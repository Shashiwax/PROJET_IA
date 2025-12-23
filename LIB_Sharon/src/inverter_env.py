# inverter_env.py
# Stable mono-env for SKY130 inverter using ngspice as a SUBPROCESS with HARD TIMEOUT.
#
# Fix in this version:
# - Your metrics stayed constant because wn/wp were not applied.
# - We now directly REWRITE the .param wn=... and .param wp=... lines in the temporary netlist.
# - We also inject .meas wn_chk/wp_chk to confirm parameter values.
#
# Still included:
# - SPICE CWD detection so sky130copy.lib.spice relative includes work.
# - Hard timeout kill of ngspice process group (no hangs).

from __future__ import annotations

import os
import re
import signal
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@contextmanager
def pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@dataclass
class RewardWeights:
    w_area: float = 0.25
    w_delay: float = 0.35
    w_edyn: float = 0.25
    w_pstat: float = 0.15


@dataclass
class RewardNorms:
    area_um2: float = 5.0
    delay_ps: float = 20.0
    edyn_fj: float = 2.0
    pstat_uw: float = 1.0


class InverterEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        netlist_path: Optional[str | Path] = None,
        wn_range: Tuple[float, float] = (0.36, 0.65),
        wp_range: Tuple[float, float] = (0.36, 1.00),
        max_steps: int = 25,
        reset_to_nominal: bool = True,
        nominal_wn: float = 0.65,
        nominal_wp: float = 1.00,
        random_reset: bool = False,
        reward_weights: RewardWeights = RewardWeights(),
        reward_norms: RewardNorms = RewardNorms(),
        seed: Optional[int] = None,
        ngspice_bin: str = "ngspice",
        timeout_s: float = 5.0,
    ) -> None:
        super().__init__()

        self.wn_min, self.wn_max = float(wn_range[0]), float(wn_range[1])
        self.wp_min, self.wp_max = float(wp_range[0]), float(wp_range[1])
        self.max_steps = int(max_steps)

        self.reset_to_nominal = bool(reset_to_nominal)
        self.nominal_wn = float(nominal_wn)
        self.nominal_wp = float(nominal_wp)
        self.random_reset = bool(random_reset)

        self.w = reward_weights
        self.n = reward_norms

        self.ngspice_bin = ngspice_bin
        self.timeout_s = float(timeout_s)

        self.netlist_path = self._resolve_netlist_path(netlist_path)
        self.spice_cwd = self._detect_spice_cwd(self.netlist_path)

        # Base netlist without final .end (we will rewrite params + append control)
        self._base_no_end = self._read_netlist_without_final_end(self.netlist_path)

        self.action_space = spaces.Box(
            low=np.array([self.wn_min, self.wp_min], dtype=np.float32),
            high=np.array([self.wn_max, self.wp_max], dtype=np.float32),
            dtype=np.float32,
        )

        low = np.array([self.wn_min, self.wp_min, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array(
            [
                self.wn_max,
                self.wp_max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._np_random, _ = gym.utils.seeding.np_random(seed)
        self._step_count = 0

    # ----------------------------
    # Path + netlist utilities
    # ----------------------------

    def _resolve_netlist_path(self, netlist_path: Optional[str | Path]) -> Path:
        if netlist_path is not None:
            p = Path(netlist_path).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"Netlist not found: {p}")
            return p

        local_candidate = (Path(__file__).resolve().parent / "netlists" / "inv.cir")
        if local_candidate.exists():
            return local_candidate

        abs_candidate = Path("/home/sharo/PROJET_IA/LIB_Sharon/netlists/inv.cir")
        if abs_candidate.exists():
            return abs_candidate.resolve()

        raise FileNotFoundError("Netlist path not provided and no default inv.cir found.")

    def _detect_spice_cwd(self, netlist_path: Path) -> Path:
        lines = netlist_path.read_text(errors="ignore").splitlines()
        lib_re = re.compile(r'^\s*\.lib\s+"([^"]+)"', re.IGNORECASE)
        inc_re = re.compile(r'^\s*\.include\s+"([^"]+)"', re.IGNORECASE)

        candidates: list[Path] = []

        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("*"):
                continue

            m = lib_re.match(line)
            if m:
                p = Path(m.group(1)).expanduser()
                p = (netlist_path.parent / p).resolve() if not p.is_absolute() else p.resolve()
                candidates.append(p)
                continue

            m = inc_re.match(line)
            if m:
                p = Path(m.group(1)).expanduser()
                if p.is_absolute():
                    candidates.append(p.resolve())

        for p in candidates:
            if p.exists():
                return p.parent

        return netlist_path.parent

    def _read_netlist_without_final_end(self, netlist_path: Path) -> str:
        lines = netlist_path.read_text(errors="ignore").splitlines()
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and lines[-1].strip().lower().startswith(".end"):
            lines.pop()
        return "\n".join(lines) + "\n"

    def _rewrite_param(self, text: str, name: str, value: float) -> Tuple[str, bool]:
        """
        Replace the first occurrence of:
          .param <name> = ...
        with:
          .param <name>=<value>
        """
        pat = re.compile(rf"(?im)^\s*\.param\s+{re.escape(name)}\s*=\s*.*$")
        new_line = f".param {name}={value}"
        new_text, n = pat.subn(new_line, text, count=1)
        return new_text, (n == 1)

    # ----------------------------
    # ngspice subprocess simulation
    # ----------------------------

    def _build_run_netlist(self, wn: float, wp: float) -> str:
        # Rewrite wn/wp directly (robust; no alterparam dependency)
        txt, ok1 = self._rewrite_param(self._base_no_end, "wn", wn)
        txt, ok2 = self._rewrite_param(txt, "wp", wp)
        if not (ok1 and ok2):
            raise RuntimeError("Could not find .param wn=... and/or .param wp=... to rewrite in inv.cir")

        # Inject checks + control block
        injected = (
            "\n"
            "* --- injected by RL wrapper ---\n"
            ".meas tran wn_chk param='wn'\n"
            ".meas tran wp_chk param='wp'\n"
            ".control\n"
            "set noaskquit\n"
            "set nomoremode\n"
            "run\n"
            "quit\n"
            ".endc\n"
            ".end\n"
        )
        return txt + injected

    def _parse_measures(self, text: str) -> Dict[str, float]:
        measures: Dict[str, float] = {}
        line_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
        for raw in text.splitlines():
            m = line_re.match(raw)
            if not m:
                continue
            measures[m.group(1)] = float(m.group(2))
        return measures

    def _run_ngspice_killgroup(self, cmd: list[str], cwd: Path, timeout_s: float) -> None:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            text=True,
        )
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            raise RuntimeError(f"ngspice timeout after {timeout_s}s")

    def _simulate(self, wn: float, wp: float) -> Dict[str, float]:
        run_netlist = self._build_run_netlist(wn, wp)

        with tempfile.TemporaryDirectory(prefix="inv_rl_") as td:
            td_path = Path(td)
            cir_path = td_path / "run.cir"
            out_path = td_path / "ngspice.out"

            cir_path.write_text(run_netlist)

            cmd = [self.ngspice_bin, "-b", "-o", str(out_path), str(cir_path)]
            self._run_ngspice_killgroup(cmd, self.spice_cwd, self.timeout_s)

            if not out_path.exists():
                raise RuntimeError("ngspice produced no output file")

            out_text = out_path.read_text(errors="ignore")
            meas = self._parse_measures(out_text)

            required = [
                "cell_area",
                "active_area",
                "delay_fall",
                "delay_rise",
                "edyn_val",
                "pstat_vdd_in0",
                "pstat_vdd_in1",
                "wn_chk",
                "wp_chk",
            ]
            missing = [k for k in required if k not in meas]
            if missing:
                raise RuntimeError(f"Missing measures: {missing}")

            # Convert units + clamp
            metrics: Dict[str, float] = {}
            metrics["cell_area_um2"] = max(0.0, meas["cell_area"])
            metrics["active_area_um2"] = max(0.0, meas["active_area"])

            metrics["delay_fall_ps"] = max(0.0, meas["delay_fall"] * 1e12)
            metrics["delay_rise_ps"] = max(0.0, meas["delay_rise"] * 1e12)

            edyn_fJ = meas["edyn_val"] * 1e15
            metrics["edyn_fJ"] = max(0.0, edyn_fJ)

            p0_w = meas["pstat_vdd_in0"]
            p1_w = meas["pstat_vdd_in1"]
            metrics["pstat_vdd_in0_uW"] = max(0.0, p0_w * 1e6)
            metrics["pstat_vdd_in1_pW"] = max(0.0, p1_w * 1e12)
            metrics["pstat_wc_uW"] = max(0.0, max(p0_w, p1_w) * 1e6)

            # Debug checks: what wn/wp did ngspice actually use?
            metrics["wn_chk"] = meas["wn_chk"]
            metrics["wp_chk"] = meas["wp_chk"]

            return metrics

    # ----------------------------
    # RL interface
    # ----------------------------

    def _make_obs(self, wn: float, wp: float, metrics: Dict[str, float]) -> np.ndarray:
        return np.array(
            [
                wn,
                wp,
                metrics["cell_area_um2"],
                metrics["active_area_um2"],
                metrics["delay_fall_ps"],
                metrics["delay_rise_ps"],
                metrics["edyn_fJ"],
                metrics["pstat_wc_uW"],
            ],
            dtype=np.float32,
        )

    def _compute_reward(self, metrics: Dict[str, float]) -> float:
        delay_ps = max(metrics["delay_rise_ps"], metrics["delay_fall_ps"])
        area_um2 = metrics["cell_area_um2"]
        edyn_fJ = metrics["edyn_fJ"]
        pstat_wc_uW = metrics["pstat_wc_uW"]

        area_n = area_um2 / max(self.n.area_um2, 1e-12)
        delay_n = delay_ps / max(self.n.delay_ps, 1e-12)
        edyn_n = edyn_fJ / max(self.n.edyn_fj, 1e-12)
        pstat_n = pstat_wc_uW / max(self.n.pstat_uw, 1e-12)

        return -float(
            self.w.w_area * area_n
            + self.w.w_delay * delay_n
            + self.w.w_edyn * edyn_n
            + self.w.w_pstat * pstat_n
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._step_count = 0

        if self.random_reset:
            wn = float(self._np_random.uniform(self.wn_min, self.wn_max))
            wp = float(self._np_random.uniform(self.wp_min, self.wp_max))
        elif self.reset_to_nominal:
            wn = float(np.clip(self.nominal_wn, self.wn_min, self.wn_max))
            wp = float(np.clip(self.nominal_wp, self.wp_min, self.wp_max))
        else:
            wn = float(0.5 * (self.wn_min + self.wn_max))
            wp = float(0.5 * (self.wp_min + self.wp_max))

        try:
            metrics = self._simulate(wn, wp)
            obs = self._make_obs(wn, wp, metrics)
            return obs, {"metrics": metrics, "spice_cwd": str(self.spice_cwd)}
        except Exception as e:
            obs = np.zeros((8,), dtype=np.float32)
            return obs, {"metrics": {"error": str(e), "where": "reset"}, "spice_cwd": str(self.spice_cwd)}

    def step(self, action: np.ndarray):
        self._step_count += 1

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        wn = float(np.clip(action[0], self.wn_min, self.wn_max))
        wp = float(np.clip(action[1], self.wp_min, self.wp_max))

        terminated = False
        truncated = self._step_count >= self.max_steps

        try:
            metrics = self._simulate(wn, wp)
            obs = self._make_obs(wn, wp, metrics)
            reward = self._compute_reward(metrics)
            info = {"metrics": metrics, "spice_cwd": str(self.spice_cwd)}
            return obs, reward, terminated, truncated, info
        except Exception as e:
            obs = np.zeros((8,), dtype=np.float32)
            reward = -1e6
            terminated = True
            truncated = True
            info = {"metrics": {"error": str(e), "wn": wn, "wp": wp, "where": "step"}, "spice_cwd": str(self.spice_cwd)}
            return obs, reward, terminated, truncated, info

    def close(self) -> None:
        super().close()


if __name__ == "__main__":
    print("ENV BACKEND = SUBPROCESS (param rewrite + hard-timeout kill group)", flush=True)

    env = InverterEnv(max_steps=10, random_reset=False, timeout_s=5.0)
    obs, info = env.reset()

    print("RESET obs:", obs, flush=True)
    print("RESET metrics:", info.get("metrics", {}), flush=True)
    print("SPICE CWD:", info.get("spice_cwd", ""), flush=True)

    for t in range(10):
        print(f"\n[DEBUG] entering step {t+1}", flush=True)
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print(f"step={t+1} action={a} reward={r:.6f} term={term} trunc={trunc}", flush=True)
        print("metrics:", info.get("metrics", {}), flush=True)
        if term or trunc:
            break

    env.close()
