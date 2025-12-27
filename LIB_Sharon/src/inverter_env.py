# inverter_env.py
from __future__ import annotations

import os
import re
import time
import math
import signal
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# -----------------------------
# Reward configuration
# -----------------------------
@dataclass(frozen=True)
class RewardConfig:
    """
    Reward shaping designed to be PPO-friendly (bounded penalties, learnable gradients).
    All units in this config are in *human units*:
      - area in um^2
      - delay in ps
      - power in uW
      - energy in fJ
    """
    # Baseline reference for normalization (your inv_1 baseline)
    ref_cell_area_um2: float = 3.7536
    ref_delay_max_ps: float = 18.96081
    ref_pstat_wc_uW: float = 1.295751
    ref_edyn_fJ: float = 1.96807

    # Weights for the 4 specs (cost terms)
    w_area: float = 0.40
    w_delay: float = 0.20
    w_pstat: float = 0.20
    w_edyn: float = 0.20

    # Soft size regularizer (discourage oversizing beyond baseline)
    w_size: float = 0.10
    wn0: float = 0.65
    wp0: float = 1.00

    # Logic constraints (soft penalties + hard cut-off)
    vdd: float = 1.8
    yhi_ratio: float = 0.95   # want Y ~ VDD when A=0
    ylo_ratio: float = 0.05   # want Y ~ 0 when A=1

    # Soft penalty weights (bounded)
    w_logic_hi: float = 0.50
    w_logic_lo: float = 0.50

    # Hard cut-off (if badly violated, treat as failure)
    hard_hi_ratio: float = 0.80
    hard_lo_ratio: float = 0.20

    # Failure reward (bounded -> PPO-friendly)
    fail_reward: float = -50.0

    # Clip ratios to avoid huge spikes
    ratio_clip: float = 10.0


# -----------------------------
# Helpers: subprocess ngspice
# -----------------------------
_MEAS_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.MULTILINE)


def _kill_process_group(p: subprocess.Popen) -> None:
    """Kill a process group (POSIX)."""
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def run_ngspice_batch(
    netlist_path: Path,
    cwd: Path,
    log_path: Path,
    timeout_s: float,
) -> str:
    """
    Run ngspice in batch mode and return the stdout/stderr text (from log file).
    Uses a new process group so we can hard-kill on timeout.
    """
    cmd = ["ngspice", "-b", "-o", str(log_path), str(netlist_path)]

    # Create a new process group (POSIX) so we can kill all children on timeout.
    preexec = os.setsid if os.name != "nt" else None

    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=preexec,
    )

    try:
        p.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_process_group(p)
        raise RuntimeError(f"ngspice timeout after {timeout_s}s")

    if not log_path.exists():
        raise RuntimeError("ngspice did not produce a log file")

    text = log_path.read_text(errors="ignore")
    if "Error: circuit not parsed" in text:
        raise RuntimeError('"Error: circuit not parsed."')

    return text


def parse_measures(log_text: str) -> Dict[str, float]:
    """
    Parse ngspice .meas results from log output.
    Returns a dict with lower-cased measure names.
    """
    out: Dict[str, float] = {}
    for m in _MEAS_RE.finditer(log_text):
        k = m.group(1).strip().lower()
        v = float(m.group(2))
        out[k] = v
    return out


# -----------------------------
# Inverter Environment
# -----------------------------
class InverterEnv(gym.Env):
    """
    PPO-friendly environment for inverter sizing (wn, wp).
    - Action space: Box([-1,1], shape=(2,))
    - Observation: current (wn, wp)
    - Each step runs exactly one SPICE simulation and returns reward + metrics.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        netlist_path: str | Path,
        max_steps: int = 1,
        reward_cfg: Optional[RewardConfig] = None,
        seed: Optional[int] = None,
        # Parameter bounds (um)
        wn_min: float = 0.20,
        wn_max: float = 1.26,
        wp_min: float = 0.20,
        wp_max: float = 1.65,
        # Discretization step (um) to avoid weird values and reduce crashes
        snap_step: float = 0.01,
        # Fast training overrides
        fast_mode: bool = True,
        fast_tran_step: str = "5p",
        fast_tsim: str = "2n",
        # ngspice timeout (seconds)
        timeout_s: float = 20.0,
        # Cache size (per-process)
        cache_max: int = 5000,
        # Debug prints
        debug: bool = False,
        # If True, run one simulation at reset (slower). Keep False for PPO.
        simulate_on_reset: bool = False,
    ) -> None:
        super().__init__()

        self.netlist_path = Path(netlist_path)
        if not self.netlist_path.exists():
            raise FileNotFoundError(f"netlist not found: {self.netlist_path}")

        self.max_steps = int(max_steps)
        self.cfg = reward_cfg or RewardConfig()
        self.debug = bool(debug)
        self.simulate_on_reset = bool(simulate_on_reset)

        self.wn_min = float(wn_min)
        self.wn_max = float(wn_max)
        self.wp_min = float(wp_min)
        self.wp_max = float(wp_max)
        self.snap_step = float(snap_step)

        self.fast_mode = bool(fast_mode)
        self.fast_tran_step = str(fast_tran_step)
        self.fast_tsim = str(fast_tsim)

        self.timeout_s = float(timeout_s)
        self.cache_max = int(cache_max)

        # PPO recommendation: symmetric action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation is current (wn, wp)
        self.observation_space = spaces.Box(
            low=np.array([self.wn_min, self.wp_min], dtype=np.float32),
            high=np.array([self.wn_max, self.wp_max], dtype=np.float32),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(seed)

        # Read and keep a template of the netlist text
        self.base_text = self.netlist_path.read_text(errors="ignore")

        # Infer SPICE CWD from the .lib path (needed for relative .include inside sky130copy.lib.spice)
        self.spice_cwd = self._infer_spice_cwd(self.base_text)

        # Per-env temp directory (important: each Subproc env has its own process -> safe)
        self.workdir = Path(tempfile.mkdtemp(prefix="inv_env_"))

        # Runtime state
        self.step_count = 0
        self.wn = self.cfg.wn0
        self.wp = self.cfg.wp0

        # Cache: (wn, wp) -> metrics dict
        self._cache: Dict[Tuple[float, float], Dict[str, float]] = {}

    # -------- netlist rewriting --------

    @staticmethod
    def _infer_spice_cwd(net_text: str) -> Path:
        """
        Find the first .lib "PATH" ... line and use its parent directory as CWD.
        This is required because sky130copy.lib.spice contains relative .include paths.
        """
        lib_re = re.compile(r'^\s*\.lib\s+"([^"]+)"\s+\S+', re.MULTILINE | re.IGNORECASE)
        m = lib_re.search(net_text)
        if m:
            lib_path = Path(m.group(1))
            if lib_path.exists():
                return lib_path.parent
        # Fallback: netlist directory
        return Path.cwd()

    def _snap(self, x: float) -> float:
        """Snap to a grid to reduce weird values and improve caching."""
        if self.snap_step <= 0:
            return float(x)
        return float(round(x / self.snap_step) * self.snap_step)

    def _action_to_params(self, action: np.ndarray) -> Tuple[float, float]:
        """
        Map action in [-1,1] to (wn, wp) within bounds, then snap.
        """
        a = np.clip(action.astype(np.float32), -1.0, 1.0)
        # Linear map [-1,1] -> [min,max]
        wn = self.wn_min + (float(a[0]) + 1.0) * 0.5 * (self.wn_max - self.wn_min)
        wp = self.wp_min + (float(a[1]) + 1.0) * 0.5 * (self.wp_max - self.wp_min)

        wn = min(max(wn, self.wn_min), self.wn_max)
        wp = min(max(wp, self.wp_min), self.wp_max)

        wn = self._snap(wn)
        wp = self._snap(wp)
        return wn, wp

    def _apply_overrides(self, text: str) -> str:
        """
        Apply fast-mode overrides (tsim + tran step).
        Assumes the netlist contains `.param tsim=...` and `.tran <step> {tsim}` lines.
        """
        if not self.fast_mode:
            return text

        # Override .param tsim=...
        text = re.sub(
            r'^\s*\.param\s+tsim\s*=\s*[^\s]+',
            f".param tsim={self.fast_tsim}",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )

        # Override .tran line first argument (tstep) but keep {tsim}
        # Example: ".tran 1p {tsim}" -> ".tran 5p {tsim}"
        text = re.sub(
            r'^\s*\.tran\s+\S+\s+\{tsim\}',
            f".tran {self.fast_tran_step} {{tsim}}",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        return text

    def _rewrite_netlist(self, wn: float, wp: float) -> Path:
        """
        Create a per-step netlist with overridden wn/wp and optional fast-mode.
        """
        txt = self.base_text

        # Replace .param wn=... and wp=...
        txt = re.sub(
            r'^\s*\.param\s+wn\s*=\s*[^\s]+',
            f".param wn={wn}",
            txt,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        txt = re.sub(
            r'^\s*\.param\s+wp\s*=\s*[^\s]+',
            f".param wp={wp}",
            txt,
            flags=re.MULTILINE | re.IGNORECASE,
        )

        # Fast overrides
        txt = self._apply_overrides(txt)

        # Write to temp file
        p = self.workdir / f"inv_wn{wn:.4f}_wp{wp:.4f}.cir"
        p.write_text(txt)
        return p

    # -------- simulation + metrics --------

    def _simulate(self, wn: float, wp: float) -> Dict[str, float]:
        """
        Run ngspice and return metrics in human units.
        Uses caching to avoid repeated sims.
        """
        key = (wn, wp)
        if key in self._cache:
            return self._cache[key]

        net_tmp = self._rewrite_netlist(wn, wp)
        log_path = self.workdir / f"ngspice_wn{wn:.4f}_wp{wp:.4f}.log"

        t0 = time.time()
        log_text = run_ngspice_batch(
            netlist_path=net_tmp,
            cwd=self.spice_cwd,
            log_path=log_path,
            timeout_s=self.timeout_s,
        )
        dt = time.time() - t0

        meas = parse_measures(log_text)

        # Required measures (case-insensitive)
        required = [
            "cell_area",
            "active_area",
            "delay_fall",
            "delay_rise",
            "edyn_val",
            "pstat_vdd_in0",
            "pstat_vdd_in1",
            "ymean_a0",
            "ymean_a1",
        ]
        missing = [k for k in required if k not in meas]
        if missing:
            raise RuntimeError(f"Missing measures: {missing}")

        # Convert to human units
        cell_area_um2 = float(meas["cell_area"])         # already um^2 by construction
        active_area_um2 = float(meas["active_area"])     # already um^2 by construction

        delay_fall_ps = float(meas["delay_fall"]) * 1e12
        delay_rise_ps = float(meas["delay_rise"]) * 1e12
        delay_max_ps = max(delay_fall_ps, delay_rise_ps)

        edyn_fJ = float(meas["edyn_val"]) * 1e15

        pstat_vdd_in0_uW = float(meas["pstat_vdd_in0"]) * 1e6
        pstat_vdd_in1_uW = float(meas["pstat_vdd_in1"]) * 1e6
        pstat_wc_uW = max(pstat_vdd_in0_uW, pstat_vdd_in1_uW)

        ymean_a0 = float(meas["ymean_a0"])
        ymean_a1 = float(meas["ymean_a1"])

        metrics = {
            "wn": wn,
            "wp": wp,
            "sim_time_s": dt,
            "cell_area_um2": cell_area_um2,
            "active_area_um2": active_area_um2,
            "delay_fall_ps": delay_fall_ps,
            "delay_rise_ps": delay_rise_ps,
            "delay_max_ps": delay_max_ps,
            "edyn_fJ": edyn_fJ,
            "pstat_vdd_in0_uW": pstat_vdd_in0_uW,
            "pstat_vdd_in1_uW": pstat_vdd_in1_uW,
            "pstat_wc_uW": pstat_wc_uW,
            "ymean_a0": ymean_a0,
            "ymean_a1": ymean_a1,
        }

        # Cache management
        if len(self._cache) >= self.cache_max:
            # Simple eviction: clear everything (fast and safe)
            self._cache.clear()
        self._cache[key] = metrics
        return metrics

    # -------- reward --------

    def _compute_reward(self, m: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Return (reward, breakdown) where breakdown helps you understand the compromise.
        Reward is negative total cost (so higher is better).
        """
        cfg = self.cfg
        eps = 1e-12

        # Hard cut-off for very broken logic (PPO-friendly bounded fail)
        hi_hard = cfg.hard_hi_ratio * cfg.vdd
        lo_hard = cfg.hard_lo_ratio * cfg.vdd
        if m["ymean_a0"] < hi_hard or m["ymean_a1"] > lo_hard:
            return float(cfg.fail_reward), {
                "fail": 1.0,
                "reason_logic_hard": 1.0,
            }

        # Normalized ratios (clipped)
        r_area = min(m["cell_area_um2"] / (cfg.ref_cell_area_um2 + eps), cfg.ratio_clip)
        r_delay = min(m["delay_max_ps"] / (cfg.ref_delay_max_ps + eps), cfg.ratio_clip)
        r_pstat = min(m["pstat_wc_uW"] / (cfg.ref_pstat_wc_uW + eps), cfg.ratio_clip)
        r_edyn = min(m["edyn_fJ"] / (cfg.ref_edyn_fJ + eps), cfg.ratio_clip)

        term_area = cfg.w_area * r_area
        term_delay = cfg.w_delay * r_delay
        term_pstat = cfg.w_pstat * r_pstat
        term_edyn = cfg.w_edyn * r_edyn

        # Soft size regularizer (only above baseline)
        sz_n = max(0.0, (m["wn"] - cfg.wn0) / (cfg.wn0 + eps))
        sz_p = max(0.0, (m["wp"] - cfg.wp0) / (cfg.wp0 + eps))
        term_size = cfg.w_size * (sz_n + sz_p)

        # Soft logic penalties around target thresholds
        hi_tgt = cfg.yhi_ratio * cfg.vdd
        lo_tgt = cfg.ylo_ratio * cfg.vdd

        # Penalize only when violating thresholds
        viol_hi = max(0.0, (hi_tgt - m["ymean_a0"]) / (hi_tgt + eps))
        viol_lo = max(0.0, (m["ymean_a1"] - lo_tgt) / (lo_tgt + eps))

        # Quadratic gives smoother gradients
        term_logic = cfg.w_logic_hi * (viol_hi ** 2) + cfg.w_logic_lo * (viol_lo ** 2)

        total_cost = term_area + term_delay + term_pstat + term_edyn + term_size + term_logic
        reward = -float(total_cost)

        breakdown = {
            "fail": 0.0,
            "r_area": r_area,
            "r_delay": r_delay,
            "r_pstat": r_pstat,
            "r_edyn": r_edyn,
            "term_area": term_area,
            "term_delay": term_delay,
            "term_pstat": term_pstat,
            "term_edyn": term_edyn,
            "term_size": term_size,
            "term_logic": term_logic,
            "cost": total_cost,
        }
        return reward, breakdown

    # -------- Gym API --------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.step_count = 0

        # Randomize reset slightly to avoid degenerate constant observation
        # (still 1-step episodes, but helps exploration a bit)
        if options and options.get("random_reset", True):
            self.wn = self._snap(self.rng.uniform(self.wn_min, self.wn_max))
            self.wp = self._snap(self.rng.uniform(self.wp_min, self.wp_max))
        else:
            self.wn = self.cfg.wn0
            self.wp = self.cfg.wp0

        obs = np.array([self.wn, self.wp], dtype=np.float32)

        info: Dict[str, Any] = {
            "spice_cwd": str(self.spice_cwd),
            "wn": self.wn,
            "wp": self.wp,
        }

        if self.simulate_on_reset:
            try:
                m = self._simulate(self.wn, self.wp)
                r, b = self._compute_reward(m)
                info["metrics"] = m
                info["breakdown"] = b
                info["reward"] = r
            except Exception as e:
                info["metrics"] = {"error": str(e)}
                info["reward"] = float(self.cfg.fail_reward)

        return obs, info

    def step(self, action):
        self.step_count += 1

        wn, wp = self._action_to_params(np.array(action, dtype=np.float32))
        self.wn, self.wp = wn, wp
        obs = np.array([wn, wp], dtype=np.float32)

        terminated = False
        truncated = False

        try:
            m = self._simulate(wn, wp)
            r, b = self._compute_reward(m)

            # Episode ends when reaching max_steps
            if self.step_count >= self.max_steps:
                truncated = True

            info = {
                "metrics": m,
                "breakdown": b,
                "spice_cwd": str(self.spice_cwd),
            }

            if self.debug:
                print(f"[DEBUG] step={self.step_count} wn={wn:.4f} wp={wp:.4f} r={r:.4f} cost={b.get('cost', None)}")

            return obs, float(r), terminated, truncated, info

        except Exception as e:
            # PPO-friendly bounded failure
            info = {
                "metrics": {"error": str(e), "wn": wn, "wp": wp, "where": "step"},
                "breakdown": {"fail": 1.0},
                "spice_cwd": str(self.spice_cwd),
            }
            # End quickly on failure (prevents wasting steps)
            terminated = True
            return obs, float(self.cfg.fail_reward), terminated, True, info

    def close(self) -> None:
        # Best effort: cleanup temp directory
        try:
            for p in self.workdir.glob("*"):
                try:
                    p.unlink()
                except Exception:
                    pass
            try:
                self.workdir.rmdir()
            except Exception:
                pass
        except Exception:
            pass
        super().close()


if __name__ == "__main__":
    # Quick manual smoke test (no SB3)
    env = InverterEnv(
        netlist_path=os.getenv("NETLIST", "../netlists/inv.cir"),
        max_steps=1,
        debug=True,
        simulate_on_reset=True,
    )
    obs, info = env.reset(options={"random_reset": False})
    print("RESET obs:", obs)
    print("RESET info:", info)

    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    print("STEP action:", a)
    print("STEP reward:", r, "term:", term, "trunc:", trunc)
    print("metrics:", info.get("metrics", {}))
    print("breakdown:", info.get("breakdown", {}))
    env.close()
