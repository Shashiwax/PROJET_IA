# inverter_env.py
# RL environment for a SKY130 inverter simulated with ngspice via pyngs.
# This version:
#  - forces the working directory to the netlist folder when loading/running (fixes relative .include issues)
#  - clamps metrics that should never be negative (fixes SB3 check_env bounds issues)

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pyngs.core import NGSpiceInstance


@contextmanager
def pushd(path: Path):
    """Temporarily change working directory (process-wide)."""
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
    """
    Gymnasium environment (single instance, sequential).

    Action: [wn, wp] (continuous)
    Observation: [wn, wp, cell_area_um2, active_area_um2, delay_fall_ps, delay_rise_ps, edyn_fJ, pstat_wc_uW]

    Netlist requirements (.meas names):
      cell_area, active_area, delay_fall, delay_rise, edyn_val, pstat_vdd_in0, pstat_vdd_in1
    Netlist requirements (.param names):
      wn, wp
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        netlist_path: Optional[str | Path] = None,
        wn_range: Tuple[float, float] = (0.20, 0.65),
        wp_range: Tuple[float, float] = (0.20, 1.00),
        max_steps: int = 25,
        reset_to_nominal: bool = True,
        nominal_wn: float = 0.65,
        nominal_wp: float = 1.00,
        random_reset: bool = False,
        reward_weights: RewardWeights = RewardWeights(),
        reward_norms: RewardNorms = RewardNorms(),
        seed: Optional[int] = None,
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

        # Resolve netlist path
        self.netlist_path = self._resolve_netlist_path(netlist_path)

        # Action space: [wn, wp]
        self.action_space = spaces.Box(
            low=np.array([self.wn_min, self.wp_min], dtype=np.float32),
            high=np.array([self.wn_max, self.wp_max], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: 8 floats
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

        # RNG
        self._np_random, _ = gym.utils.seeding.np_random(seed)

        # NGSpice instance (load once)
        self.inst = NGSpiceInstance()
        with pushd(self.netlist_path.parent):
            self.inst.load(self.netlist_path)

        self._step_count = 0
        self._last_action: Optional[Tuple[float, float]] = None

    def _resolve_netlist_path(self, netlist_path: Optional[str | Path]) -> Path:
        if netlist_path is not None:
            p = Path(netlist_path).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"Netlist not found: {p}")
            return p

        # Prefer local ./netlists/inv.cir if it exists
        local_candidate = (Path(__file__).resolve().parent / "netlists" / "inv.cir")
        if local_candidate.exists():
            return local_candidate

        # Fallback to your known absolute path (if present)
        abs_candidate = Path("/home/sharo/PROJET_IA/LIB_Sharon/netlists/inv.cir")
        if abs_candidate.exists():
            return abs_candidate.resolve()

        raise FileNotFoundError(
            "Netlist path not provided and couldn't find ./netlists/inv.cir or the default absolute path."
        )

    def _safe_reset_spice(self) -> None:
        try:
            self.inst.cmd("reset")
            return
        except Exception:
            pass
        try:
            self.inst.reset()
            return
        except Exception:
            pass

    def _set_params(self, wn: float, wp: float) -> None:
        self.inst.set_parameter("wn", float(wn))
        self.inst.set_parameter("wp", float(wp))

    def _read_metrics(self) -> Dict[str, float]:
        """
        Reads .meas results and returns metrics in user-friendly units.
        Also clamps any metric that should not be negative (for stable RL + env bounds).
        """
        m: Dict[str, float] = {}

        def gm(name: str) -> float:
            val = self.inst.get_measure(name)
            if val is None:
                raise RuntimeError(f"Missing .meas result: {name}")
            return float(val)

        # Areas (already µm² in your netlist)
        m["cell_area_um2"] = max(0.0, gm("cell_area"))
        m["active_area_um2"] = max(0.0, gm("active_area"))

        # Delays (s -> ps)
        m["delay_fall_ps"] = max(0.0, gm("delay_fall") * 1e12)
        m["delay_rise_ps"] = max(0.0, gm("delay_rise") * 1e12)

        # Energy (J -> fJ), clamp to 0 (edyn_val can go negative due to baseline subtraction)
        edyn_fJ = gm("edyn_val") * 1e15
        m["edyn_fJ"] = max(0.0, edyn_fJ)

        # Static power on VDD
        p_vdd_in0_w = gm("pstat_vdd_in0")
        p_vdd_in1_w = gm("pstat_vdd_in1")

        m["pstat_vdd_in0_uW"] = max(0.0, p_vdd_in0_w * 1e6)
        m["pstat_vdd_in1_pW"] = max(0.0, p_vdd_in1_w * 1e12)

        # Worst-case static power (µW)
        m["pstat_wc_uW"] = max(0.0, max(p_vdd_in0_w, p_vdd_in1_w) * 1e6)

        return m

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

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
            with pushd(self.netlist_path.parent):
                self._safe_reset_spice()
                self._set_params(wn, wp)
                self.inst.run()
                metrics = self._read_metrics()

            obs = self._make_obs(wn, wp, metrics)
            return obs, {"metrics": metrics}

        except Exception as e:
            obs = np.zeros((8,), dtype=np.float32)
            return obs, {"metrics": {"error": str(e), "where": "reset"}}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        wn = float(np.clip(action[0], self.wn_min, self.wn_max))
        wp = float(np.clip(action[1], self.wp_min, self.wp_max))
        self._last_action = (wn, wp)

        terminated = False
        truncated = self._step_count >= self.max_steps

        try:
            with pushd(self.netlist_path.parent):
                self._safe_reset_spice()
                self._set_params(wn, wp)
                self.inst.run()
                metrics = self._read_metrics()

            obs = self._make_obs(wn, wp, metrics)
            reward = self._compute_reward(metrics)
            info = {"metrics": metrics}
            return obs, reward, terminated, truncated, info

        except Exception as e:
            obs = np.zeros((8,), dtype=np.float32)
            reward = -1e6
            terminated = True
            truncated = True
            info = {"metrics": {"error": str(e), "wn": wn, "wp": wp, "where": "step"}}
            return obs, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            self.inst.stop()
        except Exception:
            pass
        super().close()


if __name__ == "__main__":
    # Quick sanity test: run a few random steps
    env = InverterEnv(max_steps=5, random_reset=False)
    obs, info = env.reset()
    print("RESET obs:", obs)
    print("RESET metrics:", info.get("metrics", {}))

    for t in range(5):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print(f"\nstep={t+1} action={a} reward={r:.6f} term={term} trunc={trunc}")
        print("metrics:", info.get("metrics", {}))
        if term or trunc:
            break

    env.close()
