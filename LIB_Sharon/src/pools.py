from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import os
import re
import subprocess
import tempfile

import pandas as pd


# -----------------------------
# Parsing helpers
# -----------------------------

_MEAS_LINE_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*=\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)"
)


def _infer_spice_cwd_from_netlist_text(netlist_text: str) -> Optional[Path]:
    """
    Infer ngspice working directory from the first .lib "path" line.
    This matters because model libraries often use relative .include paths.
    """
    lib_re = re.compile(r'^\s*\.lib\s+"([^"]+)"', re.IGNORECASE | re.MULTILINE)
    m = lib_re.search(netlist_text)
    if not m:
        return None
    lib_path = Path(m.group(1)).expanduser()
    return lib_path.parent


def _rewrite_params_in_netlist(netlist_text: str, params: Dict[str, float]) -> str:
    """
    Replace .param <name>=... occurrences for the given params.
    If not found, append a new .param at the end (last definition wins).
    """
    out = netlist_text

    for name, val in params.items():
        pattern = re.compile(
            rf"(^\s*\.param\s+{re.escape(name)}\s*=\s*)([^\s]+)",
            re.IGNORECASE | re.MULTILINE,
        )
        repl = rf"\g<1>{val:.6g}"
        out, n = pattern.subn(repl, out, count=1)
        if n == 0:
            out += f"\n.param {name}={val:.6g}\n"

    return out


def _run_ngspice_batch(netlist_path: Path, spice_cwd: Optional[Path], timeout_s: float) -> str:
    """
    Run ngspice in batch mode and return stdout+stderr.
    """
    cmd = ["ngspice", "-b", str(netlist_path)]
    cwd = str(spice_cwd) if spice_cwd is not None else None

    env = os.environ.copy()
    env.setdefault("NGSPICE_ASCIIRAWFILE", "1")

    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ngspice failed (code={p.returncode}): {p.stdout[-4000:]}")
    return p.stdout


def _parse_measures_from_ngspice_output(output_text: str) -> Dict[str, float]:
    """
    Parse 'name = value' lines printed by ngspice for .meas results.
    Works even if the line has extra tokens like 'targ=' and 'trig='.
    """
    measures: Dict[str, float] = {}
    for line in output_text.splitlines():
        m = _MEAS_LINE_RE.match(line)
        if not m:
            continue
        key = m.group(1)
        val = float(m.group(2))
        measures[key] = val
    return measures


# -----------------------------
# Reward config + reward logic
# -----------------------------

@dataclass(frozen=True)
class RewardConfig:
    """
    Keep this consistent between Pool+CEM and SB3 env (later).
    Defaults assume your baseline is wn0=0.65 wp0=1.0 at VDD=1.8V.
    """
    # Baseline reference (normalized ratios)
    ref_cell_area_um2: float = 3.7536
    ref_delay_max_ps: float = 18.96081
    ref_pstat_wc_uW: float = 1.295751
    ref_edyn_fJ: float = 1.96807

    # Weights for the 4 specs
    w_area: float = 0.35
    w_delay: float = 0.35
    w_pstat: float = 0.25
    w_edyn: float = 0.25

    # Soft size regularizer (penalize only above baseline)
    w_size: float = 0.1
    wn0: float = 0.65
    wp0: float = 1.0

    # Logic constraints on ymean (hard)
    vdd: float = 1.8
    yhi_ratio: float = 0.95
    ylo_ratio: float = 0.05

    fail_reward: float = -1e6


def _postprocess_robot_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert raw ngspice measures to robot-friendly units:
      - area: um^2 (already in your netlist)
      - delay: s -> ps
      - power: W -> uW
      - energy: J -> fJ
    Also compute:
      - delay_max_ps
      - pstat_wc_uW
    """
    out: Dict[str, Any] = dict(raw)

    # Areas (already in um^2 in your measurement expressions)
    if "cell_area" in out:
        out["cell_area_um2"] = float(out["cell_area"])
    if "active_area" in out:
        out["active_area_um2"] = float(out["active_area"])

    # Delays (s -> ps)
    if "delay_fall" in out:
        out["delay_fall_ps"] = float(out["delay_fall"]) * 1e12
    if "delay_rise" in out:
        out["delay_rise_ps"] = float(out["delay_rise"]) * 1e12
    if "delay_fall_ps" in out and "delay_rise_ps" in out:
        out["delay_max_ps"] = max(float(out["delay_fall_ps"]), float(out["delay_rise_ps"]))

    # Energy (J -> fJ)
    if "edyn_val" in out:
        out["edyn_fJ"] = float(out["edyn_val"]) * 1e15

    # Static power from VDD (W -> uW)
    if "pstat_vdd_in0" in out:
        out["pstat_vdd_in0_uW"] = float(out["pstat_vdd_in0"]) * 1e6
    if "pstat_vdd_in1" in out:
        out["pstat_vdd_in1_uW"] = float(out["pstat_vdd_in1"]) * 1e6

    # Worst-case static power
    if "pstat_vdd_in0_uW" in out and "pstat_vdd_in1_uW" in out:
        out["pstat_wc_uW"] = max(float(out["pstat_vdd_in0_uW"]), float(out["pstat_vdd_in1_uW"]))
    elif "pstat_vdd_in0_uW" in out:
        out["pstat_wc_uW"] = float(out["pstat_vdd_in0_uW"])
    elif "pstat_vdd_in1_uW" in out:
        out["pstat_wc_uW"] = float(out["pstat_vdd_in1_uW"])

    # Keep ymean as-is (Volts)
    # Expected measures: ymean_a0, ymean_a1

    return out


def _compute_reward(metrics: Dict[str, Any], cfg: RewardConfig) -> float:
    """
    Reward = -(weighted normalized cost + size_regularizer)
    Hard constraints:
      - ymean_a0 >= 0.95*VDD
      - ymean_a1 <= 0.05*VDD
    Delay metric:
      - D = delay_max_ps = max(delay_rise_ps, delay_fall_ps)
    """
    if "error" in metrics and metrics["error"]:
        return cfg.fail_reward

    required = ["cell_area_um2", "delay_max_ps", "pstat_wc_uW", "edyn_fJ", "ymean_a0", "ymean_a1", "wn_in", "wp_in"]
    if any(k not in metrics for k in required):
        return cfg.fail_reward

    # Hard logic constraints
    ymean_a0 = float(metrics["ymean_a0"])
    ymean_a1 = float(metrics["ymean_a1"])
    y_hi = cfg.yhi_ratio * cfg.vdd
    y_lo = cfg.ylo_ratio * cfg.vdd
    if ymean_a0 < y_hi or ymean_a1 > y_lo:
        return cfg.fail_reward

    # Normalized costs (ratios)
    area = float(metrics["cell_area_um2"])
    dmax = float(metrics["delay_max_ps"])
    pstat = float(metrics["pstat_wc_uW"])
    edyn = float(metrics["edyn_fJ"])

    a = area / (cfg.ref_cell_area_um2 + 1e-12)
    d = dmax / (cfg.ref_delay_max_ps + 1e-12)
    p = pstat / (cfg.ref_pstat_wc_uW + 1e-12)
    e = edyn / (cfg.ref_edyn_fJ + 1e-12)

    base_cost = cfg.w_area * a + cfg.w_delay * d + cfg.w_pstat * p + cfg.w_edyn * e

    # Soft size regularizer (only above baseline)
    wn = float(metrics["wn_in"])
    wp = float(metrics["wp_in"])
    dw = max(0.0, wn / (cfg.wn0 + 1e-12) - 1.0)
    dp = max(0.0, wp / (cfg.wp0 + 1e-12) - 1.0)
    size_cost = cfg.w_size * (dw * dw + dp * dp)

    return -(base_cost + size_cost)


# -----------------------------
# Single simulation job
# -----------------------------

@dataclass(frozen=True)
class SimJob:
    netlist_template: Path
    params: Dict[str, float]


def _simulate_one_job(job: SimJob, timeout_s: float, spice_cwd: Optional[Path]) -> Dict[str, Any]:
    """
    Worker-safe single simulation:
      - copy netlist to a temp folder
      - rewrite params
      - run ngspice
      - parse measures
    """
    try:
        template_text = job.netlist_template.read_text()

        inferred_cwd = _infer_spice_cwd_from_netlist_text(template_text)
        effective_cwd = spice_cwd if spice_cwd is not None else inferred_cwd

        with tempfile.TemporaryDirectory(prefix="spice_job_") as td:
            td_path = Path(td)
            work_netlist = td_path / job.netlist_template.name

            rewritten = _rewrite_params_in_netlist(template_text, job.params)
            work_netlist.write_text(rewritten)

            out = _run_ngspice_batch(work_netlist, effective_cwd, timeout_s)
            meas = _parse_measures_from_ngspice_output(out)

        # Echo the actual params used
        for k, v in job.params.items():
            meas[f"{k}_in"] = float(v)

        return meas

    except Exception as e:
        return {"error": str(e), **{f"{k}_in": float(v) for k, v in job.params.items()}}


# -----------------------------
# Pool classes
# -----------------------------

class SequentialPool:
    """
    Runs simulations one by one: run(DataFrame)->DataFrame.
    """

    def __init__(
        self,
        netlists: Iterable[str | Path],
        *,
        timeout_s: float = 20.0,
        spice_cwd: Optional[str | Path] = None,
        reward_cfg: Optional[RewardConfig] = None,
        keep_raw: bool = False,
    ) -> None:
        self.netlists = [Path(n).expanduser().resolve() for n in netlists]
        if len(self.netlists) == 0:
            raise ValueError("netlists must not be empty")

        self.timeout_s = float(timeout_s)
        self.spice_cwd = Path(spice_cwd).expanduser().resolve() if spice_cwd else None
        self.reward_cfg = reward_cfg or RewardConfig()
        self.keep_raw = bool(keep_raw)

    def run(self, values: pd.DataFrame) -> pd.DataFrame:
        results: List[Dict[str, Any]] = []
        n_templates = len(self.netlists)

        for i, row in values.iterrows():
            params = {k: float(row[k]) for k in values.columns}
            tpl = self.netlists[i % n_templates]
            job = SimJob(netlist_template=tpl, params=params)

            raw = _simulate_one_job(job, self.timeout_s, self.spice_cwd)
            met = _postprocess_robot_metrics(raw)
            met["reward"] = _compute_reward(met, self.reward_cfg)

            results.append(met)

        df = pd.DataFrame(results)
        return df if self.keep_raw else _select_robot_columns(df)


class ParallelPool:
    """
    Runs simulations in parallel using multiple processes: run(DataFrame)->DataFrame.
    """

    def __init__(
        self,
        netlists: Iterable[str | Path],
        *,
        n_workers: Optional[int] = None,
        timeout_s: float = 20.0,
        spice_cwd: Optional[str | Path] = None,
        reward_cfg: Optional[RewardConfig] = None,
        keep_raw: bool = False,
    ) -> None:
        self.netlists = [Path(n).expanduser().resolve() for n in netlists]
        if len(self.netlists) == 0:
            raise ValueError("netlists must not be empty")

        self.timeout_s = float(timeout_s)
        self.spice_cwd = Path(spice_cwd).expanduser().resolve() if spice_cwd else None
        self.reward_cfg = reward_cfg or RewardConfig()
        self.keep_raw = bool(keep_raw)

        self.n_workers = int(n_workers) if n_workers is not None else len(self.netlists)
        if self.n_workers <= 0:
            raise ValueError("n_workers must be >= 1")

    def run(self, values: pd.DataFrame) -> pd.DataFrame:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        jobs: List[SimJob] = []
        n_templates = len(self.netlists)

        for i, row in values.iterrows():
            params = {k: float(row[k]) for k in values.columns}
            tpl = self.netlists[i % n_templates]
            jobs.append(SimJob(netlist_template=tpl, params=params))

        results: List[Dict[str, Any]] = [None] * len(jobs)  # type: ignore

        with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
            future_map = {
                ex.submit(_simulate_one_job, job, self.timeout_s, self.spice_cwd): idx
                for idx, job in enumerate(jobs)
            }

            for fut in as_completed(future_map):
                idx = future_map[fut]
                raw = fut.result()
                met = _postprocess_robot_metrics(raw)
                met["reward"] = _compute_reward(met, self.reward_cfg)
                results[idx] = met

        df = pd.DataFrame(results)
        return df if self.keep_raw else _select_robot_columns(df)


def _select_robot_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns that matter for the robot + minimal debugging.
    """
    cols = [
        "wn_in",
        "wp_in",
        "cell_area_um2",
        "active_area_um2",
        "delay_fall_ps",
        "delay_rise_ps",
        "delay_max_ps",
        "edyn_fJ",
        "pstat_wc_uW",
        "pstat_vdd_in0_uW",
        "pstat_vdd_in1_uW",
        "ymean_a0",
        "ymean_a1",
        "reward",
        "error",
    ]
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy()
