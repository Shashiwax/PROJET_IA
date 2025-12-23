from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import os
import re
import shutil
import subprocess
import tempfile

import pandas as pd


_MEAS_LINE_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*=\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)"
)


def _infer_spice_cwd_from_netlist_text(netlist_text: str) -> Optional[Path]:
    """
    Infer the best ngspice working directory from the first .lib "path" line.
    This is critical for relative .include inside the sky130 model library.
    """
    # Example: .lib "/abs/path/.../sky130copy.lib.spice" tt
    lib_re = re.compile(r'^\s*\.lib\s+"([^"]+)"', re.IGNORECASE | re.MULTILINE)
    m = lib_re.search(netlist_text)
    if not m:
        return None
    lib_path = Path(m.group(1)).expanduser()
    return lib_path.parent if lib_path.exists() else lib_path.parent


def _rewrite_params_in_netlist(netlist_text: str, params: Dict[str, float]) -> str:
    """
    Replace .param <name>=... occurrences for the given params.
    Assumes params like wn/wp are defined as individual .param lines.
    """
    out = netlist_text

    for name, val in params.items():
        # Replace lines like: .param wn=650000u   OR   .param wn = 0.65
        # Keep the rest of the line unchanged.
        pattern = re.compile(
            rf"(^\s*\.param\s+{re.escape(name)}\s*=\s*)([^\s]+)",
            re.IGNORECASE | re.MULTILINE,
        )
        repl = rf"\g<1>{val:.6g}"
        out, n = pattern.subn(repl, out, count=1)
        if n == 0:
            # If not found, append a new definition (last definition wins in SPICE).
            out += f"\n.param {name}={val:.6g}\n"

    return out


def _run_ngspice_batch(
    netlist_path: Path,
    spice_cwd: Optional[Path],
    timeout_s: float,
) -> str:
    """
    Run ngspice in batch mode and return stdout+stderr text.
    """
    cmd = ["ngspice", "-b", str(netlist_path)]
    cwd = str(spice_cwd) if spice_cwd is not None else None

    env = os.environ.copy()
    # Avoid GUI / interactive behaviors
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
    Parse the 'name = value' lines printed by ngspice for .meas results.
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


@dataclass(frozen=True)
class SimJob:
    netlist_template: Path
    params: Dict[str, float]


def _simulate_one_job(
    job: SimJob,
    timeout_s: float,
    spice_cwd: Optional[Path],
) -> Dict[str, Any]:
    """
    Worker-safe single simulation: copy netlist to temp, rewrite params, run ngspice, parse measures.
    Returns a dict of measures + potential 'error'.
    """
    netlist_template = job.netlist_template

    try:
        template_text = netlist_template.read_text()

        inferred_cwd = _infer_spice_cwd_from_netlist_text(template_text)
        effective_cwd = spice_cwd if spice_cwd is not None else inferred_cwd

        with tempfile.TemporaryDirectory(prefix="spice_job_") as td:
            td_path = Path(td)
            work_netlist = td_path / netlist_template.name

            rewritten = _rewrite_params_in_netlist(template_text, job.params)
            work_netlist.write_text(rewritten)

            out = _run_ngspice_batch(work_netlist, effective_cwd, timeout_s)
            meas = _parse_measures_from_ngspice_output(out)

        # Always echo back the params used (useful for debugging)
        meas_out: Dict[str, Any] = dict(meas)
        for k, v in job.params.items():
            meas_out[f"{k}_in"] = float(v)

        return meas_out

    except Exception as e:
        return {"error": str(e), **{f"{k}_in": float(v) for k, v in job.params.items()}}


class SequentialPool:
    """
    Sequential simulation pool (spec: __init__(netlists), run(values)->DataFrame).
    Runs simulations one by one. :contentReference[oaicite:2]{index=2}
    """

    def __init__(
        self,
        netlists: Iterable[str | Path],
        *,
        timeout_s: float = 20.0,
        spice_cwd: Optional[str | Path] = None,
    ) -> None:
        self.netlists = [Path(n).expanduser().resolve() for n in netlists]
        if len(self.netlists) == 0:
            raise ValueError("netlists must not be empty")

        self.timeout_s = float(timeout_s)
        self.spice_cwd = Path(spice_cwd).expanduser().resolve() if spice_cwd else None

    def run(self, values: pd.DataFrame) -> pd.DataFrame:
        """
        values: DataFrame with columns matching the SPICE .param names (e.g. wn, wp).
        returns: DataFrame with parsed measures (one row per input row). :contentReference[oaicite:3]{index=3}
        """
        results: List[Dict[str, Any]] = []

        # Use round-robin netlist templates (useful when you have duplicates inv1.cir inv2.cir...)
        n_templates = len(self.netlists)

        for i, row in values.iterrows():
            params = {k: float(row[k]) for k in values.columns}
            tpl = self.netlists[i % n_templates]
            job = SimJob(netlist_template=tpl, params=params)
            res = _simulate_one_job(job, timeout_s=self.timeout_s, spice_cwd=self.spice_cwd)
            results.append(res)

        return pd.DataFrame(results)


class ParallelPool:
    """
    Parallel simulation pool (spec: __init__(netlists), run(values)->DataFrame). :contentReference[oaicite:4]{index=4}
    """

    def __init__(
        self,
        netlists: Iterable[str | Path],
        *,
        n_workers: Optional[int] = None,
        timeout_s: float = 20.0,
        spice_cwd: Optional[str | Path] = None,
    ) -> None:
        self.netlists = [Path(n).expanduser().resolve() for n in netlists]
        if len(self.netlists) == 0:
            raise ValueError("netlists must not be empty")

        self.timeout_s = float(timeout_s)
        self.spice_cwd = Path(spice_cwd).expanduser().resolve() if spice_cwd else None

        # Default: number of workers = number of netlists (as suggested by the spec idea of duplicating netlists)
        self.n_workers = int(n_workers) if n_workers is not None else len(self.netlists)
        if self.n_workers <= 0:
            raise ValueError("n_workers must be >= 1")

    def run(self, values: pd.DataFrame) -> pd.DataFrame:
        """
        values: DataFrame with columns matching SPICE .param names (wn, wp, ...). :contentReference[oaicite:5]{index=5}
        returns: DataFrame of measures (one row per input row).
        """
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
                results[idx] = fut.result()

        return pd.DataFrame(results)
