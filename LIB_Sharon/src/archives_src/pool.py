from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from pyngs.core import NGSpiceInstance


@contextmanager
def _chdir(path: Optional[Path]):
    """Temporarily change working directory (useful for relative .include in SPICE libs)."""
    if path is None:
        yield
        return

    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _row_to_params(row: pd.Series) -> Dict[str, float]:
    """Convert a pandas row to a plain dict of float parameters."""
    params: Dict[str, float] = {}
    for k, v in row.items():
        if pd.isna(v):
            continue
        params[str(k)] = float(v)
    return params


def _simulate_one(job: "SimJob") -> Dict[str, Any]:
    """Worker function: run one NGSpice simulation for one set of params."""
    inst = NGSpiceInstance()
    try:
        with _chdir(job.ngspice_cwd):
            inst.load(job.netlist_path)

            for name, value in job.params.items():
                inst.set_parameter(name, value)

            inst.run()

            measures = inst.list_measures()
            out: Dict[str, Any] = {}
            for m in measures:
                out[m] = inst.get_measure(m)

            return out
    finally:
        # Ensure simulator is stopped even if something fails
        try:
            inst.stop()
        except Exception:
            pass


@dataclass(frozen=True)
class SimJob:
    netlist_path: Path
    params: Dict[str, float]
    ngspice_cwd: Optional[Path] = None


class SequentialPool:
    """
    Sequential version: runs simulations one after another.

    Spec goal: pool = SequentialPool([...]); results = pool.run(values_df)
    """

    def __init__(self, netlists: Iterable[str | Path], ngspice_cwd: Optional[str | Path] = None):
        self.netlists: List[Path] = [Path(p) for p in netlists]
        if not self.netlists:
            raise ValueError("netlists must contain at least one path")
        self.ngspice_cwd: Optional[Path] = Path(ngspice_cwd) if ngspice_cwd is not None else None

        # Pre-create instances (one per netlist), as requested by the spec.
        self._instances: List[NGSpiceInstance] = []
        for nl in self.netlists:
            inst = NGSpiceInstance()
            with _chdir(self.ngspice_cwd):
                inst.load(nl)
            self._instances.append(inst)

        # Discover measure names once (requires a run in many setups).
        self._measure_names: Optional[List[str]] = None

    def _ensure_measure_names(self) -> List[str]:
        if self._measure_names is not None:
            return self._measure_names

        # Use the first instance to discover measures
        inst = self._instances[0]
        with _chdir(self.ngspice_cwd):
            inst.run()
            self._measure_names = list(inst.list_measures())
        return self._measure_names

    def run(self, values: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(values, pd.DataFrame):
            raise TypeError("values must be a pandas DataFrame")

        measure_names = self._ensure_measure_names()

        results: List[Dict[str, Any]] = []
        n_inst = len(self._instances)

        for i, (_, row) in enumerate(values.iterrows()):
            inst = self._instances[i % n_inst]
            params = _row_to_params(row)

            with _chdir(self.ngspice_cwd):
                for name, value in params.items():
                    inst.set_parameter(name, value)
                inst.run()

                out: Dict[str, Any] = {}
                for m in measure_names:
                    out[m] = inst.get_measure(m)
                results.append(out)

        return pd.DataFrame(results)

    def close(self) -> None:
        for inst in self._instances:
            try:
                inst.stop()
            except Exception:
                pass


class ParallelPool:
    """
    Parallel version: runs simulations in multiple processes.
    Each job creates its own NGSpiceInstance (safer than sharing).
    """

    def __init__(
        self,
        netlists: Iterable[str | Path],
        n_workers: Optional[int] = None,
        ngspice_cwd: Optional[str | Path] = None,
        maxtasksperchild: int = 1,
    ):
        self.netlists: List[Path] = [Path(p) for p in netlists]
        if not self.netlists:
            raise ValueError("netlists must contain at least one path")

        self.ngspice_cwd: Optional[Path] = Path(ngspice_cwd) if ngspice_cwd is not None else None
        self.n_workers = n_workers
        self.maxtasksperchild = maxtasksperchild

    def run(self, values: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(values, pd.DataFrame):
            raise TypeError("values must be a pandas DataFrame")

        if len(values) == 0:
            return pd.DataFrame()

        # Create jobs (round-robin assignment across netlists)
        jobs: List[SimJob] = []
        for i, (_, row) in enumerate(values.iterrows()):
            nl = self.netlists[i % len(self.netlists)]
            jobs.append(SimJob(netlist_path=nl, params=_row_to_params(row), ngspice_cwd=self.ngspice_cwd))

        # Decide worker count
        if self.n_workers is None:
            n_workers = min(os.cpu_count() or 1, len(jobs))
        else:
            n_workers = max(1, int(self.n_workers))

        # Use "spawn" for safety (works on Linux too)
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers, maxtasksperchild=self.maxtasksperchild) as pool:
            outs = pool.map(_simulate_one, jobs)

        return pd.DataFrame(outs)
