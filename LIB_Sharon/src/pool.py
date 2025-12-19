from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Mapping, List, Dict, Any

import pandas as pd
from pyngs.core import NGSpiceInstance


@dataclass
class _Simulator:
    """Small container to bind a netlist path with its NGSpice instance."""
    netlist_path: Path
    instance: NGSpiceInstance


class SequentialPool:
    """
    Sequential pool of NGSpice simulators.

    - Takes a list of netlist paths.
    - For each row of a pandas DataFrame (parameter values),
      it runs all netlists and collects .meas results.
    """

    def __init__(
        self,
        netlists: List[str | Path],
        param_mapping: Mapping[str, str] | None = None,
        measures: Iterable[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        netlists : list of str or Path
            Paths to SPICE netlists (.cir files). All must define the same .param and .meas.
        param_mapping : dict, optional
            Maps DataFrame column names -> SPICE parameter names.
            Example: {"R_val": "Rval", "C_val": "Cval"}.
            If None, DataFrame column names are assumed to be the SPICE parameter names.
        measures : iterable of str, optional
            Names of .meas to retrieve (e.g. ["fcut"]).
            If None, all measures from the first instance are used.
        """
        self.simulators: List[_Simulator] = []
        self.param_mapping: Dict[str, str] | None = dict(param_mapping) if param_mapping is not None else None

        # Create one NGSpiceInstance per netlist
        for path in netlists:
            p = Path(path)
            inst = NGSpiceInstance()
            inst.load(p)
            self.simulators.append(_Simulator(netlist_path=p, instance=inst))

        # Determine which measures to read
        if measures is None and self.simulators:
            measures = list(self.simulators[0].instance.list_measures())
        self.measures: List[str] = list(measures or [])

    # --------- Resource management helpers ---------

    def close(self) -> None:
        """Stop all NGSpice instances."""
        for sim in self.simulators:
            sim.instance.stop()

    def __enter__(self) -> "SequentialPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --------- Main API ---------

    def run(self, values: pd.DataFrame) -> pd.DataFrame:
        """
        Run all netlists for each row of 'values'.

        Parameters
        ----------
        values : pandas.DataFrame
            Each row contains a set of parameter values.
            Columns must match param_mapping keys (or SPICE parameter names if param_mapping is None).

        Returns
        -------
        pandas.DataFrame
            DataFrame containing:
            - original parameter columns
            - one column per (netlist, measure).
              If there is only one netlist, the column is simply the measure name (e.g. "fcut").
              If there are several netlists, columns are named "<measure>_<netlist_stem>".
        """
        records: List[Dict[str, Any]] = []

        for _, row in values.iterrows():
            # Start record with input parameters
            record: Dict[str, Any] = dict(row)

            for sim in self.simulators:
                inst = sim.instance

                # Set parameters on this instance
                if self.param_mapping:
                    for col_name, param_name in self.param_mapping.items():
                        value = float(row[col_name])
                        inst.set_parameter(param_name, value)
                else:
                    for col_name, value in row.items():
                        inst.set_parameter(col_name, float(value))

                # Run simulation
                inst.run()

                # Collect measures
                for meas_name in self.measures:
                    meas_value = inst.get_measure(meas_name)

                    if len(self.simulators) == 1:
                        col_name = meas_name
                    else:
                        col_name = f"{meas_name}_{sim.netlist_path.stem}"

                    record[col_name] = meas_value

            records.append(record)

        result_df = pd.DataFrame.from_records(records)
        return result_df
