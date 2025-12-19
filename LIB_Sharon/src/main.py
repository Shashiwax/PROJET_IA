from pathlib import Path

import pandas as pd

from pool import SequentialPool


# Project root = one level above src/
ROOT_DIR = Path(__file__).resolve().parents[1]
NETLIST_DIR = ROOT_DIR / "netlists"


def main() -> None:
    print(">>> main.py (partie 3) started")

    # For now we use a single RC netlist.
    # You can later duplicate it (rc1.cir, rc2.cir, ...) if needed.
    netlists = [NETLIST_DIR / "rc.cir"]

    # DataFrame of parameter values:
    # columns names are R_val, C_val (Python side)
    values = pd.DataFrame(
        {
            "R_val": [1e3, 1e4, 1e3, 5e3],     # Ohms
            "C_val": [1e-6, 1e-6, 1e-7, 2e-6],  # Farads
        }
    )

    # Mapping from DataFrame columns -> SPICE .param names
    # In rc.cir you used: .param Rval = ..., .param Cval = ...
    param_mapping = {
        "R_val": "Rval",
        "C_val": "Cval",
    }

    # We know our netlist defines a .meas fcut
    measures = ["fcut"]

    # Use the pool as a context manager so that instances are closed automatically
    with SequentialPool(netlists, param_mapping=param_mapping, measures=measures) as pool:
        result_df = pool.run(values)

    print("\nResults DataFrame:")
    print(result_df)

    print("\n>>> Done.")


if __name__ == "__main__":
    main()
