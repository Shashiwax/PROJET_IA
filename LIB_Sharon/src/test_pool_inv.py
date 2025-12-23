from pathlib import Path
import pandas as pd

from pools import ParallelPool

if __name__ == "__main__":
    netlist = Path("/home/sharo/PROJET_IA/LIB_Sharon/netlists/inv.cir")

    # Example batch of (wn, wp) in microns (same convention you used in the env)
    values = pd.DataFrame(
        {
            "wn": [0.65, 0.50, 0.40, 0.55],
            "wp": [1.00, 0.70, 0.90, 0.80],
        }
    )

    pool = ParallelPool([netlist], n_workers=3, timeout_s=25.0)
    df = pool.run(values)

    # Join inputs + outputs for easy reading
    out = pd.concat([values.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
    print(out.to_string(index=False))
