import numpy as np
import pandas as pd
from pathlib import Path

from pools import ParallelPool


def clamp(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def main() -> None:
    # --- Paths ---
    netlist = Path("/home/sharo/PROJET_IA/LIB_Sharon/netlists/inv.cir")

    # --- Action bounds (match your env ranges) ---
    wn_min, wn_max = 0.36, 0.65
    wp_min, wp_max = 0.36, 1.00
    lo = np.array([wn_min, wp_min], dtype=np.float32)
    hi = np.array([wn_max, wp_max], dtype=np.float32)

    # --- ParallelPool setup (2 workers is safe on your laptop) ---
    cfg = RewardConfig(
        # IMPORTANT: align with your InverterEnv reward weights/refs if needed
        w_area=0.25,
        w_delay=0.45,
        w_pstat=0.20,
        w_edyn=0.10,
    )
    pool = ParallelPool([netlist], n_workers=2, timeout_s=25.0, reward_cfg=cfg, keep_raw=False)

    # --- CEM hyperparameters ---
    n_iters = 15          # number of training iterations
    batch_size = 32       # how many (wn, wp) evaluated per iteration
    elite_frac = 0.25     # top fraction used to update distribution
    alpha = 0.7           # smoothing (0=no smoothing, 1=full replace)

    # --- Initialize policy distribution: Gaussian over (wn, wp) ---
    mean = np.array([0.55, 0.80], dtype=np.float32)
    std = np.array([0.10, 0.15], dtype=np.float32)

    best = {"reward": -1e18, "wn": None, "wp": None}
    history_rows = []

    for it in range(1, n_iters + 1):
        # --- Sample a batch from current policy ---
        samples = np.random.randn(batch_size, 2).astype(np.float32) * std + mean
        samples = clamp(samples, lo, hi)

        values = pd.DataFrame({"wn": samples[:, 0], "wp": samples[:, 1]})
        df = pool.run(values)
        # --- Make Pool output "robot-friendly" (same units as your env) ---
        df["cell_area_um2"] = df["cell_area"]
        df["active_area_um2"] = df["active_area"]

        df["delay_fall_ps"] = df["delay_fall"] * 1e12
        df["delay_rise_ps"] = df["delay_rise"] * 1e12

        df["pstat_vdd_in0_uW"] = df["pstat_vdd_in0"] * 1e6
        df["pstat_vdd_in1_uW"] = df["pstat_vdd_in1"] * 1e6
        df["pstat_wc_uW"] = df[["pstat_vdd_in0_uW", "pstat_vdd_in1_uW"]].max(axis=1)

        df["edyn_fJ"] = df["edyn_val"] * 1e15

        # --- Reward (normalized weighted cost, negative) ---
        ref_area = 3.7536
        ref_delay_sum = (5.000321 + 18.96081)
        ref_pstat = 1.295751
        ref_edyn = 1.96807

        w_area, w_delay, w_pstat, w_edyn = 0.25, 0.45, 0.20, 0.10

        delay_sum = df["delay_fall_ps"] + df["delay_rise_ps"]

        cost = (
            w_area * (df["cell_area_um2"] / ref_area)
            + w_delay * (delay_sum / ref_delay_sum)
            + w_pstat * (df["pstat_wc_uW"] / ref_pstat)
            + w_edyn * (df["edyn_fJ"] / ref_edyn)
        )

        df["reward"] = -cost

        # --- Filter failed sims ---
        if "error" in df.columns:
            ok = df["error"].isna() | (df["error"] == "")
            df_ok = df[ok].copy()
        else:
            df_ok = df.copy()

        # If everything failed (rare but possible), skip update
        if len(df_ok) == 0:
            print(f"[iter {it:02d}] all simulations failed -> skipping update")
            continue

        # --- Select elites by reward ---
        df_ok = df_ok.sort_values("reward", ascending=False).reset_index(drop=True)
        n_elite = max(2, int(elite_frac * len(df_ok)))
        elites = df_ok.head(n_elite)

        elite_actions = elites[["wn_in", "wp_in"]].to_numpy(dtype=np.float32)
        elite_mean = elite_actions.mean(axis=0)
        elite_std = elite_actions.std(axis=0) + 1e-6  # avoid collapse to zero

        # --- Smooth update ---
        mean = (1 - alpha) * mean + alpha * elite_mean
        std = (1 - alpha) * std + alpha * elite_std

        # --- Track best ---
        top = df_ok.iloc[0]
        if float(top["reward"]) > best["reward"]:
            best["reward"] = float(top["reward"])
            best["wn"] = float(top["wn_in"])
            best["wp"] = float(top["wp_in"])

        # --- Log ---
        print(
            f"[iter {it:02d}] "
            f"best_reward={float(top['reward']):.6f} "
            f"mean=(wn={mean[0]:.4f}, wp={mean[1]:.4f}) "
            f"std=(wn={std[0]:.4f}, wp={std[1]:.4f}) "
            f"global_best=(r={best['reward']:.6f}, wn={best['wn']:.4f}, wp={best['wp']:.4f})"
        )

        # Keep a row for later plotting/analysis
        history_rows.append(
            {
                "iter": it,
                "iter_best_reward": float(top["reward"]),
                "iter_best_wn": float(top["wn_in"]),
                "iter_best_wp": float(top["wp_in"]),
                "policy_mean_wn": float(mean[0]),
                "policy_mean_wp": float(mean[1]),
                "policy_std_wn": float(std[0]),
                "policy_std_wp": float(std[1]),
                "global_best_reward": float(best["reward"]),
                "global_best_wn": float(best["wn"]),
                "global_best_wp": float(best["wp"]),
            }
        )

    hist = pd.DataFrame(history_rows)
    hist.to_csv("cem_history.csv", index=False)

    print("\n=== FINAL BEST (by reward) ===")
    print(best)
    print("Saved: cem_history.csv")


if __name__ == "__main__":
    main()
