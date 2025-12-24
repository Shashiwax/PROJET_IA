import numpy as np
import pandas as pd
from pathlib import Path

from pools import ParallelPool, RewardConfig


def _clamp(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def _snap_to_bins(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """
    Snap each element of x to the nearest value in bins.
    """
    # x: (N,)
    # bins: (M,)
    idx = np.abs(x[:, None] - bins[None, :]).argmin(axis=1)
    return bins[idx]


def main() -> None:
    netlist = Path("/home/sharo/PROJET_IA/LIB_Sharon/netlists/inv.cir")

    # PDK-friendly width bins (subset up to 2.0um; edit if you want a different range)
    # English-only comments as requested.
    W_BINS = np.array(
        [
            0.36, 0.39, 0.42,
            0.52, 0.54, 0.55, 0.58, 0.60, 0.61, 0.63, 0.64, 0.65,
            0.70, 0.74, 0.75, 0.84,
            1.00, 1.12, 1.26, 1.65, 1.68,
            2.00,
        ],
        dtype=np.float32,
    )

    # Search bounds (continuous sampling is clamped, then snapped to bins)
    wn_min, wn_max = 0.36, 2.00
    wp_min, wp_max = 0.36, 2.00
    lo = np.array([wn_min, wp_min], dtype=np.float32)
    hi = np.array([wn_max, wp_max], dtype=np.float32)

    # Reward config: baseline + weights + hard logic thresholds (0.95/0.05)
    cfg = RewardConfig(
        ref_cell_area_um2=3.7536,
        ref_delay_max_ps=18.96081,
        ref_pstat_wc_uW=1.295751,
        ref_edyn_fJ=1.96807,
        w_area=0.25,
        w_delay=0.35,
        w_pstat=0.25,
        w_edyn=0.15,
        w_size=0.02,
        wn0=0.65,
        wp0=1.0,
        vdd=1.8,
        yhi_ratio=0.95,
        ylo_ratio=0.05,
        fail_reward=-1e6,
    )

    pool = ParallelPool(
        [netlist],
        n_workers=2,
        timeout_s=25.0,
        reward_cfg=cfg,
        keep_raw=False,
    )

    rng = np.random.default_rng(0)

    # CEM hyperparameters
    n_iters = 20
    batch_size = 48
    elite_frac = 0.25
    alpha = 0.7

    # Policy distribution over (wn, wp)
    mean = np.array([0.65, 1.00], dtype=np.float32)
    std = np.array([0.30, 0.35], dtype=np.float32)

    best_reward = -1e18
    best_wn = None
    best_wp = None

    history = []

    for it in range(1, n_iters + 1):
        # Sample continuous
        samples = rng.standard_normal(size=(batch_size, 2), dtype=np.float32) * std + mean
        samples = _clamp(samples, lo, hi)

        # Snap to PDK bins
        wn = _snap_to_bins(samples[:, 0], W_BINS)
        wp = _snap_to_bins(samples[:, 1], W_BINS)

        values = pd.DataFrame({"wn": wn, "wp": wp})
        df = pool.run(values)

        # Filter failures
        if "error" in df.columns:
            ok = df["error"].isna() | (df["error"] == "")
            df_ok = df[ok].copy()
        else:
            df_ok = df.copy()

        # Also drop hard-constraint failures (reward = fail_reward)
        df_ok = df_ok[df_ok["reward"] > cfg.fail_reward / 10].copy()

        if len(df_ok) == 0:
            print(f"[iter {it:02d}] all candidates failed -> widening std slightly")
            std = np.minimum(std * 1.10, np.array([0.60, 0.80], dtype=np.float32))
            continue

        # Sort by reward (descending)
        df_ok = df_ok.sort_values("reward", ascending=False).reset_index(drop=True)
        top = df_ok.iloc[0]

        # Track best so far
        if float(top["reward"]) > best_reward:
            best_reward = float(top["reward"])
            best_wn = float(top["wn_in"])
            best_wp = float(top["wp_in"])

        # Elite selection
        n_elite = max(2, int(elite_frac * len(df_ok)))
        elites = df_ok.head(n_elite)
        elite_actions = elites[["wn_in", "wp_in"]].to_numpy(dtype=np.float32)

        elite_mean = elite_actions.mean(axis=0)
        elite_std = elite_actions.std(axis=0) + 1e-6

        # Smooth update
        mean = (1 - alpha) * mean + alpha * elite_mean
        std = (1 - alpha) * std + alpha * elite_std

        print(
            f"[iter {it:02d}] "
            f"iter_best_r={float(top['reward']):.6f} "
            f"policy_mean=(wn={mean[0]:.4f}, wp={mean[1]:.4f}) "
            f"policy_std=(wn={std[0]:.4f}, wp={std[1]:.4f}) "
            f"global_best=(r={best_reward:.6f}, wn={best_wn:.4f}, wp={best_wp:.4f})"
        )

        history.append(
            {
                "iter": it,
                "iter_best_reward": float(top["reward"]),
                "iter_best_wn": float(top["wn_in"]),
                "iter_best_wp": float(top["wp_in"]),
                "policy_mean_wn": float(mean[0]),
                "policy_mean_wp": float(mean[1]),
                "policy_std_wn": float(std[0]),
                "policy_std_wp": float(std[1]),
                "global_best_reward": float(best_reward),
                "global_best_wn": float(best_wn),
                "global_best_wp": float(best_wp),
            }
        )

    hist_df = pd.DataFrame(history)
    hist_df.to_csv("cem_history.csv", index=False)

    print("\n=== FINAL BEST (by reward) ===")
    print({"reward": best_reward, "wn": best_wn, "wp": best_wp})
    print("Saved: cem_history.csv")

    if best_wn is not None and best_wp is not None:
        df_best = pool.run(pd.DataFrame({"wn": [best_wn], "wp": [best_wp]}))
        print("\n=== BEST POINT METRICS ===")
        print(df_best.to_string(index=False))


if __name__ == "__main__":
    main()
