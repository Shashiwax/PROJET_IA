import numpy as np
import pandas as pd
from pathlib import Path

from pools import ParallelPool, RewardConfig


def _clamp(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    Clamp each dimension of x to [lo, hi].
    """
    return np.minimum(np.maximum(x, lo), hi)


def main() -> None:
    # --- Netlist ---
    netlist = Path("/home/sharo/PROJET_IA/LIB_Sharon/netlists/inv.cir")

    # --- Action bounds (match what you want for design search) ---
    wn_min, wn_max = 0.36, 0.65
    wp_min, wp_max = 0.36, 1.00
    lo = np.array([wn_min, wp_min], dtype=np.float32)
    hi = np.array([wn_max, wp_max], dtype=np.float32)

    # --- Reward config (must match the reward logic you want) ---
    cfg = RewardConfig(
        w_area=0.25,
        w_delay=0.45,
        w_pstat=0.20,
        w_edyn=0.10,
        # Keep the reference point = your "baseline" (defaults are OK if your baseline is the initial cell)
        ref_cell_area_um2=3.7536,
        ref_delay_sum_ps=(5.000321 + 18.96081),
        ref_pstat_wc_uW=1.295751,
        ref_edyn_fJ=1.96807,
    )

    # --- Pool ---
    pool = ParallelPool(
        [netlist],
        n_workers=2,
        timeout_s=25.0,
        reward_cfg=cfg,
        keep_raw=False,
    )

    # --- CEM hyperparameters (batch RL / policy search) ---
    rng = np.random.default_rng(0)

    n_iters = 15
    batch_size = 32
    elite_frac = 0.25
    alpha = 0.7

    # --- Policy distribution: Gaussian over [wn, wp] ---
    mean = np.array([0.55, 0.80], dtype=np.float32)
    std = np.array([0.10, 0.15], dtype=np.float32)

    best_reward = -1e18
    best_wn = None
    best_wp = None

    history = []

    for it in range(1, n_iters + 1):
        # Sample a batch from the current policy
        samples = rng.standard_normal(size=(batch_size, 2), dtype=np.float32) * std + mean
        samples = _clamp(samples, lo, hi)

        values = pd.DataFrame({"wn": samples[:, 0], "wp": samples[:, 1]})
        df = pool.run(values)

        # Filter failed sims (if any)
        if "error" in df.columns:
            ok = df["error"].isna() | (df["error"] == "")
            df_ok = df[ok].copy()
        else:
            df_ok = df.copy()

        if len(df_ok) == 0:
            print(f"[iter {it:02d}] all simulations failed -> skipping update")
            continue

        # Sort by reward (descending)
        df_ok = df_ok.sort_values("reward", ascending=False).reset_index(drop=True)

        # Track best so far
        top = df_ok.iloc[0]
        if float(top["reward"]) > best_reward:
            best_reward = float(top["reward"])
            best_wn = float(top["wn_in"])
            best_wp = float(top["wp_in"])

        # Select elites
        n_elite = max(2, int(elite_frac * len(df_ok)))
        elites = df_ok.head(n_elite)
        elite_actions = elites[["wn_in", "wp_in"]].to_numpy(dtype=np.float32)

        elite_mean = elite_actions.mean(axis=0)
        elite_std = elite_actions.std(axis=0) + 1e-6

        # Smooth update (policy improvement)
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

    # Save training trace
    hist_df = pd.DataFrame(history)
    hist_df.to_csv("cem_history.csv", index=False)

    print("\n=== FINAL BEST (by reward) ===")
    print({"reward": best_reward, "wn": best_wn, "wp": best_wp})
    print("Saved: cem_history.csv")

    # Re-evaluate best (one clean sim) without creating another file
    if best_wn is not None and best_wp is not None:
        df_best = pool.run(pd.DataFrame({"wn": [best_wn], "wp": [best_wp]}))
        print("\n=== BEST POINT METRICS ===")
        print(df_best.to_string(index=False))


if __name__ == "__main__":
    main()
