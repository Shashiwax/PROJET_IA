import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pools import ParallelPool, RewardConfig


def _project_root() -> Path:
    # Assumes this file lives in <root>/src/
    return Path(__file__).resolve().parents[1]


def _infer_cell_and_corner(netlist_path: Path) -> tuple[str, str]:
    # netlist stem like "inv" or "inv_tt"
    stem = netlist_path.stem
    parts = stem.split("_")
    cell = parts[0] if parts and parts[0] else stem
    corner = parts[1] if len(parts) >= 2 and parts[1] else "tt"
    return cell, corner


def _next_run_id(base_dir: Path) -> str:
    best = 0
    for p in base_dir.glob("run[0-9][0-9][0-9]"):
        if not p.is_dir():
            continue
        try:
            n = int(p.name.replace("run", ""))
            best = max(best, n)
        except ValueError:
            pass
    return f"run{best + 1:03d}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run",
        type=str,
        default="",
        help="Run folder name to reuse (e.g. run001). If omitted, auto-creates runXXX.",
    )
    return p.parse_args()


def _resolve_run_dir(netlist_path: Path, run_id: str) -> tuple[Path, str, str, str]:
    cell, corner = _infer_cell_and_corner(netlist_path)
    base = _project_root() / "runs" / "cem" / cell / corner
    base.mkdir(parents=True, exist_ok=True)

    rid = run_id.strip()
    if not rid:
        rid = _next_run_id(base)

    run_dir = base / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, rid, cell, corner


def _clamp(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def _snap_to_bins(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """
    Snap each element of x to the nearest value in bins.
    """
    idx = np.abs(x[:, None] - bins[None, :]).argmin(axis=1)
    return bins[idx]


def main() -> None:
    # Keep your default behavior; allow override via env if you want.
    netlist = Path(
        str(
            Path(
                # If you want to override without editing the code:
                # NETLIST=/path/to/inv.cir uv run python train_cem_pool.py
                # Otherwise it keeps your original absolute path style.
                # (This is optional; remove env usage if you want strictly no changes.)
                __import__("os").getenv("NETLIST", "/home/sharo/PROJET_IA/LIB_Sharon/netlists/inv.cir")
            )
        )
    )

    # PDK-friendly width bins 
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

    # Search bounds 
    wn_min, wn_max = 0.36, 2.00
    wp_min, wp_max = 0.36, 2.00
    lo = np.array([wn_min, wp_min], dtype=np.float32)
    hi = np.array([wn_max, wp_max], dtype=np.float32)

    # Reward configuration
    cfg = RewardConfig()

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
    batch_size = 96
    elite_frac = 0.25
    alpha = 0.3

    # Policy distribution over (wn, wp) 
    mean = np.array([0.65, 1.00], dtype=np.float32)
    std = np.array([0.30, 0.35], dtype=np.float32)

    # -------------------------
    # NEW: Clean run folder (file management only)
    # -------------------------
    args = _parse_args()
    run_dir, RUN_ID, CELL_NAME, CORNER = _resolve_run_dir(netlist, args.run)

    history_csv = run_dir / "cem_history.csv"
    best_metrics_csv = run_dir / "best_point_metrics.csv"
    best_summary_json = run_dir / "best_summary.json"
    config_json = run_dir / "config.json"

    # Write config once per run (do not overwrite if it already exists)
    if not config_json.exists():
        config_dump = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "algo": "cem",
            "run_id": RUN_ID,
            "cell": CELL_NAME,
            "corner": CORNER,
            "netlist": str(netlist),
            "cem": {
                "n_iters": n_iters,
                "batch_size": batch_size,
                "elite_frac": elite_frac,
                "alpha": alpha,
                "mean_init": mean.tolist(),
                "std_init": std.tolist(),
                "wn_bounds": [wn_min, wn_max],
                "wp_bounds": [wp_min, wp_max],
                "w_bins": W_BINS.tolist(),
                "rng_seed": 0,
                "pool": {"n_workers": 4, "timeout_s": 25.0},
            },
            "reward_cfg": cfg.__dict__,
        }
        config_json.write_text(json.dumps(config_dump, indent=2) + "\n")

    # -------------------------
    # CEM loop (unchanged logic)
    # -------------------------
    best_reward = -1e18
    best_wn = None
    best_wp = None

    history = []

    for it in range(1, n_iters + 1):
        samples = rng.standard_normal(size=(batch_size, 2), dtype=np.float32) * std + mean
        samples = _clamp(samples, lo, hi)

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

        df_ok = df_ok[df_ok["reward"] > cfg.fail_reward / 10].copy()

        if len(df_ok) == 0:
            print(f"[iter {it:02d}] all candidates failed -> widening std slightly")
            std = np.minimum(std * 1.10, np.array([0.60, 0.80], dtype=np.float32))
            continue

        df_ok = df_ok.sort_values("reward", ascending=False).reset_index(drop=True)
        top = df_ok.iloc[0]

        if float(top["reward"]) > best_reward:
            best_reward = float(top["reward"])
            best_wn = float(top["wn_in"])
            best_wp = float(top["wp_in"])

        n_elite = max(2, int(elite_frac * len(df_ok)))
        elites = df_ok.head(n_elite)
        elite_actions = elites[["wn_in", "wp_in"]].to_numpy(dtype=np.float32)

        elite_mean = elite_actions.mean(axis=0)
        elite_std = elite_actions.std(axis=0) + 1e-6

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
                
                # --- AJOUTER CES LIGNES ---
                "iter_best_delay": float(top["delay_max_ps"]),
                "iter_best_area": float(top["cell_area_um2"]),
                "iter_best_pstat": float(top["pstat_wc_uW"]),
                "iter_best_edyn": float(top["edyn_fJ"]),
                # --------------------------

                "policy_mean_wn": float(mean[0]),
                "policy_mean_wp": float(mean[1]),
                "policy_std_wn": float(std[0]),
                "policy_std_wp": float(std[1]),
                "global_best_reward": float(best_reward),
                "global_best_wn": float(best_wn),
                "global_best_wp": float(best_wp),
            }
        )

    # -------------------------
    # NEW: Save outputs inside run_dir
    # -------------------------
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(history_csv, index=False)

    print("\n=== FINAL BEST (by reward) ===")
    print({"reward": best_reward, "wn": best_wn, "wp": best_wp})
    print(f"Saved: {history_csv}")

    if best_wn is not None and best_wp is not None:
        df_best = pool.run(pd.DataFrame({"wn": [best_wn], "wp": [best_wp]}))
        df_best.to_csv(best_metrics_csv, index=False)

        best_summary = {
            "run_id": RUN_ID,
            "run_dir": str(run_dir),
            "best_reward": float(best_reward),
            "best_wn": float(best_wn),
            "best_wp": float(best_wp),
        }
        best_summary_json.write_text(json.dumps(best_summary, indent=2) + "\n")

        print("\n=== BEST POINT METRICS ===")
        print(df_best.to_string(index=False))
        print(f"Saved: {best_metrics_csv}")
        print(f"Saved: {best_summary_json}")

    print(f"\nRun folder: {run_dir}")


if __name__ == "__main__":
    main()
