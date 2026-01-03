import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reuse the logic from your other scripts to locate run folders
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _infer_cell_and_corner(netlist_path: Path) -> tuple[str, str]:
    stem = netlist_path.stem
    parts = stem.split("_")
    cell = parts[0] if parts and parts[0] else stem
    corner = parts[1] if len(parts) >= 2 and parts[1] else "tt"
    return cell, corner

def _resolve_run_dir(netlist_path: Path, run_id: str) -> Path:
    cell, corner = _infer_cell_and_corner(netlist_path)
    # Look into runs/cem/ this time
    base = _project_root() / "runs" / "cem" / cell / corner
    
    if not base.exists():
        print(f"Error: Base directory {base} does not exist.")
        sys.exit(1)

    # If run_id is empty, pick the latest one
    if not run_id:
        runs = sorted([p for p in base.glob("run*") if p.is_dir()], key=lambda p: p.name)
        if not runs:
            print(f"No runs found in {base}")
            sys.exit(1)
        return runs[-1]
    
    run_dir = base / run_id
    if not run_dir.exists():
        print(f"Error: Run directory {run_dir} does not exist.")
        sys.exit(1)
    
    return run_dir

def main():
    parser = argparse.ArgumentParser(description="Plot CEM convergence curves.")
    parser.add_argument("--run", type=str, default="", help="Run ID (e.g., run001). Default: latest run.")
    parser.add_argument("--netlist", type=str, default="../netlists/inv.cir", help="Path to netlist (to infer cell/corner).")
    parser.add_argument("--show", action="store_true", help="Display the plots on screen.")
    args = parser.parse_args()

    # 1. Locate the CSV file
    netlist = Path(args.netlist)
    run_dir = _resolve_run_dir(netlist, args.run)
    csv_path = run_dir / "cem_history.csv"
    
    print(f"--- Analyzing Run: {run_dir.name} ---")
    print(f"File: {csv_path}")

    if not csv_path.exists():
        print("Error: cem_history.csv not found in this run.")
        sys.exit(1)

    # 2. Load Data
    df = pd.read_csv(csv_path)

    # 3. Plot Configuration
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"CEM Analysis - {run_dir.name}", fontsize=16)

    iters = df['iter']

    # --- PLOT 1: Policy Convergence (Mean ± Std Dev) ---
    # WN
    axs[0].plot(iters, df['policy_mean_wn'], label='Mean Wn', color='blue', linewidth=2)
    axs[0].fill_between(iters, 
                        df['policy_mean_wn'] - df['policy_std_wn'], 
                        df['policy_mean_wn'] + df['policy_std_wn'], 
                        color='blue', alpha=0.1)
    # WP
    axs[0].plot(iters, df['policy_mean_wp'], label='Mean Wp', color='red', linewidth=2)
    axs[0].fill_between(iters, 
                        df['policy_mean_wp'] - df['policy_std_wp'], 
                        df['policy_mean_wp'] + df['policy_std_wp'], 
                        color='red', alpha=0.1)
    
    axs[0].set_title('Policy Evolution (Mean ± Std)')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Width (µm)')
    axs[0].legend()

    # --- PLOT 2: Reward Convergence ---
    axs[1].plot(iters, df['iter_best_reward'], label='Best (Iter)', marker='o', linestyle='--', color='gray', alpha=0.6)
    axs[1].plot(iters, df['global_best_reward'], label='Best (Global)', color='green', linewidth=2.5)
    
    axs[1].set_title('Reward Convergence')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Reward')
    axs[1].legend()

    # --- PLOT 3: Uncertainty Reduction (Sigma) ---
    axs[2].plot(iters, df['policy_std_wn'], label='Sigma Wn', color='blue', marker='x')
    axs[2].plot(iters, df['policy_std_wp'], label='Sigma Wp', color='red', marker='x')
    
    axs[2].set_title("Uncertainty Reduction (Sigma)")
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Std Dev')
    axs[2].legend()

    # 4. Save and Show
    output_img = run_dir / "cem_convergence_plot.png"
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    print(f"Plot saved to: {output_img}")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()