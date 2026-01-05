import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def _resolve_run_dir(run_arg: str) -> Path:
    # (Même logique que tes autres scripts pour trouver le dossier)
    root = Path(__file__).resolve().parents[1]
    base = root / "runs" / "cem" / "inv" / "tt" # Adapté pour inv/tt par défaut
    
    if not run_arg:
        runs = sorted([p for p in base.glob("run*") if p.is_dir()], key=lambda p: p.name)
        return runs[-1]
    return base / run_arg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="", help="Run ID (ex: run005)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run)
    csv_path = run_dir / "cem_history.csv"
    
    print(f"Chargement : {csv_path}")
    if not csv_path.exists():
        print("Erreur: Fichier introuvable.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Vérification que les nouvelles colonnes existent
    required = ["iter_best_delay", "iter_best_area", "iter_best_pstat", "iter_best_edyn"]
    if not all(col in df.columns for col in required):
        print("ERREUR: Les métriques détaillées (delay, area...) sont absentes du CSV.")
        print("Avez-vous modifié train_cem_pool.py et relancé l'entraînement ?")
        sys.exit(1)

    sns.set_theme(style="whitegrid")
    
    # --- FIGURE 1 : Evolution des 4 Specs par itération ---
    fig1, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle(f"Convergence des Spécifications - {run_dir.name}", fontsize=16)

    # Delay
    sns.lineplot(data=df, x='iter', y='iter_best_delay', ax=axs[0,0], marker='o', color='tab:orange')
    axs[0,0].set_title("Délai (ps)")
    axs[0,0].set_ylabel("ps")

    # Area
    sns.lineplot(data=df, x='iter', y='iter_best_area', ax=axs[0,1], marker='o', color='tab:blue')
    axs[0,1].set_title("Surface (um²)")
    axs[0,1].set_ylabel("um²")

    # Power Static
    sns.lineplot(data=df, x='iter', y='iter_best_pstat', ax=axs[1,0], marker='o', color='tab:red')
    axs[1,0].set_title("Puissance Statique (uW)")
    axs[1,0].set_ylabel("uW")

    # Energy Dynamic
    sns.lineplot(data=df, x='iter', y='iter_best_edyn', ax=axs[1,1], marker='o', color='tab:green')
    axs[1,1].set_title("Énergie Dynamique (fJ)")
    axs[1,1].set_ylabel("fJ")

    plt.tight_layout()
    fig1.savefig(run_dir / "cem_metrics_convergence.png")
    print(f"Sauvegardé : {run_dir / 'cem_metrics_convergence.png'}")

    # --- FIGURE 2 : Wn/Wp vs Performance (Scatter) ---
    # On veut voir comment Wn et Wp influencent le Délai (le trade-off principal)
    fig2, axs2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle(f"Impact de la Taille sur le Délai - {run_dir.name}", fontsize=16)

    # Wn vs Delay
    sns.scatterplot(data=df, x='iter_best_wn', y='iter_best_delay', ax=axs2[0], hue='iter', palette='viridis', s=100)
    axs2[0].set_title("Wn vs Délai")
    axs2[0].set_xlabel("Largeur NMOS (um)")
    axs2[0].set_ylabel("Délai (ps)")
    
    # Wp vs Delay
    sns.scatterplot(data=df, x='iter_best_wp', y='iter_best_delay', ax=axs2[1], hue='iter', palette='viridis', s=100)
    axs2[1].set_title("Wp vs Délai")
    axs2[1].set_xlabel("Largeur PMOS (um)")
    axs2[1].set_ylabel("Délai (ps)")

    plt.tight_layout()
    fig2.savefig(run_dir / "cem_design_space.png")
    print(f"Sauvegardé : {run_dir / 'cem_design_space.png'}")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()