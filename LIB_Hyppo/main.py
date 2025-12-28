import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from loguru import logger
import warnings
import RL_Nand

# --- FILTRAGE WARNINGS ---
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- CONFIGURATION ---
DELAY_NETLIST = "LIB_Hyppo/nand_delay.cir"
POWER_NETLIST = "LIB_Hyppo/nand_power.cir"
MODEL_NAME = "sac_nand_normalized" 
TRAINING_STEPS = 5000
LEARNING_RATE = 0.0003

# AUGMENTATION MASSIVE POUR AVOIR UN BEAU PARETO
# Avec ton optimisation de vitesse, 2000 sims devraient prendre ~1 à 2 minutes max
N_RANDOM_SAMPLES = 2000 

MODE = "VERIFICATION" # "TRAINING" ou "VERIFICATION"

def get_env():
    if not os.path.exists(DELAY_NETLIST) or not os.path.exists(POWER_NETLIST):
        logger.error("Netlists introuvables.")
        sys.exit(1)
    return RL_Nand.NANDOpt(delay_netlist=DELAY_NETLIST, power_netlist=POWER_NETLIST)

def get_pareto_frontier(Xs, Ys, maxX=False, maxY=False):
    """
    Calcule la frontière de Pareto pour un nuage de points 2D.
    Retourne les listes X et Y triées des points optimaux.
    """
    # On combine les listes et on trie par X
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], key=lambda x: x[0])
    
    pareto_front = [sorted_list[0]]
    
    for pair in sorted_list[1:]:
        if maxY: 
            if pair[1] >= pareto_front[-1][1]: # Si on maximise Y
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]: # Si on minimise Y (cas standard ici)
                pareto_front.append(pair)
    
    # Séparation pour le plot
    p_xs = [x[0] for x in pareto_front]
    p_ys = [x[1] for x in pareto_front]
    return p_xs, p_ys

def train_new_model():
    env = get_env()
    # Vectorisation et Normalisation
    vec_env = DummyVecEnv([lambda: env])
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=100.)
    
    logger.info(f"Entraînement ({TRAINING_STEPS} steps)...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=LEARNING_RATE)
    model.learn(total_timesteps=TRAINING_STEPS)
    
    model.save(MODEL_NAME)
    env.save(f"{MODEL_NAME}_vecnormalize.pkl")
    logger.success("Modèle sauvegardé.")
    return model

def run_verification_and_plot(model=None):
    raw_env = get_env()
    vec_env = DummyVecEnv([lambda: raw_env])
    
    # Chargement normalisation
    stats_path = f"{MODEL_NAME}_vecnormalize.pkl"
    if os.path.exists(stats_path):
        env_norm = VecNormalize.load(stats_path, vec_env)
        env_norm.training = False
        env_norm.norm_reward = False
    else:
        env_norm = vec_env

    # Chargement modèle
    if model is None:
        if not os.path.exists(f"{MODEL_NAME}.zip"):
            logger.error("Modèle introuvable.")
            return
        model = PPO.load(MODEL_NAME, env=env_norm)

    # --- 1. POINT IA ---
    logger.info("Calcul du point optimal IA...")
    obs = env_norm.reset()
    last_info = {}
    ai_best_action = None
    
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, infos = env_norm.step(action)
        last_info = infos[0]
        ai_best_action = action[0]

    # Récupération IA (Unités physiques)
    ai_metrics = {
        'delay': last_info.get('delay', 0.0) / 10.0,    # ns
        'area': last_info.get('area', 0.0),             # um2
        'p_stat': last_info.get('p_stat', 0.0),         # nW (déjà à l'échelle si RL_Nand divise par 1e-9)
        'p_dyn': last_info.get('p_dyn', 0.0),           # uW (déjà à l'échelle si RL_Nand divise par 1e-6)
    }
    ai_metrics['p_tot'] = (ai_metrics['p_stat']*1e-3) + ai_metrics['p_dyn'] # uW

    print(f"IA Result -> Delay: {ai_metrics['delay']:.4f}ns | Area: {ai_metrics['area']:.2f}um2")

    # --- 2. GENERATION MASSIVE (Monte Carlo) ---
    logger.info(f"Génération du nuage de {N_RANDOM_SAMPLES} points (Patientez ~1min)...")
    logger.disable("pyngs")
    
    # Listes pour stocker les données brutes
    data = {'delay': [], 'area': [], 'p_tot': [], 'p_dyn': [], 'p_stat': []}
    
    # On utilise raw_env pour sampler (plus simple)
    # Note: On échantillonne directement l'espace physique ou l'espace d'action
    
    for i in range(N_RANDOM_SAMPLES):
        if i % 100 == 0: print(f"Sampling {i}/{N_RANDOM_SAMPLES}...", end='\r')
        
        act = raw_env.action_space.sample()
        _, _, _, _, info = raw_env.step(act)
        
        if info.get('delay', 100.0) < 10.0: # Filtre crashs
            d = info['delay'] / 10.0
            a = info['area']
            pt = (info['p_stat']*1e-9 + info['p_dyn']*1e-6) * 1e6 # uW
            
            data['delay'].append(d)
            data['area'].append(a)
            data['p_tot'].append(pt)
            data['p_stat'].append(info['p_stat']) # nW
            data['p_dyn'].append(info['p_dyn'])   # uW

    logger.enable("pyngs")
    print("\nCalcul des fronts de Pareto...")

    # --- 3. PLOTTING ---
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Optimisation IA vs Espace des Possibles ({N_RANDOM_SAMPLES} samples)", fontsize=16)

    def plot_pareto(ax, x_key, y_key, xlabel, ylabel, title):
        X = data[x_key]
        Y = data[y_key]
        
        # 1. Nuage de points (Gris, haute transparence)
        ax.scatter(X, Y, c='gray', alpha=0.3, s=20, label='Designs Possibles', edgecolors='none')
        
        # 2. Front de Pareto (Calculé mathématiquement)
        # On minimise X et Y dans tous les cas ici (Delay, Power, Area sont tous à minimiser)
        p_x, p_y = get_pareto_frontier(X, Y)
        ax.plot(p_x, p_y, 'b-', linewidth=2, label='Front de Pareto', alpha=0.8)
        ax.scatter(p_x, p_y, c='blue', s=30)

        # 3. Point IA
        ax_x = ai_metrics[x_key]
        ax_y = ai_metrics[y_key]
        ax.scatter(ax_x, ax_y, c='red', marker='*', s=300, label='IA (PPO)', zorder=10, edgecolors='black')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.4)
        ax.legend()

    # Graphique 1 : Delay vs Area (Celui que tu voulais voir)
    plot_pareto(axs[0, 0], 'delay', 'area', 'Délai (ns)', 'Surface (µm²)', 'Trade-off : Vitesse vs Coût')
    
    # Graphique 2 : Delay vs Power
    plot_pareto(axs[0, 1], 'delay', 'p_tot', 'Délai (ns)', 'Puissance Totale (µW)', 'Trade-off : Vitesse vs Conso')
    
    # Graphique 3 : Power vs Area
    plot_pareto(axs[1, 0], 'p_tot', 'area', 'Puissance (µW)', 'Surface (µm²)', 'Densité de Puissance')
    
    # Graphique 4 : Stat vs Dyn
    plot_pareto(axs[1, 1], 'p_dyn', 'p_stat', 'Puissance Dyn (µW)', 'Puissance Stat (nW)', 'Analyse des Fuites')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    if MODE == "TRAINING":
        model = train_new_model()
        run_verification_and_plot(model)
    elif MODE == "VERIFICATION":
        run_verification_and_plot()