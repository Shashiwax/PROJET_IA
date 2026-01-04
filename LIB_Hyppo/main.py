import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from loguru import logger
import warnings
import RL_Nand 

# --- FILTRAGE WARNINGS ---
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- CONFIGURATION ---
DELAY_NETLIST = "LIB_Hyppo/nand_delay.cir"
POWER_NETLIST = "LIB_Hyppo/nand_power.cir"
MODEL_NAME = "ppo_nand_normalized" 
LOG_FILE = f"{MODEL_NAME}_training_log.pkl"

TRAINING_STEPS = 10000
LEARNING_RATE = 0.0006
N_RANDOM_SAMPLES = 2000 

MODE = "TRAINING" # "TRAINING" ou "VERIFICATION"

# --- CLASSE CALLBACK POUR LOGGUER LES METRIQUES INTERNES ---
class TrainingMetricsCallback(BaseCallback):
    """
    Callback personnalisé pour enregistrer l'évolution des actions (Ws, Wp),
    de l'incertitude (Sigma) et des récompenses pendant l'entraînement.
    """
    def __init__(self, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.data = {
            'iterations': [],
            'rewards': [],
            'action_mean_0': [], # Ex: Wn
            'action_mean_1': [], # Ex: Wp
            'sigma_0': [],       # Incertitude sur Wn
            'sigma_1': []        # Incertitude sur Wp
        }

    def _on_step(self) -> bool:
        # On enregistre toutes les étapes (ou modulo X si trop lourd)
        self.data['iterations'].append(self.num_timesteps)
        
        # 1. Récupération des Rewards (Reward brut du step)
        # Note: 'rewards' est un array (car vectorisé), on prend la moyenne
        if 'rewards' in self.locals:
            self.data['rewards'].append(np.mean(self.locals['rewards']))
            
        # 2. Récupération des Actions choisies (Moyenne du batch)
        if 'actions' in self.locals:
            actions = self.locals['actions']
            # On suppose que l'action a au moins 2 dimensions (Wn, Wp)
            if actions.shape[1] >= 2:
                self.data['action_mean_0'].append(np.mean(actions[:, 0]))
                self.data['action_mean_1'].append(np.mean(actions[:, 1]))
            else:
                self.data['action_mean_0'].append(np.mean(actions[:, 0]))
                self.data['action_mean_1'].append(0)

        # 3. Récupération du Sigma (Incertitude/Exploration)
        # PPO stocke log_std. On fait exp() pour avoir l'écart-type.
        if hasattr(self.model.policy, 'log_std'):
            log_std = self.model.policy.log_std.detach().cpu().numpy()
            std = np.exp(log_std)
            if len(std) >= 2:
                self.data['sigma_0'].append(std[0])
                self.data['sigma_1'].append(std[1])
            else:
                self.data['sigma_0'].append(std[0])
                self.data['sigma_1'].append(0)
                
        return True

    def save_data(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
            
# --- FONCTIONS UTILITAIRES ---

def get_env():
    if not os.path.exists(DELAY_NETLIST) or not os.path.exists(POWER_NETLIST):
        logger.error("Netlists introuvables.")
        sys.exit(1)
    
    # Création de l'env
    env = RL_Nand.NANDOpt(delay_netlist=DELAY_NETLIST, power_netlist=POWER_NETLIST)
    # Ajout du Monitor pour faciliter le logging standard SB3 si besoin
    env = Monitor(env) 
    return env

def get_pareto_frontier(Xs, Ys, maxX=False, maxY=False):
    """ Calcule la frontière de Pareto pour un nuage de points 2D. """
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], key=lambda x: x[0])
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY: 
            if pair[1] >= pareto_front[-1][1]: pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]: pareto_front.append(pair)
    
    p_xs = [x[0] for x in pareto_front]
    p_ys = [x[1] for x in pareto_front]
    return p_xs, p_ys

# --- FONCTIONS PRINCIPALES ---

def train_new_model():
    env = get_env()
    vec_env = DummyVecEnv([lambda: env])
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=100.)
    
    logger.info(f"Entraînement ({TRAINING_STEPS} steps)...")
    
    # Création du callback
    metrics_callback = TrainingMetricsCallback()
    
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=LEARNING_RATE, n_steps = 256, batch_size=64, n_epochs=10)
    # Ajout du callback dans learn()
    model.learn(total_timesteps=TRAINING_STEPS, callback=metrics_callback)
    
    model.save(MODEL_NAME)
    env.save(f"{MODEL_NAME}_vecnormalize.pkl")
    
    # Sauvegarde des données de convergence pour le plot
    metrics_callback.save_data(LOG_FILE)
    logger.success(f"Modèle et logs sauvegardés ({LOG_FILE}).")
    
    return model

def plot_training_convergence():
    """ Affiche la fenêtre avec les courbes d'apprentissage (Figure 2) """
    if not os.path.exists(LOG_FILE):
        logger.warning("Fichier de log d'entraînement introuvable. Pas de plot de convergence.")
        return

    with open(LOG_FILE, 'rb') as f:
        data = pickle.load(f)

    # Récupération des données
    iters = np.array(data['iterations'])
    # Lissage de la reward (fenêtre glissante) car très bruité
    rewards = data['rewards']
    window_size = 50
    if len(rewards) > window_size:
        rewards_smooth = np.convolve(rewards, np.ones(window_size)/window_size, mode='same')
    else:
        rewards_smooth = rewards

    # --- FIGURE CONVERGENCE ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'PPO Training Analysis - {MODEL_NAME}', fontsize=16)

    # 1. Action Evolution
    # On plotte la moyenne des actions Wn et Wp
    ax1.plot(iters, data['action_mean_0'], label='Mean Action 0 (Ws)', color='blue', alpha=0.7)
    ax1.plot(iters, data['action_mean_1'], label='Mean Action 1 (Wp)', color='red', alpha=0.7)
    
    # Zone d'incertitude (Mean +/- Sigma)
    s0 = np.array(data['sigma_0'])
    s1 = np.array(data['sigma_1'])
    m0 = np.array(data['action_mean_0'])
    m1 = np.array(data['action_mean_1'])
    
    ax1.fill_between(iters, m0 - s0, m0 + s0, color='blue', alpha=0.1)
    ax1.fill_between(iters, m1 - s1, m1 + s1, color='red', alpha=0.1)

    ax1.set_title('Policy Evolution (Action Mean $\pm$ Sigma)')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Normalized Action Value')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. Reward Convergence
    ax2.plot(iters, rewards, color='gray', alpha=0.3, label='Raw Reward')
    ax2.plot(iters, rewards_smooth, color='green', label='Smoothed Reward')
    ax2.set_title('Reward Convergence')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Reward')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 3. Uncertainty Reduction (Sigma)
    ax3.plot(iters, s0, label='Sigma (Ws)', color='blue')
    ax3.plot(iters, s1, label='Sigma (Wp)', color='red')
    ax3.set_title('Uncertainty Reduction (Sigma)')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Std Dev')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    # On n'appelle pas plt.show() ici pour ne pas bloquer, on le fera à la fin

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
    
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, infos = env_norm.step(action)
        last_info = infos[0]

    ai_metrics = {
        'delay': last_info.get('delay', 0.0) / 10.0,
        'area': last_info.get('area', 0.0),
        'p_stat': last_info.get('p_stat', 0.0),
        'p_dyn': last_info.get('p_dyn', 0.0),
    }
    ai_metrics['p_tot'] = (ai_metrics['p_stat']*1e-3) + ai_metrics['p_dyn']

    print(f"IA Result -> Delay: {ai_metrics['delay']:.4f}ns | Area: {ai_metrics['area']:.2f}um2")

    # --- 2. GENERATION MASSIVE (Pareto) ---
    logger.info(f"Génération du nuage de {N_RANDOM_SAMPLES} points...")
    logger.disable("pyngs")
    
    data = {'delay': [], 'area': [], 'p_tot': [], 'p_dyn': [], 'p_stat': []}
    
    for i in range(N_RANDOM_SAMPLES):
        if i % 200 == 0: print(f"Sampling {i}/{N_RANDOM_SAMPLES}...", end='\r')
        
        act = raw_env.action_space.sample()
        
        # --- CORRECTION ICI ---
        # On récupère le résultat complet et on prend juste le dernier élément (info)
        # Cela marche que step() renvoie 4 ou 5 valeurs.
        step_result = raw_env.step(act)
        info = step_result[-1] 
        # ----------------------
        
        if info.get('delay', 100.0) < 10.0:
            d = info['delay'] / 10.0
            a = info['area']
            pt = (info['p_stat']*1e-9 + info['p_dyn']*1e-6) * 1e6
            
            data['delay'].append(d)
            data['area'].append(a)
            data['p_tot'].append(pt)
            data['p_stat'].append(info['p_stat']) 
            data['p_dyn'].append(info['p_dyn'])   

    logger.enable("pyngs")
    print("\nCalcul des fronts de Pareto et affichage...")

    # --- 3. AFFICHAGE DE LA CONVERGENCE ---
    plot_training_convergence()

    # --- 4. AFFICHAGE PARETO ---
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Optimisation IA vs Espace des Possibles ({N_RANDOM_SAMPLES} samples)", fontsize=16)

    def plot_pareto(ax, x_key, y_key, xlabel, ylabel, title):
        X = data[x_key]
        Y = data[y_key]
        
        ax.scatter(X, Y, c='gray', alpha=0.3, s=20, label='Designs Possibles', edgecolors='none')
        
        if len(X) > 0: # Sécurité si la liste est vide
            p_x, p_y = get_pareto_frontier(X, Y)
            ax.plot(p_x, p_y, 'b-', linewidth=2, label='Front de Pareto', alpha=0.8)
        
        ax_x = ai_metrics[x_key]
        ax_y = ai_metrics[y_key]
        ax.scatter(ax_x, ax_y, c='red', marker='*', s=300, label='IA (PPO)', zorder=10, edgecolors='black')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.4)
        ax.legend()

    plot_pareto(axs[0, 0], 'delay', 'area', 'Délai (ns)', 'Surface (µm²)', 'Trade-off : Vitesse vs Coût')
    plot_pareto(axs[0, 1], 'delay', 'p_tot', 'Délai (ns)', 'Puissance Totale (µW)', 'Trade-off : Vitesse vs Conso')
    plot_pareto(axs[1, 0], 'p_tot', 'area', 'Puissance (µW)', 'Surface (µm²)', 'Densité de Puissance')
    plot_pareto(axs[1, 1], 'p_dyn', 'p_stat', 'Puissance Dyn (µW)', 'Puissance Stat (nW)', 'Analyse des Fuites')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    if MODE == "TRAINING":
        # L'entraînement va générer le fichier .pkl de log
        model = train_new_model()
        # Puis lancer la vérification
        run_verification_and_plot(model)
    elif MODE == "VERIFICATION":
        # Charge le modèle ET le fichier de log pour les graphes
        run_verification_and_plot()