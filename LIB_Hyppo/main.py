import os
import sys
import numpy as np
from stable_baselines3 import PPO
from loguru import logger
import RL_Nand  # Ton fichier d'environnement

# --- CONFIGURATION ---
DELAY_NETLIST = "LIB_Hyppo/nand_delay.cir"
POWER_NETLIST = "LIB_Hyppo/nand_power.cir"
MODEL_NAME = "ppo_nand_multi_obj"  # Nom du fichier de sauvegarde (sans .zip)
TRAINING_STEPS = 5000             # Nombre de steps d'entrainement
LEARNING_RATE = 0.0003
TRAINING = True

def get_env():
    """Crée et retourne l'environnement configuré."""
    # Vérification basique de l'existence des fichiers
    if not os.path.exists(DELAY_NETLIST) or not os.path.exists(POWER_NETLIST):
        logger.error("Fichiers netlist introuvables. Vérifiez les chemins dans la configuration.")
        sys.exit(1)
        
    return RL_Nand.NANDOpt(delay_netlist=DELAY_NETLIST, power_netlist=POWER_NETLIST)

def train_new_model():
    """Entraîne un nouveau modèle et le sauvegarde."""
    env = get_env()
    
    logger.info(f"Démarrage de l'entraînement pour {TRAINING_STEPS} steps...")
    
    # MlpPolicy est adapté pour des vecteurs de chiffres (pas d'images)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=LEARNING_RATE)
    model.learn(total_timesteps=TRAINING_STEPS)
    
    logger.success("Entraînement terminé.")
    model.save(MODEL_NAME)
    logger.info(f"Modèle sauvegardé sous '{MODEL_NAME}.zip'")
    
    return model

def run_verification(model=None):
    """
    Charge le modèle (si nécessaire) et cherche le point optimal de fonctionnement.
    Affiche les résultats physiques interprétés.
    """
    env = get_env()
    
    # Si aucun modèle n'est passé en argument (cas du chargement direct), on le charge du disque
    if model is None:
        if not os.path.exists(f"{MODEL_NAME}.zip"):
            logger.error(f"Le fichier modèle '{MODEL_NAME}.zip' n'existe pas. Veuillez d'abord entraîner le modèle.")
            return
        logger.info(f"Chargement du modèle '{MODEL_NAME}' depuis le disque...")
        model = PPO.load(MODEL_NAME, env=env)

    logger.info("Lancement de la phase de test/vérification...")
    
    # Reset initial
    obs, _ = env.reset()
    
    final_action = None
    final_info = {}

    print("\n" + "="*80)
    print(f"{'Step':<5} | {'W_NMOS (µm)':<12} | {'W_PMOS (µm)':<12} | {'Reward':<10} | {'Délai Norm.':<12} | {'Power Norm.':<12}")
    print("="*80)

    # Boucle de convergence (Mode Déterministe)
    # On laisse l'agent stabiliser sa décision sur 10 itérations
    for i in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        final_action = action
        final_info = info
        
        d_val = info.get('delay', 0.0)
        p_val = info.get('power', 0.0)
        
        print(f"{i+1:<5} | {action[0]:.4f}       | {action[1]:.4f}       | {reward:.4f}     | {d_val:.4f}       | {p_val:.4f}")

    # --- INTERPRÉTATION DES RÉSULTATS ---
    print("="*80)
    print(" RÉSULTATS FINAUX OPTIMISÉS")
    print("="*80)
    
    # 1. Les paramètres géométriques à utiliser
    w_nmos = final_action[0]
    w_pmos = final_action[1]
    
    print(f"Paramètres du Design :")
    print(f"  -> W_NMOS : {w_nmos:.4f} µm")
    print(f"  -> W_PMOS : {w_pmos:.4f} µm")
    print(f"  -> Ratio P/N : {w_pmos/w_nmos:.2f}")

    # 2. Les performances physiques estimées
    # Note : Il faut inverser la normalisation faite dans RL_Nand pour retrouver les vraies unités
    # Dans RL_Nand : norm_delay = avg_delay_ns * 10.0  => avg_delay_ns = norm / 10.0
    # Dans RL_Nand : norm_power = p_total / 1e-6       => p_total = norm * 1e-6
    
    norm_delay = final_info.get('delay', 0.0)
    norm_power = final_info.get('power', 0.0)
    
    real_delay_ns = norm_delay / 10.0
    real_power_w = norm_power * 1e-6
    
    print(f"\nPerformances Estimées :")
    print(f"  -> Délai Moyen : {real_delay_ns:.5f} ns")
    print(f"  -> Puissance   : {real_power_w*1e6:.3f} µW")
    print("="*80)

if __name__ == "__main__":
    print("--- OPTIMISATION DE CELLULE NAND (Delay + Power) ---")
    
    if TRAINING:
        # Entraînement puis Test immédiat avec le modèle en mémoire
        trained_model = train_new_model()
        run_verification(model=trained_model)
        
    else:
        # Juste le test en chargeant depuis le disque
        run_verification(model=None)