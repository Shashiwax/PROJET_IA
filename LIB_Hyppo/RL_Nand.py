import gymnasium as gym
from stable_baselines3 import PPO
import Nand_tb as nand_tb
import numpy as np
from loguru import logger

class NANDOpt(gym.Env):
    def __init__(self, delay_netlist, power_netlist):
        super().__init__()
        
        # Initialisation du TB avec les deux fichiers
        self.testbench = nand_tb.NANDTestbench(delay_netlist, power_netlist)
        
        self.current_action = np.array([1.0, 1.0], dtype=float)
        
        # Action : Wn, Wp
        self.action_space = gym.spaces.Box(low=0.42, high=2.0, shape=(2,), dtype=float)
        
        # Observation : [Normalized_Delay, Normalized_Power, Normalized_Area]
        # L'agent a besoin de voir les 3 métriques pour comprendre le compromis
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(3,), dtype=float)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._run_sequence_and_get_results()
        return obs, {}
    
    def step(self, action):
        self.current_action = action
        
        # 1. Exécution Séquentielle
        obs = self._run_sequence_and_get_results()
        
        # obs = [norm_delay, norm_power, norm_area]
        norm_delay = obs[0]
        norm_power = obs[1]
        norm_area = obs[2] # Si tu veux optimiser l'aire aussi

        # --- CALCUL DU REWARD (FONCTION DE COÛT) ---
        # Objectif : Minimiser Délai ET Puissance.
        # Formule classique : PDP (Power Delay Product) ou somme pondérée (EDP).
        
        # Poids (A ajuster selon ce qui est prioritaire pour toi)
        # Si le délai est prioritaire, augmente ALPHA.
        ALPHA = 0.5  # Poids du Délai
        BETA = 0.5   # Poids de la Puissance
        # GAMMA = 0.1 # Poids de l'Aire (optionnel)

        # Reward négatif car on veut minimiser
        reward = - (ALPHA * norm_delay + BETA * norm_power)
        
        # Pénalité violente si simulation échouée (valeur 10.0 détectée)
        if norm_delay >= 10.0 or norm_power >= 10.0:
            reward = -50.0 # Punition très forte pour éviter ces zones
            
        done = False
        truncated = False
        info = {
            "delay": norm_delay,
            "power": norm_power,
            "area": norm_area
        }
        return obs, reward, done, truncated, info
    
    def _run_sequence_and_get_results(self):
        """
        Exécute Delay -> Power -> Retourne vecteur [D, P, A] normalisé
        """
        W_MIN_PDK = 0.42
        raw_nmos = np.clip(self.current_action[0], W_MIN_PDK, 2.0)
        raw_pmos = np.clip(self.current_action[1], W_MIN_PDK, 2.0)
        
        w_str_values = [f"{raw_nmos}e+06u", f"{raw_pmos}e+06u"]
        
        FAIL_OBS = np.array([10.0, 10.0, 10.0], dtype=float)

        try:
            # --- ETAPE 1 : DELAY & AREA ---
            self.testbench.run_specific_simulation("DELAY", w_str_values)
            res_delay = self.testbench.get_measurements("DELAY")
            
            # Extraction Delay
            delays = res_delay.get('delays')
            if not delays: return FAIL_OBS
            avg_delay_ns = np.mean(list(delays.values()))
            
            # Extraction Area
            area_val = res_delay.get('area', 0.0)
            
            # --- ETAPE 2 : POWER ---
            self.testbench.run_specific_simulation("POWER", w_str_values)
            res_power = self.testbench.get_measurements("POWER")
            
            power_data = res_power.get('power')
            if not power_data: return FAIL_OBS
            
            p_total = (power_data.get('static', 0.0) + power_data.get('dynamic', 0.0)) / 2.0

            # --- NORMALISATION ---
            # C'est CRUCIAL pour que l'IA ne favorise pas juste le chiffre le plus gros.
            # Valeurs cibles arbitraires (à ajuster selon ta techno) :
            # Delay cible ~ 0.05ns | Power cible ~ 1uW
            
            norm_delay = avg_delay_ns * 10.0  # ex: 0.1ns -> 1.0
            norm_power = p_total / 1e-6       # ex: 1uW -> 1.0
            norm_area = area_val * 1.0        # A ajuster selon l'ordre de grandeur
            
            return np.array([norm_delay, norm_power, norm_area], dtype=float)

        except Exception as e:
            logger.error(f"Erreur séquentielle : {e}")
            return FAIL_OBS