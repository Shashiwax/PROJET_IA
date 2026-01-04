import gymnasium as gym
from stable_baselines3 import PPO
import Nand_tb as nand_tb
import numpy as np
from loguru import logger

class NANDOpt(gym.Env):
    def __init__(self, delay_netlist, power_netlist):
        super().__init__()
        
        self.testbench = nand_tb.NANDTestbench(delay_netlist, power_netlist)
        self.current_action = np.array([1.0, 1.0], dtype=float)
        
        # Action : [Wn, Wp] (micromètres)
        self.action_space = gym.spaces.Box(low=0.42, high=2.0, shape=(2,), dtype=float)
        
        # Observation : [Norm_Delay, Norm_P_Static, Norm_P_Dynamic, Norm_Area]
        # On passe à 4 dimensions pour séparer les types de puissance
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(4,), dtype=float)
        
        self.step_counter = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._run_sequence_and_get_results()
        return obs, {}
    
    def step(self, action):
        self.step_counter += 1
        self.current_action = action
        
        # 1. Simulation et Récupération des 4 métriques normalisées
        obs = self._run_sequence_and_get_results()
        
        norm_delay = obs[0]
        norm_p_stat = obs[1]
        norm_p_dyn = obs[2]
        norm_area = obs[3]

        # --- REWARD SYSTEM (FONCTION DE COÛT) ---
        # Définition des poids (A ajuster selon tes priorités !)
        
        W_DELAY = 0.4       # Priorité au délai
        W_P_DYN = 0.3       # La conso active est souvent dominante
        W_P_STAT = 0.1      # Les fuites (souvent faibles, mais importantes en veille)
        W_AREA = 0.2        # Surface (coût du silicium)
        
        # Note : La somme n'a pas besoin de faire 1.0, c'est relatif.
        
        # Le reward est négatif car on minimise
        reward = - (W_DELAY * norm_delay + 
                    W_P_STAT * norm_p_stat + 
                    W_P_DYN * norm_p_dyn + 
                    W_AREA * norm_area)
        
        # Pénalité de crash (si une valeur vaut 10.0 ou 20.0, c'est un échec)
        if np.any(obs >= 10.0):
            reward = -50.0 
            
        done = False
        truncated = False
        
        # Info pour le debug/affichage
        info = {
            "delay": norm_delay,
            "p_stat": norm_p_stat,
            "p_dyn": norm_p_dyn,
            "area": norm_area
        }
        
        return obs, reward, done, truncated, info
    
    def _run_sequence_and_get_results(self):
        """
        Exécute la séquence et retourne [Delay, P_Stat, P_Dyn, Area] (Normalisés)
        """
        W_MIN_PDK = 0.42
        raw_nmos = np.clip(self.current_action[0], W_MIN_PDK, 2.0)
        raw_pmos = np.clip(self.current_action[1], W_MIN_PDK, 2.0)
        
        w_str_values = [f"{raw_nmos}e+06u", f"{raw_pmos}e+06u"]
        
        # Vecteur d'échec (4 dimensions maintenant)
        FAIL_OBS = np.array([20.0, 20.0, 20.0, 20.0], dtype=float)

        try:
            # --- 1. SIMULATION DELAY & AREA ---
            self.testbench.run_specific_simulation("DELAY", w_str_values)
            if not self.testbench.check_simulation_health(): return FAIL_OBS

            res_delay = self.testbench.get_measurements("DELAY")
            
            # Delay
            delays = res_delay.get('delays')
            if not delays: return FAIL_OBS
            avg_delay_ns = np.mean(list(delays.values()))
            
            # Area (en m², souvent très petit ex: 1e-12)
            area_val = res_delay.get('area', 0.0)

            # --- 2. SIMULATION POWER ---
            self.testbench.run_specific_simulation("POWER", w_str_values)
            if not self.testbench.check_simulation_health(): return FAIL_OBS

            res_power = self.testbench.get_measurements("POWER")
            power_data = res_power.get('power', {})
            
            p_stat_val = power_data.get('static', 0.0)
            p_dyn_val = power_data.get('dynamic', 0.0)

            # --- 3. NORMALISATION (Crucial !) ---
            # Il faut ramener toutes les valeurs autour de 1.0 pour que le réseau de neurones apprenne bien.
            
            # Cibles arbitraires pour la normalisation :
            # Delay cible : 0.1 ns
            # P_Stat cible : 10 nW (1e-8) -> Attention c'est petit !
            # P_Dyn cible : 1 µW (1e-6)
            # Area cible : 1 µm² (1e-12)
            
            norm_delay = avg_delay_ns * 10.0      # ex: 0.1ns -> 1.0
            
            norm_p_stat = p_stat_val / 1e-9       # ex: 10nW -> 10.0 (On divise par 1 nanoWatt)
            if norm_p_stat > 100: norm_p_stat = 100.0 # Clip pour éviter des explosions
            
            norm_p_dyn = p_dyn_val / 1e-6         # ex: 1µW -> 1.0
            
            norm_area = area_val        
            
            return np.array([norm_delay, norm_p_stat, norm_p_dyn, norm_area], dtype=float)

        except Exception as e:
            logger.error(f"Erreur run sequence: {e}")
            return FAIL_OBS