"""DOCSTRING Modèle PPO d'optimisation de Standard cells"""
import gymnasium as gym
from stable_baselines3 import PPO
import Nand_tb as nand_tb
import numpy as np
from loguru import logger

# Instanciation de la classe de l'agent RL
# Objectif : construire une classe qui utilise la PPO de stable baselines pour optimiser le W d'une nand 
# et optimiser la surface, le delay et la puissance
# L'agent reçoit les résultats d'une simulation pyngs et ajuste une netlist ngspice en conséquence
import gymnasium as gym
from stable_baselines3 import PPO
import Nand_tb as nand_tb
import numpy as np

class NANDOpt(gym.Env):
    def __init__(self, netlist_path):
        super().__init__()
        self.netlist_path = netlist_path
        self.testbench = nand_tb.NANDTestbench(netlist_path)
        
        # CORRECTION 1 : Initialiser avec des FLOATS, pas des strings.
        # Cela correspond à l'espace d'action (0.15 à 2.0)
        self.current_action = np.array([1.0, 2.0], dtype=float)
        
        # Espace d'action : on manipule des nombres (ex: 0.5 signifie 0.5 micromètres pour nous)
        self.action_space = gym.spaces.Box(low=0.15, high=2.0, shape=(2,), dtype=float)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=float)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Lors du reset, on utilise les valeurs par défaut définies dans __init__
        obs = self._get_1sim_results()
        return obs, {}
    
    def step(self, action):
        # L'agent envoie des floats. On les stocke tels quels.
        self.current_action = action
        
        obs = self._get_sim_results()
        
        # Calcul de reward (à adapter selon vos besoins)
        reward = -obs[0] -obs[1]
        done = False
        truncated = False
        info = {}
        return obs, reward, done, truncated, info
    
    def _get_sim_results(self):
        
        W_MIN_PDK = 0.55 

        raw_nmos = self.current_action[0]
        raw_pmos = self.current_action[1]

        safe_nmos = max(raw_nmos, W_MIN_PDK)
        safe_pmos = max(raw_pmos, W_MIN_PDK)
        
        w_values = [
            f"{safe_nmos}e+06u", 
            f"{safe_pmos}e+06u"
        ]
        logger.debug(f"Transistor widths for SPICE: {w_values}")

        # Lancement simulation
        self.testbench.run_simulation(w_values)
        measurements = self.testbench.get_all_measurements()
        
        # Gestion des résultats
        delays = measurements.get('delays', {})
        if not delays:
            # En cas d'échec simulation, on renvoie une "mauvaise" observation
            return np.array([1.0], dtype=float)

        avg_delay = np.mean(list(delays.values()))
        normalized_delay = avg_delay / 10.0 # Facteur de normalisation à ajuster

        area = measurements.get('area', {})
        avg_area = np.mean(0)
        
        return np.array([normalized_delay], dtype=float)