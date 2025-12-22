import numpy as np
from stable_baselines3 import PPO
from pathlib import Path
from loguru import logger

import RL_Nand

CellPath = "LIB_Hyppo/nand.cir"

env = RL_Nand.NANDOpt(netlist_path=CellPath)

model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
logger.info("Début de l'entraînement de l'agent PPO pour l'optimisation de la NAND.")
model.learn(total_timesteps=5000)

model.save("ppo_nand_optimization")
obs, _ = env.reset()

action, _ = model.predict(obs)
print(f"Meilleures largeurs trouvées par l'agent PPO : W_NMOS = {action[0]:.3f}, W_PMOS = {action[1]:.3f}")
print("Entraînement terminé et modèle sauvegardé sous 'ppo_nand_optimization'.")

