from inverter_env import InverterEnv
from stable_baselines3.common.env_checker import check_env

env = InverterEnv(
    netlist_path="/home/sharo/PROJET_IA/LIB_Sharon/netlists/inv.cir",
    max_steps=1,
    random_reset=False,
)

check_env(env, warn=True)
print("Env OK")
env.close()
