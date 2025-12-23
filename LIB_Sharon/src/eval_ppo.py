# eval_ppo.py
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from inverter_env import InverterEnv
import gymnasium as gym


class NormalizeAction(gym.ActionWrapper):
    """
    Same wrapper as in train_ppo.py:
    maps agent actions from [-1, 1] to env's real bounds.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)

        self.low = env.action_space.low.astype(np.float32)
        self.high = env.action_space.high.astype(np.float32)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        a01 = (action + 1.0) * 0.5
        real = self.low + a01 * (self.high - self.low)
        return np.clip(real, self.low, self.high)


def make_env():
    env = InverterEnv(
        max_steps=10,
        random_reset=True,
        timeout_s=5.0,
        wn_range=(0.36, 0.65),
        wp_range=(0.36, 1.00),
    )
    env = NormalizeAction(env)
    return env


if __name__ == "__main__":
    # Vec env + same normalization stats as training
    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load("vecnorm_inverter.pkl", venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load("ppo_inverter.zip", env=venv)

    obs = venv.reset()
    ep_reward = 0.0

    for t in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)

        ep_reward += float(reward[0])
        metrics = info[0].get("metrics", {})

        print(
            f"t={t+1:02d} r={float(reward[0]): .6f} "
            f"wn={metrics.get('wn_chk')} wp={metrics.get('wp_chk')} "
            f"area={metrics.get('cell_area_um2')} "
            f"delay_rise_ps={metrics.get('delay_rise_ps')} "
            f"pstat_wc_uW={metrics.get('pstat_wc_uW')}"
        )

        if bool(done[0]):
            break

    print(f"\nEpisode return: {ep_reward:.6f}")
