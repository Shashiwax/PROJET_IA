import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from inverter_env import InverterEnv


class NormalizeAction(gym.ActionWrapper):
    """
    Map agent actions from [-1, 1] to the env's real action bounds.
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

        # [-1, 1] -> [0, 1]
        a01 = (action + 1.0) * 0.5
        # [0, 1] -> [low, high]
        real = self.low + a01 * (self.high - self.low)
        return np.clip(real, self.low, self.high)

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.low, self.high)

        a01 = (action - self.low) / (self.high - self.low + 1e-12)
        norm = a01 * 2.0 - 1.0
        return np.clip(norm, -1.0, 1.0)


def make_env():
    env = InverterEnv(
        max_steps=10,         # episode length
        random_reset=True,    # helps exploration
        timeout_s=5.0,        # keep it safe
        # IMPORTANT: use safe mins to avoid invalid widths
        wn_range=(0.36, 0.65),
        wp_range=(0.36, 1.00),
    )
    env = NormalizeAction(env)
    env = Monitor(env)  # logs episode reward/length
    return env


if __name__ == "__main__":
    # SB3 expects a VecEnv
    venv = DummyVecEnv([make_env])

    # Normalize observations (recommended because metrics have different scales)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        policy="MlpPolicy",
        env=venv,
        verbose=1,
        n_steps=64,        # keep small: each step runs ngspice
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=256)

    model.save("ppo_inverter")
    venv.save("vecnorm_inverter.pkl")
    print("Saved: ppo_inverter.zip and vecnorm_inverter.pkl")
