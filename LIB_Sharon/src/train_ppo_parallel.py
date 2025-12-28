import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed

from inverter_env import InverterEnv, RewardConfig
import argparse
import json
from datetime import datetime

def _project_root() -> Path:
    # Assumes this file lives in <root>/src/
    return Path(__file__).resolve().parents[1]


def _infer_cell_and_corner(netlist_path: Path) -> tuple[str, str]:
    # netlist stem like "inv" or "inv_tt"
    stem = netlist_path.stem
    parts = stem.split("_")
    cell = parts[0] if parts and parts[0] else stem
    corner = parts[1] if len(parts) >= 2 and parts[1] else "tt"
    return cell, corner


def _next_run_id(base_dir: Path) -> str:
    best = 0
    for p in base_dir.glob("run[0-9][0-9][0-9]"):
        if p.is_dir():
            try:
                n = int(p.name.replace("run", ""))
                best = max(best, n)
            except ValueError:
                pass
    return f"run{best + 1:03d}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run",
        type=str,
        default="",
        help="Run folder name to reuse (e.g. run001). If omitted, auto-creates runXXX.",
    )
    p.add_argument("--resume", action="store_true", help="Resume training from existing model in the run folder.")
    p.add_argument("--resume-path", type=str, default="", help="Optional explicit path to a .zip to resume from.")

    return p.parse_args()


def _resolve_run_dir(netlist_path: Path, run_id: str) -> tuple[Path, str, str, str]:
    cell, corner = _infer_cell_and_corner(netlist_path)
    base = _project_root() / "runs" / "ppo" / cell / corner
    base.mkdir(parents=True, exist_ok=True)

    rid = run_id.strip()
    if not rid:
        rid = _next_run_id(base)

    run_dir = base / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, rid, cell, corner


class EvalTraceCallback(BaseCallback):
    """
    Logs (on every evaluation) the deterministic best design found so far,
    and stores a csv trace (time, timesteps, reward, wn, wp, metrics, etc.)
    """
    def __init__(self, eval_env: DummyVecEnv, out_csv: Path, n_eval_episodes: int = 6, eval_freq: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.out_csv = Path(out_csv)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq

        self.best = None
        self.rows = []

    def _on_step(self) -> bool:
        if (self.num_timesteps % self.eval_freq) != 0:
            return True

        # Deterministic evaluation: do n_eval_episodes rollouts
        rewards = []
        best_ep = None

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, done, info = self.eval_env.step(action)
                ep_reward += float(r)

            rewards.append(ep_reward)

            # Grab last info dict
            inf = info[0] if isinstance(info, (list, tuple)) else info
            if best_ep is None or ep_reward > best_ep["reward"]:
                best_ep = {"reward": ep_reward, **inf}

        mean_r = float(np.mean(rewards))
        if self.verbose:
            print(f"[EvalTrace] t={self.num_timesteps} mean_reward={mean_r:.4f}")

        # Track global best by reward
        if self.best is None or best_ep["reward"] > self.best["reward"]:
            self.best = best_ep

        row = {
            "wall_time_s": time.time(),
            "timesteps": self.num_timesteps,
            "eval_mean_reward": mean_r,
        }
        # Add best episode fields if present
        row.update({f"best_{k}": v for k, v in best_ep.items()})
        # Add global best fields
        if self.best is not None:
            row.update({f"global_{k}": v for k, v in self.best.items()})

        self.rows.append(row)

        # Persist CSV
        df = pd.DataFrame(self.rows)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_csv, index=False)

        return True


def eval_collect_best(model: PPO, env: InverterEnv, max_steps: int, out_csv: Path) -> dict:
    """
    Deterministic eval on a *single* env, collecting best design across the episode.
    Returns best info dict and saves trace CSV (optional).
    """
    obs, _ = env.reset()
    best = None
    rows = []

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        r = float(reward)

        if best is None or r > best["reward"]:
            best = {"reward": r, **info}

        rows.append({"reward": r, **info})

        if terminated or truncated:
            break

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    return best if best is not None else {}


def make_env_fn(rank: int, netlist: Path, max_steps: int, cfg: RewardConfig, seed: int):
    def _init():
        env = InverterEnv(
            netlist_path=netlist,
            max_steps=max_steps,
            reward_cfg=cfg,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    NETLIST = Path(os.getenv("NETLIST", "../netlists/inv.cir"))

    # Parallelism: start with 4 (good tradeoff on WSL), then try 6-8 if stable.
    N_ENVS = int(os.getenv("N_ENVS", "4"))
    MAX_STEPS = int(os.getenv("MAX_STEPS", "6"))
    TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "30000"))

    # PPO rollout params
    N_STEPS = int(os.getenv("N_STEPS", "92"))
    N_EPOCHS = int(os.getenv("N_EPOCHS", "10"))

    # rollout_size = N_STEPS * N_ENVS
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", str(N_STEPS)))
    EVAL_FREQ = int(os.getenv("EVAL_FREQ", "2000"))
    N_EVAL_EPISODES = int(os.getenv("N_EVAL_EPISODES", "4"))

    SEED = int(os.getenv("SEED", "0"))
    set_random_seed(SEED)

    cfg = RewardConfig(
        ref_cell_area_um2=3.7536,
        ref_delay_max_ps=18.96081,
        ref_pstat_wc_uW=1.295751,
        ref_edyn_fJ=1.96807,

        w_area=float(os.getenv("W_AREA", "0.35")),
        w_delay=float(os.getenv("W_DELAY", "0.25")),
        w_pstat=float(os.getenv("W_PSTAT", "0.25")),
        w_edyn=float(os.getenv("W_EDYN", "0.15")),

        w_size=float(os.getenv("W_SIZE", "0.02")),
        wn0=float(os.getenv("WN0", "0.65")),
        wp0=float(os.getenv("WP0", "1.0")),

        vdd=float(os.getenv("VDD", "1.8")),
        yhi_ratio=float(os.getenv("YHI", "0.95")),
        ylo_ratio=float(os.getenv("YLO", "0.05")),

        w_logic_hi=float(os.getenv("W_LOGIC_HI", "0.50")),
        w_logic_lo=float(os.getenv("W_LOGIC_LO", "0.50")),

        hard_hi_ratio=float(os.getenv("HARD_HI", "0.80")),
        hard_lo_ratio=float(os.getenv("HARD_LO", "0.20")),

        fail_reward=float(os.getenv("FAIL_R", "-50")),
        ratio_clip=float(os.getenv("RATIO_CLIP", "10.0")),
    )

    # -------------------------
    # Clean run folder (only file management)
    # -------------------------
    args = _parse_args()
    run_dir, RUN_ID, CELL_NAME, CORNER = _resolve_run_dir(NETLIST, args.run)
    tb_log = run_dir / "tb"
    sb3_dir = run_dir / "sb3"
    tb_log.mkdir(parents=True, exist_ok=True)
    sb3_dir.mkdir(parents=True, exist_ok=True)
    out_csv = run_dir / "ppo_eval_trace.csv"
    model_path = run_dir / "ppo_inverter_parallel.zip"
    config_json = run_dir / "config.json"

    # Write config once per run (so you can understand later)
    if not config_json.exists():
        config_dump = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "algo": "ppo",
            "run_id": RUN_ID,
            "cell": CELL_NAME,
            "corner": CORNER,
            "netlist": str(NETLIST),
            "train": {
                "N_ENVS": N_ENVS,
                "MAX_STEPS": MAX_STEPS,
                "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS,
                "N_STEPS": N_STEPS,
                "BATCH_SIZE": BATCH_SIZE,
                "EVAL_FREQ": EVAL_FREQ,
                "N_EVAL_EPISODES": N_EVAL_EPISODES,
                "SEED": SEED,
            },
            "reward_cfg": cfg.__dict__,
        }
        config_json.write_text(json.dumps(config_dump, indent=2) + "\n")

    # Vec env
    env_fns = [make_env_fn(i, NETLIST, MAX_STEPS, cfg, SEED) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")

    # Eval env (single)
    eval_env = DummyVecEnv([lambda: Monitor(InverterEnv(netlist_path=NETLIST, max_steps=MAX_STEPS, reward_cfg=cfg))])

    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(sb3_dir / "best"),
        log_path=str(sb3_dir / "eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    trace_cb = EvalTraceCallback(
        eval_env=eval_env,
        out_csv=out_csv,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        verbose=0,
    )

    device = "cpu"  # keep your choice as-is
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=str(tb_log),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=0.99,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.00,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=SEED,
        device=device
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_cb, trace_cb],
        progress_bar=True,
    )

    model.save(str(model_path))
    print(f"Saved: {model_path}")

    # Deterministic final eval on a single env + save trace in the same run folder by default
    env_single = InverterEnv(netlist_path=NETLIST, max_steps=MAX_STEPS, reward_cfg=cfg)
    best = eval_collect_best(
        model=model,
        env=env_single,
        max_steps=MAX_STEPS,
        out_csv=Path(os.getenv("FINAL_EVAL_CSV", str(out_csv))),
    )

    print("\n=== PPO BEST DESIGN (deterministic eval) ===")
    for k, v in best.items():
        print(f"{k:>16}: {v}")


if __name__ == "__main__":
    main()
