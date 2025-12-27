# train_ppo_parallel.py
from __future__ import annotations

import os
import csv
from pathlib import Path
from typing import Callable, Dict, Any, List, Tuple

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from inverter_env import InverterEnv, RewardConfig


def best_divisor_leq(n: int, max_d: int) -> int:
    """Pick the largest divisor of n that is <= max_d. Fallback to n."""
    for d in range(min(max_d, n), 0, -1):
        if n % d == 0:
            return d
    return n


def make_env_fn(
    rank: int,
    netlist: str,
    max_steps: int,
    cfg: RewardConfig,
    debug: bool,
) -> Callable[[], InverterEnv]:
    def _init() -> InverterEnv:
        env = InverterEnv(
            netlist_path=netlist,
            max_steps=max_steps,
            reward_cfg=cfg,
            seed=1000 + rank,
            debug=debug,
            simulate_on_reset=False,  # keep PPO fast
            # Fast training mode
            fast_mode=True,
            fast_tran_step=os.getenv("FAST_TRAN_STEP", "5p"),
            fast_tsim=os.getenv("FAST_TSIM", "2n"),
            timeout_s=float(os.getenv("SPICE_TIMEOUT_S", "20")),
            # Bounds + discretization (reduce crashes)
            wn_min=float(os.getenv("WN_MIN", "0.20")),
            wn_max=float(os.getenv("WN_MAX", "1.26")),
            wp_min=float(os.getenv("WP_MIN", "0.20")),
            wp_max=float(os.getenv("WP_MAX", "1.65")),
            snap_step=float(os.getenv("SNAP_STEP", "0.01")),
        )
        return Monitor(env)
    return _init


def eval_collect_best(
    model: PPO,
    env: InverterEnv,
    n_episodes: int,
    out_csv: Path,
) -> Dict[str, Any]:
    """
    Run deterministic evaluation and collect the best design (wn/wp + metrics + breakdown).
    """
    rows: List[Dict[str, Any]] = []
    best = None

    for _ in range(n_episodes):
        obs, info = env.reset(options={"random_reset": True})
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)

        m = info.get("metrics", {})
        b = info.get("breakdown", {})
        row = {
            "reward": float(r),
            "wn": float(m.get("wn", np.nan)),
            "wp": float(m.get("wp", np.nan)),
            "cell_area_um2": float(m.get("cell_area_um2", np.nan)),
            "delay_max_ps": float(m.get("delay_max_ps", np.nan)),
            "pstat_wc_uW": float(m.get("pstat_wc_uW", np.nan)),
            "edyn_fJ": float(m.get("edyn_fJ", np.nan)),
            "ymean_a0": float(m.get("ymean_a0", np.nan)),
            "ymean_a1": float(m.get("ymean_a1", np.nan)),
            "term_area": float(b.get("term_area", np.nan)),
            "term_delay": float(b.get("term_delay", np.nan)),
            "term_pstat": float(b.get("term_pstat", np.nan)),
            "term_edyn": float(b.get("term_edyn", np.nan)),
            "term_size": float(b.get("term_size", np.nan)),
            "term_logic": float(b.get("term_logic", np.nan)),
            "cost": float(b.get("cost", np.nan)),
            "fail": float(b.get("fail", 0.0)),
        }
        rows.append(row)
        if best is None or row["reward"] > best["reward"]:
            best = row

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    return best or {}


def main() -> None:
    # -------------------------
    # Config (single source here)
    # -------------------------
    NETLIST = os.getenv("NETLIST", "../netlists/inv.cir")

    N_ENVS = int(os.getenv("N_ENVS", "6"))
    MAX_STEPS = int(os.getenv("MAX_STEPS", "6"))
    TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "10000"))

    # PPO rollout params
    N_STEPS = int(os.getenv("N_STEPS", "68"))  # smaller -> faster feedback (still same total sims)
    N_EPOCHS = int(os.getenv("N_EPOCHS", "10"))

    # Compute a safe batch_size (divisor of n_steps*n_envs) to avoid SB3 warning
    rollout_size = N_STEPS * N_ENVS
    #BATCH_SIZE = int(os.getenv("BATCH_SIZE", str(best_divisor_leq(rollout_size, 64))))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "68"))

    # Eval settings (be careful: each eval episode = 1 SPICE sim here)
    EVAL_FREQ = int(os.getenv("EVAL_FREQ", "500"))
    N_EVAL_EPISODES = int(os.getenv("N_EVAL_EPISODES", "4"))

    DEBUG = bool(int(os.getenv("DEBUG", "0")))

    # Reward config (PPO-friendly: bounded fail_reward, soft logic penalties)
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
    # Vectorized training env
    # -------------------------
    env_fns = [make_env_fn(i, NETLIST, MAX_STEPS, cfg, DEBUG) for i in range(N_ENVS)]

    start_method = os.getenv("START_METHOD", "spawn")  # "spawn" safer; "fork" faster on linux sometimes
    vec_env = SubprocVecEnv(env_fns, start_method=start_method)
    vec_env = VecMonitor(vec_env)

    # -------------------------
    # Eval env (single process)
    # -------------------------
    eval_env = DummyVecEnv([lambda: Monitor(InverterEnv(
        netlist_path=NETLIST,
        max_steps=MAX_STEPS,
        reward_cfg=cfg,
        seed=999,
        debug=False,
        simulate_on_reset=False,
        fast_mode=True,
        fast_tran_step=os.getenv("FAST_TRAN_STEP", "5p"),
        fast_tsim=os.getenv("FAST_TSIM", "2n"),
        timeout_s=float(os.getenv("SPICE_TIMEOUT_S", "20")),
        wn_min=float(os.getenv("WN_MIN", "0.20")),
        wn_max=float(os.getenv("WN_MAX", "1.26")),
        wp_min=float(os.getenv("WP_MIN", "0.20")),
        wp_max=float(os.getenv("WP_MAX", "1.65")),
        snap_step=float(os.getenv("SNAP_STEP", "0.01")),
    ))])

    # Early stop if eval does not improve
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=int(os.getenv("NO_IMPROVE_EVALS", "5")),
        min_evals=int(os.getenv("MIN_EVALS", "3")),
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env=eval_env,
        callback_after_eval=stop_cb,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        best_model_save_path="./",
        log_path="./",
        verbose=1,
    )

    # -------------------------
    # PPO model
    # -------------------------
    tb_log = Path(os.getenv("TB_LOG", "./tb_ppo_parallel"))
    tb_log.mkdir(parents=True, exist_ok=True)
    """
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=str(tb_log),
        learning_rate=float(os.getenv("LR", "3e-4")),
        gamma=float(os.getenv("GAMMA", "1.0")),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        clip_range=float(os.getenv("CLIP", "0.2")),
        ent_coef=float(os.getenv("ENT", "0.0")),
    )
    """
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=str(tb_log),
        learning_rate=3e-4,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=10,
        gamma=0.95,           # Standard
        # AJOUT : Curiosité pour éviter de rester bloqué au maximum (wn=1.26, wp=1.65)
        ent_coef=0.01,        
        clip_range=0.2,
    )


    print(f"[CFG] netlist={NETLIST}")
    print(f"[CFG] N_ENVS={N_ENVS} MAX_STEPS={MAX_STEPS} TOTAL_TIMESTEPS={TOTAL_TIMESTEPS}")
    print(f"[CFG] n_steps={N_STEPS} rollout_size={rollout_size} batch_size={BATCH_SIZE} n_epochs={N_EPOCHS}")
    print(f"[CFG] fast_mode=True FAST_TSIM={os.getenv('FAST_TSIM','2n')} FAST_TRAN_STEP={os.getenv('FAST_TRAN_STEP','5p')}")
    print(f"[CFG] fail_reward={cfg.fail_reward} eval_freq={EVAL_FREQ} n_eval_episodes={N_EVAL_EPISODES}")

    # -------------------------
    # Train
    # -------------------------
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_cb,
        progress_bar=True,  # avoids tqdm/rich dependency
    )

    # Save final model
    model.save("ppo_inverter_parallel.zip")
    print("Saved: ppo_inverter_parallel.zip")

    # -------------------------
    # Final evaluation: collect chosen design + metrics
    # -------------------------
    # We want the actual design the policy chooses, not only mean reward.
    # Use a raw (non-vec) env instance for detailed metrics.
    raw_eval_env = InverterEnv(
        netlist_path=NETLIST,
        max_steps=MAX_STEPS,
        reward_cfg=cfg,
        seed=2025,
        debug=False,
        simulate_on_reset=False,
        fast_mode=True,
        fast_tran_step=os.getenv("FAST_TRAN_STEP", "5p"),
        fast_tsim=os.getenv("FAST_TSIM", "2n"),
        timeout_s=float(os.getenv("SPICE_TIMEOUT_S", "20")),
        wn_min=float(os.getenv("WN_MIN", "0.20")),
        wn_max=float(os.getenv("WN_MAX", "1.26")),
        wp_min=float(os.getenv("WP_MIN", "0.20")),
        wp_max=float(os.getenv("WP_MAX", "1.65")),
        snap_step=float(os.getenv("SNAP_STEP", "0.01")),
    )

    best = eval_collect_best(
        model=model,
        env=raw_eval_env,
        n_episodes=int(os.getenv("FINAL_EVAL_EPISODES", "25")),
        out_csv=Path(os.getenv("FINAL_EVAL_CSV", "ppo_eval_trace.csv")),
    )
    raw_eval_env.close()

    print("\n=== PPO BEST DESIGN (deterministic eval) ===")
    for k, v in best.items():
        print(f"{k:>16s}: {v}")

    # Cleanup
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
