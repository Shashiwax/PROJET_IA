`markdown
# RL-Based Inverter Sizing (SKY130A) — Usage Guide

This repository trains and evaluates agents to size a SKY130A inverter (parameters `wn`, `wp`) using SPICE-based metrics (area, delay, static power, dynamic energy) and logic constraints.

## Prerequisites

- **Python + uv**: this project is run via `uv run ...`
- **ngspice** installed and accessible in your PATH (`ngspice -v`)
- A working SKY130 setup / netlists that run correctly with ngspice (includes, models, etc.)

Typical first-time setup (from repo root):
bash
uv sync
`

---

## Repository layout

* `src/inverter_env.py`: Gymnasium environment wrapping **one SPICE simulation per step**.
* `src/train_ppo_parallel.py`: PPO training (Stable-Baselines3 + parallel environments).
* `src/run_trained_ppo.py`: Deterministic evaluation of a trained PPO model (`.zip`).
* `src/pools.py` + `src/train_cem_pool.py`: CEM + Pool-based exploration (non-SB3 baseline / comparison).
* `netlists/*.cir`: SPICE netlists with `.meas` metrics.
* `runs/`: created automatically; all outputs are stored here, per run.

---

## What you can change (user-facing knobs)

### A) Training & parallelism (PPO)

You typically change these via environment variables:

* `NETLIST` (path): which netlist to use
* `N_ENVS` (int): number of parallel env processes
* `MAX_STEPS` (int): max steps per episode (≈ max SPICE sims per episode per env)
* `TOTAL_TIMESTEPS` (int): total SB3 steps (global across all envs)
* `N_STEPS` (int): rollout horizon collected before each update
* `N_EPOCHS` (int): how many passes PPO does over each rollout
* `BATCH_SIZE` (int): minibatch size used during PPO updates
* `EVAL_FREQ` (int): every how many timesteps we run evaluation
* `N_EVAL_EPISODES` (int): number of eval episodes per evaluation point
* `SEED` (int): randomness control

**Important constraint (SB3):**

* `BATCH_SIZE` should ideally be a **divisor** of `N_ENVS * N_STEPS` (rollout buffer size), otherwise SB3 warns about a “truncated minibatch”.

### B) Reward configuration (PPO + eval)

Reward parameters are created in `train_ppo_parallel.py` as `cfg = RewardConfig(...)`.

If your script exposes them through env vars, typical overrides are:

* `W_AREA`, `W_DELAY`, `W_PSTAT`, `W_EDYN`: weights
* `W_SIZE`: soft size regularizer
* `FAIL_R`: reward assigned when a sim fails / violates hard constraints
* `RATIO_CLIP`: safety clipping for normalization ratios
* `VDD`, `YHI`, `YLO`, `W_LOGIC_HI`, `W_LOGIC_LO`, `HARD_HI`, `HARD_LO`: logic constraints & penalties

---

## How runs are organized (file management)

### PPO run folder structure

Training creates a clean run folder:


runs/ppo/<cell>/<corner>/runXYZ/
  config.json                  # frozen configuration for that run
  ppo_inverter_parallel.zip    # final model snapshot saved at end of training
  ppo_eval_trace.csv           # periodic deterministic evaluation trace (custom callback)
  tb/                          # tensorboard logs (events.out.tfevents...)
  sb3/
    best/
      best_model.zip           # best checkpoint according to EvalCallback
    eval/                      # eval logs produced by EvalCallback
  eval/
    eval_deterministic_trace.csv
    eval_deterministic_summary.json


### What are these files?

* `*.zip` (SB3 model): contains policy/value network weights + hyperparams.
  This is what you reload to **continue training** or to **run deterministic inference**.
* `tb/events.out.tfevents...`: TensorBoard event logs (scalars, etc.).
* `config.json`: stores run metadata (netlist, key hyperparams, reward config) to reproduce later.
* `ppo_eval_trace.csv`: your own periodic “best design so far” logging (deterministic eval).

---

## PPO workflows (common cases)

### 1) Start a brand-new training run (auto run id)

This creates a new `runXYZ` folder automatically.

bash
cd src
NETLIST=../netlists/inv.cir \
N_ENVS=4 MAX_STEPS=6 TOTAL_TIMESTEPS=30000 \
N_STEPS=92 N_EPOCHS=10 BATCH_SIZE=92 \
EVAL_FREQ=2000 N_EVAL_EPISODES=4 \
uv run python train_ppo_parallel.py


### 2) Start a brand-new training run with an explicit run id

bash
cd src
uv run python train_ppo_parallel.py --run run005


### 3) Resume a stopped training (same run folder)

This is the **“continue where I left off”** workflow.

bash
cd src
uv run python train_ppo_parallel.py --run run004


Stop it (Ctrl+C), then later:

bash
cd src
uv run python train_ppo_parallel.py --run run004 --resume


What happens:

* The script loads a checkpoint `.zip` (usually `sb3/best/best_model.zip` if present, otherwise `ppo_inverter_parallel.zip`) and continues optimizing from those weights.

### 4) Resume from a specific checkpoint file (`--resume-path`)

Useful if you want to resume from a particular `.zip`:

bash
cd src
uv run python train_ppo_parallel.py --run run004 --resume \
  --resume-path ../runs/ppo/inv/tt/run004/sb3/best/best_model.zip


### 5) Warm-start: resume but change training hyperparameters

You can do:

* Train → stop → change params (e.g., fewer envs, different `N_STEPS`, etc.) → resume.

Example:

bash
# initial
cd src
N_ENVS=6 N_STEPS=92 uv run python train_ppo_parallel.py --run run004

# stop, then resume with different parallelism
N_ENVS=4 N_STEPS=92 uv run python train_ppo_parallel.py --run run004 --resume


This keeps the **same network weights** and continues training.

**Do NOT resume in the same run if you changed:**

* the environment observation size,
* the action space,
* the meaning/range of the action,

because the loaded model architecture must match.

Practical rule:

* If you changed `inverter_env.py` in a way that changes `observation_space` or `action_space`, start a new run.

---

## Deterministic evaluation (run a trained PPO model)

Use `run_trained_ppo.py` to **apply** a trained policy deterministically and record metrics.

### Basic usage (latest run)

bash
cd src
uv run python run_trained_ppo.py


### Evaluate a specific run

bash
cd src
uv run python run_trained_ppo.py --run run004


Note: the correct CLI is `--run run004` (not `--run004`).

### Useful options

Print step-by-step outputs:

bash
cd src
uv run python run_trained_ppo.py --run run004 --print-steps


Choose episodes and max steps:

bash
cd src
uv run python run_trained_ppo.py --run run004 --episodes 5 --max-steps 6


Outputs are written into:


runs/ppo/<cell>/<corner>/runXXX/eval/
  eval_deterministic_trace.csv
  eval_deterministic_summary.json


---

## TensorBoard (optional but useful)

If installed, from project root:

bash
tensorboard --logdir runs/ppo


Then open the shown local URL in your browser.

If TensorBoard is not installed:

bash
uv add tensorboard


---

## CEM + Pool (baseline / alternative)

### What is “deterministic evaluation” for CEM?

CEM does **not** learn a neural network policy that needs a separate `.zip` runner.
It directly samples candidate `(wn, wp)` values, evaluates them with SPICE, and keeps the best.
So:

* A separate “run_trained_cem.py” is **optional**.
* If you want to re-check the final best point, you can simply run one more `pool.run()` on `(best_wn, best_wp)` (the script already prints “BEST POINT METRICS” at the end). 

### Current CEM outputs (as the script is written now)

By default, `train_cem_pool.py` saves:

* `cem_history.csv` in the current working directory. 

### Train CEM (example)

bash
cd src
NETLIST=../netlists/inv.cir uv run python train_cem_pool.py


### CEM knobs (currently edited inside the script)

In `train_cem_pool.py`, the main CEM hyperparameters are defined directly in code:

* `n_iters`
* `batch_size`
* `elite_frac`
* `alpha`
* initial `mean` / `std`
* `W_BINS` and `(wn_min/max, wp_min/max)`
* `n_workers` and `timeout_s` when creating `ParallelPool`

### (Planned / recommended) run folder structure for CEM

To keep outputs clean (same philosophy as PPO), the goal is:


runs/cem/<cell>/<corner>/runXYZ/
  config.json
  cem_history.csv
  best_point.csv (optional)
  logs.txt (optional)


Once implemented, CEM runs will be stored under `runs/cem/...` instead of mixing files in the repo root.

---



If you want, paste your **current** `train_cem_pool.py` after you implement the run-folder feature (or tell me you haven’t yet), and I’ll adapt the README section so it matches exactly what the code does (CLI args, exact filenames, exact folder tree).
::contentReference[oaicite:3]{index=3}

