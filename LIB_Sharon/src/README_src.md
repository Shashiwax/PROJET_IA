```markdown
# RL-Based Inverter Sizing (SKY130A) — Usage Guide

This repository trains and evaluates agents to size a SKY130A inverter (parameters `wn`, `wp`) using SPICE-based metrics (area, delay, static power, dynamic energy) and logic constraints.

## src Repository layout 

- `src/inverter_env.py`: Gymnasium environment wrapping one SPICE simulation per step.
- `src/train_ppo_parallel.py`: PPO training (SB3 + parallel environments).
- `src/run_trained_ppo.py`: Deterministic evaluation of a trained PPO model (`.zip`).
- `src/pools.py` + `src/train_cem_pool.py`: CEM + Pool-based exploration (non-SB3 baseline / comparison).
- `netlists/*.cir`: SPICE netlists with `.meas` metrics.
- `runs/`: folder created where all outputs (models, logs, traces) stored per run.


## What you can change (user-facing knobs)

### A: Training & parallelism (PPO)
You typically change these via environment variables:

- `NETLIST` (path): which netlist to use  
- `N_ENVS` (int): number of parallel env processes
- `MAX_STEPS` (int): max steps per episode (≈ max SPICE sims per episode per env)
- `TOTAL_TIMESTEPS` (int): total SB3 steps (global across all envs)
- `N_STEPS` (int): rollout horizon collected before each update
- `N_EPOCHS` (int): how many passes PPO does over each rollout
- `BATCH_SIZE` (int): minibatch size used during PPO updates
- `EVAL_FREQ` (int): every how many timesteps we run evaluation
- `N_EVAL_EPISODES` (int): number of eval episodes per evaluation point
- `SEED` (int): randomness control

Important constraint (SB3):
- `BATCH_SIZE` should ideally be a divisor of `N_ENVS * N_STEPS` (rollout buffer size), otherwise SB3 warns about a “truncated minibatch”.

### B: Reward configuration (PPO + eval)
Reward parameters are created in `train_ppo_parallel.py` as `cfg = RewardConfig(...)`.
You can override most of them via environment variables (if your script is written that way), typically:

- `W_AREA`, `W_DELAY`, `W_PSTAT`, `W_EDYN`: weights
- `W_SIZE`: soft size regularizer
- `FAIL_R`: reward assigned when a sim fails / violates hard constraints
- `RATIO_CLIP`: safety clipping for normalization ratios
- `VDD`, `YHI`, `YLO`, `W_LOGIC_HI`, `W_LOGIC_LO`, `HARD_HI`, `HARD_LO`: logic constraints & penalties

---

## How runs are organized (file management)

Training creates a clean run folder:


runs/ppo/<cell>/<corner>/runXYZ/
config.json                  # frozen configuration for that run
ppo_inverter_parallel.zip    # final model snapshot saved at end of training
ppo_eval_trace.csv           # periodic deterministic evaluation trace
tb/                          # tensorboard logs (events.out.tfevents...)
sb3/
best/
best_model.zip           # best checkpoint according to EvalCallback
eval/                      # eval logs produced by EvalCallback


### What are these files?
- `*.zip` (SB3 model): contains policy/value network weights + hyperparams. This is what you reload to continue training or to run deterministic inference.
- `tb/events.out.tfevents...`: TensorBoard event logs (scalars, etc.).
- `config.json`: stores run metadata (netlist, key hyperparams, reward config) to reproduce later.
- `ppo_eval_trace.csv`: your own periodic “best design so far” logging (deterministic eval).

---

## PPO workflows (common cases)

### 1: Start a brand-new training run (auto run id)
This creates a new `runXYZ` folder automatically.
```bash
cd src
NETLIST=../netlists/inv.cir \
N_ENVS=4 MAX_STEPS=6 TOTAL_TIMESTEPS=30000 \
N_STEPS=92 N_EPOCHS=10 BATCH_SIZE=92 \
EVAL_FREQ=2000 N_EVAL_EPISODES=4 \
uv run python train_ppo_parallel.py
````

### 2: To start a brand-new training run with an explicit run id

```bash
uv run python train_ppo_parallel.py --run run005
```

### 3: To resume a stopped training (same run folder)

This is the **“continue where I left off”** workflow.

* Start training:

```bash
uv run python train_ppo_parallel.py --run run004
```

* Stop it (Ctrl+C, or terminal kill)

* Resume later:

```bash
uv run python train_ppo_parallel.py --run run004 --resume
```

What happens:

* The script loads a checkpoint `.zip` (usually `sb3/best/best_model.zip` if present, otherwise `ppo_inverter_parallel.zip`) and continues optimizing from those weights.

### 4: Resume from a specific checkpoint file (`--resume-path`)

Useful if you want to resume from a particular `.zip`:

```bash
uv run python train_ppo_parallel.py --run run004 --resume \
  --resume-path runs/ppo/inv/tt/run004/sb3/best/best_model.zip
```

### 5: Resume but change training hyperparameters (warm-start)

you can do:

* Train → stop → change params (e.g., fewer envs, different `N_STEPS`, etc.) → resume.

Example:

```bash
# initial
N_ENVS=6 N_STEPS=92 uv run python train_ppo_parallel.py --run run004

# stop, then resume with different parallelism
N_ENVS=4 N_STEPS=92 uv run python train_ppo_parallel.py --run run004 --resume
```

This keeps the same network weights and continues training.

**Do NOT do this in the same run** if you changed:

* the environment observation size,
* the action space,
* the meaning/range of the action,
  because the loaded model architecture must match.

Practical rule:

* If you changed `inverter_env.py` in a way that changes `observation_space` or `action_space`, start a new run.

---

## Deterministic evaluation (run a trained model)

Use `run_trained_ppo.py` to **apply** a trained policy deterministically and record metrics.

### Basic usage (latest run)

```bash
uv run python run_trained_ppo.py
```

### Evaluate a specific run

```bash
uv run python run_trained_ppo.py --run run004
```

Note: the correct CLI is `--run run004` (not `--run004`).

### Useful options

* Print step-by-step outputs:

```bash
uv run python run_trained_ppo.py --run run004 --print-steps
```

* Choose episodes and max steps:

```bash
uv run python run_trained_ppo.py --run run004 --episodes 5 --max-steps 6
```

Outputs are written into:

```
runs/ppo/<cell>/<corner>/runXXX/eval/
  eval_deterministic_trace.csv
  eval_deterministic_summary.json
```

---

## TensorBoard (optional but useful)

If installed, from project root:

```bash
tensorboard --logdir runs/ppo
```

Then open the shown local URL in your browser.

---

## CEM + Pool (baseline / alternative)

### Train (example)

```bash
cd src
NETLIST=../netlists/inv.cir uv run python train_cem_pool.py
```

