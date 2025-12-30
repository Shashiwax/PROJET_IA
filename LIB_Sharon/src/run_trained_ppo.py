# src/run_trained_ppo.py
import argparse
import json
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from inverter_env import InverterEnv, RewardConfig


def _project_root() -> Path:
    # Assumes this file lives in <root>/src/
    return Path(__file__).resolve().parents[1]


def _infer_cell_and_corner(netlist_path: Path) -> Tuple[str, str]:
    # netlist stem like "inv" or "inv_tt"
    stem = netlist_path.stem
    parts = stem.split("_")
    cell = parts[0] if parts and parts[0] else stem
    corner = parts[1] if len(parts) >= 2 and parts[1] else "tt"
    return cell, corner


def _latest_run_dir(base_dir: Path) -> Optional[Path]:
    # expects run001/run002/...
    best_n = -1
    best_p = None
    for p in base_dir.glob("run[0-9][0-9][0-9]"):
        if not p.is_dir():
            continue
        try:
            n = int(p.name.replace("run", ""))
        except ValueError:
            continue
        if n > best_n:
            best_n = n
            best_p = p
    return best_p


def _resolve_run_dir(netlist_path: Path, run_id: str) -> Path:
    cell, corner = _infer_cell_and_corner(netlist_path)
    base = _project_root() / "runs" / "ppo" / cell / corner
    if not base.exists():
        raise FileNotFoundError(f"Base runs folder not found: {base}")

    rid = run_id.strip()
    if rid:
        run_dir = base / rid
        if not run_dir.exists():
            raise FileNotFoundError(f"Run folder not found: {run_dir}")
        return run_dir

    # No --run provided -> pick latest
    last = _latest_run_dir(base)
    if last is None:
        raise FileNotFoundError(f"No runXXX folder found in: {base}")
    return last


def _pick_model_path(run_dir: Path, model_name: str) -> Path:
    # default file name used by your training script
    p = run_dir / model_name
    if p.exists():
        return p

    # fallback: any .zip in the run dir
    zips = sorted(run_dir.glob("*.zip"))
    if len(zips) == 1:
        return zips[0]
    if len(zips) > 1:
        raise FileNotFoundError(
            f"Multiple .zip found in {run_dir}. Use --model-name to choose."
        )
    raise FileNotFoundError(f"No model .zip found in {run_dir} (expected {p.name}).")


def _filter_reward_cfg_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    # Keep only fields that exist in RewardConfig dataclass (robust to extra keys)
    allowed = {f.name for f in fields(RewardConfig)}
    out = {}
    for k, v in d.items():
        if k in allowed:
            out[k] = v
    return out


def _load_cfg_from_config_json(run_dir: Path) -> Tuple[Optional[RewardConfig], Dict[str, Any]]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return None, {}

    raw = json.loads(cfg_path.read_text())
    cfg_raw = raw.get("reward_cfg", {})
    if isinstance(cfg_raw, dict) and cfg_raw:
        cfg = RewardConfig(**_filter_reward_cfg_dict(cfg_raw))
    else:
        cfg = None
    return cfg, raw


def _default_reward_cfg_from_env() -> RewardConfig:
    # Fallback if config.json doesn't exist
    return RewardConfig(
        ref_cell_area_um2=float(np.float32(3.7536)),
        ref_delay_max_ps=float(np.float32(18.96081)),
        ref_pstat_wc_uW=float(np.float32(1.295751)),
        ref_edyn_fJ=float(np.float32(1.96807)),
    )


def _flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, prefix=f"{prefix}{k}_" if prefix else f"{k}_"))
    else:
        key = prefix[:-1] if prefix.endswith("_") else prefix
        out[key] = d
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default="", help="Run folder to use (e.g. run004). If omitted, uses latest runXXX.")
    p.add_argument("--netlist", type=str, default="", help="Override netlist path (default: same as in config.json or ../netlists/inv.cir).")
    p.add_argument("--episodes", type=int, default=5, help="Number of deterministic episodes.")
    p.add_argument("--max-steps", type=int, default=0, help="Override max_steps (0 = use training/config value if available).")
    p.add_argument("--model-name", type=str, default="ppo_inverter_parallel.zip", help="Model zip name inside the run folder.")
    p.add_argument("--out-csv", type=str, default="", help="Output CSV path (default: <run_dir>/eval/eval_deterministic_trace.csv).")
    p.add_argument("--print-steps", action="store_true", help="Print every step line to stdout.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Determine netlist
    default_netlist = Path(args.netlist) if args.netlist else (Path(__file__).resolve().parents[1] / "netlists" / "inv.cir")

    # If a run folder exists, prefer config.json netlist unless user overrides
    run_dir = _resolve_run_dir(default_netlist, args.run)
    cfg_from_json, raw_config = _load_cfg_from_config_json(run_dir)

    netlist = default_netlist
    if not args.netlist:
        # Try take netlist from config.json if present
        nl = raw_config.get("netlist", "")
        if nl:
            netlist = Path(nl)

    # Reward config
    cfg = cfg_from_json if cfg_from_json is not None else _default_reward_cfg_from_env()

    # Max steps: override > config.json > fallback
    max_steps = args.max_steps if args.max_steps > 0 else 0
    if max_steps <= 0:
        train_cfg = raw_config.get("train", {}) if isinstance(raw_config.get("train", {}), dict) else {}
        max_steps = int(train_cfg.get("MAX_STEPS", 25)) if train_cfg else 25

    # Load model
    model_path = _pick_model_path(run_dir, args.model_name)
    model = PPO.load(str(model_path), device="cpu")

    # Output paths
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv) if args.out_csv else (eval_dir / "eval_deterministic_trace.csv")
    out_json = eval_dir / "eval_deterministic_summary.json"

    # Create env (deterministic evaluation: random_reset=False)
    env = InverterEnv(netlist_path=netlist, max_steps=max_steps, reward_cfg=cfg)

    rows = []
    best_step = None  # best by immediate step reward
    episode_returns = []

    for ep in range(args.episodes):
        obs, info0 = env.reset()
        ep_ret = 0.0

        for t in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            r = float(reward)
            ep_ret += r

            # flatten info for CSV
            flat = _flatten(info)
            row = {
                "episode": ep,
                "t": t + 1,
                "reward": r,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "wall_time_s": time.time(),
            }
            row.update(flat)
            rows.append(row)

            if args.print_steps:
                # try show wn/wp if present
                wn = flat.get("metrics_wn_chk", flat.get("metrics_wn", None))
                wp = flat.get("metrics_wp_chk", flat.get("metrics_wp", None))
                area = flat.get("metrics_cell_area_um2", None)
                dmax = flat.get("metrics_delay_max_ps", None)
                pstat = flat.get("metrics_pstat_wc_uW", None)
                print(
                    f"ep={ep:02d} t={t+1:02d} r={r:.6f} wn={wn} wp={wp} area={area} dmax={dmax} pstat={pstat}"
                )

            if best_step is None or r > best_step["reward"]:
                best_step = {"reward": r, "episode": ep, "t": t + 1, **info}

            if terminated or truncated:
                break

        episode_returns.append(ep_ret)

    env.close()

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    summary = {
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "netlist": str(netlist),
        "episodes": args.episodes,
        "max_steps": max_steps,
        "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else None,
        "episode_return_std": float(np.std(episode_returns)) if episode_returns else None,
        "best_step": best_step if best_step is not None else {},
    }
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Saved CSV:  {out_csv}")
    print(f"Saved JSON: {out_json}")

    if best_step is not None:
        # Pretty print key fields if they exist
        flat_best = _flatten(best_step)
        print("\n=== BEST STEP (deterministic) ===")
        for k in [
            "reward",
            "metrics_wn_chk",
            "metrics_wp_chk",
            "metrics_cell_area_um2",
            "metrics_delay_max_ps",
            "metrics_pstat_wc_uW",
            "metrics_edyn_fJ",
            "metrics_ymean_a0",
            "metrics_ymean_a1",
            "breakdown_cost",
        ]:
            if k in flat_best:
                print(f"{k:>24}: {flat_best[k]}")


if __name__ == "__main__":
    main()
