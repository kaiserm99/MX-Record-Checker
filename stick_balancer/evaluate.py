"""
Watch a trained agent (numerically) and export a trajectory for the HTML viewer.

    python evaluate.py --links 1                     # prints episode returns
    python evaluate.py --links 2 --export traj.json  # also dumps cart/link positions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import make_env
from recipes import recipe


def load(run_dir: Path, algo: str, n_links: int, use_best: bool = True):
    env_kwargs = recipe(n_links)["env"]
    stats_dir = run_dir / "best" if use_best and (run_dir / "best" / "best_model.zip").exists() else run_dir
    model_path = stats_dir / ("best_model" if stats_dir.name == "best" else "model")
    venv = DummyVecEnv([lambda: make_env(n_links, **env_kwargs)])
    venv = VecNormalize.load(str(stats_dir / "vecnormalize.pkl"), venv)
    venv.training = False
    venv.norm_reward = False
    model = {"ppo": PPO, "sac": SAC}[algo].load(str(model_path), device="cpu")
    return model, venv


def run_episode(model, venv, seed: int, record: bool = False):
    raw_env = venv.envs[0].unwrapped
    venv.seed(seed)
    obs = venv.reset()
    total, frames = 0.0, []
    while True:
        if record:
            frames.append(raw_env.tip_positions().round(4).tolist())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        total += float(reward[0])
        if done[0]:
            break
    return total, frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", type=int, default=1)
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--export", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run or Path("runs") / f"{args.algo}_{args.links}links"
    model, venv = load(run_dir, args.algo, args.links)
    returns = [run_episode(model, venv, seed=i)[0] for i in range(args.episodes)]
    print(f"{args.links}-link {args.algo}: mean return {np.mean(returns):.1f} +/- {np.std(returns):.1f}  ({returns})")

    if args.export:
        total, frames = run_episode(model, venv, seed=123, record=True)
        raw = venv.envs[0].unwrapped
        args.export.write_text(json.dumps({
            "links": args.links, "algo": args.algo, "return": total,
            "dt": raw.control_dt, "x_limit": raw.x_limit,
            "link_lengths": raw.params.link_lengths.tolist(),
            "frames": frames,
        }))
        print("exported", len(frames), "frames to", args.export)


if __name__ == "__main__":
    main()
