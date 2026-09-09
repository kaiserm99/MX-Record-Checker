"""
Train an agent to balance an N-link stick with Stable-Baselines3.

    python train.py --links 1                 # CartPole-like, a few minutes on CPU
    python train.py --links 2 --algo ppo      # double inverted pendulum
    python train.py --links 3 --timesteps 4e6 # triple inverted pendulum
    python train.py --links 2 --phase shake   # continue from the balanced agent, with random pushes
    python train.py --links 1 --task swingup  # stick starts hanging down and must be lifted up
    python train.py --links 1 --task swingup --phase shake   # ... and then survive pushes

Outputs go to runs/<algo>_<N>links/:
    model.zip            trained policy
    vecnormalize.pkl     running observation/reward statistics (needed at test time!)
    progress.csv         learning curve (SB3 logger)
    best/                best model found by periodic evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from env import make_env
from recipes import FINETUNE_PPO, budget_for, env_kwargs_for, recipe, run_name, stop_threshold


class PushCurriculum(BaseCallback):
    """Grow the push strength as the agent copes.

    Starts at `start` newtons; every evaluation in which the agent scores at least
    `advance_at` raises the strength by `growth` (x1.25) up to `target`.  The "best
    model" bookkeeping is reset at each raise so the saved best model is always the
    best at the hardest level reached.  Training may only stop (reward threshold)
    once the target strength is reached.  The strength reached is written to
    config.json so evaluate.py tests the agent at that level."""

    def __init__(self, eval_cb: EvalCallback, train_env, eval_env, start: float, target: float,
                 advance_at: float, growth: float = 1.25):
        super().__init__()
        self.eval_cb, self.train_env, self.eval_env = eval_cb, train_env, eval_env
        self.force, self.target, self.advance_at, self.growth = min(start, target), target, advance_at, growth
        self._apply()

    def _apply(self):
        self.train_env.set_attr("shake_force", self.force)
        self.eval_env.set_attr("shake_force", self.force)

    @property
    def at_target(self) -> bool:
        return self.force >= self.target - 1e-9

    def _on_step(self) -> bool:  # called after every evaluation (callback_after_eval)
        self.logger.record("curriculum/shake_force", self.force)
        if not self.at_target and self.eval_cb.last_mean_reward >= self.advance_at:
            self.force = min(self.force * self.growth, self.target)
            self._apply()
            self.eval_cb.best_mean_reward = -np.inf     # "best" now means best at this level
            print(f"push curriculum: agent copes, raising push strength to {self.force:.3f} N")
        return True


class StopWhenSolvedAtTarget(BaseCallback):
    """Stop training on a new best above `threshold`, but only once the push
    curriculum (if any) has reached its target strength."""

    def __init__(self, threshold: float, curriculum: "PushCurriculum | None"):
        super().__init__()
        self.threshold, self.curriculum = threshold, curriculum

    def _on_step(self) -> bool:
        reward = self.parent.best_mean_reward
        if reward >= self.threshold and (self.curriculum is None or self.curriculum.at_target):
            print(f"Stopping training because the mean reward {reward:.2f} is above the threshold {self.threshold}")
            return False
        return True


def build_vec_env(n_links: int, env_kwargs: dict, n_envs: int, seed: int, subprocess: bool):
    """Parallel envs + VecNormalize (observation and reward scaling)."""
    vec = make_vec_env(
        make_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"n_links": n_links, **env_kwargs},
        vec_env_cls=SubprocVecEnv if subprocess else DummyVecEnv,
    )
    return VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--links", type=int, default=1)
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--task", choices=["balance", "swingup"], default="balance",
                    help="'swingup' starts the stick hanging down")
    ap.add_argument("--phase", choices=["balance", "shake"], default="balance",
                    help="'shake' warm-starts from the balanced agent and adds random pushes")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="run directory to warm-start from (default for --phase shake: the balance run)")
    ap.add_argument("--timesteps", type=float, default=None, help="override the recipe budget")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-subprocess", action="store_true", help="run envs in-process (debugging)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # Tiny MLPs gain nothing from intra-op threads; leave the cores to the env processes.
    torch.set_num_threads(1)

    r = recipe(args.links)
    env_kwargs = env_kwargs_for(args.links, args.phase, args.task)
    total_timesteps = int(args.timesteps or budget_for(args.links, args.phase, args.task))
    out = args.out or Path("runs") / run_name(args.links, args.algo, args.phase, args.task)
    out.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    init_from = args.init_from
    if init_from is None and args.phase == "shake":
        init_from = Path("runs") / run_name(args.links, args.algo, "balance", args.task)

    # SAC is off-policy and learns from a single env just fine; PPO wants many.
    n_envs = args.n_envs if args.algo == "ppo" else 1
    train_env = build_vec_env(args.links, env_kwargs, n_envs, args.seed, not args.no_subprocess)
    eval_env = build_vec_env(args.links, env_kwargs, 1, args.seed + 1000, False)
    if init_from is not None:
        # Continue with the observation statistics the previous agent was trained on.
        stats = init_from / "best" / "vecnormalize.pkl"
        stats = stats if stats.exists() else init_from / "vecnormalize.pkl"
        train_env = VecNormalize.load(str(stats), train_env.venv)
        eval_env = VecNormalize.load(str(stats), eval_env.venv)
    eval_env.training = False       # freeze statistics during evaluation
    eval_env.norm_reward = False    # report raw (un-normalised) episode returns

    max_return = make_env(args.links, **env_kwargs).max_episode_steps  # 1 reward/step
    threshold = stop_threshold(args.links, args.task, max_return)
    curriculum = None
    stopper = StopWhenSolvedAtTarget(threshold, None)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out / "best"),
        log_path=str(out),
        eval_freq=max(20_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        # stop early once the agent balances (almost) every eval episode to the end
        callback_on_new_best=stopper,
    )
    if args.phase == "shake":
        curriculum = PushCurriculum(
            eval_cb, train_env, eval_env,
            start=r["shake_start"], target=env_kwargs["shake_force"],
            advance_at=0.95 * threshold,
        )
        stopper.curriculum = curriculum
        eval_cb.callback = curriculum        # SB3 keeps the after-eval callback in .callback
        curriculum.parent = eval_cb
    callback = eval_cb

    algo_cls = {"ppo": PPO, "sac": SAC}[args.algo]
    algo_kwargs = dict(r[args.algo])
    if init_from is not None and args.algo == "ppo":
        algo_kwargs.update(FINETUNE_PPO)   # gentler updates when continuing from a trained agent
    model = algo_cls("MlpPolicy", train_env, seed=args.seed, device="cpu", verbose=0, **algo_kwargs)
    if init_from is not None:
        weights = init_from / "best" / "best_model.zip"
        weights = weights if weights.exists() else init_from / "model.zip"
        model.set_parameters(str(weights), device="cpu")   # same network shape, new task
        print("warm-started from", weights)
    model.set_logger(configure(str(out), ["stdout", "csv"]))

    (out / "config.json").write_text(json.dumps({
        "links": args.links, "algo": args.algo, "task": args.task, "phase": args.phase, "timesteps": total_timesteps,
        "n_envs": n_envs, "seed": args.seed, "recipe": r, "env": env_kwargs,
        "init_from": str(init_from) if init_from else None,
    }, indent=2, default=str))

    print(f"Training {args.algo.upper()} on {args.links}-link stick ({args.task}, {args.phase}) for {total_timesteps:,} steps -> {out}")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    model.save(out / "model")
    train_env.save(str(out / "vecnormalize.pkl"))
    # Keep the eval statistics in sync with the "best" model too.
    (out / "best").mkdir(exist_ok=True)
    train_env.save(str(out / "best" / "vecnormalize.pkl"))
    if curriculum is not None:
        cfg = json.loads((out / "config.json").read_text())
        cfg["shake_force_reached"] = curriculum.force
        (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
        print(f"push strength reached: {curriculum.force:.3f} N (target {curriculum.target} N)")
    best = eval_cb.best_mean_reward
    print("done; best eval mean reward:", "n/a (no evaluation ran)" if best == -np.inf else round(float(best), 1))


if __name__ == "__main__":
    main()
