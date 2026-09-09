"""
Step 1 of the do-or-die experiment: build a dataset of an *optimal* controller acting on
sticks whose physics is hidden and differs every episode.

Expert = LQR recomputed from the true (hidden) parameters at every reset; it outputs the
force it wants, and the *action* it must send is force / (max_force * actuator_gain) because
the motor constant is one of the hidden parameters.  We add exploration noise to the
executed action so the data covers a wide region of state space, but the label stored for
imitation is the expert's clean action.

Each episode record holds, per step:  state (4), executed action (1), expert action (1),
plus the hidden multiplicative scales of the episode (length, pole mass, actuator gain).

    python interp/collect.py --episodes 1500 --steps 200 --out interp/data.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env import make_env  # noqa: E402
from lqr import lqr_gain  # noqa: E402

RANDOMIZE = {"actuator_gain": (0.3, 3.0), "link_lengths": (0.5, 1.5), "link_masses": (0.3, 3.0)}
HIDDEN = ["link_lengths", "link_masses", "actuator_gain"]


def make(seed: int | None = None):
    # ideal physics keeps the expert exactly optimal; wide init noise for coverage
    return make_env(1, physics="ideal", randomize=RANDOMIZE, init_noise=0.1, max_episode_steps=10_000)


def collect(episodes: int, steps: int, noise: float, seed: int):
    env = make()
    rng = np.random.default_rng(seed)
    S = np.zeros((episodes, steps, 4), np.float32)
    A_exec = np.zeros((episodes, steps), np.float32)
    A_expert = np.zeros((episodes, steps), np.float32)
    H = np.zeros((episodes, len(HIDDEN)), np.float32)
    lengths = np.zeros(episodes, np.int32)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        K = lqr_gain(env)
        scale = env.max_force * env.actuator_gain
        H[ep] = [env.hidden_scales[k] for k in HIDDEN]
        for t in range(steps):
            expert = float(np.clip(-(K @ obs) / scale, -1, 1))
            executed = float(np.clip(expert + noise * rng.standard_normal(), -1, 1))
            S[ep, t] = obs
            A_exec[ep, t] = executed
            A_expert[ep, t] = expert
            obs, _, terminated, _, _ = env.step([executed])
            if terminated:
                break
        lengths[ep] = t + 1
    return dict(states=S, action_exec=A_exec, action_expert=A_expert, hidden=H, lengths=lengths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--noise", type=float, default=0.15, help="exploration noise on the executed action")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data.npz")
    args = ap.parse_args()
    d = collect(args.episodes, args.steps, args.noise, args.seed)
    np.savez_compressed(args.out, **d, hidden_names=np.array(HIDDEN))
    full = np.mean(d["lengths"] == args.steps)
    print(f"{args.episodes} episodes, {d['lengths'].sum():,} steps, {full:.0%} ran the full {args.steps} steps -> {args.out}")


if __name__ == "__main__":
    main()
