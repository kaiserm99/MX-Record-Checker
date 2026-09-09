"""
Interpretability probe: is the trained balance policy a linear controller in disguise?

Near the upright equilibrium the physics is well approximated by a linear system
x' = A x + B u, and the textbook optimal controller is the LQR gain u = -K x.
This script

  1. linearises physics.py numerically around upright (finite differences),
  2. solves the discrete-time LQR (Riccati iteration, pure NumPy) for a reference gain,
  3. takes the Jacobian of the trained policy (through its observation normaliser)
     at upright: the "gain" the network actually implements,
  4. checks whether the linearised plant is stable under the network's gain,
  5. measures how linear the policy really is along its own trajectories (R^2 of a
     linear fit to its outputs).

    python probe_linear.py --links 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import make_env
from physics import accelerations
from recipes import env_kwargs_for, run_name


def linearise(env, eps: float = 1e-6):
    """Continuous-time A (2n+2 x 2n+2) and B (2n+2) at x = 0, u = 0.  State order matches
    the observation: [x, x_dot, theta..., theta_dot...]."""
    p, n = env.params, env.n_links

    def f(state, u):
        q = np.concatenate(([state[0]], state[2:2 + n]))
        qd = np.concatenate(([state[1]], state[2 + n:]))
        acc = accelerations(q, qd, u, p)
        return np.concatenate(([qd[0], acc[0]], qd[1:], acc[1:]))

    dim = 2 * n + 2
    A = np.zeros((dim, dim))
    for i in range(dim):
        d = np.zeros(dim); d[i] = eps
        A[:, i] = (f(d, 0.0) - f(-d, 0.0)) / (2 * eps)
    B = (f(np.zeros(dim), eps) - f(np.zeros(dim), -eps)) / (2 * eps)
    return A, B


def discretise(A, B, dt):
    """Exact zero-order hold via the matrix exponential of the augmented system."""
    dim = A.shape[0]
    M = np.zeros((dim + 1, dim + 1)); M[:dim, :dim] = A * dt; M[:dim, dim] = B * dt
    # matrix exponential by scaling and squaring of a Taylor series (no scipy needed)
    s = max(0, int(np.ceil(np.log2(max(np.linalg.norm(M, 1), 1e-12)))) + 1)
    Ms = M / 2**s
    E = np.eye(dim + 1); term = np.eye(dim + 1)
    for k in range(1, 20):
        term = term @ Ms / k; E += term
    for _ in range(s):
        E = E @ E
    return E[:dim, :dim], E[:dim, dim]


def lqr(Ad, Bd, Q, R, iters=5000):
    P = Q.copy()
    for _ in range(iters):
        K = (Bd @ P @ Bd + R) ** -1 * (Bd @ P @ Ad)
        P_new = Q + Ad.T @ P @ Ad - np.outer(Ad.T @ P @ Bd, K)
        if np.abs(P_new - P).max() < 1e-10:
            break
        P = P_new
    return (Bd @ P @ Bd + R) ** -1 * (Bd @ P @ Ad)


def policy_gain(model, venv, env, obs0):
    """d(force)/d(obs) of the deterministic policy at obs0, through VecNormalize."""
    mean, var = venv.obs_rms.mean, venv.obs_rms.var
    x = torch.tensor(obs0, dtype=torch.float32, requires_grad=True)
    xn = (x - torch.tensor(mean, dtype=torch.float32)) / torch.sqrt(torch.tensor(var, dtype=torch.float32) + venv.epsilon)
    xn = torch.clamp(xn, -venv.clip_obs, venv.clip_obs)
    a = model.policy.get_distribution(xn[None]).distribution.mean.squeeze()
    a.backward()
    return env.max_force * x.grad.numpy(), env.max_force * float(a.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", type=int, default=2)
    ap.add_argument("--algo", default="ppo")
    args = ap.parse_args()
    n = args.links
    run = Path("runs") / run_name(n, args.algo, "balance", "balance")
    kw = env_kwargs_for(n)
    venv = VecNormalize.load(str(run / "best" / "vecnormalize.pkl"), DummyVecEnv([lambda: make_env(n, **kw)]))
    venv.training = False; venv.norm_reward = False
    env = venv.envs[0].unwrapped
    model = PPO.load(str(run / "best" / "best_model"), device="cpu")

    A, B = linearise(env)
    Ad, Bd = discretise(A, B, env.control_dt)
    ol = np.linalg.eigvals(A)
    print(f"{n} links: open-loop unstable poles (1/s): {np.sort(ol.real[ol.real > 1e-6])[::-1].round(2)}")

    Q = np.diag(np.concatenate(([1.0, 0.1], np.full(n, 10.0), np.full(n, 0.1))))
    K_lqr = lqr(Ad, Bd, Q, np.array(0.01))
    k_net, bias = policy_gain(model, venv, env, np.zeros(2 * n + 2, dtype=np.float32))
    labels = ["x", "x'"] + [f"th{i+1}" for i in range(n)] + [f"th{i+1}'" for i in range(n)]
    print("gain vector u = -K x (N per unit state):")
    print("   state     LQR         network")
    for lab, kl, kn in zip(labels, K_lqr, -k_net):
        print(f"   {lab:6s} {kl:9.2f} {kn:11.2f}")
    print(f"network force at exact upright: {bias:+.3f} N (bias)")
    cl_lqr = np.abs(np.linalg.eigvals(Ad - np.outer(Bd, K_lqr))).max()
    cl_net = np.abs(np.linalg.eigvals(Ad - np.outer(Bd, -k_net))).max()
    print(f"closed-loop spectral radius (<1 = stable): LQR {cl_lqr:.3f}, network's linearisation {cl_net:.3f}")
    cos = float(K_lqr @ -k_net / (np.linalg.norm(K_lqr) * np.linalg.norm(k_net)))
    print(f"cosine similarity of the two gain vectors: {cos:.3f}")

    # How linear is the policy on the states it actually visits?
    obs_list, act_list = [], []
    for ep in range(5):
        venv.seed(ep); obs = venv.reset()
        for _ in range(1000):
            a, _ = model.predict(obs, deterministic=True)
            obs_list.append(venv.get_original_obs()[0].copy()); act_list.append(float(np.asarray(a).reshape(-1)[0]) * env.max_force)
            obs, _, d, _ = venv.step(a)
            if d[0]:
                break
    X = np.array(obs_list); y = np.array(act_list)
    Xa = np.hstack([X, np.ones((len(X), 1))])
    coef, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    r2 = 1 - np.var(y - Xa @ coef) / np.var(y)
    print(f"linear fit of network force on {len(X)} visited states: R^2 = {r2:.4f}; "
          f"force range {y.min():+.2f}..{y.max():+.2f} N, state spread |theta| <= {np.abs(X[:, 2:2+n]).max():.3f} rad")


if __name__ == "__main__":
    main()
