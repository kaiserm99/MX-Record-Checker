"""
Linear-quadratic regulator for the stick: the textbook optimal controller near upright.

    A, B  = linearise(env)              continuous-time linearisation at upright (finite differences)
    Ad, Bd = discretise(A, B, dt)        exact zero-order-hold discretisation
    K     = lqr(Ad, Bd, Q, R)            u = -K x minimises sum x'Qx + R u^2
    K     = lqr_gain(env)                all of the above with the default weights

State order matches the balance observation: [x, x_dot, theta_1..N, theta_dot_1..N].
Pure NumPy (no scipy): the Riccati equation is solved by iteration.
"""

from __future__ import annotations

import numpy as np

from physics import accelerations


def linearise(env, eps: float = 1e-6):
    p, n = env.params, env.n_links

    def f(state, u):
        q = np.concatenate(([state[0]], state[2:2 + n]))
        qd = np.concatenate(([state[1]], state[2 + n:]))
        acc = accelerations(q, qd, u, p)
        return np.concatenate(([qd[0], acc[0]], qd[1:], acc[1:]))

    dim = 2 * n + 2
    A = np.zeros((dim, dim))
    for i in range(dim):
        d = np.zeros(dim)
        d[i] = eps
        A[:, i] = (f(d, 0.0) - f(-d, 0.0)) / (2 * eps)
    B = (f(np.zeros(dim), eps) - f(np.zeros(dim), -eps)) / (2 * eps)
    return A, B


def discretise(A, B, dt):
    dim = A.shape[0]
    M = np.zeros((dim + 1, dim + 1))
    M[:dim, :dim] = A * dt
    M[:dim, dim] = B * dt
    s = max(0, int(np.ceil(np.log2(max(np.linalg.norm(M, 1), 1e-12)))) + 1)
    Ms = M / 2**s
    E = np.eye(dim + 1)
    term = np.eye(dim + 1)
    for k in range(1, 20):
        term = term @ Ms / k
        E += term
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


def default_weights(n_links: int):
    Q = np.diag(np.concatenate(([1.0, 0.1], np.full(n_links, 10.0), np.full(n_links, 0.1))))
    return Q, np.array(0.01)


def lqr_gain(env) -> np.ndarray:
    """K for the env's *current* physics (recompute after every reset when randomising)."""
    A, B = linearise(env)
    Ad, Bd = discretise(A, B, env.control_dt)
    Q, R = default_weights(env.n_links)
    return lqr(Ad, Bd, Q, R)
