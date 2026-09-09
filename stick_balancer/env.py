"""
Gymnasium environment: balance an N-link stick on a cart.

Observation (Box, shape 2N+2):  [x, x_dot, theta_1..theta_N, theta_dot_1..theta_dot_N]
Action      (Box, shape 1, in [-1, 1]):  scaled to a horizontal force  u = action * max_force
Reward      1.0 for every step alive minus small quadratic penalties that
            keep the cart centred, the stick straight and the actions gentle.
Episode ends (terminated) when the cart leaves the track or any link tilts
more than `angle_limit`; it is truncated after `max_episode_steps`.

N = 1 is the familiar CartPole; N = 2 is the double, N = 3 the triple inverted
pendulum, and so on.  By default the simulation is *realistic*: air drag on the
links and the cart, viscous + Coulomb friction in the hinges and on the rail
(see physics.py).  Pass physics="ideal" for the frictionless textbook model.
Everything else (masses, lengths, time step, limits) is a constructor argument
so experiments are explicit and reproducible.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from physics import ideal, joint_positions, realistic, simulate


class StickBalanceEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 50}

    def __init__(
        self,
        n_links: int = 1,
        *,
        max_force: float = 10.0,
        control_dt: float = 0.02,       # agent acts at 50 Hz (same as CartPole-v1)
        physics_substeps: int = 4,      # RK4 sub-steps per control step (h = 5 ms)
        x_limit: float = 2.4,           # half-width of the track (m)
        angle_limit: float = 0.35,      # rad; fail when any link leans more than this (~20 deg)
        init_noise: float = 0.05,       # uniform noise on the initial state
        max_episode_steps: int = 1000,  # 20 simulated seconds
        physics: str = "realistic",     # "realistic" (air drag + bearing/rail friction) or "ideal" (textbook CartPole)
        physics_overrides: dict | None = None,  # e.g. {"link_diameter": 0.03, "cart_coulomb": 0.02}
        link_masses=None,
        link_lengths=None,
        position_cost: float = 0.05,
        angle_cost: float = 0.05,
        action_cost: float = 0.001,
    ) -> None:
        super().__init__()
        preset = {"realistic": realistic, "ideal": ideal}[physics]
        self.params = preset(
            n_links=n_links,
            link_masses=link_masses,
            link_lengths=link_lengths,
            **(physics_overrides or {}),
        )
        self.physics = physics
        self.n_links = n_links
        self.max_force = max_force
        self.control_dt = control_dt
        self.physics_substeps = physics_substeps
        self.x_limit = x_limit
        self.angle_limit = angle_limit
        self.init_noise = init_noise
        self.max_episode_steps = max_episode_steps
        self.position_cost = position_cost
        self.angle_cost = angle_cost
        self.action_cost = action_cost

        # Observation bounds are only advisory (the agent normalises them
        # itself); velocities are unbounded in principle.
        high = np.concatenate(
            (
                [x_limit * 2, np.finfo(np.float32).max],
                np.full(n_links, angle_limit * 2),
                np.full(n_links, np.finfo(np.float32).max),
            )
        ).astype(np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        self.q = np.zeros(n_links + 1)
        self.qd = np.zeros(n_links + 1)
        self.steps = 0

    # ------------------------------------------------------------------ helpers
    def _obs(self) -> np.ndarray:
        return np.concatenate(([self.q[0], self.qd[0]], self.q[1:], self.qd[1:])).astype(np.float32)

    def _fallen(self) -> bool:
        return bool(abs(self.q[0]) > self.x_limit or np.any(np.abs(self.q[1:]) > self.angle_limit))

    def tip_positions(self) -> np.ndarray:
        """Cart hinge and link tips, shape (N+1, 2); handy for plotting."""
        return joint_positions(self.q, self.params)

    # -------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        n1 = self.n_links + 1
        self.q = self.np_random.uniform(-self.init_noise, self.init_noise, size=n1)
        self.qd = self.np_random.uniform(-self.init_noise, self.init_noise, size=n1)
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))
        u = a * self.max_force
        self.q, self.qd = simulate(self.q, self.qd, u, self.control_dt, self.physics_substeps, self.params)
        self.steps += 1

        terminated = self._fallen()
        truncated = self.steps >= self.max_episode_steps

        reward = (
            1.0
            - self.position_cost * (self.q[0] / self.x_limit) ** 2
            - self.angle_cost * float(np.mean((self.q[1:] / self.angle_limit) ** 2))
            - self.action_cost * a**2
        )
        if terminated:
            reward = 0.0  # falling over is the worst thing that can happen

        info = {"force": u, "x": float(self.q[0]), "angles": self.q[1:].copy()}
        return self._obs(), float(reward), terminated, truncated, info


def make_env(n_links: int = 1, **kwargs) -> StickBalanceEnv:
    return StickBalanceEnv(n_links=n_links, **kwargs)


# Optional: register so that gym.make("StickBalance-v0", n_links=2) also works.
for _n in range(1, 6):
    gym.register(
        id=f"StickBalance{_n}-v0",
        entry_point="env:StickBalanceEnv",
        kwargs={"n_links": _n},
    )
