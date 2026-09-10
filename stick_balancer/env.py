"""
Gymnasium environment: balance (or swing up and balance) an N-link stick on a cart.

Two tasks share one environment:

  task="balance"  The stick starts upright with a little noise.
      Observation (Box, 2N+2):  [x, x_dot, theta_1..theta_N, theta_dot_1..theta_dot_N]
      Reward:  1.0 per step alive minus small quadratic penalties that keep the cart
               centred, the stick straight and the actions gentle.
      Ends (terminated) when the cart leaves the track or any link tilts more than
      `angle_limit`; truncated after `max_episode_steps`.

  task="swingup"  The stick starts hanging straight down and has to be lifted up.
      Observation (Box, 3N+2):  [x, x_dot, sin(theta_1..N), cos(theta_1..N), theta_dot_1..N]
               (sin/cos because raw angles wrap at +-pi, exactly when the stick
               passes through the bottom)
      Reward:  the dm_control cartpole swing-up reward, a product in [0, 1] of
                 upright        ((1 + tip_height / stick_length) / 2) ** 2
                 centred        1 - 0.5 (x / x_limit)^2
                 small_control  1 - 0.2 a^2
                 small_velocity 0.5 + 0.5 exp(-ln(10) (max |theta_dot| / 5)^2)
               so every sub-goal has to be met at once; nothing can be "bought"
               by sacrificing another.
      Ends when the cart hits the end of the track (a crash) or, once the stick
      has been caught upright, when it is dropped again (`drop_ends_episode`);
      otherwise truncated after `max_episode_steps`.  There is no "fell over"
      termination *before* the catch, because fallen is where it starts.  Without
      the drop rule, PPO settles for swinging through the top again and again.  For training, `upright_reset_prob`
      starts that fraction of episodes tilted-but-up so the catch is practised
      too (evaluation always starts hanging).

Action (Box, shape 1, in [-1, 1]) is scaled to a horizontal force u = action * max_force
in both tasks.

`randomize` draws hidden per-episode physics (pole length, pole mass, cart mass and the
actuator gain, i.e. the force per unit action, as multiplicative factors) so that a policy
has to identify the system from history; the factors are exposed as `env.hidden_scales`
for analysis but never in the observation.

Shakes (random disturbances).  With `shake_force > 0` the environment waits
until the stick has been balanced calmly for a while and then pushes it: a
force of random magnitude (up to `shake_force` N) and random direction
(within +-30 deg of horizontal) is applied for `shake_duration` seconds at a
random point of a random link (biased towards the tips, where it hurts most)
or on the cart.  Pushes recur at random, roughly `shake_interval` seconds
apart on average (a Poisson process), and never in the first `shake_warmup`
seconds.  The agent is *not* told about the push: it has to notice it from
the state and recover.  `info["shake"]` reports the push that is active so
the viewer can draw it.

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
        task: str = "balance",          # "balance" or "swingup"
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
        shake_force: float = 0.0,       # N; 0 disables disturbances
        shake_duration: float = 0.1,    # s
        shake_interval: float = 3.0,    # s, mean time between pushes
        shake_warmup: float = 2.0,      # s, no pushes before this
        shake_calm_angle: float = 0.1,  # rad; only push while every link is within this ...
        shake_calm_rate: float = 0.5,   # rad/s; ... and turning slower than this
        upright_reset_prob: float = 0.0,  # swing-up only: fraction of episodes that start near upright (a training curriculum)
        drop_ends_episode: bool = True,   # swing-up only: once caught upright, dropping the stick ends the episode
        upright_reset_tilt: float = 0.05,  # rad; how far those episodes start from vertical ...
        upright_reset_spin: float = 0.1,   # rad/s; ... and how fast the links are turning
        randomize: dict | None = None,  # hidden per-episode physics, e.g. {"link_lengths": (0.6, 1.4), "link_masses": (0.5, 2.0), "cart_mass": (0.6, 1.4)}
    ) -> None:
        super().__init__()
        assert task in ("balance", "swingup"), task
        self.task = task
        self._preset = {"realistic": realistic, "ideal": ideal}[physics]
        self._base_kwargs = dict(n_links=n_links, link_masses=link_masses, link_lengths=link_lengths,
                                 **(physics_overrides or {}))
        self.params = self._preset(**self._base_kwargs)
        self.physics = physics
        self.drop_ends_episode = drop_ends_episode
        self._caught = False
        self.upright_reset_prob = upright_reset_prob
        self.upright_reset_tilt = upright_reset_tilt
        self.upright_reset_spin = upright_reset_spin
        self.randomize = randomize or {}
        self.hidden_scales: dict[str, float] = {}   # the multiplicative factors drawn this episode
        self.actuator_gain = 1.0
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
        self.shake_force = shake_force
        self.shake_duration = shake_duration
        self.shake_interval = shake_interval
        self.shake_warmup = shake_warmup
        self.shake_calm_angle = shake_calm_angle
        self.shake_calm_rate = shake_calm_rate
        self._shake = None          # (link, fraction, fx, fy) while a push is active
        self._shake_steps_left = 0

        # Observation bounds are only advisory (the agent normalises them
        # itself); velocities are unbounded in principle.
        big = np.finfo(np.float32).max
        if task == "balance":
            high = np.concatenate(([x_limit * 2, big], np.full(n_links, angle_limit * 2), np.full(n_links, big)))
        else:
            high = np.concatenate(([x_limit * 2, big], np.ones(2 * n_links), np.full(n_links, big)))
        self.observation_space = spaces.Box(-high.astype(np.float32), high.astype(np.float32), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        self.q = np.zeros(n_links + 1)
        self.qd = np.zeros(n_links + 1)
        self.steps = 0

    # ------------------------------------------------------------------ helpers
    def _obs(self) -> np.ndarray:
        theta, thetad = self.q[1:], self.qd[1:]
        if self.task == "balance":
            parts = ([self.q[0], self.qd[0]], theta, thetad)
        else:
            parts = ([self.q[0], self.qd[0]], np.sin(theta), np.cos(theta), thetad)
        return np.concatenate(parts).astype(np.float32)

    @staticmethod
    def _wrap(theta: np.ndarray) -> np.ndarray:
        """Map angles to (-pi, pi] so that 'upright' is 0 whichever way the stick spun."""
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def _fallen(self) -> bool:
        return bool(abs(self.q[0]) > self.x_limit or np.any(np.abs(self._wrap(self.q[1:])) > self.angle_limit))

    def _crashed(self) -> bool:
        return bool(abs(self.q[0]) > self.x_limit)

    def _swingup_reward(self, a: float) -> float:
        tip_height = self.tip_positions()[-1, 1]
        # Squared so that "lower link up, upper link hanging" (tip at hinge height, 0.5)
        # earns 0.25 rather than 0.5: that half-up posture is a local optimum PPO otherwise settles in.
        upright = (0.5 * (1.0 + tip_height / self.params.total_length)) ** 2
        centred = 1.0 - 0.5 * (self.q[0] / self.x_limit) ** 2
        small_control = 1.0 - 0.2 * a**2
        small_velocity = 0.5 + 0.5 * np.exp(-np.log(10.0) * (np.max(np.abs(self.qd[1:])) / 5.0) ** 2)
        return float(upright * centred * small_control * small_velocity)

    def tip_positions(self) -> np.ndarray:
        """Cart hinge and link tips, shape (N+1, 2); handy for plotting."""
        return joint_positions(self.q, self.params)

    def _is_calm(self) -> bool:
        return bool(
            np.all(np.abs(self._wrap(self.q[1:])) < self.shake_calm_angle)
            and np.all(np.abs(self.qd[1:]) < self.shake_calm_rate)
        )

    def _maybe_start_shake(self) -> None:
        """Start a new random push if the stick is balanced and the dice say so."""
        if self.shake_force <= 0 or self._shake_steps_left > 0:
            return
        if self.steps * self.control_dt < self.shake_warmup or not self._is_calm():
            return
        # Poisson process: per-step probability = control_dt / mean interval.
        if self.np_random.random() > self.control_dt / self.shake_interval:
            return
        rng = self.np_random
        link = int(rng.integers(-1, self.n_links))               # -1 = cart
        fraction = 0.0 if link < 0 else float(np.sqrt(rng.random()))  # biased to the tip
        magnitude = self.shake_force * rng.uniform(0.3, 1.0)
        angle = rng.uniform(-np.pi / 6, np.pi / 6)                # within 30 deg of horizontal
        direction = rng.choice([-1.0, 1.0])
        self._shake = (link, fraction, direction * magnitude * np.cos(angle), magnitude * np.sin(angle))
        self._shake_steps_left = max(1, int(round(self.shake_duration / self.control_dt)))

    # -------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if self.randomize:
            # Draw hidden multiplicative factors (log-uniform) and rebuild the physics.
            self.hidden_scales = {
                k: float(np.exp(self.np_random.uniform(np.log(lo), np.log(hi))))
                for k, (lo, hi) in self.randomize.items()
            }
            base = self._preset(**self._base_kwargs)
            kw = dict(self._base_kwargs)
            kw["link_masses"] = base.link_masses * self.hidden_scales.get("link_masses", 1.0)
            kw["link_lengths"] = base.link_lengths * self.hidden_scales.get("link_lengths", 1.0)
            kw["cart_mass"] = base.cart_mass * self.hidden_scales.get("cart_mass", 1.0)
            self.params = self._preset(**kw)
            # an unknown motor constant: the force per unit action
            self.actuator_gain = self.hidden_scales.get("actuator_gain", 1.0)
        n1 = self.n_links + 1
        self.q = self.np_random.uniform(-self.init_noise, self.init_noise, size=n1)
        self.qd = self.np_random.uniform(-self.init_noise, self.init_noise, size=n1)
        if self.task == "swingup":
            if self.np_random.random() < self.upright_reset_prob:
                # curriculum: start already up, but tilted and moving, to practise the catch
                self.q[1:] = self.np_random.uniform(-self.upright_reset_tilt, self.upright_reset_tilt, size=self.n_links)
                self.qd[1:] = self.np_random.uniform(-self.upright_reset_spin, self.upright_reset_spin, size=self.n_links)
            else:
                self.q[1:] += np.pi       # hanging straight down (plus the same small noise)
        self.steps = 0
        self._shake = None
        self._shake_steps_left = 0
        self._caught = False
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))
        u = a * self.max_force * self.actuator_gain
        self._maybe_start_shake()
        external = [self._shake] if self._shake_steps_left > 0 else []
        self.q, self.qd = simulate(
            self.q, self.qd, u, self.control_dt, self.physics_substeps, self.params, external
        )
        self.steps += 1
        if self._shake_steps_left > 0:
            self._shake_steps_left -= 1

        truncated = self.steps >= self.max_episode_steps
        if self.task == "balance":
            terminated = self._fallen()
            reward = (
                1.0
                - self.position_cost * (self.q[0] / self.x_limit) ** 2
                - self.angle_cost * float(np.mean((self.q[1:] / self.angle_limit) ** 2))
                - self.action_cost * a**2
            )
            if terminated:
                reward = 0.0  # falling over is the worst thing that can happen
        else:
            upright_now = bool(np.all(np.abs(self._wrap(self.q[1:])) < self.angle_limit))
            dropped = self.drop_ends_episode and self._caught and not upright_now
            self._caught = self._caught or upright_now
            terminated = self._crashed() or dropped
            reward = 0.0 if terminated else self._swingup_reward(a)

        info = {"force": u, "x": float(self.q[0]), "angles": self.q[1:].copy(),
                "upright": bool(np.all(np.abs(self._wrap(self.q[1:])) < self.angle_limit)),
                "shake": external[0] if external else None}
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
