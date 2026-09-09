"""
One place for every tuning decision, per number of links.

Where do these numbers come from?
  * PPO: Stable-Baselines3's standard continuous-control settings (rollout of ~2048
    transitions, gamma 0.99, gae_lambda 0.95, lr 3e-4, clip 0.2, 10 epochs), which are
    what the rl-baselines3-zoo uses for MuJoCo locomotion.  We tried the zoo's tuned
    CartPole-v1 entry (n_steps 32, gamma 0.98, gae_lambda 0.8, 20 epochs) first: it is
    tuned for 500-step discrete-action episodes and learned *worse* on our 1000-step
    continuous version, so we dropped it.  Learning rate and clip range decay linearly
    to zero (SB3's LinearSchedule), which is what the zoo's "lin_" prefix does.
    Observation/reward normalisation (VecNormalize) is on, as the zoo does for
    InvertedDoublePendulum.
  * SAC: the zoo's PyBullet InvertedDoublePendulum recipe (lr 7.3e-4, buffer 300k,
    gamma 0.98, tau 0.02, train_freq 8, gradient_steps 8, learning_starts 10k, gSDE,
    log_std_init -3, net [400, 300]).
  * Env: force authority and the allowed lean grow with the number of links because a
    longer, jointed stick has faster unstable modes and needs more control bandwidth.
Change numbers here, not in train.py.
"""

from __future__ import annotations

from stable_baselines3.common.utils import LinearSchedule


def _ppo(lr: float, clip: float, net: int, n_steps: int, batch: int, gamma: float, lam: float, epochs: int) -> dict:
    return dict(
        n_steps=n_steps, batch_size=batch, n_epochs=epochs, gamma=gamma, gae_lambda=lam,
        learning_rate=LinearSchedule(lr, 0.0, 1.0), clip_range=LinearSchedule(clip, 0.0, 1.0),
        ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[net, net], vf=[net, net])),
    )


def _sac(net: int) -> dict:
    return dict(
        learning_rate=7.3e-4, buffer_size=300_000, learning_starts=10_000, batch_size=256,
        gamma=0.98, tau=0.02, train_freq=8, gradient_steps=8, ent_coef="auto",
        use_sde=True, policy_kwargs=dict(log_std_init=-3, net_arch=[net, net]),
    )


RECIPES: dict[int, dict] = {
    1: {  # classic CartPole: trivially easy, a minute on a laptop
        "env": dict(max_force=10.0, angle_limit=0.35),
        "total_timesteps": 200_000,
        "ppo": _ppo(lr=3e-4, clip=0.2, net=64, n_steps=512, batch=256, gamma=0.99, lam=0.95, epochs=10),
        "sac": _sac(net=256),
    },
    2: {  # double inverted pendulum: needs a few hundred thousand to a couple million steps
        "env": dict(max_force=20.0, angle_limit=0.4),
        "total_timesteps": 2_000_000,
        "ppo": _ppo(lr=3e-4, clip=0.2, net=128, n_steps=512, batch=256, gamma=0.99, lam=0.95, epochs=10),
        "sac": _sac(net=400),
    },
    3: {  # triple inverted pendulum: genuinely hard, plan for several million steps
        "env": dict(max_force=30.0, angle_limit=0.5),
        "total_timesteps": 5_000_000,
        "ppo": _ppo(lr=3e-4, clip=0.2, net=256, n_steps=1024, batch=512, gamma=0.99, lam=0.95, epochs=10),
        "sac": _sac(net=400),
    },
}


def recipe(n_links: int) -> dict:
    """Recipes for more links than we tuned fall back to the largest one."""
    return RECIPES[min(n_links, max(RECIPES))]
