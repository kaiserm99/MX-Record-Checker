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
  * Shake phase: after the stick is balanced we continue training with random pushes
    (see env.py).  The strength follows a curriculum (train.py PushCurriculum): it
    starts at `shake_start` and grows by 25% every time the agent copes with the
    current level, up to `shake_force`.  We first tried fixed strengths and found
    that the upper links of a jointed stick are so light (50 g, 33 g) that a 0.75 N
    shove for 0.1 s spins the top link at ~9 rad/s, which nothing can recover from;
    the curriculum finds the strongest recoverable push by itself.  Magnitudes are
    random in [0.3, 1] x current strength so every episode mixes easy and hard pushes.
  * Swing-up: the stick starts hanging down.  Most *training* episodes (upright_reset_prob)
    start nearly upright instead, with the balance task's small noise, so the network learns
    the hold first, exactly as the balance agents did; the rest teach the swing.  Once
    caught, dropping the stick ends the episode (env.drop_ends_episode), otherwise PPO
    settles for swinging through the top again and again.  Evaluation always starts hanging.  More force is needed to pump energy
    into it (the classic single cart-pole swing-up works with ~10 N on a 1 kg cart,
    jointed sticks need more), the reward is dm_control's product reward, and the
    "solved" threshold is lower than for balancing because the first seconds of every
    episode are necessarily spent swinging (a perfect swing-up of one link scores
    roughly 850-900 out of 1000).
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


# Warm-started phases (pushes) must not wander away from an already-good policy:
# the untouched 2-link agent survives 0.1 N pushes perfectly, yet fine-tuning it at
# the full learning rate dropped it to ~600 within 20k steps.  Smaller steps fix that.
FINETUNE_PPO = dict(learning_rate=LinearSchedule(1e-4, 0.0, 1.0), clip_range=LinearSchedule(0.1, 0.0, 1.0))


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
        "shake": dict(shake_force=1.5, shake_duration=0.1, shake_interval=3.0, shake_warmup=2.0),
        "shake_start": 0.25,
        "shake_timesteps": 600_000,
        "swingup": dict(task="swingup", max_force=15.0, init_noise=0.05, upright_reset_prob=0.5),
        "swingup_timesteps": 2_000_000,
        "swingup_stop": 800.0,
        "ppo": _ppo(lr=3e-4, clip=0.2, net=64, n_steps=512, batch=256, gamma=0.99, lam=0.95, epochs=10),
        "sac": _sac(net=256),
    },
    2: {  # double inverted pendulum: needs a few hundred thousand to a couple million steps
        "env": dict(max_force=20.0, angle_limit=0.4),
        "total_timesteps": 2_000_000,
        "shake": dict(shake_force=0.5, shake_duration=0.1, shake_interval=3.0, shake_warmup=2.0),
        "shake_start": 0.05,
        "shake_timesteps": 1_000_000,
        "swingup": dict(task="swingup", max_force=20.0, init_noise=0.05, upright_reset_prob=0.7),
        "swingup_timesteps": 4_000_000,
        "swingup_stop": 750.0,
        "ppo": _ppo(lr=3e-4, clip=0.2, net=128, n_steps=512, batch=256, gamma=0.99, lam=0.95, epochs=10),
        "sac": _sac(net=400),
    },
    3: {  # triple inverted pendulum: genuinely hard, plan for several million steps
        "env": dict(max_force=30.0, angle_limit=0.5),
        "total_timesteps": 5_000_000,
        "shake": dict(shake_force=0.3, shake_duration=0.1, shake_interval=3.0, shake_warmup=2.0),
        "shake_start": 0.03,
        "shake_timesteps": 1_500_000,
        "swingup": dict(task="swingup", max_force=35.0, init_noise=0.05, upright_reset_prob=0.7),
        "swingup_timesteps": 6_000_000,
        "swingup_stop": 700.0,
        "ppo": _ppo(lr=3e-4, clip=0.2, net=256, n_steps=1024, batch=512, gamma=0.99, lam=0.95, epochs=10),
        "sac": _sac(net=400),
    },
}


def recipe(n_links: int) -> dict:
    """Recipes for more links than we tuned fall back to the largest one."""
    return RECIPES[min(n_links, max(RECIPES))]


def env_kwargs_for(n_links: int, phase: str = "balance", task: str = "balance") -> dict:
    """Environment arguments for a task ('balance' or 'swingup') and a training
    phase ('balance' = no pushes, 'shake' = random pushes once the stick is up)."""
    r = recipe(n_links)
    kwargs = {**r["env"]}
    if task == "swingup":
        kwargs.update(r["swingup"])
        kwargs.pop("ppo_extra", None)
    if phase == "shake":
        kwargs.update(r["shake"])
        if task == "swingup":
            kwargs["shake_warmup"] = 8.0   # give the swing-up time before the first push
    return kwargs


# Swing-up needs structured exploration: generalised state-dependent exploration (gSDE),
# which the rl-baselines3-zoo uses for PPO on Pendulum swing-up.
SWINGUP_PPO = dict(use_sde=True, sde_sample_freq=4)


def budget_for(n_links: int, phase: str, task: str) -> int:
    r = recipe(n_links)
    if task == "swingup" and phase == "balance":
        return r["swingup_timesteps"]
    return r["total_timesteps"] if phase == "balance" else r["shake_timesteps"]


def stop_threshold(n_links: int, task: str, max_return: float) -> float:
    """Evaluation return at which training stops: 98% of perfect for balancing,
    a lower, per-link value for swing-up (the swing itself costs reward)."""
    return recipe(n_links)["swingup_stop"] if task == "swingup" else 0.98 * max_return


def run_name(n_links: int, algo: str, phase: str, task: str) -> str:
    return f"{algo}_{n_links}links" + ("_swingup" if task == "swingup" else "") + ("_shake" if phase == "shake" else "")
