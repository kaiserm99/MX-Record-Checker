"""Behavioural checks of env.py (run: python test_env.py)."""

import numpy as np
from gymnasium.utils.env_checker import check_env

from env import make_env


def test_gym_api_all_variants():
    for n in (1, 2, 3):
        for task in ("balance", "swingup"):
            check_env(make_env(n, task=task, shake_force=1.0), skip_render_check=True)


def test_balance_terminates_when_fallen():
    e = make_env(1, task="balance", physics="ideal")
    e.reset(seed=0)
    for t in range(1000):
        _, r, term, trunc, _ = e.step([0.0])
        if term:
            assert r == 0.0
            break
    assert term and t < 200, "an uncontrolled stick must fall within a few seconds"


def test_swingup_starts_down_and_never_terminates_for_falling():
    e = make_env(2, task="swingup")
    e.reset(seed=1)
    assert e.tip_positions()[-1, 1] < -0.9 * e.params.total_length, "should start hanging"
    for _ in range(300):
        _, r, term, trunc, info = e.step([0.0])
        assert not term and 0.0 <= r <= 1.0
    assert not info["upright"]


def test_swingup_drop_after_catch_terminates():
    e = make_env(1, task="swingup", physics="ideal")
    e.reset(seed=0)
    e.q[1:] = 0.0; e.qd[1:] = 0.0          # place it upright: this step counts as the catch
    for t in range(400):
        _, r, term, _, info = e.step([0.0])  # no control: it must fall and end the episode
        if term:
            break
    assert term and r == 0.0 and t < 200


def test_swingup_reward_is_one_when_balanced():
    e = make_env(3, task="swingup")
    e.reset(seed=0)
    e.q[:] = 0.0
    e.qd[:] = 0.0
    assert np.isclose(e._swingup_reward(0.0), 1.0)


def test_shake_only_when_calm():
    e = make_env(1, task="swingup", shake_force=1.0, shake_warmup=0.0, shake_interval=0.02)
    e.reset(seed=0)                      # hanging down => never "calm" => never pushed
    for _ in range(200):
        _, _, _, _, info = e.step([0.0])
        assert info["shake"] is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all env tests passed")
