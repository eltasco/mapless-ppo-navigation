"""Paired benchmark, OOD sampling, and confidence interval tests."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from mapless_ppo.baselines import PotentialFieldController
from mapless_ppo.benchmark import wilson_interval
from mapless_ppo.environment import GymNavigationEnv
from mapless_ppo.frame_stack import FrameStackWrapper
from mapless_ppo.stress import StressWorld


def test_wilson_intervals():
    assert wilson_interval(0, 100)[0] == pytest.approx(0)
    assert wilson_interval(100, 100)[1] == pytest.approx(1)
    lo, hi = wilson_interval(450, 500)
    assert 0.87 < lo < 0.89
    assert 0.92 < hi < 0.93


def test_baseline_observation_contract():
    policy = PotentialFieldController()
    current = np.array([2.4, 0.0, 1.0] + [3.0] * 16, dtype=np.float32)
    action, _ = policy.predict(current)
    assert action.shape == (2,)
    assert np.isfinite(action).all()
    assert np.all(np.abs(action) <= 1)
    stacked_action, _ = policy.predict(np.tile(current, 3))
    np.testing.assert_allclose(action, stacked_action)
    with pytest.raises(ValueError):
        policy.predict(np.zeros(18, dtype=np.float32))


def test_uniform_layout_deterministic_and_safe_spawn():
    env = FrameStackWrapper(StressWorld(GymNavigationEnv(5, "hard"), layout="uniform"))
    try:
        obs0, _ = env.reset(seed=63_700)
        obstacles0 = list(env.robot.obstacles)
        obs1, _ = env.reset(seed=63_700)
        np.testing.assert_array_equal(obs0, obs1)
        assert obstacles0 == list(env.robot.obstacles)
        assert len(obstacles0) == 5
        assert not env.robot.check_collision()
        assert obs0.shape == (57,)
    finally:
        env.close()


def test_noisy_sensor_does_not_modify_physical_world():
    clean = FrameStackWrapper(StressWorld(GymNavigationEnv(3, "hard"), noise_std=0))
    noisy = FrameStackWrapper(StressWorld(GymNavigationEnv(3, "hard"), noise_std=0.08, dropout=0.1))
    try:
        original, _ = clean.reset(seed=64_000)
        measured, _ = noisy.reset(seed=64_000)
        assert clean.robot.obstacles == noisy.robot.obstacles
        np.testing.assert_array_equal(original.reshape(3, 19)[:, :3], measured.reshape(3, 19)[:, :3])
        assert not np.array_equal(original, measured)
        assert noisy.observation_space.contains(measured)
    finally:
        clean.close()
        noisy.close()


def test_risk_penalty_disabled_with_same_kinematics():
    a = GymNavigationEnv(1, "hard", risk_weight=1.5)
    b = GymNavigationEnv(1, "hard", risk_weight=0.0)
    try:
        a.reset(seed=5)
        b.reset(seed=5)
        for env in (a, b):
            env.robot.obstacles = [(0.75, 0)]
            env.robot.goal_x, env.robot.goal_y = 2.5, 0.0
        obs_a, reward_a, *_ = a.step(np.array([0.5, 0.0], dtype=np.float32))
        obs_b, reward_b, *_ = b.step(np.array([0.5, 0.0], dtype=np.float32))
        np.testing.assert_allclose(obs_a, obs_b)
        assert reward_b > reward_a
    finally:
        a.close()
        b.close()
