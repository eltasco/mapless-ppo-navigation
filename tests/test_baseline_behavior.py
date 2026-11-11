"""Behavioral tests for actual baseline movement, not just output shape."""
import math

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from mapless_ppo.baselines import LegacyPotentialFieldController, PotentialFieldController
from mapless_ppo.tuning import validate_seed_split
from mapless_ppo.benchmark import make_benchmark_env
from mapless_ppo.rollouts import run_episode


def observation(bearing: float, rays=None):
    return np.array([2.5, math.sin(bearing), math.cos(bearing)] +
                    (rays if rays is not None else [3.0] * 16), dtype=np.float32)


def test_reversing_when_goal_is_behind():
    obs = observation(math.pi)
    revised, _ = PotentialFieldController().predict(obs)
    historical, _ = LegacyPotentialFieldController().predict(obs)
    assert revised[0] < -0.1  # The revised field supports signed motion.
    assert historical[0] == pytest.approx(0)
    assert abs(revised[1]) < 0.1


def test_forward_when_goal_is_ahead():
    action, _ = PotentialFieldController().predict(observation(0))
    assert action[0] > 0.0
    assert abs(action[1]) < 1e-5


def test_sensor_obstacle_changes_command_and_outputs_finite():
    clear = observation(0)
    blocked = observation(0, [3.0] * 8 + [0.45] + [3.0] * 7)
    ctrl = PotentialFieldController()
    clean_action, _ = ctrl.predict(clear)
    avoidance_action, _ = ctrl.predict(blocked)
    assert not np.allclose(avoidance_action, clean_action)
    assert np.isfinite(avoidance_action).all()
    assert np.all(np.abs(avoidance_action) <= 1)
    np.testing.assert_allclose(ctrl.predict(np.tile(blocked, 3))[0], avoidance_action)


def test_benchmark_supports_single_observation_and_ood():
    env = make_benchmark_env(2, "hard", "uniform", 0.04, 0.04, "single")
    try:
        obs, _ = env.reset(seed=33000)
        assert obs.shape == (19,)
        metrics, _ = run_episode(env, PotentialFieldController(), 33000, None)
        assert metrics.outcome in ("success", "collision", "timeout")
        assert 1 <= metrics.steps <= 900
    finally:
        env.close()


def test_validation_seeds_cannot_overlap_historical_test_seeds():
    with pytest.raises(ValueError):
        validate_seed_split(59999, 2)
    with pytest.raises(ValueError):
        validate_seed_split(60499, 2)
    validate_seed_split(30000, 30)
