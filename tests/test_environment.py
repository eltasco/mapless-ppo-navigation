"""Gymnasium API, seeded sampling and frame-history checks."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from stable_baselines3.common.env_checker import check_env
from mapless_ppo.environment import GymNavigationEnv
from mapless_ppo.scenarios import VariableObstacleCount, parse_radius, set_episode_radius
from mapless_ppo.frame_stack import FrameStackWrapper


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("stage", ["easy", "hard", "mixed", "hard_passable"])
def test_count_and_observation(n, stage):
    env = FrameStackWrapper(GymNavigationEnv(n, stage))
    try:
        for seed in [42, 60_001, 60_002]:
            obs, _ = env.reset(seed=seed)
            assert len(env.robot.obstacles) == n
            assert obs.shape == (57,)
            assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_fixed_seed_reproduces_geometry():
    env = GymNavigationEnv(5, "hard")
    obs1, _ = env.reset(seed=60_012)
    obstacles1 = list(env.robot.obstacles)
    goal1 = env.robot.goal_x, env.robot.goal_y
    obs2, _ = env.reset(seed=60_012)
    assert np.array_equal(obs1, obs2)
    assert obstacles1 == env.robot.obstacles
    assert goal1 == (env.robot.goal_x, env.robot.goal_y)
    env.close()


def test_frame_lags_are_20_10_0():
    env = FrameStackWrapper(GymNavigationEnv(1))
    try:
        s0, _ = env.reset(seed=42)
        assert np.array_equal(s0[:19], s0[19:38])
        assert np.array_equal(s0[19:38], s0[38:])
        observations = [s0[-19:].copy()]
        for _ in range(20):
            s, _, terminated, truncated, _ = env.step(np.array([0.2, 0.05], dtype=np.float32))
            assert not (terminated or truncated)
            observations.append(s[-19:].copy())
        assert np.array_equal(s[:19], observations[0])
        assert np.array_equal(s[19:38], observations[10])
        assert np.array_equal(s[-19:], observations[20])
    finally:
        env.close()


def test_variable_count_and_radius():
    env = VariableObstacleCount(FrameStackWrapper(GymNavigationEnv(2, "mixed")), 2, 5)
    try:
        counts = set()
        for seed in range(150, 180):
            r = set_episode_radius(env, parse_radius("0.25-0.50"), seed)
            obs, info = env.reset(seed=seed)
            counts.add(info["num_obstacles"])
            assert 0.25 <= r <= 0.50
            assert r == pytest.approx(env.unwrapped.robot.obstacle_radius)
            assert obs.shape == (57,)
        assert counts == {2, 3, 4, 5}
    finally:
        env.close()


def test_sb3_environment_check():
    env = FrameStackWrapper(GymNavigationEnv(2, "mixed"))
    try:
        check_env(env, warn=True)
    finally:
        env.close()
