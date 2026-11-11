"""Physics, geometry and reward sanity checks."""

import math

import pytest

pytest.importorskip("gymnasium")

from mapless_ppo.navigation import NavigationEnv


def test_forward_lidar_ray_hits_circle_surface():
    robot = NavigationEnv()
    robot.obstacles = [(1.0, 0.0)]
    # The 9th ray (index 8) points along the robot's forward axis.
    assert robot.get_lidar()[8] == pytest.approx(0.65)
    assert robot.get_lidar()[0] == pytest.approx(robot.sensor_range)


def test_360_degree_sensor_and_collision():
    robot = NavigationEnv()
    assert len(robot.sensor_angles) == 16
    assert robot.sensor_angles[0] == pytest.approx(-math.pi)
    robot.obstacles = [(0.55, 0.0)]
    assert robot.check_collision() is True
    robot.obstacles = [(0.56, 0.0)]
    assert robot.check_collision() is False


def test_action_limits_and_termination():
    robot = NavigationEnv()
    robot.goal_x = 10.0
    robot.goal_y = 0.0
    robot.obstacles = [(5.0, 5.0)]
    robot.step((99.0, 0.0))
    assert robot.x == pytest.approx(robot.max_linear_speed * robot.dt)
    assert robot.steps == 1
    robot.max_steps = 2
    _, _, terminated, truncated, _ = robot.step((0.0, 0.0))
    assert terminated is False
    assert truncated is True
    with pytest.raises(RuntimeError):
        robot.step((0.0, 0.0))


def test_contact_penalty_and_goal_reward():
    robot = NavigationEnv()
    robot.obstacles = [(0.56, 0.0)]
    robot.goal_x, robot.goal_y = 5.0, 0.0
    _, reward, terminated, truncated, info = robot.step((0.6, 0.0))
    assert terminated and not truncated
    assert info["collision"] and reward < -9.0

    robot.reset()
    robot.obstacles = [(5.0, 5.0)]
    robot.goal_x, robot.goal_y = 0.25, 0.0
    _, reward, terminated, _, info = robot.step((0.0, 0.0))
    assert terminated and info["goal_reached"] and reward > 9.0
