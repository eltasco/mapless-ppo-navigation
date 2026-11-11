"""Shared deterministic episode loop for eval, paired benchmarking and tuning."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EpisodeMetrics:
    outcome: str
    steps: int
    reward: float
    path_length_m: float
    min_clearance_m: float
    low_speed_fraction: float
    velocity_sign_flips: int
    end_goal_distance_m: float
    radius_m: float


def run_episode(env, controller, seed: int, radius, *, trace: bool = False):
    """Run one deterministic world seed and return metrics plus optional trace.

    `radius` must be the result of parse_radius(). Policies are reset between
    episodes when they expose a reset() method, preventing state carry-over.
    """
    from .scenarios import set_episode_radius

    if hasattr(controller, "reset"):
        controller.reset()
    episode_radius = set_episode_radius(env, radius, seed)
    obs, _ = env.reset(seed=seed)
    expected_count = env.unwrapped.num_obstacles
    initial_obstacles = tuple(env.robot.obstacles)
    if len(initial_obstacles) != expected_count:
        raise RuntimeError("Obstacle-count mismatch at episode reset")
    reward_sum = path_length = 0.0
    min_clearance = math.inf
    low_speed_steps = flips = 0
    prev_v = None
    trace_rows = []
    while True:
        action, _ = controller.predict(obs, deterministic=True)
        x0, y0 = env.robot.x, env.robot.y
        obs, reward, terminated, truncated, info = env.step(action)
        v, w = float(action[0]), float(action[1])
        reward_sum += float(reward)
        path_length += math.hypot(env.robot.x - x0, env.robot.y - y0)
        min_clearance = min(min_clearance, float(info["clearance"]))
        low_speed_steps += abs(v) < 0.10
        if prev_v is not None and prev_v * v < -0.04:
            flips += 1
        prev_v = v
        if trace:
            trace_rows.append({
                "seed": seed, "step": env.robot.steps,
                "x_m": round(env.robot.x, 6), "y_m": round(env.robot.y, 6),
                "yaw_rad": round(env.robot.theta, 6),
                "linear_action": round(v, 6), "angular_action": round(w, 6),
                "goal_distance_m": round(env.robot.get_observation()[0], 6),
                "clearance_m": round(float(info["clearance"]), 6),
            })
        if terminated or truncated:
            break
    if tuple(env.robot.obstacles) != initial_obstacles:
        raise RuntimeError("Obstacle configuration mutated during rollout")
    outcome = "success" if info["goal_reached"] else "collision" if info["collision"] else "timeout"
    nsteps = env.robot.steps
    return EpisodeMetrics(
        outcome=outcome, steps=nsteps, reward=reward_sum,
        path_length_m=path_length, min_clearance_m=min_clearance,
        low_speed_fraction=low_speed_steps / nsteps,
        velocity_sign_flips=flips, end_goal_distance_m=env.robot.get_observation()[0],
        radius_m=episode_radius,
    ), trace_rows
