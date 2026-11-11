"""Episode-level domain randomization of obstacle count and radius.

Episode randomization supports obstacle-count curricula and controlled radius sweeps without changing policy input dimensions.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class VariableObstacleCount(gym.Wrapper):
    """Uniformly sample an integer obstacle count before each environment reset."""

    def __init__(self, env: gym.Env, min_obstacles: int = 2, max_obstacles: int = 5):
        super().__init__(env)
        if min_obstacles < 1 or max_obstacles < min_obstacles:
            raise ValueError("Invalid obstacle-count interval")
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self._rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        n = int(self._rng.integers(self.min_obstacles, self.max_obstacles + 1))
        self.unwrapped.num_obstacles = n
        observation, info = self.env.reset(seed=seed, options=options)
        if len(self.unwrapped.robot.obstacles) != n:
            raise RuntimeError("Obstacle generator returned an unexpected count")
        return observation, {**info, "num_obstacles": n}

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            observation, reward, terminated, truncated,
            {**info, "num_obstacles": self.unwrapped.num_obstacles},
        )


def parse_radius(spec: str | None):
    """Return None, fixed radius float, or (min,max) for per-episode sampling."""
    if spec is None or str(spec).lower() in {"original", "default"}:
        return None
    text = str(spec)
    if "-" in text:
        bits = text.split("-")
        if len(bits) != 2:
            raise ValueError(f"Invalid radius interval: {text}")
        lo, hi = map(float, bits)
        if not 0.05 <= lo <= hi <= 1.0:
            raise ValueError("Radius must be in [0.05, 1.0] m")
        return lo, hi
    radius = float(text)
    if not 0.05 <= radius <= 1.0:
        raise ValueError("Radius must be in [0.05, 1.0] m")
    return radius


def set_episode_radius(env, spec, seed: int) -> float:
    """Choose episode radius using an RNG independent of geometry RNG."""
    if spec is None:
        return float(env.unwrapped.robot.obstacle_radius)
    radius = (
        float(np.random.default_rng(seed + 89173).uniform(*spec))
        if isinstance(spec, tuple) else float(spec)
    )
    env.unwrapped.robot.obstacle_radius = radius
    return radius
