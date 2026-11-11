"""Measurement-only and geometry-shift tests, without altering PPO weights.

The default training/evaluation distribution is kept unchanged. These shifts
are intentionally separate and should not be mistaken for real-world validation.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np


class StressWorld(gym.Wrapper):
    """Modify *observations* and optional obstacle layout consistently per seed.

    Layout 'uniform' independently samples disk obstacles along a wider strip
    rather than anchoring the first disk at the start-to-goal midpoint.
    'long_goal' also moves the goal farther away (3.5–5.0 m).
    Noise/dropout affect only LiDAR readings; physical collision checks remain
    exact and unchanged, allowing measurement robustness to be isolated.
    """

    VALID_LAYOUTS = ("original", "uniform", "long_goal")

    def __init__(
        self, env: gym.Env, *, layout: str = "original", noise_std: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(env)
        if layout not in self.VALID_LAYOUTS:
            raise ValueError(f"Unknown layout: {layout}")
        if not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        if not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        self.layout = layout
        self.noise_std = noise_std
        self.dropout = dropout
        self.robot = env.unwrapped.robot  # Expected by the frame-history wrapper.
        self.rng = np.random.default_rng()

    def _corrupt(self, observation: np.ndarray) -> np.ndarray:
        if self.noise_std == 0 and self.dropout == 0:
            return observation
        observation = observation.copy()
        rays = observation[3:19]
        if self.noise_std:
            rays += self.rng.normal(0.0, self.noise_std, rays.shape).astype(np.float32)
            np.clip(rays, 0.0, self.robot.sensor_range, out=rays)
        if self.dropout:
            rays[self.rng.random(len(rays)) < self.dropout] = self.robot.sensor_range
        return observation

    def _resample_obstacles(self) -> None:
        robot = self.robot
        count = self.env.unwrapped.num_obstacles
        sx, sy = robot.x, robot.y
        dx, dy = robot.goal_x - sx, robot.goal_y - sy
        length = math.hypot(dx, dy)
        nx, ny = -dy / length, dx / length
        chosen = []
        for _ in range(count):
            for _attempt in range(10000):
                t = self.rng.uniform(0.15, 0.95)
                lateral = self.rng.uniform(-1.8, 1.8)
                ox = sx + t * dx + lateral * nx
                oy = sy + t * dy + lateral * ny
                if math.hypot(ox - sx, oy - sy) < robot.robot_radius + robot.obstacle_radius + 0.15:
                    continue
                if math.hypot(ox - robot.goal_x, oy - robot.goal_y) < robot.goal_radius + robot.obstacle_radius + 0.10:
                    continue
                if any(math.hypot(ox - x, oy - y) < 2.0 * robot.obstacle_radius + 0.15 for x, y in chosen):
                    continue
                chosen.append((float(ox), float(oy)))
                break
            else:
                raise RuntimeError("Could not sample a valid out-of-distribution obstacle layout")
        robot.obstacles = chosen

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed + 1_073_823)
        observation, info = self.env.reset(seed=seed, options=options)
        if self.layout != "original":
            if self.layout == "long_goal":
                r = self.rng.uniform(3.5, 5.0)
                theta = math.atan2(self.robot.goal_y - self.robot.y, self.robot.goal_x - self.robot.x)
                self.robot.goal_x = self.robot.x + r * math.cos(theta)
                self.robot.goal_y = self.robot.y + r * math.sin(theta)
            self._resample_obstacles()
            observation = self.env.unwrapped._get_observation()
        return self._corrupt(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._corrupt(observation), reward, terminated, truncated, info
