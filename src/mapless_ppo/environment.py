"""Gymnasium environment with seeded, variable-complexity obstacle layouts.

Sampling order is deliberately retained at the default obstacle radius (0.35m)
to make the original evaluation seeds meaningful for saved models.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .navigation import NavigationEnv

STAGES = ("easy", "medium", "medium_hard", "hard", "hard_passable", "mixed")


class GymNavigationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, num_obstacles: int = 1, curriculum_stage: str = "hard", *, risk_weight: float = 1.5):
        super().__init__()
        if curriculum_stage not in STAGES:
            raise ValueError(f"Unknown difficulty: {curriculum_stage!r}")
        if not isinstance(num_obstacles, int) or num_obstacles < 1:
            raise ValueError("num_obstacles must be a positive integer")
        self.num_obstacles = num_obstacles
        self.curriculum_stage = curriculum_stage
        self.robot = NavigationEnv(risk_weight=risk_weight)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        count = len(self.robot.sensor_angles)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0] + [0.0] * count, dtype=np.float32),
            high=np.array(
                [np.inf, 1.0, 1.0] + [self.robot.sensor_range] * count,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        robot = self.robot
        robot.reset()

        goal_distance = self.np_random.uniform(2.0, 3.0)
        direction = self.np_random.uniform(-math.pi, math.pi)
        robot.goal_x = robot.x + goal_distance * math.cos(direction)
        robot.goal_y = robot.y + goal_distance * math.sin(direction)

        side = float(self.np_random.choice([-1.0, 1.0]))
        if self.np_random.random() < 0.70:
            offset = self.np_random.uniform(0.4, 0.55)
        else:
            offset = self.np_random.uniform(0.55, 0.8)
        robot.place_obstacle(offset, side)
        first_x, first_y = robot.obstacle_x, robot.obstacle_y

        if self.num_obstacles == 1:
            robot.obstacles = [(first_x, first_y)]
            return self._get_observation(), {}

        dx, dy = robot.goal_x - robot.x, robot.goal_y - robot.y
        length = math.hypot(dx, dy)
        px, py = -dy / length, dx / length
        stage = self.curriculum_stage
        if stage == "mixed":
            stage = str(
                self.np_random.choice(
                    ["easy", "medium", "medium_hard", "hard"],
                    p=[0.15, 0.25, 0.30, 0.30],
                )
            )

        # Obstacle 2: source sampling distributions, source rejection rules.
        for _ in range(1000):
            if stage == "easy":
                t = self.np_random.uniform(0.60, 0.85)
                lateral = self.np_random.choice([-1.0, 1.0]) * self.np_random.uniform(1.15, 1.50)
            elif stage == "medium":
                t = self.np_random.uniform(0.45, 0.85)
                lateral = self.np_random.choice([-1.0, 1.0]) * self.np_random.uniform(0.75, 1.05)
            elif stage == "medium_hard":
                t = self.np_random.uniform(0.40, 0.85)
                lateral = self.np_random.choice([-1.0, 1.0]) * self.np_random.uniform(0.55, 0.90)
            else:  # hard, hard_passable
                t = self.np_random.uniform(0.35, 0.85)
                extra_radius = max(0.0, robot.obstacle_radius - 0.35)
                lateral_limit = 0.65 + 5.0 * extra_radius
                lateral = self.np_random.uniform(-lateral_limit, lateral_limit)

            x2 = robot.x + t * dx + lateral * px
            y2 = robot.y + t * dy + lateral * py
            separation = math.hypot(x2 - first_x, y2 - first_y)
            gap = separation - 2 * robot.obstacle_radius
            start_dist = math.hypot(x2 - robot.x, y2 - robot.y)
            goal_dist = math.hypot(x2 - robot.goal_x, y2 - robot.goal_y)
            minimum_gap = 2 * robot.robot_radius + 0.10 if stage == "hard_passable" else 0.10
            if (
                gap >= minimum_gap
                and start_dist >= robot.robot_radius + robot.obstacle_radius + 0.15
                and goal_dist >= robot.obstacle_radius + robot.goal_radius + 0.10
            ):
                break
        else:
            raise RuntimeError(
                f"Unable to place obstacle 2 (stage={stage}, radius={robot.obstacle_radius:.2f})"
            )

        robot.obstacles = [(first_x, first_y), (float(x2), float(y2))]
        self._complete_obstacles()
        return self._get_observation(), {}

    def _complete_obstacles(self) -> None:
        robot = self.robot
        sx, sy = robot.x, robot.y
        dx, dy = robot.goal_x - sx, robot.goal_y - sy
        distance = math.hypot(dx, dy)
        if distance < 1e-8:
            raise ValueError("Goal too close to start")
        px, py = -dy / distance, dx / distance
        minimum_separation = 2 * robot.obstacle_radius + 0.15
        extra_radius = max(0.0, robot.obstacle_radius - 0.35)

        while len(robot.obstacles) < self.num_obstacles:
            obstacle_number = len(robot.obstacles) + 1
            for attempt in range(5000):
                # When radius <= 0.35 this is exactly the supplied source range.
                lateral_range = (
                    1.10 + 0.30 * (obstacle_number - 3)
                    + (5.0 + 10.0 * attempt / 5000) * extra_radius
                )
                t = self.np_random.uniform(0.25, 0.90)
                lateral = self.np_random.uniform(-lateral_range, lateral_range)
                ox = sx + t * dx + lateral * px
                oy = sy + t * dy + lateral * py
                if any(math.hypot(ox - x, oy - y) < minimum_separation for x, y in robot.obstacles):
                    continue
                if math.hypot(ox - sx, oy - sy) < robot.robot_radius + robot.obstacle_radius + 0.15:
                    continue
                if math.hypot(ox - robot.goal_x, oy - robot.goal_y) < robot.goal_radius + robot.obstacle_radius + 0.10:
                    continue
                robot.obstacles.append((float(ox), float(oy)))
                break
            else:
                raise RuntimeError(
                    f"Unable to place obstacle {obstacle_number} (radius={robot.obstacle_radius:.2f})"
                )

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if action.shape != (2,):
            raise ValueError("Expected action shape (2,)")
        velocity = float(action[0]) * self.robot.max_linear_speed
        omega = float(action[1]) * self.robot.max_angular_speed
        _, reward, terminated, truncated, info = self.robot.step((velocity, omega))
        return self._get_observation(), float(reward), bool(terminated), bool(truncated), info

    def _get_observation(self) -> np.ndarray:
        distance, bearing = self.robot.get_observation()
        return np.array(
            [distance, np.sin(bearing), np.cos(bearing)] + self.robot.get_lidar(),
            dtype=np.float32,
        )
