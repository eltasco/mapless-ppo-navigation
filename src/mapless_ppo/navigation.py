"""Deterministic 2-D kinematic robot, circular LiDAR, and navigation reward.

This module intentionally preserves the original simulator's state transition,
observation ordering, and reward computation for compatibility with saved PPO
policies. It is not a dynamics simulator for a physical quadruped.
"""

from __future__ import annotations

import math
from typing import Sequence


class NavigationEnv:
    """A circular differential-drive *kinematic* navigation agent.

    Linear velocity is signed; the learned policy may reverse. Collision
    geometry is a disk approximation. Coordinates and distances are in meters.
    """

    def __init__(self) -> None:
        self.dt = 1.0 / 30.0
        self.max_steps = 900
        self.max_linear_speed = 0.6
        self.max_angular_speed = 1.5
        self.robot_radius = 0.20
        self.obstacle_radius = 0.35
        self.goal_x = 2.0
        self.goal_y = 1.5
        self.goal_radius = 0.25
        self.obstacle_x = 0.0  # Retained for the original first-obstacle generator.
        self.obstacle_y = 0.0
        self.sensor_range = 3.0
        self.sensor_angles = tuple(-math.pi + i * 2 * math.pi / 16 for i in range(16))
        self.obstacles: list[tuple[float, float]] = []
        self.reset()

    def reset(self) -> tuple[float, float]:
        """Reset the pose and episode state (goal/obstacle sampling is external)."""
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.steps = 0
        self.episode_done = False
        # Kept from the original implementation for checkpoint compatibility.
        self.obstacles = [(self.obstacle_x, self.obstacle_y)]
        return self.get_observation()

    def get_observation(self) -> tuple[float, float]:
        """Return goal distance and relative bearing in [-pi, pi]."""
        dx, dy = self.goal_x - self.x, self.goal_y - self.y
        distance = math.hypot(dx, dy)
        angle = (math.atan2(dy, dx) - self.theta + math.pi) % (2 * math.pi) - math.pi
        return distance, angle

    def place_obstacle(self, offset: float, side: float) -> None:
        """Locate obstacle 1 near the midpoint of the start-to-goal segment."""
        mid_x = (self.x + self.goal_x) / 2.0
        mid_y = (self.y + self.goal_y) / 2.0
        dx, dy = self.goal_x - self.x, self.goal_y - self.y
        distance = math.hypot(dx, dy)
        if distance == 0:
            raise ValueError("Start and goal must not coincide")
        self.obstacle_x = mid_x - side * offset * dy / distance
        self.obstacle_y = mid_y + side * offset * dx / distance

    def check_collision(self) -> bool:
        """Detect robot-disk intersections against all circular obstacles."""
        contact_distance = self.robot_radius + self.obstacle_radius
        return any(
            math.hypot(self.x - ox, self.y - oy) <= contact_distance
            for ox, oy in self.obstacles
        )

    def get_lidar(self) -> list[float]:
        """Ray-circle intersections, 16 rays spanning 360 degrees.

        Values are distances from the robot *center* to the nearest obstacle
        surface. A max-range measurement does not assert that the world is free.
        """
        readings = []
        radius = self.obstacle_radius
        radius_sq = radius * radius
        for relative_angle in self.sensor_angles:
            angle = self.theta + relative_angle
            ux, uy = math.cos(angle), math.sin(angle)
            nearest = self.sensor_range
            for ox, oy in self.obstacles:
                cx, cy = ox - self.x, oy - self.y
                center_sq = cx * cx + cy * cy
                if center_sq <= radius_sq:
                    nearest = 0.0
                    break
                projection = cx * ux + cy * uy
                perp_sq = max(0.0, center_sq - projection * projection)
                if perp_sq > radius_sq:
                    continue
                offset = math.sqrt(radius_sq - perp_sq)
                near, far = projection - offset, projection + offset
                if near >= 0.0:
                    nearest = min(nearest, near)
                elif far >= 0.0:
                    nearest = 0.0
            readings.append(nearest)
        return readings

    def step(
        self, action: Sequence[float]
    ) -> tuple[tuple[float, float], float, bool, bool, dict]:
        """Advance one 1/30-second step using physical (v, omega) commands.

        Original danger reward intentionally uses post-update heading and
        post-update obstacle distances. Keeping this convention preserves the
        exact scoring behavior of the supplied reference implementation.
        """
        if self.episode_done:
            raise RuntimeError("Episode is finished; call reset() first")
        v, omega = action
        v = max(-self.max_linear_speed, min(self.max_linear_speed, float(v)))
        omega = max(-self.max_angular_speed, min(self.max_angular_speed, float(omega)))
        previous_distance, _ = self.get_observation()

        # Kinematic transition. Translation uses the heading at the start of
        # the step; yaw is advanced afterwards, as in the original environment.
        self.x += v * math.cos(self.theta) * self.dt
        self.y += v * math.sin(self.theta) * self.dt
        self.theta = (self.theta + omega * self.dt + math.pi) % (2 * math.pi) - math.pi
        self.steps += 1

        goal_distance, relative_bearing = self.get_observation()
        collision = self.check_collision()
        goal_reached = goal_distance <= self.goal_radius and not collision
        terminated = collision or goal_reached
        truncated = self.steps >= self.max_steps and not terminated

        # Reward components: goal progress, time, proximity, closing speed.
        reward = previous_distance - goal_distance - 0.1 * self.dt
        safe_margin = 0.35
        center_distance = min(
            math.hypot(self.x - ox, self.y - oy) for ox, oy in self.obstacles
        )
        clearance = center_distance - self.robot_radius - self.obstacle_radius
        proximity = max(0.0, min(1.0, (safe_margin - clearance) / safe_margin))
        reward -= 0.8 * proximity * self.dt

        # The original scoring used the *updated* yaw to project velocity.
        # This is part of the compatibility contract of the trained models.
        vx, vy = v * math.cos(self.theta), v * math.sin(self.theta)
        maximum_danger = 0.0
        for ox, oy in self.obstacles:
            dx, dy = ox - self.x, oy - self.y
            distance = math.hypot(dx, dy)
            obstacle_clearance = distance - self.robot_radius - self.obstacle_radius
            near_obstacle = max(
                0.0, min(1.0, (safe_margin - obstacle_clearance) / safe_margin)
            )
            closing_speed = (vx * dx + vy * dy) / max(distance, 1e-8)
            approach_speed = max(0.0, closing_speed)
            danger = near_obstacle * (approach_speed / self.max_linear_speed) ** 2
            maximum_danger = max(maximum_danger, danger)
        reward -= 1.5 * maximum_danger * self.dt

        if collision:
            reward -= 10.0
        elif goal_reached:
            reward += 10.0

        self.episode_done = terminated or truncated
        info = {
            "goal_reached": goal_reached,
            "collision": collision,
            "steps": self.steps,
            "clearance": clearance,
        }
        return (goal_distance, relative_bearing), reward, terminated, truncated, info
