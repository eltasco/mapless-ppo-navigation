"""Map-free analytical controllers for *paired* comparison against PPO.

This is a simple artificial potential-field controller inspired by Khatib
(1986), not Dynamic Window Approach (DWA) and not an optimized controller.
It uses exactly the same 19-element goal/laser observation as the policy,
without privileged obstacle coordinates or a map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PotentialFieldController:
    """Reactive attraction-to-goal plus range-dependent obstacle repulsion.

    The controller is parameter-fixed (not tuned using benchmark test seeds).
    Local minima and oscillation are expected limitations of potential fields.
    """

    attraction: float = 1.0
    repulsion: float = 0.24
    influence_radius: float = 1.15
    turn_gain: float = 2.4
    forward_gain: float = 0.85
    slowdown_distance: float = 0.65

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Return a normalized [linear_speed, angular_speed] command."""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape not in ((19,), (57,)):
            raise ValueError(f"Expected one 19-D or stacked 57-D observation, got {obs.shape}")
        # The final frame is always the most recent observation.
        recent = obs[-19:]
        bearing = math.atan2(float(recent[1]), float(recent[2]))
        distances = recent[3:]
        goal_x = self.attraction * math.cos(bearing)
        goal_y = self.attraction * math.sin(bearing)
        direction_x, direction_y = goal_x, goal_y

        for index, value in enumerate(distances):
            distance = float(value)
            if distance >= self.influence_radius:
                continue
            # Finite repulsive influence; avoid division by zero at contact.
            d = max(distance, 0.10)
            strength = self.repulsion * (1.0 / d - 1.0 / self.influence_radius) / (d * d)
            angle = -math.pi + index * (2.0 * math.pi / 16.0)
            direction_x -= strength * math.cos(angle)
            direction_y -= strength * math.sin(angle)

        desired_heading = math.atan2(direction_y, direction_x)
        # Only drive when roughly aligned; rotate in place in crowded scenes.
        alignment = max(0.0, math.cos(desired_heading))
        nearest = float(np.min(distances))
        clearance_factor = max(0.0, min(1.0, nearest / self.slowdown_distance))
        v = min(1.0, self.forward_gain * alignment * clearance_factor)
        w = max(-1.0, min(1.0, self.turn_gain * desired_heading / 1.5))
        return np.array([v, w], dtype=np.float32), None
