"""Map-free analytical baselines using only goal bearing and 16 LiDAR ranges.

The original (untuned) controller is retained verbatim to reproduce the
published exploratory comparison. The revised controller adds a bounded
repulsive field, tangential obstacle bypass, and optional reverse motion.
Its parameters must be selected on a separate validation seed range before
making comparative claims against PPO.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def _recent(observation):
    data = np.asarray(observation, dtype=np.float32)
    if data.shape not in ((19,), (57,)):
        raise ValueError(f"Expected 19- or 57-element observation, got {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError("Non-finite observation")
    return data[-19:]


@dataclass(frozen=True)
class LegacyPotentialFieldController:
    """Original, untuned controller kept for historical result reproduction."""
    attraction: float = 1.0
    repulsion: float = 0.24
    influence_radius: float = 1.15
    turn_gain: float = 2.4
    forward_gain: float = 0.85
    slowdown_distance: float = 0.65

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        recent = _recent(observation)
        bearing = math.atan2(float(recent[1]), float(recent[2]))
        distances = recent[3:]
        direction_x = self.attraction * math.cos(bearing)
        direction_y = self.attraction * math.sin(bearing)
        for index, value in enumerate(distances):
            distance = float(value)
            if distance >= self.influence_radius:
                continue
            d = max(distance, 0.10)
            strength = self.repulsion * (1.0 / d - 1.0 / self.influence_radius) / (d * d)
            angle = -math.pi + index * (2.0 * math.pi / 16.0)
            direction_x -= strength * math.cos(angle)
            direction_y -= strength * math.sin(angle)
        desired_heading = math.atan2(direction_y, direction_x)
        alignment = max(0.0, math.cos(desired_heading))
        nearest = float(np.min(distances))
        clearance_factor = max(0.0, min(1.0, nearest / self.slowdown_distance))
        v = min(1.0, self.forward_gain * alignment * clearance_factor)
        w = max(-1.0, min(1.0, self.turn_gain * desired_heading / 1.5))
        return np.array([v, w], dtype=np.float32), None


@dataclass(frozen=True)
class PotentialFieldController:
    """Bounded attractive-repulsive field with tangential bypass and reverse.

    This is a heuristic local controller, not DWA. Default gains are illustrative,
    not automatically 'tuned'. Use `tune-baseline` on disjoint seeds, and pass
    the produced JSON file to `benchmark --baseline-config`.
    """
    attraction: float = 1.0
    repulsion: float = 0.16
    influence_radius: float = 1.35
    repulsion_cap: float = 1.3
    tangent_gain: float = 0.55
    turn_gain: float = 2.4
    forward_gain: float = 0.85
    slowdown_distance: float = 0.55
    allow_reverse: bool = True

    def __post_init__(self):
        for name in ("attraction", "repulsion", "influence_radius", "repulsion_cap", "turn_gain", "forward_gain", "slowdown_distance"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.tangent_gain) or self.tangent_gain < 0:
            raise ValueError("tangent_gain must be nonnegative")

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        recent = _recent(observation)
        bearing = math.atan2(float(recent[1]), float(recent[2]))
        goal_x, goal_y = math.cos(bearing), math.sin(bearing)
        fx, fy = self.attraction * goal_x, self.attraction * goal_y
        distances = recent[3:]
        for index, value in enumerate(distances):
            d = float(value)
            if d >= self.influence_radius:
                continue
            d = max(d, 0.08)
            strength = min(self.repulsion_cap,
                           self.repulsion * (1.0 / d - 1.0 / self.influence_radius) / (d * d))
            angle = -math.pi + index * math.tau / 16.0
            ox, oy = math.cos(angle), math.sin(angle)
            fx -= strength * ox
            fy -= strength * oy
            # Prefer passing on the side toward the goal; break exact ties
            # counter-clockwise, deterministically, to avoid a zero vector.
            cross = ox * goal_y - oy * goal_x
            side = 1.0 if cross >= -0.05 else -1.0
            fx += self.tangent_gain * strength * (-oy * side)
            fy += self.tangent_gain * strength * ( ox * side)

        desired = math.atan2(fy, fx)
        reversed_motion = self.allow_reverse and abs(desired) > math.pi / 2
        heading_error = math.atan2(math.sin(desired - (math.pi if reversed_motion else 0)),
                                   math.cos(desired - (math.pi if reversed_motion else 0)))
        alignment = max(0.0, math.cos(heading_error))
        nearest = float(np.min(distances))
        slowdown = max(0.15, min(1.0, nearest / self.slowdown_distance))
        signed_v = (-1.0 if reversed_motion else 1.0) * min(1.0, self.forward_gain * alignment * slowdown)
        w = float(np.clip(self.turn_gain * heading_error / 1.5, -1.0, 1.0))
        return np.array([signed_v, w], dtype=np.float32), None


def load_baseline_config(path: str | Path) -> PotentialFieldController:
    """Read tuner output with explicit schema, rejecting unrecognized values."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("kind") != "validated_potential_field_v2":
        raise ValueError("Expected a JSON config generated by `tune-baseline`")
    allowed = set(asdict(PotentialFieldController()))
    params = data.get("parameters")
    if not isinstance(params, dict) or set(params) != allowed:
        raise ValueError("Unexpected baseline parameters")
    return PotentialFieldController(**params)
