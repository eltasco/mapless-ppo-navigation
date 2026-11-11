"""Optional fixed feature scaling for *newly trained* policies only.

The shipped 57D PPO checkpoint expects raw distances. Scaling it at inference
would silently invalidate the published architecture's learned input contract.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FixedObservationScale(gym.ObservationWrapper):
    """Divide goal range by 5 m and LiDAR distances by sensor max range.

    Bearing sine/cosine remain in [-1, 1]. No learned running statistics are
    required, so new checkpoints remain portable with the scale setting.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.robot = env.unwrapped.robot
        self._scale = np.array([5.0, 1.0, 1.0] + [self.robot.sensor_range] * 16, dtype=np.float32)
        raw = env.observation_space
        self.observation_space = spaces.Box(low=raw.low / self._scale,
                                            high=raw.high / self._scale, dtype=np.float32)

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32) / self._scale
