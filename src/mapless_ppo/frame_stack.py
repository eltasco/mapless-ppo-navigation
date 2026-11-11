"""Fixed-lag frame-history observations for a memoryless PPO MLP policy.

The fixed frame offsets preserve the exact 57-element observation contract of the published policy; changing their order would invalidate checkpoint compatibility.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FrameStackWrapper(gym.Wrapper):
    """Concatenate observations at t-20, t-10 and t.

    Source observations are 19-D [distance, sin(bearing), cos(bearing),
    16 LiDAR distances]. The resulting 57-D vector matches the original PPO
    checkpoints. Initial frames are padded with the first observation.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.robot = env.robot  # Convenient read-only access for visualization.
        self.history: deque[np.ndarray] = deque(maxlen=21)
        space = env.observation_space
        self.observation_space = spaces.Box(
            low=np.tile(space.low, 3),
            high=np.tile(space.high, 3),
            dtype=np.float32,
        )

    def _stack(self) -> np.ndarray:
        return np.concatenate(
            (self.history[-21], self.history[-11], self.history[-1])
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.history.clear()
        self.history.extend(observation.copy() for _ in range(21))
        return self._stack(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(observation.copy())
        return self._stack(), reward, terminated, truncated, info
