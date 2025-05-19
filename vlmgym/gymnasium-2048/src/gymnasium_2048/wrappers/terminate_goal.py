from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import (
    ActType,
    Env,
    ObsType,
    SupportsFloat,
    WrapperActType,
    WrapperObsType,
)


class TerminateGoalWrapper(gym.Wrapper):
    def __init__(self, env: Env[ObsType, ActType], goal: int = 2048) -> None:
        super().__init__(env=env)
        log_goal = np.log2(goal)
        assert log_goal.is_integer() and (
            2 < log_goal < 256
        ), "goal must be 0 or a power of 2 and 4 < goal < 2^256"
        self._goal = int(log_goal)

    def step(
        self,
        action: WrapperActType,
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        terminated = terminated or info["max"] == self._goal
        return observation, reward, terminated, truncated, info
