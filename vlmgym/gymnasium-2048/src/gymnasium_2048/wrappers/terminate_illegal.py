from typing import Any

import gymnasium as gym
from gymnasium.core import (
    ActType,
    Env,
    ObsType,
    SupportsFloat,
    WrapperActType,
    WrapperObsType,
)


class TerminateIllegalWrapper(gym.Wrapper):
    def __init__(self, env: Env[ObsType, ActType], illegal_reward: float) -> None:
        super().__init__(env=env)
        self._illegal_reward = illegal_reward

    def step(
        self,
        action: WrapperActType,
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        if info["is_legal"]:
            return observation, reward, terminated, truncated, info
        return observation, self._illegal_reward, True, truncated, info
