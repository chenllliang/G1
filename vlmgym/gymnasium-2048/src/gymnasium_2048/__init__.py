from gymnasium.envs.registration import register

register(
    id="gymnasium_2048/TwentyFortyEight-v0",
    entry_point="gymnasium_2048.envs:TwentyFortyEightEnv",
)
