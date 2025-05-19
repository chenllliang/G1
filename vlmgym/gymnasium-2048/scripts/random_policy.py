import argparse

import gymnasium as gym
from tqdm import trange


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        default="gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0",
        help="environment id",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random generator seed",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=100,
        help="number of timesteps",
    )
    args = parser.parse_args()
    return args


def random_policy() -> None:
    args = parse_args()

    env = gym.make(args.env, render_mode="human")

    env.reset(seed=args.seed)

    for _ in trange(args.n_timesteps, desc="Random policy"):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            env.reset()

    env.close()


if __name__ == "__main__":
    random_policy()
