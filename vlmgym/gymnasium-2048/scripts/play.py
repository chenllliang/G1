import argparse
import logging

import gymnasium as gym
import pygame

logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

FPS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a 2048 game manually",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        default="gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0",
        help="environment id",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=4,
        help="game board size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random generator seed",
    )
    args = parser.parse_args()
    return args


def play() -> None:
    args = parse_args()

    logger.info("play game %s with size %d", args.env, args.size)
    env = gym.make(args.env, size=args.size, render_mode="human")

    _, info = env.reset(seed=args.seed)
    terminated = truncated = False

    while not terminated and not truncated:
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                truncated = True

        keys_pressed = pygame.key.get_pressed()

        if keys_pressed[pygame.K_UP]:
            action = 0
        elif keys_pressed[pygame.K_RIGHT]:
            action = 1
        elif keys_pressed[pygame.K_DOWN]:
            action = 2
        elif keys_pressed[pygame.K_LEFT]:
            action = 3

        if action is not None:
            logger.info("action: %d", action)
            _, _, terminated, truncated, info = env.step(action)

        env.unwrapped.clock.tick(FPS)

    env.close()

    logger.info("game over")
    logger.info("score: %d", info["total_score"])
    logger.info("max: %d", info["max"])


if __name__ == "__main__":
    play()
