import argparse
import logging
import os
from typing import Any

import gymnasium as gym
import numpy as np
from tqdm import trange

from gymnasium_2048.agents.ntuple import (
    NTupleNetworkBasePolicy,
    NTupleNetworkQLearningPolicy,
    NTupleNetworkTDPolicy,
    NTupleNetworkTDPolicySmall,
)

logging.basicConfig(
    filename="train.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 2048 agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        default="tdl",
        help="RL Algorithm",
        choices=["ql", "tdl", "tdl-small"],
    )
    parser.add_argument(
        "--env",
        default="gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0",
        help="environment id",
    )
    parser.add_argument(
        "-i",
        "--trained-agent",
        default="",
        help="path to a pretrained agent to continue training",
    )
    parser.add_argument(
        "-n",
        "--n-episodes",
        type=int,
        default=100_000,
        help="number of episodes",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
        help="evaluate the agent every n episodes (if negative, no evaluation)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=1000,
        help="number of episodes to use for evaluation",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=-1,
        help="save the model every n steps (if negative, no checkpoint)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random generator seed",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0025,
        help="learning rate",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        default="models",
        help="path to the directory containing output models",
    )
    args = parser.parse_args()
    return args


def make_policy(algo: str, trained_agent: str) -> NTupleNetworkBasePolicy:
    """
    Makes the policy to train.

    :param algo: Name of the algorithm.
    :param trained_agent: Path to a trained agent.
    :return: Policy.
    """
    algo_policy_map = {
        "ql": NTupleNetworkQLearningPolicy,
        "tdl": NTupleNetworkTDPolicy,
        "tdl-small": NTupleNetworkTDPolicySmall,
    }
    policy = algo_policy_map[algo]
    return policy.load(trained_agent) if trained_agent else policy()


def play_game(
    env: gym.Env,
    policy: NTupleNetworkBasePolicy,
    learn: bool = False,
    learning_rate: float = 0.0025,
) -> dict[str, Any]:
    """
    Plays a 2048 game.

    :param env: Game environment.
    :param policy: Policy to use.
    :param learn: True to enable learning, False otherwise. Default: False.
    :param learning_rate: Learning rate to use during training. Default: 0.0025.
    :return: Info at the end of the game.
    """
    _observation, info = env.reset()
    terminated = truncated = False

    while not terminated and not truncated:
        state = info["board"]
        action = policy.predict(state=state)
        _observation, reward, terminated, truncated, info = env.step(action)
        if learn:
            policy.learn(
                state=state,
                action=action,
                reward=float(reward),
                next_state=info["board"],
                learning_rate=learning_rate,
            )

    return info


def evaluate(
    env: gym.Env,
    policy: NTupleNetworkBasePolicy,
    eval_episodes: int,
) -> dict[str, Any]:
    """
    Evaluates the performance of the current policy.

    It returns the three following measures:

    - Winning rate: the fraction of games the agent won (i.e. reached a 2048-tile)
    - Mean score: the average number of points obtained during a game
    - Max tile: the value of the maximum tile obtained

    :param env: Game environment.
    :param policy: Policy to evaluate.
    :param eval_episodes: Number of games to play.
    :return: Performance measures.
    """
    winning_rate = 0
    total_score = 0
    max_tile = 0

    with trange(eval_episodes, desc="Evaluate", unit="episode", leave=False) as pbar:
        for _ in pbar:
            info = play_game(env=env, policy=policy)
            winning_rate += int(2 ** info["max"] >= 2048)
            total_score += info["total_score"]
            max_tile = max(max_tile, 2 ** info["max"])
            pbar.set_postfix({"max_tile": max_tile})

    return {
        "winning_rate": winning_rate / eval_episodes,
        "mean_score": total_score / eval_episodes,
        "max_tile": max_tile,
    }


def log_eval_metrics(episode: int, metrics: dict[str, Any]) -> None:
    """
    Logs the evaluation metrics for an episode.

    :param episode: Episode number.
    :param metrics: Performance measures.
    """
    logger.info(
        "episode %d: winning rate = %.2f, mean score = %.2f, max tile = %d",
        episode,
        metrics["winning_rate"],
        metrics["mean_score"],
        metrics["max_tile"],
    )


def save_best_policy(out_dir: str, policy: NTupleNetworkBasePolicy) -> None:
    """
    Saves the best policy.

    :param out_dir: Path to output directory.
    :param policy: Policy to save.
    """
    best_model_path = os.path.join(out_dir, "best_n_tuple_network_policy.zip")
    logger.info("new best model saved to %s", best_model_path)
    policy.save(path=best_model_path)


def save_checkpoint(
    episode: int,
    out_dir: str,
    policy: NTupleNetworkBasePolicy,
) -> None:
    """
    Saves a checkpoint.

    :param episode: Episode number.
    :param out_dir: Output directory.
    :param policy: Policy to save.
    """
    checkpoint_path = os.path.join(out_dir, f"checkpoint_episode_{episode}.zip")
    logger.info("checkpoint saved to %s", checkpoint_path)
    policy.save(path=checkpoint_path)


def train() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    env = gym.make(args.env)
    policy = make_policy(algo=args.algo, trained_agent=args.trained_agent)
    best_mean_score = 0
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info("start training n-tuple network")

    with trange(1, args.n_episodes + 1, desc="Train", unit="episode") as pbar:
        for e in pbar:
            play_game(
                env=env,
                policy=policy,
                learn=True,
                learning_rate=args.learning_rate,
            )

            if e % args.eval_freq == 0 or e == args.n_episodes:
                metrics = evaluate(
                    env=env,
                    policy=policy,
                    eval_episodes=args.eval_episodes,
                )
                log_eval_metrics(episode=e, metrics=metrics)

                if metrics["mean_score"] > best_mean_score:
                    best_mean_score = metrics["mean_score"]
                    save_best_policy(out_dir=args.out_dir, policy=policy)

                metrics["best_mean_score"] = best_mean_score
                pbar.set_postfix(metrics)

            if e % args.save_freq == 0 or e == args.n_episodes:
                save_checkpoint(episode=e, out_dir=args.out_dir, policy=policy)

    env.close()

    logger.info("end training n-tuple network")


if __name__ == "__main__":
    train()
