# pylint: disable=protected-access,redefined-outer-name
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from gymnasium_2048.envs import TwentyFortyEightEnv


@pytest.fixture
def env() -> TwentyFortyEightEnv:
    return TwentyFortyEightEnv()


def test_check_env():
    env = gym.make("gymnasium_2048/TwentyFortyEight-v0")
    check_env(env=env.unwrapped)


@pytest.mark.parametrize(
    "test_input,expected",
    (
        (
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 2, 0, 0],
                ],
                dtype=np.uint8,
            ),
            {
                (0, 0, 0),
                (0, 1, 0),
                (0, 2, 0),
                (0, 3, 0),
                (1, 0, 0),
                (1, 1, 0),
                (1, 2, 0),
                (1, 3, 1),
                (2, 0, 0),
                (2, 1, 0),
                (2, 2, 0),
                (2, 3, 0),
                (3, 0, 0),
                (3, 1, 2),
                (3, 2, 0),
                (3, 3, 0),
            },
        ),
        (
            np.array(
                [
                    [3, 0, 2, 1],
                    [7, 9, 8, 5],
                    [4, 10, 11, 13],
                    [15, 14, 6, 12],
                ],
                dtype=np.uint8,
            ),
            {
                (0, 0, 3),
                (0, 1, 0),
                (0, 2, 2),
                (0, 3, 1),
                (1, 0, 7),
                (1, 1, 9),
                (1, 2, 8),
                (1, 3, 5),
                (2, 0, 4),
                (2, 1, 10),
                (2, 2, 11),
                (2, 3, 13),
                (3, 0, 15),
                (3, 1, 14),
                (3, 2, 6),
                (3, 3, 12),
            },
        ),
    ),
)
def test_get_obs(
    env: TwentyFortyEightEnv,
    test_input: np.ndarray,
    expected: set[tuple[int, int, int]],
):
    # Given
    env.reset()
    env.board = test_input

    # When
    observation = env._get_obs()

    # Then
    assert observation.shape == (4, 4, 16)
    for row in range(4):
        for col in range(4):
            for value in range(16):
                if (row, col, value) in expected:
                    assert observation[row, col, value] == 1
                else:
                    assert observation[row, col, value] == 0


def test_reset(env: TwentyFortyEightEnv):
    # Given / When
    env.reset()

    # Then
    assert hasattr(env, "board")
    assert env.board.shape == (4, 4)
    assert (env.board != 0).sum() == 2
    assert (env.board == 0).sum() == 14


@patch.object(TwentyFortyEightEnv, "_spawn_tile", MagicMock())
@pytest.mark.parametrize(
    "test_input,expected",
    (
        (
            (
                np.array(
                    [
                        [2, 2, 2, 0],
                        [1, 0, 1, 1],
                        [2, 2, 0, 2],
                        [0, 1, 1, 2],
                    ],
                    dtype=np.uint8,
                ),
                0,
            ),
            (
                np.array(
                    [
                        [2, 3, 2, 1],
                        [1, 1, 2, 3],
                        [2, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    dtype=np.uint8,
                ),
                20,
            ),
        ),
        (
            (
                np.array(
                    [
                        [2, 2, 2, 0],
                        [1, 0, 1, 1],
                        [2, 2, 0, 2],
                        [0, 1, 1, 2],
                    ],
                    dtype=np.uint8,
                ),
                1,
            ),
            (
                np.array(
                    [
                        [0, 0, 2, 3],
                        [0, 0, 1, 2],
                        [0, 0, 2, 3],
                        [0, 0, 2, 2],
                    ],
                    dtype=np.uint8,
                ),
                24,
            ),
        ),
        (
            (
                np.array(
                    [
                        [2, 2, 2, 0],
                        [1, 0, 1, 1],
                        [2, 2, 0, 2],
                        [0, 1, 1, 2],
                    ],
                    dtype=np.uint8,
                ),
                2,
            ),
            (
                np.array(
                    [
                        [0, 0, 0, 0],
                        [2, 0, 0, 0],
                        [1, 3, 2, 1],
                        [2, 1, 2, 3],
                    ],
                    dtype=np.uint8,
                ),
                20,
            ),
        ),
        (
            (
                np.array(
                    [
                        [2, 2, 2, 0],
                        [1, 0, 1, 1],
                        [2, 2, 0, 2],
                        [0, 1, 1, 2],
                    ],
                    dtype=np.uint8,
                ),
                3,
            ),
            (
                np.array(
                    [
                        [3, 2, 0, 0],
                        [2, 1, 0, 0],
                        [3, 2, 0, 0],
                        [2, 2, 0, 0],
                    ],
                    dtype=np.uint8,
                ),
                24,
            ),
        ),
    ),
)
def test_step(
    env: TwentyFortyEightEnv,
    test_input: tuple[np.ndarray, int],
    expected: tuple[np.ndarray, int],
):
    # Given
    env.reset()
    env.board, action = test_input

    # When
    _, reward, terminated, truncated, info = env.step(action)

    # Then
    assert np.array_equal(env.board, expected[0])
    assert reward == expected[1]
    assert not terminated
    assert not truncated
    assert isinstance(info, dict)
