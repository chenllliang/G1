from typing import Sequence

import numpy as np


def get_all_straight_tuples(state: np.ndarray) -> Sequence[Sequence[int]]:
    """
    Gets all horizontal and vertical straight tuples.

    :param state: The board state.
    :return: The horizontal and vertical straight tuples.
    """
    tuples = []

    # horizontal
    for row in range(state.shape[0]):
        tuples.append(tuple(state[row, :]))

    # vertical
    for col in range(state.shape[1]):
        tuples.append(tuple(state[:, col]))

    return tuples


def get_all_rectangles_tuples(state: np.ndarray) -> Sequence[Sequence[int]]:
    """
    Gets all 2x2 square tuples.

    :param state: The board state.
    :return: The 2x2 square tuples.
    """
    tuples = []

    # square
    for row in range(state.shape[0] - 1):
        for col in range(state.shape[1] - 1):
            tuples.append(
                (
                    state[row, col],
                    state[row, col + 1],
                    state[row + 1, col + 1],
                    state[row + 1, col],
                )
            )

    return tuples


def get_all_straight_3_tuples(state: np.ndarray) -> Sequence[Sequence[int]]:
    """
    Gets all horizontal and vertical straight 3-tuples.

    :param state: The board state.
    :return: The horizontal and vertical straight 3-tuples.
    """
    tuples = []

    # horizontal
    for row in range(state.shape[0]):
        for col in range(state.shape[1] - 2):
            tuples.append(tuple(state[row, col : col + 3]))

    # vertical
    for col in range(state.shape[1]):
        for row in range(state.shape[0] - 2):
            tuples.append(tuple(state[row : row + 3, col]))

    return tuples


def get_all_corners_3_tuples(state: np.ndarray) -> Sequence[Sequence[int]]:
    """
    Gets all 3-tuples in the four corners of the board.

    :param state: The board state.
    :return: The corners 3-tuples.
    """
    rows, cols = state.shape
    tuples = [
        (
            # top left
            state[0, 0],
            state[0, 1],
            state[1, 0],
        ),
        (
            # top right
            state[0, cols - 1],
            state[0, cols - 2],
            state[1, cols - 1],
        ),
        (
            # bottom left
            state[rows - 1, 0],
            state[rows - 1, 1],
            state[rows - 2, 0],
        ),
        (
            # bottom right
            state[rows - 1, cols - 1],
            state[rows - 1, cols - 2],
            state[rows - 2, cols - 1],
        ),
    ]
    return tuples
