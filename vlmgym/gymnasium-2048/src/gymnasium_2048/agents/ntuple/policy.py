from __future__ import annotations

import io
import os
import pathlib
import tempfile
import zipfile
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from gymnasium_2048.agents.ntuple.factory import (
    get_all_corners_3_tuples,
    get_all_rectangles_tuples,
    get_all_straight_3_tuples,
    get_all_straight_tuples,
)
from gymnasium_2048.agents.ntuple.network import NTupleNetwork
from gymnasium_2048.envs import TwentyFortyEightEnv


class NTupleNetworkBasePolicy(ABC):
    """
    Base class for n-tuple network policies.
    """

    def __init__(self) -> None:
        self.net = self._make_network()

    @staticmethod
    def _make_network() -> NTupleNetwork:
        """
        Makes a network containing 17 4-tuples.

        :return: N-tuple network.
        """
        return NTupleNetwork(shapes=[(15, 15, 15, 15) for _ in range(17)])

    @staticmethod
    def _get_tuples(state: np.ndarray) -> Sequence[Sequence[int]]:
        """
        Gets the n-tuples from the current state.

        :param state: The board state.
        :return: The n-tuples.
        """
        return [
            *get_all_straight_tuples(state=state),
            *get_all_rectangles_tuples(state=state),
        ]

    @abstractmethod
    def evaluate(self, state: np.ndarray, action: int) -> float:
        """
        Returns the evaluation of the current state and chosen action.

        :param state: The board state.
        :param action: The chosen action.
        :return: The evaluation value.
        """

    @abstractmethod
    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        learning_rate: float,
    ) -> None:
        """
        Learns the evaluation function for one step.

        :param state: The current board state.
        :param action: The chosen action.
        :param reward: The obtained reward.
        :param next_state: The next board state.
        :param learning_rate: The learning rate to use for the update.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str | bytes | os.PathLike) -> NTupleNetworkBasePolicy:
        """
        Loads the n-tuple network policy from a zip-file.

        :param path: The path to the file (or a file-like).
        :return: New n-tuple network policy instance with loaded parameters.
        """

    def predict(self, state: np.ndarray) -> int:
        """
        Predicts the next action to play.

        :param state: The board state.
        :return: Next action to play.
        """
        return np.argmax(
            [self.evaluate(state=state, action=action) for action in range(4)]
        )

    @abstractmethod
    def save(self, path: str | pathlib.Path | io.BufferedIOBase) -> None:
        """
        Saves the weights of the n-tuple network in a zip-file.

        :param path: The path to the file where the weights should be saved.
        """


class NTupleNetworkQLearningPolicy(NTupleNetworkBasePolicy):
    """
    N-tuple network policy using Q-Learning.

    The evaluation function consists of four n-tuple networks that provide value
    functions for each of the possible game moves.
    """

    def __init__(self) -> None:  # pylint: disable=super-init-not-called
        self.nets = [self._make_network() for _ in range(4)]

    def evaluate(self, state: np.ndarray, action: int) -> float:
        _after_state, _reward, is_legal = TwentyFortyEightEnv.apply_action(
            board=state,
            action=action,
        )

        if not is_legal:
            return -np.inf

        tuples = self._get_tuples(state=state)
        return self.nets[action].predict(tuples=tuples)

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        learning_rate: float,
    ) -> None:
        state_tuples = self._get_tuples(state=state)
        state_value = self.nets[action].predict(tuples=state_tuples)

        next_state_tuples = self._get_tuples(state=next_state)
        next_state_value = max(
            (
                self.nets[a].predict(tuples=next_state_tuples)
                if TwentyFortyEightEnv.apply_action(board=next_state, action=a)[2]
                else -np.inf
            )  # illegal action
            for a in range(4)
        )

        error = reward + next_state_value - state_value

        self.nets[action].update(tuples=state_tuples, delta=learning_rate * error)

    @classmethod
    def load(cls, path: str | bytes | os.PathLike) -> NTupleNetworkBasePolicy:
        policy = NTupleNetworkQLearningPolicy()
        with zipfile.ZipFile(path, "r") as archive:
            for i, filename in enumerate(sorted(archive.namelist())):
                with archive.open(filename) as file:
                    policy.nets[i] = NTupleNetwork.load(file)
        return policy

    def save(self, path: str | pathlib.Path | io.BufferedIOBase) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            for i, net in enumerate(self.nets):
                with tempfile.NamedTemporaryFile() as file:
                    net.save(file)
                    archive.write(file.name, f"net_{i}.zip")


class NTupleNetworkTDPolicy(NTupleNetworkBasePolicy):
    """
    N-tuple network policy using Temporal Difference Learning.
    """

    def evaluate(self, state: np.ndarray, action: int) -> float:
        after_state, reward, is_legal = TwentyFortyEightEnv.apply_action(
            board=state,
            action=action,
        )

        if not is_legal:
            return -np.inf

        after_state_tuples = self._get_tuples(state=after_state)
        return reward + self.net.predict(tuples=after_state_tuples)

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        learning_rate: float,
    ) -> None:
        after_state, _reward, is_legal = TwentyFortyEightEnv.apply_action(
            board=state,
            action=action,
        )
        assert is_legal, "the action should be legal"
        after_state_tuples = self._get_tuples(state=after_state)
        after_state_value = self.net.predict(tuples=after_state_tuples)

        next_action = np.argmax(
            [self.evaluate(state=next_state, action=a) for a in range(4)]
        )
        next_after_state, next_reward, is_legal = TwentyFortyEightEnv.apply_action(
            board=next_state,
            action=next_action,
        )
        if is_legal:
            next_after_state_tuples = self._get_tuples(state=next_after_state)
            next_after_state_value = self.net.predict(tuples=next_after_state_tuples)
        else:
            next_after_state_value = 0

        error = next_reward + next_after_state_value - after_state_value

        self.net.update(tuples=after_state_tuples, delta=learning_rate * error)

    @classmethod
    def load(cls, path: str | bytes | os.PathLike) -> NTupleNetworkBasePolicy:
        policy = NTupleNetworkTDPolicy()
        policy.net = NTupleNetwork.load(path=path)
        return policy

    def save(self, path: str | pathlib.Path | io.BufferedIOBase) -> None:
        self.net.save(path=path)


class NTupleNetworkTDPolicySmall(NTupleNetworkTDPolicy):
    """
    N-tuple network policy using Temporal Difference Learning with 3-tuples.
    """

    @staticmethod
    def _make_network() -> NTupleNetwork:
        return NTupleNetwork(shapes=[(15, 15, 15) for _ in range(20)])

    @staticmethod
    def _get_tuples(state: np.ndarray) -> Sequence[Sequence[int]]:
        return [
            *get_all_straight_3_tuples(state=state),
            *get_all_corners_3_tuples(state=state),
        ]

    @classmethod
    def load(cls, path: str | bytes | os.PathLike) -> NTupleNetworkBasePolicy:
        policy = NTupleNetworkTDPolicySmall()
        policy.net = NTupleNetwork.load(path=path)
        return policy
