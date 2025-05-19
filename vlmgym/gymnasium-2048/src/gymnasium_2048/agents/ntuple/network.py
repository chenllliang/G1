from __future__ import annotations

import io
import os
import pathlib
import tempfile
import zipfile
from collections.abc import Sequence
from typing import Any

import numpy as np


class NTupleNetwork:
    """
    N-tuple network representation.

    :param shapes: The shapes of the tuples.
    :param dtype: The data-type of the network. Default: np.float64.
    """

    def __init__(
        self,
        shapes: Sequence[Sequence[int]],
        dtype: Any = np.float64,
    ) -> None:
        self.weights = [np.zeros(shape=shape, dtype=dtype) for shape in shapes]

    @classmethod
    def load(cls, path: str | bytes | os.PathLike) -> NTupleNetwork:
        """
        Load the n-tuple network from a zip-file.

        :param path: The path to the file (or a file-like).
        :return: New n-tuple network instance with loaded parameters.
        """
        net = NTupleNetwork(shapes=[])
        with zipfile.ZipFile(path, "r") as archive:
            for i in range(len(archive.filelist)):
                with archive.open(f"weights_{i}.npy", "r") as file:
                    net.weights.append(np.load(file))
        return net

    def num_weights(self) -> int:
        """
        Get the number of weights in the network.

        :return: The number of weights in the network.
        """
        return sum(w.size for w in self.weights)

    def predict(self, tuples: Sequence[Sequence[int]]) -> float:
        """
        Get the sum of values returned by the individual n-tuples.

        :param tuples: The individual n-tuples.
        :return: The prediction of the network.
        """
        assert len(tuples) == len(self.weights)
        return sum(w.item(t) for w, t in zip(self.weights, tuples))

    def update(self, tuples: Sequence[Sequence[int]], delta: float) -> None:
        """
        Update the values of the network for individual n-tuples.

        :param tuples: The individual n-tuples.
        :param delta: The delta to add.
        """
        assert len(tuples) == len(self.weights)
        for w, t in zip(self.weights, tuples):
            w.itemset(t, w.item(t) + delta)

    def save(self, path: str | pathlib.Path | io.BufferedIOBase) -> None:
        """
        Save the weights of the n-tuple network in a zip-file.

        :param path: The path to the file where the weights should be saved.
        """
        with zipfile.ZipFile(path, "w") as archive:
            for i, w in enumerate(self.weights):
                with tempfile.NamedTemporaryFile() as file:
                    np.save(file, w)
                    archive.write(file.name, f"weights_{i}.npy")
