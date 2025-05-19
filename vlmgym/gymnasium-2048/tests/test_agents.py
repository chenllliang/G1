# pylint: disable=redefined-outer-name
from tempfile import NamedTemporaryFile

import numpy as np
import pytest

from gymnasium_2048.agents.ntuple import NTupleNetwork
from gymnasium_2048.agents.ntuple.factory import (
    get_all_corners_3_tuples,
    get_all_rectangles_tuples,
    get_all_straight_3_tuples,
    get_all_straight_tuples,
)


@pytest.fixture
def state() -> np.ndarray:
    return np.array(
        [
            [6, 0, 3, 2],
            [7, 1, 0, 1],
            [1, 3, 0, 1],
            [7, 0, 0, 0],
        ],
        dtype=np.int32,
    )


def test_ntuple_network_num_weights():
    net1 = NTupleNetwork(shapes=[(15, 15, 15, 15) for _ in range(17)])
    net2 = NTupleNetwork(
        shapes=[
            *[(15, 15, 15, 15) for _ in range(2)],
            *[(15, 15, 15, 15, 15, 15) for _ in range(2)],
        ]
    )

    assert net1.num_weights() == 860_625
    assert net2.num_weights() == 22_882_500


def test_ntuple_network_save_load():
    shape = (15, 15, 15, 15)
    net1 = NTupleNetwork(shapes=[shape for _ in range(17)])
    for i in range(17):
        net1.weights[i] = np.random.rand(*shape)

    with NamedTemporaryFile("w") as file:
        net1.save(file.name)
        net2 = NTupleNetwork.load(file.name)

    assert net1.num_weights() == net2.num_weights()

    for i in range(17):
        np.testing.assert_allclose(net1.weights[i], net2.weights[i])

    tuples = [
        (6, 0, 3, 2),
        (7, 1, 0, 1),
        (1, 3, 0, 1),
        (7, 0, 0, 0),
        (6, 7, 1, 7),
        (0, 1, 3, 0),
        (3, 0, 0, 0),
        (2, 1, 1, 0),
        (6, 0, 1, 7),
        (0, 3, 0, 1),
        (3, 2, 1, 0),
        (7, 1, 3, 1),
        (1, 0, 0, 3),
        (0, 1, 1, 0),
        (1, 3, 0, 7),
        (3, 0, 0, 0),
        (0, 1, 0, 0),
    ]
    prediction1 = net1.predict(tuples=tuples)
    prediction2 = net2.predict(tuples=tuples)

    assert prediction1 == prediction2


def test_get_all_straight_tuples(state: np.ndarray):
    tuples = get_all_straight_tuples(state=state)

    assert tuples == [
        (6, 0, 3, 2),
        (7, 1, 0, 1),
        (1, 3, 0, 1),
        (7, 0, 0, 0),
        (6, 7, 1, 7),
        (0, 1, 3, 0),
        (3, 0, 0, 0),
        (2, 1, 1, 0),
    ]


def test_get_all_rectangles_tuples(state: np.ndarray):
    tuples = get_all_rectangles_tuples(state=state)

    assert tuples == [
        (6, 0, 1, 7),
        (0, 3, 0, 1),
        (3, 2, 1, 0),
        (7, 1, 3, 1),
        (1, 0, 0, 3),
        (0, 1, 1, 0),
        (1, 3, 0, 7),
        (3, 0, 0, 0),
        (0, 1, 0, 0),
    ]


def test_get_all_straight_3_tuples(state: np.ndarray):
    tuples = get_all_straight_3_tuples(state=state)

    assert tuples == [
        (6, 0, 3),
        (0, 3, 2),
        (7, 1, 0),
        (1, 0, 1),
        (1, 3, 0),
        (3, 0, 1),
        (7, 0, 0),
        (0, 0, 0),
        (6, 7, 1),
        (7, 1, 7),
        (0, 1, 3),
        (1, 3, 0),
        (3, 0, 0),
        (0, 0, 0),
        (2, 1, 1),
        (1, 1, 0),
    ]


def test_get_all_corners_3_tuples(state: np.ndarray):
    tuples = get_all_corners_3_tuples(state=state)

    assert tuples == [
        (6, 0, 7),
        (2, 3, 1),
        (7, 0, 1),
        (0, 0, 1),
    ]
