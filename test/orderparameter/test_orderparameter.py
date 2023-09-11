"""Test the different order parameters."""
import numpy as np
import pytest

from infretis.classes.orderparameter import (
    Distancevel,
    OrderParameter,
    Position,
    _verify_pair,
    pbc_dist_coordinate,
)
from infretis.classes.system import System

DIST = [
    np.array([8.0, 7.0, 9.0]),
    np.array([11.0, 1.0, 7.0]),
]
BOX = [
    np.array([10.0, 10.0, 10.0]),
    np.array([10.0, 10.0, 10.0]),
]
PBC_DIST = [
    np.array([-2.0, -3.0, -1.0]),
    np.array([1.0, 1.0, -3.0]),
]


@pytest.mark.parametrize("dist, box, pbc_dist", zip(DIST, BOX, PBC_DIST))
def test_pbc_dist(dist, box, pbc_dist):
    """Test the method for applying periodic boundaries."""
    test_dist = pbc_dist_coordinate(dist, box)
    assert pytest.approx(test_dist) == pbc_dist


def test_verify_pair():
    """Test that we can check that a list/tuple contains a pair."""
    index = [(1, 2), [1, 1], ("cat", "dog")]
    for i in index:
        _verify_pair(i)
    with pytest.raises(ValueError):
        _verify_pair((1, 2, 3))
    with pytest.raises(TypeError):
        _verify_pair(1)


def test_orderparameter(capsys):
    """Test the OrderParameter class."""
    order = OrderParameter(
        description="pytest",
        velocity=False,
    )
    assert not order.velocity_dependent
    order = OrderParameter(
        description="pytest",
        velocity=True,
    )
    assert order.velocity_dependent
    print(order)
    captured = capsys.readouterr()
    assert "This order parameter is velocity dependent" in captured.out


def test_distancevel():
    """Test the Distancevel order parameter."""
    order = Distancevel((0, 1), periodic=True)
    system = System()
    system.pos = np.array(
        [
            [1.0, 1.0, 1.0],
            [9.0, 8.0, 10.0],
        ]
    )
    system.vel = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )

    system.box = np.array([10.0, 10.0, 10.0])
    dist = np.array([-2.0, -3.0, -1.0])
    vel = np.array([1.0, 1.0, 1.0])
    correct = [np.dot(dist, vel) / np.sqrt(np.dot(dist, dist))]
    result = order.calculate(system)
    assert pytest.approx(correct) == result


def test_position():
    """Test the Position order parameter."""
    with pytest.raises(NotImplementedError):
        order = Position((1, 2), periodic=True)
    order = Position((0, 2), periodic=False)
    system = System()
    system.pos = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    orderp = order.calculate(system)
    assert pytest.approx(orderp) == [3.0]
