"""Test the different order parameters."""
import numpy as np
import pytest

from infretis.classes.orderparameter import (
    OrderParameter,
    _verify_pair,
    pbc_dist_coordinate,
)

DIST = [
    np.array([8.0, 7.0, 9.0]),
    np.array([11.0, 1.0, 7.0]),
]
BOX = [
    np.array([10.0, 10, 10]),
    np.array([10.0, 10, 10]),
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
