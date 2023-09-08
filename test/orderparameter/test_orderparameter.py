"""Test the different order parameters."""
import numpy as np
import pytest

from infretis.classes.orderparameter import pbc_dist_coordinate

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
