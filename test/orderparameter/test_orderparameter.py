"""Test the different order parameters."""
import pathlib

import numpy as np
import pytest

from infretis.classes.orderparameter import (
    Dihedral,
    Distance,
    Distancevel,
    OrderParameter,
    Position,
    Velocity,
    _verify_pair,
    create_orderparameter,
    pbc_dist_coordinate,
)
from infretis.classes.system import System

HERE = pathlib.Path(__file__).resolve().parent

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


def test_distance():
    """Test the Distance order parameter."""
    order = Distance((0, 1), periodic=True)
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
    lamb = [np.sqrt(np.dot(dist, dist))]
    result = order.calculate(system)
    assert pytest.approx(result) == lamb


def test_velocity():
    """Test the Velocity order parameter."""
    with pytest.raises(ValueError):
        order = Velocity(0, dim="w")
    order = Velocity(1, dim="y")
    assert order.dim == 1
    system = System()
    system.vel = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 123.0, 2.0],
        ]
    )
    result = order.calculate(system)
    assert pytest.approx(result) == [123.0]


def test_dihedral_setup():
    """Test the Dihedral order parameter."""
    with pytest.raises(TypeError):
        _ = Dihedral(1, periodic=True)
    with pytest.raises(ValueError):
        _ = Dihedral((1, 2))
    # Check that the calculation is the same if we reverse the
    # indices:
    order1 = Dihedral((0, 1, 2, 3), periodic=False)
    order2 = Dihedral((3, 2, 1, 0), periodic=False)
    system = System()
    system.pos = np.random.random_sample(size=(4, 3))
    result1 = order1.calculate(system)
    result2 = order2.calculate(system)
    assert pytest.approx(result1) == result2


DIHEDRAL_POS = [
    np.array(
        [
            [0.039, -0.028, 0.000],
            [1.499, -0.043, 0.000],
            [1.956, -0.866, -1.217],
            [1.571, -1.903, -1.181],
        ]
    ),
    np.array(
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    ),
]
DIHEDRAL_ANGLE = [
    [np.radians(-59.365971)],
    [np.radians(90.0)],
]


@pytest.mark.parametrize("pos, correct", zip(DIHEDRAL_POS, DIHEDRAL_ANGLE))
def test_dihedral_calculation(pos, correct):
    """Test calculation of diheral angles."""
    system = System()
    system.pos = pos
    system.box = np.array([10.0, 10.0, 10.0])
    order = Dihedral((0, 1, 2, 3), periodic=True)
    angle = order.calculate(system)
    assert pytest.approx(angle) == correct


ORDER_SETTINGS = [
    {"class": "orderparameter", "description": "test"},
    {"class": "position", "index": (1, 2), "periodic": False},
    {"class": "velocity", "index": 1},
    {"class": "distance", "index": (1, 2)},
    {"class": "dihedral", "index": (0, 1, 2, 3)},
    {"class": "distancevel", "index": (1, 2)},
]
CORRECT_ORDER = [
    OrderParameter,
    Position,
    Velocity,
    Distance,
    Dihedral,
    Distancevel,
]


@pytest.mark.parametrize(
    "settings, correct", zip(ORDER_SETTINGS, CORRECT_ORDER)
)
def test_create_internal_orderparameter(settings, correct):
    """Test that we can create order parameters from settings."""
    all_settings = {"orderparameter": settings}
    order = create_orderparameter(all_settings)
    assert type(order) is correct


def test_create_external_orderparameter():
    """Test that we can create an external order parameter."""
    settings = {
        "orderparameter": {
            "class": "ConstantOrder",
            "module": HERE / "foo.py",
            "constant": 101,
        }
    }
    order = create_orderparameter(settings)
    assert order.constant == 101

    settings = {
        "orderparameter": {
            "class": "OrderMissingCalculate",
            "module": HERE / "foo.py",
        }
    }
    with pytest.raises(ValueError):
        _ = create_orderparameter(settings)
