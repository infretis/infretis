"""Test methods for doing TIS."""
import os

import numpy as np

from infretis.classes.engines.engineparts import read_xyz_file
from infretis.classes.engines.factory import create_engine
from infretis.classes.orderparameter import create_orderparameters
from infretis.classes.path import Path, restart_path
from infretis.classes.rgen import MockRandomGenerator
from infretis.core.tis import prepare_shooting_point, shoot

ENG_SET = {
    "class": "turtlemd",
    "engine": "turtlemd",
    "timestep": 0.025,
    "temperature": 0.07,
    "boltzmann": 1.0,
    "subcycles": 1,
    "integrator": {
        "class": "LangevinInertia",
        "settings": {"gamma": 0.3, "beta": 14.285714285714285, "seed": 69},
    },
    "potential": {
        "class": "DoubleWell",
        "settings": {"a": 1.0, "b": 2.0, "c": 0.0},
    },
    "particles": {"mass": [1.0], "name": ["Z"], "pos": [[-1.0]]},
    "box": {"periodic": [False]},
}

ENS_SET = {
    "interfaces": (-0.99, -0.3, 1.0),
    "tis_set": {
        "maxlength": 2000,
        "aimless": True,
        "allowmaxlength": False,
        "zero_momentum": False,
        "rescale_energy": False,
    },
    "mc_move": "sh",
    "eng_name": "turtlemd",
    "ens_name": "007",
    "start_cond": "L",
    "rgen": MockRandomGenerator(),
}

TURTLE = create_engine({"engine": ENG_SET})
TURTLE.rgen = ENS_SET["rgen"]
ORDP_SET = {"class": "Position", "index": [0, 0], "periodic": False}
create_orderparameters({"engine": TURTLE}, {"orderparameter": ORDP_SET})
CWD = os.getcwd()
BASEPATH = os.path.dirname(__file__)

PATH_DIR = os.path.join(BASEPATH, "load/7/path.restart")
INP_PATH = restart_path(PATH_DIR)

# overwrite relative with absolute paths:
for idx, i in enumerate(INP_PATH.phasepoints):
    tup = (os.path.join(BASEPATH, i.config[0]), i.config[1])
    INP_PATH.phasepoints[idx].config = tup


def check_smooth(path):
    """Inspect whether a path is smooth or not.

    [TODO:description]

    Args:
        path ([TODO:type]): [TODO:description]

    .. Taken from:
        https://stats.stackexchange.com/questions/24607/
        how-to-measure-smoothness-of-a-time-series-in-r

    """
    orderp = [i.order[0] for i in path.phasepoints]
    diff = [i - j for i, j in zip(orderp[1:], orderp[:-1])]

    # less than 1 is "good"
    smooth = np.std(diff) / abs(np.average(diff))
    return smooth, smooth <= 1.0


def test_shooting(tmp_path: Path) -> None:
    """Template for shooting move tests.

    Should be changed with more comprehensive tests.

    Args:
        tmp_path: Input trajectory.
    """
    f1 = tmp_path / "temp"
    f1.mkdir()
    TURTLE.exe_dir = f1
    ENS_SET["rgen"] = MockRandomGenerator()
    TURTLE.rgen = ENS_SET["rgen"]
    success, trial_seg, status = shoot(ENS_SET, INP_PATH, TURTLE)
    assert not success
    assert trial_seg.length == 24
    assert status == "BWI"
    assert check_smooth(trial_seg)[1]


def test_prepare_shooting_point(tmp_path: Path) -> None:
    """Test the prepare shooting point function.

    Args:
        tmp_path: Input trajectory.
    """
    f1 = tmp_path / "temp"
    f1.mkdir()
    TURTLE.exe_dir = f1
    TURTLE.rgen = MockRandomGenerator()

    shpt_copy, idx, dek = prepare_shooting_point(INP_PATH, TURTLE.rgen, TURTLE)
    shpt_xyz = list(read_xyz_file(shpt_copy.config[0]))
    path_xyz = list(read_xyz_file(INP_PATH.phasepoints[0].config[0]))
    assert os.path.isfile(shpt_copy.config[0])
    assert len(shpt_xyz) == 1

    for key in shpt_xyz[0].keys():
        if key in ("vx", "vy", "vz"):
            assert shpt_xyz[0][key] != path_xyz[idx][key]
        elif key == "box":
            assert len(shpt_xyz[0][key]) == 3
            assert all(
                [i == j for i, j in zip(shpt_xyz[0][key], path_xyz[idx][key])]
            )
        else:
            assert shpt_xyz[0][key] == path_xyz[idx][key]
