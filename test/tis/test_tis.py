"""Test methods for doing TIS."""
import os
from pathlib import PosixPath

import numpy as np
import pytest
from rgen import MockRandomGenerator

from infretis.classes.engines.engineparts import read_xyz_file
from infretis.classes.engines.factory import create_engine
from infretis.classes.engines.turtlemdengine import TurtleMDEngine
from infretis.classes.orderparameter import create_orderparameters

# from infretis.classes.path import Path, restart_path
from infretis.classes.path import Path, load_path
from infretis.core.tis import (
    compute_weight,
    prepare_shooting_point,
    shoot,
    wire_fencing,
)

CWD = os.getcwd()
BASEPATH = os.path.dirname(__file__)
PATH_DIR = os.path.join(BASEPATH, "load/7/")
INP_PATH = load_path(PATH_DIR)
# overwrite relative with absolute paths:
for idx, i in enumerate(INP_PATH.phasepoints):
    tup = (os.path.join(BASEPATH, i.config[0]), i.config[1])
    INP_PATH.phasepoints[idx].config = tup


def return_engset() -> dict():
    eng_set = {
        "class": "turtlemd",
        "engine": "turtlemd",
        "timestep": 0.025,
        "temperature": 0.07,
        "boltzmann": 1.0,
        "subcycles": 1,
        "integrator": {
            "class": "LangevinInertia",
            "settings": {"gamma": 0.3, "beta": 14.285714285714285},
        },
        "potential": {
            "class": "DoubleWell",
            "settings": {"a": 1.0, "b": 2.0, "c": 0.0},
        },
        "particles": {"mass": [1.0], "name": ["Z"], "pos": [[-1.0]]},
        "box": {"periodic": [False]},
    }
    return eng_set


def return_ensset() -> dict():
    ens_set = {
        "interfaces": (-0.99, -0.3, 1.0),
        "tis_set": {
            "maxlength": 2000,
            "allowmaxlength": False,
            "zero_momentum": False,
            "n_jumps": 4,
        },
        "mc_move": "sh",
        "eng_name": "turtlemd",
        "ens_name": "007",
        "start_cond": "L",
        "rgen": MockRandomGenerator(),
    }
    return ens_set


def create_ensdic_and_engine() -> (dict(), TurtleMDEngine):
    eng_set = return_engset()
    ens_set = return_ensset()
    turtle = create_engine({"engine": eng_set})
    turtle.rgen = ens_set["rgen"]
    ordp_set = {"class": "Position", "index": [0, 0], "periodic": False}
    create_orderparameters({"engine": turtle}, {"orderparameter": ordp_set})
    return ens_set, turtle


def check_smooth(path: Path) -> tuple:
    """Inspect whether a path is smooth or not.

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


def create_traj_from_list(op_list: list) -> Path:
    toy_path = INP_PATH.empty_path()
    for op in op_list:
        copy = INP_PATH.phasepoints[0].copy()
        copy.order = [op]
        toy_path.append(copy)
    return toy_path


def test_shooting(tmp_path: PosixPath) -> None:
    """Template for shooting move tests.

    Should be changed with more comprehensive tests.

    Args:
        tmp_path: Input trajectory.
    """
    ens_set, turtle = create_ensdic_and_engine()
    f1 = tmp_path / "temp"
    f1.mkdir()
    turtle.exe_dir = f1
    turtle.rgen = ens_set["rgen"]
    print("lion", INP_PATH.generated)
    success, trial_seg, status = shoot(ens_set, INP_PATH, turtle)
    assert not success
    assert trial_seg.length == 21
    assert status == "BTL"
    assert check_smooth(trial_seg)[1]
    assert pytest.approx(trial_seg.ordermax[0]) == 0.8130819530087089


def test_wirefencing(tmp_path: PosixPath) -> None:
    """Template for wirefencing move tests.

    Should be changed with more comprehensive tests.

    Args:
        tmp_path: Input trajectory.
    """
    ens_set, turtle = create_ensdic_and_engine()
    ens_set["mc_move"] = "wf"
    f1 = tmp_path / "temp"
    f1.mkdir()
    turtle.exe_dir = f1
    turtle.rgen = ens_set["rgen"]

    success, trial_seg, status = wire_fencing(ens_set, INP_PATH, turtle)
    assert success
    assert trial_seg.length == 86
    assert status == "ACC"
    assert check_smooth(trial_seg)[1]
    assert pytest.approx(trial_seg.ordermax[0]) == 1.003664395705834
    assert trial_seg.weight == 124.0


def test_compute_weight_wf(tmp_path: PosixPath) -> None:
    """[TODO:summary]

    [TODO:description]

    Args:
        tmp_path: [TODO:description]

    Returns:
        [TODO:description]
    """
    f1 = tmp_path / "temp"
    f1.mkdir()

    interfaces = (0.0, 1.0, 2.0)  # A, i, cap
    # line shape:
    toy_path = create_traj_from_list([-1, 1.5, 1.5, 1.5, 1.5, 1.5, -1])
    weight = compute_weight(toy_path, interfaces, "wf")
    assert weight == 5

    # above cap line shape:
    toy_path = create_traj_from_list([-1, 2.5, 2.5, 2.5, 2.5, 2.5, -1])
    weight = compute_weight(toy_path, interfaces, "wf")
    assert weight == 0

    # M shape:
    toy_path = create_traj_from_list([-1, 1.5, 2.5, 1.5, 2.5, 1.5, -1])
    weight = compute_weight(toy_path, interfaces, "wf")
    assert weight == 2

    # stretched M shape:
    toy_path = create_traj_from_list([-1, 1.5, 2.5, 1.5, 2.5, 0.5, -1])
    weight = compute_weight(toy_path, interfaces, "wf")
    assert weight == 1


def test_prepare_shooting_point(tmp_path: PosixPath) -> None:
    """Testing the prepare shooting point function.

    Args:
        tmp_path: Input trajectory.
    """
    ens_set, turtle = create_ensdic_and_engine()
    f1 = tmp_path / "temp"
    f1.mkdir()
    turtle.exe_dir = f1

    shpt_copy, idx, dek = prepare_shooting_point(
        INP_PATH, turtle.rgen, turtle, ens_set
    )
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
