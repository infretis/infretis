"""Test methods for doing TIS."""
import os
from collections.abc import Callable
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
from infretis.classes.path import Path as InfPath
from infretis.classes.system import System
from infretis.core.tis import (
    compute_weight,
    prepare_shooting_point,
    quantis_swap_zero,
    retis_swap_zero,
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

# paths for zero swaps
PATH_DIR_2 = PosixPath(
    BASEPATH, "../../examples/turtlemd/double_well/load_copy"
).resolve()
INP_PATH0 = load_path(str(PATH_DIR_2 / "0"))
INP_PATH1 = load_path(str(PATH_DIR_2 / "1"))


class MockEngine:
    """A mockengine that generates fake paths with a given status.

    Note: This engine does not check interface crossings.
    """

    def __init__(self, status="QS0"):
        self.status = status
        self.beta = 1.0
        self.rgen = np.random.default_rng()
        self.trajs = {
            # assuming interfaces = [2, 4]
            # single step crossing error for new path 0
            "QS0": [
                {
                    "op": [1, 1],
                    "V": [0, 0],
                    "out": ("", ""),
                }
            ],
            # single step crossing error for new path 1
            "QS1": [
                {"op": [1, 3], "V": [0, 0], "out": ("", "")},
                {"op": [1, 1, 1], "V": [0, 0, 0], "out": ("", "")},
            ],
            # swap is rejected because the energy acceptance rule is not met
            "QEA": [
                {"op": [1, 3], "V": [1e6, 0], "out": ("", "")},
                {"op": [1, 3], "V": [1e6, 0], "out": ("", "")},
            ],
            # backward trajectory (full trial path in [0-]) too short
            "BTS": 2 * [{"op": [1, 3], "V": [0, 0], "out": ("", "")}]
            + [
                {
                    "op": [
                        1,
                    ],
                    "V": [0],
                    "out": ("", ""),
                }
            ],
            # forward trajectory (full trial path in [0+]) too short
            "FTS": 3 * [{"op": [1, 3], "V": [0, 0], "out": ("", "")}]
            + [{"op": [1], "V": [0], "out": ("", "")}],
            # backward trajectory (full trial path in [0-]) >= maxlen
            "BTX": 2 * [{"op": [1, 3], "V": [0, 0], "out": ("", "")}]
            + [{"op": 99 * [2], "V": 99 * [0], "out": ("", "")}],
            "ACC": 2 * [{"op": [1, 3], "V": [0, 0], "out": ("", "")}]
            + [{"op": [1, 3], "V": [0, 0], "out": ("", "")}]
            + [{"op": [1, 3], "V": [0, 0], "out": ("", "")}],
        }

    def propagate(self, path, ens_set, shooting_point, reverse=False):
        traj = self.trajs[self.status].pop(0)
        for order, vpot in zip(traj["op"], traj["V"]):
            system = System()
            system.order = [order]
            system.vpot = vpot
            system.vel_rev = reverse
            path.append(system)
        return traj["out"]


def return_engset() -> dict:
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


def return_ensset() -> dict:
    ens_set = {
        "interfaces": (-0.99, -0.3, 1.0),
        "tis_set": {
            "maxlength": 2000,
            "allowmaxlength": False,
            "zero_momentum": False,
            "n_jumps": 4,
        },
        "mc_move": "sh",
        "ens_name": "007",
        "start_cond": "L",
        "rgen": MockRandomGenerator(),
    }
    return ens_set


def create_ensdic_and_engine() -> tuple[dict, TurtleMDEngine]:
    eng_set = return_engset()
    ens_set = return_ensset()
    turtle = create_engine({"engine": eng_set})
    turtle.rgen = ens_set["rgen"]
    ordp_set = {"class": "Position", "index": [0, 0], "periodic": False}
    create_orderparameters(
        {"engine": [[-1], [turtle]]}, {"orderparameter": ordp_set}
    )
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


def test_quantis_swap_zero_messages() -> None:
    """Make some fake paths and check that we catch all errors."""
    # the old paths we want to swap
    path0 = InfPath()
    for order in [2.5, 1.0, 2.5]:
        system = System()
        system.order = [order]
        system.vpot = 0.0
        path0.append(system)

    path1 = InfPath()
    for order in [1.0, 2.5, 1.0]:
        system = System()
        system.order = [order]
        system.vpot = 0.0
        path1.append(system)

    for true_status in ["QS0", "QS1", "QEA", "BTS", "FTS", "BTX", "ACC"]:
        engine = MockEngine(status=true_status)
        engines = {-1: [engine], 0: [engine]}

        picked = {
            -1: {
                "ens": {
                    "interfaces": (-np.inf, 2.0, 2.0),
                    "tis_set": {"maxlength": 100, "accept_all": False},
                    "rgen": np.random.default_rng(seed=123),
                    "mc_move": "sh",
                    "start_cond": "L",
                },
                "traj": path0,
            },
            0: {
                "ens": {
                    "interfaces": (2.0, 2.0, 4.0),
                    "tis_set": {"maxlength": 100},
                    "rgen": np.random.default_rng(seed=123),
                    "mc_move": "sh",
                    "start_cond": "R",
                },
                "traj": path1,
            },
        }
        success, [new_path0, new_path1], status = quantis_swap_zero(
            picked, engines
        )

        assert status == true_status


@pytest.mark.parametrize(
    "zero_swap_move",
    [retis_swap_zero, quantis_swap_zero],
)
@pytest.mark.heavy
def test_zero_swaps(
    tmp_path: PosixPath,
    zero_swap_move: Callable[
        [dict], tuple[bool, tuple[InfPath, InfPath], str]
    ],
) -> None:
    """Check that three consecutive zero swaps 1, 2 and 3 gives back the
    original path at swap 2, and that the paths obtained from swap 1 and 3
    are identical as well (for both ensembles [0-] and [0+]).
    """
    tmp_dir = tmp_path / "tmp2"
    tmp_dir.mkdir()
    ens_set0, turtle = create_ensdic_and_engine()
    ens_set1 = ens_set0.copy()
    # update settings
    ens_set0["interfaces"] = (-np.inf, -0.99, -0.99)
    ens_set1["interfaces"] = (-0.99, -0.3, 1.0)
    ens_set0["tis_set"]["allowmaxlength"] = True
    ens_set1["tis_set"]["allowmaxlength"] = True
    ens_set0["tis_set"]["accept_all"] = False
    ens_set1["tis_set"]["accept_all"] = False
    ens_set0["name"] = "000"
    ens_set1["name"] = "001"
    ens_set0["start_cond"] = "R"
    ens_set1["start_cond"] = "L"
    ens_set0["rgen"] = np.random.default_rng(seed=123)
    ens_set1["rgen"] = np.random.default_rng(seed=123)
    turtle.rgen = np.random.default_rng(seed=123)
    turtle.integrator_settings = {"beta": 1e12, "gamma": 1e-5}
    turtle.exe_dir = str(tmp_dir)
    engines = {-1: [turtle], 0: [turtle]}

    picked = {
        -1: {"ens": ens_set0, "traj": INP_PATH0},
        0: {"ens": ens_set1, "traj": INP_PATH1},
    }

    # we need to shoot here because we don't have velocities in the load paths
    # so the single step crossing would fail
    success, old_path0, status = shoot(
        ens_set0, INP_PATH0, turtle, start_cond=("R",)
    )
    success, old_path1, status = shoot(ens_set1, INP_PATH1, turtle)

    for i in range(old_path0.length):
        old_path0.phasepoints[i].vpot = 0

    for i in range(old_path1.length):
        old_path1.phasepoints[i].vpot = 0

    swapped_paths0 = [old_path0]
    swapped_paths1 = [old_path1]
    for i in range(1, 4):
        # run from separate folders so *.xyz files from previous runs
        # don't mess up the zero swaps
        pathdir = PosixPath(tmp_dir / f"{i}")
        pathdir.mkdir()
        turtle.exe_dir = str(pathdir.resolve())
        picked[-1]["traj"] = old_path0
        picked[0]["traj"] = old_path1
        success, [new_path0, new_path1], status = zero_swap_move(
            picked, engines
        )
        swapped_paths0.append(new_path0)
        swapped_paths1.append(new_path1)
        old_path0 = new_path0
        old_path1 = new_path1

    for swapped_paths in [swapped_paths0, swapped_paths1]:
        print([s.status for s in swapped_paths])
        assert swapped_paths[0].length == swapped_paths1[2].length
        assert swapped_paths[1].length == swapped_paths1[3].length
        assert [
            pp.order[0] for pp in swapped_paths[0].phasepoints
        ] == pytest.approx(
            [pp.order[0] for pp in swapped_paths[2].phasepoints], rel=1e-5
        )
        assert [
            pp.order[0] for pp in swapped_paths[1].phasepoints
        ] == pytest.approx(
            [pp.order[0] for pp in swapped_paths[3].phasepoints], rel=1e-5
        )
