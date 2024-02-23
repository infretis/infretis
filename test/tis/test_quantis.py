from pathlib import PosixPath

import numpy as np
import pytest

from infretis.classes.engines.factory import create_engine
from infretis.classes.engines.turtlemdengine import TurtleMDEngine
from infretis.classes.orderparameter import create_orderparameters
from infretis.classes.path import Path as InfPath
from infretis.classes.path import load_path
from infretis.classes.system import System
from infretis.core.tis import (
    ENGINES,
    quantis_swap_zero,
    retis_swap_zero,
    shoot,
)

HERE = PosixPath(__file__).resolve().parent

PATH_DIR = PosixPath(HERE / "../../examples/turtlemd/double_well/load_copy")
INP_PATH0 = load_path(str(PATH_DIR / "0"))
INP_PATH1 = load_path(str(PATH_DIR / "1"))


class MockEngine:
    """A mockengine that generates fake paths with a given status.

    NB: This engine does not check interface crossings."""

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


def test_error_messages():
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
        ENGINES["engine"] = engine

        picked = {
            -1: {
                "ens": {
                    "interfaces": (-np.inf, 2.0, 2.0),
                    "tis_set": {"maxlength": 100},
                    "rgen": np.random.default_rng(seed=123),
                    "mc_move": "sh",
                    "start_cond": "L",
                },
                "traj": path0,
                "eng_name": "engine",
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
                "eng_name": "engine",
            },
        }
        success, [new_path0, new_path1], status = quantis_swap_zero(picked)

        assert status == true_status


def return_ensset(interfaces, name, start_cond):
    ens_set = {
        "interfaces": interfaces,
        "tis_set": {
            "maxlength": 100,
            "allowmaxlength": True,
            "zero_momentum": False,
        },
        "mc_move": "sh",
        "eng_name": "turtlemd",
        "ens_name": name,
        "start_cond": start_cond,
        "rgen": np.random.default_rng(seed=123),
    }
    return ens_set


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
            "settings": {"gamma": 1e-5, "beta": 1e12},
        },
        "potential": {
            "class": "DoubleWell",
            "settings": {"a": 1.0, "b": 2.0, "c": 0.0},
        },
        "particles": {"mass": [1.0], "name": ["Z"], "pos": [[-1.0]]},
        "box": {"periodic": [False]},
    }
    return eng_set


def create_ensdic_and_engine() -> (dict(), TurtleMDEngine):
    ens_set0 = return_ensset((-np.inf, -0.99, -0.99), "000", "R")
    ens_set1 = return_ensset((-0.99, -0.8, 1.0), "001", "L")
    eng_set = return_engset()
    turtle = create_engine({"engine": eng_set})
    turtle.rgen = np.random.default_rng(seed=123)
    ordp_set = {"class": "Position", "index": [0, 0], "periodic": False}
    create_orderparameters({"engine": turtle}, {"orderparameter": ordp_set})
    return ens_set0, ens_set1, turtle


@pytest.mark.parametrize(
    "zero_swap_move",
    [retis_swap_zero, quantis_swap_zero],
)
def test_zero_swaps(tmp_path, zero_swap_move):
    """Check that three consecutive zero swaps 1, 2 and 3 gives back the
    original path at swap 2, and that the paths obtained from swap 1 and 3
    are identical as well (for both ensembles [0-] and [0+]).
    """
    ens_set0, ens_set1, turtle = create_ensdic_and_engine()
    turtle.exe_dir = tmp_path
    ENGINES["turtlemd"] = turtle
    picked = {
        -1: {"ens": ens_set0, "traj": INP_PATH0, "eng_name": "turtlemd"},
        0: {"ens": ens_set1, "traj": INP_PATH1, "eng_name": "turtlemd"},
    }

    # we need to shoot here because we don't have velocities in the load paths
    # so the single step crossing would fail
    success, old_path0, status = shoot(
        ens_set0, INP_PATH0, turtle, start_cond="R"
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
        pathdir = PosixPath(tmp_path / f"{i}")
        pathdir.mkdir()
        turtle.exe_dir = str(pathdir)
        picked[-1]["traj"] = old_path0
        picked[0]["traj"] = old_path1
        success, [new_path0, new_path1], status = zero_swap_move(picked)
        swapped_paths0.append(new_path0)
        swapped_paths1.append(new_path1)
        old_path0 = new_path0
        old_path1 = new_path1

    for swapped_paths in [swapped_paths0, swapped_paths1]:
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
