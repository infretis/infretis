from pathlib import PosixPath

import numpy as np
import pytest

from infretis.classes.path import Path as InfPath
from infretis.classes.system import System
from infretis.core.tis import ENGINES, quantis_swap_zero

HERE = PosixPath(__file__).resolve().parent


@pytest.fixture
def set_engine():
    def _set_engine(engine):
        global ENGINES
        ENGINES["engine"] = engine

    return _set_engine


class MockEngine:
    """A mockengine that generates fake paths with a given status."""

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
                    "out": (False, ""),
                }
            ],
            # single step crossing error for new path 1
            "QS1": [
                {"op": [1, 3], "V": [0, 0], "out": (True, "")},
                {"op": [1, 1, 1], "V": [0, 0, 0], "out": (False, "")},
            ],
            # swap is rejected because the energy acceptance rule is not met
            "QEA": [
                {"op": [1, 3], "V": [1e6, 0], "out": (True, "")},
                {"op": [1, 3], "V": [1e6, 0], "out": (True, "")},
            ],
            # backward trajectory too short
            "BTS": 2 * [{"op": [1, 3], "V": [0, 0], "out": (True, "")}]
            + [
                {
                    "op": [
                        1,
                    ],
                    "V": [0, 0],
                    "out": (True, ""),
                }
            ],
            # forward trajectory too short
            "FTS": 3 * [{"op": [1, 3], "V": [0, 0], "out": (True, "")}]
            + [
                {
                    "op": [
                        1,
                    ],
                    "V": [0, 0],
                    "out": (True, ""),
                }
            ],
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


def test_quantis_swap_zero():
    """Make some fake paths and check that we catch all error cases."""
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

    for true_status in ["QS0", "QS1", "QEA", "BTS", "FTS"]:
        engine = MockEngine(status=true_status)
        ENGINES["engine"] = engine

        picked = {
            -1: {
                "ens": {
                    "interfaces": (-np.inf, 2.0, 2.0),
                    "tis_set": {"maxlength": 100},
                    "rgen": np.random.default_rng(),
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
                    "rgen": np.random.default_rng(),
                    "mc_move": "sh",
                    "start_cond": "R",
                },
                "traj": path1,
                "eng_name": "engine",
            },
        }
        success, [new_path0, new_path1], status = quantis_swap_zero(picked)
        assert success is False
        assert status == true_status
