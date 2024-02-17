from pathlib import PosixPath

import numpy as np

from infretis.classes.path import Path as InfPath
from infretis.classes.system import System
from infretis.core.tis import quantis_swap_zero

HERE = PosixPath(__file__).resolve().parent


class MockEngine:
    """A mockengine that generates fake paths with a given status."""

    # assume interfaces = [2, 4]
    def __init__(self, status="QS0"):
        self.status = status
        self.beta = 1.0
        self.rgen = np.random.default_rng()
        self.trajs = {
            # single step crossing error for new path 0
            "QS0": [
                {
                    "order": [
                        1.0,
                        1.5,
                    ],
                    "vpot": [
                        0.0,
                        0.0,
                    ],
                    "return": (False, ""),
                }
            ],
            # single step crossing error for new path 1
            "QS1": [
                {
                    "order": [1.0, 2.1],
                    "vpot": [0.0, 0.0],
                    "return": (True, ""),
                },
                {
                    "order": [1.0, 1.5, 1.9],
                    "vpot": [0.0, 0.0, 0.0],
                    "return": (False, ""),
                },
            ],
            # swap is rejected because the energy acceptance rule is not met
            "QEA": [
                {
                    "order": [1.0, 2.1],
                    "vpot": [9999.0, 0.0],
                    "return": (True, ""),
                },
                {
                    "order": [1.0, 2.1],
                    "vpot": [9999.0, 0.0],
                    "return": (True, ""),
                },
            ],
        }

    def propagate(self, path, ens_set, shooting_point, reverse=False):
        # get the next trajectory for the given
        # error message
        traj = self.trajs[self.status].pop(0)
        for order, vpot in zip(traj["order"], traj["vpot"]):
            system = System()
            system.order = [order]
            system.vpot = vpot
            system.vel_rev = reverse
            path.append(system)
        return traj["return"]


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

    for true_status in ["QS0", "QS1", "QEA"]:
        engine = MockEngine(status=true_status)

        picked = {
            -1: {
                "ens": {
                    "interfaces": (-np.inf, 2.0, 2.0),
                    "tis_set": {"maxlength": 100},
                    "rgen": np.random.default_rng(),
                },
                "engine": engine,
                "traj": path0,
            },
            0: {
                "ens": {
                    "interfaces": (2.0, 2.0, 4.0),
                    "tis_set": {"maxlength": 100},
                    "rgen": np.random.default_rng(),
                },
                "engine": engine,
                "traj": path1,
            },
        }
        success, [new_path0, new_path1], status = quantis_swap_zero(picked)
        assert status == true_status
