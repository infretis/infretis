from pathlib import PosixPath

import numpy as np

from infretis.classes.repex import REPEX_state
from infretis.classes.system import System
from infretis.setup import setup_config

HERE = PosixPath(__file__).resolve().parent


class MockEngine:
    """A mockengine that generates fake paths with a given status."""

    # assume interfaces = [2, 4]
    def __init__(self):
        self.status = "QS0"
        self.beta = 1.0
        self.rgen = np.random.default_rng()
        self.trajs = {
            "QS0": [
                {
                    "order": [1.0, 1.5, 1.9],
                    "vpot": [0.0, 0.0, 0.0],
                    "out": (False, ""),
                }
            ],
            "QS1": [
                {"order": [1.0, 2.1], "vpot": [0.0, 0.0], "out": (True, "")},
                {
                    "order": [1.0, 1.5, 1.9],
                    "vpot": [0.0, 0.0, 0.0],
                    "out": (False, ""),
                },
            ],
            "QSE": [
                {
                    "order": [1.0, 2.1],
                    "vpot": [999.0, 0.0],
                    "return": (True, ""),
                },
                {
                    "order": [1.0, 2.1],
                    "vpot": [-999.0, 0.0],
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
            system.order = order
            system.vpot = vpot
            system.vel_rev = reverse
            path.append(system)
        return traj["out"]


# def test_quantis_swap_zero(tmp_path: PosixPath):
#    os.chdir(tmp_dir)

example_folder = (HERE / "../../examples/turtlemd/H2/").resolve()
initial_configuration = example_folder / "conf.xyz"
toml = str((example_folder / "infretis.toml").resolve())

# setup
config = setup_config(toml)
state = REPEX_state(config, minus=True)
state.initiate_ensembles()
ens_set0 = {
    "interfaces": (0.345, 0.345, 1.2),
    "tis_set": {
        "maxlength": 2000,
        "allowmaxlength": False,
        "zero_momentum": True,
        "n_jumps": 3,
        "interface_cap": 0.54,
    },
    "mc_move": "sh",
    "eng_name": "turtlemd",
    "ens_name": "001",
    "start_cond": "L",
}

ens_set1 = {
    "interfaces": (-np.inf, 0.345, 0.345),
    "tis_set": {
        "maxlength": 2000,
        "allowmaxlength": False,
        "zero_momentum": True,
        "n_jumps": 3,
        "interface_cap": 0.54,
    },
    "mc_move": "sh",
    "eng_name": "turtlemd",
    "ens_name": "000",
    "start_cond": "R",
}
