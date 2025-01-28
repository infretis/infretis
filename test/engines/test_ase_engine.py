"""Test the ase engine."""

import pathlib
import os

import ase
import numpy as np
import tomli

from infretis.classes.engines.factory import create_engine
from infretis.classes.orderparameter import create_orderparameter
from infretis.classes.path import Path
from infretis.classes.system import System

HERE = pathlib.Path(__file__).resolve().parent


def test_propagate(tmp_path: pathlib.PosixPath):
    """Test the _proagate_from function.

    We do it here by setting up two particles 10 angstrom apart with one of the
    atoms having velocities 1 angstrom/fs such that after 1 fs = 9 timesteps
    they should be 1 angstrom apart, meaning we have a path of length 1 + 9.

    """

    h2_path = HERE / "../../examples/ase/H2"

    # read .toml and change som options
    with open(h2_path / "infretis0.toml", "rb") as rfile:
        config = tomli.load(rfile)

    os.chdir(tmp_path)

    # modify some options
    config["engine"]["calculator_settings"]["module"] = str(
        (h2_path / "H2-calc.py").resolve()
    )
    config["orderparameter"]["periodic"] = False
    config["engine"]["subcycles"] = 1
    config["engine"]["timestep"] = 1.0
    config["engine"]["integrator"] = "velocityverlet"
    config["engine"]["calculator_settings"]["sigma"] = 0.0

    # setup engines and orderp
    engine = create_engine(config)
    engine.order_function = create_orderparameter(config)

    # setup toy system with velocities and positions
    atoms = ase.atoms.Atoms("H2")
    vel0 = np.zeros((2, 3))
    atoms.set_positions(vel0)
    atoms.positions[0, 0] = 10.0
    vel0[0, 0] = -1 * ase.units.Angstrom / ase.units.fs
    atoms.set_velocities(vel0)
    atoms.write("initial.traj")

    # prepare setup for propagation
    infsystem = System()
    infpath = Path(maxlen=100)
    infsystem.config = ("initial.traj", 0)

    # interfaces L, _, R
    ens_set = {"ens_name": "001", "interfaces": [1.05, None, np.inf]}

    # run the system
    engine.propagate(infpath, ens_set, infsystem)
    distances = np.array([pi.order for pi in infpath.phasepoints]).flatten()
    # check that length is 10
    assert len(distances) == 10
    # check that order is [10, 9, 8, ..., 1]
    assert np.allclose(distances, np.linspace(10, 1, 10))
