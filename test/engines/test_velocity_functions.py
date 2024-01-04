"""Test velocity functions in all engines."""
import pathlib

import numpy as np
import pytest

from infretis.classes.engines.gromacs import GromacsEngine
from infretis.classes.engines.lammps import LAMMPSEngine
from infretis.classes.system import System

HERE = pathlib.Path(__file__).resolve().parent


def maxwell_vels(mass, vel, beta):
    out = (mass * beta / (2 * np.pi)) ** (1 / 2) * np.exp(
        -mass * vel**2 * beta / 2
    )
    return out


def return_lammps_engine():
    """Set up a lammps engine for the H2 system."""
    lammps_input_path = HERE / "../../examples/lammps/H2/lammps_input"
    engine = LAMMPSEngine("foobar", lammps_input_path.resolve(), -1, -1)
    print(engine.input_path)
    engine.rgen = np.random.default_rng()
    return engine


def return_gromacs_engine():
    """Set up a gromacs engine for the H2 system."""
    gromacs_input_path = HERE / "../../examples/gromacs/H2/gromacs_input"
    # set`gmx = echo` here because __init__ calls `gmx` with subprocess
    engine = GromacsEngine(
        "echo", "foobar", gromacs_input_path.resolve(), -1, -1
    )
    # engine.beta = 1/(300*0.0083144621)
    engine.rgen = np.random.default_rng()
    return engine


return_engines = [return_lammps_engine(), return_gromacs_engine()]

engine_classes = [LAMMPSEngine, GromacsEngine]


@pytest.mark.parametrize(
    "engine", [return_lammps_engine(), return_gromacs_engine()]
)
def test_modify_velocities(tmp_path, engine):
    """Check that we can modify the velocities with an engine."""
    # folder we wil run from
    folder = tmp_path / "temp"
    folder.mkdir()
    initial_conf = engine.input_path / f"conf.{engine.ext}"
    engine.exe_dir = folder

    system = System()
    system.set_pos((str(initial_conf.resolve()), 0))
    vel_settings = {
        "zero_momentum": False,
        "infretis_genvel": True,  # generate velocities internally for gromacs
        "temperature": 300,
        "mass": [1.008, 1.008],
    }

    engine.modify_velocities(system, vel_settings)

    # check if the genvel file is written
    genvel_conf = folder / f"genvel.{engine.ext}"
    assert genvel_conf.is_file()
    # we generated velocities, so we should have non-zero kinetic energy
    assert system.ekin != 0
