"""Test velocity functions in all engines."""
import pathlib

import numpy as np
import pytest

from infretis.classes.engines.gromacs import GromacsEngine
from infretis.classes.engines.lammps import LAMMPSEngine
from infretis.classes.system import System

HERE = pathlib.Path(__file__).resolve().parent


def return_lammps_engine() -> LAMMPSEngine:
    """Set up a lammps engine for the H2 system."""
    lammps_input_path = HERE / "../../examples/lammps/H2/lammps_input"
    temperature = 300
    engine = LAMMPSEngine(
        "foobar", lammps_input_path.resolve(), -1, -1, temperature
    )
    engine.rgen = np.random.default_rng()
    return engine


def return_gromacs_engine() -> GromacsEngine:
    """Set up a gromacs engine for the H2 system."""
    gromacs_input_path = HERE / "../../examples/gromacs/H2/gromacs_input/"
    # set`gmx = echo` here because __init__ calls it
    engine = GromacsEngine("echo", "bar", gromacs_input_path, -1, -1)
    return engine


@pytest.mark.parametrize("engine", [return_lammps_engine()])
def test_modify_velocities(tmp_path: pathlib.PosixPath, engine) -> None:
    """Check that we can modify the velocities with an engine."""
    # folder we wil run from
    folder = tmp_path / "temp"
    folder.mkdir()
    initial_conf = engine.input_path / f"conf.{engine.ext}"
    engine.exe_dir = folder

    system = System()
    system.set_pos((initial_conf, 0))
    engine.modify_velocities(system, {})

    # check if the genvel is written
    genvel_conf = folder / f"genvel.{engine.ext}"
    assert genvel_conf.is_file()
    # since we generated velocities we should not have 0 kinetic energy
    assert system.ekin != 0
