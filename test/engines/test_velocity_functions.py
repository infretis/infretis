"""Test velocity functions in all engines."""
import pathlib

import numpy as np
import pytest
import tomli
import importlib.util
if importlib.util.find_spec("scm.plams") is not None:
    from infretis.classes.engines.ams import AMSEngine
from infretis.classes.engines.cp2k import CP2KEngine
from infretis.classes.engines.factory import create_engine
from infretis.classes.engines.gromacs import GromacsEngine
from infretis.classes.engines.lammps import LAMMPSEngine
from infretis.classes.system import System

HERE = pathlib.Path(__file__).resolve().parent

# velocities are from a long MD run and may include zero_momentum and stuff
EXPECTED_STDDEV = {
    "cp2k": 0.0007187852735563246,
    "turtlemd": 1.57265889690332,
    "gromacs": 1.1131628,
    "lammps": 0.01573556087968066,
    "ase": 0.1113976956951558,
    "ams": 1.1131628,
}


def return_turtlemd_engine():
    """Set up a turtlemd engine for the H2 system."""
    input_path = HERE / "../../examples/turtlemd/H2/"
    toml_file = input_path / "infretis.toml"
    with open(toml_file, "rb") as rfile:
        config = tomli.load(rfile)
    engine = create_engine(config)
    engine.input_path = input_path
    engine.rgen = np.random.default_rng()
    engine.vel_settings = {
        "zero_momentum": False,
        "temperature": 300,
    }
    return engine


def return_lammps_engine():
    """Set up a lammps engine for the H2 system."""
    lammps_input_path = HERE / "../../examples/lammps/H2/lammps_input"
    engine = LAMMPSEngine("lmp_mpi", lammps_input_path.resolve(), 0, 0, 300)
    engine.rgen = np.random.default_rng()
    engine.vel_settings = {
        "zero_momentum": False,
    }
    return engine


def return_gromacs_engine():
    """Set up a gromacs engine for the H2 system."""
    gromacs_input_path = HERE / "../../examples/gromacs/H2/gromacs_input"
    # set`gmx = echo` here because __init__ calls `gmx` with subprocess
    engine = GromacsEngine(
        "echo",
        gromacs_input_path.resolve(),
        0,
        0,
        300,
        masses=[1.008, 1.008],
        infretis_genvel=True,
    )
    engine.vel_settings = {
        "zero_momentum": True,
    }
    engine.rgen = np.random.default_rng()
    return engine


def return_cp2k_engine():
    """Set up a cp2k engine for the H2 system."""
    cp2k_input_path = HERE / "../../examples/cp2k/H2/cp2k_input"
    engine = CP2KEngine("cp2k", cp2k_input_path.resolve(), 1, 1, 300)
    engine.rgen = np.random.default_rng()
    engine.vel_settings = {
        "zero_momentum": False,
    }
    return engine

def return_ase_engine():
    """Set up an ase engine for the H2 system."""
    ase_toml_path = HERE / "../../examples/ase/H2/infretis0.toml"
    calc_path = HERE / "../../examples/ase/H2/H2-calc.py"
    toml_file = ase_toml_path
    with open(toml_file, "rb") as rfile:
        config = tomli.load(rfile)
    config["engine"]["calculator_settings"]["module"] = str(calc_path.resolve())
    engine = create_engine(config)
    engine.input_path = HERE / "../../examples/ase/H2"
    engine.vel_settings = {
        "zero_momentum": True,
    }
    return engine

def return_ams_engine():
    """Set up an ams engine for the H2 system."""
    ams_input_path = pathlib.Path("ams_inp/")
    
    engine = AMSEngine(ams_input_path, 0.001,1)
    engine.vel_settings = {
        "zero_momentum": True,
        "aimless": True

    }
    engine.ens_name = "test"
    return engine


@pytest.mark.parametrize(
    "engine",
    [
        return_gromacs_engine(),
        return_lammps_engine(),
        return_cp2k_engine(),
        return_turtlemd_engine(),
        return_ase_engine(),
        pytest.param(return_ams_engine(), 
            marks=pytest.mark.skipif(
                importlib.util.find_spec("scm.plams") is None, 
                reason="scm.plams not installed")
                ),
    ],
)
def test_modify_velocities(tmp_path, engine):
    """Check that we can modify the velocities with an engine,
    and that they are not equal to zero."""
    # folder we wil run from
    folder = tmp_path / "temp"
    folder.mkdir()
    if type(engine.input_path) == str:
        initial_conf = pathlib.Path(engine.input_path + f"/conf.{engine.ext}")
    else:
        initial_conf = engine.input_path / f"conf.{engine.ext}"
    engine.exe_dir = folder

    system = System()
    system.set_pos((str(initial_conf.resolve()), 0))
    vel_settings = engine.vel_settings

    engine.modify_velocities(system, vel_settings)
    genvel_conf = folder / f"genvel.{engine.ext}"

    assert genvel_conf.is_file() or any(folder.glob("genvel_test?????_??????.rkf"))
    # we generated velocities, so we should have non-zero kinetic energy
    assert system.ekin != 0


@pytest.mark.parametrize(
    "engine",
    [
        return_gromacs_engine(),
        return_lammps_engine(),
        return_cp2k_engine(),
        return_turtlemd_engine(),
        return_ase_engine(),
        pytest.param(return_ams_engine(), 
            marks=pytest.mark.skipif(
                importlib.util.find_spec("scm.plams") is None, 
                reason="scm.plams not installed")
                ),
    ],
)
@pytest.mark.heavy
def test_modify_velocity_distribition(tmp_path, engine):
    """Check that velocities are generated with the correct distribution.

    We compare here the generated velocitied with standard deviations from a
    long MD run with the given engine. This takes care of the units and other
    settings such as zero_momentum.

    However, the velocity distribution should be a normal distribution with
    std.dev = (Temp*u.K*u.k_B*1/(mass*u.atomic_mass_constant))**0.5
    and zero mean. For zero_momentum = True, the distribution should be
    divided by sqrt(1/1.008 + 1/1.008) for the H2 system.
    """
    # folder we wil run from
    folder = tmp_path / "temp"
    folder.mkdir()

    if type(engine.input_path) == str:
        initial_conf = pathlib.Path(engine.input_path + f"/conf.{engine.ext}")
    else:
        initial_conf = engine.input_path / f"conf.{engine.ext}"    
        
    engine.exe_dir = folder

    system = System()
    system.set_pos((str(initial_conf.resolve()), 0))
    vel_settings = engine.vel_settings

    engine.modify_velocities(system, vel_settings)
    genvel_conf = folder / f"genvel.{engine.ext}"

    assert genvel_conf.is_file() or any(folder.glob("genvel_test?????_??????.rkf"))

    # we generated velocities, so we should have non-zero kinetic energy
    assert system.ekin != 0
    # check that we have the correct velocity units
    # by comparing std.dev from equilibrium run with
    # the ones generated by modify_velocities
    N_samples = 600  # 10 # gives 6*N_samples velocities
    vel_arr = []
    for i in range(N_samples):
        engine.modify_velocities(system, vel_settings)
        genvel_conf = folder / f"genvel.{engine.ext}"
        _, vel, _, _ = engine._read_configuration(system.config[0])
        for veli in vel.flatten():
            vel_arr.append(veli)
    exp = EXPECTED_STDDEV[engine.name]
    calc = np.std(vel_arr)
    # may be a bit strict with N_samples = 600
    assert abs((exp - calc) / exp) * 100 < 5
