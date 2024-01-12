"""
A script to generate load paths for the H2 examples.

Usage:
    # run  this script in H2 example folder
    # e.g. from infretis/examples/gromacs/H2
    python ../../../infretis/tools/generate_H2_loadpaths.py
"""
from pathlib import Path

import numpy as np

from infretis.classes.engines.cp2k import write_xyz_trajectory
from infretis.classes.engines.enginebase import EngineBase
from infretis.classes.engines.factory import create_engine
from infretis.classes.engines.gromacs import (
    read_gromos96_file,
    write_gromos96_file,
)
from infretis.classes.engines.lammps import (
    read_lammpstrj,
    write_lammpstrj,
)
from infretis.classes.formatter import PathStorage
from infretis.classes.orderparameter import create_orderparameter
from infretis.classes.path import Path as InfPath
from infretis.classes.repex import REPEX_state
from infretis.classes.system import System
from infretis.core.tis import shoot
from infretis.setup import setup_config


def write_conf(engine, filename, distance, outfile):
    """Write a configuration file for the H2 example with a given distance."""
    xyz, vel, box, names = engine._read_configuration(filename)
    xyz *= 0
    xyz[0, 0] = distance
    if engine.name == "gromacs":
        txt, _, _, _ = read_gromos96_file(filename)
        write_gromos96_file(outfile, txt, xyz, vel)
    elif engine.name in ["cp2k", "turtlemd"]:
        write_xyz_trajectory(outfile, xyz, vel, names, box, append=False)
    elif engine.name == "lammps":
        id_type, _, _, box = read_lammpstrj(filename, 0, engine.n_atoms)
        write_lammpstrj(outfile, id_type, xyz, vel, box)


def create_initial_paths():
    """Create initial paths for the H2 system using shooting."""
    # initiate enembles, engines and orderparameter
    config = setup_config()
    state = REPEX_state(config, minus=True)
    state.initiate_ensembles()
    engine: EngineBase = create_engine(config)
    engine.order_function = create_orderparameter(config)

    interfaces = np.array(config["simulation"]["interfaces"])
    # the initial configurations for each ensemble are given by the
    # interface location + (interface1 - interface0)/100
    delta = (interfaces[1] - interfaces[0]) / 100
    distances = interfaces * 1
    distances[1:] = distances[:-1] + delta
    distances[0] -= delta

    # some engine setup
    exe_dir = Path("temporary_load")  # Path(engine.input_path) / "../load/"
    exe_dir = exe_dir.resolve()
    exe_dir.mkdir()
    engine.exe_dir = exe_dir
    engine.rgen = np.random.default_rng()
    engine.name = config["engine"]["class"]

    if config["engine"]["class"] == "turtlemd":
        engine.input_path = Path(".")

    for ens_name in state.ensembles.keys():
        # some ensemble setup
        ensemble = state.ensembles[ens_name]
        ensemble["rgen"] = np.random.default_rng()
        ensemble[
            "allowmaxlength"
        ] = True  # should be in simulation not tis_set
        ensemble["maxlength"] = 2000  # should be in simulation not tis_set

        # set up a temporary path to start from
        path = InfPath(maxlen=2000)
        system = System()

        # initial configuration we start shooting from
        initial_conf = exe_dir / f"tmp.{engine.ext}"
        template = engine.input_path / f"conf.{engine.ext}"
        write_conf(engine, template, distances[ens_name], initial_conf)
        system.set_pos((str(initial_conf.resolve()), 0))

        # shoot until we have an accepted path
        status = "no status yet"
        while status != "ACC":
            if ens_name == 0:
                success, new_path, status = shoot(
                    ensemble,
                    path,
                    engine,
                    shooting_point=system,
                    start_cond=("R",),
                )
            else:
                success, new_path, status = shoot(
                    ensemble, path, engine, shooting_point=system
                )
            print(
                f"Generated a path in {ens_name} with len {new_path.length} "
                + f"and status {status}."
            )
        print("Success!")

        # write order.txt, traj.txt, energy.txt
        path_dir = exe_dir / str(ens_name)
        path_dir.mkdir(exist_ok=True)
        io = PathStorage()
        io.output_path_files(int(ens_name), [new_path, status], str(path_dir))
        # move trajectories to correct load/i folder
        traj_dir = path_dir / "accepted"
        traj_dir.mkdir(exist_ok=True)
        for traj in new_path.adress:
            traj_file = Path(traj)
            traj_file.rename(traj_dir / traj_file.name)

    # remove redundant files
    for fname in exe_dir.glob("*"):
        if fname.is_file():
            fname.unlink()

    print(f"Load paths generated in {exe_dir}")


create_initial_paths()
