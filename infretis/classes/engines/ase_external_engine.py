"""An ASE integrator interface."""

import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
)
from ase.md.verlet import VelocityVerlet

from infretis.classes.engines.enginebase import EngineBase
from infretis.classes.formatter import FileIO
from infretis.classes.path import Path as InfPath
from infretis.classes.system import System
from infretis.core.core import create_external

import pickle
from time import sleep
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())

def dump_stuff(names, objects, cwd):
    for name, objct in zip(names, objects):
        if cwd:
            name = os.path.join(cwd, name)
        with open(name, "wb") as f:
            pickle.dump(objct, f)

def read_stuff(name, cwd):
    if cwd:
        name = os.path.join(cwd, name)
    with open(name, "rb") as f:
        return pickle.load(f)

class ASEExternalEngine(EngineBase):
    """An ASE external engine class."""

    def __init__(
        self,
        timestep: float,
        temperature: float,
        subcycles: int,
        input_path: str,
        integrator: str,
        calculator_settings: Dict,
        langevin_friction: float = -1.0,
        langevin_fixcm: float = -1.0,
        python: str = "python",
        exe_path: Union[str, Path] = Path(".").resolve(),
        sleep: float = 0.1,
    ):
        """
        Initialize the ase engine.

        langevin_fixcm: removes center of mass motion. Should not be used for
            low dimensional systems like double well.

        """
        super().__init__("ASE external engine", timestep, subcycles)

        self.timestep = timestep
        self.subcycles = subcycles
        self.temperature = temperature
        self.input_path = Path(exe_path) / input_path
        self.ext = "traj"
        self.name = "ase"
        self.python = python
        self.sleep = sleep

        # TODO: make this non-manual
        # by reading in from .toml or .py?
        # Create calculator

        # integrator stuff
        integrator = integrator.lower()
        integrator_map = {
            "langevin": Langevin,
            "velocityverlet": VelocityVerlet,
        }
        if integrator not in integrator_map.keys():
            raise ValueError(f"{integrator} not in integrator map.")

        self.Integrator = integrator_map[integrator]
        self.integrator_name = integrator
        if integrator == "langevin":
            if langevin_fixcm == -1.0:
                raise ValueError("'langevin_fixcm' not set in [engine]")
            if langevin_friction == -1.0:
                raise ValueError("'langevin_friction' not set in [engine]")
            self.integrator_settings = {
                "timestep": self.timestep * units.fs,
                "temperature_K": self.temperature,
                "friction": langevin_friction / units.fs,
                "fixcm": langevin_fixcm,
            }
        elif integrator == "velocityverlet":
            self.integrator_settings = {
                "timestep": self.timestep * units.fs,
            }

        self.kb = 8.61733326e-5  # eV/K
        self._beta = 1 / (self.temperature * self.kb)

    def _extract_frame(self, traj_file: str, idx: int, out_file: str) -> None:
        traj = Trajectory(traj_file)
        atoms = traj[idx]
        traj.close()
        write(out_file, atoms)

    def _read_configuration(
        self,
        filename: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, None]:
        atoms = read(filename)
        if isinstance(atoms, list):
            atoms = atoms[0]
        return (
            atoms.positions,
            atoms.get_velocities(),
            atoms.cell.diagonal(),
            None,
        )

    def set_mdrun(self, md_items: Dict) -> None:
        """Set worker stuff if needed."""
        self.exe_dir = md_items["exe_dir"]

    def _propagate_from(
        self,
        name: str,
        path: InfPath,
        system: System,
        ens_set: Dict,
        msg_file: FileIO,
        reverse: bool = False,
    ) -> Tuple[bool, str]:
        logger.info(f"Propagating with ASE (reverse = {reverse})")
        interfaces = ens_set["interfaces"]
        left, _, right = interfaces

        initial_conf = system.config[0]
        atoms = read(initial_conf)
        if isinstance(atoms, list):
            atoms = atoms[0]
        # TODO: Fix box stuff, now it only takes lengths and not angles
        order = self.calculate_order(
            system,
            xyz=atoms.positions,
            vel=atoms.get_velocities(),
            box=atoms.cell.diagonal(),
        )

        msg_file.write(
            f'# Initial order parameter: {" ".join([str(i) for i in order])}'
        )
        traj_file = os.path.join(self.exe_dir, f"{name}.traj")
        msg_file.write(f"# Trajectory file is: {traj_file}")
        msg_file.close()
        cwd = self.exe_dir
        dump_stuff(
                ["system", "path", "ens_set", "Integrator", "int_set", "reverse", "left", "right"],
                [system, path, ens_set, self.Integrator, self.integrator_settings, reverse, left, right],
                cwd,
                )
        sfile = os.path.join(self.exe_dir, "INFINITY_START")
        with open(sfile, "w") as w:
            w.write(f"{initial_conf} {self.subcycles} {traj_file} {cwd} {msg_file.filename} {self.input_path}")

        while os.path.exists(sfile) and not os.path.exists("success") and not os.path.exists("status"):
            sleep(0.5)

        #   cmd2 = " ".join(cmd)
        #   logger.debug(f"Executing {cmd2}.")

        #   out_name = "stdout.txt"
        #   err_name = "stderr.txt"

        #   if cwd:
        #       out_name = os.path.join(cwd, out_name)
        #       err_name = os.path.join(cwd, err_name)

        path_new = read_stuff("path", cwd)
        for phasepoint in path_new.phasepoints:
            path.append(phasepoint)

        success = read_stuff("success", cwd)
        status = read_stuff("status", cwd)
        return success, status

    def modify_velocities(
        self, system: System, vel_settings: Dict
    ) -> Tuple[float, float]:
        """Modify the velocities.

        TODO: what about fixcm with langevin integrator
            and zero_momentum?
        """
        fname = self.dump_frame(system)
        atoms = read(fname)
        if isinstance(atoms, list):
            atoms = atoms[0]
        kin_old = atoms.get_kinetic_energy()

        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)
        kin_new = atoms.get_kinetic_energy()
        if vel_settings.get("zero_momentum", False):
            # TODO: should we preserve temperature or not?
            # The other engines do not bother to preserve the temperature
            Stationary(atoms, preserve_temperature=False)

        conf_out = os.path.join(self.exe_dir, "genvel.traj")
        atoms.write(conf_out)
        system.config = (conf_out, 0)
        system.ekin = kin_new

        if kin_old == 0.0:
            dek = float("inf")
            logger.info(
                "Kinetic energy not found for previous point."
                "\n(This happens when the initial configuration "
                "does not contain energies.)"
            )
        else:
            dek = kin_new - kin_old
        return dek, kin_new

    def _reverse_velocities(self, filename: str, outfile: str) -> None:
        atoms = read(filename)
        if isinstance(atoms, list):
            atoms = atoms[0]
        vel = atoms.get_velocities()
        atoms.set_velocities(-vel)
        write(outfile, atoms)
