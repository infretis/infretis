"""An ASE integrator interface."""

import logging
import os
from pathlib import Path

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

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


class ASEEngine(EngineBase):
    """An ASE engine class."""

    def __init__(
        self,
        timestep: float,
        temperature: float,
        subcycles: int,
        input_path: str,
        integrator: str,
        calculator_settings: dict,
        langevin_friction: float = -1.0,
        langevin_fixcm: float = -1.0,
        exe_path: str | Path = Path(".").resolve(),
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

        # TODO: make this non-manual
        # by reading in from .toml or .py?
        # Create calculator
        self.calc = create_external(calculator_settings, "ase", ["calculate"])

        # integrator stuff
        integrator = integrator.lower()
        integrator_map = {
            "langevin": Langevin,
            "velocityverlet": VelocityVerlet,
        }
        if integrator not in integrator_map.keys():
            raise ValueError(f"{integrator.lower()} not in integrator map.")

        self.Integrator = integrator_map[integrator]
        if integrator == "langevin":
            if langevin_fixcm == -1.0:
                raise ValueError("'langevin_fixcm' not set in [engine]")
            if langevin_friction == -1.0:
                raise ValueError("'langevin_friction' not set in [engine]")
            self.integrator_settings = {
                "timestep": self.timestep * units.fs,
                "temperature_K": self.temperature,
                "friction": langevin_friction * units.fs,
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
        atoms = read(filename)
        if isinstance(atoms, list):
            atoms = atoms[0]
        return (
            atoms.positions,
            atoms.get_velocities(),
            atoms.cell.diagonal(),
            None,
        )

    def set_mdrun(self, md_items: dict) -> None:
        """Set worker stuff if needed."""
        self.exe_dir = md_items["exe_dir"]

    def _propagate_from(
        self,
        name: str,
        path: InfPath,
        system: System,
        ens_set: dict,
        msg_file: FileIO,
        reverse: bool = False,
    ) -> tuple[bool, str]:
        logger.info(f"Propagating with ASE (reverse = {reverse})")
        interfaces = ens_set["interfaces"]
        left, _, right = interfaces

        initial_conf = system.config[0]
        atoms = read(initial_conf)
        if isinstance(atoms, list):
            atoms = atoms[0]
        # TODO: Fix box stuff, now it only takes lengths andn ot angles
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
        traj = Trajectory(traj_file, "w")
        msg_file.write(f"# Trajectory file is: {traj_file}")
        dyn = self.Integrator(atoms, **self.integrator_settings)
        atoms.set_calculator(self.calc)
        # we give the calculator object the system and order
        # information in case it is needed during force calculations
        # using e.g. force mixing with quantis. Can't give it directly
        # because the oderparameter is only calculated every subcycles
        self.calc.inf_system = system
        self.calc.inf_calculate_order = self.calculate_order
        # TODO: check order, here we first calculate the forces of init conf
        # and then continue to the main loop. Is this correct?
        # This way the forces are set in self.calc.results from
        # the given initial configuration
        self.calc.calculate(atoms)
        step_nr = 0
        ekin = []
        vpot = []
        # integrator step is taken at the end of every loop,
        # such that frame 0 is also written
        for i in range(self.subcycles * path.maxlen):
            energy = self.calc.results["energy"]
            forces = self.calc.results["forces"]
            stress = self.calc.results["stress"]
            if (i) % (self.subcycles) == 0:
                ekin.append(atoms.get_kinetic_energy())
                vpot.append(self.calc.results["energy"])
                # NOTE: Writing atoms removes all results from
                # the calculator (and therefore atoms)!
                traj.write(atoms, forces=forces, energy=energy, stress=stress)
                order = self.calculate_order(
                    system,
                    xyz=atoms.positions,
                    vel=atoms.get_velocities(),
                    box=atoms.cell.diagonal(),
                )
                msg_file.write(
                    f'{step_nr} {" ".join([str(j) for j in order])}'
                )
                snapshot = {
                    "order": order,
                    "config": (traj_file, step_nr),
                    "vel_rev": reverse,
                }
                phase_point = self.snapshot_to_system(system, snapshot)
                status, success, stop, add = self.add_to_path(
                    path, phase_point, left, right
                )

                if stop:
                    logger.info(
                        "ASE propagation ended at "
                        f"step {step_nr}. Reason: {status}",
                    )
                    break
                step_nr += 1
            dyn.step(forces=forces)

        msg_file.write("# Propagation done.")
        traj.close()
        path.update_energies(ekin, vpot)
        return success, status

    def modify_velocities(
        self, system: System, vel_settings: dict
    ) -> tuple[float, float]:
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
