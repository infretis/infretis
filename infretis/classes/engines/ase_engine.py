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

from infretis.classes.engines.enginebase import EngineBase
from infretis.classes.formatter import FileIO
from infretis.classes.path import Path as InfPath
from infretis.classes.system import System
from infretis.core.core import create_external

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


class ASEEngine(EngineBase):
    def __init__(
        self,
        timestep: float,
        temperature: float,
        subcycles: int,
        input_path: str,
        calculator_settings: str,
        exe_path: str | Path = Path(".").resolve(),
    ):
        super().__init__("ASE external engine", timestep, subcycles)

        self.timestep = timestep
        self.subcycles = subcycles
        self.temperature = temperature
        self.input_path = exe_path / input_path
        self.ext = "traj"

        # TODO: make this non-manual
        # by reading in from .toml or .py?
        read(self.input_path / "initial.traj")
        Calc = create_external(calculator_settings, "ase", ["calculate"])
        self.calc = Calc(timestep, temperature)
        self.Integrator = Langevin
        self.integrator_settings = {
            "timestep": self.timestep * units.fs,
            "temperature_K": self.temperature,
            "friction": 0.01 * units.fs,
        }

    def _extract_frame(self, traj_file: str, idx: int, out_file: str) -> None:
        traj = Trajectory(traj_file)
        atoms = traj[idx]
        traj.close()
        write(out_file, atoms)

    def _read_configuration(
        self,
        filename: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        atoms = read(filename)
        return atoms.positions, atoms.get_velocities(), atoms.cell.diagonal()

    def set_mdrun(self, md_items: dict) -> None:
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
        step_nr = 0
        ekin = []
        vpot = []
        # integrator step is taken at the end of every loop,
        # such that frame 0 is also written
        for i in range(self.subcycles * path.maxlen):
            if (i) % (self.subcycles) == 0:
                # Maybe use Trajectory here instead of write()
                traj.write(atoms)
                ekin.append(atoms.get_kinetic_energy())
                vpot.append(atoms.get_potential_energy())
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
                        "ASE propagation ended at {step_nr}. Reason: {status}",
                    )
                    break
            dyn.step()
            step_nr += 1

        msg_file.write("# Propagation done.")
        traj.close()
        path.update_energies(ekin, vpot)
        return success, status

    def modify_velocities(
        self, system: System, vel_settings: dict
    ) -> tuple[float, float]:
        fname = self.dump_frame(system)
        atoms = read(fname)
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
        vel = atoms.get_velocities()
        atoms.set_velocities(-vel)
        write(outfile, atoms)
