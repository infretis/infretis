"""A TurtleMD integrator interface.

This module defines a class for using the TurtleMD.

"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from turtlemd.integrators import (
    LangevinInertia,
    LangevinOverdamped,
    VelocityVerlet,
    Verlet,
)
from turtlemd.potentials.lennardjones import LennardJonesCut
from turtlemd.potentials.well import DoubleWell
from turtlemd.simulation import MDSimulation
from turtlemd.system.box import Box as TBox
from turtlemd.system.particles import Particles as TParticles
from turtlemd.system.system import System as TSystem

from infretis.classes.engines.cp2k import kinetic_energy, reset_momentum
from infretis.classes.engines.enginebase import EngineBase
from infretis.classes.engines.engineparts import (
    convert_snapshot,
    read_xyz_file,
    write_xyz_trajectory,
)

if TYPE_CHECKING:  # pragma: no cover
    from infretis.classes.formatter import FileIO
    from infretis.classes.path import Path
    from infretis.classes.system import System


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())

POTENTIAL_MAPS = {"doublewell": DoubleWell, "lennardjones": LennardJonesCut}

INTEGRATOR_MAPS = {
    "langevininertia": LangevinInertia,
    "langevinoverdamped": LangevinOverdamped,
    "velocityverlet": VelocityVerlet,
    "verlet": Verlet,
}


class TurtleMDEngine(EngineBase):
    """Interface the TurtleMD engine.

    To do:
        * Add support for multiple potentials?
        * Velocity generation adds needs to account for
          the dimensionality of the system
    """

    def __init__(
        self,
        timestep: float,
        subcycles: int,
        temperature: float,
        boltzmann: float,
        integrator: dict[str, Any],
        potential: dict[str, Any],
        particles: dict[str, Any],
        box: dict[str, Any],
    ):
        """Initialize the TurtleMD engine.

        Args:
            timestep: The simulation timestep.
            subcycles: The number of subcycles to execute per InfRetis step.
            temperature: The temperature of the simulation.
            boltzmann: The value of Boltzmanns constant (kB).
            integrator: The name of the integrator to use for
                TurtleMD.
            potential: The name of the potential to use for TurtleMD.
            particles: The mass and name of the particles in the system.
            box: Definition of the simulation box.
        """
        self.temperature = temperature
        self.timestep = timestep
        self.subcycles = subcycles
        self.name = "turtlemd"

        super().__init__(
            "TurtleMD internal engine", self.timestep, self.subcycles
        )

        self.boltzmann = boltzmann
        self.beta = 1 / (self.boltzmann * self.temperature)

        self.subcycles = subcycles

        self.integrator = INTEGRATOR_MAPS[integrator["class"].lower()]
        self.integrator_settings = integrator["settings"]
        potential_class = POTENTIAL_MAPS[potential["class"].lower()]
        self.potential = [potential_class(**potential["settings"])]

        if integrator["class"].lower() in [
            "langevininertia",
            "langevinoverdamped",
        ]:
            # TODO:
            print("do something wrt random gen. so that only")
            print("langevin integrato needs it")

        self.dim = self.potential[0].dim

        self.mass = np.array(particles["mass"])
        self.names = particles["name"]

        self.particles = TParticles(dim=self.dim)
        self.box = TBox(**box)
        for i, pos in enumerate(particles["pos"]):
            self.particles.add_particle(
                pos, mass=self.mass[i], name=self.names[i]
            )
        self.system = TSystem(self.box, self.particles, self.potential)

    def _extract_frame(self, traj_file: str, idx: int, out_file: str) -> None:
        """
        Extract a frame from a trajectory file.

        This method is used by `self.dump_config` when we are
        dumping from a trajectory file. It is not used if we are
        dumping from a single config file.

        Args:
            traj_file: The trajectory file to dump from.
            idx: The frame number we look for.
            out_file: The file to dump to.
        """
        for i, snapshot in enumerate(read_xyz_file(traj_file)):
            if i == idx:
                box, xyz, vel, names = convert_snapshot(snapshot)
                if os.path.isfile(out_file):
                    logger.debug("TurtleMD will overwrite %s", out_file)
                write_xyz_trajectory(
                    out_file, xyz, vel, names, box, append=False
                )
                return
        logger.error(
            "TurtleMD could not extract index %i from %s!", idx, traj_file
        )

    def _propagate_from(
        self,
        name: str,
        path: Path,
        system: System,
        ens_set: dict[str, Any],
        msg_file: FileIO,
        reverse: bool = False,
    ) -> tuple[bool, str]:
        """Propagate the equations of motion from the given system.

        We assume the following:
            * Box does not change (constant volume simulation)
            * Box is orthogonal
        """
        status = f"propagating with TurtleMD (reverse = {reverse})"
        interfaces = ens_set["interfaces"]
        logger.debug(status)
        success = False
        left, _, right = interfaces
        # Get positions and velocities from the input file.
        initial_conf = system.config[0]
        # these variables will be used later
        pos, vel, box, atoms = self._read_configuration(initial_conf)
        # initialize turtlemd system
        particles = TParticles(dim=self.dim)
        for i in range(self.particles.npart):
            particles.add_particle(
                pos[i][: self.dim],
                vel=vel[i][: self.dim],
                mass=self.particles.mass[i],
                name=self.particles.name[i],
            )
        tmd_system = TSystem(
            box=self.box, particles=particles, potentials=self.potential
        )
        if hasattr(self, "rgen"):
            seed = self.rgen.integers(0, 1e9)
        else:
            raise ValueError("Missing random generator!")
        tmd_simulation = MDSimulation(
            system=tmd_system,
            integrator=self.integrator(
                timestep=self.timestep, **self.integrator_settings, seed=seed
            ),
            steps=path.maxlen * self.subcycles,
        )
        order = self.calculate_order(
            system, xyz=pos, vel=vel, box=tmd_system.box.length
        )

        traj_file = os.path.join(self.exe_dir, f"{name}.{self.ext}")
        # Create a message file with some info about this run:
        msg_file.write(
            f'# Initial order parameter: {" ".join([str(i) for i in order])}'
        )
        msg_file.write(f"# Trajectory file is: {traj_file}")
        logger.debug("Running TurtleMD")
        step_nr = 0
        # dict for storing ene  rgies
        thermo = defaultdict(list)
        # loop over n subcycles
        # The first step of the loop is the initial phase point, i.e., for i=0
        # turtlemd does not integrate the equations of motion, it just
        # returns the initial system
        for i, step in enumerate(tmd_simulation.run()):
            if (i) % (self.subcycles) == 0:
                thermoi = step.thermo(self.boltzmann)
                for key, val in thermoi.items():
                    thermo[key].append(val)
                # update coordinates, velocities and box
                # for the relevant dimensions. We need this here
                # because we use xyz format for trajecories, which has
                # 3 dimensions for coords, vel and the box.
                pos[:, : self.dim] = tmd_system.particles.pos
                vel[:, : self.dim] = tmd_system.particles.vel
                if box is not None:
                    box[: self.dim] = tmd_system.box.length
                write_xyz_trajectory(
                    traj_file, pos, vel, atoms, box, step=step_nr
                )
                order = self.calculate_order(
                    system,
                    xyz=tmd_system.particles.pos,
                    vel=tmd_system.particles.vel,
                    box=tmd_system.box.length,
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
                    logger.debug(
                        "TurtleMD propagation ended at %i. Reason: %s",
                        step_nr,
                        status,
                    )
                    break
                step_nr += 1

        msg_file.write("# Propagation done.")
        ekin = np.array(thermo["ekin"]) * tmd_system.particles.npart
        vpot = np.array(thermo["vpot"]) * tmd_system.particles.npart
        path.update_energies(ekin, vpot)
        return success, status

    @staticmethod
    def _read_configuration(
        filename: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str]]:
        """Read TurtleMD output configuration.

        This method reads the specified TurtleMD output configuration and
        extracts the configuration which includes the positions, velocities,
        box dimensions and particle names. It is used when calculating the
        order parameter from a TurtleMD simulation.

        Args:
            filename: The path to the file to read the configuration from.

        Returns:
            A tuple containing:
                - xyz: An array of atomic positions.
                - vel: An array of atomic velocities.
                - box: An array of box dimensions or None if not available.
                - names: A list of atom names found in the file.
        """
        for snapshot in read_xyz_file(filename):
            box, xyz, vel, names = convert_snapshot(snapshot)
            return xyz, vel, box, names
        raise ValueError("Missing TurtleMD configuration")

    def set_mdrun(self, md_items: dict[str, Any]) -> None:
        """Set the execute directory."""
        # TODO: REMOVE OR RENAME?
        self.exe_dir = md_items["exe_dir"]

    def _reverse_velocities(self, filename: str, outfile: str) -> None:
        """Reverse velocities in the given snapshot.

        Args:
            filename: The path to the file containing the configuration
                to reverse the velocities of.
            outfile : The path to the output file for storing the
                configuration with reversed velocities.
        """
        xyz, vel, box, names = self._read_configuration(filename)
        write_xyz_trajectory(
            outfile, xyz, -1.0 * vel, names, box, append=False
        )

    def modify_velocities(
        self, system: System, vel_settings: dict[str, Any]
    ) -> tuple[float, float]:
        """Modify the velocities of all particles.

        This method modifies the velocity of all particles. Note that default
        removes the center of mass motion, thus, we need to rescale the
        momentum to zero by default.

        Args:
            system: The system whose particle velocities are to be modified.
            vel_settings: A dictionary containing settings for
                velocity modification.

        Returns:
            A tuple containing:
                - dek: The change in kinetic energy as a result of
                    the velocity modification.
                - kin_new: The new kinetic energy of the system.

        Raises:
            NotImplementedError: If the 'rescale_energy' option is
                set but not implemented.
        """
        mass = self.mass
        beta = self.beta
        rescale = vel_settings.get(
            "rescale_energy", vel_settings.get("rescale")
        )
        pos = self.dump_frame(system)
        xyz, vel, box, atoms = self._read_configuration(pos)
        # to-do: retrieve system.vpot from previous energy file.
        if None not in ((rescale, system.vpot)) and rescale is not False:
            print("Rescale")
            if rescale > 0:
                kin_old = rescale - system.vpot
                do_rescale = True
            else:
                print("Warning")
                logger.warning("Ignored re-scale 6.2%f < 0.0.", rescale)
                return 0.0, kinetic_energy(vel, mass)[0]
        else:
            kin_old = kinetic_energy(vel, mass)[0]
            do_rescale = False
        if vel_settings.get("aimless", False):
            vel, _ = self.draw_maxwellian_velocities(vel, mass, beta)
        else:
            dvel, _ = self.draw_maxwellian_velocities(
                vel, mass, beta, sigma_v=vel_settings["sigma_v"]
            )
            vel += dvel
        # make reset momentum the default
        vel_settings["zero_momentum"] = False
        if vel_settings.get("zero_momentum", True):
            vel = reset_momentum(vel, mass)
        if do_rescale:
            # system.rescale_velocities(rescale, external=True)
            raise NotImplementedError(
                "Option 'rescale_energy' is not implemented yet."
            )
        conf_out = os.path.join(self.exe_dir, f"genvel.{self.ext}")
        write_xyz_trajectory(conf_out, xyz, vel, atoms, box, append=False)
        kin_new = kinetic_energy(vel, mass)[0]
        system.config = (conf_out, 0)
        system.ekin = kin_new
        if kin_old == 0.0:
            dek = float("inf")
            logger.debug(
                "Kinetic energy not found for previous point."
                "\n(This happens when the initial configuration "
                "does not contain energies.)"
            )
        else:
            dek = kin_new - kin_old
        return dek, kin_new
