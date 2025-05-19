# Copyright (c) 2023, infRETIS Development Team.
# Distributed under the LGPLv2.1+ License. See LICENSE for more info.
"""A AMS external MD integrator interface.

This module defines a class for using AMS as an external engine.

Important classes defined here
------------------------------

AMSEngine (:py:class:`.AMSEngine`)
    A class responsible for interfacing AMS.
"""
import copy
import logging
import os
import time
import weakref
from typing import Any, Dict, Tuple

import numpy as np
from scm.plams.interfaces.adfsuite.ams import AMSJob
from scm.plams.interfaces.adfsuite.amsworker import AMSWorker
from scm.plams.tools.units import Units
from scm.plams.trajectories.rkffile import RKFTrajectoryFile

from infretis.classes.engines.enginebase import EngineBase
from infretis.classes.engines.engineparts import (
    box_matrix_to_list,
)
from infretis.classes.formatter import FileIO
from infretis.classes.path import Path as InfPath
from infretis.classes.system import System

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


class AMSEngine(EngineBase):  # , metaclass=Singleton):
    """
    A class for interfacing AMS.

    This class defines the interface to AMS.

    Attributes
    ----------
    input_path : string
        The directory where the input files are stored.
    input_files : dict of strings
        The names of the input files. We expect to find the keys
        ``'conf'``, ``'input'`` ``'topology'``.
    ext_time : float
        The time to extend simulations by. It is equal to
        ``timestep * subcycles``.
    """

    def __init__(self, input_path, timestep, subcycles):
        """Set up the AMS engine.

        Parameters
        ----------
        input_path : string
            The absolute path to where the input files are stored.
        timestep : float
            The time step used in the GROMACS MD simulation.
        subcycles : integer
            The number of steps each GROMACS MD run is composed of.

        """
        super().__init__("AMS engine", timestep, subcycles)
        self.ext = "rkf"

        # Units of AMS output, must correspond to set PyRETIS units
        self.ene_unit = "kJ/mol"
        self.dist_unit = "nm"
        self.time_unit = "ps"
        self.name = "ams"

        # Store MD states
        self.states = {}
        self.oldstates = []
        # self.ens_name = ''

        # If input trajectories have different boxsize set to True:
        self.update_box = True

        # Add input path and the input files:
        self.input_path = os.path.abspath(input_path)

        # Expected input files
        self.input_files = {
            "input": "ams.inp",
        }

        # Read AMS input
        inpf = os.path.join(self.input_path, self.input_files["input"])
        inp = open(inpf).read()
        job = AMSJob.from_input(inp)
        settings = job.settings
        molecule = job.molecule[""]
        if self.update_box:
            self.molecule_lattice = molecule.lattice  # Check input settings
        self.temperature = (
            settings.input.ams.moleculardynamics.initialvelocities.temperature
        )
        if len(self.temperature) == 0:
            logger.error("AMS: InitialVelocities Temperature was not set!")
            quit(1)
        self.temperature = float(self.temperature)
        ams_timestep = settings.input.ams.moleculardynamics.timestep
        if len(ams_timestep) == 0:
            # Default timestep in AMS, but unknown it here
            logger.error("AMS: Timestep was not set!")
            quit(1)
        ams_timestep = float(ams_timestep) * Units.conversion_ratio(
            "fs", self.time_unit
        )
        if timestep != ams_timestep:
            logger.error("Mismatch between AMS and Pyretis timestep!")
            quit(1)

        self.random_velocities_method = None
        random_velocities_method = (
            settings.input.ams.moleculardynamics.initialvelocities.randomvelocitiesmethod
        )
        if len(random_velocities_method) > 0:
            logger.info(
                'AMS setting velocity generation method to "%s"',
                random_velocities_method,
            )
            self.random_velocities_method = random_velocities_method
        ams_dir = "."
        # Start AMS worker
        self.worker = AMSWorker(
            settings,
            workerdir_root=ams_dir,
            keep_crashed_workerdir=True,
            always_keep_workerdir=True,
        )
        self._finalize = weakref.finalize(self, self.worker.stop)

    def step(self, system, name, set_trajfile=True, set_step_to_zero=False):
        """Perform a single step with AMS.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The system we are integrating.
        name : string
            To name the output files from the AMS step.
        set_trajfile : logical
            Optional, True if new output file should be opened
        set_step_to_zero : logical
            Optional, True if new MD step should start from
            number zero
        Returns
        -------
        out : string
            The name of the output configuration, obtained after
            completing the step.

        """
        state, idx = system.config

        if idx == -1:
            prev_ams_state = state  # state already contains exe_dir
            new_ams_state = os.path.join(self.exe_dir, name)  # name does not

            new_state = new_ams_state
        else:
            prev_ams_state = (
                state + "_" + str(idx)
            )  # state already contains exe_dir
            new_ams_state = os.path.join(
                self.exe_dir, name + "_" + str(idx + 1)
            )  # name does not
            new_state = state
            idx = idx + 1

        if set_trajfile:
            logger.info("AMS setting output file: %s", new_state)
            if os.path.exists(
                new_state
            ):  # File must never be there before PrepareMD
                self._removefile(new_state, disk_only=True)

            self.worker.PrepareMD(new_state)

        # This might be removed, but nice way to check what is going on
        logger.info(f"AMS step {prev_ams_state} -> {new_ams_state}")

        self.worker.CopyMDState(prev_ams_state, new_ams_state)
        states = self.worker.MolecularDynamics(
            new_ams_state,
            nsteps=self.subcycles,
            trajectorysamplingfrequency=self.subcycles,
            checkpointfrequency=0,
            pipesamplingfrequency=self.subcycles,
            setsteptozero=set_step_to_zero,
        )

        # Update system
        system.set_pos((new_state, idx))
        system.vel_rev = False
        # Here, we are not concerned if we also got the initial state or not
        # from the AMSWorker. Next state is always the last one.
        system.vpot = states[-1].get_potentialenergy(unit=self.ene_unit)
        system.ekin = states[-1].get_kineticenergy(unit=self.ene_unit)

        # Save state
        self._add_state(new_state, states[-1])
        return name

    def _read_configuration(self, filename, idx=-1):
        """Read output from AMS snapshot/trajectory.

        Parameters
        ----------
        filename : string
            The file to read the configuration from.

        idx : integer
            Optional, frame index in trajectory

        Returns
        -------
        box : numpy.array
            The box dimensions.
        xyz : numpy.array
            The positions.
        vel : numpy.array
            The velocities.

        """
        if idx == -1:
            idx = 0

        state = self.states[filename][idx]

        box = state.get_latticevectors(unit=self.dist_unit)

        if len(box) == 0:
            box = [float("inf"), float("inf"), float("inf")]
        else:
            box = box_matrix_to_list(box)
        xyz = state.get_coords(unit=self.dist_unit)
        vel = state.get_velocities(
            dist_unit=self.dist_unit, time_unit=self.time_unit
        )
        return xyz, vel, box, None

    def set_mdrun(self, md_items):
        """
        Set up the molecular dynamics run with the given parameters.

        Parameters:
        md_items (dict): A dictionary containing the following keys
        that are of importance here:
            - "exe_dir" (str): The directory where the executable is located.
            - "ens" (dict): A dictionary containing ensemble information
                with the key:
                - "ens_name" (str): The name of the ensemble.

        This method performs the following actions:
        1. Sets the executable directory and ensemble name.
        2. Logs the executable directory information.
        3. Identifies and deletes old states that are no longer in use.
        4. Updates the list of old states to the current states.
        """
        self.exe_dir = md_items["exe_dir"]
        self.ens_name = md_items["ens"]["ens_name"] + "_"
        logger.info(
            f"self.exe_dir {self.exe_dir}"
            + f" md_items['exe_dir'] {md_items['exe_dir']}"
        )
        delete_states = []
        for state in self.oldstates:
            if state in self.states.keys():
                delete_states.append(state)
        for state in delete_states:
            self._deletestate(state)

        self.oldstates = self.states.keys()

    def _reverse_velocities(self, filename, outfile):
        """Reverse velocity in a given snapshot.

        Parameters
        ----------
        filename : string
            The configuration to reverse velocities in.
        outfile : string
            The output file for storing the configuration with
            reversed velocities.

        """
        logger.info("AMS reversing velocities for %s", filename)
        if os.path.exists(
            outfile
        ):  # File must never be there before PrepareMD
            self._removefile(outfile)
        self.worker.PrepareMD(outfile)

        self._copystate(
            filename, outfile
        )  # copy only the state, file will be written later

        # Here we are working with AMS internal representation of velocities,
        # thus, we keep the units a.u./a.u.
        vel = self.states[filename][0].get_velocities(
            dist_unit="au", time_unit="au"
        )
        rev_vel = -1.0 * vel
        self.states[outfile][0]._state["velocities"] = rev_vel
        self.worker.SetVelocities(
            outfile, rev_vel, dist_unit="au", time_unit="au"
        )

        self.worker.MolecularDynamics(outfile, nsteps=0, setsteptozero=True)

    def _extract_frame(self, traj_file, idx, out_file):
        """Extract a frame from a trajectory file.

        Parameters
        ----------
        traj_file : string
            The AMS file to open.
        idx : integer
            The frame number we look for.
        out_file : string
            The file to dump to.

        Note
        ----
        This will only properly work if the frames in the input
        trajectory are uniformly spaced in time.

        """
        if traj_file in self.states:
            logger.info(
                "AMS extracting frame: %s, %i -> %s", traj_file, idx, out_file
            )
            self._copystate(traj_file, out_file, idx=idx)
            self.worker.PrepareMD(out_file)
            self.worker.MolecularDynamics(
                out_file, nsteps=0, setsteptozero=True
            )  # Writes traj file
        else:
            logger.info(
                "AMS extracting frame from disk: %s, %i -> %s",
                traj_file,
                idx,
                out_file,
            )
            rkf = RKFTrajectoryFile(traj_file)
            rkf.store_mddata()
            seconds = 0 
            molecule = rkf.get_plamsmol()
            if len(molecule.atoms) == 0:
                print(
                    f"Waiting for RKFTrajectoryFile to be ready: {traj_file}"
                )
                while len(molecule.atoms) == 0:
                    time.sleep(1)
                    seconds += 1
                    print(seconds, traj_file)
                    rkf = RKFTrajectoryFile(traj_file)
                    rkf.store_mddata()
                    molecule = rkf.get_plamsmol()
        
                print(
                    f"Waited {seconds} seconds for RKFTrajectoryFile to be ready"
                )
            rkf.read_frame(idx, molecule=molecule)
            if self.update_box:
                molecule.lattice = self.molecule_lattice
            if os.path.exists(
                out_file
            ):  # file must never be there before PrepareMD
                self._removefile(out_file)
            if out_file in self.states:
                self._deletestate(out_file)
            self.worker.PrepareMD(out_file)
            

            try: 
                self.worker.CreateMDState(out_file, molecule)
            except Exception as e:
                if "MD state with given title already exists" in str(e):
                    print("MD state with given title already exists: ", out_file)
                    logger.error(
                        "AMS error in CreateMDState: %s", str(e)
                    )
                    self.worker.DeleteMDState(out_file)
                    self.worker.CreateMDState(out_file, molecule)

                else:
                    raise e
            if "Velocities" in rkf.mddata:
                logger.info(
                        "Copying Velocities: %s -> %s", traj_file, out_file
                    )
                vel = rkf.mddata["Velocities"]
                vel = np.reshape(
                    vel, (-1, 3)
                )  # RKFTrajectoryFile returns 1D array
                self.worker.SetVelocities(
                    out_file, vel, dist_unit="bohr", time_unit="fs"
                )  # Units used in rkf file
            state = self.worker.MolecularDynamics(
                out_file, nsteps=0, setsteptozero=True
            )  # Also writes frame into out_file
            self._add_state(out_file, state[0])

    def _propagate_from(
        self,
        name: str,
        path: InfPath,
        system: System,
        ens_set: Dict,
        msg_file: FileIO,
        reverse: bool = False,
    ) -> Tuple[bool, str]:
        """
        Propagate with AMS from the current system configuration.

        Here, we assume that this method is called after the propagate()
        has been called in the parent. The parent is then responsible
        for reversing the velocities and also for setting the initial
        state of the system.

        Parameters
        ----------
        name : string
            A name to use for the trajectory we are generating.
        path : object like :py:class:`pyretis.core.path.PathBase`
            This is the path we use to fill in phase-space points.
        ensemble: dict
            It contains:

            * `system`: object like :py:class:`.System`
              The system object gives the initial state for the
              integration. The initial state is stored and the system is
              reset to the initial state when the integration is done.
            * `order_function`: object like :py:class:`.OrderParameter`
              The object used for calculating the order parameter.
            * `interfaces`: list of floats
              These interfaces define the stopping criterion.

        msg_file : object like :py:class:`.FileIO`
            An object we use for writing out messages that are useful
            for inspecting the status of the current propagation.
        reverse : boolean, optional
            If True, the system will be propagated backward in time.

        Returns
        -------
        success : boolean
            This is True if we generated an acceptable path.
        status : string
            A text description of the current status of the propagation.

        """
        status = f"propagating with AMS (reverse = {reverse})"
        logger.info(status)
        success = False
        interfaces = ens_set["interfaces"]
        left, _, right = interfaces

        # Get the current order parameter:
        order = self.calculate_order(system)
        msg_file.write(
            f'# Initial order parameter: {" ".join([str(i) for i in order])}'
        )

        kin_enes = []
        pot_enes = []
        traj_file = os.path.join(self.exe_dir, name + "." + self.ext)
        # First, process input snapshot
        initial = system.config[0]
        self._copystate(
            initial, traj_file
        )  # Copy only state. Traj file will be written in the loop
        logger.info(
            f'AMS internal traj renaming: {traj_file} -> {traj_file+"_0"}'
        )
        self.worker.RenameMDState(traj_file, traj_file + "_0")

        # Then, add snapshot to path and propagate further if necessary
        set_trajfile = True
        set_step_to_zero = True
        for i in range(path.maxlen):
            if i == 1:
                set_trajfile = False
                set_step_to_zero = False

            # Calculate the order parameter using the current system:
            system.vel_rev = True
            system.set_pos((traj_file, i))
            out = self._read_configuration(traj_file, idx=i)
            order = self.calculate_order(
                system, xyz=out[0], vel=out[1], box=out[2]
            )
            msg_file.write(f'{i} {" ".join([str(j) for j in order])}')

            snapshot = {
                "order": order,
                "config": (traj_file, i),
                "vel_rev": reverse,
            }

            phase_point = self.snapshot_to_system(system, snapshot)
            status, success, stop, _ = self.add_to_path(
                path, phase_point, left, right
            )

            kin_enes.append(
                self.states[traj_file][i].get_kineticenergy(unit=self.ene_unit)
            )
            pot_enes.append(
                self.states[traj_file][i].get_potentialenergy(
                    unit=self.ene_unit
                )
            )

            logger.info("OP: %f in frame %s %i", order[0], traj_file, i)

            msg_file.flush()
            if stop:
                logger.info(
                    "AMS propagation ended at %i. Reason: %s", i, status
                )
                if i == 0:
                    # Write the traj file if no MD is going to be done
                    self.worker.PrepareMD(traj_file)
                    self.worker.MolecularDynamics(
                        traj_file + "_0", nsteps=0, setsteptozero=True
                    )
                break
            self.step(
                system,
                traj_file,
                set_trajfile=set_trajfile,
                set_step_to_zero=set_step_to_zero,
            )
        logger.info("AMS propagation done, obtaining energies")
        path.update_energies(kin_enes, pot_enes)
        msg_file.write("# Propagation done.")
        msg_file.flush()
        return success, status

    def modify_velocities(
        self, system: System, vel_settings: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Modify the velocities of the current state.

        This method will modify the velocities of a time slice.

        Parameters
        ----------
        ensemble : dict
            It contains:

            * `system`: object like :py:class:`.System`
              This is the system that contains the particles we are
              investigating.

        vel_settings: dict
            It contains:

            * `sigma_v`: numpy.array, optional
              These values can be used to set a standard deviation (one
              for each particle) for the generated velocities.
            * `aimless`: boolean, optional
              Determines if we should do aimless shooting or not.
            * `momentum`: boolean, optional
              If True, we reset the linear momentum to zero after
              generating.
            * `rescale or rescale_energy`: float, optional
              In some NVE simulations, we may wish to re-scale the
              energy to a fixed value. If `rescale` is a float > 0,
              we will re-scale the energy (after modification of
              the velocities) to match the given float.

        Returns
        -------
        dek : float
            The change in the kinetic energy.
        kin_new : float
            The new kinetic energy.

        """
        rescale = vel_settings.get(
            "rescale_energy", vel_settings.get("rescale")
        )

        if rescale is not None and rescale is not False and rescale > 0:
            msgtxt = "AMS engine does not support energy re-scale."
            logger.error(msgtxt)
            raise NotImplementedError(msgtxt)
        kin_old = system.ekin
        if vel_settings.get("aimless", False):
            state_name, idx = system.config
            logger.info(
                "Generating velocities for %s, idx=%s", state_name, idx
            )
            prefix = (
                self.ens_name
                + str(os.getpid())
                + "_"
                + str(int(time.time() * 1_000_000) % 1_000_000).zfill(6)
            )
            genvel = os.path.join(self.exe_dir, f"genvel_{prefix}." + self.ext)
            if state_name in self.states:
                # If kicking from new MD state, prepare it
                self._copystate(state_name, genvel, idx=idx)
            else:
                self._extract_frame(state_name, idx, genvel)
            state = self.worker.GenerateVelocities(
                genvel,
                self.temperature,
                randomvelocitiesmethod=self.random_velocities_method,
                setsteptozero=True,
            )

            # Can be removed, but nice way to check velocity generation
            logger.info(
                "AMS Epot: %f Ekin: %f",
                state.get_potentialenergy(unit=self.ene_unit),
                state.get_kineticenergy(unit=self.ene_unit),
            )
            # Write rkf file
            logger.info("AMS setting output file: %s", genvel)
            if os.path.exists(
                genvel
            ):  # File must never be there before PrepareMD
                self._removefile(genvel, disk_only=True)
            self.worker.PrepareMD(genvel)
            # self.worker.MolecularDynamics(genvel, nsteps=0)

            # Update system
            kin_new = state.get_kineticenergy(unit=self.ene_unit)
            system.set_pos((genvel, -1))
            system.vel_rev = False
            system.ekin = kin_new
            system.vpot = state.get_potentialenergy(unit=self.ene_unit)
            self._add_state(genvel, state, rewrite=True)
        else:  # Soft velocity change, from a Gaussian distribution:
            msgtxt = "AMS engine only support aimless shooting!"
            logger.error(msgtxt)
            raise NotImplementedError(msgtxt)
        if vel_settings.get("momentum", False):
            pass
        if kin_old is None or kin_new is None:
            dek = float("inf")
            logger.warning(
                "Kinetic energy not found for previous point."
                "\n(This happens when the initial configuration "
                "does not contain energies.)"
            )
        else:
            dek = kin_new - kin_old
        return dek, kin_new

    def _add_state(self, key, state, rewrite=False):
        if key not in self.states or rewrite:
            self.states[key] = []
            self.states[key].append(state)
        else:
            self.states[key].append(state)
        return

    def _renamestate(self, source, dest):
        if dest in self.states:
            self._deletestate(dest)

        self.states[dest] = self.states[source]
        # Dirty usage of filename to recognize trajectories and snapshots
        # because we do not have system object with index here...
        if "traj" in os.path.basename(source):
            logger.info("AMS Moving traj: %s -> %s", source, dest)
            for i in range(len(self.states[dest])):
                self.worker.RenameMDState(
                    source + "_" + str(i), dest + "_" + str(i)
                )
        else:
            logger.info("AMS Moving snap: %s -> %s", source, dest)
            self.worker.RenameMDState(source, dest)

        del self.states[source]

    def _copystate(self, source, dest, idx=-1):
        if source == dest:
            print(
                "-----------------------------------------------------------------------------"
            )
            print("WARNING: source == dest in ams._copystate")
            print("This should only happen in pytest")
            print(
                "-----------------------------------------------------------------------------"
            )
            pass
        else:
            if dest in self.states:
                self._deletestate(dest)

            if idx == -1:
                logger.info("AMS Copying snap to snap: %s -> %s", source, dest)
                self.states[dest] = [copy.deepcopy(self.states[source][0])]
                self.worker.CopyMDState(source, dest)
            else:
                logger.info(
                    "AMS Copying traj snap to snap: %s, %i -> %s",
                    source,
                    idx,
                    dest,
                )
                self.states[dest] = [copy.deepcopy(self.states[source][idx])]
                self.worker.CopyMDState(source + "_" + str(idx), dest)

    def _deletestate(self, filename):
        """
        Delete the state associated with the given filename.

        This method distinguishes between trajectory and snapshot files based
        on the presence of the substring "traj" in the filename. For trajectory
        files, it deletes multiple states indexed by appending an underscore
        and an index to the filename. For snapshot files, it deletes the state
        directly.

        Args:
            filename (str): The name of the file whose state is to be deleted.

        Raises:
            KeyError: If the filename is not found in the states dictionary.
        """
        # Dirty usage of filename to recognize trajectories and snapshots
        # because we do not have system object with index here...
        if "traj" in os.path.basename(filename):
            logger.info("AMS Deleting traj: %s", filename)
            # self.worker.DeleteMDState(filename)
            # self.oldstates.append(filename)
            for i in range(len(self.states[filename])):
                self.worker.DeleteMDState(filename + "_" + str(i))
        else:
            logger.info("AMS Deleting snap: %s", filename)
            try:
                self.worker.DeleteMDState(filename)
            except Exception as e:
                if "MD state with given title not found" in str(e):
                    print(
                        "MD state with given title not found: ", filename
                    )
                    logger.error(
                        "AMS error in DeleteMDState: %s", str(e)
                    )
                else:
                    print('else: tried but failed but wrong')
                    raise e

        del self.states[filename]

    def _movefile(self, source, dest):
        """
        Move a file from source to destination and update internal states.

        This method moves a file from the specified source path to the
        destination path using the superclass's _movefile method. Additionally,
        it updates the internal states if the source path is present in the
        states.

        Args:
            source (str): The path of the file to be moved.
            dest (str): The destination path where the file should be moved.

        """
        super()._movefile(source, dest)
        # When swapping paths and running from restart,
        # all MD states might not be in memory
        if source in self.states:
            self._renamestate(source, dest)

    def _copyfile(self, source, dest):
        """
        Copy a file from source to destination and updates the state.

        This method first calls the superclass's _copyfile method to perform
        the actual file copy. If the destination file is already in the states,
        it deletes the existing state. Finally, it copies the state from the
        source to the destination.

        Args:
            source (str): The path to the source file.
            dest (str): The path to the destination file.
        """
        super()._copyfile(source, dest)
        if dest in self.states:
            self._deletestate(dest)
        self._copystate(source, dest)

    def _removefile(self, filename, disk_only=False):
        """
        Remove a file from the system and optionally from the internal state.

        This method removes a file by calling the superclass's _removefile
        method. It can also remove the file from the internal state if it
        represents a molecular dynamics (MD) state.

        Args:
            filename (str): The name of the file to be removed.
            disk_only (bool, optional): If True, only remove the file from the
              disk and not from the internal state. Defaults to False.

        Returns:
            None
        """
        super()._removefile(filename)
        # We could use os.remove() directly, but let's be consistent
        if disk_only:
            return
        # We have to check if file represents a MD state,
        # because the method is also called for removing log files
        # or states are not present when restarting
        if filename in self.states:
            self._deletestate(filename)
