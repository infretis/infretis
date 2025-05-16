"""Gromacs engine."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import signal
import struct
import subprocess
from io import BufferedReader, BufferedWriter
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from infretis.classes.engines.enginebase import EngineBase
from infretis.classes.engines.engineparts import (
    box_matrix_to_list,
    look_for_input_files,
    get_ase_atoms,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator
    from io import BufferedReader, BufferedWriter

    from infretis.classes.formatter import FileIO
    from infretis.classes.path import Path as InfPath
    from infretis.classes.system import System

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_GROMACS_MAGIC = 1993
_G96_FMT = "{0:}{1:15.9f}{2:15.9f}{3:15.9f}\n"
_G96_BOX_FMT = "{:15.9f}" * 9 + "\n"
_G96_BOX_FMT_3 = "{:15.9f}" * 3 + "\n"
_GROMACS_MAGIC = 1993
_DIM = 3
_TRR_VERSION = "GMX_trn_file"
_SIZE_FLOAT = struct.calcsize("f")
_SIZE_DOUBLE = struct.calcsize("d")
_HEAD_FMT = "{}13i"
_HEAD_ITEMS = (
    "ir_size",
    "e_size",
    "box_size",
    "vir_size",
    "pres_size",
    "top_size",
    "sym_size",
    "x_size",
    "v_size",
    "f_size",
    "natoms",
    "step",
    "nre",
    "time",
    "lambda",
)
TRR_HEAD_SIZE = 1000
TRR_DATA_ITEMS = (
    "box_size",
    "vir_size",
    "pres_size",
    "x_size",
    "v_size",
    "f_size",
)


class GromacsEngine(EngineBase):
    """A class for interfacing GROMACS.

    Note:
        This method assumes that we use a GROMACS version
        of 5 or later.

    Attributes:
        gmx: The command for executing GROMACS.
        maxwarn: Setting for the GROMACS `grompp -maxwarn` option.
        gmx_format: This string selects the output format for GROMACS.
            Currently, only `"g96"` is supported.
        write_vel: True if we want to output the velocities.
        write_force: True if we want to output the forces.
    """

    def __init__(
        self,
        gmx: str,
        input_path: Union[str, Path],
        timestep: float,
        subcycles: int,
        temperature: float,
        exe_path: Union[str, Path] = Path(".").resolve(),
        maxwarn: int = 0,
        gmx_format: str = "g96",
        write_vel: bool = True,
        write_force: bool = False,
        infretis_genvel: bool = False,
        masses: Union[bool, List, str] = False,
        constraints: Union[bool, List, str] = False,
    ):
        """Set up the GROMACS engine.

        Args:
            gmx: The GROMACS executable.
            input_path: The absolute path to where the input files are stored.
            timestep: The time step used in the GROMACS MD simulation.
            subcycles: The number of steps each GROMACS MD run is composed of.
            exe_path: The absolute path at which the main simulation will be
                run.
            temperature: The temperature during the MD run, used when
                generating velocities and thermostatting by gromacs.
            maxwarn: Setting for the GROMACS `grompp -maxwarn` option.
            gmx_format: The format used for GROMACS configurations.
            write_vel: Determines if GROMACS should write velocities or not.
            write_force: Determines if GROMACS should write forces or not.
            infretis_genvel: If true we generate the velocities
                internally by drawing random numbers from a maxwell-boltzmann
                distribution. Mostly used for testing purposes.
            masses: A list or a txt file with particle masses
            constraints: a python file defining ase constraints that can be
              imported as 'ase_constraints'
        """
        super().__init__("GROMACS engine zamn", timestep, subcycles)
        self.ext = gmx_format
        self.name = "gromacs"
        if self.ext != "g96":
            msg = f"The GROMACS engines now only supports .g96. \
                    Current format: {self.ext}"
            logger.error(msg)
            raise ValueError(msg)
        # Define the GROMACS GMX command:
        self.gmx = gmx
        self.maxwarn = maxwarn
        # Define the energy terms, these are hard-coded, but
        # here we open up for changing that:
        self.energy_terms = self.select_energy_terms("path")
        self.input_path = (Path(exe_path) / input_path).resolve()
        # Set the defaults input files:
        default_files = {
            "conf": f"conf.{self.ext}",
            "input_o": "grompp.mdp",  # "o" = original input file.
            "topology": "topol.top",
        }
        extra_files = {
            "index": "index.ndx",
        }

        file_g = self.input_path / default_files["conf"]
        self.top, _, _, _ = read_gromos96_file(file_g)
        self.top["VELOCITY"] = self.top["POSITION"].copy()

        # Check the presence of the defaults input files or, if absent,
        # try to find them by their expected extension.
        self.input_files = look_for_input_files(
            self.input_path, default_files, extra_files
        )

        # Check the input file and create new input file with
        # consistent settings:
        engine_uses_constraints = False
        check_set = self._read_input_settings(self.input_files["input_o"])
        for key in check_set.keys():
            val = check_set[key]
            if (
                key == "integrator"
                and "md-vv" not in val
                and "LGTM" not in val
            ):
                msg = (
                    "Gromacs integrator should be md-vv with "
                    + "path-sampling. Add a coment LGTM to the relevant"
                    + " line to overwrite this error."
                )
                raise ValueError(msg)

            if key in ["ref-t", "ref_t", "gen-temp", "gen_temp"]:
                msg = (
                    f"Found key '{key}' in {self.input_files['input_o']}. "
                    + "This key is set by infretis and should be removed "
                    + "from this file."
                )
                raise ValueError(msg)

            if key == "constraints":
                engine_uses_constraints = True

            if key in ["tc-grps", "tc_grps"]:
                self.n_tc_grps = len(val.split(";")[0].split())

        # for generating velocities with constraints
        self.infretis_genvel = infretis_genvel

        if self.infretis_genvel:
            self.ase_atoms = get_ase_atoms(
                masses,
                constraints,
                engine_uses_constraints = engine_uses_constraints,
                engine_input_path = input_path
            )

        self.temperature = temperature
        self.kb = 0.0083144621  # kJ/(K*mol)
        self._beta = 1 / (self.temperature * self.kb)

        settings: Dict[str, Union[str, float, int]] = {
            "dt": self.timestep,
            "gen_vel": "no",
            "ref-t": " ".join(
                [str(self.temperature) for i in range(self.n_tc_grps)]
            ),
        }

        for key in (
            "nsteps",
            "nstxout",
            "nstvout",
            "nstfout",
            "nstlog",
            "nstcalcenergy",
            "nstenergy",
        ):
            settings[key] = self.subcycles
        if not write_vel:
            settings["nstvout"] = 0
        if not write_force:
            settings["nstfout"] = 0

        # Create the .mdp file name:
        self.input_files["input"] = os.path.join(
            self.input_path, "infretis.mdp"
        )
        self._modify_input(
            self.input_files["input_o"],
            self.input_files["input"],
            settings,
            delim="=",
        )
        logger.info(
            (
                "Created GROMACS mdp input from %s. You might "
                "want to check the input file: %s"
            ),
            self.input_files["input_o"],
            self.input_files["input"],
        )

        # Generate a tpr file using the input files:
        logger.info('Creating ".tpr" for GROMACS in %s', self.input_path)
        self.exe_dir = str(self.input_path)
        out_files = self._execute_grompp(
            self.input_files["input"], self.input_files["conf"], "topol"
        )

        # This will generate some noise, let's remove files we don't need:
        mdout = os.path.join(self.input_path, out_files["mdout"])
        self._removefile(mdout)
        # We also remove GROMACS backup files after creating the tpr:
        self._remove_gromacs_backup_files(self.input_path)
        # Keep the tpr file.
        self.input_files["tpr"] = os.path.join(
            self.input_path, out_files["tpr"]
        )
        logger.info('GROMACS ".tpr" created: %s', self.input_files["tpr"])

    @staticmethod
    def select_energy_terms(terms: str = "path") -> bytes:
        """Select energy terms to extract from GROMACS.

        Args:
            terms: This string selects the terms to extract.
        """
        allowed_terms = {
            "full": (
                "\n".join(
                    (
                        "Potential",
                        "Kinetic-En.",
                        "Total-Energy",
                        "Temperature",
                        "Pressure",
                    )
                )
            ).encode(),
            "path": b"Potential\nKinetic-En.\nTotal-Energy\nTemperature",
        }
        if terms not in allowed_terms:
            return allowed_terms["path"]
        return allowed_terms[terms]

    def _execute_grompp(
        self, mdp_file: Union[str, Path], config: Union[str, Path], deffnm: str
    ) -> Dict[str, str]:
        """Execute the GROMACS preprocessor.

        Args:
            mdp_file: The path to the mdp file.
            config: The path to the GROMACS config file to use as input.
            deffnm: A string used to name the GROMACS output files.

        Returns:
            A dict with the file names created by the GROMACS
            preprocessor.
        """
        topol = self.input_files["topology"]
        tpr = f"{deffnm}.tpr"
        cmd = [
            self.gmx,
            "grompp",
            "-f",
            str(mdp_file),
            "-c",
            str(config),
            "-p",
            str(topol),
            "-o",
            str(tpr),
        ]
        cmd = shlex.split(" ".join(cmd))
        if "index" in self.input_files:
            cmd.extend(["-n", str(self.input_files["index"])])
        if self.maxwarn > 0:
            cmd.extend(["-maxwarn", str(self.maxwarn)])
        self.execute_command(cmd, cwd=self.exe_dir)
        out_files = {"tpr": tpr, "mdout": "mdout.mdp"}
        return out_files

    def _execute_mdrun(self, tprfile: str, deffnm: str) -> Dict[str, str]:
        """Execute GROMACS mdrun, e.g., `gmx mdrun`.

        Note:
            This method assumes that we are *not* doing a continuation.

        Args:
            tprfile: The .tpr file to use for executing GROMACS.
            deffnm: To give the GROMACS simulation a name.

        Returns:
            A dict with the output file names created by `mdrun`.
        """
        confout = f"{deffnm}.{self.ext}"
        cmd = shlex.split(self.mdrun.format(tprfile, deffnm, confout))
        self.execute_command(cmd, cwd=self.exe_dir)
        out_files = {"conf": confout, "cpt_prev": f"{deffnm}_prev.cpt"}
        for key in ("cpt", "edr", "log", "trr"):
            out_files[key] = f"{deffnm}.{key}"
        self._remove_gromacs_backup_files(self.exe_dir)
        return out_files

    def _remove_gromacs_backup_files(self, dirname: Union[str, Path]) -> None:
        """Remove GROMACS backup files (files starting with a "#").

        Args:
            dirname: The directory to remove files from.
        """
        for entry in Path(dirname).iterdir():
            if entry.name.startswith("#") and entry.is_file():
                self._removefile(entry)

    def _extract_frame(self, traj_file: str, idx: int, out_file: str) -> None:
        """Extract a frame from a .trr, .xtc or .trj file.

        Args:
            traj_file: The GROMACS file to open.
            idx: The frame number we look for.
            out_file: The file to extract to.

        Note:
            If the extension is different from .trr, .xtc or .trj,
            we will basically just copy the given input file.
        """
        trajexts = [".trr", ".xtc", ".trj"]

        logger.debug("Extracting frame, idx = %i", idx)
        logger.debug("Source file: %s, out file: %s", traj_file, out_file)
        if traj_file[-4:] in trajexts:
            # TODO: DOES THIS ACTUALLY WORK FOR XTC?
            _, data = read_trr_frame(traj_file, idx)
            if data is None:
                msg = f"Could not extract frame from {traj_file}!"
                logger.error(msg)
                raise ValueError(msg)
            xyz = data["x"]
            vel = data.get("v", None)
            box = box_matrix_to_list(data["box"], full=True)
            write_gromos96_file(out_file, self.top, xyz, vel, box)
        elif traj_file[-4:] == ".g96" and out_file[-4:] == ".g96":
            shutil.copyfile(traj_file, out_file)
        else:
            msg = f"Can't extract frame from tajectory \
                    with format: {traj_file[-4:]}"
            logger.error(msg)
            raise ValueError(msg)

    def get_energies(
        self,
        energy_file: str,
        begin: Optional[float] = None,
        end: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Return energies from a GROMACS run.

        Args:
            energy_file: The file from which to read energies.
            begin: Time of the first frame to read. If not given,
                the GROMACS defaults are used.
            end: Time of the last frame to read. If not given,
                the GROMACS defaults are used.

        Returns:
            A dictionary with energy labels as keys and the corresponding
            energies as numpy arrays.
        """
        cmd = [self.gmx, "energy", "-f", energy_file]
        if begin is not None:
            begin = max(begin, 0)
            cmd.extend(["-b", str(begin)])
        if end is not None:
            cmd.extend(["-e", str(end)])
        self.execute_command(cmd, inputs=self.energy_terms, cwd=self.exe_dir)
        xvg_file = os.path.join(self.exe_dir, "energy.xvg")
        energy = read_xvg_file(xvg_file)
        self._removefile(xvg_file)
        return energy

    def _propagate_from(
        self,
        name: str,
        path: InfPath,
        system: System,
        ens_set: Dict[str, Any],
        msg_file: FileIO,
        reverse: bool = False,
    ) -> Tuple[bool, str]:
        """Propagate with GROMACS from the given configuration.

        This method is assumed to be called after the `propagate()` has been
        invoked in the parent. The parent is responsible for reversing the
        velocities and setting the initial state of the system.

        Args:
            name: A name for the trajectory being generated.
            path: The path for storing generated phase-space points.
            ensemble: A dictionary containing the interfaces, specified in
                the 'interfaces' key.
            msg_file: An object for writing messages useful for inspecting
                the status of the propagation to a file.
            reverse: If True, the system will be propagated backward in time.

        Returns:
            A tuple containing:
                - A boolean flag for the path; True if the path can be
                    accepted generated.
                - A text description of the current status of the
                    propagation.
        """
        status = f"propagating with GROMACS (reverse = {reverse})"
        # system = ensemble['system']
        interfaces = ens_set["interfaces"]
        logger.debug(status)
        success = False
        left, _, right = interfaces
        # Dumping of the initial config were done by the parent, here
        # we will just use it:
        initial_conf = system.config[0]
        # Get the current order parameter:
        # order = self.calculate_order(ensemble)
        order = self.calculate_order(system)
        msg_file.write(
            f'# Initial order parameter: {" ".join([str(i) for i in order])}'
        )

        # So, here we will just blast off GROMACS and check the .trr
        # output when we can.
        # 1) Create mdp_file with updated number of steps:
        settings = {
            "gen_vel": "no",
            "nsteps": path.maxlen * self.subcycles,
            "continuation": "no",
        }
        mdp_file = os.path.join(self.exe_dir, f"{name}.mdp")
        self._modify_input(
            self.input_files["input"], mdp_file, settings, delim="="
        )
        # 2) Run GROMACS preprocessor:
        out_files = self._execute_grompp(mdp_file, initial_conf, name)
        # Generate some names that will be created by mdrun:
        confout = f"{name}.{self.ext}"
        out_files["conf"] = confout
        out_files["cpt_prev"] = f"{name}_prev.cpt"
        for key in ("cpt", "edr", "log", "trr"):
            out_files[key] = f"{name}.{key}"
        # Remove some of these files if present (e.g. left over from a
        # crashed simulation). This is so that GromacsRunner will not
        # start reading a .trr left from a previous simulation.

        remove = [val for key, val in out_files.items() if key != "tpr"]
        self._remove_files(self.exe_dir, remove)
        tpr_file = out_files["tpr"]
        trr_file = os.path.join(self.exe_dir, out_files["trr"])
        edr_file = os.path.join(self.exe_dir, out_files["edr"])
        cmd = shlex.split(self.mdrun.format(tpr_file, name, confout))
        # 3) Fire off GROMACS mdrun:
        logger.debug("Executing GROMACS.")
        msg_file.write(f"# Trajectory file is: {trr_file}")
        msg_file.write("# Starting GROMACS.")
        msg_file.write("# Step order parameter cv1 cv2 ...")
        with GromacsRunner(cmd, trr_file, edr_file, self.exe_dir) as gro:
            for i, data in enumerate(gro.get_gromacs_frames()):
                # Update the configuration file:
                system.set_pos((trr_file, i))
                # Also provide the loaded positions since they are
                # available:
                system.pos = data["x"]
                system.vel = data["v"]
                system.box = box_matrix_to_list(data["box"], full=True)
                if system.vel is not None and reverse:
                    system.vel *= -1
                order = self.calculate_order(
                    system, xyz=system.pos, vel=system.vel, box=system.box
                )
                msg_file.write(f'{i} {" ".join([str(j) for j in order])}')
                snapshot = {
                    "order": order,
                    "config": (trr_file, i),
                    "vel_rev": reverse,
                }
                phase_point = self.snapshot_to_system(system, snapshot)
                status, success, stop, _ = self.add_to_path(
                    path, phase_point, left, right
                )
                if stop:
                    logger.debug(
                        "Ending propagate at %i. Reason: %s", i, status
                    )
                    break
        logger.debug("GROMACS propagation done, obtaining energies!")
        msg_file.write("# Propagation done.")
        msg_file.write(f'# Reading energies from: {out_files["edr"]}')
        energy = self.get_energies(out_files["edr"])
        path.update_energies(energy["kinetic en."], energy["potential"], energy["total energy"], energy["temperature"])
        logger.debug("Removing GROMACS output after propagate.")
        remove = [
            val
            for key, val in out_files.items()
            if key not in ("trr", "gro", "g96", "edr")
        ]
        self._remove_files(self.exe_dir, remove)
        self._remove_gromacs_backup_files(self.exe_dir)
        msg_file.flush()
        return success, status

    def _prepare_shooting_point(
        self, input_file: str
    ) -> Tuple[str, Dict[str, np.ndarray]]:
        """Create the initial configuration for a shooting move.

        This creates a new initial configuration with random velocities.
        Here, the random velocities are obtained by running a zero-step
        GROMACS simulation.

        Args:
            input_file: The input configuration to generate velocities for.

        Returns:
            A tuple containing:
                - The name of the file created.
                - The energy terms read from the GROMACS .edr file.
        """
        # gen_mdp = os.path.join(self.exe_dir, 'genvel.mdp')
        gen_mdp = os.path.join(self.exe_dir, "genvel.mdp")
        if os.path.isfile(gen_mdp):
            logger.debug("%s found. Re-using it!", gen_mdp)
        else:
            # Create output file to generate velocities:
            settings: Dict[str, Union[str, int, float]] = {
                "gen_vel": "yes",
                "gen_seed": -1,
                "nsteps": 0,
                "continuation": "no",
                "gen-temp": self.temperature,
            }
            self._modify_input(
                self.input_files["input"], gen_mdp, settings, delim="="
            )
        # Run GROMACS grompp for this input file:
        out_grompp = self._execute_grompp(gen_mdp, input_file, "genvel")
        remove = [val for _, val in out_grompp.items()]
        # Run GROMACS mdrun for this tpr file:
        out_mdrun = self._execute_mdrun(out_grompp["tpr"], "genvel")
        remove += [val for key, val in out_mdrun.items() if key != "conf"]
        confout = os.path.join(self.exe_dir, out_mdrun["conf"])
        energy = self.get_energies(out_mdrun["edr"])
        # Remove run-files:
        logger.debug("Removing GROMACS output after velocity generation.")
        self._remove_files(self.exe_dir, remove)
        return confout, energy

    def set_mdrun(self, md_items: Dict[str, Any]) -> None:
        """Set the worker terminal command to execute."""
        base = md_items["wmdrun"]
        self.mdrun = base + " -s {} -deffnm {} -c {}"
        self.exe_dir = md_items["exe_dir"]

    def _read_configuration(
        self, filename: str
    ) -> Tuple[
        np.ndarray, np.ndarray, Union[np.ndarray, None], Union[List[str], None]
    ]:
        """Read output from GROMACS .g96 files.

        Args:
            filename: The file to read the configuration from.

        Returns:
            A tuple containing:
                - The positions.
                - The velocities.
                - The box dimensions.
                - Atom names, currently we do not read that for g96
                    files, and we return just a None.
        """
        if self.ext != "g96":
            msg = f"GROMACS engine does not support reading {self.ext}"
            logger.error(msg)
            raise ValueError(msg)
        _, xyz, vel, box = read_gromos96_file(filename)
        return xyz, vel, box, None

    def _reverse_velocities(self, filename: str, outfile: str) -> None:
        """Reverse velocity in a given snapshot.

        Args:
            filename: The configuration to reverse velocities in.
            outfile: The output file for storing the configuration with
                reversed velocities.
        """
        if self.ext != "g96":
            # TODO: This check for an acceptable type should come earlier.
            msg = f"GROMACS engine does not support writing {self.ext}"
            logger.error(msg)
            raise ValueError(msg)
        txt, xyz, vel, _ = read_gromos96_file(filename)
        write_gromos96_file(outfile, txt, xyz, -1 * vel)

    def modify_velocities(
        self, system: System, vel_settings: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Modify the velocities of the current state.

        Args:
            system: The system whose particle velocities are to be modified.
            vel_settings: A dict containing
                'zero_momentum': boolean, if true we reset the linear momentum
                  to zero after generating velocities internally.


        Returns:
            A tuple containing:
                - dek: The change in kinetic energy as a result of
                    the velocity modification.
                - kin_new: The new kinetic energy of the system.
        """
        pos = self.dump_frame(system)
        # generate velocities with gromacs
        if not self.infretis_genvel:
            kin_old = system.ekin
            if vel_settings.get("zero_momentum", True) is False:
                msg = (
                    "Velocitiy generation with gromacs "
                    "doesn't support zero_momentum = False!"
                )
                raise ValueError(msg)
            posvel, energy = self._prepare_shooting_point(pos)
            kin_new = energy["kinetic en."][-1]
            system.set_pos((posvel, 0))
            system.vel_rev = False
            system.ekin = kin_new
            system.vpot = energy["potential"][-1]
            system.etot = energy["total energy"][-1]
            system.temp = energy["temperature"][-1]

        # generate velocities with ASE
        else:
            txt, xyz, vel, _ = read_gromos96_file(pos)
            if not txt["VELOCITY"]:
                print(f"{pos} did not contain velocity information.")
                txt["VELOCITY"] = txt["POSITION"]
            vel, kin_old, kin_new, conf_out = self.ase_modify_system_vel(
                system, self.ase_atoms, xyz, vel, xyz2ase=10.0, vel2ase=0.01,
            )
            write_gromos96_file(conf_out, txt, xyz, vel)

        if kin_old is None or kin_new is None:
            dek = float("inf")
            logger.debug(
                "Kinetic energy not found for previous point."
                "\n(This happens when the initial configuration "
                "does not contain energies.)"
            )
        else:
            dek = kin_new - kin_old

        return dek, kin_new


class GromacsRunner:
    """A helper class for running GROMACS.

    This class handles the reading of the TRR on the fly and
    it is used to decide when to end the GROMACS execution.

    Attributes:
        cmd: The command for executing GROMACS.
        trr_file: Path to the GROMACS TRR file we are going to read.
        edr_file: Path to the GROMACS .edr file we are going to read.
        exe_dir: Path to where we are currently running GROMACS.
        fileh: The current open TRR file object.
        running: The process running GROMACS.
        bytes_read: The number of bytes read so far from the TRR file.
        ino: The current inode we are using for the file.
        stop_read: If this is set to True, we will stop the reading.
        SLEEP: How long we wait after an unsuccessful read before
            reading again.
        data_size: The size of the data (x, v, f, box, etc.) in the TRR file.
        header_size: The size of the header in the TRR file.
        stdout_name: Path to file to use for messages to standard out.
        stderr_name: Path to file to use for messages to standard error.
        stdout: File handle to write standard out to.
        stderr: File handle to write standard error to.
    """

    SLEEP: float = 0.1

    def __init__(
        self, cmd: List[str], trr_file: str, edr_file: str, exe_dir: str
    ):
        """Set the GROMACS command and the files we need.

        Args:
            cmd: The command for executing GROMACS.
            trr_file: The GROMACS TRR file we are going to read.
            edr_file: A .edr file we are going to read.
            exe_dir: Path to where we are currently running GROMACS.
        """
        self.cmd: List[str] = cmd
        self.trr_file: str = trr_file
        self.edr_file: str = edr_file
        self.exe_dir: str = exe_dir
        self.fileh: BufferedReader
        self.running: Optional[subprocess.Popen[bytes]] = None
        self.bytes_read: int = 0
        self.ino: int = 0
        self.stop_read: bool = True
        self.data_size: int = 0
        self.header_size: int = 0
        self.stdout_name: Optional[str] = None
        self.stderr_name: Optional[str] = None
        self.stdout: Optional[BufferedWriter] = None
        self.stderr: Optional[BufferedWriter] = None

    def start(self) -> None:
        """Start execution of GROMACS and wait for output file creation."""
        logger.debug("Starting GROMACS execution in %s", self.exe_dir)

        self.stdout_name = os.path.join(self.exe_dir, "stdout.txt")
        self.stderr_name = os.path.join(self.exe_dir, "stderr.txt")
        self.stdout = open(self.stdout_name, "wb")
        self.stderr = open(self.stderr_name, "wb")

        self.running = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=self.stdout,
            stderr=self.stderr,
            shell=False,
            cwd=self.exe_dir,
            preexec_fn=os.setsid,
        )
        present = []
        # Wait for the TRR/EDR files to appear:
        for fname in (self.trr_file, self.edr_file):
            while not os.path.isfile(fname):
                logger.debug('Waiting for GROMACS file "%s"', fname)
                sleep(self.SLEEP)
                poll = self.check_poll()
                if poll is not None:
                    logger.debug("GROMACS execution stopped")
                    break
            if os.path.isfile(fname):
                present.append(fname)
        # Prepare and open the TRR file:
        self.bytes_read = 0
        # Ok, so GROMACS might have crashed in between writing the
        # files. Check that both files are indeed here:
        if self.trr_file in present and self.edr_file in present:
            self.fileh = open(self.trr_file, "rb")
            self.ino = os.fstat(self.fileh.fileno()).st_ino
            self.stop_read = False
        else:
            self.stop_read = True

    def __enter__(self):
        """Context manager to start running GROMACS."""
        self.start()
        return self

    def get_gromacs_frames(self) -> Iterator[Dict[str, np.ndarray]]:
        """Read the GROMACS TRR file on-the-fly."""
        first_header = True
        header = None
        new_bytes = 0
        while not self.stop_read:
            poll = self.check_poll()
            if poll is not None:
                # GROMACS is done, read remaining data.
                self.stop_read = True
                if os.path.getsize(self.trr_file) - self.bytes_read > 0:
                    for _, data, _ in read_remaining_trr(
                        self.trr_file, self.fileh, self.bytes_read
                    ):
                        yield data

            else:
                # First we try to get the header from the file:
                size = os.path.getsize(self.trr_file)
                if self.header_size == 0:
                    header_size = TRR_HEAD_SIZE
                else:
                    header_size = self.header_size
                if size >= self.bytes_read + header_size:
                    # Try to read next frame:
                    try:
                        header, new_bytes = read_trr_header(self.fileh)
                    except EOFError:
                        new_fileh, new_ino = reopen_file(
                            self.trr_file,
                            self.fileh,
                            self.ino,
                            self.bytes_read,
                        )
                        if new_fileh is not None and new_ino is not None:
                            self.fileh = new_fileh
                            self.ino = new_ino
                    if header is not None:
                        self.bytes_read += new_bytes
                        self.header_size = new_bytes
                        if first_header:
                            logger.debug("TRR header was: %i", new_bytes)
                            first_header = False
                        # Calculate the size of the data:
                        self.data_size = sum(
                            header[key] for key in TRR_DATA_ITEMS
                        )
                        data = None
                        while data is None:
                            size = os.path.getsize(self.trr_file)
                            if size >= self.bytes_read + self.data_size:
                                try:
                                    data, new_bytes = get_data(
                                        self.fileh, header
                                    )
                                except EOFError:
                                    new_fileh, new_ino = reopen_file(
                                        self.trr_file,
                                        self.fileh,
                                        self.ino,
                                        self.bytes_read,
                                    )
                                    if (
                                        new_fileh is not None
                                        and new_ino is not None
                                    ):
                                        self.fileh = new_fileh
                                        self.ino = new_ino
                                if data is None:
                                    # Data is not ready, just wait:
                                    sleep(self.SLEEP)
                                else:
                                    self.bytes_read += new_bytes
                                    yield data
                            else:
                                # Data is not ready, just wait:
                                sleep(self.SLEEP)
                else:
                    # Header was not ready, just wait before trying again.
                    sleep(self.SLEEP)

    def close(self) -> None:
        """Close the file, in case that is explicitly needed."""
        if self.fileh is not None and not self.fileh.closed:
            logger.debug('Closing GROMACS file: "%s"', self.trr_file)
            self.fileh.close()
        for handle in (self.stdout, self.stderr):
            if handle is not None and not handle.closed:
                handle.close()

    def stop(self) -> None:
        """Stop the current GROMACS execution."""
        if self.running:
            for handle in (
                self.running.stdin,
                self.running.stdout,
                self.running.stderr,
            ):
                if handle:
                    try:
                        handle.close()
                    except AttributeError:
                        pass
            if self.running.returncode is None:
                logger.debug("Terminating GROMACS execution")
                os.killpg(os.getpgid(self.running.pid), signal.SIGTERM)

            self.running.wait(timeout=360)
        self.stop_read = True
        self.close()  # Close the TRR file.

    def __exit__(self, *args) -> None:
        """Stop execution and close file for a context manager."""
        self.stop()

    def __del__(self):
        """Stop execution and close file."""
        self.stop()

    def check_poll(self) -> Optional[int]:
        """Check the current status of the running subprocess."""
        if self.running:
            poll = self.running.poll()
            if poll is not None:
                logger.debug("Execution of GROMACS stopped")
                logger.debug("Return code was: %i", poll)
                if poll != 0:
                    logger.error("STDOUT, see file: %s", self.stdout_name)
                    logger.error("STDERR, see file: %s", self.stderr_name)
                    raise RuntimeError("Error in GROMACS execution.")
            return poll
        raise RuntimeError("GROMACS is not running.")


def read_trr_frame(
    filename: str, index: int
) -> Tuple[Union[Dict[str, Any], None], Union[Dict[str, np.ndarray], None]]:
    """Return a given frame from a TRR file.

    Args:
        filename: The path to the file to read.
        index: The frame number to read.

    Returns:
        A tuple containing:
            - The header for the frame, if the frame exists.
            - The data in the frame, if the frame exists.
    """
    idx = 0
    with open(filename, "rb") as infile:
        while True:
            try:
                header, _ = read_trr_header(infile)
                if idx == index:
                    data = read_trr_data(infile, header)
                    return header, data
                skip_trr_data(infile, header)
                idx += 1
                if idx > index:
                    logger.error("Frame %i not found in %s", index, filename)
                    return None, None
            except EOFError:
                return None, None


def read_trr_header(fileh: BufferedReader) -> Tuple[Dict[str, Any], int]:
    """Read a header from a TRR file.

    Args:
        fileh: The file handle for the file we are reading.

    Returns:
        A tuple containing:
            - The header read from the file.
            - Number of bytes read.
    """
    start = fileh.tell()
    endian = ">"
    magic = read_struct_buff(fileh, f"{endian}1i")[0]
    if magic == _GROMACS_MAGIC:
        pass
    else:
        magic = swap_integer(magic)
        if not magic == _GROMACS_MAGIC:
            logger.critical(
                "TRR file might be inconsistent! Could find _GROMACS_MAGIC"
            )
        endian = swap_endian(endian)
    slen = read_struct_buff(fileh, f"{endian}2i")
    raw = read_struct_buff(fileh, f"{endian}{slen[0] - 1}s")
    version = raw[0].split(b"\0", 1)[0].decode("utf-8")
    if not version == _TRR_VERSION:
        raise ValueError("Unknown format")

    head_fmt = _HEAD_FMT.format(endian)
    head_s = read_struct_buff(fileh, head_fmt)
    header = {}
    for i, val in enumerate(head_s):
        key = _HEAD_ITEMS[i]
        header[key] = val
    # The next are either floats or double
    double = is_double(header)
    if double:
        fmt = f"{endian}2d"
    else:
        fmt = f"{endian}2f"
    header_r = read_struct_buff(fileh, fmt)
    header["time"] = header_r[0]
    header["lambda"] = header_r[1]
    header["endian"] = endian
    header["double"] = double
    return header, fileh.tell() - start


def gromacs_settings(settings: Dict[str, Any], input_path: str) -> None:
    """Read and processes GROMACS settings.

    Args:
        settings: The current input settings.
        input_path: The GROMACS input path.
    """
    ext = settings["engine"].get("gmx_format", "g96")
    default_files = {
        "conf": f"conf.{ext}",
        "input_o": "grompp.mdp",
        "topology": "topol.top",
        "index": "index.ndx",
    }
    settings["engine"]["input_files"] = {}
    for key in ("conf", "input_o", "topology", "index"):
        # Add input path and the input files if input is no given:
        settings["engine"]["input_files"][key] = settings["engine"].get(
            key, os.path.join(input_path, default_files[key])
        )


def read_gromos96_file(
    filename: Union[str, Path],
) -> Tuple[Dict[str, List[str]], np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Read a single configuration GROMACS .g96 file.

    Args:
        filename: The file to read.

    Returns:
        A tuple containing:
            - The raw data read from the file, grouped into sections.
                Note that this does not include the actual positions and
                velocities as these are returned separately.
            - The positions.
            - The velocities.
            - The simulation box.
    """
    _len = 15
    _pos = 24
    rawdata: Dict[str, List[str]] = {
        "TITLE": [],
        "POSITION": [],
        "VELOCITY": [],
        "BOX": [],
        "POSITIONRED": [],
        "VELOCITYRED": [],
    }
    section = ""
    with open(filename, encoding="utf-8", errors="replace") as gromosfile:
        for lines in gromosfile:
            new_section = False
            stripline = lines.strip()
            if stripline == "END":
                continue
            for key in rawdata:
                if stripline == key:
                    new_section = True
                    section = key
                    break
            if new_section:
                continue
            rawdata[section].append(lines.rstrip())
    txtdata: Dict[str, List[str]] = {}
    xyzdata: Dict[str, List[List[float]]] = {}
    for key in ("POSITION", "VELOCITY"):
        txtdata[key] = []
        xyzdata[key] = []
        for line in rawdata[key]:
            txt = line[:_pos]
            txtdata[key].append(txt)
            pos = [
                float(line[i : i + _len]) for i in range(_pos, 4 * _len, _len)
            ]
            xyzdata[key].append(pos)
        for line in rawdata[key + "RED"]:
            txt = line[:_pos]
            txtdata[key].append(txt)
            pos = [float(line[i : i + _len]) for i in range(0, 3 * _len, _len)]
            xyzdata[key].append(pos)
        # xyzdata[key] = np.array(xyzdata[key])
    rawdata["POSITION"] = txtdata["POSITION"]
    rawdata["VELOCITY"] = txtdata["VELOCITY"]
    xyz = np.array(xyzdata["POSITION"])
    if not rawdata["VELOCITY"]:
        # No velocities were found in the input file.
        vel = np.zeros_like(xyz)
        logger.info("Input g96 did not contain velocities")
    else:
        vel = np.array(xyzdata["VELOCITY"])
    if rawdata["BOX"]:
        # TODO: SHOULD ALL BOXES BE CONVERTED INTO THE SAME FORM ALREADY HERE?
        box = np.array([float(i) for i in rawdata["BOX"][0].split()])
    else:
        # TODO: IS IT BETTER TO JUST FAIL IF THE BOX IS NOT THERE?
        box = None
        logger.warning("Input g96 did not contain box vectors.")
    return rawdata, xyz, vel, box


def write_gromos96_file(
    filename: Union[str, Path],
    raw: Dict[str, List[str]],
    xyz: np.ndarray,
    vel: Union[np.ndarray, None],
    box: Union[np.ndarray, List[float], None] = None,
) -> None:
    """Write configuration in GROMACS .g96 format.

    Args:
        filename: The name of the file to create.
        raw: This contains the raw data read from a .g96 file.
        xyz: The positions to write.
        vel: The velocities to write.
        box: The box matrix.
    """
    _keys = ("TITLE", "POSITION", "VELOCITY", "BOX")
    with open(filename, "w", encoding="utf-8") as outfile:
        for key in _keys:
            if key not in raw:
                continue
            outfile.write(f"{key}\n")
            for i, line in enumerate(raw[key]):
                if key == "POSITION":
                    outfile.write(_G96_FMT.format(line, *xyz[i]))
                elif key == "VELOCITY":
                    if vel is not None:
                        outfile.write(_G96_FMT.format(line, *vel[i]))
                elif box is not None and key == "BOX":
                    if len(box) == 3:
                        outfile.write(_G96_BOX_FMT_3.format(*box))
                    else:
                        outfile.write(_G96_BOX_FMT.format(*box))
                else:
                    outfile.write(f"{line}\n")
            outfile.write("END\n")


def read_struct_buff(fileh: BufferedReader, fmt: str) -> Tuple[Any, ...]:
    """Unpack from a file handle with a given format.

    Args:
        fileh: The file handle to unpack from.
        fmt: The format to use for unpacking.

    Returns:
        The unpacked elements according to the given format.

    Raises:
        EOFError: An EOFError is raised if `fileh.read()` attempts to read
            past the end of the file.
    """
    buff = fileh.read(struct.calcsize(fmt))
    if not buff:
        raise EOFError
    return struct.unpack(fmt, buff)


def is_double(header: Dict[str, Any]) -> bool:
    """Determine if we should use double precision.

    This method determines the precision to use when reading
    a TRR file. This is based on the header read for a given
    frame which defines the sizes of certain "fields" like the box
    or the positions. From this size, the precision can be obtained.

    Args:
        header: The header read from the TRR file.

    Returns:
        True if we should use double precision, False otherwise.
    """
    key_order = ("box_size", "x_size", "v_size", "f_size")
    size = 0
    for key in key_order:
        if header[key] != 0:
            if key == "box_size":
                size = int(header[key] / _DIM**2)
                break
            size = int(header[key] / (header["natoms"] * _DIM))
            break
    if size not in (_SIZE_FLOAT, _SIZE_DOUBLE):
        raise ValueError("Could not determine size!")
    return size == _SIZE_DOUBLE


def skip_trr_data(fileh: BufferedReader, header: Dict[str, Any]) -> None:
    """Skip coordinates/box data in a frame.

    This method is used when we want to skip a data section in
    the TRR file. Rather than reading the data, it will use the
    size read in the header to skip ahead to the next frame.

    Args:
        fileh: The file handle for the file we are reading.
        header: The header read from the TRR file.
    """
    offset = sum(header[key] for key in TRR_DATA_ITEMS)
    fileh.seek(offset, 1)


def read_trr_data(
    fileh: BufferedReader, header: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """Read box, coordinates etc. from a TRR file.

    Args:
        fileh: The file handle for the file we are reading.
        header: The header read from the file.

    Returns:
        data: The data we read from the file. It may contain the following
            keys if the data was found in the frame:

            - ``box`` : the box matrix,
            - ``vir`` : the virial matrix,
            - ``pres`` : the pressure matrix,
            - ``x`` : the coordinates,
            - ``v`` : the velocities, and
            - ``f`` : the forces
    """
    data = {}
    endian = header["endian"]
    double = header["double"]
    for key in ("box", "vir", "pres"):
        header_key = f"{key}_size"
        if header[header_key] != 0:
            data[key] = read_matrix(fileh, endian, double)
    for key in ("x", "v", "f"):
        header_key = f"{key}_size"
        if header[header_key] != 0:
            data[key] = read_coord(fileh, endian, double, header["natoms"])
    return data


def read_matrix(
    fileh: BufferedReader, endian: str, double: bool
) -> np.ndarray:
    """Read a matrix from the TRR file.

    Here, we assume that the matrix will be of
    dimensions (_DIM, _DIM).

    Args:
        fileh: The file handle to read from.
        endian: Determines the byte order.
        double: If true, we will assume that the numbers
            were stored in double precision.

    Return:
        The matrix as an array.
    """
    if double:
        fmt = f"{endian}{_DIM**2}d"
    else:
        fmt = f"{endian}{_DIM**2}f"
    read = read_struct_buff(fileh, fmt)
    mat = np.zeros((_DIM, _DIM))
    for i in range(_DIM):
        for j in range(_DIM):
            mat[i, j] = read[i * _DIM + j]
    return mat


def read_coord(
    fileh: BufferedReader, endian: str, double: bool, natoms: int
) -> np.ndarray:
    """Read a coordinate section from the TRR file.

    This method will read the full coordinate section from a TRR
    file. The coordinate section may be positions, velocities or
    forces.

    Args:
        fileh: The file handle to read from.
        endian: Determines the byte order.
        double: If true, we will assume that the numbers
            were stored in double precision.
        natoms: The number of atoms we have stored coordinates for.

    Returns:
        The coordinates as a numpy array. It will have
            ``natoms`` rows and ``_DIM`` columns.
    """
    if double:
        fmt = f"{endian}{natoms * _DIM}d"
    else:
        fmt = f"{endian}{natoms * _DIM}f"
    read = read_struct_buff(fileh, fmt)
    mat = np.array(read)
    mat.shape = (natoms, _DIM)
    return mat


def read_xvg_file(filename: str) -> Dict[str, np.ndarray]:
    """Return data from a .xvg file as numpy arrays.

    Args:
        filename: Path to the .xvg file to read.

    Returns:
        A dict containing where the keys are the fields read
            in the .xvg file and the values are the corresponding
            data.
    """
    raw_data = []
    legends = []
    with open(filename, encoding="utf-8") as fileh:
        for lines in fileh:
            if lines.startswith("@ s") and lines.find("legend") != -1:
                legend = lines.split("legend")[-1].strip()
                legend = legend.replace('"', "")
                legends.append(legend.lower())
            else:
                if lines.startswith("#") or lines.startswith("@"):
                    pass
                else:
                    raw_data.append([float(i) for i in lines.split()])
    data = np.array(raw_data)
    data_dict = {"step": np.arange(tuple(data.shape)[0])}
    for i, key in enumerate(legends):
        data_dict[key] = data[:, i + 1]
    return data_dict


def get_data(
    fileh: BufferedReader, header: Dict[str, Any]
) -> Tuple[Dict[str, np.ndarray], int]:
    """Read data from the TRR file.

    Args:
        fileh: The file we are reading.
        header: The previously read header. Contains sizes and what to read.

    Returns:
        A tuple containing:
            - The data read from the file.
            - The size of the data read.
    """
    data_size = sum(header[key] for key in TRR_DATA_ITEMS)
    data = read_trr_data(fileh, header)
    return data, data_size


def read_remaining_trr(
    filename: str, fileh: BufferedReader, start: int
) -> Iterator[Tuple[Dict[str, Any], Dict[str, np.ndarray], int]]:
    """Read remaining frames from the TRR file.

    Args:
        filename: The file we are reading from.
        fileh: The file object we are reading from.
        start: The current position we are at.

    Yields:
        A tuple containing:
            - The header read from the file
            - The data read from the file.
            - The size of the data read.
    """
    stop = False
    bytes_read = start
    bytes_total = os.path.getsize(filename)
    logger.debug("Reading remaining data from: %s", filename)
    while not stop:
        if bytes_read >= bytes_total:
            stop = True
            continue
        header = None
        new_bytes = bytes_read
        try:
            header, new_bytes = read_trr_header(fileh)
        except EOFError:  # pragma: no cover
            # Just assume that we have reached the end of the
            # file and we just stop here. It should not be reached,
            # kept for safety
            stop = True
            continue
        if header is not None:
            bytes_read += new_bytes
            try:
                data, new_bytes = get_data(fileh, header)
                if data is not None:
                    bytes_read += new_bytes
                    yield header, data, bytes_read
            except EOFError:  # pragma: no cover
                # Hopefully, this code should not be reached.
                # kept for safety
                stop = True
                continue


def reopen_file(
    filename: str, fileh: BufferedReader, inode: int, bytes_read: int
) -> Tuple[Optional[BufferedReader], Optional[int]]:
    """Reopen a file if the inode has changed.

    Args:
        filename: The name of the file we are working with.
        fileh: The current open file object.
        inode: The current inode we are using.
        bytes_read: The position we should start reading at.

    Returns:
        A tuple containing:
            - The new file object.
            - The new inode.
    """
    if os.stat(filename).st_ino != inode:
        new_fileh = open(filename, "rb")
        fileh.close()
        new_inode = os.fstat(new_fileh.fileno()).st_ino
        new_fileh.seek(bytes_read)
        return new_fileh, new_inode
    return None, None


def swap_integer(integer: int) -> int:
    """Convert little/big endian."""
    return (
        ((integer << 24) & 0xFF000000)
        | ((integer << 8) & 0x00FF0000)
        | ((integer >> 8) & 0x0000FF00)
        | ((integer >> 24) & 0x000000FF)
    )


def swap_endian(endian: str) -> str:
    """Just swap the string for selecting big/little endian."""
    if endian == ">":
        return "<"
    if endian == "<":
        return ">"
    raise ValueError("Undefined swap!")
