"""Base engine class."""
from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from infretis.classes.formatter import FileIO, OutputFormatter

if TYPE_CHECKING:  # pragma: no cover
    from infretis.classes.orderparameter import OrderParameter
    from infretis.classes.path import Path as InfPath
    from infretis.classes.system import System


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class EngineBase(metaclass=ABCMeta):
    """Abstract base class for engines.

    The engines perform molecular dynamics and they are assumed to act
    on a system.

    Attributes:
        description: Short string description of the engine.
            Used for printing information about the integrator.
        exe_dir: A directory where the engine is going to be executed.
    """

    def __init__(self, description: str, timestep: float, subcycles: int):
        """Initialize the engine."""
        self.description: str = description
        self._exe_dir: str = "."
        self.timestep: float = timestep
        self.subcycles: int = subcycles
        self.ext: str = "xyz"
        self.input_files: dict[str, str | Path] = {}
        self.order_function: OrderParameter | None = None

    @property
    def exe_dir(self) -> str:
        """Return the directory we are currently using."""
        return self._exe_dir

    @exe_dir.setter
    def exe_dir(self, exe_dir: str) -> None:
        """Set the directory for executing."""
        self._exe_dir = exe_dir
        if exe_dir is not None:
            logger.debug('Setting exe_dir to "%s"', exe_dir)
            if not os.path.isdir(exe_dir):
                logger.warning(
                    (
                        '"Exe dir" for "%s" is set to "%s" which does'
                        " not exist!"
                    ),
                    self.description,
                    exe_dir,
                )

    @staticmethod
    def add_to_path(
        path: InfPath, phase_point: System, left: float, right: float
    ) -> tuple[str, bool, bool, bool]:
        """Add a phase point and perform some checks.

        This method is intended to be used by the propagate methods
        to determine if the propagation should end or not.

        Args:
            path: The path to add the phase point to.
            phase_point: The phase point to add to the path.
            left: The left interface.
            right: The right interface.
        """
        status = "Running propagate..."
        success = False
        stop = False
        add = path.append(phase_point)
        if not add:
            if path.length >= path.maxlen:
                status = "Max. path length exceeded"
            else:  # pragma: no cover
                status = "Could not add for unknown reason"
            success = False
            stop = True
        if path.phasepoints[-1].order[0] < left:
            status = "Crossed left interface!"
            success = True
            stop = True
        elif path.phasepoints[-1].order[0] > right:
            status = "Crossed right interface!"
            success = True
            stop = True
        if path.length == path.maxlen:
            status = "Max. path length exceeded!"
            success = False
            stop = True
        return status, success, stop, add

    @abstractmethod
    def modify_velocities(
        self, ensemble: System, vel_settings: dict[str, Any]
    ) -> tuple[float, float]:
        """Modify the velocities of the current state.

        Args:
            ensemble: The system to modify the velocity of.
            vel_settings: A dictionary containing the parameters
                for setting the velocity. How the settings are
                used will be defined in the specific engies.

        Returns:
            A tuple containing:
                - The change in the kinetic energy.
                - The new kinetic energy.
        """

    @abstractmethod
    def set_mdrun(self, md_items: dict[str, Any]) -> None:
        """Set exe_dir and worker terminal command to be run."""

    def calculate_order(
        self,
        system: System,
        xyz: np.ndarray | None = None,
        vel: np.ndarray | None = None,
        box: np.ndarray | None = None,
    ) -> list[float]:
        """Calculate the order parameter of the current system.

        Note, if ``xyz``, ``vel`` or ``box`` are given, we will
        **NOT** read positions, velocity and box information from the
        current configuration file. This can be used in case we
        have already read these elsewhere.

        Args:
            system: The current state of the system we are investigating.
            xyz: The positions to use (to override the system's positions).
            vel: The velocities to use (to override the system's velocities).
            box: The current box vectors (to override the system's box).

        Returns:
            The calculated order parameter(s).
        """
        # Convert system into an internal representation:
        if any((xyz is None, vel is None, box is None)):
            out = self._read_configuration(system.config[0])
            xyz = out[0]
            vel = out[1]
            box = out[2]
        if xyz is not None:
            system.pos = xyz
        if vel is not None:
            system.vel = vel * -1.0 if system.vel_rev else vel

        # system.update_box(box)
        if box is not None:
            system.box = box
        if self.order_function is None:
            raise ValueError("Order parameter is not defined!")
        return self.order_function.calculate(system)

    def dump_phasepoint(
        self, phasepoint: System, deffnm: str = "conf"
    ) -> None:
        """Dump the current configuration from a system object to a file."""
        pos_file = self.dump_config(phasepoint.config, deffnm=deffnm)
        phasepoint.set_pos((pos_file, 0))

    def _name_output(self, basename: str) -> str:
        """Create a file name for the output file.

        This method is used when we dump a configuration to add
        the correct extension.

        Args:
            basename: The base name to give to the file.

        Returns:
            A file name with the correct extension.
        """
        out_file = f"{basename}.{self.ext}"
        return os.path.join(self.exe_dir, out_file)

    def dump_config(
        self, config: tuple[str, int], deffnm: str = "conf"
    ) -> str:
        """Extract configuration frame from a system if needed.

        Args:
            config: The configuration given as (filename, index).
            deffnm: The base name for the file we dump to.

        Returns:
            The file name we dumped to. If we did not in fact dump, this is
            because the system contains a single frame and we can use it
            directly. Then we return simply this file name.

        Note:
            If the velocities should be reversed, this is handled elsewhere.

        """
        out_file = os.path.join(self.exe_dir, self._name_output(deffnm))
        pos_file, idx = config
        if idx is None:
            if pos_file != out_file:
                self._copyfile(pos_file, out_file)
        else:
            logger.debug("Config: %s", (config,))
            self._extract_frame(pos_file, idx, out_file)
        return out_file

    def dump_frame(self, system: System, deffnm: str = "conf") -> str:
        """Dump the frame from a system object."""
        return self.dump_config(system.config, deffnm=deffnm)

    @abstractmethod
    def _extract_frame(self, traj_file: str, idx: int, out_file: str) -> None:
        """Extract a frame from a trajectory file.

        Args:
            traj_file: Path to the trajectory file to open.
            idx: The frame number to dump.
            out_file: Path to the file to dump to.
        """

    def clean_up(self) -> None:
        """Remove all files from the current directory."""
        dirname = self.exe_dir
        logger.debug('Running engine clean-up in "%s"', dirname)
        files = [item.name for item in os.scandir(dirname) if item.is_file()]
        if dirname is not None:
            self._remove_files(dirname, files)

    def propagate(
        self,
        path: InfPath,
        ens_set: dict[str, Any],
        system: System,
        reverse: bool = False,
    ) -> tuple[bool, str]:
        """Propagate the equations of motion with the external code.

        This method will explicitly do the common set-up, before
        calling more specialised code for doing the actual propagation.

        Args:
            path: This is the path we use to fill in phase-space points.
                We are here not returning a new path - this since we want
                to delegate the creation of the path to the method
                that is running `propagate`.
            ens_set: The set of dictionaries containing the information
                about the simulation.
            reverse: If True, the system will be propagated backward in time.
                Otherwise, it will be propagated forward in time.

        Returns:
            A tuple containing:
                - True if the enerated path can be accepted. False otherwise.
                - A text description of the current status of the propagation.
                  This can be used to interpret the cases where the generated
                  path is not acceptable.
        """
        logger.debug('Running propagate with: "%s"', self.description)

        prefix = ens_set["ens_name"] + "_" + str(counter())
        if reverse:
            logger.debug("Running backward in time.")
            name = prefix + "_trajB"
        else:
            logger.debug("Running forward in time.")
            name = prefix + "_trajF"
        logger.debug('Trajectory name: "%s"', name)
        # Also create a message file for inspecting progress:
        msg_file_name = os.path.join(self.exe_dir, f"msg-{name}.txt")
        logger.debug("Writing propagation progress to: %s", msg_file_name)
        msg_file = FileIO(
            msg_file_name, "w", OutputFormatter("MSG_File"), backup=False
        )
        msg_file.open()
        msg_file.write(f"# Preparing propagation with {self.description}")
        msg_file.write(f"# Trajectory label: {name}")

        # initial_state = ensemble['system'].copy()
        # system = ensemble['system']

        initial_file = self.dump_frame(system, deffnm=prefix + "_conf")
        msg_file.write(f"# Initial file: {initial_file}")
        logger.debug("Initial state: %s", system)

        if reverse != system.vel_rev:
            logger.debug("Reversing velocities in initial config.")
            msg_file.write("# Reversing velocities")
            basepath = os.path.dirname(initial_file)
            localfile = os.path.basename(initial_file)
            initial_conf = os.path.join(basepath, f"r_{localfile}")
            self._reverse_velocities(initial_file, initial_conf)
        else:
            initial_conf = initial_file
        msg_file.write(f"# Initial config: {initial_conf}")

        # Update system to point to the configuration file:
        system.set_pos((initial_conf, 0))
        system.vel_rev = reverse
        # Propagate from this point:
        # msg_file.write(f'# Interfaces: {ensemble["interfaces"]}')
        success, status = self._propagate_from(
            name, path, system, ens_set, msg_file, reverse=reverse
        )
        # Reset to initial state:
        # ensemble['system'] = initial_state
        msg_file.close()
        return success, status

    @abstractmethod
    def _propagate_from(
        self,
        name: str,
        path: InfPath,
        system: System,
        ensemble: dict[str, Any],
        msg_file: FileIO,
        reverse: bool = False,
    ) -> tuple[bool, str]:
        """Execute the actual propagation with the MD engine.

        This method is called from :py:meth:`.propagate` which
        handles the preparations for the propagation.

        Args:
            name: A name to use for the trajectory we are generating.
            path: This is the path we use to fill in phase-space points.
            system: The initial state of the propagation.
            ensemble: Dictionary with simulation settings.
            msg_file: An object we use for writing out messages that are
                useful for inspecting the status of MD execution.
            reverse: If True, the system will be propagated backward in time.
                Otherwise, it will be propagated forward in time.

        Returns:
            A tuple containing:
                - True if the enerated path can be accepted. False otherwise.
                - A text description of the current status of the propagation.
                  This can be used to interpret the cases where the generated
                  path is not acceptable.
        """

    @staticmethod
    def snapshot_to_system(system: System, snapshot: dict[str, Any]) -> System:
        """Convert a snapshot to a system object."""
        system_copy = system.copy()
        system_copy.order = snapshot.get("order", None)
        # # particles = system_copy.particles
        system_copy.pos = snapshot.get("pos", None)
        system_copy.vel = snapshot.get("vel", None)
        system_copy.vpot = snapshot.get("vpot", None)
        system_copy.ekin = snapshot.get("ekin", None)
        for external in ("config", "vel_rev"):
            if hasattr(system_copy, external) and external in snapshot:
                setattr(system_copy, external, snapshot[external])
        return system_copy

    @abstractmethod
    def _read_configuration(
        self, filename: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str] | None]:
        """Read output configuration from external software.

        Args:
            filename: Path to the file read a configuration from.

        Returns:
            A tuple containing:
                - The positions read.
                - The velocities read.
                - The dimensions of the simulation box.
                - Atom labels (if found).

        """

    @abstractmethod
    def _reverse_velocities(self, filename: str, outfile: str) -> None:
        """Reverse velocities in a given snapshot.

        Args:
            filename: Path to the file to reverse velocities in.
            outfile: Path to file to write with reversed velocities.
        """

    @staticmethod
    def _modify_input(
        sourcefile: str | Path,
        outputfile: str | Path,
        settings: dict[str, Any],
        delim: str = "=",
    ) -> None:
        """Modify input file for external software.

        Here we assume that the input file has a syntax consisting of
        ``keyword = setting``. We will only replace settings for
        the keywords we find in the file that is also inside the
        ``settings`` dictionary.

        Args:
            sourcefile: Path to file to use as a template for creating
                the modified input file.
            outputfile: Path to the file to write input settings to.
            settings: A dictionary with settings to write.
            delim: The delimiter used for separation keywords from settings.
        """
        reg = re.compile(rf"(.*?){delim}")
        written = set()
        with open(sourcefile, encoding="utf-8") as infile, open(
            outputfile, mode="w", encoding="utf-8"
        ) as outfile:
            for line in infile:
                to_write = line
                match = reg.match(line)
                if match:
                    keyword = "".join([match.group(1), delim])
                    keyword_strip = match.group(1).strip()
                    if keyword_strip in settings:
                        to_write = f"{keyword} {settings[keyword_strip]}\n"
                    written.add(keyword_strip)
                outfile.write(to_write)
            # Add settings not yet written:
            for key, value in settings.items():
                if key not in written:
                    outfile.write(f"{key} {delim} {value}\n")

    @staticmethod
    def _read_input_settings(
        sourcefile: str | Path, delim: str = "="
    ) -> dict[str, Any]:
        """Read input settings for simulation input files.

        Here we assume that the input file has a syntax consisting of
        `keyword = setting`, where `=` can be any string given
        in the input parameter `delim`.

        Args:
            sourcefile: The path of the file read settings from.
            delim: The delimiter used for separation keywords from settings.

        Returns:
            settings: The settings found in the file.

        Note:
            This method assumes **only one keyword per line**.
        """
        reg = re.compile(rf"(.*?){delim}")
        settings = {}
        with open(sourcefile, encoding="utf-8") as infile:
            for line in infile:
                key = reg.match(line)
                if key:
                    keyword_strip = key.group(1).strip()
                    settings[keyword_strip] = line.split(delim)[1].strip()
        return settings

    def execute_command(
        self,
        cmd: list[str],
        cwd: str | None = None,
        inputs: bytes | None = None,
    ) -> int:
        """Execute an external command for the engine.

        We are here executing a command and then waiting until it
        finishes. The standard out and standard error are piped to
        files during the execution and can be inspected if the
        command fails. This method returns the return code of the
        issued command.

        Args:
            cmd: The command to execute.
            cwd: The current working directory to set for the command.
            inputs: Additional inputs to give to the command. These are not
                arguments but more akin to keystrokes etc. that the external
                command may take.

        Returns:
            The return code of the issued command.
        """
        cmd2 = " ".join(cmd)
        logger.debug("Executing: %s", cmd2)
        if inputs is not None:
            logger.debug("With input: %s", inputs)

        out_name = "stdout.txt"
        err_name = "stderr.txt"

        if cwd:
            out_name = os.path.join(cwd, out_name)
            err_name = os.path.join(cwd, err_name)

        return_code = None

        with open(out_name, "wb") as fout, open(err_name, "wb") as ferr:
            cmd = shlex.split(" ".join(cmd))
            exe = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=fout,
                stderr=ferr,
                shell=False,
                cwd=cwd,
            )
            exe.communicate(input=inputs)
            # Note: communicate will wait until process terminates.
            return_code = exe.returncode
            if return_code != 0:
                logger.error(
                    "Execution of external program (%s) failed!",
                    self.description,
                )
                logger.error("Attempted command: %s", cmd2)
                logger.error("Execution directory: %s", cwd)
                if inputs is not None:
                    logger.error("Input to external program was: %s", inputs)
                logger.error(
                    "Return code from external program: %i", return_code
                )
                logger.error("STDOUT, see file: %s", out_name)
                logger.error("STDERR, see file: %s", err_name)
                msg = (
                    f"Execution of external program ({self.description}) "
                    f"failed with command:\n {cmd2}.\n"
                    f"Return code: {return_code}"
                )
                raise RuntimeError(msg)
        if return_code is not None and return_code == 0:
            self._removefile(out_name)
            self._removefile(err_name)
        return return_code

    @staticmethod
    def _copyfile(source: str, dest: str) -> None:
        """Copy file from source to destination."""
        logger.debug("Copy: %s -> %s", source, dest)
        shutil.copyfile(source, dest)

    @staticmethod
    def _removefile(filename: str | Path) -> None:
        """Remove a given file if it exist."""
        try:
            Path(filename).unlink(missing_ok=True)
            logger.debug("Removing: %s", filename)
        except OSError:
            logger.debug("Could not remove: %s", filename)

    def _remove_files(self, dirname: str, files: list[str]) -> None:
        """Remove files from a directory.

        Args:
            dirname: Path to the directory to remove files from.
            files: A list with files to remove.
        """
        for i in files:
            self._removefile(os.path.join(dirname, i))

    def __eq__(self, other) -> bool:
        """Check if two engines are equal."""
        if self.__class__ != other.__class__:
            logger.debug("%s and %s.__class__ differ", self, other)
            return False

        if set(self.__dict__) != set(other.__dict__):
            logger.debug("%s and %s.__dict__ differ", self, other)
            return False

        for i in ["description", "_exe_dir", "timestep"]:
            if hasattr(self, i):
                if getattr(self, i) != getattr(other, i):
                    logger.debug(
                        "%s for %s and %s, attributes are %s and %s",
                        i,
                        self,
                        other,
                        getattr(self, i),
                        getattr(other, i),
                    )
                    return False

        if hasattr(self, "rgen"):
            # pylint: disable=no-member
            if self.rgen.__class__ != other.rgen.__class__ or set(
                self.rgen.__dict__
            ) != set(other.rgen.__dict__):
                logger.debug("rgen class differs")
                return False

            # pylint: disable=no-member
            for att1, att2 in zip(self.rgen.__dict__, other.rgen.__dict__):
                # pylint: disable=no-member
                if self.rgen.__dict__[att1] != other.rgen.__dict__[att2]:
                    logger.debug(
                        "rgen class attribute %s and %s differs", att1, att2
                    )
                    return False

        return True

    def __ne__(self, other) -> bool:
        """Check if two engines are not equal."""
        return not self == other

    def __str__(self) -> str:
        """Return the string description of the integrator."""
        return self.description

    def draw_maxwellian_velocities(
        self,
        vel: np.ndarray,
        mass: np.ndarray,
        beta: float,
        sigma_v: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Draw velocities from a Gaussian distribution.

        Args:
            vel: The input velocities to modify.
            mass: The mass of the particles to generate for.
            beta: The value of Boltzmanns constant.
            sigma_v: The standard deviation to use for drawing
                velocities. One for each particle. It will be
                estimated if not explicitly given.
        """
        # TODO: Check why cp2k and turtlemd uses this differently.
        if (
            not sigma_v or sigma_v < 0.0
        ):  # TODO: The check < 0.0 is not correct for an array.
            kbt = 1.0 / beta
            sigma_v = np.sqrt(kbt * (1 / mass))

        npart, dim = vel.shape
        ### probably need a check that we have rgen.. and move this
        ### somewhere maybe.
        if hasattr(self, "rgen"):  # TODO: Not a good solution:
            vel = self.rgen.normal(loc=0.0, scale=sigma_v, size=(npart, dim))
        else:
            raise ValueError("Did not find random generator!!")
        return vel, sigma_v


def counter():
    """Return how many times this function is called."""
    counter.count = 0 if not hasattr(counter, "count") else counter.count + 1
    return counter.count
