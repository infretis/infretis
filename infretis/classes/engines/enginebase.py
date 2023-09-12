"""Base engine class."""
from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from infretis.classes.formatter import FileIO

if TYPE_CHECKING:  # pragma: no cover
    from infretis.classes.orderparameter import OrderParameter
    from infretis.classes.path import Path
    from infretis.classes.system import System


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class EngineBase(metaclass=ABCMeta):
    """Abstract base class for engines.

    The engines perform molecular dynamics (or Monte Carlo) and they
    are assumed to act on a system. Typically they will integrate
    Newtons equation of motion in time for that system.

    Attributes
    ----------
    description : string
        Short string description of the engine. Used for printing
        information about the integrator.
    exe_dir : string
        A directory where the engine is going to be executed.
    needs_order : boolean
        Determines if the engine needs an internal order parameter
        or not. If not, it is assumed that the order parameter is
        calculated by the engine.

    """

    needs_order: bool = True

    def __init__(self, description: str, timestep: float, subcycles: int):
        """Just add the description."""
        self.description: str = description
        self._exe_dir: str = "."
        self.timestep: float = timestep
        self.subcycles: int = subcycles
        self.ext: str = "xyz"
        self.input_files: dict[str, str] = {}
        self.order_function: OrderParameter | None = None

    @property
    def exe_dir(self) -> str:
        """Return the directory we are currently using."""
        return self._exe_dir

    @exe_dir.setter
    def exe_dir(self, exe_dir: str):
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
        path: Path, phase_point: System, left: float, right: float
    ) -> tuple[str, bool, bool, bool]:
        """
        Add a phase point and perform some checks.

        This method is intended to be used by the propagate methods.

        Parameters
        ----------
        path : object like :py:class:`.Path`
            The path to add to.
        phase_point : object like py:class:`.System`
            The phase point to add to the path.
        left : float
            The left interface.
        right : float
            The right interface.

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
        self, ensemble: dict[str, Any], vel_settings: dict[str, Any]
    ) -> tuple[float, float]:
        """Modify the velocities of the current state.

        Parameters
        ----------
        ensemble: dict
            It contains all the runners:

            * `path` : object like :py:class:`.PathBase`
              This is the path we use to fill in phase-space points.
              We are here not returning a new path - this since we want
              to delegate the creation of the path (type) to the method
              that is running `propagate`.

        vel_settings: dict
            It contains all the info for the velocity:

            * `sigma_v` : numpy.array, optional
              These values can be used to set a standard deviation (one
              for each particle) for the generated velocities.
            * `aimless` : boolean, optional
              Determines if we should do aimless shooting or not.
            * `zero_momentum` : boolean, optional
              If True, we reset the linear momentum to zero after
              generating.
            * `rescale or rescale_energy` : float, optional
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

    @abstractmethod
    def set_mdrun(self, config: dict[str, Any], md_items: dict[str, Any]):
        """Sets the worker terminal command to be run"""

    def calculate_order(
        self,
        system: System,
        xyz: np.ndarray | None = None,
        vel: np.ndarray | None = None,
        box: np.ndarray | None = None,
    ) -> list[float]:
        """
        Calculate order parameter from configuration in a file.

        Note, if ``xyz``, ``vel`` or ``box`` are given, we will
        **NOT** read positions, velocity and box information from the
        current configuration file.

        Parameters
        ----------
        ensemble : dict
            It contains:

            * `system` : object like :py:class:`.System`
              This is the system that contains the particles we are
              investigating
            * `order_function` : object like :py:class:`.OrderParameter`
              The object used for calculating the order parameter.

        xyz : numpy.array, optional
            The positions to use, in case we have already read them
            somewhere else. We will then not attempt to read them again.
        vel : numpy.array, optional
            The velocities to use, in case we already have read them.
        box : numpy.array, optional
            The current box vectors, in case we already have read them.

        Returns
        -------
        out : list of floats
            The calculated order parameter(s).

        """
        # Convert system into an internal representation:
        if any((xyz is None, vel is None, box is None)):
            out = self._read_configuration(system.config[0])
            box = out[0]
            xyz = out[1]
            vel = out[2]
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

    def dump_phasepoint(self, phasepoint: System, deffnm: str = "conf"):
        """Just dump the frame from a system object."""
        pos_file = self.dump_config(phasepoint.config, deffnm=deffnm)
        phasepoint.set_pos((pos_file, None))

    def _name_output(self, basename: str):
        """
        Create a file name for the output file.

        This method is used when we dump a configuration to add
        the correct extension for GROMACS (either gro or g96).

        Parameters
        ----------
        basename : string
            The base name to give to the file.

        Returns
        -------
        out : string
            A file name with an extension.

        """
        out_file = f"{basename}.{self.ext}"
        return os.path.join(self.exe_dir, out_file)

    def dump_config(
        self, config: tuple[str, int], deffnm: str = "conf"
    ) -> str:
        """Extract configuration frame from a system if needed.

        Parameters
        ----------
        config : tuple
            The configuration given as (filename, index).
        deffnm : string, optional
            The base name for the file we dump to.

        Returns
        -------
        out : string
            The file name we dumped to. If we did not in fact dump, this is
            because the system contains a single frame and we can use it
            directly. Then we return simply this file name.

        Note
        ----
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
        """Just dump the frame from a system object."""
        return self.dump_config(system.config, deffnm=deffnm)

    @abstractmethod
    def _extract_frame(self, traj_file: str, idx: int, out_file: str):
        """Extract a frame from a trajectory file.

        Parameters
        ----------
        traj_file : string
            The trajectory file to open.
        idx : integer
            The frame number we look for.
        out_file : string
            The file to dump to.

        """

    def clean_up(self):
        """Will remove all files from the current directory."""
        dirname = self.exe_dir
        logger.debug('Running engine clean-up in "%s"', dirname)
        files = [item.name for item in os.scandir(dirname) if item.is_file()]
        if dirname is not None:
            self._remove_files(dirname, files)

    def propagate(
        self,
        path: Path,
        ens_set: dict[str, Any],
        system: System,
        reverse: bool = False,
    ) -> tuple[bool, str]:
        """
        Propagate the equations of motion with the external code.

        This method will explicitly do the common set-up, before
        calling more specialised code for doing the actual propagation.

        Parameters
        ----------
        path : object like :py:class:`.Path`
            This is the path we use to fill in phase-space points.
            We are here not returning a new path - this since we want
            to delegate the creation of the path to the method
            that is running `propagate`.
        ensemble: dict
            It contains:

            * `system` : object like :py:class:`.System`
              The system object gives the initial state for the
              integration. The initial state is stored and the system is
              reset to the initial state when the integration is done.
            * `order_function` : object like :py:class:`.OrderParameter`
              The object used for calculating the order parameter.
            * `interfaces` : list of floats
              These interfaces define the stopping criterion.

        reverse : boolean, optional
            If True, the system will be propagated backward in time.

        Returns
        -------
        success : boolean
            This is True if we generated an acceptable path.
        status : string
            A text description of the current status of the propagation.

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
        msg_file = FileIO(msg_file_name, "w", None, backup=False)
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
        system.set_pos((initial_conf, None))
        system.set_vel(reverse)
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
        path: Path,
        system: System,
        ensemble: dict[str, Any],
        msg_file: FileIO,
        reverse: bool = False,
    ) -> tuple[bool, str]:
        """
        Run the actual propagation using the specific engine.

        This method is called after :py:meth:`.propagate`. And we
        assume that the necessary preparations before the actual
        propagation (e.g. dumping of the configuration etc.) is
        handled in that method.

        Parameters
        ----------
        name : string
            A name to use for the trajectory we are generating.
        path : object like :py:class:`.PathBase`
            This is the path we use to fill in phase-space points.
        ensemble: dict
            It contains:

            * `system` : object like :py:class:`.System`
              The system object gives the initial state for the
              integration. The initial state is stored and the system is
              reset to the initial state when the integration is done.
            * `order_function` : object like :py:class:`.OrderParameter`
              The object used for calculating the order parameter.
            * `interfaces` : list of floats
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
    def step(self, system: System, name: str) -> str:
        """Perform a single step with the external engine.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The system we are integrating.
        name : string
            To name the output files from the external engine.

        Returns
        -------
        out : string
            The name of the output configuration, obtained after
            completing the step.

        """

    @abstractmethod
    def _read_configuration(
        self, filename: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read output configuration from external software.

        Parameters
        ----------
        filename : string
            The file to open and read a configuration from.

        Returns
        -------
        out[0] : numpy.array
            The dimensions of the simulation box.
        out[1] : numpy.array
            The positions found in the given filename.
        out[2] : numpy.array
            The velocities found in the given filename.

        """

    @abstractmethod
    def _reverse_velocities(self, filename: str, outfile: str):
        """Reverse velocities in a given snapshot.

        Parameters
        ----------
        filename : string
            Input file with velocities.
        outfile : string
            File to write with reversed velocities.

        """

    @staticmethod
    def _modify_input(
        sourcefile: str,
        outputfile: str,
        settings: dict[str, str],
        delim: str = "=",
    ):
        """
        Modify input file for external software.

        Here we assume that the input file has a syntax consisting of
        ``keyword = setting``. We will only replace settings for
        the keywords we find in the file that is also inside the
        ``settings`` dictionary.

        Parameters
        ----------
        sourcefile : string
            The path of the file to use for creating the output.
        outputfile : string
            The path of the file to write.
        settings : dict
            A dictionary with settings to write.
        delim : string, optional
            The delimiter used for separation keywords from settings.

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
        sourcefile: str, delim: str = "="
    ) -> dict[str, str]:
        """
        Read input settings for simulation input files.

        Here we assume that the input file has a syntax consisting of
        ``keyword = setting``, where ``=`` can be any string given
        in the input parameter ``delim``.

        Parameters
        ----------
        sourcefile : string
            The path of the file to use for creating the output.
        delim : string, optional
            The delimiter used for separation keywords from settings.

        Returns
        -------
        settings : dict of strings
            The settings found in the file.

        Note
        ----
        Important: We are here assuming that there will *ONLY* be one
        keyword per line.

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
        """
        Execute an external command for the engine.

        We are here executing a command and then waiting until it
        finishes. The standard out and standard error are piped to
        files during the execution and can be inspected if the
        command fails. This method returns the return code of the
        issued command.

        Parameters
        ----------
        cmd : list of strings
            The command to execute.
        cwd : string or None, optional
            The current working directory to set for the command.
        inputs : bytes or None, optional
            Additional inputs to give to the command. These are not
            arguments but more akin to keystrokes etc. that the external
            command may take.

        Returns
        -------
        out : int
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
    def _movefile(source: str, dest: str):
        """Move file from source to destination."""
        logger.debug("Moving: %s -> %s", source, dest)
        shutil.move(source, dest)

    @staticmethod
    def _copyfile(source: str, dest: str):
        """Copy file from source to destination."""
        logger.debug("Copy: %s -> %s", source, dest)
        shutil.copyfile(source, dest)

    @staticmethod
    def _removefile(filename: str):
        """Remove a given file if it exist."""
        try:
            os.remove(filename)
            logger.debug("Removing: %s", filename)
        except OSError:
            logger.debug("Could not remove: %s", filename)

    def _remove_files(self, dirname: str, files: list[str]):
        """Remove files from a directory.

        Parameters
        ----------
        dirname : string
            Where we are removing.
        files : list of strings
            A list with files to remove.

        """
        for i in files:
            self._removefile(os.path.join(dirname, i))

    def __eq__(self, other):
        """Check if two engines are equal."""
        if self.__class__ != other.__class__:
            logger.debug("%s and %s.__class__ differ", self, other)
            return False

        if set(self.__dict__) != set(other.__dict__):
            logger.debug("%s and %s.__dict__ differ", self, other)
            return False

        for i in ["needs_order", "description", "_exe_dir", "timestep"]:
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

    def __ne__(self, other):
        """Check if two engines are not equal."""
        return not self == other

    @classmethod
    def can_use_order_function(cls, order_function: OrderParameter):
        """Fail if the engine can't be used with an empty order parameter."""
        if order_function is None and cls.needs_order:
            raise ValueError(
                "No order parameter was defined, but the "
                "engine *does* require it."
            )

    def restart_info(self) -> dict[str, str]:
        """General method.

        Returns the info to allow an engine exact restart.

        Returns
        -------
        info : dict
            Contains all the updated simulation settings and counters.

        """
        info = {"description": self.description}
        return info

    def load_restart_info(self, info: dict[str, str] | None = None):
        """Load restart information.

        Parameters
        ----------
        info : dict
            The dictionary with the restart information, should be
            similar to the dict produced by :py:func:`.restart_info`.

        """
        if info is not None:
            self.description = info["description"]

    def __str__(self):
        """Return the string description of the integrator."""
        return self.description

    def draw_maxwellian_velocities(
        self,
        vel: np.ndarray,
        mass: np.ndarray,
        beta: np.ndarray,
        sigma_v: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw numbers from a Gaussian distribution.

        Parameters
        ----------
        system : object like :py:class:`.System`
            This is used to determine the shape (number of particles
            and dimensionality) and requires veloctities.
        engine : object like :py:class:`.Engine`
            This is used to determine the temperature parameter(s)
        sigma_v : numpy.array, optional
            The standard deviation in velocity, one for each particle.
            If it's not given it will be estimated.

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
        return vel, sigma_v


def counter():
    """Return how many times this function is called."""
    counter.count = 0 if not hasattr(counter, "count") else counter.count + 1
    return counter.count
