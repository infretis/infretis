from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any

import numpy as np

from infretis.classes.engines.cp2k import kinetic_energy, reset_momentum
from infretis.classes.engines.enginebase import EngineBase
from infretis.classes.engines.engineparts import (
    ReadAndProcessOnTheFly,
    lammpstrj_reader,
)

if TYPE_CHECKING:  # pragma: nocover
    from infretis.classes.formatter import FileIO
    from infretis.classes.path import Path as InfPath
    from infretis.classes.system import System

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


def write_lammpstrj(
    outfile: str,
    id_type: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    box: np.ndarray | None,
    append: bool = False,
) -> None:
    """Write a lammps trajectory frame in .lammpstrj format
    correspondds to dump id name x y z vx vy vz
    """
    filemode = "a" if append else "w"
    with open(outfile, filemode) as writefile:
        to_write = (
            f"ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{pos.shape[0]}\n \
ITEM: BOX BOUNDS xy xz yz pp pp pp\n"
            ""
        )

        if box is not None:
            for box_vector in box:
                to_write += " ".join(box_vector.astype(str)) + "\n"
        to_write += "ITEM: ATOMS id type x y z vx vy vz\n"
        for t, x, v in zip(id_type, pos, vel):
            to_write += (
                " ".join(t.astype(int).astype(str))
                + " "
                + " ".join(x.astype(str))
                + " "
                + " ".join(v.astype(str))
                + "\n"
            )
        writefile.write(to_write)


def read_lammpstrj(
    infile: str, frame: int, n_atoms: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read a single frame from a lammps trajectory
    Note that the atoms are sorted according to their index, such that
    order calculations are correct.
    """
    block_size = n_atoms + 9
    box = np.genfromtxt(infile, skip_header=block_size * frame + 5, max_rows=3)
    posvel = np.genfromtxt(
        infile, skip_header=block_size * frame + 9, max_rows=n_atoms
    )
    id_sorted = np.argsort(posvel[:, 0])
    pos = posvel[id_sorted, 2:5]
    vel = posvel[id_sorted, 5:]
    id_type = posvel[id_sorted, :2]
    return id_type, pos, vel, box


def read_energies(filename: str) -> dict[str, np.ndarray]:
    """From pyretis
    Read some info from a LAMMPS log file.

    In particular, this method is used to read the thermodynamic
    output from a simulation (e.g. potential and kinetic energies).

    Parameters
    ----------
    filename : string
        The path to the LAMMPS log file.

    Returns
    -------
    out : dict
        A dict containing the data we found in the file.

    """
    energy_keys = []
    energy_data: dict[str, list[float | int]] = {}
    read_energy = False
    with open(filename, encoding="utf-8") as logfile:
        for lines in logfile:
            if "Step" in lines.split():
                # Assume that this is the start of the thermo output.
                energy_keys = [i.strip() for i in lines.strip().split()]
                for key in energy_keys:
                    # Note: This will discard the previously read
                    # thermodynamic data. This is because we only want
                    # to read the final section of thermodynamic data,
                    # which, by construction of the LAMMPS input file,
                    # correspond to the output of the MD run.
                    energy_data[key] = []
                read_energy = True
                continue
            if lines.startswith("Loop time"):
                # Assume this marks the end of the thermo output.
                read_energy = False
                energy_keys = []
                continue
            if read_energy and energy_keys:
                # Assume that we are reading energies.
                try:
                    data = [float(i) for i in lines.strip().split()]
                except ValueError:
                    # Some text has snuck into the thermo output,
                    # ignore it:
                    continue
                for key, value in zip(energy_keys, data):
                    if key == "Step":
                        energy_data[key].append(int(value))
                    else:
                        energy_data[key].append(value)
    return {key: np.array(val) for key, val in energy_data.items()}


def write_for_run(
    infile: str | Path,
    outfile: str | Path,
    input_settings: dict[str, Any] | None = None,
) -> None:
    """Create input file to perform n steps with lammps.

    Currently, we define a set of variables starting with `infretis_`
    at the beginning of the lammps.input template file, which are
    simply replaced in this function. See the examples for the
    lammps.input file structures.

    The variables to be replaced are:
    "infretis_timestep": the timestep,
    "infretis_nsteps": number of infretis steps = path.maxlen * self.subcycles,
    "infretis_subcycles": self.subcycles,
    "infretis_initconf": path to initial configuration to start run
    "infretis_name": name of the current project
    "infretis_lammpsdata": self.input_files["data"],
    "infretis_temperature": self.temperature,

    Parameters
    ----------
    infile : string
        The input template to use.
    outfile : string
        The file to create.
    input_settings: dict
    """
    if input_settings is None:
        input_settings = {}
    not_found = {key: 0 for key in input_settings.keys()}
    with open(infile) as readfile:
        with open(outfile, "w") as writefile:
            for line in readfile:
                spl = line.split()
                for var in input_settings.keys():
                    if var in spl:
                        line = line.replace(var, str(input_settings[var]))
                        # remove found item from dict
                        not_found.pop(var)

                writefile.write(line)
    # check if we found all keys
    if len(not_found.keys()) != 0:
        raise ValueError(
            "Did not find the following keys"
            + f"{not_found.keys()} in {os.path.abspath(infile)}"
        )


def get_atom_masses(lammps_data: str | Path) -> np.ndarray:
    """Read a lammps.data file and get the masses of the atoms"""
    n_atoms = 0
    n_atom_types = 0
    atom_type_masses = np.zeros(0)
    atoms = np.zeros(0)
    with open(lammps_data) as readfile:
        for i, line in enumerate(readfile):
            spl = line.split()
            # skip empty lines
            if not spl:
                continue
            # get number of atoms
            if len(spl) == 2 and "atoms" == spl[1]:
                n_atoms = int(spl[0])
            elif len(spl) == 3 and spl[1] == "atom" and spl[2] == "types":
                n_atom_types = int(spl[0])
            elif spl and spl[0] == "Masses":
                atom_type_masses = np.genfromtxt(
                    lammps_data, skip_header=i + 1, max_rows=n_atom_types
                ).reshape(-1, 2)
            elif spl[0] == "Atoms":
                atoms = np.genfromtxt(
                    lammps_data, skip_header=i + 1, max_rows=n_atoms
                )
    # if we didnt find all of the information
    if n_atoms == 0 or n_atom_types == 0:
        raise ValueError(
            f"Could not read atom masses from {lammps_data}. \
                         Found {n_atoms} atoms and {n_atom_types} atom_types."
        )
    masses = np.zeros((n_atoms, 1))
    for atom_type in range(1, n_atom_types + 1):
        idx = np.where(atoms[:, 2] == atom_type)[0]
        masses[idx] = atom_type_masses[atom_type - 1, 1]
    return masses


def shift_boxbounds(
    xyz: np.ndarray, box: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Shift positions such that the lower box bounds are 0, and
    return the modified positions and the new upper box bounds
    """
    xyz -= box[:, 0]
    box[:, 1] -= box[:, 0]
    return xyz, box[:, 1].flatten()


class LAMMPSEngine(EngineBase):
    """
    TO DO:
    * Add possibility to use lammps internally to modify the velocities
    such that all constraints (e.g. between bonds) are fulfilled.
    external_orderparameter : boolean

    * Velocity generation depends on units... only 'real' atm

    * Fix "exe_path" and "exe_dir", use one but not both...

    * Fix mpi in cp2k?

    * For small systems the engine is so fast that whole trajectory
    finishes before we managed to check for a crossing, leading to
    large files.

    * Make lammps remove and replace commands in input
    instead of making variables. Much easier to run md with
    the same input file

    * Periodicity in lammps has to take into accound hi and lo box bounds:
    As of now, we subtract the lower bounds from the positions and the box
    vectors to create a regular box with 0 lower bounds. This is hard coded
    in _propagate_from() when calculating, and also in _read_configuration().

    * external_orderparameter. If this is set to true, we read
    in the orderparameter from and external file. May be usefull when the
    orderparameter is calculated internally in lammps for expensive
    calculations such as in nucleation, or any other fancy stuff.
    """

    def __init__(
        self,
        lmp: str,
        input_path: str | Path,
        timestep: float,
        subcycles: int,
        temperature: float,
        exe_path: Path = Path(".").resolve(),
        sleep: float = 0.1,
    ):
        super().__init__("LAMMPS external engine", timestep, subcycles)
        self.lmp = shlex.split(lmp)
        self.name = "lammps"
        self.sleep = sleep
        self.exe_path = exe_path
        self.input_path = (Path(exe_path) / input_path).resolve()
        self.ext = "lammpstrj"

        self.input_files = {
            "data": self.input_path / "lammps.data",
            "input": self.input_path / "lammps.input",
        }
        self.mass = get_atom_masses(self.input_files["data"])
        self.n_atoms = self.mass.shape[0]
        self.kb = 1.987204259e-3  # kcal/(mol*K)
        self.temperature = temperature
        self.beta = 1 / (self.kb * self.temperature)

    def _propagate_from(
        self,
        name: str,
        path: InfPath,
        system: System,
        ens_set: dict[str, Any],
        msg_file: FileIO,
        reverse: bool = False,
    ) -> tuple[bool, str]:
        status = f"propagating with LAMMPS (reverse = {reverse})"
        interfaces = ens_set["interfaces"]
        logger.debug(status)
        success = False
        left, _, right = interfaces
        initial_conf = system.config[0]

        xyzi, veli, boxi, _ = self._read_configuration(initial_conf)
        # shift box such that lower bounds are zero
        order = self.calculate_order(system, xyz=xyzi, vel=veli, box=boxi)
        msg_file.write(
            f'# Initial order parameter: {" ".join([str(i) for i in order])}'
        )

        traj_file = os.path.join(self.exe_dir, f"{name}.{self.ext}")
        msg_file.write(f"# Trajectory file is: {traj_file}")
        # the settings we need to write in the lammps input
        input_settings = {
            "infretis_timestep": self.timestep,
            "infretis_nsteps": path.maxlen * self.subcycles,
            "infretis_subcycles": self.subcycles,
            "infretis_initconf": initial_conf,
            "infretis_name": name,
            "infretis_lammpsdata": self.input_files["data"],
            "infretis_temperature": self.temperature,
        }
        # write the file run.input from lammps input template
        run_input = os.path.join(self.exe_dir, "run.inp")
        write_for_run(self.input_files["input"], run_input, input_settings)
        # command to run lammps
        cmd = self.lmp + ["-i", run_input]
        out_name = os.path.join(self.exe_dir, "stdout.txt")
        err_name = os.path.join(self.exe_dir, "stderr.txt")
        cwd = self.exe_dir
        cmd2 = " ".join(cmd)
        # fire off lammps
        return_code = None
        lammps_was_terminated = False
        step_nr = 0
        with open(out_name, "wb") as fout, open(err_name, "wb") as ferr:
            exe = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=fout,
                stderr=ferr,
                shell=False,
                cwd=cwd,
                preexec_fn=os.setsid,
            )
            # wait for trajectories to appear
            while not os.path.exists(traj_file):
                sleep(self.sleep)
                if exe.poll() is not None:
                    logger.debug("LAMMPS execution stopped")
                    break

            # lammps may have finished after last processing the files
            # or it may have crashed without writing to the files
            if exe.poll() is None or exe.returncode == 0:
                traj_reader = ReadAndProcessOnTheFly(
                    traj_file, lammpstrj_reader
                )
                # start reading on the fly as lammps is still running
                # if it stops, perform one more iteration to read
                # the remaning contnent in the files.
                iterations_after_stop = 0
                step_nr = 0
                trajectory: list[np.ndarray] = []
                box_trajectory: list[np.ndarray] = []
                while exe.poll() is None or iterations_after_stop <= 1:
                    # we may still have some data in the trajectory
                    # so use += here
                    frames = traj_reader.read_and_process_content()
                    trajectory += frames[0]
                    box_trajectory += frames[1]
                    # loop over the frames that are ready
                    for frame in range(len(trajectory)):
                        posvel = trajectory.pop(0)
                        box = box_trajectory.pop()
                        pos = posvel[:, :3]
                        vel = posvel[:, 3:]
                        # shift the box bounds
                        pos, box = shift_boxbounds(pos, box)
                        # calculate order, check for crossings, etc
                        order = self.calculate_order(
                            system, xyz=pos, vel=vel, box=box
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
                            # process may have terminated since we last checked
                            if exe.poll() is None:
                                logger.debug("Terminating LAMMPS execution")
                                os.killpg(os.getpgid(exe.pid), signal.SIGTERM)
                                # wait for process to die, neccessary for mpi
                                exe.wait(timeout=360)
                            logger.debug(
                                "LAMMPS propagation ended at %i. Reason: %s",
                                step_nr,
                                status,
                            )
                            # exit while loop without reading additional data
                            iterations_after_stop = 2
                            lammps_was_terminated = True
                            break

                        step_nr += 1
                    sleep(self.sleep)
                    # if lammps finished, we run one more loop
                    if exe.poll() is not None and iterations_after_stop <= 1:
                        iterations_after_stop += 1

            return_code = exe.returncode
            if return_code != 0 and not lammps_was_terminated:
                logger.error(
                    "Execution of external program (%s) failed!",
                    self.description,
                )
                logger.error("Attempted command: %s", cmd2)
                logger.error("Execution directory: %s", cwd)
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
        if (return_code is not None) and (
            return_code == 0 or lammps_was_terminated
        ):
            self._removefile(out_name)
            self._removefile(err_name)

        msg_file.write("# Propagation done.")
        energy_file = os.path.join(self.exe_dir, "log.lammps")
        msg_file.write(f"# Reading energies from: {energy_file}")
        energy = read_energies(energy_file)
        end = (step_nr + 1) * self.subcycles
        ekin: np.ndarray | list[float] = energy.get("KinEng", [])
        vpot: np.ndarray | list[float] = energy.get("PotEng", [])
        path.update_energies(ekin[:end], vpot[:end])
        self._removefile(run_input)
        return success, status

    def _extract_frame(self, traj_file: str, idx: int, out_file: str) -> None:
        """Extract a frame from a trajectory and write a new configuration,
        which is a single frame trajectory"""
        id_type, pos, vel, box = read_lammpstrj(traj_file, idx, self.n_atoms)
        write_lammpstrj(out_file, id_type, pos, vel, box)

    def _read_configuration(
        self, filename: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str] | None]:
        """Read a configuration (a single frame trajectory)"""
        id_type, pos, vel, box = read_lammpstrj(filename, 0, self.n_atoms)
        pos, box = shift_boxbounds(pos, box)
        return pos, vel, box, None

    def _reverse_velocities(self, filename: str, outfile: str) -> None:
        """Reverse the velocities of a configuration"""
        id_type, pos, vel, box = read_lammpstrj(filename, 0, self.n_atoms)
        vel *= -1.0
        write_lammpstrj(outfile, id_type, pos, vel, box)

    def modify_velocities(
        self, system: System, vel_settings: dict[str, Any]
    ) -> tuple[float, float]:
        """Draw random velocities from a boltzmann distribution, and
        write a new configuration to genvel.lammpstrj.

        Basically a shortened copy from cp2k.py. Does not
        take care of constraints.
        """
        mass = self.mass
        beta = self.beta
        # energy is in units kcal/mol which we want to convert
        # to units (g/mol)*Å^2/fs (units of m*v^2), the velocity
        # units of lammps.
        # using Unitful:
        #   uconvert((u"kcal/g")^0.5, 1u"Å/fs") = 48.88821290839617
        # so we need to scale the velocities by this factor
        scale = 48.88821290839617
        pos = self.dump_frame(system)
        id_type, xyz, vel, box = read_lammpstrj(pos, 0, self.n_atoms)
        kin_old = kinetic_energy(vel, mass)[0]

        if vel_settings.get("aimless", False):
            vel, _ = self.draw_maxwellian_velocities(vel, mass, beta)
        else:
            dvel, _ = self.draw_maxwellian_velocities(
                vel, mass, beta, sigma_v=vel_settings["sigma_v"]
            )
            vel += dvel
        vel /= scale
        # reset momentum is not the default in lammps
        if vel_settings.get("zero_momentum", False):
            vel = reset_momentum(vel, mass)

        conf_out = os.path.join(
            self.exe_dir, "{}.{}".format("genvel", self.ext)
        )

        write_lammpstrj(conf_out, id_type, xyz, vel, box)

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

    def set_mdrun(
        self, config: dict[str, Any], md_items: dict[str, Any]
    ) -> None:
        """Give worker the correct random generator and executional directory,
        and eventual alternative run stuff"""
        self.exe_dir = md_items["w_folder"]
        # self.rgen = md_items['picked']['tis_set']['rgen']
        self.rgen = md_items["picked"][md_items["ens_nums"][0]]["ens"]["rgen"]
