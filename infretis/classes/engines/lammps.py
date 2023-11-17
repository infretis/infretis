import logging
import os
import subprocess
from time import sleep

import MDAnalysis as mda
import numpy as np

from infretis.classes.engines.cp2k import kinetic_energy, reset_momentum
from infretis.classes.engines.enginebase import EngineBase
from infretis.classes.engines.engineparts import (
    ReadAndProcessOnTheFly,
    lammpstrj_processer,
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


def read_lammpstrj(infile, frame, n_atoms):
    # Read a single frame from a lammps trajectory
    block_size = n_atoms + 9
    box = np.genfromtxt(infile, skip_header=block_size * frame + 5, max_rows=3)
    posvel = np.genfromtxt(
        infile, skip_header=block_size * frame + 9, max_rows=n_atoms
    )
    return posvel[:, :2], posvel[:, 2:5], posvel[:, 5:], box


def read_energies(exe_dir, filename):
    """
    From pyretis
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
    filename = exe_dir + filename
    energy_keys = []
    energy_data = {}
    read_energy = False
    with open(filename, encoding="utf-8") as logfile:
        for lines in logfile:
            if lines.startswith("Step"):
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
    for key, val in energy_data.items():
        energy_data[key] = np.array(val)

    return energy_data


def write_lammpstrj(outfile, id_type, pos, vel, box):
    # Write a lammps trajectory frame
    with open(outfile, "w") as writefile:
        to_write = (
            f"ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{pos.shape[0]}\n \
ITEM: BOX BOUNDS xy xz yz pp pp pp\n"
            ""
        )

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


def write_for_run(
    infile,
    outfile,
    timestep,
    nsteps,
    subcycles,
    initconf,
    lammpsdata,
    name="md-run",
):
    """Create input file to perform n steps with lammps.

    Note, a single step actually consists of a number of subcycles.
    But from infretis' point of view, this is a single step.
    Further, we here assume that we start from a given lammpstrj file.

    Parameters
    ----------
    infile : string
        The input template to use.
    outfile : string
        The file to create.
    timestep : float
        The time-step to use for the simulation.
    nsteps : integer
        The number of infretis steps to perform.
    subcycles : integer
        The number of sub-cycles to perform.
    initconf : string
        The name for the input file to read positions
        and velocities from
    name : string, optional
        A name for the lammps trajectories and energies.
    """
    replace = {
        "infretis_timestep": timestep,
        "infretis_nsteps": nsteps * subcycles,
        "infretis_subcycles": subcycles,
        "infretis_initconf": initconf,
        "infretis_name": name,
        "infretis_lammpsdata": lammpsdata,
    }
    not_found = {key: 0 for key in replace.keys()}
    with open(infile) as readfile:
        with open(outfile, "w") as writefile:
            for line in readfile:
                spl = line.split()
                for var in replace.keys():
                    if var in spl:
                        line = line.replace(var, str(replace[var]))
                        # remove found item from dict
                        not_found.pop(var)

                writefile.write(line)
    # check if we found all keys
    if len(not_found.keys()) != 0:
        for key in not_found.keys():
            print(f"Did not find {key} variable in {infile}!")


class LAMMPSEngine(EngineBase):
    """
    TO DO:
    * Use lammps internally to modify the velocities such that all
    constraints (e.g. between bonds) are fulfilled.
    external_orderparameter : boolean

    * external_orderparemeter
        If this is set to true, we read in the orderparameter from
        and external file. May be usefull when the orderparameter
        is calculated internally in lammps for expensive calculations
        such as in nucleation, or any other fancy stuff.

    """

    external_orderparameter = False

    def __init__(
        self,
        lmp,
        input_path,
        timestep,
        subcycles,
        exe_path=os.path.abspath("."),
        sleep=0.1,
    ):
        super().__init__("LAMMPS external engine", timestep, subcycles)
        self.lmp = lmp
        self.sleep = sleep
        self.input_path = input_path
        self.exe_path = exe_path
        self.ext = "lammpstrj"

        default_files = {
            "data": f"{input_path}/lammps.data",
            "input": f"{input_path}/lammps.input",
        }
        print("=" * 40)
        print("Initializing LAMMPS engine")
        print(f"Lammps executable: {self.lmp}")
        print(f"Input file path: {self.input_path}")
        print(f"Executional file path: {self.exe_path}")
        print("Using MDAnalysis")
        self.u = mda.Universe(default_files["data"])
        self.mass = self.u.atoms.masses.reshape(-1, 1)
        self.n_atoms = len(self.mass)
        self.kb = 1.987204259e-3  # kcal/(mol*K)
        print("Assuming a temperature of 300K!")
        self.temperature = 300
        self.beta = 1 / (self.kb * self.temperature)

    def _propagate_from(
        self, name, path, system, ens_set, msg_file, reverse=False
    ):
        out_files = {
            "traj": "test.lammpsdump",
            "energy": "log.lammps",
        }

        print("Assuming out_files are ", out_files)
        interfaces = ens_set["interfaces"]
        left, _, right = interfaces
        initial_conf = system.config[0]
        box, xyz, vel = self._read_configuration(initial_conf)
        order = self.calculate_order(system, xyz=xyz, vel=vel, box=box)
        traj_file = os.path.join(self.exe_dir, f"{name}.{self.ext}")
        msg_file.write(
            f'# Initial order parameter: {" ".join([str(i) for i in order])}'
        )
        msg_file.write(f"# Trajectory file is: {traj_file}")
        # fire off lammps
        cmd = self.lmp + ["-i", "lammps.input"]
        print(f"Running lammps with {' '.join(cmd)}")
        out_name = "stdout.txt"
        err_name = "stderr.txt"
        cwd = self.exe_dir
        input("continue?")
        with open(out_name, "wb") as fout, open(err_name, "wb") as ferr:
            exe = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=fout,
                stderr=ferr,
                shell=False,
                cwd=cwd,
            )
            # wait for trajectories to appear
            while not os.path.exists(out_files["pos"]) or not os.path.exists(
                out_files["traj"]
            ):
                sleep(self.sleep)
                if exe.poll() is not None:
                    logger.debug("LAMMPS execution stopped")
                    break
            traj_reader = ReadAndProcessOnTheFly(
                out_files["traj"], lammpstrj_processer
            )
            for frame, box in traj_reader.read_and_process_content():
                pos = frame[:, :3]
                vel = frame[:, 3:]
                print(pos, vel, box)

    def _extract_frame(self, traj_file, idx, out_file):
        print(f"_extract_frame from {traj_file} at idx {idx}")
        id_type, pos, vel, box = read_lammpstrj(traj_file, idx, self.n_atoms)
        print(f"_extraxct_frame writing out file {out_file}")
        write_lammpstrj(out_file, id_type, pos, vel, box)
        return

    def _read_configuration(self, filename):
        print(f"read_configuration {filename}")
        id_type, pos, vel, box = read_lammpstrj(filename, 0, self.n_atoms)
        return id_type, box, pos, vel

    def _reverse_velocities(self, filename, outfile):
        print("Reversing velocities")
        id_type, pos, vel, box = read_lammpstrj(filename, 0, self.n_atoms)
        vel *= -1.0
        write_lammpstrj(outfile, id_type, pos, vel, box)

    def modify_velocities(self, system, vel_settings=None):
        """
        Basically a shortened copy from cp2k.py. Does not
        take care of constraints.

        Draw random velocities from a boltzmann distribution.
        """
        mass = self.mass
        beta = self.beta
        pos = self.dump_frame(system)
        id_type, box, xyz, vel = self._read_configuration(pos)

        kin_old = kinetic_energy(vel, mass)[0]

        if vel_settings.get("aimless", False):
            vel, _ = self.draw_maxwellian_velocities(vel, mass, beta)
        else:
            dvel, _ = self.draw_maxwellian_velocities(
                vel, mass, beta, sigma_v=vel_settings["sigma_v"]
            )
            vel += dvel
        # make reset momentum the default
        if vel_settings.get("zero_momentum", True):
            vel = reset_momentum(vel, mass)

        conf_out = os.path.join(
            self.exe_dir, "{}.{}".format("genvel", self.ext)
        )

        write_lammpstrj(conf_out, id_type, xyz, vel, box)

        kin_new = kinetic_energy(vel, mass)[0]
        system.config = (conf_out, None)
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

    def set_mdrun(self, config, md_items):
        """Remove or rename?"""
        self.exe_dir = md_items["w_folder"]
        # self.rgen = md_items['picked']['tis_set']['rgen']
        self.rgen = md_items["picked"][md_items["ens_nums"][0]]["ens"]["rgen"]
