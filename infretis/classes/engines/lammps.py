import logging
import os
import subprocess

import MDAnalysis as mda
import sleep

from infretis.classes.engines.enginebase import EngineBase

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


class LAMMPSEngine(EngineBase):
    """
    external_orderparameter : boolean
        If this is set to true, we read in the orderparameter from
        and external file. May be usefull when the orderparameter
        is calculated internally in lammps.

    Trajectory format:
        nestcdf and h5md need external libs
        use lammpstrj for now so it works out of the box
        h5md (+) easy for us because we can read and write easily.
        pyinotify (+) for checking when something changed?
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

    def _extract_frame(self, traj_file, idx, out_file):
        print(f"_extract_frame from {traj_file} at idx {idx}")
        self.u.load_new(traj_file)
        self.u.trajectory[idx]
        print(f"_extraxct_frame writing out file {out_file}")
        self.u.atoms.write(out_file)
        return

    def _propagate_from(
        self, name, path, system, ens_set, msg_file, reverse=False
    ):
        out_files = {
            "pos": "test.dcd",
            "vel": "test.dcd",
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
                out_files["vel"]
            ):
                sleep(self.sleep)
                if exe.poll() is not None:
                    logger.debug("LAMMPS execution stopped")
                    break

    def _read_configuration(self, filename):
        print(f"read_configuration {filename}")
        self.u.load_new(filename)
        box = self.u.dimensions
        xyz = self.u.atoms.positions
        vel = xyz * 0  # self.u.atoms.velocities
        return box, xyz, vel

    def _reverse_velocities(self, filename, outfile):
        self._read_configuration(filename)
        self.u.atoms.velocities *= -1.0
        self.u.atoms.write(outfile)

    def modify_velocities(self, system, vel_settings):
        # dont do anything
        return 0.0, system.ekin

    def set_mdrun(self, config, md_items):
        print("set_mdrun setting exe_dir to {md_items['w_folder']}")
        self.exe_dir = md_items["w_folder"]
