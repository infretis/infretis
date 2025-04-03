import os
import sys
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.io import read
from ase.io.trajectory import Trajectory
from ase import units as u
from infretis.classes.formatter import FileIO, OutputFormatter
from infretis.core.core import create_external
from infretis.classes.engines.enginebase import EngineBase
import sys
from infretis.classes.engines.ase_external_engine import read_stuff, dump_stuff
import time
import pathlib

# if idle for more than this nr. of seconds we shut down the engine
# TIMEOUT = 1000
START = "INFINITY_START"
SLEEP = 1.5

sys.path.append(os.path.abspath("."))
from mace.calculators import mace_mp
from orderparam import LinearCombination


# calc and orderfunction only need to be set up once
calc = mace_mp("medium")
order_function = LinearCombination()


# set exe_dir, e.g. worker0/
cwd = os.path.abspath(sys.argv[1])
wname = pathlib.Path(cwd).name
logger = open(f"{wname}_propagator.log", "a")
if not cwd:
    raise ValueError("No exe_dir specified!")
if not os.path.isdir(cwd):
    print(f"Did not find {cwd}/ dir, making it now.", file=logger, flush=True)
    os.mkdir(cwd)
# change directory
print(f"Changing dir to {cwd}")
os.chdir(cwd)

# these do not change so we don't need to set them up more than once
calculator = mace_mp("medium")
order_function = LinearCombination()

idle_time = 0

while True:
    # wait for start file to appear
    if not os.path.exists(START):
        time.sleep(SLEEP)
        print(f"{wname}: Sleeping ... now been idle for {idle_time} s", file = logger, flush=True)
        idle_time += SLEEP
    else:
        print(f"{wname}: Found {START} file", file=logger, flush=True)
        # try to read start file
        while True:
            if not os.path.exists(START):
                print(f"{wname}: Now the START file is missing... Now idle for {idle_time}", file=logger, flush=True)
                time.sleep(SLEEP)
                idle_time += SLEEP
            else:
                with open(START, "r") as rfile:
                    line = rfile.readline()
                    spl = line.split()
                    if len(spl) != 6:
                        print(f"{wname}: STARTFILE_ERR not 6 columns in file; content is '{spl}'. Now idle for {idle_time} s", file = logger, flush=True)
                        time.sleep(SLEEP)
                        idle_time += SLEEP
                    # finally escape the loop if we get 6 values
                    else:
                        print(f"{wname} " + line, file=logger, flush=True)
                        initial_conf, subcycles, traj_file, cwd, msg_file_name, input_path = line.split()
                        subcycles = int(subcycles)
                        break

        system = read_stuff("system", cwd)
        path = read_stuff("path", cwd)
        ens_set = read_stuff("ens_set", cwd)
        Integrator = read_stuff("Integrator", cwd)
        int_set = read_stuff("int_set", cwd)
        reverse = read_stuff("reverse", cwd)
        left = read_stuff("left", cwd)
        right = read_stuff("right", cwd)

        # create engine and order function
        atoms = read(initial_conf)
        if isinstance(atoms, list):
            atoms = atoms[0]

        msg_file = FileIO(
            msg_file_name, "a", OutputFormatter("MSG_File"), backup=False
        )
        msg_file.open()

        dyn = Integrator(atoms, **int_set)
        traj = traj = Trajectory(traj_file, "w")
        step_nr = 0
        ekin = []
        vpot = []
        calc.calculate(atoms)
        atoms.calc = calc
        t0 = time.time()
        # integrator step is taken at the end of every loop,
        # such that frame 0 is also written
        print(f"{wname}: Starting {subcycles*path.maxlen} steps", file=logger, flush=True)
        for i in range(subcycles * path.maxlen):
            #print(f"step {i} of {subcycles*path.maxlen}", file=logger, flush=True)
            energy = calc.results["energy"]
            forces = calc.results["forces"]
            stress = calc.results.get("stress", None)
            if (i) % (subcycles) == 0:
                ekin.append(atoms.get_kinetic_energy())
                vpot.append(calc.results["energy"])
                # NOTE: Writing atoms removes all results from
                # the calculator (and therefore atoms)!
                traj.write(atoms, forces=forces, energy=energy, stress=stress)
                system.pos = atoms.positions
                system.vel = atoms.get_velocities()
                system.box = atoms.cell.diagonal()
                order = order_function.calculate(system)
                msg_file.write(
                    f'{step_nr} {" ".join([str(j) for j in order])}'
                )
                snapshot = {
                    "order": order,
                    "config": (traj_file, step_nr),
                    "vel_rev": reverse,
                }
                phase_point = EngineBase.snapshot_to_system(system, snapshot)
                status, success, stop, add = EngineBase.add_to_path(
                    path, phase_point, left, right
                )

                if stop:
                    break
                step_nr += 1
            dyn.step(forces=forces)
        t1 = time.time()

        msg_file.write("# Propagation done.")
        msg_file.close()
        traj.close()
        path.update_energies(ekin, vpot)
        dump_stuff(["path"], [path], cwd)
        dump_stuff(["success","status"], [success, status], cwd)
        # avoid divide by zero err
        if step_nr == 0:
            step_nr += 1
        print(f"{wname}: propagation done {(t1-t0)/(step_nr)} s/step.", flush=True, file=logger)
        os.remove(START)
        idle_time = 0
