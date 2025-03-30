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


initial_conf = sys.argv[1]
subcycles = int(sys.argv[2])
traj_file = sys.argv[3]
cwd = sys.argv[4]
msg_file_name = sys.argv[5]
input_path = sys.argv[6]

system = read_stuff("system", cwd)
path = read_stuff("path", cwd)
ens_set = read_stuff("ens_set", cwd)
Integrator = read_stuff("Integrator", cwd)
int_set = read_stuff("int_set", cwd)
calc_set = read_stuff("calc_set", cwd)
order_settings = read_stuff("order_settings", cwd)
reverse = read_stuff("reverse", cwd)
left = read_stuff("left", cwd)
right = read_stuff("right", cwd)

# create engine and order function
calc_set["simulation"] = {"exe_path": input_path}
calc = create_external(calc_set, "ASE calculator", [])
order_settings["simulation"] = {"exe_path": input_path}
order_function = create_external(order_settings, "Order Parameter", [])

atoms = read(initial_conf)
if isinstance(atoms, list):
    atoms = atoms[0]

dyn = Integrator(atoms, **int_set)
traj = traj = Trajectory(traj_file, "w")
calc.calculate(atoms)
atoms.calc = calc
step_nr = 0
ekin = []
vpot = []
msg_file = FileIO(
    msg_file_name, "a", OutputFormatter("MSG_File"), backup=False
)
msg_file.open()
# integrator step is taken at the end of every loop,
# such that frame 0 is also written
for i in range(subcycles * path.maxlen):
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

msg_file.write("# Propagation done.")
msg_file.close()
traj.close()
path.update_energies(ekin, vpot)
dump_stuff(["path"], [path], cwd)
dump_stuff(["success","status"], [success, status], cwd)
