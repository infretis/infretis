NaCl in Water
===================================

Simulation
----------
task = retis
steps = 10000
interfaces = [0.27, .28, 0.30, 0.32, .45]

System
------
units = gromacs

Box settings
------------
cell = [3, 3, 3]
periodic = [True, True, True]

Engine settings
---------------
class = gromacs2    # more effective than gromacs class
# gmx = gmx_mpi
gmx = gmx
# mdrun = srun --exclusive --ntasks 2 --mem-per-cpu 500 gmx_mpi mdrun -ntomp 1
mdrun = gmx mdrun
input_path = gromacs_input
timestep = 0.0005
subcycles = 1
gmx_format = g96
maxwarn = 15

TIS settings
------------
freq = 0.5
maxlength = 100000
aimless = True
allowmaxlength = False
zero_momentum = False
rescale_energy = False
sigma_v = -1
seed = 0
# shooting_moves = ['sh', 'sh', 'wf', 'wf', 'wf']
shooting_moves = ['sh', 'sh', 'sh', 'sh', 'sh']
high_accept = True
n_jumps = 2

RETIS settings
--------------
swapfreq = 0.0
relative_shoots = None
nullmoves = True
swapsimul = True

Initial-path
------------
method = load
load_folder = trajs

Orderparameter
--------------
class = Distance
index = (0, 1)
periodic = False

Output settings
---------------
backup = 'backup'
order-file = 1
energy-file = 1
trajectory-file = 1
screen = 1


