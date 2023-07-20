NaCl in Water
===================================

Simulation
----------
task = retis
steps = 10000
interfaces = [0.27, .28, 0.30, 0.32, 0.325, 0.33, 0.45]

System
------
units = gromacs

Engine
------
class = gromacs2    # more effective than gromacs class
# gmx = srun gmx_d
# gmx = mpirun -np 1 gmx_d
# gmx = gmx
# mdrun = gmx mdrun -nt 4 -pinoffset 0 -pin on
gmx = gmx
mdrun = gmx mdrun
input_path = ./../salt_data/gromacs_input
timestep = 0.0005
subcycles = 1
gmx_format = g96
maxwarn = 15

TIS
---
freq = 0.5
maxlength = 20000
aimless = True
allowmaxlength = False
zero_momentum = False
rescale_energy = False
shooting_moves = ['sh', 'sh', 'wf', 'wf', 'wf', 'wf', 'wf']
# shooting_moves = ['sh', 'sh', 'sh', 'sh', 'sh']
n_jumps = 3
sigma_v = -1
seed = 0
high_accept = True

Initial-path
------------
method = restart
load_folder = trajs

RETIS settings
--------------
swapfreq = 0.5
relative_shoots = None
nullmoves = True
swapsimul = True

Orderparameter
--------------
class = Distance
index = (0, 1)
periodic = False

Output settings
---------------
pathensemble-file = 1
screen = 10
order-file = 1
energy-file = 1
trajectory-file = 1
