RETIS test for GROMACS
======================

Simulation
----------
task = retis
steps = 10
interfaces = [-0.26, -0.24, -0.22]

System
------
units = gromacs

Engine
------
class = GromacsEngine2R
module = gromacs.py
gmx = gmx
mdrun = gmx mdrun
input_path = ../../../test/examples/external/data/gromacs_input
timestep = 0.002
subcycles = 5
gmx_format = g96
maxwarn = 15

TIS
---
freq =  0.5
maxlength = 20000
aimless = True
allowmaxlength = False
zero_momentum = False
rescale_energy = False
# shooting_moves = ['sh', 'sh', 'wf']
shooting_moves = ['sh', 'sh', 'sh']
n_jumps = 3
sigma_v = -1
seed = 0

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
class = RingDiffusion
module = orderp.py

Output settings
---------------
pathensemble-file = 1
screen = 10
order-file = 1
energy-file = 1
trajectory-file = 1
