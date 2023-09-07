Retis 1D example
================

Simulation settings
-------------------
task = 'retis'
steps = 10000
interfaces = [0.345 , 0.355, 0.375, 0.415, 0.495, 0.560, 0.65, 1.200]

System settings
---------------
units = 'gromacs'
temperature = 300

Engine settings
---------------
class = 'gmx'
gmx = 'gmx'
input_path = 'gromacs_input'
timestep = 0.0002
subcycles = 10
gmx_format = 'g96'

Orderparameter settings
-----------------------
class = 'Distance'
index = (0, 1)
periodic = True

Output settings
---------------
order-file = 1
restart-file = 1
trajectory-file = 10
energy-file = 1
pathensemble-file = 1

TIS settings
------------
freq = 0.5
maxlength = 20000
aimless = True
allowmaxlength = False
zero_momentum = True
rescale_energy = False
sigma_v = -1
seed = 0
high_accept = False
shooting_move = 'sh'

Initial-path settings
---------------------
method = 'kick'
kick-from = 'previous'

RETIS settings
--------------
swapfreq = 0.5
relative_shoots = None
nullmoves = True
swapsimul = True
