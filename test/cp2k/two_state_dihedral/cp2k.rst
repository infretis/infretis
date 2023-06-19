Retis 1D example
================

Simulation settings
-------------------
task = 'retis'
steps = 10100
interfaces = [-1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 1.5]
exe_path = '/home/lukas/myinf/infretis/test/cp2k/pyretis'

System settings
---------------
units = 'cp2k'
temperature = 0.07

Engine settings
---------------
class = 'cp2k'
cp2k = '/home/lukas/cp2k-2023.1/exe/local/cp2k.ssmp'
input_path = 'cp2k_input'
timestep = 50.0
subcycles = 10
cp2k_format = 'xyz'
#extra_files = ['BASIS_SET', 'GTH_POTENTIALS']

Orderparameter settings
-----------------------
class = 'Dihedral'
index = (0, 1, 2, 3)
periodic = False

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
#method = 'restart'
method = 'kick'
kick-from = 'previous'

RETIS settings
--------------
swapfreq = 0.5
relative_shoots = None
nullmoves = True
swapsimul = True

