Retis 1D example
================

Simulation
----------
task = retis
steps = 200000
interfaces = [-0.99, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, 1.0] 

System 
------
units = reduced
dimensions = 1
temperature = 0.07

Box 
---
periodic = [False]

Engine
------
class = Langevin
timestep = 0.025
gamma = 0.3
high_friction = False
seed = 0

TIS settings
------------
freq = 0.0
maxlength = 50000
aimless = True
allowmaxlength = False
zero_momentum = False
rescale_energy = False
sigma_v = -1
seed = 0
# shooting_moves = ['sh', 'sh', 'sh', 'sh', 'sh', 'sh', 'sh', 'sh']
shooting_moves = ['sh', 'sh', 'wf', 'wf', 'wf', 'wf', 'wf', 'wf']
n_jumps = 20
high_accept = True
# interface_cap = -0.20

RETIS settings
--------------
swapfreq = 0.5
relative_shoots = None
nullmoves = True
swapsimul = True

Initial-path settings
---------------------
method = kick
kick-from = initial

Particles
---------
position = {'input_file': 'initial.xyz'}
mass = {'Ar': 1.0}
name = ['Ar']
ptype = [0]

Forcefield settings
-------------------
description = 1D double well

Potential
---------
class = DoubleWell
a = 1.0
b = 2.0
c = 0.0

Orderparameter
--------------
class = Position
dim = x
index = 0
periodic = False

Output
------
backup = backup
order-file = -1
trajectory-file = -1
energy-file = -1
