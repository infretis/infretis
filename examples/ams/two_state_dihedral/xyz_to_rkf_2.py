#!/usr/bin/env amspython
from scm.plams import *
from scm.plams.tools.converters import traj_to_rkf, file_to_traj
from ase.io import read
import os, sys

input_xyz = str(sys.argv[1])
output_rkf = str(sys.argv[2])
timestep = 5.
print(read(input_xyz))
traj_to_rkf(input_xyz, output_rkf, timestep=timestep)
