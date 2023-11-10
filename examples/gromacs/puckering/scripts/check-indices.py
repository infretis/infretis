import argparse
import subprocess
from types import SimpleNamespace

import numpy as np

from infretis.classes.orderparameter import Puckering

parser = argparse.ArgumentParser(
    description="Calculate the theta and phi angle for an \
            .sdf file given a set of indices"
)

parser.add_argument(
    "-sdf", help="The .sdf file of your molecule (e.g. ../mol.sdf)"
)
parser.add_argument(
    "-idx",
    help="The ordered indices of your molecule (e.g. 2 5 11 8 1 0)",
    type=int,
    nargs="+",
)

args = parser.parse_args()

orderparameter = Puckering(index=[i for i in args.idx])

subprocess.run(f"obabel -isdf {args.sdf} -oxyz -O .temp.xyz", shell=True)
x = np.loadtxt(".temp.xyz", skiprows=2, usecols=[1, 2, 3])
subprocess.run("rm .temp.xyz", shell=True)

system = SimpleNamespace(
    pos=x,
    box=[np.inf, np.inf, np.inf],
)
op = orderparameter.calculate(system)

print(f"\nTheta = {op[0]:.3f} degrees")
print(f"Phi = {op[1]:.3f} degrees")
