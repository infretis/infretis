import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tomli

parser = argparse.ArgumentParser(
    description="Plot the order parameter and interfaces from an \
                infretis simulation"
)

parser.add_argument(
    "-traj",
    help="The path to the folder containing the trajectories\
            (e.g. ../iretis0/trajs/)",
)
parser.add_argument(
    "-toml",
    help="The .toml input file for reading the interfaces\
            (e.g. ../iretis0/infretis.toml)",
)
args = parser.parse_args()

# read interfaces from .toml file
with open(args.toml, "rb") as toml_file:
    toml_dict = tomli.load(toml_file)
interfaces = toml_dict["simulation"]["interfaces"][:-1]

trajs = glob.glob(f"{args.traj}/*/order.txt")
sorted_trajs = sorted(trajs, key=os.path.getctime)  # sort by time

# plotting stuff, modify by needs
f, a = plt.subplots()

for interface in interfaces:
    a.axhline(interface, c="k", lw=0.5)

for traj in sorted_trajs:
    x = np.loadtxt(traj)
    a.plot(x[:, 0], x[:, 1], c="C0", alpha=0.25, lw=1)

plt.show()
