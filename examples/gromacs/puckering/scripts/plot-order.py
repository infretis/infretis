import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tomli

# Command line argument parser stuff
parser = argparse.ArgumentParser(
    description="Plot the order parameter and interfaces from an \
                infretis simulation"
)

parser.add_argument(
    "-traj",
    help="The path to the folder containing the trajectories\
            (e.g. '../iretis0/load/')",
)
parser.add_argument(
    "-toml",
    help="The .toml input file for reading the interfaces\
            (e.g. ../iretis0/infretis.toml)",
)
args = parser.parse_args()

# read interfaces from the .toml file
with open(args.toml, "rb") as toml_file:
    toml_dict = tomli.load(toml_file)
interfaces = toml_dict["simulation"]["interfaces"][:-1]

# get the filenames of all created paths
paths = glob.glob(f"{args.traj}/*/order.txt")

# sort filenames by time of path creation
sorted_paths = sorted(paths, key=os.path.getctime)

# plotting stuff, modify by your needs
f, a = plt.subplots()

# add horisontal lines for interfaces
for interface in interfaces:
    a.axhline(interface, c="k", lw=0.5)

# plot all paths, modify by your needs
for path in sorted_paths:
    x = np.loadtxt(path)
    # if x[-1,1] > interfaces[-1]:
    # ...
    a.plot(x[:, 0], x[:, 1], c="C0", marker="o", markersize=5)

plt.show()
