import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tomli

# Command line argument parser stuff
parser = argparse.ArgumentParser(
    description="Plot the order parameter of all paths from an \
                infretis simulation."
)

parser.add_argument(
    "-traj",
    help="The path to the folder containing the trajectories\
            (e.g. 'run1/load/')",
)
parser.add_argument(
    "-toml",
    help="The .toml input file for reading the interfaces\
            (e.g. 'infretis.toml')",
)

parser.add_argument(
    "-xy",
    help="The indices of the columns to plot (default 0 1)",
    default=[0, 1],
    metavar=("x", "y"),
    type=int,
    nargs=2,
)

args = parser.parse_args()

# read interfaces from the .toml file
with open(args.toml, "rb") as toml_file:
    toml_dict = tomli.load(toml_file)
interfaces = toml_dict["simulation"]["interfaces"]

# get the filenames of all created paths
paths = glob.glob(f"{args.traj}/*/order.txt")

# sort filenames by time of path creation
sorted_paths = sorted(paths, key=os.path.getctime)

# plotting stuff, modify by your needs
f, a = plt.subplots()

# add horisontal lines for interfaces
for interface in interfaces:
    a.axhline(interface, c="k", lw=0.5)

if 2 in args.xy:
    lw = 0

else:
    lw = 1

# plot all paths, modify by your needs
for path in sorted_paths:
    x = np.loadtxt(path)
    if x[-1, 1] > interfaces[-1]:
        print()
        print(
            f"The path in {path} is reactive with \
phi={x[-1,2]:.2f}! \U0001F389 \U0001F938 \U0001F483"
        )
    #    continue # continues to next iteration in loop
    a.plot(
        x[:, args.xy[0]],
        x[:, args.xy[1]],
        c="C0",
        marker="o",
        markersize=2.5,
        lw=lw,
    )

plt.show()
