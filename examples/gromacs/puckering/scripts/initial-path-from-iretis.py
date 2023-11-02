import argparse
import glob
import os
import shutil

import numpy as np
import tomli

parser = argparse.ArgumentParser(
    description="Generate initial paths for an infretis \
                simulation using paths from an earlier infretis \
                simulation"
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

out = {}  # ensemble - traj_idx

trajs = glob.glob(f"{args.traj}/*")  # folder to trajectories
idx = np.argsort([int(f.split("/")[-1]) for f in trajs])
trajs = sorted(trajs, key=os.path.getctime)

# iterate backwards to get decorrelated paths
for traj in trajs[::-1]:
    x = np.loadtxt(f"{traj}/order.txt", usecols=[0, 1])
    # zero minus
    if x[0, 1] > interfaces[0]:
        if 0 not in out.keys():
            out[0] = traj

    # 0+ intf
    else:
        omax = np.max(x[:, 1])
        valid_in = False
        for i, interface in enumerate(interfaces):
            if omax > interface:
                valid_in = i + 1
        if valid_in and valid_in not in out.keys():
            out[valid_in] = traj

# if we miss some lower ensembles, add to
# them the paths from the higher ensembles
for i in range(len(interfaces) + 1):
    if i not in out.keys():
        for j in range(i + 1, len(interfaces) + 1):
            if j in out.keys():
                out[i] = out[j]
                print(f"[INFO] Added path from ens{j} to ens{i}")


# Check if we have paths in all ensembles
for i in range(len(interfaces) + 1):
    assert (
        i in out.keys()
    ), f"Did not find any paths in ensemble {i}\
that cross the corresponding interface"

loaddir = "load"
if os.path.exists(loaddir):
    exit(
        f"\nDirectory {loaddir}/ exists. Will not overwrite.\
\nRename or delete it manually. Aborting."
    )
else:
    os.mkdir(loaddir)

for i, traj in zip(out.keys(), out.values()):
    shutil.copytree(traj, f"{loaddir}/{i}")

print("\nAll done! Created folder load/ with new initial paths.")
