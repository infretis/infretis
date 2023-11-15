import argparse
import os
import subprocess

import numpy as np
import tomli

parser = argparse.ArgumentParser(
    description="Generate initial paths for an infretis \
                simulation from an equilibrium run."
)

parser.add_argument("-trr", help="The .trr trajectory file")
parser.add_argument(
    "-order", help="The order file corresponding to the trajectory"
)
parser.add_argument("-toml", help="The .toml input for reading the interfaces")

args = parser.parse_args()

predir = "load"
if os.path.exists(predir):
    exit(
        f"\nDirectory {predir}/ exists."
        + " Will not overwrite."
        + " Rename or remove it and try again."
    )
else:
    os.mkdir(predir)

traj = args.trr  # trajectory  file
order = np.loadtxt(args.order)  # order file

# read interfaces from .toml file
with open(args.toml, "rb") as toml_file:
    toml_dict = tomli.load(toml_file)
interfaces = toml_dict["simulation"]["interfaces"]

for i in range(len(interfaces)):
    dirname = os.path.join(predir, str(i))
    accepted = os.path.join(dirname, "accepted")
    trajfile = os.path.join(accepted, "traj.trr")
    orderfile = os.path.join(dirname, "order.txt")
    trajtxtfile = os.path.join(dirname, "traj.txt")
    print(f"Making folder: {dirname}")
    os.makedirs(dirname)
    print(f"Making folder: {accepted}")
    os.makedirs(accepted)
    print(
        "Writing trajectory {} and order {} and trajfile {}".format(
            trajfile, orderfile, trajtxtfile
        )
    )

    # minus ensemble
    if i == 0:
        idx = (order[:, 1] > interfaces[0]).astype(int)
        grad = idx[1:] - idx[:-1]
        # hopping above interface0 grad = 1
        above = np.where(grad == 1)[0]
        # select an ending point where the path hops
        # above interface0
        end = above[-2] + 1  # we want the ending point
        # select a starting point where the path hops
        # below interface0, and has to precede the end point
        # only look at the array up til end point
        below = np.where(grad[:end] == -1)[0]
        start = below[-1]
        iterator = [i for i in range(start, end + 1)]
        print("=" * 10)
        print(iterator)
        print(order[iterator, 1])

    # plus ensembles
    else:
        idx = (order[:, 1] > interfaces[0]).astype(int)
        grad = idx[1:] - idx[:-1]
        # hopping above interface0 grad = 1
        above = np.where(grad == 1)[0]
        # select a starting point where the path hops
        # above interface0. Dont select last point
        # as we may not jump below again after that
        start = above[-2]
        # select an ending point where the path hops
        # below interface0
        # truncate where wee look for this point
        below = np.where(grad[: above[-1]] == -1)[0]
        end = below[-1] + 1  # only look at the array up til end point
        iterator = [i for i in range(start, end + 1)]
        print("=" * 10)
        print(iterator)
        print(order[iterator, 1])

        # check if valid path for wire-fencing
        idx = np.where(
            (order[iterator, 1] >= interfaces[i - 1])
            & (order[iterator, 1] <= interfaces[i])
        )[0]
        if len(idx) == 0 and i > 1:
            # no points between interface i-1 and i
            idx = np.where(
                (order[iterator, 1] >= interfaces[i - 1])
                & (order[iterator, 1] <= interfaces[i + 1])
            )[0]
            exit("Invalid path for wf!!")

    with open(".frames.ndx", "w") as index_file:
        index_file.write("[ frames ]\n")
        for idxi in iterator:
            index_file.write(f"{idxi+1}\n")

    cmd = f"gmx trjconv -f {traj} -o {trajfile} -fr .frames.ndx"
    print(cmd)
    subprocess.run(cmd, shell=True)

    # write order file
    N = len(iterator)
    np.savetxt(
        orderfile,
        np.c_[order[:N, 0], order[iterator, 1:]],
        header=f"{'time':>10} {'theta':>15} {'phi':>15} {'Qampl':>15}",
        fmt=["%10.d", "%15.4f", "%15.4f", "%15.4f"],
    )
    np.savetxt(
        trajtxtfile,
        np.c_[
            [str(i) for i in range(N)],
            ["traj.trr" for i in range(N)],
            [str(i) for i in range(N)],
            [str(1) for i in range(N)],
        ],
        header=f"{'time':>10} {'trajfile':>15} {'index':>10} {'vel':>5}",
        fmt=["%10s", "%15s", "%10s", "%5s"],
    )

print("\nAll done! Created folder load/ containing the initial paths.")
