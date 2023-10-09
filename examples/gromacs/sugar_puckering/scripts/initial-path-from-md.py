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
parser.add_argument(
    "-n",
    help="The maximum number of shooting points for each ensemble \
        (default is 50)",
    default="50",
)

args = parser.parse_args()

predir = "load"

traj = args.trr  # trajectory  file
order = np.loadtxt(args.order)  # order file

# read interfaces from .toml file
with open(args.toml, "rb") as toml_file:
    toml_dict = tomli.load(toml_file)
interfaces = toml_dict["simulation"]["interfaces"]

n_sht_pts = int(args.n)
sorted_idx = np.argsort(order[:, 1])

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
        # first point crossing i+
        start = np.where(order[sorted_idx, 1] < interfaces[0])[0][-1] + 1
        # second point crossing i+
        end = np.where(order[sorted_idx, 1] < interfaces[0])[0][-1] + 2
        middle = sorted_idx[: end - 1]
        # trajectory indices giving the path
        iterator = np.hstack(
            (
                sorted_idx[start],
                middle[:: max(len(middle) // n_sht_pts, 1)],
                sorted_idx[end],
            )
        )

    # plus ensembles
    else:
        # starting point of the path
        start = np.where(order[sorted_idx, 1] > interfaces[0])[0][0] - 1
        # ending point of the path
        end = np.where(order[sorted_idx, 1] > interfaces[0])[0][0] - 2
        # points above interface i
        over_i = np.where(order[sorted_idx, 1] > interfaces[i - 1])[0]
        middle = sorted_idx[over_i[0] : over_i[-1]]
        iterator = np.hstack(
            (
                sorted_idx[start],
                middle[:: max(len(middle) // n_sht_pts, 1)],
                sorted_idx[end],
            )
        )

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
    print(iterator)

    with open("frames.ndx", "w") as index_file:
        index_file.write("[ frames ]\n")
        for idxi in iterator:
            index_file.write(f"{idxi+1}\n")

    cmd = f"gmx trjconv -f {traj} -o {trajfile} -fr frames.ndx"
    print(cmd)
    subprocess.run(cmd, shell=True)

    # write order file
    N = len(iterator)
    np.savetxt(
        orderfile,
        np.c_[order[:N, 0], order[iterator, 1]],
        header=f"{'time':>10} {'orderparam':>15}",
        fmt=["%10.d", "%15.4f"],
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
