import argparse
import os
import subprocess

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.align import alignto
from MDAnalysis.lib.mdamath import make_whole

parser = argparse.ArgumentParser(
    description="Reverse and concatenate trajectories from RETIS simulations."
)

parser.add_argument(
    "-out", help="the outfile trajectory name (e.g. md-traj.xyz)"
)
parser.add_argument(
    "-tpr", help="the .tpr file (e.g. ../gromacs_input/topol.tpr)"
)
parser.add_argument(
    "-path", help="the filepath to the path_nr folder (e.g. run3/46/)"
)
# parser.add_argument(
#     "--selection",
#     help="The selection, e.g. 'index 1 2 3 4' or 'resname MOL0' \
#        (default 'resname MOL0')",
#     default="resname MOL0",
# )

args = parser.parse_args()
args.selection = "resname MOL0"


traj_file_arr, index_arr = np.loadtxt(
    f"{args.path}/traj.txt",
    usecols=[1, 2],
    comments="#",
    dtype=str,
    unpack=True,
)
traj_file_arr = [f"{args.path}/accepted/{traj_i}" for traj_i in traj_file_arr]
# traj_file_arr = np.char.replace(traj_file_arr,"trr","xtc")
index_arr = index_arr.astype(int)

U = {}
# reference for rmsd alignment
ref = mda.Universe(args.tpr, traj_file_arr[0]).select_atoms(args.selection)
make_whole(ref.atoms)
for traj_file in np.unique(traj_file_arr):
    print(f"Reading {traj_file} ...")
    subprocess.run(
        f"printf '1\n0\n' | gmx trjconv \
        -f {traj_file} -o tmp.{traj_file.split('/')[-1]} \
        -pbc whole -center -s {args.tpr}",
        shell=True,
    )
    if not os.path.exists(traj_file):
        exit(f"Could not find file {traj_file}.?")

    U[traj_file] = mda.Universe(args.tpr, f"tmp.{traj_file.split('/')[-1]}")

with mda.Writer(
    args.out, U[traj_file].select_atoms(args.selection).n_atoms
) as wfile:
    for traj_file, index in zip(traj_file_arr, index_arr):
        u = U[traj_file]
        ag = u.select_atoms(args.selection)
        make_whole(ag)
        u.trajectory[index]
        alignto(ag, ref)
        wfile.write(ag.atoms)

for traj_file in np.unique(traj_file_arr):
    subprocess.run(f"rm tmp.{traj_file.split('/')[-1]}", shell=True)

subprocess.run(f"perl -pi -e 'chomp if eof' {args.out}", shell=True)

print("\nAll done!")
print(f"Trajectory written to {args.out}.")
