import argparse
from types import SimpleNamespace

import numpy as np
import tomli

from infretis.classes.engines.gromacs import read_trr_file
from infretis.classes.orderparameter import create_orderparameter

parser = argparse.ArgumentParser(
    description="Recalculate the orderparameter from a .trr file"
)

parser.add_argument("-trr", help="The .trr trajectory file")
parser.add_argument(
    "-toml", help="The .toml input file defining the orderparameter"
)
parser.add_argument(
    "-out",
    help="The output file. Default: order-rec.txt",
    default="order-rec.txt",
)

args = parser.parse_args()


traj = read_trr_file(args.trr)
with open(args.toml, "rb") as toml_file:
    toml_dict = tomli.load(toml_file)

orderparameter = create_orderparameter(toml_dict)
# interfaces = toml_dict["simulation"]["interfaces"]

with open(args.out, "w") as writefile:
    writefile.write("# step\ttheta\tphi\tQ\n")
    for i, frame in enumerate(traj):
        system = SimpleNamespace(
            pos=frame[1]["x"],
            box=np.diag(frame[1]["box"]),
        )
        op = orderparameter.calculate(system)
        line = f"{i} " + " ".join([f"{opi}" for opi in op]) + "\n"
        writefile.write(line)

print(f"\nAll done!\nOrderparameter values written to {args.out}")
