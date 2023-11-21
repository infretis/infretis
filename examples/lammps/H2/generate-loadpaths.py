import os
import shutil

import numpy as np

from infretis.classes.engines.factory import create_engines
from infretis.classes.orderparameter import create_orderparameters
from infretis.classes.path import Path, paste_paths
from infretis.classes.repex import REPEX_state
from infretis.classes.system import System
from infretis.setup import setup_config

# infrtis parameters
config = setup_config("infretis.toml")
state = REPEX_state(config, minus=True)

# setup ensembles
state.initiate_ensembles()
state.engines = create_engines(config)
create_orderparameters(state.engines, config)

# initial configuration to start from
system0 = System()
engine = state.engines["lmp"]
engine.exe_dir = engine.exe_path
system0.set_pos((os.path.join(engine.input_path, "conf.lammpstrj"), 0))
# empty path we will fill forwards in time
path0 = Path(maxlen=1000)
path1 = Path(maxlen=1000)
status0, message0 = engine.propagate(path0, state.ensembles[0], system0)
status1, message1 = engine.propagate(path1, state.ensembles[1], system0)

# we did not integrate in ensemble0 because
# we started above interface0
if path0.length == 1:
    path0.phasepoints[0].set_pos(path1.phasepoints[-1].config)
    status0, message0 = engine.propagate(path0, state.ensembles[0], system0)

# or we did not integrate in ensemble1 because
# we started below interface0
elif path1.length == 1:
    path1.phasepoints[0].set_pos(path0.phasepoints[-1].config)
    status1, message1 = engine.propagate(path1, state.ensembles[1], system0)

elif path1.length != 1 and path0.length != 1:
    print("Something fishy")

else:
    raise IndexError("No MD propagation was performed!")

# backward paths
path0r = Path(maxlen=1000)
path1r = Path(maxlen=1000)

status0, message0 = engine.propagate(
    path0r, state.ensembles[0], path0.phasepoints[0], reverse=True
)
status1, message1 = engine.propagate(
    path1r, state.ensembles[1], path1.phasepoints[0], reverse=True
)

dirname = "load"
pathsf = [path0, path1]
pathsr = [path0r, path1r]
for i in range(2):
    dirname = "load"
    dirname = os.path.join(dirname, str(i))
    accepted = os.path.join(dirname, "accepted")
    orderfile = os.path.join(dirname, "order.txt")
    trajtxtfile = os.path.join(dirname, "traj.txt")
    print(f"Making folder: {dirname}")
    os.makedirs(dirname)
    print(f"Making folder: {accepted}")
    os.makedirs(accepted)
    # combine forward and backward path
    path = paste_paths(pathsr[i], pathsf[i])
    # save order paramter
    order = [pp.order[0] for pp in path.phasepoints]
    order = np.vstack((np.arange(len(order)), np.array(order))).T
    np.savetxt(orderfile, order, fmt=["%d", "%12.6f"])
    N = len(order)
    # save traj.txt
    np.savetxt(
        trajtxtfile,
        np.c_[
            [str(i) for i in range(N)],
            [pp.config[0].split("/")[-1] for pp in path.phasepoints],
            [pp.config[1] for pp in path.phasepoints],
            [1 if pp.vel_rev is True else -1 for pp in path.phasepoints],
        ],
        header=f"{'time':>10} {'trajfile':>15} {'index':>10} {'vel':>5}",
        fmt=["%10s", "%15s", "%10s", "%5s"],
    )
    # copy paths
    for trajfile in np.unique(
        [pp.config[0].split("/")[-1] for pp in path.phasepoints]
    ):
        print(trajfile)
        shutil.copy(trajfile, accepted)
