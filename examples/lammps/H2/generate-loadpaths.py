import os
import shutil

import numpy as np

from infretis.classes.engines.factory import create_engines
from infretis.classes.orderparameter import create_orderparameters
from infretis.classes.path import Path, paste_paths
from infretis.classes.repex import REPEX_state
from infretis.classes.system import System
from infretis.setup import setup_config

# Generate paths in [0-] and [0+] from which we can run
# infretis and progressively increase/add interfaces
# we only need an initial configuration to start from
initial_configuration = "conf.lammpstrj"
# maximal length of initial paths
maxlen = 30

# infretis parameters
config = setup_config("infretis.toml")
state = REPEX_state(config, minus=True)

# setup ensembles
state.initiate_ensembles()
state.engines = create_engines(config)
create_orderparameters(state.engines, config)

# initial configuration to start from
system0 = System()
engine = state.engines[config["engine"]["engine"]]
engine.exe_dir = engine.exe_path
system0.set_pos((os.path.join(engine.input_path, initial_configuration), 0))

# empty paths we will fill forwards in time in [0-] and [0+]
path0 = Path(maxlen=maxlen)
path1 = Path(maxlen=maxlen)

# propagate forwards from the intiial configuration
# note that one of these does not get integrated because
# the initial phasepoint is either below or above interface 0
status0, message0 = engine.propagate(path0, state.ensembles[0], system0)
status1, message1 = engine.propagate(path1, state.ensembles[1], system0)

# we did only one integration step in ensemble 0 because
# we started above interface 0
if path0.length == 1:
    system0.set_pos((engine.dump_config(path1.phasepoints[-1].config), 0))
    path0 = Path(maxlen=maxlen)
    status0, message0 = engine.propagate(path0, state.ensembles[0], system0)

# or we did only one integration step in ensemble 1 because
# we started below interface 0
elif path1.length == 1:
    system0.set_pos((engine.dump_config(path0.phasepoints[-1].config), 0))
    path1 = Path(maxlen=maxlen)
    status1, message1 = engine.propagate(path1, state.ensembles[1], system0)

else:
    raise ValueError("Something fishy!")

# backward paths
path0r = Path(maxlen=maxlen)
path1r = Path(maxlen=maxlen)

status0, message0 = engine.propagate(
    path0r, state.ensembles[0], path0.phasepoints[0], reverse=True
)

status1, message1 = engine.propagate(
    path1r, state.ensembles[1], path1.phasepoints[0], reverse=True
)

# make load directories
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
            [-1 if pp.vel_rev else 1 for pp in path.phasepoints],
        ],
        header=f"{'time':>10} {'trajfile':>15} {'index':>10} {'vel':>5}",
        fmt=["%10s", "%15s", "%10s", "%5s"],
    )
    # copy paths
    for trajfile in np.unique(
        [pp.config[0].split("/")[-1] for pp in path.phasepoints]
    ):
        shutil.copy(trajfile, accepted)
