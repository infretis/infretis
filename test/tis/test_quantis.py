import os
from pathlib import PosixPath

import numpy as np

from infretis.classes.engines.factory import create_engines
from infretis.classes.orderparameter import create_orderparameters
from infretis.classes.path import Path, paste_paths
from infretis.classes.repex import REPEX_state
from infretis.classes.system import System
from infretis.core.tis import quantis_swap_zero
from infretis.setup import setup_config

HERE = PosixPath(__file__).resolve().parent


def test_quantis_swap_zero(tmp_path: PosixPath):
    tmp_dir = (tmp_path).resolve()
    tmp_dir.mkdir(exist_ok=True)

    os.chdir(tmp_dir)

    example_folder = (HERE / "../../examples/turtlemd/H2/").resolve()
    initial_configuration = example_folder / "conf.xyz"
    toml = str((example_folder / "infretis.toml").resolve())
    print(toml)

    # maximal length of initial paths
    maxlen = 100

    # infretis parameters
    config = setup_config(toml)

    print(config)
    state = REPEX_state(config, minus=True)

    # setup ensembles
    state.initiate_ensembles()
    state.engines = create_engines(config)
    state.engines["turtlemd"].rgen = np.random.default_rng()
    state.engines["turtlemd"].beta = 1
    create_orderparameters(state.engines, config)

    # initial configuration to start from
    system0 = System()
    engine = state.engines[config["engine"]["engine"]]
    engine.exe_dir = str(tmp_dir.resolve())
    engine.exe_path = str(tmp_dir.resolve())
    print(engine.exe_dir)
    engine.set_mdrun(
        {"wmdrun": config["dask"]["wmdrun"][0], "exe_dir": engine.exe_dir}
    )
    system0.set_pos((initial_configuration, 0))

    # empty paths we will fill forwards in time in [0-] and [0+]
    path0 = Path(maxlen=maxlen)
    path1 = Path(maxlen=maxlen)

    # propagate forwards from the intiial configuration
    # note that one of these does not get integrated because
    # the initial phasepoint is either below or above interface 0
    print("Propagating in ensemble [0-]")
    status0, message0 = engine.propagate(path0, state.ensembles[0], system0)
    system0.set_pos((initial_configuration, 0))
    print(system0.config, system0.order)
    print("Propagating in ensemble [0+]")
    status1, message1 = engine.propagate(path1, state.ensembles[1], system0)

    # we did only one integration step in ensemble 0 because
    # we started above interface 0
    if path0.length == 1:
        print("Re-propagating [0-] since we started above lambda0")
        system0.set_pos((engine.dump_config(path1.phasepoints[-1].config), 0))
        print(system0.config, system0.order)
        path0 = Path(maxlen=maxlen)
        status0, message0 = engine.propagate(
            path0, state.ensembles[0], system0
        )

    # or we did only one integration step in ensemble 1 because
    # we started below interface 0
    elif path1.length == 1:
        print("Re-propagating [0+] since we started below lambda0")
        print(system0.config, system0.order)
        system0.set_pos((engine.dump_config(path0.phasepoints[-1].config), 0))
        path1 = Path(maxlen=maxlen)
        status1, message1 = engine.propagate(
            path1, state.ensembles[1], system0
        )

    else:
        raise ValueError(
            "Something fishy!\
                Path lengths in one of the ensembles != 1"
        )

    # backward paths
    path0r = Path(maxlen=maxlen)
    path1r = Path(maxlen=maxlen)

    print("Propagating [0-] in reverse")
    status0, message0 = engine.propagate(
        path0r, state.ensembles[0], path0.phasepoints[0], reverse=True
    )

    print("Propagating [0+] in reverse")
    status1, message1 = engine.propagate(
        path1r, state.ensembles[1], path1.phasepoints[0], reverse=True
    )

    path0 = paste_paths(path0r, path0)
    path1 = paste_paths(path1r, path1)

    picked = {
        -1: {
            "engine": state.engines["turtlemd"],
            "ens": state.ensembles[0],
            "traj": path0,
        },
        0: {
            "engine": state.engines["turtlemd"],
            "ens": state.ensembles[1],
            "traj": path1,
        },
    }

    p = quantis_swap_zero(picked)
    assert p[0]
