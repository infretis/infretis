"""Setup all that is needed for the infretis simulation."""

import logging
import os
from typing import Optional, Tuple

import tomli

from infretis.asyncrunner import aiorunner, future_list
from infretis.classes.engines.factory import create_engines
from infretis.classes.formatter import get_log_formatter
from infretis.classes.path import load_paths_from_disk
from infretis.classes.repex import REPEX_state
from infretis.core.tis import run_md

logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)


class TOMLConfigError(Exception):
    """Raised when there is an error in the .toml configuration."""

    pass
    # def __init__(self, message):
    #    super().__init__(message)


def setup_internal(config: dict) -> Tuple[dict, REPEX_state]:
    """Run the various setup functions.

    Args
        config: the configuration dictionary

    Returns:
        A blank md_items dict
        An initialized REPEX state
    """
    # setup logger
    setup_logger()

    # setup repex
    state = REPEX_state(config, minus=True)

    # setup ensembles
    state.initiate_ensembles()

    # load paths from disk and add to repex
    paths = load_paths_from_disk(config)
    state.load_paths(paths)

    # create first md_items dict
    md_items = {
        "mc_moves": state.mc_moves,
        "interfaces": state.interfaces,
        "cap": state.cap,
    }

    # setup the engine_occupation list
    _, engine_occ = create_engines(config)
    state.engine_occ = engine_occ

    return md_items, state


def setup_runner(state: REPEX_state) -> Tuple[aiorunner, future_list]:
    """Set the task runner class up.

    Args:
        state: A REPEX state from which to get the config dict
    """
    # setup client with state.workers workers
    runner = aiorunner(state.config, state.config["runner"]["workers"])

    # Attach the run_md task and start the runner's workers
    runner.set_task(run_md)
    runner.start()

    # A managed list of futures
    futures = future_list()

    return runner, futures


def setup_config(
    inp: str = "infretis.toml", re_inp: str = "restart.toml"
) -> Optional[dict]:
    """Set dict from *toml file up.

    Arg
        inp: a string specifying the input file (def: infretis.toml)
        re_inp: a string specifying the restart file (def: restart.toml)

    Return
        A dictionary containing the configuration parameters or None
    """
    # sets up the dict from *toml file.

    # load input:
    if os.path.isfile(inp):
        with open(inp, mode="rb") as read:
            config = tomli.load(read)
    else:
        logger.info("%s file not found, exit.", inp)
        return None

    # check if restart.toml exist and if its runnable:
    if inp != re_inp and os.path.isfile(re_inp):
        msg = f"Restart file '{re_inp}' found, but its not the run file!"
        raise ValueError(msg)

    # in case we restart, toml file has a 'current' subdict.
    if "current" in config:
        curr = config["current"]

        # if cstep and steps are equal, we stop here.
        if curr.get("cstep") == curr.get("restarted_from", -1):
            return None

        # set 'restarted_from'
        curr["restarted_from"] = config["current"]["cstep"]

        # check active paths:
        load_dir = config["simulation"].get("load_dir", "trajs")
        for act in config["current"]["active"]:
            store_p = os.path.join(load_dir, str(act), "traj.txt")
            if not os.path.isfile(store_p):
                return None
    else:
        # no 'current' in toml, start from step 0.
        size = len(config["simulation"]["interfaces"])
        config["current"] = {
            "traj_num": size,
            "cstep": 0,
            "active": list(range(size)),
            "locked": [],
            "size": size,
            "frac": {},
            "wsubcycles": [0 for _ in range(config["runner"]["workers"])],
            "tsubcycles": 0,
        }

        # write/overwrite infretis_data.txt
        write_header(config)

    # quantis or any other method requiring different engines in each ensemble
    has_ens_engs = config["simulation"].get("ensemble_engines", False)
    if not has_ens_engs:
        ens_engs = []
        for itnf in config["simulation"]["interfaces"]:
            ens_engs.append(["engine"])
        config["simulation"]["ensemble_engines"] = ens_engs

    # set all keywords only once, so they appear in restart.toml
    # and we can avoid the .get() in other parts
    if "seed" not in config["simulation"].keys():
        config["simulation"]["seed"] = 0

    # [output]
    keep_maxop_trajs = config["output"].get("keep_maxop_trajs", False)
    config["output"]["keep_maxop_trajs"] = keep_maxop_trajs
    delete_old = config["output"].get("delete_old", False)
    delete_old_all = config["output"].get("delete_old_all", False)
    if not delete_old and keep_maxop_trajs:
        raise TOMLConfigError("keep_maxop_trajs=True requires delete_old=True")
    if delete_old_all and keep_maxop_trajs:
        msg = (
            "delete_old_all=True will delete all trajectories. Set "
            "keep_maxop_trajs to False in the [output] section"
        )
        raise TOMLConfigError(msg)

    quantis = config["simulation"]["tis_set"].get("quantis", False)
    config["simulation"]["tis_set"]["quantis"] = quantis

    l_1 = config["simulation"]["tis_set"].get("lambda_minus_one", False)
    config["simulation"]["tis_set"]["lambda_minus_one"] = l_1

    if quantis and not has_ens_engs:
        config["simulation"]["ensemble_engines"][0] = ["engine0"]
    accept_all = config["simulation"]["tis_set"].get("accept_all", False)
    config["simulation"]["tis_set"]["accept_all"] = accept_all

    check_config(config)

    return config


def check_config(config: dict) -> None:
    """Perform some checks on the settings from the .toml file.

    Args
        config: the configuration dictionary
    """
    intf = config["simulation"]["interfaces"]
    n_ens = len(config["simulation"]["interfaces"])
    n_workers = config["runner"]["workers"]
    sh_moves = config["simulation"]["shooting_moves"]
    n_sh_moves = len(sh_moves)
    intf_cap = config["simulation"]["tis_set"].get("interface_cap", False)
    lambda_minus_one = config["simulation"]["tis_set"].get(
        "lambda_minus_one", False
    )

    if lambda_minus_one is not False and lambda_minus_one >= intf[0]:
        raise TOMLConfigError(
            "lambda_minus_one interface must be less than the first interface!"
        )

    if n_ens < 2:
        raise TOMLConfigError("Define at least 2 interfaces!")

    if n_workers > n_ens - 1:
        raise TOMLConfigError("Too many workers defined!")

    if sorted(intf) != intf:
        raise TOMLConfigError("Your interfaces are not sorted!")

    if len(set(intf)) != len(intf):
        raise TOMLConfigError("Your interfaces contain duplicate values!")

    if n_ens > n_sh_moves:
        raise TOMLConfigError(
            f"N_interfaces {n_ens} > N_shooting_moves {n_sh_moves}!"
        )

    if intf_cap and intf_cap > intf[-1]:
        raise TOMLConfigError(
            f"Interface_cap {intf_cap} > interface[-1]={intf[-1]}"
        )
    if intf_cap and intf_cap < intf[0]:
        raise TOMLConfigError(
            f"Interface_cap {intf_cap} < interface[-2]={intf[-2]}"
        )

    # engine checks
    unique_engines = []
    for engines in config["simulation"]["ensemble_engines"]:
        for engine in engines:
            if engine not in unique_engines:
                unique_engines.append(engine)

    for key1 in unique_engines:
        if key1 not in config.keys():
            raise TOMLConfigError(f"Engine '{key1}' not defined!")

    # gromacs check
    for key1 in unique_engines:
        if config[key1]["class"] == "gromacs":
            eng1 = config[key1].copy()
            inp_path1 = eng1.pop("input_path")
            for key2 in unique_engines:
                eng2 = config[key2].copy()
                inp_path2 = eng2.pop("input_path")
                if eng1 != eng2 and inp_path1 == inp_path2:
                    raise TOMLConfigError(
                        "Found differing engine settings with identic"
                        + "al 'input_path'. This would overwrite the"
                        + " settings of one of the engines in"
                        + " 'infretis.mdp'!"
                    )

    # check wsubcycles and tsubcycles in case restarting from old version
    if "wsubcycles" not in config["current"]:
        list_of_zeros = [0 for _ in range(config["runner"]["workers"])]
        config["current"]["wsubcycles"] = list_of_zeros
    if "tsubcycles" not in config["current"]:
        config["current"]["tsubcycles"] = 0
    # if increased number of workers
    wsub_num = len(config["current"]["wsubcycles"])
    if wsub_num < config["runner"]["workers"]:
        extra = config["runner"]["workers"] - wsub_num
        config["current"]["wsubcycles"] += [0] * extra


def write_header(config: dict) -> None:
    """Write infretis_data.txt header.

    Args
        config: the configuration dictionary
    """
    size = config["current"]["size"]
    data_dir = config["output"]["data_dir"]
    data_file = os.path.join(data_dir, "infretis_data.txt")
    if os.path.isfile(data_file):
        for i in range(1, 1000):
            data_file = os.path.join(data_dir, f"infretis_data_{i}.txt")
            if not os.path.isfile(data_file):
                break

    config["output"]["data_file"] = data_file
    with open(data_file, "w", encoding="utf-8") as write:
        write.write("# " + "=" * (34 + 8 * size) + "\n")
        ens_str = "\t".join([f"{i:03.0f}" for i in range(size)])
        write.write("# " + f"\txxx\tlen\tmax OP\t\t{ens_str}\n")
        write.write("# " + "=" * (34 + 8 * size) + "\n")


def setup_logger(inp: str = "sim.log") -> None:
    """Set main logger.

    Args
        inp: a string specifying the main log file
    """
    # Define a console logger. This will log to sys.stderr:
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(get_log_formatter(logging.WARNING))
    logger.addHandler(console)
    fileh = logging.FileHandler(inp, mode="a")
    log_levl = getattr(logging, "info".upper(), logging.INFO)
    fileh.setLevel(log_levl)
    fileh.setFormatter(get_log_formatter(log_levl))
    logger.addHandler(fileh)
