"""Setup all that is needed for the infretis simulation."""
import logging
import os
from typing import Dict

import tomli

from infretis.asyncrunner import aiorunner, future_list
from infretis.classes.formatter import get_log_formatter
from infretis.classes.path import load_paths_from_disk
from infretis.classes.repex import REPEX_state
from infretis.core.tis import run_md

logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)


def setup_internal(config: Dict) -> tuple[Dict, REPEX_state]:
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
        "config": config,
    }

    # write pattern header
    if state.pattern:
        state.pattern_header()

    return md_items, state


def setup_runner(state: REPEX_state) -> tuple[aiorunner, future_list]:
    """Setup task runner classes.

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
) -> Dict | None:
    """Set up dict from *toml file.

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

    # check if restart.toml exist:
    if inp != re_inp and os.path.isfile(re_inp):
        # load restart input:
        with open(re_inp, mode="rb") as read:
            re_config = tomli.load(read)

        # check if sim settings for the two are equal:
        equal = True
        for key in config.keys():
            if config[key] != re_config.get(key, {}):
                equal = False
                logger.info("We use {re_inp} instead.")
                break
        config = re_config if equal else config

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
        }

        # write/overwrite infretis_data.txt
        write_header(config)

        # set pattern
        if config["output"].get("pattern", False):
            config["output"]["pattern_file"] = os.path.join("pattern.txt")

    return config


def write_header(config: Dict) -> None:
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
