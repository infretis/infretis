"""Setup all that is needed for the infretis simulation."""
import logging
import os

import tomli
from dask.distributed import Client, as_completed, dask, get_worker

from infretis.classes.engines.factory import create_engines
# from infretis.classes.ensemble import create_ensembles
from infretis.classes.formats.formatter import get_log_formatter
from infretis.classes.orderparameter import create_orderparameters
from infretis.classes.path import load_paths_from_disk
from infretis.classes.repex import REPEX_state

logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)


def setup_internal(config):
    """Run the various setup functions."""
    # setup logger
    setup_logger()

    # setup repex and its ensembles
    state = REPEX_state(config, minus=True)

    # # setup ensembles
    state.initiate_ensembles()

    # setup engines
    state.engines = create_engines(config)

    # setup engine orderparameter functions
    create_orderparameters(state.engines, config)

    # load paths from disk and add to repex
    paths = load_paths_from_disk(config)
    state.load_paths(paths)

    # create first md_items dict
    md_items = {
        "mc_moves": state.mc_moves,
        "interfaces": state.interfaces,
        "cap": state.cap,
    }

    # write pattern header
    if state.pattern:
        state.pattern_header()

    return md_items, state


def setup_dask(state):
    """Setup dask classes."""
    # isolate each worker
    dask.config.set({"distributed.scheduler.work-stealing": False})

    # setup client with state.workers workers
    client = Client(n_workers=state.workers)

    # in case external engine or o_parameter scripts are used
    for module in state.config["dask"].get("files", []):
        client.upload_file(module)

    # create future
    futures = as_completed(None, with_results=True)

    # setup individual worker logs
    client.run(set_worker_logger)

    return client, futures


def setup_config(inp="infretis.toml", re_inp="restart.toml"):
    """Setup dict from *toml file."""
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


def write_header(config):
    """Write infretis_data.txt header."""
    size = config["current"]["size"]
    data_dir = config["output"]["data_dir"]
    data_file = os.path.join(data_dir, "infretis_data.txt")
    config["output"]["data_file"] = data_file
    with open(data_file, "w", encoding="utf-8") as write:
        write.write("# " + "=" * (34 + 8 * size) + "\n")
        ens_str = "\t".join([f"{i:03.0f}" for i in range(size)])
        write.write("# " + f"\txxx\tlen\tmax OP\t\t{ens_str}\n")
        write.write("# " + "=" * (34 + 8 * size) + "\n")


def setup_logger(inp="sim.log"):
    """Set main logger."""
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


def set_worker_logger():
    """Set logger for each worker."""
    # for each worker
    pin = get_worker().name
    logging.getLogger()
    fileh = logging.FileHandler(f"worker{pin}.log", mode="a")
    log_levl = getattr(logging, "info".upper(), logging.INFO)
    fileh.setLevel(log_levl)
    fileh.setFormatter(get_log_formatter(log_levl))
    logger.addHandler(fileh)
    logger.info("=============================")
    logger.info("Logging file for worker %s", pin)
    logger.info("=============================\n")
