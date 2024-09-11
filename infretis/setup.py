"""Setup all that is needed for the infretis simulation."""

import logging
import os

import tomli

from infretis.asyncrunner import aiorunner, future_list
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


def setup_internal(config: dict) -> tuple[dict, REPEX_state]:
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
) -> dict | None:
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

    # quantis or any other method requiring different engines in each ensemble
    multi_engine = config["simulation"].get("multi_engine", False)
    quantis = config["simulation"]["tis_set"].get("quantis", False)
    # set the keywords once
    config["simulation"]["tis_set"]["quantis"] = quantis
    config["simulation"]["multi_engine"] = multi_engine

    check_config(config)

    return config


def check_config(config: dict) -> None:
    """Perform some checks on the settings from the .toml file. Raises
    TOMLConfigError if something is wrong.

    Args
        config: the configuration dictionary
    """
    intf = config["simulation"]["interfaces"]
    n_ens = len(config["simulation"]["interfaces"])
    n_workers = config["runner"]["workers"]
    sh_moves = config["simulation"]["shooting_moves"]
    n_sh_moves = len(sh_moves)
    intf_cap = config["simulation"]["tis_set"]["interface_cap"]

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

    if intf_cap > intf[-1]:
        raise TOMLConfigError(
            f"Interface_cap {intf_cap} > interface[-1]={intf[-1]}"
        )
    if intf_cap < intf[0]:
        raise TOMLConfigError(
            f"Interface_cap {intf_cap} < interface[-2]={intf[-2]}"
        )

    if config["simulation"]["tis_set"]["quantis"]:
        if not config.get("engine0"):
            raise TOMLConfigError(
                "Quantis needs a [0-] engine definition in [engine0] section!"
            )
        if not config.get("engine1"):
            raise TOMLConfigError(
                "Quantis needs an [N+] engine definition in [engine1] section!"
            )
        if not config["simulation"]["multi_engine"]:
            raise TOMLConfigError(
                "Need 'multi_engine=true' with 'quantis=true'"
            )

    if config["simulation"]["multi_engine"]:
        for key1 in config.keys():
            if "engine" in key1 and config[key1]["class"] == "gromacs":
                eng1 = config[key1].copy()
                inp_path1 = eng1.pop("input_path")
                for key2 in config.keys():
                    if "engine" in key2:
                        eng2 = config[key2].copy()
                        inp_path2 = eng2.pop("input_path")
                        if eng1 != eng2 and inp_path1 == inp_path2:
                            raise TOMLConfigError(
                                "Found differing engine settings with identic"
                                + "al 'input_path'. This would overwrite the"
                                + "settings of one of the engines in"
                                + " 'infretis.mdp'!"
                            )


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
