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

    # check if restart.toml exist
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

    expand_ensemble_move_policy(config)
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


# --------------------------------------------------------------------------- #
# Ergonomic ensemble move policy                                              #
# --------------------------------------------------------------------------- #
#
# Optional config layer that compiles into the canonical low-level fields
# `simulation.shooting_moves`, `simulation.tis_set.mwf_subcycle_small`, and
# `simulation.tis_set.mwf_subcycle_small_by_ensemble`. Pure-Python, runs once
# during config load. Removes itself after a successful expansion so restart
# files only carry canonical fields.

_ALLOWED_MOVES = ("sh", "wf", "mwf")


def _padded(idx: int) -> str:
    return f"{idx:03d}"


def _parse_ens_token(tok: str, n_ens: int) -> int:
    """Normalize a single ensemble token (e.g. '3' or '003') to an int index."""
    if not isinstance(tok, str):
        raise TOMLConfigError(
            f"ensemble_move_policy: ensemble token must be a string, got "
            f"{type(tok).__name__}: {tok!r}"
        )
    s = tok.strip()
    if not s or not s.isdigit():
        raise TOMLConfigError(
            f"ensemble_move_policy: invalid ensemble token {tok!r}; "
            f"expected digits like '003'"
        )
    idx = int(s)
    if idx < 0 or idx >= n_ens:
        raise TOMLConfigError(
            f"ensemble_move_policy: ensemble {tok!r} out of range "
            f"[000, {_padded(n_ens - 1)}]"
        )
    return idx


def _parse_selector(spec: object, n_ens: int) -> list[str]:
    """Resolve a selector spec to a sorted list of zero-padded ens_names."""
    if not isinstance(spec, str):
        raise TOMLConfigError(
            f"ensemble_move_policy: selector must be a string, got "
            f"{type(spec).__name__}: {spec!r}"
        )
    raw = spec.strip()
    if ":" not in raw:
        return [_padded(_parse_ens_token(raw, n_ens))]

    lo_tok, hi_tok = raw.split(":", 1)
    lo = 0 if lo_tok.strip() == "" else _parse_ens_token(lo_tok, n_ens)
    hi = (n_ens - 1) if hi_tok.strip() == "" else _parse_ens_token(hi_tok, n_ens)
    if lo > hi:
        raise TOMLConfigError(
            f"ensemble_move_policy: range selector {spec!r} not ordered "
            f"(lo > hi)"
        )
    return [_padded(i) for i in range(lo, hi + 1)]


def _resolve_disjoint(
    selectors: list[object], n_ens: int, *, label: str
) -> list[str]:
    """Expand selectors to ens_names; raise on overlap inside `selectors`."""
    seen: set[str] = set()
    for spec in selectors:
        for name in _parse_selector(spec, n_ens):
            if name in seen:
                raise TOMLConfigError(
                    f"ensemble_move_policy: ensemble {name!r} matched twice "
                    f"by overlapping selectors in {label}"
                )
            seen.add(name)
    return sorted(seen)


def _is_positive_int(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def expand_ensemble_move_policy(config: dict) -> None:
    """Expand `simulation.ensemble_move_policy` into canonical fields.

    No-op when the section is absent or `enabled = false`. Validates the
    full ergonomic block before mutating `config`. On success, removes the
    `ensemble_move_policy` subsection so restarts and re-runs are idempotent.
    Raises `TOMLConfigError` on any malformed input or canonical-field
    conflict (default `conflict_policy = "error"`).
    """
    sim = config["simulation"]
    policy = sim.get("ensemble_move_policy")
    if policy is None:
        return
    if not isinstance(policy, dict):
        raise TOMLConfigError(
            "ensemble_move_policy must be a table"
        )
    if not policy.get("enabled", False):
        # Disabled: leave config unchanged but strip the section so subsequent
        # passes see a clean config and restart files do not carry it.
        sim.pop("ensemble_move_policy", None)
        return

    conflict_policy = policy.get("conflict_policy", "error")
    if conflict_policy != "error":
        raise TOMLConfigError(
            f"ensemble_move_policy: unsupported conflict_policy "
            f"{conflict_policy!r}; only 'error' is implemented"
        )

    n_ens = len(sim["interfaces"])
    if n_ens < 2:
        raise TOMLConfigError(
            "ensemble_move_policy requires at least 2 interfaces"
        )

    default_move = policy.get("default_move", "wf")
    minus_move = policy.get("minus_move", "sh")
    if default_move not in _ALLOWED_MOVES:
        raise TOMLConfigError(
            f"ensemble_move_policy: default_move {default_move!r} not in "
            f"{_ALLOWED_MOVES}"
        )
    if minus_move not in _ALLOWED_MOVES:
        raise TOMLConfigError(
            f"ensemble_move_policy: minus_move {minus_move!r} not in "
            f"{_ALLOWED_MOVES}"
        )

    default_small = policy.get("default_mwf_subcycle_small")
    if default_small is not None and not _is_positive_int(default_small):
        raise TOMLConfigError(
            f"ensemble_move_policy: default_mwf_subcycle_small must be a "
            f"positive int, got {default_small!r}"
        )

    raw_mwf_ens = policy.get("mwf_ensembles", [])
    if not isinstance(raw_mwf_ens, list):
        raise TOMLConfigError(
            "ensemble_move_policy: mwf_ensembles must be a list of strings"
        )
    mwf_names = _resolve_disjoint(
        list(raw_mwf_ens), n_ens, label="mwf_ensembles"
    )

    raw_subcycle_map = policy.get("mwf_subcycle_small", {})
    if not isinstance(raw_subcycle_map, dict):
        raise TOMLConfigError(
            "ensemble_move_policy.mwf_subcycle_small must be a table"
        )
    by_ensemble: dict[str, int] = {}
    for sel, val in raw_subcycle_map.items():
        if not _is_positive_int(val):
            raise TOMLConfigError(
                f"ensemble_move_policy.mwf_subcycle_small[{sel!r}] must be a "
                f"positive int, got {val!r}"
            )
        for name in _parse_selector(sel, n_ens):
            if name in by_ensemble:
                raise TOMLConfigError(
                    f"ensemble_move_policy.mwf_subcycle_small: ensemble "
                    f"{name!r} matched twice by overlapping selectors"
                )
            by_ensemble[name] = int(val)

    # Build the proposed shooting_moves list (without mutating yet).
    moves = [default_move] * n_ens
    moves[0] = minus_move
    mwf_set = set(mwf_names)
    for i in range(n_ens):
        if _padded(i) in mwf_set:
            moves[i] = "mwf"

    # Subcycle overrides only meaningful for ensembles that resolve to "mwf".
    for name, _ in by_ensemble.items():
        idx = int(name)
        if moves[idx] != "mwf":
            raise TOMLConfigError(
                f"ensemble_move_policy.mwf_subcycle_small: ensemble {name!r} "
                f"is not assigned 'mwf' (got {moves[idx]!r}); remove the "
                f"override or add the ensemble to mwf_ensembles"
            )

    # Conflict checks — only after validation succeeds, before any mutation.
    tis_set = sim["tis_set"]
    if "shooting_moves" in sim:
        raise TOMLConfigError(
            "ensemble_move_policy conflicts with simulation.shooting_moves; "
            "remove one (default conflict_policy='error')"
        )
    if "mwf_subcycle_small" in tis_set:
        raise TOMLConfigError(
            "ensemble_move_policy conflicts with "
            "simulation.tis_set.mwf_subcycle_small; remove one"
        )
    if "mwf_subcycle_small_by_ensemble" in tis_set:
        raise TOMLConfigError(
            "ensemble_move_policy conflicts with "
            "simulation.tis_set.mwf_subcycle_small_by_ensemble; remove one"
        )

    # Commit.
    sim["shooting_moves"] = moves
    if default_small is not None:
        tis_set["mwf_subcycle_small"] = int(default_small)
    if by_ensemble:
        tis_set["mwf_subcycle_small_by_ensemble"] = by_ensemble
    sim.pop("ensemble_move_policy", None)
