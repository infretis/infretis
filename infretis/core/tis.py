"""Transition interface sampling methods."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from infretis.classes.engines.factory import create_engines
from infretis.classes.orderparameter import create_orderparameters
from infretis.classes.path import paste_paths

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


ENGINES: dict = {}


def def_globals(config):
    """Define global engine and orderparameter variables.

    Args:
        config: Dictionary with .toml settings.
    """
    global ENGINES

    ENGINES, engine_occ = create_engines(config)
    create_orderparameters(ENGINES, config)
    return engine_occ


if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from numpy.random import Generator

    from infretis.classes.engines.enginebase import EngineBase
    from infretis.classes.path import Path as InfPath
    from infretis.classes.system import System

    # Define a signature for move methods:
    MoveMethod = Callable[..., Tuple[bool, InfPath, str]]


def log_mdlogs(inp: str) -> None:
    """Log performance metrics from log files in a specified directory.

    Args:
        inp: The directory containing the log files to process.

    Returns:
        None, but if a line containing "Performance" is found,
        this will be logged at the info level.

    """
    logs = [log for log in os.listdir(inp) if "log" in log]
    for log in logs:
        with open(os.path.join(inp, log)) as read:
            for line in read:
                if "Performance" in line:
                    logger.info(
                        log + " " + line.rstrip().split()[1] + " ns/day"
                    )


def run_md(md_items: Dict[str, Any]) -> Dict[str, Any]:
    """Execute shooting moves that require MD.

    Args:
        md_items: Parameters and objects needed by the MD simulation.

    Returns:
        The updated `md_items` dictionary with additional results from
        the MD simulation.
    """
    # record start time
    md_items["wmd_start"] = time.time()

    # perform the hw move:
    picked = md_items["picked"]
    _, trials, status = select_shoot(picked)

    # Record data
    for trial, ens_num in zip(trials, picked.keys()):
        log_mdlogs(picked[ens_num]["exe_dir"])
        md_items["moves"].append(md_items["mc_moves"][ens_num + 1])
        md_items["trial_len"].append(trial.length)
        md_items["trial_op"].append((trial.ordermin[0], trial.ordermax[0]))
        md_items["generated"].append(trial.generated)
        if status == "ACC":
            minus = True if ens_num < 0 else False
            trial.weights = calc_cv_vector(
                trial,
                md_items["interfaces"],
                md_items["mc_moves"],
                picked[ens_num]["ens"]["tis_set"]["lambda_minus_one"],
                cap=md_items["cap"],
                minus=minus,
            )
            picked[ens_num]["traj"] = trial

    md_items.update({"status": status, "wmd_end": time.time()})
    return md_items


def calc_cv_vector(
    path: InfPath,
    interfaces: List[float],
    moves: List[str],
    lambda_minus_one: Union[float, bool] = False,
    cap: Optional[float] = None,
    minus: bool = False,
) -> Tuple[float, ...]:
    """Calculate weights for the given path.

    Args:
        path: The path to calculate weights for.
        interfaces: The positions of the interfaces.
        lambda_minus_one: For permeability calculations
        moves: The MC moves performed.
        cap: The cap value for the Wire Fencing (wf) move.
        minus: Indicate if the math is a minus path or not.

    Returns:
        The vector of weights for the given path.
    """
    path_max, _ = path.ordermax

    cv = []
    if minus:
        if lambda_minus_one is not False:
            return (1.0 if lambda_minus_one <= path_max else 0.0,)
        else:
            return (1.0 if interfaces[0] <= path_max else 0.0,)

    for idx, intf_i in enumerate(interfaces[:-1]):
        if moves[idx + 1] == "wf":
            intf_cap = cap if cap is not None else interfaces[-1]
            intfs = [interfaces[0], intf_i, intf_cap]
            cv.append(compute_weight(path, intfs, moves[idx + 1]))
        else:
            cv.append(1.0 if intf_i <= path_max else 0.0)
    cv.append(0.0)
    return tuple(cv)


def compute_weight(path: InfPath, interfaces: List[float], move: str) -> float:
    """Compute the High Acceptance path weight after a MC move.

    This function calculates the weights that will be used in the
    computation of the crossing probability. This trick allows the
    use of the High Acceptance version of Wire Fencing (WF),
    allowing the acceptance of B to A paths. The drawback is that
    swapping moves also need to account for different weights.
    A weight of 1 will be returned for a path not generated by WF.

    Args:
        path: The path to be weighted.
        interfaces: A list of three floats representing interface
            positions, in the order `[left, middle, right]`.
        move: A string representing the MC move to compute
            the weights for. Expects "wf" for Wire Fencing.

    Returns:
        The weight of the path.
    """
    weight = 1.0

    if move == "wf":
        wf_weight, _ = wirefence_weight_and_pick(
            path, interfaces[1], interfaces[2]
        )
        weight = 1.0 * wf_weight

    if path.get_start_point(
        interfaces[0], interfaces[2]
    ) != path.get_end_point(interfaces[0], interfaces[2]):
        if move in ("ss", "wf"):
            weight *= 2

    return weight


def wirefence_weight_and_pick(
    path: InfPath,
    left: float,
    right: float,
    return_seg: bool = False,
    ens_set: Optional[Dict[str, Any]] = None,
) -> Tuple[int, InfPath]:
    """Calculate the weight of a path generated by Wire Fencing (WF).

    The WF path weight is determined by the total sum pasepoints in
    valid sub-paths. The valid WF sub-paths are defined by the left
    and right interfaces: `left -> left`, `left -> right`, and
    `right -> left` sub-paths.

    Args:
        path: This is the input path which will be trimmed.
        left: The position of the LEFT interface.
        right: The position of the RIGHT interface.
        return_seg: If True, return a random valid WF sub-path.

    Returns:
        A tuple containing:
            - n_frames: The weight of the path.
            - segment: Return a random valid WF sub-path if `return_seg`
                is True. Otherwise, return an empty path.

    """
    key_l, key_r = False, False
    path_arr = []
    isave = 0

    for i in range(len(path.phasepoints[:-1])):
        op1 = path.phasepoints[i].order[0]
        op2 = path.phasepoints[i + 1].order[0]

        if (op1 < left and op2 >= right) or (op2 < left and op1 >= right):
            pass
        elif op2 >= left > op1 and not key_l:
            isave, key_l = i, True
        elif op2 < right <= op1 and not key_r:
            isave, key_r = i, True
        elif key_r and op2 >= right > op1:
            key_l, key_r = False, False
        elif True in (key_l, key_r) and (
            op2 < left <= op1 or op2 >= right > op1
        ):
            key_l, key_r = False, False
            path_arr.append((isave, i + 1, i - isave))

    n_frames = sum(i[2] for i in path_arr) if path_arr else 0
    if return_seg and n_frames and ens_set is not None:
        sum_frames = 0
        subpath_select = ens_set["rgen"].random()
        for ipath in path_arr:
            sum_frames += ipath[2]
            if sum_frames / n_frames >= subpath_select:
                new_segment = path.empty_path(maxlen=path.maxlen)
                for j in range(ipath[0], ipath[1] + 1):
                    new_segment.append(path.phasepoints[j])
                new_segment.status = path.status
                new_segment.time_origin = path.time_origin
                new_segment.generated = "ct"
                return n_frames, new_segment
    return n_frames, path.empty_path(maxlen=path.maxlen)


def select_shoot(
    picked: Dict[int, Any],
    start_cond: Tuple[str, ...] = ("L",),
) -> Tuple[bool, List[InfPath], str]:
    """Select shooting move, select engines, and generate a new path.

    Since the engine used now depends on the move chosen, we set the
    engines in 'picked' via this function using the global variable ENGINE.

    Args:
        picked: A dictionary mapping the ensemble indices to their
            settings, including the move and current path.
        start_cond: The starting condition for the path.
            This is determined by the ensemble we are generating
            for - it is right ("R") or left ("L").

    Returns:
        A tuple containing:
            - True if the new path can be accepted, False otherwise.
            - The generated path.
            - A string representing the status of the path.

    """
    sh_moves: Dict[str, MoveMethod] = {
        "wf": wire_fencing,
        "sh": shoot,
        "st": staple,
    }

    # set engine, might also depend on chosen move
    engines = {}
    msg = "Selected engines "
    for ens_num in picked.keys():
        pens = picked[ens_num]
        if len(picked) == 2:
            engines[ens_num] = [
                ENGINES[eng][idx] for eng, idx in pens["eng_idx"].items()
            ]
        else:
            engines[0] = [
                ENGINES[eng][idx] for eng, idx in pens["eng_idx"].items()
            ]
        msg += f"{list(pens['eng_idx'].keys())} "
    logger.info(msg + "for MC move.")

    # Set mdrun, rng, then clean_up.
    for key, ens_num in zip(engines.keys(), picked.keys()):
        pens = picked[ens_num]
        for engine in engines[key]:
            engine.set_mdrun(pens)
            if "rgen-eng" in pens:
                engine.rgen = pens["rgen-eng"]
            engine.clean_up()

    if len(picked) == 1:
        pens = next(iter(picked.values()))
        ens_set, path = (pens[i] for i in ["ens", "traj"])
        move = ens_set["mc_move"]
        logger.info(
            f"starting {move} in {ens_set['ens_name']}"
            + f" with path_n {path.path_number}"
        )
        start_cond = ens_set["start_cond"]
        accept, new_path, status = sh_moves[move](
            ens_set, path, engines[0][0], start_cond=start_cond
        )
        new_paths = [new_path]
    else:
        if -1 in picked.keys():
            if picked[-1]["ens"]["tis_set"]["quantis"]:
                accept, new_paths, status = quantis_swap_zero(picked, engines)
            else:
                accept, new_paths, status = retis_swap_zero(picked, engines)
        else:
            # logger.info(f"Shooting sh sh in ensembles: {ens_nums[0]} <-> {ens_nums[1]}"\
            #             f" with paths: {pnums} and worker: {md_items['pin']}")
            accept, new_paths, status = ppretis_swap(picked, engines)

    logger.info(f"Move was {accept} with status {status}\n")
    return accept, new_paths, status


def shoot(
    ens_set: Dict[str, Any],
    path: InfPath,
    engine: EngineBase,
    shooting_point: Optional[System] = None,
    start_cond: Tuple[str, ...] = ("L",),
) -> Tuple[bool, InfPath, str]:
    """Perform a shooting move.

    Args:
        ens_set: Settings for the ensemble.
        path: The initial path to shoot from.
        engine: The MD engine used for generating a new path.
        shooting_point: The selected shooting point. If None,
            it will be randomly selected here.
        start_cond: The starting condition for the ensemble as
            left ("L") or right ("R").

    Returns:
        A tuple containing:
            - True if the new path can be accepted, False otherwise.
            - The generated path.
            - A string representing the status of the path.

    """
    interfaces = ens_set["interfaces"]
    # the trial path we will generate
    trial_path = path.empty_path(maxlen=ens_set["tis_set"]["maxlength"])
    if shooting_point is None:
        shooting_point, idx, dek = prepare_shooting_point(
            path, ens_set["rgen"], engine, ens_set
        )
        kick = check_kick(
            shooting_point, interfaces, trial_path, ens_set["rgen"], dek
        )
    else:
        kick = True
        idx = getattr(shooting_point, "idx", 0)

    # Store info about this point, just in case we have to return
    # before completing a full new path:
    trial_path.generated = ("sh", shooting_point.order[0], idx, 0)
    trial_path.time_origin = path.time_origin + idx
    # We now check if the kick was OK or not:
    if not kick:
        return False, trial_path, trial_path.status
    # OK: kick was either aimless or it was accepted by Metropolis
    # We should now generate trajectories, but first check how long
    # it should be (if the path comes from a load, it is assumed to not
    # respect the detail balance anyway):
    if path.get_move() == "ld" or ens_set["tis_set"].get(
        "allowmaxlength", False
    ):
        maxlen = ens_set["tis_set"]["maxlength"]
    else:
        maxlen = min(
            int((path.length - 2) / ens_set["rgen"].random()) + 2,
            ens_set["tis_set"]["maxlength"],
        )
    # Since the forward path must be at least one step, the maximum
    # length for the backward path is maxlen-1.

    # Generate the backward path:
    path_back = path.empty_path(maxlen=maxlen - 1)
    # todo this inputs are a mess
    # Set ensemble state to the selected shooting point:
    # ensemble['system'] = shooting_point.copy()
    shpt_copy = shooting_point.copy()
    if not shoot_backwards(
        path_back, trial_path, shpt_copy, ens_set, engine, start_cond
    ):
        return False, trial_path, trial_path.status

    # Generate forward path:
    # Note that the length of the forward path is adjusted to
    # account for the fact that it shares a point with the backward
    # path (i.e. the shooting point). The duplicate point is just
    # counted once when the paths are merged by the method
    # `paste_paths` by setting `overlap=True`.
    path_forw = path.empty_path(maxlen=(maxlen - path_back.length + 1))
    logger.debug("Propagating forwards for shooting move...")
    # Set ensemble state to the selected shooting point:
    # change the system state.
    # ensemble['system'] = shooting_point.copy()
    shpt_copy = shooting_point.copy()
    success_forw, _ = engine.propagate(
        path_forw, ens_set, shpt_copy, reverse=False
    )
    path_forw.time_origin = trial_path.time_origin
    # Now, the forward propagation could have failed by exceeding the
    # maximum length for the forward path. However, it could also fail
    # when we paste together so that the length is larger than the
    # allowed maximum. We paste first and ask later:
    trial_path = paste_paths(
        path_back,
        path_forw,
        overlap=True,
        maxlen=ens_set["tis_set"]["maxlength"],
    )

    # Also update information about the shooting:
    trial_path.generated = (
        "sh",
        shooting_point.order[0],
        idx,
        path_back.length - 1,
    )
    if not success_forw:
        trial_path.status = "FTL"
        # If we reached this point, the backward path was successful,
        # but the forward was not. For the case where the forward was
        # also successful, the length of the trial path cannot exceed
        # the maximum length given in the TIS settings. Thus we only
        # need to check this here, i.e. when given that the backward
        # was successful and the forward not:
        if trial_path.length == ens_set["tis_set"]["maxlength"]:
            trial_path.status = "FTX"  # exceeds "memory".
        return False, trial_path, trial_path.status

    trial_path.weight = 1.0

    # Deal with the rejections for path properties.
    # Make sure we did not hit the left interface on {0-}
    # Which is the only ensemble that allows paths starting in R
    if (
        "L" not in set(start_cond)
        and "L" in trial_path.check_interfaces(interfaces)[:2]
    ):
        trial_path.status = "0-L"
        return False, trial_path, trial_path.status

    # Last check - Did we cross the middle interface?
    if "must_cross_M" in ens_set and ens_set["must_cross_M"]: # not for [0-]

        # detect if: minorder < middle interf <= maxorder
        if not trial_path.check_interfaces(interfaces)[-1][1]:
            trial_path.status = 'NCR'
            return False, trial_path, trial_path.status

        else:
            # if we do cross middle interface
            # path still has to come from left or right
            # so we need cross[0] or cross[2] to be true
            if not (trial_path.check_interfaces(interfaces)[-1][0] or
                    trial_path.check_interfaces(interfaces)[-1][2]):
                trial_path.status = 'NCR'
                return False, trial_path, trial_path.status



    # Don't do this for paths that can start everywhere
    start_cond = ens_set.get("start_cond", start_cond)
    if set(("R", "L")) == set(start_cond):
        pass
    elif not trial_path.check_interfaces(interfaces)[-1][1]:
        # No, we did not cross the middle interface:
        trial_path.status = "NCR"
        return False, trial_path, trial_path.status

    trial_path.status = "ACC"

    return True, trial_path, trial_path.status


def wire_fencing(
    ens_set: Dict[str, Any],
    trial_path: InfPath,
    engine: EngineBase,
    start_cond: Tuple[str, ...] = ("L",),
) -> Tuple[bool, InfPath, str]:
    """Perform a Wire Fencing move from an initial path.

    Args:
        ens_set: Ensemble settings.
        trial_path: The path to perform the move from.
        engine: The MD engine used for propagating.
        start_cond: The starting condition for the ensemble as
            left ("L") or right ("R").

    Returns:
        A tuple containing:
            - True if the path can be accepted, False otherwise.
            - The generated path.
            - A string representing the status of the path.

    """
    intf_cap = ens_set["tis_set"].get(
        "interface_cap",
        ens_set["interfaces"][2],
    )
    wf_int = list([ens_set["interfaces"][1]] * 2) + [intf_cap]
    n_frames, new_segment = wirefence_weight_and_pick(
        trial_path, wf_int[0], wf_int[2], return_seg=True, ens_set=ens_set
    )

    # Check if no frames to shoot from
    if n_frames == 0:
        logger.warning("Wire fencing move not usable. N frames of Path = 0")
        logger.warning(f"between interfaces {wf_int[0]} and {wf_int[-1]}.")
        return False, trial_path, "NSG"

    sub_ens = {
        "interfaces": wf_int,
        "rgen": ens_set["rgen"],
        "ens_name": ens_set["ens_name"],
        "start_cond": ens_set["start_cond"],
        "tis_set": ens_set["tis_set"],
    }
    sub_ens["tis_set"]["allowmaxlength"] = True
    sub_ens["tis_set"]["maxlength"] = ens_set["tis_set"]["maxlength"]

    succ_seg = 0
    for i in range(ens_set["tis_set"].get("n_jumps", 2)):
        logger.debug("Trying a new web with Wire Fencing, jump %i", i)
        success, trial_seg, status = shoot(
            sub_ens, new_segment, engine, start_cond=("L", "R")
        )
        start, end, _, _ = trial_seg.check_interfaces(wf_int)
        logger.info(
            f"Jump {i}, len {trial_seg.length}, status"
            + f"{status}, intf: {start} {end}"
        )
        if not success:
            # This handles R to R (start_cond = L) paths. Counter + 1, no ups.
            logger.debug("Wire Fencing Fail.")
        else:
            logger.debug("Acceptable Wire Fence link.")
            succ_seg += 1
            new_segment = trial_seg.copy()
    if succ_seg == 0:
        # No usable segments were generated.
        trial_path.status = "NSG"
        success = False
    else:
        success, trial_path, _ = extender(
            new_segment, engine, ens_set, start_cond
        )
    if success:
        success, trial_path = subt_acceptance(
            trial_path, ens_set, engine, start_cond
        )

    trial_path.generated = ("wf", 9000, succ_seg, trial_path.length)

    logger.debug("WF move %s", trial_path.status)
    if not success:
        return False, trial_path, trial_path.status

    # This might get triggered when accepting 0-L paths.
    left, _, right = ens_set["interfaces"]
    # TODO: check this
    # print(start_cond, tuple(trial_path.get_start_point(left, right)))
    assert set(start_cond) == set(
        trial_path.get_start_point(left, right)
    ), "WF: Path has an implausible start."

    trial_path.status = "ACC"
    return True, trial_path, trial_path.status


def subt_acceptance(
    trial_path: InfPath,
    ens_set: Dict[str, Any],
    engine: EngineBase,
    start_cond: Tuple[str, ...] = ("L",),
) -> Tuple[bool, InfPath]:
    """Process and weight paths generated by Wire Fencing (WF).

    This method will weight and potentially reverse generated paths by
    WF. It will also judge if the path can be accepted or if it should
    be rejected.

    Args:
        trial_path: This is the new path that will obtain weights,
            and might be reversed and accepted.
        ens_set: Ensemble settings.
        engine: The MD engine used for propagating.
        start_cond: The starting condition for the ensemble as
            left ("L") or right ("R").

    Returns:
        A tuple containing:
            - True if the path can be accepted. False otherwise.
            - The weighed and possibly reversed path.

    """
    intf = list(ens_set["interfaces"])
    move = ens_set["mc_move"]

    if move == "wf":
        intf[2] = ens_set["tis_set"].get("interface_cap", intf[2])
    trial_path.weight = compute_weight(trial_path, intf, move)

    if set(start_cond) != set(trial_path.get_start_point(intf[0], intf[2])):
        trial_path = trial_path.reverse(engine.order_function)

    if set(start_cond) != set(trial_path.get_start_point(intf[0], intf[2])):
        trial_path.status = "BWI"
        return False, trial_path
    trial_path.status = "ACC"
    return True, trial_path


def extender(
    source_seg: InfPath,
    engine: EngineBase,
    ens_set: Dict[str, Any],
    start_cond: Tuple[str, ...] = ("R", "L"),
) -> Tuple[bool, InfPath, str]:
    """Extend a path segment backward and forward in time.

    Args:
        source_seg: The path (segment) to extend.
        engine: The MD engine used for propagating.
        ens_set: Ensemble settings.
        start_cond: The starting condition for the ensemble as
            left ("L") or right ("R").

    Returns:
        A tuple containing:
            - True if the path can be accepted, False otherwise.
            - The generated path.
            - A string representing the status of the path.
    """
    interfaces = ens_set["interfaces"]
    # ensemble['system'] = source_seg.phasepoints[0].copy()
    sh_pt = source_seg.phasepoints[0].copy()

    # Extender
    if interfaces[0] <= sh_pt.order[0] < interfaces[-1]:
        back_segment = source_seg.empty_path(
            maxlen=ens_set["tis_set"]["maxlength"]
        )
        logger.debug("Trying to extend backwards")
        source_seg_copy = source_seg.copy()

        shoot_backwards(
            back_segment, source_seg_copy, sh_pt, ens_set, engine, start_cond
        )
        trial_path = paste_paths(
            back_segment,
            source_seg,
            overlap=True,
            maxlen=ens_set["tis_set"]["maxlength"],
        )
    else:
        trial_path = source_seg.copy()

    sh_pt = trial_path.phasepoints[-1].copy()
    if interfaces[0] <= sh_pt.order[0] < interfaces[-1]:
        forth_segment = source_seg.empty_path(
            maxlen=ens_set["tis_set"]["maxlength"]
        )
        engine.propagate(forth_segment, ens_set, sh_pt)

        trial_path.phasepoints = (
            trial_path.phasepoints[:-1] + forth_segment.phasepoints
        )

    if trial_path.length >= ens_set["tis_set"]["maxlength"]:
        trial_path.status = "FTX"  # exceeds "memory".
        return False, trial_path, trial_path.status
    trial_path.status = "ACC"
    return True, trial_path, trial_path.status


def shoot_backwards(
    path_back: InfPath,
    trial_path: InfPath,
    system: System,
    ens_set: Dict[str, Any],
    engine: EngineBase,
    start_cond: Tuple[str, ...],
) -> bool:
    """Propagate a path in the backward time direction.

    Args:
        path_back: The path to be filled with phase points from the
            propagation.
        trial_path: The current trial path generated by the shooting.
        system: The phase point to shoot from.
        ens_set: Ensemble settings.
        engine: The MD engine used for propagating.
        start_cond: The starting condition for the ensemble as
            left ("L") or right ("R").

    Returns:
        True if the backward path was generated successfully, False
        otherwise.

    """
    logger.debug("Propagating backwards for the shooting move.")
    path_back.time_origin = trial_path.time_origin
    success_back, _ = engine.propagate(
        path_back, ens_set, system, reverse=True
    )
    if not success_back:
        # Something went wrong, most probably the path length was exceeded.
        trial_path.status = "BTL"  # BTL = backward trajectory too long.
        # Add the failed path to trial path for analysis:
        trial_path += path_back
        if path_back.length >= ens_set["tis_set"]["maxlength"] - 1:
            # BTX is backward trajectory longer than maximum memory.
            trial_path.status = "BTX"
        return False
    # Backward seems OK so far, check if the ending point is correct:
    left, _, right = ens_set["interfaces"]
    if path_back.get_end_point(left, right) not in set(start_cond):
        # Nope, backward trajectory end at wrong interface.
        trial_path += path_back  # Store path for analysis.
        trial_path.status = "BWI"
        return False
    return True


def prepare_shooting_point(
    path: InfPath, rgen: Generator, engine: EngineBase, ens_set: Dict[str, Any]
) -> Tuple[System, int, float]:
    """Select and modify velocities for a shooting move.

    This method will randomly select a shooting point from a given
    path and modify its velocities.

    Args:
        path: This is the input path which will be used for generating a
        new path.
        rgen: A random number generator used for selecting the shooting
            point from the path.
        engine: The MD engine used for generating velocities.

    Returns:
        A tuple containing:
            - The shooting point with modified velocities.
            - The index of the shooting point in the original path.
            - The change in kinetic energy when modifying the velocities.

    """
    shooting_point, idx = path.get_shooting_point(rgen)
    orderp = shooting_point.order
    shpt_copy = shooting_point.copy()
    logger.info("Shooting from order parameter/index: %f, %d", orderp[0], idx)
    # Copy the shooting point, so that we can modify velocities without
    # altering the original path:
    # Modify the velocities:
    dek, _ = engine.modify_velocities(shpt_copy, ens_set["tis_set"])
    orderp = engine.calculate_order(shpt_copy)
    shpt_copy.order = orderp
    return shpt_copy, idx, dek


def check_kick(
    shooting_point: System,
    interfaces: List[float],
    trial_path: InfPath,
    rgen: Generator,
    dek: float,
) -> bool:
    """Check the modification of the shooting point.

    After generating velocities for a shooting point, we
    do some additional checking to see if the shooting point is
    acceptable.

    Args:
        shooting_point: The shooting point with modified velocities.
        interfaces: The interfaces on the form `[left, middle, right]`.
        trial_path: The path we are currently generating.
        rgen: A random generator used to check if we accept the
            shooting point based on the change in kinetic energy.
        dek: The change in kinetic energy when modifying the velocities.

    Returns:
        True if the modification was acceptable.

    """
    # Check if the kick was too violent:
    left, _, right = interfaces
    if not left <= shooting_point.order[0] < right:
        # Shooting point was velocity dependent and was kicked outside
        # of boundaries when modifying velocities.
        trial_path.append(shooting_point)
        trial_path.status = "KOB"
        return False
    return True


def retis_swap_zero(
    picked: Dict[int, Any],
    engines: Dict[int, List[EngineBase]],
) -> Tuple[bool, List[InfPath], str]:
    """Perform the RETIS swapping for `[0^-] <-> [0^+]` swaps.

    Args:
        picked: A dictionary mapping the ensemble indices to their
            settings, including the move and current path.

        engines: The engines used to propagate the system.

    Returns:
        A tuple containing:
            - True if the path can be accepted, False otherwise.
            - The generated paths.
            - A string representing the status of the paths.

    Note:
        The swapping move for ensembles [0^-] and [0^+] requires some
        extra propagation. Here we are generating new paths for [0^-]
        and [0^+] in the following way:

        1) For [0^-] we take the initial point in [0^+] and integrate
           backward in time. This is merged with the second point in [0^+]
           to give the final path. The initial point in [0^+] starts to the
           left of the interface and the second point is on the right
           side - i.e., the path will cross the interface at the end points.
           If we let the last point in [0^+] be called `A_0` and the
           second last point `B`, and we let `A_1, A_2, ...` be the
           points on the backward trajectory generated from `A_0` then
           the final path will be made up of the points
           `[..., A_2, A_1, A_0, B]`. Here, `B` will be on the right
           side of the interface and the first point of the path will also
           be on the right side.

        2) For [0^+] we take the last point of [0^-] and use that as an
           initial point to generate a new trajectory for [0^+] by
           integration forward in time. We also include the second last
           point of the [0^-] trajectory which is on the left side of the
           interface. We let the second last point be `B` (this is on the
           left side of the interface), the last point `A_0` and the
           points generated from `A_0` we denote by `A_1, A_2, ...`.
           Then the resulting path will be `[B, A_0, A_1, A_2, ...]`.
           Here, `B` will be on the left side of the interface and the
           last point of the path will also be on the left side of the
           interface.

    """
    ens_set0 = picked[-1]["ens"]
    ens_set1 = picked[0]["ens"]
    engine0 = engines[-1][0]
    engine1 = engines[0][0]
    path_old0 = picked[-1]["traj"]
    path_old1 = picked[0]["traj"]
    maxlen0 = ens_set0["tis_set"]["maxlength"]
    maxlen1 = ens_set1["tis_set"]["maxlength"]

    ens_moves = [ens_set0["mc_move"], ens_set1["mc_move"]]
    intf_w = [list(ens_set0["interfaces"]), list(ens_set1["interfaces"])]

    # intf_w = [list(i) for i in (path_ensemble0.interfaces,
    #                             path_ensemble1.interfaces)]
    for i, mc_move in enumerate([ens_set0["tis_set"], ens_set1["tis_set"]]):
        intf_w[i][2] = mc_move.get("interface_cap", intf_w[i][2])

    # for i, j in enumerate([settings['ensemble'][k] for k in (0, 1)]):
    #     if ens_moves[i] == 'wf':
    #         intf_w[i][2] = j['tis'].get('interface_cap', intf_w[i][2])

    # 0. check if MD is allowed
    # allowed = (path_ensemble0.last_path.get_end_point(
    #             path_ensemble0.interfaces[0],
    #             path_ensemble0.interfaces[-1]) == 'R')
    allowed = (
        path_old0.get_end_point(
            ens_set0["interfaces"][0], ens_set0["interfaces"][-1]
        )
        == "R"
    )
    # if allowed:
    #     swap_ensemble_attributes(ensemble0, ensemble1, settings)

    # if lambda_minus_one, reject early if path_old0
    if set(ens_set0["start_cond"]) == set(["L", "R"]):
        if path_old0.check_interfaces(ens_set0["interfaces"])[1] == "L":
            return False, [path_old0, path_old1], "0-L"

    # 1. Generate path for [0^-] from [0^+]:
    # We generate from the first point of the path in [0^+]:
    logger.info("Swapping [0^-] <-> [0^+]")
    logger.info("Creating path for [0^-]")
    # system = path_ensemble1.last_path.phasepoints[0].copy()
    shpt_copy = path_old1.phasepoints[0].copy()
    # shpt_copy2 = path_old1.phasepoints[0].copy()
    logger.info("Initial point is: %s", shpt_copy.order)
    # Propagate it backward in time:
    path_tmp = path_old1.empty_path(maxlen=maxlen1 - 1)
    if allowed:
        logger.info("Propagating for [0^-]")
        engine0.propagate(path_tmp, ens_set0, shpt_copy, reverse=True)
    else:
        logger.info("Not propagating for [0^-]")
        path_tmp.append(shpt_copy)
    path0 = path_tmp.empty_path(maxlen=maxlen0)
    for phasepoint in reversed(path_tmp.phasepoints):
        path0.append(phasepoint)
    # print('lobster a', path_tmp.length, path0.length, allowed)
    # Add second point from [0^+] at the end:
    logger.info("Adding second point from [0^+]:")
    # Here we make a copy of the phase point, as we will update
    # the configuration and append it to the new path:
    # phase_point = path_ensemble1.last_path.phasepoints[1].copy()
    phase_point = path_old1.phasepoints[1].copy()
    logger.info("Point is %s", phase_point.order)
    engine1.dump_phasepoint(phase_point, "second")
    path0.append(phase_point)
    if path0.length == maxlen0:
        path0.status = "BTX"
    elif path0.length < 3:
        path0.status = "BTS"
    elif (
        "L" not in set(ens_set0["start_cond"])
        and "L" in path0.check_interfaces(ens_set0["interfaces"])[:2]
    ):
        path0.status = "0-L"
    else:
        path0.status = "ACC"

    # 2. Generate path for [0^+] from [0^-]:
    logger.info("Creating path for [0^+] from [0^-]")
    # This path will be generated starting from the LAST point of [0^-] which
    # should be on the right side of the interface. We will also add the
    # SECOND LAST point from [0^-] which should be on the left side of the
    # interface, this is added after we have generated the path and we
    # save space for this point by letting maxlen = maxlen1-1 here:
    path_tmp = path0.empty_path(maxlen=maxlen1 - 1)
    # We start the generation from the LAST point:
    # Again, the copy below is not needed as the propagate
    # method will not alter the initial state.
    # system = path_ensemble0.last_path.phasepoints[-1].copy()
    system = path_old0.phasepoints[-1].copy()
    if allowed:
        logger.info("Initial point is %s", system.order)
        # nsembles[1]['system'] = system
        logger.info("Propagating for [0^+]")
        engine1.propagate(path_tmp, ens_set1, system, reverse=False)
        # Ok, now we need to just add the SECOND LAST point from [0^-] as
        # the first point for the path:
        path1 = path_tmp.empty_path(maxlen=maxlen1)
        # phase_point = path_ensemble0.last_path.phasepoints[-2].copy()
        phase_point = path_old0.phasepoints[-2].copy()
        logger.info("Add second last point: %s", phase_point.order)
        engine0.dump_phasepoint(phase_point, "second_last")
        path1.append(phase_point)
        path1 += path_tmp  # Add rest of the path.
    else:
        path1 = path_tmp
        path1.append(system)
        logger.info("Skipping propagating for [0^+] from L")

    ##### NB if path_ensemble1.last_path.get_move() != 'ld':
    ##### NB     path0.set_move('s+')
    ##### NB else:
    ##### NB     path0.set_move('ld')

    ##### NB if path_ensemble0.last_path.get_move() != 'ld':
    ##### NB     path1.set_move('s-')
    ##### NB else:
    ##### NB     path1.set_move('ld')
    if path1.length >= maxlen1:
        path1.status = "FTX"
    elif path1.length < 3:
        path1.status = "FTS"
    else:
        path1.status = "ACC"
    logger.info("Done with swap zero!")

    # Final checks:
    accept = path0.status == "ACC" and path1.status == "ACC"
    status = (
        "ACC"
        if accept
        else (path0.status if path0.status != "ACC" else path1.status)
    )
    # High Acceptance swap is required when Wire Fencing are used
    if accept:
        if "wf" in ens_moves:
            accept, status = high_acc_swap(
                [path1, path_old1],
                ens_set0["rgen"],
                intf_w[0],
                intf_w[1],
                ens_moves,
            )

    for i, path, _, _ in (
        (0, path0, ens_set0["tis_set"], "s+"),
        (1, path1, ens_set1["tis_set"], "s-"),
    ):
        if not accept and path.status == "ACC":
            path.status = status

        # These should be 1 unless length of paths equals 3.
        # This technicality is not yet fixed. (An issue is open as a reminder)

        # ens_set = settings['ensemble'][i]
        move = ens_moves[i]
        path.weight = (
            compute_weight(path, intf_w[i], move) if move in ("wf") else 1
        )

    return accept, [path0, path1], status


def high_acc_swap(
    paths: List[InfPath],
    rgen: Generator,
    intf0: List[float],
    intf1: List[float],
    ens_moves: List[str],
) -> Tuple[bool, str]:
    """Accept/reject a swap move using the High Acceptance weights.

    Args:
        paths: The path in the LOWER and UPPER ensemble to exchange.
        rgen: The random number generator.
        intf0: The interfaces of the LOWER ensemble.
        intf1: The interfaces of the HIGHER ensemble.
        ens_moves: The moves used in the two ensembles.

    Returns:
        A tuple containing:
            - True if the move is accepted, False otherwise.
            - A string with the status of acceptance.


    Notes:
        -  This function is only needed when paths are generated via
            Wire Fencing.

    """
    # Crossing before the move
    c1_old = compute_weight(paths[0], intf0, ens_moves[0])
    c2_old = compute_weight(paths[1], intf1, ens_moves[1])
    # Crossing if the move would be accepted
    c1_new = compute_weight(paths[1], intf0, ens_moves[0])
    c2_new = compute_weight(paths[0], intf1, ens_moves[1])
    if c1_old == 0 or c2_old == 0:
        logger.warning(
            "div_by_zero. c1_old, c2_old, ens_moves: [%i,%i], %s",
            c1_old,
            c2_old,
            str(ens_moves),
        )
        p_swap_acc = 1.0
    else:
        p_swap_acc = c1_new * c2_new / (c1_old * c2_old)

    # Finally, randomly decide to accept or not:
    if rgen.random() < p_swap_acc:
        return True, "ACC"  # Accepted

    return False, "HAS"  # Rejected


def quantis_swap_zero(
    picked: Dict[int, Any],
    engines: Dict[int, List[EngineBase]],
) -> Tuple[bool, List[InfPath], str]:
    """Perform a Quantis swap between the [0-] and [0+] ensembles.

    Args:
        picked: A dictionary mapping the ensemble indices to their
            settings, including the move and the paths to be swapped.

        engines: A dictionary containing the two engines.

    Returns:
        A tuple containing:
            - True if the path can be accepted, False otherwise.
            - The generated paths.
            - A string representing the status of the paths.


    The quantis swap is similar to a retis zero swap, except that the [0-]
    and [0+] ensembles are treated at two different levels of theory. To
    obey detailed balance, we need to check if the energy differences
    between the configurations to be swapped are in accord with the metropolis
    acceptance rule. The metropolis acceptance rule is defined by the
    following energies:

        old_path0.phasepoints[-2].vpot <- V_lo(r_lo)
        old_path1.phasepoints[0].vpot <- V_hi(r_hi)
        tmp_path1.phasepoints[0].vpot <- V_hi(r_lo)
        tmp_path0.phasepoints[0].vpot <- V_lo(r_hi)
        deltaV <- V(r_lo) - V(r_hi)
        pacc = exp(-(beta_lo*deltaV_lo - beta_hi*deltaV_hi))

    The quantis swap can be viewed as a generalization of the retis zero
    swap; when the two levels of theroy are identical, the energy
    differences become zero, and the acceptance probability is unity.

    We start by integrating one step in each ensemble. This gives us
    the energy of the configuration at the different level of theory,
    which is the 0th step, but also lets us check that we crosse lambda0
    in one step, which is a condition that must be met. We could also
    start with evaluating the energy acceptance/rejection by running a
    0-step integration (which is also time consuming) and then run 1 step
    and check the crossing. However, this would mean restarting twice and
    may be slower due to wavefunction guesses, so we do it the other way
    around.

    The method is described in detail in:
        Lervik, A., & van Erp, T. S. (2015). Gluing potential energy surfaces
        with rare event simulations [https://doi.org/10.1021/acs.jctc.5b00012]

    Todo:
        * Implement the option to mix engines in [eninge] and [engine2], as
        quantis now only works properly with the same engine in all ensembles
        due to different units and file formats being used. For example, to
        extract a  configuration from [0+] into [0-] requires some processing.
        * Add options to relax crossing condition and energy acceptance rule
        * Option to do 'wf' or nah?
        * After performing a swap, another swap that happens before any moves
        in the ensembles [0-] and [0+] are performed, we get back the original
        paths. Should avoid zero swap if this is the case (see retis_swap_0)

    """
    ens_set0 = picked[-1]["ens"]
    ens_set1 = picked[0]["ens"]
    engine0 = engines[-1][0]
    engine1 = engines[0][0]
    old_path0 = picked[-1]["traj"]
    old_path1 = picked[0]["traj"]
    maxlen0 = ens_set0["tis_set"]["maxlength"]
    maxlen1 = ens_set0["tis_set"]["maxlength"]
    lambda0 = ens_set0["interfaces"][-1]

    logger.info("Quantis swapping [0^-] <-> [0^+].")

    if "wf" in [ens_set0["mc_move"], ens_set1["mc_move"]]:
        logger.warning("Quantis with 'wf' in [0-] or [0+] is not implemented")
        logger.warning("Continuing with regular shooting.")

    shooting_point0 = old_path1.phasepoints[0].copy()
    shooting_point1 = old_path0.phasepoints[-2].copy()

    tmp_path0 = old_path1.empty_path(maxlen=2)
    tmp_path1 = old_path0.empty_path(maxlen=2)
    # add some information in case we return early
    tmp_path0.generated = ("q+", shooting_point0.order[0], 0, 0)
    tmp_path1.generated = (
        "q-",
        shooting_point1.order[0],
        len(old_path0.phasepoints) - 2,
        0,
    )

    # check that we have energies in the two paths
    if None in [shooting_point0.vpot, shooting_point1.vpot]:
        message = " Shooting point in [0-] or [0+] did not contain energies!"
        logger.info(message)
        status = "QNE"
        # add shooting points for debugging purposes
        tmp_path0.append(shooting_point0)
        tmp_path1.append(shooting_point1)
        tmp_path0.status = status
        tmp_path1.status = status
        logger.info(message)
        return False, [tmp_path0, tmp_path1], status

    # check that we actually start at the left side of interface0
    # before beginning the propagation
    start_cond0 = "L" if shooting_point0.order[0] < lambda0 else "R"
    start_cond1 = "L" if shooting_point1.order[0] < lambda0 else "R"
    if start_cond0 != "L" or start_cond1 != "L":
        logger.warning(f"{start_cond0} {start_cond1} != L L!")
        logger.warning(
            "One or both of the shooting points do not start on"
            " the left side of lambda0."
        )
        logger.warning("This should not happen in a stable simulation.")
        status = "QLL"
        tmp_path0.append(shooting_point0)
        tmp_path1.append(shooting_point1)
        tmp_path0.status = status
        tmp_path1.status = status
        return False, [tmp_path0, tmp_path1], status

    # propagate one step in [0-] and check the crossing condition
    logger.info("Propagating one step in [0-]")
    logger.info(f"Initial point for [0-] is: {shooting_point0.order}")
    _, msg = engine0.propagate(tmp_path0, ens_set0, shooting_point0)

    if tmp_path0.get_end_point(lambda0) != "R":
        logger.info("One-step crossing condition failed for new [0-] path.")
        logger.info(f"Reason: {msg}")
        status = "QS0"
        tmp_path0.status = status
        tmp_path1.status = status
        tmp_path1.append(shooting_point1)
        return False, [tmp_path0, tmp_path1], status

    # propagate one step in [0+] and check the crossing condition
    logger.info("Propagating one step in [0+]")
    logger.info(f"Initial point for [0+] is: {shooting_point1.order}")
    _, msg = engine1.propagate(tmp_path1, ens_set0, shooting_point1)

    if tmp_path1.get_end_point(lambda0) != "R":
        logger.info("One-step crossing condition failed for new [0+] path.")
        logger.info("Reason: {msg}")
        status = "QS1"
        tmp_path1.status = status
        tmp_path1.status = status
        return False, [tmp_path0, tmp_path1], status

    # now we check the energy acceptance rule
    V0_r0 = old_path0.phasepoints[-2].vpot
    V0_r1 = tmp_path0.phasepoints[0].vpot
    V1_r1 = old_path1.phasepoints[0].vpot
    V1_r0 = tmp_path1.phasepoints[0].vpot
    logger.info(f"V0r0 {V0_r0:.4e} V0r1 {V0_r1:.4e} dV0 {V0_r0 - V0_r1:.4e}")
    logger.info(f"V1r0 {V1_r0:.4e} V1r1 {V1_r1:.4e} dV1 {V1_r0 - V1_r1:.4e}")
    deltaV0 = V0_r0 - V0_r1
    deltaV1 = V1_r0 - V1_r1
    pacc = min(1.0, np.exp(deltaV0 * engine0.beta - deltaV1 * engine1.beta))
    rand = ens_set0["rgen"].random()
    if ens_set0["tis_set"]["accept_all"]:
        logger.info(f"Accepting all zero swaps! Actual Pacc = {pacc}")
    elif rand <= pacc:
        logger.info(f"Energy acceptance rule checks out! Pacc = {pacc}")
    else:
        status = "QEA"
        tmp_path0.status = status
        tmp_path1.status = status
        logger.info(f"Random nr {rand} > pacc {pacc}! Rejecting zero swap.")
        return False, [tmp_path1, tmp_path1], status

    # The energies check out, now complete the two paths.
    # We start with backward propagation in [0-]

    shooting_point0 = tmp_path0.phasepoints[0].copy()
    new_path0 = tmp_path0.empty_path(maxlen=maxlen0 - 1)

    # check that we actually start on the correct side of the interface
    start_cond0 = "L" if shooting_point0.order[0] < lambda0 else "R"
    if start_cond1 != "L":
        logger.warning("The phasepoint from [0-] from L is now on R side!")
        logger.warning("This should not happen!")
        status = "QR*"
        tmp_path0.status = status
        tmp_path1.status = status
        new_path0.append(shooting_point0)
        return False, [new_path0, tmp_path1], status

    logger.info("Propagating backwards in [0-]")
    _, msg = engine0.propagate(
        new_path0,
        ens_set0,
        shooting_point0,
        reverse=True,
    )

    # obtain the final full path in [0-]:
    # append the one-step path 'tmp_path0' (which is the end point)
    # to the rest of the 'new_path0', which was propagated in reverse
    new_path0 = paste_paths(new_path0, tmp_path0, maxlen=maxlen0)

    # finished with path0, now do some checks
    if new_path0.length >= maxlen0:
        new_path0.status = "BTX"
    elif new_path0.length < 3:
        new_path0.status = "BTS"
    elif (
        "L" not in set(ens_set0["start_cond"])
        and "L" in new_path0.check_interfaces(ens_set0["interfaces"])[:2]
    ):
        new_path0.status = "0-L"
    else:
        new_path0.status = "ACC"

    if new_path0.status != "ACC":
        return False, [new_path0, tmp_path1], new_path0.status

    # Finally, propagate the [0+] path
    shooting_point1 = tmp_path1.phasepoints[-1].copy()
    new_path1 = tmp_path1.empty_path(maxlen=maxlen1 - 1)

    # but check that we actually start on the correct side of the interface
    start_cond1 = "L" if shooting_point1.order[0] < lambda0 else "R"
    if start_cond1 != "R":
        logger.warning("The phasepoint from [0+] from R is now on L side!")
        logger.warning("This should not happen!")
        status = "QLR"
        new_path0.status = status
        new_path1.status = status
        new_path1.append(shooting_point1)
        return False, [new_path0, new_path1], status

    logger.info("Continuing ropagation in [0+]")
    _, msg = engine1.propagate(new_path1, ens_set1, shooting_point1)

    # obtain the final full path in [0+]:
    # append 'new_path1' to the one-step path 'tmp_path1'
    new_path1 = paste_paths(
        tmp_path1.reverse(None, rev_v=False), new_path1, maxlen=maxlen1
    )

    # finished with path1, now do some extra checks
    if new_path1.length == maxlen1 and new_path1.get_end_point:
        new_path1.status = "FTX"
    elif new_path1.length < 3:
        new_path1.status = "FTS"
    elif new_path1.get_start_point(lambda0) != "L":
        new_path1.status = "0+R"
    else:
        new_path1.status = "ACC"

    if new_path1.status != "ACC":
        return False, [new_path0, new_path1], new_path1.status

    # everything checked out
    new_path0.weight = 1.0
    new_path1.weight = 1.0

    return True, [new_path0, new_path1], "ACC"


def ppretis_swap(
    picked: dict[int, Any],
    engines: dict[int, list[EngineBase]],
) -> tuple[bool, list[InfPath], str]:

    """Perform the RETIS swap between ``[i^+] <-> [(i+1)^+]`` PPTIS ensembles.

    First, it is verified if we can attempt a swap:
    the [i^+] path should start/end on the right of lambda_(i+1)
    AND the [(i+1)^+] path should start/end on the left of lambda_i

    This is labeled as follows:
    - allowed1: [i^+] path starts on the right of lambda_(i+1)
    - allowed2: [i^+] path ends   on the right of lambda_(i+1)
    - allowed3: [(i+1)^+] path starts on the left of lambda_i
    - allowed4: [(i+1)^+] path ends   on the left of lambda_i

    For now, only the case [allowed2 and allowed3] are considered for a swap,
    because this is easier to obey detailed balance.

    The RETIS swapping move for ensembles [i^+] and [(i+1)^+] requires some
    extra integration. Here we are generating new paths for [i^+] and
    [(i+1)^+] similarly to the moves done in the retis_swap_zero function.

    Parameters
    ----------
    ensembles : list of dictionaries of objects
        This is a list of the ensembles we are using in the RETIS method.
        It contains:

        * `path_ensemble`: object like :py:class:`.PathEnsemble`
          This is used for storing results for the simulation. It
          is also used for defining the interfaces for this simulation.
        * `system`: object like :py:class:`.System`
          System is used here since we need access to the temperature
          and to the particle list.
        * `order_function`: object like :py:class:`.OrderParameter`
          The class used for calculating the order parameter(s).
        * `engine`: object like :py:class:`.EngineBase`
          The engine to use for propagating a path.

    idx : int
        The index of the ensemble that will be swapped.
    settings : dict
        This dict contains the settings for the RETIS method.
    cycle : integer
        The current cycle number.

    Returns
    -------
    out : string
        The result of the swapping move.

    """
    ens_keys = list(picked.keys())
    ens_set0 = picked[ens_keys[0]]["ens"]
    ens_set1 = picked[ens_keys[1]]["ens"]
    engine0 = engines[ens_keys[0]][0]
    engine1 = engines[ens_keys[1]][0]
    old_path0 = picked[ens_keys[0]]["traj"]
    old_path1 = picked[ens_keys[1]]["traj"]
    maxlen0 = ens_set0["tis_set"]["maxlength"]
    maxlen1 = ens_set1["tis_set"]["maxlength"]
    lambda0 = ens_set0["interfaces"][-1]

    logger.info("Swapping: %s <-> %s",
                ens_set0["ens_name"],
                ens_set1["ens_name"])


    # print('move pp start', old_path0.path_number, old_path1.path_number)
    # WOUTER: This assumes that we have a [0-] and [0+] ensemble? If the
    #         zero_ensemble and flux ensemble are not present, this will
    #         fail. (Lacking any of them would result in nonsense?)
    # if idx == 0:
    #     return retis_swap_zero(ensembles, settings, cycle)

    # path_ensemble0 = ensembles[idx]['path_ensemble']
    # path_ensemble1 = ensembles[idx+1]['path_ensemble']
    # engine0, engine1 = ensembles[idx]['engine'], ensembles[idx+1]['engine']
    # maxlen0 = settings['ensemble'][idx]['tis']['maxlength']
    # maxlen1 = settings['ensemble'][idx+1]['tis']['maxlength']

    # 0. check if MD is allowed
    # ----------------------------
    # The first or last point of [i^+] should be
    # at the right (R) of the right border (interfaces[-1]) of the [i+] range.
    # So the path should start and/or end at the right.
    # path1 = backward, phasepoint[0], phasepoint[1], ...
    allowed1 = (old_path0.get_start_point(ens_set0["interfaces"][-1]) == 'R')
    # path1 = ..., phasepoint[-2], phasepoint[-1], forward
    allowed2 = (old_path0.get_end_point(ens_set0["interfaces"][-1]) == 'R')

    # The first or last point of [(i+1)^+] should be at the left (L) of
    # the left border (interfaces[0]) of the [(i+1)] range
    # So the path should start and/or end at the left
    # path0 = backward, phasepoint[0], phasepoint[1], ...
    allowed3 = (old_path1.get_start_point(ens_set1["interfaces"][0]) == 'L')
    # path0 = phasepoint[-2], phasepoint[-1], forward
    allowed4 = (old_path1.get_end_point(ens_set1["interfaces"][0]) == 'L')

    # Double check: this should be automatically true:
    # overlap in range with the other ensemble
    if allowed1:
        assert (old_path0.get_start_point(ens_set1["interfaces"][0]) == 'R')
    if allowed2:
        assert (old_path0.get_end_point(ens_set1["interfaces"][0]) == 'R')
    if allowed3:
        assert (old_path1.get_start_point(ens_set0["interfaces"][-1]) == 'L')
    if allowed4:
        assert (old_path1.get_end_point(ens_set0["interfaces"][-1]) == 'L')

    # we will only do allowed2 and allowed3, because of detailed balance
    allowed1 = False    # overwrite
    allowed4 = False
    # allowed = (allowed2 and allowed3)
    allowed = ((allowed1 or allowed2) and (allowed3 or allowed4))

    # deal with it when we are sure that we cannot swap
    if not allowed:
        status = 'NCR'
        accept = False
        logger.info('Swap was rejected. (%s)', status)

        # Make shallow copies:
        trial0 = old_path1.copy() # copy.copy(path_ensemble1.last_path)
        trial1 = old_path0.copy() # copy.copy(path_ensemble0.last_path)
        # trial0.set_move('s+')  # Came from left.
        # trial1.set_move('s-')  # Came from right.
        trial0.weight = 1.0
        trial1.weight = 1.0
        # path_ensemble0.add_path_data(trial0, status, cycle=cycle)
        # path_ensemble1.add_path_data(trial1, status, cycle=cycle)
        return accept, [trial0, trial1], status

    # 1. Generate path for [i^+] from [(i+1)^+]:
    # --------------------------------------------
    # We generate from the first point of the path in [(i+1)^+]:
    logger.debug('Swapping [i^+] <-> [(i+1)^+]')
    logger.debug('Creating path for [i^+]')

    # Start with copying parts of the path
    # reserve one spot for the "R" point, hence maxlen0-1
    # path_tmp0 = path_ensemble1.last_path.empty_path(maxlen=maxlen0-1)
    path_tmp0 = old_path1.empty_path(maxlen=maxlen0-1)

    if allowed3:
        trial = old_path1.phasepoints[1:]
    elif allowed4:
        trial = reversed(old_path1.phasepoints[:-1])

    # Adding more points from [(i+1)^+] at the end:
    # I will already have point[0]
    logger.debug('Copying points from [(i+1)^+]:')

    for phase_point in trial:
        # Here we make a copy of the phase point, as we will update
        # the configuration and append it to the new path:
        logger.debug('Point is %s', phase_point)
        # save for external engine
        path_tmp0.append(phase_point.copy())
        if phase_point.order[0] >= ens_set0["interfaces"][-1]:
            break  # stop adding phase points

    # The new path should have crossed the right interface, which we had
    # verified with allowed3 and allowed4
    # print("path_tmp0.phasepoints[-1].order[0]",path_tmp0.phasepoints[-1].order[0])
    # print("path_ensemble0.interfaces[-1]", path_ensemble0.interfaces[-1])
    # TODO
    # assert\
    # path_tmp0.phasepoints[-1].order[0] >= path_ensemble0.interfaces[-1],\
    #     "New path did not cross left interface. This shouldn't happen?!"
    # I had this
    # if allowed3:
    #    assert (path_ensemble1.last_path.get_start_point(
    #        path_ensemble0.interfaces[-1]) == 'L')
    # if allowed4:
    #    assert (path_ensemble1.last_path.get_end_point(
    #        path_ensemble0.interfaces[-1]) == 'L')
    #
    # Now, we will construct the rest of the path by propagation

    # Select phase point
    if allowed3:    # first phase point is L
        system = old_path1.phasepoints[0].copy()
    elif allowed4:    # last phase point is L
        system = old_path1.phasepoints[-1].copy()

    logger.debug('Initial point is: %s', system)
    #ensembles[idx]['system'] = system

    # Propagate it backward / forward in time:
    thislen = maxlen0 - path_tmp0.length - 1    # a bit shorter
    path_tmp = old_path1.empty_path(maxlen=thislen)
    logger.debug('Propagating for [i^+]')

    if allowed3:
        engine0.propagate(path_tmp, ens_set0, system, reverse=True)
    elif allowed4:
        engine0.propagate(path_tmp, ens_set0, system, reverse=False)

    # Finally, patch together
    if allowed3:
        path0 = paste_paths(path_tmp, path_tmp0,
                            overlap=False, maxlen=maxlen0)
    elif allowed4:
        path0 = paste_paths(path_tmp0, path_tmp,
                            overlap=False, maxlen=maxlen0)

    # acceptance
    path0.status = 'ACC'
    if allowed3:
        if path0.length >= maxlen0:
            path0.status = 'BTX'
        elif path0.length < 3:
            path0.status = 'BTS'

    elif allowed4:
        if path0.length >= maxlen0:
            path0.status = 'FTX'
        elif path0.length < 3:
            path0.status = 'FTS'

    assert not ('L' not in set(ens_set0["start_cond"]) and
                'L' in path0.check_interfaces(ens_set0["interfaces"])[:2]),\
        "Start condition L is not in path_ensemble0, while path0 crosses L.\
        This is not possible for PPRETIS?!"

    # 2. Generate path for [(i+1)^+] from [i^+]:
    # --------------------------------------------
    logger.debug('Creating path for [(i+1)^+] from [i^+]')

    # Start with copying parts of the path
    # reserve one spot for the "L" point, hence maxle10-1
    path_tmp0 = old_path1.empty_path(maxlen=maxlen1 - 1)

    if allowed1:
        trial = old_path0.phasepoints[1:]
    elif allowed2:
        trial = reversed(old_path0.phasepoints[:-1])

    # Adding more points from [i^+]:
    # I will already have point[0]
    logger.debug('Copying points from [i^+]:')
    for phase_point in trial:
        # Here we make a copy of the phase point, as we will update
        # the configuration and append it to the new path:
        logger.debug('Point is %s', phase_point)
        path_tmp0.append(phase_point.copy())
        if phase_point.order[0] <= ens_set1["interfaces"][0]:
            break  # stop adding phase points

    # The new path should have crossed the left interface, which we had
    # verified with allowed1 and allowed2
    # TODO
    # assert path_tmp0.phasepoints[-1].order[0]
    # < path_ensemble0.interfaces[0], \
    #    "New path did not cross left interface. This shouldn't happen?!"

    # Now, we will construct the rest of the path by propagation

    # select phase point
    if allowed1:    # first phase point is R
        system = old_path0.phasepoints[0].copy()
    elif allowed2:    # last phase point is R
        system = old_path0.phasepoints[-1].copy()

    logger.debug('Initial point is: %s', system)
    # ens_set1['system'] = system

    # Propagate it backward / forward in time:
    thelen = maxlen1 - 1 - path_tmp0.length
    path_tmp = old_path1.empty_path(maxlen=thelen)

    if allowed1:
        logger.debug('Propagating for [(i+1)^+]')
        engine1.propagate(path_tmp, ens_set1, system, reverse=True)
    elif allowed2:
        logger.debug('Propagating for [(i+1)^+]')
        engine1.propagate(path_tmp, ens_set1, system, reverse=False)

    # patch together
    if allowed1:
        path1 = paste_paths(path_tmp, path_tmp0,
                            overlap=False, maxlen=maxlen1)
    elif allowed2:
        path1 = paste_paths(path_tmp0, path_tmp,
                            overlap=False, maxlen=maxlen1)

    # acceptance
    path1.status = 'ACC'
    if allowed1:
        if path1.length >= maxlen1:
            path1.status = 'BTX'
        elif path1.length < 3:
            path1.status = 'BTS'
    elif allowed2:
        if path1.length >= maxlen1:
            path1.status = 'FTX'
        elif path1.length < 3:
            path1.status = 'FTS'

    logger.debug('Done with swap zero!')
    # 3. Do the bookkeeping
    # -----------------------

    # Final checks:
    # accept = path0.status == 'ACC' and path1.status == 'ACC'
    # status = 'ACC' if accept else (path0.status if path0.status != 'ACC' else
    #                               path1.status)
    accept = path0.status == 'ACC' and path1.status == 'ACC'
    status = 'ACC'
    if path0.status != 'ACC':
        status = path0.status
    if path1.status != 'ACC':
        status = path1.status

    # for trial, flag in ((path1, 's-'), (path0, 's+')):
    #     trial.status = status
    #     trial.weight = 1  # NB: No high acceptace allowed for ppretis.
        # trial.set_move(flag)
        # label the move as ld path, if it was ld in the previous cycle.
        # if path_ensemble.last_path.get_move() == 'ld':
        #     trial.set_move('ld')

    # if accept:
        # path0 = path_ensemble1.copy_path_to_generate(path0)
        # path1 = path_ensemble0.copy_path_to_generate(path1)
        # logger.debug("path0: "+str(path0)) RETURNS NULL
        # logger.debug("path1: "+str(path1)) RETURNS NULL
        # path_ensemble1.move_path_to_generate(path0)
        # path_ensemble0.move_path_to_generate(path1)
    # Split the loop to update objects first
    # for i, trial, path_ensemble in ((0, path0, path_ensemble0),   # s+
    #                                 (1, path1, path_ensemble1)):  # s-
    #     path_ensemble.add_path_data(trial, status, cycle=cycle)
    #     ens_set = settings['ensemble'][idx + i]
    #     if cycle % ens_set.get('output', {}).get('restart-file', 1) == 0:
    #         write_ensemble_restart(ensembles[idx+i], ens_set)

    # for name0, pat000 in (('a', path0), ('b', path1)):
    #     print('cheeze a', accept)
    #     configs = []
    #     for pp in pat000.phasepoints:
    #         configs.append(pp.config[0])
    #     print('cheeze b', set(configs))


    # for name0, pat000 in (('a', old_path0), ('b', old_path1), ('c', path0), ('d', path1)):
    #     print('cheeze a', pat000.status)
    #     for pp in pat000.phasepoints:
    #         print('wheel', name0, pp.config)
    #     print('cheeze b')


    return accept, (path0, path1), status


def staple(
    ens_set: Dict[str, Any],
    path: InfPath,
    engine: EngineBase,
    shooting_point: Optional[System] = None,
    start_cond: Tuple[str, ...] = ("L",),
) -> Tuple[bool, InfPath, str]:
    """Perform a shooting move.

    Args:
        ens_set: Settings for the ensemble.
        path: The initial path to shoot from.
        engine: The MD engine used for generating a new path.
        shooting_point: The selected shooting point. If None,
            it will be randomly selected here.
        start_cond: The starting condition for the ensemble as
            left ("L") or right ("R").

    Returns:
        A tuple containing:
            - True if the new path can be accepted, False otherwise.
            - The generated path.
            - A string representing the status of the path.

    """
    interfaces = ens_set["interfaces"]
    # the trial path we will generate
    trial_path = path.empty_path(maxlen=ens_set["tis_set"]["maxlength"])
    if shooting_point is None:
        shooting_point, idx, dek = prepare_shooting_point(
            path, ens_set["rgen"], engine, ens_set
        )
        kick = check_kick(
            shooting_point, interfaces, trial_path, ens_set["rgen"], dek
        )
    else:
        kick = True
        idx = getattr(shooting_point, "idx", 0)

    # Store info about this point, just in case we have to return
    # before completing a full new path:
    trial_path.generated = ("sh", shooting_point.order[0], idx, 0)
    trial_path.time_origin = path.time_origin + idx
    # We now check if the kick was OK or not:
    if not kick:
        return False, trial_path, trial_path.status
    # OK: kick was either aimless or it was accepted by Metropolis
    # We should now generate trajectories, but first check how long
    # it should be (if the path comes from a load, it is assumed to not
    # respect the detail balance anyway):
    if path.get_move() == "ld" or ens_set["tis_set"].get(
        "allowmaxlength", False
    ):
        maxlen = ens_set["tis_set"]["maxlength"]
    else:
        maxlen = min(
            int((path.length - 2) / ens_set["rgen"].random()) + 2,
            ens_set["tis_set"]["maxlength"],
        )
    # Since the forward path must be at least one step, the maximum
    # length for the backward path is maxlen-1.

    # Generate the backward path:
    path_back = path.empty_path(maxlen=maxlen - 1)
    # todo this inputs are a mess
    # Set ensemble state to the selected shooting point:
    # ensemble['system'] = shooting_point.copy()
    shpt_copy = shooting_point.copy()
    if not shoot_backwards(
        path_back, trial_path, shpt_copy, ens_set, engine, start_cond
    ):
        return False, trial_path, trial_path.status

    # Generate forward path:
    # Note that the length of the forward path is adjusted to
    # account for the fact that it shares a point with the backward
    # path (i.e. the shooting point). The duplicate point is just
    # counted once when the paths are merged by the method
    # `paste_paths` by setting `overlap=True`.
    path_forw = path.empty_path(maxlen=(maxlen - path_back.length + 1))
    logger.debug("Propagating forwards for shooting move...")
    # Set ensemble state to the selected shooting point:
    # change the system state.
    # ensemble['system'] = shooting_point.copy()
    shpt_copy = shooting_point.copy()
    success_forw, _ = engine.propagate(
        path_forw, ens_set, shpt_copy, reverse=False
    )
    path_forw.time_origin = trial_path.time_origin
    # Now, the forward propagation could have failed by exceeding the
    # maximum length for the forward path. However, it could also fail
    # when we paste together so that the length is larger than the
    # allowed maximum. We paste first and ask later:
    trial_path = paste_paths(
        path_back,
        path_forw,
        overlap=True,
        maxlen=ens_set["tis_set"]["maxlength"],
    )

    # Also update information about the shooting:
    trial_path.generated = (
        "sh",
        shooting_point.order[0],
        idx,
        path_back.length - 1,
    )
    if not success_forw:
        trial_path.status = "FTL"
        # If we reached this point, the backward path was successful,
        # but the forward was not. For the case where the forward was
        # also successful, the length of the trial path cannot exceed
        # the maximum length given in the TIS settings. Thus we only
        # need to check this here, i.e. when given that the backward
        # was successful and the forward not:
        if trial_path.length == ens_set["tis_set"]["maxlength"]:
            trial_path.status = "FTX"  # exceeds "memory".
        return False, trial_path, trial_path.status

    trial_path.weight = 1.0

    # Deal with the rejections for path properties.
    # Make sure we did not hit the left interface on {0-}
    # Which is the only ensemble that allows paths starting in R
    if (
        "L" not in set(start_cond)
        and "L" in trial_path.check_interfaces(interfaces)[:2]
    ):
        trial_path.status = "0-L"
        return False, trial_path, trial_path.status

    # Last check - Did we cross the middle interface?
    if "must_cross_M" in ens_set and ens_set["must_cross_M"]: # not for [0-]

        # detect if: minorder < middle interf <= maxorder
        if not trial_path.check_interfaces(interfaces)[-1][1]:
            trial_path.status = 'NCR'
            return False, trial_path, trial_path.status

        else:
            # if we do cross middle interface
            # path still has to come from left or right
            # so we need cross[0] or cross[2] to be true
            if not (trial_path.check_interfaces(interfaces)[-1][0] or
                    trial_path.check_interfaces(interfaces)[-1][2]):
                trial_path.status = 'NCR'
                return False, trial_path, trial_path.status



    # Don't do this for paths that can start everywhere
    start_cond = ens_set.get("start_cond", start_cond)
    if set(("R", "L")) == set(start_cond):
        pass
    elif not trial_path.check_interfaces(interfaces)[-1][1]:
        # No, we did not cross the middle interface:
        trial_path.status = "NCR"
        return False, trial_path, trial_path.status

    trial_path.status = "ACC"

    return True, trial_path, trial_path.status


