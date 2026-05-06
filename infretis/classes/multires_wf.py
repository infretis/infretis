"""
Multiresolution Wire Fencing move ("mwf").

This module implements a new Monte Carlo move very close to the classic
Wire Fencing (WF) move, but with a multiresolution procedure that alternates
between low- and high-resolution subpath generation as specified in the
algorithm steps (1–15).

It is designed to plug into the existing TIS code alongside `wire_fencing`
(see infretis.core.tis.wire_fencing). The same High-Acceptance weight
scheme as WF is used.

Configuration (TIS settings `ens_set["tis_set"]`):
    mwf_nsubpath: int                # N_subpath (default: 3)
    n_jumps: int                     # N_subset, reuses existing WF parameter (default: 2)
    mwf_subcycle_small: int          # small N_subcycle for high-res subpaths (default: max(1, engine.subcycles // 5))
    interface_cap: float             # optional, same meaning as in WF

Note:
    - Low-res subpaths automatically use engine.subcycles (like standard WF)
    - If a high-res segment is rejected it is also immediately deleted (to save disk space)

Notes:
- Any highres subpath with length L == 3 (three phase points) is rejected.

Integration with existing code:
- Register the move key "mwf" in `tis.select_shoot`.
- Ensure weighting treats "mwf" like "wf" (either by patching
  `compute_weight`/`calc_cv_vector` to recognize "mwf", or by the wrapper
  here which computes WF-weights internally).

"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import glob

from infretis.classes import system
from infretis.classes.path import paste_paths

# Type checking imports are local to avoid runtime deps
try:  # pragma: no cover
    from infretis.classes.path import Path as InfPath
    from infretis.classes.system import System
    from infretis.classes.engines.enginebase import EngineBase
except Exception:  # pragma: no cover
    InfPath = object  # type: ignore
    System = object  # type: ignore
    EngineBase = object  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# We reuse WF helpers from tis.py at runtime (import inside function)


@dataclass
class _MWFSettings:
    nsubpath: int
    nsubset: int  # from n_jumps
    subcycle_small: int  # only new parameter
    subcycle_large: int  # from engine.subcycles


@contextmanager
def _temporary_subcycles(engine: EngineBase, new_subcycles: int):
    old = getattr(engine, "subcycles", None)
    try:
        if new_subcycles is not None and old is not None:
            logger.debug(
                f"MWF: Changing engine subcycles from {old} to {new_subcycles}"
            )
            engine.subcycles = int(max(1, new_subcycles))
            logger.debug(
                f"MWF: Engine subcycles now set to {engine.subcycles}"
            )
        yield
    finally:
        if old is not None:
            logger.debug(f"MWF: Restoring engine subcycles to {old}")
            engine.subcycles = old


def _read_mwf_settings(
    ens_set: Dict[str, Any], engine: EngineBase
) -> _MWFSettings:
    tis = ens_set["tis_set"]
    # Use existing TIS parameters where possible
    large = getattr(engine, "subcycles", 10)  # from engine, like standard WF
    small_default = max(1, large // 5)
    return _MWFSettings(
        nsubpath=int(tis.get("mwf_nsubpath", 3)),
        nsubset=int(tis.get("n_jumps", 2)),  # reuse existing WF parameter
        subcycle_small=int(tis.get("mwf_subcycle_small", small_default)),
        subcycle_large=int(large),
    )


def _random_internal_indices(path: InfPath, rgen, k: int) -> List[int]:
    """Up to k distinct interior indices of `path`, in shuffled order."""
    n = len(path.phasepoints)
    if n <= 2:
        return []
    available = list(range(1, n - 1))
    rgen.shuffle(available)
    return available[: max(0, min(k, len(available)))]


def _pick_shooting_idx(
    path: InfPath, rgen, is_highres: bool, queue: List[int]
) -> Tuple[int, List[int], str]:
    """Pick a shooting index according to resolution.

    highres: pop the head of `queue`. Empty queue → status "EQU" (caller
        treats this as a failed subpath; the lowres branch never returns EQU).
    lowres: uniform among interior indices.
    """
    if is_highres:
        if not queue:
            return 0, queue, "EQU"
        idx = queue.pop(0)
        return int(idx), queue, "OK"
    n = len(path.phasepoints)
    if n <= 2:
        return 0, queue, "OK"
    return int(rgen.integers(1, n - 1)), queue, "OK"


def _generate_segment_using_shoot(
    si: InfPath,
    idx: int,
    ens_set_sub: Dict[str, Any],
    engine: EngineBase,
    start_cond: Tuple[str, ...],
) -> Tuple[bool, InfPath, str]:
    """Generate a segment from `si` using an MWF-chosen shooting index.

    The MWF caller owns the shooting-point policy, so we build an explicit
    shooting point from `si.phasepoints[idx]` and pass it to `shoot()`.
    Velocity randomization (modify_velocities + check_kick) is performed here
    to preserve standard MC kick semantics, which `shoot()` would otherwise
    skip on the explicit-point branch.
    """
    from infretis.core.tis import shoot, check_kick

    maxlen = ens_set_sub["tis_set"]["maxlength"]
    if idx <= 0 or idx >= len(si.phasepoints) - 1:
        return False, si.empty_path(maxlen=maxlen), "IDX"

    shooting_point = si.phasepoints[idx].copy()
    shooting_point.idx = idx

    dek, _ = engine.modify_velocities(shooting_point, ens_set_sub["tis_set"])
    shooting_point.order = engine.calculate_order(shooting_point)

    scratch = si.empty_path(maxlen=maxlen)
    if not check_kick(
        shooting_point,
        ens_set_sub["interfaces"],
        scratch,
        ens_set_sub["rgen"],
        dek,
    ):
        return False, scratch, scratch.status

    return shoot(ens_set_sub, si, engine, shooting_point, start_cond)


def multires_wire_fencing(
    ens_set: Dict[str, Any],
    trial_path: InfPath,
    engine: EngineBase,
    start_cond: Tuple[str, ...] = ("L",),
) -> Tuple[bool, InfPath, str]:
    # Import WF helpers lazily to avoid circular imports
    from infretis.core.tis import (
        wirefence_weight_and_pick,
        extender,
        subt_acceptance,
        check_kick,
    )

    interfaces_full = ens_set["interfaces"]  # [lambda_A, lambda_i, lambda_B]
    lambda_cap = ens_set["tis_set"].get("interface_cap", interfaces_full[2])
    wf_int = [
        interfaces_full[1],
        interfaces_full[1],
        lambda_cap,
    ]  # [i, i, cap]

    # Step 1: pick s0 between lambda_i and lambda_cap (exclude cap-cap)
    n_frames, s0 = wirefence_weight_and_pick(
        trial_path, wf_int[0], wf_int[2], return_seg=True, ens_set=ens_set
    )
    if n_frames == 0:
        logger.warning("MWF: No valid frames between i and cap; aborting")
        return False, trial_path, "NSG"

    # Prepare a sub-ensemble confined to [lambda_i, lambda_cap]
    sub_ens = {
        "interfaces": wf_int,
        "rgen": ens_set["rgen"],
        "ens_name": ens_set["ens_name"],
        "start_cond": ("L", "R"),  # subpaths may start anywhere
        "tis_set": dict(ens_set["tis_set"]),
    }
    sub_ens["tis_set"]["allowmaxlength"] = True
    sub_ens["tis_set"]["maxlength"] = ens_set["tis_set"]["maxlength"]

    # Read multires settings
    mwf = _read_mwf_settings(ens_set, engine)

    # Counters and state (Step 2)
    countset = 1
    succ_sets = 0
    total_succ_subpaths = 0  # Track total successful subpaths across all sets
    set_success_history = (
        []
    )  # Track which sets succeeded: [True, False, True, ...]

    # Check if s0 is valid for shooting
    if len(s0.phasepoints) <= 2:
        # s0 too short, fail immediately
        trial_path.status = "NSG"
        return False, trial_path, trial_path.status

    # For carrying the last *accepted* subpath
    si = s0.copy()
    last_acc_si = si.copy()
    logger.info(
        f"Starting MWF with s0: ordermax={si.ordermax[0]:.6f}, length={si.length}"
    )

    # Iterate over sets (Step 12)
    while True:
        logger.info(f"MWF: Starting set {countset}/{mwf.nsubset}")
        i = 0
        set_rejected = False
        si = si.copy()  # start this set from current s0
        last_acc_in_set = si.copy()
        subpaths_in_set = 0

        # Build a fresh highres shooting queue from the committed s0.
        # The queue is strictly set-local: rebuilt at each set's top, never
        # carried across sets. Length is mwf.nsubpath - 1 because the final
        # subpath is lowres and never consumes from the queue.
        shooting_queue = _random_internal_indices(
            s0, sub_ens["rgen"], mwf.nsubpath - 1
        )
        last_acc_queue = list(shooting_queue)

        # Generate up to Nsubpath subpaths
        while i < mwf.nsubpath:
            # Resolution of the subpath we are about to attempt
            # (pre-increment view of i; last subpath is lowres).
            is_highres = (i + 1) < mwf.nsubpath

            # Step 3: pick shooting index per resolution
            idx, shooting_queue, pick_status = _pick_shooting_idx(
                si, sub_ens["rgen"], is_highres, shooting_queue
            )

            if pick_status == "EQU":
                # Highres queue empty — treat as a failed subpath at slot i+1
                # so the standard terminal/non-terminal rejection rules fire.
                # EQU cannot occur on the lowres pick.
                i += 1
                gen_success = False
                seg = si.empty_path(maxlen=sub_ens["tis_set"]["maxlength"])
                status = "EQU"
                subcycles = mwf.subcycle_small
                logger.info(
                    f"Set {countset}: highres queue empty at subpath {i} (EQU)"
                )
            elif (
                len(si.phasepoints) <= 2
                or idx <= 0
                or idx >= len(si.phasepoints) - 1
            ):
                gen_success = False
                seg = si.empty_path(maxlen=sub_ens["tis_set"]["maxlength"])
                status = "SHP"  # Short path
                subcycles = mwf.subcycle_small
            else:
                # Step 5: i = i + 1
                i += 1

                # Step 6: last subpath uses lowres, others use highres
                if i == mwf.nsubpath:
                    subcycles = mwf.subcycle_large
                else:
                    subcycles = mwf.subcycle_small

                logger.debug(
                    f"Subpath {i}: Shooting from ordermax={si.ordermax[0]:.6f}, subcycles={subcycles}, idx={idx}"
                )
                with _temporary_subcycles(engine, subcycles):
                    gen_success, seg, status = _generate_segment_using_shoot(
                        si,
                        idx,
                        sub_ens,
                        engine,
                        start_cond=("L", "R"),
                    )
                ordermax_str = (
                    f"{seg.ordermax[0]:.6f}" if seg.length > 0 else "empty"
                )
                logger.debug(
                    f"Subpath {i}: Generated path with ordermax={ordermax_str}, length={seg.length}, status={status}"
                )
                subpaths_in_set += 1

            # Step 8: rejection checks specific to MWF
            if gen_success:
                start_side, end_side, _, _ = seg.check_interfaces(wf_int)
                is_cap_cap = start_side == "R" and end_side == "R"
                is_len3_highres = (
                    subcycles == mwf.subcycle_small and seg.length == 3
                )
                if is_cap_cap or is_len3_highres:
                    gen_success = False
                    status = "CCP" if is_cap_cap else "L3H"

            if not gen_success:
                # Step 9: Reject subpath
                for rej_seg_trajname in seg.adress:
                    os.remove(rej_seg_trajname)
                    logger.info(
                        f"Removing {rej_seg_trajname} because of status {status}"
                    )

                if i == 1 or i == mwf.nsubpath:
                    set_rejected = True
                    si = s0.copy()
                    logger.info(
                        f"Set {countset}: First/last subpath {i} failed ({status}), rejecting entire set"
                    )
                    break
                else:
                    # Revert path AND queue to last accepted state in this set
                    si = last_acc_in_set.copy()
                    shooting_queue = list(last_acc_queue)
                    logger.debug(
                        f"Subpath {i} failed ({status}), reverting to last accepted subpath in set {countset}"
                    )
                    continue
            else:
                # Step 10: Accept subpath
                total_succ_subpaths += 1
                si = seg.copy()
                last_acc_in_set = seg.copy()

                # Refresh highres queue from the just-generated segment for
                # any remaining highres slots (i is post-increment here, so
                # remaining highres slots = (nsubpath - 1) - i).
                remaining_highres = (mwf.nsubpath - 1) - i
                if remaining_highres > 0:
                    shooting_queue = _random_internal_indices(
                        seg, sub_ens["rgen"], remaining_highres
                    )
                else:
                    shooting_queue = []
                last_acc_queue = list(shooting_queue)

                logger.debug(f"Subpath {i} accepted in set {countset}")
                if i == mwf.nsubpath:
                    logger.debug(
                        f"Set {countset} completed successfully - final subpath accepted"
                    )
                    break

        # Step 12: Record set outcome (commit before finalization check)
        set_succeeded = not set_rejected
        set_success_history.append(set_succeeded)

        if set_succeeded:
            succ_sets += 1
            last_acc_si = si.copy()
            s0 = si.copy()
            logger.info(
                f"Set {countset} completed successfully with {subpaths_in_set} subpaths generated, final path length={si.length}"
            )

        if countset == mwf.nsubset:
            if succ_sets == 0:
                logger.info(
                    f"MWF move rejected: {succ_sets}/{mwf.nsubset} sets completed, {total_succ_subpaths} total subpaths accepted"
                )
                trial_path.status = "NSG"
                return False, trial_path, trial_path.status

            path_to_extend = last_acc_si
            logger.info(
                f"Extending last accepted lowres subpath: length={path_to_extend.length}, ordermax={path_to_extend.ordermax[0]:.6f}"
            )

            ok, full_path, _ = extender(
                path_to_extend, engine, ens_set, start_cond
            )
            if not ok:
                return False, full_path, full_path.status

            success, accepted = subt_acceptance(
                full_path, ens_set, engine, start_cond
            )
            accepted.generated = ("mwf", 9001, succ_sets, accepted.length)
            if not success:
                return False, accepted, accepted.status

            logger.info(
                f"MWF move accepted: {succ_sets}/{mwf.nsubset} sets completed, {total_succ_subpaths} total subpaths accepted, final path length={accepted.length}"
            )
            return True, accepted, accepted.status

        countset += 1
        logger.debug(
            f"Moving to next set {countset}/{mwf.nsubset}, set_rejected={set_rejected}"
        )
