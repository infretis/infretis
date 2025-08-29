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
    - This reduces MWF-specific parameters from 4 to 2

Notes:
- lambda_A and lambda_B are the *outer* interfaces.
- lambda_cap is the right bound used for subpaths, like in WF.
- "lambda_cap–lambda_cap" subpaths are rejected.
- Any highres subpath with length L == 3 (three phase points) is rejected.
- Subpath starts/ends relative to [lambda_i, lambda_cap] are tested using
  the same interface classification utilities as WF.

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
    System = object   # type: ignore
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
    """Temporarily set `engine.subcycles` and restore on exit."""
    old = getattr(engine, "subcycles", None)
    try:
        if new_subcycles is not None and old is not None:
            engine.subcycles = int(max(1, new_subcycles))
        yield
    finally:
        if old is not None:
            engine.subcycles = old


def _read_mwf_settings(ens_set: Dict[str, Any], engine: EngineBase) -> _MWFSettings:
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
    """Pick up to k distinct internal indices (exclude endpoints)."""
    n = len(path.phasepoints)
    if n <= 2:
        return []
    start, stop = 1, n - 1  # [1, n-2]
    available = list(range(start, stop))
    rgen.shuffle(available)
    return available[: max(0, min(k, len(available)))]


def _pick_shooting_idx(path: InfPath, rgen, resolution: str, queue: List[int]) -> Tuple[int, List[int]]:
    """Pick a shooting index according to resolution.

    lowres: uniform among internal points
    highres: pop from the head of `queue`; if empty, fallback to uniform
    """
    if resolution == "highres" and queue:
        idx = queue.pop(0)
        return idx, queue
    # lowres or empty queue fallback
    n = len(path.phasepoints)
    if n <= 2:
        return 0, queue
    idx = rgen.integers(1, n - 1)
    return int(idx), queue


def _generate_segment_from_point(
    si: InfPath,
    idx: int,
    ens_set_sub: Dict[str, Any],
    engine: EngineBase,
    start_cond: Tuple[str, ...],
    shooting_point: Optional[System] = None
) -> Tuple[bool, InfPath, str]:
    """Clone WF's `shoot` but with an explicit shooting point index.

    Integrates backward and forward under `ens_set_sub['interfaces']`.
    """
    from infretis.core.tis import shoot_backwards  # reuse helpers

    interfaces = ens_set_sub["interfaces"]
    maxlen = ens_set_sub["tis_set"]["maxlength"]

    if shooting_point is None: 
        shooting_point = si.phasepoints[idx].copy()

    if idx <= 0 or idx >= len(si.phasepoints) - 1:
        return False, si.empty_path(maxlen=maxlen), "IDX"

    trial_path = si.empty_path(maxlen=maxlen)
    trial_path.generated = ("mwf-sh", shooting_point.order[0], idx, 0)
    trial_path.time_origin = si.time_origin + idx
    # Ensure intermediate mwf-sh paths never get path numbers (prevent file saving)
    trial_path.path_number = None

    # Backward
    path_back = si.empty_path(maxlen=maxlen - 1)
    shpt_copy = shooting_point.copy()
    if not shoot_backwards(path_back, trial_path, shpt_copy, ens_set_sub, engine, start_cond):
        return False, trial_path, trial_path.status

    # Forward
    path_forw = si.empty_path(maxlen=(maxlen - path_back.length + 1))
    shpt_copy = shooting_point.copy()
    success_forw, _ = engine.propagate(path_forw, ens_set_sub, shpt_copy, reverse=False)
    path_forw.time_origin = trial_path.time_origin

    trial_path = paste_paths(
        path_back, path_forw, overlap=True, maxlen=maxlen
    )

    trial_path.generated = ("mwf-sh", shooting_point.order[0], idx, path_back.length - 1)
    # Ensure intermediate mwf-sh paths never get path numbers (prevent file saving)
    trial_path.path_number = None

    if not success_forw:
        trial_path.status = "FTL"
        if trial_path.length == maxlen:
            trial_path.status = "FTX"
        return False, trial_path, trial_path.status

    # Reject if wrong lambda0 crossing rules apply within sub-interfaces:
    # (We allow start anywhere for subpath generation)
    # Ensure we crossed the middle sub-interface:
    if not trial_path.check_interfaces(interfaces)[-1][1]:
        trial_path.status = "NCR"
        return False, trial_path, trial_path.status

    trial_path.status = "ACC"
    return True, trial_path, trial_path.status


def multires_wire_fencing(
    ens_set: Dict[str, Any],
    trial_path: InfPath,
    engine: EngineBase,
    start_cond: Tuple[str, ...] = ("L",),
) -> Tuple[bool, InfPath, str]:
    """Perform the Multiresolution Wire Fencing (MWF) move.

    Returns (accept, path, status) just like `wire_fencing`.
    """
    # Import WF helpers lazily to avoid circular imports
    from infretis.core.tis import (
        wirefence_weight_and_pick,
        extender,
        subt_acceptance,
        check_kick,
    )

    interfaces_full = ens_set["interfaces"]  # [lambda_A, lambda_i, lambda_B]
    lambda_cap = ens_set["tis_set"].get("interface_cap", interfaces_full[2])
    wf_int = [interfaces_full[1], interfaces_full[1], lambda_cap]  # [i, i, cap]

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
    current_resolution = "lowres"  # Step 2

    # For carrying the last *accepted* subpath and its queue
    si = s0.copy()
    last_acc_si = si.copy()
    last_acc_queue: List[int] = []

    # Iterate over sets (Step 12)
    while True:
        i = 0
        set_rejected = False
        shooting_queue: List[int] = []
        si = si.copy()  # start this set from current s0

        # Generate up to Nsubpath subpaths
        while i < mwf.nsubpath:
            # Step 3: pick shooting point by resolution rule
            idx, shooting_queue = _pick_shooting_idx(si, sub_ens["rgen"], current_resolution, shooting_queue)

            # Step 4: new velocities (Maxwell–Boltzmann)
            shpt = si.phasepoints[idx].copy()
            _, _ = engine.modify_velocities(shpt, sub_ens["tis_set"])  # dek not used for now
            shpt.order = engine.calculate_order(shpt)

            # Sanity check kick (same as WF)
            if not check_kick(shpt, wf_int, si.empty_path(maxlen=sub_ens["tis_set"]["maxlength"]), sub_ens["rgen"], 0.0):
                # KOB: treat as immediate subpath rejection pathway
                gen_success = False
                seg = si.empty_path(maxlen=sub_ens["tis_set"]["maxlength"])
                status = "KOB"
            else:
                # Step 5: i = i + 1
                i += 1

                # Step 6: decide resolution used for *integration* (current resolution determines subcycles)
                # Use current resolution for subcycle choice: lowres = large subcycles, highres = small subcycles
                subcycles = mwf.subcycle_large if current_resolution == "lowres" else mwf.subcycle_small
                
                # Decide next resolution for subsequent shooting point selection
                next_resolution = "highres" if i < mwf.nsubpath else "lowres"

                # Step 7: generate si by integrating to [i, cap] using chosen resolution
                with _temporary_subcycles(engine, subcycles):
                    gen_success, seg, status = _generate_segment_from_point(
                        si, idx, sub_ens, engine, start_cond=("L", "R"), shooting_point=shpt
                    )

                # After generation, build/update the shooting list for *next* pick
                # Store Nsubpath - i random internal indices
                k = max(0, mwf.nsubpath - i)
                shooting_queue = _random_internal_indices(seg, sub_ens["rgen"], k)

            # Step 8: rejection checks specific to MWF
            if gen_success:
                start_side, end_side, _, _ = seg.check_interfaces(wf_int)
                is_cap_cap = (start_side == "R" and end_side == "R")
                is_len3_highres = (current_resolution == "highres" and seg.length == 3)
                if is_cap_cap or is_len3_highres:
                    gen_success = False
                    status = "CCP" if is_cap_cap else "L3H"

            if not gen_success:
                # Step 9: Reject subpath
                if i == 1 or i == mwf.nsubpath:
                    # Step 11: Reject set, restart from last accepted lowres path s0
                    set_rejected = True
                    si = s0.copy()
                    si.path_number = None
                    break
                else:
                    # Replace with last accepted subpath and repeat step 3 (i unchanged)
                    si = last_acc_si.copy()
                    si.path_number = None
                    shooting_queue = list(last_acc_queue)
                    continue
            else:
                # Step 10: Accept subpath
                last_acc_si = seg.copy()
                # Ensure copied intermediate paths don't retain path numbers
                last_acc_si.path_number = None
                last_acc_queue = list(shooting_queue)
                si = seg.copy()
                si.path_number = None

                if i == mwf.nsubpath:
                    # proceed to evaluate set (Step 12)
                    break

            # On next loop, picking uses *current* resolution variable
            current_resolution = next_resolution

        # Step 11 handled inside loop via set_rejected flag
        # Step 12: Evaluate set
        if countset == mwf.nsubset:
            # Determine which path to extend: last accepted subpath or fallback s0
            path_to_extend = s0 if set_rejected else last_acc_si
            # Step 13: extend last accepted subpath to full A/B using large subcycles
            # Use standard extender but with large subcycles and full interface set
            extension_ens = dict(ens_set)
            extension_ens["interfaces"] = interfaces_full  # Ensure full A/B interfaces for extension
            with _temporary_subcycles(engine, mwf.subcycle_large):
                ok, full_path, _ = extender(path_to_extend, engine, extension_ens, start_cond)
            if not ok:
                # Could be FTX etc. Pass through status from extender in full_path
                return False, full_path, full_path.status

            # Orientation and weight (use the same scheme as WF)
            # The subt_acceptance function already handles "mwf" moves properly
            success, accepted = subt_acceptance(full_path, ens_set, engine, start_cond)
            accepted.generated = ("mwf", 9001, mwf.nsubpath, accepted.length)
            if not success:
                return False, accepted, accepted.status

            # Extra step 13 rules on terminal types
            left, _, right = interfaces_full
            start_side = accepted.get_start_point(left, right)
            end_side = accepted.get_end_point(left, right)
            if start_side == "R" and end_side == "R":
                # Step 14: Reject WF move
                trial_path.status = "BBB"
                return False, trial_path, trial_path.status
            if start_side == "R" and end_side == "L":
                # Reverse to make it A->B
                accepted = accepted.reverse(engine.order_function)

            accepted.status = "ACC"
            return True, accepted, accepted.status

        # Move to next set
        countset += 1
        if not set_rejected:
            s0 = si.copy()  # last accepted lowres (after successful set) becomes new s0
            s0.path_number = None
        current_resolution = "lowres"
        # Loop back to new set