"""Epoch-wise per-ensemble subtrajectory-count controller for WF/MWF moves.

Two controller modes are supported, selected via config["simulation"]["epoch_mode"]:

  ``global_step`` (default)
      Fires for all targeted ensembles simultaneously when ``cstep`` is a
      positive multiple of ``epoch_size``.  This reproduces the original
      behaviour.

  ``per_ensemble_moves``
      Fires independently for each targeted ensemble when that ensemble's
      completed-move counter reaches a multiple of ``epoch_move_k``.  The
      counter tracks either every attempted move or only accepted moves,
      depending on ``epoch_count`` (default: ``"attempted"``).

In both modes the n_jumps value for each targeted ensemble is cycled through
the schedule defined in ``epoch_nsubpath_vals``.
"""

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _k_list(sim: dict) -> list:
    """Normalise epoch_move_k to a per-target list of positive ints."""
    n = len(sim.get("epoch_nsubpath_ens", []))
    k_raw = sim.get("epoch_move_k", 1)
    if isinstance(k_raw, list):
        return [int(k) for k in k_raw]
    return [int(k_raw)] * n


def _update_ensemble_n_jumps(
    state, ens_i: int, vals: list, epoch_idx: int
) -> None:
    """Set ensemble ens_i's n_jumps to the scheduled value for epoch_idx."""
    if ens_i in state.ensembles and vals:
        new_val = int(vals[epoch_idx % len(vals)])
        old_val = state.ensembles[ens_i]["tis_set"].get("n_jumps")
        state.ensembles[ens_i]["tis_set"]["n_jumps"] = new_val
        logger.info(
            "Epoch %d: ensemble %03d n_jumps %s -> %d",
            epoch_idx, ens_i, old_val, new_val,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_epoch_ctrl(config: dict, state=None) -> None:
    """Validate epoch controller config keys.

    Args:
        config: full simulation config dict.
        state: unused; reserved for future live-state checks.

    Raises:
        ValueError: on any configuration inconsistency.
    """
    sim = config["simulation"]
    n_ens = len(sim["interfaces"])
    sh_moves = sim["shooting_moves"]
    n_sh_moves = len(sh_moves)

    ens_targets = sim.get("epoch_nsubpath_ens", [])
    val_schedules = sim.get("epoch_nsubpath_vals", [])

    if len(ens_targets) != len(val_schedules):
        raise ValueError(
            f"epoch_nsubpath_ens has {len(ens_targets)} entries but "
            f"epoch_nsubpath_vals has {len(val_schedules)} entries; "
            "they must be the same length."
        )
    for ens_i in ens_targets:
        if not (0 <= ens_i < n_ens):
            raise ValueError(
                f"epoch_nsubpath_ens contains invalid ensemble index {ens_i}; "
                f"valid range is 0 \u2026 {n_ens - 1}."
            )
        ens_move = sh_moves[ens_i] if ens_i < n_sh_moves else None
        if ens_move not in ("wf", "mwf"):
            raise ValueError(
                f"epoch_nsubpath_ens targets ensemble {ens_i} which uses "
                f"move '{ens_move}'; only 'wf' and 'mwf' ensembles support "
                "the n_jumps parameter."
            )

    mode = sim.get("epoch_mode", "global_step")
    if mode not in ("global_step", "per_ensemble_moves"):
        raise ValueError(
            f"epoch_mode '{mode}' is not recognised; "
            "valid options are 'global_step' and 'per_ensemble_moves'."
        )

    if mode == "per_ensemble_moves" and ens_targets:
        k_raw = sim.get("epoch_move_k")
        if k_raw is None:
            raise ValueError(
                "epoch_move_k must be set when "
                "epoch_mode='per_ensemble_moves'."
            )
        if isinstance(k_raw, list):
            if len(k_raw) != len(ens_targets):
                raise ValueError(
                    f"epoch_move_k has {len(k_raw)} entries but "
                    f"epoch_nsubpath_ens has {len(ens_targets)}; "
                    "they must be the same length."
                )
            if any(int(k) <= 0 for k in k_raw):
                raise ValueError(
                    "All epoch_move_k values must be positive integers."
                )
        elif int(k_raw) <= 0:
            raise ValueError("epoch_move_k must be a positive integer.")

    count_mode = sim.get("epoch_count", "attempted")
    if count_mode not in ("attempted", "accepted"):
        raise ValueError(
            f"epoch_count '{count_mode}' is not recognised; "
            "valid options are 'attempted' and 'accepted'."
        )


def apply_epoch_ctrl(state, cstep: int) -> None:
    """Apply epoch-wise n_jumps updates to targeted ensembles.

    Dispatches on config["simulation"]["epoch_mode"]:

    * ``global_step``: all targets fire together when ``cstep`` is a positive
      multiple of ``epoch_size``.
    * ``per_ensemble_moves``: each target fires independently when its entry
      in ``state.ensemble_move_counts`` is a positive multiple of the
      corresponding ``epoch_move_k`` value.

    Args:
        state: REPEX_state instance whose ensembles are mutated in-place.
        cstep: current completed-move step count (used in global_step mode).
    """
    sim = state.config["simulation"]
    ens_targets = sim.get("epoch_nsubpath_ens", [])
    val_schedules = sim.get("epoch_nsubpath_vals", [])
    if not ens_targets:
        return

    mode = sim.get("epoch_mode", "global_step")

    if mode == "global_step":
        epoch_size = sim.get("epoch_size", 0)
        if not (epoch_size > 0 and cstep > 0 and cstep % epoch_size == 0):
            return
        epoch_idx = cstep // epoch_size
        for ens_i, vals in zip(ens_targets, val_schedules):
            _update_ensemble_n_jumps(state, ens_i, vals, epoch_idx)

    elif mode == "per_ensemble_moves":
        for ens_i, vals, k in zip(ens_targets, val_schedules, _k_list(sim)):
            count = state.ensemble_move_counts.get(ens_i, 0)
            if count > 0 and count % k == 0:
                epoch_idx = count // k
                _update_ensemble_n_jumps(state, ens_i, vals, epoch_idx)


def mirror_epoch_ctrl(state, config: dict) -> None:
    """Mirror per-ensemble n_jumps, mwf_nsubpath, and move counts into config.

    Writes ``ensemble_nsubpath``, ``ensemble_mwf_nsubpath`` (into
    ``config["simulation"]``) and ``ensemble_move_counts`` (into
    ``config["current"]``) so that ``write_toml()`` serializes the current
    live state and ``initiate_ensembles()`` restores it exactly on restart.

    Args:
        state: REPEX_state instance to read current values from.
        config: full simulation config dict; mutated in-place.
    """
    global_tis = config["simulation"]["tis_set"]
    ens_keys = sorted(state.ensembles.keys())

    config["simulation"]["ensemble_nsubpath"] = [
        state.ensembles[i]["tis_set"].get(
            "n_jumps", global_tis.get("n_jumps", 2)
        )
        for i in ens_keys
    ]
    config["simulation"]["ensemble_mwf_nsubpath"] = [
        state.ensembles[i]["tis_set"].get(
            "mwf_nsubpath", global_tis.get("mwf_nsubpath", 3)
        )
        for i in ens_keys
    ]
    # TOML requires string keys; int keys are stored as strings and converted
    # back to int on restore in REPEX_state.__init__().
    config["current"]["ensemble_move_counts"] = {
        str(i): state.ensemble_move_counts.get(i, 0)
        for i in ens_keys
    }
