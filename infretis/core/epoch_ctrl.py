"""Epoch-wise per-ensemble subtrajectory-count controller for WF/MWF moves."""

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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


def apply_epoch_ctrl(state, cstep: int) -> None:
    """Apply epoch-wise n_jumps updates to targeted ensembles.

    Fires at each epoch boundary (cstep divisible by epoch_size and > 0).
    Only ensembles listed in config["simulation"]["epoch_nsubpath_ens"] are
    modified; all other ensembles are left unchanged.

    Args:
        state: REPEX_state instance whose ensembles are mutated in-place.
        cstep: current completed-move step count.
    """
    sim = state.config["simulation"]
    epoch_size = sim.get("epoch_size", 0)
    if not (epoch_size > 0 and cstep > 0 and cstep % epoch_size == 0):
        return
    epoch_idx = cstep // epoch_size
    ens_targets = sim.get("epoch_nsubpath_ens", [])
    val_schedules = sim.get("epoch_nsubpath_vals", [])
    for ens_i, vals in zip(ens_targets, val_schedules):
        if ens_i in state.ensembles and vals:
            new_val = int(vals[epoch_idx % len(vals)])
            old_val = state.ensembles[ens_i]["tis_set"].get("n_jumps")
            state.ensembles[ens_i]["tis_set"]["n_jumps"] = new_val
            logger.info(
                "Epoch %d: ensemble %03d n_jumps %s -> %d",
                epoch_idx, ens_i, old_val, new_val,
            )


def mirror_epoch_ctrl(state, config: dict) -> None:
    """Mirror per-ensemble n_jumps/mwf_nsubpath into config for restart.

    Writes ensemble_nsubpath and ensemble_mwf_nsubpath into
    config["simulation"] so that write_toml() serializes the current live
    per-ensemble values and initiate_ensembles() restores them exactly on
    restart.

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
