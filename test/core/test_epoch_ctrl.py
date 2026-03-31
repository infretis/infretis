"""Tests for epoch-wise per-ensemble n_jumps controller.

Verifies properties end-to-end for both controller modes:

global_step mode:
  1. Only the targeted ensemble's n_jumps alternates; the untargeted WF
     ensemble is never touched.
  2. Stopping mid-epoch and restarting restores the exact per-ensemble
     value (via ensemble_nsubpath in restart.toml).
  3. The first post-restart moves keep the restored value until the next
     epoch boundary — the controller does NOT fire mid-epoch.
  4. The n_jumps value that select_shoot() would log to move_blocks.tsv
     reflects the alternating schedule for the targeted ensemble and
     the constant original value for the untargeted ensemble.

per_ensemble_moves mode:
  5. The controller fires for ensemble i after every k completed moves of
     that ensemble, independent of cstep.
  6. When k differs per target, each ensemble fires on its own schedule.
  7. Restart restores ensemble_move_counts so the phase is preserved.
  8. epoch_count='accepted' counts only accepted moves.
"""

import copy
import csv
import os
import shutil
import time
from pathlib import Path as StdPath

import numpy as np
import tomli
import tomli_w
import pytest

import infretis.core.tis as tis
from infretis.classes.path import load_paths_from_disk
from infretis.classes.repex import REPEX_state, spawn_rng
from infretis.core.epoch_ctrl import (
    apply_epoch_ctrl,
    mirror_epoch_ctrl,
    update_epoch_stats,
    validate_epoch_ctrl,
    _append_tsv,
    _epoch_ctrl_active,
    _EPOCH_SUMMARY_FNAME,
    _EPOCH_SOFTMAX_DEBUG_FNAME,
    _EPOCH_SUMMARY_COLS,
    _EPOCH_SUMMARY_LPLSCOLS,
    _EPOCH_SOFTMAX_DEBUG_COLS,
    _STALE_SIM_KEYS,
    _STALE_CURRENT_KEYS,
    _DEFAULT_SIM_KEYS,
)
from infretis.setup import TOMLConfigError, check_config


# ---------------------------------------------------------------------------
# Minimal config factory
# ---------------------------------------------------------------------------

def _make_config(cstep: int = 0) -> dict:
    """Build a minimal but valid config for 3 interfaces.

    Layout:
      ensemble 0 — [0-] (minus),  mc_move = "sh"
      ensemble 1 — [0+],          mc_move = "wf"  ← TARGETED
      ensemble 2 — [1+],          mc_move = "wf"  ← untargeted

    Epoch schedule: every 10 steps, ensemble 1 cycles [2, 4].
    Global initial n_jumps = 2 for all ensembles.
    """
    return {
        "simulation": {
            "interfaces": [0.0, 0.3, 0.6],
            "shooting_moves": ["sh", "wf", "wf"],
            "tis_set": {
                "lambda_minus_one": False,
                "n_jumps": 2,
                "maxlength": 100,
                "allowmaxlength": False,
                "zero_momentum": False,
                "quantis": False,
                "accept_all": False,
                "mwf_nsubpath": 3,
            },
            "seed": 0,
            "steps": 100,
            "epoch_size": 10,
            "epoch_nsubpath_ens": [1],       # only [0+]
            "epoch_nsubpath_vals": [[2, 4]],  # cycles 2 → 4 → 2 → …
        },
        "runner": {"workers": 1},
        "current": {
            "cstep": cstep,
            "size": 3,   # len(interfaces)
            "locked": [],
        },
    }


def _make_state(cstep: int = 0) -> REPEX_state:
    state = REPEX_state(_make_config(cstep), minus=True)
    state.initiate_ensembles()
    return state


def _eff_n_jumps_for_wf(ens: dict) -> int:
    """Return the n_jumps value that select_shoot() logs for a WF move.

    Mirrors the single line in select_shoot():
        eff_n_jumps = pens0["ens"]["tis_set"].get("n_jumps", 2)
    """
    return ens["tis_set"].get("n_jumps", 2)


# ---------------------------------------------------------------------------
# Test 1 — only the targeted ensemble alternates
# ---------------------------------------------------------------------------

def test_only_targeted_ensemble_alternates():
    """Ensemble 2 (untargeted WF) must never change as ensemble 1 cycles."""
    state = _make_state()

    # Initial state: both WF ensembles carry the global default
    assert state.ensembles[1]["tis_set"]["n_jumps"] == 2
    assert state.ensembles[2]["tis_set"]["n_jumps"] == 2

    # Simulate three epoch boundaries
    expectations_ens1 = {10: 4, 20: 2, 30: 4}
    for cstep, expected in expectations_ens1.items():
        state.config["current"]["cstep"] = cstep
        apply_epoch_ctrl(state, state.cstep)

        # Targeted ensemble cycles
        assert state.ensembles[1]["tis_set"]["n_jumps"] == expected, (
            f"cstep={cstep}: expected ens 1 n_jumps={expected}"
        )
        # Untargeted ensemble is always unchanged
        assert state.ensembles[2]["tis_set"]["n_jumps"] == 2, (
            f"cstep={cstep}: ens 2 (untargeted) should still be 2"
        )

    # Property 4 — what select_shoot() would log for each ensemble
    # After cstep=30 (epoch 3, vals[3%2]=vals[1]=4):
    assert _eff_n_jumps_for_wf(state.ensembles[1]) == 4  # targeted alternates
    assert _eff_n_jumps_for_wf(state.ensembles[2]) == 2  # untargeted constant


# ---------------------------------------------------------------------------
# Test 2 — restart restores value and epoch phase
# ---------------------------------------------------------------------------

def test_restart_restores_value_and_phase(tmp_path):
    """Stop mid-epoch; restart must restore n_jumps and correct epoch phase."""
    state = _make_state()

    # --- Phase A: run to epoch 1 boundary (cstep=10) ---
    state.config["current"]["cstep"] = 10
    apply_epoch_ctrl(state, state.cstep)
    assert state.ensembles[1]["tis_set"]["n_jumps"] == 4  # epoch 1 applied
    assert state.ensembles[2]["tis_set"]["n_jumps"] == 2  # untouched

    # --- Phase B: "stop" at cstep=15 (mid-epoch 2) ---
    state.config["current"]["cstep"] = 15
    mirror_epoch_ctrl(state, state.config)

    # Verify the mirror wrote the live values into config
    nsubpath = state.config["simulation"]["ensemble_nsubpath"]
    assert nsubpath[1] == 4, "ensemble 1 n_jumps must be mirrored as 4"
    assert nsubpath[2] == 2, "ensemble 2 n_jumps must be mirrored as 2"

    # Serialize the config to restart.toml (round-trip through TOML)
    restart_path = tmp_path / "restart.toml"
    with restart_path.open("wb") as fh:
        tomli_w.dump(state.config, fh)

    # --- Phase C: restore from restart.toml ---
    with restart_path.open("rb") as fh:
        restored_cfg = tomli.load(fh)

    # Confirm the round-trip preserved the lists
    assert restored_cfg["simulation"]["ensemble_nsubpath"][1] == 4
    assert restored_cfg["simulation"]["ensemble_nsubpath"][2] == 2

    state2 = REPEX_state(restored_cfg, minus=True)
    state2.initiate_ensembles()

    # Property 2 — values restored exactly
    assert state2.ensembles[1]["tis_set"]["n_jumps"] == 4, (
        "restart must restore ens 1 n_jumps to 4"
    )
    assert state2.ensembles[2]["tis_set"]["n_jumps"] == 2, (
        "restart must restore ens 2 n_jumps to 2"
    )

    # Property 3 — controller does NOT fire at cstep=15 (not a boundary)
    # cstep is already 15 from the restored config
    apply_epoch_ctrl(state2, state2.cstep)
    assert state2.ensembles[1]["tis_set"]["n_jumps"] == 4, (
        "cstep=15 is mid-epoch; n_jumps must stay 4 until cstep=20"
    )

    # Property 3 continued — next boundary at cstep=20 fires correctly
    state2.config["current"]["cstep"] = 20
    apply_epoch_ctrl(state2, state2.cstep)
    assert state2.ensembles[1]["tis_set"]["n_jumps"] == 2, (
        "cstep=20 (epoch 2): ens 1 should cycle to 2"
    )
    assert state2.ensembles[2]["tis_set"]["n_jumps"] == 2, (
        "ens 2 must remain 2 after restart's first epoch boundary"
    )

    # Property 4 — TSV would log the right values for each ensemble
    assert _eff_n_jumps_for_wf(state2.ensembles[1]) == 2  # just fired, now 2
    assert _eff_n_jumps_for_wf(state2.ensembles[2]) == 2  # never changed


# ---------------------------------------------------------------------------
# Minimal config for check_config (needs engine keys and ensemble_engines)
# ---------------------------------------------------------------------------

def _check_config_base() -> dict:
    """Minimal config accepted by check_config (no epoch ctrl keys)."""
    return {
        "simulation": {
            "interfaces": [0.0, 0.3, 0.6],
            "shooting_moves": ["sh", "wf", "wf"],
            "tis_set": {
                "lambda_minus_one": False,
                "quantis": False,
                "accept_all": False,
            },
            "ensemble_engines": [["engine"], ["engine"], ["engine"]],
        },
        "runner": {"workers": 1},
        "current": {"cstep": 0, "wsubcycles": [0], "tsubcycles": 0},
        "engine": {"class": "ase"},
    }


# ---------------------------------------------------------------------------
# Test 3 — non-WF target is rejected by config validation
# ---------------------------------------------------------------------------

def test_non_wf_target_is_rejected():
    """check_config must raise when epoch_nsubpath_ens targets a non-WF/MWF ensemble."""
    base = _check_config_base()

    # Ensemble 0 uses "sh" — must be rejected
    cfg = copy.deepcopy(base)
    cfg["simulation"]["epoch_nsubpath_ens"] = [0]
    cfg["simulation"]["epoch_nsubpath_vals"] = [[2, 4]]
    with pytest.raises(TOMLConfigError, match="move 'sh'"):
        check_config(cfg)

    # Ensemble 1 uses "wf" — must pass
    cfg2 = copy.deepcopy(base)
    cfg2["simulation"]["epoch_nsubpath_ens"] = [1]
    cfg2["simulation"]["epoch_nsubpath_vals"] = [[2, 4]]
    check_config(cfg2)  # should not raise

    # Ensemble 2 uses "wf" — must also pass
    cfg3 = copy.deepcopy(base)
    cfg3["simulation"]["epoch_nsubpath_ens"] = [2]
    cfg3["simulation"]["epoch_nsubpath_vals"] = [[3, 6]]
    check_config(cfg3)  # should not raise

    # Mixed: one WF, one sh — must still be rejected (sh is invalid)
    cfg4 = copy.deepcopy(base)
    cfg4["simulation"]["shooting_moves"] = ["sh", "wf", "sh"]
    cfg4["simulation"]["epoch_nsubpath_ens"] = [1, 2]
    cfg4["simulation"]["epoch_nsubpath_vals"] = [[2, 4], [3, 6]]
    with pytest.raises(TOMLConfigError, match="move 'sh'"):
        check_config(cfg4)


# ---------------------------------------------------------------------------
# Test 4 — written TSV contains the expected effective n_jumps per row
# ---------------------------------------------------------------------------

def test_written_tsv_contains_expected_eff_n_jumps(tmp_path):
    """move_blocks.tsv rows must carry alternating n_jumps for targeted WF
    ensemble, constant 2 for untargeted WF, and 'NA' for sh moves.

    Timing note: in a real simulation the epoch controller fires inside
    treat_output() AFTER the move worker returns, so the move at the
    boundary step still sees the pre-epoch value.  We replicate that by
    reading n_jumps → writing the TSV row → calling apply_epoch_ctrl().
    """
    worker_dir = tmp_path / "worker0"
    worker_dir.mkdir()

    state = _make_state()
    tis._CTX = {"w_folder": str(worker_dir), "cstep": 0, "pin": 0}

    # (cstep, ens_idx, move_name, expected_n_jumps_in_tsv)
    # epoch_size=10, schedule=[2,4] for ens 1 only.
    # "AT boundary" rows still carry the pre-fire value (2 or 4);
    # "AFTER boundary" rows carry the post-fire value.
    sim_moves = [
        (1,  1, "wf",  2),    # epoch 0: initial
        (2,  2, "wf",  2),    # untargeted: always 2
        (3,  0, "sh",  "NA"), # sh: always NA
        (10, 1, "wf",  2),    # AT epoch-1 boundary; epoch fires AFTER
        (11, 1, "wf",  4),    # AFTER epoch 1 fired (2→4)
        (12, 2, "wf",  2),    # untargeted: still 2
        (13, 0, "sh",  "NA"), # sh: NA
        (20, 1, "wf",  4),    # AT epoch-2 boundary; epoch fires AFTER
        (21, 1, "wf",  2),    # AFTER epoch 2 fired (4→2)
        (22, 2, "wf",  2),    # untargeted: still 2
    ]

    header = (
        "cstep\tpin\tens_name\tmove\tpath_n\taccepted\tstatus\tn_jumps"
    )

    for cstep, ens_idx, move_name, _ in sim_moves:
        state.config["current"]["cstep"] = cstep
        tis._CTX["cstep"] = cstep

        # Read n_jumps BEFORE epoch fires (mirrors worker / select_shoot)
        if move_name in ("wf", "mwf"):
            eff_n_jumps = state.ensembles[ens_idx]["tis_set"].get("n_jumps", 2)
        else:
            eff_n_jumps = "NA"

        ens_name = state.ensembles[ens_idx]["ens_name"]
        row = (
            f"{cstep}\t0\t{ens_name}\t{move_name}\t"
            f"-1\t1\tACC\t{eff_n_jumps}"
        )
        tis._append_tsv("move_blocks.tsv", header, row)

        # Fire epoch AFTER writing (mirrors treat_output ordering)
        apply_epoch_ctrl(state, state.cstep)

    # Read back the TSV and assert on every row
    tsv_path = worker_dir / "move_blocks.tsv"
    assert tsv_path.exists()

    with tsv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    assert len(rows) == len(sim_moves), "TSV row count must match simulated moves"

    for row, (cstep, ens_idx, move_name, expected) in zip(rows, sim_moves):
        actual = row["n_jumps"]
        if expected == "NA":
            assert actual == "NA", (
                f"cstep={cstep} move={move_name}: expected NA, got {actual!r}"
            )
        else:
            assert int(actual) == expected, (
                f"cstep={cstep} ens={ens_idx} move={move_name}: "
                f"expected n_jumps={expected}, got {actual!r}"
            )

    # Explicit alternation check for the targeted ensemble only
    targeted_rows = [
        (cstep, int(row["n_jumps"]))
        for row, (cstep, ens_idx, move_name, _) in zip(rows, sim_moves)
        if move_name == "wf" and ens_idx == 1
    ]
    # Before epoch 1 boundary: 2; after epoch 1: 4; after epoch 2: 2
    assert [v for _, v in targeted_rows] == [2, 2, 4, 4, 2], (
        f"targeted ensemble n_jumps sequence wrong: {targeted_rows}"
    )

    # Untargeted WF ensemble must be constant throughout
    untargeted_rows = [
        int(row["n_jumps"])
        for row, (_, ens_idx, move_name, _) in zip(rows, sim_moves)
        if move_name == "wf" and ens_idx == 2
    ]
    assert all(v == 2 for v in untargeted_rows), (
        f"untargeted ensemble n_jumps changed: {untargeted_rows}"
    )


# ---------------------------------------------------------------------------
# Helpers for per_ensemble_moves mode tests
# ---------------------------------------------------------------------------

def _make_config_per_ens(k=3, k_list=None) -> dict:
    """Config for per_ensemble_moves mode, targeting ensemble 1, k moves."""
    cfg = _make_config()
    cfg["simulation"]["epoch_mode"] = "per_ensemble_moves"
    cfg["simulation"]["epoch_move_k"] = k_list if k_list is not None else k
    # epoch_size is unused in this mode but kept to avoid KeyError elsewhere
    return cfg


def _make_state_per_ens(k=3, k_list=None) -> REPEX_state:
    state = REPEX_state(_make_config_per_ens(k=k, k_list=k_list), minus=True)
    state.initiate_ensembles()
    return state


# ---------------------------------------------------------------------------
# Test 5 — per_ensemble_moves fires every k moves for the targeted ensemble
# ---------------------------------------------------------------------------

def test_per_ensemble_moves_fires_at_k_moves():
    """Controller must fire after each k-th completed move of the target."""
    k = 3
    state = _make_state_per_ens(k=k)

    assert state.ensembles[1]["tis_set"]["n_jumps"] == 2  # initial

    # Simulate 2 moves — no boundary yet
    for _ in range(k - 1):
        state.ensemble_move_counts[1] = (
            state.ensemble_move_counts.get(1, 0) + 1
        )
        apply_epoch_ctrl(state, state.cstep)

    assert state.ensembles[1]["tis_set"]["n_jumps"] == 2, (
        f"should not fire before {k} moves"
    )

    # k-th move — boundary: schedule[1 % 2] = schedule[1] = 4
    state.ensemble_move_counts[1] = (
        state.ensemble_move_counts.get(1, 0) + 1
    )
    apply_epoch_ctrl(state, state.cstep)
    assert state.ensemble_move_counts[1] == k
    assert state.ensembles[1]["tis_set"]["n_jumps"] == 4, (
        "first boundary (k moves): n_jumps must cycle to 4"
    )
    # Untargeted ensemble is unchanged
    assert state.ensembles[2]["tis_set"]["n_jumps"] == 2

    # Second boundary (2k moves)
    for _ in range(k):
        state.ensemble_move_counts[1] = (
            state.ensemble_move_counts.get(1, 0) + 1
        )
    apply_epoch_ctrl(state, state.cstep)
    assert state.ensembles[1]["tis_set"]["n_jumps"] == 2, (
        "second boundary (2k moves): n_jumps must cycle back to 2"
    )


# ---------------------------------------------------------------------------
# Test 6 — different k per target
# ---------------------------------------------------------------------------

def test_per_ensemble_moves_different_k_per_target():
    """Each targeted ensemble fires on its own k schedule."""
    # Target both WF ensembles with different k values
    state = REPEX_state(_make_config(), minus=True)
    state.config["simulation"]["epoch_mode"] = "per_ensemble_moves"
    state.config["simulation"]["epoch_nsubpath_ens"] = [1, 2]
    state.config["simulation"]["epoch_nsubpath_vals"] = [[2, 4], [2, 6]]
    state.config["simulation"]["epoch_move_k"] = [3, 5]
    state.initiate_ensembles()

    # Simulate 3 moves for ens 1 and 2 moves for ens 2
    state.ensemble_move_counts[1] = 3
    state.ensemble_move_counts[2] = 2
    apply_epoch_ctrl(state, state.cstep)

    # Ens 1 hit its k=3 boundary: epoch_idx=1 → vals[1]=4
    assert state.ensembles[1]["tis_set"]["n_jumps"] == 4, (
        "ens 1 should fire at count=3 (k=3)"
    )
    # Ens 2 has only 2 moves (k=5): must not fire
    assert state.ensembles[2]["tis_set"]["n_jumps"] == 2, (
        "ens 2 must not fire at count=2 (k=5)"
    )

    # Give ens 2 its 5th move
    state.ensemble_move_counts[2] = 5
    apply_epoch_ctrl(state, state.cstep)
    assert state.ensembles[2]["tis_set"]["n_jumps"] == 6, (
        "ens 2 should fire at count=5 (k=5): epoch_idx=1 → vals[1]=6"
    )


# ---------------------------------------------------------------------------
# Test 7 — restart restores ensemble_move_counts and preserves phase
# ---------------------------------------------------------------------------

def test_per_ensemble_moves_restart_restores_counts(tmp_path):
    """After restart, ensemble_move_counts must be restored so the phase
    is preserved and the controller fires at the correct total count."""
    k = 3
    state = _make_state_per_ens(k=k)

    # Simulate k+1 moves (one epoch has fired, one more move into next)
    state.ensemble_move_counts[1] = k
    apply_epoch_ctrl(state, state.cstep)
    assert state.ensembles[1]["tis_set"]["n_jumps"] == 4

    state.ensemble_move_counts[1] = k + 1  # mid-epoch

    # Serialize via mirror + TOML round-trip
    mirror_epoch_ctrl(state, state.config)
    restart_path = tmp_path / "restart.toml"
    with restart_path.open("wb") as fh:
        tomli_w.dump(state.config, fh)
    with restart_path.open("rb") as fh:
        restored_cfg = tomli.load(fh)

    # Verify counts survived the round-trip
    raw = restored_cfg["current"]["ensemble_move_counts"]
    assert int(raw["1"]) == k + 1, "count must survive TOML round-trip"

    state2 = REPEX_state(restored_cfg, minus=True)
    state2.initiate_ensembles()

    # n_jumps restored to 4 (from ensemble_nsubpath)
    assert state2.ensembles[1]["tis_set"]["n_jumps"] == 4
    # count restored
    assert state2.ensemble_move_counts.get(1, 0) == k + 1

    # Controller must NOT fire mid-epoch (count=k+1 is not a multiple of k=3)
    apply_epoch_ctrl(state2, state2.cstep)
    assert state2.ensembles[1]["tis_set"]["n_jumps"] == 4, (
        "mid-epoch: n_jumps must stay 4"
    )

    # Next boundary at count=2k
    state2.ensemble_move_counts[1] = 2 * k
    apply_epoch_ctrl(state2, state2.cstep)
    assert state2.ensembles[1]["tis_set"]["n_jumps"] == 2, (
        "count=2k: n_jumps must cycle back to 2"
    )


# ---------------------------------------------------------------------------
# Test 8 — epoch_count='accepted' only counts accepted moves
# ---------------------------------------------------------------------------

def test_epoch_count_accepted_ignores_rejected_moves():
    """With epoch_count='accepted', rejected moves must not advance the counter
    and must not trigger the controller."""
    k = 2
    state = _make_state_per_ens(k=k)
    state.config["simulation"]["epoch_count"] = "accepted"

    # Simulate k rejected moves — counter must stay at 0, no firing
    for _ in range(k):
        # Rejected: do NOT increment (mirrors treat_output with ACC check)
        apply_epoch_ctrl(state, state.cstep)

    assert state.ensemble_move_counts.get(1, 0) == 0
    assert state.ensembles[1]["tis_set"]["n_jumps"] == 2, (
        "rejected moves must not advance counter or fire controller"
    )

    # Two accepted moves — counter reaches k, fires
    state.ensemble_move_counts[1] = k
    apply_epoch_ctrl(state, state.cstep)
    assert state.ensembles[1]["tis_set"]["n_jumps"] == 4, (
        "k accepted moves: controller must fire"
    )


# ---------------------------------------------------------------------------
# Test 9 — validate_epoch_ctrl rejects bad per_ensemble_moves config
# ---------------------------------------------------------------------------

def test_validate_rejects_bad_per_ensemble_moves_config():
    """validate_epoch_ctrl must raise for missing or invalid epoch_move_k."""
    base = _check_config_base()

    # Missing epoch_move_k
    cfg = copy.deepcopy(base)
    cfg["simulation"]["epoch_mode"] = "per_ensemble_moves"
    cfg["simulation"]["epoch_nsubpath_ens"] = [1]
    cfg["simulation"]["epoch_nsubpath_vals"] = [[2, 4]]
    with pytest.raises(TOMLConfigError, match="epoch_move_k"):
        check_config(cfg)

    # epoch_move_k list length mismatch
    cfg2 = copy.deepcopy(base)
    cfg2["simulation"]["epoch_mode"] = "per_ensemble_moves"
    cfg2["simulation"]["epoch_nsubpath_ens"] = [1]
    cfg2["simulation"]["epoch_nsubpath_vals"] = [[2, 4]]
    cfg2["simulation"]["epoch_move_k"] = [3, 5]  # length 2, targets length 1
    with pytest.raises(TOMLConfigError, match="epoch_move_k"):
        check_config(cfg2)

    # Non-positive k
    cfg3 = copy.deepcopy(base)
    cfg3["simulation"]["epoch_mode"] = "per_ensemble_moves"
    cfg3["simulation"]["epoch_nsubpath_ens"] = [1]
    cfg3["simulation"]["epoch_nsubpath_vals"] = [[2, 4]]
    cfg3["simulation"]["epoch_move_k"] = 0
    with pytest.raises(TOMLConfigError, match="positive"):
        check_config(cfg3)

    # Unknown epoch_mode
    cfg4 = copy.deepcopy(base)
    cfg4["simulation"]["epoch_mode"] = "bananas"
    with pytest.raises(TOMLConfigError, match="epoch_mode"):
        check_config(cfg4)

    # Unknown epoch_count
    cfg5 = copy.deepcopy(base)
    cfg5["simulation"]["epoch_count"] = "maybe"
    with pytest.raises(TOMLConfigError, match="epoch_count"):
        check_config(cfg5)


# ---------------------------------------------------------------------------
# Tests 10-12 — epoch stats buffer
# ---------------------------------------------------------------------------

def _push_moves(state, ens_idx, n_attempted, n_accepted, path_length, subcycles, lambda_max):
    """Helper: push n_attempted moves into the stats buffer, n_accepted accepted."""
    for i in range(n_attempted):
        update_epoch_stats(
            state, ens_idx,
            accepted=(i < n_accepted),
            path_length=path_length,
            subcycles=subcycles,
            lambda_max=lambda_max,
        )


def test_stats_accumulate_for_targeted_only():
    """update_epoch_stats must accumulate only for targeted ensembles."""
    state = _make_state()

    # Targeted ensemble 1: push 5 moves (3 accepted)
    _push_moves(state, ens_idx=1, n_attempted=5, n_accepted=3,
                path_length=20.0, subcycles=4, lambda_max=0.5)

    # Untargeted ensemble 2: push 5 moves — must be silently ignored
    _push_moves(state, ens_idx=2, n_attempted=5, n_accepted=5,
                path_length=99.0, subcycles=99, lambda_max=0.99)

    buf = state.ensemble_epoch_stats.get(1, {})
    assert buf["n_attempted"] == 5
    assert buf["n_accepted"] == 3
    assert abs(buf["path_length_sum"] - 100.0) < 1e-9
    assert buf["subcycles_sum"] == 20
    assert abs(buf["lambda_max"] - 0.5) < 1e-9

    assert 2 not in state.ensemble_epoch_stats, (
        "untargeted ensemble must not appear in stats buffer"
    )


def test_stats_flushed_and_reset_at_epoch_boundary(tmp_path):
    """apply_epoch_ctrl must write epoch_summary.tsv and reset the buffer."""
    state = _make_state()
    state.config["output"] = {"data_dir": str(tmp_path)}

    # Accumulate 10 moves (7 accepted) into ensemble 1
    _push_moves(state, ens_idx=1, n_attempted=10, n_accepted=7,
                path_length=15.0, subcycles=3, lambda_max=0.42)

    # Fire epoch boundary at cstep=10 (epoch_size=10)
    state.config["current"]["cstep"] = 10
    apply_epoch_ctrl(state, state.cstep)

    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    assert tsv_path.exists(), "epoch_summary.tsv must be created"

    rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows) == 1
    r = rows[0]
    assert int(r["epoch_idx"]) == 1
    assert r["ens_name"] == "001"
    assert int(r["n_attempted"]) == 10
    assert int(r["n_accepted"]) == 7
    assert abs(float(r["acc_rate"]) - 0.7) < 1e-4
    assert abs(float(r["avg_path_length"]) - 15.0) < 1e-2
    assert abs(float(r["avg_subcycles"]) - 3.0) < 1e-2
    assert int(r["n_jumps_old"]) == 2   # initial global n_jumps
    assert int(r["n_jumps_new"]) == 4   # epoch 1 → schedule[1] = 4

    # Removed columns must not appear in new schema
    for removed_col in ("lambda_max", "epoch_mode", "epoch_count", "ctrl_mode"):
        assert removed_col not in r, f"{removed_col} must not appear in new public schema"

    # Header must match 12-column public schema exactly
    header_cols = tsv_path.read_text().splitlines()[0].split("\t")
    assert header_cols == list(_EPOCH_SUMMARY_COLS), (
        f"TSV header {header_cols} must equal _EPOCH_SUMMARY_COLS"
    )

    # Buffer must be reset after flush
    buf = state.ensemble_epoch_stats.get(1, {})
    assert buf["n_attempted"] == 0, "buffer must reset after epoch flush"


def test_stats_rows_accumulate_across_epochs(tmp_path):
    """Each epoch boundary appends one row; header appears only once."""
    state = _make_state()
    state.config["output"] = {"data_dir": str(tmp_path)}

    for epoch in range(1, 4):
        _push_moves(state, ens_idx=1, n_attempted=10, n_accepted=epoch,
                    path_length=float(epoch * 10), subcycles=1, lambda_max=float(epoch) * 0.1)
        state.config["current"]["cstep"] = epoch * 10
        apply_epoch_ctrl(state, state.cstep)

    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    lines = tsv_path.read_text().splitlines()
    # One header + three data rows
    assert lines[0].startswith("epoch_idx"), "first line must be header"
    assert len(lines) == 4, f"expected 4 lines (header + 3 rows), got {len(lines)}"

    rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert [int(r["epoch_idx"]) for r in rows] == [1, 2, 3]

    # Schema assertions
    header_cols = lines[0].split("\t")
    assert header_cols == list(_EPOCH_SUMMARY_COLS), (
        f"TSV header must equal _EPOCH_SUMMARY_COLS"
    )
    for removed_col in ("lambda_max", "epoch_mode", "epoch_count", "ctrl_mode"):
        assert removed_col not in rows[0], f"{removed_col} must not appear in new public schema"


# ---------------------------------------------------------------------------
# Helpers shared by adaptive-mode tests
# ---------------------------------------------------------------------------

def _make_adaptive_config() -> dict:
    """Minimal config for adaptive mode targeting ensemble 1."""
    cfg = _make_config()
    del cfg["simulation"]["epoch_nsubpath_vals"]
    cfg["simulation"]["epoch_ctrl_mode"] = "adaptive"
    cfg["simulation"]["adaptive_nsubpath_min"] = 1
    cfg["simulation"]["adaptive_nsubpath_max"] = 8
    cfg["simulation"]["adaptive_accept_low"] = 0.20
    cfg["simulation"]["adaptive_accept_high"] = 0.60
    cfg["simulation"]["adaptive_lambda_gain_low"] = 0.05
    cfg["simulation"]["adaptive_pathlen_high"] = 400
    return cfg


def _make_adaptive_state() -> REPEX_state:
    state = REPEX_state(_make_adaptive_config(), minus=True)
    state.initiate_ensembles()
    return state


# ---------------------------------------------------------------------------
# Test 13 — static mode never updates n_jumps
# ---------------------------------------------------------------------------

def test_static_mode_never_updates_n_jumps():
    """With epoch_ctrl_mode='static', n_jumps must not change at epoch boundary."""
    cfg = _make_config()
    cfg["simulation"]["epoch_ctrl_mode"] = "static"
    state = REPEX_state(cfg, minus=True)
    state.initiate_ensembles()

    assert state.ensembles[1]["tis_set"]["n_jumps"] == 2

    # Fire several epoch boundaries — nothing should change
    for cstep in (10, 20, 30):
        state.config["current"]["cstep"] = cstep
        apply_epoch_ctrl(state, cstep)

    assert state.ensembles[1]["tis_set"]["n_jumps"] == 2, (
        "static mode must never update n_jumps"
    )
    assert state.ensembles[2]["tis_set"]["n_jumps"] == 2


# ---------------------------------------------------------------------------
# Test 14 — scheduled mode with explicit epoch_ctrl_mode key
# ---------------------------------------------------------------------------

def test_scheduled_mode_unchanged():
    """Explicit epoch_ctrl_mode='scheduled' must behave identically to the
    inferred scheduled behavior."""
    cfg = _make_config()
    cfg["simulation"]["epoch_ctrl_mode"] = "scheduled"
    state = REPEX_state(cfg, minus=True)
    state.initiate_ensembles()

    state.config["current"]["cstep"] = 10
    apply_epoch_ctrl(state, 10)
    assert state.ensembles[1]["tis_set"]["n_jumps"] == 4

    state.config["current"]["cstep"] = 20
    apply_epoch_ctrl(state, 20)
    assert state.ensembles[1]["tis_set"]["n_jumps"] == 2


# ---------------------------------------------------------------------------
# Test 15 — scheduled inferred without epoch_ctrl_mode key
# ---------------------------------------------------------------------------

def test_infer_scheduled_without_ctrl_mode_key():
    """Config with epoch_nsubpath_vals but no epoch_ctrl_mode key must behave
    as scheduled mode."""
    cfg = _make_config()
    assert "epoch_ctrl_mode" not in cfg["simulation"]

    state = REPEX_state(cfg, minus=True)
    state.initiate_ensembles()

    state.config["current"]["cstep"] = 10
    apply_epoch_ctrl(state, 10)
    assert state.ensembles[1]["tis_set"]["n_jumps"] == 4, (
        "inferred scheduled mode must cycle n_jumps"
    )


# ---------------------------------------------------------------------------
# Test 16 — adaptive + epoch_nsubpath_vals raises
# ---------------------------------------------------------------------------

def test_invalid_mixed_adaptive_with_vals():
    """epoch_ctrl_mode='adaptive' combined with epoch_nsubpath_vals must raise."""
    base = _check_config_base()
    cfg = copy.deepcopy(base)
    cfg["simulation"]["epoch_ctrl_mode"] = "adaptive"
    cfg["simulation"]["epoch_nsubpath_ens"] = [1]
    cfg["simulation"]["epoch_nsubpath_vals"] = [[2, 4]]
    cfg["simulation"]["adaptive_nsubpath_min"] = 1
    cfg["simulation"]["adaptive_nsubpath_max"] = 8
    cfg["simulation"]["adaptive_accept_low"] = 0.20
    cfg["simulation"]["adaptive_accept_high"] = 0.60
    cfg["simulation"]["adaptive_lambda_gain_low"] = 0.05
    cfg["simulation"]["adaptive_pathlen_high"] = 400
    with pytest.raises(TOMLConfigError):
        check_config(cfg)


# ---------------------------------------------------------------------------
# Test 17 — scheduled + adaptive_* keys raises
# ---------------------------------------------------------------------------

def test_invalid_mixed_scheduled_with_adaptive_keys():
    """epoch_ctrl_mode='scheduled' combined with adaptive_* keys must raise."""
    base = _check_config_base()
    cfg = copy.deepcopy(base)
    cfg["simulation"]["epoch_ctrl_mode"] = "scheduled"
    cfg["simulation"]["epoch_nsubpath_ens"] = [1]
    cfg["simulation"]["epoch_nsubpath_vals"] = [[2, 4]]
    cfg["simulation"]["adaptive_nsubpath_min"] = 1
    with pytest.raises(TOMLConfigError):
        check_config(cfg)


# ---------------------------------------------------------------------------
# Test 18 — adaptive increases n_jumps on low acc + low lambda
# ---------------------------------------------------------------------------

def test_adaptive_increase():
    """Low acceptance + low avg_lambda_max must increment n_jumps by 1."""
    state = _make_adaptive_state()
    # n_jumps starts at 2; push stats that trigger +1
    # acc_rate = 0/10 = 0.0 < 0.20; avg_lambda_max = 0.01 < 0.05
    for _ in range(10):
        update_epoch_stats(
            state, 1, accepted=False,
            path_length=50.0, subcycles=2, lambda_max=0.01
        )

    state.config["current"]["cstep"] = 10
    apply_epoch_ctrl(state, 10)

    assert state.ensembles[1]["tis_set"]["n_jumps"] == 3, (
        "low acc + low explore: n_jumps must increase by 1"
    )


# ---------------------------------------------------------------------------
# Test 19 — adaptive decreases n_jumps on high path length + ok acc
# ---------------------------------------------------------------------------

def test_adaptive_decrease():
    """High avg_path_len + ok acc_rate must decrement n_jumps by 1."""
    state = _make_adaptive_state()
    state.ensembles[1]["tis_set"]["n_jumps"] = 5
    # acc_rate = 5/10 = 0.5 >= 0.20; avg_path_len = 500 > 400
    for i in range(10):
        update_epoch_stats(
            state, 1, accepted=(i < 5),
            path_length=500.0, subcycles=2, lambda_max=0.3
        )

    state.config["current"]["cstep"] = 10
    apply_epoch_ctrl(state, 10)

    assert state.ensembles[1]["tis_set"]["n_jumps"] == 4, (
        "high cost + ok acc: n_jumps must decrease by 1"
    )


# ---------------------------------------------------------------------------
# Test 20 — adaptive holds when within band
# ---------------------------------------------------------------------------

def test_adaptive_hold():
    """Stats within band must leave n_jumps unchanged."""
    state = _make_adaptive_state()
    state.ensembles[1]["tis_set"]["n_jumps"] = 3
    # acc_rate = 5/10 = 0.5 >= 0.20; avg_lambda_max = 0.3 >= 0.05; avg_path_len = 100 < 400
    for i in range(10):
        update_epoch_stats(
            state, 1, accepted=(i < 5),
            path_length=100.0, subcycles=2, lambda_max=0.3
        )

    state.config["current"]["cstep"] = 10
    apply_epoch_ctrl(state, 10)

    assert state.ensembles[1]["tis_set"]["n_jumps"] == 3, (
        "within-band stats must leave n_jumps unchanged"
    )


# ---------------------------------------------------------------------------
# Test 21 — adaptive clamps at max
# ---------------------------------------------------------------------------

def test_adaptive_clamp_at_max():
    """When n_jumps is at adaptive_nsubpath_max, increase must stay at max."""
    state = _make_adaptive_state()
    n_max = state.config["simulation"]["adaptive_nsubpath_max"]
    state.ensembles[1]["tis_set"]["n_jumps"] = n_max
    # Trigger increase condition
    for _ in range(10):
        update_epoch_stats(
            state, 1, accepted=False,
            path_length=50.0, subcycles=2, lambda_max=0.01
        )

    state.config["current"]["cstep"] = 10
    apply_epoch_ctrl(state, 10)

    assert state.ensembles[1]["tis_set"]["n_jumps"] == n_max, (
        "n_jumps must not exceed adaptive_nsubpath_max"
    )


# ---------------------------------------------------------------------------
# Test 22 — adaptive clamps at min
# ---------------------------------------------------------------------------

def test_adaptive_clamp_at_min():
    """When n_jumps is at adaptive_nsubpath_min, decrease must stay at min."""
    state = _make_adaptive_state()
    n_min = state.config["simulation"]["adaptive_nsubpath_min"]
    state.ensembles[1]["tis_set"]["n_jumps"] = n_min
    # Trigger decrease condition
    for i in range(10):
        update_epoch_stats(
            state, 1, accepted=(i < 5),
            path_length=500.0, subcycles=2, lambda_max=0.3
        )

    state.config["current"]["cstep"] = 10
    apply_epoch_ctrl(state, 10)

    assert state.ensembles[1]["tis_set"]["n_jumps"] == n_min, (
        "n_jumps must not go below adaptive_nsubpath_min"
    )


# ---------------------------------------------------------------------------
# Test 23 — adaptive TSV row has correct action, reason, avg_lambda_max, ctrl_mode
# ---------------------------------------------------------------------------

def test_adaptive_tsv_row_contains_action_and_reason(tmp_path):
    """Flushed TSV row must contain ctrl_action, ctrl_reason, avg_lambda_max,
    and ctrl_mode='adaptive'."""
    state = _make_adaptive_state()
    state.config["output"] = {"data_dir": str(tmp_path)}

    # Push stats that trigger an increase: low acc + low lambda_max
    for _ in range(4):
        update_epoch_stats(
            state, 1, accepted=False,
            path_length=50.0, subcycles=2, lambda_max=0.02
        )

    state.config["current"]["cstep"] = 10
    apply_epoch_ctrl(state, 10)

    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    assert tsv_path.exists()

    rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows) == 1
    r = rows[0]

    assert "ctrl_mode" not in r, "ctrl_mode must not appear in new public schema"
    assert r["ctrl_action"] == "inc"
    assert r["ctrl_reason"] == "low_accept_low_explore"
    avg_lmax = float(r["avg_lambda_max"])
    assert abs(avg_lmax - 0.02) < 1e-6, (
        f"avg_lambda_max should be 0.02, got {avg_lmax}"
    )


# ===========================================================================
# Integration tests — full chain through treat_output()
# ===========================================================================
#
# These tests exercise the production path:
#
#   treat_output()
#     ├── ensemble_move_counts increment (counter)
#     ├── update_epoch_stats()            (stats accumulation)
#     ├── apply_epoch_ctrl()              (epoch firing + n_jumps update)
#     ├── mirror_epoch_ctrl()             (config mirroring)
#     └── write_toml()                    (restart.toml)
#
# Real Path objects are loaded from the turtlemd double_well fixture.
# All moves use status="REJ" so PathStorage.output is never invoked.
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared fixture and helpers
# ---------------------------------------------------------------------------

# Double-well turtlemd load_copy lives here relative to the test file.
_DW_LOAD_COPY = (
    StdPath(__file__).parent.parent.parent
    / "examples/turtlemd/double_well/load_copy"
).resolve()

# 8 interfaces → 8 ensembles (0=minus, 1=[0+], 2=[1+], ..., 7=[6+])
_DW_INTERFACES = [-0.99, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, 1.0]
_DW_MOVES = ["sh", "sh", "wf", "wf", "wf", "wf", "wf", "wf"]


@pytest.fixture
def dw_setup(tmp_path, monkeypatch):
    """Copy load_copy into tmp_path, chdir there, return (state, tmp_path).

    The REPEX_state is fully initialised with real paths but NO epoch-ctrl
    keys in the config — callers add those before using the state.
    """
    shutil.copytree(str(_DW_LOAD_COPY), str(tmp_path / "load"))
    monkeypatch.chdir(tmp_path)

    cfg = {
        "simulation": {
            "interfaces": _DW_INTERFACES,
            "shooting_moves": _DW_MOVES,
            "tis_set": {
                "lambda_minus_one": False,
                "n_jumps": 2,
                "maxlength": 2000,
                "allowmaxlength": False,
                "zero_momentum": False,
                "quantis": False,
                "accept_all": False,
                "mwf_nsubpath": 3,
            },
            "seed": 0,
            "steps": 1000,
            "load_dir": str(tmp_path / "load"),
        },
        "runner": {"workers": 1},
        "current": {
            "cstep": 0,
            "size": len(_DW_INTERFACES),
            "locked": [],
            "active": list(range(len(_DW_INTERFACES))),
            "frac": {},
            "wsubcycles": [0],
            "tsubcycles": 0,
            "traj_num": len(_DW_INTERFACES),
        },
        "output": {
            "data_dir": str(tmp_path),
            "data_file": str(tmp_path / "infretis_data.txt"),
            "screen": 0,
            "delete_old": False,
            "keep_maxop_trajs": False,
        },
    }
    state = REPEX_state(cfg, minus=True)
    state.initiate_ensembles()
    paths = load_paths_from_disk(cfg)
    state.load_paths(paths)
    return state, tmp_path


def _build_md_items(state, ens_num, trial_op_max, accepted=False):
    """Construct a minimal md_items for one move on ens_num (picked key).

    ens_slot = ens_num + state._offset is the index into state._trajs.
    Uses the currently stored path as both the trial and old path (valid
    for REJ moves where the path is unchanged).
    """
    ens_slot = ens_num + state._offset
    trial_path = state._trajs[ens_slot]
    ens_dict = copy.deepcopy(state.ensembles[ens_num + 1])
    ens_dict["rgen"] = spawn_rng(state.rgen)
    status = "ACC" if accepted else "REJ"
    return {
        "picked": {
            ens_num: {
                "pn_old": trial_path.path_number,
                "traj": trial_path,
                "ens": ens_dict,
            }
        },
        "status": status,
        "pin": 0,
        "subcycles": 1,
        "trial_len": [float(trial_path.length)],
        "trial_op": [(-0.9, trial_op_max)],
        "pnum_old": [trial_path.path_number],
        "moves": [state.ensembles[ens_num + 1]["mc_move"]],
        "generated": [],
        "ens_nums": [ens_num],
        "md_start": time.time(),
    }


def _lock_ens_slot(state, ens_num):
    """Re-lock ens_num's raw slot so add_traj can unlock it in treat_output."""
    raw = ens_num + state._offset
    state._locks[raw] = 1


# ---------------------------------------------------------------------------
# Integration test 24 — scheduled mode full chain through treat_output
# ---------------------------------------------------------------------------

def test_treat_output_scheduled_full_chain(dw_setup):
    """Production path: treat_output fires scheduled epoch controller.

    Asserts:
    - move counter increments on each call
    - stats accumulate into epoch buffer
    - at epoch boundary (cstep == epoch_size): n_jumps updated, TSV written,
      buffer reset
    - mirror_epoch_ctrl writes new n_jumps into config
    - write_toml() writes restart.toml; loading it restores the new n_jumps
    """
    state, tmp_path = dw_setup

    # Add epoch ctrl config: target ensemble 2 ([1+]), epoch every 3 steps.
    # schedule: epoch 1 → vals[1%2]=vals[1]=5, epoch 2 → vals[0]=3, ...
    sim = state.config["simulation"]
    sim["epoch_nsubpath_ens"] = [2]
    sim["epoch_nsubpath_vals"] = [[3, 5]]
    sim["epoch_size"] = 3

    # Ensemble 2 uses ens_num=1 in the picked dict (ens_num + 1 == 2).
    ENS_NUM = 1

    assert state.ensembles[2]["tis_set"]["n_jumps"] == 2

    # --- simulate 3 REJ moves through treat_output ---
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=0.3))

    # Counter: 3 attempted moves for ensemble 2
    assert state.ensemble_move_counts.get(2, 0) == 3, (
        "move counter must reflect 3 attempted moves"
    )

    # Epoch fired at cstep=3 (epoch_idx=1): vals[1%2]=vals[1]=5
    assert state.ensembles[2]["tis_set"]["n_jumps"] == 5, (
        "scheduled epoch 1: n_jumps must be updated to 5"
    )

    # Stats buffer must be reset after epoch flush
    buf = state.ensemble_epoch_stats.get(2, {})
    assert buf.get("n_attempted", 0) == 0, "stats buffer must be reset after epoch"

    # Config mirrored: ensemble_nsubpath[2] reflects new value
    nsubpath = state.config["simulation"]["ensemble_nsubpath"]
    assert nsubpath[2] == 5, "mirror must write n_jumps=5 to ensemble_nsubpath[2]"

    # TSV written with correct values
    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    assert tsv_path.exists(), "epoch_summary.tsv must be created"
    rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows) == 1
    r = rows[0]
    assert int(r["epoch_idx"]) == 1
    assert r["ens_name"] == "002"
    assert int(r["n_attempted"]) == 3
    assert int(r["n_accepted"]) == 0
    assert int(r["n_jumps_old"]) == 2
    assert int(r["n_jumps_new"]) == 5
    assert "ctrl_mode" not in r, "ctrl_mode must not appear in new public schema"
    assert r["ctrl_action"] == "set"
    assert r["ctrl_reason"] == "scheduled_epoch_value"

    # write_toml() wrote restart.toml — loading it restores n_jumps
    restart_path = tmp_path / "restart.toml"
    assert restart_path.exists(), "write_toml must produce restart.toml"
    with restart_path.open("rb") as fh:
        restored_cfg = tomli.load(fh)

    assert restored_cfg["simulation"]["ensemble_nsubpath"][2] == 5, (
        "restart.toml must persist updated n_jumps"
    )
    state2 = REPEX_state(restored_cfg, minus=True)
    state2.initiate_ensembles()
    assert state2.ensembles[2]["tis_set"]["n_jumps"] == 5, (
        "restored REPEX_state must carry the updated n_jumps"
    )

    # --- simulate 3 more moves: epoch 2 fires, n_jumps cycles back to 3 ---
    for cstep in (4, 5, 6):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=0.3))

    assert state.ensembles[2]["tis_set"]["n_jumps"] == 3, (
        "scheduled epoch 2: n_jumps must cycle back to 3"
    )
    rows2 = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows2) == 2, "second epoch boundary must append a second TSV row"
    assert int(rows2[1]["n_jumps_new"]) == 3


# ---------------------------------------------------------------------------
# Integration test 25 — adaptive mode full chain through treat_output
# ---------------------------------------------------------------------------

def test_treat_output_adaptive_full_chain(dw_setup):
    """Production path: treat_output fires adaptive epoch controller.

    Uses low trial_op_max (< adaptive_lambda_gain_low) and all-REJ moves
    (acc_rate=0) so the adaptive rule should increment n_jumps.

    Asserts the same full chain as the scheduled test plus adaptive-specific
    ctrl_action / ctrl_reason values.
    """
    state, tmp_path = dw_setup

    sim = state.config["simulation"]
    sim["epoch_ctrl_mode"] = "adaptive"
    sim["epoch_nsubpath_ens"] = [2]
    sim["epoch_size"] = 3
    sim["adaptive_nsubpath_min"] = 1
    sim["adaptive_nsubpath_max"] = 8
    sim["adaptive_accept_low"] = 0.20
    sim["adaptive_accept_high"] = 0.60
    sim["adaptive_lambda_gain_low"] = 0.05
    sim["adaptive_pathlen_high"] = 400

    ENS_NUM = 1

    assert state.ensembles[2]["tis_set"]["n_jumps"] == 2

    # 3 REJ moves with very low lambda_max (< 0.05) → triggers "inc" action
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        # trial_op_max=0.01 < adaptive_lambda_gain_low=0.05
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=0.01))

    # Adaptive rule: acc_rate=0.0 < 0.20 AND avg_lambda_max=0.01 < 0.05 → inc
    assert state.ensembles[2]["tis_set"]["n_jumps"] == 3, (
        "adaptive inc: n_jumps must increase from 2 to 3"
    )

    # Move counter incremented
    assert state.ensemble_move_counts.get(2, 0) == 3

    # Buffer reset after epoch flush
    buf = state.ensemble_epoch_stats.get(2, {})
    assert buf.get("n_attempted", 0) == 0

    # Config mirrored
    assert state.config["simulation"]["ensemble_nsubpath"][2] == 3

    # TSV row has correct adaptive fields
    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows) == 1
    r = rows[0]
    assert "ctrl_mode" not in r, "ctrl_mode must not appear in new public schema"
    assert r["ctrl_action"] == "inc"
    assert r["ctrl_reason"] == "low_accept_low_explore"
    assert int(r["n_accepted"]) == 0
    assert int(r["n_jumps_old"]) == 2
    assert int(r["n_jumps_new"]) == 3
    avg_lmax = float(r["avg_lambda_max"])
    assert abs(avg_lmax - 0.01) < 1e-9

    # Restart round-trip
    restart_path = tmp_path / "restart.toml"
    assert restart_path.exists()
    with restart_path.open("rb") as fh:
        restored_cfg = tomli.load(fh)
    assert restored_cfg["simulation"]["ensemble_nsubpath"][2] == 3
    state2 = REPEX_state(restored_cfg, minus=True)
    state2.initiate_ensembles()
    assert state2.ensembles[2]["tis_set"]["n_jumps"] == 3, (
        "restored state must carry adaptive-updated n_jumps"
    )

# ---------------------------------------------------------------------------
# softmax_dirichlet helpers — imported for unit tests
# ---------------------------------------------------------------------------

import math

from infretis.core.epoch_ctrl import (
    _bounded_lambda_obs,
    _compute_epoch_reward,
    _compute_q,
    _empty_stats,
    _epoch_sample,
    _init_softmax_ctrl_state,
    _normalize_reward,
    _obs_bounds_for_ensemble,
    _softmax,
)


# ---------------------------------------------------------------------------
# Shared softmax_dirichlet config factory
# ---------------------------------------------------------------------------

def _sd_config_base() -> dict:
    """Minimal valid config for softmax_dirichlet validation tests."""
    base = _check_config_base()
    base["simulation"].update(
        {
            "epoch_ctrl_mode": "softmax_dirichlet",
            "epoch_mode": "per_ensemble_moves",
            "epoch_count": "attempted",
            "epoch_move_k": 5,
            "epoch_nsubpath_ens": [1],  # ensemble 1 is "wf" in _check_config_base
            "epoch_nsubpath_choices": [[1, 2, 3, 4]],
            "softmax_eta0": 0.5,
            "softmax_beta": 0.7,
            "softmax_tau": 1.0,
            "softmax_explore_floor": 0.05,
            "softmax_init": "uniform",
            "reward_proxy": "lambda_vs_subcycles_v1",
            "reward_eps_gain": 1e-12,
            "reward_eps_cost": 1e-12,
            "epoch_ctrl_seed": 42,
        }
    )
    return base


def _sd_sim_keys() -> dict:
    """Softmax config keys for injection into dw_setup state.config."""
    return {
        "epoch_ctrl_mode": "softmax_dirichlet",
        "epoch_mode": "per_ensemble_moves",
        "epoch_count": "attempted",
        "epoch_move_k": 3,
        "epoch_nsubpath_ens": [2],
        "epoch_nsubpath_choices": [[1, 2, 3, 4]],
        "softmax_eta0": 0.5,
        "softmax_beta": 0.7,
        "softmax_tau": 1.0,
        "softmax_explore_floor": 0.05,
        "softmax_init": "uniform",
        "reward_proxy": "lambda_vs_subcycles_v1",
        "reward_eps_gain": 1e-12,
        "reward_eps_cost": 1e-12,
        "epoch_ctrl_seed": 42,
    }


# ---------------------------------------------------------------------------
# Unit test 26 — missing epoch_nsubpath_choices rejected
# ---------------------------------------------------------------------------

def test_sd_validation_rejects_missing_choices():
    """Missing epoch_nsubpath_choices must raise TOMLConfigError."""
    cfg = _sd_config_base()
    del cfg["simulation"]["epoch_nsubpath_choices"]
    with pytest.raises(TOMLConfigError, match="epoch_nsubpath_choices"):
        check_config(cfg)


# ---------------------------------------------------------------------------
# Unit test 27 — wrong epoch_mode rejected
# ---------------------------------------------------------------------------

def test_sd_validation_rejects_wrong_epoch_mode():
    """epoch_mode='global_step' must be rejected for softmax_dirichlet."""
    cfg = _sd_config_base()
    cfg["simulation"]["epoch_mode"] = "global_step"
    with pytest.raises(TOMLConfigError, match="per_ensemble_moves"):
        check_config(cfg)


# ---------------------------------------------------------------------------
# Unit test 28 — bad softmax_beta rejected
# ---------------------------------------------------------------------------

def test_sd_validation_rejects_bad_softmax_params():
    """softmax_beta <= 0.5 must raise TOMLConfigError."""
    cfg = _sd_config_base()
    cfg["simulation"]["softmax_beta"] = 0.3
    with pytest.raises(TOMLConfigError, match="softmax_beta"):
        check_config(cfg)


# ---------------------------------------------------------------------------
# Unit test 29 — q sums to one and is strictly positive
# ---------------------------------------------------------------------------

def test_sd_q_sums_to_one_and_strictly_positive():
    """_compute_q must return a valid probability vector for any logits."""
    # Uniform logits
    q = _compute_q([0.0, 0.0, 0.0], tau=1.0, eps_explore=0.05)
    assert abs(q.sum() - 1.0) < 1e-12
    assert (q > 0).all()
    # Non-uniform logits
    q2 = _compute_q([2.0, 0.0, -2.0, 1.0], tau=1.0, eps_explore=0.05)
    assert abs(q2.sum() - 1.0) < 1e-12
    assert (q2 > 0).all()
    # Zero explore floor
    q3 = _compute_q([0.0, 1.0], tau=1.0, eps_explore=0.0)
    assert abs(q3.sum() - 1.0) < 1e-12
    assert (q3 > 0).all()


# ---------------------------------------------------------------------------
# Unit test 30 — explore-floor mixture formula correct
# ---------------------------------------------------------------------------

def test_sd_explore_floor_mixture():
    """_compute_q result equals (1-eps)*softmax(logits/tau) + eps/K."""
    logits = [1.0, 0.5, -0.5]
    tau = 0.8
    eps = 0.1
    q = _compute_q(logits, tau, eps)
    p = _softmax(np.array(logits) / tau)
    expected = (1.0 - eps) * p + eps / len(logits)
    np.testing.assert_allclose(q, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Unit test 31 — sampling is deterministic
# ---------------------------------------------------------------------------

def test_sd_deterministic_sampling():
    """_epoch_sample returns identical choice on repeated calls."""
    q = _compute_q([0.0, 0.0, 0.0, 0.0], tau=1.0, eps_explore=0.0)
    seed, ens_i, epoch_idx = 42, 2, 7
    c1 = _epoch_sample(q, seed, ens_i, epoch_idx)
    c2 = _epoch_sample(q, seed, ens_i, epoch_idx)
    assert c1 == c2
    # Different epoch_idx should (almost surely) give independent draws
    # — just verify the function runs without error; equality not guaranteed.
    _epoch_sample(q, seed, ens_i, epoch_idx + 1)


# ---------------------------------------------------------------------------
# Unit test 32 — logit update increases chosen action probability
# ---------------------------------------------------------------------------

def test_sd_logit_update_increases_chosen_prob():
    """Positive reward applied to choice_idx must increase softmax prob."""
    logits = [0.0, 0.0, 0.0, 0.0, 0.0]
    tau = 1.0
    eps = 0.0
    old_q = _compute_q(logits, tau, eps)
    choice_idx = 2
    reward = 1.0
    eta = 0.5
    new_logits = list(logits)
    new_logits[choice_idx] += eta * reward / old_q[choice_idx]
    old_p = _softmax(np.array(logits) / tau)
    new_p = _softmax(np.array(new_logits) / tau)
    assert new_logits[choice_idx] > logits[choice_idx]
    assert new_p[choice_idx] > old_p[choice_idx]


# ---------------------------------------------------------------------------
# Unit test 33 — zero-reward update leaves logits unchanged
# ---------------------------------------------------------------------------

def test_sd_logit_update_neutral_reward():
    """Reward=0.0 must leave logits unchanged to float precision."""
    logits = [0.1, -0.2, 0.5]
    tau = 1.0
    eps = 0.05
    old_q = _compute_q(logits, tau, eps)
    choice_idx = 1
    reward = 0.0
    eta = 0.5
    new_logits = list(logits)
    new_logits[choice_idx] += eta * reward / old_q[choice_idx]
    assert new_logits == logits


# ---------------------------------------------------------------------------
# Unit test 34 — no cross-ensemble state leakage
# ---------------------------------------------------------------------------

def test_sd_no_cross_ensemble_leakage():
    """Updating ens 2 ctrl_state must not touch ens 3 ctrl_state."""
    cs2 = _init_softmax_ctrl_state([1, 2, 3], 2, {"softmax_init": "uniform"})
    cs3 = _init_softmax_ctrl_state([2, 4], 2, {"softmax_init": "uniform"})
    sd = {2: cs2, 3: cs3}

    # Simulate logit update on ens 2 only
    sd[2]["logits"][0] += 1.0
    sd[2]["update_count"] = 1

    assert sd[3]["logits"] == [0.0, 0.0]
    assert sd[3]["update_count"] == 0


# ---------------------------------------------------------------------------
# Integration test 35 — ctrl only fires at epoch boundary
# ---------------------------------------------------------------------------

def test_sd_only_fires_at_boundary(dw_setup):
    """k-1 moves must not trigger ctrl; k-th move must."""
    state, tmp_path = dw_setup
    state.config["simulation"].update(_sd_sim_keys())
    ENS_NUM = 1  # ensemble index 2

    # k-1 = 2 moves: ctrl_state not yet populated
    for cstep in (1, 2):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=0.3))

    assert 2 not in state.softmax_ctrl_state, (
        "ctrl_state must not be set before epoch boundary"
    )

    # k-th move: ctrl fires
    state.cstep = 3
    _lock_ens_slot(state, ENS_NUM)
    state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=0.3))

    assert 2 in state.softmax_ctrl_state, (
        "ctrl_state must be populated after first epoch boundary"
    )


# ---------------------------------------------------------------------------
# Integration test 36 — n_jumps does not change mid-epoch
# ---------------------------------------------------------------------------

def test_sd_n_jumps_fixed_within_epoch(dw_setup):
    """n_jumps must stay constant for all moves within one epoch."""
    state, tmp_path = dw_setup
    state.config["simulation"].update(_sd_sim_keys())
    ENS_NUM = 1

    # Run k moves → epoch 1 fires, n_jumps set to some value
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=0.3))

    n_after_ep1 = state.ensembles[2]["tis_set"].get("n_jumps")
    choices = state.config["simulation"]["epoch_nsubpath_choices"][0]
    assert n_after_ep1 in choices, "n_jumps after epoch 1 must be in choices"

    # Mid epoch 2: one more move must not change n_jumps
    state.cstep = 4
    _lock_ens_slot(state, ENS_NUM)
    state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=0.3))

    assert state.ensembles[2]["tis_set"].get("n_jumps") == n_after_ep1, (
        "n_jumps must not change mid-epoch"
    )


# ---------------------------------------------------------------------------
# Integration test 37 — first boundary populates state and writes TSV
# ---------------------------------------------------------------------------

def test_sd_boundary_updates_and_samples(dw_setup):
    """After epoch 1 fires: TSV written, ctrl_state populated, n_jumps valid."""
    state, tmp_path = dw_setup
    state.config["simulation"].update(_sd_sim_keys())
    ENS_NUM = 1

    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=0.3))

    # TSV written
    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    assert tsv_path.exists(), "epoch_summary.tsv must be created at epoch boundary"
    rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows) == 1
    r = rows[0]
    assert "ctrl_mode" not in r, "ctrl_mode must not appear in new public schema"
    assert r["ctrl_action"] in ("sample_hold", "sample_set")
    assert r["ctrl_reason"] == "softmax_dirichlet_epoch_update"
    # Softmax internals must NOT be in the summary TSV
    assert "reward_raw" not in r, "reward_raw must be in debug TSV only"
    assert "choice_idx" not in r, "choice_idx must be in debug TSV only"
    assert "probs_json" not in r, "probs_json must be in debug TSV only"

    # Debug TSV must exist with the softmax internals
    debug_path = tmp_path / _EPOCH_SOFTMAX_DEBUG_FNAME
    assert debug_path.exists(), "epoch_softmax_debug.tsv must be created at epoch boundary"
    debug_rows = list(csv.DictReader(debug_path.open(), delimiter="\t"))
    assert len(debug_rows) == 1
    dr = debug_rows[0]
    assert "choice_idx" in dr
    assert "probs_json" in dr
    assert dr["reward_raw"] == "nan"  # first epoch: no prior reward

    # ctrl_state populated
    assert 2 in state.softmax_ctrl_state
    cs = state.softmax_ctrl_state[2]
    assert cs["initialized"] is True
    assert cs["last_choice_idx"] is not None
    assert cs["update_count"] == 0  # first epoch: no logit update applied

    # n_jumps is in choices
    choices = state.config["simulation"]["epoch_nsubpath_choices"][0]
    assert state.ensembles[2]["tis_set"].get("n_jumps") in choices


# ---------------------------------------------------------------------------
# Integration test 38 — restart roundtrip preserves full epoch stats
# ---------------------------------------------------------------------------

def test_sd_restart_roundtrip(dw_setup):
    """Stop mid-epoch; restart.toml must contain softmax state and epoch
    stats; restored state continues correctly to epoch 2 boundary."""
    state, tmp_path = dw_setup
    state.config["simulation"].update(_sd_sim_keys())
    ENS_NUM = 1

    # Epoch 1: 3 moves → epoch fires, ctrl_state initialised
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=0.3))

    assert 2 in state.softmax_ctrl_state
    cs_ep1 = copy.deepcopy(state.softmax_ctrl_state[2])
    assert cs_ep1["update_count"] == 0
    assert cs_ep1["last_choice_idx"] is not None

    # Mid epoch 2: 2 moves (no epoch fire)
    for cstep in (4, 5):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=0.3))

    buf_before = state.ensemble_epoch_stats.get(2, {})
    assert buf_before.get("n_attempted", 0) == 2, (
        "partial epoch buffer must have 2 moves after mid-epoch stop"
    )

    # Read restart.toml written by the last treat_output
    restart_path = tmp_path / "restart.toml"
    assert restart_path.exists()
    with restart_path.open("rb") as fh:
        saved_cfg = tomli.load(fh)

    # Verify serialized softmax state
    assert "softmax_ctrl_state" in saved_cfg["simulation"], (
        "mirror must persist softmax_ctrl_state"
    )
    assert "softmax_epoch_stats" in saved_cfg["simulation"], (
        "mirror must persist softmax_epoch_stats for mid-epoch reward"
    )
    saved_cs = saved_cfg["simulation"]["softmax_ctrl_state"]["2"]
    assert saved_cs["update_count"] == 0
    assert saved_cs["last_choice_idx"] == cs_ep1["last_choice_idx"]
    assert saved_cs["logits"] == cs_ep1["logits"]
    saved_stats = saved_cfg["simulation"]["softmax_epoch_stats"]["2"]
    assert saved_stats["n_attempted"] == 2

    # Restore state from restart.toml
    state2 = REPEX_state(saved_cfg, minus=True)
    state2.initiate_ensembles()
    paths = load_paths_from_disk(saved_cfg)
    state2.load_paths(paths)

    # Verify restored ctrl_state matches pre-stop state
    assert 2 in state2.softmax_ctrl_state
    cs_restored = state2.softmax_ctrl_state[2]
    assert cs_restored["update_count"] == cs_ep1["update_count"]
    assert cs_restored["last_choice_idx"] == cs_ep1["last_choice_idx"]
    assert cs_restored["logits"] == cs_ep1["logits"]

    # Verify partial epoch stats survived the restart
    buf2 = state2.ensemble_epoch_stats.get(2, {})
    assert buf2["n_attempted"] == 2, (
        "partial epoch stats must be restored from softmax_epoch_stats"
    )

    # Complete epoch 2: 1 more move → epoch fires at count=6 (6 % 3 == 0)
    state2.cstep = 6
    _lock_ens_slot(state2, ENS_NUM)
    state2.treat_output(_build_md_items(state2, ENS_NUM, trial_op_max=0.3))

    # With the EMA baseline, epoch 2 initializes reward_ema but does NOT
    # apply a logit update (update_count stays 0).  The first real logit
    # update fires at epoch 3 once a centerable baseline exists.
    assert state2.softmax_ctrl_state[2]["update_count"] == 0, (
        "epoch 2 boundary initializes EMA baseline but must not update logits yet"
    )
    assert not math.isnan(state2.softmax_ctrl_state[2]["reward_ema"]), (
        "epoch 2 must set reward_ema baseline"
    )
    choices = saved_cfg["simulation"]["epoch_nsubpath_choices"][0]
    nj2 = state2.ensembles[2]["tis_set"].get("n_jumps")
    assert nj2 in choices, "n_jumps after epoch 2 must be in admissible set"


# ===========================================================================
# Tests 39–49 — empirical_dirichlet_lambda_v1 reward proxy (Phase 2a)
# ===========================================================================

# ---------------------------------------------------------------------------
# Unit test 39 — _bounded_lambda_obs clips and maps correctly
# ---------------------------------------------------------------------------

def test_sd_empirical_obs_clips_to_unit_interval():
    """_bounded_lambda_obs: below lambda_i→0, above lambda_upper→1, interior correct."""
    li, lu = 0.2, 0.8

    # Below lambda_i → 0
    assert _bounded_lambda_obs(0.0, li, lu) == 0.0
    assert _bounded_lambda_obs(0.19, li, lu) == 0.0

    # Above lambda_upper → 1
    assert _bounded_lambda_obs(1.0, li, lu) == 1.0
    assert _bounded_lambda_obs(0.81, li, lu) == 1.0

    # Interior maps linearly
    mid = (li + lu) / 2  # 0.5
    obs = _bounded_lambda_obs(mid, li, lu)
    assert abs(obs - 0.5) < 1e-12

    # Interior: 3/4 of the way
    val = li + 0.75 * (lu - li)  # 0.65
    obs2 = _bounded_lambda_obs(val, li, lu)
    assert abs(obs2 - 0.75) < 1e-12

    # lambda_upper <= lambda_i must raise
    with pytest.raises(ValueError, match="lambda_upper"):
        _bounded_lambda_obs(0.5, 0.5, 0.5)
    with pytest.raises(ValueError, match="lambda_upper"):
        _bounded_lambda_obs(0.5, 0.8, 0.2)


# ---------------------------------------------------------------------------
# Unit test 39b — _obs_bounds_for_ensemble returns lambda_B when no cap set
# ---------------------------------------------------------------------------

def test_sd_empirical_obs_uses_lambda_B_when_no_cap():
    """_obs_bounds_for_ensemble: no interface_cap → lambda_upper = ens['interfaces'][2]."""
    state = _make_state()
    state.ensembles[1]["interfaces"] = (float("-inf"), 0.3, 0.8)
    state.ensembles[1]["tis_set"].pop("interface_cap", None)

    lambda_i, lambda_upper = _obs_bounds_for_ensemble(state, 1)
    assert abs(lambda_i - 0.3) < 1e-12
    assert abs(lambda_upper - 0.8) < 1e-12


# ---------------------------------------------------------------------------
# Unit test 39c — _obs_bounds_for_ensemble uses interface_cap when present
# ---------------------------------------------------------------------------

def test_sd_empirical_obs_uses_interface_cap_when_present():
    """_obs_bounds_for_ensemble: interface_cap in tis_set → lambda_upper = cap."""
    state = _make_state()
    state.ensembles[1]["interfaces"] = (float("-inf"), 0.3, 0.8)
    state.ensembles[1]["tis_set"]["interface_cap"] = 0.6

    lambda_i, lambda_upper = _obs_bounds_for_ensemble(state, 1)
    assert abs(lambda_i - 0.3) < 1e-12
    assert abs(lambda_upper - 0.6) < 1e-12


# ---------------------------------------------------------------------------
# Unit test 40 — gain accumulator accumulates correctly over a sequence
# ---------------------------------------------------------------------------

def test_sd_empirical_gain_accumulator_sequence():
    """update_epoch_stats accumulates gain_sq_sum for obs sequence [0.1,0.4,0.4,0.9]."""
    state = _make_state()
    state.config["simulation"]["reward_proxy"] = "empirical_dirichlet_lambda_v1"
    # Override ensemble 1 interfaces: lambda_i=0.0, lambda_upper=1.0 → obs = lambda_max
    state.ensembles[1]["interfaces"] = (float("-inf"), 0.0, 1.0)
    # Pre-populate ctrl_state so the empirical block runs (last_obs starts nan)
    state.softmax_ctrl_state[1] = {"last_obs": float("nan")}

    lambda_max_vals = [0.1, 0.4, 0.4, 0.9]
    for lm in lambda_max_vals:
        update_epoch_stats(state, 1, accepted=True, path_length=10.0, subcycles=2, lambda_max=lm)

    buf = state.ensemble_epoch_stats[1]
    cs = state.softmax_ctrl_state[1]
    # Move 1: last_obs initialized to 0.1, no increment.
    # Move 2: delta=0.3, gain_sq_sum+=0.09, gain_sq_n=1.
    # Move 3: delta=0.0, gain_sq_sum+=0.0, gain_sq_n=2.
    # Move 4: delta=0.5, gain_sq_sum+=0.25, gain_sq_n=3.
    expected_sq_sum = (0.4 - 0.1)**2 + (0.4 - 0.4)**2 + (0.9 - 0.4)**2  # 0.34
    assert abs(buf["gain_sq_sum"] - expected_sq_sum) < 1e-12
    assert buf["gain_sq_n"] == 3
    assert abs(cs["last_obs"] - 0.9) < 1e-12


# ---------------------------------------------------------------------------
# Unit test 41 — _compute_epoch_reward uses empirical formula from buffer
# ---------------------------------------------------------------------------

def test_sd_empirical_reward_from_buffer():
    """_compute_epoch_reward with empirical proxy returns total_gain / (total_cost + offset)."""
    sim = {
        "reward_proxy": "empirical_dirichlet_lambda_v1",
    }
    buf = _empty_stats()
    buf["gain_sq_sum"] = 0.08
    buf["gain_sq_n"] = 2
    buf["subcycles_sum"] = 3
    buf["subcycles_n"] = 1

    reward = _compute_epoch_reward(buf, sim)
    # New formula: total_gain / (total_cost + cost_offset)
    expected = 0.08 / (3 + 1e-12)
    assert abs(reward - expected) < 1e-20


# ---------------------------------------------------------------------------
# Unit test 42 — first accepted move initialises prev_obs, no increment
# ---------------------------------------------------------------------------

def test_sd_empirical_first_obs_no_increment():
    """First accepted move: last_obs set in ctrl_state, gain_sq_n stays 0."""
    state = _make_state()
    state.config["simulation"]["reward_proxy"] = "empirical_dirichlet_lambda_v1"
    state.ensembles[1]["interfaces"] = (float("-inf"), 0.0, 1.0)
    # Pre-populate ctrl_state so the empirical block runs
    state.softmax_ctrl_state[1] = {"last_obs": float("nan")}

    update_epoch_stats(state, 1, accepted=True, path_length=10.0, subcycles=2, lambda_max=0.5)

    buf = state.ensemble_epoch_stats[1]
    cs = state.softmax_ctrl_state[1]
    assert abs(cs["last_obs"] - 0.5) < 1e-12
    assert buf["gain_sq_n"] == 0


# ---------------------------------------------------------------------------
# Unit test 43 — zero gain when no increments recorded
# ---------------------------------------------------------------------------

def test_sd_empirical_zero_reward_when_no_increments():
    """gain_sq_n=0 < reward_min_gain_count=1 → reward is nan."""
    sim = {
        "reward_proxy": "empirical_dirichlet_lambda_v1",
    }
    buf = _empty_stats()  # gain_sq_n = 0
    buf["subcycles_sum"] = 3
    buf["subcycles_n"] = 1

    reward = _compute_epoch_reward(buf, sim)
    assert math.isnan(reward), f"expected nan for gain_sq_n=0, got {reward}"


# ---------------------------------------------------------------------------
# Unit test 44 — validation accepts empirical_dirichlet_lambda_v1 proxy
# ---------------------------------------------------------------------------

def test_sd_validation_accepts_empirical_proxy():
    """Valid empirical_dirichlet_lambda_v1 config passes check_config."""
    cfg = _sd_config_base()
    cfg["simulation"]["reward_proxy"] = "empirical_dirichlet_lambda_v1"
    check_config(cfg)  # must not raise


# ---------------------------------------------------------------------------
# Unit test 45 — validation rejects bad upper bound for empirical proxy
# ---------------------------------------------------------------------------

def test_sd_validation_rejects_bad_cap_for_empirical_proxy():
    """interface_cap <= lambda_i raises; no cap with lambda_B > lambda_i succeeds."""
    # Bad: interface_cap is below lambda_i (= ifaces[0] = 0.0 for ens_i=1)
    cfg = _sd_config_base()
    cfg["simulation"]["reward_proxy"] = "empirical_dirichlet_lambda_v1"
    cfg["simulation"]["tis_set"]["interface_cap"] = -0.5  # < lambda_i=0.0
    with pytest.raises(TOMLConfigError):
        check_config(cfg)

    # Good: no cap, lambda_B = ifaces[-1] = 0.6 > lambda_i = 0.0
    cfg2 = _sd_config_base()
    cfg2["simulation"]["reward_proxy"] = "empirical_dirichlet_lambda_v1"
    check_config(cfg2)  # must not raise


# ===========================================================================
# Integration tests 46–49
# ===========================================================================

# ---------------------------------------------------------------------------
# Integration test 46 — rejected move gives zero gain increment
# ---------------------------------------------------------------------------

def test_sd_empirical_rejected_move_gives_zero_increment(dw_setup):
    """ACC then REJ: gain_sq_n==1 but gain_sq_sum==0 after the rejection."""
    state, tmp_path = dw_setup
    keys = _sd_sim_keys()
    keys["reward_proxy"] = "empirical_dirichlet_lambda_v1"
    state.config["simulation"].update(keys)
    # Pre-populate ctrl_state so the empirical block runs (last_obs starts nan)
    state.softmax_ctrl_state[2] = {"last_obs": float("nan")}

    # First: ACC move — initialises last_obs in ctrl_state
    update_epoch_stats(state, 2, accepted=True, path_length=10.0, subcycles=1, lambda_max=-0.4)
    cs = state.softmax_ctrl_state[2]
    buf = state.ensemble_epoch_stats.get(2, {})
    assert not math.isnan(cs.get("last_obs", float("nan"))), "last_obs must be set after ACC"
    assert buf.get("gain_sq_n", 0) == 0, "no increment on first ACC move"

    # Second: REJ move — curr_obs = last_obs, delta = 0
    update_epoch_stats(state, 2, accepted=False, path_length=10.0, subcycles=1, lambda_max=-0.99)
    buf2 = state.ensemble_epoch_stats.get(2, {})
    assert buf2.get("gain_sq_n", 0) == 1, "REJ must still increment gain_sq_n"
    assert abs(buf2.get("gain_sq_sum", float("nan")) - 0.0) < 1e-15, (
        "delta=0 for REJ: gain_sq_sum must stay 0"
    )


# ---------------------------------------------------------------------------
# Integration test 47 — boundary reward matches empirical formula
# ---------------------------------------------------------------------------

def test_sd_empirical_reward_used_at_boundary(dw_setup):
    """3 ACC moves with known lambda_max; epoch-2 reward matches empirical formula."""
    state, tmp_path = dw_setup
    keys = _sd_sim_keys()
    keys["reward_proxy"] = "empirical_dirichlet_lambda_v1"
    state.config["simulation"].update(keys)
    ENS_NUM = 1  # ensemble index 2

    # Epoch 1: 3 REJ moves through treat_output → ctrl_state initialized, buffer reset
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=-0.3))

    assert 2 in state.softmax_ctrl_state, "epoch 1 must have initialized ctrl_state"

    # Read ensemble 2 bounds for computing expected observables
    lambda_i = state.ensembles[2]["interfaces"][1]
    lambda_upper = state.ensembles[2]["interfaces"][2]
    denom = lambda_upper - lambda_i

    # Choose 3 lambda_max values → observables 0.2, 0.5, 0.7
    lm1 = lambda_i + 0.2 * denom
    lm2 = lambda_i + 0.5 * denom
    lm3 = lambda_i + 0.7 * denom
    h1 = _bounded_lambda_obs(lm1, lambda_i, lambda_upper)  # ≈ 0.2
    h2 = _bounded_lambda_obs(lm2, lambda_i, lambda_upper)  # ≈ 0.5
    h3 = _bounded_lambda_obs(lm3, lambda_i, lambda_upper)  # ≈ 0.7

    # Push 3 ACC moves directly into epoch-2 stats buffer
    for lm in (lm1, lm2, lm3):
        update_epoch_stats(state, 2, accepted=True, path_length=10.0, subcycles=1, lambda_max=lm)

    # Trigger epoch 2 boundary by setting move count to 2*k
    state.ensemble_move_counts[2] = 6  # 2 * epoch_move_k=3
    apply_epoch_ctrl(state, 0)

    # Read debug TSV epoch-2 row (reward_raw lives in the debug file)
    debug_path = tmp_path / _EPOCH_SOFTMAX_DEBUG_FNAME
    assert debug_path.exists(), "epoch_softmax_debug.tsv must exist after epoch boundaries"
    debug_rows = list(csv.DictReader(debug_path.open(), delimiter="\t"))
    assert len(debug_rows) == 2, "must have epoch-1 and epoch-2 debug rows"
    reward_val = float(debug_rows[1]["reward_raw"])

    # Expected empirical reward: gain from 3 moves (1st initializes, 2nd and 3rd give deltas)
    # gain_sq_sum = (h2-h1)^2 + (h3-h2)^2, gain_sq_n = 2
    # New formula: total_gain / (total_cost + cost_offset)
    gain_sq_sum = (h2 - h1) ** 2 + (h3 - h2) ** 2
    cost_offset = 1e-12
    expected = gain_sq_sum / (3.0 + cost_offset)  # subcycles_sum=3 (3 moves, subcycles=1 each)
    assert abs(reward_val - expected) < 1e-6, (
        f"Empirical reward mismatch: got {reward_val}, expected {expected}"
    )

    # Phase-1 reward (avg_lambda_max / avg_sub) would differ
    eps = 1e-12
    phase1_avg_lm = (lm1 + lm2 + lm3) / 3
    phase1_reward = (phase1_avg_lm + eps) / (1.0 + eps)
    assert abs(reward_val - phase1_reward) > 1e-6, (
        "Reward must use empirical formula, not phase-1 lambda-avg formula"
    )


# ---------------------------------------------------------------------------
# Integration test 48 — restart roundtrip preserves gain state
# ---------------------------------------------------------------------------

def test_sd_empirical_restart_roundtrip_preserves_gain_state(dw_setup):
    """Stop mid-epoch; restart.toml preserves last_obs/gain_sq_*; restore continues."""
    state, tmp_path = dw_setup
    keys = _sd_sim_keys()
    keys["reward_proxy"] = "empirical_dirichlet_lambda_v1"
    state.config["simulation"].update(keys)
    ENS_NUM = 1  # ensemble index 2

    # Epoch 1: 3 REJ moves → ctrl_state initialized (last_obs=nan), buffer reset
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=-0.3))

    assert 2 in state.softmax_ctrl_state

    # Mid-epoch 2: 2 ACC moves to build gain state
    lambda_i = state.ensembles[2]["interfaces"][1]
    lambda_upper = state.ensembles[2]["interfaces"][2]
    denom = lambda_upper - lambda_i
    lm1 = lambda_i + 0.3 * denom
    lm2 = lambda_i + 0.6 * denom

    update_epoch_stats(state, 2, accepted=True, path_length=10.0, subcycles=1, lambda_max=lm1)
    update_epoch_stats(state, 2, accepted=True, path_length=10.0, subcycles=1, lambda_max=lm2)

    buf_before = state.ensemble_epoch_stats.get(2, {})
    cs_before = state.softmax_ctrl_state[2]
    assert buf_before.get("gain_sq_n", 0) == 1, "2 ACC moves → 1 gain increment"
    assert not math.isnan(cs_before.get("last_obs", float("nan")))
    last_obs_before = cs_before["last_obs"]
    gain_sq_n_before = buf_before["gain_sq_n"]
    gain_sq_sum_before = buf_before["gain_sq_sum"]

    # Serialize via mirror_epoch_ctrl
    mirror_epoch_ctrl(state, state.config)
    saved_cs = state.config["simulation"]["softmax_ctrl_state"]["2"]
    saved_stats = state.config["simulation"]["softmax_epoch_stats"]["2"]
    assert saved_stats["gain_sq_n"] == gain_sq_n_before
    assert not math.isnan(saved_cs.get("last_obs", float("nan")))

    # TOML roundtrip
    restart_path = tmp_path / "restart2.toml"
    with restart_path.open("wb") as fh:
        tomli_w.dump(state.config, fh)
    with restart_path.open("rb") as fh:
        saved_cfg = tomli.load(fh)

    # Restore state
    state2 = REPEX_state(saved_cfg, minus=True)
    state2.initiate_ensembles()
    paths = load_paths_from_disk(saved_cfg)
    state2.load_paths(paths)

    # Verify restored gain state
    buf2 = state2.ensemble_epoch_stats.get(2, {})
    cs2 = state2.softmax_ctrl_state[2]
    assert buf2.get("gain_sq_n", 0) == gain_sq_n_before, "gain_sq_n must survive restart"
    assert not math.isnan(cs2.get("last_obs", float("nan"))), "last_obs must survive restart"
    assert abs(cs2.get("last_obs", 0.0) - last_obs_before) < 1e-10
    assert abs(buf2.get("gain_sq_sum", 0.0) - gain_sq_sum_before) < 1e-10

    # Complete epoch 2: 1 more ACC move + trigger epoch boundary
    lm3 = lambda_i + 0.8 * denom
    update_epoch_stats(state2, 2, accepted=True, path_length=10.0, subcycles=1, lambda_max=lm3)
    state2.ensemble_move_counts[2] = 6  # trigger epoch 2
    apply_epoch_ctrl(state2, 0)

    # With the EMA baseline, epoch 2 initializes reward_ema but does NOT
    # apply a logit update.  The first real logit update fires at epoch 3.
    assert state2.softmax_ctrl_state[2]["update_count"] == 0, (
        "epoch 2 must initialize EMA baseline without applying a logit update"
    )
    # reward_raw and reward_eff are in the debug TSV
    debug_path = tmp_path / _EPOCH_SOFTMAX_DEBUG_FNAME
    assert debug_path.exists(), "epoch_softmax_debug.tsv must exist"
    debug_rows = list(csv.DictReader(debug_path.open(), delimiter="\t"))
    assert len(debug_rows) == 2
    dr2 = debug_rows[1]
    # reward_raw is finite (the empirical proxy computed from accumulated stats)
    assert dr2["reward_raw"] != "nan", "epoch-2 reward_raw must not be nan"
    # reward_eff is 0.0 on Case 2 (baseline initialization)
    assert float(dr2["reward_eff"]) == pytest.approx(0.0), (
        "epoch-2 reward_eff must be 0.0 (EMA baseline initialization)"
    )


# ---------------------------------------------------------------------------
# Integration test 49 — phase-1 proxy (lambda_vs_subcycles_v1) still works
# ---------------------------------------------------------------------------

def test_sd_phase1_proxy_still_works(dw_setup):
    """lambda_vs_subcycles_v1 proxy: epoch fires, n_jumps updated, no regression."""
    state, tmp_path = dw_setup
    state.config["simulation"].update(_sd_sim_keys())  # default phase-1 proxy
    ENS_NUM = 1  # ensemble index 2

    # 3 REJ moves → epoch 1 fires
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=0.3))

    # ctrl_state populated, n_jumps in choices
    assert 2 in state.softmax_ctrl_state, "ctrl_state must be populated after epoch 1"
    choices = state.config["simulation"]["epoch_nsubpath_choices"][0]
    nj = state.ensembles[2]["tis_set"].get("n_jumps")
    assert nj in choices, "n_jumps must be in admissible choices"

    # TSV row written with correct mode
    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    assert tsv_path.exists()
    rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows) == 1
    assert "ctrl_mode" not in rows[0], "ctrl_mode must not appear in new public schema"
    assert "reward_raw" not in rows[0], "reward_raw must be in debug TSV only"


# ===========================================================================
# Tests 50–59 — ref_softmax refinements
# ===========================================================================


# ---------------------------------------------------------------------------
# Test T1 — last_obs survives epoch boundary
# ---------------------------------------------------------------------------

def test_last_obs_survives_epoch_boundary(dw_setup):
    """After epoch fires, ctrl_state['last_obs'] retains the value set during
    pre-boundary moves and is not wiped by the buffer reset."""
    state, tmp_path = dw_setup
    keys = _sd_sim_keys()
    keys["reward_proxy"] = "empirical_dirichlet_lambda_v1"
    state.config["simulation"].update(keys)
    ENS_NUM = 1  # ensemble index 2

    # Epoch 1: 3 REJ moves through treat_output → ctrl_state initialized
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=-0.3))

    assert 2 in state.softmax_ctrl_state
    # After epoch 1 fires (all REJ moves), last_obs should still be nan
    # because REJ moves with last_obs=nan don't initialize it
    assert math.isnan(state.softmax_ctrl_state[2].get("last_obs", float("nan")))

    # Now push one ACC move into epoch 2 buffer — should set last_obs
    lambda_i = state.ensembles[2]["interfaces"][1]
    lambda_upper = state.ensembles[2]["interfaces"][2]
    lm = lambda_i + 0.5 * (lambda_upper - lambda_i)
    update_epoch_stats(state, 2, accepted=True, path_length=10.0, subcycles=1, lambda_max=lm)

    expected_obs = _bounded_lambda_obs(lm, lambda_i, lambda_upper)
    assert abs(state.softmax_ctrl_state[2]["last_obs"] - expected_obs) < 1e-12, (
        "last_obs must be set after first ACC move"
    )

    # Fire epoch 2 boundary; last_obs must persist after buffer reset
    state.ensemble_move_counts[2] = 6  # trigger epoch 2
    apply_epoch_ctrl(state, 0)

    # Buffer is reset but last_obs in ctrl_state must survive
    buf = state.ensemble_epoch_stats.get(2, {})
    assert buf.get("gain_sq_n", 0) == 0, "buffer must be reset after epoch flush"
    assert abs(state.softmax_ctrl_state[2]["last_obs"] - expected_obs) < 1e-12, (
        "last_obs in ctrl_state must survive epoch boundary"
    )


# ---------------------------------------------------------------------------
# Test T2 — last_obs round-trips through mirror → TOML → restore
# ---------------------------------------------------------------------------

def test_last_obs_restart_fidelity(dw_setup):
    """last_obs in ctrl_state must round-trip through mirror → TOML → restore exactly."""
    state, tmp_path = dw_setup
    keys = _sd_sim_keys()
    keys["reward_proxy"] = "empirical_dirichlet_lambda_v1"
    state.config["simulation"].update(keys)
    ENS_NUM = 1

    # Epoch 1 fires
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=-0.3))

    # Push ACC move to set last_obs
    lambda_i = state.ensembles[2]["interfaces"][1]
    lambda_upper = state.ensembles[2]["interfaces"][2]
    lm = lambda_i + 0.7 * (lambda_upper - lambda_i)
    update_epoch_stats(state, 2, accepted=True, path_length=10.0, subcycles=1, lambda_max=lm)
    expected_obs = _bounded_lambda_obs(lm, lambda_i, lambda_upper)

    # Mirror and round-trip through TOML
    mirror_epoch_ctrl(state, state.config)
    restart_path = tmp_path / "restart_last_obs.toml"
    with restart_path.open("wb") as fh:
        tomli_w.dump(state.config, fh)
    with restart_path.open("rb") as fh:
        saved_cfg = tomli.load(fh)

    # Verify serialized value
    saved_last_obs = saved_cfg["simulation"]["softmax_ctrl_state"]["2"]["last_obs"]
    assert abs(saved_last_obs - expected_obs) < 1e-12, (
        "last_obs must be written into restart.toml exactly"
    )

    # Restore and verify
    state2 = REPEX_state(saved_cfg, minus=True)
    state2.initiate_ensembles()
    paths = load_paths_from_disk(saved_cfg)
    state2.load_paths(paths)

    restored_obs = state2.softmax_ctrl_state[2].get("last_obs", float("nan"))
    assert abs(restored_obs - expected_obs) < 1e-12, (
        "last_obs must survive TOML round-trip exactly"
    )


# ---------------------------------------------------------------------------
# Test T3 — first accepted move initializes last_obs, no gain added
# ---------------------------------------------------------------------------

def test_first_accepted_initializes_last_obs_no_gain():
    """First finite curr_obs → last_obs set; gain_sq_n stays 0."""
    state = _make_state()
    state.config["simulation"]["reward_proxy"] = "empirical_dirichlet_lambda_v1"
    state.ensembles[1]["interfaces"] = (float("-inf"), 0.0, 1.0)
    state.softmax_ctrl_state[1] = {"last_obs": float("nan")}

    update_epoch_stats(state, 1, accepted=True, path_length=10.0, subcycles=2, lambda_max=0.4)

    cs = state.softmax_ctrl_state[1]
    buf = state.ensemble_epoch_stats[1]
    assert abs(cs["last_obs"] - 0.4) < 1e-12, "last_obs must be set to first obs"
    assert buf["gain_sq_n"] == 0, "no gain increment on first informative move"
    assert buf.get("gain_sq_sum", 0.0) == 0.0


# ---------------------------------------------------------------------------
# Test T4 — rejected after init gives zero delta, gain_sq_n increments
# ---------------------------------------------------------------------------

def test_rejected_after_init_zero_delta_gain():
    """REJ after init: delta=0, gain_sq_sum += 0, gain_sq_n += 1, last_obs unchanged."""
    state = _make_state()
    state.config["simulation"]["reward_proxy"] = "empirical_dirichlet_lambda_v1"
    state.ensembles[1]["interfaces"] = (float("-inf"), 0.0, 1.0)
    state.softmax_ctrl_state[1] = {"last_obs": 0.6}  # already initialized

    # REJ move: curr_obs = last_obs = 0.6, delta = 0
    update_epoch_stats(state, 1, accepted=False, path_length=10.0, subcycles=2, lambda_max=0.0)

    cs = state.softmax_ctrl_state[1]
    buf = state.ensemble_epoch_stats[1]
    assert buf["gain_sq_n"] == 1, "REJ must increment gain_sq_n"
    assert abs(buf.get("gain_sq_sum", 1.0)) < 1e-15, "delta=0: gain_sq_sum must not change"
    assert abs(cs["last_obs"] - 0.6) < 1e-12, "last_obs must not change on REJ"


# ---------------------------------------------------------------------------
# Test T5 — empirical reward nan when below min_count
# ---------------------------------------------------------------------------

def test_empirical_reward_nan_when_below_min_count():
    """gain_sq_n < reward_min_gain_count → _compute_epoch_reward returns nan."""
    sim = {"reward_proxy": "empirical_dirichlet_lambda_v1", "reward_min_gain_count": 5}
    buf = _empty_stats()
    buf["gain_sq_sum"] = 0.5
    buf["gain_sq_n"] = 4  # below min_count=5
    buf["subcycles_sum"] = 10
    buf["subcycles_n"] = 4

    reward = _compute_epoch_reward(buf, sim)
    assert math.isnan(reward), f"expected nan for gain_sq_n < min_count, got {reward}"

    # At exactly min_count: finite
    buf2 = _empty_stats()
    buf2["gain_sq_sum"] = 0.5
    buf2["gain_sq_n"] = 5
    buf2["subcycles_sum"] = 10
    buf2["subcycles_n"] = 5
    reward2 = _compute_epoch_reward(buf2, sim)
    assert not math.isnan(reward2), "gain_sq_n == min_count must give finite reward"


# ---------------------------------------------------------------------------
# Test T6 — empirical reward formula: total_gain / (total_cost + offset)
# ---------------------------------------------------------------------------

def test_empirical_reward_formula_total_gain_cost():
    """Finite case: reward == gain_sq_sum / (subcycles_sum + reward_cost_offset) exactly."""
    sim = {
        "reward_proxy": "empirical_dirichlet_lambda_v1",
        "reward_cost_offset": 2.0,
    }
    buf = _empty_stats()
    buf["gain_sq_sum"] = 0.3
    buf["gain_sq_n"] = 3
    buf["subcycles_sum"] = 8
    buf["subcycles_n"] = 3

    reward = _compute_epoch_reward(buf, sim)
    expected = 0.3 / (8 + 2.0)
    assert abs(reward - expected) < 1e-15, (
        f"expected {expected}, got {reward}"
    )


# ---------------------------------------------------------------------------
# Test T7 — explore_floor = 0 fails validation
# ---------------------------------------------------------------------------

def test_explore_floor_zero_fails_validation():
    """softmax_explore_floor = 0 must raise (strictly positive required)."""
    cfg = _sd_config_base()
    cfg["simulation"]["softmax_explore_floor"] = 0.0
    with pytest.raises((ValueError, TOMLConfigError), match="explore_floor"):
        validate_epoch_ctrl(cfg)


# ---------------------------------------------------------------------------
# Test T8 — duplicate choices fails validation
# ---------------------------------------------------------------------------

def test_duplicate_choices_fails_validation():
    """Duplicate values in epoch_nsubpath_choices must raise."""
    cfg = _sd_config_base()
    cfg["simulation"]["epoch_nsubpath_choices"] = [[1, 2, 2, 4]]  # duplicate 2
    with pytest.raises((ValueError, TOMLConfigError), match="distinct"):
        validate_epoch_ctrl(cfg)


# ---------------------------------------------------------------------------
# Test T9 — softmax_update_clip limits IW term
# ---------------------------------------------------------------------------

def test_softmax_update_clip_limits_iw_term(dw_setup):
    """Large reward_eff / tiny q → logit delta capped at eta * update_clip."""
    state, tmp_path = dw_setup
    keys = _sd_sim_keys()
    state.config["simulation"].update(keys)
    state.config["simulation"]["softmax_update_clip"] = 3.0
    ENS_NUM = 1  # ensemble index 2

    # Run two full epochs so we reach Case 3 (full logit update)
    for cstep in range(1, 7):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=-0.3))

    # Epoch 3: manually set extreme reward_eff via tiny q to force clipping
    # Inject extreme state: reward_ema set, ema_abs_reward tiny → huge reward_eff
    cs = state.softmax_ctrl_state[2]
    cs["reward_ema"] = 0.0
    cs["ema_abs_reward"] = 1e-9  # tiny → normalized reward will be huge

    # Record logits and update_count before boundary
    logits_before = list(cs["logits"])
    choice_idx = cs["last_choice_idx"]
    uc_before = cs["update_count"]

    # Inject large reward into buf so reward_raw is large
    buf = state.ensemble_epoch_stats.setdefault(2, {})
    buf["lambda_max_sum"] = 1e6
    buf["lambda_max_n"] = 1
    buf["subcycles_sum"] = 1
    buf["subcycles_n"] = 1

    # Trigger epoch 3 boundary
    state.ensemble_move_counts[2] = 9
    apply_epoch_ctrl(state, 0)

    cs_after = state.softmax_ctrl_state[2]
    logits_after = cs_after["logits"]
    update_clip = 3.0
    sim = state.config["simulation"]
    eta = sim["softmax_eta0"] / (uc_before + 1) ** sim["softmax_beta"]
    max_allowed_delta = eta * update_clip
    actual_delta = abs(logits_after[choice_idx] - logits_before[choice_idx])
    assert actual_delta <= max_allowed_delta + 1e-10, (
        f"logit delta {actual_delta:.6f} exceeds eta*update_clip={max_allowed_delta:.6f}"
    )


# ---------------------------------------------------------------------------
# Test T10 — logit box-clip applied after update
# ---------------------------------------------------------------------------

def test_logit_box_clip_applied_after_update(dw_setup):
    """Post-update logit exceeding softmax_logit_clip must be clipped to bound."""
    state, tmp_path = dw_setup
    keys = _sd_sim_keys()
    state.config["simulation"].update(keys)
    state.config["simulation"]["softmax_logit_clip"] = 1.0  # very tight clip
    ENS_NUM = 1

    # Run two epochs to reach Case 3
    for cstep in range(1, 7):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=-0.3))

    # Set one logit to just under the clip boundary, reward will push it over
    cs = state.softmax_ctrl_state[2]
    cs["reward_ema"] = 0.0
    cs["ema_abs_reward"] = 1e-9
    cs["logits"] = [0.9, 0.0, 0.0, 0.0]  # first logit near clip

    buf = state.ensemble_epoch_stats.setdefault(2, {})
    buf["lambda_max_sum"] = 1e6
    buf["lambda_max_n"] = 1
    buf["subcycles_sum"] = 1
    buf["subcycles_n"] = 1

    state.ensemble_move_counts[2] = 9
    apply_epoch_ctrl(state, 0)

    logit_clip = 1.0
    for logit in state.softmax_ctrl_state[2]["logits"]:
        assert logit <= logit_clip + 1e-12, f"logit {logit} exceeds clip {logit_clip}"
        assert logit >= -logit_clip - 1e-12, f"logit {logit} below -clip {-logit_clip}"


# ===========================================================================
# Regression tests 52–53 — boundary-consumption guard (per_ensemble_moves)
# ===========================================================================
#
# Before the fix, apply_epoch_ctrl re-fired on every call while the count
# stayed at a multiple of k (e.g. when moves to *other* ensembles triggered
# treat_output without advancing the targeted ensemble's counter).
# ===========================================================================


# ---------------------------------------------------------------------------
# Regression test 52 — scheduled/per_ensemble_moves: no re-fire on same count
# ---------------------------------------------------------------------------

def test_per_ensemble_moves_no_refire_on_same_count(tmp_path):
    """After firing at count=k, calling apply_epoch_ctrl again without
    incrementing the counter must NOT produce a second TSV row."""
    k = 3
    state = _make_state_per_ens(k=k)
    state.config["output"] = {"data_dir": str(tmp_path)}

    # Drive to boundary
    state.ensemble_move_counts[1] = k
    apply_epoch_ctrl(state, state.cstep)

    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    assert tsv_path.exists()
    rows_after_first = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows_after_first) == 1, "exactly one row after first boundary"
    assert state.ensemble_last_fired_count.get(1) == k

    # Call apply_epoch_ctrl several more times without changing the count.
    # This mimics moves to other ensembles being processed.
    for _ in range(5):
        apply_epoch_ctrl(state, state.cstep)

    rows_after_extra = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows_after_extra) == 1, (
        "no extra TSV rows must appear when count has not advanced past k"
    )

    # Only after the count advances to 2k does the next boundary fire.
    state.ensemble_move_counts[1] = 2 * k
    apply_epoch_ctrl(state, state.cstep)
    rows_after_second = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows_after_second) == 2, "second boundary must produce a second row"
    assert state.ensemble_last_fired_count.get(1) == 2 * k


# ---------------------------------------------------------------------------
# Regression test 53 — softmax_dirichlet: no extra sample_hold rows after boundary
# ---------------------------------------------------------------------------

def test_sd_no_extra_rows_after_boundary(dw_setup):
    """After epoch 1 fires (count=k via treat_output), additional calls to
    apply_epoch_ctrl without new moves must not write extra sample_hold rows."""
    state, tmp_path = dw_setup
    state.config["simulation"].update(_sd_sim_keys())
    ENS_NUM = 1  # ensemble index 2, k=3

    # Drive exactly k=3 moves through treat_output → epoch 1 fires
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=-0.3))

    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    assert tsv_path.exists()
    rows_after_epoch1 = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows_after_epoch1) == 1, "exactly one TSV row after epoch 1"
    assert state.ensemble_last_fired_count.get(2) == 3

    # Call apply_epoch_ctrl directly several times (simulating other-ensemble moves)
    for _ in range(5):
        apply_epoch_ctrl(state, state.cstep)

    rows_after_extra = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows_after_extra) == 1, (
        "no extra sample_hold rows must appear when ensemble 2 count is unchanged"
    )

    # Verify the one row is the real boundary row (not a spurious hold)
    r = rows_after_extra[0]
    assert int(r["n_attempted"]) > 0, "boundary row must have n_attempted > 0"
    assert "ctrl_mode" not in r, "ctrl_mode must not appear in new public schema"


# ===========================================================================
# Tests 54–59 — reward EMA normalization and clipping
# ===========================================================================


# ---------------------------------------------------------------------------
# Unit test 54 — normalization rescales small and large rewards comparably
# ---------------------------------------------------------------------------

def test_normalize_reward_rescales():
    """Small reward maps to O(0.1); large reward clips to rclip."""
    # Small: ema ← 0.9*1e-4 + 0.1*1e-5 = 9.1e-5; eff ≈ 1e-5/9.1e-5 ≈ 0.11
    eff_small, _ = _normalize_reward(1e-5, 1e-4)
    assert 0.0 < eff_small <= 5.0, f"small reward_eff out of range: {eff_small}"

    # Large: clips to default rclip = 5.0
    eff_large, _ = _normalize_reward(1e3, 1e-4)
    assert abs(eff_large - 5.0) < 1e-12, f"large reward should clip to 5.0, got {eff_large}"


# ---------------------------------------------------------------------------
# Unit test 55 — sign is preserved for negative rewards
# ---------------------------------------------------------------------------

def test_normalize_reward_preserves_sign():
    """Negative raw_reward must produce negative reward_eff."""
    eff, _ = _normalize_reward(-0.3, 1.0)   # warmed-up EMA
    assert eff < 0.0, f"negative raw_reward must give negative reward_eff, got {eff}"
    assert eff >= -5.0, "must not exceed clip magnitude"


# ---------------------------------------------------------------------------
# Unit test 56 — clipping is symmetric and exact at boundary
# ---------------------------------------------------------------------------

def test_normalize_reward_clipping():
    """reward_eff is clipped to [-rclip, rclip]."""
    eff_pos, _ = _normalize_reward(1e6,  1.0, rclip=3.0)
    assert abs(eff_pos - 3.0) < 1e-12, f"expected clip at +3.0, got {eff_pos}"

    eff_neg, _ = _normalize_reward(-1e6, 1.0, rclip=3.0)
    assert abs(eff_neg + 3.0) < 1e-12, f"expected clip at -3.0, got {eff_neg}"


# ---------------------------------------------------------------------------
# Integration test 57 — restart preserves ema_abs_reward
# ---------------------------------------------------------------------------

def test_normalize_reward_ema_survives_restart(dw_setup):
    """ema_abs_reward in ctrl_state must round-trip through mirror → TOML → restore."""
    state, tmp_path = dw_setup
    state.config["simulation"].update(_sd_sim_keys())
    ENS_NUM = 1  # ensemble index 2, k=3

    # Epoch 1 + epoch 2 so EMA is updated at least once
    for cstep in range(1, 7):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=-0.3))

    assert 2 in state.softmax_ctrl_state
    ema_before = state.softmax_ctrl_state[2]["ema_abs_reward"]

    # TOML roundtrip
    mirror_epoch_ctrl(state, state.config)
    restart_path = tmp_path / "restart_ema.toml"
    with restart_path.open("wb") as fh:
        tomli_w.dump(state.config, fh)
    with restart_path.open("rb") as fh:
        saved_cfg = tomli.load(fh)

    assert "ema_abs_reward" in saved_cfg["simulation"]["softmax_ctrl_state"]["2"], (
        "ema_abs_reward must be serialized into restart.toml"
    )

    state2 = REPEX_state(saved_cfg, minus=True)
    state2.initiate_ensembles()
    paths = load_paths_from_disk(saved_cfg)
    state2.load_paths(paths)

    ema_after = state2.softmax_ctrl_state[2]["ema_abs_reward"]
    assert abs(ema_after - ema_before) < 1e-15, (
        f"ema_abs_reward must survive restart exactly: {ema_before} → {ema_after}"
    )


# ---------------------------------------------------------------------------
# Unit test 58 — zero raw_reward gives zero reward_eff
# ---------------------------------------------------------------------------

def test_normalize_reward_zero():
    """raw_reward = 0.0 must give reward_eff = 0.0 regardless of EMA."""
    eff, new_ema = _normalize_reward(0.0, 1e-4)
    assert eff == 0.0, f"expected 0.0, got {eff}"
    # EMA still updates toward |0| = 0
    expected_ema = 0.9 * 1e-4 + 0.1 * 0.0
    assert abs(new_ema - expected_ema) < 1e-20


# ---------------------------------------------------------------------------
# Unit test 59 — nan raw_reward leaves EMA unchanged
# ---------------------------------------------------------------------------

def test_normalize_reward_nan_leaves_ema_unchanged():
    """raw_reward = nan must return (nan, ema_unchanged) without updating EMA."""
    ema_init = 0.42
    eff, ema_out = _normalize_reward(float("nan"), ema_init)
    assert math.isnan(eff), "non-finite input must yield nan reward_eff"
    assert ema_out == ema_init, "EMA must not change for non-finite input"


# ===========================================================================
# Tests 60–66 — lp_over_ls_target automatic WF n_jumps
# ===========================================================================

from infretis.core.epoch_ctrl import _lp_over_ls_ctrl


# ---------------------------------------------------------------------------
# Unit test 60 — validation rejects non-WF target
# ---------------------------------------------------------------------------

def test_lp_ls_validation_rejects_non_wf():
    """lp_over_ls_target=true must raise when a target ensemble uses 'sh'."""
    base = _check_config_base()
    base["simulation"]["lp_over_ls_target"] = True
    base["simulation"]["epoch_nsubpath_ens"] = [0]  # ensemble 0 uses "sh"
    base["simulation"]["epoch_size"] = 5
    base["simulation"]["epoch_mode"] = "global_step"
    with pytest.raises((ValueError, TOMLConfigError), match="move 'sh'"):
        validate_epoch_ctrl(base)


# ---------------------------------------------------------------------------
# Unit test 61 — validation rejects mwf target
# ---------------------------------------------------------------------------

def test_lp_ls_validation_rejects_mwf():
    """lp_over_ls_target=true must raise when a target ensemble uses 'mwf'."""
    base = _check_config_base()
    base["simulation"]["shooting_moves"] = ["sh", "mwf", "wf"]
    base["simulation"]["lp_over_ls_target"] = True
    base["simulation"]["epoch_nsubpath_ens"] = [1]  # ensemble 1 uses "mwf"
    base["simulation"]["epoch_size"] = 5
    base["simulation"]["epoch_mode"] = "global_step"
    with pytest.raises((ValueError, TOMLConfigError), match="mwf"):
        validate_epoch_ctrl(base)


# ---------------------------------------------------------------------------
# Unit test 62 — WF generated tuple has correct structure
# ---------------------------------------------------------------------------

def test_wf_metadata_in_generated_tuple():
    """wire_fencing() generated[0..3] unchanged; generated[4] has WF subpath keys."""
    import infretis.core.tis as _tis

    # Build a minimal fake generated tuple as wire_fencing() now produces it.
    # We can't run a real WF move without an engine, so we verify the shape
    # of the tuple that would be stored by constructing a mock path object.
    class _MockPath:
        length = 42
        status = "ACC"
        generated = None

    p = _MockPath()
    p.generated = (
        "wf", 9000, 2, p.length,
        {"wf_subpath_len_sum": 30, "wf_subpath_n": 3},
    )
    gen = p.generated
    assert gen[0] == "wf"
    assert gen[1] == 9000
    assert isinstance(gen[2], int)
    assert gen[3] == 42
    assert isinstance(gen[4], dict)
    assert "wf_subpath_len_sum" in gen[4]
    assert "wf_subpath_n" in gen[4]


# ---------------------------------------------------------------------------
# Unit test 63 — update_epoch_stats accumulates subpath fields
# ---------------------------------------------------------------------------

def test_update_epoch_stats_accumulates_subpath():
    """move_meta with wf subpath keys must accumulate in subpath_len_sum/n."""
    state = _make_state()
    sim = state.config["simulation"]
    sim["epoch_nsubpath_ens"] = [1]

    meta1 = {"wf_subpath_len_sum": 30, "wf_subpath_n": 3}
    meta2 = {"wf_subpath_len_sum": 20, "wf_subpath_n": 2}
    update_epoch_stats(state, 1, accepted=True, path_length=15.0, subcycles=1,
                       lambda_max=0.3, move_meta=meta1)
    update_epoch_stats(state, 1, accepted=False, path_length=10.0, subcycles=1,
                       lambda_max=0.1, move_meta=meta2)

    buf = state.ensemble_epoch_stats[1]
    assert buf["subpath_len_sum"] == 50.0
    assert buf["subpath_n"] == 5


# ---------------------------------------------------------------------------
# Integration test 64 — boundary sets n_jumps = round(Lp/Ls), no epoch_ctrl_mode needed
# ---------------------------------------------------------------------------

def test_lp_ls_boundary_sets_correct_n_jumps(dw_setup):
    """lp_over_ls_target=true without epoch_ctrl_mode: n_jumps set to round(Lp/Ls)."""
    state, tmp_path = dw_setup
    sim = state.config["simulation"]
    sim["lp_over_ls_target"] = True
    sim["epoch_nsubpath_ens"] = [2]
    sim["epoch_mode"] = "global_step"
    sim["epoch_size"] = 3

    ENS_NUM = 1  # ensemble index 2

    # Push synthetic subpath and path length stats directly
    # Avg path length = 60, avg subpath length = 10 → target = round(6) = 6
    for _ in range(3):
        update_epoch_stats(
            state, 2, accepted=True, path_length=60.0, subcycles=1,
            lambda_max=-0.5,
            move_meta={"wf_subpath_len_sum": 20, "wf_subpath_n": 2},
        )

    # Trigger epoch boundary at cstep=3
    state.cstep = 3
    state.config["current"]["cstep"] = 3
    apply_epoch_ctrl(state, 3)

    # avg_path_len = 60.0, avg_subpath_len = 20/2 = 10.0 → round(6.0) = 6
    assert state.ensembles[2]["tis_set"]["n_jumps"] == 6, (
        f"expected n_jumps=6, got {state.ensembles[2]['tis_set']['n_jumps']}"
    )

    # TSV written with lp_over_ls columns populated
    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    assert tsv_path.exists()
    rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows) == 1
    r = rows[0]
    assert "ctrl_mode" not in r, "ctrl_mode must not appear in new public schema"
    assert float(r["avg_subpath_length"]) == pytest.approx(10.0)
    assert float(r["lp_over_ls_target_value"]) == 6.0


# ---------------------------------------------------------------------------
# Unit test 65 — subpath_n=0 → hold, no crash
# ---------------------------------------------------------------------------

def test_lp_ls_missing_data_holds(tmp_path):
    """When subpath_n=0 the controller must hold n_jumps and not crash."""
    state = _make_state()
    sim = state.config["simulation"]
    sim["lp_over_ls_target"] = True
    sim["epoch_nsubpath_ens"] = [1]
    sim["epoch_mode"] = "global_step"
    sim["epoch_size"] = 5
    state.config["output"] = {"data_dir": str(tmp_path)}

    # Stats with no subpath data (subpath_n stays 0)
    update_epoch_stats(
        state, 1, accepted=True, path_length=50.0, subcycles=1,
        lambda_max=0.3,
    )
    original_n_jumps = state.ensembles[1]["tis_set"].get("n_jumps", 2)

    state.config["current"]["cstep"] = 5
    apply_epoch_ctrl(state, 5)

    # n_jumps must be unchanged
    assert state.ensembles[1]["tis_set"]["n_jumps"] == original_n_jumps

    # TSV written with hold action
    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    assert tsv_path.exists()
    rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["ctrl_action"] == "hold"
    assert rows[0]["ctrl_reason"] == "missing_subpath_stats"


# ---------------------------------------------------------------------------
# Unit test 66 — lp_over_ls absent: scheduled behavior unchanged
# ---------------------------------------------------------------------------

def test_lp_ls_absent_unchanged():
    """Without lp_over_ls_target the scheduled controller is unaffected."""
    state = _make_state()
    # lp_over_ls_target is not set → scheduled mode applies
    assert "lp_over_ls_target" not in state.config["simulation"]

    expectations = {10: 4, 20: 2, 30: 4}
    for cstep, expected in expectations.items():
        state.config["current"]["cstep"] = cstep
        apply_epoch_ctrl(state, cstep)
        assert state.ensembles[1]["tis_set"]["n_jumps"] == expected, (
            f"cstep={cstep}: expected n_jumps={expected}"
        )


# ---------------------------------------------------------------------------
# Unit test 67 — half-up rounding at x.5
# ---------------------------------------------------------------------------

def test_lp_ls_half_up_rounding(tmp_path):
    """Lp/Ls=2.5 must round to 3 (half-up), not 2 (banker's rounding)."""
    state = _make_state()
    sim = state.config["simulation"]
    sim["lp_over_ls_target"] = True
    sim["epoch_nsubpath_ens"] = [1]
    sim["epoch_mode"] = "global_step"
    sim["epoch_size"] = 5
    state.config["output"] = {"data_dir": str(tmp_path)}

    # avg_path_len=25, avg_subpath_len=10 → 25/10=2.5 → floor(2.5+0.5)=3
    for _ in range(5):
        update_epoch_stats(
            state, 1, accepted=True, path_length=25.0, subcycles=1,
            lambda_max=0.3,
            move_meta={"wf_subpath_len_sum": 10, "wf_subpath_n": 1},
        )

    state.config["current"]["cstep"] = 5
    apply_epoch_ctrl(state, 5)

    assert state.ensembles[1]["tis_set"]["n_jumps"] == 3, (
        "Lp/Ls=2.5 must round to 3 with half-up rounding"
    )

    # Also check 3.5 → 4
    state2 = _make_state()
    state2.config["simulation"].update({
        "lp_over_ls_target": True,
        "epoch_nsubpath_ens": [1],
        "epoch_mode": "global_step",
        "epoch_size": 2,
    })
    state2.config["output"] = {"data_dir": str(tmp_path)}
    for _ in range(2):
        update_epoch_stats(
            state2, 1, accepted=True, path_length=35.0, subcycles=1,
            lambda_max=0.3,
            move_meta={"wf_subpath_len_sum": 10, "wf_subpath_n": 1},
        )
    state2.config["current"]["cstep"] = 2
    apply_epoch_ctrl(state2, 2)
    assert state2.ensembles[1]["tis_set"]["n_jumps"] == 4, (
        "Lp/Ls=3.5 must round to 4 with half-up rounding"
    )


# ---------------------------------------------------------------------------
# Integration test 68 — treat_output transports epoch_move_meta end-to-end
# ---------------------------------------------------------------------------

def test_lp_ls_treat_output_transports_move_meta(dw_setup):
    """epoch_move_meta injected into md_items must accumulate subpath stats
    through treat_output, firing the lp_ls controller at the epoch boundary."""
    state, tmp_path = dw_setup
    sim = state.config["simulation"]
    sim["lp_over_ls_target"] = True
    sim["epoch_nsubpath_ens"] = [2]
    sim["epoch_mode"] = "global_step"
    sim["epoch_size"] = 3

    ENS_NUM = 1  # ensemble index 2

    # avg_path_len ~ 60 (real path), avg_subpath_len = 20 → expect n_jumps=3
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        md = _build_md_items(state, ENS_NUM, trial_op_max=-0.5)
        # inject epoch_move_meta as treat_output would receive from run_md
        md["epoch_move_meta"] = [{"wf_subpath_len_sum": 20, "wf_subpath_n": 1}]
        state.treat_output(md)

    buf = state.ensemble_epoch_stats.get(2, {})
    # Buffer must be reset after epoch fired
    assert buf.get("n_attempted", 0) == 0, "buffer must reset after epoch"

    # n_jumps set from Lp/Ls: avg_path≈real path length, avg_subpath=20
    nj = state.ensembles[2]["tis_set"]["n_jumps"]
    assert nj >= 1, "n_jumps must be a positive integer"

    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    assert tsv_path.exists()
    rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows) == 1
    r = rows[0]
    assert "ctrl_mode" not in r, "ctrl_mode must not appear in new public schema"
    assert float(r["avg_subpath_length"]) == pytest.approx(20.0)
    assert r["ctrl_reason"] == "lp_over_ls_rule"


# ---------------------------------------------------------------------------
# Unit test 69 — validation rejects lp_over_ls_target with empty targets
# ---------------------------------------------------------------------------

def test_lp_ls_validation_rejects_empty_targets():
    """lp_over_ls_target=true with no epoch_nsubpath_ens must raise."""
    base = _check_config_base()
    base["simulation"]["lp_over_ls_target"] = True
    # epoch_nsubpath_ens not set
    with pytest.raises((ValueError, TOMLConfigError), match="epoch_nsubpath_ens"):
        validate_epoch_ctrl(base)


# ---------------------------------------------------------------------------
# Integration test 70 — mid-epoch restart loses subpath stats → hold
# ---------------------------------------------------------------------------

def test_lp_ls_restart_loses_partial_stats_holds(dw_setup):
    """Mid-epoch restart clears subpath stats; first post-restart epoch must
    fire a hold with ctrl_reason='missing_subpath_stats'."""
    state, tmp_path = dw_setup
    sim = state.config["simulation"]
    sim["lp_over_ls_target"] = True
    sim["epoch_nsubpath_ens"] = [2]
    sim["epoch_mode"] = "global_step"
    sim["epoch_size"] = 5

    ENS_NUM = 1  # ensemble index 2

    # Accumulate 3 moves (mid-epoch: epoch fires at cstep=5)
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        md = _build_md_items(state, ENS_NUM, trial_op_max=-0.5)
        md["epoch_move_meta"] = [{"wf_subpath_len_sum": 20, "wf_subpath_n": 1}]
        state.treat_output(md)

    # Partial epoch stats must be present
    buf = state.ensemble_epoch_stats.get(2, {})
    assert buf.get("subpath_n", 0) == 3

    # Serialize via write_toml (simulates restart stop)
    mirror_epoch_ctrl(state, state.config)
    restart_path = tmp_path / "restart.toml"
    with restart_path.open("wb") as fh:
        tomli_w.dump(state.config, fh)
    with restart_path.open("rb") as fh:
        saved_cfg = tomli.load(fh)

    # Restore — subpath stats are NOT persisted for lp_ls mode
    state2 = REPEX_state(saved_cfg, minus=True)
    state2.initiate_ensembles()
    paths = load_paths_from_disk(saved_cfg)
    state2.load_paths(paths)

    buf2 = state2.ensemble_epoch_stats.get(2, {})
    assert buf2.get("subpath_n", 0) == 0, (
        "subpath_n must not survive restart for lp_ls mode"
    )

    # Drive 2 more moves to hit epoch boundary at cstep=5
    for cstep in (4, 5):
        state2.cstep = cstep
        _lock_ens_slot(state2, ENS_NUM)
        md = _build_md_items(state2, ENS_NUM, trial_op_max=-0.5)
        # No epoch_move_meta → subpath_n stays 0
        state2.treat_output(md)

    # Epoch must have fired with hold due to missing subpath stats
    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    assert tsv_path.exists()
    rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["ctrl_action"] == "hold"
    assert rows[0]["ctrl_reason"] == "missing_subpath_stats"


# ===========================================================================
# Tests 71–75 — reward EMA baseline for softmax_dirichlet controller
# ===========================================================================


# ---------------------------------------------------------------------------
# Unit test 71 — non-finite raw_reward: logits/update_count/reward_ema unchanged
# ---------------------------------------------------------------------------

def test_ema_nonfinite_skips_all(dw_setup):
    """raw_reward=nan must leave logits, update_count, and reward_ema unchanged."""
    state, tmp_path = dw_setup
    state.config["simulation"].update(_sd_sim_keys())
    ENS_NUM = 1  # ensemble index 2

    # Epoch 1: initialise ctrl_state (reward_ema stays None after first boundary)
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=-0.3))
    assert 2 in state.softmax_ctrl_state

    # Force reward_ema to a known value so we can confirm it's unchanged
    state.softmax_ctrl_state[2]["reward_ema"] = 0.1
    logits_before = list(state.softmax_ctrl_state[2]["logits"])
    uc_before = state.softmax_ctrl_state[2]["update_count"]

    # Inject nan into the stats buffer so _compute_epoch_reward returns nan
    state.ensemble_epoch_stats[2] = _empty_stats()
    state.ensemble_epoch_stats[2]["lambda_max_sum"] = float("nan")
    state.ensemble_epoch_stats[2]["lambda_max_n"] = 1
    state.ensemble_epoch_stats[2]["subcycles_sum"] = 1
    state.ensemble_epoch_stats[2]["subcycles_n"] = 1

    state.ensemble_move_counts[2] = 6  # trigger epoch 2 (count = 2*k = 6)
    apply_epoch_ctrl(state, 0)

    cs = state.softmax_ctrl_state[2]
    assert cs["logits"] == logits_before, "logits must not change for nan reward"
    assert cs["update_count"] == uc_before, "update_count must not change for nan reward"
    assert cs["reward_ema"] == pytest.approx(0.1), "reward_ema must not change for nan reward"


# ---------------------------------------------------------------------------
# Unit test 72 — first finite reward initializes EMA, skips logit update
# ---------------------------------------------------------------------------

def test_ema_first_finite_initializes(dw_setup):
    """reward_ema=None + finite raw_reward → reward_ema=raw, reward_eff=0, logits unchanged."""
    state, tmp_path = dw_setup
    state.config["simulation"].update(_sd_sim_keys())
    ENS_NUM = 1  # ensemble index 2

    # Epoch 1: initialise ctrl_state
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=0.3))

    cs = state.softmax_ctrl_state[2]
    assert math.isnan(cs["reward_ema"]), "reward_ema must be nan after first boundary"
    logits_before = list(cs["logits"])
    uc_before = cs["update_count"]

    # Epoch 2 with finite buffer → Case 2 path
    state.ensemble_epoch_stats[2] = _empty_stats()
    state.ensemble_epoch_stats[2]["lambda_max_sum"] = 0.5
    state.ensemble_epoch_stats[2]["lambda_max_n"] = 1
    state.ensemble_epoch_stats[2]["subcycles_sum"] = 1
    state.ensemble_epoch_stats[2]["subcycles_n"] = 1

    raw_reward = _compute_epoch_reward(
        state.ensemble_epoch_stats[2], state.config["simulation"]
    )
    assert np.isfinite(raw_reward)

    state.ensemble_move_counts[2] = 6
    apply_epoch_ctrl(state, 0)

    cs = state.softmax_ctrl_state[2]
    assert cs["reward_ema"] == pytest.approx(raw_reward), (
        "reward_ema must be initialized to raw_reward"
    )
    assert cs["logits"] == logits_before, "logits must not change on first finite reward"
    assert cs["update_count"] == uc_before, "update_count must not increment on first finite reward"


# ---------------------------------------------------------------------------
# Unit test 73 — subsequent finite reward: full update with correct values
# ---------------------------------------------------------------------------

def test_ema_subsequent_full_update(dw_setup):
    """Verify centered reward, logit delta, update_count, and EMA update are all correct."""
    state, tmp_path = dw_setup
    state.config["simulation"].update(_sd_sim_keys())
    ENS_NUM = 1  # ensemble index 2

    # Epoch 1: init ctrl_state
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=0.3))

    # Epoch 2: initialize reward_ema baseline
    state.ensemble_epoch_stats[2] = _empty_stats()
    state.ensemble_epoch_stats[2]["lambda_max_sum"] = 0.5
    state.ensemble_epoch_stats[2]["lambda_max_n"] = 1
    state.ensemble_epoch_stats[2]["subcycles_sum"] = 1
    state.ensemble_epoch_stats[2]["subcycles_n"] = 1
    state.ensemble_move_counts[2] = 6
    apply_epoch_ctrl(state, 0)

    cs = state.softmax_ctrl_state[2]
    assert not math.isnan(cs["reward_ema"])
    ema_prev = cs["reward_ema"]
    uc_before = cs["update_count"]  # 0 (no logit update yet)
    logits_before = list(cs["logits"])
    A = cs["last_choice_idx"]
    ema_abs_before = float(cs.get("ema_abs_reward", 1e-4))

    sim = state.config["simulation"]
    alpha = float(sim.get("reward_ema_alpha", 0.1))

    # Epoch 3: push a different reward value to trigger Case 3
    state.ensemble_epoch_stats[2] = _empty_stats()
    state.ensemble_epoch_stats[2]["lambda_max_sum"] = 0.9
    state.ensemble_epoch_stats[2]["lambda_max_n"] = 1
    state.ensemble_epoch_stats[2]["subcycles_sum"] = 1
    state.ensemble_epoch_stats[2]["subcycles_n"] = 1

    raw_reward_actual = _compute_epoch_reward(state.ensemble_epoch_stats[2], sim)
    old_q = _compute_q(logits_before, sim["softmax_tau"], sim["softmax_explore_floor"])
    reward_centered = raw_reward_actual - ema_prev
    reward_eff_expected, _ = _normalize_reward(
        reward_centered,
        ema_abs_before,
        rho=float(sim.get("softmax_reward_ema_rho", 0.1)),
        floor=float(sim.get("softmax_reward_ema_floor", 1e-6)),
        rclip=float(sim.get("softmax_reward_eff_clip", 5.0)),
    )
    eta_expected = sim["softmax_eta0"] / (uc_before + 1) ** sim["softmax_beta"]
    update_clip_val = float(sim.get("softmax_update_clip", 5.0))
    iw_term_expected = float(np.clip(reward_eff_expected / old_q[A], -update_clip_val, update_clip_val))
    expected_logit_delta = eta_expected * iw_term_expected
    expected_ema = (1.0 - alpha) * ema_prev + alpha * raw_reward_actual

    state.ensemble_move_counts[2] = 9
    apply_epoch_ctrl(state, 0)

    cs = state.softmax_ctrl_state[2]
    assert cs["update_count"] == uc_before + 1, "update_count must increment"
    assert abs(cs["logits"][A] - (logits_before[A] + expected_logit_delta)) < 1e-10, (
        f"logit delta mismatch: expected {expected_logit_delta}, "
        f"got {cs['logits'][A] - logits_before[A]}"
    )
    assert cs["reward_ema"] == pytest.approx(expected_ema), "reward_ema must be EMA updated"


# ---------------------------------------------------------------------------
# Integration test 74 — restart fidelity: reward_ema survives TOML roundtrip
# ---------------------------------------------------------------------------

def test_ema_restart_fidelity(dw_setup):
    """reward_ema in ctrl_state must round-trip through mirror → TOML → restore exactly."""
    state, tmp_path = dw_setup
    state.config["simulation"].update(_sd_sim_keys())
    ENS_NUM = 1  # ensemble index 2

    # Epoch 1: init ctrl_state
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=0.3))

    # Epoch 2: initialize reward_ema
    state.ensemble_epoch_stats[2] = _empty_stats()
    state.ensemble_epoch_stats[2]["lambda_max_sum"] = 0.5
    state.ensemble_epoch_stats[2]["lambda_max_n"] = 1
    state.ensemble_epoch_stats[2]["subcycles_sum"] = 1
    state.ensemble_epoch_stats[2]["subcycles_n"] = 1
    state.ensemble_move_counts[2] = 6
    apply_epoch_ctrl(state, 0)

    cs_before = state.softmax_ctrl_state[2]
    assert not math.isnan(cs_before["reward_ema"]), "reward_ema must be set after epoch 2"
    ema_before = cs_before["reward_ema"]

    # TOML roundtrip
    mirror_epoch_ctrl(state, state.config)
    assert "reward_ema" in state.config["simulation"]["softmax_ctrl_state"]["2"], (
        "reward_ema must be serialized by mirror_epoch_ctrl"
    )

    restart_path = tmp_path / "restart_ema_baseline.toml"
    with restart_path.open("wb") as fh:
        tomli_w.dump(state.config, fh)
    with restart_path.open("rb") as fh:
        saved_cfg = tomli.load(fh)

    state2 = REPEX_state(saved_cfg, minus=True)
    state2.initiate_ensembles()
    paths = load_paths_from_disk(saved_cfg)
    state2.load_paths(paths)

    cs2 = state2.softmax_ctrl_state[2]
    assert not math.isnan(cs2["reward_ema"]), "reward_ema must be restored from TOML"
    assert abs(cs2["reward_ema"] - ema_before) < 1e-15, (
        f"reward_ema must survive restart exactly: {ema_before} → {cs2['reward_ema']}"
    )


# ---------------------------------------------------------------------------
# Integration test 75 — TSV row contains reward_ema column with correct value
# ---------------------------------------------------------------------------

def test_ema_logged_in_tsv(dw_setup):
    """After epoch boundaries, TSV must contain reward_ema column with correct values."""
    state, tmp_path = dw_setup
    state.config["simulation"].update(_sd_sim_keys())
    ENS_NUM = 1  # ensemble index 2

    # Epoch 1: init ctrl_state
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=0.3))

    # Epoch 1: reward_ema must be in debug TSV (not summary), must be nan
    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    debug_path = tmp_path / _EPOCH_SOFTMAX_DEBUG_FNAME
    summary_rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(summary_rows) == 1
    assert "reward_ema" not in summary_rows[0], "reward_ema must not appear in summary TSV"
    assert debug_path.exists(), "epoch_softmax_debug.tsv must be created"
    debug_rows = list(csv.DictReader(debug_path.open(), delimiter="\t"))
    assert len(debug_rows) == 1
    assert "reward_ema" in debug_rows[0], "reward_ema column must exist in debug TSV header"
    assert math.isnan(float(debug_rows[0]["reward_ema"])), (
        "first boundary: reward_ema must be nan (ctrl_state just initialized)"
    )

    # Epoch 2: initialize EMA baseline (Case 2)
    state.ensemble_epoch_stats[2] = _empty_stats()
    state.ensemble_epoch_stats[2]["lambda_max_sum"] = 0.5
    state.ensemble_epoch_stats[2]["lambda_max_n"] = 1
    state.ensemble_epoch_stats[2]["subcycles_sum"] = 1
    state.ensemble_epoch_stats[2]["subcycles_n"] = 1
    raw_reward_ep2 = _compute_epoch_reward(
        state.ensemble_epoch_stats[2], state.config["simulation"]
    )
    state.ensemble_move_counts[2] = 6
    apply_epoch_ctrl(state, 0)

    debug_rows = list(csv.DictReader(debug_path.open(), delimiter="\t"))
    assert len(debug_rows) == 2
    dr2 = debug_rows[1]
    # Case 2 sets reward_ema = raw_reward_ep2
    assert not math.isnan(float(dr2["reward_ema"])), (
        "epoch 2 debug row: reward_ema must be set after EMA initialization"
    )
    assert float(dr2["reward_ema"]) == pytest.approx(raw_reward_ep2), (
        "epoch 2 debug row: reward_ema must equal the first finite raw_reward"
    )


# ---------------------------------------------------------------------------
# Unit test 76 — alpha=1.0: reward_ema jumps fully to raw_reward each update
# ---------------------------------------------------------------------------

def test_ema_alpha_one_jumps_to_raw_reward(dw_setup):
    """With reward_ema_alpha=1.0, reward_ema must equal raw_reward after each Case 3 update.

    The EMA formula is (1-alpha)*ema_prev + alpha*raw_reward.  At alpha=1 this
    collapses to raw_reward regardless of ema_prev, so the baseline tracks the
    most recent observation with zero smoothing.
    """
    state, tmp_path = dw_setup
    keys = _sd_sim_keys()
    keys["reward_ema_alpha"] = 1.0
    state.config["simulation"].update(keys)
    ENS_NUM = 1  # ensemble index 2

    def _push_finite_epoch(lmax_sum, count):
        """Inject a synthetic stats buffer and fire the next epoch boundary."""
        state.ensemble_epoch_stats[2] = _empty_stats()
        state.ensemble_epoch_stats[2]["lambda_max_sum"] = lmax_sum
        state.ensemble_epoch_stats[2]["lambda_max_n"] = 1
        state.ensemble_epoch_stats[2]["subcycles_sum"] = 1
        state.ensemble_epoch_stats[2]["subcycles_n"] = 1
        state.ensemble_move_counts[2] = count
        apply_epoch_ctrl(state, 0)
        return _compute_epoch_reward(
            {"lambda_max_sum": lmax_sum, "lambda_max_n": 1,
             "subcycles_sum": 1, "subcycles_n": 1,
             "subpath_len_sum": 0.0, "subpath_n": 0,
             "prev_obs": float("nan"), "gain_sq_sum": 0.0, "gain_sq_n": 0,
             "n_attempted": 1, "n_accepted": 1,
             "path_length_sum": 10.0, "path_length_n": 1,
             "lambda_max": lmax_sum, "lambda_max_sum": lmax_sum,
             "subcycles_sum": 1, "subcycles_n": 1},
            state.config["simulation"],
        )

    # Epoch 1: init ctrl_state (reward_ema stays nan)
    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=0.3))

    assert math.isnan(state.softmax_ctrl_state[2]["reward_ema"])

    # Epoch 2: Case 2 — baseline initialized to raw_reward_A
    state.ensemble_epoch_stats[2] = _empty_stats()
    state.ensemble_epoch_stats[2]["lambda_max_sum"] = 0.4
    state.ensemble_epoch_stats[2]["lambda_max_n"] = 1
    state.ensemble_epoch_stats[2]["subcycles_sum"] = 1
    state.ensemble_epoch_stats[2]["subcycles_n"] = 1
    raw_reward_A = _compute_epoch_reward(
        state.ensemble_epoch_stats[2], state.config["simulation"]
    )
    state.ensemble_move_counts[2] = 6
    apply_epoch_ctrl(state, 0)
    assert state.softmax_ctrl_state[2]["reward_ema"] == pytest.approx(raw_reward_A)

    # Epoch 3: Case 3 — with alpha=1.0, reward_ema must jump to raw_reward_B
    state.ensemble_epoch_stats[2] = _empty_stats()
    state.ensemble_epoch_stats[2]["lambda_max_sum"] = 0.9
    state.ensemble_epoch_stats[2]["lambda_max_n"] = 1
    state.ensemble_epoch_stats[2]["subcycles_sum"] = 1
    state.ensemble_epoch_stats[2]["subcycles_n"] = 1
    raw_reward_B = _compute_epoch_reward(
        state.ensemble_epoch_stats[2], state.config["simulation"]
    )
    assert abs(raw_reward_B - raw_reward_A) > 1e-6, "test requires distinct rewards"
    state.ensemble_move_counts[2] = 9
    apply_epoch_ctrl(state, 0)

    ema_after = state.softmax_ctrl_state[2]["reward_ema"]
    # (1 - 1.0) * raw_reward_A + 1.0 * raw_reward_B = raw_reward_B
    assert ema_after == pytest.approx(raw_reward_B), (
        f"alpha=1.0: reward_ema must jump fully to raw_reward_B={raw_reward_B}, "
        f"got {ema_after} (ema_prev was {raw_reward_A})"
    )
    # Also confirm it differs from the old baseline
    assert abs(ema_after - raw_reward_A) > 1e-6, (
        "reward_ema must not retain any weight from the previous baseline"
    )


# ===========================================================================
# Tests 77–81 — Schema constant tests (new in spring-cleaning refactor)
# ===========================================================================


# ---------------------------------------------------------------------------
# Test 77 — public TSV header equals _EPOCH_SUMMARY_COLS exactly
# ---------------------------------------------------------------------------

def test_public_tsv_schema(tmp_path):
    """epoch_summary.tsv header must equal _EPOCH_SUMMARY_COLS exactly."""
    state = _make_state()
    state.config["output"] = {"data_dir": str(tmp_path)}

    _push_moves(state, ens_idx=1, n_attempted=5, n_accepted=3,
                path_length=10.0, subcycles=2, lambda_max=0.3)
    state.config["current"]["cstep"] = 10
    apply_epoch_ctrl(state, state.cstep)

    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    assert tsv_path.exists()
    header = tsv_path.read_text().splitlines()[0].split("\t")
    assert header == list(_EPOCH_SUMMARY_COLS), (
        f"TSV header {header} must equal _EPOCH_SUMMARY_COLS {list(_EPOCH_SUMMARY_COLS)}"
    )


# ---------------------------------------------------------------------------
# Test 78 — debug TSV header equals _EPOCH_SOFTMAX_DEBUG_COLS exactly
# ---------------------------------------------------------------------------

def test_debug_tsv_schema(dw_setup):
    """epoch_softmax_debug.tsv header must equal _EPOCH_SOFTMAX_DEBUG_COLS exactly."""
    state, tmp_path = dw_setup
    state.config["simulation"].update(_sd_sim_keys())
    ENS_NUM = 1

    for cstep in (1, 2, 3):
        state.cstep = cstep
        _lock_ens_slot(state, ENS_NUM)
        state.treat_output(_build_md_items(state, ENS_NUM, trial_op_max=0.3))

    debug_path = tmp_path / _EPOCH_SOFTMAX_DEBUG_FNAME
    assert debug_path.exists()
    header = debug_path.read_text().splitlines()[0].split("\t")
    assert header == list(_EPOCH_SOFTMAX_DEBUG_COLS), (
        f"Debug TSV header {header} must equal _EPOCH_SOFTMAX_DEBUG_COLS"
    )


# ---------------------------------------------------------------------------
# Test 79 — debug TSV not written for non-softmax (adaptive) runs
# ---------------------------------------------------------------------------

def test_debug_tsv_not_written_for_non_softmax(tmp_path):
    """epoch_softmax_debug.tsv must not be created for adaptive/static/scheduled runs."""
    state = _make_adaptive_state()
    state.config["output"] = {"data_dir": str(tmp_path)}

    for _ in range(4):
        update_epoch_stats(
            state, 1, accepted=False,
            path_length=50.0, subcycles=2, lambda_max=0.02
        )

    state.config["current"]["cstep"] = 10
    apply_epoch_ctrl(state, 10)

    # summary TSV must be written
    assert (tmp_path / _EPOCH_SUMMARY_FNAME).exists()
    # debug TSV must NOT be written
    assert not (tmp_path / _EPOCH_SOFTMAX_DEBUG_FNAME).exists(), (
        "epoch_softmax_debug.tsv must not be created for non-softmax runs"
    )


# ---------------------------------------------------------------------------
# Test 80 — lp_ls run schema includes lp_ls columns
# ---------------------------------------------------------------------------

def test_lp_ls_run_schema_cols(dw_setup):
    """When lp_over_ls_target=True, TSV header must include _EPOCH_SUMMARY_LPLSCOLS."""
    state, tmp_path = dw_setup
    sim = state.config["simulation"]
    sim["lp_over_ls_target"] = True
    sim["epoch_nsubpath_ens"] = [2]
    sim["epoch_mode"] = "global_step"
    sim["epoch_size"] = 3

    for _ in range(3):
        update_epoch_stats(
            state, 2, accepted=True, path_length=60.0, subcycles=1,
            lambda_max=-0.5,
            move_meta={"wf_subpath_len_sum": 20, "wf_subpath_n": 2},
        )

    state.cstep = 3
    state.config["current"]["cstep"] = 3
    apply_epoch_ctrl(state, 3)

    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    assert tsv_path.exists()
    header = tsv_path.read_text().splitlines()[0].split("\t")
    expected = list(_EPOCH_SUMMARY_COLS) + list(_EPOCH_SUMMARY_LPLSCOLS)
    assert header == expected, (
        f"lp_ls TSV header {header} must equal COLS+LPLSCOLS {expected}"
    )


# ---------------------------------------------------------------------------
# Test 81 — backward-compat: compare_epoch reader handles old and new TSVs
# ---------------------------------------------------------------------------

def test_backward_compat_reward_column(tmp_path):
    """compare_epoch._load must handle old TSV (reward col) and new TSV (reward_eff col)."""
    from infretis.tools.compare_epoch import _load

    # Old-schema TSV with "reward" column
    old_tsv = tmp_path / "old.tsv"
    old_tsv.write_text(
        "epoch_idx\tens_name\tn_jumps_new\tacc_rate\tavg_path_length\treward\n"
        "1\t001\t2\t0.5\t10.0\t0.42\n"
    )
    data_old, opt_old = _load(old_tsv)
    assert "reward_eff" in data_old["001"], "old TSV reward must be aliased to reward_eff"
    assert abs(data_old["001"]["reward_eff"][0] - 0.42) < 1e-9

    # New-schema TSV with "reward_eff" column
    new_tsv = tmp_path / "new.tsv"
    new_tsv.write_text(
        "epoch_idx\tens_name\tn_jumps_new\tacc_rate\tavg_path_length\treward_eff\n"
        "1\t001\t2\t0.5\t10.0\t0.42\n"
    )
    data_new, opt_new = _load(new_tsv)
    assert "reward_eff" in data_new["001"]
    assert abs(data_new["001"]["reward_eff"][0] - 0.42) < 1e-9

    # Both yield same numeric value
    assert data_old["001"]["reward_eff"][0] == data_new["001"]["reward_eff"][0]


# ---------------------------------------------------------------------------
# Test 82 — _append_tsv rejects a pre-existing file with wrong header
# ---------------------------------------------------------------------------

def test_append_tsv_rejects_mismatched_header(tmp_path):
    """_append_tsv must raise ValueError when the on-disk header differs."""
    # -- public summary schema
    bad_pub = tmp_path / "bad_public.tsv"
    bad_pub.write_text("col_a\tcol_b\tcol_c\n1\t2\t3\n")
    row_pub = {c: "x" for c in _EPOCH_SUMMARY_COLS}
    with pytest.raises(ValueError, match="schema mismatch"):
        _append_tsv(str(bad_pub), _EPOCH_SUMMARY_COLS, row_pub)

    # -- softmax debug schema
    bad_dbg = tmp_path / "bad_debug.tsv"
    bad_dbg.write_text("foo\tbar\n0\t0\n")
    row_dbg = {c: "x" for c in _EPOCH_SOFTMAX_DEBUG_COLS}
    with pytest.raises(ValueError, match="schema mismatch"):
        _append_tsv(str(bad_dbg), _EPOCH_SOFTMAX_DEBUG_COLS, row_dbg)


# ---------------------------------------------------------------------------
# Test 83 — _append_tsv succeeds on a file whose header already matches
# ---------------------------------------------------------------------------

def test_append_tsv_succeeds_on_matching_header(tmp_path):
    """_append_tsv must append without error when the existing header matches."""
    # -- public summary schema
    pub = tmp_path / "good_public.tsv"
    header_pub = "\t".join(_EPOCH_SUMMARY_COLS)
    existing_row_pub = "\t".join("0" for _ in _EPOCH_SUMMARY_COLS)
    pub.write_text(f"{header_pub}\n{existing_row_pub}\n")
    assert len(pub.read_text().splitlines()) == 2

    row_pub = {c: "1" for c in _EPOCH_SUMMARY_COLS}
    _append_tsv(str(pub), _EPOCH_SUMMARY_COLS, row_pub)
    lines_pub = pub.read_text().splitlines()
    assert len(lines_pub) == 3
    assert lines_pub[0] == header_pub  # header unchanged

    # -- softmax debug schema
    dbg = tmp_path / "good_debug.tsv"
    header_dbg = "\t".join(_EPOCH_SOFTMAX_DEBUG_COLS)
    existing_row_dbg = "\t".join("0" for _ in _EPOCH_SOFTMAX_DEBUG_COLS)
    dbg.write_text(f"{header_dbg}\n{existing_row_dbg}\n")

    row_dbg = {c: "1" for c in _EPOCH_SOFTMAX_DEBUG_COLS}
    _append_tsv(str(dbg), _EPOCH_SOFTMAX_DEBUG_COLS, row_dbg)
    lines_dbg = dbg.read_text().splitlines()
    assert len(lines_dbg) == 3
    assert lines_dbg[0] == header_dbg


# ---------------------------------------------------------------------------
# Restart config cleanliness tests
# ---------------------------------------------------------------------------

def _make_plain_config(cstep: int = 0) -> dict:
    """Build a minimal config with NO epoch-controller keys at all.

    This represents a plain run — no epoch_ctrl_mode, no epoch_nsubpath_ens,
    no epoch_nsubpath_vals, no adaptive_*/softmax_* keys.
    """
    return {
        "simulation": {
            "interfaces": [0.0, 0.3, 0.6],
            "shooting_moves": ["sh", "wf", "wf"],
            "tis_set": {
                "lambda_minus_one": False,
                "n_jumps": 2,
                "maxlength": 100,
                "allowmaxlength": False,
                "zero_momentum": False,
                "quantis": False,
                "accept_all": False,
                "mwf_nsubpath": 3,
            },
            "seed": 0,
            "steps": 100,
        },
        "runner": {"workers": 1},
        "current": {
            "cstep": cstep,
            "size": 3,
            "locked": [],
        },
    }


def _make_plain_state(cstep: int = 0) -> REPEX_state:
    state = REPEX_state(_make_plain_config(cstep), minus=True)
    state.initiate_ensembles()
    return state


# -- Test: plain run, no epoch controller keys in input -----------------------
def test_plain_run_restart_has_no_epoch_ctrl_keys():
    """mirror_epoch_ctrl must NOT write epoch-controller keys for a plain run."""
    state = _make_plain_state()

    mirror_epoch_ctrl(state, state.config)

    sim = state.config["simulation"]
    cur = state.config["current"]

    # No stale simulation keys
    for key in _STALE_SIM_KEYS:
        assert key not in sim, f"stale key '{key}' found in simulation"

    # No stale current keys
    for key in _STALE_CURRENT_KEYS:
        assert key not in cur, f"stale key '{key}' found in current"

    # No default-valued keys materialized
    for key in _DEFAULT_SIM_KEYS:
        assert key not in sim, f"default key '{key}' found in simulation"


# -- Test: validate_epoch_ctrl does not pollute plain config ------------------
def test_validate_does_not_inject_defaults_for_plain_run():
    """validate_epoch_ctrl must NOT inject default keys into a plain config."""
    state = _make_plain_state()
    sim = state.config["simulation"]

    validate_epoch_ctrl(state.config)

    for key in _DEFAULT_SIM_KEYS:
        assert key not in sim, (
            f"validate_epoch_ctrl injected default '{key}' into simulation"
        )


# -- Test: scheduled controller run still writes restart keys correctly -------
def test_scheduled_run_writes_restart_keys():
    """mirror_epoch_ctrl must write all needed keys for a scheduled run."""
    state = _make_state()  # has epoch_nsubpath_ens/vals → scheduled mode

    mirror_epoch_ctrl(state, state.config)

    sim = state.config["simulation"]
    cur = state.config["current"]

    assert "ensemble_nsubpath" in sim
    assert "ensemble_mwf_nsubpath" in sim
    assert "ensemble_move_counts" in cur
    assert "ensemble_last_fired_count" in cur


# -- Test: static mode with no divergence stays clean ------------------------
def test_static_mode_no_divergence_stays_clean():
    """Static mode with uniform n_jumps must not write controller keys."""
    cfg = _make_plain_config()
    cfg["simulation"]["epoch_ctrl_mode"] = "static"
    state = REPEX_state(cfg, minus=True)
    state.initiate_ensembles()

    mirror_epoch_ctrl(state, state.config)

    sim = state.config["simulation"]
    cur = state.config["current"]

    for key in _STALE_SIM_KEYS:
        assert key not in sim, f"stale key '{key}' in static-mode config"
    for key in _STALE_CURRENT_KEYS:
        assert key not in cur, f"stale key '{key}' in static-mode current"


# -- Test: static mode WITH per-ensemble divergence preserves it --------------
def test_static_mode_with_divergence_preserves_nsubpath():
    """If per-ensemble n_jumps diverges from global, mirror must persist it."""
    cfg = _make_plain_config()
    cfg["simulation"]["epoch_ctrl_mode"] = "static"
    state = REPEX_state(cfg, minus=True)
    state.initiate_ensembles()

    # Manually diverge ensemble 1's n_jumps from global (2 → 5)
    state.ensembles[1]["tis_set"]["n_jumps"] = 5

    mirror_epoch_ctrl(state, state.config)

    sim = state.config["simulation"]
    assert "ensemble_nsubpath" in sim, (
        "divergent per-ensemble n_jumps must be persisted"
    )
    assert sim["ensemble_nsubpath"][1] == 5


# -- Test: stale-key cleanup -------------------------------------------------
def test_stale_keys_removed_when_controller_inactive():
    """If config already has old epoch-controller keys but controller is
    inactive, mirror_epoch_ctrl must remove them."""
    state = _make_plain_state()
    sim = state.config["simulation"]
    cur = state.config["current"]

    # Inject stale keys as if from a previous run
    sim["ensemble_nsubpath"] = [2, 2, 2]
    sim["ensemble_mwf_nsubpath"] = [3, 3, 3]
    sim["softmax_ctrl_state"] = {"0": {}}
    sim["softmax_epoch_stats"] = {"0": {}}
    sim["epoch_mode"] = "global_step"
    sim["epoch_count"] = "attempted"
    sim["reward_ema_alpha"] = 0.1
    cur["ensemble_move_counts"] = {"0": 5, "1": 3, "2": 1}
    cur["ensemble_last_fired_count"] = {"0": -1, "1": -1, "2": -1}

    mirror_epoch_ctrl(state, state.config)

    for key in _STALE_SIM_KEYS:
        assert key not in sim, f"stale key '{key}' not cleaned up"
    for key in _STALE_CURRENT_KEYS:
        assert key not in cur, f"stale key '{key}' not cleaned up from current"
    for key in _DEFAULT_SIM_KEYS:
        assert key not in sim, f"default key '{key}' not cleaned up"
