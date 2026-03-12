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
    _EPOCH_SUMMARY_FNAME,
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
    assert abs(float(r["lambda_max"]) - 0.42) < 1e-5
    assert int(r["n_jumps_old"]) == 2   # initial global n_jumps
    assert int(r["n_jumps_new"]) == 4   # epoch 1 → schedule[1] = 4

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

    assert r["ctrl_mode"] == "adaptive"
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
    assert r["ctrl_mode"] == "scheduled"
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
    assert r["ctrl_mode"] == "adaptive"
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
    assert r["ctrl_mode"] == "softmax_dirichlet"
    assert r["ctrl_action"] in ("sample_hold", "sample_set")
    assert r["ctrl_reason"] == "softmax_dirichlet_epoch_update"
    # First epoch: no prior reward → reward column is "nan"
    assert r["reward"] == "nan"
    assert "choice_idx" in r
    assert "probs_json" in r

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

    # update_count must now be 1 (first logit update applied at epoch 2)
    assert state2.softmax_ctrl_state[2]["update_count"] == 1, (
        "epoch 2 boundary must apply first logit update (update_count → 1)"
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

    lambda_max_vals = [0.1, 0.4, 0.4, 0.9]
    for lm in lambda_max_vals:
        update_epoch_stats(state, 1, accepted=True, path_length=10.0, subcycles=2, lambda_max=lm)

    buf = state.ensemble_epoch_stats[1]
    # Move 1: prev_obs=0.1, no increment.
    # Move 2: delta=0.3, gain_sq_sum+=0.09, gain_sq_n=1.
    # Move 3: delta=0.0, gain_sq_sum+=0.0, gain_sq_n=2.
    # Move 4: delta=0.5, gain_sq_sum+=0.25, gain_sq_n=3.
    expected_sq_sum = (0.4 - 0.1)**2 + (0.4 - 0.4)**2 + (0.9 - 0.4)**2  # 0.34
    assert abs(buf["gain_sq_sum"] - expected_sq_sum) < 1e-12
    assert buf["gain_sq_n"] == 3
    assert abs(buf["prev_obs"] - 0.9) < 1e-12


# ---------------------------------------------------------------------------
# Unit test 41 — _compute_epoch_reward uses empirical formula from buffer
# ---------------------------------------------------------------------------

def test_sd_empirical_reward_from_buffer():
    """_compute_epoch_reward with empirical proxy returns (G + eps) / (avg_sub + eps)."""
    sim = {
        "reward_proxy": "empirical_dirichlet_lambda_v1",
        "reward_eps_gain": 1e-12,
        "reward_eps_cost": 1e-12,
    }
    buf = _empty_stats()
    buf["gain_sq_sum"] = 0.08
    buf["gain_sq_n"] = 2
    buf["subcycles_sum"] = 3
    buf["subcycles_n"] = 1

    reward = _compute_epoch_reward(buf, sim)
    gain = 0.08 / 2  # 0.04
    avg_sub = 3.0
    eps = 1e-12
    expected = (gain + eps) / (avg_sub + eps)
    assert abs(reward - expected) < 1e-20


# ---------------------------------------------------------------------------
# Unit test 42 — first accepted move initialises prev_obs, no increment
# ---------------------------------------------------------------------------

def test_sd_empirical_first_obs_no_increment():
    """First accepted move: prev_obs set, gain_sq_n stays 0."""
    state = _make_state()
    state.config["simulation"]["reward_proxy"] = "empirical_dirichlet_lambda_v1"
    state.ensembles[1]["interfaces"] = (float("-inf"), 0.0, 1.0)

    update_epoch_stats(state, 1, accepted=True, path_length=10.0, subcycles=2, lambda_max=0.5)

    buf = state.ensemble_epoch_stats[1]
    assert abs(buf["prev_obs"] - 0.5) < 1e-12
    assert buf["gain_sq_n"] == 0


# ---------------------------------------------------------------------------
# Unit test 43 — zero gain when no increments recorded
# ---------------------------------------------------------------------------

def test_sd_empirical_zero_reward_when_no_increments():
    """gain_sq_n=0 → reward is eps_gain / (avg_sub + eps_cost)."""
    sim = {
        "reward_proxy": "empirical_dirichlet_lambda_v1",
        "reward_eps_gain": 1e-12,
        "reward_eps_cost": 1e-12,
    }
    buf = _empty_stats()  # gain_sq_n = 0
    buf["subcycles_sum"] = 3
    buf["subcycles_n"] = 1

    reward = _compute_epoch_reward(buf, sim)
    eps = 1e-12
    expected = eps / (3.0 + eps)
    assert abs(reward - expected) < 1e-20


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

    # First: ACC move — initialises prev_obs
    update_epoch_stats(state, 2, accepted=True, path_length=10.0, subcycles=1, lambda_max=-0.4)
    buf = state.ensemble_epoch_stats.get(2, {})
    assert not math.isnan(buf.get("prev_obs", float("nan"))), "prev_obs must be set after ACC"
    assert buf.get("gain_sq_n", 0) == 0, "no increment on first ACC move"

    # Second: REJ move — curr_obs = prev_obs, delta = 0
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

    # Read TSV epoch-2 row
    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows) == 2, "must have epoch-1 and epoch-2 TSV rows"
    r = rows[1]
    reward_val = float(r["reward"])

    # Expected empirical reward: gain from 3 moves (1st initializes, 2nd and 3rd give deltas)
    # gain_sq_sum = (h2-h1)^2 + (h3-h2)^2, gain_sq_n = 2
    gain_sq_sum = (h2 - h1) ** 2 + (h3 - h2) ** 2
    gain = gain_sq_sum / 2
    eps = 1e-12
    expected = (gain + eps) / (1.0 + eps)  # avg_sub=1.0 (subcycles=1 each)
    assert abs(reward_val - expected) < 1e-9, (
        f"Empirical reward mismatch: got {reward_val}, expected {expected}"
    )

    # Phase-1 reward (avg_lambda_max / avg_sub) would differ
    phase1_avg_lm = (lm1 + lm2 + lm3) / 3
    phase1_reward = (phase1_avg_lm + eps) / (1.0 + eps)
    assert abs(reward_val - phase1_reward) > 1e-6, (
        "Reward must use empirical formula, not phase-1 lambda-avg formula"
    )


# ---------------------------------------------------------------------------
# Integration test 48 — restart roundtrip preserves gain state
# ---------------------------------------------------------------------------

def test_sd_empirical_restart_roundtrip_preserves_gain_state(dw_setup):
    """Stop mid-epoch; restart.toml preserves prev_obs/gain_sq_*; restore continues."""
    state, tmp_path = dw_setup
    keys = _sd_sim_keys()
    keys["reward_proxy"] = "empirical_dirichlet_lambda_v1"
    state.config["simulation"].update(keys)
    ENS_NUM = 1  # ensemble index 2

    # Epoch 1: 3 REJ moves → ctrl_state initialized, buffer reset
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
    assert buf_before.get("gain_sq_n", 0) == 1, "2 ACC moves → 1 gain increment"
    assert not math.isnan(buf_before.get("prev_obs", float("nan")))
    prev_obs_before = buf_before["prev_obs"]
    gain_sq_n_before = buf_before["gain_sq_n"]
    gain_sq_sum_before = buf_before["gain_sq_sum"]

    # Serialize via mirror_epoch_ctrl
    mirror_epoch_ctrl(state, state.config)
    saved_stats = state.config["simulation"]["softmax_epoch_stats"]["2"]
    assert saved_stats["gain_sq_n"] == gain_sq_n_before
    assert not math.isnan(saved_stats["prev_obs"])

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
    assert buf2.get("gain_sq_n", 0) == gain_sq_n_before, "gain_sq_n must survive restart"
    assert not math.isnan(buf2.get("prev_obs", float("nan"))), "prev_obs must survive restart"
    assert abs(buf2.get("prev_obs", 0.0) - prev_obs_before) < 1e-10
    assert abs(buf2.get("gain_sq_sum", 0.0) - gain_sq_sum_before) < 1e-10

    # Complete epoch 2: 1 more ACC move + trigger epoch boundary
    lm3 = lambda_i + 0.8 * denom
    update_epoch_stats(state2, 2, accepted=True, path_length=10.0, subcycles=1, lambda_max=lm3)
    state2.ensemble_move_counts[2] = 6  # trigger epoch 2
    apply_epoch_ctrl(state2, 0)

    # Epoch 2 fired: update_count advanced, reward is not nan
    assert state2.softmax_ctrl_state[2]["update_count"] == 1, (
        "epoch 2 must apply first logit update"
    )
    tsv_path = tmp_path / _EPOCH_SUMMARY_FNAME
    rows = list(csv.DictReader(tsv_path.open(), delimiter="\t"))
    assert len(rows) == 2
    r2 = rows[1]
    assert r2["reward"] != "nan", "epoch-2 reward must not be nan"


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
    assert rows[0]["ctrl_mode"] == "softmax_dirichlet"
    assert rows[0]["reward"] == "nan"  # first epoch: no prior reward


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
    assert r["ctrl_mode"] == "softmax_dirichlet"
