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

import numpy as np
import tomli
import tomli_w
import pytest

import infretis.core.tis as tis
from infretis.classes.repex import REPEX_state
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
