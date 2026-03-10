"""Tests for epoch-wise per-ensemble n_jumps controller.

Verifies four properties end-to-end:
  1. Only the targeted ensemble's n_jumps alternates; the untargeted WF
     ensemble is never touched.
  2. Stopping mid-epoch and restarting restores the exact per-ensemble
     value (via ensemble_nsubpath in restart.toml).
  3. The first post-restart moves keep the restored value until the next
     epoch boundary — the controller does NOT fire mid-epoch.
  4. The n_jumps value that select_shoot() would log to move_blocks.tsv
     reflects the alternating schedule for the targeted ensemble and
     the constant original value for the untargeted ensemble.
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
        state._apply_epoch_ctrl()

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
    state._apply_epoch_ctrl()
    assert state.ensembles[1]["tis_set"]["n_jumps"] == 4  # epoch 1 applied
    assert state.ensembles[2]["tis_set"]["n_jumps"] == 2  # untouched

    # --- Phase B: "stop" at cstep=15 (mid-epoch 2) ---
    state.config["current"]["cstep"] = 15
    state._mirror_epoch_params()

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
    state2._apply_epoch_ctrl()
    assert state2.ensembles[1]["tis_set"]["n_jumps"] == 4, (
        "cstep=15 is mid-epoch; n_jumps must stay 4 until cstep=20"
    )

    # Property 3 continued — next boundary at cstep=20 fires correctly
    state2.config["current"]["cstep"] = 20
    state2._apply_epoch_ctrl()
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
    reading n_jumps → writing the TSV row → calling _apply_epoch_ctrl().
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
        state._apply_epoch_ctrl()

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
