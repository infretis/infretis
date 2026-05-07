"""Unit tests for `setup.expand_ensemble_move_policy`.

Pure-config tests. No REPEX, no MD, no I/O.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import pytest

from infretis.setup import (
    TOMLConfigError,
    expand_ensemble_move_policy,
)


# --------------------------------------------------------------------------- #
# fixtures / helpers                                                          #
# --------------------------------------------------------------------------- #


def _base_config(n_ens: int = 11) -> Dict[str, Any]:
    """Minimal config skeleton with only the fields the expander reads."""
    return {
        "simulation": {
            "interfaces": [round(0.1 * i, 4) for i in range(n_ens)],
            "tis_set": {},
        }
    }


def _with_policy(cfg: Dict[str, Any], **policy: Any) -> Dict[str, Any]:
    cfg["simulation"]["ensemble_move_policy"] = policy
    return cfg


# --------------------------------------------------------------------------- #
# 1-2. no-ops                                                                 #
# --------------------------------------------------------------------------- #


def test_no_policy_section_is_noop():
    cfg = _base_config(5)
    snap = copy.deepcopy(cfg)
    expand_ensemble_move_policy(cfg)
    assert cfg == snap


def test_disabled_policy_is_noop_and_stripped():
    cfg = _with_policy(_base_config(5), enabled=False, default_move="mwf")
    expand_ensemble_move_policy(cfg)
    assert "ensemble_move_policy" not in cfg["simulation"]
    assert "shooting_moves" not in cfg["simulation"]
    assert cfg["simulation"]["tis_set"] == {}


# --------------------------------------------------------------------------- #
# 3-7. expansion                                                              #
# --------------------------------------------------------------------------- #


def test_default_wf_with_mwf_range_expands_shooting_moves():
    cfg = _with_policy(
        _base_config(8),
        enabled=True,
        default_move="wf",
        mwf_ensembles=["002:004"],
    )
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == [
        "sh", "wf", "mwf", "mwf", "mwf", "wf", "wf", "wf",
    ]


def test_minus_move_defaults_to_sh_at_ensemble_000():
    cfg = _with_policy(_base_config(4), enabled=True, default_move="wf")
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"][0] == "sh"
    assert cfg["simulation"]["shooting_moves"][1:] == ["wf", "wf", "wf"]


def test_explicit_minus_move_is_honored():
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
        minus_move="wf",
    )
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"][0] == "wf"


def test_default_mwf_subcycle_small_becomes_canonical_scalar():
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
        default_mwf_subcycle_small=7,
    )
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["tis_set"]["mwf_subcycle_small"] == 7


def test_sparse_subcycle_table_expands_per_ensemble():
    cfg = _with_policy(
        _base_config(11),
        enabled=True,
        default_move="wf",
        default_mwf_subcycle_small=4,
        mwf_ensembles=["002:006", "009"],
        mwf_subcycle_small={"002:004": 2, "009": 8},
    )
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["tis_set"]["mwf_subcycle_small_by_ensemble"] == {
        "002": 2, "003": 2, "004": 2, "009": 8,
    }
    # ensembles 005, 006 covered by mwf_ensembles but not in subcycle map -> default
    assert cfg["simulation"]["tis_set"]["mwf_subcycle_small"] == 4


def test_missing_ensembles_use_defaults():
    cfg = _with_policy(
        _base_config(5),
        enabled=True,
        default_move="wf",
        default_mwf_subcycle_small=3,
        mwf_ensembles=["002"],
    )
    expand_ensemble_move_policy(cfg)
    moves = cfg["simulation"]["shooting_moves"]
    assert moves == ["sh", "wf", "mwf", "wf", "wf"]
    tis_set = cfg["simulation"]["tis_set"]
    assert tis_set["mwf_subcycle_small"] == 3
    assert "mwf_subcycle_small_by_ensemble" not in tis_set


# --------------------------------------------------------------------------- #
# 9-13. validation errors                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "selector",
    ["abc", "5:2", "010", "-1", "", " ", "1:abc", 3, ["003"]],
)
def test_invalid_selectors_raise(selector):
    cfg = _with_policy(
        _base_config(5),
        enabled=True,
        default_move="wf",
        mwf_ensembles=[selector],
    )
    with pytest.raises(TOMLConfigError):
        expand_ensemble_move_policy(cfg)


@pytest.mark.parametrize("bad_value", [0, -1, True, False, 1.5, "4", None])
def test_invalid_subcycle_values_raise(bad_value):
    cfg = _with_policy(
        _base_config(5),
        enabled=True,
        default_move="wf",
        mwf_ensembles=["002"],
        mwf_subcycle_small={"002": bad_value},
    )
    with pytest.raises(TOMLConfigError):
        expand_ensemble_move_policy(cfg)


def test_invalid_default_subcycle_value_raises():
    cfg = _with_policy(
        _base_config(5),
        enabled=True,
        default_move="wf",
        default_mwf_subcycle_small=0,
    )
    with pytest.raises(TOMLConfigError):
        expand_ensemble_move_policy(cfg)


def test_unknown_default_move_raises():
    cfg = _with_policy(_base_config(4), enabled=True, default_move="xx")
    with pytest.raises(TOMLConfigError):
        expand_ensemble_move_policy(cfg)


def test_unknown_minus_move_raises():
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
        minus_move="xx",
    )
    with pytest.raises(TOMLConfigError):
        expand_ensemble_move_policy(cfg)


def test_subcycle_override_for_non_mwf_ensemble_raises():
    cfg = _with_policy(
        _base_config(5),
        enabled=True,
        default_move="wf",
        # ensemble "002" is NOT in mwf_ensembles -> resolved as "wf"
        mwf_subcycle_small={"002": 4},
    )
    with pytest.raises(TOMLConfigError):
        expand_ensemble_move_policy(cfg)


def test_overlapping_mwf_ensembles_selectors_raise():
    cfg = _with_policy(
        _base_config(8),
        enabled=True,
        default_move="wf",
        mwf_ensembles=["002:004", "003"],
    )
    with pytest.raises(TOMLConfigError):
        expand_ensemble_move_policy(cfg)


def test_overlapping_subcycle_selectors_raise():
    cfg = _with_policy(
        _base_config(8),
        enabled=True,
        default_move="wf",
        mwf_ensembles=["002:004"],
        mwf_subcycle_small={"002:003": 2, "003": 4},
    )
    with pytest.raises(TOMLConfigError):
        expand_ensemble_move_policy(cfg)


# --------------------------------------------------------------------------- #
# 14-16. conflict policy                                                      #
# --------------------------------------------------------------------------- #


def test_conflict_with_existing_shooting_moves_raises():
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
    )
    cfg["simulation"]["shooting_moves"] = ["sh", "wf", "wf", "wf"]
    with pytest.raises(TOMLConfigError, match="shooting_moves"):
        expand_ensemble_move_policy(cfg)


def test_conflict_with_existing_scalar_raises():
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
        default_mwf_subcycle_small=4,
    )
    cfg["simulation"]["tis_set"]["mwf_subcycle_small"] = 9
    with pytest.raises(TOMLConfigError, match="mwf_subcycle_small"):
        expand_ensemble_move_policy(cfg)


def test_conflict_with_existing_by_ensemble_table_raises():
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
        mwf_ensembles=["002"],
    )
    cfg["simulation"]["tis_set"]["mwf_subcycle_small_by_ensemble"] = {"002": 3}
    with pytest.raises(TOMLConfigError, match="by_ensemble"):
        expand_ensemble_move_policy(cfg)


def test_unknown_conflict_policy_raises():
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
        conflict_policy="overwrite",
    )
    with pytest.raises(TOMLConfigError, match="conflict_policy"):
        expand_ensemble_move_policy(cfg)


# --------------------------------------------------------------------------- #
# 17-19. idempotence + canonical equivalence                                  #
# --------------------------------------------------------------------------- #


def test_policy_section_removed_after_successful_expansion():
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
        default_mwf_subcycle_small=4,
    )
    expand_ensemble_move_policy(cfg)
    assert "ensemble_move_policy" not in cfg["simulation"]


def test_expansion_is_idempotent():
    cfg = _with_policy(
        _base_config(11),
        enabled=True,
        default_move="wf",
        default_mwf_subcycle_small=4,
        mwf_ensembles=["002:006", "009"],
        mwf_subcycle_small={"002:004": 2, "009": 8},
    )
    expand_ensemble_move_policy(cfg)
    snap = copy.deepcopy(cfg)
    expand_ensemble_move_policy(cfg)  # no-op the second time
    assert cfg == snap


def test_ergonomic_equals_handwritten_canonical_after_expansion():
    ergonomic = _with_policy(
        _base_config(11),
        enabled=True,
        default_move="wf",
        minus_move="sh",
        default_mwf_subcycle_small=4,
        mwf_ensembles=["002:006", "009"],
        mwf_subcycle_small={"002:004": 2, "009": 8},
    )
    handwritten = _base_config(11)
    handwritten["simulation"]["shooting_moves"] = [
        "sh", "wf", "mwf", "mwf", "mwf", "mwf", "mwf", "wf", "wf", "mwf", "wf",
    ]
    handwritten["simulation"]["tis_set"]["mwf_subcycle_small"] = 4
    handwritten["simulation"]["tis_set"][
        "mwf_subcycle_small_by_ensemble"
    ] = {"002": 2, "003": 2, "004": 2, "009": 8}
    expand_ensemble_move_policy(ergonomic)
    assert ergonomic == handwritten


# --------------------------------------------------------------------------- #
# 20-21. selector edge cases                                                  #
# --------------------------------------------------------------------------- #


def test_open_right_range_works():
    cfg = _with_policy(
        _base_config(5),
        enabled=True,
        default_move="wf",
        mwf_ensembles=["003:"],
    )
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == [
        "sh", "wf", "wf", "mwf", "mwf",
    ]


def test_open_left_range_works():
    cfg = _with_policy(
        _base_config(5),
        enabled=True,
        default_move="wf",
        # ":002" expands to ["000","001","002"]; explicit selectors override
        # minus_move by design (same as test_minus_via_mwf_ensembles_overrides).
        mwf_ensembles=[":002"],
    )
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == [
        "mwf", "mwf", "mwf", "wf", "wf",
    ]


def test_unpadded_selector_normalizes():
    cfg = _with_policy(
        _base_config(5),
        enabled=True,
        default_move="wf",
        mwf_ensembles=["3"],  # equivalent to "003"
        mwf_subcycle_small={"3": 6},
    )
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == [
        "sh", "wf", "wf", "mwf", "wf",
    ]
    assert cfg["simulation"]["tis_set"]["mwf_subcycle_small_by_ensemble"] == {
        "003": 6,
    }


def test_minus_via_mwf_ensembles_overrides_minus_move():
    """If user explicitly puts '000' in mwf_ensembles, mwf wins over minus_move."""
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
        minus_move="sh",
        mwf_ensembles=["000"],
    )
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"][0] == "mwf"
