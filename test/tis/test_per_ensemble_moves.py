"""Per-ensemble selection of WF / MWF / SH via simulation.shooting_moves.

These tests document and verify that users can mix moves per ensemble using
the existing list-valued `simulation.shooting_moves` config key, e.g.::

    [simulation]
    shooting_moves = ["sh", "wf", "mwf", "mwf", "wf"]

No new TOML keys are introduced. Selection is positional: ensemble `i` runs
`shooting_moves[i]`.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from infretis.classes.repex import REPEX_state
from infretis.core import tis as tis_mod
from infretis.core.tis import compute_weight, select_shoot
from infretis.setup import TOMLConfigError, check_config


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


class _FakeEngine:
    """Stand-in for an MD engine; only the lifecycle hooks are exercised."""

    def set_mdrun(self, *_a: Any, **_kw: Any) -> None:
        pass

    def clean_up(self) -> None:
        pass


class _StubPath:
    path_number = 0


def _picked_one(ens_set: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {0: {"ens": ens_set, "traj": _StubPath(), "eng_idx": {"e": 0}}}


def _ens_set(move: str) -> Dict[str, Any]:
    return {
        "interfaces": (-1.0, 0.0, 1.0),
        "tis_set": {"quantis": False},
        "mc_move": move,
        "ens_name": "001",
        "start_cond": ("L",),
    }


# --------------------------------------------------------------------------- #
# 1. dispatcher                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "move, target",
    [("sh", "shoot"), ("wf", "wire_fencing"), ("mwf", "multires_wire_fencing")],
)
def test_select_shoot_dispatches_each_move(monkeypatch, move, target):
    """select_shoot must route each registered move key to its function."""
    calls: List[str] = []

    def make_sentinel(name: str):
        def _sentinel(ens_set, path, engine, start_cond=("L",)):
            calls.append(name)
            return True, path, "ACC"

        return _sentinel

    monkeypatch.setattr(tis_mod, "shoot", make_sentinel("shoot"))
    monkeypatch.setattr(tis_mod, "wire_fencing", make_sentinel("wire_fencing"))
    monkeypatch.setattr(
        tis_mod, "multires_wire_fencing", make_sentinel("multires_wire_fencing")
    )
    monkeypatch.setattr(tis_mod, "ENGINES", {"e": [_FakeEngine()]})

    accept, _paths, status = select_shoot(_picked_one(_ens_set(move)))

    assert calls == [target]
    assert accept is True
    assert status == "ACC"


def test_select_shoot_rejects_unknown_move(monkeypatch):
    """An unregistered move key surfaces as a KeyError at dispatch time."""
    monkeypatch.setattr(tis_mod, "ENGINES", {"e": [_FakeEngine()]})
    with pytest.raises(KeyError):
        select_shoot(_picked_one(_ens_set("does-not-exist")))


# --------------------------------------------------------------------------- #
# 2. per-ensemble plumbing through REPEX                                      #
# --------------------------------------------------------------------------- #


def test_pensembles_mc_move_matches_shooting_moves():
    """REPEX.initiate_ensembles assigns shooting_moves[i] to ensemble i."""
    moves = ["sh", "wf", "mwf", "mwf", "wf"]
    cfg = {
        "simulation": {
            "interfaces": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "shooting_moves": moves,
            "tis_set": {"lambda_minus_one": False},
        }
    }
    stub = REPEX_state.__new__(REPEX_state)
    stub.config = cfg
    REPEX_state.initiate_ensembles(stub)

    got = [stub.ensembles[i]["mc_move"] for i in range(len(moves))]
    assert got == moves
    # ens_name is the canonical zero-padded string used elsewhere
    assert [stub.ensembles[i]["ens_name"] for i in range(len(moves))] == [
        f"{i:03d}" for i in range(len(moves))
    ]


# --------------------------------------------------------------------------- #
# 3. weight equivalence                                                       #
# --------------------------------------------------------------------------- #


def test_compute_weight_wf_and_mwf_agree_on_same_path():
    """Paths tagged "wf" and "mwf" receive the same High-Acceptance weight.

    The asymmetric branch (move=="wf" and weight==0 and endp=="R") is
    explicitly avoided by choosing a path whose WF weight is non-zero.
    """
    from test.tis.test_tis import INP_PATH

    interfaces = [-1.0, 0.0, 1.0]
    w_wf = compute_weight(INP_PATH, interfaces, "wf")
    w_mwf = compute_weight(INP_PATH, interfaces, "mwf")
    assert w_wf == w_mwf


# --------------------------------------------------------------------------- #
# 4. setup validation accepts "mwf"                                           #
# --------------------------------------------------------------------------- #


def _valid_cfg_with_moves(moves: List[str]) -> Dict[str, Any]:
    return {
        "simulation": {
            "interfaces": [0.1 * i for i in range(len(moves))],
            "shooting_moves": moves,
            "tis_set": {},
            "ensemble_engines": [["engine"] for _ in moves],
        },
        "runner": {"workers": 1},
        "current": {},
        "engine": {"class": "turtlemd"},
    }


def test_check_config_accepts_mwf_in_shooting_moves():
    """`check_config` must not reject mixed wf/mwf/sh shooting_moves."""
    cfg = _valid_cfg_with_moves(["sh", "wf", "mwf", "mwf", "wf"])
    # Smoke: no exception raised.
    check_config(cfg)


def test_check_config_still_rejects_too_few_moves():
    """Sanity: pre-existing length validation is unaffected."""
    cfg = _valid_cfg_with_moves(["sh", "wf", "mwf"])
    cfg["simulation"]["interfaces"] = [0.0, 0.1, 0.2, 0.3, 0.4]  # 5 ens, 3 moves
    with pytest.raises(TOMLConfigError):
        check_config(cfg)
