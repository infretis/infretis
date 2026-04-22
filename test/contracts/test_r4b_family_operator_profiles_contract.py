"""Contracts for R4b family-level operator profiles."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from infretis.math_core.predicates.ensembles import (
    EnsembleDescriptor,
    EnsembleFiberDescriptor,
    make_ensemble_context,
)
from infretis.math_core.weighted_family import WeightedObservedPath
from infretis.reducer_core import (
    FAMILY_OPERATOR_PROFILE_V1,
    FAMILY_SELECTION_ALL_ACCEPTED_V1,
    FAMILY_SELECTION_BROAD_ENSEMBLE_V1,
    FAMILY_SELECTION_EXACT_RANK_FIBER_V1,
    LEFT_INTERVAL_MASS_V1,
    SCALAR_INTERVAL_BINS_V1,
    FamilyOperatorProfile,
    FamilySelection,
    ScalarIntervalBinDiscretization,
    compute_family_operator_profile,
    select_observed_path_family,
)
from infretis.replay_semantic_core import build_observed_path
from infretis.syntax_and_context import make_observation_only_reduction_context

pytestmark = [pytest.mark.contract]


def _pathblob(
    object_id: str,
    values: list[float],
    *,
    accepted: bool = True,
    path_number: int | None = None,
) -> dict:
    payload = {
        "accepted": accepted,
        "status": "ACC" if accepted else "REJ",
        "per_slice_cv_values": {"order_parameter": values},
        "sampling_cadence": {
            "kind": "explicit_step_intervals_v1",
            "step_intervals": [1.0 for _ in range(len(values) - 1)],
        },
    }
    if path_number is not None:
        payload["path_number_new"] = path_number
    event = {
        "object_type": "PathBlob",
        "object_id": object_id,
        "payload": payload,
    }
    if path_number is not None:
        event["logical_sequence_number"] = path_number
    return event


def _context():
    return make_observation_only_reduction_context(LEFT_INTERVAL_MASS_V1)


def _observed_paths():
    context = _context()
    return [
        build_observed_path(
            _pathblob("pb-low", [0.1, 0.2, 0.3], path_number=1), context
        ),
        build_observed_path(
            _pathblob("pb-mid", [0.1, 0.6, 0.4], path_number=2), context
        ),
        build_observed_path(
            _pathblob("pb-high", [0.1, 1.2, 0.4], path_number=3), context
        ),
        build_observed_path(
            _pathblob(
                "pb-rejected",
                [0.1, 0.8, 0.4],
                accepted=False,
                path_number=4,
            ),
            context,
        ),
    ]


def _weighted_toy_family():
    context = _context()
    p1 = build_observed_path(_pathblob("pb-1", [0.1, 0.4, 0.6]), context)
    p2 = build_observed_path(_pathblob("pb-2", [0.7, 0.2, 0.9]), context)
    return [
        WeightedObservedPath(observed_path=p1, weight=1.0),
        WeightedObservedPath(observed_path=p2, weight=1.0),
    ]


def _discretization() -> ScalarIntervalBinDiscretization:
    return ScalarIntervalBinDiscretization(
        version=SCALAR_INTERVAL_BINS_V1,
        cv_key="order_parameter",
        cell_edges=(0.0, 0.5, 1.0),
    )


def test_typed_reducer_output_construction() -> None:
    profile = compute_family_operator_profile(
        _weighted_toy_family(),
        FamilySelection.all_accepted(),
        _discretization(),
        _context(),
    )

    assert isinstance(profile, FamilyOperatorProfile)
    assert profile.reduction_version == FAMILY_OPERATOR_PROFILE_V1
    assert profile.selected_family.mode == FAMILY_SELECTION_ALL_ACCEPTED_V1
    assert profile.partition.version == "v1"
    assert profile.partition.discretization_version == SCALAR_INTERVAL_BINS_V1
    assert profile.occupation.dense == pytest.approx((3.0, 1.0))
    assert profile.transition.dense == (
        (pytest.approx(1.0), pytest.approx(2.0)),
        (pytest.approx(1.0), pytest.approx(0.0)),
    )
    assert profile.tail.values == pytest.approx((1.0, 0.0))
    assert profile.upward_cut.values == pytest.approx((2.0, 0.0))
    assert profile.cut_profile.tail.values == profile.tail.values
    assert profile.cut_profile.cut.values == profile.upward_cut.values


def test_family_selection_all_accepted_paths() -> None:
    selected = select_observed_path_family(
        _observed_paths(),
        FamilySelection.all_accepted(),
        _context(),
    )

    assert selected.selection.mode == FAMILY_SELECTION_ALL_ACCEPTED_V1
    assert selected.selection.selected_path_ids == (
        "pb-low",
        "pb-mid",
        "pb-high",
    )
    assert selected.exactness_status["explicit_rejected_path_count"] == 1


def test_family_selection_by_broad_ensemble_membership() -> None:
    ensemble_context = make_ensemble_context((0.0, 0.5, 1.0))
    selected = select_observed_path_family(
        _observed_paths(),
        FamilySelection.broad_ensemble(
            EnsembleDescriptor(threshold_index=1),
            context=ensemble_context,
        ),
        _context(),
    )

    assert selected.selection.mode == FAMILY_SELECTION_BROAD_ENSEMBLE_V1
    assert selected.selection.selected_path_ids == ("pb-mid", "pb-high")
    assert selected.exactness_status["structural_screen_source"] == (
        "infretis.math_core.predicates.ensembles"
    )


def test_family_selection_by_exact_rank_fiber() -> None:
    ensemble_context = make_ensemble_context((0.0, 0.5, 1.0))
    selected = select_observed_path_family(
        _observed_paths(),
        FamilySelection.exact_rank_fiber(
            EnsembleFiberDescriptor(rho_index=1),
            context=ensemble_context,
        ),
        _context(),
    )

    assert selected.selection.mode == FAMILY_SELECTION_EXACT_RANK_FIBER_V1
    assert selected.selection.selected_path_ids == ("pb-mid",)


def test_reducer_delegates_to_math_core_family_and_operator_surfaces() -> None:
    from infretis.reducer_core import r4b_family_operator_profiles as r4b

    with (
        patch.object(
            r4b.occupation_transition_reducts,
            "family_occupation_measure",
            wraps=r4b.occupation_transition_reducts.family_occupation_measure,
        ) as family_occupation,
        patch.object(
            r4b.occupation_transition_reducts,
            "family_transition_measure",
            wraps=r4b.occupation_transition_reducts.family_transition_measure,
        ) as family_transition,
        patch.object(
            r4b.occupation_transition_reducts,
            "family_partitioned_occupation_transition",
            wraps=(
                r4b.occupation_transition_reducts
                .family_partitioned_occupation_transition
            ),
        ) as family_partitioned,
        patch.object(
            r4b.partition_operators,
            "tail_operator",
            wraps=r4b.partition_operators.tail_operator,
        ) as tail,
        patch.object(
            r4b.partition_operators,
            "upward_cut_operator",
            wraps=r4b.partition_operators.upward_cut_operator,
        ) as upward_cut,
        patch.object(
            r4b.partition_operators,
            "cut_profile_operator",
            wraps=r4b.partition_operators.cut_profile_operator,
        ) as cut_profile,
    ):
        compute_family_operator_profile(
            _weighted_toy_family(),
            FamilySelection.all_accepted(),
            _discretization(),
            _context(),
        )

    assert family_occupation.call_count >= 1
    assert family_transition.call_count >= 1
    assert family_partitioned.call_count == 1
    assert tail.call_count >= 1
    assert upward_cut.call_count >= 1
    assert cut_profile.call_count == 1


def test_family_sum_and_operator_outputs_agree_with_toy_example() -> None:
    profile = compute_family_operator_profile(
        _weighted_toy_family(),
        FamilySelection.all_accepted(),
        _discretization(),
        _context(),
    )

    assert profile.raw_occupation.mass_by_cell == {
        0.1: pytest.approx(1.0),
        0.4: pytest.approx(1.0),
        0.7: pytest.approx(1.0),
        0.2: pytest.approx(1.0),
    }
    assert profile.raw_transition.mass_by_edge == {
        (0.1, 0.4): pytest.approx(1.0),
        (0.4, 0.6): pytest.approx(1.0),
        (0.7, 0.2): pytest.approx(1.0),
        (0.2, 0.9): pytest.approx(1.0),
    }
    assert profile.occupation.total_mass() == pytest.approx(4.0)
    assert profile.transition.total_mass() == pytest.approx(4.0)
    assert profile.tail.values == pytest.approx((1.0, 0.0))
    assert profile.upward_cut.values == pytest.approx((2.0, 0.0))


def test_provenance_and_exactness_fields_are_stable() -> None:
    profile = compute_family_operator_profile(
        _weighted_toy_family(),
        FamilySelection.all_accepted(),
        _discretization(),
        _context(),
    )

    assert (
        profile.exactness_status["reducer_version"]
        == FAMILY_OPERATOR_PROFILE_V1
    )
    assert (
        profile.exactness_status["math_core_family_aggregation"]
        == "delegated"
    )
    assert (
        profile.exactness_status["math_core_partition_pushforward"]
        == "delegated"
    )
    assert (
        profile.exactness_status["math_core_partition_operators"]
        == "delegated"
    )
    assert profile.provenance["math_core_entry_points"] == (
        "family_occupation_measure",
        "family_transition_measure",
        "family_partitioned_occupation_transition",
        "tail_operator",
        "upward_cut_operator",
        "cut_profile_operator",
    )


def test_reducer_has_no_local_tail_cut_or_cell_logic() -> None:
    from infretis.reducer_core import r4b_family_operator_profiles as r4b

    text = Path(r4b.__file__).read_text()
    forbidden_fragments = (
        "def tail_",
        "def upward_cut",
        "def cut_profile",
        "cell_index(",
        "pushforward_occupation_to_partition(",
        "pushforward_transition_to_partition(",
    )
    for fragment in forbidden_fragments:
        assert fragment not in text
    assert "family_partitioned_occupation_transition" in text
    assert "partition_operators.tail_operator" in text
    assert "partition_operators.upward_cut_operator" in text
    assert "partition_operators.cut_profile_operator" in text


def test_import_boundary_purity() -> None:
    from infretis.reducer_core import r4b_family_operator_profiles as r4b

    src = r4b.__file__
    assert src is not None
    tree = ast.parse(Path(src).read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)

    for forbidden in (
        "run_asymmetric_dw_benchmark",
        "run_bootstrap",
        "operator_geometry",
        "controller",
        "placement",
        "oracle",
        "committor",
    ):
        assert not any(forbidden in module for module in imports), imports
