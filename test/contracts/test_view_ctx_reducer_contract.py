"""Contracts for reducer-core ViewCtx governance."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from infretis.math_core.predicates.ensembles import (
    EnsembleDescriptor,
    EnsembleFiberDescriptor,
    make_ensemble_context,
)
from infretis.math_core.weighted_family import WeightedObservedPath
from infretis.reducer_core import (
    LEFT_INTERVAL_MASS_V1,
    SCALAR_INTERVAL_BINS_V1,
    FAMILY_SELECTION_ALL_ACCEPTED_V1,
    FamilySelection,
    PrimitiveFamily,
    ScalarIntervalBinDiscretization,
    ViewCtx,
    ViewReductionKind,
    ViewSelectorKind,
    VersionedTimeWeightPolicy,
    compute_family_operator_profile,
    compute_family_operator_profile_from_view,
    compute_nu_E,
    compute_nu_E_from_view,
)
from infretis.replay_semantic_core import build_observed_path
from infretis.syntax_and_context import (
    ReductionContext,
    make_observation_only_reduction_context,
)

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


def _context() -> ReductionContext:
    return make_observation_only_reduction_context(LEFT_INTERVAL_MASS_V1)


def _view(
    *,
    allowed_reductions: tuple[ViewReductionKind, ...] | None = None,
) -> ViewCtx:
    return ViewCtx.standard_observed_path_view(
        _context(),
        predicate_context=make_ensemble_context((0.0, 0.5, 1.0)),
        allowed_reductions=allowed_reductions,
    )


def _discretization() -> ScalarIntervalBinDiscretization:
    return ScalarIntervalBinDiscretization(
        version=SCALAR_INTERVAL_BINS_V1,
        cv_key="order_parameter",
        cell_edges=(0.0, 0.5, 1.0),
    )


def _observed_path(values: list[float], object_id: str = "pb") -> object:
    return build_observed_path(_pathblob(object_id, values), _context())


def _family() -> list[WeightedObservedPath]:
    return [
        WeightedObservedPath(
            observed_path=_observed_path([0.1, 0.4, 0.6], "pb-1"),
            weight=1.0,
        ),
        WeightedObservedPath(
            observed_path=_observed_path([0.7, 0.2, 0.9], "pb-2"),
            weight=1.0,
        ),
        WeightedObservedPath(
            observed_path=_observed_path([0.1, 0.8, 0.4], "pb-rej"),
            weight=1.0,
            metadata={"accepted": False, "status": "REJ"},
        ),
    ]


def test_view_ctx_construction_and_immutability() -> None:
    view = _view()

    assert view.view_id == "standard_reduction_view_v1"
    assert view.carrier() is PrimitiveFamily.ACCEPTED_OBSERVED_PATHS
    assert view.weights() == (LEFT_INTERVAL_MASS_V1.version,)
    assert ViewReductionKind.R4_NU_E in view.reductions()
    with pytest.raises(FrozenInstanceError):
        view.view_id = "mutated"  # type: ignore[misc]


def test_view_ctx_generates_default_selector_set() -> None:
    view = _view()

    assert view.selectors() == (
        ViewSelectorKind.ALL_ACCEPTED,
        ViewSelectorKind.BROAD_ENSEMBLE,
        ViewSelectorKind.EXACT_RANK_FIBER,
    )
    assert (
        view.default_selector().selector_kind
        is ViewSelectorKind.ALL_ACCEPTED
    )


def test_illegal_selector_view_combination_is_rejected() -> None:
    view = ViewCtx.standard_observed_path_view(
        _context(),
        allowed_reductions=(ViewReductionKind.R4B_FAMILY_OPERATOR_PROFILE,),
    )

    with pytest.raises(ValueError, match="requires ViewCtx.predicate_context"):
        view.broad_ensemble_selector(EnsembleDescriptor(threshold_index=1))

    other_context = make_ensemble_context((0.0, 0.5, 1.0))
    bad_selection = FamilySelection.broad_ensemble(
        EnsembleDescriptor(threshold_index=1),
        context=other_context,
    )
    with pytest.raises(ValueError, match="outside this ViewCtx"):
        compute_family_operator_profile(
            _family(),
            bad_selection,
            _discretization(),
            _view(),
        )


def test_illegal_weight_policy_reducer_combination_is_rejected() -> None:
    bad_policy = VersionedTimeWeightPolicy(
        version="unsupported_time_weight_policy_v1",
        transition_weight_convention="unsupported",
        supported_sampling_cadence_forms=("explicit_step_intervals_v1",),
    )
    bad_context = make_observation_only_reduction_context(bad_policy)

    with pytest.raises(ValueError, match="time_weight_policy"):
        ViewCtx.standard_observed_path_view(bad_context)


def test_illegal_reduction_outside_allowed_set_is_rejected() -> None:
    view = _view(
        allowed_reductions=(
            ViewReductionKind.R4B_FAMILY_OPERATOR_PROFILE,
        )
    )

    with pytest.raises(ValueError, match="not allowed"):
        compute_nu_E_from_view(
            _observed_path([0.1, 0.4, 0.6]),
            _discretization(),
            view,
        )


def test_r4_works_through_view_ctx_resolved_path() -> None:
    nu, edges = compute_nu_E_from_view(
        _observed_path([0.1, 0.4, 0.6]),
        _discretization(),
        _view(),
    )

    assert nu == pytest.approx({0: 2.0})
    assert edges == pytest.approx({(0, 0): 1.0, (0, 1): 1.0})


def test_r4b_works_through_view_ctx_resolved_path() -> None:
    profile = compute_family_operator_profile_from_view(
        _family(),
        _view(),
        _discretization(),
    )

    assert profile.selected_family.mode == FAMILY_SELECTION_ALL_ACCEPTED_V1
    assert profile.selected_family.selected_path_ids == ("pb-1", "pb-2")
    assert profile.occupation.dense == pytest.approx((3.0, 1.0))
    assert profile.tail.values == pytest.approx((1.0, 0.0))
    assert profile.upward_cut.values == pytest.approx((2.0, 0.0))
    assert profile.provenance["view_ctx"]["primitive_family"] == (
        PrimitiveFamily.ACCEPTED_OBSERVED_PATHS.value
    )


def test_all_accepted_paths_remains_default_family_selector() -> None:
    profile = compute_family_operator_profile_from_view(
        _family(),
        _view(),
        _discretization(),
    )

    assert profile.selected_family.selector == {
        "kind": ViewSelectorKind.ALL_ACCEPTED.value
    }
    assert profile.selected_family.input_path_count == 3
    assert profile.selected_family.selected_path_count == 2


def test_interval_behavior_is_partition_pushforward_derived() -> None:
    view = _view()
    profile = compute_family_operator_profile_from_view(
        _family(),
        view,
        _discretization(),
    )

    assert profile.selected_family.selected_path_ids == ("pb-1", "pb-2")
    assert profile.partition.edges == (0.0, 0.5, 1.0)
    assert profile.exactness_status["math_core_partition_pushforward"] == (
        "delegated"
    )
    assert profile.occupation.dense == pytest.approx((3.0, 1.0))


def test_ensemble_selectors_are_view_generated() -> None:
    view = _view()
    broad = view.broad_ensemble_selector(EnsembleDescriptor(threshold_index=1))
    fiber = view.exact_rank_fiber_selector(
        EnsembleFiberDescriptor(rho_index=1)
    )

    broad_profile = compute_family_operator_profile_from_view(
        _family(),
        view,
        _discretization(),
        selector=broad,
    )
    fiber_profile = compute_family_operator_profile_from_view(
        _family(),
        view,
        _discretization(),
        selector=fiber,
    )

    assert broad_profile.selected_family.selected_path_ids == ("pb-1", "pb-2")
    assert fiber_profile.selected_family.selected_path_ids == ("pb-1", "pb-2")


def test_old_reducer_entry_points_still_work() -> None:
    observed = _observed_path([0.1, 0.4, 0.6])

    old_nu, old_edges = compute_nu_E(observed, _discretization(), _context())
    new_nu, new_edges = compute_nu_E(observed, _discretization(), _view())
    assert new_nu == pytest.approx(old_nu)
    assert new_edges == pytest.approx(old_edges)

    old_profile = compute_family_operator_profile(
        _family(),
        FamilySelection.all_accepted(),
        _discretization(),
        _context(),
    )
    new_profile = compute_family_operator_profile(
        _family(),
        _view().default_selector(cv_key="order_parameter"),
        _discretization(),
        _view(),
    )
    assert new_profile.occupation.dense == pytest.approx(
        old_profile.occupation.dense
    )


def test_view_ctx_import_boundary_is_clean() -> None:
    from infretis.reducer_core import view_context

    src = view_context.__file__
    assert src is not None
    tree = ast.parse(Path(src).read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)

    for forbidden in (
        "replay_bridge",
        "replay_semantic_core",
        "operator_geometry",
        "run_asymmetric_dw_benchmark",
        "run_bootstrap",
        "controller",
        "placement",
        "oracle",
    ):
        assert not any(forbidden in module for module in imports), imports
