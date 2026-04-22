"""Derived reducer R4b: family-level operator profiles.

Semantic owner for applying the landed operator-pregeometry chain to a
selected family of observed paths:

    ObservedPath family
        -> family occupation / transition measures
        -> partition pushforwards
        -> tail / upward-cut / cut-profile
        -> typed reducer output

This module owns selection, semantic readiness, and provenance only. Raw
occupation / transition, family aggregation, partition pushforwards, and
partition-level operators are delegated to ``math_core``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from infretis.math_core.predicates.ensembles import (
    EnsembleDescriptor,
    EnsembleFiberDescriptor,
    EnsemblePredicateContext,
    ensemble_fiber_membership,
    ensemble_membership,
)
from infretis.math_core.reducts import (
    occupation_transition as occupation_transition_reducts,
)
from infretis.math_core.reducts import partition_operators
from infretis.math_core.reducts.occupation_transition import (
    OccupationMeasure,
    PartitionOccupation,
    PartitionTransition,
    TransitionMeasure,
)
from infretis.math_core.reducts.partition_operators import (
    CutProfile,
    CutVector,
    TailVector,
)
from infretis.math_core.weighted_family import (
    WeightedObservedPath,
    WeightedPathFamily,
)
from infretis.replay_semantic_core import ObservedPath, build_observed_path
from infretis.syntax_and_context import LEFT_INTERVAL_MASS_V1, ReductionContext

from .r4_nu_e import (
    SCALAR_INTERVAL_BINS_V1,
    ScalarIntervalBinDiscretization,
    build_scalar_interval_partition,
    observed_path_occupation_transition_policy,
)
from .view_context import (
    ViewCtx,
    ViewReductionKind,
    ViewSelector,
    ViewSelectorKind,
    ViewSummaryKind,
)


FAMILY_OPERATOR_PROFILE_V1 = "family_operator_profile_v1"
FAMILY_SELECTION_ALL_ACCEPTED_V1 = "all_accepted_paths_v1"
FAMILY_SELECTION_BROAD_ENSEMBLE_V1 = "broad_ensemble_membership_v1"
FAMILY_SELECTION_EXACT_RANK_FIBER_V1 = "exact_rank_fiber_v1"


class FamilySelectionMode(str, Enum):
    """Supported semantic family-selection modes."""

    ALL_ACCEPTED = FAMILY_SELECTION_ALL_ACCEPTED_V1
    BROAD_ENSEMBLE = FAMILY_SELECTION_BROAD_ENSEMBLE_V1
    EXACT_RANK_FIBER = FAMILY_SELECTION_EXACT_RANK_FIBER_V1


@dataclass(frozen=True)
class FamilySelection:
    """Typed configuration for reducer-owned family selection.

    Ensemble and fiber modes delegate structural membership to
    ``math_core.predicates.ensembles``. The acceptance guard is a
    reducer-input readiness check over already-selected path candidates,
    not an ensemble semantic substitute.
    """

    mode: FamilySelectionMode
    cv_key: str = "order_parameter"
    ensemble_descriptor: EnsembleDescriptor | None = None
    fiber_descriptor: EnsembleFiberDescriptor | None = None
    ensemble_context: EnsemblePredicateContext | None = None
    require_accepted: bool = True
    selection_version: str = "family_selection_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, FamilySelectionMode):
            object.__setattr__(
                self,
                "mode",
                FamilySelectionMode(str(self.mode)),
            )

    @classmethod
    def all_accepted(
        cls,
        *,
        cv_key: str = "order_parameter",
    ) -> "FamilySelection":
        return cls(mode=FamilySelectionMode.ALL_ACCEPTED, cv_key=cv_key)

    @classmethod
    def broad_ensemble(
        cls,
        descriptor: EnsembleDescriptor,
        *,
        context: EnsemblePredicateContext,
        cv_key: str = "order_parameter",
    ) -> "FamilySelection":
        return cls(
            mode=FamilySelectionMode.BROAD_ENSEMBLE,
            cv_key=cv_key,
            ensemble_descriptor=descriptor,
            ensemble_context=context,
        )

    @classmethod
    def exact_rank_fiber(
        cls,
        descriptor: EnsembleFiberDescriptor,
        *,
        context: EnsemblePredicateContext,
        cv_key: str = "order_parameter",
    ) -> "FamilySelection":
        return cls(
            mode=FamilySelectionMode.EXACT_RANK_FIBER,
            cv_key=cv_key,
            fiber_descriptor=descriptor,
            ensemble_context=context,
        )


@dataclass(frozen=True)
class FamilySelectionDescription:
    """Stable description of the selected path family."""

    mode: str
    selection_version: str
    cv_key: str
    input_path_count: int
    selected_path_count: int
    selected_path_ids: tuple[str, ...]
    selected_total_weight: float
    selector: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectedObservedPathFamily:
    """Typed selected-family object consumed by the operator reducer."""

    family: WeightedPathFamily[ObservedPath]
    selection: FamilySelectionDescription
    exactness_status: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class PartitionIdentity:
    """Reducer-facing identity for the math-core ordered partition."""

    name: str
    version: str
    cells: tuple[int, ...]
    edges: tuple[float, ...]
    boundary: str
    discretization_version: str
    cv_key: str


@dataclass(frozen=True)
class FamilyOperatorProfile:
    """Typed family-level operator-profile reduction."""

    selected_family: FamilySelectionDescription
    partition: PartitionIdentity
    raw_occupation: OccupationMeasure[float]
    raw_transition: TransitionMeasure[float]
    occupation: PartitionOccupation[int]
    transition: PartitionTransition[int]
    tail: TailVector[int]
    upward_cut: CutVector[int]
    cut_profile: CutProfile[int]
    exactness_status: dict[str, Any]
    provenance: dict[str, Any]
    reduction_name: str = "family_operator_profile"
    reduction_version: str = FAMILY_OPERATOR_PROFILE_V1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PathFamilyInput = Iterable[
    ObservedPath
    | dict[str, Any]
    | WeightedObservedPath[ObservedPath | dict[str, Any]]
]

WeightedFamilyInput = (
    ObservedPath
    | dict[str, Any]
    | WeightedObservedPath[ObservedPath | dict[str, Any]]
)


def _metadata_from_pathblob(pathblob: Mapping[str, Any]) -> dict[str, Any]:
    payload = pathblob.get("payload", pathblob)
    metadata: dict[str, Any] = {}
    if isinstance(payload, Mapping):
        for key in (
            "accepted",
            "status",
            "path_number_new",
            "path_number_old",
            "ens_num",
        ):
            if key in payload:
                metadata[key] = payload[key]
    if "object_id" in pathblob:
        metadata["pathblob_id"] = pathblob["object_id"]
    return metadata


def _coerce_weighted_member(
    item: WeightedFamilyInput,
    reduction_context: ReductionContext,
) -> tuple[WeightedObservedPath[ObservedPath], bool]:
    compatibility_path = False
    if isinstance(item, WeightedObservedPath):
        metadata = dict(item.metadata)
        if isinstance(item.observed_path, ObservedPath):
            observed = item.observed_path
        elif isinstance(item.observed_path, dict):
            compatibility_path = True
            observed = build_observed_path(
                item.observed_path,
                reduction_context,
            )
            metadata = {
                **_metadata_from_pathblob(item.observed_path),
                **metadata,
            }
        else:
            raise TypeError(
                "WeightedObservedPath.observed_path must be ObservedPath "
                "or PathBlob dict"
            )
        return (
            WeightedObservedPath(
                observed_path=observed,
                weight=float(item.weight),
                metadata=metadata,
            ),
            compatibility_path,
        )

    if isinstance(item, ObservedPath):
        return WeightedObservedPath(observed_path=item, weight=1.0), False
    if isinstance(item, dict):
        compatibility_path = True
        observed = build_observed_path(item, reduction_context)
        return (
            WeightedObservedPath(
                observed_path=observed,
                weight=1.0,
                metadata=_metadata_from_pathblob(item),
            ),
            compatibility_path,
        )
    raise TypeError(
        "family inputs must be ObservedPath, PathBlob dict, "
        "or WeightedObservedPath"
    )


def _accepted_marker(
    member: WeightedObservedPath[ObservedPath],
) -> bool | None:
    for source in (member.metadata, member.observed_path.source_payload):
        if not isinstance(source, Mapping):
            continue
        if isinstance(source.get("accepted"), bool):
            return bool(source["accepted"])
        status = source.get("status")
        if status is not None:
            normalized = str(status).upper()
            if normalized in {"ACC", "ACCEPTED", "OK"}:
                return True
            if normalized in {"REJ", "REJECTED", "KOB", "FAIL", "FAILED"}:
                return False
    return None


def _selector_description(selection: FamilySelection) -> dict[str, Any]:
    if selection.mode is FamilySelectionMode.ALL_ACCEPTED:
        return {"kind": selection.mode.value}
    if selection.mode is FamilySelectionMode.BROAD_ENSEMBLE:
        if selection.ensemble_descriptor is None:
            raise ValueError(
                "broad ensemble selection requires ensemble_descriptor"
            )
        if selection.ensemble_context is None:
            raise ValueError(
                "broad ensemble selection requires ensemble_context"
            )
        return {
            "kind": selection.mode.value,
            "threshold_index": selection.ensemble_descriptor.threshold_index,
            "terminal_sector": (
                selection.ensemble_descriptor.terminal_sector.value
            ),
        }
    if selection.mode is FamilySelectionMode.EXACT_RANK_FIBER:
        if selection.fiber_descriptor is None:
            raise ValueError(
                "exact-rank fiber selection requires fiber_descriptor"
            )
        if selection.ensemble_context is None:
            raise ValueError(
                "exact-rank fiber selection requires ensemble_context"
            )
        return {
            "kind": selection.mode.value,
            "rho_index": selection.fiber_descriptor.rho_index,
            "terminal_sector": (
                selection.fiber_descriptor.terminal_sector.value
            ),
        }
    raise ValueError(f"unsupported family selection mode: {selection.mode!r}")


def _passes_structural_selection(
    observed_path: ObservedPath,
    selection: FamilySelection,
) -> bool:
    if selection.mode is FamilySelectionMode.ALL_ACCEPTED:
        return True
    op_sequence = observed_path.cv_values(selection.cv_key)
    if selection.mode is FamilySelectionMode.BROAD_ENSEMBLE:
        if (
            selection.ensemble_descriptor is None
            or selection.ensemble_context is None
        ):
            raise ValueError(
                "broad ensemble selection is missing descriptor/context"
            )
        return ensemble_membership(
            op_sequence,
            selection.ensemble_descriptor,
            context=selection.ensemble_context,
        )
    if selection.mode is FamilySelectionMode.EXACT_RANK_FIBER:
        if (
            selection.fiber_descriptor is None
            or selection.ensemble_context is None
        ):
            raise ValueError(
                "exact-rank fiber selection is missing descriptor/context"
            )
        return ensemble_fiber_membership(
            op_sequence,
            selection.fiber_descriptor,
            context=selection.ensemble_context,
        )
    raise ValueError(f"unsupported family selection mode: {selection.mode!r}")


def _selection_from_view_selector(
    selector: ViewSelector,
    view_ctx: ViewCtx,
) -> FamilySelection:
    view_ctx.validate_selector(selector)
    predicate_context = (
        selector.predicate_context or view_ctx.predicate_context
    )
    if selector.selector_kind is ViewSelectorKind.ALL_ACCEPTED:
        return FamilySelection.all_accepted(cv_key=selector.cv_key)
    if selector.selector_kind is ViewSelectorKind.BROAD_ENSEMBLE:
        if selector.ensemble_descriptor is None:
            raise ValueError("broad ensemble selector requires descriptor")
        if predicate_context is None:
            raise ValueError("broad ensemble selector requires context")
        return FamilySelection.broad_ensemble(
            selector.ensemble_descriptor,
            context=predicate_context,
            cv_key=selector.cv_key,
        )
    if selector.selector_kind is ViewSelectorKind.EXACT_RANK_FIBER:
        if selector.fiber_descriptor is None:
            raise ValueError("exact-rank fiber selector requires descriptor")
        if predicate_context is None:
            raise ValueError("exact-rank fiber selector requires context")
        return FamilySelection.exact_rank_fiber(
            selector.fiber_descriptor,
            context=predicate_context,
            cv_key=selector.cv_key,
        )
    raise ValueError(f"unsupported selector kind: {selector.selector_kind!r}")


def _validate_selection_against_view(
    selection: FamilySelection,
    view_ctx: ViewCtx,
) -> None:
    view_ctx.validate_selector_kind(selection.mode.value)
    if selection.mode is FamilySelectionMode.ALL_ACCEPTED:
        return
    if view_ctx.predicate_context is None:
        raise ValueError(
            f"selector {selection.mode.value!r} requires ViewCtx context"
        )
    if selection.ensemble_context is not view_ctx.predicate_context:
        raise ValueError("FamilySelection context is outside this ViewCtx")


def _resolve_selection_context(
    selection: FamilySelection | ViewSelector,
    reduction_context_or_view: ReductionContext | ViewCtx,
) -> tuple[FamilySelection, ReductionContext, ViewCtx | None]:
    view_ctx: ViewCtx | None = None
    if isinstance(reduction_context_or_view, ViewCtx):
        view_ctx = reduction_context_or_view
        context = view_ctx.reduction_context
        if isinstance(selection, ViewSelector):
            resolved_selection = _selection_from_view_selector(
                selection,
                view_ctx,
            )
        else:
            resolved_selection = selection
            _validate_selection_against_view(resolved_selection, view_ctx)
        return resolved_selection, context, view_ctx
    if isinstance(selection, ViewSelector):
        raise TypeError("ViewSelector requires a ViewCtx")
    return selection, reduction_context_or_view, None


def select_observed_path_family(
    paths: PathFamilyInput,
    selection: FamilySelection | ViewSelector,
    reduction_context: ReductionContext | ViewCtx,
) -> SelectedObservedPathFamily:
    """Select a weighted observed-path family under typed selection."""
    selection, reduction_context, view_ctx = _resolve_selection_context(
        selection,
        reduction_context,
    )
    if (
        reduction_context.time_weight_policy.version
        != LEFT_INTERVAL_MASS_V1.version
    ):
        raise ValueError(
            "family operator profiles currently support only "
            f"{LEFT_INTERVAL_MASS_V1.version}; got "
            f"{reduction_context.time_weight_policy.version}"
        )

    members: list[WeightedObservedPath[ObservedPath]] = []
    selected: list[WeightedObservedPath[ObservedPath]] = []
    compatibility_count = 0
    missing_accepted_status_count = 0
    explicit_rejected_count = 0

    for item in paths:
        member, used_compat = _coerce_weighted_member(item, reduction_context)
        members.append(member)
        if used_compat:
            compatibility_count += 1
        accepted = _accepted_marker(member)
        if accepted is None:
            missing_accepted_status_count += 1
        if selection.require_accepted and accepted is False:
            explicit_rejected_count += 1
            continue
        if not _passes_structural_selection(member.observed_path, selection):
            continue
        selected.append(member)

    family = WeightedPathFamily.from_iterable(selected)
    selected_path_ids = tuple(
        member.observed_path.pathblob_id for member in selected
    )
    description = FamilySelectionDescription(
        mode=selection.mode.value,
        selection_version=selection.selection_version,
        cv_key=selection.cv_key,
        input_path_count=len(members),
        selected_path_count=len(selected),
        selected_path_ids=selected_path_ids,
        selected_total_weight=family.total_weight(),
        selector=_selector_description(selection),
    )
    exactness_status = {
        "selection_mode": selection.mode.value,
        "input_path_count": len(members),
        "selected_path_count": len(selected),
        "explicit_rejected_path_count": explicit_rejected_count,
        "missing_accepted_status_count": missing_accepted_status_count,
        "compatibility_path_count": compatibility_count,
        "observed_path_count": len(members) - compatibility_count,
        "requires_accepted_paths": selection.require_accepted,
        "structural_screen_source": (
            "none"
            if selection.mode is FamilySelectionMode.ALL_ACCEPTED
            else "infretis.math_core.predicates.ensembles"
        ),
    }
    if view_ctx is not None:
        exactness_status["view_id"] = view_ctx.view_id
    provenance = {
        "selection_version": selection.selection_version,
        "selection_mode": selection.mode.value,
        "cv_key": selection.cv_key,
        "reduction_context_version": reduction_context.context_version,
        "time_weight_policy_version": (
            reduction_context.time_weight_policy.version
        ),
    }
    if view_ctx is not None:
        provenance["view_ctx"] = view_ctx.to_contract_dict()
    return SelectedObservedPathFamily(
        family=family,
        selection=description,
        exactness_status=exactness_status,
        provenance=provenance,
    )


def _partition_identity(
    discretization: ScalarIntervalBinDiscretization,
) -> PartitionIdentity:
    partition = build_scalar_interval_partition(discretization)
    return PartitionIdentity(
        name=partition.name,
        version=partition.version,
        cells=tuple(int(cell) for cell in partition.cells),
        edges=tuple(float(edge) for edge in partition.edges),
        boundary=partition.boundary,
        discretization_version=discretization.version,
        cv_key=discretization.cv_key,
    )


def compute_family_operator_profile(
    paths: PathFamilyInput,
    selection: FamilySelection | ViewSelector,
    discretization: ScalarIntervalBinDiscretization,
    reduction_context: ReductionContext | ViewCtx,
) -> FamilyOperatorProfile:
    """Apply the family ``(nu,E) -> tail/cut/profile`` operator chain."""
    view_ctx: ViewCtx | None = None
    if isinstance(reduction_context, ViewCtx):
        view_ctx = reduction_context
        view_ctx.validate_reduction(
            ViewReductionKind.R4B_FAMILY_OPERATOR_PROFILE
        )
        view_ctx.validate_summary(ViewSummaryKind.SCALAR_INTERVAL_BINS)
        view_ctx.validate_summary(ViewSummaryKind.FAMILY_PARTITIONED_NU_E)
        view_ctx.validate_summary(ViewSummaryKind.TAIL_OPERATOR)
        view_ctx.validate_summary(ViewSummaryKind.UPWARD_CUT_OPERATOR)
        view_ctx.validate_summary(ViewSummaryKind.CUT_PROFILE_OPERATOR)
    if discretization.version != SCALAR_INTERVAL_BINS_V1:
        raise ValueError(
            f"unsupported discretization version: {discretization.version}"
        )
    selected = select_observed_path_family(paths, selection, reduction_context)
    partition = build_scalar_interval_partition(discretization)
    policy = observed_path_occupation_transition_policy(discretization.cv_key)

    raw_occupation = occupation_transition_reducts.family_occupation_measure(
        selected.family,
        policy,
    )
    raw_transition = occupation_transition_reducts.family_transition_measure(
        selected.family,
        policy,
    )
    occupation, transition = (
        occupation_transition_reducts.family_partitioned_occupation_transition(
            selected.family,
            policy,
            partition,
        )
    )
    tail = partition_operators.tail_operator(occupation)
    upward_cut = partition_operators.upward_cut_operator(transition)
    cut_profile = partition_operators.cut_profile_operator(
        occupation,
        transition,
    )

    exactness_status = {
        **selected.exactness_status,
        "reducer_version": FAMILY_OPERATOR_PROFILE_V1,
        "partition_name": partition.name,
        "partition_version": partition.version,
        "partition_cell_count": len(partition),
        "raw_occupation_total_mass": raw_occupation.total_mass(),
        "raw_transition_total_mass": raw_transition.total_mass(),
        "partition_occupation_total_mass": occupation.total_mass(),
        "partition_transition_total_mass": transition.total_mass(),
        "math_core_family_aggregation": "delegated",
        "math_core_partition_pushforward": "delegated",
        "math_core_partition_operators": "delegated",
    }
    provenance = {
        **selected.provenance,
        "reducer_name": "family_operator_profile",
        "reducer_version": FAMILY_OPERATOR_PROFILE_V1,
        "occupation_transition_module": (
            "infretis.math_core.reducts.occupation_transition"
        ),
        "partition_operator_module": (
            "infretis.math_core.reducts.partition_operators"
        ),
        "math_core_entry_points": (
            "family_occupation_measure",
            "family_transition_measure",
            "family_partitioned_occupation_transition",
            "tail_operator",
            "upward_cut_operator",
            "cut_profile_operator",
        ),
        "discretization_version": discretization.version,
        "partition_identity": asdict(_partition_identity(discretization)),
    }
    return FamilyOperatorProfile(
        selected_family=selected.selection,
        partition=_partition_identity(discretization),
        raw_occupation=raw_occupation,
        raw_transition=raw_transition,
        occupation=occupation,
        transition=transition,
        tail=tail,
        upward_cut=upward_cut,
        cut_profile=cut_profile,
        exactness_status=exactness_status,
        provenance=provenance,
    )


def compute_family_operator_profile_from_view(
    paths: PathFamilyInput,
    view_ctx: ViewCtx,
    discretization: ScalarIntervalBinDiscretization,
    selector: FamilySelection | ViewSelector | None = None,
) -> FamilyOperatorProfile:
    """ViewCtx-resolved entry point for R4b family operator profiles."""
    resolved_selector = selector or view_ctx.default_selector(
        cv_key=discretization.cv_key
    )
    return compute_family_operator_profile(
        paths,
        resolved_selector,
        discretization,
        view_ctx,
    )


__all__ = [
    "FAMILY_OPERATOR_PROFILE_V1",
    "FAMILY_SELECTION_ALL_ACCEPTED_V1",
    "FAMILY_SELECTION_BROAD_ENSEMBLE_V1",
    "FAMILY_SELECTION_EXACT_RANK_FIBER_V1",
    "FamilyOperatorProfile",
    "FamilySelection",
    "FamilySelectionDescription",
    "FamilySelectionMode",
    "PartitionIdentity",
    "SelectedObservedPathFamily",
    "compute_family_operator_profile_from_view",
    "compute_family_operator_profile",
    "select_observed_path_family",
]
