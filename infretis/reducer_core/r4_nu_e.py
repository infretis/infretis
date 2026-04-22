"""Derived reduct R4: per-observed-path discretization summaries.

Reducer-core wrapper. This module owns only the semantic / provenance
half of R4:

1. check semantic readiness and exactness (``ObservedPath`` carries
   exact cadence; the active time-weight policy is supported);
2. resolve the discretization / time-weight policy from context;
3. adapt the semantic ``ObservedPath`` into a math-core
   :class:`OccupationTransitionPolicy` plus an :class:`OrderedPartition`;
4. delegate *all* mass, cell-index, and partition-pushforward
   mathematics to
   :mod:`infretis.math_core.reducts.occupation_transition`;
5. package the result in the cell-ordinal shape consumed by downstream
   reducer-core callers.

No private mass, cell-index, tail, or cut mathematics remains in this
module; partition-level operators (tail / upward-cut / cut-profile) live
in :mod:`infretis.math_core.reducts.partition_operators` and are reached
by downstream consumers -- not by ``compute_nu_E`` itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from infretis.math_core.reducts.occupation_transition import (
    OccupationTransitionPolicy,
    OrderedPartition,
    PartitionOccupation,
    PartitionTransition,
    interval_left_mass_convention,
    pathwise_occupation_measure,
    pathwise_transition_measure,
    pushforward_occupation_to_partition,
    pushforward_transition_to_partition,
)
from infretis.replay_semantic_core import ObservedPath, build_observed_path
from infretis.syntax_and_context import (
    LEFT_INTERVAL_MASS_V1,
    ReductionContext,
    VersionedTimeWeightPolicy,
    make_observation_only_reduction_context,
)

from .view_context import ViewCtx, ViewReductionKind


OccupancyMeasure = dict[int, float]
TransitionCounts = dict[tuple[int, int], float]


@dataclass(frozen=True)
class ScalarIntervalBinDiscretization:
    """First concrete 1D discretization for reducer R4."""

    version: str
    cv_key: str
    cell_edges: tuple[float, ...]


SCALAR_INTERVAL_BINS_V1 = "scalar_interval_bins_v1"


def build_scalar_interval_partition(
    discretization: ScalarIntervalBinDiscretization,
) -> OrderedPartition[int]:
    """Build the math-core :class:`OrderedPartition` for this discretization.

    Validates the discretization version tag (semantic readiness) and
    delegates the structural invariants -- strictly increasing edges,
    boundary convention, deterministic cell assignment, hard failure on
    out-of-support values -- to :class:`OrderedPartition`.
    """
    if discretization.version != SCALAR_INTERVAL_BINS_V1:
        raise ValueError(
            f"unsupported discretization version: {discretization.version}"
        )
    edges = tuple(float(value) for value in discretization.cell_edges)
    if len(edges) < 2:
        raise ValueError(
            "scalar_interval_bins_v1 requires at least two cell edges"
        )
    n_cells = len(edges) - 1
    return OrderedPartition(
        cells=tuple(range(n_cells)),
        edges=edges,
        name=discretization.version,
        version="v1",
    )


def observed_path_occupation_transition_policy(
    cv_key: str,
) -> OccupationTransitionPolicy[ObservedPath, float]:
    """Adapt a semantic ``ObservedPath`` into a math-core policy.

    Pure adapter: no cadence semantics invented here; the cadence-exact
    ``step_intervals`` are read off the already-resolved
    ``ObservedPath``. The left-interval mass convention is the math-core
    mirror of :data:`LEFT_INTERVAL_MASS_V1` on the context side.
    """
    return OccupationTransitionPolicy(
        convention=interval_left_mass_convention(),
        slices=lambda op: list(op.cv_values(cv_key)),
        intervals=lambda op: list(op.step_intervals),
    )


def _coerce_observed_path(
    path_or_observed_path: ObservedPath | dict,
    context_or_policy: ReductionContext | VersionedTimeWeightPolicy | ViewCtx,
) -> tuple[ObservedPath, ReductionContext]:
    if isinstance(context_or_policy, ViewCtx):
        context_or_policy.validate_reduction(ViewReductionKind.R4_NU_E)
        context = context_or_policy.reduction_context
    elif isinstance(context_or_policy, ReductionContext):
        context = context_or_policy
    else:
        context = make_observation_only_reduction_context(context_or_policy)
    if isinstance(path_or_observed_path, ObservedPath):
        return path_or_observed_path, context
    return build_observed_path(path_or_observed_path, context), context


def _sparse_from_dense_occupation(
    u: PartitionOccupation[int],
) -> OccupancyMeasure:
    return {
        k: float(mass)
        for k, mass in enumerate(u.dense)
        if mass != 0.0
    }


def _sparse_from_dense_transition(
    F: PartitionTransition[int],
) -> TransitionCounts:
    out: TransitionCounts = {}
    for i, row in enumerate(F.dense):
        for j, mass in enumerate(row):
            if mass != 0.0:
                out[(i, j)] = float(mass)
    return out


def compute_nu_E(
    path_or_observed_path: ObservedPath | dict,
    discretization: ScalarIntervalBinDiscretization,
    reduction_context_or_policy: (
        ReductionContext | VersionedTimeWeightPolicy | ViewCtx
    ),
) -> tuple[OccupancyMeasure, TransitionCounts]:
    """Compute the frozen R4 per-path ``(nu_X, E_X)`` objects.

    Thin semantic wrapper over the math-core authoritative implementation.
    The mathematics -- raw occupation / transition measure construction
    and partition pushforward -- lives in
    :mod:`infretis.math_core.reducts.occupation_transition`. This
    function only checks readiness, resolves policy, adapts the
    semantic input, delegates, and packages the result in the legacy
    cell-ordinal sparse shape.
    """
    observed_path, reduction_context = _coerce_observed_path(
        path_or_observed_path,
        reduction_context_or_policy,
    )
    time_weight_policy = reduction_context.time_weight_policy
    if time_weight_policy.version != LEFT_INTERVAL_MASS_V1.version:
        raise ValueError(
            "unsupported time-weight policy version: "
            f"{time_weight_policy.version}"
        )

    partition = build_scalar_interval_partition(discretization)
    policy = observed_path_occupation_transition_policy(discretization.cv_key)

    raw_u = pathwise_occupation_measure(observed_path, policy)
    raw_F = pathwise_transition_measure(observed_path, policy)
    u_pushed = pushforward_occupation_to_partition(raw_u, partition)
    F_pushed = pushforward_transition_to_partition(raw_F, partition)

    nu = dict(sorted(_sparse_from_dense_occupation(u_pushed).items()))
    transitions = dict(
        sorted(_sparse_from_dense_transition(F_pushed).items())
    )
    return nu, transitions


def compute_nu_E_from_view(
    path_or_observed_path: ObservedPath | dict,
    discretization: ScalarIntervalBinDiscretization,
    view_ctx: ViewCtx,
) -> tuple[OccupancyMeasure, TransitionCounts]:
    """ViewCtx-resolved entry point for the R4 per-path ``(nu,E)`` reduct."""
    return compute_nu_E(path_or_observed_path, discretization, view_ctx)
