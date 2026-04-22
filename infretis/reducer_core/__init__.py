"""Typed replay/derived-reduct core.

Semantically, this package is the project's `derived_reducts` layer. It owns
typed reductions over replay-semantic objects plus resolved syntax/context.
It is separate from the replay-bridge compatibility layer and from the
replay-semantic core itself.

Architectural guard: derived stays derived.
================================================

The following families of objects are **always derived reducts**, never
primitive replay ontology:

- rank / order summaries (derived from primitive clocks and weights)
- coarse `(nu, E)` objects (derived from `ObservedPath` plus
  `ReductionContext`)
- local strip-transport summaries (derived from R3/R4 tables and geometry)
- committor samples (derived; committor fields belong to the reserved
  operator-geometry layer, not to raw replay ontology)
- Green / RKHS / operator kernels and resolvent/heat objects (belong in
  `infretis.operator_geometry`, not here, and not in the raw event layer)

Do **not** "solve" a reducer gap by inflating the raw event layer. If a
derived quantity is missing, it either (a) belongs as a new reduct in this
package, strictly above replay-semantic primitives, or (b) belongs in the
reserved `infretis.operator_geometry` boundary. Primitive ontology stays
small.

Naming guard: no generic `kernel` namespace.
================================================

Do **not** introduce `kernel.py`, `kernels.py`, or a generic `kernels/`
subpackage under `reducer_core`. The project distinguishes at least four
kernel families -- path / Markov kernels, construction kernels, operator
kernels, and RKHS kernels -- and a generic name hides that split. Any
kernel-shaped object should live in a namespace that names the kernel
family explicitly (for operator-side kernels, `infretis.operator_geometry`).
"""

from infretis.syntax_and_context import (
    LEFT_INTERVAL_MASS_V1,
    VersionedTimeWeightPolicy,
)

from .r1 import (
    FourPartPredicateResult,
    evaluate_predicate,
    resolve_predicate_context,
)
from .r2b_replay_predicate_bundle import (
    REPLAY_PREDICATE_BUNDLE_V1,
    REPLAY_PREDICATE_EXACTNESS_CONTRACT_V1,
    AttemptPredicateBundle,
    ReplayPredicateBundleReduction,
    ReplayPredicateSummary,
    build_attempt_predicate_bundle,
    build_replay_predicate_bundle_reduction,
    summarize_replay_predicate_bundle,
)
from .r3_attempt import (
    ATTEMPT_TIME_OCCURRENCE_RESIDENCE_V1,
    PerOccurrenceAttemptTimeWeightTable,
    VersionedAttemptTimeEstimator,
    compute_attempt_time_weights,
)
from .r3_refresh import (
    PerOccurrenceRefreshWeightTable,
    REFRESH_OCCURRENCE_FREQUENCY_V1,
    VersionedRefreshEstimator,
    compute_refresh_weights,
)
from .r4_nu_e import (
    SCALAR_INTERVAL_BINS_V1,
    OccupancyMeasure,
    ScalarIntervalBinDiscretization,
    TransitionCounts,
    build_scalar_interval_partition,
    compute_nu_E,
    compute_nu_E_from_view,
    observed_path_occupation_transition_policy,
)
from .r4b_family_operator_profiles import (
    FAMILY_OPERATOR_PROFILE_V1,
    FAMILY_SELECTION_ALL_ACCEPTED_V1,
    FAMILY_SELECTION_BROAD_ENSEMBLE_V1,
    FAMILY_SELECTION_EXACT_RANK_FIBER_V1,
    FamilyOperatorProfile,
    FamilySelection,
    FamilySelectionDescription,
    FamilySelectionMode,
    PartitionIdentity,
    SelectedObservedPathFamily,
    compute_family_operator_profile_from_view,
    compute_family_operator_profile,
    select_observed_path_family,
)
from .view_context import (
    OBSERVED_PATH_SEMANTICS_V1,
    STANDARD_REDUCTION_VIEW_V1,
    ObservationSemantics,
    PrimitiveFamily,
    ViewCtx,
    ViewReductionKind,
    ViewSelector,
    ViewSelectorKind,
    ViewSummaryKind,
)
from .r5_laws import (
    CONSTRUCTION_EMPIRICAL_LAW_ESTIMATOR_V1,
    CONSTRUCTION_EMPIRICAL_LAW_V1,
    ATTEMPT_EMPIRICAL_LAW_V1,
    REFRESH_EMPIRICAL_LAW_V1,
    AttemptEmpiricalLaw,
    ConstructionEmpiricalLaw,
    RefreshEmpiricalLaw,
    VersionedConstructionEstimator,
    WeightedOccurrenceMeasureEntry,
    WeightedConstructionMeasureEntry,
    build_attempt_law,
    build_construction_law,
    build_refresh_law,
)
from .r6_persistence import (
    PERSISTENCE_DIAGNOSTIC_V1,
    PERSISTENCE_RATIO_BY_ENSEMBLE_V1,
    PersistenceDiagnostic,
    VersionedPersistenceDiagnostic,
    compute_persistence_diagnostic,
)
from .r7_controller_inputs import (
    PLACEMENT_CONTROLLER_INPUT_V1,
    ControllerGeometryReduction,
    ControllerPlacementSignalReduction,
    ControllerReplaySummaryReduction,
    PlacementControllerReduction,
    build_placement_controller_input,
)
from .r7b_controller_signals import (
    PLACEMENT_CONTROLLER_SIGNALS_V1,
    ControllerObservedPathCoverage,
    ControllerSignalReduction,
    build_placement_controller_signals,
)
from .r7c_wham_path_weights import (
    PATH_OBSERVATION_SOURCE_COMPATIBILITY,
    PATH_OBSERVATION_SOURCE_OBSERVED_PATH,
    PATH_OBSERVATION_SOURCE_UNAVAILABLE,
    WHAM_PATH_WEIGHT_INPUTS_V1,
    WhamPathWeightRecord,
    WhamPathWeightReduction,
    build_wham_path_weight_inputs,
)

__all__ = [
    "FourPartPredicateResult",
    "AttemptPredicateBundle",
    "ReplayPredicateBundleReduction",
    "ReplayPredicateSummary",
    "REPLAY_PREDICATE_BUNDLE_V1",
    "REPLAY_PREDICATE_EXACTNESS_CONTRACT_V1",
    "build_attempt_predicate_bundle",
    "build_replay_predicate_bundle_reduction",
    "summarize_replay_predicate_bundle",
    "PerOccurrenceAttemptTimeWeightTable",
    "VersionedAttemptTimeEstimator",
    "ATTEMPT_TIME_OCCURRENCE_RESIDENCE_V1",
    "compute_attempt_time_weights",
    "PerOccurrenceRefreshWeightTable",
    "VersionedRefreshEstimator",
    "REFRESH_OCCURRENCE_FREQUENCY_V1",
    "compute_refresh_weights",
    "OccupancyMeasure",
    "TransitionCounts",
    "ScalarIntervalBinDiscretization",
    "VersionedTimeWeightPolicy",
    "SCALAR_INTERVAL_BINS_V1",
    "LEFT_INTERVAL_MASS_V1",
    "compute_nu_E",
    "compute_nu_E_from_view",
    "build_scalar_interval_partition",
    "observed_path_occupation_transition_policy",
    "OBSERVED_PATH_SEMANTICS_V1",
    "STANDARD_REDUCTION_VIEW_V1",
    "ObservationSemantics",
    "PrimitiveFamily",
    "ViewCtx",
    "ViewReductionKind",
    "ViewSelector",
    "ViewSelectorKind",
    "ViewSummaryKind",
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
    "WeightedOccurrenceMeasureEntry",
    "WeightedConstructionMeasureEntry",
    "AttemptEmpiricalLaw",
    "RefreshEmpiricalLaw",
    "ConstructionEmpiricalLaw",
    "VersionedConstructionEstimator",
    "ATTEMPT_EMPIRICAL_LAW_V1",
    "REFRESH_EMPIRICAL_LAW_V1",
    "CONSTRUCTION_EMPIRICAL_LAW_V1",
    "CONSTRUCTION_EMPIRICAL_LAW_ESTIMATOR_V1",
    "build_attempt_law",
    "build_refresh_law",
    "build_construction_law",
    "VersionedPersistenceDiagnostic",
    "PersistenceDiagnostic",
    "PERSISTENCE_DIAGNOSTIC_V1",
    "PERSISTENCE_RATIO_BY_ENSEMBLE_V1",
    "compute_persistence_diagnostic",
    "ControllerGeometryReduction",
    "ControllerReplaySummaryReduction",
    "ControllerPlacementSignalReduction",
    "ControllerObservedPathCoverage",
    "ControllerSignalReduction",
    "PlacementControllerReduction",
    "PLACEMENT_CONTROLLER_INPUT_V1",
    "PLACEMENT_CONTROLLER_SIGNALS_V1",
    "WHAM_PATH_WEIGHT_INPUTS_V1",
    "WhamPathWeightRecord",
    "WhamPathWeightReduction",
    "PATH_OBSERVATION_SOURCE_OBSERVED_PATH",
    "PATH_OBSERVATION_SOURCE_COMPATIBILITY",
    "PATH_OBSERVATION_SOURCE_UNAVAILABLE",
    "build_placement_controller_input",
    "build_placement_controller_signals",
    "build_wham_path_weight_inputs",
    "evaluate_predicate",
    "resolve_predicate_context",
]
