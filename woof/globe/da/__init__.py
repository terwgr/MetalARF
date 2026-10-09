"""WOOF global ensemble data assimilation (the dual-resolution LETKF).

The package the ``woof global da`` door drives:

* :mod:`.options`: :class:`EnsembleOptions` (what the ensemble is) and
  :class:`FilterOptions` (what an analysis does);
* :mod:`.observations`: :class:`PointObs`, the one observation contract
  every stream produces;
* :mod:`.ensemble`: :class:`GlobalEnsemble`, N resident members sharing
  one model on one card, their checkpoints and their manifest;
* :mod:`.analysis`: :func:`analyze_ensemble` (one LETKF analysis and its
  receipt; with a :class:`ControlBackground` the control's own analysis
  from the high-resolution innovation through the ensemble covariance,
  amendment A), :func:`apply_control_increment` (the tapered spectral
  transfer, amendment C), :func:`recenter`, and
  :func:`apply_mean_increment` (the mean-increment transfer, kept as the
  OSSE's comparison experiment);
* :mod:`.window`: :class:`ObservationWindow`, the observation-space
  trajectories that compare a report with the state at ITS time
  (amendment B);
* :mod:`.letkf_point`: the point-observation LETKF on the sphere, the
  arithmetic of :mod:`woof.da.letkf` under a ragged neighbour list;
* :mod:`.operators`: member-batched point operators for the neutral
  variable vocabulary;
* :mod:`.perturbations`: the spectral perturbation family (initial
  ensemble, additive inflation), linearly balanced by default;
* :mod:`.observation_errors`: the Desroziers calibration of the
  observation errors laid over the tables' assigned ones;
* :mod:`.localisation`: the vertical-correlation instrument the
  per-class localisation cutoffs are derived from;
* :mod:`.osse`: the global observing-system simulation harness and the
  calibration families.

Interface notes and every decision taken while building it:
``docs/arwen-global-ensemble-da.md``.
"""
from .analysis import (
    ANALYSIS_SCHEMA,
    REGIONS,
    ControlBackground,
    EnsembleAnalysis,
    analyze_ensemble,
    apply_control_increment,
    apply_mean_increment,
    recenter,
    taper_weights,
)
from .ensemble import (
    ENSEMBLE_MANIFEST_NAME,
    ENSEMBLE_SCHEMA,
    GlobalEnsemble,
    MemberTargets,
    StepTiming,
    embed_spectral,
    ensemble_config,
    member_checkpoint_path,
    truncate_spectral,
)
from .observations import PointObs, concatenate
from .observation_errors import DESROZIERS_ERROR_SCALE_TABLE, DESROZIERS_ERROR_TABLE, calibrated_error
from .options import (
    ANALYSIS_FIELDS,
    DEFAULT_ANALYSIS_FIELDS,
    INCREMENT_APPLICATION_MODES,
    OBSERVATION_ERROR_CALIBRATIONS,
    PERTURBATION_BALANCE_MODES,
    WIND_BALANCE_MODES,
    EnsembleOptions,
    FilterOptions,
)
from .window import ObservationWindow

__all__ = [
    "ANALYSIS_FIELDS",
    "ANALYSIS_SCHEMA",
    "DEFAULT_ANALYSIS_FIELDS",
    "ControlBackground",
    "DESROZIERS_ERROR_SCALE_TABLE",
    "DESROZIERS_ERROR_TABLE",
    "ENSEMBLE_MANIFEST_NAME",
    "ENSEMBLE_SCHEMA",
    "REGIONS",
    "WIND_BALANCE_MODES",
    "EnsembleAnalysis",
    "EnsembleOptions",
    "FilterOptions",
    "GlobalEnsemble",
    "INCREMENT_APPLICATION_MODES",
    "OBSERVATION_ERROR_CALIBRATIONS",
    "PERTURBATION_BALANCE_MODES",
    "MemberTargets",
    "ObservationWindow",
    "PointObs",
    "StepTiming",
    "analyze_ensemble",
    "apply_control_increment",
    "apply_mean_increment",
    "calibrated_error",
    "concatenate",
    "embed_spectral",
    "ensemble_config",
    "member_checkpoint_path",
    "recenter",
    "taper_weights",
    "truncate_spectral",
]
