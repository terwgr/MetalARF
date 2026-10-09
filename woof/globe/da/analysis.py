"""One ensemble analysis of WOOF global: the function the door calls.

The sequence one call of :func:`analyze_ensemble` runs, in order:

1. the operators: every batch whose ``simulated`` is None and whose
   variable is in the neutral vocabulary is evaluated on the background
   members by :class:`~woof.globe.da.operators.MemberOperators`
   (a foreign variable without ``simulated`` is refused by name); with a
   :class:`ControlBackground` given, every batch's ``control_simulated``
   (H of the high-resolution control background) is evaluated the same
   way, or must arrive filled;
2. the observation-error calibration (``FilterOptions.
   observation_error_calibration``: the Desroziers table of
   :mod:`woof.globe.da.observation_errors` laid over the rows'
   assigned errors, both recorded per stream), then quality control of
   the offered batches against the background
   ensemble: the age window and gross bounds of the v1 door, the chain
   refusal (a report the ensemble's assimilation chain already holds),
   thinning to one report per ensemble grid cell per stream and variable,
   the background check against ``sqrt(error^2 + spread_H^2)``; every
   rejection counted by name per stream;
3. the cross-validation split: a seeded fraction of each (stream,
   variable)'s rows is withheld and judged, never analysed; since the
   scorecard rule was replaced (amendment G) this is a DIAGNOSTIC of the
   receipt, not the gate of record;
4. the LETKF (:mod:`woof.globe.da.letkf_point`) on the members'
   grid fields (``FilterOptions.analysis_fields``): Gaspari-Cohn on the
   sphere in kilometres and in ln p, RTPS, Hunt's rho; the increment of
   every member on the ensemble's Gaussian grid AND, with a control
   background, the control increment ``X_L Pa~ C d_H`` from the
   high-resolution innovation ``d_H = y - H(x_H^b)`` (amendment A: the
   control gets its own analysis through the ensemble covariance; the
   ensemble-mean increment never touches it);
5. the increments into the spectral state of each member: theta, qv and
   ln ps through the transform's forward analysis (the triangular
   truncation is the increment smoothing), the wind through the vector
   analysis with the balance rule, the global-mean surface pressure kept,
   the vapor's positivity repaired by the model's own repair, the state
   enforced; under ``increment_application = "iau"`` the members' spectral
   increments are stored on the ensemble and the integration adds an
   equal portion before each step of the next window instead;
6. the control increment into the control state
   (:func:`apply_control_increment`): the grid increment analysed into
   the ensemble triangle with the same balance rule, the transfer taper
   applied by degree (amendment C: the coarse ensemble corrects the
   scales it demonstrably represents and no others; the increment spectrum
   is recorded by band before and after the taper), embedded in the
   control's triangle, added, the global-mean surface pressure kept, the
   vapor repaired, the state enforced;
7. O-A on the assimilated and the withheld rows through the same
   operators on the analysed members (and on the control analysis); the
   Desroziers diagnostics per (stream, variable) with their assumptions
   stated; the four assessments of amendment G: engineering validity (the
   only hard gate), statistical consistency, physical consistency,
   predictive value (deferred to the forecast scorecard by name);
8. additive inflation (a fresh draw of the initial perturbation family at
   ``EnsembleOptions.additive_inflation_fraction`` of its amplitude, 0 by
   default), after the O-A pass so the receipt judges the analysis itself;
9. the receipt: O-B and O-A distributions per stream, per variable, per
   region (global, NH extratropics, tropics, SH extratropics, CONUS), the
   spread before and after per field, the increment summaries and spectra,
   the rejections, the localisation and inflation settings, timings.

The ensemble-MEAN increment is still returned in spectral form at the
ensemble truncation for :func:`apply_mean_increment`, the mean-increment
transfer that was the analysis path before amendment A and is now the
OSSE's comparison experiment (``family transfer``); the door applies the
control analysis :func:`analyze_ensemble` hands back and recentres the
members on it with :func:`recenter`.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..assimilate import (
    ASSIMILATION_HISTORY_KEY,
    ASSIMILATION_HISTORY_SCHEMA,
    REJECTION_BREAKAGE,
    _balanced_wind_increment,
    _chain_from_background,
)
from ..constants import (
    DRY_AIR_GAS_CONSTANT,
    EARTH_ROTATION_RATE_S,
    KAPPA,
    REFERENCE_PRESSURE_PA,
    RESEARCH_ACKNOWLEDGEMENT,
    SPECTRAL_FIELDS,
)
from ..obs_table import VARIABLE_TABLE, parse_valid_time
from ..pins import pins_hash
from ..state import ArwenGlobalState
from .ensemble import GlobalEnsemble, embed_spectral, truncate_spectral
from .letkf_point import (
    ColumnGeometry,
    PointAnalysis,
    PointLetkfConfig,
    PointLetkfDiagnostics,
    analyze_points_with_control,
    flatten_batches,
)
from .observations import LOCALISATION_AXIS_LNP, PointObs
from .observation_errors import calibrated_error
from .operators import (
    NEUTRAL_VARIABLES,
    OPERATOR_VARIABLES,
    POLE_LATITUDE_DEG,
    MemberOperators,
    evaluate_batches,
)
from .options import FilterOptions
from .perturbations import draw_perturbation, member_rng, perturbed_state

ANALYSIS_SCHEMA = "gpuwm.arwen-global-ensemble-analysis/v2"

#: The regions O-B and O-A are reported for: (lat0, lat1, lon0, lon1) in
#: degrees, longitude in [-180, 180).
REGIONS = {
    "global": (-90.0, 90.0, -180.0, 180.0),
    "nh_extratropics": (20.0, 90.0, -180.0, 180.0),
    "tropics": (-20.0, 20.0, -180.0, 180.0),
    "sh_extratropics": (-90.0, -20.0, -180.0, 180.0),
    "conus": (24.0, 50.0, -125.0, -66.0),
}

#: The spectral bands the increment spectrum is summarised in, as
#: fractions of the ENSEMBLE truncation: the largest scales, the scales
#: the ensemble resolves well, the taper band, and everything above the
#: ensemble truncation (which a control increment may not carry).
SPECTRAL_BANDS = (
    ("planetary_n_le_20", 0, 20),
    ("resolved_to_0p6T", 21, "0.6T"),
    ("taper_0p6T_to_T", "0.6T+1", "T"),
    ("above_ensemble_T", "T+1", None),
)

#: The gate of record after amendment G: engineering validity alone.  A
#: stream that could not be judged (no operator on the analysis, a
#: non-finite H) fails it; O-A above O-B on a stream does not (a
#: background of 0 with reports +1 and -3 at equal weight analyses to
#: -2/3 and moves away from the +1 report; a system that fits every
#: report is overfitting).
GATE_RULE = (
    "engineering validity is the only hard gate: ingest, quality control, "
    "operators (O-B and O-A on every accepted stream), the localised solve "
    "(finite increments), the increment application and the state "
    "enforcement all ran; statistical, physical and predictive consistency "
    "are reported as assessments beside it and never fail the analysis"
)

DESROZIERS_ASSUMPTIONS = (
    "Desroziers et al. (2005): E[d_oa d_ob^T] = R and E[d_ab d_ob^T] = HBH^T hold "
    "when the observation operator is linear about the background, observation "
    "and background errors are uncorrelated with each other, R is diagonal on the "
    "stream, the gain is the optimal one for the true covariances, and the sample "
    "is stationary across the region and window; here d_oa is formed with the "
    "nonlinear operator on the analysed members' mean (a linearised O-A is "
    "labelled as such where used), so the ratios are diagnostics of plausibility, "
    "not estimates to retune errors from until they look right"
)

#: Extra rejection names this filter adds to the v1 door's table.
ENSEMBLE_REJECTION_BREAKAGE = {
    **REJECTION_BREAKAGE,
    "thinned": (
        "a dense network's reports inside one analysis grid cell would enter "
        "the local solve as many near-identical rows whose errors the filter "
        "treats as independent, over-weighting that cell against its "
        "neighbours; one report per cell per stream and variable is kept, the "
        "one nearest the cell centre"
    ),
    "polar_wind": (
        "a wind report at the pole has no east or north component the model "
        "can be compared with (the gradient sampler refuses the exact pole); "
        "the case's tables carry the Amundsen-Scott sounding at 90 S"
    ),
    "no_finite_equivalent": (
        "the operators handed back a non-finite model equivalent for the row "
        "(a pole, a level outside the column); a NaN in the local solve would "
        "poison every column the row reaches"
    ),
    "humidity_above_floor": (
        "a sounding's humidity above humidity_pressure_floor_pa is not a "
        "measurement the analysis can use: the sonde's humidity sensor does "
        "not read the dry upper troposphere and stratosphere, and the model's "
        "dewpoint there is its vapor floor; on the case's first real window "
        "362 of 417 sounding dewpoint rows lay above 100 hPa and read 13.5 K "
        "wetter than the background at a 2.5 K assigned error (Desroziers "
        "ratio 7.1), a moisture increment nothing supports"
    ),
    "background_check": (
        "a report whose innovation exceeds background_check_sigmas standard "
        "deviations of sqrt(error^2 + spread_H^2) is either a gross error or "
        "a state the ensemble does not span; either way the local transform "
        "would fit it with a rank-deficient covariance and paint the misfit "
        "across the localisation lens"
    ),
    "foreign_variable_unsimulated": (
        "a variable outside the neutral vocabulary has no operator in this "
        "package; without simulated H(x_k) there is no innovation to analyse"
    ),
    "foreign_variable_no_control": (
        "a control analysis forms its innovation from H(x_H^b) on every row; a "
        "stream outside the neutral vocabulary must bring control_simulated or "
        "the control cannot take it"
    ),
}


@dataclass
class ControlBackground:
    """The high-resolution (deterministic) background the control analysis
    is formed for: its state, its model and transform (its own truncation),
    its config; ``operators`` is built from them when None."""

    state: ArwenGlobalState
    model: object
    transform: object
    cfg: object
    operators: MemberOperators | None = None

    def __post_init__(self) -> None:
        if self.operators is None:
            self.operators = MemberOperators.for_model(self.model, self.transform, self.cfg)

    @property
    def truncation(self) -> int:
        return int(self.transform.truncation)


@dataclass
class EnsembleAnalysis:
    """What one analysis produced (see the module doc for the report keys)."""

    ensemble: GlobalEnsemble
    mean_increment_spectral: dict[str, object]
    report: dict[str, object]
    #: The control analysis (amendment A) when a control background was
    #: given, with its tapered spectral increment at the ENSEMBLE
    #: truncation (embed it to see the control's own) and its record.
    control_analysis: ArwenGlobalState | None = None
    control_increment_spectral: dict[str, object] | None = None
    control_record: dict[str, object] | None = None
    timings_s: dict[str, float] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return str(self.report.get("status", "fail"))


class _Stopwatch:
    def __init__(self) -> None:
        self.readings: dict[str, float] = {}
        self._start = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.readings[name] = self.readings.get(name, 0.0) + (now - self._start)
        self._start = now


def _statistics(values: np.ndarray) -> dict[str, float]:
    """The distribution of a departure sample: count, mean (the bias),
    rms, standard deviation and the quantiles the receipt reads."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"count": 0, "mean": 0.0, "rms": 0.0, "std": 0.0, "quantiles": {}}
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(v.size), "mean": float(np.mean(v)), "rms": float(math.sqrt(np.mean(v ** 2))),
        "std": float(np.std(v)),
        "quantiles": {"p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
                      "p75": float(q[3]), "p95": float(q[4])},
    }


def _region_mask(lat, lon, region):
    lat0, lat1, lon0, lon1 = REGIONS[region]
    lon_w = np.where(np.asarray(lon) > 180.0, np.asarray(lon) - 360.0, np.asarray(lon))
    return (lat >= lat0) & (lat <= lat1) & (lon_w >= lon0) & (lon_w <= lon1)


def _resolve_time(analysis_time, batches) -> dt.datetime:
    if analysis_time is None:
        times = [t for b in batches if b.valid_time for t in b.valid_time]
        if not times:
            raise ValueError("analysis_time is None and no batch carries valid times")
        return max(times)
    if isinstance(analysis_time, dt.datetime):
        moment = analysis_time if analysis_time.tzinfo else analysis_time.replace(tzinfo=dt.timezone.utc)
        return moment.astimezone(dt.timezone.utc)
    parsed = parse_valid_time(str(analysis_time))
    if parsed is None:
        raise ValueError(f"analysis time {analysis_time!r} is not an ISO-8601 instant")
    return parsed


# ---------------------------------------------------------------------------
# Spectra and the transfer taper (amendment C)
# ---------------------------------------------------------------------------

def spectral_power_by_degree(coeff) -> np.ndarray:
    """``(T+1,)`` the grid mean square a real scalar field's coefficients
    carry per total degree, ``sum_m mult |c_nm|^2 / (4 pi)`` (mult 1 at
    m = 0, 2 for the mirrored m > 0), averaged over any leading (level)
    axes.  The sum over degrees is the field's grid mean square."""
    c = coeff.get() if hasattr(coeff, "get") else np.asarray(coeff)
    c = np.asarray(c, dtype=np.complex128)
    mult = np.where(np.arange(c.shape[-1]) == 0, 1.0, 2.0)
    power = np.sum(mult * (c.real ** 2 + c.imag ** 2), axis=-1) / (4.0 * math.pi)
    while power.ndim > 1:
        power = power.mean(axis=0)
    return np.asarray(power, dtype=np.float64)


def wind_power_by_degree(vorticity, divergence, radius_m: float) -> np.ndarray:
    """``(T+1,)`` the kinetic energy (J/kg, twice it is the mean square
    wind) a vorticity and divergence coefficient pair carries per degree,
    ``a^2 / (8 pi n(n+1)) sum_m mult (|zeta|^2 + |D|^2)``, averaged over
    levels."""
    out = None
    for coeff in (vorticity, divergence):
        c = coeff.get() if hasattr(coeff, "get") else np.asarray(coeff)
        c = np.asarray(c, dtype=np.complex128)
        n = np.arange(c.shape[-2], dtype=np.float64)
        lam = n * (n + 1.0)
        lam[0] = np.inf
        mult = np.where(np.arange(c.shape[-1]) == 0, 1.0, 2.0)
        power = np.sum(mult * (c.real ** 2 + c.imag ** 2), axis=-1) * (radius_m * radius_m / (8.0 * math.pi * lam))
        while power.ndim > 1:
            power = power.mean(axis=0)
        out = power if out is None else out + power
    return np.asarray(out, dtype=np.float64)


def taper_weights(truncation: int, start: int, end: int) -> np.ndarray:
    """``(T+1,)`` the transfer taper by degree: one up to ``start``, a
    raised cosine ``cos^2(pi/2 (n - start)/(end - start))`` between, zero
    at and above ``end`` (``end == start`` is a hard cut at ``end``)."""
    n = np.arange(int(truncation) + 1, dtype=np.float64)
    w = np.ones_like(n)
    if end > start:
        band = (n > start) & (n < end)
        w[band] = np.cos(0.5 * math.pi * (n[band] - start) / (end - start)) ** 2
    w[n >= end] = 0.0
    if end == start:
        w[n >= end] = 0.0
    return w


def apply_taper(coeff, weights: np.ndarray, xp):
    """``coeff (..., n, m)`` with every degree ``n`` scaled by
    ``weights[n]`` (bitwise unchanged where the weight is one)."""
    w = xp.asarray(weights, dtype=np.float64)
    return coeff * w[:, None]


def _band_edges(name_lo, name_hi, truncation: int) -> tuple[int, int]:
    def resolve(v):
        if v is None:
            return None
        if isinstance(v, str):
            if v == "T":
                return truncation
            if v == "T+1":
                return truncation + 1
            if v == "0.6T":
                return int(round(0.6 * truncation))
            if v == "0.6T+1":
                return int(round(0.6 * truncation)) + 1
            raise ValueError(v)
        return int(v)
    return resolve(name_lo), resolve(name_hi)


def band_summary(power: np.ndarray, ensemble_truncation: int) -> dict[str, dict[str, float]]:
    """``power`` per degree summed into :data:`SPECTRAL_BANDS` (edges
    resolved against the ensemble truncation), with each band's share of
    the total."""
    total = float(np.sum(power))
    out: dict[str, dict[str, float]] = {}
    top = power.size - 1
    for name, lo, hi in SPECTRAL_BANDS:
        a, b = _band_edges(lo, hi, int(ensemble_truncation))
        b = top if b is None else min(int(b), top)
        a = min(int(a), top + 1)
        value = float(np.sum(power[a:b + 1])) if b >= a else 0.0
        out[name] = {"degrees": [int(a), int(b)], "power": value,
                     "share": float(value / total) if total > 0.0 else 0.0}
    out["total"] = {"power": total}
    return out


def increment_spectrum_record(increment_spectral: dict[str, object], radius_m: float,
                              ensemble_truncation: int, exner_mean: float = 1.0) -> dict[str, object]:
    """The increment's spectrum by band for theta (as temperature through
    a mean Exner factor, K^2), ln ps and the wind (kinetic energy J/kg);
    the ``small_scale_share`` is the share above 0.6 of the ensemble
    truncation, what a reader inspects after a localised grid-space
    analysis (amendment C)."""
    out: dict[str, object] = {}
    theta = spectral_power_by_degree(increment_spectral["theta"]) * exner_mean ** 2
    lnps = spectral_power_by_degree(increment_spectral["log_surface_pressure"])
    wind = wind_power_by_degree(increment_spectral["vorticity"], increment_spectral["divergence"], radius_m)
    for name, power in (("temperature_k2", theta), ("ln_surface_pressure2", lnps), ("wind_ke_j_kg", wind)):
        bands = band_summary(power, ensemble_truncation)
        cut = int(round(0.6 * ensemble_truncation))
        total = float(np.sum(power))
        out[name] = {
            "bands": bands,
            "small_scale_share_above_0p6T": float(np.sum(power[cut + 1:]) / total) if total > 0.0 else 0.0,
            "by_degree": [float(v) for v in power],
        }
    return out


# ---------------------------------------------------------------------------
# Quality control and the withheld diagnostic
# ---------------------------------------------------------------------------

def _quality_control(batch: PointObs, ensemble: GlobalEnsemble, options: FilterOptions,
                     moment: dt.datetime, chain: dict) -> tuple[PointObs, dict[str, int]]:
    """The gates that keep a row out, counted by name."""
    counts = {name: 0 for name in ENSEMBLE_REJECTION_BREAKAGE}
    n = batch.count
    keep = np.ones(n, dtype=bool)
    if batch.variable in VARIABLE_TABLE:
        low, high = VARIABLE_TABLE[batch.variable]["gross_bounds"]
        gross = (batch.value < low) | (batch.value > high)
        counts["gross_bounds"] = int(gross.sum())
        keep &= ~gross
    if batch.valid_time is not None:
        ages = np.array([(moment - t).total_seconds() for t in batch.valid_time])
        old = ages > options.maximum_age_s
        future = ages < -options.future_tolerance_s
        counts["age_window"] = int((old & keep).sum())
        counts["future_time"] = int((future & keep).sum())
        keep &= ~old & ~future
    if batch.variable in ("wind_u_m_s", "wind_v_m_s"):
        polar = np.abs(np.asarray(batch.latitude_deg, dtype=np.float64)) >= POLE_LATITUDE_DEG
        counts["polar_wind"] = int((polar & keep).sum())
        keep &= ~polar
    floor = options.humidity_pressure_floor_pa
    if floor is not None and batch.variable == "dewpoint_k":
        aloft = ~np.asarray(batch.surface, dtype=bool)
        ln_p = np.where(aloft, np.nan_to_num(np.asarray(batch.ln_pressure, dtype=np.float64), nan=np.inf), np.inf)
        high = aloft & (np.exp(ln_p) < float(floor))
        counts["humidity_above_floor"] = int((high & keep).sum())
        keep &= ~high
    known = chain.get("reports", {})
    if known:
        held = np.array([str(h) in known for h in batch.identity])
        counts["already_assimilated"] = int((held & keep).sum())
        keep &= ~held
    if options.thinning and keep.any():
        grid = ensemble.transform.grid
        lat_nodes = np.asarray(grid.latitude_deg)
        order = np.argsort(lat_nodes)
        j = order[np.abs(lat_nodes[order][:, None] - batch.latitude_deg[None, :]).argmin(axis=0)]
        dlon = 360.0 / grid.nlon
        i = np.mod(np.round(np.mod(batch.longitude_deg, 360.0) / dlon).astype(int), grid.nlon)
        level_bin = np.where(batch.surface, -1, np.round(batch.ln_pressure / 0.2).astype(int))
        cell_lat = lat_nodes[j]
        cell_lon = i * dlon
        distance = np.hypot(batch.latitude_deg - cell_lat,
                            (np.mod(batch.longitude_deg - cell_lon + 180.0, 360.0) - 180.0)
                            * np.cos(np.deg2rad(cell_lat)))
        best: dict[tuple, int] = {}
        for idx in np.nonzero(keep)[0]:
            key = (int(j[idx]), int(i[idx]), int(level_bin[idx]))
            held = best.get(key)
            if held is None or distance[idx] < distance[held]:
                best[key] = int(idx)
        chosen = np.zeros(n, dtype=bool)
        chosen[list(best.values())] = True
        counts["thinned"] = int((keep & ~chosen).sum())
        keep &= chosen
    if batch.simulated is not None and keep.any():
        finite = np.isfinite(np.asarray(batch.simulated, dtype=np.float64)).all(axis=0)
        if batch.control_simulated is not None:
            finite &= np.isfinite(np.asarray(batch.control_simulated, dtype=np.float64)).all(axis=0)
        counts["no_finite_equivalent"] = int((~finite & keep).sum())
        keep &= finite
    if batch.simulated is not None and keep.any():
        innovation = batch.innovation()
        spread = batch.spread()
        limit = options.background_check_sigmas * np.sqrt(batch.error ** 2 + spread ** 2)
        failed = np.abs(innovation) > limit
        counts["background_check"] = int((failed & keep).sum())
        keep &= ~failed
    rejections = {name: count for name, count in counts.items() if count}
    out = batch.subset(keep)
    out.rejections = rejections
    return out, rejections


def _withhold(batch: PointObs, options: FilterOptions, variable_index: int) -> tuple[PointObs, PointObs]:
    """The v1 door's split, kept as a diagnostic: below the minimum every
    row is used; at or above it a seeded fraction of the rows, ordered by
    identity hash (or by position when the stream carries none), is
    withheld and judged."""
    n = batch.count
    if n < options.gate_minimum_count:
        return batch, batch.subset(np.zeros(n, dtype=bool))
    keys = [
        str(h) if h else f"{lat:.4f}|{lon:.4f}|{lnp:.5f}|{val:.6g}"
        for h, lat, lon, lnp, val in zip(
            batch.identity, batch.latitude_deg, batch.longitude_deg, batch.ln_pressure, batch.value)
    ]
    ordered = np.argsort(np.array(keys, dtype=object).astype(str))
    withheld_count = int(round(options.withheld_fraction * n))
    rng = np.random.default_rng([int(options.withheld_seed), int(variable_index)])
    permutation = rng.permutation(n)
    chosen_positions = set(permutation[:withheld_count].tolist())
    held = np.zeros(n, dtype=bool)
    for position, idx in enumerate(ordered):
        if position in chosen_positions:
            held[idx] = True
    return batch.subset(~held), batch.subset(held)


def _static_draws(ensemble: GlobalEnsemble, batches: list[PointObs], operators: MemberOperators,
                  options: FilterOptions, *, chunk: int = 16) -> tuple[dict[str, object], dict[str, object]]:
    """The hybrid's static draws for one analysis (``FilterOptions.
    hybrid_beta`` below one): K draws from the static covariance table on
    the ensemble grid (``{field: (K, ...)}`` for the analysis fields) and,
    on every batch, ``static_simulated = H(x_mean + x_s) - H(x_mean)``
    through the same operators the members use (the ensemble-mean spectral
    state as the linearisation point, member 0's surface and physics
    shared by every draw; the draws are evaluated ``chunk`` at a time so
    the resident copies stay a chunk's worth).  Returns ``(static_prior,
    record)``."""
    from .static_covariance import draw_static_perturbations, load_static_covariance

    clock = time.perf_counter()
    transform = ensemble.transform
    model = ensemble.model
    backend = transform.backend
    xp = backend.xp
    table = load_static_covariance(options.static_covariance)
    if table is None:
        raise ValueError(
            f"hybrid_beta {options.hybrid_beta} needs a static covariance table and "
            "FilterOptions.static_covariance names none"
        )
    table.check_against(int(transform.truncation), int(model.nlev))
    k = int(options.static_samples)
    rng = member_rng(int(options.static_seed), 0, "static-covariance", int(ensemble.cycles))
    draws = draw_static_perturbations(table, transform, model.vector, k, rng, include_spectral=True)
    spectral = draws.pop("spectral")
    static_prior = {name: draws[name] for name in options.analysis_fields}
    # The pseudo-states the operators read: the ensemble mean plus each draw.
    mean = ensemble.mean_spectral()
    base = ensemble.members[0]
    mean_fields = [mean[name] for name in SPECTRAL_FIELDS]
    mean_state = ArwenGlobalState(base.atmosphere.with_fields(mean_fields), base.surface, base.physics_state)

    def evaluate(states):
        """H on ``states`` for every batch: the package's operators for the
        neutral vocabulary in one call, a foreign batch (a radiance, a
        refractivity) through its own operator on the same states."""
        values = evaluate_batches(operators, states, batches, target=None)
        for i, batch in enumerate(batches):
            if values[i] is None and batch.operator is not None:
                values[i] = np.asarray(batch.operator(states, batch), dtype=np.float64)
        return values

    hx_mean = evaluate([mean_state])
    static_hx = [None if h is None else np.zeros((k, np.asarray(h).shape[1])) for h in hx_mean]
    for start in range(0, k, max(1, int(chunk))):
        states = []
        for j in range(start, min(k, start + int(chunk))):
            fields = [mean[name] + xp.asarray(spectral[name][j], dtype=backend.complex_dtype)
                      for name in SPECTRAL_FIELDS]
            states.append(ArwenGlobalState(base.atmosphere.with_fields(fields), base.surface, base.physics_state))
        hx = evaluate(states)
        for i, values in enumerate(hx):
            if values is None:
                continue
            static_hx[i][start:start + len(states)] = np.asarray(values) - np.asarray(hx_mean[i])
        del states
        model.release_syntheses()
    obs_space = {}
    for batch, values in zip(batches, static_hx):
        if values is None:
            raise ValueError(
                f"the hybrid cannot evaluate the static draws on {batch.stream}/{batch.variable}: "
                "a stream outside the neutral vocabulary has no operator the draws can pass through"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"the static draws' H on {batch.stream}/{batch.variable} came back non-finite; "
                "the hybrid refuses rather than poison the augmented solve"
            )
        batch.static_simulated = values
        static_spread = float(np.sqrt(np.mean(values ** 2))) if values.size else 0.0
        ensemble_spread = float(np.sqrt(np.mean(batch.spread() ** 2))) if batch.count else 0.0
        obs_space.setdefault(batch.stream, {})[batch.variable] = {
            "rows": int(batch.count),
            "static_spread_rms": static_spread,
            "ensemble_spread_rms": ensemble_spread,
            "ratio_static_over_ensemble": (static_spread / ensemble_spread) if ensemble_spread > 0.0 else None,
        }
    to_numpy = backend.to_numpy
    grid_rms = {name: float(np.sqrt(np.mean(np.asarray(to_numpy(static_prior[name]), dtype=np.float64) ** 2)))
                for name in static_prior}
    record = {
        "beta": float(options.hybrid_beta),
        "static_samples": k,
        "static_seed": int(options.static_seed),
        "cycle": int(ensemble.cycles),
        "table": table.identity(),
        "table_path": table.receipt.get("path"),
        "form": (
            "the control's gain is that of beta L o P_ens + (1 - beta) L o P_static, one "
            "positive-semidefinite covariance in the augmented perturbation space of the members and "
            "the static draws inside the localised solve (letkf_point); the members keep the ensemble "
            "transform and receive the hybrid increment through the recentring"
        ),
        "draw_grid_rms": grid_rms,
        "observation_space": obs_space,
        "seconds": float(time.perf_counter() - clock),
    }
    return static_prior, record
def solve_on_path(prior, flat, geometry, letkf_config, diagnostics, *, solve_path: str, xp, to_numpy,
                  static_prior=None, hybrid_beta: float = 1.0):
    """:func:`analyze_points_with_control` on the namespace ``solve_path``
    names: ``auto`` runs where the prior lives, ``device`` requires a device
    namespace, ``host`` moves the prior, the observations and the geometry
    to numpy, solves there and hands the increments back on the members'
    namespace.  The two paths are one code in two array modules; the
    receipt names the one taken and its wall.  ``static_prior`` and
    ``hybrid_beta`` are the hybrid's (the static draws follow the prior to
    the host on that path)."""
    if solve_path == "device" and xp is np:
        raise ValueError(
            "solve_path 'device' asks for the localised solve on a card and the members "
            "live in numpy; run the model on a device backend or choose 'auto' or 'host'"
        )
    if solve_path == "host" and xp is not np:
        host_prior = {name: np.asarray(to_numpy(arr)) for name, arr in prior.items()}
        prior.clear()
        host_flat = dataclasses.replace(flat, **{
            f.name: (np.asarray(to_numpy(getattr(flat, f.name)))
                     if f.name != "count" and getattr(flat, f.name) is not None else getattr(flat, f.name))
            for f in dataclasses.fields(flat)
        })
        host_geometry = ColumnGeometry(
            latitude_deg=geometry.latitude_deg, longitude_deg=geometry.longitude_deg,
            ln_p_full=np.asarray(to_numpy(geometry.ln_p_full)), ln_ps=np.asarray(to_numpy(geometry.ln_ps)),
            radius_m=geometry.radius_m,
        )
        host_static = (None if static_prior is None
                       else {name: np.asarray(to_numpy(arr)) for name, arr in static_prior.items()})
        solved = analyze_points_with_control(host_prior, host_flat, host_geometry, letkf_config, diagnostics,
                                             static_prior=host_static, hybrid_beta=hybrid_beta)
        del host_prior, host_flat, host_static
        increments = {name: xp.asarray(arr) for name, arr in solved.increments.items()}
        control = (None if solved.control_increment is None
                   else {name: xp.asarray(arr) for name, arr in solved.control_increment.items()})
        return PointAnalysis(increments, control)
    return analyze_points_with_control(prior, flat, geometry, letkf_config, diagnostics,
                                       static_prior=static_prior, hybrid_beta=hybrid_beta)


def _grid_prior(ensemble: GlobalEnsemble, options: FilterOptions):
    names = tuple(options.analysis_fields)
    fields, ln_p_full, ln_ps = ensemble.grid_fields(names)
    grid = ensemble.transform.grid
    geometry = ColumnGeometry(
        latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg,
        ln_p_full=ln_p_full, ln_ps=ln_ps, radius_m=float(grid.radius_m),
    )
    return fields, geometry


# ---------------------------------------------------------------------------
# Grid increments into spectral states
# ---------------------------------------------------------------------------

def grid_increment_to_spectral(model, transform, increment: dict[str, object], options: FilterOptions,
                               *, wind_records: list | None = None) -> dict[str, object]:
    """One grid increment ``{psi, chi | u, v, theta, qv, lnps}`` (any
    subset, one member's shapes) analysed into the five spectral fields of
    the transform's triangle: theta, qv and ln ps through the forward
    analysis (the triangular truncation smooths the increment), the wind
    through the Laplacian of the potentials (or the vector analysis of the
    components) under the balance rule.  Fields not present are zero."""
    backend = transform.backend
    xp = backend.xp
    nlev = int(model.nlev)
    out = {name: None for name in SPECTRAL_FIELDS}
    if "theta" in increment:
        out["theta"] = transform.project(transform.forward(xp.asarray(increment["theta"], dtype=backend.float_dtype)))
    if "qv" in increment:
        out["qv"] = transform.project(transform.forward(xp.asarray(increment["qv"], dtype=backend.float_dtype)))
    if "lnps" in increment:
        out["log_surface_pressure"] = transform.project(
            transform.forward(xp.asarray(increment["lnps"], dtype=backend.float_dtype)))
    wind_pair = None
    if "psi" in increment and "chi" in increment:
        # The potentials' increments become vorticity and divergence
        # through the Laplacian (exact in the triangle), and the wind
        # the balance rule reads is synthesised from them.
        psi_spec = transform.project(transform.forward(xp.asarray(increment["psi"], dtype=backend.float_dtype)))
        chi_spec = transform.project(transform.forward(xp.asarray(increment["chi"], dtype=backend.float_dtype)))
        u_inc, v_inc = model.vector.wind_from_vordiv(transform.laplacian(psi_spec), transform.laplacian(chi_spec))
        wind_pair = (u_inc, v_inc)
    elif "u" in increment and "v" in increment:
        wind_pair = (increment["u"], increment["v"])
    if wind_pair is not None:
        zeta_inc, div_inc, wind_record = _balanced_wind_increment(
            model.vector, transform, wind_pair[0], wind_pair[1], options.wind_balance)
        out["vorticity"] = zeta_inc
        out["divergence"] = div_inc
        if wind_records is not None:
            wind_records.append({
                "divergent_fraction_analysed": wind_record["divergent_fraction_analysed"],
                "divergent_fraction_applied": wind_record["divergent_fraction_applied"],
                "rotational_ke_j_kg": wind_record["rotational_ke_j_kg"],
                "divergent_ke_j_kg": wind_record["divergent_ke_j_kg"],
            })
    for name in SPECTRAL_FIELDS:
        if out[name] is None:
            out[name] = transform.zeros(1)[0] if name == "log_surface_pressure" else transform.zeros(nlev)
    return out


def _mass_offset(transform, ln_ps_before, ln_ps_after) -> float:
    backend = transform.backend
    xp = backend.xp
    grid = transform.grid
    ps_before = np.asarray(backend.to_numpy(xp.exp(transform.inverse(ln_ps_before))), dtype=np.float64)
    ps_after = np.asarray(backend.to_numpy(xp.exp(transform.inverse(ln_ps_after))), dtype=np.float64)
    return math.log(grid.global_mean(ps_before) / grid.global_mean(ps_after))


def _added_state(model, transform, state: ArwenGlobalState, increment_spectral: dict[str, object],
                 options: FilterOptions) -> tuple[ArwenGlobalState, float, dict]:
    """``state`` plus the spectral increment: the global-mean surface
    pressure kept, the vapor repaired, the state enforced.  Returns
    ``(state, mass_offset, repair)``."""
    fields = list(state.atmosphere.fields())
    index = {name: i for i, name in enumerate(SPECTRAL_FIELDS)}
    for name in SPECTRAL_FIELDS:
        inc = increment_spectral.get(name)
        if inc is not None:
            fields[index[name]] = fields[index[name]] + inc
    offset = 0.0
    if options.preserve_global_mean_pressure:
        offset = _mass_offset(transform, state.atmosphere.log_surface_pressure, fields[index["log_surface_pressure"]])
        fields[index["log_surface_pressure"]] = transform.add_grid_constant(
            fields[index["log_surface_pressure"]], offset)
    new = ArwenGlobalState(state.atmosphere.with_fields(fields), state.surface, state.physics_state)
    new, negative_vapor, _tracer, fixer = model._repair_positivity(new)
    model.enforce(new)
    repair = {"largest_negative_vapor_kg_kg": float(negative_vapor), **{
        k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in dict(fixer).items()}}
    return new, offset, repair


#: The surface pressure above which a state dies in the radiation call:
#: RRTMGP's pressure tables end at 109,663.316 Pa of layer pressure (the
#: physics refuses ``play range ... outside allowed range [1.00518357,
#: 109663.316] Pa``) and the lowest layer's pressure lies within a few
#: pascals of the surface pressure.  Measured 2026-09-06 on the T63 / T127
#: twin: the T63 truncation's orography undershoots below sea level at the
#: Andes' Pacific foot (19.6 S, 73.1 W), so every T63 member carries 109.2
#: to 109.7 kPa there by construction and moves it by several hundred
#: pascals within an hour of free integration; at seed 4242 one member
#: started 48 Pa above the ceiling and died in the first radiation call
#: with no analysis at all.  The observation vocabulary's gross bound for
#: a surface-pressure REPORT (108,000 Pa) is not a bound on a state:
#: applied to the members (ff15274ff) it refused every T63 member on every
#: cycle of the reference twin and the ensemble's own update silently
#: stopped while the receipt read pass.
RADIATION_SURFACE_PRESSURE_CEILING_PA = 109_663.0

MEMBER_CEILING_BREAKAGE = (
    "a member whose surface pressure exceeds the radiation tables' ceiling "
    f"({RADIATION_SURFACE_PRESSURE_CEILING_PA:,.0f} Pa of layer pressure) dies in "
    "the window's first radiation call and one dead member ends the cycle; the "
    "member keeps its increment and is named here (members_above_radiation_ceiling) "
    "so the reader sees the death before the physics reports it"
)


def member_surface_pressure_extrema(model, transform, state: ArwenGlobalState) -> dict[str, object]:
    """A state's surface-pressure extrema on its own grid, the column of
    the maximum, and how many columns lie above the radiation ceiling."""
    backend = transform.backend
    g = model.grid_state(state.atmosphere, only=("ps",))
    ps = np.asarray(backend.to_numpy(g["ps"]), dtype=np.float64)
    model.release_syntheses()
    j, i = np.unravel_index(int(np.argmax(ps)), ps.shape)
    grid = transform.grid
    return {
        "surface_pressure_min_pa": float(ps.min()), "surface_pressure_max_pa": float(ps.max()),
        "column_of_max": {"latitude_deg": float(np.asarray(grid.latitude_deg)[j]),
                          "longitude_deg": float(np.asarray(grid.longitude_deg)[i])},
        "columns_above_ceiling": int(np.sum(~np.isfinite(ps) | (ps > RADIATION_SURFACE_PRESSURE_CEILING_PA))),
        "ceiling_pa": float(RADIATION_SURFACE_PRESSURE_CEILING_PA),
    }


def members_surface_pressure_record(extrema: list[dict]) -> dict[str, object]:
    """The ensemble's surface-pressure reading from per-member extrema: the
    maximum with its member and column, every member's maximum, the
    headroom under the ceiling, and the members above it, each named with
    the breakage."""
    if not extrema:
        return {"members_surface_pressure_pa": None, "members_above_radiation_ceiling": []}
    worst = int(np.argmax([e["surface_pressure_max_pa"] for e in extrema]))
    above = [k for k, e in enumerate(extrema) if e["columns_above_ceiling"] > 0]
    return {
        "members_surface_pressure_pa": {
            "min": float(min(e["surface_pressure_min_pa"] for e in extrema)),
            "max": float(extrema[worst]["surface_pressure_max_pa"]),
            "max_member": worst,
            "column_of_max": extrema[worst]["column_of_max"],
            "per_member_max": [float(e["surface_pressure_max_pa"]) for e in extrema],
            "ceiling_pa": float(RADIATION_SURFACE_PRESSURE_CEILING_PA),
            "headroom_pa": float(RADIATION_SURFACE_PRESSURE_CEILING_PA - extrema[worst]["surface_pressure_max_pa"]),
        },
        "members_above_radiation_ceiling": [
            {"member": int(k), **extrema[k], "breakage": MEMBER_CEILING_BREAKAGE} for k in above
        ],
    }


def _apply_increments(ensemble: GlobalEnsemble, increments: dict[str, object], options: FilterOptions,
                      ) -> tuple[dict[str, object], dict[str, object], list[ArwenGlobalState]]:
    """The grid increments into every member's spectral state (direct) or
    onto the ensemble's pending increments (IAU).  Returns
    ``(mean_increment_spectral, record, analysed_members)`` where the
    analysed members are the direct-insertion result in both modes (under
    IAU they are throwaway copies the receipt's O-A is read from, and the
    ensemble's own members stay the background until the window adds the
    portions)."""
    transform = ensemble.transform
    model = ensemble.model
    names = set(options.analysis_fields)
    before = ensemble.mean_spectral()
    wind_records: list = []
    mass_offsets = []
    spectral_per_member = []
    analysed: list[ArwenGlobalState] = []
    extrema: list[dict] = []
    for k, member in enumerate(ensemble.members):
        grid_inc = {name: increments[name][k] for name in names}
        inc_spectral = grid_increment_to_spectral(model, transform, grid_inc, options, wind_records=wind_records)
        state, offset, _repair = _added_state(model, transform, member, inc_spectral, options)
        # The analysed member's surface pressure against the radiation
        # ceiling: recorded and named, never refused (a state above the
        # observation vocabulary's report bound is a legitimate state at a
        # truncated orography; see RADIATION_SURFACE_PRESSURE_CEILING_PA).
        extrema.append(member_surface_pressure_extrema(model, transform, state))
        if options.preserve_global_mean_pressure and "lnps" in names:
            inc_spectral["log_surface_pressure"] = transform.add_grid_constant(
                inc_spectral["log_surface_pressure"], offset)
            mass_offsets.append(offset)
        analysed.append(state)
        spectral_per_member.append(inc_spectral)
        model.release_syntheses()
    if options.increment_application == "iau":
        ensemble.schedule_increments(spectral_per_member, float(options.iau_window_s))
        after = {name: before[name] + sum(inc[name] for inc in spectral_per_member) / float(ensemble.size)
                 for name in SPECTRAL_FIELDS}
    else:
        for k, state in enumerate(analysed):
            ensemble.members[k] = state
        after = ensemble.mean_spectral()
    mean_increment = {name: after[name] - before[name] for name in SPECTRAL_FIELDS}
    record = {
        "route": (
            "theta, qv and ln ps through the transform's forward analysis (the "
            "triangular truncation smooths the increment); the wind through the "
            "vector analysis under the balance rule; the global-mean surface "
            "pressure kept per member; the vapor repaired by the model's positivity "
            "repair; the state enforced"
            + ("; under iau the members keep the background and the portions are "
               "added before each step of the window" if options.increment_application == "iau" else "")
        ),
        "application": options.increment_application,
        "wind_balance": {
            "mode": options.wind_balance,
            "divergent_fraction_analysed_mean": (
                float(np.mean([w["divergent_fraction_analysed"] for w in wind_records])) if wind_records else None),
            "divergent_fraction_applied_mean": (
                float(np.mean([w["divergent_fraction_applied"] for w in wind_records])) if wind_records else None),
        },
        "mass_preserving_log_offset": {
            "mean": float(np.mean(mass_offsets)) if mass_offsets else 0.0,
            "maxabs": float(np.max(np.abs(mass_offsets))) if mass_offsets else 0.0,
        },
        **members_surface_pressure_record(extrema),
    }
    return mean_increment, record, analysed


def _grid_rms_of_spectral(transform, coeff) -> float:
    backend = transform.backend
    grid = backend.to_numpy(transform.inverse(coeff))
    return float(np.sqrt(np.mean(np.asarray(grid, dtype=np.float64) ** 2)))


def _increment_summary(model, transform, reference_atmosphere, increment: dict[str, object]) -> dict[str, float]:
    backend = transform.backend
    out = {}
    g = model.grid_state(reference_atmosphere, only=("p_full",))
    exner = np.asarray(backend.to_numpy((g["p_full"] / REFERENCE_PRESSURE_PA) ** KAPPA), dtype=np.float64)
    model.release_syntheses()
    theta = np.asarray(backend.to_numpy(transform.inverse(increment["theta"])), dtype=np.float64)
    out["temperature_k_rms"] = float(np.sqrt(np.mean((theta * exner) ** 2)))
    u, v = model.vector.wind_from_vordiv(increment["vorticity"], increment["divergence"])
    out["wind_m_s_rms"] = float(np.sqrt(np.mean(
        np.asarray(backend.to_numpy(u), dtype=np.float64) ** 2
        + np.asarray(backend.to_numpy(v), dtype=np.float64) ** 2)))
    out["ln_surface_pressure_rms"] = _grid_rms_of_spectral(transform, increment["log_surface_pressure"])
    out["qv_kg_kg_rms"] = _grid_rms_of_spectral(transform, increment["qv"])
    return out


def _mean_exner(model, transform, atmosphere) -> float:
    backend = transform.backend
    g = model.grid_state(atmosphere, only=("p_full",))
    exner = np.asarray(backend.to_numpy((g["p_full"] / REFERENCE_PRESSURE_PA) ** KAPPA), dtype=np.float64)
    model.release_syntheses()
    return float(np.sqrt(np.mean(exner ** 2)))


# ---------------------------------------------------------------------------
# The control analysis (amendment A) and its transfer (amendment C)
# ---------------------------------------------------------------------------

BALANCE_RULE = (
    "measured, never imposed: the geostrophic wind of the increment's hydrostatic geopotential "
    "is compared with the increment's own wind in the extratropics (|latitude| at or above the "
    "stated bound) at the stated pressures, the divergent share of the wind increment's kinetic "
    "energy is recorded, and no balance operator touches the increment anywhere; the balance the "
    "increment carries is the ensemble covariance's, which the linearly balanced perturbation "
    "family gives it, and the tropics are left to their own dynamics"
)


def increment_balance_record(model, transform, base_atmosphere, increment_spectral: dict[str, object], *,
                             levels_hpa=(850.0, 500.0, 250.0), latitude_bound_deg: float = 25.0) -> dict[str, object]:
    """The mass-wind consistency of one spectral increment on ``base_atmosphere``:
    at the full level nearest each of ``levels_hpa`` (the base state's
    global-mean column), the geopotential increment (the hydrostatic
    geopotential of the base plus the increment, minus the base's), its
    geostrophic wind ``(-(1/f) d phi/dy, (1/f) d phi/dx)`` and the increment's
    own wind, over the columns with ``|latitude| >= latitude_bound_deg``:
    the area-weighted rms of each, the ageostrophic residual's rms, its
    share of the wind increment, and the area-weighted correlation between
    the two wind vectors.  A balanced synoptic increment reads a correlation
    near one and an ageostrophic share well below one; the record of
    2026-09-06 read 0.23 to 0.47 and 1.3 to 2.8 (the unbalanced family).
    ``surface_pressure_increment_rms_pa`` and the geopotential increment's
    rms in metres per level are recorded beside them."""
    backend = transform.backend
    xp = backend.xp
    index = {name: i for i, name in enumerate(SPECTRAL_FIELDS)}
    fields = list(base_atmosphere.fields())
    added = list(fields)
    for name in SPECTRAL_FIELDS:
        inc = increment_spectral.get(name)
        if inc is not None:
            added[index[name]] = fields[index[name]] + xp.asarray(inc, dtype=backend.complex_dtype)
    g0 = model.grid_state(base_atmosphere, only=("geopotential", "p_full", "u", "v", "ps", "virtual_temperature"))
    p_ref = np.asarray(backend.to_numpy(g0["p_full"]), dtype=np.float64)
    weights = np.asarray(transform.grid.quadrature_weights, dtype=np.float64)
    lat = np.asarray(transform.grid.latitude_deg, dtype=np.float64)
    nlon = int(transform.grid.nlon)
    p_col = np.array([float(np.sum(weights * p_ref[k].mean(axis=1)) / np.sum(weights)) for k in range(p_ref.shape[0])])
    phi0 = g0["geopotential"]
    u0, v0 = g0["u"], g0["v"]
    ps0 = np.asarray(backend.to_numpy(g0["ps"]), dtype=np.float64)
    ln_p0 = xp.log(g0["p_full"])
    tv0 = g0["virtual_temperature"]
    model.release_syntheses()
    g1 = model.grid_state(base_atmosphere.with_fields(added), only=("geopotential", "u", "v", "ps", "p_full"))
    # The geopotential increment AT FIXED PRESSURE: the model levels move
    # with the surface pressure, and a level's own geopotential change
    # carries the level's motion (R Tv d ln p, the whole of the bottom
    # level's balanced geopotential at the surface, where B is one); the
    # geostrophic relation holds on a pressure surface.
    dphi = g1["geopotential"] - phi0 + DRY_AIR_GAS_CONSTANT * tv0 * (xp.log(g1["p_full"]) - ln_p0)
    du = g1["u"] - u0
    dv = g1["v"] - v0
    dps = np.asarray(backend.to_numpy(g1["ps"]), dtype=np.float64) - ps0
    model.release_syntheses()
    omega = float(getattr(model, "rotation_rate_s", EARTH_ROTATION_RATE_S))
    f = 2.0 * omega * np.sin(np.deg2rad(lat))
    mask = np.abs(lat) >= float(latitude_bound_deg)
    w2 = np.repeat(weights[:, None], nlon, axis=1) * mask[:, None]
    wsum = float(np.sum(w2))

    def wrms(a):
        return float(math.sqrt(np.sum(w2 * a ** 2) / wsum)) if wsum > 0 else None

    out: dict[str, object] = {
        "rule": BALANCE_RULE,
        "latitude_bound_deg": float(latitude_bound_deg),
        "surface_pressure_increment_rms_pa": float(math.sqrt(np.sum(weights[:, None] * dps ** 2) / (np.sum(weights) * nlon))),
        "levels": {},
    }
    for target in levels_hpa:
        k = int(np.argmin(np.abs(p_col - float(target) * 100.0)))
        dphi_k = transform.project(transform.forward(dphi[k]))
        d_east, d_north = transform.gradient(dphi_k)
        d_east = np.asarray(backend.to_numpy(d_east), dtype=np.float64)
        d_north = np.asarray(backend.to_numpy(d_north), dtype=np.float64)
        f_safe = np.where(mask, f, 1.0)[:, None]
        ug = np.where(mask[:, None], -d_north / f_safe, 0.0)
        vg = np.where(mask[:, None], d_east / f_safe, 0.0)
        du_k = np.asarray(backend.to_numpy(du[k]), dtype=np.float64)
        dv_k = np.asarray(backend.to_numpy(dv[k]), dtype=np.float64)
        dphi_grid = np.asarray(backend.to_numpy(dphi[k]), dtype=np.float64)
        wind_rms = wrms(np.hypot(du_k, dv_k))
        geo_rms = wrms(np.hypot(ug, vg))
        ageo_rms = wrms(np.hypot(du_k - ug, dv_k - vg))
        num = float(np.sum(w2 * (du_k * ug + dv_k * vg)))
        den = math.sqrt(float(np.sum(w2 * (du_k ** 2 + dv_k ** 2))) * float(np.sum(w2 * (ug ** 2 + vg ** 2))))
        out["levels"][f"{int(round(target))}"] = {
            "level_index": k,
            "pressure_hpa": float(p_col[k] / 100.0),
            "geopotential_increment_rms_m": float(math.sqrt(np.sum(weights[:, None] * dphi_grid ** 2) / (np.sum(weights) * nlon)) / 9.80665),
            "wind_increment_rms_m_s": wind_rms,
            "geostrophic_wind_of_increment_rms_m_s": geo_rms,
            "ageostrophic_residual_rms_m_s": ageo_rms,
            "ageostrophic_share": (float(ageo_rms / wind_rms) if wind_rms else None),
            "correlation_wind_vs_geostrophic": (float(num / den) if den > 0.0 else None),
        }
    return out


def apply_control_increment(control: ControlBackground, control_increment_grid: dict[str, object],
                            ensemble: GlobalEnsemble, options: FilterOptions,
                            ) -> tuple[ArwenGlobalState, dict[str, object], dict[str, object]]:
    """The control (high-resolution) analysis from the control increment on
    the ENSEMBLE grid: the grid increment analysed into the ensemble
    triangle under the balance rule, the transfer taper applied by degree,
    the tapered increment embedded in the control's triangle (degrees
    above the ensemble truncation exactly zero), added to the control
    background with the global-mean surface pressure kept, the vapor
    repaired and the state enforced.  Returns ``(analysis, record,
    tapered_increment_at_ensemble_truncation)``; the record carries the
    taper, the increment spectrum by band before and after it, the mass
    offset, the repair and the wind balance."""
    ens_transform = ensemble.transform
    ens_model = ensemble.model
    xp_e = ens_transform.backend.xp
    ens_t = int(ens_transform.truncation)
    det_t = control.truncation
    if det_t < ens_t:
        raise ValueError(
            f"the control is T{det_t} and the ensemble T{ens_t}; the dual-resolution "
            "design puts the control at or above the ensemble"
        )
    wind_records: list = []
    raw = grid_increment_to_spectral(ens_model, ens_transform, control_increment_grid, options,
                                     wind_records=wind_records)
    start, end = options.taper_degrees(ens_t)
    weights = taper_weights(ens_t, start, end)
    radius = float(ens_transform.grid.radius_m)
    exner_mean = _mean_exner(ens_model, ens_transform, ensemble.members[0].atmosphere)
    spectrum_before = increment_spectrum_record(raw, radius, ens_t, exner_mean)
    tapered = {name: apply_taper(raw[name], weights, xp_e) for name in SPECTRAL_FIELDS}
    spectrum_after = increment_spectrum_record(tapered, radius, ens_t, exner_mean)
    balance = increment_balance_record(ens_model, ens_transform, ensemble.members[0].atmosphere, tapered)
    balance["divergent_share_of_wind_increment_ke"] = (
        float(wind_records[0]["divergent_fraction_analysed"]) if wind_records else None)
    # Into the control's triangle.
    det_transform = control.transform
    det_backend = det_transform.backend
    xp_d = det_backend.xp
    embedded = {}
    for name in SPECTRAL_FIELDS:
        coeff = tapered[name]
        host = coeff.get() if hasattr(coeff, "get") else np.asarray(coeff)
        embedded[name] = det_transform.project(
            xp_d.asarray(embed_spectral(host, det_t), dtype=det_backend.complex_dtype))
    physics = control.state.physics_state.copy()
    background = ArwenGlobalState(control.state.atmosphere, control.state.surface, physics)
    analysis, offset, repair = _added_state(control.model, det_transform, background, embedded, options)
    control.model.release_syntheses()
    if options.preserve_global_mean_pressure:
        # The increment handed back for recentring carries the mass rule's
        # constant, as every member's own increment does in
        # _apply_increments: the members' mean increment is replaced by this
        # one at recentring, so an increment without the constant moved the
        # ensemble's global-mean surface pressure by the control's raw
        # increment mean every cycle (measured 2026-09-06 on the first arm:
        # the members' mean ln ps fell 3.1e-4 below the control's in three
        # cycles, and the members' METAR pressure O-B bias ran 78 Pa above
        # the control's after 24; on the T7 / T3 twin one analysis moved the
        # ensemble's mean by -7.2 Pa where the control kept its own).
        tapered["log_surface_pressure"] = ens_transform.add_grid_constant(
            tapered["log_surface_pressure"], offset)
    record = {
        "route": (
            f"control increment on the T{ens_t} ensemble grid analysed into the T{ens_t} triangle "
            f"(wind under the balance rule), tapered by degree, embedded in the T{det_t} triangle "
            f"(degrees above T{ens_t} exactly zero), added to the control background; the "
            "global-mean surface pressure kept; the vapor repaired; the state enforced"
        ),
        "innovation": "y - H(x_H^b) on every assimilated row, the control's own (amendment A)",
        "taper": {"start_degree": int(start), "end_degree": int(end), "shape": "raised cosine in degree",
                  "weights_by_degree": [float(w) for w in weights]},
        "spectrum_before_taper": spectrum_before,
        "spectrum_after_taper": spectrum_after,
        "wind_balance": {
            "mode": options.wind_balance,
            **({k: wind_records[0][k] for k in wind_records[0]} if wind_records else {}),
        },
        "mass_preserving_log_offset": float(offset),
        "positivity_repair": repair,
        "increment": _increment_summary(control.model, det_transform, control.state.atmosphere, embedded),
        "balance": balance,
    }
    return analysis, record, tapered


def apply_mean_increment(
    deterministic: ArwenGlobalState, det_model, det_transform,
    mean_increment_spectral: dict[str, object], *,
    options: FilterOptions | None = None,
) -> tuple[ArwenGlobalState, dict[str, object]]:
    """The COMPARISON experiment (the analysis path before amendment A):
    the ensemble-mean increment embedded in the deterministic triangle
    (degrees above the ensemble truncation exactly zero, the transfer taper
    applied) and added to the deterministic background; the global-mean
    surface pressure kept, the vapor repaired, the state enforced.  Kept
    for the OSSE's ``transfer`` family, which grades it beside the control
    analysis; the door does not call it.  Returns ``(analysis, record)``."""
    options = options or FilterOptions()
    backend = det_transform.backend
    xp = backend.xp
    det_t = int(det_transform.truncation)
    ens_t = None
    embedded = {}
    for name in SPECTRAL_FIELDS:
        inc = mean_increment_spectral.get(name)
        if inc is None:
            continue
        host = inc.get() if hasattr(inc, "get") else np.asarray(inc)
        ens_t = int(host.shape[-1]) - 1
        start, end = options.taper_degrees(ens_t)
        host = host * taper_weights(ens_t, start, end)[:, None]
        embedded[name] = det_transform.project(xp.asarray(embed_spectral(host, det_t), dtype=backend.complex_dtype))
    physics = deterministic.physics_state.copy()
    background = ArwenGlobalState(deterministic.atmosphere, deterministic.surface, physics)
    state, offset, repair = _added_state(det_model, det_transform, background, embedded, options)
    det_model.release_syntheses()
    record = {
        "embedding": f"T{ens_t} ensemble-mean increment tapered and embedded in the T{det_t} triangle, "
                     f"degrees above T{ens_t} exactly zero (comparison experiment, not the analysis path)",
        "mass_preserving_log_offset": float(offset),
        "positivity_repair": repair,
    }
    return state, record


def _member_global_mean_ps(transform, ln_ps) -> float:
    backend = transform.backend
    xp = backend.xp
    return float(transform.grid.global_mean(
        np.asarray(backend.to_numpy(xp.exp(transform.inverse(ln_ps))), dtype=np.float64)))


def recenter(
    ensemble: GlobalEnsemble, deterministic_analysis: ArwenGlobalState,
    det_transform, *, fraction: float | None = None, mode: str = "increment",
    control_increment: dict[str, object] | None = None,
    mean_increment: dict[str, object] | None = None,
    preserve_global_mean_pressure: bool = True,
) -> dict[str, object]:
    """Recentre the members on the control analysis.

    ``mode`` ``"increment"`` (the default, :data:`RECENTERING_MODES`):
    the ensemble-mean increment of the analysis (``mean_increment``, the
    spectral increment of the members' mean at the ensemble truncation) is
    replaced by the control's increment truncated to the ensemble triangle
    (``control_increment``, what :func:`apply_control_increment` handed
    back as the tapered increment), so the members keep their own
    terrain-consistent background and receive the control's analysis
    increment; both arrays are required in this mode.  ``mode``
    ``"state"``: the control spectral fields truncated to the ensemble
    truncation replace the ensemble mean (the pre-measurement form; on a
    control whose orography is finer than the ensemble's the truncated
    state carries the finer terrain's surface pressure onto the coarser
    grid).  ``fraction`` 1 applies the whole shift, a value between 0 and
    1 the partial recentring.  Each member keeps its own perturbation
    about the mean; surface, physics and grid tracers untouched; the vapor
    repaired per member.  With ``preserve_global_mean_pressure`` (the
    default) each member's global-mean surface pressure is kept across the
    shift by the same rule every increment application uses (the constant
    added to ln ps is recorded): the control's increment reaches the
    members BEFORE the control's own mass rule was applied to it, so a
    shift taken raw moved every member's global-mean surface pressure by
    the raw increment's mean, 10 to 22 Pa per analysis on the T63 / T127
    twin, and the members' conservation epoch then held the moved value.
    With increments pending (IAU) the shift is folded into the pending
    increments so the window applies it, the mass constant folded with it.
    Returns a record (the mean shift per field, the mass offsets and the
    members' global-mean surface pressure before and after)."""
    fraction = 1.0 if fraction is None else float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("recentering fraction must lie in [0, 1]")
    if mode not in ("increment", "state"):
        raise ValueError("recentering mode must be increment or state")
    transform = ensemble.transform
    model = ensemble.model
    backend = transform.backend
    xp = backend.xp
    ens_t = int(transform.truncation)
    lnps_index = SPECTRAL_FIELDS.index("log_surface_pressure")
    mean = ensemble.mean_spectral(include_pending=True)
    if mode == "increment":
        if control_increment is None or mean_increment is None:
            raise ValueError(
                "increment recentring needs the control increment and the ensemble-mean "
                "increment of the analysis (EnsembleAnalysis.control_increment_spectral and "
                ".mean_increment_spectral); pass mode='state' to recentre on the truncated state"
            )
        shift = {}
        for name in SPECTRAL_FIELDS:
            ctl = control_increment[name]
            ctl = xp.asarray(ctl.get() if hasattr(ctl, "get") else ctl, dtype=backend.complex_dtype)
            ens = mean_increment[name]
            ens = xp.asarray(ens.get() if hasattr(ens, "get") else ens, dtype=backend.complex_dtype)
            shift[name] = (ctl - ens) * fraction
    else:
        target = {}
        for name in SPECTRAL_FIELDS:
            coeff = getattr(deterministic_analysis.atmosphere, name)
            host = coeff.get() if hasattr(coeff, "get") else np.asarray(coeff)
            target[name] = xp.asarray(truncate_spectral(host, ens_t), dtype=backend.complex_dtype)
        shift = {name: (target[name] - mean[name]) * fraction for name in SPECTRAL_FIELDS}
    offsets: list[float] = []
    ps_before: list[float] = []
    ps_after: list[float] = []
    if ensemble.pending_increments is not None:
        remaining = (ensemble.pending_steps_left / ensemble.pending_steps_total
                     if ensemble.pending_steps_total else 1.0)
        for k, pending in enumerate(ensemble.pending_increments):
            member_shift = dict(shift)
            # The member as the window will leave it, with and without the
            # shift: the constant that keeps its global-mean surface pressure
            # rides in the pending increment beside the shift.
            held = ensemble.members[k].atmosphere.log_surface_pressure + pending["log_surface_pressure"] * remaining
            ps_before.append(_member_global_mean_ps(transform, held))
            if preserve_global_mean_pressure:
                offset = _mass_offset(transform, held, held + shift["log_surface_pressure"])
                member_shift["log_surface_pressure"] = transform.add_grid_constant(
                    shift["log_surface_pressure"], offset)
                offsets.append(offset)
            ps_after.append(_member_global_mean_ps(transform, held + member_shift["log_surface_pressure"]))
            for name in SPECTRAL_FIELDS:
                pending[name] = pending[name] + member_shift[name]
    else:
        for k, member in enumerate(ensemble.members):
            fields = [getattr(member.atmosphere, name) + shift[name] for name in SPECTRAL_FIELDS]
            ps_before.append(_member_global_mean_ps(transform, member.atmosphere.log_surface_pressure))
            if preserve_global_mean_pressure:
                offset = _mass_offset(transform, member.atmosphere.log_surface_pressure, fields[lnps_index])
                fields[lnps_index] = transform.add_grid_constant(fields[lnps_index], offset)
                offsets.append(offset)
            # Recorded in both forms, so a raw shift's drift is a number in
            # the receipt and not an inference from the score.
            ps_after.append(_member_global_mean_ps(transform, fields[lnps_index]))
            state = ArwenGlobalState(member.atmosphere.with_fields(fields), member.surface, member.physics_state)
            state, _n, _t, _f = model._repair_positivity(state)
            model.enforce(state)
            ensemble.members[k] = state
            model.release_syntheses()
        ensemble.open_epochs()
    # The members' surface pressure after the shift (the state the window
    # integrates under direct insertion); under IAU the window adds the
    # portions step by step and the apply record's reading stands.
    after_shift = (members_surface_pressure_record(
        [member_surface_pressure_extrema(model, transform, m) for m in ensemble.members])
        if ensemble.pending_increments is None else
        {"members_surface_pressure_pa": None, "members_above_radiation_ceiling": []})
    return {
        **after_shift,
        "route": (
            (f"the ensemble-mean increment is replaced by the control's increment at T{ens_t} (fraction "
             f"{fraction:g}); the members keep their terrain-consistent background" if mode == "increment" else
             f"the control analysis truncated to T{ens_t} replaces the ensemble mean (fraction {fraction:g})")
            + "; each member keeps its perturbation; surface, physics and grid tracers untouched; vapor repaired"
            + ("; each member's global-mean surface pressure kept across the shift"
               if preserve_global_mean_pressure else
               "; the shift taken raw, the members' global-mean surface pressure moving with its mean")
            + ("; folded into the pending IAU increments" if ensemble.pending_increments is not None else "")
        ),
        "mode": mode,
        "fraction": fraction,
        "mean_shift_grid_rms": {
            "theta": _grid_rms_of_spectral(transform, shift["theta"]),
            "ln_surface_pressure": _grid_rms_of_spectral(transform, shift["log_surface_pressure"]),
            "qv": _grid_rms_of_spectral(transform, shift["qv"]),
        },
        "preserve_global_mean_pressure": bool(preserve_global_mean_pressure),
        "mass_preserving_log_offset": {
            "mean": float(np.mean(offsets)) if offsets else 0.0,
            "maxabs": float(np.max(np.abs(offsets))) if offsets else 0.0,
            "raw_shift_global_mean_ln_ps": float(
                (shift["log_surface_pressure"][0, 0].get() if hasattr(shift["log_surface_pressure"], "get")
                 else shift["log_surface_pressure"][0, 0]).real / math.sqrt(4.0 * math.pi)),
        },
        "members_global_mean_surface_pressure_pa": {
            "before": float(np.mean(ps_before)) if ps_before else None,
            "after": float(np.mean(ps_after)) if ps_after else None,
        },
    }


# ---------------------------------------------------------------------------
# The receipt: distributions, Desroziers, the four assessments
# ---------------------------------------------------------------------------

def desroziers(d_ob: np.ndarray, d_oa: np.ndarray, error: np.ndarray, spread_h: np.ndarray) -> dict[str, object]:
    """The Desroziers consistency ratios of one (stream, variable) sample:
    ``E[d_oa d_ob] / mean(sigma_o^2)`` (one when the assigned observation
    error is consistent), ``E[d_ob (d_ob - d_oa)] / mean(spread_H^2)``
    (one when the background spread in observation space is consistent),
    and the innovation ratio ``E[d_ob^2] / (mean(spread_H^2) + mean(sigma_o^2))``.
    Assumptions: :data:`DESROZIERS_ASSUMPTIONS`."""
    d_ob = np.asarray(d_ob, dtype=np.float64)
    d_oa = np.asarray(d_oa, dtype=np.float64)
    err2 = float(np.mean(np.asarray(error, dtype=np.float64) ** 2))
    sp2 = float(np.mean(np.asarray(spread_h, dtype=np.float64) ** 2))
    n = int(d_ob.size)
    if n == 0:
        return {"count": 0}
    r_hat = float(np.mean(d_oa * d_ob))
    hbht_hat = float(np.mean(d_ob * (d_ob - d_oa)))
    innov2 = float(np.mean(d_ob ** 2))
    # The background error variance in observation space the innovations
    # leave once the estimated observation error is taken out (the same
    # identity read from E[d_ob^2] = HBH^T + R); the spread ratio is the
    # ensemble spread against its root, the calibrated-spread criterion.
    background_var = innov2 - r_hat
    spread_ratio = (float(math.sqrt(sp2 / background_var)) if background_var > 0.0 and sp2 > 0.0 else None)
    return {
        "count": n,
        "assigned_error_variance": err2,
        "estimated_error_variance": r_hat,
        "error_variance_ratio": float(r_hat / err2) if err2 > 0.0 else None,
        "background_spread_variance_h": sp2,
        "estimated_hbht": hbht_hat,
        "background_variance_ratio": float(hbht_hat / sp2) if sp2 > 0.0 else None,
        "innovation_variance": innov2,
        "innovation_ratio": float(innov2 / (sp2 + err2)) if (sp2 + err2) > 0.0 else None,
        "background_error_variance_from_innovations": background_var,
        "spread_ratio": spread_ratio,
        "assumptions": DESROZIERS_ASSUMPTIONS,
    }


def _stream_report(assimilated: list[PointObs], withheld: list[PointObs],
                   background_hx: list[np.ndarray], background_hx_withheld: list[np.ndarray],
                   analysis_hx: list, analysis_hx_withheld: list, options: FilterOptions,
                   control_hx_b: list, control_hx_a: list, control_hx_b_withheld: list,
                   control_hx_a_withheld: list):
    """``streams[stream][variable]`` with per-region O-B and O-A
    distributions on the assimilated and the withheld rows (the ensemble
    mean and, when formed, the control), the Desroziers ratios and the
    withheld diagnostic."""
    streams: dict[str, dict] = {}
    unjudged: list[str] = []
    for idx, (used, held, hb, hbw, ha, haw) in enumerate(zip(
            assimilated, withheld, background_hx, background_hx_withheld, analysis_hx, analysis_hx_withheld)):
        cb = control_hx_b[idx] if control_hx_b else None
        ca = control_hx_a[idx] if control_hx_a else None
        cbw = control_hx_b_withheld[idx] if control_hx_b_withheld else None
        caw = control_hx_a_withheld[idx] if control_hx_a_withheld else None
        entry: dict[str, object] = {"units": VARIABLE_TABLE.get(used.variable, {}).get("units", "")}
        regions: dict[str, object] = {}
        for region in REGIONS:
            m_used = _region_mask(used.latitude_deg, used.longitude_deg, region)
            m_held = _region_mask(held.latitude_deg, held.longitude_deg, region)
            row = {
                "assimilated": {
                    "count": int(m_used.sum()),
                    "o_minus_b": _statistics(used.value[m_used] - hb[m_used]),
                    "o_minus_a": None if ha is None else _statistics(used.value[m_used] - ha[m_used]),
                },
                "withheld": {
                    "count": int(m_held.sum()),
                    "o_minus_b": _statistics(held.value[m_held] - hbw[m_held]),
                    "o_minus_a": None if haw is None else _statistics(held.value[m_held] - haw[m_held]),
                },
            }
            if cb is not None:
                row["control"] = {
                    "assimilated": {
                        "o_minus_b": _statistics(used.value[m_used] - cb[m_used]),
                        "o_minus_a": None if ca is None else _statistics(used.value[m_used] - ca[m_used]),
                    },
                    "withheld": {
                        "o_minus_b": _statistics(held.value[m_held] - cbw[m_held]) if cbw is not None else None,
                        "o_minus_a": (None if caw is None or cbw is None
                                      else _statistics(held.value[m_held] - caw[m_held])),
                    },
                }
            if row["assimilated"]["count"] or row["withheld"]["count"]:
                regions[region] = row
        entry["regions"] = regions
        entry["count"] = used.count
        entry["withheld_count"] = held.count
        entry["rejections"] = dict(used.rejections)
        label = f"{used.stream}/{used.variable}"
        if ha is None:
            entry["verdict"] = "UNJUDGED: no operator re-evaluated this stream on the analysis"
            entry["desroziers"] = {"count": 0}
            unjudged.append(label)
        else:
            entry["desroziers"] = desroziers(used.value - hb, used.value - ha, used.error, used.spread())
            ob = _statistics(used.value - hb)["rms"]
            oa = _statistics(used.value - ha)["rms"]
            entry["verdict"] = (
                f"assimilated O-B rms {ob:.4g}, O-A rms {oa:.4g} (ensemble mean); a stream need not "
                "move closer, the assessments judge consistency"
            )
        gated = used.count + held.count >= options.gate_minimum_count and held.count > 0
        entry["gated"] = bool(gated)
        if gated and haw is not None:
            ob = _statistics(held.value - hbw)["rms"]
            oa = _statistics(held.value - haw)["rms"]
            entry["withheld_o_minus_a_below_o_minus_b"] = bool(oa < ob)
            entry["gate_passed"] = bool(oa < ob)
        streams.setdefault(used.stream, {})[used.variable] = entry
    return streams, unjudged


def _assessments(streams: dict, unjudged: list[str], engineering: dict, apply_record: dict,
                 control_record: dict | None, options: FilterOptions, spread_before: dict,
                 spread_after: dict) -> dict[str, object]:
    """The four assessments of amendment G."""
    # 1. engineering validity: everything ran and every stream was judged.
    eng_failures = [k for k, v in engineering.items() if v is not True]
    if unjudged:
        eng_failures.append(f"unjudged streams: {unjudged}")
    engineering_row = {
        "verdict": "pass" if not eng_failures else "fail",
        "checks": engineering,
        "failures": eng_failures,
        "rule": GATE_RULE,
    }
    # 2. statistical consistency: Desroziers ratios inside [0.5, 2], and
    #    the calibrated-spread criterion per cell (the ensemble spread in
    #    observation space over the background error the innovations
    #    leave, inside options.spread_ratio_band).
    flagged = []
    judged = 0
    spread_cells: dict[str, object] = {}
    spread_flagged = []
    lo, hi = options.spread_ratio_band
    for stream, variables in streams.items():
        for variable, entry in variables.items():
            d = entry.get("desroziers", {})
            if d.get("count", 0) < options.desroziers_minimum_count:
                continue
            judged += 1
            for key in ("error_variance_ratio", "innovation_ratio"):
                ratio = d.get(key)
                if ratio is None or not (0.5 <= ratio <= 2.0):
                    flagged.append({"stream": stream, "variable": variable, "ratio": key, "value": ratio})
            ratio = d.get("spread_ratio")
            spread_cells[f"{stream}/{variable}"] = ratio
            if ratio is None or not (lo <= ratio <= hi):
                spread_flagged.append({"stream": stream, "variable": variable, "spread_ratio": ratio})
    statistical_row = {
        "verdict": "not_assessed" if judged == 0 else ("consistent" if not flagged else "flagged"),
        "band": [0.5, 2.0],
        "judged_streams": judged,
        "flagged": flagged,
        "assumptions": DESROZIERS_ASSUMPTIONS,
        "spread_before": spread_before,
        "spread_after": spread_after,
        "spread_calibration": {
            "rule": (
                "the ensemble spread in observation space over the root of the background error "
                "variance the innovations leave once the Desroziers observation error is taken out, "
                "per stream and variable; inside the band the spread is calibrated, below it the "
                "ensemble is under-dispersive there, above it over-dispersive"
            ),
            "band": [lo, hi],
            "cells": spread_cells,
            "flagged": spread_flagged,
            "verdict": ("not_assessed" if not spread_cells else
                        ("calibrated" if not spread_flagged else "flagged")),
        },
    }
    # 3. physical consistency: mass, balance, moisture.
    physical_flags = []
    offsets = [apply_record["mass_preserving_log_offset"]["maxabs"]]
    if control_record is not None:
        offsets.append(abs(float(control_record["mass_preserving_log_offset"])))
    if max(offsets) > 1.0e-4:
        physical_flags.append({"mass_preserving_log_offset_maxabs": max(offsets), "bound": 1.0e-4})
    above_ceiling = apply_record.get("members_above_radiation_ceiling") or []
    if above_ceiling:
        physical_flags.append({
            "members_above_radiation_ceiling": [int(m["member"]) for m in above_ceiling],
            "surface_pressure_max_pa": max(float(m["surface_pressure_max_pa"]) for m in above_ceiling),
            "ceiling_pa": float(RADIATION_SURFACE_PRESSURE_CEILING_PA),
            "breakage": MEMBER_CEILING_BREAKAGE,
        })
    div_frac = apply_record["wind_balance"].get("divergent_fraction_applied_mean")
    physical_row = {
        "verdict": "within_stated_bounds" if not physical_flags else "flagged",
        "global_mass_budget": {
            "rule": "the global-mean surface pressure is kept per state; the log offset the rule "
                    "removed is recorded and bounded at 1e-4 (0.01 percent of the mean)",
            "log_offset_maxabs": max(offsets),
        },
        "wind_increment_balance": {
            "mode": options.wind_balance,
            "divergent_fraction_analysed_mean": apply_record["wind_balance"].get("divergent_fraction_analysed_mean"),
            "divergent_fraction_applied_mean": div_frac,
            "rule": "the divergent share of the increment's kinetic energy is recorded; no blanket "
                    "mid-latitude balance is imposed by latitude (the rotational mode removes the "
                    "divergent part everywhere and is itself under the OSSE's verdict)",
        },
        "moisture_bounds": {
            "rule": "the model's positivity repair runs after every increment; the largest negative "
                    "vapor it met is recorded per state",
            "control": None if control_record is None else control_record["positivity_repair"],
        },
        "control_increment_spectrum": (
            None if control_record is None else {
                name: {"small_scale_share_above_0p6T": control_record["spectrum_after_taper"][name]
                       ["small_scale_share_above_0p6T"]}
                for name in ("temperature_k2", "ln_surface_pressure2", "wind_ke_j_kg")}
        ),
        "flags": physical_flags,
    }
    predictive_row = {
        "verdict": "not_assessed_in_receipt",
        "rule": "predictive value is the forecast's skill against withheld or independent "
                "verification: the observation scorecard (stations at 18Z and 00Z, soundings at "
                "12Z and 00Z) on the forecasts from this analysis, matched samples, never the "
                "analysis receipt",
    }
    return {
        "engineering_validity": engineering_row,
        "statistical_consistency": statistical_row,
        "physical_consistency": physical_row,
        "predictive_value": predictive_row,
    }


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------

def analyze_ensemble(
    ensemble: GlobalEnsemble,
    batches: list[PointObs],
    options: FilterOptions | None = None,
    *,
    analysis_time=None,
    background: dict | None = None,
    additive_inflation: bool = True,
    control: ControlBackground | None = None,
) -> EnsembleAnalysis:
    """One LETKF analysis of ``ensemble`` against ``batches`` at
    ``analysis_time`` (UTC; ``None`` takes the latest report).  Batches
    whose ``simulated`` is None are evaluated with the filter's own
    point operators when their variable is in the neutral vocabulary and
    refused by name otherwise.  With ``control`` given, the control
    background's H(x_H^b) is evaluated the same way (or must arrive as
    ``control_simulated``) and the control analysis is formed from its
    own innovation and handed back as ``control_analysis``.
    ``background`` is the identity record the door has for the background
    (member checkpoint hashes, step, time); it rides into the report.
    ``additive_inflation`` False skips the additive draw (the OSSE's
    calibration families measure the analysis alone)."""
    options = options or FilterOptions()
    clock = _Stopwatch()
    transform = ensemble.transform
    model = ensemble.model
    backend = transform.backend
    xp = backend.xp
    cfg = ensemble.cfg
    moment = _resolve_time(analysis_time, batches)
    operators = MemberOperators.for_model(model, transform, cfg, precision=options.operator_precision)
    engineering: dict[str, object] = {}

    # 1. operators on the background (ensemble members and the control),
    #    ONE evaluation per state set over every neutral batch's rows.
    offered = []
    foreign_refused = []
    control_refused = []
    need_members = []
    need_control = []
    for batch in batches:
        if batch.count == 0:
            continue
        neutral = batch.variable in OPERATOR_VARIABLES
        if batch.simulated is None:
            if not neutral:
                foreign_refused.append(f"{batch.stream}/{batch.variable}")
                continue
            need_members.append(batch)
        if neutral:
            # The package's own operators answer for the neutral vocabulary
            # on the analysed members (a window may have built the batch
            # with another grid's operators; a foreign stream keeps its own).
            batch.operator = operators.batch_operator
        if control is not None and batch.control_simulated is None:
            if not neutral:
                control_refused.append(f"{batch.stream}/{batch.variable}")
                continue
            need_control.append(batch)
        offered.append(batch)
    if need_members:
        evaluate_batches(operators, ensemble.members, need_members, target="simulated")
    if need_control:
        evaluate_batches(control.operators, [control.state], need_control, target="control_simulated")
    if foreign_refused:
        raise ValueError(
            f"batches {foreign_refused} carry a variable outside the neutral vocabulary "
            f"and no simulated H(x_k): {ENSEMBLE_REJECTION_BREAKAGE['foreign_variable_unsimulated']}"
        )
    if control_refused:
        raise ValueError(
            f"batches {control_refused} carry a variable outside the neutral vocabulary and no "
            f"control_simulated H(x_H^b): {ENSEMBLE_REJECTION_BREAKAGE['foreign_variable_no_control']}"
        )
    engineering["ingest"] = bool(offered)
    clock.lap("operators_background_s")

    # 2. the observation-error calibration, then quality control and 3. the
    #    withheld split (diagnostic).
    chain = _chain_from_background(ensemble.provenance)
    assimilated: list[PointObs] = []
    withheld: list[PointObs] = []
    rejections: dict[str, dict[str, int]] = {}
    error_calibration: dict[str, dict[str, object]] = {}
    for index, batch in enumerate(offered):
        assigned = np.asarray(batch.error, dtype=np.float64)
        errors, table_entry = calibrated_error(
            options.observation_error_calibration, batch.stream, batch.variable, assigned)
        error_calibration.setdefault(batch.stream, {})[batch.variable] = {
            "assigned_error_rms": float(math.sqrt(np.mean(assigned ** 2))) if assigned.size else None,
            # a constant cell (the table's value in the variable's units) or a
            # scale cell (the factor over each row's own assigned error, the
            # profile-error streams); the calibrated rms says what the filter saw
            "calibrated_error": table_entry if isinstance(table_entry, float) else None,
            "calibrated_scale": table_entry["scale"] if isinstance(table_entry, dict) else None,
            "calibrated_error_rms": (float(math.sqrt(np.mean(errors ** 2)))
                                     if table_entry is not None and errors.size else None),
            "applied": table_entry is not None,
        }
        if table_entry is not None:
            batch.error = errors
        kept, rej = _quality_control(batch, ensemble, options, moment, chain)
        if rej:
            rejections.setdefault(batch.stream, {})
            for name, count in rej.items():
                rejections[batch.stream][name] = rejections[batch.stream].get(name, 0) + count
        if kept.count == 0:
            continue
        used, held = _withhold(kept, options, index)
        assimilated.append(used)
        withheld.append(held)
    if not assimilated:
        raise ValueError(
            "every offered report was rejected by quality control or the chain; an "
            "empty analysis would be the background wearing a new hash"
        )
    engineering["quality_control"] = True
    clock.lap("quality_control_s")

    # 3b. the hybrid's static draws: the control's gain takes the static
    #     covariance's share when beta is below one and there is a control
    #     to analyse (the members keep the ensemble transform either way).
    static_prior = None
    hybrid_record: dict[str, object] = {"beta": float(options.hybrid_beta), "applied": False}
    if float(options.hybrid_beta) < 1.0:
        if control is None:
            hybrid_record["reason"] = (
                "no control background: the hybrid gain is the control's; the members' own "
                "analysis is the ensemble transform"
            )
        else:
            static_prior, hybrid_record = _static_draws(ensemble, assimilated, operators, options)
            hybrid_record["applied"] = True
    clock.lap("static_draws_s")

    # 4. the LETKF on the grid fields.
    prior, geometry = _grid_prior(ensemble, options)
    spread_before = ensemble.spread()
    hcut_m = float(options.horizontal_cutoff_km) * 1000.0

    def vertical_cutoff_for(batch, surface):
        if batch.vertical_cutoff_lnp is not None:
            return np.full(batch.count, float(batch.vertical_cutoff_lnp))
        out = np.empty(batch.count)
        for is_surface in (True, False):
            cut = options.vertical_cutoff_for(batch.variable, is_surface)
            out[surface == is_surface] = math.inf if cut is None else float(cut)
        return out

    flat = flatten_batches(
        assimilated, xp, horizontal_cutoff_m=hcut_m,
        vertical_cutoff_for=vertical_cutoff_for, solve_dtype=options.solve_dtype,
        control=control is not None, static=static_prior is not None,
    )
    # The rows localised through a profile (the radiances): which streams,
    # how many rows, and where each stream's profile peaks and holds half
    # its weight, so the receipt says where a channel was allowed to act.
    profile_record = {}
    for b in assimilated:
        if b.localisation_profile is None or b.count == 0:
            continue
        prof = np.asarray(b.localisation_profile, dtype=np.float64).mean(axis=0)
        above_half = np.flatnonzero(prof >= 0.5)
        profile_record[f"{b.stream}/{b.variable}"] = {
            "rows": int(b.count),
            "cutoff_lnp": None if b.vertical_cutoff_lnp is None else float(b.vertical_cutoff_lnp),
            "peak_pressure_pa": float(math.exp(LOCALISATION_AXIS_LNP[int(prof.argmax())])),
            "half_weight_pressure_pa": [float(math.exp(LOCALISATION_AXIS_LNP[above_half[0]])),
                                        float(math.exp(LOCALISATION_AXIS_LNP[above_half[-1]]))] if above_half.size else None,
            "rule": "the row's vertical weight at a level is its profile read at the level's ln p (the weighting "
                    "function convolved with Gaspari-Cohn of the cutoff), not Gaspari-Cohn on a centroid",
        }
    letkf_config = PointLetkfConfig(
        rtps_alpha=options.rtps_alpha, prior_inflation=options.prior_inflation,
        relaxation=options.relaxation, max_local_obs=options.max_local_obs,
        chunk_rings=options.chunk_columns, memory_budget_mib=options.memory_budget_mib,
        solve_dtype=options.solve_dtype, eigensolver=options.eigensolver,
    )
    diagnostics = PointLetkfDiagnostics()
    solved = solve_on_path(prior, flat, geometry, letkf_config, diagnostics,
                           solve_path=options.solve_path, xp=xp, to_numpy=backend.to_numpy,
                           static_prior=static_prior,
                           hybrid_beta=float(options.hybrid_beta) if static_prior is not None else 1.0)
    increments = solved.increments
    control_increment_grid = solved.control_increment
    del prior, flat, static_prior
    engineering["localised_solve"] = True
    clock.lap("letkf_s")

    # 5. the increments into the members (direct or pending).
    background_hx = [b.simulated.mean(axis=0) for b in assimilated]
    background_hx_withheld = [
        (b.simulated.mean(axis=0) if b.count else np.zeros(0)) for b in withheld]
    control_hx_b = [b.control_simulated[0] for b in assimilated] if control is not None else []
    control_hx_b_withheld = [
        (b.control_simulated[0] if b.count else np.zeros(0)) for b in withheld] if control is not None else []
    mean_increment, apply_record, analysed_members = _apply_increments(ensemble, increments, options)
    del increments
    engineering["increment_application"] = True
    clock.lap("apply_s")

    # 6. the control analysis.
    control_analysis = None
    control_record = None
    control_increment_spectral = None
    if control is not None:
        control_analysis, control_record, control_increment_spectral = apply_control_increment(
            control, {name: control_increment_grid[name] for name in options.analysis_fields},
            ensemble, options)
        engineering["control_analysis"] = True
    clock.lap("control_s")

    # 7. O-A through the same operators on the analysed states: one
    #    evaluation over every neutral batch (assimilated and withheld
    #    together) per state set; a foreign batch through its own operator.
    both = list(assimilated) + list(withheld)
    neutral_hx = evaluate_batches(operators, analysed_members, both, target=None)
    analysis_hx = []
    analysis_hx_withheld = []
    n_used = len(assimilated)
    for i, (used, held) in enumerate(zip(assimilated, withheld)):
        hx_used = neutral_hx[i]
        hx_held = neutral_hx[n_used + i]
        if hx_used is None:
            if used.operator is None:
                analysis_hx.append(None)
                analysis_hx_withheld.append(None)
                continue
            hx_used = np.asarray(used.operator(analysed_members, used))
            hx_held = np.asarray(held.operator(analysed_members, held)) if held.count else np.zeros((1, 0))
        analysis_hx.append(np.asarray(hx_used).mean(axis=0))
        analysis_hx_withheld.append(np.asarray(hx_held).mean(axis=0) if held.count else np.zeros(0))
    control_hx_a: list = []
    control_hx_a_withheld: list = []
    if control is not None:
        ctl_hx = evaluate_batches(control.operators, [control_analysis], both, target=None)
        for i, (used, held) in enumerate(zip(assimilated, withheld)):
            hx_used = ctl_hx[i]
            hx_held = ctl_hx[n_used + i]
            control_hx_a.append(None if hx_used is None else np.asarray(hx_used)[0])
            control_hx_a_withheld.append(
                None if hx_used is None else (np.asarray(hx_held)[0] if held.count else np.zeros(0)))
    spread_after = ensemble.spread(states=analysed_members)
    engineering["operators_on_analysis"] = all(h is not None for h in analysis_hx)
    clock.lap("operators_analysis_s")

    streams, unjudged = _stream_report(
        assimilated, withheld, background_hx, background_hx_withheld,
        analysis_hx, analysis_hx_withheld, options,
        control_hx_b, control_hx_a, control_hx_b_withheld, control_hx_a_withheld)
    engineering["state_enforced"] = True
    assessments = _assessments(streams, unjudged, engineering, apply_record, control_record, options,
                               spread_before, spread_after)
    status = "pass" if assessments["engineering_validity"]["verdict"] == "pass" else "fail"

    # The chain: the ensemble's own record plus this cycle's assimilated rows.
    horizon = moment - dt.timedelta(seconds=float(options.maximum_age_s))
    reports = {
        key: instant for key, instant in chain.get("reports", {}).items()
        if (parse_valid_time(str(instant)) or moment) >= horizon
    }
    for used in assimilated:
        if used.valid_time is None:
            continue
        for ident, when in zip(used.identity, used.valid_time):
            if ident:
                reports[str(ident)] = when.astimezone(dt.timezone.utc).isoformat(timespec="seconds")
    cycles_record = list(chain.get("cycles", []))
    cycles_record.append({
        "analysis_time_utc": moment.isoformat(timespec="seconds"),
        "step": int(ensemble.step),
        "assimilated": int(sum(b.count for b in assimilated)),
        "withheld": int(sum(b.count for b in withheld)),
        "status": status,
        "control": control is not None,
    })
    ensemble.provenance[ASSIMILATION_HISTORY_KEY] = {
        "schema": ASSIMILATION_HISTORY_SCHEMA, "reports": reports, "cycles": cycles_record,
    }
    for member in ensemble.members:
        member.physics_state.metadata[ASSIMILATION_HISTORY_KEY] = ensemble.provenance[ASSIMILATION_HISTORY_KEY]
    if control_analysis is not None:
        control_analysis.physics_state.metadata[ASSIMILATION_HISTORY_KEY] = ensemble.provenance[ASSIMILATION_HISTORY_KEY]
    ensemble.cycles += 1
    if status == "pass" and ensemble.pending_increments is None:
        ensemble.open_epochs()

    # 8. additive inflation, after the receipt's O-A pass.
    inflation_record = {"additive_fraction": float(ensemble.options.additive_inflation_fraction), "applied": False}
    if additive_inflation and ensemble.options.additive_inflation_fraction > 0.0:
        for k, member in enumerate(ensemble.members):
            rng = member_rng(ensemble.options.seed, k, "additive", ensemble.cycles)
            inc, _record = draw_perturbation(
                model, transform, member.atmosphere, ensemble.options, rng,
                amplitude_scale=float(ensemble.options.additive_inflation_fraction))
            ensemble.members[k] = perturbed_state(model, transform, member, inc)
            ensemble.members[k].atmosphere.time_s = member.atmosphere.time_s
            ensemble.members[k].atmosphere.step = member.atmosphere.step
            model.release_syntheses()
        inflation_record["applied"] = True
        inflation_record["spread_after_additive"] = ensemble.spread()
    clock.lap("additive_inflation_s")

    exner_mean = _mean_exner(model, transform, ensemble.members[0].atmosphere)
    report = {
        "schema": ANALYSIS_SCHEMA,
        "acknowledgement": RESEARCH_ACKNOWLEDGEMENT,
        "name": cfg.name,
        "config_hash": cfg.config_hash,
        "pins_hash": pins_hash(cfg.semi_implicit_scheme, cfg.integrator),
        "analysis_time_utc": moment.isoformat(timespec="seconds"),
        "background": dict(background or {}) | {"step": int(ensemble.step), "time_s": float(ensemble.time_s)},
        "members": int(ensemble.size),
        "truncation": int(transform.truncation),
        "control_truncation": None if control is None else control.truncation,
        "options": {"filter": options.identity(), "ensemble": ensemble.options.identity()},
        "streams": streams,
        "assimilated_total": int(sum(b.count for b in assimilated)),
        "withheld_total": int(sum(b.count for b in withheld)),
        "rejections": rejections,
        "rejection_breakage": ENSEMBLE_REJECTION_BREAKAGE,
        "observation_error_calibration": {
            "name": options.observation_error_calibration,
            "streams": error_calibration,
        },
        "localisation": {
            "horizontal_cutoff_km": float(options.horizontal_cutoff_km),
            "vertical_cutoff_lnp": float(options.vertical_cutoff_lnp),
            "aloft_wind_vertical_cutoff_lnp": float(options.aloft_wind_vertical_cutoff_lnp),
            "aloft_humidity_vertical_cutoff_lnp": float(options.aloft_humidity_vertical_cutoff_lnp),
            "surface_vertical_cutoff_lnp": float(options.surface_vertical_cutoff_lnp),
            "surface_wind_vertical_cutoff_lnp": float(options.surface_wind_vertical_cutoff_lnp),
            "pressure_vertical_cutoff_lnp": options.pressure_vertical_cutoff_lnp,
            "function": "Gaspari-Cohn, cutoff is the zero (2c), one cutoff per report class; a radiance row reads its profile",
            "profile_rows": profile_record,
        },
        "inflation": {
            "relaxation": options.relaxation, "rtps_alpha": float(options.rtps_alpha),
            "prior_inflation": float(options.prior_inflation), **inflation_record,
        },
        "letkf": {
            "path": diagnostics.path,
            "wall_seconds": diagnostics.wall_seconds,
            "eigensolver": diagnostics.eigensolver,
            "level_batches": diagnostics.level_batches,
            "max_batch_points": diagnostics.max_batch_points,
            "active_columns": diagnostics.active_columns,
            "total_columns": diagnostics.total_columns,
            "active_points": diagnostics.active_points,
            "total_points": diagnostics.total_points,
            "max_local_obs": diagnostics.max_local_obs,
            "max_padded_slots": diagnostics.max_padded_slots,
            "dropped_by_cap": diagnostics.dropped_by_cap,
            "chunks": diagnostics.chunks,
            "chunk_rings": diagnostics.chunk_rings,
            "chunk_segments": diagnostics.chunk_segments,
            "chunk_oom_shrinks": diagnostics.chunk_oom_shrinks,
            "max_jacobi_sweeps": diagnostics.max_jacobi_sweeps,
            "hybrid_beta": diagnostics.hybrid_beta,
            "static_samples": diagnostics.static_samples,
            "hybrid_solver": diagnostics.hybrid_solver,
            "zero_spread_fields": list(diagnostics.zero_spread_fields),
            "seconds": {
                "setup": diagnostics.setup_seconds, "gather": diagnostics.gather_seconds,
                "solve": diagnostics.solve_seconds, "finish": diagnostics.finish_seconds,
                "hybrid_solve": diagnostics.hybrid_solve_seconds,
            },
            "grid_increment_rms": diagnostics.mean_increment_rms,
            "grid_control_increment_rms": diagnostics.control_increment_rms,
            "grid_prior_spread": diagnostics.prior_spread,
            "grid_posterior_spread": diagnostics.posterior_spread,
        },
        "increment": {
            **apply_record,
            "mean_increment": _increment_summary(model, transform, ensemble.members[0].atmosphere, mean_increment),
            "mean_increment_spectrum": increment_spectrum_record(
                mean_increment, float(transform.grid.radius_m), int(transform.truncation), exner_mean),
            "control": control_record,
        },
        "operators": operators.precision_record,
        "spread": {"before": spread_before, "after": spread_after},
        "hybrid": hybrid_record,
        "assessments": assessments,
        "gate_of_record": {
            "rule": GATE_RULE,
            "failed": list(assessments["engineering_validity"]["failures"]),
            "incomplete": list(unjudged),
            "passed": status == "pass",
        },
        "status": status,
        "timings_s": dict(clock.readings),
    }
    return EnsembleAnalysis(ensemble=ensemble, mean_increment_spectral=mean_increment,
                            report=report, control_analysis=control_analysis,
                            control_increment_spectral=control_increment_spectral,
                            control_record=control_record, timings_s=dict(clock.readings))


__all__ = [
    "ANALYSIS_SCHEMA",
    "BALANCE_RULE",
    "DESROZIERS_ASSUMPTIONS",
    "ENSEMBLE_REJECTION_BREAKAGE",
    "GATE_RULE",
    "MEMBER_CEILING_BREAKAGE",
    "RADIATION_SURFACE_PRESSURE_CEILING_PA",
    "REGIONS",
    "SPECTRAL_BANDS",
    "ControlBackground",
    "EnsembleAnalysis",
    "analyze_ensemble",
    "apply_control_increment",
    "apply_mean_increment",
    "apply_taper",
    "band_summary",
    "desroziers",
    "grid_increment_to_spectral",
    "increment_balance_record",
    "increment_spectrum_record",
    "member_surface_pressure_extrema",
    "members_surface_pressure_record",
    "recenter",
    "solve_on_path",
    "spectral_power_by_degree",
    "taper_weights",
    "wind_power_by_degree",
]
