"""The global observing-system simulation experiment: the gate on the
WOOF global ensemble filter, its calibration families and its sweeps.

:mod:`woof.da.osse` gates the regional filter on a model-free twin.  The
global filter is gated on the model itself, in the dual-resolution shape
the design puts it in: a NATURE run of the configured core at the CONTROL
truncation (the truth, integrated from the configured initial state
through a spin-up and then across the cycles), a CONTROL started
DISPLACED from the truth (the nature state at the first analysis time
plus a perturbation draw of ``start_perturbation_scale`` times the
initial-perturbation amplitude, from its own seed) and an ENSEMBLE at its
own truncation built around the displaced control truncated onto its
triangle (the surface, tracers and physics its own cold start's, the
door's decision 7).  Synthetic observations of every stream are drawn
from the nature run at the real positions (the station and sounding
tables the door reads, when given; a stated synthetic network otherwise)
with the streams' stated errors, at times spread through each window
when a bin width is set (each report's truth value is H of the nature
state at the report's own bin, amendment B), and the control has to find
its way back through the ensemble covariance (amendment A).  Every score
is against the truth the experiment knows: the area-weighted grid rmse of
the control against the nature state on the control grid and of the
ensemble mean against the nature state truncated to the ensemble grid,
for the wind, the temperature, the surface pressure and the vapor, before
and after each analysis, with the spread beside it, and the O-B and O-A
the analysis receipt carries.

The families (``--family``):

``recovery``
    The twin with noisy observations (the default): the recovery curve,
    the spread against the rmse (a calibrated ensemble has spread within a
    stated band of its rmse), O-B and O-A per stream per cycle.
``pull``
    Perfect observations (values are H(truth) with no noise, the stated
    errors kept) must pull the state to the truth: the control's grid rmse
    against the nature run must fall at EVERY analysis on temperature, wind
    and surface pressure, the mean's from the start, and the control's
    pooled normalised observation fit (the sum over every assimilated row
    of (O-B / sigma)^2, the observation term of the cost) must fall at
    every analysis.  The pooled fit replaces the per-stream "O-A below
    O-B" rule (amendment G): a least-squares analysis lowers the pooled
    cost and may move AWAY from a stream whose reports it trades against
    the others (measured at T63: the 2 m temperature O-A rose 0.43 to 0.73
    K at the second analysis while the sounding temperature fell 0.71 to
    0.51 K over eleven times the rows and the pooled fit fell by a fifth,
    5,676 to 4,526 over 13,813 rows).
    The per-stream reading is kept in the verdict as a diagnostic.
``agree``
    An observation set that agrees with the background (every value is
    the background ensemble mean of H(x), the bits the filter itself
    forms) must move the mean by nothing: the innovation is exactly zero
    by construction, the mean weight vector is exactly zero, and the
    ensemble-mean increment on the grid is the rounding residual of a
    zero-mean perturbation transform, measured in units of the state
    dtype's epsilon and bounded at ``AGREE_BAR_EPSILONS`` of the field's
    own scale (measured: float32 members on the device 7.3 epsilons at
    T63, float64 host doubles 44 epsilons on the test world; the bar is
    four times the worse of the two).
``single``
    One planted report must produce the analytic localised single-report
    increment, ``w P_xy d / (w P_yy + sigma^2)`` at every gridpoint of
    every analysis field, with the Gaspari-Cohn shape in the horizontal
    and in ln p; compared in units of the state dtype's epsilon on the
    filter's own grid fields (bounded at ``SINGLE_BAR_EPSILONS``; measured:
    float32 members on the device 0.85 epsilons at T63, float64 host
    doubles 25 epsilons on the test world) and exactly zero beyond the
    cutoff.  The bar is a rounding bar, not a tolerance: a
    wrong localisation shape, a wrong error variance or a wrong gain sign
    shows at 1e-2 relative and above, five orders away from it.
``transfer``
    The recovery twin with a SECOND control trajectory beside the
    control analysis: the mean-increment transfer of the pre-amendment
    path (:func:`apply_mean_increment`), graded on the same truth, so the
    comparison the amendment asked for is a table, not an argument.

``--sweep option=a,b,c`` repeats the chosen family with one setup option
taking each value (``observation_time_bin_s``, ``increment_application``,
``wind_balance``, ``recentering_fraction``, ``rtps_alpha``, ...): the bin
width sensitivity, direct against incremental application, the two wind
balance modes, full against partial recentring, each as one report.

Runnable: ``python -m woof.globe.da.osse --config <toml> --outdir
<dir> [--truncation 63 --control-truncation 127 --members N --cycles C
--family recovery|pull|agree|single|transfer --sweep key=a,b ...]``.  The
config is the run config; ``--control-truncation`` re-cuts it for the
nature run and the control, ``--truncation`` for the ensemble.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import load_config
from ..constants import DRY_AIR_GAS_CONSTANT, SPECTRAL_FIELDS
from ..obs_table import ObsRow, load_obs
from ..runner import build_model_and_cold_state, build_transform
from ..state import ArwenGlobalState
from .analysis import (
    ControlBackground,
    analyze_ensemble,
    apply_mean_increment,
    member_surface_pressure_extrema,
    members_surface_pressure_record,
    recenter,
)
from .ensemble import GlobalEnsemble, ensemble_config, recut_config, truncate_spectral
from .letkf_point import (
    ColumnGeometry,
    PointLetkfConfig,
    PointLetkfDiagnostics,
    analyze_points,
    flatten_batches,
    single_observation_increment,
)
from .observations import PointObs
from .operators import MemberOperators, batches_from_rows, evaluate_batches
from .options import EnsembleOptions, FilterOptions
from .perturbations import draw_perturbation, member_rng, perturbed_state
from .window import ObservationWindow, batches_unevaluated

from woof.da.letkf import gaspari_cohn

OSSE_SCHEMA = "gpuwm.arwen-global-ensemble-osse/v2"
FAMILIES = ("recovery", "pull", "agree", "single", "transfer")

#: Synthetic stream errors (standard deviations) and the levels the
#: synthetic soundings report at.
STREAM_ERRORS = {
    "osse-stations": {"surface_pressure_pa": 100.0, "temperature_k": 1.0, "dewpoint_k": 1.5,
                      "wind_u_m_s": 1.5, "wind_v_m_s": 1.5},
    "osse-soundings": {"temperature_k": 1.0, "dewpoint_k": 2.5, "wind_u_m_s": 2.5, "wind_v_m_s": 2.5},
    "osse-amv": {"wind_u_m_s": 3.0, "wind_v_m_s": 3.0},
}
MANDATORY_LEVELS_PA = (92500.0, 85000.0, 70000.0, 50000.0, 40000.0, 30000.0, 25000.0, 20000.0)
AMV_LEVELS_PA = (85000.0, 70000.0, 50000.0, 30000.0)

#: The calibrated-spread band the verdict reads: spread over rmse of the
#: ensemble mean against the truth, per field, at the last cycle.
SPREAD_BAND = (0.5, 2.0)
# The analytic families compare two algebraic routes to the same increment in
# the state's own dtype, so their bars are multiples of that dtype's epsilon
# (float32 1.19e-7, float64 2.22e-16): a rounding bar, not a tolerance, set
# at four times the worse residual measured in either dtype (single 0.85 /
# 25 epsilons, agree 7.3 / 44 epsilons for float32 device / float64 host).
SINGLE_BAR_EPSILONS = 128.0
AGREE_BAR_EPSILONS = 256.0


def rounding_bar(dtype, epsilons: float) -> dict[str, object]:
    """The relative bar ``epsilons`` machine epsilons of ``dtype`` wide, with
    the dtype and epsilon recorded so the receipt states what it measures."""
    eps = float(np.finfo(np.dtype(dtype)).eps)
    return {"state_dtype": str(np.dtype(dtype)), "epsilon": eps,
            "bar_in_epsilons": float(epsilons), "bar_relative": float(epsilons) * eps}



@dataclass(frozen=True)
class GlobalOsseSetup:
    """Everything that defines one global twin experiment."""

    config: str
    members: int = 16
    cycles: int = 6
    interval_s: float = 3600.0
    spinup_s: float = 3600.0
    start_perturbation_scale: float = 2.0
    stations: str | None = None
    soundings: str | None = None
    synthetic_stations: int = 600
    synthetic_soundings: int = 120
    amv_points: int = 400
    seed: int = 20260906
    family: str = "recovery"
    horizontal_cutoff_km: float = 1200.0
    vertical_cutoff_lnp: float = 1.5
    surface_vertical_cutoff_lnp: float = 0.6
    rtps_alpha: float = 0.9
    wind_balance: str = "rotational"
    additive_inflation_fraction: float = 0.0
    thinning: bool = True
    max_local_obs: int = 400
    gate_minimum_count: int = 50
    dt_s: float | None = None
    truncation: int | None = None
    control_truncation: int | None = None
    control_dt_s: float | None = None
    #: Width of the observation time bins; None compares every report with
    #: the analysis-time state (the instantaneous form) and draws every
    #: report at the analysis time.
    observation_time_bin_s: float | None = None
    increment_application: str = "direct"
    recentering_fraction: float = 1.0
    recentering_mode: str = "increment"
    transfer_taper_start_degree: int | None = None
    transfer_taper_end_degree: int | None = None
    #: Keep each member's global-mean surface pressure across the
    #: recentring shift (the default); False takes the shift raw, the form
    #: the first twins ran with (the members' mean surface pressure then
    #: moves by the raw control increment's mean every analysis).
    recenter_preserve_mass: bool = True
    #: The hybrid covariance of the control's gain: the ensemble weight
    #: (1.0 is the ensemble alone) and the static covariance table below
    #: one (a path, or ``packaged``).
    hybrid_beta: float = 1.0
    static_covariance: str | None = "packaged"
    static_samples: int = 64

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if int(self.cycles) < 1 or int(self.members) < 3:
            raise ValueError("cycles must be >= 1 and members >= 3")


def _configs(setup: GlobalOsseSetup):
    """``(control_cfg, ensemble_cfg)``: the file's config re-cut at the
    control truncation (the nature run and the control) and at the
    ensemble truncation (``ensemble_config``, the same physics, statics
    and analysis source, dx_m scaled)."""
    cfg = load_config(setup.config)
    control_t = int(setup.control_truncation) if setup.control_truncation is not None else (
        int(setup.truncation) if setup.truncation is not None else int(cfg.truncation))
    ensemble_t = int(setup.truncation) if setup.truncation is not None else control_t
    if control_t != int(cfg.truncation):
        control_cfg = recut_config(cfg, control_t, dt_s=setup.control_dt_s, name=f"{cfg.name}-osse-control-t{control_t}")
    else:
        control_cfg = cfg if setup.control_dt_s is None else dataclasses.replace(cfg, dt_s=float(setup.control_dt_s))
    ecfg = ensemble_config(control_cfg, EnsembleOptions(members=3, truncation=ensemble_t),
                           dt_s=setup.dt_s, name=f"{cfg.name}-osse-ens-t{ensemble_t}")
    return control_cfg, ecfg


def _positions_from_table(path: str, *, surface: bool):
    """Distinct (lat, lon, elevation) of the table's rows, and for a
    sounding table the levels each site reports."""
    _source, rows, _prov = load_obs(str(path))
    seen: dict[tuple, tuple] = {}
    levels: dict[tuple, set] = {}
    for row in rows:
        if (row.level_pa is None) != surface:
            continue
        key = (round(row.latitude_deg, 3), round(row.longitude_deg, 3))
        seen.setdefault(key, (row.latitude_deg, row.longitude_deg, row.elevation_m))
        if not surface:
            levels.setdefault(key, set()).add(float(row.level_pa))
    positions = list(seen.values())
    return positions, {seen[k]: sorted(v) for k, v in levels.items()}


def _synthetic_network(rng, count: int, *, land_bias: bool = True):
    """``count`` positions with the land-heavy distribution a station
    network has: two thirds in the northern mid-latitudes."""
    n_mid = int(round(count * 0.66)) if land_bias else count // 2
    lat = np.concatenate([
        rng.uniform(25.0, 65.0, n_mid),
        np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, count - n_mid))),
    ])
    lon = rng.uniform(-180.0, 180.0, count)
    return [(float(a), float(b), 0.0) for a, b in zip(lat, lon)]


class _Network:
    """The observing network the twin samples the nature run with."""

    def __init__(self, setup: GlobalOsseSetup, rng, terrain_height_at):
        self.setup = setup
        if setup.stations:
            self.stations, _ = _positions_from_table(setup.stations, surface=True)
            self.stations_source = str(setup.stations)
        else:
            self.stations = _synthetic_network(rng, setup.synthetic_stations)
            self.stations_source = f"synthetic ({setup.synthetic_stations})"
        # A synthetic station sits on the model terrain: the pressure and
        # 2 m reductions then carry no terrain mismatch.
        if not setup.stations:
            lat = np.array([s[0] for s in self.stations])
            lon = np.array([s[1] for s in self.stations])
            z = terrain_height_at(lat, lon)
            self.stations = [(a, b, float(h)) for (a, b, _), h in zip(self.stations, z)]
        if setup.soundings:
            self.soundings, self.sounding_levels = _positions_from_table(setup.soundings, surface=False)
            self.soundings_source = str(setup.soundings)
        else:
            self.soundings = _synthetic_network(rng, setup.synthetic_soundings)
            self.sounding_levels = {s: list(MANDATORY_LEVELS_PA) for s in self.soundings}
            self.soundings_source = f"synthetic ({setup.synthetic_soundings})"

    def rows(self, moment: dt.datetime, window_s: float, rng, *, spread_times: bool) -> list[ObsRow]:
        """Every stream's reports for the window ending at ``moment``, with
        placeholder values (the nature run fills them) and times spread
        uniformly through the window when ``spread_times``."""
        rows: list[ObsRow] = []

        def when(k: int) -> dt.datetime:
            if not spread_times:
                return moment
            return moment - dt.timedelta(seconds=float(rng.uniform(0.0, window_s)))

        for k, (lat, lon, elev) in enumerate(self.stations):
            t = when(k)
            for variable, error in STREAM_ERRORS["osse-stations"].items():
                rows.append(ObsRow("osse-stations", f"S{k:05d}", float(lat), float(lon), float(elev),
                                   None, t, variable, 0.0, error))
        k = 0
        for site in self.soundings:
            t = when(k)
            for level in self.sounding_levels.get(site, MANDATORY_LEVELS_PA):
                for variable, error in STREAM_ERRORS["osse-soundings"].items():
                    rows.append(ObsRow("osse-soundings", f"U{k:05d}", float(site[0]), float(site[1]), 0.0,
                                       float(level), t, variable, 0.0, error))
                k += 1
        n = int(self.setup.amv_points)
        if n:
            a_lat = np.rad2deg(np.arcsin(rng.uniform(-0.95, 0.95, n)))
            a_lon = rng.uniform(-180.0, 180.0, n)
            a_lev = rng.choice(np.array(AMV_LEVELS_PA), n)
            for k in range(n):
                t = when(k)
                for variable, error in STREAM_ERRORS["osse-amv"].items():
                    rows.append(ObsRow("osse-amv", f"A{k:05d}", float(a_lat[k]), float(a_lon[k]), 0.0,
                                       float(a_lev[k]), t, variable, 0.0, error))
        return rows

    def describe(self) -> dict[str, object]:
        return {
            "stations": len(self.stations), "stations_source": self.stations_source,
            "soundings": len(self.soundings), "soundings_source": self.soundings_source,
            "sounding_levels_pa": sorted({lv for levels in self.sounding_levels.values() for lv in levels}),
            "amv_points_per_cycle": int(self.setup.amv_points),
            "amv_levels_pa": list(AMV_LEVELS_PA),
            "errors": STREAM_ERRORS,
        }


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def _state_fields(model, transform, state: ArwenGlobalState) -> dict[str, np.ndarray]:
    backend = transform.backend
    g = model.grid_state(state.atmosphere, only=("u", "v", "temperature", "qv", "ps"))
    out = {
        "u": np.asarray(backend.to_numpy(g["u"]), dtype=np.float64),
        "v": np.asarray(backend.to_numpy(g["v"]), dtype=np.float64),
        "temperature_k": np.asarray(backend.to_numpy(g["temperature"]), dtype=np.float64),
        "qv": np.asarray(backend.to_numpy(g["qv"]), dtype=np.float64),
        "surface_pressure_pa": np.asarray(backend.to_numpy(g["ps"]), dtype=np.float64),
    }
    model.release_syntheses()
    return out


def _mean_fields(ensemble: GlobalEnsemble) -> dict[str, np.ndarray]:
    acc: dict[str, np.ndarray] = {}
    for member in ensemble.members:
        f = _state_fields(ensemble.model, ensemble.transform, member)
        for name, value in f.items():
            acc[name] = value if name not in acc else acc[name] + value
    return {name: value / ensemble.size for name, value in acc.items()}


def _area_rms(grid, field: np.ndarray) -> float:
    """Area-weighted rms over the sphere (levels averaged for a 3-D field)."""
    if field.ndim == 3:
        return float(math.sqrt(np.mean([grid.global_mean(level ** 2) for level in field])))
    return float(math.sqrt(max(0.0, grid.global_mean(field ** 2))))


def _truncated_truth(ensemble: GlobalEnsemble, truth: ArwenGlobalState) -> ArwenGlobalState:
    """The nature state's spectral fields truncated onto the ensemble
    triangle, in a state of the ensemble's own (surface, tracers, physics
    from member 0; only the spectral atmosphere is read from it)."""
    backend = ensemble.transform.backend
    ens_t = int(ensemble.transform.truncation)
    fields = []
    for name in SPECTRAL_FIELDS:
        coeff = getattr(truth.atmosphere, name)
        host = coeff.get() if hasattr(coeff, "get") else np.asarray(coeff)
        fields.append(backend.asarray(truncate_spectral(host, ens_t), dtype=backend.complex_dtype))
    member = ensemble.members[0]
    return ArwenGlobalState(member.atmosphere.with_fields(fields), member.surface, member.physics_state)


def _reduced_pressure(model, transform, state: ArwenGlobalState, terrain_geopotential: np.ndarray) -> np.ndarray:
    """The state's surface pressure reduced to sea level through its own
    lowest-level virtual temperature and the terrain its pressure sits on:
    ``ps exp(g z / (R Tv))``, ``(nlat, nlon)``."""
    backend = transform.backend
    g = model.grid_state(state.atmosphere, only=("temperature", "qv", "ps"))
    t_low = np.asarray(backend.to_numpy(g["temperature"][-1]), dtype=np.float64)
    q_low = np.maximum(np.asarray(backend.to_numpy(g["qv"][-1]), dtype=np.float64), 0.0)
    ps = np.asarray(backend.to_numpy(g["ps"]), dtype=np.float64)
    model.release_syntheses()
    tv = t_low * (1.0 + 0.61 * q_low)
    return ps * np.exp(np.asarray(terrain_geopotential, dtype=np.float64) / (DRY_AIR_GAS_CONSTANT * tv))


def score(ensemble: GlobalEnsemble, truth: ArwenGlobalState, *, truth_terrain_geopotential=None) -> dict[str, object]:
    """Grid rmse of the ensemble mean against the truth (truncated to the
    ensemble triangle when it is finer) and the spread, per field,
    area-weighted.  ``spread`` is the area-weighted rms of the pointwise
    spread; ``spread_equal_weight`` is the equal-weight mean over the
    Gaussian gridpoints the first measurements used, recorded so the two
    instruments can be read side by side (it over-weights the polar rings
    against the area-weighted rmse).  ``mslp_pa`` is the pressure score
    that survives a truth on finer orography: each member's and the
    truth's surface pressure reduced to sea level through its own
    lowest-level virtual temperature and the terrain its pressure sits on
    (``truth_terrain_geopotential``, the control's terrain truncated to
    the ensemble triangle, on the ensemble grid; the ensemble's own when
    None), because the raw ``surface_pressure_pa`` difference between a
    T63 state and a truncated T127 state carries the two orographies'
    difference, which no analysis at T63 can remove."""
    grid = ensemble.transform.grid
    backend = ensemble.transform.backend
    truth_l = truth if int(truth.atmosphere.theta.shape[-1]) - 1 == int(ensemble.transform.truncation) \
        else _truncated_truth(ensemble, truth)
    truth_f = _state_fields(ensemble.model, ensemble.transform, truth_l)
    mean_f = _mean_fields(ensemble)
    spread = ensemble.spread()
    spread_equal = ensemble.spread(area_weighted=False)
    out = {}
    for name in truth_f:
        rmse = _area_rms(grid, mean_f[name] - truth_f[name])
        out[name] = {"rmse": rmse, "spread": float(spread[name]),
                     "spread_equal_weight": float(spread_equal[name]),
                     "spread_over_rmse": float(spread[name] / rmse) if rmse > 0 else None}
    own_terrain = np.asarray(backend.to_numpy(ensemble.model.surface_geopotential), dtype=np.float64)
    truth_terrain = own_terrain if truth_terrain_geopotential is None else np.asarray(truth_terrain_geopotential, dtype=np.float64)
    reduced = np.stack([_reduced_pressure(ensemble.model, ensemble.transform, m, own_terrain) for m in ensemble.members])
    reduced_truth = _reduced_pressure(ensemble.model, ensemble.transform, truth_l, truth_terrain)
    r = reduced.shape[0]
    var = ((reduced - reduced.mean(axis=0)) ** 2).sum(axis=0) / max(r - 1, 1)
    mslp_spread = float(math.sqrt(max(0.0, grid.global_mean(var))))
    mslp_rmse = _area_rms(grid, reduced.mean(axis=0) - reduced_truth)
    out["mslp_pa"] = {
        "rmse": mslp_rmse, "spread": mslp_spread,
        "spread_over_rmse": float(mslp_spread / mslp_rmse) if mslp_rmse > 0 else None,
        "measures": "surface pressure reduced to sea level through each state's own lowest-level virtual "
                    "temperature and the terrain its pressure sits on (the truth's: the control terrain "
                    "truncated to the ensemble triangle); the raw surface_pressure_pa row carries the "
                    "orography difference between the two truncations as well",
    }
    return out


def score_control(model, transform, control: ArwenGlobalState, truth: ArwenGlobalState) -> dict[str, object]:
    """Grid rmse of the control against the truth on the control grid."""
    grid = transform.grid
    truth_f = _state_fields(model, transform, truth)
    ctl_f = _state_fields(model, transform, control)
    return {name: {"rmse": _area_rms(grid, ctl_f[name] - truth_f[name])} for name in truth_f}


def _time_of(start: dt.datetime, time_s: float) -> dt.datetime:
    return start + dt.timedelta(seconds=float(time_s))


def _advance(model, state: ArwenGlobalState, targets, dt_s: float, steps: int, observer=None):
    """``steps`` steps of one state under its conservation targets, the
    observer called after each with ``(state, time_s)``."""
    mass, water = targets
    for _ in range(steps):
        model.set_conservation_targets(mass, water)
        state, _m = model.step(state, dt_s)
        model.release_syntheses()
        if observer is not None:
            observer(state, float(state.time_s))
    return state


def _stream_table(report: dict) -> dict:
    streams = {}
    for stream, variables in report["streams"].items():
        streams[stream] = {}
        for variable, entry in variables.items():
            g = entry["regions"].get("global", {})
            row = {
                "count": entry["count"], "withheld": entry["withheld_count"],
                "o_minus_b_rms": g.get("assimilated", {}).get("o_minus_b", {}).get("rms"),
                "o_minus_a_rms": (g.get("assimilated", {}).get("o_minus_a") or {}).get("rms"),
                "withheld_o_minus_b_rms": g.get("withheld", {}).get("o_minus_b", {}).get("rms"),
                "withheld_o_minus_a_rms": (g.get("withheld", {}).get("o_minus_a") or {}).get("rms"),
                "desroziers": {k: entry.get("desroziers", {}).get(k) for k in
                               ("error_variance_ratio", "background_variance_ratio", "innovation_ratio")},
                "verdict": entry["verdict"],
            }
            ctl = g.get("control")
            if ctl:
                row["control_o_minus_b_rms"] = ctl["assimilated"]["o_minus_b"]["rms"]
                row["control_o_minus_a_rms"] = (ctl["assimilated"]["o_minus_a"] or {}).get("rms")
            streams[stream][variable] = row
    return streams


# ---------------------------------------------------------------------------
# The twin
# ---------------------------------------------------------------------------

def run_global_osse(setup: GlobalOsseSetup, outdir: str | Path, *, progress=print) -> dict[str, object]:
    """The twin experiment (``recovery``, ``pull`` and ``transfer``)."""
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    control_cfg, ecfg = _configs(setup)
    det_transform = build_transform(control_cfg)
    det_model, det_cold = build_model_and_cold_state(control_cfg, det_transform, scratch_destination=output)
    transform = build_transform(ecfg)
    model, ens_cold = build_model_and_cold_state(ecfg, transform, scratch_destination=output)
    start = dt.datetime(2026, 9, 1, 0, 0, tzinfo=dt.timezone.utc)
    dt_h = float(control_cfg.dt_s)
    dt_l = float(ecfg.dt_s)
    for label, value in (("interval_s", setup.interval_s), ("spinup_s", setup.spinup_s)):
        for name, step in (("control", dt_h), ("ensemble", dt_l)):
            if abs(value / step - round(value / step)) > 1e-9:
                raise ValueError(f"{label} must be a whole number of the {name} dt ({step:g} s)")
    bin_s = float(setup.observation_time_bin_s) if setup.observation_time_bin_s else float(setup.interval_s)
    options = EnsembleOptions(
        members=int(setup.members), truncation=int(ecfg.truncation), seed=int(setup.seed),
        additive_inflation_fraction=float(setup.additive_inflation_fraction),
    )
    filter_options = FilterOptions(
        horizontal_cutoff_km=setup.horizontal_cutoff_km, vertical_cutoff_lnp=setup.vertical_cutoff_lnp,
        surface_vertical_cutoff_lnp=setup.surface_vertical_cutoff_lnp, rtps_alpha=setup.rtps_alpha,
        wind_balance=setup.wind_balance, thinning=setup.thinning, max_local_obs=setup.max_local_obs,
        gate_minimum_count=setup.gate_minimum_count, increment_application=setup.increment_application,
        iau_window_s=float(setup.interval_s), recentering_fraction=setup.recentering_fraction,
        recentering_mode=setup.recentering_mode,
        transfer_taper_start_degree=setup.transfer_taper_start_degree,
        transfer_taper_end_degree=setup.transfer_taper_end_degree,
        hybrid_beta=float(setup.hybrid_beta), static_covariance=setup.static_covariance,
        static_samples=int(setup.static_samples),
    )
    det_operators = MemberOperators.for_model(det_model, det_transform, control_cfg)
    ens_operators = MemberOperators.for_model(model, transform, ecfg)
    rng = np.random.default_rng([setup.seed, 7])

    def terrain_height_at(lat, lon):
        from woof.globe.spectral.sampling import sample_scalar
        return sample_scalar(det_transform, det_operators.terrain, lat, lon) / 9.80665

    network = _Network(setup, rng, terrain_height_at)
    progress(f"osse: T{control_cfg.truncation} nature and control, T{ecfg.truncation} ensemble of {setup.members}; "
             f"network {network.describe()['stations']} stations, {network.describe()['soundings']} soundings, "
             f"{setup.amv_points} AMVs per cycle; bins {bin_s:g} s")

    # Nature run: spin-up, then the truth at every analysis time and the
    # observation-space truth of every report at its own bin.
    noisy = setup.family in ("recovery", "transfer")
    spread_times = setup.observation_time_bin_s is not None
    nature = det_cold
    nature_targets = (det_model.initialize_mass_target(nature.atmosphere), det_model.initialize_water_target(nature))
    steps_spinup_h = int(round(setup.spinup_s / dt_h))
    steps_interval_h = int(round(setup.interval_s / dt_h))
    steps_interval_l = int(round(setup.interval_s / dt_l))
    t0 = time.perf_counter()
    nature = _advance(det_model, nature, nature_targets, dt_h, steps_spinup_h)
    truths = [nature.copy()]
    cycle_batches: list[list[PointObs]] = []
    windows_record = []
    for c in range(int(setup.cycles)):
        t_end = float(truths[0].time_s) + c * float(setup.interval_s)
        moment = _time_of(start, t_end)
        rows = network.rows(moment, float(setup.interval_s), rng, spread_times=spread_times)
        batches = batches_unevaluated(rows, det_operators)
        if c == 0:
            # The first analysis has no window before it: every report is
            # drawn from the truth at the analysis time.
            evaluate_batches(det_operators, [truths[0]], batches, target="simulated")
        else:
            window = ObservationWindow(batches, start_s=t_end - float(setup.interval_s), end_s=t_end,
                                       epoch=start, bin_s=bin_s, dt_s=dt_h)
            nature = _advance(det_model, nature, nature_targets, dt_h, steps_interval_h,
                              observer=lambda state, t, w=window: w.observe([state], t, det_operators))
            window.finish([nature], det_operators)
            windows_record.append(window.record())
            truths.append(nature.copy())
        finished = []
        for b in batches:
            keep = np.all(np.isfinite(b.simulated), axis=0)
            b = b.subset(keep)
            noise = rng.normal(0.0, b.error) if noisy else np.zeros(b.count)
            b.value = b.simulated[0] + noise
            b.simulated = None
            b.control_simulated = None
            b.ln_pressure = np.where(b.surface, np.nan, b.ln_pressure)
            finished.append(b)
        cycle_batches.append(finished)
    nature_wall = time.perf_counter() - t0
    progress(f"osse: nature run {len(truths)} truths and {sum(sum(b.count for b in bs) for bs in cycle_batches)} "
             f"reports in {nature_wall:.1f} s")

    # The control starts displaced from the truth at the first analysis time.
    displaced_inc, displacement = draw_perturbation(
        det_model, det_transform, truths[0].atmosphere,
        dataclasses.replace(options, truncation=int(control_cfg.truncation)),
        member_rng(setup.seed + 1, 0, "displacement"), amplitude_scale=float(setup.start_perturbation_scale))
    control_state = perturbed_state(det_model, det_transform, truths[0], displaced_inc)
    control_targets = (det_model.initialize_mass_target(control_state.atmosphere),
                       det_model.initialize_water_target(control_state))
    # The ensemble around ITS OWN spun-up cold start (the same analysis at
    # the ensemble truncation, terrain-consistent) plus the control's
    # displacement truncated onto its triangle: a truncated control STATE
    # would carry the finer orography's surface pressure onto the coarser
    # grid (7 hPa of it on the T63 / T127 twin).
    ens_targets = (model.initialize_mass_target(ens_cold.atmosphere), model.initialize_water_target(ens_cold))
    ens_base = _advance(model, ens_cold, ens_targets, dt_l, int(round(setup.spinup_s / dt_l)))
    # The control's terrain truncated onto the ensemble triangle, on the
    # ensemble grid: the terrain the truncated truth's pressure sits on.
    if int(control_cfg.truncation) == int(ecfg.truncation):
        truth_terrain = None
    else:
        det_terrain = det_transform.forward(det_model.surface_geopotential)
        det_terrain = det_terrain.get() if hasattr(det_terrain, "get") else np.asarray(det_terrain)
        truth_terrain = np.asarray(transform.backend.to_numpy(transform.inverse(
            transform.backend.asarray(truncate_spectral(det_terrain, int(ecfg.truncation)),
                                      dtype=transform.backend.complex_dtype))), dtype=np.float64)
    base_fields = []
    for name in SPECTRAL_FIELDS:
        coeff = getattr(control_state.atmosphere, name) - getattr(truths[0].atmosphere, name)
        host = coeff.get() if hasattr(coeff, "get") else np.asarray(coeff)
        base_fields.append(getattr(ens_base.atmosphere, name) + transform.backend.asarray(
            truncate_spectral(host, int(ecfg.truncation)), dtype=transform.backend.complex_dtype))
    base = ArwenGlobalState(ens_base.atmosphere.with_fields(base_fields, time_s=control_state.time_s,
                                                            step=int(round(control_state.time_s / dt_l))),
                            ens_base.surface, ens_base.physics_state)
    base, _n, _t, _f = model._repair_positivity(base)
    model.enforce(base)
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, base, options)
    # The displaced start's surface-pressure headroom under the radiation
    # ceiling, members and control: a truncated orography undershoots below
    # sea level at the Andes' Pacific foot (T63: 109.2 to 109.7 kPa there by
    # construction) and a member that starts above the ceiling dies in the
    # twin's first radiation call with no analysis at all (seed 4242, scale
    # 1.5, 2026-09-06), so the twin refuses such a start by name.
    start_pressure = members_surface_pressure_record(
        [member_surface_pressure_extrema(model, transform, m) for m in ensemble.members])
    control_pressure = member_surface_pressure_extrema(det_model, det_transform, control_state)
    members_reading = start_pressure["members_surface_pressure_pa"]
    progress(f"osse: displaced start surface pressure: members up to {members_reading['max']:.0f} Pa "
             f"(member {members_reading['max_member']} at {members_reading['column_of_max']['latitude_deg']:.1f} N, "
             f"{members_reading['column_of_max']['longitude_deg']:.1f} E, {members_reading['headroom_pa']:.0f} Pa under "
             f"the radiation ceiling), control up to {control_pressure['surface_pressure_max_pa']:.0f} Pa")
    above = start_pressure["members_above_radiation_ceiling"]
    if above or control_pressure["columns_above_ceiling"]:
        who = ", ".join(f"member {m['member']} at {m['surface_pressure_max_pa']:.0f} Pa" for m in above)
        if control_pressure["columns_above_ceiling"]:
            who = (who + "; " if who else "") + f"the control at {control_pressure['surface_pressure_max_pa']:.0f} Pa"
        raise ValueError(
            f"the displaced start exceeds the radiation tables' ceiling "
            f"({members_reading['ceiling_pa']:,.0f} Pa): {who} (seed {setup.seed}, start_perturbation_scale "
            f"{setup.start_perturbation_scale:g}, T{ecfg.truncation} members under a T{control_cfg.truncation} "
            f"control; the truncated orography undershoots at {members_reading['column_of_max']['latitude_deg']:.1f} N, "
            f"{members_reading['column_of_max']['longitude_deg']:.1f} E); the twin would die in its first radiation "
            "call: choose another seed, a smaller start_perturbation_scale or a finer ensemble truncation"
        )
    # The comparison arm (transfer family): a second control receiving the
    # mean increment.
    mean_arm = control_state.copy() if setup.family == "transfer" else None
    mean_arm_targets = control_targets
    initial = {"control": score_control(det_model, det_transform, control_state, truths[0]),
               "ensemble": score(ensemble, truths[0], truth_terrain_geopotential=truth_terrain)}
    progress(f"osse: displaced start control rmse T {initial['control']['temperature_k']['rmse']:.3f} K, "
             f"u {initial['control']['u']['rmse']:.3f} m/s, ps {initial['control']['surface_pressure_pa']['rmse']:.1f} Pa; "
             f"ensemble mean T {initial['ensemble']['temperature_k']['rmse']:.3f} K, spread {initial['ensemble']['temperature_k']['spread']:.3f} K")

    cycles_record = []
    for c, truth in enumerate(truths):
        t_end = float(truth.time_s)
        moment = _time_of(start, t_end)
        batches = cycle_batches[c]
        window = None
        if c > 0:
            window = ObservationWindow(batches, start_s=t_end - float(setup.interval_s), end_s=t_end,
                                       epoch=start, bin_s=bin_s, dt_s=dt_l)
            ctl_window = ObservationWindow(batches, start_s=t_end - float(setup.interval_s), end_s=t_end,
                                           epoch=start, bin_s=bin_s, dt_s=dt_h)
            t_f = time.perf_counter()
            control_state = _advance(det_model, control_state, control_targets, dt_h, steps_interval_h,
                                     observer=lambda state, t, w=ctl_window: w.observe([state], t, det_operators, control=True))
            ctl_window.finish([control_state], det_operators, control=True)
            if mean_arm is not None:
                mean_arm = _advance(det_model, mean_arm, mean_arm_targets, dt_h, steps_interval_h)
            control_wall = time.perf_counter() - t_f
            t_f = time.perf_counter()
            timings = ensemble.advance_to(t_end, dt_l,
                                          observer=lambda ens, t, w=window: w.observe(ens.members, t, ens_operators))
            window.finish(ensemble.members, ens_operators)
            step_wall = float(sum(t.wall_s for t in timings))
            window_record = window.record()
        else:
            control_wall = 0.0
            step_wall = 0.0
            window_record = None
        before = {"control": score_control(det_model, det_transform, control_state, truth),
                  "ensemble": score(ensemble, truth, truth_terrain_geopotential=truth_terrain)}
        control = ControlBackground(control_state, det_model, det_transform, control_cfg, operators=det_operators)
        t_a = time.perf_counter()
        result = analyze_ensemble(ensemble, batches, filter_options, analysis_time=moment,
                                  additive_inflation=noisy, control=control)
        control_state = result.control_analysis
        control_targets = (det_model.initialize_mass_target(control_state.atmosphere),
                           det_model.initialize_water_target(control_state))
        recenter_record = recenter(ensemble, control_state, det_transform, fraction=setup.recentering_fraction,
                                   mode=setup.recentering_mode, control_increment=result.control_increment_spectral,
                                   mean_increment=result.mean_increment_spectral,
                                   preserve_global_mean_pressure=bool(setup.recenter_preserve_mass))
        analysis_wall = time.perf_counter() - t_a
        after = {"control": score_control(det_model, det_transform, control_state, truth),
                 "ensemble": score(ensemble, truth, truth_terrain_geopotential=truth_terrain)}
        record = {
            "cycle": c, "time_s": t_end, "analysis_time_utc": moment.isoformat(timespec="seconds"),
            "status": result.status, "reports": int(sum(b.count for b in batches)),
            "assimilated": result.report["assimilated_total"], "withheld": result.report["withheld_total"],
            "rejections": result.report["rejections"],
            "before": before, "after": after, "streams": _stream_table(result.report),
            "assessments": {k: v["verdict"] for k, v in result.report["assessments"].items()},
            "letkf": result.report["letkf"], "increment": result.report["increment"],
            "recenter": recenter_record, "window": window_record,
            "control_forecast_wall_s": control_wall, "ensemble_forecast_wall_s": step_wall,
            "analysis_wall_s": analysis_wall, "analysis_timings_s": result.timings_s,
        }
        if mean_arm is not None:
            mean_before = score_control(det_model, det_transform, mean_arm, truth)
            mean_arm, arm_record = apply_mean_increment(mean_arm, det_model, det_transform,
                                                        result.mean_increment_spectral, options=filter_options)
            mean_arm_targets = (det_model.initialize_mass_target(mean_arm.atmosphere),
                                det_model.initialize_water_target(mean_arm))
            record["mean_increment_arm"] = {
                "before": mean_before, "after": score_control(det_model, det_transform, mean_arm, truth),
                "record": arm_record,
            }
        cycles_record.append(record)
        progress(
            f"osse cycle {c}: control T rmse {before['control']['temperature_k']['rmse']:.3f} -> "
            f"{after['control']['temperature_k']['rmse']:.3f} K, u {before['control']['u']['rmse']:.3f} -> "
            f"{after['control']['u']['rmse']:.3f}, ps {before['control']['surface_pressure_pa']['rmse']:.1f} -> "
            f"{after['control']['surface_pressure_pa']['rmse']:.1f} Pa; mean T {before['ensemble']['temperature_k']['rmse']:.3f} -> "
            f"{after['ensemble']['temperature_k']['rmse']:.3f} (spread {after['ensemble']['temperature_k']['spread']:.3f}), "
            f"mslp {before['ensemble']['mslp_pa']['rmse']:.1f} -> {after['ensemble']['mslp_pa']['rmse']:.1f} Pa "
            f"(spread {after['ensemble']['mslp_pa']['spread']:.1f}), members mean ps "
            f"{recenter_record['members_global_mean_surface_pressure_pa']['before']} -> "
            f"{recenter_record['members_global_mean_surface_pressure_pa']['after']}; "
            f"status {result.status}, {analysis_wall:.1f} s analysis"
            + (f", mean arm T {record['mean_increment_arm']['after']['temperature_k']['rmse']:.3f}" if mean_arm is not None else "")
        )
    report = {
        "schema": OSSE_SCHEMA,
        "family": setup.family,
        "setup": asdict(setup),
        "control_config_hash": control_cfg.config_hash,
        "ensemble_config_hash": ecfg.config_hash,
        "control_truncation": int(control_cfg.truncation),
        "ensemble_truncation": int(ecfg.truncation),
        "nlev": int(model.nlev),
        "control_dt_s": dt_h, "ensemble_dt_s": dt_l,
        "observation_time_bin_s": bin_s,
        "network": network.describe(),
        "nature_windows": windows_record,
        "displacement": {"scale": float(setup.start_perturbation_scale), "record": displacement,
                         "members_surface_pressure_pa": start_pressure["members_surface_pressure_pa"],
                         "control_surface_pressure_pa": control_pressure},
        "options": {"ensemble": options.identity(), "filter": filter_options.identity()},
        "initial_score": initial,
        "cycles": cycles_record,
        "nature_wall_s": nature_wall,
        "resident_bytes": ensemble.resident_bytes(),
        "step_timing": [t.as_record() for t in ensemble.timings[-3:]],
        "verdict": _verdict(setup.family, initial, cycles_record, errors=STREAM_ERRORS),
    }
    (output / f"osse-{setup.family}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    progress(f"osse: verdict {json.dumps(report['verdict'], default=str)}")
    return report


def pooled_normalised_fit(streams: dict, errors: dict, *, prefix: str = "control_") -> dict[str, float | None]:
    """The observation term of the cost pooled over every (stream,
    variable) with rows: ``sum(count * (rms / sigma)^2)`` for O-B and O-A,
    with sigma the stream's stated error.  In the linear case the analysis
    minimises the cost in the ensemble subspace, so the pooled O-A term
    lies below the pooled O-B term even where one stream's own rms rises
    (amendment G: a stream need not move closer).  ``prefix`` selects the
    control's columns (``control_``) or the ensemble mean's (``""``)."""
    total_b = total_a = 0.0
    rows = 0
    for stream, variables in streams.items():
        for variable, row in variables.items():
            sigma = errors.get(stream, {}).get(variable)
            rms_b = row.get(prefix + "o_minus_b_rms")
            rms_a = row.get(prefix + "o_minus_a_rms")
            if not row.get("count") or sigma is None or rms_b is None or rms_a is None:
                continue
            total_b += row["count"] * (rms_b / sigma) ** 2
            total_a += row["count"] * (rms_a / sigma) ** 2
            rows += int(row["count"])
    if rows == 0:
        return {"rows": 0, "o_minus_b": None, "o_minus_a": None, "ratio": None}
    return {"rows": rows, "o_minus_b": float(total_b), "o_minus_a": float(total_a),
            "ratio": float(total_a / total_b) if total_b > 0 else None}


def _verdict(family: str, initial, cycles, *, errors: dict | None = None) -> dict[str, object]:
    errors = STREAM_ERRORS if errors is None else errors
    ctl_t = [c["after"]["control"]["temperature_k"]["rmse"] for c in cycles]
    ctl_u = [c["after"]["control"]["u"]["rmse"] for c in cycles]
    ctl_ps = [c["after"]["control"]["surface_pressure_pa"]["rmse"] for c in cycles]
    mean_t = [c["after"]["ensemble"]["temperature_k"]["rmse"] for c in cycles]
    mean_u = [c["after"]["ensemble"]["u"]["rmse"] for c in cycles]
    fell_ctl = ctl_t[-1] < initial["control"]["temperature_k"]["rmse"] and ctl_u[-1] < initial["control"]["u"]["rmse"]
    fell_mean = mean_t[-1] < initial["ensemble"]["temperature_k"]["rmse"] and mean_u[-1] < initial["ensemble"]["u"]["rmse"]
    every_analysis_improved_control = all(
        c["after"]["control"]["temperature_k"]["rmse"] < c["before"]["control"]["temperature_k"]["rmse"] for c in cycles)
    every_analysis_improved_control_all_fields = all(
        c["after"]["control"][name]["rmse"] < c["before"]["control"][name]["rmse"]
        for c in cycles for name in ("temperature_k", "u", "surface_pressure_pa"))
    pooled_control = [pooled_normalised_fit(c["streams"], errors) for c in cycles]
    pooled_mean = [pooled_normalised_fit(c["streams"], errors, prefix="") for c in cycles]
    control_pooled_fit_fell = all(p["ratio"] is not None and p["ratio"] < 1.0 for p in pooled_control)
    control_o_a_below_o_b_temperature = all(
        (v.get("control_o_minus_a_rms") is not None and v["control_o_minus_a_rms"] < v["control_o_minus_b_rms"])
        for c in cycles for s in c["streams"].values() for k, v in s.items() if v["count"] and k == "temperature_k")
    ratios = {name: c["after"]["ensemble"][name]["spread_over_rmse"] for c in [cycles[-1]]
              for name in ("temperature_k", "u", "surface_pressure_pa", "mslp_pa")}
    ratios_equal_weight = {
        name: (c["after"]["ensemble"][name]["spread_equal_weight"] / c["after"]["ensemble"][name]["rmse"]
               if c["after"]["ensemble"][name]["rmse"] > 0 else None)
        for c in [cycles[-1]] for name in ("temperature_k", "u", "surface_pressure_pa")}
    # The pressure ratio is judged on the reduced pressure: the raw row's
    # rmse against a truth on finer orography is the two orographies'
    # difference before it is an error.
    spread_calibrated = all(r is not None and SPREAD_BAND[0] <= r <= SPREAD_BAND[1]
                            for k, r in ratios.items() if k != "surface_pressure_pa")
    engineering = all(c["status"] == "pass" for c in cycles)
    out = {
        "control_temperature_rmse_by_cycle": ctl_t,
        "control_wind_rmse_by_cycle": ctl_u,
        "control_surface_pressure_rmse_by_cycle": ctl_ps,
        "mean_temperature_rmse_by_cycle": mean_t,
        "mean_wind_rmse_by_cycle": mean_u,
        "control_rmse_fell_from_start": bool(fell_ctl),
        "mean_rmse_fell_from_start": bool(fell_mean),
        "every_analysis_reduced_control_temperature_rmse": bool(every_analysis_improved_control),
        "every_analysis_reduced_control_rmse_on_temperature_wind_and_pressure": bool(every_analysis_improved_control_all_fields),
        "control_pooled_normalised_fit_by_cycle": pooled_control,
        "mean_pooled_normalised_fit_by_cycle": pooled_mean,
        "control_pooled_fit_fell_every_cycle": bool(control_pooled_fit_fell),
        "control_o_a_below_o_b_temperature_every_cycle": bool(control_o_a_below_o_b_temperature),
        "control_o_a_below_o_b_temperature_is_a_diagnostic": "amendment G: a stream need not move closer; the pooled normalised fit is the pull family's gate",
        "spread_over_rmse_last_cycle": ratios,
        "spread_over_rmse_last_cycle_equal_weight_spread": ratios_equal_weight,
        "spread_over_rmse_rule": (
            "spread is the area-weighted rms of the pointwise spread against the area-weighted rmse; "
            "the pressure ratio of record is mslp_pa (each state reduced to sea level through its own "
            "terrain), the raw surface_pressure_pa ratio carries the orography difference between the "
            "two truncations in its denominator; the equal-weight spread is the first measurements' form"
        ),
        "spread_band": list(SPREAD_BAND),
        "spread_calibrated": bool(spread_calibrated),
        "engineering_validity_every_cycle": bool(engineering),
    }
    if family in ("recovery", "pull", "transfer"):
        out["passed"] = bool(engineering and fell_ctl and fell_mean and every_analysis_improved_control)
        if family == "pull":
            out["passed"] = bool(out["passed"] and every_analysis_improved_control_all_fields and control_pooled_fit_fell)
    else:
        out["passed"] = None
    if family == "transfer":
        arm_t = [c["mean_increment_arm"]["after"]["temperature_k"]["rmse"] for c in cycles]
        arm_u = [c["mean_increment_arm"]["after"]["u"]["rmse"] for c in cycles]
        out["mean_increment_arm_temperature_rmse_by_cycle"] = arm_t
        out["mean_increment_arm_wind_rmse_by_cycle"] = arm_u
        out["control_analysis_beats_mean_increment_arm_at_last_cycle"] = {
            "temperature_k": bool(ctl_t[-1] < arm_t[-1]), "u": bool(ctl_u[-1] < arm_u[-1]),
        }
    return out


# ---------------------------------------------------------------------------
# The two analytic families, on the filter's own grid fields
# ---------------------------------------------------------------------------

def agreeing_observations_family(ensemble: GlobalEnsemble, batches: list[PointObs],
                                 filter_options: FilterOptions) -> dict[str, object]:
    """Every report's value set to the background ensemble mean of H(x)
    (the bits the filter forms): the mean increment on the grid must be
    the rounding residual of a zero-mean transform.  Returns the measured
    maxima against each field's scale."""
    prior, geometry = _grid_prior(ensemble, filter_options)
    xp = ensemble.transform.backend.xp
    flat = flatten_batches(
        batches, xp, horizontal_cutoff_m=filter_options.horizontal_cutoff_km * 1000.0,
        vertical_cutoff_for=_vertical_rule(filter_options), solve_dtype=filter_options.solve_dtype)
    flat.value = flat.simbar.copy()
    config = PointLetkfConfig(rtps_alpha=filter_options.rtps_alpha, max_local_obs=filter_options.max_local_obs)
    diag = PointLetkfDiagnostics()
    increments = analyze_points(prior, flat, geometry, config, diag)
    to_numpy = ensemble.transform.backend.to_numpy
    out = {}
    for name, inc in increments.items():
        mean_inc = np.asarray(to_numpy(inc.mean(axis=0)), dtype=np.float64)
        scale = float(np.max(np.abs(np.asarray(to_numpy(prior[name]), dtype=np.float64))))
        out[name] = {
            "max_abs_mean_increment": float(np.max(np.abs(mean_inc))),
            "field_scale": scale,
            "relative": float(np.max(np.abs(mean_inc)) / scale) if scale > 0 else 0.0,
        }
    out["innovation_maxabs"] = float(np.max(np.abs(np.asarray(to_numpy(flat.value - flat.simbar)))))
    out["active_points"] = diag.active_points
    bar = rounding_bar(next(iter(prior.values())).dtype, AGREE_BAR_EPSILONS)
    worst = max(v["relative"] for k, v in out.items() if isinstance(v, dict))
    out.update(bar)
    out["worst_relative_mean_increment"] = float(worst)
    out["worst_in_epsilons"] = float(worst / bar["epsilon"])
    out["passed"] = bool(worst <= bar["bar_relative"])
    return out


def single_observation_family(ensemble: GlobalEnsemble, filter_options: FilterOptions, *,
                              variable: str = "temperature_k", latitude_deg: float = 40.0,
                              longitude_deg: float = 260.0, level_pa: float = 50000.0,
                              innovation: float = 2.0, error: float = 1.0) -> dict[str, object]:
    """One planted report against the analytic localised gain at every
    gridpoint of every analysis field."""
    prior, geometry = _grid_prior(ensemble, filter_options)
    transform = ensemble.transform
    xp = transform.backend.xp
    to_numpy = transform.backend.to_numpy
    operators = MemberOperators.for_model(ensemble.model, transform, ensemble.cfg)
    values, ln_pressure = operators.evaluate(
        ensemble.members, np.array([latitude_deg]), np.array([longitude_deg]), np.zeros(1),
        np.array([level_pa]))
    sim = values[variable]                                       # (R, 1)
    y = float(sim.mean()) + float(innovation)
    batch = PointObs("planted", variable, [latitude_deg], [longitude_deg], [math.log(level_pa)],
                     [False], [y], [error], simulated=sim)
    hcut_m = filter_options.horizontal_cutoff_km * 1000.0
    flat = flatten_batches([batch], xp, horizontal_cutoff_m=hcut_m,
                           vertical_cutoff_for=_vertical_rule(filter_options),
                           solve_dtype=filter_options.solve_dtype)
    config = PointLetkfConfig(rtps_alpha=0.0, max_local_obs=filter_options.max_local_obs)
    diag = PointLetkfDiagnostics()
    increments = analyze_points(prior, flat, geometry, config, diag)
    # The analytic weights on the host.
    from .letkf_point import _geodesic
    grid = transform.grid
    lat_g, lon_g = np.meshgrid(np.deg2rad(grid.latitude_deg), np.deg2rad(grid.longitude_deg), indexing="ij")
    dist = _geodesic(math.radians(latitude_deg), math.radians(longitude_deg), lat_g, lon_g,
                     float(grid.radius_m), np)
    wh = np.asarray(gaspari_cohn(dist / hcut_m, 1.0))
    vcut = float(filter_options.vertical_cutoff_lnp)
    ln_p_full = np.asarray(to_numpy(geometry.ln_p_full), dtype=np.float64)
    ln_ps = np.asarray(to_numpy(geometry.ln_ps), dtype=np.float64)
    w3 = wh[None] * np.asarray(gaspari_cohn(np.abs(ln_p_full - math.log(level_pa)) / vcut, 1.0))
    w2 = wh * np.asarray(gaspari_cohn(np.abs(ln_ps - math.log(level_pa)) / vcut, 1.0))
    sim_np = np.asarray(sim[:, 0], dtype=np.float64)
    d = y - float(sim_np.mean())
    out = {"innovation": d, "active_points": diag.active_points}
    worst = 0.0
    for name, inc in increments.items():
        prior_np = np.asarray(to_numpy(prior[name]), dtype=np.float64)
        w = w3 if prior_np.ndim == 4 else w2
        analytic = single_observation_increment(prior_np, sim_np, y, error, w)
        got = np.asarray(to_numpy(inc.mean(axis=0)), dtype=np.float64)
        scale = float(np.max(np.abs(got))) if np.max(np.abs(got)) > 0 else 1.0
        rel = float(np.max(np.abs(analytic - got)) / scale)
        beyond = bool(np.all(got[w == 0.0] == 0.0)) and bool(np.all(np.asarray(to_numpy(inc))[:, w == 0.0] == 0.0))
        out[name] = {"max_abs_increment": float(np.max(np.abs(got))),
                     "max_relative_difference_from_analytic": rel,
                     "bitwise_zero_beyond_cutoff": beyond}
        worst = max(worst, rel)
    out["worst_relative_difference"] = worst
    bar = rounding_bar(next(iter(prior.values())).dtype, SINGLE_BAR_EPSILONS)
    out.update(bar)
    out["worst_in_epsilons"] = float(worst / bar["epsilon"])
    out["passed"] = bool(worst <= bar["bar_relative"] and all(
        v["bitwise_zero_beyond_cutoff"] for k, v in out.items() if isinstance(v, dict) and "bitwise_zero_beyond_cutoff" in v))
    return out


def _vertical_rule(filter_options: FilterOptions):
    def rule(batch, surface):
        if batch.vertical_cutoff_lnp is not None:
            return np.full(batch.count, float(batch.vertical_cutoff_lnp))
        out = np.empty(batch.count)
        for is_surface in (True, False):
            cut = filter_options.vertical_cutoff_for(batch.variable, is_surface)
            out[surface == is_surface] = math.inf if cut is None else float(cut)
        return out
    return rule


def _grid_prior(ensemble: GlobalEnsemble, filter_options: FilterOptions):
    fields, ln_p_full, ln_ps = ensemble.grid_fields(tuple(filter_options.analysis_fields))
    grid = ensemble.transform.grid
    geometry = ColumnGeometry(latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg,
                              ln_p_full=ln_p_full, ln_ps=ln_ps, radius_m=float(grid.radius_m))
    return fields, geometry


def run_analytic_families(setup: GlobalOsseSetup, outdir: str | Path, *, progress=print) -> dict[str, object]:
    """``agree`` and ``single`` on an ensemble built from the config's
    initial state after the spin-up, at the ensemble truncation."""
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    _control_cfg, cfg = _configs(setup)
    transform = build_transform(cfg)
    model, cold = build_model_and_cold_state(cfg, transform, scratch_destination=output)
    dt_s = float(cfg.dt_s)
    state = cold
    mass = model.initialize_mass_target(state.atmosphere)
    water = model.initialize_water_target(state)
    for _ in range(int(round(setup.spinup_s / dt_s))):
        model.set_conservation_targets(mass, water)
        state, _m = model.step(state, dt_s)
        model.release_syntheses()
    options = EnsembleOptions(members=int(setup.members), truncation=int(cfg.truncation), seed=int(setup.seed))
    filter_options = FilterOptions(
        horizontal_cutoff_km=setup.horizontal_cutoff_km, vertical_cutoff_lnp=setup.vertical_cutoff_lnp,
        surface_vertical_cutoff_lnp=setup.surface_vertical_cutoff_lnp, rtps_alpha=setup.rtps_alpha,
        max_local_obs=setup.max_local_obs, thinning=False, gate_minimum_count=setup.gate_minimum_count)
    ensemble = GlobalEnsemble.from_state(cfg, model, transform, state, options)
    out: dict[str, object] = {"schema": OSSE_SCHEMA, "family": setup.family, "setup": asdict(setup),
                              "truncation": int(cfg.truncation)}
    if setup.family == "single":
        out["single"] = single_observation_family(ensemble, filter_options)
        progress(f"single-observation family: worst relative difference "
                 f"{out['single']['worst_relative_difference']:.2e} "
                 f"({out['single']['worst_in_epsilons']:.2f} {out['single']['state_dtype']} epsilons, "
                 f"bar {out['single']['bar_in_epsilons']:.0f}), passed {out['single']['passed']}")
    else:
        operators = MemberOperators.for_model(model, transform, cfg)
        rng = np.random.default_rng([setup.seed, 11])
        network = _Network(setup, rng, lambda lat, lon: np.zeros(np.asarray(lat).size))
        moment = dt.datetime(2026, 9, 1, 0, 0, tzinfo=dt.timezone.utc)
        rows = network.rows(moment, float(setup.interval_s), rng, spread_times=False)
        batches = batches_from_rows(rows, operators, ensemble.members)
        out["agree"] = agreeing_observations_family(ensemble, batches, filter_options)
        progress(f"agreeing-observations family: worst relative mean increment "
                 f"{out['agree']['worst_relative_mean_increment']:.2e} "
                 f"({out['agree']['worst_in_epsilons']:.2f} {out['agree']['state_dtype']} epsilons, "
                 f"bar {out['agree']['bar_in_epsilons']:.0f}), passed {out['agree']['passed']}")
    (output / f"osse-{setup.family}.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return out


def run_sweep(setup: GlobalOsseSetup, outdir: str | Path, option: str, values: list, *, progress=print) -> dict:
    """The chosen family once per value of ``option``; one comparison
    report (``osse-sweep-<option>.json``) with every arm's verdict."""
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    arms = {}
    for value in values:
        arm_setup = dataclasses.replace(setup, **{option: value})
        arm_dir = output / f"{option}-{value}"
        progress(f"osse sweep: {option} = {value!r}")
        report = run_global_osse(arm_setup, arm_dir, progress=progress) if setup.family in ("recovery", "pull", "transfer") \
            else run_analytic_families(arm_setup, arm_dir, progress=progress)
        arms[str(value)] = {"verdict": report.get("verdict"), "initial_score": report.get("initial_score"),
                            "cycles": [{k: c[k] for k in ("cycle", "before", "after", "analysis_wall_s")}
                                       for c in report.get("cycles", [])],
                            "dir": str(arm_dir)}
    out = {"schema": OSSE_SCHEMA, "sweep": option, "values": [str(v) for v in values], "family": setup.family,
           "setup": asdict(setup), "arms": arms}
    (output / f"osse-sweep-{option}.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return out


def _parse_value(option: str, text: str):
    if text.lower() in ("none", "null"):
        return None
    if option in ("increment_application", "wind_balance", "family", "recentering_mode", "static_covariance"):
        return text
    if option in ("thinning", "recenter_preserve_mass"):
        return text.lower() in ("1", "true", "yes", "on")
    if option in ("members", "cycles", "max_local_obs", "gate_minimum_count", "truncation",
                  "control_truncation", "transfer_taper_start_degree", "transfer_taper_end_degree", "seed",
                  "static_samples"):
        return int(text)
    return float(text)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m woof.globe.da.osse",
        description="WOOF global ensemble filter twin experiment, calibration families and sweeps")
    p.add_argument("--config", required=True, help="WOOF global run config (re-cut by --truncation and --control-truncation)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--family", choices=FAMILIES, default="recovery")
    p.add_argument("--members", type=int, default=GlobalOsseSetup.members)
    p.add_argument("--cycles", type=int, default=GlobalOsseSetup.cycles)
    p.add_argument("--interval-s", type=float, default=GlobalOsseSetup.interval_s)
    p.add_argument("--spinup-s", type=float, default=GlobalOsseSetup.spinup_s)
    p.add_argument("--start-perturbation-scale", type=float, default=GlobalOsseSetup.start_perturbation_scale)
    p.add_argument("--stations", default=None, help="obs table (iem-asos or metar CSV) whose positions the synthetic stations take")
    p.add_argument("--soundings", default=None, help="igra2-levels CSV whose sites and levels the synthetic soundings take")
    p.add_argument("--synthetic-stations", type=int, default=GlobalOsseSetup.synthetic_stations)
    p.add_argument("--synthetic-soundings", type=int, default=GlobalOsseSetup.synthetic_soundings)
    p.add_argument("--amv-points", type=int, default=GlobalOsseSetup.amv_points)
    p.add_argument("--seed", type=int, default=GlobalOsseSetup.seed)
    p.add_argument("--horizontal-cutoff-km", type=float, default=GlobalOsseSetup.horizontal_cutoff_km)
    p.add_argument("--vertical-cutoff-lnp", type=float, default=GlobalOsseSetup.vertical_cutoff_lnp)
    p.add_argument("--surface-vertical-cutoff-lnp", type=float, default=GlobalOsseSetup.surface_vertical_cutoff_lnp)
    p.add_argument("--rtps-alpha", type=float, default=GlobalOsseSetup.rtps_alpha)
    p.add_argument("--wind-balance", choices=("rotational", "unconstrained"), default=GlobalOsseSetup.wind_balance)
    p.add_argument("--additive-inflation-fraction", type=float, default=GlobalOsseSetup.additive_inflation_fraction)
    p.add_argument("--no-thinning", action="store_true")
    p.add_argument("--max-local-obs", type=int, default=GlobalOsseSetup.max_local_obs)
    p.add_argument("--gate-minimum-count", type=int, default=GlobalOsseSetup.gate_minimum_count)
    p.add_argument("--dt-s", type=float, default=None, help="the ensemble time step (default: ensemble_config's)")
    p.add_argument("--truncation", type=int, default=None, help="the ENSEMBLE truncation (default: the config's)")
    p.add_argument("--control-truncation", type=int, default=None,
                   help="the nature run's and the control's truncation (default: the ensemble's)")
    p.add_argument("--control-dt-s", type=float, default=None)
    p.add_argument("--observation-time-bin-s", type=float, default=None,
                   help="observe reports at their own times in bins this wide (default: the analysis instant)")
    p.add_argument("--increment-application", choices=("direct", "iau"), default="direct")
    p.add_argument("--recentering-fraction", type=float, default=1.0)
    p.add_argument("--recentering-mode", choices=("increment", "state"), default="increment")
    p.add_argument("--transfer-taper-start-degree", type=int, default=None)
    p.add_argument("--transfer-taper-end-degree", type=int, default=None)
    p.add_argument("--no-recenter-preserve-mass", action="store_true",
                   help="take the recentring shift raw (the first twins' form; the members' global-mean surface pressure then moves)")
    p.add_argument("--hybrid-beta", type=float, default=1.0,
                   help="the ensemble weight of the hybrid covariance in the control's gain (1.0: the ensemble alone)")
    p.add_argument("--static-covariance", default="packaged",
                   help="the static covariance table for --hybrid-beta below one (a path or 'packaged')")
    p.add_argument("--static-samples", type=int, default=64, help="static draws per analysis")
    p.add_argument("--sweep", default=None, help="option=a,b,c: repeat the family for each value")
    a = p.parse_args(argv)
    setup = GlobalOsseSetup(
        config=a.config, members=a.members, cycles=a.cycles, interval_s=a.interval_s, spinup_s=a.spinup_s,
        start_perturbation_scale=a.start_perturbation_scale, stations=a.stations, soundings=a.soundings,
        synthetic_stations=a.synthetic_stations, synthetic_soundings=a.synthetic_soundings,
        amv_points=a.amv_points, seed=a.seed, family=a.family,
        horizontal_cutoff_km=a.horizontal_cutoff_km, vertical_cutoff_lnp=a.vertical_cutoff_lnp,
        surface_vertical_cutoff_lnp=a.surface_vertical_cutoff_lnp, rtps_alpha=a.rtps_alpha,
        wind_balance=a.wind_balance, additive_inflation_fraction=a.additive_inflation_fraction,
        thinning=not a.no_thinning, max_local_obs=a.max_local_obs, gate_minimum_count=a.gate_minimum_count,
        dt_s=a.dt_s, truncation=a.truncation, control_truncation=a.control_truncation, control_dt_s=a.control_dt_s,
        observation_time_bin_s=a.observation_time_bin_s, increment_application=a.increment_application,
        recentering_fraction=a.recentering_fraction, recentering_mode=a.recentering_mode,
        transfer_taper_start_degree=a.transfer_taper_start_degree,
        transfer_taper_end_degree=a.transfer_taper_end_degree,
        recenter_preserve_mass=not a.no_recenter_preserve_mass,
        hybrid_beta=float(a.hybrid_beta), static_covariance=a.static_covariance, static_samples=int(a.static_samples),
    )
    if a.sweep:
        option, _, text = a.sweep.partition("=")
        values = [_parse_value(option, v) for v in text.split(",") if v]
        run_sweep(setup, a.outdir, option, values)
    elif setup.family in ("agree", "single"):
        run_analytic_families(setup, a.outdir)
    else:
        run_global_osse(setup, a.outdir)
    return 0


__all__ = [
    "FAMILIES",
    "OSSE_SCHEMA",
    "SPREAD_BAND",
    "STREAM_ERRORS",
    "pooled_normalised_fit",
    "GlobalOsseSetup",
    "agreeing_observations_family",
    "main",
    "run_analytic_families",
    "run_global_osse",
    "run_sweep",
    "score",
    "score_control",
    "single_observation_family",
]


if __name__ == "__main__":
    raise SystemExit(main())
