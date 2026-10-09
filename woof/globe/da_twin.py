"""The control twin: perfect-model synthetic cycling of the DA door's
control path at two resolutions.

The validation ladder of design amendment I puts perfect-model synthetic
cycling between the numerical tests and the real case.  The ensemble
package's own twin (:mod:`woof.globe.da.osse`) gates the members'
LETKF on one grid; this twin gates what the door hands back, the CONTROL
analysed at its own truncation through the ensemble covariance at a lower
one (amendment A), on the model itself:

* a NATURE run of the configured core at the control truncation (the
  truth), spun up and then integrated across the cycles;
* the control started DISPLACED from the truth at the first analysis
  time (a spectral perturbation draw scaled by
  ``start_perturbation_scale``), the ensemble built around the control
  restricted to the ensemble truncation;
* synthetic reports of every stream drawn from the truth at the network's
  positions (stations on the terrain, soundings at the mandatory levels,
  motion vectors at four pressure levels) with the streams' stated
  errors;
* every cycle: the control steps to the analysis instant through the
  door's own filter interface (``begin_window`` / ``observe`` /
  ``analyse``), the members catch up inside the filter, the control is
  analysed, and the control's grid rmse against the truth is scored
  before and after, area-weighted, per field, with the ensemble spread
  beside it.

Three families (``family``), both directions:

``recovery``
    Noisy reports; the control's rmse against the truth must fall over
    the cycles (the verdict reads the first and the last).
``perfect``
    Perfect reports (values are H(truth), the errors kept) must pull the
    control to the truth: rmse falls at every analysis, on temperature and
    wind.
``agree``
    Reports equal to the control's own H(x_H^b) must move the control by
    nothing: the innovation is exactly zero, the mean weight vector is
    exactly zero, and the control's change is the rounding residual of
    the zero-mean perturbation transform, bounded at 1e-10 of the field's
    own scale.

``mirror``
    Reports drawn from the MIRROR of the truth about the control
    (``2 x_control - x_truth``, no noise): the same network and errors as
    the perfect family, pointing the other way.  The filter must follow
    them: the control's rmse against the truth must RISE at every
    analysis, on temperature and wind, by an amount within a factor of
    three of what the perfect family's reports take off.  The fourth
    family closes the "both directions" claim: a filter that only ever
    reads "improved" because it damps every increment would pass the
    perfect family weakly and fail this one.

``increment_source`` selects the analysis path (``control``, the default,
or ``ensemble-mean``, the comparison arm of amendment A), so the two can
be run on the same truth, network and seed and their recovery curves laid
beside each other; the verdict never says which is better, the curves do.

Runnable: ``python -m woof.globe.da_twin --config <toml> --outdir
<dir> [--control-truncation T --ensemble-truncation T --members N --cycles
C --family recovery|perfect|agree --increment-source control|ensemble-mean]``.
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

from .assimilate import AssimilationOptions
from .config import load_config
from .constants import SPECTRAL_FIELDS
from .da.ensemble import GlobalEnsemble, truncate_spectral
from .da.operators import MemberOperators, evaluate_batches
from .da.options import EnsembleOptions, FilterOptions
from .da.osse import _Network, _area_rms, _state_fields as _truth_fields
from .da.window import batches_unevaluated
from .obs_table import ObsRow
from .da.perturbations import draw_perturbation, member_rng, perturbed_state
from .da_control import ControlOptions
from .da_filter import LetkfFilter
from .runner import build_model_and_cold_state, build_transform
from .state import ArwenGlobalState

TWIN_SCHEMA = "gpuwm.arwen-global-da-control-twin/v1"
FAMILIES = ("recovery", "perfect", "agree", "mirror")
#: The agreeing family's bar: the control's change relative to the
#: field's own grid rms.
AGREE_RELATIVE_BOUND = 1.0e-10


@dataclass(frozen=True)
class ControlTwinSetup:
    """Everything that defines one control twin."""

    config: str
    control_truncation: int | None = None
    ensemble_truncation: int = 3
    members: int = 6
    cycles: int = 3
    interval_s: float = 20.0
    spinup_s: float = 20.0
    start_perturbation_scale: float = 2.0
    perturbation_balance: str = "linear"
    perturbation_ln_surface_pressure: float | None = None
    perturbation_wind_m_s: float | None = None
    seed: int = 20260906
    family: str = "recovery"
    increment_source: str = "control"
    recentre_fraction: float = 1.0
    taper_full_degree: int | None = None
    taper_zero_degree: int | None = None
    synthetic_stations: int = 120
    synthetic_soundings: int = 24
    amv_points: int = 60
    horizontal_cutoff_km: float = 6000.0
    gate_minimum_count: int = 50
    wind_balance: str = "rotational"
    dt_s: float | None = None

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if int(self.cycles) < 1 or int(self.members) < 3:
            raise ValueError("cycles must be >= 1 and members >= 3")


def _control_config(setup: ControlTwinSetup):
    cfg = load_config(setup.config)
    if setup.control_truncation is not None and int(setup.control_truncation) != int(cfg.truncation):
        cfg = dataclasses.replace(
            cfg, name=f"{cfg.name}-twin-t{int(setup.control_truncation)}",
            truncation=int(setup.control_truncation), nlat=None, nlon=None,
            dt_s=float(setup.dt_s) if setup.dt_s is not None else float(cfg.dt_s),
        )
    elif setup.dt_s is not None:
        cfg = dataclasses.replace(cfg, dt_s=float(setup.dt_s))
    if int(setup.ensemble_truncation) > int(cfg.truncation):
        raise ValueError(
            f"the ensemble truncation T{setup.ensemble_truncation} exceeds the control's T{cfg.truncation}"
        )
    return cfg


def _time_of(start: dt.datetime, time_s: float) -> dt.datetime:
    return start + dt.timedelta(seconds=float(time_s))


def _filled_rows(network: _Network, operators, state, moment: dt.datetime, window_s: float, rng, *,
                 noisy: bool) -> list[ObsRow]:
    """The network's reports of ``state`` at ``moment``: the placeholder rows
    the network lays out, their values the operators' H(state), plus the
    stated error's noise when ``noisy``; rows the operator cannot evaluate
    (a surface pressure aloft) are dropped."""
    placeholders = network.rows(moment, float(window_s), rng, spread_times=False)
    batches = batches_unevaluated(placeholders, operators)
    evaluate_batches(operators, [state], batches, target="simulated")
    out: list[ObsRow] = []
    for batch in batches:
        values = np.asarray(batch.simulated, dtype=np.float64)[0]
        noise = rng.normal(0.0, 1.0, batch.count) if noisy else np.zeros(batch.count)
        for j in range(batch.count):
            if not np.isfinite(values[j]):
                continue
            out.append(ObsRow(
                batch.stream, f"{batch.stream}-{j:05d}", float(batch.latitude_deg[j]), float(batch.longitude_deg[j]),
                float(batch.elevation_m[j]), None if batch.surface[j] else float(np.exp(batch.ln_pressure[j])),
                batch.valid_time[j], batch.variable, float(values[j] + noise[j] * float(batch.error[j])),
                float(batch.error[j]),
            ))
    return out


def mirror_state(model, control: ArwenGlobalState, truth: ArwenGlobalState) -> ArwenGlobalState:
    """The truth reflected through the control: spectral fields
    ``2 c - t`` (the control's own surface, tracers and physics), the vapor
    repaired and the state enforced, so reports drawn from it point the
    control away from the truth by the displacement it started with."""
    fields = [
        2.0 * getattr(control.atmosphere, name) - getattr(truth.atmosphere, name)
        for name in SPECTRAL_FIELDS
    ]
    state = ArwenGlobalState(control.atmosphere.with_fields(fields), control.surface, control.physics_state.copy())
    state.atmosphere.time_s = control.atmosphere.time_s
    state.atmosphere.step = control.atmosphere.step
    state, _n, _t, _f = model._repair_positivity(state)
    model.enforce(state)
    model.release_syntheses()
    return state


def score_control(model, transform, control: ArwenGlobalState, truth: ArwenGlobalState) -> dict[str, float]:
    """Area-weighted grid rmse of the control against the truth, per field."""
    grid = transform.grid
    truth_f = _truth_fields(model, transform, truth)
    control_f = _truth_fields(model, transform, control)
    return {name: _area_rms(grid, control_f[name] - truth_f[name]) for name in truth_f}


def _field_scales(model, transform, state: ArwenGlobalState) -> dict[str, float]:
    grid = transform.grid
    fields = _truth_fields(model, transform, state)
    return {name: max(_area_rms(grid, value), 1.0e-30) for name, value in fields.items()}


def _control_change(model, transform, before: ArwenGlobalState, after: ArwenGlobalState) -> dict[str, float]:
    grid = transform.grid
    a = _truth_fields(model, transform, before)
    b = _truth_fields(model, transform, after)
    return {name: _area_rms(grid, b[name] - a[name]) for name in a}


def run_control_twin(setup: ControlTwinSetup, outdir: str | Path, *, progress=print) -> dict[str, object]:
    """The twin, whole."""
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = _control_config(setup)
    transform = build_transform(cfg)
    model, cold = build_model_and_cold_state(cfg, transform, scratch_destination=output)
    start = dt.datetime(2026, 9, 1, 0, 0, tzinfo=dt.timezone.utc)
    dt_s = float(cfg.dt_s)
    for value, label in ((setup.interval_s, "interval_s"), (setup.spinup_s, "spinup_s")):
        if abs(value / dt_s - round(value / dt_s)) > 1.0e-9:
            raise ValueError(f"{label} must be a whole number of the config's dt_s")
    steps_spinup = int(round(setup.spinup_s / dt_s))
    steps_interval = int(round(setup.interval_s / dt_s))

    # The nature run.
    t0 = time.perf_counter()
    nature = cold
    nature_mass = model.initialize_mass_target(nature.atmosphere)
    nature_water = model.initialize_water_target(nature)
    for _ in range(steps_spinup):
        model.set_conservation_targets(nature_mass, nature_water)
        nature, _m = model.step(nature, dt_s)
        model.release_syntheses()
    truths = [nature.copy()]
    for _c in range(int(setup.cycles) - 1):
        for _ in range(steps_interval):
            model.set_conservation_targets(nature_mass, nature_water)
            nature, _m = model.step(nature, dt_s)
            model.release_syntheses()
        truths.append(nature.copy())
    nature_wall = time.perf_counter() - t0
    progress(f"twin: nature run {len(truths)} truths at T{cfg.truncation} in {nature_wall:.1f} s")

    # The control, displaced from the truth at the first analysis time.
    amplitudes = {
        key: float(value) for key, value in (
            ("perturbation_ln_surface_pressure", setup.perturbation_ln_surface_pressure),
            ("perturbation_wind_m_s", setup.perturbation_wind_m_s),
        ) if value is not None
    }
    # The displacement is drawn by the setup's own family (the twin's
    # readings are taken against one family, ensemble and displacement).
    det_options = EnsembleOptions(members=int(setup.members), truncation=int(cfg.truncation), seed=int(setup.seed),
                                  additive_inflation_fraction=0.0, perturbation_balance=str(setup.perturbation_balance),
                                  **amplitudes)
    displaced_inc, displacement = draw_perturbation(
        model, transform, truths[0].atmosphere, det_options, member_rng(setup.seed + 1, 0, "displacement"),
        amplitude_scale=float(setup.start_perturbation_scale))
    control = perturbed_state(model, transform, truths[0], displaced_inc)
    control.atmosphere.time_s = truths[0].atmosphere.time_s
    control.atmosphere.step = truths[0].atmosphere.step
    control_mass = model.initialize_mass_target(control.atmosphere)
    control_water = model.initialize_water_target(control)

    # The filter and its ensemble around the control at the ensemble truncation.
    ensemble_options = EnsembleOptions(
        members=int(setup.members), truncation=int(setup.ensemble_truncation), seed=int(setup.seed),
        additive_inflation_fraction=0.0, perturbation_balance=str(setup.perturbation_balance), **amplitudes,
    )
    # Direct insertion by name: the twin reads the control and the members
    # at the analysis instant (the incremental update, the door's default,
    # would leave the members the background until the window's steps).
    filter_options = FilterOptions(
        horizontal_cutoff_km=float(setup.horizontal_cutoff_km), wind_balance=setup.wind_balance,
        gate_minimum_count=int(setup.gate_minimum_count), increment_application="direct",
    )
    control_options = ControlOptions(
        increment_source=setup.increment_source, recentre_fraction=float(setup.recentre_fraction),
        taper_full_degree=setup.taper_full_degree, taper_zero_degree=setup.taper_zero_degree,
    )
    filt = LetkfFilter(ensemble_options, filter_options, control_options)
    ens_cfg, ens_model, ens_transform, ens_cold = filt._build_ensemble_model(cfg, output / "ensemble")
    ens_t = int(ens_transform.truncation)
    xp = ens_transform.backend.xp
    base_fields = [
        xp.asarray(truncate_spectral(getattr(control.atmosphere, name), ens_t), dtype=ens_transform.backend.complex_dtype)
        for name in SPECTRAL_FIELDS
    ]
    base = ArwenGlobalState(ens_cold.atmosphere.with_fields(base_fields), ens_cold.surface, ens_cold.physics_state)
    base.atmosphere.time_s = control.atmosphere.time_s
    base.atmosphere.step = control.atmosphere.step
    base, _n, _t, _f = ens_model._repair_positivity(base)
    ens_model.enforce(base)
    filt.ensemble = GlobalEnsemble.from_state(ens_cfg, ens_model, ens_transform, base, ensemble_options)
    filt._store = output / "ensemble"

    operators = MemberOperators.for_model(model, transform, cfg)
    rng = np.random.default_rng([setup.seed, 11])

    def terrain_height_at(lat, lon):
        from woof.globe.spectral.sampling import sample_scalar
        return sample_scalar(transform, operators.terrain, lat, lon) / 9.80665

    network = _Network(_network_setup(setup), rng, terrain_height_at)
    initial = score_control(model, transform, control, truths[0])
    progress(f"twin: displaced control rmse T {initial['temperature_k']:.3f} K, u {initial['u']:.3f} m/s, "
             f"ps {initial['surface_pressure_pa']:.1f} Pa; {filt.ensemble.size} members at T{ens_t}")

    noisy = setup.family == "recovery"
    cycles_record = []
    for c, truth in enumerate(truths):
        moment = _time_of(start, truth.time_s)
        step_wall = 0.0
        if c > 0:
            t_s = time.perf_counter()
            filt.begin_window(cfg, model, transform, [], float(control.time_s), float(truth.time_s), start)
            for _ in range(steps_interval):
                model.set_conservation_targets(control_mass, control_water)
                control, _m = model.step(control, dt_s)
                model.release_syntheses()
                filt.observe(control, float(control.time_s))
            step_wall = time.perf_counter() - t_s
        before = score_control(model, transform, control, truth)
        if setup.family == "agree":
            rows = _filled_rows(network, operators, control, moment, setup.interval_s, rng, noisy=False)
        elif setup.family == "mirror":
            rows = _filled_rows(
                network, operators, mirror_state(model, control, truth), moment, setup.interval_s, rng, noisy=False)
        else:
            rows = _filled_rows(network, operators, truth, moment, setup.interval_s, rng, noisy=noisy)
        t_a = time.perf_counter()
        background = {"path": None, "self_sha256": f"twin-cycle-{c}", "step": int(control.step),
                      "time_s": float(control.time_s)}
        analysis, report, phases = filt.analyse(
            cfg, model, transform, [control], rows, sources=[], background=background,
            analysis_time=moment, options=AssimilationOptions(),
        )
        analysis_wall = time.perf_counter() - t_a
        after = score_control(model, transform, analysis, truth)
        change = _control_change(model, transform, control, analysis)
        scales = _field_scales(model, transform, control)
        relative_change = {name: change[name] / scales[name] for name in change}
        control = analysis
        control_mass = model.initialize_mass_target(control.atmosphere)
        control_water = model.initialize_water_target(control)
        card = report["scorecard"]
        streams = {
            source: {
                variable: {
                    "n": row["regions"]["global"]["n"],
                    "o_minus_b_rms": row["regions"]["global"]["o_minus_b"]["rms"],
                    "o_minus_a_rms": row["regions"]["global"]["o_minus_a"]["rms"],
                    "o_a_below_o_b": row["assessments"]["statistical_consistency"]["o_a_rms_below_o_b_rms"],
                    "desroziers_ratio": row["regions"]["global"]["consistency"]["desroziers_ratio"],
                }
                for variable, row in table["variables"].items()
            }
            for source, table in card["streams"].items()
        }
        bounds = ((report.get("assessments") or {}).get("engineering") or {}).get("analysed_state_bounds") or {}
        truth_ps = _truth_fields(model, transform, truth)["surface_pressure_pa"]
        record = {
            "cycle": c, "time_s": float(truth.time_s), "analysis_time_utc": moment.isoformat(timespec="seconds"),
            "status": report["status"], "reports": len(rows), "assimilated": report["assimilated_total"],
            "analysed_state_bounds": bounds,
            "truth_surface_pressure_pa": {"min": float(truth_ps.min()), "max": float(truth_ps.max())},
            "before": before, "after": after, "control_change": change, "control_change_relative": relative_change,
            "spread": report["spread"], "streams": streams,
            "control": {k: v for k, v in report["control"].items()
                        if k in ("increment_source", "increment", "taper", "route")},
            "mean_increment_transfer": report["mean_increment_transfer"],
            "recentre": report["recentre"],
            "forecast_wall_s": step_wall, "analysis_wall_s": analysis_wall, "phases_s": phases,
        }
        cycles_record.append(record)
        progress(
            f"twin cycle {c}: control rmse T {before['temperature_k']:.4f} -> {after['temperature_k']:.4f} K, "
            f"u {before['u']:.4f} -> {after['u']:.4f} m/s, ps {before['surface_pressure_pa']:.2f} -> "
            f"{after['surface_pressure_pa']:.2f} Pa; spread T {report['spread']['after']['temperature_k']:.4f} K; "
            f"{analysis_wall:.1f} s analysis; status {report['status']}; analysed ps "
            f"{bounds.get('surface_pressure_min_pa', float('nan')):.0f} to {bounds.get('surface_pressure_max_pa', float('nan')):.0f} Pa "
            f"(truth {truth_ps.min():.0f} to {truth_ps.max():.0f} Pa)"
        )
    report = {
        "schema": TWIN_SCHEMA,
        "family": setup.family,
        "setup": asdict(setup),
        "control": {"config_hash": cfg.config_hash, "truncation": int(cfg.truncation), "nlev": int(model.nlev), "dt_s": dt_s},
        "ensemble": {"config_hash": ens_cfg.config_hash, "truncation": ens_t, "members": filt.ensemble.size,
                     "dt_s": float(ens_cfg.dt_s), "resident_bytes": filt.ensemble.resident_bytes()},
        "options": {"ensemble": ensemble_options.identity(), "filter": filter_options.identity(),
                    "control": control_options.identity()},
        "network": network.describe(),
        "displacement": {"scale": float(setup.start_perturbation_scale), "record": displacement},
        "initial_score": initial,
        "cycles": cycles_record,
        "nature_wall_s": nature_wall,
        "verdict": _verdict(setup.family, initial, cycles_record),
    }
    (output / f"control-twin-{setup.family}-{setup.increment_source}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    progress(f"twin: verdict {report['verdict']}")
    return report


def _network_setup(setup: ControlTwinSetup):
    """The ensemble twin's network takes its own setup type; this adapter
    hands it the fields it reads."""
    from .da.osse import GlobalOsseSetup

    return GlobalOsseSetup(
        config=setup.config, members=int(setup.members), cycles=int(setup.cycles),
        synthetic_stations=int(setup.synthetic_stations), synthetic_soundings=int(setup.synthetic_soundings),
        amv_points=int(setup.amv_points), seed=int(setup.seed),
    )


def _verdict(family: str, initial: dict, cycles: list[dict]) -> dict[str, object]:
    rmse_t = [c["after"]["temperature_k"] for c in cycles]
    rmse_u = [c["after"]["u"] for c in cycles]
    every_analysis_improved_t = all(c["after"]["temperature_k"] < c["before"]["temperature_k"] for c in cycles)
    every_analysis_improved_u = all(c["after"]["u"] < c["before"]["u"] for c in cycles)
    # The agreeing bar is read on the fields the filter analyses through
    # the transform (temperature, wind, surface pressure); the vapor is
    # reported beside them because the model's positivity repair runs
    # after every increment, a zero one included, and on a float32 state
    # it moves the vapor by its own clip (measured 1e-5 relative at T127),
    # which is the repair's action, not the filter's.
    dynamical = ("temperature_k", "u", "v", "surface_pressure_pa")
    largest_relative_change = max(
        (v for c in cycles for k, v in c["control_change_relative"].items() if k in dynamical), default=0.0)
    largest_qv_change = max((c["control_change_relative"].get("qv", 0.0) for c in cycles), default=0.0)
    out: dict[str, object] = {
        "family": family,
        "temperature_rmse_first_to_last": [initial["temperature_k"], rmse_t[-1]],
        "wind_rmse_first_to_last": [initial["u"], rmse_u[-1]],
        "every_analysis_improved_temperature": every_analysis_improved_t,
        "every_analysis_improved_wind": every_analysis_improved_u,
        "largest_relative_control_change": largest_relative_change,
        "largest_relative_control_change_fields": list(dynamical),
        "largest_relative_vapor_change": largest_qv_change,
        "vapor_note": (
            "the model's positivity repair runs after every increment, a zero one included; "
            "its clip of the vapor is recorded here and is not the filter's increment"
        ),
    }
    if family == "recovery":
        out["passed"] = bool(rmse_t[-1] < initial["temperature_k"])
        out["bar"] = "the control's temperature rmse against the truth falls from the displaced start to the last analysis"
    elif family == "perfect":
        out["passed"] = bool(every_analysis_improved_t and every_analysis_improved_u)
        out["bar"] = "perfect reports pull the control toward the truth at every analysis, on temperature and wind"
    elif family == "mirror":
        every_analysis_worsened_t = all(c["after"]["temperature_k"] > c["before"]["temperature_k"] for c in cycles)
        every_analysis_worsened_u = all(c["after"]["u"] > c["before"]["u"] for c in cycles)
        out["every_analysis_worsened_temperature"] = every_analysis_worsened_t
        out["every_analysis_worsened_wind"] = every_analysis_worsened_u
        out["passed"] = bool(every_analysis_worsened_t and every_analysis_worsened_u)
        out["bar"] = (
            "reports drawn from the mirror of the truth about the control push the control away from "
            "the truth at every analysis, on temperature and wind (the filter follows its reports "
            "whichever way they point)"
        )
    else:
        out["passed"] = bool(largest_relative_change <= AGREE_RELATIVE_BOUND)
        out["bar"] = (
            f"reports equal to the control's own H(x) change the control's temperature, wind and "
            f"surface pressure by at most {AGREE_RELATIVE_BOUND:g} of the field's grid rms (the vapor's "
            "positivity repair recorded beside it)"
        )
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m woof.globe.da_twin",
        description="the control twin: perfect-model synthetic cycling of the DA door's control path",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--control-truncation", type=int, default=None)
    parser.add_argument("--ensemble-truncation", type=int, default=3)
    parser.add_argument("--members", type=int, default=6)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--interval-s", type=float, default=20.0)
    parser.add_argument("--spinup-s", type=float, default=20.0)
    parser.add_argument("--dt-s", type=float, default=None)
    parser.add_argument("--family", choices=FAMILIES, default="recovery")
    parser.add_argument("--increment-source", choices=("control", "ensemble-mean"), default="control")
    parser.add_argument("--recentre-fraction", type=float, default=1.0)
    parser.add_argument("--taper-full-degree", type=int, default=None)
    parser.add_argument("--taper-zero-degree", type=int, default=None)
    parser.add_argument("--start-perturbation-scale", type=float, default=2.0)
    parser.add_argument("--perturbation-balance", choices=("linear", "none"), default="linear",
                        help="the ensemble perturbation family: linear (the balanced default) or none "
                             "(the independent draws; the smoke-truncation twins use it, where a planetary "
                             "wind balances the stated pressure at a fraction of a metre per second)")
    parser.add_argument("--synthetic-stations", type=int, default=120)
    parser.add_argument("--synthetic-soundings", type=int, default=24)
    parser.add_argument("--amv-points", type=int, default=60)
    parser.add_argument("--horizontal-cutoff-km", type=float, default=6000.0)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args(argv)
    setup = ControlTwinSetup(
        config=args.config, control_truncation=args.control_truncation,
        ensemble_truncation=args.ensemble_truncation, members=args.members, cycles=args.cycles,
        interval_s=args.interval_s, spinup_s=args.spinup_s, dt_s=args.dt_s, family=args.family,
        increment_source=args.increment_source, recentre_fraction=args.recentre_fraction,
        taper_full_degree=args.taper_full_degree, taper_zero_degree=args.taper_zero_degree,
        start_perturbation_scale=args.start_perturbation_scale,
        perturbation_balance=args.perturbation_balance,
        synthetic_stations=args.synthetic_stations, synthetic_soundings=args.synthetic_soundings,
        amv_points=args.amv_points, horizontal_cutoff_km=args.horizontal_cutoff_km, seed=args.seed,
    )
    report = run_control_twin(setup, args.outdir)
    return 0 if report["verdict"]["passed"] else 1


__all__ = ["AGREE_RELATIVE_BOUND", "FAMILIES", "TWIN_SCHEMA", "ControlTwinSetup", "mirror_state", "run_control_twin", "score_control"]


if __name__ == "__main__":
    raise SystemExit(main())
