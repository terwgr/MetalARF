"""The analysis filters the DA door drives, behind one interface.

The door (``woof global da``) does not know how an increment is formed.
It steps the resident control state between analysis instants, tells the
filter when a window opens and after every step (so the filter can take
observation-space equivalents at the reports' own times), hands the
filter the reports at the analysis instant and takes back the control
analysis and a report; the filter owns the increment arithmetic, the
ensemble if it has one, and the ensemble manifest on disk.

Two filters:

* :class:`SuccessiveCorrectionFilter` (``successive-correction``): the
  deterministic v1 door of :mod:`woof.globe.assimilate`, one
  resident state, the increment formed by :func:`assimilate.analyse`,
  every report compared at the analysis instant (it has no trajectory
  record; the report says so).
* :class:`LetkfFilter` (``letkf``): the dual-resolution ensemble filter
  under the design amendments of 2026-09-06: N members at their own
  truncation resident in one process (:mod:`woof.globe.da`), the
  members' own LETKF analysis for the perturbations, and the CONTROL's
  own analysis from the control's innovations through the ensemble
  covariance (:mod:`woof.globe.da_control`, amendment A), with
  the transfer taper and increment-spectrum inspection (amendment C),
  observation-space equivalents at the reports' own times through the
  window (:mod:`woof.globe.da_window`, amendment B), the external
  analysis as a weak low-pass constraint (:mod:`arwen_global.
  da_anchor`, amendment E), RTPS as the one inflation mechanism by default
  (amendment D; additive inflation is an option switched on by name), and
  the four assessments in the report (amendment G).

Interface (recorded here because the door, the ensemble lane and the
observation lanes build to it):

``resident_states(deterministic) -> list[ArwenGlobalState]``
    Every state the door steps with the model between analyses, the
    control first (the ensemble filter keeps its members inside itself
    and steps them at their own time step during ``analyse``).

``begin_window(cfg, model, transform, rows, window_start_s, window_end_s,
start_utc)``
    The door opens the analysis window ``(t0, t1]`` with the reports it
    has fetched for it; the filter assigns them to time bins.

``observe(state, time_s)``
    After every control step: the filter evaluates its control operators
    for the reports whose bin instant is ``time_s`` and keeps the values.

``analyse(cfg, model, transform, states, rows, *, sources, background,
analysis_time, options) -> (control_analysis, report, phases)``
    The report carries ``scorecard`` (:mod:`woof.globe.da_scorecard`,
    the four assessments), ``lineage``, ``status`` (engineering validity)
    and the records named in :class:`LetkfFilter`.

``init(cfg, out_dir, deterministic, *, members, analysis_time_utc) ->
manifest`` and ``manifest(cfg, out_dir, deterministic, *,
analysis_time_utc, previous) -> dict``
    The ensemble manifest (:data:`ENSEMBLE_MANIFEST_SCHEMA`) after init and
    after every analysis.

The manifest schema ``gpuwm.arwen-global-da-ensemble/v1``:
``filter``, ``members`` (id, checkpoint, self_sha256), ``deterministic``
(checkpoint, self_sha256), ``config_hash``, ``truncation``,
``analysis_time_utc``, ``lineage`` (the checkpoint's assimilation chain
summary), ``created_utc``, ``previous_manifest_sha256`` and its own
``self_sha256``; the letkf filter adds ``ensemble_store``,
``ensemble_manifest``, ``ensemble_options``, ``filter_options`` and
``control_options``.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .assimilate import ASSIMILATION_HISTORY_KEY, FILTER_NAME, analyse, cross_stream_duplicates
from .checkpoint import read_checkpoint

ENSEMBLE_MANIFEST_SCHEMA = "gpuwm.arwen-global-da-ensemble/v1"
ENSEMBLE_MANIFEST_NAME = "da-ensemble.json"

LETKF_NOT_IN_TREE = (
    "the ensemble filter ('letkf': N members at their own truncation "
    "resident in one process, the control analysed through the ensemble "
    "covariance from its own innovations) is the ensemble package's; the "
    "door will not substitute the deterministic filter for it silently.  "
    "Use --filter successive-correction, or --filter letkf with --members"
)

#: Amendment D: one inflation configuration first.  The ensemble package's
#: own default re-draws five percent of the initial perturbation amplitude
#: after every analysis (additive inflation); the door switches that off
#: unless asked by name, so a spread deficit or a broken observation error
#: is seen for what it is instead of being hidden by a second mechanism.
DEFAULT_ADDITIVE_INFLATION_FRACTION = 0.0

ADDITIVE_INFLATION_RULE = (
    "RTPS is the one inflation mechanism of release 1 (amendment D); additive "
    "inflation is off unless --additive-inflation names a fraction, because "
    "several adaptive mechanisms hide each other: a broken observation error "
    "looks like a spread deficit"
)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _write_json(path: Path, payload: dict) -> dict:
    payload = dict(payload)
    payload.pop("self_sha256", None)
    payload["self_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return payload


def read_ensemble_manifest(path: str | Path) -> dict:
    """Read and verify a manifest: schema, self-hash and that every
    checkpoint it names exists."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != ENSEMBLE_MANIFEST_SCHEMA:
        raise ValueError(f"{path} is not a {ENSEMBLE_MANIFEST_SCHEMA} manifest")
    stated = payload.pop("self_sha256", None)
    if stated != hashlib.sha256(_canonical(payload)).hexdigest():
        raise ValueError(f"ensemble manifest {path} self-hash mismatch")
    payload["self_sha256"] = stated
    if payload.get("filter") not in FILTERS:
        raise ValueError(
            f"ensemble manifest {path} names filter {payload.get('filter')!r}, "
            f"which this tree does not carry ({sorted(FILTERS)})"
        )
    base = path.parent
    for entry in [payload["deterministic"], *payload["members"]]:
        checkpoint = Path(entry["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = base / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"ensemble manifest {path} names {checkpoint}, which does not exist"
            )
    return payload


def _lineage_of(checkpoint: Path) -> dict:
    metadata, _ = read_checkpoint(checkpoint)
    chain = metadata["physics_metadata"].get(ASSIMILATION_HISTORY_KEY) or {}
    cycles = chain.get("cycles", []) if isinstance(chain, dict) else []
    return {
        "checkpoint_self_sha256": metadata["self_sha256"],
        "step": int(metadata["step"]), "time_s": float(metadata["time_s"]),
        "analyses": len(cycles),
        "last_analysis": cycles[-1] if cycles else None,
    }


def write_ensemble_manifest(
    out_dir: Path, *, filter_name: str, deterministic: Path, members: list[Path],
    cfg, analysis_time_utc: str | None, previous: dict | None = None,
    extra: dict | None = None,
) -> dict:
    out_dir = Path(out_dir)
    det_meta, _ = read_checkpoint(deterministic, expected_config_hash=cfg.config_hash)
    payload = {
        **(extra or {}),
        "schema": ENSEMBLE_MANIFEST_SCHEMA,
        "filter": filter_name,
        "config_hash": cfg.config_hash,
        "truncation": int(cfg.truncation),
        "analysis_time_utc": analysis_time_utc,
        "deterministic": {
            "checkpoint": os.path.relpath(deterministic, out_dir),
            "self_sha256": det_meta["self_sha256"],
            "step": int(det_meta["step"]), "time_s": float(det_meta["time_s"]),
        },
        "members": [
            {
                "id": index,
                "checkpoint": os.path.relpath(member, out_dir),
                # A member checkpoint is written under the ENSEMBLE config
                # (its own truncation), so its hash is read without the
                # deterministic config's identity.
                "self_sha256": read_checkpoint(member)[0]["self_sha256"],
            }
            for index, member in enumerate(members)
        ],
        "lineage": _lineage_of(deterministic),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "previous_manifest_sha256": None if previous is None else previous.get("self_sha256"),
    }
    return _write_json(out_dir / ENSEMBLE_MANIFEST_NAME, payload)


def _apply_anchor_if_any(filter_obj, state, model, transform, moment):
    """The weak low-pass constraint of amendment E on the control analysis,
    when the door configured one; ``(state, record)``."""
    if getattr(filter_obj, "anchor", None) is None:
        return state, None
    from .da_anchor import apply_anchor

    anchor, anchor_options = filter_obj.anchor
    return apply_anchor(state, model, transform, anchor, anchor_options, analysis_time=moment)


def _rows_in_age_window(rows, moment, options):
    """The rows whose valid time lies in ``(moment - maximum_age_s, moment +
    future_tolerance_s]`` of the package's options: what an analysis at
    ``moment`` may be offered when no window was declared."""
    out = []
    for row in rows:
        when = row.valid_time if row.valid_time.tzinfo else row.valid_time.replace(tzinfo=dt.timezone.utc)
        age = (moment - when).total_seconds()
        if -float(options.future_tolerance_s) <= age <= float(options.maximum_age_s):
            out.append(row)
    return out


def _control_increment_for_recentring(control_increment_ens, before_anchor, after_anchor, ensemble):
    """The control's spectral increment at the ensemble truncation for
    increment recentring: the package's tapered increment (already at the
    ensemble triangle) plus the anchor's increment, the control after the
    anchor minus the control before it, truncated to the ensemble triangle.
    ``None`` when the package handed back no increment (the state mode
    reads the analysis itself)."""
    if not control_increment_ens:
        return None
    from .constants import SPECTRAL_FIELDS
    from .da.ensemble import truncate_spectral

    if after_anchor is before_anchor:
        return dict(control_increment_ens)
    ens_t = int(ensemble.transform.truncation)
    folded = {}
    for name in SPECTRAL_FIELDS:
        base = control_increment_ens[name]
        base_host = base.get() if hasattr(base, "get") else np.asarray(base)
        after = getattr(after_anchor.atmosphere, name)
        before = getattr(before_anchor.atmosphere, name)
        after_host = after.get() if hasattr(after, "get") else np.asarray(after)
        before_host = before.get() if hasattr(before, "get") else np.asarray(before)
        folded[name] = base_host + truncate_spectral(after_host - before_host, ens_t)
    return folded


def _ensure_anchor(filter_obj, cfg, transform) -> None:
    """Load the anchor a ``--anchor`` spelling names the first time the
    filter has the transform to read it at (the window's opening)."""
    spec = getattr(filter_obj, "anchor_spec", None)
    if spec is None or getattr(filter_obj, "anchor", None) is not None:
        return
    from .da_anchor import anchor_options_from_spec, load_anchor, parse_anchor_spec

    options = parse_anchor_spec(spec) if isinstance(spec, str) else dict(spec)
    filter_obj.anchor = (
        load_anchor(cfg, transform, options),
        anchor_options_from_spec(options),
    )


class SuccessiveCorrectionFilter:
    """The deterministic v1 filter: one resident state, the increment of
    :func:`assimilate.analyse`, every report compared at the analysis
    instant."""

    name = FILTER_NAME
    members = 1

    def __init__(self) -> None:
        self.anchor = None
        self.anchor_spec = None
        self.observation_bin_s = None

    def resident_states(self, deterministic):
        return [deterministic]

    def begin_window(self, cfg, model, transform, rows, window_start_s, window_end_s, start_utc,
                     extra_batches=None, extra_record=None) -> dict:
        _ensure_anchor(self, cfg, transform)
        if extra_batches:
            raise ValueError(
                f"{len(extra_batches)} radiance batches were offered to the {self.name} filter, which has no "
                "radiance operator and no ensemble covariance to spread a brightness temperature with; the "
                "radiance streams run under --filter letkf"
            )
        return {"bin_s": None, "rows": 0, "note": "the successive correction compares every report at the analysis instant"}

    def observe(self, state, time_s) -> int:
        return 0

    def analyse(self, cfg, model, transform, states, rows, *, sources, background,
                analysis_time, options):
        if len(states) != 1:
            raise ValueError(
                f"the {self.name} filter carries one deterministic state, "
                f"not {len(states)}"
            )
        analysis, report, phases = analyse(
            cfg, model, transform, states[0], rows, sources=sources,
            background=background, analysis_time=analysis_time, options=options,
        )
        report["observation_times"] = {
            "bin_s": None,
            "note": "every report compared with the state at the analysis instant (no trajectory record in the v1 door)",
        }
        clock = time.perf_counter()
        moment = _moment_of(report)
        analysis, anchor_record = _apply_anchor_if_any(self, analysis, model, transform, moment)
        if anchor_record is not None:
            report["anchor"] = anchor_record
            phases = {**phases, "anchor_s": time.perf_counter() - clock}
        report.setdefault("assessments", {})
        card = report.get("scorecard") or {}
        report["assessments"] = {
            **(card.get("assessments") or {}),
            "physical_consistency": {
                **((card.get("assessments") or {}).get("physical_consistency") or {}),
                "mass_preservation": report.get("mass_preservation"),
                "wind_balance": {k: v for k, v in (report.get("wind_balance") or {}).items() if k != "breakage"},
                "anchor": None if anchor_record is None else {
                    "applied": anchor_record["applied"], "increment_grid_rms": anchor_record.get("increment_grid_rms")},
            },
        }
        return analysis, report, phases

    def init(self, cfg, out_dir: Path, deterministic: Path, *, members: int,
             analysis_time_utc: str | None) -> dict:
        if int(members) != 1:
            raise ValueError(
                f"the {self.name} filter carries one deterministic member; "
                f"{members} members need the letkf filter.  {LETKF_NOT_IN_TREE}"
            )
        return write_ensemble_manifest(
            out_dir, filter_name=self.name, deterministic=Path(deterministic),
            members=[], cfg=cfg, analysis_time_utc=analysis_time_utc,
        )

    def manifest(self, cfg, out_dir: Path, deterministic: Path, *,
                 analysis_time_utc: str | None, previous: dict | None) -> dict:
        return write_ensemble_manifest(
            out_dir, filter_name=self.name, deterministic=Path(deterministic),
            members=[], cfg=cfg, analysis_time_utc=analysis_time_utc, previous=previous,
        )


def _moment_of(report: dict) -> dt.datetime:
    from .obs_table import parse_valid_time

    parsed = parse_valid_time(str(report.get("analysis_time_utc")))
    if parsed is None:
        raise ValueError("the report carries no analysis instant")
    return parsed


LETKF_FILTER_NAME = "letkf"
#: Where the letkf filter keeps its member checkpoints and the ensemble
#: package's own manifest, under the door's output directory.
ENSEMBLE_STORE_DIR = "ensemble"


#: The ``FilterOptions`` fields the door may set by flag: where the
#: localised solve runs and how the operators contract the state.  Path
#: settings, not science: the receipt records the one taken.
FILTER_SETTING_NAMES = ("solve_path", "operator_precision")


def _filter_settings(settings) -> dict:
    out = {}
    for key, value in dict(settings or {}).items():
        if key not in FILTER_SETTING_NAMES:
            raise ValueError(f"filter setting {key!r} is not one of {FILTER_SETTING_NAMES}")
        if value is not None:
            out[key] = value
    return out


def _overlay_door_options(filter_options, options):
    """The door's ``AssimilationOptions`` settings the two filters share
    (the withheld gate, the age window, the wind balance) laid over
    ``FilterOptions``; ``options`` None leaves them as they are."""
    if options is None:
        return filter_options
    return dataclasses.replace(
        filter_options,
        withheld_fraction=float(options.withheld_fraction),
        withheld_seed=int(options.withheld_seed),
        gate_minimum_count=int(options.gate_minimum_count),
        maximum_age_s=float(options.maximum_age_s),
        future_tolerance_s=float(options.future_tolerance_s),
        wind_balance=str(options.wind_balance),
    )


ANALYSED_STATE_BOUNDS_BREAKAGE = (
    "an analysed surface pressure above the radiation tables' ceiling is not an "
    "atmosphere the physics can step: the RRTMGP pressure tables end at 1,096.6 hPa "
    "of layer pressure and the next radiation call refuses the column (the T127 "
    "over T63 twin's noisy arm of 2026-09-06 died there with 1,096.98 hPa at one "
    "column after a noisy pressure increment), so the analysis is not applied and "
    "the background is carried, with the columns named; the observation "
    "vocabulary's 108,000 Pa gross bound is a bound on a report, not on a state "
    "(a T127 control carries 107.7 kPa at the Andes' Pacific foot by construction, "
    "a T63 member 109.5 kPa)"
)


def analysed_state_within_bounds(model, transform, state) -> dict[str, object]:
    """Engineering validity of the analysed state itself: the surface
    pressure everywhere finite and at or below the radiation tables'
    ceiling (:data:`woof.globe.da.analysis.
    RADIATION_SURFACE_PRESSURE_CEILING_PA`, 109,663 Pa).  Returns the
    record (``within``, the extrema, the count of columns outside and
    the breakage when not)."""
    from .da.analysis import RADIATION_SURFACE_PRESSURE_CEILING_PA

    backend = transform.backend
    g = model.grid_state(state.atmosphere, only=("ps",))
    ps = np.asarray(backend.to_numpy(g["ps"]), dtype=np.float64)
    model.release_syntheses()
    ceiling = float(RADIATION_SURFACE_PRESSURE_CEILING_PA)
    outside = int(np.sum(~np.isfinite(ps) | (ps > ceiling)))
    record = {
        "within": outside == 0,
        "surface_pressure_min_pa": float(ps.min()), "surface_pressure_max_pa": float(ps.max()),
        "ceiling_pa": ceiling, "columns_outside": outside,
    }
    if outside:
        record["breakage"] = ANALYSED_STATE_BOUNDS_BREAKAGE
    return record


OUTSIDE_WINDOW_BREAKAGE = (
    "a report whose time lies outside the analysis window (t0, t1] is not "
    "analysed at this instant: one earlier than t0 belongs to an earlier "
    "window (the neighbours the thinning kept then carried its information "
    "into the state, and it has no trajectory in this window to be compared "
    "at its own time, amendment B), one later than t1 to the next; found by "
    "the 2026-09-06 refutation, where the rows the first cycle thinned away "
    "were analysed again at the second cycle's instant, twenty seconds after "
    "their own time, and the chain saw nothing because it holds only the rows "
    "an analysis used"
)

CARRIED_NO_REPORT = (
    "no admissible report in the window: every offered row was outside the "
    "window, already in the chain or refused by the table quality control, so "
    "the background is carried unchanged as this hour's checkpoint and the "
    "members keep their own forecast"
)


class LetkfFilter:
    """The dual-resolution ensemble filter under the 2026-09-06 amendments,
    driven through the ensemble package (:mod:`woof.globe.da`).

    One analysis, in order (:meth:`analyse`):

    1. the window's batches were built when the window opened
       (:meth:`begin_window`, the package's ``ObservationWindow``) and the
       control's equivalents were taken by :meth:`observe` as the door
       stepped it; the members now catch up to the analysis instant at
       their own step, observing the bins on the way, and both are
       finished at the instant (amendment B); reports offered inside the
       window but not batched join as batches the package evaluates at
       the instant, reports outside the window are refused by count
       (:data:`OUTSIDE_WINDOW_BREAKAGE`), and a window with nothing
       admissible hands the background back as a carried hour
       (:data:`CARRIED_NO_REPORT`);
    2. the binning sensitivity is measured on the control (one evaluation
       at the instant against the binned equivalents);
    3. the package's analysis (:func:`analyze_ensemble` with a
       ``ControlBackground``): quality control and the withheld split, one
       local solve returning the members' increments and the CONTROL's
       from the control's own innovations (amendment A), the control
       increment tapered by degree and embedded (amendment C), the
       members updated, O-B and O-A on the members and the control, the
       Desroziers readings and the four assessments (amendment G); the
       ensemble-mean increment is recorded beside the control's as the
       comparison quantity, never applied unless
       ``ControlOptions.increment_source`` says ``ensemble-mean``;
    4. the weak low-pass anchor on the control analysis (amendment E),
       when configured;
    5. the members recentred on the control analysis restricted to the
       ensemble truncation, fully or partially (``recentre_fraction``);
    6. the door's card on the control: every offered row that passed the
       table quality control, O-B at the row's own bin, the linearised O-A
       for binned rows, the assigned errors and the ensemble spread in
       observation space.

    RTPS is the one inflation mechanism unless the door names an additive
    fraction (amendment D); the control's incremental insertion is the
    cycle door's and the members take theirs over the next window (``iau``
    by default since 2026-09-06).

    Every stream whose rows the tables carry is analysed: the neutral point
    vocabulary and the radio-occultation refractivity rows through the
    package's operators (``refractivity_n`` is a neutral variable of the
    ensemble package, so the successive correction's ``no_operator``
    refusal never applies here), a motion vector's assigned error inflated
    by its height-assignment shear when ``FilterOptions.
    amv_height_assignment_sigma_pa`` is set (the member-mean wind change
    across that pressure interval, in quadrature; 0 by default, the
    inflation selectable), and the same instrument
    reported by two streams collapsed to one row before batching
    (``assimilate.cross_stream_duplicates``, counted per stream).  The
    report's ``stream_roster`` says per stream what was offered, what left
    and why, what was assimilated and withheld, the O-B and O-A per
    variable, and the latency class the streams module measured for it.
    """

    name = LETKF_FILTER_NAME

    def __init__(self, ensemble_options=None, filter_options=None, control_options=None):
        from .da.options import EnsembleOptions, FilterOptions
        from .da_control import ControlOptions

        self.control_options = control_options or ControlOptions()
        self.ensemble_options = ensemble_options or EnsembleOptions(
            additive_inflation_fraction=DEFAULT_ADDITIVE_INFLATION_FRACTION)
        self.filter_options = self.control_options.filter_options(filter_options or FilterOptions())
        self.ensemble = None
        self.ensemble_cfg = None
        self._store: Path | None = None
        #: The door's assimilation options the shared settings came from,
        #: re-applied over a manifest's filter options on attach.
        self._door_options = None
        #: ``(ExternalAnchor, AnchorOptions)`` or None (amendment E), loaded
        #: from ``anchor_spec`` when the first window opens.
        self.anchor = None
        self.anchor_spec = None
        #: The observation time bin (s) of amendment B; None compares at
        #: the analysis instant.
        self.observation_bin_s: float | None = None
        self.window = None
        self._control_operators = None
        self._window_cfg = None
        #: An additive-inflation fraction the door named explicitly: laid
        #: over a manifest's ensemble options on attach (it is a per-analysis
        #: setting, not a construction setting of the members).
        self._explicit_additive: float | None = None
        #: What the open window carries beyond the package's record: the
        #: point and radiance row counts and the radiance streams' receipts.
        self._window_extra: dict = {}

    @property
    def members(self) -> int:
        return int(self.ensemble_options.members)

    @classmethod
    def from_options(cls, options, *, members: int | None = None, truncation: int | None = None,
                     control_options=None, additive_inflation_fraction: float | None = None,
                     observation_bin_s: float | None = None, filter_overrides=None,
                     filter_settings: dict | None = None):
        """A filter whose shared settings (the withheld gate, the age
        window, the wind balance) follow the door's ``AssimilationOptions``;
        additive inflation off unless a fraction is named (amendment D);
        ``filter_settings`` are the door's path settings laid over the
        package's ``FilterOptions`` (``solve_path``, ``operator_precision``),
        re-applied over a manifest's recorded options at ``attach``;
        ``filter_overrides`` names ``FilterOptions`` fields by name (the
        door's ``--filter-option``) and wins over an ensemble manifest's
        recorded options and the settings, so a knob the operator sets is
        the run's."""
        from .da.options import EnsembleOptions, FilterOptions

        ensemble_kwargs = {
            "additive_inflation_fraction": (
                DEFAULT_ADDITIVE_INFLATION_FRACTION if additive_inflation_fraction is None
                else float(additive_inflation_fraction)),
        }
        if members is not None:
            ensemble_kwargs["members"] = int(members)
        if truncation is not None:
            ensemble_kwargs["truncation"] = int(truncation)
        settings = _filter_settings(filter_settings)
        filter_options = (_overlay_door_options(dataclasses.replace(FilterOptions(), **settings), options)
                          .with_overrides(filter_overrides))
        built = cls(EnsembleOptions(**ensemble_kwargs), filter_options, control_options)
        built._door_options = options
        built._filter_settings = settings
        built._filter_overrides = dict(filter_overrides or {})
        built.observation_bin_s = observation_bin_s
        built._explicit_additive = None if additive_inflation_fraction is None else float(additive_inflation_fraction)
        return built

    def resident_states(self, deterministic):
        return [deterministic]

    # -- the ensemble store --------------------------------------------------

    def _build_ensemble_model(self, cfg, out_dir: Path):
        from .da.ensemble import ensemble_config
        from .runner import build_model_and_cold_state, build_transform

        ens_cfg = ensemble_config(cfg, self.ensemble_options)
        transform = build_transform(ens_cfg)
        model, cold = build_model_and_cold_state(
            ens_cfg, transform, scratch_destination=out_dir,
        )
        self.ensemble_cfg = ens_cfg
        return ens_cfg, model, transform, cold

    def init(self, cfg, out_dir: Path, deterministic: Path, *, members: int,
             analysis_time_utc: str | None, lagged_states=()) -> dict:
        from .da.ensemble import GlobalEnsemble

        out_dir = Path(out_dir)
        if int(members) != self.members:
            self.ensemble_options = dataclasses.replace(self.ensemble_options, members=int(members))
        store = out_dir / ENSEMBLE_STORE_DIR
        ens_cfg, model, transform, cold = self._build_ensemble_model(cfg, store)
        self.ensemble = GlobalEnsemble.from_state(
            ens_cfg, model, transform, cold, self.ensemble_options, lagged_states=lagged_states,
        )
        self._store = store
        return self.manifest(cfg, out_dir, Path(deterministic), analysis_time_utc=analysis_time_utc, previous=None)

    def attach(self, cfg, manifest: dict, manifest_path: Path) -> None:
        """Read the members the door manifest names, under the ensemble,
        filter and control options the manifest recorded (the door's
        shared settings re-applied over the filter's)."""
        from .da.ensemble import GlobalEnsemble
        from .da.options import EnsembleOptions, FilterOptions
        from .da_control import ControlOptions

        base = Path(manifest_path).parent
        store = Path(manifest.get("ensemble_store") or (base / ENSEMBLE_STORE_DIR))
        if not store.is_absolute():
            store = base / store
        recorded = manifest.get("ensemble_options")
        if isinstance(recorded, dict):
            names = {f.name for f in dataclasses.fields(EnsembleOptions)}
            self.ensemble_options = EnsembleOptions(**{k: v for k, v in recorded.items() if k in names})
        recorded_control = manifest.get("control_options")
        if isinstance(recorded_control, dict) and self.control_options == ControlOptions():
            names = {f.name for f in dataclasses.fields(ControlOptions)}
            self.control_options = ControlOptions(**{k: v for k, v in recorded_control.items() if k in names})
        recorded_filter = manifest.get("filter_options")
        if isinstance(recorded_filter, dict):
            names = {f.name for f in dataclasses.fields(FilterOptions)}
            kwargs = {k: v for k, v in recorded_filter.items() if k in names}
            if "analysis_fields" in kwargs:
                kwargs["analysis_fields"] = tuple(kwargs["analysis_fields"])
            kwargs.update(getattr(self, "_filter_settings", None) or {})
            self.filter_options = self.control_options.filter_options(
                _overlay_door_options(FilterOptions(**kwargs), self._door_options)
                .with_overrides(getattr(self, "_filter_overrides", None)))
        ens_cfg, model, transform, _cold = self._build_ensemble_model(cfg, store)
        self.ensemble = GlobalEnsemble.read(ens_cfg, model, transform, store)
        if self._explicit_additive is not None:
            self.ensemble.options = dataclasses.replace(
                self.ensemble.options, additive_inflation_fraction=self._explicit_additive)
        self.ensemble_options = self.ensemble.options
        self._store = store

    def manifest(self, cfg, out_dir: Path, deterministic: Path, *,
                 analysis_time_utc: str | None, previous: dict | None) -> dict:
        if self.ensemble is None:
            raise ValueError("the letkf filter has no ensemble to write; init or attach first")
        out_dir = Path(out_dir)
        store = self._store or (out_dir / ENSEMBLE_STORE_DIR)
        package_manifest = self.ensemble.write(store, label=analysis_time_utc)
        payload = write_ensemble_manifest(
            out_dir, filter_name=self.name, deterministic=Path(deterministic),
            members=[store / entry["file"] for entry in package_manifest["members"]],
            cfg=cfg, analysis_time_utc=analysis_time_utc, previous=previous,
            extra={
                "ensemble_store": os.path.relpath(store, out_dir),
                "ensemble_manifest": {
                    "path": os.path.relpath(store / "arwen-global-ensemble.json", out_dir),
                    "schema": package_manifest["schema"],
                    "truncation": package_manifest["truncation"],
                    "spread": package_manifest["spread"],
                    "resident_bytes": package_manifest["resident_bytes"],
                    "cycles": package_manifest["cycles"],
                },
                "ensemble_options": self.ensemble_options.identity(),
                "filter_options": self.filter_options.identity(),
                "control_options": self.control_options.identity(),
            },
        )
        return payload

    # -- the window (amendment B) ---------------------------------------------

    def _member_operators(self):
        from .da.operators import MemberOperators

        return MemberOperators.for_model(
            self.ensemble.model, self.ensemble.transform, self.ensemble.cfg,
            precision=self.filter_options.operator_precision,
            amv_height_assignment_sigma_pa=float(self.filter_options.amv_height_assignment_sigma_pa))

    def begin_window(self, cfg, model, transform, rows, window_start_s, window_end_s, start_utc,
                     extra_batches=None, extra_record=None) -> dict:
        """The window ``(t0, t1]`` opens: the rows inside it become the
        package's batches (unevaluated), assigned to bins of
        ``observation_bin_s`` (the whole window when None, so every report
        is compared at the analysis instant).  ``extra_batches`` are the
        radiance streams' batches (their own operators bound to the
        ensemble's and the control's transforms), binned and observed like
        the rows'; ``extra_record`` their receipts, carried in the window
        record under ``radiance``."""
        from .da.operators import MemberOperators
        from .da.window import ObservationWindow, batches_unevaluated
        from .da_window import rows_in_window

        if self.ensemble is None:
            raise ValueError("the letkf filter has no ensemble; init or attach it before opening a window")
        _ensure_anchor(self, cfg, transform)
        t0, t1 = float(window_start_s), float(window_end_s)
        span = t1 - t0
        self._window_span_s = span
        bin_s = span if self.observation_bin_s is None else float(self.observation_bin_s)
        det_dt = float(cfg.dt_s)
        ens_dt = float(self.ensemble_cfg.dt_s)
        for step in (det_dt, ens_dt):
            ratio = bin_s / step
            if abs(ratio - round(ratio)) > 1.0e-6 or ratio < 1.0 - 1.0e-9:
                raise ValueError(
                    f"the observation bin {bin_s:g} s is not a whole number of the {step:g} s model "
                    "step, so no resident state is ever at a bin instant"
                )
        inside = rows_in_window(list(rows), start_utc, t0, t1)
        offered_by_source = _count_by_source(inside)
        offered_ids = {row.identity_hash() for row in inside}
        inside, dropped_by_source = cross_stream_duplicates(inside)
        # The duplicates leave the window for good: a row not batched here
        # must not come back as a late row evaluated at the instant.
        self._window_dropped = offered_ids - {row.identity_hash() for row in inside}
        member_operators = self._member_operators()
        batches = batches_unevaluated(inside, member_operators)
        radiance_rows = 0
        for batch in list(extra_batches or []):
            if batch.count == 0:
                continue
            if batch.operator is None or not getattr(batch.operator, "evaluates_states", False):
                raise ValueError(
                    f"radiance batch {batch.stream!r}/{batch.variable!r} carries no operator that evaluates "
                    "states by their truncation; the filter cannot compare it with the members or the control"
                )
            batches.append(batch)
            radiance_rows += int(batch.count)
        self.window = ObservationWindow(
            batches=batches, start_s=t0, end_s=t1, epoch=start_utc, bin_s=bin_s, dt_s=ens_dt,
        )
        self._control_operators = MemberOperators.for_model(
            model, transform, cfg, precision=self.filter_options.operator_precision,
            amv_height_assignment_sigma_pa=float(self.filter_options.amv_height_assignment_sigma_pa))
        self._window_cfg = cfg
        self._window_offered = offered_by_source
        self._window_cross_stream = dropped_by_source
        self._window_unverified = _count_by_source([r for r in inside if r.received_time is None])
        record = self.window.record()
        self._window_extra = {"point_rows": len(inside), "radiance_rows": radiance_rows}
        if extra_record:
            self._window_extra["radiance"] = extra_record
        record.update(self._window_extra)
        record["rows"] = len(inside) + radiance_rows
        record["rows_offered_by_source"] = offered_by_source
        record["cross_stream_duplicates_by_source"] = dropped_by_source
        record["binned"] = self.observation_bin_s is not None
        return record

    def observe(self, state, time_s) -> int:
        """After every control step: the control's equivalents for the
        reports whose bin closes at ``time_s``."""
        if self.window is None or self.observation_bin_s is None:
            return 0
        return self.window.observe([state], float(time_s), self._control_operators, control=True)

    def _advance_members(self, target_time_s: float, member_operators) -> float:
        """The members to the analysis instant at their own step, observing
        the bins on the way.  Returns the wall."""
        clock = time.perf_counter()
        ensemble = self.ensemble
        dt_ens = float(self.ensemble_cfg.dt_s)
        window = self.window if self.observation_bin_s is not None else None

        def observer(ens, t):
            if window is not None:
                window.observe(ens.members, float(t), member_operators)

        from .da.ensemble import MemberStepError

        try:
            ensemble.advance_to(float(target_time_s), dt_ens, observer=observer)
        except MemberStepError as exc:
            written = self._write_failed_member(exc)
            raise MemberStepError(
                f"{exc}; the member's state entering the step and its pending increment are written: {written}",
                member=exc.member, time_s=exc.time_s, dt_s=exc.dt_s, state=None, pending=None,
                pending_steps_left=exc.pending_steps_left, pending_steps_total=exc.pending_steps_total,
            ) from exc
        return time.perf_counter() - clock

    def _write_failed_member(self, exc) -> str:
        """The failed member's state (the one that entered the fatal step,
        its IAU portion already added) as a hash-bound checkpoint, its
        pending increment (the spectral fields, numpy) and a JSON record,
        under the ensemble store, so the death can be replayed offline;
        returns the paths, or the reason nothing could be written."""
        from .checkpoint import write_checkpoint

        try:
            ensemble = self.ensemble
            store = Path(self._store) if self._store is not None else Path(ENSEMBLE_STORE_DIR)
            store.mkdir(parents=True, exist_ok=True)
            stem = f"arwen_global_member{exc.member:03d}_failed_t{int(round(exc.time_s)):08d}s"
            paths = []
            if exc.state is not None:
                path = store / f"{stem}.npz"
                write_checkpoint(
                    path, exc.state, config_hash=ensemble.cfg.config_hash,
                    to_numpy=ensemble.transform.backend.to_numpy,
                    semi_implicit_scheme=ensemble.cfg.semi_implicit_scheme,
                    integrator=ensemble.cfg.integrator,
                    trajectory=exc.trajectory,
                )
                paths.append(str(path))
            if exc.pending is not None:
                to_numpy = ensemble.transform.backend.to_numpy
                path = store / f"{stem}_pending_increment.npz"
                np.savez(path, **{name: np.asarray(to_numpy(exc.pending[name])) for name in exc.pending})
                paths.append(str(path))
            record = store / f"{stem}.json"
            record.write_text(json.dumps({
                "schema": "gpuwm.arwen-global-member-failure/v1",
                "member": exc.member, "time_s": exc.time_s, "dt_s": exc.dt_s,
                "pending_steps_left": exc.pending_steps_left, "pending_steps_total": exc.pending_steps_total,
                "message": str(exc), "files": paths,
            }, indent=1), encoding="utf-8")
            paths.append(str(record))
            return ", ".join(paths)
        except Exception as inner:  # noqa: BLE001 - the write must never hide the refusal
            return f"nothing ({type(inner).__name__}: {inner})"

    # -- the analysis ----------------------------------------------------------

    def analyse(self, cfg, model, transform, states, rows, *, sources, background,
                analysis_time, options):
        from .da.analysis import ControlBackground, analyze_ensemble, apply_mean_increment, recenter
        from .da.operators import MemberOperators, batches_from_rows, evaluate_batches
        from .da.window import batches_unevaluated
        from .da_control import mean_increment_comparison
        from .da_scorecard import scorecard_from_stream_table
        from .da_window import LINEARISED_LABEL, binning_sensitivity, rows_in_window

        if self.ensemble is None:
            raise ValueError("the letkf filter has no ensemble; init or attach it before analysing")
        if len(states) != 1:
            raise ValueError("the letkf filter takes the control state alone; the members are its own")
        control = states[0]
        phases: dict[str, float] = {}
        ensemble = self.ensemble
        moment = _resolve_moment(analysis_time, rows)
        member_operators = self._member_operators()
        if self._control_operators is None or self._window_cfg is not cfg:
            self._control_operators = MemberOperators.for_model(
                model, transform, cfg, precision=self.filter_options.operator_precision,
                amv_height_assignment_sigma_pa=float(self.filter_options.amv_height_assignment_sigma_pa))
        control_ops = self._control_operators
        prior_chain = json.loads(json.dumps(ensemble.provenance.get(ASSIMILATION_HISTORY_KEY, {})))
        offered_by_source = dict(getattr(self, "_window_offered", None) or _count_by_source(rows))
        cross_stream_by_source = dict(getattr(self, "_window_cross_stream", None) or {})
        unverified_by_source = dict(getattr(self, "_window_unverified", None) or {})

        # 1. the members catch up, observing the bins; the window finishes
        #    at the instant.  A report is compared in the window it falls
        #    in and nowhere else (amendment B): with a window open, rows
        #    outside its span are refused by count (OUTSIDE_WINDOW_BREAKAGE:
        #    they belong to another window, or to none), never evaluated at
        #    this instant; rows inside the span handed over late are
        #    evaluated at the instant.  Without a window (no interval was
        #    declared) the rows inside the package's own age window around
        #    the instant are the batches, evaluated at the instant; a table
        #    spanning days is not evaluated whole on every member for the
        #    age rule to drop.
        phases["members_advance_s"] = self._advance_members(float(control.time_s), member_operators)
        clock = time.perf_counter()
        batches = []
        binned_ids: set[str] = set()
        window_record = None
        outside_window = 0
        if self.window is not None:
            self.window.finish(ensemble.members, member_operators)
            self.window.finish([control], control_ops, control=True)
            batches = list(self.window.batches)
            window_record = self.window.record()
            for batch in batches:
                binned_ids.update(str(h) for h in batch.identity)
            # Rows handed over after the window opened and lying inside its
            # span (the twin's reports drawn at the instant) are evaluated
            # at the instant.  A report earlier than the window's start
            # belongs to an earlier window (its neighbours the thinning kept
            # went in then; it has no trajectory here to be compared at its
            # own time), one later than the end to the next: neither is
            # analysed at this instant (OUTSIDE_WINDOW_BREAKAGE).
            dropped_ids = getattr(self, "_window_dropped", None) or set()
            late = [row for row in rows
                    if row.identity_hash() not in binned_ids and row.identity_hash() not in dropped_ids]
            extra_rows = rows_in_window(late, self.window.epoch, float(self.window.start_s), float(self.window.end_s))
            outside_window = len(late) - len(extra_rows)
        else:
            extra_rows = _rows_in_age_window(rows, moment, self.filter_options)
            outside_window = len(rows) - len(extra_rows)
            offered_by_source = _count_by_source(extra_rows)
        if extra_rows:
            extra_rows, dropped = cross_stream_duplicates(extra_rows)
            for source, count in dropped.items():
                cross_stream_by_source[source] = cross_stream_by_source.get(source, 0) + int(count)
        extra = batches_unevaluated(extra_rows, member_operators)
        if extra:
            evaluate_batches(member_operators, ensemble.members, extra, target="simulated")
            evaluate_batches(control_ops, [control], extra, target="control_simulated")
        batches.extend(extra)
        batches = [b for b in batches if b.count]
        # A motion vector's error of record: the assigned error plus the
        # member-mean height-assignment shear in quadrature (the operators
        # filled assignment_shear for rows measured at an assigned pressure).
        amv_inflation = _inflate_amv_errors(batches, float(self.filter_options.amv_height_assignment_sigma_pa))
        if not batches:
            # Nothing admissible in this window: the background is carried
            # (the door writes it as the hour's checkpoint) and the report
            # says why (CARRIED_NO_REPORT); an empty analysis would be the
            # background wearing a new hash (the package's own refusal), and
            # a cycle that dies on an empty hour is no door for a fresh
            # analysis.
            phases["operators_control_s"] = time.perf_counter() - clock
            return control, self._carried_report(
                cfg, transform, rows, sources, background, moment, options, prior_chain,
                outside_window=outside_window, offered=len(rows), window_record=window_record), phases
        # 2. the binning sensitivity on the control: the analysis-instant
        #    equivalent against the binned one, per batch.
        instant_control = evaluate_batches(control_ops, [control], batches, target=None)
        sensitivity: dict[str, object] = {}
        substituted = 0
        for batch, instant in zip(batches, instant_control):
            if instant is None or batch.control_simulated is None:
                continue
            instant = np.asarray(instant)[0]
            binned = np.asarray(batch.control_simulated)[0]
            moved = ~np.isclose(instant, binned, rtol=0.0, atol=0.0)
            if self.observation_bin_s is not None and moved.any():
                substituted += int(moved.sum())
                sensitivity[f"{batch.stream}/{batch.variable}"] = {
                    "control": binning_sensitivity(instant, binned), "rows_at_own_time": int(moved.sum()),
                }
        instant_by_batch = {id(b): (None if i is None else np.asarray(i)[0]) for b, i in zip(batches, instant_control)}
        phases["operators_control_s"] = time.perf_counter() - clock

        # 3. the package's analysis: the members' and the control's.
        rejections_outside = {"outside_window": int(outside_window)} if outside_window else {}
        control_background = ControlBackground(control, model, transform, cfg, operators=control_ops)
        use_control = self.control_options.increment_source == "control"
        # The members' incremental update runs over the coming window, so
        # its length is the cycle's own span (the package's default is an
        # hour; a door cycling at another interval would otherwise refuse
        # the second analysis with the first window's portions still
        # pending).
        package_options = self.filter_options
        span = getattr(self, "_window_span_s", None)
        if (package_options.increment_application == "iau" and span is not None
                and abs(float(span) - float(package_options.iau_window_s)) > 1.0e-6):
            package_options = dataclasses.replace(package_options, iau_window_s=float(span))
        result = analyze_ensemble(
            ensemble, batches, package_options,
            analysis_time=moment, background=dict(background),
            additive_inflation=self.ensemble_options.additive_inflation_fraction > 0.0,
            control=control_background if use_control else None,
        )
        report = result.report
        for key, value in result.timings_s.items():
            phases[f"ensemble_{key}"] = float(value)
        clock = time.perf_counter()
        if use_control:
            control_analysis_state = result.control_analysis
            control_record = {
                "increment_source": "control",
                **{k: v for k, v in (result.control_record or {}).items()},
            }
            control_increment_ens = result.control_increment_spectral or {}
        else:
            # The comparison experiment, by name: the ensemble-mean
            # increment tapered, embedded and applied.
            control_analysis_state, apply_record = apply_mean_increment(
                control, model, transform, result.mean_increment_spectral, options=self.filter_options,
            )
            control_record = {
                "increment_source": "ensemble-mean",
                "route": "the ensemble-mean increment tapered and embedded in the control's triangle (the comparison experiment of amendment A)",
                **apply_record,
            }
            control_increment_ens = dict(result.mean_increment_spectral)
        comparison = mean_increment_comparison(
            result.mean_increment_spectral, control_increment_ens,
            ensemble.model, ensemble.transform, ensemble.members[0].atmosphere,
        )
        phases["control_s"] = time.perf_counter() - clock

        # 4. the anchor on the control analysis.
        clock = time.perf_counter()
        before_anchor = control_analysis_state
        control_analysis_state, anchor_record = _apply_anchor_if_any(
            self, control_analysis_state, model, transform, moment)
        phases["anchor_s"] = time.perf_counter() - clock
        # The members are recentred on the control ANALYSIS, anchor included:
        # under increment recentring (the package's default, its decision 18)
        # the shift is the control's increment restricted to the ensemble
        # triangle, so the anchor's own increment (the control after the
        # anchor minus the control before it) is folded in at that truncation;
        # under state recentring the truncated control state carries it.
        recentre_increment = _control_increment_for_recentring(
            control_increment_ens, before_anchor, control_analysis_state, ensemble)

        # The analysed state's own bounds (engineering validity): a control
        # whose surface pressure exceeds the radiation ceiling is not handed
        # back; the members are still recentred on it only when it holds.
        bounds = analysed_state_within_bounds(model, transform, control_analysis_state)

        # 5. recentring.
        clock = time.perf_counter()
        recentre_mode = str(self.filter_options.recentering_mode)
        if bounds["within"]:
            recentre_record = recenter(
                ensemble, control_analysis_state, transform,
                fraction=float(self.control_options.recentre_fraction), mode=recentre_mode,
                control_increment=recentre_increment, mean_increment=result.mean_increment_spectral,
            )
        else:
            recentre_record = {"skipped": "the control analysis exceeds the radiation ceiling; the members keep their own analysis",
                               "fraction": float(self.control_options.recentre_fraction), "mode": recentre_mode}
        phases["recentre_s"] = time.perf_counter() - clock

        # 6. the door's card on the control.
        clock = time.perf_counter()
        card, variables = _control_card(
            list(rows), control_ops, control_analysis_state, batches, instant_by_batch,
            self.observation_bin_s is not None, moment, options, prior_chain, label=report["analysis_time_utc"],
        )
        phases["card_s"] = time.perf_counter() - clock
        roster = _stream_roster(
            report, offered_by_source, cross_stream_by_source, unverified_by_source, batches, amv_inflation,
            rejections_outside=int(outside_window))

        # The chain onto the control analysis.
        chain_record = json.loads(json.dumps(ensemble.provenance.get(ASSIMILATION_HISTORY_KEY, {})))
        streams_fed = sorted(report["streams"])
        if chain_record.get("cycles"):
            chain_record["cycles"][-1].update({
                "filter": self.name, "streams": streams_fed,
                "background_self_sha256": str(background["self_sha256"]),
            })
        control_analysis_state.physics_state.metadata[ASSIMILATION_HISTORY_KEY] = chain_record

        ensemble_card = scorecard_from_stream_table(report["streams"], label=report["analysis_time_utc"])
        rejections: dict[str, int] = dict(rejections_outside)
        for per_stream in report["rejections"].values():
            for name, count in per_stream.items():
                rejections[name] = rejections.get(name, 0) + int(count)
        rejections.setdefault("already_assimilated", 0)
        package_assessments = report.get("assessments") or {}
        physical = {
            **(package_assessments.get("physical_consistency") or {}),
            "control": {k: control_record.get(k) for k in ("mass_preserving_log_offset", "positivity_repair", "wind_balance", "increment", "balance")},
            "anchor": None if anchor_record is None else {
                "applied": anchor_record["applied"], "increment_grid_rms": anchor_record.get("increment_grid_rms"),
                "age_s": anchor_record["age_s"]},
            "recentre": recentre_record,
            "spread": report["spread"],
            "reads": "budgets, imbalance, moisture and the transfer after the update; the cycle door adds the first step's surface-pressure tendency",
        }
        assessments = dict(card["assessments"])
        assessments["physical_consistency"] = {**assessments.get("physical_consistency", {}), **physical}
        assessments["predictive_value"] = {
            **assessments.get("predictive_value", {}),
            "ensemble_receipt": package_assessments.get("predictive_value"),
        }
        assessments["package"] = {
            "engineering_validity": package_assessments.get("engineering_validity"),
            "statistical_consistency": package_assessments.get("statistical_consistency"),
        }
        # Engineering validity is the status (amendment G): the door's card
        # on the control, the package's own gate and the analysed state's
        # bounds all have to hold.
        package_pass = report["status"] == "pass"
        status = "pass" if card["verdict"] == "complete" and package_pass and bounds["within"] else "fail"
        assessments["engineering"] = {
            **assessments.get("engineering", {}),
            "analysed_state_bounds": bounds,
            "verdict": "pass" if status == "pass" else "fail",
        }
        window_times = {
            **(window_record or {}),
            **(self._window_extra or {}),
            "rows": int(sum(b.count for b in self.window.batches)) if self.window is not None else 0,
            "rows_offered_by_source": offered_by_source,
            "cross_stream_duplicates_by_source": cross_stream_by_source,
            "binned": self.observation_bin_s is not None,
            "bin_s": self.observation_bin_s,
            "rows_at_own_time": int(substituted),
            "rows_outside_window": int(outside_window),
            "rows_outside_window_rule": (
                "with a window open, a row outside it belongs to another window and is never "
                "evaluated at this instant; without a window, rows outside the package's age "
                "window around the instant are left alone"),
            "binning_sensitivity": sensitivity,
            "o_minus_a_label": LINEARISED_LABEL if substituted else card["o_minus_a_label"],
        }
        door_report = {
            "schema": report["schema"],
            "acknowledgement": report["acknowledgement"],
            "name": report["name"],
            "config_hash": report["config_hash"],
            "pins_hash": report["pins_hash"],
            "analysis_time_utc": report["analysis_time_utc"],
            "background": {
                "path": background.get("path"), "self_sha256": str(background["self_sha256"]),
                "step": int(background["step"]), "time_s": float(background["time_s"]),
            },
            "obs_sources": list(sources),
            "options": {
                "filter": self.name, **report["options"],
                "control": self.control_options.identity(),
                "additive_inflation": ADDITIVE_INFLATION_RULE,
                "observation_bin_s": self.observation_bin_s,
            },
            "rejections": rejections,
            "rejections_by_stream": report["rejections"],
            "stream_roster": roster,
            "amv_height_assignment": amv_inflation,
            "assimilated_total": report["assimilated_total"],
            "withheld_total": report["withheld_total"],
            "assimilated_report_ids": {},
            "assimilation_history": {
                "key": ASSIMILATION_HISTORY_KEY,
                "cycles_in_analysis_chain": len(chain_record.get("cycles", [])),
                "reports_in_analysis_chain": len(chain_record.get("reports", {})),
            },
            "observation_times": window_times,
            "control": control_record,
            "mean_increment_transfer": comparison,
            "anchor": anchor_record,
            "recentre": recentre_record,
            "increment": {**report["increment"], "recenter": recentre_record},
            "spread": report["spread"],
            "letkf": report["letkf"],
            "operators": report.get("operators"),
            "localisation": report["localisation"],
            "inflation": report["inflation"],
            "members": report["members"],
            "ensemble_truncation": report["truncation"],
            "control_truncation": int(transform.truncation),
            "lineage": {
                "filter": self.name, "streams": streams_fed,
                "background_self_sha256": str(background["self_sha256"]),
                "analysis_time_utc": report["analysis_time_utc"],
                "chain_length": len(chain_record.get("cycles", [])),
            },
            "scorecard": {**card, "assessments": assessments},
            "scorecard_judges": (
                "the control analysis handed back, every offered row that passed the "
                "table quality control, through the control's operators; O-B at the "
                "report's own bin instant where observed, O-A linearised for those rows; "
                "ensemble_scorecard is the package's own receipt on the members and the control"
            ),
            "ensemble_scorecard": ensemble_card,
            "assessments": assessments,
            "variables": variables,
            "gate_of_record": {
                "rule": report["gate_of_record"]["rule"],
                "passed": status == "pass",
                "failed_variables": [],
                "failed": list(report["gate_of_record"]["failed"])
                + ([] if card["verdict"] == "complete" else list(card["incomplete_streams"]))
                + ([] if bounds["within"] else [
                    f"analysed surface pressure {bounds['surface_pressure_min_pa']:.1f} to "
                    f"{bounds['surface_pressure_max_pa']:.1f} Pa exceeds the radiation ceiling "
                    f"{bounds['ceiling_pa']:.0f} Pa at {bounds['columns_outside']} columns"]),
                "incomplete": list(report["gate_of_record"]["incomplete"]),
            },
            "status": status,
            "ensemble_report": report,
        }
        return control_analysis_state, door_report, phases


    def _carried_report(self, cfg, transform, rows, sources, background, moment, options, chain, *,
                        outside_window: int, offered: int, window_record):
        """The report of an hour the letkf filter could not analyse: the
        background carried, the rejections counted (the rows outside the
        window, the rows the chain already holds, the table quality
        control's refusals), the card incomplete by name (nothing reached
        the operators), the status ``carried`` with the reason
        (:data:`CARRIED_NO_REPORT`), so the cycle writes the background as
        this hour's checkpoint and the receipt counts the hour as carried."""
        from .assimilate import (
            AssimilationOptions, _chain_from_background, _refuse_chain_rows, _table_quality_control,
        )
        from .da.analysis import ANALYSIS_SCHEMA, GATE_RULE
        from .da_scorecard import Departures, scorecard

        from .da.operators import OPERATOR_VARIABLES

        options = options or AssimilationOptions()
        kept, rejections = _table_quality_control(
            list(rows), moment, options, operator_variables=OPERATOR_VARIABLES)
        kept = _refuse_chain_rows(kept, _chain_from_background({ASSIMILATION_HISTORY_KEY: chain or {}}), rejections)
        rejections = {name: int(count) for name, count in rejections.items() if count}
        rejections["outside_window"] = int(outside_window)
        rejections.setdefault("already_assimilated", 0)
        label = moment.isoformat(timespec="seconds")
        card = scorecard(Departures(), label=label)
        assessments = dict(card["assessments"])
        assessments["engineering"] = {
            **assessments.get("engineering", {}),
            "verdict": "carried",
            "reason": CARRIED_NO_REPORT,
            "rejections": rejections,
            "rows_offered": int(offered),
        }
        ensemble = self.ensemble
        return {
            "schema": ANALYSIS_SCHEMA,
            "name": cfg.name,
            "config_hash": cfg.config_hash,
            "analysis_time_utc": label,
            "background": {
                "path": background.get("path"), "self_sha256": str(background["self_sha256"]),
                "step": int(background["step"]), "time_s": float(background["time_s"]),
            },
            "obs_sources": list(sources),
            "options": {"filter": self.name, "control": self.control_options.identity(),
                        "observation_bin_s": self.observation_bin_s},
            "rejections": rejections,
            "rejections_by_stream": {},
            "rejection_breakage": {"outside_window": OUTSIDE_WINDOW_BREAKAGE},
            "assimilated_total": 0,
            "withheld_total": 0,
            "assimilated_report_ids": {},
            "assimilation_history": {
                "key": ASSIMILATION_HISTORY_KEY,
                "cycles_in_analysis_chain": len((chain or {}).get("cycles", [])),
                "reports_in_analysis_chain": len((chain or {}).get("reports", {})),
            },
            "observation_times": {
                **(window_record or {}), **(self._window_extra or {}), "rows": 0,
                "binned": self.observation_bin_s is not None,
                "bin_s": self.observation_bin_s, "rows_at_own_time": 0,
                "rows_outside_window": int(outside_window),
                "rows_outside_window_rule": (
                    "with a window open, a row outside it belongs to another window and is never "
                    "evaluated at this instant; without a window, rows outside the package's age "
                    "window around the instant are left alone"),
                "binning_sensitivity": {},
                "o_minus_a_label": card["o_minus_a_label"],
            },
            "control": None,
            "mean_increment_transfer": None,
            "anchor": None,
            "recentre": None,
            "increment": None,
            "spread": None,
            "letkf": None,
            "operators": None,
            "members": None if ensemble is None else len(ensemble.members),
            "ensemble_truncation": None if ensemble is None else int(ensemble.transform.truncation),
            "control_truncation": int(transform.truncation),
            "lineage": {
                "filter": self.name, "streams": [],
                "background_self_sha256": str(background["self_sha256"]),
                "analysis_time_utc": label,
                "chain_length": len((chain or {}).get("cycles", [])),
            },
            "scorecard": {**card, "assessments": assessments},
            "scorecard_judges": "nothing reached the operators: the background is carried",
            "ensemble_scorecard": None,
            "assessments": assessments,
            "variables": {},
            "gate_of_record": {
                "rule": GATE_RULE, "passed": False, "failed_variables": [],
                "failed": [CARRIED_NO_REPORT], "incomplete": [],
            },
            "carried_reason": CARRIED_NO_REPORT,
            "status": "carried",
            "ensemble_report": None,
        }


def _resolve_moment(analysis_time, rows) -> dt.datetime:
    from .assimilate import _resolve_analysis_time

    return _resolve_analysis_time(analysis_time, list(rows))


def _control_card(rows, control_ops, analysis, batches, instant_by_batch, binned: bool,
                  moment, options, chain, *, label):
    """The DA card of the control update: every offered row that passed
    the table quality control and is not in the ensemble's chain,
    evaluated on the control background (at its bin instant where
    observed) and the control analysis (at the analysis instant; the
    linearised equivalent for binned rows).  Returns ``(card, variables)``
    with the door's per-variable rows (never gated: engineering validity
    is the status under amendment G).  The card's Desroziers reading is
    taken against the error the solve weighted each row by: the batch's
    error as the analysis left it (the observation-error calibration laid
    over the doors' figure where the table names the cell, the motion
    vectors' height-assignment inflation where it is on), with the rows'
    own assigned figure recorded beside it as ``door_error``; a card read
    against the doors' errors judged an error the analysis never used."""
    from .assimilate import (
        AssimilationOptions, _chain_from_background, _refuse_chain_rows, _table_quality_control,
    )
    from .da.operators import OPERATOR_VARIABLES, evaluate_batches
    from .da_scorecard import Departures, scorecard
    from .da_window import LINEARISED_LABEL, linearised_analysis_equivalent
    from .obs_table import VARIABLE_TABLE

    options = options or AssimilationOptions()
    kept, rejections = _table_quality_control(
        list(rows), moment, options, operator_variables=OPERATOR_VARIABLES)
    kept = _refuse_chain_rows(kept, _chain_from_background({ASSIMILATION_HISTORY_KEY: chain or {}}), rejections)
    by_identity = {row.identity_hash(): row for row in kept}
    parts = []
    variables: dict[str, object] = {}

    def stat(v):
        return {"mean": float(np.mean(v)) if v.size else 0.0,
                "rms": float(np.sqrt(np.mean(v * v))) if v.size else 0.0}

    analysis_hx = evaluate_batches(control_ops, [analysis], batches, target=None)
    for batch, hx_a_all in zip(batches, analysis_hx):
        if batch.count == 0 or hx_a_all is None or batch.control_simulated is None:
            continue
        hx_a = np.asarray(hx_a_all, dtype=np.float64)[0]
        hx_b_binned = np.asarray(batch.control_simulated, dtype=np.float64)[0]
        hx_b_instant = instant_by_batch.get(id(batch))
        if hx_b_instant is None:
            hx_b_instant = hx_b_binned
        foreign = batch.variable not in OPERATOR_VARIABLES
        if foreign:
            # A radiance batch has no obs-table rows: every finite row of
            # the batch is on the card under its own stream.
            mask = np.isfinite(hx_b_binned) & np.isfinite(hx_a)
        else:
            mask = np.array([str(h) in by_identity for h in batch.identity], dtype=bool)
        if not mask.any():
            continue
        binned_rows = np.zeros(batch.count, dtype=bool)
        offsets = np.zeros(batch.count)
        if binned:
            binned_rows = ~np.isclose(hx_b_instant, hx_b_binned, rtol=0.0, atol=0.0)
            when = np.array([(v.astimezone(dt.timezone.utc) - moment).total_seconds() for v in batch.valid_time])
            offsets = np.where(binned_rows, when, 0.0)
        hx_a_eff = np.where(
            binned_rows, linearised_analysis_equivalent(hx_b_binned, hx_b_instant, hx_a), hx_a)
        sim = np.asarray(batch.simulated, dtype=np.float64)
        spread_h = np.sqrt(((sim - sim.mean(axis=0)) ** 2).sum(axis=0) / max(sim.shape[0] - 1, 1))
        label = (LINEARISED_LABEL if binned_rows[mask].any()
                 else "O-A evaluated on the control analysis at the analysis instant")
        if foreign:
            parts.append(Departures(
                source=np.full(int(mask.sum()), batch.stream, dtype=object),
                variable=np.full(int(mask.sum()), batch.variable, dtype=object),
                latitude_deg=batch.latitude_deg[mask], longitude_deg=batch.longitude_deg[mask],
                level_pa=np.exp(batch.ln_pressure[mask]),
                o_minus_b=batch.value[mask] - hx_b_binned[mask], o_minus_a=batch.value[mask] - hx_a_eff[mask],
                withheld=np.zeros(int(mask.sum()), dtype=bool), error=batch.error[mask],
                spread_h=spread_h[mask], time_offset_s=offsets[mask], o_minus_a_label=label,
            ))
        else:
            selected = [by_identity[str(h)] for h in batch.identity[mask]]
            parts.append(Departures.from_rows(
                selected, hx_b_binned[mask], hx_a_eff[mask], withheld=False,
                spread_h=spread_h[mask], time_offset_s=offsets[mask], o_minus_a_label=label,
                error=np.asarray(batch.error, dtype=np.float64)[mask],
            ))
        omb = batch.value[mask] - hx_b_binned[mask]
        oma = batch.value[mask] - hx_a_eff[mask]
        finite = np.isfinite(omb) & np.isfinite(oma)
        row = variables.setdefault(batch.stream if foreign else batch.variable, {
            "analysed": True, "count": 0, "units": VARIABLE_TABLE.get(batch.variable, {}).get("units", ""),
            "_omb": [], "_oma": [],
            "withheld": {"count": 0, "o_minus_b": stat(np.zeros(0)), "o_minus_a": stat(np.zeros(0)), "ids": []},
            "gated": False, "gate_passed": True,
        })
        row["count"] += int(finite.sum())
        row["_omb"].append(omb[finite])
        row["_oma"].append(oma[finite])
    for name, row in variables.items():
        omb = np.concatenate(row.pop("_omb")) if row["_omb"] else np.zeros(0)
        oma = np.concatenate(row.pop("_oma")) if row["_oma"] else np.zeros(0)
        row["o_minus_b"] = stat(omb)
        row["o_minus_a"] = stat(oma)
    card = scorecard(Departures.concatenate(parts), label=label)
    return card, variables


def _count_by_source(rows) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[row.source] = out.get(row.source, 0) + 1
    return dict(sorted(out.items()))


AMV_HEIGHT_ASSIGNMENT_RULE = (
    "a satellite motion vector's error of record is its assigned error plus, in quadrature, the "
    "member-mean wind change across the height-assignment sigma either side of its assigned pressure "
    "(Forsythe and Saunders 2008): a vector in a sheared layer is worth less than one in a barotropic "
    "layer, because a one-sigma miss in its height is a wind error of that shear"
)


def _inflate_amv_errors(batches, sigma_pa: float) -> dict[str, object]:
    """The situation-dependent error of the motion-vector rows: every batch
    row with a finite ``assignment_shear`` (the operators fill it for
    ``amv_assigned_pressure`` rows) gets ``sqrt(error^2 + shear^2)`` as its
    error of record; the record says per stream how many rows were
    inflated and by how much."""
    from .da.operators import AMV_MEASUREMENT

    per_stream: dict[str, dict[str, object]] = {}
    if sigma_pa <= 0.0:
        return {"sigma_pa": 0.0, "rule": AMV_HEIGHT_ASSIGNMENT_RULE, "applied": False, "streams": per_stream}
    for batch in batches:
        if batch.count == 0:
            continue
        amv = np.asarray(batch.measurement, dtype=object) == AMV_MEASUREMENT
        shear = np.asarray(batch.assignment_shear, dtype=np.float64)
        rows = amv & np.isfinite(shear)
        if not rows.any():
            continue
        before = np.asarray(batch.error, dtype=np.float64).copy()
        after = before.copy()
        after[rows] = np.sqrt(before[rows] ** 2 + shear[rows] ** 2)
        batch.error = after
        entry = per_stream.setdefault(batch.stream, {"rows": 0, "shear_mean_m_s": 0.0, "error_before_mean": 0.0,
                                                     "error_after_mean": 0.0, "shear_max_m_s": 0.0})
        n_before = int(entry["rows"])
        n = int(rows.sum())
        total = n_before + n
        entry["shear_mean_m_s"] = (entry["shear_mean_m_s"] * n_before + float(shear[rows].sum())) / total
        entry["error_before_mean"] = (entry["error_before_mean"] * n_before + float(before[rows].sum())) / total
        entry["error_after_mean"] = (entry["error_after_mean"] * n_before + float(after[rows].sum())) / total
        entry["shear_max_m_s"] = max(float(entry["shear_max_m_s"]), float(shear[rows].max()))
        entry["rows"] = total
    return {"sigma_pa": float(sigma_pa), "rule": AMV_HEIGHT_ASSIGNMENT_RULE,
            "applied": bool(per_stream), "streams": per_stream}


def _stream_roster(report, offered_by_source, cross_stream_by_source, unverified_by_source, batches,
                   amv_inflation, *, rejections_outside: int) -> dict[str, object]:
    """Per stream, this analysis: rows offered in the window, the
    cross-stream duplicates dropped, the rows batched, the package's
    rejections by name, the rows assimilated and withheld, O-B and O-A per
    variable (the ensemble mean and the control), the latency class the
    streams module measured for the stream, and whether the stream was
    offered and refused whole (a defect the receipt names)."""
    from .obs_streams import LATENCY_CLASSES, STREAMS

    batched: dict[str, int] = {}
    for batch in batches:
        batched[batch.stream] = batched.get(batch.stream, 0) + int(batch.count)
    streams = report.get("streams") or {}
    rejections = report.get("rejections") or {}
    roster: dict[str, object] = {}
    defects: list[str] = []
    for source in sorted(set(offered_by_source) | set(batched) | set(streams) | set(rejections)):
        spec = STREAMS.get(source)
        variables: dict[str, object] = {}
        assimilated = 0
        withheld = 0
        for variable, entry in (streams.get(source) or {}).items():
            regions = entry.get("regions") or {}
            cell = regions.get("global") or {}
            used = cell.get("assimilated") or {}
            held = cell.get("withheld") or {}
            control = cell.get("control") or {}
            assimilated += int(entry.get("count", 0))
            withheld += int(entry.get("withheld_count", 0))
            variables[variable] = {
                "assimilated": int(entry.get("count", 0)),
                "withheld": int(entry.get("withheld_count", 0)),
                "o_minus_b_rms": (used.get("o_minus_b") or {}).get("rms"),
                "o_minus_a_rms": (used.get("o_minus_a") or {}).get("rms"),
                "o_minus_b_mean": (used.get("o_minus_b") or {}).get("mean"),
                "o_minus_a_mean": (used.get("o_minus_a") or {}).get("mean"),
                "withheld_o_minus_b_rms": (held.get("o_minus_b") or {}).get("rms"),
                "withheld_o_minus_a_rms": (held.get("o_minus_a") or {}).get("rms"),
                "control_o_minus_b_rms": ((control.get("assimilated") or {}).get("o_minus_b") or {}).get("rms"),
                "control_o_minus_a_rms": ((control.get("assimilated") or {}).get("o_minus_a") or {}).get("rms"),
                "desroziers_error_variance_ratio": (entry.get("desroziers") or {}).get("error_variance_ratio"),
            }
        offered = int(offered_by_source.get(source, 0))
        refused = {k: int(v) for k, v in (rejections.get(source) or {}).items()}
        duplicates = int(cross_stream_by_source.get(source, 0))
        # A stream whose every row was the same instrument as another
        # stream's was analysed through that stream; only rows of its own
        # that reached nothing count as refused.
        refused_whole = offered - duplicates > 0 and assimilated == 0 and withheld == 0
        entry = {
            "offered": offered,
            "cross_stream_duplicates": int(cross_stream_by_source.get(source, 0)),
            "batched": int(batched.get(source, 0)),
            "refused": refused,
            "assimilated": assimilated,
            "withheld": withheld,
            "latency_unverified": int(unverified_by_source.get(source, 0)),
            "latency_class": None if spec is None else spec.latency_class,
            "latency_class_meaning": None if spec is None else LATENCY_CLASSES[spec.latency_class],
            "latency_basis": None if spec is None else spec.latency_basis,
            "door": None if spec is None else spec.door,
            "in_stream_table": spec is not None,
            "variables": variables,
            "amv_height_assignment": (amv_inflation.get("streams") or {}).get(source),
            "refused_whole": refused_whole,
        }
        if refused_whole:
            defects.append(f"{source}: {offered} rows offered, none assimilated ({refused or 'no batch reached the filter'})")
        roster[source] = entry
    return {
        "streams": roster,
        "rows_outside_window": int(rejections_outside),
        "rule": (
            "every stream whose rows the tables carry is analysed; a stream that offers rows in the "
            "window and has none assimilated is a defect named here (refused_whole), never a silent gap"
        ),
        "defects": defects,
    }


#: Filter name -> factory.
FILTERS: dict[str, object] = {
    FILTER_NAME: SuccessiveCorrectionFilter,
    LETKF_FILTER_NAME: LetkfFilter,
}
DEFAULT_FILTER = FILTER_NAME


def resolve_filter(name: str | None, *, options=None, members: int | None = None,
                   truncation: int | None = None, control_options=None,
                   additive_inflation_fraction: float | None = None,
                   observation_bin_s: float | None = None, filter_overrides=None,
                   filter_settings: dict | None = None):
    """The filter object for ``name``; the letkf filter takes the door's
    assimilation options, the member count, the ensemble truncation, the
    control options, the additive-inflation fraction (off by default), the
    observation time bin and the path settings (:data:`FILTER_SETTING_NAMES`)."""
    name = DEFAULT_FILTER if name is None else str(name)
    factory = FILTERS.get(name)
    if factory is None:
        raise ValueError(
            f"unknown analysis filter {name!r}; this tree carries {sorted(FILTERS)}"
        )
    if factory is LetkfFilter:
        return LetkfFilter.from_options(
            options, members=members, truncation=truncation, control_options=control_options,
            additive_inflation_fraction=additive_inflation_fraction, observation_bin_s=observation_bin_s,
            filter_overrides=filter_overrides, filter_settings=filter_settings,
        )
    if filter_overrides:
        raise ValueError(
            f"--filter-option names FilterOptions fields of the letkf filter ({sorted(filter_overrides)}); "
            f"the {name} filter carries none of them")
    built = factory()
    built.observation_bin_s = observation_bin_s
    return built


__all__ = [
    "ADDITIVE_INFLATION_RULE",
    "ANALYSED_STATE_BOUNDS_BREAKAGE",
    "CARRIED_NO_REPORT",
    "OUTSIDE_WINDOW_BREAKAGE",
    "analysed_state_within_bounds",
    "DEFAULT_ADDITIVE_INFLATION_FRACTION",
    "DEFAULT_FILTER",
    "ENSEMBLE_MANIFEST_NAME",
    "FILTER_SETTING_NAMES",
    "ENSEMBLE_MANIFEST_SCHEMA",
    "FILTERS",
    "LETKF_NOT_IN_TREE",
    "LetkfFilter",
    "SuccessiveCorrectionFilter",
    "read_ensemble_manifest",
    "resolve_filter",
    "write_ensemble_manifest",
]
