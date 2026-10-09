"""``woof global da``: the data-assimilation door of WOOF global.

Five legs, one receipt (:data:`DA_RECEIPT_NAME`, schema
:data:`DA_RECEIPT_SCHEMA`) beside whatever the leg wrote:

``init``
    Build the ensemble from an analysis: the deterministic state (a
    checkpoint given, or the config's cold start written as step 0) and
    the filter's members, recorded in the ensemble manifest
    (:mod:`woof.globe.da_filter`).

``cycle``
    The hourly cycle (:func:`woof.globe.cycle.cycle`): fetch the
    streams for each window, quality control, analyse through the filter,
    write the deterministic analysis checkpoint and the ensemble manifest,
    a report per cycle with O-B and O-A per stream, variable and region
    (the DA scorecard) and the wall budget of every cycle.

``analyze``
    One analysis at a stated instant from stated observation tables into a
    checkpoint (the file door of :mod:`woof.globe.assimilate`),
    with its scorecard printed.

``fresh``
    The one command: fetch the newest analysis (GDAS through the tree's
    fetch door, or a GRIB on disk), derive the run configuration from a
    base TOML, ``init`` if no ensemble exists in the output, ``cycle``
    hourly through every stream up to the newest observation hour, and
    hand back the analysis checkpoint the forecast starts from.

``forecast``
    The forecast from an analysis checkpoint (the runner's restart).

Every analysis carries its lineage in the checkpoint (the assimilation
chain: instant, background identity, filter, streams) and in the report;
the receipt names the checkpoint handed back with its digest, so the
forecast door and the scorecards read the same object.

Interface decisions (the door builds against the ensemble lane's filter
and the observations lane's streams through code, not assumptions):

* the filter interface is :mod:`woof.globe.da_filter`'s
  (``resident_states`` / ``analyse`` / ``init`` / ``manifest``); the
  deterministic successive correction is the filter in this tree and the
  default; ``letkf`` refuses by name until the ensemble lane lands;
* a stream is a :data:`woof.globe.da_streams.STREAM_TABLE` entry
  whose fetch lands files an obs-table decoder entry reads; the door
  fetches each stream for ``(analysis time - interval, analysis time]``
  and records URL or path, bytes, SHA-256, wall and latency behind real
  time in the fetch manifest and the report;
* the observation vocabulary is :data:`woof.globe.obs_table.
  VARIABLE_TABLE`; the scorecard groups by the row's ``source``;
* the analysis checkpoint is the runner's own checkpoint format (schema
  v3) with the chain in the physics metadata; the ensemble manifest names
  the member checkpoints beside it;
* the receipt's gate is engineering validity (design amendment G): a
  stream whose rows never reached the operators or has no O-A is
  INCOMPLETE; O-A against O-B, the Desroziers reading and the withheld
  rows are assessments the receipt carries, never a gate;
* the control member gets its own analysis through the ensemble
  covariance (amendment A, :mod:`woof.globe.da_control`), the
  transfer is tapered and inspected (C), reports are compared at their
  own bin instants (B, ``--observation-bin-s``), the external analysis is
  a weak low-pass constraint (E, ``--anchor``), RTPS is the one inflation
  by default (D), the increment may be inserted incrementally (D,
  ``--increment-application iau``), and ``fresh`` states its information
  cutoff (F, ``--cutoff-utc``).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import time
import tomllib
from pathlib import Path

from .assimilate import AssimilationOptions, assimilate
from .checkpoint import read_checkpoint, write_checkpoint
from .config import ArwenGlobalConfig, load_config
from .cycle import ANALYSIS_PREFIX, cycle as run_cycle, resolve_start_time
from .da_filter import (
    ENSEMBLE_MANIFEST_NAME, read_ensemble_manifest, resolve_filter,
)


def _configure_filter(filter, *, anchor_spec=None):
    """The door's runtime settings the filters share (the anchor spelling;
    loaded when the first window opens)."""
    if anchor_spec is not None:
        filter.anchor_spec = anchor_spec
    return filter
from .da_scorecard import render_table
from .da_streams import fetch_analysis, resolve_stream
from .obs_table import parse_valid_time
from .runner import build_model_and_cold_state, build_transform, run

DA_RECEIPT_SCHEMA = "gpuwm.arwen-global-da/v1"
DA_RECEIPT_NAME = "da-receipt.json"
FRESH_CONFIG_NAME = "fresh-config.toml"
#: The deterministic state ``init`` writes when it builds from the cold
#: start (the runner's checkpoint format, its own file name).
INITIAL_STATE_NAME = "da-initial-state.npz"
DEFAULT_FORECAST_HOURS = 24.0
#: How far behind real time the newest observation hour is taken to be
#: when ``fresh`` decides where the cycle stops: the hourly surface record
#: is complete about an hour after the hour.
DEFAULT_OBSERVATION_LATENCY_S = 3600.0

#: The fresh door's shipped shape when the command names nothing else
#: (2026-09-06, the completed system): the ensemble filter, 32 members at
#: the ensemble truncation under the control, every report at its own
#: 600 s bin, the incremental analysis update, the balance package, the
#: hybrid covariance at the package's beta, and every stream with a
#: decoding door that a live window can carry (the two radiance streams
#: included).  Every part is selectable by name (``--filter
#: successive-correction``, ``--members``, ``--observation-bin-s``,
#: ``--increment-application direct``, ``--hybrid-beta 1``, ``--stream``):
#: naming any stream or table replaces the roster with exactly what was
#: named, so a case day runs on its tables and nothing is fetched.
DEFAULT_FRESH_FILTER = "letkf"
DEFAULT_FRESH_MEMBERS = 32
#: Three quarters the members' covariance, one quarter the packaged static
#: table (the T255 L40 estimate of record); ``--hybrid-beta 1`` is the
#: ensemble alone.
DEFAULT_FRESH_HYBRID_BETA = 0.75
DEFAULT_FRESH_OBSERVATION_BIN_S = 600.0
DEFAULT_FRESH_STREAMS = (
    "iem-metar", "awc-metar", "igra2", "ndbc", "goes-dmw", "cdaac-ro", "wis2", "atms", "goes-abi",
)
#: The radiance streams hand the filter their own batches and run under
#: the ensemble filter only, so the successive correction's roster is the
#: point streams.
DEFAULT_FRESH_POINT_STREAMS = tuple(s for s in DEFAULT_FRESH_STREAMS if s not in ("atms", "goes-abi"))


def default_observation_bin_s(dt_s: float, member_dt_s: float | None, interval_s: float,
                              target_s: float = DEFAULT_FRESH_OBSERVATION_BIN_S) -> float:
    """The fresh door's observation bin: ``target_s`` (600 s) when the
    control's step and the members' step both divide it, otherwise the
    smallest whole multiple of the control's step at or above the target
    that the members' step divides and that divides the cycle interval
    (an Eulerian T255 control at 90 s with T127 members at 180 s gets
    720 s; the semi-Lagrangian control at 300 s with members at 600 s
    gets 600 s).  The filter refuses a bin its steps do not divide, so a
    default has to be one they do."""
    dt = float(dt_s)
    mdt = dt if member_dt_s is None else float(member_dt_s)
    interval = float(interval_s)
    k = max(1, int(math.ceil(float(target_s) / dt - 1.0e-9)))
    while k * dt <= interval + 1.0e-9:
        candidate = k * dt
        if (abs(candidate / mdt - round(candidate / mdt)) < 1.0e-9
                and abs(interval / candidate - round(interval / candidate)) < 1.0e-9):
            return float(candidate)
        k += 1
    return float(interval)


def resolve_fresh_defaults(
    *, filter_name: str | None, members: int | None, observation_bin_s: float | None,
    stream_specs, obs_locations, cfg=None, ensemble_truncation: int | None = None,
    interval_s: float = 3600.0,
) -> dict:
    """What the fresh door runs after its defaults, and which settings it
    defaulted (the receipt's ``defaults``): the filter, the member count,
    the observation bin and the stream roster."""
    from .da.ensemble import recut_config
    from .da.options import EnsembleOptions

    defaulted = []
    name = DEFAULT_FRESH_FILTER if filter_name is None else str(filter_name)
    if filter_name is None:
        defaulted.append("filter")
    ensemble = name == "letkf"
    count = members
    if count is None:
        count = DEFAULT_FRESH_MEMBERS if ensemble else 1
        defaulted.append("members")
    bin_s = observation_bin_s
    if bin_s is None and ensemble:
        member_dt = None
        if cfg is not None:
            truncation = int(EnsembleOptions().truncation if ensemble_truncation is None else ensemble_truncation)
            try:
                member_dt = float(recut_config(cfg, truncation).dt_s)
            except Exception:  # noqa: BLE001 - a truncation the recut refuses is the init door's refusal to name
                member_dt = None
        bin_s = default_observation_bin_s(
            float(cfg.dt_s) if cfg is not None else DEFAULT_FRESH_OBSERVATION_BIN_S, member_dt, interval_s)
        defaulted.append("observation_bin_s")
    specs = list(stream_specs or [])
    if not specs and not (obs_locations or []):
        specs = list(DEFAULT_FRESH_STREAMS if ensemble else DEFAULT_FRESH_POINT_STREAMS)
        defaulted.append("streams")
    return {
        "filter": name, "members": int(count), "observation_bin_s": bin_s,
        "stream_specs": specs, "defaulted": defaulted,
    }


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(moment: dt.datetime) -> str:
    return moment.astimezone(dt.timezone.utc).isoformat(timespec="seconds")


def _write_receipt(path: Path, payload: dict) -> dict:
    payload = dict(payload)
    payload["schema"] = DA_RECEIPT_SCHEMA
    payload.pop("self_sha256", None)
    payload["self_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    payload["receipt_path"] = str(path)
    return payload


def read_da_receipt(path: str | Path) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != DA_RECEIPT_SCHEMA:
        raise ValueError(f"{path} is not a {DA_RECEIPT_SCHEMA} receipt")
    stated = payload.pop("self_sha256", None)
    if stated != hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest():
        raise ValueError(f"DA receipt {path} self-hash mismatch")
    payload["self_sha256"] = stated
    return payload


def _checkpoint_identity(path: Path, cfg: ArwenGlobalConfig) -> dict:
    metadata, _ = read_checkpoint(
        path, expected_config_hash=cfg.config_hash,
        semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator,
    )
    return {
        "path": str(path), "self_sha256": metadata["self_sha256"],
        "step": int(metadata["step"]), "time_s": float(metadata["time_s"]),
    }


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

def init(
    cfg: ArwenGlobalConfig, outdir: str | Path, *, filter_name: str | None = None,
    members: int = 1, from_checkpoint: str | Path | None = None,
    analysis_time_utc: str | None = None, overwrite: bool = False,
    config_path: str | Path | None = None, ensemble_truncation: int | None = None,
    options: AssimilationOptions | None = None, control_options=None,
    additive_inflation_fraction: float | None = None, filter_overrides=None,
    filter_settings: dict | None = None,
) -> dict:
    """Build the ensemble from an analysis and write its manifest."""
    start = time.perf_counter()
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / ENSEMBLE_MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"an ensemble manifest exists at {manifest_path}; pass --overwrite "
            "to build the ensemble again"
        )
    filter = resolve_filter(
        filter_name, options=options, members=members, truncation=ensemble_truncation,
        control_options=control_options, additive_inflation_fraction=additive_inflation_fraction,
        filter_overrides=filter_overrides,
        filter_settings=filter_settings,
    )
    if from_checkpoint is not None:
        deterministic = Path(from_checkpoint)
        _checkpoint_identity(deterministic, cfg)
        source = "checkpoint"
    else:
        # Its own name, outside the runner's arwen_global_step*.npz and the
        # cycle's arwen_global_analysis_step*.npz families, so the cycle
        # that restarts from it does not have to overwrite it.
        deterministic = output / INITIAL_STATE_NAME
        if deterministic.exists() and not overwrite:
            raise FileExistsError(
                f"{deterministic} exists; pass --overwrite to write the cold start again"
            )
        transform = build_transform(cfg)
        model, cold = build_model_and_cold_state(cfg, transform, scratch_destination=output)
        write_checkpoint(
            deterministic, cold, config_hash=cfg.config_hash,
            to_numpy=transform.backend.to_numpy,
            semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator,
        )
        source = "cold start from the config's initial state"
        if analysis_time_utc is None:
            configured = None
            try:
                configured = resolve_start_time(cfg, None)
            except ValueError:
                configured = None
            analysis_time_utc = None if configured is None else _iso(configured)
    manifest = filter.init(
        cfg, output, deterministic, members=int(members),
        analysis_time_utc=analysis_time_utc,
    )
    receipt = {
        "door": "woof global da init",
        "config_hash": cfg.config_hash,
        "config": None if config_path is None else str(config_path),
        "filter": filter.name,
        "members": len(manifest["members"]),
        "deterministic": {**manifest["deterministic"], "source": source},
        "analysis_checkpoint": _checkpoint_identity(deterministic, cfg),
        "ensemble_manifest": {"path": str(manifest_path), "self_sha256": manifest["self_sha256"]},
        "analysis_time_utc": analysis_time_utc,
        "wall_seconds": time.perf_counter() - start,
        "status": "pass",
    }
    return _write_receipt(output / DA_RECEIPT_NAME, receipt)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def analyze(
    cfg: ArwenGlobalConfig, checkpoint: str | Path, obs_locations: list[str],
    outdir: str | Path, *, analysis_time=None, options: AssimilationOptions | None = None,
    overwrite: bool = False, filter_name: str | None = None,
    config_path: str | Path | None = None, print_table=print,
) -> dict:
    """One analysis of a checkpoint against observation tables, the report
    with its scorecard beside the analysis checkpoint, and the receipt."""
    start = time.perf_counter()
    filter = resolve_filter(filter_name)
    if filter.members != 1:
        raise ValueError(
            f"the analyze leg forms one deterministic analysis; the {filter.name} "
            "filter analyses an ensemble through the cycle leg"
        )
    report = assimilate(
        cfg, checkpoint, obs_locations, outdir, analysis_time=analysis_time,
        options=options, overwrite=overwrite,
    )
    if print_table is not None:
        print_table(render_table(report["scorecard"]))
    output = Path(outdir)
    receipt = {
        "door": "woof global da analyze",
        "config_hash": cfg.config_hash,
        "config": None if config_path is None else str(config_path),
        "filter": filter.name,
        "background": report["background"],
        "analysis_checkpoint": _checkpoint_identity(Path(report["analysis"]["path"]), cfg),
        "report": {"path": report["report_path"], "self_sha256": report["self_sha256"]},
        "scorecard": report["scorecard"],
        "lineage": report["lineage"],
        "gate_of_record": report["gate_of_record"],
        "wall_seconds": time.perf_counter() - start,
        "status": "pass" if report["status"] == "pass" and report["scorecard"]["verdict"] == "complete"
        else ("incomplete" if report["status"] == "pass" else "fail"),
    }
    return _write_receipt(output / DA_RECEIPT_NAME, receipt)


# ---------------------------------------------------------------------------
# cycle
# ---------------------------------------------------------------------------

def cycle(
    cfg: ArwenGlobalConfig, outdir: str | Path, *, obs_locations: list[str] | None = None,
    stream_specs: list[str] | None = None, cycles: int, start_utc=None,
    interval_s: float = 3600.0, restart: str | Path | None = None,
    ensemble: str | Path | None = None, options: AssimilationOptions | None = None,
    until_s: float | None = None, keep_backgrounds: bool = False,
    partial_analyses: bool = True, filter_name: str | None = None,
    overwrite: bool = False, progress=None, analysis_progress=None,
    config_path: str | Path | None = None, fetch_dir: str | Path | None = None,
    ensemble_truncation: int | None = None, members: int | None = None,
    control_options=None, additive_inflation_fraction: float | None = None,
    observation_bin_s: float | None = None, anchor_spec: str | None = None,
    increment_application: str = "iau", information_cutoff_utc: str | None = None,
    filter_overrides=None, filter_settings: dict | None = None,
    metal: bool = False,
) -> dict:
    """The hourly cycle through the filter and the streams; the run's
    receipt with the ``cycle`` record, the ensemble manifest re-written at
    the last analysis, and the DA receipt naming the analysis handed back.
    ``observation_bin_s`` compares reports at their own bin instants
    (amendment B; None at the analysis instant), ``anchor_spec`` names the
    weak low-pass constraint (E), ``increment_application`` ``iau``
    distributes the increment through the window (D), and
    ``information_cutoff_utc`` is recorded with every window's latency
    classes (F)."""
    start = time.perf_counter()
    output = Path(outdir)
    manifest = None
    if ensemble is not None:
        manifest = read_ensemble_manifest(ensemble)
        if filter_name is not None and manifest["filter"] != filter_name:
            raise ValueError(
                f"the ensemble manifest {ensemble} was built by the "
                f"{manifest['filter']!r} filter; this cycle asked for {filter_name!r}"
            )
        filter_name = manifest["filter"]
        if manifest["config_hash"] != cfg.config_hash:
            raise ValueError(
                f"the ensemble manifest {ensemble} belongs to config "
                f"{manifest['config_hash'][:12]}, not this config's {cfg.config_hash[:12]}"
            )
        if restart is None:
            det = Path(manifest["deterministic"]["checkpoint"])
            if not det.is_absolute():
                det = Path(ensemble).parent / det
            restart = det
    filter = resolve_filter(
        filter_name, options=options, members=members, truncation=ensemble_truncation,
        control_options=control_options, additive_inflation_fraction=additive_inflation_fraction,
        observation_bin_s=observation_bin_s, filter_overrides=filter_overrides,
        filter_settings=filter_settings,
    )
    _configure_filter(filter, anchor_spec=anchor_spec)
    if manifest is not None and hasattr(filter, "attach"):
        filter.attach(cfg, manifest, Path(ensemble))
    streams = [resolve_stream(spec) for spec in (stream_specs or [])]
    receipt = run_cycle(
        cfg, output, obs_locations=list(obs_locations or []), cycles=cycles,
        start_utc=start_utc, interval_s=interval_s, restart=restart,
        overwrite=overwrite, options=options, until_s=until_s,
        keep_backgrounds=keep_backgrounds, partial_analyses=partial_analyses,
        progress=progress, analysis_progress=analysis_progress,
        filter=filter, streams=streams, fetch_dir=fetch_dir,
        increment_application=increment_application,
        information_cutoff=information_cutoff_utc,
        metal=metal,
    )
    record = receipt["cycle"]
    applied = [r for r in record["analyses"] if r.get("applied") and r.get("analysis")]
    handed_back = None
    manifest_payload = None
    if applied:
        last = applied[-1]
        handed_back = _checkpoint_identity(Path(last["analysis"]["path"]), cfg)
        manifest_payload = filter.manifest(
            cfg, output, Path(last["analysis"]["path"]),
            analysis_time_utc=last["analysis_time_utc"], previous=manifest,
        )
    da_receipt = {
        "door": "woof global da cycle",
        "config_hash": cfg.config_hash,
        "config": None if config_path is None else str(config_path),
        "filter": filter.name,
        "streams": record["streams"],
        "run_receipt": {"path": receipt["receipt_path"], "self_sha256": receipt["self_sha256"]},
        "cycles": {
            "planned": len(record["planned_analysis_steps"]),
            "completed": record["completed"], "applied": record["applied"],
            "carried": record["carried"], "partial": record["partial"],
        },
        "scorecards": record["scorecards"],
        "assessments": record.get("assessments"),
        "budget": record["budget"],
        "wall_seconds_per_model_hour_cycled": record.get("wall_seconds_per_model_hour_cycled"),
        "analysis_checkpoint": handed_back,
        "ensemble_manifest": None if manifest_payload is None else {
            "path": str(output / ENSEMBLE_MANIFEST_NAME),
            "self_sha256": manifest_payload["self_sha256"],
        },
        "lineage": None if not applied else applied[-1]["lineage"],
        "settings": {
            "observation_bin_s": observation_bin_s,
            "increment_application": increment_application,
            "anchor": anchor_spec,
            "control_options": None if control_options is None else control_options.identity(),
            "additive_inflation_fraction": additive_inflation_fraction,
            "filter_overrides": dict(filter_overrides or {}),
        },
        "causal": _causal_record(record, information_cutoff_utc),
        "stream_roster": stream_roster(record, information_cutoff_utc),
        "wall_seconds": time.perf_counter() - start,
        # Engineering validity is the status (amendment G): every cycle
        # whose streams reached the operators with an O-A is complete.
        "status": (
            "fail" if receipt["status"] != "pass" or handed_back is None
            else "pass" if record["scorecards"]["complete_cycles"] == record["completed"]
            else "incomplete"
        ),
    }
    return _write_receipt(output / DA_RECEIPT_NAME, da_receipt)


def stream_roster(record: dict, information_cutoff_utc: str | None) -> dict:
    """The receipt's stream roster: per cycle the filter's ``stream_roster``
    (offered, duplicates, batched, refused by name, assimilated, withheld,
    O-B and O-A per variable), and per stream across the cycles its
    latency class and basis from the streams module, the information
    cutoff the cycle ran under, the counts per cycle, and the cycles in
    which it was offered and refused whole (a defect, named)."""
    from .obs_streams import LATENCY_CLASSES, STREAMS

    cycles = []
    summary: dict[str, dict] = {}
    defects: list[str] = []
    fetch_failures: list[str] = []
    configured = [
        str(entry.get("name")) for entry in (record.get("streams") or [])
        if isinstance(entry, dict) and entry.get("name") in STREAMS
    ]

    def new_row(source, variables):
        spec = STREAMS.get(source)
        return {
            "latency_class": None if spec is None else spec.latency_class,
            "latency_class_meaning": None if spec is None else LATENCY_CLASSES[spec.latency_class],
            "latency_basis": None if spec is None else spec.latency_basis,
            "door": None if spec is None else spec.door,
            "in_stream_table": spec is not None,
            "cycles": [], "offered_per_cycle": [], "assimilated_per_cycle": [], "withheld_per_cycle": [],
            "cross_stream_duplicates_per_cycle": [], "latency_unverified_per_cycle": [],
            "refused_whole_cycles": [], "empty_fetch_cycles": [], "failed_fetch_cycles": [],
            "variables": sorted(variables or {}),
        }

    for analysis in record.get("analyses", []):
        roster = analysis.get("stream_roster") or {}
        label = analysis.get("analysis_time_utc")
        streams_here = dict(roster.get("streams") or {})
        # A configured stream whose fetch for this window held nothing
        # (EMPTY) or did not answer (failed) offered no rows and is absent
        # from the filter's roster: it is listed here with zero counts and
        # the reason, so the receipt states the gap instead of omitting the
        # stream as if it had never been asked for.
        for entry in analysis.get("fetch") or []:
            extra = entry.get("extra") or {}
            source = entry.get("stream")
            if not source or source in streams_here or not (extra.get("empty") or extra.get("failed")):
                continue
            outcome = "failed" if extra.get("failed") else "empty"
            streams_here[source] = {
                "offered": 0, "cross_stream_duplicates": 0, "batched": 0, "assimilated": 0, "withheld": 0,
                "refused": {}, "refused_whole": False, "latency_unverified": 0, "variables": {},
                "fetch": outcome, "reason": extra.get("reason"),
                "latency_class": extra.get("latency_class_of_stream"),
            }
            if outcome == "failed":
                fetch_failures.append(f"{label}: {source}: {extra.get('reason')}")
        cycles.append({"analysis_time_utc": label, "status": analysis.get("status"),
                       "streams": streams_here,
                       "rows_outside_window": roster.get("rows_outside_window"),
                       "defects": list(roster.get("defects") or [])})
        for name in roster.get("defects") or []:
            defects.append(f"{label}: {name}")
        for source, entry in streams_here.items():
            row = summary.setdefault(source, new_row(source, entry.get("variables")))
            row["cycles"].append(label)
            row["offered_per_cycle"].append(int(entry.get("offered", 0)))
            row["assimilated_per_cycle"].append(int(entry.get("assimilated", 0)))
            row["withheld_per_cycle"].append(int(entry.get("withheld", 0)))
            row["cross_stream_duplicates_per_cycle"].append(int(entry.get("cross_stream_duplicates", 0)))
            row["latency_unverified_per_cycle"].append(int(entry.get("latency_unverified", 0)))
            if entry.get("refused_whole"):
                row["refused_whole_cycles"].append(label)
            if entry.get("fetch") == "empty":
                row["empty_fetch_cycles"].append(label)
            if entry.get("fetch") == "failed":
                row["failed_fetch_cycles"].append(label)
    # A stream configured on the door that reached no cycle's roster at all
    # (no rows in any window, no fetch record either) is still a stream the
    # operator asked for: it is listed with zero counts per cycle.
    for source in configured:
        if source not in summary:
            row = new_row(source, {})
            row["cycles"] = [c["analysis_time_utc"] for c in cycles]
            for key in ("offered_per_cycle", "assimilated_per_cycle", "withheld_per_cycle",
                        "cross_stream_duplicates_per_cycle", "latency_unverified_per_cycle"):
                row[key] = [0] * len(cycles)
            row["note"] = "configured on the door; offered no rows in any window"
            summary[source] = row
    return {
        "information_cutoff_utc": information_cutoff_utc,
        "cycles": cycles,
        "summary": summary,
        "defects": defects,
        "fetch_failures": fetch_failures,
        "rule": (
            "every stream whose rows the tables carry is analysed by the letkf door; the streams module's "
            "measured latency class and basis ride beside each stream; a stream offered in a window and "
            "refused whole is a defect named here, never a silent gap; a configured stream whose window "
            "fetch was empty or failed is listed with zero rows and the reason"
        ),
    }


def _causal_record(record: dict, information_cutoff_utc: str | None) -> dict:
    """Amendment F: the information cutoff, the latency class of every
    fetched object per cycle, and whether a report's arrival time was
    measured at all."""
    classes: dict[str, int] = {}
    unverified = 0
    for analysis in record.get("analyses", []):
        for entry in analysis.get("fetch") or []:
            extra = entry.get("extra") or {}
            if extra.get("empty") or extra.get("failed"):
                # No object arrived: an empty or failed fetch is counted by
                # its outcome, not as an object whose arrival time nobody
                # measured (that reading belongs to a table on disk).
                label = "failed" if extra.get("failed") else "empty"
                classes[label] = classes.get(label, 0) + 1
                continue
            label = str(entry.get("latency_class") or "unverified")
            classes[label] = classes.get(label, 0) + 1
            if label == "unverified":
                unverified += 1
    arrived = {k: v for k, v in classes.items() if k not in ("empty", "failed")}
    rows = record.get("information_cutoff") or {}
    return {
        "information_cutoff_utc": information_cutoff_utc,
        "latency_classes": classes,
        "rows": {k: rows.get(k) for k in ("offered", "kept", "after_cutoff", "latency_unverified")},
        "mode": (
            "latency unverified (tables on disk: a historical case whose arrival times "
            "nobody measured)" if unverified and len(arrived) == 1
            else "fast hourly" if arrived.get("fast") and len(arrived) == 1
            else "delayed replay" if arrived and not arrived.get("fast")
            else "mixed"
        ),
        "rule": (
            "every fetched object carries its measurement window, first receipt time, "
            "publication time where known and a latency class (fast, replay, retrospective, "
            "unverified); every row of a v2 table carries its measurement, nominal, published "
            "and received times and revision, and a row received after the cutoff is dropped "
            "and counted; a report is never re-assimilated (the chain refuses it); a case "
            "whose arrival times were not measured is labelled latency unverified"
        ),
    }


# ---------------------------------------------------------------------------
# fresh
# ---------------------------------------------------------------------------

def _toml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("a TOML number must be finite")
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k} = {_toml_scalar(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"cannot write {type(value).__name__} into TOML")


def dump_toml(document: dict, *, header: str | None = None) -> str:
    """A TOML text of a nested dictionary of tables and scalars (the
    shape every WOOF global config has); inline tables for a dictionary
    inside a table whose values are scalars."""
    lines: list[str] = []
    if header:
        lines.extend(f"# {line}" if line else "#" for line in header.splitlines())
        lines.append("")

    def emit(table: dict, prefix: str) -> None:
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        children = {k: v for k, v in table.items() if isinstance(v, dict)}
        if prefix:
            lines.append(f"[{prefix}]")
        for key, value in scalars.items():
            lines.append(f"{key} = {_toml_scalar(value)}")
        if prefix or scalars:
            lines.append("")
        for key, child in children.items():
            nested = any(isinstance(v, dict) for v in child.values())
            if prefix and not nested and key in ("native_adapter_options",):
                # The physics options are read as an inline table.
                lines.insert(len(lines) - 1, f"{key} = {_toml_scalar(child)}")
                continue
            emit(child, f"{prefix}.{key}" if prefix else key)

    emit(document, "")
    return "\n".join(lines).rstrip() + "\n"


def derive_fresh_config(
    base_config: str | Path, *, analysis_grib: str | None, analysis_mapping: str | None,
    duration_s: float, name_suffix: str = "fresh",
) -> dict:
    """The run configuration ``fresh`` writes: the base TOML with the
    initial analysis pointed at the fetched object (an analysis-mode base)
    and the duration set to the cycle span plus the forecast."""
    raw = tomllib.loads(Path(base_config).read_text(encoding="utf-8"))
    document = json.loads(json.dumps(raw))  # a deep copy of plain data
    initial = document.setdefault("initial", {})
    if str(initial.get("mode", "analytic")).lower() == "analysis":
        if analysis_grib is None:
            raise ValueError(
                "the base config initialises from an analysis but fresh has none "
                "to point it at"
            )
        initial["analysis_grib"] = Path(analysis_grib).as_posix()
        if analysis_mapping:
            initial["analysis_mapping"] = analysis_mapping
    time_table = document.setdefault("time", {})
    time_table["duration_s"] = float(duration_s)
    head = document.setdefault("arwen_global", {})
    if "name" in head and not str(head["name"]).endswith(f"-{name_suffix}"):
        head["name"] = f"{head['name']}-{name_suffix}"
    return document


def _floor_hour(moment: dt.datetime) -> dt.datetime:
    return moment.replace(minute=0, second=0, microsecond=0)


def fresh(
    base_config: str | Path, outdir: str | Path, *, stream_specs: list[str] | None = None,
    obs_locations: list[str] | None = None, analysis_grib: str | None = None,
    analysis_cycle=None, start_utc=None, until_utc=None, forecast_hours: float = DEFAULT_FORECAST_HOURS,
    interval_s: float = 3600.0, filter_name: str | None = None, members: int | None = None,
    options: AssimilationOptions | None = None, observation_latency_s: float = DEFAULT_OBSERVATION_LATENCY_S,
    overwrite: bool = False, now: dt.datetime | None = None, progress=None,
    analysis_progress=None, fetch_engine: str = "auto", ensemble_truncation: int | None = None,
    control_options=None, additive_inflation_fraction: float | None = None,
    observation_bin_s: float | None = None, anchor_spec: str | None = None,
    increment_application: str = "iau", cutoff_utc=None, filter_overrides=None,
    defaulted_names: tuple[str, ...] = (), filter_settings: dict | None = None,
    keep_backgrounds: bool = False,
) -> dict:
    """The one command: a fresh global analysis, the latest constructible
    from the information available by a declared cutoff (amendment F:
    ``cutoff_utc``, default now; the newest observation hour is the
    cutoff minus ``observation_latency_s``, floored).

    1. the analysis: for an analysis-mode base config the GDAS cycle
       (``analysis_cycle``, default the newest published) is fetched
       through the tree's fetch door, or ``analysis_grib`` on disk is
       digested; an analytic base config (a smoke or OSSE case) needs
       ``start_utc`` and fetches nothing, and the receipt says so;
    2. the cycle span: hourly analyses from the analysis instant to the
       newest observation hour (``until_utc``, default the current hour
       minus ``observation_latency_s``, floored);
    3. the derived configuration (``fresh-config.toml`` in the output,
       duration = cycle span + ``forecast_hours``);
    4. ``init`` unless an ensemble manifest already sits in the output;
    5. ``cycle`` through every stream and table to the last hour
       (``keep_backgrounds`` writes each hour's background checkpoint
       beside its analysis, as ``cycle`` does; nothing analysed changes);
    6. the receipt names the analysis checkpoint handed back and the
       forecast command that starts from it.

    What the command runs when it names nothing else is the completed
    system (:func:`resolve_fresh_defaults`, the module constants
    ``DEFAULT_FRESH_*``): the ensemble filter with 32 members, 600 s
    bins and the shipped stream roster; ``filter_name``, ``members``,
    ``observation_bin_s`` and the streams named replace those parts, and
    the receipt's ``defaults`` lists which parts were defaulted.
    """
    start = time.perf_counter()
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    now = now or _now()
    if cutoff_utc is not None:
        cutoff = cutoff_utc if isinstance(cutoff_utc, dt.datetime) else parse_valid_time(str(cutoff_utc))
        if cutoff is None:
            raise ValueError(f"--cutoff-utc {cutoff_utc!r} is not an ISO-8601 instant")
        if cutoff > now + dt.timedelta(seconds=1.0):
            raise ValueError(
                f"the information cutoff {_iso(cutoff)} lies in the future of now {_iso(now)}: "
                "an analysis cannot use information that does not exist yet"
            )
    else:
        cutoff = now
    base_raw = tomllib.loads(Path(base_config).read_text(encoding="utf-8"))
    analysis_mode = str(base_raw.get("initial", {}).get("mode", "analytic")).lower() == "analysis"
    cycle_moment = None
    if analysis_cycle is not None:
        cycle_moment = analysis_cycle if isinstance(analysis_cycle, dt.datetime) else parse_valid_time(str(analysis_cycle))
        if cycle_moment is None:
            raise ValueError(f"--analysis-cycle {analysis_cycle!r} is not an ISO-8601 instant")
    analysis = None
    if analysis_mode:
        analysis = fetch_analysis(
            output / "analysis", grib=analysis_grib, cycle=cycle_moment,
            engine=fetch_engine, progress=progress or (lambda *_: None),
        )
        start_moment = parse_valid_time(analysis.cycle_utc)
        mapping = str(base_raw.get("initial", {}).get("analysis_mapping") or analysis.mapping)
    else:
        if start_utc is None and cycle_moment is None:
            raise ValueError(
                "the base config initialises analytically, so fresh needs "
                "--start-utc (the instant model time zero stands for); "
                "nothing is fetched for an analytic start"
            )
        start_moment = cycle_moment if start_utc is None else (
            start_utc if isinstance(start_utc, dt.datetime) else parse_valid_time(str(start_utc))
        )
        if start_moment is None:
            raise ValueError(f"--start-utc {start_utc!r} is not an ISO-8601 instant")
        mapping = None
    start_moment = start_moment.astimezone(dt.timezone.utc)
    if until_utc is None:
        last_hour = _floor_hour(cutoff - dt.timedelta(seconds=float(observation_latency_s)))
    else:
        last_hour = until_utc if isinstance(until_utc, dt.datetime) else parse_valid_time(str(until_utc))
        if last_hour is None:
            raise ValueError(f"--until-utc {until_utc!r} is not an ISO-8601 instant")
    span_s = (last_hour - start_moment).total_seconds()
    cycles = int(math.floor(span_s / float(interval_s) + 1.0e-9))
    if cycles < 1:
        raise ValueError(
            f"the analysis instant {_iso(start_moment)} is not a whole interval "
            f"before the newest observation hour {_iso(last_hour)}: nothing to "
            "cycle yet (the observations lag real time by about "
            f"{observation_latency_s:g} s); pass --until-utc or wait an hour"
        )
    cycle_span_s = cycles * float(interval_s)
    duration_s = cycle_span_s + float(forecast_hours) * 3600.0
    document = derive_fresh_config(
        base_config, analysis_grib=None if analysis is None else analysis.path,
        analysis_mapping=mapping, duration_s=duration_s,
    )
    config_path = output / FRESH_CONFIG_NAME
    if config_path.exists() and not overwrite:
        raise FileExistsError(f"{config_path} exists; pass --overwrite to derive it again")
    config_path.write_text(dump_toml(document, header=(
        f"Derived by woof global da fresh from {Path(base_config).as_posix()} at "
        f"{_iso(now)}: {cycles} hourly analyses from {_iso(start_moment)} to "
        f"{_iso(start_moment + dt.timedelta(seconds=cycle_span_s))}, then "
        f"{forecast_hours:g} h of forecast"
    )), encoding="utf-8", newline="\n")
    cfg = load_config(config_path)
    resolved = resolve_fresh_defaults(
        filter_name=filter_name, members=members, observation_bin_s=observation_bin_s,
        stream_specs=stream_specs, obs_locations=obs_locations, cfg=cfg,
        ensemble_truncation=ensemble_truncation, interval_s=interval_s,
    )
    filter_name = resolved["filter"]
    members = resolved["members"]
    observation_bin_s = resolved["observation_bin_s"]
    stream_specs = resolved["stream_specs"]
    defaulted = list(resolved["defaulted"]) + [n for n in defaulted_names if n not in resolved["defaulted"]]
    if control_options is None and filter_name == "letkf":
        from .da_control import ControlOptions

        control_options = ControlOptions(hybrid_beta=DEFAULT_FRESH_HYBRID_BETA)
        defaulted.append("hybrid_beta")
    manifest_path = output / ENSEMBLE_MANIFEST_NAME
    init_receipt = None
    if not manifest_path.exists() or overwrite:
        init_receipt = init(
            cfg, output, filter_name=filter_name, members=members,
            analysis_time_utc=_iso(start_moment), overwrite=True, config_path=config_path,
            ensemble_truncation=ensemble_truncation, options=options,
            control_options=control_options, additive_inflation_fraction=additive_inflation_fraction,
            filter_overrides=filter_overrides,
            filter_settings=filter_settings,
        )
    manifest = read_ensemble_manifest(manifest_path)
    cycle_receipt = cycle(
        cfg, output, obs_locations=obs_locations, stream_specs=stream_specs,
        cycles=cycles, start_utc=start_moment, interval_s=interval_s,
        ensemble=manifest_path, options=options, until_s=cycle_span_s,
        filter_name=filter_name, overwrite=True, progress=progress,
        analysis_progress=analysis_progress, config_path=config_path,
        control_options=control_options, additive_inflation_fraction=additive_inflation_fraction,
        observation_bin_s=observation_bin_s, anchor_spec=anchor_spec,
        increment_application=increment_application, information_cutoff_utc=_iso(cutoff),
        filter_overrides=filter_overrides,
        filter_settings=filter_settings, keep_backgrounds=keep_backgrounds,
    )
    handed_back = cycle_receipt["analysis_checkpoint"]
    forecast_command = None
    if handed_back is not None:
        forecast_command = (
            f"woof global da forecast {config_path.as_posix()} --analysis "
            f"{Path(handed_back['path']).as_posix()} --outdir {(output / 'forecast').as_posix()}"
        )
    receipt = {
        "door": "woof global da fresh",
        "base_config": str(base_config),
        "config": str(config_path),
        "config_hash": cfg.config_hash,
        "now_utc": _iso(now),
        "information_cutoff_utc": _iso(cutoff),
        "cutoff_rule": (
            "the analysis handed back is the latest constructible from information "
            "available by the cutoff: the newest observation hour is the cutoff minus "
            "the stated observation latency, floored to the hour; every window's "
            "fetch records carry their first receipt time and latency class"
        ),
        "analysis": None if analysis is None else analysis.__dict__,
        "initial_state": (
            "analysis (fetched)" if analysis is not None and analysis.manifest is not None
            else "analysis (given on disk)" if analysis is not None
            else "analytic (the base config's initial state; nothing fetched)"
        ),
        "start_utc": _iso(start_moment),
        "last_observation_hour_utc": _iso(start_moment + dt.timedelta(seconds=cycle_span_s)),
        "observation_latency_s": float(observation_latency_s),
        "cycles": cycles,
        "interval_s": float(interval_s),
        "forecast_hours": float(forecast_hours),
        "filter": manifest["filter"],
        "defaults": {
            "defaulted": defaulted,
            "members": int(members), "observation_bin_s": observation_bin_s,
            "streams": list(stream_specs),
            "hybrid_beta": None if control_options is None else float(control_options.hybrid_beta),
            "rule": "the fresh door runs the completed system when the command names nothing else: "
                    "the ensemble filter, 32 members, 600 s bins (or the smallest multiple of the "
                    "step above it that divides the interval), the hybrid covariance at beta 0.75 "
                    "and the shipped stream roster; naming a filter, a member count, a bin, a beta "
                    "or any stream replaces that part",
        },
        "init": None if init_receipt is None else {
            "receipt_self_sha256": init_receipt["self_sha256"],
            "ensemble_manifest": init_receipt["ensemble_manifest"],
        },
        "cycle": {
            "receipt_self_sha256": cycle_receipt["self_sha256"],
            "run_receipt": cycle_receipt["run_receipt"],
            "cycles": cycle_receipt["cycles"],
            "status": cycle_receipt["status"],
        },
        "scorecards": cycle_receipt["scorecards"],
        "assessments": cycle_receipt.get("assessments"),
        "budget": cycle_receipt["budget"],
        "settings": cycle_receipt.get("settings"),
        "causal": cycle_receipt.get("causal"),
        "stream_roster": cycle_receipt.get("stream_roster"),
        "analysis_checkpoint": handed_back,
        "ensemble_manifest": cycle_receipt["ensemble_manifest"],
        "lineage": cycle_receipt["lineage"],
        "forecast_command": forecast_command,
        "wall_seconds": time.perf_counter() - start,
        "status": cycle_receipt["status"],
    }
    return _write_receipt(output / DA_RECEIPT_NAME, receipt)


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------

def _library_store(ensemble: str | Path) -> Path:
    """The library ensemble store under whatever `--ensemble` names.

    `da init` writes two layered manifests: the door's `da-ensemble.json`
    in the directory it was given, and the library's
    `arwen-global-ensemble.json` beside the member checkpoints in the
    store the door manifest's `ensemble_store` names.  The library reader
    only knows the second, so handing it the directory `da init` was given
    made it look for a filename nothing had written and fail on an OS
    error naming a path the caller never chose.  This resolves the door
    spelling to the library store by the same rule the cycle door already
    follows, accepts either manifest or either directory, and refuses by
    name rather than by errno when it is none of them.
    """
    from .da.ensemble import ENSEMBLE_MANIFEST_NAME as LIBRARY_MANIFEST_NAME
    from .da_filter import ENSEMBLE_STORE_DIR

    given = Path(ensemble)
    if given.is_file():
        if given.name == LIBRARY_MANIFEST_NAME:
            return given
        if given.name != ENSEMBLE_MANIFEST_NAME:
            raise ValueError(
                f"{given} is neither the door's {ENSEMBLE_MANIFEST_NAME} nor the "
                f"library's {LIBRARY_MANIFEST_NAME}; --ensemble takes an ensemble "
                f"store, the directory `da init --outdir` was given, or either manifest")
        manifest_path, base = given, given.parent
    elif given.is_dir():
        if (given / LIBRARY_MANIFEST_NAME).is_file():
            return given
        manifest_path, base = given / ENSEMBLE_MANIFEST_NAME, given
        if not manifest_path.is_file():
            raise ValueError(
                f"{given} holds neither {LIBRARY_MANIFEST_NAME} nor "
                f"{ENSEMBLE_MANIFEST_NAME}, so it is not an ensemble store and not a "
                f"directory `da init` wrote; run `woof global da init --outdir {given}` first")
    else:
        raise ValueError(f"--ensemble {given} does not exist")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    store = Path(manifest.get("ensemble_store") or (base / ENSEMBLE_STORE_DIR))
    if not store.is_absolute():
        store = base / store
    if not (store / LIBRARY_MANIFEST_NAME).is_file():
        raise ValueError(
            f"{manifest_path} names its ensemble store as {store}, which carries no "
            f"{LIBRARY_MANIFEST_NAME}; the members that manifest was written for are "
            f"not there")
    return store


def localisation(
    cfg: ArwenGlobalConfig, ensemble: str | Path, out: str | Path, *, step: int | None = None,
    config_path: str | Path | None = None,
) -> dict:
    """The vertical-localisation derivation on an ensemble store: the
    members' vertical correlations per report class and region, the
    Gaspari-Cohn cutoff fitted to each, written as a receipt at ``out``
    with the cutoffs beside the :class:`FilterOptions` fields they are
    (:mod:`woof.globe.da.localisation`).  Runs on the host (a
    numpy float64 transform at the members' truncation), so it needs no
    card."""
    from .da.localisation import derive_from_store
    from .da.options import FilterOptions

    start = time.perf_counter()
    store = _library_store(ensemble)
    derivation = derive_from_store(cfg, store, step=step)
    carried = FilterOptions().identity()
    receipt = {
        "schema": "gpuwm.arwen-global-da-localisation/v1",
        "door": "woof global da localisation",
        "config_hash": cfg.config_hash,
        "config_path": None if config_path is None else str(config_path),
        "ensemble": str(ensemble),
        "step": step,
        "members": derivation["members"],
        "truncation": derivation["truncation"],
        "rule": derivation["rule"],
        "cutoffs": derivation["cutoffs"],
        "carried_by_filter_options": {key: carried.get(key) for key in sorted(derivation["cutoffs"]["global"])},
        "profiles": derivation["classes"],
        "mean_p_full_hpa_by_level": derivation["mean_p_full_hpa_by_level"],
        "noise_floor_corr2": derivation["noise_floor_corr2"],
        "members_read": derivation["members_read"],
        "wall_seconds": time.perf_counter() - start,
        "status": "pass",
    }
    return _write_receipt(Path(out), receipt)


def forecast(
    cfg: ArwenGlobalConfig, outdir: str | Path, *, analysis: str | Path,
    until_s: float | None = None, overwrite: bool = False, progress=None,
    config_path: str | Path | None = None,
) -> dict:
    """The forecast from an analysis checkpoint: the runner's restart,
    with the DA receipt naming the analysis it started from."""
    start = time.perf_counter()
    output = Path(outdir)
    identity = _checkpoint_identity(Path(analysis), cfg)
    receipt = run(cfg, output, restart=analysis, overwrite=overwrite, progress=progress, until_s=until_s)
    da_receipt = {
        "door": "woof global da forecast",
        "config_hash": cfg.config_hash,
        "config": None if config_path is None else str(config_path),
        "analysis_checkpoint": identity,
        "run_receipt": {"path": receipt["receipt_path"], "self_sha256": receipt["self_sha256"]},
        "checkpoints": receipt["checkpoints"],
        "final_step": receipt["final_diagnostics"]["step"],
        "final_time_s": receipt["final_diagnostics"]["time_s"],
        "wall_seconds": time.perf_counter() - start,
        "status": receipt["status"],
    }
    return _write_receipt(output / DA_RECEIPT_NAME, da_receipt)


def analysis_checkpoints(outdir: str | Path) -> list[Path]:
    return sorted(Path(outdir).glob(f"{ANALYSIS_PREFIX}*.npz"))


__all__ = [
    "DA_RECEIPT_NAME",
    "DA_RECEIPT_SCHEMA",
    "DEFAULT_FORECAST_HOURS",
    "DEFAULT_FRESH_FILTER",
    "DEFAULT_FRESH_HYBRID_BETA",
    "DEFAULT_FRESH_MEMBERS",
    "DEFAULT_FRESH_OBSERVATION_BIN_S",
    "DEFAULT_FRESH_POINT_STREAMS",
    "DEFAULT_FRESH_STREAMS",
    "DEFAULT_OBSERVATION_LATENCY_S",
    "FRESH_CONFIG_NAME",
    "analysis_checkpoints",
    "analyze",
    "cycle",
    "default_observation_bin_s",
    "derive_fresh_config",
    "dump_toml",
    "forecast",
    "fresh",
    "init",
    "read_da_receipt",
    "resolve_fresh_defaults",
    "stream_roster",
]
