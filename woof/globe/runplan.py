"""``woof global run-plan``: the machine-facing front door.

The 2.7 engine ships an integration contract, and a desktop application and a
terminal workspace already speak it: discover sources, discover products,
discover physics, probe the machine, resolve a plan, estimate its cost,
execute it, and reattach to a running job through three durable documents.
This module gives THIS distribution the same contract, so a client that
drives ``woof`` drives ``woof global`` by changing the module it spawns and
the console script it calls.  Nothing else about the client changes: the plan
document is the engine's own ``gpuwm.run-plan.v1``, the event stream is the
engine's ``gpuwm.run-plan.event.v1``, the manifest and the heartbeat are the
engine's, and every schema id below is IMPORTED from the engine rather than
transcribed, so the two cannot drift by a typo.

WHAT IS REUSED AND WHAT IS NOT.  The engine's machinery divides cleanly.

  reused, by import   :class:`woof.runplan.EventStream` (the append-only
                      JSONL writer with its lock, its flush and its dense
                      sequence), :func:`woof.runplan.read_events`, the event
                      tag tuple, every schema id, the heartbeat dataclass and
                      its atomic writer, and the provenance receipt block.

  NOT reused          domain fitting, WPS preparation, the prepared cache,
                      the intent wizard, the moving-nest corridor, the tile
                      planner, the VRAM itemizer.  Every one of them is a
                      question about a regional grid, and this model has one
                      global grid whose cost is priced by
                      :mod:`woof.globe.sizing`.  Calling them would put the
                      engine's name on a number about a run it cannot see.

THE ROUTES.  ``experiment`` is what ``woof global run CONFIG`` executes, and
``go`` is what ``woof global go CONFIG`` executes -- statics, forecast and
pictures.  The engine's ``prepared`` route is REFUSED BY NAME rather than
quietly mapped onto one of these: a prepared plan names a prepared-cache root
and a WPS namelist authority, this package reads neither, and a plan that
asked for it would be started as something else.

WHAT A RUN WRITES, all three in the run directory and all three the engine's
shapes: ``run-manifest.json`` (``gpuwm.run-manifest.v1``, written before any
work starts), ``run-progress.json`` (``gpuwm.run-progress/v1``, the
current-state authority) and ``events.jsonl``
(``gpuwm.run-plan.event.v1``, monotonic and dense).  They land beside the
``status.json`` and the command log the ``go`` and ``run`` doors already
write, which are not replaced: a workspace that polls the small status file
keeps working.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback as _traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

# Every schema id, the event tag tuple, the durable stream writer and its
# reader come from the engine.  A second spelling of any of them is a second
# thing to keep in step, and the one that drifts is the one a client pins.
from woof.runplan import (
    CATALOG_SCHEMA,
    ESTIMATE_SCHEMA,
    EVENT_SCHEMA,
    EVENT_TAGS,
    EVENTS_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    PHYSICS_PROFILES_SCHEMA,
    PLAN_SCHEMA,
    PROBE_SCHEMA,
    RESOLVE_SCHEMA,
    SOURCES_SCHEMA,
    EventStream,
    PlanError,
    read_events,
)

from ._version import CONSOLE_SCRIPT, DISTRIBUTION_NAME, __version__

__all__ = [
    "CATALOG_SCHEMA",
    "ESTIMATE_SCHEMA",
    "EVENTS_FILENAME",
    "EVENT_SCHEMA",
    "EVENT_TAGS",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "PHYSICS_PROFILES_SCHEMA",
    "PLAN_SCHEMA",
    "PROBE_SCHEMA",
    "PROFILE_DOCUMENT_FIELDS_NOT_CARRIED",
    "PROFILE_FIELDS_NOT_CARRIED",
    "REFUSED_ROUTES",
    "RESOLVE_SCHEMA",
    "ROUTES",
    "SOURCES_SCHEMA",
    "EventStream",
    "PlanError",
    "Route",
    "RunPlan",
    "build_plan",
    "estimate_plan",
    "execute_plan",
    "load_plan",
    "physics_profile_menu",
    "prepare_run_directory",
    "probe_environment",
    "producer_block",
    "read_events",
    "register_cli",
    "render_catalog",
    "resolve_plan",
    "route_summaries",
    "run_plan_main",
    "run_plan_module_entry",
    "source_inventory",
    "write_manifest",
]


# ---------------------------------------------------------------------------
# Who produced a document
# ---------------------------------------------------------------------------


_PER_EXPERIMENT = (
    "a profile here groups shipped experiments by the physics MODE they "
    "select, and {} is chosen per experiment in its own [physics] table, so "
    "one value on this row would report one experiment's choice for all of "
    "them.  `woof global run-plan PLAN.json --resolve` reports what the "
    "plan being reviewed actually runs.")

#: Fields the ENGINE's physics-profile rows carry that a row here does not,
#: each with the reason, on the same standard the source registry is held to:
#: an engine field is either carried with an answer or NAMED with why it has
#: none.  A picker written against the engine's document reads this block
#: instead of taking a KeyError on a field that has no meaning here.
PROFILE_FIELDS_NOT_CARRIED = {
    "cumulus_scheme_id": _PER_EXPERIMENT.format("the cumulus scheme"),
    "pbl_scheme_id": _PER_EXPERIMENT.format("the boundary-layer scheme"),
    "longwave_scheme_id": _PER_EXPERIMENT.format("the longwave scheme"),
    "shortwave_scheme_id": _PER_EXPERIMENT.format("the shortwave scheme"),
    "microphysics_scheme_id": _PER_EXPERIMENT.format(
        "the microphysics scheme"),
    "land_surface_scheme_id": _PER_EXPERIMENT.format(
        "the land-surface scheme"),
    "switches": "the engine's switches block is a WRF namelist fragment for "
                "one scheme set.  Selection here is the config's own "
                "[physics] table and a plan does not override it, so there "
                "is no switch table between the two to report.",
    "vertical_levels": "the engine reports the WRF component bounds its own "
                       "schemes aggregate.  The ladder here is the config's "
                       "hybrid A/B table, chosen per experiment, and these "
                       "columns are not WRF columns.",
    "maturity": "the same fact under this document's own names, and MEASURED "
                "against the installed engine rather than declared: "
                "`registered`, `admissible`, `why_not` and the adapter "
                "contract's `admission_status`.",
    "day_only": "the engine's flag marks ONE scheme set that runs shortwave "
                "without longwave.  A row here binds many experiments, each "
                "with its own radiation choice, so the fact is per "
                "experiment and no single value on this row would be true "
                "of all of them.",
    "day_only_reason": "the reason belongs to the field above and is absent "
                       "for the same reason.",
}

#: The same, for the fields the engine's document carries at its TOP level.
PROFILE_DOCUMENT_FIELDS_NOT_CARRIED = {
    "physics_registry_sha256": "the engine's menu is built from one registry "
                               "file it can hash.  This menu is derived from "
                               "the installed adapter manifest and the "
                               "shipped experiment TOMLs, so there is no "
                               "single file whose digest would mean what "
                               "that field means; each row's `registered` "
                               "and `admission_status` are read from that "
                               "manifest when this document is built.",
}


def producer_block() -> dict[str, Any]:
    """Which distribution answered, and which engine it answered on.

    Every document this module emits carries it.  The schema ids are the
    ENGINE's, deliberately, so a client needs no new parser -- which means the
    schema id alone can no longer say who replied, and a client holding two
    ``gpuwm.run-plan.sources.v1`` documents from two commands has to be able
    to tell them apart.  The engine half is read from the installed engine at
    the moment the document is built, never from a pin: the physics a run
    integrates is the installed engine's, and a document that named a version
    from a table would be naming the wrong one on the machine that matters.
    """

    import woof

    from woof.provenance_gate import receipt_block

    return {
        "distribution": DISTRIBUTION_NAME,
        "version": __version__,
        "console_script": CONSOLE_SCRIPT,
        "module": __name__,
        "model": "WOOF global: a hydrostatic spectral global model with its "
                 "own ensemble assimilation",
        "engine": {
            "distribution": "woof",
            "version": str(getattr(woof, "__version__", "unknown")),
            "package_path": str(Path(woof.__file__).resolve().parent),
            "provenance": receipt_block(),
        },
    }


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _jsonable(value: object) -> Any:
    """The engine's own default hook, for the same reason it has one."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


# ---------------------------------------------------------------------------
# Routes and run options
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Route:
    """One existing door of this distribution, named so a plan can ask for it.

    Route names are generic, exactly as the engine's are: a route is a way of
    running the model and never a particular experiment.
    """

    name: str
    summary: str
    door: str
    run_options: frozenset[str]
    stages: tuple[str, ...]


#: The stages a run passes through, in order, per route.  There is no
#: ``fetch`` stage: this model's input is the analysis file the config's
#: ``[initial]`` table names, and the bytes are brought by a separate door
#: (``woof global obs anchors``) that a plan does not drive.  A plan that
#: carries the engine's ``fetch`` block is refused by name below rather than
#: having it silently dropped.
_EXPERIMENT_STAGES = ("initialize", "forecast", "finalize")
_GO_STAGES = ("statics", "initialize", "forecast", "render", "finalize")

#: Options both routes take: the memory levers and the run's own switches.
_COMMON_OPTIONS = frozenset({
    "restart", "until_s", "latitude_bands", "host_spill", "overwrite",
})

ROUTES: dict[str, Route] = {
    "experiment": Route(
        name="experiment",
        summary="the config-driven forecast route: one experiment TOML "
                "integrated in this process (what `woof global run CONFIG` "
                "executes)",
        door="woof global run",
        run_options=_COMMON_OPTIONS,
        stages=_EXPERIMENT_STAGES),
    "go": Route(
        name="go",
        summary="statics, forecast and pictures in one command (what "
                "`woof global go CONFIG` executes)",
        door="woof global go",
        run_options=_COMMON_OPTIONS | frozenset({
            "render_products", "start_date", "geog_root", "statics", "render",
        }),
        stages=_GO_STAGES),
}


#: Routes the ENGINE has that this distribution does not, and the concrete
#: breakage naming each one prevents.  A refusal, never a silent remap: a
#: client that copies the kit's ``prepared-plan.json`` and points its config
#: at a global experiment has to be told that the route it named is not this
#: model's, rather than have a different route run under that name.
REFUSED_ROUTES = {
    "prepared": "the prepared route names a prepared-cache root and a WPS "
                "namelist authority, and binds its inputs through them.  This "
                "package reads neither: a global run is initialized from the "
                "analysis file its config's [initial] table names, so there is "
                "no authority, fetch, manifest or preparation stage for a "
                "prepared plan to drive.  Use route 'go' for the staged door "
                "(statics, forecast, pictures) or 'experiment' for the "
                "forecast alone.",
}


_RUN_OPTION_DEFAULTS: dict[str, Any] = {
    "restart": None,
    "until_s": None,
    "latitude_bands": None,
    "host_spill": None,
    "overwrite": False,
    "render_products": None,
    "start_date": None,
    "geog_root": None,
    "statics": True,
    "render": True,
}

_HOST_SPILL_CHOICES = ("auto", "on", "off")

_TOP_LEVEL_KEYS = frozenset({
    "schema", "name", "route", "config", "fetch", "output_root",
    "run_options"})
_REQUIRED_KEYS = ("schema", "name", "route", "config")
_CONFIG_KEYS = frozenset({"path", "inline", "intent"})

#: ``woof run``'s own default outdir, and this package's is its own.
DEFAULT_OUTPUT_ROOT = Path("out") / "arwen-global"


def route_summaries() -> dict[str, str]:
    """Every route this build has, plus every one it refuses, with reasons."""

    summaries = {name: route.summary for name, route in sorted(ROUTES.items())}
    summaries.update(
        {name: "REFUSED: " + reason
         for name, reason in sorted(REFUSED_ROUTES.items())})
    return summaries


# ---------------------------------------------------------------------------
# The plan document
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunPlan:
    """One validated plan, with every path made absolute."""

    name: str
    route: str
    config_path: Path | None
    config_inline: str | None
    config_base_dir: Path
    output_root: Path
    run_options: Mapping[str, Any]
    sha256: str
    source: str
    automatic_resolutions: tuple[Mapping[str, Any], ...]

    @property
    def run_dir(self) -> Path:
        """The one directory this run writes into.

        ``output_root`` IS the run directory, on the engine's own reasoning:
        deriving a subdirectory from the plan's name would leave a caller
        unable to predict where its outputs land.
        """

        return self.output_root

    @property
    def config_kind(self) -> str:
        return "path" if self.config_path is not None else "inline"

    def config_bytes(self) -> bytes:
        if self.config_inline is not None:
            return self.config_inline.encode("utf-8")
        try:
            return Path(self.config_path).read_bytes()
        except OSError as error:
            raise PlanError(
                "run plan {} names config.path {}, which woof global "
                "run-plan cannot read: {}.  Nothing was started.  Point "
                "'config.path' at the TOML on disk (a relative path resolves "
                "against the plan's own directory, not the working "
                "directory), name a shipped experiment, or carry the config "
                "text itself in 'config.inline'.".format(
                    self.source, self.config_path,
                    error.strerror or error)) from error


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{label} must be a non-empty string")
    result = value.strip()
    if any(character in result for character in "\r\n"):
        raise PlanError(f"{label} must fit on one line")
    return result


def _reject_unknown(mapping: Mapping[str, Any], known, label: str) -> None:
    unknown = sorted(set(mapping) - set(known))
    if unknown:
        raise PlanError(
            f"{label} does not have the key(s) {unknown}; this build knows "
            f"{sorted(known)}.  An unknown key is refused rather than "
            "ignored: a plan whose option was silently dropped runs as "
            "something other than what it says.")


def _absolute(value: object, base: Path, label: str) -> Path:
    text = _nonempty_string(value, label)
    path = Path(text)
    return path if path.is_absolute() else (base / path).resolve()


def _config_reference(value: object, base: Path, label: str) -> Path:
    """A plan's ``config.path``: a file, or the name of a shipped experiment.

    The shipped-name spelling is here because it is the one a client can
    actually offer in a picker: ``woof global configs`` lists 55 experiments
    that travel inside the wheel, and a client that could only name a file
    would have to unpack one first.  A path that exists always wins, on the
    package's own rule, so a reader who edits a copy runs their copy.
    """

    from .configs_dir import resolve_config

    text = _nonempty_string(value, label)
    candidate = Path(text)
    resolved = candidate if candidate.is_absolute() else (base / candidate)
    if resolved.exists():
        return resolved.resolve()
    try:
        return resolve_config(text).resolve()
    except FileNotFoundError as error:
        raise PlanError(f"{label}: {error}") from error


def _run_option(key: str, value: object, base: Path) -> Any:
    label = f"run plan 'run_options.{key}'"
    if key in ("overwrite", "statics", "render"):
        if not isinstance(value, bool):
            raise PlanError(f"{label} must be true or false")
        return value
    if key == "latitude_bands":
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise PlanError(f"{label} must be a positive whole number of "
                            "latitude bands, or null for the sizer's choice")
        if value < 1:
            raise PlanError(f"{label} must be at least 1 (1 is the resident "
                            "run); null lets the sizer choose")
        return value
    if key == "host_spill":
        if value is None:
            return None
        text = _nonempty_string(value, label)
        if text not in _HOST_SPILL_CHOICES:
            raise PlanError(f"{label} must be one of "
                            f"{list(_HOST_SPILL_CHOICES)}, or null for the "
                            "config's own [memory] table")
        return text
    if key == "until_s":
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PlanError(f"{label} must be a model time in seconds")
        if value <= 0:
            raise PlanError(f"{label} must be positive")
        return float(value)
    if key in ("render_products", "start_date"):
        return None if value is None else _nonempty_string(value, label)
    if key in ("restart", "geog_root"):
        return None if value is None else str(_absolute(value, base, label))
    raise PlanError(f"{label} is not a run option this build understands")


def build_plan(raw: Mapping[str, Any], *, source: str,
               base_dir: str | Path, sha256: str) -> RunPlan:
    """Validate a parsed plan document and build the :class:`RunPlan`.

    Separate from :func:`load_plan` for the reason the engine's pair are
    separate: the validation is the part a test wants to reach without a file
    on disk.
    """

    if not isinstance(raw, dict):
        raise PlanError("run plan must be a JSON object")
    _reject_unknown(raw, _TOP_LEVEL_KEYS, "run plan")
    missing = [key for key in _REQUIRED_KEYS if key not in raw]
    if missing:
        raise PlanError(
            f"run plan {source} is missing required key(s) {missing}")
    if raw["schema"] != PLAN_SCHEMA:
        raise PlanError(
            f"{source} is not a {PLAN_SCHEMA} document: schema is "
            f"{raw['schema']!r}.  The schema id carries this document's "
            "version, and a reader that accepted an unrecognized id would be "
            "guessing which fields it holds.")

    base = Path(base_dir)
    resolutions: list[dict[str, Any]] = []

    name = _nonempty_string(raw["name"], "run plan 'name'")
    route = _nonempty_string(raw["route"], "run plan 'route'")
    if route in REFUSED_ROUTES:
        raise PlanError(
            f"run plan {source} names route {route!r}, which this "
            f"distribution refuses.  {REFUSED_ROUTES[route]}")
    if route not in ROUTES:
        raise PlanError(
            f"run plan {source} names route {route!r}, which this build does "
            "not have.  Known routes: " + "; ".join(
                f"{key} ({value.summary})"
                for key, value in sorted(ROUTES.items())) + ".")

    if "fetch" in raw:
        raise PlanError(
            f"run plan {source} carries a 'fetch' block, and this "
            "distribution has no fetch stage to run it in.  A global run is "
            "initialized from the analysis file its config's [initial] table "
            "names; the bytes are brought by `woof global obs anchors`, "
            "which a plan does not drive.  Remove the block rather than "
            "leaving it: an option this door accepted and ignored would be "
            "a fetch a caller believes happened.")

    config = raw["config"]
    if not isinstance(config, dict):
        raise PlanError("run plan 'config' must be an object with exactly "
                        "one of 'path' or 'inline'")
    _reject_unknown(config, _CONFIG_KEYS, "run plan 'config'")
    if "intent" in config:
        raise PlanError(
            f"run plan {source} carries 'config.intent', which the engine's "
            "domain wizard writes into a regional configuration.  This "
            "distribution ships no wizard and a global run has no domain to "
            "fit, so there is nothing here that could turn an intent into a "
            "config.  Name an experiment TOML in 'config.path' (a shipped "
            "experiment's name is accepted) or carry its text in "
            "'config.inline'; `woof global configs` lists what shipped.")
    spelled = sorted(set(config) & {"path", "inline"})
    if len(spelled) != 1:
        raise PlanError(
            "run plan 'config' must carry exactly ONE of 'path' (a TOML on "
            "disk, or a shipped experiment's name) or 'inline' (the TOML "
            f"text itself), got {spelled}.")
    config_path = None
    config_inline = None
    if spelled == ["path"]:
        config_path = _config_reference(config["path"], base,
                                        "run plan 'config.path'")
        if config_path != Path(str(config["path"])).resolve():
            resolutions.append({
                "scope": "config", "key": "path",
                "value": str(config_path), "basis": "shipped_experiment",
                "note": "the plan named an experiment this wheel ships; the "
                        "file inside the installed package answered"})
    else:
        if not isinstance(config["inline"], str) or not config["inline"]:
            raise PlanError("run plan 'config.inline' must be non-empty TOML "
                            "text")
        config_inline = config["inline"]

    if "output_root" in raw:
        output_root = _absolute(raw["output_root"], base,
                                "run plan 'output_root'")
    else:
        output_root = (base / DEFAULT_OUTPUT_ROOT).resolve()
        resolutions.append({
            "scope": "plan", "key": "output_root",
            "value": str(output_root), "basis": "front_door_default",
            "note": "`woof global run`'s own default --outdir, resolved "
                    "against the plan's directory"})

    options = raw.get("run_options", {})
    if not isinstance(options, dict):
        raise PlanError("run plan 'run_options' must be an object")
    known = ROUTES[route].run_options
    unknown = sorted(set(options) - set(known))
    if unknown:
        elsewhere = sorted(
            key for key in unknown
            if any(key in other.run_options for other in ROUTES.values()))
        detail = ""
        if elsewhere:
            detail = ("  " + ", ".join(elsewhere) + " belong(s) to route(s) "
                      + ", ".join(sorted(
                          other.name for other in ROUTES.values()
                          if set(elsewhere) & set(other.run_options)))
                      + ".")
        raise PlanError(
            f"run plan 'run_options' names {unknown}, which route "
            f"{route!r} does not take; it takes {sorted(known)}." + detail)
    resolved_options: dict[str, Any] = {}
    for key in sorted(known):
        if key in options:
            resolved_options[key] = _run_option(key, options[key], base)
            continue
        resolved_options[key] = _RUN_OPTION_DEFAULTS[key]
        resolutions.append({
            "scope": "run_options", "key": key,
            "value": _RUN_OPTION_DEFAULTS[key], "basis": "schema_default"})

    return RunPlan(
        name=name, route=route, config_path=config_path,
        config_inline=config_inline, config_base_dir=base,
        output_root=output_root, run_options=resolved_options,
        sha256=sha256, source=source,
        automatic_resolutions=tuple(resolutions))


def load_plan(path: str | Path) -> RunPlan:
    """Read one plan document off disk and validate it."""

    plan_path = Path(path)
    try:
        payload = plan_path.read_bytes()
    except OSError as error:
        raise PlanError(f"cannot read run plan {plan_path}: "
                        f"{error.strerror or error}") from error
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as error:
        raise PlanError(f"{plan_path} is not valid JSON: {error}") from error
    return build_plan(raw, source=str(plan_path),
                      base_dir=plan_path.resolve().parent,
                      sha256=hashlib.sha256(payload).hexdigest())


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------


def _config_from_plan(plan: RunPlan, *, into: Path | None = None):
    """The loaded config and the path it was loaded from.

    An inline config is written to a real file first, because
    :func:`woof.globe.config.load_config` reads a path and every downstream
    door (the sidecar, the render door's config copy) needs one too.
    """

    from .config import load_config

    if plan.config_path is not None:
        return load_config(plan.config_path), Path(plan.config_path)
    if into is None:
        raise PlanError(
            "an inline config needs a directory to be written into before it "
            "can be loaded; resolve_plan's caller supplies one (a run writes "
            "it into the run directory, where it is the run's provenance, and "
            "a query mode into a scratch directory it then removes)")
    destination = Path(into)
    destination.mkdir(parents=True, exist_ok=True)
    written = destination / "plan-config.toml"
    written.write_bytes(plan.config_bytes())
    return load_config(written), written


def _grid_of(cfg):
    from woof.globe.spectral.grid import GaussianGrid

    return GaussianGrid.create(cfg.truncation, nlat=cfg.nlat, nlon=cfg.nlon,
                               dealias_factor=cfg.dealias_factor)


def _restart_step(restart: str | Path | None) -> tuple[int | None, str | None]:
    """The model step a restart checkpoint carries, or why it could not be read.

    The step is read from the checkpoint's own metadata record, which is what
    :mod:`woof.globe.runner` resumes at, rather than parsed out of the file
    name: a checkpoint a client copied under another name would otherwise be
    counted from a step it does not carry.  Only the metadata member is read
    (:func:`woof.globe.checkpoint.read_checkpoint_header`), so this costs
    nothing on a large state.
    """

    if restart is None:
        return None, None
    from .checkpoint import read_checkpoint_header

    try:
        return int(read_checkpoint_header(restart)["step"]), None
    except Exception as error:  # noqa: BLE001 - a query mode reports
        return None, f"{type(error).__name__}: {error}"


def _frame_schedule(cfg, until_s: float | None, *,
                    restart: str | Path | None = None) -> dict[str, Any]:
    """Exactly the checkpoints this run will write, by the runner's own rule.

    ``woof.globe.runner`` writes the cold state ONLY on a cold start, then
    one checkpoint at every ``output_interval_s / dt_s`` step after the step
    it starts from, and one at the final step whether or not it lands on that
    cadence.  A restart starts at the step its checkpoint carries, so that
    step is READ from the checkpoint rather than assumed to be zero.

    THE BREAKAGE THIS PREVENTS, measured on the Windows desktop 2026-09-10 on
    the T21 case at dt 360 s with output every 60 steps: the count was taken
    from step 0 on every plan and the cold state was always added, so a
    restart plan over-reported in three documents at once -- the resolved
    document's ``output_schedule``, the estimate's ``disk`` block, and the
    forecast ``stage_started`` event's ``expected_checkpoints``, which is what
    sizes a client's progress bar.  A restart from step 60 and one from step
    180 both reported 5 checkpoints and wrote 3 and 1.

    A restart whose checkpoint cannot be read reports ``null`` counts and says
    why: the count is exact or it is not reported, never taken from step 0.
    """

    seconds = float(cfg.duration_s if until_s is None else until_s)
    total_steps = int(round(seconds / cfg.dt_s))
    output_every = int(round(cfg.output_interval_s / cfg.dt_s))
    begins_at, unreadable = _restart_step(restart)
    schedule = {
        "model_seconds": seconds,
        "dt_s": float(cfg.dt_s),
        "total_steps": total_steps,
        "output_interval_s": float(cfg.output_interval_s),
        "output_every_steps": output_every,
        # The step this run starts at, and the file that says so.  Null on a
        # cold start, which begins at step 0 and writes that state.
        "restart": None if restart is None else str(restart),
        "restart_step": begins_at,
    }
    if restart is not None and begins_at is None:
        schedule.update({
            "checkpoints": None,
            "checkpoints_from_restart": None,
            "exact": False,
            "basis": "this plan restarts from " + str(restart) + " and the "
                     "step that checkpoint carries could not be read ("
                     + str(unreadable) + "), so no count is reported rather "
                     "than one taken from step 0",
        })
        return schedule
    start = 0 if begins_at is None else begins_at
    due = set()
    if output_every > 0:
        due = {step for step in range(output_every, total_steps + 1,
                                      output_every)
               if step > start}
    if total_steps > start:
        due.add(total_steps)
    # The cold state is written when the run does not restart from a
    # checkpoint; a restart continues an existing lineage and writes none.
    schedule.update({
        "checkpoints": len(due) + (1 if restart is None else 0),
        # The cadence-and-final checkpoints alone, without the initial state.
        "checkpoints_from_restart": len(due),
        "exact": True,
        "basis": "woof.globe.runner writes the cold state on a cold start, "
                 "one checkpoint at every output_interval_s/dt_s step after "
                 "the step it starts from, and one at the final step; the "
                 "count is arithmetic and exact"
                 + ("" if restart is None else
                    ", and this plan starts at step {}, read from {}".format(
                        start, restart)),
    })
    return schedule


def _declared_inputs(cfg, config_path: Path, *,
                     restart: str | Path | None = None
                     ) -> list[dict[str, Any]]:
    """Every file this run reads, and whether it is there.

    The config's own inputs, and the restart checkpoint the plan names: a
    restart path that is not on disk is the same class of hole as a missing
    analysis, and it is refused before anything is started for the same
    reason.
    """

    from .statics import cache_paths

    inputs: list[dict[str, Any]] = [{
        "role": "configuration",
        "path": str(config_path),
        "present": Path(config_path).is_file(),
    }]
    if restart:
        inputs.append({
            "role": "restart",
            "path": str(restart),
            "present": Path(restart).is_file(),
            "note": "run_options.restart; the forecast continues from the "
                    "step this checkpoint carries and writes no cold state",
        })
    if cfg.initial_mode == "analysis" and cfg.analysis_grib:
        inputs.append({
            "role": "analysis",
            "path": str(cfg.analysis_grib),
            "present": Path(cfg.analysis_grib).is_file(),
        })
    if cfg.analysis_fill_grib:
        inputs.append({
            "role": "analysis_fill",
            "path": str(cfg.analysis_fill_grib),
            "present": Path(cfg.analysis_fill_grib).is_file(),
        })
    if cfg.statics.source == "real":
        try:
            cache, _sidecar = cache_paths(cfg.statics, _grid_of(cfg))
        except Exception as error:  # noqa: BLE001 - a query mode reports
            inputs.append({"role": "statics_cache", "path": None,
                           "present": False,
                           "note": f"{type(error).__name__}: {error}"})
        else:
            inputs.append({
                "role": "statics_cache",
                "path": str(cache),
                "present": Path(cache).is_file(),
                "note": "built by `woof global statics`; the `go` route "
                        "builds it when it is absent",
            })
    return inputs


def _mapping_resolution(cfg) -> dict[str, Any] | None:
    """Which authority mapping answered this config's ``analysis_mapping``."""

    if not cfg.analysis_mapping:
        return None
    from .sources import mapping_facts

    return mapping_facts(cfg.analysis_mapping)


def _sizer_resolution(cfg) -> dict[str, Any]:
    """The band count and the host tier the sizer would choose for this run.

    Read from :func:`woof.globe.sizing.run_memory_gate`, which is the same
    call ``woof global run`` makes at its own door, so the plan a client
    reviews is the plan the run takes.  The gate reads the card in a
    short-lived subprocess and never in this process.
    """

    from .sizing import run_memory_gate

    gate = run_memory_gate(cfg)
    plan = gate["plan"]
    fragmentation, fragmentation_basis = plan.fragmentation
    # A DEVICE FIGURE FOR A RUN WITH NO DEVICE IS NOT A SMALL FIGURE.  On the
    # numpy backend the sizer's device terms are all zero, which reads in a
    # client as "this run needs no VRAM" rather than "there is no card in this
    # run at all", and the pool fragmentation ratio is a property of an
    # allocator that never runs.  Null, with the basis saying why: a consumer
    # must be able to tell absent from zero.
    on_device = gate["estimate"].backend == "cupy"

    def device_only(value):
        return value if on_device else None

    return {
        "verdict": gate["verdict"],
        "device_basis": (
            "the card figures below are the sizer's, for the cupy backend"
            if on_device else
            "this config runs on the numpy backend, so there is no card in "
            "it: every device figure below is null, which means ABSENT and "
            "never zero"),
        "refuse": bool(gate["refuse"]),
        "unfitted": gate.get("unfitted"),
        "free_bytes": gate.get("free_bytes"),
        "latitude_bands": device_only(plan.bands),
        "latitude_bands_chosen_by": plan.bands_chosen_by,
        "host_spill_slices": list(plan.spill_slices),
        "host_spill_chosen_by": plan.spill_chosen_by,
        "host_spill_bytes": device_only(plan.spilled_bytes),
        "live_peak_bytes": device_only(plan.live_peak_bytes),
        "card_required_bytes": device_only(plan.card_bytes),
        "pool_fragmentation": device_only(fragmentation),
        "pool_fragmentation_basis": device_only(fragmentation_basis),
        "fits": bool(plan.fits),
        "reason": plan.reason,
        "device_peak_bytes": device_only(gate["estimate"].device_peak_bytes),
        "host_peak_bytes": gate["estimate"].host_peak_bytes,
        "backend": gate["estimate"].backend,
    }


def _physics_snapshot(cfg) -> dict[str, Any]:
    from .physics.builtin_adapters import ensure_builtin_global_physics_adapters
    from .physics.registry import global_physics_manifest

    if cfg.physics_mode != "arwen-native":
        return {
            "mode": cfg.physics_mode,
            "profile_id": "reference-suite-v1",
            "adapter": None,
            "options": {},
            "reference": dataclasses.asdict(cfg.reference_physics)
            if dataclasses.is_dataclass(cfg.reference_physics)
            else dict(cfg.reference_physics.__dict__),
        }
    ensure_builtin_global_physics_adapters()
    manifest = global_physics_manifest()
    entry = manifest.get(cfg.native_adapter_name, {})
    contract = entry.get("contract", {}) if isinstance(entry, dict) else {}
    return {
        "mode": cfg.physics_mode,
        "profile_id": cfg.native_adapter_name,
        "adapter": cfg.native_adapter_name,
        "options": dict(cfg.native_adapter_options),
        "scheme_identity": contract.get("scheme_identity"),
        "admission_status": contract.get("admission_status"),
    }


def _config_snapshot(cfg, config_path: Path) -> dict[str, Any]:
    grid = _grid_of(cfg)
    return {
        "name": cfg.name,
        "config_path": str(config_path),
        "config_hash": cfg.config_hash,
        "backend": cfg.backend,
        "precision": cfg.precision,
        "grid": {
            "truncation": cfg.truncation,
            "nlat": grid.nlat,
            "nlon": grid.nlon,
            "dealias_factor": cfg.dealias_factor,
            "nlev": len(cfg.b_half) - 1,
            "vertical_coordinate": cfg.vertical_coordinate,
            # INDEX 0, the TOP of the half-level ladder, which is what the
            # name says and what `vertical.describe()` reports.  Index -1 is
            # the SURFACE end, where a = p_top * (1 - b) and b = 1, so it is
            # identically 0.0 on every hybrid ladder: this document reported
            # a model top of 0 Pa for every experiment before the index was
            # the one the model's own summary reads.
            "p_top_pa": float(cfg.a_half_pa[0]) if cfg.a_half_pa else None,
        },
        "time": {
            "dt_s": cfg.dt_s,
            "duration_s": cfg.duration_s,
            "output_interval_s": cfg.output_interval_s,
            "integrator": cfg.integrator,
            "maximum_cfl": cfg.maximum_cfl,
        },
        "initial": {
            "mode": cfg.initial_mode,
            "analysis_grib": cfg.analysis_grib,
            "analysis_mapping": cfg.analysis_mapping,
            "analysis_fill_grib": cfg.analysis_fill_grib,
            "analysis_fill_mapping": cfg.analysis_fill_mapping,
        },
        "statics": {
            "source": cfg.statics.source,
            "tokens": list(cfg.statics.tokens()),
        },
        "physics": _physics_snapshot(cfg),
    }


def _render_skip_reason(options: Mapping[str, Any]) -> str | None:
    """Why the render stage will skip itself on this plan, or ``None``.

    ONE spelling of each condition, read by :func:`_render_stage` when it
    skips and by the resolved document when it says the stage will not run:
    a reviewer's reason and the run's reason cannot be two sentences.  The
    stage's third condition -- a forecast that wrote no checkpoints -- is not
    knowable from a plan and is not answered here.
    """

    if not options.get("render", True):
        return "run_options.render is false"
    if not options.get("start_date"):
        return ("run_options.start_date was not given, and a tape with no "
                "valid time is a tape nobody can place in time")
    return None


def _render_products_check(products: str) -> dict[str, Any]:
    """Every token of a ``render_products`` spec, against the live catalog.

    THE BREAKAGE THIS PREVENTS, measured on the Windows desktop 2026-09-10: a
    ``go`` plan naming ``not_a_product`` resolved at exit 0 with an empty
    ``warnings`` list, the run then spent the statics and the whole forecast,
    and the render stage died with "exit 1".  On the T21 case that was 28 s;
    on ``arwen_global_gdas_t255_native_sl_si_24h`` it is a forecast day.

    The slug list, the group keywords and the skip token are the RENDERER's
    own, asked through :func:`render_catalog` rather than transcribed here.
    A token carrying a colon is a PARAMETERIZED form the renderer resolves
    against the tape's own stored variables (``var:<name>``, ``mesh:<...>``),
    which no catalog can answer without a tape: it is reported as unchecked
    and never as unknown, because refusing it would refuse a request the
    renderer accepts.  A machine with no staged ``rw_wrfbatch`` cannot be
    asked at all and says so with the catalog's own refusal, rather than
    passing every token silently.
    """

    tokens = [token.strip() for token in str(products).split(",")
              if token.strip()]
    try:
        catalog = render_catalog()
    except Exception as error:  # noqa: BLE001 - a query mode reports
        catalog = {"error": f"{type(error).__name__}: {error}"}
    rows = catalog.get("products")
    if not isinstance(rows, list):
        return {
            "products_checked_against": None,
            "products_check_error": str(
                catalog.get("error")
                or "the renderer's own catalog is not available on this "
                   "machine, so no slug in this plan was checked"),
            "unknown_products": [],
            "unchecked_products": tokens,
        }
    known = {row.get("name") for row in rows}
    keywords = set(catalog.get("group_keywords") or ())
    skip_token = catalog.get("skip_token")
    accepted = set(keywords)
    if skip_token:
        accepted.add(str(skip_token))
    unknown: list[str] = []
    unchecked: list[str] = []
    for token in tokens:
        if token in known or token in accepted:
            continue
        if ":" in token:
            unchecked.append(token)
            continue
        unknown.append(token)
    return {
        "products_checked_against":
            "the renderer's own --list-products, through `woof global "
            "run-plan --catalog`: {} slug(s), group keyword(s) {}, skip "
            "token {!r}".format(
                len(known), ", ".join(sorted(keywords)) or "none", skip_token),
        "products_check_error": None,
        "unknown_products": unknown,
        "unchecked_products": unchecked,
    }


def _render_products_resolution(plan: RunPlan) -> dict[str, Any] | None:
    """What the render stage will draw, whether it will run, and with what.

    The stage list a reviewer reads is the route's, and the route's list
    carries ``render`` whether or not this plan's options let the stage do
    anything.  ``_render_stage`` skips itself on two conditions a plan
    document already carries, so both are answered HERE rather than at the
    end of a run: MEASURED on the Windows desktop 2026-09-10, a ``go`` plan
    with no ``start_date`` resolved with the render stage in its list and no
    warning, then completed at exit 0 with the stage skipped and no pictures
    at all.  A reviewer who approved that plan expecting imagery got a green
    run and an empty gallery.
    """

    if "render_products" not in ROUTES[plan.route].run_options:
        return None
    from .render_door import DEFAULT_PRODUCTS

    options = plan.run_options
    chosen = options.get("render_products")
    products = chosen or DEFAULT_PRODUCTS
    skipped = _render_skip_reason(options)
    return {
        "products": products,
        "basis": "run_options.render_products" if chosen
                 else "the render door's own default",
        "catalog": "`woof global run-plan --catalog` says which of these "
                   "a global tape can draw and why the rest cannot",
        # The stage's own two skip conditions, read off this plan.  The third
        # -- a forecast that wrote no checkpoints -- is not knowable before
        # the run and is named as such.
        "will_run": skipped is None,
        "skipped_reason": skipped,
        "run_time_condition": "the render stage also skips itself when the "
                              "forecast wrote no checkpoints, which only the "
                              "run can know",
        **_render_products_check(products),
    }


def resolve_plan(plan: RunPlan, *, generate_into: Path | None = None,
                 require_inputs: bool = True) -> tuple[dict[str, Any], Any, Path]:
    """Load this plan's config through the real seam and describe it.

    Everything ``--resolve`` prints and everything the ``resolved_plan`` event
    carries is built here -- one function, so the document a caller inspects
    before a run and the event it receives during one cannot drift.
    """

    payload = plan.config_bytes()
    cfg, config_path = _config_from_plan(plan, into=generate_into)
    route = ROUTES[plan.route]
    resolutions = list(plan.automatic_resolutions)

    grid = _grid_of(cfg)
    if cfg.nlat is None or cfg.nlon is None:
        resolutions.append({
            "scope": "grid", "key": "quadrature",
            "value": {"nlat": grid.nlat, "nlon": grid.nlon},
            "basis": "dealias_factor",
            "note": "the config left the Gaussian quadrature unset, so it "
                    f"follows T{cfg.truncation} at dealias "
                    f"{cfg.dealias_factor:g}"})

    mapping = _mapping_resolution(cfg)
    if mapping is not None:
        resolutions.append({
            "scope": "initial", "key": "analysis_mapping",
            "value": mapping["resolved"], "basis": "engine_authority_table",
            "note": "the bare id resolved against the installed engine's "
                    "authority table" if mapping["present"] else
                    "the installed engine's authority table carries no row "
                    "for this id, so the run refuses at its first door: "
                    + str(mapping["refusal"])})

    sizer = _sizer_resolution(cfg)
    resolutions.append({
        "scope": "memory", "key": "latitude_bands",
        "value": sizer["latitude_bands"],
        "basis": sizer["latitude_bands_chosen_by"],
        "note": sizer["reason"]})
    resolutions.append({
        "scope": "memory", "key": "host_spill_slices",
        "value": sizer["host_spill_slices"],
        "basis": sizer["host_spill_chosen_by"],
        "note": sizer["verdict"]})

    products = _render_products_resolution(plan)
    if products is not None:
        resolutions.append({
            "scope": "render", "key": "render_products",
            "value": products["products"], "basis": products["basis"],
            "note": products["catalog"]})

    inputs = _declared_inputs(cfg, config_path,
                              restart=plan.run_options.get("restart"))
    warnings: list[dict[str, str]] = []
    missing = [entry for entry in inputs if not entry["present"]]
    if missing and require_inputs:
        raise PlanError(
            "run plan " + plan.source + " declares input(s) that are not "
            "there: " + ", ".join(
                f"{entry['role']} {entry['path']}" for entry in missing)
            + ".  Nothing was started.")
    for entry in missing:
        warnings.append({
            "scope": "declared_inputs",
            "message": f"{entry['role']} {entry['path']} is not on disk"})
    if sizer["refuse"]:
        warnings.append({"scope": "memory", "message": sizer["reason"]})
    # THE RENDER STAGE, ANSWERED BEFORE THE FORECAST IS SPENT.  A slug the
    # renderer does not carry is a run that draws nothing after paying for
    # every stage before the render, so it is a refusal on the route that
    # starts work and a warning in a query mode, which is exactly how a
    # missing declared input is handled above.
    if products is not None:
        if products["unknown_products"]:
            message = (
                "run_options.render_products names "
                + ", ".join(repr(slug) for slug in products["unknown_products"])
                + ", which the installed renderer does not carry.  "
                  "Checked against "
                + str(products["products_checked_against"])
                + ".  `woof global run-plan --catalog` lists every slug and "
                  "which of them a global tape can draw")
            if require_inputs:
                raise PlanError(message + ".  Nothing was started.")
            warnings.append({"scope": "render", "message": message})
        if products["products_check_error"]:
            warnings.append({
                "scope": "render",
                "message": "the render products in this plan were not "
                           "checked against the renderer's catalog, so a "
                           "slug this plan cannot draw will be found only "
                           "after the forecast has run: "
                           + products["products_check_error"]})
        if products["unchecked_products"]:
            warnings.append({
                "scope": "render",
                "message": ", ".join(
                    repr(slug) for slug in products["unchecked_products"])
                + " name(s) the renderer resolves against the tape's own "
                  "stored variables, so no catalog can say here whether the "
                  "tape carries them"})
        if not products["will_run"]:
            warnings.append({
                "scope": "render",
                "message": "the render stage will skip itself and this run "
                           "will write no pictures: "
                           + str(products["skipped_reason"])})

    document = {
        "schema": RESOLVE_SCHEMA,
        "producer": producer_block(),
        "plan": {
            "name": plan.name,
            "route": plan.route,
            "route_summary": route.summary,
            "door": route.door,
            "stages": list(route.stages),
            "source": plan.source,
            "sha256": plan.sha256,
            "config_kind": plan.config_kind,
            "config_source": str(config_path),
            "config_sha256": hashlib.sha256(payload).hexdigest(),
            "run_dir": str(plan.run_dir),
            "run_options": dict(plan.run_options),
        },
        "configuration": _config_snapshot(cfg, config_path),
        "analysis_mapping": mapping,
        "sizer": sizer,
        "render": products,
        "declared_inputs": inputs,
        "inputs_present": all(entry["present"] for entry in inputs),
        "output_schedule": _frame_schedule(
            cfg, plan.run_options.get("until_s"),
            restart=plan.run_options.get("restart")),
        "automatic_resolutions": resolutions,
        "warnings": warnings,
    }
    return document, cfg, config_path


# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------


def estimate_plan(plan: RunPlan, *,
                  generate_into: Path | None = None) -> dict[str, Any]:
    """What this plan will cost, from measured machinery only.

    The VRAM figure is :mod:`woof.globe.sizing`'s calibrated device-peak
    model plus the pool fragmentation and out-of-pool terms a card must also
    hold -- the same arithmetic ``woof global run`` prints at its own door.
    Frame COUNTS are exact.  Wall time is ``null``: this package publishes no
    measured rate for an arbitrary configuration, and a front end showing an
    invented duration would be showing this model's name on a number nobody
    measured.
    """

    from .sizing import (calibration_shapes, worst_calibration_residual_fraction,
                         worst_held_out_residual_fraction)

    resolution, cfg, _config_path = resolve_plan(
        plan, generate_into=generate_into, require_inputs=False)
    sizer = resolution["sizer"]
    schedule = resolution["output_schedule"]
    return {
        "schema": ESTIMATE_SCHEMA,
        "producer": producer_block(),
        "plan": resolution["plan"],
        "vram": {
            "backend": sizer["backend"],
            "device_basis": sizer["device_basis"],
            "device_peak_bytes": sizer["device_peak_bytes"],
            "banded_live_peak_bytes": sizer["live_peak_bytes"],
            "card_required_bytes": sizer["card_required_bytes"],
            "pool_fragmentation": sizer["pool_fragmentation"],
            "pool_fragmentation_basis": sizer["pool_fragmentation_basis"],
            "host_peak_bytes": sizer["host_peak_bytes"],
            "free_bytes": sizer["free_bytes"],
            "fits": sizer["fits"],
            "latitude_bands": sizer["latitude_bands"],
            "host_spill_slices": sizer["host_spill_slices"],
            "verdict": sizer["verdict"],
            "basis": "woof.globe.sizing's calibrated device-peak model, "
                     "fitted over {} measured shapes; worst calibration "
                     "residual {:.1%} in sample and {} held out.  The card "
                     "is read in a short-lived subprocess, never in this "
                     "process.".format(
                         calibration_shapes(),
                         worst_calibration_residual_fraction(),
                         "unavailable" if worst_held_out_residual_fraction()
                         is None else
                         "{:.1%}".format(worst_held_out_residual_fraction())),
        },
        "disk": {
            "checkpoints": schedule["checkpoints"],
            "total_steps": schedule["total_steps"],
            "output_every_steps": schedule["output_every_steps"],
            "model_seconds": schedule["model_seconds"],
            "bytes": None,
            "basis": schedule["basis"] + "; bytes per checkpoint are not "
                     "measured by this package, so no byte figure is "
                     "reported rather than an invented one",
        },
        "download": {
            "bytes": None,
            "basis": "this distribution has no fetch stage in a plan; the "
                     "analysis file is named by the config and brought by "
                     "`woof global obs anchors`, which reports its own size",
        },
        "wall_time": {
            "seconds": None,
            "basis": "this package publishes no exact rate for an arbitrary "
                     "configuration; the model_progress events carry the "
                     "real one from the first output step",
        },
        "automatic_resolutions": resolution["automatic_resolutions"],
        "warnings": resolution["warnings"],
    }


# ---------------------------------------------------------------------------
# Catalog, physics, sources, probe
# ---------------------------------------------------------------------------


def _measured_render_table() -> dict[str, Any] | None:
    """The shipped measurement of what a global tape can draw."""

    path = Path(__file__).resolve().parent / "data" / "render-catalog-global.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def render_catalog() -> dict[str, Any]:
    """What may be put in ``run_options.render_products``, as JSON.

    Two facts, kept apart because they are measured differently.

    THE SLUG LIST is the renderer's own, asked live rather than transcribed,
    and the INSTRUMENT is the engine's: :func:`woof.runplan.render_catalog`
    resolves the renderer through ``woof.render._resolve_engine("auto")``
    and reads its ``--list-products``.  This function adds the global-tape
    half below and never re-asks the question.  ``woof.globe.render_door``
    is this package's DRAW door and its own resolver is not what answers
    here; a box with no staged ``rw_wrfbatch`` gets ``products: null`` and
    the staging remedy in ``error`` from the engine's refusal, which is a
    document a picker can act on.

    THE AVAILABILITY of each slug ON A GLOBAL TAPE is a property of the field
    set :mod:`woof.globe.wrfout_export` writes, so it cannot be answered
    without a tape and this door takes no plan.  It is therefore MEASURED
    once, against a real exported tape, by
    ``tools/arwen_global_render_catalog.py``, and shipped as package data with
    the host, the date, the renderer digest and the renderer's own reason for
    every excluded slug.  A slug the live renderer lists that the table does
    not carry reports ``unmeasured`` -- never ``renderable``, because a
    catalog that says yes without having asked is worse than one that says it
    does not know.
    """

    from woof.runplan import render_catalog as engine_render_catalog

    from .render_door import DEFAULT_PRODUCTS

    document = dict(engine_render_catalog())
    document["producer"] = producer_block()
    document["default_products"] = DEFAULT_PRODUCTS
    document["door"] = "woof global render"

    table = _measured_render_table()
    if table is None:
        document["global_tape"] = None
        document["global_tape_basis"] = (
            "the shipped measurement (woof/globe/data/"
            "render-catalog-global.json) is not readable in this install, so "
            "no slug's availability on a global tape is claimed")
        return document

    # Computed ONCE and before the counts, because the counts carry it too:
    # the availability numbers are a stored measurement and the verdict on
    # whether it applies to this machine belongs beside them, not only in a
    # block further down that a picker reading the headline never opens.
    digest_verdict = (_renderer_digest_matches(table)
                      if document.get("engine") else None)

    single = {row["name"]: row for row in table["single_frame"]["products"]}
    series = {row["name"]: row
              for row in (table.get("series") or {}).get("products", ())}
    products = document.get("products")
    if isinstance(products, list):
        merged = []
        for entry in products:
            name = entry.get("name")
            row = single.get(name)
            series_row = series.get(name)
            merged.append({
                **entry,
                "global_tape_status": (row or {}).get("status", "unmeasured"),
                "global_tape_kind": (row or {}).get("kind"),
                "global_tape_detail": (row or {}).get("detail"),
                "global_tape_series_status": (
                    (series_row or {}).get("status", "unmeasured")),
                "global_tape_series_detail": (series_row or {}).get("detail"),
            })
        document["products"] = merged
        def _count(key: str, *statuses: str) -> int:
            return sum(1 for row in merged if row[key] in statuses)

        # TWO ARMS, and the headline says which one it is.  The single-frame
        # arm is what ONE stored frame can draw; the series arm is the same
        # measurement over a tape with more than one stored whole-hour frame,
        # which is what a windowed accumulation needs, and it is the more
        # permissive of the two.  A count whose basis is not stated is a
        # number a picker draws without knowing what it counted.
        # WHERE THE NUMBERS WERE MEASURED travels WITH them.  This block is
        # the headline a picker reads without opening `global_tape` below,
        # and the availability half of it is a stored measurement from one
        # host: on any other platform these are that host's numbers, and a
        # count with no host, no date and no digest verdict beside it cannot
        # say so.
        stamp = {key: table.get(key) for key in ("measured_utc", "host",
                                                 "platform")}
        document["global_tape_counts"] = {
            "listed": len(merged),
            **stamp,
            "renderer_digest_matches_installed": digest_verdict,
            "basis": "the SINGLE-FRAME arm: what one stored frame draws.  "
                     "`series` below is the same measurement over a tape "
                     "with more than one stored whole-hour frame, and each "
                     "row carries both under global_tape_status and "
                     "global_tape_series_status.  `listed` is this install's "
                     "live renderer; every other count here is the STORED "
                     "measurement named by host, platform and measured_utc "
                     "above, and is valid on this machine only where "
                     "renderer_digest_matches_installed is true.",
            "renderable": _count("global_tape_status", "renderable"),
            "unavailable": _count("global_tape_status", "excluded",
                                  "missing-fields"),
            "unmeasured": _count("global_tape_status", "unmeasured"),
            "series": {
                "renderable": _count("global_tape_series_status",
                                     "renderable"),
                "unavailable": _count("global_tape_series_status", "excluded",
                                      "missing-fields", "blocked"),
                "unmeasured": _count("global_tape_series_status",
                                     "unmeasured"),
            },
        }
    document["global_tape"] = {
        key: table[key] for key in
        ("measured_utc", "host", "platform", "instrument", "renderer", "tape")
        if key in table}
    document["global_tape"]["single_frame_summary"] = (
        table["single_frame"]["summary"])
    document["global_tape"]["series_summary"] = (
        (table.get("series") or {}).get("summary"))
    document["global_tape"]["status_meanings"] = {
        "renderable": "the renderer's import catalog offers this slug on a "
                      "tape this package exported",
        "missing-fields": "the tape does not store a field the product needs; "
                          "the detail names the field",
        "excluded": "the renderer's catalog excludes it for the reason in the "
                    "detail (a heavy grid not computed at import, a windowed "
                    "accumulation with too few stored frames, or a recipe the "
                    "wrfout import lane does not realize)",
        "unmeasured": "the installed renderer lists this slug and the shipped "
                      "measurement does not carry it; nothing is claimed",
    }
    document["global_tape"]["renderer_digest_matches_installed"] = digest_verdict
    return document


def _renderer_digest_matches(table: Mapping[str, Any]) -> bool | None:
    """Whether the renderer measured is the renderer installed."""

    from woof import rustwx

    renderer = rustwx.find_renderer()
    if renderer is None:
        return None
    expected = (table.get("renderer") or {}).get("sha256")
    if not expected:
        return None
    digest = hashlib.sha256()
    try:
        with Path(renderer).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest() == expected


#: The profile id each non-native physics mode is published under.  ``none``
#: is a MODE, not an absence: it selects the dry dynamical core, which is what
#: the Williamson and Held-Suarez cases run, and folding it into the reference
#: suite would tell a picker those cases carry moist physics.
_MODE_PROFILE_IDS = {
    "reference": "reference-suite-v1",
    "none": "dry-core-v1",
}


def _experiments_by_physics() -> tuple[dict[str, list[str]], list[str]]:
    """Which shipped experiments bind which physics suite, read from the TOMLs.

    The mode's DEFAULT is the config loader's own -- a config with no
    ``[physics]`` table runs the reference suite -- so a shipped experiment
    that spells nothing is counted where it will actually run rather than
    being dropped.  ``tests/test_arwen_global_run_plan.py`` holds the sum of
    the per-profile lists to the number of experiments this wheel ships, so a
    config the grouping cannot place fails a test instead of vanishing from
    the menu.
    """

    import tomllib

    from .configs_dir import config_root

    found: dict[str, list[str]] = {}
    unplaced: list[str] = []
    root = config_root()
    if not root.is_dir():
        return found, unplaced
    for path in sorted(root.glob("*.toml")):
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unplaced.append(path.stem)
            continue
        physics = payload.get("physics")
        physics = physics if isinstance(physics, dict) else {}
        mode = str(physics.get("mode", "reference")).lower()
        if mode == "arwen-native":
            key = physics.get("native_adapter_name")
            if not isinstance(key, str) or not key:
                unplaced.append(path.stem)
                continue
        else:
            key = _MODE_PROFILE_IDS.get(mode)
            if key is None:
                unplaced.append(path.stem)
                continue
        found.setdefault(str(key), []).append(path.stem)
    return found, unplaced


#: The doors a shipped experiment can be integrated through, in the order a
#: menu should offer them.  Each row is (door, module entry, loader), and the
#: LOADER is what decides: a config is offered on a door when that door's own
#: loader accepts it, measured, never inferred from the tables it spells.
_EXPERIMENT_DOORS = (
    ("woof global run", "experiment", "woof.globe.config.load_config"),
    ("python -m woof.globe.spectral run", None,
     "woof.globe.spectral.config.load_config"),
)


def _experiment_doors() -> dict[str, dict[str, Any]]:
    """Which door each shipped experiment actually loads through, measured.

    THE BREAKAGE THIS PREVENTS, measured on the Windows desktop 2026-09-10:
    the menu placed ``global_spectral_held_suarez_smoke``,
    ``global_spectral_primitive_smoke`` and ``global_spectral_williamson2``
    in ``reference-suite-v1`` and asserted every shipped experiment was
    placed, while neither ``woof global run-plan`` nor ``woof global run``
    can execute any of them -- both refuse with "unknown top-level tables:
    global_spectral, ...".  They load only through the spectral core's own
    module entry.  A desktop application building a menu from this document
    offered three dead choices.

    Both loaders are RUN, in order, and the first that accepts the config
    names the door.  A config no door loads carries ``run_plan_route: null``
    and the first door's own refusal as its reason, which is the sentence a
    user would have met several commands later.  All 55 shipped configs are
    measured in 0.04 s on the Windows desktop, so this is a measurement
    rather than a table that can rot.
    """

    import importlib

    from .configs_dir import config_root

    loaders = []
    for door, route, dotted in _EXPERIMENT_DOORS:
        module_name, _, attribute = dotted.rpartition(".")
        try:
            loader = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError) as error:  # noqa: PERF203
            loaders.append((door, route, dotted, None,
                            f"{type(error).__name__}: {error}"))
            continue
        loaders.append((door, route, dotted, loader, None))

    rows: dict[str, dict[str, Any]] = {}
    root = config_root()
    if not root.is_dir():
        return rows
    for path in sorted(root.glob("*.toml")):
        first_refusal = None
        placed = None
        for door, route, dotted, loader, unavailable in loaders:
            if loader is None:
                if first_refusal is None:
                    first_refusal = unavailable
                continue
            try:
                loader(path)
            except Exception as error:  # noqa: BLE001 - the refusal is the answer
                if first_refusal is None:
                    first_refusal = f"{type(error).__name__}: {error}"
                continue
            placed = {
                "door": door, "run_plan_route": route, "loaded_by": dotted,
                # A config that loads on a LATER door still carries the plan
                # route's own refusal: "run this through that door" without
                # the sentence saying why this one will not take it is half
                # an answer.
                "reason": None if route is not None else first_refusal}
            break
        rows[path.stem] = placed or {
            "door": None, "run_plan_route": None, "loaded_by": None,
            "reason": first_refusal or "no shipped door loads this config"}
    return rows


def _engine_physics_verdict() -> dict[str, Any]:
    """What the INSTALLED engine can run of the native suite, measured now.

    Not a claim about a version number.  ``engine_signature_gaps`` inspects
    the installed engine's own signatures and ``engine_gaps`` its own symbol
    table, so this answers for the engine on THIS machine and says plainly
    when the question cannot be answered here at all (the GPU physics modules
    import CuPy at module scope, so a host with no CUDA runtime can report on
    some rows and not others).
    """

    from .engine_compat import (GAPS, SIGNATURE_GAPS, engine_gaps,
                                engine_signature_gaps)

    symbol_gaps = [
        {"module": gap.module, "symbol": gap.symbol, "stops": gap.stops,
         "handling": gap.handling,
         "stops_a_native_forecast": gap.stops_a_native_forecast}
        for gap in engine_gaps()]
    # A GAP STOPS WHAT ITS ROW NAMES.  This document already said so in the
    # sentence below, and then contradicted it: every symbol gap was read as
    # stopping the native suite, which was true while every row here WAS a
    # native column symbol and stopped being true when the carve took that
    # physics into this package.  The two rows left on a published 2.7 stop
    # a card-pricing check no subcommand reaches and a scorecard regrid, and
    # the menu reported the native profile inadmissible because of them.
    blocking_symbol_gaps = [row for row in symbol_gaps
                            if row["stops_a_native_forecast"]]
    signature_gaps = []
    unanswerable = []
    for gap in SIGNATURE_GAPS:
        missing = gap.missing()
        if missing is None:
            unanswerable.append({
                "module": gap.module, "name": gap.name,
                "keywords": list(gap.keywords),
                "why": "the module could not be imported on this host, which "
                       "on a machine with no CUDA runtime is expected; the "
                       "question is answered on the card host"})
            continue
        if missing:
            signature_gaps.append({
                "module": gap.module, "name": gap.name,
                "missing_keywords": list(missing),
                "declared_keywords": list(gap.keywords),
                "stops": gap.stops})
    runnable = (not blocking_symbol_gaps and not signature_gaps
                and not unanswerable)
    if runnable and symbol_gaps:
        statement = (
            "the installed engine accepts every call the native suite makes; "
            "it runs here.  Standing gap(s) that stop something else: "
            + "; ".join(f"{row['module']}.{row['symbol']} stops "
                        f"{row['stops']}" for row in symbol_gaps))
    elif runnable:
        statement = ("the installed engine accepts every call this package "
                     "makes; the native suite runs here")
    elif signature_gaps:
        statement = (
            "the installed engine does not accept "
            + "; ".join(
                "{}.{} keyword(s) {}".format(
                    row["module"], row["name"],
                    ", ".join(row["missing_keywords"]))
                for row in signature_gaps)
            + ".  A native-physics forecast refuses at the door rather than "
              "integrating under physics its receipt does not describe.")
    elif blocking_symbol_gaps:
        statement = (
            "the installed engine is missing "
            + ", ".join(f"{row['module']}.{row['symbol']}"
                        for row in blocking_symbol_gaps)
            + ", and a native-physics forecast is one of the things each of "
              "those stops.")
    else:
        statement = (
            "this host cannot answer for "
            + ", ".join(f"{row['module']}.{row['name']}"
                        for row in unanswerable)
            + ": those modules need a CUDA runtime to import.  Run this probe "
              "on the card host for a verdict.")
    return {
        "native_suite_runnable_here": runnable,
        "statement": statement,
        "symbol_gaps": symbol_gaps,
        # The subset that decides `native_suite_runnable_here`, so a reader
        # never has to re-derive it from the prose.
        "symbol_gaps_stopping_a_native_forecast": blocking_symbol_gaps,
        "signature_gaps": signature_gaps,
        "unanswerable_here": unanswerable,
        "measured_rows": {"symbols": len(GAPS),
                          "signatures": len(SIGNATURE_GAPS)},
        "basis": "woof.globe.engine_compat, inspected against the installed "
                 "engine when this document was built",
    }


def physics_profile_menu() -> dict[str, Any]:
    """The physics suites this package ships, per source, as JSON.

    Two suites, and the difference between them is which arithmetic runs the
    columns: the REFERENCE suite is this package's own float64 column physics,
    and the NATIVE suite is the engine's WRF-derived CUDA column kernels
    through the adapter contract.  Which experiments bind which is read out of
    the shipped TOMLs, never listed here.

    The native suite's admissibility is MEASURED against the installed engine
    (see :func:`_engine_physics_verdict`) rather than asserted, because that
    is exactly the fact a client needs before it offers the choice: on a
    published 2.7.0 engine the calls this package makes are not accepted, and
    a picker that offered the suite anyway would send a user into a refusal
    several commands later.
    """

    from .physics.builtin_adapters import ensure_builtin_global_physics_adapters
    from .physics.registry import global_physics_manifest
    from .sources import source_rows

    ensure_builtin_global_physics_adapters()
    manifest = global_physics_manifest()
    binding, unplaced = _experiments_by_physics()
    verdict = _engine_physics_verdict()
    doors = _experiment_doors()

    def not_plannable(names) -> list[str]:
        """Which of these experiments `run-plan` cannot execute, and it says."""

        return sorted(name for name in names
                      if (doors.get(name) or {}).get("run_plan_route") is None)

    profiles: list[dict[str, Any]] = [{
        "profile_id": "reference-suite-v1",
        "summary": "the package's own reference column physics: radiation, "
                   "surface fluxes, turbulence, convection, saturation "
                   "adjustment and microphysics, each selected per config in "
                   "[reference_physics]",
        "mode": "reference",
        "arithmetic": "this package's own column code at the config's "
                      "precision; no engine kernel is called",
        "admissible": True,
        "why_not": None,
        "backends": ["numpy", "cupy"],
        "experiments": sorted(binding.get("reference-suite-v1", ())),
        "experiments_not_plannable": not_plannable(
            binding.get("reference-suite-v1", ())),
        "not_applicable": dict(PROFILE_FIELDS_NOT_CARRIED),
    }, {
        "profile_id": "dry-core-v1",
        "summary": "no physics suite at all: the dry dynamical core, which is "
                   "what the shallow-water and Held-Suarez cases integrate",
        "mode": "none",
        "arithmetic": "the dycore alone; no column physics runs",
        "admissible": True,
        "why_not": None,
        "backends": ["numpy", "cupy"],
        "experiments": sorted(binding.get("dry-core-v1", ())),
        "experiments_not_plannable": not_plannable(
            binding.get("dry-core-v1", ())),
        "not_applicable": dict(PROFILE_FIELDS_NOT_CARRIED),
    }]
    # Every native adapter the REGISTRY declares, plus any a shipped
    # experiment names that the registry does not: an experiment bound to an
    # adapter nobody registered is a row a picker has to see, because it is a
    # config that will refuse at its own door and the menu is where that is
    # cheap to learn.
    for adapter in sorted(set(manifest) | {
            key for key in binding
            if key not in ("reference-suite-v1", "dry-core-v1")}):
        entry = manifest.get(adapter)
        contract = entry.get("contract", {}) if isinstance(entry, dict) else {}
        registered = entry is not None
        admissible = registered and verdict["native_suite_runnable_here"]
        if not registered:
            why_not = ("no adapter of this name is registered in this "
                       "install, so a config that names it refuses at the "
                       "run door; `woof global physics-manifest` lists what "
                       "is admitted")
        elif not verdict["native_suite_runnable_here"]:
            why_not = verdict["statement"]
        else:
            why_not = None
        profiles.append({
            "profile_id": adapter,
            "summary": "the engine's WRF-derived CUDA column kernels through "
                       "the native adapter contract",
            "mode": "arwen-native",
            "registered": registered,
            "arithmetic": contract.get("precision"),
            "scheme_identity": contract.get("scheme_identity"),
            "admission_status": contract.get("admission_status"),
            "limitations": contract.get("limitations"),
            "admissible": admissible,
            "why_not": why_not,
            "backends": ["cupy"],
            "experiments": sorted(binding.get(adapter, ())),
            "experiments_not_plannable": not_plannable(binding.get(adapter, ())),
            "not_applicable": dict(PROFILE_FIELDS_NOT_CARRIED),
        })

    sources = []
    for row in source_rows():
        if row["role"] == "verification":
            continue
        sources.append({
            "source_id": row["source_id"],
            "display_name": row["display_name"],
            "default_profile_id": None,
            "default_basis": "there is no per-source default: the physics "
                             "suite is the config's own [physics] table, and "
                             "every shipped experiment names one",
            "admissible_count": sum(1 for profile in profiles
                                    if profile["admissible"]),
            "profiles": [{
                "profile_id": profile["profile_id"],
                "admissible": profile["admissible"],
                "why_not": profile["why_not"],
                "select_with": "an experiment whose [physics] mode is "
                               + profile["mode"],
                "experiments": profile["experiments"],
                "experiments_not_plannable":
                    profile["experiments_not_plannable"],
            } for profile in profiles],
        })

    producer = producer_block()
    return {
        "schema": PHYSICS_PROFILES_SCHEMA,
        "producer": producer,
        # The engine version, at the key the engine's own document for this
        # schema id puts it, so a picker written for that document reads it
        # here without digging.  MEASURED from the installed distribution,
        # never declared: the physics a run integrates is the installed
        # engine's, and this document's whole verdict half is about that.
        "gpuwm_version": producer["engine"]["version"],
        "registry_schema": "arwen-global-physics-suite-v1",
        "profile_count": len(profiles),
        "profiles": profiles,
        "source_count": len(sources),
        "sources": sources,
        "engine_verdict": verdict,
        "experiment_count": sum(len(profile["experiments"])
                                for profile in profiles),
        "experiments_unplaced": sorted(unplaced),
        # WHICH DOOR EACH EXPERIMENT ACTUALLY RUNS THROUGH, measured by
        # attempting the load each door performs.  A profile binds an
        # experiment by its [physics] table, which says nothing about whether
        # `run-plan` can execute it: the spectral-core cases are bound to a
        # profile and are integrated through their own module entry, and a
        # menu built from the profile lists alone offered them as plan
        # choices that refuse at the first door.
        "experiment_doors": doors,
        "experiment_doors_basis":
            "each config was loaded through each door's own loader, in the "
            "order a menu should offer them, when this document was built; "
            "the first loader that accepts a config names its door, and a "
            "config no door loads carries the first door's own refusal",
        # Nothing is invented and nothing is silently absent: an engine field
        # a row here cannot answer is NAMED with its reason, exactly as the
        # source registry names its own.
        "engine_fields_not_carried": {
            **PROFILE_FIELDS_NOT_CARRIED,
            **PROFILE_DOCUMENT_FIELDS_NOT_CARRIED},
        "admissibility_rules": [{
            "rule": "engine-signature-admissibility",
            "owner": "woof.globe.engine_compat.require_engine_signature",
            "applies_to": "every arwen-native profile",
            "declares": {
                "measured_against": "the installed engine's own signatures",
                "refusal_point": "the run door, before anything is allocated",
            },
        }, {
            "rule": "selection-is-the-config",
            "owner": "woof.globe.config.load_config",
            "applies_to": "every source",
            "declares": {
                "selector": "[physics] mode, and [physics] "
                            "native_adapter_name for the native suite",
                "run_option": "none: a plan does not override a config's "
                              "physics",
            },
        }],
    }


def source_inventory() -> dict[str, Any]:
    """Every source this package reads, as JSON.  See :mod:`woof.globe.sources`."""

    from .sources import source_inventory as build

    return build()


def probe_environment(*, readiness: bool = True) -> dict[str, Any]:
    """This machine's device inventory and readiness, as one JSON document.

    THE DEVICE HALF is read through NVML by the engine's own path
    (``woof.core.preflight`` plus ``woof.supervisor.query_gpus``), never
    through a CUDA context: a front end asking "can I run?" must not become a
    compute contender on the card it is asking about.  That half is always
    safe to poll.

    THE READINESS HALF is ``woof global doctor``'s own report, which is the
    command a user would run and therefore the only readiness answer worth
    giving here.  It reads bytes -- staged door digests, engine symbols,
    authority mappings, shipped configs -- and executes nothing, so unlike the
    engine's readiness half it creates no CUDA context either.
    ``--no-readiness`` still skips it, because it hashes every staged binary
    and that is real work on a busy machine.
    """

    from woof.provenance_gate import receipt_block

    producer = producer_block()
    document: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "producer": producer,
        # As above: the engine's own probe document carries the engine version
        # at this key, so a client reading both reads one field.
        "gpuwm_version": producer["engine"]["version"],
        # WHICH ENGINE TREE would execute, at the top level and under the
        # engine's own key, for the same reason `write_manifest` carries it
        # there: a reader written for the engine's probe looks here.  It is
        # the engine's receipt, not this package's -- `producer` says who
        # replied -- and a probe that dropped it made a client asking "what
        # will run here?" read a document with the answer removed.
        "provenance": receipt_block(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "pid": os.getpid(),
    }

    devices: list[dict[str, Any]] = []
    device_error = None
    try:
        from woof.core.preflight import (device_physical_total_bytes,
                                          device_wide_used_bytes)
        from woof.supervisor import query_gpus

        total = device_physical_total_bytes()
        used = device_wide_used_bytes()
        for identity in query_gpus():
            devices.append({
                "index": identity.index,
                "uuid": identity.uuid,
                "name": identity.name,
                "driver_version": identity.driver_version,
                "memory_total_bytes": total,
                "memory_used_bytes": used,
                "memory_free_bytes": (None if total is None
                                      else max(0, total - used)),
            })
    except Exception as error:  # noqa: BLE001 - a probe reports, never raises
        device_error = f"{type(error).__name__}: {error}"
    document["devices"] = devices
    document["device_query_error"] = device_error
    document["device_query_basis"] = (
        "NVML via nvidia-smi, through the engine's own path; no CUDA context "
        "is created by this probe.  memory_total is the card's, memory_used "
        "is device-wide across every process, so free is what a new run could "
        "actually claim.")

    if not readiness:
        document["readiness"] = {
            "collected": False,
            "ready": None,
            "basis": "readiness was not requested; `ready` is null, meaning "
                     "UNKNOWN and never READY.  The full report hashes every "
                     "staged door binary, which is real work on a busy "
                     "machine; it creates no CUDA context."}
    else:
        try:
            from .doctor import build_report

            report = build_report()
            document["readiness"] = {
                "collected": True,
                "ready": not report.gaps,
                "gaps": len(report.gaps),
                "sections": [{
                    "title": title,
                    "rows": [{"label": row.label, "finding": row.finding,
                              "verdict": row.verdict,
                              "detail": list(row.detail)} for row in rows],
                } for title, rows in report.sections],
                "basis": "`woof global doctor`'s own report, which is what a "
                         "user would run; every check reads bytes and "
                         "executes nothing, so no CUDA context is created",
            }
        except Exception as error:  # noqa: BLE001 - a probe reports
            document["readiness"] = {
                "collected": False,
                "ready": None,
                "error": f"{type(error).__name__}: {error}",
                "basis": "readiness could not be established; `ready` is "
                         "null, which means UNKNOWN and never READY"}

    document["routes"] = route_summaries()
    document["stages"] = {name: list(route.stages)
                          for name, route in sorted(ROUTES.items())}
    document["event_tags"] = list(EVENT_TAGS)
    document["schemas"] = {
        "plan": PLAN_SCHEMA, "event": EVENT_SCHEMA,
        "manifest": MANIFEST_SCHEMA, "resolve": RESOLVE_SCHEMA,
        "estimate": ESTIMATE_SCHEMA, "probe": PROBE_SCHEMA,
        "catalog": CATALOG_SCHEMA, "sources": SOURCES_SCHEMA,
        "physics_profiles": PHYSICS_PROFILES_SCHEMA}
    return json.loads(json.dumps(document, default=_jsonable))


# ---------------------------------------------------------------------------
# The durable documents a run writes
# ---------------------------------------------------------------------------


def manifest_physics(plan: RunPlan) -> dict[str, Any]:
    """The physics this plan runs, for the manifest, before any work starts.

    The engine's manifest carries a ``physics`` block from 2.8.0 on, so a
    client that reads a run's physics off the manifest finds it on this one
    too.  It is this package's own snapshot, the one the resolved document
    already carries, plus who stated it.  It never raises: a manifest that
    failed to write because a config could not be read would lose the run's
    attach point, so the error is recorded as ``unresolved`` instead, the
    same way the engine records it.
    """

    try:
        if plan.config_path is not None:
            cfg, _ = _config_from_plan(plan)
        else:
            with tempfile.TemporaryDirectory(prefix="arwen-global-manifest-") as scratch:
                cfg, _ = _config_from_plan(plan, into=Path(scratch))
        document = dict(_physics_snapshot(cfg))
    except Exception as error:  # noqa: BLE001 - the manifest is never lost to this
        return {"stated_by": f"the {plan.config_kind} configuration",
                "unresolved": f"{type(error).__name__}: {error}"}
    document["stated_by"] = f"the {plan.config_kind} configuration's [physics] table"
    return document


def write_manifest(plan: RunPlan, *, run_dir: Path, run_id: str,
                   superseded: Mapping[str, Any] | None = None,
                   started_at_utc: str) -> Path:
    """Publish the attach manifest before any work starts.

    Every filename and every schema id below is IMPORTED -- from
    :mod:`woof.runplan` for the event stream and this document, from
    :mod:`woof.supervisor` for the heartbeat and the failure capsule -- and
    the writer is the engine's own atomic one.  A consumer written for the
    engine's manifest reads this one; ``producer`` is what tells it which
    distribution replied.
    """

    from woof import proc_identity
    from woof.provenance_gate import receipt_block
    from woof.supervisor import (FAILURE_CAPSULE_NAME, FAILURE_CAPSULE_SCHEMA,
                                  HEARTBEAT_NAME, HEARTBEAT_SCHEMA,
                                  atomic_write_json)

    from .status import STATUS_NAME, SCHEMA as STATUS_SCHEMA

    document = {
        "schema": MANIFEST_SCHEMA,
        "producer": producer_block(),
        # WHICH ENGINE TREE is executing this plan, at the top level and under
        # the engine's own key, because that is where a reader written for the
        # engine's manifest looks for it.  The pid and the run_id say which
        # process; nothing else here says which code.
        "provenance": receipt_block(),
        "name": plan.name,
        "route": plan.route,
        "run_id": run_id,
        "pid": os.getpid(),
        # The pid names a process only until it ends; its creation time and
        # boot, from the engine's own identifier, let a client tell this run
        # from a later program that reused the pid before it reports the run
        # alive or signals it.
        "process": proc_identity.identify(os.getpid()),
        "started_at_utc": started_at_utc,
        "plan_source": plan.source,
        "plan_sha256": plan.sha256,
        "run_dir": str(run_dir),
        "outputs_dir": str(run_dir),
        "events_path": str(run_dir / EVENTS_FILENAME),
        "events_schema": EVENT_SCHEMA,
        "progress_path": str(run_dir / HEARTBEAT_NAME),
        "progress_schema": HEARTBEAT_SCHEMA,
        "failure_capsule_path": str(run_dir / FAILURE_CAPSULE_NAME),
        "failure_capsule_schema": FAILURE_CAPSULE_SCHEMA,
        # This distribution's own small status file, which the `run`, `go` and
        # `render` doors already write and which a workspace may already poll.
        # Named here so a client never has to know the filename.
        "status_path": str(run_dir / STATUS_NAME),
        "status_schema": STATUS_SCHEMA,
        # Where the PREVIOUS run's stream went, when this run replaced one
        # under `run_options.overwrite`; null when this directory held none.
        # A client that was reading the old stream finds it under this name
        # rather than finding its bytes gone.
        "superseded_events_path": (
            None if not superseded else superseded["rotated_to"]),
        "reattach": (
            "read progress_path for CURRENT state, replay events_path from "
            "byte zero for HISTORY, then tail it for live detail; the "
            "heartbeat is the durable anchor, the event stream is the "
            "fine-grained feed"),
        "physics": manifest_physics(plan),
    }
    path = run_dir / MANIFEST_FILENAME
    atomic_write_json(path, document)
    return path


def _write_failure_capsule(
        plan: RunPlan, *, run_dir: Path, run_id: str, stage: str | None,
        step: int, error_type: str, message: str, traceback_text: str,
        config_path: Path | None = None,
        declared_inputs: Sequence[Mapping[str, Any]] = (),
        backend: str | None = None) -> tuple[Path | None, str | None]:
    """The engine's crash report, at the path this run's manifest names.

    THE BREAKAGE THIS PREVENTS: ``run-manifest.json`` declares
    ``failure_capsule_path`` and ``failure_capsule_schema`` from the first
    millisecond of a run, so a client branches on them; a field naming a
    document nothing ever writes is worse than an absent field, because the
    client that looks finds nothing and cannot tell a missing capsule from a
    missing run.  The DOCUMENT is the engine's, written by
    :func:`woof.supervisor.write_failure_capsule`, so a support tool that
    reads the engine's capsules reads this one with no new parser.

    Never raises.  A reporter that raises replaces the failure it exists to
    report, so every step here is guarded and the return says which of the
    two happened: the path written, or ``None`` with the reason, which the
    ``failed`` event then carries instead of the path.
    """

    try:
        from woof.supervisor import (FAILURE_CAPSULE_NAME, GPUIdentity,
                                      write_failure_capsule)

        from .runner import CHECKPOINT_PREFIX

        try:
            config_bytes = plan.config_bytes()
        except Exception:  # noqa: BLE001 - an unreadable config is the crash
            config_bytes = None
        config_sha256 = ("" if config_bytes is None
                         else hashlib.sha256(config_bytes).hexdigest())
        named = config_path or plan.config_path or Path(plan.source)

        # WHICH CARD, and never a card this run did not use.  A numpy run
        # selects no device, so reporting the box's first one would put a
        # card in a crash report that never touched it.
        gpu = None
        if backend == "cupy":
            try:
                from woof.supervisor import query_gpus

                gpu = query_gpus()[0]
            except Exception as error:  # noqa: BLE001 - a report never raises
                gpu = GPUIdentity(
                    uuid="unknown", driver_version="unknown",
                    name="no device could be queried: "
                         f"{type(error).__name__}: {error}", index=None)
        if gpu is None:
            gpu = GPUIdentity(
                uuid="none", driver_version="none",
                name="no device was selected: this run's backend was "
                     + (backend or "not chosen before the failure"),
                index=None)

        checkpoints = sorted(run_dir.glob(f"{CHECKPOINT_PREFIX}*.npz"))
        inputs = {
            "{}:{}".format(entry.get("role"), entry.get("path")): {
                "path": entry.get("path"),
                "present": entry.get("present"),
                "role": entry.get("role"),
                # The engine's capsule embeds the verbatim text of a declared
                # input whose role is one it captures; this package declares
                # GRIB2 analyses, which are hundreds of megabytes, so no
                # capture identity is offered and none is taken.
                "identities": [],
            } for entry in declared_inputs}

        path = write_failure_capsule(
            run_dir / FAILURE_CAPSULE_NAME, run_id=run_id,
            config_path=named, config_sha256=config_sha256,
            input_hashes=inputs, gpu=gpu,
            last_phase=stage or "before the first stage",
            last_step=step, exception_type=error_type,
            exception_message=message, exception_traceback=traceback_text,
            last_durable_wrfout=None,
            last_checkpoint=(str(checkpoints[-1]) if checkpoints else None),
            worker_pid=os.getpid(), config_bytes=config_bytes)
        return Path(path), None
    except Exception as error:  # noqa: BLE001 - the failure outranks its report
        return None, f"{type(error).__name__}: {error}"


class _Heartbeat:
    """``run-progress.json``, through the engine's own dataclass and writer.

    The engine's :class:`woof.supervisor.RuntimeHeartbeat` is not used
    directly for one measured reason: its ``__call__`` runs
    ``validate_manifest_checkpoint`` over the last checkpoint, which reads the
    engine's restart header, and this model's checkpoints carry their own.  A
    heartbeat that raised on a healthy checkpoint would kill the run it exists
    to report on.  The DOCUMENT is the engine's, field for field, because
    :class:`woof.supervisor.Heartbeat` builds it and
    :func:`woof.supervisor.write_heartbeat` writes it.
    """

    def __init__(self, path: Path, *, run_id: str, config_sha256: str,
                 started_at_utc: str):
        self.path = Path(path)
        self.run_id = run_id
        self.config_sha256 = config_sha256
        self.started_at_utc = started_at_utc
        self.model_elapsed_seconds = 0.0
        self.outer_step = 0
        self.last_checkpoint: str | None = None

    def _write(self, status: str) -> None:
        from woof.supervisor import (HEARTBEAT_SCHEMA, Heartbeat, utc_now,
                                      write_heartbeat)

        write_heartbeat(self.path, Heartbeat(
            HEARTBEAT_SCHEMA, self.run_id, self.config_sha256, os.getpid(),
            self.started_at_utc, utc_now(), status,
            self.model_elapsed_seconds, self.outer_step, None,
            self.last_checkpoint))

    def preparing(self, phase: str) -> None:
        import re

        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", phase).strip("-")
        self._write(f"preparing:{normalized or 'stage'}")

    def finalizing(self, phase: str) -> None:
        import re

        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", phase).strip("-")
        self._write(f"finalizing:{normalized or 'stage'}")

    def integrating(self, *, model_elapsed_seconds: float, outer_step: int,
                    last_checkpoint: str | None = None) -> None:
        self.model_elapsed_seconds = float(model_elapsed_seconds)
        self.outer_step = int(outer_step)
        if last_checkpoint is not None:
            self.last_checkpoint = str(Path(last_checkpoint).resolve())
        self._write("integrating")

    def complete(self) -> None:
        self._write("complete")

    def failed(self) -> None:
        self._write("failed")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class _StageRunner:
    """One run's stages, across all three of the surfaces it reports on.

    THREE SURFACES, one call site each, because a run that told them different
    things would be worse than a run that told only one.  The event stream is
    the history a client replays, the heartbeat is the current-state authority
    a client polls, and ``status.json`` is the small file the ``run``, ``go``
    and ``render`` doors already write and a terminal workspace may already be
    polling -- so a plan run leaves one too, at the path its own manifest
    names.
    """

    def __init__(self, plan: RunPlan, *, events: EventStream,
                 heartbeat: _Heartbeat, run_dir: Path, status):
        self.plan = plan
        self.events = events
        self.heartbeat = heartbeat
        self.status = status
        self.run_dir = Path(run_dir)
        self.stage: str | None = None
        self.stage_started = 0.0
        # The last model step this run reached, for the crash report: a
        # capsule whose last_step is always zero cannot say whether a run
        # died on its first step or its ten-thousandth.
        self.last_step = 0
        self._announced: set[str] = set()
        self._first_products_announced = False

    # -- stages ---------------------------------------------------------

    def begin(self, stage: str, **fields: Any) -> None:
        self.stage = stage
        self.stage_started = time.perf_counter()
        self.heartbeat.preparing(stage)
        # A stage that announces how many steps it has SIZES the status
        # file's progress bar with the same number, read off the event it is
        # already emitting rather than passed twice.  Without it a workspace
        # polling status.json during a forecast reads `step: 47,
        # step_count: 0`, which is a numerator with no denominator and the
        # one thing that file exists to give.
        total = fields.get("total_steps")
        self.status.stage(stage, step_count=int(total) if total else 0)
        self.events.emit("stage_started", stage=stage, **fields)

    def end(self, stage: str, **fields: Any) -> None:
        self.events.emit(
            "stage_finished", stage=stage,
            elapsed_seconds=round(time.perf_counter() - self.stage_started, 3),
            **fields)
        self.stage = None

    def skip(self, stage: str, reason: str) -> None:
        """A stage that had nothing to do, said out loud on every surface.

        The engine's tags are the whole vocabulary a consumer switches on, so
        a skip is a started/finished pair carrying ``skipped`` rather than a
        tag this stream does not have.
        """

        self.begin(stage, skipped=True, reason=reason)
        self.status.note(stage + " skipped: " + reason)
        self.end(stage, skipped=True, reason=reason)

    def warn(self, scope: str, message: str) -> None:
        self.status.note("warning (" + scope + "): " + message)
        self.events.emit("warning", scope=scope, message=message)

    # -- outputs --------------------------------------------------------

    def commit_new_checkpoints(self) -> list[Path]:
        """Announce every checkpoint that is on disk and not yet announced.

        DURABILITY IS THE TEST, not the runner's intent: the checkpoint writer
        is asynchronous, so the step that scheduled a file is not the moment
        the file exists.  Only a file present on disk is announced, which is
        what ``output_committed`` has to mean for a reader that is about to
        open it.
        """

        from .runner import CHECKPOINT_PREFIX

        committed = []
        for path in sorted(self.run_dir.glob(f"{CHECKPOINT_PREFIX}*.npz")):
            key = path.name
            if key in self._announced:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            self._announced.add(key)
            self.events.emit(
                "output_committed", path=str(path), kind="checkpoint",
                bytes=size, domain="global")
            committed.append(path)
        return committed

    def announce_pictures(self, pictures: Sequence[Path], *,
                          outdir: Path) -> None:
        for picture in pictures:
            try:
                size = Path(picture).stat().st_size
            except OSError:
                size = None
            self.events.emit("output_committed", path=str(picture),
                             kind="picture", bytes=size, domain="global")
        if pictures and not self._first_products_announced:
            self._first_products_announced = True
            self.events.emit("first_products_ready", count=len(pictures),
                             outdir=str(outdir))


def _forecast(plan: RunPlan, cfg, config_path: Path, *,
              runner: _StageRunner) -> dict[str, Any]:
    """The forecast stage: the same call ``woof global run`` makes.

    ``woof.globe.runner.run`` is entered directly rather than through
    ``cli._run`` so this module can pass its own progress callback, which is
    what turns the run's own diagnostics into ``model_progress`` events and
    the heartbeat's beats.  Everything the door does BEFORE the call -- the
    memory levers, the card pricing, the config sidecar -- is done here by
    calling the door's own functions, so a plan run is sized by exactly the
    code that sizes a bare run.
    """

    from . import cli
    from .runner import run as run_forecast

    options = plan.run_options
    door = "woof global run-plan"
    lever_namespace = argparse.Namespace(
        config=config_path,
        latitude_bands=options.get("latitude_bands"),
        host_spill=options.get("host_spill"),
    )
    sized, memory_plan = cli._size_the_run(lever_namespace, cfg, door)
    cli.write_run_sidecar(runner.run_dir, config_path)

    runner.begin("initialize",
                 config_hash=sized.config_hash,
                 truncation=sized.truncation,
                 integrator=sized.integrator,
                 backend=sized.backend)
    runner.end("initialize")

    schedule = _frame_schedule(sized, options.get("until_s"),
                               restart=options.get("restart"))
    runner.begin("forecast", total_steps=schedule["total_steps"],
                 dt_s=schedule["dt_s"],
                 expected_checkpoints=schedule["checkpoints"])

    def progress(diag: Mapping[str, Any]) -> None:
        step = int(diag["step"])
        runner.last_step = step
        runner.heartbeat.integrating(
            model_elapsed_seconds=float(diag["time_s"]), outer_step=step)
        runner.status.step(step, note=(
            "step " + str(step) + " of " + str(schedule["total_steps"])
            + ", model time " + format(float(diag["time_s"]), ".0f") + " s"))
        runner.events.emit(
            "model_progress", step=step,
            total_steps=schedule["total_steps"],
            model_seconds=float(diag["time_s"]),
            maximum_wind_m_s=float(diag["maximum_wind_m_s"]),
            global_mean_total_water_kg_m2=float(
                diag["global_mean_total_water_kg_m2"]))
        committed = runner.commit_new_checkpoints()
        if committed:
            runner.heartbeat.integrating(
                model_elapsed_seconds=float(diag["time_s"]),
                outer_step=step, last_checkpoint=str(committed[-1]))

    result = run_forecast(
        sized, runner.run_dir,
        restart=options.get("restart"),
        overwrite=bool(options.get("overwrite")),
        progress=progress,
        until_s=options.get("until_s"),
        sized_by_door=True,
        door_plan=memory_plan,
    )
    # THE FINAL CHECKPOINT REACHES THE HEARTBEAT.  The last checkpoint is
    # submitted to the asynchronous writer inside the last progress callback
    # and lands on disk only when `run_forecast` joins the writer, so the
    # in-loop sweep never sees it and `last_checkpoint` at `status: complete`
    # named the SECOND-TO-LAST checkpoint on every run.  A client that
    # resumes or reports from that field lost the run's final segment.
    final_committed = runner.commit_new_checkpoints()
    if final_committed:
        final_diagnostics = result.get("final_diagnostics") or {}
        runner.heartbeat.integrating(
            model_elapsed_seconds=float(final_diagnostics.get(
                "time_s", runner.heartbeat.model_elapsed_seconds)),
            outer_step=int(final_diagnostics.get(
                "step", runner.heartbeat.outer_step)),
            last_checkpoint=str(final_committed[-1]))
    runner.end("forecast", status=result["status"],
               wall_seconds=result.get("wall_seconds"),
               receipt=result.get("receipt_path"))
    return result


class _NoteCollector:
    """A ``note`` sink for a helper that explains itself by writing a line.

    Not a StatusWriter: the reason a stage is skipped belongs in the skip
    event, and a probe that created a second status file and a second log in
    the run directory would be answering a question by writing to disk.
    """

    def __init__(self) -> None:
        self.notes: list[str] = []

    def note(self, line: str) -> None:
        self.notes.append(str(line))

    def reason(self, fallback: str) -> str:
        return self.notes[-1] if self.notes else fallback


def _statics_stage(plan: RunPlan, cfg, config_path: Path, *,
                   runner: _StageRunner) -> None:
    import types

    from . import cli
    from .go_door import _default_sector_degrees, _statics_needed

    if not plan.run_options.get("statics", True):
        runner.skip("statics", "run_options.statics is false")
        return
    # `_statics_needed` says WHY it answers no by calling `note` on whatever it
    # is handed, and the go door hands it that command's StatusWriter.  A
    # second StatusWriter here would write a second `status.json` and a second
    # log into the run directory -- a PROBE with side effects, racing the run's
    # own status file.  A collector takes the sentence instead, and the skip
    # event carries it, which is where a client reads it anyway.
    probe = _NoteCollector()
    if not _statics_needed(cfg, probe):
        runner.skip("statics", probe.reason(
            "nothing to build: a synthetic planet, or the cache is already "
            "there"))
        return
    runner.begin("statics")
    code = cli._statics(types.SimpleNamespace(
        config=config_path, out=None,
        overwrite=bool(plan.run_options.get("overwrite")),
        sector_degrees=_default_sector_degrees(),
        geog_root=plan.run_options.get("geog_root")))
    if code != 0:
        raise RuntimeError("the statics stage did not finish (exit "
                           f"{code}); its own output says why")
    runner.end("statics")


#: How much of one renderer failure line the `failed` event carries.  The
#: renderer's unknown-slug refusal ends with its whole catalog, and an event
#: field that is mostly a slug list is not a field a client reads.
_RENDER_FAILURE_CHARS = 300


def _render_report(text: str) -> dict[str, Any] | None:
    """The JSON document ``woof.globe.render_door.render`` printed, if any.

    The door prints one indented object per call, after its own human lines,
    so the last flush-left ``{`` that parses is that document.  Nothing is
    raised: this runs while a failure is being reported, and a reporter that
    crashes tells nobody anything.
    """

    import re

    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"(?m)^\{", text))):
        try:
            document, _end = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(document, dict) and "failures" in document:
            return document
    return None


def _render_failure_detail(text: str) -> str:
    """What the render door said did not draw, as a sentence for an event."""

    document = _render_report(text)
    failures = [str(line) for line in (document or {}).get("failures") or []]
    if not failures:
        # NOT a silent fallback: the door raised before it printed its report,
        # or printed one this could not read, and the sentence says which
        # artifact holds the answer instead of implying one was extracted.
        return ("; the render door named no failed product, so the render log "
                "beside the pictures carries what happened")
    shown = []
    for line in failures[:3]:
        if len(line) > _RENDER_FAILURE_CHARS:
            line = line[:_RENDER_FAILURE_CHARS].rstrip() + " [...]"
        shown.append(line)
    more = "" if len(failures) <= 3 else f" (and {len(failures) - 3} more)"
    return "; the renderer reported: " + "; ".join(shown) + more


def _render_stage(plan: RunPlan, config_path: Path, *,
                  runner: _StageRunner) -> None:
    import types

    from .render_door import DEFAULT_PRODUCTS, render
    from .runner import CHECKPOINT_PREFIX

    options = plan.run_options
    skipped = _render_skip_reason(options)
    if skipped is not None:
        runner.skip("render", skipped)
        return
    checkpoints = sorted(runner.run_dir.glob(f"{CHECKPOINT_PREFIX}*.npz"))
    if not checkpoints:
        runner.skip("render", "the forecast wrote no checkpoints")
        return
    runner.begin("render", checkpoints=len(checkpoints))
    outdir = runner.run_dir / "pictures"
    # THE RENDERER'S OWN WORDS REACH THE MACHINE CHANNEL.  `render` reports
    # what did not draw in the JSON document it prints, and every door's
    # stdout is the HUMAN channel here (`run_plan_main` redirects it to
    # stderr so the event stream carries JSONL and nothing else), so a failed
    # render told a client only "exit 1; its own output says why" -- which is
    # the human output this door exists so a client need not parse.  The
    # document is captured, relayed to that channel unchanged, and the lines
    # it names ride in the `failed` event's error block.
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            code = render(types.SimpleNamespace(
                config=config_path, inputs=checkpoints, outdir=outdir,
                start_date=options["start_date"],
                products=options.get("render_products") or DEFAULT_PRODUCTS,
                size="1600x1000", nlat=360, nlon=720, bbox=None,
                tapes_dir=None, keep_tapes=False,
                overwrite=bool(options.get("overwrite"))))
    finally:
        relayed = captured.getvalue()
        if relayed:
            print(relayed.rstrip())
    pictures = sorted(outdir.rglob("*.png"))
    runner.announce_pictures(pictures, outdir=outdir)
    if code != 0:
        raise RuntimeError("the render stage did not finish (exit "
                           f"{code}){_render_failure_detail(relayed)}")
    runner.end("render", pictures=len(pictures))


def _stream_identity(path: Path) -> tuple[str | None, int]:
    """The run a durable event stream belongs to, and how many records it has.

    Reads bytes and never raises: this runs before a run starts, on a file
    another process may have left half-written, and a reporter that crashes
    tells nobody anything.  A stream whose records name no run reports
    ``None``, which means UNKNOWN and is treated as occupied rather than as
    free.
    """

    run_id: str | None = None
    records = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not line.strip():
                    continue
                records += 1
                if run_id is None:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(record, dict):
                        value = record.get("run_id")
                        if isinstance(value, str) and value:
                            run_id = value
    except OSError:
        return None, 0
    return run_id, records


def prepare_run_directory(plan: RunPlan) -> dict[str, Any] | None:
    """Settle what this run does with a directory a previous run wrote in.

    THE BREAKAGE THIS PREVENTS, measured on the Windows desktop 2026-09-09.
    ``events.jsonl`` is the ONE document in a run directory that is appended
    to.  The manifest, the heartbeat and the small status file are each
    replaced whole by their own writers in the first milliseconds of a run,
    but :class:`woof.runplan.EventStream` opens its file in append mode and
    continues the sequence it finds there.  A second run into one directory
    therefore leaves a single stream carrying TWO run ids under one climbing
    sequence, and the Integration Kit's own ``RunReader`` -- the client this
    door exists for -- raises ``InterfaceError("Event belongs to another
    run")`` on its FIRST poll.  That is not a degraded view: the client
    cannot attach to the second run at all.

    THE TWO ANSWERS.  ``run_options.overwrite`` is this package's own word
    for a run that owns its directory, and the event stream is one of the
    artifacts it replaces: the previous stream is rotated to
    ``events-<its run id>.jsonl`` before this run writes a record, so the
    history survives and the live file is this run's alone.  Without
    ``overwrite`` the run is refused, naming the breakage and both remedies,
    because appending in silence is the one answer that leaves a client
    nothing to read.

    Returns ``None`` when the directory holds no previous stream, otherwise
    the rotation this call performed.
    """

    events_path = plan.run_dir / EVENTS_FILENAME
    try:
        occupied = events_path.stat().st_size > 0
    except OSError:
        occupied = False
    if not occupied:
        return None

    previous, records = _stream_identity(events_path)
    named = previous or "a run that named none"
    if not plan.run_options.get("overwrite"):
        raise PlanError(
            "run plan {} would run into {}, whose {} already carries {} "
            "record(s) from {}.  Nothing was started.  A durable event "
            "stream is APPENDED to, so a second run leaves one file carrying "
            "two run ids under one climbing sequence, and a client attaching "
            "through this run's manifest is refused by its own reader on the "
            "first poll (Event belongs to another run) rather than shown a "
            "degraded view.  Point 'output_root' at a directory of this "
            "run's own, or set 'run_options.overwrite' to true, which "
            "rotates that stream aside before this run's first "
            "record.".format(plan.source, plan.run_dir, EVENTS_FILENAME,
                              records, named))

    stem = (f"events-{previous}" if previous else
            "events-superseded-" + datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"))
    rotated = events_path.with_name(stem + ".jsonl")
    suffix = 2
    while rotated.exists():
        rotated = events_path.with_name(f"{stem}-{suffix}.jsonl")
        suffix += 1
    events_path.replace(rotated)
    return {"previous_run_id": previous, "records": records,
            "rotated_to": str(rotated)}


def execute_plan(plan: RunPlan, *, events: EventStream,
                 superseded: Mapping[str, Any] | None = None) -> int:
    """Run one plan to completion, or to its ``failed`` event.

    Exit 0 when the last event is ``completed``, 1 when it is ``failed``, 130
    on a Ctrl-C -- the shell's 128 + SIGINT, and the engine's own convention.

    ``superseded`` is what :func:`prepare_run_directory` did with a previous
    run's stream, so the manifest names the rotated file and this run says on
    its own stream that it replaced one.
    """

    run_dir = plan.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"arwen-global-{uuid.uuid4().hex[:12]}"
    started = _now_utc()

    from woof.supervisor import HEARTBEAT_NAME

    from .status import StatusWriter

    status = StatusWriter(run_dir, "run-plan",
                          stages=ROUTES[plan.route].stages)
    heartbeat = _Heartbeat(run_dir / HEARTBEAT_NAME, run_id=run_id,
                           config_sha256=hashlib.sha256(
                               plan.config_bytes()).hexdigest(),
                           started_at_utc=started)
    heartbeat.preparing("plan-accepted")
    manifest = write_manifest(plan, run_dir=run_dir, run_id=run_id,
                              started_at_utc=started, superseded=superseded)
    events.emit("plan_accepted", run_id=run_id, name=plan.name,
                route=plan.route, plan_source=plan.source,
                plan_sha256=plan.sha256, run_dir=str(run_dir),
                manifest_path=str(manifest), started_at_utc=started)

    status.note("run " + run_id + " from " + plan.source
                + " on route " + plan.route)
    stages = _StageRunner(plan, events=events, heartbeat=heartbeat,
                          run_dir=run_dir, status=status)
    # Bound before the try, so the crash report can still say which config and
    # which declared inputs a failure DURING resolution was working on.
    cfg = None
    config_path = None
    resolution: Mapping[str, Any] | None = None

    def capsule(stage: str | None, error_type: str, message: str,
                traceback_text: str) -> dict[str, Any]:
        """The capsule for this failure, as the fields the event carries."""

        written, error = _write_failure_capsule(
            plan, run_dir=run_dir, run_id=run_id, stage=stage,
            step=stages.last_step, error_type=error_type, message=message,
            traceback_text=traceback_text, config_path=config_path,
            declared_inputs=((resolution or {}).get("declared_inputs") or ()),
            backend=getattr(cfg, "backend", None))
        return {"failure_capsule": None if written is None else str(written),
                "failure_capsule_error": error}
    if superseded:
        stages.warn(
            "run_dir",
            "this directory held an event stream of {} record(s) from {}; "
            "run_options.overwrite rotated it to {}, so this stream carries "
            "this run alone".format(
                superseded["records"],
                superseded["previous_run_id"] or "a run that named none",
                superseded["rotated_to"]))
    try:
        resolution, cfg, config_path = resolve_plan(
            plan, generate_into=run_dir, require_inputs=True)
        events.emit("resolved_plan", run_id=run_id, **{
            key: value for key, value in resolution.items()
            if key not in ("schema", "producer")})
        for warning in resolution["warnings"]:
            stages.warn(warning["scope"], warning["message"])

        if plan.route == "go":
            _statics_stage(plan, cfg, config_path, runner=stages)
        result = _forecast(plan, cfg, config_path, runner=stages)
        if plan.route == "go":
            _render_stage(plan, config_path, runner=stages)

        stages.begin("finalize")
        stages.commit_new_checkpoints()
        receipt = result.get("receipt_path")
        stages.end("finalize", receipt=receipt, status=result["status"])
        if result["status"] != "pass":
            heartbeat.failed()
            status.failed("the run ended " + str(result["status"])
                          + "; its receipt names the gate that did not hold")
            events.emit("failed", run_id=run_id,
                        error={"type": "GateFailure",
                               "message": f"the run ended {result['status']}"},
                        receipt=receipt,
                        remedy="read the receipt beside this file; it names "
                               "the gate that did not hold",
                        **capsule("finalize", "GateFailure",
                                  f"the run ended {result['status']}",
                                  "no traceback: the run reached its own "
                                  "gates and one of them did not hold, which "
                                  "the receipt beside this file names"))
            return 1
        heartbeat.complete()
        status.done("receipt " + str(receipt))
        events.emit("completed", run_id=run_id, status=result["status"],
                    receipt=receipt,
                    wall_seconds=result.get("wall_seconds"),
                    run_dir=str(run_dir))
        return 0
    except KeyboardInterrupt:
        heartbeat.failed()
        status.failed("interrupted; partial output has no completion receipt")
        events.emit("failed", run_id=run_id,
                    error={"type": "KeyboardInterrupt",
                           "message": "interrupted"},
                    remedy="partial output has no completion receipt",
                    **capsule(stages.stage, "KeyboardInterrupt",
                              "interrupted", _traceback.format_exc()))
        return 130
    except BaseException as error:  # noqa: BLE001 - the stream is the report
        heartbeat.failed()
        if isinstance(error, PlanError):
            # `refused` is a separate state from `failed` on purpose: a
            # refusal is this package declining and naming why, and a
            # workspace should show it as an answer rather than as a crash.
            status.refused(str(error).splitlines()[0])
        else:
            status.failed(type(error).__name__ + ": " + str(error))
        events.emit("failed", run_id=run_id, stage=stages.stage,
                    error={"type": type(error).__name__,
                           "message": str(error)},
                    remedy=_remedy(error),
                    **capsule(stages.stage, type(error).__name__, str(error),
                              _traceback.format_exc()))
        if isinstance(error, Exception):
            return 1
        raise


_REMEDIES = {
    "PlanError": "fix the plan document and re-run; nothing was started",
    "FileNotFoundError": "a declared input is not at the path the config "
                         "names; `woof global run-plan PLAN.json --resolve` "
                         "lists all of them with their present flag",
    "GlobalMemoryRefusal": "the card cannot hold this configuration; "
                           "`woof global run-plan PLAN.json --estimate` "
                           "prints the figure and what it is made of",
    "MissingEngineSymbol": "the installed engine does not accept a call this "
                           "package makes; `woof global doctor` names it",
}


def _remedy(error: BaseException) -> str | None:
    return _REMEDIES.get(type(error).__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_plan_main(args: argparse.Namespace) -> int:
    """``woof global run-plan``.

    Exit codes are this distribution's own ladder, not the engine's: 0 when
    the last event was ``completed``, 1 when it was ``failed`` or a query mode
    refused (``woof.globe.cli.EXIT_REFUSED``), 2 when the command line was
    not a command line, 3 when a Rust door is missing, 4 when the card refused,
    and 130 on a Ctrl-C.  A refusal here is a ``PlanError``, which is a
    ``ValueError``, which is what every refusal in this package travels as and
    what ``cli.main`` prints as one sentence.
    """

    machine_channel = sys.stdout

    def answer(document) -> int:
        machine_channel.write(json.dumps(
            document, indent=2, sort_keys=True, default=_jsonable) + "\n")
        machine_channel.flush()
        return 0

    if getattr(args, "catalog", False):
        with contextlib.redirect_stdout(sys.stderr):
            document = render_catalog()
        return answer(document)
    if getattr(args, "sources", False):
        with contextlib.redirect_stdout(sys.stderr):
            document = source_inventory()
        return answer(document)
    if getattr(args, "physics_profiles", False):
        with contextlib.redirect_stdout(sys.stderr):
            document = physics_profile_menu()
        return answer(document)
    if getattr(args, "probe", False):
        with contextlib.redirect_stdout(sys.stderr):
            document = probe_environment(
                readiness=not getattr(args, "no_readiness", False))
        return answer(document)
    if args.plan is None:
        raise PlanError(
            "woof global run-plan needs a PLAN.json, or one of --probe / "
            "--catalog / --sources / --physics-profiles (which need no plan)")

    plan = load_plan(args.plan)
    if getattr(args, "resolve", False) or getattr(args, "estimate", False):
        # An inline config has to be written before it can be loaded, and a
        # query mode has nowhere durable to put it: the scratch directory is
        # created here and removed here, rather than left in the system temp
        # by a function that had no way to know when the caller was done.
        with tempfile.TemporaryDirectory(prefix="arwen-global-plan-") as scratch:
            with contextlib.redirect_stdout(sys.stderr):
                if getattr(args, "resolve", False):
                    document, _cfg, _path = resolve_plan(
                        plan, generate_into=Path(scratch),
                        require_inputs=False)
                else:
                    document = estimate_plan(plan,
                                             generate_into=Path(scratch))
        return answer(document)

    run_dir = plan.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    # BEFORE the stream is opened, because opening it in append mode is the
    # act that would poison it: a directory a previous run wrote in is either
    # this run's to replace or a refusal, and neither answer can be given
    # after a record has been written.
    superseded = prepare_run_directory(plan)
    # The EventStream binds the REAL stdout here, before the redirect below
    # moves everyone else's to stderr.  That split is the whole promise of
    # this front door: stdout is the machine channel and carries JSONL and
    # nothing else.  The doors this module calls all print for a person --
    # the vertical grid sentence, the statics sentence, the sizer's verdict,
    # the renderer's per-picture lines -- and every one of them would
    # otherwise land in the middle of a stream a client is parsing line by
    # line.
    with EventStream(run_dir / EVENTS_FILENAME) as events:
        with contextlib.redirect_stdout(sys.stderr):
            return execute_plan(plan, events=events, superseded=superseded)


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    """Register the machine-facing execution front door."""

    parser = subparsers.add_parser(
        "run-plan",
        help="execute one versioned run plan and emit a structured event "
             "stream (JSONL to <run_dir>/events.jsonl and to stdout) that a "
             "program can consume without parsing any human output")
    parser.add_argument(
        "plan", type=Path, nargs="?", default=None, metavar="PLAN.json",
        help=f"a {PLAN_SCHEMA} document: which route to execute, which "
             "config to execute it with, and where the outputs land")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resolve", action="store_true",
        help="print the fully resolved configuration plus every automatic "
             "resolution as one JSON document, and run nothing")
    mode.add_argument(
        "--estimate", action="store_true",
        help="print this plan's VRAM estimate and output-checkpoint counts "
             "as one JSON document, and run nothing")
    mode.add_argument(
        "--catalog", action="store_true",
        help="print the renderer's product catalog as one JSON document -- "
             "what may be put in the render_products run option, and which "
             "of those a global tape can draw -- and run nothing; needs no "
             "plan")
    mode.add_argument(
        "--sources", action="store_true",
        help="print the source registry as one JSON document -- every source "
             "this model initializes from or scores against, and whether the "
             "installed engine carries its authority mapping -- and run "
             "nothing; needs no plan")
    mode.add_argument(
        "--physics-profiles", dest="physics_profiles", action="store_true",
        help="print the physics menu as one JSON document -- the reference "
             "suite and the native suite, which shipped experiments bind "
             "each, and what the INSTALLED engine can actually run -- and "
             "run nothing; needs no plan")
    mode.add_argument(
        "--probe", action="store_true",
        help="print this machine's device inventory and readiness as one "
             "JSON document; needs no plan.  The device inventory is NVML "
             "only and creates no CUDA context")
    parser.add_argument(
        "--no-readiness", dest="no_readiness", action="store_true",
        help="with --probe, report the device inventory only: the NVML-only "
             "half, safe to poll on a card that is busy")
    parser.set_defaults(func=run_plan_main)


def run_plan_module_entry(argv: Sequence[str] | None = None) -> int:
    """``python -m woof.globe.runplan PLAN.json``.

    Delegates the WHOLE invocation to :func:`woof.globe.cli.main`, not just
    the parser: a second entry point that only shared the parser would need
    its own refusal print boundary, and one boundary with two spellings is
    what keeps a refusal a sentence rather than a traceback.
    """

    from .cli import main

    tokens = list(sys.argv[1:] if argv is None else argv)
    return main(["run-plan", *tokens])


if __name__ == "__main__":
    raise SystemExit(run_plan_module_entry())
