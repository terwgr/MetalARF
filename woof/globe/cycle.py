"""The in-process cycle door for WOOF global: ``woof global cycle``.

One process alternates the forecast and the assimilation on the resident
device state: the model integrates to each analysis instant, the door's
analysis (:func:`assimilate.analyse`) is formed against the observation
sources at that instant, the analysed state is written as that hour's
analysis checkpoint and the integration continues from it, opening a new
conservation epoch exactly as a restart from an assimilated checkpoint
does.  After the last analysis the forecast runs on to ``until_s`` (the
config's duration by default), writing the ordinary hourly checkpoints.

The per-segment chain this replaces (``run --until-s``, ``assimilate``,
``run --restart`` per hour, each a process) spent most of every model hour
outside the forecast steps: measured on the T255 hourly ASOS cycle on the
RTX 5090 (2026-09-05), 33 s of steps against 12 s of model build and
statics per segment, 13 s of checkpoint write per segment, 7.7 s of model
build plus 4.5 s of checkpoint read in the assimilation process and 211 s
of analysis arithmetic, of which 125 s were forty-level wind syntheses
for a lowest-level operator and 45 s the spreading sums on the host; on
a loaded node the same hour took 10 to 17 minutes.  This door pays the
model build once, keeps the state on the device between the two halves,
names the background by the identity its checkpoint would carry
(:func:`checkpoint.checkpoint_metadata`) instead of writing and reading
it, and publishes the analysis checkpoint off the model thread.

Bit identity with the chain is a contract, not a hope: the analysis is
formed by the one function the file door calls, on a state whose arrays
the file door would have read back unchanged from its checkpoint, and the
new epoch is opened by the same two calls the runner makes on a restart
that carries a chain.  ``tests/test_arwen_global_cycle.py`` holds the
per-segment chain and this door to the same bytes on the smoke
configuration; the T255 proof on the lane's arm is in its report.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

from .assimilate import (
    AssimilationOptions,
    _write_report,
    load_observations,
)
from .checkpoint import (
    bundle_arrays,
    checkpoint_metadata,
    normalize_trackers,
    read_checkpoint,
    state_from_checkpoint,
    write_checkpoint,
)
from .config import ArwenGlobalConfig
from .da_filter import SuccessiveCorrectionFilter
from .da_scorecard import merge_cycles
from .da_streams import fetch_streams, radiance_streams
from .device_memory import (
    device_memory_receipt,
    disable_fft_plan_cache,
    select_device_allocator,
    start_device_peak_tracking,
)
from .insitu.ledger import InsituLedger
from .obs_table import parse_valid_time
from .pins import pins_hash, pins_receipt
from .receipt import write_receipt
from .runner import (
    CHECKPOINT_PREFIX,
    DIAGNOSTICS_NAME,
    RECEIPT_NAME,
    SUPPLEMENTARY_TRACKER_KEYS,
    _CheckpointWriter,
    _append_diagnostics,
    _carries_assimilation_chain,
    _checkpoint_path,
    _owned_files,
    _prepare_outdir,
    _release_cached_device_blocks,
    _host_spill_receipt,
    _sizing_model_peak_bytes,
    latitude_bands_receipt,
    _update_trackers,
    build_model_and_cold_state,
    build_transform,
    run_gates,
    vertical_grid_receipt,
)

ANALYSIS_PREFIX = "arwen_global_analysis_step"
REPORT_PREFIX = "assimilation-report-step"
DEFAULT_INTERVAL_S = 3600.0


def _to_numpy(value):
    """Make an explicit host copy at the cycle's I/O and diagnostics seams."""
    from woof.metal.array import MetalArray

    if isinstance(value, MetalArray):
        return value.to_numpy()
    return np.asarray(value)


def _host_state(state):
    """Copy resident Metal fields for the host-based Global analysis seam."""
    atmosphere = state.atmosphere
    atmosphere = atmosphere.with_fields(
        [_to_numpy(value) for value in atmosphere.fields()]
    ).with_grid_tracers({
        name: _to_numpy(value)
        for name, value in atmosphere.grid_tracers().items()
        if value is not None
    })
    surface = replace(
        state.surface,
        **{
            name: None if value is None else _to_numpy(value)
            for name, value in vars(state.surface).items()
        },
    )
    physics_state = replace(
        state.physics_state,
        arrays={name: _to_numpy(value)
                for name, value in state.physics_state.arrays.items()},
    )
    return replace(
        state, atmosphere=atmosphere, surface=surface,
        physics_state=physics_state,
    )

# What the door does when a cycle's gate of record fails: the background
# is carried forward unchanged and the failure is recorded.  Applying an
# analysis that predicts withheld reports worse than the background did
# would write the over-fit into every later hour of the lineage.
GATE_FAILURE_CARRIES_BACKGROUND = (
    "the cycle's gate of record failed, so the background is carried "
    "forward unchanged and written as this hour's checkpoint; an analysis "
    "that fits the withheld reports worse than the background did is not "
    "an analysis (assimilate.GATE_BREAKAGE)"
)

# What the door does when SOME variables fail the gate of record: those
# variables' reports are withdrawn and the hour is analysed again from the
# same background with the rest, which then have to pass on their own.
# Under hourly cycling a variable's withheld fit reaches the report
# noise within a few hours and ties the background (measured on the T255
# cycle of 2026-09-05: temperature 1.53 K against 1.53 K at hour 3,
# dewpoint 1.85 against 1.84 K at hour 4), and a whole-hour carry would
# then throw away the other four variables' improvement for a tie that
# is not over-fitting.  Noise still fails: a variable whose withheld fit
# is worse than the background is dropped, and the second pass is judged
# by the same gate, so nothing enters the state that the withheld
# reports did not accept.
PARTIAL_ANALYSIS_RULE = (
    "the variables that failed the gate of record are withdrawn and the "
    "hour is analysed again from the same background with the remaining "
    "reports, which have to pass the gate on their own; the dropped "
    "variables are carried as the background"
)


def _analysis_path(outdir: Path, step: int) -> Path:
    return outdir / f"{ANALYSIS_PREFIX}{int(step):08d}.npz"


def _report_path(outdir: Path, step: int) -> Path:
    return outdir / f"{REPORT_PREFIX}{int(step):08d}.json"


def cycle_owned_files(outdir: Path) -> list[Path]:
    """Everything this door writes into ``outdir``: the run's own files
    plus the analysis checkpoints and their reports."""
    return [
        *_owned_files(outdir),
        *sorted(outdir.glob(f"{ANALYSIS_PREFIX}*.npz")),
        *sorted(outdir.glob(f"{REPORT_PREFIX}*.json")),
    ]


def configured_start_time(cfg: ArwenGlobalConfig) -> dt.datetime | None:
    """The run's start instant when the physics options declare one
    (``start_time_utc`` of the native suite), else None."""
    options = cfg.native_adapter_options or {}
    value = options.get("start_time_utc") if isinstance(options, dict) else None
    if value is None:
        return None
    parsed = parse_valid_time(str(value))
    if parsed is None:
        raise ValueError(
            f"the physics options' start_time_utc {value!r} is not an "
            "ISO-8601 instant"
        )
    return parsed


def resolve_start_time(cfg: ArwenGlobalConfig, start_utc) -> dt.datetime:
    """The instant model time zero stands for: ``start_utc`` when given
    (ISO-8601 or datetime), else the config's own; refused by name when
    neither exists, because every analysis time is this plus model time."""
    if start_utc is not None:
        if isinstance(start_utc, dt.datetime):
            moment = start_utc if start_utc.tzinfo else start_utc.replace(tzinfo=dt.timezone.utc)
            return moment.astimezone(dt.timezone.utc)
        parsed = parse_valid_time(str(start_utc))
        if parsed is None:
            raise ValueError(f"--start-utc {start_utc!r} is not an ISO-8601 instant")
        return parsed
    configured = configured_start_time(cfg)
    if configured is None:
        raise ValueError(
            "the cycle needs the instant model time zero stands for, because "
            "each analysis is formed at the run's start plus its model time "
            "and the observation age window is measured from it: pass "
            "--start-utc, or set start_time_utc in the config's physics options"
        )
    return configured


def analysis_steps(
    cfg: ArwenGlobalConfig, *, interval_s: float, cycles: int,
    start_step: int, total_steps: int,
) -> tuple[int, list[int]]:
    """``(interval_steps, [analysis steps])``: the analyses fall on whole
    multiples of the interval from model time zero, after ``start_step``
    and no later than ``total_steps``; refused by name when the interval
    is not a whole number of steps or the cycles do not fit."""
    if not math.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("--interval-s must be a positive, finite model time")
    ratio = interval_s / cfg.dt_s
    if abs(ratio - round(ratio)) > 1.0e-6:
        raise ValueError(
            f"--interval-s {interval_s:g} is not a whole number of "
            f"{cfg.dt_s:g} s steps; an analysis must fall on a step boundary"
        )
    interval_steps = int(round(ratio))
    if isinstance(cycles, bool) or int(cycles) != cycles or cycles < 1:
        raise ValueError("--cycles must be a positive whole number")
    steps = [
        step for step in range(interval_steps, total_steps + 1, interval_steps)
        if step > start_step
    ]
    if len(steps) < cycles:
        end_s = total_steps * cfg.dt_s
        raise ValueError(
            f"{cycles} cycles at --interval-s {interval_s:g} from step "
            f"{start_step} need model time beyond {end_s:g} s, where this run "
            "ends (--until-s or the config's duration_s); fewer cycles, a "
            "shorter interval or a longer run"
        )
    return interval_steps, steps[:cycles]


def cycle(
    cfg: ArwenGlobalConfig,
    outdir: str | Path,
    *,
    obs_locations: list[str],
    cycles: int,
    start_utc=None,
    interval_s: float = DEFAULT_INTERVAL_S,
    restart: str | Path | None = None,
    overwrite: bool = False,
    options: AssimilationOptions | None = None,
    until_s: float | None = None,
    keep_backgrounds: bool = False,
    partial_analyses: bool = True,
    progress=None,
    analysis_progress=None,
    filter=None,
    streams=None,
    fetch_dir: str | Path | None = None,
    increment_application: str = "iau",
    information_cutoff=None,
    sized_by_door: bool = False,
    door_plan=None,
    metal: bool = False,
) -> dict[str, object]:
    """Integrate ``cfg`` into ``outdir``, analysing the resident state
    against ``obs_locations`` every ``interval_s`` of model time for
    ``cycles`` analyses, then on to ``until_s`` (default the config's
    duration).  ``partial_analyses`` (default on) applies
    :data:`PARTIAL_ANALYSIS_RULE` when some variables fail the gate;
    off, a failed gate carries the whole background as the per-segment
    chain did.

    ``filter`` forms the increment (:mod:`woof.globe.da_filter`;
    the deterministic successive correction by default), ``streams`` are
    observation streams (:mod:`woof.globe.da_streams`) fetched for
    each analysis window into ``fetch_dir`` (default ``outdir/fetch``) and
    decoded beside the ``obs_locations`` tables; at least one of the two
    must offer reports.  A window's streams are fetched when the window
    OPENS (the previous analysis, or the start), so the filter can take
    observation-space equivalents at the reports' own times as the state
    steps through the window (``filter.begin_window`` / ``filter.observe``,
    design amendment B).  ``increment_application`` ``direct`` inserts the
    analysis increment at the analysis instant; ``iau`` re-integrates the
    window from its start adding the increment in equal parts at every
    step (incremental analysis update, amendment D) and hands back the
    state that arrives at the analysis instant.  Every analysis report
    carries the DA scorecard with its four assessments, the first step
    after every applied analysis is read for its surface-pressure
    tendency against the last step before it (the imbalance reading), and
    the receipt's ``cycle`` record stacks them per cycle with the wall
    budget of every cycle.  ``information_cutoff`` (an instant) drops
    every decoded row whose ``received_time`` lies after it and counts the
    rows whose receipt time nobody recorded (amendment F, row by row
    through the observation streams module).  Returns the receipt, the
    run's with a ``cycle`` record."""
    # Same order as the forecast door: the allocator first, so the slab's
    # arena is the first thing on the card, then the peak hook on top of it.
    allocator = select_device_allocator(
        cfg.device_allocator, cfg.backend,
        predicted_peak_bytes=_sizing_model_peak_bytes(cfg),
    )
    tracker = start_device_peak_tracking(cfg.backend)
    restore_plan_cache = disable_fft_plan_cache(cfg.backend)
    metal_context = None
    try:
        if metal:
            from .runner import metal_refusal
            refusal = metal_refusal(cfg)
            if refusal is not None:
                raise ValueError(refusal)
            from woof.metal.runtime import MetalContext
            metal_context = MetalContext()
        return _cycle_tracked(
            cfg, outdir, obs_locations=list(obs_locations), cycles=int(cycles),
            start_utc=start_utc, interval_s=float(interval_s), restart=restart,
            overwrite=overwrite, options=options or AssimilationOptions(),
            until_s=until_s, keep_backgrounds=bool(keep_backgrounds),
            partial_analyses=bool(partial_analyses),
            progress=progress, analysis_progress=analysis_progress,
            tracker=tracker, allocator=allocator,
            filter=filter or SuccessiveCorrectionFilter(),
            streams=list(streams or []),
            fetch_dir=None if fetch_dir is None else Path(fetch_dir),
            increment_application=str(increment_application),
            information_cutoff=information_cutoff,
            sized_by_door=bool(sized_by_door),
            door_plan=door_plan,
            metal_context=metal_context,
        )
    finally:
        if metal_context is not None:
            metal_context.close()
        restore_plan_cache()
        if tracker is not None:
            tracker.uninstall()
        if allocator is not None:
            allocator.uninstall()
            close = getattr(allocator, "close", None)
            if close is not None:
                close()


def _cycle_tracked(
    cfg: ArwenGlobalConfig,
    outdir,
    *,
    obs_locations: list[str],
    cycles: int,
    start_utc,
    interval_s: float,
    restart,
    overwrite: bool,
    options: AssimilationOptions,
    until_s,
    keep_backgrounds: bool,
    partial_analyses: bool,
    progress,
    analysis_progress,
    tracker,
    filter,
    streams,
    fetch_dir,
    increment_application="iau",
    information_cutoff=None,
    allocator=None,
    sized_by_door: bool = False,
    door_plan=None,
    metal_context=None,
) -> dict[str, object]:
    output = Path(outdir)
    start_time = resolve_start_time(cfg, start_utc)
    if increment_application not in ("direct", "iau"):
        raise ValueError("increment_application must be 'direct' or 'iau'")
    if information_cutoff is not None and not isinstance(information_cutoff, dt.datetime):
        parsed = parse_valid_time(str(information_cutoff))
        if parsed is None:
            raise ValueError(f"information cutoff {information_cutoff!r} is not an ISO-8601 instant")
        information_cutoff = parsed
    cutoff_counters = {"cutoff_utc": None if information_cutoff is None else information_cutoff.isoformat(timespec="seconds"),
                       "offered": 0, "kept": 0, "after_cutoff": 0, "latency_unverified": 0}

    def admit(new_rows):
        """The rows available by the information cutoff (row by row through
        the observation streams module), the counters accumulated."""
        from .obs_streams import apply_information_cutoff

        kept, counters = apply_information_cutoff(list(new_rows), information_cutoff)
        for key in ("offered", "kept", "after_cutoff", "latency_unverified"):
            cutoff_counters[key] += int(counters[key])
        return kept
    if fetch_dir is None:
        fetch_dir = output / "fetch"
    if until_s is None:
        until_s = float(cfg.duration_s)
    until_s = float(until_s)
    if not math.isfinite(until_s) or until_s <= 0.0:
        raise ValueError("--until-s must be a positive, finite model time")
    if until_s > cfg.duration_s + 1.0e-9:
        raise ValueError(
            f"--until-s {until_s:g} lies beyond the config's duration_s "
            f"{cfg.duration_s:g}; a cycle cannot outrun the run it belongs to"
        )
    if abs(until_s / cfg.dt_s - round(until_s / cfg.dt_s)) > 1.0e-6:
        raise ValueError(
            f"--until-s {until_s:g} is not a whole number of {cfg.dt_s:g} s "
            "steps"
        )
    total_steps = int(round(until_s / cfg.dt_s))
    # The observation tables are decoded once; each cycle's age window and
    # duplicate collapse run inside the analysis against its own instant.
    # The streams are fetched per analysis window and decoded as they land.
    if not obs_locations and not streams:
        raise ValueError(
            "the cycle needs at least one --obs source or one --stream "
            "(da_streams.STREAM_TABLE); nothing was offered"
        )
    if obs_locations:
        sources, rows = load_observations(obs_locations)
        rows = admit(rows)
    else:
        sources, rows = [], []
    decoded_digests = {str(source.get("sha256")) for source in sources}

    restart_metadata = None
    restart_arrays = None
    if restart is not None:
        restart_metadata, restart_arrays = read_checkpoint(
            restart,
            expected_config_hash=cfg.config_hash,
            semi_implicit_scheme=cfg.semi_implicit_scheme,
            integrator=cfg.integrator,
        )
    start_step = 0 if restart_metadata is None else int(restart_metadata["step"])
    interval_steps, planned = analysis_steps(
        cfg, interval_s=interval_s, cycles=cycles,
        start_step=start_step, total_steps=total_steps,
    )
    output.mkdir(parents=True, exist_ok=True)
    keep = frozenset() if restart is None else frozenset({Path(restart).resolve()})
    existing = [path for path in cycle_owned_files(output) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"WOOF global output exists in {output}; pass --overwrite to "
            "replace owned files"
        )
    for path in existing:
        if path.resolve() not in keep:
            path.unlink()
    _prepare_outdir(output, overwrite, keep=keep)
    diagnostics_path = output / DIAGNOSTICS_NAME
    receipt_path = output / RECEIPT_NAME

    transform = build_transform(cfg)
    backend = transform.backend
    model, cold = build_model_and_cold_state(
        cfg, transform, scratch_destination=output,
        metal_diffusion_context=metal_context)
    cold_diag = model.diagnostics(cold)
    target_mass = cold_diag["global_mean_surface_pressure_pa"]
    target_water = cold_diag["global_mean_total_water_kg_m2"]
    trackers = normalize_trackers()
    supplementary = {name: 0.0 for name in SUPPLEMENTARY_TRACKER_KEYS}
    epochs: list[dict[str, object]] = []
    if restart is None:
        state = cold
    else:
        state = state_from_checkpoint(restart_metadata, restart_arrays, backend)
        trackers = normalize_trackers(restart_metadata["run_trackers"])
        if state.time_s >= until_s - 1.0e-9:
            raise ValueError(
                f"restart checkpoint at {state.time_s:g} s is already at or "
                f"beyond the end --until-s {until_s:g}"
            )
        model.enforce(state)
        if _carries_assimilation_chain(restart_metadata):
            # The same two calls the runner makes on a restart that
            # carries a chain (runner._run_tracked): the increments are
            # deliberate sources, so the epoch is the restart state's own.
            model.initialize_mass_target(state.atmosphere)
            model.initialize_water_target(state)
            epoch = model.diagnostics(state)
            target_mass = epoch["global_mean_surface_pressure_pa"]
            target_water = epoch["global_mean_total_water_kg_m2"]
            epochs.append({
                "opened_by": "restart checkpoint carrying an assimilation chain",
                "step": int(state.step),
                "mass_target_pa": target_mass,
                "total_water_target_kg_m2": target_water,
            })
    del cold, restart_arrays
    _release_cached_device_blocks(backend)

    ledger = (
        InsituLedger(cfg, model, output).attach() if cfg.insitu.enabled else None
    )
    transform_check = transform.transform_check(seed=19)
    # The control's second time level rides its checkpoints, as the runner's
    # do, so a forecast from the analysis handed back continues the
    # analysed control as the uninterrupted run would (the semi-Lagrangian
    # core; the Eulerian core carries none).
    writer = _CheckpointWriter(
        cfg, _to_numpy,
        trajectory=model.trajectory_state if model.semi_lagrangian else None,
    )
    output_every = int(round(cfg.output_interval_s / cfg.dt_s))
    checkpoints: list[str] = []
    records: list[dict[str, object]] = []
    start_wall = time.perf_counter()
    initial_segment_diag = model.diagnostics(state)
    _append_diagnostics(diagnostics_path, initial_segment_diag)
    if restart is None:
        path = write_checkpoint(
            _checkpoint_path(output, state.step), state,
            config_hash=cfg.config_hash, to_numpy=_to_numpy,
            trackers=trackers, semi_implicit_scheme=cfg.semi_implicit_scheme,
            integrator=cfg.integrator,
        )
        checkpoints.append(str(path))

    pending = list(planned)
    steps_wall = 0.0
    carried = 0

    def failure_receipt(exc: BaseException) -> None:
        try:
            writer.close()
        except Exception as writer_exc:  # noqa: BLE001 - recorded beside the model's own failure
            exc.__context__ = writer_exc
        failure = {
            "name": cfg.name,
            "status": "error",
            "config_hash": cfg.config_hash,
            "pins_hash": pins_hash(cfg.semi_implicit_scheme, cfg.integrator),
            "pins": pins_receipt(cfg.semi_implicit_scheme, cfg.integrator),
            "model": "moist-hybrid-spectral",
            "execution_backend": "metal" if metal_context is not None else cfg.backend,
            "physics_mode": cfg.physics_mode,
            "physics_split": cfg.physics_split,
            "vertical": vertical_grid_receipt(cfg),
            "latitude_bands": latitude_bands_receipt(model, cfg, sized_by_door),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "completed_step": int(state.step),
            "completed_time_s": float(state.time_s),
            "run_trackers": trackers,
            "supplementary_trackers": supplementary,
            "checkpoints": checkpoints,
            "cycle": _cycle_record(
                start_time, interval_s, interval_steps, planned, records,
                carried, sources, options, keep_backgrounds, steps_wall,
                epochs, partial_analyses=partial_analyses,
                filter_name=filter.name, streams=streams,
            ),
            "device_memory": device_memory_receipt(
                tracker, cfg.backend,
                sizing_model_peak_bytes=_sizing_model_peak_bytes(cfg),
                allocator=allocator,
            ),
            "restart": None if restart_metadata is None else {
                "path": str(restart),
                "checkpoint_self_sha256": restart_metadata["self_sha256"],
            },
            "wall_seconds": float(time.perf_counter() - start_wall),
        }
        if ledger is not None:
            try:
                failure["insitu"] = ledger.on_failure(state)
            except Exception as ledger_exc:  # noqa: BLE001 - recorded, not raised
                failure["insitu"] = {
                    "error_type": type(ledger_exc).__name__,
                    "error_message": str(ledger_exc),
                }
        write_receipt(receipt_path, failure)

    # The filter's resident states: the deterministic state first, then
    # the members an ensemble filter holds; every one is stepped through
    # the one model between analyses (da_filter, the interface note).
    resident = filter.resident_states(state)

    def _ps(bundle):
        g = model.grid_state(bundle.atmosphere, only=("ps",))
        ps = np.asarray(_to_numpy(g["ps"]), dtype=np.float64)
        model.release_syntheses()
        return ps

    def _tendency(before, after):
        return float(np.sqrt(np.mean((after - before) ** 2)) / float(cfg.dt_s))

    def open_window():
        """Fetch the coming window's streams and hand the window to the
        filter (amendment B).  Returns ``(fetched, fetch_wall, window_record)``."""
        t0 = float(state.time_s)
        t1 = float(pending[0]) * float(cfg.dt_s)
        moment_end = start_time + dt.timedelta(seconds=t1)
        fetch_start = time.perf_counter()
        fetched: list[dict[str, object]] = []
        if streams:
            fetch_records, manifest = fetch_streams(
                streams, moment_end - dt.timedelta(seconds=interval_s), moment_end, fetch_dir,
            )
            # A radiance stream's record is its window manifest, decoded by
            # the stream itself into batches (below), never as a table.
            radiance_names = {stream.name for stream in radiance_streams(streams)}
            fresh_paths = [
                record.path for record in fetch_records
                if record.sha256 not in decoded_digests and record.stream not in radiance_names
            ]
            if fresh_paths:
                new_sources, new_rows = load_observations(fresh_paths)
                sources.extend(new_sources)
                rows.extend(admit(new_rows))
                decoded_digests.update(str(s.get("sha256")) for s in new_sources)
            fetched = [record.to_json() for record in fetch_records]
            for entry in fetched:
                entry["manifest"] = str(manifest)
        # The radiance streams hand the filter their own batches, screened
        # against the control background at the window's opening.
        extra_batches: list = []
        extra_record: dict = {}
        for stream in radiance_streams(streams or []):
            from .radiance_streams import RadianceContext

            ensemble = getattr(filter, "ensemble", None)
            if ensemble is None:
                raise ValueError(
                    f"the {stream.name} stream hands the filter radiance batches with their own operators; "
                    f"the {filter.name} filter has no ensemble and no radiance operator (use --filter letkf)"
                )
            context = RadianceContext(
                window_start=moment_end - dt.timedelta(seconds=interval_s), window_end=moment_end,
                epoch=start_time, control_state=state, control_model=model, control_transform=transform,
                cfg=cfg, ensemble=ensemble, out_dir=output,
                observation_bin_s=getattr(filter, "observation_bin_s", None),
            )
            new_batches, record = stream.batches(context)
            extra_batches.extend(new_batches)
            extra_record[stream.name] = record
        window_record = filter.begin_window(
            cfg, model, transform, rows, t0, t1, start_time,
            extra_batches=extra_batches, extra_record=extra_record or None,
        )
        return fetched, time.perf_counter() - fetch_start, window_record

    window_fetched, window_fetch_wall, window_record = ([], 0.0, None)
    if pending:
        window_fetched, window_fetch_wall, window_record = open_window()
    window_start_state = state if increment_application == "iau" else None
    # The second time level the semi-Lagrangian core holds at the window's
    # start, kept beside the start state: the re-integration under the
    # incremental analysis update restarts the window on the same model,
    # whose level by then is the window's END, and a first step
    # extrapolated from an hour later is not the window re-integrated.
    window_start_trajectory = model.trajectory_state() if window_start_state is not None else None
    ps_before_step = None
    pre_analysis_tendency = None
    post_analysis_pending = None  # (record, tendency before) awaiting the first step
    # The window's surface-pressure tendency, every step (one 2-D synthesis
    # a step): the gravity-wave noise level the increment application
    # leaves in the window it was applied over, read as the mean rms
    # tendency of the window's first and last quarters and their ratio (a
    # balanced increment under the incremental update reads near one; a
    # shock reads the first quarter high).  Attached to the analysis record
    # that opened the window when the next analysis closes it.
    window_tendencies: list[float] = []
    window_owner: dict | None = None
    try:
        while state.step < total_steps:
            step_start = time.perf_counter()
            reads_tendency = (bool(pending) and state.step + 1 == pending[0]) or post_analysis_pending is not None
            ps_before_step = _ps(state)
            state, metrics = model.step(state, cfg.dt_s)
            for index in range(1, len(resident)):
                resident[index], _member_metrics = model.step(resident[index], cfg.dt_s)
            resident[0] = state
            steps_wall += time.perf_counter() - step_start
            _update_trackers(trackers, supplementary, metrics)
            filter.observe(state, float(state.time_s))
            tendency = _tendency(ps_before_step, _ps(state))
            window_tendencies.append(tendency)
            if reads_tendency:
                if post_analysis_pending is not None:
                    previous_record, pre = post_analysis_pending
                    previous_record["physical_consistency"] = {
                        "surface_pressure_tendency_rms_pa_s": {
                            "last_step_before_analysis": pre,
                            "first_step_after_analysis": tendency,
                            "ratio_after_over_before": (tendency / pre) if pre and pre > 0.0 else None,
                        },
                        "reads": (
                            "the rms surface-pressure tendency of the first step after the applied "
                            "analysis against the last step before it: a ratio well above one is "
                            "the imbalance the increment inserted (gravity-wave adjustment)"
                        ),
                    }
                    post_analysis_pending = None
                if bool(pending) and state.step == pending[0]:
                    pre_analysis_tendency = tendency
            analysis_due = bool(pending) and state.step == pending[0]
            output_due = state.step % output_every == 0 or state.step == total_steps
            if output_due or analysis_due:
                diag = model.diagnostics(state)
                diag["step_metrics"] = metrics
                _append_diagnostics(diagnostics_path, diag)
                if progress is not None:
                    progress(diag)
            if analysis_due:
                if window_owner is not None and window_tendencies:
                    window_owner["window_surface_pressure_tendency"] = _window_tendency_record(
                        window_tendencies, float(cfg.dt_s))
                pending.pop(0)
                moment = start_time + dt.timedelta(seconds=float(state.time_s))
                record, state, applied = _analyse_resident(
                    cfg, model, transform, state, rows, sources=sources,
                    options=options, moment=moment,
                    output=output, writer=writer, trackers=trackers,
                    keep_backgrounds=keep_backgrounds, checkpoints=checkpoints,
                    partial_analyses=partial_analyses,
                    filter=filter, resident=resident,
                    fetched=window_fetched, fetch_wall=window_fetch_wall,
                    window_record=window_record,
                    increment_application=increment_application,
                    window_start_state=window_start_state, interval_steps=interval_steps,
                    window_start_trajectory=window_start_trajectory,
                )
                resident = filter.resident_states(state)
                record["forecast_steps_wall_s"] = steps_wall - sum(
                    float(r.get("forecast_steps_wall_s", 0.0)) for r in records
                )
                record["budget"] = _budget(record, interval_s)
                if applied and state.step < total_steps:
                    post_analysis_pending = (record, pre_analysis_tendency)
                window_owner = record
                window_tendencies = []
                if pending:
                    window_fetched, window_fetch_wall, window_record = open_window()
                if increment_application == "iau":
                    window_start_state = state
                    window_start_trajectory = model.trajectory_state()
                if applied:
                    # A new conservation epoch, as the runner opens on a
                    # restart that carries a chain: the increments are
                    # deliberate sources, not drift.
                    model.initialize_mass_target(state.atmosphere)
                    model.initialize_water_target(state)
                    epoch = model.diagnostics(state)
                    epoch["analysis_step"] = int(state.step)
                    _append_diagnostics(diagnostics_path, epoch)
                    target_mass = epoch["global_mean_surface_pressure_pa"]
                    target_water = epoch["global_mean_total_water_kg_m2"]
                    epochs.append({
                        "opened_by": "analysis",
                        "step": int(state.step),
                        "mass_target_pa": target_mass,
                        "total_water_target_kg_m2": target_water,
                    })
                    record["epoch"] = epochs[-1]
                else:
                    carried += 1
                records.append(record)
                if analysis_progress is not None:
                    analysis_progress(record)
            elif output_due:
                checkpoint = writer.submit(
                    _checkpoint_path(output, state.step), state, trackers,
                )
                checkpoints.append(str(checkpoint))
    except Exception as exc:
        failure_receipt(exc)
        raise

    writer.close()
    final_diag = model.diagnostics(state)
    gates = run_gates(
        cfg, transform_check, final_diag, target_mass, target_water,
        trackers, supplementary,
    )
    status = "pass" if all(row["passed"] for row in gates.values()) else "fail"
    wall = float(time.perf_counter() - start_wall)
    receipt = {
        "name": cfg.name,
        "status": status,
        "config_hash": cfg.config_hash,
        "config": cfg.config_identity,
        "pins_hash": pins_hash(cfg.semi_implicit_scheme, cfg.integrator),
        "pins": pins_receipt(cfg.semi_implicit_scheme, cfg.integrator),
        "model": "moist-hybrid-spectral",
        "execution_backend": "metal" if metal_context is not None else cfg.backend,
        "physics_mode": cfg.physics_mode,
        "physics_split": cfg.physics_split,
        "vertical": vertical_grid_receipt(cfg),
        "semi_implicit": model.semi_implicit.describe(model.vertical),
        "physics_identity": None if model.physics is None else model.physics.identity,
        "initial": {
            "mode": cfg.initial_mode,
            "provenance": model.initial_provenance,
        },
        "statics": (
            None if not isinstance(model.initial_provenance, dict)
            else model.initial_provenance.get("statics")
        ),
        "transform": transform.identity,
        "latitude_bands": latitude_bands_receipt(model, cfg, sized_by_door),
        "transform_check": transform_check,
        "cold_start_diagnostics": cold_diag,
        "segment_start_diagnostics": initial_segment_diag,
        "final_diagnostics": final_diag,
        "mass_target_pa": target_mass,
        "total_water_target_kg_m2": target_water,
        "gates": gates,
        "run_trackers": trackers,
        "supplementary_trackers": supplementary,
        "checkpoints": checkpoints,
        "segment_until_s": until_s,
        "restart": None if restart_metadata is None else {
            "path": str(restart),
            "checkpoint_self_sha256": restart_metadata["self_sha256"],
            "inherited_run_trackers": restart_metadata["run_trackers"],
        },
        "cycle": _cycle_record(
            start_time, interval_s, interval_steps, planned, records, carried,
            sources, options, keep_backgrounds, steps_wall, epochs, wall=wall,
            partial_analyses=partial_analyses,
            filter_name=filter.name, streams=streams, information_cutoff=cutoff_counters,
        ),
        "wall_seconds": wall,
        "insitu": None if ledger is None else ledger.close(),
        # The same three blocks the forecast receipt carries, because a
        # cycled run is sized by the same door and a reader asking what it
        # chose should not have to know which door wrote the file.
        "latitude_bands": {
            **model.pipeline.receipt(),
            "chosen_by": (
                getattr(door_plan, "bands_chosen_by", None)
                or ("config" if int(getattr(cfg, "latitude_bands", 0) or 0)
                    else "sizer")),
        },
        "host_spill": _host_spill_receipt(cfg, model, plan=door_plan),
        "sizer": None if door_plan is None else door_plan.receipt(),
        "device_memory": device_memory_receipt(
            tracker, cfg.backend,
            sizing_model_peak_bytes=_sizing_model_peak_bytes(cfg),
            allocator=allocator,
        ),
    }
    write_receipt(receipt_path, receipt)
    checked = json.loads(receipt_path.read_text(encoding="utf-8"))
    checked["receipt_path"] = str(receipt_path)
    return checked


def _budget(record: dict[str, object], interval_s: float) -> dict[str, object]:
    """The wall this cycle cost against the model time it advanced: the
    forecast steps, the stream fetch, the background identity, the
    analysis and the checkpoint publish, and the fraction of real time
    that is at this interval (below 1 the cycle keeps up with the clock)."""
    timings = record["timings_s"]
    wall = (
        float(record.get("forecast_steps_wall_s", 0.0))
        + float(timings.get("fetch_s", 0.0))
        + float(timings["background_identity_s"])
        + float(timings["analysis_s"])
        + float(timings["checkpoint_submit_s"])
    )
    phases = timings.get("analysis_phases") or {}
    return {
        "interval_s": float(interval_s),
        "wall_s": wall,
        "forecast_steps_wall_s": float(record.get("forecast_steps_wall_s", 0.0)),
        "fetch_s": float(timings.get("fetch_s", 0.0)),
        "analysis_s": float(timings["analysis_s"]),
        "members_advance_s": float(phases.get("members_advance_s", 0.0)),
        "control_analysis_s": float(phases.get("control_analysis_s", phases.get("ensemble_control_s", 0.0))),
        "ensemble_analysis_s": float(sum(v for k, v in phases.items() if k.startswith("ensemble_"))),
        "iau_reintegration_s": float(phases.get("iau_reintegration_s", 0.0)),
        "checkpoint_submit_s": float(timings["checkpoint_submit_s"]),
        "real_time_fraction": wall / float(interval_s),
        "keeps_up_with_real_time": wall < float(interval_s),
    }


def _analyse_resident(
    cfg, model, transform, state, rows, *, sources, options, moment, output,
    writer, trackers, keep_backgrounds, checkpoints, partial_analyses=True,
    filter=None, resident=None, fetched=(), fetch_wall=0.0, window_record=None,
    increment_application="iau", window_start_state=None, interval_steps=None,
    window_start_trajectory=None,
):
    """One cycle on the resident state at ``moment``.  Returns
    ``(record, state, applied)``: the analysed state when the gate of
    record passed (on every variable, or on the remaining ones after the
    failing variables were withdrawn under ``partial_analyses``), the
    background otherwise.  ``filter`` forms the increment from the
    ``resident`` states (the deterministic one first)."""
    if filter is None:
        filter = SuccessiveCorrectionFilter()
    if resident is None:
        resident = filter.resident_states(state)
    backend = transform.backend
    clock = time.perf_counter()
    step = int(state.step)
    background_path = _checkpoint_path(output, step)
    if keep_backgrounds:
        _, background = writer.submit(
            background_path, state, trackers, want_metadata=True,
        )
        checkpoints.append(str(background_path))
        background_written = True
    else:
        # The identity the background's checkpoint would carry, from the
        # same arrays, without the archive.
        background = checkpoint_metadata(
            bundle_arrays(state, _to_numpy),
            step=step, time_s=float(state.time_s),
            physics_state_schema=state.physics_state.schema,
            physics_metadata=json.loads(json.dumps(
                state.physics_state.metadata, sort_keys=True, allow_nan=False,
            )),
            config_hash=cfg.config_hash, trackers=dict(trackers),
            semi_implicit_scheme=cfg.semi_implicit_scheme,
            integrator=cfg.integrator,
        )
        background_written = False
    identity_wall = time.perf_counter() - clock

    analysis_resident = [_host_state(member) for member in resident]
    analysis, report, phases = filter.analyse(
        cfg, model, transform, analysis_resident, rows,
        sources=sources,
        background={
            "path": str(background_path) if background_written else None,
            "self_sha256": background["self_sha256"],
            "step": step,
            "time_s": float(state.time_s),
        },
        analysis_time=moment,
        options=options,
    )
    dropped: list[str] = []
    first_pass = None
    if report["status"] != "pass" and partial_analyses:
        failed = list(report["gate_of_record"]["failed_variables"])
        judged = [
            name for name, row in report["variables"].items()
            if row.get("gated") and name not in failed
        ]
        if judged:
            # Some variables passed: withdraw the failing ones and analyse
            # the hour again from the same background with the rest.
            first_pass = {
                "failed_variables": failed,
                "variables": {
                    name: {
                        "withheld_o_minus_b_rms": row["withheld"]["o_minus_b"]["rms"],
                        "withheld_o_minus_a_rms": row["withheld"]["o_minus_a"]["rms"],
                    }
                    for name, row in report["variables"].items()
                },
                "analysis_phases": phases,
            }
            remaining = [row for row in rows if row.variable not in failed]
            analysis, report, second = filter.analyse(
                cfg, model, transform, analysis_resident, remaining,
                sources=sources,
                background={
                    "path": str(background_path) if background_written else None,
                    "self_sha256": background["self_sha256"],
                    "step": step,
                    "time_s": float(state.time_s),
                },
                analysis_time=moment,
                options=options,
            )
            phases = {
                key: phases.get(key, 0.0) + second.get(key, 0.0)
                for key in set(phases) | set(second)
            }
            if report["status"] == "pass":
                dropped = failed
    applied = report["status"] == "pass"
    iau_record = None
    if applied and increment_application == "iau":
        if window_start_state is None or interval_steps is None:
            raise ValueError("the incremental analysis update needs the window's start state and step count")
        clock = time.perf_counter()
        analysis, iau_record = incremental_analysis_update(
            model, cfg, window_start_state, state, analysis, int(interval_steps),
            window_start_trajectory=window_start_trajectory,
        )
        iau_record["wall_s"] = time.perf_counter() - clock
        phases = {**phases, "iau_reintegration_s": iau_record["wall_s"]}
    report["increment_application"] = {
        "mode": increment_application,
        "iau": iau_record,
        "rule": (
            "direct: the increment inserted at the analysis instant; iau: the window "
            "re-integrated from its start with the increment added in equal parts at "
            "every step, the state arriving at the analysis instant handed back; the "
            "members take their own increment the same way over the next window"
        ),
    }
    clock = time.perf_counter()
    if applied:
        analysis_path = _analysis_path(output, step)
        _, written = writer.submit(
            analysis_path, analysis, trackers, want_metadata=True,
        )
        checkpoints.append(str(analysis_path))
        report["analysis"] = {
            "path": str(analysis_path),
            "self_sha256": written["self_sha256"],
        }
        state = analysis
        if dropped:
            report["partial"] = {
                "dropped_variables": dropped,
                "rule": PARTIAL_ANALYSIS_RULE,
                "first_pass": first_pass,
            }
    else:
        # The background is carried: it becomes this hour's checkpoint so
        # the lineage on disk has a state at every analysis step.
        if not background_written:
            writer.submit(background_path, state, trackers)
            checkpoints.append(str(background_path))
        report["analysis"] = None
        report["carried"] = {
            "background_path": str(background_path),
            "reason": report.get("carried_reason") or GATE_FAILURE_CARRIES_BACKGROUND,
            "first_pass": first_pass,
        }
    checkpoint_wall = time.perf_counter() - clock
    report["timings_s"] = {
        "background_identity_s": identity_wall,
        "checkpoint_submit_s": checkpoint_wall,
        "fetch_s": float(fetch_wall),
        "analysis_phases": phases,
    }
    report["background"]["written"] = background_written
    report["fetch"] = list(fetched)
    report["filter"] = filter.name
    report["window"] = window_record
    report_path = _report_path(output, step)
    finalized = _write_report(report_path, report)
    record = {
        "step": step,
        "time_s": float(state.time_s),
        "analysis_time_utc": moment.isoformat(timespec="seconds"),
        "status": report["status"],
        "applied": applied,
        "filter": filter.name,
        "background_self_sha256": background["self_sha256"],
        "background_written": background_written,
        "analysis": report["analysis"],
        "dropped_variables": dropped,
        "report": str(report_path),
        "report_self_sha256": finalized["self_sha256"],
        "assimilated_total": report["assimilated_total"],
        "withheld_total": report["withheld_total"],
        "refused_from_chain": report["rejections"]["already_assimilated"],
        "failed_variables": list(report["gate_of_record"]["failed_variables"]),
        "scorecard": report["scorecard"],
        "assessments": _assessment_summary(report),
        "observation_times": report.get("observation_times"),
        "stream_roster": report.get("stream_roster"),
        "increment_application": report["increment_application"]["mode"],
        "lineage": report["lineage"],
        "fetch": list(fetched),
        "variables": {
            name: {
                "count": row["count"],
                "o_minus_b_rms": row["o_minus_b"]["rms"],
                "o_minus_a_rms": row["o_minus_a"]["rms"],
                "withheld_o_minus_b_rms": row["withheld"]["o_minus_b"]["rms"],
                "withheld_o_minus_a_rms": row["withheld"]["o_minus_a"]["rms"],
            }
            for name, row in report["variables"].items()
        },
        "timings_s": {
            "background_identity_s": identity_wall,
            "analysis_s": float(sum(phases.values())),
            "checkpoint_submit_s": checkpoint_wall,
            "fetch_s": float(fetch_wall),
            "analysis_phases": phases,
        },
    }
    return record, state, applied


def _window_tendency_record(tendencies: list[float], dt_s: float) -> dict[str, object]:
    """The window's surface-pressure tendency reading: the mean rms tendency
    of the first and last quarters of the window's steps and their ratio,
    the mean over the window, and the largest step."""
    values = np.asarray(tendencies, dtype=np.float64)
    n = int(values.size)
    quarter = max(1, n // 4)
    first = float(np.mean(values[:quarter]))
    last = float(np.mean(values[-quarter:]))
    return {
        "steps": n,
        "window_s": float(n * dt_s),
        "mean_rms_pa_s": float(np.mean(values)),
        "max_rms_pa_s": float(np.max(values)),
        "first_quarter_mean_rms_pa_s": first,
        "last_quarter_mean_rms_pa_s": last,
        "first_over_last_quarter": (first / last) if last > 0.0 else None,
        "reads": (
            "the rms surface-pressure tendency at every step of the window the analysis was "
            "applied over: a balanced increment under the incremental update leaves the first "
            "quarter at the last quarter's level; a shock reads the first quarter high"
        ),
    }


def incremental_analysis_update(model, cfg, window_start_state, background, analysis, steps: int, *,
                                window_start_trajectory=None):
    """The incremental analysis update of amendment D: the spectral
    increment ``analysis - background`` (the control's, whatever formed it,
    the anchor included) is added in ``steps`` equal parts to the state
    re-integrated from the window's start, so the atmosphere adjusts to
    the increment through the dynamics instead of receiving it at once.
    The state arriving at the analysis instant carries the analysis's
    chain.  Returns ``(state, record)``."""
    from .assimilate import ASSIMILATION_HISTORY_KEY
    from .constants import SPECTRAL_FIELDS
    from .state import ArwenGlobalState

    if int(steps) < 1:
        raise ValueError("the incremental analysis update needs at least one step in the window")
    increment = [a - b for a, b in zip(analysis.atmosphere.fields(), background.atmosphere.fields())]
    part = [inc / float(steps) for inc in increment]
    state = window_start_state
    if model.semi_lagrangian:
        # The window is re-integrated from its start on ITS OWN second
        # time level, not the one the background's last step left on the
        # model (the window's end): the SETTLS extrapolation of the first
        # re-integrated step reads the level one step before the start.
        model.set_trajectory_state(window_start_trajectory)
    for _ in range(int(steps)):
        state, _metrics = model.step(state, cfg.dt_s)
        fields = [f + d for f, d in zip(state.atmosphere.fields(), part)]
        state = ArwenGlobalState(state.atmosphere.with_fields(fields), state.surface, state.physics_state)
        state, _n, _t, _f = model._repair_positivity(state)
        model.enforce(state)
        model.release_syntheses()
    if int(state.step) != int(analysis.step) or abs(float(state.time_s) - float(analysis.time_s)) > 1.0e-6:
        raise ValueError(
            f"the re-integrated window arrived at step {state.step} ({state.time_s:g} s), "
            f"the analysis is at step {analysis.step} ({analysis.time_s:g} s)"
        )
    chain = analysis.physics_state.metadata.get(ASSIMILATION_HISTORY_KEY)
    if chain is not None:
        state.physics_state.metadata[ASSIMILATION_HISTORY_KEY] = json.loads(json.dumps(chain))
    backend = model.transform.backend
    rms = {}
    for name, inc in zip(SPECTRAL_FIELDS, increment):
        grid = np.asarray(_to_numpy(model.transform.inverse(inc)), dtype=np.float64)
        rms[name] = float(np.sqrt(np.mean(grid ** 2)))
    return state, {"steps": int(steps), "increment_grid_rms": rms}


def _assessment_summary(report: dict) -> dict[str, object]:
    """The four assessments of a cycle's report, compressed for the receipt."""
    assessments = (report.get("scorecard") or {}).get("assessments") or report.get("assessments") or {}
    engineering = assessments.get("engineering") or {}
    statistical = assessments.get("statistical_consistency") or {}
    return {
        "engineering": engineering.get("verdict"),
        "engineering_failures": engineering.get("failures"),
        "o_a_below_o_b": statistical.get("o_a_below_o_b"),
        "o_a_not_below_o_b": statistical.get("o_a_not_below_o_b"),
        "unmoved": statistical.get("unmoved"),
        "plausible_cells": statistical.get("plausible_cells"),
        "implausible_cells": statistical.get("implausible_cells"),
    }


def _cycle_record(
    start_time, interval_s, interval_steps, planned, records, carried,
    sources, options, keep_backgrounds, steps_wall, epochs, *, wall=None,
    partial_analyses=True, filter_name=None, streams=(), information_cutoff=None,
) -> dict[str, object]:
    budgets = [r["budget"] for r in records if isinstance(r.get("budget"), dict)]
    record: dict[str, object] = {
        "door": "woof global cycle",
        "filter": filter_name,
        "information_cutoff": information_cutoff,
        "streams": [
            {"name": stream.name, "description": getattr(stream, "description", "")}
            for stream in streams
        ],
        "start_utc": start_time.isoformat(timespec="seconds"),
        "interval_s": float(interval_s),
        "interval_steps": int(interval_steps),
        "planned_analysis_steps": [int(s) for s in planned],
        "analyses": records,
        "completed": len(records),
        "applied": sum(1 for r in records if r.get("applied")),
        "carried": int(carried),
        "carried_rule": GATE_FAILURE_CARRIES_BACKGROUND,
        "partial": sum(1 for r in records if r.get("dropped_variables")),
        "partial_analyses": bool(partial_analyses),
        "partial_rule": PARTIAL_ANALYSIS_RULE,
        "backgrounds_written": bool(keep_backgrounds),
        "obs_sources": sources,
        "options": options.identity(),
        "epochs": epochs,
        "forecast_steps_wall_s": float(steps_wall),
        "analysis_wall_s": float(sum(
            float(r["timings_s"]["analysis_s"]) for r in records
        )),
        "fetch_wall_s": float(sum(
            float(r["timings_s"].get("fetch_s", 0.0)) for r in records
        )),
        # The DA scorecard per cycle, and the per-stream summary across
        # the cycles (da_scorecard.merge_cycles).
        "scorecards": merge_cycles([
            (r["analysis_time_utc"], r["scorecard"])
            for r in records if isinstance(r.get("scorecard"), dict)
        ]),
        # The wall budget: every cycle's wall against its interval.
        "assessments": [
            {"step": r["step"], "analysis_time_utc": r["analysis_time_utc"], **(r.get("assessments") or {}),
             "physical_consistency": r.get("physical_consistency")}
            for r in records
        ],
        "budget": {
            "interval_s": float(interval_s),
            "cycles": [
                {"step": r["step"], "analysis_time_utc": r["analysis_time_utc"], **r["budget"]}
                for r in records if isinstance(r.get("budget"), dict)
            ],
            "mean_wall_s": (sum(b["wall_s"] for b in budgets) / len(budgets)) if budgets else None,
            "max_wall_s": max((b["wall_s"] for b in budgets), default=None),
            "max_real_time_fraction": max((b["real_time_fraction"] for b in budgets), default=None),
            "every_cycle_keeps_up": all(b["keeps_up_with_real_time"] for b in budgets) if budgets else None,
        },
    }
    if wall is not None:
        record["wall_seconds"] = float(wall)
        if records:
            # Wall per model hour over the cycled span (start to the last
            # analysis), the number the door is measured by.
            cycled_span_h = (
                float(records[-1]["time_s"]) - float(records[0]["time_s"])
                + interval_s
            ) / 3600.0
            per_hour = sum(
                float(r["timings_s"]["analysis_s"])
                + float(r["timings_s"]["background_identity_s"])
                + float(r["timings_s"]["checkpoint_submit_s"])
                + float(r["timings_s"].get("fetch_s", 0.0))
                + float(r.get("forecast_steps_wall_s", 0.0))
                for r in records
            ) / max(cycled_span_h, 1.0e-9)
            record["wall_seconds_per_model_hour_cycled"] = per_hour
    return record


__all__ = [
    "ANALYSIS_PREFIX",
    "DEFAULT_INTERVAL_S",
    "GATE_FAILURE_CARRIES_BACKGROUND",
    "PARTIAL_ANALYSIS_RULE",
    "REPORT_PREFIX",
    "analysis_steps",
    "configured_start_time",
    "cycle",
    "cycle_owned_files",
    "incremental_analysis_update",
    "resolve_start_time",
]
