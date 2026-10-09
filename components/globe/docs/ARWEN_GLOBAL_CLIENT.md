# Driving WOOF global from a client

This page is for a program, not a person. A desktop application, a terminal
workspace or a CI job that already drives the WOOF engine through its
versioned CLI drives this model the same way, and this page says exactly what
changes.

**The one-line change.** A client that spawns

```text
python -P -m gpuwm.tui_worker --job-dir DIR -- run-plan PLAN.json
```

spawns

```text
python -P -m woof.globe.tui_worker --job-dir DIR -- run-plan PLAN.json
```

and a client that calls `woof` calls `woof global`. Nothing else changes:
the plan document is the engine's `gpuwm.run-plan.v1`, every reply carries the
engine's schema id, the durable documents a run leaves are the engine's, and
the job directory's handshake, marker names and result schema are the engine's.

Every reply also carries a `producer` block naming this distribution, its
version, and the engine version it answered on. The schema ids are shared on
purpose, so `producer` is what tells a client holding two
`gpuwm.run-plan.sources.v1` documents which command replied.

## The interfaces

| Task | Interface | Result |
|---|---|---|
| Discover data sources | `woof global sources --json` | `gpuwm.run-plan.sources.v1` |
| Inspect one source | `woof global sources SOURCE --json` | the same schema, narrowed to one row |
| Read the same list at a terminal | `woof global sources` | a table, printed from that document |
| Discover plot products | `woof global run-plan --catalog` | `gpuwm.run-plan.catalog.v1` |
| Discover physics choices | `woof global run-plan --physics-profiles` | `gpuwm.run-plan.physics-profiles.v1` |
| Read device inventory | `woof global run-plan --probe --no-readiness` | `gpuwm.run-plan.probe.v1` |
| Read inventory and readiness | `woof global run-plan --probe` | the same schema, `readiness.collected` true |
| Resolve a plan | `woof global run-plan PLAN.json --resolve` | `gpuwm.run-plan.resolved.v1` |
| Estimate its resources | `woof global run-plan PLAN.json --estimate` | `gpuwm.run-plan.estimate.v1` |
| Execute the reviewed plan | `woof global run-plan PLAN.json` | manifest, heartbeat, events, committed output |
| Own a detached job | `python -P -m woof.globe.tui_worker --job-dir DIR [--windows-job NAME] -- ARGS` | `gpuwm-tui-result-v1` |
| List the shipped experiments | `woof global configs` | one name per line |

Treat schema identifiers as required. Inspect both the subprocess exit code
and the returned document. Preserve unknown response fields when forwarding
documents. A query mode writes one JSON document to stdout and everything a
person would read to stderr; an execution writes the event stream to stdout
and everything else to stderr.

Exit codes are this distribution's ladder: `0` the command did what it was
asked, `1` a refusal or a failed run, `2` the command line was not a command
line, `3` a Rust door is missing or fails its pin, `4` the card refused, `130`
a Ctrl-C.

## A plan is an envelope over an experiment TOML

```json
{
  "schema": "gpuwm.run-plan.v1",
  "name": "a reviewed global forecast",
  "route": "go",
  "config": {"path": "arwen_global_gdas_t255_native_sl_si_24h"},
  "output_root": "./out/run",
  "run_options": {
    "start_date": "2026-09-01_00:00:00",
    "render_products": "2m_temperature,mslp_10m_winds"
  }
}
```

`config.path` takes a TOML on disk or **the name of a shipped experiment**. A
path that exists always wins, so a reader who copies a config beside their plan
runs their copy; otherwise the file inside the installed wheel answers and the
resolution says so. `config.inline` carries the TOML text itself.

Review does not start anything:

```bash
woof global run-plan plan.json --resolve
woof global run-plan plan.json --estimate
```

Execution is a separate, explicit operation:

```bash
woof global run-plan plan.json
```

### What review answers before anything is spent

`--resolve` and `--estimate` answer three questions a run would otherwise
answer at its own cost.

* **Are the inputs there.** Every file the config names and the `restart`
  checkpoint the plan names are listed under `declared_inputs` with
  `present`. A hole is a `warnings` row in a query mode and a refusal on the
  route that starts work.
* **Will the pictures be drawn.** `render.will_run` is false when
  `run_options.render` is false or `start_date` is absent, with
  `render.skipped_reason` naming which, and a `warnings` row saying the run
  will write no pictures. A slug the installed renderer does not carry is in
  `render.unknown_products`, and it is **refused** rather than warned when a
  run is being started: the render stage runs after the statics and the whole
  forecast, so a typo there costs a forecast day. A token with a colon
  (`var:<name>`) is one the renderer resolves against the tape's own stored
  variables, so it is reported as unchecked, never as unknown.
* **How many checkpoints.** `output_schedule` and `estimate.disk.checkpoints`
  count what this run will write, which on a restart is only the cadence steps
  **after** the step the restart checkpoint carries, and no cold state.
  `output_schedule.restart_step` names that step, read from the checkpoint
  itself. The forecast's `stage_started` event carries the same figure as
  `expected_checkpoints`, which is what sizes a progress bar.

### The two routes

| Route | What it executes | Stages |
|---|---|---|
| `experiment` | `woof global run CONFIG` | initialize, forecast, finalize |
| `go` | `woof global go CONFIG` | statics, initialize, forecast, render, finalize |

`woof global run-plan --probe` carries both, and every route this build
refuses, under `routes`.

### Run options

| Option | Routes | Meaning |
|---|---|---|
| `restart` | both | continue from this checkpoint instead of the cold state |
| `until_s` | both | stop at this model time instead of the config's `duration_s` |
| `latitude_bands` | both | override the sizer's band count; null lets it choose |
| `host_spill` | both | `auto`, `on` or `off`; null keeps the config's `[memory]` |
| `overwrite` | both | this run owns the directory and replaces the artifacts of a previous run in it, including its event stream |
| `start_date` | `go` | the analysis valid time the render stage stamps tapes with. Without it **the render stage skips itself**, and `--resolve` says so |
| `render_products` | `go` | comma-separated product slugs, or `all`. Every token is checked against the renderer's own catalog at resolve time |
| `geog_root` | `go` | the `WPS_GEOG` archive the statics stage builds from |
| `statics` | `go` | false skips the statics stage |
| `render` | `go` | false stops after the forecast |

An option a route does not take is **refused by name**, never dropped: a plan
whose option was silently discarded runs as something other than what it says.

## What a run writes

Everything lands in the run directory, which is `output_root` itself and not a
directory derived from it.

| File | Schema | What it is |
|---|---|---|
| `run-manifest.json` | `gpuwm.run-manifest.v1` | written before any work starts; names every other document below by absolute path |
| `run-progress.json` | `gpuwm.run-progress/v1` | the current-state authority; rewritten atomically |
| `events.jsonl` | `gpuwm.run-plan.event.v1` | append-only history, dense monotonic `sequence`, also mirrored to stdout |
| `status.json` | `gpuwm-global-status-v1` | this package's own small progress file, which `run`, `go` and `render` already wrote |
| `failure-capsule.json` | `gpuwm.failure-capsule/v3` | **only when a run fails**: the engine's own crash report at `failure_capsule_path` |

The capsule exists exactly when a run ended in `failed`, so its presence is
itself the answer to whether this run reached its own completion. It carries
the config text the run used, the last stage and model step, the exception, the
last checkpoint on disk, and the device the run selected or a named absence
where it selected none. The `failed` event carries the same path in
`failure_capsule`, and `failure_capsule_error` instead when the report itself
could not be written, because a failure outranks its report.

`run-progress.json`'s `last_checkpoint` is the newest checkpoint on disk at
every status it reports, `complete` included, so a client that resumes or
reports from that field never loses the run's final segment. The final
checkpoint is submitted to an asynchronous writer and lands after the last
progress callback, and this file used to name the one before it at
`status: complete`.

Read the manifest first, then the heartbeat for current state, then replay
`events.jsonl` from byte zero and tail it. Ignore an incomplete final line
until more bytes arrive. Reject a changed manifest and reconnect deliberately:
after a crash the last event alone cannot prove the process is alive.

### A second run into the same directory

Three of those four files are replaced whole by their own writers at the start
of a run. `events.jsonl` is not: it is appended to, so a run that found another
run's records there would leave one file carrying two run ids under one
climbing sequence, and a client attaching through the manifest would be refused
by its own reader on the first poll rather than shown a degraded view.

So a run either owns the directory or is refused:

| `run_options.overwrite` | What happens |
|---|---|
| `true` | the previous stream is rotated to `events-<its run id>.jsonl` before this run writes a record, and the new manifest's `superseded_events_path` names it. The live `events.jsonl` carries this run alone, dense from sequence 1, and the run's first `warning` event says what it replaced |
| absent or `false` | the run is **refused** before anything is written, naming the run whose stream is there and both remedies. The directory is left byte for byte as it was |

`superseded_events_path` is `null` on a run that found an empty directory.

Event tags are the engine's, and a consumer switching on `event` can be
exhaustive against `woof.runplan.EVENT_TAGS`. This model emits:

`plan_accepted`, `resolved_plan`, `stage_started`, `stage_finished`,
`model_progress`, `output_committed`, `first_products_ready`, `warning`,
`completed`, `failed`.

`output_committed` is emitted only for a file that **exists on disk**. The
checkpoint writer is asynchronous, so the step that scheduled a file is not the
moment a reader can open it; the tag would be worthless if it meant anything
else. Each carries `kind` (`checkpoint` or `picture`), `path` and `bytes`.

A stage with nothing to do emits a `stage_started`/`stage_finished` pair
carrying `skipped: true` and a `reason`, rather than a tag the stream does not
have.

## Owning a detached job

`woof.globe.tui_worker` is the engine's handshake with this package's CLI.
The launcher spawns it, waits for the `ready` marker, takes ownership of the
process group (POSIX) or the JobObject (`--windows-job NAME`, Windows), then
writes a `start` marker to release it. The worker leaves `result.json`
(`gpuwm-tui-result-v1`) behind whatever happens, including a refusal.

**On Windows the pid a launcher spawned is not the pid running the forecast.**
A virtual environment's `Scripts\python.exe` is a redirector, so
`process.json`'s `pid` is the authority and `--windows-job` is how a launcher
gets a handle that covers the real process. Killing the spawned pid leaves the
run on the card.

The pre-launch handshake is bounded at 60 seconds so an abandoned launch does
not leave a worker forever. That bound covers the seam alone and is not a limit
on preparation or on a forecast: a global day is hours.

## What is not available, and why

| The engine offers | This package | Why |
|---|---|---|
| route `prepared` | refused by name | a prepared plan names a prepared-cache root and a WPS namelist authority; this package reads neither, because a global run is initialized from the analysis file its config's `[initial]` table names |
| `config.intent` | refused by name | that is the domain wizard's question list. This distribution ships no wizard and a global run has no domain to fit, so nothing here could turn an intent into a config |
| a `fetch` block in a plan | refused by name | there is no fetch stage. The analysis is named by the config; the bytes are brought by `woof global obs anchors`, which a plan does not drive |
| `woof remote` | not offered | this package publishes no remote job workspace |
| the compact viewer transfer | not offered | this package's render door draws to disk; there is no processed-frame transfer contract here |
| `wall_time.seconds` in an estimate | always null | no measured rate is published for an arbitrary configuration, and the `model_progress` events carry the real one from the first output step |
| a byte figure for output | always null | bytes per checkpoint are not measured by this package |

A run on the `numpy` backend reports **null** for every device figure in an
estimate, with `device_basis` saying so. Null means absent. A zero would read
as "this run needs no VRAM" rather than "there is no card in this run".

## What the catalog is actually saying

`--catalog` answers two different questions and keeps them apart.

The **slug list** is the renderer's own, asked live rather than transcribed,
and the instrument is the engine's: `woof.runplan.render_catalog` resolves the
renderer through `woof.render._resolve_engine("auto")` and reads its
`--list-products`. A machine with no staged `rw_wrfbatch` gets `products: null`
and the staging remedy in `error`, which is a document a picker can act on.

Whether a slug can be drawn **on a global tape** depends on the field set
`woof.globe.wrfout_export` writes, so it cannot be answered without a tape
and this door takes no plan. It is measured once against a real exported tape
by `tools/arwen_global_render_catalog.py` and shipped as package data with the
host, the date, the renderer digest and the renderer's own reason for every
excluded slug. Each product row therefore carries:

| `global_tape_status` | Meaning |
|---|---|
| `renderable` | the renderer's import catalog offers this slug on a tape this package exported |
| `missing-fields` | the tape does not store a field the product needs; `global_tape_detail` names the field |
| `excluded` | the renderer's catalog excludes it, and the detail says whether that is a heavy grid not computed at import, a windowed accumulation with too few stored frames, or a recipe the wrfout import lane does not realize |
| `unmeasured` | the installed renderer lists this slug and the shipped measurement does not carry it |

`unmeasured` is a third answer and it is never `renderable`. A catalog that
said yes without having asked would be worse than one that says it does not
know. `global_tape.renderer_digest_matches_installed` says whether the
renderer measured is the renderer installed.

**Two arms, and every row carries both.** `global_tape_status` and
`global_tape_detail` are the SINGLE-FRAME measurement: what one stored frame
draws. `global_tape_series_status` and `global_tape_series_detail` are the same
measurement over a tape with more than one stored whole-hour frame, which is
what a windowed accumulation needs, so the series arm is the more permissive of
the two and a slug can be `missing-fields` on one and `renderable` on the other.

`global_tape_counts` summarises the single-frame arm and says so in its own
`basis` field, with the series arm's totals beside it under
`global_tape_counts.series`. A picker drawing the headline number is drawing
the single-frame one. The block also carries `host`, `platform`,
`measured_utc` and `renderer_digest_matches_installed`, the same stamp
`global_tape` carries: `listed` is this install's live renderer, and every
other count is the stored measurement, valid on this machine only where that
digest verdict is true.

## What the physics menu is actually saying

`--physics-profiles` reports the reference suite, the dry core and every native
CUDA adapter, with the shipped experiments each one binds read out of the
TOMLs. The native suite's `admissible` flag is **measured against the installed
engine's own signatures**, not asserted from a version number, and `why_not`
carries the sentence naming what closed it. That is the fact a picker needs
before it offers the choice: a menu that offered a suite the installed engine
cannot run would send a user into a refusal several commands later.

Two things close a native row, and the document separates them. `registered:
false` means no adapter of that name is in this install, whatever the engine
carries. Otherwise the row is closed only by a boundary gap that stops a
forecast: `engine_verdict.symbol_gaps` lists every gap standing on this
machine, and `symbol_gaps_stopping_a_native_forecast` is the subset that
decides `native_suite_runnable_here`. A gap that stops something else, such as
the two absent from every published 2.7, is reported in the first list and not
in the second, and the native suite stays admissible. Read the second list for
the verdict and the first for what else on this machine is unavailable.

`engine_verdict.unanswerable_here` lists the rows a host cannot answer at all,
which on a machine with no CUDA runtime is expected: those modules import CuPy
at module scope. Probe the card host for a verdict there.

**A profile here is not a scheme set.** The engine's row is one fixed
combination and carries its scheme ids, its namelist switches, its level bounds
and its day-only flag. A row here is a MODE that many shipped experiments
select, and each of them chooses its own schemes in its own `[physics]` table.
So every one of those engine row fields is named, with its reason, in the row's
`not_applicable` block rather than filled with one experiment's answer, and the
document repeats the list at `engine_fields_not_carried` along with the one
top-level field it does not carry, `physics_registry_sha256`. That is the same
standard the source registry is held to: an engine field is carried with an
answer or named with a reason, and a test measures the list against the
installed engine's own document so a field the engine grows cannot go quiet
here.

**A profile binds an experiment; it does not say which door runs it.** The
three `global_spectral_*` cases carry a `[physics]` mode like every other
shipped config, and neither `run-plan` nor `woof global run` can execute them:
they load only through `python -m woof.globe.spectral run`. So the document
carries `experiment_doors`, one row per shipped experiment naming the door it
loads through and its `run_plan_route`, measured by attempting each door's own
load, and every profile repeats the subset a plan cannot execute under
`experiments_not_plannable`. A menu built from `experiments` alone offered
three dead choices.

`gpuwm_version` sits at the top level, at the engine's spelling for this schema
id, measured from the installed distribution. `run-plan --probe` carries it the
same way, and carries the engine's top-level `provenance` receipt as well,
where the engine's own probe puts it: which engine tree would execute here.
`producer` is the separate fact of which distribution replied.

## What the source registry is actually saying

`--sources` carries the analyses this model can be initialized from, the
background anchor the assimilation reads, and the published fields its
scorecards grade against. Each row's `mapping` block is measured when the
document is built, against the **installed** engine's authority table first
and this package's carried copies second, and `mapping.origin` says which of
the two answered (`engine` or `package`). So the same package answers
differently on two engines, and `maturity.runnable` follows it. Against
`woof 2.7.0` and `woof 2.8.0` every row reads `origin: "package"` (measured
on the Windows desktop, 2026-09-10 and 2026-09-29): the engine's table carries none of the six, the engine
is still asked first for every row, and its copy wins the day it publishes
one. A row both tables carry with different bytes is a refusal carried in
`mapping.refusal`, naming both digests, rather than a silent choice between
them.

`source_id` and `maturity.field_mapping` are ids and never file names. Two of
the verification rows are opened as files by their own readers, so the
`<id>.mapping.json` spelling stays in `aliases` and still answers
`woof global sources ID`: the resolver accepts either spelling for every row
and reaches the same file.

`maturity.cadence_mapping` is null on every row, and named in
`engine_fields_not_carried` with the reason. The engine's rows put a
cadence-policy registry id there; this registry publishes no such table, so a
string built from the cycle interval would look like an id and resolve
nowhere. The interval itself is carried plainly, in seconds, by
`forcing_interval_seconds`.

Roles are not exclusive. `role` is the primary one, which is what the table
column shows; `roles` names every one that is true of the row with the basis
for each. The background anchor stays the assimilation's background anchor on
the day a shipped experiment also initializes from it, and `used_by` remains
the separate fact of which experiments name it.

The engine's registry rows carry fields that answer a regional preparation
chain: a coverage window, a WRF eta level mapping, a stock-WRF certification
gate, a per-source runner. Those have no meaning for a global spectral model,
so each row carries a `not_applicable` block **naming** the engine field and
the reason rather than filling it with a value. Nothing is invented; a picker
that coloured a row by an invented field would be drawing a fact nobody
measured.

## What a new front end should test

Source and mapping changes, a plan whose declared input is absent, a route or
an option this build refuses, reconnect after the client closes, a torn event
tail, a changed manifest under a pinned reader, a run that fails at its gates,
and a paused cursor while a running forecast appends output. Verify scientific
values and geometry as well as the screenshot.

The WOOF Integration Kit ships inside the installed engine wheel at
`docs/integration-kit/`. Its `examples/client.py` and its `RunReader` drive
this package unaltered once the spawned module is changed, and this package's
own suite (`tests/test_arwen_global_run_plan.py`) runs them against a real
short forecast rather than reimplementing their checks.
