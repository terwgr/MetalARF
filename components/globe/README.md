# WOOF global

A hydrostatic global spectral weather model with its own ensemble data
assimilation. CUDA is the broad forecast route; Metal support is limited to
implemented forecast paths. See the shared [backend guide](../../docs/public/BACKENDS.md).

WOOF global integrates the moist primitive equations on a Gaussian grid in
spherical-harmonic space, on a two-time-level semi-Lagrangian semi-implicit
core at a 300 s step, with a full GPU physics suite (RRTMGP radiation, Morrison
microphysics, Grell-Freitas or New Tiedtke convection, YSU boundary layer, Noah
land, MM5 surface layer) on a 52 to 25 km grid at 40 hybrid levels. It
cold-starts from a single public GDAS analysis, writes render-ready `wrfout`
tapes that the same Rust renderer draws as every other WOOF product, and can
run its own analysis cycle: a 32-member LETKF under the forecast control, fed
by surface observations, radiosondes, buoys, satellite motion vectors and radio
occultations through Rust decoders, with infrared and microwave radiance
operators beside them. At 52 km a 24 hour global forecast takes about four
minutes on a 16 GB card, and at 25 km about sixteen minutes on a 32 GB one.
The performance examples below describe specific CUDA builds and configurations; current Metal coverage is listed in the shared backend guide.

**This is an experimental research model.** Its configuration surface can move
between releases and its output is not a supported product. The claims it does
not make are listed in [docs/ARWEN_GLOBAL_FULL.md](docs/ARWEN_GLOBAL_FULL.md).

---

## Install

Install `recast-woof`, then follow the shared [backend guide](../../docs/public/BACKENDS.md)
for CUDA extras or the Apple Silicon Metal route. Stage the model's native
bridges and observation tools, then inspect the install:

```bash
woof setup
woof fetch-bridges
woof global fetch-doors
woof global doctor
```

The forecast examples below use the CUDA route; Metal coverage is limited to
the implemented paths. The model remains experimental and is not a supported
forecast product.

---

## A forecast day, in one command

```bash
woof global go arwen_global_gdas_t255_native_sl_si_24h \
    --outdir out/day --start-date 2026-09-01_00:00:00
```

That builds the static surface fields, integrates the forecast, exports
render-ready tapes and draws the product pictures, naming the stage it is in
while the detail goes to the run's log.

Fifty-five experiments ship inside the package and resolve by bare name from
any directory (`woof global configs` lists them); a path that exists always
wins over a shipped name, so an edited copy runs.

### The same day, one stage at a time

**1. Fetch the analysis.** One whole-globe GDAS object, about 500 MB, no
credentials.

```bash
woof fetch --source gdas --cycle 2026-09-01T00 --hours 0 \
  --mode full-file --out cases/baseline-2026090100
```

`--hours 0` takes the f000 analysis alone, which for GDAS is the assimilation
cycle's own estimate of the atmosphere rather than a forecast field.
`--mode full-file` takes the whole globe: the default GDAS transport is an area
crop, and the initializer refuses a longitude ring that does not close.

**2. Build the surface statics**, once per truncation, from the geography
archive.

```bash
woof global statics arwen_global_gdas_t255_native_sl_si_24h --geog-root ~/WPS_GEOG
```

**3. Run the day.** Nothing about memory is set: the door prices the card,
chooses the latitude band count and decides which state slices to park on the
host, and refuses before it allocates anything if the card cannot hold the run.

```bash
woof global run arwen_global_gdas_t255_native_sl_si_24h --outdir out/t255-day
```

288 whole steps at 300 s, a checkpoint every three hours, `diagnostics.jsonl`
and a self-hashed receipt beside them. `woof global check-receipt` validates
the receipt; `--restart` continues from a checkpoint.

**4. Draw it.**

```bash
woof global render out/t255-day \
  --outdir out/pictures --start-date 2026-09-01_00:00:00 \
  --products 2m_temperature,mslp_10m_winds,500mb_height_winds
```

`render` takes checkpoints, a whole run directory or tapes that are already
exported; it writes the intermediate tapes itself and removes them unless
`--keep-tapes` is given. No experiment is named on that line because `run` and
`go` leave a byte copy of the config beside their checkpoints, so a run
directory says what made it; `--config` states it for inputs that are not in
one. `woof render TAPE --list-products` prints the catalogue and says which
entries a given tape can actually fill. The tapes carry `MAP_PROJ = 6`, so
global frames go out through the ordinary layout,
`<outdir>/<domain>/<product>/<valid-day>/`.
Weather fields are drawn by the Rust renderer `rw_wrfbatch` and by nothing
else. `woof global export` writes the tapes without drawing them.

**No card?** `arwen_global_moist_smoke` is a T3 four-step numpy run that goes
through the identical run, export and render sequence in seconds and produces a
real global frame. It rehearses the plumbing, not the forecast.

---

## A fresh global analysis

```bash
woof global da fresh arwen_global_gdas_t255_native_sl_si_24h \
  --outdir out/fresh --stream iem-asos --filter letkf --members 32
```

`da fresh` fetches the newest GDAS analysis, derives its run configuration from
the base experiment named on the line, builds the ensemble if the output
directory has none, cycles hourly to the newest observation hour, and hands
back the analysis checkpoint. At least one observation source is required and
named: `--stream` fetches a stream for every window, `--obs` reads a table
already on disk, and both repeat. Without `--filter letkf` the analysis is the
deterministic successive correction with a single member, which is the shipped
default.

The receipt `out/fresh/da-receipt.json` names the analysis checkpoint it handed
back and carries `forecast_command`, the exact line that forecasts from it:

```bash
woof global da forecast out/fresh/fresh-config.toml \
  --analysis out/fresh/arwen_global_analysis_step00000036.npz \
  --outdir out/fresh/forecast
```

The forecast runs to the derived configuration's duration, which is the cycle
span plus `--forecast-hours` (24 by default); `--until-s` stops it earlier. The
legs are separately reachable: `da init` builds the ensemble, `da cycle` runs
the hourly windows over named streams, `da analyze` analyses one instant. Every
analysis writes a report with the observation counts, the quality-control
rejections by name, the o-minus-b and o-minus-a fits by stream and variable,
and the Desroziers diagnostics.

`woof global obs streams` prints the ten observation streams with the Rust
door each one decodes through, whether that decoder is built, and its measured
latency behind real time; `woof global obs fetch` writes a stream's table to
disk for `--obs`.

**What the cycle is worth, measured.** Six hourly LETKF cycles from the
GDAS 2026-08-31 18Z analysis, 32 T127 members under a T255 control, METAR,
IGRA2, NDBC and GOES derived motion winds, 879,488 rows offered over the six
windows: the METAR surface-pressure o-minus-b grew from 115 to 158 Pa across
the six hours while the o-minus-a held at 76 to 82 Pa (MEASURED 2026-09-06).
Scored at the stations against the GDAS analysis of the same instant, the
handed-back analysis reads 2 m temperature +0.45 / 2.10 K against GDAS's
-0.35 / 2.35, 2 m dewpoint +0.16 / 2.60 against -2.53 / 4.10, mean sea level
pressure -1.05 / 2.44 hPa against -1.68 / 2.79, and 10 m wind 2.05 against
1.91 m/s, on a common set of 1,591 stations; at the soundings it is worse than
the GDAS analysis on every row (MEASURED 2026-09-06). **The system ships
selectable, not default**: the cycle beats the analysis it started from at the
surface and does not reach it aloft, and the door says so.

The analysis runs on the card. The analysis proper (the step less the members'
forecast hour and the control re-integration) costs 26.4 s on an RTX 5090,
against 185.8 s on the host path, and the whole hourly cycle costs 390.5 s, so
a cycle keeps up with the hour it covers by a factor of eight
(MEASURED 2026-09-07).

**The quickstart itself, run from the installed wheel.** Three hourly cycles
from the GDAS 2026-09-01 00Z analysis over the public surface stream on the
shipped deterministic filter, on an RTX 5070 Ti shared with another job:
status `pass`, three of three cycles engineering-complete, 596.5 s for the
whole command, 76.1 s a cycle on average and 89.6 s at worst, which is 0.025
of the hour each cycle covers. At the third analysis 20,713 rows were
assimilated and 2,302 withheld from it for the gate of record. Station
pressure read 403.2 Pa o-minus-b against 113.2 o-minus-a on 4,519 rows,
temperature 1.946 against 1.816 K on 4,717, and the wind components 2.157
against 2.047 and 2.156 against 2.085 m/s. Dewpoint read 3.383 against 3.411,
worse after the analysis than before it, and the door printed that as a
reading rather than hiding it: `--moisture-update` is off by default, so
dewpoint rows are scored and not analysed (MEASURED 2026-09-07).

---

## What it runs on

A 24 hour forecast, 288 steps at 300 s on the shipped semi-Lagrangian core, the
full native physics suite, 40 hybrid levels, float32. Wall clock is end to end
from each run's own receipt. "Shared" means another process held the card, so
the row is a capability row rather than a timing.

Every row was measured on the date in its last column, in the engine checkout
this model was developed and graded in, whose version string reads 2.6.4. No
published engine produced them. The physics those runs integrated is carried
inside this package as `woof.globe.core`, so an install on the 2.8 engine
integrates the same bytes apart from the engine fixes taken since, which the
divergence document lists.

| Truncation | Gaussian grid | Equator spacing | Card | A forecast day | Device peak | Measured |
|---|---|---|---|---|---|---|
| T255 | 384 x 768 | 52.1 km | RTX 5070 Ti, 16 GB, alone | **4.28 min** (256.7 s) | 8.70 GiB | 2026-09-06 |
| T255 | 384 x 768 | 52.1 km | RTX 5070 Ti, bare from the door | 245.8 and 246.0 s | 8.70 GiB | 2026-09-07 |
| T255 | 384 x 768 | 52.1 km | RTX 5090, shared | 186.8 and 228.9 s | 8.70 GiB | 2026-09-07 |
| T383 | 576 x 1152 | 34.7 km | RTX 5090, shared | **5.63 min** (337.9 s) | 15.86 GiB | 2026-09-07 |
| T383 | 576 x 1152 | 34.7 km | RTX 5070 Ti, 16 GB | runs, at two latitude bands with all three state slices parked on the host | 12.41 GiB live, 14.99 GiB of card | 2026-09-07 |
| T533 | 801 x 1602 | 25 km | RTX 5090, 32 GB, alone | **15.9 min** (953 s; 2.98 s a step) | 23.23 GiB live, 28.11 held | 2026-09-07 |
| T533 | 801 x 1602 | 25 km | RTX 5070 Ti, 16 GB | reached on the selectable Eulerian core only: 32 bands, every slice parked, 10.65 s a step | 8.86 GiB live | 2026-09-07 |
| T799 | 1200 x 2400 | 17 km | either card | **refused at the door**, before a byte is allocated, with the figure that refuses it | priced 30.31 GiB live | 2026-09-07 |

At the same truncation on the same tree, the selectable Eulerian core
`imex_ssp3` takes 1,099.3 s for the T255 day at a 60 s step and 1,050.2 s at
its shipped 90 s step on the RTX 5070 Ti (MEASURED 2026-09-06 and 2026-09-07).
A 300 s semi-Lagrangian step costs about one Eulerian step, so the day is 288
steps instead of 960.

**The card is priced, not guessed.** `run` estimates the device peak, weighs it
against free VRAM, chooses the latitude band count and the host tier, and
refuses a plan that will not fit, naming which allocation dies first. Across
seventeen legs with a card reading, the gate was on the refusing side every
time, the tightest by 0.047 GiB (MEASURED 2026-09-06). Where a completed run of
a lighter plan bounds a heavier one, the door prices at that measurement
instead of at its model, and the receipt names the run that bounded it.

**Banding does not move a bit.** The physics suite runs a latitude band at a
time, and the ten-step T255 gate of record is byte-identical between one band
and eight: 309 of 309 checkpoint arrays, on both cards (MEASURED 2026-09-07).
The host tier is the same: 442 of 442 checkpoint arrays byte-identical with the
tier off against on (MEASURED 2026-09-07). The banded transforms are exact too:
93 of 93 FFT cases bit-exact at 0 ULP across T255, T383 and T533 crossed with
band counts 2 to 32, and 892 of 892 waist comparisons bit-exact on each card
(MEASURED 2026-09-06).

---

## The doors

One console script, 48 commands. `woof global <command> --help` prints every
option; [docs/CLI-REFERENCE.md](docs/CLI-REFERENCE.md) is the same surface on
one page, generated from the parser.

| Command | What it does |
|---|---|
| `go` | statics, forecast, tapes and pictures in one command |
| `run` | run a configured experiment: the forecast, its gates and its receipt |
| `statics` | build and cache the surface static fields for a truncation |
| `export` | render-ready global lat-lon `wrfout` tapes from checkpoints (`export-wrfout` is an accepted alias of the same door) |
| `render` | draw the products from a finished run through the Rust renderer |
| `configs` | list the experiments that ship inside the package |
| `doctor`, `fetch-doors` | what is installed, staged and missing; and stage what is missing |
| `fetch-analysis` | fetch the one whole-globe GDAS object a cold start needs, and print the run it feeds |
| `assimilate` | assimilate point observations into one checkpoint |
| `cycle` | forecast and assimilate in one process, an analysis every interval |
| `da init`, `da cycle`, `da analyze`, `da fresh`, `da forecast`, `da static-covariance`, `da localisation` | the ensemble door: build, cycle, analyse one instant, do all of it from nothing, forecast from the analysis, estimate the hybrid's static covariance from lagged forecast pairs, derive the localisation radii from the members |
| `obs fetch`, `obs subscribe`, `obs hours`, `obs summary`, `obs streams`, `obs anchors` | the observation tables: fetch them, subscribe to the live feed, see which hours are covered, what each stream carries and where it is anchored |
| `microwave` | the ATMS leg: fetch, decode, thin and score microwave radiances |
| `abi-score`, `abi-reference`, `abi-fast-model` | the GOES ABI infrared operator, its CRTM reference leg and its trained fast model |
| `export-parent`, `make-regional-target`, `translate-regional-frame`, `make-parent-series` (and their `inspect-` twins) | the one-way bridge that hands a limited-area model its lateral boundaries |
| `pins`, `physics-manifest`, `transform-check` | the arithmetic identity, the admitted physics adapters, the transform controls |
| `inspect`, `check-receipt`, `check-migration`, `check-native-evidence`, `check-native-candidate` | validate a checkpoint, a receipt, a migration, a device-qualification record |
| `migrate-level4-checkpoint` | carry an older checkpoint onto the current state contract |
| `native-qualify` | run the target-device qualification battery for the native physics suite |
| `run-plan` | the machine seam: discover sources, products and physics, probe the machine, resolve and estimate a plan, then execute it and leave a durable event stream |
| `sources` | every source this model initializes from or scores against, and which authority table answered its mapping: the engine's or this package's carried copy |

`run`, `go`, `render` and `run-plan` write `status.json` into their output directory,
rewritten atomically at each stage and at each step, and an append-only log
beside it: these are the four long-running doors, and a workspace driving one
of them reads its stage, its progress and its ending from that file rather
than by scraping a growing console. The shorter commands print their answer
and are read by watching stdout. Exit codes are `0` success, `1` a refusal
that names its reason, `2` an argument error, `3` a Rust door missing or
failing its pin, `4` the device declining admission. Every finished run leaves
a self-hashed receipt carrying the configuration identity, the arithmetic
pins, the library versions the numbers rode on, the engine version and the
staged binary digests, so a run can be compared to a configuration without
opening a checkpoint.

### Driving it from a program

A desktop application, a terminal workspace or a CI job drives this model
through the same versioned contract the engine publishes, so a client written
for `woof` drives `woof global` by changing the module it spawns.

```bash
woof global sources --json                          # gpuwm.run-plan.sources.v1
woof global run-plan --catalog                      # gpuwm.run-plan.catalog.v1
woof global run-plan --physics-profiles             # gpuwm.run-plan.physics-profiles.v1
woof global run-plan --probe --no-readiness         # gpuwm.run-plan.probe.v1
woof global run-plan plan.json --resolve            # gpuwm.run-plan.resolved.v1
woof global run-plan plan.json --estimate           # gpuwm.run-plan.estimate.v1
woof global run-plan plan.json                      # execute; events on stdout
```

A plan is a `gpuwm.run-plan.v1` envelope over an experiment TOML, and
`config.path` takes a shipped experiment's bare name. A run writes
`run-manifest.json`, `run-progress.json` and `events.jsonl` beside the
`status.json` above, all three in the engine's own schemas, so a client
reattaches to a run of this model with the reader it already has.
`python -P -m woof.globe.tui_worker --job-dir DIR -- <args>` is the same
durable handshake as `gpuwm.tui_worker` for a detached job.

[docs/ARWEN_GLOBAL_CLIENT.md](docs/ARWEN_GLOBAL_CLIENT.md) is the whole
contract, including what the engine offers that this package does not and why.

---

## Where things come from

- **The engine.** `woof` supplies the GRIB2 source decoder, the observation
  front door, the static-field builder, the LETKF filter core, the `wrfout`
  writer, the physics tables and the renderer. This package supplies the
  spectral core, the global dynamics, the global physics registry, the global
  data assimilation and the doors.
- **Every data path is Rust.** GRIB decode, NetCDF read and write, observation
  decoding, regridding and rendering run in Rust binaries the engine stages;
  Python is the command line and backend orchestration. The binaries arrive as
  release assets and are verified by size, SHA-256 and a contract marker before
  they are used.
- **The observation doors** the assimilation calls are `rw_asos` (surface
  networks), `rw_igra2` (radiosondes), `rw_ndbc` (buoys), `rw_amv` (satellite
  motion vectors), `rw_gnssro` (radio occultation), `rw_wis2` (the WMO
  information system), `rw_goes` (ABI) and `rw_atms` (microwave).
- **Statics** come from the public geography archive through the engine, cached
  once per truncation.

---

## Documentation

| Page | What it is |
|---|---|
| [ARWEN_GLOBAL_QUICKSTART.md](docs/ARWEN_GLOBAL_QUICKSTART.md) | the route above, with the reasoning behind each flag |
| [ARWEN_GLOBAL.md](docs/ARWEN_GLOBAL.md) | what ships, the resolution rungs, and where the envelope ends |
| [ARWEN_GLOBAL_FULL.md](docs/ARWEN_GLOBAL_FULL.md) | the model: dynamics, hybrid coordinate, physics, receipts, and the non-claims |
| [ARWEN_GLOBAL_DA.md](docs/ARWEN_GLOBAL_DA.md) | the assimilation door, its gates and its grade |
| [ARWEN_GLOBAL_LEVEL5.md](docs/ARWEN_GLOBAL_LEVEL5.md) | the native CUDA physics suite and its device-admission battery |
| [LEVEL3_GLOBAL_SPECTRAL_DYCORE.md](docs/LEVEL3_GLOBAL_SPECTRAL_DYCORE.md) | the spectral dynamical core |
| [arwen-global-effective-resolution.md](docs/arwen-global-effective-resolution.md) | what the model resolves, as against what its grid spacing says |
| [arwen-global-ensemble-da.md](docs/arwen-global-ensemble-da.md) | the ensemble filter, its localisation and its inflation |
| [arwen-global-observations.md](docs/arwen-global-observations.md) | the observation streams, their decoders and their error models |
| [arwen-global-abi-operator.md](docs/arwen-global-abi-operator.md) | the GOES ABI clear-sky infrared operator |
| [arwen-global-microwave-operator.md](docs/arwen-global-microwave-operator.md) | the ATMS clear-sky over-ocean operator |
| [ARWEN_GLOBAL_CLIENT.md](docs/ARWEN_GLOBAL_CLIENT.md) | driving this model from a program: the versioned documents, the plan envelope, the durable run files, and what is not offered |
| [CLI-REFERENCE.md](docs/CLI-REFERENCE.md) | every command and every option, generated from the parser |

---

## Licence

Apache-2.0. See [LICENSE](../../LICENSE) and [NOTICE](../../NOTICE).
