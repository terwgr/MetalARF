# WOOF global quickstart: a 24 h T255 global forecast from one analysis

Four commands, from nothing to rendered global maps. Every one of them is a
door that already ships; this page is the route through them, not a new tool.

WOOF global is an **experimental research model**. It is reached through
`woof global`, its configuration surface can move between releases, and its
output is not a supported product. What ships and where the envelope ends
are in [ARWEN_GLOBAL.md](ARWEN_GLOBAL.md); what the model is, and the list
of things it explicitly does not claim, are in
[ARWEN_GLOBAL_FULL.md](ARWEN_GLOBAL_FULL.md).

## What you get

A 24 hour global forecast at T255 -- a 384 x 768 Gaussian grid, 52.1 km at
the equator, 40 surface-stretched hybrid layers to 100 Pa -- cold-started
from a single GDAS 0.25 degree analysis, written out as nine `wrfout`
history tapes three hours apart, and drawn by the same Rust renderer that
draws every other WOOF product.

## What you need

The example below uses the CUDA route. For current backend coverage and the
Metal selector, see [Backends](../../../docs/public/BACKENDS.md). GDAS is
public and requires no credentials; allow disk space for its input and the
forecast outputs.

## 1. Fetch the analysis

```bash
woof global fetch-analysis --cycle 2026-08-30T18 --out data/gdas-analysis
```

This is the whole-globe f000 fetch with its two non-default flags bound, and
it prints the run command it feeds. The same fetch spelled by hand on the
engine's own door is:

```bash
woof fetch --source gdas --cycle 2026-08-30T18 --hours 0 \
  --mode full-file --out data/gdas-analysis
```

Either lands `data/gdas-analysis/gdas.t18z.pgrb2.0p25.f000` with a
`SHA256SUMS`, a series TSV and a `fetch-manifest.json` recording which
endpoint served each object. Where the bytes come from is table data
(`woof/authorities/rw-wps-fetch-routes.v1.json`), not a code path: GDAS
declares an ordered ladder of NOMADS and the unbounded AWS Open Data archive
`noaa-gfs-bdp-pds`, and the router HEAD-probes the archive per object and
lets it serve whenever it already has the bytes. `--transport s3` pins it.

**Two flags here are not the door's defaults, and both matter:**

- `--hours 0` takes the f000 analysis alone. GDAS is the one source that
  accepts it, because its f000 is the assimilation cycle's own estimate of
  the atmosphere rather than a forecast field -- which is exactly what a
  global cold start wants. There are no lateral boundaries to feed, so one
  object is the whole input.
- `--mode full-file` takes the whole-globe object. The default GDAS
  transport is the NOMADS grib-filter **area crop**, and a regional crop
  cannot initialize a global model: the initializer measures the longitude
  ring and refuses one that does not close (`analysis longitude ring covers
  N degrees, not the globe`). That refusal is deliberate and stays.

## 2. Size it against your card

There is no separate step: `woof global run` prices the card before it
allocates anything and refuses a plan that will not fit. The refusal is
an itemized device estimate and a verdict, the Legendre tables, the
spectral state, the grid fields and the step's working set weighed against
free VRAM less a 0.5 GiB other-process margin, and it names which
allocation dies first and the largest `[grid] truncation` that does fit.
Nothing is written and no device memory is taken when it refuses, so the
cheapest way to size a configuration is to start it.

`woof global run-plan PLAN.json --estimate` prints the same figures without
starting a run, as one JSON document, and a plan's `config.path` takes a
shipped experiment's bare name. The engine's own `woof check` does NOT
price a global config on a published engine: measured on the Windows desktop
2026-09-10 against `woof 2.7.0` and 2026-09-29 against `woof 2.8.0`, it
refuses with `unknown table(s)` naming
this model's tables, because its sizing route knows the regional ones only.
`woof global configs --paths` prints the full path of every shipped
experiment, which is the file to copy and edit.

One number worth reading off either report: the Legendre tables are **cubic in
truncation and carry no vertical levels at all**. Trimming `[vertical]`
cannot make a tables-bound configuration fit. They are also built in
float64 on the *host* whatever the device precision is, so the host figure
does not halve with `precision = "float32"` -- but the build streams one
order band at a time, so at T533 that host peak is 0.21 GiB rather than
the 9.4 GiB the dense float64 squares once needed.

## 3. Run the forecast

```bash
woof global run arwen_global_t255_quickstart \
  --outdir out/arwen-global-24h
```

The shipped configuration reads the file step 1 just wrote,
`data/gdas-analysis/gdas.t18z.pgrb2.0p25.f000`, and decodes it through the
packaged authority mapping named by its bare id,
`analysis_mapping = "gdas-global"`.
A bare id is asked of the engine's authority table first and of this
package's carried copies second, in both the spellings those tables use, and
must match exactly one document in whichever answers, so it works from an
installed wheel as well as from a checkout. Against `woof 2.7.0` and
`woof 2.8.0` the carried copy answers, because the engine's table carries none of the six
mappings this model names; `woof global doctor` says which one answered
each row. Adding a source here is a mapping document, not code.

dt is 300 s on the semi-Lagrangian core, the shipped default, and the run is
86400 s, so it takes 288 whole steps and writes a checkpoint every 36 of them:
nine files, `out/arwen-global-24h/arwen_global_step00000000.npz` through
`...step00000288.npz`, plus `diagnostics.jsonl` and a self-hashed
`arwen-global-receipt.json`. The run prints max wind and global mean total
water as it goes; `woof global run --help` has the rest, and `--restart`
continues from a checkpoint.

## 4. Export render tapes

```bash
woof global export arwen_global_t255_quickstart \
  out/arwen-global-24h/arwen_global_step*.npz \
  --outdir out/arwen-global-tapes --start-date 2026-08-30_18:00:00
```

One `wrfout_d01_<valid time>` NetCDF tape per checkpoint, on a regular
lat/lon grid (360 x 720 by default; `--nlat`/`--nlon` change it, `--bbox`
crops to a window). `--start-date` is the analysis valid time -- the cycle
you fetched -- and every checkpoint's time is offset from it.

## 5. Render

```bash
woof render out/arwen-global-tapes/wrfout_d01_* \
  --products 500mb_height_winds,mslp_10m_winds,2m_temperature \
  --out out/arwen-global-png
```

Nothing about this step is global-specific. The tapes carry `MAP_PROJ = 6`
(cylindrical equidistant) and the production Rust renderer already draws
unrotated global frames, so weather fields go out through the ordinary
[render door](https://github.com/recastsystems/woof/blob/main/docs/public/PIPELINE-STAGES.md) and land in the ordinary layout,
`<out>/<run folder>/<domain>/<product>/<valid-day>/`. `--list-products`
prints the catalogue for a tape and says which entries the tape can actually
fill; `--products all` draws every renderable one, which over nine tapes is
a large number of files.

## Moving to a current cycle

Exactly three spellings travel together, and they are all the same instant:

1. `--cycle` in step 1,
2. the `tHHz` in `initial.analysis_grib` in the config,
3. `--start-date` in step 4.

GDAS publishes on the 00/06/12/18 UTC grid and lags its cycle by about seven
hours. An older cycle still fetches: anything past the operational window
goes straight to the archive rung, which is unbounded.

## Rehearsing the chain without a card

The three model doors are backend-agnostic.
`arwen_global_moist_smoke` is a T3 four-step numpy run.
It goes through the identical `run` -> `export` -> `render` sequence in
seconds and produces a real global frame, so a machine with no CUDA card can
prove the route before renting one:

```bash
woof global run arwen_global_moist_smoke --outdir out/smoke
woof global export arwen_global_moist_smoke \
  out/smoke/arwen_global_step*.npz --outdir out/smoke-tapes \
  --nlat 90 --nlon 180 --start-date 2026-08-30_18:00:00
woof render out/smoke-tapes/wrfout_d01_* --products 2m_temperature \
  --out out/smoke-png
```

It is a rehearsal of the plumbing, not of the forecast: a four-step T3
atmosphere has nothing in it to look at.

## What is measured, and what is not

**Measured.** The `run` -> `export` -> `render` chain composes through these
doors and the Rust renderer draws the global frame;
`tests/test_arwen_global_quickstart.py` exercises it end to end at smoke
truncation, on the CPU backend, against the real renderer binary. The
shipped configuration's own arithmetic -- whole step count, output cadence
landing on steps, the analysis mapping resolving to exactly one authority --
is pinned by the same file.

**Not measured.** A wall clock for T255, and forecast skill of any kind.
Neither number is quoted anywhere on this page for that reason.

## Where to go next

- [ARWEN_GLOBAL.md](ARWEN_GLOBAL.md) -- what ships, the resolution rungs,
  and the envelope: the reference suite's three-day limit, the native
  suite's admission evidence, and what may be said about effective
  resolution.
- [ARWEN_GLOBAL_FULL.md](ARWEN_GLOBAL_FULL.md) -- the model: dynamics,
  hybrid coordinate, reference physics, receipts, and the non-claims.
- [ARWEN_GLOBAL_LEVEL5.md](ARWEN_GLOBAL_LEVEL5.md) -- the native WOOF CUDA
  physics suite and its target-device admission battery.
  `arwen_global_gdas_t255_native_24h` is this
  quickstart's grid and clock with `physics.mode = "arwen-native"`.
- `woof global assimilate` -- the third leg of the door: minutes-fresh
  point observations into a checkpoint, with an o-minus-b/o-minus-a gate of
  record.
- `woof global cycle` -- the forecast and the assimilation in one process:
  an analysis every interval on the resident state, each written as that
  hour's analysis checkpoint, then the forecast from the last one.
- `woof global --help` -- the rest of the research surface
  (pins, checkpoint and export inspection, Level-4 migration, the one-way
  regional parent bridge, the native qualification battery).
- [DATA.md](https://github.com/recastsystems/woof/blob/main/docs/public/DATA.md) -- what GDAS is, what is certified about fetching and
  decoding it, and what the regional route still refuses.
