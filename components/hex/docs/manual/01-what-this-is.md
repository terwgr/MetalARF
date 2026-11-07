# 1. What woof hex is

## For someone who has never run an atmospheric model

An atmospheric model is a program that takes a snapshot of the atmosphere
(temperature, wind, moisture, pressure, everywhere at once) and advances it
forward in time using the equations of fluid motion. Feed it this morning's
atmosphere and it computes this evening's. The snapshot it starts from is
called the **initial conditions**; the ground it stands on (terrain height,
land use, soil type) is called the **static** data; and the arrangement of
points where the atmosphere is represented is called the **mesh**.

woof hex's mesh is not a flat rectangle over one region. It is a sphere tiled
with hexagonal cells (the whole planet at once) and the cells do not all
have to be the same size. You can make them 100 km wide over the oceans and
25 km wide over the region you care about, in one seamless mesh with no nest
boundaries. That is what "global variable-resolution" means.

The model runs on GPU backends. CUDA has the broad forecast coverage; the
implemented Metal route runs on Apple Silicon for a supported subset. This
manual's measurements describe CUDA unless marked otherwise; see the shared
[backend guide](../../../../docs/public/BACKENDS.md) for current route coverage.

## What it is, precisely

- **The dynamics** (the fluid-motion core) is a port of MPAS-Atmosphere
  v8.4.2, byte-identical to native MPAS built with the pinned strict recipe on
  the measured cases (dry, and with water vapour and five condensates), which
  `woof hex verify-dycore` re-checks to 0 differing words (see
  [compatibility](../mpas-compatibility.md) for the scope), and
  deterministic: two runs of the same case produce
  byte-for-byte identical output.
- **The physics** (radiation, clouds, rain, turbulence, land surface) comes in
  two kinds. By default (since 0.4.6) it is one of MPAS-A v8.4.2's own suites,
  picked by mesh (`convection_permitting`; `mesoscale_reference` by name once
  it runs on the GPU), ported scheme by scheme
  and checked bit for bit against native MPAS; a 24-hour forecast with every
  scheme of `convection_permitting` on the GPU matches native MPAS in every
  field hex writes. On the MPAS suites hex runs **correct mode** by default,
  which fixes five defects found in native MPAS; `--physics-mode exact`
  reproduces native bit for bit.
- **The tooling** is a set of doors: `doctor` reports what your install can
  reach, `mesh-check` validates a mesh, `cull` cuts a limited-area case out
  of a global one, `init` builds initial conditions, `forecast` runs the
  model, `swath` decides where a fine grid goes, `cycle` follows weather
  across cycles, `render` turns model output into product images. Each door
  refuses bad input by name rather than producing a plausible wrong answer.

## What it can do

Measured, not promised (the README states each measurement):

- 24-hour global forecasts on one card, deterministic across dual runs.
- Full-physics limited-area forecasts cut out of a global case with the
  `cull` door and driven at their boundary: six hours, 1,080 of 1,080
  steps on an 11,020-cell cut, a 6,224 MiB footprint on a 10 GiB card.
- Weather followed across forecast cycles: `swath` places fine grids from
  a coarse run's own fields, `cycle` cuts, forces and integrates them,
  two real cycles ran back to back, both fine forecasts completing
  1,080 of 1,080 steps.
- Initial conditions built from a WPS intermediate file (GFS, ERA5, or
  whatever you drive with) without running any native Fortran tool; a
  native-built "capsule" is an explicit compatibility mode, not a
  requirement (chapter 5).
- Product PNGs rendered from history files entirely through the Rust
  renderer, filed by domain, product, and valid day (chapter 7).
- New meshes at resolutions you choose, with their static fields, generated
  by the engine's `woof mesh` door with no native toolchain (chapter 4).
- Checkpoint/restart with bitwise-identical continuation, proven in the
  full-physics proof harness (chapter 6).

## What it cannot do, stated up front

- **Each identity claim has a scope.** The dycore identity is measured with
  physics off on the listed meshes; the all-GPU MPAS physics match is one
  24-hour forecast on the 480 km mesh, one process; every other MPAS scheme is
  bitwise on captured inputs. Correct mode differs from native wherever a fix
  acts, by design. Or it is the WOOF engine's suite (`--physics-suite
  arwen_mpas_column`), reached through a pinned seam, so woof hex and the
  `woof` distribution share one physics implementation.
- **WOOF's physics is not MPAS's.** On the WOOF route a full-physics
  forecast is not identical to native MPAS, and the dycore check does not
  qualify one. Beyond the expected chaos-shaped divergence there are **three
  measured, one-signed differences**, the declared divergences of chapter 3.
  Read them before trusting a number from that route.
- **No multi-node.** Two-node execution is built and proven bitwise
  partition-invariant, but it is not part of this release. Several cards on
  one host is a door (chapter 12) whose qualification is not complete.
- **Forecasting runs from the wheel.** Since 0.3.2 `woof hex forecast`
  needs no checkout of either project: its drivers ship in the package and
  the physics seam is the installed `woof`, named in every receipt by
  version and `RECORD` digest (chapter 6.1). `cycle run` still needs a
  woof hex checkout for its per-cycle contract deck (chapter 6.9).
- **No data fetching.** woof hex ships no meshes, no static files, no
  meteorological data. `verify-dycore` downloads only its small public test
  mesh and builds its native reference. Chapter 2 states
  plainly where each asset comes from.
- **Not long-range on the WOOF route.** The measured upper-level warm drift
  (chapter 3) disqualifies that route for multi-day stratospheric work; it is
  bounded and fine at 24 h.
- **A run can stop early.** When a step fails validation (for example a
  vertical-velocity spike past 200 m/s) the model refuses to publish the
  step and stops rather than writing output it does not trust. A refusal is
  designed behavior, not a crash (chapter 3).

## What hardware it needs

Use a supported CUDA or Metal device with enough memory for the selected mesh.
Backend coverage and Metal memory admission are described in the shared
[backend guide](../../../../docs/public/BACKENDS.md); chapter 4 explains the Hex
mesh sizing model. Python 3.11 or newer, native mesh tools, and external mesh,
static, and initial-condition files are also required.
