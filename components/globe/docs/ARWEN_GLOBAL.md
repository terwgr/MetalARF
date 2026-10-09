# WOOF global: the experimental global spectral model

This guide covers the model and its CUDA route. Metal support is limited to
implemented forecast paths; see the shared [backend guide](../../../docs/public/BACKENDS.md)
for current selectors and coverage.

WOOF global is a moist hydrostatic global spectral model, shipped as its own
distribution and **experimental**. It runs on the woof engine and is a second
dycore beside woof's regional nonhydrostatic model, not a replacement for it:
an ordinary regional forecast reaches none of this code, and nothing about a
regional run changes because this package is installed.

It is reached through one console script:

```bash
woof global --help
```

This page is what ships, and where the edges are. The route from nothing
to rendered global maps is
[ARWEN_GLOBAL_QUICKSTART.md](ARWEN_GLOBAL_QUICKSTART.md); the model's
internals are [ARWEN_GLOBAL_FULL.md](ARWEN_GLOBAL_FULL.md).

## What experimental means here

Three things, and the first two are enforced rather than merely stated:

- **Every runnable configuration carries an acknowledgement.** A TOML
  without the exact line `acknowledgement = "research-only-arwen-global-v1"`
  is refused while it is being loaded. The native physics suite requires a
  second one of its own.
- **The configuration surface can move between releases.** Keys, defaults
  and checkpoint schemas are not held stable for this model the way the
  regional namelist surface is. A checkpoint archive is identity-locked to
  the physics-options era it was written under.
- **The output is not a supported product.** No forecast-skill claim is
  made anywhere in this document. What is measured is listed; what is not
  is listed beside it.

## The door

Three legs, and they are the shape of a run:

```bash
woof global run CONFIG.toml --outdir out/global

woof global assimilate CONFIG.toml CHECKPOINT.npz --obs OBS.csv --out analysis.npz

woof global export CONFIG.toml out/global/arwen_global_step*.npz \
  --outdir out/global-tapes --start-date 2026-08-30_18:00:00
```

The core a bare config runs is the semi-Lagrangian one, `[time] integrator =
"sl_si"` at a 300 s step at every truncation: a config that says nothing
about the core gets it, with the core's own drain and gather
(`arwen_global_gdas_t255_native_24h_bare` is that file in
the tree, and the quickstart runs it). `integrator = "imex_ssp3"` selects the
Eulerian core by name at the step its CFL gate admits, written out in every
shipped Eulerian config. The grade the default rests on is the equal-cost
scorecard of record (MEASURED 2026-09-06 on the RTX 5090: `sl_si` at T383 and
300 s against `imex_ssp3` at T255 and 120 s for the same card-time, 5 rows
better, 4 worse, 9 level on the ASOS stations at 18Z and 00Z and the IGRA2
soundings at 12Z and 00Z). The four rows it loses are 18Z sea-level pressure
(+0.076 hPa), 00Z 2 m dewpoint (+0.311 K), 12Z 500 hPa height (+0.26 m) and
00Z 850 hPa temperature (+0.072 K); the winning rows, the same-truncation
record and the wall per forecast day are in the full document's
semi-Lagrangian section.

Sizing is not a separate step. `run`, `go`, `cycle` and `assimilate` price
the card before they allocate anything and refuse a plan that will not fit,
and the same estimate is a document of its own for a program:

```bash
woof global run-plan PLAN.json --estimate
```

The engine's own preflight door does not price a global configuration on a
published engine. MEASURED on the Windows desktop 2026-09-10 against
`woof 2.7.0` and `woof 2.7.2`, and 2026-09-29 against `woof 2.8.0`:
`woof check` on a shipped experiment refuses by name with `unknown table(s) ['arwen_global', 'time', 'semilag', ...]`,
because the engine's sizing route recognises the regional tables and not
these. The
itemizing estimator is this package's (`woof.globe.sizing`) and the engine
calls it the day it routes a global TOML to it; until then the itemization
is what a refusal prints and what `--estimate` carries.

The rest of the research surface stays on the module door and is not part
of the product: pins, checkpoint and export inspection, Level-4 checkpoint
migration, the one-way regional parent bridge, and the native physics
qualification battery.

```bash
woof global --help
```

## Resolution rungs

The Gaussian grid a truncation implies is arithmetic, not a table:
`dealias_factor` (1.5 in every shipped configuration) sets
`nlon = ceil(2 * 1.5 * (T + 1))` and `nlat = ceil(1.5 * (T + 1))`.

| truncation | Gaussian grid (nlon x nlat) | spacing at the equator | truncation wavelength |
|---|---|---|---|
| T63 | 192 x 96 | 208.5 km | 630.4 km |
| T255 | 768 x 384 | 52.1 km | 156.7 km |
| T533 | 1602 x 801 | 25.0 km | 75.0 km |

Spacing is `2 pi a / nlon` at `a = 6371.22 km`. The truncation wavelength
is the isotropic convention `2 pi a / sqrt(n(n+1))` at `n = T`, which is
the convention the instrument in
[docs/arwen-global-effective-resolution.md](arwen-global-effective-resolution.md)
reads spectra under; the T63 and T533 entries are quoted from it, and the
T255 entry is the same formula at T = 255. Nothing on this page is quoted
in multiples of grid spacing. Where that document recomputes a superseded
reading it uses the plainer isotropic `2 pi a / T`, which is 0.8 % longer
at T63 and 0.09 % longer at T533; the one place this page quotes such a
figure says so.

T533 is the finest rung the model has run, and since 2026-09-06 the door
sizes each rung against the card in front of it rather than refusing what
does not fit resident: T383 runs on a 16 GB card and T533 on a 32 GB one,
with nothing set (see Sizing). Vertically, the default is a
surface-stretched hybrid coordinate at 40 levels to a 100 Pa lid, whose
lowest full level sits about 23 m above ground in the standard atmosphere,
because every surface scheme assumes tens of metres rather than hundreds.
A second layout is selectable as `[vertical] coordinate = "jet_refined"`
(48 levels): the same stack with its 120 to 400 hPa band re-laid as
twelve 23.4 hPa layers and a three-layer taper to the kept 56 hPa layer
below, every half level outside 118.73 to 504.19 hPa the default's own.
It exists because the upper-air scorecard's representation floor read the
default's 44 to 60 hPa layers across the jet costing 1.85 m/s of 250 hPa
vector wind and -0.53 m/s of speed before any forecast; on the candidate
the same floor reads 0.57 m/s and -0.16 m/s (`python -m
woof.globe.upper_air_scorecard --floor ANALYSIS --config CFG`
prices a level set before a card runs it). A config that names the
coordinate and no `nlev` gets 48; a count whose band would be coarser
than 25 hPa is refused naming 48. The receipt's `vertical` block reports
the thickest layer in the 120 to 400 hPa band for either layout.
Graded against the 40-level control over the same 24 h at T255 at the
control's own settings (GDAS 2026-09-01 00Z, dt 50, hourly checkpoints,
order-4 hyperdiffusion; 2026-09-05): the
250 hPa vector wind rmse at 24 h falls from 5.00 to 4.73 m/s and its speed
bias from -0.84 to -0.49, Z500 from 8.78 to 8.67 m (northern extratropics
8.77 to 8.51), the northern 250 km kinetic-energy ratio at 250 hPa rises
from 0.67 to 0.89, while the CONUS 2 m warm bias grows 0.06 K, the wall
17 percent and the allocator peak from 8.51 to 10.08 GiB. The 40-level
stack stays the default: the 35 km target at 48 levels prices at
18.51 GiB of pool at the default radiation chunk and 16.68 at a
5,000-column chunk, both above what a 16 GB card can hold RESIDENT, and the
24 h wind gain is a fifth of the floor's.  Since 2026-09-06 that shape
still runs on that card: the sizer streams grid space through it and parks
the persistent state in pinned host memory, with no flag set. The shipped door,
`arwen_global_gdas_t255_jet48_24h`, is the 40-level
verify config with its vertical table changed and nothing else, so it runs
that config's case at the tree's order-8 hyperdiffusion default; the
48-against-40 pair of those two files has not been graded.

The prognostic spectral state is relative vorticity, divergence, potential
temperature, log surface pressure, six water species and five number
moments, at every level, plus two-dimensional surface pressure. The number
moments are transported, diffused, repaired, checkpointed and restarted as
first-class fields rather than hidden inside a microphysics wrapper.

## Two cores, and the step each one takes

WOOF global has two time integrators and every truncation runs either.
`[time] integrator = "sl_si"`, the two-time-level semi-Lagrangian
semi-implicit core, is the default at T255, T383 and T533 and steps 300 s
when `dt_s` is omitted: a config that names no integrator runs it
(`config.SHIPPED_INTEGRATOR`), and the configs of record
(`configs/arwen_global_gdas_t{255,383,533}_native_24h.toml`) name
neither an integrator nor a step, so a bare run of one is the default with
nothing set. `"imex_ssp3"`, the Eulerian IMEX pair, is selectable by name
at every truncation at the step its own refusal sets (below), with its
ten-step identity pinned in the changelog. The default was chosen on
2026-09-06 on two measured things, the observation grade at equal cost and
the wall per forecast day, and this page carries both;
[ARWEN_GLOBAL_FULL.md](ARWEN_GLOBAL_FULL.md) carries the scheme. What a
config leaves out under `sl_si` reads that core's own defaults:
`[semi_implicit] off_centring_weight` 0.55 and `[time] maximum_lipschitz`
0.75 (the values every graded arm ran at), and the drain and gather the dry
ladder of 2026-09-07 chose for the 300 s step, order 16 at 720 s at the
truncation on the six-point quintic gather with three trajectory iterations
(`config.SEMILAG_DIFFUSION_ORDER`, `_EFOLD_S`,
`semilag.options.SemiLagrangianOptions`). The arms of the equal-cost grade
below ran the quasi-monotone cubic gather at order 8 and 2,160 s, the
core's values before the ladder was read; the shipped package's own rows
against the same Eulerian arms are in the second table.

**The grade at equal cost.** One forecast day from the GDAS 2026-09-01 00Z
analysis, the whole native suite, both cores on one tree, scored at the
ASOS stations at 18Z and 00Z and the IGRA2 radiosondes at 12Z and 00Z
with the GFS and IFS forecasts beside them, eighteen rows, each
difference a paired bootstrap over stations and sites with the project's
0.03 admission bar in the row's unit (`tools/semilag_grade_tables.py`).
MEASURED 2026-09-06, the scorecard of record:

| pair (candidate against reference) | wins | losses | ties | the losing rows | wall, same card |
|---|---|---|---|---|---|
| `sl_si` T255 at 300 s against `imex_ssp3` T255 at 120 s | 3 | 3 | 12 | 12Z Z500 +0.39 m, 00Z T850 +0.060 K, 00Z T500 +0.034 K | 279 s against 673 s, RTX 5070 Ti |
| `sl_si` T383 at 300 s against `imex_ssp3` T255 at 120 s (the equal-cost pair) | 5 | 4 | 9 | 18Z MSLP +0.076 hPa, 00Z Td2 +0.311 K, 12Z Z500 +0.26 m, 00Z T850 +0.072 K | 394 s on the RTX 5090 against 673 s on the RTX 5070 Ti |
| `sl_si` T383 at 300 s against `imex_ssp3` T383 at 80 s | 4 | 5 | 9 | 18Z MSLP +0.080 hPa, 00Z Td2 +0.259 K, 12Z Z500 +0.43 m, 12Z W250 +0.129 m/s, 00Z T850 +0.101 K | 394 s against 1,267 s (shared card), RTX 5090 |
| `imex_ssp3` T383 at 80 s against `imex_ssp3` T255 at 120 s (resolution alone) | 4 | 1 | 13 | 18Z Td2 +0.039 K | 1,267 s against 673 s, two cards |

The semi-Lagrangian core wins at the surface on temperature and wind and
loses on dewpoint and aloft, at every truncation; in the equal-cost pair
its 00Z 2 m dewpoint is 0.31 K worse than the T255 Eulerian arm's, ten
times the bar, while its 00Z 2 m temperature is 0.24 K better, eight times
the bar. The rule this line uses is that the core which, at the same
card-time, grades more rows better than worse on the observation
scorecard and runs the forecast day in less wall takes the default, with
every gate of record green and the rows it loses named; so `sl_si` is the
default at all three truncations, and a user who weighs the 00Z dewpoint
or the 850 hPa column above 2 m temperature and 10 m wind selects
`imex_ssp3` by name at its rule step. Where the choice is not clean it is
said: the dewpoint loss above is the largest single row in any pair, and
the T533 pair was not graded (the `sl_si` T533 day never had an empty
RTX 5090 in the window, and the door prices it at 24.45 GiB), so the T533
default follows the two graded truncations and this page says so.

**The same arms on the fixed core.** The lid fix of the semi-Lagrangian
core (the whole-theta gather, ARWEN_GLOBAL_FULL.md) landed while the arms
above ran, so they are the core before the fix; the shipped core is the
one after it, and the same arms re-run on it were scored in the same
table against the Eulerian core at its shipped 90 s. MEASURED 2026-09-07,
same case, same suite, same stations and sites:

| pair (candidate against reference) | wins | losses | ties | the losing rows | wall |
|---|---|---|---|---|---|
| `sl_si` T255 fixed against `sl_si` T255 before the fix | 4 | 1 | 13 | 18Z T2 +0.057 K (the wins: 00Z MSLP -0.034 hPa, 12Z Z500 -0.05 m, 12Z W250 -0.069 m/s, 00Z Z500 -0.21 m) | 293 s against 279 s, RTX 5070 Ti |
| `sl_si` T383 fixed against `sl_si` T383 before the fix | 5 | 2 | 11 | 18Z T2 +0.067 K, 00Z T2 +0.079 K (the wins: 00Z Td2 -0.057 K, 12Z W250 -0.093, 00Z W250 -0.169, 00Z W850 -0.110 m/s, 00Z Z500 -0.15 m) | 338 s against 394 s, RTX 5090 |
| `sl_si` T255 fixed at 300 s against `imex_ssp3` T255 at 90 s | 4 | 4 | 10 | 18Z MSLP +0.041 hPa, 12Z Z500 +0.45 m, 00Z T850 +0.054 K, 00Z T500 +0.035 K (the wins: 18Z T2 -0.056 K, 18Z wind -0.058 m/s, 00Z Td2 -0.121 K, 00Z Z500 -0.45 m) | 293 s against 1,050 s, RTX 5070 Ti |
| `sl_si` T383 fixed at 300 s against `imex_ssp3` T255 at 90 s (the equal-cost pair) | 2 | 4 | 12 | 18Z MSLP +0.057 hPa, 00Z Td2 +0.130 K, 12Z Z500 +0.34 m, 00Z T850 +0.055 K (the wins: 18Z wind -0.066 m/s, 00Z Z500 -0.55 m) | 338 s on the RTX 5090 against 1,050 s on the RTX 5070 Ti |
| `sl_si` T383 fixed at 300 s against `imex_ssp3` T383 at 80 s | 3 | 4 | 11 | 18Z MSLP +0.059 hPa, 00Z Td2 +0.203 K, 12Z Z500 +0.40 m, 00Z T850 +0.086 K | 338 s against 1,267 s, RTX 5090 |
| the shipped package (`sl_si` T255, six-point gather, order 16 at 720 s, the lid fix; the bare door, no flag) against `imex_ssp3` T255 at 90 s | 4 | 4 | 10 | 00Z T2 +0.097 K, 00Z wind +0.048 m/s, 12Z Z500 +0.31 m, 00Z T850 +0.050 K (the wins: 18Z T2 -0.050 K, 18Z Td2 -0.032 K, 18Z wind -0.055 m/s, 00Z Td2 -0.129 K) | 246 s against 1,050 s, RTX 5070 Ti |
| the shipped package against `imex_ssp3` T255 at 120 s | 3 | 2 | 13 | 12Z Z500 +0.20 m, 00Z T850 +0.052 K (the wins: 18Z T2 -0.035 K, 18Z wind -0.050 m/s, 00Z Z500 -0.25 m) | 246 s against 673 s |
| the shipped package against `sl_si` T255 fixed (the cubic gather at order 8 and 2,160 s the rows above ran) | 1 | 3 | 14 | 00Z Z500 +0.47 m, 12Z W850 +0.051 m/s, 00Z W850 +0.064 m/s (the win: 12Z Z500 -0.14 m); it keeps 2.05 times the cubic package's vorticity power at wavenumbers 181 to 230 | 246 s against 293 s, RTX 5070 Ti |

The rows the fix and the package add are small and aloft. The shipped
package's rows (MEASURED 2026-09-07, the bare door run on the RTX 5070 Ti,
246.0 s a forecast day, 8.701 GiB, every gate green, the tripwire silent,
scored with the same chain on the same analysis) are a wash against the
Eulerian core at either step, the same count the cubic package reads at
90 s, and give back 0.47 m of 00Z 500 hPa height and 0.05 to 0.06 m/s of
850 hPa wind head to head with the cubic package while keeping twice its
power at the scales the scorecard cannot read; a second run of the same
door, scored on its own station set, read 2 wins, 2 losses, 14 ties against the 120 s arm,
the 18Z 2 m temperature win at the interval's edge by 0.001 K. The T383
rows are the cubic package's: the shipped package was not run at T383. A
verdict count moves with the station set at one marginal row: the T255
fixed pair against the 90 s arm reads 4 wins, 5 losses, 9 ties on a
T255-only set of 1,577 stations (the 00Z 10 m wind row, +0.045 m/s with an
interval of [+0.003, +0.093], a loss there and a tie on the 1,519-station
set above); every other row keeps its verdict.

The fix moves the column the way its mechanism says it should: the wins
it adds are aloft (250 hPa wind at both hours, 850 hPa wind, 500 hPa
height at 00Z) and the dewpoint loss shrinks from 0.31 to 0.13 K, while
2 m temperature gives back 0.06 to 0.08 K of its win. Every gate stayed
green on every arm. The decision above stands on the scorecard of record
and the wall; these rows are the shipped core's, on the page so a user
reads the core they run and not the one that was graded first. One thing
the rows cannot see is stated here because it is measured: the bare run
of the T255 config of record on the fixed core (MEASURED 2026-09-07, the
archived GDAS 2026-08-30 18Z day, RTX 5090, 186.8 s for the forecast day,
every gate green), read level by level against the Eulerian 24 h
checkpoint of the same day, disagrees at the model lid by +16.4 K at the
top level and -12.0 K at the second, zonally uniform at every latitude,
within 0.1 K from the fourth level down, with the eddies at the lid agreeing
to 2 percent: half the +33.6 K the fix removed, of the same shape, inside
the three-level sponge of a column whose top is 1 hPa, forty levels above
the highest scorecard row. It is a defect against the Eulerian core's lid
and is carried as open (`tools/semilag_lid_structure.py` reads it from any
two checkpoints).

**By scale.** `tools/arwen_global_spectrum_bands.py`, 24 h, vorticity
power by total wavenumber in the candidate over the reference, read on the
model level nearest 500 hPa (level 19 of the 40-level stack, 475 hPa at the
reference surface pressure). Until 2026-09-07 the instrument never found
the level table in the receipt and silently read level 26, which is
793 hPa, so the numbers first published under this heading were the lower
troposphere's; these are the re-read of the same checkpoints (MEASURED
2026-09-07). The semi-Lagrangian core keeps 0.30 of the Eulerian arm's
power at wavenumbers 181 to 230 at T255 (0.34 after the fix, against the
90 s arm) and 0.22 at T383, 0.82 of the whole spectrum at T255 (0.79 after
the fix), while doubling the Eulerian step from 60 to 120 s moves that
band by 3 percent and the whole spectrum by 2 percent (the truncation tail
at 231 to 255 is the one band the Eulerian step moves: 0.64 at 90 s and
1.07 at 120 s against 60 s): the drain is the scheme's, not the step's.
The equal-cost pair carries 0.55 of the T255 Eulerian arm's power at 181
to 230 (0.62 after the fix), 0.84 of its power at 231 to 255 (2.0 after
the fix; the T255 arm's own truncation tail is drained by its
hyperdiffusion) and everything above wavenumber 255 that T255 does not
have. A band ratio on one model level moves with the level: the fixed
T255 core against the 90 s arm reads between 0.19 and 0.71 at 181 to 230
across levels 12 to 21 (100 to 590 hPa) and 0.50 at 240 hPa, and the
pressure-surface instrument (`tools/semilag_scale_diagnostics.py`, linear
in ln p to the 500 hPa surface) reads the T255 core against the 120 s arm
at 0.47 before the fix and 0.45 after it. Surface pressure agrees to
2 percent in every band between arms of one truncation; between the T383
and T255 arms the 231 to 255 band of surface pressure differs by 23
percent, the T255 arm's truncation tail.

**The Eulerian step is a rule, not a number.** The runner refuses an
Eulerian step whose spectral CFL `dt |V|max sqrt(N(N+1)) / a` exceeds
`[time] maximum_cfl` (0.75). The shipped step is the largest multiple of
5 s dividing the hour whose CFL on the strongest analysis day on disk
stays at or under 0.70 of that gate, so it survives a stronger day than
the one it was set on; an Eulerian config that omits `dt_s` gets it
(`config.default_eulerian_step_s`), and a day stronger than any on disk is
refused by name, the refusal and the receipt's `cfl` block both saying
which step that day would have admitted. MEASURED 2026-09-06
(`tools/arwen_global_cfl_sweep.py`): nine GDAS analyses, every cycle on
the nodes from 2026-08-30 18Z to 2026-09-02 00Z and the 2026-09-21 18Z
medicane start, each run for a whole forecast day at T255 on the Eulerian
core with the whole native suite, and each read again at T383 by a
one-step probe on the RTX 5090 (MEASURED 2026-09-07 01:10 to 01:32 UTC).
On every day the maximum is the analysis's own jet at step 1, and the flow
relaxes to 0.37 to 0.46 of it by hour 6; the strongest day is 2026-08-31
00Z at 136.9 m/s implied at T255 and 139.0 m/s at T383 (the probes read
1.015 times the T255 wind on that day, 1.019 across the nine; the table
below keeps the 139.6 the sweep tool derives, the more conservative of
the two). The T533 row reads the same analyses through the ratio a
two-step T533 probe measured on 2026-09-01 00Z (1.033).

| truncation | strongest day on disk, implied wind | shipped step of `imex_ssp3` | CFL on that day at it | step the gate itself admits | shipped before 2026-09-06 |
|---|---|---|---|---|---|
| T255 | 136.9 m/s (2026-08-31 00Z) | **90 s** | 0.494 (66 percent of the gate) | 120 s | 60 s (43 percent of the gate on the day of record) |
| T383 | 139.6 m/s (139.0 measured by the probe) | **60 s** | 0.504 (67 percent) | 80 s | 80 s in the graded arm |
| T533 | 141.3 m/s | **40 s** | 0.473 (63 percent) | 60 s | 25 s (set against a day-2 jet of an earlier tree) |

100 s at T255 fails the rule on 2026-08-31 00Z (0.549, which is 0.73 of
the gate); 45 s at T533 would need that day under 139.3 m/s and it is
not. 120 s at T255 and 80 s at T383 stay selectable
(`arwen_global_gdas_t255_native_imex_24h_dt120`,
`..._t383_native_imex_24h_dt80.toml`) as the steps the CFL gate itself
admits on the day of record, 0.86 and 0.88 of the gate, with no margin
for a stronger day. The Eulerian configs are
`..._t255_native_imex_24h.toml` (90 s), `..._t383_native_imex_24h.toml`
(60 s) and `..._t533_native_imex_24h.toml` (40 s); the semi-Lagrangian
core's spectral CFL is written into the same receipt block as a reading
(about 1.55 at 300 s on the day of record at T255, 2.36 at T383), not a
bound: its trajectories are gathered, not stepped, and the Lipschitz gate
refuses instead where the trajectory map would fold.

**Walls and memory.** MEASURED 2026-09-06 and 07, one forecast day,
in-process wall from the receipt; the card and its other tenants are
stated because a wall shared with another process is a capability row and
not a timing:

| truncation | integrator | dt | steps | wall | minutes a day | device peak | card and tenants |
|---|---|---|---|---|---|---|---|
| T255 | `sl_si` (the shipped package: six-point gather, order 16 at 720 s, the lid fix; the bare door, no flag) | 300 s | 288 | 245.8 and 246.0 s (two runs) | 4.10 | 8.701 GiB | RTX 5070 Ti, a 222 to 342 MiB tenant at the edges of the run (2026-09-07); 228.9 s on an RTX 5090 shared with four processes |
| T255 | `sl_si` (fixed core, the cubic gather at order 8 and 2,160 s the graded arms ran) | 300 s | 288 | 292.8 s | 4.88 | 8.701 GiB | RTX 5070 Ti, shared: four distinct processes seen on the card during the run |
| T255 | `sl_si` (before the fix) | 300 s | 288 | 279.0 s | 4.65 | 8.701 GiB | RTX 5070 Ti, shared (the clean timing of record is 256.7 s, 4.28 minutes) |
| T383 | `sl_si` (fixed core, the cubic gather) | 300 s | 288 | 337.9 s | 5.63 | 15.859 GiB | RTX 5090, shared: three distinct processes |
| T383 | `sl_si` (before the fix) | 300 s | 288 | 394.2 s | 6.57 | 15.858 GiB | RTX 5090, shared: six processes, five over 1 GiB |
| T255 | `imex_ssp3` | 90 s (the rule's step) | 960 | 1,050.2 s | 17.50 | 6.991 GiB | RTX 5070 Ti, shared: two distinct processes |
| T255 | `imex_ssp3` | 100 s | 864 | 723.9 s (715.4 to 728.9 over eight days) | 12.07 | 8.568 GiB | RTX 5070 Ti, ALONE on all eight days (the CFL sweep's own analyses): the one clean Eulerian timing taken |
| T255 | `imex_ssp3` | 60 s | 1,440 | 1,108.8 s | 18.48 | 8.568 GiB | RTX 5070 Ti, the arm of record (its clean timing, card verified empty, is 1,099.3 s) |
| T255 | `imex_ssp3` | 120 s | 720 | 673.4 s | 11.22 | 8.568 GiB | RTX 5070 Ti, shared |
| T383 | `imex_ssp3` | 80 s | 1,080 | 1,266.7 s | 21.11 | 15.560 GiB | RTX 5090, shared: three processes, all over 1 GiB |

Only the sweep row was taken alone; every other row shared its card and
is a capability reading. The timings of record for the two T255 cores at
60 and 300 s, taken on a card verified empty, are 18.322 and 4.278
minutes a day (ARWEN_GLOBAL_FULL.md). On the RTX 5090 the Eulerian core
costs 357 ms a step at T255 (MEASURED 2026-09-06), so the
rule's 960 steps are about 5.7 minutes a day plus initialisation, against
the semi-Lagrangian T383 arm's 5.6 minutes on a shared card: that is the
equal-cost pairing, the T383 semi-Lagrangian day for the price of the
T255 Eulerian one, and it holds on the 5090 within about 15 percent. On
the RTX 5070 Ti the semi-Lagrangian T383 arm does not fit (15.9 GiB of
16.3). The sizing door prices `sl_si` at 8.60, 15.01, 24.45 and 53.78 GiB
at T255, T383, T533 and T799; T383 measured 5.6 percent above it.

## Two drains at the truncation, and which one is the default

Both cores remove the energy that piles up at the truncation, and there are
two operators for it. `[diffusion] closure` picks one.

**`hyperdiffusion` is the default and nothing changes it.** An exact
exponential factor per total spherical degree at a fixed e-folding time at
the truncation, applied after the semi-implicit map. Under the
semi-Lagrangian core a config that names no drain gets order 16 at 720 s
(`config.SEMILAG_DIFFUSION_ORDER` and `_EFOLD_S`, set on the dry T255 ladder
of 2026-09-07); under the Eulerian core it gets order 8 at 36 min
(`config.DEFAULT_DIFFUSION_ORDER`, 2026-09-04). Every graded arm, every
record hash and every number on this page ran with it.

**`spectral_eddy_viscosity` is selectable and derived rather than tuned.**
`woof.globe.spectral.eddy_viscosity` replaces the chosen e-folding time
with the eddy viscosity the two-point closure theory of turbulence assigns
to a truncation inside a forward energy cascade (EDQNM; Kraichnan 1976,
Chollet and Lesieur 1981, Lesieur and Metais 1996, and on the sphere
Frederiksen and Davies 1997):

```text
nu(k | k_c) = nu_plus(k / k_c) * sqrt(E(k_c) / k_c)
nu_plus(x)  = plateau + cusp_amplitude * exp(-cusp_decay / x)
```

`E(k_c)` is the model's own kinetic-energy spectrum density at the cutoff,
read out of the coefficients every step and per level, as the mean over the
last `tail_degrees` degrees compensated to the truncation along the inertial
slope, so one noisy last degree does not set the drain. On the sphere the
wavenumber of degree n is `sqrt(n (n + 1)) / a`. The constants are the EDQNM
values for a Kolmogorov constant of 1.4: plateau 0.267, cusp 9.21 at decay
3.03, and the eddy Prandtl number 0.6 for potential temperature, vapour and
log surface pressure. The plateau is what the unresolved eddies exert on the
large scales, the cusp is the local transfer across the cutoff, and on a
k^-5/3 range the total drain below the cutoff equals the cascade rate the
spectrum implies, which is the property the constants were derived for and
the property the suite holds. Application is the same exact exponential
factor per degree and per level as the hyperdiffusion, unconditionally
stable, with degrees up to `preserve_degree` untouched. The constants are
options so a sweep can measure the sensitivity to them; at their defaults
nothing in it is tuned. The backscatter term of the same closure, the energy
it also predicts returning from below the cutoff, is not built.

**Record hashes do not move.** The closure's six configuration fields join a
configuration's identity only when the closure is selected, so every
hyperdiffusion config carries the identity it carried before this operator
existed, and the suite holds the two identities apart.
`arwen_global_gdas_t255_native_closure_24h` is the record T255 experiment
with its `[diffusion]` table changed and nothing else.

**It is ungraded on a card.** There is no card run of the closure arm and no
observation score for it: it ships selectable and unmeasured, and the drain
of record is the hyperdiffusion the graded arms ran with. A reader who
selects it is running an arm, not a configuration this model has been judged
on.

## Initialization from one GDAS analysis

A cold start reads a single whole-globe GDAS 0.25 degree pgrb2 f000 object
and decodes it through the Rust mapped-source engine
(`woof.mapped_source.decode_mapped_source`), the same engine the regional
route uses. The source is named as table data, not as a code path:

```toml
[initial]
mode = "analysis"
analysis_grib = "data/gdas-analysis/gdas.t18z.pgrb2.0p25.f000"
analysis_mapping = "gdas-global"
```

A bare id is asked of two authority tables in a fixed order, and it must
resolve to exactly one document in the table that answers, so the spelling
works from an installed wheel as well as from a checkout. Adding an analysis
source here is a mapping document.

**Which table answers, and why there are two.** A source mapping is metadata
the engine owns, which is what keeps adding a source declarative rather than
a code path, so the ENGINE'S table (`woof/authorities`) is asked first for
every spec, every time. No published engine carries any of the six mappings
this model names, and a package whose every shipped GDAS experiment refuses
at its first door is not shipped, so the six travel inside this package
(`woof/globe/data/authorities`) and answer only what the engine's table
does not have. The engine's copy wins the moment it publishes one.
`woof global doctor` prints which of the two answered each row, with the
SHA-256 of the file that answered it (measured on the Windows desktop
2026-09-10 against `woof 2.7.0`, `2.7.1` and `2.7.2`, there 2026-09-12
against `woof 2.7.3`, and there 2026-09-29 against `woof 2.8.0`: 6 of 6
carried on all five), and
`woof global sources` carries the same fact per row as `mapping.origin`. A
spec both tables carry with DIFFERENT bytes is refused by name with both
digests rather than chosen between: the engine's row having moved past the
one this model was graded with is exactly the case where picking a winner
silently would grade a run against a source table nobody looked at.

Both spellings the tables use reach the same file. A row is asked for as the
file whose name ends at the id (`<id>.mapping.json`) and then as the family
glob (`rw-wps-<id>-*.mapping.json`), because the six documents are named in
both shapes and asking only one of the two questions reports a gap on a
table that carries the row. A bare name is a key into the tables and never a
file in the working directory; a spec that names a path (a directory part,
an absolute path, or the explicit `./name`) opens that file, and the receipt
records it resolved rather than as typed.

**Where the decode stages its frames.** The decode of a whole-globe object
stages several GB of float64 frames, and this model names the directory that
holds them. A published engine takes the same placement from the
`WOOF_COMPOSE_SCRATCH` environment variable instead of as an argument, so
`woof.globe.mapped_source_compat` translates between the two spellings:
the read of the caller's value, the placement and the restore are one
critical section held across the decode, so two decodes in one process take
turns rather than staging into each other, and each decode's receipt names
which of the two placed the directory it used. A nested decode names this
package's own enclosing placement rather than crediting a caller who set
nothing. A value the caller set is left alone and reported as theirs.

Two refusals belong to this path and both stay:

- A **cropped** analysis cannot initialize a global model. The initializer
  measures the longitude ring and refuses one that does not close. The
  default GDAS transport is an area crop, so a global cold start asks for
  `--mode full-file` deliberately.
- A **masked or partial-coverage** field is refused by name after
  regridding rather than filled, because a non-finite cell in the initial
  state is a forecast that fails somewhere later with no trace of why.

Model terrain is the spectrally resolved version of the analysis orography
and surface pressure moves hypsometrically onto it, so the lowest layers
stay hydrostatically consistent with the smoothed mountains.

A second analysis source is another mapping row. The ECMWF IFS open-data
step-0 object (0.25 degree, 14 pressure levels from 1000 to 10 hPa,
surface pressure, skin temperature, land-sea mask, 2 m and 10 m fields,
four soil layers) initializes the same way through
`analysis_mapping = "ecmwf-open-data-global-forecast"`. That product
carries no sea-ice concentration and no snow depth (it publishes a sea-ice
thickness without a concentration, and snow as water equivalent with a
density), so the cold start takes those surface groups from a second
analysis of the same hour, named beside the first:

```toml
[initial]
mode = "analysis"
analysis_grib = "data/ifs-open/20260901000000-0h-oper-fc.grib2"
analysis_mapping = "ecmwf-open-data-global-forecast"
analysis_fill_grib = "data/gdas-analysis/gdas.t00z.pgrb2.0p25.f000"
analysis_fill_mapping = "gdas-global"
```

The fill is by surface group (soil temperature with soil moisture, snow
water with snow depth, sea-ice concentration with thickness): a group the
primary product lacks any field of is read whole from the second product,
so a pair the seeding checks against itself (the snow unit by density) is
never half from each. The atmosphere is never filled. A fill of another
hour, a field neither product carries, or an atmospheric field missing
from the primary refuses by name, and the run receipt names every field's
product (`initial.provenance.field_sources`, `initial.provenance.fill`).
The two keys join the config identity only when set, so a GDAS cold start
keeps its hash. Where the two products' land masks disagree, a filled
plane's bitmap is grown over the primary's coast from its finite
neighbours (eight cells, counted in the receipt) and primary land still
inside the second product's water starts snow-free, counted and located;
a land column whose analysed soil water is zero takes WRF's glacial
convention (saturated) on the land-ice class and its soil class's dry
limit elsewhere, because Noah's frozen-soil relation has no value at
zero water. The IFS route is selectable; the default stays GDAS.

## Two physics suites

### The reference suite

A compact executable suite that the model always has: gray shortwave and
longwave radiation, bulk surface exchange, finite grid-resident surface
water and heat reservoirs, implicit vertical diffusion, Betts-Miller deep
convection, saturation adjustment with latent heating, warm-rain
autoconversion and accretion, rain evaporation, freezing, ice-to-snow and
snow-to-graupel conversion, finite-speed fallout, and accumulated surface
precipitation by category. It closes local atmosphere-plus-surface water
and records the global water repair it applied as a run tracker.

It is a model-integration and conservation reference. It is not advertised
as WRF physics parity, and its long-lead envelope is stated below.

### The native WOOF CUDA suite

`physics.mode = "arwen-native"` runs existing WOOF CUDA column physics on
the Gaussian grid, through the launchers the regional model uses rather
than copied equations, in this fixed order:

```text
RTE+RRTMGP -> MM5 SFCLAY -> Noah LSM -> YSU PBL -> Grell-Freitas cumulus -> Morrison two-moment
```

Grell-Freitas is default on; `cumulus = "none"` removes it. The adapter
owns the top-to-surface reversal and the contiguous FP32 packing at the
boundary, and that conversion is part of its arithmetic identity. A native
half step is all-or-nothing: the caller's exchange cannot be partially
mutated if a later scheme fails.

Admission is fail-closed and by evidence, not by name. Every adapter
declares its scheme identity, backend and precision, required fields,
pressure ordering, restart and budget contracts, and two hashes; a TOML
naming an unregistered adapter is refused before a transform is built.
There is no silent fallback to the reference suite. What the built-in
adapter's evidence currently covers is in the envelope below.

### Where that physics lives, and what is still the engine's

The native suite's schemes are inside this package, at
`woof.globe.core`: the radiation, cumulus, surface-layer, boundary-layer,
land-surface and microphysics modules, the land-use rulebook, the float64
mirror the scorecards grade against, the CUDA loader, twelve kernels and
three headers, and the four Noah and land-use tables.

They are carried rather than imported because the published engine's copy of
the schemes this model executes is a different scheme. Measured on the
Windows desktop 2026-09-10 against published woof 2.7.0, line endings
normalised: nine of the eleven carried modules differ, and its radiation
constructor has no effective-size bounding, its Grell-Freitas takes no
arguments, its New Tiedtke no column chunk, its surface layer no vegetation
fraction, its boundary layer no free-atmosphere mixing-length flag, and its
land-use rulebook does not assign the ice soil category. Eight of the fifteen
carried kernel files differ too: the seven translation units `gf.cu`,
`ntiedtke.cu`, `sfclay.cu`, `ysu.cu`, `noah.cu`, `morrison.cu` and
`rrtmgp_rte.cu`, and the header `glibc_flt32.cuh`. A forecast against those
raises inside the physics minutes into a run, or, worse, if the arguments
were adapted away, integrates under a scheme no receipt in the run names.

The other two carried modules are `noah` and `morrison`, and they match the
published engine's copies apart from the carve's import rewrites. They travel
anyway because their kernels are two of the eight that differ and the loader
binds its own directory, so a module left on the engine would compile the
engine's `.cu`. The nine-file figure that used to stand here counted the
whole engine kernel directory rather than the fifteen files this package
carries.

Everything else the package reaches is still the engine's, and the files it
reaches are pinned by path, size and SHA-256 in
`woof/globe/data/engine-seam.json`. `woof global doctor` hashes the
installed engine's copies and prints an `engine seam` section:

```text
engine seam
-----------
  ok   seam                       47/47 files proven
                                  pinned against woof 2.8.0
                                  the scope is the DIRECT engine imports of
                                  the carried physics, plus the assimilation's
                                  filter, the local-GPU switch and two files
                                  those reach; it is not the import closure,
                                  which nothing on the run path enters
                                  the engine's DOORS are not pinned here:
                                  they are measured by symbol in the boundary
                                  section above
```

The scope is stated in the row because the row is a coverage claim, and it is
stated as measured. The 47 rows are the 43 engine modules the carried physics
imports DIRECTLY, plus the LETKF the assimilation runs on the engine, the
local-GPU switch, the eigensolver kernel and the scheme-limits table those two
reach. It is not the import closure: following every import from the 46
pinned then reached 109 modules, and the 63 that were not pinned are entered
only through runtimes this package's door cannot select (measured on the
Windows desktop 2026-09-10 against `woof 2.7.0`). The engine's front doors are deliberately
not pinned by bytes: what this package depends on there is an API, measured by
symbol in `tools/measure_boundary.py` and by signature in
`tools/measure_engine_signatures.py`, and pinning them would print a moved
row on every engine release while saying nothing about the physics.

A file whose bytes moved is named, with what this package reaches in it, and
the summary row above it turns to `note` as well, so a reader scanning
verdict tokens is not shown green over a moved file. It
is a warning and not a refusal: the dependency ceiling (`woof<2.9`) is the
refusal, and "these bytes are not the ones I measured" does not name a
breakage on its own. The pin that matters most is `woof/core/constants.py`,
which supplies the `#define` values to every carried kernel's preamble and so
reaches every kernel's assembled source.

Every run receipt records which of these modules actually integrated, its
origin and the SHA-256 of the file that was imported, under
`physics_modules`, with one digest over the kernel directory the loader
bound, and the seam verdict beside it under `engine_seam`: the version the
pins were taken against, the version that resolved, the proven count and any
file whose bytes moved. The module hashes cannot see the staying half:
`woof/core/constants.py` reaches every carried kernel's assembled source, so
a run against an engine that moved it would otherwise write a receipt
identical to yesterday's while every compiled kernel had changed. Two copies of several of these modules exist in one process -- the
carried ones this model calls and the engine's, which the engine's own
regional drivers import -- so the engine's version number does not answer
"whose physics produced this", and the receipt does.

`tools/pin_engine_seam.py --check` re-measures the manifest against whatever
engine is installed, and writes nothing.

## Assimilation

`woof global assimilate` folds minutes-fresh point observations into a
checkpoint and writes a restartable analysis beside an
o-minus-b/o-minus-a report. It is a data-density-normalised successive
correction against one deterministic background, with the single-report
limit of the optimal-interpolation gain.

Four things about it are worth knowing before reading a report:

- **Reports are compared at their own height.** Surface pressure reduced
  to the station, temperature to 2 m, wind to the 10 m anemometer, by the
  same similarity diagnostic that writes U10/V10 to the render tapes.
- **Wind increments are applied as their rotational part only.** Spreading
  two scalars puts 40 to 51 percent of the increment's kinetic energy into
  divergence, which is gravity-wave energy the dycore keeps and no report
  measured. `--wind-balance unconstrained` applies the predecessor for
  measurement.
- **The gate of record is cross-validation**, not the assimilated rows. A
  seeded tenth of the reports for each variable with at least 50 accepted
  reports is withheld, and O-A rms must fall below O-B rms on those. The
  older assimilated-row gate passed for any positive gain, including
  reports made of pure noise.
- **A cycled background carries its own assimilation chain.** Every
  assimilated report's hashed identity is written into the analysis
  checkpoint, so a later cycle refuses a report the chain already holds and
  refuses outright when nothing new survives.
- **Dewpoint reports can move the vapor (`--moisture-update on`, off by
  default).** The v1 door left water untouched.
  A 2 m dewpoint report is compared against the dewpoint of the model's
  lowest-level vapor at the station pressure; the innovation is spread by
  the same successive correction with its own vertical localization
  (`exp(-z / 1500 m)` above the surface), and at every level the
  background dewpoint moves by the spread increment and the vapor is
  re-derived through Bolton's relation, so the specific-humidity increment
  is the dewpoint change through the local Clausius-Clapeyron slope. It is
  capped at saturation against the analysed temperature, floored at zero,
  passed through the truncation and closed by the model's positivity
  repair; the dewpoint has its own withheld-tenth gate. Graded whole on
  the 24 h hourly T255 cycle against the CONUS stations (the same cycle
  with and without it), the update took the 18 h dewpoint from -2.76 /
  5.68 to -0.97 / 3.59 K bias / rmse and the sea-level pressure from
  2.70 to 4.19 hPa rmse, so it is selectable and not the default. Sounding humidity
  joins through the `igra2-levels-csv` table entry
  (`tools/arwen_global_igra2_levels_csv.py` writes it from the IGRA2
  archive): a level row carries its own pressure and is compared against
  the vapor profile interpolated to it. The IEM ASOS record (`rw_asos
  fetch`, the file the observation scorecard scores against) decodes
  through its own table entry, dewpoint included; with the update off
  the dewpoint rows are still read and their fit reported, never
  applied.
- **A cycle is `--until-s`.** `woof global run CONFIG --until-s 3600`
  ends a segment at that model time with the config identity unchanged,
  and the next segment restarts from the checkpoint it ends on; a 48 h
  config therefore carries a 24 h hourly cycle and the forecast from its
  final analysis on one checkpoint lineage.

## Observation streams

The observations the assimilation reads come from Rust front doors, each
writing one neutral table (`gpuwm-obs.table.v2`) that every door of the
tree reads unchanged: worldwide METAR through the IEM archive (266
networks; `rw_asos networks`, `fetch --networks`, `table`) and the
Aviation Weather Center cache (`rw_asos awc`), IGRA2 radiosondes with
every level at its own time (`rw_igra2`), GOES-18 and GOES-19 derived
motion winds (`rw_amv`), NDBC buoys and coastal stations (`rw_ndbc`),
GNSS radio-occultation refractivity (`rw_gnssro`; the public bucket ends
at 2025-07-29), and the WMO Information System 2 notification stream
(`rw_wis2`; messages and BUFR payloads archived and measured, not yet
decoded).  A row carries what it measures (a station pressure recovered
from the altimeter setting is never a reduced sea-level pressure), when it
was measured, when its source published it and when this system first
held it.

```
woof obs streams streams                       every stream, its door, latency class
woof obs streams fetch --stream iem-metar --start 2026-09-01T00:00:00Z     --end 2026-09-01T06:00:00Z --out obs        fetch and decode through the Rust door
woof obs streams hours --tables obs/*/*.csv --start 2026-09-01T00:00:00Z     --end 2026-09-01T06:00:00Z --out hours      one table per analysis instant
woof obs streams anchors                       the analyses usable as the background constraint
```

An hourly cycle is a window, not an instant: `hours` writes the reports of
`(t - 30 min, t + 30 min]` per instant and never moves a report's time.
`--cutoff` applies an information cutoff on the receipt time (a report
received after the instant is dropped and counted; one whose arrival was
never recorded is kept and counted latency unverified).  Streams carry a
measured latency class: METAR, buoys and motion vectors are minutes late
and feed the hourly analysis; the IGRA2 archive rebuilds once a day and
feeds the delayed replay; the occultation bucket is retrospective; MADIS
aircraft and the CDAAC feed need accounts and are reported, never fetched.
The external analysis (GDAS, IFS open data; about 7 hours behind their
cycle) is an anchor source for a weak low-pass background constraint, not
a stream of pseudo-observations.  The interface notes and measurements are
in `docs/arwen-global-observations.md`.

## Cycling

`woof global cycle` runs the forecast and the assimilation in one process:

```
woof global cycle CONFIG.toml --obs OBS.csv --outdir cycle \
    --cycles 24 --interval-s 3600 --start-utc 2026-08-31T00:00:00Z
```

The model integrates to each analysis instant (the run's start plus a
whole number of intervals), the resident state is analysed against the
observation sources at that instant with the same arithmetic
`woof global assimilate` applies to a checkpoint, the analysed state is
written as that hour's `arwen_global_analysis_step*.npz` beside its
`assimilation-report-step*.json`, and the integration continues from it
under a new conservation epoch, exactly as a restart from an assimilated
checkpoint opens one. After the last analysis the forecast runs on to
`--until-s` (the config's `duration_s` by default) writing the ordinary
hourly checkpoints, so the analysis of the last cycle is the forecast's
initial condition and the whole lineage shares one config hash.

A cycle whose gate of record fails on some variables withdraws those
variables' reports and analyses the hour again with the rest, which have
to pass on their own (`--partial-analyses`, default on: under hourly
cycling a variable's withheld fit reaches the report noise within a few
hours and ties the background, and a whole-hour carry would throw away
the other variables' improvement for a tie that is not over-fitting); a
cycle that fails on every gated variable, or fails again on the second
pass, carries the background forward unchanged, writes it as that hour's
checkpoint and records the failure in the report and the receipt; a cycle
offered nothing new (every report already in the chain) refuses with a
failure receipt, because that is a defect of the offer rather than
weather. The receipt's `cycle` record
carries every analysis with its withheld verdicts and the wall of every
phase (forecast steps, the background's identity, the analysis phases,
the checkpoint publish), and the seconds per cycled model hour.

The per-segment chain the door replaces (`run --until-s`, `assimilate`,
`run --restart` once per hour, each a process) paid the model build, the
statics, a checkpoint write and a checkpoint read every hour and rebuilt
the analysis operators for every variable; the door pays the build once,
keeps the state on the device between the two halves and names the
background by the identity its checkpoint would carry instead of writing
and reading it. Its analysis checkpoints and the forecast after them are
bit for bit the chain's (`tests/test_arwen_global_cycle.py` holds the two
to the same bytes).

## The data-assimilation door

`woof global da` is the door a fresh global analysis is made and started
from, built on the cycle above: `fresh` fetches the newest GDAS analysis
(or reads one on disk), derives the run configuration, builds the
ensemble manifest, cycles hourly through observation streams fetched and
recorded per window (`iem-asos` through the Rust `rw_asos`;
`local-tables` for tables already decoded) up to the newest observation
hour before a declared information cutoff, and hands back the analysis
checkpoint with the forecast command; `init`, `cycle`, `analyze` and
`forecast` are its legs one by one. Two filters form the increment: the
deterministic door above (`--filter successive-correction`, the default)
and the dual-resolution ensemble filter (`--filter letkf`: resident
members at their own truncation, the control analysed from its own
innovations through the ensemble covariance with the increment tapered
per total degree, the members recentred on it, RTPS the one inflation by
default). Reports are compared at their own bin instants along the
trajectory, the increment is inserted at once or through the window's
re-integration, an external analysis may hold the largest scales as a
weak low-pass constraint, and every analysis report carries the DA
scorecard as O-B and O-A distributions per stream, variable and region
with four separated assessments (engineering validity the only hard
gate), the lineage (the checkpoint chain names the filter and the streams
of every link) and the wall budget of the cycle. The page a user follows
is `ARWEN_GLOBAL_DA.md`.

### The ensemble filter

Beside the v1 door sits the dual-resolution ensemble filter,
`woof.globe.da`: N members (32 by default) at T127 resident on one
card in one process, stepping sequentially through one model, analysed by
a point-observation LETKF on the sphere (Gaspari-Cohn in kilometres and in
ln p, RTPS) that also forms the T255 control's own analysis from the
control's innovation through the ensemble covariance, the increment
tapered by degree and embedded in the control's triangle; the members are
recentred on the control analysis, every report is compared with the
state at its own time bin, and the receipt carries O-B and O-A
distributions for the ensemble mean and the control with Desroziers
ratios and four assessments (engineering validity the only hard gate).
Its contract, every decision taken while building it and its measurements
are in `docs/arwen-global-ensemble-da.md`; `python -m
woof.globe.da.osse` runs the dual-resolution twin that gates it
and its sweeps, `python -m woof.globe.da.measure` reads a card's
bytes per member and wall per member-step. The `woof global da` door that
drives it is documented where it lands.

The microwave leg measures ATMS brightness temperatures as an observation
for that filter: `woof global microwave` (also `woof
global microwave`) fetches a day of NOAA-20 or NOAA-21 Sensor Data Records
from the public JPSS bucket with a manifest, decodes and thins them in Rust
(`rw_atms`), scores a clear-sky over-ocean forward operator against GDAS
analysis columns, and writes an operator entry for the channels inside the
1 K bar (channels 4 to 14 on 2026-09-01; channel 15 refused with its term
named). The entry is registered for the filter, not in its default stream
set. Notes: `docs/arwen-global-microwave-operator.md`.

## Render tapes, and the standard render path

`woof global export` writes one `wrfout_d01_<valid time>` NetCDF tape per
checkpoint on a regular lat/lon grid (360 x 720 by default; `--bbox` crops
to a window). The tapes carry `MAP_PROJ = 6`, so weather fields go out
through the ordinary render door and land in the ordinary layout:

```bash
woof render out/global-tapes/wrfout_d01_* \
  --products 500mb_height_winds,mslp_10m_winds,2m_temperature \
  --out out/global-png
```

Nothing about that step is global-specific, and nothing about it is a
second renderer. The layout is the shipped one,
`<out>/<run folder>/<domain>/<product>/<valid-day>/`.

## Checkpoints and restart

A checkpoint binds the resolved config hash, the model pins, the step and
time, every spectral array, every grid-resident surface and soil array,
shape/dtype/SHA-256 per array, the whole-run maxima of every repair
mechanism, and a metadata self-hash. Persistent native scheme state lives
under its own namespace, so a resumed native run does not cold-start Noah,
radiation or Morrison at a checkpoint boundary.

`woof global run --restart CHECKPOINT` continues from it and refuses a
checkpoint whose config hash is not this configuration's. Restart is
bit-exact at a midpoint: that equality, over the complete checkpoint array
inventory including run trackers, is what the target-device qualification
battery measures.

A terminal run receipt carries the transform controls, start and final
measurements, all gates, inherited trackers, checkpoint identities,
physics identity and a self-hash. A numerical exception after output setup
writes a self-hashed `status = "error"` receipt and then re-raises.

## Sizing

`woof global run`, `woof global go`, `woof global cycle` and
`woof global assimilate` price a global configuration itemized against free
VRAM and refuse before they allocate anything;
`woof global run-plan PLAN.json --estimate` carries the same figures as a
`gpuwm.run-plan.estimate.v1` document, which is the way to read them without
starting a run. On a published 2.7.0 and 2.8.0 the engine's `woof check`
refuses a global config outright (measured, desktop, 2026-09-10 and
2026-09-29): its sizing route knows the regional tables only. `woof.globe.sizing.global_check_main` is
the itemizing entry the engine's door would call. No command reaches it
today, so `woof global doctor` prints the engine symbol it also needs,
`preflight.measured_free_vram_bytes`, as an optional note that does not move
the exit code; the row becomes a gap the day a door is wired to it.

### What the door decides, with nothing set

A bare run names no band count and no host tier, and the sizer chooses
both. It reads the card in a short-lived subprocess before a byte is
allocated, walks the band ladder from one upward, and at each count parks
the minimum number of slices of the persistent grid state in pinned host
memory, coldest first: the native physics namespace, then the surface
reservoirs, then the grid tracers. The first combination whose card
requirement fits is taken, so the cheaper relief is bought first, and the
cost order is measured on both sides: banding costs 18 to 48 percent of a
T255 step and 26 percent of a T533 step, where the tier's exposed
transfer time reads 5.9 percent of a T383 step (RTX 5090 and RTX 5070 Ti,
2026-09-06).

One pass, and it is monotone: the plan is the first candidate in that
cost-ordered enumeration whose card figure fits what the card has free,
so freeing memory can only ever buy a cheaper plan. Preferring the
cheapest plan that fits three quarters of free, and falling back to the
whole card when none does, is not monotone once the host tier is a
candidate -- just below a threshold the fallback hands back a light plan
and just above it the budget pass hands back a heavier one, so a user who
frees memory watches the model start spilling -- and swept over T255,
T383 and T533 from two to thirty-four gigabytes free that rule flipped
the parked slice count three times on one truncation. The quarter-free
share survives as a figure in the receipt and a clause at the door,
saying whether the plan leaves a second process anywhere to go. A genuine
out-of-memory prediction is the only refusal below the fitted ceiling.

The receipt's `sizer` block records all of it: the free bytes read, the
budget weighed against, the fragmentation and out-of-pool terms charged,
the predicted live peak and card requirement, what was chosen, who chose
it, and one sentence saying why. `latitude_bands` and `host_spill` in
`[memory]`, and `--latitude-bands` / `--host-spill` on all three doors,
override the choice; a named value is priced rather than searched.

### What a card has to hold

The device figure is a model fitted to allocator-measured peaks, and it
says what it predicts: the maximum live bytes of the CuPy memory pool over
a whole run of the native suite, as the runner's allocator hook reads them
and the receipt records them (`device_memory.peak_used_bytes`).

**A card must hold more than the pool does**, and the door charges the
difference as two measured terms:

```
card required  =  pool live peak  x  fragmentation  +  out-of-pool
```

*Fragmentation* is the pool holding blocks it is not lending. It is
measured per allocation pattern, because the pattern is what moves it
(2026-09-06, twelve runs at T63, T255, T383 and T533 on both cards): at
the default radiation chunk a resident run holds x1.0871 to x1.1668 of
its live bytes, a banded run x1.0695 to x1.1972, a resident run with the
host tier x1.3360, and a banded run with the tier x1.0296 to x1.0748.
**The radiation chunk is part of the pattern**, and two independent pairs
put it there: the same T383 resident run reads x1.2020 at a 5,000-column
chunk against x1.1011 at the default, and T383 at eight bands with the
tier reads x1.1626 against x1.0296, because a smaller chunk takes about
1.5 GiB off the live peak while the pool still grows for the same
allocations. Each class is charged the largest of its own measured rows;
a class with no row falls back to the largest of all of them and the
receipt says it did.

*Out-of-pool* is everything on the card the pool never sees: the CUDA
primary context (measured 0.486 GiB on the RTX 5090), the module images,
and the cuBLAS and cuFFT workspaces. Ten measurements spanning a
sixteenfold grid move by a third of a gigabyte, so it is charged flat at
the largest of them, 0.7437 GiB, rather than scaled through points that do
not trend.

Charging neither is what the door used to do, and the concrete breakage is
on record: at T533 L40 on a 32 GiB RTX 5090 it printed 24.45 GiB against
30.90 GiB free, admitted the run, and the run reached a 27.57 GiB live
peak before the allocator killed it. The check on the corrected form: for
a MEASURED T383 live peak of 16,706,958,848 B the model asks 20.35 GiB of
card, and `nvidia-smi` read 18,300 MiB against that process for the life
of the run, so the door stands above the card reading rather than under
it.

A slice in the pinned host tier is credited with more than its own bytes,
because the device copy the physics suite makes of the scheme namespace
stops existing beside the original: measured x1.675 and x1.641 at T255 and
x1.564 across a T383 pair, and the smallest of the three is what the sizer
credits.

### The calibration, and where it stops

The calibration table in `woof.globe.sizing.DEVICE_PEAK_CALIBRATION`
lists every measured run: T63, T127 and T255 at 40 and 20 levels and at
radiation chunks of 12,500 and 5,000 columns, T255 at 48 levels, a T63 day,
the T255 24 h control, and **T383 at 40 levels at both radiation chunks**,
fourteen runs over twelve distinct shapes, every one on the RTX 5090,
between 2026-09-05 and 2026-09-06. The four fitted coefficients (the step's working set per grid
point, the radiation chunk's per-column workspace, the cumulus chunk's
per-column workspace, which stops growing at the 131,072-column chunk, and
a fixed remainder) are solved from that table at import, and the estimator
carries each point's measured and modelled figure beside the prediction, in
sample (every row within 6.9 percent) and held out (every shape priced by a
refit without it, within 8.9 percent). The `--estimate` document's `basis`
states both residuals wherever it reports a peak.

The probe behind the table is ten steps with the radiation called at step 0
and again at step 6, the ledger every five steps and checkpoints at steps
0, 5 and 10 (`sizing.PROBE_RECIPE`): it reads the T255 control day's peak to
0.0003 percent, where the same ten steps with one radiation call read 18.7
percent under it (`sizing.PROBE_CALIBRATION` holds the pairs). The second
radiation call is what makes it the probe of record: it lands on a full
step working set, which is where a real forecast's peak is.

The door's margin is not a typed number. It is one plus the largest amount
the model has ever under-read a measured run by, in sample or held out,
solved at import; with the two T383 rows it reads 1.0770, and adding a measured
row moves it. Only under-reads count, because over-predicting costs a
parked slice and under-predicting costs the run.

**Above the largest measured truncation the run doors refuse.** The grid
scaling beyond it is the structure's arithmetic and not a measurement, and
it has only ever erred in the direction that admits a run the card cannot
hold. The refusal names the ceiling, the run that died, and the one thing
that lifts it: a ten-step probe of the shape at the recipe above, added to
`DEVICE_PEAK_CALIBRATION` as a row. `run-plan --estimate` still prices the
shape and carries the whole itemization; what is refused is starting a
forecast on a number nothing has weighed.

The term that decides everything below that is the Legendre tables: **cubic
in truncation, and carrying no vertical levels at all**, so trimming
`[vertical]` cannot make a tables-bound configuration fit. They are packed
by order band with a lazy derivative and built one band at a time, and the
recurrence runs in float64 on the host whatever the device precision is, so
the host construction peak does not halve at `precision = "float32"` -- but
it is now small: 0.21 GiB at T533, against the 9.4 GiB the dense float64
squares needed before the packed and streamed build landed.

A configuration outside the measured domain in a way other than truncation
(another physics mode, float64, another dealias factor, a level count other
than 20, 40 or 48) is priced by the same structure and the door labels the
figure extrapolated.

## The envelope, stated plainly

### The reference suite is good for about three days

Reference-suite integrations grow their upper-level winds without bound by
day 3 to 4, until the run is refused by its own spectral CFL gate. This is
systematic rather than case-specific: it reproduced on separate
initializations and at several time steps, so it is not a time-step
choice.

The named mechanism, and it is the working explanation rather than a
closed measurement: gray radiation carries no ozone shortwave heating, so
the polar-night upper stratosphere cools without a floor, and the
resulting temperature gradient grows the jet through thermal-wind balance.
The dycore's top absorber tames waves, not the mean jet, so it does not
reach this.

What the suite carries against it is a one-sided Held-Suarez-style
relaxation above 5,000 Pa toward 195 K, default on
(`stratospheric_floor_k <= 0` disables it, for arms measuring the
unfloored sag). That
scaffold stops the cold-pocket deaths it was sized for -- the first T533
run crossed the 140 K research floor at hour 3 without it -- but it does
not reach the 50 to 200 hPa cooling that drives the thermal wind. **Treat
a reference-suite forecast beyond about three days as outside the
envelope.** The native suite is the arm under test against this, because
RRTMGP carries an ozone climatology; it carries the same floor, added
after a 384 h native arm died at hour 83.4 by a slow polar-top sag with
its winds healthy.

### The native suite is experimental, and its evidence has a gap

The target-device qualification battery **passed on an RTX 5090** on
2026-09-01 for the five-scheme stack (RRTMGP, SFCLAY, Noah, YSU,
Morrison): every gate green, terminal checkpoints bit-exact including run
trackers across a midpoint restart, device identity bound, evidence
`self_sha256 c682dbcc73ea544af04d4892cbeea91af08dc8f7f35f1161fe2485f9c53a04fe`.
Six admission defects were found and fixed by that battery on the way in.

**That evidence does not cover what the adapter runs today.** Grell-Freitas
cumulus joined the suite as a sixth component after the battery, because
at 25 to 52 km spacing the stack had no convection parameterization and
every convective column rained as a grid-point storm through Morrison. The
registration is therefore back at `device-pending`, and the superseded
digest is recorded in the adapter's limitations rather than sitting in
`device_evidence_sha256` as cover for a stack it never measured. The
battery is re-run with `cumulus = "gf"` in the order before that field
carries a value again:

```bash
woof global native-qualify CONFIG.toml --outdir out/native-qualification
```

Read the registration itself rather than this paragraph, because it moves:

```bash
woof global physics-manifest
```

Named limitations that stand either way: the cumulus advective forcing
lanes are zero and convective momentum tendencies are not coupled; the
scheme reads one scalar `dx_m` rather than the per-column Gaussian
spacing; Noah land categories are configured constants until real static
fields are supplied; and native Strang half-step scheduling is not WRF's
RK3 held-tendency scheduling.

### Effective resolution: what can be said today

The instrument of record is `dynamics.spectrum`, rebuilt on
spherical-harmonic total-degree spectra with an absolute anchor in
physical units:

```text
python -m woof.verify.harness run dynamics.spectrum --config CONFIG.toml \
  --checkpoints out/global/arwen_global_step*.npz
```

**This core has been read by it at T255, and only there.** Measured
2026-09-01 and 2026-09-02 on a 48 h native run and on both arms of a 24 h
A/B, at 250 and 500 hPa. The reading is that there is no half-variance
scale to quote: the control's spectrum never falls below half of the
observed reference inside the resolved range, so every sample returns
`unresolved`, and the gate that would divide such a scale by the
truncation wavelength (156.68 km at T255) carries no number for that
reason rather than a flattering one. What can be quoted without a
crossing is how much of the observed spectrum the core holds, globally at
250 hPa over 13 frames: 0.72 +- 0.08 of it at 500 km and 1.15 +- 0.27 at
250 km. The T533 arm produced no reading at all, because both launches
ran out of device memory before a checkpoint old enough to read, and T63
has not been re-read. Every number, its provenance and its instrument
status are in the effective-resolution document linked below. Nothing is
summarised into a single kilometre here, because no reading of record is
a single kilometre.

The readings that stood before it are **superseded** and are being
re-measured: 560 +- 23 km at T63, quoted then as 1.25x the truncation
scale, and 70 km at T533, quoted then as 1.3x the truncation scale. They
were produced by a 10-row band-FFT instrument in which an independent
audit reproduced six defects, among them a self-anchored reference (a
model with half the real atmosphere's synoptic energy read the same
effective resolution) and a grid-scale fallback returned unflagged when no
departure was found. Both quoted multiples are also against the zonal
wavelength at 45 degrees, a convention the document has retired; against
the isotropic `2 pi a / T` wavelength that document recomputes them under
(635.4 km at T63, 75.1 km at T533), 560 km reads 0.88 and 70 km reads
0.93. Those two figures are quoted as recomputed, so they are not the
table above: that table is `2 pi a / sqrt(n(n+1))` at `n = T`, which is
0.8 % shorter at T63 and 0.09 % shorter at T533, and under it the T63
reading would be 0.89 rather than 0.88.

The full account, including what does and does not rest on the retired
readings, is
[docs/arwen-global-effective-resolution.md](arwen-global-effective-resolution.md).
Quote from that document, and quote its wavelengths against the truncation
scale, never as a multiple of grid spacing.

### What is not claimed

Operational forecast skill. GPU validation without a target-device
receipt. Direct reuse of unadapted regional WOOF physics wrappers. A
complete energy and angular-momentum conserving hybrid vertical scheme.
The T533 grade of the two cores (the semi-Lagrangian T533 day was not
run in the window; the T533 default follows the two graded truncations,
the two-cores section says so). A fully implicit
three-dimensional primitive-equation solver. Conservation without the explicitly measured
repair mechanisms. An admitted global-to-regional forcing path.
Replacement of the regional nonhydrostatic model.

The wall clocks on this page are one forecast day under the shipped
40-level configuration, each with its card, its tenants and its date;
none is a throughput claim for another card.

## Where to go next

- [ARWEN_GLOBAL_QUICKSTART.md](ARWEN_GLOBAL_QUICKSTART.md) -- four
  commands from nothing to rendered global maps, plus a no-card rehearsal.
- [ARWEN_GLOBAL_FULL.md](ARWEN_GLOBAL_FULL.md) -- dynamics, hybrid
  coordinate, reference physics, water repair, receipts, non-claims.
- [ARWEN_GLOBAL_LEVEL5.md](ARWEN_GLOBAL_LEVEL5.md) -- the native adapter,
  its transaction and persistence contracts, and the regional parent
  bridge.
- [ARWEN_GLOBAL_CLIENT.md](ARWEN_GLOBAL_CLIENT.md) -- driving this model
  from a program: the versioned documents `run-plan` and `sources` answer,
  the plan envelope, the durable run files a client reattaches to, and what
  the engine offers that this package does not.
- [docs/arwen-global-effective-resolution.md](arwen-global-effective-resolution.md)
  -- the spectral instrument, its calibration and its refusals.
- [DATA.md](https://github.com/recastsystems/woof/blob/main/docs/public/DATA.md) -- what GDAS is, and what the regional route still
  refuses.
