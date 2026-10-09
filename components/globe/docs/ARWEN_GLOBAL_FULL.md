# WOOF global: moist hybrid spherical-harmonic research model

## Status

`woof.globe` is an additive research model built on the Level-3
`woof.globe.spectral` transform library. It does not replace, import, or
silently alter an ordinary regional WOOF forecast. Every leg a reader runs,
the three a forecast is made of (integrate, assimilate, write render-ready
tapes) and the whole research surface beside them, is a command of one
console script:

```bash
woof global --help
```

This distribution carries the model and the physics the model was graded
with, and depends on a published engine (`woof>=2.8.0,<2.9`) for the source
decoders, the observation front door, the static-field builder, the LETKF
filter core, the tape writer and the renderer. The schemes the native suite
integrates are inside the package as `woof.globe.core`, because a
published engine's copy of the schemes this model executes is a different
scheme (`noah` and `morrison` match it and are carried because their kernels
do not); the engine files
the carried code still reaches are pinned by path, size and SHA-256 in
`woof/globe/data/engine-seam.json`, and `woof global doctor` re-hashes
them and reports each as proven or moved. Neither mechanism is a version
check: the whole account, including which files are pinned and why a moved
file is a note rather than a refusal, is in
[ARWEN_GLOBAL.md](ARWEN_GLOBAL.md), section "Where that physics lives, and
what is still the engine's".

**Start here:** [ARWEN_GLOBAL.md](ARWEN_GLOBAL.md) is what ships and where
the envelope ends, and [ARWEN_GLOBAL_QUICKSTART.md](ARWEN_GLOBAL_QUICKSTART.md)
is the route from nothing to rendered global maps -- fetch one GDAS
analysis, run 24 h at T255, export, render -- in four commands. This page
is the model itself.

Every runnable TOML must contain the exact acknowledgement:

```toml
acknowledgement = "research-only-arwen-global-v1"
```

The model is executable with a complete compact reference-physics suite. The
existing WRF-derived WOOF CUDA schemes are **not** automatically declared
compatible. They enter only through the evidence-hashed native-adapter
registry, and an unregistered adapter is refused before step zero.

## Prognostic atmosphere

The global spectral state is

\[
\zeta,\quad D,\quad \theta,\quad \ln p_s,\quad
q_v, q_c, q_r, q_i, q_s, q_g,\quad
n_c, n_r, n_i, n_s, n_g.
\]

Fifteen fields: vorticity, divergence, potential temperature, log surface
pressure, six water species and five number moments. The moments are
prognostic in their own right, transported, diffused, repaired,
checkpointed, exported and restarted beside the masses rather than kept
inside a physics wrapper. The ten-field state without them is the Level-4
state, which the model still reads through an explicit checkpoint
migration and never by silently zeroing the five.

All three-dimensional fields use triangular spherical-harmonic coefficients at
one or more hybrid pressure levels. The surface pressure is two-dimensional.
The transform, vector spherical-harmonic machinery, compression, arbitrary
sampling, and Gaussian grid are inherited from Level 3 and remain bound to the
Level-3 arithmetic pin.

## Hybrid vertical coordinate

Half-level pressure is

\[
p_{k+1/2}=A_{k+1/2}+B_{k+1/2}p_s.
\]

The bottom identity is exactly `A=0, B=1`. Configuration may provide explicit
A/B arrays or ask for the deterministic pressure-blend generator. Every
configured coordinate is checked for positive, strictly increasing pressure at
multiple surface-pressure values.

The dry layer mass per unit area is

\[
\Delta m_k=\frac{p_{k+1/2}-p_{k-1/2}}{g}.
\]

Terrain geopotential remains a gridpoint field. Hydrostatic geopotential is
integrated upward from that surface field: each full layer with its virtual
temperature (the midpoint rule, exact for Tv linear in ln p since the full
level is the ln-p midpoint), and the half layer between the lower interface
and the full level with Tv varying linearly in ln p, its gradient taken from
the neighbouring levels. Integrating that half layer with the level
temperature alone omits R (dTv/dlnp) dlnp^2/8, and the terrain-following
variation of that term is a spurious pressure-gradient force at rest (1.06
m/s per hour at the top level over a 2 km mountain on a 20-level grid; 0.002
after). The momentum pressure gradient R Tv grad(ln p) uses the exact
full-level gradient, grad(ln p_k) = (B_{k-1/2}/p_{k-1/2} + B_{k+1/2}/p_{k+1/2})
p_s grad(ln p_s)/2, so a column with Tv linear in ln p balances pointwise.

## Dynamics

The model advances relative vorticity and divergence in vector-invariant form.
Wind is reconstructed from vorticity and divergence through streamfunction and
velocity potential. Nonlinear momentum and tracer fluxes are evaluated on the
dealiased Gaussian grid and transformed back to the triangular spectral state.

Surface-pressure tendency is diagnosed from the vertically integrated
horizontal pressure-mass flux. Half-level pressure velocity is then diagnosed
from layer continuity with zero top and bottom fluxes. Potential temperature
and all eleven advected tracers, the six water species and the five number
moments, use horizontal and vertical pressure-mass-flux form.
The vertical flux through each interface is a monotone second-order upwind
reconstruction: the upstream layer's van Leer (harmonic-mean) limited
gradient in pressure, evaluated at the interface and bounded by the two
adjacent layer values. Momentum keeps its second-order centred advective
form. A first-order donor-cell flux carries a numerical vertical diffusivity
of |omega| dp / 2 on every thermodynamic field (5.6 h e-fold for a one-layer
feature at omega 0.5 Pa/s); the limited flux measures 4.2e-5 K/s against the
donor 1.3e-4 and the centred 3.8e-5 on a smooth profile, converges at second
order in L1, and admits no new extrema.

A strict spectral CFL estimate is evaluated before each large-step state
update. Exceeding the configured limit raises an error and leaves the incoming
state unchanged.

## Semi-implicit vertical-mode treatment

Every vertical gravity-wave mode is integrated implicitly through the linear
operator of the hydrostatic hybrid-coordinate equations about an isothermal
reference state (320 K, 1000 hPa by default), built in the model's own
discrete forms for continuity, interface omega, the flux-form theta tendency,
the hydrostatic geopotential and the pressure-gradient force. That operator is
subtracted from the explicit right-hand side, and the default `[time]`
integrator, `imex_ssp3`, advances the two together as one IMEX Runge-Kutta
pair: the third-order SSP explicit tableau for the residual right-hand side
and a diagonally implicit tableau with the same abscissae and weights for the
operator (two solves of `(I - 3 dt L / 10)` per step). Sharing every abscissa
makes any balanced state an exact fixed point of the step, at every dt and
over any terrain: at rest over a 2 km mountain the step reproduces the
explicit right-hand side's own residual response (4e-4 m/s and 6e-4 K after an
hour at dt = 60 s and at dt = 300 s), where the earlier symmetric split (the
square root of a Crank-Nicolson step on either side of the explicit step,
still selectable as `integrator = "ssprk3"` or `"rk4"`, bit-identical to its
era) drifted 8e-3 m/s and 0.25 K at dt = 60 s and 0.18 m/s and 1.5 K at
dt = 300 s. The implicit stability function is neutral to fourth order for
resolved modes and damps unresolved ones up to omega dt = 17.3, past which the
step is refused by name; the linearized rest ceilings on the 40-level default
stack are 583 s at T533 and 1248 s at T255 against the split's 395 s and
868 s, and the Doppler growth at the advective gate is 1.000000 against
1.000745. The Helmholtz solve is one nlev-by-nlev matrix per
total spherical degree, diagonalised in the vertical-mode basis because

\[
\nabla_h^2Y_n^m=-\frac{n(n+1)}{a^2}Y_n^m,
\]

so each mode receives exactly the scalar Crank-Nicolson factor of its own
phase speed (354.9, 227.4, 151.5, 109.7 m/s and below on the 40-level default
stack). The global degree-zero divergence and mass modes pass through
untouched. The earlier external-mode proxy (one reference speed coupling the
vertically averaged divergence to log surface pressure, applied after the
explicit step) stays selectable as `[semi_implicit] scheme = "external"`,
bit-identical to its era under that era's arithmetic pin. The reference
temperature, reference surface pressure and off-centring weight are
`[semi_implicit]` keys and join the config identity under the vertical-mode
scheme.

## Two-time-level semi-Lagrangian semi-implicit integrator

`[time] integrator = "sl_si"` runs the whole moist hydrostatic system with a
two-time-level semi-Lagrangian semi-implicit scheme instead of the Eulerian
IMEX pair. It is the DEFAULT core of `woof global run` and `woof global da`
at every truncation since 2026-09-06 (`config.DEFAULT_TIME_INTEGRATOR`, aliased
`config.SHIPPED_INTEGRATOR`): a config that omits `[time] integrator` gets it,
one that omits `dt_s` gets its 300 s step, and one that omits the `[diffusion]`
drain, the `[semilag]` gather or the `[semi_implicit]` off-centring gets the
core's own values (order 16 at 720 s at the truncation, the six-point gather,
0.55). The choice is the decision of 2026-09-06, made on the observation grade at
equal cost and the wall per forecast day (the decision paragraphs at the end of
this section, and the door page's two-cores section); `imex_ssp3` stays
selectable by name at every truncation at its rule step, its ten-step identity
pinned. This section carries the numbers the choice was made on and the rows it
loses.

Why it exists. The explicit Eulerian advection is bound by the fastest wind on
the planet, not by anything the forecast needs: at T255 the measured analysis
jet is 114.7 m/s and the advective CFL refuses any step past 187 s, and a
day-2 jet past 225 m/s halves that again. Physics is 59 percent of the T255
step and is paid once per step whatever the step is, so the step count is the
forecast wall. A semi-Lagrangian advection has no advective CFL limit at all;
what bounds it is the deformation of the flow.

The scheme. For each advected variable, with `A` an arrival point (a grid
point, always) and `D` its departure point,

```text
X^{n+1} - alpha dt (L X)^{n+1} = Q_X(D) + (dt/2) [2 N^n - N^{n-1}]_A
Q_X = X^n + dt [ (1 - alpha) (L X)^n + N^n / 2 ]
```

with `L` the same semi-implicit operator the Eulerian arm subtracts and
`N = A - L` the residual of the ADVECTIVE-form tendency. The departure-side
bundle is pre-combined before the interpolation, so one field is read per
advected variable and not two: three geocentric Cartesian wind components,
potential temperature minus the reference profile, vapour, the ten grid
tracers, and log surface pressure. The left-hand side is the shipped
Helmholtz solve at `tau = alpha dt`, called ONCE per step where the IMEX pair
calls it twice.

Four things are arithmetic rather than preference. Momentum travels as
Cartesian components and is carried to the arrival point by the minimal
rotation, which is the exact parallel transport of the great-circle
trajectory: projecting instead would shorten the wind by 1.46e-5 per step at
the measured 34.4 km T255 displacement, which is 4.2e-3 of the jet per
forecast day. In Cartesian components every `tan(phi)` metric term cancels, so
no momentum term has to be right at a pole where the local basis turns through
2 pi in 326 m. Potential temperature is advected as `theta - theta_ref(k)` and
its tendency is the semi-implicit operator's OWN flux form of the reference
profile's vertical advection with the run's mass flux, because the difference
between two discretizations of that term is a gravity wave and would ride the
explicit budget. The quasi-monotone clip runs on vapour and the ten tracers and
not on the dynamical bundle, which is a state plus a tendency whose physical
bounds are not its neighbours' values; `[semilag] quasi_monotone_dynamics`
reaches the other arm.

The gates, and what each refuses. The advective CFL becomes a MEASUREMENT on
this path and the Lipschitz number becomes the refusal: above one the
trajectory map is not invertible, two arrival points share a departure point,
and the model does not blow up when that happens, it mislocates silently.
`[time] maximum_lipschitz` defaults to 0.75. The departure-point search's last
iteration is measured every step against `[semilag]
trajectory_convergence_cells` (0.01 of the local meridional grid length),
because an unconverged trajectory reads as a Rossby phase error and produces
no other symptom. The flux-form tracer sweep is routed around rather than
sub-cycled 37 times at T255 and 160 at T533, with its own path still
selectable as `[semilag] tracer_scheme = "flux_form"`. The barotropic
semi-implicit proxy, a disabled operator and a partial `[semi_implicit]
weight` are each refused by name under this integrator, because a 300 s step
has no explicit budget for any part of a 350 m/s mode.

The second time level is checkpointed. A run under this integrator writes
schema `gpuwm.arwen-global-checkpoint/v4`, which is the v3 archive plus seven
trajectory arrays; every other integrator keeps writing v3 byte for byte, a v3
archive carrying a trajectory array is refused by name and so is a v4 archive
missing one. Carrying the level is what makes a midpoint restart bit-exact
against the uninterrupted run, which is the property the device-qualification
pin asserts.

Its arithmetic is a distinct pin (`vertical_modes/sl_si`), naming the four
things that change: the gravity-wave composition, the momentum, the scalar
transport and the checkpoint. A checkpoint of one integrator never resumes
under the other.

What is measured, and what is not. MEASURED 2026-09-06 on one RTX 5070 Ti with
nothing else on it: at T255 with 40 levels and physics off, a whole 24 h
forecast day from the GDAS analysis completes at 100, 200, 300 and 450 s, in
2.98, 1.57, 1.08 and 0.86 minutes, with every mass and total-water gate of
record green and an advective Courant number reaching 2.197; the Eulerian arm
does not complete that day at its own 60 s step or at 40 s, because with no
physics to damp the jet its Courant number reaches 0.760 and 0.752 against the
0.750 its gate allows. Per step, 270.59 ms at dt = 300 s against 352.46 ms at
dt = 60 s, which is 6.5x on one adiabatic forecast day. On the Jablonowski and
Williamson 2006 baroclinic wave at T85, both cores at dt = 300 s, the
perturbation's e-folding time between days 4 and 8 is 1.481 days against 1.449,
the day-9 l2 of the surface-pressure perturbation 319.40 Pa against 320.33 and
the day-9 minimum surface pressure 941.80 hPa against 940.01. At T255 the same
pair reads 1.583 days against 1.448, 9.3 percent apart where T85 is 2.2, so the
growth rate this core gives the instability is resolution-dependent and it is
the coarser resolution that flatters it; the day-8 perturbation l2 is 188.73 Pa
against 188.14, within 0.3 percent, and the comparison stops at day 8 because
the Eulerian arm of that pair did not complete day 9.

The moist package on a real day. T255 L40 with the whole native suite from
the GDAS 2026-09-01 00Z analysis, dt = 300 s, 24 hours, on one card: it runs,
and every gate of record is green. Mass drift 7.5e-7 against 1e-4, total water
drift 1.1e-8 against 1e-3, the largest Lipschitz number 0.222 against 0.75, the
trajectory's last iteration moving 2.5e-4 of a cell against 0.01. 288 steps in
259 s, which is 4.3 minutes per forecast day against the Eulerian core's 1,440
steps and 18.4 minutes on the same case, the same suite and the same card: 4.3x.
Read that ratio with the baseline's own headroom beside it. The Eulerian arm's
receipt for this day reads a maximum spectral CFL of 0.326 against the 0.750 its
gate allows, so 60 s is 43 percent of the step its own refusal would admit, and
MEASURED 2026-09-06 on an RTX 5070 Ti the shipped Eulerian core completes the same
forecast day at dt = 120 s: status pass, every gate of record green, spectral
CFL 0.644 with one pre-warning at step 10, 1.2 percent less precipitation, and
1.56x less wall than its own 60 s arm on the same card. Against the observations
that doubled step costs it 0.65 m of 500 hPa height at 00Z and buys back 2 m
temperature, dewpoint and both wind rows, which is a wash rather than a loss.
At 135 s the same core still completes the day with every gate green, a
spectral CFL of 0.722 (96 percent of its refusal, which is where that core's
ladder ends on this flow), 753.7 s of wall and a surface scorecard that is again
a wash against 60 s. So roughly a third of the 4.3x is headroom the shipped
config was not using: against the Eulerian core at 120 s the semi-Lagrangian
core at 300 s is 2.5x on a pair measured in one window on one card, and 2.3x
against it at 135 s, not 4.3x. Both numbers are here because 4.3x is what a
user switching from the shipped config sees and 2.5x is what the integrator is
worth.

What the longer step does to that day's weather, measured and not yet judged.
Against the Eulerian arm, the semi-Lagrangian day is 20 percent drier in the
global mean rain (0.928 against 1.159 kg/m2), 31 percent short on cells above
50 kg/m2 (312 against 451) and 54 percent short on its largest cell (181
against 397 kg/m2), and carries 18 percent more condensate in the air. That is
the shape of a 150 second physics half through a convective suite. The in-situ
energy ledger's truncation-scale tripwire fires on the semi-Lagrangian day and
not at all on the Eulerian one: three times on the day of record's own ledger
(steps 130, 270 and 280; its receipt's `trips_by_tripwire` reads 3) and four on
a second run of that day (steps 130, 260, 270 and 280, on model
levels 1, 0, 1 and 2, reading 2.286, 2.311, 2.791 and 2.284 against its 2.10
threshold), so one firing at hour 11 and the rest inside the last 100 minutes. That
makes the hyperdiffusion and the top sponge, both tuned at a 60 second step, the
named follow-up, and the top of the model the place to look: at 24 h the two
cores' potential temperature differs by 1.39 percent of its own value on the top
two levels and by 0.06 percent on the levels the radiosondes read.

How it grades against the observations. MEASURED 2026-09-06 on an RTX 5070 Ti: the same
day, both cores on the same tree from the same analysis, scored at the ASOS
stations at 18Z and 00Z and the IGRA2 radiosondes at 12Z and 00Z with the GFS
and the IFS forecasts beside them, 18 rows, root mean square error, each
difference read with a paired bootstrap over the stations and sites. **Four
rows better, four worse, ten level.** The semi-Lagrangian day is the better one
at the surface: 2 m temperature 2.430 against 2.540 K at 18Z and 2.790 against
2.889 at 00Z, 2 m dewpoint 4.244 against 4.277 K, 10 m wind 1.548 against 1.601
m/s. It is the worse one aloft, and the four losses are one signature rather
than four: its column is 0.06 to 0.18 K colder at 850 and 500 hPa (1.576
against 1.520 K and 1.163 against 1.126 K at 00Z) and its 500 hPa height is
about 1.2 m lower, which reads as +0.695 m of height error at 12Z, and its
sea-level pressure is 0.032 hPa worse at 18Z. Every one of those four is small:
the largest, the 500 hPa height at 12Z, is 4.6 percent of that row's own error,
and the surface wins are two to four times the project's 0.03 admission bar
where the losses sit at one to two times it. Read instead against the noise this
comparison carries -- two Eulerian arms of the same case differ by 0.0187 K,
0.0067 hPa and 0.112 m -- the four losses are 4.1 to 7.7 times that floor and
the four wins 1.9 to 5.9 times it, so the smallest win, the 18Z dewpoint at
1.9x, is weaker than any of the losses. Both readings are here because they
order the rows differently and the decision should not rest on which one a
reader happens to see.

Read the ledger against the ladder, and it is the integrator AND its step. The
same eighteen rows scored at dt = 100 s, the rung nearest the shipped arm's
60 s, separate the two: the 18Z sea-level-pressure loss is entirely the scheme
(it is already +0.032 hPa at 100 s and does not grow), while the 500 hPa height
loss at 12Z is 27 percent the scheme and 73 percent the longer step (+0.189 m at
100 s against +0.695 at 300 s), and the 850 and 500 hPa temperature losses are
18 and 49 percent the scheme. One win reverses: at 100 s the semi-Lagrangian
core is WORSE than the Eulerian one on the 00Z 2 m temperature (2.924 against
2.889) and only the longer step turns that row into a win. So three of the four
losses are mostly the price of the step rather than a property of the scheme,
which is the same price the 20 percent rain deficit above is paying.

What the eighteen rows cannot see. The scorecard reads 850, 500 and 250 hPa and
the surface; this stack carries eleven full levels above 50 hPa and no row
touches them. Measured on the model's own levels at 24 h, the global mean
potential temperature of the two cores differs by 0.016 percent near the ground,
0.063 percent through the band the soundings read, 0.310 percent on levels 8 to
13, and 1.385 percent on the top two levels, which is about 4.7 K of temperature
at the lid and twenty-two times the disagreement the soundings can report. By
scale the same asymmetry: the surface pressure field agrees to within 1.3
percent of power in every wavenumber band, while the 500 hPa vorticity keeps
0.71, 0.49, 0.17 and 0.53 of the Eulerian arm's power in the bands above n = 60.
The two arms' 500 hPa vorticity correlates at 0.999 below n = 20, which is why
the charts are the same synoptic map, and at 0.762 over all scales, which is why
they are not the same field. Anything that reads the semi-Lagrangian arm as
carrying MORE small-scale texture is reading the picture rather than the field:
it carries less, by a factor of six at the wavenumbers where the gap is widest. That spectral shape -- too much drain at high
wavenumber with a mild pile-up just below it -- is the hyperdiffusion follow-up
named above, now measured rather than suspected.

Where the 4.7 K at the lid came from, and its fix. MEASURED 2026-09-06 on
an RTX 5070 Ti, the lid disagreement is not eddies and not the sponge: at 24 h the
semi-Lagrangian day of record is warmer than the Eulerian one by 33.6 K of
potential temperature on level 0 and colder by 12.5 K on level 1 in every
latitude band (29 to 37 K and -10.5 to -14.9 K over the six 30-degree bands,
zonal standard deviation 1.7 K), and the difference grows linearly through the
day (+19.2 K at 12 h, +26.6 K at 18 h). The mechanism is the way the core
carried the reference profile. Until this fix it advected `theta - theta_ref(k)`
and added the reference's material tendency, `-omega dtheta_ref/dp` in the
operator's own one-sided flux form, as a grid tendency at the arrival point.
Under descent at the lid the parcel arriving at level 0 departs from level 0,
because nothing lies above it and the trajectory is clamped there, so the gather
reads no reference change, and the tendency warmed it anyway by
`-omega_1/2 (face_1 - theta_ref_0) / dp_0`, where the Eulerian core's upwind flux
carries the layer's own value out of it and warms nothing. Applied to the
Eulerian arm's own 24 h state, one 300 s step of that form changes level-0 theta
by +0.1127 K in the global mean (+0.206 K over the descending half of the
globe, +0.016 K over the ascending half), which is +32.5 K a forecast day
against the +33.6 K measured, and the same one-sided form at level 1 is 29
percent too steep for any displacement. The core now gathers the WHOLE
potential temperature and its nonlinear residual is minus the operator's own
thermodynamic row: a parcel's reference change is whatever the vertical stencil
reads between its departure and arrival levels, exactly consistent with what
that stencil reads of everything else, the semi-implicit treatment of the row
is unchanged, and on the same state the step reads -0.043 K at level 0, the sign
and half the size of the Eulerian upwind form's -0.090 K. The scalar-transport
pin moved to v3 with it, so no checkpoint of the earlier arithmetic resumes
under this one, and the increment-convergence, rest-state, mountain and restart
gates pass unchanged.

The wall it buys, and the step it is chosen at. One forecast day at T255 with
the whole native suite on an RTX 5070 Ti, timed on a card verified empty for the whole
run and end to end from each run's own receipt: **4.278 minutes at dt = 300 s
against 18.322 for the shipped Eulerian arm at dt = 60 s, which is 4.28x.**
The ladder, at the scored output cadence rather than the timing one, is 9.99,
5.77, 4.31, 3.31 and 3.13 minutes at 100, 200, 300, 450 and 600 s, and every
rung completes a forecast day with every gate of record green. At T383 the same
core completes a forecast day with every gate green and a device peak of 15.858
GiB; that row's wall is not reported because its card was shared. T533 needs
about 25.8 GiB of a 32 GiB card, more than the three quarters the sharing rule
leaves, and T799 about 57 GiB, which the sizing door refuses on this hardware.

Past 300 s the wall stops being step-count bound: halving
the steps again from 300 to 600 s buys 27 percent, not half. The scorecard is
nearly flat across the ladder at the surface and degrades monotonically aloft
(500 hPa height rmse at 12Z 15.23, 15.57, 15.74, 15.81, 15.85 m), and the
trajectory search's own residual rises from 2.2e-5 to 7.6e-3 of a grid cell
against its 1e-2 refusal. **dt = 300 s is the shipping step**: it is where the
upper-air cost is still inside a tenth of a metre of the shortest rung's, the
trajectory residual keeps a factor of 40 to its refusal, and the wall is
already 4.3x. dt = 450 s is available and costs about 0.4 m of 500 hPa height
rmse for another 23 percent of the wall.

What the tracers cost, measured rather than assumed. The ten grid condensate
species and number moments ride the same stencil as everything else and their
mass is closed per species by a Bermejo and Conde fixer. The mass that fixer
has to move is large for a trace species: up to 10.1 percent of the graupel per
step over the forecast day, and 6.0 percent over the six-hour arms, which run
the other case this tree carries (the forecast day is the 2026-09-01 00Z
analysis, the six-hour arms the 2026-08-30 18Z one, so the two readings are two
cases and not two points on one run). It is the ADVECTION's error, not the
fixer's: over those six hours it reads the same to two figures whichever form
the fixer runs AND with the quasi-monotone limiter turned off (6.4 percent).
What it measures is what POSITIVITY costs, for a species a few cells across and
one to three model layers deep, which is what a condensate species is at T255
on forty levels; MEASURED on the numpy specification, the same correction runs
from 13.5 percent for a species one layer deep to 1.9 percent for one ten
layers deep while the unlimited gather's own mass error holds at 0.5 percent,
so the number rises at a sharper truncation or on a thinner ladder and the
0.25 limit owes a re-reading at T533 and T799. A cubic through a field that is zero at
three of its four stencil points undershoots below zero on the shoulder of
every maximum, something has to lift it back because a negative mixing ratio is
not a state this model has, and turning the limiter off only moves that lift
from the gather's clip to the mass fixer's own floor. Turning both off stops
it, and the run is then refused at its FIRST step on a cloud water of -1.49e-5
against a maximum of 1.02e-3. Against the atmosphere's own water, which is what
the conservation breakage is about, the fixer's net correction is 3.4e-5 of the
column per step and the day's total water drift is 1.1e-8.

The gate that reads it moved with the measurement. The 1e-4 the specification
of record estimated was written before the scheme ran and is three orders of
magnitude away from what a condensate species does; the per-species row is now
0.25, priced at 2.5 times the worst reading of the T255 L40 forecast day and
naming a different breakage (a species the fixer is placing rather than the
flow transporting), and a second row holds the fixer's net water correction at
1e-4 of the column, three times the day's reading. The `proportional` fixer
form is retired and refused by name: once every conservative form floors the
field at zero before it measures the mass, its weighting and `bermejo_conde`'s
are the same array, and it was a door whose two values produced the same run
under two config hashes.

`[semilag] tracer_scheme = "flux_form"` runs the same day with the ten tracers
on the shipped van Leer sweep instead, everything else identical, for the arm
that says what the semi-Lagrangian tracer path is worth.

What users run, and at what step. `configs/arwen_global_gdas_t255
_native_sl_si_24h.toml` is the arm of record: T255, 40 levels, the whole native
suite, one forecast day from the GDAS 2026-09-01 00Z analysis at dt = 300 s.
The four `_dt100`, `_dt200`, `_dt450` and `_dt600` files beside it are the
ladder, the three `_alpha0**` files the off-centring sub-ladder, and
`arwen_global_gdas_t{255,383,533,799}_native_sl_si_wall.toml` the wall table's
rows. `arwen_global_gdas_t255_native_imex_24h.toml` is the Eulerian twin every
one of them is graded against, the Eulerian core at its rule step and
buckets on the same case (90 s since 2026-09-06; the grade of record was read
against it at 60 s and, at the CFL-admitted step, against `..._dt120.toml`),
and the config the Eulerian core's ten-step identity gate is pinned on now
that the config of record runs the default core. `arwen_global_gdas_t383_native_{imex,sl_si}_24h.toml` and
`arwen_global_gdas_t533_native_sl_si_24h.toml` are the T383 and T533 arms of
the equal-cost grade, written every six hours for the scorecard, and
`..._t383_native_imex_24h_dt80.toml` the CFL-admitted T383 Eulerian arm it
was read against.

What the core reads on its gates after the lid fix and the retune (MEASURED
2026-09-07 on an RTX 5090 unless said otherwise). On the analytic steady state at
T255 the maximum surface-pressure change over nine days is 0.71 hPa (day 8;
0.53 hPa at day 9), inside the case's 1 hPa bar where the retired arithmetic
read 1.153 hPa; at a 600 s step the same case reads 1.17 hPa, which is one
reason 300 s is the step. Neither core is an exact fixed point of that state
(it is a steady solution of the continuous equations and of neither
discretization), which is why the Eulerian pair's own error reads 0.097 and
0.153 hPa rather than zero. The perturbed twin's surface-pressure error
e-folds in 1.86 days over days 4 to 8. The in-situ energy ledger's
truncation-scale tripwire, which fired three times on the retired arithmetic's
moist day of record (steps 130, 270 and 280; its receipt's `trips_by_tripwire`
reads 3) and four times on a second run of that day (steps 130,
260, 270 and 280), stays silent over 288 steps on eight of the thirteen dry
semi-Lagrangian arms of the ladder, the default among them, and trips once or
twice between steps 150 and 200 on the other five (no drain, order 8 at 3,456
and 4,320 s, order 16 at 1,440 and 2,160 s). It reads the growth of the top
decile's energy from one reading to the next and not its level, so order 8 at
8,640 s, whose truncation band ends the day at 1.73x, stays silent while order
8 at 3,456 s at 0.98x trips once (the
dry Eulerian reference at 30 s trips it at step 30, a truncation tail its
physics would otherwise damp), and not once on the bare default day at T255 (the six-point gather, order 16 at 720 s, 288 steps, every gate green, the Lipschitz number 0.24 of its 0.75 gate; MEASURED 2026-09-07 on an RTX 5090 from `arwen_global_gdas_t255_native_24h_bare`); the firings belonged to the retired lid arithmetic. Against the Eulerian arm
at 120 s on the moist day of record the lid disagreement at 24 h is +16.5 K
at the top model level (was +33.7 K before the fix) and -12.2 K at the second
(unchanged by it), the sounding band (levels 16 to 30) within 0.046 percent
and the near-ground levels within 0.022 percent; the top level itself drifts
-62 K in a day on this core and -79 K on the Eulerian one, so what remains
there is a difference between two absorbers at a level no scorecard row
reads. The bare default day reads the same lid (+16.3 K and -12.2 K at the top two levels, the sounding band within 0.041 percent, the near-ground levels within 0.012 percent) at the same 8.70 GiB device peak as the cubic gather, and against the Eulerian arm at 120 s on the moist day its 500 hPa vorticity spectrum keeps 0.983 / 0.894 / 0.927 / 0.809 at n = 61-120 / 121-180 / 181-230 / 231-255 with a half-variance degree of 58.8 against the reference's 62.2, where the cubic gather at order 8 and 2,160 s after the same lid fix kept 0.869 / 0.627 / 0.446 / 0.399 and read 47.2, and the day of record before the fix kept 0.920 / 0.658 / 0.466 / 0.414 and read 48.1 (an earlier reading of 0.71 / 0.49 / 0.17 / 0.53 was against the 60 s Eulerian arm). The 250 hPa kinetic energy moves the same way, 0.866 / 0.741 / 0.756 / 0.613 against 0.788 / 0.524 / 0.362 / 0.308.

The small-scale drain, set for the step (MEASURED 2026-09-07, T255 L40, one
dry forecast day from the GDAS 2026-09-01 00Z analysis, every arm at 300 s
against the Eulerian core at 30 s, `tools/semilag_scale_diagnostics.py`; the
fraction of the Eulerian 500 hPa vorticity power kept at total wavenumbers
61-120 / 121-180 / 181-230 / 231-255, and the half-variance degree against
the reference's 36.6):

| arm | 61-120 | 121-180 | 181-230 | 231-255 | half-variance n |
|---|---|---|---|---|---|
| a second Eulerian step (24 s against 30 s), the noise | 1.001 | 0.999 | 0.997 | 0.986 | 36.6 |
| cubic gather, order 8 at 2,160 s (what shipped) | 0.835 | 0.576 | 0.474 | 0.502 | 31.3 |
| cubic gather, order 8 at 4,320 s | 0.837 | 0.579 | 0.507 | 0.730 | 31.5 |
| quintic gather, order 8 at 2,160 s | 0.969 | 0.863 | 0.766 | 0.700 | 35.2 |
| quintic gather, order 8 at 3,456 s | 0.972 | 0.871 | 0.822 | 0.980 | 35.5 |
| quintic gather, order 8 at 4,320 s | 0.973 | 0.874 | 0.848 | 1.147 | 35.6 |
| quintic gather, no explicit drain | 0.983 | 0.969 | 1.292 | 4.473 | 38.5 |
| quintic gather, order 16 at 2,160 s | 0.977 | 0.904 | 1.031 | 1.591 | 36.3 |
| quintic gather, order 16 at 1,080 s | 0.975 | 0.911 | 1.026 | 1.191 | 36.2 |
| **quintic gather, order 16 at 720 s (the default)** | **0.973** | **0.915** | **1.025** | **0.965** | **36.1** |

The gather is the drain: with no explicit diffusion at all the six-point
stencil keeps 0.97 at 121-180 and lets the truncation band pile up 4.5x, and
order 16 at 720 s is the shape that removes the pile-up without reaching
below n = 0.87 T, where it is weaker than the Eulerian core's own order 8 at
2,160 s. The 0.915 left at 121-180 is the stencil's own filtering at 288
passes a day, not the drain (no drain reads 0.969 there, and no candidate
reads above it). The lid, the SETTLS extrapolation and the sponge were
measured on the same ladder: the whole-theta gather leaves the top level
drifting -12 K a day with the sponge and -66 K without it, so the sponge
stays at a 300 s step; turning SETTLS off moves no band by more than 0.01.

**Why it is the default, per truncation.** The decision was made on
2026-09-06: the semi-Lagrangian core is fast enough that everything is
built, run, tested and graded with it, so it is the shipped default at every
truncation and the Eulerian core is selectable by name. The grade it rests
on is stated under the rule this line uses, which is that
the core which, at the same card-time, grades more rows better than worse on
the observation scorecard and runs the forecast day in less wall takes the
default, with every gate of record green and the rows it loses named; a core
that grades more rows worse than better at equal cost does not take it
whatever its wall. The equal-cost grade (MEASURED 2026-09-06, one forecast day
from the GDAS 2026-09-01 00Z analysis, the whole native suite, both cores on
one tree, eighteen rows, paired bootstrap, 0.03 bar;
`tools/semilag_grade_tables.py`,
`configs/arwen_global_gdas_t{255,383,533}_native_{imex,sl_si}_24h*.toml`)
reads:

- At T255, `sl_si` at 300 s against `imex_ssp3` at 120 s (the step the
  Eulerian gate itself admits on this day): three rows better, three worse,
  twelve level. The losses are the 12Z 500 hPa height (+0.39 m), the 00Z
  850 hPa temperature (+0.060 K) and the 00Z 500 hPa temperature (+0.034 K);
  the wins are 18Z 2 m temperature (-0.098 K), 18Z 10 m wind (-0.052 m/s) and
  the 00Z 500 hPa height (-0.51 m). Wall 279 s against 673 s on the same
  card.
- At the T255 budget, the equal-cost pair: `sl_si` at T383 and 300 s (394 s a
  day on the RTX 5090) against `imex_ssp3` at T255 and 120 s (673 s on the
  RTX 5070 Ti; about 300 s on the 5090 at its measured 357 ms a step): five
  rows better, four worse, nine level. The losses are the 18Z sea-level
  pressure (+0.076 hPa), the 00Z 2 m dewpoint (+0.311 K, ten times the bar),
  the 12Z 500 hPa height (+0.26 m) and the 00Z 850 hPa temperature
  (+0.072 K); the wins are 2 m temperature at both hours (-0.083 and
  -0.242 K), 10 m wind at both (-0.064 and -0.049 m/s) and the 00Z 500 hPa
  height (-0.68 m). This is the pair the default is decided on.
- At T383, `sl_si` at 300 s against `imex_ssp3` at 80 s (the CFL-admitted
  step): four better, five worse, nine level. The losses are the 18Z
  sea-level pressure (+0.080 hPa), the 00Z 2 m dewpoint (+0.259 K), the 12Z
  500 hPa height (+0.43 m), the 12Z 250 hPa wind (+0.129 m/s) and the 00Z
  850 hPa temperature (+0.101 K). Wall 394 s against 1,267 s on a shared
  card, so the ratio is a capability reading and not a timing.
- Resolution on its own, `imex_ssp3` at T383 and 80 s against T255 and
  120 s: four better, one worse (18Z 2 m dewpoint, +0.039 K), thirteen
  level; the 500 hPa height improves 0.17 and 0.58 m.
- The same arms on the fixed core (the whole-theta gather above; MEASURED
  2026-09-07, against the Eulerian core at its shipped 90 s): T255 fixed
  against itself before the fix four better, one worse (18Z 2 m temperature
  +0.057 K), thirteen level; T383 fixed against itself before the fix five
  better (the 250 hPa wind at both hours, the 00Z 850 hPa wind, the 00Z
  500 hPa height, the 00Z dewpoint), two worse (2 m temperature at both
  hours, +0.067 and +0.079 K), eleven level. T255 fixed against `imex_ssp3`
  T255 at 90 s four better, four worse, ten level; the equal-cost pair on
  the fixed core, T383 against `imex_ssp3` T255 at 90 s, two better (18Z
  10 m wind -0.066 m/s, 00Z 500 hPa height -0.55 m), four worse (18Z
  sea-level pressure +0.057 hPa, 00Z 2 m dewpoint +0.130 K, 12Z 500 hPa
  height +0.34 m, 00Z 850 hPa temperature +0.055 K), twelve level, at 338 s
  on the RTX 5090 against 1,050 s on the RTX 5070 Ti; T383 fixed against
  `imex_ssp3` T383 at 80 s three better, four worse, eleven level. A
  verdict count moves with the station set at one marginal row: the T255
  fixed pair against the 90 s arm reads four better, five worse, nine level
  on a T255-only set of 1,577 stations (the 00Z 10 m wind row, +0.045 m/s
  with an interval of [+0.003, +0.093], a loss there and a tie on the
  1,519-station set that held the T383 arms too); every other row keeps
  its verdict.
- The shipped package itself, the six-point gather at order 16 and 720 s
  with the lid fix (the ladder above), run from the bare door
  (`..._t255_native_24h_bare.toml`, no flag) on the same analysis and
  scored with the same chain (MEASURED 2026-09-07, RTX 5070 Ti, 246.0 s a
  forecast day in-process, 8.701 GiB, twelve gates green, the tripwire
  silent): against `imex_ssp3` T255 at 90 s four better (18Z 2 m
  temperature -0.050 K, 18Z dewpoint -0.032 K, 18Z 10 m wind -0.055 m/s,
  00Z dewpoint -0.129 K), four worse (00Z 2 m temperature +0.097 K, 00Z
  10 m wind +0.048 m/s, 12Z 500 hPa height +0.31 m, 00Z 850 hPa temperature
  +0.050 K), ten level; against the 120 s arm three better (18Z 2 m
  temperature -0.035 K, 18Z 10 m wind -0.050 m/s, 00Z 500 hPa height
  -0.25 m), two worse (12Z 500 hPa height +0.20 m, 00Z 850 hPa temperature
  +0.052 K), thirteen level (two better, two worse, fourteen level on a
  second run of the same door, the 18Z 2 m temperature
  win at the interval's edge by 0.001 K); head to head against the graded
  cubic package one better (12Z 500 hPa height -0.14 m), three worse (00Z
  500 hPa height +0.47 m, 850 hPa wind +0.051 and +0.064 m/s), fourteen
  level, while keeping 2.05 times its vorticity power at wavenumbers 181
  to 230 and 1.55 times at 121 to 180. The T383 rows above are the cubic
  package's; the shipped package was not run at T383.

The pattern is the same at every truncation: the semi-Lagrangian core is the
better one at the surface on temperature and wind and the worse one on
dewpoint and aloft, and the fix moves it the way its mechanism says (the wins
it adds are aloft, the dewpoint loss shrinks from 0.31 to 0.13 K, 2 m
temperature gives back 0.06 to 0.08 K). What the fix did not remove is
measured too: the bare run of the T255 config of record on the fixed core
(MEASURED 2026-09-07, the archived GDAS 2026-08-30 18Z day) against the
Eulerian 24 h checkpoint of the same day reads +16.4 K at the top level and
-12.0 K at the second, zonally uniform, within 0.1 K from the fourth level
down, the eddies at the lid agreeing to 2 percent; half the +33.6 K of the
mechanism above and of the same shape, inside the sponge, unread by every
scorecard row, and carried as open against the Eulerian core's lid. Under the decision, on the scorecard of
record and the wall, `sl_si` is the shipped default
(`config.DEFAULT_TIME_INTEGRATOR`, `config.SHIPPED_INTEGRATOR`) at T255, T383
and T533, at about a quarter of the Eulerian wall per forecast day (T255
256.7 s alone on the RTX 5070 Ti against 1,099.3 s at 60 s and 1,050.2 s at
90 s; T383 337.9 s on a shared RTX 5090 at a 15.86 GiB device peak; the T533
and T799 days are sized at 24.45 and 53.78 GiB and are not timed), and
`imex_ssp3` ships selectable by name with the rows above named in both
directions and its ten-step T255 identity pinned on
`..._t255_native_imex_24h.toml` at 90 s (config hash `ae84e41e...`, pins hash
`c5d0545d...`, 125 arrays bit-identical across the two trees the merge joined
and across both cards, MEASURED 2026-09-07; the 60 s content it was pinned on
until 2026-09-06, `cf964993...`, reads the same); `imex.DEFAULT_INTEGRATOR` keeps the Eulerian
pair's name because it is the library default of the pins, checkpoint and
migration signatures under which every hash written before this core existed
was computed, not the door's default. The T533 pair was not graded: the
semi-Lagrangian T533 day (24.45 GiB by the sizing door) never had an empty
RTX 5090 in the window, so the T533 default follows the two graded
truncations and the door page says so. By scale
(`tools/arwen_global_spectrum_bands.py`, 24 h, on the model level nearest
500 hPa, which is level 19 at 475 hPa at the reference surface pressure; the
instrument read level 26 at 793 hPa under that label until 2026-09-07, and
these numbers are the re-read of the same checkpoints): the semi-Lagrangian
arm keeps 0.30 of the Eulerian arm's vorticity power at wavenumbers 181 to
230 at T255 (0.34 on the fixed core against the 90 s arm) and 0.22 at T383,
0.34 at 231 to 255 and 0.22 above 255 (0.48 and 0.15 on the fixed core);
doubling the Eulerian step from 60 to 120 s moves the 181 to 230 band by
3 percent and the whole spectrum by 2 percent, so the step is spectrally
free there and the drain is the scheme's. The door page carries the level
scan and the pressure-surface reading beside these. On the Jablonowski
and Williamson steady state at T255 the fixed core reads 0.71 hPa over nine
days on the cubic gather at order 8 and 2,160 s (above), inside the case's
1 hPa bar where the retired arithmetic read 1.153 hPa; the shipped package's
own nine-day arm has not been run. The verify configs are the door, and every speed number on this page
carries the card it was measured on and the tenants it shared it with.

**The Eulerian step is a rule.** The runner refuses an Eulerian step whose
spectral CFL `dt |V|max sqrt(N(N+1)) / a` exceeds `[time] maximum_cfl`
(0.75); the shipped step is the largest multiple of 5 s dividing the hour
whose CFL on the strongest analysis day on disk stays at or under 0.70 of
that gate (`config.default_eulerian_step_s`, applied when `dt_s` is
omitted), and a day stronger than any on disk is refused by name with the
step it would have admitted, in the refusal and in the receipt's `cfl` block.
MEASURED 2026-09-06 over nine GDAS analyses (`tools/arwen_global_cfl_sweep.py`;
the day maximum is the analysis's own jet at step 1 on every day, relaxing to
0.37 to 0.46 of itself by hour 6): the strongest day is 2026-08-31 00Z at
136.9 m/s implied at T255, 139.6 at T383 and 141.3 at T533, so the shipped
steps are 90 s at T255 (0.494 of the gate on that day; 100 s reads 0.732 and
fails the rule), 60 s at T383 (0.504) and 40 s at T533 (0.473; 45 s would
need the day under 139.3 m/s). Until 2026-09-06 the T255 Eulerian config
shipped at 60 s, which used 43 percent of its own refusal on the day of
record; 90 s is 1.5x fewer steps. 120 s at T255 and 80 s at T383 stay
selectable as the steps the gate itself admits on the day of record, with
no margin for a stronger day. The T383 row was read again by one-step
probes of the same nine analyses on the RTX 5090 (MEASURED 2026-09-07):
the strongest day reads 139.0 m/s implied at T383, 0.502 at 60 s against a
gate of 1.0, so 60 s is 0.669 of the shipped 0.75 gate on it and the rule
holds with the derived 139.6 kept as the more conservative ceiling. The
Eulerian configs are `..._t255_native_imex_24h.toml`,
`..._t383_native_imex_24h.toml` and `..._t533_native_imex_24h.toml`; the
configs of record run the default core and name no step.

## Exact spectral diffusion

Level 3's exact exponential hyperdiffusion is applied by total spherical
degree. Divergence, log surface pressure, water species and number moments
each may use a distinct strength multiplier while sharing the registered
transfer function. It is the DEFAULT drain at every truncation and on both
cores, and every graded arm ran with it.

`[diffusion] closure = "spectral_eddy_viscosity"` selects the other operator
in its place: a spectral eddy viscosity derived from the two-point closure
theory of turbulence (`woof.globe.spectral.eddy_viscosity`; EDQNM, Chollet
and Lesieur 1981, and on the sphere Frederiksen and Davies 1997) instead of
a fixed e-folding time. Every level reads its own kinetic-energy spectrum
density at the cutoff out of the model's coefficients each step, averaged
over the last `tail_degrees` degrees compensated along the inertial slope,
and drains at `nu_plus(k / k_c) sqrt(E(k_c) / k_c)` with `nu_plus(x) =
plateau + cusp_amplitude exp(-cusp_decay / x)`. The constants are the EDQNM
values for a Kolmogorov constant of 1.4 (plateau 0.267, cusp 9.21 at decay
3.03) with the eddy Prandtl number 0.6 for the scalars; on a k^-5/3 range
the total drain below the cutoff equals the cascade rate the spectrum
implies. Application is the same exact exponential factor per degree and per
level, unconditionally stable, degrees up to `preserve_degree` untouched,
and the theory constants are exposed as options so their sensitivity can be
measured; at their defaults nothing in it is tuned. The closure's fields
join a configuration's identity only when the closure is selected, so no
record hash moves, and `arwen_global_gdas_t255_native_closure_24h` is the
record T255 experiment with its `[diffusion]` table changed and nothing
else. It has no card run and no observation score of its own: it ships
selectable and ungraded, and the drain of record stays the hyperdiffusion.
The backscatter term of the same closure is not built.

## Large-step order

One model step is transactional at the physics boundary:

```text
reference/native physics half step
    -> positivity and local water repair
    -> semi-implicit half map (square root of Crank-Nicolson on the vertical-mode operator)
    -> SSPRK3/RK4 moist hybrid dynamics on the explicit residual
    -> semi-implicit half map (the external scheme: identity before, Crank-Nicolson after)
    -> exact spectral diffusion
    -> dry-mass fixer
    -> physics half step
    -> positivity repair
    -> global total-water fixer through finite surface reservoir
    -> validation, diagnostics, checkpoint
```

A physics adapter receives a complete immutable exchange and returns a complete
result. Partial scheme output is not committed.

## Reference physics

The included reference suite is deliberately compact but executable:

- gray shortwave and longwave radiation;
- bulk surface heat, moisture, and momentum exchange;
- finite grid-resident surface-water and heat-capacity reservoirs;
- implicit vertical diffusion for wind, potential temperature, vapor, cloud
  water, and cloud ice;
- deterministic dry/moist convective adjustment;
- two-pass saturation adjustment with latent heating;
- warm-cloud autoconversion and accretion;
- rain evaporation;
- temperature-dependent cloud/rain freezing;
- cloud-ice to snow and snow to graupel conversion;
- finite-speed rain, snow, and graupel fallout;
- accumulated surface precipitation by category;
- local atmosphere-plus-surface water closure;
- explicit global water-drift repair recorded as a run tracker.

These equations are a model-integration and conservation reference. They are
not advertised as WRF physics parity or as an operational forecast package.

## Native WOOF physics admission boundary

The following existing WOOF launchers are column-oriented and are plausible
future adapters, but their current wrappers still assume regional `DomainState`
layout, C-grid ownership, terrain-coordinate geometry, scratch slots,
precipitation accumulators, physics scheduling, and restart/output contracts:

- RTE+RRTMGP and legacy RRTMG;
- revised/classic MM5 and MYNN surface layers;
- Noah, Noah-MP, and RUC land models;
- YSU, MYNN, MYJ, Shin-Hong, and SASE PBL closures;
- Kessler, WSM6/WDM6, Thompson variants, Morrison, NSSL-2, Milbrandt-Yau,
  and P3 microphysics;
- Kain-Fritsch and Grell-Freitas convection.

`woof.globe.physics.registry` requires every native adapter to declare:

- exact scheme identity;
- backend and precision;
- required atmospheric, hydrometeor, number-moment, surface, and soil fields;
- pressure ordering and vertical-coordinate convention;
- restart and budget contracts;
- an evidence-receipt SHA-256;
- an adapter-arithmetic SHA-256.

A TOML requesting `physics.mode = "arwen-native"` refuses unless its named
adapter is registered. There is no fallback to the reference suite.

## Surface and water repair

Spectral projection rings a sharp positive grid field below zero, and clipping
that ringing creates water. WOOF global closes the clip inside the atmosphere
(the standard hole-filling fixer):

1. clips the physical water field;
2. transforms and projects it;
3. rescales the positive part of each water species on each level so its
   mass-weighted global integral is exactly what it was before the clip;
4. tolerates the small residual ringing of the rescaled truncated field, which
   every consumer clamps at its own boundary (the exchange clamp applies the
   same per-level rescale before physics sees the fields).

No clamp moves water between the atmosphere and the surface reservoir. The
fixer's per-step magnitude - the global-mean column water the clips created
and the rescale removed, in kg/m2 and relative to the atmospheric column, and
the largest per-level rescale - is a step metric, a receipt tracker, and a
receipt gate (`positivity_fixer_max_step_relative`, measured 8e-5..8e-4 on
the T63 real-analysis reference).

A separate global water fixer removes remaining long-integration transform and
mass-coordinate drift through one spatially uniform surface-reservoir
correction. Its maximum absolute correction is checkpointed and receipted.
This is an explicit numerical source/sink accounting mechanism, not an
unreported conservation claim.

## Checkpoints and receipts

Every checkpoint binds:

- exact resolved config hash;
- WOOF global and inherited Level-3 pins;
- model step and time;
- every atmospheric spectral array;
- every grid-resident surface and soil array;
- shape, dtype, and SHA-256 for every array;
- whole-run maxima for CFL, dry-mass repair, total-water repair, positivity
  repair, semi-implicit increment, and reference-physics water repair;
- metadata self-hash.

Restart validates all of those identities before constructing model state.
The cold deterministic initial state remains the conservation target, so a
checkpoint cannot redefine the mass or water quantity the run claims to
preserve.

A terminal run receipt contains transform controls, start and final
measurements, all gates, inherited trackers, checkpoint identities, physics
identity, and a self-hash. A numerical exception after output setup writes a
self-hashed `status = "error"` receipt and then re-raises the original error.

## Parent export

`export-parent` samples a checkpoint onto a pole-excluding, cell-centred
regular latitude/longitude grid. It exports pressure, potential temperature,
temperature, geopotential, height, vector wind, vorticity, divergence, all
six water species, all five number moments, terrain, and grid-resident
surface/soil state. The reader refuses an export whose inventory is missing
any of them.

The export is self-hashed and tamper-evident. It is explicitly a neutral parent
artifact, not an admitted regional `wrfinput`, `wrfbdy`, or nested boundary
stream. Conservative remapping, vertical interpolation, wind rotation,
regional staggering, clocking, and boundary construction remain a separately
gated adapter task.

## Commands

The product door, `woof global`, carries the three legs a run is made of.
The complete route it belongs to, with the data acquisition in front of it
and the render behind it, is
[ARWEN_GLOBAL_QUICKSTART.md](ARWEN_GLOBAL_QUICKSTART.md).

```bash
woof global run arwen_global_t255_quickstart \
  --outdir out/arwen-global-24h

woof global export arwen_global_t255_quickstart \
  out/arwen-global-24h/arwen_global_step*.npz \
  --outdir out/arwen-global-tapes --start-date 2026-08-30_18:00:00

woof global assimilate CONFIG CHECKPOINT --obs OBS.csv --out analysis.npz

woof global cycle CONFIG --obs OBS.csv --outdir out/cycle   --cycles 24 --interval-s 3600 --start-utc 2026-08-31T00:00:00Z
```

The assimilation leg is a data-density-normalised successive correction
against one deterministic background: per column the increment is
`sum(w g d) / (1 + sum(w g))`, whose single-report limit is the optimal-
interpolation gain and whose co-located stacks saturate as OI's do; there
is no observation-observation solve, so disagreeing neighbours are
under-fitted (two opposite-sign reports one length scale apart: 0.20 of
the innovation at the report against OI's 0.52).  Reports are compared at their own height: surface
pressure reduced to the station, temperature to 2 m, and surface wind to
the 10 m anemometer by the same Monin-Obukhov similarity diagnostic that
writes U10/V10 to the render tapes (the lowest full level is 23 m up on the
default grid and is never compared raw).  Horizontal weights are Gaussian
in great-circle distance with a hard cut at 3.5 length scales.  Vertical
weights: a surface report decays over 2 km above the column's surface; an
aloft report carries the ADAS height-separation model
`exp(-(dz / 1000 m)^2)` with `dz` the ln p separation on a 7.6 km scale
height, zero below the 1e-3 floor (2.63 km-equivalent), so a 250 hPa
aircraft report reaches 177-353 hPa and nothing above 100 hPa.  Rings the
dycore's top absorber owns (ring-mean pressure below the sponge base, 50
hPa by default) accept no increment from a report that is not itself in
that region.  Wind increments are spread as u and v, analysed through
vorticity/divergence, and applied as their rotational (streamfunction)
part only: spreading two scalars puts 40-51 % of the increment's kinetic
energy into divergence (T63, one u-only report 0.50), and that divergence
is gravity-wave energy the dycore keeps rather than anything a report
measured - integrated one hour on the T3 smoke configuration from a
wind-only analysis, the unconstrained increment left 1.17 J/kg of excess
divergent kinetic energy over the control, the rotational projection of
the same increment 0.043 J/kg (T21: 1.62 against 0.074).  The report
carries the analysed and applied divergent fractions; `--wind-balance
unconstrained` applies the predecessor for measurement.  Every increment
passes the model's triangular truncation; the global-mean surface
pressure is preserved for the mass fixer.

The gate of record is cross-validation: for every variable with at least
50 accepted reports a seeded tenth of them is withheld from the analysis,
and O-A rms must fall below O-B rms on those withheld reports.  Judged on
the assimilated rows themselves the old gate passed for any positive
gain: reports made of white noise at ten times the table errors passed on
every variable while 3.1 K and 10.4 m/s of that noise went into the state
(smoke configuration; on T21 with 400 stations and a 1500 km scale the
assimilated rows still "improve" on pure noise at every background-error
multiplier from 0.1 to 1000 while the withheld forty fit worse on three
variables of four).  The receipt records the assimilated and withheld
identities and both fits; the assimilated-row numbers are diagnostics.

Every assimilated report's identity (source, platform, valid instant,
variable, position and level, hashed) is written into the analysis
checkpoint's physics-state metadata as well, so a cycled background
carries its own assimilation chain: a later cycle refuses a report the
chain already holds (`rejections.already_assimilated`), analyses only
what it has never seen, and refuses outright when nothing new survives.
Before this, the same hourly reports offered every 15 minutes inside the
90-minute age window were re-accepted at full gain up to seven times, and
the temperature O-B rms against reports that never changed fell 1.21 ->
0.51 -> 0.42 -> 0.41 K cycle over cycle; now it moves once (1.21 -> 0.52
K), once more by the previously withheld tenth (0.46 K), and then stands.
Chain entries older than the age window at an analysis time are dropped,
since they fail that window at every later cycle.

The moisture update is the door's one divergence from its v1 form, which
left water untouched.  A dewpoint report (the 2 m dewpoint of the IEM ASOS
record, or an aloft dewpoint) is compared against the dewpoint of the
model vapor at the report's pressure: the lowest full level's vapor at
the surface pressure reduced to the station for a surface report, the
profile interpolated in ln p for an aloft one.  The innovation is spread
by the same successive correction with a vertical localization of its own,
`exp(-z / 1500 m)` above the column's surface (the mixed layer the 2 m
dewpoint samples; the temperature's 2 km reaches into the free
troposphere), and at every level the background dewpoint, from the vapor
there through Bolton's relation, moves by the spread increment and the
vapor is re-derived through the same relation, so the specific-humidity
increment is the dewpoint change through the local Clausius-Clapeyron
slope and a zero change is exactly zero vapor.  Before the increment
enters the spectral basis it is capped at saturation against the analysed
temperature (a point already above saturation is not moistened) and
floored at zero; after the truncation the model's own positivity repair
closes what rang below zero inside its column, and the report records the
capped and floored points, the newly supersaturated points the truncation
left, and what the repair moved.  The dewpoint's background error is 3 K
against the reports' 1.5 K, and it carries its own withheld-tenth gate.
The update is selectable (`--moisture-update on`) and off by default:
graded whole on the 24 h hourly T255 cycle against the CONUS stations
(the same cycle with and without it) it took the 18 h dewpoint from
-2.76 / 5.68 to -0.97 / 3.59 K bias / rmse and the sea-level pressure
from 2.70 to 4.19 hPa rmse, beyond the admission rule, so off leaves
water untouched as the v1 door did and on is a measured workaround for
the dewpoint alone.  The `iem-asos-csv` table entry decodes the ASOS
download the observation scorecard scores against, so the same record
feeds both.

A cycling segment is `--until-s`: the integration ends at that model
time with the config identity unchanged (the receipt records
`segment_until_s`), and the next segment restarts from the checkpoint it
ends on; a 48 h config carries a 24 h hourly cycle and the forecast from
its final analysis on one checkpoint lineage.  A restart from a checkpoint
that carries an assimilation chain opens a new conservation epoch: the
mass and water targets become the restart state's own global means
(recorded as `restart_targets` beside the cold start's), because an
analysis increment is a deliberate source and the fixers would otherwise
remove it on the first step and trip their absorption gates (the moisture
update's 0.15 kg/m2 read 2.1e-4 of the column against the 1e-5 gate on
the T255 cycle).  A restart without a chain keeps the cold start's
targets bit for bit.

The research surface is the module door, and it is unchanged:

```bash
woof global pins
woof global physics-manifest
woof global transform-check --truncation 63

woof global run \
  arwen_global_moist_smoke \
  --outdir out/arwen-global-smoke

woof global run \
  arwen_global_hybrid_6h_reference \
  --outdir out/arwen-global-six-hour

woof global export-parent \
  CONFIG CHECKPOINT parent.npz --nlat 361 --nlon 720
```

`tools/arwen_global.py` delegates to the same CLI from a source checkout.

## Current evidence

The recovered release gates:

- Level-3 scalar and vector transform controls;
- hybrid-pressure monotonicity and continuity closure;
- hydrostatic monotonicity;
- reference-physics local water closure and transactionality;
- pre-update CFL refusal;
- checkpoint and parent-export tamper detection;
- self-hashed success and error receipts;
- bit-exact midpoint restart including inherited trackers;
- native-adapter refusal;
- a four-step moist smoke campaign;
- a 360-step, six-hour T7/eight-layer reference-physics campaign.

The six-hour control completed with maximum spectral CFL below 0.023, exact
reported global-water closure after the explicit fixer, and mass drift near
machine precision.

## Non-claims

This release does not claim:

- operational forecast skill;
- GPU validation without a target-device receipt;
- direct reuse of unadapted regional WOOF physics wrappers;
- a complete energy/angular-momentum-conserving hybrid vertical scheme;
- the T533 grade of the two cores (the semi-Lagrangian T533 day was not run;
  the T533 default follows the two graded truncations);
- a fully implicit three-dimensional primitive-equation solver;
- conservation without the explicitly measured repair mechanisms;
- an admitted global-to-regional forcing path;
- replacement of the regional nonhydrostatic model.
