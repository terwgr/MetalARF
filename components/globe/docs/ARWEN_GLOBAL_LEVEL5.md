# WOOF global Level 5

## Status

Level 5 turns the Level-4 moist hybrid global model into an experimental
**global parent system** with two new production-shaped seams:

1. a fail-closed adapter that can execute selected existing WOOF CUDA column
   physics on the global Gaussian grid; and
2. a hash-bound global-to-regional translation path that produces the exact
   coupled fields consumed by WOOF's existing specified lateral-boundary
   machinery.

It remains research-only. Every run still requires:

```toml
acknowledgement = "research-only-arwen-global-v1"
```

The native adapter additionally requires:

```toml
acknowledgement = "device-pending-arwen-native-physics-v1"
```

Level 5 does not alter an ordinary regional WOOF run. A validated parent
frame affects a regional state only when a caller explicitly invokes the
install/attach API.

## Level-5 identity

The Level-5 pin document binds:

- 15 spectral atmospheric fields;
- explicit Level-4 checkpoint migration;
- native-physics scheme order and vertical conversion;
- persistent native scheme state;
- the native transaction and budget contract;
- target-device qualification rules;
- complete parent exports;
- horizontal and vertical translation methods;
- earth-to-grid wind rotation and C-grid staggering;
- WRF coupled-unit construction;
- lateral-side ordering; and
- attachment through the existing WOOF `LateralBoundaries` classes.

Print the exact identity with:

```bash
woof global pins
```

The delivered identity is recorded in every Level-5 checkpoint, receipt,
parent export, translated frame, target, parent series, migration receipt,
and native-device evidence file.

## Expanded moist state

The spectral atmosphere now carries:

\[
\zeta,\;D,\;\theta,\;\ln p_s,
q_v,q_c,q_r,q_i,q_s,q_g,
n_c,n_r,n_i,n_s,n_g.
\]

The five number moments are transported, checkpointed, exported and
restarted as first-class prognostic fields. They are not hidden inside a
physics wrapper.

**Representation (since 2026-09-02).** The dynamical quartet and water vapor
live in the spectral basis. The five condensate species and the five number
moments are grid-point fields on the Gaussian grid: they are carried once per
step by a positive-definite, directionally split, van Leer limited flux-form
transport (`woof.globe.transport`) driven by the step's own mass
fluxes and pressure velocity, and they are never analysed into the spectral
basis. Positivity holds by construction, the transport's limiter is their
only dissipation, and no clamp or rescale ever touches them. Vapor, the one
water field that still rings below zero in grid space, is clipped at the
physics boundary and after each repair and closed inside its own column
(each column's vapor integral is unchanged by the clip; no other column and
no reservoir pays). Checkpoints carry the grid tracers as real
`(nlev, nlat, nlon)` arrays (schema v3); a schema-v2 checkpoint, whose ten
tracers were spectral coefficients, can be inspected but not resumed.

Why: with every hydrometeor a truncated spectral tracer, the positivity
machinery that kept those fields nonnegative rescaled every column on a level
by the level's clipped-to-unclipped mass ratio, moving the mass the clip
created in one feature's negative lobes out of every cloud and rain shaft on
the planet into clear air, four passes per step, where the microphysics
evaporated it. Measured at 52 km: 37 percent of the world's cloud water per
repair pass, grid-scale surface precipitation 0.008 kg/m2 in 24 h, total
precipitation equal to convective precipitation to four decimals.

Level-4 checkpoints contain the six water masses but not the number moments.
Level 5 therefore refuses them by default. Migration is explicit:

```bash
woof global migrate-level4-checkpoint \
  TARGET_LEVEL5_CONFIG.toml \
  level4-checkpoint.npz \
  level5-checkpoint.npz \
  --receipt level5-checkpoint.migration.json
```

The migration validates the old checkpoint's schema, pin, config identity,
metadata hash and every array hash, then seeds all five moments to exact zero
and records that decision. A migration into native Morrison physics is refused
unless `--allow-native-zero-moments` is stated, because zero moments are a
scientific initialization choice rather than a neutral file-format change.

## Native WOOF CUDA physics adapter

### Admitted suite

The built-in adapter is:

```text
arwen-cuda-column-suite-v1
```

It calls existing WOOF modules in this fixed order:

```text
RTE+RRTMGP
    ↓
MM5 SFCLAY
    ↓
Noah LSM
    ↓
YSU PBL
    ↓
cumulus: Grell-Freitas (default) or New Tiedtke (`cumulus = "ntiedtke"`);
         `cumulus = "none"` removes the slot
    ↓
Morrison two-moment microphysics
```

The adapter uses the current WOOF launchers rather than copied equations:

- `woof.globe.core.rrtmgp.RRTMGPRadiation`;
- `woof.globe.core.sfclay.launch_sfclay`;
- `woof.globe.core.noah.launch_noah`;
- `woof.globe.core.ysu.launch_ysu`;
- `woof.globe.core.gf.GrellFreitas` (the regional `cu_physics=3` seam over
  `woof/globe/core/kernels/gf.cu`) or `woof.globe.core.ntiedtke.NewTiedtke`
  (the regional `cu_physics=16` seam over
  `woof/globe/core/kernels/ntiedtke.cu`); and
- `woof.globe.core.morrison.launch_morrison`.

The cumulus scheme reads the column as WRF's cumulus driver does: the state
before the call's radiation and PBL forcing with those two rates as its
forcing lanes, the pressure velocity the dycore's continuity closure
diagnoses (`w = -omega / (rho g)`), the terrain height, YSU's PBL-top index,
the surface fluxes and the land flag. Its heating, drying and detrainment
integrate into the column; its convective rain is credited to the surface
reservoir at the measured column loss, and one number, the scheme's reported
rain capped at that loss, is accumulated as `RAINC` on the render tape and
handed to Noah in the land bucket. Grell-Freitas's deep arm reports more rain
than its own tendencies remove (1.27 times on WRF v4.6.1's oracle columns,
1.77 times over the globe and 2.28 times over land in the first day of a T255
GDAS forecast), so neither the land nor `RAINC` carries the excess; each call
reports it as the diagnostic `cumulus_rain_withheld_kg_m2`, beside
`cumulus_rain_reported_kg_m2` and `cumulus_water_removed_kg_m2`. The advective forcing lanes
(WRF's `RTHFTEN`/`RQVFTEN`) are the dynamics' own theta and vapor
tendencies over the previous dynamics interval, measured by the runtime
between its two calls of a step and checkpointed with the physics
namespace; they lag WRF's stage-1 reading by one interval. No unlisted
scheme is silently replaced by reference physics.

The two schemes are fed through one step and differ in two stated places:

- Grid spacing. Grell-Freitas reads the scalar `dx_m` option (the value its
  baselines and its bit-identity anchor were recorded with). New Tiedtke is
  handed each column's own spacing, `sqrt(dx * dy)` on the Gaussian grid
  (`dx = R cos(lat) 2 pi / nlon`, `dy` the ring's meridional extent), because
  its deep closure is scale-aware through `log(dxref / dx)` per column and the
  zonal spacing at 60 degrees is half the equatorial value; a scalar would
  hand a polar column the equator's adjustment time.
- Momentum. New Tiedtke returns convective `u`/`v` tendencies (WRF's
  `lmfdudv` is a parameter of that scheme, not an option) and they reach the
  wind on the lane YSU's `du`/`dv` already use. Grell-Freitas returns none and
  the wind is untouched. A result carrying one component and not the other is
  refused by name.

`gf_resolved_convergence_closure = true` (default `false`, read only under
`cumulus = "gf"`) runs the coarse-column form of the Grell-Freitas deep arm,
a declared divergence from WRF v4.6.1 for a 52 km column that carries the
whole of deep convection as resolved forcing
(`woof/globe/core/kernels/gf.cu`, `GF_RESOLVED_CONVERGENCE_CLOSURE`): the
per-level 300.01 K/day heating cap that scales every deep tendency yields
to the latent heat of the column's own resolved moisture convergence (the
cap becomes the larger of the two,
so a column under 300 K/day is untouched); the downdraft sweeps continue
past levels the downdraft's profile reaches with no mass, carrying the
environment there, where WRF divides by zero and hands the column to the
grid scale with exit 51; a downdraft that cannot form (exit 7, the
saturated deep column) leaves the updraft running; and the cloud-base mass
flux is floored at the Kuo moisture-convergence member of the ensemble
(`mconv / den / pr_ens[7]`, sig-scaled like the mean; the scheme's own
member 7) times the Kuo-Anthes share of the converged moisture that rains
out, `1 - b` with `b = min(1, max(0, 10 (1 - RH)))` and `RH` the
pressure-weighted mean relative humidity of the cloud layer (Anthes 1977
with n = 1 and the critical humidity at 0.9): a saturated column
precipitates all of the moisture the grid converges into it, a column at 95
percent half, one under 90 percent none, the rest moistening the column as
WRF's members already do. The critical humidity is measured, not Anthes's
0.5 for MM4's grids: on the 2026-09-01 control's hour 12 under the 0.5 form,
82 percent of the rain the floor added came from columns whose cloud layer
sat under 90 percent relative humidity, columns the ensemble moistens and
which never became grid-point storms, while the storms sat at 96 to 100
percent. The sixteen-member mean
gives the Kuo member a quarter of the weight, a hedge that is right where
the grid resolves part of the convection and carries the rest of the
converging moisture itself, and wrong on a saturated 52 km column that
resolves none of it: without the floor the remainder reaches Morrison as a
grid-point storm. The closure ensemble, its members and their weights are
otherwise WRF's; the cap and the floor both scale with the column's own
resolved moisture convergence, the one number the scheme already integrates
per column, and with nothing tuned to a grid spacing. `false` is WRF's kernel word for word and keeps the identity every
earlier Grell-Freitas checkpoint carried; `true` joins the hash. The
arm is an opt-in, a workaround for the grid-point storms and not a fix, by
its grade of 2026-09-05 on the 2026-09-01 00Z T255 case against the
control: on it takes the grid-scale cells above 200 mm per day from 27 to
0 and the heaviest cell from 448 to 177 mm and keeps every surface and
upper-air score inside 0.03, but takes the northern 250 km kinetic-energy
ratio from 0.825 to 0.792 (0.034, past the grade's 0.03 bar), the global
grid-scale rain from 1.500 to 1.187 mm per day (0.313, past its 0.3 bar)
and the CONUS diurnal composite only from 12.59 to 13.00 LST against
Stage-IV's 14.60. The surface gains the same grade shows (CONUS T2 rmse
2.548 to 2.481 K, MSLP rmse 1.676 to 1.575 hPa) belong to the dynamics
forcing lanes the tree carries with the arm off: against that baseline the
arm reads 2.477 to 2.481 K and 1.588 to 1.575 hPa, 0.806 to 0.792 on the
northern 250 km ratio, 1.418 to 1.187 mm per day of grid-scale rain and 19
to 0 cells above 200 mm. The floor is compared with the ensemble mean after
the diurnal-cycle term is taken off it, so a column that term silenced (a
zero request, exit 19 in WRF) convects on the floor alone: 1,298 of the
15,357 deep-active columns at the arm's hour 12, named by the census as
`floored.columns_with_zero_request`. A floored column is one the kernel
floored (floor above request on a deep-active column), 3,063 at that hour
where the census's earlier row, which demanded a request and an applied flux
above it, read 1,750. The per-column closure reading the kernel exports
under both (requested against applied cloud-base mass flux, the floor, the
four family requests, the resolved convergence, the cap used, the exit) is
read by `python -m woof.globe.massflux_diagnostic --read CONFIG
CHECKPOINT` (`--calibrate` prints the census calibration rows, the floor and
downdraft rows among them). The regional `cu_physics = 3` seam keeps the
WRF-faithful kernel, its parity anchor.

`ntiedtke_tiedtke_closure = true` (default `false`, read only under
`cumulus = "ntiedtke"`) runs the New Tiedtke kernels with classic Tiedtke's
deep closure; `docs/cumulus-new-tiedtke.md` records what that changes. The
flag joins the adapter identity only under the scheme that reads it, so a
Grell-Freitas checkpoint keeps its hash and keeps restarting; the two New
Tiedtke closures never share a hash. `cumulus_column_chunk` bounds the
per-pass column packing of whichever scheme fills the slot, and the device
peak in the receipt is measured at the allocator, so the scheme's workspace
is counted whatever the chunk.

#### YSU above the boundary layer: the mixing length is not the layer thickness

WRF's YSU sets the asymptotic mixing length of its local Richardson-number
diffusivity above the boundary layer from the layer thickness,
`rlamdz = min(max(0.1 dz, 30 m), 300 m)` (`bl_ysu.F90:1003`, the same line
in `kernels/ysu.cu`), so `K = l^2 |dV/dz| f(Ri)` grows with the square of
the spacing for the same resolved shear and Richardson number. The 40-level
stack puts 55 hPa layers (about 1500 m) across the jet, where a regional
column has 300 m: the rule reads 150 m against 30 m, a 25x diffusivity, and
the energy ledger read the physics draining the 100 to 400 km kinetic energy
at 237 hPa at 0.7 to 1.2 of the band's energy per day, YSU's momentum lane
being the only physics that moves wind there. The native option
`ysu_free_atmosphere_mixing_length` (`woof.globe.core.ysu_contract`) selects the
rule: `"wrf-layer"`, the default, is WRF's rule, bit for bit; `"fixed"`
holds the length at WRF's own `rlam = 30 m` at every spacing (the value
WRF's rule gives every layer thinner than 300 m). Nothing inside the
boundary layer changes under either (the profile K, the countergradient and
entrainment terms are untouched), the regional model has no such option and
keeps WRF's arithmetic, and the option joins the config identity only as
`"fixed"`, so every checkpoint written under WRF's rule keeps its hash, bare
or spelled. `"fixed"` is a declared divergence from WRF (the free-atmosphere
flux of a resolved shear no longer grows with the vertical spacing) and it
is selectable, not default: it is reported as a workaround, because its
grade below lost two of the lines the rule that admits a default asked for.

The hash-stable spelling had a trap, found when the probe restarted the
control's f012 checkpoint under a config spelling `"wrf-layer"`: the restart
was admitted (the hash matched) and the kernel ran `"fixed"`, its mean K at
every interface equal to the fixed-length formula. The config had kept only
the identity payload of the options, which drops the key under WRF's rule,
and the suite rebuilt from that payload read the dropped key as its default.
The registry now separates the two roles (`woof.globe.physics.registry`:
`options_validator` returns every normalized option, what the suite is built
from; `options_identity` reduces them to the hash payload), the config's
identity applies the reduction and the bridge's receipt identity carries it,
so every config hash is unchanged and the bridge the runner builds reads the
spelled rule. Nothing that runs a scheme may be built from an identity
payload again: `NativePhysicsOptions.normalized` is the build payload,
`NativePhysicsOptions.identity` the hash payload, and the door test loads the
smoke config bare, with `"fixed"` and with `"wrf-layer"` and holds the two
apart.

The grade of `"fixed"` (one 24 h arm on the control case, scored with the
control's chain unchanged plus the upper-air scorecard; recorded
2026-09-05, every number recomputed from the score files by a second
reading): the northern 250 km ratio at 250 hPa
moves from 0.670 (the tree before this change) and 0.655 (the order 8
package, the arm's other reference) to 0.850, 55 percent of the gap to 1,
0.015 above the half-gap mark with a sample standard error of 0.023 over the
13 hourly samples; the northern effective resolution reads 185 km in 13 of
13 samples (package 197 km); the southern spectrum falls to half of observed
at 236 km instead of 410; the grid-limit ratio is 0.053 against the
package's 0.038. Every surface score against the GFS f024 analysis sits
within 0.026 K, 0.024 hPa and 0.017 m/s of both references, the water-budget
residual is 0.16 percent of max(E, P), and every receipt gate passes. On the
upper-air scorecard the 250 hPa speed bias at 24 h improves (northern -0.711
to -0.669 m/s, tropics -0.621 to -0.438) while the northern 250 hPa vector
rmsve worsens by 0.16 m/s (4.41 to 4.56 against the GFS f024, 4.69 to 4.85
against the GDAS analysis; the arm is better than the control through hour
8 and the gap opens monotonically from hour 9), the northern 250 hPa speed
rmse by 0.05 m/s and the northern Z500 rmse by 0.14 m. The rule that admits
a default asked for the grid-limit ratio not to rise above the package's
and for the W250 rmsve not to worsen, and the arm lost both lines, so WRF's
rule stays the default and `"fixed"` stays selectable. What would settle
the loss: a scale-filtered rmsve (does the added 250 to 1000 km energy
carry a wrong phase, or only variance the reference does not hold), a
diffusion e-folding retuned with the 30 m length (the tail rose because the
order 8 operator was set against WRF's drain), and a 48 h arm (the 250 km
band was still filling at hour 24). The ledger at 237 hPa reads the physics
draining the 330 to 200 km band at 0.27 of its energy per day where the
control and package read 1.19 and 1.10, four times less at every band from
660 km down, with the band's energy up 36 percent and the sum over
operators at 250 km still +0.38 per day over hours 12 to 24. The regional
path is proven unchanged: the WRF v4.6.1 fixture parity gate holds its
recorded ULP table on the card, the kernel under WRF's rule is byte-identical
to the reference kernel on every output of the 24-column WRF fixture, and
a 10-step cold start of this tree spelling WRF's rule is byte-identical to
the order 8 package's tree in every state array.

The probe behind it, `woof.globe.pbl_free_atmosphere`, restarts a run
from a checkpoint, captures the first YSU call and reads the K profile, the
formula term by term (thickness, shear, Richardson number, both lengths, the
stability functions, the floor and cap) with the kernel's word checked
against the recomputed formula on every free-atmosphere interface outside
the entrainment zone, the per-level and per-band kinetic tendency of the
call, and exports columns for `tools/ysu_wrf461_oracle/run_bl_ysu_columns.F90`,
which drives the byte-unmodified Fortran on them. Its calibration, two
synthetic families in both directions: a planted jet reads its Richardson
number back to the scheme's own 1e-9 shear floor and its K to 2e-16 of the
formula under both rules, the rules differ by exactly the square of their
Blackadar lengths, a stable shear-free column reads exactly the 0.1 m2/s
floor and moves nothing, a neutral shear-free layer reads the floor plus
`rl^2 sqrt(1e-9)` and nothing else, and a single-degree tendency lands in
its spectral band alone.
`gf_updraft_only_when_downdraft_dry = true` (default `false`, read only under
`cumulus = "gf"`) keeps the Grell-Freitas deep arm running updraft-only in
the columns WRF's GFDRV rejects because their downdraft cannot form (the
ierr 7 exits and cup_dd_moisture's ierr 51 exit); the overridden exit is
exported per column beside the deep exit code. It is an opt-in workaround,
not a fix: on the 2026-09-01 00Z T255 control it halves the grid-scale
cells above 200 mm/day but takes the northern 250 km and grid-limit energy
ratios 0.034 and 0.039 below the control and leaves the CONUS diurnal
composite at 12.9 LST against Stage-IV's 14.6, and the grid-point storm
mechanism (resolved convergence feeding a saturated 52 km column) survives
it. Only `true` joins the adapter identity, so the default keeps the hash
every earlier Grell-Freitas checkpoint carried.

### Radiation as the global core runs it

RRTMGP fires every `radiation_interval_s` (the GDAS verify configs set
3120 s, the regional one-minute-per-km convention at 52 km; the adapter
default is 1800 s) on the model clock's bucket, so an interval that is
not a whole number of steps fires on the first step past each bucket
edge (3120 s at dt 50 alternates 3100 and 3150 s). The planes it returns
are held and applied every step until the next bucket.

The cloud optics reads the grid tracers: WRF's `cal_cldfra1` cloud
fraction from the checkpointed vapor, cloud water, ice and snow; in-cloud
paths from the layer masses; Morrison's post-update effective radii
(`effc`, `effi`, `effs`) when the scheme has run, the number-moment
reconstruction before its first call. Ice and snow enter RRTMGP's one ice
species each at the solid-ice effective diameter that carries its area
per unit mass, `2 re_x rho_x / 917` (the kernel's cloud-ice 500 and snow
100 kg/m3 against the table's solid ice, whose extinction at 10 um is the
geometric `3 / (rho D)` of 917 kg/m3 ice), merged at the area-conserving
size `(qi + qs) / (qi / d_i + qs / d_s)`. A size outside the loaded
tables' domain (2.5 to 21.5 um
liquid radius, 10 to 180 um ice diameter in the shipped v1.9 tables) is
counted and carried: above the upper bound the in-cloud path is scaled by
bound/size at the bound's optical properties (the geometric-optics limit,
where extinction per unit mass goes as 1/size), below it the size is
clipped. The WRF-transcribed explicit-radius (WSM6, Thompson, NSSL) and
P3 couplings clip at the bound instead, as `module_ra_rrtmg` does past
its own cap, because their arithmetic is fixture-pinned to WRF's size cap
and snow discount. The counts ride the checkpoint metadata and the
suite's diagnostics (`radiation_size_bounding_*`), each with the share of
the in-cloud path the counted cells held and the cloud-fraction-weighted
share, which is their radiative weight: the in-cloud path is the grid-mean
path over max(0.01, cloud fraction) and the McICA generator samples a cell
cloudy with the probability of its fraction, so a cell of zero fraction is
counted (`*_unsampled_cells`) and never radiates.

The radiation scorecard reads a finished run:

```
python -m woof.globe.radiation_scorecard --run-dir DIR --out JSON
    [--reference-grib f000,f024 --reference-step 0 --reference-step N]
    [--window-hours 12 24] [--label TEXT]
python -m woof.globe.radiation_scorecard --calibrate
```

It reports, per checkpoint and per region (global, land, ocean, six
latitude bands), the surface net shortwave, the surface downward and
upward longwave, the outgoing longwave, the reflected and incoming
shortwave, the derived surface and top-of-column net radiation and
the atmospheric column's net radiation, the column cloud cover and the daylit fraction,
with interval means from the time integrals (or from the held planes at
checkpoint times for an archive without them, and the JSON says which);
the cloud-optics view above under the shipped coupling and under the
pre-2026-09-04 one (number-weighted merge, bare clip) so the counts read
before and after; the model's cloud cover against the instantaneous
total, low, middle and high cover of the GFS/GDAS pgrb2 products the arms
read; and the fluxes against annual global-mean climatological ranges
with the caveat that a single forecast day is not an annual mean. The
case files' radiative fluxes are interval-averaged GRIB2 products the
mapped engine cannot bind, so no flux reference is read from them.

### Vertical and memory boundary

The global spectral model stores levels from top to surface. Existing WOOF
column kernels consume C-contiguous FP32 arrays from surface to top. The native
batch adapter owns the conversion:

```text
spectral coefficients
       ↓ inverse transform
Gaussian grid, top → surface
       ↓ explicit reversal and contiguous FP32 packing
WOOF native columns, surface → top
       ↓ existing CUDA schemes
native output, surface → top
       ↓ explicit reversal and validation
Gaussian grid, top → surface
       ↓ forward transform
spectral state
```

The reversal, pressure/interface ordering, dimensions and precision are part of
the adapter arithmetic identity.

### The four memory levers

Four settings decide how much of the card a run holds while it computes the
same forecast. They live in a `[memory]` table and each has a `woof global
run` flag that overrides it; a run without the table and without the flags is
byte-identical to every run written before the table existed.

```toml
[memory]
spectral_chunk = 6        # fields per transform call
synthesis_memo = true     # serve repeated syntheses of one state from a memo
legendre_band = 32        # orders per packed Legendre band
streaming = false         # hold no Legendre table at all
```

Two of them are measured bit-neutral and two are measured arithmetic, and the
identity treatment follows the measurement rather than the intention.

| Lever | What it trades | Bits | Identity |
|---|---|---|---|
| `spectral_chunk` | a narrower stack bounds the complex Fourier temporaries a wide one materializes, at one more pass of the per-order loop | the Legendre GEMM's M dimension. Bit-neutral at forty levels (T255 and T533 on an RTX 5070 Ti, and T21 on numpy); the synthesis moves 1 ulp at a two-, five- or ten-level ladder | carried when it is not 6 |
| `synthesis_memo` | off recomputes every synthesis, which is the memory-tightest form | identical. A ten-step T255 native A/B is byte-identical across all 126 checkpoint arrays, metadata included | never carried |
| `legendre_band` | a narrower band holds less expansion scratch and runs more, smaller GEMMs | the GEMM's batch count on cupy, and **arithmetic** there: at T255 float32 a band other than 32 changes the analysis of a single plane by up to 7.6e-06, and a ten-step run at band 16 left 66 of the 125 checkpoint state arrays differing | carried when it is not 32, in the config AND in the transform identity |
| `streaming` | holds no Legendre table at all, so a truncation whose tables do not fit still runs, at 37.7x a resident synthesis and 15,144x a resident analysis | identical at the transform's own band, for a single plane and for a forty-level stack | never carried |

A checkpoint written under a lever that is carried will not restart under a
different value of it, by name, which is the point: the two runs computed
different states.

Measured table floor at float32, the part of the card that never divides with
any decomposition (RTX 5070 Ti, all three tables resident, 2026-09-06):
0.158 GiB at T255, 0.514 at T383, 1.352 at T533 and 4.463 at T799.

### Streaming grid space: `latitude_bands`

A truncation whose grid state does not fit the card no longer has to fit it.
Grid space is cut into latitude bands and streamed through the card a band at
a time; **spectral space stays whole**, and the two meet at a full-latitude
Fourier buffer. The Legendre contraction therefore runs at exactly the operand
shapes it runs at today -- `K = N = nlat` -- whatever the band count, which is
why a banded run returns the resident run's bits.

```toml
[memory]
latitude_bands = 0     # 0 = the sizer chooses, 1 = the resident run
```

```
woof global run <config> --latitude-bands 8
```

**It buys capacity, not speed.** Banding removes no work and changes no
tendency. A truncation that already fits gains nothing from it and pays a
small streaming cost; a truncation that does not fit runs.

**Zero is the default and the sizer chooses.** A bare run prices itself
against the card's free VRAM and takes the resident run when that fits and the
smallest band count that fits when it does not, so a shape the card cannot
hold resident starts rather than dying in the allocator several minutes in.
The door's memory gate prices the band count the run will take, not the
resident peak it will never reach, and refuses on a genuine out-of-memory
prediction: the count it priced does not fit the card's free VRAM at all.
When no count fits inside the run's three-quarter budget the sizer takes the
widest the grid allows and the gate weighs the resident peak, because a gate
that fired on its own margin would refuse runs that fit.

**It enters no identity, at any value.** `config_hash`, `pins_hash` and
`transform.identity_hash` are the same at every band count, so a banded run
shares a config hash, a checkpoint lineage and a receipt with the resident run
and a checkpoint written under one restarts under the other. The receipt
records the schedule and who chose it under `latitude_bands`.

| Gate | Case | Result, MEASURED 2026-09-06 |
|---|---|---|
| BIT-1, BIT-3, BIT-4 | T255 L40 native five-scheme, ten steps, checkpoints at steps 0, 5 and 10, RTX 5070 Ti | 292 checkpoint arrays byte-identical at band counts 1, 4, 8 and 16, and the resident run repeated itself exactly; one config hash across all of them |
| BIT-1, BIT-4 | the shipped door, one model hour at T21 L10 | 85 checkpoint arrays byte-identical between the bare run and `--latitude-bands 4`, with equal config, pins and transform identity hashes and equal final diagnostics |
| BIT-1, BIT-3, BIT-4 | the shipped step at T21 L10, three steps, numpy float64 | 44 arrays -- the five spectral fields, the ten grid tracers, the surface reservoirs and the physics namespace -- and every scalar metric, byte-identical at band counts 1, 2, 3, 4 and 8 |
| HALO-1 | the meridional tracer sweep behind its deep halo, numpy float64 | every advanced tracer, the pseudo-density and every metric byte-identical at band counts 1 to 8, two sweep orders and two Courant limits, at halo depths of 4, 14 and 62 latitude rows |

**A band is a slab, not a stride.** This is the one thing a reader
implementing anything band-wise on a card has to know. MEASURED 2026-09-06 on
an RTX 5070 Ti at T255 and T533 shapes, float32 and float64: a CuPy reduction
over the last axis returns different bits for a STRIDED latitude slice of a
levelled array than for the same rows in a CONTIGUOUS one -- up to 78 percent
of the output cells and 1.5e-02 relative -- while the same rows handed over as
a contiguous slab are byte-identical to the whole reduction's, at every band
count. The band count never entered it; the layout did. The model materialises
every band where it cuts it for exactly that reason.

**What is banded, and what is not.** The stacked synthesis and everything
derived from it, the mass-flux divergence and its continuity closure, the
explicit right-hand side, the tracer transport and all three of its sweeps,
the positivity repair, the physics exchange and the physics return, the lid
absorber, the water fixer, the guards and the output-step diagnostics. The
physics SUITE itself is not banded: its radiation diagnostics are flat means
over the whole plane, and a flat mean folded band by band is a different
number for a different band count, so the twelve levelled fields the exchange
hands it are assembled whole while everything that builds them runs a band at
a time.

### The allocator a run spends through

A fifth `[memory]` setting picks the allocator itself. It changes what the
card HOLDS, never what the run computes: an address is not an operand, and the
ten-step T255 checkpoint gate is byte-identical under all three (40 arrays at
step 0 and 126 at step 10, 0 differing, on an RTX 5070 Ti and again on an RTX
5090, 2026-09-06). It is therefore absent from the config
identity at every value, so a run that moves it shares a config hash, a
checkpoint lineage and a receipt with one that does not.

```toml
[memory]
device_allocator = "default"   # or "slab", or "async"
```

| Allocator | What it is | Held over live, MEASURED 2026-09-06, RTX 5070 Ti |
|---|---|---|
| `default` | the CuPy memory pool the process already carries, which is what every run before this setting used | 1.195 at T85 reference, 1.222 at T255 native |
| `slab` | one contiguous device arena, taken before the first model byte and cut by an exact-fit, coalescing free list; it extends by a further segment when a request fits nowhere, capped at a quarter of a GiB a segment, and it never returns bytes to a pool's size bins | 1.424 at T85, 1.239 at T255 (the shipped capped growth; the uncapped form it replaced read 1.367) |
| `async` | the CUDA driver's own `cudaMallocAsync` pool | 1.53 to 1.69 at T85, 1.197 at T255 |

The pool the process already carries is the default because it measured best on
both counts: least held, and least wall (36 steps at T85: 5.88 s against the
slab's 6.85 and the async pool's 6.33). The slab is there for the case the
pool's own free bins cannot serve, which is what killed a T533 run in the
record: the live bytes fitted the card with 1.9 GiB to spare and the pool's 1.8
GiB of free chunks, in the shapes the dynamics had asked for, could not serve a
0.38 GiB block.

Every run's receipt now carries `device_memory.held_over_live_peak`, the
allocator's held high-water over the run's live high-water, beside the
allocator that spent the bytes. The held total is folded at every allocation
rather than read at the end, because a total read after the last free is not
the run's.

### Transaction boundary

A native half-step is all-or-nothing:

1. Copy the caller exchange and persistent physics state.
2. Construct and validate the native batch.
3. Run the complete admitted scheme order on the copies.
4. Validate finite values, non-negativity, inventory and shape.
5. Measure atmospheric, surface, soil, precipitation, water and energy
   changes.
6. Return a new exchange and a new persistent physics state.

The caller's input exchange cannot be partially mutated if a later scheme
fails. The global Strang split commits a native result only after the complete
transaction succeeds.

### Persistent scheme state

Checkpoint schema v2 stores native state under a `physics__` namespace. It
includes, as applicable:

- radiation scheduler/update state, the held radiation planes (WRF's
  SWDOWN, GLW, GSW, OLR and, since 2026-09-04, the upward and downward
  shortwave at the top of the radiation column, the upward longwave at
  the surface and the column cloud cover the scheme's overlap implies)
  and their time integrals (`acc_*`, J/m2, advanced by plane times dt on
  every physics call so two checkpoints give the exact interval-mean
  flux), the radiation call count and the cloud-optics size-bounding
  record (how many cloudy cells and columns carried a particle size
  outside the RRTMGP tables, last call and running sum);
- SFCLAY in/out fields;
- Noah soil, snow, canopy and diagnostic state;
- Morrison number moments, radii and precipitation accumulators;
- the cumulus `RAINC` accumulator and call count (neither cumulus scheme
  holds a state between calls: `RAINCV` is a per-call increment consumed
  once, so a resumed run continues the accumulator bit for bit); and
- JSON-scalar scheduler/identity metadata.

A resumed run therefore does not cold-start Noah, radiation or Morrison at a
checkpoint boundary.

### Cold-start surface seeding

A real-analysis cold start seeds the surface from the analysis it decodes
(`woof.globe.surface_seeding`): the sea-ice fraction and thickness
(GRIB2 ICEC and ICETK) into the surface state, and the snow water
equivalent, depth and cover flag (WEASD and SNOD) into Noah's store, beside
the skin temperature, the four soil layers and the land mask the initializer
took before. On open water the skin is the analysis skin over the analysis's
own water points only, so a lake or coastal column whose stencil reaches the
analysis's land never starts, and holds for the run, a land skin
(`analysis_initial.open_water_skin_temperature`; the rule and the columns it
changed are in `initial.provenance.open_water_skin`). On the ocean that water
temperature is held for the run, the usual medium-range choice for a sea
surface the analysis observes. An inland lake is not that surface: the
analysis's lake temperature is weakly observed, and held, a lake the analysis
carries too warm evaporates at that rate for the whole forecast. GDAS carried
Lake Tana (11.9 N, 37.5 E) at 304.8 to 305.1 K at all eight cycles of
2026-09-29 and 2026-09-30, 9.2 K above the 2 m air temperature the same
analyses carry over that water averaged over the eight cycles, and held for a
240 h T255 forecast that column drew about 22 kg/m2 a day from its surface
reservoir. So a lake column (`statics.lake_columns`, WRF's LAKEMASK rule:
open water whose lake share in the statics' land-use fractions beats its
ocean share; the statics carry the share as the surface member
`lake_fraction`) integrates its skin on the land cadence from its own surface
energy budget (`native_runtime._lake_surface_step`): the shortwave and
downward longwave it absorbs, its own emission, and the surface layer's
sensible and latent fluxes of the call, over the water heat capacity the
state carries (4e7 J/m2/K, a 10 m mixed layer, blended with the column's land
share), never cooled below 273.15 K (there is no lake ice; an analysed ice
cover is a frozen column). On the GDAS 2026-09-25 12Z case the lake surface
temperature error against OISST on the lakes it covers fell from 1.37 to
1.11 K RMSE at 120 h, and Lake Tana's column cooled from 305.3 to 299.6 K.
A checkpoint written before the plane reads with no lakes. The receipt's `initial.provenance.
surface_seeding` names every source, its unit, value range, masked count
and the columns seeded. The seeding refuses instead of guessing: an
analysis without one of the four fields is refused by name, the snow unit
is verified by value (the bulk density the two snow planes imply must read
as snow, so a water equivalent in metres cannot pass as kg m-2), and the
snow bitmap is read as no snow on open water only. The statics rulebook
freezes every column at or above a sea-ice fraction of one half (the ice
class and ice soil, WRF's non-fractional rule) and puts the snow-covered
albedo on snowy land. The columns Noah skips as sea ice or land ice run a
four-node heat conduction column in their soil-temperature layers
(`physics/frozen_surface.py`: the skin node under held radiation and the
surface layer's fluxes, each node the snow or ice at its depth, conduction
to the freezing point of sea water at the analysed ice bottom or to the
deep-soil climatology at 8 m through firn, implicit, capped at melting; a
partial pack presents the
fraction-weighted blend of the ice skin and open water at the freezing
point) in place of WRF's seaice_noah and SFLX_GLACIAL, which are not
ported; that divergence is stated in the adapter contract. The cold start
seeds the column: linear from the analysed skin to the freezing point on
sea ice, the analysed soil temperatures under a skin node on land ice.
An ice-free planet never enters that arithmetic and its land/water flag is
bit-for-bit the pre-seeding construction. The instrument is calibrated by
`python -m woof.globe.surface_seeding --calibrate` on synthetic
analyses in both hemispheres and both latitude orderings. Checkpoints
written before the seeding carry no sea-ice planes; the reader fills zero
ice and records the absence.

### Pure fail-closed configuration

Adapter registration and option validation are dependency-light. Bad adapter
names, acknowledgements, selectors, intervals or option types fail while TOML
is loaded, before a transform is built and before CuPy or a CUDA device is
accessed.

List the current registry with:

```bash
woof global physics-manifest
```

The built-in adapter reads `device-pending`, and it has been there twice.
The qualification battery passed on an RTX 5090 on 2026-09-01 and
promoted the registration to `experimental` on that evidence, which
covered a five-scheme stack: RRTMGP, SFCLAY, Noah, YSU and Morrison.
Grell-Freitas cumulus then joined the suite as a sixth component, default
on, so the evidence on file no longer describes the stack the adapter
runs, and the registration went back to `device-pending`. The superseded
digest sits in the contract's `limitations` rather than in
`device_evidence_sha256`, where it would read as cover for a stack it
never measured, and `device_evidence_sha256` is the zero digest until the
battery is re-run with `cumulus = "gf"` in the order. `physics-manifest`
is the current answer; this paragraph is the history behind it.

## Target-device qualification

Source tests prove transaction behavior, restart serialization, refusal order,
option identity and artifact integrity. They do not prove that a particular
CUDA wheel, driver and GPU execute the adapter correctly.

Run the qualification battery on the target card:

```bash
woof global native-qualify \
  arwen_global_level5_native_smoke \
  --outdir out/arwen-global-native-qualification
```

The battery binds evidence to:

- Level-5 pins;
- adapter contract and arithmetic hashes;
- all adapter source-file hashes;
- Python, NumPy and CuPy versions;
- CUDA runtime/driver and GPU identity;
- uninterrupted terminal checkpoint;
- midpoint-restarted terminal checkpoint; and
- bit-exact equality of the complete checkpoint array inventory.

An error still emits a self-hashed `native-device-evidence.json` and no
candidate. A passing battery may emit an **experimental candidate**; it does
not automatically rewrite the built-in source contract to `validated`.

## Complete parent export

`export-parent` writes a hash-bound regular-lat/lon artifact from a global
checkpoint:

```bash
woof global export-parent \
  CONFIG.toml CHECKPOINT.npz parent.npz \
  --nlat 361 --nlon 720
```

In addition to the atmospheric and surface state, Level-5 exports include:

- all six water masses and five number moments;
- pressure and pressure interfaces;
- virtual temperature;
- geopotential and terrain;
- east/north wind and vertical motion;
- pressure tendency and diagnosed omega;
- continuity residual diagnostics;
- dry column mass; and
- complete grid-resident surface/soil state needed for provenance.

The export remains a neutral global artifact. It becomes a WOOF regional
forcing only after translation against a specific target identity.

## Regional target contract

A target artifact binds everything needed to turn global physical fields into
one particular WOOF regional state:

- latitude and longitude at mass points;
- terrain height;
- map rotation (`sina`, `cosa`);
- total-pressure hybrid A/B coefficients, top to surface;
- WOOF base dry mass;
- `c1h/c2h/c1f/c2f` coupling coefficients;
- base potential temperature and geopotential;
- mass/u/v map factors; and
- a caller-supplied source identity such as config, geography and base-state
  hashes.

From a live initialized regional state:

```python
from woof.globe.regional import write_regional_target_from_state

write_regional_target_from_state(
    "regional-target.npz",
    state,
    latitude_deg=xlat,
    longitude_deg=xlong,
    a_half_pa=a_half,
    b_half=b_half,
    name="d01-global-parent-target",
    grid_id=cfg.grid_id,
    source_identity={"regional_config_sha256": config_sha},
)
```

The generic CLI can also wrap a prepared NPZ:

```bash
woof global make-regional-target \
  target-input.npz regional-target.npz \
  --name d01-global-parent-target \
  --grid-id d01
```

## Translation arithmetic

Translate one parent export with:

```bash
woof global translate-regional-frame \
  parent.npz regional-target.npz regional-frame.npz
```

The frozen Level-5 translation is:

1. Periodic bilinear interpolation on the regular global lat/lon source.
   The dateline wraps; a regional domain is never forced to treat longitude
   0/360 as a discontinuity.
2. A target-terrain surface-pressure correction using virtual temperature.
3. Target total-pressure interfaces from the target A/B coordinate.
4. Independent-column log-pressure interpolation for thermodynamics, water,
   moments and horizontal wind.
5. Earth-relative wind rotation into the regional map frame.
6. Nonperiodic mass-to-C-grid staggering for U and V.
7. Hydrostatic geopotential reintegration from the target terrain.
8. Dry column mass from pressure thickness divided by `1 + total water`.
9. Construction of regional perturbation fields.
10. Construction of the exact WRF-coupled fields consumed by WOOF:

```text
u, v, theta, phi, mu, qv
```

The coupled fields follow the same formulas as WOOF's existing
`domain_boundary_snapshot` path. East and north boundary sides are stored
outermost-first, matching the existing external LBC layout.

## Parent series and existing LBC attachment

Bind two or more frames:

```bash
woof global make-parent-series \
  regional-target.npz parent-series.json \
  frame-000.npz frame-003.npz frame-006.npz
```

The series verifies every frame file hash, frame self-hash, target hash,
monotonic time and field inventory.

In the complete WOOF repository:

```python
from woof.globe.regional import install_initial_and_attach_parent

result = install_initial_and_attach_parent(
    state,
    "parent-series.json",
    "regional-target.npz",
    spec_bdy_width=5,
    spec_zone=1,
    relax_zone=4,
    streaming=True,
    receipt_directory="out/global-parent-receipts",
)
```

This call:

- transactionally installs the first translated frame into the existing
  `DomainState`;
- seeds diagnostic pressure and inverse density when those arrays exist;
- builds the existing `BoundaryInterval`, `FieldBoundary`, `SideBoundary` and
  `LateralBoundaries` objects; and
- invokes either `attach_lateral_boundaries` or
  `attach_streaming_lateral_boundaries`.

There is no second boundary kernel and no alternate hot-loop time arithmetic.
Once attached, WOOF's established specified/relaxation path owns application.

## What Level 5 proves now

The CPU/reference battery proves:

- old global transform, shallow-water and primitive controls remain green;
- Level-4 moist/hybrid/reference-physics behavior remains green;
- the 15-field checkpoint/restart inventory;
- explicit Level-4 migration and tamper refusal;
- native adapter registration and pure option refusal;
- native transaction no-mutation behavior using controlled scheme doubles;
- persistence of native physics state through checkpoint/restart;
- transform and parent artifact integrity;
- dateline-safe interpolation;
- target binding and frame tamper refusal;
- exact WRF coupled-unit construction;
- first-frame transactional install;
- conversion to the existing LBC object model; and
- eager and streaming attachment dispatch.

On a target card, the qualification battery has run. It passed on an RTX
5090 on 2026-09-01 for the five-scheme stack: every gate green, terminal
checkpoints bit-exact including run trackers across a midpoint restart,
device identity bound, six admission defects found and fixed on the way
in. That evidence does not cover the six-scheme suite the adapter runs
today, so the registration is back at `device-pending` and no CUDA
numerical claim rests on it. No performance claim is made from it either.
An environment without CuPy still gets the same treatment it always did:
the battery emits a durable error receipt naming the missing dependency
and emits no candidate.

## Remaining admission work

A production global-parent claim still requires:

1. passing target-device qualification on the intended GPU/CUDA/CuPy stack
   for the suite as it stands, which today means re-running the battery
   with `cumulus = "gf"` in the scheme order;
2. scheme-by-scheme comparison against the existing regional physics wrappers
   on identical column batches;
3. long moist global integrations and climatological/budget evaluation;
4. real static land/soil/vegetation fields rather than configured constants;
5. real global-analysis initialization and cycling;
6. global-parent versus ordinary-parent regional A/B forecasts;
7. LBC seam, restart and long-series streaming tests on a complete regional
   model; and
8. explicit promotion of a measured adapter contract from `device-pending` to
   `experimental` or `validated` using its evidence hash.

Level 5 supplies the executable seams and their refusal/evidence contracts. It
does not relabel those remaining meteorological campaigns as completed.
