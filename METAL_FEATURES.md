# Metal feature contracts and remaining gaps

These contracts describe the scoped regional and Hex additions. They do not
establish complete numerical parity with CUDA WOOF or WRF. A supported selector,
a successful forecast, checkpoint identity, and agreement with an independent
numerical reference are separate claims.

This integration carries upstream WOOF 1.1.5 and the earlier scoped Metal
implementations. Native packaging and release-specific public forecast
acceptance remain in progress. Reconciled interfaces and component comparisons
do not establish numerical acceptance for every configuration.

## Integration admission

Regional wind farms run on resident Metal domains: Fitch (including automatic
selection), EWP, Jensen and the LES generalized actuator disk (GAD). Turbine
momentum, retained/advected MYNN TKE and turbine/farm power use the upstream
CUDA launch order and software IEEE arithmetic. Host-source comparisons
matched 344,282 words (three turbine repeats), including sparse and
full-column forcing, wake superposition, GAD forces and history power reductions. Power and TKE source
fields are published through the standard history boundary.

Hex complete-globe wind farms use Fitch on spherical Voronoi cells through
MYNN, with turbine/farm power in history. Its RUC cold-start fractional
sea-ice optics use the existing Metal land-use kernel, matching the CUDA
column helper. Public regional MYNN/Noah and 12-level globe MYNN/RUC runs
published positive power and nonzero turbine TKE; the globe used the
existing non-monotonic Metal transport. Culled regional Hex meshes retain
an explicit refusal: outside-domain turbine ownership is not assigned, so
nearest-cell mapping could incorrectly force boundary cells. Resident
whole-domain ownership remains required for regional structured grids.

Thompson's post-column negative vapour repair runs on Metal before moist
physics finish. It walks each column in increasing level order and uses
rounded binary32 operations for the neighbour mean. The fill matched all
255 words against the host-built CUDA source under each Metal arithmetic
policy, including negative subnormals, signed zero, NaNs and adjacent repairs.

Resident and tiled restart headers preserve whether the held shortwave split
came from the radiation scheme. The producer-marker omission is shared with upstream CUDA;
this integration intentionally carries the marker to preserve diagnostics.
Older checkpoints with SWDDIR but no producer marker are
refused unless their radiation call count is known to be zero. Use a checkpoint
with the marker or start from prepared state. Public tiled restart comparisons
cover both held marker values and their published fields.

Ordinary FAST_SBM (`mp_physics=30`) runs its spectral column on Metal. A cold start
with bulk moisture initializes the bins on the host with NumPy and SciPy;
later spectral steps stay resident on Metal. Its breakup helper is kept out of
line as a compilation-context workaround; the underlying cause is unresolved.
The full column matches the host-built CUDA source in the checked warm, mixed
and dry cases. An active wet idealized run and a public aerosol-only forecast
both preserved their carried state across checkpoint restore. These checks do
not establish CUDA-device agreement or wet public-forecast parity. Strict
mode and polar radar diagnostics remain refused. Untested compiler contexts
remain a risk while the inline-only discrepancy is unresolved.

Aerosol Thompson (`mp_physics=28`) carries both source networks, ordered
accumulation, fallout, terminal state and each generation's distinct radar
reflector on Metal. Consecutive full-wrapper source comparisons preserve live
prognostics, precipitation, accumulators and output-due reflectivity without
resetting intermediate stages. The reference is the retained CUDA source built
for the host; public coupled cycling and actual CUDA-device agreement remain
separate acceptance evidence.

Purdue Lin (`mp_physics=2`) is a mechanical translation of the CUDA column and
its melting-aware reflectivity: every binary32 operation and comparison runs on
soft32 words, so subnormals behave as in the host reference. It runs 2 to 128
levels; the CUDA 256 and 512 tiers size only the kernel's own arrays and overrun
the helpers' 128-level storage, so they are not reproduced.

Ferrier-Aligo (`mp_physics=5`) is a mechanical translation of the CUDA table
initialisation and column, including its reflectivity and analysis-only moments,
on soft32 words with the packaged expanded rain table. The table buffer and
every updated field match the host reference word for word.

CCPP sa-TKE-EDMF (`bl_pbl_physics=901`) is a mechanical translation of the CUDA
column and its saturation-pressure table on soft binary64, with the pinned glibc
exp, log and pow; the table, Exner function and every output match the host
reference word for word. It needs a land-surface scheme, whose vegetation
fraction and roughness it reads, and the host binds its float64 columns.

Metal implements `solar_generation="wrf_480"` geometry with integer software
binary32/binary64 and the pinned upstream trigonometric functions; the earlier
`wrf_461` ordinary paths retain their behavior. Metal implements both
`ysu_version` generations (`wrf_480` as its own kernel, word-for-word with the CUDA
source's arithmetic). Both `noahmp_snow_age` generations are implemented in
radiation and glacier columns, including the `wrf_470` night-aging default.
Both `urban_water_balance` generations are implemented for the WRF 4.7.1
canopy through Noah and Noah-MP handoffs, including the `wrf_480` default.
The full `urban_ucm_generation="wrf_480"` canopy also executes through
both handoffs, including tree and vegetated-ground state and the retained
upstream control restrictions. New Tiedtke dry-mass coupling, Thompson
settled-cloud booking and hydrometeor handover corrections have Metal dispatch
and state handling; their numerical and public acceptance remains specific to
the exercised caller. Conservative WSM/WDM surface-return corrections use their
shared scheme dispatch.
Select exact physics mode to retain matched scheme algorithms. The Hex command exposes `--physics-mode exact`; regional experiment
files express the resolved correction switches explicitly. This physics-mode selector does not enable full WRF-exact arithmetic.

Hex's existing column providers and the bounded generic plan support
`--physics-mode correct`, including `--physics-correct-skip`, as well as
`--physics-mode exact`. The public default follows upstream's correct mode.
WSM6 passes the conservative surface-booking switch to its existing kernel;
Thompson allocates its settled-cloud totals before the first checkpoint.
Receipts name corrections in effect in the column seam and identify the
host handover correction separately. Mode and enabled correction settings
bind restart identity; restoring under different settings is refused.
These paths have short public continuation comparisons. The discriminating
WSM6 budget check preserves atmospheric words while reducing the surface
budget residual; this does not establish complete CUDA-device parity.
Its batch uses upstream 1.1 RunConfig generation defaults, replacing the temporary 4.6.1
pins used while the new generations were unavailable. This intentionally
changes the selected source algorithms in those profiles to match upstream. Its receipt and restart
identity carry the selected source generations. Provider history metadata reads
those resolved values rather than assuming the frozen WSM6 profile. Metal
provider checkpoints also bind the shared resolved RunConfig fingerprint, so
implicit profile defaults cannot change across a restore. Earlier experimental
payloads without that binding are refused. Metal's ordinary kernel arithmetic
has not been established as either upstream fast numerics or strict numerics;
strict numerics remains refused.

Noah sea-ice columns run on Metal (word-matched to the CUDA source).
Source-initialized lake state (`lake_state_source` other than `forcing`) binds on
Metal through the same host-side source read and a word-matched column kernel.
Active eclipse obscuration (`ra_sw_eclipse`) runs on Metal: the obscuration
field and the shortwave dimming word-match the CUDA source.
The ocean mixed layer (`sf_ocean_physics=1`) runs on Metal and word-matches
the CUDA source. Prognostic sea ice (`seaice_prognostic`) runs on Metal, alone or under the mixed
layer. The current sea-ice caller carries the upstream flux-ownership flag;
the mixed layer uses ice heat only where that flag or the land classification
assigns ownership. A focused seven-column caller comparison matches all stored
words against host-compiled source, including warm water and newly formed ice.
The 3DPWP ocean (`sf_ocean_physics=2`) runs on Metal and word-matches the CUDA
source; its WOA18, analysis and configured initializations build the same
profiles on the host. Sea-ice/PWP heat reconciliation and cold-profile seeding
also run on Metal, carrying unresolved heat in the restart state. Selected
host-compiled source comparisons cover both surface-flux ownership modes and
mixed-layer heat exchange. A synthetic full-physics checkpoint comparison
retains active ice growth and unresolved heat exactly; this does not establish
public long-run or NVIDIA-device parity. Coupled SFIRE (`ifire=2`) runs on
Metal, including `woof fire-ideal --backend metal` and its checkpoint resume;
each fire unit and each coupled fire step of a small ideal case word-match the
host-compiled CUDA source. The bulk smoke tracer (`fire_smoke`), firebrand
spotting (`fs_firebrand_gen_lim > 0`) and `fmoist_run` without a surface
package remain CUDA only and are refused on Metal. WRF-native
restart initialization stages source arrays on the host. The DA IAU, hydrostatic
rebalancing, time-window and extended-heating paths operate on Metal states as
described below; packed members remain a CUDA batching mechanism. These paths'
public and numerical evidence is configuration-specific.

Aerosol-aware Thompson (`mp_physics=28`) carries the newer source-network
accumulators on Metal. Full-wrapper comparisons cover physical and booked
fields, but do not establish full WRF-exact physics or CUDA-device agreement.
Settled-cloud booking has device kernels, restart/history inventory and public
default-profile execution. This does not establish full strict-forecast parity.

The Hex clock uses the shared timestep resolver, mesh admission and schedule.
One resolved timestep and radiation cadence reach physics, dynamics, history
labels and restart configuration. Noah-MP carries the additional soil-drainage
diagnostic and retained ozone uses the canonical restart inventory.

TKE self-diffusion now forms the doubled increment before adding it to the
retained tendency, following upstream's expression order. This intentionally
corrects the earlier Metal sequence of two additions. Reporting accumulation
still follows the software binary64 contract below.

| Feature | Implemented path | Remaining restriction |
| --- | --- | --- |
| Coordinate mixing | `diff_opt=1` coefficient preparation, initial thermal reference, and horizontal mixing at all four staggers | Shared upstream mixing and boundary compatibility rules apply |
| No exchange mixing | `km_opt=0` uses the common configuration validator | Outside SASE, the upstream `no-horizontal-mixing-operator-v1` acknowledgement is required; this is a deliberate research control |
| TKE budget | Device term accumulation and binary64 slab reduction, published through the existing budget diagnostics | `km_opt=2`; the shared MYNN budget restriction remains |
| Radiation | Valid combinations of LW/SW IDs 0, 1, 4, 5 and 90 use the common spectrum adapters | Scheme-specific geography, moisture, compatibility and surface-flux requirements remain |
| Urban land surface | Noah SLUCM/BEP rural-state handoff; BEM integration repairs | Noah BEP+BEM admitted with word-for-word reference agreement on the regime fixtures; Noah mosaic SLUCM (`sf_surface_mosaic=1` with `sf_urban_physics=1`) admitted with word-for-word agreement on the WRF 4.7.1 mosaic oracle |
| Lake | Noah, RUC and Noah-MP snapshot precipitation before the land driver consumes it and then run the lake column | The selected model still needs its lake mask, latitude and depth policy |
| NSSL sedimentation | Local 64/128-level tiers and a device-workspace 256-level tier | 256 levels, matching the upstream sedimentation extent |
| HAILCAST | The upstream operational HRRR default and the selectable WRF 4.6.1 kernel share the existing Metal diagnostic and restart fields | The shared 256-level limit applies |
| Modern radiation capacity | Admission counts model and added top layers; the isolated SW solver uses budgeted device workspace above 128 | Shared forecast profiles and LW Planck sources remain limited to 128 total layers; the isolated SW ceiling is removed |
| Tiled stochastic physics | Global spectral owner, windowed SPPT/SKEBS/SPP and the existing checkpoint payload | Supported Metal tile forecasts; pattern and rate transforms are host staged |
| Prepared nested DA | Shared Metal context, live-parent SINT initialization and correction, device wind and analysis addition, hot-start height, ramp and pointwise increments, resident IAU rates and mass coupling, hydrostatic integration and live geopotential application, and tree-checkpoint leg joins | Perturbation generation, hot-start thermodynamics and diagnostics, IAU time weights and unsupported layouts, hydrostatic staging and receipts, and analysis constraints retain host work. Both backends refuse nested radar tendency forcing |
| Regional Hex LTS | Gathered regional acoustic kernels, class schedule, reflux, history and checkpoint identity | Single device and the shared supported acoustic schedule; no class interface in the driven boundary zone |
| Hex providers | Shared dispatch, scalar inventory and allocation pricing; a MYNN/RUC/GSL seam adds nine-layer soil and internal drag | WSM6, Thompson and MYNN/RUC coupled execution and restart identity are established for a short regional case; MYNN/RUC requires genuine mesh orographic statistics and nine-layer soil initialization |
| Hex source treatment | Provider-owned source-table validation | Built-in rows reject source treatment on CUDA too; no source-capable optional provider is installed, and Metal `hex pair` remains unavailable |
| Full WRF-exact mode | Refusal retained | Complete dynamics and physics arithmetic paths are unfinished |

The operational HRRR HAILCAST default matches 72 output words across eight
upstream storm columns with positive hail. The full Metal diagnostic caller
matches 30,403 words across three prepared column sets, including 256 levels.
The existing WRF 4.6.1 selection separately matches all 72 compared words on
the same eight storm columns. These are source comparisons, not forecast
verification.

Cumulus-fed Thompson rain receives the upstream per-step rain-number seed on
Metal. The calculation stages bounded slabs to NumPy and preserves the source
expression order; it is not GPU-native. Preparation EOS calls also stage through
the shared host diagnostic, including pinned log1p/pow words, while ordinary
forecast diagnostics retain their existing arithmetic. Component comparisons
cover both base-state layouts, both hypsometric options and windowed writes.

Original Tiedtke supports the upstream scalar and four-column libmvec exp
selectors, including scalar row tails. Referees before glibc 2.41 are explicitly
refused because their tangent implementation is not ported. Strict revised MM5
uses products for the four squared terms folded by the upstream referee build.
Strict Thompson remains refused; the upstream strict-only graupel-table and
fallout-flag changes do not alter the ordinary Metal forecast adapter.

A one-step public HRRR-initialized specified-domain forecast runs with KF,
classic Thompson and either operational HRRR or WRF 4.6.1 HAILCAST. Both
history frames publish finite hail diagnostics, reflectivity, rain, winds and
theta. Rain and hail
remain zero in this dry case; wet coupled forecasts remain unqualified.

## Regional mixing and budget ownership

Coordinate mixing uses explicit binary32 operation boundaries, including
subnormal arithmetic. It preserves the upstream initial thermal reference;
reconstructing that reference from evolved theta on restart would change the
operator. The ordinary `diff_opt=2` dispatch stays separate.

`km_opt=0` skips exchange mixing through the common dispatcher. The sixth-order
filter remains an independent selector. The acknowledgement does not assert that
a configuration without horizontal exchange mixing is suitable for production.

The TKE budget uses the upstream 256-lane strided accumulation and reduction
order. Double-precision terms and totals use software binary64. Prognostic TKE
belongs to the restart state. The reporting accumulator is rebuilt and begins a
new window after restart, matching the upstream report-only contract.

Admitted WRF-exact regional paths consume the stored `MF_VX_INV` for V flux
coupling, acoustic initialization and updates, dry mixing, and specified
relaxation. Small-step finish uses the forward factor to undo the coupling.
The focused comparisons exercise an inverse that differs from the recomputed
reciprocal. Single mapped steps match all 6,265 periodic and 14,489 specified
words compared against the host-built source reference.

Strict implicit vertical advection uses the stored V-map inverse in its column
mass update, explicit binary32 rounding in the split and solve coefficients,
and software binary64 in the column solver. The operator comparisons cover
both WRF variants and all three workspace tiers through 256 mass levels.
The pinned WRF operator fixtures match word for word over their defined
comparison regions. The strict forecast admits both WRF variants for the
checked flat, physics-off profiles without mixing or damping. The admitted
profiles cover dry forecasts and qv/qc/qr transport, including mapped domains
and both supported scalar transport forms. The checked dry steps match 9,308
host-source words per case. The checked moist steps match 16,551 words per
case, including the post-acoustic split and all three scalar solves. A bounded
public `woof verify` case using the standard external case registry exercises
modern unmapped and legacy mapped moist forecasts. Each fresh-process
continuation matches all 4,275 saved values and scientific checkpoint headers.
The checked dry and moist specified-boundary profiles use clock-bound
snapshots. Both mapped dry variants match 39,070 host-source words across a
boundary interval change. The mapped legacy moist case matches 76,272 words
across the same change, including the vapour boundary tendency and scalar
solves. The strict snapshot producer preserves the rounded mass, face
averaging and theta-offset
sequence, and divides momentum by the forward map factor on both axes.
The boundary collector matches 17,220 retained values and tendencies against
the host expressions. In the public specified case, Metal-produced snapshots
match the host-prepared inputs and all 7,507 saved dry forecast values.
The public moist case also exercises Metal-produced snapshots and a
fresh-process tree continuation. All 10,147 saved values and scientific
headers match, including the boundary clock.
Both variants also admit the unmapped, periodic moist terrain profile with
`km_opt=0`, the research acknowledgement, and positive-definite transport.
Each matches 25,598 host-source words with both splits active over sloped
columns. The public terrain case matches all 4,275 saved values on a
fresh-process continuation.
The unmapped, flat, dry periodic profile also admits vertical Courant
damping. The IEVA caller uses the configured onset, including valid
thresholds above 1. The compared full steps match 11,324 host-source words
per variant. The public damping case exercises thresholds 0.5 and 2, and
the fresh-process continuation at 0.5 matches all 3,155 saved values.

Strict resident nest coupling has software binary32 force-time round trips,
saved west/south mass-halo corrections, and distinct T and THM history fields.
The comparisons match 13,965 host-operator words for mapped and unmapped
faces and history. A private, fixed-step, one-way tree exercises three real
force intervals. Six child SINT calls and 18 four-side table sets match 83,718
host-compiled CUDA source words. The complete parent and child history from
the host source run matches 256,648 words, including the initial T and THM
inventory. Fresh-process continuation matches 128,324 history values,
138,068 checkpoint values, and scientific headers. Public `woof verify`
through the external case registry exercises the admitted fixed, resident,
flat, unmapped ratio-3 one-way profiles with a specified parent, aligned
history, and parent-derived child initialization. The moist physics-off
profile uses positive-definite transport and carries active vapour, cloud and
rain. Its initialization and forcing match 130,338 host-source words. Public
full and resumed history match 330,088 and 165,044 source words, and all
187,028 resumed checkpoint values match the uninterrupted run.

The resident two-way terrain feedback transaction matches 35,515 selected
CUDA source words, including all four updated terrain base fields. Public
`woof verify` cases exercise the fixed resident, unmapped ratio-3,
physics-off two-way profiles with smoothing option 1. Dry flat history
matches 256,648 host-compiled CUDA source words. Moist sloped history and
feedback-stage bases and diagnostics match 445,972 words. The latter uses
positive-definite transport and the `km_opt=0` research acknowledgement.
Fresh-process dry and moist continuations match 138,068 and 213,650 saved
checkpoint values and all scientific headers, respectively. The terrain
checkpoint restores the updated base and the per-edge feedback marker and
rebuilds the terrain caches. The initial transaction follows WOOF's CUDA
path, which re-diagnoses the parent feedback window.

Public moist terrain one-way history matches 330,088 host-compiled CUDA
source words. Moist flat two-way history and feedback stages match 380,284
words. Fresh-process continuation matches 187,028 checkpoint values and
scientific headers for each profile.

Each admitted two-way profile accepts the shared smoothing options 0, 1
and 2. Public history and feedback comparisons for options 0 and 2 match
306,844 selected source words each on dry flat terrain, 380,284 and
380,644 on moist flat terrain, and 445,972 and 446,332 on moist sloped
terrain. The corresponding fresh restarts match 138,068, 187,028 and
213,650 checkpoint values per option. The moist-flat option-2 fixture
activates the shared WOOF nonnegative clamp on 60 cloud-water values.
Its pre- and post-clamp words match the CUDA WOOF caller.

Non-positive-definite moist, microphysics, ratio-2, moving, streamed and
WRF-file nest initialization remain refused in these strict profiles.

## Radiation and surface state

The ordinary Noah Eta surface path applies `CHKLOWQ` on Metal using the
source's rounded humidity conversions and threshold comparisons. It matches
all 4,096 compared words, including equality at the humidity thresholds and
unchanged inactive words. The active Noah and Eta driver completes a bounded
step. Noah mosaic retains the constant marker used by upstream CUDA.

Classic RRTM and legacy RRTMG resolve `ghg_input=1` CAMtr gases at each
radiation valid time, with explicit overrides applied last. The CAMtr asset
and per-call policy are bound into Metal restart identity. RRTMGP uses its
existing start-time gas selection. The legacy public forecast compared
2,995,200 prepared longwave and shortwave gas words to the upstream table
at two radiation calls. Its fresh-process restart matches 5,236,848 saved
values and 2,617,897 history values at each held and due endpoint. Classic
longwave matches 144 selected reference words across a year boundary.

Strict Noah forcing, rain clearing, SFCDIAGS, land columns and sea-ice columns
have separate software-arithmetic paths compared with selected source bodies
compiled for host execution. Finite land outputs matched in the sampled cases;
two nonfinite ground-flux words differed in NaN sign. The sea-ice translation
retains a function boundary for source-word agreement; its inline divergence
remains unexplained. Strict soil initialization copies water only when the
source's frozen branch is inactive, and refuses that branch because its CUDA
power operation is not bound. The strict radiation profiles also admit revised
MM5 surface exchange and Noah, with no PBL, prognostic ocean ice, urban or
mosaic model, monthly albedo or prescribed LAI. A composed source comparison
covers carried fields and restart. A four-second public prepared forecast at
a 0.1-second step also matched its checkpoint continuation. A one-second-step
case failed health checks; its captured vertical-mixing leaf matched the
selected host-compiled source, but a complete stability bound was not proved.
These checks do not establish long-term stability or NVIDIA-device parity.
The same bounded profile supports every nonzero LW/SW combination of selectors
0, 1 and 5, using the shared spectrum constructor and adapter checks. Any
Goddard leaf retains the 128-model-level capacity limit. Selected-source
comparisons cover composed dynamics, warm Noah, radiation cadence and tree
checkpoint restore for the additional combinations. Each also passed a short,
genuinely prepared public daytime forecast and restart comparison, including
all saved arrays, semantic checkpoint metadata and radiation call counts.
The SW-only case preserves its explicitly declared GLW. Public cases have zero
cloud fraction; active-cloud and subnormal checks remain separate component
evidence. Analytic option 90 remains outside the strict arithmetic profile.

Strict classic RRTMG supports standalone `(0,4)` and `(4,0)`, paired
`(4,4)`, and the mixed `(1,4)`, `(4,1)`, `(4,5)`, and `(5,4)` profiles.
The radiation-only profiles require WRF 4.8 solar geometry, `icloud=1`,
`o3input=0`, `ghg_input=0`, and no aerosol or microphysics. Shortwave
alone retains the declared downward longwave input. Column preparation
and McICA run on the host, with the source's operand order on preparation
ties. The solvers, rounded shortwave heating and SWDOWN publication run
on Metal. Longwave heating conversion
uses two source-ordered binary32 divisions on the host. The paired profile
computes longwave first and uses it in the rounded shortwave heating
calculation. The restart identity binds the arithmetic policy and held
radiation state of each active spectrum.

In a prepared daylight forecast, two sampled columns at each radiation
call match 1,424 selected-source solver words. The fresh process restarts
between calls and matches all 5,501,814 checkpoint values and 2,730,223
published history values at the final endpoint. The fixture has active
shortwave heating and zero cloud fraction. Active-cloud and near-terminator
comparisons use separate source fixtures. In the paired public forecast,
two sampled columns at each call match 896 upstream NumPy longwave and
1,424 selected-source shortwave words. A fresh restart between calls
matches 5,550,774 checkpoint and 2,779,183 history values at the final
endpoint. Separate selected longwave checks cover all 16 gas bands,
column setup, cloud optics, and transfer, including active clouds.
The prepared paired and standalone longwave forecasts above have zero
cloud fraction. Standalone longwave retains CUDA's zero-solar-flux SWDOWN
calculation, with 265 selected source words including signed-zero and
albedo edge cases. Its fresh between-call restart matches 5,496,054
checkpoint and 2,724,463 history values with longwave heating active.
The mixed `(1,4)` forecast preserves the classic shortwave leaf's
rounding and the subsequent composed driver residual. Live captures
match both rounding operations, the routed fluxes, and 1,424 selected
shortwave solver words per call. Its fresh between-call restart matches
5,504,694 checkpoint and 2,733,103 history values.
The other mixed forecasts match the composed heating and held diagnostics
from the selected leaf outputs. Selected classic longwave calls match
896 upstream NumPy words each, and the classic shortwave calls match 1,424
selected source words each. Fresh restarts for `(4,1)`, `(4,5)`, and
`(5,4)` match 5,504,694, 5,524,854, and 5,516,214 checkpoint values and
2,724,463, 2,744,623, and 2,744,623 history values, respectively. The
public mixed forecasts have zero cloud fraction.

Strict regional Kessler runs in the checked flat, periodic, radiation-free
profile with zero prescribed diffusivities and no surface physics,
HAILCAST or reflectivity. Column preparation, sedimentation, and limited
heating use source-ordered binary32 arithmetic. The carried precipitation
accumulator uses the strict maximum and a separately rounded addition,
including subnormals. An active 3-D warm-rain case matches 10,062 selected
host-compiled CUDA-source words across two steps, including cloud, rain,
latent heating and pending precipitation. Separate public verification
processes match all 18,510 final checkpoint values and semantic headers,
including 65 cells with nonzero pending precipitation.

The radiation-free flat periodic Kessler and WSM6 profiles also accept
map factors. The selected two-step cases use varying T, U and V map
factors, a stored V inverse distinct from a recomputed reciprocal, and
active winds and latent heating. They match 10,062 Kessler and 15,732
WSM6 host-compiled CUDA-source output words. Their public fresh restarts
match 18,510 and 21,264 saved values, respectively. Paired radiation
retains its unmapped restriction.

Radiation-free strict Kessler also accepts the flat unmapped specified
profile with the existing clock-bound lateral forcing. Selected two-step
cases use prescribed vapour with flow-dependent cloud and rain, or prescribe
all three masses. Both moisture advection forms are exercised with all
three supplied. Each case matches 16,334 host-compiled CUDA-source output
words and 30,166 public fresh tree-restart values. Prescribing the three
time-varying masses changes 55 west-edge words per species against the
constant-forcing control.

The same flat periodic warm-rain profile also supports paired classic
RRTMG `(4,4)` with the pinned solar, gas and cloud options. A public
active-cloud case retains condensate and rain between two radiation
alarms. Its fresh restart between alarms matches 77,874 checkpoint values
and the semantic headers, including the held heating and pending
precipitation. At each alarm, two active cloudy columns match 368
upstream NumPy longwave words and 724 selected shortwave CUDA-source words.

The strict Noah column also has a single-layer urban handoff entry. Sampled
carried calls matched selected host-compiled source, including rural humidity
and height handoff. The canopy and full strict urban driver remain closed.

Spectrum composition retains each leaf's initialization and restart identity.
The existing radiation clock owns cadence and held atmospheric heating and
surface fluxes. Analytic option 90 follows upstream's clear-sky surface proxy and
supplies zero atmospheric heating. Calendar calculations remain on the host. WRF 4.8 solar geometry executes on
Metal with software arithmetic and pinned trig; analytic surface fluxes execute
on Metal. The WRF 4.6.1 ordinary geometry remains host computed. A paired
modern RRTMGP verification matches all 224 shortwave source words for the
selected WRF 4.6.1 irradiance and exercises active shortwave heating. Full radiation
arithmetic has not been established as WRF-exact. The strict 4.6.1 radiation
clock has a source-compared Metal cosine provider. The Metal analytic adapter
refuses strict calls before evaluating surface fluxes: its CuPy transcendental
operations are not yet bound to the Metal path.

Classic RRTM longwave uses dynamically sized device workspaces, including its
added model-top layers. Admission prices all live columns and checks spectral
index and individual-buffer limits. A 256-level model with 269 radiation layers
has completed a forecast and midpoint restart with nonzero sedimentation.
These capacity and restart checks do not establish CUDA-device parity.
Strict interface-temperature preparation uses software binary32 arithmetic
and is compared to the shared array source, including subnormal values.
Strict MM5ATM column preparation also follows the shared pressure, ozone,
trace-gas and cloud-optical-depth operations, including source-pinned power
and tiny rain. Its source comparison covers model tops with and without
added radiation layers. The strict component reads its inputs back to check
that arrays and gas constants are finite, and requires positive layer
temperature, layer pressure, gas constant and gravity before submission.
Strict SETCOEF uses pinned logarithm and exponential operations; its source
comparison includes the MM5ATM-to-SETCOEF caller chain. The typed strict
component reads inputs back to check finiteness and positive layer pressure,
temperature and dry mass before submission.
Strict GASABS bands 1 and 2 follow the shared source interpolation and
use the pinned coefficient tables. Their caller comparison includes both
pressure regimes and subnormal optical depth. The strict typed component reads
its coefficient fields back to reject nonfinite values before submission.
The strict classic RRTM adapter supports `icloud` 0 and 1 without active
microphysics or PBL. It clips packed moisture on the host before cloud
selection. Its transfer and source-ordered caller were compared with the
shared column source; the RRTM–Dudhia pair has a narrow strict physics
profile. `icloud=2` remains refused in strict mode. The driver checkpoint carries held heating, OLR, cloud fraction and the
cumulative radiation-call state.

The classic RRTM implementation corrects the earlier Metal band-9 offset,
g-point fraction-table layout, eight-way interpolation weights, per-g-point
self continuum and added-layer water units against the shared upstream source.
This intentionally changes the affected ordinary Metal radiation results.
Clear and cloudy profile comparisons bound heating differences by
2.4e-9 K/s and flux differences by 3.1e-5 W/m² in the profiles exercised;
they do not certify other profiles or WRF-exact radiation arithmetic.
The shared cloud-fraction preparation applies Ferrier's all-ice saturation
weight below -40 C, matching its upstream selector; this changes ordinary
Metal results for affected Ferrier clouds.

Goddard (option 5) runs its five upstream stages as a mechanical translation of
the CUDA unit: binary64 band solves on software arithmetic with the portable
libm, resident packing and a per-column device workspace sized from the actual
column depth. The cloud optics table keeps its source hash check. A selected
Goddard leaf publishes only its own spectrum's heating, fluxes and diagnostics.
Strict cloud-fraction preparation has a separate source-order binary32 path;
it reads back inputs and results to refuse nonfinite values, whose NaN word
semantics are not established. Strict Goddard direct calls support no
microphysics or PBL and `icloud` 0 or 1; inputs and results are checked for
finiteness. Active shortwave columns whose selected 400 or 700 hPa index is
the top layer or whose indices are equal or reversed are refused because the shared
solver accesses index zero or uninitialized values.
The strict direct-normal flux calculation uses source-order
binary32 arithmetic and checks finite inputs and results before publication.
Restart restores the driver's radiation call count; the model clock controls cadence.
The separately reported adapter update count starts again on a restored run.

WSM7 (`mp_physics=24`) runs its column and REFL_10CM kernels, hail included, as
the same translation through the WSM-family table for 2 to 80 levels, with the
conservative surface option as its own entry points.

WSM6R (`mp_physics=106`) runs its column kernel and the shared effective-radius
kernel as the same translation, at any depth, with its fall-speed scratch on
the device.

WSM3 (`mp_physics=3`) runs its column kernel as the same translation, with the
resolved vertical velocity read at each level's lower face, for 2 to 80 levels.
Its REFL_10CM diagnostic is the shared WSM6 operator; the conservative surface
option runs as its own entry points.

WSM5 (`mp_physics=4`) runs its column and REFL_10CM kernels as a mechanical
translation of the CUDA unit on soft32/soft64 arithmetic, bit-matching the
host build of that unit, for 2 to 80 levels. The conservative surface option
runs as its own entry points.

The strict modern RRTMGP longwave transport uses the pinned exponential,
source-ordered binary32 operations, and an ascending radiance sum scaled
once at the final tile. Its fused Planck path uses the interface-pressure
span and gas columns, including the upstream eta and temperature endpoint
rules. A public solver verification case matches 437,376 selected
host-compiled CUDA-source words, including both vertical orientations,
masked clouds, fused and supplied Planck sources, partial tiles, and
128-layer columns.

Strict shortwave transport uses the budgeted column workspace at all
supported depths. It keeps direct and diffuse fluxes separate until the
ascending broadband fold, with two rounded additions to downward flux
per g-point. A public solver verification case matches 159,300 selected
host-compiled CUDA-source words through 160 layers. It covers both vertical
orientations, night columns, masked clouds, zero gas asymmetry, and partial
column and g-point tiles. This is solver coverage within the existing
forecast admission rules.

Strict gas interpolation reproduces the source's pinned fused logarithm,
reciprocal order, and tropopause and table endpoint rules. Gas optical
depths preserve the major and ordered minor contributions, density and
complement scaling, and shortwave Rayleigh scattering. Strict cloud
optics uses the medium ice roughness table and the source interpolation
and mixture order. Public solver cases match 6,105 metadata words, 33,792
gas-optics words, and 92,160 cloud-optics words against host-compiled CUDA
source. The optics cases use the packaged longwave and shortwave tables.
The strict regional McICA mask uses the retained pressure seeding and
maximum-random overlap arithmetic. Public verification matches 432
mask and padding bytes against the host-compiled CUDA kernel across two
seed counts and fresh or reused storage. The strict delta scale, gas
stores from supplied binary64 ozone, and conversion from broadband
fluxes to heating and diagnostics match 756 selected values against
the host-compiled CUDA kernels and upstream array expressions.
Full strict forecasts still require the CUDA binary32 pressure logarithm,
the CuPy interpolation on binary32 log knots with a binary64 result,
remaining cloud preparation, and joined forecast verification. The ordinary
Metal ozone path uses a software binary64 pressure logarithm.

Strict cloud water paths use software binary32 operations for fixed-radius
Kessler and Ferrier, supplied-radius WSM3, WSM6, Thompson and NSSL, and
the P3 ice remap. Public component verification matches 229,376 path and
radius words against the upstream array expressions, including subnormal
inputs and both snow policies. The WSM3 comparison starts after the phase
split. Number-moment radii for Morrison and Milbrandt remain gated in
strict mode.

Metal modern radiation splits WSM3 cloud and precipitation at 273.15 K
after calculating cloud fraction from the original carriers. Ferrier uses
its total ice with fixed radii. Public ordinary paired forecasts exercise
both cases with active longwave and shortwave heating and match 5,184
source path and radius words per scheme. WSM3 also matches 5,184 live
phase words and 28 strict threshold and subnormal words.

Strict cloudy-background radii preserve positive subnormal cloud fractions
and the source-ordered ice temperature table interpolation. Ordinary and
strict component checks each match 4,900 words, including land, water,
the neutral land flag, carried MYNN masks, and unchanged P3 ice radii.
The MYNN supplied-cloud masks select the same legacy or GSD 4.1 phase
partition as the condensate merge. Both Metal modes now preserve subnormal
amounts in these operations. The GSD branch uses both resolved-phase thresholds
and the temperature-dependent split before deciding whether MYNN supplied each
phase. Public component verification matched 160 mask and merged cloud words
per mode, including initial and subsequent cloud fractions, subnormal amounts
and a NaN temperature. This comparison ends at cloud preparation.

Ordinary Morrison radiation preserves effective radii in columns where the
scheme ran. It reconstructs radii on the specified-zone ring using the
carried moments and the fixed Morrison droplet number. A direct paired
driver check compares 3,136 selected path and radius words across the
cold and carried calls, including active ring and interior differences.

Strict interface temperatures preserve the upstream operation order at
the lower, interior and upper faces. Strict moisture and number reads
clamp a separate device buffer, retaining positive subnormal and NaN
words. The longwave cap uses the shared evenly spaced fallback when a
column's live top pressure would put its penultimate added interface
at or below the coefficient floor. Other columns retain the 400 Pa
ladder. Public verification matches 170 strict interface temperatures
and nine strict clamp edge values against the upstream array expressions.
The mixed-top profile check matches 2,278 pressure and carried vapour
values in each arithmetic mode, with validation owned by its caller.
The strict cap also matches 11,248 temperature and 16,758 pressure and
vapour words against the noncontracted CuPy interpolation expression and
host-compiled CUDA append kernel. All 59 adjacent climatology temperature
differences are exact in binary32, so widening before subtraction gives
the same result. This check covers uniform and mixed model-top pressures.

With supplied binary64 ozone, a direct paired driver check through
`woof verify` runs two strict radiation calls with day and night columns,
active cloud masks, a partial final chunk, and reused gas storage. It
matches 9,330 profile, 34,800 gas-store and 1,110 publication words at the
live source boundaries. The check uses prescribed ozone and leaves the
strict forecast gate unchanged.

Modern RRTMGP's upper-column policy depends on model-top pressure and the active
spectra. The shared forecast profile builder remains limited to 128 total layers,
including its added layers. CUDA's LW Planck-source kernel also holds 128 layers.
The isolated CUDA SW solver specializes to the requested depth. Metal now mirrors
that capacity with a reusable device frame above 128 layers, partitioning columns
and g-points against the device working-set and buffer limits. The ordinary
128-layer solver and ordered broadband reduction are retained. This does not
raise the shared forecast-profile or LW limits.

Urban Noah preserves the rural temperature, humidity, surface state and flux
handoff required by each canopy model. The BEM integration uses ten wall and roof
layers, typed building/level views, and registered canopy stages in the bundle.
The BEM canopy, including green roofs and rooftop PV, reproduces the WRF 4.6.1
reference word for word on the regime fixtures, and Noah BEP+BEM is admitted.
Canopy input packing still stages some forcing through the host. The Noah
mosaic column runs the SLUCM canopy inside its reverse tile loop, carrying the
per-tile roof, wall and road state and blending each urban tile with the grid
fraction; the mosaic column itself uses the exact binary32 operations and
glibc `expf`/`logf`/`powf` words of the CUDA source.

Lake coupling captures the accumulated precipitation before Noah, RUC or
Noah-MP consumes its land-surface carrier. The lake receives that snapshot once, and the
pending carrier is cleared after the lake update. The existing lake state and
restart inventory remain the owners of water, snow and thermal memory.
The lake converts the accumulated millimetres to a rate over its surface-step
interval and uses the WRF near-surface temperature rule for rain or snow.
Bounded comparisons exercise twelve retained WRF columns over 300 steps,
including cold snowfall and warm rain. The float power now selects the same
pinned implementation as CUDA, correcting an ordinary Metal difference in
reported two-metre temperature. The binary64 latitude sine now follows the selected source's double-double
algorithm, and three qsat coefficients preserve source binary32 literal rounding
before widening. Initial state, static fields and all 504,000 trajectory state
and output words match both retained Fortran and selected host-built source.
Binary64 exponential, logarithm, base-ten logarithm, power and arctangent also
follow the selected source's double-double graphs, tables, explicit fused
operations and final rounding, using integer binary64 arithmetic. Primitive
word comparisons include special values and underflow boundaries. The sine
reduction is limited to the source's small-argument domain, which contains
valid lake latitude angles. This remains bounded source and fixture evidence;
full lake and NVIDIA-device parity is unproved.

## Cumulus

Original Tiedtke (option 6) is a mechanical translation of the CUDA column unit:
one thread per column with its arrays in a per-call device arena, float division
and square root on the soft binary32 helpers, glibc's own `exp`, `log` and `tan`.
The per-row early return of the source is decided from the first ascent's
activity words, so the result matches the CUDA launcher for any row layout, and
a column whose cloud top reaches the source's undefined level is refused.

New SAS with its shallow arm (option 96) is translated the same way from the table-driven CUDA
unit. A small device kernel packs the per-column tables from the live surface fields, deep-search limits are
reduced per row, and the persistent rain rate and its clock-step increment stay on the device for the
driver's `advance_ppt` bookkeeping.

Grell 3D (option 5) is translated the same way, one thread per band column over a
device workspace of level arrays, followed by the neighbour-spreading kernel. It
runs at the default ensemble dimensions. The `cu_diag` averaging window and the
cloud-top hand-off for `lightning_option = 11` are supported.

## Workspaces and stochastic execution

NSSL columns above 128 levels use reusable device scratch instead of larger
per-thread arrays. Chunk size is derived from the device working-set budget, and
admission includes the deep-column workspace. Category order is unchanged.

Tiled SPPT, SKEBS and SPP share the resident Metal spectral owner. Patterns retain
global coordinates, member seeds, update cadence and checkpoint state; overlapping
tiles gather the same realization. FFT/RNG work and the stochastic tendency
transform run on the host, with SPP samples uploaded for the actual tile's physics
consumers. This is not a GPU-native stochastic transform. Four-byte integer tile
carriers are copied as words, preserving category bits.

On WOOF 1.1.5, a prepared `woof ensemble --backend metal` case with active
SPPT, SKEBS and land-surface SPP matched resident and tiled history and checkpoint
words across two one-second updates. A fresh tiled run restarted after the first
update matched the uninterrupted endpoint, including all four stochastic spectra
and their seed and update metadata.

Tiled restarts preserve whether held PBL raw rates are available. An older
checkpoint without those rates retains its coupled tendencies until the next
due PBL call; tile buffers reuse their existing raw-rate storage when it runs.

## Ensemble products

A Metal ensemble runs the CUDA product engine: the same threshold table, the
same files, and the same Rust CPU reducer and maps. Member diagnostics (wind,
dewpoint, humidity, rain and QPF windows, reflectivity composite, visibility,
ice accretion, joint events) are the CUDA kernels transcribed to NumPy on host
copies of the member planes, with no fused multiply-add; transcendental calls
use the host libm. `retain_member_diagnostics` and threshold overrides for every
field are honoured. `cyclone_products` and `sounding_points` read CUDA device
state and are refused. Gust and UH-style fields appear only when the member's
history carries them, as on CUDA.

## WRF restart initialization

`woof run --wrfinput` with a WRF `wrfrst` file starts a Metal run: the shared
restore takes WRF's restart branch, and the physics overlay fills every cross-step
carrier from the file on Metal storage (host words written through the array
module, dry-mass recoupling through the Metal radiation sum). The clock opens at
the restart instant exactly as on the CUDA path.

## Data assimilation modes

Hydrostatic rebalance, incremental analysis updates (split, 3-D and 4-D), the
surface-pressure balance record, and general state and boundary perturbations
run on Metal states. Their host routines stage the fields they read and write
the result back through the array module.

Radar latent-heating tendencies run on Metal with the PBL extension, strict
suppression and window offsets (a slab of a larger domain) on the same kernels
as the CUDA path, and the apply step takes the domain extent as the CUDA kernel
does.

Grid and spectral analysis nudging run on resident, fixed-step regional grids,
including explicitly selected fixed nests. WRF PBL suppression, vertical masks,
field coefficients, moisture, held tendencies, `fgdt`, `fgdtzero` and end ramps
follow the source kernels. Spectral transforms use host-staged NumPy FFTs, as
host stochastic patterns do; this is a host-transform reference, not a claim of
cuFFT word identity. Physical-temperature pressure powers use host libm to match
the host-built CUDA reference. CUDA `globaltimer` stamps are unavailable; Metal
receipts omit per-stage device timings. Moving, delayed and tiled analysis grids
retain their explicit refusals.

Latent heat nudging (`--radar-tten-mode lhn`) runs on Metal: the observed rain
rate and the nudging step are the CUDA kernels with glibc powf and expf.

Metal latent heat nudging retains cumulative 64-bit point-step counts using
atomic low-word increments and explicit high-word carries. It keeps the case's
microphysics tendency limit, and the cycling report now distinguishes that
contract from the heating arm's companion clamp. Radar-window reading and
receipt reductions remain host computations.

Packed members are a CUDA batching mechanism; Metal runs members serially.
Prepared LETKF keeps Metal Jacobi in the process that owns the context.
Host staging and purely host worker paths retain their existing roles.

## Nested research cycling and Hex

The prepared cycling tool runs parent and child forecasts on Metal. Analysis
addition and the difference of the two SINT interpolations execute on Metal,
with binary32 rounding and gradual underflow for the sums and differences.
The child atmosphere also initializes through Metal SINT. The geometry
tables use the child's existing manifest-owned nesting slots.
The held hydrometeor envelope is built on the host and uploaded to the
selected resident boundary buffers.
Supported binary32 hot-start heights, vertical ramps, masks, warm and
clear-air increments, and clamps execute on Metal. Height uses rounded
full-level heights, then the mass point average and terrain subtraction.
The ramp uses the resident heights and feeds the resident pointwise kernel.
Tent and uniform weights preserve the source operations. The sine-squared
path requires the verified NumPy 2.5.4 Apple Silicon SIMD arithmetic.
Unsupported bounds, tiny sine arguments and nonfinite heights use the host
ramp. Temperature, saturation and provenance reductions remain on the host.
The host reads the ramp for diagnostics and pointwise fallback.
The registered Metal ramp matches the NumPy source across all three shapes
and two geometries. A public prepared cycle used the Metal ramp for two
observed members before and after a fresh-process restart. The 30,699,624
compared checkpoint and pending values at each endpoint matched the prior
host-ramp run, along with the scientific diagnostics.

Resident FP32 IAU uploads the attached increments once, forms ordered node
rates and applies mass coupling, boundary masks and map factors on Metal.
The host integrates the time weights and handles unsupported layouts and
workspace sizes. The device comparison matched 24,110 NumPy source values
across multiple nodes, both vertical staggers, mapped and unmapped domains,
open and nested boundaries, signed zero and subnormal inputs. A public
prepared 3D cycle saved and resumed in a fresh process. Its 30,699,624
checkpoint and pending values matched the host-staged run. The two active
parent trajectories and one child used device coupling for all five forced
steps.

Supported single-child binary32 correction positivity runs on Metal for
`clip` and `mean-preserving`. It widens the background and correction before
the zero comparison, and the host reduces the negative words in the original
order for scientific receipts. Other policies and unsupported layouts retain
the host path. Component comparisons match 168 words and exact receipts,
including 24 negative decisions. The public fresh-process prepared cycle
exercises the device path on the carried vapour correction and matches
30,699,624 saved values, with zero negative decisions in that forecast case.
Perturbation draws and filtering, other increment validation and constraints,
and child static and surface inventory remain host work.
For analysis on a resident Metal state, species loading and hydrostatic
pressure and geopotential integration run in software binary64 on Metal.
Live binary32 geopotential application widens the existing value, adds in
binary64, and rounds once to binary32. Zero increments preserve the original
word, and the live field is updated only after all columns validate. Staged
perturbation writes and layouts without enough workspace use the host writer.
Before/after capture, hydrostatic operand preparation and solver result
readback for receipts retain host work. Multi-field member
perturbations use the same Metal column solve after their host draws and
bounds. Tree checkpoints carry the child's evolved state and clock between
legs.

The pointwise hot-start outputs match the existing NumPy operator, including
nonzero subnormal increments. In a bounded MP28 cycle using an authenticated
background and synthetic radar observations, the Metal path and the host
pointwise reference match all 1,313 saved arrays at the first observed leg
and at the resumed endpoint. The comparisons include nonzero pending wind,
theta and vapour increments, parent and child state, and scientific headers.
The device height matches 5,187 source words across three geopotential layouts.
The public prepared cycle uses Metal height for both observed members and
matches the host-height baseline at both saved and fresh-process resumed
endpoints. Each endpoint compares 30,699,624 saved values and the scientific
headers and hot-start diagnostics.
The Metal hydrostatic solver matches 83,105 Rust host words for both
hypsometric options and mass modes, including 450 nonzero subnormal outputs.
With device species loading, it also matches 89,870 Rust output words.
Reversing the species order changes 490 reference pressure words in the
order-sensitive case. Additional species are packed in bounded column chunks.
The same public prepared case exercises three active Metal integrations in the
fresh-process resumed leg. Its checkpoint and pending forcing match all
30,699,624 values and the scientific headers of the host-integration run.
A separate public case with thermodynamic and hydrometeor perturbations
matches 31,264,104 saved values and both active member balance receipts
against the host-integration baseline.
In public one-shot cycling, two live device applications move nine columns
each. The device and original host writers match all 30,699,624 saved values
and the scientific headers across five domain checkpoints. A fresh process
matches the uninterrupted run on those same fields. Selected application
checks cover signed zero, subnormals, binary64 addition before the final
round, and preserving the live field when a later chunk fails.
A parent-inherited surface refines the atmosphere while retaining
the parent's surface column; active fine-grid terrain drag requires prepared
child orographic statistics.

For initial additive wind perturbations, the prepared cycling tool uploads
the host-generated increments and applies the masked additions to resident
winds. The same path is used when the member also perturbs temperature,
vapour and hydrometeors. Draws, filtering, thermodynamic and species updates,
and bounds remain on the host. Unsupported wind layouts and insufficient
device workspace use the host application. Untouched rim values retain their
original bits. Comparison against the existing host routine matches all 25,600
words. Two prepared HRRR free-leg runs match 1,150,128
pre- and post-perturbation wind words and complete draw provenance across both
perturbed members. Both wind components change in the interior, and each nested
forecast completes its leg. A prepared nested cycle with wind, temperature,
vapour and hydrometeor perturbations matched 31,264,104 saved values at each
of two endpoints against the host-wind path, including a fresh-process resume.
The two active members used the resident additions with no wind readback
during perturbation. Their draw and balance receipts matched the host-wind path.

`woof go --backend metal` runs a prepared multi-domain tree through the same tree runner
CUDA uses, with the Metal boundary interpolation, feedback, smoothing and tree-checkpoint
resume. The single-domain runner refuses nests and hierarchy bundles on both backends.

Combining nesting with radar tendency forcing remains a shared upstream refusal:
the cycling tool has no child radar-forcing owner. Removing that refusal would
apply inconsistent forcing across domains.

Regional Hex LTS follows CUDA WOOF's rate classing and reflux scheme, including the
hold at rate 1 in the specified/relaxation zone. Explicit classing is an instrument
and is identified as such in the receipt; its contents are bound to a checkpoint.
The shared ladder, limiter and partition restrictions still apply. As upstream
documents, local time stepping is an intentional WOOF extension to native MPAS;
its interface conservation includes binary32 rounding.
A short public regional run with an explicitly placed interior rate interface
exercised reflux and produced identical continuous and resumed state. Its
comparison with the fixed-step control establishes neither broad numerical
accuracy nor a spacing-based performance result.

The port's built-in `wsm6_column` provider rejects `--source-table` itself.
The `thompson_aero_column` provider uses the shared two-phase Metal engine,
including its eleven transported scalars, aerosol initialization, tendencies,
flux diagnostics, and checkpoint inventory. Its allocation estimate includes
additional transport copies and the column scratch registry. Backend identity
is bound to the checkpoint. The `mynn_ruc_gsl_column` Metal seam shares the selected regional MYNN, RUC
and internal GSL drivers, seals the nine-layer soil and static identities, and
prices their workspace. A short genuine GFS-driven regional case has identical
repeated and resumed histories and checkpoint arrays. This is not CUDA parity
or forecast skill; it requires genuine GSL mesh statistics and nine-layer soil.
The same suite admits synthesized MYNN `bl_mynn_version` and
`bl_mynn_mixlength` selections supported by the upstream generation table,
with the existing Thompson, RUC and GSL inventory. Both WRF 4.6.1 and GSD 4.1
run with mixing length 1 or 2; other generation controls remain closed. A short GFS-driven case
has identical continuous and resumed histories and checkpoint arrays, with the
selected generation bound to the checkpoint. All four version/length combinations preserve their generation and carried
state through public forecast restart. The MYNN caller also matches
host-compiled selected CUDA source on separate cold/warm column comparisons.
This does not establish complete suite or NVIDIA-device parity.
The Thompson row executes on Metal with the upstream 1.1 source generations;
its public forecast has word-identical resumed/continuous checkpoint and
history arrays across a radiation call.
Classic Thompson `thompson_version="wrf_44"` (`mp_physics=8`) runs the generated WRF v4.4 column
on Metal for up to 256 levels, translated from the CUDA unit by `tools/thompson_column_metal`; it matches the host-built
CUDA column word for word on every output and carries identical resumed and continuous restarts.
TEMPO (`mp_physics=900`) runs its binary64 column on Metal for up to 512 levels, using 64/128/256/512-level kernel tiers and translated from the CUDA unit by
`tools/tempo_metal` over binary64-on-integer arithmetic; it matches the host-built CUDA column word for word on every output
and carries identical resumed and continuous restarts. Current TEMPO numerical evidence uses synthetic table data; execution through the public command with authoritative TEMPO assets is still unverified. CUDA TEMPO has no corresponding fixed layer ceiling, so Metal's 512-level limit remains an additional restriction.
All three in-tree upstream providers reject source tables. The point-source treatment belongs to an optional
`woof.hex.rows` provider; none of these built-in rows supplies that algorithm.
Source treatment and `woof hex pair` therefore remain unavailable without a
source-capable provider and its corresponding Metal implementation.

Generic registry selection on Metal currently supports the WSM6, YSU,
revised MM5, Noah-MP and RRTMG 4/4 plan with `icloud=2`, gravity-wave drag
off and convection off. The shared upstream planner checks its physics,
transported species, carried fields, radii, soil and diagnostic inventory;
the checkpoint seals the generic identity. Public continuous and resumed
fields and metadata agree. A separate caller comparison covers held-PBL
continuation. This is ordinary Metal arithmetic, with no independent whole-provider CUDA
comparison. Other generic plans remain closed.

Native `mpasphys` supports the bounded regional convection-permitting route
described below. Other suites and selector combinations remain gated.
Native Thompson coefficient transport authenticates all four lookup files,
the auxiliary file and the complete upstream decoded-output certificate.
The host reads record framing and converts integer byte order; FP64 arrays
reach Metal as exact uint64 words. No table or scheme arithmetic runs on the
host in this transport. The existing CUDA Rust-decoder path is preserved.
Native Thompson input/output conversion and precipitation reset/accumulation
are implemented as explicit components, including top-level handling,
all-level number tendencies and the named WB3 correction. Bounded selected-source
checks cover 2, 9, 129 and 256 levels, sampled subnormal fields and bucket
boundaries. The native Thompson column process now uses source-sized device
workspaces for 93 binary32, 64 binary64-word and seven Boolean planes through
256 levels. Ordered column chunks reuse one workspace; the caller still owns
other resident allocations and outstanding submissions. Bounded selected-source
checks with authenticated tables match at 4, 49 and 256 levels for the tested
effective-radius flags. Native relative humidity also matches selected source,
including a subnormal-vapour case. A diagnostics-off native Thompson SlotCall
now composes precipitation, conversion, ordered microphysics substeps and
publication through the shared Registry. Bounded selected-source checks cover
two heterogeneous columns, two substeps, radii enabled/disabled and WB3 on/off; a hydrated Python-driver
check preserves all pool fields through in-memory rollback. That check seeds
manager state and isolates other schemes. Radar and nest-look remain refused.
The bounded default-profile native initialization and public continuation below
join these components; broader native provider configurations remain unfinished.
The native MPAS v8.4.2 work includes resident pool views, preparation and
transport operators, held tendencies, cloud fraction, MYNN surface layer, gravity-wave drag,
monthly surface updates and diagnostic buckets. Ozone includes latitude,
cyclic time and vertical interpolation; held-source coupling includes the
single-rank dummy-cell convention and refuses unresolved halo endpoints.
Native Noah has land-use, soil initialization, glacier and land-column kernels
compared with selected source using authenticated tables. Its staged driver
preserves live values across pre/solver/post kernels; captured native and
selected-source branch comparisons cover its bounded input domain. Caller-bound
pool conversion, module-location state and checkpoint replay are implemented.
The native Noah text-table decoder is packaged for Darwin with its unchanged
Rust sources and a source/artifact manifest, as a companion to the complete
Darwin wheel. Source distributions exclude the compiled helper; clean its
staging with `python tools/mpasphys_v842_noah/build_decoder.py --clean`. Rebuild it with
`python tools/mpasphys_v842_noah/build_decoder.py`. An explicit resource packet
may omit `rust_decoder` to select this helper. Table parsing remains host work;
existing explicit decoder descriptors retain their certificate checks.
With explicit decoded Noah resources, the session binds registry, manager,
complete `physics_init` and finalization callbacks, including all local cells
and refreshed cached views. Cold initialization sets Noah layer thickness;
restart initialization preserves it. Common directions, calendar and metadata
setup still run on the host, including the microphysics top-level scan and
Thompson droplet defaults. Table startup records the explicit resource-owning
services selected by source metadata; missing owners refuse before manager
initialization. Complete scheme wiring remains caller-bound. When RP6 interval coupling is selected, the session owns
the native rain, snow, graupel and frozen-fraction window on Metal and hands
its totals to land and sea-ice calls before restoring the current-step fields.
Bounded source checks include subnormal precipitation, presence branches and
reset retention; actual-driver due/held/due checks preserve the window through
in-memory rollback. Native restart-file wiring remains unfinished.
The native MYNN surface layer also applies the upstream RP3 fractional sea-ice
skin-temperature correction when selected. Its source comparison covers cold
and warm calls, the two concentration safeguards and the recovered-temperature
floor; ordinary calls retain the uncorrected source path.
Native MYNN PBL components implement mixing/specific conversions, tridiagonal
solves, level-2 closure, state prediction, stability, boundary-layer height,
mixing-length selector 2, five-pass initialization, closure-2.5 turbulence,
cloud-PDF selector 2 condensation and the tendency/moisture solves. Bounded
selected-source checks cover these components, including discriminating FMA
and subnormal words. The eight-plume EDMF path and its iterative condensation
also have bounded selected-source comparisons with active and inactive columns.
Per-column surface controls and full interface carriers are preserved in the
predictor. Native cold/warm preparation, selector-1 surface controls, active
dissipation heating and exchange coefficients also have bounded source checks.
The heating stage requires the positive-normal QKE domain established by the
preceding source predictor; the disabled branch remains the caller's zeroing
operation. A bounded PBL SlotCall now composes these stages, preserves
prepared and Registry tendency destinations and clears disabled diagnostics.
An independent unpartitioned selected-source host comparison covers 49 outputs
in cold/warm, original/RP1 and absent-species cases; real Registry driver checks
cover original every-step and RP6 interval scheduling with pool, driver and
service rollback. Other schemes and framework startup are isolated in those
checks. The caller currently requires non-cycling default closure/selectors,
EDMF enabled with its output disabled, and jointly enabled or disabled aerosol
number species. RP7 now uses the native density/height pressure difference,
with a separate rounded-arithmetic kernel and the dedicated gravity constant.
Selected-source checks include its zero pressure-difference discriminator
and RP1 combinations. Top-down/downdraft, stochastic PBL, TKE advection and
unproved options remain refused. The native predictor's budget-on branch reads
an uninitialized top-level source value and stays refused; this is separate
from supported regional `tke_budget=1`. Validation statuses and scan checks are
read on the host. Complete native forecast and restart-file wiring remain open.
Native Grell–Freitas now gathers source inputs, executes all nineteen native
procedures with structured control flow and publishes tendencies, mass flux,
precipitation and status diagnostics on Metal. The current caller supports the
default deep/shallow closure selectors and at least sixteen levels. Its scratch
arena is sized from the source requirements and reused sequentially across
columns. Merged active/inactive fifty-level column comparisons match all
2,136 selected pinned-host-source input/output words. Actual Registry checks
cover due/held/due cadence, precipitation accumulation, halo preservation and
whole-session rollback with other schemes isolated. Shared service registration
owns its source-empty first convection update; the second update uses the
existing precipitation accumulator. Full-provider composition remains open;
these results do not establish NVIDIA-device or independent Fortran parity.
Native output diagnostics retain unclipped prognostic water. Cold-start EOS
and base-only restart refresh use the native binary64-FMA power core in software
binary64, retaining carried perturbation words in the base-only branch. Bounded
finite-input comparisons, including sampled subnormal fields, match selected
source. Complete native cold-start and dycore publication wiring remain open.
Native post-land-surface diagnostics publish `q2`, `t2` and `th2` on Metal
using the current surface pressure. Both Noah and Noah-MP pool bindings and
transaction replay match selected source in bounded checks; the disabled arm
preserves all words. Constants and callback scheduling remain host work.
Native sea-ice columns also carry their module locations and publish pool
updates only after source fatal checks succeed. Their selected-source checks
cover mixed columns and restart replay. The RP3 correction uses actual sea
surface temperature and the skin temperature at the sea-ice call, retaining
the upstream decomposition and clamp order. Its component and composed
selected-source comparisons match every compared word. An actual driver
fixture exercises fractional ice, due/held cadence and transaction replay
with stationary injected dynamics; this does not establish forecast parity.
Noah and sea ice bind through shared
session registration, using the existing source Noah binder and descriptor
profile aliases. Separate actual Python-driver checks exercise solve/halo
extents, module state and whole-session replay with other slots isolated;
independent numerical parity of a complete surface-coupled native forecast
remains unverified. A real public Hex dycore candidate has also exercised name-mapped native transport, including the dummy
column. A device adapter now writes core state carriers, scalar rows and native
output diagnostics into the resident pool views without per-carrier host
staging. It is exercised on both state levels of a real public forecast
candidate; cold EOS also matches selected source on private candidate copies.
Scheme initialization remains isolated in this public-candidate check. Native
wind reconstruction now uses the same pinned GNU trig graph selected by the
v8.4.2 CUDA carrier compiler, preserving source accumulation order and sealed
geometry. A bounded regional comparison includes signed zero, subnormal
coordinates, explicit donor selection and refusal after geometry mutation.
The real fifty-five-level public candidate matches all 44,110 wind words against
that selected source and publishes both native state levels. The ordinary
v8.4.1 trig path is unchanged. The shared Metal geometry now retains and checks
an explicit regional donor, as CUDA does. Wind validation synchronously reads
one four-byte status word before publication; the native CUDA resident path
defers that check. The enclosing transport boundary still downloads resident
fields for explicit native publication. Selected-source
comparisons and composed driver checks cover these components. Ozone initialization is also
available as an explicit component callback. These isolated checks do not
establish full forecast parity; the bounded public integration is described below.
Framework calendar handling,
pool hydration/publication and checkpoint encoding use host code.
Native radiation raw-table ingest also uses host code: Metal reuses the existing
Fortran-record reader and declaration schema on the authenticated MPAS/WRF
assets, preserving float and integer words before device initialization.
All 115 LW and 103 SW raw fields match the independent Fortran manifests.
This intentionally replaces the CUDA route's Rust transport executable;
it does not claim that executable's serialized-output certificate. Device LW
table initialization matches all independent reduced coefficient digests.
All sixteen LW gas bands, transfer march and fixed-depth solver are implemented.
Bounded comparisons match selected source with authenticated coefficients,
including sampled subnormal branches. The LW SlotCall groups columns by the
native added-top-layer count, prepares the extended atmosphere and 38 species,
samples clouds, handles ozone climatology and publishes fluxes, radii and heating.
The source capacity is 128 radiation layers including added model-top layers.

SW initialization, calendar controls, sounding preparation, KISS cloud sampling
and output conversion also match selected source in bounded component checks.
The SW SlotCall includes day selection, the native Thompson aerosol route and
publication of flux diagnostics and tendencies. Fourteen complete comparisons
match host-compiled selected source across clear/cloudy, day/night/zero-cosine,
overlap and land/ocean aerosol cases; the host solver also matches independent
captured Fortran fluxes. Independent LW comparisons cover four captured native
Fortran columns: boundary fluxes and radii match, while one column differs in
four internal downward-flux/heating words. Metal matches host-compiled selected
CUDA source for that column. The default comparison fails on these differences;
an explicit diagnostic mode checks only the four recorded word pairs.
These results do not establish NVIDIA-device parity.

Both radiation services bind through the shared session slot registry and use
its resident pool views and transaction state. Hydrated Registry checks exercise
the actual Python PhysicsDriver/PhysicsManager, held cadence and replay with
SW aerosol off/on and LW ozone off/on, including two LW column-depth groups.
Other scheme slots and framework preparation are isolated in these checks.
Slot-pool access shares existing resident caches and reads live host calendar
and scalar words. Noah also uses its shared source pool binder through these
views. SW orchestration reads day counts and validation statuses on the host;
LW also reads layer counts and groups columns on the host. Weather and flux
arrays remain resident between explicit boundaries. The bounded default-profile
public route below joins the supported slots and carrier restart; other native
providers and full independent forecast parity remain unverified.

A coupled caller check now runs the actual surface layer, Noah, MYNN PBL,
Thompson, held tendencies and RP6 precipitation window in source driver order.
Its ice-free columns exercise the sea-ice slot boundary without activating the
sea-ice column solver. Five live Thompson dispatches match 26,160 selected-host-
source words. A live cold MYNN call after the actual surface handoff matches
49 outputs and 4,806 words using all eighteen native constants; a successful
live Noah call matches 286 published state, flux and location words. Whole pool
and service state also match after a fresh-process sealed-carrier continuation
across the PBL alarm. Dynamics and winds are
explicit fixture inputs, and serialized dycore carriers are checked separately;
this does not exercise a resumed dycore or admit a public native forecast.
The fixture retains the saved RP6 configuration for the state-bearing Metal
carrier while using restart initialization. The shared native-file resolver
still drops RP6. The public Metal carrier uses its already settled configuration
and carries the session-owned window. The public numerical comparisons below
use the default surface cadence; RP6 cadence is exercised in this isolated fixture.

The native dycore callback boundary now supports Registry-selected mass loading,
first-stage density sources and microphysics before regional boundary resets.
It adds carried microphysics heating before mixing and held physics, zeroes the
carry in specified cells, and removes it at final RK recovery. Native recovery
uses the CUDA caller's separate-rounding power graph; ordinary Metal recovery
keeps its existing arithmetic. Bounded selected-host-source comparisons match
644 heating, specified-cell and recovery words. Nonpositive, subnormal or
nonfinite Exner arguments are explicitly refused because CUDA's double-power
fallback is not implemented. The ordinary public Hex comparison retains every
saved field word. These component and preservation checks do not establish a
complete native forecast or restart.

Native pool and stream transports now use the existing dynamic netCDF loader
on Darwin, with installed Python-wheel library discovery for child processes.
They build and stage through the native bridge tool without linking a private
netCDF library. Real four-soil input has been decoded and its soil carriers
round-tripped. A default-profile constructor also binds all native Metal
services and authenticated tables. The four framework startup callbacks run
on that decoded input. Initialization probes supply explicit diagnostic location seeds; their
pre-call values do not establish native module-state parity. The real regional
startup also reconstructs the core geometry and state before framework startup.
Scientific input reconstruction and native table transport execute on the host;
cold thermodynamics, native winds and resident carrier packing execute on Metal.

The native step wrapper joins actual limited-area dynamics and default-profile
physics, publishes the post-boundary endpoint before advancing the native clock,
and uses the mode-specific physics clock described below. Live MYNN, Noah and
Thompson calls match 2,162,404 selected-host-source words in an exact-mode step;
510,195 published endpoint words match the returned post-boundary atmosphere.
Other callbacks are exercised without independent source comparisons in that
joined execution. Correct mode settles corrections against the same executable
slot inventory used for dispatch: RP1, RP3, RP7 and WB3 are applied where
applicable. Ordinary routes reject Metal slot overrides.
Correct-mode convection forcing clears the source-defined boundary mask after
transport handoff, including the padded dummy column; its environment opt-out
is preserved.

A three-step correct-mode case crosses due, held and due radiation/convection
calls. A fresh process restores its sealed first-step carrier into actual padded
dycore arrays and native physics, preserving held tendencies, endpoint state,
services, boundary time weights and absolute alarm phase. Both resumed endpoints
match the uninterrupted Metal execution byte for byte across 14,865,664 array
elements each. This is same-Metal continuation evidence, not an independent CUDA
forecast comparison.

`woof hex forecast --backend metal --physics-backend mpasphys --physics-suite convection_permitting`
dispatches the native convection-permitting profile through shared native pool decoding,
namelist resolution, stream scheduling and Rust output encoding. It requires
regional grid, static, init and LBC inputs. Broader native providers, strict
numerics, moving meshes, stochastic/chemistry state and diagnostic fields without
a computed Metal authority remain refused. The route requires Darwin native
transport executables. The complete 80-artifact bridge roster was verified separately.

Metal writes supported state streams on the stream writer's output clock.
The default profile keeps the MPAS physics manager's `output` and
`diagnostics` alarms inactive. The Kessler profile below supports
relative humidity on the `diagnostics` alarm.
An enabled scientific output stream that requests native relative humidity
or radar fields without a computed producer is refused regardless of its
name. The native restart writer can still serialize those raw pool fields.
The public stream comparisons cover the written fields and saved Metal
state at the exercised endpoints.

The native profile also supports `config_o3climatology=.false.`, using the
MPAS fixed vertical ozone profile in both RRTMG spectra. The existing
authenticated resource load remains required. The selected host-compiled
CUDA slot comparisons match all sampled shortwave and longwave output
words. A three-step public daylight forecast exercises both spectra across
due, held and due calls. Its fresh-process continuation matches 14,865,664
checkpoint elements, 75 saved native variable elements and 380,724 history
elements at each resumed endpoint. Changing the ozone option at restart is
refused by the native checkpoint physics identity.

Native MYNN accepts `config_mynn_dheat_opt=0` and `1`. Disabling it sets
the dissipation heating to positive zero as in the source. All 49 published
slot fields match selected host-compiled CUDA source in cold and warm
columns at both settings. The public heating-off forecast changes the
thermal tendency relative to heating on and its fresh restart matches
14,865,664 checkpoint elements and 380,724 history elements at both held
and due endpoints. Changing the option on restart is refused.

Native MYNN also accepts `config_mynn_edmf_mom=1`. The selected-source
column comparison exercises active plume momentum flux in both wind
components and matches 4,806 published words. A public daylight case
changes 30 words in each first-step wind tendency relative to momentum off.
It matches 14,865,664 checkpoint values, 75 saved native variable values
and 380,724 history values at each fresh-resumed endpoint, with identical
active plume counts. Changing this option at restart is refused.

Native MYNN also accepts `config_mynn_mixlength=1` and
`config_mynn_closure=2.6` with `config_mynn_edmf_mom=1`. It uses BouLac
mixing lengths and five source initialization iterations. The Metal
validator checks that each prepared height scan can finish within the
column. The `icloud_bl` option feeds the separate radiation merge. In
the pinned source, MYNN itself uses it only to copy unused scratch fields.

Native convection-permitting forecasts accept `config_gwdo_scheme='off'`.
The shared driver skips orographic drag and retains MYNN wind tendencies for
edge projection. The public first-step comparison against active drag
matches the source-defined wind-tendency differences and leaves the thermal
tendency identical. With drag off, the six drag diagnostics remain zero on
this cold-start fixture. A fresh restart matches 14,865,664 checkpoint
values, 75 saved native variable values and 380,724 history values at each
held and due radiation endpoint. Changing the drag selector on restart is
refused.

Native convection-permitting forecasts also accept
`config_convection_scheme='off'`. They omit the Grell–Freitas service and
skip its scheduled calls. The shared first-step hook has no physical work
for this selector. A daylight control has active convective tendencies
and precipitation, while the disabled fixture leaves those fields absent
or zero. Its fresh restart matches 14,662,357 checkpoint values, 75 saved
native variable values and 358,324 history values at each held and due
radiation endpoint. Changing the convection selector on restart is refused.
With native Thompson aerosols and both convection and drag off, a daylight
continuation matches 15,148,770 checkpoint values, 75 saved native variable
values and 689,524 history values per endpoint. Optical-depth diagnostics
and shortwave heating are active in both runs.

The native convection-permitting suite also accepts
`config_microp_scheme='mp_wsm6'` and a state history stream that leaves
the MPAS diagnostics alarm inactive. WSM6 uses
the native scalar order, precipitation accumulations and rain bucket.
Its initialized constants and column workspace are owned by the Metal
service. The working-set and buffer limits bound the admitted mesh size.
Native correct mode applies WB3 precipitable-water integration and WB5
precipitation transport when enabled by the shared correction ledger.

The selected source comparisons cover initialization, active rain, snow
and graupel, and the hydrated slot through two calls and checkpoint replay.
The public state history and fresh restart match 14,688,422 checkpoint
values, 75 saved native variable values and 380,724 history values at
each resumed endpoint. The shared `history` stream also writes the
restored boundary record, whose 380,724 values match the original run.
Selected source comparisons also cover all four WB3 and WB5 selections.
Public correction comparisons changed 400 precipitable-water words for
WB3 and 104 accumulated and 104 step-rain words for WB5. The WB5 fixture
had positive surface rain and zero surface snow and graupel. The
selected-source column comparison includes active snow transport.

With paired RRTMG, `config_microp_re=.true.` computes WSM6 cloud, ice
and snow radii from the updated column before native writeback. Both
radiation schemes consume those saved diagnostics on their own cadence.
The public radius witness has 115 cloud, 1,498 ice and 1,465 snow values
above background after the first step. On the next due step, 14,045 LW
and 13,942 SW heating values differ from the radii-disabled control.
Both spectra retain their heating exactly on the held step. The
radii-enabled fresh restart matches 14,688,422 checkpoint values,
75 saved native values and 380,724 history values at both endpoints.

The native convection-permitting suite also accepts
`config_microp_scheme='mp_kessler'`. It uses the native Kessler column and
writeback, with device workspace sized for the mesh and native WB3
precipitable-water integration in correct mode. The named `diagnostics`
stream can produce relative humidity on a recurring interval aligned
with the timestep. Other history streams may read the held humidity.
The initial record carries input or Registry-default humidity. The
native `output` alarm and reflectivity fields remain unavailable with
this configuration.

Selected source comparisons cover active rain, small-magnitude operands,
rain bucket rollover and two carried calls with WB3 enabled and disabled.
They also cover due, held and replayed relative humidity. A public
three-step forecast produced Kessler rain in 164 to 188 cells. Its
diagnostics call changed 22,000 humidity values. A fresh restart matched
14,222,460 checkpoint values, 75 native variable values and 136,259 history
values at both resumed endpoints with 35 fractional-ice cells.

The native convection-permitting suite also supports the paired
`config_pbl_scheme='bl_ysu'` with either `sf_monin_obukhov` or
`sf_monin_obukhov_rev` as its surface-layer selector. Other suite
selectors use their defaults. State history must leave the MPAS
diagnostics alarm inactive. YSU carries momentum, temperature and moisture
tendencies and boundary-layer diagnostics. The two MM5 schemes use their
own source tables and column physics. They share the land and fractional-ice
exchange fields and apply RP3 in correct mode.
Top-down YSU requires finite positive layer thicknesses and a prepared
interface above 500 m in each column, which the host checks before dispatch.

Selected source comparisons cover both YSU top-down settings and both
MM5 schemes' initialization, surface diagnostics and fractional-ice
exchange through native pools and replay. For each scheme, the daylight
public continuation matches 14,182,761 checkpoint values, 75 saved native
variable values and 381,924 history values at each of two endpoints with
35 fractional-ice cells. The original and revised schemes differ in
surface fluxes and winds on the public case.

The native regional `none` suite runs with the upstream all-off selectors.
It skips moist services and initialization, supplies zero physics sources
to the RK step, and publishes resolved `rthdynten` every step. Dry boundary
conversion uses zero vapour with an empty scalar inventory. The CUDA
regional boundary loader still requires a vapour scalar.
Public `woof hex forecast` exact and correct runs both pass fresh restart
at two endpoints. Each comparison matches 361,052 dycore values,
7,669,859 native-pool values, 12 native variables, and 268,324 history
values. A separate live check matches the source-ordered heating expression
and verifies zero sources with active resolved heating.

The native regional `mesoscale_reference` suite runs its default WSM6,
New Tiedtke, YSU, revised MM5, Noah and RRTMG selectors on Metal. New
Tiedtke retains the whole solve slab for its ascent trigger and publishes
thermal, moisture and momentum tendencies, step rain and rain rate to
the native pools. Held rates and accumulated rain use the shared manager
and surface update. The alarm follows the configured interval and the
scheme uses MPAS's rounded count of dynamical steps. The host forms that
binary32 scalar timestep and checks one device status word before
publishing. An active downdraft requires a native half-level within 60 hPa
of the surface.

Selected source comparisons cover the assembled active scheme on two
calls, native pool publication, subnormal forcing and the downdraft guard.
In a seeded daylight public forecast, convective rain reaches 23 and 24
columns on the two due steps and the middle step holds the tendencies
exactly. A fresh restart matches 13,825,871 checkpoint values, 75 saved
native values and 380,724 history values at each resumed endpoint. This
suite uses state history without the MPAS diagnostics alarm.

The native convection-permitting suite accepts
`config_microp_scheme='mp_thompson_aerosols'` with the native aerosol table
and regional aerosol boundary fields. Initialization fills globally zero
profiles and preserves prescribed profiles. It stages native host pools
through the Metal initializer, then copies the initialized scalars into the
resident dynamics state. The aerosol column, substep emissions, and scalar
and tendency publication run on Metal. The certified aerosol auxiliary uses
its own table order and decoded-word identity.

The startup and joined slot match 274,540 and 518 selected CUDA source
words, respectively. The slot matches in exact mode and with the WB3
correction. The public regional check uses controlled aerosol concentrations
and exercises live aerosol state, emissions, and tendencies. Correct-mode
fresh restart matches 15,352,077 checkpoint values, 75 saved native
variables and 711,924 history values at each resumed daylight endpoint,
including active optical-depth diagnostics. Exact mode
retains the shared native step-one radiation cadence on restart.

The correct-mode `--physics-profile hrrr-like-aerosol` uses the upstream
prepared aerosol fields and merges carried MYNN cloud into the radiation
copies before both spectra. After the first physics step it also publishes
the carried cloud fraction. The prognostic condensate fields are unchanged
by this merge. The public regional forecast uses authenticated WIF climatology.
It matches 66,000 source-calculated merge words at each radiation alarm,
including active liquid, ice and fraction changes after the first step.
Its fresh restart matches 15,352,077 checkpoint values, 75 saved native
variable values and 777,524 history values at each held and due endpoint.
The correct-mode `--physics-profile hrrr-like` also uses this aerosol and
cloud coupling with the BouLac MYNN settings above. A public daylight
case with WIF input matches 49 published MYNN fields and 1,034,800
selected-source words on each of its hydrated cold and fresh-resumed
calls. It also matches 66,000 source-calculated cloud merge words per
alarm, with liquid, ice and cloud fraction replacement active on the
later alarm. Its fresh restart matches 15,352,077 checkpoint values,
75 saved native variable values and 799,924 history values at both
resumed endpoints. The checkpoint identity includes the selected
profile. Native radar and nest-look outputs remain closed on Metal.

The RRFS MYNN-SFC v600 surface-layer component generates its four stability
tables and runs the packed land, water and ice columns on Metal. Software
binary32 comparisons preserve the source's distinction between zero and
subnormal surface humidity. The direct native driver packs the atmospheric
levels and surface fields, passes the first call's boundary-layer height
to the water call, blends the upstream fractional-ice fields, and publishes
the sea-ice handoff. Public `woof verify` matches 75,488 table and packed
column words and 110,334 native-driver words against the retained CUDA
source compiled for the host. A separate subnormal probe matches 17,871
selected words. Native-driver
coverage includes cold, carried and restart initialization inputs, plus both
sides of the ice thresholds. The fork Registry binds the same driver through
its prepared profile aliases. The Registry comparison matched 8,488 words
with the resolved HRRRv5 flux options, including the fractional-ice handoff,
untouched halo and profile storage, and prepared-state restore.

RRFS prepared aerosol-number carriers share the native selector mapping and
the same device statements as the equivalent v8.4.2 carriers. The selected
TEMPO/MYNN-EDMF Registry comparison matched 1,024 retained-source words across
the four number fields and preserved their prepared top and native scalar
halo words.

The RRFS startup path authenticates its TEMPO witness, land-use, ozone and
RRTMG tables before constructing the native slot services. Its monthly
surface callback updates the RUC leaf-area index, conditionally updates
vegetation and albedo, and uses the fork's declared dust fields when enabled.
The hydrated startup and monthly surface comparison matched 1,348 source
words. The forecast reads source-prepared fork inputs.

The RRFS TEMPO v303 prepared-column component uses software binary32 and
binary64, native-initialized lookup tables, and budgeted device workspaces
through 128 levels. A synchronous `woof verify` fixture matched 6,528 selected
words, including 384 independently captured native Fortran words from the
first call. The host-compiled CUDA comparisons include carried state, varying
optional fields, and active precipitation and hail diagnostics. Separate
active PBL radii and forced single-slot batching matched 1,536 source words.
Native MPAS preparation, once-per-step aerosol emissions and the carried
temperature conversion matched 6,360 selected words against host-compiled
CUDA. A separate solve-only run matched 1,152 source words and preserved
the two halo columns. The post-column precipitation and tendency wrappers
matched 2,530 selected words with two active substeps, rain bucket rollover,
a vertical cutoff and protected halos. An external `woof verify` case matched
103,482 boundary words over two steps with two substeps each and replayed the
second step after reloading the first-step state into a new Metal context.
The component supports dry reflectivity.
Melting reflectivity remains gated on the upstream CUDA vendor-math contract.
The native radar and rolling-rate wrappers matched 290 selected source words,
including full and held outputs, device hourly overrides and active windows.
Full radar validates the 1 km interpolation geometry on Metal and reads status
back to the host before publication. Cold-start aerosol profiles and surface
sources match 40,280 selected solve-cell words from the retained framework
initializer, with independent extrema switches and a separate restart skip.
The initialized-table transport promotes native kinds on the host in bounded
blocks and preserves all 122,457,796 words of the authenticated reference blob.
`python -m tools.hexrrfs_tempo.build_native_tables` can rebuild the mixed-kind
Darwin Fortran witness from pinned source and lookup files. The verified
non-melting case reproduced the standalone native witness word for word.
The builder uses system libm and requires an independent reference digest.
Equivalence to the fork CUDA framework's decoder remains open.
The resident TEMPO slot binds level-two state and prepared pools. An external
`woof verify` case matched 6,921 selected source words, including the carried
second call. The active rolling-rate fixture matched 4,122 words per endpoint
after writing and loading slot state in separate processes. The portable
table witness gave the same result with table re-upload, and a changed table
word or checkpoint table digest was refused.

RRFS RUC surface forcing and fractional sea-ice boundaries use the retained
fork's source-order precipitation sum and distinct albedo, emissivity and
skin-temperature masks. The preparation and blend wrappers matched 1,536
selected source words across optional forcing, MYNN and other surface-layer
branches, both ice thresholds and untouched halos. The native saturation table
matched all 5,001 source words. Land and ice interpolation matched the source
at both clamps, around freezing and with halo sentinels. The packed sea-ice
solver and its bound workspace matched 4,592 retained source words through
two carried calls in an external `woof verify` case. The fixtures cover bare
and snow-covered ice, zeroed and NaN-filled workspaces, active precipitation,
per-column SPP flags, entry-context tracing and untouched halo cells. The
two-metre diagnostic wrapper matched 880 retained source words in a separate
external `woof verify` case with MYNN and non-MYNN exchange coefficients,
both saturation phases, active stability adjustments and untouched halos.
The pinned RUC table decoder matches the standalone upstream Fortran witness
for USGS-RUC and MODI-RUC with STAS-RUC soil. Each 955-word adapter packet
includes zero-filled unused category slots and integer character tokens.
The cold-start column initializer matched 408 retained source words using the
MODI-RUC witness in an external `woof verify` case. It covered frozen and warm
soil, water, fractional ice, restart and cycling skips, untouched halo cells
and refusal of uninitialized categories. A separate `woof verify` case
matched 6,044 words through the real RRFS Registry, including packet-backed
table publication, full-local startup, cached solve views, an untouched
dummy cell, and the live diagnostic callback.
The packed land solver matched 6,180 source words through an external
`woof verify` case that exercises gather, tile preflight, solve and soil
copyback. Cases cover the whole-tile first-lake stop and its preparation-only
prefix, bare and snow-covered active columns, carried calls with both
synthetic and native-initialized tables, and untouched halo cells.
The retained land and sea-ice callbacks also matched 1,426 words in a
bounded `woof verify` case using the actual fork slot registry, native
constants and solve-only Metal views, including the sea-ice blend and
published native pool words. The fixture constructs its own selected
column inputs and exercises the callback boundary.
RUC slot gathering requires the retained Registry precipitation fields. In
the RRFS profile with microphysics disabled, the Registry omits `rainncv`
and the retained CUDA gather still requires it.
Four live land columns matched 2,520 retained-source words at the packed
solver boundary, including the sensible, moisture and latent heat fluxes.

The RRFS held-tendency owner uses the retained fork selector mapping for
MYNN-EDMF and TEMPO. Its `woof verify` fixture matched 1,264 source words
across edge projection, tendency assembly and pool publication with TEMPO
aerosols enabled and disabled. The fixture covers native dummy endpoints.
The owner requires the first Runge–Kutta stage and refuses RRFS SPPT or
physical edge layouts that would require halo exchange or wider publication.
The RRFS startup pool adapter matched 2,387 source words through the actual
Registry in an external `woof verify` case. The cases cover full-local wind
directions, cold accumulators, species flags, DA cycling and restart,
positive, zero and negative ocean mixed-layer depths, and both fork versions
with and without the optional microphysics top. The RRFS framework binds
land-use, RUC and TEMPO startup through the retained
callback sequence. External `woof verify` cases matched 339 land-use pool
words, 1,069,540 aerosol pool words and 1,308 words across selected startup
phases in a hydrated Metal session. The aerosol extrema decision uses the
host rank reducer and the verified cases exercise its returned values.
The integrated startup case leaves PBL, radiation and gravity-wave drag off.

The RRFS cloudiness callbacks publish TEMPO and MYNN radiation carriers through
the native Registry, including the pinned TEMPO zero-fraction default and the
optional Grell–Freitas–Li fraction and condensate additions. A Registry
comparison matched 2,030 source and preservation words across the first and
subsequent steps, unchanged halo diagnostics, prepared carrier restore and
refusal before publication when an output carrier has the wrong shape.
RRFS RRTMG shortwave selects the equation of time from the configured source
revision and applies it before day-column selection. Its public component case
matched 3,586 retained-source words across both RRFS revisions and v8.4.2,
including heating, flux profiles, an elapsed-day rollover and TEMPO aerosol
optical depths from the original vapour and aerosol profiles. The two RRFS
coefficient sets produced distinct cosine words. The selected longwave slots
matched 434 source words, including ozone, heating, fluxes and TEMPO and NSSL
radius diagnostics with enabled, disabled and inapplicable output handling.
A public one-step v8.4.2 forecast retained its checkpoint and history words.
On the daylight forecast, the two due cloudiness calls each matched 338,709
retained-source words at the carrier and publication boundaries. The later
call included nonzero cloud fraction and condensate. Four live shortwave
columns at each due call matched 1,508 retained-source words, including
heating, aerosol optical depth and full flux profiles.

The RRFS route binds the orographic UGWP slot when nonstationary gravity waves
are disabled. Its device workspace is sized from the actual column depth and
priced against the Metal buffer and working-set limits. A public component
comparison matched 216,752 source words at 12 and 160 levels across the grid
spacing taper, directional boundaries, active drag and diagnostic branches,
and two initial workspace patterns. A hydrated Registry comparison matched
18,626 source and preservation words, including rounded wind-tendency
differences, held optional diagnostics, untouched halo and prepared top
storage, prepared-state restore, and invalid PBL-depth refusal before
publication. The slot has no private checkpoint state.

The RRFS 44c0849 route binds the c6d08709 MYNN-EDMF boundary-layer slot.
The older RRFS route binds its separate c942418 kernel and diagnostic
inventory. Both pack solve columns into a reusable device workspace sized
from the retained source allocation plan and the Metal working-set limit.
The two revisions use distinct column types in the linked Metal library.
Corrected source
comparisons include cold and carried state, active cloud condensate and plume
mass flux, and partial batches that preserve halo and upper-interface words.
A column that ends inside a source height scan reports an invalid extent
before the slot publishes Registry tendencies or diagnostics.
A hydrated Registry comparison matches 126,836 current-source words and
126,596 older-source words across incoming fields, published tendencies
and diagnostics, and prepared-state restore. Chemistry, ozone and black-carbon
scalar coupling, fire mixing, SPP and DA cycling remain gated in this slot.
The native Metal writer captures published Registry host words and uses the
same netCDF runtime discovery as the ordinary native writer. With authentic
fork-prepared inputs, a verification run exercised the current HRRRv5 route
through a due, held and due radiation sequence. Fresh continuation in
correction mode matched 17,132,061 checkpoint values and 881,000 history
values at each of two endpoints. Longwave heating was active. Shortwave
heating was zero in this nighttime case. Exact mode follows native MPAS
restart initialization and starts the radiation cadence anew.

`woof hex forecast --backend metal --physics-backend mpasphys --physics-suite hrrrv5`
admits the current RRFS source row, its default scheme selectors and
supported native options on a fixed regional mesh. For moist native regional
scalar advection, select `config_monotonic=false` and
`config_positive_definite=false`. Metal refuses requests for the scalar limiter
before device startup. The daylight public run
covered 381 columns and 59 layers. After a checkpoint during the six-minute
radiation hold, the continued run matched 16,324,769 checkpoint values and
839,293 history values at each of two endpoints. Both spectra had 22,479
active heating values. The held heating and surface fluxes kept their words
until the next due call. Older RRFS forecast rows, strict Hex numerics, and
unsupported physics options and output streams retain their admission checks.
The joined restart compares the same Metal configuration, and the independent
retained-source comparisons cover the selected scheme boundaries above.
Regional stage-three heating uses the density tendency after the first two
boundary adjustments. At each compared call in the daylight run, all 22,538
heating words matched the retained source. Using the pre-adjustment density
tendency changed 17,700 words across 300 cells.
Regional boundary sampling and scalar density interpolation follow the
resolved RK order. Relaxation uses the named divergence damping coefficient.
Public order-2 and order-3 forecasts matched all 473,298 compared scalar
density words against the retained source. With a nondefault damping
coefficient, the order-2 run also matched the uninterrupted checkpoint and
history words after a fresh-process restart.
The current NOAA row accepts the explicit source pin and
`config_lbc_w='nearest'`. The resident update copies the nearest relaxation
cell to the interior interfaces of the specified zone after stage diagnostics.
It preserves the lower and upper interfaces and leaves the prognostic
vertical momentum in its existing owner. In the public unlimited-transport
case, 1,237,680 live boundary words matched the retained source over 54
stage calls, and 7,192 values changed in each call. Fresh resume matched
the uninterrupted checkpoint and history words at both endpoints.

On WOOF 1.1.3, public cold start, checkpoint export and fresh-process
continuation from both due and held boundaries match the retained uninterrupted
Metal state. Each
compared checkpoint contains 14,865,664 equal array elements; each completed-step
history record contains 380,595 equal elements after native dummy trimming and
declared dimension permutation. Timestamps, backend metadata and absolute alarm
phase are also checked. The initial first-call diagnostic frame has no retained
endpoint reference and is listed as excluded. Comparison manifests identify
fields and transformations. The ordinary public route retains all 104 arrays
and 1,148,932 saved values byte for byte. These are bounded same-Metal and selected
source comparisons, not NVIDIA-device or complete upstream forecast parity.

On WOOF 1.1.3, a public fractional sea-ice case also matched at both held and
due endpoints after a fresh-process restart: 14,865,664 checkpoint elements
and 380,724 history elements per endpoint. The prepared case had 43 active
ice cells, of which 35 had fractional concentration. This comparison checks
continuation on Metal; the separate sea-ice source comparison checks the
correction itself.

WOOF 1.1.5 correct-mode Metal continuation carries the original physics start
hour and continues the absolute physics step and alarm time. Output and sealed
checkpoint scheduling retain the segment clock. Exact mode retains the native
step-one restart clock. A correct-mode carrier without the physics start hour
is refused, as are foreign scheme flags or output accumulators that the Metal
session cannot restore. A public 1.1.5 fractional-ice restart matched the
uninterrupted run at both the held and due endpoints: 14,865,664 checkpoint
elements, 75 saved native variable elements and 380,724 history elements per
endpoint. The original physics start hour, absolute step and alarm phase
matched; the restart did not duplicate the checkpoint-boundary record. This is
same-Metal continuation evidence, separate from the sea-ice source comparison.

Scientific input reconstruction, native pool publication, calendar and monthly
weights, longwave grouping, table transport, output encoding and checkpoint
serialization retain explicit host work. Unused pre-call Noah and sea-ice
location indices receive deterministic zero rather than upstream undefined
storage; this intentional diagnostic-only divergence has no scientific consumer.
Active branches subsequently preserve the source I/J values.

Native Metal resources resolve portable packets with rebased native table paths
and the authenticated packaged Noah decoder. Packets without an ozone row use
their adjacent pinned ozone tables; partial local tables fail explicitly. The
installed-DATA route uses the existing manifest and installer without assembling
CUDA decoders. Both routes match all returned resource arrays against the
authenticated loaders. Offline installation verifies every pinned DATA file;
online first-use availability remains unverified after a download timeout.
Table decoding and transport remain host work.

A same-mesh Metal checkpoint adapter reuses the existing sealed forecast carrier
and adds provider-owned numeric, text, prepared and service state. Fresh-process
checks cover original alarm reset and correct-mode phase continuation at the
same timestep and an aligned timestep change. New-segment initialization keeps
its timestep-derived variables and clock; internal transaction rollback retains
its stricter clock identity. Earlier isolated checks use synthetic dycore carriers; the bounded joined
regional case above also exercises live dynamics continuation. Moving-mesh remap,
stochastic/chemistry state and broader provider restart configurations remain
unavailable through this adapter. The public same-mesh default-profile route uses
this carrier directly. Exact mode retains the upstream alarm reset
at a fresh segment; correct mode carries its phase.

The exact-policy library carries integer binary32 implementations of the regional
`diff_opt=2`, `km_opt=4` Smagorinsky coefficients, horizontal stress and scalar-flux
tendencies at all four staggers, and the open-boundary strip zeroing. They match the strict upstream
source word for word; the W stress uses the direct source form. This does not
open the full exact mode.

Strict source-word comparisons now also cover moisture sources, positive-definite
transport, forced scalar boundaries, Coriolis, carried mass, nest field coupling
and atmosphere preparation. The retained inverse nest-coupling operation has no
current upstream feedback caller, so its isolated comparison does not establish
live feedback coverage. Actual unsmoothed resident and windowed feedback
restriction and theta framing also have source-word comparisons. Strict
resident feedback smoothing and forced SINT boundary interpolation have actual
caller comparisons, including both boundary-strip orientations. Streamed strict
feedback smoothing remains refused. Retained heating uses software binary32
arithmetic with source-word comparisons covering mapped and unmapped cases,
subnormals and signed zero. The strict Dudhia column preserves the retained
source recurrence, table interpolation and pinned logarithm/power operations;
typed-caller comparisons cover clear/cloudy, day/night and tiny-water inputs.
The shared MYNN cloud merge also preserves strict binary32 order and gradual
subnormals for both legacy and GSD selection. Actual Dudhia caller comparisons
cover cloud selectors 0, 1 and 2. The full driver and dycore comparison also
exercises the shared upstream host solar calculation, but does not independently
verify it against WRF solar geometry. Ordinary Metal now follows upstream GSD temperature
partitioning and skips the boundary-layer merge for selector 2. These are
intentional corrections to ordinary cloud preparation; ordinary column
transcendentals are unchanged. Exceptional clipping semantics are checked
separately against the shared array expression.
Strict updraft-helicity columns and running maxima use software binary32
arithmetic, with source comparisons and resident reset, tracker-window and
restart checks. A seven-step specified-boundary strict dynamics comparison
also exercises active helicity, a diagnostic reset and both tracker windows
through a synthetic lifecycle tree checkpoint. Strict NWP diagnostics are
admitted for the supported specified-boundary profile; other diagnostic gates
remain in force. The shared completed-step sampling, low-top stand-in and halo
rules remain as documented in `woof/core/uh_diag.py`.

Strict climate statistics use software binary32 operation order for extrema,
means, variances, wind and precipitation flux statistics. Their 50 persistent
planes retain the shared output and restart lifecycle. Source comparisons
include synthetic nonzero producers, distinct windows and restart calls.
The admitted dynamics comparison also crosses a native one-minute finish and
reset with tree-checkpoint continuation and active skin-temperature and
shortwave producers. Invalid source window sample counts are refused before a strict step or a
direct diagnostic submission can mutate state. Ordinary climate dispatch is
unchanged.

Reference CUDA source compiled for host execution is not an NVIDIA-device
authority.

With both lateral axes open, a flat physics-off strict profile admits dry or
moist states without mixing, sixth-order filtering, upper damping, or climate
and precipitation diagnostics. Active inflow and outflow source comparisons
cover all stages of seven model steps and tree-checkpoint continuation. As in the
shared WOOF caller, open moisture uses unlimited transport with no strict clamp
even when `moist_adv_opt=1`; this does not implement WRF open-boundary PD
radiation. The ordinary Metal x-edge w mask still reads the previous row
where the shared caller reads the previous vertical level; strict Metal uses
the source level. Ordinary open-boundary parity is not established.

Restricted flat strict dynamics steps are admitted with all strict controls
enabled. The `diff_opt=2`, `km_opt=4` profile also carries Smagorinsky mixing
with periodic or specified boundaries, including active constant surface heat
and drag, with physics off or with the Dudhia-only profile below. Mapped dry
and moist step-and-checkpoint comparisons cover surface flux options 0, 1 and
2. The unmixed moist physics-free profile also supports specified boundaries.
All specified profiles retain the clock, width and linear-forcing guards.
Direct-step tree-checkpoint comparisons cover forcing crossovers and both
sides of radiation calls, using positive-definite and ordinary transport.
Other mixing configurations remain refused. A flat, one-dimensional-base moist profile with
periodic or specified boundaries also admits Dudhia shortwave only: LW, microphysics, PBL, surface, cumulus and auxiliary physics
are off, with icloud=1, either supported solar generation and eclipse disabled.
The WRF 4.8 generation uses device software arithmetic and pinned source trig;
the WRF 4.6.1 generation retains shared host geometry. The call calendar and
WRF 4.6.1 host calculation are not independently verified against WRF; public
CLI execution of this profile has not been established. Seven-step source
comparisons cover radiation cadence, all three dynamics stages and checkpoint
continuation. Periodic transport includes vapor, cloud water and rain with
either positive-definite or ordinary transport; extra moist species remain
refused. Flat mapped and rotating cases are included. Specified boundaries
support linear forcing, a bound matching clock, default relaxation widths and
no exponential sponge or W relaxation. Nonlinear boundary time laws remain refused. A synthetic
one-root tree checkpoint comparison covers forcing crossovers and resumes on
both sides of a radiation call, using direct dycore steps. It does not establish
tree-executor or public CLI behavior. Admission reads base
theta, terrain and dry mass to the
host on every step;
for a 3-D base it also reads the base pressure, geopotential, specific volume
and geopotential residual. These checks synchronize the device. Prepared-state theta-carrier adoption
uses the device binary32 sum-then-subtract expression, retains the original base
metadata and is idempotent across repeated initialization and restart. A
one-row flat, dry, periodic profile with inactive mixing admits the stock IGW
configuration. Two-step selected-source comparisons cover its RK stages and
tree-checkpoint continuation, including a separate nonzero staggered V-face
case. The public stock IGW benchmark passes its analytical gates; its metrics
use perturbations relative to the original sounding, while history `T` uses
the WRF theta-minus-300 convention. This does not establish NVIDIA-device
parity. Other one-row physics, mixing, terrain and moisture profiles remain
refused. Two-step
comparisons and actual checkpoint continuations against host-compiled selected
source cover this scope; it does not
establish NVIDIA-device or full forecast parity. Other strict configurations
remain refused.

The strict revised MM5 surface-layer component (`sf_sfclay_physics=1`) uses
software binary32 arithmetic and the selected pinned single-precision math.
Before its column writes, a device preflight and host status readback refuse
columns that may need the source's native binary64 logarithm rescue. The
preflight is conservative: it checks all five possible numerator forms even
if a conditional branch would not use them. This restriction does not assert
that excluded roughness is physically invalid. The strict RUC sea-ice wrapper
and classic option 91 remain refused. Component comparisons cover the real
surface caller and carried fields against host-compiled selected source; the
full strict physics gate remains closed, and NVIDIA-device parity is not
established.

These strict dynamics profiles support `h_sca_adv_order=2` or `5` with
`v_sca_adv_order=3` or `5` and `v_mom_adv_order=3` or `5`. In the selected
WOOF caller, the momentum order selects the vertical u and v stencils, while
the scalar order selects scalar and w vertical stencils. This horizontal-order
setting selects the geopotential stencil; moisture uses its own existing
transport dispatch. Comparisons cover mapped flat and terrain cases, both
moist transport modes, the fifth-order vertical limiter, rigid and open top
boundaries, external-mode and vertical-velocity damping, implicit upper
damping with either the WRF 4.6.1 or NOAA upper-wind form, the sixth-order
filter with the WRF 4.6.1
form, and checkpoint continuation. The unlimited filter is restricted to dry
runs; moist runs use its monotonic form. The selected WOOF dry
filter caller uses padded edge values in this form; its scalar caller uses
post-acoustic mass and bounded stencils. This preserves WOOF behavior and
does not establish the WRF dry caller's boundary behavior. The NOAA WRF 3.9
filter form is also admitted. In the selected WOOF caller, specified-boundary scalar rows use padded edges before the acoustic
loop; periodic scalar rows use post-acoustic mass. Dry rows use padded edges
in both cases.

Selected strict `diff_opt=2` configurations also support constant-coefficient
`km_opt=1` mixing with finite nonnegative horizontal and vertical diffusivities.
Source-stage and real tree checkpoint comparisons cover unequal coefficients,
active prescribed surface flux, mapped specified boundaries and Dudhia cadence.
These remain targeted caller comparisons, without NVIDIA-device or public
forecast parity claims.

Strict coordinate-surface mixing (`diff_opt=1`, `km_opt=4`) uses separate
software-RN32 coefficient and tendency kernels. The admitted subset requires
`isfflx=1` and zero prescribed diffusivities, with the existing geometry and
physics restrictions. Source-stage and checkpoint comparisons cover dry/moist
periodic and specified boundaries, including Dudhia cadence and subnormal
coefficients. Ordinary coordinate mixing retains its existing kernels.
With Dudhia radiation and either the existing Noah or no-surface profile,
specified-boundary moist terrain also admits `km_opt=4`, `diff_opt=1`,
`hybrid_opt=2` and `hypsometric_opt=2`, with sixth-order diffusion off and
advection orders 2/3/3. A mapped, nonuniform-terrain source comparison covers
surface coupling, boundary crossover and tree-format checkpoint continuation
through direct dycore steps; it is not a tree-executor or public forecast proof.

The moist physics-off and Dudhia-only profiles also admit spatially varying
terrain with finite three-dimensional base fields when `km_opt=0`, or when
`km_opt` in 2/3/4 uses `diff_opt=2`, with uniformly 300 K base theta. Source comparisons
cover mapped periodic and specified boundaries, both moisture transport modes,
radiation cadence, surface flux with mixing and checkpoint continuation. Other
mixing and physics with varying terrain remain refused. These are host-compiled
selected-source comparisons, without NVIDIA or public forecast claims.

The same restricted dynamics profile admits `diff_opt=2` with prognostic TKE
(`km_opt=2`) or 3-D Smagorinsky mixing (`km_opt=3`). Coordinate mixing
(`diff_opt=1`) also admits `km_opt=2` with `isfflx=1` and zero prescribed
constant diffusivities. Mapped source comparisons cover anisotropic and
isotropic lengths, surface forcing, moisture, radiation cadence and checkpoint
continuation. The optional TKE budget uses a separate strict binary64
reduction, with its diagnostic window rebuilt after restart. These comparisons
use host-compiled selected WOOF CUDA source, not an NVIDIA device. The shared
WOOF TKE transport pre-fold and mass-carrier choices differ from WRF's TKE
caller, so this scope does not claim WRF caller equivalence.

Strict NSSL has separate software-arithmetic callers for precipitation
reduction, scatter, hot gather, supersaturation, fused-GS preparation, effective
radii and radar diagnostics. Selected-source comparisons cover operation order,
CCN selection, subnormals and both radius concentration conventions, including
their distinct constants and RN32 micron conversion. Radar preserves the mixed
RN32/RN64 expression and pinned logarithm and power helpers; supersaturation
retains raw vapor and the source temperature table. Ordinary paths retain their
existing kernels. First-step and cumulus gather still require an unported
binary64 power contract and refuse before writing. Separate serial cloud, rain, ice and
snow sedimentation components cover up to 256 levels with source-ordered
transport and adaptive substeps. They match the standalone CUDA serial entry
points, including shallow serial calls; the separate shallow parallel
production route remains unproved. The strict NUCOND caller carries optional
frozen concentration fields and preserves source-ordered adaptive condensation,
CCN, evaporation and frozen-category updates. Actual caller comparisons cover
both number-concentration layouts, optional frozen state and CCN selection
against host-compiled selected source. Strict temperature publication uses
gradual binary32 multiplication; its current arithmetic evidence is host FP32
multiplication rather than a captured CuPy device body. Complete strict NSSL
physics remains
unadmitted. Its unported composed runtime, gather/sedimentation and full fused-GS
entry points refuse strict mode before preparation or kernel submission; the
independently matched component callers remain available.

Standalone strict regional WSM6 admits a flat periodic mixed-phase profile
with no other physics, positive-definite moisture
transport, 5–80 levels and
a timestep of at most 120 seconds. The 64- and 80-level kernel tiers support
the upstream graupel and hail coefficient sets. The conservative surface
correction is incompatible with upstream WRF-exact mode on both backends.
Source-ordered binary32 and pinned transcendental operations cover
preparation, the column and heating finish. Selected column comparisons at
64, 65 and 80 levels match 25,332 host-compiled CUDA-source words with
both coefficient sets. The active graupel and 80-level hail forecasts
match 15,732 and 136,598 selected full-step source words. Separate public
verification processes match all 21,264 and 156,778 final checkpoint
values and semantic headers, including carried frozen species, effective
radii and pending precipitation. Strict radar remains outside the profile.

Radiation-free strict WSM6 also accepts flat unmapped specified domains
with the existing clock-bound lateral forcing and positive-definite
transport. In two active steps with supplied vapour, both the supplied
and flow-dependent mixed-phase boundary cases match 25,574 selected
host-compiled CUDA-source output words and 34,654 public fresh
tree-restart values. Time-varying supplied vapour, cloud, rain, ice,
snow and graupel each change 55 west-edge words against the control.

Strict WSM6 also supports paired classic RRTMG in the same flat periodic
profile with the upstream diagnostic or supplied effective radii.
Diagnostic radii use cloud optics 2/3/1. The mixed-cloud public cases
with graupel and hail coefficients each save
immediately after the second radiation call, then restore a held step and
reach the next call. Fresh continuation matches 62,718 checkpoint values
and semantic headers in each case. At each of three
alarms, the sampled frozen cloud matches 6,160 selected CUDA-source
longwave optical-depth words, 272 upstream NumPy longwave solver words,
and 500 selected CUDA-source shortwave solver words. The default
`use_mp_re=1` retains supplied radius words in microns and uses 5/5/1
optics with separate snow and ice paths. The public case exercises those
paths at all three alarms and matches the same saved-value count after
restart. In the selected graupel columns, the previous conversion through
meters changed 12 radius words at the second alarm. The checkpoint binds
the radius policy.

Strict precipitation accumulation uses the same selected-source binary64
expression and preserves both windows, reset timing, history fields and carried
checkpoint planes. Standalone strict WSM6 runs both windows with active
rain, snow and graupel. Across five steps and three window resets, the
Metal result matches 42,570 selected host-compiled CUDA-source words.
Fresh public processes starting from a partially accumulated window
match 21,912 checkpoint values, semantic headers and 3,240 trace words.
Paired classic radiation also runs with both WSM6 windows. A checkpoint
after the second radiation call and inside both windows restores held
heating, the next radiation call and both independent resets. The public
case matches 63,366 checkpoint values and semantic headers and compares
3,888 active bucket words with the selected CUDA source.
With the hail coefficient set, fresh processes resume from immediately
before and after a step that calls radiation and resets the first bucket.
Both continuations match 63,366 saved values, including both radiation
heating rates and surface fluxes. The ten-step case matches 6,480 selected
CUDA-source bucket words.

Strict Kessler uses the same two windows. Its missing frozen precipitation
inputs follow the shared zero path. Standalone and paired classic radiation
cases each match 3,888 selected CUDA-source bucket words and, respectively,
48,462 and 55,266 saved values after a fresh restart. In the paired case the
save follows a coincident radiation call and bucket reset.

Full WRF-exact mode still needs complete chained dynamics, transport, boundary
and physics caller coverage, with the upstream operation order, precision,
rounding, subnormal handling and pinned transcendental functions. An exact-named
library or an isolated soft-arithmetic kernel does not satisfy that contract.

## Output diagnostics

`prec_acc_dt` and `prec_acc_dt1` interval precipitation buckets run on Metal
with the CUDA expressions' binary64 sums rounded once to binary32; reset follows
the step-start boundary and the buckets are part of the checkpoint inventory.

`lightning_option` 1, 2, 3 and 11 run on Metal from a mechanical translation of
`lightning.cu` (`tools/lightning_metal/generate.py`): the patch reduction stays
a single thread, the extended-level fields are staged on the host, and the
no-CuPy physics initializer now creates the lightning scratch and clock.

`gsd_diagnostics=1` runs on Metal from a mechanical translation of the GSD and
helicity units (`tools/gsd_metal/generate.py`): the Thompson hail size routine
uses soft binary64 with glibc `pow` and `exp`, the domain lightning maxima are
read back as one scalar each, and the admission refusal is removed.

Constant-K diffusion (`km_opt=1`, `khdif`/`kvdif` > 0) runs on Metal with open or specified lateral
boundaries and on a non-flat base: the open-ring surface-flux restore uses a new `add_open_ring`
kernel and full-height scratch planes, and the two Metal-only refusals are removed.

`bl_mynn_tkeadvect=True` runs on Metal for MYNN 4.6.1 and GSD: the transported `qke_adv` carrier replaces QKE on each
warm call and takes the predicted QKE back, as in the CUDA caller.

Grell-Freitas on Metal runs up to 256 levels (a second build of the same kernel at a 256-level stride serves 128 < nz <= 256)
and reads the lower-face W the CUDA caller passes instead of the two-face mean.

RRTMGP (`ra_lw_physics`/`ra_sw_physics` 4) runs on Metal with `icloud=2`: the binary cloud fraction is a new
`rrtmgp_cal_cldfra2` kernel, and the Metal admission and caller gates accept icloud 1 or 2 as CUDA does.

`gwd_opt=3` with `spp_pbl=1` runs on Metal inside an ensemble member: the GSL drag kernel reads the pattern's lowest
level as the CUDA kernel does, perturbing VAR, VARSS and the two variance caps.
The GSL path uses the pinned scalar math and retains gradual underflow in form
drag and tendency rotation. Literal-word comparisons cover partially and fully
active large-scale drag; the coupled restart check is separate evidence.

External lateral boundaries with the rational time law (the moist-theta boundary tables WRF inputs carry) run on Metal:
a new `evaluate_rational_boundary` kernel evaluates the binary64 law with soft arithmetic, word for word with the CUDA kernel.

The GSD 4.1 MYNN Metal caller supports mixing lengths 1 and 2, both source
TKE conversions, and both cloud tendency forms. Cold initialization, carried
fields, closure and clipping follow the selected generation, including gradual
underflow. Its current literal-word evidence compares the full driver and
coupled caller with retained CUDA source compiled for host execution; it does
not establish CUDA-device or full WRF-exact forecast parity. The shared GSD
SPP and TKE-budget restrictions remain.

WSM6 and WDM6 run the upstream conservative surface-return option through the
same typed column dispatch as their ordinary paths, including hail and deeper
column tiers. Their generated columns now follow the current CUDA source's
sedimentation integration, constants and operation order; this intentionally
corrects the previous handwritten Metal arithmetic. The WSM6 preparation and
finish stages remain shared, WDM6 retains its status accumulator, and full
WRF-exact mode remains refused. Source-word comparisons cover ordinary and
conservative returns separately and verify that the option changes both
backends by the same words.

Nested 3-D IAU carries the parent's pending spread through the same device
SINT difference as one-shot analysis, with geometry owned by the child's
manifest-backed state. Parent-plus-spread addition and the hydrostatic column
solve run on Metal. Child positivity, hydrostatic operand preparation, IAU
rate construction and receipt reductions remain host computations. The child
retains its own state and clock. The upstream multi-slot/4-D nesting restriction
remains.

Post-PBL `scalar_pblmix` uses the existing Metal column solver with the
registry-selected heat exchange field, scalar package and mass coupling. The
MYNN wrapper retains ownership of its internally mixed package. Per-call work
and raw rates rebuild after restart; the shared `bldt=0` requirement remains.
Actual caller comparisons cover the registered producer and species mappings,
including poisoned work buffers and retained WRF column fixtures. A short,
separately prepared NSSL forecast exercises nonzero QNN diffusion on both steps;
its resumed history and checkpoint arrays match the uninterrupted run. This
establishes that caller and continuation case, not full WRF forecast parity.

The offline-child warm-start route runs on Metal. Checkpoint overlap, land
continuation and parent precipitation strips use explicit host staging;
diagnostics and RK seed copies use the existing Metal paths. Focused checks
cover shifted overlaps, nonzero strip and ring replacement, and state restart.
A short public shifted-child case uses real parent history and a real donor
checkpoint; its zero precipitation strip does not establish wet-strip behavior.

The 1.1.3 CFL diagnostic includes the upstream 32-bin vertical histogram
on Metal, with the shared quantile reports and per-member ownership. The
whole-domain fold remains supported; tiled CFL reduction remains CUDA-only.

The CFL report retains upstream physical ring-slot ordering after its
32,768-step ring wraps; this reporting restriction is shared with CUDA.

Strict slow buoyancy, pressure gradient and moist-CQ preparation use integer
binary32 operations, independent diagnostic and big-step controls, and the
source's pressure carriers. Combined caller comparisons match retained CUDA
source compiled for host execution. Strict pressure diagnostics also implement
both hypsometric options, windowed output and the distinct carried-MUTS path,
including pinned glibc log/pow and perturbation pressure. These component checks
do not admit a full exact forecast; remaining stage contracts still apply.

Hydrometeor handover numbers use the existing host preparation closure,
including direct Metal initialization of analyzed moments. On
Metal, a specified boundary's record-end closure stages only its boundary
strips to the host for that same closure, then copies updated numbers back;
this closure is not GPU-native. Mixed-scheme nest number carry runs on Metal
with the existing coupling arithmetic. Strict WRF and exact-mode behavior
remain governed by the upstream handover switch.

Strict classic RRTM GASABS band 3 uses the selected source interpolation
and binary32 operations. Direct strict calls read back finite coefficients and
require nonnegative water and CO2 with a nonzero combined carrier; invalid inputs are refused before
output mutation.

Strict classic RRTM GASABS bands 4–5 use source-ordered self-continuum
arithmetic and the source band 4 split interval. Direct strict calls read back
finite coefficients and trace gases, and require nonnegative carriers with
nonzero water–CO2 and ozone–CO2 combinations before output mutation.

Strict classic RRTM GASABS band 6 uses source-ordered binary32 arithmetic,
including CFC11 and CFC12 in both atmospheric branches. Direct strict calls
read back finite coefficients and trace gases before output mutation.

Strict classic RRTM GASABS band 7 uses source-ordered self-continuum
arithmetic. Direct strict calls read back finite coefficients and require
nonnegative water and ozone with a nonzero combination before output mutation.

Strict classic RRTM GASABS band 8 uses source-ordered binary32 arithmetic
and its layer switch, including CFC12 and CFC22 in both branches. Direct
strict calls read back finite coefficients and trace gases before output
mutation.

Strict classic RRTM GASABS band 9 follows the source split intervals and
retains the most recently fired layer event, with the switch winning ties.
Direct strict calls read back finite coefficients and require nonnegative,
nonzero combined water and methane carriers.

Strict classic RRTM GASABS bands 10–11 use the source interpolation and
self-continuum grouping. Direct strict calls read back finite coefficients
and validate output shape before writing.

Strict classic RRTM GASABS bands 12–16 use source-ordered gas mixtures
and continuum arithmetic. Direct strict calls read back finite, nonnegative
carriers and reject a zero or nonfinite rounded mixture before output writes.

The strict classic RRTM GASABS helper composes all 16 bands through their
source-ordered implementations.

Classic Metal RRTM packs Exner from model-grid storage into bottom-up
columns before dividing heating by it, matching the upstream caller. This
corrects heating where Exner varies across columns or levels; radiative
fluxes, cloud fraction, and the ordinary transfer arithmetic are unchanged.
The ordinary downward transfer initializes its top boundary to zero, so
model-top heating does not read uninitialized radiance when no buffer layers
are added.

Strict classic RRTM Pade indices use source-ordered binary32 operations and
the source-rounded default coefficient. Nonfinite inputs and positive
optical-depth arithmetic that overflows are refused. The strict transfer
uses source-ordered binary32 operations, including the clear-sky stream;
its comparison covers cloud transitions, the 340 K Planck row and subnormals.

Strict Goddard limits each band dispatch to 32 columns. Its software-float
band kernels support at most 128 model levels. Above 49 levels, longwave
bands execute in source order across separate launches. Deeper strict columns
are refused before radiation preparation. Ordinary Goddard retains its
existing depth and chunk policy.
