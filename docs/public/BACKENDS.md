# Backends

WOOF has NVIDIA/CUDA and Apple Silicon/Metal forecast routes, plus NumPy-backed
CPU paths for preparation, analysis tools, and scientific reference work. The
available forecast route depends on the model and feature; NumPy is not a
complete fallback for every forecast.

| Model | Metal | CUDA | NumPy |
|---|---|---|---|
| Regional | Caller paths through `woof go`; limits below | Production route | Preparation and selected tools; no general forecast fallback |
| Hex | Preview forecast path | Forecast route | Initialization and reference tools |
| Global | Preview forecast path | Forecast route | Preparation and reference tools |

Metal requires macOS 14 or newer on Apple Silicon (`arm64`), Python 3.11
or newer, and Cargo for the native data bridges. Install MetalARF from its
repository, using the same route as the [README](../../README.md#install):

```sh
git clone --depth 1 https://github.com/terwgr/MetalARF.git
cd MetalARF
python3 -m venv .venv
source .venv/bin/activate
python tools/build_darwin_native.py --all --output-root build/native
python tools/stage_wheel_bridges.py --from-native-dir build/native
python -m pip install -e .
woof setup
woof fetch-tables
woof fetch-geog --datasets wrf
woof selftest --backend metal
```

The Metal shader bundle ships with this clone; Xcode is needed to rebuild
shaders after changing them, not to use the prebuilt bundle. The PyPI package
`recast-woof` installs upstream WOOF, which has no Metal backend. MetalARF
currently keeps that distribution name and upstream version, so package
metadata alone does not identify the fork. Record the clone's commit
(`git rev-parse HEAD`) with the run; an editable install points to that clone.
The imported module and editable-install metadata must both identify the
intended clone. To inspect the module origin, run:

```sh
python -c 'from pathlib import Path; import woof.metal; print(Path(woof.metal.__file__).resolve())'
```

Choose the backend on the public forecast command:

```sh
woof go CONFIG.toml --backend metal
woof hex forecast ... --backend metal
woof global run CONFIG.toml --metal
```

Other commands select Metal the same way:

- `woof run --backend metal` runs `--wrfinput` / `--met-em` inputs and
  `[case_data]` configs in process, nested trees included (`woof branch` and
  `woof resume` take the same flag). A `[case_data]` ensemble runs its
  members serially on one Metal context; each member decodes its own
  inputs, and stochastic members stay refused as on every Metal ensemble.
  Delayed, spawned and moving nests (scripted `[relocation]` moves,
  `[relocation.follow]` and per-domain `follow`) run as on CUDA, and the
  tree is checked against the Metal memory plan first, as `woof go` does.
- `woof sim --backend metal`; `woof downscale --backend metal` (the parent is
  interpolated on the CPU, a microphysics change converts through the Metal
  edge kernel, and the child is held resident: `[tiles]` steps it under the
  Metal tiling rule below, and `--auto-vram` is refused);
  `woof hex cycle run --backend metal` (the CUDA contract deck, a CUDA
  kernel self-check, is skipped and recorded as skipped); and `woof stream`
  with `[run] backend = "metal"` in its plan.
- `woof speedrun --backend metal` records the warm class only: the Metal
  pipeline cache belongs to the system and cannot be emptied for one run.
- `woof warm-kernels` on a Mac builds every Metal pipeline once (this can take
  minutes on a cold cache), so later forecasts skip the driver's first-use
  compile. `--profile`, `--all-profiles` and `--levels` select CUDA compile
  domains and are refused on Metal.
- `woof local-da --precip-analysis clear` clears precipitation where the
  radar observed no echo, on either backend.
- `woof local-da --radar-tten` (opt-in, either backend) adds HRRR's radar
  latent heating (`mp_tend_radar`): each forecast leg that starts from a
  radar analysis is heated from that window's radar file for one cadence,
  and so is the first cadence of the short forecast. It needs a radar route
  and microphysics, and is refused before anything is fetched otherwise;
  continuous windows (`--continuous`) refuse it for now. The forcing stays
  attached for the whole leg, so HRRR's microphysics clamp (`mp_tend_lim`,
  0.07 K/s) applies past the heated cadence too; its slots and builder volumes
  are counted in the plan's memory price. Each member
  writes `radar-tten.json` beside its outputs. The Metal kernels follow the
  CUDA source's arithmetic operation for operation.
- `tools/da_cycle_prepared.py --state-backend metal` runs the research
  radar-DA cycle on a Mac: perturbed members, restart sets between legs,
  the LETKF solve (`--solve-device metal` by default there, or `host`),
  hot start, `--radar-tten`, `--hydrometeors`, reflectivity and clear-air
  analysis, `--precip-analysis clear` and `--save-ensemble` /
  `--resume-ensemble`. The perturbation, the increments and the hot start
  are computed on host copies of the state and written back. The nest
  options stay CUDA only and are refused.
- `woof cycle --parent-kind mpas-metal` cycles an MPAS-mesh parent on the
  Metal Hex engine (no `--port-root`): each leg runs in its own process and
  hands its state on through anchors. A null arm reproduces the
  uninterrupted run bit for bit.

Regional Metal
coverage follows the capability table and admitted scheme combinations, and
Metal tile streaming runs only `[tiles] mode = "on"` with a pinned tile
extent and `store = "device"` (no `auto` mode and no host store, so no
larger-than-memory domains); Hex
and Global are preview models, and their Metal coverage is limited to the
implemented forecast paths. Global's `--metal` is on `woof global run`, `go`,
`cycle` and `da cycle`, `da forecast` and `da fresh`. It runs configs with
`backend = "numpy"` and `precision = "float64"` only, without `arwen-native`
real-data physics; any other Global config, including the float32 ones, is
refused before the run writes anything. `da init`, `da analyze` and
`assimilate` integrate no model and run on the CPU. CUDA installation uses the matching `all-cu12` or
`all-cu13` extra on Linux; install one CUDA/CuPy major per environment.

Check an install with `woof selftest` (one tiny regional, Hex and Global
step on the chosen backend).

## Regional Metal admission and evidence

The selectors below come from `METAL_FAMILY_CAPS` in
[`woof/core/physics.py`](../../woof/core/physics.py). They describe caller
availability at this source revision. They are not a list of numerically
verified combinations; config validation and coupling restrictions still
apply. Admission establishes that a configuration passes the checked gates;
execution establishes that it runs. Numerical agreement is limited to the
particular inputs, fields and tolerances compared.

<!-- metal-capabilities:start -->
| Family | Selectors listed by Metal admission |
|---|---|
| Microphysics (`mp_physics`) | 0, 1, 6, 8, 9, 10, 16, 18, 28, 50 |
| PBL (`bl_pbl_physics`) | 0, 1, 2, 5, 9, 11, 900 |
| MYNN source form (`bl_mynn_version`) | `gsd_41`, `wrf_461` |
| Surface layer (`sf_sfclay_physics`) | 0, 1, 2, 5, 91 |
| Land surface (`sf_surface_physics`) | 0, 2, 3, 4 |
| Cumulus (`cu_physics`) | 0, 1, 3, 16 |
| Radiation (LW, SW) | (0, 0), (0, 1), (1, 1), (4, 0), (4, 4) |
| Urban (`sf_urban_physics`) | 0, 1, 2, 3 |
<!-- metal-capabilities:end -->

Important restrictions from `METAL_KNOWN_CONFLICTS` and the dynamics gate:

| Option or coupling | Metal requirement or limitation |
|---|---|
| WRF-exact mode | No Metal physics callers; dynamics also refuses WRF-exact arithmetic. |
| TKE budget | No implementation; use `tke_budget=0`. Initialization, direct stepping and admission inspection reject a nonzero budget. |
| RUC | Nine soil levels. |
| Lake model | Requires RUC (`sf_surface_physics=3`) on Metal; upstream's broader LSM allowance does not apply. |
| MYNN GSD 4.1 | Requires PBL 5, mixlength 2 and the registered option set listed in `METAL_KNOWN_CONFLICTS`. |
| MYJ / Eta surface layer | PBL 2 and surface-layer option 2 must be paired. |
| Grell–Freitas | Requires a PBL caller for KPBL and `cudt_minutes=0`. |
| Urban | Requires Noah-MP (`sf_surface_physics=4`) in Metal admission. |
| NSSL (`mp_physics=18`) | At most 128 levels on Metal (CUDA: 256); the 256-level sedimentation column does not fit the GPU stack. |
| IEVA (`zadvect_implicit`) | Both variants, up to 256 levels, as on CUDA. |
| MYNN | `bl_mynn_mixscalars=1` runs on Metal with `wrf_461`; with `gsd_41` it stays refused, as on CUDA. `spp_pbl` is refused: the Metal MYNN kernels support SPP, but no Metal route supplies a perturbation pattern yet. |
| Radiation / cumulus adapters | Must match the configured selectors; no radiation adapter when radiation is disabled. |
| Noah-MP | Admission checks that its required SFLX and glacier callers exist. |
| Dynamics | Further checks cover base-state layout, boundary forcing, diffusion, damping and diagnostics in `dycore.step`. |

The runtime checks in [physics](../../woof/core/physics.py) and
[dynamics](../../woof/core/dycore.py) are authoritative for the exact selected
options. A selector appearing in this table does not mean every coupling
or public command can execute it.

Metal smoke commands return status 0 for finite executed rows or expected
configuration refusals, status 1 for crashes, nonfinite rows or incomplete
inner deadlines, and status 124 for parent timeouts. Pairwise and suite
commands propagate child failures. An expected refusal means the
configuration did not execute.

`tools/metal_combo_smoke.py` labels finite executed rows `ok`; preflight
`ok` means initialization gates resolved without device execution. Neither
result measures numerical agreement. The [software binary64 contract](../../woof/metal/soft_binary64_contract.json)
reports zero mismatches in bounded base-arithmetic probes, but explicitly
excludes the later `soft64_fma` addition. It remains `implemented-unverified`.
The [exp-helper contract](../../woof/metal/soft_gfk_exp_contract.json) is a
dependency-blocked source proposal with `production_registration=false`,
although production shaders include the helper. These records do not
certify all production arithmetic. Historical upstream WRF comparisons do
not certify Metal. Published Metal release notes do not
provide a reproducible cross-model bitwise comparison or speed benchmark.

The [real-data divergence report](https://github.com/terwgr/MetalARF/issues/7)
remains open. [PR #8](https://github.com/terwgr/MetalARF/pull/8) addresses a
longwave band-table defect, but its idealized comparison does not replace a
rerun of the reported forecast.

## MetalARF 1.0.3 bounded validation

MP50 has exact FP32 NumPy agreement on the retained tiny and cold profiles
in `tools/metal_p3_parity_regression.py`. Prepared public P3 forecasts ran
for ten minutes in a clean installed environment; three repeats and three
checkpoint resumes produced identical final canonical states. This does not
certify every profile, long-duration stability or a P3 radar DA operator.

The release also exercised evolving prepared UW/Thompson, NSSL, Thompson,
Milbrandt–Yau and WDM6 forecasts, with three identical final states and finite
history fields per case. The radar regression uses the actual observation
operator, including repeated calls without forecast-state mutation; the
public synthetic DA analysis is a separate bounded check.

Morrison reflectivity requires positive finite number moments for active
hydrometeor mass. Boundary forcing can violate this requirement; the
reference deliberately produces NaN for an invalid active population.
One UW/Morrison boundary receipt did so and is retained as a restriction,
not a finite-output acceptance result. Use compatible mass/number forcing.

Radiation comparisons use the numerical reference for corrected paths;
cloudy outputs intentionally change after the band-index correction.
Issue #7's Vermont CUDA comparison remains outstanding. Admission, smoke
success and execution capsules alone do not certify numerical agreement.

## Idealized verification cases

Run a case with `woof verify CASE --backend metal`. Omitting `--backend`
retains the default CUDA route. Add `--outdir OUT` to write case plots and
NetCDF output; omitting it computes metrics only.

| Case | Duration |
|---|---|
| `straka` | 900 seconds |
| `igw` | 3,000 seconds |
| `hill2d` | 10 hours |
| `wk82` | 2 hours |
| `moist_bubble` | 1 hour |
| `cbl_dry` | 2 hours |
| `convective_boundary_layer` | 15 minutes |
| `cloud_topped_boundary_layer` | 15 minutes |
| `gabls1` | 9 hours |

These runners retain the case configurations, initial conditions and gate
bounds. Regional cases use the native Metal engine; GABLS1 uses its separate
single-column protocol with software binary64 prognostics. The two LES cases
cover their small-grid smoke configurations. Unsupported optional modes,
including `tke_budget=1`, remain subject to Metal admission restrictions.

WK82 preserves WOOF's configuration, metric definitions and thirteen gate
bounds on both backends. Its snapshot geometry measures the two strongest
updraft cores at 60, 90 and 120 minutes, classified by their projection onto
the right normal of the initial shear. These snapshots describe the changing
core population.

Separate `tracked_*` diagnostics follow fixed opposite-rotation identities.
Lineage compares advected prior rotation-supporting cells with complete
current updraft components. Ambiguous continuations retain all admissible
paths and conservative separation bounds. Lost, merged, reversed or capped
paths fail the tracked measurement, leaving subsequent tracked separation
unavailable. Weak confidence may persist between analysis times; required
analysis times need confident identities. A different surviving core never
replaces a lost identity. The mass-closure diagnostic independently observes
lateral acoustic fluxes.

The command returns zero when all case gates pass. Passing case gates does
not establish field-by-field equivalence with CUDA or WRF.

## Metal memory and environment

Before allocating forecast state, Metal estimates device use against the
recommended working-set size. The default budget is 65% of that size. If the
estimated forecast exceeds its budget, WOOF refuses before launch; reduce the
domain or use a device with more available memory.

| Variable | Effect |
|---|---|
| `WOOF_METAL_MEMORY_FRACTION` | Budget share of the recommended working set; default `0.65`, valid range `(0, 1]`. |
| `WOOF_METAL_RRTMG_CHUNK` | Positive integer columns per legacy RRTMG pass; unset lets WOOF size the chunk for the device. |
| `WOOF_METAL_GPU_SLOTS` | Opt-in shared process-lock slot count; set to `1` to serialize cooperating Metal runs. Unset disables the lock. |

See [hardware sizing](HARDWARE.md) for domain sizing and [the CLI manual](CLI-USER-MANUAL.md)
for general workflow help.
