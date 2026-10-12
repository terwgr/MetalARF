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

Choose the backend on the public forecast command:

```sh
woof go CONFIG.toml --backend metal
woof hex forecast ... --backend metal
woof global run CONFIG.toml --metal
```

The legacy `woof run` command does not expose a Metal selector. Regional Metal
coverage follows the capability table and admitted scheme combinations; Hex
and Global are preview models, and their Metal coverage is limited to the
implemented forecast paths. CUDA installation uses the matching `all-cu12` or
`all-cu13` extra on Linux; install one CUDA/CuPy major per environment.

Check an install with `woof selftest` (one tiny regional, Hex and Global
step on the chosen backend).

## Regional Metal admission and evidence

The selectors below come from `METAL_FAMILY_CAPS` in
[`woof/core/physics.py`](../../woof/core/physics.py). They describe caller
availability at this source revision. They are not a list of numerically
verified combinations; config validation and coupling restrictions still
apply.

| Family | Selectors listed by Metal admission |
|---|---|
| Microphysics (`mp_physics`) | 0, 1, 6, 8, 9, 10, 16, 18, 28, 50 |
| PBL (`bl_pbl_physics`) | 0, 1, 2, 5, 9, 11, 900 |
| MYNN source form (`bl_mynn_version`) | `wrf_461`, `gsd_41` |
| Surface layer (`sf_sfclay_physics`) | 0, 1, 2, 5, 91 |
| Land surface (`sf_surface_physics`) | 0, 2, 3, 4 |
| Cumulus (`cu_physics`) | 0, 1, 3, 16 |
| Radiation (LW, SW) | (0, 0), (0, 1), (1, 1), (4, 0), (4, 4) |
| Urban (`sf_urban_physics`) | 0, 1, 2, 3 |

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
| Radiation / cumulus adapters | Must match the configured selectors; no radiation adapter when radiation is disabled. |
| Noah-MP | Admission checks that its required SFLX and glacier callers exist. |
| Dynamics | Further checks cover base-state layout, boundary forcing, diffusion, damping and diagnostics in `dycore.step`. |

The runtime checks in [physics](../../woof/core/physics.py) and
[dynamics](../../woof/core/dycore.py) are authoritative for the exact selected
options. A selector appearing in this table does not mean every coupling
or public command can execute it.

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
