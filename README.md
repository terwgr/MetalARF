# MetalARF

MetalARF is a port of [Recast WOOF](https://github.com/recastsystems/woof)
1.1.5 to Apple Silicon. This integration is under development. WOOF is a GPU weather model with WRF-class physics,
preprocessing, data assimilation and plotting. Upstream it runs on NVIDIA
GPUs through CUDA; MetalARF adds a Metal backend so the same models run on
the GPU of an Apple Silicon Mac, without CUDA or CuPy.

The integration carries the upstream solar, YSU, Noah-MP snow-age and urban
water-budget generations. Numerical acceptance remains configuration-specific;
see [Metal feature contracts](METAL_FEATURES.md) for implementation coverage,
verification limits and remaining refusals.

The package contains three models:

| Model | Command | Status | Metal |
| --- | --- | --- | --- |
| Regional (WRF-ARW class, nests, DA) | `woof` | Upstream interface promise | Selected combinations; see backend guide |
| Hex (MPAS-style unstructured mesh) | `woof hex` | Preview | Forecast with the ported WOOF physics providers, global and limited-area, restart, history |
| Global (spectral, with ensemble DA) | `woof global` | Preview | `sl_si`, `ssprk3`, restart, DA cycle |

The Metal backend runs most of the regional physics families (microphysics
from Kessler to TEMPO, FAST_SBM and aerosol Thompson; YSU, MYNN, MYJ and
sa-TKE-EDMF; Noah, Noah-MP and RUC with urban, lake, sea-ice and ocean
models; Grell-Freitas, Kain-Fritsch, Tiedtke, SAS and Grell 3D; RRTM,
Dudhia, legacy RRTMG, RTE+RRTMGP and Goddard radiation), plus nests, ensembles,
WRF restarts, history products and data assimilation. Availability depends
on the selected options and their coupling; the
[backend guide](docs/public/BACKENDS.md) summarises it and refused options
are named at admission.

Apple GPUs have no hardware double precision. Metal's double-precision
physics uses software binary64. Ported kernels are translations of the CUDA
source, compared word for word with that source compiled for the host in the
cases METAL_FEATURES.md lists. That is not a general claim that Metal output
matches a CUDA device, NumPy or WRF for every configuration. The
[Metal feature contracts](METAL_FEATURES.md) describe the scoped additions,
remaining restrictions and evidence limits; the [changelog](CHANGELOG.md)
lists changes.

The regional commands and their config and checkpoint formats inherit the
upstream major-version promise. This describes the interface, not Metal
forecast accuracy; see Science and limits below.

## Requirements

- Apple Silicon Mac (M1 or newer), macOS 14 or newer.
- Python 3.11 or newer, and Rust 1.94 or newer (`cargo`) to build the native
  data bridges.
- Unified memory sets the largest domain you can run; WOOF checks this
  before it starts (see Memory below).

## Install

```bash
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

The command is installed as both `metalarf` and `woof`; they are the same
program, and WOOF's messages and manuals say `woof`. `--depth 1` fetches only
the current release; drop it to get the full
history. The first native build downloads the bridges' Rust dependencies
from crates.io, at the versions pinned in each `Cargo.lock`.

The Metal shader library ships prebuilt in `woof/libexec/metal`. After you
change a shader, rebuild it with
`python tools/build_metal_backend.py --output woof/libexec/metal`.

The PyPI distribution `recast-woof` is upstream WOOF and does not provide
MetalARF's Metal backend. Use the clone install above on a Mac. MetalARF
currently retains upstream's distribution name and version; `pip show`
alone does not distinguish the fork. Keep the clone's commit recorded with
your run when identifying the installed source.

On Linux with an NVIDIA GPU, follow the upstream install:
`python -m pip install 'recast-woof[all-cu12]'` (or `all-cu13`).

## Run a forecast

```bash
woof domain --point 35.3,-97.5 --source hrrr --cycle latest --hours 3 \
     --root-dx 3 --point-extent-km 300 --out forecast.toml
woof go forecast.toml --dry-run --backend metal
woof go forecast.toml --backend metal
```

`--dry-run` reviews the route, sizes it for your GPU and names anything
missing; the second line fetches, prepares, runs and draws. Pictures are drawn
as each history frame lands, beside the running forecast.

Coupled hydro is opt-in. `woof go forecast.toml --hydro on` couples NextGen
hydro wherever a hydrofabric covers the domain; a forecast without it has no
hydro.

Compare two runs, any product, as run A minus run B:

```bash
woof render --diff runA/ runB/ --out diff/
```

The pair must share the grid and the valid times; a mismatch is refused by
name.

The Hex and Global models take the same backend choice:
`woof hex forecast --backend metal` and `woof global run CONFIG.toml --metal`.
`woof hex --help` and `woof global --help` list their options.

`woof hex forecast --backend metal` runs the ported WOOF physics providers
(the WSM6, aerosol Thompson, MYNN/RUC/GSL and GSD 4.1 column suites, and a
generic WSM6/YSU/Noah-MP/RRTMG plan) with `--physics-mode correct` or
`exact`. The native MPAS-A v8.4.2 physics routes run on CUDA only and are
refused on Metal. It needs `--grid` and `--static` as the CUDA backend does,
and `--lbc-dir` on a limited-area mesh. `--checkpoint` and `--restart` carry
the physics state and its selected generations, and a restarted run
reproduces the uninterrupted one in the checked cases. Local time stepping
works on global and limited-area meshes as it does on CUDA.

## Memory

Before it allocates any forecast state, the Metal backend estimates how much
GPU memory the run needs and compares it with the working set macOS
recommends for the device. If the domain will not fit, WOOF stops and reports
both numbers. Use a smaller domain or a Mac with more memory.

| Variable | Effect |
| --- | --- |
| `WOOF_METAL_MEMORY_FRACTION` | Share of the recommended working set WOOF may use (default `0.65`). |
| `WOOF_METAL_RRTMG_CHUNK` | Columns per legacy RRTMG pass; unset sizes it from the device. |
| `WOOF_METAL_GPU_SLOTS` | Opt-in limit on concurrent WOOF Metal processes. |

## Self-test

`woof selftest` runs one tiny step of each model through its public path: a
regional physics row, the 12-cell Hex dry mesh and the shipped Global moist
smoke. It prints PASS, FAIL or SKIP for each, with the largest wind and the
theta or temperature range, and finishes in seconds. On Metal the Hex step
must also agree with the NumPy step. `--backend auto` (the
default) uses Metal when a Metal device opens and NumPy otherwise; `--backend
metal` or `--backend numpy` forces one. The regional engine needs a GPU
backend, so it is skipped on NumPy. `--json` prints the results as one
document. The command exits nonzero when any check fails, so contributors can
run it before and after a change to the Metal path.

## Documentation

Nine idealized cases support `woof verify CASE --backend metal`; see the
[backend guide](docs/public/BACKENDS.md#idealized-verification-cases) for case
names and durations.

- `docs/public/`: the CLI manual, physics, hardware guidance and data sources.
  The [code verification record](docs/public/VERIFICATION.md) documents
  historical WRF v4.6.1 comparisons and their configuration and coverage
  limits.
- `components/hex/` and `components/globe/`: the hex and global models' manuals,
  tests and tools.
- `woof --help-all` lists every command; `woof hex --help` and
  `woof global --help` list the preview models' commands.
- [AGENTS.md](AGENTS.md): how the port is laid out and the rules for
  changing it, for contributors and coding agents.
  [MAINTAINING.md](MAINTAINING.md) covers upstream merges and releases.

## Science and limits

Metal numerical coverage is incomplete. The software binary64 contracts
([base arithmetic](woof/metal/soft_binary64_contract.json) and
[exp helper](woof/metal/soft_gfk_exp_contract.json)) do not certify all
production callers. The combo smoke tool checks admission and finite state;
its success does not establish physical accuracy or parity. Historical
upstream verification results below apply to the builds and configurations
that produced them, and do not transfer automatically to this Metal port.

A reported [Vermont forecast comparison](https://github.com/terwgr/MetalARF/issues/7)
shows substantial Metal/CUDA divergence. The proposed
[longwave fix](https://github.com/terwgr/MetalARF/pull/8) has a small idealized
cloud comparison, but the original real-data comparison remains outstanding.

WOOF implements a WRF-ARW-class regional model and WRF-derived physics on the
GPU. Comparisons against WRF Fortran, ULP and byte-identity checks, and the
matched-run and ensemble consistency comparisons are **code verification**:
evidence that the tested code computes its intended equations. Each result
belongs to the build, reference version and configuration it tested. None
establishes general equivalence with WRF or skill for every combination of
physics, input data and hardware.

**Solution verification** estimates numerical error in a particular run.
Routine numerical-error budgets for weather forecasts remain a gap.
**Validation** compares model results with observations. Meteorology calls
that scoring forecast verification; in computational-science V&V terms it is
validation. Matching WRF is code verification and never validates a model.
WOOF can inherit WRF's published validation record only to the extent the
two are statistically indistinguishable for the configuration in question.

Published observation evidence is limited to specific runs and variables,
including the ASOS snow/soil comparison, the station/ceilometer fog-suite
comparison and an MRMS model-top comparison. The observation scoreboard is
in development. These do not establish a broad validation record across
regions, seasons or forecast lengths. The
[evidence page](docs/public/VERIFICATION.md) documents historical WRF v4.6.1
code comparisons and their configuration and coverage limits.

The hex model is a GPU port of MPAS-Atmosphere v8.4.2, with MPAS-A's own
physics suites as its default and this engine's physics selectable; it is not
MPAS and is not endorsed by UCAR. Use official
meteorological services for forecasts and warnings.

WOOF continues an engine released before under another name: WOOF 1.0 is
that engine's release 2.8.0 and the fixes made since, under the WOOF name,
with the hex model inside it. Settings spelled `GPUWM_*` keep
working beside the new `WOOF_*` spelling.

## Credits and licence

MetalARF is a port of Recast WOOF by Recast Systems; the models, physics
and tools are theirs, and the Metal backend is added here. It is not an
official Recast release. Apache License 2.0. Third-party code, tables and datasets keep their own
terms; see [NOTICE](NOTICE) and [licenses/](licenses/).
