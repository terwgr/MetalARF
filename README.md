# MetalARF

MetalARF is a port of [Recast WOOF](https://github.com/recastsystems/woof)
1.0.3 to Apple Silicon. WOOF is a GPU weather model with WRF-class physics,
preprocessing, data assimilation and plotting. Upstream it runs on NVIDIA
GPUs through CUDA; MetalARF adds a Metal backend so the same models run on
the GPU of an Apple Silicon Mac, without CUDA or CuPy.

The package contains three models:

| Model | Command | Status | Metal |
| --- | --- | --- | --- |
| Regional (WRF-ARW class, nests, DA) | `woof` | Production | Every admitted physics combination |
| Hex (MPAS-style unstructured mesh) | `woof hex` | Preview | Forecast, restart, history |
| Global (spectral, with ensemble DA) | `woof global` | Preview | `sl_si`, `ssprk3`, restart, DA cycle |

On Metal, the regional model runs the operational HRRR configuration
(Thompson, MYNN GSD 4.1, RUC and legacy RRTMG), as well as nests, ensembles,
restarts, history products and local data assimilation. Apple GPUs have no
hardware double precision, so the physics that WRF computes in double uses a
software implementation, and results match the NumPy reference. The [backend guide](docs/public/BACKENDS.md)
lists coverage by feature, and the [changelog](CHANGELOG.md) lists changes.

Production means the regional commands and their config and checkpoint
formats are covered by the upstream major-version promise. It is not a
statement about forecast accuracy; see Science and limits below.

## Requirements

- Apple Silicon Mac (M1 or newer), macOS 14 or newer.
- Python 3.11 or newer, and a Rust toolchain (`cargo`) to build the native
  data bridges.
- Unified memory sets the largest domain you can run; WOOF checks this
  before it starts (see Memory below).

## Install

```bash
git clone https://github.com/terwgr/MetalARF.git
cd MetalARF
python3 -m venv .venv
source .venv/bin/activate
python tools/build_darwin_native.py --all --output-root build/native
python tools/stage_wheel_bridges.py --from-native-dir build/native
python -m pip install -e .
woof setup
woof fetch-tables
woof fetch-geog --datasets wrf
woof selftest
```

The Metal shader library ships prebuilt in `woof/libexec/metal`. After you
change a shader, rebuild it with
`python tools/build_metal_backend.py --output woof/libexec/metal`.

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

Compare two runs, any product, as run A minus run B:

```bash
woof render --diff runA/ runB/ --out diff/
```

The pair must share the grid and the valid times; a mismatch is refused by
name.

The Hex and Global models take the same backend choice:
`woof hex forecast --backend metal` and `woof global run CONFIG.toml --metal`.
`woof hex --help` and `woof global --help` list their options.

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
theta or temperature range, and finishes in seconds. `--backend auto` (the
default) uses Metal when a Metal device opens and NumPy otherwise; `--backend
metal` or `--backend numpy` forces one. The regional engine needs a GPU
backend, so it is skipped on NumPy. `--json` prints the results as one
document. The command exits nonzero when any check fails, so contributors can
run it before and after a change to the Metal path.

## Documentation

- `docs/public/`: the CLI manual, physics, hardware guidance and data sources.
  The [code verification record](docs/public/VERIFICATION.md) documents
  historical WRF v4.6.1 comparisons and their configuration and coverage
  limits.
- `components/hex/` and `components/globe/`: the hex and global models' manuals,
  tests and tools.
- `woof --help-all` lists every command; `woof hex --help` and
  `woof global --help` list the preview models' commands.

## Science and limits

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

The hex model is a port of the MPAS-A v8.4.1 dynamical core with this engine's
physics; it is not MPAS and is not endorsed by UCAR. Use official
meteorological services for forecasts and warnings.

WOOF continues an engine released before under another name: WOOF 1.0 is
that engine's release 2.8.0 and the fixes made since, under the WOOF name,
with the hex and global models inside it. Settings spelled `GPUWM_*` keep
working beside the new `WOOF_*` spelling.

## Credits and licence

MetalARF is a port of Recast WOOF by Recast Systems; the models, physics
and tools are theirs, and the Metal backend is added here. It is not an
official Recast release. Apache License 2.0. Third-party code, tables and datasets keep their own
terms; see [NOTICE](NOTICE) and [licenses/](licenses/).
