# Backends

WOOF has NVIDIA/CUDA and Apple Silicon/Metal forecast routes, plus NumPy-backed
CPU paths for preparation, analysis tools, and scientific reference work. The
available forecast route depends on the model and feature; NumPy is not a
complete fallback for every forecast.

| Model | Metal | CUDA | NumPy |
|---|---|---|---|
| Regional | Supported combinations through `woof go` | Production route | Preparation and selected tools; no general forecast fallback |
| Hex | Preview forecast path | Forecast route | Initialization and reference tools |
| Global | Preview forecast path | Forecast route | Preparation and reference tools |

Metal requires macOS 14 or newer on Apple Silicon (`arm64`). Install the wheel
and stage its native tools and tables:

```sh
python -m pip install recast-woof
woof setup
```

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
