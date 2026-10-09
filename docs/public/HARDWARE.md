# Hardware and memory sizing

For supported devices, backend selection, and Metal's device preflight, see
[Backends](BACKENDS.md). This page covers domain sizing, host memory, and
execution tradeoffs.

## Size before running

For an NVIDIA forecast, `woof domain` can size a starter case against a
named card tier or declared capacity:

```sh
woof domain --point 35.3,-97.5 --source gfs --cycle latest \
  --hours 6 --card 16gb --out configs/myarea.toml
```

Use `--vram-gib N` when you want to declare a capacity directly. `--nz N`
sets mass levels; the starter config and `woof check` report the resulting
memory estimate. `woof go CONFIG --dry-run` reviews the actual route before
execution, and launch checks the target device again.

Metal-specific memory admission and its controls are described in
[Backends](BACKENDS.md).

## Grid size and streaming

Higher horizontal or vertical resolution increases the state and workspace
cost. Keep the forecast duration and output interval appropriate to the
question being studied. A shorter forecast does not reduce the memory needed
to retain all supplied boundary times; use a narrower input window when that
is the limiting factor.

Prepared forecasts can choose tiled execution with `--tiles auto`, force it
with `--tiles on`, or disable it with `--tiles off`. Streaming reduces the
resident device grid; preparation, output, and host-side input storage still
have their own limits. See [tile streaming](TILES.md).

## Host memory

Preprocessing and forecast state use different memory budgets. CPU preparation
can consume substantial host RAM while decoding source files, building static
fields, and composing input grids. For the CPU preparation route, see
[CPU preparation](CPU-PREPARATION.md). For input formats and acquisition, see
[Data](DATA.md).

## Multiple devices

CUDA installations can run independent forecasts on separate devices through
[`woof multi-run`](MULTIGPU.md). This is separate from one forecast spanning
multiple devices. Metal concurrency controls are described in [Backends](BACKENDS.md).

## FP32, subnormals, and GPU-model caveats

The model state uses FP32, matching WRF's default `REAL`. Arithmetic and
subnormal handling can vary with the backend, device, compiler, and build, so
bit identity across those environments is not implied. See
[determinism](DETERMINISM.md) and [verification](VERIFICATION.md) for the
scope of numerical comparisons.
