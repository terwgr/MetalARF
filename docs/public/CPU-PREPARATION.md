# CPU preparation

WOOF can fetch and prepare forecast inputs without a forecast device. This
route is useful when preparing a case for another machine. CPU preparation
does not imply a NumPy forecast fallback. For Apple Silicon/Metal and CUDA
forecast options, see [Backends](BACKENDS.md).

## Prepare inputs

Use `woof domain` to create a configuration and print the fetch and prep
commands. If no sizing device is present, declare the capacity being targeted
with `--card` or `--vram-gib`; that estimate is not a measurement of the
forecast machine.

For an existing acquisition, `woof prep --preprocess-backend cpu` selects
the CPU preparation path. `--preprocess-backend auto` uses a certified CUDA
preprocessor when available and otherwise selects the CPU path; the receipt
records the choice. This option controls preparation, not forecast execution.

A prepared bundle carries the inputs and receipts needed by the forecast
command printed at the end of preparation. Move the complete bundle and its
bound configuration together to the target machine, then choose its forecast
backend as documented in [Backends](BACKENDS.md).

## Draw existing output

`woof render` can draw a history file produced elsewhere. Rendering does not
run or resume the forecast. See [output and products](OUTPUT-VARIABLES.md).
