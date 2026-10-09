# 8. Operational envelope

The supported backend, model routes, and device preflight vary by model and
feature. Use the current [backend guide](../public/BACKENDS.md) for selectors
and platform coverage.

## Capacity and admission

Before launch, `woof domain` and `woof check` help size a configuration;
`woof go CONFIG --dry-run` reviews its actual route. Launch repeats the
resource checks on the target device. CUDA and Metal use different device
budgets, described in [hardware sizing](../public/HARDWARE.md) and the
[Metal preflight guide](../public/BACKENDS.md).

Host memory is separate from forecast-device memory. Preparation can retain
and decode input windows, while forecast state and radiation workspaces use
the selected execution backend. Shortening a forecast does not always reduce
input storage; narrow the supplied window if host memory is the constraint.

Tiled execution can reduce resident forecast state, but does not remove the
limits on preparation, output, or input storage. See [tile streaming](../public/TILES.md).

## Reading performance claims

A timing or capacity measurement applies to its recorded build, hardware,
configuration, and input window. It does not imply capacity or forecast skill
for another backend or configuration. See [verification](../public/VERIFICATION.md)
for the scope of WRF comparisons and observations.
