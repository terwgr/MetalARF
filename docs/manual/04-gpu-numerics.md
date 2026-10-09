# 4. Numerical behavior across backends

WOOF stores model state in FP32, matching WRF's default `REAL`. CUDA and Metal
use separate implementations and compilation paths; byte identity between
backends is not implied. Supported routes and backend selection are listed in
[Backends](../public/BACKENDS.md).

## 4.1 Precision and compiler path

Floating-point results can depend on the backend, device architecture, compiler,
and math libraries. CUDA-specific observations do not establish how Metal
behaves. Use the backend-specific evidence for the route being evaluated; see
[determinism](../public/DETERMINISM.md) and
[physics evidence](../public/PHYSICS.md).

## 4.2 Reduction order

Parallel reductions may sum values in a different order and therefore produce
small rounding differences. Comparisons that need reproducible results should
use the same backend, build, inputs, and configuration in both runs.

## 4.3 Determinism and comparison

A deterministic replay claim is scoped to its recorded environment. A dual-run
byte comparison detects differences between its two outputs; equality does not
establish numerical correctness or forecast skill. The limits and conditions
are stated in the [determinism guide](../public/DETERMINISM.md).

## 4.4 Restart identity

A restart is valid only for a compatible configuration and execution contract.
Use the public [restart guide](../public/PREPARED-RESTART.md) and retain the
receipt with its checkpoint.
