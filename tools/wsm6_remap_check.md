# WSM6 sedimentation conservation checks

The regular-grid remap integrates each limited linear arrival-cell profile
over its geometric intersection with the target cell. Dividing by the full
target width counts uncovered space as vacuum. Surface precipitation integrates
the same profile below zero. These two integrals partition the initial mass,
including partially fallen cells and a populated uppermost layer.

`wsm6_remap_check.py` compiles the production Metal helper bodies as C++ with
binary32 inputs, contraction disabled and fast math disabled. It compares the
complete arrival calculation and remap with an independent binary64 two-point
Gauss integral. This checks remap geometry and integration; its limited slope
definition is shared with the scheme, so it cannot independently establish that
the slope limiter itself is correct. This CPU path is a source check, not CUDA
hardware validation.

The deterministic remap sweep has 576 columns, 2–64 layers, concentration scales
from 1e-12 to 1e-2, variable density and layer width, and fall distances next to
cell interfaces. It also exercises 152 complete columns from primitive fields,
both hail modes, populated and empty upper layers, and a four-column neighboring
rain reproducer. No downstream sedimentation intermediate is supplied to the
complete-column calculation. A copied source with doubled retained mass must
fail the conservation gate; production source is never mutated.

Before comparison, remap integrated error and conservation budgets are set to
64 binary32 epsilons times initial mass. Complete-column water budgets are
32 epsilons times initial mass plus 8 epsilons times 0.1 kg/m², allowing the
scheme's depletion/clipping floor. Native/source output differences use
128 epsilons times a field scale, with a 1e-8 absolute floor. Radii are checked
for finiteness but excluded from pointwise parity because depleted condensate
can cross their discrete lower bounds. Water, positivity and independent
integration remain separate gates. None of these checks proves radiative-flux
accuracy, forecast skill or CUDA parity.

Run CPU checks with a task-owned output directory:

```sh
GPUWM_NO_LOCAL_GPU=1 python tools/wsm6_remap_check.py --output /tmp/wsm6-remap-check
```

Add `--native` only after verifying all forecast and other GPU processes are
idle. The checker refuses visible models, an existing machine lock or queued
jobs. It uses the existing machine PID lock and runtime slot coordination,
synchronous packaged-kernel submissions and a 60-second parent deadline. It
checks identical transferred input bits, complete-column outputs and water
balance, followed by a two-step public `run_steps` WSM6 driver smoke with wind
and theta ranges. A blocked native pass is recorded explicitly; it is not a
pass. The driver smoke does not replace a public forecast CLI check.

The output records the source revision, shader hash, compiler settings,
coverage and budget ratios. A first remap failure preserves primitive inputs,
arrival interfaces, remapped concentrations and integral results in an NPZ.
The regression tests live in `tests/test_metal_wsm6_remap.py`.

The conservative boundary result intentionally differs from the legacy WRF
PLM search/remap when the arrival column leaves a partially covered top cell.
Agreement with that nonconservative result is not a valid mass oracle. The
overlap integral is exact for a linear profile in real arithmetic; finite
precision differences are assessed with the stated budgets.
