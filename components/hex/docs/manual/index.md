# woof hex User Manual

woof hex runs a global, variable-resolution atmospheric model. Its CUDA route
has broad forecast coverage, and an implemented forecast path is available on
Apple Silicon/Metal. See the shared [backend guide](../../../../docs/public/BACKENDS.md);
examples and measurements in this manual describe CUDA unless noted.

woof hex is a research and educational tool. It is never a substitute for
official forecasts and warnings from your national meteorological service. Do
not use it to make safety decisions.

*(MPAS-Atmosphere is a registered project of NCAR and LANL. woof hex is an
independent port and is not affiliated with or endorsed by them.)*

## Chapters

1. [What woof hex is](01-what-this-is.md), plain-language introduction: what
   it does, what it cannot do, and what hardware it needs.
2. [Quickstart](02-quickstart.md): install, `woof hex doctor`, closing every
   gap doctor names, obtaining assets, first initial conditions, first
   forecast, first rendered products.
3. [Concepts](03-concepts.md): mesh, static, init; the doors; why the
   dynamical core is pinned and the physics is judged by observations; what a
   refusal is and why every refusal names its remedy.
4. [Meshes and static fields](04-meshes-and-static.md): validate a mesh pair,
   generate a new mesh and build its static, capacity planning with the
   measured footprint model.
5. [Initial conditions](05-initial-conditions.md), the init door: the
   native-free vertical path, the capsule compatibility mode, and every
   physics switch with its native namelist key.
6. [Running a forecast](06-forecast.md): what the run verifies at launch,
   registered meshes and timestep admission, local time stepping and its
   measured cost, restart, and what a run receipt claims and does not claim.
7. [Rendering products](07-render.md): history files to product PNGs through
   the Rust renderer, the delivered layout, and the render manifest.
8. [Troubleshooting](08-troubleshooting.md): the most common named refusals,
   what each means, and the remedy for each.
9. [Reference](09-reference.md): doors and flags, environment variables, the
   mesh registry, the test battery, and where receipts live.

## Conventions

Measured claims name their hardware and configuration. Declared differences
from native MPAS-Atmosphere are described in chapter 3 and quantified in
[the divergence record](../declared-divergences.md). The package and command are
both `woof hex`; the import namespace is `woof.hex`. The engine supplies the
shared physics seam and native tools.
