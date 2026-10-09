# Level 3: Global spherical-harmonic research dynamical core

## Status

Level 3 is an additive, executable, research-only global model inside the
`woof.globe.spectral` package. It does **not** replace WOOF's regional
nonhydrostatic ARW core, does not alter an ordinary forecast, and is not called
from `woof run`. Its public door is:

```bash
python -m woof.globe.spectral --help
```

Every runnable TOML must carry the exact acknowledgement:

```toml
acknowledgement = "research-only-global-spectral-v1"
```

The current implementation supplies all of the following in one self-contained
CPU-reference/CuPy-optional subsystem:

- Gaussian latitude/longitude grids;
- triangular complex spherical-harmonic analysis and synthesis;
- analytic scalar gradients, Laplacian and inverse Laplacian;
- direct vector spherical-harmonic analysis through vorticity/divergence;
- vorticity/divergence wind inversion through streamfunction and velocity
  potential;
- a rotating vector-invariant shallow-water model;
- Williamson test case 2 at the verified `alpha_rad = 0` orientation;
- a multilayer dry hydrostatic primitive-equation model in sigma coordinates;
- SSPRK3 and classical RK4;
- exact exponential total-degree hyperdiffusion;
- a global-mean surface-pressure mass fixer;
- optional Held-Suarez thermal relaxation and lower-tropospheric drag;
- hash-bound checkpoints and self-hashed pass-or-failure run receipts;
- progressive quantized scalar and wind compression;
- log-space handling for positive scalar fields;
- direct arbitrary latitude/longitude sampling with a bounded chunk workspace;
- a hash-bound regular-lat/lon parent-export artifact.

## Transform definition

The stored basis is the positive-zonal-wavenumber triangular set

\[
Y_n^m(\phi,\lambda)
=
N_n^m P_n^m(\sin\phi)e^{im\lambda},
\qquad 0\le m\le n\le T,
\]

with complex orthonormal normalization and the Condon-Shortley phase. For real
fields only `m >= 0` is stored; negative `m` is reconstructed by conjugate
symmetry.

Latitude analysis uses Gauss-Legendre quadrature in
`x = sin(latitude)`. Longitude uses an equispaced complex FFT. The default
nonlinear grid is 3/2-dealiased relative to the triangular truncation.

The transform is normalized so that

\[
\int_{S^2} |f|^2\,d\Omega
=
\sum_n\left(|a_{n0}|^2+2\sum_{m=1}^n|a_{nm}|^2\right).
\]

`transform-check` gates both a coefficient round trip and Parseval closure.

## Vector transform

Independent scalar fits of eastward and northward wind are not a vector
spherical-harmonic transform. Level 3 instead computes vorticity and divergence
directly by integration by parts, then reconstructs wind from

\[
\nabla_h^2\psi=\zeta,
\qquad
\nabla_h^2\chi=D,
\]

\[
u=-\frac{1}{a}\frac{\partial\psi}{\partial\phi}
  +\frac{1}{a\cos\phi}\frac{\partial\chi}{\partial\lambda},
\]

\[
v=\frac{1}{a\cos\phi}\frac{\partial\psi}{\partial\lambda}
  +\frac{1}{a}\frac{\partial\chi}{\partial\phi}.
\]

The global vector degree-zero mode is absent by construction.

## Shallow-water model

The shallow-water state is relative vorticity, divergence, and geopotential.
Nonlinear terms are evaluated in grid space and transformed back into the
triangular spectral state. The model uses the vector-invariant momentum form
and conservative geopotential continuity:

\[
\frac{\partial \mathbf v}{\partial t}
=-(\zeta+f)\,\mathbf k\times\mathbf v
-\nabla_h\left(\Phi+\frac{|\mathbf v|^2}{2}\right),
\]

\[
\frac{\partial\Phi}{\partial t}
=-\nabla_h\cdot(\Phi\mathbf v).
\]

The shipped five-day gate is Williamson test case 2 at `alpha_rad = 0`.
Its admitted initial vorticity is formed independently from the analytic
solid-body result `zeta = 2*u0*sin(latitude)/a`, rather than being derived by
the vector transform whose dynamics the case is testing. The completion gate
checks geopotential, vorticity, vector wind, scaled divergence, mass, total
energy, and potential enstrophy. The analytic tilted-axis state exists as a
helper for future work, but a run config with nonzero alpha is refused: that
arm does not yet close the discrete Coriolis balance and therefore has no
verification receipt.

## Dry primitive-equation model

The hydrostatic primitive state is

- relative vorticity `zeta(n,m,k)`;
- divergence `D(n,m,k)`;
- temperature `T(n,m,k)`;
- log surface pressure `ln(ps)(n,m)`.

The vertical coordinate is pure sigma:

\[
p(\sigma)=\sigma p_s.
\]

Horizontal momentum uses the vector-invariant vorticity/divergence form.
Hydrostatic geopotential is integrated upward through piecewise-isothermal
sigma layers. The sigma-coordinate continuity equation provides
`d ln(ps)/dt` and half-level `sigma_dot` with exact zero top and bottom boundary
fluxes. Temperature includes horizontal advection, centered sigma-coordinate advection,
and dry adiabatic compression/expansion.

This is intentionally an explicit research core. It has no semi-implicit
Helmholtz solve, semi-Lagrangian transport, hybrid pressure coordinate,
moisture, or regional ARW acoustics. A strict spectral CFL check runs before
every state update.

## Exact exponential diffusion

For total spherical degree `n`, the per-step response is

\[
G_n=
\exp\left[-\frac{\Delta t}{\tau_T}
\left(
\frac{n(n+1)}{T(T+1)}
\right)^p\right].
\]

Low degrees through `preserve_degree` are exactly one. Divergence and log
surface pressure may use independent strengths while sharing the same
registered response family.

## Global field compression

### Scalar fields

A scalar field is transformed, optionally degree-truncated, and quantized with
an independent symmetric `int16` scale for each total degree and leading
field. Payloads carry:

- arithmetic pin hash;
- spectral-grid identity;
- source shape, dtype and SHA-256;
- retained degree;
- per-degree scales;
- post-quantization physical-space error metrics;
- a payload hash over metadata and every coefficient array.

Progressive decoding can request any degree no greater than the stored degree.

### Positive fields

Strictly nonnegative fields can select `space = "log"`:

\[
y=\log(\max(x,x_{\min})).
\]

Negative input is refused. Decoding exponentiates and can restore the original
arithmetic global mean through one multiplicative correction. This is useful
for positive continuous carriers; it is not a generic occurrence/no-occurrence
codec for zero-inflated precipitation.

### Wind

Wind is compressed as vorticity and divergence triangles, not as independent
scalar U/V fields. Decoding reconstructs the vector wind through the same
streamfunction/velocity-potential operator used by the model.

## Arbitrary sampling and regional handoff

`sample_scalar`, `sample_gradient`, and `sample_wind` evaluate spectral fields
at arbitrary finite latitude/longitude points. Scalar values are defined at
exact poles; vector gradients are refused there because eastward orientation is
singular. Associated-Legendre values are built in automatically sized point
chunks whose raw/basis workspace is bounded to approximately 64 MiB, so a
high-resolution regular export does not materialize one `(n,m,point)` cube for
the entire globe. `chunk_points` can be stated explicitly for controlled
experiments without changing per-point arithmetic order.

`export-latlon` samples a checkpoint onto a pole-excluding, cell-centred regular
global grid. The NPZ contains coordinate arrays, all model state needed for an
offline parent handoff, source checkpoint identity, per-array SHA-256 values,
and a metadata self-hash.

The export is deliberately neutral. It is **not** yet a WOOF boundary file or
an initial-condition file. The migration handoff defines the additional
interpolation and coupling gates required before it may force a regional run.

## CLI

```bash
# Arithmetic identity
python -m woof.globe.spectral pins

# Transform/Parseval control
python -m woof.globe.spectral transform-check --truncation 63

# Five-day Williamson test
python -m woof.globe.spectral run \
  global_spectral_williamson2 \
  --outdir out/williamson2

# Four-layer primitive smoke
python -m woof.globe.spectral run \
  global_spectral_primitive_smoke \
  --outdir out/primitive

# Resume the exact same config identity
python -m woof.globe.spectral run CONFIG \
  --restart out/run/global_spectral_step00000100.npz \
  --outdir out/run

# Benchmark transform and RHS (first-call and warmed steady-state separately)
python -m woof.globe.spectral benchmark CONFIG --iterations 20 --warmup 3

# Scalar compression from a Gaussian-grid NPZ field
python -m woof.globe.spectral compress-scalar CONFIG source.npz field.shc.npz \
  --field surface_pressure --space log --floor 1.0 --keep-degree 31

# Vector wind compression
python -m woof.globe.spectral compress-wind CONFIG source.npz wind.shc.npz \
  --u-field u --v-field v --keep-degree 31

# Parent-export artifact
python -m woof.globe.spectral export-latlon CONFIG CHECKPOINT parent.npz \
  --nlat 361 --nlon 720
```

Every output door refuses an existing target unless `--overwrite` is stated.
`run --overwrite` removes only files owned by the Level-3 runner.

## Checkpoints and receipts

A checkpoint binds:

- schema;
- model;
- exact resolved config hash;
- arithmetic pin hash;
- model step and time;
- array inventory, shapes, dtypes and SHA-256 values;
- metadata self-hash;
- inherited whole-run CFL and mass-fixer maxima.

A successful completion receipt records transform controls, cold-start and segment-start
diagnostics, final diagnostics, CFL maximum, mass-fixer maximum, every gate,
checkpoint paths, restart provenance, timing, and its own self-hash.
A failed integration that has already published its initial checkpoint writes a
self-hashed `status = "error"` receipt before re-raising the original exception.
The receipt names the exception, last good state, checkpoints and accumulated
run trackers; it never converts failure into a passing exit status.


The primitive mass target is reconstructed from the deterministic cold state on
resume, not from the checkpoint, so a restart cannot redefine what the run
claims to conserve. Checkpoints also carry validated whole-run CFL and
mass-fixer trackers; the resumed receipt reports both inherited whole-run
maxima and maxima observed only in the resumed segment, rather than grading the
second half of a run as though the first half never happened.

## Current admitted evidence

The shipped CPU reference gates include:

- T3 through T63 transform round trips and Parseval controls;
- direct vector transform/inversion controls;
- exact alpha-zero Williamson-2 discrete balance from independent analytic
  vorticity;
- a five-day T15 Williamson integration with height, wind, vorticity,
  divergence, mass, energy, and potential-enstrophy gates;
- horizontally uniform primitive rest-state balance;
- a four-layer thermal-wave primitive smoke integration;
- pre-update CFL refusals;
- checkpoint metadata and array tamper detection;
- bit-exact primitive restart continuation;
- scalar/wind compression, progressive decoding, positivity and mean repair;
- regular lat/lon export validation;
- CLI and no-CuPy CPU operation.

CuPy arithmetic is implemented but is not called “GPU validated” until the
same battery and benchmark run on the actual target card.

## Non-claims and limitations

Level 3 currently does **not** claim:

- production forecast skill;
- replacement of the regional nonhydrostatic core;
- moist primitive equations;
- orographic primitive dynamics;
- an energy- and angular-momentum-conserving Simmons-Burridge vertical scheme;
- semi-implicit or semi-Lagrangian stability;
- arbitrary tilted Williamson-2 balance;
- a completed global-to-regional lateral-boundary adapter;
- coupled execution with WOOF nests;
- GPU performance or parity without a device receipt.

Those are explicit migration stages, not silently implied by the existence of
an executable prototype.

## References used for the formulation

- Williamson, Drake, Hack, Jakob, and Swarztrauber (1992), *A standard test
  set for numerical approximations to the shallow water equations in spherical
  geometry*.
- Simmons and Burridge (1981), *An energy and angular-momentum conserving
  vertical finite-difference scheme and hybrid vertical coordinates*.
- Held and Suarez (1994), *A proposal for the intercomparison of the
  dynamical cores of atmospheric general circulation models*.
- The open SpeedyWeather primitive-equation documentation and implementation
  were used as a modern cross-check on state choice, sigma continuity, and
  transform ordering; no source code was copied.
