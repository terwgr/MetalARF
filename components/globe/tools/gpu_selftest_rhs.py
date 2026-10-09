#!/usr/bin/env python
"""Device gate for the fused dynamics kernels in ``woof.globe.spectral.fused``.

Run ON A GPU MACHINE with CuPy installed, from a tree that carries this
file (the process must NOT set ``GPUWM_NO_LOCAL_GPU``; the tool opens the
device by design and refuses when the variable forbids it)::

    python tools/gpu_selftest_rhs.py

Each fused op is compared against its numpy specification - the numpy
branch it bypasses - on random and adversarial inputs (zeros, saturated
magnitudes, negative-moisture edges, single-point extremes, exact-zero
interface velocities, both van Leer limiter branches).  Per-op maximum
absolute and field-scale relative diffs are printed.  Exit codes:
0 = all ops within fp reordering tolerance, 1 = at least one op beyond
tolerance, 2 = the environment cannot run the test (no CuPy, or the
local device is forbidden).

**The references are not transcriptions.**  ``momentum_bernoulli_reference``,
``tracer_flux_reference`` and ``scalar_tendency_reference`` below are the
numpy specifications themselves, and ``tests/test_global_spectral_fused_kernels_cpu.py``
compiles each kernel's C body for the CPU and proves it BIT-IDENTICAL to
these exact functions; the vertical flux has no transcription at all - its
specification is the numpy branch of
``MoistHybridModel._vertical_scalar_flux_divergence``, reached here through
``specification_model()``.  So a spec that drifts from the kernel fails on
a CPU box before any card is booked, and this tool measures only what a
CPU cannot: the device launch (cupy CArray indexing, nvcc contraction
policy, GEMM/FFT reduction order).

Current kernel specifications (both moved 2026-09-01, audit task 5):

* momentum/Bernoulli takes the pressure-gradient input ``pg``, the
  full-level ``grad(ln p_k)`` per unit ``grad(ln ps)`` from
  ``MoistHybridModel._pressure_gradient_factor`` (commit 08791fb8d, DN-3);
  ``ge``/``gn`` are now the 2-D gradient of ``ln ps``.
* the vertical flux kernel is the van Leer limited (MUSCL) form and takes
  ``(scalar, omega_half, p_full, p_half)`` instead of the donor-cell pair
  (commit b38462606, DN-4).

Tolerances: the fused kernels keep the specification's association, so
the only licensed divergence is compiler multiply-add contraction (last
bits).  project() performs no arithmetic at all and must match EXACTLY.
The end-to-end rhs() comparison additionally crosses GEMM/FFT reduction
reordering between CPU and GPU BLAS, so it carries a wider tolerance.

fp32 note: single-point-extreme cases use the smallest NORMAL float32,
not subnormals - sm_120 flushes fp32 subnormals in all arithmetic, so a
subnormal case would measure the hardware, not the kernel.
"""
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np

from woof.globe.dynamics import MoistHybridModel
from woof.globe.semi_implicit import BarotropicSemiImplicit
from woof.globe.vertical import HybridCoordinate
from woof.globe.spectral.backend import get_backend

# The kernel factories build lazily, so this module imports on a machine
# with no CuPy and the CPU kernel test can pull the specifications below
# out of it.
from woof.globe.spectral.fused import (
    momentum_bernoulli_kernel,
    scalar_tendency_kernel,
    tracer_flux_kernel,
    vertical_flux_divergence_kernel,
)
from woof.globe.spectral.transform import SphericalHarmonicTransform
from woof.local_gpu import NO_LOCAL_GPU_ENV, no_local_gpu

# Field-scale relative tolerance per dtype: multiply-add contraction moves
# last bits only, so a few ulp at the field's own magnitude.
REL_TOL = {np.float64: 1.0e-12, np.float32: 1.0e-5}
RHS_REL_TOL = 1.0e-9  # crosses CPU-vs-GPU GEMM/FFT reduction order
CONTINUITY_REL_TOL = 1.0e-12  # scan vs loop reassociation over nlev terms

FAILURES: list[str] = []


# ----------------------------------------------------------------------
# The numpy specifications.  These are the functions the CPU kernel test
# compiles the kernel C against, so they are shared, never duplicated.
#
# Each one's parameters are named and ordered exactly as its kernel's
# ``in_params``, and ``tests/test_gpu_selftest_tools.py`` compares the two
# lists: a kernel that gains, loses or reorders an input (the momentum
# kernel gained ``pg`` on 2026-09-01, DN-3) then cannot pass its CPU proof
# until the specification moves with it.
# ----------------------------------------------------------------------

def momentum_bernoulli_reference(zeta, cor, u, v, wu, wv, tv, pg, ge, gn, phi, rgas):
    """Numpy branch of ``MoistHybridModel.rhs``: both forcings and Bernoulli.

    ``pg`` is the full-level ``grad(ln p_k)`` factor (DN-3) and ``ge``/``gn``
    the 2-D ``grad(ln ps)``; the association is the kernel's, statement for
    statement.
    """
    absolute_vorticity = zeta + cor
    pressure_force = rgas * tv * pg
    momentum_u = absolute_vorticity * v - wu - pressure_force * ge
    momentum_v = -absolute_vorticity * u - wv - pressure_force * gn
    bernoulli = 0.5 * (u ** 2 + v ** 2) + phi
    return momentum_u, momentum_v, bernoulli


def tracer_flux_reference(dp, s, u, v):
    """Numpy branch of ``MoistHybridModel._scalar_tendency``: the flux pair.

    ``dp * s`` is associated first in both paths, so the shared product
    rounds once.
    """
    return dp * s * u, dp * s * v


def scalar_tendency_reference(h, vert, s, dpt, dp):
    """Numpy branch of ``MoistHybridModel._scalar_tendency``: the combine.

    ``h`` is the horizontal flux divergence, ``vert`` the vertical one.
    """
    return (-h - vert - s * dpt) / dp


def specification_model(nlev: int, precision: str = "float64", truncation: int = 5):
    """A numpy-backed ``MoistHybridModel`` whose methods ARE the specification.

    Its ``_vertical_scalar_flux_divergence`` takes the numpy branch (the
    van Leer reconstruction the kernel implements), so nothing about that
    flux is transcribed anywhere.  ``truncation`` only sizes the grid the
    model validates its surface geopotential against; the flux method
    reads shapes from its arguments.
    """
    transform = SphericalHarmonicTransform.create(
        truncation, backend="numpy", precision=precision
    )
    return MoistHybridModel(
        transform=transform,
        vertical=HybridCoordinate.pressure_blend(nlev, 100.0),
        surface_geopotential=np.zeros(transform.grid.shape),
        physics=None,
        diffusion=None,
        semi_implicit=BarotropicSemiImplicit(enabled=False),
        mass_fixer=False,
        water_fixer=False,
        positivity_repair=False,
    )


def hybrid_pressures(nlev: int, surface_pressure, dtype):
    """``(p_full, p_half)`` from the real hybrid coordinate.

    Built by ``HybridCoordinate.pressure``, which refuses a nonpositive
    layer thickness, so the limiter's ``p_full`` differences divide safely
    and the case data cannot accidentally invert a level.
    """
    precision = "float64" if dtype == np.float64 else "float32"
    coordinate = HybridCoordinate.pressure_blend(nlev, 100.0)
    pressure = coordinate.pressure(surface_pressure, get_backend("numpy", precision))
    return (
        np.ascontiguousarray(np.asarray(pressure["p_full"], dtype=dtype)),
        np.ascontiguousarray(np.asarray(pressure["p_half"], dtype=dtype)),
    )


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def report(name: str, max_abs: float, rel: float, tol: float) -> None:
    status = "ok  " if rel <= tol else "FAIL"
    print(f"  {status} {name:<58s} max_abs={max_abs:.3e} rel={rel:.3e} tol={tol:.0e}")
    if rel > tol:
        FAILURES.append(name)


def compare(cp, name: str, got_gpu, want_np, tol: float) -> None:
    got = cp.asnumpy(got_gpu)
    want = np.asarray(want_np)
    if got.shape != want.shape:
        print(f"  FAIL {name}: shape {got.shape} != {want.shape}")
        FAILURES.append(name)
        return
    if not (np.isfinite(got).all() and np.isfinite(want).all()):
        # Same non-finite pattern (inf sign / nan placement included) and
        # in-tolerance finite entries counts as a match.
        finite_mask = np.isfinite(want)
        pattern = bool(
            np.array_equal(finite_mask, np.isfinite(got))
            and np.array_equal(
                got[~finite_mask], want[~finite_mask], equal_nan=True
            )
        )
        if not pattern:
            print(f"  FAIL {name}: non-finite pattern differs")
            FAILURES.append(name)
            return
        got, want = got[finite_mask], want[finite_mask]
    max_abs = float(np.max(np.abs(got - want))) if got.size else 0.0
    scale = float(np.max(np.abs(want))) if want.size else 0.0
    rel = max_abs / max(scale, 1.0e-300)
    report(name, max_abs, rel, tol)


def compare_exact(cp, name: str, got_gpu, want_np) -> None:
    got = cp.asnumpy(got_gpu)
    want = np.asarray(want_np)
    equal = got.shape == want.shape and np.array_equal(got, want)
    print(f"  {'ok  ' if equal else 'FAIL'} {name:<58s} exact={equal}")
    if not equal:
        FAILURES.append(name)


def rng():
    return np.random.default_rng(20260831)


# ----------------------------------------------------------------------
# Per-op checks: fused kernel vs the numpy specification it bypasses.
# Shapes carry a leading tracer axis where the call site stacks tracers.
# ----------------------------------------------------------------------

def momentum_cases(dtype):
    r = rng()
    nlev, nlat, nlon = 6, 8, 16
    shape = (nlev, nlat, nlon)
    base = {
        "zeta": r.normal(0.0, 1.0e-5, shape),
        "cor": (2.0 * 7.292e-5 * np.sin(np.linspace(-1.4, 1.4, nlat)))[
            :, None
        ],
        "u": r.uniform(-60.0, 60.0, shape),
        "v": r.uniform(-60.0, 60.0, shape),
        "wu": r.normal(0.0, 1.0e-3, shape),
        "wv": r.normal(0.0, 1.0e-3, shape),
        "tv": r.uniform(180.0, 320.0, shape),
        # grad(ln p_k) per unit grad(ln ps): 1 at the surface where the
        # coordinate is pure sigma, falling toward 0 at the pressure top
        # (DN-3).  A real column's factor is monotone in k, so the ladder
        # is used rather than noise.
        "pg": np.broadcast_to(
            np.linspace(0.02, 1.0, nlev)[:, None, None], shape
        ),
        "ge": np.broadcast_to(r.normal(0.0, 1.0e-5, (nlat, nlon)), shape),
        "gn": np.broadcast_to(r.normal(0.0, 1.0e-5, (nlat, nlon)), shape),
        "phi": r.uniform(0.0, 3.0e5, shape),
    }
    base = {k: np.ascontiguousarray(v) for k, v in base.items()}
    yield "random", base
    yield "zeros", {k: np.zeros_like(v) for k, v in base.items()}
    sat = {k: np.array(v, copy=True) for k, v in base.items()}
    sat["u"][0, 0, 0] = 1.0e30
    sat["v"][2, 3, 4] = -1.0e30
    yield "single-point-extreme-wind", sat
    tiny = {k: np.array(v, copy=True) for k, v in base.items()}
    small = np.finfo(dtype).tiny  # smallest normal; see fp32 note above
    tiny["zeta"][1, 1, 1] = small
    tiny["ge"][1, 1, 1] = -small
    tiny["pg"][0] = 0.0  # a rigid-lid top level contributes no pressure force
    yield "single-point-smallest-normal", tiny


def check_momentum(cp, dtype):
    tol = REL_TOL[dtype]
    rgas = dtype(287.0528)
    order = ("zeta", "cor", "u", "v", "wu", "wv", "tv", "pg", "ge", "gn", "phi")
    for label, c in momentum_cases(dtype):
        c = {k: np.asarray(v, dtype=dtype) for k, v in c.items()}
        # The single-point-extreme case overflows the Bernoulli square on
        # purpose (1e30 squared is inf in float32); compare() judges the
        # non-finite pattern, so numpy's warning carries no information.
        with np.errstate(over="ignore"):
            want_mu, want_mv, want_b = momentum_bernoulli_reference(
                *(c[name] for name in order), rgas
            )
        d = {k: cp.asarray(v) for k, v in c.items()}
        mu, mv, bern = momentum_bernoulli_kernel(cp)(
            *(d[name] for name in order), rgas
        )
        compare(cp, f"momentum_u[{np.dtype(dtype).name}/{label}]", mu, want_mu, tol)
        compare(cp, f"momentum_v[{np.dtype(dtype).name}/{label}]", mv, want_mv, tol)
        compare(cp, f"bernoulli[{np.dtype(dtype).name}/{label}]", bern, want_b, tol)


def check_tracer_flux(cp, dtype):
    tol = REL_TOL[dtype]
    r = rng()
    ntr, nlev, nlat, nlon = 3, 6, 8, 16
    dp = r.uniform(50.0, 5000.0, (nlev, nlat, nlon))
    u = r.uniform(-80.0, 80.0, (nlev, nlat, nlon))
    v = r.uniform(-80.0, 80.0, (nlev, nlat, nlon))
    s = r.uniform(0.0, 0.02, (ntr, nlev, nlat, nlon))
    s[0, 0, 0, 0] = -1.0e-12  # negative-moisture ringing edge
    s[1, 2, 3, 4] = -4.0e-3
    cases = {"random": (dp, s, u, v), "zero-scalar": (dp, np.zeros_like(s), u, v)}
    sat = np.array(s, copy=True)
    sat[2, 5, 7, 15] = 1.0e30
    cases["single-point-saturated"] = (dp, sat, u, v)
    for label, (a, b, cu, cv) in cases.items():
        a, b, cu, cv = (np.asarray(x, dtype=dtype) for x in (a, b, cu, cv))
        want_fx, want_fy = tracer_flux_reference(a, b, cu, cv)
        fx, fy = tracer_flux_kernel(cp)(
            cp.asarray(a), cp.asarray(b), cp.asarray(cu), cp.asarray(cv)
        )
        compare(cp, f"tracer_flux_east[{np.dtype(dtype).name}/{label}]", fx, want_fx, tol)
        compare(cp, f"tracer_flux_north[{np.dtype(dtype).name}/{label}]", fy, want_fy, tol)


def check_scalar_tendency(cp, dtype):
    tol = REL_TOL[dtype]
    r = rng()
    ntr, nlev, nlat, nlon = 3, 6, 8, 16
    h = r.normal(0.0, 1.0e-3, (ntr, nlev, nlat, nlon))
    vert = r.normal(0.0, 1.0e-3, (ntr, nlev, nlat, nlon))
    s = r.uniform(-1.0e-6, 0.02, (ntr, nlev, nlat, nlon))
    dpt = r.normal(0.0, 1.0e-2, (nlev, nlat, nlon))
    dp = r.uniform(50.0, 5000.0, (nlev, nlat, nlon))
    cases = {"random": (h, vert, s, dpt, dp)}
    thin = np.array(dp, copy=True)
    thin[0] = 1.0e-3  # sub-pascal layer: the division must not be floored
    cases["thin-layer"] = (h, vert, s, dpt, thin)
    cases["zeros"] = tuple(np.zeros_like(x) for x in (h, vert, s, dpt)) + (dp,)
    for label, args in cases.items():
        a = [np.asarray(x, dtype=dtype) for x in args]
        want = scalar_tendency_reference(*a)
        got = scalar_tendency_kernel(cp)(*(cp.asarray(x) for x in a))
        compare(cp, f"scalar_tendency[{np.dtype(dtype).name}/{label}]", got, want, tol)


def vertical_flux_cases(dtype):
    """Case data for the van Leer flux: fields, pressures, and the model.

    The three stacked tracers are a smooth profile (limiter active on
    every interior interface), a sign-changing noisy one (both limiter
    branches and both upstream directions) and a top hat (the face bound
    clips), matching the shapes the CPU kernel test exercises.
    """
    r = rng()
    for label, (lead, nlev, nlat, nlon) in {
        "tracer-stack": (3, 6, 8, 16),
        "no-lead": (0, 6, 8, 16),
        "minimum-nlev-2": (2, 2, 4, 8),
    }.items():
        shape = (nlat, nlon)
        ps = np.asarray(
            1.0e5 * (1.0 + 0.05 * r.standard_normal(shape)), dtype=dtype
        )
        p_full, p_half = hybrid_pressures(nlev, ps, dtype)
        smooth = 300.0 + 0.01 * p_full + r.standard_normal((nlev, *shape))
        noisy = r.standard_normal((nlev, *shape))
        hat = np.where((p_full > 3.0e4) & (p_full < 7.0e4), 1.0, 0.0)
        stack = np.stack([smooth, noisy, hat])
        scalar = stack[0] if lead == 0 else np.ascontiguousarray(stack[:lead])
        omega = np.asarray(0.5 * r.standard_normal((nlev + 1, *shape)), dtype=dtype)
        omega[1, 0, :] = 0.0  # exact zero rides the >= branch (upper level)
        omega[0] = 123.0  # top interface value must be IGNORED (flux pinned 0)
        omega[-1] = -321.0  # bottom interface likewise
        for sub, sc in {
            "": scalar,
            "/zeros": np.zeros_like(scalar),
            "/single-point-saturated": np.where(
                np.arange(scalar.size).reshape(scalar.shape) == 1, 1.0e30, scalar
            ),
        }.items():
            yield (
                f"{label}{sub}",
                nlev,
                np.ascontiguousarray(np.asarray(sc, dtype=dtype)),
                omega,
                p_full,
                p_half,
            )


def limiter_branch_census(nlev, scalar, omega, p_full):
    """Which van Leer branches a case reaches: (limited, flat, down, up)."""
    if nlev < 3:
        return (False, False, bool(np.any(omega[1:nlev] >= 0.0)),
                bool(np.any(omega[1:nlev] < 0.0)))
    above = (scalar[..., 1:-1, :, :] - scalar[..., :-2, :, :]) / (
        p_full[1:-1] - p_full[:-2]
    )
    below = (scalar[..., 2:, :, :] - scalar[..., 1:-1, :, :]) / (
        p_full[2:] - p_full[1:-1]
    )
    product = above * below
    interior = omega[1:nlev]
    return (
        bool(np.any(product > 0.0)),
        bool(np.any(product <= 0.0)),
        bool(np.any(interior >= 0.0)),
        bool(np.any(interior < 0.0)),
    )


def check_vertical_flux(cp, dtype):
    tol = REL_TOL[dtype]
    precision = "float64" if dtype == np.float64 else "float32"
    models: dict[int, object] = {}
    reached = [False, False, False, False]
    for label, nlev, scalar, omega, p_full, p_half in vertical_flux_cases(dtype):
        model = models.setdefault(
            nlev, specification_model(nlev, precision=precision)
        )
        want = model._vertical_scalar_flux_divergence(
            scalar, omega, p_full, p_half
        )
        out = cp.empty(scalar.shape, dtype=dtype)
        vertical_flux_divergence_kernel(cp)(
            cp.asarray(scalar), cp.asarray(omega), cp.asarray(p_full),
            cp.asarray(p_half), nlev, scalar.shape[-2] * scalar.shape[-1], out,
        )
        compare(cp, f"vertical_flux[{np.dtype(dtype).name}/{label}]", out, want, tol)
        census = limiter_branch_census(nlev, scalar, omega, p_full)
        reached = [a or b for a, b in zip(reached, census)]
    # A suite that stops reaching a branch measures nothing about it, and
    # the donor-cell path this replaced had no limiter to lose (DN-4).
    name = f"vertical_flux[{np.dtype(dtype).name}/branch-census]"
    labels = ("limited-gradient", "flat-gradient", "upstream-above", "upstream-below")
    missing = [n for n, hit in zip(labels, reached) if not hit]
    print(f"  {'ok  ' if not missing else 'FAIL'} {name:<58s} "
          f"reached={[n for n, hit in zip(labels, reached) if hit]}")
    if missing:
        FAILURES.append(name)


def check_vertical_flux_noncontiguous(cp, dtype):
    """Raw-array linear indexing must honor device strides.

    The kernel's flat-index math addresses the LOGICAL C-order element,
    which CuPy's CArray decomposes through the view's strides, so a
    non-contiguous device view must produce the same field the numpy
    specification computes from the contiguous copy.  All four of the van
    Leer kernel's inputs are ``raw`` (the reconstruction reaches level
    neighbours), so all four are passed as views here.  No current call
    site passes one, but the raw indexing contract is the kernel's one
    unchecked assumption, so it is pinned.
    """
    tol = REL_TOL[dtype]
    precision = "float64" if dtype == np.float64 else "float32"
    r = rng()
    lead, nlev, nlat, nlon = 3, 6, 8, 16
    shape = (nlat, nlon)
    ps = np.asarray(1.0e5 * (1.0 + 0.05 * r.standard_normal(shape)), dtype=dtype)
    p_full, p_half = hybrid_pressures(nlev, ps, dtype)

    def strided(dense):
        """A device view whose last axis strides by 2 over a padded buffer."""
        padded = np.zeros((*dense.shape[:-1], 2 * dense.shape[-1]), dtype=dtype)
        padded[..., ::2] = dense
        view = cp.asarray(padded)[..., ::2]
        assert not view.flags.c_contiguous
        return view

    scalar = np.ascontiguousarray(
        np.asarray(r.uniform(-1.0e-3, 0.02, (lead, nlev, *shape)), dtype=dtype)
    )
    omega = np.asarray(r.normal(0.0, 0.5, (nlev + 1, *shape)), dtype=dtype)
    model = specification_model(nlev, precision=precision)
    want = model._vertical_scalar_flux_divergence(scalar, omega, p_full, p_half)
    out = cp.empty(scalar.shape, dtype=dtype)
    vertical_flux_divergence_kernel(cp)(
        strided(scalar), strided(omega), strided(p_full), strided(p_half),
        nlev, nlat * nlon, out,
    )
    compare(
        cp, f"vertical_flux[{np.dtype(dtype).name}/raw-noncontiguous]",
        out, want, tol,
    )


def check_project(cp, precision):
    """project() does no arithmetic: fused output must match EXACTLY."""
    t_np = SphericalHarmonicTransform.create(
        21, backend="numpy", precision=precision
    )
    t_cp = SphericalHarmonicTransform.create(
        21, backend="cupy", precision=precision
    )
    r = rng()
    n = t_np.truncation + 1
    for label, lead in {"2d": (), "stack": (5,), "double-stack": (3, 4)}.items():
        coeff = (
            r.normal(size=(*lead, n, n)) + 1j * r.normal(size=(*lead, n, n))
        )
        # Above-triangle garbage must be zeroed; m=0 imaginary parts dropped.
        coeff[..., 0, :] += 1.0e30
        coeff[..., :, 0] += 1j * r.normal(size=(*lead, n))
        want = t_np.project(coeff)
        got = t_cp.project(cp.asarray(coeff))
        compare_exact(cp, f"project[{precision}/{label}]", got, want)
        zeros = np.zeros_like(coeff)
        compare_exact(
            cp, f"project[{precision}/{label}/zeros]",
            t_cp.project(cp.asarray(zeros)),
            t_np.project(zeros),
        )


# ----------------------------------------------------------------------
# Artifact-level checks: the real model methods, numpy vs cupy backend.
# ----------------------------------------------------------------------

MODEL_TOML = """
[arwen_global]
schema = "gpuwm.arwen-global-run/v1"
name = "fusion-gpu-selftest"
acknowledgement = "research-only-arwen-global-v1"
backend = "numpy"
precision = "float64"

[grid]
truncation = 21
dealias_factor = 1.5

[time]
dt_s = 60.0
duration_s = 120.0
output_interval_s = 60.0
maximum_cfl = 0.90
integrator = "ssprk3"

[vertical]
coordinate = "pressure_blend"
nlev = 5
p_top_pa = 100.0

[initial]
surface_pressure_pa = 100000.0
surface_temperature_k = 286.0
top_temperature_k = 220.0
qv_surface = 0.006
zonal_wind_m_s = 12.0
perturbation_amplitude = 0.0001
zonal_wavenumber = 2
terrain_amplitude_m = 250.0
surface_water_kg_m2 = 500.0

[physics]
mode = "reference"

[reference_physics]
radiation = true
surface_fluxes = true
turbulence = true
convection = true
saturation_adjustment = true
microphysics = true
solar_constant_w_m2 = 1361.0
atmospheric_shortwave_absorptivity = 0.08
atmospheric_longwave_emissivity = 0.35
bulk_exchange_coefficient = 0.0005
background_diffusivity_m2_s = 0.05
pbl_diffusivity_m2_s = 2.0
pbl_depth_m = 1200.0
cloud_autoconversion_threshold = 0.001
cloud_autoconversion_time_s = 1800.0
ice_to_snow_time_s = 2400.0
snow_to_graupel_time_s = 4800.0
rain_fall_speed_m_s = 4.0
snow_fall_speed_m_s = 0.7
graupel_fall_speed_m_s = 2.0
soil_exchange_time_s = 86400.0

[diffusion]
enabled = true
order = 4
e_folding_time_s_at_truncation = 21600.0
preserve_degree = 1
divergence_strength = 1.5
pressure_strength = 0.25
water_strength = 0.5

[semi_implicit]
enabled = true
external_wave_speed_m_s = 450.0
weight = 1.0

[repair]
mass_fixer = true
water_fixer = true
positivity_repair = true

[gates]
transform_roundtrip_relative_linf = 5.0e-11
transform_parseval_relative_error = 5.0e-12
mass_relative_drift = 1.0e-9
total_water_relative_drift = 1.0e-8
"""


def build_models():
    from woof.globe.config import load_config
    from woof.globe.runner import build_model_and_cold_state

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "selftest.toml"
        path.write_text(MODEL_TOML, encoding="utf-8")
        cfg_np = load_config(path)
        cfg_cp = dataclasses.replace(cfg_np, backend="cupy")
    model_np, cold_np = build_model_and_cold_state(cfg_np)
    model_cp, _ = build_model_and_cold_state(cfg_cp)
    return cfg_np, model_np, cold_np, model_cp


def check_rhs_end_to_end(cp, model_np, cold_np, model_cp):
    from woof.globe.constants import SPECTRAL_FIELDS

    # Perturb every spectral field with tiny projected noise so the cold
    # state's zero moments and near-zero condensates still exercise the
    # tracer branches; both backends receive the identical state.
    r = rng()
    fields = []
    for f in cold_np.atmosphere.fields():
        noise = r.normal(0.0, 1.0e-8, f.shape) + 1j * r.normal(
            0.0, 1.0e-8, f.shape
        )
        fields.append(f + model_np.transform.project(noise))
    state_np = cold_np.atmosphere.with_fields(fields)
    state_cp = state_np.with_fields(
        [cp.asarray(f) for f in state_np.fields()]
    )
    out_np = model_np.rhs(state_np)
    out_cp = model_cp.rhs(state_cp)
    for name in SPECTRAL_FIELDS:
        compare(
            cp,
            f"rhs.{name}[artifact/T21]",
            getattr(out_cp, name),
            getattr(out_np, name),
            RHS_REL_TOL,
        )


def check_vertical_flux_artifact(cp, model_np, model_cp):
    r = rng()
    nlev = model_np.nlev
    nlat, nlon = model_np.transform.grid.shape
    ps = 1.0e5 * (1.0 + 0.05 * r.standard_normal((nlat, nlon)))
    pressure = model_np.vertical.pressure(ps, model_np.transform.backend)
    p_full, p_half = pressure["p_full"], pressure["p_half"]
    scalar = r.uniform(-1.0e-3, 0.02, (4, nlev, nlat, nlon))
    omega = r.normal(0.0, 0.5, (nlev + 1, nlat, nlon))
    omega[2, :, :] = 0.0
    want = model_np._vertical_scalar_flux_divergence(
        scalar, omega, p_full, p_half
    )
    got = model_cp._vertical_scalar_flux_divergence(
        cp.asarray(scalar), cp.asarray(omega),
        cp.asarray(p_full), cp.asarray(p_half),
    )
    compare(cp, "vertical_flux[artifact/T21]", got, want, REL_TOL[np.float64])


def check_continuity(cp, cfg, model_np, model_cp):
    r = rng()
    vertical = cfg.vertical
    nlat, nlon = model_np.transform.grid.shape
    divm = r.normal(0.0, 1.0e-2, (vertical.nlev, nlat, nlon))
    ps_t = -np.sum(divm, axis=0)
    omega_np, residual_np = vertical.continuity(
        divm, ps_t, model_np.transform.backend
    )
    omega_cp, residual_cp = vertical.continuity(
        cp.asarray(divm), cp.asarray(ps_t), model_cp.transform.backend
    )
    compare(cp, "continuity.omega[artifact]", omega_cp, omega_np, CONTINUITY_REL_TOL)
    # residual scale is roundoff; compare at the omega field's own scale.
    got = cp.asnumpy(residual_cp)
    max_abs = float(np.max(np.abs(got - residual_np)))
    scale = max(float(np.max(np.abs(omega_np))), 1.0e-300)
    report("continuity.residual[artifact]", max_abs, max_abs / scale,
           CONTINUITY_REL_TOL)


def main() -> int:
    if no_local_gpu():
        print(
            f"REFUSED: {NO_LOCAL_GPU_ENV} is set, and this self-test opens the "
            "local CUDA device by design - it compiles and launches every "
            "fused kernel and builds a cupy-backend model. There is nothing "
            "for it to measure without a card. Unset the variable on the node "
            "that owns the device, or run the CPU half instead: "
            "pytest tests/test_global_spectral_fused_kernels_cpu.py, which "
            "compiles the same kernel C for the CPU and proves it "
            "bit-identical to the numpy specifications this tool uses."
        )
        return 2
    try:
        import cupy as cp
    except Exception as exc:  # noqa: BLE001 - report and refuse to fake a pass
        print(f"SKIP-FAIL: CuPy unavailable ({exc!r}); run on a GPU machine")
        return 2

    print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    print("== per-op fused kernels vs numpy specification ==")
    for dtype in (np.float64, np.float32):
        check_momentum(cp, dtype)
        check_tracer_flux(cp, dtype)
        check_scalar_tendency(cp, dtype)
        check_vertical_flux(cp, dtype)
        check_vertical_flux_noncontiguous(cp, dtype)
    for precision in ("float64", "float32"):
        check_project(cp, precision)
    print("== artifact-level: real model methods, numpy vs cupy backend ==")
    cfg, model_np, cold_np, model_cp = build_models()
    check_rhs_end_to_end(cp, model_np, cold_np, model_cp)
    check_vertical_flux_artifact(cp, model_np, model_cp)
    check_continuity(cp, cfg, model_np, model_cp)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} op(s) beyond tolerance:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("ALL FUSED OPS MATCH THE NUMPY SPECIFICATION")
    return 0


if __name__ == "__main__":
    sys.exit(main())
