"""The compiled semi-Lagrangian kernels against the numpy specification.

A separate module from the two instrument-gate files because the suite marks
every module that imports CuPy as a device module and skips all of it when
the machine may not open a card.  Keeping these three here leaves the
twenty-nine tests that need no device runnable on a CPU-only box, which is
where the full suite runs.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from woof.globe.spectral.grid import GaussianGrid
from woof.local_gpu import NO_LOCAL_GPU_ENV, no_local_gpu
from woof.globe.semilag.cases import (
    CASE_PERIOD_S,
    _mesh,
    solid_body_wind,
)
from woof.globe.semilag.interpolate import (
    Stencil,
    gather,
    gather_batch,
    zero_stencil,
)
from woof.globe.semilag.tables import SphericalGridTables
from woof.globe.semilag.trajectory import (
    CartesianWind,
    cartesian_wind,
    departure_points,
)

NLEV = 4


def _solid_body(truncation, alpha, *, xp=np, dtype=np.float64):
    grid = GaussianGrid.create(truncation)
    tables = SphericalGridTables.create(grid, xp=xp, dtype=dtype)
    lam, phi = _mesh(grid)
    speed = 2.0 * math.pi * grid.radius_m / CASE_PERIOD_S
    u2, v2 = solid_body_wind(lam, phi, u0=speed, alpha=alpha)
    u = xp.asarray(np.repeat(u2[None], NLEV, axis=0).astype(dtype))
    v = xp.asarray(np.repeat(v2[None], NLEV, axis=0).astype(dtype))
    vx, vy, vz = cartesian_wind(u, v, tables, xp=xp, dtype=dtype)
    wind = CartesianWind(
        xp.ascontiguousarray(vx), xp.ascontiguousarray(vy),
        xp.ascontiguousarray(vz), xp.zeros_like(vx),
    )
    return grid, tables, wind, speed


@pytest.fixture()
def cupy_module():
    cupy = pytest.importorskip("cupy")
    if no_local_gpu():
        pytest.skip(
            f"{NO_LOCAL_GPU_ENV} is set: no device for the semi-Lagrangian "
            "kernels"
        )
    try:
        cupy.zeros(1)
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no usable CUDA device: {exc}")
    return cupy



@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("monotone", [False, True])
def test_kernel_matches_the_numpy_specification(cupy_module, dtype, monotone):
    """The compiled gather answers what the numpy specification answers."""
    cupy = cupy_module
    grid = GaussianGrid.create(31)
    nlev = 6
    shape = (nlev, grid.nlat, grid.nlon)
    rng = np.random.default_rng(4242)
    fields = [rng.standard_normal(shape).astype(dtype) for _ in range(3)]
    coords = {
        "xi": rng.uniform(-3.0, grid.nlon + 3.0, shape).astype(dtype),
        "phi": rng.uniform(-math.pi / 2, math.pi / 2, shape).astype(dtype),
        "level": rng.uniform(-0.5, nlev + 0.5, shape).astype(dtype),
    }

    host_tables = SphericalGridTables.create(grid, xp=np, dtype=dtype)
    host = gather_batch(
        fields, Stencil(tables=host_tables, **coords), monotone=monotone
    )

    dev_tables = SphericalGridTables.create(grid, xp=cupy, dtype=dtype)
    dev_stencil = Stencil(
        tables=dev_tables,
        **{k: cupy.asarray(v) for k, v in coords.items()},
    )
    device = gather_batch(
        [cupy.asarray(f) for f in fields], dev_stencil, monotone=monotone
    )

    tol = 3e-6 if dtype == np.float32 else 1e-12
    for a, b in zip(host, device):
        assert float(np.max(np.abs(a - cupy.asnumpy(b)))) < tol


def test_kernel_zero_displacement_is_bit_exact(cupy_module):
    cupy = cupy_module
    grid = GaussianGrid.create(31)
    tables = SphericalGridTables.create(grid, xp=cupy, dtype=np.float32)
    nlev = 8
    stencil = zero_stencil(tables, nlev, xp=cupy, dtype=np.float32)
    rng = np.random.default_rng(5)
    field = cupy.asarray(
        rng.standard_normal((nlev, grid.nlat, grid.nlon)).astype(np.float32))
    for monotone in (False, True):
        out = gather(field, stencil, monotone=monotone)
        assert bool((out == field).all())




@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("monotone", [False, True])
def test_quintic_kernel_matches_the_numpy_specification(cupy_module, dtype, monotone):
    """The six-point gather (sl_gather5_*) answers what the numpy
    specification answers at order 6, departure points anywhere on the
    sphere including both polar caps."""
    cupy = cupy_module
    grid = GaussianGrid.create(31)
    nlev = 6
    shape = (nlev, grid.nlat, grid.nlon)
    rng = np.random.default_rng(2424)
    fields = [rng.standard_normal(shape).astype(dtype) for _ in range(3)]
    coords = {
        "xi": rng.uniform(-3.0, grid.nlon + 3.0, shape).astype(dtype),
        "phi": rng.uniform(-math.pi / 2, math.pi / 2, shape).astype(dtype),
        "level": rng.uniform(-0.5, nlev + 0.5, shape).astype(dtype),
    }
    host_tables = SphericalGridTables.create(grid, xp=np, dtype=dtype)
    host = gather_batch(
        fields, Stencil(tables=host_tables, **coords), monotone=monotone,
        order=6,
    )
    dev_tables = SphericalGridTables.create(grid, xp=cupy, dtype=dtype)
    dev_stencil = Stencil(
        tables=dev_tables,
        **{k: cupy.asarray(v) for k, v in coords.items()},
    )
    device = gather_batch(
        [cupy.asarray(f) for f in fields], dev_stencil, monotone=monotone,
        order=6,
    )
    tol = 3e-6 if dtype == np.float32 else 1e-12
    for a, b in zip(host, device):
        assert float(np.max(np.abs(a - cupy.asnumpy(b)))) < tol


def test_quintic_kernel_zero_displacement_is_bit_exact(cupy_module):
    cupy = cupy_module
    grid = GaussianGrid.create(31)
    tables = SphericalGridTables.create(grid, xp=cupy, dtype=np.float32)
    nlev = 8
    stencil = zero_stencil(tables, nlev, xp=cupy, dtype=np.float32)
    rng = np.random.default_rng(6)
    field = cupy.asarray(
        rng.standard_normal((nlev, grid.nlat, grid.nlon)).astype(np.float32))
    for monotone in (False, True):
        out = gather(field, stencil, monotone=monotone, order=6)
        assert bool((out == field).all())


def test_the_cubic_kernel_is_the_same_kernel_after_the_quintic_joined(cupy_module):
    """The four-point entry points read the four-point tables and answer
    the four-point specification exactly as before: the template's NH = 4
    branch is the arithmetic the gates of record were measured on."""
    cupy = cupy_module
    grid = GaussianGrid.create(31)
    nlev = 5
    shape = (nlev, grid.nlat, grid.nlon)
    rng = np.random.default_rng(99)
    field = rng.standard_normal(shape).astype(np.float64)
    coords = {
        "xi": rng.uniform(-3.0, grid.nlon + 3.0, shape),
        "phi": rng.uniform(-math.pi / 2, math.pi / 2, shape),
        "level": rng.uniform(-0.5, nlev + 0.5, shape),
    }
    host = gather(field, Stencil(tables=SphericalGridTables.create(
        grid, xp=np, dtype=np.float64), **coords), monotone=False, order=4)
    dev_tables = SphericalGridTables.create(grid, xp=cupy, dtype=np.float64)
    device = gather(cupy.asarray(field), Stencil(
        tables=dev_tables, **{k: cupy.asarray(v) for k, v in coords.items()}
    ), monotone=False, order=4)
    assert float(np.max(np.abs(host - cupy.asnumpy(device)))) < 1e-12
    quintic = gather(cupy.asarray(field), Stencil(
        tables=dev_tables, **{k: cupy.asarray(v) for k, v in coords.items()}
    ), monotone=False, order=6)
    # a different stencil answers differently on a random field
    assert float(np.max(np.abs(cupy.asnumpy(quintic) - host))) > 1e-6


def test_departure_kernel_matches_the_numpy_specification(cupy_module):
    cupy = cupy_module
    dt = CASE_PERIOD_S / 256.0
    grid, host_tables, host_wind, _ = _solid_body(31, math.pi / 3.0)
    host_stencil, host_diag = departure_points(
        host_wind, host_tables, dt, iterations=3)

    dev_tables = SphericalGridTables.create(grid, xp=cupy, dtype=np.float64)
    dev_wind = CartesianWind(
        *[cupy.asarray(a) for a in host_wind.arrays()])
    dev_stencil, dev_diag = departure_points(
        dev_wind, dev_tables, dt, iterations=3)

    assert float(np.max(np.abs(
        np.asarray(host_stencil.xi) - cupy.asnumpy(dev_stencil.xi)))) < 1e-9
    assert float(np.max(np.abs(
        np.asarray(host_stencil.phi) - cupy.asnumpy(dev_stencil.phi)))) < 1e-12
    assert dev_diag.displacement_max_m == pytest.approx(
        host_diag.displacement_max_m, rel=1e-9)


def test_fused_lipschitz_norms_match_their_numpy_specification(cupy_module):
    """The one-launch norm kernel answers what the array form answers.

    The array form is kept in place as the specification and is what the CPU
    path runs; the kernel exists because the array form ran about a hundred
    grid-sized passes and cost 18.6 ms at T255 and 147.2 ms at T533 on the
    5070 Ti, five times the fourteen-field gather it is a diagnostic on.
    """
    cupy = cupy_module
    from woof.globe.semilag.trajectory import lipschitz

    for dtype in (np.float32, np.float64):
        grid = GaussianGrid.create(31)
        tables = SphericalGridTables.create(grid, xp=cupy, dtype=dtype)
        nlev = 10
        shape = (nlev, grid.nlat, grid.nlon)
        lam = np.asarray(grid.lon_rad)[None, None, :]
        phi = np.asarray(grid.lat_rad)[None, :, None]
        k = np.arange(nlev)[:, None, None] / (nlev - 1)
        u = np.broadcast_to(45.0 * np.cos(phi) * (0.2 + k)
                            * (1.0 + 0.3 * np.sin(2.0 * lam)), shape)
        v = np.broadcast_to(9.0 * np.sin(3.0 * lam) * np.cos(phi) * (1.0 - k),
                            shape)
        rate = np.broadcast_to(
            3.0e-4 * np.sin(math.pi * k) * (1.0 + 0.2 * np.cos(lam)), shape)
        vx, vy, vz = cartesian_wind(
            cupy.asarray(u.astype(dtype)), cupy.asarray(v.astype(dtype)),
            tables, xp=cupy, dtype=dtype)
        wind = CartesianWind(
            cupy.ascontiguousarray(vx), cupy.ascontiguousarray(vy),
            cupy.ascontiguousarray(vz),
            cupy.asarray(np.ascontiguousarray(rate).astype(dtype)))
        fused = lipschitz(wind, tables, 300.0, fused_norms=True)
        plain = lipschitz(wind, tables, 300.0, fused_norms=False)
        rel = 3e-6 if dtype == np.float32 else 1e-12
        assert fused.lipschitz == pytest.approx(plain.lipschitz, rel=rel)
        assert fused.lipschitz_frobenius == pytest.approx(
            plain.lipschitz_frobenius, rel=rel)
        assert fused.lipschitz_horizontal == pytest.approx(
            plain.lipschitz_horizontal, rel=rel)
        assert fused.lipschitz_balanced == pytest.approx(
            plain.lipschitz_balanced, rel=rel)
        assert fused.lipschitz > 0.0


def _cartesian_polynomial(lam, phi):
    """A polynomial in the Cartesian unit vector.

    Smooth everywhere on the sphere, poles included, and the rows the polar
    reflection reads are its own continuation.  The order gate's own field,
    ``sin(2 lam) cos^2(phi) + 0.5 cos(3 lam) sin(phi)``, is not: its second
    term is ``0.5 z cos(3 lam)`` and ``cos(3 lam)`` has no limit at a pole,
    which is why that gate reads only ``|lat| < 1.0`` rad and why it cannot
    grade the polar stencil at all.
    """
    x = np.cos(phi) * np.cos(lam)
    y = np.cos(phi) * np.sin(lam)
    z = np.sin(phi) * np.ones_like(x)
    return (1.0 + 0.7 * x - 0.4 * y + 0.9 * z + 1.3 * x * y * z
            + 0.6 * x * x - 0.5 * z * z * z + 0.35 * y * y * x)


def test_the_gather_is_fourth_order_inside_the_polar_cap(cupy_module):
    """Named breakage: a polar stencil that is right but not accurate.

    The bit-exact identity at the poles and the KERN norms both pass with a
    meridional stencil that reads the reflected rows in the wrong order or
    with the wrong weights, because the identity is exact for any weight set
    that sums to one at a node and the norms average the two outermost rings
    into 245 interior ones.  What separates a correct polar stencil from a
    plausible one is its ORDER inside the cap, where the reflected rows are
    the ones being read.  MEASURED 2026-09-06 on the 16 GB host: 4.15 and 4.09
    poleward of 80 degrees, against 4.00 and 3.99 equatorward of 57.
    """
    cupy = cupy_module
    errors = {"interior": [], "cap": []}
    spacing = []
    for truncation in (63, 127):
        grid = GaussianGrid.create(truncation)
        tables = SphericalGridTables.create(grid, xp=cupy, dtype=np.float64)
        lat = np.asarray(grid.lat_rad)
        shape = (NLEV, grid.nlat, grid.nlon)
        field = cupy.asarray(np.broadcast_to(_cartesian_polynomial(
            np.asarray(grid.lon_rad)[None, None, :], lat[None, :, None]),
            shape).astype(np.float64))
        target = np.minimum(lat + 0.25 * np.gradient(lat),
                            math.pi / 2.0 - 1e-9)
        stencil = Stencil(
            xi=cupy.asarray(np.broadcast_to(
                np.arange(grid.nlon, dtype=np.float64) + 0.5, shape).copy()),
            phi=cupy.asarray(np.broadcast_to(
                target[None, :, None], shape).copy()),
            level=cupy.asarray(np.broadcast_to(
                np.arange(NLEV, dtype=np.float64)[:, None, None],
                shape).copy()),
            tables=tables,
        )
        got = cupy.asnumpy(gather(field, stencil, monotone=False))
        exact = _cartesian_polynomial(
            ((np.arange(grid.nlon) + 0.5) * tables.dlam)[None, None, :],
            target[None, :, None])
        err = got - np.broadcast_to(exact, shape)
        cap = np.abs(target) >= 1.396
        interior = np.abs(target) < 1.0
        assert int(cap.sum()) >= 4
        errors["cap"].append(float(np.sqrt(np.mean(err[:, cap, :] ** 2))))
        errors["interior"].append(
            float(np.sqrt(np.mean(err[:, interior, :] ** 2))))
        spacing.append(1.0 / grid.nlon)
    for band, values in errors.items():
        slope = (math.log(values[0] / values[1])
                 / math.log(spacing[0] / spacing[1]))
        assert slope > 3.5, (band, values, slope)


def test_the_gather_is_fourth_order_in_the_vertical(cupy_module):
    """Named breakage: a vertical stencil nobody grades.

    Every KERN case is two-dimensional and runs four levels with a zero
    level rate, so the vertical direction is exercised as an exact identity
    and its ORDER is measured by nothing.  A vertical weight set that is
    second order, or that reads the levels in the wrong order below the
    bracketing pair, passes every other gate in this package.  MEASURED
    2026-09-06 on the 16 GB host: 4.06, 4.04 and 4.02 over 20, 40, 80 and 160
    levels.
    """
    cupy = cupy_module
    grid = GaussianGrid.create(21)
    tables = SphericalGridTables.create(grid, xp=cupy, dtype=np.float64)
    errors, spacing = [], []

    def profile(t):
        return np.sin(3.0 * t) + 0.4 * np.cos(5.0 * t) - 0.2 * t ** 2

    for nlev in (20, 40, 80):
        shape = (nlev, grid.nlat, grid.nlon)
        z = np.arange(nlev, dtype=np.float64) / (nlev - 1)
        field = cupy.asarray(np.broadcast_to(
            profile(z)[:, None, None], shape).copy())
        target = np.clip(np.arange(nlev, dtype=np.float64) + 0.37,
                         0.0, nlev - 1)
        stencil = Stencil(
            xi=cupy.asarray(np.broadcast_to(
                np.arange(grid.nlon, dtype=np.float64)[None, None, :],
                shape).copy()),
            phi=cupy.asarray(np.broadcast_to(
                np.asarray(grid.lat_rad)[None, :, None], shape).copy()),
            level=cupy.asarray(np.broadcast_to(
                target[:, None, None], shape).copy()),
            tables=tables,
        )
        got = cupy.asnumpy(gather(field, stencil, monotone=False))[:, 0, 0]
        # the outermost two levels use the one-sided cubic through the
        # boundary, which is a different stencil and not what is graded here
        err = got[2:-2] - profile(target / (nlev - 1))[2:-2]
        errors.append(float(np.sqrt(np.mean(err ** 2))))
        spacing.append(1.0 / nlev)
    slopes = [math.log(errors[i] / errors[i + 1])
              / math.log(spacing[i] / spacing[i + 1])
              for i in range(len(errors) - 1)]
    assert min(slopes) > 3.5, (errors, slopes)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_deficit_kernel_matches_the_specification_and_the_plain_gather(
    cupy_module, dtype
):
    """The clip-deficit entry point, on the card.

    Two things at once, because they are the two ways it could be wrong.
    Its VALUES have to be the plain quasi-monotone gather's, bit for bit,
    or every arm run with the additive mass fixer would be measuring a
    different interpolation than the default arm runs.  Its DEFICIT has to
    be the numpy specification's, which is what says the mass the fixer
    puts back is the mass the limiter actually took.
    """
    cupy = cupy_module
    grid = GaussianGrid.create(31)
    nlev = 6
    shape = (nlev, grid.nlat, grid.nlon)
    rng = np.random.default_rng(90210)
    # A field like a condensate species: zero nearly everywhere, with
    # isolated maxima.  A smooth field barely clips at all and would let a
    # deficit that was always zero pass.
    fields = []
    for _ in range(3):
        field = np.zeros(shape, dtype=dtype)
        picks = rng.integers(0, field.size, 200)
        field.reshape(-1)[picks] = rng.uniform(0.1, 1.0, picks.size)
        fields.append(field)
    coords = {
        "xi": rng.uniform(-3.0, grid.nlon + 3.0, shape).astype(dtype),
        "phi": rng.uniform(-math.pi / 2, math.pi / 2, shape).astype(dtype),
        "level": rng.uniform(-0.5, nlev + 0.5, shape).astype(dtype),
    }

    host_tables = SphericalGridTables.create(grid, xp=np, dtype=dtype)
    host = gather_batch(
        fields, Stencil(tables=host_tables, **coords), monotone=True,
        deficit=True,
    )

    dev_tables = SphericalGridTables.create(grid, xp=cupy, dtype=dtype)
    dev_stencil = Stencil(
        tables=dev_tables,
        **{k: cupy.asarray(v) for k, v in coords.items()},
    )
    dev_fields = [cupy.asarray(f) for f in fields]
    plain = gather_batch(dev_fields, dev_stencil, monotone=True)
    device = gather_batch(dev_fields, dev_stencil, monotone=True, deficit=True)

    tol = 3e-6 if dtype == np.float32 else 1e-12
    clipped = 0
    for (host_value, host_cut), (value, cut), bare in zip(host, device, plain):
        assert bool((value == bare).all())
        assert float(np.max(np.abs(host_value - cupy.asnumpy(value)))) < tol
        assert float(np.max(np.abs(host_cut - cupy.asnumpy(cut)))) < tol
        clipped += int((cupy.asnumpy(cut) != 0.0).sum())
    assert clipped > 0
