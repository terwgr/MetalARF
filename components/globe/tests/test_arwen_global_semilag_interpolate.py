"""Instrument gates for the semi-Lagrangian interpolation kernels.

Every test here grades the interpolator against something it cannot fake: an
identity, a closed form, or a convergence rate.  Nothing in this file runs the
model and nothing in it opens a device: the tests that grade the compiled
kernel against this numpy specification live in
``test_arwen_global_semilag_cuda.py``, kept in a separate module because the
suite marks every module that imports CuPy as a device module and skips the
whole of it on a machine that may not open a card.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from woof.globe.spectral.grid import GaussianGrid
from woof.globe.semilag.interpolate import (
    Stencil,
    gather,
    zero_stencil,
)
from woof.globe.semilag.tables import SphericalGridTables


def _tables(truncation=21, *, xp=np, dtype=np.float64):
    grid = GaussianGrid.create(truncation)
    return grid, SphericalGridTables.create(grid, xp=xp, dtype=dtype)


def _smooth_field(grid, nlev, dtype=np.float64):
    lam = np.asarray(grid.lon_rad)[None, None, :]
    phi = np.asarray(grid.lat_rad)[None, :, None]
    k = np.arange(nlev)[:, None, None]
    return (np.sin(2.0 * lam) * np.cos(phi) ** 2
            + 0.5 * np.cos(3.0 * lam) * np.sin(phi)
            + 0.25 * np.cos(k * math.pi / max(nlev - 1, 1))
            ).astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("monotone", [False, True])
def test_zero_displacement_is_bit_exact(dtype, monotone):
    """A stencil that departs from its own arrival point returns the field.

    Bit for bit, over the WHOLE grid: the polar rows, where the stencil
    reaches across a pole into reflected data, and the top and bottom levels,
    where the vertical stencil start is clamped and the departure point sits
    at position 0 or 3 of four rather than in the middle.  Those are the
    three places a weight set is wrong without any other symptom, and an
    interior-only identity check passes while every one of them is broken.
    """
    grid, tables = _tables(dtype=dtype)
    nlev = 6
    stencil = zero_stencil(tables, nlev, xp=np, dtype=dtype)
    rng = np.random.default_rng(20260906)
    field = rng.standard_normal((nlev, grid.nlat, grid.nlon)).astype(dtype)
    out = gather(field, stencil, monotone=monotone)
    assert np.array_equal(out, field)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_constant_field_is_reproduced(dtype):
    """The three weight sets each sum to one, so a constant survives."""
    grid, tables = _tables(dtype=dtype)
    nlev = 8
    rng = np.random.default_rng(11)
    shape = (nlev, grid.nlat, grid.nlon)
    stencil = Stencil(
        xi=(rng.uniform(-5.0, grid.nlon + 5.0, shape)).astype(dtype),
        phi=(rng.uniform(-math.pi / 2, math.pi / 2, shape)).astype(dtype),
        level=(rng.uniform(-1.0, nlev, shape)).astype(dtype),
        tables=tables,
    )
    field = np.full(shape, 3.25, dtype=dtype)
    out = gather(field, stencil, monotone=False)
    tol = 8.0 * np.finfo(dtype).eps * 3.25
    assert float(np.max(np.abs(out - 3.25))) <= tol


def test_meridional_weights_are_exact_for_a_cubic():
    """A cubic in latitude is interpolated exactly on the true nodes.

    This is the test that fails if the meridional weights are built as if
    the Gauss-Legendre nodes were equispaced.  It is deliberately a cubic:
    a four-point Lagrange stencil reproduces every cubic exactly on ITS OWN
    nodes and no cubic at all on somebody else's.
    """
    grid, tables = _tables(41)
    nlev = 4
    phi = np.asarray(grid.lat_rad)

    def cubic(x):
        return 1.0 - 0.7 * x + 0.3 * x ** 2 + 0.11 * x ** 3

    field = np.broadcast_to(
        cubic(phi)[None, :, None], (nlev, grid.nlat, grid.nlon)
    ).copy()
    rng = np.random.default_rng(3)
    target = rng.uniform(phi[2], phi[-3], (nlev, grid.nlat, grid.nlon))
    stencil = Stencil(
        xi=np.broadcast_to(np.arange(grid.nlon, dtype=np.float64),
                           (nlev, grid.nlat, grid.nlon)).copy(),
        phi=target,
        level=np.broadcast_to(np.arange(nlev, dtype=np.float64)[:, None, None],
                              (nlev, grid.nlat, grid.nlon)).copy(),
        tables=tables,
    )
    out = gather(field, stencil, monotone=False)
    assert float(np.max(np.abs(out - cubic(target)))) < 1e-12


def test_pole_row_mapping():
    """The extended rows read the rings they reflect, half a grid around."""
    grid, tables = _tables(21)
    n = grid.nlon
    off = tables.rowoff_host // n
    shift = tables.rowshift_host
    # ext rows -2, -1 are rings 1 and 0 reflected through the south pole
    assert list(off[:4]) == [1, 0, 0, 1]
    assert list(shift[:4]) == [n // 2, n // 2, 0, 0]
    # and the same at the north
    assert list(off[-4:]) == [grid.nlat - 2, grid.nlat - 1,
                              grid.nlat - 1, grid.nlat - 2]
    assert list(shift[-4:]) == [0, 0, n // 2, n // 2]
    # the extended latitudes are the reflections, and strictly ascending
    assert tables.lat_ext_host[1] == pytest.approx(-math.pi - grid.lat_rad[0])
    assert tables.lat_ext_host[-2] == pytest.approx(
        math.pi - grid.lat_rad[-1])
    assert np.all(np.diff(tables.lat_ext_host) > 0.0)


def test_polar_cap_matches_an_independent_extended_interpolation():
    """A departure point inside the polar cap, graded against a hand build.

    The reference is assembled here rather than read from the module: the
    four rings the stencil needs are written out explicitly, the two that
    lie across the pole are rolled by half the grid by hand, and the cubic
    is evaluated on the true node latitudes with numpy's own Lagrange.  A
    reflection that forgot the roll, or rolled the wrong ring, disagrees.
    """
    grid, tables = _tables(31)
    nlev, n = 4, grid.nlon
    lam = np.asarray(grid.lon_rad)[None, :]
    phi = np.asarray(grid.lat_rad)[:, None]
    plane = (2.0 + np.sin(phi) + np.cos(lam) * np.cos(phi)
             + np.sin(2.0 * lam) * np.cos(phi) ** 2)
    field = np.broadcast_to(plane[None], (nlev, grid.nlat, n)).copy()

    # halfway between the south pole and the outermost ring
    target = 0.5 * (-math.pi / 2.0 + grid.lat_rad[0])
    shape = (nlev, grid.nlat, n)
    stencil = Stencil(
        xi=np.broadcast_to(np.arange(n, dtype=np.float64), shape).copy(),
        phi=np.full(shape, target),
        level=np.broadcast_to(np.arange(nlev, dtype=np.float64)[:, None, None],
                              shape).copy(),
        tables=tables,
    )
    out = gather(field, stencil, monotone=False)

    rings = [np.roll(plane[1], n // 2), np.roll(plane[0], n // 2),
             plane[0], plane[1]]
    nodes = tables.lat_ext_host[0:4]
    expected = np.zeros(n)
    for m in range(4):
        w = 1.0
        for q in range(4):
            if q != m:
                w *= (target - nodes[q]) / (nodes[m] - nodes[q])
        expected = expected + w * rings[m]
    assert np.allclose(out[0, 0], expected, atol=1e-12)
    assert np.allclose(out[3, 7], expected, atol=1e-12)


def test_the_pole_is_single_valued():
    """Every longitude reads the same value at the pole, and the right one.

    The field carries zonal wavenumbers zero and one only.  That is not a
    convenience: a latitude-only stencil cannot make a wavenumber-two field
    single-valued at a pole, because the two rings it reflects together
    carry the same phase of ``sin(2*lambda)`` rather than opposite ones, and
    what survives is the field's own value on the ring, which vanishes with
    ``cos(phi)^2`` and so is a fourth-order remainder rather than an error.
    Wavenumber one is the case that has to be exact, because it is the
    leading polar behaviour of every smooth field and of every Cartesian
    wind component.
    """
    grid, tables = _tables(63)
    nlev, n = 4, grid.nlon
    lam = np.asarray(grid.lon_rad)[None, :]
    phi = np.asarray(grid.lat_rad)[:, None]
    plane = 2.0 + np.sin(phi) + np.cos(lam) * np.cos(phi)
    field = np.broadcast_to(plane[None], (nlev, grid.nlat, n)).copy()
    shape = (nlev, grid.nlat, n)
    stencil = Stencil(
        xi=np.broadcast_to(np.arange(n, dtype=np.float64), shape).copy(),
        phi=np.full(shape, -math.pi / 2.0),
        level=np.broadcast_to(np.arange(nlev, dtype=np.float64)[:, None, None],
                              shape).copy(),
        tables=tables,
    )
    out = gather(field, stencil, monotone=False)
    at_pole = out[0, 0]
    assert float(np.max(at_pole) - np.min(at_pole)) < 1e-12
    assert float(np.mean(at_pole)) == pytest.approx(1.0, abs=1e-5)


def test_quasi_monotone_result_stays_inside_its_cell():
    """With the clip on, every value lies inside the surrounding 2x2x2 box.

    Including at the vertical boundaries, where the stencil start is clamped
    and the box therefore is NOT the middle pair of the four taps.  A clip
    that always used the middle pair would clamp a lid-level departure point
    to a box that does not contain it, which is a monotone limiter that
    breaks the identity it exists to protect.
    """
    grid, tables = _tables(21)
    nlev = 6
    rng = np.random.default_rng(7)
    shape = (nlev, grid.nlat, grid.nlon)
    field = rng.standard_normal(shape)
    stencil = Stencil(
        xi=rng.uniform(0.0, grid.nlon, shape),
        phi=rng.uniform(-math.pi / 2, math.pi / 2, shape),
        level=rng.uniform(0.0, nlev - 1, shape),
        tables=tables,
    )
    out = gather(field, stencil, monotone=True)
    plain = gather(field, stencil, monotone=False)

    # rebuild the bounding box independently of the interpolator
    xi = np.clip(np.asarray(stencil.xi), 0.0, grid.nlon)
    i0 = np.floor(xi).astype(int)
    lat_ext = tables.lat_ext_host
    m0 = np.clip(np.searchsorted(lat_ext, np.asarray(stencil.phi),
                                 side="right") - 1, 1, grid.nlat + 1)
    kb = np.clip(np.floor(np.asarray(stencil.level)).astype(int), 0, nlev - 2)
    lo = np.full(shape, np.inf)
    hi = np.full(shape, -np.inf)
    for dk in (0, 1):
        for dj in (0, 1):
            rows = tables.rowoff_host[m0 + dj] // grid.nlon
            shift = tables.rowshift_host[m0 + dj] > 0
            for di in (0, 1):
                col = (i0 + di) % grid.nlon
                col = np.where(shift, (col + grid.nlon // 2) % grid.nlon, col)
                vals = field[np.clip(kb + dk, 0, nlev - 1), rows, col]
                lo = np.minimum(lo, vals)
                hi = np.maximum(hi, vals)
    assert np.all(out >= lo - 1e-12)
    assert np.all(out <= hi + 1e-12)
    # and the clip actually did something, so the test is not vacuous
    assert float(np.max(np.abs(out - plain))) > 0.0


def test_convergence_is_fourth_order_in_the_interior():
    """Refining the grid at a fixed displacement drops the error like h^4."""
    errors = []
    spacings = []
    for truncation in (31, 63, 127):
        grid, tables = _tables(truncation)
        nlev = 4
        field = _smooth_field(grid, nlev)
        shape = (nlev, grid.nlat, grid.nlon)
        # half a cell east, a quarter cell north, in the tropics only, so the
        # measurement is of the stencil and not of the polar geometry
        xi = np.broadcast_to(
            np.arange(grid.nlon, dtype=np.float64) + 0.5, shape).copy()
        lat = np.asarray(grid.lat_rad)
        dphi = np.gradient(lat)
        phi = np.broadcast_to((lat + 0.25 * dphi)[None, :, None], shape).copy()
        level = np.broadcast_to(
            np.arange(nlev, dtype=np.float64)[:, None, None], shape).copy()
        stencil = Stencil(xi=xi, phi=phi, level=level, tables=tables)
        out = gather(field, stencil, monotone=False)

        lam_t = (np.arange(grid.nlon) + 0.5) * tables.dlam
        phi_t = lat + 0.25 * dphi
        exact = (np.sin(2.0 * lam_t)[None, None, :]
                 * np.cos(phi_t)[None, :, None] ** 2
                 + 0.5 * np.cos(3.0 * lam_t)[None, None, :]
                 * np.sin(phi_t)[None, :, None]
                 + 0.25 * np.cos(np.arange(nlev)
                                 * math.pi / (nlev - 1))[:, None, None])
        band = np.abs(phi_t) < 1.0
        err = np.abs(out[:, band, :] - exact[:, band, :])
        errors.append(float(np.sqrt(np.mean(err ** 2))))
        spacings.append(1.0 / grid.nlon)

    slopes = [
        math.log(errors[i] / errors[i + 1])
        / math.log(spacings[i] / spacings[i + 1])
        for i in range(len(errors) - 1)
    ]
    assert min(slopes) > 3.0, (errors, slopes)


def test_grid_refusals_name_their_breakage():
    grid = GaussianGrid.create(21)
    with pytest.raises(ValueError, match="strictly ascending"):
        SphericalGridTables.create(
            type(grid)(**{**grid.__dict__, "lat_rad": grid.lat_rad[::-1]}),
            xp=np, dtype=np.float64,
        )
    with pytest.raises(ValueError, match="even"):
        SphericalGridTables.create(
            type(grid)(**{**grid.__dict__, "nlon": grid.nlon - 1,
                          "lon_rad": grid.lon_rad[:-1]}),
            xp=np, dtype=np.float64,
        )


def test_stencil_refusals():
    grid, tables = _tables(21)
    shape = (4, grid.nlat, grid.nlon)
    good = np.zeros(shape)
    with pytest.raises(ValueError, match="nlev >= 4"):
        Stencil(xi=np.zeros((3, grid.nlat, grid.nlon)),
                phi=np.zeros((3, grid.nlat, grid.nlon)),
                level=np.zeros((3, grid.nlat, grid.nlon)), tables=tables)
    with pytest.raises(ValueError, match="where xi is"):
        Stencil(xi=good, phi=np.zeros((4, grid.nlat, grid.nlon - 2)),
                level=good, tables=tables)
    stencil = Stencil(xi=good, phi=good, level=good, tables=tables)
    with pytest.raises(ValueError, match="float32"):
        gather(np.zeros(shape, dtype=np.float32), stencil)
