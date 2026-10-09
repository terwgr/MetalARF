"""The quintic-horizontal gather: its tables, its numpy specification and
the door that selects it.

The compiled kernel is graded against the specification in
``test_arwen_global_semilag_cuda.py`` (the device module); everything here
runs on a CPU-only box.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from woof.globe.spectral.grid import GaussianGrid
from woof.globe.semilag.interpolate import (
    ORDERS,
    Stencil,
    gather,
    gather_batch,
    zero_stencil,
)
from woof.globe.semilag.options import (
    HORIZONTAL_INTERPOLATIONS,
    HORIZONTAL_ORDER,
    SemiLagrangianOptions,
)
from woof.globe.semilag.tables import SphericalGridTables

# CPU only: the device switch is set for each test in this module by
# `conftest._cpu_only_marked_tests` and put back afterwards, because a
# module that set it at import time decided it for the whole session.
pytestmark = pytest.mark.cpu_only


def _tables(truncation=21, *, xp=np, dtype=np.float64):
    grid = GaussianGrid.create(truncation)
    return grid, SphericalGridTables.create(grid, xp=xp, dtype=dtype)


def _smooth_field(grid, nlev, dtype=np.float64):
    lam = np.asarray(grid.lon_rad)[None, :]
    phi = np.asarray(grid.lat_rad)[:, None]
    plane = np.sin(2.0 * lam) * np.cos(phi) ** 2 + 0.5 * np.cos(3.0 * lam) * np.sin(phi)
    levels = 0.25 * np.cos(np.arange(nlev) * math.pi / (nlev - 1))
    return (plane[None] + levels[:, None, None]).astype(dtype)


def test_the_orders_are_four_and_six():
    assert ORDERS == (4, 6)
    assert HORIZONTAL_INTERPOLATIONS == ("cubic_lagrange", "quintic_lagrange")
    assert HORIZONTAL_ORDER == {"cubic_lagrange": 4, "quintic_lagrange": 6}


def test_the_six_point_tables_reflect_three_rows_through_each_pole():
    grid, tables = _tables(31)
    nlat, nlon = tables.shape
    lat = np.asarray(grid.lat_rad)
    ext = tables.lat_ext6_host
    assert ext.shape == (nlat + 6,)
    assert np.array_equal(ext[3:nlat + 3], lat)
    for g in range(3):
        assert ext[2 - g] == pytest.approx(-math.pi - lat[g])
        assert ext[nlat + 3 + g] == pytest.approx(math.pi - lat[nlat - 1 - g])
    assert np.all(np.diff(ext) > 0.0)
    # the reflected rows read the data rows they mirror, half a grid around
    rowoff = tables.rowoff6_host // nlon
    shift = tables.rowshift6_host
    assert list(rowoff[:3]) == [2, 1, 0]
    assert list(rowoff[nlat + 3:]) == [nlat - 1, nlat - 2, nlat - 3]
    assert np.all(shift[:3] == nlon // 2) and np.all(shift[nlat + 3:] == nlon // 2)
    assert np.all(shift[3:nlat + 3] == 0)
    assert np.array_equal(rowoff[3:nlat + 3], np.arange(nlat))
    # the reciprocal denominators are those of the six-node Lagrange basis
    assert tables.mrden6_host.shape == (nlat + 1, 6)
    sst = 7
    nodes = ext[sst:sst + 6]
    for m in range(6):
        den = 1.0
        for q in range(6):
            if q != m:
                den *= nodes[m] - nodes[q]
        assert tables.mrden6_host[sst, m] == pytest.approx(1.0 / den, rel=1e-12)


def test_the_cubic_tables_are_untouched_by_the_six_point_ones():
    """Adding the second table set moves nothing the cubic gather reads."""
    grid, tables = _tables(31)
    nlat, nlon = tables.shape
    lat = np.asarray(grid.lat_rad)
    assert tables.lat_ext_host.shape == (nlat + 4,)
    assert tables.lat_ext_host[0] == pytest.approx(-math.pi - lat[1])
    assert tables.lat_ext_host[1] == pytest.approx(-math.pi - lat[0])
    assert tables.mrden_host.shape == (nlat + 1, 4)
    assert tables.lookup_bins == 4 * (nlat + 4)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("monotone", [False, True])
def test_zero_displacement_is_bit_exact_at_order_six(dtype, monotone):
    grid, tables = _tables(31, dtype=dtype)
    nlev = 6
    stencil = zero_stencil(tables, nlev, xp=np, dtype=dtype)
    rng = np.random.default_rng(7)
    field = rng.standard_normal((nlev, *grid.shape)).astype(dtype)
    out = gather(field, stencil, monotone=monotone, order=6)
    assert np.array_equal(out, field)


def test_a_constant_field_is_reproduced_at_order_six():
    grid, tables = _tables(31)
    nlev = 5
    shape = (nlev, grid.nlat, grid.nlon)
    rng = np.random.default_rng(11)
    stencil = Stencil(
        xi=rng.uniform(-3.0, grid.nlon + 3.0, shape),
        phi=rng.uniform(-math.pi / 2.0, math.pi / 2.0, shape),
        level=rng.uniform(0.0, nlev - 1.0, shape),
        tables=tables,
    )
    field = np.full(shape, 3.25)
    out = gather(field, stencil, monotone=False, order=6)
    assert np.allclose(out, 3.25, atol=1e-12)


def test_convergence_is_sixth_order_in_the_interior():
    """Refining the grid at a fixed displacement drops the error like h^6,
    against the cubic gather's h^4 on the same points."""
    errors = {4: [], 6: []}
    spacings = []
    for truncation in (31, 63, 127):
        grid, tables = _tables(truncation)
        nlev = 4
        field = _smooth_field(grid, nlev)
        shape = (nlev, grid.nlat, grid.nlon)
        xi = np.broadcast_to(
            np.arange(grid.nlon, dtype=np.float64) + 0.5, shape).copy()
        lat = np.asarray(grid.lat_rad)
        dphi = np.gradient(lat)
        phi = np.broadcast_to((lat + 0.25 * dphi)[None, :, None], shape).copy()
        level = np.broadcast_to(
            np.arange(nlev, dtype=np.float64)[:, None, None], shape).copy()
        stencil = Stencil(xi=xi, phi=phi, level=level, tables=tables)
        lam_t = (np.arange(grid.nlon) + 0.5) * tables.dlam
        phi_t = lat + 0.25 * dphi
        exact = (np.sin(2.0 * lam_t)[None, None, :]
                 * np.cos(phi_t)[None, :, None] ** 2
                 + 0.5 * np.cos(3.0 * lam_t)[None, None, :]
                 * np.sin(phi_t)[None, :, None]
                 + 0.25 * np.cos(np.arange(nlev)
                                 * math.pi / (nlev - 1))[:, None, None])
        band = np.abs(phi_t) < 1.0
        for order in (4, 6):
            out = gather(field, stencil, monotone=False, order=order)
            err = np.abs(out[:, band, :] - exact[:, band, :])
            errors[order].append(float(np.sqrt(np.mean(err ** 2))))
        spacings.append(1.0 / grid.nlon)

    def slopes(errs):
        return [
            math.log(errs[i] / errs[i + 1]) / math.log(spacings[i] / spacings[i + 1])
            for i in range(len(errs) - 1)
        ]

    assert min(slopes(errors[6])) > 5.0, (errors, slopes(errors[6]))
    # and at every resolution the quintic is the more accurate of the two
    assert all(e6 < e4 for e4, e6 in zip(errors[4], errors[6])), errors


def test_polar_cap_matches_an_independent_six_ring_interpolation():
    """A departure point inside the polar cap, graded against a hand build
    of the six rings: three across the pole, rolled by half the grid."""
    grid, tables = _tables(31)
    nlev, n = 4, grid.nlon
    lam = np.asarray(grid.lon_rad)[None, :]
    phi = np.asarray(grid.lat_rad)[:, None]
    plane = (2.0 + np.sin(phi) + np.cos(lam) * np.cos(phi)
             + np.sin(2.0 * lam) * np.cos(phi) ** 2)
    field = np.broadcast_to(plane[None], (nlev, grid.nlat, n)).copy()
    target = 0.5 * (-math.pi / 2.0 + grid.lat_rad[0])
    shape = (nlev, grid.nlat, n)
    stencil = Stencil(
        xi=np.broadcast_to(np.arange(n, dtype=np.float64), shape).copy(),
        phi=np.full(shape, target),
        level=np.broadcast_to(np.arange(nlev, dtype=np.float64)[:, None, None],
                              shape).copy(),
        tables=tables,
    )
    out = gather(field, stencil, monotone=False, order=6)
    rings = [np.roll(plane[2], n // 2), np.roll(plane[1], n // 2),
             np.roll(plane[0], n // 2), plane[0], plane[1], plane[2]]
    nodes = tables.lat_ext6_host[0:6]
    expected = np.zeros(n)
    for m in range(6):
        w = 1.0
        for q in range(6):
            if q != m:
                w *= (target - nodes[q]) / (nodes[m] - nodes[q])
        expected = expected + w * rings[m]
    assert np.allclose(out[0, 0], expected, atol=1e-12)
    assert np.allclose(out[3, 7], expected, atol=1e-12)


def test_the_pole_is_single_valued_at_order_six():
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
    out = gather(field, stencil, monotone=False, order=6)
    at_pole = out[0, 0]
    assert float(np.max(at_pole) - np.min(at_pole)) < 1e-12
    assert float(np.mean(at_pole)) == pytest.approx(1.0, abs=1e-6)


def test_the_quasi_monotone_cell_is_the_same_inner_cell_at_both_orders():
    """The limiter bounds by the cell around the point, whichever stencil
    width computed the raw value, so a limited quintic never leaves the
    bounds the limited cubic holds."""
    grid, tables = _tables(31)
    nlev = 5
    shape = (nlev, grid.nlat, grid.nlon)
    rng = np.random.default_rng(3)
    stencil = Stencil(
        xi=rng.uniform(0.0, grid.nlon, shape),
        phi=rng.uniform(-1.2, 1.2, shape),
        level=rng.uniform(0.0, nlev - 1.0, shape),
        tables=tables,
    )
    field = np.where(rng.uniform(size=shape) > 0.7, 1.0, 0.0)
    raw6 = gather(field, stencil, monotone=False, order=6)
    kept6 = gather(field, stencil, monotone=True, order=6)
    kept4 = gather(field, stencil, monotone=True, order=4)
    assert raw6.min() < 0.0 and raw6.max() > 1.0
    assert kept6.min() >= 0.0 and kept6.max() <= 1.0
    assert kept4.min() >= 0.0 and kept4.max() <= 1.0


def test_the_quintic_refuses_a_grid_without_six_point_tables():
    """A grid too small for six rows keeps its cubic tables and the quintic
    gather names the missing set instead of reading the four-point one."""
    grid, built = _tables(21)
    tables = dataclasses.replace(
        built, lat_ext6=None, mrden6=None, mlut6=None, rowoff6=None,
        rowshift6=None, lookup_bins6=0, lut_scale6=0.0,
    )
    stencil = zero_stencil(tables, 4, xp=np, dtype=np.float64)
    field = np.zeros((4, *grid.shape))
    assert gather(field, stencil, monotone=False, order=4).shape == field.shape
    with pytest.raises(ValueError, match="six meridional rows"):
        gather(field, stencil, monotone=False, order=6)


def test_the_deficit_is_the_cubic_gathers_alone():
    grid, tables = _tables(31)
    stencil = zero_stencil(tables, 4, xp=np, dtype=np.float64)
    field = np.zeros((4, *grid.shape))
    with pytest.raises(ValueError, match="cubic gather only"):
        gather_batch([field], stencil, monotone=True, deficit=True, order=6)
    with pytest.raises(ValueError, match="horizontal order"):
        gather_batch([field], stencil, monotone=False, order=5)


def test_the_door_names_the_quintic_and_carries_it_in_the_identity():
    cubic = SemiLagrangianOptions(horizontal_interpolation="cubic_lagrange")
    quintic = SemiLagrangianOptions()
    # the six-point gather is the default since the dry ladder (2026-09-07)
    assert quintic.horizontal_interpolation == "quintic_lagrange"
    assert cubic.identity["horizontal_interpolation"] == "cubic_lagrange"
    assert quintic.identity["horizontal_interpolation"] == "quintic_lagrange"
    assert cubic.identity != quintic.identity
    with pytest.raises(ValueError, match="horizontal_interpolation"):
        SemiLagrangianOptions(horizontal_interpolation="septic_lagrange")


# ---------------------------------------------------------------------------
# The physics sub-step instrument


def test_physics_substeps_default_is_one_and_joins_the_identity():
    options = SemiLagrangianOptions()
    assert options.physics_substeps == 1
    assert options.identity["physics_substeps"] == 1
    five = SemiLagrangianOptions(physics_substeps=5)
    assert five.identity["physics_substeps"] == 5
    assert five.identity != options.identity


@pytest.mark.parametrize("value", [0, -1, 17])
def test_physics_substeps_outside_one_to_sixteen_is_refused(value):
    with pytest.raises(ValueError, match="physics_substeps"):
        SemiLagrangianOptions(physics_substeps=value)


def test_physics_substeps_table_parse_refuses_a_non_integer():
    from woof.globe.semilag.options import semilag_options_from_table

    with pytest.raises(ValueError, match="integer"):
        semilag_options_from_table({"physics_substeps": 2.5})
    assert semilag_options_from_table({"physics_substeps": 3}).physics_substeps == 3


def test_the_sub_stepped_half_is_the_same_operator_on_the_sponge_only_pass():
    """With no suite loaded a physics half is the lid absorber alone, whose
    exact-integral form makes n calls of dt/n equal one call of dt to
    roundoff; the loop is therefore graded on the arithmetic it wraps and
    on the record it carries, and an Eulerian model never enters it."""
    import dataclasses
    from woof.globe.vertical import HybridCoordinate
    from test_arwen_global_semilag_step import _baroclinic, _model, _transform

    transform = _transform(21)
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    base = _model(transform, vertical,
                  options=SemiLagrangianOptions(physics_substeps=4))
    model = dataclasses.replace(base, sponge_base_pa=5000.0)
    bundle = _baroclinic(transform, vertical)
    one, record_one = model.apply_physics(bundle, 300.0)
    four, record_four = model._apply_physics_half(bundle, 300.0)
    assert record_one == {"physics_mode": "none"}
    assert record_four["physics_substeps"] == 4
    moved = False
    for name in ("vorticity", "divergence"):
        before = np.asarray(getattr(bundle.atmosphere, name))
        a = np.asarray(getattr(one.atmosphere, name))
        b = np.asarray(getattr(four.atmosphere, name))
        moved = moved or bool(np.max(np.abs(a - before)) > 0.0)
        assert np.max(np.abs(a - b)) <= 1e-9 * max(1.0, float(np.max(np.abs(a))))
    assert moved, "the absorber did nothing, so the gate read nothing"
    eulerian = dataclasses.replace(model, integrator="imex_ssp3",
                                   semilag=SemiLagrangianOptions())
    assert eulerian._apply_physics_half(bundle, 300.0)[1] == {"physics_mode": "none"}
