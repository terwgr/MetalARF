"""Instrument gates for the departure-point solver and its diagnostics."""
from __future__ import annotations

import math

import numpy as np
import pytest

from woof.globe.spectral.grid import GaussianGrid
from woof.globe.semilag.cases import (
    CASE_PERIOD_S,
    _mesh,
    deformational_wind,
    solid_body_departure,
    solid_body_wind,
)
from woof.globe.semilag.tables import SphericalGridTables
from woof.globe.semilag.trajectory import (
    CartesianWind,
    LipschitzDiagnostics,
    TrajectoryDiagnostics,
    cartesian_wind,
    convergence,
    departure_points,
    level_rate_from_mass_flux,
    lipschitz,
    local_wind,
    refuse_beyond_lipschitz,
)

NLEV = 4


def _solid_body(truncation, alpha, *, xp=np, dtype=np.float64, u0=None):
    grid = GaussianGrid.create(truncation)
    tables = SphericalGridTables.create(grid, xp=xp, dtype=dtype)
    lam, phi = _mesh(grid)
    speed = (2.0 * math.pi * grid.radius_m / CASE_PERIOD_S if u0 is None
             else u0)
    u2, v2 = solid_body_wind(lam, phi, u0=speed, alpha=alpha)
    u = np.repeat(u2[None], NLEV, axis=0).astype(dtype)
    v = np.repeat(v2[None], NLEV, axis=0).astype(dtype)
    vx, vy, vz = cartesian_wind(xp.asarray(u), xp.asarray(v), tables,
                                xp=xp, dtype=dtype)
    wind = CartesianWind(
        xp.ascontiguousarray(vx), xp.ascontiguousarray(vy),
        xp.ascontiguousarray(vz), xp.zeros_like(vx),
    )
    return grid, tables, wind, speed


@pytest.mark.parametrize("alpha", [0.0, math.pi / 4.0, math.pi / 2.0])
def test_departure_matches_the_closed_form(alpha):
    """The solid-body trajectory has an exact answer; the solver finds it."""
    grid, tables, wind, u0 = _solid_body(21, alpha)
    dt = CASE_PERIOD_S / 256.0
    stencil, diag = departure_points(wind, tables, dt, iterations=3)
    lam, phi = _mesh(grid)
    exact_lam, exact_phi = solid_body_departure(
        lam, phi, u0=u0, alpha=alpha, dt=dt, radius=grid.radius_m)
    got_lam = np.asarray(stencil.xi)[0] * tables.dlam
    got_phi = np.asarray(stencil.phi)[0]
    dot = (np.sin(exact_phi) * np.sin(got_phi)
           + np.cos(exact_phi) * np.cos(got_phi) * np.cos(exact_lam - got_lam))
    err = grid.radius_m * np.arccos(np.clip(dot, -1.0, 1.0))
    # the displacement itself is 156 km at this step
    assert diag.displacement_max_m > 1.5e5
    assert float(err.max()) < 200.0
    # and the fixed point has converged far below the trajectory's own error
    assert diag.move_max_m < 1.0


def test_trajectory_error_is_bounded_by_the_wind_interpolation():
    """What actually limits the departure point, measured rather than assumed.

    The fixed point of the trapezoidal map with an EXACT wind is the Cayley
    transform of the rotation, whose error is dt^3 * omega^3 / 12 and is
    0.003 m at a 300 s step.  The measured error is forty times that, and
    the reason is the trilinear gather of the wind at the trial departure
    point: with a displacement much shorter than a cell, the fractional
    offset inside the cell is itself proportional to dt/h, so the linear
    interpolation error is O(dt*V*h) and the trajectory error is first order
    in the grid spacing at a fixed step, not second.  Linear interpolation
    for the trajectory wind is the standard choice and this is its price;
    the test pins both the size and the direction so a future cubic wind
    gather has something to be measured against.
    """
    errors = []
    for truncation in (21, 42, 84):
        grid, tables, wind, u0 = _solid_body(truncation, math.pi / 2.0)
        stencil, diag = departure_points(wind, tables, 300.0, iterations=4)
        lam, phi = _mesh(grid)
        exact_lam, exact_phi = solid_body_departure(
            lam, phi, u0=u0, alpha=math.pi / 2.0, dt=300.0,
            radius=grid.radius_m)
        got_lam = np.asarray(stencil.xi)[0] * tables.dlam
        got_phi = np.asarray(stencil.phi)[0]
        dot = (np.sin(exact_phi) * np.sin(got_phi)
               + np.cos(exact_phi) * np.cos(got_phi)
               * np.cos(exact_lam - got_lam))
        errors.append(float(grid.radius_m
                            * np.max(np.arccos(np.clip(dot, -1.0, 1.0)))))
        # the displacement itself is 11.6 km at this step
        assert diag.displacement_max_m == pytest.approx(1.158e4, rel=0.02)
    # first order in the grid spacing, and small against the displacement
    slopes = [math.log(errors[i] / errors[i + 1]) / math.log(2.0)
              for i in range(len(errors) - 1)]
    assert min(slopes) > 0.85, (errors, slopes)
    assert errors[-1] < 1.0
    assert errors[-1] / 1.158e4 < 1e-4


def test_more_iterations_do_not_move_the_converged_fixed_point():
    """Three iterations reach the fixed point; the residual says so."""
    _, tables, wind, _ = _solid_body(42, math.pi / 4.0)
    previous = None
    for iterations in (2, 3, 4, 6):
        stencil, diag = departure_points(
            wind, tables, 300.0, iterations=iterations)
        if previous is not None:
            assert float(np.max(np.abs(
                np.asarray(stencil.phi) - previous))) < 1e-9
        previous = np.asarray(stencil.phi).copy()
        if iterations >= 3:
            assert diag.move_max_cells < 1e-6


def test_zero_wind_departs_from_the_arrival_point():
    grid, tables, wind, _ = _solid_body(21, 0.0, u0=0.0)
    stencil, diag = departure_points(wind, tables, 300.0, iterations=3)
    index = np.broadcast_to(np.arange(grid.nlon, dtype=np.float64),
                            stencil.shape)
    lat = np.broadcast_to(np.asarray(grid.lat_rad)[None, :, None],
                          stencil.shape)
    level = np.broadcast_to(np.arange(NLEV, dtype=np.float64)[:, None, None],
                            stencil.shape)
    # the departure longitude comes back on the atan2 branch, so the index
    # is compared modulo the grid rather than on its principal value
    offset = (np.asarray(stencil.xi) - index + grid.nlon / 2.0) % grid.nlon
    assert float(np.max(np.abs(offset - grid.nlon / 2.0))) < 1e-9
    assert float(np.max(np.abs(np.asarray(stencil.phi) - lat))) < 1e-12
    assert np.array_equal(np.asarray(stencil.level), level)
    assert diag.displacement_max_m < 1e-6


@pytest.mark.parametrize("alpha", [0.0, math.pi / 2.0])
def test_lipschitz_of_a_rigid_rotation_is_the_rotation_rate(alpha):
    """A closed form the diagnostic has to reproduce, poles included.

    The velocity gradient tensor of a rigid rotation is antisymmetric with
    magnitude omega everywhere on the sphere, so the diagnostic must read
    exactly ``omega * dt`` and must read it at the poles too.  Building the
    Jacobian from the LOCAL east-north components instead of the Cartesian
    ones puts 0.037 per second into the outermost T255 ring for a flow that
    is not deforming at all, because the local basis turns through 2*pi
    across 326 m of zonal spacing there.  That is the failure this test
    exists to catch, and the ``alpha = pi/2`` arm runs the flow straight
    over both poles so it cannot be missed.
    """
    grid, tables, wind, u0 = _solid_body(31, alpha)
    dt = 300.0
    diag = lipschitz(wind, tables, dt)
    exact = dt * u0 / grid.radius_m
    assert diag.lipschitz == pytest.approx(exact, rel=0.02)
    assert diag.lipschitz_horizontal == pytest.approx(exact, rel=0.02)
    # the Frobenius norm of an antisymmetric 3x3 with one rate is sqrt(2)
    # times its spectral norm
    assert diag.lipschitz_frobenius == pytest.approx(
        math.sqrt(2.0) * exact, rel=0.05)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lipschitz_is_a_number_where_the_flow_is_degenerate(dtype):
    """No NaN from a point whose Jacobian is isotropic or zero.

    The closed-form spectrum of a symmetric 3x3 divides by the cube of the
    matrix's deviation from isotropy.  A purely zonal flow with no vertical
    motion has an exactly zero third row and column, and on a real float32
    Jacobian the deviation is 1e-6, whose cube underflows to exactly zero;
    the division then returned 0/0 and a single NaN made the whole
    reduction NaN.  A diagnostic that reports "not a number" for a flow with
    a perfectly good deformation is worse than one that reports nothing, so
    both precisions are pinned here.
    """
    _, tables, wind, u0 = _solid_body(31, 0.0, dtype=dtype)
    diag = lipschitz(wind, tables, 300.0)
    assert math.isfinite(diag.lipschitz)
    assert math.isfinite(diag.lipschitz_frobenius)
    assert math.isfinite(diag.lipschitz_horizontal)
    assert diag.lipschitz > 0.0

    # and a flow that is exactly zero everywhere reads exactly zero
    still = CartesianWind(*[np.zeros_like(np.asarray(a)) if dtype is None
                            else a * 0 for a in wind.arrays()])
    zero = lipschitz(still, tables, 300.0)
    assert zero.lipschitz == 0.0
    assert zero.lipschitz_frobenius == 0.0


def test_lipschitz_does_not_depend_on_its_chunking():
    """The level chunk is a memory choice, not an arithmetic one.

    The diagnostic walks the column in slabs because the full-volume form
    held about twenty-five grid arrays at once and ran a 16 GiB card out of
    memory at T533 L40.  A slab needs one level of halo for the vertical
    derivative, and a missing halo shows up as a wrong answer only on the
    two levels at each slab boundary, which is exactly the kind of error a
    single-chunk test would never see.
    """
    grid = GaussianGrid.create(31)
    tables = SphericalGridTables.create(grid, xp=np, dtype=np.float64)
    nlev = 12
    rng = np.random.default_rng(19)
    shape = (nlev, grid.nlat, grid.nlon)
    lam = np.asarray(grid.lon_rad)[None, None, :]
    phi = np.asarray(grid.lat_rad)[None, :, None]
    k = np.arange(nlev)[:, None, None] / (nlev - 1)
    u = np.broadcast_to(40.0 * np.cos(phi) * (0.2 + k)
                        * (1.0 + 0.3 * np.sin(2.0 * lam)), shape).copy()
    v = np.broadcast_to(9.0 * np.sin(3.0 * lam) * np.cos(phi) * (1.0 - k),
                        shape).copy()
    vx, vy, vz = cartesian_wind(u, v, tables, xp=np, dtype=np.float64)
    rate = np.broadcast_to(
        3.0e-4 * np.sin(math.pi * k) * (1.0 + 0.2 * np.cos(lam)), shape).copy()
    wind = CartesianWind(np.ascontiguousarray(vx), np.ascontiguousarray(vy),
                         np.ascontiguousarray(vz), rate)
    whole = lipschitz(wind, tables, 300.0, level_chunk=nlev)
    for chunk in (1, 2, 3, 5, 7):
        part = lipschitz(wind, tables, 300.0, level_chunk=chunk)
        assert part.lipschitz == pytest.approx(whole.lipschitz, rel=1e-12)
        assert part.lipschitz_frobenius == pytest.approx(
            whole.lipschitz_frobenius, rel=1e-12)
        assert part.lipschitz_horizontal == pytest.approx(
            whole.lipschitz_horizontal, rel=1e-12)
    # and the vertical rate is actually contributing, so the test is not
    # measuring a Jacobian whose third row is zero
    flat = CartesianWind(wind.vx, wind.vy, wind.vz, np.zeros_like(rate))
    assert lipschitz(flat, tables, 300.0).lipschitz < whole.lipschitz


def test_lipschitz_scales_with_the_time_step():
    _, tables, wind, _ = _solid_body(21, 0.0)
    a = lipschitz(wind, tables, 300.0)
    b = lipschitz(wind, tables, 600.0)
    assert b.lipschitz == pytest.approx(2.0 * a.lipschitz, rel=1e-12)
    assert a.jacobian_spectral_s == pytest.approx(b.jacobian_spectral_s,
                                                  rel=1e-12)


def test_level_rate_boundaries_are_zero():
    nlev = 6
    dp = np.full((nlev, 2, 4), 2500.0)
    omega = np.zeros((nlev + 1, 2, 4))
    omega[1:nlev] = 5.0
    rate = level_rate_from_mass_flux(omega, dp)
    assert rate.shape == (nlev, 2, 4)
    # interior interfaces give 5/2500 per interface, averaged onto levels
    assert rate[0] == pytest.approx(0.5 * 5.0 / 2500.0)
    assert rate[-1] == pytest.approx(0.5 * 5.0 / 2500.0)
    assert rate[2] == pytest.approx(5.0 / 2500.0)
    with pytest.raises(ValueError, match="interfaces"):
        level_rate_from_mass_flux(np.zeros((nlev, 2, 4)), dp)


def test_wind_round_trip():
    grid, tables, wind, _ = _solid_body(21, math.pi / 3.0)
    u, v = local_wind(wind.vx, wind.vy, wind.vz, tables, xp=np,
                      dtype=np.float64)
    vx, vy, vz = cartesian_wind(u, v, tables, xp=np, dtype=np.float64)
    assert float(np.max(np.abs(vx - wind.vx))) < 1e-9
    assert float(np.max(np.abs(vy - wind.vy))) < 1e-9
    assert float(np.max(np.abs(vz - wind.vz))) < 1e-9


def test_refusals_name_their_breakage():
    _, tables, wind, _ = _solid_body(21, 0.0)
    with pytest.raises(ValueError, match="at least one iteration"):
        departure_points(wind, tables, 300.0, iterations=0)

    diag = TrajectoryDiagnostics(
        iterations=3, dt_s=300.0, move_max_m=900.0, move_max_cells=0.02,
        move_max_levels=0.0, displacement_max_m=0.0,
        displacement_max_cells=0.0, displacement_max_levels=0.0,
    )
    with pytest.raises(ValueError, match="did not converge"):
        convergence(diag, 0.01)
    convergence(diag, 0.05)

    lip = LipschitzDiagnostics(
        dt_s=300.0, reference_length_m=52000.0, lipschitz=0.9,
        lipschitz_frobenius=1.2, lipschitz_horizontal=0.4,
        jacobian_spectral_s=0.003, jacobian_frobenius_s=0.004,
        jacobian_horizontal_s=0.001,
    )
    with pytest.raises(ValueError, match="Lipschitz number"):
        refuse_beyond_lipschitz(lip, 0.75)
    refuse_beyond_lipschitz(lip, 1.0)
    with pytest.raises(ValueError, match="not invertible at all"):
        refuse_beyond_lipschitz(lip, 1.5)


def test_the_departure_latitude_survives_float32_near_a_pole():
    """The reconstruction of latitude has to hold up in the run precision.

    Named breakage: reading the departure latitude as asin of the vertical
    component of a unit vector amplifies its last bit by 1/cos(phi), which
    is 80 on the outermost T127 ring and 156 at T255.  In float32, the
    production precision, that turned a resting atmosphere's departure point
    into a point 8 m away and put 127 m of error on the outermost T255 ring
    of a solid-body rotation, where the interior read 3 m.  Latitude is read
    by atan2 against the horizontal radius instead, which carries the same
    relative precision at every latitude.  The tolerance below is met by the
    atan2 form by two orders and failed by the asin form.
    """
    grid, tables, wind, _ = _solid_body(127, 0.0, dtype=np.float32, u0=0.0)
    stencil, diag = departure_points(wind, tables, 300.0, iterations=3)
    lat = np.broadcast_to(np.asarray(grid.lat_rad)[None, :, None],
                          stencil.shape)
    error_rad = float(np.max(np.abs(np.asarray(stencil.phi) - lat)))
    assert error_rad < 1e-6, error_rad
    # and it is the outermost rings that the asin form loses, so pin them
    outer = np.abs(np.asarray(stencil.phi)[:, [0, 1, -2, -1], :]
                   - lat[:, [0, 1, -2, -1], :])
    assert float(np.max(outer)) < 1e-6, float(np.max(outer))
    assert diag.displacement_max_m < 5.0


@pytest.mark.parametrize("t_over_period", [0.0, 0.25, 0.5])
def test_the_prescribed_fields_are_the_published_ones(t_over_period):
    """The KERN-2 flows are pinned to the published coefficients.

    Named breakage: a gate that calls itself a published case and carries
    different coefficients cannot be compared with a published number, and
    its convergence slope measures a different problem.  The divergent field
    was coded at the non-divergent field's amplitude and with no background
    rotation, which is twice the deformation of the published case; its
    measured convergence slope was 0.88 where the other three read 2.8 to
    3.8.  The closed forms below are written out here independently of the
    module so that a coefficient cannot drift on either side alone.
    """
    grid = GaussianGrid.create(21)
    lam, phi = _mesh(grid)
    period = CASE_PERIOD_S
    radius = grid.radius_m
    t = t_over_period * period
    taper = math.cos(math.pi * t / period)
    background = 2.0 * math.pi * radius / period
    shifted = lam - background * t / radius

    u, v = deformational_wind("nondivergent_two_cell_background", lam, phi, t,
                              radius=radius, period=period)
    u_pub = (10.0 * radius / period * np.sin(shifted) ** 2
             * np.sin(2.0 * phi) * taper + background * np.cos(phi))
    v_pub = (10.0 * radius / period * np.sin(2.0 * shifted)
             * np.cos(phi) * taper)
    assert np.allclose(u, u_pub, rtol=0, atol=1e-9)
    assert np.allclose(v, v_pub, rtol=0, atol=1e-9)

    u, v = deformational_wind("divergent", lam, phi, t,
                              radius=radius, period=period)
    u_pub = (-5.0 * radius / period * np.sin(0.5 * shifted) ** 2
             * np.sin(2.0 * phi) * np.cos(phi) ** 2 * taper
             + background * np.cos(phi))
    v_pub = (2.5 * radius / period * np.sin(shifted)
             * np.cos(phi) ** 3 * taper)
    assert np.allclose(u, u_pub, rtol=0, atol=1e-9)
    assert np.allclose(v, v_pub, rtol=0, atol=1e-9)
