"""The stacked tracer sweeps return the first version's bits.

CPU only: the stacked numpy specification of every sweep is compared,
exact bits, against a verbatim copy of the per-tracer loop it replaced,
and the ``fused`` flag is shown inert on the numpy backend.  The device
half of the proof (the fused cupy kernels against this specification)
lives in ``test_arwen_global_transport_fused.py``, whose every test needs
a card; this module names no device library so its tests run on the CPU
test host.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from woof.globe.transport import GridTracerTransport, _idx
from woof.globe.vertical import HybridCoordinate
from woof.globe.spectral.transform import SphericalHarmonicTransform


# -- the first version's per-tracer sweeps, verbatim, as the reference ----

def _harmonic(left, right, xp):
    product = left * right
    monotone = product > 0.0
    denominator = xp.where(monotone, left + right, 1.0)
    return xp.where(monotone, 2.0 * product / denominator, 0.0)


def _reference_periodic_step(mass, density, psi, dt, xp):
    """``GridTracerTransport._periodic_step`` as first written (per
    tracer, a dict of masses)."""
    flux_dt = psi * dt
    density_next = xp.roll(density, -1, axis=-1)
    c_lo = flux_dt / density
    c_hi = -flux_dt / density_next
    upwind_lo = psi >= 0.0
    w_lo = 0.5 * (1.0 - c_lo)
    w_hi = -0.5 * (1.0 - c_hi)
    new_density = density - (flux_dt - xp.roll(flux_dt, 1, axis=-1))
    new_mass = {}
    for name, value in mass.items():
        q = value / density
        q_next = xp.roll(q, -1, axis=-1)
        dq = q_next - q
        slope = _harmonic(xp.roll(dq, 1, axis=-1), dq, xp)
        slope_next = xp.roll(slope, -1, axis=-1)
        face_value = xp.where(
            upwind_lo, q + slope * w_lo, q_next + slope_next * w_hi
        )
        q_up = xp.where(upwind_lo, q, q_next)
        face_value = xp.minimum(
            xp.maximum(face_value, xp.minimum(q, q_next)),
            xp.minimum(xp.maximum(q, q_next), 2.0 * q_up),
        )
        tracer_flux = flux_dt * face_value
        new_mass[name] = value - (
            tracer_flux - xp.roll(tracer_flux, 1, axis=-1)
        )
    return new_density, new_mass


def _reference_bounded_step(mass, density, psi, sub, *, axis, centre, face, inv_width, xp):
    """One sub-step of ``GridTracerTransport._bounded`` as first written."""
    lo_cells = _idx(axis, slice(None, -1))
    hi_cells = _idx(axis, slice(1, None))
    interior = _idx(axis, slice(1, -1))
    d_face = face[interior]
    flux_dt = psi * sub
    c_lo = flux_dt * inv_width[lo_cells] / density[lo_cells]
    c_hi = -flux_dt * inv_width[hi_cells] / density[hi_cells]
    upwind_lo = psi >= 0.0
    w_lo = (d_face - centre[lo_cells]) * (1.0 - c_lo)
    w_hi = (d_face - centre[hi_cells]) * (1.0 - c_hi)
    new_density = density.copy()
    new_density[lo_cells] -= flux_dt * inv_width[lo_cells]
    new_density[hi_cells] += flux_dt * inv_width[hi_cells]
    out = {}
    for name, value in mass.items():
        q = value / density
        dq = (q[hi_cells] - q[lo_cells]) / (
            centre[hi_cells] - centre[lo_cells]
        )
        slope = xp.zeros_like(q)
        slope[interior] = _harmonic(
            dq[_idx(axis, slice(None, -1))],
            dq[_idx(axis, slice(1, None))], xp,
        )
        q_lo = q[lo_cells]
        q_hi = q[hi_cells]
        face_value = xp.where(
            upwind_lo,
            q_lo + slope[lo_cells] * w_lo,
            q_hi + slope[hi_cells] * w_hi,
        )
        q_up = xp.where(upwind_lo, q_lo, q_hi)
        face_value = xp.minimum(
            xp.maximum(face_value, xp.minimum(q_lo, q_hi)),
            xp.minimum(xp.maximum(q_lo, q_hi), 2.0 * q_up),
        )
        tracer_flux = flux_dt * face_value
        updated = value.copy()
        updated[lo_cells] -= tracer_flux * inv_width[lo_cells]
        updated[hi_cells] += tracer_flux * inv_width[hi_cells]
        out[name] = updated
    return new_density, out


# -- fields ---------------------------------------------------------------

def _transport(T=21, nlev=6, *, backend="numpy", precision="float32", fused=True):
    transform = SphericalHarmonicTransform.create(
        T, backend=backend, precision=precision, dealias_factor=1.5,
    )
    vertical = HybridCoordinate.pressure_blend(nlev)
    return transform, vertical, GridTracerTransport(transform, vertical, fused=fused)


def _fields(transform, vertical, seed, *, family):
    """Tracers, density and fluxes of one synthetic family, float32 host
    arrays."""
    rng = np.random.default_rng(seed)
    nlat, nlon = transform.grid.shape
    nlev = vertical.nlev
    shape = (nlev, nlat, nlon)
    ps = 1.0e5 * (1.0 + 0.05 * rng.standard_normal((nlat, nlon)))
    dp = np.asarray(
        transform.backend.to_numpy(
            vertical.pressure(np.asarray(ps), transform.backend)["dp"]
        ), dtype=np.float32,
    )
    lat = transform.grid.lat_rad[:, None]
    lon = transform.grid.lon_rad[None, :]
    if family == "random":
        tracers = {
            name: rng.random(shape, dtype=np.float32) * np.float32(1.0e-3)
            for name in ("qc", "qr", "qi", "nc", "nr")
        }
        u = 60.0 * rng.standard_normal(shape)
        v = 40.0 * rng.standard_normal(shape)
        omega = np.zeros((nlev + 1, nlat, nlon))
        omega[1:-1] = 2.0 * rng.standard_normal((nlev - 1, nlat, nlon))
    elif family == "spikes":
        tracers = {name: np.zeros(shape, np.float32) for name in ("qc", "qr", "ns")}
        for name in tracers:
            for _ in range(12):
                k, j, i = rng.integers(nlev), rng.integers(nlat), rng.integers(nlon)
                tracers[name][k, j, i] = np.float32(rng.random() * 5.0e-3)
        u = 80.0 * np.cos(lat) * np.ones(shape)
        v = 30.0 * np.sin(2.0 * lon) * np.cos(lat) * np.ones(shape)
        omega = np.zeros((nlev + 1, nlat, nlon))
        omega[1:-1] = 3.0 * np.sin(lon) * np.cos(lat)
    elif family == "uniform_polar":
        # A uniform field under a strong cross-polar flow: the polar band
        # sub-cycles and every sweep must leave the ratio uniform.
        tracers = {name: np.full(shape, 2.0e-3, np.float32) for name in ("qs", "qg", "ng")}
        alpha = math.pi / 2.0 - 0.05
        u0 = 2.0 * math.pi * transform.grid.radius_m / (12.0 * 86400.0)
        lat2 = lat * np.ones((1, nlon))
        lon2 = lon * np.ones((nlat, 1))
        u = (u0 * (np.cos(lat2) * np.cos(alpha) + np.sin(lat2) * np.cos(lon2) * np.sin(alpha)))[None] * np.ones(shape)
        v = (-u0 * np.sin(lon2) * np.sin(alpha))[None] * np.ones(shape)
        omega = np.zeros((nlev + 1, nlat, nlon))
    else:
        raise KeyError(family)
    dp_u = (dp * u).astype(np.float32)
    dp_v = (dp * v).astype(np.float32)
    return tracers, dp, dp_u, dp_v, omega.astype(np.float32)


# -- CPU: the stacked specification against the per-tracer loop ---------

@pytest.mark.parametrize("family", ["random", "spikes", "uniform_polar"])
def test_stacked_sweeps_return_the_per_tracer_loop_bits(family):
    transform, vertical, transport = _transport(fused=False)
    tracers, dp, dp_u, dp_v, omega = _fields(transform, vertical, 5, family=family)
    names = list(tracers)
    xp = np
    dt = np.float32(120.0)
    mass_dict = {name: tracers[name] * dp for name in names}
    stack = np.stack([mass_dict[name] for name in names])
    psi_x = 0.5 * (dp_u + np.roll(dp_u, -1, axis=-1)) * transport._x_scale
    psi_y = 0.5 * (dp_v[:, :-1, :] + dp_v[:, 1:, :]) * transport._y_face
    psi_z = omega[1:-1]
    # zonal
    d_ref, m_ref = _reference_periodic_step(mass_dict, dp, psi_x, dt, xp)
    m_new, d_new = transport._periodic_step(stack, dp, psi_x, dt)
    assert np.array_equal(d_ref, d_new)
    for index, name in enumerate(names):
        assert np.array_equal(m_ref[name], m_new[index]), name
    # meridional
    d_ref, m_ref = _reference_bounded_step(
        mass_dict, dp, psi_y, dt, axis=1, centre=transport._mu,
        face=transport._mu_half, inv_width=transport._y_inv_dmu, xp=xp,
    )
    m_new, d_new = transport._walled_step(
        stack, dp, psi_y, dt, axis=1, centre=transport._mu,
        face=transport._mu_half, inv_width=transport._y_inv_dmu,
    )
    assert np.array_equal(d_ref, d_new)
    for index, name in enumerate(names):
        assert np.array_equal(m_ref[name], m_new[index]), name
    # vertical
    p_half, p_full = transport._pressure_coordinates(dp)
    ones = np.ones_like(dp)
    d_ref, m_ref = _reference_bounded_step(
        mass_dict, dp, psi_z, dt, axis=0, centre=p_full, face=p_half,
        inv_width=ones, xp=xp,
    )
    m_new, d_new = transport._walled_step(
        stack, dp, psi_z, dt, axis=0, centre=p_full, face=p_half, inv_width=ones,
    )
    assert np.array_equal(d_ref, d_new)
    for index, name in enumerate(names):
        assert np.array_equal(m_ref[name], m_new[index]), name


def test_the_fused_flag_is_inert_on_numpy():
    transform, vertical, transport = _transport(fused=True)
    assert transport._use_fused is False
    tracers, dp, dp_u, dp_v, omega = _fields(transform, vertical, 2, family="random")
    out, metrics = transport.advance(tracers, dp, dp_u, dp_v, omega, 300.0)
    assert set(out) == set(tracers)
    assert metrics["floor_clip_kg_m2"] >= 0.0
