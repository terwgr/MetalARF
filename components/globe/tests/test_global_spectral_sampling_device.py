"""The device point sampler against the host one, both directions.

The members of the global ensemble live on the card, and the point
operators sample their spectral fields at report positions; the device
path of :mod:`woof.globe.spectral.sampling` builds the packed Legendre
basis of a point chunk with one fused kernel and contracts every field
of a member stack with one GEMM per chunk and output.  In float64 it has
to give the host path's numbers to rounding (the host computes in
complex128); at ``precision="float32"`` (float32 GEMMs over 256-term
blocks summed in float64, the state's own precision for a float32
state) it has to agree to a stated bound; the packed basis has to be the
host recurrence's values to rounding; zero coefficients have to sample
to exactly zero; and a device result can stay on the card.  Runs where
cupy and a card are present; skipped elsewhere (the host path is pinned
by ``test_global_spectral_sampling.py``).
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof.globe.spectral.sampling import (  # noqa: E402
    packed_basis,
    sample_gradient,
    sample_scalar,
    sample_wind,
    triangle_index,
)
from woof.globe.spectral.transform import SphericalHarmonicTransform  # noqa: E402


def _card_present() -> bool:
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # noqa: BLE001 - no driver, no card
        return False


pytestmark = pytest.mark.skipif(not _card_present(), reason="the device sampler needs a card")


def _coefficients(rng, shape, truncation):
    c = rng.standard_normal((*shape, truncation + 1, truncation + 1)) + 1j * rng.standard_normal(
        (*shape, truncation + 1, truncation + 1))
    n_idx, m_idx = np.indices((truncation + 1, truncation + 1))
    c[..., m_idx > n_idx] = 0.0
    c[..., :, 0] = c[..., :, 0].real
    return c.astype(np.complex64)


@pytest.mark.parametrize("truncation", [21, 63])
def test_the_device_sampler_matches_the_host_sampler_to_rounding(truncation):
    transform = SphericalHarmonicTransform.create(truncation, backend="cupy", precision="float32")
    rng = np.random.default_rng(truncation)
    npts = 700
    lat = rng.uniform(-88.0, 88.0, npts)
    lon = rng.uniform(-180.0, 360.0, npts)
    lat[:40] = lat[40:80]                       # duplicated points, as unique-point gathers produce
    lon[:40] = lon[40:80]
    for shape in ((4, 3), (10,)):
        c = _coefficients(rng, shape, truncation)
        host = sample_scalar(transform, c, lat, lon)
        device = sample_scalar(transform, cp.asarray(c), lat, lon)
        assert host.shape == device.shape == (*shape, npts)
        assert np.abs(host - device).max() <= 1e-12 * np.abs(host).max()
        he, hn = sample_gradient(transform, c, lat, lon)
        de, dn = sample_gradient(transform, cp.asarray(c), lat, lon)
        assert np.abs(he - de).max() <= 1e-12 * np.abs(he).max()
        assert np.abs(hn - dn).max() <= 1e-12 * np.abs(hn).max()
    vort = _coefficients(rng, (4, 10), truncation)
    div = _coefficients(rng, (4, 10), truncation)
    hu, hv = sample_wind(transform, vort, div, lat, lon)
    du, dv = sample_wind(transform, cp.asarray(vort), cp.asarray(div), lat, lon)
    assert np.abs(hu - du).max() <= 1e-12 * np.abs(hu).max()
    assert np.abs(hv - dv).max() <= 1e-12 * np.abs(hv).max()


def test_zero_coefficients_sample_to_exactly_zero_on_the_device():
    transform = SphericalHarmonicTransform.create(21, backend="cupy", precision="float32")
    zeros = cp.zeros((2, 22, 22), dtype=cp.complex64)
    lat = np.linspace(-80.0, 80.0, 50)
    lon = np.linspace(0.0, 350.0, 50)
    assert np.all(sample_scalar(transform, zeros, lat, lon) == 0.0)
    east, north = sample_gradient(transform, zeros, lat, lon)
    assert np.all(east == 0.0) and np.all(north == 0.0)


def test_the_device_sampler_refuses_the_poles_for_gradients_like_the_host():
    transform = SphericalHarmonicTransform.create(21, backend="cupy", precision="float32")
    c = cp.asarray(_coefficients(np.random.default_rng(0), (1,), 21))
    with pytest.raises(ValueError, match="poles"):
        sample_gradient(transform, c, np.array([90.0]), np.array([0.0]))


def test_the_packed_basis_is_the_host_recurrence_to_rounding():
    from woof.globe.spectral.legendre import normalized_associated_legendre

    t = 31
    rng = np.random.default_rng(3)
    lat = np.deg2rad(rng.uniform(-88.0, 88.0, 37))
    lon = np.deg2rad(rng.uniform(0.0, 360.0, 37))
    n_idx, m_idx = triangle_index(t)
    ktri = n_idx.size
    basis, derivative = normalized_associated_legendre(t, np.sin(lat))
    mult = np.where(m_idx == 0, 1.0, 2.0)[:, None]
    phase = m_idx[:, None] * lon[None, :]
    p_tri = basis[n_idx, m_idx]
    dp_tri = derivative[n_idx, m_idx]
    b0 = cp.asnumpy(packed_basis(cp, t, lat, lon, gradient=False))
    assert b0.shape == (2 * ktri, lat.size)
    assert np.allclose(b0[:ktri], mult * p_tri * np.cos(phase), rtol=1e-13, atol=1e-15)
    assert np.allclose(b0[ktri:], -mult * p_tri * np.sin(phase), rtol=1e-13, atol=1e-15)
    bz, bm = (cp.asnumpy(v) for v in packed_basis(cp, t, lat, lon, gradient=True))
    mm = mult * m_idx[:, None]
    assert np.allclose(bz[:ktri], -mm * p_tri * np.sin(phase), rtol=1e-13, atol=1e-15)
    assert np.allclose(bz[ktri:], -mm * p_tri * np.cos(phase), rtol=1e-13, atol=1e-15)
    assert np.allclose(bm[:ktri], mult * dp_tri * np.cos(phase), rtol=1e-13, atol=1e-13)
    assert np.allclose(bm[ktri:], -mult * dp_tri * np.sin(phase), rtol=1e-13, atol=1e-13)


def test_state_precision_agrees_to_its_stated_bound_and_a_device_result_stays_on_the_card():
    transform = SphericalHarmonicTransform.create(63, backend="cupy", precision="float32")
    rng = np.random.default_rng(5)
    npts = 900
    lat = rng.uniform(-85.0, 85.0, npts)
    lon = rng.uniform(0.0, 360.0, npts)
    c = _coefficients(rng, (6, 4), 63)
    c[..., 0, 0] += 40.0                        # a mean, as a temperature field carries
    d = cp.asarray(c)
    exact = sample_scalar(transform, d, lat, lon, precision="float64")
    fast = sample_scalar(transform, d, lat, lon, precision="float32")
    assert np.abs(fast - exact).max() <= 1e-5 * np.abs(exact).max()
    ee, en = sample_gradient(transform, d, lat, lon, precision="float64")
    fe, fn = sample_gradient(transform, d, lat, lon, precision="float32")
    assert np.abs(fe - ee).max() <= 1e-5 * np.abs(ee).max()
    assert np.abs(fn - en).max() <= 1e-5 * np.abs(en).max()
    on_card = sample_scalar(transform, d, lat, lon, precision="float64", as_device=True)
    assert isinstance(on_card, cp.ndarray) and on_card.dtype == cp.float64
    assert np.array_equal(cp.asnumpy(on_card), exact)
    u_card, v_card = sample_wind(transform, d, d, lat, lon, as_device=True)
    u_host, v_host = sample_wind(transform, d, d, lat, lon)
    assert isinstance(u_card, cp.ndarray)
    assert np.array_equal(cp.asnumpy(u_card), u_host) and np.array_equal(cp.asnumpy(v_card), v_host)
    with pytest.raises(ValueError, match="precision"):
        sample_scalar(transform, d, lat, lon, precision="float16")
