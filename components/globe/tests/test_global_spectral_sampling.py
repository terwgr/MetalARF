from __future__ import annotations

import numpy as np

from woof.globe.spectral.sampling import sample_scalar, sample_wind
from woof.globe.spectral.transform import SphericalHarmonicTransform
from woof.globe.spectral.vector import VorticityDivergenceOperator


def _random_coeff(transform, seed=1, degree=7, scale=1.0):
    rng = np.random.default_rng(seed)
    coeff = np.zeros(transform.spectral_shape, dtype=np.complex128)
    for n in range(degree + 1):
        for m in range(n + 1):
            coeff[n, m] = scale * (rng.normal() + (0j if m == 0 else 1j * rng.normal()))
    return transform.backend.asarray(coeff, dtype=transform.backend.complex_dtype)


def test_arbitrary_scalar_sampling_matches_gaussian_grid_synthesis():
    transform = SphericalHarmonicTransform.create(12, precision="float64")
    coeff = _random_coeff(transform, degree=8)
    latitude, longitude = np.meshgrid(
        transform.grid.latitude_deg,
        transform.grid.longitude_deg,
        indexing="ij",
    )
    direct = sample_scalar(transform, coeff, latitude, longitude)
    native = transform.backend.to_numpy(transform.inverse(coeff))
    assert np.max(np.abs(direct - native)) < 2.0e-13


def test_arbitrary_wind_sampling_matches_gaussian_grid_inversion():
    transform = SphericalHarmonicTransform.create(12, precision="float64")
    zeta = _random_coeff(transform, seed=2, degree=8, scale=1.0e-5)
    divergence = _random_coeff(transform, seed=3, degree=8, scale=1.0e-5)
    zeta[0, 0] = 0.0
    divergence[0, 0] = 0.0
    latitude, longitude = np.meshgrid(
        transform.grid.latitude_deg,
        transform.grid.longitude_deg,
        indexing="ij",
    )
    sampled_u, sampled_v = sample_wind(
        transform, zeta, divergence, latitude, longitude
    )
    native_u, native_v = VorticityDivergenceOperator(
        transform
    ).wind_from_vordiv(zeta, divergence)
    assert np.max(
        np.abs(sampled_u - transform.backend.to_numpy(native_u))
    ) < 2.0e-12
    assert np.max(
        np.abs(sampled_v - transform.backend.to_numpy(native_v))
    ) < 2.0e-12


def test_chunked_scalar_and_wind_sampling_are_arithmetic_identical():
    transform = SphericalHarmonicTransform.create(12, precision="float64")
    scalar = _random_coeff(transform, seed=11, degree=8)
    zeta = _random_coeff(transform, seed=12, degree=8, scale=1.0e-5)
    divergence = _random_coeff(transform, seed=13, degree=8, scale=1.0e-5)
    zeta[0, 0] = 0.0
    divergence[0, 0] = 0.0
    latitude = np.linspace(-87.0, 87.0, 37)
    longitude = np.linspace(-720.0, 720.0, 37)
    whole = sample_scalar(
        transform, scalar, latitude, longitude, chunk_points=10_000
    )
    chunked = sample_scalar(
        transform, scalar, latitude, longitude, chunk_points=3
    )
    np.testing.assert_array_equal(chunked, whole)
    u_whole, v_whole = sample_wind(
        transform,
        zeta,
        divergence,
        latitude,
        longitude,
        chunk_points=10_000,
    )
    u_chunked, v_chunked = sample_wind(
        transform,
        zeta,
        divergence,
        latitude,
        longitude,
        chunk_points=2,
    )
    np.testing.assert_array_equal(u_chunked, u_whole)
    np.testing.assert_array_equal(v_chunked, v_whole)


def test_default_sampler_stays_bounded_at_t127_point_sets():
    transform = SphericalHarmonicTransform.create(127, precision="float64")
    coeff = transform.zeros()
    coeff[1, 0] = 1.0
    latitude = np.linspace(-89.0, 89.0, 600)
    longitude = np.linspace(0.0, 359.0, 600)
    sampled = sample_scalar(transform, coeff, latitude, longitude)
    assert sampled.shape == (600,)
    assert np.isfinite(sampled).all()
