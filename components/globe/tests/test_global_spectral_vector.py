from __future__ import annotations

import numpy as np

from woof.globe.spectral.transform import SphericalHarmonicTransform
from woof.globe.spectral.vector import VorticityDivergenceOperator


def _random_triangle(transform, *, seed: int, max_degree: int):
    rng = np.random.default_rng(seed)
    out = np.zeros(transform.spectral_shape, dtype=np.complex128)
    for n in range(1, max_degree + 1):
        for m in range(n + 1):
            out[n, m] = rng.normal() + (0.0j if m == 0 else 1j * rng.normal())
    return transform.backend.asarray(out, dtype=transform.backend.complex_dtype)


def test_direct_vector_transform_roundtrips_low_degree_vordiv():
    transform = SphericalHarmonicTransform.create(15, precision="float64")
    vector = VorticityDivergenceOperator(transform)
    zeta = _random_triangle(transform, seed=3, max_degree=9)
    divergence = _random_triangle(transform, seed=4, max_degree=9)
    u, v = vector.wind_from_vordiv(zeta, divergence)
    zeta_back, divergence_back = vector.vordiv_from_wind(u, v)
    zeta_error = np.max(
        np.abs(transform.backend.to_numpy(zeta_back - zeta))
    )
    div_error = np.max(
        np.abs(transform.backend.to_numpy(divergence_back - divergence))
    )
    assert zeta_error < 2.0e-12
    assert div_error < 2.0e-12


def test_vector_analysis_has_no_degree_zero_vorticity_or_divergence():
    transform = SphericalHarmonicTransform.create(9, precision="float64")
    vector = VorticityDivergenceOperator(transform)
    rng = np.random.default_rng(8)
    u = rng.normal(size=transform.grid.shape)
    v = rng.normal(size=transform.grid.shape)
    zeta, divergence = vector.vordiv_from_wind(u, v)
    zeta = transform.backend.to_numpy(zeta)
    divergence = transform.backend.to_numpy(divergence)
    assert zeta[0, 0] == 0.0
    assert divergence[0, 0] == 0.0


def test_solid_body_rotation_matches_analytic_vorticity():
    transform = SphericalHarmonicTransform.create(15, precision="float64")
    vector = VorticityDivergenceOperator(transform)
    u0 = 40.0
    u = u0 * transform.grid.cos_lat[:, None] * np.ones((1, transform.grid.nlon))
    v = np.zeros_like(u)
    zeta, divergence = vector.vordiv_from_wind(u, v)
    zeta_grid = transform.backend.to_numpy(transform.inverse(zeta))
    divergence_grid = transform.backend.to_numpy(transform.inverse(divergence))
    expected = 2.0 * u0 * transform.grid.sin_lat[:, None] / transform.grid.radius_m
    assert np.max(np.abs(zeta_grid - expected)) < 2.0e-18
    assert np.max(np.abs(divergence_grid)) < 2.0e-18
