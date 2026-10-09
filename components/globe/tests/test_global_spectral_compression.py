from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from woof.globe.spectral.compression import (
    QuantizedTriangle,
    compress_scalar,
    compress_wind,
    decode_scalar,
    decode_wind,
    quantize_coefficients,
    read_compressed_scalar,
    read_compressed_wind,
    write_compressed_scalar,
    write_compressed_wind,
)
from woof.globe.spectral.initial_conditions import williamson2_state
from woof.globe.spectral.transform import SphericalHarmonicTransform
from woof.globe.spectral.vector import VorticityDivergenceOperator


def test_full_degree_scalar_quantization_is_small_and_archive_roundtrips(tmp_path):
    transform = SphericalHarmonicTransform.create(15, precision="float64")
    coefficients = transform.zeros()
    coefficients[0, 0] = 280.0 * np.sqrt(4.0 * np.pi)
    coefficients[3, 2] = 2.5 - 1.25j
    coefficients[7, 4] = -0.4 + 0.8j
    field = transform.backend.to_numpy(transform.inverse(coefficients))
    compressed = compress_scalar(transform, field, keep_degree=15)
    assert compressed.header["metrics"]["relative_l2"]["maximum"] < 3.0e-5
    path = write_compressed_scalar(tmp_path / "scalar.npz", compressed)
    restored = read_compressed_scalar(path)
    np.testing.assert_array_equal(
        decode_scalar(transform, compressed), decode_scalar(transform, restored)
    )
    assert restored.header["payload_to_raw_ratio"] < 0.3


def test_zero_mean_linear_fields_survive_the_default_mean_restoration():
    # Every anomaly field the codec is pointed at -- vorticity, divergence,
    # any departure from a mean -- has an area mean that is float64 rounding
    # noise.  In linear space the mean is the n=0 coefficient, so restoring
    # it must be additive; a multiplicative restoration divides by that noise.
    # 3.0e-5 is one int16 quantum, 1/32767, the codec's own per-degree scale.
    transform = SphericalHarmonicTransform.create(15, precision="float64")
    vorticity = transform.backend.to_numpy(
        transform.inverse(williamson2_state(transform).vorticity)
    )
    assert abs(transform.grid.global_mean(vorticity)) < 1.0e-20
    compressed = compress_scalar(transform, vorticity, keep_degree=15)
    decoded = decode_scalar(transform, compressed)
    assert (
        np.max(np.abs(decoded - vorticity)) / np.max(np.abs(vorticity)) < 3.0e-5
    )

    coefficients = transform.zeros()
    coefficients[1, 0] = 3.0
    coefficients[3, 2] = 2.5 - 1.25j
    coefficients[7, 4] = -0.4 + 0.8j
    field = transform.backend.to_numpy(transform.inverse(coefficients))
    compressed = compress_scalar(transform, field, keep_degree=15)
    decoded = decode_scalar(transform, compressed)
    assert np.max(np.abs(decoded - field)) / np.max(np.abs(field)) < 3.0e-5


def test_positive_field_log_compression_bounds_the_field_it_returns():
    transform = SphericalHarmonicTransform.create(15, precision="float64")
    lat, lon = transform.grid.mesh()
    field = np.exp(0.25 * np.cos(lat) * np.cos(2.0 * lon)) + 1.0e-3
    compressed = compress_scalar(
        transform, field, space="log", floor=1.0e-12, preserve_mean=True
    )
    decoded = decode_scalar(transform, compressed)
    # exp() of a cosine has no finite spherical-harmonic expansion, so the
    # T15 fit is truncation-limited rather than quantization-limited:
    # measured 9.11e-3 relative Linf here, and the same 9.04e-3 in linear
    # space and with the mean restoration off, which is what identifies the
    # error as truncation.
    error = np.max(np.abs(decoded - field)) / np.max(np.abs(field))
    assert error < 1.0e-2
    assert compressed.header["metrics"]["relative_linf"]["maximum"] < 1.0e-2
    assert np.min(decoded) > 0.0
    assert abs(transform.grid.global_mean(decoded) - transform.grid.global_mean(field)) < 2.0e-14

    # exp() manufactures positivity and the mean restoration manufactures the
    # mean, for any payload at all: without the field bound above, a codec
    # that returned nothing would still pass this test.
    hollow = replace(
        compressed,
        triangle=QuantizedTriangle(
            np.zeros_like(compressed.triangle.q_real),
            np.zeros_like(compressed.triangle.q_imag),
            compressed.triangle.scales,
            compressed.triangle.truncation,
        ),
    )
    empty = decode_scalar(transform, hollow, validate_payload=False)
    assert np.min(empty) > 0.0
    assert abs(transform.grid.global_mean(empty) - transform.grid.global_mean(field)) < 2.0e-14
    assert np.max(np.abs(empty - field)) / np.max(np.abs(field)) > 20.0 * error


def test_quantization_refuses_a_nonreal_m_zero_coefficient():
    # The m=0 imaginary part sets the degree scale before it is dropped, so
    # zeroing it silently coarsens every other coefficient at that degree.
    coefficients = np.zeros((4, 4), dtype=np.complex128)
    coefficients[3, 0] = 1.0 + 1.0e6j
    coefficients[3, 1] = 2.0 + 1.5j
    coefficients[3, 2] = -0.75 + 0.25j
    with pytest.raises(ValueError, match="m=0 imaginary part"):
        quantize_coefficients(coefficients)


def test_log_compression_refuses_negative_input():
    transform = SphericalHarmonicTransform.create(7)
    field = np.ones(transform.grid.shape)
    field[0, 0] = -1.0
    with pytest.raises(ValueError, match="negative"):
        compress_scalar(transform, field, space="log")


def test_progressive_scalar_decode_never_requests_missing_degrees():
    transform = SphericalHarmonicTransform.create(12)
    lat, lon = transform.grid.mesh()
    field = 5.0 + np.cos(lat) * np.cos(8.0 * lon)
    compressed = compress_scalar(transform, field, keep_degree=8)
    coarse = decode_scalar(transform, compressed, keep_degree=4)
    fine = decode_scalar(transform, compressed, keep_degree=8)
    assert np.sqrt(np.mean((fine - field) ** 2)) < np.sqrt(np.mean((coarse - field) ** 2))
    with pytest.raises(ValueError, match="exceeds stored"):
        decode_scalar(transform, compressed, keep_degree=9)


def _broadband_wind(transform, seed: int = 11):
    """Wind whose vorticity and divergence populate every admitted degree.

    The Williamson-2 solid body puts all its vorticity in one coefficient and
    no divergence at all, and the codec's per-degree scale is the largest
    magnitude at that degree, so a lone coefficient quantizes to exactly
    +-32767 and dequantizes with no error whatever.  A quantization gate has
    no power on such a field.
    """
    rng = np.random.default_rng(seed)
    zeta = transform.zeros()
    divergence = transform.zeros()
    for n in range(1, transform.truncation + 1):
        for m in range(n + 1):
            amplitude = 3.0e-6 / (1.0 + n)
            phase = 0.0 if m == 0 else 1j * rng.normal()
            zeta[n, m] = amplitude * (rng.normal() + phase)
            phase = 0.0 if m == 0 else 1j * rng.normal()
            divergence[n, m] = 0.4 * amplitude * (rng.normal() + phase)
    return VorticityDivergenceOperator(transform).wind_from_vordiv(
        transform.project(zeta), transform.project(divergence)
    )


def test_wind_codec_quantization_is_bounded_on_a_broadband_wind():
    transform = SphericalHarmonicTransform.create(15, precision="float64")
    u, v = _broadband_wind(transform)
    un = transform.backend.to_numpy(u)
    vn = transform.backend.to_numpy(v)
    assert np.max(np.sqrt(un * un + vn * vn)) > 1.0
    compressed = compress_wind(transform, un, vn, keep_degree=15)
    # One int16 quantum is 1/32767 = 3.05e-5 of a degree's largest
    # coefficient, and rounding costs at most half of one; measured 1.05e-5
    # for this field against the 3.0e-5 whole-quantum bound.
    assert compressed.header["metrics"]["wind_vector_relative_l2"]["maximum"] < 3.0e-5
    assert compressed.header["metrics"]["wind_vector_relative_l2"]["maximum"] > 1.0e-8

    # Dropping resolved degrees is the failure the bound has to see.
    coarse = compress_wind(transform, un, vn, keep_degree=8)
    assert coarse.header["metrics"]["wind_vector_relative_l2"]["maximum"] > 3.0e-5


def test_wind_codec_uses_vordiv_and_roundtrips_archive(tmp_path):
    transform = SphericalHarmonicTransform.create(15, precision="float64")
    state = williamson2_state(transform)
    u, v = VorticityDivergenceOperator(transform).wind_from_vordiv(
        state.vorticity, state.divergence
    )
    compressed = compress_wind(transform, u, v, keep_degree=15)
    assert compressed.header["representation"].startswith("vorticity-divergence")
    path = write_compressed_wind(tmp_path / "wind.npz", compressed)
    restored = read_compressed_wind(path)
    u0, v0 = decode_wind(transform, compressed)
    u1, v1 = decode_wind(transform, restored)
    np.testing.assert_array_equal(u0, u1)
    np.testing.assert_array_equal(v0, v1)
