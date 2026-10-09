from __future__ import annotations

import math
import tracemalloc

import numpy as np
import pytest

from woof.globe.spectral.legendre import DEFAULT_BAND
from woof.globe.spectral.grid import GaussianGrid
from woof.globe.spectral.transform import SphericalHarmonicTransform


def test_constant_mode_is_exact_and_mean_normalized():
    transform = SphericalHarmonicTransform.create(9, precision="float64")
    coeff = transform.constant_coeff(7.25)
    grid = transform.backend.to_numpy(transform.inverse(coeff))
    assert np.max(np.abs(grid - 7.25)) < 2.0e-14
    assert abs(transform.grid.global_mean(grid) - 7.25) < 2.0e-14
    back = transform.backend.to_numpy(transform.forward(grid))
    assert abs(back[0, 0] - 7.25 * math.sqrt(4.0 * math.pi)) < 5.0e-14
    assert np.max(np.abs(back[1:])) < 5.0e-14


def test_t15_roundtrip_and_parseval_close_at_float64_precision():
    transform = SphericalHarmonicTransform.create(15, precision="float64")
    result = transform.transform_check(seed=19)
    assert result["roundtrip_relative_linf"] < 5.0e-13
    assert result["parseval_relative_error"] < 5.0e-14


def test_analysis_recovers_closed_form_harmonics_it_did_not_synthesize():
    # forward(inverse(x)) cannot fail: __post_init__ replaces the analysis
    # rows with solve(rows @ basis.T, rows), whose product with basis.T is the
    # identity for any rows at all.  Corrupt the basis normalization and the
    # round trip still returns 3.6e-16.  So the analysis operator has to be
    # measured against harmonics built from their closed forms instead, with
    # no inverse() in the path: a real field carrying 2*Re(Y_nm) for m>0, or
    # Y_n0 for m=0, must analyse to exactly one at (n, m) and nothing else.
    # Measured: 8.4e-16 worst case here, 0.47 with the degree normalization
    # scaled by (1 + 0.3n).
    transform = SphericalHarmonicTransform.create(15, precision="float64")
    lat, lon = transform.grid.mesh()
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    harmonics = {
        (0, 0): np.full(lat.shape, 1.0 / math.sqrt(4.0 * math.pi)),
        (1, 0): math.sqrt(3.0 / (4.0 * math.pi)) * sin_lat,
        (2, 0): math.sqrt(5.0 / (16.0 * math.pi)) * (3.0 * sin_lat**2 - 1.0),
        (2, 2): 2.0
        * np.real(
            math.sqrt(15.0 / (32.0 * math.pi)) * cos_lat**2 * np.exp(2j * lon)
        ),
        (3, 2): 2.0
        * np.real(
            math.sqrt(105.0 / (32.0 * math.pi))
            * sin_lat
            * cos_lat**2
            * np.exp(2j * lon)
        ),
    }
    for (n, m), field in harmonics.items():
        coefficients = transform.backend.to_numpy(transform.forward(field))
        assert abs(coefficients[n, m] - 1.0) < 5.0e-15
        coefficients[n, m] = 0.0
        assert np.max(np.abs(coefficients)) < 5.0e-15


def test_laplacian_and_inverse_laplacian_are_degree_exact():
    transform = SphericalHarmonicTransform.create(12, precision="float64")
    coeff = transform.zeros()
    coeff[7, 3] = 1.25 - 0.75j
    lap = transform.backend.to_numpy(transform.laplacian(coeff))
    eigen = -7.0 * 8.0 / transform.grid.radius_m**2
    assert abs(lap[7, 3] - eigen * coeff[7, 3]) < 1.0e-28
    recovered = transform.backend.to_numpy(transform.inverse_laplacian(lap))
    assert np.max(np.abs(recovered - transform.backend.to_numpy(coeff))) < 1.0e-14


def test_dealiased_grid_has_enough_latitudes_and_longitudes():
    # T+1 and 2T+1 are the linear-representation minima, which a
    # dealias_factor of 1.0 already meets; the 1.5x rule asks for
    # ceil(1.5*(T+1)) latitudes and ceil(3.0*(T+1)) longitudes so quadratic
    # products at T21 do not alias back into the resolved modes.
    truncation = 21
    transform = SphericalHarmonicTransform.create(truncation, dealias_factor=1.5)
    assert transform.grid.nlat >= math.ceil(1.5 * (truncation + 1))
    assert transform.grid.nlon >= math.ceil(3.0 * (truncation + 1))
    assert transform.grid.nlon % 2 == 0


def test_high_truncation_transform_is_finite_and_roundtrips():
    # The unnormalized Legendre recurrence carried a (2m-1)!! diagonal that
    # overflows float64 near m=150, so every transform above ~T150 silently
    # returned NaN while config.MAXIMUM_TRUNCATION admits truncation up to
    # 255.  The normalized recurrence stays bounded for the whole admitted
    # range.
    transform = SphericalHarmonicTransform.create(170, precision="float64")
    result = transform.transform_check(seed=7)
    assert math.isfinite(result["roundtrip_relative_linf"])
    assert result["roundtrip_relative_linf"] < 5.0e-13
    assert result["parseval_relative_error"] < 5.0e-14


def test_basis_matches_pinned_scipy_spherical_harmonic_reference():
    # scipy.special.sph_harm_y exists only in scipy>=1.15 while the project
    # floor is scipy>=1.11, so the reference values are committed as a pinned
    # fixture (tests/data/sph_harm_reference_scipy115.json) and the gate runs
    # on every in-spec install with no scipy dependency at all.
    import json
    from pathlib import Path

    from woof.globe.spectral.legendre import normalized_associated_legendre_values

    fixture = json.loads(
        (Path(__file__).parent / "data" / "sph_harm_reference_scipy115.json")
        .read_text(encoding="utf-8")
    )
    latitude = np.deg2rad(np.asarray(fixture["latitude_deg"]))
    longitude = np.deg2rad(np.asarray(fixture["longitude_deg"]))
    basis = normalized_associated_legendre_values(10, np.sin(latitude))
    for key, row in fixture["values"].items():
        n, m = (int(part) for part in key.split(","))
        ours = basis[n, m] * np.exp(1j * m * longitude)
        reference = np.asarray(row["real"]) + 1j * np.asarray(row["imag"])
        np.testing.assert_allclose(ours, reference, rtol=2.0e-14, atol=2.0e-14)

    try:
        from scipy.special import sph_harm_y
    except ImportError:
        return
    for key, row in fixture["values"].items():
        n, m = (int(part) for part in key.split(","))
        live = sph_harm_y(n, m, np.pi / 2.0 - latitude, longitude)
        pinned = np.asarray(row["real"]) + 1j * np.asarray(row["imag"])
        np.testing.assert_allclose(live, pinned, rtol=2.0e-14, atol=2.0e-14)


def _packed_table_bytes(truncation: int, nlat: int, band: int, itemsize: int) -> int:
    """Bytes of one banded table: sum over bands of orders x (T+1-m0) x nlat."""
    total = 0
    for m0 in range(0, truncation + 1, band):
        total += (min(m0 + band, truncation + 1) - m0) * (truncation + 1 - m0) * nlat
    return total * itemsize


def test_the_tables_are_packed_by_band_and_the_derivative_is_lazy():
    # A dense (T+1, T+1, nlat) table carries a zero for every n < m: half
    # of it.  At T1534 on the 1536-latitude GFS grid that half is 14.5 GB
    # per table and the analysis process was OOM-killed at 120 GB on a
    # 128 GB host (2026-09-01).  The packed tables hold the triangle plus
    # at most band-1 spare columns per order, the derivative is absent
    # until a gradient asks for it, and the transposed views that the
    # vector transform reads are views of the expansion scratch, not
    # copies.
    truncation, band = 63, 8
    transform = SphericalHarmonicTransform.create(
        truncation, precision="float64", legendre_band=band
    )
    nlat = transform.grid.nlat
    dense = (truncation + 1) ** 2 * nlat * 8
    packed = _packed_table_bytes(truncation, nlat, band, 8)
    assert packed < 0.6 * dense
    held = transform.legendre_table_nbytes
    assert held["basis"] == packed
    assert held["analysis"] == packed
    assert held["derivative"] == 0
    assert held["scratch"] == 0

    coeff = transform.zeros()
    coeff[3, 1] = 1.0
    transform.forward(transform.inverse(coeff))
    assert transform.legendre_table_nbytes["derivative"] == 0, (
        "analysis and synthesis must not build the derivative table"
    )
    # Two expansion scratches (one per layout), each one band of the
    # dense table.
    assert transform.legendre_table_nbytes["scratch"] == 2 * band * (truncation + 1) * nlat * 8

    transform.gradient(coeff)
    held = transform.legendre_table_nbytes
    assert held["derivative"] == packed
    assert held["scratch"] == 3 * band * (truncation + 1) * nlat * 8


def test_the_band_is_a_layout_under_numpy_and_an_identity_everywhere():
    """Under numpy every order is its own BLAS GEMM whatever the band, so
    the band moves no number here.  Under cupy the band is the strided-
    batched GEMM's batch count and it DOES move numbers: MEASURED
    2026-09-06 on an RTX 5070 Ti at T255 float32, the analysis of a single
    plane differs by up to 7.6e-06 between the shipped band 32 and bands
    8, 16 and 128, and a ten-step T255 native run at band 16 left 67 of
    125 checkpoint arrays differing from the band-32 run.  So the identity
    carries a non-default band on every backend rather than being
    backend-dependent, and the shipped band keeps every hash ever
    written."""
    coeff = np.zeros((16, 16), dtype=np.complex128)
    rng = np.random.default_rng(5)
    for n in range(16):
        for m in range(n + 1):
            coeff[n, m] = rng.normal() + (0.0 if m == 0 else 1j * rng.normal())
    grids = {}
    hashes = {}
    for band in (1, 4, 16, 32):
        transform = SphericalHarmonicTransform.create(15, legendre_band=band)
        grids[band] = transform.backend.to_numpy(transform.inverse(coeff))
        hashes[band] = transform.identity_hash
    for band in (4, 16, 32):
        assert np.array_equal(grids[1], grids[band])
    assert len(set(hashes.values())) == 4
    default = SphericalHarmonicTransform.create(15)
    assert hashes[DEFAULT_BAND] == default.identity_hash
    assert "legendre_band" not in default.identity
    with pytest.raises(ValueError, match="legendre_band must be >= 1"):
        SphericalHarmonicTransform.create(15, legendre_band=0)


# --- capped exact analysis on a grid the truncation did not derive --------

def test_a_native_grid_states_its_content_and_its_exact_truncation():
    # 1536x3072 is the native GFS grid: T_in = 1535, and 2*1536-1-1535 =
    # 1536, so every T <= 1535 is an exact projection there.  The default
    # 1.5x grid of T533 (801x1602) carries T_in = 800 and is exact to 800.
    native = GaussianGrid.for_shape(1536, 3072)
    assert native.truncation == 1535
    assert native.zonal_content_truncation == 1535
    assert native.exact_analysis_truncation == 1535
    default = GaussianGrid.create(533)
    assert (default.nlat, default.nlon) == (801, 1602)
    assert default.zonal_content_truncation == 800
    assert default.exact_analysis_truncation == 800
    capped = native.truncated_to(767)
    assert capped.truncation == 767
    assert capped.nlat == 1536 and capped.nlon == 3072
    assert np.array_equal(capped.sin_lat, native.sin_lat)
    assert np.array_equal(capped.quadrature_weights, native.quadrature_weights)


def test_an_aliasing_cap_is_refused_with_the_condition_spelt_out():
    # 22 latitudes integrate degree 43 exactly; a 64-longitude grid carries
    # T_in = 31, so analysing its content at T21 needs 31 + 21 = 52 <= 43,
    # which fails: degrees above 12 would alias.
    grid = GaussianGrid.create(21, nlat=22, nlon=64, dealias_factor=1.0)
    assert grid.exact_analysis_truncation == 12
    with pytest.raises(ValueError, match=r"31 \+ 21 = 52 > 43") as caught:
        grid.truncated_to(21)
    assert "T_in + T <= 2*nlat - 1" in str(caught.value)
    assert "T <= 12 is exact" in str(caught.value)
    grid.truncated_to(12)
    # An order the real FFT does not carry is refused on its own terms.
    with pytest.raises(ValueError, match=r"m <= nlon/2 - 1 = 31"):
        grid.truncated_to(32)
    with pytest.raises(ValueError, match="cannot be combined"):
        SphericalHarmonicTransform.create(12, grid=grid, nlat=22)
    with pytest.raises(ValueError, match=r"52 > 43"):
        SphericalHarmonicTransform.create(21, grid=grid)


def _capped_analysis_is_the_exact_projection(nlat: int, content: int, cap: int):
    # Content of degree `content` on an nlat x 2 nlat grid (T_in = nlat -
    # 1; T_in + content <= 2 nlat - 1): the full coefficients come back
    # from the grid to roundoff, and the capped analysis of the same field
    # on the same grid equals those coefficients restricted to n <= cap --
    # the projection is exact, nothing aliases down from the degrees above
    # the cap.  The zonal cut keeps m <= cap from every column; no column
    # is dropped.  The capped error is pure quadrature roundoff, measured
    # 2026-09-01 at 6.6e-15, 4.0e-14 and 4.0e-14 of the field's maximum
    # for (192, 95, 47), (384, 191, 95) and (768, 383, 191); the full
    # round trip is 4e-15-7e-15 absolute because the Gram-solved analysis
    # is the synthesis's inverse by construction.
    native = GaussianGrid.for_shape(nlat, 2 * nlat)
    full = SphericalHarmonicTransform.create(content, grid=native)
    assert full.grid.shape == (nlat, 2 * nlat)
    rng = np.random.default_rng(content)
    size = content + 1
    coeff = full.project(
        rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    )
    field = full.inverse(coeff)
    back = full.forward(field)
    assert np.max(np.abs(back - coeff)) < 1.0e-13

    capped = SphericalHarmonicTransform.create(cap, grid=native)
    assert capped.grid.shape == (nlat, 2 * nlat)
    assert capped.geometry_hash != full.geometry_hash
    low = capped.forward(field)
    assert low.shape == (cap + 1, cap + 1)
    error = np.max(np.abs(low - coeff[: cap + 1, : cap + 1]))
    assert error < 2.0e-13 * np.max(np.abs(field))
    # ... and the capped transform is a complete transform on that grid:
    # its own round trip and Parseval hold there too.
    check = capped.transform_check(seed=3)
    assert check["roundtrip_relative_linf"] < 5.0e-13
    assert check["parseval_relative_error"] < 5.0e-14


def test_a_capped_analysis_on_a_wider_grid_is_the_exact_projection():
    _capped_analysis_is_the_exact_projection(192, 95, 47)


@pytest.mark.slow
def test_a_capped_t191_analysis_of_t383_content_on_the_768_grid_is_exact():
    # The size the ruling named: T383 content on 768x1536, analysed at
    # T191 and at T383 (767 + 383 = 1150 <= 1535).  Slow-marked for the
    # build, which the per-order Gram solves dominate: 10 s on a quiet
    # host and 350 s on a loaded one (OpenBLAS's spinning threads),
    # measured 2026-09-01.
    _capped_analysis_is_the_exact_projection(768, 383, 191)


@pytest.mark.slow
def test_t767_on_the_native_gfs_grid_builds_and_analyses_within_budget():
    # The verification harness's configuration: T767 (52 km) on the
    # 1536x3072 native grid, an exact projection (767 + 1535 <= 3071).
    # Two packed float64 tables of 3.52 GiB each replace a dense build
    # that held six 7.25 GB squares at its peak.  Measured 2026-09-01
    # (numpy 2.2.6, this tree): 7.032 GiB held after construction, 7.419
    # GiB peak through one forward() (0.30 GiB of it the analysis
    # scratch), 48 s to build on a quiet host.  The bounds below are
    # those figures with headroom, re-taken every run.
    native = GaussianGrid.for_shape(1536, 3072)
    tracemalloc.start()
    transform = SphericalHarmonicTransform.create(767, grid=native)
    held, _ = tracemalloc.get_traced_memory()
    field = np.cos(3.0 * transform.grid.lat_rad)[:, None] * np.cos(
        5.0 * transform.grid.lon_rad
    )[None, :]
    coeff = transform.forward(field)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tables = transform.legendre_table_nbytes
    assert tables["derivative"] == 0
    assert tables["basis"] == _packed_table_bytes(767, 1536, 32, 8)
    assert held < T767_HELD_BOUND_BYTES
    assert peak < T767_PEAK_BOUND_BYTES
    # cos(5 lon) cos(3 lat) has order 5 only; nothing lands elsewhere.
    energy = np.abs(coeff) ** 2
    assert energy[:, 5].sum() > 0.999 * energy.sum()


#: Bounds for the T767 native-grid smoke: measured 7.032 GiB held and
#: 7.419 GiB peak (2026-09-01), stated with ~7% headroom.
T767_HELD_BOUND_BYTES = int(7.5 * 2**30)
T767_PEAK_BOUND_BYTES = int(8.0 * 2**30)
