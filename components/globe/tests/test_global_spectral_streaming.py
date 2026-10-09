"""The streamed transform: no resident Legendre table, the same bits.

``forward_streaming`` / ``inverse_streaming`` and a transform created
with ``streaming=True`` regenerate the Legendre basis one chunk of orders
at a time and discard it, so an analysis runs in O(order_chunk x (T+1)
x nlat) memory whatever the truncation: T1534 on the 1536x3072 native
GFS grid, whose dense tables (28.9 GB each) OOM-killed the analysis
process at 120 GB on 2026-09-01, streams at 1.89 GiB peak RSS in 159 s
(measured 2026-09-01, numpy 2.2.6, 16-core Zen 5, chunk 32; T767 on
the same grid 0.99 GiB and 39-79 s across two runs, against 7.03 GiB
held by the resident tables).  The bits are
the resident table's because the recurrence, the per-order Gram solve,
the dtype cast, the dense expansion and the per-order GEMM are the same
functions in the same order.
"""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from woof.globe.spectral.grid import GaussianGrid
from woof.globe.spectral.transform import SphericalHarmonicTransform
from woof.globe.spectral.vector import VorticityDivergenceOperator


def _same(a, b) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def _relative_linf(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _random_coeff(transform, rng, *lead):
    t1 = transform.truncation + 1
    return transform.project(
        rng.standard_normal((*lead, t1, t1)) + 1j * rng.standard_normal((*lead, t1, t1))
    )


@pytest.mark.parametrize("truncation,precision", [(127, "float64"), (63, "float32")])
@pytest.mark.parametrize("order_chunk", [1, 5, 32, 128])
def test_forward_streaming_returns_the_resident_table_s_bits(
    truncation, precision, order_chunk
):
    # The requirement was <= 1e-13 relative at T127 float64; the measured
    # difference is exactly zero for every chunk, so the test demands the
    # bits (the relative figure is asserted too, so a host where the two
    # ever part company reports the size of the gap, not just its
    # existence).
    transform = SphericalHarmonicTransform.create(truncation, precision=precision)
    rng = np.random.default_rng(truncation)
    for lead in ((), (3,)):
        field = rng.standard_normal((*lead, *transform.grid.shape))
        resident = transform.forward(field)
        streamed = transform.forward_streaming(field, order_chunk=order_chunk)
        assert streamed.dtype == resident.dtype
        assert _relative_linf(streamed, resident) <= 1e-13
        assert _same(streamed, resident), f"chunk {order_chunk} lead {lead}"


@pytest.mark.parametrize("truncation,precision", [(127, "float64"), (63, "float32")])
@pytest.mark.parametrize("order_chunk", [1, 7, 32])
def test_inverse_streaming_returns_the_resident_table_s_bits(
    truncation, precision, order_chunk
):
    transform = SphericalHarmonicTransform.create(truncation, precision=precision)
    rng = np.random.default_rng(truncation + 1)
    for lead in ((), (2, 3)):
        coeff = _random_coeff(transform, rng, *lead)
        resident = transform.inverse(coeff)
        streamed = transform.inverse_streaming(coeff, order_chunk=order_chunk)
        assert streamed.dtype == resident.dtype
        assert _same(streamed, resident), f"chunk {order_chunk} lead {lead}"


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_a_streaming_transform_holds_no_table_and_every_operation_matches(precision):
    truncation = 63
    resident = SphericalHarmonicTransform.create(truncation, precision=precision)
    streaming = SphericalHarmonicTransform.create(
        truncation, precision=precision, streaming=True, legendre_band=11
    )
    # Streaming changes where the table lives, not the arithmetic, so a
    # streamed transform shares the identity of the resident one AT THE
    # SAME BAND -- and the band is not free, because under cupy it is the
    # strided-batched GEMM's batch count and MEASURED to move bits
    # (2026-09-06, RTX 5070 Ti, T255 float32).
    same_band = SphericalHarmonicTransform.create(
        truncation, precision=precision, legendre_band=11
    )
    assert streaming.identity_hash == same_band.identity_hash
    assert streaming.identity_hash != resident.identity_hash
    assert "streaming" not in streaming.identity
    assert SphericalHarmonicTransform.create(
        truncation, precision=precision, streaming=True
    ).identity_hash == resident.identity_hash
    assert streaming.legendre_table_nbytes == {
        "basis": 0, "analysis": 0, "derivative": 0, "scratch": 0,
    }
    rng = np.random.default_rng(7)
    nlat, nlon = resident.grid.shape
    field = rng.standard_normal((2, nlat, nlon))
    coeff = _random_coeff(resident, rng, 2)
    u = rng.standard_normal((nlat, nlon))
    v = rng.standard_normal((nlat, nlon))
    assert _same(streaming.forward(field), resident.forward(field))
    assert _same(streaming.inverse(coeff), resident.inverse(coeff))
    for have, want in zip(streaming.gradient(coeff), resident.gradient(coeff), strict=True):
        assert _same(have, want)
    assert _same(
        streaming.inverse_zonal_derivative(coeff), resident.inverse_zonal_derivative(coeff)
    )
    s_op = VorticityDivergenceOperator(streaming)
    r_op = VorticityDivergenceOperator(resident)
    for have, want in zip(s_op.vordiv_from_wind(u, v), r_op.vordiv_from_wind(u, v), strict=True):
        assert _same(have, want)
    for have, want in zip(
        s_op.wind_from_vordiv(coeff[0], coeff[1]),
        r_op.wind_from_vordiv(coeff[0], coeff[1]),
        strict=True,
    ):
        assert _same(have, want)
    # Nothing was cached along the way: the derivative table in particular.
    assert streaming._dbasis is None
    assert streaming.legendre_table_nbytes["derivative"] == 0
    assert streaming.legendre_table_nbytes["scratch"] == 0


def _streaming_peak_bytes(transform, field, order_chunk: int) -> int:
    tracemalloc.start()
    transform.forward_streaming(field, order_chunk=order_chunk)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def _field_sized_allowance(transform) -> int:
    """Bytes the analysis holds beyond the chunk: the field, its Fourier
    transform, the stacked real/imaginary inputs and outputs (eight
    field-sized float64 buffers cover them), and the per-order Gram
    solve's temporaries (four (T+1) x nlat rows)."""
    nlat, nlon = transform.grid.shape
    return 8 * nlat * nlon * 8 + 4 * (transform.truncation + 1) * nlat * 8


def test_streaming_memory_is_one_order_chunk():
    # The claim is O(order_chunk x (T+1) x nlat), priced by
    # streaming_working_bytes: three chunk-sized blocks under numpy
    # float64 (recurrence, Gram-solved rows, dense expansion).  Measured
    # 2026-09-01 at T127 (192 x 384): 4.1 MiB peak for a chunk of 4 and
    # 19.9 MiB for 32 against a resident table set of 30.0 MiB, the
    # priced blocks being 2.2 MiB and 18.0 MiB.
    transform = SphericalHarmonicTransform.create(127)
    field = np.random.default_rng(2).standard_normal(transform.grid.shape)
    allowance = _field_sized_allowance(transform)
    peaks = {}
    for chunk in (4, 32):
        peaks[chunk] = _streaming_peak_bytes(transform, field, chunk)
        priced = sum(transform.streaming_working_bytes(chunk).values())
        assert peaks[chunk] <= priced + allowance, (
            f"chunk {chunk}: peak {peaks[chunk]} exceeds the priced "
            f"{priced} + allowance {allowance}"
        )
    resident = sum(transform.legendre_table_nbytes.values())
    assert peaks[32] < resident
    assert peaks[4] < peaks[32] / 3
    # The float32 pricing carries the cast copy and the narrower scratch.
    narrow = SphericalHarmonicTransform.create(127, precision="float32")
    priced = narrow.streaming_working_bytes(32)
    assert priced["device"] == priced["scratch"] == priced["recurrence"] // 2
    assert priced["analysis"] == priced["recurrence"]


def test_the_order_chunk_is_validated():
    transform = SphericalHarmonicTransform.create(15)
    field = np.zeros(transform.grid.shape)
    with pytest.raises(ValueError, match="order_chunk must be >= 1"):
        transform.forward_streaming(field, order_chunk=0)
    with pytest.raises(ValueError, match="legendre_band must be >= 1"):
        SphericalHarmonicTransform.create(15, streaming=True, legendre_band=0)


def test_streaming_analysis_on_the_native_grid_is_the_exact_projection():
    # 96 x 192 carries T95 content; T47 is an exact projection there
    # (95 + 47 <= 191).  The streamed analysis returns the full
    # coefficients restricted to n <= 47: the same claim the resident
    # capped analysis makes, here without a table.
    native = GaussianGrid.for_shape(96, 192)
    full = SphericalHarmonicTransform.create(95, grid=native)
    rng = np.random.default_rng(11)
    coeff = _random_coeff(full, rng)
    field = full.inverse(coeff)
    capped = SphericalHarmonicTransform.create(47, grid=native, streaming=True)
    got = capped.forward(field)
    want = coeff[:48, :48]
    assert np.max(np.abs(got - want)) <= 2e-13 * np.max(np.abs(field))


#: Bound for the T767 native-grid streamed analysis: measured 0.936 GiB
#: tracemalloc peak (2026-09-01), stated with ~17% headroom.
T767_STREAMING_PEAK_BOUND_BYTES = int(1.1 * 2 ** 30)


@pytest.mark.slow
def test_t767_on_the_native_gfs_grid_streams_within_one_chunk():
    # The instrument's configuration without a table: T767 on 1536x3072.
    # The resident build holds 7.03 GiB; the stream holds three 0.28 GiB
    # chunk blocks and the field-sized buffers -- 0.936 GiB tracemalloc
    # peak, 0.99 GiB process RSS, measured 2026-09-01.  Re-taken every
    # run against the bound above.
    native = GaussianGrid.for_shape(1536, 3072)
    transform = SphericalHarmonicTransform.create(767, grid=native, streaming=True)
    field = np.cos(3.0 * transform.grid.lat_rad)[:, None] * np.cos(
        5.0 * transform.grid.lon_rad
    )[None, :]
    tracemalloc.start()
    coeff = transform.forward(field)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert sum(transform.legendre_table_nbytes.values()) == 0
    assert peak < T767_STREAMING_PEAK_BOUND_BYTES
    energy = np.abs(coeff) ** 2
    assert energy[:, 5].sum() > 0.999 * energy.sum()
