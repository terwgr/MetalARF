"""Orthonormal associated-Legendre basis and meridional derivatives.

The basis is generated in BANDS of consecutive orders.  A band holding
orders ``m0 <= m < m1`` is one dense ``(m1 - m0, T + 1 - m0, nlat)`` block
whose entry ``[i, r, j]`` is ``N_nm P_n^m(x_j)`` for ``m = m0 + i`` and
``n = m0 + r``; the entries with ``n < m`` (``r < i``) are the zeros the
triangle has there.  A full ``(T+1, T+1, nlat)`` square carries a zero for
every ``n < m`` -- half the table -- and at T1534 on a 1536-latitude grid
that half is 14.5 GB per table (measured 2026-09-01, the analysis process
reaching 120 GB and being OOM-killed on a 128 GB host).  Bands keep the
waste at ``(band - 1) / 2`` columns per order, 5.8% at T533 for the
default band of 32 and 2% at T1534, while every order still lives in a
dense stack a batched GEMM can read.

The recurrence runs on the normalized values.  The unnormalized form
carries a ``(2m-1)!!`` diagonal that overflows float64 near m=150 while
``config.MAXIMUM_TRUNCATION`` admits truncation up to 255; the normalized
values stay below ``sqrt((2n+1)/(4pi))`` for every admitted degree.
Condon-Shortley phase comes from the negated diagonal step.

Every arithmetic expression here is elementwise and keeps the operand
order of the per-(n, m) scalar loop it replaced, so the vectorised band
produces the same bits the scalar loop did (pinned by
``tests/test_global_spectral_bit_identity.py``).
"""
from __future__ import annotations

import math

import numpy as np

#: Orders per band.  Under numpy the band is a memory/launch layout only:
#: the contraction is one BLAS GEMM per order whatever the band (numpy's
#: batched matmul loops over the leading axis), so the band never touches
#: the arithmetic.  Under cupy a band is one strided-batched GEMM.
DEFAULT_BAND = 32


def band_bounds(truncation: int, band: int = DEFAULT_BAND) -> list[tuple[int, int]]:
    """``[(m0, m1), ...]`` covering orders ``0..truncation`` in bands."""
    t = int(truncation)
    b = int(band)
    if b < 1:
        raise ValueError(f"band must be >= 1, got {band}")
    return [(m0, min(m0 + b, t + 1)) for m0 in range(0, t + 1, b)]


def packed_table_elements(truncation: int, nlat: int, band: int = DEFAULT_BAND) -> int:
    """Elements one banded table holds: ``sum over bands of orders x (T+1-m0) x nlat``.

    The memory estimator prices from this, not from its own copy of the
    band arithmetic, so the two cannot drift apart.
    """
    t = int(truncation)
    return sum((m1 - m0) * (t + 1 - m0) * int(nlat) for m0, m1 in band_bounds(t, band))


def expansion_scratch_elements(truncation: int, nlat: int, band: int = DEFAULT_BAND) -> int:
    """Elements of one table's dense-expansion scratch: ``band x (T+1) x nlat``."""
    return int(band) * (int(truncation) + 1) * int(nlat)


def _validated_x(sin_lat, *, strict: bool) -> np.ndarray:
    x = np.asarray(sin_lat, dtype=np.float64)
    if strict:
        if x.ndim != 1 or np.any(np.abs(x) >= 1.0):
            raise ValueError(
                "sin_lat derivatives require a one-dimensional array strictly "
                "inside (-1, 1)"
            )
    elif x.ndim != 1 or np.any(np.abs(x) > 1.0):
        raise ValueError("sin_lat must be a one-dimensional array in [-1, 1]")
    return x


def legendre_bands(truncation: int, sin_lat, band: int = DEFAULT_BAND):
    """Yield ``(m0, m1, block)`` for the orders of ``0..truncation`` in bands.

    ``block[i, r, j] = N_nm P_n^m(sin_lat[j])`` with ``m = m0 + i`` and
    ``n = m0 + r``, zero where ``n < m``.  Each block is a fresh float64
    array the caller owns; nothing from earlier bands is retained beyond
    the running diagonal ``P_mm``.
    """
    t = int(truncation)
    x = _validated_x(sin_lat, strict=False)
    root = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    diagonal = np.full(x.shape, 1.0 / math.sqrt(4.0 * math.pi))
    for m0, m1 in band_bounds(t, band):
        count = m1 - m0
        block = np.zeros((count, t + 1 - m0, x.size), dtype=np.float64)
        for i in range(count):
            m = m0 + i
            if m > 0:
                diagonal = -math.sqrt((2 * m + 1) / (2.0 * m)) * root * diagonal
            block[i, i] = diagonal
            if m < t:
                block[i, i + 1] = math.sqrt(2 * m + 3) * x * diagonal
        orders = np.arange(m0, m1, dtype=np.int64)
        for n in range(m0 + 2, t + 1):
            # Orders m <= n - 2 advance together: the coefficients are exact
            # integer ratios under one division and one square root, the
            # same operations the scalar loop performed per (n, m).
            k = min(count, n - m0 - 1)
            r = n - m0
            ms = orders[:k]
            a = np.sqrt((4.0 * n * n - 1.0) / (n * n - ms * ms))
            b = np.sqrt(
                ((n - 1.0) * (n - 1.0) - ms * ms)
                / (4.0 * (n - 1.0) * (n - 1.0) - 1.0)
            )
            block[:k, r] = a[:, None] * (
                x[None, :] * block[:k, r - 1] - b[:, None] * block[:k, r - 2]
            )
        yield m0, m1, block
        # The generator frame must not keep the yielded band alive while
        # the caller contracts it: a streamed transform holds one chunk.
        del block


def legendre_derivative_block(sin_lat, m0: int, block: np.ndarray) -> np.ndarray:
    """``dP/dphi`` for one band block, same shape and zero pattern.

    ``(x^2-1) dP/dx = n x P_nm - c_nm P_{n-1,m}`` on normalized values, with
    ``c_nm`` carrying the ``N_nm/N_{n-1,m}`` normalization ratio; the
    ``n = m`` row has no ``P_{n-1,m}`` term.  Requires non-polar latitudes.
    """
    x = _validated_x(sin_lat, strict=True)
    count, width, _ = block.shape
    deriv = np.zeros_like(block)
    denom = x * x - 1.0
    coslat = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    for i in range(count):
        m = m0 + i
        # The n = m row (r = i): numerator = n x P_mm, skipped for n = 0.
        if m >= 1:
            deriv[i, i] = coslat * (m * x * block[i, i]) / denom
        if i + 1 >= width:
            continue
        n = np.arange(m + 1, m0 + width, dtype=np.int64)
        c = np.sqrt((n * n - m * m) * (2.0 * n + 1.0) / (2.0 * n - 1.0))
        numerator = (
            n[:, None] * x[None, :] * block[i, i + 1:]
            - c[:, None] * block[i, i:-1]
        )
        deriv[i, i + 1:] = coslat[None, :] * numerator / denom[None, :]
    return deriv


def _assemble(truncation: int, x: np.ndarray, blocks) -> np.ndarray:
    t = int(truncation)
    table = np.zeros((t + 1, t + 1, x.size), dtype=np.float64)
    for m0, m1, block in blocks:
        for i in range(m1 - m0):
            m = m0 + i
            table[m:, m, :] = block[i, i:, :]
    return table


def normalized_associated_legendre_values(
    truncation: int, sin_lat: np.ndarray
) -> np.ndarray:
    """Orthonormal ``N_nm P_n^m(sin(phi))`` as a full ``(n, m, j)`` square.

    Includes poles.  This is the point-sampling form (``sampling.py`` reads
    ``[n, m]`` slices of it for arbitrary point chunks); the transform
    itself never builds the square and reads :func:`legendre_bands`.
    """
    x = _validated_x(sin_lat, strict=False)
    return _assemble(truncation, x, legendre_bands(truncation, x))


def normalized_associated_legendre(
    truncation: int, sin_lat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(P, dP_dphi)`` as full ``(n, m, j)`` squares.

    ``P[n,m,j]`` is ``N_nm P_n^m(sin(phi_j))`` with the Condon--Shortley
    phase included by the recurrence.  The full harmonic is
    ``Y_nm = P[n,m] * exp(i*m*lambda)`` and integrates to one over the sphere.
    Entries with m > n are zero.  Derivatives require non-polar latitudes;
    scalar-only sampling at a pole uses
    :func:`normalized_associated_legendre_values` instead.
    """
    x = _validated_x(sin_lat, strict=True)
    basis_blocks = []
    deriv_blocks = []
    for m0, m1, block in legendre_bands(truncation, x):
        basis_blocks.append((m0, m1, block))
        deriv_blocks.append((m0, m1, legendre_derivative_block(x, m0, block)))
    return _assemble(truncation, x, basis_blocks), _assemble(truncation, x, deriv_blocks)
