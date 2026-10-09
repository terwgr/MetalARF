"""Evaluate spectral scalars and winds at arbitrary latitude/longitude points.

The direct evaluator is chunked automatically.  A regular T127 export can
contain hundreds of thousands of points, while a full ``(n,m,point)``
associated-Legendre cube would require many gigabytes.  Chunking changes no
per-point arithmetic order; it only bounds the host workspace.
"""
from __future__ import annotations

import math

import numpy as np

from .legendre import (
    normalized_associated_legendre,
    normalized_associated_legendre_values,
)
from .transform import SphericalHarmonicTransform

# Peak basis construction also holds the raw recurrence.  Scalar sampling
# therefore has roughly two triangular float64 work cubes alive; gradient
# sampling has raw + normalized basis + derivative.  Keep those cubes below a
# conservative 64 MiB by default, independent of target-grid size.
DEFAULT_SAMPLE_WORKSPACE_BYTES = 64 * 1024**2

#: The device path's workspace per point chunk: the packed basis blocks one
#: GEMM contracts (``2K`` rows per block, ``K = (T+1)(T+2)/2``, one block for
#: a scalar and two for a gradient).  512 MiB bounds a T255 float64
#: gradient chunk to about 500 points and a T127 scalar chunk to about
#: 4,000; the float32 (state-precision) blocks take twice the points.
DEFAULT_DEVICE_SAMPLE_WORKSPACE_BYTES = 512 * 1024**2

#: The precisions the device sampler evaluates in.  ``"float64"``: the
#: basis and the contraction in float64 (the host path's arithmetic to
#: rounding, complex128 there).  ``"float32"``: the packed basis rounded to
#: float32 and the contraction as float32 GEMMs over blocks of
#: :data:`STATE_PRECISION_BLOCK` terms whose partial sums are added in
#: float64, so the error is the float32 rounding of the terms and of one
#: block's sum, not of a 33,000-term accumulation.  A float32 state carries
#: 6e-8 of relative quantisation already, so the second is the state's own
#: precision; the point operators choose by ``operator_precision``.
SAMPLE_PRECISIONS = ("float64", "float32")

#: Terms per float32 GEMM block before the partial sums are added in float64.
STATE_PRECISION_BLOCK = 256

#: Threads per block of the basis kernel (one thread per order and point).
_BASIS_THREADS = 128

_BASIS_KERNELS: dict = {}
_RECURRENCE_TABLES: dict = {}


def _device_module(transform: SphericalHarmonicTransform, coeff):
    """The array module of ``coeff`` when it is a device array of the
    transform's backend (cupy), else ``None``: the device path is taken
    only for coefficients that already live on the card."""
    xp = transform.backend.xp
    if xp is np:
        return None
    return xp if isinstance(coeff, xp.ndarray) else None


def _resolve_precision(precision) -> str:
    if precision is None:
        return "float64"
    if precision not in SAMPLE_PRECISIONS:
        raise ValueError(f"sample precision must be one of {SAMPLE_PRECISIONS}, got {precision!r}")
    return str(precision)


def triangle_index(truncation: int) -> tuple[np.ndarray, np.ndarray]:
    """``(n, m)`` of every coefficient inside the triangle ``n >= m`` in the
    packed order the device sampler uses: order-major, ``m = 0..T`` and
    ``n = m..T`` inside each order (the host path's ``c[..., m:, m]``
    slices laid end to end)."""
    t = int(truncation)
    ms = np.concatenate([np.full(t + 1 - m, m, dtype=np.int64) for m in range(t + 1)])
    ns = np.concatenate([np.arange(m, t + 1, dtype=np.int64) for m in range(t + 1)])
    return ns, ms


def _recurrence_tables(xp, truncation: int):
    """The recurrence coefficients of :mod:`legendre`, one value per packed
    ``(n, m)`` (``a``, ``b`` for the two-term upward recurrence at
    ``n >= m + 2``, ``c`` for the derivative at ``n > m``) and per order
    (the diagonal step ``f_m = -sqrt((2m+1)/2m)`` and ``g_m = sqrt(2m+3)``
    for ``P_{m+1,m}``), computed with the host recurrence's own numpy
    expressions so the device basis matches the host one to rounding."""
    key = (id(xp), int(truncation))
    hit = _RECURRENCE_TABLES.get(key)
    if hit is not None:
        return hit
    t = int(truncation)
    n, m = triangle_index(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.sqrt((4.0 * n * n - 1.0) / (n * n - m * m))
        b = np.sqrt(((n - 1.0) * (n - 1.0) - m * m) / (4.0 * (n - 1.0) * (n - 1.0) - 1.0))
        c = np.sqrt((n * n - m * m) * (2.0 * n + 1.0) / (2.0 * n - 1.0))
    a = np.where(n >= m + 2, a, 0.0)
    b = np.where(n >= m + 2, b, 0.0)
    c = np.where(n > m, c, 0.0)
    f = np.zeros(t + 1)
    f[1:] = [-math.sqrt((2 * k + 1) / (2.0 * k)) for k in range(1, t + 1)]
    g = np.array([math.sqrt(2 * k + 3) for k in range(t + 1)], dtype=np.float64)
    tables = tuple(xp.asarray(v, dtype=xp.float64) for v in (a, b, c, f, g))
    _RECURRENCE_TABLES[key] = tables
    return tables


_BASIS_PREAMBLE = r"""
// One thread per (order m, point p), the point index fastest so a warp's
// writes for one packed row land on adjacent columns.  The normalized
// associated Legendre values of the thread's order at the point's
// sin(latitude) by the band recurrence of woof/globe/spectral/legendre.py
// (the diagonal P_mm by the diagonal step from P_00, then P_{m+1,m}, then
// the two-term upward recurrence in n with the exact integer ratios
// precomputed on the host; the derivative from n x P_nm - c P_{n-1,m}),
// every operation a rounded IEEE double so the compiler cannot contract a
// multiply-add and move the last bit away from the host basis.  Each value
// is multiplied by the order's zonal phase (cos, sin of m x longitude,
// tabulated per order and point) and the doubling of the m > 0 orders, and
// written into the packed real blocks one GEMM contracts: row k of the
// first half carries the cosine term, row K + k of the second half the
// sine term, for the packed (n, m) index k of triangle_index().
const long long npts_l = (long long)npts;
const long long pidx = i % npts_l;
const int m = (int)(i / npts_l);
const double xi = x[pidx];
const double root = __dsqrt_rn(fmax(0.0, __dsub_rn(1.0, __dmul_rn(xi, xi))));
const double denom = __dsub_rn(__dmul_rn(xi, xi), 1.0);
const long long kk = (long long)ktri;
double diag = inv_root_4pi;
for (int mm_ = 1; mm_ <= m; ++mm_) diag = __dmul_rn(__dmul_rn(diag_f[mm_], root), diag);
const double cm = cosm[(long long)m * npts_l + pidx];
const double sm = sinm[(long long)m * npts_l + pidx];
const double mult = (m == 0) ? 1.0 : 2.0;
const double mm = mult * (double)m;
// The packed row of (n = m, m): the orders below m fill (t + 1 - m') rows each.
long long k = (long long)m * (long long)(t + 1) - ((long long)m * (long long)(m - 1)) / 2;
double prev2 = 0.0;
double prev = diag;
double dp;
dp = (m >= 1) ? __ddiv_rn(__dmul_rn(root, __dmul_rn(__dmul_rn((double)m, xi), diag)), denom) : 0.0;
EMIT(k, pidx, diag, dp)
++k;
if (m < t) {
    const double p1 = __dmul_rn(__dmul_rn(diag_g[m], xi), diag);
    dp = __ddiv_rn(__dmul_rn(root, __dsub_rn(__dmul_rn(__dmul_rn((double)(m + 1), xi), p1), __dmul_rn(rec_c[k], prev))), denom);
    EMIT(k, pidx, p1, dp)
    ++k;
    prev2 = diag;
    prev = p1;
    for (int n = m + 2; n <= t; ++n) {
        const double pn = __dmul_rn(rec_a[k], __dsub_rn(__dmul_rn(xi, prev), __dmul_rn(rec_b[k], prev2)));
        dp = __ddiv_rn(__dmul_rn(root, __dsub_rn(__dmul_rn(__dmul_rn((double)n, xi), pn), __dmul_rn(rec_c[k], prev))), denom);
        EMIT(k, pidx, pn, dp)
        ++k;
        prev2 = prev;
        prev = pn;
    }
}
"""

_EMIT_SCALAR = r"""
#define EMIT(K_, P_IDX, P_, DP_) \
    b0[(K_) * npts_l + (P_IDX)] = (T)(mult * (P_) * cm); \
    b0[((K_) + kk) * npts_l + (P_IDX)] = (T)(-(mult * (P_) * sm));
"""

_EMIT_GRADIENT = r"""
#define EMIT(K_, P_IDX, P_, DP_) \
    bz[(K_) * npts_l + (P_IDX)] = (T)(-(mm * (P_) * sm)); \
    bz[((K_) + kk) * npts_l + (P_IDX)] = (T)(-(mm * (P_) * cm)); \
    bm[(K_) * npts_l + (P_IDX)] = (T)(mult * (DP_) * cm); \
    bm[((K_) + kk) * npts_l + (P_IDX)] = (T)(-(mult * (DP_) * sm));
"""


def _basis_kernel(xp, gradient: bool):
    """The packed-basis ElementwiseKernel (built once per variant)."""
    key = "gradient" if gradient else "scalar"
    ker = _BASIS_KERNELS.get(key)
    if ker is None:
        common = ("raw float64 x, raw float64 cosm, raw float64 sinm, raw float64 rec_a, raw float64 rec_b, "
                  "raw float64 rec_c, raw float64 diag_f, raw float64 diag_g, float64 inv_root_4pi, "
                  "int32 t, int32 npts, int64 ktri")
        if gradient:
            ker = xp.ElementwiseKernel(common, "raw T bz, raw T bm", _EMIT_GRADIENT + _BASIS_PREAMBLE,
                                       "arwen_packed_legendre_gradient_basis")
        else:
            ker = xp.ElementwiseKernel(common, "raw T b0", _EMIT_SCALAR + _BASIS_PREAMBLE,
                                       "arwen_packed_legendre_scalar_basis")
        _BASIS_KERNELS[key] = ker
    return ker


def packed_basis(xp, truncation: int, lat_rad, lon_rad, *, gradient: bool, dtype=np.float64):
    """The packed real basis blocks of a point chunk, on the device.

    Scalar (``gradient`` False): one ``(2K, p)`` block ``B`` with
    ``B[k] = mult_m P_nm cos(m lambda)`` and ``B[K + k] = -mult_m P_nm
    sin(m lambda)``, so that with ``C = [Re c | Im c]`` packed by
    :func:`triangle_index` the sample is ``C @ B``.  Gradient: the zonal
    block (``-m mult P sin``, ``-m mult P cos``) and the meridional block
    (``mult dP cos``, ``-mult dP sin``); ``C @ Bz`` over the sphere's radius
    times cos(latitude) is the eastward derivative and ``C @ Bm`` over the
    radius the northward one, exactly the host path's contractions with the
    phase folded into the basis.  ``lat_rad``, ``lon_rad`` are host float64.
    """
    t = int(truncation)
    lat = np.asarray(lat_rad, dtype=np.float64).ravel()
    lon = np.asarray(lon_rad, dtype=np.float64).ravel()
    npts = int(lat.size)
    ktri = (t + 1) * (t + 2) // 2
    x = xp.asarray(np.sin(lat))
    lon_d = xp.asarray(lon)
    orders = xp.arange(t + 1, dtype=xp.float64)
    angle = orders[:, None] * lon_d[None, :]
    cosm = xp.cos(angle)
    sinm = xp.sin(angle)
    del angle
    rec_a, rec_b, rec_c, diag_f, diag_g = _recurrence_tables(xp, t)
    inv_root_4pi = 1.0 / math.sqrt(4.0 * math.pi)
    out_dtype = xp.dtype(dtype)
    kernel = _basis_kernel(xp, gradient)
    args = (x, cosm, sinm, rec_a, rec_b, rec_c, diag_f, diag_g, inv_root_4pi,
            np.int32(t), np.int32(npts), np.int64(ktri))
    threads = npts * (t + 1)
    if gradient:
        bz = xp.empty((2 * ktri, npts), dtype=out_dtype)
        bm = xp.empty((2 * ktri, npts), dtype=out_dtype)
        kernel(*args, bz, bm, size=threads, block_size=_BASIS_THREADS)
        return bz, bm
    b0 = xp.empty((2 * ktri, npts), dtype=out_dtype)
    kernel(*args, b0, size=threads, block_size=_BASIS_THREADS)
    return b0


def packed_coefficients(xp, coeff, truncation: int, dtype=np.float64):
    """``[Re c | Im c]`` of ``coeff (..., T+1, T+1)`` over the packed
    triangle, ``(fields, 2K)`` in ``dtype`` (the leading dimensions
    flattened)."""
    t = int(truncation)
    c = xp.asarray(coeff)
    lead = tuple(c.shape[:-2])
    fields = int(np.prod(lead)) if lead else 1
    n_idx, m_idx = triangle_index(t)
    tri = xp.asarray(n_idx * (t + 1) + m_idx)
    flat = c.reshape(fields, (t + 1) * (t + 1))[:, tri]
    packed = xp.empty((fields, 2 * int(tri.size)), dtype=xp.dtype(dtype))
    packed[:, :tri.size] = flat.real
    packed[:, tri.size:] = flat.imag
    return packed, lead


def _contract(xp, packed, block, precision: str):
    """``packed (F, 2K) @ block (2K, p)`` as float64: one GEMM in
    float64, or float32 GEMMs over :data:`STATE_PRECISION_BLOCK`-term
    slices whose partial sums are added in float64."""
    if precision == "float64":
        return packed @ block
    rows = int(block.shape[0])
    out = xp.zeros((int(packed.shape[0]), int(block.shape[1])), dtype=xp.float64)
    for start in range(0, rows, STATE_PRECISION_BLOCK):
        stop = min(rows, start + STATE_PRECISION_BLOCK)
        out += packed[:, start:stop] @ block[start:stop, :]
    return out


def _device_chunk(transform: SphericalHarmonicTransform, *, blocks: int, itemsize: int, requested) -> int:
    if requested is not None:
        return _chunk_points(transform, live_basis_arrays=blocks, requested=requested)
    t = int(transform.truncation)
    ktri = (t + 1) * (t + 2) // 2
    bytes_per_point = max(1, blocks * 2 * ktri * int(itemsize) + 2 * (t + 1) * 8)
    return max(1, DEFAULT_DEVICE_SAMPLE_WORKSPACE_BYTES // bytes_per_point)


def _sample_device(xp, transform: SphericalHarmonicTransform, coeff, latitude_deg, longitude_deg,
                   chunk_points, *, gradient: bool, precision: str | None = None,
                   as_device: bool = False):
    """The device evaluation of :func:`sample_scalar` (``gradient`` False)
    or :func:`sample_gradient` (True) for coefficients resident on the
    card: the packed Legendre basis of every point chunk built by one
    fused kernel (one thread per point, the host recurrence's operations
    in the host's order), the coefficients of every leading dimension
    packed once as ``[Re | Im]`` over the triangle, and ONE real GEMM per
    chunk and output (float64, or float32 over 256-term blocks summed in
    float64 under ``precision = "float32"``), the result handed back as
    numpy.  The arithmetic is the host path's (amplitude per order times
    the zonal phase, the m > 0 orders doubled) with the phase folded into
    the basis before the sum instead of after it.  Why: the previous
    device path contracted one small complex matmul per order per point
    chunk (256 launches a chunk at T255) and promoted the basis slice to
    complex128 each time; measured on the case's real hourly tables (about
    30,000 unique report positions) it took 53 s per control operator call
    and about 17 s per 32-member call at the card's float64 rate
    (2026-09-06, RTX 5090).
    """
    t = int(transform.truncation)
    c = xp.asarray(coeff)
    if c.shape[-2:] != (t + 1, t + 1):
        raise ValueError(
            f"spectral coefficients must end in ({t + 1}, {t + 1}), got {tuple(c.shape)}")
    precision = _resolve_precision(precision)
    lat, lon, point_shape = _points(latitude_deg, longitude_deg)
    coslat = np.cos(lat)
    if gradient and np.any(np.abs(coslat) < 1.0e-12):
        raise ValueError("vector/gradient sampling is undefined at the exact poles")
    gemm_dtype = xp.float64 if precision == "float64" else xp.float32
    packed, lead = packed_coefficients(xp, c, t, dtype=gemm_dtype)
    fields = int(packed.shape[0])
    npts = int(lat.size)
    first = xp.zeros((fields, npts), dtype=xp.float64)
    second = xp.zeros((fields, npts), dtype=xp.float64) if gradient else None
    chunk = _device_chunk(transform, blocks=2 if gradient else 1,
                          itemsize=xp.dtype(gemm_dtype).itemsize, requested=chunk_points)
    for start in range(0, npts, chunk):
        stop = min(npts, start + chunk)
        if gradient:
            bz, bm = packed_basis(xp, t, lat[start:stop], lon[start:stop], gradient=True, dtype=gemm_dtype)
            first[:, start:stop] = _contract(xp, packed, bz, precision)
            del bz
            second[:, start:stop] = _contract(xp, packed, bm, precision)
            del bm
        else:
            b0 = packed_basis(xp, t, lat[start:stop], lon[start:stop], gradient=False, dtype=gemm_dtype)
            first[:, start:stop] = _contract(xp, packed, b0, precision)
            del b0
    del packed
    if as_device:
        if not gradient:
            return first.reshape((*lead, *point_shape))
        east = first / xp.asarray(transform.grid.radius_m * coslat)[None, :]
        north = second / transform.grid.radius_m
        return east.reshape((*lead, *point_shape)), north.reshape((*lead, *point_shape))
    to_numpy = transform.backend.to_numpy
    if not gradient:
        return np.asarray(to_numpy(first)).reshape((*lead, *point_shape))
    east = np.asarray(to_numpy(first)) / (transform.grid.radius_m * coslat)[None, :]
    north = np.asarray(to_numpy(second)) / transform.grid.radius_m
    return east.reshape((*lead, *point_shape)), north.reshape((*lead, *point_shape))


def _points(latitude_deg, longitude_deg) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    lat, lon = np.broadcast_arrays(
        np.asarray(latitude_deg, dtype=np.float64),
        np.asarray(longitude_deg, dtype=np.float64),
    )
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("sample coordinates must be finite")
    if np.any((lat < -90.0) | (lat > 90.0)):
        raise ValueError("latitude must lie in [-90, 90] degrees")
    shape = lat.shape
    return np.deg2rad(lat.ravel()), np.deg2rad(lon.ravel()), shape


def _validate_coeff(transform: SphericalHarmonicTransform, coeff) -> np.ndarray:
    host = transform.backend.to_numpy(coeff).astype(np.complex128, copy=False)
    transform._validate_spectral(host)
    return host


def _chunk_points(
    transform: SphericalHarmonicTransform,
    *,
    live_basis_arrays: int,
    requested: int | None,
) -> int:
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, (int, np.integer)):
            raise ValueError("chunk_points must be a positive integer")
        if int(requested) < 1:
            raise ValueError("chunk_points must be a positive integer")
        return int(requested)
    triangle = (transform.truncation + 1) ** 2
    bytes_per_point = max(1, live_basis_arrays * triangle * 8)
    return max(1, DEFAULT_SAMPLE_WORKSPACE_BYTES // bytes_per_point)


def sample_scalar(
    transform: SphericalHarmonicTransform,
    coeff,
    latitude_deg,
    longitude_deg,
    *,
    chunk_points: int | None = None,
    precision: str | None = None,
    as_device: bool = False,
) -> np.ndarray:
    """Synthesize one or more real scalar fields at arbitrary points.

    Leading coefficient dimensions are retained.  Exact poles are admitted
    for scalars; m>0 harmonics vanish there by the associated-Legendre limit.
    The default chunk size bounds the raw+basis workspace to about 64 MiB.
    ``precision`` (:data:`SAMPLE_PRECISIONS`, default float64) is the
    device path's contraction precision; the host path is complex128.
    ``as_device`` hands the device path's result back on the card (the
    host path always returns numpy).
    """
    xp = _device_module(transform, coeff)
    if xp is not None:
        return _sample_device(xp, transform, coeff, latitude_deg, longitude_deg, chunk_points,
                              gradient=False, precision=precision, as_device=as_device)
    _resolve_precision(precision)
    c = _validate_coeff(transform, coeff)
    lat, lon, point_shape = _points(latitude_deg, longitude_deg)
    result = np.zeros((*c.shape[:-2], lat.size), dtype=np.float64)
    chunk = _chunk_points(transform, live_basis_arrays=2, requested=chunk_points)
    for start in range(0, lat.size, chunk):
        stop = min(lat.size, start + chunk)
        basis = normalized_associated_legendre_values(
            transform.truncation, np.sin(lat[start:stop])
        )
        target = result[..., start:stop]
        for m in range(transform.truncation + 1):
            amplitude = np.einsum(
                "...n,np->...p", c[..., m:, m], basis[m:, m]
            )
            phase = np.exp(1j * m * lon[start:stop])
            if m == 0:
                target += (amplitude * phase).real
            else:
                target += 2.0 * (amplitude * phase).real
    return result.reshape((*c.shape[:-2], *point_shape))


def sample_gradient(
    transform: SphericalHarmonicTransform,
    coeff,
    latitude_deg,
    longitude_deg,
    *,
    chunk_points: int | None = None,
    precision: str | None = None,
    as_device: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Physical eastward/northward scalar gradient at non-polar points.

    The default chunk size bounds the raw+basis+derivative workspace to about
    64 MiB.  ``precision`` as :func:`sample_scalar`.
    """
    xp = _device_module(transform, coeff)
    if xp is not None:
        return _sample_device(xp, transform, coeff, latitude_deg, longitude_deg, chunk_points,
                              gradient=True, precision=precision, as_device=as_device)
    _resolve_precision(precision)
    c = _validate_coeff(transform, coeff)
    lat, lon, point_shape = _points(latitude_deg, longitude_deg)
    coslat = np.cos(lat)
    if np.any(np.abs(coslat) < 1.0e-12):
        raise ValueError("vector/gradient sampling is undefined at the exact poles")
    zonal = np.zeros((*c.shape[:-2], lat.size), dtype=np.float64)
    meridional = np.zeros_like(zonal)
    chunk = _chunk_points(transform, live_basis_arrays=3, requested=chunk_points)
    for start in range(0, lat.size, chunk):
        stop = min(lat.size, start + chunk)
        basis, derivative = normalized_associated_legendre(
            transform.truncation, np.sin(lat[start:stop])
        )
        zonal_target = zonal[..., start:stop]
        meridional_target = meridional[..., start:stop]
        for m in range(transform.truncation + 1):
            phase = np.exp(1j * m * lon[start:stop])
            amp = np.einsum("...n,np->...p", c[..., m:, m], basis[m:, m])
            damp = np.einsum(
                "...n,np->...p", c[..., m:, m], derivative[m:, m]
            )
            multiplier = 1.0 if m == 0 else 2.0
            zonal_target += multiplier * (1j * m * amp * phase).real
            meridional_target += multiplier * (damp * phase).real
    east = zonal / (transform.grid.radius_m * coslat)
    north = meridional / transform.grid.radius_m
    return (
        east.reshape((*c.shape[:-2], *point_shape)),
        north.reshape((*c.shape[:-2], *point_shape)),
    )


def sample_wind(
    transform: SphericalHarmonicTransform,
    vorticity,
    divergence,
    latitude_deg,
    longitude_deg,
    *,
    chunk_points: int | None = None,
    precision: str | None = None,
    as_device: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert vorticity/divergence and sample eastward/northward wind.
    ``precision`` as :func:`sample_scalar`; ``as_device`` hands the device
    path's result back on the card (a host path ignores it)."""
    psi = transform.inverse_laplacian(vorticity)
    chi = transform.inverse_laplacian(divergence)
    psi_east, psi_north = sample_gradient(
        transform,
        psi,
        latitude_deg,
        longitude_deg,
        chunk_points=chunk_points,
        precision=precision,
        as_device=as_device,
    )
    chi_east, chi_north = sample_gradient(
        transform,
        chi,
        latitude_deg,
        longitude_deg,
        chunk_points=chunk_points,
        precision=precision,
        as_device=as_device,
    )
    return -psi_north + chi_east, psi_east + chi_north


def regular_latlon_coordinates(
    nlat: int,
    nlon: int,
    *,
    include_poles: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre regular global coordinates, or pole-inclusive latitudes."""
    if isinstance(nlat, bool) or isinstance(nlon, bool):
        raise ValueError("regular-grid dimensions must be integers")
    ny, nx = int(nlat), int(nlon)
    if ny != nlat or nx != nlon:
        raise ValueError("regular-grid dimensions must be integers")
    if ny < 2 or nx < 4:
        raise ValueError("regular grid needs nlat>=2 and nlon>=4")
    if include_poles:
        lat = np.linspace(-90.0, 90.0, ny, dtype=np.float64)
    else:
        lat = -90.0 + (np.arange(ny, dtype=np.float64) + 0.5) * 180.0 / ny
    lon = np.arange(nx, dtype=np.float64) * 360.0 / nx
    return lat, lon


def regular_latlon_scalar(
    transform: SphericalHarmonicTransform,
    coeff,
    *,
    nlat: int,
    nlon: int,
    include_poles: bool = False,
    chunk_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat, lon = regular_latlon_coordinates(
        nlat, nlon, include_poles=include_poles
    )
    lon2, lat2 = np.meshgrid(lon, lat)
    return lat, lon, sample_scalar(
        transform, coeff, lat2, lon2, chunk_points=chunk_points
    )


__all__ = [
    "DEFAULT_DEVICE_SAMPLE_WORKSPACE_BYTES",
    "DEFAULT_SAMPLE_WORKSPACE_BYTES",
    "SAMPLE_PRECISIONS",
    "STATE_PRECISION_BLOCK",
    "packed_basis",
    "packed_coefficients",
    "regular_latlon_coordinates",
    "regular_latlon_scalar",
    "sample_gradient",
    "sample_scalar",
    "sample_wind",
    "triangle_index",
]
