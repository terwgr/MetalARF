"""Positive-definite flux-form transport of the grid-point tracers.

The five condensate species and the five number moments live on the
Gaussian grid, not in the spectral basis (constants.GRID_TRACERS).  A
truncated spectral representation of a rain shaft or a cloud edge rings
below zero, and the clip-and-rescale that kept those fields nonnegative
moved 37 percent of the planet's cloud water and 9-32 percent of its
rain, ice and graupel out of the columns that held them on EVERY repair
pass (four per step), into clear air where the microphysics evaporated
it; total surface precipitation equalled convective precipitation to
four decimals and grid-scale rain never reached the ground (finding
2026-09-02).  Vapor is smooth enough for the spectral basis (its ringing
fraction measured 0.00 percent in every pass) and stays spectral.

This module carries the grid tracers with the classic finite-volume
scheme every operational lat-lon or reduced-grid model uses for
condensate: a directionally split, conservative flux form on the
Gaussian cells, van Leer (harmonic-mean) limited linear reconstruction
with the (1 - C) time-centring of the upwind face, and a pseudo-density
that is advanced by the same fluxes as the tracer mass so that a
uniform mixing ratio stays uniform to roundoff under any divergent
flow.  Positivity holds by construction: every face value is bounded by
the adjacent cell values and by twice the upwind value, and each sweep
is sub-cycled so no face carries more than ``courant_limit`` (0.25) of
the upwind cell per sub-step, so the two outflow faces of a cell can
take at most the cell's own mass.  Rows near the poles, where the zonal
Courant number of a 52 km run exceeds one (330 m cells at the first
Gaussian ring of T255), are sub-cycled as a contiguous polar band; every
other row runs one sweep.  The meridional faces at the poles carry
exactly zero flux (their length ``cos(lat)`` is zero at ``|mu| = 1``),
which closes the polar cap without a special case.

Cells: ring ``j`` spans ``mu`` (sin latitude) from ``-1 + sum(w[:j])`` to
``-1 + sum(w[:j+1])`` with ``w`` the Gaussian quadrature weights, so the
cell areas are the weights the model's own global integrals use and the
scheme conserves exactly the area-weighted mass those integrals
measure.  The zonal cell width is one longitude step; the meridional
reconstruction is linear in ``mu``; the vertical reconstruction is
linear in pressure between the full levels, the coordinate the model's
spectral scalar advection already uses (DN-4).

Winds: the caller supplies cell-centred mass fluxes ``dp*u``, ``dp*v``
(Pa m/s) and the half-level pressure velocity (Pa/s), time-averaged over
the step by the dynamics; interface fluxes are the arithmetic mean of
the adjacent cell values.  The tracer mass and the pseudo-density are
advanced by the same interface fluxes, so ``q = mass / density`` is
exact for a constant; the pseudo-density's end value differs from the
spectral continuity's ``dp`` by the two discretizations' truncation
difference, which is what the mixing ratio then rides in.  The scheme
alternates its sweep order (x y z, then z y x) on consecutive steps so
the splitting error does not accumulate with one sign.

LAYOUT AND THE FUSED SWEEPS.  The ten tracers travel as ONE stacked
``(ntracer, nlev, nlat, nlon)`` mass array through every sweep, so each
elementwise operation of the scheme is one launch for all ten instead
of ten.  The numpy expressions on that stack are the specification: per
element they are the same operations, in the same order, the per-tracer
loop of the first version computed, so the stacked numpy path returns
the bits that loop returned.  On the cupy backend each sub-step of a
sweep is ONE fused kernel per sweep (``_fused``): the same expressions,
in the same association, with multiply-add contraction disabled
(``-fmad=false``) so every intermediate is rounded exactly where the
numpy specification rounds it, and IEEE division and comparison
throughout.  The fused path is bit-identical to the numpy specification
on the same float32 inputs (tests pin it on random and adversarial
fields when a device is present) and to the unfused ten-tracer loop it
replaced (the 10-step T255 case, 2026-09-04).  Why: the unfused polar
band ran twenty-four sub-steps of about twenty elementwise launches per
tracer, each launch streaming the whole band through device memory
once, 0.14 s of a 0.72 s step on the RTX 5090; the fused sweep reads
the band's mass, density and flux once per sub-step.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .spill import prefetch, resident
from .bands import (
    BandPipeline,
    LatitudeAccumulator,
    associative_over as _associative_over,
)
from .spectral.backend import device_cache_key
from .constants import GRAVITY_M_S2
from .profile import profiler_of


def _intersect(piece: slice, bands) -> list[slice]:
    """The row ranges a sweep piece and the band schedule share.

    Both cuts are contiguous runs of latitude rows, so their overlap is
    one run per band and the runs tile the piece exactly.
    """
    out = []
    for rows in bands:
        start = max(piece.start, rows.start)
        stop = min(piece.stop, rows.stop)
        if stop > start:
            out.append(slice(start, stop))
    return out


def _idx(axis: int, s) -> tuple:
    """Index tuple selecting ``s`` along ``axis`` of a 3-D (nlev, nlat,
    nlon) array, or along ``axis + 1`` of a stacked 4-D tracer array
    when ``s`` is applied through :func:`_tidx`."""
    index = [slice(None), slice(None), slice(None)]
    index[axis] = s
    return tuple(index)


def _tidx(axis: int, s) -> tuple:
    """:func:`_idx` for the stacked ``(ntracer, nlev, nlat, nlon)`` mass."""
    return (slice(None), *_idx(axis, s))


# ---------------------------------------------------------------- fused

_FUSED_PREAMBLE = r"""
// The van Leer harmonic-mean limiter, the numpy _harmonic specification:
//   product = left * right; monotone = product > 0
//   denominator = monotone ? left + right : 1
//   result = monotone ? 2 * product / denominator : 0
template <typename T>
__device__ __forceinline__ T arwen_harmonic(T left, T right) {
    T product = left * right;
    bool monotone = product > T(0);
    T denominator = monotone ? (left + right) : T(1);
    return monotone ? ((T(2) * product) / denominator) : T(0);
}
// numpy minimum / maximum on finite operands.
template <typename T>
__device__ __forceinline__ T arwen_min(T a, T b) { return (a < b) ? a : b; }
template <typename T>
__device__ __forceinline__ T arwen_max(T a, T b) { return (a > b) ? a : b; }
"""

# One sub-step of the periodic zonal sweep for every tracer of the stack.
# Element i runs over (ntracer, n0, n1, n2); the flux of face j (between
# cells j and j+1 mod n2) is formed twice per cell, once for its east
# face and once for its west, exactly as the specification forms the
# whole face array and differences it.
_PERIODIC_OPERATION = r"""
    const long long cell = i % ncell;           // index into density/psi
    const long long tbase = i - cell;           // this tracer's block
    const long long j = cell % n2;
    const long long row = cell - j;             // start of this ring
    // q at ring position p (periodic).
    #define ARWEN_Q(p) (mass[tbase + row + (p)] / density[row + (p)])
    const long long jm2 = (j + n2 - 2) % n2;
    const long long jm1 = (j + n2 - 1) % n2;
    const long long jp1 = (j + 1) % n2;
    const long long jp2 = (j + 2) % n2;
    const T q_m2 = ARWEN_Q(jm2);
    const T q_m1 = ARWEN_Q(jm1);
    const T q_0 = ARWEN_Q(j);
    const T q_p1 = ARWEN_Q(jp1);
    const T q_p2 = ARWEN_Q(jp2);
    #undef ARWEN_Q
    // dq[p] = q[p+1] - q[p]; slope[p] = H(dq[p-1], dq[p])
    const T dq_m2 = q_m1 - q_m2;
    const T dq_m1 = q_0 - q_m1;
    const T dq_0 = q_p1 - q_0;
    const T dq_p1 = q_p2 - q_p1;
    const T slope_m1 = arwen_harmonic<T>(dq_m2, dq_m1);
    const T slope_0 = arwen_harmonic<T>(dq_m1, dq_0);
    const T slope_p1 = arwen_harmonic<T>(dq_0, dq_p1);
    // face f between cells f and f+1: flux_dt = psi[f] * dt
    T flux_here, flux_west;
    {
        const T fdt = psi[row + j] * dt;
        const T c_lo = fdt / density[row + j];
        const T c_hi = (-fdt) / density[row + jp1];
        const bool upwind_lo = psi[row + j] >= T(0);
        const T w_lo = T(0.5) * (T(1) - c_lo);
        const T w_hi = T(-0.5) * (T(1) - c_hi);
        T face_value = upwind_lo ? (q_0 + slope_0 * w_lo) : (q_p1 + slope_p1 * w_hi);
        const T q_up = upwind_lo ? q_0 : q_p1;
        face_value = arwen_min<T>(
            arwen_max<T>(face_value, arwen_min<T>(q_0, q_p1)),
            arwen_min<T>(arwen_max<T>(q_0, q_p1), T(2) * q_up));
        flux_here = fdt * face_value;
    }
    {
        const T fdt = psi[row + jm1] * dt;
        const T c_lo = fdt / density[row + jm1];
        const T c_hi = (-fdt) / density[row + j];
        const bool upwind_lo = psi[row + jm1] >= T(0);
        const T w_lo = T(0.5) * (T(1) - c_lo);
        const T w_hi = T(-0.5) * (T(1) - c_hi);
        T face_value = upwind_lo ? (q_m1 + slope_m1 * w_lo) : (q_0 + slope_0 * w_hi);
        const T q_up = upwind_lo ? q_m1 : q_0;
        face_value = arwen_min<T>(
            arwen_max<T>(face_value, arwen_min<T>(q_m1, q_0)),
            arwen_min<T>(arwen_max<T>(q_m1, q_0), T(2) * q_up));
        flux_west = fdt * face_value;
    }
    out = mass[i] - (flux_here - flux_west);
    // The pseudo-density advanced by the same face fluxes, the
    // specification's density - (flux_dt - roll(flux_dt, 1)), written
    // once per cell by the first tracer's thread.
    if (tbase == 0) {
        const T fdt_here = psi[row + j] * dt;
        const T fdt_west = psi[row + jm1] * dt;
        out_density[cell] = density[cell] - (fdt_here - fdt_west);
    }
"""

# One sub-step of a walled sweep along ``axis`` for every tracer of the
# stack.  ``psi`` holds the n-1 interior faces, ``face`` the n+1 face
# coordinates, ``centre``/``inv_width`` the cells; the three coordinate
# arrays are full (n0, n1, n2) arrays (broadcast by the caller).
_WALLED_OPERATION = r"""
    const long long cell = i % ncell;
    const long long tbase = i - cell;
    const long long k2 = cell % n2;
    const long long k1 = (cell / n2) % n1;
    const long long k0 = cell / (n2 * n1);
    const long long k = (axis == 0) ? k0 : ((axis == 1) ? k1 : k2);
    const long long n = (axis == 0) ? n0 : ((axis == 1) ? n1 : n2);
    const long long stride = (axis == 0) ? (n1 * n2) : ((axis == 1) ? n2 : 1);
    // Flat index of the cell at position p along the axis, other
    // coordinates fixed; and of face p in the (n-1)-face psi array and
    // the (n+1)-face face array.
    const long long cell0 = cell - k * stride;
    #define ARWEN_CELL(p) (cell0 + (p) * stride)
    const long long psi_stride = (axis == 0) ? (n1 * n2) : ((axis == 1) ? n2 : 1);
    const long long psi0 = (axis == 0) ? (k1 * n2 + k2)
                         : ((axis == 1) ? (k0 * (n1 - 1) * n2 + k2)
                                        : (k0 * n1 * (n2 - 1) + k1 * (n2 - 1)));
    const long long face0 = (axis == 0) ? (k1 * n2 + k2)
                          : ((axis == 1) ? (k0 * (n1 + 1) * n2 + k2)
                                         : (k0 * n1 * (n2 + 1) + k1 * (n2 + 1)));
    #define ARWEN_PSI(f) psi[psi0 + (f) * psi_stride]
    #define ARWEN_FACE(f) face[face0 + (f) * psi_stride]
    #define ARWEN_Q(p) (mass[tbase + ARWEN_CELL(p)] / density[ARWEN_CELL(p)])
    #define ARWEN_C(p) centre[ARWEN_CELL(p)]
    // slope at cell p: zero on the two wall cells, H(dq[p-1], dq[p]) inside,
    // dq[f] = (q[f+1] - q[f]) / (centre[f+1] - centre[f]).
    #define ARWEN_DQ(f) ((ARWEN_Q((f) + 1) - ARWEN_Q(f)) / (ARWEN_C((f) + 1) - ARWEN_C(f)))
    T out_value = mass[i];
    if (k <= n - 2) {
        // The east/north/down face f = k between cells k and k+1.
        const long long f = k;
        const T fdt = ARWEN_PSI(f) * dt;
        const T c_lo = (fdt * inv_width[ARWEN_CELL(f)]) / density[ARWEN_CELL(f)];
        const T c_hi = ((-fdt) * inv_width[ARWEN_CELL(f + 1)]) / density[ARWEN_CELL(f + 1)];
        const bool upwind_lo = ARWEN_PSI(f) >= T(0);
        const T d_face = ARWEN_FACE(f + 1);
        const T w_lo = (d_face - ARWEN_C(f)) * (T(1) - c_lo);
        const T w_hi = (d_face - ARWEN_C(f + 1)) * (T(1) - c_hi);
        const T q_lo = ARWEN_Q(f);
        const T q_hi = ARWEN_Q(f + 1);
        const T slope_lo = (f >= 1) ? arwen_harmonic<T>(ARWEN_DQ(f - 1), ARWEN_DQ(f)) : T(0);
        const T slope_hi = (f + 1 <= n - 2) ? arwen_harmonic<T>(ARWEN_DQ(f), ARWEN_DQ(f + 1)) : T(0);
        T face_value = upwind_lo ? (q_lo + slope_lo * w_lo) : (q_hi + slope_hi * w_hi);
        const T q_up = upwind_lo ? q_lo : q_hi;
        face_value = arwen_min<T>(
            arwen_max<T>(face_value, arwen_min<T>(q_lo, q_hi)),
            arwen_min<T>(arwen_max<T>(q_lo, q_hi), T(2) * q_up));
        const T tracer_flux = fdt * face_value;
        out_value = out_value - tracer_flux * inv_width[cell];
    }
    if (k >= 1) {
        // The west/south/up face f = k-1 between cells k-1 and k.
        const long long f = k - 1;
        const T fdt = ARWEN_PSI(f) * dt;
        const T c_lo = (fdt * inv_width[ARWEN_CELL(f)]) / density[ARWEN_CELL(f)];
        const T c_hi = ((-fdt) * inv_width[ARWEN_CELL(f + 1)]) / density[ARWEN_CELL(f + 1)];
        const bool upwind_lo = ARWEN_PSI(f) >= T(0);
        const T d_face = ARWEN_FACE(f + 1);
        const T w_lo = (d_face - ARWEN_C(f)) * (T(1) - c_lo);
        const T w_hi = (d_face - ARWEN_C(f + 1)) * (T(1) - c_hi);
        const T q_lo = ARWEN_Q(f);
        const T q_hi = ARWEN_Q(f + 1);
        const T slope_lo = (f >= 1) ? arwen_harmonic<T>(ARWEN_DQ(f - 1), ARWEN_DQ(f)) : T(0);
        const T slope_hi = (f + 1 <= n - 2) ? arwen_harmonic<T>(ARWEN_DQ(f), ARWEN_DQ(f + 1)) : T(0);
        T face_value = upwind_lo ? (q_lo + slope_lo * w_lo) : (q_hi + slope_hi * w_hi);
        const T q_up = upwind_lo ? q_lo : q_hi;
        face_value = arwen_min<T>(
            arwen_max<T>(face_value, arwen_min<T>(q_lo, q_hi)),
            arwen_min<T>(arwen_max<T>(q_lo, q_hi), T(2) * q_up));
        const T tracer_flux = fdt * face_value;
        out_value = out_value + tracer_flux * inv_width[cell];
    }
    if (tbase == 0) {
        // The pseudo-density advanced by the same face fluxes, in the
        // specification's order: the east face's share subtracted, then
        // the west face's added.
        T d = density[cell];
        if (k <= n - 2) {
            d = d - (ARWEN_PSI(k) * dt) * inv_width[cell];
        }
        if (k >= 1) {
            d = d + (ARWEN_PSI(k - 1) * dt) * inv_width[cell];
        }
        out_density[cell] = d;
    }
    #undef ARWEN_CELL
    #undef ARWEN_PSI
    #undef ARWEN_FACE
    #undef ARWEN_Q
    #undef ARWEN_C
    #undef ARWEN_DQ
    out = out_value;
"""

#: The deep halo a CARD boundary exchanges, in latitude rows, when no
#: value is configured.  It has to cover ``2n`` for the step's meridional
#: sub-step count ``n``; the design's worked T533 case runs n = 4 and the
#: polar rows of a real case reach into the teens (R12), so 16 rows covers
#: n = 8 and a run that needs more is REFUSED by name rather than swept
#: from stale neighbour values.  One row of the tracer stack is 1.2 MB at
#: T255 L40 float32, so 16 rows is 19.6 MB a boundary a step.
DEFAULT_CARD_HALO_ROWS = 16


def xp_of(transport):
    return transport.transform.backend.xp


#: The fused sweep kernels, one entry per (name, module, DEVICE).  See
#: :func:`~woof.globe.spectral.backend.device_cache_key` for why the
#: device id is in the key; gate CARD-1 covers it.
_KERNELS: dict[tuple, object] = {}


def _fused(xp, name: str):
    """The fused sweep kernels, built once per process and per device.

    ``-fmad=false``: the specification rounds after every multiply and
    every add (each numpy ufunc is one rounding), so contraction into a
    fused multiply-add would move last bits; the kernels are asked for
    the same rounding points.  Division and comparison are IEEE by
    default under NVRTC (no fast-math).
    """
    key = (name, *device_cache_key(xp))
    kernel = _KERNELS.get(key)
    if kernel is None:
        if name == "periodic":
            kernel = xp.ElementwiseKernel(
                "raw T mass, raw T density, raw T psi, T dt, int64 ncell, int64 n2",
                "T out, raw T out_density",
                _PERIODIC_OPERATION,
                "arwen_tracer_periodic_sweep",
                preamble=_FUSED_PREAMBLE,
                options=("-fmad=false",),
            )
        elif name == "walled":
            kernel = xp.ElementwiseKernel(
                "raw T mass, raw T density, raw T psi, raw T centre, raw T face, "
                "raw T inv_width, T dt, int64 ncell, int64 n0, int64 n1, int64 n2, "
                "int32 axis",
                "T out, raw T out_density",
                _WALLED_OPERATION,
                "arwen_tracer_walled_sweep",
                preamble=_FUSED_PREAMBLE,
                options=("-fmad=false",),
            )
        else:
            raise KeyError(name)
        _KERNELS[key] = kernel
    return kernel


@dataclass
class GridTracerTransport:
    """Conservative positive-definite transport on the Gaussian grid.

    ``courant_limit`` is the largest per-face fraction of the upwind cell
    any sub-step may move; 0.25 is the largest value for which two
    outflow faces (each bounded by twice the upwind value times its
    Courant number) cannot exceed the cell's mass, so a larger limit
    would let a divergent flow make negative tracer mass and a positive
    scheme would stop being one.

    ``fused`` selects the fused cupy sweeps (the default; ignored on the
    numpy backend, which always runs the specification).  It exists so
    the equivalence of the two paths can be proved on one device in one
    process; it is not a tuning knob.
    """

    transform: object
    vertical: object
    courant_limit: float = 0.25
    fused: bool = True
    metal_context: object | None = None
    # Step profiler slot (woof.globe.profile); None is the no-op.
    profiler: object | None = None

    def __post_init__(self) -> None:
        limit = float(self.courant_limit)
        if not (math.isfinite(limit) and 0.0 < limit <= 0.25):
            raise ValueError(
                "tracer courant_limit must lie in (0, 0.25]: the limited "
                "face value is bounded by twice the upwind cell value, so "
                "two outflow faces at 0.25 each can take exactly the cell's "
                "mass and a larger limit lets a divergent flow drive the "
                "tracer negative"
            )
        self.courant_limit = limit
        backend = self.transform.backend
        grid = self.transform.grid
        mu = np.asarray(grid.sin_lat, dtype=np.float64)
        weights = np.asarray(grid.quadrature_weights, dtype=np.float64)
        nlat, nlon = grid.shape
        if not np.all(np.diff(mu) > 0.0):
            raise ValueError("Gaussian latitudes must ascend south to north")
        mu_half = np.empty(nlat + 1, dtype=np.float64)
        mu_half[0] = -1.0
        mu_half[1:] = -1.0 + np.cumsum(weights)
        mu_half[-1] = 1.0
        phi_half = np.arcsin(np.clip(mu_half, -1.0, 1.0))
        dphi = np.diff(phi_half)
        # Ring-mean cos(lat) of the cell, exactly d(mu)/d(phi) over the
        # cell, so a uniform zonal flux diverges to zero on every ring.
        cos_eff = weights / dphi
        cos_half = np.sqrt(np.maximum(0.0, 1.0 - mu_half * mu_half))
        cos_half[0] = 0.0
        cos_half[-1] = 0.0
        radius = float(grid.radius_m)
        dlon = 2.0 * math.pi / nlon
        dtype = backend.float_dtype
        self.nlat = int(nlat)
        self.nlon = int(nlon)
        self.nlev = int(self.vertical.nlev)
        # Zonal: psi = (dp u)_face / (a cos_eff dlon), Pa/s per cell.
        self._x_scale = backend.asarray(
            (1.0 / (radius * cos_eff * dlon))[None, :, None], dtype=dtype
        )
        # Meridional: psi = (dp v)_face cos_half / (a dmu_j), Pa/s per
        # cell; the interface factor and the cell divisor are separate
        # because the flux is one number per face and the divisor one per
        # cell on either side.
        self._y_face = backend.asarray(
            (cos_half[1:-1] / radius)[None, :, None], dtype=dtype
        )
        self._y_inv_dmu = backend.asarray(
            (1.0 / weights)[None, :, None], dtype=dtype
        )
        self._mu = backend.asarray(mu[None, :, None], dtype=dtype)
        self._mu_half = backend.asarray(mu_half[None, :, None], dtype=dtype)
        self._x_centre = backend.asarray(
            np.arange(nlon, dtype=np.float64)[None, None, :], dtype=dtype
        )
        self._x_face = backend.asarray(
            (np.arange(nlon + 1, dtype=np.float64) - 0.5)[None, None, :],
            dtype=dtype,
        )
        self._cell_weight = backend.asarray(
            (weights / (2.0 * nlon))[None, :, None], dtype=dtype
        )
        # The meridional sweep's coordinates broadcast to the full grid
        # once, for the fused kernel's flat indexing (memory: three
        # (nlev, nlat, nlon) arrays, 0.14 GiB at T255 float32).  Above one
        # band they are built at the band's width instead and not cached:
        # the cache is the resident run's, and a banded run pays a
        # broadcast per band rather than holding 0.574 GiB at T533.
        self._y_full = None
        # How many latitude bands grid space is streamed through.  The
        # model hands its own pipeline down after construction; the
        # default is the resident single band.
        self.pipeline = BandPipeline(self.nlat, 1)
        # The deep halo a CARD boundary needs, in latitude rows.  The
        # model hands its configured value down with the pipeline; a
        # single-card run never reads it.
        self.card_halo_rows = DEFAULT_CARD_HALO_ROWS

    @property
    def _use_fused(self) -> bool:
        return bool(self.fused) and self.transform.backend.name == "cupy"

    # ---- public entry --------------------------------------------------

    def advance(
        self,
        tracers: dict[str, object],
        dp,
        mass_flux_u,
        mass_flux_v,
        omega_half,
        dt_s: float,
        *,
        step: int = 0,
    ) -> tuple[dict[str, object], dict[str, float]]:
        """Transport every tracer over ``dt_s``.

        ``tracers`` maps name to a nonnegative (nlev, nlat, nlon) mixing
        ratio on the Gaussian grid; ``dp`` is the layer thickness (Pa)
        the tracers sit in at the start of the step; ``mass_flux_u`` and
        ``mass_flux_v`` are the cell-centred ``dp*u`` and ``dp*v`` (Pa
        m/s) and ``omega_half`` the (nlev+1, nlat, nlon) half-level
        pressure velocity (Pa/s, positive downward, zero at the top and
        the surface), all time-averaged over the step.  Returns the
        advanced tracers and a metrics record: the sub-cycle counts, the
        largest Courant number of each direction, the tracer mass the
        roundoff floor removed (kg/m2, global mean over all tracers) and,
        under ``"pseudo_density"``, the layer thickness the fluxes carried
        ``dp`` to (the caller measures it against the continuity's own).
        """
        if self.metal_context is not None:
            return self._advance_metal(
                tracers, dp, mass_flux_u, mass_flux_v, omega_half, dt_s,
                step=step,
            )
        xp = self.transform.backend.xp
        dtype = self.transform.backend.float_dtype
        dt = dtype(float(dt_s))
        names = list(tracers)
        # The sweeps advance the pseudo-density IN PLACE band by band, so
        # it is this call's own array and never the caller's dp.  The
        # resident form paid the same copy inside every sweep
        # (``new_density = density.copy()``), plus a whole tracer stack
        # beside it; this pays one, once.
        density = xp.array(dp, dtype=dtype)
        if tuple(density.shape) != (self.nlev, self.nlat, self.nlon):
            raise ValueError(
                f"dp shape {tuple(density.shape)} is not the model's "
                f"{(self.nlev, self.nlat, self.nlon)}"
            )
        for name in names:
            value = tracers[name]
            if tuple(value.shape) != tuple(density.shape):
                raise ValueError(
                    f"grid tracer {name} shape {tuple(value.shape)} != "
                    f"{tuple(density.shape)}"
                )
        # Interface fluxes in Pa/s per cell (see the class docstring).
        flux_u = xp.asarray(mass_flux_u, dtype=dtype)
        flux_v = xp.asarray(mass_flux_v, dtype=dtype)
        psi_x = 0.5 * (flux_u + xp.roll(flux_u, -1, axis=-1)) * self._x_scale
        exchange = self.pipeline.exchange
        cards = 1 if exchange is None else int(getattr(exchange, "world", 1))
        if cards > 1:
            # The meridional face velocity of the boundary face reads the
            # neighbouring card's mass flux, and the extended band a deep
            # halo sweeps reads it for every halo row.  It does not change
            # during the sweeps, so it is exchanged once, here.  One row
            # is (nlev, nlon) floats: 0.12 MB at T255 L40 float32, so a
            # generous width costs nothing beside the tracer stack's.
            exchange.halo(
                xp, flux_v, 1, int(self.card_halo_rows) + 1,
                name="transport_flux_v",
            )
        psi_y = 0.5 * (flux_v[:, :-1, :] + flux_v[:, 1:, :]) * self._y_face
        psi_z = xp.asarray(omega_half, dtype=dtype)[1:-1]
        del flux_u, flux_v
        # Mass per unit area (Pa) is what the sweeps conserve: one stacked
        # (ntracer, nlev, nlat, nlon) array for every sweep.
        mass = xp.empty((len(names), *density.shape), dtype=dtype)
        prefetch(tracers[names[0]]) if names else None
        for index, name in enumerate(names):
            # ``resident`` is the identity for a tracer already on the
            # card and stages one held by the pinned host tier.  ONE at a
            # time, into the stack the sweeps need anyway: a tier run
            # never has the whole ten-tracer slice on the card (1.9 GiB
            # at T533), and a resident run copies nothing it did not copy
            # before.  The NEXT tracer's copy is started first, so the
            # link works while this one is multiplied into the stack.
            if index + 1 < len(names):
                prefetch(tracers[names[index + 1]])
            value = resident(xp, tracers[name])
            xp.multiply(
                xp.asarray(value, dtype=dtype), density, out=mass[index]
            )
            del value
        metrics: dict[str, float] = {}
        order = ("x", "y", "z") if int(step) % 2 == 0 else ("z", "y", "x")
        prof = profiler_of(self)
        for direction in order:
            with prof.section(f"sweep_{direction}"):
                if direction == "x":
                    mass, density = self._zonal(mass, density, psi_x, dt, metrics)
                elif direction == "y":
                    mass, density = self._meridional(mass, density, psi_y, dt, metrics)
                else:
                    mass, density = self._vertical(mass, density, psi_z, dt, metrics)
        metrics["order"] = "".join(order)
        metrics["pseudo_density"] = density
        # The roundoff floor: the sweeps are positive by construction, so
        # anything below zero here is float cancellation at the 1e-7
        # relative scale of the working dtype.  It is measured, reported
        # and bounded; a sweep that made real negative mass is a scheme
        # defect and refuses below.
        out: dict[str, object] = {}
        clipped = 0.0
        largest_negative = 0.0
        floor = prof.section("floor")
        floor.__enter__()
        # One device reduction for every tracer's negative and positive
        # column mass, then one host read (the per-tracer read of the
        # first version was ten synchronisations per step).
        #
        # THE BAND LOOP, and the allocation this lane exists to remove.
        # ``ratio = mass / density`` is one whole ten-tracer grid volume:
        # 2,053,123,584 B at T533 L40 float32, the exact request that
        # failed at step 2 on a 32 GiB RTX 5090 (MEASURED 2026-09-06, this
        # file, this line).  Its negative and positive parts are two more.
        # A band computes its own rows of all three, hands them to the
        # resident (ntracer, nlat) accumulators, and writes the floored
        # mixing ratio straight into the output the caller keeps, so at B
        # bands the three volumes exist at 1/B of full size and the whole
        # sum still runs once, over latitude, in grid order.
        neg_rows = LatitudeAccumulator(
            xp, (len(names), self.nlat), density.dtype,
            name="transport_floor_negative", exchange=self.pipeline.exchange,
        )
        pos_rows = LatitudeAccumulator(
            xp, (len(names), self.nlat), density.dtype,
            name="transport_floor_positive", exchange=self.pipeline.exchange,
        )
        # The divide runs IN PLACE into the mass stack: the mass is dead
        # the moment its ratio exists, so the allocation that failed is
        # not made smaller, it is not made at all.
        ratio = mass
        for rows in self.pipeline.local_slices():
            band = xp.divide(
                mass[:, :, rows, :], density[:, rows, :],
                out=mass[:, :, rows, :],
            )
            neg_rows.add_band(rows, self._global_mean_columns_rows(
                xp.minimum(band, 0.0), density[:, rows, :], rows=rows,
            ))
            pos_rows.add_band(rows, self._global_mean_columns_rows(
                xp.maximum(band, 0.0), density[:, rows, :], rows=rows,
            ))
            del band
        neg_host = np.asarray(self.transform.backend.to_numpy(
            neg_rows.total() / GRAVITY_M_S2
        ), dtype=np.float64)
        pos_host = np.asarray(self.transform.backend.to_numpy(
            pos_rows.total() / GRAVITY_M_S2
        ), dtype=np.float64)
        del neg_rows, pos_rows
        for index, name in enumerate(names):
            value = ratio[index]
            if neg_host[index] != 0.0:
                clipped -= float(neg_host[index])
                largest_negative = max(
                    largest_negative,
                    -float(neg_host[index]) / max(float(pos_host[index]), 1.0e-300),
                )
                value = xp.maximum(value, 0.0)
            out[name] = value
        del ratio
        metrics["floor_clip_kg_m2"] = float(clipped)
        metrics["floor_clip_relative"] = float(largest_negative)
        # THE ONE PLACE THE GRID TRACERS CROSS A CARD BOUNDARY.  Every
        # other writer of the ten grid tracers leaves the whole globe on
        # every card -- the positivity repair floors a whole array, the
        # physics half-step runs its bands on every card and assembles the
        # globe (dynamics.whole_globe_slices) -- so the sweeps are the
        # only band-wise writer and the state is whole again the moment
        # this returns.  Eleven volumes, 498 MB at T255 L40
        # float32 per card per step; that figure and the physics
        # duplication are what the two-card verdict is made of.
        if cards > 1:
            for name in names:
                out[name] = exchange.fill_rows(
                    xp, out[name], out[name].ndim - 2, name=f"advanced_{name}")
            exchange.fill_rows(xp, density, density.ndim - 2,
                               name="advanced_density")
        floor.__exit__(None, None, None)
        eps = float(xp.finfo(dtype).eps)
        if largest_negative > 64.0 * eps:
            raise FloatingPointError(
                f"grid tracer transport made negative mass {largest_negative:.3g} "
                "of a tracer's positive mass, beyond the 64-ulp roundoff "
                f"floor ({64.0 * eps:.3g}) of a positive-definite scheme"
            )
        return out, metrics

    def _advance_metal(self, tracers, dp, mass_flux_u, mass_flux_v,
                       omega_half, dt_s, *, step):
        """Resident FP64 Metal implementation of :meth:`advance`.

        MetalArray intentionally has no implicit NumPy compatibility.  This
        path therefore keeps the packed mass, pseudo-density, interface
        fluxes, sweep scratch and reductions in Metal buffers throughout the
        timestep.  Only scalar Courant/floor diagnostics are read back.
        """
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.generated import kernel_specs

        shape = (self.nlev, self.nlat, self.nlon)
        names = list(tracers)
        if not names:
            raise ValueError("grid tracer transport requires at least one tracer")
        for name, value in (("dp", dp), ("mass_flux_u", mass_flux_u),
                            ("mass_flux_v", mass_flux_v),
                            ("omega_half", omega_half)):
            if not isinstance(value, MetalArray):
                raise TypeError(f"resident Global {name} must be a MetalArray")
            if value.buffer.context is not self.metal_context:
                raise ValueError(f"resident Global {name} belongs to another Metal context")
            if value.dtype != np.dtype(np.float64):
                raise TypeError(f"resident Global {name} must use FP64 storage")
        if dp.shape != shape or mass_flux_u.shape != shape or mass_flux_v.shape != shape:
            raise ValueError(f"resident Global transport fields must have shape {shape}")
        if omega_half.shape != (self.nlev + 1, self.nlat, self.nlon):
            raise ValueError("resident Global omega_half has the wrong shape")
        if self.pipeline.exchange is not None and int(
                getattr(self.pipeline.exchange, "world", 1)) > 1:
            raise NotImplementedError(
                "resident Metal Global tracer transport requires a single card"
            )
        for name in names:
            value = tracers[name]
            if (not isinstance(value, MetalArray) or value.shape != shape
                    or value.dtype != np.dtype(np.float64)
                    or value.buffer.context is not self.metal_context):
                raise TypeError(
                    f"resident Global tracer {name} must be FP64 Metal with shape {shape}"
                )

        metal = MetalArrayModule(self.metal_context)
        count = self.nlev * self.nlat * self.nlon
        psi_x = metal.empty(shape, dtype=np.float64)
        psi_y = metal.empty((self.nlev, self.nlat - 1, self.nlon), dtype=np.float64)
        psi_z = metal.empty((self.nlev - 1, self.nlat, self.nlon), dtype=np.float64)
        mass = metal.empty((len(names), *shape), dtype=np.float64)
        for index, name in enumerate(names):
            packed = mass.view((mass.size,)).view(
                (count,), byte_offset=index * count * np.dtype(np.float64).itemsize
            )
            self._metal_launch(
                "global_tracer_pack_fragment", kernel_specs,
                {"tracer": tracers[name].view((count,)),
                 "density": dp.view((count,)), "mass": packed},
                {"count": count}, count,
            )
        x_scale = metal.asarray(np.ascontiguousarray(self._x_scale[0, :, 0]), dtype=np.float64)
        y_face = metal.asarray(np.ascontiguousarray(self._y_face[0, :, 0]), dtype=np.float64)
        self._metal_launch(
            "global_tracer_flux_prepare_fragment", kernel_specs,
            {"flux_u": mass_flux_u.view((count,)),
             "flux_v": mass_flux_v.view((count,)),
             "omega": omega_half.view(((self.nlev + 1) * self.nlat * self.nlon,)),
             "x_scale": x_scale, "y_face": y_face,
             "psi_x": psi_x.view((count,)),
             "psi_y": psi_y.view((self.nlev * (self.nlat - 1) * self.nlon,)),
             "psi_z": psi_z.view(((self.nlev - 1) * self.nlat * self.nlon,))},
            {"nlevel": self.nlev, "nlatitude": self.nlat,
             "nlongitude": self.nlon}, count,
        )
        density = metal.asarray(dp)
        metrics: dict[str, float] = {}
        order = ("x", "y", "z") if int(step) % 2 == 0 else ("z", "y", "x")
        prof = profiler_of(self)
        for direction in order:
            with prof.section(f"sweep_{direction}"):
                if direction == "x":
                    mass, density = self._metal_zonal(
                        mass, density, psi_x, dt_s, metrics, metal, kernel_specs,
                    )
                elif direction == "y":
                    mass, density = self._metal_meridional(
                        mass, density, psi_y, dt_s, metrics, metal, kernel_specs,
                    )
                else:
                    mass, density = self._metal_vertical(
                        mass, density, psi_z, dt_s, metrics, metal, kernel_specs,
                    )
        metrics["order"] = "".join(order)
        metrics["pseudo_density"] = density
        with prof.section("floor"):
            clipped, largest_negative = self._metal_floor(
                mass, density, names, metal, kernel_specs,
            )
        metrics["floor_clip_kg_m2"] = clipped
        metrics["floor_clip_relative"] = largest_negative
        eps = np.finfo(np.float64).eps
        if largest_negative > 64.0 * eps:
            raise FloatingPointError(
                f"grid tracer transport made negative mass {largest_negative:.3g} "
                "of a tracer's positive mass, beyond the 64-ulp roundoff "
                f"floor ({64.0 * eps:.3g}) of a positive-definite scheme"
            )
        return {
            name: mass.view(
                shape, byte_offset=index * count * np.dtype(np.float64).itemsize
            )
            for index, name in enumerate(names)
        }, metrics

    def _metal_launch(self, name, kernel_specs, resources, parameters, count):
        threads = getattr(kernel_specs, name.upper() + "_THREADS_PER_GROUP")
        self.metal_context.submit_kernel(
            name, resources, parameters, exact_policy=False,
            grid_size=int(count), threads_per_group=int(threads),
        )

    def _metal_scalar_courant(self, density, psi, dt, *, axis, inv_width,
                              metrics, metal, kernel_specs, label):
        """Reduce a walled sweep's largest face Courant number on Metal."""
        ncell = density.size
        chunk = 256
        nblock = max(1, math.ceil(psi.size / chunk))
        partial = metal.empty((nblock,), dtype=np.float64)
        dt_array = metal.asarray(np.asarray([float(dt)], dtype=np.float64))
        limit = metal.asarray(np.asarray([self.courant_limit], dtype=np.float64))
        inv = (metal.asarray(np.ascontiguousarray(inv_width), dtype=np.float64)
               if inv_width is not None else metal.zeros((1,), dtype=np.float64))
        self._metal_launch(
            "global_tracer_courant_scalar_fragment", kernel_specs,
            {"density": density.view((ncell,)), "psi": psi.view((psi.size,)),
             "inv_width": inv, "dt": dt_array, "partial": partial},
            {"nlevel": self.nlev, "nlatitude": self.nlat,
             "nlongitude": self.nlon, "axis": axis, "chunk": chunk,
             "count": int(psi.size), "nblock": nblock}, nblock,
        )
        summary = metal.empty((2,), dtype=np.float64)
        self._metal_launch(
            "global_tracer_courant_scalar_finalize_fragment", kernel_specs,
            {"partial": partial, "limit": limit, "summary": summary},
            {"npartial": nblock}, 1,
        )
        host = summary.to_numpy()
        maximum = float(host[0])
        n = max(1, int(host[1]))
        if not math.isfinite(maximum):
            raise FloatingPointError("tracer transport Courant number is non-finite")
        metrics["max_courant_" + label] = maximum
        metrics["substeps_" + label] = n
        return n

    def _metal_zonal(self, mass, density, psi, dt, metrics, metal, kernel_specs):
        rowmax = metal.empty((self.nlat,), dtype=np.float64)
        dt_array = metal.asarray(np.asarray([float(dt)], dtype=np.float64))
        limit = metal.asarray(np.asarray([self.courant_limit], dtype=np.float64))
        self._metal_launch(
            "global_tracer_courant_row_fragment", kernel_specs,
            {"density": density.view((density.size,)),
             "psi": psi.view((psi.size,)), "dt": dt_array, "row_max": rowmax},
            {"nlevel": self.nlev, "nlatitude": self.nlat,
             "nlongitude": self.nlon}, self.nlat,
        )
        row_steps = metal.empty((self.nlat,), dtype=np.int32)
        summary = metal.empty((3,), dtype=np.float64)
        self._metal_launch(
            "global_tracer_courant_row_finalize_fragment", kernel_specs,
            {"row_max": rowmax, "limit": limit, "row_steps": row_steps,
             "summary": summary},
            {"nlatitude": self.nlat}, 1,
        )
        host = summary.to_numpy()
        maximum, n, polar_rows = float(host[0]), max(1, int(host[1])), int(host[2])
        if not math.isfinite(maximum):
            raise FloatingPointError("tracer transport Courant number is non-finite")
        metrics["max_courant_x"] = maximum
        metrics["substeps_x"] = n
        metrics["polar_band_rows_x"] = polar_rows
        new_mass = metal.empty(mass.shape, dtype=np.float64)
        new_density = metal.empty(density.shape, dtype=np.float64)
        dt_array = metal.asarray(np.asarray([float(dt)], dtype=np.float64))
        for iteration in range(n):
            self._metal_launch(
                "global_tracer_zonal_rows_fragment", kernel_specs,
                {"mass": mass.view((mass.size,)), "density": density.view((density.size,)),
                 "psi": psi.view((psi.size,)), "row_steps": row_steps,
                 "dt": dt_array, "new_mass": new_mass.view((new_mass.size,)),
                 "new_density": new_density.view((new_density.size,))},
                {"ntracer": mass.shape[0], "nlevel": self.nlev,
                 "nlatitude": self.nlat, "nlongitude": self.nlon,
                 "iteration": iteration}, density.size,
            )
            mass, new_mass = new_mass, mass
            density, new_density = new_density, density
        return mass, density

    def _metal_meridional(self, mass, density, psi, dt, metrics, metal, kernel_specs):
        weights = np.asarray(self.transform.grid.quadrature_weights, dtype=np.float64)
        n = self._metal_scalar_courant(
            density, psi, dt, axis=1, inv_width=1.0 / weights,
            metrics=metrics, metal=metal, kernel_specs=kernel_specs, label="y",
        )
        sub = metal.asarray(np.asarray([float(dt) / n], dtype=np.float64))
        centre = metal.asarray(np.asarray(self.transform.grid.sin_lat, dtype=np.float64))
        mu_half = -1.0 + np.concatenate(([0.0], np.cumsum(weights)))
        mu_half[-1] = 1.0
        face = metal.asarray(mu_half[1:])
        inv_width = metal.asarray(1.0 / weights)
        new_mass = metal.empty(mass.shape, dtype=np.float64)
        new_density = metal.empty(density.shape, dtype=np.float64)
        for _ in range(n):
            self._metal_launch(
                "global_tracer_walled_resident_fragment", kernel_specs,
                {"mass": mass.view((mass.size,)), "density": density.view((density.size,)),
                 "psi": psi.view((psi.size,)), "centre": centre, "face": face,
                 "inv_width": inv_width, "sub": sub,
                 "new_mass": new_mass.view((new_mass.size,)),
                 "new_density": new_density.view((new_density.size,))},
                {"ntracer": mass.shape[0], "nouter": self.nlev,
                 "axis_length": self.nlat, "stride": self.nlon}, density.size,
            )
            mass, new_mass = new_mass, mass
            density, new_density = new_density, density
        return mass, density

    def _metal_vertical(self, mass, density, psi, dt, metrics, metal, kernel_specs):
        n = self._metal_scalar_courant(
            density, psi, dt, axis=0, inv_width=None, metrics=metrics,
            metal=metal, kernel_specs=kernel_specs, label="z",
        )
        ncolumn = self.nlat * self.nlon
        p_half = metal.empty(((self.nlev + 1) * ncolumn,), dtype=np.float64)
        p_full = metal.empty((self.nlev * ncolumn,), dtype=np.float64)
        p_top = metal.asarray(np.asarray([float(self.vertical.a_half_pa[0])], dtype=np.float64))
        self._metal_launch(
            "global_tracer_pressure_coords_fragment", kernel_specs,
            {"density": density.view((density.size,)), "p_half": p_half,
             "p_full": p_full, "p_top": p_top},
            {"nlevel": self.nlev, "ncolumn": ncolumn}, ncolumn,
        )
        sub = metal.asarray(np.asarray([float(dt) / n], dtype=np.float64))
        new_mass = metal.empty(mass.shape, dtype=np.float64)
        new_density = metal.empty(density.shape, dtype=np.float64)
        for _ in range(n):
            self._metal_launch(
                "global_tracer_vertical_fragment", kernel_specs,
                {"mass": mass.view((mass.size,)), "density": density.view((density.size,)),
                 "psi": psi.view((psi.size,)), "p_half": p_half, "p_full": p_full,
                 "sub": sub, "new_mass": new_mass.view((new_mass.size,)),
                 "new_density": new_density.view((new_density.size,))},
                {"ntracer": mass.shape[0], "nlevel": self.nlev,
                 "ncolumn": ncolumn}, density.size,
            )
            mass, new_mass = new_mass, mass
            density, new_density = new_density, density
        return mass, density

    def _metal_floor(self, mass, density, names, metal, kernel_specs):
        ncell = density.size
        chunk = 256
        nblock = math.ceil(ncell / chunk)
        negative = metal.empty((len(names) * nblock,), dtype=np.float64)
        positive = metal.empty((len(names) * nblock,), dtype=np.float64)
        row_weight = metal.asarray(np.broadcast_to(
            np.asarray(self._cell_weight[0], dtype=np.float64),
            (self.nlat, self.nlon),
        ).reshape(-1).copy())
        nblock_array = nblock
        self._metal_launch(
            "global_tracer_floor_fragment", kernel_specs,
            {"mass": mass.view((mass.size,)), "density": density.view((ncell,)),
             "row_weight": row_weight, "negative": negative,
             "positive": positive},
            {"ntracer": len(names), "nlevel": self.nlev,
             "nlatitude": self.nlat, "nlongitude": self.nlon,
             "chunk": chunk, "nblock": nblock_array}, len(names) * nblock,
        )
        neg = metal.empty((len(names),), dtype=np.float64)
        pos = metal.empty((len(names),), dtype=np.float64)
        self._metal_launch(
            "global_tracer_floor_finalize_fragment", kernel_specs,
            {"negative": negative, "positive": positive,
             "neg_total": neg, "pos_total": pos},
            {"ntracer": len(names), "nblock": nblock}, len(names),
        )
        neg_host, pos_host = neg.to_numpy(), pos.to_numpy()
        clipped = 0.0
        largest = 0.0
        for n, p in zip(neg_host, pos_host):
            if n != 0.0:
                clipped -= float(n) / GRAVITY_M_S2
                largest = max(largest, -float(n) / max(float(p), 1.0e-300))
        return clipped, largest

    # ---- helpers -------------------------------------------------------

    def _global_mean_columns_rows(self, value, density, rows=None):
        """The band-local stage: each tracer's area-weighted column mass
        per latitude row, ``(ntracer, nlat)``.  ``rows`` slices the cell
        weights to the band's own rows; every row is independent of every
        other, so a band computes exactly the rows it holds."""
        xp = self.transform.backend.xp
        weight = self._cell_weight[0]
        if rows is not None:
            weight = weight[rows]
        column = xp.sum(value * density, axis=1)
        return xp.sum(column * weight, axis=-1)

    def _global_mean_columns(self, value, density):
        """Global-mean column mass (kg/m2) of each tracer of a stacked
        ``(ntracer, nlev, nlat, nlon)`` ratio field, on the device.
        Same arithmetic per tracer as the first version's per-tracer
        ``_global_mean_column``: ``value`` is ``q`` (ratio) or its clipped
        part and the column is ``sum over levels of value * density``
        weighted by the cell areas.

        Two stages over a resident ``(ntracer, nlat)`` buffer (32 KB at
        T533): the rows a band holds, then one sum over latitude in grid
        order.  The resident path is that buffer filled by a single band."""
        xp = self.transform.backend.xp
        rows = self._global_mean_columns_rows(value, density)
        accumulator = LatitudeAccumulator(
            xp, rows.shape, rows.dtype, name="transport_global_mean_columns", exchange=self.pipeline.exchange,
        )
        accumulator.add_band(slice(0, rows.shape[-1]), rows)
        return accumulator.total() / GRAVITY_M_S2

    @staticmethod
    def _harmonic(left, right, xp):
        product = left * right
        monotone = product > 0.0
        denominator = xp.where(monotone, left + right, 1.0)
        return xp.where(monotone, 2.0 * product / denominator, 0.0)

    # -- walled sweeps ---------------------------------------------------

    def _walled_step(self, mass, density, psi, sub, *, axis, centre, face, inv_width):
        """One sub-step of a walled sweep: the numpy specification on the
        stacked mass, or the fused kernel on cupy.  ``centre``, ``face``
        and ``inv_width`` are full-grid arrays on the fused path and
        broadcastable ones on the specification path (same values)."""
        xp = self.transform.backend.xp
        lo_cells = _idx(axis, slice(None, -1))
        hi_cells = _idx(axis, slice(1, None))
        if self._use_fused:
            # The kernel advances the pseudo-density with the tracers
            # (the specification's three launches and a copy per
            # sub-step, folded; same products in the same order).
            n0, n1, n2 = density.shape
            out = xp.empty_like(mass)
            new_density = xp.empty_like(density)
            _fused(xp, "walled")(
                mass, density, xp.ascontiguousarray(psi),
                centre, face, inv_width, sub,
                np.int64(n0 * n1 * n2), np.int64(n0), np.int64(n1), np.int64(n2),
                np.int32(axis), out, new_density,
            )
            return out, new_density
        flux_dt = psi * sub
        new_density = density.copy()
        new_density[lo_cells] -= flux_dt * inv_width[lo_cells]
        new_density[hi_cells] += flux_dt * inv_width[hi_cells]
        interior = _idx(axis, slice(1, -1))
        t_lo = _tidx(axis, slice(None, -1))
        t_hi = _tidx(axis, slice(1, None))
        t_interior = _tidx(axis, slice(1, -1))
        c_lo = flux_dt * inv_width[lo_cells] / density[lo_cells]
        c_hi = -flux_dt * inv_width[hi_cells] / density[hi_cells]
        upwind_lo = psi >= 0.0
        d_face = face[interior]
        w_lo = (d_face - centre[lo_cells]) * (1.0 - c_lo)
        w_hi = (d_face - centre[hi_cells]) * (1.0 - c_hi)
        q = mass / density
        dq = (q[t_hi] - q[t_lo]) / (centre[hi_cells] - centre[lo_cells])
        slope = xp.zeros_like(q)
        slope[t_interior] = self._harmonic(
            dq[_tidx(axis, slice(None, -1))],
            dq[_tidx(axis, slice(1, None))], xp,
        )
        q_lo = q[t_lo]
        q_hi = q[t_hi]
        face_value = xp.where(
            upwind_lo,
            q_lo + slope[t_lo] * w_lo,
            q_hi + slope[t_hi] * w_hi,
        )
        q_up = xp.where(upwind_lo, q_lo, q_hi)
        face_value = xp.minimum(
            xp.maximum(face_value, xp.minimum(q_lo, q_hi)),
            xp.minimum(xp.maximum(q_lo, q_hi), 2.0 * q_up),
        )
        tracer_flux = flux_dt * face_value
        updated = mass.copy()
        updated[t_lo] -= tracer_flux * inv_width[lo_cells]
        updated[t_hi] += tracer_flux * inv_width[hi_cells]
        return updated, new_density

    def _substeps_and_metrics(
        self, density, psi, dt, *, axis, inv_width, label, metrics
    ):
        """The whole globe's largest face Courant number, its sub-step
        count, and both metrics -- folded band by band.

        THE COUNT STAYS GLOBAL.  A band that scanned only its own rows
        would sub-cycle them differently from its neighbours, and two
        bands running different sub-step counts on one field diverge
        silently rather than failing.  A maximum is exactly associative,
        so the fold returns the whole array's maximum whatever the
        schedule, and the Courant volume the resident form materialised
        (0.205 GiB at T533 L40 float32) exists at 1/B of that size.
        """
        xp = self.transform.backend.xp
        lo_cells = _idx(axis, slice(None, -1))
        hi_cells = _idx(axis, slice(1, None))
        largest = None
        for rows in self.pipeline.local_slices():
            if axis == 1:
                # The sweep axis IS the latitude axis: a band of cells
                # [a, b) owns the faces [a, b - 1), and the last band's
                # last cell has a wall rather than a face.
                stop = min(rows.stop, self.nlat - 1)
                if stop <= rows.start:
                    continue
                cells = slice(rows.start, stop + 1)
                faces = slice(rows.start, stop)
            else:
                cells = faces = rows
            d = density[:, cells, :]
            flux_dt = psi[:, faces, :] * dt
            if inv_width is None:
                # The vertical sweep's cell divisor is exactly one, and a
                # multiply by 1.0 is exact, so the resident form's
                # ``ones_like(density)`` volume is not built to be read.
                courant = xp.maximum(
                    flux_dt / d[lo_cells], -flux_dt / d[hi_cells]
                )
            else:
                w = inv_width[:, cells, :]
                courant = xp.maximum(
                    flux_dt * w[lo_cells] / d[lo_cells],
                    -flux_dt * w[hi_cells] / d[hi_cells],
                )
            band = _associative_over(
                xp, "max", courant, name="transport_courant_" + label
            )
            largest = band if largest is None else xp.maximum(largest, band)
            del courant, flux_dt, d
        # THE SUB-STEP COUNT IS A LOOP BOUND, and it must be the globe's.
        # A card that scanned only its own rows would take a different
        # number of sub-steps from its partner on the same physical sweep
        # and the two halves of the answer would diverge silently rather
        # than fail.  Maximum is exactly associative, so folding it across
        # the cards costs the answer nothing and costs the wire 4 bytes.
        exchange = self.pipeline.exchange
        if exchange is not None and int(getattr(exchange, "world", 1)) > 1:
            largest = exchange.fold(
                xp, largest, "max", name="transport_courant_cards_" + label)
        maximum = float(self.transform.backend.to_numpy(largest))
        if not math.isfinite(maximum):
            raise FloatingPointError("tracer transport Courant number is non-finite")
        n = max(1, int(math.ceil(maximum / self.courant_limit)))
        metrics["max_courant_" + label] = maximum
        metrics["substeps_" + label] = int(n)
        return n

    def _bounded_block(
        self, mass, density, psi, sub, n, *, axis, centre, face, inv_width
    ):
        """``n`` sub-steps of a walled sweep over one contiguous block."""
        for _ in range(n):
            mass, density = self._walled_step(
                mass, density, psi, sub, axis=axis, centre=centre, face=face,
                inv_width=inv_width,
            )
        return mass, density

    def _meridional_coordinates(self, rows: slice):
        """``(centre, face, inv_width)`` of the meridional sweep over the
        latitude rows ``rows``, in the layout its path wants: broadcast
        views for the specification, contiguous arrays for the fused
        kernel.

        The resident sweep caches its three full-grid arrays (0.574 GiB at
        T533 float32, MEASURED); a banded sweep builds them at the band's
        width per band and does not cache them, because holding one set
        per band is the memory the band count exists to divide.
        """
        xp = self.transform.backend.xp
        centre = self._mu[:, rows, :]
        face = self._mu_half[:, rows.start : rows.stop + 1, :]
        inv_width = self._y_inv_dmu[:, rows, :]
        if not self._use_fused:
            return centre, face, inv_width
        nrows = rows.stop - rows.start
        if nrows == self.nlat:
            if self._y_full is None:
                shape = (self.nlev, self.nlat, self.nlon)
                self._y_full = (
                    xp.ascontiguousarray(xp.broadcast_to(centre, shape)),
                    xp.ascontiguousarray(xp.broadcast_to(
                        face, (self.nlev, self.nlat + 1, self.nlon)
                    )),
                    xp.ascontiguousarray(xp.broadcast_to(inv_width, shape)),
                )
            return self._y_full
        shape = (self.nlev, nrows, self.nlon)
        return (
            xp.ascontiguousarray(xp.broadcast_to(centre, shape)),
            xp.ascontiguousarray(xp.broadcast_to(
                face, (self.nlev, nrows + 1, self.nlon)
            )),
            xp.ascontiguousarray(xp.broadcast_to(inv_width, shape)),
        )

    def _meridional(self, mass, density, psi, dt, metrics):
        """The meridional sweep, band by band behind a deep halo.

        THE DEEP HALO (gate HALO-1).  ``_walled_step`` reads
        ``q[j-2 .. j+2]`` through the van Leer slope, so one sub-step of a
        band is exact on every row except the two at each end, and ``n``
        sub-steps are exact except for ``2n`` rows at each end.  So the
        band is widened by ``2n`` rows once, all ``n`` sub-steps run with
        no further exchange, and the widened rows are dropped.  The
        redundant arithmetic on those rows reads identical inputs and
        produces identical values, so this is bit-exact rather than an
        approximation -- and it replaces ``n`` neighbour exchanges with
        one, which is what a card boundary will need.

        The band writes its result back into the mass stack in place, so
        the rows a later band still needs as halo are original values by
        then only if somebody kept them: ``carry`` is that keeper, ``2n``
        latitude rows of the tracer stack (20 MB at T533 L40 at four
        sub-steps, against the 1.91 GiB a second whole stack would cost).
        """
        pipeline = self.pipeline
        exchange = pipeline.exchange
        cards = 1 if exchange is None else int(getattr(exchange, "world", 1))
        if cards > 1:
            # THE DEEP HALO AT A CARD BOUNDARY.  Exchanged HERE and not at
            # the top of the step: the zonal sweep has already written this
            # card's rows in place, so the neighbour's rows this card holds
            # are one sweep out of date until they are refetched.  One
            # exchange serves the sub-step count's scan (which reads one
            # cell past the block) and all n sub-steps behind it.
            width = int(self.card_halo_rows)
            exchange.halo(xp_of(self), mass, 2, width, name="transport_mass")
            exchange.halo(xp_of(self), density, 1, width, name="transport_density")
        n = self._substeps_and_metrics(
            density, psi, dt, axis=1, inv_width=self._y_inv_dmu,
            label="y", metrics=metrics,
        )
        if cards > 1 and 2 * n > int(self.card_halo_rows):
            raise ValueError(
                f"this step's meridional sweep takes {n} sub-steps, which "
                f"reads {2 * n} rows past each card boundary, and the card "
                f"halo is {int(self.card_halo_rows)} rows.  A short halo "
                "would sweep the boundary rows from stale neighbour values "
                "and the two cards would stop agreeing with one card.  Raise "
                "[memory].card_halo_rows to at least "
                f"{2 * n} (the receipt's sub-step histogram is what sets it)"
            )
        sub = dt / n
        if pipeline.resident:
            centre, face, inv_width = self._meridional_coordinates(
                slice(0, self.nlat)
            )
            return self._bounded_block(
                mass, density, psi, sub, n, axis=1, centre=centre, face=face,
                inv_width=inv_width,
            )
        xp = self.transform.backend.xp
        width = 2 * n
        carry_mass = carry_density = None
        for rows in pipeline.local_slices():
            ext, lead, _trail = pipeline.halo(rows, width)
            block_mass = xp.array(mass[:, :, ext, :])
            block_density = xp.array(density[:, ext, :])
            if lead and carry_mass is not None:
                # The rows below this band were swept in place by the band
                # before it, so the halo reads the copy kept for exactly
                # this.  On the FIRST band a card runs there is no such
                # band: below it are either the pole (a wall) or the
                # neighbouring card's rows, which the boundary exchange
                # above has just filled with their un-swept values.
                block_mass[:, :, :lead, :] = carry_mass[:, :, -lead:, :]
                block_density[:, :lead, :] = carry_density[:, -lead:, :]
            # The originals this band is about to overwrite, kept for the
            # next band's halo.
            keep = slice(
                max(0, rows.stop - width) - ext.start, rows.stop - ext.start
            )
            carry_mass = xp.array(block_mass[:, :, keep, :])
            carry_density = xp.array(block_density[:, keep, :])
            centre, face, inv_width = self._meridional_coordinates(ext)
            block_mass, block_density = self._bounded_block(
                block_mass, block_density,
                xp.ascontiguousarray(psi[:, ext.start : ext.stop - 1, :]),
                sub, n, axis=1, centre=centre, face=face, inv_width=inv_width,
            )
            inner = slice(lead, lead + rows.stop - rows.start)
            mass[:, :, rows, :] = block_mass[:, :, inner, :]
            density[:, rows, :] = block_density[:, inner, :]
            del block_mass, block_density, centre, face, inv_width
        return mass, density

    def _vertical(self, mass, density, psi, dt, metrics):
        """The vertical sweep: the mass coordinate is pressure, so the
        cell divisor is one and the reconstruction runs in pressure
        between the full levels.

        The reconstruction coordinate is the pseudo-density column as
        the sweep finds it (its full and half pressures), held over the
        sweep's sub-steps: the first version built it once before its
        sub-cycle and this one keeps that arithmetic.

        Every column is independent of every other, so a latitude band
        needs no halo and writes its result back in place; the two
        pressure arrays and the unit divisor exist at the band's width
        (0.615 GiB of the three at T533 L40 float32 at one band).
        """
        xp = self.transform.backend.xp
        n = self._substeps_and_metrics(
            density, psi, dt, axis=0, inv_width=None, label="z", metrics=metrics,
        )
        sub = dt / n
        pipeline = self.pipeline
        if pipeline.resident:
            p_half, p_full = self._pressure_coordinates(density)
            return self._bounded_block(
                mass, density, psi, sub, n, axis=0, centre=p_full, face=p_half,
                inv_width=xp.ones_like(density),
            )
        for rows in pipeline.local_slices():
            block_mass = xp.ascontiguousarray(mass[:, :, rows, :])
            block_density = xp.ascontiguousarray(density[:, rows, :])
            p_half, p_full = self._pressure_coordinates(block_density)
            block_mass, block_density = self._bounded_block(
                block_mass, block_density,
                xp.ascontiguousarray(psi[:, rows, :]), sub, n, axis=0,
                centre=p_full, face=p_half,
                inv_width=xp.ones_like(block_density),
            )
            mass[:, :, rows, :] = block_mass
            density[:, rows, :] = block_density
            del block_mass, block_density, p_half, p_full
        return mass, density

    def _pressure_coordinates(self, density):
        """Full and half pressures of the pseudo-density column."""
        xp = self.transform.backend.xp
        dtype = self.transform.backend.float_dtype
        p_top = dtype(float(self.vertical.a_half_pa[0]))
        p_half = xp.concatenate([
            xp.zeros_like(density[:1]) + p_top,
            xp.cumsum(density, axis=0) + p_top,
        ], axis=0)
        p_full = xp.sqrt(p_half[:-1] * p_half[1:])
        if self._use_fused:
            p_half = xp.ascontiguousarray(p_half)
            p_full = xp.ascontiguousarray(p_full)
        return p_half, p_full

    # -- the periodic zonal sweep ---------------------------------------

    def _zonal(self, mass, density, psi, dt, metrics):
        """The periodic zonal sweep, with the polar bands sub-cycled.

        Every row is independent of every other here (the sweep is
        periodic in longitude), so the pieces the polar sub-cycling cuts
        and the latitude bands the pipeline cuts are the same kind of
        cut: the sweep runs over their intersection and writes back in
        place.  The two whole-stack copies the resident form made (2.05
        GiB of mass and 0.205 of density at T533 L40 float32) become one
        block per intersection.
        """
        xp = self.transform.backend.xp
        host = self.transform.backend.to_numpy
        # Per-ring largest Courant number, on the host: the sub-cycle
        # count is a loop bound.  The per-row maxima land in a resident
        # nlat vector (a band writes its own rows) and the polar-band scan
        # below reads the whole vector, so the band construction is the
        # globe's whatever the schedule was.
        row_courant = LatitudeAccumulator(
            xp, (self.nlat,), density.dtype, name="transport_row_courant", exchange=self.pipeline.exchange,
        )
        for rows in self.pipeline.local_slices():
            flux_dt = psi[:, rows, :] * dt
            d = density[:, rows, :]
            courant = xp.maximum(
                flux_dt / d, -flux_dt / xp.roll(d, -1, axis=-1)
            )
            row_courant.add_band(rows, xp.max(courant, axis=(0, 2)))
            del flux_dt, d, courant
        per_row = host(row_courant.complete())
        if not np.all(np.isfinite(per_row)):
            raise FloatingPointError("tracer transport Courant number is non-finite")
        rows_n = np.maximum(
            1, np.ceil(per_row / self.courant_limit).astype(int)
        )
        metrics["max_courant_x"] = float(np.max(per_row))
        nlat = self.nlat
        half = nlat // 2
        south = np.flatnonzero(rows_n[:half] > 1)
        north = np.flatnonzero(rows_n[half:] > 1) + half
        south_end = int(south.max()) + 1 if south.size else 0
        north_start = int(north.min()) if north.size else nlat
        bands = []
        if south_end > 0:
            bands.append((0, south_end, int(rows_n[:south_end].max())))
        if north_start < nlat:
            bands.append((north_start, nlat, int(rows_n[north_start:].max())))
        metrics["substeps_x"] = max([1] + [n for _, _, n in bands])
        metrics["polar_band_rows_x"] = int(sum(b - a for a, b, _ in bands))
        # The middle band (one sweep) and each polar band (n sweeps of
        # dt/n) are independent rows of the same periodic sweep.
        pieces = [(south_end, north_start, 1)] + bands
        for piece_start, piece_stop, n in pieces:
            if piece_stop <= piece_start:
                continue
            piece = slice(piece_start, piece_stop)
            sub = dt / n
            for rows in _intersect(piece, self.pipeline.local_slices()):
                d = xp.ascontiguousarray(density[:, rows, :])
                m = xp.ascontiguousarray(mass[:, :, rows, :])
                p = xp.ascontiguousarray(psi[:, rows, :])
                for _ in range(n):
                    m, d = self._periodic_step(m, d, p, sub)
                density[:, rows, :] = d
                mass[:, :, rows, :] = m
                del d, m, p
        return mass, density

    def _periodic_step(self, mass, density, psi, dt):
        xp = self.transform.backend.xp
        if self._use_fused:
            out = xp.empty_like(mass)
            new_density = xp.empty_like(density)
            _fused(xp, "periodic")(
                mass, density, psi, dt,
                np.int64(density.size), np.int64(density.shape[-1]),
                out, new_density,
            )
            return out, new_density
        flux_dt = psi * dt
        new_density = density - (flux_dt - xp.roll(flux_dt, 1, axis=-1))
        density_next = xp.roll(density, -1, axis=-1)
        c_lo = flux_dt / density
        c_hi = -flux_dt / density_next
        upwind_lo = psi >= 0.0
        w_lo = 0.5 * (1.0 - c_lo)
        w_hi = -0.5 * (1.0 - c_hi)
        q = mass / density
        q_next = xp.roll(q, -1, axis=-1)
        dq = q_next - q
        slope = self._harmonic(xp.roll(dq, 1, axis=-1), dq, xp)
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
        new_mass = mass - (
            tracer_flux - xp.roll(tracer_flux, 1, axis=-1)
        )
        return new_mass, new_density


def sample_grid_field(grid, value, latitude_deg, longitude_deg):
    """Bilinear sample of a Gaussian-grid field at arbitrary points.

    Linear in ``mu`` (sin latitude) between the rings, periodic and
    linear in longitude; beyond the outermost rings the nearest ring's
    value is held (the polar cap of a Gaussian grid has no node).
    Bilinear weights are convex, so a nonnegative field samples
    nonnegative and a bounded field stays inside its bounds: the
    condensate a regional child receives never rings the way a spectral
    synthesis of the same field did.  Leading dimensions of ``value``
    are retained; the points' shape is retained after them.
    """
    value = np.asarray(value, dtype=np.float64)
    nlat, nlon = grid.shape
    if value.shape[-2:] != (nlat, nlon):
        raise ValueError(
            f"field shape {value.shape[-2:]} is not the grid's {(nlat, nlon)}"
        )
    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    point_shape = np.broadcast(lat, lon).shape
    lat = np.broadcast_to(lat, point_shape).reshape(-1)
    lon = np.broadcast_to(lon, point_shape).reshape(-1)
    mu_nodes = np.asarray(grid.sin_lat, dtype=np.float64)
    mu = np.sin(np.deg2rad(lat))
    j_hi = np.clip(np.searchsorted(mu_nodes, mu), 1, nlat - 1)
    j_lo = j_hi - 1
    span = mu_nodes[j_hi] - mu_nodes[j_lo]
    t = np.clip((mu - mu_nodes[j_lo]) / span, 0.0, 1.0)
    dlon = 360.0 / nlon
    x = np.mod(lon, 360.0) / dlon
    i_lo = np.floor(x).astype(int) % nlon
    i_hi = (i_lo + 1) % nlon
    s = x - np.floor(x)
    flat = value.reshape(*value.shape[:-2], nlat * nlon)
    def take(j, i):
        return flat[..., j * nlon + i]
    sampled = (
        (1.0 - t) * ((1.0 - s) * take(j_lo, i_lo) + s * take(j_lo, i_hi))
        + t * ((1.0 - s) * take(j_hi, i_lo) + s * take(j_hi, i_hi))
    )
    return sampled.reshape(*value.shape[:-2], *point_shape)


__all__ = ["GridTracerTransport", "sample_grid_field"]
