"""FROZEN dense-table reference for the spherical-harmonic transform.

This is ``woof/global_spectral/{legendre,transform,vector}.py`` as they
stood at commit 8f7c5d7d7 (2026-09-01), before the Legendre tables were
packed into bands and the derivative table became lazy.  It builds the
full ``(T+1, T+1, nlat)`` squares and contracts the whole triangle in one
batched GEMM.  ``tests/test_global_spectral_bit_identity.py`` runs it
beside the live transform on the same machine and demands EXACT equality
of every output: the live transform's memory layout may change, its
arithmetic may not.

Do not edit the code below.  Only the three relative imports were made
absolute so the file is one self-contained module.
"""
# ---- legendre.py @ 8f7c5d7d7 ------------------------------------------

import math

import numpy as np


def _normalized_table(truncation: int, x: np.ndarray) -> np.ndarray:
    """Orthonormal ``N_nm P_n^m`` values by the bounded normalized recurrence.

    The unnormalized recurrence carries a ``(2m-1)!!`` diagonal that
    overflows float64 near m=150 while ``config.MAXIMUM_TRUNCATION`` admits
    truncation up to 255, so the recurrence must run on the normalized
    values, which stay below ``sqrt((2n+1)/(4pi))`` for every admitted
    degree.  Condon-Shortley phase comes from the negated diagonal step.
    """
    t = int(truncation)
    table = np.zeros((t + 1, t + 1, x.size), dtype=np.float64)
    root = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    table[0, 0] = 1.0 / math.sqrt(4.0 * math.pi)
    for m in range(1, t + 1):
        table[m, m] = -math.sqrt((2 * m + 1) / (2.0 * m)) * root * table[m - 1, m - 1]
    for m in range(0, t):
        table[m + 1, m] = math.sqrt(2 * m + 3) * x * table[m, m]
    for m in range(0, t + 1):
        for n in range(m + 2, t + 1):
            a = math.sqrt((4.0 * n * n - 1.0) / (n * n - m * m))
            b = math.sqrt(
                ((n - 1.0) * (n - 1.0) - m * m)
                / (4.0 * (n - 1.0) * (n - 1.0) - 1.0)
            )
            table[n, m] = a * (x * table[n - 1, m] - b * table[n - 2, m])
    return table


def normalized_associated_legendre_values(
    truncation: int, sin_lat: np.ndarray
) -> np.ndarray:
    """Orthonormal ``N_nm P_n^m(sin(phi))`` values, including poles."""
    x = np.asarray(sin_lat, dtype=np.float64)
    if x.ndim != 1 or np.any(np.abs(x) > 1.0):
        raise ValueError("sin_lat must be a one-dimensional array in [-1, 1]")
    return _normalized_table(truncation, x)


def normalized_associated_legendre(
    truncation: int, sin_lat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(P, dP_dphi)`` for orthonormal complex spherical harmonics.

    ``P[n,m,j]`` is ``N_nm P_n^m(sin(phi_j))`` with the Condon--Shortley
    phase included by the recurrence.  The full harmonic is
    ``Y_nm = P[n,m] * exp(i*m*lambda)`` and integrates to one over the sphere.
    Entries with m > n are zero.  Derivatives require non-polar latitudes;
    scalar-only sampling at a pole uses
    :func:`normalized_associated_legendre_values` instead.
    """
    t = int(truncation)
    x = np.asarray(sin_lat, dtype=np.float64)
    if x.ndim != 1 or np.any(np.abs(x) >= 1.0):
        raise ValueError(
            "sin_lat derivatives require a one-dimensional array strictly "
            "inside (-1, 1)"
        )
    basis = _normalized_table(t, x)
    deriv = np.zeros_like(basis)
    denom = x * x - 1.0
    coslat = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    for n in range(1, t + 1):
        for m in range(0, n + 1):
            # (x^2-1) dP/dx = n x P_nm - c_nm P_{n-1,m} on normalized values,
            # with c_nm carrying the N_nm/N_{n-1,m} normalization ratio.
            if m <= n - 1:
                c = math.sqrt((n * n - m * m) * (2.0 * n + 1.0) / (2.0 * n - 1.0))
                numerator = n * x * basis[n, m] - c * basis[n - 1, m]
            else:
                numerator = n * x * basis[n, m]
            deriv[n, m] = coslat * numerator / denom
    return basis, deriv

# ---- transform.py @ 8f7c5d7d7 -----------------------------------------

import contextlib
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from woof.globe.spectral.backend import Backend, get_backend
from woof.globe.spectral.grid import GaussianGrid


@dataclass
class SphericalHarmonicTransform:
    grid: GaussianGrid
    backend: Backend
    # Opt-in TF32 tensor-core compute for the Legendre GEMM contractions.
    # TF32 truncates the float32 mantissa to 10 bits inside the GEMM, so this
    # CHANGES NUMERICS: it stays default-off, joins the identity hash only
    # when enabled, and needs its own transform_check gate re-measurement on
    # the GPU before any production default flips.
    tensor_core_contractions: bool = False

    def __post_init__(self) -> None:
        if self.tensor_core_contractions:
            if self.backend.name != "cupy":
                raise ValueError(
                    "tensor_core_contractions=True requires backend='cupy': "
                    "numpy has no tensor-core path, so the flag would stamp "
                    "a TF32 identity onto arithmetic that never changes"
                )
            if np.dtype(self.backend.float_dtype) != np.dtype(np.float32):
                raise ValueError(
                    "tensor_core_contractions=True requires precision="
                    "'float32': CuPy's TF32 compute type acts on float32 "
                    "GEMMs only, so under float64 the flag would move the "
                    "identity hash while every contraction stays fp64"
                )
        basis, derivative = normalized_associated_legendre(
            self.grid.truncation, self.grid.sin_lat
        )
        xp = self.backend.xp
        self._weights = self.backend.asarray(
            self.grid.quadrature_weights, dtype=self.backend.float_dtype
        )
        # Gauss-quadrature exactness holds in real arithmetic only: the
        # computed basis values and weights carry O(truncation * eps)
        # correlated error, so the raw analysis of a resolved field leaks
        # ~1e-13 into modes that must vanish (measured at T9, numpy 2.2.6).
        # Solving against the analysis-synthesis Gram matrix per order makes
        # forward(inverse(x)) = x on the resolved subspace at working
        # precision on any in-spec numpy, which is the transform's own
        # correctness claim.
        analysis = basis * self.grid.quadrature_weights[None, None, :] * (2.0 * math.pi)
        for m in range(self.grid.truncation + 1):
            rows = analysis[m:, m, :]
            gram = rows @ basis[m:, m, :].T
            analysis[m:, m, :] = np.linalg.solve(gram, rows)
        # Batch-first (order-leading) table layouts: the whole-triangle
        # contraction in forward/_synthesize is then one strided-batched real
        # GEMM over every order at once instead of a per-order loop, and the
        # above-triangle zeros in these dense tables make the full
        # contraction mathematically identical to the loop it replaced.
        # The old (n, m, j) copies are dropped, so resident table bytes are
        # unchanged by the layout swap.
        self._analysis_mjn = self.backend.asarray(
            np.ascontiguousarray(analysis.transpose(1, 2, 0)),
            dtype=self.backend.float_dtype,
        )
        self._basis_mnj = self.backend.asarray(
            np.ascontiguousarray(basis.transpose(1, 0, 2)),
            dtype=self.backend.float_dtype,
        )
        self._dbasis_mnj = self.backend.asarray(
            np.ascontiguousarray(derivative.transpose(1, 0, 2)),
            dtype=self.backend.float_dtype,
        )
        # Analysis-direction (m, j, n) views for the vector transform share
        # the synthesis buffers; their F-contiguous inner matrices map onto
        # the BLAS/cuBLAS transpose flags without a copy.
        self._basis_mjn = self._basis_mnj.transpose(0, 2, 1)
        self._dbasis_mjn = self._dbasis_mnj.transpose(0, 2, 1)
        self._zonal_wavenumber = self.backend.asarray(
            1j * np.arange(self.grid.truncation + 1, dtype=np.float64),
            dtype=self.backend.complex_dtype,
        )
        n = np.arange(self.grid.truncation + 1, dtype=np.float64)
        eigen = -(n * (n + 1.0)) / (self.grid.radius_m * self.grid.radius_m)
        self._laplacian_eigen = self.backend.asarray(eigen, dtype=self.backend.float_dtype)
        tri = np.tri(self.grid.truncation + 1, dtype=bool)
        self._tri_mask = self.backend.asarray(tri, dtype=bool)
        # Per-coefficient action table for the fused cupy project(): 0 above
        # the triangle (zeroed), 2 on the in-triangle m=0 column (imaginary
        # part dropped), 1 elsewhere in the triangle (copied).  The numpy
        # project() below stays the specification and never reads it.
        code = np.where(tri, 1, 0).astype(np.int8)
        code[:, 0] = np.where(tri[:, 0], 2, 0)
        self._project_code = self.backend.asarray(code)
        self._coslat = self.backend.asarray(
            self.grid.cos_lat[:, None], dtype=self.backend.float_dtype
        )
        self._degree = self.backend.asarray(n, dtype=self.backend.float_dtype)
        self._order = self.backend.asarray(n, dtype=self.backend.float_dtype)

    @classmethod
    def create(
        cls,
        truncation: int,
        *,
        nlat: int | None = None,
        nlon: int | None = None,
        dealias_factor: float = 1.5,
        radius_m: float | None = None,
        backend: str = "numpy",
        precision: str = "float64",
        tensor_core_contractions: bool = False,
    ) -> "SphericalHarmonicTransform":
        kwargs = {}
        if radius_m is not None:
            kwargs["radius_m"] = radius_m
        grid = GaussianGrid.create(
            truncation,
            nlat=nlat,
            nlon=nlon,
            dealias_factor=dealias_factor,
            **kwargs,
        )
        return cls(
            grid,
            get_backend(backend, precision),
            tensor_core_contractions=tensor_core_contractions,
        )

    @property
    def truncation(self) -> int:
        return self.grid.truncation

    @property
    def spectral_shape(self) -> tuple[int, int]:
        n = self.truncation + 1
        return n, n

    @property
    def geometry_identity(self) -> dict:
        return {
            "truncation": int(self.truncation),
            "nlat": int(self.grid.nlat),
            "nlon": int(self.grid.nlon),
            "radius_m": float(self.grid.radius_m),
            "normalization": "complex-orthonormal-condon-shortley",
            "latitude_grid": "gauss-legendre-in-sin-latitude",
        }

    @property
    def geometry_hash(self) -> str:
        raw = json.dumps(
            self.geometry_identity, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(raw).hexdigest()

    @property
    def identity(self) -> dict:
        payload = {
            **self.geometry_identity,
            "backend": self.backend.name,
            "float_dtype": str(np.dtype(self.backend.float_dtype)),
        }
        # TF32 joins the identity only when enabled: it changes the GEMM
        # arithmetic, so an enabled transform must not share a hash with the
        # fp32 transform it diverges from, while the disabled default keeps
        # every pre-existing identity hash byte-identical.
        if self.tensor_core_contractions:
            payload["tensor_core_contractions"] = True
        return payload

    @property
    def identity_hash(self) -> str:
        raw = json.dumps(
            self.identity, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(raw).hexdigest()

    def zeros(self, *leading: int):
        return self.backend.xp.zeros(
            (*leading, *self.spectral_shape), dtype=self.backend.complex_dtype
        )

    def _validate_grid(self, field) -> None:
        if tuple(field.shape[-2:]) != self.grid.shape:
            raise ValueError(
                f"grid field has trailing shape {field.shape[-2:]}, expected {self.grid.shape}"
            )

    def _validate_spectral(self, coeff) -> None:
        if tuple(coeff.shape[-2:]) != self.spectral_shape:
            raise ValueError(
                f"spectral field has trailing shape {coeff.shape[-2:]}, "
                f"expected {self.spectral_shape}"
            )

    def project(self, coeff):
        xp = self.backend.xp
        if self.backend.name == "cupy":
            # One fused launch replaces the asarray+copy+mask-multiply+
            # realify chain (four launches per call, ~470 calls/step at
            # T533).  Same arithmetic as the numpy specification below for
            # every finite input; the kernel always writes a fresh output,
            # matching .copy().  One divergence: above-triangle entries
            # become exact zeros, where the mask multiply would turn a
            # non-finite entry there into NaN (in-triangle non-finites
            # still propagate identically on both paths).
            from woof.globe.spectral.fused import project_kernel

            arr = xp.asarray(coeff, dtype=self.backend.complex_dtype)
            self._validate_spectral(arr)
            return project_kernel(xp)(arr, self._project_code)
        arr = xp.asarray(coeff, dtype=self.backend.complex_dtype).copy()
        self._validate_spectral(arr)
        arr *= self._tri_mask
        arr[..., :, 0] = arr[..., :, 0].real
        return arr

    def _contraction_scope(self):
        """Scope the Legendre GEMM contractions onto TF32 tensor cores.

        Default (flag off) is a no-op, so every hash-checked fp32/fp64
        identity is untouched.  With the flag on (cupy float32 only, enforced
        in ``__post_init__``) the float32 compute type is switched to TF32
        for the duration of the contraction and restored afterwards: CuPy's
        compute-type table is process-global state, and leaving it set would
        silently change the arithmetic of every other float32 GEMM in the
        process.
        """
        if not self.tensor_core_contractions:
            return contextlib.nullcontext()
        # backend='cupy' already imported cupy; CPU paths never reach this.
        from cupy import _core as cupy_core
        from cupy._core import _routines_linalg as cupy_linalg

        xp = self.backend.xp

        @contextlib.contextmanager
        def scope():
            previous = cupy_core.get_compute_type(xp.float32)
            cupy_core.set_compute_type(
                xp.float32, cupy_linalg.COMPUTE_TYPE_TF32
            )
            try:
                yield
            finally:
                cupy_core.set_compute_type(xp.float32, previous)

        return scope()

    def _batched_contract(self, values, table):
        """Contract ``values[..., k, m] * table[m, k, n] -> out[..., n, m]``.

        The whole triangle rides one strided-batched real GEMM per
        real/imaginary component: ``matmul`` on an (orders, batch, k) stack
        against an (orders, k, n) table lands on cuBLAS gemmStridedBatched
        under the cupy backend (where the TF32 compute-type scope can
        apply) and on per-order BLAS GEMM under numpy.  An einsum with the
        order axis shared between operands has no guaranteed GEMM lowering,
        which is why this formulation was chosen.  Zeros above the triangle
        in every table make the full contraction mathematically identical
        to the per-order loop it replaced; the launch count no longer
        scales with truncation.
        """
        xp = self.backend.xp
        lead = values.shape[:-2]
        k = values.shape[-2]
        m_count = values.shape[-1]
        n_count = table.shape[-1]
        stacked = xp.moveaxis(values, -1, 0).reshape(m_count, -1, k)
        real = xp.ascontiguousarray(stacked.real) @ table
        imag = xp.ascontiguousarray(stacked.imag) @ table
        out = xp.empty(real.shape, dtype=self.backend.complex_dtype)
        out.real = real
        out.imag = imag
        return xp.moveaxis(out.reshape(m_count, *lead, n_count), 0, -1)

    def forward(self, field):
        xp = self.backend.xp
        f = xp.asarray(field, dtype=self.backend.float_dtype)
        self._validate_grid(f)
        # Grid fields are real, so the real FFT's m = 0..nlon/2 output covers
        # every retained order (nlon >= 2*truncation+2) at half the work of
        # the complex transform it replaces.
        fourier = xp.fft.rfft(f, axis=-1) / self.grid.nlon
        with self._contraction_scope():
            out = self._batched_contract(
                fourier[..., : self.truncation + 1], self._analysis_mjn
            )
        out[..., :, 0] = out[..., :, 0].real
        return out

    def _synthesize(self, coeff, basis_mnj):
        xp = self.backend.xp
        c = xp.asarray(coeff, dtype=self.backend.complex_dtype)
        self._validate_spectral(c)
        # Only orders 0..truncation are populated and the negative-m half of
        # the old full spectrum was the conjugate mirror, so the half-spectrum
        # inverse real FFT reproduces ifft(full).real to fp roundoff (irfft
        # reads only the real part of the m=0 bin, exactly the part .real
        # kept) at half the FFT work and half the temporary.
        spectrum = xp.zeros(
            (*c.shape[:-2], self.grid.nlat, self.grid.nlon // 2 + 1),
            dtype=self.backend.complex_dtype,
        )
        with self._contraction_scope():
            values = self._batched_contract(c, basis_mnj)
        spectrum[..., : self.truncation + 1] = self.grid.nlon * values
        return xp.fft.irfft(spectrum, n=self.grid.nlon, axis=-1).astype(
            self.backend.float_dtype
        )

    def inverse(self, coeff):
        return self._synthesize(coeff, self._basis_mnj)

    def inverse_meridional_derivative(self, coeff):
        """Return ∂field/∂latitude in grid space."""
        return self._synthesize(coeff, self._dbasis_mnj)

    def inverse_zonal_derivative(self, coeff):
        xp = self.backend.xp
        c = xp.asarray(coeff, dtype=self.backend.complex_dtype)
        self._validate_spectral(c)
        # The 1j*m diagonal is a precomputed vector broadcast along the
        # order axis; above-triangle entries are zero in every projected
        # state, so scaling the full rectangle equals the retired per-order
        # loop.
        return self.inverse(c * self._zonal_wavenumber)

    def gradient(self, coeff):
        """Return physical eastward and northward derivatives (per metre)."""
        east = self.inverse_zonal_derivative(coeff) / (
            self.grid.radius_m * self._coslat
        )
        north = self.inverse_meridional_derivative(coeff) / self.grid.radius_m
        return east, north

    def laplacian(self, coeff):
        xp = self.backend.xp
        c = xp.asarray(coeff, dtype=self.backend.complex_dtype)
        self._validate_spectral(c)
        return c * self._laplacian_eigen[:, None]

    def inverse_laplacian(self, coeff):
        xp = self.backend.xp
        c = xp.asarray(coeff, dtype=self.backend.complex_dtype)
        self._validate_spectral(c)
        out = xp.zeros_like(c)
        eig = self._laplacian_eigen
        out[..., 1:, :] = c[..., 1:, :] / eig[1:, None]
        return self.project(out)

    def spectral_mean_square(self, coeff) -> float:
        c = self.backend.to_numpy(coeff)
        self._validate_spectral(c)
        weight = np.ones(self.truncation + 1, dtype=np.float64)
        weight[1:] = 2.0
        total = np.sum(np.abs(c) ** 2 * weight[None, :], axis=(-2, -1))
        return float(np.asarray(total).mean() / (4.0 * np.pi))

    def power_by_degree(self, coeff) -> np.ndarray:
        c = self.backend.to_numpy(coeff)
        weight = np.ones(self.truncation + 1, dtype=np.float64)
        weight[1:] = 2.0
        return np.sum(np.abs(c) ** 2 * weight[None, :], axis=-1)

    def constant_coeff(self, value: float):
        out = self.zeros()
        out[0, 0] = float(value) * math.sqrt(4.0 * math.pi)
        return out

    def add_grid_constant(self, coeff, value: float):
        out = self.backend.xp.asarray(coeff).copy()
        out[..., 0, 0] += float(value) * math.sqrt(4.0 * math.pi)
        return out

    def transform_check(self, *, seed: int = 0, max_degree: int | None = None) -> dict:
        rng = np.random.default_rng(seed)
        t = self.truncation if max_degree is None else min(self.truncation, int(max_degree))
        coeff = np.zeros(self.spectral_shape, dtype=np.complex128)
        for n in range(t + 1):
            for m in range(n + 1):
                coeff[n, m] = rng.normal() + (0.0j if m == 0 else 1j * rng.normal())
        grid = self.inverse(self.backend.asarray(coeff, dtype=self.backend.complex_dtype))
        back = self.backend.to_numpy(self.forward(grid))
        scale = max(1.0, float(np.max(np.abs(coeff))))
        error = float(np.max(np.abs(back - coeff)) / scale)
        spatial_ms = self.grid.global_mean(self.backend.to_numpy(grid) ** 2)
        spectral_ms = float(
            np.sum(
                np.abs(coeff) ** 2
                * np.where(np.arange(self.truncation + 1)[None, :] == 0, 1.0, 2.0)
            )
            / (4.0 * np.pi)
        )
        parseval = abs(spatial_ms - spectral_ms) / max(1.0, abs(spectral_ms))
        return {
            "truncation": self.truncation,
            "nlat": self.grid.nlat,
            "nlon": self.grid.nlon,
            "roundtrip_relative_linf": error,
            "parseval_relative_error": float(parseval),
        }

# ---- vector.py @ 8f7c5d7d7 --------------------------------------------

from dataclasses import dataclass
import math

import numpy as np



@dataclass
class VorticityDivergenceOperator:
    transform: SphericalHarmonicTransform

    def __post_init__(self) -> None:
        # State-independent analysis weights, built once at construction:
        # the retired per-order loop rebuilt them for every m of every call.
        # The 1j*m rotational diagonal and the quadrature/metric factors
        # fold into two small per-(latitude, order) tables applied to the
        # Fourier coefficients before the batched Legendre contraction.
        t = self.transform
        b = t.backend
        scale = 2.0 * math.pi / t.grid.radius_m
        m = np.arange(t.grid.truncation + 1, dtype=np.float64)
        self._rotation_weight = b.asarray(
            (1j * m)[None, :]
            * (t.grid.quadrature_weights / t.grid.cos_lat * scale)[:, None],
            dtype=b.complex_dtype,
        )
        self._shear_weight = b.asarray(
            (t.grid.quadrature_weights * scale)[:, None],
            dtype=b.float_dtype,
        )

    def wind_from_vordiv(self, vorticity, divergence):
        """Invert relative vorticity/divergence to eastward/northward wind.

        The scalar streamfunction and velocity potential satisfy
        ``laplacian(psi)=vorticity`` and ``laplacian(chi)=divergence``.
        Their analytic spherical-harmonic derivatives then produce the wind.
        """
        xp = self.transform.backend.xp
        psi = self.transform.inverse_laplacian(vorticity)
        chi = self.transform.inverse_laplacian(divergence)
        # One gradient over the stacked pair: the synthesis contraction
        # inside runs per call, so two separate gradients pay it twice for
        # the same arithmetic.
        d_east, d_north = self.transform.gradient(xp.stack([psi, chi]))
        u = -d_north[0] + d_east[1]
        v = d_east[0] + d_north[1]
        return u, v

    def vordiv_from_wind(self, u, v):
        """Direct vector spherical-harmonic analysis of a grid wind.

        This uses integration by parts rather than differentiating projected
        scalar U/V components.  The latter is tempting but is not the vector
        transform: it loses several percent even for low-degree wind fields.
        For ``Y_nm=P_nm(phi) exp(i m lambda)`` the coefficients are

        ``D_nm = 1/a ∫ [i m u P/cos(phi) - v dP/dphi] e^-imλ dΩ``
        ``Z_nm = 1/a ∫ [i m v P/cos(phi) + u dP/dphi] e^-imλ dΩ``.
        """
        t = self.transform
        xp = t.backend.xp
        u = xp.asarray(u, dtype=t.backend.float_dtype)
        v = xp.asarray(v, dtype=t.backend.float_dtype)
        t._validate_grid(u)
        t._validate_grid(v)
        if u.shape != v.shape:
            raise ValueError(f"u/v shape mismatch: {u.shape} != {v.shape}")

        # Real wind components: the real FFT's m = 0..nlon/2 output covers
        # every retained order (nlon >= 2*truncation+2) at half the work of
        # the complex transform it replaces.  u and v stack so each of the
        # two Legendre contractions below is one batched GEMM for the pair.
        fourier = xp.fft.rfft(xp.stack([u, v]), axis=-1) / t.grid.nlon
        fm = fourier[..., : t.grid.truncation + 1]
        with t._contraction_scope():
            rotational = t._batched_contract(
                fm * self._rotation_weight, t._basis_mjn
            )
            sheared = t._batched_contract(
                fm * self._shear_weight, t._dbasis_mjn
            )
        divergence = rotational[0] - sheared[1]
        zeta = rotational[1] + sheared[0]
        return t.project(zeta), t.project(divergence)

    def flux_curl_divergence(self, eastward_flux, northward_flux):
        return self.vordiv_from_wind(eastward_flux, northward_flux)
