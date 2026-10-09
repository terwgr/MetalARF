"""Vector spherical-harmonic analysis through vorticity/divergence."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .transform import FourierWaist, SphericalHarmonicTransform


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
        from woof.metal.array import MetalArray, MetalArrayModule

        if isinstance(vorticity, MetalArray) or isinstance(divergence, MetalArray):
            context = getattr(self.transform, "metal_context", None)
            if (not isinstance(vorticity, MetalArray)
                    or not isinstance(divergence, MetalArray)
                    or context is None
                    or vorticity.buffer.context is not context
                    or divergence.buffer.context is not context
                    or vorticity.dtype != np.dtype(np.complex128)
                    or divergence.dtype != np.dtype(np.complex128)
                    or vorticity.shape != divergence.shape):
                raise TypeError(
                    "resident Global wind synthesis needs matching complex128 "
                    "vorticity and divergence on one transform context"
                )
            self.transform._validate_spectral(vorticity)
            shape = tuple(int(extent) for extent in vorticity.shape)
            nlevel, nlat, nlon = shape[0], *self.transform.grid.shape
            count = nlevel * nlat * nlon
            metal = MetalArrayModule(context)
            # Match the NumPy path: winds are derivatives of the stream
            # function and velocity potential, not of vorticity and
            # divergence themselves. Skipping these inverse Laplacians
            # leaves only roundoff-sized winds after the 1/radius metric.
            streamfunction = self.transform.inverse_laplacian(vorticity)
            velocity_potential = self.transform.inverse_laplacian(divergence)
            pair = metal.empty((2, *shape), dtype=np.complex128)
            context.submit_bookkeeping((
                (streamfunction, pair.view(
                    shape, dtype=np.complex128,
                    byte_offset=pair.byte_offset,
                )),
                (velocity_potential, pair.view(
                    shape, dtype=np.complex128,
                    byte_offset=pair.byte_offset + streamfunction.nbytes,
                )),
            ))
            del streamfunction, velocity_potential
            east_waist, north_waist = self.transform.gradient_waists(pair)
            east = self.transform.waist_to_grid(
                east_waist, keep_resident=True
            )
            north = self.transform.waist_to_grid(
                north_waist, keep_resident=True
            )
            u = metal.empty((count,), dtype=np.float64)
            v = metal.empty((count,), dtype=np.float64)
            from woof.metal.generated import kernel_specs

            context.submit_kernel(
                "global_semilag_wind_synthesis_fragment",
                {"east": east.view((2 * count,), dtype=np.float64),
                 "north": north.view((2 * count,), dtype=np.float64),
                 "coslat": metal.asarray(np.ascontiguousarray(
                     np.asarray(self.transform._coslat, dtype=np.float64)
                     .reshape(-1))),
                 "radius": metal.asarray(np.asarray(
                     [self.transform.grid.radius_m], dtype=np.float64)),
                 "u": u, "v": v},
                {"nlevel": nlevel, "nlat": nlat, "nlon": nlon},
                exact_policy=False, grid_size=count,
                threads_per_group=(
                    kernel_specs.GLOBAL_SEMILAG_WIND_SYNTHESIS_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            grid_shape = (nlevel, nlat, nlon)
            return (
                u.view(grid_shape, dtype=np.float64),
                v.view(grid_shape, dtype=np.float64),
            )
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
        from woof.metal.array import MetalArray
        if (isinstance(u, MetalArray) or isinstance(v, MetalArray)):
            if not isinstance(u, MetalArray) or not isinstance(v, MetalArray):
                raise TypeError(
                    "resident Global wind analysis requires both components on Metal"
                )
            context = getattr(t, "metal_context", None)
            if (context is None or u.buffer.context is not context
                    or v.buffer.context is not context
                    or u.dtype != np.dtype(np.float64)
                    or v.dtype != np.dtype(np.float64)
                    or u.shape != v.shape):
                raise ValueError(
                    "resident Global wind components need one FP64 transform context and shape"
                )
            from woof.metal.array import MetalArrayModule
            from woof.metal.generated import kernel_specs
            shape = tuple(map(int, u.shape))
            if shape[-2:] != t.grid.shape:
                raise ValueError(f"wind grid is {shape[-2:]}, expected {t.grid.shape}")
            count = int(u.size)
            metal = MetalArrayModule(context)
            pair = metal.empty((2 * count,), dtype=np.float64)
            context.submit_kernel(
                "global_semilag_wind_pair_fragment",
                {"u": metal.asarray(u).view((count,), dtype=np.float64),
                 "v": metal.asarray(v).view((count,), dtype=np.float64),
                 "pair": pair},
                {"count": count}, exact_policy=False, grid_size=count,
                threads_per_group=(
                    kernel_specs.GLOBAL_SEMILAG_WIND_PAIR_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            pair = pair.view((2, *shape), dtype=np.float64)
            return self._vordiv_from_fourier(self._wind_fourier(pair))
        u = xp.asarray(u, dtype=t.backend.float_dtype)
        v = xp.asarray(v, dtype=t.backend.float_dtype)
        t._validate_grid(u)
        t._validate_grid(v)
        if u.shape != v.shape:
            raise ValueError(f"u/v shape mismatch: {u.shape} != {v.shape}")
        return self._vordiv_from_fourier(self._wind_fourier(xp.stack([u, v])))

    def vordiv_from_wind_waist(self, waist: FourierWaist):
        """:meth:`vordiv_from_wind` from a waist a band pipeline filled.

        The two weight tables are ``(nlat, T+1)`` and both contractions
        reduce the latitude axis, so the waist is read WHOLE however many
        bands wrote it and the coefficients are the resident call's.
        """
        return self._vordiv_from_fourier(waist)

    def _wind_fourier(self, pair, *, bands: int | None = None) -> FourierWaist:
        """The Fourier waist of a stacked ``(2, ..., nlat, nlon)`` wind pair.

        The first half of :meth:`vordiv_from_wind`, split out so a caller
        that owns the pair can release it before the Legendre contractions
        run (the dycore's scalar advection: a 2.29 GiB pair per chunk at
        T533 float32).  Real wind components: the real FFT's m = 0..nlon/2
        output covers every retained order (nlon >= 2*truncation+2) at
        half the work of the complex transform it replaces.  u and v are
        stacked so each of the two Legendre contractions is one batched
        GEMM for the pair.

        This split predates the design that named it: the seam it cuts is
        exactly the transform's waist, so it is that object now rather
        than a second, private, full-width zonal spectrum.  The retained
        orders are cut here, where :meth:`_vordiv_from_fourier` used to
        cut them, so the waist a banded fill writes is the operand the
        weights read.  Cutting them here makes the ONE-BAND call cheaper
        rather than neutral.  What this method returned before was the
        full-width ``(2, ..., nlat, nlon//2+1)`` quotient, and it stayed
        alive through both contractions because the cut downstream was a
        view of it; what it returns now is the ``(2, ..., nlat, T+1)``
        waist, two thirds of that width (1.643 against 2.467 GB for a
        six-field forty-level chunk at T533 float32), and the full-width
        rfft output dies with the divide instead of outliving it.  At B
        bands the full-width buffer never exists at full size at all.
        """
        return self.transform.fourier_waist(pair, bands=bands)

    def _vordiv_from_fourier(self, fourier):
        """The second half of :meth:`vordiv_from_wind`: the contractions.

        Takes the waist :meth:`_wind_fourier` fills, or a raw zonal
        spectrum wide enough to cut, and reads it WHOLE IN LATITUDE:
        both weight tables are ``(nlat, T+1)`` and both contractions
        reduce the latitude axis, so nothing here bands and nothing here
        moves when the fill did.
        """
        t = self.transform
        raw = fourier.values if isinstance(fourier, FourierWaist) else fourier
        from woof.metal.array import MetalArray, MetalArrayModule
        if isinstance(raw, MetalArray):
            context = getattr(t, "metal_context", None)
            if (context is None or raw.buffer.context is not context
                    or raw.dtype != np.dtype(np.complex128)):
                raise TypeError("resident Global vector waist must be complex128 on its transform context")
            nlat = int(t.grid.nlat)
            nfreq = int(t.grid.truncation + 1)
            if raw.shape[-2:] != (nlat, nfreq):
                raise ValueError("resident Global vector waist has an unexpected latitude/order shape")
            from woof.metal.generated import kernel_specs
            metal = MetalArrayModule(context)
            if raw.ndim > 4:
                batch_shape = tuple(map(int, raw.shape[1:-3]))
                batch_count = int(np.prod(batch_shape, dtype=np.int64))
                nlevel = int(raw.shape[-3])
                plane_shape = (nlevel, nlat, nfreq)
                plane_bytes = int(np.prod(plane_shape, dtype=np.int64)) * np.dtype(
                    np.complex128
                ).itemsize
                results = [[], []]
                for index in range(batch_count):
                    pair = metal.empty((2, *plane_shape), dtype=np.complex128)
                    context.submit_bookkeeping((
                        (raw.view(
                            plane_shape, dtype=np.complex128,
                            byte_offset=raw.byte_offset + index * plane_bytes,
                        ), pair.view(
                            plane_shape, dtype=np.complex128,
                            byte_offset=pair.byte_offset,
                        )),
                        (raw.view(
                            plane_shape, dtype=np.complex128,
                            byte_offset=raw.byte_offset
                            + (batch_count + index) * plane_bytes,
                        ), pair.view(
                            plane_shape, dtype=np.complex128,
                            byte_offset=pair.byte_offset + plane_bytes,
                        )),
                    ))
                    vorticity, divergence = self._vordiv_from_fourier(pair)
                    results[0].append(vorticity)
                    results[1].append(divergence)
                output_shape = tuple(map(int, results[0][0].shape))
                outputs = [
                    metal.empty((*batch_shape, *output_shape), dtype=np.complex128),
                    metal.empty((*batch_shape, *output_shape), dtype=np.complex128),
                ]
                for index in range(batch_count):
                    for output_index in range(2):
                        value = results[output_index][index]
                        context.submit_bookkeeping(((value, outputs[output_index].view(
                            output_shape, dtype=np.complex128,
                            byte_offset=outputs[output_index].byte_offset
                            + index * value.nbytes,
                        )),))
                return tuple(outputs)
            count = int(raw.size)
            rotated = metal.empty(raw.shape, dtype=np.complex128)
            sheared = metal.empty(raw.shape, dtype=np.complex128)
            rotation_weight = metal.asarray(np.ascontiguousarray(
                self._rotation_weight, dtype=np.complex128
            ))
            shear_weight = metal.asarray(np.ascontiguousarray(
                np.broadcast_to(self._shear_weight, (nlat, nfreq)),
                dtype=np.float64,
            ))
            context.submit_kernel(
                "global_semilag_vector_weights_fragment",
                {"values": metal.asarray(raw).view((2 * count,), dtype=np.float64),
                 "rotation_weight": rotation_weight.view(
                     (2 * rotation_weight.size,), dtype=np.float64
                 ),
                 "shear_weight": shear_weight.view((shear_weight.size,), dtype=np.float64),
                 "rotated": rotated.view((2 * count,), dtype=np.float64),
                 "sheared": sheared.view((2 * count,), dtype=np.float64)},
                {"count": count, "nlat": nlat, "nfreq": nfreq},
                exact_policy=False, grid_size=count,
                threads_per_group=(
                    kernel_specs.GLOBAL_SEMILAG_VECTOR_WEIGHTS_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            with t._contraction_scope():
                rotational = t._batched_contract(
                    rotated, t._basis, transposed=True
                )
                sheared_contract = t._batched_contract(
                    sheared, t._derivative_basis, transposed=True
                )
            nlevel, ndegree, mcount = map(int, rotational.shape[1:])
            output_count = nlevel * ndegree * mcount
            divergence = metal.empty((output_count,), dtype=np.complex128)
            vorticity = metal.empty((output_count,), dtype=np.complex128)
            context.submit_kernel(
                "global_semilag_vordiv_combine_fragment",
                {"rotated": rotational.view((2 * rotational.size,), dtype=np.float64),
                 "sheared": sheared_contract.view(
                     (2 * sheared_contract.size,), dtype=np.float64
                 ),
                 "divergence": divergence.view(
                     (2 * output_count,), dtype=np.float64
                 ),
                 "vorticity": vorticity.view(
                     (2 * output_count,), dtype=np.float64
                 )},
                {"nlevel": nlevel, "ndegree": ndegree, "mcount": mcount},
                exact_policy=False, grid_size=output_count,
                threads_per_group=(
                    kernel_specs.GLOBAL_SEMILAG_VORDIV_COMBINE_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            shape = tuple(map(int, rotational.shape[1:]))
            return (vorticity.view(shape, dtype=np.complex128),
                    divergence.view(shape, dtype=np.complex128))
        fm = raw[..., : t.grid.truncation + 1]
        with t._contraction_scope():
            rotational = t._batched_contract(
                fm * self._rotation_weight, t._basis, transposed=True
            )
            sheared = t._batched_contract(
                fm * self._shear_weight, t._derivative_basis, transposed=True
            )
        divergence = rotational[0] - sheared[1]
        zeta = rotational[1] + sheared[0]
        return t.project(zeta), t.project(divergence)

    def flux_curl_divergence(self, eastward_flux, northward_flux):
        return self.vordiv_from_wind(eastward_flux, northward_flux)
