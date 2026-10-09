"""Spectral kinetic energy by total degree from the dycore's own coefficients.

The state IS spectral, so no transform is paid: with the transform's
complex-orthonormal convention (grid mean square of a field
= sum_{n,m} w_m |c_nm|^2 / (4 pi), w_0 = 1, w_{m>0} = 2 - the identity
``transform_check`` verifies) and the streamfunction psi_nm = -a^2
zeta_nm / (n(n+1)), the rotational kinetic energy per unit mass (global
mean of 0.5 |v_rot|^2) at total degree n is

    KE_rot(n) = a^2 / (2 n(n+1)) * sum_m w_m |zeta_nm|^2 / (4 pi)

and KE_div(n) the same with the divergence coefficients.  The cross term
v_rot . v_div = J(psi, chi) integrates to zero over the sphere, so
sum_n KE_rot(n) + KE_div(n) equals the grid global mean of 0.5 (u^2 + v^2)
of the synthesised wind - exactly, on the quadratic (dealiased) Gaussian
grid, because (u cos phi)^2 / cos^2 phi and (v cos phi)^2 / cos^2 phi are
polynomials in sin(phi) of degree <= 2T.
``tests/test_arwen_global_insitu.py::test_spectral_kinetic_energy_is_parseval_anchored``
proves that identity to roundoff.
"""
from __future__ import annotations

import math

import numpy as np


class SpectralKineticEnergy:
    """Per-level rotational/divergent KE by total degree; device-resident."""

    def __init__(self, transform):
        xp = transform.backend.xp
        self.xp = xp
        self.truncation = int(transform.truncation)
        n = np.arange(self.truncation + 1, dtype=np.float64)
        radius = float(transform.grid.radius_m)
        factor = np.zeros_like(n)
        factor[1:] = radius * radius / (2.0 * n[1:] * (n[1:] + 1.0)) / (4.0 * math.pi)
        order_weight = np.full(self.truncation + 1, 2.0)
        order_weight[0] = 1.0
        self._factor = xp.asarray(factor, dtype=xp.float64)
        self._order_weight = xp.asarray(order_weight, dtype=xp.float64)
        # Top decile of total degrees: the pile-up band the tripwire watches.
        self.band_start = self.truncation - self.truncation // 10

    def by_degree(self, coeff):
        """``coeff[..., n, m]`` -> KE per unit mass by degree, ``[..., n]``."""
        xp = self.xp
        power = xp.sum(
            (coeff.real.astype(xp.float64) ** 2 + coeff.imag.astype(xp.float64) ** 2)
            * self._order_weight,
            axis=-1,
        )
        return power * self._factor

    def sample(self, vorticity, divergence):
        """One flat device vector: level-summed rot/div spectra by degree,
        then per-level top-decile and total KE for rot and div."""
        from ...metal.array import MetalArray, MetalArrayModule
        if isinstance(vorticity, MetalArray) or isinstance(divergence, MetalArray):
            if not isinstance(vorticity, MetalArray) or not isinstance(divergence, MetalArray):
                raise TypeError("Global spectral observer requires both resident coefficient arrays")
            if (vorticity.dtype != np.dtype(np.complex128)
                    or divergence.dtype != np.dtype(np.complex128)
                    or vorticity.shape != divergence.shape
                    or len(vorticity.shape) != 3
                    or vorticity.buffer.context is not divergence.buffer.context):
                raise ValueError("Global spectral observer requires matching resident complex128 coefficients")
            from ...metal.generated import kernel_specs

            context = vorticity.buffer.context
            metal = MetalArrayModule(context)
            nlevel, nterm, norder = (int(value) for value in vorticity.shape)
            if nterm != self.truncation + 1 or norder < nterm:
                raise ValueError("Global spectral coefficient shape does not match truncation")
            per_degree = metal.empty((2 * nlevel * nterm,), dtype=np.float64)
            context.submit_kernel(
                "global_insitu_spectra_modes_fragment",
                {"vorticity": vorticity.view((2 * vorticity.size,), dtype=np.float64),
                 "divergence": divergence.view((2 * divergence.size,), dtype=np.float64),
                 "factor": metal.asarray(np.asarray(self._factor, dtype=np.float64)),
                 "order_weight": metal.asarray(np.asarray(self._order_weight, dtype=np.float64)),
                 "per_degree": per_degree},
                {"nlevel": nlevel, "nterm": nterm},
                exact_policy=False, grid_size=2 * nlevel * nterm,
                threads_per_group=kernel_specs.GLOBAL_INSITU_SPECTRA_MODES_FRAGMENT_THREADS_PER_GROUP,
            )
            row_size = 2 * nterm + 4 * nlevel
            row = metal.empty((row_size,), dtype=np.float64)
            context.submit_kernel(
                "global_insitu_spectra_pack_fragment",
                {"per_degree": per_degree, "row": row},
                {"nlevel": nlevel, "nterm": nterm,
                 "band_start": self.band_start},
                exact_policy=False, grid_size=row_size,
                threads_per_group=kernel_specs.GLOBAL_INSITU_SPECTRA_PACK_FRAGMENT_THREADS_PER_GROUP,
            )
            return row
        xp = self.xp
        rot = self.by_degree(vorticity)
        div = self.by_degree(divergence)
        return xp.concatenate([
            xp.sum(rot, axis=0),
            xp.sum(div, axis=0),
            xp.sum(rot[..., self.band_start:], axis=-1),
            xp.sum(div[..., self.band_start:], axis=-1),
            xp.sum(rot, axis=-1),
            xp.sum(div, axis=-1),
        ])

    def unpack(self, vector, nlev: int) -> dict[str, list[float]]:
        t1 = self.truncation + 1
        vector = np.asarray(vector, dtype=np.float64)
        expected = 2 * t1 + 4 * nlev
        if vector.shape != (expected,):
            raise ValueError(f"spectra vector has {vector.shape}, expected ({expected},)")
        cursor = 0

        def take(count):
            nonlocal cursor
            out = vector[cursor:cursor + count]
            cursor += count
            return [float(v) for v in out]

        return {
            "rot_by_degree": take(t1),
            "div_by_degree": take(t1),
            "rot_top_decile_by_level": take(nlev),
            "div_top_decile_by_level": take(nlev),
            "rot_total_by_level": take(nlev),
            "div_total_by_level": take(nlev),
        }


__all__ = ["SpectralKineticEnergy"]
