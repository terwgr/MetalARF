"""Exact exponential hyperdiffusion in total spherical wavenumber."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .transform import SphericalHarmonicTransform


@dataclass(frozen=True)
class ExponentialHyperdiffusion:
    order: int = 4
    e_folding_time_s_at_truncation: float = 21_600.0
    preserve_degree: int = 1

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("hyperdiffusion order must be >= 1")
        if not math.isfinite(self.e_folding_time_s_at_truncation) or self.e_folding_time_s_at_truncation <= 0:
            raise ValueError("e-folding time must be finite and positive")
        if self.preserve_degree < 0:
            raise ValueError("preserve_degree must be >= 0")

    def factors(self, transform: SphericalHarmonicTransform, dt_s: float):
        xp = transform.backend.xp
        n = np.arange(transform.truncation + 1, dtype=np.float64)
        lam = n * (n + 1.0)
        top = max(1.0, lam[-1])
        exponent = -(float(dt_s) / self.e_folding_time_s_at_truncation) * (lam / top) ** self.order
        factor = np.exp(exponent)
        factor[: self.preserve_degree + 1] = 1.0
        return xp.asarray(factor, dtype=transform.backend.float_dtype)

    def apply(self, coeff, transform: SphericalHarmonicTransform, dt_s: float, *,
              strength: float = 1.0, metal_context=None):
        if strength < 0 or not math.isfinite(strength):
            raise ValueError("diffusion strength must be finite and nonnegative")
        # The Global Metal slice operates on contiguous FP64 triangular
        # coefficient planes (including stacks), while transform tables and
        # exact factor construction remain on the NumPy reference path. This
        # is a device kernel, not a Metal transform or forecast backend.
        from woof.metal.array import MetalArray
        if isinstance(coeff, MetalArray):
            if transform.backend.name != "numpy":
                raise TypeError(
                    "Global Metal spectral diffusion currently requires the "
                    "NumPy FP64 transform reference for factor construction"
                )
            if np.dtype(transform.backend.float_dtype) != np.dtype(np.float64):
                raise TypeError(
                    "Global Metal spectral diffusion requires an FP64 "
                    "transform reference"
                )
            if coeff.dtype != np.dtype(np.complex128):
                raise TypeError(
                    "Global Metal spectral diffusion requires complex128 "
                    "coefficients; lower precision is unsupported"
                )
            expected = (transform.truncation + 1, transform.truncation + 1)
            if (coeff.ndim < 2 or tuple(coeff.shape[-2:]) != expected
                    or not coeff.flags.c_contiguous):
                raise ValueError(
                    "Global Metal spectral diffusion requires a C-contiguous "
                    f"field or stack ending in shape {expected}, "
                    f"got {tuple(coeff.shape)}"
                )
            factor = self.factors(transform, dt_s) ** float(strength)
            from woof.metal.global_spectral_diffusion import (
                submit_global_spectral_diffusion,
            )
            return submit_global_spectral_diffusion(coeff, factor)
        if (coeff.ndim < 2 or tuple(coeff.shape[-2:]) != (
                transform.truncation + 1, transform.truncation + 1)):
            raise ValueError(
                "Global spectral diffusion requires coefficient fields whose "
                "last two dimensions match the transform's spectral shape"
            )
        factor = self.factors(transform, dt_s) ** float(strength)
        if metal_context is not None:
            if transform.backend.name != "numpy" or transform.backend.float_dtype != np.dtype(np.float64):
                raise TypeError(
                    "staged Global Metal diffusion requires the NumPy FP64 "
                    "transform reference"
                )
            if coeff.dtype != np.dtype(np.complex128):
                raise TypeError(
                    "staged Global Metal diffusion requires complex128 "
                    "coefficients"
                )
            from woof.metal.array import MetalArrayModule
            from woof.metal.global_spectral_diffusion import (
                submit_global_spectral_diffusion,
            )
            metal_coeff = MetalArrayModule(metal_context).asarray(
                np.ascontiguousarray(coeff))
            metal_result = submit_global_spectral_diffusion(metal_coeff, factor)
            return transform.project(metal_result.to_numpy())
        degree_factors = factor.reshape(
            (1,) * (coeff.ndim - 2) + (transform.truncation + 1, 1)
        )
        return transform.project(coeff * degree_factors)
