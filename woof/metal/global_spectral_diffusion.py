"""FP64 Metal diffusion entrypoint for Global spectral coefficient fields."""
from __future__ import annotations

import numpy as np

from .array import MetalArray


def submit_global_spectral_diffusion(coeff: MetalArray, factors) -> MetalArray:
    """Apply degree-wise hyperdiffusion to one or more stacked fields.

    ``factors`` are the FP64 factors produced by the Global NumPy reference
    expression. The trailing two coefficient dimensions are one triangular
    spectral plane; any leading dimensions are independent contiguous fields.
    Coefficients stay resident on Metal; the shader uses integer-only
    binary64 primitives and preserves the real m=0 column and triangular mask
    in every plane.
    """
    if not isinstance(coeff, MetalArray):
        raise TypeError("Global spectral diffusion requires a MetalArray")
    if coeff.dtype != np.dtype(np.complex128):
        raise TypeError("Global spectral diffusion requires complex128 storage")
    if not coeff.flags.c_contiguous or coeff.ndim < 2:
        raise TypeError(
            "Global spectral diffusion requires a contiguous field with at "
            "least two spectral dimensions"
        )
    truncation = int(coeff.shape[-2]) - 1
    if (tuple(coeff.shape[-2:]) != (truncation + 1, truncation + 1)
            or truncation < 1):
        raise ValueError(
            "Global spectral coefficient fields must end in square "
            "(T + 1, T + 1) planes, T >= 1"
        )
    factor_host = np.asarray(factors)
    if factor_host.dtype != np.dtype(np.float64):
        raise TypeError("Global spectral diffusion factors must remain FP64")
    if factor_host.shape != (truncation + 1,):
        raise ValueError("Global spectral diffusion factor count must equal T + 1")
    if not np.isfinite(factor_host).all():
        raise ValueError("Global spectral diffusion factors must be finite")
    if coeff.size > 0x7FFFFFFF:
        raise ValueError("Global spectral coefficient count exceeds the int32 ABI")

    context = coeff.buffer.context
    factor_array = MetalArray(
        context.buffer(max(factor_host.nbytes, np.dtype(np.float64).itemsize)),
        factor_host.shape,
        np.float64,
    )
    factor_array.buffer.write(np.ascontiguousarray(factor_host).tobytes())
    output = MetalArray(
        context.buffer(max(coeff.nbytes, np.dtype(np.complex128).itemsize)),
        coeff.shape,
        np.complex128,
    )
    if coeff.size == 0:
        return output
    # The typed runtime binds ``float64_bits`` resources from FP64 storage;
    # the shader reads the same eight bytes through its ulong pointer.
    coeff_bits = coeff.view((int(coeff.size), 2), np.float64)
    output_bits = output.view((int(output.size), 2), np.float64)
    factor_bits = factor_array.view(factor_array.shape, np.float64)
    context.submit_kernel(
        "global_spectral_diffusion",
        {"coefficients": coeff_bits, "factors": factor_bits,
         "output": output_bits},
        {"truncation": truncation, "coefficient_count": int(coeff.size)},
        exact_policy=False,
        grid_size=int(coeff.size),
        threads_per_group=128,
    )
    return output
