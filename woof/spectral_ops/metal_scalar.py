"""Scalar hyperdiffusion on contiguous FP32 Metal fields.

The field stays resident on the device. Only compact statistic/status vectors
cross back to the host for the public spectral receipt and refusal contract.
Reflect mode uses the pinned even extension; explicitly admitted periodic mode
transforms the original plane without extending or cropping it.
"""

from __future__ import annotations

import math
import numpy as np

from woof.metal.array import MetalArray

from .scalar import ScalarResult
from .transfer import Hyperdiffusion


def _submit(context, name, arrays, params, size):
    return context.submit_kernel(name, arrays, params, exact_policy=False,
                                 grid_size=size)


def _reduce_partials(context, allocator, partials, batches: int,
                     count: int, width: int):
    """Reduce block summaries independently within each leading batch."""
    while count > 1:
        output_count = (count + 255) // 256
        reduced = allocator.zeros((batches, output_count, width), np.float32)
        _submit(context, "spectral_stats_reduce", {
            "partials": partials,
            "reduced": reduced,
        }, {"batches": batches, "input_count": count,
            "output_count": output_count,
            "width": width}, batches * output_count)
        partials, count = reduced, output_count
    return partials.view((batches, width), np.float32)


def hyperdiffuse_metal(array, *, dy_m: float, dx_m: float, dt_s: float,
                       spec: Hyperdiffusion, boundary: str = "reflect",
                       periodic_domain: bool = False,
                       preserve_mean: bool = True
                       ) -> ScalarResult:
    """Apply linear scalar hyperdiffusion using Metal kernels and MPSGraph."""
    if not isinstance(array, MetalArray):
        raise TypeError("Metal scalar hyperdiffusion requires a MetalArray")
    if boundary not in ("reflect", "periodic"):
        raise NotImplementedError(
            "Metal scalar hyperdiffusion supports reflect and periodic boundaries")
    if boundary == "periodic" and not periodic_domain:
        raise ValueError(
            "periodic spectral wrapping requires periodic_domain=true")
    if boundary == "reflect" and periodic_domain:
        raise NotImplementedError(
            "Metal reflect hyperdiffusion does not admit periodic_domain=true")
    if array.dtype != np.dtype(np.float32):
        raise TypeError("Metal scalar hyperdiffusion requires FP32 storage")
    if not array.flags.c_contiguous or array.byte_offset != 0:
        raise ValueError("Metal spectral fields must be contiguous base arrays")
    if array.ndim < 2 or array.size == 0 or any(n < 2 for n in array.shape[-2:]):
        raise ValueError("a scalar spectral operand needs two non-empty axes of length >= 2")
    if not all(math.isfinite(v) and v > 0.0 for v in (dx_m, dy_m)):
        raise ValueError("dx_m and dy_m must be positive and finite")
    if not math.isfinite(dt_s) or dt_s < 0.0:
        raise ValueError("dt_s must be non-negative and finite")
    spec.validate()
    ny, nx = array.shape[-2:]
    batches = int(array.size // (ny * nx))
    plane_count = ny * nx
    if boundary == "periodic":
        # Keep the CPU/source geometry and wavenumber basis exactly on the
        # caller's plane. MetalFFT does not resize, so this also avoids the
        # reflect path's extension/crop while preserving original dx and dy.
        ey, ex = ny, nx
    else:
        ey, ex = 2 * ny, 2 * nx
    if max(array.size, batches * ey * ex, ey, ex, batches) > 0x7FFFFFFF:
        raise ValueError("Metal scalar spectral extent exceeds typed dispatch limits")
    context = array.buffer.context
    original = array.view((array.size,))
    # MetalArray views intentionally do not own an array module; allocations
    # use the already-associated context through its standard array facade.
    from woof.metal.array import MetalArrayModule
    allocator = MetalArrayModule(context)
    if boundary == "periodic":
        transform_input = original.view((batches, ny, nx), np.float32)
    else:
        transform_input = allocator.zeros((batches, ey, ex), np.float32)
        _submit(context, "spectral_reflect_extend",
                {"input": original, "output": transform_input},
                {"count": array.size, "batches": batches, "ny": ny, "nx": nx,
                 "ey": ey, "ex": ex}, batches * ey * ex)

    half_width = ex // 2 + 1
    coefficients = allocator.fft.rfft2(transform_input, axes=(-2, -1))
    # Typed Metal resources are FP32, so expose each complex64 value as its
    # two adjacent FP32 lanes without moving or copying the spectrum.
    coefficient_lanes = coefficients.view(
        (batches, ey, half_width, 2), np.float32)
    _submit(context, "spectral_transfer", {"coefficients": coefficient_lanes}, {
        "batches": batches, "ey": ey, "ex": ex,
        "half_width": half_width, "dy": float(dy_m), "dx": float(dx_m),
        "dt": float(dt_s),
        "reference_wavelength": float(spec.reference_wavelength_m),
        "efold": float(spec.e_fold_time_s),
        "protect_wavelength": float(spec.protect_wavelength_m or 0.0),
        "max_damping": float(spec.maximum_damping_fraction),
        "order": int(spec.order),
    }, batches * ey * half_width * 2)
    filtered = allocator.fft.irfft2(
        coefficients, s=(ey, ex), axes=(-2, -1))

    partial_count = (plane_count + 255) // 256
    pre_stats = allocator.zeros((batches, partial_count, 3), np.float32)
    _submit(context, "spectral_pair_stats", {
        "original": original.view((batches, plane_count), np.float32),
        "filtered": filtered,
        "stats": pre_stats,
    }, {"plane_count": plane_count, "batches": batches,
        "partial_count": partial_count, "ny": ny, "nx": nx,
        "ey": ey, "ex": ex}, batches * partial_count)
    before_after = _reduce_partials(
        context, allocator, pre_stats, batches, partial_count, 3).view(
            (batches, 3), np.float32)
    corrections = allocator.zeros((batches,), np.float32)
    status = allocator.zeros((1,), np.float32)
    _submit(context, "spectral_plane_correction", {
        "partial_sums": before_after,
        "corrections": corrections,
        "status": status,
    }, {"plane_count": plane_count, "batches": batches,
        "preserve_mean": int(bool(preserve_mean))}, 1)
    if float(status.to_numpy()[0]) != 1.0:
        raise ValueError("scalar spectral operand or transform result is non-finite")

    output_flat = allocator.zeros((array.size,), np.float32)
    _submit(context, "spectral_recenter_crop", {
        "filtered": filtered,
        "corrections": corrections,
        "output": output_flat,
    }, {"count": array.size, "batches": batches, "ny": ny, "nx": nx,
        "ey": ey, "ex": ex}, array.size)
    output = output_flat.view(array.shape, np.float32)

    final_partial_count = (array.size + 255) // 256
    stats = allocator.zeros((1, final_partial_count, 7), np.float32)
    _submit(context, "spectral_final_stats", {
        "original": original, "output": output_flat, "stats": stats,
    }, {"count": array.size, "partial_count": final_partial_count},
       final_partial_count)
    values = _reduce_partials(
        context, allocator, stats, 1, final_partial_count, 7).to_numpy()[0]
    if values[6] != 1.0:
        raise ValueError("scalar spectral result is non-finite")
    # The GPU reduction preserves sums for stable hierarchical aggregation;
    # normalize the public mean and RMS metrics only after the final row.
    return ScalarResult(
        values=output,
        mean_before=float(values[0] / array.size),
        mean_after=float(values[1] / array.size),
        rms_increment=float(np.sqrt(values[2] / array.size)),
        max_abs_increment=float(values[3]),
        minimum_before=float(values[4]), minimum_after=float(values[5]),
    )


__all__ = ["hyperdiffuse_metal"]
