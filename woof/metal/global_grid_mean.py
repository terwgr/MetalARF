"""FP64 Gaussian-grid reduction caller for the global forecast setup."""
from __future__ import annotations

import numpy as np


def submit_global_grid_mean(context, values, quadrature_weights) -> float:
    """Reduce a Gaussian-grid field to its area-weighted mean on Metal.

    Host fields are uploaded; a same-context MetalArray is consumed in place.
    Longitude and latitude accumulation stay on the device; only the final
    scalar returns to the Python caller.
    """
    from .array import MetalArray, MetalArrayModule
    from .generated import kernel_specs

    resident = isinstance(values, MetalArray)
    if resident:
        if values.buffer.context is not context:
            raise ValueError("global Metal mean field belongs to another context")
        if values.dtype != np.dtype(np.float64) or values.ndim != 2:
            raise TypeError("global Metal mean expects a 2-D FP64 field")
    else:
        values = np.asarray(values)
    weights = np.asarray(quadrature_weights)
    if not resident and (values.dtype != np.dtype(np.float64) or values.ndim != 2):
        raise TypeError("global Metal mean expects a 2-D FP64 field")
    nlat, nlon = map(int, values.shape)
    if weights.dtype != np.dtype(np.float64) or weights.shape != (nlat,):
        raise TypeError("global Metal mean quadrature weights must match latitude")
    if nlat < 1 or nlon < 1:
        raise ValueError("global Metal mean needs nonempty grid extents")
    metal = MetalArrayModule(context)
    rows = metal.empty((nlat,), dtype=np.float64)
    mean = metal.empty((1,), dtype=np.float64)
    resources = {
        "values": (values.view((nlat, nlon), byte_offset=values.byte_offset)
                   if resident else metal.asarray(np.ascontiguousarray(values))),
        "quadrature_weights": metal.asarray(np.ascontiguousarray(weights)),
        "weighted_rows": rows,
    }
    context.submit_kernel(
        "global_grid_mean_rows_fragment", resources,
        {"nlat": nlat, "nlon": nlon}, exact_policy=False,
        grid_size=nlat,
        threads_per_group=kernel_specs.GLOBAL_GRID_MEAN_ROWS_FRAGMENT_THREADS_PER_GROUP,
    )
    context.submit_kernel(
        "global_grid_mean_finalize_fragment",
        {"weighted_rows": rows, "mean": mean}, {"nlat": nlat},
        exact_policy=False, grid_size=1,
        threads_per_group=kernel_specs.GLOBAL_GRID_MEAN_FINALIZE_FRAGMENT_THREADS_PER_GROUP,
    )
    return float(mean.to_numpy()[0])
