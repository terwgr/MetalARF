"""Metal scaling of the small table metadata in a physics parameter set."""

from __future__ import annotations

import numpy as np

from .array import MetalArrayModule


def scale_physics_param_values(context, values, factors, crop_flags):
    """Multiply in binary64 and round once to binary32, as the table source does."""
    count = len(values)
    if len(factors) != count or len(crop_flags) != count:
        raise ValueError("physics parameter values, factors and crop flags differ in length")
    if not count:
        return ()
    if count > 0x7fffffff:
        raise ValueError("physics parameter edit count exceeds int32")
    inputs = np.asarray(values, dtype=np.float64)
    multipliers = np.asarray(factors, dtype=np.float64)
    crops = np.asarray(crop_flags, dtype=np.int32)
    if any(array.shape != (count,) for array in (inputs, multipliers, crops)):
        raise ValueError("physics parameter values, factors and crop flags must be one-dimensional")
    arrays = MetalArrayModule(context)
    outputs = arrays.empty((count,), dtype=np.float32)
    status = arrays.empty((count,), dtype=np.int32)
    context.submit_kernel("scale_physics_param_values", {
        "values": arrays.asarray(inputs),
        "factors": arrays.asarray(multipliers),
        "crop_flags": arrays.asarray(crops), "outputs": outputs, "status": status},
        {"count": count}, exact_policy=False, grid_size=count).wait()
    for index, code in enumerate(status.to_numpy()):
        if code == 1:
            raise ValueError(f"physics parameter edit {index} produced a nonfinite value")
        if code == 2:
            raise ValueError(f"physics parameter edit {index} produced a nonpositive value")
        if code == 3:
            raise ValueError(
                f"physics parameter edit {index} produced seasonal-crop roughness "
                "at or below the 0.125 m surface-kernel decrement")
    return tuple(outputs.to_numpy().tolist())
