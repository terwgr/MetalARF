"""Metal submission helpers for ordinary lateral flow boundaries."""

from __future__ import annotations

from .array import MetalArray


def _validate_arrays(arrays, *, shapes, name):
    if any(not isinstance(array, MetalArray) for array in arrays):
        raise TypeError(f"Metal {name} requires MetalArray resources")
    if any(array.dtype.name != "float32" or not array.flags.c_contiguous
           for array in arrays):
        raise TypeError(f"Metal {name} requires contiguous float32 resources")
    context = arrays[0].buffer.context
    for array, shape in zip(arrays, shapes):
        if array.shape != shape:
            raise ValueError(f"Metal {name} resource shape {array.shape} != {shape}")
        if array.buffer.context is not context:
            raise ValueError(f"Metal {name} resources must share a context")
    return context


def submit_flow_dependent_boundaries(fields, u_flux, v_flux, spec_zone, *,
                                     inflow_value=0.0):
    """Submit WRF flow dependent boundaries for 1..9 resident fields.

    The caller owns the public CUDA/Metal dispatch and fallback selection.
    The field sequence and u/v mass-flux staggering match ``lbc_flow.cu``.
    """
    fields = tuple(fields)
    if not fields or len(fields) > 9:
        raise ValueError("flow-dependent boundary batch needs 1..9 fields")
    if fields[0].ndim != 3:
        raise ValueError("flow-dependent boundary fields must be 3-D")
    nz, ny, nx = fields[0].shape
    if spec_zone < 1 or min(ny, nx) <= 2 * spec_zone:
        raise ValueError("spec_zone leaves no flow-dependent boundary interior")
    if any(field.shape != (nz, ny, nx) for field in fields):
        raise ValueError("flow-dependent boundary fields must share a shape")
    if u_flux.shape != (nz, ny, nx + 1):
        raise ValueError("u_flux must have shape (nz, ny, nx + 1)")
    if v_flux.shape != (nz, ny + 1, nx):
        raise ValueError("v_flux must have shape (nz, ny + 1, nx)")
    padded = fields + (fields[-1],) * (9 - len(fields))
    arrays = (*padded, u_flux, v_flux)
    frame_count = sum(
        2 * (nx - 2 * d) + 2 * max(ny - 2 * d - 2, 0)
        for d in range(spec_zone))
    context = _validate_arrays(
        arrays, shapes=(*( (nz, ny, nx) for _ in padded),
                        (nz, ny, nx + 1), (nz, ny + 1, nx)),
        name="flow-dependent boundary")
    return context.submit_kernel(
        "lbc_dependent_batch",
        {f"f{i}": padded[i] for i in range(9)}
        | {"u_flux": u_flux, "v_flux": v_flux},
        {"nfield": len(fields), "spec_zone": spec_zone,
         "nz": nz, "ny": ny, "nx": nx, "frame_count": frame_count,
         "inflow_value": float(inflow_value)},
        exact_policy=False, grid_size=nz * frame_count)


def submit_zero_gradient_w(field, spec_zone):
    """Submit the specified-domain zero-gradient rule for staggered w."""
    if not isinstance(field, MetalArray) or field.ndim != 3:
        raise TypeError("Metal zero-gradient w requires a 3-D MetalArray")
    nz, ny, nx = field.shape
    if spec_zone < 1 or min(ny, nx) <= 2 * spec_zone:
        raise ValueError("spec_zone leaves no zero-gradient boundary interior")
    frame_count = sum(
        2 * (nx - 2 * d) + 2 * max(ny - 2 * d - 2, 0)
        for d in range(spec_zone))
    _validate_arrays((field,), shapes=((nz, ny, nx),), name="zero-gradient w")
    return field.buffer.context.submit_kernel(
        "lbc_zero_gradient_w", {"field": field},
        {"spec_zone": spec_zone, "nz": nz, "ny": ny, "nx": nx,
         "frame_count": frame_count},
        exact_policy=False, grid_size=nz * frame_count)
