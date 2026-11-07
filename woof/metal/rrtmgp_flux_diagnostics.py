"""Copy radiation-column boundary diagnostics through existing Metal kernels."""

from __future__ import annotations

import numpy as np

from .array import MetalArray, MetalArrayModule
from .nest_family import submit_rect_gather_f32
from .rrtmgp_cloud_family import submit_store_model_flux_pair


def extract_flux_level(flux, destination, columns, level, *, daylight=None):
    """Copy one interface from a contiguous column window, preserving words."""
    if not isinstance(flux, MetalArray) or flux.ndim != 2:
        raise TypeError("radiation flux must be a rank-two MetalArray")
    ncol, nlev = flux.shape
    if (not isinstance(destination, MetalArray) or destination.ndim != 1
            or flux.dtype != np.dtype(np.float32)
            or destination.dtype != flux.dtype
            or not flux.flags.c_contiguous or not destination.flags.c_contiguous
            or flux.buffer.context is not destination.buffer.context):
        raise TypeError("radiation flux and destination must share contiguous FP32 storage")
    if not isinstance(columns, slice):
        raise TypeError("Metal radiation diagnostics require a contiguous column window")
    start, stop, step = columns.indices(destination.size)
    if step != 1 or stop - start != ncol:
        raise ValueError("radiation diagnostic window does not match the flux columns")
    level = int(level)
    if level < 0:
        level += nlev
    if not 0 <= level < nlev:
        raise ValueError("radiation diagnostic interface is outside the column")
    context = flux.buffer.context
    target = destination.view((ncol, 1),
        byte_offset=destination.byte_offset + start * destination.dtype.itemsize)
    mask = None
    if daylight is not None:
        mask = MetalArrayModule(context).asarray(daylight, dtype=np.float32)
        if mask.shape != (ncol,) or mask.buffer.context is not context:
            raise ValueError("radiation daylight mask must match the column window")
    if not ncol:
        return
    submit_rect_gather_f32(context, source=flux.view((1, ncol, nlev)),
        packed=target.view((1, ncol, 1)), j0=0, i0=level,
        nz=1, ny=ncol, nx=1)
    if mask is not None:
        submit_store_model_flux_pair(context, target, target, mask,
            target, target, ncol=ncol, source_nlev=1, model_nlev=1,
            apply_daylight=True)


def columns_to_grid(columns, *, ny, nx):
    """Invert the column pack using the same transpose kernel."""
    if not isinstance(columns, MetalArray) or columns.ndim != 2:
        raise TypeError("radiation columns must be a rank-two MetalArray")
    ncol, nz = columns.shape
    if int(ny) * int(nx) != ncol:
        raise ValueError("radiation columns do not match the model grid")
    from .rrtmgp_prepare_family import submit_pack_columns

    return submit_pack_columns(columns.view((ncol, 1, nz))).view(
        (nz, int(ny), int(nx)))
