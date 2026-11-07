"""Typed submissions for the two MP28 reflectivity source contracts."""
from __future__ import annotations

import numpy as np

from .array import MetalArray


def _fields(qv, qr, nr, qs, qg, ng, temperature, pressure, refl):
    arrays = {"qv": qv, "qr": qr, "nr": nr, "qs": qs, "qg": qg,
              "ng": ng, "temperature": temperature, "pressure": pressure,
              "refl": refl}
    if not isinstance(refl, MetalArray) or refl.ndim != 3:
        raise TypeError("Thompson reflectivity requires a 3-D MetalArray")
    shape = tuple(map(int, refl.shape))
    nz, ny, nx = shape
    count = nz * ny * nx
    if min(shape) < 1 or count > np.iinfo(np.int32).max:
        raise ValueError("Thompson reflectivity dimensions must be nonempty and fit int32")
    context = refl.buffer.context
    for name, value in arrays.items():
        if (not isinstance(value, MetalArray) or value.shape != shape
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"Thompson reflectivity {name} must be contiguous float32 with shape {shape}")
        if value.buffer.context is not context:
            raise ValueError("Thompson reflectivity fields must share one Metal context")
    flat = {name: value.view((count,), byte_offset=value.byte_offset)
            for name, value in arrays.items()}
    return flat, context, {"nz": nz, "ny": ny, "nx": nx}


def _tables(context):
    from woof.core.refl import radar_init

    rc = radar_init()
    words = np.ascontiguousarray(np.concatenate((
        rc.xxds, rc.xdts, rc.xxdg, rc.xdtg, rc.simpson)),
        dtype=np.float64)
    tables = MetalArray(context.buffer(words.nbytes), words.shape, np.float64)
    tables.buffer.write(words.tobytes())
    return tables, rc


def submit_refl10cm(qv, qr, nr, qs, qg, ng, temperature, pressure, refl,
                     *, melting=True):
    resources, context, params = _fields(
        qv, qr, nr, qs, qg, ng, temperature, pressure, refl)
    resources["tables"], _ = _tables(context)
    params["melting"] = int(bool(melting))
    return context.submit_kernel(
        "thompson_aa_refl10cm", resources, params,
        exact_policy=False, grid_size=params["ny"] * params["nx"], threads_per_group=32)


def submit_wrf39_refl10cm(qv, qr, nr, qs, qg, ng, temperature, pressure, refl,
                           *, melting=True):
    resources, context, params = _fields(
        qv, qr, nr, qs, qg, ng, temperature, pressure, refl)
    if params["nz"] > 256:
        raise ValueError("Thompson fork reflectivity supports at most 256 levels")
    if not melting:
        return context.submit_kernel(
            "thompson_aa_wrf39_refl10cm_dry", resources,
            {"n": refl.size}, exact_policy=False, grid_size=refl.size,
            threads_per_group=32)
    resources["tables"], rc = _tables(context)
    values = np.asarray((rc.k_w, rc.m_w_0.real, rc.m_w_0.imag,
                         rc.m_i_0.real, rc.m_i_0.imag), dtype=np.float64)
    constants = MetalArray(context.buffer(values.nbytes), values.shape, np.float64)
    constants.buffer.write(values.tobytes())
    resources["radar_constants"] = constants
    return context.submit_kernel(
        "thompson_aa_wrf39_refl10cm", resources, params,
        exact_policy=False, grid_size=params["ny"] * params["nx"], threads_per_group=32)
