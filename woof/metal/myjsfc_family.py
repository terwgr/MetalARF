"""Metal submission boundary for the MYJ Eta-similarity surface layer."""

from __future__ import annotations

import threading

import numpy as np

from woof.core.myjsfc_tables import KZTM, build_psi_tables
from woof.wrf_exact import ENABLED as WRF_EXACT


_COLUMN_INPUTS = ("dz", "tke")
_SURFACE_INPUTS = (
    "u1", "v1", "t1", "th1", "qv1", "qc1", "p1", "psfc",
    "tsk", "xland", "mavail", "z0base",
)
_INOUT = ("ust", "znt", "thz0", "qz0", "uz0", "vz0", "qsfc",
          "akhs", "akms")
_OUTPUTS = (
    "rmol", "ct", "pblh", "rib", "chs", "chs2", "cqs2", "hfx",
    "qfx", "lh", "flhc", "flqc", "qgh", "cpm", "u10", "v10",
    "t2", "th2", "tshltr", "th10", "q2", "qshltr", "q10",
    "pshltr", "u10e", "v10e",
)
_TABLES = ("psim1", "psih1", "psim2", "psih2")
_TABLE_LOCK = threading.RLock()


def _device_tables(context):
    """Upload the shared MYJSFCINIT tables once per Metal context."""
    cached = getattr(context, "_myjsfc_psi_tables", None)
    if cached is not None:
        return cached
    with _TABLE_LOCK:
        cached = getattr(context, "_myjsfc_psi_tables", None)
        if cached is None:
            from .array import MetalArrayModule

            tables = build_psi_tables()
            module = MetalArrayModule(context)
            arrays = {name: module.asarray(tables[name], dtype=np.float32)
                      for name in _TABLES}
            scalars = {name: float(np.float32(tables[name])) for name in (
                "dzeta1", "dzeta2", "ztmin1", "ztmax1", "ztmin2",
                "ztmax2", "fh01", "fh02")}
            cached = (arrays, scalars)
            # The context owns the device buffers; their lifetime matches its
            # pipeline and command queue, just as the CUDA launcher cache does.
            context._myjsfc_psi_tables = cached
    return cached


def submit_myjsfc_column(columns, surface, state, outputs, *, itimestep: int):
    """Submit one MYJ column per surface point, preserving CUDA field order.

    ``columns`` carries the bottom-up ``(nz, ny, nx)`` ``dz`` and ``tke``
    fields. ``surface``, ``state``, and ``outputs`` carry 2-D float32 planes.
    The caller remains responsible for the WRF driver epilogue after this
    kernel, including publishing ``ustm`` from the updated ``ust``.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal MYJSFC supports ordinary float32 arithmetic only; "
            "WRF exact arithmetic and its pinned transcendental layer are "
            "not implemented")
    from .array import MetalArray

    if set(columns) != set(_COLUMN_INPUTS):
        raise ValueError("columns must contain exactly dz and tke")
    if set(surface) != set(_SURFACE_INPUTS):
        raise ValueError("surface fields do not match the MYJSFC input set")
    if set(state) != set(_INOUT):
        raise ValueError("state fields do not match the MYJSFC INOUT set")
    if set(outputs) != set(_OUTPUTS):
        raise ValueError("output fields do not match the MYJSFC output set")

    psfc = surface["psfc"]
    if not isinstance(psfc, MetalArray) or psfc.ndim != 2:
        raise TypeError("MYJSFC surface fields must be 2-D MetalArray values")
    ny, nx = psfc.shape
    dz = columns["dz"]
    if not isinstance(dz, MetalArray) or dz.ndim != 3:
        raise TypeError("MYJSFC dz must be a 3-D MetalArray column")
    nz = dz.shape[0]
    if nz < 2:
        raise ValueError("the MYJ surface layer requires nz >= 2")
    n = ny * nx
    if not ny or not nx or max(n, nz, ny, nx) > 2_147_483_647:
        raise ValueError("MYJSFC dimensions must be positive int32 values")
    for name in _COLUMN_INPUTS:
        array = columns[name]
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or array.shape != (nz, ny, nx)
                or not array.flags.c_contiguous):
            raise TypeError(
                f"{name} must be contiguous float32 with shape {(nz, ny, nx)}")
    surface_arrays = {name: surface[name] for name in _SURFACE_INPUTS}
    state_arrays = {name: state[name] for name in _INOUT}
    output_arrays = {name: outputs[name] for name in _OUTPUTS}
    for group_name, group in (("surface", surface_arrays),
                              ("state", state_arrays),
                              ("outputs", output_arrays)):
        for name, array in group.items():
            if (not isinstance(array, MetalArray)
                    or array.dtype != np.dtype(np.float32)
                    or array.shape != (ny, nx)
                    or not array.flags.c_contiguous):
                raise TypeError(
                    f"{group_name}.{name} must be contiguous float32 with "
                    f"shape {(ny, nx)}")

    context = psfc.buffer.context
    named_arrays = {
        **columns, **surface_arrays, **state_arrays, **output_arrays,
    }
    for name, array in named_arrays.items():
        if array.buffer.context is not context:
            raise ValueError(f"{name} belongs to a different Metal context")
    table_arrays, scalars = _device_tables(context)
    named_arrays.update(table_arrays)
    arrays = named_arrays
    parameters = {
        **scalars,
        "itimestep": int(itimestep), "nz": nz, "n": n,
        "ny": ny, "nx": nx, "npsi": KZTM,
    }
    return context.submit_kernel(
        "myjsfc_column", arrays, parameters, exact_policy=False,
        grid_size=n, threads_per_group=128)


def submit_myj_driver_wspd(u, v, wspd):
    """Apply the surface-driver MYJ wind diagnostic to 2-D level-zero views."""
    from .array import MetalArray

    if (not isinstance(u, MetalArray) or not isinstance(v, MetalArray)
            or not isinstance(wspd, MetalArray)
            or u.dtype != np.dtype(np.float32)
            or v.dtype != np.dtype(np.float32)
            or wspd.dtype != np.dtype(np.float32)
            or u.ndim != 2 or v.shape != u.shape or wspd.shape != u.shape
            or not u.flags.c_contiguous or not v.flags.c_contiguous
            or not wspd.flags.c_contiguous
            or u.buffer.context is not v.buffer.context
            or u.buffer.context is not wspd.buffer.context):
        raise TypeError("MYJ WSPD inputs must be same-context contiguous FP32 planes")
    ny, nx = u.shape
    n = ny * nx
    if not n or n > 2_147_483_647:
        raise ValueError("MYJ WSPD dimensions must fit positive int32 indexing")
    u_flat, v_flat, wspd_flat = (array.view((n,)) for array in (u, v, wspd))
    u.buffer.context.submit_kernel(
        "myj_driver_wspd", {"u": u_flat, "v": v_flat, "wspd": wspd_flat},
        {"n": n}, exact_policy=False, grid_size=n, threads_per_group=128)


__all__ = ["submit_myjsfc_column"]
