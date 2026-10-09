"""Typed Metal launchers for Noah-MP ERROR and SFLX marshalling kernels."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray


_THREADS_ERROR = 128
_THREADS_MARSHAL = 64
_NSOIL = 4
_NSNOW = 3
_NFULL = _NSOIL + _NSNOW
_NSC = 34
_NOUT = 9
_I32_MAX = np.iinfo(np.int32).max


def _validate(arrays, shapes, int_names, *, context=None):
    shared_context = context
    for name, array in arrays.items():
        if not isinstance(array, MetalArray):
            raise TypeError(f"Noah-MP SFLX {name} must be a MetalArray")
        if array.shape != shapes[name]:
            raise ValueError(
                f"Noah-MP SFLX {name} shape {array.shape} != {shapes[name]}")
        dtype = np.dtype(np.int32 if name in int_names else np.float32)
        if array.dtype != dtype or not array.flags.c_contiguous:
            raise TypeError(
                f"Noah-MP SFLX {name} requires contiguous {dtype.name} storage")
        if shared_context is None:
            shared_context = array.buffer.context
        elif array.buffer.context is not shared_context:
            raise ValueError("Noah-MP SFLX resources must share one context")
    return shared_context


def submit_k_sflx_error(*, sc, smc, dzsnso, ist, calc, out, status):
    """Submit the bitwise-ordered ERROR gates for a batch of columns.

    ``sc`` is the source's 34-scalar row; ``smc`` and ``dzsnso`` contain the
    four soil values read by the water balance. Outputs are caller-owned.
    """
    if not isinstance(sc, MetalArray) or sc.ndim != 2 or sc.shape[1] != _NSC:
        raise TypeError(f"Noah-MP SFLX sc must have shape (ncase, {_NSC})")
    ncase = int(sc.shape[0])
    if ncase > _I32_MAX:
        raise ValueError("Noah-MP SFLX ncase exceeds the int32 ABI")
    arrays = {"sc": sc, "smc": smc, "dzsnso": dzsnso, "ist": ist,
              "calc": calc, "out": out, "status": status}
    shapes = {"sc": (ncase, _NSC), "smc": (ncase, _NSOIL),
              "dzsnso": (ncase, _NSOIL), "ist": (ncase,),
              "calc": (ncase,), "out": (ncase, _NOUT),
              "status": (ncase,)}
    context = _validate(arrays, shapes, {"ist", "calc", "status"})
    if ncase == 0:
        return None
    grid = ((ncase + _THREADS_ERROR - 1) // _THREADS_ERROR) * _THREADS_ERROR
    return context.submit_kernel(
        "k_sflx_error", arrays, {"ncase": ncase}, exact_policy=False,
        grid_size=grid, threads_per_group=_THREADS_ERROR)


def submit_k_sflx_marshal(*, zsnso, stc, smc, zsoil, canliq, canice, sneqv,
                          wa, isnow, nroot, ist, dzsnso, troot, beg_wb):
    """Submit DZSNSO, TROOT, and BEG_WB for a batch of columns."""
    if not isinstance(zsnso, MetalArray) or zsnso.ndim != 2:
        raise TypeError("Noah-MP SFLX zsnso must be a two-dimensional MetalArray")
    ncol = int(zsnso.shape[0])
    if ncol > _I32_MAX:
        raise ValueError("Noah-MP SFLX ncol exceeds the int32 ABI")
    arrays = {
        "zsnso": zsnso, "stc": stc, "smc": smc, "zsoil": zsoil,
        "canliq": canliq, "canice": canice, "sneqv": sneqv, "wa": wa,
        "isnow": isnow, "nroot": nroot, "ist": ist, "dzsnso": dzsnso,
        "troot": troot, "beg_wb": beg_wb,
    }
    shapes = {
        "zsnso": (ncol, _NFULL), "stc": (ncol, _NFULL),
        "smc": (ncol, _NSOIL), "zsoil": (ncol, _NSOIL),
        "canliq": (ncol,), "canice": (ncol,), "sneqv": (ncol,),
        "wa": (ncol,), "isnow": (ncol,), "nroot": (ncol,),
        "ist": (ncol,), "dzsnso": (ncol, _NFULL),
        "troot": (ncol,), "beg_wb": (ncol,),
    }
    context = _validate(arrays, shapes, {"isnow", "nroot", "ist"})
    if ncol == 0:
        return None
    grid = ((ncol + _THREADS_MARSHAL - 1) // _THREADS_MARSHAL) * _THREADS_MARSHAL
    return context.submit_kernel(
        "k_sflx_marshal", arrays, {"ncol": ncol}, exact_policy=False,
        grid_size=grid, threads_per_group=_THREADS_MARSHAL)


__all__ = ["submit_k_sflx_error", "submit_k_sflx_marshal"]
