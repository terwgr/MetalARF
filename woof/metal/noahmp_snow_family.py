"""Typed launchers for the independently portable Noah-MP snow leaves."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray


_THREADS = 64
_FLOAT = np.dtype(np.float32)
_INT = np.dtype(np.int32)


def _submit(name, state_in, state_out, n, *, ints_in=None, ints_out=None):
    if not isinstance(n, (int, np.integer)):
        raise TypeError("Noah-MP snow n must be an integer")
    n = int(n)
    if n > np.iinfo(np.int32).max:
        raise ValueError("Noah-MP snow n must fit the CUDA int32 ABI")
    if n <= 0:
        return None
    arrays = {"in": state_in, "out": state_out}
    if ints_in is not None:
        arrays["iin"] = ints_in
        arrays["iout"] = ints_out
    for name0, array in arrays.items():
        if not isinstance(array, MetalArray):
            raise TypeError(f"Noah-MP snow {name0} must be a MetalArray")
        if not array.flags.c_contiguous:
            raise TypeError(f"Noah-MP snow {name0} must be contiguous")
    if state_in.dtype != _FLOAT or state_in.shape != (n, 53):
        raise TypeError("Noah-MP snow input must be float32 with shape (n, 53)")
    if state_out.dtype != _FLOAT or state_out.shape != (n, 41):
        raise TypeError("Noah-MP snow output must be float32 with shape (n, 41)")
    if ints_in is not None:
        if ints_in.dtype != _INT or ints_in.shape != (n, 4):
            raise TypeError("Noah-MP snow iin must be int32 with shape (n, 4)")
        if not isinstance(ints_out, MetalArray) or ints_out.dtype != _INT or ints_out.shape != (n,):
            raise TypeError("Noah-MP snow iout must be int32 with shape (n,)")
    context = state_in.buffer.context
    if any(array.buffer.context is not context for array in arrays.values()):
        raise ValueError("Noah-MP snow arrays must share one Metal context")
    return context.submit_kernel(
        name, arrays, {"n": n}, exact_policy=False,
        grid_size=((n + _THREADS - 1) // _THREADS) * _THREADS,
        threads_per_group=_THREADS)


def submit_combo(state_in, state_out, n):
    """Combine two scalar snow elements from the packed CUDA-compatible ABI."""
    return _submit("noahmp_snow_combo", state_in, state_out, n)


def submit_snowfall(state_in, iin, state_out, iout, n):
    """Run SNOWFALL; scalar inputs occupy packed input slots 37 through 40."""
    return _submit("noahmp_snow_snowfall", state_in, state_out, n,
                   ints_in=iin, ints_out=iout)


def submit_combine(state_in, iin, state_out, iout, n):
    """Run COMBINE, preserving incoming ponding values on untouched paths."""
    return _submit("noahmp_snow_combine", state_in, state_out, n,
                   ints_in=iin, ints_out=iout)


def submit_divide(state_in, iin, state_out, iout, n):
    """Run DIVIDE over the fixed three snow layers."""
    return _submit("noahmp_snow_divide", state_in, state_out, n,
                   ints_in=iin, ints_out=iout)


def submit_snowh2o(state_in, iin, state_out, iout, n):
    """Run SNOWH2O; scalar inputs occupy packed input slots 37 through 44."""
    return _submit("noahmp_snow_snowh2o", state_in, state_out, n,
                   ints_in=iin, ints_out=iout)


def submit_compact(*args, **kwargs):
    """Refuse COMPACT until faithful binary64 glibc expf is available in MSL."""
    raise NotImplementedError(
        "Noah-MP COMPACT requires the CUDA source's binary64 glibc expf "
        "transcription; the Metal family does not narrow that arithmetic")


def submit_snowwater(*args, **kwargs):
    """Refuse SNOWWATER because it composes the unsupported COMPACT leaf."""
    raise NotImplementedError(
        "Noah-MP SNOWWATER calls COMPACT, which requires binary64 glibc expf")


__all__ = ["submit_combo", "submit_snowfall", "submit_combine",
           "submit_divide", "submit_snowh2o", "submit_compact",
           "submit_snowwater"]
