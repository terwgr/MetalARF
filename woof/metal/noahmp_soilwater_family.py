"""Typed launcher for the arithmetic-only Noah-MP SSTEP CUDA export."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray


_FLOAT = np.dtype(np.float32)
_THREADS = 64
_UNSUPPORTED = {
    "k_canwater": "CANWATER calls pinned glibc powf for its frozen wetness exponent",
    "k_infil": "INFIL calls pinned glibc expf and WDFCND2 calls pinned glibc powf",
    "k_srt": "SRT calls WDFCND1, which uses pinned glibc powf",
    "k_soilwater": "SOILWATER composes pinned glibc expf and powf paths",
    "k_expf_sweep": "diagnostic sweep requires the unreviewed pinned glibc expf implementation",
    "k_powf_sweep": "diagnostic sweep requires the unreviewed pinned glibc powf implementation",
}


def submit_sstep(par: MetalArray, fin: MetalArray, fout: MetalArray):
    """Submit k_sstep for rows using the CUDA par/fin/fout packed layouts."""
    arrays = {"par": par, "fin": fin, "fout": fout}
    for name, array in arrays.items():
        if not isinstance(array, MetalArray):
            raise TypeError(f"Noah-MP soilwater {name} must be a MetalArray")
        if array.dtype != _FLOAT:
            raise TypeError(f"Noah-MP soilwater {name} must have dtype float32")
        if not array.flags.c_contiguous:
            raise TypeError(f"Noah-MP soilwater {name} must be contiguous")
    if par.ndim != 2 or par.shape[1] != 24:
        raise ValueError("Noah-MP soilwater par must have shape (n, 24)")
    n = int(par.shape[0])
    if fin.shape != (n, 40):
        raise ValueError("Noah-MP soilwater fin must have shape (n, 40)")
    if fout.shape != (n, 32):
        raise ValueError("Noah-MP soilwater fout must have shape (n, 32)")
    if n > np.iinfo(np.int32).max:
        raise ValueError("Noah-MP soilwater n must fit the CUDA int32 ABI")
    if n == 0:
        return None
    context = par.buffer.context
    if any(array.buffer.context is not context for array in arrays.values()):
        raise ValueError("Noah-MP soilwater arrays must share one Metal context")
    return context.submit_kernel(
        "noahmp_soilwater_sstep", arrays, {"n": n}, exact_policy=False,
        grid_size=((n + _THREADS - 1) // _THREADS) * _THREADS,
        threads_per_group=_THREADS)


def require_noahmp_soilwater_metal_supported(cuda_symbol: str) -> None:
    """Fail closed for soilwater exports that depend on pinned binary64 math."""
    if cuda_symbol == "k_sstep":
        return
    reason = _UNSUPPORTED.get(cuda_symbol)
    if reason is None:
        raise ValueError(f"unknown Noah-MP soilwater CUDA export {cuda_symbol!r}")
    raise NotImplementedError(
        f"Metal {cuda_symbol} has no faithful translation: {reason}; "
        "the shared software-binary64 and pinned-math packets are not "
        "centrally reviewed for production use")


def submit_canwater(*args, **kwargs):
    require_noahmp_soilwater_metal_supported("k_canwater")


def submit_infil(*args, **kwargs):
    require_noahmp_soilwater_metal_supported("k_infil")


def submit_srt(*args, **kwargs):
    require_noahmp_soilwater_metal_supported("k_srt")


def submit_soilwater(*args, **kwargs):
    require_noahmp_soilwater_metal_supported("k_soilwater")


__all__ = ["submit_sstep", "submit_canwater", "submit_infil", "submit_srt",
           "submit_soilwater", "require_noahmp_soilwater_metal_supported"]
