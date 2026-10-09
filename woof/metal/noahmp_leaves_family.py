"""Metal launcher for the independently supportable Noah-MP ESAT leaf."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def submit_leaf_atm(xs: MetalArray, ys: MetalArray) -> MetalArray:
    """Run the packed 11-to-17-slot ATM leaf on Metal.

    Native MSL ``pow``/``exp`` serve ATM's r_pow/r_exp operations under the
    default Metal arithmetic policy.
    """
    if not isinstance(xs, MetalArray) or not isinstance(ys, MetalArray):
        raise TypeError("Noah-MP ATM requires MetalArray inputs and output")
    if (xs.dtype != np.dtype(np.float32) or ys.dtype != np.dtype(np.float32)
            or not xs.flags.c_contiguous or not ys.flags.c_contiguous):
        raise TypeError("Noah-MP ATM requires contiguous float32 arrays")
    if xs.ndim != 2 or xs.shape[1] != 11:
        raise ValueError("Noah-MP ATM input must have shape (ncase, 11)")
    ncase = int(xs.shape[0])
    if ys.shape != (ncase, 17):
        raise ValueError(f"Noah-MP ATM output must have shape {(ncase, 17)}")
    if xs.buffer.context is not ys.buffer.context:
        raise ValueError("Noah-MP ATM arrays must share one Metal context")
    if ncase:
        xs.buffer.context.submit_kernel(
            "noahmp_leaf_atm", {"xs": xs, "ys": ys},
            {"ncase": ncase}, exact_policy=False,
            grid_size=((ncase + 63) // 64) * 64, threads_per_group=64)
    return ys


def submit_leaf_esat(xs: MetalArray, ys: MetalArray) -> MetalArray:
    """Evaluate the four-source FP32 ESAT polynomial for each case.

    ``xs`` is ``(ncase, 1)`` and ``ys`` is a caller-owned
    ``(ncase, 4)`` contiguous float32 Metal array. Other Noah-MP leaves are
    intentionally not redirected to approximate transcendental math.
    """
    if not isinstance(xs, MetalArray) or not isinstance(ys, MetalArray):
        raise TypeError("Noah-MP ESAT requires MetalArray inputs and output")
    if (xs.dtype != np.dtype(np.float32)
            or ys.dtype != np.dtype(np.float32)
            or not xs.flags.c_contiguous or not ys.flags.c_contiguous):
        raise TypeError("Noah-MP ESAT requires contiguous float32 arrays")
    if xs.ndim != 2 or xs.shape[1] != 1:
        raise ValueError("Noah-MP ESAT input must have shape (ncase, 1)")
    ncase = xs.shape[0]
    if ys.shape != (ncase, 4):
        raise ValueError(
            f"Noah-MP ESAT output must have shape {(ncase, 4)}")
    if (xs.buffer is ys.buffer
            and xs.byte_offset < ys.byte_offset + ys.nbytes
            and ys.byte_offset < xs.byte_offset + xs.nbytes):
        raise ValueError("Noah-MP ESAT input/output must not overlap")
    if ncase:
        xs.buffer.context.submit_kernel(
            "noahmp_leaf_esat", {"xs": xs, "ys": ys},
            {"ncase": ncase, "n_in": 1, "n_out": 4},
            exact_policy=False, grid_size=ncase, threads_per_group=64)
    return ys


def submit_leaf_rosr12(xs: MetalArray, ys: MetalArray, ix) -> MetalArray:
    """Run the pure-FP32 ROSR12 solve, passing per-case ``ntop`` as a scalar.

    The int32 topology is host metadata, not a numeric input. Keeping it as a
    scalar parameter avoids the current runtime's unsupported int32 resource
    path while retaining each CUDA case's original topology value.
    """
    if not isinstance(xs, MetalArray) or not isinstance(ys, MetalArray):
        raise TypeError("Noah-MP ROSR12 requires MetalArray inputs and output")
    if (xs.dtype != np.dtype(np.float32)
            or ys.dtype != np.dtype(np.float32)
            or not xs.flags.c_contiguous or not ys.flags.c_contiguous):
        raise TypeError("Noah-MP ROSR12 requires contiguous float32 arrays")
    if xs.ndim != 2 or xs.shape[1] != 28:
        raise ValueError("Noah-MP ROSR12 input must have shape (ncase, 28)")
    ncase = xs.shape[0]
    if ys.shape != (ncase, 21):
        raise ValueError(
            f"Noah-MP ROSR12 output must have shape {(ncase, 21)}")
    host_ix = np.ascontiguousarray(np.asarray(ix, dtype=np.int32))
    if host_ix.shape != (ncase, 1):
        raise ValueError(
            f"Noah-MP ROSR12 topology must have shape {(ncase, 1)} int32")
    if (xs.buffer is ys.buffer
            and xs.byte_offset < ys.byte_offset + ys.nbytes
            and ys.byte_offset < xs.byte_offset + xs.nbytes):
        raise ValueError("Noah-MP ROSR12 input/output must not overlap")
    context = xs.buffer.context
    for case, ntop0 in enumerate(host_ix[:, 0]):
        x_case = xs.view((28,), byte_offset=xs.byte_offset + case * 28 * 4)
        y_case = ys.view((21,), byte_offset=ys.byte_offset + case * 21 * 4)
        context.submit_kernel(
            "noahmp_leaf_rosr12", {"xs": x_case, "ys": y_case},
            {"ntop0": int(ntop0), "n_in": 28, "n_out": 21},
            exact_policy=False, grid_size=1, threads_per_group=1)
    return ys


def submit_leaf(leaf: str, xs, ys, *, ix=None):
    """Dispatch the supported leaf and fail closed for the others."""
    if leaf == "atm":
        if ix is not None:
            raise ValueError("Noah-MP ATM does not consume integer topology")
        return submit_leaf_atm(xs, ys)
    if leaf == "esat":
        if ix is not None:
            raise ValueError("Noah-MP ESAT does not consume integer topology")
        return submit_leaf_esat(xs, ys)
    if leaf == "rosr12":
        if ix is None:
            raise ValueError("Noah-MP ROSR12 requires per-case ntop topology")
        return submit_leaf_rosr12(xs, ys, ix)
    if leaf in {"csnow", "tdfcnd", "thermoprop",
                "wdfcnd1", "wdfcnd2"}:
        raise NotImplementedError(
            f"Noah-MP {leaf} needs the reviewed shared binary64-backed "
            "glibc r_pow/r_exp/r_log10 implementation; no local helper or "
            "CPU fallback is available")
    raise KeyError(f"unknown Noah-MP leaf {leaf!r}")
