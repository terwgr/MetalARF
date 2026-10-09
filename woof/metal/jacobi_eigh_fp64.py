"""Low-level software-binary64 Metal launcher for batched Jacobi eigh."""

from __future__ import annotations

import math

import numpy as np

from woof.core.jacobi_eigh import JacobiEighError, plan


def submit_jacobi_eigh_fp64(a, *, return_sweeps: bool = False):
    """Return ascending FP64 eigenpairs for a contiguous Metal batch.

    Numerical work uses integer-only software binary64 operations in Metal;
    the buffers retain ordinary NumPy ``float64`` storage. This is a low-level
    kernel entrypoint and does not provide LETKF array algebra or dispatch.
    CUDA FMA contraction parity is not established.
    """
    from .array import MetalArray

    if not isinstance(a, MetalArray):
        raise TypeError("Metal Jacobi FP64 requires a MetalArray input")
    if a.dtype != np.dtype(np.float64):
        raise TypeError("Metal Jacobi FP64 requires float64 input")
    if not a.flags.c_contiguous:
        raise TypeError("Metal Jacobi FP64 requires contiguous input matrices")
    if len(a.shape) != 3 or a.shape[1] != a.shape[2]:
        raise JacobiEighError(
            f"jacobi_eigh wants a stack of square matrices (n, k, k), got "
            f"shape {tuple(a.shape)}")

    n, k, _ = (int(extent) for extent in a.shape)
    tier = plan(k, np.dtype(np.float64).str)
    if n > 0x7FFFFFFF:
        raise JacobiEighError(
            "Metal Jacobi batch count exceeds the typed int32 extent ABI")

    context = a.buffer.context

    def allocate(shape, dtype):
        dtype = np.dtype(dtype)
        count = math.prod(shape)
        buffer = context.buffer(max(dtype.itemsize, count * dtype.itemsize))
        return MetalArray(buffer, shape, dtype)

    m = tier.m
    w = allocate((n, k), np.float64)
    v = allocate((n, k, k), np.float64)
    work = allocate((n, m, m), np.float64)
    status = allocate((n,), np.int32)
    if n == 0:
        return (w, v, 0) if return_sweeps else (w, v)

    context.submit_kernel(
        "jacobi_eigh_fp64_batched",
        {"a_in": a, "work": work, "w_out": w, "v_out": v,
         "status_out": status},
        {"n_matrices": n, "k": k, "m": m,
         "tpb": tier.threads_per_matrix, "sweep_cap": tier.sweep_cap},
        # The registered kernel is admitted in both bundle variants. Exact
        # selection pins the arithmetic-policy cache identity for this path.
        exact_policy=True, grid_size=n * tier.threads_per_matrix,
        threads_per_group=tier.threads_per_matrix)

    host_status = status.to_numpy()
    worst = int(host_status.min())
    if worst < 0:
        if worst == -2:
            bad = int(np.count_nonzero(host_status == -2))
            raise JacobiEighError(
                f"jacobi_eigh was handed a non-finite entry in {bad} of {n} "
                "matrices. No eigendecomposition was produced for them; "
                "the caller's finiteness guard is the place to fix this.")
        bad = int(np.count_nonzero(host_status == -1))
        raise JacobiEighError(
            f"jacobi_eigh reached its {tier.sweep_cap}-sweep cap on {bad} of "
            f"{n} matrices of size {k} without convergence. These matrices "
            "are not answers and are not returned.")
    if return_sweeps:
        return w, v, int(host_status.max())
    return w, v
