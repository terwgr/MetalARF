"""Typed Metal launcher for the batched cyclic Jacobi eigensolver."""

from __future__ import annotations

import math

import numpy as np

from woof.core.jacobi_eigh import JacobiEighError, plan


def submit_jacobi_eigh(a, *, return_sweeps: bool = False):
    """Return ascending eigenvalues and column eigenvectors for Metal FP32.

    The input must already be a contiguous Metal array shaped ``(n,k,k)``.
    FP64 is explicitly refused: the current typed Metal kernel ABI only binds
    contiguous FP32 resources, and this helper never downcasts matrix values.
    """
    from .array import MetalArray

    if not isinstance(a, MetalArray):
        raise TypeError("Metal Jacobi requires a MetalArray input")
    if a.dtype != np.dtype(np.float32):
        if a.dtype == np.dtype(np.float64):
            raise NotImplementedError(
                "Metal Jacobi FP64 is unavailable in the typed kernel ABI; "
                "input matrices are not downcast")
        raise TypeError("Metal Jacobi supports float32 input only")
    if not a.flags.c_contiguous:
        raise TypeError("Metal Jacobi requires contiguous input matrices")
    if len(a.shape) != 3 or a.shape[1] != a.shape[2]:
        raise JacobiEighError(
            f"jacobi_eigh wants a stack of square matrices (n, k, k), got "
            f"shape {tuple(a.shape)}")

    n, k, _ = (int(extent) for extent in a.shape)
    tier = plan(k, np.dtype(np.float32).str)
    if n > 0x7FFFFFFF:
        raise JacobiEighError(
            "Metal Jacobi batch count exceeds the typed int32 extent ABI")

    context = a.buffer.context

    def allocate(shape):
        count = math.prod(shape)
        buffer = context.buffer(max(4, count * np.dtype(np.float32).itemsize))
        return MetalArray(buffer, shape, np.float32)

    w = allocate((n, k))
    v = allocate((n, k, k))
    # CUDA uses an int32 status buffer. The typed Metal ABI currently accepts
    # only float32 resources, so small integer status codes are carried exactly
    # in FP32 and converted only after the status-only readback.
    status = allocate((n,))
    if n == 0:
        return (w, v, 0) if return_sweeps else (w, v)

    context.submit_kernel(
        "jacobi_eigh_batched",
        {"a_in": a, "w_out": w, "v_out": v, "status_out": status},
        {"n_matrices": n, "k": k, "m": tier.m,
         "tpb": tier.threads_per_matrix, "sweep_cap": tier.sweep_cap},
        # The native API treats grid_size as a logical thread count and rounds
        # it up by threads_per_group. One Jacobi matrix owns each group, so
        # dispatch exactly n groups by supplying n * tpb logical threads.
        exact_policy=False, grid_size=n * tier.threads_per_matrix,
        threads_per_group=tier.threads_per_matrix)

    host_status = status.to_numpy().astype(np.int32, copy=False)
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
