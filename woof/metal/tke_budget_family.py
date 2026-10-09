"""Refusal boundary for the FP64-only Metal TKE budget reduction.

The CUDA receipt accumulates per-cell values in FP64 and writes an FP64
per-level accumulator.  The current WOOF Metal typed-kernel ABI only describes
32-bit resources and parameters, and its eight-byte array values are storage
only.  Do not substitute FP32 accumulation or copy the reduction to the host.
"""


def submit_tke_budget_accumulate(*args, **kwargs):
    """Refuse until Metal has a faithful FP64 reduction path."""
    raise NotImplementedError(
        "Metal tke_budget_accumulate requires FP64 device arithmetic and an "
        "FP64 writable resource; the current WOOF Metal kernel ABI does not "
        "provide them. No FP32 or host fallback is permitted.")


__all__ = ["submit_tke_budget_accumulate"]
