"""Fail-closed admission for the Noah-MP flux-preparation CUDA exports.

All three kernels depend on pinned glibc FP32 transcendental routines whose
implementations use binary64 intermediates, plus source-ordered RN FP32
arithmetic. This packet does not provide a faithful Metal translation.
"""

from __future__ import annotations


_UNSUPPORTED_EXPORTS = {
    "noahmp_fluxprep_ragrb": (
        "r_pow and r_exp use pinned glibc binary64 intermediates; exact "
        "rounding and gradual underflow are not available in the reviewed "
        "Metal arithmetic layer"),
    "noahmp_fluxprep_sfcdif1": (
        "r_log, r_pow, and r_atan plus source-ordered non-FTZ RN FP32 "
        "arithmetic are not available as reviewed Metal implementations"),
    "noahmp_fluxprep_stomata": (
        "r_pow and r_exp use pinned glibc binary64 intermediates; exact "
        "rounding and gradual underflow are not available in the reviewed "
        "Metal arithmetic layer"),
}


def require_noahmp_fluxprep_metal_supported(cuda_symbol: str) -> None:
    """Refuse a flux-preparation export until its CUDA behavior is ported."""
    reason = _UNSUPPORTED_EXPORTS.get(cuda_symbol)
    if reason is None:
        raise ValueError(
            f"unknown Noah-MP flux-preparation CUDA export {cuda_symbol!r}")
    raise NotImplementedError(
        f"Metal {cuda_symbol} has no faithful translation: {reason}; "
        "no kernel signature is admitted")
