"""Metal launcher coverage for preprocessing's horizontal CUDA family.

Only ``horizontal_envelope`` is currently supported. The other entry points
depend on the pinned binary64/subnormal arithmetic in ``horizontal.cu`` and
fail closed until that arithmetic has a verified Metal implementation.
"""

from __future__ import annotations

import numpy as np


_FP64_REFUSAL = {
    "horizontal_cast": "binary64-to-binary32 rounding with subnormal preservation",
    "horizontal_divide": "binary64 fallback for binary32 subnormal division",
    "horizontal_regular": "binary64 emulation of correctly-rounded binary32 interpolation",
    "horizontal_nearest": "binary64 rounding and distance ordering",
    "horizontal_rotate": "binary64 fallback for correctly-rounded binary32 operations",
    "horizontal_rh_water": "binary64 RN arithmetic and portable log/exp",
}


def require_supported_horizontal(name: str) -> None:
    """Refuse horizontal operators whose source arithmetic is not supported."""
    requirement = _FP64_REFUSAL.get(name)
    if requirement is not None:
        raise NotImplementedError(
            f"Metal {name} is unavailable: it requires {requirement}; "
            "the backend has no verified binary64/subnormal arithmetic layer")
    if name != "horizontal_envelope":
        raise ValueError(f"unknown horizontal kernel {name!r}")


def submit_horizontal_envelope(context, src, bounds):
    """Reduce finite FP32 values to ordered-bit bounds and an invalid flag.

    ``bounds`` is a caller-initialized three-word float32 MetalArray whose
    storage is interpreted as uint32 bit patterns by the shader. Its initial
    words must be ``0xffffffff, 0, 0`` as in the CUDA caller.
    """
    require_supported_horizontal("horizontal_envelope")
    if src.dtype != np.dtype(np.float32) or bounds.dtype != np.dtype(np.float32):
        raise TypeError("horizontal_envelope requires float32 Metal storage")
    count = int(src.size)
    if count > 0x7fffffff:
        raise NotImplementedError(
            "Metal horizontal_envelope currently limits count to int32 range")
    if count == 0:
        return None
    groups = min(1024, (count + 255) // 256)
    return context.submit_kernel(
        "horizontal_envelope",
        {"src": src, "bounds": bounds},
        {"count": count, "n_bounds": 3},
        exact_policy=False, grid_size=groups * 256,
        threads_per_group=256)


def submit_horizontal_cast(*_args, **_kwargs):
    require_supported_horizontal("horizontal_cast")


def submit_horizontal_divide(*_args, **_kwargs):
    require_supported_horizontal("horizontal_divide")


def submit_horizontal_regular(*_args, **_kwargs):
    require_supported_horizontal("horizontal_regular")


def submit_horizontal_nearest(*_args, **_kwargs):
    require_supported_horizontal("horizontal_nearest")


def submit_horizontal_rotate(*_args, **_kwargs):
    require_supported_horizontal("horizontal_rotate")


def submit_horizontal_rh_water(*_args, **_kwargs):
    require_supported_horizontal("horizontal_rh_water")
