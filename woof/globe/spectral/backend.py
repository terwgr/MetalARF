"""Small NumPy/CuPy backend boundary.

The transform arithmetic is written once.  NumPy is the CPU reference and is
always available; CuPy is optional and imported only when explicitly selected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Backend:
    name: str
    xp: Any
    float_dtype: Any
    complex_dtype: Any

    def asarray(self, value, *, dtype=None):
        return self.xp.asarray(value, dtype=dtype)

    def to_numpy(self, value) -> np.ndarray:
        # An array the pinned host tier holds is already on the host: it
        # answers ``np.asarray`` with its own slot (after waiting for the
        # slot's outstanding write) and never touches the card, which is
        # what makes a checkpoint, an export and a diagnostic read of a
        # spilled run cost nothing on the device side.
        if self.name == "numpy":
            explicit_copy = getattr(value, "to_numpy", None)
            if callable(explicit_copy):
                return np.asarray(explicit_copy())
            return np.asarray(value)
        if hasattr(value, "host"):
            return np.asarray(value)
        return self.xp.asnumpy(value)

    def synchronize(self) -> None:
        if self.name == "cupy":
            self.xp.cuda.runtime.deviceSynchronize()


def get_backend(name: str = "numpy", precision: str = "float64") -> Backend:
    key = str(name).strip().lower()
    prec = str(precision).strip().lower()
    if prec not in {"float32", "float64"}:
        raise ValueError(f"precision must be 'float32' or 'float64', got {precision!r}")
    if key == "numpy":
        xp = np
    elif key == "cupy":
        from woof.local_gpu import NO_LOCAL_GPU_ENV, no_local_gpu

        if no_local_gpu():
            # The desktop card belongs to whichever lane is running device
            # proofs on it; a global-model run that opened it unannounced
            # (2026-09-01, noticed by the release lane during its device set)
            # would land those proofs on a contended card.
            raise RuntimeError(
                f"backend='cupy' refused: {NO_LOCAL_GPU_ENV} is set, so this "
                "process may not open the local CUDA device. Select "
                "backend='numpy' (the CPU reference) or run on a node where "
                "the variable is unset."
            )
        try:
            import cupy as xp  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "backend='cupy' requires CuPy; install the WOOF GPU runtime "
                "or select backend='numpy' for the CPU reference"
            ) from exc
    else:
        raise ValueError(f"backend must be 'numpy' or 'cupy', got {name!r}")
    f = xp.float32 if prec == "float32" else xp.float64
    c = xp.complex64 if prec == "float32" else xp.complex128
    return Backend(key, xp, f, c)


def device_cache_key(xp=None) -> tuple:
    """``(module name, device id)``: the key every module-level cache of a
    device object must carry.

    A cache that holds a CuPy array, a compiled kernel or an operator built
    on one device and hands it back on another is a defect with no error
    message on a consumer card: CuPy refuses a cross-device array outright
    (``ValueError: The device where the array resides (0) is different from
    the current device (1). Peer access is unavailable``) only when it
    notices, and a kernel launched against the wrong device's module
    silently reads the wrong memory.  ``woof/globe/core/rrtmgp.py`` carries the
    same helper for the radiation tables, written when a four-card box hit
    exactly this on the second card's first radiation step; this one is for
    the global model, whose caches were NOT among those five.

    Under NumPy there is no device, so the key is one element and every
    CPU-backed cache keeps exactly the behaviour it had.
    """
    if xp is None:
        name = "cupy"
    else:
        name = getattr(xp, "__name__", None) or str(xp)
    if name == "cupy":
        import cupy as cp  # local: the CPU reference must not import it

        return (name, int(cp.cuda.Device().id))
    return (name,)
