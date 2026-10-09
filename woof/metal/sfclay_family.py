"""Typed Metal submit helper for WRF MM5 SFCLAY options 91 and 1."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from .runtime import MetalContext


_INPUTS = ("u", "v", "t", "qv", "p", "dz8w", "psfc", "tsk",
           "pblh", "mavail", "xland", "lakemask")
_INOUTS = ("znt", "ust", "ustm", "mol", "hfx", "qfx", "qsfc", "zol")
_OUTPUTS = ("regime", "psim", "psih", "fm", "fh", "lh", "u10", "v10",
            "th2", "t2", "q2", "chs", "chs2", "cqs2", "flhc", "flqc",
            "qgh", "rmol", "wspd", "br", "gz1oz0", "cpm", "ck", "cka",
            "cd", "cda")


def submit_sfclay_column(
    context: MetalContext, u: MetalArray, v: MetalArray, t: MetalArray,
    qv: MetalArray, p: MetalArray, dz8w: MetalArray, psfc: MetalArray,
    tsk: MetalArray, pblh: MetalArray, mavail: MetalArray,
    xland: MetalArray, lakemask: MetalArray, result,
    *, option: int, dx: float, isfflx: bool = True,
    isftcflx: int = 0, iz0tlnd: int = 0,
) -> None:
    """Submit one surface column per 2-D grid point into ``result``.

    ``result`` may be a mapping or an object exposing the 34 canonical
    ``SFCLAY_OUTPUTS`` arrays. Inputs/results must be same-context contiguous
    float32 MetalArrays; the first eight result arrays preserve WRF inout
    state. No allocation, host arithmetic, or fallback is performed here.
    """
    from woof.core.physics_inventory import SFCLAY_OUTPUTS

    if not isinstance(context, MetalContext):
        raise TypeError("Metal sfclay requires a MetalContext")
    if option not in (1, 91):
        raise ValueError(f"sfclay option must be 1 or 91, got {option}")
    if isftcflx not in (0, 1, 2):
        raise ValueError(f"isftcflx must be 0, 1, or 2, got {isftcflx}")
    if iz0tlnd not in (0, 1, 2):
        raise ValueError(f"iz0tlnd must be 0, 1, or 2, got {iz0tlnd}")

    inputs = dict(zip(_INPUTS, (u, v, t, qv, p, dz8w, psfc, tsk,
                               pblh, mavail, xland, lakemask)))
    if isinstance(result, dict):
        outputs = result
    else:
        outputs = {name: getattr(result, name) for name in SFCLAY_OUTPUTS}
    missing = set(SFCLAY_OUTPUTS) - set(outputs)
    if missing:
        raise KeyError(f"sfclay result is missing {sorted(missing)}")

    first = u
    if not isinstance(first, MetalArray) or first.ndim != 2:
        raise TypeError("sfclay u must be a 2-D MetalArray")
    shape = first.shape
    arrays = {}
    for name, value in inputs.items():
        if (not isinstance(value, MetalArray)
                or value.dtype != np.dtype(np.float32)
                or value.shape != shape or not value.flags.c_contiguous):
            raise TypeError(f"sfclay {name} must be contiguous float32 {shape}")
        if value.buffer.context is not context:
            raise ValueError(f"sfclay {name} belongs to a different Metal context")
        arrays[name] = value
    for name in SFCLAY_OUTPUTS:
        value = outputs[name]
        if (not isinstance(value, MetalArray)
                or value.dtype != np.dtype(np.float32)
                or value.shape != shape or not value.flags.c_contiguous):
            raise TypeError(f"sfclay {name} must be contiguous float32 {shape}")
        if value.buffer.context is not context:
            raise ValueError(f"sfclay {name} belongs to a different Metal context")
        arrays[name] = value

    ny, nx = shape
    n = int(np.prod(shape))
    context.submit_kernel(
        "sfclay_column", arrays,
        {"dx": float(dx), "option": int(option),
         "isfflx": int(bool(isfflx)), "isftcflx": int(isftcflx),
         "iz0tlnd": int(iz0tlnd), "nx": nx, "ny": ny, "n": n},
        exact_policy=False, grid_size=n)
