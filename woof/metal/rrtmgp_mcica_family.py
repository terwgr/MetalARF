"""Typed Metal launchers for WRF's RRTMGP MCICA mask generator."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray, MetalArrayModule
from woof.wrf_exact import ENABLED as WRF_EXACT


_FLOAT = np.dtype(np.float32)
_U32 = np.dtype(np.uint32)
_THREADS = 128
_PREPARE_RESOURCES = {"play", "seed"}
_MAXRAN_RESOURCES = {
    "cldfra", "seed", "jump_s1", "jump_s2", "jump_s3", "jump_s4",
    "mask_words",
}


def _require_array(value, name, dtype, shape, context=None):
    if (not isinstance(value, MetalArray) or value.dtype != dtype
            or not value.flags.c_contiguous):
        raise TypeError(
            f"RRTMGP MCICA {name} must be a contiguous {dtype} MetalArray")
    if value.shape != shape:
        raise ValueError(
            f"RRTMGP MCICA {name} shape {value.shape} != expected {shape}")
    if context is not None and value.buffer.context is not context:
        raise ValueError("RRTMGP MCICA resources must share one Metal context")
    return value


def _submit_seed_prepare(play, seed, *, ncol, nlay, permuteseed):
    context = play.buffer.context
    return context.submit_kernel(
        "rrtmgp_mcica_prepare_seed",
        {"play": play, "seed": seed},
        {"ncol": int(ncol), "nlay": int(nlay),
         "permuteseed": int(permuteseed)},
        exact_policy=False, grid_size=int(ncol),
        threads_per_group=_THREADS)


def submit_mcica_maxran(play, cldfra, jump_s1, jump_s2, jump_s3, jump_s4,
                        *, ngpt: int, permuteseed: int,
                        mask_words: MetalArray | None = None):
    """Prepare column seeds and submit one invocation per (column, g-point).

    The logical output has shape ``(ncol, nlay, ngpt)`` and byte values 0/1.
    It is stored in a uint32 MetalArray so the existing resource ABI's
    four-byte range/alignment contract is retained. The shader writes through
    a device ``uchar`` view; any final padding bytes are ignored by consumers.
    The returned array is the backing words, not a numerically converted mask.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal RRTMGP MCICA does not implement WRF exact-D arithmetic")
    if (isinstance(ngpt, bool) or not isinstance(ngpt, (int, np.integer))
            or int(ngpt) <= 0):
        raise ValueError("RRTMGP MCICA ngpt must be a positive integer")
    if (isinstance(permuteseed, bool)
            or not isinstance(permuteseed, (int, np.integer))
            or int(permuteseed) < 0):
        raise ValueError("RRTMGP MCICA permuteseed must be nonnegative")
    ngpt, permuteseed = int(ngpt), int(permuteseed)
    if not isinstance(play, MetalArray) or play.ndim != 2:
        raise TypeError("RRTMGP MCICA play must be a two-dimensional MetalArray")
    ncol, nlay = play.shape
    if ncol <= 0 or nlay < 4:
        raise ValueError("RRTMGP MCICA requires ncol > 0 and nlay >= 4")
    context = play.buffer.context
    _require_array(play, "play", _FLOAT, (ncol, nlay), context)
    _require_array(cldfra, "cldfra", _FLOAT, (ncol, nlay), context)
    _require_array(jump_s1, "jump_s1", _U32, (ngpt, 2), context)
    _require_array(jump_s2, "jump_s2", _U32, (32, ngpt), context)
    _require_array(jump_s3, "jump_s3", _U32, (ngpt,), context)
    _require_array(jump_s4, "jump_s4", _U32, (ngpt,), context)
    if ncol > np.iinfo(np.int32).max or nlay > np.iinfo(np.int32).max:
        raise ValueError("RRTMGP MCICA dimensions must fit int32 parameters")
    if ngpt > np.iinfo(np.int32).max or permuteseed > np.iinfo(np.int32).max:
        raise ValueError("RRTMGP MCICA g-points and seed must fit int32")

    nbytes = ncol * nlay * ngpt
    if nbytes > np.iinfo(np.int32).max:
        raise ValueError("RRTMGP MCICA logical mask must fit int32 indexing")
    nwords = (nbytes + 3) // 4
    xp = MetalArrayModule(context)
    if mask_words is None:
        mask_words = xp.zeros((nwords,), dtype=np.uint32)
    else:
        _require_array(mask_words, "mask_words", _U32, (nwords,), context)
        mask_words.fill(np.uint32(0))

    seed = xp.empty((ncol, 4), dtype=np.uint32)
    _submit_seed_prepare(
        play, seed, ncol=ncol, nlay=nlay, permuteseed=permuteseed)
    context.submit_kernel(
        "rrtmgp_mcica_maxran",
        {"cldfra": cldfra, "seed": seed,
         "jump_s1": jump_s1, "jump_s2": jump_s2,
         "jump_s3": jump_s3, "jump_s4": jump_s4,
         "mask_words": mask_words},
        {"ncol": ncol, "nlay": nlay, "ngpt": ngpt,
         "nmask_words": nwords},
        exact_policy=False, grid_size=ncol * ngpt,
        threads_per_group=_THREADS)
    return mask_words


__all__ = ["submit_mcica_maxran"]
