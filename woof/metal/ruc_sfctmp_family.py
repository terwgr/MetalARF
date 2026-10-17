"""Typed launchers for the three packed RUC SFCTMP Metal stages."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


_THREADS = 128
_STAGES = {0, 1, 2}
_I32_MAX = np.iinfo(np.int32).max
_PARAMETERS = (
    "n", "run_base", "alive_base", "int_base", "table_base",
    "table_offset_rstbl", "table_offset_rgltbl", "table_offset_z0tbl",
    "table_offset_lemitbl", "table_offset_tbq", "table_offset_zshalf",
    "delt", "c1sn", "c2sn", "rsmax", "isice", "urban", "ncategory",
    "limit", "forms",
)


def _overlaps(left: MetalArray, right: MetalArray) -> bool:
    return (left.buffer is right.buffer
            and left.byte_offset < right.byte_offset + right.nbytes
            and right.byte_offset < left.byte_offset + left.nbytes)


def submit_stage(stage: int, *, raw, flags, parameters: dict[str, object],
                 pattern_spp_lsm=None, field_sf=None):
    """Submit one source-ordered stage over the packed nine-level RUC slab.

    The 815 CUDA pointer aliases have already been resolved to offsets in
    ``raw`` by :mod:`woof.metal.ruc_metal_layout`. Flag words remain separate
    so Metal's native atomic storage rules are explicit.
    """
    if type(stage) is not int or stage not in _STAGES:
        raise ValueError("RUC SFCTMP Metal stage must be 0, 1, or 2")
    parameters = {"spp_lsm": 0, "has_field_sf": 0, **parameters}
    stage_parameters = _PARAMETERS + ("spp_lsm", "has_field_sf")
    missing = [name for name in stage_parameters if name not in parameters]
    if missing:
        raise TypeError(f"missing RUC SFCTMP parameters: {', '.join(missing)}")
    n = parameters["n"]
    if type(n) is not int or not 0 <= n <= _I32_MAX - (_THREADS - 1):
        raise ValueError("RUC SFCTMP Metal n must fit the int32 dispatch extent")
    for name, array, dtype in (("raw", raw, np.dtype(np.uint32)),
                               ("flags", flags, np.dtype(np.uint64))):
        if not isinstance(array, MetalArray):
            raise TypeError(f"RUC SFCTMP Metal {name} must be a MetalArray")
        if array.ndim != 1 or not array.flags.c_contiguous or array.dtype != dtype:
            raise TypeError(
                f"RUC SFCTMP Metal {name} requires contiguous flat {dtype.name} storage")
    if raw.buffer.context is not flags.buffer.context:
        raise ValueError("RUC SFCTMP Metal resources must share one context")
    if n == 0:
        return None
    if raw.size < int(parameters["alive_base"]) + n or flags.size < 15:
        raise ValueError("RUC SFCTMP Metal workspace is smaller than the dispatch")
    for name in stage_parameters:
        if name == "n":
            continue
        value = parameters[name]
        limit = np.finfo(np.float32).max if name in {
            "delt", "c1sn", "c2sn", "rsmax"
        } else _I32_MAX
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"RUC SFCTMP Metal parameter {name} must be numeric")
        if abs(float(value)) > limit:
            raise ValueError(f"RUC SFCTMP Metal parameter {name} is out of range")
    launch_parameters = {name: parameters[name] for name in stage_parameters}
    resources = {"raw": raw, "flags": flags}
    enabled = parameters["spp_lsm"] == 1
    has_field_sf = parameters["has_field_sf"] == 1
    dummy = raw.view((raw.size,), np.float32)
    pattern = pattern_spp_lsm if enabled else dummy
    output = field_sf if enabled and has_field_sf else dummy
    for name, array in (("pattern_spp_lsm", pattern), ("field_sf", output)):
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or array.ndim != 1 or not array.flags.c_contiguous):
            raise TypeError(
                f"RUC SFCTMP Metal {name} must be a contiguous flat float32 MetalArray")
        if array.buffer.context is not raw.buffer.context:
            raise ValueError("RUC SFCTMP SPP resources must share one context")
        if enabled and array.size < 9 * n:
            raise ValueError(f"RUC SFCTMP Metal {name} must contain 9*n values")
        if (enabled and (name == "pattern_spp_lsm"
                         or has_field_sf and name == "field_sf")
                and _overlaps(raw, array)):
            raise ValueError(f"RUC SFCTMP Metal {name} must not alias the packed workspace")
    if enabled and _overlaps(pattern, output):
        raise ValueError("RUC SFCTMP Metal field_sf must not alias the SPP pattern")
    if has_field_sf and field_sf is None:
        raise ValueError("RUC SFCTMP Metal field_sf flag has no output resource")
    resources.update({"pattern_spp_lsm": pattern, "field_sf": output})
    return raw.buffer.context.submit_kernel(
        f"ruc_sfctmp_stage{stage}", resources,
        launch_parameters, exact_policy=True,
        grid_size=((n + _THREADS - 1) // _THREADS) * _THREADS,
        threads_per_group=_THREADS)


def submit_stage0(*, raw, flags, parameters):
    return submit_stage(0, raw=raw, flags=flags, parameters=parameters)


def submit_stage1(*, raw, flags, parameters):
    return submit_stage(1, raw=raw, flags=flags, parameters=parameters)


def submit_stage2(*, raw, flags, parameters):
    return submit_stage(2, raw=raw, flags=flags, parameters=parameters)


__all__ = ["submit_stage", "submit_stage0", "submit_stage1", "submit_stage2"]
