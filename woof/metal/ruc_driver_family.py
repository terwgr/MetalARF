"""Typed launchers for the packed RUC driver prologue, epilogue and commit."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from .ruc_metal_layout import (
    ALIVE_BASE_UNITS,
    INTEGER_BASE_UNITS,
    RUN_BASE_UNITS,
)


_THREADS = 128
_I32_MAX = np.iinfo(np.int32).max
_PROLOGUE = {
    "n", "int_base", "run_base", "pending_base", "frzfrac_mode",
    "ktau", "dt", "iswater", "isice", "nv", "ns",
    "icealbedo", "cn",
    "table_base", "table_offset_ifortbl", "table_offset_z0tbl",
    "table_offset_lemitbl", "table_offset_pctbl", "table_offset_laitbl",
    "table_offset_bb", "table_offset_drysmc", "table_offset_hc",
    "table_offset_maxsmc", "table_offset_refsmc", "table_offset_satpsi",
    "table_offset_satdk", "table_offset_wltsmc", "table_offset_qtz",
    "table_offset_tbq",
    "nlcat", "nscat", "mosaic_lu", "mosaic_soil", "lakemodel", "qvg_air",
    "rdlai2d", "xice_threshold", "forms", "table_offset_landusef",
    "table_offset_soilctop",
}
_EPILOGUE = {
    "n", "dt", "table_base", "table_offset_tbq", "table_offset_lemitbl",
    "table_offset_zshalf", "table_offset_landusef", "nlcat", "mosaic_lu",
    "crop", "natural", "irrigation", "log_profile", "xice_threshold",
}


def _packed_resources(raw, flags, n: int):
    if type(n) is not int or not 0 <= n <= _I32_MAX - (_THREADS - 1):
        raise ValueError("RUC Metal dispatch extent must fit int32")
    for name, array, dtype in (("raw", raw, np.dtype(np.uint32)),
                               ("flags", flags, np.dtype(np.uint64))):
        if not isinstance(array, MetalArray):
            raise TypeError(f"RUC Metal {name} must be a MetalArray")
        if array.ndim != 1 or not array.flags.c_contiguous or array.dtype != dtype:
            raise TypeError(
                f"RUC Metal {name} requires contiguous flat {dtype.name} storage")
    if raw.buffer.context is not flags.buffer.context:
        raise ValueError("RUC Metal packed resources must share one context")
    if raw.size < ALIVE_BASE_UNITS * max(n, 1) + n or flags.size < 20:
        raise ValueError("RUC Metal packed workspace is undersized")
    return raw.buffer.context


def _submit(name, raw, flags, parameters, expected):
    if not isinstance(parameters, dict) or set(parameters) != expected:
        missing = sorted(expected - set(parameters or {}))
        extra = sorted(set(parameters or {}) - expected)
        raise TypeError(f"RUC Metal {name} parameters differ; missing={missing}, extra={extra}")
    n = parameters["n"]
    context = _packed_resources(raw, flags, n)
    if n == 0:
        return None
    return context.submit_kernel(
        name, {"raw": raw, "flags": flags}, parameters, exact_policy=True,
        grid_size=((n + _THREADS - 1) // _THREADS) * _THREADS,
        threads_per_group=_THREADS)


def submit_prologue(*, raw, flags, parameters):
    """Dispatch the WRF driver/SOILVEGIN prologue on the packed RUC slab."""
    return _submit("ruc_driver_prologue", raw, flags, parameters, _PROLOGUE)


def submit_epilogue(*, raw, flags, parameters):
    """Dispatch the WRF post-SFCTMP driver epilogue on the packed slab."""
    return _submit("ruc_driver_epilogue", raw, flags, parameters, _EPILOGUE)


def submit_commit(*, raw, flags, n: int, alive_base: int):
    """Dispatch the final all-or-nothing RUC output admission stage."""
    context = _packed_resources(raw, flags, n)
    if flags.size < 35:
        raise ValueError("RUC Metal shared flag slab is undersized")
    if n == 0:
        return None
    return context.submit_kernel(
        "ruc_driver_commit",
        {"raw": raw, "flags": flags},
        {"n": n, "sf_flag_words": 15, "sf_flag_base": 20,
         "alive_base": alive_base}, exact_policy=True,
        grid_size=((n + _THREADS - 1) // _THREADS) * _THREADS,
        threads_per_group=_THREADS)


def dispatch_full_chain(*, workspace, inputs, prologue_parameters,
                        sfctmp_parameters, epilogue_parameters,
                        pending_rainbl=None, pattern_spp_lsm=None,
                        field_sf=None):
    """Queue the six production RUC stages against one packed workspace.

    The caller owns categorical/finite validation and table uploads. Public
    fields are first blitted into `workspace`; output blits remain a separate
    explicit step after the caller has inspected the shared failure flags.
    """
    n = workspace.n
    if n == 0:
        return ()
    workspace.flag_slab.fill(0)
    workspace.pack_fields(inputs)
    if pending_rainbl is None:
        workspace.pending_rainbl.fill(0.0)
    else:
        workspace.transfer(((pending_rainbl, workspace.pending_rainbl),))
    driver_flags = workspace.driver_flags
    sf_flags = workspace.sf_flags
    raw = workspace.raw
    prologue = dict(prologue_parameters)
    prologue.update({
        "n": n,
        "int_base": INTEGER_BASE_UNITS * n,
        "run_base": RUN_BASE_UNITS * n,
        "pending_base": workspace.pending_base,
        "table_base": workspace.table_base,
    })
    sfctmp = dict(sfctmp_parameters)
    sfctmp.update({
        "n": n,
        "int_base": INTEGER_BASE_UNITS * n,
        "run_base": RUN_BASE_UNITS * n,
        "alive_base": ALIVE_BASE_UNITS * n,
        "table_base": workspace.table_base,
    })
    epilogue = dict(epilogue_parameters)
    epilogue.update({"n": n, "table_base": workspace.table_base})
    submissions = [submit_prologue(
        raw=raw, flags=driver_flags, parameters=prologue)]
    from .ruc_sfctmp_family import submit_stage
    for stage in (0, 1, 2):
        submissions.append(submit_stage(
            stage, raw=raw, flags=sf_flags, parameters=sfctmp,
            pattern_spp_lsm=pattern_spp_lsm, field_sf=field_sf))
    submissions.append(submit_epilogue(
        raw=raw, flags=driver_flags, parameters=epilogue))
    submissions.append(submit_commit(
        raw=raw, flags=workspace.flag_slab, n=n,
        alive_base=ALIVE_BASE_UNITS * n))
    return tuple(submissions)


def scatter_committed(*, workspace, outputs, admitted: bool):
    """Blit result rows to public state only after flag admission succeeds."""
    if type(admitted) is not bool:
        raise TypeError("RUC Metal output admission must be a bool")
    if not admitted:
        raise ValueError("RUC Metal output scatter requires a clean commit")
    workspace.unpack_fields(outputs)


__all__ = [
    "dispatch_full_chain", "scatter_committed", "submit_commit",
    "submit_epilogue", "submit_prologue",
]
