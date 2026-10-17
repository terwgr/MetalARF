"""Metal helpers for RUC's fractional-seaice SFCLAY wrapper."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


_SEA_OUTPUTS = (
    "znt_sea", "ust_sea", "mol_sea", "zol_sea", "flhc_sea",
    "flqc_sea", "cpm_sea", "cqs2_sea", "chs2_sea", "chs_sea",
    "qsfc_sea", "qgh_sea", "hfx_sea", "qfx_sea", "lh_sea",
)


def _arrays(values, *, context, shape, label):
    for name, value in values.items():
        if (not isinstance(value, MetalArray)
                or value.dtype != np.dtype(np.float32)
                or value.shape != shape
                or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise TypeError(
                f"RUC Metal {label} {name} must be contiguous FP32 {shape} "
                "on the caller's Metal context")


def _threshold(value):
    value = np.float32(value)
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("RUC Metal xice_threshold must lie in (0, 1]")
    return value


def submit_prepare(*, xice, tsk, tsk_save, tsk_sea, local_tsk,
                   itimestep: int, xice_threshold=0.5) -> None:
    """Select RUC's ice temperature without changing the held SST."""
    threshold = _threshold(xice_threshold)
    if type(itimestep) is not int or itimestep < 1:
        raise ValueError("RUC Metal surface timestep is one-based")
    values = {"xice": xice, "tsk": tsk, "tsk_save": tsk_save,
              "tsk_sea": tsk_sea, "local_tsk": local_tsk}
    if not isinstance(xice, MetalArray) or xice.ndim != 2:
        raise TypeError("RUC Metal xice must be a 2-D MetalArray")
    _arrays(values, context=xice.buffer.context, shape=xice.shape,
            label="seaice preparation field")
    n = int(np.prod(xice.shape))
    if n == 0:
        return
    xice.buffer.context.submit_kernel(
        "ruc_sfclay_prepare", values,
        {"n": n, "itimestep": itimestep, "xice_threshold": threshold}, exact_policy=False,
        grid_size=n)


def submit_scatter(*, raw, xice, outputs, xice_threshold=0.5) -> None:
    """Retain second-call open-water outputs in fractional-ice columns."""
    threshold = _threshold(xice_threshold)
    if not isinstance(raw, MetalArray) or raw.ndim != 1:
        raise TypeError("RUC Metal SFCLAY workspace must be flat")
    if not isinstance(xice, MetalArray) or xice.ndim != 2:
        raise TypeError("RUC Metal xice must be a 2-D MetalArray")
    if not isinstance(outputs, dict) or set(outputs) != set(_SEA_OUTPUTS):
        raise TypeError("RUC Metal open-water SFCLAY output map is incomplete")
    context, shape = xice.buffer.context, xice.shape
    _arrays({"xice": xice, **outputs}, context=context, shape=shape,
            label="open-water SFCLAY field")
    n = int(np.prod(shape))
    if raw.dtype != np.dtype(np.float32) or not raw.flags.c_contiguous:
        raise TypeError("RUC Metal SFCLAY workspace must be contiguous FP32")
    if raw.buffer.context is not context or raw.size < 34 * n:
        raise ValueError("RUC Metal SFCLAY workspace is undersized or misplaced")
    if n == 0:
        return
    context.submit_kernel(
        "ruc_sfclay_scatter", {"raw": raw, "xice": xice, **outputs},
        {"n": n, "xice_threshold": threshold}, exact_policy=False, grid_size=n)


def submit_mynn_prepare(*, xice, tsk, mavail, xland, tsk_sea, znt_sea,
                        local_tsk, sea_tsk, sea_mavail, sea_xland,
                        itimestep: int, xice_threshold=0.5) -> None:
    """MYNN_SEAICE_WRAPPER's ice TSK and open-water call inputs."""
    threshold = _threshold(xice_threshold)
    if type(itimestep) is not int or itimestep < 1:
        raise ValueError("MYNN Metal surface timestep is one-based")
    values = {"xice": xice, "tsk": tsk, "mavail": mavail, "xland": xland,
              "tsk_sea": tsk_sea, "znt_sea": znt_sea, "local_tsk": local_tsk,
              "sea_tsk": sea_tsk, "sea_mavail": sea_mavail,
              "sea_xland": sea_xland}
    if not isinstance(xice, MetalArray) or xice.ndim != 2:
        raise TypeError("MYNN Metal xice must be a 2-D MetalArray")
    _arrays(values, context=xice.buffer.context, shape=xice.shape,
            label="seaice preparation field")
    n = int(np.prod(xice.shape))
    if n:
        xice.buffer.context.submit_kernel(
            "mynn_seaice_prepare", values,
            {"n": n, "itimestep": itimestep, "xice_threshold": threshold}, exact_policy=False,
            grid_size=n)


def submit_mynn_blend(*, xice, value, sea, select: bool = False,
                      xice_threshold=0.5) -> None:
    """Blend (or, for REGIME, select) the open-water call on ice cells."""
    threshold = _threshold(xice_threshold)
    if not isinstance(xice, MetalArray) or xice.ndim != 2:
        raise TypeError("MYNN Metal xice must be a 2-D MetalArray")
    _arrays({"xice": xice, "value": value, "sea": sea},
            context=xice.buffer.context, shape=xice.shape,
            label="seaice blend field")
    n = int(np.prod(xice.shape))
    if n:
        xice.buffer.context.submit_kernel(
            "mynn_seaice_blend", {"xice": xice, "value": value, "sea": sea},
            {"n": n, "select": int(bool(select)), "xice_threshold": threshold}, exact_policy=False,
            grid_size=n)


__all__ = ["submit_prepare", "submit_scatter", "submit_mynn_prepare",
           "submit_mynn_blend"]
