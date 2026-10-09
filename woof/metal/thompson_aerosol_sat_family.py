"""Typed launchers for aerosol-aware Thompson saturation kernels."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


_FLOAT = np.dtype(np.float32)
_DOUBLE = np.dtype(np.float64)
_U64 = np.dtype(np.uint64)
_INT = np.dtype(np.int32)
_THREADS = 64
_TNCCN_SIZE = 7 * 9 * 7 * 5 * 4
_TNC_WEV_SIZE = 100 * 37 * 100


def _active_arrays(fields: dict[str, MetalArray | None],
                   shape: tuple[int, ...], context) -> dict[str, MetalArray]:
    """Validate and flatten active float32 resources, preserving offsets."""
    result: dict[str, MetalArray] = {}
    for name, array in fields.items():
        if array is None:
            continue
        if (not isinstance(array, MetalArray) or array.dtype != _FLOAT
                or not array.flags.c_contiguous or array.shape != shape):
            raise TypeError(
                f"Thompson saturation {name} must be contiguous float32 "
                f"with shape {shape}")
        if array.buffer.context is not context:
            raise ValueError("Thompson saturation arrays must share one Metal context")
        result[name] = array.view((array.size,), byte_offset=array.byte_offset)
    return result


def _table_bits(array: MetalArray, name: str, count: int,
                context) -> MetalArray:
    if not isinstance(array, MetalArray) or not array.flags.c_contiguous:
        raise TypeError(f"Thompson saturation {name} must be contiguous Metal storage")
    if array.dtype == _DOUBLE:
        bits = array.view_uint64_bits()
    elif array.dtype == _U64:
        bits = array
    else:
        raise TypeError(f"Thompson saturation {name} must store float64 bits")
    if bits.size != count:
        raise ValueError(f"Thompson saturation {name} has {bits.size} words, expected {count}")
    if bits.buffer.context is not context:
        raise ValueError("Thompson saturation tables and state must share one Metal context")
    return bits.view((count,), byte_offset=bits.byte_offset)


def _check_no_overlap(resources: dict[str, MetalArray],
                      allowed_aliases: set[str]) -> None:
    items = list(resources.items())
    for i, (left_name, left) in enumerate(items):
        left_start = left.byte_offset
        left_end = left_start + left.nbytes
        for right_name, right in items[:i]:
            if left_name in allowed_aliases or right_name in allowed_aliases:
                continue
            if left.buffer.handle != right.buffer.handle:
                continue
            right_start = right.byte_offset
            right_end = right_start + right.nbytes
            if left_start < right_end and right_start < left_end:
                raise ValueError(
                    f"restricted Thompson saturation resources overlap: "
                    f"{left_name}, {right_name}")


def _launch(name: str, resources: dict[str, MetalArray], parameters: dict,
            count: int, allowed_aliases: set[str] | None = None):
    if count == 0:
        return None
    _check_no_overlap(resources, allowed_aliases or set())
    return next(iter(resources.values())).buffer.context.submit_kernel(
        name, resources, parameters, exact_policy=True,
        grid_size=count, threads_per_group=_THREADS)


def submit_saturation_adjust(
    temperature: MetalArray,
    pressure: MetalArray,
    qv: MetalArray,
    qc: MetalArray,
    nc_entry: MetalArray,
    ncten: MetalArray,
    nwfaten: MetalArray,
    nwfa_work_m3: MetalArray,
    w: MetalArray,
    tnccn_act_bits: MetalArray,
    tnc_wev_bits: MetalArray,
    dt: float,
    *,
    reference_density: MetalArray | None = None,
    reference_temperature: MetalArray | None = None,
    condensation_rate: MetalArray | None = None,
    cloud_presence: MetalArray | None = None,
):
    """Apply MP28 condensation/evaporation and aerosol number tendencies.

    Optional reference and diagnostic outputs are bound to a harmless input
    resource when omitted; presence flags prevent their access in the kernel.
    Lookup tables may be float64 arrays or same-byte uint64 views.
    """
    if not isinstance(temperature, MetalArray) or temperature.dtype != _FLOAT:
        raise TypeError("Thompson saturation temperature must be float32 MetalArray")
    shape = temperature.shape
    if not shape or temperature.ndim > 3 or not temperature.flags.c_contiguous:
        raise ValueError("Thompson saturation fields need contiguous 1-D to 3-D storage")
    count = temperature.size
    if count > np.iinfo(np.int32).max:
        raise ValueError("Thompson saturation size must fit int32")
    context = temperature.buffer.context
    optional = {
        "reference_density": reference_density,
        "reference_temperature": reference_temperature,
        "condensation_rate": condensation_rate,
        "cloud_presence": cloud_presence,
    }
    fields = {
        "temperature": temperature, "pressure": pressure, "qv": qv,
        "qc": qc, "nc_entry": nc_entry, "ncten": ncten,
        "nwfaten": nwfaten, "nwfa_work_m3": nwfa_work_m3, "w": w,
    }
    resources = _active_arrays(fields, shape, context)
    resources["tnccn_act_bits"] = _table_bits(
        tnccn_act_bits, "tnccn_act_bits", _TNCCN_SIZE, context)
    resources["tnc_wev_bits"] = _table_bits(
        tnc_wev_bits, "tnc_wev_bits", _TNC_WEV_SIZE, context)
    optional_resources = _active_arrays(optional, shape, context)
    resources.update(optional_resources)
    for name, array in optional.items():
        if array is None:
            resources[name] = pressure.view((pressure.size,), byte_offset=pressure.byte_offset)
    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    parameters = {
        "dt": float(dt32), "size": count,
        "has_reference_density": int(reference_density is not None),
        "has_reference_temperature": int(reference_temperature is not None),
        "has_condensation_rate": int(condensation_rate is not None),
        "has_cloud_presence": int(cloud_presence is not None),
    }
    dummy_names = {name for name, array in optional.items() if array is None}
    return _launch("thompson_aa_saturation_adjust", resources, parameters,
                   count, dummy_names)


def submit_rain_evaporation(
    qr: MetalArray,
    nr: MetalArray,
    temperature: MetalArray,
    pressure: MetalArray,
    qv: MetalArray,
    dt: float,
    *,
    nwfaten: MetalArray | None = None,
    reference_density: MetalArray | None = None,
    reference_temperature: MetalArray | None = None,
    graupel_melt_marker: MetalArray | None = None,
    condensation_rate: MetalArray | None = None,
    entry_density: MetalArray | None = None,
):
    """Apply MP28 rain evaporation, latent heating, and CCN return."""
    if not isinstance(qr, MetalArray) or qr.dtype != _FLOAT:
        raise TypeError("Thompson rain evaporation qr must be float32 MetalArray")
    shape = qr.shape
    if not shape or qr.ndim > 3 or not qr.flags.c_contiguous:
        raise ValueError("Thompson rain fields need contiguous 1-D to 3-D storage")
    count = qr.size
    if count > np.iinfo(np.int32).max:
        raise ValueError("Thompson rain evaporation size must fit int32")
    context = qr.buffer.context
    optional = {
        "nwfaten": nwfaten,
        "reference_density": reference_density,
        "reference_temperature": reference_temperature,
        "graupel_melt_marker": graupel_melt_marker,
        "condensation_rate": condensation_rate,
        "entry_density": entry_density,
    }
    fields = {
        "qr": qr, "nr": nr, "temperature": temperature,
        "pressure": pressure, "qv": qv, **optional,
    }
    resources = _active_arrays(fields, shape, context)
    for name, array in optional.items():
        if array is None:
            resources[name] = pressure.view((pressure.size,), byte_offset=pressure.byte_offset)
    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    parameters = {
        "dt": float(dt32), "size": count,
        "has_nwfaten": int(nwfaten is not None),
        "has_reference_density": int(reference_density is not None),
        "has_reference_temperature": int(reference_temperature is not None),
        "has_graupel_marker": int(graupel_melt_marker is not None),
        "has_condensation_rate": int(condensation_rate is not None),
        "has_entry_density": int(entry_density is not None),
    }
    dummy_names = {name for name, array in optional.items() if array is None}
    return _launch("thompson_aa_rain_evaporation", resources, parameters,
                   count, dummy_names)


__all__ = ["submit_saturation_adjust", "submit_rain_evaporation"]
