"""Typed Metal boundary for Thompson aerosol warm-family CUDA exports."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray




def submit_warm_source_network(
        qc: MetalArray, qr: MetalArray, nr: MetalArray, qs: MetalArray,
        qg: MetalArray, graupel_number_per_kg: MetalArray,
        graupel_melt_marker: MetalArray, snow_melt_marker: MetalArray,
        temperature: MetalArray, pressure: MetalArray, qv: MetalArray,
        nc_entry: MetalArray, nwfa_entry: MetalArray, nifa_entry: MetalArray,
        ncten: MetalArray, nwfaten: MetalArray, nifaten: MetalArray,
        rain_cloud_efficiency: MetalArray, snow_cloud_efficiency: MetalArray,
        tcs_racs1: MetalArray, tmr_racs1: MetalArray,
        tcs_racs2: MetalArray, tmr_racs2: MetalArray,
        tcr_sacr1: MetalArray, tms_sacr1: MetalArray,
        tcr_sacr2: MetalArray, tms_sacr2: MetalArray,
        tnr_racs1: MetalArray, tnr_racs2: MetalArray,
        tnr_sacr1: MetalArray, tnr_sacr2: MetalArray,
        tcg_racg: MetalArray, tmr_racg: MetalArray,
        tcr_gacr: MetalArray, tnr_racg: MetalArray,
        tnr_gacr: MetalArray, dt: float, size: int, *, wrf39: bool = False):
    """Submit Thompson's warm source network over the supplied flat fields."""
    float_fields = {
        "qc": qc, "qr": qr, "nr": nr, "qs": qs, "qg": qg,
        "graupel_number_per_kg": graupel_number_per_kg,
        "graupel_melt_marker": graupel_melt_marker,
        "snow_melt_marker": snow_melt_marker, "temperature": temperature,
        "pressure": pressure, "qv": qv, "nc_entry": nc_entry,
        "nwfa_entry": nwfa_entry, "nifa_entry": nifa_entry,
        "ncten": ncten, "nwfaten": nwfaten, "nifaten": nifaten,
    }
    table_fields = {
        "rain_cloud_efficiency": rain_cloud_efficiency,
        "snow_cloud_efficiency": snow_cloud_efficiency,
        "tcs_racs1": tcs_racs1, "tmr_racs1": tmr_racs1,
        "tcs_racs2": tcs_racs2, "tmr_racs2": tmr_racs2,
        "tcr_sacr1": tcr_sacr1, "tms_sacr1": tms_sacr1,
        "tcr_sacr2": tcr_sacr2, "tms_sacr2": tms_sacr2,
        "tnr_racs1": tnr_racs1, "tnr_racs2": tnr_racs2,
        "tnr_sacr1": tnr_sacr1, "tnr_sacr2": tnr_sacr2,
        "tcg_racg": tcg_racg, "tmr_racg": tmr_racg,
        "tcr_gacr": tcr_gacr, "tnr_racg": tnr_racg,
        "tnr_gacr": tnr_gacr,
    }
    if isinstance(size, (bool, np.bool_)) or not isinstance(
            size, (int, np.integer)):
        raise TypeError("Thompson warm-network size must be an integer")
    size = int(size)
    if size < 0 or size > np.iinfo(np.int32).max:
        raise ValueError("Thompson warm-network size must fit int32")
    if isinstance(dt, (bool, np.bool_)):
        raise TypeError("Thompson warm-network dt must be a positive scalar")
    try:
        dt = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "Thompson warm-network dt must be a positive scalar") from exc
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Thompson warm-network dt must be finite and positive")

    first = qc
    if (not isinstance(first, MetalArray)
            or first.dtype != np.dtype(np.float32)
            or not first.flags.c_contiguous or first.ndim == 0):
        raise TypeError(
            "Thompson warm-network fields must be contiguous float32 arrays")
    if size > first.size:
        raise ValueError("Thompson warm-network size exceeds field size")
    shape = first.shape
    context = first.buffer.context
    flat = {}
    ranges = []
    for name, array in float_fields.items():
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous or array.shape != shape):
            raise TypeError(
                f"Thompson warm-network {name} must be contiguous float32 "
                f"with shape {shape}")
        if array.buffer.context is not context:
            raise ValueError(
                "Thompson warm-network arrays must share one Metal context")
        flat[name] = array.view((array.size,), byte_offset=array.byte_offset)
        ranges.append((name, array, name in {
            "qc", "qr", "nr", "qs", "qg", "graupel_number_per_kg",
            "graupel_melt_marker", "snow_melt_marker", "temperature", "qv",
            "ncten", "nwfaten", "nifaten"}))

    table_shapes = {
        "rain_cloud_efficiency": (100, 100),
        "snow_cloud_efficiency": (100, 100),
        # Graupel collision tables carry the classic hail-density axis
        # (size 1 for mp=28); the kernel's ts stride is 37*37*1*37*37.
        **{graupel_name: ((28, 28, 37, 37) if wrf39 else (37, 37, 1, 37, 37)) for graupel_name in (
            "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr")},
        **{snow_name: ((28, 9, 37, 37) if wrf39 else (37, 9, 37, 37)) for snow_name in (
            "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
            "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
            "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2")},
    }
    for name, array in table_fields.items():
        expected_shape = table_shapes.get(name, (37, 9, 37, 37))
        # The shared classic table owner stores every table as flat
        # Fortran-order float64 (the cold network takes the same buffers);
        # the kernel indexes that storage directly, so either form binds.
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float64)
                or not array.flags.c_contiguous
                or (array.shape != expected_shape
                    and array.shape != (int(np.prod(expected_shape)),))):
            raise TypeError(
                f"Thompson warm-network {name} must be contiguous float64 "
                f"with shape {expected_shape} or its flat storage")
        if array.buffer.context is not context:
            raise ValueError(
                "Thompson warm-network arrays must share one Metal context")
        # The generated ABI binds float64 storage; the shader reads the
        # binary64 values as their uint64 bit words.
        flat[name] = array.view((array.size,), byte_offset=array.byte_offset)
        ranges.append((name, array, False))

    for index, (left_name, left, left_write) in enumerate(ranges):
        for right_name, right, right_write in ranges[index + 1:]:
            if (left.buffer.handle != right.buffer.handle
                    or not (left_write or right_write)):
                continue
            if (left.byte_offset < right.byte_offset + right.nbytes
                    and right.byte_offset < left.byte_offset + left.nbytes):
                raise ValueError(
                    f"Thompson warm-network resources {left_name} and "
                    f"{right_name} overlap")
    if size == 0:
        return None
    return context.submit_kernel(
        "thompson_aa_warm_source_network", flat,
        {"dt": dt, "n": size, "wrf39": int(bool(wrf39))},
        exact_policy=True, grid_size=size, threads_per_group=128)


def submit_ncten_balance(
        qc_entry: MetalArray, qc_after: MetalArray, nc_entry: MetalArray,
        density: MetalArray, ncten: MetalArray, dt: float, size: int):
    """Apply Thompson's once-per-call cloud-water number limiter."""
    arrays = {
        "qc_entry": qc_entry,
        "qc_after": qc_after,
        "nc_entry": nc_entry,
        "density": density,
        "ncten": ncten,
    }
    if isinstance(size, (bool, np.bool_)) or not isinstance(
            size, (int, np.integer)):
        raise TypeError("Thompson ncten balance size must be an integer")
    size = int(size)
    if size < 0 or size > np.iinfo(np.int32).max:
        raise ValueError("Thompson ncten balance size must fit int32")
    if isinstance(dt, (bool, np.bool_)):
        raise TypeError("Thompson ncten balance dt must be a positive scalar")
    try:
        dt = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "Thompson ncten balance dt must be a positive scalar") from exc
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Thompson ncten balance dt must be finite and positive")

    first = qc_entry
    if not isinstance(first, MetalArray):
        raise TypeError("Thompson ncten balance qc_entry must be a MetalArray")
    if (first.dtype != np.dtype(np.float32)
            or not first.flags.c_contiguous or first.ndim == 0):
        raise TypeError(
            "Thompson ncten balance fields must be contiguous float32 arrays")
    if size > first.size:
        raise ValueError("Thompson ncten balance size exceeds the array size")
    shape = first.shape
    context = first.buffer.context
    flat = {}
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous or array.shape != shape):
            raise TypeError(
                f"Thompson ncten balance {name} must be contiguous float32 "
                f"with shape {shape}")
        if array.buffer.context is not context:
            raise ValueError(
                "Thompson ncten balance arrays must share one Metal context")
        flat[name] = array.view((array.size,), byte_offset=array.byte_offset)

    if qc_entry.buffer.handle == qc_after.buffer.handle:
        entry_start = qc_entry.byte_offset
        entry_end = entry_start + qc_entry.nbytes
        after_start = qc_after.byte_offset
        after_end = after_start + qc_after.nbytes
        if entry_start < after_end and after_start < entry_end:
            raise ValueError(
                "Thompson qc_entry must not overlap post-source qc_after")

    output_start = ncten.byte_offset
    output_end = output_start + ncten.nbytes
    for name, array in arrays.items():
        if name == "ncten" or array.buffer.handle != ncten.buffer.handle:
            continue
        if output_start < array.byte_offset + array.nbytes and \
                array.byte_offset < output_end:
            raise ValueError(
                f"Thompson ncten output overlaps resource {name}")
    if size == 0:
        return None
    return context.submit_kernel(
        "thompson_aa_ncten_balance", flat, {"dt": dt, "n": size},
        exact_policy=True, grid_size=size, threads_per_group=128)


__all__ = ["submit_ncten_balance", "submit_warm_source_network"]
