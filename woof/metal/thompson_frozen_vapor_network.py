"""Typed Metal launcher for the classic Thompson frozen-vapor source stage."""
from __future__ import annotations

import numpy as np

from .array import MetalArray

_TABLE_SHAPES = {
    "ice_deposition_partition": (64, 55),
    "ice_to_snow_mass": (64, 55),
    "ice_to_snow_number": (64, 55),
    **{name: (37, 9, 37, 37) for name in (
        "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
        "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
        "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2")},
    **{name: (37, 37, 1, 37, 37) for name in (
        "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr")},
    **{name: (37, 37, 45, 55) for name in (
        "rain_to_ice_mass", "rain_to_ice_number",
        "rain_to_graupel_mass", "rain_to_graupel_number")},
    "rain_cloud_efficiency": (100, 100),
    "cloud_to_ice_mass": (37, 100, 45, 55),
    "cloud_to_ice_number": (37, 100, 45, 55),
}
_TABLE_LOGICAL_NAMES = {
    "ice_deposition_partition": "tpi_ide",
    "ice_to_snow_mass": "tps_iaus",
    "ice_to_snow_number": "tni_iaus",
    "rain_cloud_efficiency": "t_Efrw",
    "cloud_to_ice_mass": "tpi_qcfz",
    "cloud_to_ice_number": "tni_qcfz",
    "rain_to_ice_mass": "tpi_qrfz",
    "rain_to_ice_number": "tni_qrfz",
    "rain_to_graupel_mass": "tpg_qrfz",
    "rain_to_graupel_number": "tnr_qrfz",
}


def submit_frozen_vapor_network(
        qi, ni, qs, qg, qr, nr, temperature, pressure, qv, table_owner, dt,
        *, qc, graupel_number_shadow, snow_velocity_boost):
    """Submit the caller-selected full classic cold/cloud source stage.

    ``DeviceClassicTableSet`` stores every coefficient in physical
    Fortran-order one-dimensional float64 storage. The shader indexes those
    exact offsets; no reshaping or narrowing is performed here.
    """
    from woof.core.thompson_runtime import DeviceClassicTableSet
    if not isinstance(table_owner, DeviceClassicTableSet) or not table_owner.roundtrip_verified:
        raise TypeError("table_owner must be a verified DeviceClassicTableSet")
    state = {"qi": qi, "ni": ni, "qs": qs, "qg": qg, "qr": qr, "nr": nr,
             "temperature": temperature, "pressure": pressure, "qv": qv,
             "qc": qc, "graupel_number_shadow": graupel_number_shadow,
             "snow_velocity_boost": snow_velocity_boost}
    first = qi
    if not isinstance(first, MetalArray) or first.ndim not in (1, 2, 3):
        raise TypeError("Thompson state fields must be 1-D, 2-D, or 3-D MetalArrays")
    shape = first.shape
    if first.dtype != np.dtype(np.float32) or not first.flags.c_contiguous:
        raise TypeError("Thompson state fields require contiguous float32 storage")
    context = first.buffer.context
    for name, value in state.items():
        if (not isinstance(value, MetalArray) or value.shape != shape
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"{name} must match the contiguous float32 state shape")
        if value.buffer.context is not context:
            raise ValueError("Thompson resources must share one Metal context")
    n = int(np.prod(shape, dtype=np.int64))
    if n <= 0 or n > 0x7fffffff:
        raise ValueError("Thompson state size must fit positive CUDA int indexing")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0:
        raise ValueError("dt must be representable as a positive float32")
    arrays = {name: value.view((value.size,))
              for name, value in state.items()}
    tables = table_owner.cold_source_tables
    named = {
        "ice_deposition_partition": tables.ice_deposition_partition,
        "ice_to_snow_mass": tables.ice_to_snow_mass,
        "ice_to_snow_number": tables.ice_to_snow_number,
        "rain_cloud_efficiency": tables.rain_cloud_efficiency,
        "cloud_to_ice_mass": tables.cloud_freezing_tables[0],
        "cloud_to_ice_number": tables.cloud_freezing_tables[1],
    }
    for field, names in (
        ("rain_snow_tables", ("tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
                               "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
                               "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2")),
        ("rain_graupel_tables", ("tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr")),
        ("rain_freezing_tables", ("rain_to_ice_mass", "rain_to_ice_number",
                                   "rain_to_graupel_mass", "rain_to_graupel_number")),
    ):
        named.update(zip(names, getattr(tables, field), strict=True))
    for name, expected in _TABLE_SHAPES.items():
        logical_name = _TABLE_LOGICAL_NAMES.get(name, name)
        actual = tuple(table_owner.logical_shapes.get(logical_name, ()))
        if actual != expected:
            raise ValueError(f"{name} logical shape {actual} != {expected}")
        array = named[name]
        if (not isinstance(array, MetalArray) or array.dtype != np.dtype(np.float64)
                or not array.flags.c_contiguous or array.ndim != 1
                or array.size != int(np.prod(expected))):
            raise TypeError(f"{name} must be flat contiguous float64 physical-Fortran storage")
        if array.buffer.context is not context:
            raise ValueError("Thompson table and state resources must share one Metal context")
        arrays[name] = array
    for name, left in arrays.items():
        start, end = left.byte_offset, left.byte_offset + left.nbytes
        for other_name, right in list(arrays.items())[:list(arrays).index(name)]:
            if left.buffer is right.buffer and start < right.byte_offset + right.nbytes and right.byte_offset < end:
                raise ValueError(f"restricted Thompson resources overlap: {name}, {other_name}")
    return context.submit_kernel(
        "thompson_frozen_vapor_cloud_network", arrays,
        {"n": n, "dt": float(dt32), "track_graupel_number": 1,
         "include_snow_rime_conversion": 1, "include_cold_rain": 1,
         "include_cold_cloud": 1}, exact_policy=False,
        grid_size=n, threads_per_group=64)
