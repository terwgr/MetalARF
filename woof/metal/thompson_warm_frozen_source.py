"""Typed Metal launcher for the classic Thompson warm frozen-source stage."""
from __future__ import annotations

import numpy as np

from .array import MetalArray

_TABLE_SHAPES = {
    "rain_cloud_efficiency": (100, 100),
    "snow_cloud_efficiency": (100, 100),
    **{name: (37, 9, 37, 37) for name in (
        "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
        "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
        "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2")},
    **{name: (37, 37, 1, 37, 37) for name in (
        "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr")},
}
_TABLE_LOGICAL_NAMES = {
    "rain_cloud_efficiency": "t_Efrw",
    "snow_cloud_efficiency": "t_Efsw",
}


def submit_warm_frozen_source_network(
        qc, qr, nr, qs, qg, graupel_number_per_kg,
        graupel_melt_marker, snow_melt_marker, temperature, pressure, qv,
        table_owner, dt):
    """Submit the source-ordered, entry-mask-held warm network stage."""
    from woof.core.thompson_runtime import DeviceClassicTableSet
    if not isinstance(table_owner, DeviceClassicTableSet) or not table_owner.roundtrip_verified:
        raise TypeError("table_owner must be a verified DeviceClassicTableSet")
    state = {
        "qc": qc, "qr": qr, "nr": nr, "qs": qs, "qg": qg,
        "graupel_number_per_kg": graupel_number_per_kg,
        "graupel_melt_marker": graupel_melt_marker,
        "snow_melt_marker": snow_melt_marker, "temperature": temperature,
        "pressure": pressure, "qv": qv,
    }
    first = qc
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

    tables = table_owner.cold_source_tables
    named = {
        "rain_cloud_efficiency": tables.rain_cloud_efficiency,
        "snow_cloud_efficiency": table_owner.t_Efsw,
    }
    named.update(zip(
        ("tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
         "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
         "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2"),
        tables.rain_snow_tables, strict=True))
    named.update(zip(
        ("tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr"),
        tables.rain_graupel_tables, strict=True))
    arrays = {name: value.view((value.size,))
              for name, value in state.items()}
    collision_arrays = {
        name: arrays[name] for name in (
            "temperature", "pressure", "qv", "qr", "nr", "qs", "qg",
            "graupel_number_per_kg", "graupel_melt_marker")}
    cloud_arrays = {
        name: arrays[name] for name in (
            "qc", "qr", "nr", "qs", "qg", "graupel_number_per_kg",
            "graupel_melt_marker", "temperature", "pressure", "qv")}
    for name, expected in _TABLE_SHAPES.items():
        logical_name = _TABLE_LOGICAL_NAMES.get(name, name)
        if tuple(table_owner.logical_shapes.get(logical_name, ())) != expected:
            raise ValueError(f"{name} logical shape does not match {expected}")
        array = named[name]
        if (not isinstance(array, MetalArray) or array.dtype != np.dtype(np.float64)
                or not array.flags.c_contiguous or array.ndim != 1
                or array.size != int(np.prod(expected))):
            raise TypeError(f"{name} must be flat contiguous float64 physical-Fortran storage")
        if array.buffer.context is not context:
            raise ValueError("Thompson table and state resources must share one Metal context")
        if name == "rain_cloud_efficiency":
            cloud_arrays[name] = array
        elif name == "snow_cloud_efficiency":
            frozen_cloud_table = array
        else:
            collision_arrays[name] = array
    wet_bulb = MetalArray(
        context.buffer(n * np.dtype(np.float32).itemsize), (n,), np.float32)
    context.submit_kernel(
        "thompson_warm_source_wet_bulb",
        {"temperature": arrays["temperature"],
         "pressure": arrays["pressure"], "qv": arrays["qv"],
         "graupel_melt_marker": arrays["graupel_melt_marker"],
         "wet_bulb": wet_bulb},
        {"n": n}, exact_policy=False, grid_size=n,
        threads_per_group=64)
    collision_arrays["wet_bulb"] = wet_bulb
    cloud_arrays["wet_bulb"] = wet_bulb
    cloud_state = MetalArray(
        context.buffer(64 * n * np.dtype(np.float64).itemsize),
        (64 * n,), np.float64)
    cloud_arrays["cloud_state"] = cloud_state
    context.submit_kernel(
        "thompson_warm_source_cloud_rates", cloud_arrays,
        {"n": n, "dt": float(dt32)}, exact_policy=False,
        grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_snow_cloud_rate",
        {"snow_cloud_efficiency": frozen_cloud_table,
         "cloud_state": cloud_state},
        {"n": n}, exact_policy=False, grid_size=n,
        threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_log_reduce",
        {"cloud_state": cloud_state}, {"mode": 0, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_log_poly",
        {"cloud_state": cloud_state}, {"mode": 0, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_log_compensate_product",
        {"cloud_state": cloud_state}, {"mode": 0, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_log_compensate_sum",
        {"cloud_state": cloud_state}, {"mode": 0, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_multiply",
        {"cloud_state": cloud_state}, {"mode": 0, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_t",
        {"cloud_state": cloud_state}, {"mode": 0, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_u",
        {"cloud_state": cloud_state}, {"mode": 0, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_v",
        {"cloud_state": cloud_state}, {"mode": 0, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_z",
        {"cloud_state": cloud_state}, {"mode": 0, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_w",
        {"cloud_state": cloud_state}, {"mode": 0, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_poly",
        {"cloud_state": cloud_state}, {"mode": 0, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_rational",
        {"cloud_state": cloud_state}, {"mode": 0, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_scale",
        {"cloud_state": cloud_state}, {"mode": 0, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_cloud_rate_base",
        {"cloud_state": cloud_state}, {"n": n}, exact_policy=False,
        grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_log_reduce",
        {"cloud_state": cloud_state}, {"mode": 1, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_log_poly",
        {"cloud_state": cloud_state}, {"mode": 1, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_log_compensate_product",
        {"cloud_state": cloud_state}, {"mode": 1, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_log_compensate_sum",
        {"cloud_state": cloud_state}, {"mode": 1, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_multiply",
        {"cloud_state": cloud_state}, {"mode": 1, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_t",
        {"cloud_state": cloud_state}, {"mode": 1, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_u",
        {"cloud_state": cloud_state}, {"mode": 1, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_v",
        {"cloud_state": cloud_state}, {"mode": 1, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_z",
        {"cloud_state": cloud_state}, {"mode": 1, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_w",
        {"cloud_state": cloud_state}, {"mode": 1, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_poly",
        {"cloud_state": cloud_state}, {"mode": 1, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_rational",
        {"cloud_state": cloud_state}, {"mode": 1, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_scale",
        {"cloud_state": cloud_state}, {"mode": 1, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_log_reduce",
        {"cloud_state": cloud_state}, {"mode": 2, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_log_poly",
        {"cloud_state": cloud_state}, {"mode": 2, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_log_compensate_product",
        {"cloud_state": cloud_state}, {"mode": 2, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_log_compensate_sum",
        {"cloud_state": cloud_state}, {"mode": 2, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_multiply",
        {"cloud_state": cloud_state}, {"mode": 2, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_t",
        {"cloud_state": cloud_state}, {"mode": 2, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_u",
        {"cloud_state": cloud_state}, {"mode": 2, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_v",
        {"cloud_state": cloud_state}, {"mode": 2, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_z",
        {"cloud_state": cloud_state}, {"mode": 2, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_w",
        {"cloud_state": cloud_state}, {"mode": 2, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_poly",
        {"cloud_state": cloud_state}, {"mode": 2, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_rational",
        {"cloud_state": cloud_state}, {"mode": 2, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_scale",
        {"cloud_state": cloud_state}, {"mode": 2, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_square",
        {"cloud_state": cloud_state}, {"n": n}, exact_policy=False,
        grid_size=n, threads_per_group=64)
    collision_rates = MetalArray(
        context.buffer(8 * n * np.dtype(np.float64).itemsize),
        (8 * n,), np.float64)
    collision_arrays["collision_rates"] = collision_rates
    context.submit_kernel(
        "thompson_warm_source_collision_rates", collision_arrays,
        {"n": n, "dt": float(dt32)}, exact_policy=False,
        grid_size=n, threads_per_group=64)
    arrays["cloud_state"] = cloud_state
    arrays["collision_rates"] = collision_rates
    context.submit_kernel(
        "thompson_warm_frozen_source_network_melt", arrays,
        {"n": n, "dt": float(dt32)}, exact_policy=False,
        grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_frozen_source_network_vapor", arrays,
        {"n": n, "dt": float(dt32)}, exact_policy=False,
        grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_frozen_source_network_limits", arrays,
        {"n": n, "dt": float(dt32)}, exact_policy=False,
        grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_frozen_source_network_apply", arrays,
        {"n": n, "dt": float(dt32)}, exact_policy=False,
        grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_log_reduce",
        {"cloud_state": cloud_state}, {"mode": 3, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_log_poly",
        {"cloud_state": cloud_state}, {"mode": 3, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_log_compensate_product",
        {"cloud_state": cloud_state}, {"mode": 3, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_log_compensate_sum",
        {"cloud_state": cloud_state}, {"mode": 3, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_multiply",
        {"cloud_state": cloud_state}, {"mode": 3, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_t",
        {"cloud_state": cloud_state}, {"mode": 3, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_u",
        {"cloud_state": cloud_state}, {"mode": 3, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_v",
        {"cloud_state": cloud_state}, {"mode": 3, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_z",
        {"cloud_state": cloud_state}, {"mode": 3, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_w",
        {"cloud_state": cloud_state}, {"mode": 3, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_poly",
        {"cloud_state": cloud_state}, {"mode": 3, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_rational",
        {"cloud_state": cloud_state}, {"mode": 3, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    context.submit_kernel(
        "thompson_warm_source_graupel_pow_exp_scale",
        {"cloud_state": cloud_state}, {"mode": 3, "n": n},
        exact_policy=False, grid_size=n, threads_per_group=64)
    return context.submit_kernel(
        "thompson_warm_frozen_source_network_finalize", arrays,
        {"n": n, "dt": float(dt32)}, exact_policy=False,
        grid_size=n, threads_per_group=64)
