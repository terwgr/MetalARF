"""Typed Metal caller for Thompson's aerosol-aware cold source family."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray


_STATE_NAMES = (
    "qi", "ni", "qs", "qg", "qr", "nr", "qc", "temperature",
    "pressure", "qv", "nc_entry", "nwfa_entry", "nifa_entry", "ncten",
    "nwfaten", "nifaten", "graupel_number_shadow", "snow_velocity_boost",
)
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
        "rain_to_ice_mass", "rain_to_ice_number", "rain_to_graupel_mass",
        "rain_to_graupel_number")},
    "rain_cloud_efficiency": (100, 100),
    "cloud_to_ice_mass": (37, 100, 45, 55),
    "cloud_to_ice_number": (37, 100, 45, 55),
}
_TABLE_NAMES = tuple(_TABLE_SHAPES)
def require_thompson_aerosol_cold_metal_supported(cuda_symbol: str) -> None:
    """Accept the production network export name."""
    if cuda_symbol != "thompson_aa_cold_network":
        raise ValueError(f"unknown Thompson aerosol-cold CUDA export {cuda_symbol!r}")


def _checked_resources(state: dict[str, MetalArray], tables: dict[str, MetalArray],
                       size: int, *, wrf39: bool = False
                       ) -> tuple[dict[str, MetalArray], int]:
    first = state["qi"]
    if (not isinstance(first, MetalArray) or first.ndim not in (1, 2, 3)
            or first.dtype != np.dtype(np.float32)
            or not first.flags.c_contiguous):
        raise TypeError("Thompson cold state requires contiguous float32 MetalArrays")
    shape, context = first.shape, first.buffer.context
    for name, array in state.items():
        if (not isinstance(array, MetalArray) or array.shape != shape
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous):
            raise TypeError(f"{name} must match the contiguous float32 state shape")
        if array.buffer.context is not context:
            raise ValueError("Thompson cold resources must share one Metal context")

    n = int(np.prod(shape, dtype=np.int64))
    if isinstance(size, (bool, np.bool_)) or not isinstance(
            size, (int, np.integer)):
        raise TypeError("Thompson cold size must be an integer")
    size = int(size)
    if size <= 0 or size > n or size > 0x7fffffff:
        raise ValueError("Thompson cold size must fit the state and int32 indexing")

    if set(tables) != set(_TABLE_SHAPES):
        missing = sorted(set(_TABLE_SHAPES) - set(tables))
        extra = sorted(set(tables) - set(_TABLE_SHAPES))
        raise ValueError(f"cold table set mismatch; missing={missing}, extra={extra}")
    table_shapes = dict(_TABLE_SHAPES)
    if wrf39:
        table_shapes.update({name: (28, 9, 37, 37) for name in (
            "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
            "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
            "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2")})
        table_shapes.update({name: (28, 28, 37, 37) for name in (
            "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr")})
    for name, logical_shape in table_shapes.items():
        array = tables[name]
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float64)
                or not array.flags.c_contiguous or array.ndim != 1
                or array.size != int(np.prod(logical_shape))):
            raise TypeError(
                f"{name} must be flat contiguous float64 Fortran-order storage "
                f"with {int(np.prod(logical_shape))} values")
        if array.buffer.context is not context:
            raise ValueError("Thompson cold tables and state must share one Metal context")

    resources = {name: state[name].view((state[name].size,))
                 for name in _STATE_NAMES}
    resources.update(tables)
    # The CUDA signature uses restrict for all resources. Reject overlapping
    # views before asking Metal to bind them, including table/table aliases.
    items = list(resources.items())
    for i, (left_name, left) in enumerate(items):
        if left.nbytes == 0:
            continue
        left_start, left_end = left.byte_offset, left.byte_offset + left.nbytes
        for right_name, right in items[:i]:
            if left.buffer.handle != right.buffer.handle or right.nbytes == 0:
                continue
            right_start = right.byte_offset
            right_end = right_start + right.nbytes
            if left_start < right_end and right_start < left_end:
                raise ValueError(
                    f"restricted Thompson cold resources overlap: "
                    f"{left_name}, {right_name}")
    return resources, size


def submit_cold_network(
        qi: MetalArray, ni: MetalArray, qs: MetalArray, qg: MetalArray,
        qr: MetalArray, nr: MetalArray, qc: MetalArray,
        temperature: MetalArray, pressure: MetalArray, qv: MetalArray,
        nc_entry: MetalArray, nwfa_entry: MetalArray, nifa_entry: MetalArray,
        ncten: MetalArray, nwfaten: MetalArray, nifaten: MetalArray,
        graupel_number_shadow: MetalArray, snow_velocity_boost: MetalArray,
        ice_deposition_partition: MetalArray, ice_to_snow_mass: MetalArray,
        ice_to_snow_number: MetalArray, tcs_racs1: MetalArray,
        tmr_racs1: MetalArray, tcs_racs2: MetalArray, tmr_racs2: MetalArray,
        tcr_sacr1: MetalArray, tms_sacr1: MetalArray,
        tcr_sacr2: MetalArray, tms_sacr2: MetalArray,
        tnr_racs1: MetalArray, tnr_racs2: MetalArray,
        tnr_sacr1: MetalArray, tnr_sacr2: MetalArray,
        tcg_racg: MetalArray, tmr_racg: MetalArray, tcr_gacr: MetalArray,
        tnr_racg: MetalArray, tnr_gacr: MetalArray,
        rain_to_ice_mass: MetalArray, rain_to_ice_number: MetalArray,
        rain_to_graupel_mass: MetalArray, rain_to_graupel_number: MetalArray,
        rain_cloud_efficiency: MetalArray, cloud_to_ice_mass: MetalArray,
        cloud_to_ice_number: MetalArray, dt: float, size: int, *,
        wrf39: bool = False):
    """Submit the complete production network with every live mp=28 table.

    Arrays are flattened as storage views without converting the float64
    Fortran-order coefficient tables. Entry aerosol/droplet state is read-only;
    tendencies remain additive scratch for the terminal balance stage.
    """
    state_values = locals()
    state = {name: state_values[name] for name in _STATE_NAMES}
    tables = {name: state_values[name] for name in _TABLE_NAMES}
    if not isinstance(wrf39, (bool, np.bool_)):
        raise TypeError("Thompson cold wrf39 selector must be boolean")
    resources, n = _checked_resources(state, tables, size, wrf39=bool(wrf39))
    if isinstance(dt, (bool, np.bool_)):
        raise TypeError("Thompson cold dt must be a positive scalar")
    try:
        dt32 = np.float32(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("Thompson cold dt must be a positive scalar") from exc
    if not np.isfinite(dt32) or dt32 <= 0.0:
        raise ValueError("Thompson cold dt must be finite, positive, and representable as float32")
    return qi.buffer.context.submit_kernel(
        "thompson_aa_cold_network", resources,
        {"n": n, "dt": float(dt32), "wrf39": int(wrf39)},
        exact_policy=False,
        grid_size=n, threads_per_group=64)



__all__ = ["require_thompson_aerosol_cold_metal_supported",
           "submit_cold_network"]
