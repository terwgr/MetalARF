"""Typed Metal launchers for supportable Thompson scalar-stage kernels."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


_VOLUME_SHAPES = ("nz", "ny", "nx")


def _ensure_disjoint(arrays):
    ranges = []
    for name, array in arrays.items():
        start, end = array.byte_offset, array.byte_offset + array.nbytes
        for other_name, other_buffer, other_start, other_end in ranges:
            if (array.buffer is other_buffer and start < other_end
                    and other_start < end):
                raise ValueError(
                    f"Thompson resources {name} and {other_name} overlap; "
                    "the CUDA entrypoints declare restricted pointers")
        ranges.append((name, array.buffer, start, end))


def _flat_view(array):
    """Return a zero-copy one-dimensional view for linear-index kernels."""
    return array if array.ndim == 1 else array.view((array.size,))


def _flat_arrays(**arrays):
    first_name, first = next(iter(arrays.items()))
    if not isinstance(first, MetalArray) or first.ndim not in (1, 2, 3):
        raise TypeError(f"{first_name} must be a 1-D, 2-D, or 3-D MetalArray")
    shape = tuple(map(int, first.shape))
    if first.dtype != np.dtype(np.float32) or not first.flags.c_contiguous:
        raise TypeError(f"{first_name} requires contiguous float32 storage")
    for name, array in arrays.items():
        if not isinstance(array, MetalArray):
            raise TypeError(f"{name} must be a MetalArray")
        if array.shape != shape:
            raise ValueError(f"{name} shape {array.shape} != {shape}")
        if array.dtype != np.dtype(np.float32) or not array.flags.c_contiguous:
            raise TypeError(f"{name} requires contiguous float32 storage")
        if array.buffer.context is not first.buffer.context:
            raise ValueError("Thompson resources must share one Metal context")
    n = int(np.prod(shape, dtype=np.int64))
    dims = ((1, 1, shape[0]) if len(shape) == 1 else
            (1, shape[0], shape[1]) if len(shape) == 2 else shape)
    if max((*shape, n, *dims)) > 0x7FFFFFFF:
        raise ValueError("Thompson dimensions must fit CUDA int indexing")
    _ensure_disjoint(arrays)
    return first.buffer.context, n, dims


def _submit(entrypoint, arrays, *, threads=256):
    context, n, (nz, ny, nx) = _flat_arrays(**arrays)
    if n <= 0:
        return None
    params = {"n": n, "nz": nz, "ny": ny, "nx": nx}
    flat_arrays = {name: _flat_view(array)
                   for name, array in arrays.items()}
    return context.submit_kernel(entrypoint, flat_arrays, params,
                                 exact_policy=False, grid_size=n,
                                 threads_per_group=threads)


def _submit_column(entrypoint, arrays, *, volume_names, output_names):
    first = arrays[volume_names[0]]
    if not isinstance(first, MetalArray) or first.ndim != 3:
        raise TypeError("Thompson column kernels require 3-D volume fields")
    nz, ny, nx = map(int, first.shape)
    volume_shape, surface_shape = (nz, ny, nx), (ny, nx)
    context = first.buffer.context
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous):
            raise TypeError(f"{name} requires contiguous float32 Metal storage")
        expected = surface_shape if name in output_names else volume_shape
        if array.shape != expected:
            raise ValueError(f"{name} shape {array.shape} != {expected}")
        if array.buffer.context is not context:
            raise ValueError("Thompson resources must share one Metal context")
    _ensure_disjoint(arrays)
    if max(nz, ny, nx, ny * nx) > 0x7FFFFFFF:
        raise ValueError("Thompson column dimensions must fit CUDA int indexing")
    params = {"nz": nz, "ny": ny, "nx": nx}
    if entrypoint == "thompson_adapter_masks":
        params["columns"] = ny * nx
    return context.submit_kernel(
        entrypoint, arrays, params,
        exact_policy=False, grid_size=ny * nx, threads_per_group=32)


def submit_warm_saturation_adjust(temperature, pressure, qv, qc):
    arrays = {"temperature": temperature, "pressure": pressure,
              "qv": qv, "qc": qc}
    return _submit("thompson_warm_saturation_adjust", arrays)


def submit_exner_power(pressure, exner, *, reference_pressure,
                       exponent):
    """Apply the adapter's source p/P0 power on Metal device storage."""
    arrays = {"pressure": pressure, "exner": exner}
    context, n, (nz, ny, nx) = _flat_arrays(**arrays)
    if n <= 0:
        return None
    p0, power = np.float32(reference_pressure), np.float32(exponent)
    if not np.isfinite(p0) or p0 <= 0.0 or not np.isfinite(power):
        raise ValueError("Exner power constants must be finite and P0 positive")
    return context.submit_kernel(
        "thompson_exner_power",
        {name: _flat_view(array) for name, array in arrays.items()},
        {"size": n, "nz": nz, "ny": ny, "nx": nx,
         "reference_pressure": float(p0), "exponent": float(power)},
        exact_policy=False, grid_size=n, threads_per_group=256)


def submit_cloud_saturation_adjust(
        temperature, pressure, qv, qc, *, reference_density=None,
        reference_temperature=None, condensation_marker=None,
        cloud_presence=None):
    """Submit a typed saturation variant with caller-owned optional outputs.

    The presence form requires the full held-state/history tuple. The history
    form currently requires both held density and held temperature because
    the typed Metal resource ABI cannot encode CUDA null device pointers.
    """
    if cloud_presence is not None:
        if (reference_density is None or reference_temperature is None
                or condensation_marker is None):
            raise ValueError(
                "cloud_presence requires held density, held temperature, "
                "and condensation marker")
        entrypoint = "thompson_cloud_saturation_adjust_with_presence"
        arrays = {"temperature": temperature, "pressure": pressure,
                  "qv": qv, "qc": qc,
                  "reference_density": reference_density,
                  "reference_temperature": reference_temperature,
                  "condensation_marker": condensation_marker,
                  "cloud_presence": cloud_presence}
    elif condensation_marker is not None:
        arrays = {"temperature": temperature, "pressure": pressure,
                  "qv": qv, "qc": qc,
                  "condensation_marker": condensation_marker}
        if reference_density is None:
            entrypoint = "thompson_cloud_saturation_adjust_history_marker_only"
        elif reference_temperature is None:
            entrypoint = "thompson_cloud_saturation_adjust_history_density_marker"
            arrays["reference_density"] = reference_density
        else:
            entrypoint = "thompson_cloud_saturation_adjust_with_history"
            arrays["reference_density"] = reference_density
            arrays["reference_temperature"] = reference_temperature
    elif reference_temperature is not None:
        if reference_density is None:
            raise ValueError("reference_temperature requires reference_density")
        entrypoint = "thompson_cloud_saturation_adjust_with_state"
        arrays = {"temperature": temperature, "pressure": pressure,
                  "qv": qv, "qc": qc,
                  "reference_density": reference_density,
                  "reference_temperature": reference_temperature}
    elif reference_density is not None:
        entrypoint = "thompson_cloud_saturation_adjust_with_density"
        arrays = {"temperature": temperature, "pressure": pressure,
                  "qv": qv, "qc": qc,
                  "reference_density": reference_density}
    else:
        entrypoint = "thompson_cloud_saturation_adjust"
        arrays = {"temperature": temperature, "pressure": pressure,
                  "qv": qv, "qc": qc}
    return _submit(entrypoint, arrays)


def submit_graupel_fallout_column_mask(
        entry_active, qg, active_columns):
    arrays = {"entry_active": entry_active, "qg": qg,
              "active_columns": active_columns}
    return _submit_column(
        "thompson_graupel_fallout_column_mask", arrays,
        volume_names=("entry_active", "qg"),
        output_names=("active_columns",))


def submit_hydrometeor_column_mask(mixing_ratio, active_columns):
    arrays = {"mixing_ratio": mixing_ratio, "active_columns": active_columns}
    return _submit_column(
        "thompson_hydrometeor_column_mask", arrays,
        volume_names=("mixing_ratio",), output_names=("active_columns",))


def submit_microphysics_columns(
        qc, qi, qr, qs, qg, temperature, pressure, qv, micro_columns):
    arrays = {"qc": qc, "qi": qi, "qr": qr, "qs": qs, "qg": qg,
              "temperature": temperature, "pressure": pressure, "qv": qv,
              "micro_columns": micro_columns}
    return _submit_column(
        "thompson_microphysics_columns", arrays,
        volume_names=("qc", "qi", "qr", "qs", "qg", "temperature",
                      "pressure", "qv"),
        output_names=("micro_columns",))


def submit_adapter_masks(qr, qg, entry_graupel, rain_columns,
                         graupel_columns, *, nz):
    arrays = {"qr": qr, "qg": qg, "entry_graupel": entry_graupel,
              "rain_columns": rain_columns,
              "graupel_columns": graupel_columns}
    if not isinstance(qr, MetalArray) or qr.ndim != 3:
        raise TypeError("Thompson adapter masks require 3-D volume fields")
    nz_actual, ny, nx = map(int, qr.shape)
    if int(nz) != nz_actual:
        raise ValueError(f"nz {nz} != volume nz {nz_actual}")
    context = qr.buffer.context
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous):
            raise TypeError(f"{name} requires contiguous float32 Metal storage")
        expected = (ny, nx) if name in ("rain_columns", "graupel_columns") else (nz, ny, nx)
        if array.shape != expected:
            raise ValueError(f"{name} shape {array.shape} != {expected}")
        if array.buffer.context is not context:
            raise ValueError("Thompson resources must share one Metal context")
    _ensure_disjoint(arrays)
    return context.submit_kernel(
        "thompson_adapter_masks", arrays,
        {"nz": nz_actual, "ny": ny, "nx": nx, "columns": ny * nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=32)


def submit_adapter_finish(temperature, pii, th, thp, h_diabatic,
                          rainncv, snowncv, graupelncv, sr, *,
                          limit, dt, no_mp_heating=False):
    """Submit WRF's final theta/tendency and surface frozen-fraction writes."""
    if not isinstance(temperature, MetalArray) or temperature.ndim != 3:
        raise TypeError("Thompson adapter finish requires 3-D volume fields")
    nz, ny, nx = map(int, temperature.shape)
    columns = ny * nx
    arrays = {"temperature": temperature, "pii": pii, "th": th, "thp": thp,
              "h_diabatic": h_diabatic, "rainncv": rainncv,
              "snowncv": snowncv, "graupelncv": graupelncv, "sr": sr}
    context = temperature.buffer.context
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous):
            raise TypeError(f"{name} requires contiguous float32 Metal storage")
        expected = (ny, nx) if name in ("rainncv", "snowncv", "graupelncv", "sr") else (nz, ny, nx)
        if array.shape != expected:
            raise ValueError(f"{name} shape {array.shape} != {expected}")
        if array.buffer.context is not context:
            raise ValueError("Thompson resources must share one Metal context")
    _ensure_disjoint(arrays)
    if max(nz, ny, nx, columns, nz * columns) > 0x7FFFFFFF:
        raise ValueError("Thompson adapter dimensions must fit CUDA int indexing")
    limit32, dt32 = np.float32(limit), np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt!r}")
    return context.submit_kernel(
        "thompson_adapter_finish", arrays,
        {"limit": float(limit32), "dt": float(dt32),
         "no_mp_heating": int(bool(no_mp_heating)),
         "nz": nz, "ny": ny, "nx": nx, "columns": columns,
         "size": nz * columns},
        exact_policy=False, grid_size=nz * columns, threads_per_group=256)


def submit_adapter_prepare(
        thb, thp, phb, php, th, pii, temperature, dz, saved_theta,
        qc, qi, ni, qr, nr, qs, qg, entry_graupel, warm_entry,
        graupelncv, micro_columns, *, gravity):
    """Submit WRF entry thermodynamics, masks, and hydrometeor R1 cleanup."""
    if not isinstance(thp, MetalArray) or thp.ndim != 3:
        raise TypeError("Thompson adapter prepare requires 3-D cell fields")
    nz, ny, nx = map(int, thp.shape)
    columns, size = ny * nx, nz * ny * nx
    interface_shape = (nz + 1, ny, nx)
    arrays = {"thb": thb, "thp": thp, "phb": phb, "php": php,
              "th": th, "pii": pii, "temperature": temperature, "dz": dz,
              "saved_theta": saved_theta, "qc": qc, "qi": qi, "ni": ni,
              "qr": qr, "nr": nr, "qs": qs, "qg": qg,
              "entry_graupel": entry_graupel, "warm_entry": warm_entry,
              "graupelncv": graupelncv, "micro_columns": micro_columns}
    surface = {"graupelncv", "micro_columns"}
    context = thp.buffer.context
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous):
            raise TypeError(f"{name} requires contiguous float32 Metal storage")
        if name == "thb":
            valid = array.shape in ((nz,), (nz, ny, nx))
        elif name == "phb":
            valid = array.shape in ((nz + 1,), interface_shape)
        elif name == "php":
            valid = array.shape == interface_shape
        else:
            expected = (ny, nx) if name in surface else (nz, ny, nx)
            valid = array.shape == expected
        if not valid:
            raise ValueError(f"{name} has unsupported Thompson adapter shape {array.shape}")
        if array.buffer.context is not context:
            raise ValueError("Thompson resources must share one Metal context")
    _ensure_disjoint(arrays)
    if max(nz + 1, ny, nx, columns, size) > 0x7FFFFFFF:
        raise ValueError("Thompson adapter dimensions must fit CUDA int indexing")
    gravity32 = np.float32(gravity)
    return context.submit_kernel(
        "thompson_adapter_prepare", arrays,
        {"gravity": float(gravity32), "size": size, "columns": columns,
         "thb_full": int(thb.ndim == 3), "phb_full": int(phb.ndim == 3),
         "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=size, threads_per_group=256)


def submit_classic_graupel_number_init(
        qg, temperature, pressure, qv, graupel_number_per_kg):
    """Submit the classic mp=8 graupel-number diagnosis for one call entry."""
    arrays = {"qg": qg, "temperature": temperature, "pressure": pressure,
              "qv": qv, "graupel_number_per_kg": graupel_number_per_kg}
    return _submit("thompson_classic_graupel_number_init", arrays)


def submit_classic_graupel_number_finalize(
        qg, temperature, pressure, qv, graupel_number_per_kg):
    """Submit WRF's final graupel R1 cleanup and MVD number reconstruction."""
    arrays = {"qg": qg, "temperature": temperature, "pressure": pressure,
              "qv": qv, "graupel_number_per_kg": graupel_number_per_kg}
    return _submit("thompson_classic_graupel_number_finalize", arrays)


def submit_adapter_entry(
        qc, qi, qr, qs, qg, temperature, pressure, qv,
        graupel_number_per_kg, micro_columns):
    """Submit classic entry graupel diagnosis and no-microphysics marking.

    This preserves the CUDA entry's level-parallel writes of ``1.0`` to the
    owning column flag. The caller must clear ``micro_columns`` beforehand.
    """
    arrays = {"qc": qc, "qi": qi, "qr": qr, "qs": qs, "qg": qg,
              "temperature": temperature, "pressure": pressure, "qv": qv,
              "graupel_number_per_kg": graupel_number_per_kg,
              "micro_columns": micro_columns}
    first = arrays["qc"]
    if not isinstance(first, MetalArray) or first.ndim != 3:
        raise TypeError("Thompson adapter entry requires 3-D volume fields")
    nz, ny, nx = map(int, first.shape)
    columns, size = ny * nx, nz * ny * nx
    context = first.buffer.context
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous):
            raise TypeError(f"{name} requires contiguous float32 Metal storage")
        expected = (ny, nx) if name == "micro_columns" else (nz, ny, nx)
        if array.shape != expected:
            raise ValueError(f"{name} shape {array.shape} != {expected}")
        if array.buffer.context is not context:
            raise ValueError("Thompson resources must share one Metal context")
    _ensure_disjoint(arrays)
    if max(nz, ny, nx, columns, size) > 0x7FFFFFFF:
        raise ValueError("Thompson adapter dimensions must fit CUDA int indexing")
    return context.submit_kernel(
        "thompson_adapter_entry", arrays,
        {"size": size, "columns": columns, "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=columns, threads_per_group=256)
