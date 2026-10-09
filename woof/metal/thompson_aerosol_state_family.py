"""Typed Metal submitters for supportable Thompson aerosol state kernels."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


_FLOAT = np.dtype(np.float32)
_THREADS = 128


def _overlap(left: MetalArray, right: MetalArray) -> bool:
    if left.buffer.handle != right.buffer.handle:
        return False
    return (left.byte_offset < right.byte_offset + right.nbytes
            and right.byte_offset < left.byte_offset + left.nbytes)


def _submit_linear(name: str, arrays: dict[str, MetalArray],
                   outputs: tuple[str, ...],
                   parameters: dict[str, int | float] | None = None):
    if not arrays:
        raise ValueError("Thompson aerosol submission needs resources")
    first_name, first = next(iter(arrays.items()))
    if not isinstance(first, MetalArray):
        raise TypeError(f"Thompson aerosol {first_name} must be a MetalArray")
    shape = first.shape
    if not shape:
        raise ValueError("Thompson aerosol fields must have at least one axis")
    count = first.size
    if count > np.iinfo(np.int32).max:
        raise ValueError("Thompson aerosol element count must fit int32")
    if count == 0:
        return None

    flat_arrays = {}
    context = first.buffer.context
    for field, array in arrays.items():
        if (not isinstance(array, MetalArray) or array.dtype != _FLOAT
                or not array.flags.c_contiguous):
            raise TypeError(
                f"Thompson aerosol {field} must be contiguous float32 storage")
        if array.shape != shape:
            raise ValueError(
                f"Thompson aerosol {field} shape {array.shape} != {shape}")
        if array.buffer.context is not context:
            raise ValueError("Thompson aerosol arrays must share one Metal context")
        flat_arrays[field] = array.view((count,), byte_offset=array.byte_offset)

    for output_name in outputs:
        output = arrays[output_name]
        output_start = output.byte_offset
        output_end = output_start + output.nbytes
        for other_name, other in arrays.items():
            if other_name == output_name or other.buffer.handle != output.buffer.handle:
                continue
            other_start = other.byte_offset
            other_end = other_start + other.nbytes
            if output_start < other_end and other_start < output_end:
                raise ValueError(
                    f"Thompson aerosol output {output_name} overlaps "
                    f"resource {other_name}")

    launch_parameters = {"n": count}
    if parameters:
        launch_parameters.update(parameters)
    return context.submit_kernel(
        name, flat_arrays, launch_parameters, exact_policy=True,
        grid_size=count, threads_per_group=_THREADS)


def submit_aerosol_entry_snapshot(
    temperature: MetalArray,
    pressure: MetalArray,
    qv: MetalArray,
    nwfa: MetalArray,
    nifa: MetalArray,
    rho_out: MetalArray,
    nwfa_entry_m3: MetalArray,
    nifa_entry_m3: MetalArray,
):
    """Write WRF's entry density and bounded per-m3 aerosol snapshots.

    Input aerosol state is read-only. The outputs must be caller-provided
    same-shape float32 Metal arrays. The operation uses the entry qv floor and
    WRF's independently bounded NWFA/NIFA snapshots; it allocates no scratch.
    """
    return _submit_linear("thompson_aa_entry_snapshot", {
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "nwfa": nwfa,
        "nifa": nifa,
        "rho_out": rho_out,
        "nwfa_entry_m3": nwfa_entry_m3,
        "nifa_entry_m3": nifa_entry_m3,
    }, outputs=("rho_out", "nwfa_entry_m3", "nifa_entry_m3"))


def submit_aerosol_micro_columns(
    qc: MetalArray,
    qi: MetalArray,
    qr: MetalArray,
    qs: MetalArray,
    qg: MetalArray,
    temperature: MetalArray,
    pressure: MetalArray,
    qv: MetalArray,
    micro_columns: MetalArray,
):
    """Classify entry columns using WRF's condensate/supersaturation test."""
    fields = {
        "qc": qc, "qi": qi, "qr": qr, "qs": qs, "qg": qg,
        "temperature": temperature, "pressure": pressure, "qv": qv,
    }
    if not isinstance(qc, MetalArray) or qc.ndim != 3:
        raise TypeError("Thompson aerosol qc must be a 3-D MetalArray")
    nz, ny, nx = qc.shape
    if nz <= 0 or ny <= 0 or nx <= 0:
        raise ValueError("Thompson aerosol entry columns must be nonempty")
    if not isinstance(micro_columns, MetalArray):
        raise TypeError("Thompson aerosol micro_columns must be a MetalArray")
    if micro_columns.shape != (ny, nx) or micro_columns.dtype != _FLOAT:
        raise TypeError(
            "Thompson aerosol micro_columns must be float32 with shape "
            f"{(ny, nx)}")
    context = qc.buffer.context
    if micro_columns.buffer.context is not context:
        raise ValueError("Thompson aerosol arrays must share one Metal context")
    if not micro_columns.flags.c_contiguous:
        raise ValueError("Thompson aerosol micro_columns must be C-contiguous")
    ncolumns = ny * nx
    if ncolumns > np.iinfo(np.int32).max or nz > np.iinfo(np.int32).max:
        raise ValueError("Thompson aerosol dimensions must fit int32")
    resources = {}
    for name, array in fields.items():
        if (not isinstance(array, MetalArray) or array.dtype != _FLOAT
                or array.shape != qc.shape or not array.flags.c_contiguous):
            raise TypeError(
                f"Thompson aerosol {name} must be C-contiguous float32 "
                f"with shape {qc.shape}")
        if array.buffer.context is not context:
            raise ValueError("Thompson aerosol arrays must share one Metal context")
        resources[name] = array.view((array.size,), byte_offset=array.byte_offset)
    output_start = micro_columns.byte_offset
    output_end = output_start + micro_columns.nbytes
    for name, array in fields.items():
        if array.buffer.handle != micro_columns.buffer.handle:
            continue
        start, end = array.byte_offset, array.byte_offset + array.nbytes
        if output_start < end and start < output_end:
            raise ValueError(
                f"Thompson aerosol output micro_columns overlaps resource {name}")
    resources["micro_columns"] = micro_columns.view(
        (ncolumns,), byte_offset=micro_columns.byte_offset)
    return context.submit_kernel(
        "thompson_aa_micro_columns", resources,
        {"nz": nz, "ncol": ncolumns}, exact_policy=True,
        grid_size=ncolumns, threads_per_group=_THREADS)


def submit_aerosol_entry_cloud_number(
    qc: MetalArray,
    nc: MetalArray,
    rho: MetalArray,
    rc_out: MetalArray,
    nc_entry_m3: MetalArray,
    nu_c_out: MetalArray,
    l_qc_out: MetalArray,
):
    """Diagnose WRF's entry cloud distribution and apply its dry-level reset."""
    fields = {
        "qc": qc, "nc": nc, "rho": rho, "rc_out": rc_out,
        "nc_entry_m3": nc_entry_m3,
    }
    if not isinstance(qc, MetalArray) or qc.ndim != 3:
        raise TypeError("Thompson aerosol qc must be a 3-D MetalArray")
    shape = qc.shape
    if any(dimension <= 0 for dimension in shape):
        raise ValueError("Thompson aerosol entry fields must be nonempty")
    count = qc.size
    if count > np.iinfo(np.int32).max:
        raise ValueError("Thompson aerosol element count must fit int32")
    context = qc.buffer.context
    resources = {}
    for name, array in fields.items():
        if (not isinstance(array, MetalArray) or array.dtype != _FLOAT
                or array.shape != shape or not array.flags.c_contiguous):
            raise TypeError(
                f"Thompson aerosol {name} must be C-contiguous float32 "
                f"with shape {shape}")
        if array.buffer.context is not context:
            raise ValueError("Thompson aerosol arrays must share one Metal context")
        resources[name] = array.view((count,), byte_offset=array.byte_offset)
    int_outputs = {"nu_c_out": nu_c_out, "l_qc_out": l_qc_out}
    for name, array in int_outputs.items():
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.int32)
                or array.shape != shape or not array.flags.c_contiguous):
            raise TypeError(
                f"Thompson aerosol {name} must be C-contiguous int32 "
                f"with shape {shape}")
        if array.buffer.context is not context:
            raise ValueError("Thompson aerosol arrays must share one Metal context")
        resources[name] = array.view((count,), byte_offset=array.byte_offset)

    # qc and nc are intentionally read/write in place on the dry-level branch.
    # Every distinct output must remain disjoint from those state arrays and
    # from the other outputs.
    output_arrays = {"rc_out": rc_out, "nc_entry_m3": nc_entry_m3,
                     **int_outputs}
    occupied = {"qc": qc, "nc": nc, "rho": rho}
    for name, output in output_arrays.items():
        start, end = output.byte_offset, output.byte_offset + output.nbytes
        for other_name, other in {**occupied, **{
                key: value for key, value in output_arrays.items()
                if key != name}}.items():
            if (output.buffer.handle == other.buffer.handle
                    and start < other.byte_offset + other.nbytes
                    and other.byte_offset < end):
                raise ValueError(
                    f"Thompson aerosol output {name} overlaps {other_name}")
    return context.submit_kernel(
        "thompson_aa_entry_cloud_number", resources, {"n": count},
        exact_policy=True, grid_size=count, threads_per_group=_THREADS)


def submit_aerosol_state_finalize_with_columns(
    qc: MetalArray,
    nc: MetalArray,
    nwfa: MetalArray,
    nifa: MetalArray,
    ncten: MetalArray,
    nwfaten: MetalArray,
    nifaten: MetalArray,
    rho: MetalArray,
    dt: float,
    nc_out: MetalArray,
    nwfa_out: MetalArray,
    nifa_out: MetalArray,
    qv: MetalArray,
    micro_columns: MetalArray,
):
    """Apply MP28 terminal tendencies while honoring entry column exits."""
    fields = {
        "qc": qc, "nc": nc, "nwfa": nwfa, "nifa": nifa,
        "ncten": ncten, "nwfaten": nwfaten, "nifaten": nifaten,
        "rho": rho, "nc_out": nc_out, "nwfa_out": nwfa_out,
        "nifa_out": nifa_out, "qv": qv,
    }
    if not isinstance(qc, MetalArray) or qc.ndim != 3:
        raise TypeError("Thompson aerosol qc must be a 3-D MetalArray")
    shape = qc.shape
    if any(dimension <= 0 for dimension in shape):
        raise ValueError("Thompson aerosol finalize fields must be nonempty")
    nz, ny, nx = shape
    count = qc.size
    ncolumns = ny * nx
    if count > np.iinfo(np.int32).max or ncolumns > np.iinfo(np.int32).max:
        raise ValueError("Thompson aerosol dimensions must fit int32")
    context = qc.buffer.context
    resources = {}
    for name, array in fields.items():
        if (not isinstance(array, MetalArray) or array.dtype != _FLOAT
                or array.shape != shape or not array.flags.c_contiguous):
            raise TypeError(
                f"Thompson aerosol {name} must be C-contiguous float32 "
                f"with shape {shape}")
        if array.buffer.context is not context:
            raise ValueError("Thompson aerosol arrays must share one Metal context")
        resources[name] = array.view((count,), byte_offset=array.byte_offset)
    if (not isinstance(micro_columns, MetalArray)
            or micro_columns.dtype != _FLOAT
            or micro_columns.shape != (ny, nx)
            or not micro_columns.flags.c_contiguous):
        raise TypeError(
            "Thompson aerosol micro_columns must be C-contiguous float32 "
            f"with shape {(ny, nx)}")
    if micro_columns.buffer.context is not context:
        raise ValueError("Thompson aerosol arrays must share one Metal context")
    resources["micro_columns"] = micro_columns.view(
        (ncolumns,), byte_offset=micro_columns.byte_offset)

    dt_f32 = np.float32(dt)
    if not np.isfinite(dt_f32) or dt_f32 <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")

    mask_start = micro_columns.byte_offset
    mask_end = mask_start + micro_columns.nbytes
    for name, array in fields.items():
        if (array.buffer.handle == micro_columns.buffer.handle
                and mask_start < array.byte_offset + array.nbytes
                and array.byte_offset < mask_end):
            raise ValueError(
                f"Thompson aerosol micro_columns overlaps resource {name}")

    # In-place output aliases are legal only for their matching state field;
    # shifted/partial aliases can make one thread overwrite another's input.
    allowed_pairs = {"nc_out": "nc", "nwfa_out": "nwfa",
                     "nifa_out": "nifa"}
    output_names = ("nc_out", "nwfa_out", "nifa_out")
    for name in output_names:
        output = fields[name]
        start, end = output.byte_offset, output.byte_offset + output.nbytes
        for other_name, other in fields.items():
            if other_name == name:
                continue
            overlaps = (output.buffer.handle == other.buffer.handle
                        and start < other.byte_offset + other.nbytes
                        and other.byte_offset < end)
            if not overlaps:
                continue
            exact_pair = (allowed_pairs.get(name) == other_name
                          and start == other.byte_offset
                          and end == other.byte_offset + other.nbytes)
            if not exact_pair:
                raise ValueError(
                    f"Thompson aerosol output {name} overlaps {other_name}")
        if (output.buffer.handle == micro_columns.buffer.handle
                and start < micro_columns.byte_offset + micro_columns.nbytes
                and micro_columns.byte_offset < end):
            raise ValueError(
                f"Thompson aerosol output {name} overlaps micro_columns")

    return context.submit_kernel(
        "thompson_aa_state_finalize_with_columns", resources,
        {"dt": float(dt_f32), "ncol": ncolumns, "n": count},
        exact_policy=True, grid_size=count, threads_per_group=_THREADS)


def submit_tau1_density(
    temperature: MetalArray,
    pressure: MetalArray,
    qv: MetalArray,
    rho_out: MetalArray,
):
    """Write the source's TAU+1 density from already updated temperature/qv."""
    return _submit_linear("thompson_aa_tau1_density", {
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "rho_out": rho_out,
    }, outputs=("rho_out",))


def submit_working_number(
    nwfa: MetalArray,
    nwfaten: MetalArray,
    rho: MetalArray,
    dt: float,
    nwfa_work_m3: MetalArray,
):
    """Refresh WRF's uncapped working water-friendly aerosol number."""
    dt_f32 = np.float32(dt)
    if not np.isfinite(dt_f32) or dt_f32 <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    return _submit_linear(
        "thompson_aa_working_number", {
            "nwfa": nwfa,
            "nwfaten": nwfaten,
            "rho": rho,
            "nwfa_work_m3": nwfa_work_m3,
        }, outputs=("nwfa_work_m3",), parameters={"dt": float(dt_f32)})


def submit_surface_emission(
    nwfa: MetalArray,
    nifa: MetalArray,
    nwfa2d: MetalArray,
    nifa2d: MetalArray,
    dt: float,
):
    """Apply WRF's unclamped lowest-level aerosol emission update.

    The volume inputs must share shape ``(nz, ny, nx)`` and the two source
    fields shape ``(ny, nx)``. Only the first plane of each volume is written.
    """
    if not isinstance(nwfa, MetalArray) or nwfa.ndim != 3:
        raise TypeError("Thompson aerosol nwfa must be a 3-D MetalArray")
    nz, ny, nx = nwfa.shape
    if nz < 1:
        raise ValueError("Thompson aerosol surface emission needs nz >= 1")
    expected = (ny, nx)
    arrays = {"nwfa": nwfa, "nifa": nifa,
              "nwfa2d": nwfa2d, "nifa2d": nifa2d}
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray)
                or array.dtype != _FLOAT
                or not array.flags.c_contiguous):
            raise TypeError(
                f"Thompson aerosol {name} must be contiguous float32 storage")
        shape = nwfa.shape if name in ("nwfa", "nifa") else expected
        if array.shape != shape:
            raise ValueError(
                f"Thompson aerosol {name} shape {array.shape} != {shape}")
        if array.buffer.context is not nwfa.buffer.context:
            raise ValueError(
                "Thompson aerosol arrays must share one Metal context")

    ranges = []
    for name, array in arrays.items():
        start, end = array.byte_offset, array.byte_offset + array.nbytes
        for other_name, other_buffer, other_start, other_end in ranges:
            if (array.buffer.handle == other_buffer.handle
                    and start < other_end and other_start < end):
                raise ValueError(
                    f"Thompson aerosol resources {name} and {other_name} "
                    "overlap")
        ranges.append((name, array.buffer, start, end))

    dt_f32 = np.float32(dt)
    ncolumns = ny * nx
    if not np.isfinite(dt_f32) or dt_f32 <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if ncolumns > np.iinfo(np.int32).max:
        raise ValueError("Thompson aerosol columns must fit int32")
    if ncolumns == 0:
        return None
    flat = {name: array.view((array.size,), byte_offset=array.byte_offset)
            for name, array in arrays.items()}
    return nwfa.buffer.context.submit_kernel(
        "thompson_aa_surface_emission", flat,
        {"dt": float(dt_f32), "ncolumns": ncolumns},
        exact_policy=True, grid_size=ncolumns, threads_per_group=128)


def submit_effective_radius(
    temperature: MetalArray,
    pressure: MetalArray,
    qv: MetalArray,
    qc: MetalArray,
    nc: MetalArray,
    qi: MetalArray,
    ni: MetalArray,
    qs: MetalArray,
    effc: MetalArray,
    effi: MetalArray,
    effs: MetalArray,
):
    """Write MP28 cloud, ice, and snow effective radii in microns."""
    return _submit_linear("thompson_aa_effective_radius", {
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "qc": qc,
        "nc": nc,
        "qi": qi,
        "ni": ni,
        "qs": qs,
        "effc": effc,
        "effi": effi,
        "effs": effs,
    }, outputs=("effc", "effi", "effs"))


def submit_wrf39_effective_radius(
    temperature: MetalArray, pressure: MetalArray, qv: MetalArray,
    qc: MetalArray, nc: MetalArray, qi: MetalArray, ni: MetalArray,
    qs: MetalArray, effc: MetalArray, effi: MetalArray, effs: MetalArray,
):
    """Write WRF39 radii with fork-specific cloud-number and floors."""
    return _submit_linear("thompson_aa_wrf39_effective_radius", {
        "temperature": temperature, "pressure": pressure, "qv": qv,
        "qc": qc, "nc": nc, "qi": qi, "ni": ni, "qs": qs,
        "effc": effc, "effi": effi, "effs": effs,
    }, outputs=("effc", "effi", "effs"))


def submit_wrf39_graupel_intercept(
    qg: MetalArray, qr: MetalArray, nr: MetalArray,
    temperature: MetalArray, pressure: MetalArray, qv: MetalArray,
    n0_out: MetalArray, *, mode: int,
):
    """Submit the WRF39 top-down graupel intercept/slope pass."""
    if (isinstance(mode, (bool, np.bool_))
            or not isinstance(mode, (int, np.integer)) or int(mode) not in (0, 1, 2)):
        raise ValueError("WRF39 graupel intercept mode must be 0, 1, or 2")
    arrays = {"qg": qg, "qr": qr, "nr": nr, "temperature": temperature,
              "pressure": pressure, "qv": qv, "n0_out": n0_out}
    if not isinstance(qg, MetalArray) or qg.ndim != 3:
        raise TypeError("WRF39 graupel intercept requires 3-D MetalArrays")
    shape = tuple(map(int, qg.shape))
    nz, ny, nx = shape
    if min(shape) <= 0 or max(nz, ny, nx, ny * nx) > np.iinfo(np.int32).max:
        raise ValueError("WRF39 graupel intercept dimensions must fit int32")
    context = qg.buffer.context
    for name, value in arrays.items():
        if (not isinstance(value, MetalArray) or value.shape != shape
                or value.dtype != _FLOAT or not value.flags.c_contiguous):
            raise TypeError(f"WRF39 intercept {name} must be contiguous float32 with shape {shape}")
        if value.buffer.context is not context:
            raise ValueError("WRF39 intercept arrays must share one Metal context")
    for name, output in (("n0_out", n0_out),):
        for other_name, other in arrays.items():
            if other_name == name:
                continue
            if _overlap(output, other):
                raise ValueError(f"WRF39 intercept output overlaps {other_name}")

    from woof.core.thompson_aerosol_contract import AM_R, D0R_M
    f32 = lambda x: float(np.float32(x))
    # The registered WRF39 intercept ABI binds each field as (nz, ny, nx).
    # Keep those validated contiguous arrays at their registered rank.
    resources = dict(arrays)
    parameters = {
        "r1": 1.0e-12, "r2": 1.0e-6, "d0r": f32(D0R_M),
        "am_r": f32(AM_R), "obmr": f32(1.0 / 3.0),
        "gonv_min": f32(1.0e4), "gonv_max": f32(3.0e6),
        "pi": f32(3.1415926536), "rho_g": f32(500.0),
        "mvd_numerator": f32(3.672), "qv_floor": f32(1.0e-10),
        "nz": nz, "ny": ny, "nx": nx, "mode": int(mode),
    }
    return context.submit_kernel(
        "thompson_aa_wrf39_graupel_intercept", resources, parameters,
        exact_policy=True, grid_size=ny * nx, threads_per_group=32)


def submit_working_cloud(
    qc: MetalArray,
    qcten: MetalArray,
    nc: MetalArray,
    ncten: MetalArray,
    rho: MetalArray,
    dt: float,
    rc_work: MetalArray,
    nc_work_m3: MetalArray,
    l_qc_out: MetalArray,
):
    """Refresh WRF's working cloud water, number and presence flag."""
    arrays = {
        "qc": qc,
        "qcten": qcten,
        "nc": nc,
        "ncten": ncten,
        "rho": rho,
        "rc_work": rc_work,
        "nc_work_m3": nc_work_m3,
    }
    if not isinstance(qc, MetalArray) or qc.dtype != _FLOAT:
        raise TypeError("Thompson working cloud qc must be float32 MetalArray")
    if not isinstance(l_qc_out, MetalArray) or l_qc_out.dtype != np.dtype(np.int32):
        raise TypeError("Thompson working cloud flag must be int32 MetalArray")
    if l_qc_out.shape != qc.shape or not l_qc_out.flags.c_contiguous:
        raise ValueError("Thompson working cloud flag must match the state shape")
    shape, context = qc.shape, qc.buffer.context
    if l_qc_out.buffer.context is not context:
        raise ValueError("Thompson working cloud arrays must share one Metal context")
    if not shape:
        raise ValueError("Thompson working cloud fields must have at least one axis")
    count = qc.size
    if count > np.iinfo(np.int32).max:
        raise ValueError("Thompson working cloud element count must fit int32")
    if isinstance(dt, (bool, np.bool_)):
        raise TypeError("Thompson working cloud dt must be a positive scalar")
    try:
        dt32 = np.float32(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("Thompson working cloud dt must be a positive scalar") from exc
    if not np.isfinite(dt32) or dt32 <= 0.0:
        raise ValueError(
            "Thompson working cloud dt must be finite and float32 positive")
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray) or array.dtype != _FLOAT
                or array.shape != shape or not array.flags.c_contiguous):
            raise TypeError(
                f"Thompson working cloud {name} must match contiguous float32 state")
        if array.buffer.context is not context:
            raise ValueError("Thompson working cloud arrays must share one Metal context")
    all_arrays = {**arrays, "l_qc_out": l_qc_out}
    flat = {name: array.view((array.size,), byte_offset=array.byte_offset)
            for name, array in all_arrays.items()}
    for output_name in ("rc_work", "nc_work_m3", "l_qc_out"):
        output = all_arrays[output_name]
        start, end = output.byte_offset, output.byte_offset + output.nbytes
        for other_name, other in all_arrays.items():
            if other_name == output_name or other.buffer.handle != output.buffer.handle:
                continue
            other_start, other_end = other.byte_offset, other.byte_offset + other.nbytes
            if start < other_end and other_start < end:
                raise ValueError(
                    f"Thompson working cloud output {output_name} overlaps "
                    f"resource {other_name}")
    if count == 0:
        return None
    return context.submit_kernel(
        "thompson_aa_working_cloud", flat,
        {"dt": float(dt32), "n": count}, exact_policy=True,
        grid_size=count, threads_per_group=_THREADS)


def submit_init_profile(
    hgt: MetalArray,
    nwfa: MetalArray,
    nifa: MetalArray,
    nwfa2d: MetalArray,
    *,
    fill_ccn: bool,
    fill_in: bool,
):
    """Fill WRF's synthetic CCN/IN profiles for selected fields."""
    if not isinstance(hgt, MetalArray) or hgt.dtype != _FLOAT:
        raise TypeError("Thompson profile height must be a float32 MetalArray")
    if hgt.ndim != 3 or hgt.shape[0] < 2:
        raise ValueError("Thompson profile height must be (nz, ny, nx), nz >= 2")
    shape, context = hgt.shape, hgt.buffer.context
    if any(not isinstance(value, (bool, np.bool_)) for value in (fill_ccn, fill_in)):
        raise TypeError("Thompson profile fill selectors must be booleans")
    arrays = {"hgt": hgt, "nwfa": nwfa, "nifa": nifa}
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray) or array.dtype != _FLOAT
                or array.shape != shape or not array.flags.c_contiguous):
            raise TypeError(
                f"Thompson profile {name} must be contiguous float32 with shape {shape}")
        if array.buffer.context is not context:
            raise ValueError("Thompson profile arrays must share one Metal context")
    surface_shape = shape[1:]
    if (not isinstance(nwfa2d, MetalArray) or nwfa2d.dtype != _FLOAT
            or nwfa2d.shape != surface_shape or not nwfa2d.flags.c_contiguous):
        raise TypeError(
            f"Thompson profile nwfa2d must be contiguous float32 with shape {surface_shape}")
    if nwfa2d.buffer.context is not context:
        raise ValueError("Thompson profile arrays must share one Metal context")
    all_arrays = {**arrays, "nwfa2d": nwfa2d}
    flat = {name: array.view((array.size,), byte_offset=array.byte_offset)
            for name, array in all_arrays.items()}
    outputs = ("nwfa", "nifa", "nwfa2d")
    for output_name in outputs:
        output = all_arrays[output_name]
        start, end = output.byte_offset, output.byte_offset + output.nbytes
        for other_name, other in all_arrays.items():
            if other_name == output_name or other.buffer.handle != output.buffer.handle:
                continue
            other_start, other_end = other.byte_offset, other.byte_offset + other.nbytes
            if start < other_end and other_start < end:
                raise ValueError(
                    f"Thompson profile output {output_name} overlaps {other_name}")
    ncolumns = int(np.prod(surface_shape, dtype=np.int64))
    if (shape[0] > np.iinfo(np.int32).max
            or ncolumns > np.iinfo(np.int32).max):
        raise ValueError("Thompson profile dimensions must fit int32")
    if ncolumns == 0:
        return None
    return context.submit_kernel(
        "thompson_aa_init_profile", flat,
        {"fill_ccn": int(fill_ccn), "fill_in": int(fill_in),
         "nz": shape[0], "ncolumns": ncolumns},
        exact_policy=True, grid_size=ncolumns, threads_per_group=_THREADS)


def submit_profile_needs_fill(field: MetalArray, needs_fill: MetalArray):
    """Reduce one aerosol species to WRF's domain-wide profile-fill flag."""
    if (not isinstance(field, MetalArray) or field.dtype != _FLOAT
            or field.ndim != 3 or not field.flags.c_contiguous):
        raise TypeError("Thompson profile reduction needs a 3-D float32 field")
    size = int(field.size)
    if size <= 0 or size > np.iinfo(np.int32).max:
        raise ValueError("Thompson profile reduction size must fit int32")
    if (not isinstance(needs_fill, MetalArray)
            or needs_fill.dtype != np.dtype(np.int32)
            or needs_fill.shape != (1,) or not needs_fill.flags.c_contiguous):
        raise TypeError("Thompson profile flag must be one contiguous int32")
    context = field.buffer.context
    if needs_fill.buffer.context is not context:
        raise ValueError("Thompson profile reduction arrays must share context")
    if (field.buffer.handle == needs_fill.buffer.handle
            and field.byte_offset < needs_fill.byte_offset + needs_fill.nbytes
            and needs_fill.byte_offset < field.byte_offset + field.nbytes):
        raise ValueError("Thompson profile reduction output aliases its input")
    return context.submit_kernel(
        "thompson_aa_profile_needs_fill",
        {"field": field.view((size,), byte_offset=field.byte_offset),
         "needs_fill": needs_fill},
        {"size": size, "threshold": 1.0e-12},
        exact_policy=False, grid_size=1, threads_per_group=1)


def submit_profile_height(phb: MetalArray, php: MetalArray,
                          z8w: MetalArray, *, gravity: float):
    """Build geopotential interface heights without MetalArray slicing."""
    if (not isinstance(php, MetalArray) or php.ndim != 3
            or php.dtype != _FLOAT or not php.flags.c_contiguous):
        raise TypeError("Thompson profile php must be 3-D contiguous float32")
    nz1, ny, nx = map(int, php.shape)
    nz = nz1 - 1
    if nz < 2 or ny <= 0 or nx <= 0:
        raise ValueError("Thompson profile height needs nz >= 2")
    shape = (nz1, ny, nx)
    if (not isinstance(z8w, MetalArray) or z8w.dtype != _FLOAT
            or z8w.shape != shape or not z8w.flags.c_contiguous):
        raise TypeError(
            f"Thompson profile z8w must be contiguous float32 with shape {shape}")
    if (not isinstance(phb, MetalArray) or phb.dtype != _FLOAT
            or phb.shape not in ((nz1,), shape)
            or not phb.flags.c_contiguous):
        raise TypeError(
            f"Thompson profile phb must be contiguous float32 with shape "
            f"{(nz1,)} or {shape}")
    context = php.buffer.context
    if phb.buffer.context is not context or z8w.buffer.context is not context:
        raise ValueError("Thompson profile height arrays must share context")
    if (phb.buffer.handle == z8w.buffer.handle
            and phb.byte_offset < z8w.byte_offset + z8w.nbytes
            and z8w.byte_offset < phb.byte_offset + phb.nbytes):
        raise ValueError("Thompson profile z8w must not alias phb")
    if (php.buffer.handle == z8w.buffer.handle
            and php.byte_offset < z8w.byte_offset + z8w.nbytes
            and z8w.byte_offset < php.byte_offset + php.nbytes):
        raise ValueError("Thompson profile z8w must not alias php")
    size = nz1 * ny * nx
    if size > np.iinfo(np.int32).max:
        raise ValueError("Thompson profile height size must fit int32")
    gravity32 = np.float32(gravity)
    if not np.isfinite(gravity32) or gravity32 <= 0:
        raise ValueError("Thompson profile gravity must be finite and positive")
    return context.submit_kernel(
        "thompson_aa_profile_height",
        {"phb": phb.view((phb.size,), byte_offset=phb.byte_offset),
         "php": php.view((size,), byte_offset=php.byte_offset),
         "z8w": z8w.view((size,), byte_offset=z8w.byte_offset)},
        {"gravity": float(gravity32), "nz": nz, "ny": ny, "nx": nx,
         "phb_full": int(phb.ndim == 3)},
        exact_policy=True, grid_size=size, threads_per_group=_THREADS)


def submit_aerosol_surface_rain_fraction(
        rainncv: MetalArray, snowncv: MetalArray,
        graupelncv: MetalArray, sr: MetalArray):
    """Write ``(snow + graupel) / (rain + 1e-12)`` with WRF float32 order."""
    if (not isinstance(rainncv, MetalArray) or rainncv.ndim != 2
            or rainncv.dtype != _FLOAT or not rainncv.flags.c_contiguous):
        raise TypeError("MP28 rain fraction needs 2-D float32 rainncv")
    shape = rainncv.shape
    context = rainncv.buffer.context
    arrays = {"rainncv": rainncv, "snowncv": snowncv,
              "graupelncv": graupelncv, "sr": sr}
    for name, value in arrays.items():
        if (not isinstance(value, MetalArray) or value.shape != shape
                or value.dtype != _FLOAT or not value.flags.c_contiguous):
            raise TypeError(
                f"MP28 rain fraction {name} must be contiguous float32 "
                f"with shape {shape}")
        if value.buffer.context is not context:
            raise ValueError("MP28 rain fraction arrays must share context")
    if sr.buffer.handle in {rainncv.buffer.handle, snowncv.buffer.handle,
                            graupelncv.buffer.handle}:
        for name, value in arrays.items():
            if name != "sr" and _overlap(sr, value):
                raise ValueError(f"MP28 rain-fraction output aliases {name}")
    ncolumns = int(np.prod(shape, dtype=np.int64))
    if ncolumns <= 0 or ncolumns > np.iinfo(np.int32).max:
        raise ValueError("MP28 surface rain-fraction size must fit int32")
    flat = {name: value.view((ncolumns,), byte_offset=value.byte_offset)
            for name, value in arrays.items()}
    return context.submit_kernel(
        "thompson_aa_surface_rain_fraction", flat,
        {"ncolumns": ncolumns}, exact_policy=True,
        grid_size=ncolumns, threads_per_group=_THREADS)


def submit_wrf39_graupel_finalize(qg: MetalArray):
    """Apply the WRF-3.9 terminal trace-graupel cleanup."""
    if (not isinstance(qg, MetalArray) or qg.dtype != _FLOAT
            or not qg.flags.c_contiguous or not qg.shape):
        raise TypeError("WRF39 graupel must be a contiguous float32 MetalArray")
    if qg.size > np.iinfo(np.int32).max:
        raise ValueError("WRF39 graupel size must fit int32")
    if qg.size == 0:
        return None
    return qg.buffer.context.submit_kernel(
        "thompson_aa_wrf39_graupel_finalize",
        {"qg": qg.view((qg.size,), byte_offset=qg.byte_offset)},
        {"r1": 1.0e-12, "n": qg.size}, exact_policy=True,
        grid_size=qg.size, threads_per_group=_THREADS)


def submit_wrf39_start_emission(nwfa: MetalArray, nwfa2d: MetalArray,
                                *, dx: float, dy: float):
    """Initialize WRF-3.9 surface CCN emission from level zero."""
    if (not isinstance(nwfa, MetalArray) or nwfa.ndim != 3
            or nwfa.dtype != _FLOAT or not nwfa.flags.c_contiguous):
        raise TypeError("WRF39 nwfa must be contiguous float32 (nz, ny, nx)")
    nz, ny, nx = nwfa.shape
    if min(nz, ny, nx) <= 0 or ny * nx > np.iinfo(np.int32).max:
        raise ValueError("WRF39 aerosol dimensions must fit positive int32")
    context = nwfa.buffer.context
    if (not isinstance(nwfa2d, MetalArray) or nwfa2d.shape != (ny, nx)
            or nwfa2d.dtype != _FLOAT or not nwfa2d.flags.c_contiguous
            or nwfa2d.buffer.context is not context):
        raise TypeError("WRF39 nwfa2d must share context and shape (ny, nx)")
    if _overlap(nwfa, nwfa2d):
        raise ValueError("WRF39 nwfa2d must not overlap nwfa")
    dx32, dy32 = np.float32(dx), np.float32(dy)
    if not np.isfinite(dx32) or not np.isfinite(dy32):
        raise ValueError("WRF39 emission spacings must be finite")
    ncolumns = ny * nx
    return context.submit_kernel(
        "thompson_aa_wrf39_start_emission",
        {"nwfa": nwfa, "nwfa2d": nwfa2d},
        {"dx": float(dx32), "dy": float(dy32), "ncolumns": ncolumns},
        exact_policy=True, grid_size=ncolumns, threads_per_group=_THREADS)


__all__ = ["submit_aerosol_entry_cloud_number",
           "submit_aerosol_entry_snapshot", "submit_aerosol_micro_columns",
           "submit_aerosol_state_finalize_with_columns",
           "submit_surface_emission", "submit_tau1_density",
           "submit_working_number", "submit_effective_radius",
           "submit_working_cloud", "submit_init_profile",
           "submit_profile_needs_fill", "submit_profile_height",
           "submit_aerosol_surface_rain_fraction",
           "submit_wrf39_graupel_finalize", "submit_wrf39_start_emission",
           "submit_wrf39_effective_radius", "submit_wrf39_graupel_intercept"]
