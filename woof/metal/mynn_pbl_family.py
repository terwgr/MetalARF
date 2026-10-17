"""Experimental source proposals for selected MYNN PBL CUDA routines.

Sixteen CUDA exports have Metal proposals; one additional proposal adapts
the internal ``mynn_mym_level2_column`` helper for column dispatch. They use
shared math or explicitly recorded native MSL boundaries.
They remain opt-in experiments: their fragment is not centrally admitted,
their arithmetic dependencies are unverified, and they are not connected to
the production MYNN callers.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .array import MetalArray


_EXPERIMENTAL_EXPORTS = {
    "mynn_driver_exchange_columns",
    "mynn_driver_diss_heat_columns",
    "mynn_mixlength_default_columns",
    "mynn_initialize_default_columns",
    "mynn_driver_prep_columns",
    "mynn_driver_surface_columns",
    "mynn_condensation_default_columns",
    "mynn_level2_pairs",
    "mynn_level2_columns",
    "mynn_pblh_scale_columns",
    "mynn_predict_default_columns",
    "mynn_turbulence_default_interfaces",
    "mynn_tendencies_columns",
    "mynn_wrapper_to_specific",
    "mynn_wrapper_from_specific",
    "mynn_dmp_mf_columns",
}

_UNSUPPORTED_EXPORTS = {}


def require_mynn_pbl_metal_supported(cuda_symbol: str) -> None:
    """Fail closed for production admission, including experimental kernels."""
    if cuda_symbol in _EXPERIMENTAL_EXPORTS:
        raise NotImplementedError(
            f"Metal {cuda_symbol} is an experimental source proposal only; "
            "the shared arithmetic is unverified and no production admission "
            "or caller integration is enabled"
        )
    reason = _UNSUPPORTED_EXPORTS.get(cuda_symbol)
    if reason is None:
        raise ValueError(f"unknown MYNN PBL CUDA export {cuda_symbol!r}")
    raise NotImplementedError(
        f"Metal {cuda_symbol} has no source proposal: {reason}; "
        "no production kernel signature is admitted"
    )


def _flat_fp32_arrays(arrays: Sequence[object], names: Sequence[str]):
    if len(arrays) != len(names):
        raise ValueError("internal MYNN resource list mismatch")
    reference_shape = None
    flattened = []
    for name, array in zip(names, arrays):
        if not isinstance(array, MetalArray):
            raise TypeError(f"{name} must be a MetalArray")
        if array.dtype != np.dtype(np.float32) or not array.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous FP32 storage")
        if reference_shape is None:
            reference_shape = array.shape
        elif array.shape != reference_shape:
            raise ValueError("MYNN resources must have identical shapes")
        flattened.append(array.view((array.size,)))
    if reference_shape is None or any(dim <= 0 for dim in reference_shape):
        raise ValueError("MYNN resources must be nonempty")
    return flattened, int(np.prod(reference_shape, dtype=np.int64))


def _reshape_fp32(array, name: str, shape):
    if not isinstance(array, MetalArray):
        raise TypeError(f"{name} must be a MetalArray")
    if array.dtype != np.dtype(np.float32) or not array.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous FP32 storage")
    size = int(np.prod(shape, dtype=np.int64))
    if array.size != size:
        raise ValueError(f"{name} does not contain the required MYNN elements")
    if array.shape not in (tuple(shape), (size,)):
        raise ValueError(f"{name} must have shape {tuple(shape)} or {(size,)}")
    return array.view(tuple(shape))


def _submit(context, name: str, resources, parameters, count: int,
            experimental: bool):
    if experimental is not True:
        raise NotImplementedError(
            f"{name} is an unverified source proposal; pass "
            "experimental=True to opt in; it still requires an explicitly "
            "assembled experimental registry because this packet is not "
            "registered in the production context"
        )
    if count > 0x7FFFFFFF:
        raise ValueError("MYNN element count exceeds the CUDA int32 ABI")
    return context.submit_kernel(
        name, resources, parameters, exact_policy=False, grid_size=count)


def submit_mynn_tendencies(context, resource_arrays, *, nz: int, ncol: int,
                           onoff, workspace=None, gsd41=False,
                           experimental=False):
    """Experimentally solve MYNN momentum, heat, and moisture tendencies.

    ``resource_arrays`` maps the input and visible output names to contiguous
    FP32 MetalArrays. Layer resources use ``(nz, ncol)``; mass-flux
    interfaces use ``(nz+1, ncol)``; column controls use ``(ncol,)``. The
    fifteen internal work fields share one caller-owned level-major
    workspace, keeping this proposal below Metal's per-kernel resource limit.
    Production callers should pass a persistent ``DomainState.scratch`` view;
    standalone callers may omit it and receive a temporary allocation.
    """
    for name, extent in (("nz", nz), ("ncol", ncol)):
        if (isinstance(extent, (bool, np.bool_))
                or not isinstance(extent, (int, np.integer))
                or extent <= 0 or extent > 0x7FFFFFFF):
            raise ValueError(f"{name} must be a positive int32")
    nz, ncol = int(nz), int(ncol)
    if nz < 2:
        raise ValueError("MYNN tendencies require nz >= 2")
    gsd41_names = ('qc_original', 'qi_original') if gsd41 else ()
    layer_names = ('dz', 'rho', 'u', 'v', 'th', 'tk', 'qv', 'p', 'exner', 'thl_in', 'sqv', 'sqc', 'sqi', 'sqs', 'ozone', 'tcd', 'qcd', 'dfm', 'dfh', 'diss_heat', 'sub_thl', 'sub_sqv', 'sub_u', 'sub_v', 'det_thl', 'det_sqv', 'det_sqc', 'det_u', 'det_v', 'du', 'dv', 'dth', 'dqv', 'dqc', 'dqi', 'dqs', 'dozone', 'thl', 'dtz', 'rhoinv', 'delp', 'a', 'b', 'c', 'd', 'cpw', 'dpw', 'sqv2', 'sqc2', 'sqi2', 'sqs2')
    scratch_names = ('dtz', 'rhoinv', 'delp', 'khdz', 'kmdz', 'a', 'b', 'c', 'd', 'cpw', 'dpw', 'sqv2', 'sqc2', 'sqi2', 'sqs2')
    visible_layer_names = tuple(n for n in layer_names if n not in scratch_names)
    interface_names = ('s_aw', 's_awthl', 's_awqv', 's_awqc', 's_awu', 's_awv', 'sd_aw', 'sd_awthl', 'sd_awqv', 'sd_awqc', 'sd_awu', 'sd_awv')
    column_names = ('delt', 'psfc', 'ust', 'wspd', 'uoce', 'voce', 'flt', 'flqv', 'flqc')
    required = (set(visible_layer_names) | set(interface_names)
                | set(column_names) | set(gsd41_names))
    # Accept legacy dictionaries that still carry caller-allocated work
    # arrays, but bind only the compact workspace below.
    accepted = required | set(scratch_names) | {'khdz', 'kmdz'}
    if not required <= set(resource_arrays) or set(resource_arrays) - accepted:
        missing = sorted(required - set(resource_arrays))
        extra = sorted(set(resource_arrays) - accepted)
        raise ValueError(f"MYNN tendency resources differ; missing={missing}, extra={extra}")
    layer_values, layer_count = _flat_fp32_arrays(
        tuple(resource_arrays[n] for n in visible_layer_names), visible_layer_names)
    if layer_count != nz * ncol:
        raise ValueError("MYNN tendency layer resources must have nz*ncol values")
    interface_shape = (nz + 1, ncol)
    interface_values = {
        n: _reshape_fp32(resource_arrays[n], n, interface_shape)
        for n in interface_names
    }
    for name in ('khdz', 'kmdz'):
        if name in resource_arrays:
            _reshape_fp32(resource_arrays[name], name, interface_shape)
    for name in scratch_names:
        if name in resource_arrays and name not in ('khdz', 'kmdz'):
            _reshape_fp32(resource_arrays[name], name, (nz, ncol))
    for name in column_names:
        _reshape_fp32(resource_arrays[name], name, (ncol,))
    resources = dict(zip(visible_layer_names,
                         (a.view((nz, ncol)) for a in layer_values)))
    resources.update(interface_values)
    resources.update({n: resource_arrays[n].view((ncol,)) for n in column_names})
    resources.update({n: _reshape_fp32(resource_arrays[n], n, (nz, ncol))
                      for n in gsd41_names})
    workspace_shape = (15, nz + 1, ncol)
    if workspace is None:
        workspace = MetalArray(
            context.buffer(15 * (nz + 1) * ncol * 4),
            (15 * (nz + 1) * ncol,), np.float32)
        workspace = workspace.view(workspace_shape)
    elif (not isinstance(workspace, MetalArray)
          or workspace.dtype != np.dtype(np.float32)
          or workspace.shape != workspace_shape
          or not workspace.flags.c_contiguous
          or workspace.buffer.context is not context):
        raise ValueError(
            "MYNN tendency workspace must be a contiguous FP32 MetalArray "
            f"with shape {workspace_shape} in the submit context")
    resources['scratch'] = workspace

    # These constants follow the MYNN Fortran parameter chain, rounding each
    # constant operation to FP32 before passing it through the typed ABI.
    rd = np.float32(287.0)
    rv = np.float32(461.6)
    cp = np.float32(np.float32(7.0) * rd / np.float32(2.0))
    xlv = np.float32(2.5e6)
    xlf = np.float32(3.50e5)
    p608 = np.float32(np.float32(rv / rd) - np.float32(1.0))
    xlvcp = np.float32(xlv / cp)
    xlscp = np.float32(np.float32(xlv + xlf) / cp)
    parameters = {
        "onoff": np.float32(onoff), "rd": rd, "p608": p608,
        "xlvcp": xlvcp, "xlscp": xlscp, "nz": nz, "ncol": ncol,
    }
    return _submit(context, "mynn_tendencies_gsd41_columns" if gsd41
                   else "mynn_tendencies_columns", resources,
                   parameters, ncol, experimental)


def submit_mynn_wrapper_to_specific(context, qv, qc, qi, qs, sqv, sqc, sqi,
                                    sqs, *, experimental=False):
    """Experimentally convert mixing ratios to specific values, in place.

    All eight arrays are same-shaped, contiguous FP32 MetalArrays. The source
    CUDA kernel applies the shared ``1 + qv`` denominator to all four species.
    """
    names = ("qv", "qc", "qi", "qs", "sqv", "sqc", "sqi", "sqs")
    arrays, count = _flat_fp32_arrays(
        (qv, qc, qi, qs, sqv, sqc, sqi, sqs), names)
    resources = dict(zip(names, arrays))
    return _submit(context, "mynn_wrapper_to_specific", resources,
                   {"count": count}, count, experimental)


def submit_mynn_wrapper_from_specific(context, sqv, rqvblten, rqcblten,
                                      rqiblten, qc_bl, qi_bl, *,
                                      experimental=False):
    """Experimentally convert five specific quantities back, in place."""
    names = ("sqv", "rqvblten", "rqcblten", "rqiblten", "qc_bl", "qi_bl")
    arrays, count = _flat_fp32_arrays(
        (sqv, rqvblten, rqcblten, rqiblten, qc_bl, qi_bl), names)
    resources = dict(zip(names, arrays))
    return _submit(context, "mynn_wrapper_from_specific", resources,
                   {"count": count}, count, experimental)


def submit_mynn_scalar_pblmix(context, qn, dz, rho, exch_h, rate, scratch,
                              *, dt, nz: int, ncol: int):
    """Solve WRF's scalar_pblmix diffusion for one number scalar.

    ``qn``/``dz``/``rho``/``exch_h``/``rate`` are contiguous FP32
    ``(nz, ncol)`` MetalArrays; ``scratch`` holds ``4 * nz * ncol`` floats.
    Writes the rate per second into ``rate``; the caller mass-couples it.
    """
    for name, extent in (("nz", nz), ("ncol", ncol)):
        if (isinstance(extent, (bool, np.bool_))
                or not isinstance(extent, (int, np.integer))
                or extent <= 0 or extent > 0x7FFFFFFF):
            raise ValueError(f"{name} must be a positive int32 extent")
    names = ("qn", "dz", "rho", "exch_h", "rate")
    arrays, count = _flat_fp32_arrays((qn, dz, rho, exch_h, rate), names)
    if count != int(nz) * int(ncol):
        raise ValueError("scalar_pblmix layers must hold nz * ncol values")
    (work,), work_count = _flat_fp32_arrays((scratch,), ("scratch",))
    if work_count < 4 * count:
        raise ValueError("scalar_pblmix scratch must hold 4 * nz * ncol values")
    resources = {name: array.view((int(nz), int(ncol)))
                 for name, array in zip(names, arrays)}
    resources["scratch"] = work[:4 * count].view((4, int(nz), int(ncol)))
    return _submit(context, "mynn_scalar_pblmix_columns", resources,
                   {"dt": np.float32(dt), "nz": int(nz), "ncol": int(ncol)},
                   int(ncol), True)


def submit_mynn_driver_exchange(context, dz, dfm, dfh, k_m, k_h, *, nz: int,
                                experimental=False):
    """Experimentally form interface exchange coefficients in level-major order.

    Flattened storage must follow the CUDA source layout: ``count / nz``
    columns per level, with adjacent levels one leading-dimension apart.
    """
    if (isinstance(nz, (bool, np.bool_))
            or not isinstance(nz, (int, np.integer)) or nz <= 0):
        raise ValueError("nz must be a positive integer")
    names = ("dz", "dfm", "dfh", "k_m", "k_h")
    arrays, count = _flat_fp32_arrays((dz, dfm, dfh, k_m, k_h), names)
    if count % int(nz) != 0:
        raise ValueError("MYNN exchange element count must be divisible by nz")
    return _submit(context, "mynn_driver_exchange_columns",
                   dict(zip(names, arrays)),
                   {"nz": int(nz), "count": count}, count, experimental)


def submit_mynn_dmp_mf(context, inputs, outputs, *, plume_workspace,
                       work_workspace, nz: int, ncol: int,
                       experimental=False):
    """Submit the default MYNN EDMF mass-flux column solve.

    Profiles are level-major ``(nz,ncol)`` arrays. The plume and work
    workspaces are per-column, caller-owned scratch so forecast storage can be
    bound to persistent domain scratch.
    """
    for name, extent in (("nz", nz), ("ncol", ncol)):
        if (isinstance(extent, (bool, np.bool_))
                or not isinstance(extent, (int, np.integer))
                or extent <= 0 or extent > 0x7FFFFFFF):
            raise ValueError(f"{name} must be a positive int32")
    nz, ncol = int(nz), int(ncol)
    if nz < 5:
        raise ValueError("MYNN mass flux requires nz >= 5")

    layer_inputs = (
        "dz", "p", "rho", "u", "v", "w", "th", "thl", "thv",
        "tk", "qt", "qv", "qc", "exner", "rstoch", "qc_bl_in",
        "cldfra_bl_in", "vt_in", "vq_in",
    )
    column_inputs = ("flt", "fltv", "flq", "pblh", "dx", "landsea",
                     "ts", "psig_shcu")
    layer_outputs = (
        "edmf_a", "edmf_w", "edmf_qt", "edmf_thl", "edmf_ent",
        "edmf_qc", "qc_bl", "cldfra_bl", "vt", "vq",
    )
    interface_outputs = ("s_aw", "s_awthl", "s_awqt", "s_awqv", "s_awqc",
                         "s_awu", "s_awv")
    scalar_outputs = ("maxwidth", "ztop", "maxmf")
    required_inputs = set(layer_inputs) | set(column_inputs) | {"zw"}
    required_outputs = (set(layer_outputs) | set(interface_outputs)
                        | set(scalar_outputs) | {"ktop"})
    if set(inputs) != required_inputs:
        raise ValueError(
            "MYNN DMP input names differ; "
            f"missing={sorted(required_inputs - set(inputs))}, "
            f"extra={sorted(set(inputs) - required_inputs)}")
    if set(outputs) != required_outputs:
        raise ValueError(
            "MYNN DMP output names differ; "
            f"missing={sorted(required_outputs - set(outputs))}, "
            f"extra={sorted(set(outputs) - required_outputs)}")

    resources = {}
    for name in layer_inputs:
        resources[name] = _reshape_fp32(inputs[name], name, (nz, ncol))
    resources["zw"] = _reshape_fp32(inputs["zw"], "zw", (nz + 1, ncol))
    for name in column_inputs:
        resources[name] = _reshape_fp32(inputs[name], name, (ncol,))
    for name in layer_outputs:
        resources[name] = _reshape_fp32(outputs[name], name, (nz, ncol))
    for name in interface_outputs:
        resources[name] = _reshape_fp32(outputs[name], name, (nz + 1, ncol))
    for name in scalar_outputs:
        resources[name] = _reshape_fp32(outputs[name], name, (ncol,))
    ktop = outputs["ktop"]
    if (not isinstance(ktop, MetalArray)
            or ktop.dtype != np.dtype(np.int32)
            or not ktop.flags.c_contiguous or ktop.size != ncol):
        raise ValueError("MYNN DMP ktop must be contiguous int32 (ncol,)")
    resources["ktop"] = ktop.view((ncol,))
    resources["plume_scratch"] = _reshape_fp32(
        plume_workspace, "plume_workspace", (ncol, 8, 8, nz + 1))
    resources["work_scratch"] = _reshape_fp32(
        work_workspace, "work_workspace", (ncol, 11, nz))
    return _submit(context, "mynn_dmp_mf_columns", resources,
                   {"nz": nz, "ncol": ncol}, ncol, experimental)


def submit_mynn_dmp_mf_gsd41(context, inputs, outputs, *, sgm,
                             plume_workspace, work_workspace,
                             nz: int, ncol: int, experimental=False):
    """Submit the GSD 4.1 ten-class mass-flux column kernel.

    The visible DMP inputs and outputs have the same names/layouts as the
    v4.6.1 path. GSD adds ``sgm`` (read-only, ``(nz,ncol)``), extends the
    plume scratch to eight vectors by ten plume classes, and uses thirteen
    work planes (three layer work arrays plus ten entrainment rows).
    """
    for name, extent in (("nz", nz), ("ncol", ncol)):
        if (isinstance(extent, (bool, np.bool_))
                or not isinstance(extent, (int, np.integer))
                or extent <= 0 or extent > 0x7FFFFFFF):
            raise ValueError(f"{name} must be a positive int32")
    nz, ncol = int(nz), int(ncol)
    if nz < 5:
        raise ValueError("MYNN GSD 4.1 mass flux requires nz >= 5")
    layer_inputs = (
        "dz", "p", "rho", "u", "v", "w", "th", "thl", "thv",
        "tk", "qt", "qv", "qc", "exner", "rstoch", "qc_bl_in",
        "cldfra_bl_in", "vt_in", "vq_in",
    )
    column_inputs = ("flt", "fltv", "flq", "pblh", "dx", "landsea",
                     "ts", "psig_shcu")
    layer_outputs = (
        "edmf_a", "edmf_w", "edmf_qt", "edmf_thl", "edmf_ent",
        "edmf_qc", "qc_bl", "cldfra_bl", "vt", "vq",
    )
    interface_outputs = ("s_aw", "s_awthl", "s_awqt", "s_awqv", "s_awqc",
                         "s_awu", "s_awv")
    scalar_outputs = ("maxwidth", "ztop", "maxmf")
    required_inputs = set(layer_inputs) | set(column_inputs) | {"zw"}
    required_outputs = (set(layer_outputs) | set(interface_outputs)
                        | set(scalar_outputs) | {"ktop"})
    if set(inputs) != required_inputs:
        raise ValueError(
            "MYNN GSD 4.1 DMP input names differ; "
            f"missing={sorted(required_inputs - set(inputs))}, "
            f"extra={sorted(set(inputs) - required_inputs)}")
    if set(outputs) != required_outputs:
        raise ValueError(
            "MYNN GSD 4.1 DMP output names differ; "
            f"missing={sorted(required_outputs - set(outputs))}, "
            f"extra={sorted(set(outputs) - required_outputs)}")
    resources = {}
    for name in layer_inputs:
        resources[name] = _reshape_fp32(inputs[name], name, (nz, ncol))
    resources["zw"] = _reshape_fp32(inputs["zw"], "zw", (nz + 1, ncol))
    resources["sgm"] = _reshape_fp32(sgm, "sgm", (nz, ncol))
    for name in column_inputs:
        resources[name] = _reshape_fp32(inputs[name], name, (ncol,))
    for name in layer_outputs:
        resources[name] = _reshape_fp32(outputs[name], name, (nz, ncol))
    for name in interface_outputs:
        resources[name] = _reshape_fp32(outputs[name], name, (nz + 1, ncol))
    for name in scalar_outputs:
        resources[name] = _reshape_fp32(outputs[name], name, (ncol,))
    ktop = outputs["ktop"]
    if (not isinstance(ktop, MetalArray)
            or ktop.dtype != np.dtype(np.int32)
            or not ktop.flags.c_contiguous or ktop.size != ncol):
        raise ValueError("MYNN GSD 4.1 DMP ktop must be contiguous int32 (ncol,)")
    resources["ktop"] = ktop.view((ncol,))
    resources["plume_scratch"] = _reshape_fp32(
        plume_workspace, "plume_workspace", (ncol, 8, 10, nz + 1))
    resources["work_scratch"] = _reshape_fp32(
        work_workspace, "work_workspace", (ncol, 13, nz))
    return _submit(context, "mynn_dmp_mf_gsd41_columns", resources,
                   {"nz": nz, "ncol": ncol}, ncol, experimental)




def submit_mynn_driver_prep(context, dz, exner, sqv, sqc, sqi, th, ust,
                            zw, qv1, sqw, thl, thetav, qke_seed, *,
                            nz: int, ncol: int, experimental=False):
    """Experimentally prepare MYNN driver columns in source memory order.

    Layer arrays must be C-contiguous ``(nz, ncol)`` MetalArrays. This has
    the same flat level-major bytes as the CUDA caller's Fortran ``(ncol,
    nz)`` arrays. ``zw`` is ``(nz+1, ncol)`` and ``ust`` is ``(ncol,)``.
    """
    for name, extent in (("nz", nz), ("ncol", ncol)):
        if (isinstance(extent, (bool, np.bool_))
                or not isinstance(extent, (int, np.integer))
                or extent <= 0 or extent > 0x7FFFFFFF):
            raise ValueError(f"{name} must be a positive int32")
    nz, ncol = int(nz), int(ncol)
    layer_shape = (nz, ncol)
    input_names = ("dz", "exner", "sqv", "sqc", "sqi", "th")
    input_arrays = (dz, exner, sqv, sqc, sqi, th)
    inputs = {name: _reshape_fp32(array, name, layer_shape)
              for name, array in zip(input_names, input_arrays)}
    ust_view = _reshape_fp32(ust, "ust", (ncol,))
    output_names = ("zw", "qv1", "sqw", "thl", "thetav", "qke_seed")
    output_arrays = (zw, qv1, sqw, thl, thetav, qke_seed)
    output_shapes = ((nz + 1, ncol),) + (layer_shape,) * 5
    outputs = {name: _reshape_fp32(array, name, shape)
               for name, array, shape in zip(output_names, output_arrays,
                                             output_shapes)}
    resources = {**inputs, "ust": ust_view, **outputs}
    return _submit(context, "mynn_driver_prep_columns", resources,
                   {"nz": nz, "ncol": ncol}, ncol, experimental)






def submit_mynn_driver_surface(context, rho, exner, dz, qv1, ust, hfx, qfx,
                               ts, flt, fltv, flq, flqv, flqc, th_sfc, rmol,
                               zet, pmz, phh, *, nz: int, ncol: int,
                               experimental=False):
    """Experimentally prepare surface fluxes and stability per column.

    Layer resources are level-major `(nz, ncol)` or flat with the same bytes;
    column resources and outputs are `(ncol,)` or flat.
    """
    for name, extent in (("nz", nz), ("ncol", ncol)):
        if (isinstance(extent, (bool, np.bool_))
                or not isinstance(extent, (int, np.integer))
                or extent <= 0 or extent > 0x7FFFFFFF):
            raise ValueError(f"{name} must be a positive int32")
    nz, ncol = int(nz), int(ncol)
    layer_names = ("rho", "exner", "dz", "qv1")
    layer_values, layer_count = _flat_fp32_arrays(
        (rho, exner, dz, qv1), layer_names)
    if layer_count != nz * ncol:
        raise ValueError("MYNN surface layer resources must have nz*ncol values")
    column_names = ("ust", "hfx", "qfx", "ts")
    column_values, column_count = _flat_fp32_arrays(
        (ust, hfx, qfx, ts), column_names)
    if column_count != ncol:
        raise ValueError("MYNN surface scalar resources must have ncol values")
    output_names = ("flt", "fltv", "flq", "flqv", "flqc", "th_sfc",
                    "rmol", "zet", "pmz", "phh")
    output_values, output_count = _flat_fp32_arrays(
        (flt, fltv, flq, flqv, flqc, th_sfc, rmol, zet, pmz, phh),
        output_names)
    if output_count != ncol:
        raise ValueError("MYNN surface outputs must have ncol values")
    resources = dict(zip(layer_names,
                         (array.view((nz, ncol)) for array in layer_values)))
    resources.update(dict(zip(column_names,
                              (array.view((ncol,)) for array in column_values))))
    resources.update(dict(zip(output_names,
                              (array.view((ncol,)) for array in output_values))))
    return _submit(context, "mynn_driver_surface_columns", resources,
                   {"nz": nz, "ncol": ncol}, ncol, experimental)


def submit_mynn_driver_surface_gsd41(context, rho, exner, dz, qv1, ust,
                                     hfx, qfx, ts, rmol_in, flt, fltv, flq,
                                     flqv, flqc, th_sfc, rmol, zet, pmz, phh,
                                     *, nz: int, ncol: int,
                                     experimental=False):
    """Submit the GSD v4.1 surface-flux and stability column variant."""
    for name, extent in (("nz", nz), ("ncol", ncol)):
        if (isinstance(extent, (bool, np.bool_))
                or not isinstance(extent, (int, np.integer))
                or extent <= 0 or extent > 0x7FFFFFFF):
            raise ValueError(f"{name} must be a positive int32")
    nz, ncol = int(nz), int(ncol)
    layer_names = ("rho", "exner", "dz", "qv1")
    layers, layer_count = _flat_fp32_arrays(
        (rho, exner, dz, qv1), layer_names)
    if layer_count != nz * ncol:
        raise ValueError("MYNN GSD surface layers need nz*ncol values")
    column_names = ("ust", "hfx", "qfx", "ts", "rmol_in")
    columns, column_count = _flat_fp32_arrays(
        (ust, hfx, qfx, ts, rmol_in), column_names)
    if column_count != ncol:
        raise ValueError("MYNN GSD surface controls need ncol values")
    output_names = ("flt", "fltv", "flq", "flqv", "flqc", "th_sfc",
                    "rmol", "zet", "pmz", "phh")
    output_values, output_count = _flat_fp32_arrays(
        (flt, fltv, flq, flqv, flqc, th_sfc, rmol, zet, pmz, phh),
        output_names)
    if output_count != ncol:
        raise ValueError("MYNN GSD surface outputs need ncol values")
    resources = dict(zip(layer_names,
                         (a.view((nz, ncol)) for a in layers)))
    resources.update(dict(zip(column_names,
        (a.view((ncol,)) for a in columns))))
    resources.update(dict(zip(output_names,
        (a.view((ncol,)) for a in output_values))))
    return _submit(context, "mynn_driver_surface_gsd41_columns", resources,
                   {"nz": nz, "ncol": ncol}, ncol, experimental)


def submit_mynn_predict_default(context, dz, rho, dfq, pdk, pdt, pdq, pdc,
                                el, s_aw, ust, pmz, phh, delt, qke_in,
                                qsq_in, qke_out, tsq_out, qsq_out, cov_out,
                                qkw, rhoinv, kqdz, kmdz, a, b, c, d, cp, dp,
                                *, nz: int, ncol: int, gsd41=False,
                                experimental=False):
    """Experimentally submit the default MYNN prediction column solve.

    Layer buffers are level-major `(nz, ncol)` or flat with identical bytes;
    the interface field `s_aw` is `(nz+1, ncol)`. Scalar column buffers are
    `(ncol,)` or flat. The solver owns its scratch arrays (`qkw`, `rhoinv`,
    `kqdz`, `kmdz`, `a`–`d`, `cp`, `dp`).
    """
    for name, extent in (("nz", nz), ("ncol", ncol)):
        if (isinstance(extent, (bool, np.bool_))
                or not isinstance(extent, (int, np.integer))
                or extent <= 0 or extent > 0x7FFFFFFF):
            raise ValueError(f"{name} must be a positive int32")
    nz, ncol = int(nz), int(ncol)
    if nz < 2:
        raise ValueError("MYNN prediction requires nz >= 2")
    layer_names = (
        "dz", "rho", "dfq", "pdk", "pdt", "pdq", "pdc", "el",
        "qke_in", "qsq_in", "qke_out", "tsq_out", "qsq_out",
        "cov_out", "qkw", "rhoinv", "kqdz", "kmdz", "a", "b", "c",
        "d", "cp", "dp",
    )
    interface_values, interface_count = _flat_fp32_arrays((s_aw,), ("s_aw",))
    if interface_count != (nz + 1) * ncol:
        raise ValueError("MYNN prediction s_aw must contain (nz+1)*ncol values")
    layer_values, layer_count = _flat_fp32_arrays(
        (dz, rho, dfq, pdk, pdt, pdq, pdc, el, qke_in, qsq_in,
         qke_out, tsq_out, qsq_out, cov_out, qkw, rhoinv, kqdz, kmdz,
         a, b, c, d, cp, dp), layer_names)
    if layer_count != nz * ncol:
        raise ValueError("MYNN prediction layer resources must have nz*ncol values")
    column_names = ("ust", "pmz", "phh", "delt")
    column_values, column_count = _flat_fp32_arrays(
        (ust, pmz, phh, delt), column_names)
    if column_count != ncol:
        raise ValueError("MYNN prediction column resources must have ncol values")
    resources = dict(zip(layer_names,
                         (array.view((nz, ncol)) for array in layer_values)))
    resources["s_aw"] = interface_values[0].view((nz + 1, ncol))
    resources.update(dict(zip(column_names,
                              (array.view((ncol,)) for array in column_values))))
    return _submit(context, "mynn_predict_gsd41_columns" if gsd41
                   else "mynn_predict_default_columns", resources,
                   {"nz": nz, "ncol": ncol}, ncol, experimental)


def submit_mynn_level2_pairs(context, dz_a, u_a, v_a, thl_a, qw_a, vt_a,
                             vq_a, dz_prev_a, u_prev_a, v_prev_a,
                             thl_prev_a, qw_prev_a, vt_prev_a, vq_prev_a,
                             dtl_o, dqw_o, dtv_o, gm_o, gh_o, sm_o, sh_o, *,
                             experimental=False):
    """Experimentally evaluate the source's adjacent-level closure pairs.

    Every input and output is a same-shaped contiguous FP32 MetalArray. The
    implementation uses shared RN binary32 operators, including soft32 sqrt;
    this consumer and shared primitive remain unverified.
    """
    names = (
        "dz_a", "u_a", "v_a", "thl_a", "qw_a", "vt_a", "vq_a",
        "dz_prev_a", "u_prev_a", "v_prev_a", "thl_prev_a", "qw_prev_a",
        "vt_prev_a", "vq_prev_a", "dtl_o", "dqw_o", "dtv_o", "gm_o",
        "gh_o", "sm_o", "sh_o",
    )
    arrays, count = _flat_fp32_arrays(
        (dz_a, u_a, v_a, thl_a, qw_a, vt_a, vq_a, dz_prev_a, u_prev_a,
         v_prev_a, thl_prev_a, qw_prev_a, vt_prev_a, vq_prev_a, dtl_o,
         dqw_o, dtv_o, gm_o, gh_o, sm_o, sh_o), names)
    return _submit(context, "mynn_level2_pairs", dict(zip(names, arrays)),
                   {"n": count}, count, experimental)


def submit_mynn_level2_columns(context, dz, u, v, thl, qw, vt, vq, dtl, dqw,
                               dtv, gm, gh, sm, sh, *, nz: int, ncol: int,
                               experimental=False):
    """Experimentally evaluate MYNN level-2 closure column by column.

    Layer arrays use level-major ``(nz, ncol)`` storage. The five zero-based
    tendency/gradient outputs are initialized to zero at level zero; ``sm``
    and ``sh`` retain their caller-provided surface values there, matching
    ``mym_initialize`` before its level-2 call. The implementation depends on
    shared RN binary32 arithmetic and soft32 sqrt and remains unverified.
    """
    for name, extent in (("nz", nz), ("ncol", ncol)):
        if (isinstance(extent, (bool, np.bool_))
                or not isinstance(extent, (int, np.integer))
                or extent <= 0 or extent > 0x7FFFFFFF):
            raise ValueError(f"{name} must be a positive int32")
    nz, ncol = int(nz), int(ncol)
    if nz < 2:
        raise ValueError("MYNN level-2 closure requires nz >= 2")
    names = ("dz", "u", "v", "thl", "qw", "vt", "vq", "dtl", "dqw", "dtv",
             "gm", "gh", "sm", "sh")
    arrays = tuple(_reshape_fp32(array, name, (nz, ncol))
                   for array, name in zip(
                       (dz, u, v, thl, qw, vt, vq, dtl, dqw, dtv, gm, gh,
                        sm, sh),
                       names))
    return _submit(context, "mynn_level2_columns", dict(zip(names, arrays)),
                   {"nz": nz, "ncol": ncol}, ncol, experimental)


def submit_mynn_pblh_scale(context, thetav, qke, zw, dz, landsea, dx, zi,
                           kzi, psig_bl, psig_shcu, *, nz: int, ncol: int,
                           gsd41=False, experimental=False):
    """Experimentally compute the source's PBL height and scale columns.

    Layer arrays are level-major `(nz, ncol)`; `zw` is `(nz+1, ncol)`.
    Column arrays are `(ncol,)`. The int32 `kzi` output is checked separately.
    This source proposal uses shared RN arithmetic and glibc powf, with native
    MSL tanh for CUDA tanhf; math behavior remains unverified.
    """
    for name, extent in (("nz", nz), ("ncol", ncol)):
        if (isinstance(extent, (bool, np.bool_))
                or not isinstance(extent, (int, np.integer))
                or extent <= 0 or extent > 0x7FFFFFFF):
            raise ValueError(f"{name} must be a positive int32")
    nz, ncol = int(nz), int(ncol)
    if nz < 2:
        raise ValueError("MYNN PBLH scaling requires nz >= 2")
    layer_shape = (nz, ncol)
    layer_names = ("thetav", "qke", "dz")
    layer_values, layer_count = _flat_fp32_arrays(
        (thetav, qke, dz), layer_names)
    if layer_count != nz * ncol:
        raise ValueError("MYNN PBLH layer resources must have nz*ncol values")
    zw_values, zw_count = _flat_fp32_arrays((zw,), ("zw",))
    if zw_count != (nz + 1) * ncol:
        raise ValueError("MYNN PBLH zw must have (nz+1)*ncol values")
    zw_view = _reshape_fp32(zw_values[0], "zw", (nz + 1, ncol))

    column_names = ("landsea", "dx", "zi", "psig_bl", "psig_shcu")
    column_values, column_count = _flat_fp32_arrays(
        (landsea, dx, zi, psig_bl, psig_shcu), column_names)
    if column_count != ncol:
        raise ValueError("MYNN PBLH column resources must have ncol values")
    if not isinstance(kzi, MetalArray) or kzi.dtype != np.dtype(np.int32):
        raise TypeError("MYNN PBLH kzi must be an int32 MetalArray")
    if (not kzi.flags.c_contiguous or kzi.size != ncol
            or kzi.shape not in ((ncol,), (ncol, 1), (1, ncol))):
        raise ValueError("MYNN PBLH kzi must be contiguous with ncol values")

    resources = dict(zip(layer_names,
                         (array.view(layer_shape) for array in layer_values)))
    resources["zw"] = zw_view
    resources.update(dict(zip(column_names,
                              (array.view((ncol,)) for array in column_values))))
    resources["kzi"] = kzi.view((ncol,))
    return _submit(context, "mynn_pblh_scale_gsd41_columns" if gsd41
                   else "mynn_pblh_scale_columns", resources,
                   {"nz": nz, "ncol": ncol}, ncol, experimental)


def submit_mynn_turbulence_default(context, dz, u, v, cldfra, edmf_w,
                                   edmf_a, tkeprodtd, el, qkw, dtl, dqw,
                                   gm, gh, sm, sh, dfm, dfh, dfq, pdk, pdt,
                                   pdq, pdc, tcd, qcd, *, nz: int,
                                   gsd41=False, experimental=False):
    """Experimentally evaluate source-ordered interface turbulence updates.

    Resources use contiguous level-major storage. `nz` must divide the shared
    flattened interface extent, matching the CUDA kernel's ld calculation.
    The source's mixed FP32/binary64 arithmetic uses shared bit operations;
    exceptional-value and consumer behavior remain unverified.
    """
    if (isinstance(nz, (bool, np.bool_))
            or not isinstance(nz, (int, np.integer))
            or nz <= 0 or nz > 0x7FFFFFFF):
        raise ValueError("nz must be a positive int32")
    names = (
        "dz", "u", "v", "cldfra", "edmf_w", "edmf_a", "tkeprodtd",
        "el", "qkw", "dtl", "dqw", "gm", "gh", "sm", "sh", "dfm",
        "dfh", "dfq", "pdk", "pdt", "pdq", "pdc", "tcd", "qcd",
    )
    arrays, count = _flat_fp32_arrays(
        (dz, u, v, cldfra, edmf_w, edmf_a, tkeprodtd, el, qkw, dtl, dqw,
         gm, gh, sm, sh, dfm, dfh, dfq, pdk, pdt, pdq, pdc, tcd, qcd),
        names)
    nz = int(nz)
    if count % nz != 0 or count < nz:
        raise ValueError("MYNN turbulence count must be a positive multiple of nz")
    resources = dict(zip(names, arrays))
    resources["sm"] = resources["sm"].view((count,))
    resources["sh"] = resources["sh"].view((count,))
    return _submit(context, "mynn_turbulence_gsd41_interfaces" if gsd41
                   else "mynn_turbulence_default_interfaces", resources,
                   {"nz": nz, "count": count}, count, experimental)


def submit_mynn_condensation_default(context, dz, th, qw, qc, qi, qs, p,
                                     exner, qsq, rstoch, xland, pblh,
                                     sgm_in, qc_bl, qi_bl, cldfra, vt, vq,
                                     sgm, *, nz: int, ncol: int,
                                     experimental=False):
    """Experimentally submit the default cloud-PDF condensation columns.

    Layer resources are level-major `(nz, ncol)`; column controls are
    `(ncol,)`. Shared soft math handles sqrt, exp, pow, and atan. The two
    source log10f calls use native MSL log10 and remain unverified.
    """
    for name, extent in (("nz", nz), ("ncol", ncol)):
        if (isinstance(extent, (bool, np.bool_))
                or not isinstance(extent, (int, np.integer))
                or extent <= 0 or extent > 0x7FFFFFFF):
            raise ValueError(f"{name} must be a positive int32")
    nz, ncol = int(nz), int(ncol)
    if nz < 2:
        raise ValueError("MYNN condensation requires nz >= 2")
    layer_names = (
        "dz", "th", "qw", "qc", "qi", "qs", "p", "exner", "qsq",
        "rstoch", "sgm_in",
    )
    layer_values, layer_count = _flat_fp32_arrays(
        (dz, th, qw, qc, qi, qs, p, exner, qsq, rstoch, sgm_in),
        layer_names)
    if layer_count != nz * ncol:
        raise ValueError("MYNN condensation layer resources must have nz*ncol values")
    output_names = ("qc_bl", "qi_bl", "cldfra", "vt", "vq", "sgm")
    output_values, output_count = _flat_fp32_arrays(
        (qc_bl, qi_bl, cldfra, vt, vq, sgm), output_names)
    if output_count != nz * ncol:
        raise ValueError("MYNN condensation outputs must have nz*ncol values")
    column_names = ("xland", "pblh")
    column_values, column_count = _flat_fp32_arrays(
        (xland, pblh), column_names)
    if column_count != ncol:
        raise ValueError("MYNN condensation controls must have ncol values")
    layer_shape = (nz, ncol)
    resources = dict(zip(layer_names,
                         (a.view(layer_shape) for a in layer_values)))
    resources.update(dict(zip(output_names,
                              (a.view(layer_shape) for a in output_values))))
    resources.update(dict(zip(column_names,
                              (a.view((ncol,)) for a in column_values))))
    return _submit(context, "mynn_condensation_default_columns", resources,
                   {"nz": nz, "ncol": ncol}, ncol, experimental)


def submit_mynn_condensation_gsd41(context, dz, zw, th, thl, qw, p, exner,
                                   el, dx, pblh, hfx, sgm_in, qc_bl, qi_bl,
                                   cldfra, vt, vq, sgm, work, *, nz: int,
                                   ncol: int, experimental=False):
    """Submit GSD MYNN v4.1 CASE(2) condensation columns.

    Layer buffers use level-major ``(nz,ncol)`` storage; ``zw`` uses
    ``(nz+1,ncol)``. Five work planes (q1, rh, a, b, cloud fraction) share a
    ``(5*nz,ncol)`` buffer. ``sgm_in`` and output ``sgm`` are distinct because
    the top level is copied from the input after the lower levels are filled.
    """
    for name, extent in (("nz", nz), ("ncol", ncol)):
        if (isinstance(extent, (bool, np.bool_))
                or not isinstance(extent, (int, np.integer))
                or extent <= 0 or extent > 0x7FFFFFFF):
            raise ValueError(f"{name} must be a positive int32")
    nz, ncol = int(nz), int(ncol)
    if nz < 4:
        raise ValueError("MYNN GSD 4.1 condensation requires nz >= 4")
    layer_names = ("dz", "th", "thl", "qw", "p", "exner", "el",
                   "sgm_in")
    layer_values, layer_count = _flat_fp32_arrays(
        (dz, th, thl, qw, p, exner, el, sgm_in), layer_names)
    if layer_count != nz * ncol:
        raise ValueError("MYNN GSD 4.1 condensation layers need nz*ncol values")
    resources = dict(zip(layer_names,
                         (a.view((nz, ncol)) for a in layer_values)))
    zw_array = _reshape_fp32(zw, "zw", (nz + 1, ncol))
    resources["zw"] = zw_array
    column_names = ("dx", "pblh", "hfx")
    column_values, column_count = _flat_fp32_arrays(
        (dx, pblh, hfx), column_names)
    if column_count != ncol:
        raise ValueError("MYNN GSD 4.1 controls need ncol values")
    resources.update(dict(zip(column_names,
        (a.view((ncol,)) for a in column_values))))
    output_names = ("qc_bl", "qi_bl", "cldfra", "vt", "vq", "sgm")
    output_values, output_count = _flat_fp32_arrays(
        (qc_bl, qi_bl, cldfra, vt, vq, sgm), output_names)
    if output_count != nz * ncol:
        raise ValueError("MYNN GSD 4.1 outputs need nz*ncol values")
    resources.update(dict(zip(output_names,
        (a.view((nz, ncol)) for a in output_values))))
    resources["work"] = _reshape_fp32(work, "gsd41_condensation_work",
                                       (ncol, 5, nz))
    return _submit(context, "mynn_condensation_gsd41_columns", resources,
                   {"nz": nz, "ncol": ncol}, ncol, experimental)


def submit_mynn_gsd41_thvl(context, th, exner, sqv, sqc, sqi, tk, qc_bl,
                           cldfra_bl, thvl, *, experimental=False):
    """Theta-v of the liquid-water theta that GSD 4.1 feeds to GET_PBLH."""
    names = ("th", "exner", "sqv", "sqc", "sqi", "tk", "qc_bl",
             "cldfra_bl", "thvl")
    arrays, count = _flat_fp32_arrays(
        (th, exner, sqv, sqc, sqi, tk, qc_bl, cldfra_bl, thvl), names)
    return _submit(context, "mynn_gsd41_thvl_columns",
                   dict(zip(names, arrays)), {"count": count}, count,
                   experimental)


def submit_mynn_gsd41_cloud_decay(context, cldfra_bl, qc_bl, cldfra_old, u,
                                  v, dx, delt, *, nz: int, ncol: int,
                                  experimental=False):
    """Limit how fast the carried GSD 4.1 subgrid cloud may decay."""
    layer_names = ("cldfra_bl", "qc_bl", "cldfra_old", "u", "v")
    resources = {name: _reshape_fp32(array, name, (nz, ncol))
                 for name, array in zip(
                     layer_names, (cldfra_bl, qc_bl, cldfra_old, u, v))}
    resources.update({name: _reshape_fp32(array, name, (ncol,))
                      for name, array in (("dx", dx), ("delt", delt))})
    return _submit(context, "mynn_gsd41_cloud_decay", resources,
                   {"nz": nz, "ncol": ncol}, nz * ncol, experimental)


def submit_mynn_driver_diss_heat(context, el, qke, p, diss_heat, *,
                                 nz: int, ncol: int, gsd41=False,
                                 experimental=False):
    """Experimentally compute source dissipation heating by column.

    Inputs and output are contiguous, level-major FP32 `(nz, ncol)` buffers.
    Shared glibc powf/expf stand in for source double pow/exp rounded to FP32;
    their equivalence for this consumer remains unverified.
    """
    for name, extent in (("nz", nz), ("ncol", ncol)):
        if (isinstance(extent, (bool, np.bool_))
                or not isinstance(extent, (int, np.integer))
                or extent <= 0 or extent > 0x7FFFFFFF):
            raise ValueError(f"{name} must be a positive int32")
    nz, ncol = int(nz), int(ncol)
    if nz < 1:
        raise ValueError("MYNN dissipation heating requires nz >= 1")
    names = ("el", "qke", "p", "diss_heat")
    arrays, count = _flat_fp32_arrays((el, qke, p, diss_heat), names)
    if count != nz * ncol:
        raise ValueError("MYNN dissipation resources must have nz*ncol values")
    layer_shape = (nz, ncol)
    resources = dict(zip(names,
                         (array.view(layer_shape) for array in arrays)))
    return _submit(context, "mynn_driver_diss_heat_gsd41_columns" if gsd41
                   else "mynn_driver_diss_heat_columns", resources,
                   {"nz": nz, "ncol": ncol}, ncol, experimental)


def submit_mynn_mixlength_default(context, dz, zw, u, v, qke, dtv, theta,
                                  edmf_w, edmf_a, rmo, fltv, zi, psig_bl,
                                  el, qkw, qtke, thetaw, elblavg, dlu, dld,
                                  *, nz: int, ncol: int, mixlength: int = 1,
                                  experimental=False):
    """Experimentally compute MYNN mixing length and its column scratch.

    Level resources use C-contiguous level-major `(nz, ncol)` storage; `zw`
    uses `(nz+1, ncol)` and controls use `(ncol,)`. The source fdlibm tanhf
    path is transcribed; shared pow/sqrt and RN arithmetic remain unverified.
    """
    if type(mixlength) is not int or mixlength not in (1, 2):
        raise ValueError("MYNN mixing length requires mixlength=1 or 2")
    for name, extent in (("nz", nz), ("ncol", ncol)):
        if (isinstance(extent, (bool, np.bool_))
                or not isinstance(extent, (int, np.integer))
                or extent <= 0 or extent > 0x7FFFFFFF):
            raise ValueError(f"{name} must be a positive int32")
    nz, ncol = int(nz), int(ncol)
    if nz < 2:
        raise ValueError("MYNN mixing length requires nz >= 2")
    level_names = ("dz", "u", "v", "qke", "dtv", "theta", "edmf_w", "edmf_a")
    level_values, level_count = _flat_fp32_arrays(
        (dz, u, v, qke, dtv, theta, edmf_w, edmf_a), level_names)
    if level_count != nz * ncol:
        raise ValueError("MYNN mixing-length layers must have nz*ncol values")
    zw_values, zw_count = _flat_fp32_arrays((zw,), ("zw",))
    if zw_count != (nz + 1) * ncol:
        raise ValueError("MYNN mixing-length zw must have (nz+1)*ncol values")
    column_names = ("rmo", "fltv", "zi", "psig_bl")
    column_values, column_count = _flat_fp32_arrays(
        (rmo, fltv, zi, psig_bl), column_names)
    if column_count != ncol:
        raise ValueError("MYNN mixing-length controls must have ncol values")
    output_names = ("el", "qkw", "qtke", "thetaw", "elblavg", "dlu", "dld")
    output_values, output_count = _flat_fp32_arrays(
        (el, qkw, qtke, thetaw, elblavg, dlu, dld), output_names)
    if output_count != nz * ncol:
        raise ValueError("MYNN mixing-length outputs must have nz*ncol values")
    layer_shape = (nz, ncol)
    resources = dict(zip(level_names,
                         (a.view(layer_shape) for a in level_values)))
    resources["zw"] = _reshape_fp32(zw_values[0], "zw", (nz + 1, ncol))
    resources.update(dict(zip(column_names,
                              (a.view((ncol,)) for a in column_values))))
    resources.update(dict(zip(output_names,
                              (a.view(layer_shape) for a in output_values))))
    return _submit(context, "mynn_mixlength_default_columns", resources,
                   {"nz": nz, "ncol": ncol, "mixlength": mixlength}, ncol, experimental)


def submit_mynn_mixlength_gsd41(context, dz, zw, qke, dtv, theta,
                                edmf_w, edmf_a, cldfra, vt, vq, flt, flq,
                                rmo, zi, psig_bl, el, qkw, qtke, thetaw,
                                *, nz: int, ncol: int, experimental=False):
    """Run the WRF v4.1 GSD ``mym_length`` CASE(2) column kernel."""
    for name, extent in (("nz", nz), ("ncol", ncol)):
        if (isinstance(extent, (bool, np.bool_))
                or not isinstance(extent, (int, np.integer))
                or extent <= 0 or extent > 0x7FFFFFFF):
            raise ValueError(f"{name} must be a positive int32")
    nz, ncol = int(nz), int(ncol)
    if nz < 2:
        raise ValueError("MYNN GSD 4.1 mixing length requires nz >= 2")
    level_names = ("dz", "qke", "dtv", "theta", "edmf_w", "edmf_a",
                   "cldfra", "vt", "vq")
    level_values = (dz, qke, dtv, theta, edmf_w, edmf_a, cldfra, vt, vq)
    levels = tuple(_reshape_fp32(value, name, (nz, ncol))
                   for name, value in zip(level_names, level_values))
    zw_values, count = _flat_fp32_arrays((zw,), ("zw",))
    if count != (nz + 1) * ncol:
        raise ValueError("MYNN GSD 4.1 zw must have (nz+1)*ncol values")
    control_names = ("flt", "flq", "rmo", "zi", "psig_bl")
    control_values, count = _flat_fp32_arrays(
        (flt, flq, rmo, zi, psig_bl), control_names)
    if count != ncol:
        raise ValueError("MYNN GSD 4.1 controls must have ncol values")
    output_names = ("el", "qkw", "qtke", "thetaw")
    output_values = tuple(_reshape_fp32(value, name, (nz, ncol))
                          for name, value in zip(
                              output_names, (el, qkw, qtke, thetaw)))
    resources = dict(zip(level_names, levels))
    resources["zw"] = zw_values[0].view((nz + 1, ncol))
    resources.update(dict(zip(control_names,
                              (a.view((ncol,)) for a in control_values))))
    resources.update(dict(zip(output_names, output_values)))
    return _submit(context, "mynn_mixlength_gsd41_columns", resources,
                   {"nz": nz, "ncol": ncol}, ncol, experimental)


def submit_mynn_initialize_default(context, dz, zw, u, v, thl, qw, theta,
                                   edmf_w, edmf_a, sm_in, sh_in, qke_in,
                                   rmo, ust, zi, psig_bl, el, qke, tsq, qsq,
                                   cov, sm, sh, scratch, *, nz: int,
                                   ncol: int, initialize_qke: bool,
                                   mixlength: int = 1,
                                   _kernel_name: str = "mynn_initialize_default_columns",
                                   experimental=False):
    """Run the gated cold-start proposal with level-major field storage.

    Every profile uses ``(nz,ncol)`` storage; scalar controls use ``(ncol,)``.
    The submitter-owned ABI requires a contiguous FP32 scratch buffer shaped
    ``(15*nz,ncol)``. ``initialize_qke`` controls the source's optional QKE
    initialization path. Shared portable-libm pow is only a candidate for the
    CUDA binary64 pow helper; this proposal is not admitted to production.
    """
    if type(mixlength) is not int or mixlength not in (1, 2):
        raise ValueError("MYNN initialization requires mixlength=1 or 2")
    for name, extent in (("nz", nz), ("ncol", ncol)):
        if (isinstance(extent, (bool, np.bool_))
                or not isinstance(extent, (int, np.integer))
                or extent <= 0 or extent > 0x7FFFFFFF):
            raise ValueError(f"{name} must be a positive int32")
    if not isinstance(initialize_qke, (bool, np.bool_)):
        raise TypeError("initialize_qke must be boolean")
    nz, ncol = int(nz), int(ncol)
    if nz < 2:
        raise ValueError("MYNN initialization requires nz >= 2")
    input_layers = ("dz", "u", "v", "thl", "qw", "theta", "edmf_w",
                    "edmf_a", "sm_in", "sh_in", "qke_in")
    inputs = (dz, u, v, thl, qw, theta, edmf_w, edmf_a, sm_in, sh_in, qke_in)
    in_arrays = tuple(_reshape_fp32(a, name, (nz, ncol))
                      for a, name in zip(inputs, input_layers))
    zw_values, zw_count = _flat_fp32_arrays((zw,), ("zw",))
    if zw_count != (nz + 1) * ncol:
        raise ValueError("MYNN initialization zw must have (nz+1)*ncol values")
    controls = ("rmo", "ust", "zi", "psig_bl")
    control_arrays, control_count = _flat_fp32_arrays(
        (rmo, ust, zi, psig_bl), controls)
    if control_count != ncol:
        raise ValueError("MYNN initialization controls must have ncol values")
    output_names = ("el", "qke", "tsq", "qsq", "cov", "sm", "sh")
    output_arrays = tuple(_reshape_fp32(a, name, (nz, ncol))
                          for a, name in zip(
                              (el, qke, tsq, qsq, cov, sm, sh), output_names))
    scratch_values, scratch_count = _flat_fp32_arrays((scratch,), ("scratch",))
    if scratch_count != 15 * nz * ncol:
        raise ValueError("MYNN initialization scratch must contain 15*nz*ncol values")
    resources = dict(zip(input_layers,
                         (a.view((nz, ncol)) for a in in_arrays)))
    resources["zw"] = zw_values[0].view((nz + 1, ncol))
    resources.update(dict(zip(controls,
                              (a.view((ncol,)) for a in control_arrays))))
    resources.update(dict(zip(output_names,
                              (a.view((nz, ncol)) for a in output_arrays))))
    resources["scratch"] = scratch_values[0].view((15 * nz * ncol,))
    parameters = {"nz": nz, "ncol": ncol,
                  "initialize_qke": int(initialize_qke)}
    if _kernel_name == "mynn_initialize_default_columns":
        parameters["mixlength"] = mixlength
    return _submit(context, _kernel_name, resources, parameters, ncol, experimental)


def submit_mynn_initialize_gsd41(context, *args, **kwargs):
    """Run initialization with the separate WRF v4.1 GSD length kernel."""
    if "_kernel_name" in kwargs:
        raise TypeError("_kernel_name is reserved by the MYNN submitter")
    kwargs["_kernel_name"] = "mynn_initialize_gsd41_columns"
    return submit_mynn_initialize_default(context, *args, **kwargs)
