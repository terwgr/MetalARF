"""Typed Metal launchers for FP32 RRTMGP cloud/profile preparation kernels."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def submit_mask_words_to_float(mask, output=None):
    """Expand the workspace's logical byte mask to an FP32 0/1 plane."""
    words = getattr(mask, "words", None)
    shape = tuple(getattr(mask, "shape", ()))
    if (not isinstance(words, MetalArray)
            or words.dtype != np.dtype(np.uint32)
            or not words.flags.c_contiguous or not shape):
        raise TypeError("RRTMGP mask conversion needs a word-backed Metal mask")
    n = int(np.prod(shape, dtype=np.int64))
    nwords = (n + 3) // 4
    if words.size < nwords:
        raise ValueError("RRTMGP packed mask does not cover its logical shape")
    context = words.buffer.context
    if output is None:
        output = MetalArray(context.buffer(max(n * 4, 4)), (n,), np.float32)
    elif (not isinstance(output, MetalArray)
          or output.dtype != np.dtype(np.float32)
          or output.shape != (n,)
          or output.buffer.context is not context):
        raise TypeError("RRTMGP float mask output has incompatible storage")
    context.submit_kernel(
        "rrtmgp_mask_words_to_float",
        {"words": words.view((nwords,), np.float32), "mask": output},
        {"n": n, "nwords": nwords}, exact_policy=False,
        grid_size=n)
    return output.view(shape)


def submit_expand_band_to_gpoint(context, values, gpoint_bands, output, *,
                                 ncol, nband, ngpt):
    ncol, nband, ngpt = map(int, (ncol, nband, ngpt))
    arrays = {
        "values": _array(values, "values", (ncol, nband)),
        "gpoint_bands": _flat(gpoint_bands, "gpoint_bands"),
        "output": _array(output, "output", (ncol, ngpt)),
    }
    if arrays["gpoint_bands"].size != ngpt:
        raise ValueError("g-point band map has an invalid length")
    return context.submit_kernel(
        "rrtmgp_expand_band_to_gpoint", arrays,
        {"ncol": ncol, "nband": nband, "ngpt": ngpt},
        exact_policy=False, grid_size=ncol * ngpt)


def _array(value, name, shape=None):
    if (not isinstance(value, MetalArray)
            or value.dtype != np.dtype(np.float32)
            or not value.flags.c_contiguous):
        raise TypeError(f"{name} must be a contiguous FP32 MetalArray")
    if shape is not None and value.shape != tuple(shape):
        raise ValueError(f"{name} shape {value.shape} != {tuple(shape)}")
    return value


def _flat(value, name):
    value = _array(value, name)
    return value.view((value.size,))


def submit_cloud_optics(context, clwp, ciwp, reliq, dgice, extliq, ssaliq,
                        asyliq, extice, ssaice, asyice, tau, ssa, asym, *,
                        ncell, nband, nliq, nice, nrgh, liq_offset, liq_step,
                        ice_offset, ice_step):
    ncell, nband = int(ncell), int(nband)
    nliq, nice, nrgh = int(nliq), int(nice), int(nrgh)
    if min(ncell, nband, nrgh) <= 0 or min(nliq, nice) < 2:
        raise ValueError("RRTMGP tables need at least two sizes and positive other dimensions")
    arrays = {
        "clwp": _flat(clwp, "clwp"), "ciwp": _flat(ciwp, "ciwp"),
        "reliq": _flat(reliq, "reliq"), "dgice": _flat(dgice, "dgice"),
        "extliq": _array(extliq, "extliq", (nliq, nband)),
        "ssaliq": _array(ssaliq, "ssaliq", (nliq, nband)),
        "asyliq": _array(asyliq, "asyliq", (nliq, nband)),
        "extice": _array(extice, "extice", (nice, nband, nrgh)),
        "ssaice": _array(ssaice, "ssaice", (nice, nband, nrgh)),
        "asyice": _array(asyice, "asyice", (nice, nband, nrgh)),
        "tau": _flat(tau, "tau"), "ssa": _flat(ssa, "ssa"),
        "asym": _flat(asym, "asym"),
    }
    if any(arrays[name].size != ncell for name in
           ("clwp", "ciwp", "reliq", "dgice")):
        raise ValueError("cloud path/radius arrays must contain ncell values")
    noptics = ncell * nband
    if any(arrays[name].size != noptics for name in ("tau", "ssa", "asym")):
        raise ValueError("cloud optics outputs must contain ncell*nband values")
    arrays["tau"] = arrays["tau"].view((ncell, nband))
    arrays["ssa"] = arrays["ssa"].view((ncell, nband))
    arrays["asym"] = arrays["asym"].view((ncell, nband))
    return context.submit_kernel(
        "rrtmgp_cloud_optics", arrays,
        {"ncell": ncell, "nband": nband, "nliq": nliq, "nice": nice,
         "nrgh": nrgh,
         "liq_offset": float(np.float32(liq_offset)),
         "liq_step": float(np.float32(liq_step)),
         "ice_offset": float(np.float32(ice_offset)),
         "ice_step": float(np.float32(ice_step))},
        exact_policy=False, grid_size=noptics)


def submit_finalize_cloud_lw(context, tau_gas, tau_cloud, ssa_cloud,
                              gpoint_bands_f32, cloud_mask_f32, tau, *,
                              ncell, ngpt, nband, have_mask=False):
    """Finalize LW optics using exact FP32 encodings of band IDs/mask bits."""
    ncell, ngpt, nband = int(ncell), int(ngpt), int(nband)
    n = ncell * ngpt
    arrays = {
        "tau_gas": _flat(tau_gas, "tau_gas"),
        "tau_cloud": _array(tau_cloud, "tau_cloud", (ncell, nband)),
        "ssa_cloud": _array(ssa_cloud, "ssa_cloud", (ncell, nband)),
        "gpoint_bands": _flat(gpoint_bands_f32, "gpoint_bands_f32"),
        "cloud_mask": _flat(cloud_mask_f32, "cloud_mask_f32"),
        "tau": _flat(tau, "tau"),
    }
    if arrays["tau_gas"].size != n or arrays["tau"].size != n:
        raise ValueError("LW gas/output arrays must contain ncell*ngpt values")
    if arrays["gpoint_bands"].size != ngpt or arrays["cloud_mask"].size != n:
        raise ValueError("LW band map or cloud mask has an invalid size")
    arrays["tau_gas"] = arrays["tau_gas"].view((n,))
    arrays["tau"] = arrays["tau"].view((n,))
    return context.submit_kernel(
        "rrtmgp_finalize_cloud_lw", arrays,
        {"n": n, "ncell": ncell, "ngpt": ngpt, "nband": nband,
         "have_mask": int(bool(have_mask))},
        exact_policy=False, grid_size=n)


def submit_finalize_cloud_sw(context, tau_gas, ssa_gas, tau_cloud,
                              ssa_cloud, g_cloud, gpoint_bands_f32,
                              cloud_mask_f32, tau, ssa, asym, *,
                              ncell, ngpt, nband, have_mask=False):
    ncell, ngpt, nband = int(ncell), int(ngpt), int(nband)
    n = ncell * ngpt
    arrays = {
        "tau_gas": _flat(tau_gas, "tau_gas"),
        "ssa_gas": _flat(ssa_gas, "ssa_gas"),
        "tau_cloud": _array(tau_cloud, "tau_cloud", (ncell, nband)),
        "ssa_cloud": _array(ssa_cloud, "ssa_cloud", (ncell, nband)),
        "g_cloud": _array(g_cloud, "g_cloud", (ncell, nband)),
        "gpoint_bands": _flat(gpoint_bands_f32, "gpoint_bands_f32"),
        "cloud_mask": _flat(cloud_mask_f32, "cloud_mask_f32"),
        "tau": _flat(tau, "tau"), "ssa": _flat(ssa, "ssa"),
        "asym": _flat(asym, "asym"),
    }
    if any(arrays[name].size != n for name in
           ("tau_gas", "ssa_gas", "cloud_mask", "tau", "ssa", "asym")):
        raise ValueError("SW gas/mask/output arrays must contain ncell*ngpt values")
    if arrays["gpoint_bands"].size != ngpt:
        raise ValueError("SW g-point band map must contain ngpt values")
    for name in ("tau_gas", "ssa_gas", "cloud_mask", "tau", "ssa", "asym"):
        arrays[name] = arrays[name].view((n,))
    return context.submit_kernel(
        "rrtmgp_finalize_cloud_sw", arrays,
        {"n": n, "ncell": ncell, "ngpt": ngpt, "nband": nband,
         "have_mask": int(bool(have_mask))},
        exact_policy=False, grid_size=n)


def submit_clear_upper_layers(context, values, outputs, *, ncol,
                              model_nlay, upper_nlay):
    if len(values) != 5 or len(outputs) != 5:
        raise ValueError("clear-upper-layers needs five inputs and outputs")
    ncol, model_nlay, upper_nlay = map(int, (ncol, model_nlay, upper_nlay))
    nlay = model_nlay + upper_nlay
    arrays = {}
    for index, (value, output) in enumerate(zip(values, outputs)):
        arrays[f"in_{index}"] = _array(value, f"input {index}",
                                        (ncol, model_nlay))
        arrays[f"out_{index}"] = _array(output, f"output {index}",
                                         (ncol, nlay))
    return context.submit_kernel(
        "rrtmgp_clear_upper_layers", arrays,
        {"ncol": ncol, "model_nlay": model_nlay, "nlay": nlay},
        exact_policy=False, grid_size=ncol * nlay)


def submit_store_model_flux_pair(context, up, dn, daylight_f32,
                                 out_up, out_dn, *, ncol, source_nlev,
                                 model_nlev, apply_daylight=False):
    ncol, source_nlev, model_nlev = map(int, (ncol, source_nlev, model_nlev))
    arrays = {
        "up": _array(up, "up", (ncol, source_nlev)),
        "dn": _array(dn, "dn", (ncol, source_nlev)),
        "daylight": _flat(daylight_f32, "daylight_f32"),
        "out_up": _array(out_up, "out_up", (ncol, model_nlev)),
        "out_dn": _array(out_dn, "out_dn", (ncol, model_nlev)),
    }
    if arrays["daylight"].size != ncol:
        raise ValueError("daylight mask must contain one value per column")
    arrays["daylight"] = arrays["daylight"].view((ncol,))
    return context.submit_kernel(
        "rrtmgp_store_model_flux_pair", arrays,
        {"ncol": ncol, "source_nlev": source_nlev,
         "model_nlev": model_nlev, "apply_daylight": int(bool(apply_daylight))},
        exact_policy=False, grid_size=ncol * model_nlev)


def submit_prepare_sw_inputs(context, albedo, solar, mu, albedo_gpt,
                             inc_gpt, mu0, *, ncol, ngpt, nlay):
    ncol, ngpt, nlay = map(int, (ncol, ngpt, nlay))
    arrays = {
        "albedo": _flat(albedo, "albedo"), "solar": _flat(solar, "solar"),
        "mu": _flat(mu, "mu"),
        "albedo_gpt": _array(albedo_gpt, "albedo_gpt", (ncol, ngpt)),
        "inc_gpt": _array(inc_gpt, "inc_gpt", (ncol, ngpt)),
        "mu0": _array(mu0, "mu0", (ncol, nlay)),
    }
    if any(arrays[name].size != ncol for name in ("albedo", "mu")):
        raise ValueError("SW albedo and mu must contain ncol values")
    if arrays["solar"].size != ngpt:
        raise ValueError("SW incident spectrum must contain ngpt values")
    for name, size in (("albedo", ncol), ("solar", ngpt), ("mu", ncol)):
        arrays[name] = arrays[name].view((size,))
    return context.submit_kernel(
        "rrtmgp_prepare_sw_inputs", arrays,
        {"ncol": ncol, "ngpt": ngpt, "nlay": nlay},
        exact_policy=False, grid_size=max(ncol * ngpt, ncol * nlay))


def submit_append_profile_fp32(context, model, upper, outputs, *, ncol,
                               model_nlay, upper_nlay):
    if len(model) != 5 or len(upper) != 5 or len(outputs) != 5:
        raise ValueError("profile append needs five model, upper, and output arrays")
    ncol, model_nlay, upper_nlay = map(int, (ncol, model_nlay, upper_nlay))
    nlay = model_nlay + upper_nlay
    model_shapes = ((ncol, model_nlay), (ncol, model_nlay + 1),
                    (ncol, model_nlay), (ncol, model_nlay + 1),
                    (ncol, model_nlay))
    upper_shapes = ((ncol, upper_nlay), (ncol, upper_nlay + 1),
                    (ncol, upper_nlay), (ncol, upper_nlay + 1),
                    (ncol, upper_nlay))
    output_shapes = ((ncol, nlay), (ncol, nlay + 1), (ncol, nlay),
                     (ncol, nlay + 1), (ncol, nlay))
    names = ("play", "plev", "tlay", "tlev", "qv")
    arrays = {}
    for group, values, shapes in (("model", model, model_shapes),
                                  ("upper", upper, upper_shapes),
                                  ("output", outputs, output_shapes)):
        for name, value, shape in zip(names, values, shapes):
            arrays[f"{group}_{name}"] = _array(
                value, f"{group}_{name}", shape)
    # MetalArray is C-contiguous; retain source stride parameters explicitly.
    upper_strides = [int(stride // 4) for value in upper
                     for stride in value.strides]
    stride_names = ("spc", "spl", "sec", "sel", "stc", "stl",
                    "sic", "sil", "sqc", "sql")
    parameters = {"ncol": ncol, "model_nlay": model_nlay,
                  "upper_nlay": upper_nlay}
    parameters.update(zip(stride_names, upper_strides))
    count = ncol * (nlay + 1)
    return context.submit_kernel(
        "rrtmgp_append_profile_fp32", arrays, parameters,
        exact_policy=False, grid_size=count)
