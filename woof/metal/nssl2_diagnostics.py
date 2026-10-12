"""Typed Metal submitters for NSSL radar and effective-radius diagnostics."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray


_TPB = 256


def _fields(context, fields):
    first = next(iter(fields.values()))
    if not isinstance(first, MetalArray) or len(first.shape) != 3:
        raise ValueError("NSSL diagnostic fields must be 3-D MetalArrays")
    shape = first.shape
    for name, value in fields.items():
        if not isinstance(value, MetalArray):
            raise TypeError(f"{name} must be a MetalArray")
        if value.shape != shape or value.dtype != np.dtype(np.float32):
            raise ValueError(f"{name} must be FP32 with shape {shape}")
        if value.buffer.context is not context:
            raise ValueError("NSSL diagnostic resources must share one context")
    return shape, int(np.prod(shape, dtype=np.int64))


def _output(context, output, name, shape):
    if not isinstance(output, MetalArray):
        raise TypeError(f"{name} must be a MetalArray")
    if output.shape != shape or output.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must be FP32 with shape {shape}")
    if output.buffer.context is not context:
        raise ValueError("NSSL diagnostic resources must share one context")


def submit_nssl2_radardd02(
        context, air_density, temperature_k, qr, qi, qs, qg, qh,
        qnr, qni, qns, qng, qnh, qvolg, qvolh, refl_10cm, *,
        concentration_space=False, output_due=True):
    if type(output_due) is not bool or type(concentration_space) is not bool:
        raise TypeError("NSSL diagnostic switches must be bool")
    if not output_due:
        return False
    fields = {name: value for name, value in locals().copy().items()
              if name in {"air_density", "temperature_k", "qr", "qi", "qs",
                          "qg", "qh", "qnr", "qni", "qns", "qng",
                          "qnh", "qvolg", "qvolh"}}
    shape, n = _fields(context, fields)
    nz, ny, nx = shape
    _output(context, refl_10cm, "refl_10cm", shape)
    return context.submit_kernel(
        "nssl2_radardd02", {**fields, "refl_10cm": refl_10cm},
        {"concentration_space": int(concentration_space), "n": n,
         "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=n, threads_per_group=_TPB)


def submit_nssl2_effective_radius_concentration(
        context, air_density, qc, cloud_number, qi, ice_number, qs,
        snow_number, re_cloud, re_ice, re_snow, *,
        registry_unit_numbers=False):
    if type(registry_unit_numbers) is not bool:
        raise TypeError("registry_unit_numbers must be bool")
    fields = {"air_density": air_density, "qc": qc,
              "cloud_number": cloud_number, "qi": qi,
              "ice_number": ice_number, "qs": qs,
              "snow_number": snow_number}
    shape, n = _fields(context, fields)
    nz, ny, nx = shape
    for name, output in (("re_cloud", re_cloud), ("re_ice", re_ice),
                         ("re_snow", re_snow)):
        _output(context, output, name, shape)
    parameters = {"n": n, "nz": nz, "ny": ny, "nx": nx}
    kernel_name = "nssl2_effective_radius_concentration"
    if registry_unit_numbers:
        kernel_name = "nssl2_effective_radius_registry_units"
        parameters["registry_unit_numbers"] = 1
    return context.submit_kernel(
        kernel_name,
        {**fields, "re_cloud": re_cloud, "re_ice": re_ice,
         "re_snow": re_snow},
        parameters, exact_policy=False,
        grid_size=n, threads_per_group=_TPB)


__all__ = ["submit_nssl2_effective_radius_concentration", "submit_nssl2_radardd02"]
