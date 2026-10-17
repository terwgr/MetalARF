"""Metal launches for the existing mixed-microphysics nest edge contracts."""

from __future__ import annotations

import math

import numpy as np

from .array import MetalArray, MetalArrayModule
from .generated import kernel_specs


_SYMBOLS = {"mp8_to_mp18_mass_diagnosed_field", "microphysics_edge_field"}


def my2_constants(context):
    """Keep the scheme's constant vector on the context that owns it."""
    constants = getattr(context, "_nest_microphysics_my2_constants", None)
    if constants is None:
        from woof.core.milbrandt2_constants import ck_vector
        constants = MetalArrayModule(context).asarray(ck_vector())
        context._nest_microphysics_my2_constants = constants
    return constants


def submit_edge(symbol, arguments):
    """Bind the shared launch tuple without copying any state to the host."""
    if symbol not in _SYMBOLS:
        raise ValueError(f"unknown nest microphysics kernel {symbol!r}")
    name = "nest_" + symbol
    prefix = name.upper()
    resource_specs = getattr(kernel_specs, prefix + "_RESOURCES")
    parameter_specs = getattr(kernel_specs, prefix + "_PARAM_FIELDS")
    if len(arguments) != len(resource_specs) + len(parameter_specs):
        raise TypeError("nest microphysics launch arguments differ from the registered ABI")
    arrays = arguments[:len(resource_specs)]
    parameters = dict(zip((field for field, _ in parameter_specs),
                          arguments[len(resource_specs):]))
    out = arrays[-1]
    if not isinstance(out, MetalArray) or out.ndim != 3:
        raise TypeError("nest microphysics output must be a 3-D MetalArray")
    shape = tuple(int(parameters[name]) for name in ("nz", "ny", "nx"))
    count = math.prod(shape)
    if out.shape != shape or min(shape) < 1 or count > np.iinfo(np.int32).max:
        raise ValueError("nest microphysics output extent must fit the int32 kernel")
    context = out.buffer.context
    resources = {}
    for (field, _, _), array in zip(resource_specs, arrays):
        if (not isinstance(array, MetalArray) or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous or array.buffer.context is not context):
            raise TypeError(f"nest microphysics {field} must be contiguous FP32 on the output context")
        if field != "out" and array.buffer is out.buffer:
            if (array.byte_offset < out.byte_offset + out.nbytes
                    and out.byte_offset < array.byte_offset + array.nbytes):
                raise ValueError("nest microphysics output must not overlap its inputs")
        if field in ("c1h", "c2h"):
            expected = (shape[0],)
        elif field in ("mub2d", "mup"):
            expected = shape[1:]
        elif field == "my2_ck":
            if parameters["target_mp"] == 9 and array.size < 141:
                raise ValueError("Milbrandt-Yau edge constants are incomplete")
            expected = array.shape
        elif field == "target_thb" and parameters["target_mp"] == 9 and parameters["thb_columnar"]:
            expected = (shape[0],)
        else:
            expected = shape
        if array.shape != expected:
            raise ValueError(f"nest microphysics {field} requires shape {expected}")
        resources[field] = array.view((array.size,))
    context.submit_kernel(name, resources, parameters, exact_policy=False,
                          grid_size=count, threads_per_group=128)
    return out


def source_window(state, window, names):
    """Gather the existing donor rectangle using resident Metal copies."""
    from types import SimpleNamespace

    from .nest_family import submit_rect_gather_f32

    context = state.qv.buffer.context
    module = MetalArrayModule(context)
    j, i = window
    ny, nx = j.stop - j.start, i.stop - i.start

    def gather(value):
        if value is None:
            return None
        if value.ndim == 1:
            return value
        levels = 1 if value.ndim == 2 else value.shape[0]
        shape = (ny, nx) if value.ndim == 2 else (levels, ny, nx)
        packed = module.empty(shape, value.dtype)
        submit_rect_gather_f32(context,
            source=value.view((levels, *value.shape[-2:])),
            packed=packed.view((levels, ny, nx)), j0=j.start, i0=i.start,
            nz=levels, ny=ny, nx=nx)
        return packed

    fields = {name: gather(getattr(state, name, None)) for name in names}
    fields.update(thb=gather(getattr(state, "thb", None)),
                  c1h=state.c1h, c2h=state.c2h)
    return SimpleNamespace(**fields)
