"""Metal submission for supported nest microphysics edge closures.

The NSSL MP8-to-MP18 closure remains refused until its source's required
binary64 operations have a faithful Metal implementation.
"""

from __future__ import annotations

from .array import MetalArray


_EDGE_RESOURCES = (
    "alt", "source_qv", "source_qc", "source_qr", "source_qi",
    "source_qs", "source_qg", "source_qh", "source_qir", "source_qib",
    "target_thb", "target_thp", "target_pres", "my2_ck", "mub2d",
    "mup", "c1h", "c2h", "out",
)


def submit_mp8_to_mp18_mass_diagnosed_field(*args, **kwargs):
    """Refuse the NSSL mass closure without substituting float32 pow."""
    raise NotImplementedError(
        "MP8-to-MP18 nest translation requires faithful binary64 pow, "
        "binary64 arithmetic, and conversion rounding; Metal support is "
        "not established")


def submit_microphysics_edge_field(arrays, parameters):
    """Submit supported FP32 arms of ``microphysics_edge_field``.

    The coordinator's caller branch must construct the resource and parameter
    dictionaries with the same field/source codes as
    ``microphysics_transition.py``. MP18 is refused for required binary64;
    MP9 is refused because its CK table is still prepared as a CuPy device
    array by the current caller.
    """
    if not isinstance(arrays, dict) or set(arrays) != set(_EDGE_RESOURCES):
        raise ValueError("nest microphysics resource names do not match the signature")
    if not isinstance(parameters, dict):
        raise TypeError("nest microphysics parameters must be a dictionary")
    target_mp = int(parameters.get("target_mp", -1))
    if target_mp == 18:
        raise NotImplementedError(
            "MP18 nest edge closure requires faithful binary64 operations")
    if target_mp == 9:
        raise NotImplementedError(
            "MP9 nest edge closure needs CK constants prepared in Metal storage")
    required_parameters = {
        "qv_source", "qc_source", "qr_source", "qi_source", "qs_source",
        "qg_source", "qh_source", "field", "source_mp", "target_mp",
        "morr_rhog", "my2_p0", "my2_rcp", "thb_columnar", "wdm6_ccn",
        "coupled", "nz", "ny", "nx",
    }
    if set(parameters) != required_parameters:
        raise ValueError("nest microphysics scalar names do not match the signature")
    nz, ny, nx = (int(parameters[name]) for name in ("nz", "ny", "nx"))
    field_shape = (nz, ny, nx)
    shapes = {name: field_shape for name in _EDGE_RESOURCES}
    shapes.update(mub2d=(ny, nx), mup=(ny, nx),
                  c1h=(nz,), c2h=(nz,))
    context = None
    for name in _EDGE_RESOURCES:
        array = arrays[name]
        if not isinstance(array, MetalArray):
            raise TypeError(f"nest microphysics {name} must be a MetalArray")
        if array.dtype.name != "float32" or not array.flags.c_contiguous:
            raise TypeError(
                f"nest microphysics {name} requires contiguous float32 storage")
        if array.shape != shapes[name]:
            raise ValueError(
                f"nest microphysics {name} shape {array.shape} != {shapes[name]}")
        if context is None:
            context = array.buffer.context
        elif array.buffer.context is not context:
            raise ValueError("nest microphysics resources must share a context")
    return context.submit_kernel(
        "microphysics_edge_field", arrays, parameters,
        exact_policy=False, grid_size=nz * ny * nx)
