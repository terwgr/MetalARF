"""Metal submission preparation for WRF tendency coupling.

All scalar arithmetic is delegated to the shared software RN32 primitives in
``soft_binary32.metalh``. This module owns resource validation, placeholders
for masked typed bindings, output allocation, and submission lifetime.
"""

from __future__ import annotations

import numpy as np

def ready_metal(state, *arrays, mut=None) -> bool:
    """Fail closed unless every rate/metric is a same-context FP32 Metal view."""
    from woof.metal.array import MetalArray

    nz, ny, nx = tuple(int(extent) for extent in state.p.shape)
    expected = [(state.p, (nz, ny, nx)),
                (state.c1h, (nz,)), (state.c2h, (nz,))]
    if mut is not None:
        expected.append((mut, (ny, nx)))
    if state.has_msf:
        expected.extend(((state.msft, (ny, nx)),
                         (state.msfu, (ny, nx + 1)),
                         (state.msfv, (ny + 1, nx))))
    expected.extend((array, (nz, ny, nx)) for array in arrays)
    context = state._array_module.context
    for array, shape in expected:
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous or array.shape != shape):
            raise TypeError(
                "Metal tendency coupling requires contiguous float32 arrays "
                f"with the registered shape {shape}")
        if array.buffer.context is not context:
            raise ValueError("Metal tendency coupling arrays must share one context")
    return True


def _size(*extents) -> int:
    size = 1
    for extent in extents:
        extent = int(extent)
        if extent < 0 or extent > 0x7fffffff:
            raise ValueError("Metal tendency coupling extents must fit int32")
        size *= extent
    if size > 0x7fffffff:
        raise ValueError("Metal tendency coupling volume must fit int32")
    return size


def couple_mass_metal(state, mut, slots, *, theta_slot: int,
                      mask_ring: bool, momentum=None):
    """Allocate and submit fused mass/rate coupling for the WRF callers."""
    slots = list(slots)
    if len(slots) > 6:
        raise ValueError(f"at most 6 rate slots, got {len(slots)}")
    nz, ny, nx = tuple(int(extent) for extent in state.p.shape)
    total = _size(nz, ny, nx)
    if not 0 <= int(theta_slot) < max(1, len(slots)):
        raise ValueError("theta_slot must name an existing rate slot")
    rates = [rate for rate, _wanted in slots if rate is not None]
    inputs = [*rates]
    if momentum is not None:
        if len(momentum) != 2:
            raise ValueError("momentum must contain A-grid du and dv")
        inputs.extend(momentum)
    ready_metal(state, *inputs, mut=mut)

    module = state._array_module
    context = module.context
    volume_shape = (nz, ny, nx)
    # Typed argument buffers require every bound resource to meet its
    # declared shape even when masks disable a slot. Persistent scratch owns
    # the exact volume geometry, same context, and backing lifetime; masked
    # inputs/outputs are never dereferenced or written by the shader.
    dummy_volume = state.scratch(volume_shape,
                                 "p01_couple_dummy_volume")
    outputs = [module.zeros(volume_shape, np.float32) if wanted else None
               for _rate, wanted in slots]
    present = sum(1 << index for index, (rate, _wanted) in enumerate(slots)
                  if rate is not None)
    wanted_mask = sum(1 << index for index, (_rate, wanted) in enumerate(slots)
                      if wanted)

    # Every typed resource is bound even for disabled slots. The presence and
    # wanted masks prevent dereference/store for these real, same-context
    # placeholders; submission retains all participating backing buffers.
    dummy_msft = (state.msft if state.has_msf else
                  state.scratch((ny, nx), "p01_couple_dummy_msft"))
    resources = {
        "c1h": state.c1h, "c2h": state.c2h, "mut": mut,
        "msft": dummy_msft,
    }
    for index in range(6):
        rate = slots[index][0] if index < len(slots) else None
        out = outputs[index] if index < len(outputs) else None
        resources[f"in{index}"] = rate if rate is not None else dummy_volume
        resources[f"out{index}"] = out if out is not None else dummy_volume

    if momentum is None:
        du = dv = mass_u = mass_v = dummy_volume
    else:
        du, dv = momentum
        mass_u = module.zeros(volume_shape, np.float32)
        mass_v = module.zeros(volume_shape, np.float32)
    resources.update({"du": du, "dv": dv,
                      "mass_u": mass_u, "mass_v": mass_v})

    from woof.wrf_exact import ENABLED as WRF_EXACT

    if total:
        context.submit_kernel(
            "couple_mass_rates", resources,
            {"present": present, "wanted": wanted_mask,
             "nslot": len(slots), "theta_slot": int(theta_slot),
             "momentum": int(momentum is not None),
             "mask_ring": int(bool(mask_ring)),
             "has_msf": int(bool(state.has_msf)),
             "nz": nz, "ny": ny, "nx": nx},
            exact_policy=bool(WRF_EXACT), grid_size=total,
            threads_per_group=128)
    if momentum is None:
        mass_u = mass_v = None
    return outputs, mass_u, mass_v


def couple_faces_metal(state, mass_u, mass_v, *, mask_ring: bool,
                       open_x: bool, open_y: bool):
    """Submit periodic face interpolation, masks, map factors and alias closure."""
    from woof.wrf_exact import (BIGSTEP_ENABLED as WRF_EXACT_C_BIGSTEP,
                                ENABLED as WRF_EXACT)

    nz, ny, nx = tuple(int(extent) for extent in state.p.shape)
    ready_metal(state, mass_u, mass_v)
    nu = _size(nz, ny, nx + 1)
    nv = _size(nz, ny + 1, nx)
    total = nu + nv
    if total > 0x7fffffff:
        raise ValueError("Metal tendency-face grid must fit int32")
    module = state._array_module
    ru = module.zeros((nz, ny, nx + 1), np.float32)
    rv = module.zeros((nz, ny + 1, nx), np.float32)
    dummy_msfu = (state.msfu if state.has_msf else
                  state.scratch((ny, nx + 1), "p01_couple_dummy_msfu"))
    dummy_msfv = (state.msfv if state.has_msf else
                  state.scratch((ny + 1, nx), "p01_couple_dummy_msfv"))
    resources = {
        "mass_u": mass_u, "mass_v": mass_v,
        "msfu": dummy_msfu,
        "msfv": dummy_msfv,
        "ru": ru, "rv": rv,
    }
    if total:
        module.context.submit_kernel(
            "couple_faces", resources,
            {"mask_ring": int(bool(mask_ring)),
             "open_x": int(bool(open_x)), "open_y": int(bool(open_y)),
             "has_msf": int(bool(state.has_msf)),
             "wrf_exact": int(bool(WRF_EXACT_C_BIGSTEP)),
             "nz": nz, "ny": ny, "nx": nx},
            exact_policy=bool(WRF_EXACT), grid_size=total,
            threads_per_group=128)
    return ru, rv


__all__ = ["ready_metal", "couple_mass_metal", "couple_faces_metal"]
