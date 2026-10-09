"""Metal implementation of the ARW ``phy_prep`` atmosphere products.

This slice owns the preparation math used by ``PhysicsDriver._prepare_atmosphere``
for a Metal-resident mass grid. It allocates output fields on the state context
and performs all field arithmetic in one Metal kernel per horizontal column.
"""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def submit_prepare_physics_atmosphere(state):
    """Build the CuPy ``_prepare_atmosphere`` dictionary on Metal.

    Inputs and outputs are contiguous FP32 arrays on the state's Metal
    context. Missing hydrometeors receive independent zero-filled Metal
    buffers, matching the existing scratch-field contract. This function
    performs no host numerical fallback or implicit transfer.
    """
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("physics atmosphere preparation requires a Metal DomainState")
    xp = getattr(state, "_array_module", None)
    context = getattr(xp, "context", None)
    if context is None:
        raise TypeError("Metal DomainState has no Metal array context")

    nz, ny, nx = state.p.shape
    shape3 = (nz, ny, nx)
    shape2 = (ny, nx)
    shapei = (nz + 1, ny, nx)
    required = {
        "thb": ((nz, ny, nx), (nz,)), "thp": shape3, "p": shape3,
        "u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx),
        "phb": ((nz + 1, ny, nx), (nz + 1,)), "php": shapei,
        "c1h": (nz,), "c2h": (nz,), "dnw": (nz,), "alt": shape3,
    }
    arrays = {}
    for name, shapes in required.items():
        value = getattr(state, name, None)
        allowed = shapes if isinstance(shapes[0], tuple) else (shapes,)
        if (not isinstance(value, MetalArray) or value.shape not in allowed
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"Metal physics preparation {name} must be contiguous FP32 with shape in {allowed}")
        if value.buffer.context is not context:
            raise ValueError(f"Metal physics preparation {name} belongs to another context")
        arrays[name] = value

    species = {}
    has = {}
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh"):
        value = getattr(state, name, None)
        if value is None:
            # A unique allocation preserves independent mutable scratch fields.
            value = xp.zeros(shape3, dtype=np.float32)
            has[name] = 0
        else:
            if (not isinstance(value, MetalArray) or value.shape != shape3
                    or value.dtype != np.dtype(np.float32)
                    or not value.flags.c_contiguous):
                raise TypeError(f"Metal physics preparation {name} must be contiguous FP32 {shape3}")
            if value.buffer.context is not context:
                raise ValueError(f"Metal physics preparation {name} belongs to another context")
            has[name] = 1
        species[name] = value
    arrays.update(species)

    outputs = {name: xp.zeros(shape, dtype=np.float32) for name, shape in {
        "theta": shape3, "temperature": shape3, "pressure": shape3,
        "p_interface": shapei, "exner": shape3, "u_mass": shape3,
        "v_mass": shape3, "z_interface": shapei, "dz": shape3,
        "rho": shape3,
    }.items()}
    arrays.update(outputs)
    arrays["total_mu"] = state.total_mu()
    if not isinstance(arrays["total_mu"], MetalArray):
        raise TypeError("Metal total_mu preparation did not return a MetalArray")

    context.submit_kernel(
        "physics_atmosphere_prepare", arrays,
        {"nz": nz, "ny": ny, "nx": nx,
         "thb_3d": int(len(state.thb.shape) == 3),
         "phb_3d": int(len(state.phb.shape) == 3),
         **{f"has_{name}": has[name] for name in species},
         "p_top": np.float32(state.p_top)},
        exact_policy=False, grid_size=ny * nx)

    return {
        "theta": outputs["theta"], "temperature": outputs["temperature"],
        "pressure": outputs["pressure"], "p_interface": outputs["p_interface"],
        "exner": outputs["exner"], "u": outputs["u_mass"],
        "v": outputs["v_mass"], "z_interface": outputs["z_interface"],
        "dz": outputs["dz"], "rho": outputs["rho"],
        "qv": species["qv"], "qc": species["qc"],
        "qi": species["qi"], "qs": species["qs"],
    }
