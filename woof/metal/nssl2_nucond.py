"""Typed Metal submitters for the default NSSL option-18 NUCOND stage."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from woof.metal.array import MetalArray


_TPB = 256


def _validate_float_array(array, name, context, shape=None):
    if not isinstance(array, MetalArray):
        raise TypeError(f"{name} must be a MetalArray")
    if array.dtype != np.dtype(np.float32) or not array.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous FP32 storage")
    if array.buffer.context is not context:
        raise ValueError("NSSL resources must share the submitting MetalContext")
    if shape is not None and array.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {array.shape}")


def _dimensions(nz, ny, nx):
    dims = (nz, ny, nx)
    if any(isinstance(v, (bool, np.bool_))
           or not isinstance(v, (int, np.integer)) or v < 1 for v in dims):
        raise ValueError("NSSL dimensions must be positive integers")
    if nz < 3:
        raise ValueError("NSSL NUCOND requires nz >= 3")
    return tuple(int(v) for v in dims)


def _step(dt_s):
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("NSSL dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("NSSL dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("NSSL dt_s must be representable as positive float32")
    return step32


def _cell_resources(context, nz, ny, nx, fields):
    for name, array in fields.items():
        _validate_float_array(array, name, context, (nz, ny, nx))


def submit_nssl2_nucond_supersaturation(
        context, full_theta, pressure_pa, exner, qv, scratch, *, nz, ny, nx):
    """Compute WRF's immutable pre-stage supersaturation field."""
    nz, ny, nx = _dimensions(nz, ny, nx)
    fields = {"full_theta": full_theta, "pressure_pa": pressure_pa,
              "exner": exner, "qv": qv, "supersaturation_percent": scratch}
    _cell_resources(context, nz, ny, nx, fields)
    n = nz * ny * nx
    return context.submit_kernel(
        "nssl2_nucond_supersaturation", fields,
        {"n": n, "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=n, threads_per_group=_TPB)


def submit_nssl2_nucond_default(
        context, full_theta, air_density, pressure_pa, exner, w_interface,
        initial_supersaturation_percent, qv, qc, qr, qi, qs, qndrop, qnr,
        qni, qns, qnn, dt_s, *, nz, ny, nx,
        concentration_space=False, predicted_ccn=True):
    """Advance the default WRF ``ipconc=5`` warm NUCOND stage in place."""
    nz, ny, nx = _dimensions(nz, ny, nx)
    if type(concentration_space) is not bool or type(predicted_ccn) is not bool:
        raise TypeError("NSSL NUCOND switches must be bool")
    step32 = _step(dt_s)
    fields = {
        "full_theta": full_theta, "air_density": air_density,
        "pressure_pa": pressure_pa, "exner": exner,
        "initial_supersaturation_percent": initial_supersaturation_percent,
        "qv": qv, "qc": qc, "qr": qr, "qi": qi, "qs": qs,
        "qndrop": qndrop, "qnr": qnr, "qni": qni, "qns": qns,
        "qnn": qnn,
    }
    _cell_resources(context, nz, ny, nx, fields)
    _validate_float_array(w_interface, "w_interface", context,
                          (nz + 1, ny, nx))
    n = nz * ny * nx
    resources = {**fields, "w_interface": w_interface}
    params = {"dt": step32, "concentration_space": int(concentration_space),
              "n": n, "nz": nz, "horizontal_size": ny * nx,
              "predicted_ccn": int(predicted_ccn), "ny": ny, "nx": nx}
    return context.submit_kernel(
        "nssl2_nucond_default", resources, params, exact_policy=False,
        grid_size=n, threads_per_group=_TPB)


def launch_nucond(
        context, full_theta, air_density, pressure_pa, exner, w_interface,
        qv, qc, qr, qi, qs, qndrop, qnr, qni, qns, qnn, dt_s, *,
        supersaturation_scratch, concentration_space=False,
        predicted_ccn=True):
    """Queue the prepass and default warm-phase NUCOND kernel in order.

    The caller owns ``supersaturation_scratch`` so it can reuse storage across
    timesteps. Fields have ``(nz, ny, nx)`` shape, with staggered velocity
    ``(nz + 1, ny, nx)``. Number and CCN values use the driver's selected
    concentration convention.
    """
    if not isinstance(full_theta, MetalArray) or len(full_theta.shape) != 3:
        raise TypeError("full_theta must be a 3-D MetalArray")
    nz, ny, nx = full_theta.shape
    nz, ny, nx = _dimensions(nz, ny, nx)
    context = context or full_theta.buffer.context
    submit_nssl2_nucond_supersaturation(
        context, full_theta, pressure_pa, exner, qv,
        supersaturation_scratch, nz=nz, ny=ny, nx=nx)
    submit_nssl2_nucond_default(
        context, full_theta, air_density, pressure_pa, exner, w_interface,
        supersaturation_scratch, qv, qc, qr, qi, qs, qndrop, qnr, qni,
        qns, qnn, dt_s, nz=nz, ny=ny, nx=nx,
        concentration_space=concentration_space, predicted_ccn=predicted_ccn)


@dataclass(frozen=True)
class NSSL2MetalNucond:
    """Default-hooks compatible callback using workspace-owned scratch."""

    supersaturation_scratch: object
    dt_s: float
    concentration_space: bool = True
    predicted_ccn: bool = True

    def __call__(self, workspace, fields, /) -> None:
        state = workspace.state
        if state.shape[0] != 16:
            raise ValueError("NSSL state must have 16 moment planes")
        nz, ny, nx = workspace.shape
        # The driver's canonical slab order is qv, qc, qr, qi, qs, qndrop,
        # qnr, qni, qns, qnn in state planes 0 through 9.
        def layer(index):
            return state.view(
                (nz, ny, nx),
                byte_offset=state.byte_offset + index * nz * ny * nx * 4)

        launch_nucond(
            state.buffer.context, fields.theta, fields.rho, fields.pressure,
            fields.pii, fields.w, layer(0), layer(1), layer(2), layer(3),
            layer(4), layer(7), layer(8), layer(9), layer(10), layer(13),
            self.dt_s, supersaturation_scratch=self.supersaturation_scratch,
            concentration_space=self.concentration_space,
            predicted_ccn=self.predicted_ccn)


__all__ = [
    "NSSL2MetalNucond", "launch_nucond", "submit_nssl2_nucond_default",
    "submit_nssl2_nucond_supersaturation",
]
