"""Typed Metal submitters for the native NSSL option-18 fused GS stage."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from woof.metal.array import MetalArray


def _validate_float_array(array, name, context, shape=None):
    if not isinstance(array, MetalArray):
        raise TypeError(f"{name} must be a MetalArray")
    if array.dtype != np.dtype(np.float32) or not array.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous FP32 storage")
    if array.buffer.context is not context:
        raise ValueError("NSSL resources must share the submitting MetalContext")
    if shape is not None and array.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {array.shape}")


def _validate_step(dt_s):
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


def submit_nssl2_prepare_fused_gs(
        context, temperature_k, target_m3, state, full_theta, air_density,
        pressure_pa, exner, *, nz: int, ncol: int):
    """Snapshot the immutable full-step temperature and primary ice target."""
    if (isinstance(nz, (bool, np.bool_))
            or not isinstance(nz, (int, np.integer)) or nz < 2
            or isinstance(ncol, (bool, np.bool_))
            or not isinstance(ncol, (int, np.integer)) or ncol < 1):
        raise ValueError("NSSL fused-GS nz and ncol must be positive integers")
    nz, ncol = int(nz), int(ncol)
    n = nz * ncol
    fields = {
        "temperature_k": temperature_k, "target_m3": target_m3,
        "full_theta": full_theta, "air_density": air_density,
        "pressure_pa": pressure_pa, "exner": exner,
    }
    for name, array in fields.items():
        _validate_float_array(array, name, context, (nz, ncol))
    _validate_float_array(state, "state", context, (16, nz, ncol))
    return context.submit_kernel(
        "nssl2_prepare_fused_gs",
        {**fields, "state": state},
        {"n": n, "nz": nz, "ncol": ncol}, exact_policy=False,
        grid_size=n, threads_per_group=128)


def submit_nssl2_fused_gs(
        context, state, full_theta, air_density, pressure_pa, exner,
        temperature_k, vertical_velocity, primary_ice_target_m3, dz, dt_s, *,
        nz: int, ncol: int, hail_on: bool = True):
    """Advance every NSSL mass, number, volume, and theta moment once."""
    if (isinstance(nz, (bool, np.bool_))
            or not isinstance(nz, (int, np.integer)) or nz < 2
            or isinstance(ncol, (bool, np.bool_))
            or not isinstance(ncol, (int, np.integer)) or ncol < 1):
        raise ValueError("NSSL fused-GS nz and ncol must be positive integers")
    if type(hail_on) is not bool:
        raise TypeError("NSSL hail_on must be bool")
    nz, ncol = int(nz), int(ncol)
    n = nz * ncol
    step32 = _validate_step(dt_s)
    _validate_float_array(state, "state", context, (16, nz, ncol))
    for name, array in (
            ("full_theta", full_theta), ("air_density", air_density),
            ("pressure_pa", pressure_pa), ("exner", exner),
            ("temperature_k", temperature_k),
            ("primary_ice_target_m3", primary_ice_target_m3), ("dz", dz)):
        _validate_float_array(array, name, context, (nz, ncol))
    _validate_float_array(vertical_velocity, "vertical_velocity", context)
    if vertical_velocity.size != (nz + 1) * ncol:
        raise ValueError("NSSL vertical_velocity must contain (nz+1)*ncol values")
    resources = {
        "state": state, "full_theta": full_theta,
        "air_density": air_density, "pressure_pa": pressure_pa,
        "exner": exner, "temperature_k": temperature_k,
        "vertical_velocity": vertical_velocity,
        "primary_ice_target_m3": primary_ice_target_m3, "dz": dz,
    }
    return context.submit_kernel(
        "nssl2_fused_gs", resources,
        {"dt": step32, "nz": nz, "ncol": ncol, "n": n,
         "hail_on": int(hail_on)},
        exact_policy=False, grid_size=n, threads_per_group=128)


def launch_nssl2_fused_gs(
        workspace, fields, temperature_k, primary_ice_target_m3, dt_s, *,
        hail_on=True):
    """Submit the two ordered kernels that make the production GS stage."""
    shape = tuple(workspace.shape)
    if len(shape) != 3:
        raise ValueError("NSSL workspace shape must be (nz, ny, nx)")
    nz, ny, nx = shape
    ncol = ny * nx
    context = workspace.state.buffer.context

    def layer(value, name):
        _validate_float_array(value, name, context, shape)
        return value.view((nz, ncol))

    state = workspace.state.view((16, nz, ncol))
    theta = layer(fields.theta, "full_theta")
    rho = layer(fields.rho, "air_density")
    pressure = layer(fields.pressure, "pressure_pa")
    exner = layer(fields.pii, "exner")
    dz = layer(fields.dz, "dz")
    temp = layer(temperature_k, "temperature_k")
    target = layer(primary_ice_target_m3, "primary_ice_target_m3")
    submit_nssl2_prepare_fused_gs(
        context, temp, target, state, theta, rho, pressure, exner,
        nz=nz, ncol=ncol)
    submit_nssl2_fused_gs(
        context, state, theta, rho, pressure, exner, temp, fields.w.view(
            ((nz + 1), ncol)), target, dz, dt_s,
        nz=nz, ncol=ncol, hail_on=hail_on)


@dataclass(frozen=True)
class NSSL2MetalFusedGS:
    """Default-hooks compatible callback with domain-owned t0/t7 scratch."""

    temperature_k: object
    primary_ice_target_m3: object
    dt_s: float
    hail_on: bool = True

    def __call__(self, workspace, fields, /) -> None:
        launch_nssl2_fused_gs(
            workspace, fields, self.temperature_k,
            self.primary_ice_target_m3, self.dt_s, hail_on=self.hail_on)


__all__ = [
    "NSSL2MetalFusedGS", "launch_nssl2_fused_gs",
    "submit_nssl2_fused_gs", "submit_nssl2_prepare_fused_gs",
]
