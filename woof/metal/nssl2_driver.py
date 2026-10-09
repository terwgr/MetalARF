"""Metal implementations of the NSSL production driver's transport seams.

These functions mirror the NSSL CUDA driver helper signatures so the
production coordinator can select them as a backend without changing field
ordering or the state ABI. They do not select ``mp_physics=18`` themselves.
"""

from __future__ import annotations

import numpy as np

from woof.core.nssl2_driver_support import NSSL2DriverWorkspace
from woof.metal.array import MetalArray, MetalArrayModule

from .nssl2_diagnostics import (
    submit_nssl2_effective_radius_concentration,
    submit_nssl2_radardd02,
)
from .nssl2_driver_transport import (
    submit_nssl2_driver_gather_initialize,
    submit_nssl2_driver_reduce_precipitation,
    submit_nssl2_driver_scatter,
)
from .nssl2_sedimentation import submit_nssl2_sediment_all


_FIELD_INDEX = {
    "qv": 0, "qc": 1, "qr": 2, "qi": 3, "qs": 4, "qg": 5,
    "qh": 6, "qndrop": 7, "qnr": 8, "qni": 9, "qns": 10,
    "qng": 11, "qnh": 12, "qnn": 13, "qvolg": 14, "qvolh": 15,
}


def nssl2_workspace_field(workspace: NSSL2DriverWorkspace, name: str):
    """Return a zero-copy field view from CPU or Metal driver workspace."""
    if not isinstance(workspace, NSSL2DriverWorkspace):
        raise TypeError("workspace must be NSSL2DriverWorkspace")
    if name not in _FIELD_INDEX:
        raise KeyError(f"unknown NSSL driver field {name!r}")
    state = workspace.state
    if isinstance(state, MetalArray):
        nz, ny, nx = workspace.shape
        index = _FIELD_INDEX[name]
        return state.view(
            (nz, ny, nx),
            byte_offset=state.byte_offset + index * nz * ny * nx * 4)
    return workspace.field(name)


def gather_initialize_and_sediment(
        air_density, dz,
        qv, qc, qr, qi, qs, qg, qh,
        qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
        dt_s: float, *, temperature_k,
        first_step: bool = False, cu_used: bool = False,
        qrcuten=None, qscuten=None, qicuten=None, qccuten=None,
        workspace: NSSL2DriverWorkspace | None = None,
        predicted_ccn: bool = True) -> NSSL2DriverWorkspace:
    """Metal counterpart of the production gather/initialize/sediment path."""
    if not isinstance(air_density, MetalArray):
        raise TypeError("Metal NSSL driver requires MetalArray fields")
    context = air_density.buffer.context
    shape = air_density.shape
    if len(shape) != 3:
        raise ValueError("NSSL driver fields must be 3-D")
    nz, ny, nx = shape
    if workspace is None:
        xp = MetalArrayModule(context)
        workspace = NSSL2DriverWorkspace(
            state=xp.empty((16, nz, ny, nx), dtype=np.float32),
            category_surface_export=xp.empty((5, ny, nx), dtype=np.float32),
            shape=(nz, ny, nx),
            ignored_accumulator=xp.empty((ny, nx), dtype=np.float32),
        )
    if not isinstance(workspace, NSSL2DriverWorkspace):
        raise TypeError("workspace must be NSSL2DriverWorkspace or None")
    if tuple(workspace.shape) != tuple(shape):
        raise ValueError("NSSL workspace and Registry fields must share a shape")

    rates = (qrcuten, qscuten, qicuten, qccuten)
    if cu_used:
        if any(value is None for value in rates):
            xp = MetalArrayModule(context)
            zero = xp.zeros(shape, dtype=np.float32)
            rates = tuple(zero if value is None else value for value in rates)
    else:
        # The CUDA driver does not dereference these rates when CU is off.
        rates = tuple(qv if value is None else value for value in rates)

    submit_nssl2_driver_gather_initialize(
        context, workspace.state,
        air_density=air_density, qv=qv, qc=qc, qr=qr, qi=qi, qs=qs,
        qg=qg, qh=qh, qndrop=qndrop, qnr=qnr, qni=qni, qns=qns,
        qng=qng, qnh=qnh, qnn=qnn, qvolg=qvolg, qvolh=qvolh,
        qrcuten=rates[0], qscuten=rates[1], qicuten=rates[2],
        qccuten=rates[3], dt_s=dt_s, first_step=first_step,
        cu_used=cu_used, predicted_ccn=predicted_ccn)
    submit_nssl2_sediment_all(
        context, air_density, temperature_k, workspace.state, dz,
        workspace.ignored_accumulator, workspace.category_surface_export,
        dt_s, nz=nz, ny=ny, nx=nx)
    return workspace


def reduce_nssl2_precipitation(
        workspace: NSSL2DriverWorkspace,
        rainnc, rainncv, snownc, snowncv,
        graupelnc, graupelncv, hailnc, hailncv, sr) -> None:
    """Metal counterpart of WRF's four-category precipitation reducer."""
    if not isinstance(workspace, NSSL2DriverWorkspace):
        raise TypeError("workspace must be NSSL2DriverWorkspace")
    context = workspace.state.buffer.context
    submit_nssl2_driver_reduce_precipitation(
        context, workspace.category_surface_export, rainnc=rainnc,
        rainncv=rainncv, snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv,
        hailnc=hailnc, hailncv=hailncv, sr=sr)


def scatter_nssl2_driver_workspace(
        workspace: NSSL2DriverWorkspace, air_density,
        qv, qc, qr, qi, qs, qg, qh,
        qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
        *, predicted_ccn: bool = True) -> None:
    """Metal counterpart of the driver's one final Registry scatter."""
    if not isinstance(workspace, NSSL2DriverWorkspace):
        raise TypeError("workspace must be NSSL2DriverWorkspace")
    submit_nssl2_driver_scatter(
        workspace.state.buffer.context, workspace.state, air_density,
        qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, qh=qh,
        qndrop=qndrop, qnr=qnr, qni=qni, qns=qns, qng=qng,
        qnh=qnh, qnn=qnn, qvolg=qvolg, qvolh=qvolh,
        predicted_ccn=predicted_ccn)


def launch_radardd02(
        air_density, temperature_k, qr, qi, qs, qg, qh,
        qnr, qni, qns, qng, qnh, qvolg, qvolh, refl_10cm, *,
        output_due: bool, concentration_space: bool = False,
        validate_values: bool = True) -> bool:
    """Metal counterpart of the output-gated NSSL reflectivity launcher."""
    if type(output_due) is not bool:
        raise TypeError("output_due must be bool")
    if not output_due:
        return False
    del validate_values
    return bool(submit_nssl2_radardd02(
        air_density.buffer.context, air_density, temperature_k,
        qr, qi, qs, qg, qh, qnr, qni, qns, qng, qnh, qvolg, qvolh,
        refl_10cm, concentration_space=concentration_space,
        output_due=output_due))


def launch_effective_radius_concentration(
        air_density, qc, cloud_number, qi, ice_number, qs, snow_number,
        re_cloud, re_ice, re_snow, *, validate_values: bool = True) -> None:
    """Metal counterpart of the concentration-space radius diagnostic."""
    del validate_values
    submit_nssl2_effective_radius_concentration(
        air_density.buffer.context, air_density, qc, cloud_number, qi,
        ice_number, qs, snow_number, re_cloud, re_ice, re_snow)


__all__ = [
    "gather_initialize_and_sediment", "launch_effective_radius_concentration",
    "launch_radardd02", "nssl2_workspace_field",
    "reduce_nssl2_precipitation", "scatter_nssl2_driver_workspace",
]
