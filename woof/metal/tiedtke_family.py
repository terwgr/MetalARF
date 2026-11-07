"""Metal launcher for the original Tiedtke cumulus (cu_physics=6)."""

from __future__ import annotations

import numpy as np

from woof.core.tiedtke import (INPUTS, LEVEL_OUTPUTS, REFUSAL, workspace_shapes)
from woof.metal.array import MetalArray
from woof.metal.cumulus_launch import COLUMNS_PER_SUBMIT, submit_column_kernel

_FAMILY = "tiedtke_family"
_ENTRY = "tiedtke_column"


def _launch(inputs, outputs, activity_flag, workspace, *, dt, itimestep, stepcu,
            nz, ncol, f_qc, f_qi, force_continue):
    resources = {name: inputs[name] for name in INPUTS}
    resources.update(outputs)
    resources.update(cu_act_flag=activity_flag, workspace_f=workspace[0],
                     workspace_i=workspace[1], workspace_b=workspace[2])
    submit_column_kernel(
        _FAMILY, _ENTRY, resources,
        dict(dt=dt, itimestep=itimestep, stepcu=stepcu, nz=nz, ncol=ncol,
             f_qc=f_qc, f_qi=f_qi, force_continue=force_continue),
        grid_size=ncol)


def column_step(module, inputs, outputs, *, dt, itimestep, stepcu, nz, ncol,
                width, f_qc, f_qi, guess=1):
    """Run the original driver over ``ncol`` columns of flat Metal arrays.

    The per-row early return of F:960 is decided from the first ascent's
    activity words exactly as the CUDA launcher does; the columns of a row
    that took the other path are recomputed in place from the same inputs.
    Returns the per-column flag array, negative for a refused column.
    """
    shapes = workspace_shapes(nz, ncol)
    workspace = (module.empty((int(np.prod(shapes[0])),), np.float32),
                 module.empty((int(np.prod(shapes[1])),), np.int32),
                 module.zeros(((int(np.prod(shapes[2])) + 3) // 4,), np.uint32))
    flag = module.empty((ncol,), np.int32)
    flag.fill(np.int32(1))
    common = dict(dt=dt, itimestep=itimestep, stepcu=stepcu, nz=nz, ncol=ncol,
                  f_qc=f_qc, f_qi=f_qi)
    if ncol == 1 or width == 1:
        _launch(inputs, outputs, flag, workspace, force_continue=0, **common)
    else:
        words = workspace[1]
        activity = words.view((ncol,), byte_offset=(nz + 8) * ncol * 4)
        activity.fill(np.int32(guess))
        _launch(inputs, outputs, flag, workspace, force_continue=2, **common)
        active = activity.to_numpy().reshape(-1, width) != 0
        row = active.any(axis=1, keepdims=True)
        redo = ~active & (row != bool(guess))
        activity[...] = np.where(redo, row.astype(np.int32),
                                 np.int32(-1)).reshape(ncol)
        _launch(inputs, outputs, flag, workspace, force_continue=2, **common)
    return flag


def _flat(array, shape, name, context):
    if (not isinstance(array, MetalArray) or array.shape != tuple(shape)
            or array.dtype != np.dtype(np.float32)
            or not array.flags.c_contiguous
            or array.buffer.context is not context):
        raise TypeError(f"Tiedtke {name} must be a same-context contiguous "
                        f"FP32 MetalArray with shape {tuple(shape)}")
    return array.view((array.size,))


def run_tiedtke(owner, *, atmosphere, fields, state, cfg):
    """Run cu_physics=6 with Metal storage and native column launches."""
    from woof.core.physics import _NativeNtCumulusResult, _model_clock_dt
    from woof.core.physics_inventory import PBL_RQI_MICROPHYSICS
    from woof.metal.ntiedtke_family import _pack_chunk, _scatter_chunk

    module = state._array_module
    context = module.context
    nz, ny, nx = state.p.shape
    ncol = ny * nx
    volume, interface, surface = (nz, ny, nx), (nz + 1, ny, nx), (ny, nx)

    def lane(name):
        value = None if owner._driver is None else getattr(owner._driver, name, None)
        return module.zeros(volume, dtype=np.float32) if value is None else value

    sources = {
        "u3d": (atmosphere["u"], volume), "v3d": (atmosphere["v"], volume),
        "w": (state.w, interface), "t3d": (atmosphere["temperature"], volume),
        "qv3d": (atmosphere["qv"], volume), "qc3d": (atmosphere["qc"], volume),
        "qi3d": (atmosphere["qi"], volume), "pi3d": (atmosphere["exner"], volume),
        "rho3d": (atmosphere["rho"], volume),
        "qvften": (lane("gf_rqvdynten"), volume),
        "qvpblten": (lane("gf_rqvblten"), volume),
        "dz8w": (atmosphere["dz"], volume),
        "pcps": (atmosphere["pressure"], volume),
        "p8w": (atmosphere["p_interface"], interface),
        "xland": (fields["xland"], surface), "qfx": (fields["qfx"], surface)}
    flat = {name: _flat(a, shape, name, context) for name, (a, shape) in sources.items()}
    flat["znu"] = module.asarray(np.asarray(
        state.znu.to_numpy() if isinstance(state.znu, MetalArray) else state.znu,
        dtype=np.float32))
    f_qi = int(cfg.mp_physics in PBL_RQI_MICROPHYSICS)
    dt = float(_model_clock_dt(cfg))
    epoch = float(getattr(state, "domain_start_offset", 0.0) or 0.0)
    itimestep = int(np.floor((float(state.elapsed_seconds) - epoch) / cfg.dt + 0.5)) + 1
    if not np.isfinite(dt) or dt <= 0 or itimestep < 1:
        raise ValueError("cu6 requires a finite positive dt")
    names = LEVEL_OUTPUTS + ("raincv",)
    results = {n: module.zeros((nz, ncol), np.float32) if n in LEVEL_OUTPUTS
               else module.empty((ncol,), np.float32) for n in names}
    rows = max(1, min(ny, COLUMNS_PER_SUBMIT // nx))
    flags = []
    for r0 in range(0, ny, rows):
        lo, n = r0 * nx, min(ny - r0, rows) * nx
        whole = n == ncol
        if whole:
            chunk = dict(flat)
            out = {k: v for k, v in results.items()}
        else:
            chunk = {"znu": flat["znu"]}
            for name, (a, shape) in sources.items():
                levels = shape[0] if len(shape) == 3 else 1
                target = module.empty((levels * n,), np.float32)
                _pack_chunk(flat[name], target,
                            stage=("ntiedtke_gather_level" if levels == nz else
                                   "ntiedtke_gather_interface" if levels > nz else
                                   "ntiedtke_gather_surface"),
                            full_ncol=ncol, start=lo, count=n, width=n,
                            levels=levels)
                chunk[name] = target
            out = {k: module.zeros((nz * n,), np.float32) if k in LEVEL_OUTPUTS
                   else module.empty((n,), np.float32) for k in names}
        outputs = dict(out)
        for k in outputs:
            outputs[k] = outputs[k].view((outputs[k].size,))
        outputs["pratec"] = module.empty((n,), np.float32)
        flags.append(column_step(
            module, chunk, outputs, dt=dt, itimestep=itimestep, stepcu=1,
            nz=nz, ncol=n, width=nx, f_qc=1, f_qi=f_qi))
        if not whole:
            for k in names:
                _scatter_chunk(out[k], results[k],
                               stage=("ntiedtke_scatter_level" if k in LEVEL_OUTPUTS
                                      else "ntiedtke_scatter_surface"),
                               full_ncol=ncol, start=lo, count=n, width=n,
                               levels=nz)
    if min(int(f.to_numpy().min()) for f in flags) < 0:
        raise ValueError(REFUSAL)

    def grid(name):
        return results[name].view(volume)
    return _NativeNtCumulusResult(
        owner=owner, rthcuten=grid("rthcuten"), rqvcuten=grid("rqvcuten"),
        rqccuten=grid("rqccuten"),
        rqicuten=grid("rqicuten") if getattr(state, "qi", None) is not None else None,
        rucuten=grid("rucuten"), rvcuten=grid("rvcuten"),
        rainc=results["raincv"].view(surface))
