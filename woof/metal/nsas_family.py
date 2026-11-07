"""Metal launcher for the table-driven New SAS column cumulus (cu_physics=96)."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray
from woof.metal.cumulus_launch import COLUMNS_PER_SUBMIT, submit_column_kernel

_FAMILY = "nsas_family"
_MASS = ("t", "qv", "qc", "qi", "p", "pi", "rho", "dz8w", "u", "v")
_KERNEL_NAMES = dict(t="t3d", qv="qv3d", qc="qc3d", qi="qi3d", p="p3d",
                     pi="pi3d", rho="rho3d", dz8w="dz8w", u="u3d", v="v3d")


def _flat(array, shape, name, context):
    if (not isinstance(array, MetalArray) or array.shape != tuple(shape)
            or array.dtype != np.dtype(np.float32)
            or not array.flags.c_contiguous
            or array.buffer.context is not context):
        raise TypeError(f"New SAS {name} must be a same-context contiguous "
                        f"FP32 MetalArray with shape {tuple(shape)}")
    return array.view((array.size,))


def run_nsas(owner, *, atmosphere, fields, state, cfg):
    """Run cu_physics=96 with Metal storage and native column launches."""
    from woof.core.physics import _NativeNtCumulusResult, _model_clock_dt

    scheme = owner.scheme
    module = state._array_module
    context = module.context
    nz, ny, nx = state.p.shape
    ncol = ny * nx
    volume, interface, surface = (nz, ny, nx), (nz + 1, ny, nx), (ny, nx)

    mass_sources = {"t": atmosphere["temperature"], "qv": atmosphere["qv"],
                    "qc": atmosphere["qc"], "qi": atmosphere["qi"],
                    "p": atmosphere["pressure"], "pi": atmosphere["exner"],
                    "rho": atmosphere["rho"], "dz8w": atmosphere["dz"],
                    "u": atmosphere["u"], "v": atmosphere["v"]}
    inputs = {_KERNEL_NAMES[name]: _flat(mass_sources[name], volume, name, context)
              for name in _MASS}
    inputs["p3di"] = _flat(atmosphere["p_interface"], interface, "p8w", context)
    inputs["w"] = _flat(state.w, interface, "w", context)
    columns = {name: _flat(fields[name], surface, name, context)
               for name in ("xland", "pblh", "hfx", "qfx")}

    clock_dt = float(_model_clock_dt(cfg))
    floats = {"dx": float(cfg.dx), "dt": clock_dt}
    for field_name, slot in scheme.config_floats:
        floats[slot] = float(getattr(cfg, field_name))
    ints = {"stepcu": owner._stepcu(),
            "ncloud": int(scheme.ncloud(cfg.mp_physics))}
    for field_name, slot in scheme.config_ints:
        ints[slot] = int(getattr(cfg, field_name))

    sc = module.empty((ncol * 7,), np.float32)
    ic = module.empty((ncol * 3,), np.int32)
    submit_column_kernel(
        _FAMILY, "nsas_pack_columns",
        {"xland": columns["xland"], "hpbl": columns["pblh"],
         "hfx": columns["hfx"], "qfx": columns["qfx"], "sc": sc, "ic": ic},
        dict(dx=floats["dx"], dt=floats["dt"], pgcon=floats["pgcon"],
             stepcu=ints["stepcu"], ncloud=ints["ncloud"],
             dx_factor=ints["dx_factor"], ncol=ncol), grid_size=ncol)

    # WRF decides the deep search limits per complete i row.
    limits = module.empty((ny * 3,), np.int32)
    submit_column_kernel(
        _FAMILY, "nsas_row_limits",
        {"p": inputs["p3d"], "p8w": inputs["p3di"], "limits": limits},
        dict(ncol=ncol, nz=nz, row_width=nx), grid_size=ny)

    tile = max(1, min(ncol, COLUMNS_PER_SUBMIT))
    workspace = module.empty((scheme.workspace_slots * (nz + 2) * tile,), np.float32)
    out = {name: module.empty((nz * ncol,), np.float32) for name in (
        "rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten")}
    sfc = module.empty((4 * ncol,), np.float32)
    for col0 in range(0, ncol, tile):
        width = min(tile, ncol - col0)
        submit_column_kernel(
            _FAMILY, "nsas_column",
            {**inputs, "sc": sc, "ic": ic, **out, "sfc": sfc, "ws": workspace,
             "row_limits": limits},
            dict(col0=col0, ntile=width, ncol=ncol, nz=nz, row_width=nx,
                 has_qc=1, has_qi=1), grid_size=width)
    pratec = sfc.view((ncol,), byte_offset=ncol * 4)
    clock_raincv = module.empty((ncol,), np.float32)
    submit_column_kernel(
        _FAMILY, "nsas_clock_rain", {"pratec": pratec, "raincv": clock_raincv},
        dict(dt=clock_dt, ncol=ncol), grid_size=ncol)

    def grid(name):
        return out[name].view(volume)
    return _NativeNtCumulusResult(
        owner=owner, rthcuten=grid("rthcuten"), rqvcuten=grid("rqvcuten"),
        rqccuten=grid("rqccuten"),
        rqicuten=grid("rqicuten") if getattr(state, "qi", None) is not None else None,
        rucuten=grid("rucuten"), rvcuten=grid("rvcuten"),
        pratec=pratec.view(surface), persistent_pratec=True,
        clock_raincv=clock_raincv.view(surface))
