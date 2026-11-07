"""WRF coordinate-surface mixing with explicit binary32 operation boundaries."""
from __future__ import annotations
import numpy as np
from woof.metal.array import MetalArray
from woof.wrf_exact import ENABLED as WRF_EXACT


def _submit(name, arrays, params, count):
    context = next(iter(arrays.values())).buffer.context
    for label, value in arrays.items():
        if (not isinstance(value, MetalArray) or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous or value.buffer.context is not context):
            raise TypeError(f"coordinate mixing {label} needs contiguous FP32 in one context")
    if WRF_EXACT:
        name = {'wrf_diff_opt1_horizontal':'wrf_exact_diff_opt1_horizontal',
                'wrf_diff_opt1_km4':'wrf_exact_diff_opt1_km4',
                'diff_opt1_theta':'wrf_exact_diff_opt1_theta'}[name]
    return context.submit_kernel(name, arrays, params, exact_policy=WRF_EXACT,
                                 grid_size=count, threads_per_group=128)


def prepare_theta(state, thp, out):
    from woof.core import constants
    nz, ny, nx = thp.shape
    return _submit("diff_opt1_theta", {"thp": thp, "thb": state.thb, "theta": out},
                   dict(t0=np.float32(constants.T0), nz=nz, ny=ny, nx=nx,
                        thb3d=int(state.thb.ndim == 3)), thp.size)


def coefficients(state, cfg, d11, d22, d12, km, kh, prandtl):
    nz, ny, nx = km.shape
    return _submit("wrf_diff_opt1_km4",
        dict(d11=d11, d22=d22, d12=d12, msft=state.msft, km=km, kh=kh),
        dict(dx=np.float32(cfg.dx), dy=np.float32(cfg.dy), cs=np.float32(cfg.c_s),
             pr=np.float32(prandtl), nz=nz, ny=ny, nx=nx,
             bx=int(cfg.open_x or cfg.specified or cfg.nested),
             by=int(cfg.open_y or cfg.specified or cfg.nested)), km.size)


def horizontal(field, km, mu, c1, c2, msft, msfu, msfv, dx, dy, tendency,
               *, stagger="", boundary_x=False, boundary_y=False, theta_initial=None):
    nz, ny, nx = km.shape
    expected = (nz + (stagger == "z"), ny + (stagger == "y"), nx + (stagger == "x"))
    if field.shape != expected or tendency.shape != expected:
        raise ValueError("coordinate mixing stagger does not match field and tendency")
    return _submit("wrf_diff_opt1_horizontal",
        dict(field=field, km=km, mu=mu, c1=c1, c2=c2,
             base=field if theta_initial is None else theta_initial,
             msft=msft, msfu=msfu, msfv=msfv, tendency=tendency),
        dict(rdx=np.float32(1.0/dx), rdy=np.float32(1.0/dy), nz=nz, ny=ny, nx=nx,
             stag={"":0, "x":1, "y":2, "z":3}[stagger], bx=int(boundary_x),
             by=int(boundary_y), perturb=int(theta_initial is not None)), field.size)
