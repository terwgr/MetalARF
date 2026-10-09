"""Typed Metal submissions for the WRF 3.9 Thompson fallout family."""
from __future__ import annotations

import numpy as np
from woof.metal.array import MetalArray

_FLOAT = np.dtype(np.float32)
_THREADS = 32
_KMAX = 256
_SHALLOW_KMAX = 64


def _validate(name, value, shape, context):
    if (not isinstance(value, MetalArray) or value.dtype != _FLOAT
            or value.shape != shape or not value.flags.c_contiguous):
        raise TypeError(f"WRF39 Thompson sedimentation {name} must be contiguous "
                        f"float32 with shape {shape}")
    if value.buffer.context is not context:
        raise ValueError("WRF39 Thompson fields must share one Metal context")


def _common(primary, arrays, *, surface=(), dt=None):
    if not isinstance(primary, MetalArray) or primary.ndim != 3:
        raise TypeError("WRF39 Thompson sedimentation requires 3-D MetalArray")
    nz, ny, nx = map(int, primary.shape)
    if not (2 <= nz <= _KMAX and ny > 0 and nx > 0):
        raise ValueError(f"WRF39 fallout requires 2 <= nz <= {_KMAX} and a nonempty grid")
    if max(nz * ny * nx, ny * nx) > np.iinfo(np.int32).max:
        raise ValueError("WRF39 Thompson dimensions must fit int32")
    if dt is not None and (not np.isfinite(np.float32(dt)) or np.float32(dt) <= 0):
        raise ValueError(f"WRF39 fallout dt must be finite and positive, got {dt!r}")
    ctx = primary.buffer.context
    for name, arr in arrays.items():
        shape = (ny, nx) if name in surface else (nz, ny, nx)
        _validate(name, arr, shape, ctx)
    return nz, ny, nx, ctx


def _submit(context, name, arrays, params, columns):
    return context.submit_kernel(name, arrays, params, exact_policy=False,
                                 grid_size=columns, threads_per_group=_THREADS)


def _suffix(nz):
    return "64" if nz <= _SHALLOW_KMAX else "256"


def submit_wrf39_ice(qi, ni, temperature, pressure, qv, reference_density,
                      dz, rainnc, rainncv, snownc, snowncv, *, dt):
    arrays = {"qi": qi, "ni": ni, "temperature": temperature,
              "pressure": pressure, "qv": qv,
              "reference_density": reference_density, "dz": dz,
              "rainnc": rainnc, "rainncv": rainncv,
              "snownc": snownc, "snowncv": snowncv}
    nz, ny, nx, ctx = _common(qi, arrays,
        surface=("rainnc", "rainncv", "snownc", "snowncv"), dt=dt)
    params = {"dt": float(np.float32(dt)), "nz": nz, "ny": ny, "nx": nx}
    return _submit(ctx, f"thompson_aa_wrf39_ice_sediment_{_suffix(nz)}",
                   arrays, params, ny * nx)


def submit_wrf39_graupel(qg, graupel_intercept, temperature, pressure, qv,
        reference_density, melt_rain_qr, melt_rain_nr, melt_rain_density, dz,
        rainnc, rainncv, graupelnc, graupelncv, active_columns, *, dt):
    arrays = {"qg": qg, "graupel_intercept": graupel_intercept,
        "temperature": temperature, "pressure": pressure, "qv": qv,
        "reference_density": reference_density, "melt_rain_qr": melt_rain_qr,
        "melt_rain_nr": melt_rain_nr, "melt_rain_density": melt_rain_density,
        "dz": dz, "rainnc": rainnc, "rainncv": rainncv,
        "graupelnc": graupelnc, "graupelncv": graupelncv,
        "active_columns": active_columns}
    surf=("rainnc","rainncv","graupelnc","graupelncv","active_columns")
    nz,ny,nx,ctx=_common(qg,arrays,surface=surf,dt=dt)
    params={"dt":float(np.float32(dt)),"nz":nz,"ny":ny,"nx":nx}
    return _submit(ctx,f"thompson_aa_wrf39_graupel_sediment_{_suffix(nz)}",arrays,params,ny*nx)


def submit_wrf39_snow(qs, snow_melt_marker, melt_rain_qr, melt_rain_nr,
        temperature, pressure, qv, reference_density, reference_temperature,
        velocity_boost, melt_rain_density, dz, rainnc, rainncv, snownc,
        snowncv, *, dt, singular_fall):
    arrays={"qs":qs,"snow_melt_marker":snow_melt_marker,
        "melt_rain_qr":melt_rain_qr,"melt_rain_nr":melt_rain_nr,
        "temperature":temperature,"pressure":pressure,"qv":qv,
        "reference_density":reference_density,
        "reference_temperature":reference_temperature,
        "velocity_boost":velocity_boost,"melt_rain_density":melt_rain_density,
        "dz":dz,"rainnc":rainnc,"rainncv":rainncv,"snownc":snownc,
        "snowncv":snowncv}
    surf=("rainnc","rainncv","snownc","snowncv")
    nz,ny,nx,ctx=_common(qs,arrays,surface=surf,dt=dt)
    params={"singular_fall":int(bool(singular_fall)),"dt":float(np.float32(dt)),
            "nz":nz,"ny":ny,"nx":nx}
    return _submit(ctx,f"thompson_aa_wrf39_snow_sediment_{_suffix(nz)}",arrays,params,ny*nx)


def submit_wrf39_warm_snow_boost(entry_warm_mask, velocity_boost):
    if (not isinstance(entry_warm_mask, MetalArray)
            or entry_warm_mask.ndim != 3):
        raise TypeError("WRF39 warm snow boost requires a 3-D MetalArray")
    shape=tuple(map(int,entry_warm_mask.shape));ctx=entry_warm_mask.buffer.context
    _validate("velocity_boost",velocity_boost,shape,ctx)
    n=int(np.prod(shape,dtype=np.int64))
    if n > np.iinfo(np.int32).max: raise ValueError("WRF39 boost size must fit int32")
    arrays={"entry_warm_mask":entry_warm_mask,"velocity_boost":velocity_boost}
    flat={name:value.view((n,),byte_offset=value.byte_offset)
          for name,value in arrays.items()}
    return _submit(ctx,"thompson_aa_wrf39_warm_snow_boost",flat,{"n":n},n)


__all__=["submit_wrf39_ice","submit_wrf39_graupel","submit_wrf39_snow",
         "submit_wrf39_warm_snow_boost"]
