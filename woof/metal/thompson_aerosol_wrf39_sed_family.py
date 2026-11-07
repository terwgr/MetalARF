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


def submit_wrf39_rain(qr, nr, temperature, pressure, qv, reference_density,
                      dz, rainnc, rainncv, *, dt):
    arrays = {"qr": qr, "nr": nr, "temperature": temperature,
              "pressure": pressure, "qv": qv,
              "reference_density": reference_density, "dz": dz,
              "rainnc": rainnc, "rainncv": rainncv}
    nz, ny, nx, ctx = _common(qr, arrays, surface=("rainnc", "rainncv"), dt=dt)
    params = {"dt": float(np.float32(dt)), "nz": nz, "ny": ny, "nx": nx}
    return _submit(ctx, f"thompson_aa_wrf39_rain_sediment_{_suffix(nz)}",
                   arrays, params, ny * nx)


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
         "submit_wrf39_rain",
         "submit_wrf39_warm_snow_boost"]


# ---------------------------------------------------------------------------
# WRF v4.6.1 accumulator-form fallout and terminal applies.
# ---------------------------------------------------------------------------

def _flat(arrays, names):
    out = {}
    for name in names:
        a = arrays[name]
        n = a.size
        out[name] = a.view((n,), byte_offset=a.byte_offset)
    return out


def _col_submit(stem, primary, arrays, surface, dt, **options):
    nz, ny, nx, ctx = _common(primary, arrays, surface=surface, dt=dt)
    params = {"dt": float(np.float32(dt)), "nz": nz, "ny": ny, "nx": nx}
    params.update(options)
    return _submit(ctx, f"{stem}_{_suffix(nz)}", arrays, params, ny * nx)


def submit_aa_rain_accumulate(qr1d, nr1d, qrten, nrten, temperature, pressure,
                              qv, reference_density, dz, rainnc, rainncv, *,
                              dt, accumulate_surface=True, export_surface=False):
    arrays = {"qr1d": qr1d, "nr1d": nr1d, "qrten": qrten, "nrten": nrten,
              "temperature": temperature, "pressure": pressure, "qv": qv,
              "reference_density": reference_density, "dz": dz,
              "rainnc": rainnc, "rainncv": rainncv}
    return _col_submit("thompson_aa_rain_sediment_accumulate", qr1d, arrays,
                       ("rainnc", "rainncv"), dt,
                       accumulate_surface=2 if export_surface else int(bool(accumulate_surface)))


def submit_aa_ice_accumulate(qi1d, ni1d, qiten, niten, temperature, pressure,
                             qv, reference_density, rain_active_columns, dz,
                             rainnc, rainncv, snownc, snowncv, *, dt,
                             export_surface=False):
    arrays = {"qi1d": qi1d, "ni1d": ni1d, "qiten": qiten, "niten": niten,
              "temperature": temperature, "pressure": pressure, "qv": qv,
              "reference_density": reference_density,
              "rain_active_columns": rain_active_columns, "dz": dz,
              "rainnc": rainnc, "rainncv": rainncv, "snownc": snownc,
              "snowncv": snowncv}
    return _col_submit("thompson_aa_ice_sediment_accumulate", qi1d, arrays,
                       ("rain_active_columns", "rainnc", "rainncv", "snownc", "snowncv"),
                       dt, export_only=int(bool(export_surface)))


def submit_aa_snow(qs, snow_melt_marker, qr1d, nr1d, qrten, nrten,
                   rain_density, temperature, pressure, qv,
                   reference_density, reference_temperature, velocity_boost,
                   dz, snow_precipitation, qsten, *, dt):
    arrays = {"qs": qs, "snow_melt_marker": snow_melt_marker, "qr1d": qr1d,
              "nr1d": nr1d, "qrten": qrten, "nrten": nrten,
              "rain_density": rain_density, "temperature": temperature,
              "pressure": pressure, "qv": qv,
              "reference_density": reference_density,
              "reference_temperature": reference_temperature,
              "velocity_boost": velocity_boost, "dz": dz,
              "snow_precipitation": snow_precipitation,
              "qsten": qsten if qsten is not None else qs}
    return _col_submit("thompson_aa_snow_sediment", qs, arrays,
                       ("snow_precipitation",), dt,
                       accumulate=int(qsten is not None))


def submit_aa_graupel(qg, graupel_number, rain_density, temperature,
                      pressure, qv, reference_density, dz, active_columns,
                      graupel_precipitation, qgten, ngten, *, dt):
    arrays = {"qg": qg, "graupel_number": graupel_number,
              "rain_density": rain_density, "temperature": temperature,
              "pressure": pressure, "qv": qv,
              "reference_density": reference_density, "dz": dz,
              "active_columns": active_columns,
              "graupel_precipitation": graupel_precipitation,
              "qgten": qgten if qgten is not None else qg,
              "ngten": ngten if ngten is not None else qg}
    return _col_submit("thompson_aa_graupel_sediment", qg, arrays,
                       ("active_columns", "graupel_precipitation"), dt,
                       accumulate=int(qgten is not None))


def submit_aa_surface_precipitation(rainnc, rainncv, snownc, snowncv,
                                    graupelnc, graupelncv, sr):
    names = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
             "graupelncv", "sr")
    arrays = dict(zip(names, (rainnc, rainncv, snownc, snowncv, graupelnc,
                              graupelncv, sr)))
    ctx = rainnc.buffer.context
    for name, arr in arrays.items():
        _validate(name, arr, tuple(rainnc.shape), ctx)
    n = rainnc.size
    return _submit(ctx, "thompson_aa_surface_precipitation",
                   _flat(arrays, names), {"ncol": n}, n)


def submit_aa_entry_warm_mask(temperature, warm_mask):
    shape = tuple(map(int, temperature.shape))
    ctx = temperature.buffer.context
    arrays = {"temperature": temperature, "warm_mask": warm_mask}
    for name, arr in arrays.items():
        _validate(name, arr, shape, ctx)
    nz, ny, nx = shape
    return _submit(ctx, "thompson_aa_entry_warm_mask",
                   _flat(arrays, arrays), {"nz": nz, "ncol": ny * nx},
                   ny * nx)


def submit_aa_terminal_rain_ice(qr, nr, qi, ni, qrten, nrten, qiten, niten,
                                rho, micro_columns, *, dt):
    arrays = {"qr": qr, "nr": nr, "qi": qi, "ni": ni, "qrten": qrten,
              "nrten": nrten, "qiten": qiten, "niten": niten, "rho": rho,
              "micro_columns": micro_columns}
    nz, ny, nx, ctx = _common(qr, arrays, surface=("micro_columns",), dt=dt)
    return _submit(ctx, "thompson_aa_terminal_rain_ice",
                   _flat(arrays, arrays),
                   {"dt": float(np.float32(dt)), "ncol": ny * nx,
                    "n": nz * ny * nx}, nz * ny * nx)


def submit_aa_graupel_number_finalize(qg, graupel_number, terminal_density):
    arrays = {"qg": qg, "graupel_number": graupel_number,
              "terminal_density": terminal_density}
    nz, ny, nx, ctx = _common(qg, arrays)
    n = nz * ny * nx
    return _submit(ctx, "thompson_aa_graupel_number_finalize",
                   _flat(arrays, arrays), {"n": n}, n)


def submit_aa_final_phase_cleanup(
        qc, qi, ni, temperature, cloud_number_entry, ice_number_entry,
        cloud_number_tendency, pressure, qv, qcten, *, dt, qiten=None,
        niten=None, tten=None, heat_ocp=None, heat_lvap=None, theta=None,
        exner=None, wrf39=False):
    arrays = {"qc": qc, "qi": qi, "ni": ni, "temperature": temperature,
              "cloud_number_entry": cloud_number_entry,
              "ice_number_entry": ice_number_entry,
              "cloud_number_tendency": cloud_number_tendency,
              "pressure": pressure, "qv": qv,
              "qcten": qcten if qcten is not None else qc}
    accumulate_ice = qiten is not None
    accumulate_heat = tten is not None
    # Accumulators not in play bind a harmless in-bounds array; the kernel's
    # flags keep it from being read or written.
    arrays.update({
        "qiten": qiten if accumulate_ice else qi,
        "niten": niten if accumulate_ice else qi,
        "tten": tten if accumulate_heat else qi,
        "heat_ocp": heat_ocp if accumulate_heat else qi,
        "heat_lvap": heat_lvap if accumulate_heat else qi,
        "theta": theta if accumulate_heat else qi,
        "exner": exner if accumulate_heat else qi})
    nz, ny, nx, ctx = _common(qc, arrays, dt=dt)
    n = nz * ny * nx
    return _submit(ctx, "thompson_aa_final_phase_cleanup",
                   _flat(arrays, arrays),
                   {"dt": float(np.float32(dt)), "size": n,
                    "accumulate_cloud": int(qcten is not None),
                    "wrf39": int(bool(wrf39)),
                    "accumulate_ice": int(accumulate_ice),
                    "accumulate_heat": int(accumulate_heat)}, n)
