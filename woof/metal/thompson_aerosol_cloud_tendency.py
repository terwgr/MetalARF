"""Current aerosol cloud fallout with an independently retained tendency."""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from .array import MetalArray
from .thompson_aerosol_sed_family import (
    _validate_array, _overlap, _KMAX, _SHALLOW_KMAX, _THREADS,
    _R1, _R2, _OBMR, _SED_W_LIMIT, _SED_HGT_AGL,
    AM_R, AV_C, NC_SEDIMENT_FLOOR,
)


def submit_cloud_tendency(qc, cloud_number_entry, cloud_number_tendency,
                         temperature, pressure, qv, vertical_velocity, dz, *,
                         qcten, reference_density, rain_active_columns,
                         cloud_active_columns, gamma_moments, dt):
    """Keep qc read-only and add source-ordered fallout to qcten and ncten."""
    if not isinstance(qc, MetalArray) or qc.ndim != 3:
        raise TypeError("cloud tendency requires 3-D MetalArray storage")
    nz, ny, nx = map(int, qc.shape)
    if not 2 <= nz <= _KMAX or ny <= 0 or nx <= 0:
        raise ValueError("cloud tendency dimensions are outside the column ABI")
    if qc.size > np.iinfo(np.int32).max:
        raise ValueError("cloud tendency extent exceeds its int32 ABI")
    context = qc.buffer.context
    arrays = {"qc": qc, "qcten": qcten,
              "cloud_number_entry": cloud_number_entry,
              "cloud_number_tendency": cloud_number_tendency,
              "temperature": temperature, "pressure": pressure, "qv": qv,
              "vertical_velocity": vertical_velocity, "dz": dz,
              "reference_density": reference_density,
              "rain_active_columns": rain_active_columns,
              "cloud_active_columns": cloud_active_columns}
    for name, value in arrays.items():
        shape = (ny, nx) if name.endswith("_columns") else qc.shape
        _validate_array(name, value, shape, context)
    if not isinstance(gamma_moments, Mapping) or set(gamma_moments) != {
            "ccg2", "ccg4", "ccg5", "ocg1", "ocg2"}:
        raise ValueError("cloud tendency needs all five verified gamma moments")
    for name, value in gamma_moments.items():
        _validate_array(name, value, (16,), context)
        arrays[name] = value
    entries = list(arrays.items())
    for i, (name, value) in enumerate(entries):
        for other_name, other in entries[:i]:
            if _overlap(value, other):
                raise ValueError(f"cloud tendency resources overlap: {name}, {other_name}")
    step = np.float32(dt)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("cloud tendency step must be finite and positive")
    suffix = "64" if nz <= _SHALLOW_KMAX else "256"
    return context.submit_kernel(
        f"thompson_aa_cloud_sediment_{suffix}_tendency", arrays,
        {"dt": float(step), "r1": float(_R1), "r2": float(_R2),
         "am_r": float(np.float32(AM_R)), "obmr": float(_OBMR),
         "av_c": float(np.float32(AV_C)),
         "nc_sed_floor": float(np.float32(NC_SEDIMENT_FLOOR)),
         "sed_w_limit": float(_SED_W_LIMIT), "sed_hgt_agl": float(_SED_HGT_AGL),
         "nz": nz, "ny": ny, "nx": nx,
         "has_rain_mask": 1, "has_cloud_mask": 1},
        exact_policy=False, grid_size=ny * nx, threads_per_group=_THREADS)
