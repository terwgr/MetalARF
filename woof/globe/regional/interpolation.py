"""Pinned numerical interpolation choices for Level-5 parent translation."""
from __future__ import annotations

import numpy as np


def periodic_bilinear(
    source_latitude_deg,
    source_longitude_deg,
    values,
    target_latitude_deg,
    target_longitude_deg,
):
    """Bilinear interpolation on a regular, periodic-longitude global grid."""
    lat = np.asarray(source_latitude_deg, np.float64)
    lon = np.asarray(source_longitude_deg, np.float64)
    field = np.asarray(values)
    target_lat = np.asarray(target_latitude_deg, np.float64)
    target_lon = np.asarray(target_longitude_deg, np.float64)
    if lat.ndim != 1 or lon.ndim != 1 or lat.size < 2 or lon.size < 4:
        raise ValueError("parent latitude/longitude must be regular 1-D arrays")
    if field.shape[-2:] != (lat.size, lon.size):
        raise ValueError("parent field shape does not match coordinates")
    if target_lat.shape != target_lon.shape:
        raise ValueError("target latitude/longitude shapes differ")
    dlat = np.diff(lat)
    if not (np.all(dlat > 0.0) or np.all(dlat < 0.0)) or np.any(np.diff(lon) <= 0.0):
        # A latitude row that merely avoids repeats still breaks searchsorted:
        # the bracket it returns then does not contain the target point and the
        # weights leave [0, 1], so the interpolation is wrong without raising.
        raise ValueError("parent coordinates must be strictly monotonic")
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        field = field[..., ::-1, :]
    dlon = np.diff(lon)
    if not np.allclose(dlon, dlon[0], rtol=0.0, atol=1.0e-10):
        raise ValueError("parent longitude must be uniformly spaced")
    spacing = float(dlon[0])
    if abs(spacing * lon.size - 360.0) > 1.0e-7:
        raise ValueError("parent longitude grid must span exactly one period")

    flat_lat = target_lat.reshape(-1)
    if np.any(flat_lat < lat[0]) or np.any(flat_lat > lat[-1]):
        # The parent export builds its grid with include_poles=False, so the
        # global field genuinely stops short of +/-90.  Filling a poleward
        # target row from the parent's outermost latitude is constant
        # extrapolation over up to half a parent cell of latitude and leaves
        # no trace in the frame, so it is refused instead.
        raise ValueError(
            "target latitude lies outside the parent grid "
            f"[{lat[0]}, {lat[-1]}]: the parent excludes the polar caps"
        )
    j1 = np.searchsorted(lat, flat_lat, side="right")
    j1 = np.clip(j1, 1, lat.size - 1)
    j0 = j1 - 1
    wy = (flat_lat - lat[j0]) / (lat[j1] - lat[j0])

    x = ((target_lon.reshape(-1) - lon[0]) % 360.0) / spacing
    i0 = np.floor(x).astype(np.int64) % lon.size
    i1 = (i0 + 1) % lon.size
    wx = x - np.floor(x)

    f00 = field[..., j0, i0]
    f01 = field[..., j0, i1]
    f10 = field[..., j1, i0]
    f11 = field[..., j1, i1]
    lower = f00 * (1.0 - wx) + f01 * wx
    upper = f10 * (1.0 - wx) + f11 * wx
    result = lower * (1.0 - wy) + upper * wy
    return result.reshape((*field.shape[:-2], *target_lat.shape))


#: ICAO standard-atmosphere tropospheric lapse rate.  WPS/real.exe use the
#: same value to continue temperature below a parent's lowest level; holding
#: potential temperature there instead would impose a dry adiabat (9.8 K/km).
STANDARD_LAPSE_RATE_K_M = 6.5e-3


def extrapolation_fractions(source_pressure, target_pressure) -> dict[str, float]:
    """Fraction of target points outside the parent's own full-level range."""
    source_p = np.asarray(source_pressure, np.float64)
    target_p = np.asarray(target_pressure, np.float64)
    total = float(target_p.size)
    below = float(np.count_nonzero(target_p > source_p[-1][None])) / total
    above = float(np.count_nonzero(target_p < source_p[0][None])) / total
    return {"below_parent_bottom": below, "above_parent_top": above}


def standard_lapse_theta_below(
    source_pressure,
    source_theta,
    target_pressure,
    *,
    reference_pressure_pa: float,
    kappa: float,
    gas_constant: float,
    gravity: float,
):
    """Potential temperature continued below the parent's lowest full level.

    Temperature follows ``STANDARD_LAPSE_RATE_K_M`` downward from the parent's
    bottom full level, with depth taken hypsometrically at that level's own
    temperature; theta is then rebuilt from the continued temperature.  Values
    at or above the parent's bottom level are returned unchanged from it and
    are never used by the caller.
    """
    source_p = np.asarray(source_pressure, np.float64)
    theta = np.asarray(source_theta, np.float64)
    target_p = np.asarray(target_pressure, np.float64)
    bottom_p = source_p[-1][None]
    bottom_theta = theta[-1][None]
    bottom_t = bottom_theta * (bottom_p / reference_pressure_pa) ** kappa
    depth_m = (gas_constant * bottom_t / gravity) * np.log(target_p / bottom_p)
    continued_t = bottom_t + STANDARD_LAPSE_RATE_K_M * depth_m
    return continued_t * (reference_pressure_pa / target_p) ** kappa


def log_pressure_interpolate(
    source_pressure, source_values, target_pressure, *, bottom_values=None
):
    """Independent-column linear interpolation in log pressure.

    Above the parent's top full level the value is held.  Below the parent's
    bottom full level it is held too, unless ``bottom_values`` supplies an
    explicit continuation on the target levels; either way the extrapolated
    depth is bounded by the parent's own bottom layer thickness.
    """
    source_p = np.asarray(source_pressure, np.float64)
    source = np.asarray(source_values, np.float64)
    target_p = np.asarray(target_pressure, np.float64)
    if source_p.shape != source.shape or source_p.ndim != 3:
        raise ValueError("source pressure/value must be equal (nlev,ny,nx) arrays")
    if target_p.ndim != 3 or target_p.shape[1:] != source_p.shape[1:]:
        raise ValueError("target pressure has incompatible shape")
    if np.any(source_p <= 0.0) or np.any(target_p <= 0.0):
        raise ValueError("log-pressure interpolation requires positive pressure")
    if np.any(np.diff(source_p, axis=0) <= 0.0):
        raise ValueError("source pressure must increase top-to-surface")
    if source_p.shape[0] < 2:
        raise ValueError("log-pressure interpolation needs at least two levels")
    # Below the parent's bottom level nothing is interpolated, only continued.
    # One parent bottom layer is the depth over which the parent still resolves
    # the gradient being continued; past it the continuation is invention, and
    # for theta it would be silently reported as a translated parent value.
    limit = np.log(source_p[-1]) - np.log(source_p[-2])
    overshoot = np.log(target_p.max(axis=0)) - np.log(source_p[-1])
    if np.any(overshoot > limit):
        raise ValueError(
            "target pressure extends more than one parent bottom layer below "
            f"the parent's lowest full level (max {float(overshoot.max()):.4f} "
            f"vs {float(limit.min()):.4f} in ln p)"
        )
    flat_sp = np.log(source_p.reshape(source_p.shape[0], -1))
    flat_sv = source.reshape(source.shape[0], -1)
    flat_tp = np.log(target_p.reshape(target_p.shape[0], -1))
    out = np.empty_like(flat_tp)
    for column in range(flat_sp.shape[1]):
        out[:, column] = np.interp(
            flat_tp[:, column], flat_sp[:, column], flat_sv[:, column],
            left=flat_sv[0, column], right=flat_sv[-1, column],
        )
    result = out.reshape(target_p.shape)
    if bottom_values is not None:
        continued = np.asarray(bottom_values, np.float64)
        if continued.shape != target_p.shape:
            raise ValueError("bottom continuation shape does not match target")
        result = np.where(target_p > source_p[-1][None], continued, result)
    return result


def rotate_earth_to_grid(eastward, northward, cosa, sina):
    east = np.asarray(eastward)
    north = np.asarray(northward)
    c = np.asarray(cosa)
    s = np.asarray(sina)
    return east * c + north * s, -east * s + north * c


def stagger_u_nonperiodic(mass_u):
    # The outermost faces are zeroth-order copies of the adjacent mass point,
    # so the prescribed normal wind at each domain edge is displaced half a
    # grid cell inward.  Linear extrapolation would remove the displacement
    # only for a linear profile and would overshoot a sheared jet at exactly
    # the columns the specified zone forces; removing it properly needs u- and
    # v-point latitude/longitude and rotation angles in the regional target
    # contract, which this artifact does not carry.  The half-cell offset is
    # declared in the frame's 'wind' method string.
    value = np.asarray(mass_u)
    out = np.empty((*value.shape[:-1], value.shape[-1] + 1), dtype=value.dtype)
    out[..., 1:-1] = 0.5 * (value[..., :-1] + value[..., 1:])
    out[..., 0] = value[..., 0]
    out[..., -1] = value[..., -1]
    return out


def stagger_v_nonperiodic(mass_v):
    # Same zeroth-order outer faces as stagger_u_nonperiodic.
    value = np.asarray(mass_v)
    out = np.empty(
        (*value.shape[:-2], value.shape[-2] + 1, value.shape[-1]),
        dtype=value.dtype,
    )
    out[..., 1:-1, :] = 0.5 * (value[..., :-1, :] + value[..., 1:, :])
    out[..., 0, :] = value[..., 0, :]
    out[..., -1, :] = value[..., -1, :]
    return out


def side_tables(field, width: int):
    value = np.asarray(field)
    if value.ndim == 2:
        value = value[None]
    if value.ndim != 3:
        raise ValueError("boundary source field must be 2-D or 3-D")
    nz, ny, nx = value.shape
    if width < 1 or 2 * width >= min(ny, nx):
        raise ValueError("boundary width leaves no unique regional interior")
    # Arwen/WRF boundary tables are outermost-first.  West/south already
    # have that order in array storage; east/north must be reversed exactly
    # as woof.ingest.lateral_bc._field_boundary does.
    return {
        "west": np.ascontiguousarray(value[:, :, :width]),
        "east": np.ascontiguousarray(value[:, :, -width:][:, :, ::-1]),
        "south": np.ascontiguousarray(value[:, :width, :]),
        "north": np.ascontiguousarray(value[:, -width:, :][:, ::-1, :]),
    }


__all__ = [
    "STANDARD_LAPSE_RATE_K_M", "extrapolation_fractions",
    "log_pressure_interpolate", "periodic_bilinear", "rotate_earth_to_grid",
    "side_tables", "standard_lapse_theta_below", "stagger_u_nonperiodic",
    "stagger_v_nonperiodic",
]
