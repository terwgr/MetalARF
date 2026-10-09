"""Observation scorecard for WOOF global: the run, a control and the GFS
forecast read the same way against MEASURED observations, never against an
analysis.

What it measures
----------------
Two instruments, one sampling rule.

*Surface.*  ASOS station reports (the ``gpuwm-obs.asos-surface.v2`` or ``.v1`` record
``rw_asos`` decodes) at one valid time, four variables in the seam's units:
2 m temperature (K), 2 m dewpoint (K), 10 m wind speed (m/s) and mean sea
level pressure (Pa; the tables print hPa).  Every model is a
:class:`SurfaceFields`: its 2 m temperature and specific humidity, 10 m wind
components, surface pressure, terrain height and land mask on its own
rectilinear latitude-longitude grid.  The station set is admitted by the
verification library's own rule (:func:`woof.verify.obs.stations.
freeze_station_set`: land at the nearest cell, model terrain within the
registered tolerance of the station elevation, the gross-error screen and
the reporting fraction) with EACH model's own land mask and terrain, and
the scored set is the INTERSECTION of the admitted sets, so every model
meets the same stations and the same report.  A station where any model's
stencil is refused leaves that variable for every model.  The model value
is bilinear in latitude and longitude on the model's own grid
(:class:`RectilinearSampler`); the residual is model minus observation; the
rows are n, bias, rmse and mae.

*Upper air.*  Radiosonde mandatory levels from the IGRA2 year-to-date
archive (:func:`parse_igra2`), at the nominal 00Z and 12Z hours: 500 hPa
geopotential height (m), 850 and 500 hPa temperature (K), and the vector
wind at 250 and 850 hPa (m/s: root-mean-square vector error, wind-speed
bias, u and v biases).  The run's fields come from the upper-air
scorecard's own column interpolation (linear in ln p between the bracketing
full levels, geopotential hydrostatic, below-ground targets refused per
column) on the Gaussian grid; the GFS fields are the product's own isobaric
levels on its 0.25 degree grid, refused where the product's surface pressure
sits below the level.  Both are then sampled at the site bilinearly in
latitude and longitude; a site whose stencil touches a refused column is
refused for every model at that level.  Rows by hemisphere: ``nh`` is
latitude >= 0, ``sh`` latitude < 0, ``global`` everything.

How the two models are sampled differently (stated, not hidden)
-----------------------------------------------------------------
* Our 2 m temperature and specific humidity are the physics suite's
  SFCDIAGS books on land (T2 = TSK - HFX / (rho cp CHS2), Q2 from the
  moisture flux) and the surface layer's bulk profile over water
  (``physics__t2``, ``physics__q2`` in the checkpoint); the GFS 2 m fields
  are diagnosed by the GFS's own surface scheme inside the GFS and written
  to the product.  Each model's 2 m value is the model's own claim about
  2 m; nothing is re-diagnosed here.
* Our 10 m wind is the surface layer's (``physics__u10``, ``physics__v10``);
  the GFS's is its 10 m product field.
* Our terrain is the analysis orography the run integrated with (regridded
  to the Gaussian grid and spectrally truncated, ``phi_s / g`` with the
  model's g 9.80616); the GFS terrain is the product's surface height in gpm.
  The terrain-mismatch admission uses each model's own terrain, so the
  admitted sets differ (the truncated T255 orography is smoother than the
  0.25 degree GFS terrain); the intersection is what is scored.
* Dewpoint is derived on BOTH sides from 2 m specific humidity and surface
  pressure with one formula (Bolton 1980, ``surface_energy.
  dewpoint_from_specific_humidity``), never read from a product dewpoint.
* Mean sea level pressure is reduced on BOTH sides with one formula, the
  scoring chain of record's (``tools/arwen_global_bias_vs_gfs.py``):
  ``p_msl = p_s exp(g h / (R (T2 + 0.0065 h / 2)))`` with g 9.80665 and
  R 287.05.  The GFS product's own PRMSL is not read.  The station's own
  reduction is the ASOS algorithm's.
* Our grid is Gaussian (T255: 384 rows, 768 columns, about 0.47 degree);
  the GFS product's is 0.25 degree.  Bilinear sampling on each model's own
  grid means the GFS is read at roughly twice our resolution; no field is
  regridded before sampling.
* Geopotential height: ours is ``phi / g`` with g 9.80616; the GFS product's
  is gpm (g0 9.80665); IGRA2 reports geopotential metres.  At 500 hPa the
  constants differ by 0.27 m of thickness, well below the rows' rmse.
* A sounding is valid at its nominal hour; the balloon is released up to
  an hour before it and reaches 500 hPa about 20 minutes after release.
  Both models are read at the nominal hour.

Calibration (``python -m woof.globe.obs_scorecard calibrate``;
``tests/test_arwen_global_obs_scorecard.py`` holds every bar): the
families in CALIBRATION below are planted in both directions.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .constants import GRAVITY_M_S2
from .surface_energy import dewpoint_from_specific_humidity
from .upper_air_scorecard import (
    LEVELS_PA,
    PressureLevelFields,
    ModelReader,
    _sha256_file,
    build_transform_from_receipt,
    decode_reference,
    model_cache_path,
    read_receipt,
    receipt_initial_analysis,
    receipt_start_utc,
    resolve_mapping,
    surface_geopotential_for_run,
)

SCHEMA = "gpuwm.arwen-global-obs-scorecard/v1"

#: The scored surface variables, the seam names the station record uses.
SURFACE_VARIABLES: tuple[str, ...] = ("temperature_2m", "dewpoint_2m", "wind_speed_10m", "mslp")
#: Display name, unit and the factor from seam units to the printed unit.
SURFACE_DISPLAY: dict[str, tuple[str, str, float]] = {
    "temperature_2m": ("T2", "K", 1.0),
    "dewpoint_2m": ("Td2", "K", 1.0),
    "wind_speed_10m": ("WS10", "m/s", 1.0),
    "mslp": ("MSLP", "hPa", 0.01),
}
#: The checkpoint arrays a model's surface reading needs, by name.
MODEL_SURFACE_ARRAYS: tuple[str, ...] = (
    "physics__t2", "physics__q2", "physics__u10", "physics__v10",
    "surface__land_fraction", "atmosphere__log_surface_pressure",
)
#: The decoded product fields a GFS surface reading needs, by name.
PRODUCT_SURFACE_FIELDS: tuple[str, ...] = (
    "air_temperature_2m", "specific_humidity_2m", "eastward_wind_10m",
    "northward_wind_10m", "surface_pressure", "terrain_height", "land_fraction",
)
PRODUCT_LEVEL_FIELDS: tuple[str, ...] = (
    "geopotential_height", "air_temperature", "eastward_wind", "northward_wind", "surface_pressure",
)
#: The scoring chain of record's reduction constants (arwen_global_bias_vs_gfs.py).
MSLP_G = 9.80665
MSLP_R = 287.05
MSLP_LAPSE_K_PER_M = 0.0065

#: (score name, field, level Pa) for the upper-air rows.
UPPER_TARGETS: tuple[tuple[str, str, float], ...] = (
    ("z500", "z", 50_000.0),
    ("t850", "t", 85_000.0),
    ("t500", "t", 50_000.0),
    ("w250", "wind", 25_000.0),
    ("w850", "wind", 85_000.0),
)
UPPER_LEVELS_PA: tuple[float, ...] = tuple(sorted({level for _, _, level in UPPER_TARGETS}))
#: Hemisphere name -> (south edge inclusive, north edge inclusive); ``nh``
#: takes the equator.
HEMISPHERES: dict[str, tuple[float, float]] = {
    "nh": (0.0, 90.0),
    "sh": (-90.0, -1.0e-12),
    "global": (-90.0, 90.0),
}
#: Plausibility screen for a radiosonde mandatory-level value; a value
#: outside is dropped and counted, never corrected.
SOUNDING_SCREEN: dict[str, tuple[float, float]] = {
    "z": (-500.0, 20_000.0),
    "t": (150.0, 350.0),
    "wspd": (0.0, 150.0),
}
IGRA2_MISSING = (-9999, -8888)

CALIBRATION = """
Recorded 2026-09-05 on the CPU test host (Python 3.14, numpy 2.5, float64)
by `calibrate`; the test file holds every bar.  Grids: the T21 Gaussian
grid (33 x 66, rows ascending from the south) and a regular 1 degree grid
with rows descending from the north (the product's order), both periodic
in longitude.

Family S1, the sampler against a closed form: a field 3 + 0.7 lat - 0.02 lon
at 400 random points that never cross the longitude seam reads the closed
form to 1.4e-14 on both grids (bar 1e-12); 40 points inside the seam cell
(between the last column and the first) read the four-corner hand
computation to 7.1e-15 (Gaussian) and 0 (regular); the library's
index-space bilinear (woof.verify.obs.stations.sample_field) and this
sampler agree to 8.9e-16 at 400 points on the regular grid; a point
poleward of the first Gaussian row is refused (NaN); a point whose stencil
touches one NaN cell is refused while the clean cell beside it reads 1.0.

Family S2, a model against itself: pseudo-observations equal to the model's
own samples at 300 stations score bias 0, rmse 0, mae 0 exactly for every
surface variable (the intersection with a second model on the regular grid
keeps all 300), and at 200 sites every upper-air target reads exactly 0
(rmsve 0, every wind bias 0).

Family S3, planted differences (observation = model sample + delta; the
row reads model minus observation):

    planted                   bias read          rmse read   mae read
    T2 +1.5 K                 -1.5 (4e-16 off)   1.5         1.5
    Td2 -2.0 K                +2.0               2.0         2.0
    WS10 +0.75 m/s            -0.75              0.75        0.75
    MSLP +120 Pa              -1.2 hPa           1.2         1.2
    z500 +12 m                -12                12          12
    t850 -1.5 K               +1.5               1.5         1.5
    t500 +0.8 K               -0.8 (1e-14 off)   0.8         0.8
    w250 (du,dv) = (+3,-4)    u -3, v +4         rmsve 5
    w850 (du,dv) = (-3,+4)    u +3, v -4         rmsve 5
    z500 +12 m plus a zero-mean texture of rms 5 m: bias -12.000, rmse 13.000 (2e-14 off)

Family S4, the two derived fields: dewpoint from specific humidity and back
through Bolton's saturation vapour pressure round-trips to 0.0 K over 233
to 313 K; a station at terrain 0 reads mean sea level pressure equal to
surface pressure exactly, and terrain 1000 m at T2 288.15 K reads the chain
of record's closed form to 0.0 Pa.

Family S5, the IGRA2 record: a synthetic two-sounding file in the archive's
own fixed columns (verified against the archive's 2026-09-01 12Z records)
reads back z850 1512 m, t850 294.35 K, u850 +10 / v850 0 (270 degrees,
10 m/s), z500 5860 m, t500 262.65 K, u500 0 / v500 -5 (360 degrees, 5 m/s)
to 2e-15; a -9999 height and temperature read None with the 30 m/s wind
speed beside them kept; no 700 hPa row reads None; a 12Z filter keeps one
of the two soundings.

Family S6, refusals by name: a checkpoint lacking physics__t2 refuses
naming physics__t2 (and every other missing array); a decoded product
lacking specific_humidity_2m refuses naming it; a product without an
850 hPa level refuses naming the level.

Family S7, admission parity: the fractional positions this sampler hands
the admission library on the regular grid equal the direct index formula
to 0 (x) and 1.4e-14 (y), and the library admits the same station set from
either.
"""


# --------------------------------------------------------------------------
# the sampler
# --------------------------------------------------------------------------


class RectilinearSampler:
    """Bilinear sampling in latitude and longitude on a rectilinear global
    grid: rows at any strictly monotone latitudes (Gaussian or regular, in
    either order), columns at uniform longitudes covering the globe once,
    periodic across the seam."""

    def __init__(self, latitude_deg, longitude_deg) -> None:
        lat = np.asarray(latitude_deg, dtype=np.float64).ravel()
        lon = np.asarray(longitude_deg, dtype=np.float64).ravel()
        if lat.size < 2 or lon.size < 2:
            raise ValueError("a rectilinear grid needs at least two rows and two columns")
        self.flip = bool(lat[0] > lat[-1])
        ascending = lat[::-1] if self.flip else lat
        if np.any(np.diff(ascending) <= 0.0):
            raise ValueError("grid latitudes must be strictly monotone")
        lon = np.mod(lon, 360.0)
        dlon = np.diff(lon)
        dlon = np.where(dlon < 0.0, dlon + 360.0, dlon)
        if np.max(np.abs(dlon - dlon[0])) > 1.0e-6 or dlon[0] <= 0.0:
            raise ValueError("grid longitudes must be uniform and increasing")
        if abs(float(dlon[0]) * lon.size - 360.0) > 1.0e-3:
            raise ValueError(
                f"grid longitudes span {float(dlon[0]) * lon.size:.4f} degrees, not the globe"
            )
        self.latitude_asc = ascending
        self.nlat = int(lat.size)
        self.nlon = int(lon.size)
        self.lon0 = float(lon[0])
        self.dlon = float(dlon[0])

    @property
    def shape(self) -> tuple[int, int]:
        return self.nlat, self.nlon

    def weights(self, latitude_deg, longitude_deg):
        """Stencil rows (ascending order), columns and weights for points."""
        lat_pts = np.asarray(latitude_deg, dtype=np.float64).ravel()
        lon_pts = np.asarray(longitude_deg, dtype=np.float64).ravel()
        if lat_pts.shape != lon_pts.shape:
            raise ValueError("latitude and longitude point arrays must match")
        rows = self.latitude_asc
        inside = (lat_pts >= rows[0]) & (lat_pts <= rows[-1]) & np.isfinite(lat_pts) & np.isfinite(lon_pts)
        j1 = np.clip(np.searchsorted(rows, np.where(inside, lat_pts, rows[0]), side="right"), 1, rows.size - 1)
        j0 = j1 - 1
        wy = (lat_pts - rows[j0]) / (rows[j1] - rows[j0])
        wy = np.where(inside, wy, 0.0)
        fx = np.mod(np.where(inside, lon_pts, 0.0) - self.lon0, 360.0) / self.dlon
        base = np.floor(fx)
        i0 = np.mod(base.astype(np.int64), self.nlon)
        wx = fx - base
        i1 = np.mod(i0 + 1, self.nlon)
        return j0, j1, wy, i0, i1, wx, inside

    def sample(self, field_2d, latitude_deg, longitude_deg) -> np.ndarray:
        """Bilinear values at the points; NaN where the point is poleward
        of the outermost rows or any stencil corner is NaN."""
        f = np.asarray(field_2d, dtype=np.float64)
        if f.shape != (self.nlat, self.nlon):
            raise ValueError(f"field shape {f.shape} is not the grid's {(self.nlat, self.nlon)}")
        if self.flip:
            f = f[::-1]
        j0, j1, wy, i0, i1, wx, inside = self.weights(latitude_deg, longitude_deg)
        a = f[j0, i0]
        b = f[j0, i1]
        c = f[j1, i0]
        d = f[j1, i1]
        out = (a * (1.0 - wx) + b * wx) * (1.0 - wy) + (c * (1.0 - wx) + d * wx) * wy
        # A NaN corner poisons the value whatever its weight: a refused
        # column anywhere in the stencil refuses the point.
        corners_finite = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d)
        return np.where(inside & corners_finite, out, np.nan)

    def library_positions(self, latitude_deg, longitude_deg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fractional (x, y) in the grid's OWN index order, the convention
        ``woof.verify.obs.stations.StationPosition`` takes: y is the
        fractional row between the bracketing rows (linear in latitude),
        x the fractional column from the first column.  The admission reads
        the nearest cell, round(x) and round(y), so a point in the seam cell
        (between the last column and the first, which the library cannot
        wrap) is handed its nearest column outright: the last column or
        column 0.  Values are never sampled through these positions; the
        sampler above does that with the periodic stencil."""
        j0, _j1, wy, _i0, _i1, wx, inside = self.weights(latitude_deg, longitude_deg)
        y = j0 + wy
        if self.flip:
            y = (self.nlat - 1) - y
        lon_pts = np.asarray(longitude_deg, dtype=np.float64).ravel()
        x = np.mod(lon_pts - self.lon0, 360.0) / self.dlon
        seam = x > self.nlon - 1
        x = np.where(seam, np.where(x - (self.nlon - 1) < 0.5, float(self.nlon - 1), 0.0), x)
        return x, y, inside


# --------------------------------------------------------------------------
# surface fields of one model
# --------------------------------------------------------------------------


def mslp_reduction(surface_pressure_pa, terrain_m, t2_k) -> np.ndarray:
    """The scoring chain of record's sea-level reduction, applied to both
    models alike."""
    ps = np.asarray(surface_pressure_pa, dtype=np.float64)
    h = np.asarray(terrain_m, dtype=np.float64)
    t2 = np.asarray(t2_k, dtype=np.float64)
    tmean = t2 + MSLP_LAPSE_K_PER_M * h / 2.0
    return ps * np.exp(MSLP_G * h / (MSLP_R * tmean))


@dataclass
class SurfaceFields:
    """One model's surface reading on its own rectilinear grid."""

    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    t2_k: np.ndarray
    q2_kg_kg: np.ndarray
    u10_m_s: np.ndarray
    v10_m_s: np.ndarray
    surface_pressure_pa: np.ndarray
    terrain_m: np.ndarray
    land: np.ndarray
    source: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        lat = np.asarray(self.latitude_deg, dtype=np.float64).ravel()
        lon = np.asarray(self.longitude_deg, dtype=np.float64).ravel()
        shape = (lat.size, lon.size)
        for name in ("t2_k", "q2_kg_kg", "u10_m_s", "v10_m_s", "surface_pressure_pa", "terrain_m"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.shape != shape:
                raise ValueError(f"{name} has shape {arr.shape}, the grid is {shape}")
            setattr(self, name, arr)
        land = np.asarray(self.land, dtype=bool)
        if land.shape != shape:
            raise ValueError(f"land has shape {land.shape}, the grid is {shape}")
        self.land = land
        self.latitude_deg, self.longitude_deg = lat, lon

    def variable(self, name: str) -> np.ndarray:
        """One scored surface variable in seam units (K, K, m/s, Pa)."""
        if name == "temperature_2m":
            return self.t2_k
        if name == "dewpoint_2m":
            return dewpoint_from_specific_humidity(self.q2_kg_kg, self.surface_pressure_pa)
        if name == "wind_speed_10m":
            return np.hypot(self.u10_m_s, self.v10_m_s)
        if name == "mslp":
            return mslp_reduction(self.surface_pressure_pa, self.terrain_m, self.t2_k)
        raise KeyError(f"unknown surface variable {name!r}; the scorecard reads {SURFACE_VARIABLES}")

    def sampler(self) -> RectilinearSampler:
        return RectilinearSampler(self.latitude_deg, self.longitude_deg)


def surface_fields_from_checkpoint_arrays(
    arrays: dict[str, np.ndarray], transform, surface_geopotential: np.ndarray, *, source: dict | None = None,
    checkpoint: str = "checkpoint",
) -> SurfaceFields:
    """The model's own surface books from a checkpoint's arrays: refused by
    name when the checkpoint does not carry them."""
    missing = [name for name in MODEL_SURFACE_ARRAYS if name not in arrays]
    if missing:
        raise KeyError(
            f"{checkpoint}: the checkpoint carries no {', '.join(missing)}; the 2 m and 10 m "
            "fields are the physics suite's own SFCDIAGS and surface-layer books and nothing "
            "is substituted for them"
        )
    grid = transform.grid
    logps = np.asarray(
        transform.inverse(np.asarray(arrays["atmosphere__log_surface_pressure"]).astype(np.complex128)),
        dtype=np.float64,
    )
    phi = np.asarray(surface_geopotential, dtype=np.float64)
    if phi.shape != logps.shape:
        raise ValueError(f"surface geopotential shape {phi.shape} is not the grid's {logps.shape}")
    return SurfaceFields(
        latitude_deg=grid.latitude_deg,
        longitude_deg=grid.longitude_deg,
        t2_k=arrays["physics__t2"],
        q2_kg_kg=arrays["physics__q2"],
        u10_m_s=arrays["physics__u10"],
        v10_m_s=arrays["physics__v10"],
        surface_pressure_pa=np.exp(logps),
        terrain_m=phi / GRAVITY_M_S2,
        land=np.asarray(arrays["surface__land_fraction"], dtype=np.float64) >= 0.5,
        source=dict(source or {}),
    )


def build_streaming_transform(receipt: dict):
    """The run's transform without its resident Legendre tables.  The
    surface reading synthesizes one field (log surface pressure); the
    streaming basis does that in well under a second where the resident
    table build takes minutes on a CPU host."""
    from woof.globe.spectral.transform import SphericalHarmonicTransform

    cfg = receipt["config"]
    block = receipt["transform"]
    return SphericalHarmonicTransform.create(
        int(cfg["truncation"]),
        nlat=int(block["nlat"]),
        nlon=int(block["nlon"]),
        dealias_factor=float(cfg["dealias_factor"]),
        radius_m=float(block["radius_m"]),
        backend="numpy",
        precision="float64",
        streaming=True,
    )


def model_surface_fields(run_dir: str | Path, checkpoint: str | Path, *, surface_geopotential: np.ndarray) -> SurfaceFields:
    """One checkpoint's surface reading through the run's own (streaming)
    transform.  Only the surface arrays are loaded; a checkpoint lacking one
    is refused by name before anything is read."""
    receipt = read_receipt(Path(run_dir))
    transform = build_streaming_transform(receipt)
    with np.load(checkpoint, allow_pickle=False) as archive:
        if "__metadata__" not in archive:
            raise ValueError(f"checkpoint {checkpoint} has no metadata")
        metadata = json.loads(str(archive["__metadata__"].item()))
        names = set(archive.files)
        missing = [name for name in MODEL_SURFACE_ARRAYS if name not in names]
        if missing:
            raise KeyError(
                f"{checkpoint}: the checkpoint carries no {', '.join(missing)}; the 2 m and 10 m "
                "fields are the physics suite's own SFCDIAGS and surface-layer books and nothing "
                "is substituted for them"
            )
        arrays = {name: np.array(archive[name]) for name in MODEL_SURFACE_ARRAYS}
    valid = receipt_start_utc(receipt) + timedelta(seconds=float(metadata["time_s"]))
    source = {
        "kind": "model", "run_dir": str(run_dir), "checkpoint": str(checkpoint),
        "checkpoint_self_sha256": metadata.get("self_sha256"),
        "step": int(metadata["step"]), "time_s": float(metadata["time_s"]),
        "valid_time": valid.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_hash": receipt.get("config_hash"),
        "t2_q2": "physics__t2 / physics__q2 (SFCDIAGS on land after the land surface; the surface layer over water)",
        "u10_v10": "physics__u10 / physics__v10 (the surface layer)",
        "terrain": "the run's surface geopotential (analysis orography regridded and truncated) / g 9.80616",
        "land": "surface__land_fraction >= 0.5",
    }
    return surface_fields_from_checkpoint_arrays(
        arrays, transform, surface_geopotential, source=source, checkpoint=str(checkpoint)
    )


def _frame_axes(frame) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(frame.latitude, dtype=np.float64)
    lon = np.asarray(frame.longitude, dtype=np.float64)
    if lat.ndim == 2:
        lat, lon = lat[:, 0], lon[0, :]
    return lat, lon


def _frame_plane(frame, name: str) -> np.ndarray:
    values = np.asarray(frame.fields[name].values, dtype=np.float64)
    if values.ndim == 3:
        values = values[-1]
    if values.ndim != 2:
        raise ValueError(f"{name} has {values.ndim} dimensions, not a plane")
    return values


def product_surface_fields(frame, *, path: str | None = None) -> SurfaceFields:
    """The GFS product's own surface fields, on its own grid, refused by
    name when the decoded frame lacks one."""
    missing = [name for name in PRODUCT_SURFACE_FIELDS if name not in frame.fields]
    if missing:
        raise ValueError(
            f"{path or 'product'}: the decoded product lacks {', '.join(missing)}; the scorecard "
            "reads the product's own 2 m, 10 m, surface pressure, terrain and land fields and "
            "substitutes nothing"
        )
    lat, lon = _frame_axes(frame)
    valid_time = getattr(frame, "valid_time", None)
    cycle = getattr(frame, "source_cycle", None)
    lead_h = None if valid_time is None or cycle is None else (valid_time - cycle).total_seconds() / 3600.0
    return SurfaceFields(
        latitude_deg=lat, longitude_deg=lon,
        t2_k=_frame_plane(frame, "air_temperature_2m"),
        q2_kg_kg=_frame_plane(frame, "specific_humidity_2m"),
        u10_m_s=_frame_plane(frame, "eastward_wind_10m"),
        v10_m_s=_frame_plane(frame, "northward_wind_10m"),
        surface_pressure_pa=_frame_plane(frame, "surface_pressure"),
        terrain_m=_frame_plane(frame, "terrain_height"),
        land=_frame_plane(frame, "land_fraction") >= 0.5,
        source={
            "kind": "analysis" if lead_h == 0.0 else "forecast",
            "product": Path(path).name if path else "product", "path": path, "lead_h": lead_h,
            "valid_time": None if valid_time is None else valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cycle": None if cycle is None else cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input_sha256": dict(getattr(frame, "input_sha256", {}) or {}),
            "mapping_sha256": getattr(frame, "mapping_sha256", None),
            "t2_q2": "the product's own 2 m temperature and specific humidity (the producing model's surface scheme's diagnosis; a product without a 2 m humidity field carries its mapping's derivation from the product's own 2 m dewpoint)",
            "u10_v10": "the product's own 10 m wind",
            "terrain": "the product's surface height (gpm)",
            "land": "the product's land flag >= 0.5",
        },
    )


# --------------------------------------------------------------------------
# station scoring
# --------------------------------------------------------------------------


def _stats(residuals: np.ndarray) -> dict[str, float | int | None]:
    r = np.asarray(residuals, dtype=np.float64)
    if r.size == 0:
        return {"n": 0, "bias": None, "rmse": None, "mae": None}
    return {
        "n": int(r.size), "bias": float(np.mean(r)),
        "rmse": float(np.sqrt(np.mean(r * r))), "mae": float(np.mean(np.abs(r))),
    }


def admit_stations(fields: SurfaceFields, observations, valid_times: list[str], *, elevation_tolerance_m: float,
                   minimum_reporting_fraction: float, match_tolerance_seconds: int, maximum_screen_fraction: float):
    """The verification library's admission on one model's own land mask
    and terrain, with positions from this sampler."""
    from woof.verify.obs import stations as st

    sampler = fields.sampler()
    ids = [s.station_id for s in observations.stations]
    lats = np.array([s.latitude for s in observations.stations], dtype=np.float64)
    lons = np.array([s.longitude for s in observations.stations], dtype=np.float64)
    x, y, inside = sampler.library_positions(lats, lons)
    positions = {
        sid: st.StationPosition(station_id=sid, x=float(x[k]), y=float(y[k]))
        for k, sid in enumerate(ids) if inside[k]
    }
    interior = np.ones(fields.land.shape, dtype=bool)
    return st.freeze_station_set(
        observations.stations, positions, observations=observations, valid_times=list(valid_times),
        interior_mask=interior, land_mask=fields.land, terrain_m=fields.terrain_m,
        elevation_tolerance_m=float(elevation_tolerance_m),
        minimum_reporting_fraction=float(minimum_reporting_fraction),
        match_tolerance_seconds=int(match_tolerance_seconds),
        maximum_screen_fraction=float(maximum_screen_fraction),
    )


def score_surface_stations(
    models: dict[str, SurfaceFields], observations, valid_time: str, *,
    elevation_tolerance_m: float | None = None, minimum_reporting_fraction: float | None = None,
    match_tolerance_seconds: int | None = None, maximum_screen_fraction: float | None = None,
) -> dict[str, object]:
    """Every model against the same stations and the same reports at one
    valid time; the scored set is the intersection of the models' admitted
    sets.  ``observations`` is a ``StationObsSet`` (AsosSurfaceSource.
    observations([valid_time]))."""
    from woof.verify.obs import registration, stations as st

    tol_m = registration.DEFAULT_ELEVATION_TOLERANCE_M if elevation_tolerance_m is None else elevation_tolerance_m
    frac = registration.DEFAULT_MINIMUM_REPORTING_FRACTION if minimum_reporting_fraction is None else minimum_reporting_fraction
    tol_s = registration.DEFAULT_MATCH_TOLERANCE_SECONDS if match_tolerance_seconds is None else match_tolerance_seconds
    screen = registration.DEFAULT_MAXIMUM_SCREEN_FRACTION if maximum_screen_fraction is None else maximum_screen_fraction
    if not models:
        raise ValueError("no models to score")
    frozen = {
        label: admit_stations(
            fields, observations, [valid_time], elevation_tolerance_m=tol_m, minimum_reporting_fraction=frac,
            match_tolerance_seconds=tol_s, maximum_screen_fraction=screen,
        )
        for label, fields in models.items()
    }
    common = sorted(set.intersection(*[set(f.station_ids) for f in frozen.values()]))
    matched = st.match_reports(observations, [valid_time], tolerance_seconds=tol_s)
    by_id = {s.station_id: s for s in observations.stations}
    lats = np.array([by_id[sid].latitude for sid in common], dtype=np.float64)
    lons = np.array([by_id[sid].longitude for sid in common], dtype=np.float64)
    samples: dict[str, dict[str, np.ndarray]] = {}
    for label, fields in models.items():
        sampler = fields.sampler()
        samples[label] = {v: sampler.sample(fields.variable(v), lats, lons) for v in SURFACE_VARIABLES}
    residuals: dict[str, dict[str, list[float]]] = {label: {v: [] for v in SURFACE_VARIABLES} for label in models}
    refused: dict[str, int] = {v: 0 for v in SURFACE_VARIABLES}
    unreported: dict[str, int] = {v: 0 for v in SURFACE_VARIABLES}
    rows: list[dict[str, object]] = []
    for k, sid in enumerate(common):
        report = matched.get((sid, valid_time))
        if report is None:
            continue
        failed = st.screen_report(report)
        station = by_id[sid]
        row: dict[str, object] = {
            "station_id": sid, "lat": station.latitude, "lon": station.longitude,
            "elevation_m": station.elevation_m, "report_time": report.valid_time,
        }
        for v in SURFACE_VARIABLES:
            tag, _unit, factor = SURFACE_DISPLAY[v]
            if v in failed or v not in report.values:
                unreported[v] += 1
                continue
            values = {label: float(samples[label][v][k]) for label in models}
            if not all(math.isfinite(m) for m in values.values()):
                refused[v] += 1
                continue
            obs = float(report.values[v])
            row[f"{tag}_obs"] = obs * factor
            for label, m in values.items():
                residuals[label][v].append(m - obs)
                row[f"{tag}_{label}"] = m * factor
        rows.append(row)
    scores: dict[str, dict[str, dict]] = {}
    for label in models:
        scores[label] = {}
        for v in SURFACE_VARIABLES:
            tag, unit, factor = SURFACE_DISPLAY[v]
            s = _stats(np.asarray(residuals[label][v], dtype=np.float64) * factor)
            s.update({"display": tag, "unit": unit})
            scores[label][v] = s
    return {
        "schema": SCHEMA,
        "kind": "surface-stations",
        "valid_time": valid_time,
        "observations": {
            "provenance": observations.provenance.record() if hasattr(observations.provenance, "record") else {},
            "station_count": len(observations.stations),
            "report_count": len(observations.reports),
        },
        "admission": {
            label: {
                "admitted": len(f.station_ids),
                "dropped_by_reason": f.record()["dropped_by_reason"],
                "parameters": f.record()["parameters"],
            }
            for label, f in frozen.items()
        },
        "common_station_count": len(common),
        "unreported_or_screened_by_variable": unreported,
        "stencil_refused_by_variable": refused,
        "interpolation": "bilinear in latitude and longitude on each model's own grid; admission at the nearest cell",
        "residual": "model minus observation; MSLP rows in hPa, the rest in seam units",
        "models": {label: fields.source for label, fields in models.items()},
        "scores": scores,
        "stations": rows,
    }


# --------------------------------------------------------------------------
# radiosondes: the IGRA2 record
# --------------------------------------------------------------------------


@dataclass
class SoundingLevel:
    level_type: int
    pressure_pa: float | None
    height_m: float | None
    temperature_k: float | None
    wind_direction_deg: float | None
    wind_speed_m_s: float | None


@dataclass
class Sounding:
    station_id: str
    nominal: datetime
    release_hhmm: str
    latitude: float
    longitude: float
    levels: list[SoundingLevel]

    def mandatory(self, level_pa: float) -> dict[str, float | None] | None:
        """The row at exactly ``level_pa`` (standard level first): height,
        temperature, u and v; None per value when IGRA2 marks it missing,
        None as a whole when the sounding carries no such row."""
        rows = [lv for lv in self.levels if lv.pressure_pa is not None and abs(lv.pressure_pa - level_pa) < 0.5]
        if not rows:
            return None
        rows.sort(key=lambda lv: 0 if lv.level_type == 1 else 1)
        lv = rows[0]
        u, v = wind_components(lv.wind_direction_deg, lv.wind_speed_m_s)
        return {"z": lv.height_m, "t": lv.temperature_k, "u": u, "v": v, "wspd": lv.wind_speed_m_s}


def wind_components(direction_deg: float | None, speed_m_s: float | None) -> tuple[float | None, float | None]:
    """Meteorological direction (degrees the wind blows FROM) and speed to
    eastward and northward components."""
    if direction_deg is None or speed_m_s is None:
        return None, None
    rad = math.radians(float(direction_deg))
    return -float(speed_m_s) * math.sin(rad), -float(speed_m_s) * math.cos(rad)


def _igra_int(text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    value = int(text)
    return None if value in IGRA2_MISSING else value


def parse_igra2(text: str, *, wanted: set[datetime] | None = None) -> list[Sounding]:
    """Soundings from one IGRA2 data file (the ``igra2-data-format.txt``
    fixed columns); ``wanted`` keeps only the nominal instants listed."""
    soundings: list[Sounding] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("#"):
            index += 1
            continue
        station = line[1:12].strip()
        year, month, day, hour = int(line[13:17]), int(line[18:20]), int(line[21:23]), int(line[24:26])
        reltime = line[27:31]
        numlev = int(line[32:36])
        lat = int(line[55:62]) / 10_000.0
        lon = int(line[63:71]) / 10_000.0
        index += 1
        block = lines[index:index + numlev]
        index += numlev
        if hour == 99:
            continue
        nominal = datetime(year, month, day, hour)
        if wanted is not None and nominal not in wanted:
            continue
        levels: list[SoundingLevel] = []
        for row in block:
            if len(row) < 51:
                row = row.ljust(51)
            # igra2-data-format.txt columns (1-indexed): LVLTYP1 1, LVLTYP2 2,
            # ETIME 4-8, PRESS 10-15, PFLAG 16, GPH 17-21, ZFLAG 22, TEMP 23-27,
            # TFLAG 28, RH 29-33, DPDP 35-39, WDIR 41-45, WSPD 47-51; verified
            # against the archive's own 2026-09-01 12Z records.
            level_type = int(row[0])
            press = _igra_int(row[9:15])
            gph = _igra_int(row[16:21])
            temp = _igra_int(row[22:27])
            wdir = _igra_int(row[40:45])
            wspd = _igra_int(row[46:51])
            levels.append(SoundingLevel(
                level_type=level_type,
                pressure_pa=None if press is None else float(press),
                height_m=None if gph is None else float(gph),
                temperature_k=None if temp is None else temp / 10.0 + 273.15,
                wind_direction_deg=None if wdir is None else float(wdir),
                wind_speed_m_s=None if wspd is None else wspd / 10.0,
            ))
        soundings.append(Sounding(
            station_id=station, nominal=nominal, release_hhmm=reltime, latitude=lat, longitude=lon, levels=levels,
        ))
    return soundings


def extract_igra2_zips(zip_dir: str | Path, wanted: list[datetime], *, levels_pa=UPPER_LEVELS_PA) -> dict[str, object]:
    """Every sounding at the wanted nominal instants across a directory of
    IGRA2 station zips, reduced to the mandatory levels, with the fetch
    manifest (size and SHA-256 of every zip read)."""
    zip_dir = Path(zip_dir)
    wanted_set = set(wanted)
    manifest = []
    records = []
    for path in sorted(zip_dir.glob("*.zip")):
        digest = _sha256_file(path)
        manifest.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest})
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                with archive.open(member) as stream:
                    text = io.TextIOWrapper(stream, encoding="ascii", errors="replace").read()
                for snd in parse_igra2(text, wanted=wanted_set):
                    records.append({
                        "station_id": snd.station_id,
                        "nominal": snd.nominal.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "release_hhmm": snd.release_hhmm,
                        "latitude": snd.latitude, "longitude": snd.longitude,
                        "levels": {str(int(level)): snd.mandatory(level) for level in levels_pa},
                    })
    return {
        "schema": SCHEMA, "kind": "igra2-mandatory-levels",
        "source": "IGRA2 year-to-date station files (NCEI, access/data-y2d)",
        "wanted": [w.strftime("%Y-%m-%dT%H:%M:%SZ") for w in wanted],
        "levels_pa": [float(v) for v in levels_pa],
        "zip_count": len(manifest), "zip_bytes": int(sum(m["bytes"] for m in manifest)),
        "manifest": manifest, "soundings": records,
    }


# --------------------------------------------------------------------------
# upper-air fields of one model at the target levels, on its own grid
# --------------------------------------------------------------------------


@dataclass
class LevelFields:
    """z, t, u, v at the target pressures on one rectilinear grid, NaN
    where the column is refused at that level."""

    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    levels_pa: tuple[float, ...]
    fields: dict[str, dict[float, np.ndarray]]
    source: dict = field(default_factory=dict)

    def sampler(self) -> RectilinearSampler:
        return RectilinearSampler(self.latitude_deg, self.longitude_deg)

    @classmethod
    def from_pressure_level_fields(cls, plf: PressureLevelFields, *, levels_pa=UPPER_LEVELS_PA) -> "LevelFields":
        missing = [level for level in levels_pa if level not in plf.levels_pa]
        if missing:
            raise ValueError(f"the model's pressure-level fields lack {missing} Pa (they hold {plf.levels_pa})")
        fields: dict[str, dict[float, np.ndarray]] = {name: {} for name in ("z", "t", "u", "v")}
        for level in levels_pa:
            mask = plf.valid[level]
            for name in fields:
                fields[name][level] = np.where(mask, plf.fields[name][level], np.nan)
        return cls(
            latitude_deg=plf.latitude_deg, longitude_deg=plf.longitude_deg,
            levels_pa=tuple(float(v) for v in levels_pa), fields=fields, source=dict(plf.source),
        )


def product_level_fields(frame, *, levels_pa=UPPER_LEVELS_PA, path: str | None = None) -> LevelFields:
    """The product's own isobaric levels on its own grid, refused where its
    surface pressure sits below the level."""
    missing = [name for name in PRODUCT_LEVEL_FIELDS if name not in frame.fields]
    if missing:
        raise ValueError(f"{path or 'product'}: the decoded product lacks {', '.join(missing)}")
    if getattr(frame, "vertical_kind", "pressure") != "pressure":
        raise ValueError(f"reference vertical kind {frame.vertical_kind!r} is not 'pressure'")
    lat, lon = _frame_axes(frame)
    levels = np.asarray(frame.vertical_values, dtype=np.float64)
    ps = _frame_plane(frame, "surface_pressure")
    fields: dict[str, dict[float, np.ndarray]] = {name: {} for name in ("z", "t", "u", "v")}
    names = {"z": "geopotential_height", "t": "air_temperature", "u": "eastward_wind", "v": "northward_wind"}
    for level in levels_pa:
        index = np.where(np.abs(levels - float(level)) < 0.5)[0]
        if index.size != 1:
            raise ValueError(
                f"{path or 'product'}: no {float(level):g} Pa level in the product (levels {levels.tolist()}); "
                "the scorecard reads the product's own isobaric levels, never an interpolation"
            )
        k = int(index[0])
        inside = ps > float(level)
        for short, name in names.items():
            values = np.asarray(frame.fields[name].values, dtype=np.float64)[k]
            fields[short][level] = np.where(inside, values, np.nan)
    valid_time = getattr(frame, "valid_time", None)
    cycle = getattr(frame, "source_cycle", None)
    lead_h = None if valid_time is None or cycle is None else (valid_time - cycle).total_seconds() / 3600.0
    return LevelFields(
        latitude_deg=lat, longitude_deg=lon, levels_pa=tuple(float(v) for v in levels_pa), fields=fields,
        source={
            "kind": "analysis" if lead_h == 0.0 else "forecast",
            "product": Path(path).name if path else "product", "path": path, "lead_h": lead_h,
            "valid_time": None if valid_time is None else valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cycle": None if cycle is None else cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input_sha256": dict(getattr(frame, "input_sha256", {}) or {}),
            "levels": "the product's own isobaric levels; refused where surface pressure <= level",
            "height": "the product's geopotential height (gpm)",
        },
    )


def model_level_fields(run_dir: str | Path, checkpoint: str | Path, *, surface_geopotential: np.ndarray,
                       cache_dir: str | Path | None = None, phi_digest: str = "") -> LevelFields:
    """The run's pressure-level fields at the upper-air scorecard's levels,
    through its ModelReader (or that instrument's cache when the checkpoint
    and surface geopotential digests match)."""
    receipt = read_receipt(Path(run_dir))
    cache = None
    if cache_dir is not None and phi_digest:
        cache = model_cache_path(cache_dir, checkpoint, phi_digest)
    if cache is not None and cache.exists():
        plf = PressureLevelFields.load(cache)
    else:
        reader = ModelReader(receipt, surface_geopotential, levels_pa=LEVELS_PA)
        plf = reader.sample(checkpoint)
        if cache is not None:
            plf.save(cache)
    out = LevelFields.from_pressure_level_fields(plf)
    out.source["run_dir"] = str(run_dir)
    out.source["interpolation"] = (
        "linear in ln p between bracketing full levels; geopotential hydrostatic from the level above; "
        "below-ground targets refused per column (upper_air_scorecard.interpolate_column_fields)"
    )
    out.source["height"] = f"phi / g with the model's g {GRAVITY_M_S2}"
    return out


# --------------------------------------------------------------------------
# sounding scoring
# --------------------------------------------------------------------------


def _hemisphere_mask(latitudes: np.ndarray, name: str) -> np.ndarray:
    south, north = HEMISPHERES[name]
    return (latitudes >= south) & (latitudes <= north)


def _screened(value: float | None, name: str) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    low, high = SOUNDING_SCREEN.get(name, (-math.inf, math.inf))
    return None if not (low <= float(value) <= high) else float(value)


def score_soundings(models: dict[str, LevelFields], soundings: list[dict], *, targets=UPPER_TARGETS) -> dict[str, object]:
    """Every model at the same sites; a site counts for a target only when
    the observation carries it and every model's sample is finite."""
    if not models:
        raise ValueError("no models to score")
    sites: dict[str, dict] = {}
    duplicates = 0
    for record in soundings:
        sid = record["station_id"]
        if sid in sites:
            duplicates += 1
            continue
        sites[sid] = record
    ids = sorted(sites)
    lats = np.array([sites[s]["latitude"] for s in ids], dtype=np.float64)
    lons = np.array([sites[s]["longitude"] for s in ids], dtype=np.float64)
    samples: dict[str, dict[str, dict[float, np.ndarray]]] = {}
    for label, lf in models.items():
        sampler = lf.sampler()
        samples[label] = {
            name: {level: sampler.sample(lf.fields[name][level], lats, lons) for level in lf.levels_pa}
            for name in ("z", "t", "u", "v")
        }
    screened: dict[str, int] = {}
    scores: dict[str, dict[str, dict[str, dict]]] = {label: {} for label in models}
    site_rows: dict[str, list[dict]] = {}
    for score_name, kind, level in targets:
        key = str(int(level))
        obs_values = []
        keep = []
        for k, sid in enumerate(ids):
            row = sites[sid]["levels"].get(key)
            if row is None:
                continue
            if kind == "wind":
                u, v = row.get("u"), row.get("v")
                speed = _screened(row.get("wspd"), "wspd")
                if u is None or v is None or speed is None:
                    if row.get("wspd") is not None and speed is None:
                        screened[score_name] = screened.get(score_name, 0) + 1
                    continue
                if not all(math.isfinite(samples[label]["u"][level][k]) and math.isfinite(samples[label]["v"][level][k]) for label in models):
                    continue
                obs_values.append((float(u), float(v)))
            else:
                value = _screened(row.get(kind), kind)
                if value is None:
                    if row.get(kind) is not None:
                        screened[score_name] = screened.get(score_name, 0) + 1
                    continue
                if not all(math.isfinite(samples[label][kind][level][k]) for label in models):
                    continue
                obs_values.append(value)
            keep.append(k)
        keep_arr = np.asarray(keep, dtype=np.int64)
        site_rows[score_name] = []
        for label in models:
            per_hemisphere = {}
            for hemi in HEMISPHERES:
                mask = _hemisphere_mask(lats[keep_arr], hemi) if keep_arr.size else np.zeros(0, dtype=bool)
                if kind == "wind":
                    ou = np.array([o[0] for o in obs_values], dtype=np.float64)[mask]
                    ov = np.array([o[1] for o in obs_values], dtype=np.float64)[mask]
                    mu = samples[label]["u"][level][keep_arr][mask]
                    mv = samples[label]["v"][level][keep_arr][mask]
                    n = int(mu.size)
                    if n == 0:
                        per_hemisphere[hemi] = {"n": 0, "rmsve": None, "speed_bias": None, "u_bias": None, "v_bias": None}
                    else:
                        du, dv = mu - ou, mv - ov
                        per_hemisphere[hemi] = {
                            "n": n,
                            "rmsve": float(np.sqrt(np.mean(du * du + dv * dv))),
                            "speed_bias": float(np.mean(np.hypot(mu, mv) - np.hypot(ou, ov))),
                            "u_bias": float(np.mean(du)), "v_bias": float(np.mean(dv)),
                        }
                else:
                    obs_arr = np.asarray(obs_values, dtype=np.float64)[mask]
                    mod = samples[label][kind][level][keep_arr][mask]
                    per_hemisphere[hemi] = _stats(mod - obs_arr)
            scores[label][score_name] = per_hemisphere
        for j, k in enumerate(keep):
            row = {"station_id": ids[k], "lat": float(lats[k]), "lon": float(lons[k])}
            if kind == "wind":
                row["obs_u"], row["obs_v"] = obs_values[j]
                for label in models:
                    row[f"{label}_u"] = float(samples[label]["u"][level][k])
                    row[f"{label}_v"] = float(samples[label]["v"][level][k])
            else:
                row["obs"] = obs_values[j]
                for label in models:
                    row[label] = float(samples[label][kind][level][k])
            site_rows[score_name].append(row)
    return {
        "schema": SCHEMA,
        "kind": "soundings",
        "site_count": len(ids),
        "duplicate_soundings_skipped": duplicates,
        "screened_by_target": screened,
        "hemispheres": {name: list(edges) for name, edges in HEMISPHERES.items()},
        "targets": [{"name": n, "field": k, "level_pa": l} for n, k, l in targets],
        "interpolation": "bilinear in latitude and longitude on each model's own grid at the site; a refused stencil corner refuses the site",
        "residual": "model minus observation",
        "models": {label: lf.source for label, lf in models.items()},
        "scores": scores,
        "sites": site_rows,
    }


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


class _FakeFrame:
    def __init__(self, latitude, longitude, fields, *, vertical_values=None, valid_time=None, source_cycle=None):
        self.latitude = np.asarray(latitude, dtype=np.float64)
        self.longitude = np.asarray(longitude, dtype=np.float64)
        self.fields = {name: _FakeField(values) for name, values in fields.items()}
        self.vertical_kind = "pressure"
        self.vertical_values = None if vertical_values is None else np.asarray(vertical_values, dtype=np.float64)
        self.valid_time = valid_time
        self.source_cycle = source_cycle
        self.input_sha256 = {}
        self.mapping_sha256 = None


class _FakeField:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)


class _FakeReport:
    def __init__(self, station_id, valid_time, values):
        self.station_id = station_id
        self.valid_time = valid_time
        self.values = dict(values)
        self.flags = ()


class _FakeProvenance:
    def record(self):
        return {"source": "synthetic"}


class _FakeObsSet:
    def __init__(self, stations, reports):
        self.stations = tuple(stations)
        self.reports = tuple(reports)
        self.provenance = _FakeProvenance()

    def by_station(self):
        out: dict[str, list] = {}
        for report in self.reports:
            out.setdefault(report.station_id, []).append(report)
        return out


def synthetic_surface_fields(latitude_deg, longitude_deg, *, seed: int = 0) -> SurfaceFields:
    """Smooth planted surface fields on a grid, land everywhere, terrain 0
    to 300 m."""
    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    lon2, lat2 = np.meshgrid(lon, lat)
    rng = np.random.default_rng(seed)
    t2 = 288.0 - 0.4 * np.abs(lat2) + 2.0 * np.cos(np.deg2rad(lon2))
    terrain = 150.0 + 150.0 * np.sin(np.deg2rad(lon2)) * np.cos(np.deg2rad(lat2))
    ps = 101325.0 * np.exp(-terrain / 8400.0) + 200.0 * rng.standard_normal(lat2.shape)
    # 60 percent of saturation, so every planted dewpoint sits below its
    # temperature and the station screen keeps every station
    es = 611.2 * np.exp(17.67 * (t2 - 273.15) / (t2 - 29.65))
    q2 = 0.6 * 0.622 * es / (ps - 0.378 * es)
    u10 = 3.0 * np.sin(np.deg2rad(2.0 * lat2)) + 1.0
    v10 = 2.0 * np.cos(np.deg2rad(lon2))
    return SurfaceFields(
        latitude_deg=lat, longitude_deg=lon, t2_k=t2, q2_kg_kg=q2, u10_m_s=u10, v10_m_s=v10,
        surface_pressure_pa=ps, terrain_m=terrain, land=np.ones(lat2.shape, dtype=bool), source={"kind": "synthetic"},
    )


def synthetic_stations(fields: SurfaceFields, count: int, *, seed: int = 1):
    """Stations at random interior points of the grid, at the model's own
    terrain (so the admission keeps every one)."""
    from woof.verify.obs.contracts import Station

    rng = np.random.default_rng(seed)
    lat = fields.latitude_deg
    lo, hi = float(min(lat[0], lat[-1])) + 1.0, float(max(lat[0], lat[-1])) - 1.0
    lats = rng.uniform(lo, hi, count)
    lons = rng.uniform(-179.0, 179.0, count)   # the station contract wants [-180, 180)
    terrain = fields.sampler().sample(fields.terrain_m, lats, lons)
    return [
        Station(station_id=f"S{k:04d}", latitude=float(lats[k]), longitude=float(lons[k]), elevation_m=float(terrain[k]))
        for k in range(count)
    ]


def synthetic_observations(fields: SurfaceFields, stations, valid_time: str, offsets: dict[str, float] | None = None):
    """Pseudo-reports equal to the model's own samples plus planted offsets."""
    offsets = dict(offsets or {})
    lats = np.array([s.latitude for s in stations])
    lons = np.array([s.longitude for s in stations])
    sampler = fields.sampler()
    values = {v: sampler.sample(fields.variable(v), lats, lons) for v in SURFACE_VARIABLES}
    reports = []
    for k, s in enumerate(stations):
        reports.append(_FakeReport(s.station_id, valid_time, {
            v: float(values[v][k]) + offsets.get(v, 0.0) for v in SURFACE_VARIABLES
        }))
    return _FakeObsSet(stations, reports)


def synthetic_level_fields(latitude_deg, longitude_deg, *, levels_pa=UPPER_LEVELS_PA, offsets=None) -> LevelFields:
    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    lon2, lat2 = np.meshgrid(lon, lat)
    offsets = dict(offsets or {})
    fields: dict[str, dict[float, np.ndarray]] = {name: {} for name in ("z", "t", "u", "v")}
    for level in levels_pa:
        scale = math.log(101325.0 / level)
        fields["z"][level] = 7000.0 * scale + 100.0 * np.cos(np.deg2rad(lat2)) * np.sin(np.deg2rad(lon2)) + offsets.get(("z", level), 0.0)
        fields["t"][level] = 288.0 - 50.0 * scale / math.log(4.0) + 5.0 * np.cos(np.deg2rad(lat2)) + offsets.get(("t", level), 0.0)
        fields["u"][level] = 10.0 * scale * np.sin(np.deg2rad(2.0 * lat2)) + offsets.get(("u", level), 0.0)
        fields["v"][level] = 3.0 * np.cos(np.deg2rad(lon2)) + offsets.get(("v", level), 0.0)
    return LevelFields(latitude_deg=lat, longitude_deg=lon, levels_pa=tuple(float(v) for v in levels_pa), fields=fields, source={"kind": "synthetic"})


def synthetic_soundings(lf: LevelFields, count: int, *, seed: int = 2, offsets=None) -> list[dict]:
    """Pseudo-soundings equal to the model's own samples plus planted offsets."""
    rng = np.random.default_rng(seed)
    lat = lf.latitude_deg
    lo, hi = float(min(lat[0], lat[-1])) + 1.0, float(max(lat[0], lat[-1])) - 1.0
    lats = rng.uniform(lo, hi, count)
    lons = rng.uniform(0.0, 360.0, count)
    sampler = lf.sampler()
    offsets = dict(offsets or {})
    records = []
    for k in range(count):
        levels = {}
        for level in lf.levels_pa:
            u = float(sampler.sample(lf.fields["u"][level], lats[k:k + 1], lons[k:k + 1])[0]) + offsets.get(("u", level), 0.0)
            v = float(sampler.sample(lf.fields["v"][level], lats[k:k + 1], lons[k:k + 1])[0]) + offsets.get(("v", level), 0.0)
            levels[str(int(level))] = {
                "z": float(sampler.sample(lf.fields["z"][level], lats[k:k + 1], lons[k:k + 1])[0]) + offsets.get(("z", level), 0.0),
                "t": float(sampler.sample(lf.fields["t"][level], lats[k:k + 1], lons[k:k + 1])[0]) + offsets.get(("t", level), 0.0),
                "u": u, "v": v, "wspd": math.hypot(u, v),
            }
        records.append({"station_id": f"R{k:04d}", "nominal": "2026-01-01T00:00:00Z", "release_hhmm": "2300",
                        "latitude": float(lats[k]), "longitude": float(lons[k]), "levels": levels})
    return records


def igra2_header(station_id: str, year: int, month: int, day: int, hour: int, reltime: str, numlev: int,
                 latitude_deg: float, longitude_deg: float) -> str:
    """One IGRA2 header record in the archive's fixed columns."""
    return (f"#{station_id:<11s} {year:04d} {month:02d} {day:02d} {hour:02d} {reltime:>4s} {numlev:>4d} "
            f"{'ncdc-gts':<8s} {'':<8s} {int(round(latitude_deg * 10_000)):>7d} {int(round(longitude_deg * 10_000)):>8d}")


def igra2_row(level_type1: int, level_type2: int, press: int, gph: int, temp_tenths: int, rh_tenths: int,
              dpdp_tenths: int, wdir: int, wspd_tenths: int, *, pflag: str = " ", zflag: str = "B", tflag: str = "B") -> str:
    """One IGRA2 data record in the archive's fixed columns (-9999 missing)."""
    return (f"{level_type1}{level_type2} {-9999:>5d} {press:>6d}{pflag}{gph:>5d}{zflag}{temp_tenths:>5d}{tflag}"
            f"{rh_tenths:>5d} {dpdp_tenths:>5d} {wdir:>5d} {wspd_tenths:>5d}")


#: Two synthetic soundings in the archive's own fixed columns.
IGRA2_SYNTHETIC = "\n".join([
    igra2_header("USM00072469", 2026, 9, 1, 12, "1100", 4, 39.75, -104.83),
    igra2_row(2, 1, 84400, 1611, 298, 450, 50, 180, 35, pflag="B"),
    igra2_row(1, 0, 85000, 1512, 212, 600, 100, 270, 100),
    igra2_row(1, 0, 50000, 5860, -105, 300, 200, 360, 50),
    igra2_row(1, 0, 25000, -9999, -9999, -9999, -9999, 250, 300, zflag=" ", tflag=" "),
    igra2_header("USM00072469", 2026, 9, 2, 0, "2300", 2, 39.75, -104.83),
    igra2_row(2, 1, 84500, 1611, 251, 600, 40, 200, 20, pflag="B"),
    igra2_row(1, 0, 50000, 5872, -120, 200, 220, 90, 80),
]) + "\n"


def calibrate() -> dict[str, object]:
    """Every family in CALIBRATION, planted and read back."""
    from woof.verify.obs import stations as st

    from .upper_air_scorecard import synthetic_grid

    out: dict[str, object] = {"schema": SCHEMA, "kind": "calibration", "families": {}}
    gauss = synthetic_grid(21)
    regular_lat = np.arange(90.0, -90.0 - 0.5, -1.0)
    regular_lon = np.arange(0.0, 360.0, 1.0)

    # -- S1: the sampler against a closed form -------------------------
    rng = np.random.default_rng(7)
    s1 = {}
    for name, lat, lon in (("gaussian", gauss.latitude_deg, gauss.longitude_deg), ("regular", regular_lat, regular_lon)):
        lon2, lat2 = np.meshgrid(lon, lat)
        planted = 3.0 + 0.7 * lat2 - 0.02 * lon2
        sampler = RectilinearSampler(lat, lon)
        lo, hi = float(min(lat[0], lat[-1])), float(max(lat[0], lat[-1]))
        plats = rng.uniform(lo, hi, 400)
        dlon = 360.0 / lon.size
        plons = rng.uniform(0.0, 360.0 - dlon - 1.0e-6, 400)  # never in the seam cell
        read = sampler.sample(planted, plats, plons)
        s1[f"{name}_linear_max_abs_error"] = float(np.max(np.abs(read - (3.0 + 0.7 * plats - 0.02 * plons))))
        # the seam cell by hand: corners at the last column and column 0
        seam_lons = rng.uniform(360.0 - dlon + 1.0e-6, 360.0 - 1.0e-6, 40)
        seam_lats = rng.uniform(lo, hi, 40)
        wave = np.cos(np.deg2rad(lon2)) + 0.5 * lat2
        read = sampler.sample(wave, seam_lats, seam_lons)
        asc_lat = lat[::-1] if lat[0] > lat[-1] else lat
        fld = wave[::-1] if lat[0] > lat[-1] else wave
        hand = np.empty(40)
        for k in range(40):
            j1 = int(np.searchsorted(asc_lat, seam_lats[k], side="right"))
            j1 = min(max(j1, 1), asc_lat.size - 1)
            j0 = j1 - 1
            wy = (seam_lats[k] - asc_lat[j0]) / (asc_lat[j1] - asc_lat[j0])
            wx = (seam_lons[k] - lon[-1]) / dlon
            hand[k] = (fld[j0, -1] * (1 - wx) + fld[j0, 0] * wx) * (1 - wy) + (fld[j1, -1] * (1 - wx) + fld[j1, 0] * wx) * wy
        s1[f"{name}_seam_max_abs_error"] = float(np.max(np.abs(read - hand)))
    # the library's index-space bilinear against this sampler on the regular grid
    sampler = RectilinearSampler(regular_lat, regular_lon)
    lon2, lat2 = np.meshgrid(regular_lon, regular_lat)
    texture = np.sin(np.deg2rad(3.0 * lat2)) * np.cos(np.deg2rad(2.0 * lon2)) + 0.01 * lat2
    plats = rng.uniform(-89.0, 89.0, 400)
    plons = rng.uniform(0.5, 357.5, 400)
    x, y, inside = sampler.library_positions(plats, plons)
    lib = np.array([st.sample_field(texture, st.StationPosition("p", float(x[k]), float(y[k]))) for k in range(400)])
    s1["library_vs_sampler_max_abs_error"] = float(np.max(np.abs(lib - sampler.sample(texture, plats, plons))))
    gs = RectilinearSampler(gauss.latitude_deg, gauss.longitude_deg)
    polar = gs.sample(np.ones(gs.shape), np.array([89.9, -89.9]), np.array([10.0, 10.0]))
    s1["poleward_of_first_row_refused"] = bool(np.isnan(polar).all())
    holed = np.ones(gs.shape)
    holed[10, 20] = np.nan
    lat_hole = gauss.latitude_deg[10] + 0.3 * (gauss.latitude_deg[11] - gauss.latitude_deg[10])
    lon_hole = gauss.longitude_deg[20] + 0.4 * (gauss.longitude_deg[21] - gauss.longitude_deg[20])
    s1["nan_corner_refused"] = bool(np.isnan(gs.sample(holed, [lat_hole], [lon_hole])[0]))
    s1["clean_cell_beside_hole_reads"] = float(gs.sample(holed, [lat_hole], [gauss.longitude_deg[22] + 0.5 * (gauss.longitude_deg[23] - gauss.longitude_deg[22])])[0])
    out["families"]["S1_sampler"] = s1

    # -- S2 and S3: surface, a model against itself and planted offsets --
    valid = "2026-01-01T00:00:00"
    model = synthetic_surface_fields(gauss.latitude_deg, gauss.longitude_deg)
    stations = synthetic_stations(model, 300)
    offsets = {"temperature_2m": 1.5, "dewpoint_2m": -2.0, "wind_speed_10m": 0.75, "mslp": 120.0}
    s2 = {}
    s3 = {}
    for tag, planted in (("self", {}), ("planted", offsets)):
        obs = synthetic_observations(model, stations, valid, planted)
        # a second model on the product-style regular grid, so the
        # intersection of two admitted sets is exercised
        second = SurfaceFields(
            latitude_deg=regular_lat, longitude_deg=regular_lon,
            **{k: _regrid_to(model, k, regular_lat, regular_lon)
               for k in ("t2_k", "q2_kg_kg", "u10_m_s", "v10_m_s", "surface_pressure_pa", "terrain_m")},
            land=np.ones((regular_lat.size, regular_lon.size), dtype=bool), source={"kind": "synthetic-regular"},
        )
        result = score_surface_stations({"model": model, "other": second}, obs, valid)
        table = {}
        for v in SURFACE_VARIABLES:
            s = result["scores"]["model"][v]
            factor = SURFACE_DISPLAY[v][2]
            table[v] = {"n": s["n"], "bias": s["bias"], "rmse": s["rmse"], "mae": s["mae"],
                        "planted": -planted.get(v, 0.0) * factor}
        (s2 if tag == "self" else s3)["surface"] = table
        (s2 if tag == "self" else s3)["surface_common_stations"] = result["common_station_count"]
    # -- upper air, self and planted --
    upper = synthetic_level_fields(gauss.latitude_deg, gauss.longitude_deg)
    planted_upper = {("z", 50_000.0): 12.0, ("t", 85_000.0): -1.5, ("t", 50_000.0): 0.8, ("u", 25_000.0): 3.0, ("v", 25_000.0): -4.0,
                     ("u", 85_000.0): -3.0, ("v", 85_000.0): 4.0}
    for tag, planted in (("self", {}), ("planted", planted_upper)):
        obs = synthetic_soundings(upper, 200, offsets=planted)
        result = score_soundings({"model": upper}, obs)
        table = {}
        for score_name, kind, level in UPPER_TARGETS:
            row = result["scores"]["model"][score_name]["global"]
            if kind == "wind":
                table[score_name] = {**row, "planted_u": -planted.get(("u", level), 0.0), "planted_v": -planted.get(("v", level), 0.0),
                                     "planted_rmsve": math.hypot(planted.get(("u", level), 0.0), planted.get(("v", level), 0.0))}
            else:
                table[score_name] = {**row, "planted": -planted.get((kind, level), 0.0)}
        (s2 if tag == "self" else s3)["upper_air"] = table
    # a texture on top of the 12 m
    rng = np.random.default_rng(11)
    obs = synthetic_soundings(upper, 400, offsets={("z", 50_000.0): 12.0})
    noise = rng.standard_normal(400)
    noise = 5.0 * (noise - noise.mean()) / np.sqrt(np.mean((noise - noise.mean()) ** 2))
    for k, record in enumerate(obs):
        record["levels"]["50000"]["z"] += float(noise[k])
    result = score_soundings({"model": upper}, obs)
    row = result["scores"]["model"]["z500"]["global"]
    s3["z500_texture"] = {"bias": row["bias"], "rmse": row["rmse"], "expected_rmse": 13.0}
    out["families"]["S2_self"] = s2
    out["families"]["S3_planted"] = s3

    # -- S4: derived fields --------------------------------------------
    t = np.linspace(233.0, 313.0, 41)
    p = np.full_like(t, 95_000.0)
    e = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 273.15 + 243.5))
    q = 0.622 * e / (p - (1.0 - 0.622) * e)
    td = dewpoint_from_specific_humidity(q, p)
    s4 = {"dewpoint_roundtrip_max_abs_error_k": float(np.max(np.abs(td - t)))}
    s4["mslp_at_sea_level_equals_ps"] = bool(np.array_equal(mslp_reduction(np.array([101325.0, 98000.0]), np.zeros(2), np.array([288.15, 300.0])), np.array([101325.0, 98000.0])))
    closed = 90000.0 * math.exp(9.80665 * 1000.0 / (287.05 * (288.15 + 0.0065 * 500.0)))
    s4["mslp_1000m_closed_form_error_pa"] = float(abs(mslp_reduction(np.array([90000.0]), np.array([1000.0]), np.array([288.15]))[0] - closed))
    out["families"]["S4_derived"] = s4

    # -- S5: the IGRA2 record -----------------------------------------
    all_soundings = parse_igra2(IGRA2_SYNTHETIC)
    twelve = parse_igra2(IGRA2_SYNTHETIC, wanted={datetime(2026, 9, 1, 12)})
    first = all_soundings[0]
    m850 = first.mandatory(85_000.0)
    m500 = first.mandatory(50_000.0)
    m250 = first.mandatory(25_000.0)
    s5 = {
        "sounding_count": len(all_soundings), "twelve_z_count": len(twelve),
        "station_id": first.station_id, "latitude": first.latitude, "longitude": first.longitude,
        "z850": m850["z"], "t850": m850["t"], "u850": m850["u"], "v850": m850["v"],
        "z500": m500["z"], "t500": m500["t"], "u500": m500["u"], "v500": m500["v"],
        "z250_missing": m250["z"], "t250_missing": m250["t"], "wspd250": m250["wspd"],
        "no_700": first.mandatory(70_000.0),
        "wind_270_10": wind_components(270.0, 10.0), "wind_360_5": wind_components(360.0, 5.0),
    }
    out["families"]["S5_igra2"] = s5

    # -- S6: refusals by name -----------------------------------------
    s6 = {}
    try:
        surface_fields_from_checkpoint_arrays({"physics__q2": np.zeros(gs.shape)}, None, np.zeros(gs.shape), checkpoint="ck")
        s6["missing_t2"] = "not refused"
    except KeyError as exc:
        s6["missing_t2"] = str(exc)
    frame = _FakeFrame(regular_lat, regular_lon, {name: np.zeros((regular_lat.size, regular_lon.size)) for name in PRODUCT_SURFACE_FIELDS if name != "specific_humidity_2m"})
    try:
        product_surface_fields(frame, path="gfs.f018")
        s6["missing_q2"] = "not refused"
    except ValueError as exc:
        s6["missing_q2"] = str(exc)
    level_frame = _FakeFrame(
        regular_lat, regular_lon,
        {"geopotential_height": np.zeros((2, regular_lat.size, regular_lon.size)), "air_temperature": np.zeros((2, regular_lat.size, regular_lon.size)),
         "eastward_wind": np.zeros((2, regular_lat.size, regular_lon.size)), "northward_wind": np.zeros((2, regular_lat.size, regular_lon.size)),
         "surface_pressure": np.full((regular_lat.size, regular_lon.size), 101325.0)},
        vertical_values=[25_000.0, 50_000.0],
    )
    try:
        product_level_fields(level_frame, path="gfs.f012")
        s6["missing_850"] = "not refused"
    except ValueError as exc:
        s6["missing_850"] = str(exc)
    out["families"]["S6_refusals"] = s6

    # -- S7: admission parity -----------------------------------------
    x, y, inside = sampler.library_positions(plats, plons)
    direct_x = (plons - regular_lon[0]) / 1.0
    direct_y = (regular_lat[0] - plats) / 1.0
    s7 = {"x_max_abs_error": float(np.max(np.abs(x - direct_x))), "y_max_abs_error": float(np.max(np.abs(y - direct_y)))}
    out["families"]["S7_admission"] = s7
    return out


def _regrid_to(fields: SurfaceFields, name: str, lat, lon) -> np.ndarray:
    lon2, lat2 = np.meshgrid(lon, lat)
    sampler = fields.sampler()
    source = getattr(fields, name)
    values = sampler.sample(source, lat2.ravel(), lon2.ravel()).reshape(lat2.shape)
    # rows poleward of the Gaussian grid's outermost row read that row
    for j in range(lat.size):
        if np.isnan(values[j]).all():
            nearest = float(fields.latitude_deg[int(np.argmin(np.abs(fields.latitude_deg - lat[j])))])
            values[j] = sampler.sample(source, np.full(lon.size, nearest), lon)
    return values


# --------------------------------------------------------------------------
# the door
# --------------------------------------------------------------------------


def _parse_model_spec(spec: str) -> tuple[str, Path, Path]:
    label, _, rest = spec.partition("=")
    run_dir, _, checkpoint = rest.rpartition(":")
    if not label or not run_dir or not checkpoint:
        raise argparse.ArgumentTypeError(f"--model wants label=RUN_DIR:CHECKPOINT, got {spec!r}")
    return label, Path(run_dir), Path(checkpoint)


def _parse_product_spec(spec: str) -> tuple[str, Path, str | None]:
    """``label=PATH`` reads the product with the run's analysis mapping (or
    ``--mapping``); ``label=PATH@MAPPING`` names the product's own mapping,
    an authority id or a path, so a second product family (the IFS open
    data beside the GFS) is one more row on the command line and not a
    branch in here."""
    label, _, rest = spec.partition("=")
    path, _, mapping = rest.partition("@")
    if not label or not path:
        raise argparse.ArgumentTypeError(f"--gfs wants label=PATH or label=PATH@MAPPING, got {spec!r}")
    return label, Path(path), (mapping or None)


def _phi_for(run_dir: Path, mapping: str | None, cache_dir: Path | None, initial_analysis: Path | None):
    """The run's surface geopotential: the upper-air scorecard's cache when
    it holds the rebuild for this initial analysis and grid (no resident
    transform is built), else that instrument's own rebuild."""
    receipt = read_receipt(run_dir)
    mapping_path = resolve_mapping(mapping, receipt)
    if cache_dir is not None:
        _mapping, inputs, _sha = receipt_initial_analysis(receipt)
        candidates = [Path(initial_analysis)] if initial_analysis else [Path(p) for p in inputs]
        path = next((c for c in candidates if c.exists()), None)
        if path is not None:
            digest = _sha256_file(path)
            if digest not in set(inputs.values()):
                raise ValueError(
                    f"{path}: SHA-256 {digest[:16]} is not the initial analysis the run receipt recorded"
                )
            nlat, nlon = int(receipt["transform"]["nlat"]), int(receipt["transform"]["nlon"])
            cache = Path(cache_dir) / f"phi-surface-{digest[:16]}-{nlat}x{nlon}.npz"
            if cache.exists():
                with np.load(cache, allow_pickle=False) as archive:
                    phi = np.array(archive["phi"])
                provenance = {
                    "initial_analysis": str(path), "sha256": digest,
                    "construction": "analysis terrain regridded bilinearly to the Gaussian grid, "
                                    "times g, spectrally truncated, float64 rebuild of the run's float32 field",
                    "cache": str(cache),
                }
                return phi, provenance, mapping_path
    transform = build_transform_from_receipt(receipt)
    phi, provenance = surface_geopotential_for_run(receipt, transform, initial_analysis, mapping_path, cache_dir=cache_dir)
    return phi, provenance, mapping_path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _surface_table(result: dict) -> list[str]:
    # Hoisted out of the f-string: a third quote level inside an f-string is
    # Python 3.12 syntax, and this package declares 3.11 as its floor.
    admitted = ", ".join(f"{k} {v['admitted']}" for k, v in result["admission"].items())
    lines = [f"valid {result['valid_time']}  common stations {result['common_station_count']}  "
             f"admitted {admitted}"]
    lines.append(f"{'model':16s} {'var':5s} {'n':>5s} {'bias':>9s} {'rmse':>9s} {'mae':>9s}  unit")
    for label, table in result["scores"].items():
        for v in SURFACE_VARIABLES:
            s = table[v]
            if s["n"]:
                lines.append(f"{label:16s} {s['display']:5s} {s['n']:5d} {s['bias']:9.3f} {s['rmse']:9.3f} {s['mae']:9.3f}  {s['unit']}")
            else:
                lines.append(f"{label:16s} {s['display']:5s} {0:5d}")
    return lines


def _sounding_table(result: dict) -> list[str]:
    lines = [f"sites {result['site_count']}"]
    lines.append(f"{'model':16s} {'target':6s} {'hemi':6s} {'n':>4s} {'bias/rmsve':>11s} {'rmse/spdbias':>13s} {'mae/ubias':>10s} {'vbias':>8s}")
    for label, table in result["scores"].items():
        for score_name, kind, _level in UPPER_TARGETS:
            for hemi in HEMISPHERES:
                row = table[score_name][hemi]
                if row["n"] == 0:
                    lines.append(f"{label:16s} {score_name:6s} {hemi:6s} {0:4d}")
                elif kind == "wind":
                    lines.append(f"{label:16s} {score_name:6s} {hemi:6s} {row['n']:4d} {row['rmsve']:11.3f} {row['speed_bias']:13.3f} {row['u_bias']:10.3f} {row['v_bias']:8.3f}")
                else:
                    lines.append(f"{label:16s} {score_name:6s} {hemi:6s} {row['n']:4d} {row['bias']:11.3f} {row['rmse']:13.3f} {row['mae']:10.3f}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="woof.globe.obs_scorecard", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibrate", help="plant every calibration family and read it back")
    cal.add_argument("--out", type=Path)

    ext = sub.add_parser("igra2-extract", help="reduce a directory of IGRA2 station zips to the mandatory levels at the wanted instants")
    ext.add_argument("--zips", type=Path, required=True)
    ext.add_argument("--valid", action="append", required=True, help="nominal instant, e.g. 2026-09-01T12:00:00Z")
    ext.add_argument("--out", type=Path, required=True)

    for name in ("surface", "upper-air"):
        p = sub.add_parser(name)
        p.add_argument("--model", action="append", default=[], type=_parse_model_spec, help="label=RUN_DIR:CHECKPOINT")
        p.add_argument("--gfs", "--product", action="append", default=[], type=_parse_product_spec, dest="gfs",
                       help="label=PRODUCT_PATH[@MAPPING]: a decoded product (the GFS, the IFS open data) read by "
                            "--mapping, or by its own authority mapping id or path after the @")
        p.add_argument("--mapping", help="analysis mapping for the products (default: the first model's receipt)")
        p.add_argument("--cache-dir", type=Path)
        p.add_argument("--initial-analysis", type=Path)
        p.add_argument("--out", type=Path, required=True)
        if name == "surface":
            p.add_argument("--valid", required=True, help="seam instant, e.g. 2026-09-01T18:00:00")
            p.add_argument("--obs", type=Path, required=True, help="gpuwm-obs.asos-surface.v2 (or v1) record")
            p.add_argument("--csv", type=Path)
        else:
            p.add_argument("--valid", required=True, help="nominal instant, e.g. 2026-09-01T12:00:00Z")
            p.add_argument("--soundings", type=Path, required=True, help="igra2-extract output")
    args = parser.parse_args(argv)

    if args.command == "calibrate":
        payload = calibrate()
        text = json.dumps(payload, indent=2, sort_keys=True, default=str)
        if args.out:
            _write_json(args.out, payload)
        print(text)
        return 0

    if args.command == "igra2-extract":
        wanted = [datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ") for v in args.valid]
        payload = extract_igra2_zips(args.zips, wanted)
        _write_json(args.out, payload)
        by_time: dict[str, int] = {}
        for record in payload["soundings"]:
            by_time[record["nominal"]] = by_time.get(record["nominal"], 0) + 1
        print(f"zips {payload['zip_count']} bytes {payload['zip_bytes']} soundings {len(payload['soundings'])} by time {by_time}")
        print(f"wrote {args.out}")
        return 0

    if not args.model and not args.gfs:
        parser.error("at least one --model or --gfs")
    cache_dir = args.cache_dir
    mapping_path = None
    phi_by_run: dict[Path, tuple[np.ndarray, dict]] = {}
    for _label, run_dir, _ck in args.model:
        if run_dir not in phi_by_run:
            phi, provenance, mapping_path = _phi_for(run_dir, args.mapping, cache_dir, args.initial_analysis)
            phi_by_run[run_dir] = (phi, provenance)
    from .analysis_initial import resolve_analysis_mapping

    if any(spec is None for _label, _path, spec in args.gfs) and mapping_path is None:
        if not args.mapping:
            parser.error("--mapping is needed to decode products when no --model supplies a receipt")
        mapping_path = resolve_analysis_mapping(args.mapping)
    product_mappings = {
        label: (resolve_analysis_mapping(spec) if spec else mapping_path) for label, _path, spec in args.gfs
    }

    if args.command == "surface":
        from woof.obs.sources import AsosSurfaceSource

        models: dict[str, SurfaceFields] = {}
        for label, run_dir, checkpoint in args.model:
            phi, _prov = phi_by_run[run_dir]
            models[label] = model_surface_fields(run_dir, checkpoint, surface_geopotential=phi)
        for label, path, _spec in args.gfs:
            frame, decode = decode_reference(product_mappings[label], path)
            models[label] = product_surface_fields(frame, path=str(path))
            models[label].source["mapping"] = str(product_mappings[label])
            models[label].source["decode"] = decode
        observations = AsosSurfaceSource(args.obs).observations([args.valid])
        result = score_surface_stations(models, observations, args.valid)
        result["surface_geopotential"] = {str(k): v[1] for k, v in phi_by_run.items()}
        result["obs_record"] = str(args.obs)
        _write_json(args.out, result)
        if args.csv:
            import csv

            rows = result["stations"]
            keys: list[str] = []
            for row in rows:
                for key in row:
                    if key not in keys:
                        keys.append(key)
            with args.csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
        print("\n".join(_surface_table(result)))
        print(f"wrote {args.out}")
        return 0

    # upper-air
    payload = json.loads(args.soundings.read_text(encoding="utf-8"))
    soundings = [r for r in payload["soundings"] if r["nominal"] == args.valid]
    if not soundings:
        raise SystemExit(f"{args.soundings}: no soundings at {args.valid} (it holds {payload.get('wanted')})")
    level_models: dict[str, LevelFields] = {}
    for label, run_dir, checkpoint in args.model:
        phi, provenance = phi_by_run[run_dir]
        level_models[label] = model_level_fields(
            run_dir, checkpoint, surface_geopotential=phi, cache_dir=cache_dir, phi_digest=provenance["sha256"]
        )
    for label, path, _spec in args.gfs:
        frame, decode = decode_reference(product_mappings[label], path)
        level_models[label] = product_level_fields(frame, path=str(path))
        level_models[label].source["mapping"] = str(product_mappings[label])
        level_models[label].source["decode"] = decode
    result = score_soundings(level_models, soundings)
    result["valid_time"] = args.valid
    result["soundings_record"] = str(args.soundings)
    result["soundings_manifest"] = {"zip_count": payload.get("zip_count"), "zip_bytes": payload.get("zip_bytes"), "source": payload.get("source")}
    _write_json(args.out, result)
    print("\n".join(_sounding_table(result)))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
