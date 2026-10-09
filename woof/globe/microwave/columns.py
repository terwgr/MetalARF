"""GDAS analysis columns for the microwave operator.

The 0.25-degree GDAS pgrb2 analyses (the same product the global model
initializes from) are decoded through the Rust mapped engine with the
``rw-wps-gdas-pgrb2-0p25-microwave-columns`` mapping: temperature and
specific humidity on all 41 isobaric levels (1000 hPa to 0.01 hPa),
surface pressure, terrain, skin and 2 m temperature, 10 m wind, the land
and sea-ice masks, and the column water records the clear-sky screen
reads (cloud water path, precipitable water).  The analysis total cloud
cover record of the GDAS pgrb2 f000 file carries no data points (the
mapped engine refuses it: "Section 5 declares 0 data points"), so the
analysis side of the cloud screen is the cloud water path alone.

A day of ATMS overpasses is compared against columns interpolated
linearly in time between the two bracketing six-hourly analyses (00, 06,
12, 18, 24Z) and sampled bilinearly at the thinned cell's mean position.
Bilinear sampling of a regular latitude-longitude grid is arithmetic on
four neighbours; the field decode and the beam colocation are the Rust
steps and both stay in Rust.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rte import Column

MAPPING_NAME = "rw-wps-gdas-pgrb2-0p25-microwave-columns.mapping.json"

SURFACE_FIELDS = (
    "surface_pressure", "terrain_height", "skin_temperature", "air_temperature_2m",
    "eastward_wind_10m", "northward_wind_10m", "land_fraction", "sea_ice_fraction",
    "cloud_water_path", "precipitable_water",
)


def mapping_path() -> Path:
    """The ATMS column mapping, from whichever authority table has it.

    One resolver answers every mapping this package names
    (:func:`woof.globe.analysis_initial.resolve_analysis_mapping`): the
    engine's table first, this package's carried copies second.  This
    module used to join the file name onto the engine's directory itself,
    which on a published engine that does not carry the row produced a
    path to nothing and a decode that failed on the open rather than on
    the missing row.
    """

    from woof.globe.analysis_initial import resolve_analysis_mapping

    return resolve_analysis_mapping(MAPPING_NAME)


@dataclass(frozen=True)
class Analysis:
    """One decoded analysis on its regular grid (latitude descending or
    ascending as the product carries it; longitude 0..360)."""

    valid_time: dt.datetime
    latitude: np.ndarray  # (ny,)
    longitude: np.ndarray  # (nx,)
    pressure_pa: np.ndarray  # (nlev,)
    temperature_k: np.ndarray  # (nlev, ny, nx)
    specific_humidity: np.ndarray  # (nlev, ny, nx)
    surface: dict[str, np.ndarray]  # each (ny, nx)
    provenance: dict[str, object]


def decode_analysis(grib: str | Path, *, scratch_destination: str | Path | None = None) -> Analysis:
    """Decode one GDAS pgrb2 analysis through the mapped engine."""
    from woof.globe.mapped_source_compat import decode_through_engine

    grib = Path(grib)
    decoded = decode_through_engine(
        mapping_path(), [grib], scratch_destination=scratch_destination)
    frames = decoded.frames
    if len(frames) != 1:
        raise ValueError(f"{grib}: decoded {len(frames)} valid times, expected one analysis")
    frame = frames[0]
    if getattr(frame, "vertical_kind", "pressure") != "pressure":
        raise ValueError(f"{grib}: vertical kind {frame.vertical_kind!r} is not pressure")
    surface = {}
    for name in SURFACE_FIELDS:
        if name not in frame.fields:
            raise ValueError(f"{grib}: the microwave mapping decoded no {name}")
        surface[name] = np.asarray(frame.fields[name].values, dtype=np.float64)
    return Analysis(
        valid_time=frame.valid_time.replace(tzinfo=dt.timezone.utc),
        latitude=np.asarray(frame.latitude, dtype=np.float64),
        longitude=np.asarray(frame.longitude, dtype=np.float64),
        pressure_pa=np.asarray(frame.vertical_values, dtype=np.float64),
        temperature_k=np.asarray(frame.fields["air_temperature"].values, dtype=np.float64),
        specific_humidity=np.clip(
            np.asarray(frame.fields["specific_humidity"].values, dtype=np.float64), 0.0, None
        ),
        surface=surface,
        provenance={
            "grib": str(grib),
            "input_sha256": dict(frame.input_sha256),
            "mapping_sha256": frame.mapping_sha256,
            "grid_fingerprint": frame.grid_fingerprint,
            "valid_time": frame.valid_time.isoformat(),
            # HOW the decode was obtained, not only what it read: which
            # mechanism placed the frame stream's scratch, and whether the
            # installed engine's soil-only preserve_mask narrowing had to
            # be adapted for this mapping's masked snow records (it does
            # declare two).  These columns feed the assimilation, so the
            # product says which decoder path produced it rather than
            # leaving a reader to infer it from an engine version.
            "decode": decoded.receipt,
        },
    )


class BilinearSampler:
    """Bilinear weights from a regular global lat-lon grid to points."""

    def __init__(self, latitude: np.ndarray, longitude: np.ndarray, lat_pts, lon_pts):
        lat = np.asarray(latitude, dtype=np.float64)
        lon = np.asarray(longitude, dtype=np.float64)
        self.flip = lat[0] > lat[-1]
        if self.flip:
            lat = lat[::-1]
        dlat = lat[1] - lat[0]
        dlon = lon[1] - lon[0]
        if abs(dlon * lon.size - 360.0) > 1.0e-3:
            raise ValueError("analysis longitude ring does not cover the globe")
        y = np.clip((np.asarray(lat_pts, dtype=np.float64) - lat[0]) / dlat, 0.0, lat.size - 1.0)
        self.y0 = np.minimum(y.astype(np.int64), lat.size - 2)
        self.wy = y - self.y0
        x = np.mod(np.asarray(lon_pts, dtype=np.float64) - lon[0], 360.0) / dlon
        self.x0 = np.mod(x.astype(np.int64), lon.size)
        self.wx = x - np.floor(x)
        self.x1 = np.mod(self.x0 + 1, lon.size)

    def __call__(self, field: np.ndarray) -> np.ndarray:
        values = np.asarray(field, dtype=np.float64)
        if self.flip:
            values = values[..., ::-1, :]
        y0, y1 = self.y0, self.y0 + 1
        a = values[..., y0, self.x0]
        b = values[..., y0, self.x1]
        c = values[..., y1, self.x0]
        d = values[..., y1, self.x1]
        return (1.0 - self.wy) * ((1.0 - self.wx) * a + self.wx * b) + self.wy * (
            (1.0 - self.wx) * c + self.wx * d
        )


@dataclass(frozen=True)
class SampledColumns:
    #: None when the sampling was ``surface_only``.
    column: Column | None
    surface: dict[str, np.ndarray]
    weights_earlier: np.ndarray  # time weight of the earlier analysis per point
    earlier_index: np.ndarray
    later_index: np.ndarray


def _ordered_analyses(analyses: list[Analysis]) -> list[Analysis]:
    if len(analyses) < 1:
        raise ValueError("at least one analysis is needed")
    ordered = sorted(analyses, key=lambda a: a.valid_time)
    for later in ordered[1:]:
        if later.pressure_pa.shape != ordered[0].pressure_pa.shape or not np.allclose(
            later.pressure_pa, ordered[0].pressure_pa
        ):
            raise ValueError("analyses carry different pressure level sets")
        if later.latitude.shape != ordered[0].latitude.shape:
            raise ValueError("analyses carry different grids")
    return ordered


def time_weights(ordered: list[Analysis], time_unix_s) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(earlier_index, later_index, w_earlier)`` of the linear time
    interpolation between the bracketing analyses; instants outside the
    span are held at the end analysis (``w_earlier`` exactly 0 or 1)."""
    times = np.array([a.valid_time.timestamp() for a in ordered])
    t = np.asarray(time_unix_s, dtype=np.float64)
    t_clip = np.clip(t, times[0], times[-1])
    if len(times) > 1:
        later_index = np.clip(np.searchsorted(times, t_clip, side="right"), 1, len(times) - 1)
        earlier_index = np.maximum(later_index - 1, 0)
    else:
        later_index = np.zeros(t.shape, dtype=np.int64)
        earlier_index = later_index
    span = times[later_index] - times[earlier_index]
    with np.errstate(invalid="ignore", divide="ignore"):
        w_later = np.where(span > 0, (t_clip - times[earlier_index]) / np.where(span > 0, span, 1.0), 0.0)
    return earlier_index, later_index, 1.0 - w_later


def sample_columns(
    analyses: list[Analysis],
    lat_pts,
    lon_pts,
    time_unix_s,
    *,
    surface_only: bool = False,
) -> SampledColumns:
    """Columns at points and instants: linear in time between the two
    bracketing analyses, bilinear in space.  Instants outside the analysis
    span are held at the nearest analysis and counted by the caller from
    ``weights_earlier`` (0 or 1 exactly at the ends).

    ``surface_only`` samples the surface records alone (the screens read
    them before any column is built) and leaves ``column`` None."""
    ordered = _ordered_analyses(analyses)
    earlier_index, later_index, w_earlier = time_weights(ordered, time_unix_s)
    w_later = 1.0 - w_earlier
    sampler = BilinearSampler(ordered[0].latitude, ordered[0].longitude, lat_pts, lon_pts)
    npts = int(np.asarray(time_unix_s).size)
    nlev = ordered[0].pressure_pa.size
    temperature = None if surface_only else np.zeros((nlev, npts))
    humidity = None if surface_only else np.zeros((nlev, npts))
    surface = {name: np.zeros(npts) for name in SURFACE_FIELDS}
    for k, analysis in enumerate(ordered):
        weight = np.where(earlier_index == k, w_earlier, 0.0) + np.where(later_index == k, w_later, 0.0)
        if len(ordered) == 1:
            weight = np.ones(npts)
        if not np.any(weight > 0):
            continue
        active = weight > 0
        if not surface_only:
            temperature[:, active] += weight[active] * sampler(analysis.temperature_k)[:, active]
            humidity[:, active] += weight[active] * sampler(analysis.specific_humidity)[:, active]
        for name in SURFACE_FIELDS:
            surface[name][active] += weight[active] * sampler(analysis.surface[name])[active]
    column = None
    if not surface_only:
        column = Column(
            pressure_pa=ordered[0].pressure_pa,
            temperature_k=temperature,
            specific_humidity=humidity,
            surface_pressure_pa=surface["surface_pressure"],
            skin_temperature_k=surface["skin_temperature"],
            air_temperature_2m_k=surface["air_temperature_2m"],
        )
    return SampledColumns(
        column=column,
        surface=surface,
        weights_earlier=w_earlier,
        earlier_index=earlier_index,
        later_index=later_index,
    )


def save_analysis(analysis: Analysis, path: str | Path) -> None:
    """Cache a decoded analysis as one npz beside its provenance."""
    path = Path(path)
    np.savez_compressed(
        path,
        latitude=analysis.latitude,
        longitude=analysis.longitude,
        pressure_pa=analysis.pressure_pa,
        temperature_k=analysis.temperature_k.astype(np.float32),
        specific_humidity=analysis.specific_humidity.astype(np.float32),
        **{f"surface_{name}": values.astype(np.float32) for name, values in analysis.surface.items()},
    )
    sidecar = {"valid_time": analysis.valid_time.isoformat(), **analysis.provenance}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=1), encoding="utf-8")


def load_analysis(path: str | Path) -> Analysis:
    path = Path(path)
    data = np.load(path)
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return Analysis(
        valid_time=dt.datetime.fromisoformat(sidecar["valid_time"]),
        latitude=data["latitude"],
        longitude=data["longitude"],
        pressure_pa=data["pressure_pa"],
        temperature_k=data["temperature_k"].astype(np.float64),
        specific_humidity=data["specific_humidity"].astype(np.float64),
        surface={name: data[f"surface_{name}"].astype(np.float64) for name in SURFACE_FIELDS},
        provenance={k: v for k, v in sidecar.items() if k != "valid_time"},
    )
