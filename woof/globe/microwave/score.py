"""The microwave O-B scorecard.

Observation minus background for the ATMS channels against GDAS analysis
columns, clear sky over open ocean, per channel: bias and rmse before
and after a linear bias correction, with the diagnostics that name the
residual's terms (dependence on scan angle, latitude band, 10 m wind,
precipitable water, and the instrument noise the cells carry).

Screening, every stage counted in the receipt:

* ``candidate``: latitude inside ``max_abs_latitude_deg`` (60), at least
  ``min_beams_per_cell`` (3) beams in the cell, satellite zenith at or
  below ``max_zenith_deg`` (60).  Only the candidates have the analysis
  surface sampled under them.
* ``ocean``: land fraction below 0.01 and sea-ice fraction below 0.01 at
  the cell (bilinear from the analysis masks).
* ``analysis_clear``: analysis cloud water path at or below
  ``max_cloud_water_kg_m2`` (0.01), and total cloud cover at or below
  ``max_cloud_cover`` (0.05) when the column set carries it (the GDAS
  analysis record has no data points, so the GDAS columns do not).
* ``obs_clear``: cloud liquid water retrieved from ATMS channels 1 and 2
  (Grody et al., 2001, the AMSU-A window-channel retrieval; ATMS channels
  1 and 2 sit at the same frequencies) at or below
  ``max_retrieved_lwp_kg_m2`` (0.02), and the 23.8 minus 31.4 GHz
  difference positive (a negative difference marks precipitation or a
  sea-ice edge).
* ``time_span``: the cell's instant inside the analysis span, so no
  column is held at an end analysis.
* ``spread`` (per channel, inside the scoring): the within-cell standard
  deviation of the channel being scored at or below ``max_cell_std_k``
  (3 K) for channels 1 to 3 and ``max_sounding_cell_std_k`` (1.5 K) for
  the sounding channels, so a cell straddling a front or a cloud edge is
  not scored as clear.

Only the survivors have full columns sampled and the operator run, so a
day of 2 million cells costs the operator its clear-sky ocean cells alone.

Bias correction, per channel, every model fitted by least squares on the
cells whose position in the sorted time order is even and scored on the
odd cells, so every corrected rmse is out of sample:

* ``constant``: ``O - B = a``;
* ``linear``: ``a + b (B - mean_B)`` (the design's linear correction in
  the background brightness temperature);
* ``scan``: ``a + c (sec z - 1)``;
* ``geometry``: ``a + b (B - mean_B) + c (sec z - 1)``, the model the
  operator entry carries, because the member operator knows the
  background and the viewing geometry of every row and nothing else;
* ``wind``: ``geometry`` plus ``d W10`` with the analysis 10 m wind, the
  diagnostic that names the missing ocean roughness: a channel that
  passes only with the wind predictor is surface-limited and is not
  admitted, because the filter operator carries no wind.

The reading of record per channel is the ``geometry`` rmse against the
bar; ``linear`` and ``wind`` are reported beside it.  The noise floor per
channel is the rms over the scored cells of the within-cell standard
deviation divided by the square root of the beam count (the instrument
noise left in a cell mean); the part of the corrected rmse above that
floor is what the operator and the analysis own.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .channels import CHANNELS, TEMPERATURE_SOUNDING_CHANNELS
from .rte import Column, brightness_temperature

ALL_CHANNELS = tuple(range(1, 23))

SURFACE_TERM = "ocean surface emissivity (no wind roughness or foam in the specular model)"
NOISE_TERM = "instrument noise left in the cell mean (NEdT over the square root of the beam count)"
UPPER_TERM = "the analysis and the absorption above 5 hPa (Zeeman splitting is not modelled)"


@dataclass(frozen=True)
class ScreenOptions:
    max_land_fraction: float = 0.01
    max_sea_ice_fraction: float = 0.01
    max_abs_latitude_deg: float = 60.0
    max_cloud_cover: float = 0.05
    max_cloud_water_kg_m2: float = 0.01
    max_retrieved_lwp_kg_m2: float = 0.02
    max_zenith_deg: float = 60.0
    max_cell_std_k: float = 3.0
    max_sounding_cell_std_k: float = 1.5
    min_beams_per_cell: int = 3


def grody_cloud_liquid_water(tb_23_k, tb_31_k, zenith_deg) -> np.ndarray:
    """Cloud liquid water path (kg m-2) over ocean from the 23.8 and 31.4
    GHz brightness temperatures, Grody et al. (2001), J. Geophys. Res.
    106(D3), the AMSU-A algorithm.  Returns NaN where the logarithms are
    undefined (brightness temperatures at or above 285 K)."""
    mu = np.cos(np.deg2rad(np.asarray(zenith_deg, dtype=np.float64)))
    t23 = np.asarray(tb_23_k, dtype=np.float64)
    t31 = np.asarray(tb_31_k, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        a0 = 8.24 - (2.622 - 1.846 * mu) * mu
        lwp = mu * (a0 + 0.754 * np.log(285.0 - t23) - 2.265 * np.log(285.0 - t31))
    return np.where((t23 < 285.0) & (t31 < 285.0), lwp, np.nan)


@dataclass
class Screen:
    """Boolean stages and the survivor count after each."""

    stages: dict[str, int] = field(default_factory=dict)
    mask: np.ndarray | None = None


def candidate_mask(thinned, options: ScreenOptions) -> np.ndarray:
    """The geometry screen that needs no analysis under it."""
    lat = np.asarray(thinned.lat_mean_deg, dtype=np.float64)
    return (
        (np.abs(lat) <= options.max_abs_latitude_deg)
        & (np.asarray(thinned.count) >= options.min_beams_per_cell)
        & (np.asarray(thinned.zenith_mean_deg, dtype=np.float64) <= options.max_zenith_deg)
    )


def screen_cells(thinned, surface: dict[str, np.ndarray], time_weights_earlier: np.ndarray,
                 options: ScreenOptions) -> Screen:
    """The analysis-dependent screens; the geometry stages are counted
    again here so a caller that skipped :func:`candidate_mask` still gets
    every stage."""
    tb = thinned.tb_mean_k
    n = thinned.ncell
    screen = Screen()
    mask = np.ones(n, dtype=bool)
    screen.stages["cells"] = int(n)

    mask &= thinned.count >= options.min_beams_per_cell
    screen.stages["beams"] = int(mask.sum())

    lat = thinned.lat_mean_deg.astype(np.float64)
    ocean = (
        (surface["land_fraction"] <= options.max_land_fraction)
        & (surface["sea_ice_fraction"] <= options.max_sea_ice_fraction)
        & (np.abs(lat) <= options.max_abs_latitude_deg)
    )
    mask &= ocean
    screen.stages["ocean"] = int(mask.sum())

    clear = surface["cloud_water_path"] <= options.max_cloud_water_kg_m2
    if "total_cloud_cover" in surface:
        clear &= surface["total_cloud_cover"] <= options.max_cloud_cover
    mask &= clear
    screen.stages["analysis_clear"] = int(mask.sum())

    lwp = grody_cloud_liquid_water(tb[:, 0], tb[:, 1], thinned.zenith_mean_deg)
    obs_clear = np.isfinite(lwp) & (lwp <= options.max_retrieved_lwp_kg_m2) & (tb[:, 0] > tb[:, 1])
    mask &= obs_clear
    screen.stages["obs_clear"] = int(mask.sum())

    mask &= thinned.zenith_mean_deg <= options.max_zenith_deg
    screen.stages["zenith"] = int(mask.sum())

    inside = (time_weights_earlier > 0.0) & (time_weights_earlier < 1.0)
    mask &= inside
    screen.stages["time_span"] = int(mask.sum())

    finite = np.all(np.isfinite(tb[:, :16]), axis=1)
    mask &= finite
    screen.stages["finite"] = int(mask.sum())
    screen.mask = mask
    return screen


def _fit(predictors: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients of ``y ~ predictors`` (``(n, k)``); too
    few rows or a rank-deficient design fall back to the mean alone."""
    k = predictors.shape[1]
    fallback = np.zeros(k)
    if y.size == 0:
        return fallback
    fallback[0] = float(np.mean(y))
    if y.size < k + 1:
        return fallback
    coefficients, _, rank, _ = np.linalg.lstsq(predictors, y, rcond=None)
    if rank < k:
        return fallback
    return coefficients


def _fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least squares y = a + b x."""
    if x.size < 2 or np.ptp(x) == 0.0:
        return float(np.mean(y)) if y.size else 0.0, 0.0
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def _stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"count": 0, "bias": float("nan"), "rmse": float("nan"), "std": float("nan")}
    return {
        "count": int(values.size),
        "bias": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "std": float(np.std(values)),
    }


@dataclass
class ChannelScore:
    channel: int
    centre_ghz: float
    nedt_k: float
    fit_cells: int
    score_cells: int
    raw: dict[str, float]
    constant_corrected: dict[str, float]
    linear_corrected: dict[str, float]
    scan_corrected: dict[str, float]
    geometry_corrected: dict[str, float]
    wind_corrected: dict[str, float]
    linear_coefficients: dict[str, float]
    scan_coefficient: float
    #: ``a + b (B - mean_B) + c (sec z - 1)``: the operator entry's model.
    geometry_coefficients: dict[str, float]
    #: The geometry model plus ``d W10``.
    wind_coefficients: dict[str, float]
    mean_background_k: float
    #: rms over the scored cells of the within-cell standard deviation.
    beam_noise_k: float
    #: rms over the scored cells of cell_std / sqrt(count): the noise left in a cell mean.
    noise_floor_k: float
    #: sqrt(max(geometry rmse^2 - noise_floor^2, 0)).
    rmse_above_noise_k: float
    mean_beams_per_cell: float
    #: The reading of record: the geometry-corrected rmse at or under the bar.
    within_bar: bool
    within_bar_background_only: bool
    within_bar_with_wind: bool
    by_scan_angle: list[dict[str, float]]
    by_latitude_band: list[dict[str, float]]
    by_wind_speed: list[dict[str, float]]
    by_precipitable_water: list[dict[str, float]]
    by_hour: list[dict[str, float]]


def _binned(values: np.ndarray, key: np.ndarray, edges: np.ndarray, label: str) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (key >= lo) & (key < hi)
        stats = _stats(values[inside])
        rows.append({label + "_lo": float(lo), label + "_hi": float(hi), **stats})
    return rows


def score_channels(
    observed: np.ndarray,
    background: np.ndarray,
    *,
    zenith_deg: np.ndarray,
    latitude_deg: np.ndarray,
    wind_speed_m_s: np.ndarray,
    precipitable_water_kg_m2: np.ndarray,
    time_unix_s: np.ndarray,
    cell_std_k: np.ndarray,
    beam_count: np.ndarray | None = None,
    channels=ALL_CHANNELS,
    options: ScreenOptions | None = None,
    bar_k: float = 1.0,
) -> list[ChannelScore]:
    """Per-channel O-B statistics.  ``observed``, ``background`` and
    ``cell_std_k`` are ``(ncell, nchan)`` in the order of ``channels``;
    ``beam_count`` is ``(ncell,)`` or ``(ncell, nchan)`` (the beams that
    entered each cell mean; ones when None)."""
    options = options or ScreenOptions()
    n = observed.shape[0]
    order = np.argsort(time_unix_s, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    fit_half = rank % 2 == 0
    sec = 1.0 / np.cos(np.deg2rad(zenith_deg))
    hours = (np.asarray(time_unix_s, dtype=np.float64) % 86400.0) / 3600.0
    if beam_count is None:
        counts = np.ones((n, len(channels)))
    else:
        counts = np.asarray(beam_count, dtype=np.float64)
        if counts.ndim == 1:
            counts = np.repeat(counts[:, None], len(channels), axis=1)
    results = []
    for k, number in enumerate(channels):
        channel = CHANNELS[number - 1]
        limit = (options.max_sounding_cell_std_k if number in TEMPERATURE_SOUNDING_CHANNELS
                 else options.max_cell_std_k)
        good = np.isfinite(observed[:, k]) & np.isfinite(background[:, k]) & (cell_std_k[:, k] <= limit)
        o = observed[good, k]
        b = background[good, k]
        z = sec[good]
        w = wind_speed_m_s[good]
        fit = fit_half[good]
        score = ~fit
        omb = o - b
        mean_b = float(np.mean(b)) if b.size else float("nan")
        db = b - mean_b
        dz = z - 1.0
        a_const = float(np.mean(omb[fit])) if fit.any() else 0.0
        a_lin, b_lin = _fit_linear(db[fit], omb[fit]) if fit.any() else (0.0, 0.0)
        a_scan, c_scan = _fit_linear(dz[fit], omb[fit]) if fit.any() else (0.0, 0.0)
        design_geometry = np.stack([np.ones(omb.size), db, dz], axis=1)
        geometry = _fit(design_geometry[fit], omb[fit])
        design_wind = np.concatenate([design_geometry, w[:, None]], axis=1)
        wind = _fit(design_wind[fit], omb[fit])
        corrected_linear = omb[score] - (a_lin + b_lin * db[score])
        corrected_const = omb[score] - a_const
        corrected_scan = omb[score] - (a_scan + c_scan * dz[score])
        corrected_geometry = omb[score] - design_geometry[score] @ geometry
        corrected_wind = omb[score] - design_wind[score] @ wind
        linear = _stats(corrected_linear)
        geometry_stats = _stats(corrected_geometry)
        wind_stats = _stats(corrected_wind)
        std_scored = cell_std_k[good, k][score]
        count_scored = np.maximum(counts[good, k][score], 1.0)
        beam_noise = float(np.sqrt(np.mean(std_scored ** 2))) if std_scored.size else float("nan")
        floor = (float(np.sqrt(np.mean(std_scored ** 2 / count_scored)))
                 if std_scored.size else float("nan"))
        above = (float(np.sqrt(max(geometry_stats["rmse"] ** 2 - floor ** 2, 0.0)))
                 if std_scored.size else float("nan"))
        results.append(ChannelScore(
            channel=number,
            centre_ghz=channel.centre_ghz,
            nedt_k=channel.nedt_k,
            fit_cells=int(fit.sum()),
            score_cells=int(score.sum()),
            raw=_stats(omb[score]),
            constant_corrected=_stats(corrected_const),
            linear_corrected=linear,
            scan_corrected=_stats(corrected_scan),
            geometry_corrected=geometry_stats,
            wind_corrected=wind_stats,
            linear_coefficients={"a": a_lin, "b": b_lin},
            scan_coefficient=c_scan,
            geometry_coefficients={"a": float(geometry[0]), "b": float(geometry[1]),
                                   "c": float(geometry[2]), "mean_background_k": mean_b},
            wind_coefficients={"a": float(wind[0]), "b": float(wind[1]), "c": float(wind[2]),
                               "d": float(wind[3]), "mean_background_k": mean_b},
            mean_background_k=mean_b,
            beam_noise_k=beam_noise,
            noise_floor_k=floor,
            rmse_above_noise_k=above,
            mean_beams_per_cell=float(np.mean(count_scored)) if count_scored.size else float("nan"),
            within_bar=bool(geometry_stats["count"] > 0 and geometry_stats["rmse"] <= bar_k),
            within_bar_background_only=bool(linear["count"] > 0 and linear["rmse"] <= bar_k),
            within_bar_with_wind=bool(wind_stats["count"] > 0 and wind_stats["rmse"] <= bar_k),
            by_scan_angle=_binned(omb, zenith_deg[good], np.array([0, 15, 30, 45, 60.001]), "zenith"),
            by_latitude_band=_binned(omb, latitude_deg[good], np.array([-60, -30, 0, 30, 60.001]), "lat"),
            by_wind_speed=_binned(omb, w, np.array([0, 4, 8, 12, 100.0]), "wind"),
            by_precipitable_water=_binned(
                omb, precipitable_water_kg_m2[good], np.array([0, 15, 30, 45, 200.0]), "pwat"
            ),
            by_hour=_binned(omb, hours[good], np.arange(0.0, 24.001, 6.0), "hour"),
        ))
    return results


def background_for_cells(column, zenith_deg, scan_angle_deg, channels=ALL_CHANNELS,
                         *, chunk: int = 256, rayleigh_jeans: bool = False) -> np.ndarray:
    """Forward operator over a column batch in chunks; ``(ncell, nchan)``."""
    n = column.ncol
    out = np.empty((n, len(channels)))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        part = Column(
            pressure_pa=column.pressure_pa,
            temperature_k=column.temperature_k[:, start:stop],
            specific_humidity=column.specific_humidity[:, start:stop],
            surface_pressure_pa=column.surface_pressure_pa[start:stop],
            skin_temperature_k=column.skin_temperature_k[start:stop],
            air_temperature_2m_k=None if column.air_temperature_2m_k is None
            else column.air_temperature_2m_k[start:stop],
        )
        out[start:stop] = brightness_temperature(
            part, channels, zenith_deg[start:stop], scan_angle_deg[start:stop],
            rayleigh_jeans=rayleigh_jeans,
        ).T
    return out


class _Subset:
    """The thinned arrays restricted to an index set, for the screens."""

    _FIELDS = ("tb_mean_k", "tb_std_k", "tb_count", "count", "lat_mean_deg", "lon_mean_deg",
               "zenith_mean_deg", "scan_angle_abs_mean_deg", "time_mean_unix_s")

    def __init__(self, thinned, index: np.ndarray):
        for name in self._FIELDS:
            setattr(self, name, np.asarray(getattr(thinned, name))[index])
        self.ncell = int(index.size)


@dataclass
class DayScore:
    """Everything one scoring pass produced."""

    scores: list[ChannelScore]
    screen: Screen
    #: Indices into the thinned cells of the scored (screen-surviving) cells.
    index: np.ndarray
    observed: np.ndarray  # (ncell, nchan)
    background: np.ndarray  # (ncell, nchan)
    surface: dict[str, np.ndarray]
    column: Column
    zenith_deg: np.ndarray
    scan_angle_deg: np.ndarray
    wind_speed_m_s: np.ndarray
    cells_by_hour: dict[str, int]
    operator_seconds: float
    sampling_seconds: float


def score_day(thinned, analyses, *, options: ScreenOptions | None = None, bar_k: float = 1.0,
              max_cells: int | None = None, seed: int = 20260901, channels=ALL_CHANNELS,
              column_chunk: int = 20000, operator_chunk: int = 256) -> DayScore:
    """Screen a day of thinned cells against the analyses, run the operator
    on the survivors and score every channel.

    Two sampling stages keep the cost on the survivors: the surface records
    are sampled under every candidate cell for the screens, and full
    columns only under the cells that pass, in chunks of ``column_chunk``.
    ``max_cells`` subsamples the candidates at random (recorded; a probe,
    not the reading of record)."""
    from .columns import _ordered_analyses, sample_columns

    options = options or ScreenOptions()
    ordered = _ordered_analyses(list(analyses))
    lat = np.asarray(thinned.lat_mean_deg, dtype=np.float64)
    lon = np.asarray(thinned.lon_mean_deg, dtype=np.float64)
    t = np.asarray(thinned.time_mean_unix_s, dtype=np.float64)

    candidate = candidate_mask(thinned, options)
    stages = {"cells_total": int(thinned.ncell), "cells_candidate": int(candidate.sum())}
    if max_cells is not None and candidate.sum() > max_cells:
        rng = np.random.default_rng(seed)
        keep = rng.choice(np.flatnonzero(candidate), size=max_cells, replace=False)
        candidate = np.zeros_like(candidate)
        candidate[keep] = True
        stages["cells_subsampled"] = int(max_cells)
    index = np.flatnonzero(candidate)

    sampling_started = time.monotonic()
    surface_all: dict[str, np.ndarray] = {}
    w_earlier = np.zeros(index.size)
    for start in range(0, index.size, column_chunk):
        part = index[start:start + column_chunk]
        sampled = sample_columns(ordered, lat[part], lon[part], t[part], surface_only=True)
        for name, values in sampled.surface.items():
            surface_all.setdefault(name, np.zeros(index.size))[start:start + part.size] = values
        w_earlier[start:start + part.size] = sampled.weights_earlier
    sub = _Subset(thinned, index)
    screen = screen_cells(sub, surface_all, w_earlier, options)
    screen.stages = {**stages, **screen.stages}
    keep = screen.mask
    scored_index = index[keep]

    pressure = ordered[0].pressure_pa
    nlev = pressure.size
    temperature = np.zeros((nlev, scored_index.size))
    humidity = np.zeros((nlev, scored_index.size))
    for start in range(0, scored_index.size, column_chunk):
        part = scored_index[start:start + column_chunk]
        sampled = sample_columns(ordered, lat[part], lon[part], t[part])
        temperature[:, start:start + part.size] = sampled.column.temperature_k
        humidity[:, start:start + part.size] = sampled.column.specific_humidity
    surface = {name: values[keep] for name, values in surface_all.items()}
    sampling_seconds = time.monotonic() - sampling_started
    column = Column(
        pressure_pa=pressure,
        temperature_k=temperature,
        specific_humidity=humidity,
        surface_pressure_pa=surface["surface_pressure"],
        skin_temperature_k=surface["skin_temperature"],
        air_temperature_2m_k=surface["air_temperature_2m"],
    )
    zenith = np.asarray(thinned.zenith_mean_deg, dtype=np.float64)[scored_index]
    scan = np.asarray(thinned.scan_angle_abs_mean_deg, dtype=np.float64)[scored_index]
    operator_started = time.monotonic()
    background = background_for_cells(column, zenith, scan, channels, chunk=operator_chunk)
    operator_seconds = time.monotonic() - operator_started

    wind = np.hypot(surface["eastward_wind_10m"], surface["northward_wind_10m"])
    channel_index = np.asarray([c - 1 for c in channels])
    observed = np.asarray(thinned.tb_mean_k, dtype=np.float64)[scored_index][:, channel_index]
    cell_std = np.asarray(thinned.tb_std_k, dtype=np.float64)[scored_index][:, channel_index]
    beam_count = np.asarray(thinned.tb_count, dtype=np.float64)[scored_index][:, channel_index]
    scores = score_channels(
        observed, background,
        zenith_deg=zenith, latitude_deg=lat[scored_index], wind_speed_m_s=wind,
        precipitable_water_kg_m2=surface["precipitable_water"], time_unix_s=t[scored_index],
        cell_std_k=cell_std, beam_count=beam_count, channels=channels, options=options, bar_k=bar_k,
    )
    hours: dict[str, int] = {}
    for stamp in t[scored_index]:
        key = dt.datetime.fromtimestamp(float(stamp), tz=dt.timezone.utc).strftime("%Y-%m-%dT%HZ")
        hours[key] = hours.get(key, 0) + 1
    return DayScore(
        scores=scores, screen=screen, index=scored_index, observed=observed, background=background,
        surface=surface, column=column, zenith_deg=zenith, scan_angle_deg=scan, wind_speed_m_s=wind,
        cells_by_hour=dict(sorted(hours.items())), operator_seconds=operator_seconds,
        sampling_seconds=sampling_seconds,
    )


def nearest_term(score: ChannelScore, bar_k: float) -> str:
    """Name the term a failing channel is closest to, from its own diagnostics."""
    if score.within_bar:
        return "inside the bar"
    if score.within_bar_with_wind:
        return SURFACE_TERM
    if np.isfinite(score.noise_floor_k) and score.noise_floor_k > 0.7 * bar_k:
        return NOISE_TERM
    if score.channel >= 14:
        return UPPER_TERM
    if score.channel <= 5:
        return SURFACE_TERM
    return "the analysis column (no single diagnostic dominates)"


def write_scorecard(path: str | Path, *, scores: list[ChannelScore], screen: Screen,
                    options: ScreenOptions, provenance: dict, bar_k: float) -> dict:
    sounding = [s for s in scores if s.channel in TEMPERATURE_SOUNDING_CHANNELS]
    admitted = sorted(s.channel for s in sounding if s.within_bar)
    outside = sorted(s.channel for s in sounding if not s.within_bar)
    verdict = {
        "bar_k": bar_k,
        "correction_of_record": "geometry (a + b (B - mean_B) + c (sec z - 1)), fitted on the even "
                                "half of the time order, scored on the odd half",
        "sounding_channels": list(TEMPERATURE_SOUNDING_CHANNELS),
        "within_bar": admitted,
        "outside_bar": outside,
        "within_bar_background_only": sorted(s.channel for s in sounding if s.within_bar_background_only),
        "within_bar_with_wind": sorted(s.channel for s in sounding if s.within_bar_with_wind),
        "nearest_term": {s.channel: nearest_term(s, bar_k) for s in sounding if not s.within_bar},
        "gap_k": {s.channel: round(s.geometry_corrected["rmse"] - bar_k, 3)
                  for s in sounding if not s.within_bar and s.geometry_corrected["count"] > 0},
        "admitted_channels": admitted,
        "operator_entry_ships": bool(admitted),
        "every_sounding_channel_within_bar": not outside,
    }
    document = {
        "schema": "gpuwm-arwen-global-microwave-scorecard-v2",
        "screen": screen.stages,
        "screen_options": asdict(options),
        "channels": [asdict(s) for s in scores],
        "verdict": verdict,
        "provenance": provenance,
    }
    Path(path).write_text(json.dumps(document, indent=1, default=_json_default), encoding="utf-8")
    return document


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    raise TypeError(f"not JSON serialisable: {type(value)!r}")
