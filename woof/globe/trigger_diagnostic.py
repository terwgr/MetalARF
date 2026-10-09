"""The convection trigger diagnostic: grid-scale rain against the cumulus scheme, per land cell and hour.

What it measures.  For every interval between two consecutive checkpoints
of a run and every cell, the grid-scale rain the microphysics put on the
ground (the surface rain, snow and graupel accumulators) and the
convective rain the cumulus scheme booked (``physics__rainc``), and
whether the cumulus scheme was ACTIVE in that column that interval,
defined as: the scheme booked convective rain there (its accumulator
grew).  That is the deep-convective trigger as the ground sees it; a
shallow plume that rains nothing is not counted as active, and the
reading says so.  With those two per-cell series it reports, per region
and per hour of local solar time (UTC + longitude / 15, the diurnal
instrument's own convention),

* the composite rates of grid-scale, convective and total rain (mm/h,
  area-weighted over the region's cells, every cell's interval binned by
  the local solar hour of its midpoint),
* the fraction of the grid-scale rain that fell in columns where the
  cumulus scheme was inactive that hour, and the same fraction over the
  noon window (local 11 to 13) as the headline: "grid-scale condensation
  before the trigger" as one number,
* per cell, the lead of the first grid-scale rain over the first
  convective rain (hours; positive when the grid scale rained first),
  the fraction of the region's raining cells in each order, and the share
  of the region's grid-scale rain carried by columns the cumulus scheme
  never touched in the window,
* the state of the raining columns at the end of the interval
  (:mod:`woof.globe.column_sounding`): surface-based and
  most-unstable CAPE, the column's highest relative humidity, the mean
  relative humidity of its cloudy levels (grid-scale condensate present,
  the levels the microphysics condensed at) and the pressure of its
  highest cloudy level, split by active and inactive columns.

Calibration (``--calibrate``; ``tests/test_trigger_diagnostic_instrument.py``
asserts every row): planted accumulator series on a T21 grid in both
directions of every reading (the inactive fraction 0, 0.3 and 1 by
area, the lead +3 h and -3 h, a column saturated at planted levels and
a dry one), and the analytic CAPE soundings of the sounding module.

Reads hash-verified checkpoints through ``read_checkpoint``; the run's
receipt supplies the grid and the vertical coordinate; the instrument
tree must be the tree that wrote the checkpoints (schema v3).  Writes
one JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import column_sounding as cs
from .diurnal_phase import REGIONS, _iso, gaussian_grid_coordinates, region_mask

SCHEMA = "gpuwm.arwen-global-trigger-diagnostic/v1"
#: Local solar hours of the noon window, [start, end).
NOON_WINDOW_LST_H = (11.0, 13.0)
#: A level counts as saturated at or above this relative humidity.
SATURATED_RH = 0.99
#: An inactive raining column counts as convectively capable (the cumulus
#: scheme had instability to work with while the grid scale condensed)
#: at or above this most-unstable CAPE, with at least one saturated level.
CAPABLE_CAPE_J_KG = 200.0
HOURS_PER_DAY = 24.0
DEFAULT_REGIONS = ("conus_land", "global_land")


@dataclass(frozen=True)
class IntervalSample:
    """One checkpoint interval: rain increments and the end-of-interval
    column state, all ``(ny, nx)``.  State arrays may be None when the
    caller has no soundings (accumulator-only readings still work)."""

    start_utc_s: float
    end_utc_s: float
    grid_kg_m2: np.ndarray
    convective_kg_m2: np.ndarray
    cape_sb_j_kg: np.ndarray | None = None
    cape_mu_j_kg: np.ndarray | None = None
    rh_max: np.ndarray | None = None
    rh_cloudy_mean: np.ndarray | None = None
    cloudy_levels: np.ndarray | None = None
    cloud_top_pa: np.ndarray | None = None
    saturated_levels: np.ndarray | None = None

    @property
    def hours(self) -> float:
        return (self.end_utc_s - self.start_utc_s) / 3600.0

    @property
    def active(self) -> np.ndarray:
        return self.convective_kg_m2 > 0.0


def local_solar_hour(utc_s: float, longitude_deg: np.ndarray) -> np.ndarray:
    lon = np.where(longitude_deg > 180.0, longitude_deg - 360.0, longitude_deg)
    utc_hour = (float(utc_s) % 86400.0) / 3600.0
    return np.mod(utc_hour + lon / 15.0, HOURS_PER_DAY)


def column_state(sounding: cs.Sounding) -> dict[str, np.ndarray]:
    """The per-column state readings of one sounding."""
    cape = cs.parcel_cape(sounding.temperature_k, sounding.qv, sounding.p_full, sounding.p_half)
    rh = sounding.relative_humidity
    cloudy = sounding.cloudy()
    n_cloudy = np.sum(cloudy, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rh_cloudy = np.where(n_cloudy > 0, np.sum(np.where(cloudy, rh, 0.0), axis=0) / np.maximum(n_cloudy, 1), np.nan)
    nlev = rh.shape[0]
    first_cloudy = np.argmax(cloudy, axis=0)
    cloud_top = np.where(n_cloudy > 0, np.take_along_axis(sounding.p_full, first_cloudy[None], axis=0)[0], np.nan)
    return {
        "cape_sb_j_kg": cape["cape_sb_j_kg"],
        "cape_mu_j_kg": cape["cape_mu_j_kg"],
        "rh_max": np.max(rh, axis=0),
        "rh_cloudy_mean": rh_cloudy,
        "cloudy_levels": n_cloudy.astype(np.int64),
        "cloud_top_pa": cloud_top,
        "saturated_levels": np.sum(rh >= SATURATED_RH, axis=0).astype(np.int64),
        "nlev": nlev,
    }


def interval_from_soundings(
    previous: cs.Sounding, current: cs.Sounding, start_utc_s: float, *, state: bool = True,
) -> IntervalSample:
    grid = current.grid_scale_kg_m2 - previous.grid_scale_kg_m2
    conv = current.convective_kg_m2 - previous.convective_kg_m2
    fields = column_state(current) if state else {}
    return IntervalSample(
        start_utc_s=start_utc_s + previous.time_s,
        end_utc_s=start_utc_s + current.time_s,
        grid_kg_m2=np.maximum(grid, 0.0),
        convective_kg_m2=np.maximum(conv, 0.0),
        cape_sb_j_kg=fields.get("cape_sb_j_kg"),
        cape_mu_j_kg=fields.get("cape_mu_j_kg"),
        rh_max=fields.get("rh_max"),
        rh_cloudy_mean=fields.get("rh_cloudy_mean"),
        cloudy_levels=fields.get("cloudy_levels"),
        cloud_top_pa=fields.get("cloud_top_pa"),
        saturated_levels=fields.get("saturated_levels"),
    )


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> dict[str, float | None]:
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    keep = np.isfinite(v) & (w > 0.0)
    v, w = v[keep], w[keep]
    if v.size == 0:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    order = np.argsort(v)
    v, w = v[order], w[order]
    cumulative = np.cumsum(w) / np.sum(w)

    def quantile(q: float) -> float:
        return float(v[min(int(np.searchsorted(cumulative, q)), v.size - 1)])

    return {
        "n": int(v.size),
        "mean": float(np.sum(v * w) / np.sum(w)),
        "median": quantile(0.5),
        "p10": quantile(0.1),
        "p90": quantile(0.9),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0.0 else None


def measure(
    samples: list[IntervalSample],
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    land: np.ndarray,
    cell_weights: np.ndarray,
    *,
    regions=DEFAULT_REGIONS,
    noon_window=NOON_WINDOW_LST_H,
    capable_cape: float = CAPABLE_CAPE_J_KG,
) -> dict[str, object]:
    """Every reading of the module docstring, per region."""
    if len(samples) < 1:
        raise ValueError("at least one checkpoint interval is needed")
    samples = sorted(samples, key=lambda s: s.start_utc_s)
    lat2, lon2 = np.meshgrid(latitude_deg, longitude_deg, indexing="ij")
    ny, nx = lat2.shape
    for s in samples:
        if s.grid_kg_m2.shape != (ny, nx) or s.convective_kg_m2.shape != (ny, nx):
            raise ValueError("every sample must be on the (ny, nx) grid of the coordinates")
        if s.end_utc_s <= s.start_utc_s:
            raise ValueError("interval end must follow its start")
    have_state = all(s.cape_sb_j_kg is not None for s in samples)
    noon_lo, noon_hi = float(noon_window[0]), float(noon_window[1])
    out_regions: dict[str, object] = {}
    for region in regions:
        mask = region_mask(region, lat2, lon2, land)
        w = np.where(mask, cell_weights, 0.0)
        region_area = float(np.sum(w))
        if region_area <= 0.0:
            out_regions[region] = {"status": "no cells"}
            continue
        bins = np.arange(int(HOURS_PER_DAY))
        grid_sum = np.zeros(bins.size)
        conv_sum = np.zeros(bins.size)
        grid_inactive_sum = np.zeros(bins.size)
        hours_sum = np.zeros(bins.size)
        raining_cells = np.zeros(bins.size)
        active_cells = np.zeros(bins.size)
        noon_grid = 0.0
        noon_grid_inactive = 0.0
        noon_grid_inactive_unstable = 0.0
        noon_grid_inactive_capable = 0.0
        noon_count = 0
        noon_count_inactive = 0
        noon_state = {"active": {}, "inactive": {}}
        state_values = {"active": {}, "inactive": {}}
        first_grid = np.full((ny, nx), np.nan)
        first_conv = np.full((ny, nx), np.nan)
        total_grid_cell = np.zeros((ny, nx))
        total_conv_cell = np.zeros((ny, nx))
        saturated_any = 0.0
        raining_any = 0.0
        for s in samples:
            mid = 0.5 * (s.start_utc_s + s.end_utc_s)
            lst = local_solar_hour(mid, lon2)
            bin_index = np.minimum(np.floor(lst).astype(np.int64), bins.size - 1)
            active = s.active
            g = s.grid_kg_m2
            c = s.convective_kg_m2
            inactive_grid = np.where(active, 0.0, g)
            np.add.at(grid_sum, bin_index[mask], (w * g)[mask])
            np.add.at(conv_sum, bin_index[mask], (w * c)[mask])
            np.add.at(grid_inactive_sum, bin_index[mask], (w * inactive_grid)[mask])
            np.add.at(hours_sum, bin_index[mask], (w * s.hours)[mask])
            np.add.at(raining_cells, bin_index[mask], (w * (g > 0.0))[mask])
            np.add.at(active_cells, bin_index[mask], (w * active)[mask])
            in_noon = mask & (lst >= noon_lo) & (lst < noon_hi)
            noon_grid += float(np.sum((w * g)[in_noon]))
            noon_grid_inactive += float(np.sum((w * inactive_grid)[in_noon]))
            noon_raining = in_noon & (g > 0.0)
            noon_count += int(np.count_nonzero(noon_raining))
            noon_count_inactive += int(np.count_nonzero(noon_raining & ~active))
            if have_state:
                unstable = s.cape_mu_j_kg >= capable_cape
                capable = unstable & (s.saturated_levels >= 1)
                noon_grid_inactive_unstable += float(np.sum((w * inactive_grid)[in_noon & unstable]))
                noon_grid_inactive_capable += float(np.sum((w * inactive_grid)[in_noon & capable]))
                for label, sel in (("active", noon_raining & active), ("inactive", noon_raining & ~active)):
                    for name in ("cape_sb_j_kg", "cape_mu_j_kg", "rh_max", "rh_cloudy_mean", "cloud_top_pa", "cloudy_levels", "saturated_levels"):
                        field = getattr(s, name)
                        state_values[label].setdefault(name, []).append((field[sel], w[sel]))
                raining = mask & (g > 0.0)
                raining_any += float(np.sum(w[raining]))
                saturated_any += float(np.sum(w[raining & (s.saturated_levels > 0)]))
            newly_grid = mask & (g > 0.0) & np.isnan(first_grid)
            first_grid = np.where(newly_grid, s.start_utc_s, first_grid)
            newly_conv = mask & (c > 0.0) & np.isnan(first_conv)
            first_conv = np.where(newly_conv, s.start_utc_s, first_conv)
            total_grid_cell += np.where(mask, g, 0.0)
            total_conv_cell += np.where(mask, c, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            grid_rate = np.where(hours_sum > 0, grid_sum / hours_sum, np.nan)
            conv_rate = np.where(hours_sum > 0, conv_sum / hours_sum, np.nan)
            total_rate = grid_rate + conv_rate
            inactive_fraction = np.where(grid_sum > 0, grid_inactive_sum / grid_sum, np.nan)
        for label in ("active", "inactive"):
            for name, parts in state_values[label].items():
                values = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0)
                weights = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0)
                noon_state[label][name] = _weighted_stats(values, weights)
        both = mask & np.isfinite(first_grid) & np.isfinite(first_conv)
        lead_h = (first_conv - first_grid) / 3600.0
        grid_only = mask & np.isfinite(first_grid) & np.isnan(first_conv)
        conv_only = mask & np.isnan(first_grid) & np.isfinite(first_conv)
        region_grid_total = float(np.sum(w * total_grid_cell))
        region_conv_total = float(np.sum(w * total_conv_cell))
        both_area = float(np.sum(w[both]))
        peak_bin = int(np.nanargmax(total_rate)) if np.any(np.isfinite(total_rate)) else None

        def centroid(rate: np.ndarray) -> float | None:
            r = np.where(np.isfinite(rate), rate, 0.0)
            if np.sum(r) <= 0.0:
                return None
            angle = 2.0 * math.pi * (bins + 0.5) / HOURS_PER_DAY
            x = float(np.sum(r * np.cos(angle)))
            y = float(np.sum(r * np.sin(angle)))
            return float((math.atan2(y, x) / (2.0 * math.pi) * HOURS_PER_DAY) % HOURS_PER_DAY)

        out_regions[region] = {
            "status": "measured",
            "area_fraction_of_globe": region_area,
            "by_local_hour": {
                "hour": [int(b) for b in bins],
                "grid_scale_mm_h": [_finite(v) for v in grid_rate],
                "convective_mm_h": [_finite(v) for v in conv_rate],
                "total_mm_h": [_finite(v) for v in total_rate],
                "grid_scale_inactive_fraction": [_finite(v) for v in inactive_fraction],
                "raining_cell_fraction": [_finite(v) for v in np.where(hours_sum > 0, raining_cells / (hours_sum / np.mean([s.hours for s in samples])), np.nan)],
                "active_cell_fraction": [_finite(v) for v in np.where(hours_sum > 0, active_cells / (hours_sum / np.mean([s.hours for s in samples])), np.nan)],
            },
            "composite": {
                "peak_local_hour_bin": peak_bin,
                "centroid_local_hour_total": centroid(total_rate),
                "centroid_local_hour_grid_scale": centroid(grid_rate),
                "centroid_local_hour_convective": centroid(conv_rate),
                "grid_scale_mm_day": region_grid_total / region_area,
                "convective_mm_day": region_conv_total / region_area,
            },
            "noon_window": {
                "local_hours": [noon_lo, noon_hi],
                "grid_scale_mm_region_mean": noon_grid / region_area,
                "grid_scale_in_inactive_columns_mm_region_mean": noon_grid_inactive / region_area,
                "fraction_of_grid_scale_rain_in_inactive_columns": _ratio(noon_grid_inactive, noon_grid),
                "fraction_of_grid_scale_rain_in_inactive_unstable_columns": _ratio(noon_grid_inactive_unstable, noon_grid) if have_state else None,
                "fraction_of_grid_scale_rain_in_inactive_capable_columns": _ratio(noon_grid_inactive_capable, noon_grid) if have_state else None,
                "capable_definition": f"cumulus inactive, most-unstable CAPE >= {capable_cape:g} J/kg and at least one saturated level at the interval end; unstable drops the saturation condition",
                "raining_cell_intervals": noon_count,
                "raining_cell_intervals_inactive": noon_count_inactive,
                "fraction_of_raining_cell_intervals_inactive": _ratio(noon_count_inactive, noon_count),
                "column_state_at_interval_end": noon_state if have_state else "no soundings",
            },
            "all_hours": {
                "fraction_of_grid_scale_rain_in_inactive_columns": _ratio(float(np.sum(grid_inactive_sum)), float(np.sum(grid_sum))),
                "fraction_of_raining_columns_saturated_at_interval_end": _ratio(saturated_any, raining_any) if have_state else None,
            },
            "order_of_first_rain": {
                "definition": "first interval with grid-scale rain against first interval with convective rain, per cell; lead = t_conv - t_grid in hours, positive when the grid scale rained first",
                "cells_with_both_area": both_area,
                "grid_first_fraction": _ratio(float(np.sum(w[both & (lead_h > 0.0)])), both_area),
                "same_interval_fraction": _ratio(float(np.sum(w[both & (lead_h == 0.0)])), both_area),
                "convective_first_fraction": _ratio(float(np.sum(w[both & (lead_h < 0.0)])), both_area),
                "lead_hours": _weighted_stats(np.where(both, lead_h, np.nan), w),
                "grid_scale_rain_share_in_columns_without_convection": _ratio(float(np.sum((w * total_grid_cell)[grid_only])), region_grid_total),
                "convective_rain_share_in_columns_without_grid_scale": _ratio(float(np.sum((w * total_conv_cell)[conv_only])), region_conv_total),
            },
        }
    return {
        "schema": SCHEMA,
        "measures": (
            "per checkpoint interval and cell: grid-scale rain (surface rain + snow + graupel accumulators), "
            "convective rain (physics__rainc), cumulus active := convective rain booked that interval; "
            "local solar hour = UTC + longitude/15 of the interval midpoint; area-weighted by Gaussian cell area"
        ),
        "active_definition": "the cumulus scheme booked convective rain in the column during the interval (RAINC grew); a shallow plume without rain is not counted",
        "noon_window_lst_h": [noon_lo, noon_hi],
        "n_intervals": len(samples),
        "interval_hours": [float(s.hours) for s in samples],
        "window_utc": [_iso(samples[0].start_utc_s), _iso(samples[-1].end_utc_s)],
        "column_state": (
            {
                "saturation": cs.SATURATION_FORMULA,
                "cloudy_level": f"qc + qi > {cs.CLOUDY_LEVEL_KG_KG:g} kg/kg",
                "saturated_level": f"relative humidity >= {SATURATED_RH}",
                "cape": "surface-based and most-unstable parcel, pseudo-adiabatic, virtual-temperature buoyancy (column_sounding)",
                "when": "the checkpoint at the END of the interval",
            }
            if have_state else "not read"
        ),
        "regions": out_regions,
    }


def _finite(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def summary_line(result: dict, label: str = "") -> str:
    parts = [f"trigger {label}".strip()]
    for region, block in result["regions"].items():
        if block.get("status") != "measured":
            parts.append(f"{region}: {block.get('status')}")
            continue
        noon = block["noon_window"]
        order = block["order_of_first_rain"]
        comp = block["composite"]
        frac = noon["fraction_of_grid_scale_rain_in_inactive_columns"]
        capable = noon.get("fraction_of_grid_scale_rain_in_inactive_capable_columns")
        parts.append(
            f"{region}: noon grid-scale rain in inactive columns "
            f"{'n/a' if frac is None else f'{frac:.2f}'} "
            f"(capable {'n/a' if capable is None else f'{capable:.2f}'}) "
            f"(all hours {block['all_hours']['fraction_of_grid_scale_rain_in_inactive_columns']:.2f}); "
            f"grid first {order['grid_first_fraction']}, conv first {order['convective_first_fraction']}, "
            f"lead median {order['lead_hours']['median']} h; "
            f"centroid LST total {comp['centroid_local_hour_total']}, grid {comp['centroid_local_hour_grid_scale']}, "
            f"conv {comp['centroid_local_hour_convective']}; "
            f"mm/day grid {comp['grid_scale_mm_day']:.2f} conv {comp['convective_mm_day']:.2f}"
        )
    return " | ".join(parts)


# --------------------------------------------------------------------------
# run door
# --------------------------------------------------------------------------


def measure_run(
    run_dir: str | Path,
    *,
    start_utc_s: float | None = None,
    checkpoints: list[str | Path] | None = None,
    config: str | Path | None = None,
    regions=DEFAULT_REGIONS,
    state: bool = True,
    progress=None,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    receipt = cs.read_receipt_or_config(run_dir, config)
    if start_utc_s is None:
        start_utc_s = cs.run_start_utc_s(receipt)
    if start_utc_s is None:
        raise ValueError("neither the receipt nor the config carries start_time_utc; pass --start")
    paths = sorted(Path(p) for p in (checkpoints or run_dir.glob("arwen_global_step*.npz")))
    if len(paths) < 2:
        raise ValueError(f"{len(paths)} checkpoints give no interval; at least 2 are needed")
    reader = cs.SoundingReader(receipt, omega=False)
    samples: list[IntervalSample] = []
    previous = None
    for path in paths:
        current = reader.sounding(path)
        if progress is not None:
            progress(f"read {path.name} step {current.step}")
        if previous is not None:
            if current.time_s <= previous.time_s:
                raise ValueError("checkpoint times are not strictly increasing")
            samples.append(interval_from_soundings(previous, current, start_utc_s, state=state))
        previous = current
    land = previous.land
    result = measure(samples, reader.latitude_deg, reader.longitude_deg, land, reader.cell_weights, regions=regions)
    result["model_source"] = {
        "run_dir": str(run_dir),
        "checkpoints": [str(p) for p in paths],
        "config_hash": reader.config_hash,
        "grid": {"kind": "gaussian", "nlat": reader.shape[0], "nlon": reader.shape[1]},
        "land_mask": f"{cs.CHECKPOINT_LAND_FRACTION} >= 0.5",
        "synthesis": "numpy float64 from the checkpoint's coefficients; grid tracers read as stored",
    }
    return result


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def synthetic_geometry(nlat: int = 32, nlon: int = 64):
    from woof.globe.spectral.grid import GaussianGrid

    grid = GaussianGrid.for_shape(nlat, nlon)
    lat = np.asarray(grid.latitude_deg, dtype=np.float64)
    lon = np.asarray(grid.longitude_deg, dtype=np.float64)
    weights = np.repeat((np.asarray(grid.quadrature_weights) / (2.0 * nlon))[:, None], nlon, axis=1)
    return lat, lon, weights


def synthetic_samples(
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    weights: np.ndarray,
    region: str = "conus_land",
    inactive_area_fraction: float = 0.5,
    grid_lst_h: float = 12.0,
    convective_lst_h: float = 12.0,
    grid_mm: float = 3.0,
    convective_mm: float = 1.0,
    start_utc_s: float = 0.0,
    hours: int = 24,
) -> tuple[list[IntervalSample], np.ndarray, float]:
    """Hourly intervals over one day on an all-land grid.  Every cell of
    ``region`` receives ``grid_mm`` of grid-scale rain in the interval
    whose local solar midpoint hour is ``grid_lst_h``; the cells making up
    the LOWEST-area ``inactive_area_fraction`` of the region (by cell
    weight, cells sorted by longitude then latitude) receive no convective
    rain, the rest receive ``convective_mm`` in the interval at
    ``convective_lst_h``.  Returns the samples, the land mask and the
    inactive fraction actually planted (by area, exactly)."""
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    land = np.ones(lat2.shape, dtype=bool)
    mask = region_mask(region, lat2, lon2, land)
    cells = np.argwhere(mask)
    order = np.lexsort((cells[:, 0], cells[:, 1]))
    cells = cells[order]
    w_cells = weights[cells[:, 0], cells[:, 1]]
    target = inactive_area_fraction * float(np.sum(w_cells))
    cumulative = np.cumsum(w_cells)
    n_inactive = int(np.searchsorted(cumulative, target, side="right"))
    if inactive_area_fraction >= 1.0:
        n_inactive = cells.shape[0]
    inactive = np.zeros(lat2.shape, dtype=bool)
    inactive[cells[:n_inactive, 0], cells[:n_inactive, 1]] = True
    planted_fraction = float(np.sum(w_cells[:n_inactive]) / np.sum(w_cells)) if grid_lst_h == convective_lst_h else 1.0
    samples = []
    grid_acc = np.zeros(lat2.shape)
    conv_acc = np.zeros(lat2.shape)
    for h in range(hours):
        start = start_utc_s + 3600.0 * h
        end = start + 3600.0
        lst = local_solar_hour(0.5 * (start + end), lon2)
        g = np.where(mask & (np.floor(lst) == math.floor(grid_lst_h)), grid_mm, 0.0)
        c = np.where(mask & ~inactive & (np.floor(lst) == math.floor(convective_lst_h)), convective_mm, 0.0)
        grid_acc += g
        conv_acc += c
        samples.append(IntervalSample(start, end, g, c))
    return samples, land, planted_fraction


def synthetic_sounding(
    nlev: int = 20, *, cloudy_levels: tuple[int, int] | None = (8, 11), rh_elsewhere: float = 0.5,
) -> cs.Sounding:
    """One column (a 1 x 1 grid) at 280 K isothermal, saturated with cloud
    water on ``cloudy_levels`` (inclusive, top-down indices) and at
    ``rh_elsewhere`` elsewhere."""
    ph = np.exp(np.linspace(np.log(10000.0), np.log(100000.0), nlev + 1))[:, None, None]
    pf = np.sqrt(ph[:-1] * ph[1:])
    t = np.full(pf.shape, 280.0)
    qs = cs.saturation_mixing_ratio(t, pf)
    qv = rh_elsewhere * qs
    qc = np.zeros(pf.shape)
    if cloudy_levels is not None:
        k0, k1 = cloudy_levels
        qv[k0:k1 + 1] = qs[k0:k1 + 1]
        qc[k0:k1 + 1] = 1.0e-4
    zeros = np.zeros(pf.shape)
    return cs.Sounding(
        time_s=0.0, step=0, p_full=pf, p_half=ph, temperature_k=t, qv=qv,
        condensate={"qc": qc, "qr": zeros.copy(), "qi": zeros.copy(), "qs": zeros.copy(), "qg": zeros.copy()},
        relative_humidity=cs.relative_humidity(t, pf, qv), omega_half_pa_s=None,
        convective_kg_m2=np.zeros((1, 1)), grid_scale_kg_m2=np.zeros((1, 1)),
        land=np.ones((1, 1), dtype=bool), latitude_deg=np.zeros(1), longitude_deg=np.zeros(1),
    )


def calibrate() -> dict[str, object]:
    """Every synthetic family in both directions; each row carries its
    bar and whether it met it."""
    from .constants import DRY_AIR_GAS_CONSTANT, KAPPA

    rows = []
    lat, lon, weights = synthetic_geometry()
    for planted in (0.0, 0.3, 1.0):
        samples, land, exact = synthetic_samples(lat=lat, lon=lon, weights=weights, inactive_area_fraction=planted)
        result = measure(samples, lat, lon, land, weights, regions=("conus_land",))
        read = result["regions"]["conus_land"]["noon_window"]["fraction_of_grid_scale_rain_in_inactive_columns"]
        rows.append({
            "family": "inactive fraction", "planted": exact, "read": read,
            "bar": 1.0e-12, "ok": abs(read - exact) <= 1.0e-12,
        })
        centroid = result["regions"]["conus_land"]["composite"]["centroid_local_hour_grid_scale"]
        rows.append({
            "family": "grid-scale centroid LST", "planted": 12.5, "read": centroid,
            "bar": 1.0e-9, "ok": abs(centroid - 12.5) <= 1.0e-9,
        })
    for capable_fraction in (0.0, 0.5, 1.0):
        samples, land, exact = synthetic_samples(lat=lat, lon=lon, weights=weights, inactive_area_fraction=1.0)
        planted = []
        for s in samples:
            ny, nx = s.grid_kg_m2.shape
            cape = np.zeros((ny, nx))
            sat = np.zeros((ny, nx), dtype=np.int64)
            # Capable cells: the first fraction of the raining cells by longitude index.
            raining = s.grid_kg_m2 > 0.0
            idx = np.argwhere(raining)
            n_cap = int(round(capable_fraction * idx.shape[0]))
            for j, i in idx[:n_cap]:
                cape[j, i] = 2.0 * CAPABLE_CAPE_J_KG
                sat[j, i] = 1
            zeros = np.zeros((ny, nx))
            planted.append(IntervalSample(
                s.start_utc_s, s.end_utc_s, s.grid_kg_m2, s.convective_kg_m2,
                cape_sb_j_kg=cape, cape_mu_j_kg=cape, rh_max=zeros, rh_cloudy_mean=zeros,
                cloudy_levels=sat, cloud_top_pa=zeros, saturated_levels=sat,
            ))
        result = measure(planted, lat, lon, land, weights, regions=("conus_land",))
        noon = result["regions"]["conus_land"]["noon_window"]
        # The planted share by area: the capable cells' weight over the raining cells' weight.
        expect = 0.0
        total = 0.0
        lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
        for s, ps in zip(samples, planted):
            lst = local_solar_hour(0.5 * (s.start_utc_s + s.end_utc_s), lon2)
            in_noon = (lst >= NOON_WINDOW_LST_H[0]) & (lst < NOON_WINDOW_LST_H[1]) & (s.grid_kg_m2 > 0.0)
            total += float(np.sum((weights * s.grid_kg_m2)[in_noon]))
            expect += float(np.sum((weights * s.grid_kg_m2)[in_noon & (ps.cape_mu_j_kg > 0.0)]))
        expect = expect / total if total > 0 else None
        read = noon["fraction_of_grid_scale_rain_in_inactive_capable_columns"]
        rows.append({
            "family": "capable fraction of noon inactive rain", "planted": expect, "read": read,
            "bar": 1.0e-12, "ok": abs(read - expect) <= 1.0e-12
            and noon["fraction_of_grid_scale_rain_in_inactive_unstable_columns"] == read,
        })
    for grid_h, conv_h, lead in ((10.0, 13.0, 3.0), (13.0, 10.0, -3.0)):
        samples, land, _ = synthetic_samples(
            lat=lat, lon=lon, weights=weights, inactive_area_fraction=0.0,
            grid_lst_h=grid_h, convective_lst_h=conv_h,
        )
        result = measure(samples, lat, lon, land, weights, regions=("conus_land",))
        order = result["regions"]["conus_land"]["order_of_first_rain"]
        read = order["lead_hours"]["median"]
        first = order["grid_first_fraction"] if lead > 0 else order["convective_first_fraction"]
        rows.append({
            "family": "lead of first grid rain", "planted": lead, "read": read,
            "bar": 1.0e-9, "ok": abs(read - lead) <= 1.0e-9 and abs(first - 1.0) <= 1.0e-12,
        })
        noon = result["regions"]["conus_land"]["noon_window"]
        rows.append({
            "family": "noon window empty when rain falls outside it", "planted": 0.0,
            "read": noon["grid_scale_mm_region_mean"], "bar": 0.0,
            "ok": noon["grid_scale_mm_region_mean"] == 0.0 and noon["fraction_of_grid_scale_rain_in_inactive_columns"] is None,
        })
    for cloudy, rh_else in (((8, 11), 0.5), (None, 0.5), ((3, 3), 0.9)):
        sounding = synthetic_sounding(cloudy_levels=cloudy, rh_elsewhere=rh_else)
        state = column_state(sounding)
        if cloudy is None:
            ok = (
                state["cloudy_levels"][0, 0] == 0 and math.isnan(state["rh_cloudy_mean"][0, 0])
                and abs(state["rh_max"][0, 0] - rh_else) <= 1.0e-12 and state["saturated_levels"][0, 0] == 0
            )
            rows.append({"family": "column state, no cloud", "planted": rh_else, "read": float(state["rh_max"][0, 0]), "bar": 1.0e-12, "ok": bool(ok)})
        else:
            n = cloudy[1] - cloudy[0] + 1
            ok = (
                int(state["cloudy_levels"][0, 0]) == n
                and abs(state["rh_cloudy_mean"][0, 0] - 1.0) <= 1.0e-12
                and abs(state["rh_max"][0, 0] - 1.0) <= 1.0e-12
                and int(state["saturated_levels"][0, 0]) == n
                and abs(state["cloud_top_pa"][0, 0] - sounding.p_full[cloudy[0], 0, 0]) == 0.0
            )
            rows.append({"family": "column state, cloudy levels", "planted": n, "read": int(state["cloudy_levels"][0, 0]), "bar": 0, "ok": bool(ok)})
    nlev = 20
    ph = np.exp(np.linspace(np.log(10000.0), np.log(100000.0), nlev + 1))
    pf = np.sqrt(ph[:-1] * ph[1:])
    dry = 300.0 * (pf / pf[-1]) ** KAPPA
    for delta in (2.0, -2.0):
        env = dry.copy()
        env[:-1] -= delta
        cape = cs.parcel_cape(env[:, None], np.zeros((nlev, 1)), pf[:, None], ph[:, None])
        expect = DRY_AIR_GAS_CONSTANT * delta * math.log(ph[nlev - 1] / ph[0])
        read = float(cape["cape_sb_j_kg"][0]) if delta > 0 else float(cape["cin_sb_j_kg"][0])
        rows.append({
            "family": "analytic CAPE" if delta > 0 else "analytic CIN", "planted": expect, "read": read,
            "bar": 1.0e-9 * abs(expect), "ok": abs(read - expect) <= 1.0e-9 * abs(expect),
        })
    return {"schema": SCHEMA + "/calibration", "rows": rows, "all_ok": all(r["ok"] for r in rows)}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, allow_nan=False, default=_json_default))


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not serialisable: {type(value)!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", help="directory of arwen_global_step*.npz checkpoints and the receipt")
    parser.add_argument("--checkpoints", nargs="*", help="explicit checkpoint paths (else every step file in --run-dir)")
    parser.add_argument("--config", help="the run's TOML, for a run still writing (no receipt yet)")
    parser.add_argument("--start", help="run start, ISO UTC; else the receipt's start_time_utc")
    parser.add_argument("--regions", nargs="*", default=list(DEFAULT_REGIONS))
    parser.add_argument("--no-state", action="store_true", help="accumulator readings only (no soundings)")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", help="JSON output path")
    parser.add_argument("--calibrate", action="store_true", help="run the synthetic families and print their numbers")
    args = parser.parse_args(argv)
    if args.calibrate:
        payload = calibrate()
        for row in payload["rows"]:
            print(f"{row['family']:40s} planted {row['planted']!s:>22} read {row['read']!s:>22} {'ok' if row['ok'] else 'FAIL'}")
        if args.out:
            _write_json(Path(args.out), payload)
        return 0 if payload["all_ok"] else 1
    if not args.run_dir and not args.checkpoints:
        parser.error("--run-dir or --checkpoints is required (or --calibrate)")
    start = None
    if args.start:
        from .diurnal_phase import parse_utc

        start = parse_utc(args.start)
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.checkpoints[0]).parent
    result = measure_run(
        run_dir, start_utc_s=start, checkpoints=args.checkpoints, config=args.config, regions=tuple(args.regions),
        state=not args.no_state, progress=lambda text: print(text, file=sys.stderr),
    )
    result["label"] = args.label
    print(summary_line(result, args.label))
    if args.out:
        _write_json(Path(args.out), result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
