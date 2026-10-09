"""Diurnal-phase instrument for surface precipitation.

What it measures
----------------
The first diurnal harmonic of the surface precipitation rate, fitted per
land cell over a window of interval-mean rates whose abscissa is LOCAL
SOLAR TIME.  For every cell the model is

    rate(t) = a0 + a1 cos(w t) + b1 sin(w t),      w = 2 pi / 24 h,

with t the local solar hour of each interval's midpoint (UTC hour plus
longitude / 15; the equation of time, at most 16 minutes through the
year and under one minute on 2026-09-01, is not applied).  The fit is an
ordinary least-squares solve of the 3 x 3 normal equations per cell, so
irregular or coarse sampling (3-hourly checkpoints, a missing hour) is
handled exactly rather than through a Fourier projection that assumes
uniform hourly samples.  Reported per cell:

* amplitude   ``A = hypot(a1, b1)``  in mm/h;
* phase       local solar hour of the harmonic's maximum,
              ``atan2(b1, a1) * 24 / (2 pi)`` modulo 24;
* noise       ``sigma_A``, the standard error of the amplitude from the
              residual variance ``SSR / (n - 3)`` and the normal-equation
              covariance (``sqrt(2/n) * sigma_resid`` for uniform samples);
* resolvable  ``A > absolute_floor_mm_h`` AND ``A > z * sigma_A``.

The noise floor is therefore two-sided and stated in every output: an
absolute floor (default 0.02 mm/h, about half a millimetre per day) that
rejects cells that barely rained, and a z-score floor (default z = 2)
that rejects a cell whose harmonic is not separable from its own
residual.  A single 3 h shower in eight samples has ``A / sigma_A =
sqrt(2)`` and is rejected by the second test, which is the point of it.

Regional readings (CONUS land: 24..50 N, 125..66 W, land fraction >= 0.5;
global land) are cos-latitude weighted over the fitted land cells:

* ``amplitude_mean_mm_h``            weighted mean of ``A`` over all fitted cells;
* ``resolvable_fraction``            weighted fraction of cells passing both floors;
* ``phase_h`` / ``phase_spread_h``   amplitude-weighted circular mean (and
                                     circular standard deviation) of the
                                     resolvable cells' phases;
* ``composite_*``                    the harmonic of the region-mean diurnal
                                     cycle in local time, which is exactly the
                                     weighted mean of the per-cell (a1, b1);
                                     its status decides the region's status.

``status`` is ``"measured"`` when the composite amplitude clears both floors,
``"no-signal"`` when it does not.  A region with no fitted cells raises,
because on a global grid that is a mask or geometry defect, never a reading.

Observations: an hourly (or finer) accumulation series on any lat/lon grid
is aggregated onto the model's intervals (each model interval must be tiled
exactly by observation intervals) and then binned onto the model grid by
AREA-WEIGHTED BINNING through the tree's registered ``cell_average`` remap
(:mod:`woof.verify.obs.regrid`): every observation cell is assigned to
the model cell whose centre is nearest, and the model cell takes the mean
of its observation cells; a model cell with no observation cell within
``max_distance_m`` is unobserved.  It is not a conservative (overlap-area)
remap and the output names it ``cell_average``; the Stage-IV HRAP cells are
equal-area to well under a percent so the mean is the area-weighted mean.
The same harmonic is fitted to the binned observations, and for every cell
where both fits are resolvable the paired difference ``model - obs`` is
wrapped to (-12, 12] h; its cos-latitude-weighted circular mean and
circular spread are reported with ``status`` ``"measured"`` (at least
``min_paired`` cells), ``"no-signal"`` (cells overlap but too few resolve
on both sides) or ``"unpaired"`` (no observations supplied or no overlap).

Model input: WOOF global checkpoints (``arwen_global_step*.npz``) read
directly; consecutive checkpoints give interval-mean rates from the
accumulators (convective ``physics__rainc``; total = that plus the surface
rain, snow and graupel accumulators, the same split the render tape's
RAINC / RAINNC carry).  The grid is the model's own Gaussian grid for the
array shape; the land mask is ``surface__land_fraction >= 0.5``.

Calibration (``python -m woof.globe.diurnal_phase --calibrate``,
tests/test_diurnal_phase_instrument.py, 31 tests), recorded 2026-09-02 on
the CPU test host (numpy 2.5.2, Python 3.14.4, observation binning on the
announced Python fallback because that host stages no Rust bridges):

Family A, pure sinusoid, known phase 16.50 h, 12 x 24 cell grid spanning
all longitudes, hourly (24) and 3-hourly (8) samples, seeded noise:

    amplitude   noise  samples  composite phase err  cell median err  resolvable  status
    1.00 mm/h   0.00   24       0.0 min              0.0 min          1.00        measured
    1.00 mm/h   0.05   24       0.1 min              2.3 min          1.00        measured
    1.00 mm/h   0.20   24       0.3 min              9.0 min          1.00        measured
    1.00 mm/h   0.20   8        0.1 min              16.0 min         1.00        measured
    0.30 mm/h   0.20   24       1.1 min              30.7 min         1.00        measured
    0.05 mm/h   0.20   24       6.2 min              116.9 min        0.31        measured (composite only)
    0.005 mm/h  0.00   24       below absolute floor                  0.00        no-signal
    0.00 mm/h   0.20   24       noise only                            0.14        no-signal

Family B, first harmonic at 17.00 h plus a phase-locked second harmonic
(0.4 of the first), true maximum 17.00 h: composite phase error 0.0 min
at hourly and at 3-hourly sampling, resolvable 1.00, status measured; the
flat family (constant 0.3 mm/h) returns no-signal at zero noise
(resolvable 0.00) and at 0.2 mm/h noise (resolvable 0.13, composite
below the floor).

Both directions, observation field on a 0.5 degree grid binned to the
model grid through ``cell_average``, 288 paired cells: imposed shift +3 h
read as +3.00 h (spread 0.00 h), -3 h read as -3.00 h, no shift read as
0.00 h; with the second harmonic and 0.05 mm/h noise on both sides, +3 h
read as +3.00 h with spread 0.12 h.  All within the 15 minute bar.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "gpuwm.arwen-global-diurnal-phase/v1"

#: Two-sided noise floor defaults; both are stated in every output.
ABSOLUTE_FLOOR_MM_H = 0.02
Z_FLOOR = 2.0
#: Paired readings need this many cells resolvable on both sides.
MIN_PAIRED_CELLS = 20
#: Observation cells farther than this from every model cell centre are
#: left unbinned (a 52 km Gaussian cell's centre-to-corner distance is
#: about 37 km).
DEFAULT_MAX_BIN_DISTANCE_M = 60.0e3
#: The window must sample at least this much of the diurnal cycle, or the
#: first harmonic is not separable from the mean.
MIN_COVERAGE_HOURS = 20.0

HOURS_PER_DAY = 24.0
OMEGA = 2.0 * math.pi / HOURS_PER_DAY

REGIONS: dict[str, dict[str, object]] = {
    "conus_land": {
        "latitude_deg": (24.0, 50.0),
        "longitude_deg": (-125.0, -66.0),
        "land": True,
    },
    "global_land": {"land": True},
}

CHECKPOINT_TOTAL_ACCUMULATORS = (
    "physics__rainc",
    "surface__accumulated_rain_kg_m2",
    "surface__accumulated_snow_kg_m2",
    "surface__accumulated_graupel_kg_m2",
)
CHECKPOINT_CONVECTIVE_ACCUMULATOR = "physics__rainc"
CHECKPOINT_LAND_FRACTION = "surface__land_fraction"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RateSeries:
    """Interval-mean precipitation rates on a lat/lon grid.

    ``start_s`` / ``end_s`` are UTC seconds since the epoch per interval;
    ``rate_mm_h`` is ``(n, ny, nx)``; latitude/longitude are 1-D
    ``(ny,)`` / ``(nx,)`` for a regular grid or 2-D ``(ny, nx)`` for a
    curvilinear one; ``valid`` marks the cells the series observes.
    """

    start_s: np.ndarray
    end_s: np.ndarray
    rate_mm_h: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    valid: np.ndarray | None = None

    def __post_init__(self) -> None:
        start = np.asarray(self.start_s, dtype=np.float64)
        end = np.asarray(self.end_s, dtype=np.float64)
        rate = np.asarray(self.rate_mm_h, dtype=np.float64)
        if start.ndim != 1 or end.shape != start.shape:
            raise ValueError("start_s and end_s must be matching 1-D arrays")
        if rate.ndim != 3 or rate.shape[0] != start.size:
            raise ValueError(
                f"rate_mm_h must be (n, ny, nx) with n = {start.size}; got {rate.shape}"
            )
        if not np.all(end > start):
            raise ValueError("every interval must have end_s > start_s")
        if not np.isfinite(rate).all():
            raise ValueError("rate_mm_h carries non-finite values")
        lat = np.asarray(self.latitude_deg, dtype=np.float64)
        lon = np.asarray(self.longitude_deg, dtype=np.float64)
        ny, nx = rate.shape[1:]
        if lat.ndim == 1:
            if lat.shape != (ny,) or lon.shape != (nx,):
                raise ValueError("1-D coordinates must be (ny,) and (nx,)")
        elif lat.ndim == 2:
            if lat.shape != (ny, nx) or lon.shape != (ny, nx):
                raise ValueError("2-D coordinates must both be (ny, nx)")
        else:
            raise ValueError("coordinates must be 1-D or 2-D")
        if self.valid is not None and np.asarray(self.valid).shape != (ny, nx):
            raise ValueError("valid must be (ny, nx)")
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "end_s", end)
        object.__setattr__(self, "rate_mm_h", rate)
        object.__setattr__(self, "latitude_deg", lat)
        object.__setattr__(self, "longitude_deg", lon)
        if self.valid is not None:
            object.__setattr__(self, "valid", np.asarray(self.valid, dtype=bool))

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.rate_mm_h.shape[1]), int(self.rate_mm_h.shape[2])

    @property
    def n_samples(self) -> int:
        return int(self.rate_mm_h.shape[0])

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        if self.latitude_deg.ndim == 2:
            return self.latitude_deg, self.longitude_deg
        lon2, lat2 = np.meshgrid(self.longitude_deg, self.latitude_deg)
        return lat2, lon2

    def mid_s(self) -> np.ndarray:
        return 0.5 * (self.start_s + self.end_s)

    def coverage_hours(self) -> float:
        """Hours of the 24 h cycle the interval midpoints span."""
        hours = np.sort(np.mod(self.mid_s() / 3600.0, HOURS_PER_DAY))
        if hours.size < 2:
            return 0.0
        gaps = np.diff(np.concatenate([hours, [hours[0] + HOURS_PER_DAY]]))
        return float(HOURS_PER_DAY - gaps.max())


@dataclass(frozen=True)
class HarmonicFit:
    """Per-cell first-harmonic fit; NaN where a cell was not fitted."""

    mean_mm_h: np.ndarray
    a1: np.ndarray
    b1: np.ndarray
    amplitude_mm_h: np.ndarray
    phase_h: np.ndarray
    sigma_amplitude_mm_h: np.ndarray
    resolvable: np.ndarray
    fitted: np.ndarray
    n_samples: int
    absolute_floor_mm_h: float
    z_floor: float


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------


def local_solar_hours(mid_s: np.ndarray, longitude_deg: np.ndarray) -> np.ndarray:
    """Local solar hour of each sample for each longitude: ``(n, ...)``."""
    utc_hours = np.mod(np.asarray(mid_s, dtype=np.float64) / 3600.0, HOURS_PER_DAY)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    return np.mod(
        utc_hours.reshape((-1,) + (1,) * lon.ndim) + lon[None] / 15.0,
        HOURS_PER_DAY,
    )


def fit_first_harmonic(
    series: RateSeries,
    *,
    fit_mask: np.ndarray | None = None,
    absolute_floor_mm_h: float = ABSOLUTE_FLOOR_MM_H,
    z_floor: float = Z_FLOOR,
) -> HarmonicFit:
    """Least-squares first diurnal harmonic per cell in local solar time."""
    n = series.n_samples
    if n < 4:
        raise ValueError(
            f"{n} samples cannot carry a mean plus one harmonic with a "
            "residual; at least 4 are needed"
        )
    coverage = series.coverage_hours()
    if coverage < MIN_COVERAGE_HOURS:
        raise ValueError(
            f"samples cover {coverage:.1f} h of the 24 h cycle; below "
            f"{MIN_COVERAGE_HOURS:.0f} h the first harmonic is not separable "
            "from the mean"
        )
    ny, nx = series.shape
    mask = np.ones((ny, nx), dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    if mask.shape != (ny, nx):
        raise ValueError("fit_mask must match the grid")
    if series.valid is not None:
        mask = mask & series.valid
    cells = np.flatnonzero(mask)
    lat2, lon2 = series.mesh()
    lon_cells = lon2.reshape(-1)[cells]
    rates = series.rate_mm_h.reshape(n, -1)[:, cells]  # (n, m)
    hours = local_solar_hours(series.mid_s(), lon_cells)  # (n, m)
    angle = OMEGA * hours
    design = np.stack(
        [np.ones_like(angle), np.cos(angle), np.sin(angle)], axis=-1
    )  # (n, m, 3)
    design = np.transpose(design, (1, 0, 2))  # (m, n, 3)
    gram = np.einsum("mni,mnk->mik", design, design)  # (m, 3, 3)
    rhs = np.einsum("mni,nm->mi", design, rates)  # (m, 3)
    coef = np.linalg.solve(gram, rhs[..., None])[..., 0]  # (m, 3)
    fitted_values = np.einsum("mni,mi->nm", design, coef)
    residual = rates - fitted_values
    ssr = np.sum(residual * residual, axis=0)
    sigma2 = ssr / float(n - 3)
    gram_inv = np.linalg.inv(gram)
    variance_factor = 0.5 * (gram_inv[:, 1, 1] + gram_inv[:, 2, 2])
    sigma_amp = np.sqrt(np.maximum(sigma2 * variance_factor, 0.0))
    amplitude = np.hypot(coef[:, 1], coef[:, 2])
    phase = np.mod(np.arctan2(coef[:, 2], coef[:, 1]) / OMEGA, HOURS_PER_DAY)
    resolvable = (amplitude > absolute_floor_mm_h) & (amplitude > z_floor * sigma_amp)

    def full(values, fill=np.nan, dtype=np.float64):
        out = np.full(ny * nx, fill, dtype=dtype)
        out[cells] = values
        return out.reshape(ny, nx)

    return HarmonicFit(
        mean_mm_h=full(coef[:, 0]),
        a1=full(coef[:, 1]),
        b1=full(coef[:, 2]),
        amplitude_mm_h=full(amplitude),
        phase_h=full(phase),
        sigma_amplitude_mm_h=full(sigma_amp),
        resolvable=full(resolvable, False, bool),
        fitted=mask,
        n_samples=n,
        absolute_floor_mm_h=float(absolute_floor_mm_h),
        z_floor=float(z_floor),
    )


# --------------------------------------------------------------------------
# regions
# --------------------------------------------------------------------------


def region_mask(
    name: str, lat2: np.ndarray, lon2: np.ndarray, land: np.ndarray
) -> np.ndarray:
    spec = REGIONS[name]
    mask = np.ones(lat2.shape, dtype=bool)
    if spec.get("land"):
        mask &= land
    if "latitude_deg" in spec:
        lo, hi = spec["latitude_deg"]
        mask &= (lat2 >= lo) & (lat2 <= hi)
    if "longitude_deg" in spec:
        lo, hi = spec["longitude_deg"]
        lon_signed = np.where(lon2 > 180.0, lon2 - 360.0, lon2)
        mask &= (lon_signed >= lo) & (lon_signed <= hi)
    return mask


def _circular_mean_hours(phase_h: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    """Weighted circular mean (h), circular std (h), resultant length."""
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan"), float("nan"), 0.0
    angle = OMEGA * np.asarray(phase_h, dtype=np.float64)
    c = float(np.sum(weights * np.cos(angle))) / total
    s = float(np.sum(weights * np.sin(angle))) / total
    length = math.hypot(c, s)
    mean_h = math.atan2(s, c) / OMEGA % HOURS_PER_DAY
    if length <= 0.0:
        spread = float("nan")
    else:
        spread = abs(math.sqrt(max(-2.0 * math.log(min(length, 1.0)), 0.0)) / OMEGA)
    return mean_h, spread, length


def summarize_region(
    fit: HarmonicFit,
    *,
    region: np.ndarray,
    lat2: np.ndarray,
) -> dict[str, object]:
    cells = region & fit.fitted & np.isfinite(fit.amplitude_mm_h)
    n_cells = int(np.count_nonzero(cells))
    if n_cells == 0:
        raise ValueError("region selects no fitted cells; check the land mask and grid")
    weight = np.cos(np.deg2rad(lat2[cells]))
    weight_total = float(np.sum(weight))
    amplitude = fit.amplitude_mm_h[cells]
    resolvable = fit.resolvable[cells]
    a1 = float(np.sum(weight * fit.a1[cells]) / weight_total)
    b1 = float(np.sum(weight * fit.b1[cells]) / weight_total)
    composite_amp = math.hypot(a1, b1)
    composite_phase = math.atan2(b1, a1) / OMEGA % HOURS_PER_DAY
    sigma = fit.sigma_amplitude_mm_h[cells]
    composite_sigma = math.sqrt(float(np.sum((weight * sigma) ** 2))) / weight_total
    composite_ok = (
        composite_amp > fit.absolute_floor_mm_h
        and composite_amp > fit.z_floor * composite_sigma
    )
    out: dict[str, object] = {
        "status": "measured" if composite_ok else "no-signal",
        "n_cells": n_cells,
        "n_resolvable": int(np.count_nonzero(resolvable)),
        "resolvable_fraction": float(np.sum(weight[resolvable]) / weight_total),
        "mean_rate_mm_h": float(np.sum(weight * fit.mean_mm_h[cells]) / weight_total),
        "amplitude_mean_mm_h": float(np.sum(weight * amplitude) / weight_total),
        "composite_amplitude_mm_h": composite_amp,
        "composite_phase_h": composite_phase,
        "composite_sigma_amplitude_mm_h": composite_sigma,
        "phase_h": None,
        "phase_spread_h": None,
        "amplitude_mean_resolvable_mm_h": None,
        "weighting": "cos(latitude); phase_h is additionally amplitude-weighted",
    }
    if out["n_resolvable"] > 0:
        w_res = weight[resolvable] * amplitude[resolvable]
        mean_h, spread_h, _ = _circular_mean_hours(fit.phase_h[cells][resolvable], w_res)
        out["phase_h"] = mean_h
        out["phase_spread_h"] = spread_h
        out["amplitude_mean_resolvable_mm_h"] = float(
            np.sum(weight[resolvable] * amplitude[resolvable]) / np.sum(weight[resolvable])
        )
    if not composite_ok:
        out["reason"] = (
            f"composite amplitude {composite_amp:.4f} mm/h is not above the "
            f"absolute floor {fit.absolute_floor_mm_h} mm/h and "
            f"{fit.z_floor} x its standard error {composite_sigma:.4f} mm/h"
        )
    return out


# --------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------


def aggregate_to_intervals(obs: RateSeries, target: RateSeries) -> RateSeries:
    """Sum finer observation intervals onto the target's intervals.

    Every target interval must be tiled exactly by observation intervals
    (no gaps, no partial overlap); anything else is refused, because a
    3 h model interval scored against 2 h of observation is a biased rate
    that nothing downstream could detect.
    """
    rates = []
    for start, end in zip(target.start_s, target.end_s):
        inside = (obs.start_s >= start - 1e-6) & (obs.end_s <= end + 1e-6)
        idx = np.flatnonzero(inside)
        covered = float(np.sum(obs.end_s[idx] - obs.start_s[idx]))
        if idx.size == 0 or abs(covered - (end - start)) > 1.0:
            raise ValueError(
                f"observation intervals cover {covered / 3600.0:.2f} h of the "
                f"target interval {_iso(start)}..{_iso(end)} "
                f"({(end - start) / 3600.0:.2f} h); the target must be tiled exactly"
            )
        order = idx[np.argsort(obs.start_s[idx])]
        starts, ends = obs.start_s[order], obs.end_s[order]
        if np.any(np.abs(starts[1:] - ends[:-1]) > 1.0):
            raise ValueError("observation intervals overlap or leave a gap inside a target interval")
        weights = (ends - starts) / (end - start)
        rates.append(np.tensordot(weights, obs.rate_mm_h[order], axes=(0, 0)))
    return RateSeries(
        start_s=target.start_s,
        end_s=target.end_s,
        rate_mm_h=np.stack(rates),
        latitude_deg=obs.latitude_deg,
        longitude_deg=obs.longitude_deg,
        valid=obs.valid,
    )


def bin_to_grid(
    obs: RateSeries,
    target: RateSeries,
    *,
    max_distance_m: float = DEFAULT_MAX_BIN_DISTANCE_M,
) -> tuple[RateSeries, dict[str, object]]:
    """Area-weighted binning of an observation series onto the target grid.

    Uses the tree's registered ``cell_average`` remap: each observation
    cell goes to the nearest target centre, and a target cell is the mean
    of its observation cells.  A target cell is valid only if it received
    at least one valid observation cell in EVERY interval.
    """
    from woof import obs_regrid_bridge
    from woof.verify.obs import regrid as obs_regrid

    src_lat, src_lon = obs.mesh()
    dst_lat, dst_lon = target.mesh()
    plan = obs_regrid.build_plan(
        source_latitude=src_lat,
        source_longitude=src_lon,
        destination_latitude=dst_lat,
        destination_longitude=dst_lon,
        method=obs_regrid.CELL_AVERAGE,
        max_distance_m=float(max_distance_m),
    )
    base_valid = np.ones(src_lat.shape, dtype=bool) if obs.valid is None else obs.valid
    rates = []
    valid_all = np.ones(dst_lat.shape, dtype=bool)
    for k in range(obs.n_samples):
        values, valid = obs_regrid.apply_plan(plan, obs.rate_mm_h[k], base_valid)
        rates.append(values)
        valid_all &= valid
    reason = obs_regrid_bridge.unavailable_reason()
    record = {
        "method": "cell_average (area-weighted binning: nearest target centre, mean of members)",
        "engine": "rust obs-regrid" if reason is None and not obs_regrid_bridge.python_fallback_requested() else f"python fallback ({reason or 'requested'})",
        **plan.record(),
        "target_cells_observed": int(np.count_nonzero(valid_all)),
    }
    binned = RateSeries(
        start_s=obs.start_s,
        end_s=obs.end_s,
        rate_mm_h=np.where(valid_all[None], np.stack(rates), 0.0),
        latitude_deg=target.latitude_deg,
        longitude_deg=target.longitude_deg,
        valid=valid_all,
    )
    return binned, record


def wrap_hours(delta_h: np.ndarray) -> np.ndarray:
    """Wrap a phase difference to (-12, 12] h."""
    return -np.mod(-np.asarray(delta_h, dtype=np.float64) + 12.0, HOURS_PER_DAY) + 12.0


def pair_phases(
    model_fit: HarmonicFit,
    obs_fit: HarmonicFit,
    *,
    region: np.ndarray,
    lat2: np.ndarray,
    min_paired: int = MIN_PAIRED_CELLS,
) -> dict[str, object]:
    """Paired phase difference model minus observation over a region."""
    both_fitted = region & model_fit.fitted & obs_fit.fitted
    both = both_fitted & model_fit.resolvable & obs_fit.resolvable
    n_overlap = int(np.count_nonzero(both_fitted))
    n_paired = int(np.count_nonzero(both))
    out: dict[str, object] = {
        "n_overlap_cells": n_overlap,
        "n_paired_cells": n_paired,
        "min_paired": int(min_paired),
        "sign": "model minus observation, hours, wrapped to (-12, 12]",
        "weighting": "cos(latitude)",
        "delta_phase_mean_h": None,
        "delta_phase_spread_h": None,
        "composite_delta_phase_h": None,
    }
    if n_overlap == 0:
        out["status"] = "unpaired"
        out["reason"] = "no cell is fitted on both the model and the observation side"
        return out
    weight_overlap = np.cos(np.deg2rad(lat2[both_fitted]))
    # Composite over the common coverage so both sides see the same cells.
    ma = float(np.sum(weight_overlap * model_fit.a1[both_fitted]))
    mb = float(np.sum(weight_overlap * model_fit.b1[both_fitted]))
    oa = float(np.sum(weight_overlap * obs_fit.a1[both_fitted]))
    ob = float(np.sum(weight_overlap * obs_fit.b1[both_fitted]))
    total = float(np.sum(weight_overlap))
    out["composite_model"] = {
        "amplitude_mm_h": math.hypot(ma, mb) / total,
        "phase_h": math.atan2(mb, ma) / OMEGA % HOURS_PER_DAY,
    }
    out["composite_observation"] = {
        "amplitude_mm_h": math.hypot(oa, ob) / total,
        "phase_h": math.atan2(ob, oa) / OMEGA % HOURS_PER_DAY,
    }
    out["composite_delta_phase_h"] = float(
        wrap_hours(out["composite_model"]["phase_h"] - out["composite_observation"]["phase_h"])
    )
    if n_paired < min_paired:
        out["status"] = "no-signal"
        out["reason"] = (
            f"{n_paired} cells resolvable on both sides, fewer than {min_paired}"
        )
        return out
    delta = wrap_hours(model_fit.phase_h[both] - obs_fit.phase_h[both])
    weight = np.cos(np.deg2rad(lat2[both]))
    mean_h, spread_h, _ = _circular_mean_hours(delta, weight)
    out["status"] = "measured"
    out["delta_phase_mean_h"] = float(wrap_hours(mean_h))
    out["delta_phase_spread_h"] = spread_h
    out["delta_phase_median_h"] = float(np.median(delta))
    return out


# --------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(text: str) -> float:
    value = str(text).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def gaussian_grid_coordinates(nlat: int, nlon: int) -> tuple[np.ndarray, np.ndarray]:
    """The model's own Gaussian grid for a checkpoint array shape."""
    from woof.globe.spectral.grid import GaussianGrid

    grid = GaussianGrid.for_shape(int(nlat), int(nlon))
    return np.asarray(grid.latitude_deg, dtype=np.float64), np.asarray(grid.longitude_deg, dtype=np.float64)


def run_start_utc(run_dir: Path) -> float | None:
    """The run's start time from its receipt, if the receipt carries one."""
    receipt = Path(run_dir) / "arwen-global-receipt.json"
    if not receipt.exists():
        return None
    with receipt.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    options = payload.get("config", {}).get("native_adapter_options", {})
    start = options.get("start_time_utc") if isinstance(options, dict) else None
    return parse_utc(start) if start else None


def read_checkpoint_series(
    paths: list[str | Path],
    *,
    start_utc_s: float,
) -> tuple[RateSeries, RateSeries, np.ndarray, dict[str, object]]:
    """Total and convective interval-mean rates from consecutive checkpoints.

    Returns ``(total, convective, land_mask, record)``.  Accumulator
    decrements between checkpoints (float32 rounding) are clipped at zero
    and counted in the record.
    """
    times: list[float] = []
    total: list[np.ndarray] = []
    convective: list[np.ndarray] = []
    land = None
    cold_starts: list[str] = []
    for path in sorted(Path(p) for p in paths):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["__metadata__"].item()))
            times.append(float(metadata["time_s"]))
            names = set(archive.files)
            if CHECKPOINT_CONVECTIVE_ACCUMULATOR not in names:
                # A cold-start checkpoint (step 0) carries no accumulator:
                # no physics has run and its buckets are zero by
                # construction.  The physics arrays it may carry are the
                # surface stores the cold start seeds (Noah's snow from
                # the analysis), never a bucket.  A checkpoint past step 0
                # without this accumulator, or a step-0 checkpoint that
                # carries any physics bucket but not this one, is a
                # different tree's state and is refused: reading it as
                # zero would print a convective phase for a run that never
                # booked convective rain.
                buckets = sorted(
                    name for name in names
                    if name.startswith("physics__") and not name.startswith("physics__noah_")
                )
                if int(metadata["step"]) != 0 or buckets:
                    raise ValueError(
                        f"{path}: no {CHECKPOINT_CONVECTIVE_ACCUMULATOR} array although the "
                        f"checkpoint is past the cold start or carries physics buckets {buckets[:6]}; "
                        "not a cumulus-bearing run"
                    )
                cold_starts.append(str(path))
            acc = None
            for name in CHECKPOINT_TOTAL_ACCUMULATORS:
                if name not in names and path.name in {Path(c).name for c in cold_starts}:
                    continue
                value = np.asarray(archive[name], dtype=np.float64)
                acc = value if acc is None else acc + value
            total.append(acc)
            if CHECKPOINT_CONVECTIVE_ACCUMULATOR in names:
                convective.append(np.asarray(archive[CHECKPOINT_CONVECTIVE_ACCUMULATOR], dtype=np.float64))
            else:
                convective.append(np.zeros_like(acc))
            if land is None:
                land = np.asarray(archive[CHECKPOINT_LAND_FRACTION], dtype=np.float64) >= 0.5
    if len(times) < 5:
        raise ValueError(f"{len(times)} checkpoints give {max(len(times) - 1, 0)} intervals; at least 4 are needed")
    order = np.argsort(times)
    times_arr = np.asarray(times, dtype=np.float64)[order]
    if np.any(np.diff(times_arr) <= 0.0):
        raise ValueError("checkpoint times are not strictly increasing")
    total_arr = np.stack([total[i] for i in order])
    conv_arr = np.stack([convective[i] for i in order])
    dt_h = np.diff(times_arr) / 3600.0
    total_diff = np.diff(total_arr, axis=0)
    conv_diff = np.diff(conv_arr, axis=0)
    negatives = int(np.count_nonzero(total_diff < 0.0)) + int(np.count_nonzero(conv_diff < 0.0))
    most_negative = float(min(total_diff.min(), conv_diff.min()))
    total_rate = np.maximum(total_diff, 0.0) / dt_h[:, None, None]
    conv_rate = np.maximum(conv_diff, 0.0) / dt_h[:, None, None]
    ny, nx = land.shape
    lat, lon = gaussian_grid_coordinates(ny, nx)
    start = start_utc_s + times_arr[:-1]
    end = start_utc_s + times_arr[1:]
    record = {
        "checkpoints": [str(Path(p)) for p in sorted(Path(p) for p in paths)],
        "n_intervals": int(dt_h.size),
        "interval_hours": [float(v) for v in dt_h],
        "window_utc": [_iso(start[0]), _iso(end[-1])],
        "cold_start_checkpoints_without_physics_state": cold_starts,
        "negative_increments_clipped": negatives,
        "most_negative_increment_kg_m2": most_negative,
        "grid": {"kind": "gaussian", "nlat": int(ny), "nlon": int(nx)},
        "land_mask": "surface__land_fraction >= 0.5",
        "total": " + ".join(CHECKPOINT_TOTAL_ACCUMULATORS),
        "convective": CHECKPOINT_CONVECTIVE_ACCUMULATOR,
    }
    return (
        RateSeries(start, end, total_rate, lat, lon),
        RateSeries(start, end, conv_rate, lat, lon),
        land,
        record,
    )


def read_stage4_series(pack_paths: list[str | Path], geo_pack: str | Path) -> RateSeries:
    """Hourly Stage-IV accumulation packs as a rate series on the HRAP grid.

    A pack's ``valid_time`` is the END of its accumulation window.
    """
    from woof.obs.obspack import read_geo_pack, read_grid_pack

    latitude, longitude = read_geo_pack(geo_pack)
    starts, ends, rates, valids = [], [], [], []
    for path in sorted(Path(p) for p in pack_paths):
        pack = read_grid_pack(path)
        if pack.meta.get("quantity") != "precipitation_accumulation" or pack.meta.get("units") != "mm":
            raise ValueError(f"{path}: not a precipitation accumulation in mm")
        hours = float(pack.meta["accumulation_hours"])
        end = parse_utc(str(pack.meta["valid_time"]))
        values = np.asarray(pack.array("values"), dtype=np.float64)
        valid = np.asarray(pack.array("valid"), dtype=bool)
        if values.shape != latitude.shape:
            raise ValueError(f"{path}: values {values.shape} do not match the geometry {latitude.shape}")
        starts.append(end - hours * 3600.0)
        ends.append(end)
        rates.append(np.where(valid, values, 0.0) / hours)
        valids.append(valid)
    if not rates:
        raise ValueError("no observation packs")
    return RateSeries(
        start_s=np.asarray(starts),
        end_s=np.asarray(ends),
        rate_mm_h=np.stack(rates),
        latitude_deg=latitude,
        longitude_deg=longitude,
        valid=np.logical_and.reduce(np.stack(valids)),
    )


# --------------------------------------------------------------------------
# the reading
# --------------------------------------------------------------------------


def measure(
    *,
    total: RateSeries,
    convective: RateSeries | None,
    land: np.ndarray,
    observation: RateSeries | None = None,
    absolute_floor_mm_h: float = ABSOLUTE_FLOOR_MM_H,
    z_floor: float = Z_FLOOR,
    max_bin_distance_m: float = DEFAULT_MAX_BIN_DISTANCE_M,
    min_paired: int = MIN_PAIRED_CELLS,
    regions: tuple[str, ...] = tuple(REGIONS),
) -> dict[str, object]:
    """The instrument: regional readings per species and the paired reading."""
    lat2, lon2 = total.mesh()
    land = np.asarray(land, dtype=bool)
    masks = {name: region_mask(name, lat2, lon2, land) for name in regions}
    fits: dict[str, HarmonicFit] = {
        "total": fit_first_harmonic(
            total, fit_mask=land, absolute_floor_mm_h=absolute_floor_mm_h, z_floor=z_floor
        )
    }
    if convective is not None:
        fits["convective"] = fit_first_harmonic(
            convective, fit_mask=land, absolute_floor_mm_h=absolute_floor_mm_h, z_floor=z_floor
        )
    result: dict[str, object] = {
        "schema": SCHEMA,
        "measures": (
            "first diurnal harmonic of the interval-mean precipitation rate per "
            "land cell in local solar time (UTC + longitude/15, equation of time "
            "not applied); phase is the local hour of the harmonic's maximum"
        ),
        "sampling": {
            "n_intervals": total.n_samples,
            "interval_hours": [float(v) for v in (total.end_s - total.start_s) / 3600.0],
            "window_utc": [_iso(total.start_s[0]), _iso(total.end_s[-1])],
            "coverage_hours_of_cycle": total.coverage_hours(),
        },
        "noise_floor": {
            "absolute_mm_h": float(absolute_floor_mm_h),
            "z": float(z_floor),
            "rule": "resolvable when amplitude > absolute AND amplitude > z * sigma_amplitude",
        },
        "regions": {},
        "observation": {"status": "unpaired", "reason": "no observation series supplied"},
    }
    for name, mask in masks.items():
        result["regions"][name] = {
            species: summarize_region(fit, region=mask, lat2=lat2)
            for species, fit in fits.items()
        }
    if observation is not None:
        aggregated = aggregate_to_intervals(observation, total)
        binned, record = bin_to_grid(aggregated, total, max_distance_m=max_bin_distance_m)
        obs_fit = fit_first_harmonic(
            binned, fit_mask=land, absolute_floor_mm_h=absolute_floor_mm_h, z_floor=z_floor
        )
        obs_block: dict[str, object] = {
            "status": "measured",
            "binning": record,
            "regions": {},
        }
        any_measured = False
        for name, mask in masks.items():
            observed = mask & binned.valid
            block: dict[str, object] = {"n_observed_cells": int(np.count_nonzero(observed))}
            if block["n_observed_cells"] == 0:
                block["status"] = "unpaired"
                block["reason"] = "no observed cell in this region"
                block["observation_fit"] = None
            else:
                block["observation_fit"] = summarize_region(obs_fit, region=observed, lat2=lat2)
                block["status"] = "measured"
            for species, fit in fits.items():
                if block["n_observed_cells"] == 0:
                    block[species] = {"status": "unpaired", "reason": "no observed cell in this region"}
                    continue
                paired = pair_phases(fit, obs_fit, region=mask, lat2=lat2, min_paired=min_paired)
                block[species] = paired
                any_measured |= paired["status"] == "measured"
            obs_block["regions"][name] = block
        obs_block["status"] = "measured" if any_measured else "unpaired"
        if not any_measured:
            obs_block["reason"] = "no region reached the paired-cell minimum"
        result["observation"] = obs_block
    result["summary"] = summary_line(result)
    return result


def _fmt_phase(value) -> str:
    return "n/a" if value is None else f"{value:05.2f}"


def summary_line(result: dict[str, object], label: str = "") -> str:
    parts = []
    regions = result["regions"]
    for name in regions:
        for species, block in regions[name].items():
            parts.append(
                f"{name}/{species}: phase {_fmt_phase(block['composite_phase_h'])} LST "
                f"(cells {_fmt_phase(block['phase_h'])}) amp {block['composite_amplitude_mm_h']:.3f} mm/h "
                f"resolvable {block['resolvable_fraction']:.2f} [{block['status']}]"
            )
    obs = result["observation"]
    if obs["status"] == "unpaired" and "regions" not in obs:
        parts.append(f"obs: [unpaired] {obs['reason']}")
    else:
        for name, block in obs["regions"].items():
            for species in ("total", "convective"):
                if species not in block:
                    continue
                pair = block[species]
                if pair["status"] == "measured":
                    parts.append(
                        f"{name}/{species} vs obs: d(model-obs) {pair['delta_phase_mean_h']:+.2f} h "
                        f"spread {pair['delta_phase_spread_h']:.2f} h composite {pair['composite_delta_phase_h']:+.2f} h "
                        f"n={pair['n_paired_cells']} [measured]"
                    )
                else:
                    parts.append(f"{name}/{species} vs obs: [{pair['status']}] {pair.get('reason', '')}")
    head = f"diurnal-phase {label}: " if label else "diurnal-phase: "
    return head + "; ".join(parts)


# --------------------------------------------------------------------------
# synthetic families for calibration
# --------------------------------------------------------------------------


def synthetic_grid(ny: int = 12, nx: int = 24) -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(-60.0, 60.0, ny)
    lon = (np.arange(nx) + 0.5) * (360.0 / nx) - 180.0
    return lat, lon


def synthetic_series(
    *,
    phase_h: float,
    amplitude_mm_h: float,
    mean_mm_h: float = 0.5,
    noise_mm_h: float = 0.0,
    second_harmonic_fraction: float = 0.0,
    interval_hours: float = 1.0,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    start_utc_s: float = 0.0,
    seed: int = 0,
) -> RateSeries:
    """A field whose diurnal cycle in LOCAL time peaks at ``phase_h``.

    The second harmonic, when present, is phase-locked so the true maximum
    stays at ``phase_h``.  Rates are clipped at zero only when the noise
    would make them negative in a way the fit does not see (the clip is
    skipped, because a clipped noise field is no longer the family under
    test).
    """
    if lat is None or lon is None:
        lat, lon = synthetic_grid()
    n = int(round(HOURS_PER_DAY / interval_hours))
    start = start_utc_s + np.arange(n) * interval_hours * 3600.0
    end = start + interval_hours * 3600.0
    mid = 0.5 * (start + end)
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    hours = local_solar_hours(mid, lon2)  # (n, ny, nx)
    cycle = amplitude_mm_h * np.cos(OMEGA * (hours - phase_h))
    if second_harmonic_fraction:
        cycle = cycle + amplitude_mm_h * second_harmonic_fraction * np.cos(
            2.0 * OMEGA * (hours - phase_h)
        )
    rate = mean_mm_h + cycle
    if noise_mm_h:
        rate = rate + np.random.default_rng(seed).normal(0.0, noise_mm_h, size=rate.shape)
    return RateSeries(start, end, rate, lat, lon)


def calibrate() -> dict[str, object]:
    """Run the calibration families and return their numbers."""
    lat, lon = synthetic_grid()
    land = np.ones((lat.size, lon.size), dtype=bool)
    rows = []
    truth = 16.5
    for amplitude, noise, interval in (
        (1.0, 0.0, 1.0), (1.0, 0.05, 1.0), (1.0, 0.2, 1.0), (1.0, 0.2, 3.0),
        (0.3, 0.2, 1.0), (0.05, 0.2, 1.0), (0.005, 0.0, 1.0), (0.0, 0.2, 1.0),
    ):
        series = synthetic_series(
            phase_h=truth, amplitude_mm_h=amplitude, noise_mm_h=noise,
            interval_hours=interval, seed=7,
        )
        fit = fit_first_harmonic(series, fit_mask=land)
        region = summarize_region(fit, region=land, lat2=series.mesh()[0])
        err = abs(float(wrap_hours(region["composite_phase_h"] - truth))) * 60.0
        cell_err = np.abs(wrap_hours(fit.phase_h[fit.resolvable] - truth)) * 60.0
        rows.append({
            "family": "A", "amplitude_mm_h": amplitude, "noise_mm_h": noise,
            "samples": series.n_samples, "composite_phase_error_min": err,
            "cell_median_error_min": float(np.median(cell_err)) if cell_err.size else None,
            "resolvable_fraction": region["resolvable_fraction"], "status": region["status"],
        })
    truth_b = 17.0
    for interval in (1.0, 3.0):
        series = synthetic_series(
            phase_h=truth_b, amplitude_mm_h=0.5, second_harmonic_fraction=0.4,
            interval_hours=interval,
        )
        fit = fit_first_harmonic(series, fit_mask=land)
        region = summarize_region(fit, region=land, lat2=series.mesh()[0])
        rows.append({
            "family": "B", "samples": series.n_samples,
            "composite_phase_error_min": abs(float(wrap_hours(region["composite_phase_h"] - truth_b))) * 60.0,
            "resolvable_fraction": region["resolvable_fraction"], "status": region["status"],
        })
    for noise in (0.0, 0.2):
        series = synthetic_series(phase_h=0.0, amplitude_mm_h=0.0, mean_mm_h=0.3, noise_mm_h=noise, seed=3)
        fit = fit_first_harmonic(series, fit_mask=land)
        region = summarize_region(fit, region=land, lat2=series.mesh()[0])
        rows.append({"family": "flat", "noise_mm_h": noise, "status": region["status"],
                     "resolvable_fraction": region["resolvable_fraction"]})
    obs_lat = np.arange(-59.75, 60.0, 0.5)
    obs_lon = np.arange(-179.75, 180.0, 0.5)
    for shift, second, noise in ((3.0, 0.0, 0.0), (-3.0, 0.0, 0.0), (0.0, 0.0, 0.0), (3.0, 0.4, 0.05)):
        model = synthetic_series(phase_h=15.0 + shift, amplitude_mm_h=0.5, second_harmonic_fraction=second, noise_mm_h=noise, seed=11)
        obs = synthetic_series(phase_h=15.0, amplitude_mm_h=0.5, second_harmonic_fraction=second, noise_mm_h=noise, lat=obs_lat, lon=obs_lon, seed=12)
        reading = measure(total=model, convective=None, land=land, observation=obs, regions=("global_land",))
        pair = reading["observation"]["regions"]["global_land"]["total"]
        rows.append({"family": "shift", "imposed_h": shift, "second_harmonic": second, "noise_mm_h": noise,
                     "read_h": pair["delta_phase_mean_h"], "spread_h": pair["delta_phase_spread_h"],
                     "composite_h": pair["composite_delta_phase_h"], "n_paired": pair["n_paired_cells"],
                     "status": pair["status"]})
    return {"schema": SCHEMA + "/calibration", "rows": rows}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="diurnal-phase instrument for surface precipitation")
    parser.add_argument("--run-dir", help="directory of arwen_global_step*.npz checkpoints")
    parser.add_argument("--checkpoints", nargs="*", help="explicit checkpoint paths (else every step file in --run-dir)")
    parser.add_argument("--start", help="run start, ISO UTC; else the run receipt's start_time_utc")
    parser.add_argument("--obs-packs", help="glob of Stage-IV hourly packs (gpuwm-obs.obs-grid.v1)")
    parser.add_argument("--obs-geo", help="the packs' geometry pack (gpuwm-obs.obs-geo.v1)")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", help="JSON output path")
    parser.add_argument("--absolute-floor-mm-h", type=float, default=ABSOLUTE_FLOOR_MM_H)
    parser.add_argument("--z-floor", type=float, default=Z_FLOOR)
    parser.add_argument("--max-bin-distance-m", type=float, default=DEFAULT_MAX_BIN_DISTANCE_M)
    parser.add_argument("--calibrate", action="store_true", help="run the synthetic families and print their numbers")
    args = parser.parse_args(argv)

    if args.calibrate:
        payload = calibrate()
        for row in payload["rows"]:
            print(json.dumps(_finite(row), sort_keys=True))
        if args.out:
            _write_json(Path(args.out), _finite(payload))
        return 0

    if not args.run_dir and not args.checkpoints:
        parser.error("--run-dir or --checkpoints is required")
    paths = list(args.checkpoints or [])
    if args.run_dir:
        paths.extend(sorted(glob.glob(str(Path(args.run_dir) / "arwen_global_step*.npz"))))
    if not paths:
        parser.error("no checkpoints found")
    if args.start:
        start_s = parse_utc(args.start)
    else:
        start_s = run_start_utc(Path(args.run_dir)) if args.run_dir else None
        if start_s is None:
            parser.error("--start is required when the run directory carries no receipt with start_time_utc")
    total, convective, land, record = read_checkpoint_series(paths, start_utc_s=start_s)
    observation = None
    obs_record: dict[str, object] = {}
    if args.obs_packs:
        if not args.obs_geo:
            parser.error("--obs-geo is required with --obs-packs")
        packs = sorted(glob.glob(args.obs_packs))
        if not packs:
            parser.error(f"no packs match {args.obs_packs}")
        observation = read_stage4_series(packs, args.obs_geo)
        obs_record = {"packs": packs, "geometry": args.obs_geo, "valid_time_is": "end of accumulation"}
    reading = measure(
        total=total, convective=convective, land=land, observation=observation,
        absolute_floor_mm_h=args.absolute_floor_mm_h, z_floor=args.z_floor,
        max_bin_distance_m=args.max_bin_distance_m,
    )
    reading["label"] = args.label
    reading["model_source"] = record
    reading["observation_source"] = obs_record
    reading["summary"] = summary_line(reading, args.label)
    reading = _finite(reading)
    if args.out:
        _write_json(Path(args.out), reading)
    print(reading["summary"])
    return 0


__all__ = [
    "ABSOLUTE_FLOOR_MM_H", "HarmonicFit", "RateSeries", "REGIONS", "SCHEMA", "Z_FLOOR",
    "aggregate_to_intervals", "bin_to_grid", "calibrate", "fit_first_harmonic",
    "local_solar_hours", "measure", "pair_phases", "read_checkpoint_series",
    "read_stage4_series", "region_mask", "summarize_region", "summary_line",
    "synthetic_series", "wrap_hours",
]


if __name__ == "__main__":
    sys.exit(main())
