"""The grid-point-storm reader: the heaviest cells of a run and what feeds them.

What it measures.  From the surface accumulators of every checkpoint of a
run (the grid-scale rain, snow and graupel buckets and the cumulus
scheme's ``physics__rainc``), per checkpoint interval and cell, the
grid-scale, convective and total rain rates (mm/h), and from them

* per interval: the largest cell rate (grid-scale and total) and where it
  fell, and the number and area fraction of cells raining above each rate
  threshold,
* over the run: every cell's accumulation, the number and area fraction of
  cells above each accumulation threshold and the share of the global
  mean those cells carry (a grid-point storm is a cell whose 24 h total is
  hundreds of millimetres while its neighbours are dry), the largest
  cell, and the ``top_n`` cells by grid-scale accumulation,
* for the top cells, the column state at every checkpoint
  (:mod:`woof.globe.column_sounding`): the strongest ascent in the
  column (the most negative interface pressure velocity, Pa/s, diagnosed
  from the checkpointed mass flux) and the pressure it sits at, the
  condensate path by species, precipitable water, surface-based and
  most-unstable CAPE and CIN, the highest relative humidity, the number of
  saturated levels and the cloud top, beside the interval's grid-scale and
  convective rain and whether the cumulus scheme booked rain there,
* with a GPU and the run's configuration (``--tendencies``), the
  cumulus and microphysics tendencies on those cells: the model is
  rebuilt from the run's TOML, restarted from a chosen checkpoint and
  stepped, and after the cumulus step and after the microphysics step of
  every physics call the column heating (K/h), the vapor and condensate
  changes (kg/m2 per hour) and the rain each scheme put on the ground are
  read on the target columns, the same wrapping the precipitation trace
  tool uses.

Every rate is an interval mean; every count is a cell count on the run's
Gaussian grid and every fraction is area-weighted by the quadrature
weights (what the model's own integrals measure).  Reads hash-verified
checkpoints through ``read_checkpoint``; the run's receipt (or, for a run
still writing, its TOML) supplies the grid and the vertical coordinate.
Writes one JSON.
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
from .diurnal_phase import _iso

SCHEMA = "gpuwm.arwen-global-storm-reader/v1"
RATE_THRESHOLDS_MM_H = (5.0, 10.0, 20.0, 50.0)
TOTAL_THRESHOLDS_MM = (50.0, 100.0, 200.0, 300.0)
DEFAULT_TOP_CELLS = 6
TENDENCY_STAGES = ("_cumulus_step", "_microphysics_step")
#: A column raining above this rate (mm/h) from either scheme in one physics
#: call is counted as heavy in the tendency trace's exit census.
HEAVY_RATE_MM_H = 20.0


@dataclass(frozen=True)
class IntervalRain:
    """One checkpoint interval's rain increments, ``(ny, nx)`` in kg/m2."""

    start_utc_s: float
    end_utc_s: float
    grid_kg_m2: np.ndarray
    convective_kg_m2: np.ndarray

    @property
    def hours(self) -> float:
        return (self.end_utc_s - self.start_utc_s) / 3600.0


def _cell(j: int, i: int, lat: np.ndarray, lon: np.ndarray) -> dict[str, float | int]:
    return {"j": int(j), "i": int(i), "latitude_deg": float(lat[j]), "longitude_deg": float(lon[i])}


def storm_statistics(
    intervals: list[IntervalRain],
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    cell_weights: np.ndarray,
    *,
    rate_thresholds=RATE_THRESHOLDS_MM_H,
    total_thresholds=TOTAL_THRESHOLDS_MM,
    top_n: int = DEFAULT_TOP_CELLS,
) -> dict[str, object]:
    """The rate and accumulation statistics of the module docstring."""
    if not intervals:
        raise ValueError("at least one checkpoint interval is needed")
    intervals = sorted(intervals, key=lambda s: s.start_utc_s)
    ny, nx = int(latitude_deg.size), int(longitude_deg.size)
    w = np.asarray(cell_weights, dtype=np.float64)
    if w.shape != (ny, nx):
        raise ValueError("cell weights must be on the (ny, nx) grid of the coordinates")
    total_area = float(np.sum(w))
    grid_total = np.zeros((ny, nx))
    conv_total = np.zeros((ny, nx))
    peak_grid_rate = np.zeros((ny, nx))
    peak_total_rate = np.zeros((ny, nx))
    peak_grid_interval = np.full((ny, nx), -1, dtype=np.int64)
    per_interval = []
    for index, s in enumerate(intervals):
        if s.grid_kg_m2.shape != (ny, nx) or s.convective_kg_m2.shape != (ny, nx):
            raise ValueError("every interval must be on the (ny, nx) grid of the coordinates")
        if s.end_utc_s <= s.start_utc_s:
            raise ValueError("interval end must follow its start")
        g = np.maximum(np.asarray(s.grid_kg_m2, dtype=np.float64), 0.0)
        c = np.maximum(np.asarray(s.convective_kg_m2, dtype=np.float64), 0.0)
        grid_rate = g / s.hours
        total_rate = (g + c) / s.hours
        grid_total += g
        conv_total += c
        newly = grid_rate > peak_grid_rate
        peak_grid_interval = np.where(newly, index, peak_grid_interval)
        peak_grid_rate = np.maximum(peak_grid_rate, grid_rate)
        peak_total_rate = np.maximum(peak_total_rate, total_rate)
        jg, ig = np.unravel_index(int(np.argmax(grid_rate)), grid_rate.shape)
        jt, it = np.unravel_index(int(np.argmax(total_rate)), total_rate.shape)
        per_interval.append({
            "start_utc": _iso(s.start_utc_s), "end_utc": _iso(s.end_utc_s), "hours": float(s.hours),
            "max_grid_scale_rate_mm_h": float(grid_rate[jg, ig]),
            "max_grid_scale_cell": _cell(jg, ig, latitude_deg, longitude_deg),
            "max_total_rate_mm_h": float(total_rate[jt, it]),
            "max_total_cell": _cell(jt, it, latitude_deg, longitude_deg),
            "convective_share_at_max_total": float(c[jt, it] / (g[jt, it] + c[jt, it])) if (g[jt, it] + c[jt, it]) > 0 else None,
            "global_mean_grid_scale_mm": float(np.sum(w * g) / total_area),
            "global_mean_convective_mm": float(np.sum(w * c) / total_area),
            "cells_above_rate": {
                f"{t:g}": {
                    "grid_scale_count": int(np.count_nonzero(grid_rate >= t)),
                    "grid_scale_area_fraction": float(np.sum(w[grid_rate >= t]) / total_area),
                    "total_count": int(np.count_nonzero(total_rate >= t)),
                    "total_area_fraction": float(np.sum(w[total_rate >= t]) / total_area),
                }
                for t in rate_thresholds
            },
        })
    all_total = grid_total + conv_total
    mean_grid = float(np.sum(w * grid_total) / total_area)
    mean_conv = float(np.sum(w * conv_total) / total_area)
    mean_total = mean_grid + mean_conv
    above = {}
    for t in total_thresholds:
        sel_g = grid_total >= t
        sel_t = all_total >= t
        above[f"{t:g}"] = {
            "grid_scale_count": int(np.count_nonzero(sel_g)),
            "grid_scale_area_fraction": float(np.sum(w[sel_g]) / total_area),
            "grid_scale_share_of_global_mean": float(np.sum((w * grid_total)[sel_g]) / total_area / mean_grid) if mean_grid > 0 else None,
            "total_count": int(np.count_nonzero(sel_t)),
            "total_area_fraction": float(np.sum(w[sel_t]) / total_area),
            "total_share_of_global_mean": float(np.sum((w * all_total)[sel_t]) / total_area / mean_total) if mean_total > 0 else None,
        }
    order = np.argsort(grid_total, axis=None)[::-1][: max(int(top_n), 0)]
    top = []
    for flat in order:
        j, i = np.unravel_index(int(flat), grid_total.shape)
        if grid_total[j, i] <= 0.0:
            break
        top.append({
            **_cell(j, i, latitude_deg, longitude_deg),
            "grid_scale_mm": float(grid_total[j, i]), "convective_mm": float(conv_total[j, i]),
            "total_mm": float(all_total[j, i]),
            "peak_grid_scale_rate_mm_h": float(peak_grid_rate[j, i]),
            "peak_grid_scale_interval": int(peak_grid_interval[j, i]),
            "peak_total_rate_mm_h": float(peak_total_rate[j, i]),
        })
    jm, im = np.unravel_index(int(np.argmax(all_total)), all_total.shape)
    jg, ig = np.unravel_index(int(np.argmax(grid_total)), grid_total.shape)
    return {
        "window_utc": [_iso(intervals[0].start_utc_s), _iso(intervals[-1].end_utc_s)],
        "n_intervals": len(intervals),
        "interval_hours": [float(s.hours) for s in intervals],
        "global_mean_mm": {"grid_scale": mean_grid, "convective": mean_conv, "total": mean_total},
        "max_cell": {
            "grid_scale_mm": float(grid_total[jg, ig]), "grid_scale_cell": _cell(jg, ig, latitude_deg, longitude_deg),
            "total_mm": float(all_total[jm, im]), "total_cell": _cell(jm, im, latitude_deg, longitude_deg),
            "convective_share_at_max_total": float(conv_total[jm, im] / all_total[jm, im]) if all_total[jm, im] > 0 else None,
        },
        "cells_above_total": above,
        "max_rate_over_run_mm_h": {
            "grid_scale": float(np.max(peak_grid_rate)), "total": float(np.max(peak_total_rate)),
        },
        "top_cells_by_grid_scale": top,
        "per_interval": per_interval,
    }


# --------------------------------------------------------------------------
# column state of the top cells through the run
# --------------------------------------------------------------------------


def _omega_full(omega_half: np.ndarray) -> np.ndarray:
    return 0.5 * (omega_half[:-1] + omega_half[1:])


def column_readings(sounding: cs.Sounding, cells: list[tuple[int, int]]) -> list[dict[str, object]]:
    """The state of ``cells`` (``(j, i)`` pairs) in one sounding."""
    js = np.asarray([c[0] for c in cells], dtype=np.int64)
    is_ = np.asarray([c[1] for c in cells], dtype=np.int64)
    t = sounding.temperature_k[:, js, is_]
    q = sounding.qv[:, js, is_]
    pf = sounding.p_full[:, js, is_]
    ph = sounding.p_half[:, js, is_]
    cape = cs.parcel_cape(t, q, pf, ph)
    rh = sounding.relative_humidity[:, js, is_]
    dp_g = (ph[1:] - ph[:-1]) / cs.GRAVITY_M_S2
    rows = []
    for n, (j, i) in enumerate(cells):
        row: dict[str, object] = {
            "j": int(j), "i": int(i),
            "time_s": float(sounding.time_s), "step": int(sounding.step),
            "precipitable_water_kg_m2": float(np.sum(q[:, n] * dp_g[:, n])),
            "condensate_kg_m2": {
                name: float(np.sum(field[:, j, i] * dp_g[:, n]))
                for name, field in sounding.condensate.items()
            },
            "cape_sb_j_kg": float(cape["cape_sb_j_kg"][n]), "cin_sb_j_kg": float(cape["cin_sb_j_kg"][n]),
            "cape_mu_j_kg": float(cape["cape_mu_j_kg"][n]), "cin_mu_j_kg": float(cape["cin_mu_j_kg"][n]),
            "rh_max": float(np.max(rh[:, n])),
            "saturated_levels": int(np.count_nonzero(rh[:, n] >= 0.99)),
            "rh_profile": [float(v) for v in rh[:, n]],
            "temperature_profile_k": [float(v) for v in t[:, n]],
            "p_full_pa": [float(v) for v in pf[:, n]],
            "surface_pressure_pa": float(ph[-1, n]),
            "lowest_level_temperature_k": float(t[-1, n]),
        }
        cloudy = sounding.cloudy()[:, j, i]
        row["cloud_top_pa"] = float(pf[int(np.argmax(cloudy)), n]) if np.any(cloudy) else None
        if sounding.omega_half_pa_s is not None:
            omega = _omega_full(sounding.omega_half_pa_s[:, j, i])
            k = int(np.argmin(omega))
            row["omega_min_pa_s"] = float(omega[k])
            row["omega_min_pressure_pa"] = float(pf[k, n])
            row["omega_profile_pa_s"] = [float(v) for v in omega]
            k500 = int(np.argmin(np.abs(pf[:, n] - 50000.0)))
            row["omega_500_pa_s"] = float(omega[k500])
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# run door
# --------------------------------------------------------------------------


def _accumulators(path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Metadata, grid-scale and convective accumulators and the land mask
    of one checkpoint (hash-verified), without a synthesis."""
    from .checkpoint import read_checkpoint

    metadata, arrays = read_checkpoint(path)
    names = set(arrays)
    grid = sum(np.asarray(arrays[n], dtype=np.float64) for n in cs.CHECKPOINT_GRID_SCALE_BUCKETS)
    if cs.CHECKPOINT_CONVECTIVE in names:
        conv = np.asarray(arrays[cs.CHECKPOINT_CONVECTIVE], dtype=np.float64)
    elif cs.convective_accumulator_is_absent_by_construction(names, int(metadata["step"])):
        # The cold start: no physics call has run, and the physics arrays it
        # may carry are the seeded surface stores, not buckets.
        conv = np.zeros_like(grid)
    else:
        raise ValueError(cs.convective_accumulator_refusal(path, names, int(metadata["step"])))
    land = np.asarray(arrays[cs.CHECKPOINT_LAND_FRACTION], dtype=np.float64) >= 0.5
    return metadata, grid, conv, land


def measure_run(
    run_dir: str | Path,
    *,
    start_utc_s: float | None = None,
    checkpoints: list[str | Path] | None = None,
    config: str | Path | None = None,
    top_n: int = DEFAULT_TOP_CELLS,
    extra_cells: list[tuple[int, int]] | None = None,
    state: bool = True,
    rate_thresholds=RATE_THRESHOLDS_MM_H,
    total_thresholds=TOTAL_THRESHOLDS_MM,
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
    reader = cs.SoundingReader(receipt, omega=True)
    lat, lon = reader.latitude_deg, reader.longitude_deg
    intervals: list[IntervalRain] = []
    previous = None
    land = None
    times = []
    for path in paths:
        metadata, grid, conv, land = _accumulators(path)
        if progress is not None:
            progress(f"accumulators {path.name} step {metadata['step']}")
        t = float(metadata["time_s"])
        if previous is not None:
            if t <= previous[0]:
                raise ValueError("checkpoint times are not strictly increasing")
            intervals.append(IntervalRain(
                start_utc_s + previous[0], start_utc_s + t,
                grid - previous[1], conv - previous[2],
            ))
        previous = (t, grid, conv)
        times.append(t)
    stats = storm_statistics(
        intervals, lat, lon, reader.cell_weights,
        rate_thresholds=rate_thresholds, total_thresholds=total_thresholds, top_n=top_n,
    )
    cells = [(int(c["j"]), int(c["i"])) for c in stats["top_cells_by_grid_scale"]]
    for extra in extra_cells or ():
        if tuple(extra) not in cells:
            cells.append((int(extra[0]), int(extra[1])))
    traces: dict[str, list[dict]] = {f"{j},{i}": [] for j, i in cells}
    if state and cells:
        for index, path in enumerate(paths):
            sounding = reader.sounding(path)
            if progress is not None:
                progress(f"sounding {path.name} step {sounding.step}")
            for row in column_readings(sounding, cells):
                key = f"{row['j']},{row['i']}"
                if index > 0:
                    s = intervals[index - 1]
                    row["interval_grid_scale_mm"] = float(s.grid_kg_m2[row["j"], row["i"]])
                    row["interval_convective_mm"] = float(s.convective_kg_m2[row["j"], row["i"]])
                    row["cumulus_active"] = bool(s.convective_kg_m2[row["j"], row["i"]] > 0.0)
                row["land"] = bool(land[row["j"], row["i"]])
                row["utc"] = _iso(start_utc_s + row["time_s"])
                traces[key].append(row)
    return {
        "schema": SCHEMA,
        "measures": (
            "per checkpoint interval and cell: grid-scale rain (surface rain + snow + graupel accumulators) and "
            "convective rain (physics__rainc) as interval-mean rates; accumulation thresholds over the run; "
            "the top cells' column state per checkpoint (omega from the checkpointed mass-flux continuity, "
            "condensate paths, parcel CAPE/CIN, relative humidity over liquid water); counts are cells, "
            "fractions are area-weighted by the Gaussian quadrature weights"
        ),
        "column_state": {
            "saturation": cs.SATURATION_FORMULA,
            "omega": "interface pressure velocity from div(v dp) of the checkpointed vorticity, divergence and ln ps, averaged to full levels; negative is ascent",
            "cape": "surface-based and most-unstable parcel, pseudo-adiabatic, virtual-temperature buoyancy (column_sounding)",
        } if state else "not read",
        "rate_thresholds_mm_h": [float(t) for t in rate_thresholds],
        "total_thresholds_mm": [float(t) for t in total_thresholds],
        "statistics": stats,
        "cells": cells,
        "column_traces": traces,
        "model_source": {
            "run_dir": str(run_dir), "checkpoints": [str(p) for p in paths],
            "config_hash": reader.config_hash,
            "grid": {"kind": "gaussian", "nlat": reader.shape[0], "nlon": reader.shape[1]},
            "land_mask": f"{cs.CHECKPOINT_LAND_FRACTION} >= 0.5",
        },
    }


def summary_line(result: dict, label: str = "") -> str:
    st = result["statistics"]
    parts = [f"storms {label}".strip()]
    parts.append(
        f"global mean mm grid {st['global_mean_mm']['grid_scale']:.3f} conv {st['global_mean_mm']['convective']:.3f}; "
        f"max cell grid {st['max_cell']['grid_scale_mm']:.1f} mm at "
        f"{st['max_cell']['grid_scale_cell']['latitude_deg']:.1f}N {st['max_cell']['grid_scale_cell']['longitude_deg']:.1f}E; "
        f"max rate grid {st['max_rate_over_run_mm_h']['grid_scale']:.1f} mm/h"
    )
    for t, row in st["cells_above_total"].items():
        share = row["grid_scale_share_of_global_mean"]
        parts.append(
            f">{t} mm: grid {row['grid_scale_count']} cells ({'n/a' if share is None else f'{100 * share:.2f}%'} of mean), "
            f"total {row['total_count']}"
        )
    return " | ".join(parts)


# --------------------------------------------------------------------------
# tendencies on named columns: restart, step, wrap the two schemes
# --------------------------------------------------------------------------


def trace_tendencies(
    config: str | Path, checkpoint: str | Path, cells: list[tuple[int, int]], *, steps: int = 12,
    progress=None,
) -> dict[str, object]:
    """Restart the model of ``config`` from ``checkpoint`` and record, on
    ``cells``, what the cumulus step and the microphysics step do in every
    physics call of ``steps`` steps.  Needs the run's backend (cupy)."""
    import functools

    from woof.globe.checkpoint import read_checkpoint, state_from_checkpoint
    from woof.globe.config import load_config
    from woof.globe.constants import CONDENSATE_SPECIES, GRAVITY_M_S2
    from woof.globe.physics import native_runtime as nr
    from woof.globe.runner import build_model_and_cold_state, build_transform

    cfg = load_config(config)
    transform = build_transform(cfg)
    model, cold = build_model_and_cold_state(cfg, transform)
    del cold
    metadata, arrays = read_checkpoint(
        checkpoint, expected_config_hash=cfg.config_hash, semi_implicit_scheme=cfg.semi_implicit_scheme,
    )
    state = state_from_checkpoint(metadata, arrays, transform.backend)
    model.enforce(state)
    del arrays
    host = transform.backend.to_numpy
    rows: list[dict] = []
    census: list[dict] = []
    context = {"step": 0, "call": 0, "gf_ierr_deep": None, "gf_downdraft_dry_exit": None}
    js = np.asarray([c[0] for c in cells], dtype=np.int64)
    is_ = np.asarray([c[1] for c in cells], dtype=np.int64)

    def histogram(codes: np.ndarray) -> dict[str, int]:
        flat = np.asarray(codes, dtype=np.int64).ravel()
        counts = np.bincount(flat) if flat.size else np.zeros(0, dtype=np.int64)
        return {str(int(code)): int(n) for code, n in enumerate(counts) if n}

    def columns(batch) -> dict[str, np.ndarray]:
        xp = batch.xp
        dp = host(batch.arrays["dp"][:, js, is_]).astype(np.float64)
        moist = 1.0 + host(batch.arrays["qv"][:, js, is_]).astype(np.float64)
        dp_g = dp / moist / GRAVITY_M_S2
        out = {
            "theta_k": host(batch.arrays["theta"][:, js, is_]).astype(np.float64),
            "qv_kg_m2": np.sum(host(batch.arrays["qv"][:, js, is_]).astype(np.float64) * dp_g, axis=0),
            "condensate_kg_m2": np.sum(sum(
                host(batch.arrays[n][:, js, is_]).astype(np.float64) for n in CONDENSATE_SPECIES
            ) * dp_g, axis=0),
        }
        del xp
        return out

    def surface(persistent, name: str) -> np.ndarray:
        return host(persistent.arrays[name][js, is_]).astype(np.float64)

    originals = {}
    for stage in TENDENCY_STAGES:
        original = getattr(nr.NativePhysicsRuntime, stage)
        originals[stage] = original

        def make(original, stage):
            @functools.wraps(original)
            def wrapped(self, batch, persistent, *args, **kwargs):
                before = columns(batch)
                rain_name = "rainc" if stage == "_cumulus_step" else "rainncv"
                rain_before = surface(persistent, rain_name)
                rain_before_field = (
                    host(persistent.arrays[rain_name]).astype(np.float64) if stage == "_cumulus_step" else None
                )
                out = original(self, batch, persistent, *args, **kwargs)
                after = columns(batch)
                rain_after = surface(persistent, rain_name)
                dt_h = float(batch.dt_s) / 3600.0
                dtheta = after["theta_k"] - before["theta_k"]
                # The Grell-Freitas deep exit code per column (gf.cu ierr, 0
                # where deep convection ran) and the downdraft exit its
                # updraft-only switch overrode (0 none, else 7 or 51), when
                # the scheme in the slot carries them: the cells' codes ride
                # on their rows, and every call adds one census row -- the
                # codes over all columns and over the columns whose cumulus
                # rain this call exceeds the heavy threshold, then (from the
                # microphysics stage) over the columns whose grid-scale rain
                # does.
                codes = None
                dry = None
                if stage == "_cumulus_step":
                    held = getattr(self, "_cumulus_column_diagnostics", None) or {}
                    exported = held.get("gf_ierr_deep")
                    context["gf_ierr_deep"] = None if exported is None else host(exported).astype(np.int64)
                    codes = context["gf_ierr_deep"]
                    exported = held.get("gf_downdraft_dry_exit")
                    context["gf_downdraft_dry_exit"] = None if exported is None else host(exported).astype(np.int64)
                    dry = context["gf_downdraft_dry_exit"]
                    if codes is not None:
                        conv_rate = (host(persistent.arrays["rainc"]).astype(np.float64) - rain_before_field) / dt_h
                        heavy_conv = conv_rate > HEAVY_RATE_MM_H
                        entry = {
                            "step": context["step"], "physics_call": context["call"],
                            "all_columns": histogram(codes),
                            "columns_with_convective_rain_above_threshold": histogram(codes[heavy_conv]),
                            "n_columns_with_convective_rain_above_threshold": int(np.count_nonzero(heavy_conv)),
                        }
                        if dry is not None:
                            entry["downdraft_dry_exit_all_columns"] = histogram(dry)
                            entry["downdraft_dry_exit_columns_with_convective_rain_above_threshold"] = histogram(dry[heavy_conv])
                        census.append(entry)
                elif context["gf_ierr_deep"] is not None and census:
                    grid_rate = host(persistent.arrays["rainncv"]).astype(np.float64) / dt_h
                    heavy = grid_rate > HEAVY_RATE_MM_H
                    census[-1]["columns_with_grid_scale_rain_above_threshold"] = histogram(context["gf_ierr_deep"][heavy])
                    census[-1]["n_columns_with_grid_scale_rain_above_threshold"] = int(np.count_nonzero(heavy))
                    if context["gf_downdraft_dry_exit"] is not None:
                        census[-1]["downdraft_dry_exit_columns_with_grid_scale_rain_above_threshold"] = histogram(
                            context["gf_downdraft_dry_exit"][heavy]
                        )
                for n, (j, i) in enumerate(cells):
                    row = {
                        "step": context["step"], "physics_call": context["call"], "stage": stage,
                        "j": int(j), "i": int(i), "dt_s": float(batch.dt_s),
                        "column_heating_k_per_h": float(np.mean(dtheta[:, n]) / dt_h),
                        "max_level_heating_k_per_h": float(np.max(dtheta[:, n]) / dt_h),
                        "dqv_kg_m2_per_h": float((after["qv_kg_m2"][n] - before["qv_kg_m2"][n]) / dt_h),
                        "dcondensate_kg_m2_per_h": float((after["condensate_kg_m2"][n] - before["condensate_kg_m2"][n]) / dt_h),
                        "rain_mm_per_h": float((rain_after[n] - rain_before[n]) / dt_h) if stage == "_cumulus_step" else float(rain_after[n] / dt_h),
                        "dtheta_profile_k": [float(v) for v in dtheta[:, n]],
                    }
                    if codes is not None:
                        row["gf_ierr_deep"] = int(codes[j, i])
                    if dry is not None:
                        row["gf_downdraft_dry_exit"] = int(dry[j, i])
                    rows.append(row)
                return out
            return wrapped

        setattr(nr.NativePhysicsRuntime, stage, make(original, stage))
    original_run = nr.NativePhysicsRuntime.run

    @functools.wraps(original_run)
    def run_wrapped(self, batch, cfg_, persistent=None):
        context["call"] += 1
        return original_run(self, batch, cfg_, persistent)

    nr.NativePhysicsRuntime.run = run_wrapped
    try:
        for _ in range(int(steps)):
            context["step"] += 1
            state, _metrics = model.step(state, cfg.dt_s)
            if progress is not None:
                progress(f"step {context['step']} of {steps}")
    finally:
        for stage, original in originals.items():
            setattr(nr.NativePhysicsRuntime, stage, original)
        nr.NativePhysicsRuntime.run = original_run
    summary = {}
    for stage in TENDENCY_STAGES:
        for j, i in cells:
            sel = [r for r in rows if r["stage"] == stage and r["j"] == j and r["i"] == i]
            if not sel:
                continue
            entry = {
                "calls": len(sel),
                "mean_column_heating_k_per_h": float(np.mean([r["column_heating_k_per_h"] for r in sel])),
                "mean_dqv_kg_m2_per_h": float(np.mean([r["dqv_kg_m2_per_h"] for r in sel])),
                "mean_dcondensate_kg_m2_per_h": float(np.mean([r["dcondensate_kg_m2_per_h"] for r in sel])),
                "mean_rain_mm_per_h": float(np.mean([r["rain_mm_per_h"] for r in sel])),
            }
            if stage == "_cumulus_step" and all("gf_ierr_deep" in r for r in sel):
                entry["gf_ierr_deep_by_call"] = [int(r["gf_ierr_deep"]) for r in sel]
                entry["gf_ierr_deep_histogram"] = histogram(np.asarray([r["gf_ierr_deep"] for r in sel]))
            if stage == "_cumulus_step" and all("gf_downdraft_dry_exit" in r for r in sel):
                entry["gf_downdraft_dry_exit_by_call"] = [int(r["gf_downdraft_dry_exit"]) for r in sel]
                entry["gf_downdraft_dry_exit_histogram"] = histogram(
                    np.asarray([r["gf_downdraft_dry_exit"] for r in sel])
                )
            summary.setdefault(f"{j},{i}", {})[stage] = entry
    return {
        "schema": SCHEMA + "/tendencies",
        "config": str(config), "checkpoint": str(checkpoint), "steps": int(steps), "dt_s": float(cfg.dt_s),
        "cells": [[int(j), int(i)] for j, i in cells],
        "definition": (
            "per physics call: the change the cumulus step and the microphysics step each make to the target "
            "column, divided by the call's dt; column heating is the level-mean theta change; rain is the "
            "scheme's own surface increment (rainc for the cumulus scheme, rainncv for Morrison)"
        ),
        "summary": summary,
        "gf_exit_census": {
            "definition": (
                "per physics call, the Grell-Freitas deep exit code (gf.cu ierr; 0 = deep convection ran, "
                "WRF's numbered exits otherwise) over every column, over the columns whose convective rain "
                f"that call exceeded {HEAVY_RATE_MM_H:g} mm/h and over the columns whose grid-scale rain did; "
                "empty when the cumulus slot carries no such code; the downdraft_dry_exit_* rows count, over "
                "the same column sets, the downdraft exit the updraft-only switch overrode (0 = none, else "
                "WRF's 7 or 51), so the columns the switch rescued are the nonzero entries"
            ),
            "heavy_rate_mm_h": HEAVY_RATE_MM_H,
            "calls": census,
            "downdraft_dry_exit_all_columns_all_calls": _merge_histograms(
                c.get("downdraft_dry_exit_all_columns", {}) for c in census
            ),
            "downdraft_dry_exit_grid_scale_heavy_columns_all_calls": _merge_histograms(
                c.get("downdraft_dry_exit_columns_with_grid_scale_rain_above_threshold", {}) for c in census
            ),
            "grid_scale_heavy_columns_all_calls": _merge_histograms(
                c.get("columns_with_grid_scale_rain_above_threshold", {}) for c in census
            ),
            "convective_heavy_columns_all_calls": _merge_histograms(
                c.get("columns_with_convective_rain_above_threshold", {}) for c in census
            ),
        },
        "rows": rows,
    }


def _merge_histograms(parts) -> dict[str, int]:
    merged: dict[str, int] = {}
    for part in parts:
        for code, n in part.items():
            merged[code] = merged.get(code, 0) + int(n)
    return dict(sorted(merged.items(), key=lambda kv: int(kv[0])))


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


def synthetic_storm(
    *, lat: np.ndarray, lon: np.ndarray, cell: tuple[int, int], rates_mm_h: tuple[float, ...],
    background_mm_h: float = 0.0, convective_mm_h: float = 0.0, hours: int = 24, start_utc_s: float = 0.0,
) -> list[IntervalRain]:
    """Hourly intervals: ``background_mm_h`` of grid-scale rain everywhere,
    ``rates_mm_h`` on ``cell`` in the first ``len(rates_mm_h)`` hours, and
    ``convective_mm_h`` of convective rain everywhere."""
    ny, nx = lat.size, lon.size
    out = []
    for h in range(hours):
        g = np.full((ny, nx), background_mm_h)
        if h < len(rates_mm_h):
            g[cell] += rates_mm_h[h]
        c = np.full((ny, nx), convective_mm_h)
        out.append(IntervalRain(start_utc_s + 3600.0 * h, start_utc_s + 3600.0 * (h + 1), g, c))
    return out


def synthetic_omega_arrays(truncation: int = 21, nlev: int = 8, divergence_s1: float = 1.0e-5, ps_pa: float = 1.0e5):
    """A receipt and checkpoint-like arrays for a resting atmosphere with a
    flat surface pressure and a uniform divergence of ``-D`` on the upper
    half of the levels and ``+D`` on the lower half (low-level convergence
    of ``-D`` would be ascent; this is the descending pattern), for which
    the continuity's interface pressure velocity is analytic:
    ``ps_t = -sum_k D_k dp_k`` and ``omega_half[k] = -sum_{j<k} (delta_b_j
    ps_t + D_j dp_j)``, zero at both boundaries."""
    from woof.globe.vertical import HybridCoordinate
    from woof.globe.spectral.transform import SphericalHarmonicTransform

    transform = SphericalHarmonicTransform.create(truncation, backend="numpy", precision="float64")
    grid = transform.grid
    vertical = HybridCoordinate.pressure_blend(nlev, p_top_pa=100.0)
    receipt = {
        "config": {
            "truncation": truncation, "dealias_factor": 1.5,
            "a_half_pa": [float(v) for v in vertical.a_half_pa], "b_half": [float(v) for v in vertical.b_half],
        },
        "transform": {"nlat": int(grid.nlat), "nlon": int(grid.nlon), "radius_m": float(grid.radius_m)},
        "config_hash": "synthetic",
    }
    ny, nx = int(grid.nlat), int(grid.nlon)
    per_level = np.asarray([-divergence_s1 if k < nlev // 2 else divergence_s1 for k in range(nlev)])
    div = np.stack([transform.forward(np.full((ny, nx), d)) for d in per_level])
    arrays = {
        "atmosphere__vorticity": np.zeros_like(div),
        "atmosphere__divergence": div,
        "atmosphere__log_surface_pressure": transform.forward(np.full((ny, nx), math.log(ps_pa))),
    }
    p_half = vertical.a_half_pa + vertical.b_half * ps_pa
    dp = np.diff(p_half)
    ps_t = -float(np.sum(per_level * dp))
    expected = np.zeros(nlev + 1)
    for k in range(nlev):
        expected[k + 1] = expected[k] - vertical.delta_b[k] * ps_t - per_level[k] * dp[k]
    expected[-1] = 0.0
    return receipt, arrays, expected


def calibrate() -> dict[str, object]:
    """Every synthetic family in both directions; each row carries its
    bar and whether it met it."""
    rows = []
    lat, lon, weights = synthetic_geometry()
    cell = (10, 20)
    rates = (2.0, 30.0, 60.0, 120.0, 12.0)
    intervals = synthetic_storm(lat=lat, lon=lon, cell=cell, rates_mm_h=rates, background_mm_h=0.5, convective_mm_h=0.25)
    stats = storm_statistics(intervals, lat, lon, weights, top_n=3)
    planted_total = sum(rates) + 0.5 * 24
    rows.append({
        "family": "storm cell accumulation", "planted": planted_total,
        "read": stats["max_cell"]["grid_scale_mm"], "bar": 1.0e-12,
        "ok": abs(stats["max_cell"]["grid_scale_mm"] - planted_total) <= 1.0e-12
        and (stats["max_cell"]["grid_scale_cell"]["j"], stats["max_cell"]["grid_scale_cell"]["i"]) == cell,
    })
    rows.append({
        "family": "storm cell peak rate", "planted": max(rates) + 0.5,
        "read": stats["max_rate_over_run_mm_h"]["grid_scale"], "bar": 1.0e-12,
        "ok": abs(stats["max_rate_over_run_mm_h"]["grid_scale"] - (max(rates) + 0.5)) <= 1.0e-12
        and stats["top_cells_by_grid_scale"][0]["peak_grid_scale_interval"] == int(np.argmax(rates)),
    })
    for t in (50.0, 100.0, 200.0, 300.0):
        expect_count = 1 if planted_total >= t else 0
        row = stats["cells_above_total"][f"{t:g}"]
        expect_share = (weights[cell] * planted_total / float(np.sum(weights))) / stats["global_mean_mm"]["grid_scale"] if expect_count else 0.0
        rows.append({
            "family": f"cells above {t:g} mm", "planted": expect_count, "read": row["grid_scale_count"],
            "bar": 0,
            "ok": row["grid_scale_count"] == expect_count and abs((row["grid_scale_share_of_global_mean"] or 0.0) - expect_share) <= 1.0e-12,
        })
    interval_counts = [s["cells_above_rate"]["20"]["grid_scale_count"] for s in stats["per_interval"]]
    expect_counts = [1 if (r + 0.5) >= 20.0 else 0 for r in rates] + [0] * (24 - len(rates))
    rows.append({
        "family": "cells above 20 mm/h per interval", "planted": expect_counts, "read": interval_counts,
        "bar": 0, "ok": interval_counts == expect_counts,
    })
    quiet = synthetic_storm(lat=lat, lon=lon, cell=cell, rates_mm_h=(), background_mm_h=0.2)
    qs = storm_statistics(quiet, lat, lon, weights)
    rows.append({
        "family": "no storm: uniform rain", "planted": 0,
        "read": qs["cells_above_total"]["50"]["grid_scale_count"], "bar": 0,
        "ok": qs["cells_above_total"]["50"]["grid_scale_count"] == 0
        and abs(qs["max_rate_over_run_mm_h"]["grid_scale"] - 0.2) <= 1.0e-12
        and abs(qs["global_mean_mm"]["grid_scale"] - 4.8) <= 1.0e-9,
    })
    for d in (1.0e-5, 0.0):
        receipt, arrays, expected = synthetic_omega_arrays(divergence_s1=d)
        reader = cs.SoundingReader(receipt, omega=True)
        ps = np.full(reader.shape, 1.0e5)
        pressure = reader.vertical.pressure(ps, reader.transform.backend)
        omega = reader.omega_from_arrays(arrays, ps, pressure)
        err = float(np.max(np.abs(omega - expected[:, None, None])))
        scale = float(np.max(np.abs(expected))) if d else 1.0
        mid = len(expected) // 2
        rows.append({
            "family": "omega from a two-layer divergence" if d else "omega at rest", "planted": float(expected[mid]),
            "read": float(omega[mid, 0, 0]), "bar": 1.0e-9 * scale, "ok": err <= 1.0e-9 * scale,
        })
    return {"schema": SCHEMA + "/calibration", "rows": rows, "all_ok": all(r["ok"] for r in rows)}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"not serialisable: {type(value)!r}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=_json_default))


def _parse_cells(values: list[str] | None) -> list[tuple[int, int]]:
    cells = []
    for text in values or ():
        j, i = text.split(",")
        cells.append((int(j), int(i)))
    return cells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", help="directory of arwen_global_step*.npz checkpoints and the receipt")
    parser.add_argument("--checkpoints", nargs="*", help="explicit checkpoint paths (else every step file in --run-dir)")
    parser.add_argument("--config", help="the run's TOML, for a run still writing (no receipt yet)")
    parser.add_argument("--start", help="run start, ISO UTC; else the receipt's start_time_utc")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_CELLS, help="how many top cells to trace")
    parser.add_argument("--cells", nargs="*", help="extra j,i cells to trace")
    parser.add_argument("--no-state", action="store_true", help="statistics only (no soundings)")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", help="JSON output path")
    parser.add_argument("--calibrate", action="store_true", help="run the synthetic families and print their numbers")
    parser.add_argument("--tendencies", nargs=2, metavar=("CONFIG", "CHECKPOINT"),
                        help="restart from CHECKPOINT with CONFIG and record the two schemes' tendencies on --cells (GPU)")
    parser.add_argument("--steps", type=int, default=12, help="steps for --tendencies")
    args = parser.parse_args(argv)
    if args.calibrate:
        payload = calibrate()
        for row in payload["rows"]:
            print(f"{row['family']:40s} planted {row['planted']!s:>22} read {row['read']!s:>22} {'ok' if row['ok'] else 'FAIL'}")
        if args.out:
            _write_json(Path(args.out), payload)
        return 0 if payload["all_ok"] else 1
    if args.tendencies:
        cells = _parse_cells(args.cells)
        if not cells:
            parser.error("--tendencies needs --cells j,i ...")
        result = trace_tendencies(
            args.tendencies[0], args.tendencies[1], cells, steps=args.steps,
            progress=lambda text: print(text, file=sys.stderr),
        )
        result["label"] = args.label
        for key, block in result["summary"].items():
            for stage, row in block.items():
                print(
                    f"tendencies {args.label} cell {key} {stage}: heating {row['mean_column_heating_k_per_h']:+.3f} K/h, "
                    f"dqv {row['mean_dqv_kg_m2_per_h']:+.3f}, dcond {row['mean_dcondensate_kg_m2_per_h']:+.3f} kg/m2/h, "
                    f"rain {row['mean_rain_mm_per_h']:.3f} mm/h over {row['calls']} calls"
                )
        if args.out:
            _write_json(Path(args.out), result)
        return 0
    if not args.run_dir and not args.checkpoints:
        parser.error("--run-dir or --checkpoints is required (or --calibrate / --tendencies)")
    start = None
    if args.start:
        from .diurnal_phase import parse_utc

        start = parse_utc(args.start)
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.checkpoints[0]).parent
    result = measure_run(
        run_dir, start_utc_s=start, checkpoints=args.checkpoints, config=args.config, top_n=args.top,
        extra_cells=_parse_cells(args.cells), state=not args.no_state,
        progress=lambda text: print(text, file=sys.stderr),
    )
    result["label"] = args.label
    print(summary_line(result, args.label))
    if args.out:
        _write_json(Path(args.out), result)
    return 0


__all__ = [
    "IntervalRain", "SCHEMA", "calibrate", "column_readings", "measure_run", "storm_statistics",
    "summary_line", "synthetic_omega_arrays", "synthetic_storm", "trace_tendencies",
]

if __name__ == "__main__":
    sys.exit(main())
