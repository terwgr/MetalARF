"""The Grell-Freitas mass-flux reading: what the closure asked for, what the column got, and which family asked.

What it measures.  On every physics call the Grell-Freitas seam
(``woof.globe.core.gf``) exports, per column, the deep closure reading the
kernel writes beside its tendencies (gf.cu ``GfClosureReading``): the
cloud-base mass flux the sixteen-member ensemble REQUESTED
(``xmb_request``, kg/m2/s, before WRF's 100 kg/m2/s ceiling), the one
the column was APPLIED (``xmb_applied``: the same flux after that
ceiling, neg_check's per-level heating cap and the cuten gate), the four
family requests the mean was built from (members 1, 4, 7 and 10:
quasi-equilibrium, cloud-base omega, Kuo moisture convergence, ECMWF
CAPE relaxation), the resolved moisture convergence ``mconv`` those Kuo
members divide (kg/m2/s), the precipitation per unit mass flux
``pr_ens7``, the neg_check factor (1 where nothing was capped) and the
heating cap it held the column to (K/day), the family with the largest
request, the deep exit code, the downdraft exit the kernel overrode and
the levels its downdraft reached with no mass.

This module turns those arrays into numbers: per region, how many
columns convected, how many were capped and by what factor, how much
rain the request would have made against what was applied and against
the resolved convergence, which family led, and how the exits split; per
named cell, every value.  ``read`` restarts the model of a config from a
hash-verified checkpoint for a few steps and takes the reading on every
cumulus call, with the cumulus and grid-scale rain of the same call
beside it, so the storm cells of the timing lane can be read at the hour
they stormed.

Calibration (``--calibrate``; ``tests/test_arwen_global_gf_massflux.py``
asserts every row): the census arithmetic on planted readings in both
directions (a planted capped fraction and factor read back exactly and a
planet with nothing capped reads zero; planted families and exits count
exactly; a planted convergence reads its own rain rate).  The kernel side
of the calibration (a planted vertical velocity requests the moisture
convergence's own flux; a column over the cap reads capped by the cap's
own factor; untouched columns bitwise under either kernel) runs on the
device in the same test file.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from .diurnal_phase import gaussian_grid_coordinates, region_mask

SCHEMA = "gpuwm.arwen-global-massflux-diagnostic/v1"
#: closure_family codes, gf.cu GfClosureReading.
FAMILIES = {0: "none", 1: "quasi_equilibrium", 2: "omega",
            3: "moisture_convergence", 4: "ecmwf"}
FAMILY_REQUESTS = {1: "xf_quasi_equilibrium", 2: "xf_omega",
                   3: "xf_moisture_convergence", 4: "xf_ecmwf"}
#: A column counts as capped when the applied flux is under this fraction
#: of the request (float32 products round at 1e-7; the caps move by tenths).
CAPPED_BELOW = 1.0 - 1.0e-4
#: WRF's deep neg_check threshold, K/day (gf.cu K_THRESH_DEEP).
WRF_HEATING_CAP_K_DAY = 300.01
SECONDS_PER_HOUR = 3600.0
#: The regions the census reports, beyond the whole planet, land and ocean.
DEFAULT_REGIONS = ("conus_land", "global_land")
#: The arrays one reading carries (``gf_`` prefix stripped).
READING_NAMES = (
    "xmb_request", "xmb_applied", "xmb_floor", "floor_fraction", "xf_quasi_equilibrium", "xf_omega",
    "xf_moisture_convergence", "xf_ecmwf", "xf_dicycle", "mconv", "mconv_den",
    "pr_ens7", "sig", "closure_n", "neg_check_factor", "heating_cap_k_day",
    "closure_family", "downdraft_massless_levels", "ierr_deep",
    "downdraft_dry_exit",
)


# --------------------------------------------------------------------------
# the census
# --------------------------------------------------------------------------

def _stats(values: np.ndarray, weights: np.ndarray) -> dict[str, float | None]:
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    keep = np.isfinite(v) & (w > 0)
    v, w = v[keep], w[keep]
    if v.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "max": None}
    order = np.argsort(v)
    cw = np.cumsum(w[order]) / np.sum(w)
    def q(p):
        return float(v[order][min(int(np.searchsorted(cw, p)), v.size - 1)])
    return {"n": int(v.size), "mean": float(np.sum(v * w) / np.sum(w)),
            "p50": q(0.5), "p90": q(0.9), "max": float(np.max(v))}


def _hist(codes: np.ndarray, names: dict[int, str] | None = None) -> dict[str, int]:
    flat = np.asarray(codes, dtype=np.int64).ravel()
    if flat.size == 0:
        return {}
    values, counts = np.unique(flat, return_counts=True)
    out = {}
    for value, count in zip(values, counts):
        key = names.get(int(value), str(int(value))) if names else str(int(value))
        out[key] = int(count)
    return out


def census(
    reading: dict[str, np.ndarray],
    weights: np.ndarray,
    mask: np.ndarray,
    *,
    grid_rain_mm_h: np.ndarray | None = None,
    convective_rain_mm_h: np.ndarray | None = None,
) -> dict[str, object]:
    """The mass-flux census of one reading over the columns of ``mask``.

    ``reading`` maps every READING_NAMES entry to a ``(ny, nx)`` array;
    ``weights`` are the columns' area weights.  Rain rates, when given,
    add the heavy-column rows (columns above 20 mm/h of grid-scale rain).
    """
    m = np.asarray(mask, dtype=bool)
    w = np.asarray(weights, dtype=np.float64)
    req = np.asarray(reading["xmb_request"], dtype=np.float64)
    app = np.asarray(reading["xmb_applied"], dtype=np.float64)
    ierr = np.asarray(reading["ierr_deep"], dtype=np.int64)
    fam = np.asarray(reading["closure_family"], dtype=np.int64)
    factor = np.asarray(reading["neg_check_factor"], dtype=np.float64)
    cap = np.asarray(reading["heating_cap_k_day"], dtype=np.float64)
    mconv = np.asarray(reading["mconv"], dtype=np.float64)
    pr7 = np.asarray(reading["pr_ens7"], dtype=np.float64)
    dry = np.asarray(reading["downdraft_dry_exit"], dtype=np.int64)
    massless = np.asarray(reading["downdraft_massless_levels"], dtype=np.int64)
    area = float(np.sum(w[m]))
    active = m & (ierr == 0) & (app > 0.0)
    requested = m & (req > 0.0)
    capped = requested & (app < req * CAPPED_BELOW)
    floor = np.asarray(reading["xmb_floor"], dtype=np.float64)
    # The floor BOUND where it exceeded the ensemble's request on a column
    # the deep arm carried through (the kernel's own test, cup_output_ens_3d:
    # xmb_floor > xmb).  The request may be zero: a column whose sixteen-
    # member mean fell under the diurnal-cycle term convects on the floor
    # alone, and WRF's kernel would have left it with ierr 19.  An earlier
    # form of this row demanded request > 0 and applied > request and so
    # missed those columns and every floored column neg_check then capped
    # under its request: 1,313 of 3,063 floored columns at the arm's hour
    # 12 (measured 2026-09-05, both kernels' readings).
    floored = m & (ierr == 0) & (floor > req)
    floored_zero_request = floored & (req <= 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cap_factor = np.where(req > 0.0, app / req, np.nan)
    out: dict[str, object] = {
        "columns": int(np.sum(m)),
        "area_fraction_of_planet": float(area / max(float(np.sum(w)), 1e-300)),
        "deep_active_columns": int(np.sum(active)),
        "deep_active_area_fraction": float(np.sum(w[active]) / area) if area > 0 else None,
        "requesting_columns": int(np.sum(requested)),
        "exit_code_histogram": _hist(ierr[m]),
        "closure_family_histogram_active": _hist(fam[active], FAMILIES),
        "closure_family_area_share_active": {
            FAMILIES[k]: float(np.sum(w[active & (fam == k)]) / max(np.sum(w[active]), 1e-300))
            for k in (1, 2, 3, 4)
        },
        "capped": {
            "columns": int(np.sum(capped)),
            "fraction_of_requesting_columns": float(np.sum(capped) / max(np.sum(requested), 1)),
            "area_fraction_of_requesting": float(np.sum(w[capped]) / max(np.sum(w[requested]), 1e-300)),
            "applied_over_request": _stats(cap_factor[capped], w[capped]),
            "neg_check_factor": _stats(factor[capped], w[capped]),
            "heating_cap_at_wrf_value_columns": int(np.sum(capped & (np.abs(cap - WRF_HEATING_CAP_K_DAY) < 1e-2))),
            "heating_cap_raised_columns": int(np.sum(capped & (cap > WRF_HEATING_CAP_K_DAY + 1e-2))),
            "heating_cap_k_day": _stats(cap[capped], w[capped]),
        },
        "floored": {
            # the resolved-convergence floor bound (floor > request on a
            # deep-active column); 0 under WRF's kernel
            "columns": int(np.sum(floored)),
            "fraction_of_active_columns": float(np.sum(floored) / max(np.sum(active), 1)),
            "fraction_of_requesting_columns": float(np.sum(floored & requested) / max(np.sum(requested), 1)),
            # the request was zero: the sixteen-member mean sat under the
            # diurnal-cycle term and the column convects on the floor alone
            "columns_with_zero_request": int(np.sum(floored_zero_request)),
            "applied_over_request": _stats(cap_factor[floored & requested], w[floored & requested]),
            "floor_over_request": _stats(np.where(req > 0, floor / np.where(req > 0, req, 1.0), np.nan)[floored & requested], w[floored & requested]),
        },
        "mass_flux_kg_m2_s": {
            "request_active": _stats(req[active], w[active]),
            "applied_active": _stats(app[active], w[active]),
            "request_area_mean_all_columns": float(np.sum(req[m] * w[m]) / area) if area > 0 else None,
            "applied_area_mean_all_columns": float(np.sum(app[m] * w[m]) / area) if area > 0 else None,
        },
        "rain_mm_h": {
            # the rain a mass flux makes: xmb * pr_ens7, kg/m2/s -> mm/h
            "request_would_make_active": _stats(req[active] * pr7[active] * SECONDS_PER_HOUR, w[active]),
            "applied_makes_active": _stats(app[active] * pr7[active] * SECONDS_PER_HOUR, w[active]),
            "resolved_moisture_convergence_active": _stats(mconv[active] * SECONDS_PER_HOUR, w[active]),
            "resolved_moisture_convergence_requesting": _stats(mconv[requested] * SECONDS_PER_HOUR, w[requested]),
            "request_area_mean_all_columns": float(np.sum(req[m] * pr7[m] * w[m]) * SECONDS_PER_HOUR / area) if area > 0 else None,
            "applied_area_mean_all_columns": float(np.sum(app[m] * pr7[m] * w[m]) * SECONDS_PER_HOUR / area) if area > 0 else None,
        },
        "family_requests_active_mean_kg_m2_s": {
            FAMILIES[k]: (float(np.sum(np.asarray(reading[name], dtype=np.float64)[active] * w[active]) / np.sum(w[active]))
                          if np.any(active) else None)
            for k, name in FAMILY_REQUESTS.items()
        },
        "downdraft": {
            "dry_exit_overridden_histogram": _hist(dry[m]),
            "massless_levels_histogram": _hist(massless[m]),
            "columns_with_massless_levels": int(np.sum(m & (massless > 0))),
        },
    }
    if grid_rain_mm_h is not None:
        g = np.asarray(grid_rain_mm_h, dtype=np.float64)
        heavy = m & (g > 20.0)
        heavy_capped = heavy & capped
        out["grid_scale_heavy_columns"] = {
            "threshold_mm_h": 20.0,
            "columns": int(np.sum(heavy)),
            "exit_code_histogram": _hist(ierr[heavy]),
            "deep_active_columns": int(np.sum(heavy & active)),
            "capped_columns": int(np.sum(heavy_capped)),
            "applied_over_request": _stats(cap_factor[heavy_capped], w[heavy_capped]),
            "closure_family_histogram_active": _hist(fam[heavy & active], FAMILIES),
            "resolved_moisture_convergence_mm_h": _stats(mconv[heavy] * SECONDS_PER_HOUR, w[heavy]),
            "request_would_make_mm_h": _stats(req[heavy] * pr7[heavy] * SECONDS_PER_HOUR, w[heavy]),
            "applied_makes_mm_h": _stats(app[heavy] * pr7[heavy] * SECONDS_PER_HOUR, w[heavy]),
            "grid_scale_rain_mm_h": _stats(g[heavy], w[heavy]),
            "downdraft_dry_exit_overridden_histogram": _hist(dry[heavy]),
        }
        if convective_rain_mm_h is not None:
            c = np.asarray(convective_rain_mm_h, dtype=np.float64)
            out["grid_scale_heavy_columns"]["convective_rain_mm_h"] = _stats(c[heavy], w[heavy])
    return out


def column_rows(reading: dict[str, np.ndarray], cells: list[tuple[int, int]], **extra) -> list[dict]:
    rows = []
    for j, i in cells:
        row = {"j": int(j), "i": int(i)}
        for name in READING_NAMES:
            value = reading[name][j, i]
            row[name] = int(value) if name in ("closure_family", "downdraft_massless_levels", "ierr_deep", "downdraft_dry_exit") else float(value)
        row["closure_family_name"] = FAMILIES.get(row["closure_family"], str(row["closure_family"]))
        pr = row["pr_ens7"]
        row["request_would_make_mm_h"] = row["xmb_request"] * pr * SECONDS_PER_HOUR
        row["applied_makes_mm_h"] = row["xmb_applied"] * pr * SECONDS_PER_HOUR
        row["resolved_moisture_convergence_mm_h"] = row["mconv"] * SECONDS_PER_HOUR
        row["applied_over_request"] = (row["xmb_applied"] / row["xmb_request"]) if row["xmb_request"] > 0 else None
        for key, arr in extra.items():
            row[key] = float(np.asarray(arr)[j, i])
        rows.append(row)
    return rows


def planet_masks(lat: np.ndarray, lon: np.ndarray, land: np.ndarray, regions=DEFAULT_REGIONS) -> dict[str, np.ndarray]:
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    masks = {
        "planet": np.ones(lat2.shape, dtype=bool),
        "land": np.asarray(land, dtype=bool),
        "ocean": ~np.asarray(land, dtype=bool),
        "tropics_20s_20n": np.abs(lat2) <= 20.0,
    }
    for name in regions:
        masks[name] = region_mask(name, lat2, lon2, np.asarray(land, dtype=bool))
    return masks


def area_weights(lat: np.ndarray, nlon: int) -> np.ndarray:
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    return np.repeat(w[:, None], int(nlon), axis=1)


# --------------------------------------------------------------------------
# the reading of a run's checkpoint
# --------------------------------------------------------------------------

def read_calls(
    config: str | Path, checkpoint: str | Path, cells: list[tuple[int, int]], *,
    steps: int = 1, progress=None, unverified_config: bool = False,
) -> dict[str, object]:
    """Restart ``config`` from ``checkpoint``, take the closure reading on
    every cumulus call of ``steps`` steps, and census it.  Needs the run's
    backend (cupy).  Returns the census per call and region, the named
    cells' rows per call, and the last call's planet arrays.

    ``unverified_config`` reads a checkpoint whose config hash is not the
    config's: the instrument's own use, reading one archived state under
    two kernels (the WRF-faithful one that wrote it and the coarse-column
    one) so the two closures can be compared on the same columns.  The
    payload records both hashes and says the check was waived; the
    restart door never takes this path."""
    import functools

    from woof.globe.checkpoint import read_checkpoint, state_from_checkpoint
    from woof.globe.config import load_config
    from woof.globe.physics import native_runtime as nr
    from woof.globe.runner import build_model_and_cold_state, build_transform

    cfg = load_config(config)
    transform = build_transform(cfg)
    model, cold = build_model_and_cold_state(cfg, transform)
    del cold
    metadata, arrays = read_checkpoint(
        checkpoint, expected_config_hash=None if unverified_config else cfg.config_hash,
        semi_implicit_scheme=cfg.semi_implicit_scheme,
    )
    state = state_from_checkpoint(metadata, arrays, transform.backend)
    model.enforce(state)
    del arrays
    host = transform.backend.to_numpy
    calls: list[dict] = []
    planets: list[dict[str, np.ndarray]] = []
    context = {"step": 0, "call": 0, "rainc_before": None}
    original_cumulus = nr.NativePhysicsRuntime._cumulus_step
    original_mp = nr.NativePhysicsRuntime._microphysics_step
    original_run = nr.NativePhysicsRuntime.run
    geometry = {}

    @functools.wraps(original_cumulus)
    def cumulus_wrapped(self, batch, persistent, *args, **kwargs):
        rainc_before = host(persistent.arrays["rainc"]).astype(np.float64)
        out = original_cumulus(self, batch, persistent, *args, **kwargs)
        dt_h = float(batch.dt_s) / SECONDS_PER_HOUR
        held = self._cumulus_column_diagnostics
        reading = {}
        for name in READING_NAMES:
            arr = held.get(f"gf_{name}")
            if arr is None:
                raise ValueError(
                    f"the cumulus slot carries no gf_{name} reading: the mass-flux "
                    "diagnostic reads woof.globe.core.gf's closure export and no other scheme"
                )
            reading[name] = host(arr)
        conv_rate = (host(persistent.arrays["rainc"]).astype(np.float64) - rainc_before) / dt_h
        if not geometry:
            land = host(batch.surface.land_fraction) >= 0.5
            lat, lon = gaussian_grid_coordinates(*land.shape)
            geometry.update(lat=lat, lon=lon, land=land, weights=area_weights(lat, land.shape[1]),
                            masks=planet_masks(lat, lon, land))
        calls.append({"step": context["step"], "physics_call": context["call"], "dt_s": float(batch.dt_s),
                      "reading": reading, "convective_rain_mm_h": conv_rate, "grid_rain_mm_h": None})
        return out

    @functools.wraps(original_mp)
    def mp_wrapped(self, batch, persistent, *args, **kwargs):
        out = original_mp(self, batch, persistent, *args, **kwargs)
        dt_h = float(batch.dt_s) / SECONDS_PER_HOUR
        if calls and calls[-1]["grid_rain_mm_h"] is None:
            calls[-1]["grid_rain_mm_h"] = host(persistent.arrays["rainncv"]).astype(np.float64) / dt_h
        return out

    @functools.wraps(original_run)
    def run_wrapped(self, batch, cfg_, persistent=None):
        context["call"] += 1
        return original_run(self, batch, cfg_, persistent)

    nr.NativePhysicsRuntime._cumulus_step = cumulus_wrapped
    nr.NativePhysicsRuntime._microphysics_step = mp_wrapped
    nr.NativePhysicsRuntime.run = run_wrapped
    try:
        for _ in range(int(steps)):
            context["step"] += 1
            state, _metrics = model.step(state, cfg.dt_s)
            if progress is not None:
                progress(f"step {context['step']} of {steps}")
    finally:
        nr.NativePhysicsRuntime._cumulus_step = original_cumulus
        nr.NativePhysicsRuntime._microphysics_step = original_mp
        nr.NativePhysicsRuntime.run = original_run
    if not calls:
        raise ValueError("no cumulus call ran: the config carries no cumulus scheme")
    masks = geometry["masks"]
    weights = geometry["weights"]
    per_call = []
    for entry in calls:
        reading = entry["reading"]
        regions = {
            name: census(reading, weights, mask, grid_rain_mm_h=entry["grid_rain_mm_h"],
                         convective_rain_mm_h=entry["convective_rain_mm_h"])
            for name, mask in masks.items()
        }
        extra = {"convective_rain_mm_h": entry["convective_rain_mm_h"]}
        if entry["grid_rain_mm_h"] is not None:
            extra["grid_rain_mm_h"] = entry["grid_rain_mm_h"]
        per_call.append({
            "step": entry["step"], "physics_call": entry["physics_call"], "dt_s": entry["dt_s"],
            "regions": regions, "cells": column_rows(reading, cells, **extra),
        })
    last = calls[-1]
    planet = {name: np.asarray(last["reading"][name]) for name in READING_NAMES}
    planet["convective_rain_mm_h"] = np.asarray(last["convective_rain_mm_h"], dtype=np.float32)
    if last["grid_rain_mm_h"] is not None:
        planet["grid_rain_mm_h"] = np.asarray(last["grid_rain_mm_h"], dtype=np.float32)
    planet["latitude_deg"] = geometry["lat"]
    planet["longitude_deg"] = geometry["lon"]
    planet["land"] = geometry["land"]
    return {
        "schema": SCHEMA,
        "config": str(config), "checkpoint": str(checkpoint), "config_hash": cfg.config_hash,
        "checkpoint_config_hash": metadata.get("config_hash"),
        "config_hash_verified": not unverified_config,
        "checkpoint_step": int(metadata.get("step", -1)), "checkpoint_time_s": float(metadata.get("time_s", float("nan"))),
        "native_adapter_options": dict(cfg.native_adapter_options),
        "steps": int(steps), "calls": per_call,
        "cells": [{"j": int(j), "i": int(i), "latitude_deg": float(geometry["lat"][j]),
                   "longitude_deg": float(geometry["lon"][i])} for j, i in cells],
        "cell_means": _cell_means(per_call),
        "reading_names": list(READING_NAMES),
    }, planet


def _cell_means(per_call: list[dict]) -> list[dict]:
    if not per_call:
        return []
    out = []
    n_cells = len(per_call[0]["cells"])
    keys = ("xmb_request", "xmb_applied", "request_would_make_mm_h", "applied_makes_mm_h",
            "resolved_moisture_convergence_mm_h", "neg_check_factor", "heating_cap_k_day",
            "convective_rain_mm_h", "grid_rain_mm_h", "mconv_den", "pr_ens7")
    for n in range(n_cells):
        rows = [c["cells"][n] for c in per_call]
        entry = {"j": rows[0]["j"], "i": rows[0]["i"], "calls": len(rows)}
        for key in keys:
            values = [r[key] for r in rows if key in r and r[key] is not None]
            entry[f"mean_{key}"] = float(np.mean(values)) if values else None
        entry["exit_code_histogram"] = _hist(np.asarray([r["ierr_deep"] for r in rows]))
        entry["closure_family_histogram"] = _hist(np.asarray([r["closure_family"] for r in rows]), FAMILIES)
        entry["downdraft_dry_exit_histogram"] = _hist(np.asarray([r["downdraft_dry_exit"] for r in rows]))
        entry["capped_calls"] = int(sum(1 for r in rows if r["xmb_request"] > 0 and r["xmb_applied"] < r["xmb_request"] * CAPPED_BELOW))
        for k, name in FAMILY_REQUESTS.items():
            entry[f"mean_{name}"] = float(np.mean([r[name] for r in rows]))
        out.append(entry)
    return out


# --------------------------------------------------------------------------
# calibration: the census arithmetic on planted readings, both directions
# --------------------------------------------------------------------------

def synthetic_reading(
    nlat: int = 16, nlon: int = 32, *, capped_fraction: float = 0.0, cap_factor: float = 1.0,
    family: int = 3, active_fraction: float = 0.5, mconv_mm_h: float = 12.0, pr7: float = 0.01,
    exit_code: int = 2, seed: int = 0,
    floored_fraction: float = 0.0, floor_multiple: float = 2.0, floored_capped_factor: float = 1.0,
    zero_request_floored: int = 0, dry_exit_7: int = 0, dry_exit_51: int = 0,
    massless_levels: dict[int, int] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float | int]]:
    """A planted reading on ``(nlat, nlon)``: the first ``active_fraction``
    of the columns (by count) convect with family ``family`` at a request
    of ``mconv / pr7`` (the Kuo member's own answer), the first
    ``capped_fraction`` of THOSE are applied ``cap_factor`` times their
    request, the rest of the planet carries ``exit_code``.

    The floor and downdraft rows are planted from the far end of the active
    block so they never overlap the capped columns: the last
    ``floored_fraction`` of the active columns carry a floor of
    ``floor_multiple`` times their request and are applied that floor times
    ``floored_capped_factor`` (1: the floor as it stands; under 1: a floored
    column neg_check then capped, which can land its applied flux UNDER its
    request while the floor still bound); the ``zero_request_floored``
    active columns before them request nothing and carry a floor of the
    Kuo member (a column the diurnal-cycle term silenced); ``dry_exit_7`` and
    ``dry_exit_51`` active columns report those downdraft exits overridden;
    ``massless_levels`` plants a histogram {levels: columns} on active
    columns."""
    rng = np.random.default_rng(seed)
    n = nlat * nlon
    n_active = int(round(active_fraction * n))
    n_capped = int(round(capped_fraction * n_active))
    mconv = mconv_mm_h / SECONDS_PER_HOUR
    req = np.zeros(n)
    app = np.zeros(n)
    ierr = np.full(n, int(exit_code), dtype=np.int64)
    fam = np.zeros(n, dtype=np.int64)
    factor = np.ones(n)
    cap = np.full(n, WRF_HEATING_CAP_K_DAY)
    floor = np.zeros(n)
    dry = np.zeros(n, dtype=np.int64)
    massless = np.zeros(n, dtype=np.int64)
    req[:n_active] = mconv / pr7 * (1.0 + 0.1 * rng.standard_normal(n_active))
    req[:n_active] = np.abs(req[:n_active]) + 1e-6
    app[:n_active] = req[:n_active]
    app[:n_capped] = req[:n_capped] * cap_factor
    factor[:n_capped] = cap_factor
    ierr[:n_active] = 0
    fam[:n_active] = int(family)
    n_floored = int(round(floored_fraction * n_active))
    n_zero = int(zero_request_floored)
    if n_capped + n_zero + n_floored > n_active:
        raise ValueError("the planted capped, zero-request and floored blocks overlap: shrink one")
    if n_floored:
        lo, hi = n_active - n_floored, n_active
        floor[lo:hi] = req[lo:hi] * float(floor_multiple)
        app[lo:hi] = floor[lo:hi] * float(floored_capped_factor)
        factor[lo:hi] = float(floored_capped_factor)
    if n_zero:
        lo, hi = n_active - n_floored - n_zero, n_active - n_floored
        req[lo:hi] = 0.0
        floor[lo:hi] = mconv / pr7
        app[lo:hi] = floor[lo:hi]
    if dry_exit_7 or dry_exit_51:
        dry[n_capped:n_capped + int(dry_exit_7)] = 7
        dry[n_capped + int(dry_exit_7):n_capped + int(dry_exit_7) + int(dry_exit_51)] = 51
    if massless_levels:
        start = n_capped
        for levels, count in sorted(massless_levels.items()):
            massless[start:start + int(count)] = int(levels)
            start += int(count)
        if start > n_active:
            raise ValueError("the planted massless-level columns run past the active block")
    shape = (nlat, nlon)
    reading = {
        "xmb_request": req.reshape(shape), "xmb_applied": app.reshape(shape),
        "xmb_floor": floor.reshape(shape),
        "floor_fraction": np.where(floor > 0, 1.0, 0.0).reshape(shape),
        "xf_quasi_equilibrium": np.where(fam == 1, req, 0.1 * req).reshape(shape),
        "xf_omega": np.where(fam == 2, req, 0.1 * req).reshape(shape),
        "xf_moisture_convergence": np.where(fam == 3, req, 0.1 * req).reshape(shape),
        "xf_ecmwf": np.where(fam == 4, req, 0.1 * req).reshape(shape),
        "xf_dicycle": np.zeros(shape), "mconv": np.where(ierr == 0, mconv, 0.0).reshape(shape),
        "mconv_den": np.ones(shape), "pr_ens7": np.where(ierr == 0, pr7, 0.0).reshape(shape),
        "sig": np.full(shape, 0.98), "closure_n": np.full(shape, 16.0),
        "neg_check_factor": factor.reshape(shape), "heating_cap_k_day": cap.reshape(shape),
        "closure_family": fam.reshape(shape), "downdraft_massless_levels": massless.reshape(shape),
        "ierr_deep": ierr.reshape(shape), "downdraft_dry_exit": dry.reshape(shape),
    }
    weights = np.ones(shape)
    exact = {"active": n_active, "capped": n_capped, "capped_fraction": (n_capped / n_active) if n_active else 0.0,
             "cap_factor": cap_factor, "family": family, "mconv_mm_h": mconv_mm_h, "exit_code": exit_code,
             "floored": n_floored + n_zero, "floored_with_request": n_floored, "floored_zero_request": n_zero,
             "floor_multiple": float(floor_multiple), "floored_applied_over_request": float(floor_multiple) * float(floored_capped_factor),
             "dry_exit_7": int(dry_exit_7), "dry_exit_51": int(dry_exit_51),
             "massless_levels": dict(massless_levels or {})}
    return reading, weights, exact


def calibrate() -> dict[str, object]:
    rows = []

    def row(family, planted, read, ok, note=""):
        rows.append({"family": family, "planted": planted, "read": read, "ok": bool(ok), "note": note})

    # capped fraction and factor, both directions
    for capped_fraction, cap_factor in ((0.0, 1.0), (0.3, 0.25), (1.0, 0.6), (0.5, 0.0)):
        reading, weights, exact = synthetic_reading(capped_fraction=capped_fraction, cap_factor=cap_factor)
        c = census(reading, weights, np.ones(weights.shape, dtype=bool))
        got = c["capped"]["columns"]
        row("capped columns", exact["capped"], got, got == exact["capped"], f"cap_factor {cap_factor}")
        got_f = c["capped"]["fraction_of_requesting_columns"]
        row("capped fraction of requesting", exact["capped_fraction"], got_f, abs(got_f - exact["capped_fraction"]) < 1e-12)
        mean = c["capped"]["applied_over_request"]["mean"]
        if exact["capped"] == 0:
            row("applied over request, nothing capped", None, mean, mean is None)
        else:
            row("applied over request", cap_factor, mean, abs(mean - cap_factor) < 1e-12)
        row("deep active columns", exact["active"], c["deep_active_columns"],
            c["deep_active_columns"] == (exact["active"] if cap_factor > 0 else exact["active"] - exact["capped"]),
            "an applied flux of zero is not active")
    # families and exits count exactly
    for family in (1, 2, 3, 4):
        reading, weights, exact = synthetic_reading(family=family, exit_code=7)
        c = census(reading, weights, np.ones(weights.shape, dtype=bool))
        got = c["closure_family_histogram_active"].get(FAMILIES[family], 0)
        row("closure family count", exact["active"], got, got == exact["active"], FAMILIES[family])
        share = c["closure_family_area_share_active"][FAMILIES[family]]
        row("closure family area share", 1.0, share, abs(share - 1.0) < 1e-12, FAMILIES[family])
        got = c["exit_code_histogram"].get("7", 0)
        row("exit code count", weights.size - exact["active"], got, got == weights.size - exact["active"])
    # a planted convergence reads its own rain rate, and none reads none
    for mconv_mm_h in (0.0, 12.0, 122.0):
        reading, weights, exact = synthetic_reading(mconv_mm_h=mconv_mm_h)
        c = census(reading, weights, np.ones(weights.shape, dtype=bool))
        got = c["rain_mm_h"]["resolved_moisture_convergence_active"]["mean"]
        row("resolved convergence mm/h", mconv_mm_h, got, got is not None and abs(got - mconv_mm_h) < 1e-9)
        # the Kuo request would make its own rain, request * pr7: the planted
        # request carries a spread (and a 1e-6 floor at zero convergence),
        # so the expectation is the planted columns' own mean
        got = c["rain_mm_h"]["request_would_make_active"]["mean"]
        want = float(np.mean(reading["xmb_request"][reading["ierr_deep"] == 0] * 0.01 * SECONDS_PER_HOUR))
        row("request would make mm/h", want, got, got is not None and abs(got - want) < 1e-9)
    # a mask that selects nothing reads zeros, not NaNs
    reading, weights, exact = synthetic_reading()
    c = census(reading, weights, np.zeros(weights.shape, dtype=bool))
    row("empty mask", 0, c["deep_active_columns"], c["deep_active_columns"] == 0 and c["capped"]["columns"] == 0)
    # heavy columns: planted grid-scale rain above the threshold
    reading, weights, exact = synthetic_reading(capped_fraction=1.0, cap_factor=0.5)
    grid = np.zeros(weights.shape)
    grid.ravel()[: exact["capped"]] = 50.0
    c = census(reading, weights, np.ones(weights.shape, dtype=bool), grid_rain_mm_h=grid)
    got = c["grid_scale_heavy_columns"]["capped_columns"]
    row("heavy capped columns", exact["capped"], got, got == exact["capped"])
    # the floor rows, both directions: nothing planted reads zero; a planted
    # floored block reads its count, its floor over request and its applied
    # over request exactly, whether the floor stands or neg_check then capped
    # the column under its own request; zero-request floored columns count
    # as floored and are named apart
    for floored_fraction, multiple, capped_factor, n_zero in ((0.0, 2.0, 1.0, 0), (0.3, 2.0, 1.0, 0), (0.25, 1.5, 0.5, 0), (0.2, 3.0, 1.0, 7), (0.0, 2.0, 1.0, 5)):
        reading, weights, exact = synthetic_reading(capped_fraction=0.1, cap_factor=0.5, floored_fraction=floored_fraction,
                                                    floor_multiple=multiple, floored_capped_factor=capped_factor,
                                                    zero_request_floored=n_zero)
        c = census(reading, weights, np.ones(weights.shape, dtype=bool))
        note = f"floor x{multiple}, applied factor {capped_factor}, zero-request {n_zero}"
        got = c["floored"]["columns"]
        row("floored columns", exact["floored"], got, got == exact["floored"], note)
        got = c["floored"]["columns_with_zero_request"]
        row("floored columns with zero request", exact["floored_zero_request"], got, got == exact["floored_zero_request"], note)
        got = c["floored"]["fraction_of_active_columns"]
        want = exact["floored"] / exact["active"]
        row("floored fraction of active", want, got, abs(got - want) < 1e-12, note)
        mean = c["floored"]["floor_over_request"]["mean"]
        if exact["floored_with_request"] == 0:
            row("floor over request, nothing floored with a request", None, mean, mean is None, note)
        else:
            row("floor over request", multiple, mean, abs(mean - multiple) < 1e-12, note)
            got = c["floored"]["applied_over_request"]["mean"]
            row("floored applied over request", exact["floored_applied_over_request"], got,
                abs(got - exact["floored_applied_over_request"]) < 1e-12, note)
        # the capped row is untouched by the floor block unless the floored
        # column was itself capped under its request
        capped_expected = exact["capped"] + (exact["floored_with_request"] if exact["floored_applied_over_request"] < 1.0 else 0)
        row("capped columns beside a floor block", capped_expected, c["capped"]["columns"], c["capped"]["columns"] == capped_expected, note)
        # a floored column's applied flux is not "active" arithmetic the floor
        # changes: every planted active column stays active
        row("deep active columns beside a floor block", exact["active"], c["deep_active_columns"], c["deep_active_columns"] == exact["active"], note)
    # the downdraft rows, both directions
    for dry7, dry51, massless in ((0, 0, None), (3, 4, None), (0, 0, {1: 6, 3: 2, 5: 1}), (2, 0, {2: 4})):
        reading, weights, exact = synthetic_reading(dry_exit_7=dry7, dry_exit_51=dry51, massless_levels=massless)
        c = census(reading, weights, np.ones(weights.shape, dtype=bool))
        hist = c["downdraft"]["dry_exit_overridden_histogram"]
        row("downdraft exit 7 overridden count", dry7, hist.get("7", 0), hist.get("7", 0) == dry7)
        row("downdraft exit 51 overridden count", dry51, hist.get("51", 0), hist.get("51", 0) == dry51)
        row("downdraft exit 0 (none overridden) count", weights.size - dry7 - dry51, hist.get("0", 0), hist.get("0", 0) == weights.size - dry7 - dry51)
        want = {str(k): v for k, v in (massless or {}).items()}
        got = {k: v for k, v in c["downdraft"]["massless_levels_histogram"].items() if k != "0"}
        row("downdraft massless-level histogram", want, got, got == want)
        n_massless = sum((massless or {}).values())
        row("columns with massless levels", n_massless, c["downdraft"]["columns_with_massless_levels"], c["downdraft"]["columns_with_massless_levels"] == n_massless)
    return {"schema": SCHEMA + "/calibration", "rows": rows, "all_ok": all(r["ok"] for r in rows)}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def summary_lines(payload: dict, label: str = "") -> list[str]:
    lines = []
    tag = f"[{label}] " if label else ""
    for call in payload["calls"]:
        p = call["regions"]["planet"]
        heavy = p.get("grid_scale_heavy_columns", {})
        lines.append(
            f"{tag}call {call['physics_call']} (step {call['step']}): deep active {p['deep_active_columns']} of "
            f"{p['columns']}; capped {p['capped']['columns']} and floored {p['floored']['columns']} of {p['requesting_columns']} requesting "
            f"(applied/request mean {p['capped']['applied_over_request']['mean']}); exits {p['exit_code_histogram']}; "
            f"families {p['closure_family_histogram_active']}; heavy columns {heavy.get('columns')} with exits "
            f"{heavy.get('exit_code_histogram')} and capped {heavy.get('capped_columns')}"
        )
    for cell in payload["cell_means"]:
        lines.append(
            f"{tag}cell {cell['j']},{cell['i']}: request {cell['mean_request_would_make_mm_h']} mm/h, applied "
            f"{cell['mean_applied_makes_mm_h']} mm/h, resolved convergence {cell['mean_resolved_moisture_convergence_mm_h']} mm/h, "
            f"conv rain {cell['mean_convective_rain_mm_h']} mm/h, grid rain {cell['mean_grid_rain_mm_h']} mm/h, "
            f"exits {cell['exit_code_histogram']}, families {cell['closure_family_histogram']}, capped {cell['capped_calls']} of {cell['calls']} calls"
        )
    return lines


def _parse_cells(values: list[str] | None) -> list[tuple[int, int]]:
    cells = []
    for value in values or []:
        j, i = value.split(",")
        cells.append((int(j), int(i)))
    return cells


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"not serialisable: {type(value)!r}")


def _write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=1, default=_json_default)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibrate", action="store_true", help="run the census calibration rows and print them")
    parser.add_argument("--read", nargs=2, metavar=("CONFIG", "CHECKPOINT"),
                        help="restart CONFIG from CHECKPOINT and take the reading on every cumulus call")
    parser.add_argument("--steps", type=int, default=1, help="steps for --read")
    parser.add_argument("--cells", nargs="*", help="j,i cells whose rows are recorded per call")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", help="JSON output path")
    parser.add_argument("--npz", help="the last call's planet arrays (compressed npz)")
    parser.add_argument("--unverified-config", action="store_true",
                        help="read a checkpoint another kernel wrote (the config hash check is waived and recorded)")
    args = parser.parse_args(argv)
    if args.calibrate:
        payload = calibrate()
        for r in payload["rows"]:
            print(f"{'ok ' if r['ok'] else 'BAD'} {r['family']}: planted {r['planted']} read {r['read']} {r['note']}")
        print("all_ok", payload["all_ok"])
        if args.out:
            _write_json(Path(args.out), payload)
        return 0 if payload["all_ok"] else 1
    if args.read:
        config, checkpoint = args.read
        payload, planet = read_calls(config, checkpoint, _parse_cells(args.cells), steps=args.steps,
                                     progress=lambda m: print(m, file=sys.stderr, flush=True),
                                     unverified_config=args.unverified_config)
        payload["label"] = args.label
        for line in summary_lines(payload, args.label):
            print(line)
        if args.out:
            _write_json(Path(args.out), payload)
        if args.npz:
            Path(args.npz).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.npz, **planet)
        return 0
    parser.error("one of --calibrate or --read is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
