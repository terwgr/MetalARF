"""YSU free-atmosphere diffusivity and momentum-drain probe.

What it measures.  On one checkpoint of a WOOF global run the model is
rebuilt from the run's TOML, restarted from the checkpoint and stepped
until the first YSU call of the first physics half-step; that call's
inputs (the column state and the surface-layer coupling fields the
scheme consumed) and outputs (``exch_m``, ``exch_h``, ``du``, ``dv``,
``hpbl``, ``kpbl``, ``delta``) are captured and read three ways:

* the K profile: per interface (``exch_m[k + 1]`` is the momentum
  diffusivity between full levels ``k`` and ``k + 1``, surface-first),
  the area-weighted mean and the 50, 90 and 99th percentiles over the
  sphere and per hemisphere of ``exch_m`` and ``exch_h`` on the
  free-atmosphere columns (``k + 1 >= kpbl``), with the level pressures;
* the formula terms behind that K, recomputed in float64 from the same
  inputs level by level exactly as ``kernels/ysu.cu`` (WRF v4.6.1
  ``bl_ysu.F90:986-1024``) forms them: the layer thickness ``dza``, the
  shear ``sqrt(ss)``, the gradient Richardson number ``ri``, ``zk =
  kappa z``, the asymptotic length ``rlamdz = min(dz, clip(0.1 dz, 30 m,
  300 m))``, the Blackadar length ``rl``, ``dk = rl^2 sqrt(ss)``, the
  stability functions ``f_m(ri)`` and ``f_h(ri)``, the products ``km``
  and ``kh``, the floors ``xkzminm = 0.1`` and ``xkzminh = 0.01`` and the
  cap ``xkzmax = 1000``; the kernel's word is checked against the
  recomputed ``min(km + floor, cap)`` on every interface outside the
  entrainment zone (where WRF overwrites ``xkzm`` with the entrainment
  diffusivity, ``bl_ysu.F90:1265``-equivalent ``ysu.cu`` momentum matrix),
  and the counterfactual K under the ``"fixed"`` length
  (:mod:`woof.globe.core.ysu_contract`, carried) is read beside it;
* the momentum tendency: per level the sphere and hemisphere means of
  the kinetic-energy tendency the call applies, ``u du + v dv`` times
  ``dp/g`` (W/m2 per level), and the same per spectral band (the energy
  ledger's bands, ``woof.globe.insitu.energy.spectral_bands``)
  after the wind and its tendency are both band-filtered through the
  vector spherical-harmonic analysis, so a drain is placed at a level
  AND a band; the free-atmosphere and boundary-layer parts are summed
  separately.

Calibration (``calibrate``), two synthetic families in both directions:
a planted jet column with a known gradient Richardson number and shear
must read that ``ri`` back and a K equal to the formula's under both
length modes; a stably stratified zero-shear column must read exactly
the floor and a zero shear term; and a single-degree rotational
tendency on a resting sphere must be read in its band and in no other,
a zero tendency as zero everywhere.  ``tests/test_arwen_global_pbl_
free_atmosphere.py`` holds the families on the float64 mirror; the same
function on the CUDA kernel (``--backend cupy``) is the calibration of
record for the probe's readings.

It is a reader: nothing here changes a scheme.  The WRF column
reference (``export_columns`` and ``compare_wrf_reference``) hands the
captured columns to the byte-unmodified Fortran through
``tools/ysu_wrf461_oracle/run_bl_ysu_columns.F90`` and reads its
``exch_mx`` back beside the kernel's.

    python -m woof.globe.pbl_free_atmosphere probe --config TOML
        --checkpoint NPZ --out JSON [--levels LO HI]
        [--export-columns PREFIX --n-columns N]
    python -m woof.globe.pbl_free_atmosphere calibrate --out JSON
        [--backend numpy|cupy]
    python -m woof.globe.pbl_free_atmosphere compare-wrf
        --export PREFIX --wrf-levels CSV --out JSON
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from woof.core.constants import CP, G, RD, RV, XLV
from .core.ysu_contract import (
    YSU_FIXED_ASYMPTOTIC_LENGTH_M,
    YSU_FREE_ATMOSPHERE_MIXING_LENGTHS,
)

from .insitu.energy import HEMISPHERES, band_label, spectral_bands

SCHEMA = "gpuwm.arwen-global-pbl-free-atmosphere/v1"

KARMAN = 0.4
XKZMINM, XKZMINH, XKZMAX = 0.1, 0.01, 1000.0
RIMIN, RLAM, PRMAX = -100.0, 30.0, 4.0
#: bl_ysu.F90 (ysu.cu) leaves the entrainment override where
#: ``entfac = ((zq[k+1] - hpbl) / delta)^2 < 4.6``.
ENTRAINMENT_ENTFAC = 4.6

#: Latitude belts the K profile is read over, beside the hemispheres.
REGIONS = {
    "global": (-90.0, 90.0),
    "northern": (0.0, 90.0),
    "southern": (-90.0, 0.0),
    "north_20_70": (20.0, 70.0),
    "south_20_70": (-70.0, -20.0),
}

MODES = tuple(YSU_FREE_ATMOSPHERE_MIXING_LENGTHS)


# --------------------------------------------------------------------------
# the formula, level by level, in float64
# --------------------------------------------------------------------------


def column_geometry(dz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``zq`` (nz + 1, ...), ``za`` (nz, ...) and ``dza`` (nz, ...) from the
    layer depths, exactly as ysu.cu forms them (surface-first)."""
    dz = np.asarray(dz, dtype=np.float64)
    zq = np.concatenate([np.zeros((1, *dz.shape[1:])), np.cumsum(dz, axis=0)], axis=0)
    za = 0.5 * (zq[:-1] + zq[1:])
    dza = np.empty_like(za)
    dza[0] = za[0]
    dza[1:] = za[1:] - za[:-1]
    return zq, za, dza


def free_atmosphere_terms(
    u, v, theta, qv, qc, qi, exner, dz, kpbl, *, modes: tuple[str, ...] = MODES,
) -> dict[str, np.ndarray]:
    """The free-atmosphere diffusivity of ysu.cu, term by term, on
    ``(nz, ...)`` surface-first columns; every array returned is
    ``(nz - 1, ...)``, entry ``k`` for the interface between full levels
    ``k`` and ``k + 1``, and ``active`` marks the interfaces the scheme
    treats as free atmosphere (``k + 1 >= kpbl``, WRF's one-based
    ``kpbl``).  Per mode ``m`` in ``modes``: ``rlamdz_m``, ``rl_m``,
    ``dk_m``, ``km_m``, ``kh_m``, ``xkzm_m``, ``xkzh_m``."""
    u, v, theta, qv, qc, qi, exner, dz = (
        np.asarray(a, dtype=np.float64) for a in (u, v, theta, qv, qc, qi, exner, dz)
    )
    # One kpbl per column: a scalar for a single column, (ny, nx) for a grid.
    kpbl = np.asarray(kpbl, dtype=np.int64).reshape(theta.shape[1:])
    nz = theta.shape[0]
    ep1 = RV / RD - 1.0
    thv = theta * (1.0 + ep1 * qv)
    temp = theta * exner
    zq, _za, dza = column_geometry(dz)
    k = np.arange(nz - 1)
    active = (k + 1).reshape((nz - 1,) + (1,) * (theta.ndim - 1)) >= kpbl[None]
    dzai = dza[1:]
    ud = u[1:] - u[:-1]
    vd = v[1:] - v[:-1]
    ss = (ud * ud + vd * vd) / (dzai * dzai) + 1.0e-9
    govrthv = G / (0.5 * (thv[1:] + thv[:-1]))
    ri = govrthv * (thv[1:] - thv[:-1]) / (ss * dzai)
    cloudy = ((qc[:-1] + qi[:-1]) > 1.0e-5) & ((qc[1:] + qi[1:]) > 1.0e-5)
    if np.any(cloudy):
        qmean = 0.5 * (qv[:-1] + qv[1:])
        tmean = 0.5 * (temp[:-1] + temp[1:])
        alph = XLV * qmean / RD / tmean
        chi = XLV * XLV * qmean / CP / RV / (tmean * tmean)
        moist = (1.0 + alph) * (ri - G * G / ss / tmean / CP * ((chi - alph) / (1.0 + chi)))
        ri = np.where(cloudy, moist, ri)
    zk = KARMAN * zq[1:nz]
    unstable = ri < 0.0
    ri_c = np.where(unstable, np.maximum(ri, RIMIN), ri)
    sri = np.sqrt(np.maximum(-ri_c, 0.0))
    f_m = np.where(
        unstable,
        1.0 + 8.0 * (-ri_c) / (1.0 + 1.746 * sri),
        np.minimum(1.0 + 2.1 * np.maximum(ri_c, 0.0), PRMAX) / (1.0 + 5.0 * np.maximum(ri_c, 0.0)) ** 2,
    )
    f_h = np.where(
        unstable,
        1.0 + 8.0 * (-ri_c) / (1.0 + 1.286 * sri),
        1.0 / (1.0 + 5.0 * np.maximum(ri_c, 0.0)) ** 2,
    )
    out = {
        "active": active,
        "dza": dzai,
        "zq": zq[1:nz],
        "shear": np.sqrt(ss),
        "ri": ri,
        "cloudy": cloudy,
        "zk": zk,
        "f_m": f_m,
        "f_h": f_h,
    }
    for mode in modes:
        if mode == "wrf-layer":
            rlamdz = np.minimum(dzai, np.clip(0.1 * dzai, RLAM, 300.0))
        elif mode == "fixed":
            rlamdz = np.minimum(dzai, YSU_FIXED_ASYMPTOTIC_LENGTH_M)
        else:
            raise ValueError(f"unknown mixing-length mode {mode!r}")
        rl = zk * rlamdz / (rlamdz + zk)
        dk = rl * rl * np.sqrt(ss)
        km = dk * f_m
        kh = dk * f_h
        out[f"rlamdz_{mode}"] = rlamdz
        out[f"rl_{mode}"] = rl
        out[f"dk_{mode}"] = dk
        out[f"km_{mode}"] = km
        out[f"kh_{mode}"] = kh
        out[f"xkzm_{mode}"] = np.minimum(km + XKZMINM, XKZMAX)
        out[f"xkzh_{mode}"] = np.minimum(kh + XKZMINH, XKZMAX)
    return out


def entrainment_zone(dz, hpbl, delta, kpbl) -> np.ndarray:
    """``(nz - 1, ...)`` mask of the interfaces where ysu.cu replaces the
    free-atmosphere ``xkzm`` by the entrainment diffusivity: ``k + 1 >=
    kpbl`` and ``((zq[k + 1] - hpbl) / delta)^2 < 4.6`` with ``delta > 0``
    (the ``pblflg`` condition is folded into ``delta > 0``: the kernel
    leaves ``delta`` at zero when the layer is not convective)."""
    zq, _za, _dza = column_geometry(dz)
    nz = zq.shape[0] - 1
    plane = zq.shape[1:]
    hpbl = np.asarray(hpbl, dtype=np.float64).reshape(plane)
    delta = np.asarray(delta, dtype=np.float64).reshape(plane)
    kpbl = np.asarray(kpbl, dtype=np.int64).reshape(plane)
    k = np.arange(nz - 1)
    above = (k + 1).reshape((nz - 1,) + (1,) * len(plane)) >= kpbl[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        e = (zq[1:nz] - hpbl[None]) / delta[None]
    inside = np.where(delta[None] > 0.0, e * e < ENTRAINMENT_ENTFAC, False)
    return above & inside


# --------------------------------------------------------------------------
# sphere means and the band projection
# --------------------------------------------------------------------------


def _region_weights(latitude_deg: np.ndarray, quadrature: np.ndarray) -> dict[str, np.ndarray]:
    """Per region the ``(nlat,)`` quadrature weights of its rows, halved
    on an equator row, zero elsewhere: ``0.5 * sum(w * zonal_mean)`` is
    the sphere mean restricted to the region (the ledger's convention)."""
    lat = np.asarray(latitude_deg, dtype=np.float64)
    w = np.asarray(quadrature, dtype=np.float64)
    out = {}
    for name, (lo, hi) in REGIONS.items():
        inside = ((lat > lo) & (lat < hi)).astype(np.float64)
        inside += 0.5 * ((lat == lo) | (lat == hi))
        out[name] = w * np.clip(inside, 0.0, 1.0)
    return out


def region_mean(field, weights, xp=np) -> float:
    """``0.5 * sum_j w_j * mean_i field[j, i]`` (the sphere mean over the
    region's rows, in the region's share of the sphere)."""
    zonal = xp.mean(field, axis=-1, dtype=xp.float64)
    return float(0.5 * xp.sum(zonal * xp.asarray(weights, dtype=xp.float64)))


def region_fraction(weights) -> float:
    return float(0.5 * np.sum(np.asarray(weights, dtype=np.float64)))


def band_kinetic_tendency(vector, transform, u, v, du, dv, dp_g, weights_by_region) -> dict[str, object]:
    """Per level and per band the region means of ``u_b du_b + v_b dv_b``
    times ``dp/g`` (W/m2 per level), the unfiltered total beside them, on
    surface-first ``(nz, nlat, nlon)`` grids (the level order is the
    caller's; the transform acts per level)."""
    xp = transform.backend.xp
    truncation = int(transform.truncation)
    bands = spectral_bands(truncation)
    labels = [band_label(lo, hi) for lo, hi in bands]
    dtype = transform.backend.float_dtype
    u = xp.asarray(u, dtype=dtype)
    v = xp.asarray(v, dtype=dtype)
    du = xp.asarray(du, dtype=dtype)
    dv = xp.asarray(dv, dtype=dtype)
    dp_g = xp.asarray(dp_g, dtype=xp.float64)
    regions = list(weights_by_region)
    nz = int(u.shape[0])

    def reading(uu, vv, duu, dvv):
        rate = (uu.astype(xp.float64) * duu.astype(xp.float64)
                + vv.astype(xp.float64) * dvv.astype(xp.float64)) * dp_g
        return np.asarray([[region_mean(rate[k], weights_by_region[r], xp) for r in regions]
                           for k in range(nz)])

    total = reading(u, v, du, dv)
    zeta, div = vector.vordiv_from_wind(u, v)
    zeta_t, div_t = vector.vordiv_from_wind(du, dv)
    masks = np.zeros((len(bands), truncation + 1), dtype=np.float64)
    for index, (lo, hi) in enumerate(bands):
        masks[index, lo:hi + 1] = 1.0
    per_band = []
    for index in range(len(bands)):
        mask = xp.asarray(masks[index], dtype=dtype)[:, None]
        u_b, v_b = vector.wind_from_vordiv(zeta * mask, div * mask)
        du_b, dv_b = vector.wind_from_vordiv(zeta_t * mask, div_t * mask)
        per_band.append(reading(u_b, v_b, du_b, dv_b))
        del u_b, v_b, du_b, dv_b
    return {
        "bands": labels,
        "band_edges": [[int(lo), int(hi)] for lo, hi in bands],
        "regions": regions,
        "units": "W/m2 per level: region mean of (u du + v dv) dp/g, the wind and its tendency band-filtered by the vector analysis",
        "total": total.tolist(),
        "per_band": [b.tolist() for b in per_band],
        "cross_band": (total - sum(per_band)).tolist(),
    }


# --------------------------------------------------------------------------
# the probe on one checkpoint
# --------------------------------------------------------------------------


class _Captured(Exception):
    pass


def capture_first_pbl_call(config: str | Path, checkpoint: str | Path) -> dict[str, object]:
    """Restart the model of ``config`` from ``checkpoint`` and capture the
    inputs and outputs of the first YSU call of the first physics
    half-step (needs the run's backend).  Returns host numpy arrays,
    surface-first, plus the model's transform and vector operator."""
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
        checkpoint, expected_config_hash=cfg.config_hash,
        semi_implicit_scheme=cfg.semi_implicit_scheme,
    )
    state = state_from_checkpoint(metadata, arrays, transform.backend)
    model.enforce(state)
    del arrays
    host = transform.backend.to_numpy
    captured: dict[str, object] = {}
    original = nr.NativePhysicsRuntime._pbl_step

    @functools.wraps(original)
    def wrapped(self, batch, persistent):
        f = persistent.arrays
        inputs = {
            name: host(batch.arrays[name]).astype(np.float32)
            for name in ("u", "v", "theta", "qv", "qc", "qi", "exner", "p_full", "p_half", "dp",
                         "temperature", "latitude_deg", "longitude_deg")
        }
        inputs["dz"] = host(self._dz(batch)).astype(np.float32)
        inputs["rthraten"] = host(f["rad_rthratenlw"] + f["rad_rthratensw"]).astype(np.float32)
        for name, key in (("ust", "ust"), ("hfx", "hfx"), ("qfx", "qfx"), ("wspd", "wspd"),
                          ("br", "br"), ("psim", "fm"), ("psih", "fh"), ("znt", "znt"),
                          ("u10", "u10"), ("v10", "v10")):
            inputs[name] = host(f[key]).astype(np.float32)
        inputs["xland"] = host(1.0 + (1.0 - batch.surface.land_fraction)).astype(np.float32)
        out = original(self, batch, persistent)
        outputs = {name: host(out[name]) for name in ("exch_m", "exch_h", "du", "dv", "hpbl", "kpbl", "delta", "wstar")}
        captured.update(
            inputs=inputs, outputs=outputs, dt_s=float(batch.dt_s), time_s=float(batch.time_s),
            mixing_length=str(self.options.ysu_free_atmosphere_mixing_length),
        )
        raise _Captured()

    nr.NativePhysicsRuntime._pbl_step = wrapped
    try:
        try:
            model.step(state, cfg.dt_s)
        except _Captured:
            pass
    finally:
        nr.NativePhysicsRuntime._pbl_step = original
    if not captured:
        raise RuntimeError("the step made no YSU call (physics mode is not arwen-native, or pbl is not ysu)")
    captured["config_hash"] = cfg.config_hash
    captured["step"] = int(metadata["step"])
    captured["transform"] = transform
    captured["vector"] = model.vector
    captured["quadrature_weights"] = np.asarray(transform.grid.quadrature_weights, dtype=np.float64)
    captured["latitude_rows_deg"] = np.asarray(transform.grid.latitude_deg, dtype=np.float64)
    return captured


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan"), "max": float("nan")}
    p50, p90, p99 = np.percentile(values, [50.0, 90.0, 99.0])
    return {"p50": float(p50), "p90": float(p90), "p99": float(p99), "max": float(np.max(values))}


def read_capture(captured: dict[str, object], *, levels: tuple[int, int] | None = None) -> dict[str, object]:
    """Every reading of the probe on one captured call: the K profile,
    the formula terms, the transcription check, the counterfactual, the
    per-level per-band momentum tendency.  ``levels`` (lo, hi inclusive,
    surface-first full-level indices) names the levels of interest in the
    summary; the arrays carry every level."""
    inp = captured["inputs"]
    out = captured["outputs"]
    nz, nlat, nlon = inp["theta"].shape
    kpbl = np.asarray(out["kpbl"], dtype=np.int64)
    terms = free_atmosphere_terms(
        inp["u"], inp["v"], inp["theta"], inp["qv"], inp["qc"], inp["qi"], inp["exner"], inp["dz"], kpbl,
    )
    active = terms["active"]
    zone = entrainment_zone(inp["dz"], out["hpbl"], out["delta"], kpbl)
    checkable = active & ~zone
    exch_m = np.asarray(out["exch_m"], dtype=np.float64)[1:]      # interface k+1 -> index k
    exch_h = np.asarray(out["exch_h"], dtype=np.float64)[1:]
    mode = captured["mixing_length"]
    formula_m = terms[f"xkzm_{mode}"]
    formula_h = terms[f"xkzh_{mode}"]
    diff_m = np.abs(exch_m - formula_m)[checkable]
    rel_m = (diff_m / np.maximum(np.abs(formula_m[checkable]), XKZMINM))
    diff_h = np.abs(exch_h - formula_h)[checkable]
    quadrature = captured["quadrature_weights"]
    lat_rows = captured["latitude_rows_deg"]
    weights = _region_weights(lat_rows, quadrature)
    # Column weights on the (nlat, nlon) planes: quadrature per row, halved
    # over the longitudes so that sum(w2d * field) is the sphere mean.
    w2d = {r: np.repeat((0.5 * w / nlon)[:, None], nlon, axis=1) for r, w in weights.items()}
    p_full_mean = np.asarray([region_mean(inp["p_full"][k].astype(np.float64), weights["global"]) for k in range(nz)])
    p_half_mean = np.asarray([region_mean(inp["p_half"][k].astype(np.float64), weights["global"]) for k in range(nz + 1)])

    def masked_mean(field, mask, region):
        w = w2d[region] * mask
        total = float(np.sum(w))
        if total <= 0.0:
            return float("nan"), 0.0
        return float(np.sum(w * field) / total), total / region_fraction(weights[region])

    per_level = []
    for k in range(nz - 1):
        row = {
            "interface": k + 1,
            "levels": [k, k + 1],
            "p_full_below_pa": float(p_full_mean[k]),
            "p_full_above_pa": float(p_full_mean[k + 1]),
            "p_interface_pa": float(p_half_mean[k + 1]),
            "regions": {},
        }
        for region in REGIONS:
            mean_m, frac = masked_mean(exch_m[k], active[k], region)
            mean_h, _ = masked_mean(exch_h[k], active[k], region)
            entry = {
                "free_atmosphere_fraction": frac,
                "entrainment_zone_fraction": masked_mean(np.ones_like(exch_m[k]), zone[k], region)[1] if np.any(zone[k]) else 0.0,
                "exch_m_mean": mean_m,
                "exch_h_mean": mean_h,
                "exch_m_percentiles": _percentiles(exch_m[k][active[k] & (w2d[region] > 0.0)]),
                "dza_mean": masked_mean(terms["dza"][k], active[k], region)[0],
                "shear_mean": masked_mean(terms["shear"][k], active[k], region)[0],
                "ri_median": float(np.median(terms["ri"][k][active[k] & (w2d[region] > 0.0)])) if np.any(active[k] & (w2d[region] > 0.0)) else float("nan"),
                "stable_fraction": masked_mean((terms["ri"][k] >= 0.0).astype(np.float64), active[k], region)[0],
                "f_m_mean": masked_mean(terms["f_m"][k], active[k], region)[0],
            }
            for m in MODES:
                entry[f"rlamdz_{m}_mean"] = masked_mean(terms[f"rlamdz_{m}"][k], active[k], region)[0]
                entry[f"rl_{m}_mean"] = masked_mean(terms[f"rl_{m}"][k], active[k], region)[0]
                entry[f"xkzm_{m}_mean"] = masked_mean(terms[f"xkzm_{m}"][k], active[k], region)[0]
                entry[f"xkzh_{m}_mean"] = masked_mean(terms[f"xkzh_{m}"][k], active[k], region)[0]
            # The floor's share and the shear term's share of the mean K.
            entry["floor_share_of_exch_m"] = XKZMINM / mean_m if mean_m and np.isfinite(mean_m) and mean_m > 0 else float("nan")
            row["regions"][region] = entry
        per_level.append(row)

    # Momentum tendency: per level and per band.
    dp_g = inp["dp"].astype(np.float64) / G
    bands = band_kinetic_tendency(
        captured["vector"], captured["transform"], inp["u"], inp["v"], out["du"], out["dv"], dp_g, weights,
    )
    total = np.asarray(bands["total"])            # (nz, nregions)
    free_mask = np.zeros((nz, nlat, nlon), dtype=np.float64)
    for k in range(nz):
        free_mask[k] = (k + 1 >= kpbl)            # the level's own index against kpbl
    rate = (inp["u"].astype(np.float64) * out["du"].astype(np.float64)
            + inp["v"].astype(np.float64) * out["dv"].astype(np.float64)) * dp_g
    regions = list(REGIONS)
    free_part = np.asarray([[region_mean(rate[k] * free_mask[k], weights[r]) for r in regions] for k in range(nz)])
    pbl_part = total - free_part
    level_ke = np.asarray([[region_mean(0.5 * (inp["u"][k].astype(np.float64) ** 2 + inp["v"][k].astype(np.float64) ** 2) * dp_g[k], weights[r]) for r in regions] for k in range(nz)])
    lo, hi = (0, nz - 1) if levels is None else (int(levels[0]), int(levels[1]))
    summary = {
        "levels_of_interest": [lo, hi],
        "transcription_check": {
            "interfaces_checked": int(np.count_nonzero(checkable)),
            "interfaces_in_entrainment_zone": int(np.count_nonzero(zone & active)),
            "exch_m_max_abs_diff_m2_s": float(np.max(diff_m)) if diff_m.size else 0.0,
            "exch_m_max_rel_diff": float(np.max(rel_m)) if rel_m.size else 0.0,
            "exch_m_rel_diff_percentiles": _percentiles(rel_m),
            "exch_m_interfaces_above_1e-3_rel": int(np.count_nonzero(rel_m > 1.0e-3)),
            "exch_m_interfaces_above_1e-2_rel": int(np.count_nonzero(rel_m > 1.0e-2)),
            "exch_h_max_abs_diff_m2_s": float(np.max(diff_h)) if diff_h.size else 0.0,
            "definition": "kernel exch_m[k+1] against the float64 recomputation min(km + 0.1, 1000) of ysu.cu:538-572 on every free-atmosphere interface outside the entrainment zone; the bulk of the difference is the float32 rounding of the kernel, the tail is the interfaces whose cloudy-branch test (qc + qi > 1e-5) or Richardson number sits at a branch and rounds to the other side in float32",
        },
        "kinetic_tendency_w_m2": {
            "regions": regions,
            "definition": "sum over the levels of interest of the region mean of (u du + v dv) dp/g; free = levels at or above kpbl, pbl = below",
            "levels_of_interest_total": [float(np.sum(total[lo:hi + 1, r])) for r in range(len(regions))],
            "levels_of_interest_free": [float(np.sum(free_part[lo:hi + 1, r])) for r in range(len(regions))],
            "levels_of_interest_pbl": [float(np.sum(pbl_part[lo:hi + 1, r])) for r in range(len(regions))],
            "column_total": [float(np.sum(total[:, r])) for r in range(len(regions))],
            "column_free": [float(np.sum(free_part[:, r])) for r in range(len(regions))],
            "column_pbl": [float(np.sum(pbl_part[:, r])) for r in range(len(regions))],
            "levels_of_interest_per_band": [
                [float(np.sum(np.asarray(bands["per_band"][b])[lo:hi + 1, r])) for r in range(len(regions))]
                for b in range(len(bands["bands"]))
            ],
            "levels_of_interest_kinetic_energy_j_m2": [float(np.sum(level_ke[lo:hi + 1, r])) for r in range(len(regions))],
        },
    }
    # Per-band per-level e-folding rate (per day) against the level's band energy needs the band energies:
    return {
        "schema": SCHEMA,
        "mixing_length_mode": mode,
        "config_hash": captured["config_hash"],
        "checkpoint_step": captured["step"],
        "time_s": captured["time_s"],
        "dt_s": captured["dt_s"],
        "shape": [int(nz), int(nlat), int(nlon)],
        "regions": {name: {"latitude_deg": list(REGIONS[name]), "sphere_fraction": region_fraction(weights[name])} for name in REGIONS},
        "summary": summary,
        "per_interface": per_level,
        "per_level_kinetic_tendency_w_m2": {
            "regions": regions,
            "p_full_pa": p_full_mean.tolist(),
            "total": total.tolist(),
            "free_atmosphere": free_part.tolist(),
            "boundary_layer": pbl_part.tolist(),
            "level_kinetic_energy_j_m2": level_ke.tolist(),
        },
        "per_band": bands,
        "definition": __doc__.split("Calibration")[0].strip(),
    }


# --------------------------------------------------------------------------
# WRF column reference: export and compare
# --------------------------------------------------------------------------

LEVEL_INPUT_FIELDS = ("ux", "vx", "tx", "qvx", "qcx", "qix", "p2d", "pi2d", "p2di_k", "p2di_kp1", "dz8w", "rthraten")
SURFACE_INPUT_FIELDS = ("psfcpa", "znt", "ust", "hfx", "qfx", "wspd", "br", "psim", "psih", "xland", "u10", "v10")


def select_columns(captured: dict[str, object], n_columns: int, *, level: int | None = None, seed: int = 20260905) -> list[tuple[int, int]]:
    """``n_columns`` cells: the half with the largest ``exch_m`` at the
    interface above ``level`` (the level nearest 250 hPa by default)
    between 20N and 70N, and the other half drawn uniformly from the same
    belt with a fixed seed."""
    inp = captured["inputs"]
    out = captured["outputs"]
    lat = inp["latitude_deg"]
    p_full = inp["p_full"]
    nz = p_full.shape[0]
    if level is None:
        column_mean = np.asarray([float(np.mean(p_full[k])) for k in range(nz)])
        level = int(np.argmin(np.abs(column_mean - 25_000.0)))
    belt = (lat >= 20.0) & (lat <= 70.0)
    k_face = min(level + 1, nz - 1)
    score = np.where(belt, out["exch_m"][k_face], -np.inf).ravel()
    top = max(1, n_columns // 2)
    order = np.argsort(-score)[:top]
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(belt.ravel())
    pool = pool[~np.isin(pool, order)]
    rest = rng.choice(pool, size=max(0, n_columns - top), replace=False) if pool.size else np.zeros(0, dtype=np.int64)
    cells = np.concatenate([order, rest])
    nlon = lat.shape[1]
    return [(int(c // nlon), int(c % nlon)) for c in cells]


def export_columns(captured: dict[str, object], prefix: str | Path, cells: list[tuple[int, int]]) -> dict[str, str]:
    """Write ``<prefix>-levels.csv`` and ``<prefix>-surface.csv`` in the
    oracle harness's input layout (one-based ``k``, surface-first) for
    ``run_bl_ysu_columns``, and ``<prefix>-kernel.json`` with the
    kernel's outputs on the same cells."""
    inp = captured["inputs"]
    out = captured["outputs"]
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    nz = inp["theta"].shape[0]
    levels_path = prefix.with_name(prefix.name + "-levels.csv")
    surface_path = prefix.with_name(prefix.name + "-surface.csv")
    kernel_path = prefix.with_name(prefix.name + "-kernel.json")

    def f32(x):
        return repr(float(np.float32(x)))

    with levels_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("case,k," + ",".join(LEVEL_INPUT_FIELDS) + "\n")
        for case, (j, i) in enumerate(cells, start=1):
            for k in range(nz):
                values = [
                    inp["u"][k, j, i], inp["v"][k, j, i], inp["temperature"][k, j, i],
                    inp["qv"][k, j, i], inp["qc"][k, j, i], inp["qi"][k, j, i],
                    inp["p_full"][k, j, i], inp["exner"][k, j, i],
                    inp["p_half"][k, j, i], inp["p_half"][k + 1, j, i],
                    inp["dz"][k, j, i], inp["rthraten"][k, j, i],
                ]
                stream.write(f"{case},{k + 1}," + ",".join(f32(v) for v in values) + "\n")
    with surface_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("case,nz,dt,topdown," + ",".join(SURFACE_INPUT_FIELDS) + "\n")
        for case, (j, i) in enumerate(cells, start=1):
            values = [
                inp["p_half"][0, j, i], inp["znt"][j, i], inp["ust"][j, i], inp["hfx"][j, i], inp["qfx"][j, i],
                inp["wspd"][j, i], inp["br"][j, i], inp["psim"][j, i], inp["psih"][j, i], inp["xland"][j, i],
                inp["u10"][j, i], inp["v10"][j, i],
            ]
            stream.write(f"{case},{nz},{f32(captured['dt_s'])},1," + ",".join(f32(v) for v in values) + "\n")
    kernel = {
        "schema": SCHEMA + "/kernel-columns",
        "mixing_length_mode": captured["mixing_length"],
        "cells": [[int(j), int(i)] for j, i in cells],
        "latitude_deg": [float(inp["latitude_deg"][j, i]) for j, i in cells],
        "longitude_deg": [float(inp["longitude_deg"][j, i]) for j, i in cells],
        "p_full_pa": [[float(inp["p_full"][k, j, i]) for k in range(nz)] for j, i in cells],
        "exch_m": [[float(out["exch_m"][k, j, i]) for k in range(nz)] for j, i in cells],
        "exch_h": [[float(out["exch_h"][k, j, i]) for k in range(nz)] for j, i in cells],
        "du": [[float(out["du"][k, j, i]) for k in range(nz)] for j, i in cells],
        "dv": [[float(out["dv"][k, j, i]) for k in range(nz)] for j, i in cells],
        "hpbl": [float(out["hpbl"][j, i]) for j, i in cells],
        "kpbl": [int(out["kpbl"][j, i]) for j, i in cells],
    }
    kernel_path.write_text(json.dumps(kernel, indent=1), encoding="utf-8")
    return {"levels": str(levels_path), "surface": str(surface_path), "kernel": str(kernel_path)}


def _ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    ia = a32.view(np.int32).astype(np.int64)
    ib = b32.view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, np.int64(-2**31) - ia, ia)
    ib = np.where(ib < 0, np.int64(-2**31) - ib, ib)
    return np.abs(ia - ib)


def compare_wrf_reference(kernel_json: str | Path, wrf_levels_csv: str | Path, *, levels: tuple[int, int] | None = None) -> dict[str, object]:
    """The byte-unmodified Fortran's ``exch_mx`` and momentum tendencies on
    the exported cells beside the kernel's: per level the max ULP and
    relative distance, and the WRF K profile itself."""
    kernel = json.loads(Path(kernel_json).read_text(encoding="utf-8"))
    rows: dict[int, dict[int, dict[str, float]]] = {}
    with Path(wrf_levels_csv).open("r", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.setdefault(int(row["case"]), {})[int(row["k"])] = {key: float(value) for key, value in row.items()}
    ncase = len(kernel["cells"])
    nz = len(kernel["exch_m"][0])
    if set(rows) != set(range(1, ncase + 1)):
        raise ValueError(f"{wrf_levels_csv}: cases {sorted(rows)} do not match the {ncase} exported cells")
    wrf_m = np.asarray([[rows[c][k]["exch_mx"] for k in range(1, nz + 1)] for c in range(1, ncase + 1)])
    wrf_h = np.asarray([[rows[c][k]["exch_hx"] for k in range(1, nz + 1)] for c in range(1, ncase + 1)])
    wrf_du = np.asarray([[rows[c][k]["utnp"] for k in range(1, nz + 1)] for c in range(1, ncase + 1)])
    wrf_dv = np.asarray([[rows[c][k]["vtnp"] for k in range(1, nz + 1)] for c in range(1, ncase + 1)])
    ker_m = np.asarray(kernel["exch_m"])
    ker_h = np.asarray(kernel["exch_h"])
    ker_du = np.asarray(kernel["du"])
    ker_dv = np.asarray(kernel["dv"])
    lo, hi = (0, nz - 1) if levels is None else (int(levels[0]), int(levels[1]))
    per_level = []
    for k in range(nz):
        ulp_m = _ulp_distance(ker_m[:, k], wrf_m[:, k])
        rel_m = np.abs(ker_m[:, k] - wrf_m[:, k]) / np.maximum(np.abs(wrf_m[:, k]), XKZMINM)
        per_level.append({
            "level": k,
            "p_full_mean_pa": float(np.mean(np.asarray(kernel["p_full_pa"])[:, k])),
            "wrf_exch_m_mean": float(np.mean(wrf_m[:, k])),
            "kernel_exch_m_mean": float(np.mean(ker_m[:, k])),
            "wrf_exch_m_max": float(np.max(wrf_m[:, k])),
            "kernel_exch_m_max": float(np.max(ker_m[:, k])),
            "exch_m_max_ulp": int(np.max(ulp_m)),
            "exch_m_max_rel": float(np.max(rel_m)),
            "exch_h_max_ulp": int(np.max(_ulp_distance(ker_h[:, k], wrf_h[:, k]))),
            "du_max_abs_diff": float(np.max(np.abs(ker_du[:, k] - wrf_du[:, k]))),
            "dv_max_abs_diff": float(np.max(np.abs(ker_dv[:, k] - wrf_dv[:, k]))),
            "wrf_du_rms": float(np.sqrt(np.mean(wrf_du[:, k] ** 2))),
        })
    band = per_level[lo:hi + 1]
    return {
        "schema": SCHEMA + "/wrf-reference",
        "cells": ncase,
        "mixing_length_mode_of_kernel": kernel["mixing_length_mode"],
        "levels_of_interest": [lo, hi],
        "summary": {
            "exch_m_max_ulp_levels_of_interest": int(max(r["exch_m_max_ulp"] for r in band)),
            "exch_m_max_rel_levels_of_interest": float(max(r["exch_m_max_rel"] for r in band)),
            "exch_m_max_ulp_all_levels": int(max(r["exch_m_max_ulp"] for r in per_level)),
            "du_max_abs_diff_all_levels": float(max(r["du_max_abs_diff"] for r in per_level)),
            "wrf_exch_m_mean_levels_of_interest": [r["wrf_exch_m_mean"] for r in band],
            "kernel_exch_m_mean_levels_of_interest": [r["kernel_exch_m_mean"] for r in band],
        },
        "per_level": per_level,
        "definition": "WRF v4.6.1 bl_ysu_run (byte-unmodified, -O0, ctopo absent) driven on the exported cells by tools/ysu_wrf461_oracle/run_bl_ysu_columns.F90; distances are the kernel's float32 word against WRF's",
    }


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def _run_column(backend: str, column: dict[str, np.ndarray], surface: dict[str, float], dt: float, mode: str) -> dict[str, np.ndarray]:
    """One column through the float64 mirror (``numpy``) or the CUDA
    kernel (``cupy``); outputs as numpy."""
    if backend == "numpy":
        from woof.globe.core.npref import np_ysu_column

        out = np_ysu_column(
            column["u"], column["v"], column["theta"], column["qv"], column["qc"], column["qi"],
            column["p"], column["p_interface"], column["exner"], column["dz"],
            psfc=surface["psfc"], znt=surface["znt"], ust=surface["ust"], hfx=surface["hfx"],
            qfx=surface["qfx"], wspd=surface["wspd"], br=surface["br"], psim=surface["psim"],
            psih=surface["psih"], xland=surface["xland"], u10=surface["u10"], v10=surface["v10"],
            dt=dt, ysu_topdown_pblmix=1, free_atmosphere_mixing_length=mode,
        )
        return {name: np.asarray(out[name], dtype=np.float64) for name in ("exch_m", "exch_h", "du", "dv")} | {"kpbl": int(out["kpbl"]), "hpbl": float(out["hpbl"]), "delta": float(out["delta"])}
    if backend == "cupy":
        import cupy as cp

        from woof.globe.core.ysu import launch_ysu

        def vol(a):
            return cp.ascontiguousarray(cp.asarray(np.asarray(a, dtype=np.float32)[:, None, None]))

        def plane(x):
            return cp.full((1, 1), np.float32(x), dtype=cp.float32)

        out = launch_ysu(
            vol(column["u"]), vol(column["v"]), vol(column["theta"]), vol(column["qv"]), vol(column["qc"]),
            vol(column["qi"]), vol(column["p"]), vol(column["p_interface"]), vol(column["exner"]), vol(column["dz"]),
            psfc=plane(surface["psfc"]), znt=plane(surface["znt"]), ust=plane(surface["ust"]), hfx=plane(surface["hfx"]),
            qfx=plane(surface["qfx"]), wspd=plane(surface["wspd"]), br=plane(surface["br"]), psim=plane(surface["psim"]),
            psih=plane(surface["psih"]), xland=plane(surface["xland"]), u10=plane(surface["u10"]), v10=plane(surface["v10"]),
            dt=dt, ysu_topdown_pblmix=1, free_atmosphere_mixing_length=mode,
        )
        host = {name: cp.asnumpy(out[name])[:, 0, 0].astype(np.float64) for name in ("exch_m", "exch_h", "du", "dv")}
        host["kpbl"] = int(cp.asnumpy(out["kpbl"])[0, 0])
        host["hpbl"] = float(cp.asnumpy(out["hpbl"])[0, 0])
        host["delta"] = float(cp.asnumpy(out["delta"])[0, 0])
        return host
    raise ValueError(f"backend must be numpy or cupy, got {backend!r}")


def synthetic_column(*, nz: int = 40, dz_m: float = 1500.0, shear_per_s: float = 0.0,
                     ri: float | None = None, jet_levels: tuple[int, int] = (10, 22),
                     dthv_per_layer_k: float = 3.0, theta0_k: float = 300.0) -> dict[str, np.ndarray]:
    """A dry column of ``nz`` layers ``dz_m`` thick, ``dthv_per_layer_k``
    of potential temperature per layer above the second level, with a
    linear shear ``shear_per_s`` (in u) between the full levels
    ``jet_levels``.  With ``ri`` given, the potential temperature inside
    the jet is built layer by layer so that every interface there carries
    the gradient Richardson number ``ri`` EXACTLY under the scheme's own
    definition ``g / thv_mean * (dthv / dz) / shear^2`` (``theta[k+1] =
    theta[k] (1 + c/2) / (1 - c/2)`` with ``c = ri shear^2 dz / g``);
    with ``ri`` None the jet keeps the background stratification (so a
    zero shear with ``dthv_per_layer_k = 0`` inside the jet plants a
    neutral, shear-free layer)."""
    dz = np.full(nz, dz_m, dtype=np.float64)
    zq = np.concatenate([[0.0], np.cumsum(dz)])
    za = 0.5 * (zq[:-1] + zq[1:])
    lo, hi = jet_levels
    theta = np.empty(nz, dtype=np.float64)
    theta[:2] = theta0_k
    for k in range(2, nz):
        if ri is not None and lo <= k - 1 < hi:
            c = ri * shear_per_s * shear_per_s * dz_m / G
            theta[k] = theta[k - 1] * (1.0 + 0.5 * c) / (1.0 - 0.5 * c)
        else:
            theta[k] = theta[k - 1] + dthv_per_layer_k
    u = np.zeros(nz)
    for k in range(nz):
        if lo <= k <= hi:
            u[k] = shear_per_s * (za[k] - za[lo])
        elif k > hi:
            u[k] = shear_per_s * (za[hi] - za[lo])
    v = np.zeros(nz)
    zeros = np.zeros(nz)
    psfc = 100_000.0
    p_interface = psfc * np.exp(-zq / 8500.0)
    p = 0.5 * (p_interface[:-1] + p_interface[1:])
    exner = (p / 100_000.0) ** (RD / CP)
    return {"u": u, "v": v, "theta": theta, "qv": zeros.copy(), "qc": zeros.copy(), "qi": zeros.copy(),
            "p": p, "p_interface": p_interface, "exner": exner, "dz": dz, "za": za, "zq": zq}


STABLE_SURFACE = {
    "psfc": 100_000.0, "znt": 0.1, "ust": 0.2, "hfx": -10.0, "qfx": 0.0, "wspd": 3.0, "br": 0.3,
    "psim": 6.5, "psih": 8.5, "xland": 1.0, "u10": 0.0, "v10": 0.0,
}

#: Planted (Richardson number, shear 1/s) pairs of the jet family.
PLANTED_JETS = ((0.25, 0.02), (1.0, 0.01), (4.0, 0.005), (-0.1, 0.01))


def _column_terms(col, out, mode):
    return free_atmosphere_terms(col["u"], col["v"], col["theta"], col["qv"], col["qc"], col["qi"],
                                 col["exner"], col["dz"], out["kpbl"], modes=(mode,))


def _column_zone(col, out):
    return entrainment_zone(col["dz"], out["hpbl"], out["delta"], out["kpbl"])


def calibrate(backend: str = "numpy", *, dt: float = 50.0) -> dict[str, object]:
    """The synthetic families in both directions, on ``backend``:
    ``planted_jet`` (a known Ri and shear read back, the kernel's K
    against the formula's under both length modes), ``zero_shear``
    (stable: exactly the floor, no tendency; neutral layer: the floor
    plus the 1e-9 shear-floor leak ``rl^2 sqrt(1e-9)`` and nothing
    else)."""
    families: dict[str, object] = {}
    jets = []
    dz_m = 1500.0
    k_face = np.arange(12, 20)                          # interfaces 13..20, inside the jet
    for ri_target, shear in PLANTED_JETS:
        col = synthetic_column(dz_m=dz_m, shear_per_s=shear, ri=ri_target)
        for mode in MODES:
            out = _run_column(backend, col, STABLE_SURFACE, dt, mode)
            terms = _column_terms(col, out, mode)
            zone = _column_zone(col, out)
            if out["kpbl"] > 12 or np.any(zone[k_face]):
                raise AssertionError(f"the planted jet is not free atmosphere: kpbl {out['kpbl']}, zone {zone[k_face]}")
            read_ri = terms["ri"][k_face]
            formula = terms[f"xkzm_{mode}"][k_face]
            kernel_k = out["exch_m"][k_face + 1]
            expected_length = (min(dz_m, min(max(0.1 * dz_m, RLAM), 300.0)) if mode == "wrf-layer"
                               else min(dz_m, YSU_FIXED_ASYMPTOTIC_LENGTH_M))
            jets.append({
                "mode": mode, "planted_ri": ri_target, "planted_shear_per_s": shear, "dz_m": dz_m,
                "read_ri": [float(x) for x in read_ri],
                "read_ri_max_rel_error": float(np.max(np.abs(read_ri - ri_target) / abs(ri_target))),
                "read_shear_max_rel_error": float(np.max(np.abs(terms["shear"][k_face] - shear) / shear)),
                "read_rlamdz_m": float(terms[f"rlamdz_{mode}"][k_face][0]), "expected_rlamdz_m": float(expected_length),
                "kernel_exch_m_m2_s": [float(x) for x in kernel_k],
                "formula_exch_m_m2_s": [float(x) for x in formula],
                "max_rel_error_kernel_vs_formula": float(np.max(np.abs(kernel_k - formula) / np.abs(formula))),
                "kpbl": int(out["kpbl"]),
            })
    families["planted_jet"] = jets
    zero = []
    for dz_m in (500.0, 1500.0):
        col = synthetic_column(dz_m=dz_m, shear_per_s=0.0)
        for mode in MODES:
            out = _run_column(backend, col, STABLE_SURFACE, dt, mode)
            terms = _column_terms(col, out, mode)
            faces = np.arange(12, 30)
            kernel_k = out["exch_m"][faces + 1]
            zero.append({
                "mode": mode, "dz_m": dz_m, "family": "stable",
                "kernel_exch_m_max_minus_floor": float(np.max(np.abs(kernel_k - XKZMINM))),
                "reads_exactly_the_floor": bool(np.all(np.asarray(kernel_k, dtype=np.float32) == np.float32(XKZMINM))),
                "shear_term_max_m2_s": float(np.max(terms[f"km_{mode}"][faces])),
                "du_max_abs": float(np.max(np.abs(out["du"]))),
                "kpbl": int(out["kpbl"]),
            })
        # A neutral, shear-free layer inside a stable free atmosphere: Ri = 0
        # there, so the K is the floor plus rl^2 sqrt(1e-9) from the shear
        # floor the scheme adds to ss, and nothing else.
        col = synthetic_column(dz_m=dz_m, shear_per_s=0.0, ri=0.0)
        for mode in MODES:
            out = _run_column(backend, col, STABLE_SURFACE, dt, mode)
            terms = _column_terms(col, out, mode)
            faces = np.arange(11, 21)
            kernel_k = out["exch_m"][faces + 1]
            leak = terms[f"rl_{mode}"][faces] ** 2 * math.sqrt(1.0e-9)
            zero.append({
                "mode": mode, "dz_m": dz_m, "family": "neutral_layer",
                "read_ri_max_abs": float(np.max(np.abs(terms["ri"][faces]))),
                "kernel_exch_m_minus_floor_mean": float(np.mean(kernel_k - XKZMINM)),
                "expected_shear_floor_leak_m2_s": float(np.mean(leak)),
                "max_rel_error_kernel_vs_floor_plus_leak": float(np.max(np.abs(kernel_k - (XKZMINM + leak)) / (XKZMINM + leak))),
                "du_max_abs": float(np.max(np.abs(out["du"]))),
                "kpbl": int(out["kpbl"]),
            })
    families["zero_shear"] = zero
    return {"schema": SCHEMA + "/calibration", "backend": backend, "dt_s": dt, "families": families}


def calibrate_band_projection(transform, vector, *, degree: int = 30, order: int = 5) -> dict[str, object]:
    """A single-degree rotational wind on one level, with its tendency
    equal to itself: the reading must land in that degree's band alone,
    equal to twice the level's kinetic energy over the sphere, and a zero
    tendency must read zero in every band."""
    xp = transform.backend.xp
    truncation = int(transform.truncation)
    nlat, nlon = transform.grid.shape
    if not 0 <= order <= degree <= truncation:
        raise ValueError("degree/order outside the truncation")
    coefficients = np.zeros((1, *transform.spectral_shape), dtype=np.complex128)
    coefficients[0, degree, order] = 1.0e-5 * (1.0 + 0.5j)
    vort = transform.project(coefficients)
    zero = xp.zeros_like(vort)
    u, v = vector.wind_from_vordiv(vort, zero)
    dp_g = np.ones((1, nlat, nlon), dtype=np.float64)
    weights = _region_weights(np.asarray(transform.grid.latitude_deg), np.asarray(transform.grid.quadrature_weights))
    reading = band_kinetic_tendency(vector, transform, u, v, u, v, dp_g, weights)
    uh = transform.backend.to_numpy(u)[0].astype(np.float64)
    vh = transform.backend.to_numpy(v)[0].astype(np.float64)
    ke = 0.5 * region_mean(uh * uh + vh * vh, weights["global"])
    per_band = np.asarray(reading["per_band"])[:, 0, 0]      # global, level 0
    bands = reading["band_edges"]
    home = [b for b, (lo, hi) in enumerate(bands) if lo <= degree <= hi][0]
    null = band_kinetic_tendency(vector, transform, u, v, xp.zeros_like(u), xp.zeros_like(v), dp_g, weights)
    return {
        "degree": degree, "order": order, "home_band": reading["bands"][home],
        "twice_kinetic_energy": 2.0 * ke,
        "read_in_home_band": float(per_band[home]),
        "home_band_rel_error": float(abs(per_band[home] - 2.0 * ke) / (2.0 * ke)),
        "max_abs_elsewhere": float(max(abs(per_band[b]) for b in range(len(bands)) if b != home)) if len(bands) > 1 else 0.0,
        "zero_tendency_max_abs": float(np.max(np.abs(np.asarray(null["per_band"])))),
        "total_vs_bands_rel": float(abs(np.asarray(reading["total"])[0, 0] - per_band.sum()) / (2.0 * ke)),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _table(reading: dict[str, object], levels: tuple[int, int]) -> str:
    lo, hi = levels
    nz = int(reading["shape"][0])
    lines = [
        f"YSU free-atmosphere probe: mode {reading['mixing_length_mode']}, step {reading['checkpoint_step']}, t = {reading['time_s'] / 3600.0:.1f} h; "
        f"levels {lo}..{hi} surface-first (top-first {nz - 1 - hi}..{nz - 1 - lo}; the ledger and the checkpoint count from the top)",
        "interface  p_int hPa  region        free%  dza m   shear 1/s   Ri med  rlamdz wrf/fixed  Km mean wrf-mode/fixed-mode  Km kernel mean  p99",
    ]
    for row in reading["per_interface"]:
        k = row["interface"] - 1
        if not lo <= k <= hi:
            continue
        for region in ("global", "north_20_70", "south_20_70"):
            e = row["regions"][region]
            lines.append(
                f"{row['interface']:>9d}  {row['p_interface_pa'] / 100.0:>9.1f}  {region:<12s}  {100 * e['free_atmosphere_fraction']:>5.1f}  "
                f"{e['dza_mean']:>6.0f}  {e['shear_mean']:>10.2e}  {e['ri_median']:>7.2f}  {e['rlamdz_wrf-layer_mean']:>6.1f}/{e['rlamdz_fixed_mean']:<5.1f}  "
                f"{e['xkzm_wrf-layer_mean']:>10.3f}/{e['xkzm_fixed_mean']:<10.3f}  {e['exch_m_mean']:>10.3f}  {e['exch_m_percentiles']['p99']:>8.2f}"
            )
    s = reading["summary"]
    t = s["kinetic_tendency_w_m2"]
    tc = s["transcription_check"]
    lines.append(
        f"transcription check: {tc['interfaces_checked']} interfaces, exch_m rel diff p50 {tc['exch_m_rel_diff_percentiles']['p50']:.1e} "
        f"p99 {tc['exch_m_rel_diff_percentiles']['p99']:.1e} max {tc['exch_m_max_rel_diff']:.2e}; "
        f"above 1e-3: {tc['exch_m_interfaces_above_1e-3_rel']}, above 1e-2: {tc['exch_m_interfaces_above_1e-2_rel']}"
    )
    lines.append("kinetic tendency W/m2, levels %d..%d: %s" % (lo, hi, ", ".join(f"{r} {v:+.4f} (free {f:+.4f})" for r, v, f in zip(t["regions"], t["levels_of_interest_total"], t["levels_of_interest_free"]))))
    lines.append("column: %s" % ", ".join(f"{r} {v:+.4f} (free {f:+.4f}, pbl {p:+.4f})" for r, v, f, p in zip(t["regions"], t["column_total"], t["column_free"], t["column_pbl"])))
    for b, label in enumerate(reading["per_band"]["bands"]):
        lines.append(f"  band {label:<8s}: " + ", ".join(f"{r} {v:+.4f}" for r, v in zip(t["regions"], t["levels_of_interest_per_band"][b])))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe", help="capture the first YSU call of a checkpoint restart and read it")
    probe.add_argument("--config", required=True)
    probe.add_argument("--checkpoint", required=True)
    probe.add_argument("--out", required=True)
    probe.add_argument("--levels", nargs=2, type=int, default=(12, 18), metavar=("LO", "HI"))
    probe.add_argument("--export-columns", default=None, metavar="PREFIX")
    probe.add_argument("--n-columns", type=int, default=48)
    cal = sub.add_parser("calibrate", help="the synthetic families on the mirror or the kernel")
    cal.add_argument("--out", required=True)
    cal.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    cmp_ = sub.add_parser("compare-wrf", help="the Fortran reference's exch_mx beside the kernel's")
    cmp_.add_argument("--export", required=True, metavar="PREFIX")
    cmp_.add_argument("--wrf-levels", required=True)
    cmp_.add_argument("--out", required=True)
    cmp_.add_argument("--levels", nargs=2, type=int, default=(12, 18))
    args = parser.parse_args(argv)
    if args.command == "probe":
        captured = capture_first_pbl_call(args.config, args.checkpoint)
        reading = read_capture(captured, levels=tuple(args.levels))
        if args.export_columns:
            cells = select_columns(captured, int(args.n_columns))
            reading["export"] = export_columns(captured, args.export_columns, cells)
        Path(args.out).write_text(json.dumps(reading, indent=1), encoding="utf-8")
        print(_table(reading, tuple(args.levels)))
        print(f"wrote {args.out}")
        return 0
    if args.command == "calibrate":
        result = calibrate(args.backend)
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
        for row in result["families"]["planted_jet"]:
            print(f"jet {row['mode']:<9s} Ri {row['planted_ri']:>5.2f}: read Ri rel err {row['read_ri_max_rel_error']:.2e}, shear rel err {row['read_shear_max_rel_error']:.2e}, rlamdz {row['read_rlamdz_m']:.1f} (expected {row['expected_rlamdz_m']:.1f}), kernel vs formula {row['max_rel_error_kernel_vs_formula']:.2e}, K {row['kernel_exch_m_m2_s'][0]:.4f}")
        for row in result["families"]["zero_shear"]:
            if row["family"] == "neutral_layer":
                print(f"neutral layer {row['mode']:<9s} dz {row['dz_m']:.0f}: read Ri {row['read_ri_max_abs']:.1e}, K - floor {row['kernel_exch_m_minus_floor_mean']:.4e} (leak {row['expected_shear_floor_leak_m2_s']:.4e}), rel err {row['max_rel_error_kernel_vs_floor_plus_leak']:.2e}, |du| {row['du_max_abs']:.2e}")
            else:
                print(f"stable zero shear {row['mode']:<9s} dz {row['dz_m']:.0f}: floor exactly {row['reads_exactly_the_floor']}, shear term {row['shear_term_max_m2_s']:.2e}, |du| {row['du_max_abs']:.2e}")
        print(f"wrote {args.out}")
        return 0
    if args.command == "compare-wrf":
        result = compare_wrf_reference(Path(args.export).with_name(Path(args.export).name + "-kernel.json"), args.wrf_levels, levels=tuple(args.levels))
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
        s = result["summary"]
        print(f"WRF reference on {result['cells']} cells: exch_m max ULP {s['exch_m_max_ulp_levels_of_interest']} (levels {args.levels[0]}..{args.levels[1]}), {s['exch_m_max_ulp_all_levels']} all levels; du max abs diff {s['du_max_abs_diff_all_levels']:.2e}")
        for k, (w, m) in enumerate(zip(s["wrf_exch_m_mean_levels_of_interest"], s["kernel_exch_m_mean_levels_of_interest"])):
            print(f"  level {args.levels[0] + k}: WRF Km mean {w:.4f}  kernel {m:.4f}")
        print(f"wrote {args.out}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
