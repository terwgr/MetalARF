"""The ABI clear-sky fast forward model: its coefficient table and its trainer.

The operator the ensemble filter runs is Rust (``rw_goes forward``, in
``tools/rustwx/crates/rw-goes/src/forward.rs``): it takes the model
columns of a ``gpuwm-da.abi-columns.v2`` stream, evaluates per-layer
optical depths from a coefficient table, marches the emission radiative
transfer, converts through the instrument's own band-corrected Planck
relation and writes brightness temperatures with finite-difference
Jacobians.  This module is the table's trainer and its Python oracle:

* the **trainer** fits, per model layer and per band, the layer nadir
  optical depth CRTM (the numerical reference, ``abi_reference``) computed
  on the analysed columns of the case, as a function of named per-layer
  features (the layer's vapor path, its temperature and pressure, the
  slant vapor path above it, its neighbours' vapor) in one of two forms:
  ``linear`` (the window band: dry and continuum terms add) or
  ``two_term`` (the water-vapor band: ``exp(wet polynomial) + dry``,
  fitted by damped Gauss-Newton on the log residual so the emitting layers
  are fitted to relative accuracy);
* the **oracle** evaluates a table in numpy exactly as the Rust operator
  does, so a test can pin the Rust to it bit for bit on a small column
  set, and the validation numbers a table carries come from it.

The table is tied to the vertical coordinate it was trained on (per-layer
coefficients); it carries the coordinate and the Rust refuses a column set
on another.  The Planck constants are the instrument's own, read from the
Level 1b granules (``fk1, fk2, bc1, bc2``); the transmittance model is
trained on CRTM with the GOES-16 ABI coefficients (the closest public
set), and the difference to the GOES-18 set is recorded as the sensor term
(0.04 K in band 13, 0.2 K in band 8 on this case).

Nothing here is a data path of the shipped system: the trainer runs once
per coordinate, the oracle runs in tests.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

FAST_MODEL_SCHEMA = "gpuwm-da.abi-fast-model.v1"

#: The feature vocabulary, by name.  The Rust operator computes exactly
#: these (``forward.rs::features``); a table may only name terms built
#: from them.
FEATURE_NAMES: tuple[str, ...] = (
    "one", "dp", "u", "e", "lnp", "Tn", "lu", "lUs", "Twn", "lP", "lsec", "lu_up", "lu_dn", "Tn_up", "Tn_dn",
)
FEATURE_DEFINITIONS: dict[str, str] = {
    "one": "1",
    "dp": "layer thickness, hPa (p_half below minus p_half above)",
    "u": "layer vapor path, q[g/kg] * dp[hPa]",
    "e": "vapor pressure proxy, q[g/kg] * p_full[hPa] * 1.608e-3 (hPa)",
    "lnp": "ln(p_full[hPa])",
    "Tn": "(T[K] - 250) / 50",
    "lu": "ln(max(u, u_floor))",
    "lUs": "ln(max(U_mid, u_floor) * sec(zenith)), U_mid the vapor path from the top of the atmosphere to the layer middle",
    "Twn": "(Tw - 250) / 50, Tw the vapor-path-weighted temperature from the top to the layer middle",
    "lP": "ln(max(Pw, 1e-3)), Pw the vapor-path-weighted pressure from the top to the layer middle (hPa)",
    "lsec": "ln(sec(zenith))",
    "lu_up": "lu of the layer above (the top layer repeats its own)",
    "lu_dn": "lu of the layer below (the bottom layer repeats its own)",
    "Tn_up": "Tn of the layer above",
    "Tn_dn": "Tn of the layer below",
}
FEATURE_CONSTANTS = {"t_ref_k": 250.0, "t_scale_k": 50.0, "u_floor": 1.0e-8, "e_factor": 1.608e-3, "pw_floor_hpa": 1.0e-3}

#: The window band (13): dry (pressure, temperature) and vapor (continuum,
#: self and foreign) terms add; linear least squares per layer.
LINEAR_TERMS_WINDOW: tuple[tuple[str, ...], ...] = (
    ("dp",), ("dp", "Tn"), ("dp", "Tn", "Tn"), ("u",), ("u", "e"), ("u", "e", "Tn"), ("u", "lnp"), ("u", "Tn"),
    ("u", "Tn", "Tn"), ("u", "u"), ("u", "e", "e"),
)
#: The water-vapor band (8): ``exp(wet) + dry``.
WET_TERMS: tuple[tuple[str, ...], ...] = (
    ("one",), ("lu",), ("lu", "lu"), ("lu", "lu", "lu"), ("lnp",), ("Tn",), ("lu", "Tn"), ("lu", "lnp"),
    ("Tn", "Tn"), ("lUs",), ("lUs", "lUs"), ("lu", "lUs"), ("Twn",), ("Twn", "lu"), ("lP",), ("lUs", "Tn"),
    ("lsec",), ("lsec", "lu"), ("lsec", "lsec"), ("lsec", "lUs"), ("lUs", "lUs", "lUs"), ("lsec", "Tn"),
    ("lnp", "lnp"), ("Tn", "Tn", "Tn"), ("lu", "Tn", "Tn"),
    ("lu_up",), ("lu_dn",), ("lu_up", "lu"), ("lu_dn", "lu"), ("lu_up", "lu_up"), ("lu_dn", "lu_dn"), ("Tn_up",), ("Tn_dn",),
    ("lu_up", "lsec"), ("lu_dn", "lsec"),
)
DRY_TERMS: tuple[tuple[str, ...], ...] = (("dp",), ("dp", "Tn"), ("dp", "Tn", "Tn"), ("dp", "lnp"))

#: Finite-difference steps the Rust Jacobians use (and the oracle).
JACOBIAN_STEPS = {"temperature_k": 0.1, "vapor_relative": 0.01, "skin_k": 0.1}


class FastModelError(ValueError):
    """The table cannot be built or applied as asked.  Names the breakage."""


# ---------------------------------------------------------------------------
# features and terms
# ---------------------------------------------------------------------------

def features(temperature_k: np.ndarray, q_gkg: np.ndarray, p_half_hpa: np.ndarray, p_full_hpa: np.ndarray,
             zenith_deg: np.ndarray) -> dict[str, np.ndarray]:
    """Every named feature, ``(n, nlay)``, top of atmosphere first."""
    c = FEATURE_CONSTANTS
    T = np.asarray(temperature_k, dtype=np.float64)
    q = np.asarray(q_gkg, dtype=np.float64)
    p_half = np.asarray(p_half_hpa, dtype=np.float64)
    p_full = np.asarray(p_full_hpa, dtype=np.float64)
    sec = 1.0 / np.cos(np.radians(np.asarray(zenith_deg, dtype=np.float64)))
    dp = p_half[:, 1:] - p_half[:, :-1]
    u = q * dp
    lnp = np.log(p_full)
    Tn = (T - c["t_ref_k"]) / c["t_scale_k"]
    lu = np.log(np.maximum(u, c["u_floor"]))
    cum = np.cumsum(u, axis=1)
    U_mid = np.maximum(cum - 0.5 * u, c["u_floor"])
    lUs = np.log(U_mid * sec[:, None])
    Tw = (np.cumsum(u * T, axis=1) - 0.5 * u * T) / U_mid
    Twn = (Tw - c["t_ref_k"]) / c["t_scale_k"]
    Pw = (np.cumsum(u * p_full, axis=1) - 0.5 * u * p_full) / U_mid
    lP = np.log(np.maximum(Pw, c["pw_floor_hpa"]))
    lsec = np.log(sec)[:, None] * np.ones_like(u)
    e = q * p_full * c["e_factor"]
    lu_up = np.concatenate([lu[:, :1], lu[:, :-1]], axis=1)
    lu_dn = np.concatenate([lu[:, 1:], lu[:, -1:]], axis=1)
    Tn_up = np.concatenate([Tn[:, :1], Tn[:, :-1]], axis=1)
    Tn_dn = np.concatenate([Tn[:, 1:], Tn[:, -1:]], axis=1)
    return {"one": np.ones_like(u), "dp": dp, "u": u, "e": e, "lnp": lnp, "Tn": Tn, "lu": lu, "lUs": lUs, "Twn": Twn,
            "lP": lP, "lsec": lsec, "lu_up": lu_up, "lu_dn": lu_dn, "Tn_up": Tn_up, "Tn_dn": Tn_dn}


def clip_features(feat: dict[str, np.ndarray], clip: dict[str, dict[str, list[float]]]) -> dict[str, np.ndarray]:
    """Hold every clipped feature inside the per-layer range the table
    recorded from its training set (the Rust does the same), so a column
    outside the training envelope extrapolates no further than its edge."""
    out = dict(feat)
    for name, bounds in clip.items():
        lo = np.asarray(bounds["min"], dtype=np.float64)[None, :]
        hi = np.asarray(bounds["max"], dtype=np.float64)[None, :]
        out[name] = np.minimum(np.maximum(feat[name], lo), hi)
    return out


def design(feat: dict[str, np.ndarray], terms) -> np.ndarray:
    """``(n, nlay, n_terms)`` products of named features."""
    n, nlay = feat["one"].shape
    X = np.ones((n, nlay, len(terms)))
    for j, term in enumerate(terms):
        for name in term:
            if name not in feat:
                raise FastModelError(f"term {term} names an unknown feature {name!r}; the vocabulary is {FEATURE_NAMES}")
            X[:, :, j] *= feat[name]
    return X


# ---------------------------------------------------------------------------
# Planck and the emission march (the arithmetic the Rust carries)
# ---------------------------------------------------------------------------

def planck_radiance(planck: dict, temperature_k: np.ndarray) -> np.ndarray:
    """Band-corrected Planck radiance, the L1b convention: ``L = fk1 / (exp(fk2 / (bc1 + bc2 T)) - 1)``."""
    teff = planck["bc1"] + planck["bc2"] * np.asarray(temperature_k, dtype=np.float64)
    return planck["fk1"] / (np.exp(planck["fk2"] / teff) - 1.0)


def planck_temperature(planck: dict, radiance: np.ndarray) -> np.ndarray:
    """The inverse: ``T = (fk2 / ln(fk1 / L + 1) - bc1) / bc2``."""
    return (planck["fk2"] / np.log(planck["fk1"] / np.asarray(radiance, dtype=np.float64) + 1.0) - planck["bc1"]) / planck["bc2"]


def emission_radiance(planck: dict, od_slant: np.ndarray, temperature_k: np.ndarray, skin_k: np.ndarray,
                      emissivity: np.ndarray) -> np.ndarray:
    """The clear-sky emission march CRTM's solver takes (specular
    downwelling along the same slant path): upwelling layer emission plus
    the surface's emission and its reflection of the downwelling."""
    n = od_slant.shape[0]
    cum = np.cumsum(od_slant, axis=1)
    t_top = np.concatenate([np.ones((n, 1)), np.exp(-cum)], axis=1)            # top of layer k to space
    B = planck_radiance(planck, temperature_k)
    up = np.sum(B * (t_top[:, :-1] - t_top[:, 1:]), axis=1)
    t_sfc = t_top[:, -1]
    rev = np.cumsum(od_slant[:, ::-1], axis=1)[:, ::-1]                          # layer k to the surface
    t_bot_sfc = np.concatenate([np.exp(-rev[:, 1:]), np.ones((n, 1))], axis=1)
    down = np.sum(B * (t_bot_sfc - np.exp(-rev)), axis=1)
    return up + emissivity * planck_radiance(planck, skin_k) * t_sfc + (1.0 - emissivity) * down * t_sfc


# ---------------------------------------------------------------------------
# the table: evaluation (the oracle)
# ---------------------------------------------------------------------------

def layer_optical_depth(band_table: dict, feat: dict[str, np.ndarray]) -> np.ndarray:
    """Nadir layer optical depth ``(n, nlay)`` from a band's table."""
    feat = clip_features(feat, band_table.get("clip", {}))
    layers = band_table["layers"]
    nlay = len(layers)
    if feat["one"].shape[1] != nlay:
        raise FastModelError(f"the columns carry {feat['one'].shape[1]} layers, the table {nlay}")
    if band_table["form"] == "linear":
        X = design(feat, [tuple(t) for t in band_table["terms_linear"]])
        coef = np.asarray([row["linear"] for row in layers])                    # (nlay, nt)
        return np.maximum(np.einsum("nkt,kt->nk", X, coef), 0.0)
    if band_table["form"] == "two_term":
        Xw = design(feat, [tuple(t) for t in band_table["terms_wet"]])
        Xd = design(feat, [tuple(t) for t in band_table["terms_dry"]])
        w = np.asarray([row["wet"] for row in layers])
        d = np.asarray([row["dry"] for row in layers])
        cap = np.asarray([row["ln_od_max"] for row in layers])[None, :]
        wet = np.exp(np.minimum(np.einsum("nkt,kt->nk", Xw, w), cap))
        dry = np.maximum(np.einsum("nkt,kt->nk", Xd, d), 0.0)
        return wet + dry
    raise FastModelError(f"unknown table form {band_table['form']!r}")


def evaluate(table: dict, band: int, columns: dict, *, emissivity: np.ndarray | None = None) -> dict:
    """Brightness temperature and the pieces, exactly as ``rw_goes forward``
    computes them.  ``columns`` carries ``temperature_k, q_gkg, p_half_hpa,
    p_full_hpa, zenith, skin_k`` and, for the table's own emissivity,
    ``land_fraction, landuse_category, sea_ice_fraction, snowh_m``."""
    bt_table = table["bands"][str(int(band))]
    feat = features(columns["temperature_k"], columns["q_gkg"], columns["p_half_hpa"], columns["p_full_hpa"], columns["zenith"])
    od = layer_optical_depth(bt_table, feat)
    sec = 1.0 / np.cos(np.radians(np.asarray(columns["zenith"], dtype=np.float64)))
    if emissivity is None:
        emissivity = table_emissivity(bt_table["emissivity"], columns)
    rad = emission_radiance(bt_table["planck"], od * sec[:, None], columns["temperature_k"], columns["skin_k"], emissivity)
    return {"bt": planck_temperature(bt_table["planck"], rad), "radiance": rad, "od_nadir": od, "emissivity": emissivity}


def table_emissivity(emis_table: dict, columns: dict) -> np.ndarray:
    """The table's surface emissivity per column: water by the reference's
    sea-water mean, land by IGBP class, snow-covered land and sea ice by
    their own values; fractions composed linearly."""
    land = np.clip(np.asarray(columns["land_fraction"], dtype=np.float64), 0.0, 1.0)
    water = 1.0 - land
    ice = np.clip(np.asarray(columns.get("sea_ice_fraction", np.zeros_like(land)), dtype=np.float64), 0.0, 1.0) * water
    water = water - ice
    cls = np.asarray(columns["landuse_category"], dtype=int)
    by_class = np.asarray(emis_table["land_by_igbp_class"], dtype=np.float64)
    land_e = by_class[np.clip(cls, 1, by_class.size) - 1]
    snow = np.asarray(columns.get("snowh_m", np.zeros_like(land)), dtype=np.float64) > emis_table.get("snow_depth_threshold_m", 0.01)
    snow = snow | (cls == 15)
    land_e = np.where(snow, emis_table["snow"], land_e)
    return water * emis_table["water"] + ice * emis_table["ice"] + land * land_e


def jacobians(table: dict, band: int, columns: dict, *, emissivity: np.ndarray | None = None) -> dict:
    """Central finite differences in layer temperature, layer vapor
    (relative) and skin, the steps of :data:`JACOBIAN_STEPS`; per g/kg for
    the vapor row.  The oracle for the Rust ``forward`` Jacobians."""
    base = evaluate(table, band, columns, emissivity=emissivity)
    emis = base["emissivity"]
    n, nlay = columns["temperature_k"].shape
    jac_t = np.zeros((n, nlay))
    jac_q = np.zeros((n, nlay))
    hT = JACOBIAN_STEPS["temperature_k"]
    rq = JACOBIAN_STEPS["vapor_relative"]
    for k in range(nlay):
        for sign in (+1, -1):
            c = dict(columns)
            T = np.array(columns["temperature_k"], dtype=np.float64, copy=True)
            T[:, k] += sign * hT
            c["temperature_k"] = T
            jac_t[:, k] += sign * evaluate(table, band, c, emissivity=emis)["bt"] / (2 * hT)
            c = dict(columns)
            q = np.array(columns["q_gkg"], dtype=np.float64, copy=True)
            dq = np.maximum(q[:, k] * rq, 1e-9)
            q[:, k] += sign * dq
            c["q_gkg"] = q
            jac_q[:, k] += sign * evaluate(table, band, c, emissivity=emis)["bt"] / (2 * dq)
    hS = JACOBIAN_STEPS["skin_k"]
    plus = dict(columns); plus["skin_k"] = np.asarray(columns["skin_k"]) + hS
    minus = dict(columns); minus["skin_k"] = np.asarray(columns["skin_k"]) - hS
    jac_skin = (evaluate(table, band, plus, emissivity=emis)["bt"] - evaluate(table, band, minus, emissivity=emis)["bt"]) / (2 * hS)
    return {"bt": base["bt"], "jac_t": jac_t, "jac_q": jac_q, "jac_tskin": jac_skin, "emissivity": emis}


# ---------------------------------------------------------------------------
# the trainer
# ---------------------------------------------------------------------------

def _fit_linear(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    scale = np.maximum(np.abs(X).max(axis=0), 1e-12)
    A = X / scale
    return np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]) * A.shape[0], A.T @ y) / scale


def _fit_two_term(y: np.ndarray, Xw: np.ndarray, Xd: np.ndarray, *, iterations: int, ridge: float,
                  damping: float = 1e-3) -> tuple[np.ndarray, np.ndarray, float]:
    """``OD = max(Xd d, 0) + exp(Xw w)`` by damped Gauss-Newton on ``ln OD``."""
    ly = np.log(np.maximum(y, 1e-12))
    sw = np.maximum(np.abs(Xw).max(axis=0), 1e-12)
    sd = np.maximum(np.abs(Xd).max(axis=0), 1e-12)
    A = Xw / sw
    w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]) * A.shape[0], A.T @ ly) / sw
    d = np.zeros(Xd.shape[1])
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for _ in range(iterations):
        wet = np.exp(np.minimum(Xw @ w, 50.0))
        dry = np.maximum(Xd @ d, 0.0)
        m = wet + dry
        r = ly - np.log(np.maximum(m, 1e-12))
        cost = float(np.mean(r * r))
        if best is None or cost < best[0]:
            best = (cost, w.copy(), d.copy())
        J = np.concatenate([(wet / m)[:, None] * Xw / sw, (1.0 / m)[:, None] * Xd / sd], axis=1)
        H = J.T @ J
        step = np.linalg.solve(H + damping * np.diag(np.diag(H)) + 1e-12 * np.eye(H.shape[0]), J.T @ r)
        w = w + step[:Xw.shape[1]] / sw
        d = d + step[Xw.shape[1]:] / sd
    assert best is not None
    return best[1], best[2], best[0]


def _clip_ranges(feat: dict[str, np.ndarray], train: np.ndarray, margin: float = 0.05) -> dict:
    """Per-layer min and max of every non-trivial feature over the
    training columns, widened by ``margin`` of the range."""
    clip: dict[str, dict[str, list[float]]] = {}
    for name in FEATURE_NAMES:
        if name in ("one", "dp", "u", "e", "lsec"):
            continue
        lo = feat[name][train].min(axis=0)
        hi = feat[name][train].max(axis=0)
        pad = margin * np.maximum(hi - lo, 1e-6)
        clip[name] = {"min": (lo - pad).tolist(), "max": (hi + pad).tolist()}
    return clip


def train_band(band: int, columns: dict, od_nadir: np.ndarray, *, train: np.ndarray, form: str,
               planck: dict, ridge: float = 1e-12, iterations: int = 12) -> dict:
    """One band's table: per-layer coefficients in ``form`` (``linear`` for
    the window band, ``two_term`` for the vapor band), the clip ranges and
    the validation on the held-out columns (BT of the oracle against the
    reference's BT recomputed through the same march, so the number is
    the transmittance model's alone)."""
    feat = features(columns["temperature_k"], columns["q_gkg"], columns["p_half_hpa"], columns["p_full_hpa"], columns["zenith"])
    clip = _clip_ranges(feat, train)
    feat_c = clip_features(feat, clip)
    n, nlay = od_nadir.shape
    layers: list[dict] = []
    if form == "linear":
        X = design(feat_c, LINEAR_TERMS_WINDOW)
        for k in range(nlay):
            c = _fit_linear(X[train, k, :], od_nadir[train, k], ridge)
            layers.append({"linear": c.tolist()})
        table = {"form": form, "terms_linear": [list(t) for t in LINEAR_TERMS_WINDOW]}
    elif form == "two_term":
        Xw = design(feat_c, WET_TERMS)
        Xd = design(feat_c, DRY_TERMS)
        for k in range(nlay):
            w, d, cost = _fit_two_term(od_nadir[train, k], Xw[train, k, :], Xd[train, k, :], iterations=iterations, ridge=ridge)
            layers.append({"wet": w.tolist(), "dry": d.tolist(), "ln_od_max": float(np.log(od_nadir[train, k].max() * 3.0 + 1e-12)),
                           "fit_log_rms": float(np.sqrt(cost))})
        table = {"form": form, "terms_wet": [list(t) for t in WET_TERMS], "terms_dry": [list(t) for t in DRY_TERMS]}
    else:
        raise FastModelError(f"form must be linear or two_term, got {form!r}")
    table.update({"band": int(band), "planck": dict(planck), "layers": layers, "clip": clip})
    return table


def validate_band(band_table: dict, columns: dict, od_nadir_reference: np.ndarray, *, test: np.ndarray,
                  emissivity: np.ndarray) -> dict:
    """The oracle's BT against the BT of the reference's own optical depths
    through the same march (isolating the transmittance model), on the
    held-out columns; plus the layer relative error where the band emits."""
    band = int(band_table["band"])
    table = {"bands": {str(band): band_table}}
    ours = evaluate(table, band, columns, emissivity=emissivity)
    sec = 1.0 / np.cos(np.radians(columns["zenith"]))
    ref_rad = emission_radiance(band_table["planck"], od_nadir_reference * sec[:, None], columns["temperature_k"],
                                columns["skin_k"], emissivity)
    ref_bt = planck_temperature(band_table["planck"], ref_rad)
    d = ours["bt"] - ref_bt
    rel = np.abs(ours["od_nadir"] - od_nadir_reference) / np.maximum(od_nadir_reference, 1e-9)
    land = np.asarray(columns["land_fraction"]) >= 0.5
    zen = np.asarray(columns["zenith"])
    by_zenith = {}
    for lo, hi in ((0, 20), (20, 40), (40, 50), (50, 60), (60, 70)):
        m = test & (zen >= lo) & (zen < hi)
        if m.any():
            by_zenith[f"{lo}_{hi}"] = {"n": int(m.sum()), "rms_k": float(np.sqrt(np.mean(d[m] ** 2))), "bias_k": float(d[m].mean())}
    return {
        "against": "the reference's own layer optical depths through the same emission march (the transmittance model alone)",
        "test_columns": int(test.sum()), "train_columns": int((~test).sum()),
        "test_rms_k": float(np.sqrt(np.mean(d[test] ** 2))), "test_bias_k": float(d[test].mean()),
        "test_p99_abs_k": float(np.percentile(np.abs(d[test]), 99)), "test_max_abs_k": float(np.abs(d[test]).max()),
        "train_rms_k": float(np.sqrt(np.mean(d[~test] ** 2))),
        "test_water_rms_k": float(np.sqrt(np.mean(d[test & ~land] ** 2))) if np.any(test & ~land) else None,
        "test_land_rms_k": float(np.sqrt(np.mean(d[test & land] ** 2))) if np.any(test & land) else None,
        "by_zenith": by_zenith,
        "median_relative_od_error_per_layer_test": np.median(rel[test], axis=0).round(4).tolist(),
    }


def emissivity_table_from_reference(reference_run: dict, band: int, columns: dict) -> dict:
    """The per-class emissivity the reference used on the case: sea water
    (its Nalli model's mean at the case's winds and angles), land by IGBP
    class (the class means; a class absent from the case takes the land
    mean), snow and sea ice from CRTM's NPOESS tables' means where the
    case had them, else stated defaults."""
    e = np.asarray(reference_run["emissivity"][int(band)], dtype=np.float64)
    land = np.asarray(columns["land_fraction"]) >= 0.5
    cls = np.asarray(columns["landuse_category"], dtype=int)
    snow = np.asarray(columns.get("snowh_m", np.zeros(cls.shape))) > 0.01
    ice = (np.asarray(columns.get("sea_ice_fraction", np.zeros(cls.shape))) >= 0.5) & ~land
    water = ~land & ~ice
    land_mean = float(e[land & ~snow].mean()) if np.any(land & ~snow) else 0.96
    by_class = []
    counts = []
    for c in range(1, 21):
        m = land & ~snow & (cls == c)
        counts.append(int(m.sum()))
        by_class.append(float(e[m].mean()) if m.sum() >= 5 else land_mean)
    return {
        "water": float(e[water].mean()) if water.any() else 0.98,
        "water_std": float(e[water].std()) if water.any() else None,
        "land_by_igbp_class": by_class,
        "land_class_counts": counts,
        "snow": float(e[land & snow].mean()) if np.any(land & snow) else 0.985,
        "ice": float(e[ice].mean()) if ice.any() else 0.98,
        "snow_depth_threshold_m": 0.01,
        "source": "the reference run's own surface emissivity on the case's columns (sea water: Nalli; land: IGBP table)",
    }


def coordinate_identity(a_half_pa, b_half) -> dict:
    a = np.asarray(a_half_pa, dtype=np.float64)
    b = np.asarray(b_half, dtype=np.float64)
    digest = hashlib.sha256(a.tobytes() + b.tobytes()).hexdigest()
    return {"nlev": int(a.size - 1), "a_half_pa": a.tolist(), "b_half": b.tolist(), "sha256": digest}


def write_table(path: str | os.PathLike, table: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = dict(table)
    table.setdefault("schema", FAST_MODEL_SCHEMA)
    table.setdefault("written_utc", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    path.write_text(json.dumps(table, indent=1, sort_keys=True), encoding="utf-8")
    return path


def read_table(path: str | os.PathLike) -> dict:
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    if table.get("schema") != FAST_MODEL_SCHEMA:
        raise FastModelError(f"{path} declares schema {table.get('schema')!r}, not {FAST_MODEL_SCHEMA}")
    return table


__all__ = [
    "DRY_TERMS", "FAST_MODEL_SCHEMA", "FEATURE_CONSTANTS", "FEATURE_DEFINITIONS", "FEATURE_NAMES", "JACOBIAN_STEPS",
    "LINEAR_TERMS_WINDOW", "WET_TERMS", "FastModelError", "clip_features", "coordinate_identity", "design",
    "emission_radiance", "emissivity_table_from_reference", "evaluate", "features", "jacobians", "layer_optical_depth",
    "planck_radiance", "planck_temperature", "read_table", "table_emissivity", "train_band", "validate_band", "write_table",
]
