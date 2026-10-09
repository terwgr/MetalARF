"""CRTM as the numerical reference for the ABI infrared operator.

The ABI operator lane measured SimSat's clear-sky brightness temperatures
against the real GOES-19 scan and failed the 1.5 K gate in both bands
(``docs/arwen-global-abi-operator.md``).  A failure against the instrument
does not say which term failed: the absorption, the source function, the
surface emissivity, or the analysed state itself.  This module puts a
reference forward model beside the operator on the SAME columns so the
gap is named by term instead of guessed:

* the analysed columns under the both-clear blocks of the colocation
  (block centres of ``rw_goes colocate``'s table) are read from the
  export tapes and the checkpoint's own surface planes and written as a
  ``gpuwm-da.abi-columns.v1`` stream;
* ``tools/crtm_reference/abi_crtm_reference`` (Fortran, CRTM v3.1.1 with
  JCSDA's public coefficients) runs the K-matrix model on them: the
  brightness temperature, the emissivity it used, the skin, temperature
  and vapor Jacobians, the layer optical depths;
* the scorer decomposes SimSat minus observed into SimSat minus CRTM at
  SimSat's own emissivity (the absorption and source-function term), CRTM
  at SimSat's emissivity minus CRTM with its own surface models (the
  emissivity term), and CRTM minus observed (the state and the reference's
  own residual), per surface class and zenith band, with the linear fit
  and the rmse after it that the operator gate reads.

The Jacobians are the operator's acceptance contract: a radiance is not a
one-level observation, and the ensemble filter localises it in model space
with the weighting function this module summarises (peak pressure, the
pressure band holding half the temperature sensitivity, the skin share).

This module is orchestration and arithmetic: the tape is read through the
Rust NetCDF bridge, the reference is CRTM, the Rust operator is measured,
nothing here is a data path of the shipped system.
"""
from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any

import numpy as np

COLUMNS_SCHEMA = "gpuwm-da.abi-columns.v2"
CRTM_OUTPUT_SCHEMA = "gpuwm-da.abi-crtm.v1"
REFERENCE_SCHEMA = "gpuwm-da.abi-reference-score.v1"
#: ``struct.unpack('<i', b'ABIC')`` and ``b'ABIR'``: what the Fortran driver checks.
COLUMNS_MAGIC = 1128874561
OUTPUT_MAGIC = 1380532801

#: The quality control the radiance stream applies to a block before it
#: becomes one observation of the filter, and therefore the population the
#: operator entries are graded on (one block, one row, one error; no pixel
#: weighting): at least this many both-clear pixels, at least this fraction
#: of the block's paired pixels both-clear (a block that is 2 percent clear
#: is a cloud edge, and on the case its residual is cloud, not the
#: operator), and a satellite zenith inside the admitted band.
STREAM_QC = {"minimum_pixels": 12, "minimum_clear_fraction": 0.5, "zenith_max_deg": 60.0}

#: CRTM climatology codes (CRTM_Atmosphere_Define).
TROPICAL, MIDLATITUDE_SUMMER, MIDLATITUDE_WINTER, SUBARCTIC_SUMMER, SUBARCTIC_WINTER, US_STANDARD = 1, 2, 3, 4, 5, 6

#: The IGBP land classes CRTM's ``IGBP.IRland.EmisCoeff`` table indexes, in the
#: order the model's MODIFIED_IGBP_MODIS_NOAH ``landuse_category`` uses (1-based).
IGBP_CLASSES: tuple[str, ...] = (
    "evergreen_needleleaf_forest", "evergreen_broadleaf_forest", "deciduous_needleleaf_forest",
    "deciduous_broadleaf_forest", "mixed_forests", "closed_shrublands", "open_shrublands",
    "woody_savannas", "savannas", "grasslands", "permanent_wetlands", "croplands",
    "urban_and_built-up", "cropland/natural_vegetation_mosaic", "snow_and_ice",
    "barren_or_sparsely_vegetated", "water", "wooded_tundra", "mixed_tundra", "bare_ground_tundra",
)

#: The US standard atmosphere ozone (ppmv) on CRTM's 92-layer test profile
#: (``test/mains/regression/k_matrix/test_ClearSky/Load_Atm_Data.inc``,
#: profile 1), interpolated in ln p onto the analysed layers; the analysis
#: carries no ozone and the two bands are nearly ozone-blind, so this is a
#: stated climatological fill, not a retrieved quantity.
_US_STANDARD_LAYER_HPA = np.array([
    0.838, 1.129, 1.484, 1.910, 2.416, 3.009, 3.696, 4.485, 5.385, 6.402, 7.545, 8.822, 10.240, 11.807,
    13.532, 15.423, 17.486, 19.730, 22.163, 24.793, 27.626, 30.671, 33.934, 37.425, 41.148, 45.113,
    49.326, 53.794, 58.524, 63.523, 68.797, 74.353, 80.198, 86.338, 92.778, 99.526, 106.586, 113.965,
    121.669, 129.703, 138.072, 146.781, 155.836, 165.241, 175.001, 185.121, 195.606, 206.459, 217.685,
    229.287, 241.270, 253.637, 266.392, 279.537, 293.077, 307.014, 321.351, 336.091, 351.236, 366.789,
    382.751, 399.126, 415.914, 433.118, 450.738, 468.777, 487.236, 506.115, 525.416, 545.139, 565.285,
    585.854, 606.847, 628.263, 650.104, 672.367, 695.054, 718.163, 741.693, 765.645, 790.017, 814.807,
    840.016, 865.640, 891.679, 918.130, 944.993, 972.264, 999.942, 1028.025, 1056.510, 1085.394])
_US_STANDARD_O3_PPMV = np.array([
    3.035, 3.943, 4.889, 5.812, 6.654, 7.308, 7.660, 7.745, 7.696, 7.573, 7.413, 7.246, 7.097, 6.959,
    6.797, 6.593, 6.359, 6.110, 5.860, 5.573, 5.253, 4.937, 4.625, 4.308, 3.986, 3.642, 3.261, 2.874,
    2.486, 2.102, 1.755, 1.450, 1.208, 1.087, 1.030, 1.005, 1.010, 1.028, 1.068, 1.109, 1.108, 1.071,
    0.9928, 0.8595, 0.7155, 0.5778, 0.4452, 0.3372, 0.2532, 0.1833, 0.1328, 0.09394, 0.06803, 0.05152,
    0.04569, 0.04855, 0.05461, 0.06398, 0.07205, 0.07839, 0.08256, 0.08401, 0.08412, 0.08353, 0.08269,
    0.08196, 0.08103, 0.07963, 0.07741, 0.07425, 0.07067, 0.06702, 0.06368, 0.06070, 0.05778, 0.05481,
    0.05181, 0.04920, 0.04700, 0.04478, 0.04207, 0.03771, 0.03012, 0.01941, 0.009076, 0.002980, 0.005117,
    0.01160, 0.01428, 0.01428, 0.01428, 0.01428])


class AbiReferenceError(ValueError):
    """The reference cannot be built as asked.  Always names the breakage."""


# ---------------------------------------------------------------------------
# blocks and columns
# ---------------------------------------------------------------------------

def read_block_tables(csv_by_band: dict[int, str | os.PathLike], *, zenith_max_deg: float = 70.0) -> dict:
    """The both-clear blocks of the colocation, joined across bands on the
    block index.  Returns arrays keyed by name plus ``obs_<band>`` and
    ``sim_<band>`` (the both-clear block means) and ``n_<band>``."""
    tables = {int(b): np.genfromtxt(p, delimiter=",", names=True) for b, p in csv_by_band.items()}
    bands = sorted(tables)
    first = tables[bands[0]]
    key = {(int(r["block_x"]), int(r["block_y"])): i for i, r in enumerate(first)}
    keep = np.ones(first.size, dtype=bool)
    index = {bands[0]: np.arange(first.size)}
    for band in bands[1:]:
        table = tables[band]
        other = {(int(r["block_x"]), int(r["block_y"])): i for i, r in enumerate(table)}
        idx = np.full(first.size, -1)
        for k, i in key.items():
            j = other.get(k)
            if j is not None:
                idx[i] = j
        keep &= idx >= 0
        index[band] = idx
    keep &= (first["n_both_clear"] > 0) & (first["zenith_mean_deg"] <= zenith_max_deg)
    for band in bands[1:]:
        j = index[band]
        both = np.where(keep, tables[band]["n_both_clear"][np.maximum(j, 0)], 0)
        if np.any(both[keep] != first["n_both_clear"][keep]):
            raise AbiReferenceError(
                f"band {band} and band {bands[0]} disagree on the both-clear count of a block; the two "
                "colocations were not made against one clear-sky mask")
    out = {
        "block_x": first["block_x"][keep].astype(int), "block_y": first["block_y"][keep].astype(int),
        "lat": first["lat_mean_deg"][keep].astype(float), "lon": first["lon_mean_deg"][keep].astype(float),
        "zenith": first["zenith_mean_deg"][keep].astype(float),
        "n_both_clear": first["n_both_clear"][keep].astype(int),
        "n_pair": first["n_pair"][keep].astype(int),
    }
    for band in bands:
        rows = tables[band][index[band][keep]]
        out[f"obs_{band}"] = rows["obs_mean_both_clear_k"].astype(float)
        out[f"sim_{band}"] = rows["sim_mean_both_clear_k"].astype(float)
    out["bands"] = bands
    return out


@dataclass
class Tape:
    path: str
    lat: np.ndarray      # (ny,)
    lon: np.ndarray      # (nx,)
    fields: dict[str, np.ndarray]

    def index_of(self, lat_deg: float, lon_deg: float) -> tuple[int, int] | None:
        dlat = self.lat[1] - self.lat[0]
        dlon = self.lon[1] - self.lon[0]
        i = int(round((lat_deg - self.lat[0]) / dlat))
        j = int(round((lon_deg - self.lon[0]) / dlon))
        if 0 <= i < self.lat.size and 0 <= j < self.lon.size:
            return i, j
        return None


_TAPE_3D = ("PB", "T", "QVAPOR")
_TAPE_2D = ("PSFC", "TSK", "HGT", "LANDMASK", "U10", "V10", "SEAICE", "SNOWH")


def read_tapes(paths) -> list[Tape]:
    """The export tapes through the Rust NetCDF bridge (whole fields; a
    tile is a few tens of megabytes)."""
    from woof.netcdf_bridge import open_dataset

    tapes: list[Tape] = []
    for path in paths:
        ds = open_dataset(path)

        def plane(name):
            arr = np.asarray(ds.variables[name][:])
            return arr[0] if arr.ndim in (3, 4) and arr.shape[0] == 1 else arr

        xlat = plane("XLAT")
        xlong = plane("XLONG")
        fields = {name: plane(name) for name in (*_TAPE_3D, *_TAPE_2D)}
        tapes.append(Tape(str(path), xlat[:, 0].astype(float), xlong[0, :].astype(float), fields))
    return tapes


def gaussian_grid(nlat: int, nlon: int) -> tuple[np.ndarray, np.ndarray]:
    """The model's Gaussian latitudes (south to north, degrees) and
    longitudes (from 0, degrees), as ``woof.globe.spectral.grid`` builds
    them."""
    mu, _ = np.polynomial.legendre.leggauss(int(nlat))
    return np.degrees(np.arcsin(mu)), np.arange(int(nlon)) * (360.0 / int(nlon))


def ozone_ppmv(layer_hpa: np.ndarray) -> np.ndarray:
    """US standard ozone at the layer pressures (ln p interpolation, the
    ends held)."""
    x = np.log(_US_STANDARD_LAYER_HPA)
    return np.interp(np.log(layer_hpa), x, _US_STANDARD_O3_PPMV)


def climatology_for(lat_deg: np.ndarray, month: int) -> np.ndarray:
    """CRTM's climatology code by latitude band and hemisphere season:
    used only for the layers CRTM adds above the model top."""
    lat = np.asarray(lat_deg, dtype=float)
    north_summer = month in (4, 5, 6, 7, 8, 9)
    out = np.full(lat.shape, TROPICAL, dtype=np.int32)
    mid = (np.abs(lat) >= 30.0) & (np.abs(lat) < 60.0)
    high = np.abs(lat) >= 60.0
    summer_here = np.where(lat >= 0.0, north_summer, not north_summer)
    out[mid] = np.where(summer_here[mid], MIDLATITUDE_SUMMER, MIDLATITUDE_WINTER)
    out[high] = np.where(summer_here[high], SUBARCTIC_SUMMER, SUBARCTIC_WINTER)
    return out


def build_columns(blocks: dict, tapes: list[Tape], *, a_half_pa: np.ndarray, b_half: np.ndarray,
                  checkpoint: str | os.PathLike, month: int, kappa: float, reference_pressure_pa: float,
                  theta_offset_k: float = 300.0, emissivity_by_band: dict[int, float] | None = None) -> dict:
    """The analysed columns under the blocks: the tape's atmosphere at the
    nearest tape point, the checkpoint's own surface planes at the nearest
    Gaussian node.  Returns the column arrays (top of atmosphere first),
    the block rows kept, and the consistency checks."""
    a_half = np.asarray(a_half_pa, dtype=np.float64)
    b_half = np.asarray(b_half, dtype=np.float64)
    nlev = a_half.size - 1
    n = blocks["lat"].size
    placed = np.zeros(n, dtype=bool)
    cols: dict[str, list] = {k: [] for k in ("p_full", "t", "q", "psfc", "tsk", "hgt", "landmask", "u10", "v10", "seaice", "snowh")}
    for k in range(n):
        for tape in tapes:
            ij = tape.index_of(blocks["lat"][k], blocks["lon"][k])
            if ij is None:
                continue
            i, j = ij
            f = tape.fields
            cols["p_full"].append(f["PB"][:, i, j].astype(np.float64))
            cols["t"].append(f["T"][:, i, j].astype(np.float64))
            cols["q"].append(f["QVAPOR"][:, i, j].astype(np.float64))
            for name, var in (("psfc", "PSFC"), ("tsk", "TSK"), ("hgt", "HGT"), ("landmask", "LANDMASK"),
                              ("u10", "U10"), ("v10", "V10"), ("seaice", "SEAICE"), ("snowh", "SNOWH")):
                cols[name].append(float(f[var][i, j]))
            placed[k] = True
            break
    if not placed.any():
        raise AbiReferenceError("no block centre lies on any tape")
    arrays = {k: np.asarray(v) for k, v in cols.items()}
    p_full_bottom_up = arrays["p_full"]          # (n, nlev), tape order: lowest layer first
    if p_full_bottom_up.shape[1] != nlev:
        raise AbiReferenceError(f"the tape carries {p_full_bottom_up.shape[1]} layers, the coordinate {nlev}")
    if not np.all(np.diff(p_full_bottom_up, axis=1) < 0):
        raise AbiReferenceError("the tape's PB is not monotonic bottom-up; the level order is not what the export writes")
    ps = arrays["psfc"]
    # half levels top-first from the model's own coordinate at the tape's surface pressure
    p_half = a_half[None, :] + b_half[None, :] * ps[:, None]           # (n, nlev+1), top first
    p_full_top_first = p_full_bottom_up[:, ::-1]
    p_full_from_half = np.sqrt(p_half[:, :-1] * p_half[:, 1:])
    coord_mismatch = float(np.max(np.abs(p_full_from_half - p_full_top_first) / p_full_top_first))
    theta = arrays["t"][:, ::-1] + theta_offset_k
    temperature = theta * (p_full_top_first / reference_pressure_pa) ** kappa
    q_gkg = np.maximum(arrays["q"][:, ::-1], 0.0) * 1000.0
    o3 = np.stack([ozone_ppmv(p_full_top_first[k] / 100.0) for k in range(placed.sum())])

    # the checkpoint's surface planes at the nearest Gaussian node
    z = np.load(checkpoint, allow_pickle=False)
    land_fraction = z["surface__land_fraction"]
    nlat, nlon = land_fraction.shape
    glat, glon = gaussian_grid(nlat, nlon)
    lat_k = blocks["lat"][placed]
    lon_k = np.mod(blocks["lon"][placed], 360.0)
    gi = np.argmin(np.abs(glat[None, :] - lat_k[:, None]), axis=1)
    gj = np.mod(np.round(lon_k / (360.0 / nlon)).astype(int), nlon)
    surface = {
        "land_fraction": land_fraction[gi, gj].astype(np.float64),
        "landuse_category": z["surface__landuse_category"][gi, gj].astype(np.int32),
        "surface_emissivity": z["surface__surface_emissivity"][gi, gj].astype(np.float64),
        "sea_ice_fraction": z["surface__sea_ice_fraction"][gi, gj].astype(np.float64),
        "skin_checkpoint_k": z["surface__surface_temperature_k"][gi, gj].astype(np.float64),
    }
    landmask_agreement = float(np.mean((surface["land_fraction"] >= 0.5) == (arrays["landmask"] >= 0.5)))
    skin_gap = surface["skin_checkpoint_k"] - arrays["tsk"]

    emissivity_by_band = dict(emissivity_by_band or {})
    bands = sorted(emissivity_by_band)
    emis_user = np.stack([np.full(placed.sum(), float(emissivity_by_band[b])) for b in bands], axis=1) if bands else np.zeros((placed.sum(), 0))
    return {
        "kept": placed,
        "lat": lat_k, "lon": blocks["lon"][placed], "zenith": blocks["zenith"][placed],
        "land_fraction": surface["land_fraction"], "landuse_category": surface["landuse_category"],
        "surface_emissivity": surface["surface_emissivity"], "sea_ice_fraction": surface["sea_ice_fraction"],
        "skin_k": arrays["tsk"], "psfc_hpa": ps / 100.0,
        "wind10_ms": np.hypot(arrays["u10"], arrays["v10"]), "snowh_m": arrays["snowh"],
        "landmask_tape": arrays["landmask"], "elevation_m": arrays["hgt"],
        "p_half_hpa": p_half / 100.0, "p_full_hpa": p_full_top_first / 100.0,
        "temperature_k": temperature, "q_gkg": q_gkg, "o3_ppmv": o3,
        "climatology": climatology_for(lat_k, month),
        "a_half_pa": a_half, "b_half": b_half,
        "emissivity_channels": bands, "emissivity_user": emis_user,
        "checks": {
            "blocks_asked": int(n), "blocks_placed": int(placed.sum()),
            "coordinate_mismatch_max_relative": coord_mismatch,
            "landmask_agreement_checkpoint_vs_tape": landmask_agreement,
            "skin_checkpoint_minus_tape_k": {"mean": float(skin_gap.mean()), "rms": float(np.sqrt(np.mean(skin_gap ** 2))),
                                             "max_abs": float(np.abs(skin_gap).max())},
            "layers": int(nlev),
            "top_half_level_hpa": float(p_half[:, 0].mean() / 100.0),
        },
    }


def write_columns(path: str | os.PathLike, columns: dict, *, provenance: dict | None = None) -> Path:
    """The ``gpuwm-da.abi-columns.v2`` stream the Fortran reference driver
    and the Rust operator read, and its JSON sidecar.  Layout
    (little-endian): int32 magic, version 2, n, nlay, n_user_chan; float64
    a_half_pa, b_half (each nlay+1, the model's vertical coordinate, top
    first); int32 user channels; float64 lat, lon, zenith, land_fraction,
    skin, psfc_hpa, wind10, snowh, seaice (each n); int32 climatology,
    land_type (each n); float64 p_half (n, nlay+1), p_full, t, q_gkg, o3
    (each n, nlay), top first; float64 emissivity (n, n_user_chan)."""
    path = Path(path)
    n = int(columns["lat"].size)
    nlay = int(columns["p_full_hpa"].shape[1])
    chans = [int(c) for c in columns["emissivity_channels"]]
    a_half = np.ascontiguousarray(columns["a_half_pa"], dtype="<f8")
    b_half = np.ascontiguousarray(columns["b_half"], dtype="<f8")
    if a_half.size != nlay + 1 or b_half.size != nlay + 1:
        raise AbiReferenceError(f"the coordinate has {a_half.size} / {b_half.size} half levels for {nlay} layers")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<5i", COLUMNS_MAGIC, 2, n, nlay, len(chans)))
        fh.write(a_half.tobytes())
        fh.write(b_half.tobytes())
        if chans:
            fh.write(np.asarray(chans, dtype="<i4").tobytes())
        for name in ("lat", "lon", "zenith", "land_fraction", "skin_k", "psfc_hpa", "wind10_ms", "snowh_m", "sea_ice_fraction"):
            fh.write(np.ascontiguousarray(columns[name], dtype="<f8").tobytes())
        fh.write(np.ascontiguousarray(columns["climatology"], dtype="<i4").tobytes())
        fh.write(np.ascontiguousarray(columns["landuse_category"], dtype="<i4").tobytes())
        for name in ("p_half_hpa", "p_full_hpa", "temperature_k", "q_gkg", "o3_ppmv"):
            fh.write(np.ascontiguousarray(columns[name], dtype="<f8").tobytes())
        if chans:
            fh.write(np.ascontiguousarray(columns["emissivity_user"], dtype="<f8").tobytes())
    sidecar = {
        "schema": COLUMNS_SCHEMA, "n": n, "layers": nlay, "emissivity_channels": chans,
        "coordinate_sha256": hashlib.sha256(a_half.tobytes() + b_half.tobytes()).hexdigest(),
        "emissivity_user_first_row": [float(v) for v in columns["emissivity_user"][0]] if chans and n else [],
        "checks": columns["checks"], "units": {
            "pressure": "hPa, top of atmosphere first", "temperature": "K", "water_vapor": "g/kg mass mixing ratio",
            "ozone": "ppmv (US standard atmosphere fill)", "wind10": "m/s", "snow_depth": "m",
            "land_type": "MODIFIED_IGBP_MODIS_NOAH category = CRTM IGBP.IRland table index",
        },
        "provenance": provenance or {},
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    Path(f"{path}.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_crtm_output(path: str | os.PathLike) -> dict:
    """The ``gpuwm-da.abi-crtm.v1`` stream the driver writes."""
    raw = Path(path).read_bytes()
    magic, version, n, nlay, nchan, emis_mode, n_bad = struct.unpack_from("<7i", raw, 0)
    if magic != OUTPUT_MAGIC or version != 1:
        raise AbiReferenceError(f"{path} is not a {CRTM_OUTPUT_SCHEMA} stream (magic {magic}, version {version})")
    offset = 28
    channels = np.frombuffer(raw, dtype="<i4", count=nchan, offset=offset).tolist()
    offset += 4 * nchan
    expected = offset + 8 * nchan * n * (5 + 3 * nlay)
    if len(raw) != expected:
        raise AbiReferenceError(
            f"{path} holds {len(raw)} bytes where its header ({nchan} channels, {n} columns, {nlay} layers) "
            f"says {expected}")

    def take(shape):
        nonlocal offset
        count = int(np.prod(shape))
        arr = np.frombuffer(raw, dtype="<f8", count=count, offset=offset).reshape(shape)
        offset += 8 * count
        return arr

    out: dict[str, Any] = {"schema": CRTM_OUTPUT_SCHEMA, "n": n, "layers": nlay, "channels": channels,
                           "emissivity_mode": emis_mode, "refused": n_bad}
    for name in ("bt", "emissivity", "radiance", "jac_tskin", "surface_planck"):
        arr = take((nchan, n))
        out[name] = {ch: arr[c] for c, ch in enumerate(channels)}
    for name in ("jac_t", "jac_q", "layer_od"):
        arr = take((nchan, n, nlay))
        out[name] = {ch: arr[c] for c, ch in enumerate(channels)}
    return out


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def paired_stats(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> dict:
    """``y`` against ``x`` (y minus x is the bias), weighted by ``w`` (pixel
    counts): n, means, bias, rmse, correlation, the least-squares ``x = a +
    b y`` (the gate's form, observed from simulated) and the rmse after it."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if w is None:
        w = np.ones(x.shape)
    w = np.asarray(w, dtype=np.float64)
    x, y, w = x[ok], y[ok], w[ok]
    n = int(x.size)
    if n == 0:
        return {"n": 0, "pixels": 0}
    d = y - x
    bias = float(np.average(d, weights=w))
    rmse = float(np.sqrt(np.average(d * d, weights=w)))
    row = {"n": n, "pixels": int(w.sum()), "mean_x_k": float(np.average(x, weights=w)),
           "mean_y_k": float(np.average(y, weights=w)), "bias_k": bias, "rmse_k": rmse,
           "rmse_debiased_k": float(math.sqrt(max(rmse * rmse - bias * bias, 0.0)))}
    if n >= 3 and np.average((y - row["mean_y_k"]) ** 2, weights=w) > 0:
        slope, intercept = np.polyfit(y, x, 1, w=np.sqrt(w))
        resid = x - (intercept + slope * y)
        sx = math.sqrt(np.average((x - row["mean_x_k"]) ** 2, weights=w))
        sy = math.sqrt(np.average((y - row["mean_y_k"]) ** 2, weights=w))
        row["correlation"] = float(np.average((x - row["mean_x_k"]) * (y - row["mean_y_k"]), weights=w) / (sx * sy)) if sx > 0 and sy > 0 else float("nan")
        row["fit_intercept_k"] = float(intercept)
        row["fit_slope"] = float(slope)
        row["rmse_after_k"] = float(np.sqrt(np.average(resid * resid, weights=w)))
    return row


def residual_shape(resid: np.ndarray, *, zenith: np.ndarray | None = None, lat: np.ndarray | None = None) -> dict:
    """The shape of a residual distribution beside its rms: skew, excess
    kurtosis, the fraction beyond two and three standard deviations (a
    Gaussian holds 4.55 and 0.27 percent), and the correlation with the
    zenith angle and latitude where given.  Reported, not gated: an
    observation error is one number and these say how far the residual is
    from the Gaussian that number stands for."""
    r = np.asarray(resid, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 3 or r.std() == 0:
        return {"n": int(r.size)}
    z = (r - r.mean()) / r.std()
    out = {"n": int(r.size), "skew": float(np.mean(z ** 3)), "excess_kurtosis": float(np.mean(z ** 4) - 3.0),
           "fraction_beyond_2_sigma": float(np.mean(np.abs(z) > 2.0)), "fraction_beyond_3_sigma": float(np.mean(np.abs(z) > 3.0)),
           "gaussian_beyond_3_sigma": 0.0027, "p1_k": float(np.percentile(r, 1)), "p99_k": float(np.percentile(r, 99))}
    for name, x in (("zenith", zenith), ("latitude", lat)):
        if x is not None:
            x = np.asarray(x, dtype=np.float64)
            if x.size == r.size and x.std() > 0:
                out[f"correlation_with_{name}"] = float(np.corrcoef(r, x)[0, 1])
    return out


def filter_facing_stats(obs: np.ndarray, sim: np.ndarray, *, n_both_clear: np.ndarray, n_pair: np.ndarray, zenith: np.ndarray,
                        lat: np.ndarray | None = None, qc: dict | None = None) -> dict:
    """The statistics of the blocks the stream hands the filter: the
    :data:`STREAM_QC` selection (pixels, clear fraction, zenith), each block
    one row with one error, no pixel weighting.  Carries the QC, the count
    it rejected, the paired statistics and the residual shape after the
    linear correction."""
    qc = dict(STREAM_QC, **(qc or {}))
    n_both = np.asarray(n_both_clear, dtype=np.float64)
    pairs = np.asarray(n_pair, dtype=np.float64)
    zen = np.asarray(zenith, dtype=np.float64)
    frac = np.where(pairs > 0, n_both / np.maximum(pairs, 1.0), 0.0)
    keep = (n_both >= qc["minimum_pixels"]) & (frac >= qc["minimum_clear_fraction"]) & (zen <= qc["zenith_max_deg"])
    row = paired_stats(np.asarray(obs)[keep], np.asarray(sim)[keep])
    row["qc"] = qc
    row["blocks_offered"] = int(n_both.size)
    row["blocks_rejected"] = {"below_minimum_pixels": int(np.sum(n_both < qc["minimum_pixels"])),
                              "below_minimum_clear_fraction": int(np.sum((n_both >= qc["minimum_pixels"]) & (frac < qc["minimum_clear_fraction"]))),
                              "beyond_zenith": int(np.sum((n_both >= qc["minimum_pixels"]) & (frac >= qc["minimum_clear_fraction"]) & (zen > qc["zenith_max_deg"])))}
    row["weighting"] = "none: one block is one observation with one error"
    if "fit_slope" in row:
        o = np.asarray(obs)[keep]
        s = np.asarray(sim)[keep]
        resid = o - (row["fit_intercept_k"] + row["fit_slope"] * s)
        row["residual_shape_after_correction"] = residual_shape(
            resid, zenith=zen[keep], lat=(np.asarray(lat)[keep] if lat is not None else None))
    return row


def jacobian_summary(jac_t: np.ndarray, jac_q: np.ndarray, jac_tskin: np.ndarray, p_half_hpa: np.ndarray,
                     p_full_hpa: np.ndarray) -> dict:
    """The weighting function the operator contract carries: the mean
    temperature Jacobian per unit ln p, its peak pressure and the pressure
    band holding the middle half of the sensitivity per column, the skin
    share, and the vapor Jacobian's peak."""
    dlnp = np.log(p_half_hpa[:, 1:] / p_half_hpa[:, :-1])          # (n, nlay)
    density = np.abs(jac_t) / dlnp                                   # K per K per unit ln p
    total_t = jac_t.sum(axis=1)
    peaks = p_full_hpa[np.arange(jac_t.shape[0]), np.argmax(density, axis=1)]
    cum = np.cumsum(np.abs(jac_t), axis=1)
    cum = cum / np.maximum(cum[:, -1:], 1e-30)
    lo = np.array([np.interp(0.25, cum[k], p_full_hpa[k]) for k in range(jac_t.shape[0])])
    hi = np.array([np.interp(0.75, cum[k], p_full_hpa[k]) for k in range(jac_t.shape[0])])
    q_peaks = p_full_hpa[np.arange(jac_q.shape[0]), np.argmax(np.abs(jac_q), axis=1)]
    mean_p = np.exp(np.mean(np.log(p_full_hpa), axis=0))
    return {
        "columns": int(jac_t.shape[0]),
        "mean_layer_pressure_hpa": mean_p.round(3).tolist(),
        "mean_temperature_jacobian_k_per_k": jac_t.mean(axis=0).round(6).tolist(),
        "mean_temperature_jacobian_per_lnp": density.mean(axis=0).round(6).tolist(),
        "mean_vapor_jacobian_k_per_gkg": jac_q.mean(axis=0).round(6).tolist(),
        "temperature_jacobian_sum_mean": float(total_t.mean()),
        "skin_jacobian_mean": float(jac_tskin.mean()),
        "skin_jacobian_percentiles_5_50_95": np.percentile(jac_tskin, [5, 50, 95]).round(4).tolist(),
        "peak_pressure_hpa_percentiles_5_25_50_75_95": np.percentile(peaks, [5, 25, 50, 75, 95]).round(1).tolist(),
        "half_sensitivity_band_hpa_median": [float(np.median(lo)), float(np.median(hi))],
        "vapor_peak_pressure_hpa_percentiles_5_50_95": np.percentile(q_peaks, [5, 50, 95]).round(1).tolist(),
        "sensitivity_below_500hpa_fraction_mean": float(np.mean(np.sum(np.abs(jac_t) * (p_full_hpa > 500.0), axis=1)
                                                              / np.maximum(np.sum(np.abs(jac_t), axis=1), 1e-30))),
    }


def jacobian_agreement(primary: dict, other: dict, band: int, q_gkg: np.ndarray, valid: np.ndarray) -> dict:
    """How another run's Jacobians follow the primary's: the skin Jacobian,
    the layer temperature Jacobian (per layer rms of the difference against
    the rms of the primary's) and the vapor Jacobian in its relative form
    ``dTb/dln q = jac_q * q`` (the per-g/kg form explodes where the air is
    dry and the sensitivity is nil)."""
    jt_p = np.asarray(primary["jac_t"][band])[valid]
    jt_o = np.asarray(other["jac_t"][band])[valid]
    q = np.asarray(q_gkg)[valid]
    jq_p = np.asarray(primary["jac_q"][band])[valid] * q
    jq_o = np.asarray(other["jac_q"][band])[valid] * q
    js_p = np.asarray(primary["jac_tskin"][band])[valid]
    js_o = np.asarray(other["jac_tskin"][band])[valid]
    def rms(a):
        return float(np.sqrt(np.mean(a * a)))
    return {
        "columns": int(valid.sum()),
        "skin": {"rms_primary": rms(js_p), "rms_difference": rms(js_o - js_p), "mean_primary": float(js_p.mean()), "mean_other": float(js_o.mean())},
        "temperature": {"rms_primary": rms(jt_p), "rms_difference": rms(jt_o - jt_p),
                        "column_sum_mean_primary": float(jt_p.sum(axis=1).mean()), "column_sum_mean_other": float(jt_o.sum(axis=1).mean())},
        "vapor_dln_q": {"rms_primary": rms(jq_p), "rms_difference": rms(jq_o - jq_p),
                        "column_sum_mean_primary": float(jq_p.sum(axis=1).mean()), "column_sum_mean_other": float(jq_o.sum(axis=1).mean())},
    }


def score_reference(blocks: dict, columns: dict, runs: dict[str, dict], *, primary: str, simsat_emissivity_run: str | None,
                    zenith_gate_deg: float = 60.0, gate_k: float = 1.5) -> dict:
    """Per band and surface class: SimSat, the reference with its own surface
    (``primary``), the reference at SimSat's emissivity, each against the
    observation; and the decomposition SimSat - obs = (SimSat - CRTM_A) +
    (CRTM_A - CRTM_B) + (CRTM_B - obs)."""
    kept = columns["kept"]
    w_all = blocks["n_both_clear"][kept].astype(float)
    n_pair = blocks["n_pair"][kept].astype(float) if "n_pair" in blocks else None
    land = columns["land_fraction"] >= 0.5
    zen = columns["zenith"] <= zenith_gate_deg
    classes = {"all": zen, "water": zen & ~land, "land": zen & land, "all_to70": np.ones(zen.shape, dtype=bool)}
    stream_qc = dict(STREAM_QC, zenith_max_deg=zenith_gate_deg)
    out: dict[str, Any] = {"schema": REFERENCE_SCHEMA, "primary_run": primary, "simsat_emissivity_run": simsat_emissivity_run,
                           "zenith_gate_deg": zenith_gate_deg, "gate_k": gate_k, "stream_qc": stream_qc, "bands": {}}
    for band in blocks["bands"]:
        obs = blocks[f"obs_{band}"][kept]
        sim = blocks[f"sim_{band}"][kept]
        ref_b = runs[primary]["bt"][band]
        ref_a = runs[simsat_emissivity_run]["bt"][band] if simsat_emissivity_run else None
        valid = (ref_b > 0) & ((ref_a > 0) if ref_a is not None else True)
        rows: dict[str, Any] = {}
        for cname, mask in classes.items():
            m = mask & valid
            w = w_all[m]
            row = {
                "simsat_vs_obs": paired_stats(obs[m], sim[m], w),
                "reference_vs_obs": paired_stats(obs[m], ref_b[m], w),
            }
            if ref_a is not None:
                row["reference_at_simsat_emissivity_vs_obs"] = paired_stats(obs[m], ref_a[m], w)
                row["simsat_vs_reference_at_simsat_emissivity"] = paired_stats(ref_a[m], sim[m], w)
                row["emissivity_term_refA_minus_refB"] = paired_stats(ref_b[m], ref_a[m], w)
            for name, run in runs.items():
                if name in (primary, simsat_emissivity_run):
                    continue
                other = run["bt"][band]
                ok = m & (other > 0)
                row[f"{name}_vs_obs"] = paired_stats(obs[ok], other[ok], w_all[ok])
                row[f"{name}_minus_{primary}"] = paired_stats(ref_b[ok], other[ok], w_all[ok])
            # the population the stream hands the filter (one block, one row), the primary run
            base = (mask & valid) if cname != "all_to70" else valid
            if n_pair is None:
                row["filter_facing"] = {"n": 0, "verdict": "INCOMPLETE",
                                        "reason": "the block table carries no n_pair, so the clear fraction of a block is unknown; "
                                                  "rebuild the columns with the current columns door"}
            else:
                row["filter_facing"] = filter_facing_stats(
                    obs[base], ref_b[base], n_both_clear=w_all[base], n_pair=n_pair[base], zenith=columns["zenith"][base],
                    lat=columns["lat"][base], qc=stream_qc)
            rows[cname] = row
        agreement = {}
        for name, run in runs.items():
            if name == primary:
                continue
            agreement[name] = jacobian_agreement(runs[primary], run, band, columns["q_gkg"], valid)
        gate = rows["all"]["reference_vs_obs"]
        verdict = "INCOMPLETE" if gate.get("n", 0) < 100 or "rmse_after_k" not in gate else (
            "PASS" if gate["rmse_after_k"] <= gate_k else "FAIL")
        emis_b = runs[primary]["emissivity"][band]
        out["bands"][str(band)] = {
            "classes": rows,
            "jacobian_agreement_with_primary": agreement,
            "reference_gate": {"class": "all", "zenith_max_deg": zenith_gate_deg, "rmse_after_linear_k_max": gate_k,
                               "verdict": verdict, "rmse_after_k": gate.get("rmse_after_k"), "bias_k": gate.get("bias_k"),
                               "n_blocks": gate.get("n", 0)},
            "reference_emissivity": {"water_mean": float(emis_b[valid & ~land].mean()) if np.any(valid & ~land) else None,
                                     "land_mean": float(emis_b[valid & land].mean()) if np.any(valid & land) else None,
                                     "land_percentiles_5_50_95": np.percentile(emis_b[valid & land], [5, 50, 95]).round(4).tolist() if np.any(valid & land) else None},
            "jacobians": jacobian_summary(runs[primary]["jac_t"][band][valid], runs[primary]["jac_q"][band][valid],
                                          runs[primary]["jac_tskin"][band][valid], columns["p_half_hpa"][valid],
                                          columns["p_full_hpa"][valid]),
            "refused_columns": int((~valid).sum()),
        }
    return out


def write_reference_charts(out_dir: Path, score: dict, columns: dict, blocks: dict, runs: dict[str, dict], *,
                           primary: str, simsat_emissivity_run: str | None) -> list[str]:
    """Analysis charts, not weather fields: the term decomposition per band
    and surface, the mean Jacobian profiles, the reference-against-observed
    scatter."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []
    written: list[str] = []
    kept = columns["kept"]
    for band_key, band_score in score["bands"].items():
        band = int(band_key)
        # 1. decomposition
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, cname in zip(axes, ("water", "land")):
            rows = band_score["classes"][cname]
            names = ["simsat_vs_obs", "simsat_vs_reference_at_simsat_emissivity", "emissivity_term_refA_minus_refB", "reference_vs_obs"]
            labels = ["SimSat - obs", "SimSat - CRTM\n(same emissivity)", "emissivity term\n(0.99 - CRTM surface)", "CRTM - obs"]
            present = [(l, rows[n]) for l, n in zip(labels, names) if n in rows and rows[n].get("n", 0) > 0]
            x = np.arange(len(present))
            ax.bar(x - 0.2, [r["bias_k"] for _, r in present], 0.4, label="bias")
            ax.bar(x + 0.2, [r["rmse_k"] for _, r in present], 0.4, label="rmse")
            ax.axhline(0, color="k", lw=0.6)
            ax.axhline(score["gate_k"], color="r", lw=0.8, ls="--", label=f"gate {score['gate_k']} K")
            ax.set_xticks(x)
            ax.set_xticklabels([l for l, _ in present], fontsize=8)
            ax.set_title(f"band {band}, {cname} both-clear blocks, zenith <= {score['zenith_gate_deg']:.0f} "
                         f"(n={rows['simsat_vs_obs'].get('n', 0)})", fontsize=9)
            ax.set_ylabel("K")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"ABI band {band}: the clear-sky gap by term, reference {primary}", fontsize=10)
        fig.tight_layout()
        path = out_dir / f"band{band:02d}-reference-decomposition.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(str(path))
        # 2. Jacobians
        jac = band_score["jacobians"]
        p = np.asarray(jac["mean_layer_pressure_hpa"])
        fig, axes = plt.subplots(1, 2, figsize=(9, 5))
        axes[0].plot(jac["mean_temperature_jacobian_per_lnp"], p, "-o", ms=3)
        axes[0].set_xlabel("dTb/dT per unit ln p (K/K)")
        axes[0].set_title(f"band {band} temperature weighting function\npeak median {jac['peak_pressure_hpa_percentiles_5_25_50_75_95'][2]:.0f} hPa, "
                          f"skin share {jac['skin_jacobian_mean']:.2f}", fontsize=9)
        axes[1].plot(jac["mean_vapor_jacobian_k_per_gkg"], p, "-o", ms=3, color="tab:green")
        axes[1].set_xlabel("dTb/dq (K per g/kg)")
        axes[1].set_title(f"band {band} vapor Jacobian\npeak median {jac['vapor_peak_pressure_hpa_percentiles_5_50_95'][1]:.0f} hPa", fontsize=9)
        for ax in axes:
            ax.set_yscale("log")
            ax.invert_yaxis()
            ax.set_ylabel("pressure (hPa)")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"band{band:02d}-reference-jacobians.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(str(path))
        # 3. scatter
        obs = blocks[f"obs_{band}"][kept]
        sim = blocks[f"sim_{band}"][kept]
        ref = runs[primary]["bt"][band]
        land = columns["land_fraction"] >= 0.5
        ok = (ref > 0) & (columns["zenith"] <= score["zenith_gate_deg"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        for ax, (name, val) in zip(axes, (("SimSat", sim), (f"CRTM {primary}", ref))):
            ax.scatter(obs[ok & ~land], val[ok & ~land], s=3, alpha=0.35, label="water")
            ax.scatter(obs[ok & land], val[ok & land], s=3, alpha=0.35, label="land")
            lo, hi = float(np.nanmin(obs[ok])), float(np.nanmax(obs[ok]))
            ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
            ax.set_xlabel("observed block mean (K)")
            ax.set_ylabel(f"{name} (K)")
            row = band_score["classes"]["all"]["simsat_vs_obs" if name == "SimSat" else "reference_vs_obs"]
            ax.set_title(f"{name}: bias {row['bias_k']:+.2f} K, rmse {row['rmse_k']:.2f}, after fit {row.get('rmse_after_k', float('nan')):.2f}", fontsize=9)
            ax.legend(fontsize=8)
        fig.suptitle(f"ABI band {band}, both-clear block means, zenith <= {score['zenith_gate_deg']:.0f}", fontsize=10)
        fig.tight_layout()
        path = out_dir / f"band{band:02d}-reference-scatter.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(str(path))
    return written


def write_json(path: str | os.PathLike, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("written_utc", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = [
    "COLUMNS_SCHEMA", "CRTM_OUTPUT_SCHEMA", "REFERENCE_SCHEMA", "IGBP_CLASSES", "STREAM_QC", "AbiReferenceError",
    "build_columns", "climatology_for", "filter_facing_stats", "gaussian_grid", "jacobian_summary", "ozone_ppmv",
    "paired_stats", "read_block_tables", "read_crtm_output", "read_tapes", "residual_shape", "score_reference",
    "write_columns", "write_reference_charts", "write_json",
]
