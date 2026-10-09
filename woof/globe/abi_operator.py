"""ABI infrared brightness temperature as a forward operator for the global
analysis: SimSat renders the ABI window (band 13, 10.3 um) and upper-level
water-vapor (band 8, 6.2 um) brightness temperatures from a render tape of
a WOOF global checkpoint on the exact ABI 2 km fixed-grid lattice, the
real GOES-R Level 1b radiances of the same scan arrive through the Rust
front door (``rw_goes bt``), the two meet pixel for pixel by lattice index
(``rw_goes colocate``, never an interpolation), and this module turns the
paired statistics into a scorecard with a gate.

Every data-path step is Rust: the L1b decode, the brightness
temperature inversion, the clear-sky mask, the colocation and the block
means live in ``tools/rustwx/crates/rw-goes``; the emission march is
SimSat (Rust, through its Python binding ``simsat``); the render tape is
the model's own export door.  This module is orchestration and arithmetic
on the statistics the Rust tools return.

Interface decisions, recorded here because the ensemble filter and the DA
door are built beside this lane:

* An **operator entry** is a :class:`BandSpec` row of :data:`BANDS`: the
  SimSat call that renders a band, the sensor response it uses, the
  scene class the entry is admitted for (clear sky first), the linear
  bias correction measured on the case, and the observation error the
  after-correction rmse supports.  A consumer (the ensemble filter)
  renders every member's tape through :func:`render_tile`, reads the
  observations through ``rw_goes bt`` and superobs them with ``rw_goes
  colocate --block N``; the block table is the observation vector.
* A simulated plane travels as north-first float32 little-endian with a
  JSON sidecar of schema :data:`SIM_PLANE_SCHEMA` carrying the SimSat
  ``abi_fixed_grid_crop`` block verbatim; the Rust colocation refuses a
  plane whose sidecar is not on the lattice.
* SimSat's raster is capped at 4096 samples per axis, so a full disk is
  rendered as tiles whose every corner is on the visible disk
  (:data:`DEFAULT_TILES`); tiles compose on the lattice and an overlap
  keeps the first tile's pixel.
* The gate of record for a band's entry is
  :data:`CLEAR_SKY_GATE_K`: the rmse of both-clear columns after the
  linear correction, at satellite zenith 60 degrees or less.  A band
  outside it ships no operator entry and the gap is reported by term.

Assumptions of the emission and absorption path are listed by name in
:data:`ASSUMPTIONS`; they are SimSat's, read from its optics table, and a
receipt carries them beside every number.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np

#: The sidecar schema a simulated plane carries (mirrored in
#: ``tools/rustwx/crates/rw-goes/src/radiance.rs``).
SIM_PLANE_SCHEMA = "gpuwm-da.simsat-plane.v1"
#: The pack schema ``rw_goes bt`` writes.
BT_PACK_SCHEMA = "gpuwm-obs.goes-bt.v1"
#: The receipt this module writes.
SCORE_SCHEMA = "gpuwm-da.abi-operator-score.v1"
CALIBRATION_SCHEMA = "gpuwm-da.abi-operator-calibration.v1"

#: The gate of record: both-clear columns at zenith <= 60 degrees, rmse
#: after the linear correction, in Kelvin.
CLEAR_SKY_GATE_K = 1.5
#: The zenith band the gate is read in.
GATE_ZENITH_BAND = "le60"
#: The scene class the gate is read in.
GATE_CLASS = "both_clear"

#: Tiles whose every corner sits on the GOES-East visible disk (the
#: sub-satellite point 75.2 W; a corner is visible while its earth-central
#: angle from the sub-satellite point is below about 81 degrees, and
#: (62, -140) reads 78.6).  Each tile's ABI 2 km crop stays under SimSat's
#: 4096-sample axis cap.  (label, lat_min, lat_max, lon_min, lon_max).
DEFAULT_TILES: tuple[tuple[str, float, float, float, float], ...] = (
    ("nw", 0.0, 62.0, -140.0, -75.0),
    ("ne", 0.0, 62.0, -75.0, -10.0),
    ("sw", -62.0, 0.0, -140.0, -75.0),
    ("se", -62.0, 0.0, -75.0, -10.0),
)

#: What the emission and absorption path assumes, term by term (SimSat
#: v0.3.0, ``crates/simsat/src/optics.rs``, ``ir.rs``, ``wv.rs``,
#: ``thermal_sensor.rs``, ``ingest.rs``).  Read from the code, not from
#: memory; a number here is the constant the march uses.
ASSUMPTIONS: tuple[str, ...] = (
    "geometry: the tape is a regular lat/lon grid (MAP_PROJ 6) resampled by SimSat to a 250 m "
    "vertical brick from 0 to 19.75 km MSL (80 layers); air above 20 km is not marched",
    "navigation: GOES-R ABI fixed grid on the GRS80 ellipsoid (a 6378137 m, b 6356752.31414 m, "
    "h 35786023 m, sub-satellite 75.0 W, sweep x) for the scan angles; the march itself runs on "
    "WRF's 6370 km sphere, so a pixel's ray and its navigation differ by the ellipsoid flattening",
    "temperature: T = (theta + 300) (p / 1e5)^(2/7) from the tape's T and PB, resampled to the brick",
    "source function band 13: Planck spectral radiance integrated over NOAA's FM4 (GOES-19) channel-13 "
    "spectral response, inverted through the same response (thermal_sensor GoesRAbiBand13Fm4)",
    "source function band 8: Planck at the single centre wavelength 6.2 um (fast gray), inverted "
    "at the same wavelength",
    "cloud absorption: gray per hydrometeor class, beta = kappa M with M recovered from the visible "
    "geometric-optics extinction at fixed effective radii (liquid 10 um, ice 40 um, snow 150 um, "
    "rain and graupel 1 mm); kappa 0.15 (liquid), 0.07 (ice), 9.33e-3 (snow, 2x geometric), "
    "7.0e-4 (rain, graupel) m^2 g^-1; no scattering in either band",
    "water-vapor absorption: beta = kappa_wv rho_std(z) q_v with a standard-atmosphere density "
    "rho_std = 1.225 exp(-z / 8500 m) kg m^-3 (the brick carries no pressure); kappa_wv 5.0e-3 m^2 "
    "kg^-1 in band 13 (window continuum) and 3.0 m^2 kg^-1 in band 8; no line-by-line, no CO2, "
    "no ozone, no temperature dependence of the absorption",
    "surface: gray emissivity 0.99 in both bands, emitting at the tape's TSK (the analysis skin; "
    "SST over water); no surface reflection of downwelling radiance",
    "sub-terrain: vapor below the tape's terrain is clipped by the layer's above-terrain fraction",
    "cloud fraction: none (the thermal march fills a cell with its condensate; SimSat's thermal "
    "path carries no subcolumn closure)",
    "no instrument spatial response (the 2 km sample is one ray), no limb darkening beyond the "
    "slant path itself, no scan-time offset (the model state is the 18:00 analysis, the scan runs "
    "18:00:20 to 18:09:52 UTC)",
)


@dataclass(frozen=True)
class BandSpec:
    """One operator entry: how a band is rendered and what it is for."""

    band: int
    name: str
    wavelength_um: float
    simsat_function: str
    simsat_kwargs: dict[str, Any]
    #: What the band constrains, for the filter's localisation choice.
    constrains: str
    #: The scene class the entry is admitted for.
    admitted_class: str = GATE_CLASS
    #: Zenith band of the admission.
    admitted_zenith: str = GATE_ZENITH_BAND

    def render(self, simsat, tape: str, **overrides):
        function = getattr(simsat, self.simsat_function)
        kwargs = dict(self.simsat_kwargs)
        kwargs.update(overrides)
        return function(tape, **kwargs)


#: The two bands this lane measures.  Both render in the from-space view
#: on the exact GOES-R ABI 2 km lattice; band 13 through the FM4 spectral
#: response, band 8 at its centre wavelength (SimSat carries no band-8
#: response).
BANDS: dict[int, BandSpec] = {
    13: BandSpec(
        band=13,
        name="abi-band13-window",
        wavelength_um=10.3,
        simsat_function="render_ir",
        simsat_kwargs={
            "sat": "goes-east",
            "geo_navigation": "goes-r-abi",
            "view": "geo",
            "resolution": "abi2km",
            "sensor": "goes-r-abi-band13-fm4",
        },
        constrains="skin temperature and cloud-top temperature; the clear column is the skin "
                   "seen through a weak vapor continuum",
    ),
    8: BandSpec(
        band=8,
        name="abi-band08-upper-water-vapor",
        wavelength_um=6.2,
        simsat_function="render_water_vapor",
        simsat_kwargs={
            "band": "6.2",
            "sat": "goes-east",
            "geo_navigation": "goes-r-abi",
            "view": "geo",
            "resolution": "abi2km",
        },
        constrains="upper-tropospheric water vapor and temperature (the weighting function "
                   "peaks in the upper troposphere); a moister column reads colder",
    ),
}


class AbiOperatorError(ValueError):
    """The operator cannot run as asked.  Always names the breakage."""


def require_simsat():
    """The SimSat binding, or a refusal naming what is missing."""
    try:
        import simsat  # type: ignore
    except ModuleNotFoundError as error:
        raise AbiOperatorError(
            "the SimSat Python binding is not installed: build the wheel from the SimSat "
            "tree (cd crates/simsat_py && maturin build --release) and pip install it; "
            f"import failed with {error}"
        ) from error
    return simsat


def find_rw_goes(explicit: str | os.PathLike | None = None) -> Path:
    """The ``rw_goes`` front door: ``--rw-goes``, ``WOOF_RW_GOES``, the
    tree's own build, then PATH."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("WOOF_RW_GOES")
    if env:
        candidates.append(Path(env))
    repo = Path(__file__).resolve().parents[2]
    for name in ("rw_goes", "rw_goes.exe"):
        candidates.append(repo / "tools" / "rustwx" / "target" / "release" / name)
    found = shutil.which("rw_goes")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise AbiOperatorError(
        "rw_goes is not available: pass --rw-goes, set WOOF_RW_GOES, or build it with "
        "`cargo build --release --locked --offline -p rw-goes` in tools/rustwx"
    )


def _run_rw_goes(rw_goes: Path, args: list[str]) -> dict:
    command = [str(rw_goes), *args]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    wall = time.perf_counter() - started
    if completed.returncode != 0:
        raise AbiOperatorError(
            f"rw_goes {args[0]} failed (rc {completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    record = json.loads(completed.stdout)
    record["_wall_s"] = round(wall, 3)
    record["_command"] = command
    return record


# ---------------------------------------------------------------------------
# simulated planes
# ---------------------------------------------------------------------------

@dataclass
class SimPlane:
    """One rendered tile on disk: the plane, its sidecar, and what SimSat
    said about it."""

    path: Path
    sidecar: dict
    shape: tuple[int, int]
    finite: int
    wall_s: float

    @property
    def crop(self) -> dict:
        return self.sidecar["abi_fixed_grid_crop"]


def write_sim_plane(path: str | os.PathLike, values: np.ndarray, *, band: int,
                    crop: dict, label: str, tape: str, sensor: str,
                    science_warnings: list[str] | None = None,
                    mask_plane: str | None = None, extra: dict | None = None) -> SimPlane:
    """Write a north-first float32 plane and its lattice sidecar."""
    path = Path(path)
    array = np.asarray(values, dtype="<f4")
    if array.ndim != 2:
        raise AbiOperatorError(f"a simulated plane is 2-D, got shape {array.shape}")
    ny, nx = array.shape
    required = ("x_index_min", "x_index_max", "y_index_min", "y_index_max", "nx", "ny")
    missing = [key for key in required if key not in crop]
    if missing:
        raise AbiOperatorError(
            f"the render carries no exact ABI lattice crop ({missing} missing); the plane cannot "
            "be colocated by index. Render with geo_navigation='goes-r-abi', view='geo', "
            "resolution='abi2km'"
        )
    if int(crop["nx"]) != nx or int(crop["ny"]) != ny:
        raise AbiOperatorError(
            f"the plane is {ny}x{nx} but the crop says {crop['ny']}x{crop['nx']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)
    sidecar = {
        "schema": SIM_PLANE_SCHEMA,
        "dtype": "<f4",
        "north_first": True,
        "shape": [ny, nx],
        "band": int(band),
        "label": label,
        "tape": str(tape),
        "sensor": sensor,
        "abi_fixed_grid_crop": {
            key: (float(crop[key]) if key == "sample_angle_urad" else int(crop[key]))
            for key in (*required, "sample_angle_urad") if key in crop
        },
        "mask_plane": mask_plane,
        "science_warnings": list(science_warnings or ()),
        "finite": int(np.isfinite(array).sum()),
        **(extra or {}),
    }
    Path(f"{path}.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True),
                                    encoding="utf-8")
    return SimPlane(path=path, sidecar=sidecar, shape=(ny, nx),
                    finite=sidecar["finite"], wall_s=float(sidecar.get("wall_s", 0.0)))


def read_sim_plane(path: str | os.PathLike) -> tuple[np.ndarray, dict]:
    path = Path(path)
    sidecar = json.loads(Path(f"{path}.json").read_text(encoding="utf-8"))
    if sidecar.get("schema") != SIM_PLANE_SCHEMA:
        raise AbiOperatorError(f"{path}.json declares schema {sidecar.get('schema')!r}")
    ny, nx = (int(v) for v in sidecar["shape"])
    values = np.fromfile(path, dtype="<f4")
    if values.size != ny * nx:
        raise AbiOperatorError(f"{path} holds {values.size} values, the sidecar says {ny}x{nx}")
    return values.reshape(ny, nx), sidecar


def render_tile(tape: str | os.PathLike, band: int, out_dir: str | os.PathLike, *,
                label: str, cache: str | os.PathLike | None = None,
                threads: int | None = None, simsat=None) -> SimPlane:
    """Render one band of one tape through SimSat on the exact ABI lattice
    and write the plane with its sidecar under ``out_dir``."""
    spec = BANDS.get(int(band))
    if spec is None:
        raise AbiOperatorError(f"band {band} has no operator entry; this lane measures {sorted(BANDS)}")
    simsat = simsat or require_simsat()
    overrides: dict[str, Any] = {}
    if cache is not None:
        overrides["cache"] = str(cache)
    if threads is not None:
        overrides["threads"] = int(threads)
    started = time.perf_counter()
    result = spec.render(simsat, str(tape), **overrides)
    wall = time.perf_counter() - started
    bt, geo = result[0], result[-1]
    crop = getattr(geo, "abi_fixed_grid_crop", None)
    if not crop:
        raise AbiOperatorError(
            f"SimSat returned no abi_fixed_grid_crop for {tape} band {band}; the render is not "
            "on the exact ABI lattice (geo_navigation must be goes-r-abi with resolution abi2km "
            "in the from-space view)"
        )
    out_dir = Path(out_dir)
    plane = write_sim_plane(
        out_dir / f"{label}-band{band:02d}.f32", np.asarray(bt), band=band, crop=dict(crop),
        label=label, tape=str(tape), sensor=str(spec.simsat_kwargs.get("sensor", "fast-gray")),
        science_warnings=list(getattr(geo, "science_warnings", []) or []),
        extra={"wall_s": round(wall, 3), "geo_navigation": getattr(geo, "geo_navigation", None)},
    )
    plane.wall_s = wall
    return plane


def attach_mask(plane: SimPlane, mask_path: str | os.PathLike) -> SimPlane:
    """Record a condensate mask plane (u8, same raster, from SimSat's
    ``cloud-mask-out``) in the sidecar, after proving its size."""
    mask_path = Path(mask_path)
    ny, nx = plane.shape
    size = mask_path.stat().st_size
    if size != ny * nx:
        raise AbiOperatorError(f"{mask_path} holds {size} bytes, the plane is {ny}x{nx}")
    plane.sidecar["mask_plane"] = mask_path.name if mask_path.parent == plane.path.parent else str(mask_path)
    Path(f"{plane.path}.json").write_text(json.dumps(plane.sidecar, indent=2, sort_keys=True),
                                          encoding="utf-8")
    return plane


# ---------------------------------------------------------------------------
# the Rust doors
# ---------------------------------------------------------------------------

def build_bt_pack(rw_goes: Path, rad: str | os.PathLike, out: str | os.PathLike, *,
                  acm: str | os.PathLike | None = None,
                  cmip: str | os.PathLike | None = None, band: int | None = None,
                  received_utc: str | None = None) -> dict:
    """``rw_goes bt``; ``received_utc`` (the fetch manifest's wall) rides
    into the pack's provenance row beside the granule's own publication
    time and identity, so a later reader can class the observation's
    latency instead of guessing it."""
    args = ["bt", "--rad", str(rad), "--out", str(out)]
    if acm is not None:
        args += ["--acm", str(acm)]
    if cmip is not None:
        args += ["--cmip", str(cmip)]
    if band is not None:
        args += ["--band", str(int(band))]
    if received_utc:
        args += ["--received-utc", str(received_utc)]
    return _run_rw_goes(rw_goes, args)


def colocate(rw_goes: Path, pack: str | os.PathLike, planes: list[SimPlane | str | os.PathLike],
             out_csv: str | os.PathLike, stats_json: str | os.PathLike, *,
             block: int = 24, zenith_max_deg: float | None = None) -> dict:
    args = ["colocate", "--pack", str(pack), "--out", str(out_csv), "--stats", str(stats_json),
            "--block", str(int(block))]
    for plane in planes:
        path = plane.path if isinstance(plane, SimPlane) else Path(plane)
        args += ["--sim", str(path)]
    if zenith_max_deg is not None:
        args += ["--zenith-max", str(float(zenith_max_deg))]
    return _run_rw_goes(rw_goes, args)


def quicklook(rw_goes: Path, out_png: str | os.PathLike, *, pack: str | os.PathLike | None = None,
              plane_name: str | None = None, planes: list[SimPlane | str | os.PathLike] | None = None,
              band: int | None = None, bbox_index: tuple[int, int, int, int] | None = None,
              downsample: int = 4) -> dict:
    args = ["quicklook", "--out", str(out_png), "--downsample", str(int(downsample))]
    if pack is not None:
        args += ["--pack", str(pack)]
        if plane_name:
            args += ["--plane-name", plane_name]
    for plane in planes or ():
        path = plane.path if isinstance(plane, SimPlane) else Path(plane)
        args += ["--sim", str(path)]
    if band is not None:
        args += ["--band", str(int(band))]
    if bbox_index is not None:
        args += ["--bbox-index", ",".join(str(int(v)) for v in bbox_index)]
    return _run_rw_goes(rw_goes, args)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

@dataclass
class ClassScore:
    """One scene class in one zenith band, before and after the linear
    correction the class itself supports."""

    cls: str
    zenith: str
    n: int
    mean_obs_k: float
    mean_sim_k: float
    bias_k: float
    rmse_k: float
    correlation: float
    fit_intercept_k: float | None
    fit_slope: float | None
    rmse_after_k: float | None
    #: rmse after removing the mean bias alone (a slope of one).
    rmse_debiased_k: float

    def row(self) -> dict:
        return asdict(self)


def _number(value) -> float:
    """A statistic from the Rust record: serde writes a NaN as ``null``, and
    an empty class carries nothing but nulls."""
    return float("nan") if value is None else float(value)


def class_score(cls: str, zenith: str, stats: dict) -> ClassScore:
    m = stats["moments"]
    n = int(m["n"])
    bias = _number(stats["bias_k"])
    rmse = _number(stats["rmse_k"])
    debiased = math.sqrt(max(rmse * rmse - bias * bias, 0.0)) if n > 0 else float("nan")
    return ClassScore(
        cls=cls, zenith=zenith, n=n,
        mean_obs_k=_number(stats["mean_obs_k"]), mean_sim_k=_number(stats["mean_sim_k"]),
        bias_k=bias, rmse_k=rmse, correlation=_number(stats["correlation"]),
        fit_intercept_k=stats.get("linear_fit_intercept_k"),
        fit_slope=stats.get("linear_fit_slope"),
        rmse_after_k=stats.get("rmse_after_linear_k"),
        rmse_debiased_k=debiased,
    )


def score_band(colocation: dict, *, gate_k: float = CLEAR_SKY_GATE_K,
               gate_class: str = GATE_CLASS, gate_zenith: str = GATE_ZENITH_BAND,
               minimum_pairs: int = 1000) -> dict:
    """Turn one colocation record into a scorecard with the gate decided.

    The gate reads the after-correction rmse of ``gate_class`` in
    ``gate_zenith``.  A class with fewer than ``minimum_pairs`` pairs
    measured nothing and reads INCOMPLETE, never PASS.
    """
    classes = colocation["classes"]
    scores = {key: class_score(*key.split("/", 1), value).row() for key, value in classes.items()}
    gate_key = f"{gate_class}/{gate_zenith}"
    gate = scores.get(gate_key)
    if gate is None:
        verdict, reason = "INCOMPLETE", f"the colocation carries no class {gate_key}"
    elif gate["n"] < minimum_pairs:
        verdict = "INCOMPLETE"
        reason = f"{gate_key} holds {gate['n']} pairs, below the {minimum_pairs} the gate needs"
    elif gate["rmse_after_k"] is None:
        verdict = "INCOMPLETE"
        reason = f"{gate_key} supports no linear fit (constant simulation or too few pairs)"
    elif gate["rmse_after_k"] <= gate_k:
        verdict = "PASS"
        reason = (f"{gate_key}: rmse {gate['rmse_k']:.3f} K, bias {gate['bias_k']:+.3f} K, after "
                  f"obs = {gate['fit_intercept_k']:.3f} + {gate['fit_slope']:.4f} sim the rmse is "
                  f"{gate['rmse_after_k']:.3f} K, within {gate_k} K")
    else:
        verdict = "FAIL"
        reason = (f"{gate_key}: rmse {gate['rmse_k']:.3f} K, bias {gate['bias_k']:+.3f} K, after "
                  f"the linear fit {gate['rmse_after_k']:.3f} K, outside {gate_k} K")
    return {
        "schema": SCORE_SCHEMA,
        "band": int(colocation["band"]),
        "satellite": colocation["satellite"],
        "scan_start": colocation["scan_start"],
        "scan_end": colocation["scan_end"],
        "pairs": int(colocation["counts"]["pairs"]),
        "counts": colocation["counts"],
        "block_pixels": colocation["block_pixels"],
        "block_table": colocation["block_table"],
        "has_clear_sky_mask": bool(colocation["has_clear_sky_mask"]),
        "gate": {
            "class": gate_class, "zenith": gate_zenith, "rmse_after_linear_k_max": gate_k,
            "minimum_pairs": minimum_pairs, "verdict": verdict, "reason": reason,
        },
        "classes": scores,
    }


def operator_entry(score: dict, band: int) -> dict | None:
    """The shipped operator entry a passing band earns: the band's spec,
    its correction and the observation error its residual supports.
    ``None`` when the gate did not pass."""
    if score["gate"]["verdict"] != "PASS":
        return None
    spec = BANDS[int(band)]
    gate = score["classes"][f"{score['gate']['class']}/{score['gate']['zenith']}"]
    return {
        "band": spec.band,
        "name": spec.name,
        "wavelength_um": spec.wavelength_um,
        "simsat_function": spec.simsat_function,
        "simsat_kwargs": spec.simsat_kwargs,
        "constrains": spec.constrains,
        "admitted_class": score["gate"]["class"],
        "admitted_zenith": score["gate"]["zenith"],
        "bias_correction": {
            "form": "obs = intercept + slope * simulated",
            "intercept_k": gate["fit_intercept_k"],
            "slope": gate["fit_slope"],
            "measured_on": f"{score['satellite']} {score['scan_start']}, {gate['n']} pairs",
        },
        "observation_error_k": gate["rmse_after_k"],
        "assumptions": list(ASSUMPTIONS),
    }


# ---------------------------------------------------------------------------
# calibration: planted perturbations must read back
# ---------------------------------------------------------------------------

def plant_perturbation(checkpoint_in: str | os.PathLike, checkpoint_out: str | os.PathLike, *,
                       skin_delta_k: float | None = None, qv_factor: float | None = None,
                       qv_levels: slice | None = None, semi_implicit_scheme: str | None = None,
                       integrator: str | None = None) -> dict:
    """Write a copy of a checkpoint with a planted change: a uniform skin
    temperature offset, or the spectral vapor coefficients of a level
    range scaled by a factor (the field scales linearly).  Everything
    else, the metadata included, is carried through; the writer recomputes
    the self hash.  Returns a receipt of what was planted."""
    from .checkpoint import read_checkpoint, write_checkpoint_arrays

    metadata, arrays = read_checkpoint(checkpoint_in)
    planted: dict[str, Any] = {"source": str(checkpoint_in), "target": str(checkpoint_out)}
    if skin_delta_k is not None:
        key = "surface__surface_temperature_k"
        if key not in arrays:
            raise AbiOperatorError(f"{checkpoint_in} carries no {key}")
        arrays[key] = (arrays[key] + np.asarray(skin_delta_k, dtype=arrays[key].dtype)).astype(
            arrays[key].dtype)
        planted["skin_delta_k"] = float(skin_delta_k)
    if qv_factor is not None:
        key = "atmosphere__qv"
        if key not in arrays:
            raise AbiOperatorError(f"{checkpoint_in} carries no {key}")
        qv = np.array(arrays[key], copy=True)
        levels = qv_levels if qv_levels is not None else slice(0, qv.shape[0])
        qv[levels] = qv[levels] * qv_factor
        arrays[key] = qv.astype(arrays[key].dtype)
        planted["qv_factor"] = float(qv_factor)
        planted["qv_levels"] = [levels.start, levels.stop] if isinstance(levels, slice) else list(levels)
        planted["qv_shape"] = list(qv.shape)
    if skin_delta_k is None and qv_factor is None:
        planted["identity"] = True
    kwargs: dict[str, Any] = {}
    if semi_implicit_scheme is not None:
        kwargs["semi_implicit_scheme"] = semi_implicit_scheme
    if integrator is not None:
        kwargs["integrator"] = integrator
    write_checkpoint_arrays(
        checkpoint_out, arrays,
        step=int(metadata["step"]), time_s=float(metadata["time_s"]),
        physics_state_schema=metadata["physics_state_schema"],
        physics_metadata=metadata["physics_metadata"],
        config_hash=metadata["config_hash"],
        trackers=metadata.get("run_trackers"),
        **kwargs,
    )
    written, _ = read_checkpoint(checkpoint_out)
    from .pins import scheme_of_pins_hash

    # the source may carry the v2 pin WOOF 1.0.0 wrote; the plant is written
    # under v3, and either is the same arithmetic when their labels agree
    if written["pins_hash"] != metadata["pins_hash"] and (
            scheme_of_pins_hash(written["pins_hash"]) is None
            or scheme_of_pins_hash(written["pins_hash"])
            != scheme_of_pins_hash(metadata["pins_hash"])):
        Path(checkpoint_out).unlink(missing_ok=True)
        raise AbiOperatorError(
            f"the planted checkpoint would carry pins {written['pins_hash'][:12]} where the source "
            f"carries {metadata['pins_hash'][:12]}: pass the config's semi_implicit_scheme and "
            "integrator so the plant stays on the source's arithmetic")
    planted["pins_hash"] = metadata["pins_hash"]
    return planted


def levels_above(cfg, pressure_pa: float, *, surface_pressure_pa: float = 1.0e5) -> slice:
    """The model levels whose full-level pressure at a reference surface
    pressure is below ``pressure_pa`` (the top of the atmosphere is level
    0), as a slice for :func:`plant_perturbation`."""
    a_half = np.asarray(cfg.a_half_pa, dtype=np.float64)
    b_half = np.asarray(cfg.b_half, dtype=np.float64)
    p_half = a_half + b_half * surface_pressure_pa
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    above = np.where(p_full < pressure_pa)[0]
    if above.size == 0:
        raise AbiOperatorError(f"no level lies above {pressure_pa} Pa")
    return slice(0, int(above[-1]) + 1)


def readback(control: np.ndarray, planted: np.ndarray, *, mask: np.ndarray | None = None) -> dict:
    """What a planted change did to a plane: the mean, median and rms of
    (planted minus control) over pixels finite in both (and inside
    ``mask`` when given), the fraction that moved, and the sign census.
    Two identical planes read exactly zero."""
    if control.shape != planted.shape:
        raise AbiOperatorError(f"planes differ in shape: {control.shape} against {planted.shape}")
    both = np.isfinite(control) & np.isfinite(planted)
    if mask is not None:
        both &= np.asarray(mask, dtype=bool)
    delta = (planted[both].astype(np.float64) - control[both].astype(np.float64))
    if delta.size == 0:
        return {"n": 0}
    return {
        "n": int(delta.size),
        "mean_k": float(delta.mean()),
        "median_k": float(np.median(delta)),
        "rms_k": float(np.sqrt(np.mean(delta * delta))),
        "min_k": float(delta.min()),
        "max_k": float(delta.max()),
        "moved_fraction": float(np.mean(np.abs(delta) > 0.0)),
        "positive_fraction": float(np.mean(delta > 0.0)),
        "negative_fraction": float(np.mean(delta < 0.0)),
        "identical": bool(np.all(delta == 0.0)),
    }


def judge_calibration(band: int, plant: str, expected_sign: int, result: dict, *,
                      minimum_magnitude_k: float) -> dict:
    """Both directions: a planted change must read back with its sign and
    at least ``minimum_magnitude_k`` in the mean; an identity plant must
    read exactly zero."""
    if expected_sign == 0:
        passed = bool(result.get("identical", False))
        reason = ("identical planes read exactly zero" if passed
                  else f"an unchanged state moved the plane by {result.get('rms_k')} K rms")
    else:
        mean = result.get("mean_k", float("nan"))
        passed = bool(result.get("n", 0) > 0 and math.copysign(1.0, mean) == expected_sign
                      and abs(mean) >= minimum_magnitude_k)
        reason = (f"mean read-back {mean:+.3f} K over {result.get('n', 0)} pixels, expected sign "
                  f"{'+' if expected_sign > 0 else '-'} and at least {minimum_magnitude_k} K")
    return {"band": int(band), "plant": plant, "expected_sign": expected_sign,
            "passed": passed, "reason": reason, "readback": result}


def write_receipt(path: str | os.PathLike, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("written_utc", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# the case: export, render, decode, colocate, score, calibrate
# ---------------------------------------------------------------------------

def _tape_of(tapes_dir: Path) -> Path | None:
    found = sorted(tapes_dir.glob("wrfout_d01_*")) if tapes_dir.is_dir() else []
    return found[0] if found else None


def export_tile_tape(cfg, checkpoint: str | os.PathLike, tapes_dir: str | os.PathLike, *,
                     start_date: str, bbox: tuple[float, float, float, float] | None,
                     nlat: int = 720, nlon: int = 1440, reuse: bool = True) -> Path:
    """One render tape of one checkpoint on a lat/lon window, through the
    model's own export door; an existing tape is reused when asked."""
    from .wrfout_export import export_wrfout

    tapes_dir = Path(tapes_dir)
    if reuse:
        existing = _tape_of(tapes_dir)
        if existing is not None:
            return existing
    written = export_wrfout(cfg, [Path(checkpoint)], tapes_dir, nlat=nlat, nlon=nlon,
                            start_date=start_date, overwrite=True, bbox=bbox)
    return Path(written[0])


def simsat_cli_mask(simsat_cli: str | os.PathLike, tape: str | os.PathLike, out_dir: str | os.PathLike,
                    *, label: str, cache: str | os.PathLike | None, threads: int | None) -> dict:
    """SimSat's headless band-13 render of the same tape on the same lattice,
    for its condensate mask (the binding carries none) and a plane to
    cross-check the binding's render against."""
    out_dir = Path(out_dir)
    png = out_dir / f"{label}-band13-cli.png"
    bt_out = out_dir / f"{label}-band13-cli.f32"
    mask_out = out_dir / f"{label}-band13.mask"
    command = [str(simsat_cli), f"input={tape}", f"out={png}", f"bt-out={bt_out}",
               f"cloud-mask-out={mask_out}", "view=geo", "sat=goes-east",
               "geo-navigation=goes-r-abi", "resolution=abi2km", "sensor=goes-r-abi-band13-fm4",
               "enhancement=cimss"]
    if cache is not None:
        command.append(f"cache={cache}")
    if threads is not None:
        command.append(f"threads={int(threads)}")
    started = time.perf_counter()
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise AbiOperatorError(f"simsat-render-ir failed (rc {done.returncode}): {done.stderr.strip()[-2000:]}")
    return {"png": str(png), "bt_out": str(bt_out), "mask": str(mask_out),
            "wall_s": round(time.perf_counter() - started, 3),
            "summary": [line for line in done.stdout.splitlines() if line.startswith(("SUMMARY", "IRSUMMARY"))]}


def cross_check_cli(plane: SimPlane, cli_bt: str | os.PathLike) -> dict:
    """The binding's plane against the CLI's plane of the same tape: the
    same march on the same lattice, so they must be identical; the mask
    the CLI wrote is aligned exactly when they are."""
    values, _ = read_sim_plane(plane.path)
    cli = np.fromfile(cli_bt, dtype="<f4")
    if cli.size != values.size:
        raise AbiOperatorError(
            f"the CLI plane holds {cli.size} values, the binding's {values.size}; the two renders "
            "are not on one raster, so the CLI mask cannot be attached")
    cli = cli.reshape(values.shape)
    both = np.isfinite(values) & np.isfinite(cli)
    diff = np.abs(values[both] - cli[both])
    return {
        "n_both": int(both.sum()),
        "finite_binding": int(np.isfinite(values).sum()),
        "finite_cli": int(np.isfinite(cli).sum()),
        "max_abs_k": float(diff.max()) if diff.size else None,
        "identical": bool(diff.size and float(diff.max()) == 0.0),
    }


def _write_charts(out_dir: Path, band: int, score: dict, colocation: dict, block_csv: Path,
                  state_label: str = "the checkpoint") -> list[str]:
    """Analysis charts (not weather fields): bias and rmse per class, the
    difference histogram of the gate class, the block-mean scatter."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []
    written: list[str] = []
    classes = score["classes"]
    zen = score["gate"]["zenith"]
    names = [c for c in ("all", "both_clear", "obs_clear", "obs_cloudy", "both_cloudy",
                         "obs_clear_sim_cloudy", "obs_cloudy_sim_clear")
             if f"{c}/{zen}" in classes and classes[f"{c}/{zen}"]["n"] > 0]
    if names:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(names))
        bias = [classes[f"{c}/{zen}"]["bias_k"] for c in names]
        rmse = [classes[f"{c}/{zen}"]["rmse_k"] for c in names]
        after = [classes[f"{c}/{zen}"]["rmse_after_k"] or float("nan") for c in names]
        ax.bar(x - 0.27, bias, 0.27, label="bias (sim minus obs)")
        ax.bar(x, rmse, 0.27, label="rmse")
        ax.bar(x + 0.27, after, 0.27, label="rmse after linear correction")
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(CLEAR_SKY_GATE_K, color="r", lw=0.8, ls="--", label=f"gate {CLEAR_SKY_GATE_K} K")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\nn={classes[f'{c}/{zen}']['n']}" for c in names], fontsize=8)
        ax.set_ylabel("K")
        ax.set_title(f"ABI band {band}: SimSat from {state_label} against {score['satellite']} "
                     f"L1b, {score['scan_start']}, zenith {zen}", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"band{band:02d}-class-scores.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(str(path))
    key = f"{score['gate']['class']}/{zen}"
    if key in colocation["classes"]:
        hist = colocation["classes"][key]["diff_histogram_1k"]
        edges = np.arange(-60, 61)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(edges[:-1] + 0.5, hist, width=1.0)
        ax.set_xlabel("simulated minus observed brightness temperature (K)")
        ax.set_ylabel("pixels")
        ax.set_title(f"band {band} {key}: n={classes[key]['n']}, bias {classes[key]['bias_k']:+.2f} K, "
                     f"rmse {classes[key]['rmse_k']:.2f} K")
        fig.tight_layout()
        path = out_dir / f"band{band:02d}-{score['gate']['class']}-histogram.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(str(path))
    try:
        table = np.genfromtxt(block_csv, delimiter=",", names=True)
    except (OSError, ValueError):
        table = None
    if table is not None and table.size:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ok = table["n_both_clear"] > 0
        ax.scatter(table["obs_mean_k"], table["sim_mean_k"], s=4, alpha=0.4, label="every block")
        if ok.any():
            ax.scatter(table["obs_mean_both_clear_k"][ok], table["sim_mean_both_clear_k"][ok], s=4,
                       alpha=0.6, label="both-clear pixels of the block")
        lo = float(np.nanmin(table["obs_mean_k"]))
        hi = float(np.nanmax(table["obs_mean_k"]))
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
        ax.set_xlabel("observed block mean (K)")
        ax.set_ylabel("simulated block mean (K)")
        ax.set_title(f"band {band}: {colocation['block_pixels']}-pixel blocks, {int(table.size)} blocks")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"band{band:02d}-block-scatter.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(str(path))
    return written


def run_case(cfg, checkpoint: str | os.PathLike, *, start_date: str, out_dir: str | os.PathLike,
             goes_rad: dict[int, str | os.PathLike], goes_acm: str | os.PathLike | None = None,
             goes_cmip: dict[int, str | os.PathLike] | None = None,
             tiles=DEFAULT_TILES, bands: tuple[int, ...] = (13, 8), rw_goes: str | os.PathLike | None = None,
             simsat_cli: str | os.PathLike | None = None, threads: int | None = None, block: int = 24,
             zenith_max_deg: float | None = None, calibrate: bool = False, charts: bool = True,
             nlat: int = 720, nlon: int = 1440, reuse_tapes: bool = True,
             skin_delta_k: float = 2.0, qv_factor: float = 1.5, qv_above_pa: float = 50_000.0,
             received_utc: dict[int, str] | None = None) -> dict:
    """The whole measurement of item 5a for one analysis and one scan.

    Tapes of every tile (the export door), SimSat planes of every band on
    the exact ABI lattice, the L1b packs through ``rw_goes bt`` (with the
    clear-sky mask and the CMIP cross-check), the colocation by index, the
    scorecard with its gate, the operator entries the gate admits, the
    calibration read-backs when asked, the quicklooks and charts, and one
    receipt that carries it all with the assumptions named.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rw_goes_path = find_rw_goes(rw_goes)
    goes_cmip = {int(k): v for k, v in (goes_cmip or {}).items()}
    goes_rad = {int(k): v for k, v in goes_rad.items()}
    for band in bands:
        if band not in goes_rad:
            raise AbiOperatorError(f"band {band} is asked for but no L1b radiance file was given for it")
        if band not in BANDS:
            raise AbiOperatorError(f"band {band} has no operator entry; this lane measures {sorted(BANDS)}")
    receipt: dict[str, Any] = {
        "schema": SCORE_SCHEMA,
        "config_hash": cfg.config_hash,
        "config_name": cfg.name,
        "checkpoint": str(checkpoint),
        "analysis_valid": start_date,
        "rw_goes": str(rw_goes_path),
        "simsat_cli": str(simsat_cli) if simsat_cli else None,
        "tiles": [list(t) for t in tiles],
        "bands": list(bands),
        "block_pixels": block,
        "zenith_max_deg": zenith_max_deg,
        "assumptions": list(ASSUMPTIONS),
        "gate": {"class": GATE_CLASS, "zenith": GATE_ZENITH_BAND, "rmse_after_linear_k_max": CLEAR_SKY_GATE_K},
        "steps": {},
    }
    cache = out_dir / "simsat-cache"
    cache.mkdir(exist_ok=True)

    # 1. tapes
    tapes: dict[str, Path] = {}
    started = time.perf_counter()
    for label, lat_min, lat_max, lon_min, lon_max in tiles:
        tapes[label] = export_tile_tape(cfg, checkpoint, out_dir / "tapes" / label, start_date=start_date,
                                        bbox=(lat_min, lat_max, lon_min, lon_max), nlat=nlat, nlon=nlon,
                                        reuse=reuse_tapes)
    receipt["steps"]["tapes"] = {"wall_s": round(time.perf_counter() - started, 1),
                                 "tapes": {k: str(v) for k, v in tapes.items()}}

    # 2. simulated planes, 3. the CLI mask
    planes: dict[int, dict[str, SimPlane]] = {band: {} for band in bands}
    render_rows: dict[str, Any] = {}
    sim_dir = out_dir / "sim"
    for label, tape in tapes.items():
        row: dict[str, Any] = {}
        for band in bands:
            plane = render_tile(tape, band, sim_dir, label=label, cache=cache, threads=threads)
            planes[band][label] = plane
            row[str(band)] = {"plane": str(plane.path), "shape": list(plane.shape), "finite": plane.finite,
                              "crop": plane.crop, "wall_s": round(plane.wall_s, 1),
                              "science_warnings": plane.sidecar.get("science_warnings")}
        if simsat_cli:
            cli = simsat_cli_mask(simsat_cli, tape, sim_dir, label=label, cache=cache, threads=threads)
            check = cross_check_cli(planes[13][label], cli["bt_out"]) if 13 in planes else None
            cli["cross_check"] = check
            if check is None or check["identical"]:
                for band in bands:
                    attach_mask(planes[band][label], cli["mask"])
                mask = np.fromfile(cli["mask"], dtype=np.uint8)
                cli["mask_census"] = {"clear": int((mask == 0).sum()), "condensate": int((mask == 1).sum()),
                                      "no_data": int((mask == 255).sum())}
            else:
                cli["mask_not_attached"] = ("the CLI plane differs from the binding's plane "
                                            f"(max {check['max_abs_k']} K), so its mask is not on the same raster")
            row["cli"] = cli
        render_rows[label] = row
    receipt["steps"]["render"] = render_rows

    # 4. the observations
    packs: dict[int, Path] = {}
    pack_rows: dict[str, Any] = {}
    pack_dir = out_dir / "packs"
    pack_dir.mkdir(exist_ok=True)
    for band in bands:
        pack = pack_dir / f"band{band:02d}.goespack"
        record = build_bt_pack(rw_goes_path, goes_rad[band], pack, acm=goes_acm, cmip=goes_cmip.get(band), band=band,
                               received_utc=(received_utc or {}).get(band))
        packs[band] = pack
        pack_rows[str(band)] = {k: v for k, v in record.items() if k not in ("_command",)}
    receipt["steps"]["packs"] = pack_rows

    # 5. colocation, 6. scores
    scores: dict[str, Any] = {}
    entries: dict[str, Any] = {}
    for band in bands:
        csv_path = out_dir / f"band{band:02d}-blocks.csv"
        stats_path = out_dir / f"band{band:02d}-colocation.json"
        record = colocate(rw_goes_path, packs[band], list(planes[band].values()), csv_path, stats_path,
                          block=block, zenith_max_deg=zenith_max_deg)
        score = score_band(record)
        scores[str(band)] = score
        entries[str(band)] = operator_entry(score, band)
        if charts:
            score["charts"] = _write_charts(
                out_dir, band, score, record, csv_path,
                state_label=f"{cfg.name} {Path(checkpoint).name} valid {start_date}")
    receipt["scores"] = scores
    receipt["operator_entries"] = entries

    # 7. quicklooks through the rw-sat palette: observed and simulated, same band, same crop
    looks: dict[str, Any] = {}
    for band in bands:
        crops = [p.crop for p in planes[band].values()]
        bbox = (min(c["x_index_min"] for c in crops), max(c["x_index_max"] for c in crops),
                min(c["y_index_min"] for c in crops), max(c["y_index_max"] for c in crops))
        obs_png = out_dir / f"band{band:02d}-observed.png"
        sim_png = out_dir / f"band{band:02d}-simulated.png"
        looks[str(band)] = {
            "observed": quicklook(rw_goes_path, obs_png, pack=packs[band], plane_name="bt", band=band,
                                  bbox_index=bbox, downsample=4),
            "simulated": quicklook(rw_goes_path, sim_png, planes=list(planes[band].values()), band=band,
                                   bbox_index=bbox, downsample=4),
        }
    receipt["quicklooks"] = looks

    # 8. calibration: planted changes must read back in the right band, the identity must read zero
    if calibrate:
        receipt["calibration"] = calibrate_case(
            cfg, checkpoint, out_dir / "calibration", start_date=start_date, tiles=tiles, planes=planes,
            threads=threads, nlat=nlat, nlon=nlon, skin_delta_k=skin_delta_k,
            qv_factor=qv_factor, qv_above_pa=qv_above_pa)

    receipt["verdicts"] = {str(band): scores[str(band)]["gate"]["verdict"] for band in bands}
    write_receipt(out_dir / "abi-operator-receipt.json", receipt)
    return receipt


def calibrate_case(cfg, checkpoint, out_dir: Path, *, start_date: str, tiles, planes, threads,
                   nlat: int, nlon: int, skin_delta_k: float, qv_factor: float, qv_above_pa: float) -> dict:
    """Both directions on the first tile: a warmer skin in band 13 must read
    back warmer and a cooler one cooler, moistening the upper troposphere
    in band 8 must read back colder and drying it warmer, and an identity
    plant must read exactly zero in both.  The control planes are the
    case's own renders of that tile."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label, lat_min, lat_max, lon_min, lon_max = tiles[0]
    bbox = (lat_min, lat_max, lon_min, lon_max)
    results: dict[str, Any] = {"schema": CALIBRATION_SCHEMA, "tile": label, "plants": {}}
    controls: dict[int, np.ndarray] = {}
    clear_mask = None
    for band, by_label in planes.items():
        values, sidecar = read_sim_plane(by_label[label].path)
        controls[band] = values
        if band == 13 and sidecar.get("mask_plane"):
            mask_path = Path(by_label[label].path).parent / sidecar["mask_plane"]
            clear_mask = np.fromfile(mask_path, dtype=np.uint8).reshape(values.shape) == 0
    upper = levels_above(cfg, qv_above_pa)
    plants = [
        ("identity", {}, [(13, 0, 0.0), (8, 0, 0.0)]),
        ("skin_plus", {"skin_delta_k": skin_delta_k}, [(13, +1, 0.5 * skin_delta_k)]),
        ("skin_minus", {"skin_delta_k": -skin_delta_k}, [(13, -1, 0.5 * skin_delta_k)]),
        ("upper_moistening", {"qv_factor": qv_factor, "qv_levels": upper}, [(8, -1, 0.2)]),
        ("upper_drying", {"qv_factor": 1.0 / qv_factor, "qv_levels": upper}, [(8, +1, 0.2)]),
    ]
    judged: list[dict] = []
    for name, kwargs, expectations in plants:
        wanted = {band for band, _, _ in expectations}
        if not wanted & set(planes):
            continue
        plant_dir = out_dir / name
        plant_dir.mkdir(exist_ok=True)
        planted_ck = plant_dir / "planted.npz"
        planted = plant_perturbation(checkpoint, planted_ck, semi_implicit_scheme=cfg.semi_implicit_scheme,
                                     integrator=cfg.integrator, **kwargs)
        tape = export_tile_tape(cfg, planted_ck, plant_dir / "tape", start_date=start_date, bbox=bbox,
                                nlat=nlat, nlon=nlon, reuse=False)
        row: dict[str, Any] = {"planted": planted, "tape": str(tape), "bands": {}}
        for band, sign, magnitude in expectations:
            if band not in planes:
                continue
            plane = render_tile(tape, band, plant_dir, label=f"{label}-{name}", cache=plant_dir / "cache",
                                threads=threads)
            values, _ = read_sim_plane(plane.path)
            masked = band == 13 and clear_mask is not None
            result = readback(controls[band], values, mask=clear_mask if masked else None)
            verdict = judge_calibration(band, name, sign, result, minimum_magnitude_k=magnitude)
            verdict["plane"] = str(plane.path)
            verdict["masked_to"] = ("clear columns of the control (SimSat condensate mask)" if masked
                                    else "every finite pixel")
            row["bands"][str(band)] = verdict
            judged.append(verdict)
        results["plants"][name] = row
    results["passed"] = bool(judged) and all(v["passed"] for v in judged)
    results["count"] = {"judged": len(judged), "passed": sum(1 for v in judged if v["passed"])}
    write_receipt(out_dir / "calibration-receipt.json", results)
    return results


__all__ = [
    "ASSUMPTIONS", "BANDS", "BandSpec", "CLEAR_SKY_GATE_K", "DEFAULT_TILES", "SIM_PLANE_SCHEMA",
    "AbiOperatorError", "attach_mask", "build_bt_pack", "calibrate_case", "class_score", "colocate",
    "cross_check_cli", "export_tile_tape", "find_rw_goes", "judge_calibration", "levels_above",
    "operator_entry", "plant_perturbation", "quicklook", "read_sim_plane", "readback", "render_tile",
    "require_simsat", "run_case", "score_band", "simsat_cli_mask", "write_receipt", "write_sim_plane",
]


# ---------------------------------------------------------------------------
# the fast operator's entry and the four assessments (design amendments G, H)
# ---------------------------------------------------------------------------

#: The stream name and variable the ensemble filter sees.
FAST_STREAM = "goes-abi-l1b-bt"
FAST_VARIABLE = "brightness_temperature_k"
#: The classes an entry may be admitted for, in the order they are judged.
ADMISSION_CLASSES = ("water", "land")


def acceptance_contract(band: int, table_band: dict, gate_zenith_deg: float, *, qc: dict | None = None) -> dict:
    """What the operator claims to measure and how (amendment H): the
    measurement, its time and location, its vertical coordinate, its
    representativeness, its bias treatment and its error correlations,
    each stated so a reader of the receipt can dispute it.  ``qc`` is the
    stream's block selection (:data:`abi_reference.STREAM_QC`), the
    population the entry's numbers are graded on."""
    from .abi_reference import STREAM_QC
    qc = dict(STREAM_QC, **(qc or {}))
    jac = table_band.get("reference_jacobians", {})
    peak = jac.get("peak_pressure_hpa_percentiles_5_25_50_75_95", [None] * 5)
    band_half = jac.get("half_sensitivity_band_hpa_median", [None, None])
    planck = table_band["planck"]
    return {
        "measurement": (f"ABI band {band} brightness temperature: the Level 1b radiance inverted with the granule's own "
                        f"band-corrected Planck constants (fk1 {planck['fk1']:.6g}, fk2 {planck['fk2']:.6g}, bc1 {planck['bc1']:.5g}, "
                        f"bc2 {planck['bc2']:.6g}); the superobservation is the mean over the both-clear pixels of a 24-pixel "
                        "block (about 48 km at the sub-satellite point), clear by the ACM mask and by the model's own condensate"),
        "time": ("the block's position in the scan window: the full disk scans north to south over about 9.5 minutes, a row's "
                 "fraction from the north places it linearly in the window (swath structure not modelled, about 26 s uncertainty); "
                 "the nominal scan start is the analysis instant compared today"),
        "location": "the block's mean pixel latitude and longitude on the GOES-R fixed grid (GRS80 navigation); no parallax "
                    "correction is applied to a clear-sky column",
        "vertical_coordinate": (f"not one level: the temperature weighting function of the reference peaks at {peak[2]} hPa "
                                f"(5 to 95 percent of columns {peak[0]} to {peak[4]} hPa) and holds half its sensitivity between "
                                f"{band_half[0]} and {band_half[1]} hPa (median); the operator's Jacobians (temperature, vapor, skin) "
                                "ride in its output for model-space localisation; release 1 places the row at the median peak with "
                                f"the band's own vertical cutoff and says so; skin share {jac.get('skin_jacobian_mean')}"),
        "representativeness": ("one 24-pixel block against one model column at the block centre (T255, about 52 km); the block "
                               "mean stands for the column, the pixel-to-pixel variance inside a clear block is not carried"),
        "bias_treatment": ("the linear correction obs = a + b sim is measured per class on the blocks the stream hands the "
                           "filter (one block, one row, no pixel weighting) and recorded; the entry ships with that correction "
                           "and the residual after it as the observation error, and the stream applies the correction to the "
                           "operator's output so the filter's O-B is in the corrected frame; the sensor term (GOES-16 "
                           "transmittance coefficients under GOES-19 Planck constants) is inside the correction"),
        "error_correlations": ("adjacent blocks share the analysed column's errors and the scan's calibration; the filter treats "
                               "blocks as independent with the error the filter-facing residual after the correction; a block is "
                               "24 ABI pixels (about 48 km at the sub-satellite point, a larger footprint toward the limb) and no "
                               f"further thinning is applied; a block is kept only when it holds at least {qc['minimum_pixels']} "
                               f"both-clear pixels and at least {qc['minimum_clear_fraction']:.0%} of its paired pixels are both-clear "
                               f"(the cloud-edge gate); zenith beyond {gate_zenith_deg} degrees is not admitted; the residual's "
                               "shape (tails, skew, its correlation with zenith and latitude) rides in the entry so the Gaussian "
                               "the error stands for can be disputed"),
    }


def fast_operator_entries(score: dict, table: dict, *, gate_k: float = CLEAR_SKY_GATE_K, operator_run: str = "fast_own",
                          reference_run: str | None = None) -> dict:
    """The operator entries the reference scorecard admits, per band and
    per surface class.  A class is judged on the population the stream
    hands the filter (the ``filter_facing`` row of the score: the stream's
    own block QC, one block one row, no pixel weighting): inside the gate
    on at least 1000 blocks it is admitted with that row's correction and
    residual as its error; otherwise it is reported with its numbers.  The
    pixel-weighted statistics over every both-clear block ride beside as
    the record of the whole scan.  The Jacobian agreement and the
    brightness-temperature difference are measured against the named
    reference run (``reference_run``, or the score's own ``reference_run``);
    without one the entries are refused, because an entry that compares
    the operator with an arbitrary other run says nothing about the
    reference.  ``score`` is an ``abi-reference score`` receipt whose
    primary run is the operator (``fast_own``)."""
    reference = reference_run or score.get("reference_run")
    qc = score.get("stream_qc")
    entries: dict[str, Any] = {"schema": "gpuwm-da.abi-operator-entries.v1", "stream": FAST_STREAM, "variable": FAST_VARIABLE,
                               "operator": {"binary": "rw_goes forward", "table_schema": table.get("schema"),
                                            "coordinate_sha256": table.get("vertical", {}).get("sha256"),
                                            "table_written_utc": table.get("written_utc"), "run": operator_run},
                               "reference_run": reference,
                               "gate": {"rmse_after_linear_k_max": gate_k, "minimum_blocks": 1000, "zenith_max_deg": score["zenith_gate_deg"],
                                        "population": "the blocks the stream hands the filter (stream_qc), one block one row, unweighted"},
                               "stream_qc": qc,
                               "bands": {}}
    for band_key, row in score["bands"].items():
        table_band = table["bands"][band_key]
        agreement = row.get("jacobian_agreement_with_primary", {})
        if reference is None or reference not in agreement:
            raise AbiOperatorError(
                f"band {band_key}: the operator entries carry the Jacobian agreement against the reference run, and none is "
                f"named (reference_run={reference!r}; the score compares the primary {score.get('primary_run')!r} with "
                f"{sorted(agreement)}); without it the entry would compare the operator with an arbitrary other run")
        classes: dict[str, Any] = {}
        for cls in ADMISSION_CLASSES:
            crow = row["classes"][cls]
            pixel = crow["reference_vs_obs"]
            ff = crow.get("filter_facing", {})
            n = ff.get("n", 0)
            after = ff.get("rmse_after_k")
            if ff.get("verdict") == "INCOMPLETE":
                verdict, reason = "INCOMPLETE", ff.get("reason", "the filter-facing statistics were not measured")
            elif n < 1000:
                verdict, reason = "INCOMPLETE", f"{n} blocks after the stream QC, below the 1000 the gate needs"
            elif after is None:
                verdict, reason = "INCOMPLETE", "no linear fit"
            elif after <= gate_k:
                verdict = "ADMITTED"
                reason = (f"{n} blocks after the stream QC: bias {ff['bias_k']:+.3f} K, rmse {ff['rmse_k']:.3f} K, after obs = "
                          f"{ff['fit_intercept_k']:.3f} + {ff['fit_slope']:.4f} sim the rmse is {after:.3f} K, inside {gate_k} K")
            else:
                verdict = "NOT ADMITTED"
                reason = (f"{n} blocks after the stream QC: bias {ff['bias_k']:+.3f} K, rmse {ff['rmse_k']:.3f} K, after the fit "
                          f"{after:.3f} K, outside {gate_k} K")
            keys = ("n", "bias_k", "rmse_k", "correlation", "fit_intercept_k", "fit_slope", "rmse_after_k")
            against = crow.get(f"{reference}_minus_{operator_run}", {})
            classes[cls] = {
                "verdict": verdict, "reason": reason, "n_blocks": n,
                "bias_correction": ({"form": "obs = intercept + slope * simulated", "intercept_k": ff.get("fit_intercept_k"),
                                     "slope": ff.get("fit_slope")} if verdict == "ADMITTED" else None),
                "observation_error_k": after if verdict == "ADMITTED" else None,
                "filter_facing": {k: ff.get(k) for k in keys} | {
                    "blocks_rejected": ff.get("blocks_rejected"),
                    "residual_shape_after_correction": ff.get("residual_shape_after_correction")},
                "pixel_weighted_all_blocks": {k: pixel.get(k) for k in keys},
                "reference_minus_operator_pixel_weighted": {k: against.get(k) for k in ("n", "bias_k", "rmse_k", "rmse_after_k")},
                # the earlier receipts carried the pixel-weighted numbers under this key; kept so a reader of both sees the change
                "raw": {k: pixel.get(k) for k in ("bias_k", "rmse_k", "correlation", "rmse_after_k")},
            }
        entries["bands"][band_key] = {
            "band": int(band_key),
            "name": BANDS[int(band_key)].name if int(band_key) in BANDS else f"abi-band{int(band_key):02d}",
            "constrains": BANDS[int(band_key)].constrains if int(band_key) in BANDS else None,
            "form": table_band["form"],
            "transmittance_model_vs_reference": {k: table_band.get("validation", {}).get(k) for k in
                                                 ("test_rms_k", "test_bias_k", "test_p99_abs_k", "test_columns")} | {
                "against": "the reference's layer optical depths through the same march and the reference's own per-column "
                           "emissivity: the transmittance model alone, not the operator as shipped"},
            "jacobians_vs_reference": {"reference_run": reference, **agreement[reference]},
            "classes": classes,
            "admitted_classes": [c for c, v in classes.items() if v["verdict"] == "ADMITTED"],
            "acceptance_contract": acceptance_contract(int(band_key), table_band, score["zenith_gate_deg"], qc=qc),
            "vertical_placement_release_1_hpa": (table_band.get("reference_jacobians", {})
                                                 .get("peak_pressure_hpa_percentiles_5_25_50_75_95", [None] * 5)[2]),
        }
    return entries


def four_assessments(*, engineering: dict, statistical: dict, physical: dict, predictive: dict) -> dict:
    """The receipt's four assessments (amendment G), each a dict with a
    ``verdict`` (PASS, FAIL, INCOMPLETE or NOT MEASURED) and its facts;
    engineering validity is the only hard gate and a row that measured
    nothing never reads PASS."""
    out = {}
    for name, row in (("engineering_validity", engineering), ("statistical_consistency", statistical),
                      ("physical_consistency", physical), ("predictive_value", predictive)):
        row = dict(row)
        row.setdefault("verdict", "NOT MEASURED")
        out[name] = row
    return out


__all__ += ["ADMISSION_CLASSES", "FAST_STREAM", "FAST_VARIABLE", "acceptance_contract", "fast_operator_entries", "four_assessments"]
