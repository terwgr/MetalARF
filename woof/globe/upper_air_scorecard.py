"""Upper-air scorecard for WOOF global: the standard deterministic scores
of a global model against an analysis, at every checkpoint hour.

What it measures
----------------
For one checkpoint and one reference product valid at the same instant,
on the run's own Gaussian grid, area-weighted by the grid's quadrature
weights, over four regions (``REGIONS``: the northern extratropics 20N-90N,
the southern extratropics 90S-20S, the tropics 20S-20N, and the globe):

* ``z500``  500 hPa geopotential height, m: bias (model minus reference),
            rmse, mae, and the anomaly correlation;
* ``t850``  850 hPa temperature, K: bias, rmse, mae, anomaly correlation;
* ``w250``  250 hPa vector wind, m/s: the root-mean-square vector error
            ``sqrt(mean(du^2 + dv^2))``, the wind-speed bias and the u and
            v component biases;
* ``w850``  850 hPa vector wind, the same readings;
* ``rh700`` 700 hPa relative humidity, percent: bias, rmse, mae, anomaly
            correlation.

Read at every checkpoint of a run, the scores are growth curves rather
than one number.  A reference whose valid time equals its source cycle is
an ANALYSIS (the reading of record); one valid later than its cycle is a
FORECAST of the reference model at that lead, and its row says so: the
model's distance from another model's forecast is a divergence curve, not
an error.

Definitions, stated so the numbers are reproducible:

* Model fields come from the checkpoint's own arrays on the Gaussian
  grid: theta, vorticity, divergence, log surface pressure and vapor are
  synthesized from their spectral coefficients in float64 (the wind
  through the vector operator the dycore uses), the condensate species are
  grid tracers (schema v3) or synthesized and floored at zero (the
  spectral-tracer era, schema v2).  Temperature is theta times the Exner
  function of the full-level pressure, virtual temperature the dycore's
  ``T (1 + 0.61 qv - condensate)``, and geopotential the dycore's own
  hydrostatic integration (``HybridCoordinate.hydrostatic_geopotential``)
  from the surface geopotential the run integrated with: the analysis
  orography of the run's initial state regridded to the Gaussian grid and
  spectrally truncated, rebuilt here in float64 from that file (its
  SHA-256 must be the one the run receipt recorded, else the reader
  refuses by name).  The run truncated it in float32 on the card; the
  float64 rebuild differs at the 0.01 m level.
* Interpolation to a target pressure is linear in ln p between the two
  full levels that bracket it, column by column (the floor of that
  choice: a column of constant potential temperature, whose temperature
  is exponential in ln p, reads 0.006 K high at 500 hPa on the 40-level
  grid, 0.65 K on the 4-level test grid).  Geopotential is
  interpolated hydrostatically from the bracketing level above with the
  trapezoid of virtual temperature in ln p, which is exact for a column
  whose virtual temperature is linear in ln p, as the dycore's full-level
  geopotential is.  Below the lowest full level and above the surface the
  lowest level's temperature, wind and vapor are held and geopotential
  integrates hydrostatically from the surface with the held virtual
  temperature.  A target below the surface (target pressure above the
  column's surface pressure) or above the top full level is REFUSED for
  that column: the column is masked out of every score at that level,
  and each region row reports the masked area fraction.  The reference
  is masked the same way by its own surface pressure, so a below-ground
  extrapolation of either product never enters a score.
* Reference fields are decoded through the Rust mapped-source engine with
  the run's own analysis mapping (geopotential height, temperature,
  specific humidity, wind on isobaric levels; surface pressure; terrain)
  and regridded bilinearly onto the Gaussian grid with the regridder the
  model initializes with.
* Relative humidity is DERIVED on both sides from specific humidity,
  temperature and pressure with respect to liquid water (Bolton 1980
  saturation vapour pressure), never read from the reference's own RH
  field, whose saturation convention over ice is the reference model's.
* Geopotential height of the model is ``phi / g`` with the model's own
  ``g`` (9.80616), the constant it turned the analysis orography into
  geopotential with, so the surface reads back the analysis terrain
  exactly; the reference's height is the product's gpm (g0 9.80665).  At
  500 hPa the two constants differ by 0.27 m of thickness, below the
  1 m reading floor of any row here.
* The anomaly correlation is the centred correlation of the departures of
  model and reference from a climatology; no climatology file exists for
  this model, so the climatology is the reference's OWN zonal mean at the
  valid time (per Gaussian row, over that row's unmasked columns).  This
  measures the correlation of the eddy fields (departures from the zonal
  mean), which is what the standard AC measures when the climatology is a
  zonal-mean-like field; every output names the climatology it used.

Calibration (``python -m woof.globe.upper_air_scorecard
--calibrate``; ``tests/test_arwen_global_upper_air_scorecard.py`` holds
every bar): the families in CALIBRATION below are planted in both
directions and read back to float64 rounding.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .constants import (
    DRY_AIR_GAS_CONSTANT,
    GRAVITY_M_S2,
    KAPPA,
    REFERENCE_PRESSURE_PA,
)

SCHEMA = "gpuwm.arwen-global-upper-air-scorecard/v1"
RECEIPT_NAME = "arwen-global-receipt.json"

#: (score name, field, target pressure Pa).  The field names are the
#: PressureLevelFields keys: z (m), t (K), u and v (m/s), rh (percent).
TARGETS: tuple[tuple[str, str, float], ...] = (
    ("z500", "z", 50_000.0),
    ("t850", "t", 85_000.0),
    ("w250", "wind", 25_000.0),
    ("w850", "wind", 85_000.0),
    ("rh700", "rh", 70_000.0),
)
LEVELS_PA: tuple[float, ...] = tuple(sorted({level for _, _, level in TARGETS}))
SCALAR_FIELDS = ("z", "t", "u", "v", "rh")
#: Region name -> (south edge, north edge), rows assigned by
#: ``south <= latitude <= north``.
REGIONS: dict[str, tuple[float, float]] = {
    "nh_extratropics": (20.0, 90.0),
    "sh_extratropics": (-90.0, -20.0),
    "tropics": (-20.0, 20.0),
    "global": (-90.0, 90.0),
}
CLIMATOLOGY = "reference zonal mean at the valid time (per Gaussian row, unmasked columns)"
RH_CONVENTION = (
    "derived on both sides from specific humidity, temperature and pressure "
    "with respect to liquid water (Bolton 1980)"
)
EPSILON_WATER = 0.622

CALIBRATION = """
Recorded 2026-09-04 on the CPU test host (Python 3.14, numpy 2.5, float64)
by --calibrate; the test file holds every bar.  Synthetic Gaussian grid
T21 (33 x 66), 40-level surface_stretched coordinate, surface pressure
1013.25 hPa minus a 60 hPa cos(lat) cos(2 lon) pattern, terrain a 300 m
pattern; u and v linear in ln p; temperature isothermal per column with
specific humidity linear in ln p (family A1) or linear in ln p with both
signs of the lapse and constant humidity (family A2), so virtual
temperature is linear in ln p and the closed forms are exact.

Family A, column interpolation and hydrostatics against the closed form
(largest absolute error over every column and every target level):

    family  lapse K/lnp   z m       t K       u m/s     v m/s     rh %
    A1      0             1.9e-11   5.7e-14   3.4e-14   8.9e-16   1.6e-13
    A2      +30           1.9e-11   1.1e-13   3.4e-14   8.9e-16   2.2e-12
    A2      -30           2.0e-11   1.1e-13   3.4e-14   8.9e-16   2.1e-13
    bars: 1e-9 m, 1e-12 K, 1e-12 m/s, 1e-10 percent

Family B, planted differences between a reference and a model built from
it, every region (read back to 2e-15 relative):

    planted                 bias read    rmse read
    z500 +12 m / -12 m      +12 / -12    12
    t850 +1.5 K / -1.5 K    +1.5 / -1.5  1.5
    w250 (du,dv) = (3,-4)   u +3, v -4   rmsve 5;  (-3,4): u -3, v +4, rmsve 5
    w850 (3,-4) / (-3,4)    the same
    rh700 +8 % / -8 %       +8 / -8      8
    z500 +12 m plus a zero-mean texture of rms 5 m: bias 12.000, rmse 13.000

Family C, the reference scored against itself: every bias, rmse, mae and
rmsve reads exactly 0 and every anomaly correlation exactly 1; on the
real GDAS 2026-09-02 00Z analysis on the T255 384 x 768 grid
(--self-check) the same, 0.000e+00 and 0.000e+00.

Family D, anomaly correlation of a planted rotation (model eddy =
cos(theta) x reference eddy + sin(theta) x its zonal quadrature, AC =
cos(theta)):

    theta   0    30        60        90        120        180
    AC read 1.0  0.866025  0.500000  1.1e-15   -0.500000  -1.0   (bar 1e-12)

Family E, refusals in both directions: a planted mountain region whose
surface pressure sits 400 hPa below the pattern (55 to 61 kPa) is masked
at 850 and 700 hPa and not at 500 or 250; the masked global area fraction
reads the planted 0.0774548786 to 1.6e-16 whether the mountain is the
model's or the reference's, and a +50 K (t850) / +30 % (rh700) offset
planted inside it leaves both biases at exactly 0.

Family F, the analysis regrid: a reference field bilinear in latitude and
longitude on a descending-latitude 1 degree grid reads at every Gaussian
node to 5.7e-14 m; ascending latitude the same.

Family G (test file): a T3 checkpoint holding an isothermal solid-body
rotation reads its planted wind, surface pressure and temperature to
1e-9 through the same door the run archives take.

Family H, the representation floor (--floor) against the artifact: the
hour-0 checkpoint of the T255 control (12e9aaf9f, written in float32 by
the card) scored against its own initial GDAS 2026-09-01 00Z analysis,
beside the floor's cold_start row (the same analysis through the cold
start in float64), global / NH / SH / tropics:

    reading            checkpoint hour 0            cold_start floor            difference
    Z500 rmse m        1.741 / 1.850 / 1.560 / 1.795  1.746 / 1.852 / 1.573 / 1.797  0.005 / 0.001 / 0.013 / 0.002
    Z500 bias m        -0.22 / -0.14 / -0.50 / -0.04  -0.26 / -0.17 / -0.54 / -0.09  0.04 / 0.03 / 0.04 / 0.05
    T850 rmse K        0.3443 / 0.2892 / 0.3716 / 0.3638   the same to 1e-4
    W250 rmsve m/s     1.9730 / 1.9198 / 2.1103 / 1.8857   the same to 1e-4
    W850 rmsve m/s     0.7759 / 0.8443 / 0.8142 / 0.6688   the same to 1e-4
    RH700 rmse %       4.252 / 4.371 / 4.695 / 3.669   4.251 / 4.371 / 4.691 / 3.668  0.004 at most

The height residue (0.01 m rmse, 0.05 m bias) is the float32 orography
and log surface pressure the card holds against the float64 rebuild;
the temperature, wind and humidity rows are the artifact's to four
places.  The floor rows of that analysis on the T255 40-level grid
(global rmse or rmsve): vertical Z500 1.67 m, T850 0.28 K, W250 1.85
m/s (speed bias -0.53), W850 0.27 m/s, RH700 2.94 %; spectral 0.45 m,
0.21 K, 0.77 m/s, 0.74 m/s, 3.11 % (the product's own fields with every
column included through the truncation, masked below ground after: the
level-mean fill of the masked columns this row first carried read 0.69 K,
0.77 m/s and 3.14 % on the three masked targets, the ringing of the
fill's step at every mountain edge, and the same 0.45 m and 0.77 m/s on
the two targets no column is masked at); cold_start 1.75 m, 0.34 K, 1.97
m/s, 0.78 m/s, 4.25 %.  The synthetic families (test file) read the
vertical row to 1e-6, the spectral row to the truncation's own loss of
the regridded field (1e-9 relative, and the regrid residue under a 60 kPa
plateau that masks the 850 and 700 hPa targets, where the level-mean fill
read 0.19 m/s of W850), the cold-start row inside the 1 degree bilinear
regrid's residue (0.01 m, 0.01 m/s), and a checkpoint's cache key follows
its content digest and its surface geopotential, not its config hash or
file name (two arms of different code on one config share both).
"""


# --------------------------------------------------------------------------
# thermodynamics
# --------------------------------------------------------------------------




def saturation_vapor_pressure_pa(temperature_k):
    """Bolton (1980) saturation vapour pressure over liquid water, Pa."""
    t = np.asarray(temperature_k, dtype=np.float64)
    return 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))


def relative_humidity_percent(specific_humidity, temperature_k, pressure_pa):
    """RH in percent from specific humidity (kg per kg of moist air)."""
    q = np.asarray(specific_humidity, dtype=np.float64)
    p = np.asarray(pressure_pa, dtype=np.float64)
    vapor = q * p / (EPSILON_WATER + (1.0 - EPSILON_WATER) * q)
    return 100.0 * vapor / saturation_vapor_pressure_pa(temperature_k)


def specific_humidity_from_rh(rh_percent, temperature_k, pressure_pa):
    """The inverse of :func:`relative_humidity_percent` (calibration)."""
    vapor = np.asarray(rh_percent, dtype=np.float64) / 100.0 * saturation_vapor_pressure_pa(temperature_k)
    p = np.asarray(pressure_pa, dtype=np.float64)
    return EPSILON_WATER * vapor / (p - (1.0 - EPSILON_WATER) * vapor)


# --------------------------------------------------------------------------
# pressure-level fields
# --------------------------------------------------------------------------


@dataclass
class PressureLevelFields:
    """Scalar fields at the scorecard's target pressures on one lat-lon
    grid, with the per-level validity mask (False where the column was
    refused at that level) and the row area weights."""

    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    row_weights: np.ndarray
    levels_pa: tuple[float, ...]
    fields: dict[str, dict[float, np.ndarray]]
    valid: dict[float, np.ndarray]
    surface_pressure_pa: np.ndarray
    source: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lat = np.asarray(self.latitude_deg, dtype=np.float64)
        lon = np.asarray(self.longitude_deg, dtype=np.float64)
        w = np.asarray(self.row_weights, dtype=np.float64)
        if lat.ndim != 1 or lon.ndim != 1 or w.shape != lat.shape:
            raise ValueError("latitude_deg, longitude_deg and row_weights must be 1-D with matching rows")
        shape = (lat.size, lon.size)
        self.levels_pa = tuple(float(v) for v in self.levels_pa)
        for name in SCALAR_FIELDS:
            if name not in self.fields:
                raise ValueError(f"pressure-level fields lack {name!r}")
            for level in self.levels_pa:
                arr = np.asarray(self.fields[name][level], dtype=np.float64)
                if arr.shape != shape:
                    raise ValueError(f"{name} at {level:g} Pa has shape {arr.shape}, expected {shape}")
                self.fields[name][level] = arr
        for level in self.levels_pa:
            mask = np.asarray(self.valid[level], dtype=bool)
            if mask.shape != shape:
                raise ValueError(f"valid mask at {level:g} Pa has shape {mask.shape}, expected {shape}")
            self.valid[level] = mask
        ps = np.asarray(self.surface_pressure_pa, dtype=np.float64)
        if ps.shape != shape:
            raise ValueError("surface_pressure_pa shape mismatch")
        self.latitude_deg, self.longitude_deg, self.row_weights = lat, lon, w
        self.surface_pressure_pa = ps

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.latitude_deg.size), int(self.longitude_deg.size)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "row_weights": self.row_weights,
            "levels_pa": np.asarray(self.levels_pa, dtype=np.float64),
            "surface_pressure_pa": self.surface_pressure_pa,
        }
        for name in SCALAR_FIELDS:
            for level in self.levels_pa:
                arrays[f"{name}@{int(round(level))}"] = self.fields[name][level]
        for level in self.levels_pa:
            arrays[f"valid@{int(round(level))}"] = self.valid[level]
        temporary = target.with_name(f".{target.name}.partial-{os.getpid()}")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                __source__=np.asarray(json.dumps(self.source, sort_keys=True)),
                **arrays,
            )
        os.replace(temporary, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "PressureLevelFields":
        with np.load(path, allow_pickle=False) as archive:
            levels = tuple(float(v) for v in archive["levels_pa"])
            fields = {
                name: {level: np.array(archive[f"{name}@{int(round(level))}"]) for level in levels}
                for name in SCALAR_FIELDS
            }
            valid = {level: np.array(archive[f"valid@{int(round(level))}"]) for level in levels}
            return cls(
                latitude_deg=np.array(archive["latitude_deg"]),
                longitude_deg=np.array(archive["longitude_deg"]),
                row_weights=np.array(archive["row_weights"]),
                levels_pa=levels,
                fields=fields,
                valid=valid,
                surface_pressure_pa=np.array(archive["surface_pressure_pa"]),
                source=json.loads(str(archive["__source__"].item())),
            )


# --------------------------------------------------------------------------
# column interpolation
# --------------------------------------------------------------------------


def _bracket(ln_p: np.ndarray, ln_target: float):
    """For a column stack ``ln_p`` (levels top to bottom, ascending along
    axis 0) return the lower and upper bracket indices of ``ln_target``,
    the linear weight of the upper (higher-pressure) level and the mask
    of columns where the target lies inside the stack."""
    count_above = np.sum(ln_p <= ln_target, axis=0)
    n = ln_p.shape[0]
    inside = (count_above >= 1) & (count_above <= n - 1)
    upper = np.clip(count_above, 1, n - 1)
    lower = upper - 1
    lower_index = lower[None]
    upper_index = upper[None]
    ln_lower = np.take_along_axis(ln_p, lower_index, axis=0)[0]
    ln_upper = np.take_along_axis(ln_p, upper_index, axis=0)[0]
    weight = (ln_target - ln_lower) / (ln_upper - ln_lower)
    return lower_index, upper_index, weight, inside


def interpolate_column_fields(
    *,
    p_full: np.ndarray,
    geopotential: np.ndarray,
    temperature: np.ndarray,
    virtual_temperature: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    specific_humidity: np.ndarray,
    surface_pressure: np.ndarray,
    surface_geopotential: np.ndarray,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    row_weights: np.ndarray,
    levels_pa=LEVELS_PA,
    source: dict[str, object] | None = None,
) -> PressureLevelFields:
    """Interpolate full-level columns (axis 0 top to bottom) to the target
    pressures as the module docstring states."""
    p_full = np.asarray(p_full, dtype=np.float64)
    ps = np.asarray(surface_pressure, dtype=np.float64)
    if np.any(np.diff(p_full, axis=0) <= 0.0):
        raise ValueError("full-level pressure must increase from the top down in every column")
    if np.any(p_full[-1] >= ps):
        raise ValueError("the lowest full level must lie above the surface in every column")
    # The surface is appended as one more row: held T, u, v, q; the
    # surface geopotential; so a target between the lowest level and the
    # ground interpolates against the ground.
    ln_p = np.log(np.concatenate([p_full, ps[None]], axis=0))
    tv = np.asarray(virtual_temperature, dtype=np.float64)
    stacks = {
        "t": np.concatenate([temperature, temperature[-1:]], axis=0),
        "u": np.concatenate([u, u[-1:]], axis=0),
        "v": np.concatenate([v, v[-1:]], axis=0),
        "q": np.concatenate([specific_humidity, specific_humidity[-1:]], axis=0),
        "tv": np.concatenate([tv, tv[-1:]], axis=0),
        "phi": np.concatenate([geopotential, np.asarray(surface_geopotential, dtype=np.float64)[None]], axis=0),
    }
    fields: dict[str, dict[float, np.ndarray]] = {name: {} for name in SCALAR_FIELDS}
    valid: dict[float, np.ndarray] = {}
    for level in levels_pa:
        ln_target = math.log(float(level))
        lower, upper, w, inside = _bracket(ln_p, ln_target)

        def at(name: str) -> np.ndarray:
            low = np.take_along_axis(stacks[name], lower, axis=0)[0]
            high = np.take_along_axis(stacks[name], upper, axis=0)[0]
            return low * (1.0 - w) + high * w

        t_here = at("t")
        tv_here = at("tv")
        # Hydrostatic step from the bracketing level above (index lower)
        # down to the target with the trapezoid of Tv in ln p.
        ln_lower = np.take_along_axis(ln_p, lower, axis=0)[0]
        phi_lower = np.take_along_axis(stacks["phi"], lower, axis=0)[0]
        tv_lower = np.take_along_axis(stacks["tv"], lower, axis=0)[0]
        phi_here = phi_lower - DRY_AIR_GAS_CONSTANT * 0.5 * (tv_lower + tv_here) * (ln_target - ln_lower)
        fields["z"][level] = np.where(inside, phi_here / GRAVITY_M_S2, np.nan)
        fields["t"][level] = np.where(inside, t_here, np.nan)
        fields["u"][level] = np.where(inside, at("u"), np.nan)
        fields["v"][level] = np.where(inside, at("v"), np.nan)
        fields["rh"][level] = np.where(
            inside, relative_humidity_percent(at("q"), np.where(inside, t_here, 250.0), float(level)), np.nan
        )
        valid[level] = inside
    return PressureLevelFields(
        latitude_deg=latitude_deg, longitude_deg=longitude_deg, row_weights=row_weights,
        levels_pa=tuple(float(v) for v in levels_pa), fields=fields, valid=valid,
        surface_pressure_pa=ps, source=dict(source or {}),
    )


# --------------------------------------------------------------------------
# the model side
# --------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def read_receipt(run_dir: Path) -> dict:
    path = Path(run_dir) / RECEIPT_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path}: the run receipt is needed for the grid, the vertical tables and the run start")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def receipt_start_utc(receipt: dict) -> datetime:
    options = receipt.get("config", {}).get("native_adapter_options", {})
    start = options.get("start_time_utc") if isinstance(options, dict) else None
    if not start:
        raise ValueError("the run receipt carries no start_time_utc; checkpoints cannot be paired with a valid time")
    value = str(start)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(timezone.utc).replace(tzinfo=None)
    return stamp


def receipt_initial_analysis(receipt: dict) -> tuple[str | None, dict[str, str], str | None]:
    """(mapping path, {input path: sha256}, mapping sha) the run initialized from."""
    initial = receipt.get("initial") or {}
    provenance = initial.get("provenance") or {}
    if provenance.get("mode") != "analysis":
        raise ValueError(
            "the run was not initialized from an analysis; its surface geopotential "
            "cannot be rebuilt from an analysis orography"
        )
    return provenance.get("mapping"), dict(provenance.get("input_sha256") or {}), provenance.get("mapping_sha256")


def build_transform_from_receipt(receipt: dict):
    from woof.globe.spectral.transform import SphericalHarmonicTransform

    cfg = receipt["config"]
    block = receipt["transform"]
    return SphericalHarmonicTransform.create(
        int(cfg["truncation"]),
        nlat=int(block["nlat"]),
        nlon=int(block["nlon"]),
        dealias_factor=float(cfg["dealias_factor"]),
        radius_m=float(block["radius_m"]),
        backend="numpy",
        precision="float64",
    )


def surface_geopotential_from_frame(frame, transform) -> np.ndarray:
    """The model's surface geopotential: the analysis terrain regridded to
    the Gaussian grid and spectrally truncated, as analysis_initial does."""
    from .analysis_initial import _global_regridder

    regrid = _global_regridder(frame.latitude, frame.longitude, transform.grid)
    phi_src = GRAVITY_M_S2 * regrid(frame.fields["terrain_height"].values)
    return np.asarray(transform.inverse(transform.forward(phi_src)), dtype=np.float64)


def resolve_mapping(spec: str | None, receipt: dict) -> Path:
    from .analysis_initial import MappingTablesDisagree, resolve_analysis_mapping

    if spec:
        return resolve_analysis_mapping(spec)
    recorded, _inputs, _sha = receipt_initial_analysis(receipt)
    if recorded and Path(recorded).exists():
        return Path(recorded)
    name = Path(recorded).name if recorded else ""
    if name:
        # The receipt recorded an absolute path from the machine that ran
        # the forecast.  The FILE NAME is still a key both authority
        # tables answer, so a run scored on a second machine resolves
        # through the same resolver every other command uses rather than
        # by joining the name onto one directory.
        try:
            return resolve_analysis_mapping(name)
        except MappingTablesDisagree:
            # BOTH tables answer this name, with different bytes.  Falling
            # through to the refusal below would call a row that has MOVED
            # a file that is not on this machine, and send a reader looking
            # for something that is present twice.  That confusion is the
            # whole reason the resolver raises its own class here.
            raise
        except (FileNotFoundError, ValueError):
            pass
    raise FileNotFoundError(
        f"the run's analysis mapping {recorded!r} is not on this machine; pass --mapping"
    )


def decode_reference(mapping: Path, path: Path):
    """One reference product through this package's decode door, as the
    pair ``(frame, decode receipt)``.

    The receipt is the block ``mapped_source_compat.decode_through_engine``
    produced: which mechanism placed the frame stream's scratch, whether
    the installed engine's soil-only preserve_mask narrowing had to be
    adapted for this document, the fields it was adapted for, and the
    engine version that did it.  It comes back as a PAIR rather than being
    dropped here because every caller keeps a provenance block, and one
    that records the mapping digest but not the mechanism that read it
    leaves a reader inferring the mechanism from a version number.
    """
    from .mapped_source_compat import decode_through_engine

    decoded = decode_through_engine(mapping, [Path(path)])
    frames = decoded.frames
    if len(frames) != 1:
        raise ValueError(f"{path}: expected one valid time, decoded {len(frames)}")
    return frames[0], decoded.receipt


def surface_geopotential_for_run(
    receipt: dict, transform, initial_analysis: str | Path | None, mapping: Path,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Rebuild the run's surface geopotential from its initial analysis
    file (``initial_analysis`` or the receipt's recorded path), refusing a
    file whose SHA-256 is not the one the receipt recorded."""
    _mapping, inputs, _sha = receipt_initial_analysis(receipt)
    if not inputs:
        raise ValueError("the run receipt records no initial analysis input hash")
    candidates = [Path(initial_analysis)] if initial_analysis else [Path(p) for p in inputs]
    path = next((c for c in candidates if c.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "the run's initial analysis file is not on this machine "
            f"({', '.join(str(c) for c in candidates)}); pass --initial-analysis"
        )
    digest = _sha256_file(path)
    if digest not in set(inputs.values()):
        raise ValueError(
            f"{path}: SHA-256 {digest[:16]} is not the initial analysis the run "
            f"receipt recorded ({', '.join(v[:16] for v in inputs.values())}); the "
            "surface geopotential would not be the one the run integrated with"
        )
    provenance = {
        "initial_analysis": str(path), "sha256": digest,
        "construction": "analysis terrain regridded bilinearly to the Gaussian grid, "
                        "times g, spectrally truncated, float64 rebuild of the run's float32 field",
    }
    nlat, nlon = transform.grid.shape
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"phi-surface-{digest[:16]}-{nlat}x{nlon}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as archive:
                return np.array(archive["phi"]), {**provenance, "cache": str(cache)}
    frame, decode = decode_reference(mapping, path)
    provenance["decode"] = decode
    phi = surface_geopotential_from_frame(frame, transform)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, phi=phi)
        provenance["cache"] = str(cache)
    return phi, provenance


class ModelReader:
    """Pressure-level fields of a run's checkpoints on the run's own grid."""

    def __init__(self, receipt: dict, surface_geopotential: np.ndarray, *, levels_pa=LEVELS_PA):
        from woof.globe.spectral.vector import VorticityDivergenceOperator

        from .vertical import HybridCoordinate

        self.transform = build_transform_from_receipt(receipt)
        cfg = receipt["config"]
        self.vertical = HybridCoordinate(
            np.asarray(cfg["a_half_pa"], dtype=np.float64),
            np.asarray(cfg["b_half"], dtype=np.float64),
        )
        self.vector = VorticityDivergenceOperator(self.transform)
        self.surface_geopotential = np.asarray(surface_geopotential, dtype=np.float64)
        if self.surface_geopotential.shape != self.transform.grid.shape:
            raise ValueError(
                f"surface geopotential shape {self.surface_geopotential.shape} is not the "
                f"grid's {self.transform.grid.shape}"
            )
        self.levels_pa = tuple(float(v) for v in levels_pa)
        # The run start pairs checkpoints with valid times; a receipt
        # without one (a synthetic floor receipt) can still read columns,
        # and sample() carries receipt_start_utc's own refusal.
        self._receipt = receipt
        options = receipt.get("config", {}).get("native_adapter_options", {})
        self.start_utc = receipt_start_utc(receipt) if isinstance(options, dict) and options.get("start_time_utc") else None
        self.config_hash = receipt.get("config_hash")

    def _grid_field(self, value: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(value):
            return np.asarray(self.transform.inverse(value.astype(np.complex128)), dtype=np.float64)
        return np.asarray(value, dtype=np.float64)

    def column_fields(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The full-level column fields the interpolation takes, as the
        dycore defines them."""
        theta = self._grid_field(arrays["atmosphere__theta"])
        qv = self._grid_field(arrays["atmosphere__qv"])
        condensate = np.zeros_like(theta)
        for name in ("qc", "qr", "qi", "qs", "qg"):
            key = f"atmosphere__{name}"
            if key in arrays:
                condensate += np.maximum(self._grid_field(arrays[key]), 0.0)
        logps = self._grid_field(arrays["atmosphere__log_surface_pressure"])
        ps = np.exp(logps)
        pressure = self.vertical.pressure(ps, self.transform.backend)
        p_full = np.asarray(pressure["p_full"], dtype=np.float64)
        p_half = np.asarray(pressure["p_half"], dtype=np.float64)
        temperature = theta * (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
        virtual = temperature * (1.0 + 0.61 * qv - condensate)
        u, v = self.vector.wind_from_vordiv(
            arrays["atmosphere__vorticity"].astype(np.complex128),
            arrays["atmosphere__divergence"].astype(np.complex128),
        )
        geopotential = self.vertical.hydrostatic_geopotential(
            virtual, self.surface_geopotential, p_half
        )
        return {
            "p_full": p_full, "geopotential": np.asarray(geopotential, dtype=np.float64),
            "temperature": temperature, "virtual_temperature": virtual,
            "u": np.asarray(u, dtype=np.float64), "v": np.asarray(v, dtype=np.float64),
            "specific_humidity": qv, "surface_pressure": ps,
        }

    def sample(self, path: str | Path) -> PressureLevelFields:
        from .checkpoint import read_checkpoint

        metadata, arrays = read_checkpoint(path)
        columns = self.column_fields(arrays)
        start = self.start_utc if self.start_utc is not None else receipt_start_utc(self._receipt)
        valid_time = start + timedelta(seconds=float(metadata["time_s"]))
        grid = self.transform.grid
        return interpolate_column_fields(
            **columns,
            surface_geopotential=self.surface_geopotential,
            latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg,
            row_weights=grid.quadrature_weights,
            levels_pa=self.levels_pa,
            source={
                "kind": "model", "checkpoint": str(path), "schema": metadata.get("schema"),
                "step": int(metadata["step"]), "time_s": float(metadata["time_s"]),
                "valid_time": valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "config_hash": metadata.get("config_hash"),
            },
        )


# --------------------------------------------------------------------------
# the reference side
# --------------------------------------------------------------------------


def reference_fields(frame, grid, *, levels_pa=LEVELS_PA, path: str | None = None,
                     decode: dict | None = None) -> PressureLevelFields:
    """A decoded isobaric analysis or forecast frame on the Gaussian grid
    at the target pressures, masked below its own surface.

    ``decode`` is the receipt :func:`decode_reference` returned beside the
    frame; it lands in the ``source`` block this writes, so a cached
    reference and every scorecard that reads one say which decoder path
    produced the bytes.  ``None`` when the frame did not come from a
    decode, which is the synthetic frames the calibration builds."""
    from .analysis_initial import _global_regridder

    needed = ("geopotential_height", "air_temperature", "specific_humidity",
              "eastward_wind", "northward_wind", "surface_pressure", "terrain_height")
    missing = [name for name in needed if name not in frame.fields]
    if missing:
        raise ValueError(f"reference frame lacks {', '.join(missing)}")
    if getattr(frame, "vertical_kind", "pressure") != "pressure":
        raise ValueError(f"reference vertical kind {frame.vertical_kind!r} is not 'pressure'")
    levels = np.asarray(frame.vertical_values, dtype=np.float64)
    regrid = _global_regridder(frame.latitude, frame.longitude, grid)
    ps = regrid(frame.fields["surface_pressure"].values)
    fields: dict[str, dict[float, np.ndarray]] = {name: {} for name in SCALAR_FIELDS}
    valid: dict[float, np.ndarray] = {}
    for level in levels_pa:
        index = np.where(np.abs(levels - float(level)) < 0.5)[0]
        if index.size != 1:
            raise ValueError(
                f"reference carries no {float(level):g} Pa level (levels {levels.tolist()}); "
                "the scorecard reads the product's own isobaric levels, never an interpolation"
            )
        k = int(index[0])
        inside = ps > float(level)
        t = regrid(frame.fields["air_temperature"].values[k])
        q = np.clip(regrid(frame.fields["specific_humidity"].values[k]), 0.0, None)
        fields["z"][level] = np.where(inside, regrid(frame.fields["geopotential_height"].values[k]), np.nan)
        fields["t"][level] = np.where(inside, t, np.nan)
        fields["u"][level] = np.where(inside, regrid(frame.fields["eastward_wind"].values[k]), np.nan)
        fields["v"][level] = np.where(inside, regrid(frame.fields["northward_wind"].values[k]), np.nan)
        fields["rh"][level] = np.where(inside, relative_humidity_percent(q, np.where(inside, t, 250.0), float(level)), np.nan)
        valid[level] = inside
    valid_time = getattr(frame, "valid_time", None)
    cycle = getattr(frame, "source_cycle", None)
    lead_h = None
    if valid_time is not None and cycle is not None:
        lead_h = (valid_time - cycle).total_seconds() / 3600.0
    return PressureLevelFields(
        latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg,
        row_weights=grid.quadrature_weights, levels_pa=tuple(float(v) for v in levels_pa),
        fields=fields, valid=valid, surface_pressure_pa=ps,
        source={
            "kind": "analysis" if lead_h == 0.0 else "forecast",
            "product": Path(path).name.split(".")[0] if path else "reference",
            "path": path, "lead_h": lead_h,
            "valid_time": None if valid_time is None else valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cycle": None if cycle is None else cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input_sha256": dict(getattr(frame, "input_sha256", {}) or {}),
            "mapping_sha256": getattr(frame, "mapping_sha256", None),
            "decode": decode,
            "regrid": "bilinear from the product's regular grid to the Gaussian grid (analysis_initial._global_regridder)",
        },
    )


# --------------------------------------------------------------------------
# scores
# --------------------------------------------------------------------------


def region_mask(latitude_deg: np.ndarray, region: str) -> np.ndarray:
    south, north = REGIONS[region]
    lat = np.asarray(latitude_deg, dtype=np.float64)
    return (lat >= south) & (lat <= north)


def _weights(plf: PressureLevelFields, level: float, both_valid: np.ndarray, region: str) -> np.ndarray:
    rows = region_mask(plf.latitude_deg, region)
    w = np.repeat(plf.row_weights[:, None], plf.shape[1], axis=1) * rows[:, None]
    return np.where(both_valid, w, 0.0)


def zonal_mean_climatology(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-row mean over the unmasked columns, broadcast to the field."""
    count = np.sum(valid, axis=1)
    total = np.sum(np.where(valid, values, 0.0), axis=1)
    mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return np.repeat(mean[:, None], values.shape[1], axis=1)


def _scalar_scores(model: np.ndarray, reference: np.ndarray, w: np.ndarray, climatology: np.ndarray) -> dict[str, float]:
    total = float(np.sum(w))
    if total <= 0.0:
        return {"bias": None, "rmse": None, "mae": None, "anomaly_correlation": None, "n": 0}
    diff = np.where(w > 0, model - reference, 0.0)
    bias = float(np.sum(w * diff) / total)
    rmse = float(math.sqrt(max(0.0, np.sum(w * diff * diff) / total)))
    mae = float(np.sum(w * np.abs(diff)) / total)
    fa = np.where(w > 0, model - climatology, 0.0)
    aa = np.where(w > 0, reference - climatology, 0.0)
    fa = fa - np.sum(w * fa) / total
    aa = aa - np.sum(w * aa) / total
    num = float(np.sum(w * fa * aa))
    den = math.sqrt(float(np.sum(w * fa * fa)) * float(np.sum(w * aa * aa)))
    ac = num / den if den > 0.0 else None
    return {"bias": bias, "rmse": rmse, "mae": mae, "anomaly_correlation": ac, "n": int(np.count_nonzero(w))}


def _wind_scores(mu, mv, ru, rv, w) -> dict[str, float]:
    total = float(np.sum(w))
    if total <= 0.0:
        return {"rmsve": None, "speed_bias": None, "speed_rmse": None, "u_bias": None, "v_bias": None, "n": 0}
    du = np.where(w > 0, mu - ru, 0.0)
    dv = np.where(w > 0, mv - rv, 0.0)
    ms = np.hypot(np.where(w > 0, mu, 0.0), np.where(w > 0, mv, 0.0))
    rs = np.hypot(np.where(w > 0, ru, 0.0), np.where(w > 0, rv, 0.0))
    ds = ms - rs
    return {
        "rmsve": float(math.sqrt(max(0.0, np.sum(w * (du * du + dv * dv)) / total))),
        "speed_bias": float(np.sum(w * ds) / total),
        "speed_rmse": float(math.sqrt(max(0.0, np.sum(w * ds * ds) / total))),
        "u_bias": float(np.sum(w * du) / total),
        "v_bias": float(np.sum(w * dv) / total),
        "n": int(np.count_nonzero(w)),
    }


def score_pair(model: PressureLevelFields, reference: PressureLevelFields, *, regions=tuple(REGIONS)) -> dict[str, object]:
    """Every target over every region for one model/reference pair on the
    same grid."""
    if model.shape != reference.shape or not np.allclose(model.latitude_deg, reference.latitude_deg):
        raise ValueError("model and reference fields must share one grid")
    if not np.allclose(model.row_weights, reference.row_weights):
        raise ValueError("model and reference fields must share one set of row weights")
    out: dict[str, object] = {}
    for name, kind, level in TARGETS:
        if level not in model.valid or level not in reference.valid:
            raise ValueError(f"{name}: {level:g} Pa is not among the interpolated levels")
        both = model.valid[level] & reference.valid[level]
        rows = {}
        for region in regions:
            w = _weights(model, level, both, region)
            region_rows = region_mask(model.latitude_deg, region)
            region_area = float(np.sum(model.row_weights[region_rows]) * model.shape[1])
            masked_fraction = 1.0 - float(np.sum(w)) / region_area if region_area > 0 else None
            if kind == "wind":
                scores = _wind_scores(
                    model.fields["u"][level], model.fields["v"][level],
                    reference.fields["u"][level], reference.fields["v"][level], w,
                )
            else:
                clim = zonal_mean_climatology(reference.fields[kind][level], both)
                scores = _scalar_scores(model.fields[kind][level], reference.fields[kind][level], w, clim)
            scores["masked_fraction"] = masked_fraction
            rows[region] = scores
        out[name] = {"field": kind, "level_pa": float(level), "regions": rows}
    return out


# --------------------------------------------------------------------------
# a run against its references
# --------------------------------------------------------------------------


_WORKER: dict[str, object] = {}


def _worker_init(receipt: dict, phi: np.ndarray, levels: tuple[float, ...]) -> None:
    _WORKER["reader"] = ModelReader(receipt, phi, levels_pa=levels)


def _model_task(args: tuple[str, str]) -> str:
    path, cache = args
    reader = _WORKER["reader"]
    reader.sample(path).save(cache)
    return cache


def _reference_task(args: tuple[str, str, str, int, int, tuple[float, ...]]) -> str:
    path, cache, mapping, nlat, nlon, levels = args
    from woof.globe.spectral.grid import GaussianGrid

    grid = GaussianGrid.for_shape(int(nlat), int(nlon))
    frame, decode = decode_reference(Path(mapping), Path(path))
    reference_fields(frame, grid, levels_pa=levels, path=path,
                     decode=decode).save(cache)
    return cache


def _parse_valid(text: str | None) -> datetime | None:
    if not text:
        return None
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def checkpoint_content_digest(path: str | Path) -> str:
    """The checkpoint's own ``self_sha256`` (the digest of its metadata,
    which carries the SHA-256 of every array it holds), read from the
    metadata member alone, without loading the arrays."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "__metadata__" not in archive:
            raise ValueError(f"checkpoint {source} has no metadata")
        metadata = json.loads(str(archive["__metadata__"].item()))
    digest = metadata.get("self_sha256") if isinstance(metadata, dict) else None
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"checkpoint {source} carries no self_sha256; its fields cannot be cached by content")
    return digest


def model_cache_path(cache_dir: str | Path, checkpoint: str | Path, surface_geopotential_digest: str) -> Path:
    """Where one checkpoint's pressure-level fields are cached: keyed by
    the checkpoint's content digest and the initial analysis its surface
    geopotential was rebuilt from, never by the run's config hash or the
    file name.  Two arms of different code and one config share a config
    hash and name their checkpoints alike, and a cache keyed by those
    handed the second arm the first arm's fields without a word."""
    return Path(cache_dir) / f"model-{checkpoint_content_digest(checkpoint)[:16]}-phi{str(surface_geopotential_digest)[:8]}.npz"


REFERENCE_SKILL_NOTE = (
    "the reference model's own forecast scored against the analysis valid at the same instant, on the "
    "run's grid by the same instrument and masks: the bar the run's analysis rows are read against "
    "(the forecast keeps the product's own scales, the run's fields stop at its truncation)"
)


def reference_skill_rows(references: list[PressureLevelFields], start: datetime) -> list[dict]:
    """Every (forecast, analysis) pair of reference products valid at one
    instant, scored forecast against analysis, one row per pair with the
    hour counted from ``start``: the reference model's own error at the
    run's leads, read by the same instrument."""
    by_time: dict[datetime, list[PressureLevelFields]] = {}
    for plf in references:
        when = _parse_valid(plf.source.get("valid_time"))
        if when is not None:
            by_time.setdefault(when, []).append(plf)
    rows = []
    for when in sorted(by_time):
        group = by_time[when]
        analyses = [r for r in group if r.source.get("kind") == "analysis"]
        forecasts = [r for r in group if r.source.get("kind") == "forecast"]
        for forecast in forecasts:
            for analysis in analyses:
                rows.append({
                    "hour": (when - start).total_seconds() / 3600.0,
                    "valid_time": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "forecast": forecast.source, "reference": analysis.source,
                    "scores": score_pair(forecast, analysis),
                })
    return rows


def measure_run(
    run_dir: Path,
    references: list[Path],
    *,
    cache_dir: Path,
    workers: int = 1,
    label: str = "",
    initial_analysis: Path | None = None,
    mapping: str | None = None,
    checkpoints: list[Path] | None = None,
    levels_pa=LEVELS_PA,
) -> dict[str, object]:
    receipt = read_receipt(run_dir)
    mapping_path = resolve_mapping(mapping, receipt)
    transform = build_transform_from_receipt(receipt)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    phi, phi_provenance = surface_geopotential_for_run(
        receipt, transform, initial_analysis, mapping_path, cache_dir=cache_dir
    )
    del transform
    paths = sorted(
        Path(p) for p in (checkpoints or glob.glob(str(Path(run_dir) / "arwen_global_step*.npz")))
    )
    if not paths:
        raise FileNotFoundError(f"{run_dir}: no arwen_global_step*.npz checkpoints")
    nlat, nlon = int(receipt["transform"]["nlat"]), int(receipt["transform"]["nlon"])
    levels = tuple(float(v) for v in levels_pa)
    model_caches = {path: model_cache_path(cache_dir, path, phi_provenance["sha256"]) for path in paths}
    model_jobs = []
    for path in paths:
        cache = model_caches[path]
        if not cache.exists():
            model_jobs.append((str(path), str(cache)))
    reference_jobs = []
    reference_caches: list[Path] = []
    for ref in references:
        digest = _sha256_file(ref)
        cache = cache_dir / f"ref-{digest[:16]}-{nlat}x{nlon}.npz"
        reference_caches.append(cache)
        if not cache.exists():
            reference_jobs.append((str(ref), str(cache), str(mapping_path), nlat, nlon, levels))
    if workers > 1 and (model_jobs or reference_jobs):
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(receipt, phi, levels)) as pool:
            for _ in pool.map(_reference_task, reference_jobs):
                pass
            for _ in pool.map(_model_task, model_jobs):
                pass
    else:
        for job in reference_jobs:
            _reference_task(job)
        if model_jobs:
            _worker_init(receipt, phi, levels)
            for job in model_jobs:
                _model_task(job)
    refs_by_time: dict[datetime, list[PressureLevelFields]] = {}
    for cache in reference_caches:
        plf = PressureLevelFields.load(cache)
        when = _parse_valid(plf.source.get("valid_time"))
        if when is None:
            raise ValueError(f"{cache}: reference carries no valid time")
        refs_by_time.setdefault(when, []).append(plf)
    start = receipt_start_utc(receipt)
    rows = []
    unpaired = []
    for path in paths:
        model = PressureLevelFields.load(model_caches[path])
        when = _parse_valid(model.source["valid_time"])
        hour = (when - start).total_seconds() / 3600.0
        matches = refs_by_time.get(when, [])
        if not matches:
            unpaired.append({"checkpoint": str(path), "valid_time": model.source["valid_time"], "hour": hour})
            continue
        for ref in matches:
            rows.append({
                "hour": hour, "step": model.source["step"], "time_s": model.source["time_s"],
                "valid_time": model.source["valid_time"], "checkpoint": str(path),
                "reference": ref.source, "scores": score_pair(model, ref),
            })
    rows.sort(key=lambda r: (r["hour"], r["reference"].get("kind", ""), r["reference"].get("path") or ""))
    skill = reference_skill_rows([plf for group in refs_by_time.values() for plf in group], start)
    return {
        "schema": SCHEMA,
        "label": label,
        "run_dir": str(run_dir),
        "config_hash": receipt.get("config_hash"),
        "integrator": receipt.get("config", {}).get("integrator"),
        "dt_s": receipt.get("config", {}).get("dt_s"),
        "start_time_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "grid": {"kind": "gaussian", "nlat": nlat, "nlon": nlon, "truncation": receipt["config"]["truncation"]},
        "targets": [{"name": n, "field": k, "level_pa": l} for n, k, l in TARGETS],
        "regions": {name: list(edges) for name, edges in REGIONS.items()},
        "climatology": CLIMATOLOGY,
        "relative_humidity": RH_CONVENTION,
        "geopotential_height": (
            f"model phi / g with the model's g {GRAVITY_M_S2}; reference gpm (g0 9.80665); "
            "0.27 m of 500 hPa thickness between the two conventions"
        ),
        "interpolation": "linear in ln p between bracketing full levels; geopotential hydrostatic (trapezoid of Tv in ln p) from the level above; below-ground and above-top targets refused per column",
        "surface_geopotential": phi_provenance,
        "mapping": str(mapping_path),
        "rows": rows,
        "unpaired_checkpoints": unpaired,
        "reference_skill": skill,
        "reference_skill_note": REFERENCE_SKILL_NOTE,
        "reference_files": [str(p) for p in references],
    }


# --------------------------------------------------------------------------
# the representation floor of an analysis on the run's grid
# --------------------------------------------------------------------------


def floor_receipt_from_config(cfg, *, mapping: str | None = None) -> dict[str, object]:
    """A receipt-shaped document for a config that has not run: what the
    floor and the checkpoint reader take from a run receipt (the grid,
    the vertical tables, the start time, the analysis mapping), built
    from the config alone so a candidate level set can be priced against
    the analysis before a card runs it.  The document says so
    (``"source": "config"``), carries no run and no checkpoint provenance,
    and the transform is the numpy float64 one the instrument reads with,
    the same geometry the run's would have."""
    from woof.globe.spectral.transform import SphericalHarmonicTransform

    from .analysis_initial import resolve_analysis_mapping

    transform = SphericalHarmonicTransform.create(
        int(cfg.truncation), nlat=cfg.nlat, nlon=cfg.nlon,
        dealias_factor=float(cfg.dealias_factor), backend="numpy", precision="float64",
    )
    spec = mapping or getattr(cfg, "analysis_mapping", None)
    resolved = str(resolve_analysis_mapping(spec)) if spec else None
    options = dict(getattr(cfg, "native_adapter_options", None) or {})
    return {
        "schema": SCHEMA + "/config-receipt",
        "source": "config",
        "name": cfg.name,
        "config_hash": cfg.config_hash,
        "config": {
            "truncation": int(cfg.truncation),
            "dealias_factor": float(cfg.dealias_factor),
            "a_half_pa": [float(v) for v in cfg.a_half_pa],
            "b_half": [float(v) for v in cfg.b_half],
            "nlev": int(cfg.vertical.nlev),
            "vertical_coordinate": cfg.vertical_coordinate,
            "native_adapter_options": options,
            "analysis_mapping": spec,
        },
        "transform": dict(transform.geometry_identity),
        "initial": {"provenance": {"mode": "analysis", "mapping": resolved, "input_sha256": {}}},
    }


def vertical_round_trip(
    vertical, backend, *, ln_source: np.ndarray, t_src: np.ndarray, q_src: np.ndarray,
    u_src: np.ndarray, v_src: np.ndarray, ps: np.ndarray, phi_s: np.ndarray, grid,
    levels_pa=LEVELS_PA, tag: str = "vertical",
) -> PressureLevelFields:
    """The analysis's isobaric columns put on ``vertical``'s full levels
    at the analysis surface pressure (linear in ln p, as the cold start
    does) and read back at the target pressures by this instrument, with
    geopotential integrated hydrostatically from the analysis terrain: the
    vertical row of :func:`representation_floor`, shared with the level-set
    calibration so both read one arithmetic."""
    from .analysis_initial import _to_model_levels

    pressure = vertical.pressure(ps, backend)
    p_full = np.asarray(pressure["p_full"], dtype=np.float64)
    p_half = np.asarray(pressure["p_half"], dtype=np.float64)
    ln_target = np.log(p_full)
    t = _to_model_levels(t_src, ln_source, ln_target, extrapolate_below=True)
    q = np.clip(_to_model_levels(q_src, ln_source, ln_target), 0.0, None)
    u = _to_model_levels(u_src, ln_source, ln_target)
    v = _to_model_levels(v_src, ln_source, ln_target)
    tv = t * (1.0 + 0.61 * q)
    phi = np.asarray(vertical.hydrostatic_geopotential(tv, phi_s, p_half), dtype=np.float64)
    return interpolate_column_fields(
        p_full=p_full, geopotential=phi, temperature=t, virtual_temperature=tv,
        u=u, v=v, specific_humidity=q, surface_pressure=ps, surface_geopotential=phi_s,
        latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg, row_weights=grid.quadrature_weights,
        levels_pa=levels_pa, source={"kind": "model", "floor": tag},
    )


def kink_read_factor(vertical, target_pa: float, surface_pressure_pa: float = 101_325.0) -> float:
    """What the vertical round trip reads of a slope change at ``target_pa``.

    A profile linear in ln p on either side of a target pressure with
    slopes ``s_above`` and ``s_below`` (per unit ln p) is exact on every
    full level; reading the target back linearly in ln p between the
    bracketing full levels ``p_a < p_t < p_b`` then misses by ``(s_below
    - s_above) * f`` with ``f = (x_b - x_t) (x_t - x_a) / (x_b - x_a)``
    in ``x = ln p``.  ``f`` is the level set's own number at the target:
    0.0363 for the 40-level surface_stretched stack at 250 hPa (55 hPa
    layers), 0.0093 for the 48-level jet_refined one (23 hPa), 0 when a
    full level sits on the target.  The level-set calibration family
    plants a jet with a known slope change and reads this back."""
    ps = float(surface_pressure_pa)
    p_half = vertical.a_half_pa + vertical.b_half * ps
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    x = np.log(p_full)
    x_t = math.log(float(target_pa))
    if x_t <= x[0] or x_t >= x[-1]:
        raise ValueError(f"{target_pa:g} Pa is outside the full levels {p_full[0]:.1f} to {p_full[-1]:.1f} Pa")
    below = int(np.searchsorted(x, x_t))
    x_a, x_b = float(x[below - 1]), float(x[below])
    return (x_b - x_t) * (x_t - x_a) / (x_b - x_a)


def level_set_floor_family(
    vertical, *, truncation: int = 21, jet_slope_m_s_per_lnp: float = 25.0, target_pa: float = 25_000.0,
) -> dict[str, object]:
    """Family G, one level set: flat-surface columns (``ps`` = 101325 Pa
    everywhere, terrain 0) on isobaric source levels, put through the
    vertical round trip.  G1 is linear in ln p in every field (the round
    trip must read 0 on every target); G2 adds a jet to ``u``, linear in
    ln p on either side of ``target_pa`` with slope ``+S`` above and
    ``-S`` below (a maximum at the target), so the read-back at the
    target misses by exactly ``-2 S f`` with ``f`` the level set's
    :func:`kink_read_factor`: the expected wind-speed bias is ``-2 S f``
    and the rmsve ``2 S f``, uniform over the sphere, and every other
    target and field still reads 0.  Both directions: a level set that
    resolves the jet must read a smaller number and the number is
    predicted, not fitted."""
    from woof.globe.spectral.transform import SphericalHarmonicTransform

    transform = SphericalHarmonicTransform.create(int(truncation), backend="numpy", precision="float64")
    grid = transform.grid
    lat2, lon2 = grid.mesh()
    ps = np.full(lat2.shape, 101_325.0)
    phi_s = np.zeros(lat2.shape)
    source_pa = np.asarray([10_000.0, 15_000.0, 20_000.0, 25_000.0, 30_000.0, 40_000.0, 50_000.0,
                            60_000.0, 70_000.0, 85_000.0, 92_500.0, 100_000.0])
    ln_source = np.log(source_pa)
    x = (ln_source - math.log(101_325.0))[:, None, None]
    t = 288.0 + 3.0 * np.cos(lat2)[None] + 28.0 * x
    q = np.full(t.shape, 0.002)
    u_linear = 12.0 + 6.0 * np.cos(lat2)[None] - 15.0 * x
    v = 2.0 * np.sin(2.0 * lon2)[None] * np.cos(lat2)[None] + 0.0 * x
    S = float(jet_slope_m_s_per_lnp)
    x_t = math.log(float(target_pa)) - math.log(101_325.0)
    jet = 30.0 - S * np.abs(x - x_t)
    u_jet = u_linear + jet
    f = kink_read_factor(vertical, target_pa, 101_325.0)
    rows = {}
    for label, u_src in (("G1_linear", u_linear), ("G2_jet", u_jet)):
        plf = vertical_round_trip(
            vertical, transform.backend, ln_source=ln_source, t_src=t, q_src=q, u_src=u_src, v_src=v,
            ps=ps, phi_s=phi_s, grid=grid, tag=label,
        )
        truth_fields = {name: {} for name in SCALAR_FIELDS}
        valid = {}
        for level in LEVELS_PA:
            xl = math.log(float(level)) - math.log(101_325.0)
            tt = 288.0 + 3.0 * np.cos(lat2) + 28.0 * xl
            qq = np.full(lat2.shape, 0.002)
            uu = 12.0 + 6.0 * np.cos(lat2) - 15.0 * xl
            if label == "G2_jet":
                uu = uu + 30.0 - S * abs(xl - x_t)
            c0 = (288.0 + 3.0 * np.cos(lat2)) * (1.0 + 0.61 * 0.002)
            c1 = 28.0 * (1.0 + 0.61 * 0.002)
            truth_fields["z"][level] = -DRY_AIR_GAS_CONSTANT * (c0 * xl + c1 * xl * xl / 2.0) / GRAVITY_M_S2
            truth_fields["t"][level] = tt
            truth_fields["u"][level] = uu
            truth_fields["v"][level] = 2.0 * np.sin(2.0 * lon2) * np.cos(lat2)
            truth_fields["rh"][level] = relative_humidity_percent(qq, tt, np.full(lat2.shape, float(level)))
            valid[level] = np.ones(lat2.shape, dtype=bool)
        truth = PressureLevelFields(
            latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg, row_weights=grid.quadrature_weights,
            levels_pa=tuple(float(v) for v in LEVELS_PA), fields=truth_fields, valid=valid,
            surface_pressure_pa=np.array(ps, copy=True), source={"kind": "analysis", "synthetic": True},
        )
        scores = score_pair(plf, truth)
        rows[label] = {
            "read": {
                name: {region: {k: r[k] for k in ("rmse", "bias", "rmsve", "speed_bias") if k in r}
                       for region, r in scores[name]["regions"].items()}
                for name in scores
            },
        }
    rows["G2_jet"]["expected"] = {
        "w250_rmsve": 2.0 * S * f, "w250_speed_bias": -2.0 * S * f, "kink_read_factor": f,
        "every_other_target_and_field": 0.0,
    }
    return {"nlev": int(vertical.nlev), "kink_read_factor_at_target": f, "target_pa": float(target_pa),
            "jet_slope_m_s_per_lnp": S, "rows": rows}


def representation_floor(frame, receipt: dict, *, levels_pa=LEVELS_PA, path: str | None = None,
                         decode: dict | None = None) -> dict[str, object]:
    """What the run's representation of ``frame`` costs before any
    forecast: the analysis regridded to the Gaussian grid scored against
    (1) ``vertical``, its own round trip through the run's hybrid levels
    (fields interpolated linearly in ln p onto the full levels of the
    analysis's own surface pressure and read back by this instrument,
    geopotential integrated hydrostatically from the analysis terrain),
    (2) ``spectral``, its own truncation at the run's truncation (scalar
    fields through the transform, wind through the vector operator), and
    (3) ``cold_start``, the analysis through the run's own cold start as
    ``analysis_initial`` builds it (the truncated orography, surface
    pressure moved hypsometrically onto it, the columns interpolated to
    the model levels of that surface pressure, theta, ln ps and vapor
    through the spectral basis, the wind through the vector operator)
    and read back by the reader the checkpoints take.  The third row is
    the run's hour-0 checkpoint rebuilt in float64: the hour-0 row of a
    run scored against its own initial analysis must read it to the
    float32 the card wrote, which is how this floor is calibrated
    against the artifact."""
    from woof.globe.spectral.vector import VorticityDivergenceOperator

    from .analysis_initial import _global_regridder, _to_model_levels, surface_virtual_temperature
    from .vertical import HybridCoordinate

    transform = build_transform_from_receipt(receipt)
    grid = transform.grid
    cfg = receipt["config"]
    vertical = HybridCoordinate(np.asarray(cfg["a_half_pa"], dtype=np.float64), np.asarray(cfg["b_half"], dtype=np.float64))
    vector = VorticityDivergenceOperator(transform)
    reference = reference_fields(frame, grid, levels_pa=levels_pa, path=path,
                                 decode=decode)
    regrid = _global_regridder(frame.latitude, frame.longitude, grid)
    levels = np.asarray(frame.vertical_values, dtype=np.float64)
    ln_source = np.log(levels)
    ps = regrid(frame.fields["surface_pressure"].values)
    phi_s = GRAVITY_M_S2 * regrid(frame.fields["terrain_height"].values)
    t_src = regrid(frame.fields["air_temperature"].values)
    q_src = np.clip(regrid(frame.fields["specific_humidity"].values), 0.0, None)
    u_src = regrid(frame.fields["eastward_wind"].values)
    v_src = regrid(frame.fields["northward_wind"].values)
    vertical_only = vertical_round_trip(
        vertical, transform.backend, ln_source=ln_source, t_src=t_src, q_src=q_src, u_src=u_src, v_src=v_src,
        ps=ps, phi_s=phi_s, grid=grid, levels_pa=levels_pa, tag="vertical",
    )

    def truncate(field):
        return np.asarray(transform.inverse(transform.forward(field)), dtype=np.float64)

    def truncate_wind(uu, vv):
        zeta, div = vector.vordiv_from_wind(uu, vv)
        ut, vt = vector.wind_from_vordiv(zeta, div)
        return np.asarray(ut, dtype=np.float64), np.asarray(vt, dtype=np.float64)

    # Spectral truncation alone: the analysis's own pressure-level fields
    # through the basis with EVERY column included (the product carries
    # its own continuous extrapolation of each isobaric field under its
    # terrain, so the field the transform sees has no step at a mountain
    # edge), masked below ground only after the truncation, the mask
    # unchanged.  Filling the masked columns with the level mean before
    # the transform put a step at every mountain edge, and the ringing of
    # that step read 0.69 K of T850 on the T255 grid where the truncation
    # of the continuous field reads 0.21 K; Z500 and W250, whose columns
    # are never masked, read the same either way.
    z_src = regrid(frame.fields["geopotential_height"].values)
    spectral_fields = {name: {} for name in SCALAR_FIELDS}
    for level in reference.levels_pa:
        mask = reference.valid[level]
        k = int(np.where(np.abs(levels - float(level)) < 0.5)[0][0])
        whole = {
            "z": z_src[k], "t": t_src[k], "u": u_src[k], "v": v_src[k],
            "rh": relative_humidity_percent(q_src[k], t_src[k], float(level)),
        }
        for name in ("z", "t", "rh"):
            spectral_fields[name][level] = np.where(mask, truncate(whole[name]), np.nan)
        ut, vt = truncate_wind(whole["u"], whole["v"])
        spectral_fields["u"][level] = np.where(mask, ut, np.nan)
        spectral_fields["v"][level] = np.where(mask, vt, np.nan)
    spectral_only = PressureLevelFields(
        latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg, row_weights=grid.quadrature_weights,
        levels_pa=reference.levels_pa, fields=spectral_fields,
        valid={level: np.array(m, copy=True) for level, m in reference.valid.items()},
        surface_pressure_pa=np.array(ps, copy=True), source={"kind": "model", "floor": "spectral"},
    )
    # The cold start, as analysis_initial builds it: the model's terrain is
    # the truncated analysis orography, surface pressure moves onto it
    # hypsometrically over the air at the analysis surface, the columns go
    # to the model levels of that surface pressure, and the prognostic
    # fields (theta, ln ps, vapor, vorticity, divergence) go through the
    # spectral basis.  The reader then rebuilds temperature and the
    # hydrostatic geopotential from those coefficients exactly as it does
    # for a checkpoint, so this row is the hour-0 checkpoint in float64.
    phi_model = truncate(phi_s)
    virtual_surface = surface_virtual_temperature(t_src, q_src, ln_source, ps)
    ps_model = ps * np.exp((phi_s - phi_model) / (DRY_AIR_GAS_CONSTANT * virtual_surface))
    p_full_model = np.asarray(vertical.pressure(ps_model, transform.backend)["p_full"], dtype=np.float64)
    ln_model = np.log(p_full_model)
    t_model = _to_model_levels(t_src, ln_source, ln_model, extrapolate_below=True)
    q_model = np.clip(_to_model_levels(q_src, ln_source, ln_model), 0.0, None)
    u_model = _to_model_levels(u_src, ln_source, ln_model)
    v_model = _to_model_levels(v_src, ln_source, ln_model)
    theta_model = t_model / (p_full_model / REFERENCE_PRESSURE_PA) ** KAPPA
    zeta, div = vector.vordiv_from_wind(u_model, v_model)
    reader = ModelReader(receipt, phi_model, levels_pa=levels_pa)
    columns = reader.column_fields({
        "atmosphere__theta": np.asarray(transform.forward(theta_model)),
        "atmosphere__qv": np.asarray(transform.forward(q_model)),
        "atmosphere__log_surface_pressure": np.asarray(transform.forward(np.log(ps_model))),
        "atmosphere__vorticity": np.asarray(zeta),
        "atmosphere__divergence": np.asarray(div),
    })
    cold_start = interpolate_column_fields(
        **columns, surface_geopotential=phi_model,
        latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg, row_weights=grid.quadrature_weights,
        levels_pa=levels_pa, source={"kind": "model", "floor": "cold_start"},
    )
    return {
        "schema": SCHEMA + "/representation-floor",
        "reference": reference.source,
        "grid": {"kind": "gaussian", "nlat": grid.nlat, "nlon": grid.nlon, "truncation": grid.truncation},
        "nlev": int(vertical.nlev),
        "vertical_coordinate": cfg.get("vertical_coordinate"),
        "config_hash": receipt.get("config_hash"),
        "receipt_source": receipt.get("source", "run"),
        "jet_band_thickest_layer_pa": vertical.describe().get("jet_band_thickest_layer_pa"),
        "kink_read_factor_250hpa": kink_read_factor(vertical, 25_000.0),
        "rows": {
            "vertical": score_pair(vertical_only, reference),
            "spectral": score_pair(spectral_only, reference),
            "cold_start": score_pair(cold_start, reference),
        },
        "note": (
            "vertical: linear-in-ln p interpolation onto the run's full levels at the analysis surface "
            "pressure and back, geopotential integrated hydrostatically from the analysis terrain "
            "(so its z rows also carry the difference between the product's own hydrostatic height and "
            "this integration of its virtual temperature); spectral: the run's truncation of the "
            "product's own pressure-level fields with every column included (its below-ground "
            "extrapolation keeps each field continuous at mountain edges), masked below ground after; "
            "cold_start: the analysis through the run's own cold start "
            "(analysis_initial: truncated orography, hypsometric surface pressure, model levels, "
            "theta, ln ps, vapor and the wind through the spectral basis) read back as a checkpoint is, "
            "the hour-0 state in float64"
        ),
    }


# --------------------------------------------------------------------------
# synthetic families and calibration
# --------------------------------------------------------------------------


def synthetic_grid(truncation: int = 21):
    from woof.globe.spectral.grid import GaussianGrid

    return GaussianGrid.create(int(truncation))


def synthetic_columns(
    grid, *, nlev: int = 40, lapse_k_per_lnp: float = 0.0, mountain_pa: float = 0.0,
    mountain_mask: np.ndarray | None = None,
):
    """An analytic atmosphere on the run's own hybrid grid.

    Surface pressure ``101325 - 6000 cos(lat) cos(2 lon)`` Pa (minus
    ``mountain_pa`` where ``mountain_mask``), terrain ``300 (1 + sin(lat)
    sin(lon))`` m, temperature ``Ts + lapse ln(p / ps)`` with ``Ts = 288 +
    15 cos(lat)`` (isothermal columns at lapse 0), ``u = 10 cos(lat) + 20
    ln(ps/p)``, ``v = 5 sin(2 lon) cos(lat)``, ``q`` linear in ln p from
    0.012 at the surface toward 0 at the top, all with the closed forms
    the calibration reads against.  Returns the full-level column fields
    (top to bottom) and a callable ``truth(level_pa)`` giving the exact
    fields at a target pressure."""
    from .vertical import HybridCoordinate

    lat2, lon2 = grid.mesh()
    ps = 101_325.0 - 6_000.0 * np.cos(lat2) * np.cos(2.0 * lon2)
    if mountain_mask is not None:
        ps = np.where(mountain_mask, ps - float(mountain_pa), ps)
    terrain = 300.0 * (1.0 + np.sin(lat2) * np.sin(lon2))
    phi_s = GRAVITY_M_S2 * terrain
    coordinate = HybridCoordinate.surface_stretched(int(nlev))
    a, b = coordinate.a_half_pa, coordinate.b_half
    p_half = a[:, None, None] + b[:, None, None] * ps[None]
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    ts = 288.0 + 15.0 * np.cos(lat2)
    q_surface = 0.012 * (0.5 + 0.5 * np.cos(lat2))
    ln_top = math.log(float(a[0]))
    # Tv = T (1 + 0.61 q) must stay LINEAR in ln p for the closed form
    # and the dycore's integration to be exact: isothermal columns take q
    # linear in ln p (family A1), columns with a lapse take q constant
    # (family A2); ``slope`` is 1 or 0 accordingly.
    slope = 0.0 if float(lapse_k_per_lnp) != 0.0 else 1.0

    def profile(p):
        x = np.log(p / ps[None] if p.ndim == 3 else p / ps)
        t = ts + float(lapse_k_per_lnp) * x
        u = 10.0 * np.cos(lat2) - 20.0 * x
        v = 5.0 * np.sin(2.0 * lon2) * np.cos(lat2) + 0.0 * x
        depth = np.log(ps) - ln_top
        q = q_surface * (1.0 + slope * x / depth)
        tv = t * (1.0 + 0.61 * q)
        return t, u, v, q, tv

    t, u, v, q, tv = profile(p_full)
    phi = coordinate.hydrostatic_geopotential(tv, phi_s, p_half)

    def truth(level_pa: float):
        p = np.full(ps.shape, float(level_pa))
        tt, uu, vv, qq, _ = profile(p)
        x = np.log(p / ps)
        # closed-form geopotential for Tv = (Ts + L x)(1 + 0.61 q(x)),
        # integrated in x from the surface: phi = phi_s - R int_0^x Tv dx'
        # with q(x') = q_s (1 + x'/depth) and Ts, L, q_s, depth per column.
        depth = np.log(ps) - ln_top
        L = float(lapse_k_per_lnp)
        c0 = ts * (1.0 + 0.61 * q_surface)
        c1 = L * (1.0 + 0.61 * q_surface) + ts * 0.61 * q_surface * slope / depth
        c2 = L * 0.61 * q_surface * slope / depth
        integral = c0 * x + c1 * x * x / 2.0 + c2 * x ** 3 / 3.0
        zz = (phi_s - DRY_AIR_GAS_CONSTANT * integral) / GRAVITY_M_S2
        rh = relative_humidity_percent(qq, tt, p)
        return {"z": zz, "t": tt, "u": uu, "v": vv, "rh": rh, "inside": (p < ps) & (p > p_full[0])}

    columns = {
        "p_full": p_full, "geopotential": phi, "temperature": t, "virtual_temperature": tv,
        "u": u, "v": v, "specific_humidity": q, "surface_pressure": ps, "surface_geopotential": phi_s,
        "latitude_deg": grid.latitude_deg, "longitude_deg": grid.longitude_deg,
        "row_weights": grid.quadrature_weights,
    }
    return columns, truth


def synthetic_reference(grid, *, seed: int = 0, levels_pa=LEVELS_PA) -> PressureLevelFields:
    """Analysis-shaped fields: smooth planetary patterns plus a seeded
    zero-mean texture, all columns valid."""
    lat2, lon2 = grid.mesh()
    rng = np.random.default_rng(seed)
    fields: dict[str, dict[float, np.ndarray]] = {name: {} for name in SCALAR_FIELDS}
    valid = {}
    for level in levels_pa:
        scale = math.log(101_325.0 / float(level))
        fields["z"][level] = 7_000.0 * scale * (1.0 - 0.05 * np.sin(lat2) ** 2) + 80.0 * np.cos(3.0 * lon2) * np.cos(lat2) + 5.0 * rng.standard_normal(lat2.shape)
        fields["t"][level] = 288.0 - 45.0 * scale + 12.0 * np.cos(lat2) + 2.0 * np.sin(2.0 * lon2) * np.cos(lat2) + 0.5 * rng.standard_normal(lat2.shape)
        fields["u"][level] = 10.0 + 25.0 * scale * np.sin(2.0 * lat2) ** 2 + 3.0 * np.cos(4.0 * lon2) + 1.0 * rng.standard_normal(lat2.shape)
        fields["v"][level] = 6.0 * np.sin(3.0 * lon2) * np.cos(lat2) + 1.0 * rng.standard_normal(lat2.shape)
        fields["rh"][level] = 55.0 + 25.0 * np.cos(2.0 * lon2) * np.cos(lat2) + 3.0 * rng.standard_normal(lat2.shape)
        valid[level] = np.ones(lat2.shape, dtype=bool)
    return PressureLevelFields(
        latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg,
        row_weights=grid.quadrature_weights, levels_pa=tuple(float(v) for v in levels_pa),
        fields=fields, valid=valid, surface_pressure_pa=np.full(lat2.shape, 101_000.0),
        source={"kind": "analysis", "synthetic": True},
    )


def shifted_copy(reference: PressureLevelFields, offsets: dict[tuple[str, float], np.ndarray | float]) -> PressureLevelFields:
    fields = {name: {level: np.array(arr, copy=True) for level, arr in per.items()} for name, per in reference.fields.items()}
    for (name, level), delta in offsets.items():
        fields[name][level] = fields[name][level] + delta
    return PressureLevelFields(
        latitude_deg=reference.latitude_deg, longitude_deg=reference.longitude_deg,
        row_weights=reference.row_weights, levels_pa=reference.levels_pa, fields=fields,
        valid={level: np.array(mask, copy=True) for level, mask in reference.valid.items()},
        surface_pressure_pa=np.array(reference.surface_pressure_pa, copy=True),
        source={"kind": "model", "synthetic": True},
    )


class SyntheticFrame:
    """A reference frame on a regular grid, the shape the Rust decoder
    returns, for the regrid family."""

    def __init__(self, latitude, longitude, levels_pa, fields, *, valid_time=None, source_cycle=None):
        from datetime import datetime as _dt

        self.latitude = np.asarray(latitude, dtype=np.float64)
        self.longitude = np.asarray(longitude, dtype=np.float64)
        self.vertical_kind = "pressure"
        self.vertical_values = np.asarray(levels_pa, dtype=np.float64)
        self.fields = {name: _Field(np.asarray(values, dtype=np.float64)) for name, values in fields.items()}
        self.valid_time = valid_time or _dt(2000, 1, 1)
        self.source_cycle = source_cycle or self.valid_time
        self.input_sha256 = {}
        self.mapping_sha256 = None


class _Field:
    def __init__(self, values):
        self.values = values


def calibrate() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    grid = synthetic_grid(21)
    # family A: interpolation and hydrostatics against the closed form
    for family, lapse in (("A1", 0.0), ("A2", 30.0), ("A2", -30.0)):
        columns, truth = synthetic_columns(grid, lapse_k_per_lnp=lapse)
        plf = interpolate_column_fields(**columns)
        worst = {name: 0.0 for name in SCALAR_FIELDS}
        for level in LEVELS_PA:
            exact = truth(level)
            inside = plf.valid[level]
            assert np.array_equal(inside, exact["inside"])
            for name in SCALAR_FIELDS:
                err = float(np.max(np.abs(plf.fields[name][level][inside] - exact[name][inside])))
                worst[name] = max(worst[name], err)
        rows.append({"family": family, "lapse_k_per_lnp": lapse, "max_abs_error": worst})
    # family B: planted differences
    reference = synthetic_reference(grid, seed=1)
    for name, kind, level in TARGETS:
        for sign in (1.0, -1.0):
            if kind == "wind":
                du, dv = sign * 3.0, -sign * 4.0
                model = shifted_copy(reference, {("u", level): du, ("v", level): dv})
                scores = score_pair(model, reference)[name]["regions"]
                rows.append({
                    "family": "B", "target": name, "planted": {"du": du, "dv": dv},
                    "read": {region: {"rmsve": r["rmsve"], "u_bias": r["u_bias"], "v_bias": r["v_bias"]} for region, r in scores.items()},
                    "expected_rmsve": 5.0,
                })
            else:
                delta = sign * {"z": 12.0, "t": 1.5, "rh": 8.0}[kind]
                model = shifted_copy(reference, {(kind, level): delta})
                scores = score_pair(model, reference)[name]["regions"]
                rows.append({
                    "family": "B", "target": name, "planted": delta,
                    "read": {region: {"bias": r["bias"], "rmse": r["rmse"]} for region, r in scores.items()},
                })
    rng = np.random.default_rng(5)
    texture = rng.standard_normal(grid.shape)
    texture = texture - grid.global_mean(texture)
    texture = 5.0 * texture / grid.rms(texture)
    model = shifted_copy(reference, {("z", 50_000.0): 12.0 + texture})
    g = score_pair(model, reference)["z500"]["regions"]["global"]
    rows.append({"family": "B", "target": "z500", "planted": "12 m + zero-mean texture of rms 5 m",
                 "read": {"bias": g["bias"], "rmse": g["rmse"]}, "expected": {"bias": 12.0, "rmse": 13.0}})
    # family C: self
    self_scores = score_pair(reference, reference)
    worst_bias = max(abs(r["bias"] if "bias" in r else r["rmsve"]) for t in self_scores.values() for r in t["regions"].values())
    worst_ac = max(abs(1.0 - r["anomaly_correlation"]) for t in self_scores.values() for r in t["regions"].values() if "anomaly_correlation" in r)
    rows.append({"family": "C", "self_max_abs_bias_or_rmsve": worst_bias, "self_max_ac_departure": worst_ac})
    # family D: planted rotation of the eddy field
    lat2, lon2 = grid.mesh()
    eddy = 60.0 * np.cos(3.0 * lon2) * np.cos(lat2)
    quadrature = 60.0 * np.sin(3.0 * lon2) * np.cos(lat2)
    base = np.array(reference.fields["z"][50_000.0], copy=True)
    zonal = zonal_mean_climatology(base, np.ones(base.shape, dtype=bool))
    ref_d = shifted_copy(reference, {("z", 50_000.0): (zonal + eddy) - base})
    for theta_deg in (0.0, 30.0, 60.0, 90.0, 120.0, 180.0):
        th = math.radians(theta_deg)
        model = shifted_copy(ref_d, {("z", 50_000.0): (math.cos(th) - 1.0) * eddy + math.sin(th) * quadrature})
        ac = score_pair(model, ref_d)["z500"]["regions"]["global"]["anomaly_correlation"]
        rows.append({"family": "D", "theta_deg": theta_deg, "ac_read": ac, "ac_expected": math.cos(th)})
    # family E: refusals
    mountain = (np.abs(lat2) < math.radians(15.0)) & (np.cos(lon2) > 0.5)
    for direction in ("model", "reference"):
        columns, _ = synthetic_columns(grid, mountain_pa=40_000.0, mountain_mask=mountain)
        columns_flat, _ = synthetic_columns(grid)
        hilly = interpolate_column_fields(**columns)
        flat = interpolate_column_fields(**columns_flat)
        spoiled = shifted_copy(flat, {("t", 85_000.0): np.where(mountain, 50.0, 0.0)})
        if direction == "model":
            model, ref = hilly, spoiled
        else:
            model, ref = spoiled, hilly
        scores = score_pair(model, ref)
        planted_fraction = float(np.sum(grid.quadrature_weights[:, None] * mountain) / (2.0 * grid.nlon))
        rows.append({
            "family": "E", "direction": direction, "planted_masked_fraction_global": planted_fraction,
            "read_masked_fraction": {t: scores[t]["regions"]["global"]["masked_fraction"] for t in ("t850", "rh700", "z500", "w250")},
            "t850_global_bias_with_planted_50K_inside_mountain": scores["t850"]["regions"]["global"]["bias"],
            "rh700_global_bias": scores["rh700"]["regions"]["global"]["bias"],
        })
    # family F: regrid
    lat_src = np.arange(90.0, -90.001, -1.0)
    lon_src = np.arange(0.0, 360.0, 1.0)
    lon_s, lat_s = np.meshgrid(lon_src, lat_src)
    linear = 2.0 * lat_s + 0.5 * np.where(lon_s > 180.0, 360.0 - lon_s, lon_s)
    for flip in (False, True):
        lat_use = lat_src[::-1] if flip else lat_src
        field_use = linear[::-1] if flip else linear
        levels = np.asarray(LEVELS_PA)
        stack = np.repeat(field_use[None], levels.size, axis=0)
        frame = SyntheticFrame(lat_use, lon_src, levels, {
            "geopotential_height": stack, "air_temperature": 250.0 + stack / 10.0, "specific_humidity": np.full(stack.shape, 0.001),
            "eastward_wind": stack / 100.0, "northward_wind": -stack / 100.0,
            "surface_pressure": np.full(field_use.shape, 101_000.0), "terrain_height": field_use,
        })
        plf = reference_fields(frame, grid)
        glat, glon = grid.mesh()
        glat_deg, glon_deg = np.degrees(glat), np.degrees(glon)
        expected = 2.0 * glat_deg + 0.5 * np.where(glon_deg > 180.0, 360.0 - glon_deg, glon_deg)
        err = max(float(np.max(np.abs(plf.fields["z"][level] - expected))) for level in LEVELS_PA)
        rows.append({"family": "F", "ascending_latitude": flip, "max_abs_error_m": err})
    # family G: the vertical round trip on two level sets, linear (reads
    # zero) and with a planted jet (reads the level set's own predicted
    # number, smaller on the level set that resolves the jet)
    from .vertical import HybridCoordinate

    for label, vertical in (("surface_stretched_40", HybridCoordinate.surface_stretched(40)),
                            ("jet_refined_48", HybridCoordinate.jet_refined(48))):
        family = level_set_floor_family(vertical)
        g = family["rows"]
        worst_linear = max(
            abs(v) for target in g["G1_linear"]["read"].values() for r in target.values() for v in r.values()
        )
        rows.append({
            "family": "G", "level_set": label, "nlev": family["nlev"],
            "kink_read_factor_250hpa": family["kink_read_factor_at_target"],
            "G1_linear_max_abs_read": worst_linear,
            "G2_jet_w250_read": g["G2_jet"]["read"]["w250"]["global"],
            "G2_jet_expected": g["G2_jet"]["expected"],
            "G2_jet_other_targets_max_abs_read": max(
                abs(v) for name, target in g["G2_jet"]["read"].items() if name != "w250"
                for r in target.values() for v in r.values()
            ),
        })
    return {"schema": SCHEMA + "/calibration", "rows": rows}


# --------------------------------------------------------------------------
# the chart
# --------------------------------------------------------------------------


def analysis_series(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split scorecard rows into the analysis series of record (one row per
    hour, the product most analysis rows carry), the other analysis
    products at hours the series already holds, and the forecast rows."""
    analyses = [r for r in rows if r["reference"].get("kind") == "analysis"]
    forecasts = [r for r in rows if r["reference"].get("kind") == "forecast"]
    counts: dict[str, int] = {}
    for r in analyses:
        product = r["reference"].get("product", "")
        counts[product] = counts.get(product, 0) + 1
    order = sorted(counts, key=lambda p: (-counts[p], p))
    primary: dict[float, dict] = {}
    extra: list[dict] = []
    for r in sorted(analyses, key=lambda r: (r["hour"], order.index(r["reference"].get("product", "")))):
        if r["hour"] in primary:
            extra.append(r)
        else:
            primary[r["hour"]] = r
    return [primary[h] for h in sorted(primary)], extra, sorted(forecasts, key=lambda r: r["hour"])


def _reading(row: dict, target: str, region: str, key: str):
    value = row["scores"][target]["regions"][region].get(key)
    return None if value is None else float(value)


def render_chart(payloads: list[dict], out_png: Path, *, title: str = "") -> Path:
    """Growth curves of the five targets over the four regions for one or
    more scorecard payloads (analysis chart, not a weather field)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("z500", "rmse", "Z500 rmse, m"), ("z500", "anomaly_correlation", "Z500 anomaly correlation"),
        ("t850", "rmse", "T850 rmse, K"), ("t850", "bias", "T850 bias, K"),
        ("w250", "rmsve", "250 hPa vector wind rmse, m/s"), ("w850", "rmsve", "850 hPa vector wind rmse, m/s"),
        ("rh700", "rmse", "RH700 rmse, %"), ("rh700", "bias", "RH700 bias, %"),
    )
    regions = list(REGIONS)
    fig, axes = plt.subplots(len(panels), len(regions), figsize=(4.2 * len(regions), 2.6 * len(panels)), sharex=True)
    colors = ("#1f5fbf", "#c0392b", "#2e8b57", "#8e44ad", "#d68910")
    for i, (target, key, label) in enumerate(panels):
        for j, region in enumerate(regions):
            ax = axes[i][j]
            for k, payload in enumerate(payloads):
                color = colors[k % len(colors)]
                name = payload.get("label") or "run"
                primary, extra, forecasts = analysis_series(payload["rows"])
                for rows, style, marker, lw, ms, tag in (
                    (primary, "-", "o", 1.5, 4, "analysis"),
                    (forecasts, ":", ".", 1.0, 3, "forecast of the reference model at the same lead"),
                ):
                    if not rows:
                        continue
                    hours = [r["hour"] for r in rows]
                    values = [np.nan if (v := _reading(r, target, region, key)) is None else v for r in rows]
                    products = "/".join(sorted({r["reference"].get("product", "") for r in rows}))
                    ax.plot(hours, values, style, marker=marker, color=color, lw=lw, ms=ms,
                            label=f"{name} vs {products} {tag}" if (i == 0 and j == 0) else None)
                if extra:
                    hours = [r["hour"] for r in extra]
                    values = [np.nan if (v := _reading(r, target, region, key)) is None else v for r in extra]
                    products = "/".join(sorted({r["reference"].get("product", "") for r in extra}))
                    ax.plot(hours, values, linestyle="none", marker="s", color=color, ms=5, mfc="none",
                            label=f"{name} vs {products} analysis (second product)" if (i == 0 and j == 0) else None)
            skill = next((p.get("reference_skill") for p in payloads if p.get("reference_skill")), None)
            if skill:
                # The reference model's own forecast against the analysis at
                # the same instant (identical for every arm scored with one
                # reference set, so drawn once): the bar the arms read against.
                primary, _extra, _forecasts = analysis_series(skill)
                hours = [r["hour"] for r in primary]
                values = [np.nan if (v := _reading(r, target, region, key)) is None else v for r in primary]
                forecast_products = "/".join(sorted({r["forecast"].get("product", "") for r in primary}))
                analysis_products = "/".join(sorted({r["reference"].get("product", "") for r in primary}))
                ax.plot(hours, values, "--", marker="d", color="0.35", lw=1.2, ms=3.5,
                        label=(f"{forecast_products} forecast vs {analysis_products} analysis (the reference model's own error)"
                               if (i == 0 and j == 0) else None))
            if i == 0:
                ax.set_title(region.replace("_", " "), fontsize=11)
            if j == 0:
                ax.set_ylabel(label, fontsize=9)
            if key in ("bias",):
                ax.axhline(0.0, color="0.6", lw=0.8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            if i == len(panels) - 1:
                ax.set_xlabel("forecast hour", fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    rows_of_legend = 0
    if handles:
        ncol = 2 if len(labels) > 4 else min(3, len(labels))
        rows_of_legend = -(-len(labels) // ncol)
        fig.legend(handles, labels, loc="upper center", ncol=ncol, fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, 0.985))
    top = 0.985 - 0.013 * rows_of_legend
    fig.suptitle(title or "WOOF global upper-air scorecard", fontsize=12, y=top - 0.008)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top - 0.03))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def scorecard_table(payloads: list[dict], *, hour: float, kind: str = "analysis") -> list[dict[str, object]]:
    """The numbers at one hour against one reference kind, one row per
    (target, reading, region) with a column per payload."""
    readings = {
        "z500": ("rmse", "bias", "anomaly_correlation"), "t850": ("rmse", "bias", "anomaly_correlation"),
        "w250": ("rmsve", "speed_bias"), "w850": ("rmsve", "speed_bias"), "rh700": ("rmse", "bias", "anomaly_correlation"),
    }
    # Against the analysis the row of record is the analysis series' own
    # (one product per hour, the product most hours carry), so an hour
    # with two analysis products reads the same one as the chart.
    candidates = []
    for payload in payloads:
        primary, _extra, forecasts = analysis_series(payload["rows"])
        pool = primary if kind == "analysis" else forecasts
        candidates.append([r for r in pool if abs(r["hour"] - hour) < 1e-6])
    table = []
    for target, keys in readings.items():
        for key in keys:
            for region in REGIONS:
                row: dict[str, object] = {"target": target, "reading": key, "region": region}
                for payload, hit in zip(payloads, candidates):
                    row[payload.get("label") or "run"] = _reading(hit[0], target, region, key) if hit else None
                table.append(row)
    return table


def render_scorecard_table(payloads: list[dict], out_png: Path, *, hour: float, kind: str = "analysis", title: str = "") -> Path:
    """The scorecard numbers at one hour as a table image: rows per
    target, reading and region, one column per payload plus the
    difference of every later column from the first (analysis chart)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [p.get("label") or f"run {k}" for k, p in enumerate(payloads)]
    table = scorecard_table(payloads, hour=hour, kind=kind)
    units = {"z500": "m", "t850": "K", "w250": "m/s", "w850": "m/s", "rh700": "%"}
    header = ["target", "reading", "region", *labels]
    if len(payloads) > 1:
        header += [f"{lab} minus {labels[0]}" for lab in labels[1:]]
    cells = []
    for row in table:
        values = [row[lab] for lab in labels]
        digits = 4 if row["reading"] == "anomaly_correlation" else 2
        unit = "1" if row["reading"] == "anomaly_correlation" else units[row["target"]]
        line = [row["target"], f"{row['reading'].replace('_', ' ')} ({unit})", row["region"].replace("_", " ")]
        line += ["n/a" if v is None else f"{v:.{digits}f}" for v in values]
        if len(payloads) > 1:
            for v in values[1:]:
                line.append("n/a" if v is None or values[0] is None else f"{v - values[0]:+.{digits}f}")
        cells.append(line)
    fig_h = 0.28 * (len(cells) + 2) + 0.8
    widths = [max(len(str(line[c])) for line in [header, *cells]) for c in range(len(header))]
    fig_w = 0.075 * sum(widths) + 0.4 * len(header)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tab = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="left", colLoc="left",
                   colWidths=[w / sum(widths) for w in widths])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1.0, 1.15)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("0.8")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("0.92")
        elif cells[r - 1][2] == "global":
            cell.set_facecolor("0.97")
    ax.set_title(title or f"upper-air scorecard at hour {hour:g} against the {kind}", fontsize=11, pad=8)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


def summary_lines(payload: dict) -> list[str]:
    lines = [f"== {payload.get('label') or payload.get('run_dir')} upper-air scorecard =="]
    for row in payload["rows"]:
        ref = row["reference"]
        s = row["scores"]
        g = lambda t, r, k: s[t]["regions"][r].get(k)  # noqa: E731

        def fmt(value, width=7, digits=2):
            return f"{value:{width}.{digits}f}" if value is not None else " " * (width - 3) + "n/a"

        lines.append(
            f"h{row['hour']:5.1f} vs {ref.get('kind', '?'):8s} "
            f"Z500 rmse NH {fmt(g('z500', 'nh_extratropics', 'rmse'))} SH {fmt(g('z500', 'sh_extratropics', 'rmse'))} "
            f"AC NH {fmt(g('z500', 'nh_extratropics', 'anomaly_correlation'), 7, 4)} | "
            f"T850 rmse NH {fmt(g('t850', 'nh_extratropics', 'rmse'))} bias {fmt(g('t850', 'nh_extratropics', 'bias'))} | "
            f"W250 rmsve NH {fmt(g('w250', 'nh_extratropics', 'rmsve'))} trop {fmt(g('w250', 'tropics', 'rmsve'))} | "
            f"W850 rmsve NH {fmt(g('w850', 'nh_extratropics', 'rmsve'))} | "
            f"RH700 rmse NH {fmt(g('rh700', 'nh_extratropics', 'rmse'))} bias {fmt(g('rh700', 'nh_extratropics', 'bias'))} | "
            f"global Z500 {fmt(g('z500', 'global', 'rmse'))} T850 {fmt(g('t850', 'global', 'rmse'))}"
        )
    for row in payload.get("unpaired_checkpoints", []):
        lines.append(f"h{row['hour']:5.1f} no reference valid at {row['valid_time']}")
    return lines


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=1, sort_keys=False, allow_nan=False)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="upper-air scorecard against an analysis, every checkpoint hour")
    parser.add_argument("--run-dir", help="directory of arwen_global_step*.npz checkpoints with the run receipt")
    parser.add_argument("--checkpoints", nargs="*", help="explicit checkpoint paths (else every step file in --run-dir)")
    parser.add_argument("--reference", nargs="*", default=[], help="reference GRIB files (analyses and forecasts; paired by valid time)")
    parser.add_argument("--reference-glob", nargs="*", default=[], help="globs of reference GRIB files")
    parser.add_argument("--initial-analysis", help="the run's initial analysis GRIB (for the surface geopotential; SHA-256 checked against the receipt)")
    parser.add_argument("--mapping", help="analysis mapping id or path (default: the run receipt's)")
    parser.add_argument("--cache-dir", help="where decoded reference and model pressure-level fields are kept")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", help="JSON output path")
    parser.add_argument("--chart", help="PNG growth-curve chart of --inputs (or of this run)")
    parser.add_argument("--inputs", nargs="*", default=[], help="scorecard JSONs to chart together")
    parser.add_argument("--title", default="")
    parser.add_argument("--calibrate", action="store_true", help="run the synthetic families and print their numbers")
    parser.add_argument("--self-check", help="score this reference GRIB against itself on the grid of --run-dir's receipt")
    parser.add_argument("--floor", help="representation floor of this analysis GRIB on --run-dir's grid and levels (or --config's)")
    parser.add_argument("--config", help="a WOOF global config TOML standing in for --run-dir under --floor: the floor of a level set that has not run")
    parser.add_argument("--table", help="PNG table of the numbers at --table-hour against --table-kind, for --inputs (or this run)")
    parser.add_argument("--table-hour", type=float, default=24.0)
    parser.add_argument("--table-kind", default="analysis")
    args = parser.parse_args(argv)

    if args.calibrate:
        payload = calibrate()
        print(json.dumps(payload, indent=1))
        if args.out:
            _write_json(Path(args.out), payload)
        return 0

    if args.self_check:
        if not args.run_dir:
            parser.error("--self-check needs --run-dir for the grid")
        receipt = read_receipt(Path(args.run_dir))
        mapping = resolve_mapping(args.mapping, receipt)
        from woof.globe.spectral.grid import GaussianGrid

        grid = GaussianGrid.for_shape(int(receipt["transform"]["nlat"]), int(receipt["transform"]["nlon"]))
        self_check_frame, self_check_decode = decode_reference(mapping, Path(args.self_check))
        plf = reference_fields(self_check_frame, grid, path=args.self_check, decode=self_check_decode)
        scores = score_pair(plf, plf)
        payload = {"schema": SCHEMA + "/self-check", "reference": plf.source, "scores": scores}
        worst = 0.0
        worst_ac = 0.0
        for t in scores.values():
            for r in t["regions"].values():
                worst = max(worst, abs(r.get("bias", r.get("rmsve", 0.0)) or 0.0), r.get("rmse", r.get("rmsve", 0.0)) or 0.0)
                if r.get("anomaly_correlation") is not None:
                    worst_ac = max(worst_ac, abs(1.0 - r["anomaly_correlation"]))
        payload["max_abs_score"] = worst
        payload["max_ac_departure"] = worst_ac
        print(f"self-check {args.self_check}: max |bias|,rmse {worst:.3e}; max |1 - AC| {worst_ac:.3e}")
        if args.out:
            _write_json(Path(args.out), payload)
        return 0

    if args.floor:
        if args.config and args.run_dir:
            parser.error("--floor takes --run-dir or --config, not both")
        if args.config:
            from .config import load_config

            receipt = floor_receipt_from_config(load_config(args.config), mapping=args.mapping)
        elif args.run_dir:
            receipt = read_receipt(Path(args.run_dir))
        else:
            parser.error("--floor needs --run-dir (a run) or --config (a level set that has not run) for the grid and the vertical tables")
        mapping = resolve_mapping(args.mapping, receipt)
        floor_frame, floor_decode = decode_reference(mapping, Path(args.floor))
        payload = representation_floor(floor_frame, receipt, path=args.floor, decode=floor_decode)
        print(f"floor of {payload['nlev']} levels ({payload.get('vertical_coordinate')}), receipt from {payload['receipt_source']}: "
              f"jet band thickest layer {(payload.get('jet_band_thickest_layer_pa') or 0.0) / 100.0:.1f} hPa, "
              f"kink read factor at 250 hPa {payload['kink_read_factor_250hpa']:.4f}")
        for tag, scores in payload["rows"].items():
            z = scores["z500"]["regions"]["global"]
            w = scores["w250"]["regions"]["global"]
            t = scores["t850"]["regions"]["global"]
            rh = scores["rh700"]["regions"]["global"]
            print(f"floor {tag:9s} global: Z500 rmse {z['rmse']:.2f} m bias {z['bias']:+.2f} | T850 rmse {t['rmse']:.2f} K | "
                  f"W250 rmsve {w['rmsve']:.2f} speed bias {w['speed_bias']:+.2f} m/s | RH700 rmse {rh['rmse']:.2f} %")
        if args.out:
            _write_json(Path(args.out), payload)
        return 0

    if (args.chart or args.table) and not args.run_dir:
        payloads = [json.load(open(p, "r", encoding="utf-8")) for p in args.inputs]
        if not payloads:
            parser.error("--chart/--table need --inputs or --run-dir")
        if args.chart:
            render_chart(payloads, Path(args.chart), title=args.title)
            print(f"chart {args.chart}")
        if args.table:
            render_scorecard_table(payloads, Path(args.table), hour=args.table_hour, kind=args.table_kind, title=args.title)
            print(f"table {args.table}")
        return 0

    if not args.run_dir:
        parser.error("--run-dir is required (or --calibrate, --self-check, --chart with --inputs)")
    references = [Path(p) for p in args.reference]
    for pattern in args.reference_glob:
        references.extend(Path(p) for p in sorted(glob.glob(pattern)))
    if not references:
        parser.error("at least one --reference or --reference-glob is required")
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.run_dir) / "scorecard-cache"
    payload = measure_run(
        Path(args.run_dir), references, cache_dir=cache_dir, workers=max(1, int(args.workers)),
        label=args.label, initial_analysis=Path(args.initial_analysis) if args.initial_analysis else None,
        mapping=args.mapping, checkpoints=[Path(p) for p in args.checkpoints] if args.checkpoints else None,
    )
    for line in summary_lines(payload):
        print(line)
    if args.out:
        _write_json(Path(args.out), payload)
        print(f"wrote {args.out}")
    if args.chart:
        render_chart([payload], Path(args.chart), title=args.title)
        print(f"chart {args.chart}")
    if args.table:
        render_scorecard_table([payload], Path(args.table), hour=args.table_hour, kind=args.table_kind, title=args.title)
        print(f"table {args.table}")
    return 0


__all__ = [
    "CALIBRATION", "CLIMATOLOGY", "LEVELS_PA", "REGIONS", "RH_CONVENTION", "SCHEMA", "TARGETS",
    "ModelReader", "PressureLevelFields", "SyntheticFrame", "calibrate", "checkpoint_content_digest",
    "floor_receipt_from_config", "interpolate_column_fields", "kink_read_factor", "level_set_floor_family",
    "model_cache_path", "vertical_round_trip",
    "analysis_series", "measure_run", "reference_fields", "relative_humidity_percent", "render_chart",
    "render_scorecard_table", "representation_floor", "score_pair",
    "scorecard_table", "shifted_copy", "summary_lines", "surface_geopotential_for_run",
    "surface_geopotential_from_frame", "synthetic_columns", "synthetic_grid", "synthetic_reference",
    "zonal_mean_climatology",
]


if __name__ == "__main__":
    sys.exit(main())
