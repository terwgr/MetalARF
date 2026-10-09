"""Column soundings from WOOF global checkpoints, in float64, on the run's grid.

The shared reader of the convection-timing instruments
(:mod:`woof.globe.trigger_diagnostic` and
:mod:`woof.globe.storm_reader`).  From one checkpoint it
synthesizes temperature, vapor and pressure from the five spectral fields
(numpy float64, the transform of the receipt's own geometry), reads the
ten grid tracers as they are (schema v3) or synthesizes them (a
spectral-tracer-era archive, inspection only), and diagnoses

* relative humidity over liquid water (Bolton 1980 saturation vapour
  pressure, ``q_s = eps e_s / (p - e_s)``; the microphysics kernel uses
  the Flatau polynomial, which differs by under one percent between
  -40 and +40 C, and every reading states which formula it used),
* the interface pressure velocity ``omega`` from the continuity of the
  checkpointed mass flux (``div(v dp)`` per layer with ``dp`` built from
  the hybrid coefficients, ``ps_t = -sum``, the vertical module's own
  recurrence), the same diagnostic the dynamics feed the physics,
* parcel CAPE and CIN, surface-based and most-unstable: the parcel of the
  lowest full level (surface-based) or of the highest equivalent
  potential temperature in the lowest 300 hPa (most-unstable) lifted
  dry-adiabatically to its lifting condensation level (Bolton 1980
  T_LCL) and pseudo-adiabatically above it (condensate removed, two-stage
  Runge-Kutta in ln p with four sub-steps per layer), buoyancy in virtual
  temperature against the environment's ``T (1 + 0.61 q_v)`` (condensate
  loading of the environment not counted), CAPE the whole positive area
  above the parcel level, CIN the negative area between the parcel level
  and the level of free convection (the first positive layer above the
  LCL).  Every layer contributes ``R_d (T_v,p - T_v,e) ln(p_below /
  p_above)`` over its own half-level bounds, so an analytic sounding
  whose buoyancy is constant per layer integrates exactly.

Levels are the model's own: index 0 is the model top and ``nlev - 1``
the layer above the surface.  Nothing here writes; nothing here is
approximate without saying so.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import (
    CONDENSATE_SPECIES,
    DRY_AIR_CP,
    DRY_AIR_GAS_CONSTANT,
    EPSILON,
    GRAVITY_M_S2,
    KAPPA,
    LATENT_HEAT_VAPORIZATION,
    REFERENCE_PRESSURE_PA,
)

#: Bolton (1980) saturation vapour pressure over liquid water, Pa.
SATURATION_FORMULA = "Bolton 1980, e_s = 611.2 exp(17.67 (T - 273.15) / (T - 29.65)) Pa"
#: The most-unstable parcel is searched this deep above the surface.
MOST_UNSTABLE_SEARCH_DEPTH_PA = 30000.0
#: Pseudo-adiabatic sub-steps per model layer.
MOIST_ASCENT_SUBSTEPS = 4
#: Parcels are lifted no higher than this (the stratosphere never holds
#: positive buoyancy for a tropospheric parcel).
PARCEL_TOP_PA = 5000.0
#: A layer counts as cloudy (grid-scale condensate present) above this
#: mixing ratio of cloud water plus cloud ice.
CLOUDY_LEVEL_KG_KG = 1.0e-6

CHECKPOINT_LAND_FRACTION = "surface__land_fraction"
CHECKPOINT_CONVECTIVE = "physics__rainc"
CHECKPOINT_GRID_SCALE_BUCKETS = (
    "surface__accumulated_rain_kg_m2",
    "surface__accumulated_snow_kg_m2",
    "surface__accumulated_graupel_kg_m2",
)
#: Physics arrays a cold-start checkpoint may carry before any physics call:
#: the surface stores the cold start seeds from the analysis (Noah's snow
#: planes, written at step 0 since the cold-start seeding).  They are not
#: buckets, so a step-0 checkpoint carrying only these and no convective
#: accumulator is a cumulus-bearing run before its first call.
COLD_START_PHYSICS_PREFIXES = ("physics__noah_",)


def convective_accumulator_is_absent_by_construction(names, step: int) -> bool:
    """True when a checkpoint without ``CHECKPOINT_CONVECTIVE`` is the cold
    start of a cumulus-bearing run (step 0, no physics bucket beyond the
    seeded surface stores), so its convective rain is zero by construction.
    False when the checkpoint is past step 0 or carries any other physics
    array: that is a different tree's state, and reading it as zero would
    print a convective reading for a run that never booked convective rain.
    Measured 2026-09-05: the cold-start seeding writes ``physics__noah_snow``
    (and its cover and depth) into the step-0 namespace, and both readers
    took the seeded surface for a physics call and refused the whole run."""
    if int(step) != 0:
        return False
    buckets = [
        n for n in names
        if n.startswith("physics__") and not n.startswith(COLD_START_PHYSICS_PREFIXES)
    ]
    return not buckets


def convective_accumulator_refusal(path, names, step: int) -> str:
    buckets = sorted(
        n for n in names
        if n.startswith("physics__") and not n.startswith(COLD_START_PHYSICS_PREFIXES)
    )
    return (
        f"{path}: no {CHECKPOINT_CONVECTIVE} array although the checkpoint is past "
        f"the cold start (step {int(step)}) or carries physics buckets {buckets[:6]}; "
        "not a cumulus-bearing run"
    )


def saturation_vapor_pressure_pa(temperature_k) -> np.ndarray:
    t = np.asarray(temperature_k, dtype=np.float64)
    return 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))


def saturation_mixing_ratio(temperature_k, pressure_pa) -> np.ndarray:
    es = saturation_vapor_pressure_pa(temperature_k)
    p = np.asarray(pressure_pa, dtype=np.float64)
    es = np.minimum(es, 0.99 * p)
    return EPSILON * es / (p - es)


def relative_humidity(temperature_k, pressure_pa, qv) -> np.ndarray:
    """``q_v / q_s`` over liquid water (dimensionless, may exceed 1)."""
    return np.asarray(qv, dtype=np.float64) / saturation_mixing_ratio(temperature_k, pressure_pa)


def dewpoint_k(pressure_pa, qv) -> np.ndarray:
    """Bolton's inversion of the saturation formula for the vapour
    pressure ``e = q p / (eps + q)``; a dry column reads a very cold dew
    point, never an error (``q`` is floored at 1e-12)."""
    q = np.maximum(np.asarray(qv, dtype=np.float64), 1.0e-12)
    e = q * np.asarray(pressure_pa, dtype=np.float64) / (EPSILON + q)
    ln = np.log(e / 611.2)
    return 273.15 + 243.5 * ln / (17.67 - ln)


def lcl_temperature_k(temperature_k, dewpoint) -> np.ndarray:
    """Bolton (1980) equation 15."""
    t = np.asarray(temperature_k, dtype=np.float64)
    td = np.minimum(np.asarray(dewpoint, dtype=np.float64), t)
    return 1.0 / (1.0 / (td - 56.0) + np.log(t / td) / 800.0) + 56.0


def equivalent_potential_temperature_k(temperature_k, pressure_pa, qv) -> np.ndarray:
    """Bolton (1980) equation 43 (the mixing ratio taken as ``q_v``)."""
    t = np.asarray(temperature_k, dtype=np.float64)
    p = np.asarray(pressure_pa, dtype=np.float64)
    q = np.maximum(np.asarray(qv, dtype=np.float64), 1.0e-12)
    tl = lcl_temperature_k(t, dewpoint_k(p, q))
    return (
        t * (REFERENCE_PRESSURE_PA / p) ** (0.2854 * (1.0 - 0.28 * q))
        * np.exp((3.376 / tl - 0.00254) * 1.0e3 * q * (1.0 + 0.81 * q))
    )


def _moist_lapse_dlnp(temperature_k: np.ndarray, pressure_pa: np.ndarray) -> np.ndarray:
    """``dT / d ln p`` along the pseudo-adiabat (Rogers and Yau form)."""
    qs = saturation_mixing_ratio(temperature_k, pressure_pa)
    lv = LATENT_HEAT_VAPORIZATION
    numerator = DRY_AIR_GAS_CONSTANT * temperature_k + lv * qs
    denominator = DRY_AIR_CP + lv * lv * qs * EPSILON / (
        DRY_AIR_GAS_CONSTANT * temperature_k * temperature_k
    )
    return numerator / denominator


@dataclass(frozen=True)
class ParcelResult:
    """``(n,)`` arrays per column."""

    cape_j_kg: np.ndarray
    cin_j_kg: np.ndarray
    lcl_pa: np.ndarray
    lfc_pa: np.ndarray
    origin_level: np.ndarray


def lift_parcel(temperature_k, qv, p_full, p_half, origin) -> ParcelResult:
    """Lift the parcel of level ``origin`` (per column, top-down index)
    through every layer above it.  Arrays are ``(nlev, n)`` and
    ``(nlev + 1, n)``; the surface is the last index."""
    t = np.asarray(temperature_k, dtype=np.float64)
    q = np.asarray(qv, dtype=np.float64)
    p = np.asarray(p_full, dtype=np.float64)
    ph = np.asarray(p_half, dtype=np.float64)
    nlev, n = t.shape
    cols = np.arange(n)
    origin = np.asarray(origin, dtype=np.int64)
    t0 = t[origin, cols]
    q0 = np.maximum(q[origin, cols], 1.0e-12)
    p0 = p[origin, cols]
    tlcl = lcl_temperature_k(t0, dewpoint_k(p0, q0))
    plcl = np.minimum(p0 * (tlcl / t0) ** (1.0 / KAPPA), p0)
    cape = np.zeros(n)
    cin = np.zeros(n)
    lfc = np.full(n, np.nan)
    found_lfc = np.zeros(n, dtype=bool)
    tp = t0.copy()
    pp = p0.copy()
    for k in range(nlev - 1, -1, -1):
        active = (k < origin) & (p[k] >= PARCEL_TOP_PA)
        if not np.any(active):
            continue
        target = p[k]
        # Dry stage down to the LCL, then pseudo-adiabatic.
        dry_to = np.maximum(target, plcl)
        dry = active & (pp > dry_to)
        tp = np.where(dry, tp * (dry_to / pp) ** KAPPA, tp)
        pp = np.where(dry, dry_to, pp)
        moist = active & (target < pp)
        if np.any(moist):
            lnp0 = np.log(pp)
            lnp1 = np.log(np.where(moist, target, pp))
            h = (lnp1 - lnp0) / MOIST_ASCENT_SUBSTEPS
            tt = tp.copy()
            lp = lnp0.copy()
            for _ in range(MOIST_ASCENT_SUBSTEPS):
                k1 = _moist_lapse_dlnp(tt, np.exp(lp))
                k2 = _moist_lapse_dlnp(tt + h * k1, np.exp(lp + h))
                tt = tt + 0.5 * h * (k1 + k2)
                lp = lp + h
            tp = np.where(moist, tt, tp)
            pp = np.where(moist, target, pp)
        qp = np.where(pp <= plcl, np.minimum(q0, saturation_mixing_ratio(tp, pp)), q0)
        tv_p = tp * (1.0 + 0.61 * qp)
        tv_e = t[k] * (1.0 + 0.61 * np.maximum(q[k], 0.0))
        dlnp = np.log(ph[k + 1] / ph[k])
        b = DRY_AIR_GAS_CONSTANT * (tv_p - tv_e) * dlnp
        positive = active & (b > 0.0)
        above_lcl = active & (p[k] <= plcl)
        newly = positive & above_lcl & ~found_lfc
        lfc = np.where(newly, p[k], lfc)
        found_lfc |= newly
        cape = np.where(positive, cape + b, cape)
        negative = active & (b < 0.0) & ~found_lfc
        cin = np.where(negative, cin + b, cin)
    return ParcelResult(cape, cin, plcl, lfc, origin)


def parcel_cape(temperature_k, qv, p_full, p_half) -> dict[str, np.ndarray]:
    """Surface-based and most-unstable CAPE/CIN of ``(nlev, ...)`` fields;
    every returned array has the trailing shape of the input."""
    t = np.asarray(temperature_k, dtype=np.float64)
    trailing = t.shape[1:]
    nlev = t.shape[0]
    t2 = t.reshape(nlev, -1)
    q2 = np.asarray(qv, dtype=np.float64).reshape(nlev, -1)
    p2 = np.asarray(p_full, dtype=np.float64).reshape(nlev, -1)
    ph2 = np.asarray(p_half, dtype=np.float64).reshape(nlev + 1, -1)
    n = t2.shape[1]
    sb = lift_parcel(t2, q2, p2, ph2, np.full(n, nlev - 1, dtype=np.int64))
    theta_e = equivalent_potential_temperature_k(t2, p2, q2)
    within = p2 >= (ph2[-1] - MOST_UNSTABLE_SEARCH_DEPTH_PA)[None]
    mu_origin = np.argmax(np.where(within, theta_e, -np.inf), axis=0)
    mu = lift_parcel(t2, q2, p2, ph2, mu_origin)
    out = {
        "cape_sb_j_kg": sb.cape_j_kg, "cin_sb_j_kg": sb.cin_j_kg,
        "lcl_sb_pa": sb.lcl_pa, "lfc_sb_pa": sb.lfc_pa,
        "cape_mu_j_kg": mu.cape_j_kg, "cin_mu_j_kg": mu.cin_j_kg,
        "lcl_mu_pa": mu.lcl_pa, "lfc_mu_pa": mu.lfc_pa,
        "mu_origin_level": mu.origin_level,
    }
    return {k: v.reshape(trailing) for k, v in out.items()}


@dataclass(frozen=True)
class Sounding:
    """One checkpoint's column fields on the model grid, float64, level
    index 0 at the model top."""

    time_s: float
    step: int
    p_full: np.ndarray
    p_half: np.ndarray
    temperature_k: np.ndarray
    qv: np.ndarray
    condensate: dict[str, np.ndarray]
    relative_humidity: np.ndarray
    omega_half_pa_s: np.ndarray | None
    convective_kg_m2: np.ndarray
    grid_scale_kg_m2: np.ndarray
    land: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray

    @property
    def dp_g(self) -> np.ndarray:
        return (self.p_half[1:] - self.p_half[:-1]) / GRAVITY_M_S2

    def column_integral(self, field) -> np.ndarray:
        return np.sum(np.asarray(field, dtype=np.float64) * self.dp_g, axis=0)

    @property
    def cloud_mixing_ratio(self) -> np.ndarray:
        return self.condensate["qc"] + self.condensate["qi"]

    def cloudy(self) -> np.ndarray:
        return self.cloud_mixing_ratio > CLOUDY_LEVEL_KG_KG


class SoundingReader:
    """Reads a run's checkpoints on the run's own geometry from its receipt."""

    def __init__(self, receipt: dict, *, omega: bool = True):
        from woof.globe.spectral.transform import SphericalHarmonicTransform
        from woof.globe.spectral.vector import VorticityDivergenceOperator

        from .vertical import HybridCoordinate

        cfg = receipt["config"]
        block = receipt["transform"]
        # A receipt names the grid; a config that leaves it to the
        # truncation (nlat/nlon None) gets the dealias rule's grid.
        self.transform = SphericalHarmonicTransform.create(
            int(cfg["truncation"]),
            nlat=None if block.get("nlat") is None else int(block["nlat"]),
            nlon=None if block.get("nlon") is None else int(block["nlon"]),
            dealias_factor=float(cfg["dealias_factor"]),
            radius_m=None if block.get("radius_m") is None else float(block["radius_m"]),
            backend="numpy",
            precision="float64",
        )
        self.vertical = HybridCoordinate(
            np.asarray(cfg["a_half_pa"], dtype=np.float64),
            np.asarray(cfg["b_half"], dtype=np.float64),
        )
        self.vector = VorticityDivergenceOperator(self.transform)
        self.omega = bool(omega)
        self.config_hash = receipt.get("config_hash")
        grid = self.transform.grid
        self.latitude_deg = np.asarray(grid.latitude_deg, dtype=np.float64)
        self.longitude_deg = np.asarray(grid.longitude_deg, dtype=np.float64)
        self.cell_weights = np.repeat(
            (np.asarray(grid.quadrature_weights, dtype=np.float64) / (2.0 * grid.nlon))[:, None],
            grid.nlon, axis=1,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.latitude_deg.size), int(self.longitude_deg.size)

    def _grid_field(self, value) -> np.ndarray:
        if np.iscomplexobj(value):
            return np.asarray(self.transform.inverse(value.astype(np.complex128)), dtype=np.float64)
        return np.asarray(value, dtype=np.float64)

    def omega_from_arrays(self, arrays: dict, ps: np.ndarray, pressure: dict) -> np.ndarray:
        """Interface pressure velocity (Pa/s, ``(nlev + 1, ny, nx)``) from
        the checkpointed vorticity, divergence and surface pressure."""
        vor = arrays["atmosphere__vorticity"].astype(np.complex128)
        div = arrays["atmosphere__divergence"].astype(np.complex128)
        u, v = self.vector.wind_from_vordiv(vor, div)
        divergence = self._grid_field(div)
        d_east, d_north = self.transform.gradient(
            arrays["atmosphere__log_surface_pressure"].astype(np.complex128)
        )
        delta_b = np.asarray(self.vertical.delta_b, dtype=np.float64)
        dp = np.asarray(pressure["dp"], dtype=np.float64)
        # div(v dp) = dp div(v) + v . grad(dp), grad(dp) = delta_b ps grad(ln ps).
        divm = dp * divergence + delta_b[:, None, None] * ps[None] * (
            np.asarray(u) * d_east[None] + np.asarray(v) * d_north[None]
        )
        ps_t = -np.sum(divm, axis=0)
        omega, _residual = self.vertical.continuity(divm, ps_t, self.transform.backend)
        return np.asarray(omega, dtype=np.float64)

    def sounding_from_arrays(self, metadata: dict, arrays: dict) -> Sounding:
        backend = self.transform.backend
        logps = self._grid_field(arrays["atmosphere__log_surface_pressure"])
        ps = np.exp(logps)
        pressure = self.vertical.pressure(ps, backend)
        p_full = np.asarray(pressure["p_full"], dtype=np.float64)
        p_half = np.asarray(pressure["p_half"], dtype=np.float64)
        theta = self._grid_field(arrays["atmosphere__theta"])
        temperature = theta * (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
        del theta
        qv = self._grid_field(arrays["atmosphere__qv"])
        condensate = {
            name: self._grid_field(arrays[f"atmosphere__{name}"]) for name in CONDENSATE_SPECIES
        }
        rh = relative_humidity(temperature, p_full, qv)
        omega = self.omega_from_arrays(arrays, ps, pressure) if self.omega else None
        names = set(arrays)
        ny, nx = self.shape
        if CHECKPOINT_CONVECTIVE in names:
            convective = np.asarray(arrays[CHECKPOINT_CONVECTIVE], dtype=np.float64)
        elif convective_accumulator_is_absent_by_construction(names, int(metadata["step"])):
            convective = np.zeros((ny, nx))
        else:
            raise ValueError(
                convective_accumulator_refusal(f"step {metadata['step']}", names, int(metadata["step"]))
            )
        grid_scale = sum(
            np.asarray(arrays[n], dtype=np.float64) for n in CHECKPOINT_GRID_SCALE_BUCKETS
        )
        land = np.asarray(arrays[CHECKPOINT_LAND_FRACTION], dtype=np.float64) >= 0.5
        return Sounding(
            time_s=float(metadata["time_s"]), step=int(metadata["step"]),
            p_full=p_full, p_half=p_half, temperature_k=temperature, qv=qv,
            condensate=condensate, relative_humidity=rh, omega_half_pa_s=omega,
            convective_kg_m2=convective, grid_scale_kg_m2=grid_scale, land=land,
            latitude_deg=self.latitude_deg, longitude_deg=self.longitude_deg,
        )

    def sounding(self, path: str | Path) -> Sounding:
        from .checkpoint import read_checkpoint

        metadata, arrays = read_checkpoint(path)
        return self.sounding_from_arrays(metadata, arrays)


def read_receipt(run_dir: str | Path) -> dict:
    import json

    path = Path(run_dir) / "arwen-global-receipt.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path}: the run receipt is needed for the grid and the vertical coordinate"
        )
    return json.loads(path.read_text())


def receipt_from_config(config: str | Path, shape: tuple[int, int] | None = None) -> dict:
    """A receipt-shaped mapping (the ``config`` and ``transform`` blocks the
    reader needs) from a run's TOML, for a run that has not written its
    receipt yet.  The grid is the config's, or ``shape`` when the config
    leaves it to the truncation."""
    from woof.globe.spectral.grid import EARTH_RADIUS_M

    from .config import load_config

    cfg = load_config(config)
    identity = dict(cfg.config_identity)
    nlat, nlon = identity.get("nlat"), identity.get("nlon")
    if (nlat is None or nlon is None) and shape is not None:
        nlat, nlon = int(shape[0]), int(shape[1])
    return {
        "config": identity,
        "transform": {"nlat": nlat, "nlon": nlon, "radius_m": EARTH_RADIUS_M},
        "config_hash": cfg.config_hash,
        "source": f"config {config} (no receipt)",
    }


def read_receipt_or_config(run_dir: str | Path, config: str | Path | None = None) -> dict:
    """The run's receipt when it exists, else the config's equivalent; a
    run without either is refused (the grid would be a guess)."""
    path = Path(run_dir) / "arwen-global-receipt.json"
    if path.exists():
        return read_receipt(run_dir)
    if config is None:
        raise FileNotFoundError(
            f"{path}: no receipt yet; pass the run's TOML so the grid and the vertical "
            "coordinate come from the configuration that is writing the checkpoints"
        )
    shape = None
    first = sorted(Path(run_dir).glob("arwen_global_step*.npz"))
    if first:
        with np.load(first[0], allow_pickle=False) as archive:
            if CHECKPOINT_LAND_FRACTION in archive.files:
                shape = tuple(archive[CHECKPOINT_LAND_FRACTION].shape)
    return receipt_from_config(config, shape)


def run_start_utc_s(receipt: dict) -> float | None:
    """The run's start instant from the receipt's physics options, or None."""
    from datetime import datetime, timezone

    options = (receipt.get("config") or {}).get("native_adapter_options") or {}
    text = options.get("start_time_utc")
    if not text:
        return None
    text = str(text)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(timezone.utc).timestamp()


__all__ = [
    "CLOUDY_LEVEL_KG_KG", "SATURATION_FORMULA", "Sounding", "SoundingReader",
    "ParcelResult", "lift_parcel", "parcel_cape", "relative_humidity",
    "saturation_mixing_ratio", "saturation_vapor_pressure_pa", "read_receipt",
    "read_receipt_or_config", "receipt_from_config", "run_start_utc_s",
]
