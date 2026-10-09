"""Fail-closed TOML schema for the Level-3 global spectral prototype."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import tomllib

import numpy as np

from .constants import (
    EARTH_RADIUS_M,
    EARTH_ROTATION_RATE_S,
    RESEARCH_ACKNOWLEDGEMENT,
    RUN_SCHEMA,
)
from .primitive import SURFACE_PRESSURE_CEILING_PA

# The transform builds dense (T+1, T+1, nlat) float64 tables for the basis,
# its meridional derivative and the Gram-corrected analysis, and solves a
# per-order Gram system on top of them.  Measured peak host allocation in
# SphericalHarmonicTransform.__post_init__ (numpy 2.2.6, float64): 769 MiB at
# T255 in 6.9 s, 1500 MiB at T319 in 17.0 s.  T255 is the last truncation the
# constructor fits under a gigabyte, and it is also the range the normalized
# Legendre recurrence in legendre.py was written and verified for.
MAXIMUM_TRUNCATION = 255
INITIAL_SURFACE_PRESSURE_FLOOR_PA = 50_000.0


@dataclass(frozen=True)
class GlobalSpectralRunConfig:
    name: str
    model: str
    acknowledgement: str
    truncation: int
    backend: str
    precision: str
    nlat: int | None
    nlon: int | None
    dealias_factor: float
    radius_m: float
    rotation_rate_s: float
    dt_s: float
    duration_s: float
    output_interval_s: float
    integrator: str
    diffusion_enabled: bool
    diffusion_order: int
    diffusion_efold_s: float
    diffusion_preserve_degree: int
    divergence_diffusion_strength: float
    pressure_diffusion_strength: float
    williamson_alpha_rad: float
    williamson_u0_m_s: float | None
    williamson_mean_geopotential: float
    sigma_half: tuple[float, ...]
    primitive_surface_pressure_pa: float
    primitive_surface_temperature_k: float
    primitive_top_temperature_k: float
    primitive_temperature_perturbation_k: float
    primitive_zonal_wavenumber: int
    held_suarez: bool
    mass_fixer: bool
    maximum_cfl: float
    gate_transform_roundtrip: float
    gate_transform_parseval: float
    gate_mass_relative_drift: float
    gate_williamson_l2: float

    @property
    def steps(self) -> int:
        return int(round(self.duration_s / self.dt_s))

    @property
    def output_steps(self) -> int:
        return int(round(self.output_interval_s / self.dt_s))

    @property
    def nlev(self) -> int:
        return len(self.sigma_half) - 1

    def canonical_dict(self) -> dict:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        raw = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(raw).hexdigest()


def _table(raw: dict, name: str) -> dict:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return value


def _unknown(table: dict, known: set[str], where: str) -> None:
    extra = sorted(set(table) - known)
    if extra:
        raise ValueError(f"unknown key(s) in [{where}]: {extra}")


def _whole_multiple(value: float, dt: float, name: str) -> None:
    ratio = value / dt
    if abs(ratio - round(ratio)) > 1.0e-10:
        raise ValueError(
            f"{name}={value:g} must be a whole multiple of dt_s={dt:g}"
        )


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a TOML boolean, got {value!r}")
    return bool(value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a TOML integer, got {value!r}")
    return int(value)


def _string(value: object, name: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a TOML string, got {value!r}")
    result = value.strip()
    if nonempty and not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _positive_gate(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def load_config(path: str | Path) -> GlobalSpectralRunConfig:
    source = Path(path)
    with source.open("rb") as stream:
        raw = tomllib.load(stream)
    known_top = {
        "global_spectral",
        "grid",
        "time",
        "diffusion",
        "shallow_water",
        "primitive",
        "gates",
    }
    extra_top = sorted(set(raw) - known_top)
    if extra_top:
        raise ValueError(f"unknown top-level table(s) in {source}: {extra_top}")

    root = _table(raw, "global_spectral")
    _unknown(
        root,
        {"schema", "name", "model", "acknowledgement", "backend", "precision"},
        "global_spectral",
    )
    if root.get("schema") != RUN_SCHEMA:
        raise ValueError(f"[global_spectral].schema must be {RUN_SCHEMA!r}")
    acknowledgement = _string(
        root.get("acknowledgement", ""), "global_spectral.acknowledgement"
    )
    if acknowledgement != RESEARCH_ACKNOWLEDGEMENT:
        raise ValueError(
            "the global spectral dycore is research-only; set "
            f"acknowledgement={RESEARCH_ACKNOWLEDGEMENT!r} exactly"
        )
    model = _string(root.get("model", ""), "global_spectral.model").lower()
    if model not in {"shallow-water", "primitive-dry"}:
        raise ValueError("model must be 'shallow-water' or 'primitive-dry'")
    name = _string(root.get("name", ""), "global_spectral.name")
    backend = _string(
        root.get("backend", "numpy"), "global_spectral.backend"
    ).lower()
    precision = _string(
        root.get("precision", "float64"), "global_spectral.precision"
    ).lower()
    if backend not in {"numpy", "cupy"}:
        raise ValueError("backend must be 'numpy' or 'cupy'")
    if precision not in {"float32", "float64"}:
        raise ValueError("precision must be 'float32' or 'float64'")

    grid = _table(raw, "grid")
    _unknown(
        grid,
        {"truncation", "nlat", "nlon", "dealias_factor", "radius_m", "rotation_rate_s"},
        "grid",
    )
    truncation = _integer(grid.get("truncation", 15), "grid.truncation")
    if truncation < 3:
        raise ValueError("truncation must be >= 3")
    if truncation > MAXIMUM_TRUNCATION:
        raise ValueError(
            f"truncation T{truncation} exceeds T{MAXIMUM_TRUNCATION}: the "
            "transform's dense Legendre and Gram-corrected analysis tables "
            "peak at 769 MiB of host memory at T255 and 1500 MiB at T319, so "
            "a larger truncation exhausts host memory in the transform "
            "constructor before a single step is integrated"
        )
    nlat = (
        None
        if grid.get("nlat") is None
        else _integer(grid["nlat"], "grid.nlat")
    )
    nlon = (
        None
        if grid.get("nlon") is None
        else _integer(grid["nlon"], "grid.nlon")
    )
    dealias = _finite(grid.get("dealias_factor", 1.5), "grid.dealias_factor")
    if dealias < 1.0:
        raise ValueError("grid.dealias_factor must be >= 1")
    required_lat = max(
        truncation + 1, int(math.ceil(dealias * (truncation + 1)))
    )
    required_lon = max(
        2 * truncation + 1,
        int(math.ceil(2.0 * dealias * (truncation + 1))),
    )
    if nlat is not None and nlat < required_lat:
        raise ValueError(
            f"nlat={nlat} is too small for T{truncation} with "
            f"dealias_factor={dealias:g}; need at least {required_lat}"
        )
    if nlon is not None:
        if nlon % 2:
            raise ValueError("explicit grid.nlon must be even")
        if nlon < required_lon:
            raise ValueError(
                f"nlon={nlon} is too small for T{truncation} with "
                f"dealias_factor={dealias:g}; need at least {required_lon}"
            )
    radius = _finite(grid.get("radius_m", EARTH_RADIUS_M), "grid.radius_m")
    if radius <= 0.0:
        raise ValueError("grid.radius_m must be positive")
    omega = _finite(
        grid.get("rotation_rate_s", EARTH_ROTATION_RATE_S),
        "grid.rotation_rate_s",
    )
    if omega < 0.0:
        raise ValueError("grid.rotation_rate_s must be nonnegative")

    time_table = _table(raw, "time")
    _unknown(
        time_table,
        {"dt_s", "duration_s", "output_interval_s", "integrator", "maximum_cfl"},
        "time",
    )
    dt = _finite(time_table.get("dt_s", 300.0), "time.dt_s")
    duration = _finite(time_table.get("duration_s", 21_600.0), "time.duration_s")
    output_interval = _finite(
        time_table.get("output_interval_s", duration), "time.output_interval_s"
    )
    integrator = _string(
        time_table.get("integrator", "ssprk3"), "time.integrator"
    ).lower()
    maximum_cfl = _finite(
        time_table.get("maximum_cfl", 0.65 if model == "primitive-dry" else 0.95),
        "time.maximum_cfl",
    )
    for key, value in {
        "time.dt_s": dt,
        "time.duration_s": duration,
        "time.output_interval_s": output_interval,
    }.items():
        if value <= 0.0:
            raise ValueError(f"{key} must be positive")
    if integrator not in {"ssprk3", "rk4"}:
        raise ValueError("integrator must be 'ssprk3' or 'rk4'")
    _whole_multiple(duration, dt, "duration_s")
    _whole_multiple(output_interval, dt, "output_interval_s")
    if output_interval > duration:
        raise ValueError("output_interval_s cannot exceed duration_s")
    if not 0.0 < maximum_cfl <= 0.95:
        raise ValueError("maximum_cfl must lie in (0, 0.95]")

    diff = _table(raw, "diffusion")
    _unknown(
        diff,
        {
            "enabled",
            "order",
            "e_folding_time_s_at_truncation",
            "preserve_degree",
            "divergence_strength",
            "pressure_strength",
        },
        "diffusion",
    )
    diffusion_enabled = _strict_bool(diff.get("enabled", True), "diffusion.enabled")
    diffusion_order = _integer(diff.get("order", 4), "diffusion.order")
    diffusion_efold = _finite(
        diff.get("e_folding_time_s_at_truncation", 21_600.0),
        "diffusion.e_folding_time_s_at_truncation",
    )
    diffusion_preserve = _integer(
        diff.get("preserve_degree", 1), "diffusion.preserve_degree"
    )
    divergence_strength = _finite(
        diff.get("divergence_strength", 1.5 if model == "primitive-dry" else 1.0),
        "diffusion.divergence_strength",
    )
    # Scales the hyperdiffusion of the model's mass field: log surface
    # pressure for primitive-dry, geopotential for shallow-water.  The
    # shallow-water default is the neutral 1.0 the model has always applied
    # to its geopotential anomaly.
    pressure_strength = _finite(
        diff.get("pressure_strength", 0.25 if model == "primitive-dry" else 1.0),
        "diffusion.pressure_strength",
    )
    if diffusion_order < 1:
        raise ValueError("diffusion.order must be >= 1")
    if diffusion_efold <= 0.0:
        raise ValueError("diffusion e-folding time must be positive")
    if not 0 <= diffusion_preserve <= truncation:
        raise ValueError("diffusion.preserve_degree must lie in 0..truncation")
    if divergence_strength < 0.0 or pressure_strength < 0.0:
        raise ValueError("diffusion strengths must be nonnegative")

    sw = _table(raw, "shallow_water")
    _unknown(sw, {"alpha_rad", "u0_m_s", "mean_geopotential_m2_s2"}, "shallow_water")
    primitive = _table(raw, "primitive")
    _unknown(
        primitive,
        {
            "sigma_half",
            "nlev",
            "surface_pressure_pa",
            "surface_temperature_k",
            "top_temperature_k",
            "temperature_perturbation_k",
            "zonal_wavenumber",
            "held_suarez",
            "mass_fixer",
        },
        "primitive",
    )
    if model == "shallow-water" and "primitive" in raw:
        raise ValueError(
            "[primitive] is not read by model='shallow-water'; remove the table "
            "rather than carrying ignored settings"
        )
    if model == "primitive-dry" and "shallow_water" in raw:
        raise ValueError(
            "[shallow_water] is not read by model='primitive-dry'; remove the "
            "table rather than carrying ignored settings"
        )

    alpha = _finite(sw.get("alpha_rad", 0.0), "shallow_water.alpha_rad")
    if model == "shallow-water" and abs(alpha) > 1.0e-14:
        raise ValueError(
            "research gate: Williamson test case 2 is admitted only at "
            "alpha_rad=0 in this implementation. The tilted-axis analytic "
            "state does not yet close the discrete Coriolis balance and is "
            "refused rather than presented as verified."
        )
    u0 = None if sw.get("u0_m_s") is None else _finite(
        sw["u0_m_s"], "shallow_water.u0_m_s"
    )
    mean_phi = _finite(
        sw.get("mean_geopotential_m2_s2", 2.94e4),
        "shallow_water.mean_geopotential_m2_s2",
    )
    if mean_phi <= 0.0:
        raise ValueError("shallow_water.mean_geopotential_m2_s2 must be positive")

    if "sigma_half" in primitive and "nlev" in primitive:
        raise ValueError("set either primitive.sigma_half or primitive.nlev, not both")
    if "sigma_half" in primitive:
        raw_sigma = primitive["sigma_half"]
        if not isinstance(raw_sigma, list):
            raise ValueError("primitive.sigma_half must be a TOML array")
        sigma_half = tuple(
            _finite(x, "primitive.sigma_half[]") for x in raw_sigma
        )
    else:
        nlev = _integer(primitive.get("nlev", 4), "primitive.nlev")
        if not 2 <= nlev <= 32:
            raise ValueError("primitive.nlev must lie in 2..32 for this explicit research core")
        sigma_half = tuple(np.linspace(0.0, 1.0, nlev + 1).tolist())
    if (
        len(sigma_half) < 3
        or abs(sigma_half[0]) > 1.0e-14
        or abs(sigma_half[-1] - 1.0) > 1.0e-14
    ):
        raise ValueError(
            "primitive sigma_half must start at 0, end at 1, and define >=2 layers"
        )
    if len(sigma_half) - 1 > 32:
        raise ValueError("primitive sigma_half defines more than the admitted 32 layers")
    if any(b <= a for a, b in zip(sigma_half, sigma_half[1:])):
        raise ValueError("primitive sigma_half must be strictly increasing")
    ps = _finite(primitive.get("surface_pressure_pa", 100_000.0), "primitive.surface_pressure_pa")
    ts = _finite(primitive.get("surface_temperature_k", 288.0), "primitive.surface_temperature_k")
    tt = _finite(primitive.get("top_temperature_k", 215.0), "primitive.top_temperature_k")
    perturb = _finite(
        primitive.get("temperature_perturbation_k", 0.25),
        "primitive.temperature_perturbation_k",
    )
    wave = _integer(
        primitive.get("zonal_wavenumber", 4), "primitive.zonal_wavenumber"
    )
    held = _strict_bool(primitive.get("held_suarez", False), "primitive.held_suarez")
    mass_fixer = _strict_bool(primitive.get("mass_fixer", True), "primitive.mass_fixer")
    if not INITIAL_SURFACE_PRESSURE_FLOOR_PA <= ps <= SURFACE_PRESSURE_CEILING_PA:
        raise ValueError(
            "primitive.surface_pressure_pa must lie in "
            f"{INITIAL_SURFACE_PRESSURE_FLOOR_PA:g}..{SURFACE_PRESSURE_CEILING_PA:g}"
        )
    if not 150.0 <= tt <= 350.0 or not 150.0 <= ts <= 350.0:
        raise ValueError("primitive top/surface temperatures must lie in 150..350 K")
    if abs(perturb) > 10.0:
        raise ValueError("primitive.temperature_perturbation_k exceeds the 10 K research bound")
    if not 0 <= wave <= truncation:
        raise ValueError("primitive.zonal_wavenumber must lie in 0..truncation")

    gates = _table(raw, "gates")
    _unknown(
        gates,
        {
            "transform_roundtrip_relative_linf",
            "transform_parseval_relative_error",
            "mass_relative_drift",
            "williamson_normalized_l2",
        },
        "gates",
    )
    roundtrip_gate = _positive_gate(
        gates.get(
            "transform_roundtrip_relative_linf",
            5.0e-11 if precision == "float64" else 5.0e-5,
        ),
        "gates.transform_roundtrip_relative_linf",
    )
    parseval_gate = _positive_gate(
        gates.get(
            "transform_parseval_relative_error",
            5.0e-12 if precision == "float64" else 5.0e-5,
        ),
        "gates.transform_parseval_relative_error",
    )
    mass_gate = _positive_gate(
        gates.get("mass_relative_drift", 1.0e-10 if mass_fixer else 1.0e-6),
        "gates.mass_relative_drift",
    )
    williamson_gate = _positive_gate(
        gates.get("williamson_normalized_l2", 1.0e-8),
        "gates.williamson_normalized_l2",
    )

    return GlobalSpectralRunConfig(
        name=name,
        model=model,
        acknowledgement=acknowledgement,
        truncation=truncation,
        backend=backend,
        precision=precision,
        nlat=nlat,
        nlon=nlon,
        dealias_factor=dealias,
        radius_m=radius,
        rotation_rate_s=omega,
        dt_s=dt,
        duration_s=duration,
        output_interval_s=output_interval,
        integrator=integrator,
        diffusion_enabled=diffusion_enabled,
        diffusion_order=diffusion_order,
        diffusion_efold_s=diffusion_efold,
        diffusion_preserve_degree=diffusion_preserve,
        divergence_diffusion_strength=divergence_strength,
        pressure_diffusion_strength=pressure_strength,
        williamson_alpha_rad=alpha,
        williamson_u0_m_s=u0,
        williamson_mean_geopotential=mean_phi,
        sigma_half=sigma_half,
        primitive_surface_pressure_pa=ps,
        primitive_surface_temperature_k=ts,
        primitive_top_temperature_k=tt,
        primitive_temperature_perturbation_k=perturb,
        primitive_zonal_wavenumber=wave,
        held_suarez=held,
        mass_fixer=mass_fixer,
        maximum_cfl=maximum_cfl,
        gate_transform_roundtrip=roundtrip_gate,
        gate_transform_parseval=parseval_gate,
        gate_mass_relative_drift=mass_gate,
        gate_williamson_l2=williamson_gate,
    )
