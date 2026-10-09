"""Typed atmospheric, surface, and persistent physics state for WOOF global."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping

import numpy as np

from .constants import (
    ADVECTED_TRACERS,
    GRID_TRACERS,
    NUMBER_MOMENTS,
    PHYSICS_STATE_SCHEMA,
    PROGNOSTIC_FIELDS,
    SPECTRAL_FIELDS,
    WATER_SPECIES,
)


@dataclass
class MoistHybridState:
    """Prognostic atmosphere: five spectral fields and ten grid tracers.

    The spectral fields (``SPECTRAL_FIELDS``: vorticity, divergence,
    theta, log surface pressure, water vapor) have shape ``(nlev, T+1,
    T+1)``, ``log_surface_pressure`` ``(T+1, T+1)``; the triangular
    projection is owned by the transform rather than encoded in the
    shape.  The grid tracers (``GRID_TRACERS``: the five condensate
    species and the five number moments) are real ``(nlev, nlat, nlon)``
    arrays on the Gaussian grid, nonnegative by construction, carried by
    the positive-definite flux-form transport and never analysed into
    the spectral basis (finding 2026-09-02).

    :meth:`fields` and :meth:`with_fields` address the SPECTRAL fields:
    they are what the explicit integrator, the semi-implicit maps, the
    hyperdiffusion and the fixers combine linearly, and the grid tracers
    ride through those maps by reference (no copy).  :meth:`grid_tracers`
    and :meth:`with_grid_tracers` address the grid tracers.  A tendency
    state (the right-hand side) may leave the grid tracers ``None``: it
    has no grid-tracer rows and nothing reads them there.
    """

    vorticity: object
    divergence: object
    theta: object
    log_surface_pressure: object
    qv: object
    qc: object = None
    qr: object = None
    qi: object = None
    qs: object = None
    qg: object = None
    nc: object = None
    nr: object = None
    ni: object = None
    ns: object = None
    ng: object = None
    time_s: float = 0.0
    step: int = 0

    def fields(self) -> tuple[object, ...]:
        """The spectral fields, in ``SPECTRAL_FIELDS`` order."""
        return tuple(getattr(self, name) for name in SPECTRAL_FIELDS)

    def with_fields(
        self,
        fields: Iterable[object],
        *,
        time_s: float | None = None,
        step: int | None = None,
    ) -> "MoistHybridState":
        """A copy with the spectral fields replaced.

        Accepts the five spectral fields (the grid tracers are carried
        from ``self`` by reference), the ten Level-4 fields (the four
        dynamical fields and six water species; the number moments are
        carried, never silently zeroed) or all fifteen prognostic fields.
        """
        values = tuple(fields)
        if len(values) == len(SPECTRAL_FIELDS):
            kwargs = dict(zip(SPECTRAL_FIELDS, values))
        elif len(values) == len(PROGNOSTIC_FIELDS):
            kwargs = dict(zip(PROGNOSTIC_FIELDS, values))
        elif len(values) == len(PROGNOSTIC_FIELDS) - len(NUMBER_MOMENTS):
            legacy_names = SPECTRAL_FIELDS[:4] + WATER_SPECIES
            kwargs = dict(zip(legacy_names, values))
        else:
            raise ValueError(
                f"expected {len(SPECTRAL_FIELDS)} spectral fields, "
                f"{len(PROGNOSTIC_FIELDS)} prognostic fields or "
                f"{len(PROGNOSTIC_FIELDS) - len(NUMBER_MOMENTS)} Level-4 "
                f"fields, got {len(values)}"
            )
        kwargs.update(
            time_s=self.time_s if time_s is None else float(time_s),
            step=self.step if step is None else int(step),
        )
        return replace(self, **kwargs)

    def grid_tracers(self) -> dict[str, object]:
        """The grid tracers by name, in ``GRID_TRACERS`` order."""
        return {name: getattr(self, name) for name in GRID_TRACERS}

    def with_grid_tracers(
        self, tracers: Mapping[str, object]
    ) -> "MoistHybridState":
        """A copy with the named grid tracers replaced; the spectral
        fields and the unnamed tracers are carried by reference."""
        unknown = set(tracers) - set(GRID_TRACERS)
        if unknown:
            raise ValueError(
                f"unknown grid tracers {sorted(unknown)}; the grid tracers "
                f"are {list(GRID_TRACERS)}"
            )
        return replace(self, **dict(tracers))

    def prognostic_fields(self) -> tuple[object, ...]:
        """Every prognostic field, in ``PROGNOSTIC_FIELDS`` order."""
        return tuple(getattr(self, name) for name in PROGNOSTIC_FIELDS)

    def water_fields(self) -> tuple[object, ...]:
        return tuple(getattr(self, name) for name in WATER_SPECIES)

    def moment_fields(self) -> tuple[object, ...]:
        return tuple(getattr(self, name) for name in NUMBER_MOMENTS)

    def tracer_fields(self) -> tuple[object, ...]:
        return tuple(getattr(self, name) for name in ADVECTED_TRACERS)


#: SurfaceState member -> stored/exported array name, in checkpoint order.
#: The static-field members (from ``landuse_category`` on) are seeded once
#: by :mod:`woof.globe.statics` -- real WPS_GEOG fields or the
#: declared synthetic planet -- and carried unchanged; categories are
#: stored as floats (exact for small integers) so every surface array
#: shares one dtype through the checkpoint and the exchange.
SURFACE_ARRAY_NAMES = {
    "temperature_k": "surface_temperature_k",
    "water_kg_m2": "surface_water_kg_m2",
    "land_fraction": "land_fraction",
    "albedo": "surface_albedo",
    "emissivity": "surface_emissivity",
    "roughness_m": "surface_roughness_m",
    "heat_capacity_j_m2_k": "surface_heat_capacity_j_m2_k",
    "soil_temperature_k": "soil_temperature_k",
    "soil_water_fraction": "soil_water_fraction",
    "accumulated_rain_kg_m2": "accumulated_rain_kg_m2",
    "accumulated_snow_kg_m2": "accumulated_snow_kg_m2",
    "accumulated_graupel_kg_m2": "accumulated_graupel_kg_m2",
    "landuse_category": "landuse_category",
    "soil_category_top": "soil_category_top",
    "soil_category_bottom": "soil_category_bottom",
    "vegetation_fraction": "vegetation_fraction",
    "vegetation_fraction_min": "vegetation_fraction_min",
    "vegetation_fraction_max": "vegetation_fraction_max",
    "leaf_area_index": "leaf_area_index",
    "background_albedo": "background_albedo",
    "snow_albedo": "snow_albedo",
    "deep_soil_temperature_k": "deep_soil_temperature_k",
    # Seeded from the analysis by woof.globe.surface_seeding
    # (cold-start lane, 2026-09-04): the sea-ice fraction is the
    # runtime's second land/water rule (a column at or above one half is
    # a frozen surface, xland = 1) and the thickness feeds the frozen-
    # surface conduction.  Absent in checkpoints written before the
    # seeding; the reader fills zeros and says so (checkpoint.py).
    "sea_ice_fraction": "sea_ice_fraction",
    "sea_ice_thickness_m": "sea_ice_thickness_m",
    # The static lake share of each column (LANDUSEF of the land-use
    # set's lake class, 0 on the synthetic planet), seeded by
    # woof.globe.statics: an open-water column whose water is mostly
    # lake is an inland lake (statics.lake_columns, WRF's LAKEMASK rule)
    # and integrates its skin from its own surface energy budget
    # (native_runtime._lake_surface_step).  Absent in checkpoints written
    # before it; the reader fills zeros and says so (checkpoint.py).
    "lake_fraction": "lake_fraction",
}
#: Surface members a checkpoint from before their seeding lacks; filled
#: with zero on read (the ice-free planet with no lakes that checkpoint
#: ran: its open water held the starting skin for the whole run).
SEEDED_SURFACE_MEMBERS = ("sea_ice_fraction", "sea_ice_thickness_m", "lake_fraction")


@dataclass
class SurfaceState:
    """Grid-resident surface and land/ocean reservoirs, plus the static
    surface fields the land surface is driven with.

    ``landuse_category``/``soil_category_top``/``soil_category_bottom`` are
    one-based MODIS and STAS categories (lake already folded to water,
    water soil forced on water columns); ``vegetation_fraction`` and its
    annual min/max, ``leaf_area_index``, ``background_albedo`` and
    ``snow_albedo`` are fractions (LAI in m2/m2) resolved to the run's
    start date; ``deep_soil_temperature_k`` is Noah's lower boundary.
    ``sea_ice_fraction`` (0 to 1) and ``sea_ice_thickness_m`` are the
    analysis's sea ice, seeded once and carried; a builder that has no
    ice (the analytic planet, a Level-4 migration) leaves them ``None``
    and they materialize as zero planes shaped like the land fraction.
    ``lake_fraction`` (0 to 1) is the statics' lake share of the column,
    the same way: ``None`` is a planet without lakes.
    """

    temperature_k: object
    water_kg_m2: object
    land_fraction: object
    albedo: object
    emissivity: object
    roughness_m: object
    heat_capacity_j_m2_k: object
    soil_temperature_k: object
    soil_water_fraction: object
    accumulated_rain_kg_m2: object
    accumulated_snow_kg_m2: object
    accumulated_graupel_kg_m2: object
    landuse_category: object
    soil_category_top: object
    soil_category_bottom: object
    vegetation_fraction: object
    vegetation_fraction_min: object
    vegetation_fraction_max: object
    leaf_area_index: object
    background_albedo: object
    snow_albedo: object
    deep_soil_temperature_k: object
    sea_ice_fraction: object = None
    sea_ice_thickness_m: object = None
    lake_fraction: object = None

    def __post_init__(self) -> None:
        for member in SEEDED_SURFACE_MEMBERS:
            if getattr(self, member) is None:
                # Same array module, dtype and shape as the land fraction:
                # an ice-free plane, not a hidden constant the physics
                # could mistake for a seeded one (the receipt names the
                # seeding source separately).
                setattr(self, member, 0.0 * self.land_fraction)

    def arrays(self) -> dict[str, object]:
        return {
            stored: getattr(self, member)
            for member, stored in SURFACE_ARRAY_NAMES.items()
        }

    def copy(self) -> "SurfaceState":
        return SurfaceState(**{
            member: getattr(self, member).copy()
            for member in SURFACE_ARRAY_NAMES
        })


@dataclass
class PhysicsState:
    """Checkpointed gridpoint state owned by a native physics adapter.

    Arrays are named, numeric, and backend-resident. Metadata is restricted to
    JSON scalar/list/dict values by the checkpoint writer. Constant lookup
    tables and callable objects are deliberately reconstructed after restart.
    """

    schema: str = PHYSICS_STATE_SCHEMA
    arrays: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def copy(self) -> "PhysicsState":
        return PhysicsState(
            schema=self.schema,
            arrays={name: value.copy() for name, value in self.arrays.items()},
            metadata=_copy_json(self.metadata),
        )

    def validate(self) -> None:
        if self.schema != PHYSICS_STATE_SCHEMA:
            raise ValueError(f"physics-state schema mismatch: {self.schema!r}")
        for name, value in self.arrays.items():
            if not isinstance(name, str) or not name or name.startswith("__"):
                raise ValueError(f"invalid physics-state array name {name!r}")
            if not hasattr(value, "shape") or not hasattr(value, "dtype"):
                raise TypeError(f"physics-state array {name!r} is not numeric")
        _validate_json(self.metadata, "physics_state.metadata")


@dataclass
class ArwenGlobalState:
    atmosphere: MoistHybridState
    surface: SurfaceState
    physics_state: PhysicsState = field(default_factory=PhysicsState)

    @property
    def time_s(self) -> float:
        return float(self.atmosphere.time_s)

    @property
    def step(self) -> int:
        return int(self.atmosphere.step)

    def copy(self) -> "ArwenGlobalState":
        atmosphere = self.atmosphere.with_fields(
            [value.copy() for value in self.atmosphere.fields()]
        ).with_grid_tracers({
            name: value.copy()
            for name, value in self.atmosphere.grid_tracers().items()
            if value is not None
        })
        return ArwenGlobalState(
            atmosphere, self.surface.copy(), self.physics_state.copy()
        )


@dataclass
class PhysicsTendencies:
    du: object
    dv: object
    dtheta: object
    dqv: object
    dqc: object
    dqr: object
    dqi: object
    dqs: object
    dqg: object
    surface_temperature_tendency: object
    surface_water_tendency: object
    rain_rate_kg_m2_s: object
    snow_rate_kg_m2_s: object
    graupel_rate_kg_m2_s: object
    dnc: object | None = None
    dnr: object | None = None
    dni: object | None = None
    dns: object | None = None
    dng: object | None = None

    def water_tendencies(self) -> tuple[object, ...]:
        return self.dqv, self.dqc, self.dqr, self.dqi, self.dqs, self.dqg

    def moment_tendencies(self) -> tuple[object | None, ...]:
        return self.dnc, self.dnr, self.dni, self.dns, self.dng


def _copy_json(value):
    if isinstance(value, dict):
        return {str(k): _copy_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_json(v) for v in value]
    if isinstance(value, tuple):
        return [_copy_json(v) for v in value]
    return value


def _validate_json(value, path: str) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json(child, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            _validate_json(child, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


def to_metal_atmosphere(atmosphere: MoistHybridState, context) -> MoistHybridState:
    """Move atmospheric state arrays to one Metal context in FP64 storage.

    ``MetalArrayModule.asarray`` returns a compatible resident array unchanged,
    so this is also the stage-boundary operation for outputs that are already
    on the run's device.  Spectral coefficients are complex128, grid-point
    prognostics are float64, and already-resident arrays with the wrong dtype
    or context are rejected instead of being copied back to the host.  Scalar
    clocks and JSON metadata stay on the host.
    """
    from woof.metal.array import MetalArray, MetalArrayModule

    module = MetalArrayModule(context)

    def resident(value, dtype=None):
        if value is None:
            return None
        if isinstance(value, MetalArray):
            if value.buffer.context is not context:
                raise ValueError("Global state contains arrays from another Metal context")
            if dtype is not None and value.dtype != np.dtype(dtype):
                raise TypeError(
                    "resident Global state arrays must already have the "
                    f"required dtype {np.dtype(dtype)}, got {value.dtype}"
                )
            return value
        return module.asarray(value, dtype=dtype)

    return replace(
        atmosphere,
        **{
            name: resident(
                getattr(atmosphere, name),
                np.complex128 if name in SPECTRAL_FIELDS else np.float64,
            )
            for name in PROGNOSTIC_FIELDS
        },
    )


def to_metal_state(bundle: ArwenGlobalState, context) -> ArwenGlobalState:
    """Move all numeric Global state arrays to one Metal context."""
    from woof.metal.array import MetalArray, MetalArrayModule

    module = MetalArrayModule(context)

    def resident(value, dtype=None):
        if value is None:
            return None
        if isinstance(value, MetalArray):
            if value.buffer.context is not context:
                raise ValueError("Global state contains arrays from another Metal context")
            if dtype is not None and value.dtype != np.dtype(dtype):
                raise TypeError(
                    "resident Global state arrays must already have the "
                    f"required dtype {np.dtype(dtype)}, got {value.dtype}"
                )
            return value
        return module.asarray(value, dtype=dtype)

    atmosphere = to_metal_atmosphere(bundle.atmosphere, context)
    surface = replace(
        bundle.surface,
        **{member: resident(getattr(bundle.surface, member))
           for member in SURFACE_ARRAY_NAMES},
    )
    physics_state = replace(
        bundle.physics_state,
        arrays={name: resident(value)
                for name, value in bundle.physics_state.arrays.items()},
        metadata=_copy_json(bundle.physics_state.metadata),
    )
    return ArwenGlobalState(atmosphere, surface, physics_state)


def to_host_atmosphere(atmosphere: MoistHybridState) -> MoistHybridState:
    """Materialize resident atmospheric arrays for a host-only stage."""
    from woof.metal.array import MetalArray

    def host(value):
        return value.to_numpy() if isinstance(value, MetalArray) else value

    return replace(
        atmosphere,
        **{name: host(getattr(atmosphere, name)) for name in PROGNOSTIC_FIELDS},
    )


def to_host_state(bundle: ArwenGlobalState) -> ArwenGlobalState:
    """Materialize resident state arrays for diagnostics or disk I/O."""
    from woof.metal.array import MetalArray

    def host(value):
        return value.to_numpy() if isinstance(value, MetalArray) else value

    atmosphere = to_host_atmosphere(bundle.atmosphere)
    surface = replace(
        bundle.surface,
        **{member: host(getattr(bundle.surface, member))
           for member in SURFACE_ARRAY_NAMES},
    )
    physics_state = replace(
        bundle.physics_state,
        arrays={name: host(value)
                for name, value in bundle.physics_state.arrays.items()},
        metadata=_copy_json(bundle.physics_state.metadata),
    )
    return ArwenGlobalState(atmosphere, surface, physics_state)


__all__ = [
    "SEEDED_SURFACE_MEMBERS",
    "SURFACE_ARRAY_NAMES",
    "ArwenGlobalState",
    "MoistHybridState",
    "PhysicsState",
    "PhysicsTendencies",
    "SurfaceState",
    "to_host_state",
    "to_host_atmosphere",
    "to_metal_atmosphere",
    "to_metal_state",
]
