"""Per-step conservation ledger, computed on the device as one small vector.

Every term is an AREA-WEIGHTED GLOBAL MEAN (the Gaussian quadrature mean
``grid.global_mean`` uses: 0.5 * sum_j w_j * mean_i f_ji), so a column
integral appears as "per unit area" and the terms have the units their
names carry.  Column integrals are ``sum_k f_k dp_k / g``.  Reductions
accumulate in float64 whatever the state's precision; the products inside
them carry the state's own precision.

Definitions (the audit of 2026-09-01, dycore lens DN-8, found no discrete
energy or angular-momentum identity in the discretisation and, thermo lens
VTW-5, no moist energy budget anywhere; these terms are the budgets those
findings ask for):

* mass_pa: global-mean surface pressure = g * total (moist) column mass.
* water_<species>_kg_m2: column water per species; water_atmosphere is
  their sum; water_surface / soil / native / outflow are the reservoir
  accounts of water.total_water_column, and water_total is the held +
  booked-exit total the global water fixer and the drift gate read.
* dry_enthalpy_j_m2 = int cp T dp/g.
* kinetic_energy_j_m2 = int 0.5 (u^2 + v^2) dp/g.
* potential_energy_j_m2 = int Phi dp/g, Phi the model's own full-level
  geopotential (gravitational potential energy above sea level).
* surface_potential_energy_j_m2 = Phi_s ps / g.
* total_dry_energy_j_m2 = dry_enthalpy + kinetic + surface_potential.
  This is the hydrostatic primitive-equation invariant
  int (cp T + K) dp/g + Phi_s ps/g (Kasahara 1974); it is NOT
  dry_enthalpy + kinetic + potential_energy, which double-counts int R T
  dp/g (int Phi dp/g = Phi_s ps/g + int R T dp/g by parts).
* latent_vapor_j_m2 = int Lv qv dp/g; latent_frozen_j_m2 = int Lf (qi +
  qs + qg) dp/g.  The combination the suite's own phase conversions
  conserve is cp T + Lv qv - Lf q_frozen (liquid water as the zero of
  latent energy: condensation moves cp dT = +Lv dq, freezing cp dT = +Lf
  dq_frozen, sublimation cp dT = -Ls dq with Ls = Lv + Lf), so
  moist_total_energy_j_m2 = total_dry + latent_vapor - latent_frozen.
* relative_angular_momentum_kg_s = int u a cos(phi) dp/g (per unit area);
  axial_angular_momentum_kg_s adds the planetary part Omega a^2 cos^2(phi)
  dp/g.
* the extremes (temperature, surface pressure, wind, per-species grid
  minima, reservoir minimum) are the quantities the research-bound
  refusals in dynamics.enforce read and the negatives the next consumer
  will clamp; recorded here every step so a refusal has a lead-in.
"""
from __future__ import annotations

import numpy as np

from ..constants import (
    DRY_AIR_CP,
    GRAVITY_M_S2,
    LATENT_HEAT_FUSION,
    LATENT_HEAT_VAPORIZATION,
    LIQUID_WATER_DENSITY,
    WATER_SPECIES,
)
from ..spill import resident
from ..water import (
    NATIVE_CANOPY_STORE_NAMES,
    NATIVE_SNOW_STORE_NAMES,
    SOIL_LAYER_THICKNESS_M,
    WATER_OUTFLOW_NAMES,
)

FROZEN_SPECIES = ("qi", "qs", "qg")


def reservoir_columns(bundle, xp) -> dict[str, object]:
    """Surface, soil, native-store and outflow columns at exactly
    water.total_water_column's pricing, WITHOUT its driven-negative
    refusals.

    Those refusals read ``float(xp.min(store))`` - a device sync per store
    on the cupy backend - and they already run every step inside the
    global water fixer on the same arrays, so a copy here would be a
    second identical guard bought with the per-step syncs the ledger
    exists to avoid.  The pricing must stay identical to water.py's or
    the ledger's water_total stops being the fixer's total;
    ``test_ledger_row_matches_host_budgets`` compares the two to 1e-12.
    """
    surface = bundle.surface
    # The five planes and one soil stack this ledger reads may live in
    # the pinned host tier; ``resident`` stages exactly those and no
    # more, and they die with the call.  Same values, same pricing.
    soil_fraction = resident(xp, surface.soil_water_fraction)
    land_fraction = resident(xp, surface.land_fraction)
    surface_water = resident(xp, surface.water_kg_m2)
    thickness = xp.asarray(
        SOIL_LAYER_THICKNESS_M, dtype=soil_fraction.dtype
    )[:, None, None]
    soil = xp.sum(
        xp.maximum(soil_fraction, 0.0)
        * thickness
        * LIQUID_WATER_DENSITY
        * land_fraction[None],
        axis=0,
    )
    arrays = getattr(bundle.physics_state, "arrays", {})
    native = xp.zeros_like(surface_water)
    for choices in (NATIVE_CANOPY_STORE_NAMES, NATIVE_SNOW_STORE_NAMES):
        name = next((name for name in choices if name in arrays), None)
        if name is not None:
            native = native + xp.maximum(resident(xp, arrays[name]), 0.0)
    name = next((name for name in WATER_OUTFLOW_NAMES if name in arrays), None)
    outflow = (
        resident(xp, arrays[name]) if name is not None
        else xp.zeros_like(surface_water)
    )
    return {
        "surface": surface_water,
        "soil": soil,
        "native": native,
        "outflow": outflow,
    }

#: (name, definition) in the exact order ``ledger_row`` packs them.
LEDGER_TERMS: tuple[tuple[str, str], ...] = (
    ("mass_pa", "global-mean surface pressure (g * total column mass)"),
    *(
        (f"water_{name}_kg_m2", f"global-mean column integral of {name}")
        for name in WATER_SPECIES
    ),
    ("water_atmosphere_kg_m2", "sum of the six species columns"),
    ("water_surface_kg_m2", "explicit surface reservoir"),
    ("water_soil_kg_m2", "soil water at land-fraction weight"),
    ("water_native_kg_m2", "native canopy + snow stores"),
    ("water_outflow_kg_m2", "cumulative booked runoff exits"),
    ("water_total_kg_m2", "held + booked exits: the fixer/gate total"),
    ("dry_enthalpy_j_m2", "int cp T dp/g"),
    ("kinetic_energy_j_m2", "int 0.5 (u^2+v^2) dp/g"),
    ("potential_energy_j_m2", "int Phi dp/g, full-level geopotential"),
    ("surface_potential_energy_j_m2", "Phi_s ps / g"),
    ("total_dry_energy_j_m2", "dry_enthalpy + kinetic + surface_potential"),
    ("latent_vapor_j_m2", "int Lv qv dp/g"),
    ("latent_frozen_j_m2", "int Lf (qi+qs+qg) dp/g"),
    ("moist_total_energy_j_m2", "total_dry + latent_vapor - latent_frozen"),
    ("relative_angular_momentum_kg_s", "int u a cos(phi) dp/g"),
    ("axial_angular_momentum_kg_s", "relative + int Omega a^2 cos^2(phi) dp/g"),
    ("temperature_min_k", "grid minimum temperature"),
    ("temperature_max_k", "grid maximum temperature"),
    ("surface_pressure_min_pa", "grid minimum surface pressure"),
    ("surface_pressure_max_pa", "grid maximum surface pressure"),
    ("wind_max_m_s", "grid maximum wind speed"),
    ("surface_water_min_kg_m2", "minimum column of the explicit reservoir"),
    *(
        (f"min_{name}_kg_kg", f"grid minimum of {name} before any clamp")
        for name in WATER_SPECIES
    ),
)
LEDGER_NAMES: tuple[str, ...] = tuple(name for name, _ in LEDGER_TERMS)
LEDGER_INDEX: dict[str, int] = {name: i for i, name in enumerate(LEDGER_NAMES)}

#: Terms whose per-hour drift the receipt summarises (the conserved ones).
CONSERVED_TERMS = (
    "mass_pa",
    "water_total_kg_m2",
    "total_dry_energy_j_m2",
    "moist_total_energy_j_m2",
    "axial_angular_momentum_kg_s",
)

GRID_KEYS = ("ps", "dp", "temperature", "u", "v", "geopotential", *WATER_SPECIES)


def global_mean_device(xp, fields, weights):
    """Area-weighted global mean of ``fields[..., nlat, nlon]`` on the device.

    Same quadrature as ``GaussianGrid.global_mean``; accumulates in
    float64 on every backend.
    """
    zonal = xp.mean(fields, axis=-1, dtype=xp.float64)
    return 0.5 * xp.sum(zonal * weights, axis=-1, dtype=xp.float64)


class LedgerGeometry:
    """Device-resident constants the ledger row needs, built once."""

    def __init__(self, transform, rotation_rate_s: float):
        xp = transform.backend.xp
        grid = transform.grid
        self.xp = xp
        self.weights = xp.asarray(grid.quadrature_weights, dtype=xp.float64)
        radius = float(grid.radius_m)
        cos_lat = np.asarray(grid.cos_lat, dtype=np.float64)
        self.a_coslat = xp.asarray(
            (radius * cos_lat)[:, None], dtype=xp.float64
        )
        self.planetary = xp.asarray(
            (float(rotation_rate_s) * radius * radius * cos_lat * cos_lat)[:, None],
            dtype=xp.float64,
        )


def ledger_row(model, bundle, geometry: LedgerGeometry):
    """One device vector of ``len(LEDGER_TERMS)`` float64 entries.

    No host round trip: every value is a 0-d or 1-d device array until the
    ledger flushes the batch.
    """
    xp = geometry.xp
    g = model.grid_state(bundle.atmosphere, only=GRID_KEYS)
    context = (getattr(model, "metal_diffusion_context", None)
               or getattr(model.transform, "metal_context", None))
    if context is not None:
        return _metal_ledger_row(model, bundle, geometry, g, context)
    weights = geometry.weights
    a_coslat = geometry.a_coslat
    planetary = geometry.planetary
    host_diagnostic = any(hasattr(value, "to_numpy") for value in g.values())
    if host_diagnostic:
        # The in-situ ledger is an output diagnostic. Grid state remains
        # resident for the forecast; explicitly read only the fields this
        # ledger records, then compute its small diagnostic row on host.
        def host(value):
            return value.to_numpy() if hasattr(value, "to_numpy") else np.asarray(value)

        g = {name: host(value) for name, value in g.items()}
        weights = host(weights)
        a_coslat = host(a_coslat)
        planetary = host(planetary)
        surface = bundle.surface
        soil_fraction = host(surface.soil_water_fraction)
        land_fraction = host(surface.land_fraction)
        surface_water = host(surface.water_kg_m2)
        soil = np.sum(
            np.maximum(soil_fraction, 0.0)
            * np.asarray(SOIL_LAYER_THICKNESS_M, dtype=np.float64)[:, None, None]
            * LIQUID_WATER_DENSITY * land_fraction[None], axis=0,
        )
        arrays = getattr(bundle.physics_state, "arrays", {})
        native = np.zeros_like(surface_water)
        for choices in (NATIVE_CANOPY_STORE_NAMES, NATIVE_SNOW_STORE_NAMES):
            name = next((name for name in choices if name in arrays), None)
            if name is not None:
                native += np.maximum(host(arrays[name]), 0.0)
        name = next((name for name in WATER_OUTFLOW_NAMES if name in arrays), None)
        stores = {
            "surface": surface_water,
            "soil": soil,
            "native": native,
            "outflow": host(arrays[name]) if name is not None else np.zeros_like(surface_water),
        }
        xp = np
    dp_g = g["dp"] / GRAVITY_M_S2
    # Per species, never a stacked (6, nlev, nlat, nlon) temporary: at T533
    # float64 that stack alone is 2.5 GB.
    species_columns = xp.stack([
        xp.sum(g[name] * dp_g, axis=0, dtype=xp.float64)
        for name in WATER_SPECIES
    ])
    if not host_diagnostic:
        stores = reservoir_columns(bundle, xp)
    temperature = g["temperature"]
    u = g["u"]
    v = g["v"]
    speed2 = u * u + v * v
    dry_enthalpy = xp.sum(temperature * dp_g, axis=0, dtype=xp.float64) * DRY_AIR_CP
    kinetic = xp.sum(0.5 * speed2 * dp_g, axis=0, dtype=xp.float64)
    potential = xp.sum(g["geopotential"] * dp_g, axis=0, dtype=xp.float64)
    surface_geopotential = model.surface_geopotential
    if hasattr(surface_geopotential, "to_numpy"):
        surface_geopotential = surface_geopotential.to_numpy()
    surface_potential = (
        np.asarray(surface_geopotential, dtype=np.float64)
        * g["ps"] / GRAVITY_M_S2
    )
    latent_vapor = xp.sum(g["qv"] * dp_g, axis=0, dtype=xp.float64) * (
        LATENT_HEAT_VAPORIZATION
    )
    frozen = sum(g[name] for name in FROZEN_SPECIES)
    latent_frozen = xp.sum(frozen * dp_g, axis=0, dtype=xp.float64) * (
        LATENT_HEAT_FUSION
    )
    column_mass = xp.sum(dp_g, axis=0, dtype=xp.float64)
    relative_momentum = xp.sum(u * dp_g, axis=0, dtype=xp.float64) * a_coslat
    planetary_momentum = column_mass * planetary
    atmosphere = xp.sum(species_columns, axis=0)
    held = (
        stores["surface"].astype(xp.float64)
        + stores["soil"].astype(xp.float64)
        + stores["native"].astype(xp.float64)
        + stores["outflow"].astype(xp.float64)
    )
    columns = xp.stack([
        g["ps"].astype(xp.float64),
        *species_columns,
        atmosphere,
        stores["surface"].astype(xp.float64),
        stores["soil"].astype(xp.float64),
        stores["native"].astype(xp.float64),
        stores["outflow"].astype(xp.float64),
        atmosphere + held,
        dry_enthalpy,
        kinetic,
        potential,
        surface_potential,
        dry_enthalpy + kinetic + surface_potential,
        latent_vapor,
        latent_frozen,
        dry_enthalpy + kinetic + surface_potential + latent_vapor - latent_frozen,
        relative_momentum,
        relative_momentum + planetary_momentum,
    ])
    means = global_mean_device(xp, columns, weights)
    extremes = xp.stack([
        xp.min(temperature),
        xp.max(temperature),
        xp.min(g["ps"]),
        xp.max(g["ps"]),
        xp.sqrt(xp.max(speed2)),
        # ``stores["surface"]`` is the same plane, already staged from the
        # pinned host tier when the tier holds it, so the reservoir
        # minimum costs no second transfer.
        xp.min(stores["surface"]),
    ]).astype(xp.float64)
    minima = xp.stack([xp.min(g[name]) for name in WATER_SPECIES]).astype(
        xp.float64
    )
    row = xp.concatenate([means, extremes, minima])
    return row


def _metal_ledger_row(model, bundle, geometry, grid, context):
    """Compute the default Global conservation row on its Metal context.

    A column kernel writes the 35 per-column diagnostics. Two bounded
    reductions produce the weighted global means and extrema without
    converting any grid field to a host array.
    """
    from ..spill import resident
    from ...metal.array import MetalArray, MetalArrayModule
    from ...metal.generated import kernel_specs

    metal = MetalArrayModule(context)
    nlat, nlon = model.transform.grid.shape
    nlevel = int(model.nlev)
    ncell = nlat * nlon

    def flat(value, *, name):
        if isinstance(value, MetalArray):
            if value.buffer.context is not context:
                raise ValueError(f"Global in-situ {name} belongs to another Metal context")
            if value.dtype != np.dtype(np.float64):
                raise TypeError(f"Global in-situ {name} must be float64 on Metal")
            source = value
        else:
            host = resident(np, value)
            host = np.ascontiguousarray(np.asarray(host, dtype=np.float64))
            source = metal.asarray(host)
        return source.view((source.size,), dtype=np.float64)

    names = ("dp", "temperature", "u", "v", "geopotential", "ps",
             *WATER_SPECIES)
    fields = {name: flat(grid[name], name=name) for name in names}
    surface = bundle.surface
    surface_water = flat(surface.water_kg_m2, name="surface_water")
    soil_fraction = flat(surface.soil_water_fraction, name="soil_water_fraction")
    land_fraction = flat(surface.land_fraction, name="land_fraction")
    arrays = getattr(bundle.physics_state, "arrays", {})

    def store(choices, label):
        name = next((candidate for candidate in choices if candidate in arrays), None)
        if name is None:
            return metal.zeros((ncell,), dtype=np.float64)
        return flat(arrays[name], name=label)

    native_canopy = store(NATIVE_CANOPY_STORE_NAMES, "native_canopy")
    native_snow = store(NATIVE_SNOW_STORE_NAMES, "native_snow")
    outflow = store(WATER_OUTFLOW_NAMES, "outflow")
    surface_geopotential = flat(
        model.surface_geopotential, name="surface_geopotential")
    weights = flat(geometry.weights, name="quadrature_weights")
    a_coslat = flat(geometry.a_coslat, name="a_coslat")
    planetary = flat(geometry.planetary, name="planetary_momentum")
    soil_thickness = flat(
        np.asarray(SOIL_LAYER_THICKNESS_M, dtype=np.float64),
        name="soil_thickness")

    columns = metal.empty((len(LEDGER_NAMES) * ncell,), dtype=np.float64)
    context.submit_kernel(
        "global_insitu_ledger_columns_fragment",
        {**fields,
         "surface_water": surface_water,
         "soil_fraction": soil_fraction,
         "land_fraction": land_fraction,
         "native_canopy": native_canopy,
         "native_snow": native_snow,
         "outflow": outflow,
         "surface_geopotential": surface_geopotential,
         "a_coslat": a_coslat,
         "planetary": planetary,
         "weights": weights,
         "soil_thickness": soil_thickness,
         "columns": columns},
        {"nlevel": nlevel, "nlat": nlat, "nlon": nlon,
         "nsoil": len(SOIL_LAYER_THICKNESS_M)},
        exact_policy=False, grid_size=ncell,
        threads_per_group=(
            kernel_specs.GLOBAL_INSITU_LEDGER_COLUMNS_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    chunk_size = 256
    nchunk = max(1, (ncell + chunk_size - 1) // chunk_size)
    partial = metal.empty((len(LEDGER_NAMES) * nchunk,), dtype=np.float64)
    context.submit_kernel(
        "global_insitu_ledger_reduce_fragment",
        {"columns": columns, "partial": partial},
        {"ncell": ncell, "nterm": len(LEDGER_NAMES),
         "nchunk": nchunk, "chunk_size": chunk_size},
        exact_policy=False, grid_size=len(LEDGER_NAMES) * nchunk,
        threads_per_group=(
            kernel_specs.GLOBAL_INSITU_LEDGER_REDUCE_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    row = metal.empty((len(LEDGER_NAMES),), dtype=np.float64)
    context.submit_kernel(
        "global_insitu_ledger_finalize_fragment",
        {"partial": partial, "row": row},
        {"nterm": len(LEDGER_NAMES), "nchunk": nchunk},
        exact_policy=False, grid_size=len(LEDGER_NAMES),
        threads_per_group=(
            kernel_specs.GLOBAL_INSITU_LEDGER_FINALIZE_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    return row


__all__ = [
    "CONSERVED_TERMS",
    "LEDGER_INDEX",
    "LEDGER_NAMES",
    "LEDGER_TERMS",
    "LedgerGeometry",
    "global_mean_device",
    "ledger_row",
    "reservoir_columns",
]
