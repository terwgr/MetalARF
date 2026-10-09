"""Skin, screen and surface energy balance of a checkpoint against the GFS.

The afternoon 2 m warm bias over land has four places to live: the skin
(the land surface's energy balance), the surface layer between the skin
and the first full level (the exchange closure and the 2 m diagnostic
drawn from it), the column above (the boundary layer), or the forcing
(net radiation).  A station score of T2 alone cannot tell them apart.
This instrument reads one checkpoint on the run's own Gaussian grid and
scores, against the GFS 0.25 degree product valid at the same instant,
every rung of that ladder at once:

    TSK      the skin (GFS TMP:surface)
    T2       the screen (GFS TMP:2 m), and Td2 (GFS DPT:2 m)
    T80/T100 the column just above the surface layer (GFS TMP:80 m,
             TMP:100 m; the model column is interpolated in height)
    H, LE, G, Rn   sensible, latent, ground heat flux and net radiation
             (GFS SHTFL, LHTFL, -GFLUX (the GFS ground flux is positive upward),
             DSWRF - USWRF + DLWRF - ULWRF, time-averaged records of the
             forecast product handed in, read record by record through the
             GRIB2 bridges because the pdt-8 interval records cannot bind
             as a model input)
    VEGFRA, SMOIS(0-10 cm), TSLB(0-10 cm)   the land state Noah partitions
             with (GFS VEG, SOILW, TSOIL)

and decomposes the 2 m bias into the three differences that add up to it
exactly:

    bias(T2) = bias(T80) + [d(T2 - T80)]
    bias(TSK) = bias(T2) + [d(TSK - T2)]

where d(x) is the model-minus-GFS difference of the area mean of x.  A
bias that sits in bias(T80) is the column's; one that sits in d(T2 - T80)
is the surface layer's or the diagnostic's; d(TSK - T2) is the skin's
excess over the screen.  The surface energy balance of the model's own
books (Rn - H - LE - G, per land cell) is reported beside the GFS's so the
partition is a number and not a story.

What it measures, exactly: area means over the named regions with
cos(latitude) weights on the Gaussian grid, model minus reference; the
reference is bilinearly interpolated from the 0.25 degree grid.  The GFS
state is instantaneous (one analysis or forecast record); the GFS fluxes
are the averaging period of the product's own pdt-8 records (one hour for
an f001, the 6-7 h hour of an f007), so a model instantaneous flux at the
period's start is compared with a one-hour mean.  The model's fluxes are
the physics state's held values at the checkpoint instant (the surface
layer's and Noah's last call; SWDOWN/GLW the last radiation bucket).
Nothing here falls back: a missing physics array, a GFS record the
mapping cannot find, or a region with no land cell refuses by name.  The
one GFS record with a bitmap, the vegetation fraction (off over water),
is decoded under the mapping's preserve_mask policy: its cells are NaN
where the bitmap is off and on every land cell whose bilinear stencil
touched water, and those cells are left out of that field's statistics
with their count reported as n_missing beside n.

The frozen footprints (the columns the cold start seeds and the frozen
column integrates) are scored beside the three land regions: the pack by
hemisphere and by analysed fraction (the GFS product's own sea-ice
fraction at or above one half, split at PARTIAL_PACK_FRACTION), the
Antarctic and Greenland ice sheets (land carrying the model's ice class),
the snow-covered land (GFS water equivalent at or above 10 kg/m2).  On a
frozen column the latent flux is the sublimation the frozen step charges
(L_s QFX) and G is the conduction from the skin node into the node below,
so Rn - H - LE - G there is the skin node's storage.  Every footprint
also reads the 10 m wind and the roughness both sides carry, with the
part of the wind difference the roughness alone accounts for in the
neutral log law from the surface layer's own first-level wind,

    dU10 = U1 [ln(10/z0_model)/ln(z1/z0_model) - ln(10/z0_gfs)/ln(z1/z0_gfs)]

(stability and the boundary layer's response to the drag are not in it;
it is the floor of the roughness effect, stated as such).

Calibration (tests/test_arwen_global_surface_energy.py): identical sides
read zero; a planted +2 K skin offset is read as +2 K in TSK and 0 in
T2/T80 (and the decomposition puts it all in d(TSK - T2)); a planted
column offset lands in bias(T80) with the surface-layer term zero; the
80 m interpolation is exact on a linear profile; the energy-balance
residual of a closed synthetic column is zero and a planted 50 W/m2 hole
is read as 50; the sign convention of the GFS ground heat flux is checked
against the GFS's own closure residual and refused when it does not
close within GFS_CLOSURE_TOLERANCE_W_M2 either way; a plant that is not
uniform (+4 K north of 40N only) reads its cos(latitude)-weighted share
in the bias and the square root of that share in the rmse, computed from
the masks and weights alone, where an unweighted mean would read a
different number; and the same plants on the reference side (skin, 80 m,
sensible flux) read back as their negatives in the same terms.  On a
synthetic frozen column: a skin offset planted on the pack reads in the
pack's skin term and nowhere else; a roughness planted on either side,
with that side's 10 m wind re-diagnosed by the log law, reads in the wind
bias and in the roughness term by the same amount, so the wind at the
reference roughness reads zero.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .physics import frozen_surface
from .state import SURFACE_ARRAY_NAMES
from .statics import SURFACE_STATICS_METADATA_KEY, xland_plane
from .constants import (
    DRY_AIR_GAS_CONSTANT,
    GRAVITY_M_S2,
    KAPPA,
    LATENT_HEAT_VAPORIZATION,
    REFERENCE_PRESSURE_PA,
    STEFAN_BOLTZMANN,
)

SCHEMA = "arwen-global.surface-energy/v1"
#: Mapping authorities (bare ids; resolve_analysis_mapping globs
#: rw-wps-<id>-*.mapping.json): the GDAS/GFS pgrb2 0.25 source plus the
#: screen/80 m/100 m/vegetation state, and the same plus the pdt-8 fluxes.
STATE_MAPPING_ID = "gfs-surface-state"
FLUX_MAPPING_ID = "gfs-surface-flux"
#: Heights (m AGL) of the GFS above-ground temperature records compared.
COLUMN_HEIGHTS_M = (80.0, 100.0)
#: Region masks over which the area means are taken (the three land
#: regions of the skin lane; their numbers are unchanged by the frozen
#: footprints added after them).
REGIONS = ("conus_land", "nh_midlat_land", "global_land")
#: The frozen footprints, from the GFS product's own ice and snow planes
#: and the model's ice class (a footprint with no cell is reported empty,
#: never averaged).
FROZEN_REGIONS = (
    "sea_ice_north", "sea_ice_north_partial", "sea_ice_north_full",
    "sea_ice_south", "sea_ice_south_partial", "sea_ice_south_full",
    "antarctic_land_ice", "antarctic_interior", "greenland_ice",
    "snow_covered_land", "nh_snow_covered_land",
)
#: WRF's non-fractional sea-ice rule (the seeding's SEA_ICE_THRESHOLD),
#: the fraction above which a pack cell is read as full, the snow water
#: real.exe reads as snow cover, the interior's terrain height.
SEA_ICE_THRESHOLD = 0.5
PARTIAL_PACK_FRACTION = 0.85
SNOW_COVER_KG_M2 = 10.0
INTERIOR_TERRAIN_M = 2000.0
#: Height (m) the log-law reading refers the wind to.
WIND_DIAGNOSTIC_HEIGHT_M = 10.0
#: GFS closure residual (W/m2, CONUS land mean) beyond which the flux
#: product is refused: it means the sign convention assumed for its ground
#: heat flux is not the one its own balance closes with.
GFS_CLOSURE_TOLERANCE_W_M2 = 40.0
#: Lowest-level virtual temperature and the top of the interpolation stack.
_COLUMN_LEVELS = 8
#: Bolton (1980) saturation vapour pressure constants for the 2 m dewpoint.
_BOLTON_A, _BOLTON_B, _BOLTON_C = 611.2, 17.67, 243.5
_EPSILON = 0.622

MEASURES = (
    "area means over the named regions with cos(latitude) weights on the run's "
    "Gaussian grid, model minus GFS, the GFS interpolated bilinearly from 0.25 deg; "
    "GFS state instantaneous, GFS fluxes the pdt-8 averaging period of the product; "
    "model fluxes the physics state's held values at the checkpoint instant"
)
MODEL_FIELDS = (
    "tsk", "t2", "td2", "t80", "t100", "t1", "z1", "hfx", "lh", "g", "rnet",
    "swdown", "glw", "albedo", "emiss", "vegfra", "smois1", "tslb1", "ps",
    "land", "residual", "q2",
    # the frozen footprints' rungs
    "u10", "v10", "wspd10", "wspd1", "z0", "seaice", "snow", "frozen", "landice",
)
STATE_FIELDS = {
    "tsk": "skin_temperature",
    "t2": "air_temperature_2m",
    "td2": "dewpoint_2m",
    "q2": "specific_humidity_2m",
    "t80": "air_temperature_80m",
    "t100": "air_temperature_100m",
    "ps": "surface_pressure",
    "hgt": "terrain_height",
    "land": "land_fraction",
    "vegfra": "vegetation_fraction",
    "z0": "surface_roughness",
    "tslb": "soil_temperature",
    "smois": "volumetric_soil_moisture",
    "u10": "eastward_wind_10m",
    "v10": "northward_wind_10m",
    "seaice": "sea_ice_fraction",
    "snow": "snow_water_equivalent",
}
FLUX_FIELDS = {
    "hfx": "sensible_heat_flux",
    "lh": "latent_heat_flux",
    "gflux": "ground_heat_flux",
    "dswrf": "downward_shortwave",
    "uswrf": "upward_shortwave",
    "dlwrf": "downward_longwave",
    "ulwrf": "upward_longwave",
    "land": "land_fraction",
}
#: (name, unit, the difference reported, the model field, the GFS field)
COMPARED = (
    ("tsk_k", "K", "tsk"),
    ("t2_k", "K", "t2"),
    ("td2_k", "K", "td2"),
    ("t80_k", "K", "t80"),
    ("t100_k", "K", "t100"),
    ("hfx_w_m2", "W/m2", "hfx"),
    ("lh_w_m2", "W/m2", "lh"),
    ("g_w_m2", "W/m2", "g"),
    ("rnet_w_m2", "W/m2", "rnet"),
    ("vegfra_1", "1", "vegfra"),
    ("smois1_m3_m3", "m3/m3", "smois1"),
    ("tslb1_k", "K", "tslb1"),
    ("wspd10_m_s", "m/s", "wspd10"),
    ("seaice_1", "1", "seaice"),
)


# --------------------------------------------------------------------------
# the model side
# --------------------------------------------------------------------------


@dataclass
class SurfaceSample:
    """One checkpoint's surface ladder on the Gaussian grid (float64)."""

    step: int
    time_s: float
    lat: np.ndarray          # (nlat, nlon) degrees
    lon: np.ndarray          # (nlat, nlon) degrees, 0..360
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def hydrostatic_full_level_heights(temperature, qv, p_half, p_full):
    """Height (m AGL) of every full level from the surface up, top first.

    ``temperature``/``qv``/``p_full`` are (nlev, ...) with index 0 the top
    and index -1 the lowest layer; ``p_half`` is (nlev + 1, ...) with the
    surface last.  Layer thickness from the virtual temperature of the
    layer; the full level sits at its own pressure inside the layer.
    """
    nlev = temperature.shape[0]
    virtual = temperature * (1.0 + 0.61 * qv)
    z_half = np.zeros_like(p_half)
    for k in range(nlev - 1, -1, -1):
        z_half[k] = z_half[k + 1] + DRY_AIR_GAS_CONSTANT * virtual[k] / GRAVITY_M_S2 * np.log(
            p_half[k + 1] / p_half[k]
        )
    z_full = z_half[1:] + DRY_AIR_GAS_CONSTANT * virtual / GRAVITY_M_S2 * np.log(
        p_half[1:] / p_full
    )
    return z_full


def interpolate_in_height(values, heights, target_m, *, levels: int = _COLUMN_LEVELS):
    """Linear-in-height interpolation of a (nlev, ...) field to one height.

    The lowest ``levels`` full levels are used (index -1 the lowest).  A
    target below the lowest full level extrapolates linearly from the two
    lowest levels; a target above the stack is refused by name, because a
    reading that silently clamped would be the top level dressed as the
    target.
    """
    v = np.asarray(values, dtype=np.float64)[-levels:]
    z = np.asarray(heights, dtype=np.float64)[-levels:]
    # Bottom-up ordering for the search.
    v = v[::-1]
    z = z[::-1]
    if bool(np.any(z[-1] < target_m)):
        raise ValueError(
            f"interpolation target {target_m} m lies above the {levels}-level "
            f"stack (top full level minimum {float(np.min(z[-1])):.1f} m AGL)"
        )
    upper = np.sum(z < target_m, axis=0)          # first level index at or above target
    upper = np.clip(upper, 1, levels - 1)[None]
    lower = upper - 1
    z_lo = np.take_along_axis(z, lower, axis=0)[0]
    z_hi = np.take_along_axis(z, upper, axis=0)[0]
    v_lo = np.take_along_axis(v, lower, axis=0)[0]
    v_hi = np.take_along_axis(v, upper, axis=0)[0]
    weight = (target_m - z_lo) / (z_hi - z_lo)
    return v_lo + weight * (v_hi - v_lo)


def dewpoint_from_specific_humidity(q, p_pa):
    """2 m dewpoint (K) from specific humidity and pressure, Bolton 1980."""
    q = np.clip(np.asarray(q, dtype=np.float64), 1.0e-9, None)
    e = q * p_pa / (_EPSILON + (1.0 - _EPSILON) * q)
    ln = np.log(np.maximum(e, 1.0e-6) / _BOLTON_A)
    return _BOLTON_C * ln / (_BOLTON_B - ln) + 273.15


def net_radiation(swdown, glw, albedo, emissivity, tsk):
    """Rn = SWDOWN (1 - albedo) + emiss GLW - emiss sigma TSK^4, the
    balance Noah closes (noah.cu solnet + lwdn - emissi sigma T1^4)."""
    return swdown * (1.0 - albedo) + emissivity * (glw - STEFAN_BOLTZMANN * tsk ** 4)


def model_surface_sample(reader, checkpoint_path) -> SurfaceSample:
    """The surface ladder of one checkpoint through the run's own grid.

    ``reader`` is a water_budget.CheckpointReader (the receipt's transform
    and vertical tables, numpy float64).  Every field is refused by name
    when the checkpoint does not carry it: this instrument reads the native
    suite's own books and substitutes nothing.
    """
    from .checkpoint import read_checkpoint

    metadata, arrays = read_checkpoint(checkpoint_path)
    transform = reader.transform
    lat1 = np.asarray(transform.grid.latitude_deg, dtype=np.float64)
    lon1 = np.asarray(transform.grid.longitude_deg, dtype=np.float64) % 360.0
    lon2d, lat2d = np.meshgrid(lon1, lat1)

    def surface(name):
        key = f"surface__{SURFACE_ARRAY_NAMES[name]}"
        if key not in arrays:
            raise KeyError(f"{checkpoint_path}: no {key} in the checkpoint")
        return np.asarray(arrays[key], dtype=np.float64)

    def physics(name):
        key = f"physics__{name}"
        if key not in arrays:
            raise KeyError(
                f"{checkpoint_path}: the physics state carries no {name!r}; "
                "this instrument reads the native suite's surface books "
                "(sfclay, Noah, RRTMGP) and substitutes nothing"
            )
        return np.asarray(arrays[key], dtype=np.float64)

    logps = transform.inverse(arrays["atmosphere__log_surface_pressure"].astype(np.complex128))
    ps = np.exp(np.asarray(logps, dtype=np.float64))
    pressure = reader.vertical.pressure(ps, transform.backend)
    p_half = np.asarray(pressure["p_half"], dtype=np.float64)
    p_full = np.asarray(pressure["p_full"], dtype=np.float64)
    theta = reader._grid_field(arrays["atmosphere__theta"])
    qv = reader._grid_field(arrays["atmosphere__qv"])
    temperature = theta * (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
    z_full = hydrostatic_full_level_heights(temperature, qv, p_half, p_full)

    land_fraction = surface("land_fraction")
    land = land_fraction >= 0.5
    tsk = surface("temperature_k")
    albedo = surface("albedo")
    emiss = surface("emissivity")
    hfx = physics("hfx")
    qfx = physics("qfx")
    noah_lh = physics("noah_lh")
    grdflx = physics("noah_grdflx")
    swdown = physics("swdown")
    glw = physics("glw")
    t2 = physics("t2")
    q2 = physics("q2")
    notes = [
        f"lowest full level {float(np.mean(z_full[-1])):.1f} m AGL (area mean)",
        "lh: Noah energy flux on land, XLV*QFX on water, L_s*QFX on frozen columns; "
        "g: -GRDFLX (into the soil) on land, skin-node conduction into the column on frozen columns",
    ]
    # The frozen columns (sea ice and land ice): the seeded ice planes
    # (zero on a pre-seeding checkpoint, which the reader records), the
    # ice class from the statics convention row the checkpoint carries.
    def seeded(name):
        # A checkpoint written before the cold-start seeding carries no ice
        # planes: it ran an ice-free planet, read as zero here exactly as
        # checkpoint.py reads it, and the note says so.
        key = f"surface__{SURFACE_ARRAY_NAMES[name]}"
        if key not in arrays:
            notes.append(f"{key} absent (a pre-seeding checkpoint): read as zero, an ice-free planet")
            return np.zeros(land.shape, dtype=np.float64)
        return surface(name)

    sea_ice = seeded("sea_ice_fraction")
    thickness = seeded("sea_ice_thickness_m")
    landuse = surface("landuse_category")
    row = dict(metadata.get("physics_metadata") or {}).get(SURFACE_STATICS_METADATA_KEY)
    if row is None:
        if np.any(sea_ice >= SEA_ICE_THRESHOLD):
            raise ValueError(
                f"{checkpoint_path}: carries sea ice but no "
                f"physics_metadata[{SURFACE_STATICS_METADATA_KEY!r}] row naming the ice "
                "class; the frozen footprints cannot be identified"
            )
        frozen = np.zeros(land.shape, dtype=bool)
        notes.append("no statics convention row: no frozen column identified")
    else:
        ice_category = int(row["ice_category"])
        xland = xland_plane(land_fraction, sea_ice, np)
        frozen = np.asarray(
            frozen_surface.frozen_columns(xland, sea_ice, landuse, ice_category, np), dtype=bool
        )
    seaice_columns = frozen & (sea_ice >= SEA_ICE_THRESHOLD)
    landice = frozen & ~seaice_columns
    snowh = physics("noah_snowh") if "physics__noah_snowh" in arrays else np.zeros(land.shape)
    snow = physics("noah_snow") if "physics__noah_snow" in arrays else np.zeros(land.shape)
    soil_t = surface("soil_temperature_k")
    # Latent heat: Noah's own energy flux on land; the surface layer's
    # XLV * QFX over water, where nothing else computes one; the
    # sublimation the frozen step charges on a frozen column.
    lh = np.where(land, noah_lh, LATENT_HEAT_VAPORIZATION * qfx)
    lh = np.where(frozen, frozen_surface.LATENT_HEAT_SUBLIMATION_J_KG * qfx, lh)
    # WRF's GRDFLX is the flux INTO the surface from the soil (noah.cu
    # ssoil_out = -ssoil): negative by day.  G here is positive into the
    # soil, so Rn - H - LE - G is Noah's own closure residual.  A frozen
    # column's G is the conduction from its skin node into the node below
    # (the same node thicknesses and media the frozen step used).
    g = np.where(land, -grdflx, 0.0)
    if frozen.any():
        dz = np.where(
            seaice_columns[None],
            frozen_surface.sea_ice_layer_thickness(thickness, snowh, np),
            frozen_surface.land_ice_layer_thickness(tsk, np),
        )
        g_frozen = frozen_surface.skin_conduction_w_m2(soil_t, dz, snowh, np)
        g = np.where(frozen, np.asarray(g_frozen, dtype=np.float64), g)
    rnet = net_radiation(swdown, glw, albedo, emiss, tsk)
    u10 = physics("u10")
    v10 = physics("v10")
    fields = {
        "u10": u10,
        "v10": v10,
        "wspd10": np.hypot(u10, v10),
        # The surface layer's own first-level wind speed (sfclay wspd, its
        # floor included), the U1 its 10 m diagnostic was drawn from.
        "wspd1": physics("wspd"),
        "z0": surface("roughness_m"),
        "seaice": sea_ice,
        "snow": snow,
        "frozen": frozen,
        "landice": landice,
        "tsk": tsk,
        "t2": t2,
        "q2": q2,
        "td2": dewpoint_from_specific_humidity(q2, ps),
        "t1": temperature[-1],
        "z1": z_full[-1],
        "hfx": hfx,
        "lh": lh,
        "g": g,
        "rnet": rnet,
        "residual": rnet - hfx - lh - g,
        "swdown": swdown,
        "glw": glw,
        "albedo": albedo,
        "emiss": emiss,
        "vegfra": surface("vegetation_fraction"),
        "smois1": surface("soil_water_fraction")[0],
        "tslb1": surface("soil_temperature_k")[0],
        "ps": ps,
        "land": land_fraction,
    }
    for height in COLUMN_HEIGHTS_M:
        fields[f"t{int(height)}"] = interpolate_in_height(temperature, z_full, height)
    notes.append(
        f"frozen columns {int(frozen.sum())} (sea ice {int(seaice_columns.sum())}, "
        f"land ice {int(landice.sum())})"
    )
    return SurfaceSample(
        step=int(metadata["step"]), time_s=float(metadata["time_s"]),
        lat=lat2d, lon=lon2d, fields=fields, notes=notes,
    )


# --------------------------------------------------------------------------
# the GFS side
# --------------------------------------------------------------------------


def _mapping(mapping_id: str) -> Path:
    from .analysis_initial import resolve_analysis_mapping

    return resolve_analysis_mapping(mapping_id)


def _decode(mapping_id: str, path) -> dict:
    from .mapped_source_compat import decode_through_engine

    decoded = decode_through_engine(_mapping(mapping_id), [str(path)])
    frames = decoded.frames
    if len(frames) != 1:
        raise ValueError(f"{path}: expected one frame, decoded {len(frames)}")
    frame = frames[0]
    lat = np.asarray(frame.latitude, dtype=np.float64)
    lon = np.asarray(frame.longitude, dtype=np.float64) % 360.0
    # The decode receipt travels with the product.  The state mapping
    # declares two masked surface records (snow water equivalent and
    # vegetation fraction), so on a published engine this decode really
    # does run the validator adaptation, and the scorecard that reads it
    # says so instead of leaving a reader to infer it from a version.
    return {"lat": lat, "lon": lon, "valid_time": frame.valid_time.isoformat(),
            "frame": frame, "decode": decoded.receipt}


def _plane(frame, name):
    values = np.asarray(frame.fields[name].values, dtype=np.float64)
    return values


def decode_gfs_state(path) -> dict:
    """Skin, screen, 80/100 m, soil and vegetation of one GFS product."""
    out = _decode(STATE_MAPPING_ID, path)
    frame = out.pop("frame")
    fields = {}
    for short, name in STATE_FIELDS.items():
        values = _plane(frame, name)
        if name in ("soil_temperature", "volumetric_soil_moisture"):
            fields[short + "1"] = values[0]
        else:
            fields[short] = values
    fields["snow"] = snow_with_bitmap_policy(fields["snow"], fields["land"], fields["seaice"], str(path))
    out["fields"] = fields
    out["path"] = str(path)
    return out


def snow_with_bitmap_policy(snow, land_fraction, sea_ice_fraction, source: str) -> np.ndarray:
    """The GFS water-equivalent record carries a bitmap that is off over
    open water: there a masked point is no snow (the cold-start seeding's
    rule).  A masked point on land or on the pack is a missing value and
    is refused by name, never zero-filled."""
    snow = np.asarray(snow, dtype=np.float64)
    missing = ~np.isfinite(snow)
    if not missing.any():
        return snow
    covered = (np.asarray(land_fraction) >= 0.5) | (np.asarray(sea_ice_fraction) >= SEA_ICE_THRESHOLD)
    bad = missing & covered
    if bad.any():
        raise ValueError(
            f"{source}: the snow water equivalent is masked on {int(bad.sum())} land "
            "or sea-ice points; only open water may carry the bitmap's absence as no snow"
        )
    return np.where(missing, 0.0, snow)


#: Selector keys a flux record must match exactly (the GFS local-table
#: records also pin the producer identity octets).
_RECORD_KEYS = (
    "discipline", "category", "parameter", "level_type", "pdt",
    "center", "subcenter", "master_table_version", "local_table_version",
)


def record_matches_selector(selector: dict, row: dict) -> bool:
    """Does one GRIB2 inventory row satisfy one mapping selector?

    Integer keys compare exactly; ``level_value`` as a float; a key the
    selector does not name is unconstrained.  ``pdt`` is the product
    definition template, so a selector that names 8 picks the interval
    (time-averaged) record and never the instantaneous twin."""
    for key in _RECORD_KEYS:
        if key in selector and int(row[key]) != int(selector[key]):
            return False
    if "level_value" in selector and float(row["level_value"]) != float(selector["level_value"]):
        return False
    return True


def decode_gfs_flux(path) -> dict:
    """The pdt-8 surface energy fluxes of one GFS forecast product, with
    Rn assembled from its four radiation records and G taken positive
    into the soil (checked by check_gfs_closure).

    The flux records are product-definition-template 8 (a time interval:
    the hour after the reference for an f001), which the rw-wps.mapping.v1
    grammar cannot bind as a model input, so the mapping authority serves
    here as the selector table only and each record is read through the
    GRIB2 inventory and dump bridges (the same decoders the mapped engine
    runs).  The plane is the record's own averaging period; the valid
    time stated is the interval start the record embeds.  A selector that
    matches no record, or more than one, refuses by name.
    """
    from woof.mapped_source import (
        _build_grib2_tools, _grib2_inventory, _grib2_records,
    )

    # The mapping document comes through this package's door, not the
    # engine's function directly: that door is where the published
    # validator's soil-only preserve_mask narrowing is adapted, and four of
    # the six carried mappings declare a masked SURFACE record.  Reading
    # this one document around the door made the flux ladder the single
    # site that would refuse the day a re-cut gave the flux mapping such a
    # record, with the decode receipt reporting nothing about it.
    from .mapped_source_compat import (
        load_mapping, surface_preserve_mask_fields,
    )

    source = Path(path)
    mapping_path = _mapping(FLUX_MAPPING_ID)
    mapping = load_mapping(mapping_path)
    inventory_executable, dump_executable = _build_grib2_tools()
    rows = _grib2_inventory(source, inventory_executable)
    wanted: dict[str, int] = {}
    for short, name in FLUX_FIELDS.items():
        selectors = mapping["fields"][name]["selectors"]
        hits = [row for row in rows if any(record_matches_selector(s, row) for s in selectors)]
        if len(hits) != 1:
            raise ValueError(
                f"{source.name}: {name} matched {len(hits)} GRIB2 records "
                f"(selectors {selectors}); the flux ladder needs exactly one"
            )
        wanted[short] = int(hits[0]["index"])
    records = {
        record.index: record
        for record in _grib2_records(
            source, inventory_executable, dump_executable, set(wanted.values())
        )
    }
    fields = {}
    lat = lon = None
    valid = None
    for short, index in wanted.items():
        record = records[index]
        scale = float(mapping["fields"][FLUX_FIELDS[short]]["units"].get("scale", 1.0))
        fields[short] = np.asarray(record.values, dtype=np.float64) * scale
        if lat is None:
            lat = np.asarray(record.latitude, dtype=np.float64)
            lon = np.asarray(record.longitude, dtype=np.float64) % 360.0
            valid = record.valid_time
        elif (record.values.shape != fields[next(iter(fields))].shape
              or not np.array_equal(np.asarray(record.latitude), lat)):
            raise ValueError(f"{source.name}: {short} is not on the grid of the other flux records")
    fields["rnet"] = fields["dswrf"] - fields["uswrf"] + fields["dlwrf"] - fields["ulwrf"]
    # GFS GFLUX is positive UPWARD (from the soil to the surface), the
    # convention of its SHTFL/LHTFL: negative by day.  G here is positive
    # into the soil, the sign Rn - H - LE - G closes with; measured on the
    # 2026-09-01 18Z f001 product the CONUS land residual reads +1.2 W/m2
    # this way and +153.8 W/m2 the other, and check_gfs_closure refuses
    # the product if that ever inverts.
    fields["g"] = -fields["gflux"]
    fields["residual"] = fields["rnet"] - fields["hfx"] - fields["lh"] - fields["g"]
    return {
        "lat": lat, "lon": lon, "valid_time": valid.isoformat(), "fields": fields,
        "path": str(source),
        "records": {short: index for short, index in wanted.items()},
        "time_semantics": "pdt 8: each plane is the averaging period of its record; valid_time is the interval start",
        # This route has no decode receipt because it runs no decode: the
        # mapping serves as the selector table and the records come through
        # the GRIB2 bridges.  What it CAN say is how the document was read
        # and whether it declares a record the published engine's validator
        # would have narrowed, which is the one adaptation that reaches it.
        "mapping": {
            "path": str(mapping_path),
            "read_through": "woof.globe.mapped_source_compat.load_mapping",
            "masked_surface_fields": list(
                surface_preserve_mask_fields(mapping_path)),
            "records_read_by": "the engine's GRIB2 inventory and dump "
                               "bridges, not decode_mapped_source",
        },
    }


def interpolate_to_grid(values, src_lat, src_lon, lat2d, lon2d):
    """Bilinear interpolation from a regular lat-lon grid (any axis order)
    onto 2-D target coordinates, the wrap column appended."""
    from .engine_compat import interpolate_to_tape

    return interpolate_to_tape(values, src_lat, src_lon, lat2d, lon2d)


def regrid_reference(reference: dict, lat2d, lon2d, names) -> dict:
    return {
        name: interpolate_to_grid(reference["fields"][name], reference["lat"], reference["lon"], lat2d, lon2d)
        for name in names
        if name in reference["fields"]
    }


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def region_masks(lat, lon, land_model, land_reference, *, gfs_ice=None, gfs_snow=None,
                 model_landice=None, gfs_terrain=None) -> dict[str, np.ndarray]:
    """The three land regions, and (when the reference's ice and snow
    planes and the model's ice class are handed in) the frozen footprints:
    the pack from the reference's own fraction so both arms of a
    comparison score the same cells, the ice sheets from the model's ice
    class on land both sides call land, the snow-covered land from the
    reference's water equivalent."""
    land = land_model & land_reference
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    masks = {
        "conus_land": land & (lat >= 25) & (lat <= 50) & (lon180 >= -125) & (lon180 <= -65),
        "nh_midlat_land": land & (lat >= 30) & (lat <= 60),
        "global_land": land,
    }
    if gfs_ice is None:
        return masks
    ice = gfs_ice >= SEA_ICE_THRESHOLD
    partial = ice & (gfs_ice < PARTIAL_PACK_FRACTION)
    full = gfs_ice >= PARTIAL_PACK_FRACTION
    north, south = lat > 0.0, lat < 0.0
    landice = land & np.asarray(model_landice, dtype=bool)
    snow_land = land & (gfs_snow >= SNOW_COVER_KG_M2)
    antarctic = landice & (lat <= -60.0)
    masks.update({
        "sea_ice_north": ice & north,
        "sea_ice_north_partial": partial & north,
        "sea_ice_north_full": full & north,
        "sea_ice_south": ice & south,
        "sea_ice_south_partial": partial & south,
        "sea_ice_south_full": full & south,
        "antarctic_land_ice": antarctic,
        "antarctic_interior": antarctic & (gfs_terrain > INTERIOR_TERRAIN_M) & (lat < -70.0),
        "greenland_ice": landice & (lat >= 60.0) & (lon180 >= -75.0) & (lon180 <= -10.0),
        "snow_covered_land": snow_land,
        "nh_snow_covered_land": snow_land & north,
    })
    return masks


def log_law_wind(wspd1, z1, z0, height_m: float = WIND_DIAGNOSTIC_HEIGHT_M):
    """Neutral log-law wind at ``height_m`` from the first-level wind
    speed at ``z1`` over roughness ``z0``: U1 ln(h/z0) / ln(z1/z0)."""
    z0 = np.asarray(z0, dtype=np.float64)
    return (np.asarray(wspd1, dtype=np.float64) * np.log(height_m / z0)
            / np.log(np.asarray(z1, dtype=np.float64) / z0))


def roughness_reading(m: dict, ref: dict, mask, weights) -> dict:
    """The roughness both sides carry over ``mask`` and the part of the
    10 m wind difference it alone accounts for in the neutral log law from
    the model's own first-level wind (cells whose roughness is not inside
    (0, z1) on both sides are left out and counted)."""
    z0m = np.asarray(m["z0"], dtype=np.float64)
    z0g = np.asarray(ref["z0"], dtype=np.float64)
    z1 = np.asarray(m["z1"], dtype=np.float64)
    usable = mask & (z0m > 0.0) & (z0g > 0.0) & (z0m < z1) & (z0g < z1) & np.isfinite(z0g)
    if not usable.any():
        return {"n": 0, "n_left_out": int(mask.sum())}
    effect = log_law_wind(m["wspd1"], z1, z0m) - log_law_wind(m["wspd1"], z1, z0g)
    bias = weighted(m["wspd10"] - ref["wspd10"], usable, weights)
    effect_mean = weighted(effect, usable, weights)
    return {
        "n": int(usable.sum()),
        "n_left_out": int(mask.sum() - usable.sum()),
        "model_z0_m": weighted(z0m, usable, weights),
        "gfs_z0_m": weighted(z0g, usable, weights),
        "ln_z0_model_over_gfs": weighted(np.log(z0m / z0g), usable, weights),
        "model_wspd1_m_s": weighted(m["wspd1"], usable, weights),
        "model_wspd10_m_s": weighted(m["wspd10"], usable, weights),
        "gfs_wspd10_m_s": weighted(ref["wspd10"], usable, weights),
        "wspd10_bias_m_s": bias,
        "log_law_wind_effect_m_s": effect_mean,
        "wspd10_bias_at_gfs_roughness_m_s": bias - effect_mean,
        "measures": (
            "neutral log law from the surface layer's first-level wind speed; stability and "
            "the boundary layer's response to the drag are not in it (a floor of the roughness effect)"
        ),
    }


def weighted(values, mask, weights):
    """Area mean of ``values`` over the finite cells of ``mask``.

    A reference record decoded under the preserve_mask policy (the GFS
    vegetation fraction, whose bitmap is off over water) is NaN where it
    was missing and NaN on every land cell whose bilinear stencil touched
    one; those cells carry no value and are left out of the mean rather
    than poisoning it.  The count of cells left out is reported beside
    every statistic by weighted_stats.
    """
    finite = mask & np.isfinite(values)
    if not finite.any():
        raise ValueError("region has no finite cell; a mean over nothing is not a reading")
    w = weights[finite]
    v = values[finite]
    total = float(np.sum(w))
    mean = float(np.sum(v * w) / total)
    return mean


def weighted_stats(diff, mask, weights) -> dict:
    if not mask.any():
        raise ValueError("region has no cell; a mean over nothing is not a reading")
    finite = mask & np.isfinite(diff)
    missing = int(mask.sum() - finite.sum())
    if not finite.any():
        raise ValueError(
            f"region has no finite cell ({missing} masked as missing); a mean "
            "over nothing is not a reading"
        )
    w = weights[finite]
    d = diff[finite]
    total = float(np.sum(w))
    bias = float(np.sum(d * w) / total)
    rmse = float(math.sqrt(np.sum(d * d * w) / total))
    mae = float(np.sum(np.abs(d) * w) / total)
    return {"bias": bias, "rmse": rmse, "mae": mae, "n": int(finite.sum()),
            "n_missing": missing}


def check_gfs_closure(flux: dict, masks: dict, weights) -> dict:
    """The GFS product's own Rn - H - LE - G over the regions, with G read
    positive into the soil.  Refuses when the CONUS land residual exceeds
    the tolerance with G in either sign: then the product's records are
    not what the mapping says they are."""
    out = {}
    for region, mask in masks.items():
        f = flux["fields"]
        with_g_down = weighted(f["rnet"] - f["hfx"] - f["lh"] - f["g"], mask, weights)
        with_g_up = weighted(f["rnet"] - f["hfx"] - f["lh"] + f["g"], mask, weights)
        out[region] = {"residual_g_into_soil": with_g_down, "residual_g_out_of_soil": with_g_up}
    conus = out["conus_land"]
    if abs(conus["residual_g_into_soil"]) > GFS_CLOSURE_TOLERANCE_W_M2:
        raise ValueError(
            "the GFS flux product does not close with its ground heat flux "
            f"read positive into the soil (CONUS land residual "
            f"{conus['residual_g_into_soil']:+.1f} W/m2, the other sign "
            f"{conus['residual_g_out_of_soil']:+.1f}); the records the mapping "
            "selected are not the balance this instrument assumes"
        )
    return out


def localize(model: SurfaceSample, state: dict, flux: dict | None) -> dict:
    """Per region: bias/rmse of every rung, the decomposition of the 2 m
    bias, the surface energy partition of both sides."""
    lat, lon = model.lat, model.lon
    weights = np.cos(np.deg2rad(lat))
    names = list(STATE_FIELDS) + ["tslb1", "smois1"]
    ref = regrid_reference(state, lat, lon, names)
    ref_flux = regrid_reference(flux, lat, lon, list(FLUX_FIELDS) + ["rnet", "g", "residual"]) if flux else {}
    land_model = model.fields["land"] >= 0.5
    land_ref = ref["land"] >= 0.5
    frozen_rungs = all(name in ref for name in ("seaice", "snow", "u10", "v10", "hgt")) and all(
        name in model.fields for name in ("wspd10", "wspd1", "z0", "seaice", "landice")
    )
    if frozen_rungs:
        ref["wspd10"] = np.hypot(ref["u10"], ref["v10"])
        masks = region_masks(
            lat, lon, land_model, land_ref, gfs_ice=ref["seaice"], gfs_snow=ref["snow"],
            model_landice=model.fields["landice"], gfs_terrain=ref["hgt"],
        )
    else:
        masks = region_masks(lat, lon, land_model, land_ref)
    closure = None
    if flux is not None:
        flux_masks = region_masks(lat, lon, land_model, ref_flux["land"] >= 0.5)
        closure = check_gfs_closure({"fields": ref_flux}, flux_masks, weights)
    # The closure residuals are computed here from the four terms, never
    # read from a stored plane, so a sample whose fluxes were edited (a
    # calibration plant, a substituted reference) is scored on what it holds.
    m = dict(model.fields)
    m["residual"] = m["rnet"] - m["hfx"] - m["lh"] - m["g"]
    if ref_flux:
        ref_flux["residual"] = ref_flux["rnet"] - ref_flux["hfx"] - ref_flux["lh"] - ref_flux["g"]
    regions = {}
    for region, mask in masks.items():
        entry = {"n": int(mask.sum())}
        if entry["n"] == 0:
            if region in REGIONS:
                raise ValueError(f"region {region} has no cell; a mean over nothing is not a reading")
            entry["empty"] = True   # a frozen footprint the case does not carry: reported, not averaged
            regions[region] = entry
            continue
        for key, unit, name in COMPARED:
            if name in ref:
                theirs = ref[name]
            elif name in ref_flux:
                theirs = ref_flux[name]
            else:
                continue
            if name not in m:
                continue
            diff = m[name] - theirs
            if region not in REGIONS and not np.any(mask & np.isfinite(diff)):
                # A reference record whose bitmap is off over the whole
                # footprint (the vegetation fraction and the soil records
                # over the pack): reported missing, never averaged and
                # never a refusal of the footprint's other rungs.
                entry[key] = {"bias": None, "rmse": None, "mae": None, "n": 0,
                              "n_missing": int(mask.sum()), "unit": unit, "missing": True}
                continue
            entry[key] = {**weighted_stats(diff, mask, weights), "unit": unit}
        model_means = ["tsk", "t2", "td2", "t80", "t100", "t1", "z1", "hfx", "lh", "g",
                       "rnet", "residual", "swdown", "glw", "albedo", "vegfra", "smois1", "tslb1"]
        gfs_means = ["tsk", "t2", "td2", "t80", "t100", "vegfra", "smois1", "tslb1", "z0"]
        if frozen_rungs:
            model_means += ["emiss", "wspd10", "wspd1", "z0", "seaice", "snow"]
            gfs_means += ["wspd10", "seaice", "snow", "hgt"]

        def mean_or_missing(values):
            if region not in REGIONS and not np.any(mask & np.isfinite(values)):
                return None
            return weighted(values, mask, weights)

        means = {
            "model": {name: mean_or_missing(m[name]) for name in model_means},
            "gfs": {name: mean_or_missing(ref[name]) for name in gfs_means},
        }
        if frozen_rungs:
            means["model"]["frozen_share"] = weighted(model.fields["frozen"].astype(np.float64), mask, weights)
            entry["roughness"] = roughness_reading(m, ref, mask, weights)
        if ref_flux:
            means["gfs"].update({
                name: weighted(ref_flux[name], mask, weights)
                for name in ("hfx", "lh", "g", "rnet", "residual", "dswrf", "dlwrf", "uswrf", "ulwrf")
            })
            dswrf = ref_flux["dswrf"]
            lit = mask & (dswrf > 50.0)
            if lit.any():
                means["gfs"]["albedo_effective"] = weighted(
                    ref_flux["uswrf"] / np.maximum(dswrf, 50.0), lit, weights)
        entry["means"] = means
        # The decomposition: exact by construction on the area means.
        d = lambda a, b: means["model"][a] - means["gfs"][b]  # noqa: E731
        t2_bias = d("t2", "t2")
        t80_bias = d("t80", "t80")
        tsk_bias = d("tsk", "tsk")
        entry["decomposition_k"] = {
            "t2_bias": t2_bias,
            "column_t80_bias": t80_bias,
            "surface_layer_d_t2_minus_t80": t2_bias - t80_bias,
            "skin_d_tsk_minus_t2": tsk_bias - t2_bias,
            "tsk_bias": tsk_bias,
            "model_t2_minus_t80": means["model"]["t2"] - means["model"]["t80"],
            "gfs_t2_minus_t80": means["gfs"]["t2"] - means["gfs"]["t80"],
            "model_tsk_minus_t2": means["model"]["tsk"] - means["model"]["t2"],
            "gfs_tsk_minus_t2": means["gfs"]["tsk"] - means["gfs"]["t2"],
            "model_tsk_minus_t1": means["model"]["tsk"] - means["model"]["t1"],
        }
        h, le = means["model"]["hfx"], means["model"]["lh"]
        entry["partition"] = {
            "model": {
                "bowen_ratio": h / le if abs(le) > 1.0 else None,
                "evaporative_fraction": le / (h + le) if abs(h + le) > 1.0 else None,
                "closure_residual_w_m2": means["model"]["residual"],
            },
        }
        if ref_flux:
            hg, leg = means["gfs"]["hfx"], means["gfs"]["lh"]
            entry["partition"]["gfs"] = {
                "bowen_ratio": hg / leg if abs(leg) > 1.0 else None,
                "evaporative_fraction": leg / (hg + leg) if abs(hg + leg) > 1.0 else None,
                "closure_residual_w_m2": means["gfs"]["residual"],
            }
        regions[region] = entry
    return {"regions": regions, "gfs_closure": closure}


def verdict(regions: dict, region: str = "conus_land") -> dict:
    """Which rung carries the 2 m bias, by the share of each term."""
    d = regions[region]["decomposition_k"]
    t2 = d["t2_bias"]
    terms = {
        "column": d["column_t80_bias"],
        "surface_layer": d["surface_layer_d_t2_minus_t80"],
    }
    shares = {k: (v / t2 if abs(t2) > 1.0e-9 else None) for k, v in terms.items()}
    largest = max(terms, key=lambda k: abs(terms[k]))
    return {
        "region": region,
        "t2_bias_k": t2,
        "terms_k": terms,
        "shares_of_t2_bias": shares,
        "largest_term": largest,
        "skin_excess_over_screen_k": d["skin_d_tsk_minus_t2"],
        "tsk_bias_k": d["tsk_bias"],
    }


# --------------------------------------------------------------------------
# tapes for the renderer
# --------------------------------------------------------------------------


BIAS_PLANES = (
    ("TSK_MODEL", "tsk", None), ("TSK_GFS", None, "tsk"), ("TSK_BIAS", "tsk", "tsk"),
    ("T2_MODEL", "t2", None), ("T2_GFS", None, "t2"), ("T2_BIAS", "t2", "t2"),
    ("T80_BIAS", "t80", "t80"),
    ("HFX_MODEL", "hfx", None), ("HFX_GFS", None, "hfx"), ("HFX_BIAS", "hfx", "hfx"),
    ("LH_MODEL", "lh", None), ("LH_GFS", None, "lh"), ("LH_BIAS", "lh", "lh"),
    ("RNET_MODEL", "rnet", None), ("RNET_GFS", None, "rnet"), ("RNET_BIAS", "rnet", "rnet"),
)


def bias_planes(model: SurfaceSample, state: dict, flux: dict | None) -> dict[str, np.ndarray]:
    """Model, reference and difference planes on the Gaussian grid, named
    for a render tape (``var:<name>`` products of the Rust renderer)."""
    lat, lon = model.lat, model.lon
    ref = regrid_reference(state, lat, lon, ["tsk", "t2", "t80"])
    if flux is not None:
        ref.update(regrid_reference(flux, lat, lon, ["hfx", "lh", "rnet"]))
    land = model.fields["land"] >= 0.5
    planes = {}
    for name, ours, theirs in BIAS_PLANES:
        if theirs is not None and theirs not in ref:
            continue
        if ours is not None and theirs is not None:
            value = model.fields[ours] - ref[theirs]
        elif ours is not None:
            value = model.fields[ours]
        else:
            value = ref[theirs]
        planes[name] = np.where(land, value, np.nan) if name.endswith("_BIAS") else value
    return planes


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m woof.globe.surface_energy",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="the run directory (its receipt supplies the grid)")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--gfs-state", required=True, type=Path,
                        help="GFS/GDAS pgrb2 0.25 product valid at the checkpoint instant")
    parser.add_argument("--gfs-flux", type=Path, default=None,
                        help="GFS pgrb2 0.25 forecast product whose pdt-8 fluxes average the hour after the instant")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--planes-npz", type=Path, default=None,
                        help="write the bias planes on the Gaussian grid for the export door")
    args = parser.parse_args(argv)

    from .water_budget import CheckpointReader, read_receipt

    t0 = time.time()
    receipt = read_receipt(args.run_dir)
    reader = CheckpointReader(receipt, transport=False)
    model = model_surface_sample(reader, args.checkpoint)
    print(f"model sample: step {model.step} t={model.time_s / 3600:.1f} h ({time.time() - t0:.0f} s)", flush=True)
    state = decode_gfs_state(args.gfs_state)
    print(f"gfs state {args.gfs_state.name} valid {state['valid_time']} ({time.time() - t0:.0f} s)", flush=True)
    flux = decode_gfs_flux(args.gfs_flux) if args.gfs_flux else None
    if flux:
        print(f"gfs flux {args.gfs_flux.name} valid {flux['valid_time']} ({time.time() - t0:.0f} s)", flush=True)
    result = localize(model, state, flux)
    result.update({
        "schema": SCHEMA,
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "step": model.step,
        "time_s": model.time_s,
        "gfs_state": {"path": state["path"], "valid_time": state["valid_time"],
                      "decode": state.get("decode")},
        "gfs_flux": ({"path": flux["path"], "valid_time": flux["valid_time"],
                      "mapping": flux.get("mapping")} if flux else None),
        "measures": __doc__.split("\n\n")[5].strip(),
        "notes": model.notes,
        "verdict": verdict(result["regions"]),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1, default=float)
    if args.planes_npz:
        planes = bias_planes(model, state, flux)
        np.savez_compressed(args.planes_npz, **{k: v.astype(np.float32) for k, v in planes.items()})
    print(report(result))
    return 0


def report(result: dict) -> str:
    lines = [f"== {result['label']} surface ladder vs GFS, step {result['step']} t={result['time_s'] / 3600:.1f} h =="]
    for region, entry in result["regions"].items():
        if entry.get("empty"):
            lines.append(f"[{region}] EMPTY footprint (n=0): INCOMPLETE")
            continue
        lines.append(f"[{region}] n={entry['n']}")
        for key, unit, _name in COMPARED:
            if key in entry:
                s = entry[key]
                if s.get("missing"):
                    lines.append(f"  {key:14s} MISSING on every cell of the footprint ({s['n_missing']} masked)")
                    continue
                lines.append(f"  {key:14s} bias {s['bias']:+8.3f} rmse {s['rmse']:7.3f} {unit}")
        d = entry["decomposition_k"]
        lines.append(
            f"  T2 bias {d['t2_bias']:+.3f} = column(T80) {d['column_t80_bias']:+.3f} "
            f"+ surface layer d(T2-T80) {d['surface_layer_d_t2_minus_t80']:+.3f}; "
            f"skin d(TSK-T2) {d['skin_d_tsk_minus_t2']:+.3f} (TSK bias {d['tsk_bias']:+.3f})"
        )
        lines.append(
            f"  model TSK-T2 {d['model_tsk_minus_t2']:+.2f} (GFS {d['gfs_tsk_minus_t2']:+.2f}); "
            f"model T2-T80 {d['model_t2_minus_t80']:+.2f} (GFS {d['gfs_t2_minus_t80']:+.2f}); "
            f"model TSK-T1 {d['model_tsk_minus_t1']:+.2f}"
        )
        nan = float("nan")
        m = {k: (nan if v is None else v) for k, v in entry["means"]["model"].items()}
        lines.append(
            f"  model Rn {m['rnet']:6.1f} H {m['hfx']:6.1f} LE {m['lh']:6.1f} G {m['g']:6.1f} "
            f"resid {m['residual']:+6.1f} W/m2; SWDOWN {m['swdown']:.0f} GLW {m['glw']:.0f} alb {m['albedo']:.3f} "
            f"veg {m['vegfra']:.2f} smois1 {m['smois1']:.3f} tslb1 {m['tslb1']:.1f}"
        )
        g = {k: (nan if v is None else v) for k, v in entry["means"]["gfs"].items()}
        if "hfx" in g:
            lines.append(
                f"  GFS   Rn {g['rnet']:6.1f} H {g['hfx']:6.1f} LE {g['lh']:6.1f} G {g['g']:6.1f} "
                f"resid {g['residual']:+6.1f} W/m2; DSWRF {g['dswrf']:.0f} DLWRF {g['dlwrf']:.0f} "
                f"alb {g.get('albedo_effective', nan):.3f} veg {g['vegfra']:.2f} smois1 {g['smois1']:.3f} tslb1 {g['tslb1']:.1f}"
            )
        p = entry["partition"]
        lines.append(f"  partition model {p['model']}" + (f"; gfs {p['gfs']}" if "gfs" in p else ""))
        r = entry.get("roughness")
        if r and r.get("n"):
            lines.append(
                f"  wind10 model {r['model_wspd10_m_s']:.2f} GFS {r['gfs_wspd10_m_s']:.2f} bias {r['wspd10_bias_m_s']:+.2f} m/s; "
                f"z0 model {r['model_z0_m']:.4f} GFS {r['gfs_z0_m']:.4f} m (ln ratio {r['ln_z0_model_over_gfs']:+.2f}); "
                f"log-law share {r['log_law_wind_effect_m_s']:+.2f}, wind at GFS z0 {r['wspd10_bias_at_gfs_roughness_m_s']:+.2f} m/s"
            )
            if "emiss" in m:
                lines.append(
                    f"  model emiss {m['emiss']:.3f} frozen share {m.get('frozen_share', float('nan')):.2f} "
                    f"seaice {m.get('seaice', float('nan')):.2f} (GFS {g.get('seaice', float('nan')):.2f}) "
                    f"snow {m.get('snow', float('nan')):.0f} (GFS {g.get('snow', float('nan')):.0f}) kg/m2"
                )
    v = result["verdict"]
    lines.append(f"verdict ({v['region']}): T2 bias {v['t2_bias_k']:+.3f} K, largest term {v['largest_term']} "
                 f"{v['terms_k']}, skin excess over screen {v['skin_excess_over_screen_k']:+.3f} K")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
