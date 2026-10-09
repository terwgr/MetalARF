"""Cold-start seeding of sea ice and snow from the analysis.

A global model started in September with every column ice-free and
snow-free is wrong at both poles: the Antarctic pack is at its yearly
maximum and the Greenland and Antarctic sheets are snow-covered all
year.  The GDAS pgrb2 analysis the initializer already decodes carries
the surface it needs: sea-ice fraction (ICEC, GRIB2 10/2/0) and
thickness (ICETK, 10/2/1), snow water equivalent (WEASD, 0/1/13, kg m-2)
and snow depth (SNOD, 0/1/11, m), beside the skin temperature (which is
the SST on open water), the four soil temperature and moisture layers
and the land mask the initializer took before.  This module turns those
four planes into the surface state's sea-ice fraction and thickness,
Noah's snow store, depth and cover flag, and a provenance document the
receipt carries, and it refuses instead of guessing:

- an analysis without one of the four fields is refused by name; nothing
  is zero-filled for a field that was never read;
- the snow unit trap (the audit's finding: a legacy name that means
  metres of water on one route and kg m-2 on another) is closed by
  VALUE, not by the declared unit: the bulk density the two snow planes
  imply, SWE / depth, must read as kg m-3 of snow (tens to hundreds);
  a field of metres reads as a fraction of one and is refused;
- the snow planes' source bitmap (unset over open water on GDAS) is read
  as no snow on water and refused on any land or sea-ice point.

Interpolation onto the Gaussian grid follows WPS METGRID.TBL's treatment
of SNOW and SNOWH (masked by the land mask, zero on water): the snow
planes are averaged over land and sea-ice source points only, so a
coastal land column is not diluted by the bare ocean beside it, and the
ice thickness over ice points only.  The sea-ice fraction itself is the
plain bilinear regrid; the statics rulebook then freezes every column at
or above :data:`woof.globe.statics.SEA_ICE_THRESHOLD`.

``python -m woof.globe.surface_seeding --calibrate`` runs the
instrument on synthetic analyses in both hemispheres and both latitude
orderings (a planted ice edge and snow line must land in the right
Gaussian rows) and through every refusal, and writes the readings.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .statics import SEA_ICE_THRESHOLD, frozen_water_columns

SEEDING_SCHEMA = "gpuwm.arwen-global-surface-seeding/v1"
#: The analysis fields this module seeds from, with the unit the mapping
#: declares for each.  The initializer's other surface fields (skin
#: temperature, the soil layers, the land mask) are seeded by
#: analysis_initial itself and only NAMED in the provenance here.
SEEDED_ANALYSIS_FIELDS = {
    "sea_ice_fraction": "1",
    "sea_ice_thickness": "m",
    "snow_water_equivalent": "kg m-2",
    "snow_depth": "m",
}
#: WRF real.exe's snow-cover flag: SNOWC = 1 where SNOW >= 10 kg/m2
#: (module_initialize_real.F), the same test woof.globe.core.landuse applies
#: for the snow-covered albedo.
SNOW_COVER_THRESHOLD_KG_M2 = 10.0
#: Bulk density of snow, kg m-3, that a kg m-2 water equivalent divided
#: by a depth in metres must read as: fresh snow sits near 50, old
#: settled snow and firn up to about 500, and nothing that is snow
#: reaches ice (917).  A water equivalent in METRES divided by the same
#: depth reads as a fraction of one, so the trap is caught by value.
SNOW_DENSITY_BOUNDS_KG_M3 = (20.0, 917.0)
#: Depth below which a cell's density is not read (the ratio of two
#: near-zero numbers says nothing about units).
SNOW_DENSITY_MIN_DEPTH_M = 0.02
#: Depth derived for a cell that holds water equivalent but no depth
#: (WRF real.exe: SNOWH = SNOW * 0.005, a 200 kg m-3 pack).
DERIVED_SNOW_DENSITY_KG_M3 = 200.0
#: Sanity ceilings.  GDAS holds 415 kg/m2 on the 2026-09-01 analysis and
#: sea ice of 5 m; the ceilings are far above what any analysis carries
#: and catch a scale error (a 1000x factor) by an order of magnitude.
MAX_SNOW_WATER_KG_M2 = 1.0e5
MAX_SNOW_DEPTH_M = 100.0
MAX_SEA_ICE_THICKNESS_M = 50.0


@dataclass(frozen=True)
class SeededSurface:
    """The seeded planes on the Gaussian grid, float64, plus provenance."""

    sea_ice_fraction: np.ndarray
    sea_ice_thickness_m: np.ndarray
    snow_water_kg_m2: np.ndarray
    snow_depth_m: np.ndarray
    snow_cover: np.ndarray
    provenance: dict

    def planes(self) -> dict[str, np.ndarray]:
        return {
            "sea_ice_fraction": self.sea_ice_fraction,
            "sea_ice_thickness_m": self.sea_ice_thickness_m,
            "snow_water_kg_m2": self.snow_water_kg_m2,
            "snow_depth_m": self.snow_depth_m,
            "snow_cover": self.snow_cover,
        }


def _stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def _plane(frame, name: str, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(frame.fields[name].values, dtype=np.float64)
    if values.ndim != 2 or values.shape != shape:
        raise ValueError(
            f"analysis field {name} has shape {values.shape}; the surface "
            f"seeding needs the analysis's {shape} surface plane"
        )
    return values


def _masked_regrid(regrid, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Bilinear regrid of ``values`` over the source points ``mask`` only:
    the weighted mean of the masked points under each target column, zero
    where no masked point contributes (METGRID.TBL's masked interpolation
    with fill_missing = 0)."""
    weight = regrid(mask.astype(np.float64))
    total = regrid(np.where(mask, values, 0.0))
    out = np.zeros_like(total)
    np.divide(total, weight, out=out, where=weight > 1.0e-9)
    return out


def snow_cover_flag(snow_water_kg_m2) -> np.ndarray:
    """WRF's SNOWC at initialization: 1 where SNOW >= 10 kg/m2, else 0."""
    return (
        np.asarray(snow_water_kg_m2, dtype=np.float64)
        >= SNOW_COVER_THRESHOLD_KG_M2
    ).astype(np.float64)


def seed_surface_from_analysis(frame, regrid, *, target_land_fraction) -> SeededSurface:
    """Seed sea ice and snow from an analysis ``frame`` onto the Gaussian
    grid through ``regrid`` (analysis_initial's periodic bilinear
    regridder).  ``target_land_fraction`` is the run's land fraction on
    the Gaussian grid (the statics' water fraction or the analysis
    regrid); snow is zeroed on its open-water columns, where no scheme
    integrates it.  Refuses by name on a missing field, a masked land or
    ice point, a value outside its physical range, or a snow plane whose
    density says its unit is not the declared one.
    """
    missing = [name for name in SEEDED_ANALYSIS_FIELDS if name not in frame.fields]
    if missing:
        raise ValueError(
            "analysis frame lacks the surface seeding fields "
            f"{', '.join(missing)}; the cold start seeds sea ice and snow "
            "from the analysis and never starts a column ice-free or "
            "snow-free by default -- decode through a mapping that carries "
            "them (rw-wps-gdas-global-analysis-grib2 does)"
        )
    land_src = np.asarray(frame.fields["land_fraction"].values, dtype=np.float64)
    if land_src.ndim != 2:
        raise ValueError("analysis land_fraction must be a surface plane")
    shape = land_src.shape
    ice = _plane(frame, "sea_ice_fraction", shape)
    thickness = _plane(frame, "sea_ice_thickness", shape)
    swe = _plane(frame, "snow_water_equivalent", shape)
    depth = _plane(frame, "snow_depth", shape)

    # Sea ice: stored without a bitmap (0 on land and open water).
    for name, values in (("sea_ice_fraction", ice), ("sea_ice_thickness", thickness)):
        bad = int(np.size(values) - np.count_nonzero(np.isfinite(values)))
        if bad:
            raise ValueError(
                f"analysis field {name} carries {bad} missing values; the "
                "sea-ice planes are stored without a bitmap and a hole in "
                "one is a decode defect, not open water"
            )
    if ice.min() < -1.0e-3 or ice.max() > 1.0 + 1.0e-3:
        raise ValueError(
            f"analysis sea_ice_fraction spans {ice.min():.4g} to "
            f"{ice.max():.4g}; a fraction lies in [0, 1] (a percent field "
            "would read up to 100)"
        )
    ice = np.clip(ice, 0.0, 1.0)
    if thickness.min() < -1.0e-6 or thickness.max() > MAX_SEA_ICE_THICKNESS_M:
        raise ValueError(
            f"analysis sea_ice_thickness spans {thickness.min():.4g} to "
            f"{thickness.max():.4g} m; sea ice is metres thick, not "
            f"centimetres or kilometres (ceiling {MAX_SEA_ICE_THICKNESS_M} m)"
        )
    thickness = np.maximum(thickness, 0.0)
    ice_src = ice >= SEA_ICE_THRESHOLD
    land_points = land_src >= 0.5
    ice_without_thickness = int(np.count_nonzero(ice_src & (thickness <= 0.0)))

    # Snow: the bitmap (NaN under preserve_mask) means no snow on open
    # water; on a land or ice point it means the analysis has no snow
    # value there, which is refused rather than read as bare ground.
    snow_masked_on_land_or_ice = 0
    masked_counts = {}
    for name, values in (("snow_water_equivalent", swe), ("snow_depth", depth)):
        masked = ~np.isfinite(values)
        masked_counts[name] = int(masked.sum())
        on_land_or_ice = int(np.count_nonzero(masked & (land_points | ice_src)))
        snow_masked_on_land_or_ice += on_land_or_ice
        if on_land_or_ice:
            raise ValueError(
                f"analysis field {name} is masked on {on_land_or_ice} land "
                "or sea-ice points; the seeding reads a snow bitmap as no "
                "snow on open water only and does not invent bare ground "
                "where the analysis carries no value"
            )
    swe = np.where(np.isfinite(swe), swe, 0.0)
    depth = np.where(np.isfinite(depth), depth, 0.0)
    if swe.min() < -1.0e-6 or swe.max() > MAX_SNOW_WATER_KG_M2:
        raise ValueError(
            f"analysis snow_water_equivalent spans {swe.min():.4g} to "
            f"{swe.max():.4g}; kg m-2 of water equivalent lie between 0 and "
            f"{MAX_SNOW_WATER_KG_M2:g}"
        )
    if depth.min() < -1.0e-6 or depth.max() > MAX_SNOW_DEPTH_M:
        raise ValueError(
            f"analysis snow_depth spans {depth.min():.4g} to "
            f"{depth.max():.4g}; metres of snow lie between 0 and "
            f"{MAX_SNOW_DEPTH_M:g}"
        )
    swe = np.maximum(swe, 0.0)
    depth = np.maximum(depth, 0.0)

    # The unit trap, settled by value: the bulk density the two planes
    # imply on every cell deep enough to read.
    readable = (depth >= SNOW_DENSITY_MIN_DEPTH_M) & (swe > 0.0)
    density = None
    if np.any(readable):
        density = float(np.median(swe[readable] / depth[readable]))
        low, high = SNOW_DENSITY_BOUNDS_KG_M3
        if density < low:
            # A ratio below the band has two readings and the value alone
            # cannot tell them apart: the water plane in metres of water
            # (the ERA5 'sd' route) or the depth plane in centimetres.  The
            # refusal names both; a refusal that named only the first sent
            # the reader to the wrong plane when the depth was the one in
            # the wrong unit.
            raise ValueError(
                "analysis snow_water_equivalent reads as metres of water, "
                "not kg m-2, or snow_depth reads as centimetres, not metres: "
                f"the median SWE / snow_depth over {int(readable.sum())} "
                f"snow-covered cells is {density:.3g} kg m-3 where snow is "
                f"{low:g} to {high:g} kg m-3 (the ERA5 'sd' route publishes "
                "metres of water equivalent; the mapping declares "
                f"{SEEDED_ANALYSIS_FIELDS['snow_water_equivalent']!r} and "
                f"{SEEDED_ANALYSIS_FIELDS['snow_depth']!r})"
            )
        if density > high:
            raise ValueError(
                "analysis snow_water_equivalent and snow_depth disagree: "
                f"the median SWE / snow_depth over {int(readable.sum())} cells is "
                f"{density:.4g} kg m-3, denser than ice ({high:g}); the depth "
                "plane is not metres or the water equivalent is not kg m-2"
            )
    # Depth derived where the analysis holds water but no depth.
    no_depth = (swe > 0.0) & (depth <= 0.0)
    derived_depth_cells = int(np.count_nonzero(no_depth))
    depth = np.where(no_depth, swe / DERIVED_SNOW_DENSITY_KG_M3, depth)

    # Onto the Gaussian grid.
    ice_target = np.clip(regrid(ice), 0.0, 1.0)
    frozen_target = frozen_water_columns(ice_target)
    thickness_target = _masked_regrid(regrid, thickness, ice > 0.0)
    snow_mask = land_points | ice_src
    swe_target = np.maximum(_masked_regrid(regrid, swe, snow_mask), 0.0)
    depth_target = np.maximum(_masked_regrid(regrid, depth, snow_mask), 0.0)
    target_land = np.asarray(target_land_fraction, dtype=np.float64)
    if target_land.shape != ice_target.shape:
        raise ValueError("target land fraction does not match the Gaussian grid")
    open_water = (target_land <= 0.5) & ~frozen_target
    zeroed = float(np.sum(swe_target[open_water]))
    swe_target = np.where(open_water, 0.0, swe_target)
    depth_target = np.where(open_water, 0.0, depth_target)
    thickness_target = np.where(frozen_target, thickness_target, thickness_target)
    cover = snow_cover_flag(swe_target)
    for name, values in (
        ("sea_ice_fraction", ice_target), ("sea_ice_thickness_m", thickness_target),
        ("snow_water_kg_m2", swe_target), ("snow_depth_m", depth_target),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"seeded {name} is not finite after regridding")

    provenance = {
        "schema": SEEDING_SCHEMA,
        "seeded_from_analysis": {
            "sea_ice_fraction": {
                "source_field": "sea_ice_fraction", "units": "1",
                "missing_policy": "reject (stored without a bitmap)",
                "source": _stats(ice), "target": _stats(ice_target),
                "source_points_at_or_above_threshold": int(ice_src.sum()),
                "threshold": SEA_ICE_THRESHOLD,
                "regrid": "bilinear",
            },
            "sea_ice_thickness_m": {
                "source_field": "sea_ice_thickness", "units": "m",
                "missing_policy": "reject (stored without a bitmap)",
                "source": _stats(thickness), "target": _stats(thickness_target),
                "ice_points_without_thickness": ice_without_thickness,
                "regrid": "bilinear over ice points (fraction > 0)",
            },
            "snow_water_kg_m2": {
                "source_field": "snow_water_equivalent", "units": "kg m-2",
                "missing_policy": "preserve_mask: masked = no snow on open water, refused on land or ice",
                "masked_source_points": masked_counts["snow_water_equivalent"],
                "source": _stats(swe), "target": _stats(swe_target),
                "median_density_kg_m3": density,
                "density_bounds_kg_m3": list(SNOW_DENSITY_BOUNDS_KG_M3),
                "density_cells": int(readable.sum()),
                "regrid": "bilinear over land and sea-ice points, zero on open water",
                "zeroed_on_open_water_kg_m2_sum": zeroed,
            },
            "snow_depth_m": {
                "source_field": "snow_depth", "units": "m",
                "missing_policy": "preserve_mask: masked = no snow on open water, refused on land or ice",
                "masked_source_points": masked_counts["snow_depth"],
                "source": _stats(depth), "target": _stats(depth_target),
                "derived_from_water_equivalent_cells": derived_depth_cells,
                "derived_density_kg_m3": DERIVED_SNOW_DENSITY_KG_M3,
                "regrid": "bilinear over land and sea-ice points, zero on open water",
            },
            "snow_cover": {
                "rule": f"1 where snow water >= {SNOW_COVER_THRESHOLD_KG_M2:g} kg/m2 (WRF real.exe SNOWC)",
                "target_columns": int(cover.sum()),
            },
        },
        "target_columns": {
            "sea_ice": int(frozen_target.sum()),
            "snow_covered": int(cover.sum()),
            "snow_covered_land": int(np.count_nonzero((cover > 0.5) & (target_land > 0.5))),
            "snow_covered_sea_ice": int(np.count_nonzero((cover > 0.5) & frozen_target)),
            "columns": int(cover.size),
        },
        "named_but_seeded_elsewhere": {
            "skin_temperature": "analysis skin temperature on land and sea-ice columns; on open water the analysis skin over the analysis's own water points only (analysis_initial.open_water_skin_temperature), held for the run on the ocean (no ocean model); an inland lake's skin then follows its own surface energy budget (native_runtime._lake_surface_step)",
            "sea_surface_temperature": "the analysis skin temperature over the analysis's own water points, on open-water columns",
            "soil_temperature": "analysis soil temperature, four layers, water columns filled from the skin in source space",
            "volumetric_soil_moisture": "analysis volumetric soil moisture, four layers, water columns filled at 0.25 in source space",
            "land_fraction": "static water fraction (real statics) or the analysis land mask (synthetic)",
        },
    }
    return SeededSurface(
        sea_ice_fraction=ice_target,
        sea_ice_thickness_m=thickness_target,
        snow_water_kg_m2=swe_target,
        snow_depth_m=depth_target,
        snow_cover=cover,
        provenance=provenance,
    )


def sea_ice_initial_soil_temperature(soil_temperature_k, skin_temperature_k,
                                     sea_ice_fraction, sea_ice_thickness_m,
                                     snow_depth_m) -> np.ndarray:
    """The four soil-temperature layers with every sea-ice column's heat
    conduction column (physics/frozen_surface) seeded in them: linear in
    depth from the ice skin through the analysed snow and ice to the
    freezing point at the ice bottom, capped at the melting point.  The
    analysed skin over a partial pack is the composite the reference
    presents (its ice weighted by the fraction, open water at the
    freezing point for the rest), so the ice skin the column starts from
    is that composite read back through the same blend the runtime
    presents; the step-0 skin the atmosphere sees is then the analysed
    one.  Every other column's layers are returned untouched.  Runs
    before the statics resolve, so the deep-soil temperature they derive
    from the bottom layer sees the seeded column."""
    from .physics import frozen_surface

    soil = np.array(soil_temperature_k, dtype=np.float64, copy=True)
    if soil.ndim != 3 or soil.shape[0] != frozen_surface.COLUMN_NODES:
        raise ValueError(
            f"the frozen column needs {frozen_surface.COLUMN_NODES} soil layers, "
            f"got soil temperature shaped {soil.shape}"
        )
    seaice = frozen_water_columns(sea_ice_fraction)
    if np.any(seaice):
        ice_skin = frozen_surface.ice_skin_from_composite(
            np.asarray(skin_temperature_k, dtype=np.float64), sea_ice_fraction, np,
            threshold=SEA_ICE_THRESHOLD,
        )
        column = frozen_surface.initial_sea_ice_column(
            ice_skin, sea_ice_thickness_m, snow_depth_m, np)
        soil = np.where(
            seaice[None], np.minimum(column.astype(np.float64), frozen_surface.MELT_POINT_K), soil)
    return soil


def land_ice_initial_skin_node(soil_temperature_k, skin_temperature_k, sea_ice_fraction,
                               landuse_category, ice_category: int) -> np.ndarray:
    """The four soil-temperature layers with every land-ice column's skin
    node (the top layer) set to the analysed skin, capped at the melting
    point; the analysed layers below it and every other column are
    untouched.  Runs after the statics resolve, which name the ice class."""
    from .physics import frozen_surface

    soil = np.array(soil_temperature_k, dtype=np.float64, copy=True)
    if soil.ndim != 3 or soil.shape[0] != frozen_surface.COLUMN_NODES:
        raise ValueError(
            f"the frozen column needs {frozen_surface.COLUMN_NODES} soil layers, "
            f"got soil temperature shaped {soil.shape}"
        )
    seaice = frozen_water_columns(sea_ice_fraction)
    landice = (np.rint(np.asarray(landuse_category)) == int(ice_category)) & ~seaice
    if np.any(landice):
        soil[0] = np.where(
            landice,
            np.minimum(np.asarray(skin_temperature_k, dtype=np.float64), frozen_surface.MELT_POINT_K),
            soil[0],
        )
    return soil


# ---------------------------------------------------------------------------
# calibration on synthetic analyses
# ---------------------------------------------------------------------------


class _Field:
    def __init__(self, values):
        self.values = values


class _Frame:
    def __init__(self, latitude, longitude, fields):
        self.latitude = latitude
        self.longitude = longitude
        self.fields = fields


def synthetic_analysis(*, nlat: int = 181, nlon: int = 360, descending: bool = True,
                       ice_edge_deg: float = 72.0, ice_south: bool = False,
                       snow_line_deg: float = 58.0, snow_south: bool = False,
                       land_band=(30.0, 66.0), swe_kg_m2: float = 60.0,
                       snow_depth_m: float = 0.30, thickness_m: float = 1.5,
                       swe_in_metres: bool = False, mask_snow_on_land: bool = False,
                       drop: str | None = None):
    """A regular analysis with a planted ice edge and snow line.

    Land is the latitude band ``land_band`` (degrees, either sign; the
    default leaves water poleward of 66 degrees for the ice), ice covers
    the water poleward of ``ice_edge_deg`` (north unless ``ice_south``),
    snow covers the land poleward of ``snow_line_deg``.
    The snow planes carry the GDAS bitmap (NaN on open water).
    """
    lat = np.linspace(90.0, -90.0, nlat) if descending else np.linspace(-90.0, 90.0, nlat)
    lon = np.arange(nlon) * (360.0 / nlon)
    lat2 = lat[:, None] * np.ones((1, nlon))
    lo, hi = land_band
    land = ((lat2 >= min(lo, hi)) & (lat2 <= max(lo, hi))).astype(np.float64)
    poleward_ice = (lat2 <= -abs(ice_edge_deg)) if ice_south else (lat2 >= abs(ice_edge_deg))
    ice = np.where((land < 0.5) & poleward_ice, 1.0, 0.0)
    thickness = np.where(ice >= 0.5, thickness_m, 0.0)
    poleward_snow = (lat2 <= -abs(snow_line_deg)) if snow_south else (lat2 >= abs(snow_line_deg))
    snow_on = (land >= 0.5) & poleward_snow
    swe = np.where(snow_on, swe_kg_m2 / 1000.0 if swe_in_metres else swe_kg_m2, 0.0)
    depth = np.where(snow_on, snow_depth_m, 0.0)
    open_water = (land < 0.5) & (ice < 0.5)
    swe = np.where(open_water, np.nan, swe)
    depth = np.where(open_water, np.nan, depth)
    if mask_snow_on_land:
        swe = np.where(land >= 0.5, np.nan, swe)
    fields = {
        "land_fraction": _Field(land),
        "sea_ice_fraction": _Field(ice),
        "sea_ice_thickness": _Field(thickness),
        "snow_water_equivalent": _Field(swe),
        "snow_depth": _Field(depth),
    }
    if drop is not None:
        fields.pop(drop)
    return _Frame(lat, lon, fields)


def _edge_reading(plane: np.ndarray, threshold: float, grid_lat: np.ndarray,
                  planted_deg: float, south: bool, columns_mask: np.ndarray):
    """Where a planted latitude edge landed on the Gaussian grid.

    ``plane`` is the seeded plane, ``columns_mask`` the columns where the
    edge is meaningful (water columns for ice, land columns for snow).
    Each Gaussian row is classified by its majority over the masked
    columns; the read edge is the midpoint between the last covered row
    and the first bare row.  Returns the read edge, its error against the
    planted latitude, the rows that disagree with the planted rule, and
    the Gaussian row spacing at the edge.
    """
    lat = np.asarray(grid_lat, dtype=np.float64)
    covered = plane >= threshold
    rows = []
    for j in range(lat.size):
        mask = columns_mask[j]
        if not mask.any():
            continue
        rows.append((float(lat[j]), float(np.mean(covered[j][mask]))))
    rows.sort(key=lambda item: item[0])
    lats = np.asarray([r[0] for r in rows])
    frac = np.asarray([r[1] for r in rows])
    if south:
        expected = lats <= -abs(planted_deg)
    else:
        expected = lats >= abs(planted_deg)
    read = frac >= 0.5
    wrong = [float(l) for l, e, r in zip(lats, expected, read) if e != r]
    # The edge: between the last row on one side and the first on the other.
    edge = None
    for k in range(lats.size - 1):
        if read[k] != read[k + 1]:
            edge = 0.5 * (lats[k] + lats[k + 1])
            spacing = lats[k + 1] - lats[k]
            break
    if edge is None:
        return {"read_edge_deg": None, "error_deg": None, "wrong_rows": wrong,
                "row_spacing_deg": None, "rows": int(lats.size)}
    return {
        "read_edge_deg": float(edge),
        "error_deg": float(edge - (-abs(planted_deg) if south else abs(planted_deg))),
        "wrong_rows": wrong,
        "row_spacing_deg": float(spacing),
        "rows": int(lats.size),
    }


def calibrate(truncation: int = 21) -> dict:
    """Run the seeding on synthetic analyses both directions and through
    every refusal; the readings are the instrument's calibration."""
    from woof.globe.spectral.grid import GaussianGrid

    from .analysis_initial import _global_regridder

    grid = GaussianGrid.create(truncation)
    grid_lat = np.asarray(grid.latitude_deg)
    readings: dict[str, object] = {
        "schema": SEEDING_SCHEMA + "/calibration",
        "truncation": int(truncation),
        "grid_shape": list(grid.shape),
        "cases": {},
    }

    def run(frame):
        regrid = _global_regridder(frame.latitude, frame.longitude, grid)
        land_target = regrid(frame.fields["land_fraction"].values)
        return seed_surface_from_analysis(frame, regrid, target_land_fraction=land_target), land_target

    # 1. northern edges on a descending source; 2. southern edges on an
    # ascending source: a planted edge must land in the Gaussian row it
    # belongs to in either hemisphere and either latitude ordering.
    for label, kwargs, ice_south, snow_south in (
        ("north-descending", dict(descending=True, ice_edge_deg=72.0, snow_line_deg=58.0,
                                  land_band=(30.0, 66.0)), False, False),
        ("south-ascending", dict(descending=False, ice_edge_deg=72.0, ice_south=True,
                                 snow_line_deg=50.0, snow_south=True,
                                 land_band=(-66.0, -30.0)), True, True),
        ("north-ascending", dict(descending=False, ice_edge_deg=76.0, snow_line_deg=64.0,
                                 land_band=(30.0, 70.0)), False, False),
        ("south-descending", dict(descending=True, ice_edge_deg=66.0, ice_south=True,
                                  snow_line_deg=45.0, snow_south=True,
                                  land_band=(-62.0, -30.0)), True, True),
    ):
        frame = synthetic_analysis(**kwargs)
        seeded, land_target = run(frame)
        water_columns = land_target <= 0.5
        land_columns = ~water_columns
        ice_reading = _edge_reading(
            seeded.sea_ice_fraction, SEA_ICE_THRESHOLD, grid_lat,
            kwargs["ice_edge_deg"], ice_south, water_columns)
        snow_reading = _edge_reading(
            seeded.snow_water_kg_m2, SNOW_COVER_THRESHOLD_KG_M2, grid_lat,
            kwargs["snow_line_deg"], snow_south, land_columns)
        source_spacing = 180.0 / (frame.latitude.size - 1)
        readings["cases"][label] = {
            "planted": {
                "ice_edge_deg": (-1 if ice_south else 1) * kwargs["ice_edge_deg"],
                "snow_line_deg": (-1 if snow_south else 1) * kwargs["snow_line_deg"],
                "source_row_spacing_deg": source_spacing,
            },
            "ice_edge": ice_reading,
            "snow_line": snow_reading,
            "sea_ice_columns": seeded.provenance["target_columns"]["sea_ice"],
            "snow_covered_columns": seeded.provenance["target_columns"]["snow_covered"],
            "thickness_on_ice_m": _stats(seeded.sea_ice_thickness_m[frozen_water_columns(seeded.sea_ice_fraction)]),
            "swe_on_snow_kg_m2": _stats(seeded.snow_water_kg_m2[seeded.snow_cover > 0.5]),
            "snow_on_open_water_kg_m2": float(np.sum(seeded.snow_water_kg_m2[water_columns & ~frozen_water_columns(seeded.sea_ice_fraction)])),
            "median_density_kg_m3": seeded.provenance["seeded_from_analysis"]["snow_water_kg_m2"]["median_density_kg_m3"],
            # The pass rule: every row the planted rule covers is read as
            # covered and vice versa, except rows within one source cell
            # plus half a Gaussian row of the planted edge, where a
            # bilinear regrid legitimately lands on either side.
            "pass": all(
                abs(abs(row) - kwargs[key]) <= source_spacing + 0.5 * abs(reading["row_spacing_deg"] or 0.0)
                for reading, key in ((ice_reading, "ice_edge_deg"), (snow_reading, "snow_line_deg"))
                for row in reading["wrong_rows"]
            ) and ice_reading["read_edge_deg"] is not None and snow_reading["read_edge_deg"] is not None,
        }

    # 3. Every refusal names its cause.
    refusals = {}
    for label, kwargs, expect in (
        ("missing-sea-ice-fraction", dict(drop="sea_ice_fraction"), "lacks the surface seeding fields sea_ice_fraction"),
        ("missing-snow-depth", dict(drop="snow_depth"), "lacks the surface seeding fields snow_depth"),
        ("swe-in-metres", dict(swe_in_metres=True), "reads as metres of water"),
        ("snow-masked-on-land", dict(mask_snow_on_land=True), "masked on"),
    ):
        frame = synthetic_analysis(**kwargs)
        try:
            run(frame)
        except ValueError as exc:
            refusals[label] = {"refused": True, "message": str(exc), "names_cause": expect in str(exc)}
        else:
            refusals[label] = {"refused": False, "message": None, "names_cause": False}
    readings["refusals"] = refusals

    # 4. An analysis with the fields but no ice and no snow seeds zeros
    # and says so (a present field of zeros is not a missing field).
    frame = synthetic_analysis(ice_edge_deg=95.0, snow_line_deg=95.0)
    seeded, _ = run(frame)
    readings["ice-free-snow-free"] = {
        "sea_ice_columns": seeded.provenance["target_columns"]["sea_ice"],
        "snow_covered_columns": seeded.provenance["target_columns"]["snow_covered"],
        "max_ice": float(seeded.sea_ice_fraction.max()),
        "max_swe": float(seeded.snow_water_kg_m2.max()),
        "refused": False,
    }
    readings["pass"] = (
        all(case["pass"] for case in readings["cases"].values())
        and all(r["refused"] and r["names_cause"] for r in refusals.values())
        and readings["ice-free-snow-free"]["max_ice"] == 0.0
        and readings["ice-free-snow-free"]["max_swe"] == 0.0
    )
    return readings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m woof.globe.surface_seeding",
        description="Calibrate the cold-start surface seeding on synthetic analyses.",
    )
    parser.add_argument("--calibrate", action="store_true", required=True)
    parser.add_argument("--truncation", type=int, default=21)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    readings = calibrate(args.truncation)
    text = json.dumps(readings, indent=1)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    def edge(reading):
        if reading["read_edge_deg"] is None:
            return "unresolved"
        return (f"{reading['read_edge_deg']:+7.2f} (err {reading['error_deg']:+5.2f} deg, "
                f"wrong rows {len(reading['wrong_rows'])})")

    for label, case in readings["cases"].items():
        print(
            f"{label:18s} ice edge planted {case['planted']['ice_edge_deg']:+6.1f} read "
            f"{edge(case['ice_edge'])}; snow line planted "
            f"{case['planted']['snow_line_deg']:+6.1f} read {edge(case['snow_line'])}; "
            f"density {case['median_density_kg_m3']:.0f} kg/m3; pass {case['pass']}"
        )
    for label, refusal in readings["refusals"].items():
        print(f"{label:24s} refused={refusal['refused']} names_cause={refusal['names_cause']}")
    print(f"ice-free/snow-free: {readings['ice-free-snow-free']}")
    print(f"CALIBRATION {'PASS' if readings['pass'] else 'FAIL'}")
    return 0 if readings["pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DERIVED_SNOW_DENSITY_KG_M3",
    "SEEDED_ANALYSIS_FIELDS",
    "SEEDING_SCHEMA",
    "SNOW_COVER_THRESHOLD_KG_M2",
    "SNOW_DENSITY_BOUNDS_KG_M3",
    "SeededSurface",
    "calibrate",
    "land_ice_initial_skin_node",
    "sea_ice_initial_soil_temperature",
    "seed_surface_from_analysis",
    "snow_cover_flag",
    "synthetic_analysis",
]
