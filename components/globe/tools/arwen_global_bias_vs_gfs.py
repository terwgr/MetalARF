"""Surface-field bias of a WOOF global render tape against a GFS product,
with the polar footprints the cold-start seeding is measured on.

usage: arwen_global_bias_vs_gfs.py <gfs_pgrb2_list_csv> <frame_index> <wrfout_tape_global> <label> <out_json>

The control's scoring chain, which has run a copy of this file since
2026-09-05, reads the GFS product through the forcing-series mapping
and scores five footprints: CONUS land (25-50N,
125-65W), NH mid-latitude land (30-60N), global land, NH extratropics all
points (30-70N) and the globe, with bilinear GFS onto the tape's regular
lat-lon grid, cos-latitude area weights and land where both products say
land.  This tool adds:

- polar_north_all (60-90N, all points), polar_north_land, polar_south_all
  (60-90S), polar_south_land;
- sea_ice (the GFS product's own sea-ice fraction >= 0.5 at the tape's
  valid time, both hemispheres) and snow_covered_land (GFS water
  equivalent >= 10 kg/m2 on land), the footprints where the seeded
  surface lives.  The GFS ice and snow planes come from a second decode
  of the selected product through the global-analysis mapping, which
  carries them; the forcing-series mapping does not.

The pressure row grades the mass field.  ``mslp_hpa`` reduces each side's
own surface pressure over its own terrain height to sea level through ONE
reduction column shared by both sides: the tape's lowest-model-level
temperature (WRF's T and PB at the first level, a prognostic state
variable) brought to the ground down the standard 6.5 K/km lapse, then the
mean of the fictitious column.  With the column shared, the difference of
the two sides is the surface-pressure difference scaled by the hydrostatic
factor plus a static term from the two products' terrain mismatch; a model
temperature reaches it only through that mismatch and through the factor
itself, which rescales the surface-pressure difference by a relative
g h / (R Tm^2) per kelvin (about 4e-4 per K per km of terrain at 280 K),
so where the surface pressures agree no column temperature can create a
difference; both paths are recorded per footprint as one sensitivity in
hPa per K, and the 2 m diagnostic never enters the row.  Why: the row
used to reduce each side through its own 2 m temperature, and a change to
the surface layer's 2 m diagnostic moved every MSLP row while surface
pressure was byte-identical (CONUS rmse +0.067 hPa, NH mid-latitude
+0.057; the 2026-09-04 skin measurement and its independent check).  Each side's own lowest
level is not an option: the GFS pgrb2 0.25 product carries no temperature
on its first hybrid level, and its PRMSL and MSLET records are NCEP's own
reductions (Shuell with its hot-terrain cap; the membrane relaxation),
which would need a model temperature on the tape to mirror.

``mslp_t2_hpa`` is the retired reduction (each side through its own 2 m
temperature), kept for one release so tables written before 2026-09-05
read beside new ones; its sensitivity to the tape's T2 is recorded.

The reduction arithmetic here is the same as
woof/verify/harness/surface_bias.py (tests hold the two together); it is
inlined so the file runs against a tree whose harness predates it.

Environment: ARWEN_TREE (the checkout the mappings and the mapped-source
engine come from).
"""
import json
import os
import sys

import netCDF4
import numpy as np
from scipy.interpolate import RegularGridInterpolator

TREE = os.environ.get("ARWEN_TREE", os.getcwd())
sys.path.insert(0, TREE)
# THE PACKAGE'S DOOR, not the engine's function.  Every decode in this
# package goes through `decode_through_engine`, which places the decoder's
# scratch beside the run output and adapts the one mapping rule a published
# engine narrowed: woof 2.7.0 refuses a surface `preserve_mask` outright,
# and the global-analysis mapping below declares two.  Calling the engine's
# `decode_mapped_source` directly made this the one instrument in the tree
# that failed against a published engine while every shipped route worked.
from woof.globe.mapped_source_compat import decode_through_engine  # noqa: E402

SERIES_MAPPING = f"{TREE}/woof/authorities/rw-wps-gdas-pgrb2-0p25-grib2.mapping.json"
ANALYSIS_MAPPING = f"{TREE}/woof/authorities/rw-wps-gdas-global-analysis-grib2.mapping.json"
G, R = 9.80665, 287.05
LAPSE_K_M = 0.0065
# The tape's WRF conventions: T is theta less 300 K, PB plus P the full
# pressure, PHB plus PH the geopotential on half levels, index 0 the ground.
TAPE_THETA_OFFSET_K = 300.0
TAPE_REFERENCE_PRESSURE_PA = 100_000.0
TAPE_KAPPA = 287.0 / 1004.0
GROUND_TOLERANCE_M = 1.0
SEA_ICE_THRESHOLD = 0.5
SNOW_COVER_KG_M2 = 10.0

MSLP_REDUCTION = {
    "mslp_hpa": (
        "each side's own surface pressure over its own terrain height, reduced "
        "to sea level with the standard 6.5 K/km lapse through ONE column shared "
        "by both sides: the tape's lowest-model-level temperature brought to the "
        "ground down the lapse (surface air temperature), then the mean of the "
        "fictitious column; the row reads the mass field, never the 2 m diagnostic"
    ),
    "mslp_t2_hpa": (
        "retired reduction, each side through its own 2 m temperature; kept one "
        "release so tables written before 2026-09-05 read beside new ones; moves "
        "with the 2 m diagnostic while surface pressure is unchanged"
    ),
}


def surface_air_temperature(t_level_k, z_level_m):
    """The lowest model level's temperature brought to the ground down the lapse."""
    return np.asarray(t_level_k, dtype=np.float64) + LAPSE_K_M * np.asarray(z_level_m, dtype=np.float64)


def reduce_mslp(ps_pa, hgt_m, t_surface_k):
    """Standard-lapse reduction through a column whose surface air temperature is t_surface_k."""
    hgt = np.asarray(hgt_m, dtype=np.float64)
    tmean = np.asarray(t_surface_k, dtype=np.float64) + LAPSE_K_M * hgt / 2.0
    return np.asarray(ps_pa, dtype=np.float64) * np.exp(G * hgt / (R * tmean))


def mslp_temperature_sensitivity(ps_pa, hgt_m, t_surface_k):
    """d(MSLP)/d(T_surface), Pa per K."""
    hgt = np.asarray(hgt_m, dtype=np.float64)
    tmean = np.asarray(t_surface_k, dtype=np.float64) + LAPSE_K_M * hgt / 2.0
    return -reduce_mslp(ps_pa, hgt, t_surface_k) * G * hgt / (R * tmean ** 2)


def lowest_level_temperature(theta_minus_offset_k, pressure_pa):
    theta = np.asarray(theta_minus_offset_k, dtype=np.float64) + TAPE_THETA_OFFSET_K
    return theta * (np.asarray(pressure_pa, dtype=np.float64) / TAPE_REFERENCE_PRESSURE_PA) ** TAPE_KAPPA


def lowest_level_height(geopotential_half_m2_s2, hgt_m):
    phi = np.asarray(geopotential_half_m2_s2, dtype=np.float64)
    off = np.abs(phi[0] / G - np.asarray(hgt_m, dtype=np.float64)).max()
    if not off <= GROUND_TOLERANCE_M:
        raise ValueError(
            f"the tape's first half level sits {off:.1f} m from its terrain: index 0 "
            "is not the ground, so the lowest-level reduction would read the wrong "
            "end of the column; the tape's vertical order is not WRF's"
        )
    return (phi[0] + phi[1]) / 2.0 / G - np.asarray(hgt_m, dtype=np.float64)


def read_tape_column(dataset, hgt):
    """(t1, z1) of the tape's lowest model level; a tape without the column is refused."""
    missing = [name for name in ("T", "P", "PB", "PH", "PHB") if name not in dataset.variables]
    if missing:
        raise ValueError(
            f"tape carries no {', '.join(missing)}: the MSLP row reduces through the "
            "lowest model level and cannot be read from the 2 m diagnostic"
        )
    theta = np.asarray(dataset["T"][0, 0], dtype=np.float64)
    pressure = np.asarray(dataset["PB"][0, 0], dtype=np.float64) + np.asarray(dataset["P"][0, 0], dtype=np.float64)
    phi = np.asarray(dataset["PHB"][0, :2], dtype=np.float64) + np.asarray(dataset["PH"][0, :2], dtype=np.float64)
    return lowest_level_temperature(theta, pressure), lowest_level_height(phi, hgt)


def reduce_sides(ours, theirs):
    """Both rows on both sides (the shared column is the tape's), and the per-cell sensitivities."""
    ours = dict(ours)
    theirs = dict(theirs)
    t_surface = surface_air_temperature(ours["t1"], ours["z1"])
    for s in (ours, theirs):
        s["wspd"] = np.hypot(s["u10"], s["v10"])
        s["mslp"] = reduce_mslp(s["ps"], s["hgt"], t_surface)
        s["mslp_t2"] = reduce_mslp(s["ps"], s["hgt"], s["t2"])
    sensitivity = {
        "mslp_hpa_per_k_of_shared_column": 0.01 * (
            mslp_temperature_sensitivity(ours["ps"], ours["hgt"], t_surface)
            - mslp_temperature_sensitivity(theirs["ps"], theirs["hgt"], t_surface)
        ),
        "mslp_t2_hpa_per_k_of_tape_t2": 0.01 * mslp_temperature_sensitivity(ours["ps"], ours["hgt"], ours["t2"]),
        "terrain_mismatch_m": ours["hgt"] - theirs["hgt"],
    }
    return ours, theirs, sensitivity


def weighted_stats(diff, w, m):
    if not m.any():
        return {"bias": None, "rmse": None, "n": 0}
    d = diff[m]
    ww = w[m]
    bias = float(np.sum(d * ww) / np.sum(ww))
    rmse = float(np.sqrt(np.sum(d * d * ww) / np.sum(ww)))
    return {"bias": bias, "rmse": rmse, "n": int(m.sum())}


def score_regions(ours, theirs, regions, lat):
    """Every row and the sensitivities per footprint; both sides carry ps, hgt, t2, u10, v10 and ours t1, z1."""
    ours, theirs, sensitivity = reduce_sides(ours, theirs)
    w = np.cos(np.deg2rad(lat))
    out = {}
    for rname, m in regions.items():
        out[rname] = {
            "t2_k": weighted_stats(ours["t2"] - theirs["t2"], w, m),
            "wspd10_m_s": weighted_stats(ours["wspd"] - theirs["wspd"], w, m),
            "mslp_hpa": weighted_stats((ours["mslp"] - theirs["mslp"]) / 100.0, w, m),
            "mslp_t2_hpa": weighted_stats((ours["mslp_t2"] - theirs["mslp_t2"]) / 100.0, w, m),
            "sensitivity": {k: weighted_stats(v, w, m) for k, v in sensitivity.items()},
        }
    return out


def main():
    gfs_list, frame_index, tape_path, label, out_path = sys.argv[1:6]
    gfs_paths = gfs_list.split(",")
    frame_index = int(frame_index)
    if len(gfs_paths) == 1:
        # One product (an analysis at hour 0): the forcing-series mapping
        # needs two times to define a cadence, and the global-analysis
        # mapping carries every surface field this tool reads, so the
        # single product is decoded through it alone.
        decoded = decode_through_engine(ANALYSIS_MAPPING, gfs_paths)
    else:
        decoded = decode_through_engine(SERIES_MAPPING, gfs_paths)
    frames = decoded.frames
    fr = frames[frame_index]
    glat = np.asarray(fr.latitude, dtype=np.float64)
    glon = np.asarray(fr.longitude, dtype=np.float64) % 360.0
    if glat.ndim == 2:
        glat, glon = glat[:, 0], glon[0, :]

    def gfield(frame, name):
        v = np.asarray(frame.fields[name].values, dtype=np.float64)
        return v[-1] if v.ndim == 3 else v

    gfs = {n: gfield(fr, n) for n in (
        "air_temperature_2m", "eastward_wind_10m", "northward_wind_10m",
        "surface_pressure", "terrain_height", "land_fraction")}
    # The selected product alone, through the mapping that carries the ice
    # and snow planes (a single valid time is what that mapping decodes).
    analysis_decode = None
    if len(gfs_paths) == 1:
        analysis_frame = fr
    else:
        analysis_decode = decode_through_engine(
            ANALYSIS_MAPPING, [gfs_paths[frame_index]])
        (analysis_frame,) = analysis_decode.frames
    # A tree whose analysis mapping predates the seeded planes (the control
    # 12e9aaf9f) cannot read the sea-ice and snow footprints: they are
    # reported unread by name, never as zero, and every other footprint
    # scores.  The chain's five footprints and the polar bands need no
    # plane.
    unread = {}
    missing_planes = [n for n in ("sea_ice_fraction", "snow_water_equivalent") if n not in analysis_frame.fields]
    if missing_planes:
        reason = (
            f"the tree's global-analysis mapping ({ANALYSIS_MAPPING}) carries no "
            f"{', '.join(missing_planes)} plane: the sea_ice, sea_ice_north, sea_ice_south and "
            "snow_covered_land footprints are unread (n=0), not zero"
        )
        for n in ("sea_ice", "sea_ice_north", "sea_ice_south", "snow_covered_land"):
            unread[n] = reason
        gfs["sea_ice_fraction"] = np.full_like(gfs["land_fraction"], np.nan)
        gfs["snow_water_equivalent"] = np.full_like(gfs["land_fraction"], np.nan)
    else:
        gfs["sea_ice_fraction"] = gfield(analysis_frame, "sea_ice_fraction")
        swe = gfield(analysis_frame, "snow_water_equivalent")
        gfs["snow_water_equivalent"] = np.where(np.isfinite(swe), swe, 0.0)
    order = np.argsort(glat)
    glat_s = glat[order]
    lon_order = np.argsort(glon)
    glon_s = glon[lon_order]

    def to_tape(field, lat2d, lon2d):
        f = field[order][:, lon_order]
        f = np.concatenate([f, f[:, :1]], axis=1)
        lons = np.concatenate([glon_s, [glon_s[0] + 360.0]])
        itp = RegularGridInterpolator((glat_s, lons), f, bounds_error=False, fill_value=None)
        pts = np.stack([lat2d.ravel(), (lon2d.ravel() % 360.0)], axis=1)
        return itp(pts).reshape(lat2d.shape)

    d = netCDF4.Dataset(tape_path)
    lat = np.asarray(d["XLAT"][0], dtype=np.float64)
    lon = np.asarray(d["XLONG"][0], dtype=np.float64)
    ours = {
        "t2": np.asarray(d["T2"][0], dtype=np.float64),
        "u10": np.asarray(d["U10"][0], dtype=np.float64),
        "v10": np.asarray(d["V10"][0], dtype=np.float64),
        "ps": np.asarray(d["PSFC"][0], dtype=np.float64),
        "hgt": np.asarray(d["HGT"][0], dtype=np.float64),
        "land": np.asarray(d["LANDMASK"][0], dtype=np.float64),
        "tsk": np.asarray(d["TSK"][0], dtype=np.float64),
    }
    ours["t1"], ours["z1"] = read_tape_column(d, ours["hgt"])
    tape_seaice = np.asarray(d["SEAICE"][0], dtype=np.float64) if "SEAICE" in d.variables else None
    tape_snow = np.asarray(d["SNOW"][0], dtype=np.float64) if "SNOW" in d.variables else None
    d.close()

    theirs = {
        "t2": to_tape(gfs["air_temperature_2m"], lat, lon),
        "u10": to_tape(gfs["eastward_wind_10m"], lat, lon),
        "v10": to_tape(gfs["northward_wind_10m"], lat, lon),
        "ps": to_tape(gfs["surface_pressure"], lat, lon),
        "hgt": to_tape(gfs["terrain_height"], lat, lon),
        "land": to_tape(gfs["land_fraction"], lat, lon),
        "seaice": to_tape(gfs["sea_ice_fraction"], lat, lon),
        "snow": to_tape(gfs["snow_water_equivalent"], lat, lon),
    }

    land = (ours["land"] >= 0.5) & (theirs["land"] >= 0.5)
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    with np.errstate(invalid="ignore"):
        gfs_ice = theirs["seaice"] >= SEA_ICE_THRESHOLD  # NaN planes compare False: unread footprints are empty
        gfs_snow_land = land & (theirs["snow"] >= SNOW_COVER_KG_M2)
    regions = {
        "conus_land": land & (lat >= 25) & (lat <= 50) & (lon180 >= -125) & (lon180 <= -65),
        "nh_midlat_land": land & (lat >= 30) & (lat <= 60),
        "global_land": land,
        "nh_extratropics_all": (lat >= 30) & (lat <= 70),
        "global_all": np.ones_like(land, dtype=bool),
        "polar_north_all": lat >= 60,
        "polar_north_land": land & (lat >= 60),
        "polar_south_all": lat <= -60,
        "polar_south_land": land & (lat <= -60),
        "sea_ice": gfs_ice,
        "sea_ice_north": gfs_ice & (lat > 0),
        "sea_ice_south": gfs_ice & (lat < 0),
        "snow_covered_land": gfs_snow_land,
    }

    result = {
        "label": label, "gfs": gfs_paths[frame_index], "tape": tape_path,
        # What the decode actually did: which mapping document, which
        # engine, where the scratch went and whether the surface
        # preserve_mask rule was adapted.  An instrument that does not say
        # what it read is a number without a source.
        "decode": [dict(decoded.receipt)] + (
            [] if analysis_decode is None else [dict(analysis_decode.receipt)]),
        "mslp_reduction": dict(MSLP_REDUCTION),
        "footprints_unread": unread,
        "footprints": {
            "sea_ice": f"GFS sea-ice fraction >= {SEA_ICE_THRESHOLD} at the tape's valid time",
            "snow_covered_land": f"GFS water equivalent >= {SNOW_COVER_KG_M2} kg/m2 on land",
            "polar": "60-90 degrees, all points and land",
        },
        "tape_surface": {
            "seaice_columns": None if tape_seaice is None else int((tape_seaice >= SEA_ICE_THRESHOLD).sum()),
            "gfs_seaice_columns": int(gfs_ice.sum()),
            "seaice_agreement": None if tape_seaice is None else int(((tape_seaice >= SEA_ICE_THRESHOLD) == gfs_ice).sum()),
            "snow_covered_columns": None if tape_snow is None else int((tape_snow >= SNOW_COVER_KG_M2).sum()),
            "gfs_snow_covered_land_columns": int(gfs_snow_land.sum()),
        },
        "tape_column": {
            "lowest_level_height_m": {
                "min": float(ours["z1"].min()), "median": float(np.median(ours["z1"])), "max": float(ours["z1"].max()),
            },
        },
        "regions": score_regions(ours, theirs, regions, lat),
    }
    json.dump(result, open(out_path, "w"), indent=1)
    print(f"== {label} vs GFS ==")
    print("MSLP: mslp_hpa reduces both sides through the tape's lowest-model-level column (the row of record); "
          "MSLP(T2) is the retired reduction through each side's own 2 m temperature")
    print(f"tape surface: {result['tape_surface']}")
    for reason in sorted(set(unread.values())):
        print(f"UNREAD: {reason}")
    for rname, r in result["regions"].items():
        t, ws, p, q = r["t2_k"], r["wspd10_m_s"], r["mslp_hpa"], r["mslp_t2_hpa"]
        if t["n"] == 0:
            print(f"{rname:22s} {'UNREAD' if rname in unread else 'EMPTY'} footprint (n=0): INCOMPLETE")
            continue
        print(
            f"{rname:22s} T2 bias {t['bias']:+6.2f} K rmse {t['rmse']:5.2f} | 10m wind bias "
            f"{ws['bias']:+5.2f} rmse {ws['rmse']:4.2f} m/s | MSLP bias {p['bias']:+6.2f} rmse "
            f"{p['rmse']:5.2f} hPa | MSLP(T2) bias {q['bias']:+6.2f} rmse {q['rmse']:5.2f} hPa  (n={t['n']})"
        )


if __name__ == "__main__":
    main()
