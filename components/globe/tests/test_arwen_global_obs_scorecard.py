"""The observation scorecard instrument: every calibration family in both
directions (the sampler against a closed form and against the verification
library's own bilinear, a model against itself, planted differences at the
stations and the sounding sites, the two derived fields, the IGRA2 record,
refusals by name, admission parity).

Every planted number is read back to float64 rounding; the bars are the
ones the module's CALIBRATION table records.  No network, no card, no run.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from woof.globe import obs_scorecard as oc
from woof.globe.surface_energy import dewpoint_from_specific_humidity
from woof.globe.upper_air_scorecard import synthetic_grid

ROUNDOFF = 1.0e-12


@pytest.fixture(scope="module")
def gauss():
    return synthetic_grid(21)


@pytest.fixture(scope="module")
def regular():
    return np.arange(90.0, -90.0 - 0.5, -1.0), np.arange(0.0, 360.0, 1.0)


# -- S1: the sampler --------------------------------------------------------


@pytest.mark.parametrize("which", ["gaussian", "regular"])
def test_sampler_reads_a_bilinear_field_exactly(gauss, regular, which):
    lat, lon = (gauss.latitude_deg, gauss.longitude_deg) if which == "gaussian" else regular
    lon2, lat2 = np.meshgrid(lon, lat)
    planted = 3.0 + 0.7 * lat2 - 0.02 * lon2
    sampler = oc.RectilinearSampler(lat, lon)
    rng = np.random.default_rng(3)
    lo, hi = float(min(lat[0], lat[-1])), float(max(lat[0], lat[-1]))
    plats = rng.uniform(lo, hi, 400)
    dlon = 360.0 / lon.size
    plons = rng.uniform(0.0, 360.0 - dlon - 1.0e-6, 400)
    read = sampler.sample(planted, plats, plons)
    assert np.all(np.isfinite(read))
    assert np.max(np.abs(read - (3.0 + 0.7 * plats - 0.02 * plons))) < ROUNDOFF
    # negative longitudes are the same points
    again = sampler.sample(planted, plats, plons - 360.0)
    assert np.max(np.abs(again - read)) < ROUNDOFF


def test_sampler_crosses_the_longitude_seam(regular):
    lat, lon = regular
    lon2, lat2 = np.meshgrid(lon, lat)
    wave = np.cos(np.deg2rad(lon2)) + 0.5 * lat2
    sampler = oc.RectilinearSampler(lat, lon)
    pts_lon = np.array([359.25, 359.9, 359.01])
    pts_lat = np.array([10.4, -33.7, 71.2])
    read = sampler.sample(wave, pts_lat, pts_lon)
    for k in range(3):
        j1 = int(np.searchsorted(lat[::-1], pts_lat[k], side="right"))
        asc = lat[::-1]
        j0 = j1 - 1
        wy = (pts_lat[k] - asc[j0]) / (asc[j1] - asc[j0])
        wx = (pts_lon[k] - 359.0) / 1.0
        f = wave[::-1]
        hand = (f[j0, -1] * (1 - wx) + f[j0, 0] * wx) * (1 - wy) + (f[j1, -1] * (1 - wx) + f[j1, 0] * wx) * wy
        assert abs(read[k] - hand) < ROUNDOFF


def test_sampler_agrees_with_the_library_bilinear(regular):
    from woof.verify.obs import stations as st

    lat, lon = regular
    lon2, lat2 = np.meshgrid(lon, lat)
    texture = np.sin(np.deg2rad(3.0 * lat2)) * np.cos(np.deg2rad(2.0 * lon2)) + 0.01 * lat2
    sampler = oc.RectilinearSampler(lat, lon)
    rng = np.random.default_rng(5)
    plats = rng.uniform(-89.0, 89.0, 400)
    plons = rng.uniform(0.5, 357.5, 400)
    x, y, inside = sampler.library_positions(plats, plons)
    assert inside.all()
    lib = np.array([st.sample_field(texture, st.StationPosition("p", float(x[k]), float(y[k]))) for k in range(400)])
    assert np.max(np.abs(lib - sampler.sample(texture, plats, plons))) < ROUNDOFF


def test_sampler_refuses_poleward_points_and_nan_corners(gauss):
    sampler = oc.RectilinearSampler(gauss.latitude_deg, gauss.longitude_deg)
    ones = np.ones(sampler.shape)
    assert np.isnan(sampler.sample(ones, [89.9, -89.9], [10.0, 10.0])).all()
    assert sampler.sample(ones, [gauss.latitude_deg[0]], [0.0])[0] == 1.0   # the first row itself reads
    holed = ones.copy()
    holed[10, 20] = np.nan
    lat_pt = gauss.latitude_deg[10] + 0.3 * (gauss.latitude_deg[11] - gauss.latitude_deg[10])
    dlon = gauss.longitude_deg[1] - gauss.longitude_deg[0]
    assert np.isnan(sampler.sample(holed, [lat_pt], [gauss.longitude_deg[20] + 0.4 * dlon])[0])
    assert np.isnan(sampler.sample(holed, [lat_pt], [gauss.longitude_deg[19] + 0.4 * dlon])[0])   # the hole is its east corner
    assert sampler.sample(holed, [lat_pt], [gauss.longitude_deg[22] + 0.5 * dlon])[0] == 1.0


def test_sampler_refuses_a_grid_that_is_not_global():
    with pytest.raises(ValueError, match="not the globe"):
        oc.RectilinearSampler(np.arange(-45.0, 46.0, 1.0), np.arange(0.0, 180.0, 1.0))
    with pytest.raises(ValueError, match="strictly monotone"):
        oc.RectilinearSampler(np.array([0.0, 1.0, 1.0, 2.0]), np.arange(0.0, 360.0, 90.0))


# -- S2 and S3: surface stations -------------------------------------------


VALID = "2026-01-01T00:00:00"


def _two_models(gauss, regular):
    model = oc.synthetic_surface_fields(gauss.latitude_deg, gauss.longitude_deg)
    lat, lon = regular
    second = oc.SurfaceFields(
        latitude_deg=lat, longitude_deg=lon,
        **{k: oc._regrid_to(model, k, lat, lon)
           for k in ("t2_k", "q2_kg_kg", "u10_m_s", "v10_m_s", "surface_pressure_pa", "terrain_m")},
        land=np.ones((lat.size, lon.size), dtype=bool), source={"kind": "synthetic-regular"},
    )
    return model, second


def test_a_model_against_its_own_samples_scores_exactly_zero(gauss, regular):
    model, second = _two_models(gauss, regular)
    stations = oc.synthetic_stations(model, 300)
    obs = oc.synthetic_observations(model, stations, VALID)
    result = oc.score_surface_stations({"model": model, "other": second}, obs, VALID)
    assert result["common_station_count"] == 300
    for v in oc.SURFACE_VARIABLES:
        s = result["scores"]["model"][v]
        assert s["n"] == 300
        assert s["bias"] == 0.0 and s["rmse"] == 0.0 and s["mae"] == 0.0
        assert result["scores"]["other"][v]["n"] == 300


def test_planted_station_offsets_read_back(gauss, regular):
    model, second = _two_models(gauss, regular)
    stations = oc.synthetic_stations(model, 300)
    offsets = {"temperature_2m": 1.5, "dewpoint_2m": -2.0, "wind_speed_10m": 0.75, "mslp": 120.0}
    obs = oc.synthetic_observations(model, stations, VALID, offsets)
    result = oc.score_surface_stations({"model": model, "other": second}, obs, VALID)
    for v, delta in offsets.items():
        factor = oc.SURFACE_DISPLAY[v][2]
        s = result["scores"]["model"][v]
        assert s["n"] == 300
        assert abs(s["bias"] - (-delta * factor)) < 1.0e-9
        assert abs(s["rmse"] - abs(delta) * factor) < 1.0e-9
        assert abs(s["mae"] - abs(delta) * factor) < 1.0e-9


def test_the_scored_set_is_the_intersection_and_a_refused_stencil_drops_the_station(gauss, regular):
    model, second = _two_models(gauss, regular)
    stations = oc.synthetic_stations(model, 100)
    # the second model calls one station's cell water: the intersection loses it
    sampler = second.sampler()
    x, y, _ = sampler.library_positions([stations[0].latitude], [stations[0].longitude])
    second.land[int(round(y[0])), int(round(x[0]))] = False
    # and the first model refuses a column under another station
    x2, y2, _ = model.sampler().library_positions([stations[1].latitude], [stations[1].longitude])
    j0 = int(math.floor(y2[0]))
    i0 = int(math.floor(x2[0]))
    model.t2_k[j0, i0] = np.nan
    obs = oc.synthetic_observations(second, stations, VALID)
    result = oc.score_surface_stations({"model": model, "other": second}, obs, VALID)
    assert result["common_station_count"] == 99
    assert result["admission"]["other"]["dropped_by_reason"]["not-land"] == 1
    assert result["stencil_refused_by_variable"]["temperature_2m"] == 1
    assert result["scores"]["model"]["temperature_2m"]["n"] == 98
    assert result["scores"]["other"]["temperature_2m"]["n"] == 98
    assert result["scores"]["other"]["wind_speed_10m"]["n"] == 99


# -- S2 and S3: soundings ----------------------------------------------------


def test_upper_air_self_and_planted(gauss):
    upper = oc.synthetic_level_fields(gauss.latitude_deg, gauss.longitude_deg)
    obs = oc.synthetic_soundings(upper, 200)
    result = oc.score_soundings({"model": upper}, obs)
    for name, kind, _level in oc.UPPER_TARGETS:
        for hemi in oc.HEMISPHERES:
            row = result["scores"]["model"][name][hemi]
            assert row["n"] > 0
            if kind == "wind":
                assert row["rmsve"] == 0.0 and row["u_bias"] == 0.0 and row["v_bias"] == 0.0 and row["speed_bias"] == 0.0
            else:
                assert row["bias"] == 0.0 and row["rmse"] == 0.0 and row["mae"] == 0.0
    nh = result["scores"]["model"]["z500"]["nh"]["n"]
    sh = result["scores"]["model"]["z500"]["sh"]["n"]
    assert nh + sh == result["scores"]["model"]["z500"]["global"]["n"] == 200
    planted = {("z", 50_000.0): 12.0, ("t", 85_000.0): -1.5, ("t", 50_000.0): 0.8,
               ("u", 25_000.0): 3.0, ("v", 25_000.0): -4.0, ("u", 85_000.0): -3.0, ("v", 85_000.0): 4.0}
    obs = oc.synthetic_soundings(upper, 200, offsets=planted)
    result = oc.score_soundings({"model": upper}, obs)
    table = result["scores"]["model"]
    assert abs(table["z500"]["global"]["bias"] + 12.0) < ROUNDOFF * 1e3
    assert abs(table["z500"]["global"]["rmse"] - 12.0) < ROUNDOFF * 1e3
    assert abs(table["t850"]["global"]["bias"] - 1.5) < ROUNDOFF * 1e2
    assert abs(table["t500"]["global"]["bias"] + 0.8) < ROUNDOFF * 1e2
    assert abs(table["w250"]["global"]["u_bias"] + 3.0) < ROUNDOFF * 1e2
    assert abs(table["w250"]["global"]["v_bias"] - 4.0) < ROUNDOFF * 1e2
    assert abs(table["w250"]["global"]["rmsve"] - 5.0) < ROUNDOFF * 1e2
    assert abs(table["w850"]["global"]["u_bias"] - 3.0) < ROUNDOFF * 1e2
    assert abs(table["w850"]["global"]["v_bias"] + 4.0) < ROUNDOFF * 1e2
    assert abs(table["w850"]["global"]["rmsve"] - 5.0) < ROUNDOFF * 1e2


def test_a_texture_on_a_planted_height_offset_reads_the_quadrature_sum(gauss):
    upper = oc.synthetic_level_fields(gauss.latitude_deg, gauss.longitude_deg)
    obs = oc.synthetic_soundings(upper, 400, offsets={("z", 50_000.0): 12.0})
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(400)
    noise = 5.0 * (noise - noise.mean()) / np.sqrt(np.mean((noise - noise.mean()) ** 2))
    for k, record in enumerate(obs):
        record["levels"]["50000"]["z"] += float(noise[k])
    row = oc.score_soundings({"model": upper}, obs)["scores"]["model"]["z500"]["global"]
    assert abs(row["bias"] + 12.0) < 1.0e-9
    assert abs(row["rmse"] - 13.0) < 1.0e-9


def test_a_refused_model_column_refuses_the_site_for_every_model(gauss):
    upper = oc.synthetic_level_fields(gauss.latitude_deg, gauss.longitude_deg)
    other = oc.synthetic_level_fields(gauss.latitude_deg, gauss.longitude_deg)
    obs = oc.synthetic_soundings(upper, 50)
    site = obs[0]
    sampler = upper.sampler()
    j0, j1, wy, i0, i1, wx, inside = sampler.weights([site["latitude"]], [site["longitude"]])
    other.fields["t"][85_000.0][int(j0[0]), int(i0[0])] = np.nan
    result = oc.score_soundings({"model": upper, "other": other}, obs)
    assert result["scores"]["model"]["t850"]["global"]["n"] == 49
    assert result["scores"]["other"]["t850"]["global"]["n"] == 49
    assert result["scores"]["model"]["t500"]["global"]["n"] == 50
    # a missing observation leaves the target for every model, too
    obs[1]["levels"]["50000"]["z"] = None
    result = oc.score_soundings({"model": upper, "other": other}, obs)
    assert result["scores"]["model"]["z500"]["global"]["n"] == 49


# -- S4: derived fields -----------------------------------------------------


def test_dewpoint_round_trips_and_the_reduction_matches_the_chain_of_record():
    t = np.linspace(233.0, 313.0, 41)
    p = np.full_like(t, 95_000.0)
    e = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 273.15 + 243.5))
    q = 0.622 * e / (p - (1.0 - 0.622) * e)
    assert np.max(np.abs(dewpoint_from_specific_humidity(q, p) - t)) < 1.0e-9
    ps = np.array([101325.0, 98000.0])
    assert np.array_equal(oc.mslp_reduction(ps, np.zeros(2), np.array([288.15, 300.0])), ps)
    closed = 90000.0 * math.exp(9.80665 * 1000.0 / (287.05 * (288.15 + 0.0065 * 500.0)))
    assert abs(oc.mslp_reduction(np.array([90000.0]), np.array([1000.0]), np.array([288.15]))[0] - closed) < 1.0e-9
    fields = oc.synthetic_surface_fields(np.linspace(-80, 80, 9), np.arange(0.0, 360.0, 45.0))
    assert np.array_equal(fields.variable("mslp"), oc.mslp_reduction(fields.surface_pressure_pa, fields.terrain_m, fields.t2_k))
    assert np.array_equal(fields.variable("wind_speed_10m"), np.hypot(fields.u10_m_s, fields.v10_m_s))
    with pytest.raises(KeyError, match="unknown surface variable"):
        fields.variable("skin")


# -- S5: the IGRA2 record ---------------------------------------------------


def test_igra2_record_reads_back_its_planted_values():
    soundings = oc.parse_igra2(oc.IGRA2_SYNTHETIC)
    assert len(soundings) == 2
    first = soundings[0]
    assert first.station_id == "USM00072469"
    assert first.nominal == datetime(2026, 9, 1, 12)
    assert first.release_hhmm == "1100"
    assert abs(first.latitude - 39.75) < 1.0e-9 and abs(first.longitude + 104.83) < 1.0e-9
    assert len(first.levels) == 4
    m850 = first.mandatory(85_000.0)
    assert m850["z"] == 1512.0
    assert abs(m850["t"] - 294.35) < ROUNDOFF
    assert abs(m850["u"] - 10.0) < ROUNDOFF and abs(m850["v"]) < ROUNDOFF
    m500 = first.mandatory(50_000.0)
    assert m500["z"] == 5860.0 and abs(m500["t"] - 262.65) < ROUNDOFF
    assert abs(m500["u"]) < ROUNDOFF and abs(m500["v"] + 5.0) < ROUNDOFF
    m250 = first.mandatory(25_000.0)
    assert m250["z"] is None and m250["t"] is None and m250["wspd"] == 30.0
    assert first.mandatory(70_000.0) is None
    twelve = oc.parse_igra2(oc.IGRA2_SYNTHETIC, wanted={datetime(2026, 9, 1, 12)})
    assert [s.nominal for s in twelve] == [datetime(2026, 9, 1, 12)]
    assert oc.parse_igra2(oc.IGRA2_SYNTHETIC, wanted={datetime(2026, 9, 2, 0)})[0].mandatory(50_000.0)["z"] == 5872.0


def test_igra2_columns_match_the_archive_layout():
    row = oc.igra2_row(1, 0, 85000, 1609, 262, -9999, 220, 225, 21, zflag="B", tflag="B")
    assert row == "10 -9999  85000  1609B  262B-9999   220   225    21"
    header = oc.igra2_header("AGM00060390", 2026, 9, 1, 12, "1131", 117, 36.6899, 3.2166)
    assert header == "#AGM00060390 2026 09 01 12 1131  117 ncdc-gts           366899    32166"


def test_wind_components():
    u, v = oc.wind_components(270.0, 10.0)
    assert abs(u - 10.0) < ROUNDOFF and abs(v) < ROUNDOFF
    u, v = oc.wind_components(360.0, 5.0)
    assert abs(u) < ROUNDOFF and abs(v + 5.0) < ROUNDOFF
    u, v = oc.wind_components(90.0, 2.0)
    assert abs(u + 2.0) < ROUNDOFF and abs(v) < ROUNDOFF
    assert oc.wind_components(None, 3.0) == (None, None)


# -- S6: refusals by name ---------------------------------------------------


def test_refusals_name_the_missing_field(regular):
    lat, lon = regular
    with pytest.raises(KeyError, match="physics__t2"):
        oc.surface_fields_from_checkpoint_arrays({"physics__q2": np.zeros((3, 4))}, None, np.zeros((3, 4)), checkpoint="ck")
    shape = (lat.size, lon.size)
    frame = oc._FakeFrame(lat, lon, {name: np.zeros(shape) for name in oc.PRODUCT_SURFACE_FIELDS if name != "specific_humidity_2m"})
    with pytest.raises(ValueError, match="specific_humidity_2m"):
        oc.product_surface_fields(frame, path="gfs.f018")
    level_frame = oc._FakeFrame(
        lat, lon,
        {"geopotential_height": np.zeros((2,) + shape), "air_temperature": np.zeros((2,) + shape),
         "eastward_wind": np.zeros((2,) + shape), "northward_wind": np.zeros((2,) + shape),
         "surface_pressure": np.full(shape, 101325.0)},
        vertical_values=[25_000.0, 50_000.0],
    )
    with pytest.raises(ValueError, match="85000 Pa"):
        oc.product_level_fields(level_frame, path="gfs.f012")


def test_product_levels_are_refused_below_the_products_surface(regular):
    lat, lon = regular
    shape = (lat.size, lon.size)
    ps = np.full(shape, 101325.0)
    ps[100:110, :] = 80_000.0   # a plateau where 850 hPa is underground
    fields = {
        "geopotential_height": np.stack([np.full(shape, 10_000.0), np.full(shape, 5_500.0), np.full(shape, 1_450.0)]),
        "air_temperature": np.stack([np.full(shape, 220.0), np.full(shape, 255.0), np.full(shape, 285.0)]),
        "eastward_wind": np.zeros((3,) + shape), "northward_wind": np.zeros((3,) + shape),
        "surface_pressure": ps,
    }
    frame = oc._FakeFrame(lat, lon, fields, vertical_values=[25_000.0, 50_000.0, 85_000.0])
    lf = oc.product_level_fields(frame, path="gfs.f012")
    assert np.isnan(lf.fields["t"][85_000.0][100:110]).all()
    assert np.isfinite(lf.fields["t"][50_000.0]).all()
    assert lf.fields["z"][50_000.0][0, 0] == 5_500.0


def test_model_level_fields_carry_the_scorecards_mask(gauss):
    from woof.globe import upper_air_scorecard as sc

    reference = sc.synthetic_reference(gauss)
    lf = oc.LevelFields.from_pressure_level_fields(reference)
    for level in oc.UPPER_LEVELS_PA:
        assert np.array_equal(np.isnan(lf.fields["z"][level]), ~reference.valid[level])
    with pytest.raises(ValueError, match="lack"):
        oc.LevelFields.from_pressure_level_fields(reference, levels_pa=(30_000.0,))


# -- S7: admission parity ---------------------------------------------------


def test_library_positions_match_the_direct_index_formula(regular):
    lat, lon = regular
    sampler = oc.RectilinearSampler(lat, lon)
    rng = np.random.default_rng(9)
    plats = rng.uniform(-89.0, 89.0, 400)
    plons = rng.uniform(0.5, 357.5, 400)
    x, y, inside = sampler.library_positions(plats, plons)
    assert np.max(np.abs(x - plons)) < ROUNDOFF
    assert np.max(np.abs(y - (90.0 - plats))) < ROUNDOFF
    assert inside.all()


def test_calibrate_reports_every_family():
    payload = oc.calibrate()
    families = payload["families"]
    assert set(families) == {"S1_sampler", "S2_self", "S3_planted", "S4_derived", "S5_igra2", "S6_refusals", "S7_admission"}
    s1 = families["S1_sampler"]
    assert s1["gaussian_linear_max_abs_error"] < ROUNDOFF and s1["regular_linear_max_abs_error"] < ROUNDOFF
    assert s1["gaussian_seam_max_abs_error"] < ROUNDOFF and s1["regular_seam_max_abs_error"] < ROUNDOFF
    assert s1["library_vs_sampler_max_abs_error"] < ROUNDOFF
    assert s1["poleward_of_first_row_refused"] and s1["nan_corner_refused"]
    for v in oc.SURFACE_VARIABLES:
        assert families["S2_self"]["surface"][v]["bias"] == 0.0
        assert abs(families["S3_planted"]["surface"][v]["bias"] - families["S3_planted"]["surface"][v]["planted"]) < 1.0e-9
    assert families["S4_derived"]["dewpoint_roundtrip_max_abs_error_k"] < 1.0e-9
    assert families["S5_igra2"]["z850"] == 1512.0
    assert "physics__t2" in families["S6_refusals"]["missing_t2"]
    assert "specific_humidity_2m" in families["S6_refusals"]["missing_q2"]
    assert families["S7_admission"]["x_max_abs_error"] < ROUNDOFF


# -- the IFS open data as one more product row ------------------------------


def test_a_product_spec_may_name_its_own_mapping():
    import argparse
    from pathlib import Path

    from woof.globe.obs_scorecard import _parse_product_spec

    assert _parse_product_spec("gfs_f018=/data/gfs.f018") == ("gfs_f018", Path("/data/gfs.f018"), None)
    label, path, mapping = _parse_product_spec("ifs_f018=/data/ifs-18h.grib2@ecmwf-open-data-global-forecast")
    assert (label, path, mapping) == ("ifs_f018", Path("/data/ifs-18h.grib2"), "ecmwf-open-data-global-forecast")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_product_spec("ifs_f018")


def test_the_ifs_open_data_single_time_mapping_is_an_authority_row():
    """The open-data feed publishes the IFS orography once per cycle (the
    step-0 object) and one forecast step per file; the single-time mapping
    declares the terrain cycle-invariant and asks for no forcing series, so
    one step file with the step-0 orography record appended decodes as one
    frame.  Every field the scorecard reads is declared."""
    from woof.globe.analysis_initial import resolve_analysis_mapping
    from woof.globe.mapped_source_compat import load_mapping
    from woof.globe.obs_scorecard import PRODUCT_LEVEL_FIELDS, PRODUCT_SURFACE_FIELDS

    path = resolve_analysis_mapping("ecmwf-open-data-global-forecast")
    assert path.name == "rw-wps-ecmwf-open-data-global-forecast-grib2.mapping.json"
    mapping = load_mapping(path)
    assert mapping["fields"]["terrain_height"]["time_binding"] == "cycle_invariant"
    assert mapping["target"]["require_lateral_boundaries"] is False
    for name in PRODUCT_SURFACE_FIELDS + PRODUCT_LEVEL_FIELDS:
        assert name in mapping["fields"], name
    assert 85000.0 in mapping["coordinates"]["vertical"]["levels"]
    # The composed oper source is the engine's row and the single-time one is
    # carried here, so the sibling is resolved through the same resolver
    # rather than as a file beside the first: the two now live in different
    # directories and `with_name` would look for one of them in the other's.
    oper = load_mapping(resolve_analysis_mapping(
        "rw-wps-ecmwf-open-data-oper-grib2.mapping.json"))
    assert {k: v for k, v in oper["fields"].items() if k != "terrain_height"} == \
        {k: v for k, v in mapping["fields"].items() if k != "terrain_height"}
