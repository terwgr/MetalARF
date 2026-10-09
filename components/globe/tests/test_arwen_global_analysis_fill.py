"""The analysis fill: a second analysis of the same hour supplies the
surface groups the primary product lacks (analysis_initial.FILL_GROUPS),
each group whole from one source, every field's source in the receipt.

Calibrated both ways: a planted secondary reads back exactly on an
identical grid in either latitude order and with the longitude origin
moved, to the bilinear floor on a coarser grid with its mask kept; a
missing field, a wrong hour and a missing atmosphere refuse by name; the
composite cold start is complete and hydrostatic; a config without the
fill keys keeps its hash.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from datetime import datetime

import numpy as np
import pytest

from woof.globe.analysis_initial import (
    FILL_GROUPS,
    analysis_initial_state,
    fill_frame,
    floor_soil_water,
)
from woof.globe.config import load_config
from woof.globe.runner import build_transform

from test_arwen_global_analysis_initial import (
    _Field, _Frame, _analysis_cfg, _synthetic_frame,
)

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
FILL_NAMES = tuple(name for _group, names in FILL_GROUPS for name in names)


def _split(frame, *, drop=FILL_NAMES, descending_fill=False, lon0=0.0, coarse=1):
    """The synthetic frame as (primary lacking ``drop``, secondary carrying
    them), the secondary optionally in the other latitude order, with its
    longitude ring started at ``lon0`` and coarsened by ``coarse``."""
    primary = _Frame(
        frame.latitude, frame.longitude, frame.vertical_kind, frame.vertical_values,
        {k: v for k, v in frame.fields.items() if k not in drop},
        valid_time=frame.valid_time, source_cycle=frame.source_cycle,
        mapping_sha256="primary-mapping", input_sha256="primary-input",
    )
    lat = frame.latitude[::coarse]
    lon = frame.longitude[::coarse]
    fields = {}
    for name in drop:
        values = np.asarray(frame.fields[name].values)[..., ::coarse, ::coarse]
        fields[name] = _Field(values)
    if descending_fill != (frame.latitude[0] > frame.latitude[-1]):
        lat = lat[::-1]
        fields = {k: _Field(np.asarray(v.values)[..., ::-1, :]) for k, v in fields.items()}
    shift = int(np.argmin(np.abs(lon - lon0)))
    if shift:
        lon = np.roll(lon, -shift)
        fields = {k: _Field(np.roll(np.asarray(v.values), -shift, axis=-1)) for k, v in fields.items()}
    secondary = _Frame(
        lat, lon, "pressure", frame.vertical_values, fields,
        valid_time=frame.valid_time, source_cycle=frame.source_cycle,
        mapping_sha256="fill-mapping", input_sha256="fill-input",
    )
    return primary, secondary


@pytest.mark.parametrize("descending_fill", [False, True])
@pytest.mark.parametrize("lon0", [0.0, 180.0])
def test_identical_grid_fill_reads_back_exactly_in_either_order(descending_fill, lon0):
    frame = _synthetic_frame()
    primary, secondary = _split(frame, descending_fill=descending_fill, lon0=lon0)
    composite = fill_frame(primary, secondary, primary_label="P", fill_label="F")
    for name in FILL_NAMES:
        got = np.asarray(composite.fields[name].values)
        want = np.asarray(frame.fields[name].values)
        # Exact, mask included: the snow bitmap over water stays NaN where
        # it was and nowhere else.
        assert np.array_equal(np.isnan(got), np.isnan(want)), name
        assert np.array_equal(got[np.isfinite(got)], want[np.isfinite(want)]), name
        assert composite.fields[name].source == "fill"
    for name in ("air_temperature", "surface_pressure", "skin_temperature"):
        assert composite.fields[name].source == "primary"
        assert composite.fields[name].values is primary.fields[name].values
    fill = composite.fill
    assert fill["fill_grid"]["onto_primary"] == "re-indexed"
    assert {g: v["source"] for g, v in fill["groups"].items()} == {
        "soil": "fill", "snow": "fill", "sea_ice": "fill",
    }
    assert fill["fill_input_sha256"] == "fill-input"
    assert fill["fill_valid_time"] == str(frame.valid_time)
    # The composite carries the primary's identity and grid.
    assert composite.input_sha256 == "primary-input"
    assert np.array_equal(composite.latitude, primary.latitude)


def test_coarser_fill_regrids_to_the_bilinear_floor_and_keeps_the_mask():
    frame = _synthetic_frame(nlat=73, nlon=144)
    # A smooth plane on the fine grid, sampled onto a grid twice as coarse
    # for the secondary; read back through the fill it must return to the
    # fine grid within the bilinear floor of that spacing.
    lat2 = np.deg2rad(frame.latitude)[:, None] * np.ones((1, frame.longitude.size))
    lon2 = np.deg2rad(frame.longitude)[None, :] * np.ones((frame.latitude.size, 1))
    smooth = 280.0 + 5.0 * np.cos(lat2) * np.cos(lon2)
    frame.fields["soil_temperature"] = _Field(np.repeat(smooth[None], 4, axis=0))
    frame.fields["volumetric_soil_moisture"] = _Field(np.full((4, *smooth.shape), 0.3))
    primary, secondary = _split(frame, drop=("soil_temperature", "volumetric_soil_moisture"), coarse=2, descending_fill=True)
    composite = fill_frame(primary, secondary, primary_label="P", fill_label="F")
    got = np.asarray(composite.fields["soil_temperature"].values)
    assert got.shape == (4, frame.latitude.size, frame.longitude.size)
    # Bilinear floor for a wavenumber-one field at 5 degree spacing: the
    # second-derivative bound (h^2 / 8) |f''| with |f''| <= 5 * 2 (rad^-2)
    # on a 0.087 rad step is 0.01 K; the truncated coarse ring's last
    # column reads through the periodic wrap.
    assert float(np.max(np.abs(got - smooth[None]))) < 0.02
    assert np.isfinite(got).all()
    assert composite.fill["fill_grid"]["onto_primary"].startswith("bilinear")
    # The snow group with its bitmap: NaN over open water on the coarse
    # grid stays NaN on the fine grid where every corner is masked, and a
    # land point next to the coast reads its finite neighbours only.
    primary2, secondary2 = _split(frame, drop=("snow_water_equivalent", "snow_depth"), coarse=2, descending_fill=True)
    composite2 = fill_frame(primary2, secondary2, primary_label="P", fill_label="F")
    swe = np.asarray(composite2.fields["snow_water_equivalent"].values)
    coarse = np.asarray(secondary2.fields["snow_water_equivalent"].values)
    # Points that coincide with coarse points read the coarse value, mask
    # included; snow is 40 on the ridge, 0 elsewhere on land.
    assert np.array_equal(np.isnan(swe[::2, ::2]), np.isnan(coarse))
    finite = np.isfinite(swe)
    assert set(np.unique(np.round(swe[finite], 6))) <= {0.0, 40.0} | set(
        np.round(np.linspace(0.0, 40.0, 5), 6)
    )


def test_a_coast_the_two_land_masks_draw_apart_is_grown_from_its_neighbours():
    """The second product's snow bitmap is NaN over ITS water; where the
    primary calls land one cell further out, the plane takes its finite
    neighbours' mean (counted in the receipt), water keeps the bitmap, and
    a masked land point beyond the reach refuses by name."""
    frame = _synthetic_frame()
    primary, secondary = _split(frame, drop=("snow_water_equivalent", "snow_depth"))
    land = np.asarray(frame.fields["land_fraction"].values)
    # The primary's land mask one cell wider at every coast.
    wider = land.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            wider = np.maximum(wider, np.roll(np.roll(land, dy, 0), dx, 1))
    primary.fields["land_fraction"] = _Field(wider)
    composite = fill_frame(primary, secondary, primary_label="P", fill_label="F")
    swe = np.asarray(composite.fields["snow_water_equivalent"].values)
    grown = (wider >= 0.5) & (land < 0.5)
    assert grown.any()
    assert np.isfinite(swe[wider >= 0.5]).all()
    # Grown cells read a mean of neighbouring finite values (40 on the
    # snowy ridge, 0 elsewhere), never a value the plane does not carry.
    assert np.all((swe[grown] >= 0.0) & (swe[grown] <= 40.0))
    # Water beyond the wider coast keeps the bitmap.
    deep_water = ~(wider >= 0.5)
    ice = np.asarray(frame.fields["sea_ice_fraction"].values) >= 0.5
    assert np.isnan(swe[deep_water & ~ice]).all()
    group = composite.fill["groups"]["snow"]
    assert group["coastal_filled_points"]["snow_water_equivalent"] == int(grown.sum())
    assert group["coastal_filled_points"]["snow_depth"] == int(grown.sum())
    assert "3 x 3" in group["coastal_fill"]
    # An island the primary calls land in the middle of the second
    # product's ocean, out of reach (eight cells) of any value, starts
    # snow-free and is counted and located in the receipt (never
    # silently).  On the 2.5 degree planet the ocean between the ridge's
    # southern shore (5N) and the ice (70S) is 30 rows wide.
    fine = _synthetic_frame(nlat=73, nlon=144)
    primary, secondary = _split(fine, drop=("snow_water_equivalent", "snow_depth"))
    land = np.asarray(fine.fields["land_fraction"].values)
    far = land.copy()
    far[54, 72] = 1.0  # 45S, 180E
    assert land[45:64, 63:82].max() == 0.0
    assert np.asarray(fine.fields["sea_ice_fraction"].values)[45:64, 63:82].max() == 0.0
    primary.fields["land_fraction"] = _Field(far)
    composite = fill_frame(primary, secondary, primary_label="P", fill_label="F")
    swe = np.asarray(composite.fields["snow_water_equivalent"].values)
    assert swe[54, 72] == 0.0
    assumed = composite.fill["groups"]["snow"]["assumed_snow_free_points"]
    assert assumed["snow_water_equivalent"]["points"] == 1
    assert assumed["snow_water_equivalent"]["latitude_range_deg"] == [-45.0, -45.0]
    assert assumed["snow_water_equivalent"]["northern_points"] == 0
    assert assumed["snow_depth"]["points"] == 1
    # And the neighbouring water keeps its bitmap.
    assert np.isnan(swe[54, 71]) and np.isnan(swe[53, 72])


def test_partial_group_is_taken_whole_and_the_unused_field_is_named():
    frame = _synthetic_frame()
    # The primary carries the ice THICKNESS but not the concentration (the
    # IFS open-data shape): the pair comes whole from the secondary.
    primary, secondary = _split(frame, drop=("sea_ice_fraction",))
    thick = np.asarray(frame.fields["sea_ice_thickness"].values) + 0.5
    primary.fields["sea_ice_thickness"] = _Field(thick)
    secondary.fields["sea_ice_thickness"] = frame.fields["sea_ice_thickness"]
    composite = fill_frame(primary, secondary, primary_label="P", fill_label="F")
    got = np.asarray(composite.fields["sea_ice_thickness"].values)
    assert np.array_equal(got, np.asarray(frame.fields["sea_ice_thickness"].values))
    assert composite.fields["sea_ice_thickness"].source == "fill"
    group = composite.fill["groups"]["sea_ice"]
    assert group["source"] == "fill"
    assert "carries no sea_ice_fraction" in group["reason"]
    assert "sea_ice_thickness not used" in group["reason"]
    assert composite.fill["groups"]["soil"]["source"] == "primary"


def test_missing_field_in_both_products_refuses_by_name():
    frame = _synthetic_frame()
    primary, secondary = _split(frame, drop=("snow_water_equivalent", "snow_depth"))
    del secondary.fields["snow_depth"]
    with pytest.raises(ValueError, match="fill analysis \\(F\\) lacks snow_depth"):
        fill_frame(primary, secondary, primary_label="P", fill_label="F")


def test_a_fill_of_another_hour_refuses_by_name():
    frame = _synthetic_frame()
    primary, secondary = _split(frame)
    secondary.valid_time = datetime(2026, 8, 30, 12)
    with pytest.raises(ValueError, match="is not the primary analysis valid time"):
        fill_frame(primary, secondary, primary_label="P", fill_label="F")


def test_the_atmosphere_is_never_filled():
    frame = _synthetic_frame()
    primary, secondary = _split(frame, drop=("specific_humidity", "snow_depth", "snow_water_equivalent"))
    with pytest.raises(ValueError, match="lacks specific_humidity; only the surface groups"):
        fill_frame(primary, secondary, primary_label="P", fill_label="F")


def test_the_composite_cold_start_matches_the_single_product_cold_start(tmp_path):
    """The same planet through two products is the same cold start: the
    fill re-indexes the secondary's planes exactly, so every array of the
    state is bit-identical to the single-frame route, and the receipt
    names each field's product."""
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    frame = _synthetic_frame()
    primary, secondary = _split(frame, descending_fill=True, lon0=180.0)
    composite = fill_frame(primary, secondary, primary_label="P", fill_label="F")
    one, phi_one, prov_one = analysis_initial_state(cfg, transform, frame=frame)
    two, phi_two, prov_two = analysis_initial_state(cfg, transform, frame=composite)
    host = transform.backend.to_numpy
    for name in ("theta", "qv", "log_surface_pressure", "vorticity", "divergence"):
        assert np.array_equal(host(getattr(one.atmosphere, name)), host(getattr(two.atmosphere, name))), name
    for name in (
        "temperature_k", "soil_temperature_k", "soil_water_fraction",
        "sea_ice_fraction", "sea_ice_thickness_m", "land_fraction",
    ):
        assert np.array_equal(host(getattr(one.surface, name)), host(getattr(two.surface, name))), name
    for name, value in one.physics_state.arrays.items():
        assert np.array_equal(host(value), host(two.physics_state.arrays[name])), name
    assert np.array_equal(host(phi_one), host(phi_two))
    assert prov_one["fill"] is None
    sources = prov_two["field_sources"]
    assert set(sources) >= set(FILL_NAMES) | {"air_temperature", "skin_temperature"}
    assert all(sources[name] == "fill" for name in FILL_NAMES)
    assert sources["air_temperature"] == prov_two["mapping"].split("/")[-1].split("\\")[-1]
    assert prov_two["fill"]["groups"]["snow"]["source"] == "fill"
    assert prov_two["surface_seeding"] == prov_one["surface_seeding"]


def test_fill_keys_join_the_config_identity_only_when_set(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    assert cfg.analysis_fill_grib is None and cfg.analysis_fill_mapping is None
    assert "analysis_fill_grib" not in cfg.config_identity
    assert "analysis_fill_mapping" not in cfg.config_identity
    text = (tmp_path / "analysis.toml").read_text(encoding="utf-8")
    with_fill = text.replace(
        "[initial]\n",
        "[initial]\nanalysis_fill_grib = \"fill.grib\"\n"
        "analysis_fill_mapping = \"gdas-global\"\n", 1,
    )
    path = tmp_path / "fill.toml"
    path.write_text(with_fill, encoding="utf-8")
    filled = load_config(path)
    assert filled.analysis_fill_grib == "fill.grib"
    assert filled.config_identity["analysis_fill_mapping"] == "gdas-global"
    assert filled.config_hash != cfg.config_hash
    # One without the other is refused by name.
    half = text.replace("[initial]\n", "[initial]\nanalysis_fill_grib = \"fill.grib\"\n", 1)
    (tmp_path / "half.toml").write_text(half, encoding="utf-8")
    with pytest.raises(ValueError, match="name one secondary analysis together"):
        load_config(tmp_path / "half.toml")


def test_zero_soil_water_on_land_takes_the_ice_convention_or_the_dry_limit():
    """The floor both ways: a land-ice column at zero reads 1.0, a soil
    column at zero reads its class's smcdry, every positive value and
    every water column is untouched bit for bit, and a planet without a
    zero changes nothing."""
    rng = np.random.default_rng(5)
    shape = (4, 6, 8)
    moisture = rng.uniform(0.05, 0.45, shape)
    land = np.ones((6, 8)); land[0, :] = 0.0                     # row 0 water
    soil = np.full((6, 8), 3.0); soil[2, :] = 7.0; soil[0, :] = 14.0
    landuse = np.full((6, 8), 7.0); landuse[4, :] = 15.0         # row 4 land ice
    drysmc = np.linspace(0.01, 0.19, 19)
    same, record = floor_soil_water(moisture, soil, landuse, 15, land, drysmc=drysmc)
    assert same is moisture and record["land_ice_cells_set_saturated"] == 0
    planted = moisture.copy()
    planted[:, 4, 2] = 0.0            # land ice, all layers
    planted[1, 2, 5] = 0.0            # soil class 7, layer 1
    planted[0, 3, 1] = -1.0e-6        # soil class 3, layer 0, negative
    planted[2, 0, 4] = 0.0            # water column: left alone
    out, record = floor_soil_water(planted, soil, landuse, 15, land, drysmc=drysmc)
    assert record["land_ice_cells_set_saturated"] == 4
    assert record["soil_cells_set_to_smcdry"] == 2
    assert np.all(out[:, 4, 2] == 1.0)
    assert out[1, 2, 5] == drysmc[6] and out[0, 3, 1] == drysmc[2]
    assert out[2, 0, 4] == 0.0
    changed = np.zeros(shape, dtype=bool)
    changed[:, 4, 2] = True; changed[1, 2, 5] = True; changed[0, 3, 1] = True
    assert np.array_equal(out[~changed], planted[~changed])


def test_a_finer_offset_fill_places_every_layer_and_a_planted_spike_where_they_belong():
    """Third family, both directions: the secondary on a FINER grid (2 x
    2.5 degrees under a 5 degree primary) whose longitude ring is offset a
    quarter primary cell, so no primary point coincides with a secondary
    point in longitude and every value is interpolated.  Each soil layer
    carries its own smooth plane, 5 K and 0.05 apart, and must land on
    its own layer to the bilinear floor (a layer swapped or repeated reads
    5 K off).  A single fine-cell spike planted in the snow plane reaches
    exactly the one primary point whose stencil holds it, at its bilinear
    weight (a quarter), and no other point moves; the water keeps its
    bitmap."""
    coarse = _synthetic_frame()                    # 37 x 72, 5 degrees
    fine = _synthetic_frame(nlat=91, nlon=144)     # 2 x 2.5 degrees
    primary, _unused = _split(coarse, drop=FILL_NAMES)
    lon_fine = fine.longitude + 1.25               # a quarter primary cell east
    land_fine = np.asarray(fine.fields["land_fraction"].values) >= 0.5
    lat2 = np.deg2rad(fine.latitude)[:, None] * np.ones((1, lon_fine.size))
    lon2 = np.deg2rad(lon_fine)[None, :] * np.ones((fine.latitude.size, 1))
    layer = np.arange(4)[:, None, None]
    soil_t = np.where(land_fine[None], 270.0 + 5.0 * layer + 3.0 * np.cos(lat2) * np.cos(lon2), np.nan)
    soil_m = np.where(land_fine[None], 0.10 + 0.05 * layer + 0.02 * np.cos(lat2) * np.sin(lon2), np.nan)
    swe = np.array(fine.fields["snow_water_equivalent"].values, dtype=np.float64, copy=True)
    i_spike = int(np.argmin(np.abs(fine.latitude - 44.0)))
    j_spike = int(np.argmin(np.abs(lon_fine - 101.25)))
    assert land_fine[i_spike, j_spike] and swe[i_spike, j_spike] == 40.0
    swe[i_spike, j_spike] += 100.0
    secondary = _Frame(
        fine.latitude, lon_fine, "pressure", coarse.vertical_values,
        {
            "soil_temperature": _Field(soil_t),
            "volumetric_soil_moisture": _Field(soil_m),
            "snow_water_equivalent": _Field(swe),
            "snow_depth": fine.fields["snow_depth"],
            "sea_ice_fraction": fine.fields["sea_ice_fraction"],
            "sea_ice_thickness": fine.fields["sea_ice_thickness"],
        },
        valid_time=coarse.valid_time, source_cycle=coarse.source_cycle,
        mapping_sha256="fill-mapping", input_sha256="fill-input",
    )
    composite = fill_frame(primary, secondary, primary_label="P", fill_label="F")
    assert composite.fill["fill_grid"] == {
        "nlat": 91, "nlon": 144,
        "onto_primary": "bilinear, weights renormalised over finite corners",
    }

    # Direction one: every layer reads its own analytic plane on the
    # primary's interior land (land with eight land neighbours, so every
    # fine corner is land and no coastal growth is involved).  The
    # bilinear floor for a wavenumber-one field at 0.044 rad spacing is
    # (h^2 / 8) |f''| ~ 1.5e-3 K; a layer swapped reads 5 K off.
    land = np.asarray(coarse.fields["land_fraction"].values) >= 0.5
    interior = land.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            interior &= np.roll(np.roll(land, dy, 0), dx, 1)
    assert interior.sum() > 100
    plat = np.deg2rad(coarse.latitude)[:, None] * np.ones((1, coarse.longitude.size))
    plon = np.deg2rad(coarse.longitude)[None, :] * np.ones((coarse.latitude.size, 1))
    got_t = np.asarray(composite.fields["soil_temperature"].values)
    got_m = np.asarray(composite.fields["volumetric_soil_moisture"].values)
    assert got_t.shape == (4, *land.shape) and got_m.shape == (4, *land.shape)
    for k in range(4):
        want_t = 270.0 + 5.0 * k + 3.0 * np.cos(plat) * np.cos(plon)
        want_m = 0.10 + 0.05 * k + 0.02 * np.cos(plat) * np.sin(plon)
        assert float(np.max(np.abs(got_t[k][interior] - want_t[interior]))) < 0.01, k
        assert float(np.max(np.abs(got_m[k][interior] - want_m[interior]))) < 1.0e-4, k
        assert abs(float(np.mean(got_t[k][interior])) - (270.0 + 5.0 * k)) < 1.0
    assert np.isfinite(got_t[:, land]).all() and np.isfinite(got_m[:, land]).all()
    assert composite.fields["soil_temperature"].source == "fill"

    # Direction two: the spike lands on the one primary point (45N, 100E)
    # whose four fine corners include the planted cell, at the bilinear
    # weight of that corner (half in latitude, half in longitude), and
    # nowhere else: the rest of the snowy ridge (land north of 38) reads
    # 40, the snow-free land 0, and the primary's open water away from
    # the coast keeps NaN (at the coast itself a primary water point
    # whose stencil holds fine land reads the finite corners, which the
    # seeding zeroes on the run's open-water columns).
    swe_c = np.asarray(composite.fields["snow_water_equivalent"].values)
    i_hit = int(np.argmin(np.abs(coarse.latitude - 45.0)))
    j_hit = int(np.argmin(np.abs(coarse.longitude - 100.0)))
    assert abs(swe_c[i_hit, j_hit] - (40.0 + 0.25 * 100.0)) < 1.0e-9
    lat_deg = np.rad2deg(plat)
    ice = np.asarray(coarse.fields["sea_ice_fraction"].values) >= 0.5
    snowy = land & (lat_deg >= 38.0)
    elsewhere = np.ones(land.shape, dtype=bool)
    elsewhere[i_hit, j_hit] = False
    assert float(np.max(np.abs(swe_c[snowy & elsewhere] - 40.0))) < 1.0e-9
    assert float(np.max(np.abs(swe_c[land & ~snowy]))) < 1.0e-9
    water = ~land
    deep = water.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            deep &= np.roll(np.roll(water, dy, 0), dx, 1)
    assert (deep & ~ice).sum() > 100
    assert np.isnan(swe_c[deep & ~ice]).all()
    assert np.all(swe_c[ice] == 0.0)
    snow = composite.fill["groups"]["snow"]
    assert snow["source"] == "fill"
    assert "assumed_snow_free_points" not in snow
    assert snow.get("coastal_filled_points", {}).get("snow_water_equivalent", 0) == 0
