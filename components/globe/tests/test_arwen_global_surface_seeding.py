"""Cold-start seeding of sea ice and snow from the analysis.

The instrument (woof.globe.surface_seeding) is calibrated on
synthetic analyses in both hemispheres and both latitude orderings, its
refusals are named, the seeded planes reach the surface state, Noah's
store, the frozen-surface skin balance, the checkpoint and the render
tape, and an ice-free planet leaves the runtime's arithmetic untouched.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_netcdf_writer  # noqa: E402

from woof.globe import surface_seeding as seeding
from woof.globe.analysis_initial import _global_regridder, analysis_initial_state
from woof.globe.physics import frozen_surface
from woof.globe.physics.native_runtime import NativePhysicsRuntime
from woof.globe.physics.native_suite import ArwenCudaColumnSuite
from woof.globe.runner import build_transform
from woof.globe.state import SEEDED_SURFACE_MEMBERS, SurfaceState
from woof.globe.statics import (
    SEA_ICE_THRESHOLD, frozen_water_columns, water_columns, xland_plane,
)
from woof.globe.spectral.grid import GaussianGrid

from test_arwen_global_analysis_initial import _analysis_cfg, _synthetic_frame
from test_arwen_global_level5_native import _exchange, _fake_modules, _options


# ---------------------------------------------------------------------------
# the instrument
# ---------------------------------------------------------------------------


def _seed(frame, truncation=21):
    grid = GaussianGrid.create(truncation)
    regrid = _global_regridder(frame.latitude, frame.longitude, grid)
    land = regrid(frame.fields["land_fraction"].values)
    return seeding.seed_surface_from_analysis(frame, regrid, target_land_fraction=land), grid, land


def test_calibration_lands_planted_edges_in_the_right_rows_both_directions():
    """A planted ice edge and snow line land within one source cell plus
    half a Gaussian row of where they were planted, in both hemispheres
    and both source latitude orderings, with no row on the wrong side."""
    readings = seeding.calibrate(21)
    assert readings["pass"], readings
    for label, case in readings["cases"].items():
        ice, snow = case["ice_edge"], case["snow_line"]
        source = case["planted"]["source_row_spacing_deg"]
        assert ice["read_edge_deg"] is not None and snow["read_edge_deg"] is not None, label
        assert abs(ice["error_deg"]) <= source + 0.5 * ice["row_spacing_deg"], (label, ice)
        assert abs(snow["error_deg"]) <= source + 0.5 * snow["row_spacing_deg"], (label, snow)
        assert ice["wrong_rows"] == [] and snow["wrong_rows"] == [], label
        assert case["sea_ice_columns"] > 0 and case["snow_covered_columns"] > 0
        assert case["snow_on_open_water_kg_m2"] == 0.0
        # The planted 60 kg/m2 over 0.30 m reads as 200 kg/m3 of snow.
        assert abs(case["median_density_kg_m3"] - 200.0) < 1.0e-6
    for label, refusal in readings["refusals"].items():
        assert refusal["refused"] and refusal["names_cause"], (label, refusal)
    assert readings["ice-free-snow-free"]["max_ice"] == 0.0
    assert readings["ice-free-snow-free"]["max_swe"] == 0.0


def test_a_missing_field_is_refused_by_name_never_zeroed():
    for name in seeding.SEEDED_ANALYSIS_FIELDS:
        frame = seeding.synthetic_analysis(drop=name)
        with pytest.raises(ValueError, match=f"lacks the surface seeding fields {name}"):
            _seed(frame)


def test_the_snow_unit_trap_is_caught_by_value():
    """Water equivalent published in metres reads as a density below 1
    kg/m3 against the depth plane and is refused; a depth plane in
    centimetres reads the same way (60 kg/m2 over "30 m" is 2 kg/m3), so
    that refusal names both planes; a depth too shallow for its water
    reads denser than ice and is refused by the other sentence."""
    with pytest.raises(ValueError, match="reads as metres of water") as metres:
        _seed(seeding.synthetic_analysis(swe_in_metres=True))
    assert "snow_depth reads as centimetres" in str(metres.value)
    with pytest.raises(ValueError, match="snow_depth reads as centimetres") as centimetres:
        _seed(seeding.synthetic_analysis(swe_kg_m2=60.0, snow_depth_m=30.0))
    assert "2 kg m-3" in str(centimetres.value)
    with pytest.raises(ValueError, match="denser than ice"):
        _seed(seeding.synthetic_analysis(swe_kg_m2=60.0, snow_depth_m=0.03))
    seeded, _, _ = _seed(seeding.synthetic_analysis(swe_kg_m2=60.0, snow_depth_m=0.30))
    assert seeded.provenance["seeded_from_analysis"]["snow_water_kg_m2"]["median_density_kg_m3"] == pytest.approx(200.0)


def test_a_snow_bitmap_means_no_snow_on_open_water_only():
    with pytest.raises(ValueError, match="masked on .* land or sea-ice points"):
        _seed(seeding.synthetic_analysis(mask_snow_on_land=True))
    seeded, grid, land = _seed(seeding.synthetic_analysis())
    frozen = frozen_water_columns(seeded.sea_ice_fraction)
    open_water = (land <= 0.5) & ~frozen
    assert open_water.any()
    assert np.all(seeded.snow_water_kg_m2[open_water] == 0.0)
    assert np.all(seeded.snow_depth_m[open_water] == 0.0)
    prov = seeded.provenance["seeded_from_analysis"]["snow_water_kg_m2"]
    assert prov["masked_source_points"] > 0
    assert "no snow on open water" in prov["missing_policy"]


def test_out_of_range_planes_are_refused():
    frame = seeding.synthetic_analysis()
    frame.fields["sea_ice_fraction"].values[...] *= 100.0  # a percent field
    with pytest.raises(ValueError, match="a fraction lies in"):
        _seed(frame)
    frame = seeding.synthetic_analysis(thickness_m=150.0)  # centimetres read as metres
    with pytest.raises(ValueError, match="sea ice is metres thick"):
        _seed(frame)


def test_a_depth_plane_of_zeros_is_derived_from_the_water_equivalent():
    frame = seeding.synthetic_analysis(snow_depth_m=0.0)
    seeded, _, _ = _seed(frame)
    prov = seeded.provenance["seeded_from_analysis"]["snow_depth_m"]
    assert prov["derived_from_water_equivalent_cells"] > 0
    snowy = seeded.snow_cover > 0.5
    assert snowy.any()
    assert np.allclose(
        seeded.snow_depth_m[snowy],
        seeded.snow_water_kg_m2[snowy] / seeding.DERIVED_SNOW_DENSITY_KG_M3,
    )


# ---------------------------------------------------------------------------
# the cold start
# ---------------------------------------------------------------------------


def test_the_analysis_cold_start_seeds_the_surface_the_store_and_the_receipt(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    state, _phi, provenance = analysis_initial_state(
        cfg, transform, frame=_synthetic_frame()
    )
    host = transform.backend.to_numpy
    lat = np.asarray(transform.grid.latitude_deg)
    ice = host(state.surface.sea_ice_fraction)
    thickness = host(state.surface.sea_ice_thickness_m)
    land = host(state.surface.land_fraction)
    frozen = frozen_water_columns(ice)
    # The frame plants ice on the water poleward of 70 degrees, 1.2 m thick.
    assert frozen.any() and np.all(np.abs(lat[frozen.any(axis=1)]) > 60.0)
    assert np.all(land[frozen] <= 0.5)
    assert np.all(thickness[frozen] > 0.5)
    assert np.all(thickness[ice == 0.0] == 0.0)
    # Snow on the land poleward of 60 degrees reaches Noah's store, depth
    # and cover flag in the physics namespace, and nowhere on open water.
    arrays = state.physics_state.arrays
    snow = host(arrays["noah_snow"])
    depth = host(arrays["noah_snowh"])
    cover = host(arrays["noah_snowc"])
    snowy = cover > 0.5
    assert snowy.any() and np.all(snow[snowy] >= seeding.SNOW_COVER_THRESHOLD_KG_M2)
    assert np.all(land[snowy] > 0.5)
    assert np.allclose(depth[snowy], snow[snowy] / 200.0, rtol=1.0e-6)
    open_water = water_columns(land, sea_ice_fraction=ice)
    assert np.all(snow[open_water] == 0.0)
    # The statics rulebook froze the ice columns over and left every other
    # column's class alone; the receipt names every source.
    categories = host(state.surface.landuse_category)
    assert np.all(categories[frozen] == 15)
    assert not np.any(categories[~frozen] == 15)
    seeded = provenance["surface_seeding"]
    assert seeded["schema"] == seeding.SEEDING_SCHEMA
    assert seeded["target_columns"]["sea_ice"] == int(frozen.sum())
    assert seeded["target_columns"]["snow_covered"] == int(snowy.sum())
    for name in ("sea_ice_fraction", "sea_ice_thickness_m", "snow_water_kg_m2", "snow_depth_m"):
        assert seeded["seeded_from_analysis"][name]["source_field"]
    assert "sea_surface_temperature" in seeded["named_but_seeded_elsewhere"]


def test_the_cold_start_refuses_an_analysis_without_the_seeding_fields(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    frame = _synthetic_frame()
    frame.fields.pop("snow_water_equivalent")
    with pytest.raises(ValueError, match="snow_water_equivalent"):
        analysis_initial_state(cfg, transform, frame=frame)


# ---------------------------------------------------------------------------
# the state and the checkpoint
# ---------------------------------------------------------------------------


def test_a_surface_built_without_ice_materializes_zero_planes():
    exchange = _exchange()
    surface = exchange.surface
    members = {name: getattr(surface, name) for name in SurfaceState.__dataclass_fields__}
    for name in SEEDED_SURFACE_MEMBERS:
        members.pop(name)
    rebuilt = SurfaceState(**members)
    for name in SEEDED_SURFACE_MEMBERS:
        plane = getattr(rebuilt, name)
        assert plane.shape == surface.land_fraction.shape
        assert plane.dtype == surface.land_fraction.dtype
        assert not np.any(plane)
    assert set(rebuilt.arrays()) == set(surface.arrays())


def test_a_pre_seeding_checkpoint_reads_with_zero_ice_and_says_so(tmp_path):
    from woof.globe.checkpoint import (
        read_checkpoint, state_from_checkpoint, write_checkpoint,
    )
    from woof.globe.config import load_config
    from woof.globe.runner import build_model_and_cold_state

    cfg = load_config(str(_shipped_configs() / "arwen_global_moist_smoke.toml"))
    model, state = build_model_and_cold_state(cfg)
    backend = model.transform.backend
    ice = backend.xp.zeros_like(state.surface.land_fraction)
    ice[0, :4] = 1.0
    state.surface.sea_ice_fraction[...] = ice
    state.surface.sea_ice_thickness_m[...] = 0.8 * ice
    path = write_checkpoint(
        tmp_path / "seeded.npz", state, config_hash=cfg.config_hash,
        to_numpy=backend.to_numpy, semi_implicit_scheme=cfg.semi_implicit_scheme,
        integrator=cfg.integrator,
    )
    metadata, arrays = read_checkpoint(
        path, expected_config_hash=cfg.config_hash,
        semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator,
    )
    assert metadata["absent_seeded_surface_arrays"] == []
    back = state_from_checkpoint(metadata, arrays, backend)
    assert np.array_equal(backend.to_numpy(back.surface.sea_ice_fraction), backend.to_numpy(ice))
    assert np.array_equal(
        backend.to_numpy(back.surface.sea_ice_thickness_m), 0.8 * backend.to_numpy(ice)
    )
    # The same checkpoint minus the two planes, as a tree before the
    # seeding wrote it: the reader fills zeros and records the absence.
    import json

    with np.load(path) as stored:
        kept = {name: stored[name] for name in stored.files if name != "__metadata__"}
        old_metadata = json.loads(str(stored["__metadata__"]))
    for member in SEEDED_SURFACE_MEMBERS:
        kept.pop(f"surface__{member}")
        old_metadata["arrays"].pop(f"surface__{member}")
    import hashlib

    from woof.globe.checkpoint import _canonical

    old_metadata.pop("self_sha256")
    old_metadata["self_sha256"] = hashlib.sha256(_canonical(old_metadata)).hexdigest()
    older = tmp_path / "older.npz"
    np.savez_compressed(
        older, __metadata__=np.asarray(json.dumps(old_metadata, sort_keys=True)), **kept
    )
    metadata, arrays = read_checkpoint(
        older, expected_config_hash=cfg.config_hash,
        semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator,
    )
    assert metadata["absent_seeded_surface_arrays"] == sorted(
        f"surface__{member}" for member in SEEDED_SURFACE_MEMBERS
    )
    back = state_from_checkpoint(metadata, arrays, backend)
    for member in SEEDED_SURFACE_MEMBERS:
        assert not np.any(backend.to_numpy(getattr(back.surface, member)))


# ---------------------------------------------------------------------------
# the frozen-surface skin balance
# ---------------------------------------------------------------------------


def _column_forcing(shape, *, glw, dt_s, layers=None, dz=None, snow=0.0, bottom=260.0,
                    extra=None, emissivity=0.98):
    eps = np.full(shape, emissivity, np.float32)
    if layers is None:
        layers = np.full((4,) + shape, 260.0, np.float32)
    if dz is None:
        dz = frozen_surface.land_ice_layer_thickness(layers[0], np)
    if extra is None:
        extra = np.full(shape, (frozen_surface.LAND_ICE_BOTTOM_DEPTH_M
                                - sum(frozen_surface.LAND_ICE_LAYER_THICKNESS_M))
                        / frozen_surface.ICE_CONDUCTIVITY_W_M_K, np.float32)
    return dict(
        layers=layers, dz=dz, snow_depth_m=np.full(shape, snow, np.float32),
        bottom_k=np.full(shape, bottom, np.float32), bottom_extra_resistance=extra,
        swdown=np.zeros(shape), albedo=np.full(shape, 0.7), glw=np.full(shape, glw),
        emissivity=eps, hfx=np.zeros(shape), qfx=np.zeros(shape), dt_s=dt_s, xp=np,
    )


def test_the_column_holds_a_balance_conserves_energy_and_caps_at_melting():
    """A column in radiative balance with its sky and its bottom stays put;
    under a deficit the skin cools and the column's energy change equals
    the surface deficit less the bottom conduction (the implicit step's
    own arithmetic); above the melting point every node is capped."""
    shape = (2, 3)
    sigma = frozen_surface.STEFAN_BOLTZMANN_W_M2_K4
    emission = 0.98 * sigma * 260.0 ** 4
    balanced = frozen_surface.column_step(**_column_forcing(shape, glw=emission / 0.98, dt_s=60.0))
    assert np.allclose(balanced, 260.0, atol=1.0e-3)
    forcing = _column_forcing(shape, glw=150.0, dt_s=600.0)
    cooled = frozen_surface.column_step(**forcing)
    assert cooled.dtype == np.float32 and cooled.shape == (4,) + shape
    assert np.all(cooled[0] < 260.0)
    # Deeper nodes lag the skin: the column has inertia the skin alone lacks.
    assert np.all(cooled[0] < cooled[1]) and np.all(cooled[1] <= cooled[2]) and np.all(cooled[2] <= cooled[3])
    assert np.all(cooled[3] > 259.99)
    capacity = frozen_surface.ICE_VOLUMETRIC_HEAT_CAPACITY_J_M3_K * np.asarray(forcing["dz"], np.float64)
    stored = float(np.sum(capacity[:, 0, 0] * (cooled[:, 0, 0].astype(np.float64) - 260.0)))
    t0 = cooled[0, 0, 0].astype(np.float64)
    surface_net = (0.98 * 150.0 - 0.98 * sigma * 260.0 ** 4
                   - 4.0 * 0.98 * sigma * 260.0 ** 3 * (t0 - 260.0))
    g_bottom = 1.0 / (0.5 * forcing["dz"][3, 0, 0] / frozen_surface.ICE_CONDUCTIVITY_W_M_K
                      + float(forcing["bottom_extra_resistance"][0, 0]))
    bottom_flux = g_bottom * (cooled[3, 0, 0].astype(np.float64) - 260.0)
    assert stored == pytest.approx(600.0 * (surface_net - bottom_flux), rel=1.0e-3)
    warm = _column_forcing(shape, glw=320.0, dt_s=3600.0,
                           layers=np.full((4,) + shape, 272.0, np.float32), bottom=271.36)
    warm["swdown"] = np.full(shape, 800.0)
    warm["albedo"] = np.full(shape, 0.2)
    melted = frozen_surface.column_step(**warm)
    assert np.all(melted[0] == np.float32(frozen_surface.MELT_POINT_K))
    assert np.all(melted <= np.float32(frozen_surface.MELT_POINT_K))


def test_snow_is_the_top_of_the_column_and_thin_ice_is_floored():
    """The nodes take the medium at their depth: with 0.3 m of snow on
    1.5 m of ice the top nodes are snow (its conductivity and capacity)
    and the deeper ones ice, the column is the snow plus the ice, and the
    skin under the same sky cools further in an hour than on bare ice
    (snow conducts less heat up from the warm ice below) but not as far
    as a skin cut off by a lumped resistance would; a thickness of zero
    takes the floor, not an infinite conductance."""
    shape = (1, 2)
    bare_dz = frozen_surface.sea_ice_layer_thickness(
        np.full(shape, 1.5, np.float32), np.zeros(shape, np.float32), np)
    snowy_dz = frozen_surface.sea_ice_layer_thickness(
        np.full(shape, 1.5, np.float32), np.full(shape, 0.3, np.float32), np)
    assert bare_dz.sum(axis=0)[0, 0] == pytest.approx(1.5, rel=1e-5)
    assert snowy_dz.sum(axis=0)[0, 0] == pytest.approx(1.8, rel=1e-5)
    # the nodes follow the snow: 0.10 m skin node, the other 0.20 m of snow
    # as the second node, the ice split in two below
    np.testing.assert_allclose(snowy_dz[:, 0, 0], [0.10, 0.20, 0.75, 0.75], rtol=1e-5)
    k, c = frozen_surface.column_properties(snowy_dz, np.full(shape, 0.3, np.float32), np)
    assert k[0, 0, 0] == np.float32(frozen_surface.SNOW_CONDUCTIVITY_W_M_K)
    assert k[1, 0, 0] == np.float32(frozen_surface.SNOW_CONDUCTIVITY_W_M_K)
    assert k[2, 0, 0] == np.float32(frozen_surface.ICE_CONDUCTIVITY_W_M_K)
    assert c[0, 0, 0] == np.float32(frozen_surface.SNOW_VOLUMETRIC_HEAT_CAPACITY_J_M3_K)
    assert k[3, 0, 0] == np.float32(frozen_surface.ICE_CONDUCTIVITY_W_M_K)
    # a 0.25 m snow cover (the Antarctic pack's 81 kg/m2 at 330 kg/m3) is
    # the whole insulation: the conductance from the skin to the ice is
    # that of the snow, a quarter of the earlier layout's
    quarter = frozen_surface.sea_ice_layer_thickness(
        np.full(shape, 1.5, np.float32), np.full(shape, 0.25, np.float32), np)
    np.testing.assert_allclose(quarter[:, 0, 0], [0.10, 0.15, 0.75, 0.75], rtol=1e-5)
    kq, _ = frozen_surface.column_properties(quarter, np.full(shape, 0.25, np.float32), np)
    g01 = 1.0 / (0.5 * 0.10 / float(kq[0, 0, 0]) + 0.5 * 0.15 / float(kq[1, 0, 0]))
    assert g01 == pytest.approx(1.0 / (0.125 / frozen_surface.SNOW_CONDUCTIVITY_W_M_K), rel=1e-5)
    # a cover shallower than the skin cap keeps the bare-ice layout (the
    # Arctic pack's 3 to 5 cm of September snow): the skin node stays
    # 0.10 m, the rest split in three, and the midpoint rule decides the
    # medium (a 0.06 m cover reaches the skin node's midpoint, snow; a
    # 0.01 m dusting does not, ice)
    for cover, medium in ((0.01, frozen_surface.ICE_CONDUCTIVITY_W_M_K), (0.06, frozen_surface.SNOW_CONDUCTIVITY_W_M_K)):
        thin = frozen_surface.sea_ice_layer_thickness(
            np.full(shape, 1.5, np.float32), np.full(shape, cover, np.float32), np)
        assert thin[0, 0, 0] == pytest.approx(0.10, rel=1e-5)
        assert thin[1, 0, 0] == pytest.approx((1.5 + cover - 0.10) / 3.0, rel=1e-5)
        kt, _ = frozen_surface.column_properties(thin, np.full(shape, cover, np.float32), np)
        assert kt[0, 0, 0] == np.float32(medium)
    assert frozen_surface.SNOW_LAYER_MIN_M == frozen_surface.SEA_ICE_SKIN_LAYER_MAX_M
    k, c = frozen_surface.column_properties(bare_dz, np.zeros(shape, np.float32), np)
    assert np.all(k == np.float32(frozen_surface.ICE_CONDUCTIVITY_W_M_K))
    bare_layers = frozen_surface.initial_sea_ice_column(
        np.full(shape, 262.0, np.float32), np.full(shape, 1.5, np.float32),
        np.zeros(shape, np.float32), np)
    snowy_layers = frozen_surface.initial_sea_ice_column(
        np.full(shape, 262.0, np.float32), np.full(shape, 1.5, np.float32),
        np.full(shape, 0.3, np.float32), np)
    bare = frozen_surface.column_step(**_column_forcing(
        shape, glw=150.0, dt_s=3600.0, layers=bare_layers, dz=bare_dz, bottom=271.36,
        extra=np.zeros(shape, np.float32)))
    snowy = frozen_surface.column_step(**_column_forcing(
        shape, glw=150.0, dt_s=3600.0, layers=snowy_layers, dz=snowy_dz, bottom=271.36,
        extra=np.zeros(shape, np.float32), snow=0.3))
    assert np.all(snowy[0] < bare[0]) and np.all(bare[0] < 262.0)
    assert np.all(snowy[0] > 240.0)
    thin = frozen_surface.sea_ice_layer_thickness(
        np.array([0.0, 10.0], np.float32), np.zeros(2, np.float32), np)
    assert thin[:, 0].sum() == pytest.approx(frozen_surface.MIN_SEA_ICE_THICKNESS_M, rel=1e-5)
    assert thin[:, 1].sum() == pytest.approx(frozen_surface.MAX_SEA_ICE_THICKNESS_M, rel=1e-5)
    assert thin[0, 1] == pytest.approx(frozen_surface.SEA_ICE_SKIN_LAYER_MAX_M, rel=1e-5)
    q = frozen_surface.saturation_specific_humidity_over_ice(
        np.array([253.15, 273.15], np.float32), np.array([100000.0, 100000.0], np.float32), np
    )
    assert q[0] == pytest.approx(0.622 * 103.2 / (100000.0 - 0.378 * 103.2), rel=0.02)
    assert q[1] == pytest.approx(0.622 * 611.15 / (100000.0 - 0.378 * 611.15), rel=1e-4)


def test_the_cold_start_seeds_the_frozen_column_and_the_pack_blends_its_skin():
    """A sea-ice column starts linear from the analysed skin to the
    freezing point at the ice bottom, a land-ice column keeps the
    analysed soil under a skin node, other columns are untouched; a
    partial pack presents the fraction-weighted skin."""
    soil = np.full((4, 1, 4), 280.0)
    skin = np.array([[250.0, 240.0, 290.0, 285.0]])
    ice = np.array([[1.0, 0.0, 0.0, 0.0]])
    thickness = np.array([[1.5, 0.0, 0.0, 0.0]])
    landuse = np.array([[15.0, 15.0, 7.0, 15.0]])
    snow_depth = np.array([[0.2, 0.0, 0.0, 0.0]])
    out = seeding.sea_ice_initial_soil_temperature(soil, skin, ice, thickness, snow_depth)
    expected = frozen_surface.initial_sea_ice_column(
        np.full((1, 1), 250.0, np.float32), np.full((1, 1), 1.5, np.float32),
        np.full((1, 1), 0.2, np.float32), np)
    np.testing.assert_allclose(out[:, 0, 0], expected[:, 0, 0], rtol=1e-6)
    np.testing.assert_array_equal(out[:, 0, 1:], 280.0)  # every other column untouched
    out = seeding.land_ice_initial_skin_node(out, skin, ice, landuse, 15)
    np.testing.assert_allclose(out[:, 0, 0], expected[:, 0, 0], rtol=1e-6)  # sea ice keeps its column
    assert out[0, 0, 1] == 240.0 and np.all(out[1:, 0, 1] == 280.0)  # land ice: skin node, analysed soil below
    np.testing.assert_array_equal(out[:, 0, 2], 280.0)  # bare land untouched
    assert out[0, 0, 3] == 273.15  # a land-ice skin above melting is capped
    with pytest.raises(ValueError, match="4 soil layers"):
        seeding.sea_ice_initial_soil_temperature(soil[:2], skin, ice, thickness, snow_depth)
    with pytest.raises(ValueError, match="4 soil layers"):
        seeding.land_ice_initial_skin_node(soil[:2], skin, ice, landuse, 15)
    blend = frozen_surface.blended_surface_temperature(
        np.array([250.0, 250.0, 250.0], np.float32), np.array([1.0, 0.6, 0.5], np.float32), np)
    np.testing.assert_allclose(blend, [250.0, 0.6 * 250.0 + 0.4 * 271.36, 0.5 * 250.0 + 0.5 * 271.36], rtol=1e-6)
    # A partial pack's analysed skin is the composite: the seeded column
    # starts from the ice skin whose blend is that composite (a cell at
    # fraction 0.6 reading 258.5 K holds 250 K ice), so the step-0 skin
    # the runtime presents is the analysis's.
    partial = np.array([[0.6]])
    composite = np.array([[0.6 * 250.0 + 0.4 * 271.36]])
    out = seeding.sea_ice_initial_soil_temperature(
        np.full((4, 1, 1), 280.0), composite, partial, np.array([[1.5]]), np.array([[0.0]]))
    from_ice_skin = frozen_surface.initial_sea_ice_column(
        np.full((1, 1), 250.0, np.float32), np.full((1, 1), 1.5, np.float32), np.zeros((1, 1), np.float32), np)
    np.testing.assert_allclose(out[:, 0, 0], from_ice_skin[:, 0, 0], atol=1e-3)
    assert out[0, 0, 0] < composite[0, 0]   # the ice under a partial pack is colder than the composite


def _freeze(exchange, *, sea_ice_rows=(0, 1), land_ice_columns=()):
    """Plant sea ice on the water columns of ``sea_ice_rows`` (1.5 m, no
    snow) and the land-ice class on ``land_ice_columns``; the categories
    follow the statics rulebook (ice class 15, ice soil 16)."""
    surface = exchange.surface
    lf = np.asarray(surface.land_fraction)
    water = water_columns(lf)
    ice = np.zeros_like(lf)
    for row in sea_ice_rows:
        ice[row][water[row]] = 1.0
    surface.sea_ice_fraction[...] = ice
    surface.sea_ice_thickness_m[...] = 1.5 * ice
    frozen = ice >= SEA_ICE_THRESHOLD
    surface.landuse_category[...] = np.where(frozen, 15.0, surface.landuse_category)
    surface.soil_category_top[...] = np.where(frozen, 16.0, surface.soil_category_top)
    for j, i in land_ice_columns:
        assert not water[j, i]
        surface.landuse_category[j, i] = 15.0
    surface.soil_temperature_k[...] = seeding.land_ice_initial_skin_node(
        seeding.sea_ice_initial_soil_temperature(
            surface.soil_temperature_k, surface.temperature_k, surface.sea_ice_fraction,
            surface.sea_ice_thickness_m, 0.0 * surface.sea_ice_fraction,
        ),
        surface.temperature_k, surface.sea_ice_fraction, surface.landuse_category, 15,
    )
    return frozen


def test_frozen_columns_are_land_to_the_flag_and_skipped_by_noah():
    exchange = _exchange()
    frozen = _freeze(exchange)
    assert frozen.any()
    calls = []
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules(noah_calls=calls)
    )
    result = suite.step(exchange)
    (first,) = calls
    # The kernel's xland is 1 on every frozen column and the pre-seeding
    # construction everywhere else; xice is the analysed fraction.
    assert np.all(first["xland"][frozen] == 1.0)
    lf = np.asarray(exchange.surface.land_fraction)
    expected = np.asarray(1.0 + (1.0 - lf), np.float32)
    assert np.array_equal(first["xland"][~frozen], expected[~frozen])
    assert np.array_equal(first["xice"], np.asarray(exchange.surface.sea_ice_fraction, np.float32))
    assert np.array_equal(first["ivgtyp"][frozen], np.full(int(frozen.sum()), 15, np.int32))
    diagnostics = result.diagnostics if hasattr(result, "diagnostics") else None
    assert diagnostics is None or diagnostics.get("frozen_surface_columns", int(frozen.sum())) == int(frozen.sum())


def test_the_frozen_surface_step_moves_the_skin_and_refreshes_qsfc():
    """With the fake radiation (no shortwave, 300 W/m2 downward longwave,
    a 269.7 K radiative balance) a 272.5 K sea-ice skin emits more than
    it receives and conducts to the 271.36 K ice bottom, so it cools;
    every other column's skin is untouched, and the surface layer's
    saturation humidity on the frozen columns is the value over ice at
    the new skin."""
    baseline = _exchange()
    exchange = _exchange()
    lf = np.asarray(exchange.surface.land_fraction)
    planted = np.zeros_like(lf, dtype=bool)
    for row in (0, 1):
        planted[row][water_columns(lf)[row]] = True
    for planet in (baseline, exchange):
        planet.surface.temperature_k[...] = np.where(planted, 272.5, planet.surface.temperature_k)
    frozen = _freeze(exchange)
    assert np.array_equal(frozen, planted)
    skin_before = np.array(exchange.surface.temperature_k, copy=True)
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=_fake_modules())
    result = suite.step(exchange)
    unseeded = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules()
    ).step(baseline)
    skin_after = np.asarray(result.surface.temperature_k)
    assert np.all(skin_after[frozen] < skin_before[frozen])
    assert np.all(skin_after[frozen] > 269.0)
    # The columns outside the ice are bit-for-bit the ice-free planet's.
    assert np.array_equal(
        skin_after[~frozen], np.asarray(unseeded.surface.temperature_k)[~frozen]
    )
    assert np.array_equal(
        np.asarray(result.theta)[:, ~frozen], np.asarray(unseeded.theta)[:, ~frozen]
    )
    arrays = result.physics_state.arrays
    psfc = np.asarray(exchange.p_half)[-1]
    expected = frozen_surface.saturation_specific_humidity_over_ice(
        skin_after, psfc, np
    )
    assert np.allclose(arrays["qsfc"][frozen], expected[frozen], rtol=1.0e-5)
    assert np.allclose(arrays["noah_qsfc"][frozen], expected[frozen], rtol=1.0e-5)
    assert result.physics_state.metadata["frozen_surface_calls"] == 1


def test_a_land_ice_column_conducts_to_the_deep_soil_temperature():
    exchange = _exchange()
    lf = np.asarray(exchange.surface.land_fraction)
    land = np.argwhere(~water_columns(lf))
    j, i = (int(v) for v in land[0])
    exchange.surface.temperature_k[j, i] = 250.0
    exchange.surface.deep_soil_temperature_k[j, i] = 250.0
    exchange.surface.soil_temperature_k[:, j, i] = 250.0
    _freeze(exchange, sea_ice_rows=(), land_ice_columns=((j, i),))
    # A skin in radiative balance with its deep temperature and the fake
    # sky stays put only if conduction and emission balance; here the fake
    # 300 W/m2 sky exceeds the 250 K emission, so the skin warms toward it
    # while the deep temperature holds it back through one metre of firn.
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=_fake_modules())
    result = suite.step(exchange)
    skin = float(np.asarray(result.surface.temperature_k)[j, i])
    assert 250.0 < skin < 273.15


def test_an_ice_free_planet_never_enters_the_frozen_arithmetic(monkeypatch):
    exchange = _exchange()
    assert not np.any(exchange.surface.sea_ice_fraction)
    assert not np.any(np.asarray(exchange.surface.landuse_category) == 15)

    def refuse(**kwargs):
        raise AssertionError("column_step ran on an ice-free planet")

    monkeypatch.setattr(frozen_surface, "column_step", refuse)
    calls = []
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules(noah_calls=calls)
    )
    result = suite.step(exchange)
    (first,) = calls
    assert np.all(first["xice"] == 0.0)
    assert "frozen_surface_calls" not in result.physics_state.metadata
    assert np.array_equal(
        first["xland"],
        np.asarray(1.0 + (1.0 - np.asarray(exchange.surface.land_fraction)), np.float32),
    )


def test_frozen_columns_hand_the_surface_layer_the_ice_class_moisture_availability():
    exchange = _exchange()
    frozen = _freeze(exchange)
    seen = []
    modules = _fake_modules()
    stock = modules["woof.globe.core.sfclay"].sfclay

    def sfclay(*args, **kwargs):
        seen.append(np.array(args[10], copy=True))  # mavail
        return stock(*args, **kwargs)

    modules["woof.globe.core.sfclay"] = SimpleNamespace(sfclay=sfclay)
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=modules)
    suite.step(exchange)
    mavail = seen[0]
    # LANDUSE.TBL MODIS snow-and-ice row: SLMO 0.95; open water stays 1.
    assert np.allclose(mavail[frozen], 0.95)
    water = water_columns(np.asarray(exchange.surface.land_fraction), sea_ice_fraction=np.asarray(exchange.surface.sea_ice_fraction))
    assert np.all(mavail[water] == 1.0)


def test_xland_plane_is_shared_by_the_statics_and_the_runtime():
    lf = np.array([[0.0, 0.3, 0.5, 0.7, 1.0]])
    ice = np.array([[1.0, 0.5, 0.49, 0.0, 0.0]])
    plane = xland_plane(lf, ice)
    assert plane.dtype == np.float32
    assert plane.tolist() == [[1.0, 1.0, np.float32(1.5), np.float32(1.3), 1.0]]
    assert water_columns(lf, sea_ice_fraction=ice).tolist() == [[False, False, True, False, False]]
    batch = SimpleNamespace(xp=np, surface=SimpleNamespace(land_fraction=lf, sea_ice_fraction=ice))
    assert np.array_equal(NativePhysicsRuntime._xland(batch), plane)


# ---------------------------------------------------------------------------
# the render tape
# ---------------------------------------------------------------------------


@requires_netcdf_writer
def test_the_render_tape_carries_the_seeded_surface(tmp_path):
    import netCDF4

    from woof.globe.checkpoint import write_checkpoint
    from woof.globe.config import load_config
    from woof.globe.runner import build_model_and_cold_state
    from woof.globe.wrfout_export import export_wrfout

    cfg = load_config(str(_shipped_configs() / "arwen_global_moist_smoke.toml"))
    model, state = build_model_and_cold_state(cfg)
    backend = model.transform.backend
    xp = backend.xp
    state.surface.sea_ice_fraction[0, :] = 1.0
    state.surface.sea_ice_thickness_m[0, :] = 2.0
    shape = state.surface.land_fraction.shape
    state.physics_state.arrays["noah_snow"] = xp.full(shape, 25.0, dtype=backend.float_dtype)
    state.physics_state.arrays["noah_snowh"] = xp.full(shape, 0.125, dtype=backend.float_dtype)
    state.physics_state.arrays["noah_snowc"] = xp.ones(shape, dtype=backend.float_dtype)
    path = write_checkpoint(
        tmp_path / "step.npz", state, config_hash=cfg.config_hash,
        to_numpy=backend.to_numpy, semi_implicit_scheme=cfg.semi_implicit_scheme,
        integrator=cfg.integrator,
    )
    (tape,) = export_wrfout(
        cfg, [path], tmp_path / "tape", nlat=18, nlon=36,
        start_date="2026-09-01_00:00:00",
    )
    with netCDF4.Dataset(tape) as d:
        seaice = np.asarray(d["SEAICE"][0])
        depth = np.asarray(d["ICEDEPTH"][0])
        snow = np.asarray(d["SNOW"][0])
        snowh = np.asarray(d["SNOWH"][0])
        snowc = np.asarray(d["SNOWC"][0])
    assert seaice.max() > 0.5 and seaice.min() == 0.0
    assert depth.max() > 1.0
    assert np.allclose(snow, 25.0) and np.allclose(snowh, 0.125) and np.allclose(snowc, 1.0)


# ---------------------------------------------------------------------------
# the partial pack: the ice column's own fluxes, the composite surface
# ---------------------------------------------------------------------------


def test_the_ice_skin_reads_back_from_the_composite_and_round_trips():
    f = np.array([1.0, 0.6, 0.5, 0.3], np.float32)
    ice = np.array([250.0, 250.0, 250.0, 250.0], np.float32)
    composite = frozen_surface.blended_surface_temperature(ice, f, np)
    back = frozen_surface.ice_skin_from_composite(composite, f, np)
    np.testing.assert_allclose(back[:3], 250.0, atol=2e-4)        # frozen columns invert the blend
    assert back[3] == composite[3]                                  # below the threshold: returned as given
    # a composite that would put the ice above melting is capped
    hot = frozen_surface.ice_skin_from_composite(np.array([273.0], np.float32), np.array([0.6], np.float32), np)
    assert hot[0] == np.float32(frozen_surface.MELT_POINT_K)
    assert frozen_surface.composite_albedo(np.float32(0.65), np.array([1.0, 0.5], np.float32), np).tolist() == pytest.approx([0.65, 0.5 * 0.65 + 0.5 * 0.08], rel=1e-6)
    assert frozen_surface.composite_emissivity(np.float32(0.98), np.array([1.0, 0.5], np.float32), np).tolist() == pytest.approx([0.98, 0.98], rel=1e-6)


def test_the_saturation_slope_over_ice_is_the_derivative_of_the_saturation():
    t = np.array([230.0, 250.0, 265.0, 272.0], np.float32)
    p = np.full(4, 98000.0, np.float32)
    h = np.float32(0.01)
    finite = (frozen_surface.saturation_specific_humidity_over_ice(t + h, p, np).astype(np.float64)
              - frozen_surface.saturation_specific_humidity_over_ice(t - h, p, np).astype(np.float64)) / (2.0 * float(h))
    slope = frozen_surface.saturation_slope_over_ice(t, p, np)
    np.testing.assert_allclose(slope, finite, rtol=2e-2)
    assert np.all(slope > 0.0)


def _pack_forcing(shape, *, glw, dt_s, layers=None, tref=None, rcc=None, rqc=None):
    dz = frozen_surface.sea_ice_layer_thickness(np.full(shape, 1.5, np.float32), np.zeros(shape, np.float32), np)
    if layers is None:
        layers = frozen_surface.initial_sea_ice_column(
            np.full(shape, 250.0, np.float32), np.full(shape, 1.5, np.float32), np.zeros(shape, np.float32), np)
    forcing = dict(
        layers=layers, dz=dz, snow_depth_m=np.zeros(shape, np.float32),
        bottom_k=np.full(shape, frozen_surface.SEA_ICE_BOTTOM_K, np.float32),
        bottom_extra_resistance=np.zeros(shape, np.float32),
        swdown=np.zeros(shape), albedo=np.full(shape, 0.65), glw=np.full(shape, glw),
        emissivity=np.full(shape, 0.98), hfx=np.full(shape, 20.0), qfx=np.full(shape, 5.0e-6),
        dt_s=dt_s, xp=np,
    )
    if rcc is not None:
        forcing["exchange_heat_w_m2_k"] = np.full(shape, rcc, np.float32)
    if rqc is not None:
        forcing["exchange_moisture_kg_m2_s"] = np.full(shape, rqc, np.float32)
        forcing["psfc_pa"] = np.full(shape, 98000.0, np.float32)
    if tref is not None:
        forcing["skin_reference_k"] = np.full(shape, tref, np.float32)
    return forcing


def test_the_ice_column_receives_its_own_fluxes_not_the_blends():
    """A partial pack's surface layer saw a skin 10 K warmer than the ice
    (the leads) and returned 20 W/m2 upward for the cell.  With the
    exchange slope the ice's own sensible flux is 20 - rcc * 10: at
    rcc = 5 W/m2/K a 30 W/m2 downward flux, so the ice warms where the
    explicit column (charged the cell's 20 W/m2) cooled; the latent flux
    corrects the same way through the saturation over ice; and with no
    slopes the arithmetic is bit-for-bit the explicit form."""
    shape = (1, 2)
    start = _pack_forcing(shape, glw=200.0, dt_s=600.0)["layers"][0]
    explicit = frozen_surface.column_step(**_pack_forcing(shape, glw=200.0, dt_s=600.0))
    corrected = frozen_surface.column_step(**_pack_forcing(shape, glw=200.0, dt_s=600.0, tref=260.0, rcc=5.0))
    assert np.all(explicit[0] < start)
    assert np.all(corrected[0] > explicit[0])
    # The ice's own flux: the cell's plus the slope times the skin difference, at the new skin.
    t_new = float(corrected[0, 0, 0])
    h_ice = 20.0 + 5.0 * (t_new - 260.0)
    assert h_ice < 0.0
    with_moisture = frozen_surface.column_step(**_pack_forcing(shape, glw=200.0, dt_s=600.0, tref=260.0, rcc=5.0, rqc=0.02))
    # q_sat,ice(250) < q_sat,ice(260): the ice sublimates less than the cell's blend, so it warms further.
    assert np.all(with_moisture[0] > corrected[0])
    zero = frozen_surface.column_step(**_pack_forcing(shape, glw=200.0, dt_s=600.0, tref=250.0, rcc=0.0, rqc=0.0))
    np.testing.assert_array_equal(zero, explicit)
    # Without a reference the slope couples the skin to itself: the step's
    # own change is damped (implicit), never a source or sink at rest.
    damped = frozen_surface.column_step(**_pack_forcing(shape, glw=200.0, dt_s=600.0, rcc=50.0))
    assert np.all(damped[0] < start) and np.all(damped[0] > explicit[0])
    with pytest.raises(ValueError, match="surface pressure"):
        forcing = _pack_forcing(shape, glw=200.0, dt_s=600.0, rqc=0.02)
        forcing.pop("psfc_pa")
        frozen_surface.column_step(**forcing)


def test_a_partial_pack_presents_the_composite_surface_and_charges_the_ice_its_own_flux():
    """Two planets differing only in the pack's analysed fraction (1.0 and
    0.6) under the fake surface layer: the fractional cells show the
    radiation WRF's blended albedo and emissivity and the air the blended
    skin, the full cells the ice's own; the ice column under the partial
    cells is warmer than the composite skin it presents would suggest
    because the leads' share of the cell's upward flux stays with the
    water (the ice skin, node 0, of the partial cells is not colder than
    the full cells' at the same forcing)."""
    full = _exchange()
    partial = _exchange()
    frozen_full = _freeze(full)
    frozen_partial = _freeze(partial)
    assert np.array_equal(frozen_full, frozen_partial)
    partial.surface.sea_ice_fraction[...] = np.where(frozen_partial, 0.6, partial.surface.sea_ice_fraction)
    for planet in (full, partial):
        planet.surface.temperature_k[...] = np.where(frozen_full, 255.0, planet.surface.temperature_k)
        planet.surface.soil_temperature_k[...] = seeding.sea_ice_initial_soil_temperature(
            planet.surface.soil_temperature_k, planet.surface.temperature_k, planet.surface.sea_ice_fraction,
            planet.surface.sea_ice_thickness_m, 0.0 * planet.surface.sea_ice_fraction,
        )
    out_full = ArwenCudaColumnSuite(_options(), array_module=np, modules=_fake_modules()).step(full)
    out_partial = ArwenCudaColumnSuite(_options(), array_module=np, modules=_fake_modules()).step(partial)
    frozen = frozen_full
    alb_full = np.asarray(out_full.surface.albedo)[frozen]
    alb_partial = np.asarray(out_partial.surface.albedo)[frozen]
    eps_full = np.asarray(out_full.surface.emissivity)[frozen]
    eps_partial = np.asarray(out_partial.surface.emissivity)[frozen]
    assert np.allclose(alb_full, frozen_surface.SEA_ICE_ALBEDO)
    assert np.allclose(alb_partial, 0.6 * frozen_surface.SEA_ICE_ALBEDO + 0.4 * frozen_surface.OPEN_WATER_ALBEDO, atol=1e-6)
    assert np.allclose(eps_full, frozen_surface.SEA_ICE_EMISSIVITY)
    assert np.allclose(eps_partial, frozen_surface.SEA_ICE_EMISSIVITY)
    ice_full = np.asarray(out_full.surface.soil_temperature_k)[0][frozen]
    ice_partial = np.asarray(out_partial.surface.soil_temperature_k)[0][frozen]
    skin_partial = np.asarray(out_partial.surface.temperature_k)[frozen]
    np.testing.assert_allclose(
        skin_partial, 0.6 * ice_partial + 0.4 * frozen_surface.SEA_ICE_BOTTOM_K, rtol=1e-5)
    # the composite skin the partial cells present is warmer than the ice under it
    assert np.all(skin_partial > ice_partial)
    # the columns outside the pack are bit-for-bit the same on both planets
    assert np.array_equal(
        np.asarray(out_full.surface.temperature_k)[~frozen], np.asarray(out_partial.surface.temperature_k)[~frozen])
    assert np.array_equal(np.asarray(out_full.surface.emissivity)[~frozen], np.asarray(out_partial.surface.emissivity)[~frozen])


def test_a_partial_pack_runs_the_leads_as_a_second_tile_and_composites_the_fluxes():
    """Under a surface layer whose sensible flux follows the skin it is
    handed (hfx = 10 (T_skin - 260)), a planet with a 0.6-fraction pack
    calls the surface layer twice: the ice tile at the ice's own skin
    (the column's top node, not the blended skin the state carries) with
    the ice class's land flag, the leads at the freezing point with the
    water flag and unit moisture availability.  The atmosphere receives
    f hfx_ice + (1 - f) hfx_lead on the partial cells and the ice tile's
    flux elsewhere; the friction velocity is the stress-weighted composite;
    the ice tile's own flux is kept for the frozen column; the leads'
    inout state lives in its own planes.  A full pack and an ice-free
    planet call the surface layer once and keep every plane at its seed."""
    calls = []
    modules = _fake_modules()
    stock = modules["woof.globe.core.sfclay"].sfclay

    def sfclay(*args, **kwargs):
        tsk = np.array(args[7], np.float32, copy=True)
        xland = np.array(args[11], np.float32, copy=True)
        mavail = np.array(args[10], np.float32, copy=True)
        calls.append({"tsk": tsk, "xland": xland, "mavail": mavail, "znt": np.array(args[8], np.float32, copy=True)})
        out = stock(*args, **kwargs)
        out.hfx = (np.float32(10.0) * (tsk - np.float32(260.0))).astype(np.float32)
        out.qfx = (np.float32(1.0e-6) * (tsk - np.float32(250.0))).astype(np.float32)
        out.ust = np.where(xland >= 1.5, np.float32(0.3), np.float32(0.1)).astype(np.float32)
        return out

    modules["woof.globe.core.sfclay"] = SimpleNamespace(sfclay=sfclay)
    exchange = _exchange()
    frozen = _freeze(exchange)
    exchange.surface.sea_ice_fraction[...] = np.where(frozen, 0.6, exchange.surface.sea_ice_fraction)
    exchange.surface.temperature_k[...] = np.where(frozen, 255.0, exchange.surface.temperature_k)
    exchange.surface.soil_temperature_k[...] = seeding.sea_ice_initial_soil_temperature(
        exchange.surface.soil_temperature_k, exchange.surface.temperature_k, exchange.surface.sea_ice_fraction,
        exchange.surface.sea_ice_thickness_m, 0.0 * exchange.surface.sea_ice_fraction,
    )
    ice_skin = np.array(exchange.surface.soil_temperature_k[0], np.float32, copy=True)
    result = ArwenCudaColumnSuite(_options(), array_module=np, modules=modules).step(exchange)
    per_physics_call = 2
    assert len(calls) % per_physics_call == 0 and len(calls) >= per_physics_call
    ice_call, lead_call = calls[0], calls[1]
    # the ice tile: the ice skin, the land flag, the ice class's moisture availability
    assert np.allclose(ice_call["tsk"][frozen], ice_skin[frozen])
    assert np.all(ice_call["xland"][frozen] == 1.0) and np.allclose(ice_call["mavail"][frozen], 0.95)
    # the lead tile: the freezing point, the water flag, unit availability, the water roughness seed
    assert np.all(lead_call["tsk"] == np.float32(frozen_surface.SEA_ICE_BOTTOM_K))
    assert np.all(lead_call["xland"] == 2.0) and np.all(lead_call["mavail"] == 1.0)
    assert np.allclose(lead_call["znt"], 1.0e-4)
    arrays = result.physics_state.arrays
    hfx_ice = 10.0 * (ice_call["tsk"].astype(np.float64) - 260.0)
    hfx_lead = 10.0 * (frozen_surface.SEA_ICE_BOTTOM_K - 260.0)
    # the atmosphere's flux on the partial cells is the area composite; elsewhere the ice tile's
    np.testing.assert_allclose(arrays["hfx"][frozen], 0.6 * hfx_ice[frozen] + 0.4 * hfx_lead, rtol=1e-5)
    assert np.allclose(arrays["ice_hfx"][frozen], hfx_ice[frozen], rtol=1e-5)
    # The lead tile exists on the partial columns and writes nothing
    # elsewhere: outside the pack the plane keeps its seed, which is what
    # makes the checkpoint independent of the band count the physics ran
    # in (the kernel runs on whichever batch carries the column).
    assert np.allclose(arrays["lead_hfx"][frozen], hfx_lead, rtol=1e-5)
    assert np.all(arrays["lead_hfx"][~frozen] == 0.0)
    np.testing.assert_allclose(arrays["ust"][frozen], np.sqrt(0.6 * 0.1 ** 2 + 0.4 * 0.3 ** 2), rtol=1e-5)
    water = water_columns(np.asarray(exchange.surface.land_fraction), sea_ice_fraction=np.asarray(exchange.surface.sea_ice_fraction))
    assert np.all(arrays["ust"][~frozen & ~water] == np.float32(0.1))
    assert np.all(arrays["ust"][water] == np.float32(0.3))
    assert result.physics_state.metadata["lead_tile_calls"] >= 1
    # the columns off the pack carry the ice tile's own values untouched
    off = ~frozen
    assert np.allclose(arrays["hfx"][off], hfx_ice[off], rtol=1e-5)
    # a full pack runs one tile per call and leaves the lead planes at their seed
    calls.clear()
    full = _exchange()
    _freeze(full)
    out_full = ArwenCudaColumnSuite(_options(), array_module=np, modules=modules).step(full)
    assert all(np.all(c["xland"] <= 1.5) or not np.any(c["xland"][frozen] == 2.0) for c in calls)
    assert len(calls) == len([c for c in calls if not np.all(c["tsk"] == np.float32(frozen_surface.SEA_ICE_BOTTOM_K))])
    assert "lead_tile_calls" not in out_full.physics_state.metadata
    assert np.all(out_full.physics_state.arrays["lead_hfx"] == 0.0)
    # an ice-free planet: one tile, nothing kept
    calls.clear()
    bare = ArwenCudaColumnSuite(_options(), array_module=np, modules=modules).step(_exchange())
    assert "lead_tile_calls" not in bare.physics_state.metadata
    assert np.all(bare.physics_state.arrays["ice_hfx"] == 0.0)


def test_the_implicit_skin_balance_closes_with_the_ice_own_flux_in_both_directions():
    """The skin node's discrete energy balance, recomputed in float64 from
    the step's float32 output, closes with the ice's OWN fluxes at the new
    skin (H = H_cell + rcc (T_new - T_ref), E = E_cell + rqc (q_sat(T_old)
    - q_sat(T_ref)) + rqc dq_sat/dT (T_new - T_old)), the emission
    linearised about the old skin and the conduction into the node below,
    with the reference skin above the ice (the leads of a pack) and, the
    other direction, below it; and the ice's own sensible flux is the
    cell's less rcc times the skin difference."""
    shape = (1, 3)
    dt = 600.0
    for tref, glw in ((262.0, 200.0), (245.0, 200.0), (262.0, 320.0)):
        forcing = _pack_forcing(shape, glw=glw, dt_s=dt, tref=tref, rcc=5.0, rqc=0.02)
        layers = forcing["layers"]
        dz = forcing["dz"]
        new = frozen_surface.column_step(**forcing)
        assert np.all(new < frozen_surface.MELT_POINT_K)     # the cap did not act: the balance is the step's own
        k, cv = frozen_surface.column_properties(dz, forcing["snow_depth_m"], np)
        k = k.astype(np.float64)
        cv = cv.astype(np.float64)
        dz64 = dz.astype(np.float64)
        t_old = layers.astype(np.float64)
        t_new = new.astype(np.float64)
        c0 = cv[0] * dz64[0] / dt
        g01 = 1.0 / (0.5 * dz64[0] / k[0] + 0.5 * dz64[1] / k[1])
        eps, sigma = 0.98, frozen_surface.STEFAN_BOLTZMANN_W_M2_K4
        p = forcing["psfc_pa"].astype(np.float32)

        def qs(t):
            return frozen_surface.saturation_specific_humidity_over_ice(
                np.asarray(t, np.float32), p, np).astype(np.float64)

        dqs = frozen_surface.saturation_slope_over_ice(t_old[0].astype(np.float32), p, np).astype(np.float64)
        h_ice = 20.0 + 5.0 * (t_new[0] - tref)
        e_ice = (5.0e-6 + 0.02 * (qs(t_old[0]) - qs(np.full(shape, tref)))
                 + 0.02 * dqs * (t_new[0] - t_old[0]))
        emission = eps * sigma * t_old[0] ** 3 * (4.0 * t_new[0] - 3.0 * t_old[0])
        net = eps * glw - emission - h_ice - frozen_surface.LATENT_HEAT_SUBLIMATION_J_KG * e_ice
        residual = c0 * (t_new[0] - t_old[0]) - net + g01 * (t_new[0] - t_new[1])
        assert np.all(np.abs(residual) < 0.5), (tref, glw, residual)   # W/m2, float32 output on terms of 100 W/m2
        # the ice's own sensible flux: below the cell's when the reference skin is warmer, above when colder
        if tref > t_old[0, 0, 0]:
            assert np.all(h_ice < 20.0)
        else:
            assert np.all(h_ice > 20.0)
