from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest


from woof.globe.analysis_initial import (
    DRY_AIR_GAS_CONSTANT,
    _global_regridder,
    analysis_initial_state,
    resolve_analysis_mapping,
    surface_virtual_temperature,
)
from woof.globe.config import load_config
from woof.globe.constants import GRAVITY_M_S2
from woof.globe.runner import build_transform
from woof.globe.spectral.transform import SphericalHarmonicTransform

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
#: The two spellings that resolve from an INSTALL.  The third,
#: `woof/authorities/rw-wps-...json`, is a checkout-relative path: it named
#: a real file only while this model lived inside the engine's tree, and
#: these gates were written there.  Every shipped config carries the bare id.
MAPPING = "gdas-global"
MAPPING_NAME = "rw-wps-gdas-global-analysis-grib2.mapping.json"


@dataclass
class _Field:
    values: np.ndarray


@dataclass
class _Frame:
    latitude: np.ndarray
    longitude: np.ndarray
    vertical_kind: str
    vertical_values: np.ndarray
    fields: dict
    mapping_sha256: str = "test-mapping"
    input_sha256: str = "test-input"
    source_cycle: datetime = datetime(2026, 8, 30, 18)
    valid_time: datetime = datetime(2026, 8, 30, 18)


def _synthetic_frame(
    nlat=37, nlon=72, descending=True, ridge_sigma_deg=15.0,
    terrain_zonal_wavenumber=0,
):
    lat = np.linspace(90.0, -90.0, nlat) if descending else np.linspace(-90.0, 90.0, nlat)
    lon = np.arange(nlon) * (360.0 / nlon)
    levels = np.asarray([10000.0, 30000.0, 50000.0, 70000.0, 85000.0, 100000.0])
    lat2 = np.deg2rad(lat)[:, None] * np.ones((1, nlon))
    lon2 = np.deg2rad(lon)[None, :] * np.ones((nlat, 1))
    shape3 = (levels.size, nlat, nlon)
    temperature = 220.0 + 70.0 * (levels / 100000.0)[:, None, None] * np.cos(lat2)[None] ** 2
    humidity = 0.01 * (levels / 100000.0)[:, None, None] ** 3 * np.ones(shape3)
    u = 20.0 * np.cos(lat2)[None] * np.ones(shape3)
    v = np.zeros(shape3)
    envelope = np.exp(-((np.rad2deg(lat2) - 30.0) / ridge_sigma_deg) ** 2)
    if terrain_zonal_wavenumber:
        terrain = 700.0 * (
            1.0 + np.cos(terrain_zonal_wavenumber * lon2)
        ) * envelope
    else:
        terrain = 1500.0 * envelope
    ps = 101000.0 * np.exp(-terrain / 8000.0)
    land = (terrain > 100.0).astype(np.float64)
    skin = 288.0 - 30.0 * np.sin(lat2) ** 2
    soil_t = np.where(land[None] > 0.5, skin[None] - 1.0, np.nan) * np.ones((4, 1, 1))
    soil_m = np.where(land[None] > 0.5, 0.3, np.nan) * np.ones((4, 1, 1))
    # The surface seeding's four planes, GDAS-shaped: sea ice on the water
    # poleward of 70 degrees (1.2 m thick), snow on the ridge's land north
    # of 38 degrees (40 kg/m2 over 0.2 m), the snow planes masked (NaN)
    # on open water.
    lat_deg = np.rad2deg(lat2)
    ice = np.where((land < 0.5) & (np.abs(lat_deg) >= 70.0), 1.0, 0.0)
    thickness = np.where(ice >= 0.5, 1.2, 0.0)
    snowy = (land >= 0.5) & (lat_deg >= 38.0)
    open_water = (land < 0.5) & (ice < 0.5)
    swe = np.where(open_water, np.nan, np.where(snowy, 40.0, 0.0))
    snow_depth = np.where(open_water, np.nan, np.where(snowy, 0.2, 0.0))
    fields = {
        "sea_ice_fraction": _Field(ice),
        "sea_ice_thickness": _Field(thickness),
        "snow_water_equivalent": _Field(swe),
        "snow_depth": _Field(snow_depth),
        "air_temperature": _Field(temperature),
        "specific_humidity": _Field(humidity),
        "eastward_wind": _Field(u),
        "northward_wind": _Field(v),
        "surface_pressure": _Field(ps),
        "terrain_height": _Field(terrain),
        "skin_temperature": _Field(skin),
        "land_fraction": _Field(land),
        "soil_temperature": _Field(soil_t),
        "volumetric_soil_moisture": _Field(soil_m),
    }
    return _Frame(lat, lon, "pressure", levels, fields)


def _analysis_cfg(tmp_path, *, statics='[statics]\nsource = "synthetic"\n'):
    """An analysis-mode config on the smoke grid.

    These gates drive a synthetic frame, so the planet is the declared
    synthetic one; ``statics=""`` leaves the table out and takes the
    analysis-mode default (real statics, refused without a cache).
    """
    text = Path(CONFIG).read_text(encoding="utf-8")
    replaced = text.replace(
        "[initial]\n",
        "[initial]\nmode = \"analysis\"\n"
        "analysis_grib = \"unused.grib\"\n"
        f"analysis_mapping = \"{MAPPING}\"\n",
        1,
    )
    kept = []
    for line in replaced.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(key) for key in (
            "surface_pressure_pa", "surface_temperature_k", "top_temperature_k",
            "qv_surface", "zonal_wind_m_s", "perturbation_amplitude",
            "zonal_wavenumber", "terrain_amplitude_m",
        )):
            continue
        kept.append(line)
    path = tmp_path / "analysis.toml"
    path.write_text("\n".join(kept) + "\n" + statics, encoding="utf-8")
    return load_config(path)


# The mapping refusal fires BEFORE the statics refusal this test grades, so
# unlike its neighbours it cannot be answered by the refusal-to-skip rule in
# conftest: `pytest.raises` turns the engine's own sentence into an
# AssertionError of this file's making.  It is marked instead.
def test_an_analysis_run_defaults_to_real_statics_and_refuses_without_a_cache(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path / "case-data"))
    cfg = _analysis_cfg(tmp_path, statics="")
    assert cfg.statics.source == "real" and not cfg.statics.declared
    transform = build_transform(cfg)
    with pytest.raises(FileNotFoundError, match="woof global statics"):
        analysis_initial_state(cfg, transform, frame=_synthetic_frame())


def test_a_declared_synthetic_planet_is_recorded_as_synthetic(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    state, _phi, provenance = analysis_initial_state(
        cfg, transform, frame=_synthetic_frame()
    )
    from woof.globe.statics import (
        SURFACE_STATICS_METADATA_KEY, SYNTHETIC_CONVENTION, frozen_water_columns,
        water_columns,
    )

    statics = provenance["statics"]
    assert statics["source"] == "synthetic" and statics["declared"] is True
    host = transform.backend.to_numpy
    # One vegetation and soil class on land; the runtime's water columns
    # carry the MODIS water class and water soil, as real.exe leaves them;
    # the water columns the analysis freezes over carry the ice class and
    # ice soil (the synthetic frame plants ice poleward of 70 degrees).
    land_fraction = host(state.surface.land_fraction)
    ice = host(state.surface.sea_ice_fraction)
    water = water_columns(land_fraction, sea_ice_fraction=ice)
    frozen = frozen_water_columns(ice)
    assert water.any() and (~water).any() and frozen.any()
    assert not np.any(water & frozen) and np.all(land_fraction[frozen] <= 0.5)
    categories = host(state.surface.landuse_category)
    assert np.all(categories[~water & ~frozen] == 7)
    assert np.all(categories[water] == 17) and np.all(categories[frozen] == 15)
    soil = host(state.surface.soil_category_top)
    assert np.all(soil[~water & ~frozen] == 8)
    assert np.all(soil[water] == 14) and np.all(soil[frozen] == 16)
    assert np.allclose(host(state.surface.leaf_area_index), 3.0)
    assert state.physics_state.metadata[SURFACE_STATICS_METADATA_KEY] == (
        SYNTHETIC_CONVENTION.as_metadata("synthetic")
    )
    assert statics["convention"] == SYNTHETIC_CONVENTION.as_metadata("synthetic")
    assert np.allclose(
        host(state.surface.deep_soil_temperature_k),
        host(state.surface.soil_temperature_k)[-1],
    )


def test_analysis_state_is_complete_finite_and_hydrostatic(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    state, phi_surface, provenance = analysis_initial_state(
        cfg, transform, frame=_synthetic_frame()
    )
    nlev = cfg.vertical.nlev
    shape = (nlev, *transform.spectral_shape)
    assert state.atmosphere.theta.shape == shape
    assert state.atmosphere.log_surface_pressure.shape == transform.spectral_shape
    assert state.atmosphere.qv.shape == shape
    # The condensate species and number moments are grid tracers: real,
    # nonnegative arrays on the Gaussian grid.
    for name in ("qc", "nc", "ng"):
        value = getattr(state.atmosphere, name)
        assert value.shape == (nlev, *transform.grid.shape)
        assert value.dtype.kind == "f"
        assert float(np.min(value)) >= 0.0
    grid_theta = transform.backend.to_numpy(transform.inverse(state.atmosphere.theta))
    assert np.isfinite(grid_theta).all()
    assert float(grid_theta.min()) > 200.0
    ps = np.exp(transform.backend.to_numpy(
        transform.inverse(state.atmosphere.log_surface_pressure)
    ))
    assert np.isfinite(ps).all() and 40000.0 < ps.min() and ps.max() < 110000.0
    # Condensate and moments start at exact spectral zero: the pgrb2 ladder
    # carries no condensate analyses.
    assert np.all(transform.backend.to_numpy(state.atmosphere.qc) == 0.0)
    assert np.all(transform.backend.to_numpy(state.atmosphere.ng) == 0.0)
    assert np.isfinite(transform.backend.to_numpy(phi_surface)).all()
    assert provenance["mode"] == "analysis"
    assert provenance["analysis_levels"] == 6


def test_ocean_masked_soil_is_filled_from_skin_temperature(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    state, _phi, _prov = analysis_initial_state(
        cfg, transform, frame=_synthetic_frame()
    )
    soil = transform.backend.to_numpy(state.surface.soil_temperature_k)
    moisture = transform.backend.to_numpy(state.surface.soil_water_fraction)
    assert np.isfinite(soil).all()
    assert np.isfinite(moisture).all()
    assert soil.shape[0] == 4 and moisture.shape[0] == 4


def test_regional_subset_is_refused_by_name(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    frame = _synthetic_frame()
    frame = _Frame(
        frame.latitude, frame.longitude[:40], frame.vertical_kind,
        frame.vertical_values,
        {name: _Field(np.asarray(field.values)[..., :40])
         for name, field in frame.fields.items()},
    )
    with pytest.raises(ValueError, match="not the\n?.*globe|globe"):
        analysis_initial_state(cfg, transform, frame=frame)


def test_missing_required_field_is_refused_by_name(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    frame = _synthetic_frame()
    del frame.fields["skin_temperature"]
    with pytest.raises(ValueError, match="skin_temperature"):
        analysis_initial_state(cfg, transform, frame=frame)


def test_non_pressure_vertical_kind_is_refused(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    frame = _synthetic_frame()
    frame = _Frame(
        frame.latitude, frame.longitude, "hybrid_sigma_pressure",
        frame.vertical_values, frame.fields,
    )
    with pytest.raises(ValueError, match="pressure"):
        analysis_initial_state(cfg, transform, frame=frame)


def test_surface_pressure_moves_hypsometrically_with_terrain_smoothing(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    # Zonal wavenumber 4 is resolvable on the T3 Gaussian grid (nlon=8) but
    # beyond the T3 truncation, so the spectrally smoothed model terrain
    # must DROP the +-700 m alternation entirely and the surface pressure
    # must move hypsometrically against the dropped relief.
    frame = _synthetic_frame(terrain_zonal_wavenumber=4)
    state, phi_model, _prov = analysis_initial_state(cfg, transform, frame=frame)
    ps_model = np.exp(transform.backend.to_numpy(
        transform.inverse(state.atmosphere.log_surface_pressure)
    ))
    gauss_lat = transform.grid.latitude_deg
    gauss_lon = transform.grid.longitude_deg
    envelope = np.exp(-((gauss_lat - 30.0) / 15.0) ** 2)[:, None]
    terrain_src = 700.0 * (
        1.0 + np.cos(4.0 * np.deg2rad(gauss_lon))[None, :]
    ) * envelope
    ps_src = 101000.0 * np.exp(-terrain_src / 8000.0)
    delta_phi = transform.backend.to_numpy(phi_model) - 9.80665 * terrain_src
    strong = np.abs(delta_phi) > 0.5 * np.abs(delta_phi).max()
    assert np.abs(delta_phi).max() > 2000.0
    agreement = np.mean(
        np.sign(ps_model - ps_src)[strong] == -np.sign(delta_phi)[strong]
    )
    assert agreement > 0.9


def test_surface_pressure_reduction_uses_the_air_at_the_source_surface():
    # A plateau column: the analysis surface sits at 600 hPa, the profile
    # above it is a 6.5 K/km atmosphere, and the isobaric levels BELOW the
    # surface carry the analysis's own below-ground continuation - 28 K
    # warmer at 1000 hPa than the air at the surface (the audit's Tibetan
    # column: T1000 312.2 K vs Tsfc 284.3 K).
    levels = np.asarray([30000.0, 50000.0, 60000.0, 70000.0, 85000.0, 100000.0])
    ln_source = np.log(levels)
    t_surface = 284.3
    scale_height_m = DRY_AIR_GAS_CONSTANT * t_surface / 9.80665
    heights = -scale_height_m * np.log(levels / 60000.0)
    temperature = (t_surface - 0.0065 * heights)[:, None, None]
    humidity = np.clip(0.008 * (levels / 60000.0) ** 2, 0.0, 0.02)[:, None, None]
    ps_src = np.full((1, 1), 60000.0)
    virtual = surface_virtual_temperature(temperature, humidity, ln_source, ps_src)
    expected = t_surface * (1.0 + 0.608 * 0.008)
    assert virtual.shape == (1, 1)
    assert virtual[0, 0] == pytest.approx(expected, abs=1.0e-9)
    # The bottom source level is 27.9 K warmer: the value the reduction used
    # before the fix, over a layer that lies between 600 and ~800 hPa.
    bottom = temperature[-1, 0, 0] * (1.0 + 0.608 * humidity[-1, 0, 0])
    assert bottom - virtual[0, 0] > 25.0
    # Smoothing the plateau down 2000 m (the audit's dz) reduces ps by
    # 1780 Pa less with the surface air than with the underground level.
    dphi = 9.80665 * 2000.0
    ps_fix = 60000.0 * np.exp(dphi / (DRY_AIR_GAS_CONSTANT * virtual[0, 0]))
    ps_old = 60000.0 * np.exp(dphi / (DRY_AIR_GAS_CONSTANT * bottom))
    assert ps_fix - ps_old > 1500.0
    # A sea-level column below the bottom source level continues the
    # bottom layer's lapse rate rather than holding 1000 hPa's value.
    ps_low = np.full((1, 1), 101300.0)
    low = surface_virtual_temperature(temperature, humidity, ln_source, ps_low)
    slope = (temperature[-1, 0, 0] - temperature[-2, 0, 0]) / (ln_source[-1] - ln_source[-2])
    t_low = temperature[-1, 0, 0] + slope * (np.log(101300.0) - ln_source[-1])
    assert low[0, 0] == pytest.approx(t_low * (1.0 + 0.608 * humidity[-1, 0, 0]), abs=1.0e-9)


def test_analysis_state_reduces_surface_pressure_with_surface_air(tmp_path):
    # End to end through the pipeline: the receipted ps adjustment equals
    # the hypsometric reduction computed with the surface-interpolated Tv
    # on the regridded source fields, not with the bottom source level.
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    frame = _synthetic_frame(terrain_zonal_wavenumber=4)
    # Make the below-ground levels hot, the way an isobaric analysis
    # continues under a plateau, so the two choices separate clearly.
    temperature = np.asarray(frame.fields["air_temperature"].values).copy()
    temperature[-1] += 25.0
    frame.fields["air_temperature"] = _Field(temperature)
    _state, phi_model, provenance = analysis_initial_state(cfg, transform, frame=frame)
    assert provenance["surface_pressure_reduction"].startswith("hypsometric-virtual-temperature-interpolated")
    regrid = _global_regridder(frame.latitude, frame.longitude, transform.grid)
    ps_src = regrid(frame.fields["surface_pressure"].values)
    phi_src = GRAVITY_M_S2 * regrid(frame.fields["terrain_height"].values)
    ln_source = np.log(np.asarray(frame.vertical_values))
    t_src = regrid(temperature)
    q_src = np.clip(regrid(frame.fields["specific_humidity"].values), 0.0, None)
    dphi = phi_src - transform.backend.to_numpy(phi_model)
    with_surface_air = ps_src * np.exp(
        dphi / (DRY_AIR_GAS_CONSTANT * surface_virtual_temperature(t_src, q_src, ln_source, ps_src))
    ) - ps_src
    with_bottom_level = ps_src * np.exp(
        dphi / (DRY_AIR_GAS_CONSTANT * t_src[-1] * (1.0 + 0.608 * q_src[-1]))
    ) - ps_src
    assert provenance["surface_pressure_adjustment_pa"]["max"] == pytest.approx(
        float(with_surface_air.max()), rel=1.0e-9
    )
    assert provenance["surface_pressure_adjustment_pa"]["min"] == pytest.approx(
        float(with_surface_air.min()), rel=1.0e-9
    )
    assert abs(float(with_surface_air.max()) - float(with_bottom_level.max())) > 50.0


def test_analysis_config_refuses_analytic_shape_keys(tmp_path):
    text = Path(CONFIG).read_text(encoding="utf-8").replace(
        "[initial]\n", "[initial]\nmode = \"analysis\"\nanalysis_grib = \"x.grib\"\n", 1
    )
    path = tmp_path / "bad.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys in \\[initial\\]"):
        load_config(path)


def test_analysis_config_requires_grib(tmp_path):
    cfg_text = Path(CONFIG).read_text(encoding="utf-8")
    kept = [
        line for line in cfg_text.splitlines()
        if not any(line.strip().startswith(key) for key in (
            "surface_pressure_pa", "surface_temperature_k", "top_temperature_k",
            "qv_surface", "zonal_wind_m_s", "perturbation_amplitude",
            "zonal_wavenumber", "terrain_amplitude_m",
        ))
    ]
    text = "\n".join(kept).replace("[initial]", "[initial]\nmode = \"analysis\"")
    path = tmp_path / "bad.toml"
    path.write_text(text + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="analysis_grib"):
        load_config(path)


def test_mapping_id_resolution_refuses_ambiguity():
    with pytest.raises(ValueError, match="exactly one"):
        resolve_analysis_mapping("gdas")
    assert resolve_analysis_mapping(MAPPING).name == MAPPING_NAME


def test_the_bare_id_and_the_file_name_resolve_to_one_mapping(
    tmp_path, monkeypatch
):
    """One mapping file, two spellings, both resolving from an install.

    Configurations used to name the authority by its checkout-relative path,
    which resolves to nothing inside an installed wheel -- there is no
    ``woof/authorities/...`` under the working directory of a user who pip
    installed, and the resolver says so by name rather than substituting a
    file the caller did not ask for.  The two spellings that DO resolve are
    the bare id a config carries and the file name a reader module opens,
    and both go through one resolver, so they cannot answer differently.

    Both must name the same file, or the two would be different runs.  And
    because the node chains launch these configs from a checkout root, the
    bare id is exercised from there too, next to a directory whose name
    could shadow it.
    """

    packaged = resolve_analysis_mapping(MAPPING)
    by_name = resolve_analysis_mapping(MAPPING_NAME)
    assert packaged.resolve() == by_name.resolve()
    assert packaged.read_bytes() == by_name.read_bytes()
    # Packaged, not working-directory-relative: absolute, and under an
    # authorities directory that ships beside a module.
    assert packaged.is_absolute()
    assert packaged.parent.name == "authorities"

    checkout_root = Path(__file__).resolve().parents[1]
    for cwd in (checkout_root, tmp_path):
        monkeypatch.chdir(cwd)
        assert resolve_analysis_mapping(MAPPING).resolve() == packaged.resolve(), cwd

    # The checkout spelling is refused BY NAME, never substituted: a caller
    # who typed a path meant that file, and quietly answering with another is
    # how a run gets initialized from a source nobody named.
    with pytest.raises(FileNotFoundError):
        resolve_analysis_mapping("woof/authorities/" + MAPPING_NAME)


def test_every_shipped_global_analysis_config_names_the_bare_id():
    """A path spelling in a shipped config is an install-time refusal."""

    for path in (
        str(_shipped_configs() / "arwen_global_t255_quickstart.toml"),
        str(_shipped_configs() / "arwen_global_gdas_t63_48h.toml"),
        str(_shipped_configs() / "arwen_global_gdas_t255_native_24h.toml"),
        str(_shipped_configs() / "arwen_global_gdas_t533_24h.toml"),
    ):
        cfg = load_config(path)
        assert cfg.initial_mode == "analysis", path
        assert cfg.analysis_mapping == "gdas-global", path
        assert resolve_analysis_mapping(cfg.analysis_mapping).exists(), path
        # And the GRIB each names is relative, so the config is runnable
        # from a working directory the fetch door wrote into.
        assert not Path(cfg.analysis_grib).is_absolute(), path
