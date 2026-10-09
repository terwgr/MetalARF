"""The open-water skin a cold start holds for the run is a water temperature.

The breakage (GDAS 2026-09-30 12Z, T255 L40, 120 h): an open-water column
holds its initial skin for the whole run (no ocean or lake model), and the
plain bilinear regrid of the analysis skin handed the north basin of Lake
Turkana (4.4473 N, 36.0938 E on the T255 Gaussian grid, land fraction 0.485)
the 15:00 desert ground around the lake, 321.1 K, where the one analysis
water point in its stencil read 300.8 K.  A 48 C saturated lake under the
Turkana jet evaporated about 1.2e-3 kg/m2/s day and night, the booked flux
drained that column's 500 kg/m2 surface reservoir at 4.3 kg/m2/h, and the run
died at hour 117 on "native physics water closure exceeds the explicit
surface reservoir".  The stencil values below are that analysis's own.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from woof.globe.analysis_initial import (
    _global_regridder,
    _TargetGrid,
    analysis_initial_state,
    open_water_skin_temperature,
)
from woof.globe.config import load_config
from woof.globe.runner import build_transform
from woof.globe.statics import water_columns

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")

#: The T255 Gaussian column the run died on, and its static land fraction.
TURKANA_LATITUDE, TURKANA_LONGITUDE, TURKANA_LAND_FRACTION = 4.4473305, 36.09375, 0.485
#: GDAS 2026-09-30 12Z f000 at the four 0.25-degree points of that column's
#: stencil: (latitude, longitude) -> (land-sea mask, skin temperature K).
TURKANA_STENCIL = {
    (4.50, 36.00): (1.0, 321.97324219),
    (4.50, 36.25): (1.0, 327.97324219),
    (4.25, 36.00): (0.0, 300.77324219),
    (4.25, 36.25): (1.0, 323.87324219),
}


def _global_source(stencil, *, land_skin=325.0):
    """A 0.25-degree global analysis plane pair (descending latitude, as
    GDAS stores it): land at ``land_skin`` everywhere except ``stencil``."""
    lat = np.linspace(90.0, -90.0, 721)
    lon = np.arange(1440) * 0.25
    land = np.ones((lat.size, lon.size))
    skin = np.full((lat.size, lon.size), land_skin)
    for (la, lo), (mask, value) in stencil.items():
        j = int(np.argmin(np.abs(lat - la)))
        i = int(np.argmin(np.abs(lon - lo)))
        land[j, i] = mask
        skin[j, i] = value
    return lat, lon, land, skin


def test_the_failing_lake_column_starts_on_its_analysis_water_point_not_the_desert():
    lat, lon, land, skin = _global_source(TURKANA_STENCIL)
    regrid = _global_regridder(
        lat, lon, _TargetGrid(np.asarray([TURKANA_LATITUDE]), np.asarray([TURKANA_LONGITUDE]))
    )
    open_water = water_columns(np.asarray([[TURKANA_LAND_FRACTION]], dtype=np.float32))
    assert open_water.all()
    # The defect: the plain regrid is the afternoon desert, held as a lake.
    assert regrid(skin)[0, 0] > 320.0
    fixed, record = open_water_skin_temperature(skin, land, regrid, open_water)
    assert fixed[0, 0] == pytest.approx(300.77324219, abs=1.0e-9)
    assert record["open_water_columns"] == 1
    assert record["stencil_with_analysis_land"] == 1
    assert record["searched_columns"] == 0
    assert record["largest_changes"][0]["plain_regrid_k"] > 320.0


def test_a_water_column_with_no_analysis_water_in_its_stencil_reads_the_nearest_water():
    # Only the analysis water point two cells south of the stencil exists.
    stencil = {(3.75, 36.0): (0.0, 300.5)}
    lat, lon, land, skin = _global_source(stencil)
    regrid = _global_regridder(
        lat, lon, _TargetGrid(np.asarray([TURKANA_LATITUDE]), np.asarray([TURKANA_LONGITUDE]))
    )
    open_water = np.ones((1, 1), dtype=bool)
    fixed, record = open_water_skin_temperature(skin, land, regrid, open_water)
    assert fixed[0, 0] == pytest.approx(300.5, abs=1.0e-9)
    assert record["searched_columns"] == 1
    assert record["unreached_columns_kept_plain_regrid"] == 0
    assert record["search_passes"] >= 2


def test_land_columns_keep_the_plain_regrid():
    lat, lon, land, skin = _global_source(TURKANA_STENCIL)
    regrid = _global_regridder(
        lat, lon, _TargetGrid(np.asarray([TURKANA_LATITUDE]), np.asarray([TURKANA_LONGITUDE]))
    )
    fixed, record = open_water_skin_temperature(skin, land, regrid, np.zeros((1, 1), dtype=bool))
    assert fixed[0, 0] == regrid(skin)[0, 0]
    assert record["changed_by_more_than_1_k"] == 0


# --------------------------------------------------------------------------
# Through the cold start: no open-water column starts on an analysis land skin.
# --------------------------------------------------------------------------


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
    source_cycle: datetime = datetime(2026, 9, 30, 12)
    valid_time: datetime = datetime(2026, 9, 30, 12)


WATER_SKIN_K = 290.0
LAND_SKIN_K = 330.0


def _hot_coast_frame(nlat=37, nlon=72):
    """A coarse analysis whose land is an afternoon desert (330 K skin) and
    whose water is 290 K everywhere: a run water column whose stencil
    touches the coast would start hotter than any water the analysis has."""
    lat = np.linspace(90.0, -90.0, nlat)
    lon = np.arange(nlon) * (360.0 / nlon)
    levels = np.asarray([10000.0, 30000.0, 50000.0, 70000.0, 85000.0, 100000.0])
    lat2 = np.deg2rad(lat)[:, None] * np.ones((1, nlon))
    shape3 = (levels.size, nlat, nlon)
    temperature = 220.0 + 70.0 * (levels / 100000.0)[:, None, None] * np.cos(lat2)[None] ** 2
    humidity = 0.01 * (levels / 100000.0)[:, None, None] ** 3 * np.ones(shape3)
    u = 20.0 * np.cos(lat2)[None] * np.ones(shape3)
    v = np.zeros(shape3)
    terrain = 1500.0 * np.exp(-((np.rad2deg(lat2) - 30.0) / 15.0) ** 2)
    ps = 101000.0 * np.exp(-terrain / 8000.0)
    # A desert band from 45 to 75 N: on the smoke grid the 41.4 N row's
    # stencil reaches its south edge (land weight 0.28), so those columns
    # are open water with the desert in their stencil, and the 68.8 N row
    # is land.
    land = ((lat2 >= np.deg2rad(42.5)) & (lat2 <= np.deg2rad(77.5))).astype(np.float64)
    skin = np.where(land > 0.5, LAND_SKIN_K, WATER_SKIN_K)
    soil_t = np.where(land[None] > 0.5, 300.0, np.nan) * np.ones((4, 1, 1))
    soil_m = np.where(land[None] > 0.5, 0.3, np.nan) * np.ones((4, 1, 1))
    zero = np.zeros((nlat, nlon))
    open_water = land < 0.5
    fields = {
        "sea_ice_fraction": _Field(zero.copy()),
        "sea_ice_thickness": _Field(zero.copy()),
        "snow_water_equivalent": _Field(np.where(open_water, np.nan, 0.0)),
        "snow_depth": _Field(np.where(open_water, np.nan, 0.0)),
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


def _analysis_cfg(tmp_path):
    text = Path(CONFIG).read_text(encoding="utf-8")
    replaced = text.replace(
        "[initial]\n",
        "[initial]\nmode = \"analysis\"\n"
        "analysis_grib = \"unused.grib\"\n"
        "analysis_mapping = \"gdas-global\"\n",
        1,
    )
    kept = [
        line for line in replaced.splitlines()
        if not any(line.strip().startswith(key) for key in (
            "surface_pressure_pa", "surface_temperature_k", "top_temperature_k",
            "qv_surface", "zonal_wind_m_s", "perturbation_amplitude",
            "zonal_wavenumber", "terrain_amplitude_m",
        ))
    ]
    path = tmp_path / "analysis.toml"
    path.write_text("\n".join(kept) + "\n[statics]\nsource = \"synthetic\"\n", encoding="utf-8")
    return load_config(path)


def test_no_open_water_column_starts_on_an_analysis_land_skin(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    frame = _hot_coast_frame()
    state, _phi, provenance = analysis_initial_state(cfg, transform, frame=frame)
    host = transform.backend.to_numpy
    skin = host(state.surface.temperature_k)
    land = host(state.surface.land_fraction)
    ice = host(state.surface.sea_ice_fraction)
    open_water = water_columns(land, sea_ice_fraction=ice)
    regrid = _global_regridder(frame.latitude, frame.longitude, transform.grid)
    plain = regrid(frame.fields["skin_temperature"].values)
    # The case is a real one: coastal water columns whose plain regrid
    # mixes in the desert, hotter than any water the analysis carries.
    assert np.count_nonzero(open_water & (plain > WATER_SKIN_K + 1.0)) > 0
    assert open_water.any() and (~open_water).any()
    # Every open-water column starts on the analysis's own water.
    assert np.allclose(skin[open_water], WATER_SKIN_K, rtol=0.0, atol=1.0e-3)
    # Land columns keep the analysis skin exactly as regridded.
    assert np.array_equal(skin[~open_water], plain[~open_water].astype(skin.dtype))
    record = provenance["open_water_skin"]
    assert record["open_water_columns"] == int(open_water.sum())
    assert record["stencil_with_analysis_land"] > 0
    assert record["changed_by_more_than_1_k"] > 0
    assert record["largest_cooling_k"] > 1.0
    assert "own water points" in (
        provenance["surface_seeding"]["named_but_seeded_elsewhere"]["skin_temperature"]
    )
