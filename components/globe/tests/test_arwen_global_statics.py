"""Static surface fields for the global model: the rows grid, the
WPS_GEOG build on a Gaussian grid through the Rust static-field builder,
the once-per-truncation cache, the refusals, and the resolved surface
state the land surface is driven with."""
from __future__ import annotations

from conftest import (  # noqa: E402
    STATIC_FIELDS_SPEAKS_ROWS,
    requires_rows_static_fields,
)
from woof.globe.configs_dir import config_root as _shipped_configs
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pytest

from woof.globe import statics
from woof.globe.config import load_config
from woof.globe.statics import StaticsOptions
from woof.globe.spectral.grid import GaussianGrid
from woof.static import rust_bridge
from woof.globe.statics_rows import RowsGrid

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
#: The bare id every shipped config carries.  The checkout-relative path
#: this line used to hold named a file only inside the engine's tree, and
#: the test that reads it was skipped everywhere else until 0.1.2, by a
#: rows-kind probe that could not call the 2.8 bridge.
MAPPING = "gdas-global"
VALID_TIME = datetime(2026, 8, 30, 18)

# The MODIS 21-class land-use metadata the real index declares.
ISWATER, ISLAKE, ISICE, ISURBAN = 17, 21, 15, 13
TILE = 90


def _needs_builder():
    reason = rust_bridge.unavailable_reason()
    if reason is not None:
        pytest.skip(f"Rust static-fields library not loadable: {reason}")


# ---------------------------------------------------------------------------
# a synthetic one-degree WPS_GEOG archive (the nine datasets the build reads)
# ---------------------------------------------------------------------------

def _land(lat, lon):
    return (np.abs(lat) < 60.0) & (
        ((lon > -140.0) & (lon < -40.0)) | ((lon > 0.0) & (lon < 130.0)))


def _lake(lat, lon):
    return (lat > 40.0) & (lat < 45.0) & (lon > 20.0) & (lon < 25.0)


def _write_index(directory: Path, **kv) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in kv.items():
        if isinstance(value, str) and " " in value:
            value = f'"{value}"'
        lines.append(f"{key}={value}")
    (directory / "index").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return kv


def _write_tiles(directory: Path, data: np.ndarray, kv: dict) -> None:
    nz, nyg, nxg = data.shape
    ws = int(kv["wordsize"])
    signed = str(kv.get("signed", "no")).lower() in ("yes", "true")
    base = {1: "i1", 2: "i2"}[ws]
    if not signed:
        base = "u" + base[1:]
    dtype = np.dtype(">" + base) if ws > 1 else np.dtype(base)
    for ys in range(1, nyg + 1, TILE):
        for xs in range(1, nxg + 1, TILE):
            tile = data[:, ys - 1:ys - 1 + TILE, xs - 1:xs - 1 + TILE]
            name = f"{xs:05d}-{xs + TILE - 1:05d}.{ys:05d}-{ys + TILE - 1:05d}"
            np.ascontiguousarray(tile).astype(dtype).tofile(directory / name)


def _geography(*, north_first: bool):
    """(lat, lon, land, lake) meshes on the one-degree grid, rows in the
    dataset's own row order."""
    lat = np.arange(-89.5, 90.0, 1.0)
    if north_first:
        lat = lat[::-1]
    lon = np.arange(-179.5, 180.0, 1.0)
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    return lat2, lon2, _land(lat2, lon2), _lake(lat2, lon2)


def write_geog_archive(root: Path) -> Path:
    common = dict(projection="regular_ll", dx=1.0, dy=1.0, known_x=1.0,
                  known_y=1.0, known_lat=-89.5, known_lon=-179.5,
                  tile_x=TILE, tile_y=TILE)
    top_down = dict(common, dy=-1.0, known_lat=89.5)
    lat, lon, land, lake = _geography(north_first=False)
    i = np.arange(360)[None, :] + 1
    j = np.arange(180)[:, None] + 1

    kv = _write_index(root / "topo_gmted2010_30s", type="continuous",
                      signed="yes", wordsize=2, tile_z=1, units="meters MSL",
                      **common)
    terrain = np.where(land, 300 + 10 * ((i + j) % 50), 0)
    _write_tiles(root / "topo_gmted2010_30s", terrain[None], kv)

    kv = _write_index(root / "modis_landuse_20class_30s_with_lakes",
                      type="categorical", category_min=1, category_max=21,
                      wordsize=1, tile_z=1, mminlu="MODIFIED_IGBP_MODIS_NOAH",
                      iswater=ISWATER, islake=ISLAKE, isice=ISICE,
                      isurban=ISURBAN, **common)
    landuse = np.where(land, 1 + (i + j) % 14, ISWATER)
    landuse = np.where(lake, ISLAKE, landuse)
    _write_tiles(root / "modis_landuse_20class_30s_with_lakes", landuse[None], kv)

    for name in ("soiltype_top_30s", "soiltype_bot_30s"):
        kv = _write_index(root / name, type="categorical", category_min=1,
                          category_max=16, wordsize=1, tile_z=1, **common)
        soil = np.where(land & ~lake, 1 + (3 * i + j) % 12, 14)
        _write_tiles(root / name, soil[None], kv)

    kv = _write_index(root / "greenfrac_fpar_modis", type="continuous",
                      wordsize=1, tile_z_start=1, tile_z_end=12,
                      scale_factor=0.01, **common)
    planes = np.arange(12)[:, None, None]
    _write_tiles(root / "greenfrac_fpar_modis",
                 np.where(land[None], 5 * (planes + 1), 0), kv)

    kv = _write_index(root / "lai_modis_10m", type="continuous", wordsize=1,
                      tile_z_start=1, tile_z_end=12, scale_factor=0.1,
                      **common)
    _write_tiles(root / "lai_modis_10m",
                 np.where(land[None], 5 * (planes + 1), 0), kv)

    lat_n, lon_n, land_n, _ = _geography(north_first=True)
    kv = _write_index(root / "albedo_modis", type="continuous", wordsize=1,
                      tile_z_start=1, tile_z_end=12, **top_down)
    _write_tiles(root / "albedo_modis",
                 np.where(land_n[None], 15 + planes, 8), kv)

    kv = _write_index(root / "maxsnowalb_modis", type="continuous",
                      wordsize=1, tile_z=1, **top_down)
    _write_tiles(root / "maxsnowalb_modis", np.where(land_n, 60, 0)[None], kv)

    kv = _write_index(root / "soiltemp_1deg", type="continuous",
                      signed="yes", wordsize=2, tile_z=1, scale_factor=0.01,
                      missing_value=0, **common)
    soiltemp = np.where(land, 28000 + 100 * (j % 5), 0)
    _write_tiles(root / "soiltemp_1deg", soiltemp[None], kv)
    return root


@pytest.fixture(scope="module")
def geog_root(tmp_path_factory) -> Path:
    return write_geog_archive(tmp_path_factory.mktemp("WPS_GEOG"))


@pytest.fixture(scope="module")
def t21_grid() -> GaussianGrid:
    return GaussianGrid.create(21, dealias_factor=1.5)


@pytest.fixture(scope="module")
def built(geog_root, t21_grid):
    _needs_builder()
    # The build crosses the seam as a `rows` grid, which a published engine's
    # static-fields library does not know.  Skipping HERE rather than on each
    # consumer keeps the reason on the one line that actually needs the door.
    if not STATIC_FIELDS_SPEAKS_ROWS:
        pytest.skip(
            "the static_fields library the installed engine loads does not "
            "know the `rows` grid kind a Gaussian grid crosses the seam as; "
            "the engine's own bundled library carries it from woof 2.8.0, "
            "so a library staged from an older build is shadowing it")
    options = StaticsOptions(source="real", geog_root=str(geog_root))
    return statics.build_statics(t21_grid, options, sector_degrees=90.0)


# ---------------------------------------------------------------------------
# the rows grid
# ---------------------------------------------------------------------------

def test_rows_grid_describes_the_gaussian_grid_to_the_crate(t21_grid):
    ring = RowsGrid.from_gaussian(t21_grid)
    assert (ring.nlat, ring.nlon) == t21_grid.shape
    assert ring.closes
    spec = ring._rust_spec()
    assert spec["kind"] == "rows"
    assert spec["e_we"] == t21_grid.nlon + 1 and spec["e_sn"] == t21_grid.nlat + 1
    assert spec["lat_deg"] == [float(v) for v in t21_grid.latitude_deg]
    assert spec["lon0_deg"] == 0.0 and spec["dlon_deg"] == pytest.approx(360.0 / t21_grid.nlon)
    assert ring.dx >= 1000.0
    sectors = ring.sectors(90.0)
    assert sum(piece.nlon for piece in sectors) == ring.nlon
    assert [piece._translation_offset[0] for piece in sectors][0] == 0
    assert all(piece._translation_reference is ring for piece in sectors)
    with pytest.raises(ValueError, match="outside the ring"):
        ring.sector(ring.nlon - 1, 2)
    with pytest.raises(ValueError, match="strictly ascending"):
        RowsGrid([0.0, -1.0], 0.0, 1.0, 360)


# ---------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------

def _column(grid, lat_deg, lon_deg):
    j = int(np.argmin(np.abs(grid.latitude_deg - lat_deg)))
    i = int(np.argmin(np.abs(((grid.longitude_deg - lon_deg + 180.0) % 360.0) - 180.0)))
    return j, i


def test_build_on_the_gaussian_grid_reads_the_archive_through_the_crate(built, t21_grid):
    fields, provenance = built
    for name in statics.CACHED_FIELDS:
        assert fields[name].shape[-2:] == t21_grid.shape, name
        assert fields[name].dtype == np.float32
    assert provenance["schema"] == statics.STATICS_SCHEMA
    # `num_land_cat` is the engine's reading of the index's category_max,
    # recorded by the 2.8 static builder beside the four class numbers.
    assert provenance["landuse"] == {
        "mminlu": "MODIFIED_IGBP_MODIS_NOAH", "iswater": ISWATER,
        "islake": ISLAKE, "isice": ISICE, "isurban": ISURBAN,
        "num_land_cat": 21,
    }
    assert len(provenance["sectors"]) == 4
    assert set(provenance["coverage"]) == set(statics.GEOG_ROLES)
    assert all(receipt["status"] == "PASS"
               for receipts in provenance["coverage"].values() for receipt in receipts)
    assert provenance["builder"]["library"].endswith(rust_bridge.library_names()[0])
    # Deep inside the land box: land, a land category, a land soil class.
    j, i = _column(t21_grid, 0.0, 60.0)
    assert fields["LANDMASK"][j, i] == 1.0
    assert 1 <= fields["LU_INDEX"][j, i] <= 14
    assert 1 <= fields["SCT_DOM"][j, i] <= 12
    assert fields["GREENFRAC"][:, j, i] == pytest.approx(0.05 * np.arange(1, 13), abs=1e-6)
    assert fields["LAI12M"][:, j, i] == pytest.approx(0.5 * np.arange(1, 13), abs=1e-6)
    assert fields["ALBEDO12M"][:, j, i] == pytest.approx(15.0 + np.arange(12), abs=1e-6)
    assert fields["SNOALB"][j, i] == pytest.approx(60.0)
    assert 279.0 < fields["SOILTEMP"][j, i] < 285.0
    assert fields["HGT_M"][j, i] > 250.0
    # Mid-ocean: water, water category, water soil, masked climatologies.
    j, i = _column(t21_grid, 0.0, -160.0)
    assert fields["LANDMASK"][j, i] == 0.0
    assert fields["LU_INDEX"][j, i] == ISWATER
    assert fields["SCT_DOM"][j, i] == 14
    assert np.all(fields["GREENFRAC"][:, j, i] == 0.0)
    assert np.all(fields["ALBEDO12M"][:, j, i] == 8.0)
    assert fields["SOILTEMP"][j, i] == 0.0
    # The lake is the dominant water type where it fills the cell.
    j, i = _column(t21_grid, 42.5, 22.5)
    assert fields["LANDUSEF"][ISLAKE - 1, j, i] > 0.0
    # The whole-globe land fraction is the archive's, not a formula.
    water = fields["LANDUSEF"][ISWATER - 1] + fields["LANDUSEF"][ISLAKE - 1]
    weights = t21_grid.quadrature_weights[:, None] / 2.0 / t21_grid.nlon
    lat, lon, land, _ = _geography(north_first=False)
    expected = float(np.mean(np.cos(np.deg2rad(lat)) * land) / np.mean(np.cos(np.deg2rad(lat))))
    assert float(np.sum(weights * (1.0 - water))) == pytest.approx(expected, abs=0.02)


@requires_rows_static_fields
def test_sector_width_does_not_change_the_fields(built, geog_root, t21_grid):
    fields, _ = built
    options = StaticsOptions(source="real", geog_root=str(geog_root))
    again, _ = statics.build_statics(t21_grid, options, sector_degrees=180.0)
    for name in statics.CACHED_FIELDS:
        if name == "HGT_M":
            # The builder's terrain takes geogrid's smoother-desmoother
            # pass per sector, so sector-edge columns differ; the model
            # keeps its own spectral terrain and the deep-soil correction
            # is recomputed against it, so HGT_M is carried for the
            # record only.
            continue
        assert np.array_equal(fields[name], again[name]), name


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------

def test_cache_round_trip_verifies_every_hash(built, t21_grid, tmp_path):
    fields, provenance = built
    npz, sidecar = statics.write_cache(tmp_path / "cache" / "t21.npz", fields, provenance)
    read, document = statics.read_cache(npz, t21_grid)
    for name, value in fields.items():
        assert np.array_equal(read[name], value), name
    assert document["self_sha256"] and document["npz_sha256"]
    assert document["arrays"]["LU_INDEX"]["sha256"] == provenance["arrays"]["LU_INDEX"]["sha256"]
    with pytest.raises(FileExistsError, match="--overwrite"):
        statics.write_cache(npz, fields, provenance)
    statics.write_cache(npz, fields, provenance, overwrite=True)

    # a different grid is refused by name
    other = GaussianGrid.create(15, dealias_factor=1.5)
    with pytest.raises(ValueError, match="this run's grid"):
        statics.read_cache(npz, other)

    # a tampered array is refused
    with np.load(npz, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    values["LU_INDEX"][0, 0] += 1.0
    with npz.open("wb") as stream:
        np.savez_compressed(stream, **values)
    with pytest.raises(ValueError, match="npz_sha256"):
        statics.read_cache(npz, t21_grid)

    # a tampered sidecar is refused
    statics.write_cache(npz, fields, provenance, overwrite=True)
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    document["geog_root"] = "elsewhere"
    sidecar.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash"):
        statics.read_cache(npz, t21_grid)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_missing_archive_refuses_with_the_fetch_remedy(tmp_path, monkeypatch):
    monkeypatch.delenv("GPUWM_WPS_GEOG", raising=False)
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path / "case-data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    with pytest.raises(FileNotFoundError, match="woof fetch-geog"):
        statics.resolve_geog_root(StaticsOptions(source="real"))
    (tmp_path / "partial").mkdir()
    with pytest.raises(FileNotFoundError, match="lacks 9 dataset"):
        statics.resolve_geog_root(
            StaticsOptions(source="real", geog_root=str(tmp_path / "partial")))


def test_missing_cache_refuses_naming_the_statics_door(tmp_path, t21_grid):
    options = StaticsOptions(source="real", cache_dir=str(tmp_path / "none"))
    with pytest.raises(FileNotFoundError, match="woof global statics"):
        statics.load_statics(options, t21_grid)


def test_python_fallback_is_refused_for_the_rows_grid(monkeypatch):
    monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
    with pytest.raises(RuntimeError, match="no body for the global model"):
        statics.require_rust_builder()


# ---------------------------------------------------------------------------
# resolving to the run's surface state
# ---------------------------------------------------------------------------

def test_resolved_statics_follow_the_regional_rulebook(built, t21_grid):
    from woof.globe.core.landuse import load_landuse_table
    from woof.static.build import monthly_interp_to_date

    fields, provenance = built
    shape = t21_grid.shape
    terrain = np.where(fields["LANDMASK"] > 0.5, 400.0, 0.0)
    soil = np.full((4, *shape), 290.0)
    skin = np.full(shape, 291.0)
    resolved, detail = statics.resolve_surface_statics(
        fields, provenance, valid_time=VALID_TIME,
        latitude_deg=t21_grid.latitude_deg, terrain_height_m=terrain,
        soil_temperature_k=soil, skin_temperature_k=skin,
    )
    assert set(resolved) == {"land_fraction", *statics.SURFACE_STATIC_FIELDS}
    for name, value in resolved.items():
        assert value.shape == shape and np.isfinite(value).all(), name
    table = load_landuse_table("MODIFIED_IGBP_MODIS_NOAH")
    water_row = table.values[0, ISWATER - 1]

    j, i = _column(t21_grid, -30.0, 60.0)  # southern-hemisphere land
    assert resolved["land_fraction"][j, i] == 1.0
    assert 1 <= resolved["landuse_category"][j, i] <= 14
    assert 1 <= resolved["soil_category_top"][j, i] <= 12
    assert resolved["vegetation_fraction"][j, i] == pytest.approx(
        monthly_interp_to_date(0.05 * np.arange(1, 13), VALID_TIME), abs=1e-6)
    assert resolved["vegetation_fraction_min"][j, i] == pytest.approx(0.05, abs=1e-6)
    assert resolved["vegetation_fraction_max"][j, i] == pytest.approx(0.60, abs=1e-6)
    assert resolved["leaf_area_index"][j, i] == pytest.approx(
        monthly_interp_to_date(0.5 * np.arange(1, 13), VALID_TIME), abs=1e-6)
    assert resolved["background_albedo"][j, i] == pytest.approx(
        monthly_interp_to_date(0.15 + 0.01 * np.arange(12), VALID_TIME), abs=1e-6)
    assert resolved["snow_albedo"][j, i] == pytest.approx(0.60, abs=1e-6)
    assert resolved["deep_soil_temperature_k"][j, i] == pytest.approx(
        fields["SOILTEMP"][j, i] - 0.0065 * 400.0, abs=1e-4)
    category = int(resolved["landuse_category"][j, i])
    land_row = table.values[detail["season_southern"] - 1, category - 1]
    assert resolved["albedo"][j, i] == pytest.approx(land_row[0] / 100.0, abs=1e-6)
    assert resolved["emissivity"][j, i] == pytest.approx(land_row[2], abs=1e-6)
    assert resolved["roughness_m"][j, i] == pytest.approx(land_row[3] / 100.0, abs=1e-7)

    j, i = _column(t21_grid, 0.0, -160.0)  # open water
    assert resolved["land_fraction"][j, i] == 0.0
    assert resolved["landuse_category"][j, i] == ISWATER
    assert resolved["soil_category_top"][j, i] == 14
    assert resolved["soil_category_bottom"][j, i] == 14
    assert resolved["deep_soil_temperature_k"][j, i] == 290.0
    assert resolved["albedo"][j, i] == pytest.approx(water_row[0] / 100.0)
    assert resolved["emissivity"][j, i] == pytest.approx(water_row[2])
    assert resolved["roughness_m"][j, i] == pytest.approx(water_row[3] / 100.0)

    j, i = _column(t21_grid, 42.5, 22.5)  # the lake folds to water
    if fields["LANDMASK"][j, i] == 0.0:
        assert resolved["landuse_category"][j, i] == ISWATER
    # the northern and southern seasons differ in August
    assert detail["season_northern"] != detail["season_southern"]
    assert (resolved["landuse_category"] <= 20).all()
    # The land/water split the runtime derives from the land fraction is
    # the split the categories carry, on every column and in either
    # precision; a coastal fraction never sits in the float32 dead band.
    lf = resolved["land_fraction"]
    water = statics.water_columns(lf)
    assert np.array_equal(resolved["landuse_category"] == ISWATER, water)
    assert np.array_equal(resolved["soil_category_top"] == 14, water)
    assert np.array_equal(statics.water_columns(lf.astype(np.float32)), water)
    assert not np.any((lf > 0.5) & (lf < 0.5 + statics.LAND_FRACTION_MARGIN))
    assert (0.0 < lf[(lf > 0.0) & (lf < 1.0)]).any()  # fractional coasts exist
    assert np.array_equal(water, fields["LANDMASK"] < 0.5)
    assert detail["convention"] == statics.real_convention(provenance).as_metadata("real")
    assert detail["convention"]["landuse_dataset"] == "MODIFIED_IGBP_MODIS_NOAH"
    assert detail["convention"]["water_category"] == ISWATER


def test_a_cache_whose_mask_and_categories_disagree_is_refused(built, t21_grid):
    fields, provenance = built
    shape = t21_grid.shape
    tampered = {name: np.array(value, copy=True) for name, value in fields.items()}
    j, i = _column(t21_grid, -30.0, 60.0)  # land by LANDMASK
    assert tampered["LANDMASK"][j, i] == 1.0
    tampered["LU_INDEX"][j, i] = ISWATER
    with pytest.raises(ValueError, match="1 column\\(s\\) disagree between the static land mask"):
        statics.resolve_surface_statics(
            tampered, provenance, valid_time=VALID_TIME,
            latitude_deg=t21_grid.latitude_deg,
            terrain_height_m=np.zeros(shape), soil_temperature_k=np.full((4, *shape), 290.0),
            skin_temperature_k=np.full(shape, 291.0),
        )


def test_synthetic_planet_is_the_former_constant_planet_on_land():
    # Two boundary columns: 0.5 is water by the runtime's rule and a
    # fraction 1e-9 above it rounds to 1.5 in float32 and is water too, so
    # the planet caps it at 0.5 and puts the water classes on it; a
    # fraction two float32 ulps above one half is land and untouched.
    land = np.array([[0.0, 0.3, 0.5], [0.7, 1.0, 0.5 + 1.0e-9],
                     [0.5 + 2.0 ** -23, 0.51, 0.49]])
    soil = np.stack([np.full((3, 3), 280.0), np.full((3, 3), 281.0)])
    planet = statics.synthetic_surface_statics(land, soil)
    assert set(planet) == {"land_fraction", *statics.SURFACE_STATIC_FIELDS}
    water = np.array([[True, True, True], [False, False, True],
                      [False, False, True]])
    assert np.array_equal(statics.water_columns(planet["land_fraction"]), water)
    assert np.array_equal(planet["land_fraction"][1, 2], 0.5)
    untouched = ~(np.arange(9).reshape(3, 3) == 5)
    assert np.array_equal(planet["land_fraction"][untouched], land[untouched])
    assert np.all(planet["landuse_category"][~water] == 7)
    assert np.all(planet["landuse_category"][water] == 17)
    for name in ("soil_category_top", "soil_category_bottom"):
        assert np.all(planet[name][~water] == 8)
        assert np.all(planet[name][water] == 14)
    assert np.array_equal(planet["vegetation_fraction"], planet["land_fraction"])
    assert np.all(planet["vegetation_fraction_min"] == 0.0)
    assert np.all(planet["vegetation_fraction_max"] == 1.0)
    assert np.all(planet["leaf_area_index"] == 3.0)
    assert np.all(planet["snow_albedo"] == 0.6)
    assert np.all(planet["emissivity"] == 0.96)
    assert np.array_equal(planet["deep_soil_temperature_k"], soil[-1])
    assert np.allclose(planet["albedo"], 0.08 + 0.12 * planet["land_fraction"])
    assert np.allclose(planet["background_albedo"], planet["albedo"])
    assert np.allclose(planet["roughness_m"], 1.0e-4 + 0.08 * planet["land_fraction"])
    assert statics.synthetic_provenance(StaticsOptions())["convention"] == {
        "source": "synthetic", "landuse_dataset": "MODIFIED_IGBP_MODIS_NOAH",
        "soil_dataset": "STAS", "water_category": 17, "lake_category": 21,
        "ice_category": 15, "urban_category": 13, "water_soil_category": 14,
        "ice_soil_category": 16, "sea_ice_roughness_m": statics.SEA_ICE_ROUGHNESS_M,
    }


def test_water_columns_is_the_runtime_rule_in_both_precisions():
    """statics.water_columns and native_runtime._xland >= 1.5 are one
    rule, and consistent_land_fraction puts every column on the same side
    of it whether the state is float32 or float64."""
    from types import SimpleNamespace

    from woof.globe.physics.native_runtime import NativePhysicsRuntime

    values = np.array([
        0.0, 0.25, 0.5 - 2.0 ** -25, 0.5, 0.5 + 2.0 ** -26, 0.5 + 2.0 ** -25,
        0.5 + 2.0 ** -24, 0.5 + 2.0 ** -23, 0.5 + 1.0e-6, 0.75, 1.0,
    ])
    for dtype in (np.float32, np.float64):
        lf = values.astype(dtype)[None]
        batch = SimpleNamespace(xp=np, surface=SimpleNamespace(
            land_fraction=lf, sea_ice_fraction=np.zeros_like(lf)))
        assert np.array_equal(
            statics.water_columns(lf),
            NativePhysicsRuntime._xland(batch) >= np.float32(1.5),
        )
        # An ice-free planet's flag is bit-for-bit the pre-seeding
        # construction; a frozen column is exactly 1 (land) whatever its
        # fraction, and no longer water.
        assert np.array_equal(
            NativePhysicsRuntime._xland(batch),
            np.asarray(1.0 + (1.0 - lf), dtype=np.float32),
        )
        ice = np.zeros_like(lf)
        ice[0, :3] = 1.0
        frozen_batch = SimpleNamespace(xp=np, surface=SimpleNamespace(
            land_fraction=lf, sea_ice_fraction=ice))
        flag = NativePhysicsRuntime._xland(frozen_batch)
        assert np.all(flag[0, :3] == 1.0)
        assert np.array_equal(flag[0, 3:], np.asarray(1.0 + (1.0 - lf), dtype=np.float32)[0, 3:])
        assert not statics.water_columns(lf, sea_ice_fraction=ice)[0, :3].any()
    # float32 rounding moves a fraction 2^-26 above one half onto the water
    # side; the held fraction agrees with the categories in both precisions
    land = ~statics.water_columns(values)
    assert not land[values <= 0.5].any() and land[values >= 0.5 + 2.0 ** -23].all()
    held = statics.consistent_land_fraction(values, land)
    assert np.all(held[land] >= 0.5 + statics.LAND_FRACTION_MARGIN)
    assert np.all(held[~land] <= 0.5)
    assert np.max(np.abs(held - values)) <= statics.LAND_FRACTION_MARGIN
    for dtype in (np.float32, np.float64):
        assert np.array_equal(statics.water_columns(held.astype(dtype)), ~land)
    with pytest.raises(ValueError, match="shapes differ"):
        statics.consistent_land_fraction(values, land[:3])


def test_category_convention_round_trips_and_refuses_malformed_rows():
    row = statics.SYNTHETIC_CONVENTION.as_metadata("synthetic")
    assert statics.CategoryConvention.from_metadata(row) == statics.SYNTHETIC_CONVENTION
    real = statics.CategoryConvention.from_landuse_attrs({
        "mminlu": "USGS", "iswater": 16, "islake": 28, "isice": 24, "isurban": 1})
    assert (real.landuse_dataset, real.water_category, real.ice_category) == ("USGS", 16, 24)
    with pytest.raises(ValueError, match="lacks water_category"):
        statics.CategoryConvention.from_metadata({k: v for k, v in row.items() if k != "water_category"})
    with pytest.raises(ValueError, match="positive integer"):
        statics.CategoryConvention.from_metadata({**row, "ice_category": 0})
    with pytest.raises(ValueError, match="non-empty string"):
        statics.CategoryConvention.from_metadata({**row, "landuse_dataset": ""})
    with pytest.raises(ValueError, match="is not an object"):
        statics.CategoryConvention.from_metadata("MODIS")
    with pytest.raises(ValueError, match="source must be one of"):
        statics.SYNTHETIC_CONVENTION.as_metadata("constant")
    assert statics.surface_statics_metadata("synthetic", statics.SYNTHETIC_CONVENTION) == {
        "surface_statics": row}
    # The ice-class roughness travels in the row; a row from before the
    # field reads the default (its roughness plane is the one it ran with);
    # a value that is not a roughness in metres is refused.
    assert row["sea_ice_roughness_m"] == statics.SEA_ICE_ROUGHNESS_M == 0.01
    older = {k: v for k, v in row.items() if k != "sea_ice_roughness_m"}
    assert statics.CategoryConvention.from_metadata(older).sea_ice_roughness_m == statics.SEA_ICE_ROUGHNESS_M
    assert statics.CategoryConvention.from_metadata({**row, "sea_ice_roughness_m": 0.001}).sea_ice_roughness_m == 0.001
    for bad in (0.0, 1.5, "1 cm", True):
        with pytest.raises(ValueError, match="roughness in metres"):
            statics.CategoryConvention.from_metadata({**row, "sea_ice_roughness_m": bad})


# ---------------------------------------------------------------------------
# config and doors
# ---------------------------------------------------------------------------

def _analysis_frame(nlat=37, nlon=72):
    """A regular global analysis frame shaped like the mapped GDAS decode."""
    from dataclasses import dataclass
    from types import SimpleNamespace

    @dataclass
    class Field:
        values: np.ndarray

    lat = np.linspace(90.0, -90.0, nlat)
    lon = np.arange(nlon) * (360.0 / nlon)
    levels = np.asarray([10000.0, 30000.0, 50000.0, 70000.0, 85000.0, 100000.0])
    lat2 = np.deg2rad(lat)[:, None] * np.ones((1, nlon))
    shape3 = (levels.size, nlat, nlon)
    temperature = 220.0 + 70.0 * (levels / 100000.0)[:, None, None] * np.cos(lat2)[None] ** 2
    humidity = 0.01 * (levels / 100000.0)[:, None, None] ** 3 * np.ones(shape3)
    terrain = 1500.0 * np.exp(-((np.rad2deg(lat2) - 30.0) / 15.0) ** 2)
    ps = 101000.0 * np.exp(-terrain / 8000.0)
    land = (terrain > 100.0).astype(np.float64)
    skin = 288.0 - 30.0 * np.sin(lat2) ** 2
    soil_t = np.where(land[None] > 0.5, skin[None] - 1.0, np.nan) * np.ones((4, 1, 1))
    soil_m = np.where(land[None] > 0.5, 0.3, np.nan) * np.ones((4, 1, 1))
    # The surface seeding's four planes, GDAS-shaped: sea ice on the water
    # poleward of 70 degrees (1.2 m thick), snow on the land north of 38
    # degrees (40 kg/m2 over 0.2 m), the snow planes masked on open water.
    lat_deg = np.rad2deg(lat2)
    ice = np.where((land < 0.5) & (np.abs(lat_deg) >= 70.0), 1.0, 0.0)
    thickness = np.where(ice >= 0.5, 1.2, 0.0)
    snowy = (land >= 0.5) & (lat_deg >= 38.0)
    open_water = (land < 0.5) & (ice < 0.5)
    swe = np.where(open_water, np.nan, np.where(snowy, 40.0, 0.0))
    snow_depth = np.where(open_water, np.nan, np.where(snowy, 0.2, 0.0))
    fields = {
        "sea_ice_fraction": Field(ice),
        "sea_ice_thickness": Field(thickness),
        "snow_water_equivalent": Field(swe),
        "snow_depth": Field(snow_depth),
        "air_temperature": Field(temperature),
        "specific_humidity": Field(humidity),
        "eastward_wind": Field(20.0 * np.cos(lat2)[None] * np.ones(shape3)),
        "northward_wind": Field(np.zeros(shape3)),
        "surface_pressure": Field(ps),
        "terrain_height": Field(terrain),
        "skin_temperature": Field(skin),
        "land_fraction": Field(land),
        "soil_temperature": Field(soil_t),
        "volumetric_soil_moisture": Field(soil_m),
    }
    return SimpleNamespace(
        latitude=lat, longitude=lon, vertical_kind="pressure",
        vertical_values=levels, fields=fields, mapping_sha256="test",
        input_sha256="test", source_cycle=VALID_TIME, valid_time=VALID_TIME,
    )


def _config(tmp_path, body: str, *, analysis: bool):
    text = Path(CONFIG).read_text(encoding="utf-8")
    if analysis:
        text = text.replace(
            "[initial]\n",
            "[initial]\nmode = \"analysis\"\nanalysis_grib = \"unused.grib\"\n"
            f"analysis_mapping = \"{MAPPING}\"\n", 1)
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith((
                "surface_pressure_pa", "surface_temperature_k",
                "top_temperature_k", "qv_surface", "zonal_wind_m_s",
                "perturbation_amplitude", "zonal_wavenumber",
                "terrain_amplitude_m"))) + "\n"
    path = tmp_path / "run.toml"
    path.write_text(text + body, encoding="utf-8")
    return path


def test_statics_table_defaults_and_identity(tmp_path):
    analytic = load_config(_config(tmp_path, "", analysis=False))
    assert analytic.statics.source == "synthetic" and not analytic.statics.declared
    assert "statics" not in analytic.config_identity
    baseline = analytic.config_hash
    analysis = load_config(_config(tmp_path, "", analysis=True))
    assert analysis.statics.source == "real"
    assert analysis.config_identity["statics"] == {"source": "real", "geog_data_res": "default"}
    declared = load_config(_config(tmp_path, '[statics]\nsource = "synthetic"\n', analysis=True))
    assert declared.statics.source == "synthetic" and declared.statics.declared
    assert "statics" not in declared.config_identity
    # the analytic hash is untouched by an explicit synthetic table
    explicit = load_config(_config(tmp_path, '[statics]\nsource = "synthetic"\n', analysis=False))
    assert explicit.config_hash == baseline
    real_analytic = load_config(_config(
        tmp_path, '[statics]\nsource = "real"\nvalid_time = "2026-08-30T18:00:00Z"\n',
        analysis=False))
    assert real_analytic.config_identity["statics"]["valid_time"] == "2026-08-30T18:00:00Z"
    with pytest.raises(ValueError, match="statics.valid_time"):
        load_config(_config(tmp_path, '[statics]\nsource = "real"\n', analysis=False))
    with pytest.raises(ValueError, match="read only with initial.mode='analytic'"):
        load_config(_config(
            tmp_path, '[statics]\nvalid_time = "2026-08-30T18:00:00Z"\n', analysis=True))
    with pytest.raises(ValueError, match="read only with statics.source='real'"):
        load_config(_config(
            tmp_path, '[statics]\nsource = "synthetic"\ngeog_root = "x"\n', analysis=True))
    with pytest.raises(ValueError, match="statics.source must be"):
        load_config(_config(tmp_path, '[statics]\nsource = "guess"\n', analysis=True))


@requires_rows_static_fields
def test_statics_door_builds_the_cache_the_run_reads(geog_root, tmp_path, capsys):
    _needs_builder()
    from woof.globe.analysis_initial import analysis_initial_state
    from woof.globe.cli import main
    from woof.globe.runner import build_transform

    cache_dir = tmp_path / "statics-cache"
    body = (f'[statics]\nsource = "real"\ngeog_root = "{geog_root.as_posix()}"\n'
            f'cache_dir = "{cache_dir.as_posix()}"\n')
    path = _config(tmp_path, body, analysis=True)
    # the analysis grid is T3 on the smoke config: cheap, and the door
    # names the same cache path the run resolves
    assert main(["statics", str(path), "--sector-degrees", "120"]) == 0
    out = capsys.readouterr().out
    assert "statics real T3" in out
    summary = json.loads(out[out.index("{"):])
    cache = Path(summary["cache"])
    assert cache.parent == cache_dir and cache.name.startswith("arwen-global-statics-T3-")
    cfg = load_config(path)
    transform = build_transform(cfg)
    state, _phi, provenance = analysis_initial_state(
        cfg, transform, frame=_analysis_frame())
    row = provenance["statics"]
    assert row["source"] == "real" and row["cache"] == str(cache)
    assert row["cache_self_sha256"] == summary["self_sha256"]
    assert row["coverage"]["landuse"]["status"] == ["PASS"]
    host = transform.backend.to_numpy
    categories = host(state.surface.landuse_category)
    assert categories.min() >= 1 and categories.max() <= 20
    assert (categories == ISWATER).any() and (categories != ISWATER).any()
    ice = host(state.surface.sea_ice_fraction)
    assert np.array_equal(
        categories == ISWATER,
        statics.water_columns(host(state.surface.land_fraction), sea_ice_fraction=ice))
    # the seeded sea ice carries the ice class on exactly the frozen columns
    assert (ice >= statics.SEA_ICE_THRESHOLD).any()
    assert np.all(categories[ice >= statics.SEA_ICE_THRESHOLD] == 15)
    # the convention row travels with the state for the land surface's check
    assert state.physics_state.metadata[statics.SURFACE_STATICS_METADATA_KEY] == {
        "source": "real", "landuse_dataset": "MODIFIED_IGBP_MODIS_NOAH",
        "soil_dataset": "STAS", "water_category": ISWATER, "lake_category": ISLAKE,
        "ice_category": ISICE, "urban_category": ISURBAN,
        "water_soil_category": 14, "ice_soil_category": 16,
        "sea_ice_roughness_m": statics.SEA_ICE_ROUGHNESS_M,
    }
    # the seeded pack carries the convention row's roughness; the land ice
    # (the ice class on land) keeps LANDUSE.TBL's 0.1 cm; nothing else
    # carries the pack's value
    roughness = host(state.surface.roughness_m)
    pack = ice >= statics.SEA_ICE_THRESHOLD
    landice = (categories == ISICE) & ~pack
    assert np.allclose(roughness[pack], statics.SEA_ICE_ROUGHNESS_M, atol=1e-7)
    if landice.any():
        assert np.allclose(roughness[landice], 1.0e-3, atol=1e-7)
    assert not np.any(np.isclose(roughness[~pack], statics.SEA_ICE_ROUGHNESS_M, atol=1e-7))
    assert row["convention"] == state.physics_state.metadata[statics.SURFACE_STATICS_METADATA_KEY]
    assert 0.0 <= host(state.surface.vegetation_fraction).max() <= 1.0
    assert not np.allclose(host(state.surface.albedo), 0.08 + 0.12 * host(state.surface.land_fraction))
    # the door line names the cache it read, and a second build without
    # --overwrite is a refusal: exit 1, the code this package's doors give
    # every refusal (the 2 this line used to expect is the owner tree's
    # boundary, which this door never passes through)
    assert main(["statics", str(path)]) == 1
    err = capsys.readouterr().err
    assert "--overwrite" in err


def test_run_door_prints_the_synthetic_planet(tmp_path, capsys):
    from woof.globe.cli import main

    path = _config(tmp_path, "", analysis=False)
    assert main(["run", str(path), "--outdir", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "statics synthetic (analytic default)" in out
    receipt = json.loads((tmp_path / "out" / "arwen-global-receipt.json").read_text(encoding="utf-8"))
    assert receipt["statics"]["source"] == "synthetic"
    assert receipt["statics"]["declared"] is False
    with pytest.raises(ValueError, match="nothing to build"):
        from woof.globe.cli import _statics
        import argparse
        _statics(argparse.Namespace(config=path, out=None, overwrite=False,
                                    sector_degrees=30.0))


def test_a_rows_grid_answers_every_handle_the_engine_static_build_asks_for(monkeypatch):
    """The engine's routed static build calls both grid handles.

    ``woof.static.build._build_static_routed`` takes the Rust route for a
    grid that has ``_rust_handle`` and then samples through
    ``_rust_sampling_handle``; from 2.8.0 a grid with only the first raised
    AttributeError on the first sector of every global statics build.  A
    rows grid has no nest ancestry, so the sampling handle is the
    coordinate handle, for the whole grid and for a sector alike.
    """

    import inspect
    import re

    from woof.static import build as engine_build

    from woof.globe import statics_rows

    # The fake handles below must never reach the real library's free.
    monkeypatch.setattr(statics_rows, "_free_rust_grid_handle", lambda handle: None)

    source = inspect.getsource(engine_build)
    asked = sorted(set(re.findall(r"grid\.(_rust_[a-z_]*handle)\(", source)))
    assert asked, "the engine's static build no longer asks the grid for a handle"
    for name in asked:
        assert callable(getattr(RowsGrid, name, None)), name

    class _Bridge:
        def __init__(self):
            self.calls = []

        def grid_new(self, spec):
            self.calls.append(("new", spec["e_we"]))
            return 7

        def grid_translated(self, base, di, dj, e_we, e_sn):
            self.calls.append(("translated", base, di, dj))
            return 11

    grid = RowsGrid([-10.0, 0.0, 10.0], 0.0, 1.0, 360)
    bridge = _Bridge()
    assert grid._rust_sampling_handle(bridge) == grid._rust_handle(bridge) == 7
    sector = grid.sector(4, 8)
    assert sector._rust_sampling_handle(bridge) == 11
    assert bridge.calls == [("new", grid.e_we), ("translated", 7, 4, 0)]
