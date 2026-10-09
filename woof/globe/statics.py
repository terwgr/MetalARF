"""Static surface fields for the global model, on its own Gaussian grid.

The land surface needs what every real-data model gets from geogrid:
land-use and soil categories, monthly vegetation fraction, leaf area,
albedo, maximum snow albedo and the deep-soil boundary temperature.  Until
2026-09-01 the native suite ran the planet on one vegetation class, one
soil class, LAI 3.0, snow albedo 0.6 and a deep-soil temperature equal to
the bottom soil layer -- and the 24 h GDAS forecast measured +4.2 K warm
and -4.1 K dry at 2 m against 1,675 ASOS stations.

This module is the orchestration around the Rust static-field builder
(``tools/rustwx/crates/static-fields``, driven through
:mod:`woof.static.build`): it describes the Gaussian grid to the crate as
a ``rows`` grid (:class:`woof.globe.statics_rows.RowsGrid`), cuts the ring into
longitude sectors so a whole-globe 30-arc-second window is never read at
once, caches the crate's fields once per truncation with provenance and
per-array hashes, and resolves them to the run's date and to the
categories Noah is driven with through the SAME rulebook the regional
model uses (:mod:`woof.globe.core.landuse`, :mod:`woof.globe.core.noah`).  Every
byte of source data is read and sampled by the crate; nothing here
touches a tile.

Two sources exist.  ``real`` (the default for an analysis-initialized
run) reads the WPS_GEOG archive through the crate and refuses, with the
remedy, when the archive or the cache is missing.  ``synthetic`` is the
pre-2026-09-01 constant planet, kept for smoke and test configurations
that have no geography and must say so: it is selected explicitly, printed
at the door and recorded in the receipt as synthetic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np

STATICS_SCHEMA = "gpuwm.arwen-global-statics/v1"
STATICS_SOURCES = ("real", "synthetic")
#: Longitude width of one build sector.  A 30-degree sector of the 30
#: arc-second archive is a 3,600 x 21,600 source window (78 M cells, 0.6
#: GB widened) plus the ~72 decoded tiles behind it; the twelve-plane
#: vegetation climatology is the largest at ~18 GB peak.  A whole-globe
#: window of that dataset would be 90 GB of tiles alone.
SECTOR_DEGREES = 30.0
#: The crate fields the cache keeps (the two 16-plane soil fraction
#: stacks are dropped: only their dominant categories drive anything).
CACHED_FIELDS = (
    "HGT_M", "LANDMASK", "LU_INDEX", "SCT_DOM", "SCB_DOM", "LANDUSEF",
    "GREENFRAC", "LAI12M", "ALBEDO12M", "SNOALB", "SOILTEMP",
)
#: The nine GEOG roles the crate reads, in ``GeogSelection`` field order.
GEOG_ROLES = (
    "terrain", "landuse", "soil_top", "soil_bottom", "greenfrac", "lai",
    "albedo", "snow_albedo", "soil_temperature",
)
#: SurfaceState members this module seeds (both sources).
SURFACE_STATIC_FIELDS = (
    "landuse_category", "soil_category_top", "soil_category_bottom",
    "vegetation_fraction", "vegetation_fraction_min",
    "vegetation_fraction_max", "leaf_area_index", "background_albedo",
    "snow_albedo", "deep_soil_temperature_k",
    "albedo", "emissivity", "roughness_m", "lake_fraction",
)
#: physics_state.metadata key under which the category convention of the
#: surface state's static fields travels with the state (cold start,
#: checkpoint, restart, migration): the VEGPARM/SOILPARM sections the
#: categories index and the special categories, so the land surface can
#: check the tables it loads against the planet it is handed.
SURFACE_STATICS_METADATA_KEY = "surface_statics"
#: The SOILPARM.TBL section every static source here indexes.
SOIL_DATASET = "STAS"
#: A column whose analysed sea-ice fraction reaches this is a frozen
#: surface: WRF's non-fractional sea-ice rule (xice_threshold 0.5,
#: woof.globe.core.landuse._derive_categories), the same test the runtime's
#: land/water flag and the Noah kernel's sea-ice skip apply, so the
#: columns carrying the ice class are exactly the columns run as ice.
SEA_ICE_THRESHOLD = 0.5

# The synthetic planet: the constants the native suite ran on before real
# statics existed, kept as the explicit test arm.  MODIS category 7 is
# open shrubland, STAS soil 8 silty clay loam.
SYNTHETIC_VEGETATION_CATEGORY = 7
SYNTHETIC_SOIL_CATEGORY = 8
SYNTHETIC_LEAF_AREA_INDEX = 3.0
SYNTHETIC_SNOW_ALBEDO = 0.6
SYNTHETIC_EMISSIVITY = 0.96
#: Sea-ice columns of the synthetic planet: WRF's seaice_albedo_default;
#: their roughness is the convention row's sea-ice value.
SYNTHETIC_SEA_ICE_ALBEDO = 0.65
#: The momentum roughness (m) the sea-ice columns carry (the ice class on
#: water at or above the sea-ice threshold), in place of LANDUSE.TBL's
#: 0.1 cm for the class.  WRF's tables give the class 0.1 cm (USGS 24,
#: MODIS 15), 1 cm (the RUC MODIS rows), 1.2 cm (SSiB 22) and 5 cm (SiB
#: winter); the GFS carries 1.0 cm on sea ice.  Measured skin roughness
#: over smooth snow is 0.01 to 1 mm, over ridged or broken pack 1 mm to
#: 1 cm (Andreas et al. 2010; Lupkes et al. 2012 for the form drag of
#: floe edges), and a 52 km column has no floe-edge drag scheme, so the
#: pack carries the effective value that stands in for it: 1 cm, the
#: rough end of the measured range and the value the reference model
#: runs.  Graded on the polar footprints (frozen-columns lane): at 0.1 cm
#: the merged tip ran the pack's 10 m wind 1.3 m/s too fast against the
#: GFS analysis; at 1 cm the pack reads +0.05 m/s with its 2 m temperature
#: inside the admission rule.  A stated divergence from WRF's LANDUSE.TBL;
#: the regional rulebook (woof.globe.core.landuse) keeps the table's value.
#: Land ice keeps the table's 0.1 cm: 1 cm on the ice sheets cut the
#: Antarctic 10 m wind from +2.30 to +0.83 m/s but raised the Antarctic
#: land and snow-covered land 2 m rmse by 0.08 K (3.03 to 3.11, 2.93 to
#: 3.01), past the admission rule's 0.03 K; the sheets' form drag is a
#: named follow-up, not a roughness.
SEA_ICE_ROUGHNESS_M = 0.01
#: SOILPARM STAS water and land-ice soil categories (WRF namelist
#: isoilwater/isoilice defaults).
ISOILWATER = 14
ISOILICE = 16
#: WRF real's TMN elevation correction (share/module_soil_pre.F:973).
TMN_LAPSE_K_PER_M = 0.0065
#: Land columns carry a land fraction at least this far above one half.
#: The native suite decides open water per column as WRF's
#: xland = 1 + (1 - land_fraction) >= 1.5, formed in the surface state's
#: precision and cast to float32 (native_runtime._xland): a land fraction
#: in (0.5, 0.5 + 2^-23) can round to exactly 0.5 in float32 and land on
#: the water side while its categories say land.  Two float32 ulps at 0.5
#: is the smallest margin exact in both precisions: 1 - lf <= 0.5 - 2^-23
#: is exact (Sterbenz) and 1.5 - 2^-23 is representable, so the test lands
#: on the land side whatever the state's dtype.  A water column's
#: fraction is <= 0.5, which every rounding preserves.
LAND_FRACTION_MARGIN = 2.0 ** -23


@dataclass(frozen=True)
class CategoryConvention:
    """Which table sections the surface state's categories index.

    ``landuse_dataset`` is WRF's MMINLU (the VEGPARM.TBL / LANDUSE.TBL
    section), ``soil_dataset`` its MMINSL (the SOILPARM.TBL section); the
    special categories are the ones geogrid's index declares for that
    land-use set (ISWATER/ISLAKE/ISICE/ISURBAN) and WRF's namelist soil
    defaults (isoilwater/isoilice).  Noah's kernel indexes its tables with
    ``vegtyp - 1`` and ``soiltyp - 1`` and skips ``vegtyp == isice``
    columns (noah.cu:976, :1056-1057), so a state whose categories belong
    to another section would read another class's parameters without any
    error: the native runtime refuses on a mismatch instead.
    """

    landuse_dataset: str
    soil_dataset: str
    water_category: int
    lake_category: int
    ice_category: int
    urban_category: int
    water_soil_category: int
    ice_soil_category: int
    #: The momentum roughness (m) the sea-ice columns take
    #: (:data:`SEA_ICE_ROUGHNESS_M`); a row written before the field
    #: existed reads the default, and the roughness plane such a state
    #: carries is the one it ran with.  Land ice keeps LANDUSE.TBL's value.
    sea_ice_roughness_m: float = SEA_ICE_ROUGHNESS_M

    @classmethod
    def from_landuse_attrs(cls, attrs: dict) -> "CategoryConvention":
        """From the lower-cased geogrid index attributes a cache sidecar
        records (``provenance["landuse"]``)."""
        return cls(
            landuse_dataset=str(attrs["mminlu"]), soil_dataset=SOIL_DATASET,
            water_category=int(attrs["iswater"]),
            lake_category=int(attrs["islake"]),
            ice_category=int(attrs["isice"]),
            urban_category=int(attrs["isurban"]),
            water_soil_category=ISOILWATER, ice_soil_category=ISOILICE,
        )

    def as_metadata(self, source: str) -> dict[str, object]:
        """The JSON row stored under :data:`SURFACE_STATICS_METADATA_KEY`."""
        if source not in STATICS_SOURCES:
            raise ValueError(f"statics source must be one of {STATICS_SOURCES}, got {source!r}")
        return {"source": source, **asdict(self)}

    @classmethod
    def from_metadata(cls, payload) -> "CategoryConvention":
        if not isinstance(payload, dict):
            raise ValueError(
                f"physics_state.metadata[{SURFACE_STATICS_METADATA_KEY!r}] "
                "is not an object")
        names = tuple(cls.__dataclass_fields__)
        optional = ("sea_ice_roughness_m",)
        missing = [name for name in names if name not in payload and name not in optional]
        if missing:
            raise ValueError(
                f"physics_state.metadata[{SURFACE_STATICS_METADATA_KEY!r}] "
                f"lacks {', '.join(missing)}")
        values = {}
        for name in names:
            kind = cls.__dataclass_fields__[name].type
            if name in optional and name not in payload:
                continue   # a row from before the field: the dataclass default
            raw = payload[name]
            if kind == "str":
                if not isinstance(raw, str) or not raw:
                    raise ValueError(f"surface statics {name} must be a non-empty string")
                values[name] = raw
            elif kind == "float":
                if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not (0.0 < float(raw) < 1.0):
                    raise ValueError(f"surface statics {name} must be a roughness in metres inside (0, 1)")
                values[name] = float(raw)
            else:
                if isinstance(raw, bool) or int(raw) != raw or int(raw) < 1:
                    raise ValueError(f"surface statics {name} must be a positive integer")
                values[name] = int(raw)
        return cls(**values)


#: The synthetic planet's categories are MODIS / STAS ones (7 open
#: shrubland, 8 silty clay loam), so it declares that convention.
SYNTHETIC_CONVENTION = CategoryConvention(
    landuse_dataset="MODIFIED_IGBP_MODIS_NOAH", soil_dataset=SOIL_DATASET,
    water_category=17, lake_category=21, ice_category=15, urban_category=13,
    water_soil_category=ISOILWATER, ice_soil_category=ISOILICE,
)


def frozen_water_columns(sea_ice_fraction, xp=np):
    """The water columns the analysed sea ice freezes over: fraction at or
    above :data:`SEA_ICE_THRESHOLD`, evaluated in float32 like the kernel's
    ``xice >= xice_threshold`` test (noah.cu:969)."""
    return xp.asarray(sea_ice_fraction, dtype=xp.float32) >= xp.float32(SEA_ICE_THRESHOLD)


def xland_plane(land_fraction, sea_ice_fraction=None, xp=np):
    """WRF's land/water flag for the native suite, float32.

    THE rule: ``1 + (1 - land_fraction)`` (1 land, 2 water, fractional
    between) with every sea-ice column set to exactly 1, as real.exe's
    ``adjust_for_seaice`` sets XLAND = 1 on the columns it hands the ice
    class.  The runtime's kernel argument (native_runtime._xland), the
    statics' water test (:func:`water_columns`) and the land-step skip
    mask are all this plane, so the columns whose categories say water
    are exactly the columns sfclay runs its water branch on, Noah skips
    (noah.cu:968) and GF flags as water, and the columns whose categories
    say ice are exactly the columns sfclay runs as land and Noah skips as
    ice (noah.cu:969).  ``sea_ice_fraction`` None is the ice-free planet:
    the plane is then bit-for-bit the pre-seeding construction.
    """
    plane = xp.asarray(1.0 + (1.0 - land_fraction), dtype=xp.float32)
    if sea_ice_fraction is None:
        return plane
    return xp.where(
        frozen_water_columns(sea_ice_fraction, xp), xp.float32(1.0), plane
    ).astype(xp.float32)


def water_columns(land_fraction, xp=np, *, sea_ice_fraction=None):
    """The columns the native suite treats as open water: xland at or
    above 1.5 on :func:`xland_plane` (so a sea-ice column is not water).
    Evaluated on the array in its own precision, as the runtime does on
    the surface state.
    """
    return xland_plane(land_fraction, sea_ice_fraction, xp) >= xp.float32(1.5)


def lake_columns(land_fraction, lake_fraction, xp=np, *, sea_ice_fraction=None):
    """The open-water columns that are inland lakes: :func:`water_columns`
    whose water is mostly lake, ``lake_fraction > (1 - land_fraction) -
    lake_fraction`` in float32.

    WRF's rule (module_initialize_real.F, LAKEMASK): a column is a lake
    where the land-use set's lake class is its dominant water, which on a
    water column is the lake share beating the ocean-water share.  The
    lake fraction is the statics' LANDUSEF of the lake class, so the rule
    names the lakes the land-use set names (MODIS with lakes), not a
    connectivity or size threshold on the run grid.  A state without the
    plane (the synthetic planet, a checkpoint from before it) has none.
    """
    land = xp.asarray(land_fraction, dtype=xp.float32)
    lake = xp.asarray(lake_fraction, dtype=xp.float32)
    return water_columns(land, xp, sea_ice_fraction=sea_ice_fraction) & (
        lake > (xp.float32(1.0) - land) - lake
    )


def consistent_land_fraction(land_fraction, land) -> np.ndarray:
    """``land_fraction`` (float64) placed on the side of one half that
    ``land`` says, by :data:`LAND_FRACTION_MARGIN`: a land column's
    fraction is raised to at least 0.5 + 2^-23, a water column's capped at
    0.5.  Real statics need this only for a fraction within 1.2e-7 of one
    half; the correction is that small and the receipt's water test then
    agrees with the categories in every precision."""
    lf = np.asarray(land_fraction, dtype=np.float64)
    land = np.asarray(land, dtype=bool)
    if land.shape != lf.shape:
        raise ValueError("land mask and land fraction shapes differ")
    out = np.where(land, np.maximum(lf, 0.5 + LAND_FRACTION_MARGIN), np.minimum(lf, 0.5))
    return np.clip(out, 0.0, 1.0)


def surface_statics_metadata(source: str, convention: CategoryConvention) -> dict[str, object]:
    """``{SURFACE_STATICS_METADATA_KEY: row}`` for a cold state's
    ``PhysicsState.metadata``."""
    return {SURFACE_STATICS_METADATA_KEY: convention.as_metadata(source)}


@dataclass(frozen=True)
class StaticsOptions:
    """The ``[statics]`` table.

    ``source`` is ``real`` or ``synthetic``.  The dataclass default is
    ``synthetic`` so a config built without the table behaves as the
    analytic controls always did; :func:`woof.globe.config.
    load_config` selects ``real`` for every analysis-initialized run
    (fixed means default) and ``synthetic`` for the analytic planet,
    whose land mask is a formula and carries no geography to look up.
    ``declared`` records whether the config named the source itself.
    """

    source: str = "synthetic"
    geog_root: str | None = None
    geog_data_res: str = "default"
    cache_dir: str | None = None
    valid_time: str | None = None
    declared: bool = False

    @property
    def identity(self) -> dict[str, object] | None:
        """What enters the config identity: only the real arm, and only
        the choices that change results (source, dataset tokens, the
        analytic-arm date).  Paths are machine-local and stay out; the
        synthetic arm returns None so every pre-existing analytic config
        hash stays byte-identical."""
        if self.source != "real":
            return None
        payload: dict[str, object] = {
            "source": "real", "geog_data_res": self.geog_data_res,
        }
        if self.valid_time is not None:
            payload["valid_time"] = self.valid_time
        return payload

    @property
    def valid_time_utc(self) -> datetime | None:
        if self.valid_time is None:
            return None
        return parse_utc(self.valid_time, "statics.valid_time")

    def tokens(self) -> tuple[str, ...]:
        from woof.static.build import GeogSelection

        return GeogSelection.from_tokens(Path("."), self.geog_data_res).resolution_tokens


def parse_utc(text: str, name: str) -> datetime:
    value = str(text).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not an ISO-8601 instant: {text!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must carry an explicit UTC offset: {text!r}")
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# locations
# ---------------------------------------------------------------------------

def default_cache_dir() -> Path:
    """Beside the case data, like the geography archive itself."""
    from woof.case_data import case_data_root

    return case_data_root() / "arwen-global-statics"


def cache_dir_for(options: StaticsOptions) -> Path:
    if options.cache_dir:
        return Path(options.cache_dir)
    return default_cache_dir()


def cache_stem(grid, tokens) -> str:
    return (f"arwen-global-statics-T{int(grid.truncation)}-"
            f"{int(grid.nlat)}x{int(grid.nlon)}-{'+'.join(tokens)}")


def cache_paths(options: StaticsOptions, grid) -> tuple[Path, Path]:
    """``(npz, sidecar)`` for this truncation and dataset selection."""
    stem = cache_stem(grid, options.tokens())
    directory = cache_dir_for(options)
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def resolve_geog_root(options: StaticsOptions):
    """``(root, GeogSelection)`` or a refusal naming the remedy."""
    from woof.rustwx_static import geog_root_candidates
    from woof.static.build import GeogSelection

    if options.geog_root:
        root = Path(options.geog_root)
        looked = [root]
    else:
        looked = list(geog_root_candidates())
        root = next((path for path in looked if path.is_dir()), None)
    if root is None or not root.is_dir():
        rendered = "\n".join(f"    {path}" for path in looked)
        raise FileNotFoundError(
            "no WPS_GEOG archive, so the global model's static fields "
            "cannot be built; looked here and none is a directory:\n"
            f"{rendered}\n"
            "what does work:\n"
            "  woof fetch-geog, which stages the archive into the "
            "case-data root\n"
            "  set GPUWM_WPS_GEOG to an archive already on this box\n"
            "  [statics] geog_root = \"...\" in the run config\n"
            "  [statics] source = \"synthetic\" for a smoke or test "
            "configuration that has no geography and says so")
    selection = GeogSelection.from_tokens(root, options.geog_data_res)
    missing = [role for role in GEOG_ROLES
               if not (selection.path(role) / "index").is_file()]
    if missing:
        raise FileNotFoundError(
            f"the WPS_GEOG archive at {root} lacks "
            f"{len(missing)} dataset(s) the global statics read: "
            + ", ".join(f"{role} ({selection.path(role).name})"
                        for role in missing)
            + ".  A static built without one would hand the model zeros "
            "as geography.  woof fetch-geog stages every one of them; "
            "or point [statics] geog_root at a complete archive.")
    return root, selection


def require_rust_builder():
    """The static-fields bridge, or a refusal: no Python body exists for
    a rows grid, so the fallback that reports itself as a workaround for
    WPS domains is a refusal here."""
    from woof.static import rust_bridge

    if rust_bridge.python_fallback_requested():
        raise RuntimeError(
            f"{rust_bridge.STATIC_PYTHON_ENV}=1 selects the pure-Python "
            "static build, which has no body for the global model's rows "
            "grid; unset it -- every byte of the global statics is built "
            "by the Rust static-fields library")
    reason = rust_bridge.unavailable_reason()
    if reason is not None:
        # A door that is not there is exit 3, not exit 1, and its remedy is
        # the bundle that publishes it.  The engine's own resolver ends its
        # account with a `cargo build` in a checkout, which is the remedy for
        # the person developing the engine and no remedy at all for the
        # person who installed two wheels.
        from .doors import missing_door_refusal

        raise missing_door_refusal("static_fields", str(reason))
    return rust_bridge


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------

def _array_hash(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(arr.dtype.str.encode())
    digest.update(str(arr.shape).encode())
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def build_statics(grid, options: StaticsOptions, *,
                  sector_degrees: float = SECTOR_DEGREES,
                  progress=None) -> tuple[dict[str, np.ndarray], dict]:
    """Build the crate's static fields on ``grid`` (a ``GaussianGrid``).

    Returns ``(fields, provenance)``: the :data:`CACHED_FIELDS` as float32
    arrays on ``(nlat, nlon)`` (stacks lead with their plane axis), plus
    the grid's own latitudes/longitudes, and the provenance document the
    cache sidecar is written from.
    """
    from woof.static.build import build_static
    from woof.globe.statics_rows import RowsGrid

    bridge = require_rust_builder()
    root, selection = resolve_geog_root(options)
    landuse = selection.landuse_global_attrs()
    ring = RowsGrid.from_gaussian(grid)
    if not ring.closes:
        raise ValueError("the global statics build needs a closed ring of "
                         f"columns; {ring.nlon} x {ring.dlon_deg} deg is not 360")
    sectors = ring.sectors(sector_degrees)
    started = time.perf_counter()
    fields: dict[str, np.ndarray] = {}
    coverage: dict[str, list] = {role: [] for role in GEOG_ROLES}
    sector_rows = []
    for index, piece in enumerate(sectors):
        first, _ = piece._translation_offset
        report: dict[str, object] = {}
        timing: dict[str, object] = {}
        built = build_static(piece, root, 0, selection=selection,
                             source_coverage_report=report,
                             timing_report=timing)
        for name in CACHED_FIELDS:
            value = np.asarray(built[name], dtype=np.float64)
            if name not in fields:
                fields[name] = np.empty(
                    (*value.shape[:-1], ring.nlon), dtype=np.float64)
            fields[name][..., first:first + piece.nlon] = value
        for role in GEOG_ROLES:
            coverage[role].append(report[role])
        sector_rows.append({
            "index": index, "first_column": int(first),
            "columns": int(piece.nlon),
            "seconds": float(timing["seconds"]),
        })
        if progress is not None:
            progress(index + 1, len(sectors), float(timing["seconds"]))
    wall = time.perf_counter() - started
    out = {name: np.ascontiguousarray(value, dtype=np.float32)
           for name, value in fields.items()}
    out["latitude_deg"] = np.ascontiguousarray(grid.latitude_deg, dtype=np.float64)
    out["longitude_deg"] = np.ascontiguousarray(grid.longitude_deg, dtype=np.float64)
    library = bridge.resolve_static_bridge()
    provenance = {
        "schema": STATICS_SCHEMA,
        "truncation": int(grid.truncation),
        "nlat": int(grid.nlat), "nlon": int(grid.nlon),
        "grid": {
            "kind": "rows", "lat_first_deg": float(grid.latitude_deg[0]),
            "lat_last_deg": float(grid.latitude_deg[-1]),
            "lon0_deg": ring.lon0_deg, "dlon_deg": ring.dlon_deg,
        },
        "geog_root": str(root),
        "geog_data_res": list(selection.resolution_tokens),
        "datasets": {role: getattr(selection, role) for role in GEOG_ROLES},
        "landuse": {key.lower(): value for key, value in landuse.items()},
        "halo": 0,
        "sector_degrees": float(sector_degrees),
        "sectors": sector_rows,
        "builder": {
            "library": str(library), "sha256": _file_hash(library),
            "abi": int(bridge.STATIC_ABI),
        },
        "built_at_utc": datetime.now(timezone.utc).replace(microsecond=0)
        .isoformat().replace("+00:00", "Z"),
        # The KIND of machine that built the cache, never its name.  Until
        # 0.1.2 this field carried the hostname, on the reasoning that the
        # sidecar stays in the user's own cache.  It does not stay there: a
        # geog cache is built once and copied between machines, attached to
        # support requests and published beside a run, and every copy named
        # the machine it came from.  The kind is what a reader can act on
        # (whether a stored build applies to theirs), the same rule the
        # render catalog follows in `tools/arwen_global_render_catalog
        # ._machine_class`; two caches are told apart by `built_at_utc` and
        # their array digests, which the name never added to.
        "host": machine_class(),
        "wall_seconds": float(wall),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": value.dtype.str,
                   "sha256": _array_hash(value)}
            for name, value in out.items()
        },
        "coverage": coverage,
    }
    return out, provenance


def machine_class() -> str:
    """The kind of machine this process runs on: system and architecture.

    Never a hostname, a user name or an operating-system build number.
    """

    return f"{platform.system().lower()}-{platform.machine().lower()}"


def write_cache(npz_path: Path, fields: dict[str, np.ndarray],
                provenance: dict, *, overwrite: bool = False) -> tuple[Path, Path]:
    """Write the arrays and the self-hashed sidecar beside them."""
    npz_path = Path(npz_path)
    sidecar = npz_path.with_suffix(".json")
    for path in (npz_path, sidecar):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"statics cache {path} exists; pass --overwrite to rebuild it")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    document = dict(provenance)
    document.pop("self_sha256", None)
    _canonical(document)
    temporary = npz_path.with_name(f".{npz_path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **fields)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, npz_path)
    document["npz_sha256"] = _file_hash(npz_path)
    document["self_sha256"] = hashlib.sha256(_canonical(document)).hexdigest()
    sidecar.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    return npz_path, sidecar


def read_cache(npz_path: Path, grid=None) -> tuple[dict[str, np.ndarray], dict]:
    """Read a cache, verifying the sidecar's self-hash, every array's hash
    and (when ``grid`` is given) that the arrays sit on that grid."""
    npz_path = Path(npz_path)
    sidecar = npz_path.with_suffix(".json")
    for path in (npz_path, sidecar):
        if not path.is_file():
            raise FileNotFoundError(missing_cache_remedy(npz_path))
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != STATICS_SCHEMA:
        raise ValueError(f"statics sidecar {sidecar} is not a {STATICS_SCHEMA} document")
    stated = document.pop("self_sha256", None)
    if stated != hashlib.sha256(_canonical(document)).hexdigest():
        raise ValueError(f"statics sidecar {sidecar} self-hash mismatch")
    document["self_sha256"] = stated
    if _file_hash(npz_path) != document.get("npz_sha256"):
        raise ValueError(
            f"statics cache {npz_path} does not match its sidecar's "
            "npz_sha256; rebuild it with the statics door")
    with np.load(npz_path, allow_pickle=False) as archive:
        fields = {name: np.array(archive[name], copy=True) for name in archive.files}
    expected = document["arrays"]
    if set(expected) != set(fields):
        raise ValueError(f"statics cache {npz_path} array inventory mismatch")
    for name, row in expected.items():
        value = fields[name]
        if list(value.shape) != row["shape"] or value.dtype.str != row["dtype"]:
            raise ValueError(f"statics cache array {name} shape/dtype mismatch")
        if _array_hash(value) != row["sha256"]:
            raise ValueError(f"statics cache array {name} hash mismatch")
        if not np.isfinite(value).all():
            raise ValueError(f"statics cache array {name} contains non-finite values")
    if grid is not None:
        if (int(document["nlat"]), int(document["nlon"])) != (int(grid.nlat), int(grid.nlon)):
            raise ValueError(
                f"statics cache {npz_path} is {document['nlat']}x{document['nlon']}, "
                f"this run's grid is {grid.nlat}x{grid.nlon}")
        if not np.allclose(fields["latitude_deg"], grid.latitude_deg, atol=1.0e-9):
            raise ValueError(
                f"statics cache {npz_path} row latitudes are not this grid's "
                "Gauss-Legendre nodes; rebuild it with the statics door")
        if not np.allclose(fields["longitude_deg"], grid.longitude_deg, atol=1.0e-9):
            raise ValueError(
                f"statics cache {npz_path} column longitudes are not this "
                "grid's; rebuild it with the statics door")
    return fields, document


def missing_cache_remedy(npz_path: Path) -> str:
    return (
        f"no static-field cache for this truncation at {npz_path}; a "
        "real-data run needs the land-use, soil, vegetation, albedo and "
        "deep-soil fields Noah is driven with, and without them the "
        "planet would run on one vegetation class and one soil class "
        "(the +4.2 K / -4.1 K 2 m bias measured 2026-08-31).  Build it "
        "once for this truncation:\n"
        "  woof global statics <this run's config>\n"
        "or select [statics] source = \"synthetic\" in a smoke or test "
        "configuration that has no geography and says so")


def load_statics(options: StaticsOptions, grid) -> tuple[dict[str, np.ndarray], dict]:
    """The cached real statics for ``grid``, or the refusal with remedy."""
    npz_path, _ = cache_paths(options, grid)
    return read_cache(npz_path, grid)


# ---------------------------------------------------------------------------
# resolving to the run
# ---------------------------------------------------------------------------

def resolve_surface_statics(fields: dict[str, np.ndarray], provenance: dict, *,
                            valid_time: datetime, latitude_deg,
                            terrain_height_m, soil_temperature_k,
                            skin_temperature_k, sea_ice_fraction=None,
                            snow_water_kg_m2=None) -> tuple[dict[str, np.ndarray], dict]:
    """The surface state the real statics imply at ``valid_time``.

    ``sea_ice_fraction`` and ``snow_water_kg_m2`` are the analysis's
    planes on the Gaussian grid (woof.globe.surface_seeding);
    the rulebook puts the ice class and ice soil on every column at or
    above :data:`SEA_ICE_THRESHOLD` and the snow-covered albedo on every
    land column holding 10 kg/m2 or more, exactly as real.exe does with
    SEAICE and SNOW.  None is the ice-free, snow-free planet.

    Categories and the seasonal albedo/emissivity/roughness go through
    the regional model's rulebook (:func:`woof.globe.core.landuse.
    initialize_landuse`: lake folded to water, water soil forced, the
    real.exe landmask/soil reconciliation with the analysis soil and
    skin temperatures as its evidence, LANDUSE.TBL by season), and then
    the sea-ice columns (the ice class on water at or above the sea-ice
    threshold) take the convention row's roughness
    (:data:`SEA_ICE_ROUGHNESS_M`) in place of the table's 0.1 cm, while
    land ice keeps the table's.  A global domain has both hemispheres,
    so the season is chosen per hemisphere rather than from one domain
    centre.  Vegetation fraction, LAI and
    background albedo are the monthly climatologies at the run's date
    (WRF real's ``monthly_interp_to_date``), the deep-soil temperature is
    the 1-degree climatology corrected to the MODEL's terrain
    (``SOILTEMP - 0.0065 * HGT`` on land, real.exe's rule against the
    terrain the run actually carries).
    """
    from woof.globe.core.landuse import initialize_landuse, reconciled_soil_category
    from woof.globe.core.noah import load_tables, noah_initial_snow_albedo, pack_params
    from woof.static.build import monthly_interp_to_date

    convention = CategoryConvention.from_landuse_attrs(provenance["landuse"])
    iswater = convention.water_category
    islake = convention.lake_category
    isice = convention.ice_category
    mminlu = convention.landuse_dataset
    lu_index = np.asarray(fields["LU_INDEX"], dtype=np.float64)
    landmask = np.asarray(fields["LANDMASK"], dtype=np.float64)
    shape = lu_index.shape
    lat = np.asarray(latitude_deg, dtype=np.float64).reshape(-1, 1)
    if lat.shape[0] != shape[0]:
        raise ValueError("latitude table does not match the statics rows")
    luf = np.asarray(fields["LANDUSEF"], dtype=np.float64)
    water_fraction = luf[iswater - 1] + luf[islake - 1]
    # The crate's LANDMASK is 0 where that water fraction is >= 0.5 and
    # its LU_INDEX is the dominant land type on land cells and the
    # dominant water type on water cells (static-fields fields.rs, the
    # geogrid rule); the runtime's water test is land_fraction <= 0.5.
    # The fraction is placed on the side of one half the mask names so
    # the two agree in every precision (LAND_FRACTION_MARGIN).
    land = landmask > 0.5
    land_fraction = consistent_land_fraction(1.0 - water_fraction, land)
    zeros = np.zeros(shape, dtype=np.float32)
    xice = zeros if sea_ice_fraction is None else np.asarray(
        sea_ice_fraction, dtype=np.float32)
    snow = zeros if snow_water_kg_m2 is None else np.asarray(
        snow_water_kg_m2, dtype=np.float32)
    for name, value in (("sea_ice_fraction", xice), ("snow_water_kg_m2", snow)):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} is not a finite {shape} plane")
    frozen = frozen_water_columns(xice)
    soil_t = np.asarray(soil_temperature_k, dtype=np.float64)
    skin = np.asarray(skin_temperature_k, dtype=np.float64)
    common = dict(
        soil_type=fields["SCT_DOM"], landmask=landmask, snow=snow,
        xice=xice, valid_time=valid_time, mminlu=mminlu, iswater=iswater,
        islake=islake, isice=isice, isoilwater=ISOILWATER,
        isoilice=ISOILICE, soil_temperature=soil_t, sst=skin,
    )
    north = initialize_landuse(lu_index, cen_lat=1.0, **common)
    south = initialize_landuse(lu_index, cen_lat=-1.0, **common)
    northern = lat >= 0.0

    def by_hemisphere(name):
        return np.where(northern, getattr(north, name), getattr(south, name))

    ivgtyp = np.asarray(north.ivgtyp, dtype=np.int32)
    isltyp = np.asarray(north.isltyp, dtype=np.int32)
    # The rulebook keeps the crate's land/water split (lake folds to
    # water on water cells, a land cell carrying water soil keeps its land
    # use and takes soil 8, as real.exe matches it) and freezes the sea-ice columns over (the ice
    # class on water cells at or above the threshold), so the category
    # water test and the runtime's xland test name the same columns.
    # Checked, not assumed: a column on the wrong side would be
    # integrated by Noah with VEGPARM's water row, or skipped while
    # sfclay runs it as land.
    disagree = (ivgtyp == iswater) != ~(land | frozen)
    if np.any(disagree):
        rows, columns = np.nonzero(disagree)
        raise ValueError(
            f"{int(disagree.sum())} column(s) disagree between the static "
            "land mask and the land-use category (first at row "
            f"{int(rows[0])}, column {int(columns[0])}: LANDMASK "
            f"{landmask[rows[0], columns[0]]:g}, category "
            f"{int(ivgtyp[rows[0], columns[0]])}, water category {iswater}); "
            "the cache was not built by the static-fields library's "
            "landmask/dominant-category rule -- rebuild it with the statics door")
    bottom = reconciled_soil_category(
        lu_index, soil_type=fields["SCB_DOM"], xice=xice, iswater=iswater,
        islake=islake, isice=isice, isoilwater=ISOILWATER, isoilice=ISOILICE,
        soil_temperature=soil_t, sst=skin)
    greenfrac = np.asarray(fields["GREENFRAC"], dtype=np.float64)
    vegetation = np.clip(monthly_interp_to_date(greenfrac, valid_time), 0.0, 1.0)
    lai = np.clip(monthly_interp_to_date(
        np.asarray(fields["LAI12M"], dtype=np.float64), valid_time), 0.0, None)
    background = np.clip(monthly_interp_to_date(
        np.asarray(fields["ALBEDO12M"], dtype=np.float64), valid_time) / 100.0,
        0.0, 1.0)
    params = pack_params(load_tables(mminlu=mminlu))
    snow_albedo = noah_initial_snow_albedo(
        fields["SNOALB"], ivgtyp, params, rdmaxalb=True)
    soiltemp = np.asarray(fields["SOILTEMP"], dtype=np.float64)
    terrain = np.asarray(terrain_height_m, dtype=np.float64)
    deep = np.where(land, soiltemp - TMN_LAPSE_K_PER_M * terrain, soil_t[-1])
    resolved = {
        "land_fraction": land_fraction,
        "landuse_category": ivgtyp.astype(np.float64),
        "soil_category_top": isltyp.astype(np.float64),
        "soil_category_bottom": np.asarray(bottom, dtype=np.float64),
        "vegetation_fraction": vegetation,
        "vegetation_fraction_min": np.clip(greenfrac.min(axis=0), 0.0, 1.0),
        "vegetation_fraction_max": np.clip(greenfrac.max(axis=0), 0.0, 1.0),
        "leaf_area_index": lai,
        "background_albedo": background,
        "snow_albedo": np.asarray(snow_albedo, dtype=np.float64),
        "deep_soil_temperature_k": deep,
        "albedo": np.asarray(by_hemisphere("albedo"), dtype=np.float64),
        "emissivity": np.asarray(by_hemisphere("emiss"), dtype=np.float64),
        "roughness_m": np.where(
            (ivgtyp == isice) & frozen, float(convention.sea_ice_roughness_m),
            np.asarray(by_hemisphere("z0"), dtype=np.float64)),
        # The lake class's share of the column (lake_columns): carried so
        # the runtime can tell an inland lake from the ocean after the
        # rulebook folded the lake category into water.
        "lake_fraction": np.clip(luf[islake - 1], 0.0, 1.0),
    }
    for name, value in resolved.items():
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"resolved static {name} is not a finite {shape} plane")
    detail = {
        "valid_time": valid_time.isoformat(),
        "season_northern": int(north.season),
        "season_southern": int(south.season),
        "land_fraction": (
            "1 - LANDUSEF water and lake fractions, held >= 0.5 + 2^-23 on "
            "LANDMASK land and <= 0.5 on water"),
        "deep_soil_temperature": (
            "SOILTEMP - 0.0065 * model terrain on land, bottom soil layer "
            "over water"),
        "land_columns": int(land.sum()),
        "columns": int(land.size),
        "sea_ice_columns": int(frozen.sum()),
        "sea_ice_rule": (
            f"ice class {isice} and ice soil {ISOILICE} on water columns "
            f"with analysed sea-ice fraction >= {SEA_ICE_THRESHOLD}"),
        "snow_covered_land_columns": int((land & (snow >= 10.0)).sum()),
        "snow_rule": "LANDUSE.TBL snow-covered albedo on land holding >= 10 kg/m2",
        "ice_class_columns": int((ivgtyp == isice).sum()),
        "sea_ice_roughness_rule": (
            f"roughness {convention.sea_ice_roughness_m} m on the {int(frozen.sum())} sea-ice "
            f"columns (ice class {isice} on water at or above {SEA_ICE_THRESHOLD}), in place of "
            "LANDUSE.TBL's value; land ice keeps the table's"),
        "convention": convention.as_metadata("real"),
    }
    return resolved, detail


def real_convention(cache_provenance: dict) -> CategoryConvention:
    """The convention a real cache's categories index."""
    return CategoryConvention.from_landuse_attrs(cache_provenance["landuse"])


def synthetic_surface_statics(land_fraction, soil_temperature_k,
                              sea_ice_fraction=None) -> dict[str, np.ndarray]:
    """The constant planet, as arrays shaped like ``land_fraction``.

    Kept for smoke and test configurations only; the door and the
    receipt name it as synthetic.  Albedo and roughness are the former
    land-fraction formulas.  The one vegetation and soil class sit on the
    land columns; the columns the runtime treats as water
    (:func:`water_columns`) carry the water category and water soil, as
    WRF's real.exe leaves every water column, and the returned
    ``land_fraction`` is the input held on the side of one half those
    categories name (:func:`consistent_land_fraction`).
    """
    lf = np.asarray(land_fraction, dtype=np.float64)
    soil = np.asarray(soil_temperature_k, dtype=np.float64)
    if soil.ndim != 3 or soil.shape[1:] != lf.shape:
        raise ValueError("synthetic statics need a (nsoil, nlat, nlon) soil temperature")
    water = water_columns(lf)
    lf = consistent_land_fraction(lf, ~water)
    # Sea ice freezes water columns over (ice class, ice soil, sea-ice
    # albedo and roughness) without moving their land fraction, exactly
    # as the real rulebook does.
    frozen = (np.zeros(lf.shape, dtype=bool) if sea_ice_fraction is None
              else frozen_water_columns(np.asarray(sea_ice_fraction)))
    water = water & ~frozen
    full = lambda value: np.full(lf.shape, float(value), dtype=np.float64)  # noqa: E731
    convention = SYNTHETIC_CONVENTION
    category = np.where(
        water, float(convention.water_category), float(SYNTHETIC_VEGETATION_CATEGORY))
    category = np.where(frozen, float(convention.ice_category), category)
    soil_category = np.where(
        water, float(convention.water_soil_category), float(SYNTHETIC_SOIL_CATEGORY))
    soil_category = np.where(frozen, float(convention.ice_soil_category), soil_category)
    albedo = np.where(frozen, SYNTHETIC_SEA_ICE_ALBEDO, 0.08 + 0.12 * lf)
    roughness = np.where(frozen, convention.sea_ice_roughness_m, 1.0e-4 + 0.08 * lf)
    return {
        "land_fraction": lf,
        "landuse_category": category,
        "soil_category_top": soil_category.copy(),
        "soil_category_bottom": soil_category.copy(),
        "vegetation_fraction": lf.copy(),
        "vegetation_fraction_min": full(0.0),
        "vegetation_fraction_max": full(1.0),
        "leaf_area_index": full(SYNTHETIC_LEAF_AREA_INDEX),
        "background_albedo": albedo.copy(),
        "snow_albedo": full(SYNTHETIC_SNOW_ALBEDO),
        "deep_soil_temperature_k": soil[-1].copy(),
        "albedo": albedo,
        "emissivity": full(SYNTHETIC_EMISSIVITY),
        "roughness_m": roughness,
        "lake_fraction": full(0.0),
    }


def synthetic_provenance(options: StaticsOptions) -> dict:
    return {
        "source": "synthetic",
        "declared": bool(options.declared),
        "vegetation_category": SYNTHETIC_VEGETATION_CATEGORY,
        "soil_category": SYNTHETIC_SOIL_CATEGORY,
        "water_category": SYNTHETIC_CONVENTION.water_category,
        "water_soil_category": SYNTHETIC_CONVENTION.water_soil_category,
        "leaf_area_index": SYNTHETIC_LEAF_AREA_INDEX,
        "snow_albedo": SYNTHETIC_SNOW_ALBEDO,
        "emissivity": SYNTHETIC_EMISSIVITY,
        "albedo": "0.08 + 0.12 * land_fraction",
        "roughness_m": "1e-4 + 0.08 * land_fraction",
        "deep_soil_temperature": "bottom soil layer",
        "convention": SYNTHETIC_CONVENTION.as_metadata("synthetic"),
    }


def real_provenance(options: StaticsOptions, npz_path: Path, cache: dict,
                    detail: dict) -> dict:
    """The receipt's statics row: where the fields came from, without the
    per-sector coverage receipts (those live in the cache sidecar)."""
    coverage = {}
    for role, receipts in cache.get("coverage", {}).items():
        coverage[role] = {
            "sectors": len(receipts),
            "status": sorted({str(r.get("status")) for r in receipts}),
            "required_cells": int(sum(int(r.get("required_cells", 0)) for r in receipts)),
            "covered_cells": int(sum(int(r.get("covered_cells", 0)) for r in receipts)),
            "required_tiles": int(sum(int(r.get("required_tile_count", 0)) for r in receipts)),
        }
    return {
        "source": "real",
        "declared": bool(options.declared),
        "cache": str(npz_path),
        "cache_self_sha256": cache.get("self_sha256"),
        "npz_sha256": cache.get("npz_sha256"),
        "truncation": cache.get("truncation"),
        "nlat": cache.get("nlat"), "nlon": cache.get("nlon"),
        "geog_root": cache.get("geog_root"),
        "geog_data_res": cache.get("geog_data_res"),
        "datasets": cache.get("datasets"),
        "landuse": cache.get("landuse"),
        "builder": cache.get("builder"),
        "built_at_utc": cache.get("built_at_utc"),
        "coverage": coverage,
        **detail,
    }


def statics_sentence(cfg) -> str:
    """The one line the run door prints about its statics."""
    options = cfg.statics
    if options.source != "real":
        how = "declared in [statics]" if options.declared else "analytic default"
        return (
            f"statics synthetic ({how}): one vegetation class "
            f"{SYNTHETIC_VEGETATION_CATEGORY} and one soil class "
            f"{SYNTHETIC_SOIL_CATEGORY} on land (water class "
            f"{SYNTHETIC_CONVENTION.water_category} on water), LAI "
            f"{SYNTHETIC_LEAF_AREA_INDEX}, snow albedo {SYNTHETIC_SNOW_ALBEDO}, "
            "deep soil = bottom soil layer, albedo and roughness from the "
            "land fraction")
    from woof.globe.spectral.grid import GaussianGrid

    grid = GaussianGrid.create(cfg.truncation, nlat=cfg.nlat, nlon=cfg.nlon,
                               dealias_factor=cfg.dealias_factor)
    npz_path, _ = cache_paths(options, grid)
    state = "present" if npz_path.is_file() else "MISSING"
    return (f"statics real (WPS_GEOG {'+'.join(options.tokens())}, cache "
            f"{npz_path} {state})")


__all__ = [
    "CACHED_FIELDS", "GEOG_ROLES", "LAND_FRACTION_MARGIN", "SEA_ICE_THRESHOLD",
    "STATICS_SCHEMA",
    "STATICS_SOURCES", "SURFACE_STATICS_METADATA_KEY",
    "SURFACE_STATIC_FIELDS", "SYNTHETIC_CONVENTION",
    "SYNTHETIC_LEAF_AREA_INDEX", "SYNTHETIC_SNOW_ALBEDO",
    "SYNTHETIC_SOIL_CATEGORY", "SYNTHETIC_VEGETATION_CATEGORY",
    "CategoryConvention", "StaticsOptions", "build_statics", "cache_paths",
    "consistent_land_fraction", "lake_columns", "load_statics", "parse_utc", "read_cache",
    "real_convention", "real_provenance", "resolve_geog_root",
    "resolve_surface_statics", "statics_sentence", "surface_statics_metadata",
    "frozen_water_columns", "synthetic_provenance", "synthetic_surface_statics",
    "water_columns", "write_cache", "xland_plane",
]
