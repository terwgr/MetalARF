"""Real-analysis initial state for WOOF global.

Builds the moist hybrid cold start from a decoded global analysis, replacing
nothing about the model: the return contract is identical to
``analytic_initial_state`` plus a provenance document for the run receipt.

The decode is the engine's, reached through
:mod:`woof.globe.mapped_source_compat` rather than by calling
``woof.mapped_source.decode_mapped_source`` here.  That module is where the
two published-engine differences that stop a real-analysis run are
translated -- where the frame stream's scratch is placed, and the one mapping
rule a published validator narrowed -- and it records both in a receipt block
this function carries into the provenance under ``decode``.  A run therefore
states how it was obtained instead of leaving a reader to infer it from a
version number.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

from woof.globe.spectral.vector import VorticityDivergenceOperator

from .constants import GRAVITY_M_S2, KAPPA, REFERENCE_PRESSURE_PA
from .state import ArwenGlobalState, MoistHybridState, PhysicsState, SurfaceState
from .surface_seeding import (
    SEEDED_ANALYSIS_FIELDS, land_ice_initial_skin_node,
    sea_ice_initial_soil_temperature, seed_surface_from_analysis,
)

DRY_AIR_GAS_CONSTANT = 287.0
#: The surface groups a secondary analysis of the same valid time may
#: supply when the primary product lacks any field of the group.  A group
#: is taken WHOLE from one source: the seeding reads sea-ice concentration
#: with its thickness and snow water with its depth as pairs (a density
#: check settles the snow unit by value), and a pair read half from each
#: product would be checked against itself.  Nothing atmospheric is ever
#: filled: temperature, humidity, wind, surface pressure and terrain are
#: the analysis, and a product lacking one is not an initial state.
FILL_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("soil", ("soil_temperature", "volumetric_soil_moisture")),
    ("snow", ("snow_water_equivalent", "snow_depth")),
    ("sea_ice", ("sea_ice_fraction", "sea_ice_thickness")),
)
_ATMOSPHERIC_FIELDS = (
    "air_temperature", "specific_humidity", "eastward_wind",
    "northward_wind", "surface_pressure", "terrain_height",
)
_REQUIRED_FIELDS = (
    "air_temperature",
    "specific_humidity",
    "eastward_wind",
    "northward_wind",
    "surface_pressure",
    "terrain_height",
    "skin_temperature",
    "land_fraction",
    "soil_temperature",
    "volumetric_soil_moisture",
    # The cold-start surface seeding (surface_seeding.py): sea ice and
    # snow are read from the analysis, never zero-filled.
    *SEEDED_ANALYSIS_FIELDS,
)
_SOIL_LAYERS = 4
#: Noah's snow store, depth and cover flag in the physics namespace
#: (native_state.PersistentNativeState seeds its own from these when the
#: cold state carries them), from the seeded planes of the same name.
_NOAH_SNOW_ARRAYS = (
    ("noah_snow", "snow_water_kg_m2"),
    ("noah_snowh", "snow_depth_m"),
    ("noah_snowc", "snow_cover"),
)


#: This package's own copy of the source mappings the model names.
#:
#: A source is metadata by rule, so a mapping belongs in the ENGINE's
#: authority table and that is where this resolver looks first.  It is not
#: where a published engine has them: woof 2.7.0's `woof/authorities/`
#: carries none of the six this model reads, so every shipped GDAS
#: experiment refused at its first door with `found 0 (none)`, and the ATMS
#: columns, the radiation reference cover and both surface-energy products
#: had no mapping document to read (the flux ladder reads its one as a
#: selector table rather than decoding through it, and needs it just as
#: much).  Waiting for the engine to take the rows
#: is not a shipping state, so the bytes ride here too, unchanged, and
#: retire themselves: the day the engine publishes a row, the engine's copy
#: is the one that answers and this one is never opened.
PACKAGE_AUTHORITIES_DIR = Path(__file__).resolve().parent / "data" / "authorities"


class MappingResolution(NamedTuple):
    """One resolved mapping file, and which table answered for it."""

    #: The file to hand the decoder.
    path: Path
    #: ``"engine"``, ``"package"`` or ``"path"`` (the caller named a file).
    origin: str
    #: The engine's copy, when its table carries this mapping.
    engine_path: Path | None
    #: This package's copy, when it carries this mapping.
    package_path: Path | None


class MappingTablesDisagree(ValueError):
    """Both authority tables carry one spec, with different bytes.

    Its own class rather than a message to match, because
    ``woof global doctor`` prints a different row for it than for a spec
    NEITHER table answers, and telling a reader "no authority table answers
    it" when both answered and disagreed sends them looking for a missing
    file instead of a moved one.  A ``ValueError`` still, so every caller
    that refuses on the resolver's word keeps refusing.
    """


def _engine_authorities_dir() -> Path:
    """The engine's ``authorities`` directory, resolved through the engine.

    It used to be reached as ``Path(__file__).parent.parent / "authorities"``,
    which was true only while this package lived inside ``gpuwm/``.  From a
    distribution of its own that expression names ``site-packages/authorities``
    and matches nothing, so every bare source id fails to resolve and a run
    dies before it allocates.  Asking ``woof`` where it is keeps the source
    table in one place, which is what a bare id is for.
    """
    import woof

    return Path(woof.__file__).resolve().parent / "authorities"


def _engine_authorities_table_label() -> str:
    try:
        return str(_engine_authorities_dir())
    except Exception as failure:  # pragma: no cover - a broken engine install
        return f"unavailable: {failure}"


def _authority_matches(directory: Path, spec: str) -> list[Path]:
    """Every mapping in ``directory`` a spec names, by file name or by id.

    Two spellings reach the callers and both are answered here, so that
    nobody answers one of them by hand.  A FILE NAME names one file
    exactly; a BARE ID is the family prefix a config carries and is
    globbed.  The suite once answered the id form by dropping its last
    segment, which turned ``gdas-global`` into ``gdas`` -- a family that
    matches three different products -- and reported a mapping present
    that nobody had asked for.
    """

    try:
        if not directory.is_dir():
            return []
    except OSError:  # pragma: no cover - an unreadable site-packages
        return []
    # BOTH QUESTIONS, IN THIS ORDER, FOR EITHER SPELLING.  The authority
    # tables hold two shapes of name: a file whose name ends at the id
    # (`rw-wps-gfs-pgrb2-0p25-cloud-cover.mapping.json`) and a file the
    # family glob finds (`rw-wps-gdas-global-analysis-grib2.mapping.json`
    # answers `gdas-global`).  Asking only one of the two reports a gap on a
    # table that carries the row: the glob alone misses the first shape,
    # because `rw-wps-rw-wps-gfs-pgrb2-0p25-cloud-cover-*` matches nothing,
    # and the file name alone misses the second.  Both spellings therefore
    # reach the same file, which is what lets `woof global doctor` and the
    # source registry label every row with its id while the readers that
    # open one by name -- `radiation_scorecard.REFERENCE_MAPPING`,
    # `microwave.columns.MAPPING_NAME` -- keep resolving, and it is why the
    # registry can publish the file name as an ALIAS of the id.
    stem = (spec[:-len(".mapping.json")] if spec.endswith(".mapping.json")
            else spec)
    named = directory / f"{stem}.mapping.json"
    if named.is_file():
        return [named]
    return sorted(directory.glob(f"rw-wps-{stem}-*.mapping.json"))


def _digest(path: Path) -> str:
    """The sha256 of one mapping file, as the run receipts record it."""

    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_analysis_mapping_row(spec: str) -> MappingResolution:
    """Resolve a mapping spec, saying which authority table answered.

    THE ORDER IS THE RULE, and it is why this package carrying a copy does
    not fork the source table.  THE ENGINE IS ASKED FIRST, every time, for
    every spec: the engine publishes the source table, adding a source is
    metadata rather than code, and the day a published engine carries one
    of these rows its copy is the one every command opens.  This package's
    copy answers only what the engine's table does not have, and
    ``woof global doctor`` prints which of the two answered each row.

    THE BREAKAGE THIS PREVENTS.  Two copies of one source mapping on one
    machine decide between them what a forecast was initialized from while
    every receipt records the same file name.  So when BOTH tables carry a
    spec and the bytes differ, this refuses by name rather than picking a
    winner: the engine's row has moved past the one this model was graded
    with, and continuing would grade a run against a source table nobody
    looked at.  Identical bytes are not a conflict -- that is the row
    having landed, and the engine's copy is returned.

    A spec that NAMES A PATH wins outright and comes back as
    ``origin="path"``, resolved to an absolute path so the receipt names a
    file a second machine can open: a caller who typed a path meant that
    file, and `./name` in a receipt names none.  Naming a
    path means a directory part, an absolute path, or the explicit ``./``
    spelling; a BARE mapping file name is not one, it is one of the two keys
    the authority tables answer, and it falls through to them whether or not
    a file of that name happens to sit in the working directory.

    THE BREAKAGE THAT DISTINCTION PREVENTS.  Four of the specs this package
    names in its own source are bare file names -- the ATMS columns, the
    radiation scorecard's reference cover, and the two doctor rows that
    print them.  While any existing file of that name won on the working
    directory alone, ``woof global microwave columns`` run from a directory
    a fetch had populated could decode through a document neither authority
    table publishes; the receipt would record a bare relative name, and the
    both-tables-differ refusal below could not fire, because only one table
    was ever asked.
    """

    candidate = Path(spec)
    named_a_path = (
        len(candidate.parts) > 1
        or candidate.is_absolute()
        or spec.startswith(("./", "../"))
        or (os.name == "nt" and spec.startswith((".\\", "..\\")))
    )
    if named_a_path:
        if candidate.exists():
            # RESOLVED, not as typed.  This path goes into the run receipt
            # beside the mapping digest, and `./name` recorded as `name` is
            # a receipt a second machine cannot follow: it names whatever
            # file of that name sits in whatever directory the reader
            # happens to be in.  The caller still gets the file they typed.
            return MappingResolution(candidate.resolve(), "path", None, None)
        # A spec that named a path named a file, and the file is not there.
        raise FileNotFoundError(f"analysis mapping {candidate} does not exist")

    try:
        engine_matches = _authority_matches(_engine_authorities_dir(), spec)
    except Exception:
        # The engine failing to import is a separate finding with its own
        # sentence (`woof global doctor`); it must not turn into "this
        # mapping does not exist".
        engine_matches = []
    package_matches = _authority_matches(PACKAGE_AUTHORITIES_DIR, spec)

    for matches, where in ((engine_matches, "the engine's authority table"),
                           (package_matches, "this package's carried copies")):
        if len(matches) > 1:
            names = ", ".join(path.name for path in matches)
            raise ValueError(
                f"analysis mapping id {spec!r} must match exactly one "
                f"authority mapping, found {len(matches)} ({names}) in "
                f"{where}; pass an explicit mapping path instead"
            )

    engine_path = engine_matches[0] if engine_matches else None
    package_path = package_matches[0] if package_matches else None
    if engine_path is None and package_path is None:
        raise ValueError(
            f"analysis mapping id {spec!r} must match exactly one authority "
            f"mapping, found 0 (none) in the engine's authority table "
            f"({_engine_authorities_table_label()}) or in this package's "
            f"carried copies ({PACKAGE_AUTHORITIES_DIR}); pass an explicit "
            "mapping path instead"
        )
    if engine_path is None:
        return MappingResolution(package_path, "package", None, package_path)
    if package_path is not None and (engine_path.read_bytes()
                                     != package_path.read_bytes()):
        # NAMED WITH DIGESTS, not just "they differ": the reader's next
        # question is whether the engine's row is one they have seen, and
        # a refusal that cannot be compared to a receipt is a dead end.
        raise MappingTablesDisagree(
            f"analysis mapping {spec!r} is carried twice with different "
            f"bytes: the engine's {engine_path} "
            f"(sha256 {_digest(engine_path)}) and this package's "
            f"{package_path} (sha256 {_digest(package_path)}).\n"
            "The engine publishes the source table and its row wins, but a "
            "row that has MOVED means a run decoded through it is not the "
            "run this model was graded against, and picking a winner "
            "silently is how that goes unnoticed.\n"
            "Install a woof whose row matches, or name the mapping file "
            "you mean by path.")
    return MappingResolution(engine_path, "engine", engine_path, package_path)


def resolve_analysis_mapping(spec: str) -> Path:
    """Resolve a mapping spec to a mapping JSON path.

    A spec that names a path wins, and is refused by name when that path is
    not there.  Otherwise the spec is a bare source id or a bare mapping
    file name, answered by :func:`resolve_analysis_mapping_row`:
    the engine's authority table first, this package's carried copies
    second, so that adding a source stays declarative metadata rather than
    code and an engine that has taken the row is the one that answers.
    """

    return resolve_analysis_mapping_row(spec).path


class _FilledField:
    """One plane (or stack) of the composite frame with the name of the
    product it came from."""

    __slots__ = ("values", "source")

    def __init__(self, values, source: str):
        self.values = values
        self.source = source


class CompositeFrame:
    """The primary analysis frame with the surface groups it lacks taken
    from a secondary frame of the same valid time (``fill_frame``).

    Carries the primary's grid, levels, times and digests; ``fields``
    maps every name to an object with ``.values`` and ``.source``
    (``"primary"`` or ``"fill"``); ``fill`` is the receipt block naming
    every group's source and the reason.
    """

    def __init__(self, primary, fields: dict, fill: dict):
        self.latitude = primary.latitude
        self.longitude = primary.longitude
        self.vertical_kind = getattr(primary, "vertical_kind", "pressure")
        self.vertical_values = primary.vertical_values
        self.fields = fields
        self.fill = fill
        for name in ("mapping_sha256", "input_sha256", "source_cycle", "valid_time"):
            setattr(self, name, getattr(primary, name, None))


class _TargetGrid:
    __slots__ = ("latitude_deg", "longitude_deg")

    def __init__(self, latitude_deg, longitude_deg):
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg


def _onto_primary_grid(values: np.ndarray, source_lat, source_lon,
                       target_lat, target_lon) -> np.ndarray:
    """A secondary product's plane (or stack) on the primary product's
    grid.  Identical grids (the two 0.25-degree products differ only in
    latitude order and the longitude origin) are re-indexed exactly, mask
    included; otherwise the plane is bilinearly regridded with the weights
    renormalised over the finite corners, so a masked source point (a snow
    bitmap over water) stays masked where no finite corner reaches and
    never bleeds a NaN into a land column."""
    field = np.asarray(values, dtype=np.float64)
    slat = np.asarray(source_lat, dtype=np.float64)
    slon = np.asarray(source_lon, dtype=np.float64)
    tlat = np.asarray(target_lat, dtype=np.float64)
    tlon = np.asarray(target_lon, dtype=np.float64)
    same_lat = slat.size == tlat.size and (
        np.allclose(slat, tlat, atol=1.0e-9)
        or np.allclose(slat[::-1], tlat, atol=1.0e-9)
    )
    if same_lat and slon.size == tlon.size:
        # The longitude ring may start anywhere; find the source column
        # under the target's first column and require the rest to line up.
        offsets = np.mod(slon - tlon[0] + 180.0, 360.0) - 180.0
        shift = int(np.argmin(np.abs(offsets)))
        rolled = np.roll(slon, -shift)
        same_lon = abs(offsets[shift]) < 1.0e-9 and np.allclose(
            np.mod(rolled - tlon + 180.0, 360.0) - 180.0, 0.0, atol=1.0e-9
        )
    else:
        same_lon = False
    if same_lat and same_lon:
        if not np.allclose(slat, tlat, atol=1.0e-9):
            field = field[..., ::-1, :]
        if shift:
            field = np.roll(field, -shift, axis=-1)
        return field
    regrid = _global_regridder(slat, slon, _TargetGrid(tlat, tlon))
    finite = np.isfinite(field)
    weight = regrid(finite.astype(np.float64))
    total = regrid(np.where(finite, field, 0.0))
    out = np.full(total.shape, np.nan)
    np.divide(total, weight, out=out, where=weight > 1.0e-9)
    return out


#: How far (in source cells) a filled plane's bitmap is grown over the
#: primary product's land where the two land masks disagree: eight passes
#: of a 3 x 3 neighbourhood, two degrees on the 0.25-degree products.  A
#: primary-land point still masked beyond that reach lies inside water the
#: second product draws for more than two degrees around (the IFS calls
#: 98 such cells land on 2026-09-01 00Z where the GDAS snow bitmap is
#: water: inland lakes and shelf edges); it starts snow-free and the
#: receipt counts and locates it (``assumed_snow_free_points``).
COASTAL_FILL_PASSES = 8

#: The open-water skin's search reach, in analysis cells: a run water
#: column whose bilinear stencil holds none of the analysis's own water
#: points reads the analysis water values grown outward over its land this
#: many passes at most (the reach doubles from one pass until the stencil
#: is filled).  At 0.25 degrees, 512 passes cross any continent, so the
#: cap is a guarantee that the search ends, not a reach anything needs:
#: the GDAS 2026-09-30 12Z analysis on the T255 grid searched 220 columns
#: (Antarctic ice-shelf water the analysis calls land) and reached all of
#: them in 7 passes.
OPEN_WATER_SKIN_SEARCH_PASSES = 512


def open_water_skin_temperature(skin_src, land_src, regrid, open_water, *,
                                latitude_deg=None, longitude_deg=None):
    """The skin temperature of the run's open-water columns, from the
    analysis's OWN water points only.

    An open-water column starts from this skin and the ocean holds it for
    the whole run (no ocean model: Noah skips it, the surface layer reads
    it as the sea surface temperature; an inland lake's skin then follows
    its own surface energy budget, native_runtime._lake_surface_step), so
    that skin must be a water temperature.  The plain bilinear regrid of the analysis skin is not
    one wherever the run's land fraction calls a column water and the
    analysis calls a point of its stencil land: the stencil mixes in a
    land skin, at the analysis hour.  On the GDAS 2026-09-30 12Z analysis
    the north basin of Lake Turkana (4.45 N, 36.09 E, land fraction
    0.485) started at 321.1 K, the 15:00 desert ground around the lake,
    where the one analysis water point in its stencil read 300.8 K; held
    for the run, a 48 C saturated lake under the Turkana jet evaporated
    1.2e-3 kg/m2/s (about 3000 W/m2 latent) day and night, and the column
    drained its 500 kg/m2 surface reservoir by hour 117 ("native physics
    water closure exceeds the explicit surface reservoir").

    The rule is metgrid's masked interpolation of SST over the analysis
    land-sea mask (WPS METGRID.TBL, masked=land with search): the bilinear
    weights renormalised over the stencil's analysis water points
    (``land_src < 0.5``); a column whose stencil holds none reads the
    analysis water values grown outward over the analysis land
    (:func:`_grow_over_primary_land`, the 3 x 3 finite-neighbour mean)
    until its stencil is filled.  Land and sea-ice columns keep the plain
    regrid (Noah and the frozen-surface step integrate their skins from
    the first call).  ``open_water`` is the run's open-water plane
    (:func:`woof.globe.statics.water_columns`); ``latitude_deg`` and
    ``longitude_deg`` (the target grid's axes) only locate the record's
    largest changes.

    Returns ``(skin, record)``: the open-water skin plane (equal to the
    plain regrid everywhere else) and the provenance record that counts
    and locates what the rule changed.
    """
    skin_src = np.asarray(skin_src, dtype=np.float64)
    water_src = np.asarray(land_src, dtype=np.float64) < 0.5
    plain = regrid(skin_src)
    open_water = np.asarray(open_water, dtype=bool)
    weight = regrid(water_src.astype(np.float64))
    total = regrid(np.where(water_src, skin_src, 0.0))
    has_water = weight > 0.0
    skin = np.where(
        open_water & has_water, total / np.where(has_water, weight, 1.0), plain
    )
    need = open_water & ~has_water
    searched = int(np.count_nonzero(need))
    passes = 0
    if searched and water_src.any():
        grown = np.where(water_src, skin_src, np.nan)
        step = 1
        while True:
            grown, _filled, _left = _grow_over_primary_land(grown, ~water_src, step)
            passes += step
            values = regrid(grown)
            reached = need & np.isfinite(values)
            skin = np.where(reached, values, skin)
            need = need & ~reached
            if not need.any() or passes >= OPEN_WATER_SKIN_SEARCH_PASSES:
                break
            step = min(step * 2, OPEN_WATER_SKIN_SEARCH_PASSES - passes)
    unreached = int(np.count_nonzero(need))
    change = np.where(open_water, skin - plain, 0.0)
    order = np.argsort(np.abs(change), axis=None)[::-1][:5]
    rows, columns = np.unravel_index(order, change.shape)
    record = {
        "rule": (
            "open-water columns take the analysis skin over the analysis's "
            "own water points only: bilinear weights renormalised over the "
            "stencil's water points (land_fraction < 0.5 in the analysis), "
            "and a stencil with none reads the water values grown outward "
            "over the analysis land by the 3 x 3 finite-neighbour mean; "
            "land and sea-ice columns keep the plain bilinear regrid"
        ),
        "open_water_columns": int(np.count_nonzero(open_water)),
        "stencil_with_analysis_land": int(np.count_nonzero(
            open_water & has_water & (weight < 1.0 - 1.0e-12))),
        "searched_columns": searched,
        "search_passes": int(passes),
        "unreached_columns_kept_plain_regrid": unreached,
        "changed_by_more_than_1_k": int(np.count_nonzero(np.abs(change) > 1.0)),
        "largest_cooling_k": float(-min(0.0, float(change.min()))),
        "largest_warming_k": float(max(0.0, float(change.max()))),
        "largest_changes": [
            {"latitude_deg": None if latitude_deg is None else float(np.asarray(latitude_deg)[j]),
             "longitude_deg": None if longitude_deg is None else float(np.asarray(longitude_deg)[i]),
             "row": int(j), "column": int(i),
             "plain_regrid_k": float(plain[j, i]), "open_water_k": float(skin[j, i])}
            for j, i in zip(rows, columns) if change[j, i] != 0.0
        ],
    }
    return skin, record


def _grow_over_primary_land(values: np.ndarray, land: np.ndarray, passes: int):
    """A plane carrying a bitmap (NaN over the second product's water)
    completed over the PRIMARY product's land: every masked point the
    primary calls land takes the mean of its finite 3 x 3 neighbours,
    repeated ``passes`` times so a coast that the two land masks draw a
    few cells apart reads the nearest values of the same plane.  Points
    the primary calls water keep their mask (the seeding reads it as no
    snow on open water).  Returns ``(plane, filled_count, unfilled_count)``
    where ``unfilled`` counts masked primary-land points still out of
    reach."""
    plane = np.array(values, dtype=np.float64, copy=True)
    want = land & ~np.isfinite(plane)
    filled = 0
    for _pass in range(passes):
        if not want.any():
            break
        finite = np.isfinite(plane)
        weight = np.zeros_like(plane)
        total = np.zeros_like(plane)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                shifted = np.roll(np.roll(plane, dy, axis=-2), dx, axis=-1)
                mask = np.roll(np.roll(finite, dy, axis=-2), dx, axis=-1)
                if dy:
                    # Latitude does not wrap; the rolled-in row is not a neighbour.
                    edge = 0 if dy > 0 else -1
                    mask[..., edge, :] = False
                total += np.where(mask, shifted, 0.0)
                weight += mask
        reach = want & (weight > 0)
        plane = np.where(reach, total / np.maximum(weight, 1.0), plane)
        filled += int(np.count_nonzero(reach))
        want = want & ~reach
    return plane, filled, int(np.count_nonzero(want))


def fill_frame(primary, secondary, *, primary_label: str, fill_label: str):
    """The primary frame completed from the secondary where a surface
    group is missing (``FILL_GROUPS``); refuses by name when an
    atmospheric field is missing, when the two products are not valid at
    the same instant, or when the secondary lacks a field it is asked for.
    A group of which the primary carries only part is taken whole from the
    secondary and the unused primary field is named in the receipt."""
    missing_atmosphere = [
        name for name in _ATMOSPHERIC_FIELDS if name not in primary.fields
    ]
    if missing_atmosphere:
        raise ValueError(
            f"primary analysis ({primary_label}) lacks "
            f"{', '.join(missing_atmosphere)}; only the surface groups "
            f"{', '.join(group for group, _ in FILL_GROUPS)} may be filled "
            "from a secondary analysis, the atmosphere is the analysis itself"
        )
    primary_time = getattr(primary, "valid_time", None)
    fill_time = getattr(secondary, "valid_time", None)
    if primary_time is None or fill_time is None or str(primary_time) != str(fill_time):
        raise ValueError(
            f"analysis fill valid time {fill_time} ({fill_label}) is not the "
            f"primary analysis valid time {primary_time} ({primary_label}); "
            "a surface group from another hour would seed ice, snow or soil "
            "of a different day under this atmosphere"
        )
    fields = {
        name: _FilledField(field.values, "primary")
        for name, field in primary.fields.items()
    }
    groups: dict[str, dict[str, object]] = {}
    for group, names in FILL_GROUPS:
        present = [name for name in names if name in primary.fields]
        absent = [name for name in names if name not in primary.fields]
        if not absent:
            groups[group] = {"source": "primary", "fields": list(names)}
            continue
        lacking = [name for name in names if name not in secondary.fields]
        if lacking:
            raise ValueError(
                f"primary analysis ({primary_label}) lacks {', '.join(absent)} "
                f"and the fill analysis ({fill_label}) lacks "
                f"{', '.join(lacking)}; the {group} group is read whole from "
                "one product and neither carries it"
            )
        coastal: dict[str, int] = {}
        assumed: dict[str, dict] = {}
        for name in names:
            plane = _onto_primary_grid(
                secondary.fields[name].values, secondary.latitude,
                secondary.longitude, primary.latitude, primary.longitude,
            )
            if not np.isfinite(plane).all() and "land_fraction" in primary.fields:
                # The second product's bitmap is drawn on ITS land mask;
                # where the primary calls land and the second water, the
                # plane is grown from its finite neighbours (the seeding
                # would otherwise refuse the coast as "masked on land").
                land = np.asarray(primary.fields["land_fraction"].values, dtype=np.float64) >= 0.5
                if plane.ndim == 3:
                    land = np.broadcast_to(land[None], plane.shape)
                plane, filled, unfilled = _grow_over_primary_land(
                    plane, land, COASTAL_FILL_PASSES
                )
                coastal[name] = filled
                if unfilled:
                    # Primary land inside the second product's water for
                    # more than the reach: no value exists in either
                    # product.  The plane starts at zero there (no snow,
                    # no water equivalent), counted and located in the
                    # receipt rather than refused, since 98 cells of lake
                    # and shelf edge are not a reason to refuse a planet.
                    still = land & ~np.isfinite(plane)
                    where = np.argwhere(still.reshape(-1, *still.shape[-2:])[0])
                    lat_axis = np.asarray(primary.latitude, dtype=np.float64)
                    lon_axis = np.asarray(primary.longitude, dtype=np.float64)
                    assumed.setdefault(name, {
                        "points": int(unfilled),
                        "value": 0.0,
                        "latitude_range_deg": [
                            float(lat_axis[where[:, 0]].min()),
                            float(lat_axis[where[:, 0]].max()),
                        ] if where.size else None,
                        "northern_points": int(np.count_nonzero(lat_axis[where[:, 0]] > 0.0)) if where.size else 0,
                        "longitude_range_deg": [
                            float(lon_axis[where[:, 1]].min()),
                            float(lon_axis[where[:, 1]].max()),
                        ] if where.size else None,
                    })
                    plane = np.where(still, 0.0, plane)
            fields[name] = _FilledField(plane, "fill")
        reason = f"primary product carries no {', '.join(absent)}"
        if present:
            reason += (
                f"; its {', '.join(present)} not used, the group is read "
                "whole from one product"
            )
        groups[group] = {"source": "fill", "fields": list(names), "reason": reason}
        if coastal:
            groups[group]["coastal_filled_points"] = coastal
            groups[group]["coastal_fill"] = (
                "masked points the primary land mask calls land took the "
                "mean of their finite 3 x 3 neighbours over "
                f"{COASTAL_FILL_PASSES} passes; points the primary calls "
                "water keep the bitmap"
            )
        if assumed:
            groups[group]["assumed_snow_free_points"] = assumed
            groups[group]["assumed_snow_free"] = (
                "primary land inside the second product's water beyond "
                f"{COASTAL_FILL_PASSES} cells of any value in either "
                "product starts at zero (no snow); counted and located here"
            )
    same_grid = (
        np.size(secondary.latitude) == np.size(primary.latitude)
        and np.size(secondary.longitude) == np.size(primary.longitude)
    )
    fill = {
        "primary": primary_label,
        "fill": fill_label,
        "fill_mapping_sha256": getattr(secondary, "mapping_sha256", None),
        "fill_input_sha256": getattr(secondary, "input_sha256", None),
        "fill_valid_time": None if fill_time is None else str(fill_time),
        "fill_grid": {
            "nlat": int(np.size(secondary.latitude)),
            "nlon": int(np.size(secondary.longitude)),
            "onto_primary": (
                "re-indexed" if same_grid
                else "bilinear, weights renormalised over finite corners"
            ),
        },
        "groups": groups,
        "field_sources": {name: field.source for name, field in fields.items()},
    }
    return CompositeFrame(primary, fields, fill)


def floor_soil_water(soil_moisture, soil_category_top, landuse_category,
                     ice_category: int, land_fraction, *, drysmc=None):
    """Volumetric soil water inside Noah's domain on every land column.

    A land column whose analysed soil water is zero or negative has no
    water for Noah's frozen-soil relation (``noah_frh2o`` divides by
    ``smc - swl``), and the first land call returns non-finite fluxes
    that the surface layer carries into the wind (the IFS open-data
    product writes 0.0 over the Antarctic ice sheet, 3,688 cells at 0.25
    degrees, where the GDAS analysis writes 1.0).  Such a column takes
    WRF's glacial-ice convention, saturated (1.0), when the statics put
    the land-ice class on it, and its soil class's dry limit ``smcdry``
    from the Noah soil table otherwise.  Every other value is untouched,
    bit for bit; a water column is untouched.  Returns ``(soil_moisture,
    record)`` with the counts."""
    if drysmc is None:
        from woof.globe.core.noah import load_tables

        drysmc = np.asarray(load_tables().drysmc, dtype=np.float64)
    moisture = np.asarray(soil_moisture, dtype=np.float64)
    land = np.asarray(land_fraction, dtype=np.float64) >= 0.5
    category = np.rint(np.asarray(soil_category_top, dtype=np.float64)).astype(np.int64)
    dry = drysmc[np.clip(category - 1, 0, drysmc.size - 1)]
    ice = np.rint(np.asarray(landuse_category, dtype=np.float64)).astype(np.int64) == int(ice_category)
    empty = (moisture <= 0.0) & land[None]
    on_ice = empty & ice[None]
    on_soil = empty & ~ice[None]
    if not empty.any():
        return soil_moisture, {
            "rule": "no land column carried zero or negative soil water; nothing changed",
            "land_ice_cells_set_saturated": 0, "soil_cells_set_to_smcdry": 0,
        }
    out = np.where(on_ice, 1.0, np.where(on_soil, np.broadcast_to(dry[None], moisture.shape), moisture))
    return out, {
        "rule": (
            "a land column with zero or negative analysed soil water takes "
            "1.0 on the land-ice class (WRF's glacial convention, the value "
            "the GDAS analysis carries there) and its soil class's smcdry "
            "from the Noah soil table elsewhere; every other column is "
            "untouched"
        ),
        "land_ice_cells_set_saturated": int(on_ice.sum()),
        "soil_cells_set_to_smcdry": int(on_soil.sum()),
    }


def _global_regridder(latitude: np.ndarray, longitude: np.ndarray, grid):
    """Periodic bilinear interpolation weights from a regular global
    lat-lon grid onto the transform's Gaussian grid."""
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    flip = lat[0] > lat[-1]
    if flip:
        lat = lat[::-1]
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    if lat.size < 2 or lon.size < 2:
        raise ValueError("analysis grid must be two-dimensional")
    if np.max(np.abs(dlat - dlat[0])) > 1.0e-6 or np.max(np.abs(dlon - dlon[0])) > 1.0e-6:
        raise ValueError("analysis grid must be regular in latitude and longitude")
    span = float(dlon[0]) * lon.size
    if abs(span - 360.0) > 1.0e-3:
        raise ValueError(
            f"analysis longitude ring covers {span:.4f} degrees, not the "
            "globe; a regional subset cannot initialize the global model"
        )
    target_lat = np.asarray(grid.latitude_deg, dtype=np.float64)
    target_lon = np.asarray(grid.longitude_deg, dtype=np.float64)
    if target_lat[0] < lat[0] - 1.0e-9 or target_lat[-1] > lat[-1] + 1.0e-9:
        raise ValueError(
            "analysis latitudes do not cover the Gaussian grid; the source "
            "must reach both poles"
        )
    fy = np.clip((target_lat - lat[0]) / dlat[0], 0.0, lat.size - 1.0)
    y0 = np.minimum(fy.astype(np.int64), lat.size - 2)
    wy = fy - y0
    fx = np.mod(target_lon - lon[0], 360.0) / dlon[0]
    x0 = np.mod(fx.astype(np.int64), lon.size)
    wx = fx - np.floor(fx)
    x1 = np.mod(x0 + 1, lon.size)

    def regrid(values: np.ndarray) -> np.ndarray:
        field = np.asarray(values, dtype=np.float64)
        if flip:
            field = field[..., ::-1, :]
        yl = y0[:, None]
        yu = (y0 + 1)[:, None]
        a = field[..., yl, x0[None, :]]
        b = field[..., yl, x1[None, :]]
        c = field[..., yu, x0[None, :]]
        d = field[..., yu, x1[None, :]]
        wyc = wy[:, None]
        wxc = wx[None, :]
        return (
            (1.0 - wyc) * ((1.0 - wxc) * a + wxc * b)
            + wyc * ((1.0 - wxc) * c + wxc * d)
        )

    return regrid


def _to_model_levels(values, ln_source, ln_target, *, extrapolate_below=False):
    """Linear-in-ln(p) interpolation from source pressure levels to model
    full levels.  Above the top source level the value is held; below the
    bottom the value is held unless ``extrapolate_below`` continues the
    bottom layer's ln(p) gradient (temperature's lapse continuation)."""
    upper = np.clip(np.searchsorted(ln_source, ln_target), 1, ln_source.size - 1)
    lower = upper - 1
    weight = (ln_target - ln_source[lower]) / (ln_source[upper] - ln_source[lower])
    weight = np.maximum(weight, 0.0)
    if not extrapolate_below:
        weight = np.minimum(weight, 1.0)
    low = np.take_along_axis(values, lower, axis=0)
    high = np.take_along_axis(values, upper, axis=0)
    return low * (1.0 - weight) + high * weight


def surface_virtual_temperature(temperature_src, humidity_src, ln_source, ps_src):
    """Virtual temperature of the air AT the source surface pressure.

    ``temperature_src``/``humidity_src`` are (nlevel, ...) isobaric fields
    on ascending ``ln_source``; the result is their value interpolated in
    ln p to ``ps_src`` (temperature continues the bottom layer's lapse
    below the lowest source level, humidity is held), times
    ``1 + 0.608 q``.  This is the temperature the hypsometric surface
    pressure reduction integrates over: the layer between the source
    surface and the model surface.  The bottom SOURCE level (1000 hPa)
    is not that air over high terrain - it is the analysis's own
    below-ground extrapolation, +14.9 K mean and +35.5 K max against the
    surface-interpolated value over the 1569 T63 columns with ps below
    800 hPa on the 2026-08-30 18Z GDAS frame, which put the reduced ps
    1781 Pa low on the Tibetan column and the initial Z500 -11.2 m mean
    over |terrain change| > 500 m columns (audit 2026-09-01 VTW-3).
    """
    ln_ps = np.log(np.asarray(ps_src, dtype=np.float64))[None]
    temperature = _to_model_levels(
        temperature_src, ln_source, ln_ps, extrapolate_below=True
    )[0]
    humidity = np.clip(_to_model_levels(humidity_src, ln_source, ln_ps)[0], 0.0, None)
    return temperature * (1.0 + 0.608 * humidity)


def analysis_initial_state(cfg, transform, frame=None, statics=None,
                           scratch_destination=None):
    """Return ``(state, surface_geopotential, provenance)`` from a real
    global analysis.  ``frame`` is injectable for unit gates; by default the
    configured GRIB decodes through the Rust mapped-source engine.
    ``statics`` is the ``(fields, provenance)`` pair of a real static cache
    (injectable the same way); by default a real-statics config loads its
    cache for this truncation and refuses with the remedy when it is
    missing.  ``scratch_destination`` is the output directory of the run
    this state starts: the engine stages the decoded frame stream (several
    GB of f64 at full level count) beside it instead of in the system
    temp, which on a tmpfs ``/tmp`` with a user quota killed the decode
    with "Disk quota exceeded (os error 122)"."""
    from .statics import (
        SURFACE_STATICS_METADATA_KEY, SYNTHETIC_CONVENTION, load_statics, real_convention, real_provenance,
        resolve_surface_statics, surface_statics_metadata,
        synthetic_provenance, synthetic_surface_statics,
    )

    mapping_path = resolve_analysis_mapping(cfg.analysis_mapping)
    if cfg.statics.source == "real" and statics is None:
        # Before the GRIB decode: a missing cache is a door refusal, not
        # something found after minutes of decoding.
        from .statics import cache_paths

        statics = load_statics(cfg.statics, transform.grid)
        statics_path = cache_paths(cfg.statics, transform.grid)[0]
    else:
        statics_path = None
    decode_receipts: list[dict] = []
    if frame is None:
        from .mapped_source_compat import decode_through_engine

        grib = Path(cfg.analysis_grib)
        if not grib.exists():
            raise FileNotFoundError(f"analysis GRIB {grib} does not exist")
        # The scratch is the run directory's, not the system temp's: the
        # decoder stages several GB of float64 frames and a quota-limited
        # tmpfs killed that with "Disk quota exceeded (os error 122)".  The
        # seam module places it whichever way the installed engine spells
        # it and records which one in the receipt, so the placement is a
        # printed fact rather than a version to look up.
        decoded = decode_through_engine(
            mapping_path, [grib], scratch_destination=scratch_destination)
        frames = decoded.frames
        decode_receipts.append(decoded.receipt)
        if len(frames) != 1:
            raise ValueError(
                f"analysis initialization needs exactly one valid time, "
                f"decoded {len(frames)}"
            )
        frame = frames[0]
    fill_mapping_path = None
    if getattr(cfg, "analysis_fill_grib", None):
        # A second product of the same hour supplies the surface groups
        # the primary lacks (FILL_GROUPS); decoded through its own mapping,
        # composed before any field is read, every source in the receipt.
        from .mapped_source_compat import decode_through_engine

        fill_mapping_path = resolve_analysis_mapping(cfg.analysis_fill_mapping)
        fill_grib = Path(cfg.analysis_fill_grib)
        if not fill_grib.exists():
            raise FileNotFoundError(
                f"analysis fill GRIB {fill_grib} does not exist"
            )
        decoded_fill = decode_through_engine(
            fill_mapping_path, [fill_grib],
            scratch_destination=scratch_destination)
        fill_frames = decoded_fill.frames
        decode_receipts.append(decoded_fill.receipt)
        if len(fill_frames) != 1:
            raise ValueError(
                "analysis fill needs exactly one valid time, decoded "
                f"{len(fill_frames)}"
            )
        frame = fill_frame(
            frame, fill_frames[0],
            primary_label=mapping_path.name, fill_label=fill_mapping_path.name,
        )

    missing = [name for name in _REQUIRED_FIELDS if name not in frame.fields]
    if missing:
        raise ValueError(
            f"analysis frame lacks required fields: {', '.join(missing)}"
        )
    if getattr(frame, "vertical_kind", "pressure") != "pressure":
        raise ValueError(
            f"analysis vertical kind {frame.vertical_kind!r} is not "
            "'pressure'; only isobaric analyses are supported"
        )
    levels = np.asarray(frame.vertical_values, dtype=np.float64)
    if levels.ndim != 1 or levels.size < 5 or np.any(np.diff(levels) <= 0.0):
        raise ValueError("analysis pressure levels must be ascending top-to-bottom")

    grid = transform.grid
    regrid = _global_regridder(frame.latitude, frame.longitude, grid)

    def field(name: str) -> np.ndarray:
        return regrid(frame.fields[name].values)

    temperature_src = field("air_temperature")
    humidity_src = np.clip(field("specific_humidity"), 0.0, None)
    u_src = field("eastward_wind")
    v_src = field("northward_wind")
    ps_src = field("surface_pressure")
    terrain_src = field("terrain_height")
    for name, values in (
        ("air_temperature", temperature_src),
        ("specific_humidity", humidity_src),
        ("eastward_wind", u_src),
        ("northward_wind", v_src),
        ("surface_pressure", ps_src),
        ("terrain_height", terrain_src),
    ):
        bad = int(np.size(values) - np.count_nonzero(np.isfinite(values)))
        if bad:
            raise ValueError(
                f"analysis field {name} carries {bad} non-finite values "
                "after regridding; a masked or partial-coverage source "
                "cannot initialize the global model"
            )

    # The model's terrain is the spectrally resolved version of the analysis
    # orography; surface pressure moves hypsometrically onto it so the lowest
    # layers stay hydrostatically consistent with the smoothed mountains.
    # The reduction integrates over the air at the source surface, so its
    # virtual temperature is interpolated to ps_src (see
    # surface_virtual_temperature for why the bottom source level is not
    # that air).
    phi_src = GRAVITY_M_S2 * terrain_src
    phi_model = transform.backend.to_numpy(
        transform.inverse(transform.forward(phi_src))
    )
    ln_source = np.log(levels)
    virtual_surface = surface_virtual_temperature(
        temperature_src, humidity_src, ln_source, ps_src
    )
    ps_model = ps_src * np.exp(
        (phi_src - phi_model) / (DRY_AIR_GAS_CONSTANT * virtual_surface)
    )

    pressure = cfg.vertical.pressure(ps_model, transform.backend)
    p_full = transform.backend.to_numpy(pressure["p_full"])
    ln_target = np.log(p_full)

    temperature = _to_model_levels(
        temperature_src, ln_source, ln_target, extrapolate_below=True
    )
    humidity = np.clip(_to_model_levels(humidity_src, ln_source, ln_target), 0.0, None)
    u = _to_model_levels(u_src, ln_source, ln_target)
    v = _to_model_levels(v_src, ln_source, ln_target)

    exner = (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
    theta = temperature / exner

    b = transform.backend
    vector = VorticityDivergenceOperator(transform)
    vorticity, divergence = vector.vordiv_from_wind(
        b.asarray(u, dtype=b.float_dtype), b.asarray(v, dtype=b.float_dtype)
    )
    zeros_grid = np.zeros_like(humidity)
    atmosphere = MoistHybridState(
        vorticity=vorticity,
        divergence=divergence,
        theta=transform.forward(theta),
        log_surface_pressure=transform.forward(np.log(ps_model)),
        # The model's convention is SPECIFIC humidity (GDAS SPFH, kg per kg
        # of moist air), stored as analysed; the native bridge converts to
        # the kernels' dry mixing ratio at its own boundary
        # (physics/native_batch.py) and back, so nothing here re-labels
        # the field (audit 2026-09-01 NB-3).
        qv=transform.forward(humidity),
        # The pgrb2 mapping carries no condensate or number analyses; the
        # model starts condensate-free and lets physics build cloud, which
        # the provenance records as a stated limit of this initialization.
        # These are grid tracers: exact zeros on the Gaussian grid.
        **{
            name: b.asarray(zeros_grid, dtype=b.float_dtype)
            for name in ("qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng")
        },
    )

    land_fraction = np.clip(field("land_fraction"), 0.0, 1.0)
    skin_temperature = field("skin_temperature")
    soil_temperature_src = np.asarray(frame.fields["soil_temperature"].values)
    soil_moisture_src = np.asarray(frame.fields["volumetric_soil_moisture"].values)
    if soil_temperature_src.ndim != 3 or soil_moisture_src.ndim != 3:
        raise ValueError("analysis soil fields must carry a layer dimension")
    # GRIB soil fields are masked over water.  The fill happens in SOURCE
    # space, before regridding, because bilinear weights would otherwise
    # bleed NaN one cell inland along every coastline.
    skin_src = np.asarray(frame.fields["skin_temperature"].values)
    soil_temperature_src = np.where(
        np.isfinite(soil_temperature_src), soil_temperature_src, skin_src[None]
    )
    soil_moisture_src = np.where(
        np.isfinite(soil_moisture_src), soil_moisture_src, 0.25
    )
    soil_temperature = regrid(soil_temperature_src)
    soil_moisture = np.clip(regrid(soil_moisture_src), 0.0, 1.0)

    def _soil_layers(stack: np.ndarray) -> np.ndarray:
        if stack.shape[0] >= _SOIL_LAYERS:
            return np.ascontiguousarray(stack[:_SOIL_LAYERS])
        pad = np.repeat(stack[-1:], _SOIL_LAYERS - stack.shape[0], axis=0)
        return np.concatenate([stack, pad], axis=0)

    soil_temperature = _soil_layers(soil_temperature)
    soil_moisture = _soil_layers(soil_moisture)

    # Sea ice and snow, from the analysis (surface_seeding: unit trap
    # closed by value, bitmap read as no snow on open water only, every
    # missing field refused by name).  The statics rulebook below puts
    # the ice class on the frozen columns and the snow-covered albedo on
    # the snowy land; Noah's store starts from the seeded snow.
    if cfg.statics.source == "real":
        # The crate's land mask is the split the resolved land fraction
        # is held to (statics.consistent_land_fraction), so the seeding's
        # open-water test names the same columns the runtime will.
        seeding_land_fraction = np.asarray(statics[0]["LANDMASK"], dtype=np.float64)
    else:
        seeding_land_fraction = land_fraction
    seeded = seed_surface_from_analysis(
        frame, regrid, target_land_fraction=seeding_land_fraction
    )
    # The frozen columns' heat conduction column (physics/frozen_surface)
    # lives in the four soil-temperature layers.  A sea-ice column starts
    # linear from the analysed skin to the freezing point at the ice
    # bottom, before the statics derive their deep-soil temperature from
    # the bottom layer; a land-ice column's skin node is set once the
    # statics have named the ice class (below).
    soil_temperature = sea_ice_initial_soil_temperature(
        soil_temperature, skin_temperature, seeded.sea_ice_fraction,
        seeded.sea_ice_thickness_m, seeded.snow_depth_m,
    )

    # The static surface fields.  Real: the WPS_GEOG build resolved at the
    # analysis valid time, whose land fraction (the MODIS water and lake
    # fractions) replaces the analysis's bilinear 0/1 LAND regrid so the
    # categories, the mask and Noah's water test agree by construction.
    # Synthetic: the declared constant planet on the analysis land mask.
    if cfg.statics.source == "real":
        valid_time = getattr(frame, "valid_time", None)
        if valid_time is None:
            raise ValueError(
                "real statics resolve monthly climatologies at the analysis "
                "valid time and this frame carries none"
            )
        static_fields, cache_provenance = statics
        resolved, detail = resolve_surface_statics(
            static_fields, cache_provenance, valid_time=valid_time,
            latitude_deg=grid.latitude_deg,
            terrain_height_m=phi_model / GRAVITY_M_S2,
            soil_temperature_k=soil_temperature,
            skin_temperature_k=skin_temperature,
            sea_ice_fraction=seeded.sea_ice_fraction,
            snow_water_kg_m2=seeded.snow_water_kg_m2,
        )
        land_fraction = resolved.pop("land_fraction")
        statics_provenance = real_provenance(
            cfg.statics, statics_path, cache_provenance, detail
        )
        statics_provenance["land_fraction_source"] = "static-water-fraction"
        statics_metadata = surface_statics_metadata(
            "real", real_convention(cache_provenance)
        )
    else:
        resolved = synthetic_surface_statics(
            land_fraction, soil_temperature,
            sea_ice_fraction=seeded.sea_ice_fraction,
        )
        # The regridded analysis mask, held on the side of one half its
        # water columns' categories name (statics.consistent_land_fraction).
        land_fraction = resolved.pop("land_fraction")
        statics_provenance = synthetic_provenance(cfg.statics)
        statics_provenance["land_fraction_source"] = "analysis-land-regrid"
        statics_metadata = surface_statics_metadata("synthetic", SYNTHETIC_CONVENTION)

    # Soil water inside Noah's domain on every land column (floor_soil_water):
    # the real statics name the soil and land-ice classes; the synthetic
    # planet's single class is left as declared.
    soil_water_floor = None
    if cfg.statics.source == "real":
        soil_moisture, soil_water_floor = floor_soil_water(
            soil_moisture, resolved["soil_category_top"],
            resolved["landuse_category"],
            statics_metadata[SURFACE_STATICS_METADATA_KEY]["ice_category"],
            land_fraction,
        )

    # A land-ice column keeps the analysed soil temperatures under a skin
    # node at the analysed skin, capped at melting.
    soil_temperature = land_ice_initial_skin_node(
        soil_temperature, skin_temperature, seeded.sea_ice_fraction,
        resolved["landuse_category"],
        statics_metadata[SURFACE_STATICS_METADATA_KEY]["ice_category"],
    )

    # The open-water skin is a water temperature the ocean holds for the
    # run (open_water_skin_temperature): the analysis's own water points,
    # never a land skin its stencil mixed in.  The columns are the runtime's own
    # open-water test on the planes in the state's precision.
    from .statics import water_columns

    surface_dtype = np.dtype(b.float_dtype)
    open_water_skin, open_water_skin_record = open_water_skin_temperature(
        frame.fields["skin_temperature"].values,
        frame.fields["land_fraction"].values,
        regrid,
        water_columns(
            np.asarray(land_fraction, dtype=surface_dtype),
            sea_ice_fraction=np.asarray(seeded.sea_ice_fraction, dtype=surface_dtype),
        ),
        latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg,
    )

    surface = SurfaceState(
        temperature_k=b.asarray(open_water_skin, dtype=b.float_dtype),
        water_kg_m2=b.xp.full(
            grid.shape, float(cfg.surface_water_kg_m2), dtype=b.float_dtype
        ),
        land_fraction=b.asarray(land_fraction, dtype=b.float_dtype),
        # Ocean keeps a mixed-layer-scale capacity; land gets a ~5 cm soil
        # skin (2e5 J/m2/K).  The former 2e7 land value was five metres of
        # water equivalent: deserts held their skin temperature like an
        # ocean, so a tropical cyclone moving inland never lost its warm
        # surface - measured 992.6 hPa over the Sonoran desert at f021
        # where GFS's forecast for the same valid time bottomed at 1008.5.
        heat_capacity_j_m2_k=b.asarray(
            4.0e7 * (1.0 - land_fraction) + 2.0e5 * land_fraction,
            dtype=b.float_dtype,
        ),
        soil_temperature_k=b.asarray(soil_temperature, dtype=b.float_dtype),
        soil_water_fraction=b.asarray(soil_moisture, dtype=b.float_dtype),
        accumulated_rain_kg_m2=b.xp.zeros(grid.shape, dtype=b.float_dtype),
        accumulated_snow_kg_m2=b.xp.zeros(grid.shape, dtype=b.float_dtype),
        accumulated_graupel_kg_m2=b.xp.zeros(grid.shape, dtype=b.float_dtype),
        sea_ice_fraction=b.asarray(seeded.sea_ice_fraction, dtype=b.float_dtype),
        sea_ice_thickness_m=b.asarray(seeded.sea_ice_thickness_m, dtype=b.float_dtype),
        **{name: b.asarray(value, dtype=b.float_dtype)
           for name, value in resolved.items()},
    )
    # Noah's snow store, depth and cover start from the analysis: the
    # persistent native state takes an array already in the namespace
    # over its zero fill, and the water target the fixer holds is formed
    # from this cold state, so the seeded snow is inside the conservation
    # total from step zero rather than appearing as created water on the
    # first land call.
    planes = seeded.planes()
    physics_arrays = {
        name: b.asarray(planes[source], dtype=b.float_dtype)
        for name, source in _NOAH_SNOW_ARRAYS
    }

    def _json_safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        if hasattr(value, "items"):
            return {str(key): _json_safe(item) for key, item in value.items()}
        return str(value)

    provenance = {
        "mode": "analysis",
        "mapping": str(mapping_path),
        "mapping_sha256": _json_safe(getattr(frame, "mapping_sha256", None)),
        "input_sha256": _json_safe(getattr(frame, "input_sha256", None)),
        "source_cycle": None if getattr(frame, "source_cycle", None) is None
        else str(frame.source_cycle),
        "valid_time": None if getattr(frame, "valid_time", None) is None
        else str(frame.valid_time),
        "analysis_levels": int(levels.size),
        "surface_pressure_adjustment_pa": {
            "max": float(np.max(ps_model - ps_src)),
            "min": float(np.min(ps_model - ps_src)),
        },
        "surface_pressure_reduction": (
            "hypsometric-virtual-temperature-interpolated-in-ln-p-to-"
            "source-surface-pressure"
        ),
        "wind_balance": "none-analysis-winds-taken-as-analysed",
        "condensate": "not-ingested-model-starts-condensate-free",
        "statics": _json_safe(statics_provenance),
        "surface_seeding": _json_safe(seeded.provenance),
        # Every required field's product: the primary mapping's name, or
        # the fill's where a surface group came from the second product.
        "field_sources": {
            name: (
                mapping_path.name
                if getattr(frame.fields[name], "source", "primary") == "primary"
                else (fill_mapping_path.name if fill_mapping_path else "fill")
            )
            for name in _REQUIRED_FIELDS
        },
        # How each decode was obtained: where its scratch went, whether
        # the engine's mapping validator had to be adapted, and which
        # engine version answered.  A run that says nothing about this
        # leaves a reader to infer it from a version number.
        "decode": _json_safe(decode_receipts) or None,
        "fill": _json_safe(getattr(frame, "fill", None)),
        "soil_water_floor": _json_safe(soil_water_floor),
        "open_water_skin": _json_safe(open_water_skin_record),
    }
    return (
        ArwenGlobalState(
            atmosphere, surface,
            PhysicsState(arrays=physics_arrays, metadata=statics_metadata),
        ),
        b.asarray(phi_model, dtype=b.float_dtype),
        provenance,
    )


__all__ = [
    "COASTAL_FILL_PASSES",
    "OPEN_WATER_SKIN_SEARCH_PASSES",
    "open_water_skin_temperature",
    "FILL_GROUPS",
    "CompositeFrame",
    "analysis_initial_state",
    "fill_frame",
    "floor_soil_water",
    "MappingResolution",
    "PACKAGE_AUTHORITIES_DIR",
    "resolve_analysis_mapping",
    "resolve_analysis_mapping_row",
    "surface_virtual_temperature",
]
