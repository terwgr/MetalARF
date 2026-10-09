"""Every route that keeps a provenance block keeps the DECODE receipt.

`woof.globe.mapped_source_compat` adapts two published-engine differences
in place: where the decoder's multi-GB frame stream is staged, and the
soil-only narrowing the published `load_mapping` puts on a masked surface
record.  Both are silent from the outside.  The only thing that says which
mechanism a run actually used is the receipt block the door hands back, and
a caller that takes `.frames` and drops `.receipt` turns that into a
version number a reader has to guess from.

THE BREAKAGE THIS PREVENTS, measured.  The ATMS column receipt delivered
with the 2026-09-09 release proofs carried the mapping digest and no decode
block at all, while the ATMS mapping declares two masked surface records
and therefore really did run the validator adaptation on the published
engine.  Three of the five decode routes were in that state: the columns
that feed the assimilation, the radiation reference cover and the
surface-energy state product, all adapted, none saying so.

The fake decoder here stands in for the engine's byte work only.  The
mapping documents are the real carried ones and the adaptation in front of
them is the real one, so the receipt these assertions read is produced the
way a run produces it.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from woof.globe.analysis_initial import PACKAGE_AUTHORITIES_DIR


def _receipt_is_complete(decode: dict, *, masked: tuple[str, ...] = ()) -> None:
    """The rows a reader needs to reconstruct how the decode was obtained."""

    assert isinstance(decode, dict), decode
    assert decode["mapping"]
    assert decode["inputs"]
    assert decode["scratch"]["mechanism"]
    assert decode["scratch"]["placed_by"]
    assert decode["surface_preserve_mask"]
    assert decode["engine_version"]
    if masked:
        assert tuple(decode["masked_surface_fields"]) == masked


class _Field:
    def __init__(self, values):
        self.values = values


class _Frame:
    """The shape the engine's decoder returns, with the fields one route
    reads.  Nothing here is scored; the routes are exercised for what they
    record, not for what they compute."""

    vertical_kind = "pressure"

    def __init__(self, names, *, nlev=0, ny=7, nx=12):
        self.latitude = np.linspace(90.0, -90.0, ny)
        self.longitude = np.arange(0.0, 360.0, 360.0 / nx)
        self.vertical_values = np.linspace(100000.0, 10000.0, nlev) if nlev else np.zeros(0)
        self.valid_time = dt.datetime(2026, 9, 1, 0, 0)
        self.source_cycle = dt.datetime(2026, 9, 1, 0, 0)
        self.input_sha256 = {"x.grib2": "0" * 64}
        self.mapping_sha256 = "1" * 64
        self.grid_fingerprint = "2" * 64
        shape = (nlev, ny, nx) if nlev else (ny, nx)
        self.fields = {}
        for name in names:
            if nlev and name in _SURFACE_ONLY:
                self.fields[name] = _Field(np.full((ny, nx), 1.0))
            else:
                self.fields[name] = _Field(np.full(shape, 1.0))


_SURFACE_ONLY = {
    "surface_pressure", "terrain_height", "skin_temperature", "air_temperature_2m",
    "eastward_wind_10m", "northward_wind_10m", "land_fraction", "sea_ice_fraction",
    "cloud_water_path", "precipitable_water",
}


def _install(monkeypatch, frame):
    """Put one frame behind the engine's decoder, leaving the mapping
    validator and the adaptation in front of it exactly as installed."""

    import woof.mapped_source as engine

    calls: list[tuple] = []

    def fake_decode(mapping_path, files, **_rest):
        calls.append((str(mapping_path), [str(f) for f in files]))
        return [frame]

    monkeypatch.setattr(engine, "decode_mapped_source", fake_decode)
    return calls


# ------------------------------------------------------- the ATMS columns

def test_the_atms_columns_receipt_carries_the_decode_block(monkeypatch, tmp_path):
    """The product that feeds the assimilation says how it was decoded.

    Its mapping declares two masked surface records, so on a published
    engine this decode runs the validator adaptation; the sidecar the
    `microwave columns` door writes is where a reader finds that out.
    """

    from woof.globe.microwave import columns

    names = list(_SURFACE_ONLY) + ["air_temperature", "specific_humidity"]
    _install(monkeypatch, _Frame(names, nlev=5))
    analysis = columns.decode_analysis(tmp_path / "gdas.t00z.pgrb2.0p25.f000")
    _receipt_is_complete(analysis.provenance["decode"],
                         masked=("snow_depth", "snow_water_equivalent"))

    # It survives the cache the door writes and the reader that opens it,
    # which is the file the release actually delivers.
    columns.save_analysis(analysis, tmp_path / "columns.npz")
    sidecar = json.loads((tmp_path / "columns.json").read_text(encoding="utf-8"))
    _receipt_is_complete(sidecar["decode"])
    assert columns.load_analysis(tmp_path / "columns.npz").provenance["decode"] == \
        analysis.provenance["decode"]


# ------------------------------------------------ the surface-energy state

def test_the_surface_energy_state_product_carries_the_decode_block(
        monkeypatch, tmp_path):
    """The state mapping declares two masked surface records too."""

    from woof.globe import surface_energy as se

    _install(monkeypatch, _Frame(list(se.STATE_FIELDS.values()), nlev=4))
    state = se.decode_gfs_state(tmp_path / "gfs.t00z.pgrb2.0p25.f000")
    _receipt_is_complete(state["decode"],
                         masked=("snow_water_equivalent", "vegetation_fraction"))


def test_the_flux_ladder_records_how_its_mapping_was_read(monkeypatch, tmp_path):
    """The one route that runs no decode still says what it did.

    The pdt-8 flux records cannot be bound as model inputs, so this ladder
    reads them through the engine's GRIB2 bridges and reaches the package
    door only for the mapping document.  It has no decode receipt because
    there is no decode; what it records instead is which door read the
    document and which of its records the published validator would have
    narrowed (none, today).
    """

    from woof.globe import surface_energy as se

    class _Record:
        def __init__(self, index):
            self.index = index
            self.values = np.full((7, 12), 1.0)
            self.latitude = np.linspace(90.0, -90.0, 7)
            self.longitude = np.arange(0.0, 360.0, 30.0)
            self.valid_time = dt.datetime(2026, 9, 1, 0, 0)

    import woof.mapped_source as engine

    from woof.globe.analysis_initial import resolve_analysis_mapping
    from woof.globe.mapped_source_compat import load_mapping

    mapping = load_mapping(resolve_analysis_mapping(se.FLUX_MAPPING_ID))
    first = {short: mapping["fields"][name]["selectors"][0]
             for short, name in se.FLUX_FIELDS.items()}
    order = list(se.FLUX_FIELDS)
    rows = [{"index": str(i), "field": short} for i, short in enumerate(order)]

    monkeypatch.setattr(engine, "_build_grib2_tools", lambda: ("wgrib2", "wgrib2"))
    monkeypatch.setattr(engine, "_grib2_inventory", lambda source, tool: rows)
    monkeypatch.setattr(
        engine, "_grib2_records",
        lambda source, inv, dump, wanted: [_Record(i) for i in sorted(wanted)])
    monkeypatch.setattr(se, "record_matches_selector",
                        lambda selector, row: selector == first[row["field"]])

    flux = se.decode_gfs_flux(tmp_path / "gfs.t00z.pgrb2.0p25.f001")
    block = flux["mapping"]
    assert Path(block["path"]).name.endswith(".mapping.json")
    assert block["read_through"].endswith("load_mapping")
    assert block["masked_surface_fields"] == []
    assert "decode_mapped_source" in block["records_read_by"]


# ------------------------------------------------- the radiation reference

def test_the_radiation_reference_carries_the_decode_block_to_its_checkpoint(
        monkeypatch, tmp_path):
    """The receipt reaches the entry that is scored against the product.

    Dropping it at the decode left the scorecard recording a reference
    product's valid time with nothing about the mechanism that read it,
    and the cover mapping declares two masked surface records.
    """

    from woof.globe import radiation_scorecard as rs

    frame = _Frame(list(rs.REFERENCE_COVER_FIELDS), ny=21, nx=36)
    _install(monkeypatch, frame)
    decoded = rs.decode_reference_cover([str(tmp_path / "gfs.f001")])
    assert decoded[0]["status"] == "measured"
    _receipt_is_complete(decoded[0]["decode"],
                         masked=("snow_depth", "snow_water_equivalent"))

    grid = rs.synthetic_grid(21)
    lat2d, _lon2d = rs.model_lat_lon(grid)
    planes = {"swdown": 0.0, "gsw": 0.0, "glw": 0.0, "olr": 0.0,
              "cldfra_total": np.full(lat2d.shape, 1.0)}
    samples = [rs.synthetic_sample(grid, time_s=0.0, step=0, planes=planes),
               rs.synthetic_sample(grid, time_s=3600.0, step=72, planes=planes)]
    result = rs.scorecard(samples, grid, reference_frames={72: decoded[0]})
    _receipt_is_complete(result["reference"]["cloud_cover"]["72"]["decode"])


# ---------------------------------------------------- the upper-air reference

def test_the_upper_air_reference_hands_back_its_receipt(monkeypatch, tmp_path):
    """`decode_reference` returns the pair, and the source block keeps it.

    Every caller of that function writes a provenance block; the pair is
    what stops each of them from having to remember to ask.
    """

    from woof.globe import upper_air_scorecard as ua
    from woof.globe.spectral.grid import GaussianGrid

    names = ("geopotential_height", "air_temperature", "specific_humidity",
             "eastward_wind", "northward_wind", "surface_pressure", "terrain_height")
    frame = _Frame(names, nlev=len(ua.LEVELS_PA), ny=21, nx=36)
    frame.vertical_values = np.asarray(ua.LEVELS_PA, dtype=np.float64)
    for name in ("surface_pressure", "terrain_height"):
        frame.fields[name] = _Field(np.full((21, 36), 101325.0 if "pressure" in name else 0.0))
    _install(monkeypatch, frame)

    gdas = PACKAGE_AUTHORITIES_DIR / "rw-wps-gdas-global-analysis-grib2.mapping.json"
    returned, decode = ua.decode_reference(gdas, tmp_path / "gdas.f000")
    assert returned is frame
    _receipt_is_complete(decode, masked=("snow_depth", "snow_water_equivalent"))

    grid = GaussianGrid.for_shape(22, 44)
    fields = ua.reference_fields(frame, grid, path=str(tmp_path / "gdas.f000"),
                                 decode=decode)
    _receipt_is_complete(fields.source["decode"])
    # It survives the reference cache, which is what the parallel worker
    # writes and every scorecard run reads back instead of decoding again.
    cache = tmp_path / "ref.npz"
    fields.save(cache)
    _receipt_is_complete(type(fields).load(cache).source["decode"])
    # A frame that came from no decode says so rather than inventing a row.
    assert ua.reference_fields(frame, grid).source["decode"] is None


# ------------------------------------- the receipt never says the wrong thing

def test_a_document_refused_for_another_reason_does_not_read_not_needed(
        monkeypatch, tmp_path):
    """A receipt that says "not needed" about a document declaring masked
    records says the opposite of what the file says.

    Two branches used to yield that block: an engine refusal that is not
    the narrowing, and a validator that could not be run at all.  The
    first always ends in the engine's own refusal so its block is never
    returned; the second can reach a caller.
    """

    from woof.globe import mapped_source_compat as compat

    gdas = PACKAGE_AUTHORITIES_DIR / "rw-wps-gdas-global-analysis-grib2.mapping.json"

    monkeypatch.setattr(compat, "_engine_load_mapping",
                        lambda: _raiser(ValueError("some other objection")))
    with compat._surface_preserve_mask_adaptation(gdas) as block:
        assert "another reason" in block["surface_preserve_mask"]
        assert block["masked_surface_fields"] == ["snow_depth",
                                                  "snow_water_equivalent"]

    monkeypatch.setattr(compat, "_engine_load_mapping",
                        lambda: _raiser(MemoryError("the validator died")))
    with compat._surface_preserve_mask_adaptation(gdas) as block:
        assert "could not be run" in block["surface_preserve_mask"]
        assert "MemoryError" in block["surface_preserve_mask"]
        assert block["masked_surface_fields"] == ["snow_depth",
                                                  "snow_water_equivalent"]


def _raiser(error):
    def call(_path, **_kw):
        raise error
    return call


def test_a_document_with_no_masked_record_still_reads_not_needed():
    """The one place the sentence is true keeps saying it."""

    from woof.globe import mapped_source_compat as compat

    flux = PACKAGE_AUTHORITIES_DIR / "rw-wps-gfs-surface-flux-grib2.mapping.json"
    assert compat.surface_preserve_mask_fields(flux) == ()
    with compat._surface_preserve_mask_adaptation(flux) as block:
        assert block["surface_preserve_mask"] == "not needed"
        assert "masked_surface_fields" not in block


# ------------------------------------------------- the disagreement refusal

def test_a_moved_engine_row_is_a_disagreement_on_the_upper_air_route(
        monkeypatch, tmp_path):
    """A ValueError subclass swallowed by a broad handler is a refusal lost.

    `resolve_mapping` fell through to "not on this machine; pass
    --mapping" when both tables answered one name with different bytes,
    which sends a reader to look for a missing file that is present twice.
    """

    from woof.globe import analysis_initial as ai
    from woof.globe import upper_air_scorecard as ua

    name = "rw-wps-gdas-global-analysis-grib2.mapping.json"
    planted = tmp_path / "engine-authorities"
    planted.mkdir()
    carried = (PACKAGE_AUTHORITIES_DIR / name).read_bytes()
    (planted / name).write_bytes(carried + b"\n")
    monkeypatch.setattr(ai, "_engine_authorities_dir", lambda: planted)

    # The receipt records an absolute path from another machine, so the
    # resolver is reached through the FILE NAME branch, which is the one
    # that swallowed the refusal.
    receipt = {"initial": {"provenance": {
        "mode": "analysis", "mapping": f"/elsewhere/{name}",
        "input_sha256": {}, "mapping_sha256": None}}}
    with pytest.raises(ai.MappingTablesDisagree) as caught:
        ua.resolve_mapping(None, receipt)
    text = str(caught.value)
    assert str(planted) in text and str(PACKAGE_AUTHORITIES_DIR) in text
    assert "pass --mapping" not in text
