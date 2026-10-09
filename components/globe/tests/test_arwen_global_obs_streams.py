"""The observation streams module: the stream roster, windows at the
reports' own times, the information cutoff, thinning, the v2 table round
trip, the hourly cut and the manifests."""
from __future__ import annotations

import datetime as dt
import json

import pytest

from woof.globe import obs_streams
from woof.globe.obs_streams import (
    ANCHOR_SOURCES,
    LATENCY_CLASSES,
    STREAMS,
    account_gated_sources,
    analysis_window,
    apply_information_cutoff,
    cut_hours,
    fetchable_streams,
    rows_in_window,
    stream_manifest,
    streams_table,
    thin_to_grid,
    write_table,
)
from woof.globe.obs_table import (
    MEASUREMENT_TABLE,
    TABLE_HEADER,
    VARIABLE_TABLE,
    ObsRow,
    load_obs,
)

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def row(source, station, lat, lon, minutes, variable="temperature_k", value=280.0,
        level=None, received=None, elevation=10.0, measurement="screen_temperature_2m"):
    return ObsRow(
        source=source, station_id=station, latitude_deg=lat, longitude_deg=lon,
        elevation_m=elevation, level_pa=level,
        valid_time=T0 + dt.timedelta(minutes=minutes), variable=variable, value=value,
        error=1.5, measurement=measurement, received_time=received, revision="abcdef012345",
    )


def test_the_roster_is_consistent_with_the_vocabulary_and_the_classes():
    for spec in STREAMS.values():
        assert set(spec.variables) <= set(VARIABLE_TABLE)
        assert spec.latency_class in LATENCY_CLASSES
        assert set(spec.measurements) <= set(MEASUREMENT_TABLE)
        assert spec.latency_basis
        if spec.account_gated:
            assert spec.door is None and spec.latency_class == "gated" and not spec.public
    gated = {s.name for s in account_gated_sources()}
    assert gated == {"madis-aircraft"}  # the CDAAC route answered anonymously on 2026-09-06
    fetchable = {s.name for s in fetchable_streams()}
    assert {"iem-metar", "igra2", "goes-dmw", "ndbc", "gnss-ro", "cdaac-ro", "awc-metar"} <= fetchable
    assert "wis2" in fetchable  # the archived BUFR payloads decode into the table (rw_wis2 table)
    assert set(STREAMS["wis2"].variables) == {"surface_pressure_pa", "temperature_k", "dewpoint_k",
                                              "wind_u_m_s", "wind_v_m_s"}
    assert STREAMS["gnss-ro"].latency_class == "retrospective"
    assert STREAMS["cdaac-ro"].latency_class == "replay"
    assert STREAMS["cdaac-ro"].errors == STREAMS["gnss-ro"].errors == obs_streams.REFRACTIVITY_ERRORS
    assert STREAMS["igra2"].latency_class == "replay"
    assert STREAMS["iem-metar"].latency_class == "fast"
    # the external analysis is an anchor source, never a stream
    assert "analysis-pseudo" not in STREAMS
    assert set(ANCHOR_SOURCES) == {"gdas", "ifs-open-data"}
    table = streams_table()
    assert {r["stream"] for r in table} == set(STREAMS)


def test_a_stream_naming_an_unknown_variable_or_class_is_refused():
    with pytest.raises(ValueError, match="outside the neutral vocabulary"):
        obs_streams.StreamSpec(
            name="x", door=None, subject="", sources=(), variables=("cloud_base_m",), errors={},
            cadence_s=1, public=True, latency_class="fast", latency_basis="b")
    with pytest.raises(ValueError, match="latency class"):
        obs_streams.StreamSpec(
            name="x", door=None, subject="", sources=(), variables=(), errors={},
            cadence_s=1, public=True, latency_class="soon", latency_basis="b")
    with pytest.raises(ValueError, match="must be class 'gated'"):
        obs_streams.StreamSpec(
            name="x", door=None, subject="", sources=(), variables=(), errors={},
            cadence_s=1, public=False, latency_class="fast", latency_basis="b", account_gated=True)


def test_the_anchor_record_states_the_source_cycle_and_its_availability_latency():
    anchor = ANCHOR_SOURCES["gdas"]
    cycle = dt.datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    record = anchor.record(
        cycle, published_time=dt.datetime(2026, 9, 1, 7, 7, 37, tzinfo=UTC),
        received_time=dt.datetime(2026, 9, 6, 0, 34, 50, tzinfo=UTC), bytes_=471773591)
    assert record["schema"] == "gpuwm-obs.anchor.v1"
    assert record["availability_latency_s"] == 7 * 3600 + 7 * 60 + 37
    assert record["url"].endswith("gdas.20260901/00/atmos/gdas.t00z.pgrb2.0p25.f000")
    assert "never pseudo-observations" in record["role"]
    unknown = anchor.record(cycle, published_time=None, received_time=None)
    assert unknown["availability_latency_s"] is None


def test_windows_are_half_open_and_keep_the_reports_own_times():
    start, end = analysis_window(T0, 3600)
    assert start == T0 - dt.timedelta(minutes=30) and end == T0 + dt.timedelta(minutes=30)
    rows = [row("a", "s", 0, 0, -30), row("a", "s", 0, 0, -29), row("a", "s", 0, 0, 30),
            row("a", "s", 0, 0, 31)]
    kept = rows_in_window(rows, start, end)
    assert [r.valid_time for r in kept] == [rows[1].valid_time, rows[2].valid_time]
    # nothing moved a row's time to the analysis instant
    assert all(r.valid_time != T0 for r in kept)
    with pytest.raises(ValueError):
        analysis_window(T0, 0)


def test_the_information_cutoff_drops_late_receipts_and_labels_unknown_ones():
    early = row("a", "s1", 0, 0, 0, received=T0 - dt.timedelta(minutes=5))
    late = row("a", "s2", 0, 1, 0, received=T0 + dt.timedelta(days=5))
    unknown = row("a", "s3", 0, 2, 0, received=None)
    kept, counters = apply_information_cutoff([early, late, unknown], T0)
    assert [r.station_id for r in kept] == ["s1", "s3"]
    assert counters == {"offered": 3, "kept": 2, "after_cutoff": 1, "latency_unverified": 1}
    kept_all, counters_all = apply_information_cutoff([early, late, unknown], None)
    assert len(kept_all) == 3 and counters_all["after_cutoff"] == 0
    assert counters_all["latency_unverified"] == 1


def test_thinning_keeps_the_row_nearest_the_instant_per_source_cell_and_layer():
    rows = [
        row("a", "s1", 10.2, 20.2, -20),   # same cell as s2, further from the instant
        row("a", "s2", 10.3, 20.3, 5),
        row("b", "s3", 10.25, 20.25, 15),  # another source in the same cell: kept
        row("a", "s4", 10.2, 20.2, 0, variable="wind_u_m_s", level=50000.0, measurement="sonde_level"),
        row("a", "s5", 10.2, 20.2, 1, variable="wind_u_m_s", level=52000.0, measurement="sonde_level"),
        row("a", "s6", 10.2, 20.2, 1, variable="wind_u_m_s", level=30000.0, measurement="sonde_level"),
    ]
    kept, counters = thin_to_grid(rows, analysis_time=T0, nlat=90, nlon=180, layer_ln_p=0.1)
    ids = [r.station_id for r in kept]
    assert "s2" in ids and "s1" not in ids and "s3" in ids
    # 500 and 520 hPa share a 0.1 ln p layer (ln 50000 = 10.82, ln 52000 = 10.86); 300 hPa is another
    assert "s4" in ids and "s5" not in ids and "s6" in ids
    assert counters["offered"] == 6 and counters["kept"] == 4 and counters["thinned"] == 2
    assert counters["thinned_by_source"] == {"a": 2}
    with pytest.raises(ValueError):
        thin_to_grid(rows, analysis_time=T0, nlat=0, nlon=1)


def test_write_table_round_trips_every_bookkeeping_column(tmp_path):
    rows = [
        row("igra2", "USM00072365", 35.04, -106.62, -35, variable="temperature_k", value=263.15,
            level=50000.0, received=T0 - dt.timedelta(hours=1), elevation=5870.0,
            measurement="sonde_level"),
        row("ndbc", "41001", 34.5, 287.478, 0, variable="surface_pressure_pa", value=100860.0,
            elevation=0.0, measurement="sea_level_pressure"),
    ]
    rows[0] = ObsRow(**{**rows[0].__dict__, "nominal_time": T0,
                        "published_time": T0 + dt.timedelta(days=1)})
    path = tmp_path / "t.csv"
    record = write_table(rows, path)
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == ",".join(TABLE_HEADER)
    assert record["rows"] == 2 and record["rows_latency_unverified"] == 1
    assert record["rows_by_source"] == {"igra2": 1, "ndbc": 1}
    source, back, provenance = load_obs(str(path))
    assert provenance["counters"]["table_version"] == 2
    assert len(back) == 2
    sonde = next(r for r in back if r.source == "igra2")
    assert sonde.nominal_time == T0 and sonde.published_time == T0 + dt.timedelta(days=1)
    assert sonde.received_time == T0 - dt.timedelta(hours=1)
    assert sonde.measurement == "sonde_level" and sonde.revision == "abcdef012345"
    assert sonde.valid_time == rows[0].valid_time
    buoy = next(r for r in back if r.source == "ndbc")
    # longitude written in [-180, 180)
    assert abs(buoy.longitude_deg - (287.478 - 360.0)) < 1e-9


def test_cut_hours_writes_one_table_per_instant_with_the_cutoff_and_thinning_counted(tmp_path):
    rows = [
        row("a", "s1", 0, 0, -20, received=T0 - dt.timedelta(minutes=1)),
        row("a", "s1", 0, 0, 40, received=T0 + dt.timedelta(minutes=62)),
        row("a", "s1", 0, 0, 65, received=None),
        row("b", "s9", 0, 0, 0, received=T0 + dt.timedelta(days=5)),  # received long after: dropped under 'analysis'
    ]
    source = tmp_path / "stream.csv"
    write_table(rows, source)
    out = tmp_path / "hours"
    manifest = cut_hours([source], start=T0, end=T0 + dt.timedelta(hours=1), out_dir=out,
                         cutoff="analysis", thin_grid=(90, 180))
    assert manifest["schema"] == "gpuwm-obs.hours.v2"
    assert manifest["information_cutoff"] == "each hour's analysis instant"
    assert [h["analysis_time"] for h in manifest["hours"]] == ["2026-09-01T12:00:00Z", "2026-09-01T13:00:00Z"]
    first, second = manifest["hours"]
    # 12Z window (11:30, 12:30]: s1 at -20 (received before 12Z, kept), s9 (received days later, dropped)
    assert first["cutoff_counters"]["after_cutoff"] == 1
    assert first["table"]["rows"] == 1
    # 13Z window (12:30, 13:30]: s1 at +40 (received 13:02, at or before 13:00? no: 13:02 > 13:00 -> dropped)
    # and s1 at +65 (no receipt time: kept, unverified)
    assert second["cutoff_counters"]["after_cutoff"] == 1
    assert second["cutoff_counters"]["latency_unverified"] == 1
    assert second["table"]["rows"] == 1
    assert second["thinning"]["nlat"] == 90
    written = json.loads((out / "hours.json").read_text(encoding="utf-8"))
    assert written["hours"][0]["table"]["path"].endswith("2026-09-01T12.csv")
    # without a cutoff every row of the window is kept and only labelled
    manifest_all = cut_hours([source], start=T0, end=T0, out_dir=tmp_path / "all")
    assert manifest_all["hours"][0]["table"]["rows"] == 2
    assert manifest_all["hours"][0]["cutoff_counters"]["after_cutoff"] == 0
    with pytest.raises(ValueError):
        cut_hours([source], start=T0, end=T0 - dt.timedelta(hours=1), out_dir=tmp_path / "bad")


def test_stream_manifest_wraps_the_doors_records_and_names_the_class(tmp_path):
    table_record = tmp_path / "igra2.json"
    table_record.write_text(json.dumps({
        "schema": "gpuwm-obs.igra2-table.v1", "path": "igra2.csv", "sha256": "ab", "rows": 3,
        "bytes": 300, "status": "READY", "counters": {"files_read": 1},
        "latency_behind_real_time_s": 120998,
        "latency_basis": "archive file Last-Modified minus the latest nominal hour it holds",
    }), encoding="utf-8")
    out = tmp_path / "manifest.json"
    manifest = stream_manifest("igra2", table_record=table_record, out=out)
    assert manifest["schema"] == "gpuwm-obs.stream.v2"
    assert manifest["latency_behind_real_time_s"] == 120998
    assert manifest["latency_class"] == "replay"
    assert manifest["table"]["rows"] == 3
    assert manifest["measurements"] == ["station_pressure", "sonde_level"]
    assert json.loads(out.read_text(encoding="utf-8"))["stream"] == "igra2"
    # the metar door states an upper bound instead
    metar_record = tmp_path / "metar.json"
    metar_record.write_text(json.dumps({
        "schema": "gpuwm-obs.asos-table.v1", "path": "m.csv", "sha256": "cd", "rows": 1, "bytes": 1,
        "status": "READY", "latency_upper_bound_s": 137,
        "latency_basis": "fetch record's fetched_at minus the latest report kept (an upper bound)",
    }), encoding="utf-8")
    m2 = stream_manifest("iem-metar", table_record=metar_record)
    assert m2["latency_behind_real_time_s"] == 137 and m2["latency_class"] == "fast"


def test_the_cli_lists_streams_and_anchors(capsys):
    assert obs_streams.main(["streams"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {r["stream"] for r in listed} == set(STREAMS)
    assert obs_streams.main(["anchors"]) == 0
    anchors = json.loads(capsys.readouterr().out)
    assert {a["anchor"] for a in anchors} == {"gdas", "ifs-open-data"}
