"""The neutral observation table as a file: gpuwm-obs.table.v2 and v1 read
without conversion, the bookkeeping columns carried, and the v1 door's
refusal by name of the variable it has no operator for."""
from __future__ import annotations

import datetime as dt

import pytest

from woof.globe import obs_table
from woof.globe.assimilate import (
    AssimilationOptions,
    REJECTION_BREAKAGE,
    VARIABLES_WITHOUT_OPERATOR,
    _table_quality_control,
)
from woof.globe.obs_table import (
    MEASUREMENT_TABLE,
    TABLE_HEADER,
    TABLE_HEADER_V1,
    TABLE_SCHEMA,
    VARIABLE_TABLE,
    ObsRow,
    decode_neutral_csv,
    decode_obs_csv,
    neutral_header_version,
)

UTC = dt.timezone.utc

V2_ROWS = """source,station_id,latitude_deg,longitude_deg,elevation_m,level_pa,valid_time,variable,value,error,measurement,nominal_time,published_time,received_time,revision
igra2,USM00072365,35.04000,-106.62000,5870.0,50000.0,2026-08-31T23:25:15Z,temperature_k,263.15,1,sonde_level,2026-09-01T00:00:00Z,2026-09-02T21:36:42Z,2026-09-06T01:01:18Z,16d059c6d0e2
ndbc,41001,34.50200,-72.52200,0.0,,2026-09-01T00:00:00Z,surface_pressure_pa,100860,100,sea_level_pressure,,,,5812ae1540f8
goes-dmw,G18-C02,29.25010,-83.29076,1350.8,86115.3,2026-09-01T12:00:20Z,wind_u_m_s,-4.386662017252213,3,amv_assigned_pressure,,2026-09-01T12:20:06Z,2026-09-06T01:01:28Z,37e568c80450
gnss-ro,cosmic2e1-E03,10.35062,-104.70860,1191.5,100103.7,2025-07-29T00:58:01Z,refractivity_n,323.1253356933594,3.231253356933594,ro_refractivity_tangent_point,,2025-08-01T13:55:32Z,2026-09-06T01:42:37Z,97c4cf5b5e03
iem-metar,FNLU,-8.85840,13.23120,70.0,,2026-08-31T18:00:00Z,surface_pressure_pa,100348.35974082419,100,station_pressure_from_altimeter,,,,096a3cbdffc7
iem-metar,FNLU,-8.85840,13.23120,70.0,,2026-08-31T18:00:00Z,temperature_k,296.15,1.5,made_up_label,,,,096a3cbdffc7
iem-metar,FNLU,-8.85840,13.23120,70.0,,2026-08-31T18:00:00Z,cloud_base_m,900,50,,,,,096a3cbdffc7
iem-metar,FNLU,-8.85840,13.23120,70.0,,2026-08-31T18:00:00Z,dewpoint_k,290.15,1.5,screen_dewpoint_2m,not-a-time,,,096a3cbdffc7
"""

V1_ROWS = """source,station_id,latitude_deg,longitude_deg,elevation_m,level_pa,valid_time,variable,value,error
igra2,USM00072365,35.04000,-106.62000,5870.0,50000.0,2026-09-01T00:00:00Z,temperature_k,263.15,1
"""


def test_the_headers_are_versioned_and_v2_extends_v1():
    assert TABLE_SCHEMA == "gpuwm-obs.table.v2"
    assert TABLE_HEADER[:10] == TABLE_HEADER_V1
    assert TABLE_HEADER[10:] == (
        "measurement", "nominal_time", "published_time", "received_time", "revision")
    assert neutral_header_version(list(TABLE_HEADER)) == 2
    assert neutral_header_version(list(TABLE_HEADER_V1)) == 1
    assert neutral_header_version(["station", "valid", "lon"]) is None


def test_v2_rows_carry_their_bookkeeping_and_bad_rows_are_counted():
    source, rows, counters = decode_neutral_csv(V2_ROWS)
    assert counters["table_version"] == 2
    assert counters["rows_scanned"] == 8
    # the unknown variable is dropped, the unparsable nominal time is malformed
    assert counters["rows_unknown_variable"] == 1
    assert counters["rows_malformed"] == 1
    # an unknown measurement label is counted and the row kept as written
    assert counters["rows_unknown_measurement"] == 1
    assert counters["rows_decoded"] == 6
    assert set(source.split("+")) == {"igra2", "ndbc", "goes-dmw", "gnss-ro", "iem-metar"}
    sonde = next(r for r in rows if r.source == "igra2")
    assert sonde.valid_time == dt.datetime(2026, 8, 31, 23, 25, 15, tzinfo=UTC)
    assert sonde.nominal_time == dt.datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    assert sonde.published_time == dt.datetime(2026, 9, 2, 21, 36, 42, tzinfo=UTC)
    assert sonde.received_time == dt.datetime(2026, 9, 6, 1, 1, 18, tzinfo=UTC)
    assert sonde.revision == "16d059c6d0e2"
    assert sonde.measurement == "sonde_level"
    assert sonde.level_pa == 50000.0
    buoy = next(r for r in rows if r.source == "ndbc")
    assert buoy.measurement == "sea_level_pressure"
    assert buoy.level_pa is None and buoy.elevation_m == 0.0
    assert buoy.received_time is None and buoy.published_time is None and buoy.nominal_time is None
    ro = next(r for r in rows if r.source == "gnss-ro")
    assert ro.variable == "refractivity_n" and ro.elevation_m == 1191.5
    # rows without a receipt time are counted for the latency label
    assert counters["rows_without_received_time"] == 3
    assert "refractivity_n" in VARIABLE_TABLE
    for row in rows:
        if row.measurement and row.measurement != "made_up_label":
            assert row.measurement in MEASUREMENT_TABLE


def test_v1_rows_read_with_empty_bookkeeping():
    source, rows, counters = decode_neutral_csv(V1_ROWS)
    assert counters["table_version"] == 1
    assert source == "igra2" and len(rows) == 1
    row = rows[0]
    assert row.measurement == "" and row.revision == ""
    assert row.nominal_time is None and row.published_time is None and row.received_time is None
    assert counters["rows_without_received_time"] == 1


def test_decode_obs_csv_recognises_both_neutral_headers_before_the_source_tables():
    source_v2, rows_v2, _ = decode_obs_csv(V2_ROWS)
    source_v1, rows_v1, _ = decode_obs_csv(V1_ROWS)
    assert len(rows_v2) == 6 and len(rows_v1) == 1
    assert "igra2" in source_v2 and source_v1 == "igra2"
    with pytest.raises(ValueError, match="not a gpuwm-obs.table.v2 file"):
        decode_neutral_csv("station,valid,lon\nX,2026-09-01T00:00:00Z,1\n")


def test_obsrow_identity_is_unchanged_by_the_bookkeeping():
    base = dict(source="igra2", station_id="USM00072365", latitude_deg=35.04, longitude_deg=-106.62,
                elevation_m=5870.0, level_pa=50000.0,
                valid_time=dt.datetime(2026, 8, 31, 23, 25, 15, tzinfo=UTC),
                variable="temperature_k", value=263.15, error=1.0)
    plain = ObsRow(**base)
    booked = ObsRow(**base, measurement="sonde_level", revision="16d059c6d0e2",
                    nominal_time=dt.datetime(2026, 9, 1, tzinfo=UTC))
    assert plain.identity() == booked.identity()
    assert plain.identity_hash() == booked.identity_hash()


def test_the_v1_door_refuses_refractivity_rows_by_name_and_covers_the_vocabulary():
    options = AssimilationOptions()
    assert set(dict(options.background_errors)) == set(VARIABLE_TABLE)
    assert "refractivity_n" in VARIABLES_WITHOUT_OPERATOR
    assert "no_operator" in REJECTION_BREAKAGE
    moment = dt.datetime(2025, 7, 29, 1, 0, tzinfo=UTC)
    rows = [
        ObsRow("gnss-ro", "rx-tx", 10.0, -104.0, 1191.5, 100103.7,
               dt.datetime(2025, 7, 29, 0, 58, 1, tzinfo=UTC), "refractivity_n", 323.1, 3.2),
        ObsRow("ndbc", "41001", 34.5, -72.5, 0.0, None,
               dt.datetime(2025, 7, 29, 1, 0, tzinfo=UTC), "surface_pressure_pa", 100860.0, 100.0),
    ]
    kept, rejections = _table_quality_control(rows, moment, options)
    assert rejections["no_operator"] == 1
    assert [r.variable for r in kept] == ["surface_pressure_pa"]
