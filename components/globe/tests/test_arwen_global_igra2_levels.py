"""The radiosonde level table (``igra2-levels-csv``) and the tool that
writes it from the IGRA2 archive.

A level row carries its own pressure (``pressure_level`` mode): the door
compares it against the profile interpolated to that pressure, never
against the surface.  The tool's parser is held against a synthetic IGRA2
record with planted values in both directions: every planted level reads
back, the missing sentinels leave blanks, and a variable left out of
``--variables`` is blank so the door finds it underivable.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from woof.globe.obs_table import (
    LEVEL_MODES, VARIABLE_TABLE, decode_obs_csv, load_obs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
import arwen_global_igra2_levels_csv as tool  # noqa: E402

HEADER = ("station,valid,lon,lat,gph_m,pressure_hpa,temp_c,dewpoint_c,"
          "wdir_deg,wspd_m_s")


def test_a_level_row_carries_its_own_pressure():
    text = "\n".join([
        HEADER,
        "USM00072456,2026-09-01 12:00,-95.4700,39.0700,1543.0,850,17.20,11.40,200,6.50",
        "USM00072456,2026-09-01 12:00,-95.4700,39.0700,5860.0,500,-8.30,,270,18.00",
        "USM00072456,2026-09-01 12:00,-95.4700,39.0700,,700,,,,",
        "USM00072456,2026-09-01 12:00,-95.4700,39.0700,3100.0,0,5.0,1.0,,",
    ]) + "\n"
    source, rows, counters = decode_obs_csv(text)
    assert source == "igra2-levels-csv"
    assert "pressure_level" in LEVEL_MODES
    by = {(r.level_pa, r.variable): r for r in rows}
    t850 = by[(85000.0, "temperature_k")]
    assert t850.value == pytest.approx(290.35)
    assert t850.elevation_m == 1543.0
    assert t850.error == 1.0
    td850 = by[(85000.0, "dewpoint_k")]
    assert td850.value == pytest.approx(284.55)
    assert td850.error == 2.5
    # 200 degrees at 6.5 m/s: u = -6.5 sin(200), v = -6.5 cos(200), in m/s
    # straight from the archive's own unit.
    u = by[(85000.0, "wind_u_m_s")]
    v = by[(85000.0, "wind_v_m_s")]
    assert u.value == pytest.approx(-6.5 * np.sin(np.deg2rad(200.0)))
    assert v.value == pytest.approx(-6.5 * np.cos(np.deg2rad(200.0)))
    assert u.error == 2.5
    # A blank dewpoint leaves that variable underivable; the others stand.
    assert (50000.0, "dewpoint_k") not in by
    assert (50000.0, "temperature_k") in by
    assert (50000.0, "wind_u_m_s") in by
    # A blank height is a malformed row; a pressure of 0 is a sentinel.
    assert counters["rows_malformed"] == 2
    assert all(r.valid_time == dt.datetime(2026, 9, 1, 12, tzinfo=dt.timezone.utc) for r in rows)
    assert all(r.variable in VARIABLE_TABLE for r in rows)


def _igra2_record(dpdp_at_500: int = -9999) -> str:
    """One synthetic IGRA2 v2 sounding in the archive's fixed columns."""
    header = (
        "#USM00072456 2026 09 01 12 1131    3 ncdc-gts ncdc-gts  390700  -954700"
    )
    assert len(header) >= 71

    def level(press_pa, gph, temp, dpdp, wdir, wspd):
        return (
            f"21 {0:5d} {press_pa:6d} {gph:5d} {temp:5d}A{-9999:5d} {dpdp:5d} "
            f"{wdir:5d} {wspd:5d}"
        )

    lines = [
        header,
        level(85000, 1543, 172, 58, 200, 65),
        level(70000, 3100, 50, 40, 250, 120),
        level(50000, 5860, -83, dpdp_at_500, 270, 180),
    ]
    return "\n".join(lines) + "\n"


def test_the_tool_reads_planted_levels_back_and_blanks_the_sentinels(tmp_path):
    text = _igra2_record()
    soundings = tool.parse_igra2(text, {"2026-09-01T12:00:00Z"})
    assert len(soundings) == 1
    sounding = soundings[0]
    assert sounding["station"] == "USM00072456"
    assert sounding["latitude"] == pytest.approx(39.07)
    assert sounding["longitude"] == pytest.approx(-95.47)
    assert set(sounding["levels"]) == {85000, 70000, 50000}
    assert sounding["levels"][85000] == {
        "gph_m": 1543, "temp_c": 17.2, "dpdp_c": 5.8, "wdir_deg": 200, "wspd_m_s": 6.5,
    }
    assert sounding["levels"][50000]["dpdp_c"] is None
    # A sounding at another nominal hour is not kept.
    assert tool.parse_igra2(text, {"2026-09-01T00:00:00Z"}) == []

    rows = tool.rows_for(sounding, (850.0, 700.0, 500.0), {"temperature", "dewpoint", "wind"})
    by = {row["pressure_hpa"]: row for row in rows}
    assert by["850"]["temp_c"] == "17.20" and by["850"]["dewpoint_c"] == "11.40"
    assert by["850"]["wdir_deg"] == "200.00" and by["850"]["wspd_m_s"] == "6.50"
    assert by["850"]["gph_m"] == "1543.0" and by["850"]["valid"] == "2026-09-01 12:00"
    # The missing dewpoint depression leaves the dewpoint blank, nothing else.
    assert by["500"]["dewpoint_c"] == "" and by["500"]["temp_c"] == "-8.30"
    # A variable left out is blank so the door finds it underivable.
    dewpoint_only = tool.rows_for(sounding, (850.0,), {"dewpoint"})
    assert dewpoint_only[0]["temp_c"] == "" and dewpoint_only[0]["wdir_deg"] == ""
    assert dewpoint_only[0]["dewpoint_c"] == "11.40"

    # The whole tool, end to end, through the door's own table entry.
    source_dir = tmp_path / "zips"
    source_dir.mkdir()
    (source_dir / "USM00072456-data.txt").write_text(text, encoding="ascii")
    out = tmp_path / "levels.csv"
    assert tool.main([
        "--zips", str(source_dir), "--nominal", "2026-09-01T12:00:00Z",
        "--levels-hpa", "850,700,500", "--out", str(out),
    ]) == 0
    record = json.loads(out.with_suffix(".json").read_text())
    assert record["rows"] == 3 and record["soundings_at_nominal"] == 1
    assert record["soundings_with_rows_by_nominal"] == {"2026-09-01T12:00:00Z": 1}
    source, decoded, _provenance = load_obs(str(out))
    assert source == "igra2-levels-csv"
    by_level = {(r.level_pa, r.variable): r.value for r in decoded}
    assert by_level[(85000.0, "temperature_k")] == pytest.approx(290.35)
    assert by_level[(85000.0, "dewpoint_k")] == pytest.approx(284.55)
    assert (50000.0, "dewpoint_k") not in by_level
    assert len(decoded) == 4 + 4 + 3


def test_unknown_variables_are_refused_by_name(tmp_path):
    source_dir = tmp_path / "zips"
    source_dir.mkdir()
    (source_dir / "x.txt").write_text(_igra2_record(), encoding="ascii")
    with pytest.raises(SystemExit, match="unknown variables \\['humidity'\\]"):
        tool.main([
            "--zips", str(source_dir), "--nominal", "2026-09-01T12:00:00Z",
            "--variables", "humidity", "--out", str(tmp_path / "o.csv"),
        ])
