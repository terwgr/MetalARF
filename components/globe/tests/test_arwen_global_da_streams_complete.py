"""Every observation stream that exists as a table assimilates: the
refractivity rows through the ensemble package's own operators (no
``no_operator`` refusal on the letkf door), a motion vector's error
inflated by its height-assignment shear, the same instrument reported by
two streams collapsed to one row, the stream roster in every report with
the latency class and the counts, and the door's ``--stream NAME`` for
every stream of the observation streams module."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from woof.globe import da_door, da_streams, obs_streams
from woof.globe.assimilate import (
    REJECTION_BREAKAGE,
    AssimilationOptions,
    _table_quality_control,
    cross_stream_duplicates,
)
from woof.globe.checkpoint import read_checkpoint, state_from_checkpoint
from woof.globe.da import FilterOptions, analyze_ensemble
from woof.globe.da.operators import (
    AMV_MEASUREMENT,
    OPERATOR_VARIABLES,
    WIND_SHEAR_VARIABLE,
    MemberOperators,
    batches_from_rows,
)
from woof.globe.da_filter import ENSEMBLE_MANIFEST_NAME
from woof.globe.obs_table import ObsRow, isa_pressure_pa
from woof.globe.runner import build_model_and_cold_state, build_transform

from test_arwen_global_assimilate import CONFIG, spun_up  # noqa: F401 - the fixture rides the import
from test_arwen_global_cycle import OBS_TIME, START_TEXT, OPTIONS
from test_arwen_global_da_ensemble import WHEN, _ensemble, _truth, world  # noqa: F401

UTC = dt.timezone.utc


def _row(source, station, lat, lon, variable, value, error, *, level=None, elev=0.0, measurement="",
         minutes=0.0, when=OBS_TIME):
    return ObsRow(source=source, station_id=station, latitude_deg=lat, longitude_deg=lon,
                  elevation_m=elev, level_pa=level, valid_time=when + dt.timedelta(minutes=minutes),
                  variable=variable, value=value, error=error, measurement=measurement)


def test_the_same_instrument_through_two_streams_is_one_report_and_the_rule_names_its_breakage():
    assert "duplicate_cross_stream" in REJECTION_BREAKAGE
    rows = [
        _row("iem-metar", "KORD", 41.98, -87.90, "temperature_k", 290.0, 1.5),
        # the AWC cache carrying the same station's report, coordinates a
        # few thousandths apart, the same minute: one instrument
        _row("awc-metar", "KORD", 41.979, -87.904, "temperature_k", 290.0, 1.5),
        # a WIS2 SYNOP of a station with another id but the same position
        # and minute: the position cell joins it
        _row("wis2", "0-20000-0-72530", 41.981, -87.901, "temperature_k", 290.1, 1.2),
        # the same station's pressure: a different variable, kept
        _row("awc-metar", "KORD", 41.979, -87.904, "surface_pressure_pa", 99000.0, 100.0),
        # a buoy under the station's cell is a second instrument only when
        # its id and cell differ; here a different cell
        _row("ndbc", "45007", 42.67, -87.03, "temperature_k", 289.0, 1.5),
        # the same station an hour earlier is another report
        _row("awc-metar", "KORD", 41.979, -87.904, "temperature_k", 289.0, 1.5, minutes=-60.0),
        # a SYNOP at the hour and the METAR seven minutes before it are one
        # report of one sensor: joined across the bin edge
        _row("iem-metar", "KMKE", 42.955, -87.904, "dewpoint_k", 285.0, 1.5, minutes=-7.0),
        _row("wis2", "0-20000-0-72640", 42.9551, -87.9041, "dewpoint_k", 285.2, 1.5),
        # a report eleven minutes apart is another observation of the station
        _row("iem-metar", "KGRB", 44.48, -88.13, "dewpoint_k", 283.0, 1.5, minutes=-11.0),
        _row("wis2", "0-20000-0-72645", 44.4805, -88.1301, "dewpoint_k", 283.3, 1.5),
        # two rows of ONE stream never collapse here
        _row("iem-metar", "KMDW", 41.79, -87.75, "temperature_k", 291.0, 1.5),
        _row("iem-metar", "KMDW", 41.79, -87.75, "temperature_k", 291.2, 1.5, minutes=1.0),
    ]
    kept, dropped = cross_stream_duplicates(rows)
    # the smaller error wins the temperature group (wis2 at 1.2 K), the
    # other two are counted against their streams
    # other two are counted against their streams; the SYNOP and METAR
    # seven minutes apart tie on error and the stream name keeps the METAR
    assert dropped == {"awc-metar": 1, "iem-metar": 1, "wis2": 1}
    assert [(r.source, r.variable) for r in kept] == [
        ("wis2", "temperature_k"), ("awc-metar", "surface_pressure_pa"), ("ndbc", "temperature_k"),
        ("awc-metar", "temperature_k"), ("iem-metar", "dewpoint_k"), ("iem-metar", "dewpoint_k"),
        ("wis2", "dewpoint_k"), ("iem-metar", "temperature_k"), ("iem-metar", "temperature_k"),
    ]
    # through the table quality control: counted by name
    moment = OBS_TIME + dt.timedelta(minutes=5)
    kept2, rejections = _table_quality_control(rows, moment, AssimilationOptions())
    assert rejections["duplicate_cross_stream"] == 3
    # the in-stream rule then keeps KMDW's latest and the earlier KORD row
    # falls to the same station's later one
    assert rejections["duplicate_superseded"] == 2
    assert {r.source for r in kept2} == {"wis2", "awc-metar", "ndbc", "iem-metar"}


def test_the_letkf_door_takes_refractivity_rows_while_the_successive_correction_refuses_them():
    moment = OBS_TIME
    rows = [_row("gnss-ro", "C2E1-G09", 10.0, 20.0, "refractivity_n", 200.0, 2.0, level=50000.0, elev=5500.0,
                 measurement="ro_refractivity_tangent_point")]
    kept, rejections = _table_quality_control(rows, moment, AssimilationOptions())
    assert rejections["no_operator"] == 1 and kept == []
    kept, rejections = _table_quality_control(rows, moment, AssimilationOptions(), operator_variables=OPERATOR_VARIABLES)
    assert rejections["no_operator"] == 0 and len(kept) == 1
    assert "refractivity_n" in OPERATOR_VARIABLES
    assert "successive correction" in REJECTION_BREAKAGE["no_operator"]


def test_the_member_operators_read_refractivity_and_the_height_assignment_shear(world):
    cfg, ecfg, options, transform, model, cold = world
    # the shear is read when the option is set; the bare default (0) reads none (the inflation is selectable)
    assert MemberOperators.for_model(model, transform, ecfg).amv_height_assignment_sigma_pa == 0.0
    operators = MemberOperators.for_model(model, transform, ecfg, amv_height_assignment_sigma_pa=10_000.0)
    truth = _truth(world)
    lat = np.array([10.0, -30.0, 45.0, 60.0])
    lon = np.array([20.0, 100.0, 250.0, 300.0])
    heights = np.array([3000.0, 6000.0, 9000.0, 12000.0])
    levels = np.array([isa_pressure_pa(z) for z in heights])
    values, lnp = operators.evaluate([truth], lat, lon, heights, levels, variables=("refractivity_n",))
    n = values["refractivity_n"][0]
    assert np.all(np.isfinite(n)) and np.all(n > 20.0) and np.all(n < 400.0)
    assert np.all(np.diff(n) < 0.0)  # refractivity falls with height
    assert lnp == pytest.approx(np.log(levels))
    # the column vocabulary is untouched by a refractivity request, and the
    # refractivity is untouched by a column request
    assert np.all(np.isnan(values["temperature_k"]))
    column, _ = operators.evaluate([truth], lat, lon, np.zeros(4), np.full(4, 50000.0),
                                   variables=("temperature_k", "wind_u_m_s", "wind_v_m_s", WIND_SHEAR_VARIABLE))
    assert np.all(np.isnan(column["refractivity_n"]))
    assert np.all(np.isfinite(column[WIND_SHEAR_VARIABLE])) and np.all(column[WIND_SHEAR_VARIABLE] >= 0.0)
    # a target above the column is refused (NaN), never extrapolated
    high, _ = operators.evaluate([truth], lat[:1], lon[:1], np.array([90_000.0]), np.array([1.0]),
                                 variables=("refractivity_n",))
    assert np.isnan(high["refractivity_n"][0, 0])
    # the same rows through the calibrated column function of obs_operators
    from woof.globe.obs_operators import RefractivityOperator

    class _Batch:
        latitude_deg, longitude_deg, elevation_m, variable = lat, lon, heights, "refractivity_n"

    reference = RefractivityOperator.for_model(model, transform)([truth], _Batch())
    assert reference[0] == pytest.approx(n, rel=1e-9)


def test_a_refractivity_batch_is_analysed_by_the_ensemble_and_localised_by_its_own_cutoff(world):
    cfg, ecfg, options, transform, model, cold = world
    ensemble = _ensemble(world)
    ensemble.step_all(ecfg.dt_s)
    truth = _truth(world)
    operators = MemberOperators.for_model(model, transform, ecfg)
    rng = np.random.default_rng(5)
    lat = rng.uniform(-60.0, 60.0, 40)
    lon = rng.uniform(0.0, 360.0, 40)
    heights = rng.uniform(2000.0, 12000.0, 40)
    levels = np.array([isa_pressure_pa(z) for z in heights])
    values, _ = operators.evaluate([truth], lat, lon, heights, levels, variables=("refractivity_n",))
    rows = [
        ObsRow("gnss-ro", f"C2E1-G{k:02d}", lat[k], lon[k], heights[k], levels[k], WHEN, "refractivity_n",
               float(values["refractivity_n"][0, k]), 0.01 * float(values["refractivity_n"][0, k]),
               measurement="ro_refractivity_tangent_point")
        for k in range(40)
    ]
    batches = batches_from_rows(rows, operators, ensemble.members)
    assert [(b.stream, b.variable) for b in batches] == [("gnss-ro", "refractivity_n")]
    batch = batches[0]
    # a target under the smoke column's lowest full level (four levels) is
    # refused, never extrapolated: the batch holds the rows inside the span
    assert 30 <= batch.count <= 40 and np.all(np.isfinite(batch.simulated))
    assert batch.simulated.shape == (6, batch.count)
    assert np.all(batch.measurement == "ro_refractivity_tangent_point")
    filter_options = FilterOptions(horizontal_cutoff_km=6000.0, gate_minimum_count=1000, thinning=False)
    assert filter_options.vertical_cutoff_for("refractivity_n", False) == 1.0
    assert filter_options.identity()["refractivity_vertical_cutoff_lnp"] == 1.0
    result = analyze_ensemble(ensemble, batches, filter_options, analysis_time=WHEN)
    report = result.report
    assert report["status"] == "pass", report["gate_of_record"]
    entry = report["streams"]["gnss-ro"]["refractivity_n"]
    cell = entry["regions"]["global"]["assimilated"]
    assert cell["count"] > 0
    assert cell["o_minus_a"]["rms"] < cell["o_minus_b"]["rms"]
    assert entry["desroziers"]["count"] > 0
    assert "refractivity" not in json.dumps(report["rejections"])


def _synthetic_stream_tables(cfg, checkpoint: Path, directory: Path) -> list[Path]:
    """One v2 table per stream at the fixture's instant, every value the
    background's own read plus a small pattern: METAR rows (iem-metar)
    and the same stations through the AWC cache (awc-metar), motion
    vectors at 500 hPa (goes-dmw, measured at an assigned pressure) and
    refractivity profiles (gnss-ro)."""
    transform = build_transform(cfg)
    model, _ = build_model_and_cold_state(cfg, transform)
    metadata, arrays = read_checkpoint(checkpoint)
    state = state_from_checkpoint(metadata, arrays, transform.backend)
    operators = MemberOperators.for_model(model, transform, cfg)
    lats = np.repeat(np.linspace(-60.0, 60.0, 8), 8)
    lons = np.tile(np.arange(0.0, 360.0, 45.0), 8)
    pattern = np.sin(np.deg2rad(lons)) * np.cos(np.deg2rad(lats))
    surface, _ = operators.evaluate([state], lats, lons, np.zeros(lats.size), np.full(lats.size, np.nan),
                                    variables=("surface_pressure_pa", "temperature_k"))
    metar, awc = [], []
    for k in range(lats.size):
        for variable, error, measurement, bump in (
                ("surface_pressure_pa", 100.0, "station_pressure_from_altimeter", 500.0 * pattern[k]),
                ("temperature_k", 1.5, "screen_temperature_2m", 2.0 * pattern[k])):
            value = float(surface[variable][0, k]) + bump
            metar.append(_row("iem-metar", f"ST{k:03d}", lats[k], lons[k], variable, value, error,
                              measurement=measurement))
            awc.append(_row("awc-metar", f"ST{k:03d}", lats[k], lons[k], variable, value, error,
                            measurement=measurement))
    amv_lat = np.linspace(-50.0, 50.0, 16)
    amv_lon = np.arange(10.0, 330.0, 20.0)
    winds, _ = operators.evaluate([state], amv_lat, amv_lon, np.zeros(16), np.full(16, 50000.0),
                                  variables=("wind_u_m_s", "wind_v_m_s"))
    amv = []
    for k in range(16):
        for variable in ("wind_u_m_s", "wind_v_m_s"):
            amv.append(_row("goes-dmw", f"G18-{k:02d}", amv_lat[k], amv_lon[k], variable,
                            float(winds[variable][0, k]) + 1.5 * np.sin(np.deg2rad(amv_lon[k])), 4.0,
                            level=50000.0, elev=5500.0, measurement=AMV_MEASUREMENT))
    ro_lat = np.linspace(-40.0, 40.0, 10)
    ro_lon = np.arange(5.0, 360.0, 36.0)
    heights = np.array([3000.0, 6000.0, 9000.0])
    ro = []
    for k in range(10):
        levels = np.array([isa_pressure_pa(z) for z in heights])
        values, _ = operators.evaluate([state], np.full(3, ro_lat[k]), np.full(3, ro_lon[k]), heights, levels,
                                       variables=("refractivity_n",))
        for j in range(3):
            n = float(values["refractivity_n"][0, j])
            ro.append(_row("gnss-ro", f"C2E1-G{k:02d}", ro_lat[k], ro_lon[k], "refractivity_n",
                           n * (1.0 + 0.01 * np.sin(np.deg2rad(ro_lon[k]))), 0.01 * n,
                           level=float(levels[j]), elev=float(heights[j]), measurement="ro_refractivity_tangent_point"))
    paths = []
    for name, rows in (("iem-metar", metar), ("awc-metar", awc), ("goes-dmw", amv), ("gnss-ro", ro)):
        path = directory / f"{name}.csv"
        obs_streams.write_table(rows, path)
        paths.append(path)
    return paths


def test_every_stream_in_the_tables_is_assimilated_and_the_roster_says_so(spun_up, tmp_path):
    cfg, checkpoint = spun_up
    tables = _synthetic_stream_tables(cfg, checkpoint, tmp_path)
    out = tmp_path / "streams"
    da_door.init(cfg, out, filter_name="letkf", members=3, ensemble_truncation=3,
                 analysis_time_utc=START_TEXT, options=OPTIONS,
                 filter_overrides={"amv_height_assignment_sigma_pa": "10000"})
    receipt = da_door.cycle(
        cfg, out, stream_specs=["local-tables:paths=" + ",".join(str(p) for p in tables)], cycles=1,
        start_utc=START_TEXT, interval_s=20.0, ensemble=out / ENSEMBLE_MANIFEST_NAME, options=OPTIONS,
        observation_bin_s=10.0, filter_overrides={"amv_height_assignment_sigma_pa": "10000"},
    )
    assert receipt["cycles"]["applied"] == 1 and receipt["status"] == "pass"
    report = json.loads((out / "assimilation-report-step00000002.json").read_text())
    assert report["status"] == "pass"
    roster = report["stream_roster"]
    assert roster["defects"] == []
    streams = roster["streams"]
    assert set(streams) == {"iem-metar", "awc-metar", "goes-dmw", "gnss-ro"}
    # the refractivity rows reached the filter: no refusal by name anywhere
    assert report["rejections"].get("no_operator", 0) == 0
    ro = streams["gnss-ro"]
    assert ro["offered"] == 30 and ro["assimilated"] > 0 and ro["refused_whole"] is False
    assert ro["latency_class"] == "retrospective" and ro["door"] == "rw_gnssro"
    assert ro["variables"]["refractivity_n"]["o_minus_b_rms"] > 0.0
    assert ro["variables"]["refractivity_n"]["o_minus_a_rms"] is not None
    assert "gnss-ro" in report["ensemble_scorecard"]["streams"]
    assert report["variables"]["refractivity_n"]["count"] > 0
    # the AWC copy of every METAR station is one instrument: the ties fall
    # to the stream name and the other stream's rows are counted
    metar, awc = streams["iem-metar"], streams["awc-metar"]
    assert metar["offered"] == 128 and awc["offered"] == 128
    assert metar["cross_stream_duplicates"] + awc["cross_stream_duplicates"] == 128
    assert metar["batched"] + awc["batched"] == 128
    assert metar["latency_class"] == "fast" and awc["latency_class"] == "fast"
    # the motion vectors carry their height-assignment shear in the error when the option is set
    amv = streams["goes-dmw"]
    assert amv["offered"] == 32 and amv["assimilated"] > 0
    inflation = report["amv_height_assignment"]
    assert inflation["applied"] is True and inflation["sigma_pa"] == 10_000.0
    assert inflation["streams"]["goes-dmw"]["rows"] == 32
    assert inflation["streams"]["goes-dmw"]["error_after_mean"] >= inflation["streams"]["goes-dmw"]["error_before_mean"]
    assert amv["amv_height_assignment"]["rows"] == 32
    assert report["options"]["filter"]["refractivity_vertical_cutoff_lnp"] == 1.0
    assert report["options"]["filter"]["amv_height_assignment_sigma_pa"] == 10_000.0
    # the window record carries the offered counts and the duplicates
    times = report["observation_times"]
    assert times["rows_offered_by_source"]["gnss-ro"] == 30
    assert sum(times["cross_stream_duplicates_by_source"].values()) == 128
    # the receipt's roster stacks the cycles with the streams module's classes
    assert receipt["stream_roster"]["cycles"][0]["streams"]["gnss-ro"]["assimilated"] > 0
    summary = receipt["stream_roster"]["summary"]
    assert summary["gnss-ro"]["latency_class"] == "retrospective"
    assert summary["gnss-ro"]["assimilated_per_cycle"] == [ro["assimilated"]]
    assert receipt["stream_roster"]["defects"] == []
    assert receipt["stream_roster"]["information_cutoff_utc"] is None


def test_a_filter_option_is_set_by_name_and_an_unknown_one_is_refused_with_the_list():
    from woof.globe.da.options import parse_filter_overrides
    from woof.globe.da_filter import resolve_filter

    base = FilterOptions()
    # the inflation is selectable and off by default: a bare analysis applies none
    assert base.amv_height_assignment_sigma_pa == 0.0 and base.refractivity_vertical_cutoff_lnp == 1.0
    from woof.globe.da_filter import _inflate_amv_errors
    bare = _inflate_amv_errors([], base.amv_height_assignment_sigma_pa)
    assert bare["applied"] is False and bare["sigma_pa"] == 0.0
    overrides = parse_filter_overrides(["amv_height_assignment_sigma_pa=10000", "refractivity_vertical_cutoff_lnp=1.5"])
    changed = base.with_overrides(overrides)
    assert changed.amv_height_assignment_sigma_pa == 10_000.0 and changed.refractivity_vertical_cutoff_lnp == 1.5
    assert changed.rtps_alpha == base.rtps_alpha
    with pytest.raises(ValueError, match="unknown filter option 'amv_sigma'.*amv_height_assignment_sigma_pa"):
        base.with_overrides({"amv_sigma": "0"})
    with pytest.raises(ValueError, match="NAME=VALUE"):
        parse_filter_overrides(["amv_height_assignment_sigma_pa"])
    # the validation still runs: a negative sigma is refused by the dataclass
    with pytest.raises(ValueError):
        base.with_overrides({"amv_height_assignment_sigma_pa": "-1"})
    # through the factory: the letkf filter carries it, the successive correction refuses it by name
    letkf = resolve_filter("letkf", options=AssimilationOptions(), members=3, truncation=3,
                           filter_overrides=overrides)
    assert letkf.filter_options.amv_height_assignment_sigma_pa == 10_000.0
    assert letkf.filter_options.identity()["amv_height_assignment_sigma_pa"] == 10_000.0
    with pytest.raises(ValueError, match="carries none of them"):
        resolve_filter("successive-correction", filter_overrides=overrides)
    # the command line hands the option to the door
    from woof.globe import cli
    parser = cli.build_parser() if hasattr(cli, "build_parser") else None
    if parser is not None:
        # A SHIPPED experiment, not a made-up filename.  This distribution
        # resolves a config at PARSE time, so that a mistyped experiment name
        # is an argument error with the shipped list beside it rather than a
        # traceback ten minutes into a run; a name that resolves to nothing is
        # refused before the namespace is built, and `c.toml` resolves to
        # nothing.
        from woof.globe.configs_dir import config_root

        config = str(config_root() / "arwen_global_moist_smoke.toml")
        args = parser.parse_args(["da", "cycle", config, "--outdir", "o", "--cycles", "1",
                                  "--filter-option", "amv_height_assignment_sigma_pa=0"])
        assert args.filter_option == ["amv_height_assignment_sigma_pa=0"]


def test_the_door_carries_every_stream_of_the_streams_module(monkeypatch, tmp_path):
    fetchable = {s.name for s in obs_streams.fetchable_streams()}
    assert fetchable <= set(da_streams.STREAM_TABLE)
    assert "cdaac-ro" in da_streams.STREAM_TABLE and "igra2" in da_streams.STREAM_TABLE
    assert "madis-aircraft" not in da_streams.STREAM_TABLE
    stream = da_streams.resolve_stream("cdaac-ro:missions=cosmic2;refresh_s=7200")
    assert isinstance(stream, da_streams.ObsStreamsStream)
    assert stream.name == "cdaac-ro" and stream.refresh_s == 7200.0 and stream.options == {"missions": "cosmic2"}
    assert da_streams.resolve_stream("igra2").refresh_s == da_streams.STREAM_REFRESH_S["igra2"]
    calls = []

    def fake_fetch(name, start, end, out_root, **options):
        calls.append((name, start, end, options))
        out = Path(out_root) / name
        out.mkdir(parents=True, exist_ok=True)
        table = out / f"{name}.csv"
        obs_streams.write_table([_row("gnss-ro", "C2E1-G01", 10.0, 20.0, "refractivity_n", 200.0, 2.0,
                                      level=50000.0, elev=5500.0, measurement="ro_refractivity_tangent_point")], table)
        manifest = {"table": {"rows": 1}, "latency_behind_real_time_s": 17594.0, "counters": {"profiles_read": 1}}
        return obs_streams.StreamFetch(stream=name, table=table, manifest=manifest, fetch_records=[])

    monkeypatch.setattr(obs_streams, "fetch_stream", fake_fetch)
    start = dt.datetime(2026, 9, 1, 18, 0, tzinfo=UTC)
    records = stream.fetch(start, start + dt.timedelta(hours=1), tmp_path / "fetch")
    assert len(records) == 1 and records[0].latency_behind_real_time_s == 17594.0
    assert records[0].latency_class == "replay" and records[0].decoder == "gpuwm-obs.table.v2"
    assert records[0].extra["latency_class_of_stream"] == "replay"
    assert calls[0][3]["missions"] == "cosmic2"
    assert calls[0][1] == "2026-09-01T17:30:00Z" and calls[0][2] == "2026-09-01T19:30:00Z"
    # inside the refresh window the same table serves the next window
    again = stream.fetch(start + dt.timedelta(minutes=30), start + dt.timedelta(minutes=90), tmp_path / "fetch")
    assert len(calls) == 1 and again[0].sha256 == records[0].sha256
    assert again[0].extra["reused_within_refresh_s"] == 7200.0
    # a stream that holds nothing for the window is an empty record with the reason

    def empty_fetch(name, start, end, out_root, **options):
        raise obs_streams.StreamEmpty("the archive ends at 2025-07-29")

    monkeypatch.setattr(obs_streams, "fetch_stream", empty_fetch)
    empty = da_streams.resolve_stream("gnss-ro").fetch(start, start + dt.timedelta(hours=1), tmp_path / "fetch")
    assert empty[0].bytes == 0 and empty[0].extra["empty"] is True
    assert "2025-07-29" in empty[0].extra["reason"]
    # a source that will not answer is one stream's failed record, not the
    # window's death: the reason is carried and the other streams go on

    def dead_fetch(name, start, end, out_root, **options):
        raise RuntimeError("rw_wis2 subscribe: globalbroker.meteo.fr:8883: connection refused")

    monkeypatch.setattr(obs_streams, "fetch_stream", dead_fetch)
    failed = da_streams.resolve_stream("wis2").fetch(start, start + dt.timedelta(hours=1), tmp_path / "fetch")
    assert failed[0].bytes == 0 and failed[0].extra["failed"] is True
    assert "connection refused" in failed[0].extra["reason"]
    assert failed[0].extra["latency_class_of_stream"] == "fast"


def test_a_configured_stream_whose_window_fetch_was_empty_or_failed_is_in_the_receipt_roster_with_zero_rows():
    """A stream the operator asked for that held nothing for a window (an
    EMPTY fetch record) or did not answer (a failed one) offers no rows and
    never reaches the filter's roster; the receipt's roster lists it all
    the same, with zero counts and the reason, and the causal record counts
    the fetch by its outcome rather than as an object whose arrival time
    nobody measured."""
    label = "2026-09-01T19:00:00+00:00"
    record = {
        "streams": [{"name": "wis2", "description": ""}, {"name": "cdaac-ro", "description": ""},
                    {"name": "local-tables", "description": ""}, {"name": "iem-metar", "description": ""}],
        "analyses": [{
            "analysis_time_utc": label, "status": "pass",
            "stream_roster": {
                "defects": [], "rows_outside_window": 0,
                "streams": {"iem-metar": {"offered": 10, "cross_stream_duplicates": 0, "batched": 10,
                                          "assimilated": 9, "withheld": 1, "refused": {}, "refused_whole": False,
                                          "latency_unverified": 10, "variables": {"temperature_k": {}}}},
            },
            "fetch": [
                {"stream": "local-tables", "latency_class": "unverified", "extra": {}},
                {"stream": "wis2", "latency_class": "unverified",
                 "extra": {"empty": True, "reason": "no payload in the window", "latency_class_of_stream": "fast"}},
                {"stream": "cdaac-ro", "latency_class": "unverified",
                 "extra": {"failed": True, "reason": "the portal answered 503", "latency_class_of_stream": "replay"}},
            ],
        }],
    }
    roster = da_door.stream_roster(record, None)
    cycle = roster["cycles"][0]["streams"]
    assert set(cycle) == {"iem-metar", "wis2", "cdaac-ro"}
    assert cycle["wis2"]["fetch"] == "empty" and cycle["wis2"]["offered"] == 0
    assert cycle["wis2"]["reason"] == "no payload in the window"
    assert cycle["cdaac-ro"]["fetch"] == "failed" and cycle["cdaac-ro"]["assimilated"] == 0
    summary = roster["summary"]
    assert summary["wis2"]["latency_class"] == "fast" and summary["wis2"]["door"] == "rw_wis2"
    assert summary["wis2"]["offered_per_cycle"] == [0] and summary["wis2"]["empty_fetch_cycles"] == [label]
    assert summary["cdaac-ro"]["failed_fetch_cycles"] == [label] and summary["cdaac-ro"]["assimilated_per_cycle"] == [0]
    assert summary["iem-metar"]["empty_fetch_cycles"] == [] and summary["iem-metar"]["assimilated_per_cycle"] == [9]
    assert roster["fetch_failures"] == [f"{label}: cdaac-ro: the portal answered 503"]
    assert roster["defects"] == []
    assert "local-tables" not in summary
    causal = da_door._causal_record(record, None)
    assert causal["latency_classes"] == {"unverified": 1, "empty": 1, "failed": 1}
    assert causal["mode"].startswith("latency unverified")
    # a configured stream that reached no cycle at all is still listed
    bare = da_door.stream_roster({"streams": [{"name": "ndbc"}], "analyses": record["analyses"]}, None)
    assert bare["summary"]["ndbc"]["offered_per_cycle"] == [0]
    assert bare["summary"]["ndbc"]["note"].startswith("configured on the door")


def test_the_cross_stream_rule_collapses_rows_within_ten_minutes_and_keeps_rows_further_apart_whatever_the_bin_edges():
    """The rule says ten minutes: two streams' reports of one station nine
    minutes apart are one report and fifteen minutes apart are two, at
    every offset against the time bins the rule uses to limit its
    candidates (a 20-minute bin also holds pairs 11 to 19 minutes apart)."""
    a = _row("iem-metar", "KAAA", 40.0, -100.0, "temperature_k", 290.0, 1.0, minutes=0.0)
    b = _row("awc-metar", "KAAA", 40.0, -100.0, "temperature_k", 290.5, 1.2, minutes=9.0)
    c = _row("wis2", "KAAA", 40.0, -100.0, "temperature_k", 291.0, 0.9, minutes=25.0)
    kept, dropped = cross_stream_duplicates([a, b, c])
    assert [r.source for r in kept] == ["iem-metar", "wis2"] and dropped == {"awc-metar": 1}
    for start in range(0, 41, 3):
        far = [_row("iem-metar", "KBBB", 41.0, -101.0, "temperature_k", 290.0, 1.0, minutes=start),
               _row("awc-metar", "KBBB", 41.0, -101.0, "temperature_k", 290.0, 1.0, minutes=start + 15.0)]
        near = [_row("iem-metar", "KCCC", 42.0, -102.0, "temperature_k", 290.0, 1.0, minutes=start),
                _row("awc-metar", "KCCC", 42.0, -102.0, "temperature_k", 290.0, 1.0, minutes=start + 9.0)]
        assert len(cross_stream_duplicates(far)[0]) == 2, start
        assert len(cross_stream_duplicates(near)[0]) == 1, start
        # the same by position cell, no station id
        near_cell = [_row("ndbc", "", 43.004, -103.004, "temperature_k", 290.0, 1.0, minutes=start),
                     _row("iem-metar", "", 43.006, -103.006, "temperature_k", 290.0, 1.0, minutes=start + 9.0)]
        far_cell = [_row("ndbc", "", 43.004, -103.004, "temperature_k", 290.0, 1.0, minutes=start),
                    _row("iem-metar", "", 43.006, -103.006, "temperature_k", 290.0, 1.0, minutes=start + 15.0)]
        assert len(cross_stream_duplicates(near_cell)[0]) == 1, start
        assert len(cross_stream_duplicates(far_cell)[0]) == 2, start


def test_the_error_table_scales_the_refractivity_profile_error_and_keeps_the_constant_cells():
    """The refractivity rows' assigned error is a profile (the Kuo fraction
    of each row's own value), so the calibration lays a SCALE over it, the
    root of the record's Desroziers ratio (1.965 over the six analyses of
    the grade of record); a constant cell still broadcasts its value, a cell
    neither table names keeps the rows' own errors, and the receipt's block
    says which of the two was laid over the batch."""
    from woof.globe.da import DESROZIERS_ERROR_SCALE_TABLE, DESROZIERS_ERROR_TABLE
    from woof.globe.da.observation_errors import calibrated_error

    profile = np.array([7.54, 0.62, 0.18])  # the Kuo fraction of 314 N at 0.8 km, 60 N at 25 km, ...
    for route in ("cdaac-ro", "gnss-ro"):
        errors, entry = calibrated_error("desroziers-2026-09-06", route, "refractivity_n", profile)
        assert entry == {"scale": 1.4} and DESROZIERS_ERROR_SCALE_TABLE[(route, "refractivity_n")] == 1.4
        assert errors.tolist() == pytest.approx((profile * 1.4).tolist())
        assert 1.4 == pytest.approx(np.sqrt(1.965), abs=0.003)
    assert ("cdaac-ro", "refractivity_n") not in DESROZIERS_ERROR_TABLE
    errors, entry = calibrated_error("desroziers-2026-09-06", "igra2", "wind_u_m_s", np.full(4, 2.5))
    assert entry == 3.1 and errors.tolist() == [3.1] * 4
    errors, entry = calibrated_error("desroziers-2026-09-06", "wis2", "refractivity_n", profile)
    assert entry is None and errors.tolist() == profile.tolist()
    errors, entry = calibrated_error(None, "cdaac-ro", "refractivity_n", profile)
    assert entry is None and errors.tolist() == profile.tolist()
