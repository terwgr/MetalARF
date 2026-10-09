"""`woof global da`: init, cycle, analyze, fresh and forecast.

The door is driven end to end on the CPU at the smoke truncation: a
synthetic cycle through a stream (the door's local-tables stream over the
synthetic METAR and aircraft tables), every report with its DA scorecard,
the lineage in the checkpoint chain, the ensemble manifest re-written at
the last analysis, the wall budget in the receipt; then `fresh` on the
analytic smoke config from `init` to the handed-back checkpoint and
`forecast` from it.  Every refusal names its breakage.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from woof.globe import da_door
from woof.globe.assimilate import ASSIMILATION_HISTORY_KEY
from woof.globe.checkpoint import read_checkpoint
from woof.globe.cli import EXIT_REFUSED, _da_progress, main as cli_main
from woof.globe.config import load_config
from woof.globe.da_filter import (
    ENSEMBLE_MANIFEST_NAME,
    read_ensemble_manifest,
    resolve_filter,
)
from woof.globe import da_streams
from woof.globe.da_streams import (
    FETCH_MANIFEST_SCHEMA,
    LocalTableStream,
    _url_file_name,
    resolve_stream,
)

from test_arwen_global_assimilate import (  # noqa: F401 - the fixture rides the import
    CONFIG, _synthetic_obs_files, spun_up,
)
from test_arwen_global_cycle import OBS_TIME, START, START_TEXT, OPTIONS


def _stream_spec(paths) -> str:
    return "local-tables:paths=" + ",".join(str(p) for p in paths)


def _shifted_obs_files(paths, seconds: int) -> list[Path]:
    """Copies of the fixture's tables with every report ``seconds`` later
    (the fixture writes one instant, 2026-08-31T12:00:00Z): a later
    instant is a new report identity, so the copies are a second
    window's network."""
    later = (OBS_TIME + dt.timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        assert "2026-08-31T12:00:00Z" in text
        target = Path(path).with_name(f"later-{Path(path).name}")
        target.write_text(text.replace("2026-08-31T12:00:00Z", later), encoding="utf-8")
        out.append(target)
    return out


def test_a_window_with_no_report_carries_the_background_and_says_so(spun_up, tmp_path):
    """The second of two windows holds no report (every row sits at
    12:00:00, the first window's end, and a report earlier than a
    window's start is not analysed at that window's instant: the
    2026-09-06 refutation found the rows the first cycle thinned away
    analysed again twenty seconds after their own time): the letkf filter
    carries the background as that hour's checkpoint, the hour is CARRIED
    by name, the rows outside the window and the rows the chain already
    holds are counted, the first analysis is what the door hands back,
    and the members advanced through the window all the same."""
    cfg, checkpoint = spun_up
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    out = tmp_path / "letkf-empty"
    da_door.init(cfg, out, filter_name="letkf", members=3, ensemble_truncation=3,
                 analysis_time_utc=START_TEXT, options=OPTIONS)
    receipt = da_door.cycle(
        cfg, out, stream_specs=[_stream_spec(obs)], cycles=2, start_utc=START_TEXT,
        interval_s=20.0, ensemble=out / ENSEMBLE_MANIFEST_NAME, options=OPTIONS,
        observation_bin_s=10.0,
    )
    assert receipt["cycles"] == {"planned": 2, "completed": 2, "applied": 1, "carried": 1, "partial": 0}
    assert receipt["status"] == "incomplete"
    handed = receipt["analysis_checkpoint"]
    assert handed["step"] == 2 and Path(handed["path"]).is_file()
    metadata, _ = read_checkpoint(handed["path"], expected_config_hash=cfg.config_hash)
    assert len(metadata["physics_metadata"][ASSIMILATION_HISTORY_KEY]["cycles"]) == 1
    assert receipt["lineage"]["chain_length"] == 1
    after = read_ensemble_manifest(out / ENSEMBLE_MANIFEST_NAME)
    assert after["deterministic"]["self_sha256"] == handed["self_sha256"]
    assert len(after["members"]) == 3 and after["ensemble_manifest"]["cycles"] == 1
    carried = json.loads((out / "assimilation-report-step00000004.json").read_text())
    assert carried["status"] == "carried" and carried["assimilated_total"] == 0
    assert carried["rejections"]["outside_window"] == 304
    assert carried["rejections"]["already_assimilated"] > 0
    assert "outside_window" in carried["rejection_breakage"]
    assert "not analysed at this instant" in carried["rejection_breakage"]["outside_window"]
    assert carried["carried"]["reason"] == carried["carried_reason"]
    assert "no admissible report in the window" in carried["carried"]["reason"]
    assert carried["gate_of_record"]["failed"] == [carried["carried_reason"]]
    assert carried["scorecard"]["verdict"] == "incomplete"
    assert carried["assessments"]["engineering"]["verdict"] == "carried"
    assert carried["assessments"]["engineering"]["rows_offered"] == 304
    assert carried["observation_times"]["rows"] == 0 and carried["observation_times"]["rows_per_bin"] == [0, 0]
    assert carried["observation_times"]["rows_outside_window"] == 304
    assert (out / "arwen_global_step00000004.npz").is_file()
    assert receipt["scorecards"]["complete_cycles"] == 1
    run_record = json.loads(Path(receipt["run_receipt"]["path"]).read_text())["cycle"]
    assert [a["applied"] for a in run_record["analyses"]] == [True, False]
    assert run_record["analyses"][1]["refused_from_chain"] == carried["rejections"]["already_assimilated"]
    second = json.loads(Path(run_record["analyses"][1]["report"]).read_text())
    assert second["status"] == "carried" and second["carried_reason"] == carried["carried_reason"]


def test_init_writes_the_initial_state_the_manifest_and_the_receipt(tmp_path):
    cfg = load_config(CONFIG)
    out = tmp_path / "init"
    receipt = da_door.init(cfg, out, analysis_time_utc=START_TEXT, config_path=CONFIG)
    assert receipt["status"] == "pass" and receipt["filter"] == "successive-correction"
    assert receipt["members"] == 0
    initial = Path(receipt["analysis_checkpoint"]["path"])
    assert initial == out / da_door.INITIAL_STATE_NAME and initial.is_file()
    metadata, _ = read_checkpoint(initial, expected_config_hash=cfg.config_hash)
    assert metadata["step"] == 0 and receipt["analysis_checkpoint"]["self_sha256"] == metadata["self_sha256"]
    manifest = read_ensemble_manifest(out / ENSEMBLE_MANIFEST_NAME)
    assert manifest["filter"] == "successive-correction" and manifest["members"] == []
    assert manifest["deterministic"]["self_sha256"] == metadata["self_sha256"]
    assert manifest["analysis_time_utc"] == START_TEXT
    assert manifest["lineage"]["analyses"] == 0
    assert receipt["ensemble_manifest"]["self_sha256"] == manifest["self_sha256"]
    # The receipt reads back under its own schema and hash.
    back = da_door.read_da_receipt(out / da_door.DA_RECEIPT_NAME)
    assert back["door"] == "woof global da init"
    # A second init is refused until --overwrite.
    with pytest.raises(FileExistsError, match="--overwrite"):
        da_door.init(cfg, out, analysis_time_utc=START_TEXT)


def test_refusals_name_their_breakage(tmp_path):
    cfg = load_config(CONFIG)
    with pytest.raises(ValueError, match="letkf"):
        da_door.init(cfg, tmp_path / "two", members=2)
    # The ensemble filter refuses what its options cannot run, by name.
    with pytest.raises(ValueError, match="at least 3"):
        resolve_filter("letkf", members=2)
    with pytest.raises(ValueError, match="init or attach"):
        resolve_filter("letkf", members=3, truncation=3).analyse(
            cfg, None, None, [None], [], sources=[], background={}, analysis_time=None, options=None)
    with pytest.raises(ValueError, match="unknown analysis filter"):
        resolve_filter("nudging")
    with pytest.raises(ValueError, match="unknown observation stream"):
        resolve_stream("madis")
    with pytest.raises(ValueError, match="key=value"):
        resolve_stream("iem-asos:networks")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        LocalTableStream(("nowhere.csv",)).fetch(START, OBS_TIME, tmp_path)
    # fresh on an analytic base needs the instant model time zero stands for.
    with pytest.raises(ValueError, match="--start-utc"):
        da_door.fresh(CONFIG, tmp_path / "fresh-a", stream_specs=["local-tables:paths=x.csv"])
    # ...and refuses when the newest observation hour is not a whole
    # interval after the start: nothing to cycle yet.
    with pytest.raises(ValueError, match="nothing to cycle yet"):
        da_door.fresh(
            CONFIG, tmp_path / "fresh-b", stream_specs=["local-tables:paths=x.csv"],
            start_utc=START_TEXT, until_utc=START_TEXT, interval_s=20.0,
        )
    # The cycle needs something to analyse.
    with pytest.raises(ValueError, match="at least one --obs source"):
        da_door.cycle(cfg, tmp_path / "empty", cycles=1, start_utc=START_TEXT, interval_s=20.0)


def test_a_table_fetched_from_a_url_lands_under_a_name_without_its_query(tmp_path, monkeypatch):
    """The IEM archive's CSV service answers at ``asos.py?station=...``;
    the query is the request, not the file, and a name carrying ``?``,
    ``&`` and ``%`` is refused by Windows (found on the first real fetch
    through this route, 2026-09-06).  The record still carries the URL,
    the bytes and the digest of what was written."""
    assert _url_file_name("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=DSM&data=tmpf&tz=Etc%2FUTC") == "asos.py"
    assert _url_file_name("https://host.example/tables/obs-2026.csv.gz#top") == "obs-2026.csv.gz"
    assert _url_file_name("https://host.example/") == "table.csv"
    assert _url_file_name("https://host.example/a b?c") == "a_b"
    text = "station,valid,lon,lat,elevation,tmpf,dwpf,sknt,drct,alti\nAMW,2026-09-01 00:53,-93.6,41.9,280.0,80.0,77.0,5.0,210.0,29.90\n"
    monkeypatch.setattr(da_streams, "fetch_obs", lambda location: (text, {"sha256": "ab" * 32, "location": location}))
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=AMW&data=tmpf&year1=2026"
    records = LocalTableStream((url,)).fetch(START, OBS_TIME, tmp_path / "fetch")
    assert len(records) == 1
    landed = Path(records[0].path)
    assert landed.parent == tmp_path / "fetch" and landed.name == "abababababababab-asos.py"
    assert landed.read_text(encoding="utf-8") == text
    assert records[0].location == url and records[0].bytes == len(text.encode("utf-8"))
    assert records[0].latency_class in ("fast", "replay", "retrospective")
    assert records[0].decoder == "iem-asos-csv"


def test_the_synthetic_cycle_runs_end_to_end_through_a_stream(spun_up, tmp_path):
    """init from the cold start, two hourly (here 20 s) cycles through the
    local-tables stream, the deterministic analysis handed back with its
    lineage, the manifest re-written, the scorecard complete on every
    cycle and the wall budget in the receipt."""
    cfg, checkpoint = spun_up
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    out = tmp_path / "da"
    init = da_door.init(cfg, out, analysis_time_utc=START_TEXT)
    receipt = da_door.cycle(
        cfg, out, stream_specs=[_stream_spec(obs)], cycles=2, start_utc=START_TEXT,
        interval_s=20.0, ensemble=out / ENSEMBLE_MANIFEST_NAME, options=OPTIONS,
        config_path=CONFIG,
    )
    # Both cycles applied and engineering-complete (amendment G: the status
    # is engineering validity); the second cycle's aircraft u-wind reads
    # O-A above O-B on its one row and that is a recorded reading below,
    # not a failure.
    assert receipt["status"] == "pass"
    assert receipt["cycles"] == {"planned": 2, "completed": 2, "applied": 2, "carried": 0, "partial": 0}
    assert receipt["settings"]["increment_application"] == "iau"
    assert receipt["causal"]["latency_classes"] == {"unverified": 4}
    assert receipt["causal"]["mode"].startswith("latency unverified")
    # The row-level cutoff (none declared here): every decoded row kept,
    # every one counted as latency-unverified (the cache tables carry no
    # receipt time).
    rows = receipt["causal"]["rows"]
    assert rows["kept"] == rows["offered"] > 0 and rows["after_cutoff"] == 0
    assert rows["latency_unverified"] == rows["offered"]
    assert receipt["filter"] == "successive-correction"
    assert [s["name"] for s in receipt["streams"]] == ["local-tables"]

    # The analysis handed back: the last applied analysis, at step 4.
    handed = receipt["analysis_checkpoint"]
    analysis = Path(handed["path"])
    assert analysis == out / "arwen_global_analysis_step00000004.npz" and analysis.is_file()
    metadata, _ = read_checkpoint(analysis, expected_config_hash=cfg.config_hash)
    assert metadata["self_sha256"] == handed["self_sha256"] and handed["step"] == 4

    # Lineage: two links in the checkpoint's chain, each naming the filter
    # and the streams (obs-table sources) that fed it; the receipt's
    # lineage is the last link's.
    chain = metadata["physics_metadata"][ASSIMILATION_HISTORY_KEY]
    assert len(chain["cycles"]) == 2
    assert chain["cycles"][0]["streams"] == ["awc-aircraft-cache", "awc-metar-cache"]
    assert all(link["filter"] == "successive-correction" for link in chain["cycles"])
    assert receipt["lineage"]["chain_length"] == 2
    assert receipt["lineage"]["filter"] == "successive-correction"

    # The manifest now names the analysis, with the init manifest as its
    # predecessor.
    manifest = read_ensemble_manifest(out / ENSEMBLE_MANIFEST_NAME)
    assert manifest["deterministic"]["self_sha256"] == handed["self_sha256"]
    assert manifest["lineage"]["analyses"] == 2
    assert manifest["previous_manifest_sha256"] == init["ensemble_manifest"]["self_sha256"]
    assert manifest["analysis_time_utc"] == (START + dt.timedelta(seconds=40)).isoformat(timespec="seconds")

    # The scorecard per cycle.  Both cycles are engineering-complete on
    # both streams.  The second cycle's offer is the withheld tenth of the
    # first (the split is per variable over every source, so it holds
    # six or seven METAR rows and ONE aircraft row per variable); the
    # METAR stream moves closer on every variable, and the aircraft
    # stream's single u-wind row reads O-A above O-B (0.878 against
    # 0.874 m/s: one row the rotational wind increment could not pull),
    # which amendment G records as a reading, never as a failure.
    cards = receipt["scorecards"]
    second = (START + dt.timedelta(seconds=40)).isoformat(timespec="seconds")
    assert cards["complete_cycles"] == 2 and cards["incomplete_cycles"] == []
    metar = cards["streams"]["awc-metar-cache"]
    assert metar["surface_pressure_pa"]["complete"] == 2 and metar["surface_pressure_pa"]["o_a_below_o_b"] == 2
    assert all(a < b for a, b in zip(metar["temperature_k"]["o_minus_a_rms"], metar["temperature_k"]["o_minus_b_rms"]))
    aircraft = cards["streams"]["awc-aircraft-cache"]
    assert aircraft["temperature_k"]["cycles"] == 2 and aircraft["temperature_k"]["o_a_below_o_b"] == 2
    assert aircraft["wind_u_m_s"]["o_a_not_below_o_b_cycles"] == [second]
    assert aircraft["wind_u_m_s"]["incomplete_cycles"] == []
    assert aircraft["wind_u_m_s"]["rows"][1] == 1
    assert aircraft["wind_u_m_s"]["o_minus_a_rms"][1] >= aircraft["wind_u_m_s"]["o_minus_b_rms"][1]
    # The four assessments per cycle ride in the receipt; the imbalance
    # reading (the first step after the analysis) is filled for the
    # first cycle, which had a step after it.
    assessments = receipt["assessments"]
    assert [a["engineering"] for a in assessments] == ["pass", "pass"]
    assert assessments[1]["o_a_not_below_o_b"] == ["awc-aircraft-cache/wind_u_m_s"]
    tendency = assessments[0]["physical_consistency"]["surface_pressure_tendency_rms_pa_s"]
    assert tendency["first_step_after_analysis"] > 0.0 and tendency["last_step_before_analysis"] > 0.0

    # The wall budget: every cycle's wall against its 20 s interval.
    budget = receipt["budget"]
    assert budget["interval_s"] == 20.0 and len(budget["cycles"]) == 2
    for row in budget["cycles"]:
        assert row["wall_s"] > 0.0
        assert row["real_time_fraction"] == pytest.approx(row["wall_s"] / 20.0)
        assert row["wall_s"] == pytest.approx(
            row["forecast_steps_wall_s"] + row["fetch_s"] + row["analysis_s"]
            + row["checkpoint_submit_s"]
            + json.loads(Path(receipt["run_receipt"]["path"]).read_text())["cycle"]["analyses"][budget["cycles"].index(row)]["timings_s"]["background_identity_s"])
        # The deterministic filter has no members to advance; under the
        # incremental update (the default) the control re-integrates its
        # window, and the time is recorded.
        assert row["members_advance_s"] == 0.0 and row["iau_reintegration_s"] >= 0.0
    assert budget["max_wall_s"] >= budget["mean_wall_s"] > 0.0

    # The fetch: one manifest per window naming the tables with their
    # digests, and the report carrying the records and the filter.
    manifests = sorted((out / "fetch").glob("da-fetch-*.json"))
    assert len(manifests) == 2
    fetched = json.loads(manifests[0].read_text())
    assert fetched["schema"] == FETCH_MANIFEST_SCHEMA
    assert {r["path"] for r in fetched["records"]} == {str(p) for p in obs}
    assert all(len(r["sha256"]) == 64 and r["bytes"] > 0 for r in fetched["records"])
    assert {r["decoder"] for r in fetched["records"]} == {"awc-metar-cache", "awc-aircraft-cache"}
    assert fetched["latency_classes"] == {"unverified": 2}
    assert all(r["first_receipt_utc"] == r["fetched_utc"] and r["publication_utc"] is None for r in fetched["records"])
    report = json.loads((out / "assimilation-report-step00000002.json").read_text())
    assert report["filter"] == "successive-correction"
    assert report["fetch"][0]["stream"] == "local-tables" and report["fetch"][0]["manifest"] == str(manifests[0])
    assert report["scorecard"]["verdict"] == "complete"
    assert report["scorecard"]["assessments"]["engineering"]["verdict"] == "pass"
    assert report["observation_times"]["bin_s"] is None
    assert report["increment_application"]["mode"] == "iau"
    assert report["lineage"]["chain_length"] == 1
    assert report["timings_s"]["fetch_s"] >= 0.0
    # The second cycle decoded nothing new (the same tables) and analysed
    # the withheld tenth the chain lacked.
    second = json.loads((out / "assimilation-report-step00000004.json").read_text())
    assert second["rejections"]["already_assimilated"] == report["assimilated_total"]
    # The run receipt is the cycle door's own, with the same records.
    run_receipt = json.loads(Path(receipt["run_receipt"]["path"]).read_text())
    assert run_receipt["cycle"]["filter"] == "successive-correction"
    assert run_receipt["cycle"]["budget"]["every_cycle_keeps_up"] is True
    assert run_receipt["cycle"]["scorecards"]["complete_cycles"] == 2


def test_the_letkf_filter_drives_the_ensemble_package_through_the_door(spun_up, tmp_path):
    """The dual-resolution ensemble filter at the smoke truncation: three
    members at T3 built by init from the ensemble config's own cold start,
    two cycles through the stream, the deterministic analysis handed back
    with the ensemble-mean increment applied and the members recentred,
    the door manifest naming the member checkpoints and the package's
    manifest, the card judged on the deterministic state."""
    cfg, checkpoint = spun_up
    # Reports for both windows: the fixture's instant (the end of the first
    # window) and the same network 20 s later (the end of the second); a
    # report is compared in the window it falls in and nowhere else.
    (tmp_path / "later").mkdir()
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path) + _shifted_obs_files(
        _synthetic_obs_files(cfg, checkpoint, tmp_path / "later"), 20)
    out = tmp_path / "letkf"
    init = da_door.init(
        cfg, out, filter_name="letkf", members=3, ensemble_truncation=3,
        analysis_time_utc=START_TEXT, options=OPTIONS,
    )
    assert init["filter"] == "letkf" and init["members"] == 3
    manifest = read_ensemble_manifest(out / ENSEMBLE_MANIFEST_NAME)
    assert manifest["filter"] == "letkf" and len(manifest["members"]) == 3
    assert manifest["ensemble_store"] == "ensemble"
    assert manifest["ensemble_options"]["members"] == 3 and manifest["ensemble_options"]["truncation"] == 3
    assert (out / "ensemble" / "arwen-global-ensemble.json").is_file()
    for entry in manifest["members"]:
        assert (out / entry["checkpoint"]).is_file()
    assert manifest["ensemble_manifest"]["spread"]["temperature_k"] > 0.0

    assert manifest["control_options"]["increment_source"] == "control"
    assert manifest["ensemble_options"]["additive_inflation_fraction"] == 0.0

    receipt = da_door.cycle(
        cfg, out, stream_specs=[_stream_spec(obs)], cycles=2, start_utc=START_TEXT,
        interval_s=20.0, ensemble=out / ENSEMBLE_MANIFEST_NAME, options=OPTIONS,
        observation_bin_s=10.0,
    )
    assert receipt["filter"] == "letkf"
    assert receipt["cycles"]["applied"] == 2
    assert receipt["settings"]["observation_bin_s"] == 10.0
    handed = receipt["analysis_checkpoint"]
    assert handed["step"] == 4 and Path(handed["path"]).is_file()
    metadata, _ = read_checkpoint(handed["path"], expected_config_hash=cfg.config_hash)
    chain = metadata["physics_metadata"][ASSIMILATION_HISTORY_KEY]
    assert len(chain["cycles"]) == 2
    assert chain["cycles"][-1]["filter"] == "letkf"
    assert chain["cycles"][-1]["streams"] == ["awc-aircraft-cache", "awc-metar-cache"]
    assert receipt["lineage"]["filter"] == "letkf" and receipt["lineage"]["chain_length"] == 2
    after = read_ensemble_manifest(out / ENSEMBLE_MANIFEST_NAME)
    assert after["deterministic"]["self_sha256"] == handed["self_sha256"]
    assert after["previous_manifest_sha256"] == manifest["self_sha256"]
    assert len(after["members"]) == 3 and after["ensemble_manifest"]["cycles"] == 2
    report = json.loads((out / "assimilation-report-step00000002.json").read_text())
    assert report["filter"] == "letkf" and report["members"] == 3
    assert report["scorecard"]["verdict"] == "complete" and report["status"] == "pass"
    assert "control analysis" in report["scorecard_judges"]
    assert set(report["ensemble_scorecard"]["streams"]) == {"awc-metar-cache", "awc-aircraft-cache"}
    # Amendment A: the control's own increment, the ensemble-mean increment
    # recorded beside it and not applied.
    control = report["control"]
    assert control["increment_source"] == "control"
    assert "embedded in the T3 triangle" in control["route"]
    assert "the control's own (amendment A)" in control["innovation"]
    assert report["letkf"]["active_columns"] > 0
    assert report["letkf"]["grid_control_increment_rms"]["theta"] > 0.0
    comparison = report["mean_increment_transfer"]
    assert "NOT applied" in comparison["note"]
    assert comparison["control_increment_rms"]["temperature_k_rms"] > 0.0
    assert comparison["difference_rms"]["temperature_k_rms"] > 0.0
    # Amendment C: the taper (the package's default at T3: one to degree
    # 2, zero at 3) and the spectrum inspection by band.
    taper = control["taper"]
    assert taper["start_degree"] == 2 and taper["end_degree"] == 3
    assert taper["weights_by_degree"] == pytest.approx([1.0, 1.0, 1.0, 0.0])
    before = control["spectrum_before_taper"]["temperature_k2"]
    after = control["spectrum_after_taper"]["temperature_k2"]
    assert after["by_degree"][3] == 0.0 and before["by_degree"][:3] == pytest.approx(after["by_degree"][:3])
    assert after["small_scale_share_above_0p6T"] <= before["small_scale_share_above_0p6T"]
    assert control["increment"]["temperature_k_rms"] > 0.0
    # Amendment B: reports at their own times.  The synthetic reports sit
    # at the analysis instant of the FIRST cycle (12:00:00, the last bin),
    # so every row is observed at the window's end there; the second
    # cycle's window holds no report at all and the bins are recorded
    # empty.
    times = report["observation_times"]
    assert times["bin_s"] == 10.0 and times["bins"] == 2 and times["bin_observe_times_s"] == [10.0, 20.0]
    # 64 stations x 4 variables plus 16 aircraft x 3 variables: one row per
    # variable per report.
    assert times["rows"] == 304 and times["rows_per_bin"] == [0, 304]
    assert times["rows_at_own_time"] == 0 and times["complete"] == {"members": True, "control": True}
    # Amendment D: RTPS alone; recentring full.
    assert report["inflation"]["additive_fraction"] == 0.0 and report["inflation"]["applied"] is False
    assert report["recentre"]["fraction"] == 1.0
    assert report["recentre"]["mean_shift_grid_rms"]["theta"] >= 0.0
    # Amendment G: the four assessments, the package's own beside them.
    assessments = report["assessments"]
    assert assessments["engineering"]["verdict"] == "pass"
    assert set(assessments) == {"engineering", "statistical_consistency", "physical_consistency", "predictive_value", "package"}
    assert assessments["package"]["engineering_validity"]["verdict"] == "pass"
    assert assessments["physical_consistency"]["control"]["mass_preserving_log_offset"] is not None
    assert assessments["physical_consistency"]["global_mass_budget"]["log_offset_maxabs"] >= 0.0
    assert report["spread"]["before"]["temperature_k"] > 0.0
    assert report["timings_s"]["analysis_phases"]["members_advance_s"] >= 0.0
    assert report["timings_s"]["analysis_phases"]["ensemble_control_s"] >= 0.0
    # A cycle that names the wrong filter for the manifest is refused by name.
    with pytest.raises(ValueError, match="was built by the 'letkf' filter"):
        da_door.cycle(
            cfg, tmp_path / "wrong", stream_specs=[_stream_spec(obs)], cycles=1,
            start_utc=START_TEXT, interval_s=20.0, ensemble=out / ENSEMBLE_MANIFEST_NAME,
            filter_name="successive-correction", options=OPTIONS,
        )
    # analyze forms one deterministic analysis; the ensemble filter goes
    # through the cycle leg.
    with pytest.raises(ValueError, match="through the cycle leg"):
        da_door.analyze(cfg, checkpoint, [str(p) for p in obs], tmp_path / "an", filter_name="letkf")


def test_analyze_is_the_file_door_with_its_scorecard_and_receipt(spun_up, tmp_path, capsys):
    cfg, checkpoint = spun_up
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    out = tmp_path / "analyze"
    receipt = da_door.analyze(
        cfg, checkpoint, [str(p) for p in obs], out, analysis_time=OBS_TIME,
        options=OPTIONS, config_path=CONFIG,
    )
    printed = capsys.readouterr().out
    assert "DA scorecard" in printed and "COMPLETE" in printed
    assert receipt["status"] == "pass"
    assert receipt["scorecard"]["verdict"] == "complete"
    assert Path(receipt["analysis_checkpoint"]["path"]).is_file()
    assert receipt["report"]["path"] == str(out / "assimilation-report.json")
    assert receipt["lineage"]["chain_length"] == 1
    # The scorecard door renders the report's card.
    from woof.globe.da_scorecard import main as scorecard_main

    assert scorecard_main(["show", receipt["report"]["path"], "--region", "global"]) == 0
    assert "awc-metar-cache" in capsys.readouterr().out


def test_fresh_on_an_analytic_config_inits_cycles_and_hands_back_a_forecast_start(spun_up, tmp_path):
    """The one command on the smoke case: nothing fetched (analytic base),
    the derived config, init, two cycles to the newest observation hour,
    the handed-back checkpoint and the forecast command; then the
    forecast door runs from it to the derived duration."""
    cfg, checkpoint = spun_up
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    out = tmp_path / "fresh"
    until = START + dt.timedelta(seconds=40)
    receipt = da_door.fresh(
        CONFIG, out, stream_specs=[_stream_spec(obs)], start_utc=START_TEXT,
        until_utc=until.isoformat(), forecast_hours=20.0 / 3600.0, interval_s=20.0,
        options=OPTIONS, now=START + dt.timedelta(hours=1), filter_name="successive-correction",
    )
    # The receipt says which parts the door defaulted (here the member
    # count of the named filter; the streams were named).
    assert receipt["defaults"]["defaulted"] == ["members"] and receipt["defaults"]["members"] == 1
    # Both cycles applied and handed back; the verdict is the scorecard's
    # (a six-row second offer at T3 can read incomplete on one wind
    # component), and the door says which.
    assert receipt["status"] in ("pass", "incomplete")
    assert receipt["cycle"]["cycles"]["applied"] == 2
    cards = receipt["scorecards"]
    assert cards["complete_cycles"] + len(cards["incomplete_cycles"]) == 2
    assert (receipt["status"] == "pass") == (cards["complete_cycles"] == 2)
    assert receipt["initial_state"].startswith("analytic")
    assert receipt["analysis"] is None
    assert receipt["cycles"] == 2 and receipt["interval_s"] == 20.0
    assert receipt["last_observation_hour_utc"] == until.isoformat(timespec="seconds")
    derived = Path(receipt["config"])
    assert derived == out / da_door.FRESH_CONFIG_NAME and derived.is_file()
    fresh_cfg = load_config(derived)
    assert fresh_cfg.duration_s == pytest.approx(60.0)
    assert fresh_cfg.name.endswith("-fresh")
    assert receipt["config_hash"] == fresh_cfg.config_hash
    assert receipt["init"] is not None and receipt["ensemble_manifest"] is not None
    handed = receipt["analysis_checkpoint"]
    assert handed["step"] == 4 and Path(handed["path"]).is_file()
    assert receipt["budget"]["every_cycle_keeps_up"] is True
    assert receipt["forecast_command"].startswith("woof global da forecast ")
    assert Path(handed["path"]).as_posix() in receipt["forecast_command"]

    forecast = da_door.forecast(
        fresh_cfg, out / "forecast", analysis=handed["path"], config_path=derived,
    )
    assert forecast["status"] == "pass"
    assert forecast["analysis_checkpoint"]["self_sha256"] == handed["self_sha256"]
    assert forecast["final_step"] == 6 and forecast["final_time_s"] == pytest.approx(60.0)
    final = Path(forecast["checkpoints"][-1])
    assert final.name == "arwen_global_step00000006.npz"
    # The forecast's checkpoints carry the analysis lineage forward.
    chain = read_checkpoint(final)[0]["physics_metadata"][ASSIMILATION_HISTORY_KEY]
    assert len(chain["cycles"]) == 2
    # A second fresh into the same output is refused until --overwrite.
    with pytest.raises(FileExistsError, match="--overwrite"):
        da_door.fresh(
            CONFIG, out, stream_specs=[_stream_spec(obs)], start_utc=START_TEXT,
            until_utc=until.isoformat(), forecast_hours=20.0 / 3600.0, interval_s=20.0,
            options=OPTIONS, now=START + dt.timedelta(hours=1), filter_name="successive-correction",
        )


def test_fresh_keeps_each_hours_background_beside_its_analysis_when_asked(spun_up, tmp_path):
    """``keep_backgrounds`` on the one command writes every cycle's
    background (the free forecast to the instant) beside the analysis
    handed back, as ``cycle`` does, so the increment can be read from the
    two files; the receipt's analyses say the background was written and
    what is analysed does not change (the same handed-back digest as a
    fresh without it)."""
    cfg, checkpoint = spun_up
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    until = START + dt.timedelta(seconds=40)
    common = dict(
        stream_specs=[_stream_spec(obs)], start_utc=START_TEXT, until_utc=until.isoformat(),
        forecast_hours=20.0 / 3600.0, interval_s=20.0, options=OPTIONS,
        now=START + dt.timedelta(hours=1), filter_name="successive-correction",
    )
    kept = da_door.fresh(CONFIG, tmp_path / "kept", keep_backgrounds=True, **common)
    plain = da_door.fresh(CONFIG, tmp_path / "plain", **common)
    assert kept["cycle"]["cycles"]["applied"] == 2 and plain["cycle"]["cycles"]["applied"] == 2
    for step in (2, 4):
        background = tmp_path / "kept" / f"arwen_global_step{step:08d}.npz"
        analysis = tmp_path / "kept" / f"arwen_global_analysis_step{step:08d}.npz"
        assert background.is_file() and analysis.is_file()
        assert not (tmp_path / "plain" / f"arwen_global_step{step:08d}.npz").exists()
        for out, written in (("kept", True), ("plain", False)):
            report = json.loads((tmp_path / out / f"assimilation-report-step{step:08d}.json").read_text())
            assert report["background"]["written"] is written
            assert (report["background"]["path"] is not None) is written
        assert read_checkpoint(background)[0]["self_sha256"] == json.loads(
            (tmp_path / "kept" / f"assimilation-report-step{step:08d}.json").read_text())["background"]["self_sha256"]
    assert kept["analysis_checkpoint"]["self_sha256"] == plain["analysis_checkpoint"]["self_sha256"]
    # The flag is on the one command's own parser.
    from woof.globe.cli import build_parser
    args = build_parser().parse_args(["da", "fresh", str(CONFIG), "--outdir", str(tmp_path / "x"), "--keep-backgrounds"])
    assert args.keep_backgrounds is True


def test_the_cli_legs_run_the_door_and_print_the_summary(spun_up, tmp_path, capsys):
    cfg, checkpoint = spun_up
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    out = tmp_path / "cli"
    code = cli_main([
        "da", "cycle", CONFIG, "--stream", _stream_spec(obs), "--outdir", str(out),
        "--cycles", "1", "--interval-s", "20", "--start-utc", START_TEXT,
        "--length-scale-km", "4000",
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "DA scorecard" in printed and "of real time" in printed
    assert "scorecard: 1 of 1 cycles engineering-complete" in printed
    assert "wall budget" in printed
    summary = json.loads(printed[printed.index("{"):printed.index("}\n") + 1])
    assert summary["status"] == "pass"
    assert summary["analysis_checkpoint"] == str(out / "arwen_global_analysis_step00000002.npz")
    assert (out / da_door.DA_RECEIPT_NAME).is_file()

    code = cli_main(["da", "init", CONFIG, "--outdir", str(tmp_path / "cli-init"),
                     "--analysis-time", START_TEXT])
    assert code == 0
    printed = capsys.readouterr().out
    assert json.loads(printed[printed.index("{"):])["door"] == "woof global da init"
    assert (tmp_path / "cli-init" / ENSEMBLE_MANIFEST_NAME).is_file()

    fresh_out = tmp_path / "cli-fresh"
    code = cli_main([
        "da", "fresh", CONFIG, "--stream", _stream_spec(obs), "--outdir", str(fresh_out),
        "--filter", "successive-correction", "--start-utc", START_TEXT,
        "--until-utc", (START + dt.timedelta(seconds=20)).isoformat(),
        "--forecast-hours", str(20.0 / 3600.0), "--interval-s", "20",
        "--length-scale-km", "4000",
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "forecast_command" in printed
    fresh_receipt = da_door.read_da_receipt(fresh_out / da_door.DA_RECEIPT_NAME)
    code = cli_main([
        "da", "forecast", str(fresh_out / da_door.FRESH_CONFIG_NAME),
        "--analysis", fresh_receipt["analysis_checkpoint"]["path"],
        "--outdir", str(fresh_out / "forecast"),
    ])
    assert code == 0
    assert (fresh_out / "forecast" / "arwen-global-receipt.json").is_file()
    # A refusal on the CLI arrives as one sentence, at EXIT_REFUSED.
    code = cli_main(["da", "init", CONFIG, "--outdir", str(tmp_path / "cli-init")])
    assert code == EXIT_REFUSED
    assert "--overwrite" in capsys.readouterr().err


def test_da_progress_prints_the_engine_fetch_sentence_instead_of_dying(capsys):
    """The DA progress callback is handed a STRING by the engine's fetch door.

    `da fresh` without `--analysis-grib` brings the analysis down through
    `woof.fetch`, and that door reports its stages as plain sentences on the
    same callback the model integrator uses for step diagnostics.  The
    callback used to subscript both, so the FIRST fetch sentence raised
    `TypeError: string indices must be integers` and the door printed it as
    its refusal: no command, no stage, no remedy, before a single byte of the
    analysis had been read.
    """

    progress, _ = _da_progress("woof global da fresh")
    progress("fetch gdas: --all-levels declares the whole published ladder")
    progress({
        "step": 12, "time_s": 3600.0, "maximum_wind_m_s": 91.25,
        "global_mean_total_water_kg_m2": 710.659958,
    })
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        "woof global da fresh: fetch gdas: --all-levels declares the whole "
        "published ladder")
    assert lines[1].startswith("step=12 time=3600s max_wind=91.250m/s")
def test_the_bare_fresh_command_runs_the_completed_system():
    """`woof global da fresh BASE.toml --outdir DIR` with nothing else named
    runs the completed system (2026-09-06): the ensemble filter, 32
    members, 600 s bins and the shipped stream roster with the radiance
    streams; every part is replaced by naming it, and naming any stream
    or table replaces the roster (a case day on its tables fetches
    nothing).  The bin follows the steps: 600 s where the control's and
    the members' steps divide it, the next multiple that does otherwise."""
    from woof.globe.da_door import (
        DEFAULT_FRESH_POINT_STREAMS, DEFAULT_FRESH_STREAMS, default_observation_bin_s,
        resolve_fresh_defaults,
    )

    bare = resolve_fresh_defaults(filter_name=None, members=None, observation_bin_s=None,
                                  stream_specs=None, obs_locations=None)
    assert bare["filter"] == "letkf" and bare["members"] == 32
    assert bare["observation_bin_s"] == 600.0
    from woof.globe.da_door import DEFAULT_FRESH_HYBRID_BETA
    from woof.globe.da.options import DEFAULT_HYBRID_BETA
    assert DEFAULT_FRESH_HYBRID_BETA == 0.75 and DEFAULT_HYBRID_BETA == 1.0
    assert tuple(bare["stream_specs"]) == DEFAULT_FRESH_STREAMS
    assert "atms" in bare["stream_specs"] and "goes-abi" in bare["stream_specs"]
    assert bare["defaulted"] == ["filter", "members", "observation_bin_s", "streams"]
    # Naming replaces: the filter, the count, the bin, the roster.
    named = resolve_fresh_defaults(filter_name="successive-correction", members=None, observation_bin_s=None,
                                   stream_specs=None, obs_locations=None)
    assert named["filter"] == "successive-correction" and named["members"] == 1
    assert named["observation_bin_s"] is None
    assert tuple(named["stream_specs"]) == DEFAULT_FRESH_POINT_STREAMS
    tables = resolve_fresh_defaults(filter_name=None, members=16, observation_bin_s=300.0,
                                    stream_specs=["local-tables:paths=a.csv"], obs_locations=None)
    assert tables["members"] == 16 and tables["observation_bin_s"] == 300.0
    assert tables["stream_specs"] == ["local-tables:paths=a.csv"] and tables["defaulted"] == ["filter"]
    given_obs = resolve_fresh_defaults(filter_name=None, members=None, observation_bin_s=None,
                                       stream_specs=None, obs_locations=["obs.csv"])
    assert given_obs["stream_specs"] == [] and "streams" not in given_obs["defaulted"]
    # The bin and the steps: the semi-Lagrangian shape, the Eulerian T255 shape
    # at 90 s with T127 members at 180 s, and a step that divides 600 s.
    assert default_observation_bin_s(300.0, 600.0, 3600.0) == 600.0
    assert default_observation_bin_s(90.0, 180.0, 3600.0) == 720.0
    assert default_observation_bin_s(50.0, 100.0, 3600.0) == 600.0
    assert default_observation_bin_s(40.0, 120.0, 3600.0) == 600.0
    # On a real config the members' step comes from the re-cut (the smoke
    # config's step divides 600 s, so the bin is 600 s).
    cfg = load_config(CONFIG)
    with_cfg = resolve_fresh_defaults(filter_name=None, members=None, observation_bin_s=None,
                                      stream_specs=None, obs_locations=None, cfg=cfg,
                                      ensemble_truncation=3, interval_s=3600.0)
    assert with_cfg["observation_bin_s"] == default_observation_bin_s(cfg.dt_s, None, 3600.0)


def test_the_iau_reintegration_restarts_the_window_on_its_own_second_time_level():
    """The incremental analysis update re-integrates the window from its
    start on the same model, whose semi-Lagrangian second time level by
    then is the window's END; the door now hands the level it held at the
    window's start and the re-integration installs it before its first
    step (2026-09-07)."""
    import numpy as np

    from woof.globe.cycle import incremental_analysis_update

    calls = []

    class _Atm:
        def __init__(self, tag):
            self.tag = tag

        def fields(self):
            return [np.zeros(3) + self.tag]

        def with_fields(self, fields):
            return _Atm(float(fields[0][0]))

    class _State:
        def __init__(self, tag, step, time_s):
            self.atmosphere = _Atm(tag)
            self.surface = None
            self.physics_state = type("P", (), {"metadata": {}})()
            self.step = step
            self.time_s = time_s

    class _Backend:
        @staticmethod
        def to_numpy(a):
            return np.asarray(a)

    class _Transform:
        backend = _Backend()

        @staticmethod
        def inverse(a):
            return np.asarray(a)

    class _Model:
        semi_lagrangian = True
        transform = _Transform()

        def trajectory_state(self):
            return calls[-1][1] if calls else None

        def set_trajectory_state(self, trajectory):
            calls.append(("set", trajectory))

        def step(self, state, dt_s):
            calls.append(("step", state.step))
            return _State(state.atmosphere.tag, state.step + 1, state.time_s + dt_s), {}

        def _repair_positivity(self, state):
            return state, 0, 0, 0

        def enforce(self, state):
            return None

        def release_syntheses(self):
            return None

    import woof.globe.state as state_module
    real = state_module.ArwenGlobalState
    state_module.ArwenGlobalState = lambda atmosphere, surface, physics_state: _State(atmosphere.tag, 0, 0.0)
    try:
        cfg = type("C", (), {"dt_s": 10.0})()
        start = _State(1.0, 0, 0.0)
        background = _State(1.0, 2, 20.0)
        analysis = _State(3.0, 2, 20.0)
        level = object()
        # the fake ArwenGlobalState loses step and time, so the arrival check is bypassed by giving 2 steps of 0
        try:
            incremental_analysis_update(_Model(), cfg, start, background, analysis, 2, window_start_trajectory=level)
        except ValueError:
            pass  # the fake state's arrival step is not tracked; the calls are what this test reads
    finally:
        state_module.ArwenGlobalState = real
    assert calls and calls[0] == ("set", level), "the window's own level is installed before the first re-integrated step"
    assert calls[1][0] == "step"


def test_localisation_takes_the_directory_da_init_was_given(spun_up, tmp_path):
    """`da localisation --ensemble` accepts the spelling the sibling door
    hands the user.

    `da init --outdir DIR` writes the door manifest in DIR and the member
    checkpoints, with the library manifest beside them, in DIR/ensemble.
    The derivation reads the library manifest, so passing DIR made it look
    for a filename nothing had written and die on an errno naming a path
    the caller never typed, while the same command with DIR/ensemble
    worked: a door leg reachable only by knowing an internal layout.  This
    drives all four spellings a user can hold and asserts one derivation
    comes back from every one of them, and that a directory which is
    neither is refused by name rather than by errno.
    """
    cfg, _ = spun_up
    out = tmp_path / "loc"
    da_door.init(
        cfg, out, filter_name="letkf", members=3, ensemble_truncation=3,
        analysis_time_utc=START_TEXT, options=OPTIONS,
    )

    spellings = [
        out,                                            # what `da init --outdir` was given
        out / ENSEMBLE_MANIFEST_NAME,                   # the door manifest itself
        out / "ensemble",                               # the library store
        out / "ensemble" / "arwen-global-ensemble.json",  # the library manifest
    ]
    receipts = []
    for i, spelling in enumerate(spellings):
        receipt = da_door.localisation(cfg, spelling, tmp_path / f"loc-{i}.json")
        assert receipt["status"] == "pass"
        assert receipt["members"] == 3
        receipts.append(receipt)
    assert all(r["cutoffs"] == receipts[0]["cutoffs"] for r in receipts), (
        "the four spellings name one store, so they must derive one answer")

    empty = tmp_path / "not-an-ensemble"
    empty.mkdir()
    with pytest.raises(ValueError) as refusal:
        da_door.localisation(cfg, empty, tmp_path / "never.json")
    assert "arwen-global-ensemble.json" in str(refusal.value)
    assert ENSEMBLE_MANIFEST_NAME in str(refusal.value)
