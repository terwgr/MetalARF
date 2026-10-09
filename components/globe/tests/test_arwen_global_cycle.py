"""`woof global cycle`: forecast and assimilation in one process.

The door is held to the per-segment chain it replaces (``run --until-s``,
``assimilate``, ``run --restart``) byte for byte on the smoke
configuration: the analysis checkpoint it writes and the forecast
checkpoint after it carry the same arrays, trackers and chain as the
chain's.  A failed gate of record carries the background and says so;
every refusal names its breakage; the receipt measures where the wall
went.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from woof.globe.assimilate import (
    ASSIMILATION_HISTORY_KEY,
    AssimilationOptions,
    assimilate,
)
from woof.globe.checkpoint import read_checkpoint
from woof.globe.cli import main as cli_main
from woof.globe.config import load_config
from woof.globe.cycle import (
    GATE_FAILURE_CARRIES_BACKGROUND,
    PARTIAL_ANALYSIS_RULE,
    analysis_steps,
    cycle,
    resolve_start_time,
)
from woof.globe.runner import run

from test_arwen_global_assimilate import (  # noqa: F401 - the fixture rides the import
    CONFIG, _pure_noise_metars, _synthetic_obs_files, spun_up,
)

#: The synthetic reports are stamped 2026-08-31T12:00:00Z; the smoke
#: config's first checkpoint is 20 s in, so model time zero is 20 s before.
OBS_TIME = dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc)
START = OBS_TIME - dt.timedelta(seconds=20.0)
START_TEXT = "2026-08-31T11:59:40Z"
OPTIONS = AssimilationOptions(length_scale_km=4000.0)


def _arrays(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    metadata, arrays = read_checkpoint(path)
    return metadata, arrays


def _assert_same_arrays(left: Path, right: Path) -> None:
    lm, la = _arrays(left)
    rm, ra = _arrays(right)
    assert set(la) == set(ra)
    for name in la:
        assert la[name].dtype == ra[name].dtype, name
        assert np.array_equal(la[name], ra[name]), name
    assert lm["step"] == rm["step"] and lm["time_s"] == rm["time_s"]
    assert lm["run_trackers"] == rm["run_trackers"]
    assert lm["physics_metadata"] == rm["physics_metadata"]
    assert lm["arrays"] == rm["arrays"]


def _chain(cfg, obs, root: Path) -> tuple[Path, Path, dict]:
    """The per-segment chain: one 20 s segment, the file door's analysis
    of its checkpoint, one restarted segment to the end."""
    seg0 = root / "seg0"
    run(cfg, seg0, until_s=20.0)
    background = seg0 / "arwen_global_step00000002.npz"
    anl = root / "anl"
    report = assimilate(
        cfg, background, [str(p) for p in obs], anl,
        analysis_time=OBS_TIME, options=OPTIONS,
    )
    assert report["status"] == "pass"
    seg1 = root / "seg1"
    result = run(cfg, seg1, restart=Path(report["analysis"]["path"]))
    assert result["status"] == "pass"
    return (
        Path(report["analysis"]["path"]),
        seg1 / "arwen_global_step00000004.npz",
        report,
    )


def test_the_door_matches_the_per_segment_chain_bit_for_bit(spun_up, tmp_path):
    cfg, checkpoint = spun_up
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    chain_analysis, chain_forecast, chain_report = _chain(cfg, obs, tmp_path / "chain")

    out = tmp_path / "door"
    # The per-segment chain inserts its increment at the analysis instant,
    # so the door is asked for direct insertion by name (its default is the
    # incremental analysis update, which re-integrates the window).
    receipt = cycle(
        cfg, out, obs_locations=[str(p) for p in obs], cycles=1,
        start_utc=START_TEXT, interval_s=20.0, options=OPTIONS, increment_application="direct",
    )
    assert receipt["status"] == "pass"
    record = receipt["cycle"]
    assert record["completed"] == 1 and record["applied"] == 1 and record["carried"] == 0
    assert record["planned_analysis_steps"] == [2]
    analysis = record["analyses"][0]
    assert analysis["step"] == 2 and analysis["applied"] is True
    assert analysis["analysis_time_utc"] == OBS_TIME.isoformat(timespec="seconds")

    # The analysis checkpoint: the same arrays, trackers and chain.  The
    # chain's background identity is the file the chain wrote; the door
    # named its resident background by the identity that file carries.
    door_analysis = out / "arwen_global_analysis_step00000002.npz"
    _assert_same_arrays(chain_analysis, door_analysis)
    background_sha = read_checkpoint(chain_analysis)[0]["physics_metadata"][
        ASSIMILATION_HISTORY_KEY]["cycles"][0]["background_self_sha256"]
    assert analysis["background_self_sha256"] == background_sha
    assert analysis["background_written"] is False
    assert not (out / "arwen_global_step00000002.npz").exists()

    # The forecast from the analysis: the same bytes at the end, so the
    # new conservation epoch was opened exactly as the restart opens it.
    door_forecast = out / "arwen_global_step00000004.npz"
    _assert_same_arrays(chain_forecast, door_forecast)
    assert receipt["cycle"]["epochs"][0]["step"] == 2
    assert receipt["run_trackers"] == read_checkpoint(chain_forecast)[0]["run_trackers"]

    # The report beside the analysis is the file door's report with the
    # same verdicts, plus the phase timings.
    report = json.loads((out / "assimilation-report-step00000002.json").read_text())
    assert report["status"] == "pass"
    for variable, row in chain_report["variables"].items():
        assert report["variables"][variable]["withheld"]["o_minus_a"]["rms"] == pytest.approx(
            row["withheld"]["o_minus_a"]["rms"], rel=1e-12)
        assert report["variables"][variable]["o_minus_a"]["rms"] == pytest.approx(
            row["o_minus_a"]["rms"], rel=1e-12)
    assert report["assimilated_report_ids"] == chain_report["assimilated_report_ids"]
    assert report["analysis"]["path"] == str(door_analysis)
    assert report["analysis"]["self_sha256"] == read_checkpoint(door_analysis)[0]["self_sha256"]
    assert report["background"]["path"] is None
    phases = report["timings_s"]["analysis_phases"]
    for key in ("quality_control_s", "background_operators_s", "spread_s",
                "moisture_s", "mass_and_positivity_s", "analysis_operators_s"):
        assert key in phases and phases[key] >= 0.0
    assert analysis["timings_s"]["analysis_s"] == pytest.approx(sum(phases.values()))
    assert analysis["forecast_steps_wall_s"] > 0.0
    assert record["wall_seconds_per_model_hour_cycled"] > 0.0
    assert set(map(Path, receipt["checkpoints"])) == {
        out / "arwen_global_step00000000.npz", door_analysis, door_forecast,
    }


def test_a_failed_gate_carries_the_background_and_says_so(spun_up, tmp_path):
    """Reports made of noise: the gate of record fails and the whole hour
    is carried.  The partial rule is off here so the carry path itself is
    what is judged (on, a noise variable that a six-row withheld set lets
    through by chance would be applied; the rule's own test is below)."""
    cfg, checkpoint = spun_up
    noise = _pure_noise_metars(cfg, checkpoint, tmp_path / "noise.csv", 10.0)
    out = tmp_path / "door"
    receipt = cycle(
        cfg, out, obs_locations=[str(noise)], cycles=1,
        start_utc=START_TEXT, interval_s=20.0, options=OPTIONS,
        partial_analyses=False,
    )
    record = receipt["cycle"]
    assert record["carried"] == 1 and record["applied"] == 0
    analysis = record["analyses"][0]
    assert analysis["applied"] is False and analysis["status"] == "fail"
    assert analysis["failed_variables"]
    assert analysis["analysis"] is None
    assert not (out / "arwen_global_analysis_step00000002.npz").exists()
    # The background is this hour's checkpoint, unchanged from the
    # forecast the chain would have written.
    background = out / "arwen_global_step00000002.npz"
    assert background.exists()
    reference = tmp_path / "reference"
    run(cfg, reference, until_s=20.0)
    _assert_same_arrays(reference / "arwen_global_step00000002.npz", background)
    report = json.loads((out / "assimilation-report-step00000002.json").read_text())
    assert report["carried"]["reason"] == GATE_FAILURE_CARRIES_BACKGROUND
    assert report["carried"]["background_path"] == str(background)
    assert report["analysis"] is None
    # No epoch opened: the forecast carried on under the cold start's targets.
    assert record["epochs"] == []
    assert receipt["mass_target_pa"] == receipt["cold_start_diagnostics"]["global_mean_surface_pressure_pa"]


def test_keep_backgrounds_writes_the_background_beside_the_analysis(spun_up, tmp_path):
    cfg, checkpoint = spun_up
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    out = tmp_path / "door"
    receipt = cycle(
        cfg, out, obs_locations=[str(p) for p in obs], cycles=1,
        start_utc=START_TEXT, interval_s=20.0, options=OPTIONS,
        keep_backgrounds=True,
    )
    analysis = receipt["cycle"]["analyses"][0]
    background = out / "arwen_global_step00000002.npz"
    assert analysis["background_written"] is True and background.exists()
    assert read_checkpoint(background)[0]["self_sha256"] == analysis["background_self_sha256"]
    report = json.loads((out / "assimilation-report-step00000002.json").read_text())
    assert report["background"]["path"] == str(background)


def test_refusals_name_their_breakage(spun_up, tmp_path):
    cfg, checkpoint = spun_up
    obs = [str(p) for p in _synthetic_obs_files(cfg, checkpoint, tmp_path)]
    with pytest.raises(ValueError, match="whole number of 10 s steps"):
        cycle(cfg, tmp_path / "a", obs_locations=obs, cycles=1,
              start_utc=START_TEXT, interval_s=25.0)
    with pytest.raises(ValueError, match="3 cycles at --interval-s 20 .* need model time beyond 40"):
        cycle(cfg, tmp_path / "b", obs_locations=obs, cycles=3,
              start_utc=START_TEXT, interval_s=20.0)
    with pytest.raises(ValueError, match="pass --start-utc"):
        cycle(cfg, tmp_path / "c", obs_locations=obs, cycles=1, interval_s=20.0)
    with pytest.raises(ValueError, match="beyond the config's duration_s"):
        cycle(cfg, tmp_path / "d", obs_locations=obs, cycles=1,
              start_utc=START_TEXT, interval_s=20.0, until_s=60.0)
    with pytest.raises(ValueError, match="at least one --obs source"):
        cycle(cfg, tmp_path / "e", obs_locations=[], cycles=1,
              start_utc=START_TEXT, interval_s=20.0)
    # The same reports offered at the second analysis are all in the
    # chain (no withheld tenth here: the gate minimum is above the count,
    # so every row was analysed): nothing new to analyse is a defect of
    # the offer, not weather, and the door refuses with a failure receipt
    # rather than carrying on.
    out = tmp_path / "f"
    everything = AssimilationOptions(length_scale_km=4000.0, gate_minimum_count=10**6)
    with pytest.raises(ValueError, match="nothing new to analyse"):
        cycle(cfg, out, obs_locations=obs, cycles=2,
              start_utc=START_TEXT, interval_s=20.0, options=everything)
    failure = json.loads((out / "arwen-global-receipt.json").read_text())
    assert failure["status"] == "error"
    assert failure["cycle"]["completed"] == 1
    assert failure["completed_step"] == 4


def test_a_second_cycle_analyses_only_what_the_chain_lacks(spun_up, tmp_path):
    """The withheld tenth of the first cycle was never in the state, so
    the second cycle may analyse it; everything else is refused from the
    chain and the report counts the refusal."""
    cfg, checkpoint = spun_up
    obs = [str(p) for p in _synthetic_obs_files(cfg, checkpoint, tmp_path)]
    out = tmp_path / "two"
    receipt = cycle(cfg, out, obs_locations=obs, cycles=2,
                    start_utc=START_TEXT, interval_s=20.0, options=OPTIONS)
    first, second = receipt["cycle"]["analyses"]
    assert first["refused_from_chain"] == 0
    assert second["refused_from_chain"] == first["assimilated_total"]
    assert second["assimilated_total"] + second["withheld_total"] == first["withheld_total"]
    assert receipt["cycle"]["completed"] == 2


def test_analysis_steps_fall_on_whole_intervals_after_the_start():
    cfg = load_config(CONFIG)
    assert analysis_steps(cfg, interval_s=20.0, cycles=2, start_step=0, total_steps=4) == (2, [2, 4])
    assert analysis_steps(cfg, interval_s=20.0, cycles=1, start_step=2, total_steps=4) == (2, [4])
    assert analysis_steps(cfg, interval_s=10.0, cycles=2, start_step=1, total_steps=4) == (1, [2, 3])
    with pytest.raises(ValueError, match="positive whole number"):
        analysis_steps(cfg, interval_s=20.0, cycles=0, start_step=0, total_steps=4)


def test_the_start_instant_comes_from_the_flag_or_the_physics_options():
    cfg = load_config(CONFIG)
    assert resolve_start_time(cfg, START_TEXT) == START
    assert resolve_start_time(cfg, START) == START
    naive = START.replace(tzinfo=None)
    assert resolve_start_time(cfg, naive) == START
    with pytest.raises(ValueError, match="not an ISO-8601 instant"):
        resolve_start_time(cfg, "yesterday")
    with pytest.raises(ValueError, match="start_time_utc"):
        resolve_start_time(cfg, None)


def test_the_cli_leg_runs_the_door_and_prints_its_summary(spun_up, tmp_path, capsys):
    cfg, checkpoint = spun_up
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    out = tmp_path / "cli"
    code = cli_main([
        "cycle", CONFIG, "--obs", str(obs[0]), "--obs", str(obs[1]),
        "--outdir", str(out), "--cycles", "1", "--interval-s", "20",
        "--start-utc", START_TEXT, "--length-scale-km", "4000",
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "analysis step=2 at 2026-08-31T12:00:00+00:00: applied" in printed
    summary = json.loads(printed[printed.index("{"):])
    assert summary["status"] == "pass"
    assert summary["cycles_completed"] == 1 and summary["cycles_applied"] == 1
    assert summary["analyses"][0]["analysis"] == str(out / "arwen_global_analysis_step00000002.npz")
    assert (out / "arwen-global-receipt.json").exists()
    # A second run into the same directory is refused until --overwrite,
    # as one sentence on stderr under the door's refusal contract.
    code = cli_main([
        "cycle", CONFIG, "--obs", str(obs[0]), "--outdir", str(out),
        "--cycles", "1", "--interval-s", "20", "--start-utc", START_TEXT,
    ])
    assert code != 0
    assert "--overwrite" in capsys.readouterr().err


def _temperature_noise_metars(cfg, checkpoint: Path, path: Path) -> Path:
    """The end-to-end fixture's METARs with the temperature column replaced
    by the background plus white noise at ten times the table error, so
    temperature alone has nothing real to fit while pressure and wind
    carry the smooth signal."""
    metar, _aircraft = _synthetic_obs_files(cfg, checkpoint, path.parent)
    lines = metar.read_text(encoding="utf-8").splitlines()
    rng = np.random.default_rng(5)
    out = [lines[0]]
    for line in lines[1:]:
        fields = line.split(",")
        # RAW,station,time,lat,lon,temp_c,dir,speed,altim,elev
        fields[5] = f"{float(fields[5]) + rng.normal(0.0, 15.0):.4f}"
        out.append(",".join(fields))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def test_a_variable_that_fails_the_gate_is_withdrawn_and_the_rest_applied(spun_up, tmp_path):
    """PARTIAL_ANALYSIS_RULE: temperature made of noise fails its withheld
    gate; pressure and wind pass; the door withdraws the temperature
    reports, analyses the hour again with the rest, applies that analysis
    and carries temperature as the background.  With the rule off the
    whole hour is carried as the chain did."""
    cfg, checkpoint = spun_up
    noisy = _temperature_noise_metars(cfg, checkpoint, tmp_path / "noisy_t.csv")
    out = tmp_path / "partial"
    receipt = cycle(
        cfg, out, obs_locations=[str(noisy)], cycles=1,
        start_utc=START_TEXT, interval_s=20.0, options=OPTIONS,
    )
    record = receipt["cycle"]
    analysis = record["analyses"][0]
    assert analysis["dropped_variables"] == ["temperature_k"]
    assert analysis["applied"] is True and record["partial"] == 1 and record["carried"] == 0
    report = json.loads((out / "assimilation-report-step00000002.json").read_text())
    assert report["partial"]["dropped_variables"] == ["temperature_k"]
    assert report["partial"]["rule"] == PARTIAL_ANALYSIS_RULE
    first = report["partial"]["first_pass"]
    assert first["failed_variables"] == ["temperature_k"]
    t_first = first["variables"]["temperature_k"]
    assert t_first["withheld_o_minus_a_rms"] >= t_first["withheld_o_minus_b_rms"]
    # The second pass never saw a temperature report: none in the chain,
    # no temperature increment, and the other variables still pass.
    assert "temperature_k" not in report["variables"]
    assert "temperature_k" not in report["increment_maxabs"]
    assert report["status"] == "pass"
    chain = read_checkpoint(out / "arwen_global_analysis_step00000002.npz")[0][
        "physics_metadata"][ASSIMILATION_HISTORY_KEY]
    assert set(chain["reports"]) == set(
        h for name, ids in report["assimilated_report_ids"].items() for h in ids
    )
    # Both passes are timed into the record.
    assert analysis["timings_s"]["analysis_s"] == pytest.approx(
        sum(report["timings_s"]["analysis_phases"].values()))

    # The rule off: the hour is carried whole.
    out_off = tmp_path / "whole"
    receipt_off = cycle(
        cfg, out_off, obs_locations=[str(noisy)], cycles=1,
        start_utc=START_TEXT, interval_s=20.0, options=OPTIONS,
        partial_analyses=False,
    )
    off = receipt_off["cycle"]["analyses"][0]
    assert off["applied"] is False and off["dropped_variables"] == []
    assert off["failed_variables"] == ["temperature_k"]
    assert receipt_off["cycle"]["partial_analyses"] is False


def test_the_cycle_door_records_the_band_schedule_it_ran(spun_up, tmp_path):
    """The forecast door records the band count and the cycle door did not.

    A door that prices a band count in its verdict and writes a receipt
    that names none is a run nobody can check afterwards: the gate reads
    the card before a byte is allocated and the builder reads it again
    after the Legendre tables are on it, so the two can disagree and
    nothing in the output says they did.  The smoke grid is six latitude
    rows, below the four-row band floor at any count above one, so what
    is held here is the record and who it says chose the count.
    """
    from dataclasses import replace

    cfg, checkpoint = spun_up
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)

    declared = cycle(
        replace(cfg, latitude_bands=1), tmp_path / "declared",
        obs_locations=[str(p) for p in obs], cycles=1,
        start_utc=START_TEXT, interval_s=20.0, options=OPTIONS,
    )
    assert declared["status"] == "pass"
    record = declared["latitude_bands"]
    assert record["latitude_bands"] == 1
    assert record["latitude_rows"] >= 1
    assert record["resident"] is True
    assert record["chosen_by"] == "config"

    sized = cycle(
        cfg, tmp_path / "sized",
        obs_locations=[str(p) for p in obs], cycles=1,
        start_utc=START_TEXT, interval_s=20.0, options=OPTIONS,
    )
    assert sized["status"] == "pass"
    assert sized["latitude_bands"]["latitude_bands"] == 1
    assert sized["latitude_bands"]["chosen_by"] == "sizer"
    # The band count is a streaming granularity here too: one config
    # hash, one lineage, the same analysis and the same forecast.
    assert declared["config_hash"] == sized["config_hash"]
    for name in ("arwen_global_analysis_step00000002.npz",
                 "arwen_global_step00000004.npz"):
        _assert_same_arrays(tmp_path / "sized" / name,
                            tmp_path / "declared" / name)


def test_the_cycle_door_takes_the_memory_levers(tmp_path):
    """`--latitude-bands` reaches the cycle door, not only `run`."""
    from woof.globe.cli import _memory_lever_overrides, build_parser

    parser = build_parser()
    args = parser.parse_args([
        "cycle", str(CONFIG), "--outdir", str(tmp_path / "out"),
        "--cycles", "1", "--obs", str(tmp_path / "nothing.json"),
        "--latitude-bands", "4",
    ])
    assert _memory_lever_overrides(args) == {"latitude_bands": 4}
