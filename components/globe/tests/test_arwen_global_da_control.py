"""The design amendments of 2026-09-06 in the DA door, each with its
numerical test on the CPU at the smoke truncation: the control's own
analysis and the transfer layer (A, C), observations at their own times
(B), the incremental analysis update and the imbalance reading (D), the
external analysis as a weak low-pass constraint (E), the causal
bookkeeping (F).
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pytest

from woof.globe import da_door
from woof.globe.assimilate import ASSIMILATION_HISTORY_KEY
from woof.globe.checkpoint import read_checkpoint, state_from_checkpoint
from woof.globe.config import load_config
from woof.globe.constants import SPECTRAL_FIELDS
from woof.globe.da.analysis import apply_taper, taper_weights
from woof.globe.da.ensemble import embed_spectral, truncate_spectral
from woof.globe.da.options import FilterOptions
from woof.globe.da_anchor import (
    AnchorOptions,
    ExternalAnchor,
    anchor_options_from_spec,
    apply_anchor,
    load_anchor,
    parse_anchor_spec,
)
from woof.globe.da_control import ControlOptions, degree_power
from woof.globe.da_filter import ENSEMBLE_MANIFEST_NAME, read_ensemble_manifest
from woof.globe.da_streams import classify_latency
from woof.globe.da_window import (
    LINEARISED_LABEL,
    binning_sensitivity,
    linearised_analysis_equivalent,
    rows_in_window,
)
from woof.globe.obs_table import ObsRow
from woof.globe.runner import build_model_and_cold_state, build_transform

from test_arwen_global_assimilate import (  # noqa: F401 - the fixture rides the import
    CONFIG, _synthetic_obs_files, spun_up,
)
from test_arwen_global_cycle import OBS_TIME, START, START_TEXT, OPTIONS


# ---------------------------------------------------------------------------
# C: the transfer layer (the package's taper, driven through the door's options)
# ---------------------------------------------------------------------------

def test_the_door_options_lay_the_taper_and_the_recentring_over_the_package_options():
    base = FilterOptions()
    assert ControlOptions().taper_degrees_for(127) == base.taper_degrees(127) == (76, 127)
    laid = ControlOptions(taper_full_degree=40, taper_zero_degree=100, recentre_fraction=0.5).filter_options(base)
    assert laid.taper_degrees(127) == (40, 100) and laid.recentering_fraction == 0.5
    assert laid.increment_application == "iau"
    w = taper_weights(12, 4, 9)
    assert w[:5].tolist() == [1.0] * 5 and w[9:].tolist() == [0.0] * 4
    assert w[6] == pytest.approx(math.cos(0.5 * math.pi * 2.0 / 5.0) ** 2)
    rng = np.random.default_rng(1)
    coeff = (rng.standard_normal((3, 13, 13)) + 1j * rng.standard_normal((3, 13, 13))) * np.tri(13)[None]
    out = apply_taper(coeff, w, np)
    assert np.array_equal(out[:, :5, :], coeff[:, :5, :]) and not out[:, 9:, :].any()
    power = degree_power(out)
    assert power[9:].tolist() == [0.0] * 4 and power[:5] == pytest.approx(degree_power(coeff)[:5])


def test_the_transfer_is_the_identity_both_ways_in_native_variables():
    rng = np.random.default_rng(2)
    det_t, ens_t = 15, 7
    field = (rng.standard_normal((2, det_t + 1, det_t + 1)) + 1j * rng.standard_normal((2, det_t + 1, det_t + 1)))
    field *= np.tri(det_t + 1)[None]
    low = truncate_spectral(field, ens_t)
    assert low.shape == (2, ens_t + 1, ens_t + 1)
    back = embed_spectral(low, det_t)
    assert np.array_equal(back[:, : ens_t + 1, : ens_t + 1], field[:, : ens_t + 1, : ens_t + 1])
    assert not back[:, ens_t + 1:, :].any() and not back[:, :, ens_t + 1:].any()
    assert np.array_equal(truncate_spectral(embed_spectral(low, det_t), ens_t), low)
    with pytest.raises(ValueError, match="embedding, not a truncation"):
        truncate_spectral(low, det_t)
    with pytest.raises(ValueError, match="truncation, not an embedding"):
        embed_spectral(field, ens_t)


def test_control_options_refuse_what_the_release_does_not_carry():
    with pytest.raises(ValueError, match="names no table"):
        ControlOptions(hybrid_beta=0.5, static_covariance=None)
    assert ControlOptions(hybrid_beta=0.5).filter_options(FilterOptions()).hybrid_beta == 0.5
    with pytest.raises(ValueError, match="increment_source"):
        ControlOptions(increment_source="mean")
    with pytest.raises(ValueError, match="recentre_fraction"):
        ControlOptions(recentre_fraction=1.5)
    with pytest.raises(ValueError, match="increment_application"):
        ControlOptions(increment_application="nudge")
    assert ControlOptions(recentre_fraction=0.5, increment_source="ensemble-mean").identity()["recentre_fraction"] == 0.5


# ---------------------------------------------------------------------------
# B: the door's side of observations at their own times
# ---------------------------------------------------------------------------

def _row(minute: int, second: int, k: int = 0, variable: str = "temperature_k") -> ObsRow:
    return ObsRow(
        source="s", station_id=f"K{k}", latitude_deg=40.0, longitude_deg=-100.0 + k,
        elevation_m=0.0, level_pa=None,
        valid_time=dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc) + dt.timedelta(minutes=minute, seconds=second),
        variable=variable, value=280.0, error=1.0,
    )


def test_the_trailing_window_and_the_linearised_equivalent():
    start = dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc)
    rows = [_row(0, 0, 9), _row(0, 30), _row(7, 0, 1), _row(60, 0, 4), _row(61, 0, 5)]
    inside = rows_in_window(rows, start, 0.0, 3600.0)
    # (t0, t1]: the 12:00:00 report belongs to the previous window, 13:01 to the next.
    assert [r.station_id for r in inside] == ["K0", "K1", "K4"]
    sens = binning_sensitivity([5.0, 5.0, 5.0], [1.0, 2.0, 5.0])
    assert sens["rows"] == 3 and sens["max_abs"] == 4.0 and sens["rms"] == pytest.approx(math.sqrt(25.0 / 3.0))
    assert binning_sensitivity(None, [1.0]) is None
    assert linearised_analysis_equivalent([1.0], [5.0], [5.5]).tolist() == [1.5]
    assert "linearised" in LINEARISED_LABEL


# ---------------------------------------------------------------------------
# F: latency classes
# ---------------------------------------------------------------------------

def test_latency_classes_follow_the_measured_latency():
    assert classify_latency(None) == "unverified"
    assert classify_latency(600.0) == "fast"
    assert classify_latency(3600.0) == "fast"
    assert classify_latency(7200.0) == "replay"
    assert classify_latency(90000.0) == "retrospective"


# ---------------------------------------------------------------------------
# E: the anchor on the smoke state
# ---------------------------------------------------------------------------

def test_the_anchor_moves_the_constrained_degrees_by_the_weight_and_nothing_else(spun_up):
    cfg, checkpoint = spun_up
    transform = build_transform(cfg)
    model, _cold = build_model_and_cold_state(cfg, transform)
    metadata, arrays = read_checkpoint(checkpoint, expected_config_hash=cfg.config_hash)
    state = state_from_checkpoint(metadata, arrays, transform.backend)
    model.enforce(state)
    when = OBS_TIME
    fields = {name: np.asarray(getattr(state.atmosphere, name), dtype=np.complex128).copy() for name in SPECTRAL_FIELDS}
    same = ExternalAnchor("checkpoint", "self", when, when, fields, {})
    options = AnchorOptions(weight=0.5, full_degree=1, zero_degree=2, max_age_s=3600.0)
    # An anchor equal to the state moves nothing, bitwise.
    out, record = apply_anchor(state, model, transform, same, options, analysis_time=when)
    assert record["applied"] is True
    for name in SPECTRAL_FIELDS:
        assert np.array_equal(np.asarray(getattr(out.atmosphere, name)), np.asarray(getattr(state.atmosphere, name)))
    assert all(v == 0.0 for v in record["increment_grid_rms"].values())
    # A difference at degree 3 only (above zero_degree 2) moves nothing.
    high = {k: v.copy() for k, v in fields.items()}
    high["theta"][:, 3, :] += 0.7
    out, record = apply_anchor(state, model, transform, ExternalAnchor("checkpoint", "high", when, when, high, {}),
                               options, analysis_time=when)
    assert np.array_equal(np.asarray(out.atmosphere.theta), np.asarray(state.atmosphere.theta))
    assert record["departure_power"]["theta"]["fraction_in_band"] == 0.0
    # A difference at degree 1 (full weight) moves half the way, exactly.
    low = {k: v.copy() for k, v in fields.items()}
    low["theta"][:, 1, 0] += 0.4
    low["log_surface_pressure"][0, 0] += 0.01  # degree 0 of ln ps: excluded
    low["log_surface_pressure"][1, 1] += 0.001
    out, record = apply_anchor(state, model, transform, ExternalAnchor("checkpoint", "low", when, when, low, {}),
                               options, analysis_time=when)
    theta = np.asarray(out.atmosphere.theta)
    assert theta[:, 1, 0] == pytest.approx(fields["theta"][:, 1, 0] + 0.2)
    lnps = np.asarray(out.atmosphere.log_surface_pressure)
    assert lnps[0, 0] == pytest.approx(fields["log_surface_pressure"][0, 0])
    assert lnps[1, 1] == pytest.approx(fields["log_surface_pressure"][1, 1] + 0.0005)
    assert record["increment_grid_rms"]["theta"] > 0.0
    assert record["weights_per_degree"] == pytest.approx([1.0, 1.0, 0.0, 0.0])
    # Out of its age window: not applied, and the record says why.
    out, record = apply_anchor(state, model, transform, same, options,
                               analysis_time=when + dt.timedelta(hours=2))
    assert record["applied"] is False and "beyond max_age_s" in record["reason"]
    assert out is state
    # The spelling and the loader.
    spec = parse_anchor_spec(f"{checkpoint}:valid_utc={when.isoformat()};weight=0.25;full_degree=1;zero_degree=2")
    assert spec["path"] == str(checkpoint) and spec["weight"] == "0.25"
    loaded = load_anchor(cfg, transform, spec)
    assert loaded.kind == "checkpoint" and loaded.valid_utc == when
    assert np.array_equal(loaded.fields["theta"], fields["theta"])
    assert anchor_options_from_spec(spec).weight == 0.25
    with pytest.raises(ValueError, match="valid_utc"):
        load_anchor(cfg, transform, {"path": str(checkpoint)})
    with pytest.raises(ValueError, match="hybrid|weight"):
        AnchorOptions(weight=1.5)


# ---------------------------------------------------------------------------
# A, B, D through the door on the smoke case
# ---------------------------------------------------------------------------

def _mid_window_obs(cfg, checkpoint, directory: Path) -> list[Path]:
    """The synthetic tables with every report fifteen seconds BEFORE the
    analysis instant (11:59:45), inside the window's first ten-second bin
    (the package observes a bin at its end, and a report exactly on a bin
    boundary joins the later bin)."""
    files = _synthetic_obs_files(cfg, checkpoint, directory)
    out = []
    for path in files:
        text = path.read_text(encoding="utf-8").replace("2026-08-31T12:00:00Z", "2026-08-31T11:59:45Z")
        target = path.with_name(f"mid-{path.name}")
        target.write_text(text, encoding="utf-8")
        out.append(target)
    return out


def test_reports_inside_the_window_are_compared_at_their_own_bin(spun_up, tmp_path):
    cfg, checkpoint = spun_up
    obs = _mid_window_obs(cfg, checkpoint, tmp_path)
    out = tmp_path / "bins"
    da_door.init(cfg, out, filter_name="letkf", members=3, ensemble_truncation=3,
                 analysis_time_utc=START_TEXT, options=OPTIONS)
    receipt = da_door.cycle(
        cfg, out, stream_specs=["local-tables:paths=" + ",".join(map(str, obs))], cycles=1,
        start_utc=START_TEXT, interval_s=20.0, ensemble=out / ENSEMBLE_MANIFEST_NAME,
        options=OPTIONS, observation_bin_s=10.0,
    )
    assert receipt["status"] == "pass"
    report = json.loads((out / "assimilation-report-step00000002.json").read_text())
    times = report["observation_times"]
    assert times["bins"] == 2 and times["rows_per_bin"] == [304, 0]
    assert times["rows_observed_at_own_time"] == {"members": 304, "control": 304}
    assert times["rows_observed_at_end"] == {"members": 0, "control": 0}
    assert times["rows_at_own_time"] == 304
    assert times["o_minus_a_label"] == LINEARISED_LABEL
    # The trajectory moved between 11:59:50 (the first bin's end) and
    # 12:00:00: an analysis-instant comparison would have read that motion
    # as an innovation.
    sensitivity = times["binning_sensitivity"]
    assert set(sensitivity) >= {"awc-metar-cache/temperature_k", "awc-aircraft-cache/wind_u_m_s"}
    for entry in sensitivity.values():
        assert entry["control"]["rows"] > 0 and entry["control"]["rms"] > 0.0
        assert entry["rows_at_own_time"] > 0
    assert report["scorecard"]["o_minus_a_label"] == LINEARISED_LABEL
    card = report["scorecard"]["streams"]["awc-metar-cache"]["variables"]["temperature_k"]["regions"]["global"]
    # The reports sit fifteen seconds before the analysis instant and were
    # compared with the state five seconds after them (the first bin's
    # end); the card records the offset to the analysis instant.
    assert card["time_offset_max_abs_s"] == pytest.approx(15.0)
    assert card["consistency"]["innovation_variance_ratio"] is not None
    # A cell the observation-error table does not name is judged at the
    # door's own error, and the card says so.
    assert card["consistency"]["calibrated"] is False
    assert card["consistency"]["door_sigma_o"] == pytest.approx(card["consistency"]["assigned_sigma_o"])
    # The same reports at the analysis instant read a different O-B: the
    # binning is a measured difference, not a relabelling.
    (tmp_path / "instant").mkdir()
    instant_obs = _synthetic_obs_files(cfg, checkpoint, tmp_path / "instant")
    out2 = tmp_path / "instant-run"
    da_door.init(cfg, out2, filter_name="letkf", members=3, ensemble_truncation=3,
                 analysis_time_utc=START_TEXT, options=OPTIONS)
    da_door.cycle(
        cfg, out2, stream_specs=["local-tables:paths=" + ",".join(map(str, instant_obs))], cycles=1,
        start_utc=START_TEXT, interval_s=20.0, ensemble=out2 / ENSEMBLE_MANIFEST_NAME,
        options=OPTIONS, observation_bin_s=10.0,
    )
    other = json.loads((out2 / "assimilation-report-step00000002.json").read_text())
    other_card = other["scorecard"]["streams"]["awc-metar-cache"]["variables"]["temperature_k"]["regions"]["global"]
    assert other["observation_times"]["rows_at_own_time"] == 0
    assert other_card["o_minus_b"]["rms"] != card["o_minus_b"]["rms"]


def test_the_incremental_analysis_update_hands_back_a_balanced_state_with_the_chain(spun_up, tmp_path):
    cfg, checkpoint = spun_up
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    spec = "local-tables:paths=" + ",".join(map(str, obs))
    results = {}
    for mode in ("direct", "iau"):
        out = tmp_path / mode
        da_door.init(cfg, out, analysis_time_utc=START_TEXT)
        receipt = da_door.cycle(
            cfg, out, stream_specs=[spec], cycles=1, start_utc=START_TEXT, interval_s=20.0,
            ensemble=out / ENSEMBLE_MANIFEST_NAME, options=OPTIONS, increment_application=mode,
            until_s=40.0,
        )
        assert receipt["status"] == "pass" and receipt["settings"]["increment_application"] == mode
        report = json.loads((out / "assimilation-report-step00000002.json").read_text())
        assert report["increment_application"]["mode"] == mode
        handed = receipt["analysis_checkpoint"]
        metadata, arrays = read_checkpoint(handed["path"], expected_config_hash=cfg.config_hash)
        assert metadata["step"] == 2 and len(metadata["physics_metadata"][ASSIMILATION_HISTORY_KEY]["cycles"]) == 1
        results[mode] = (receipt, report, metadata, arrays)
    direct_receipt, direct_report, direct_meta, direct_arrays = results["direct"]
    iau_receipt, iau_report, iau_meta, iau_arrays = results["iau"]
    iau = iau_report["increment_application"]["iau"]
    assert iau["steps"] == 2 and iau["wall_s"] > 0.0
    assert iau["increment_grid_rms"]["theta"] > 0.0
    assert direct_report["increment_application"]["iau"] is None
    # The two hand back different states at the same step: the increment
    # entered through two steps of dynamics, not at once.
    assert direct_meta["self_sha256"] != iau_meta["self_sha256"]
    theta_d = direct_arrays["atmosphere/theta"] if "atmosphere/theta" in direct_arrays else None
    assert iau_receipt["budget"]["cycles"][0]["iau_reintegration_s"] > 0.0
    assert direct_receipt["budget"]["cycles"][0]["iau_reintegration_s"] == 0.0
    # The imbalance reading rides in both receipts.
    for receipt in (direct_receipt, iau_receipt):
        tendency = receipt["assessments"][0]["physical_consistency"]["surface_pressure_tendency_rms_pa_s"]
        assert tendency["first_step_after_analysis"] > 0.0 and tendency["ratio_after_over_before"] > 0.0
    del theta_d


def test_the_cli_carries_the_amendment_flags(spun_up, tmp_path, capsys):
    from woof.globe.cli import main as cli_main

    cfg, checkpoint = spun_up
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    out = tmp_path / "cli"
    code = cli_main([
        "da", "init", CONFIG, "--outdir", str(out), "--filter", "letkf", "--members", "3",
        "--ensemble-truncation", "3", "--analysis-time", START_TEXT, "--length-scale-km", "4000",
        "--recentre-fraction", "0.5", "--taper-full-degree", "1", "--taper-zero-degree", "3",
    ]) if False else cli_main([
        "da", "init", CONFIG, "--outdir", str(out), "--filter", "letkf", "--members", "3",
        "--ensemble-truncation", "3", "--analysis-time", START_TEXT,
        "--recentre-fraction", "0.5", "--taper-full-degree", "1", "--taper-zero-degree", "3",
    ])
    assert code == 0
    capsys.readouterr()
    code = cli_main([
        "da", "cycle", CONFIG, "--stream", "local-tables:paths=" + ",".join(map(str, obs)),
        "--outdir", str(out), "--cycles", "1", "--interval-s", "20", "--start-utc", START_TEXT,
        "--ensemble", str(out / ENSEMBLE_MANIFEST_NAME), "--length-scale-km", "4000",
        "--observation-bin-s", "10", "--increment-application", "iau",
        "--recentre-fraction", "0.5", "--taper-full-degree", "1", "--taper-zero-degree", "3",
        "--additive-inflation", "0.02", "--until-s", "40",
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "engineering-complete" in printed and "information:" in printed
    receipt = da_door.read_da_receipt(out / da_door.DA_RECEIPT_NAME)
    assert receipt["settings"]["observation_bin_s"] == 10.0
    assert receipt["settings"]["increment_application"] == "iau"
    assert receipt["settings"]["control_options"]["recentre_fraction"] == 0.5
    assert receipt["settings"]["additive_inflation_fraction"] == 0.02
    report = json.loads((out / "assimilation-report-step00000002.json").read_text())
    assert report["inflation"]["additive_fraction"] == 0.02 and report["inflation"]["applied"] is True
    assert report["recentre"]["fraction"] == 0.5
    assert report["increment_application"]["mode"] == "iau"


def test_the_anchor_rides_through_the_door_and_is_recorded(spun_up, tmp_path):
    cfg, checkpoint = spun_up
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    spec = "local-tables:paths=" + ",".join(map(str, obs))
    out = tmp_path / "anchored"
    da_door.init(cfg, out, filter_name="letkf", members=3, ensemble_truncation=3,
                 analysis_time_utc=START_TEXT, options=OPTIONS)
    anchor = f"{checkpoint}:valid_utc={OBS_TIME.isoformat()};weight=0.5;full_degree=1;zero_degree=2"
    receipt = da_door.cycle(
        cfg, out, stream_specs=[spec], cycles=1, start_utc=START_TEXT, interval_s=20.0,
        ensemble=out / ENSEMBLE_MANIFEST_NAME, options=OPTIONS, anchor_spec=anchor,
    )
    assert receipt["status"] == "pass" and receipt["settings"]["anchor"] == anchor
    report = json.loads((out / "assimilation-report-step00000002.json").read_text())
    record = report["anchor"]
    assert record["applied"] is True and record["age_s"] == 0.0
    assert record["source"]["kind"] == "checkpoint" and record["affected_band"]["zero_degree"] == 2
    assert record["increment_grid_rms"]["theta"] > 0.0
    assert "does not make the cycle unable to drift" in record["claim"]
    assert report["assessments"]["physical_consistency"]["anchor"]["applied"] is True
    # The successive-correction filter takes the same anchor.
    out2 = tmp_path / "anchored-sc"
    da_door.init(cfg, out2, analysis_time_utc=START_TEXT)
    receipt2 = da_door.cycle(
        cfg, out2, stream_specs=[spec], cycles=1, start_utc=START_TEXT, interval_s=20.0,
        ensemble=out2 / ENSEMBLE_MANIFEST_NAME, options=OPTIONS, anchor_spec=anchor,
    )
    report2 = json.loads((out2 / "assimilation-report-step00000002.json").read_text())
    assert receipt2["status"] == "pass" and report2["anchor"]["applied"] is True
    manifest = read_ensemble_manifest(out2 / ENSEMBLE_MANIFEST_NAME)
    assert manifest["filter"] == "successive-correction"


def test_partial_recentring_and_the_comparison_arm_are_selectable_by_name(spun_up, tmp_path):
    cfg, checkpoint = spun_up
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    spec = "local-tables:paths=" + ",".join(map(str, obs))
    out = tmp_path / "partial"
    control = ControlOptions(recentre_fraction=0.5, increment_source="ensemble-mean")
    da_door.init(cfg, out, filter_name="letkf", members=3, ensemble_truncation=3,
                 analysis_time_utc=START_TEXT, options=OPTIONS, control_options=control)
    manifest = read_ensemble_manifest(out / ENSEMBLE_MANIFEST_NAME)
    assert manifest["control_options"]["recentre_fraction"] == 0.5
    receipt = da_door.cycle(
        cfg, out, stream_specs=[spec], cycles=1, start_utc=START_TEXT, interval_s=20.0,
        ensemble=out / ENSEMBLE_MANIFEST_NAME, options=OPTIONS, control_options=control,
    )
    assert receipt["status"] == "pass"
    report = json.loads((out / "assimilation-report-step00000002.json").read_text())
    assert report["control"]["increment_source"] == "ensemble-mean"
    assert "comparison experiment" in report["control"]["route"]
    assert report["recentre"]["fraction"] == 0.5
    comparison = report["mean_increment_transfer"]
    assert comparison["difference_rms"]["temperature_k_rms"] == pytest.approx(0.0, abs=1e-12)


def test_fresh_with_the_ensemble_filter_hands_back_a_control_analysis_and_a_forecast_start(spun_up, tmp_path):
    """The one command a user reaches, with the ensemble filter and time
    bins on the smoke case: init, one cycle, the handed-back control
    analysis, the information cutoff and the forecast from it."""
    cfg, checkpoint = spun_up
    obs = _mid_window_obs(cfg, checkpoint, tmp_path)
    out = tmp_path / "fresh-letkf"
    until = START + dt.timedelta(seconds=20)
    receipt = da_door.fresh(
        CONFIG, out, stream_specs=["local-tables:paths=" + ",".join(map(str, obs))], start_utc=START_TEXT,
        until_utc=until.isoformat(), forecast_hours=20.0 / 3600.0, interval_s=20.0, options=OPTIONS,
        now=START + dt.timedelta(hours=1), filter_name="letkf", members=3, ensemble_truncation=3,
        observation_bin_s=10.0, cutoff_utc=START + dt.timedelta(minutes=30),
        control_options=ControlOptions(hybrid_beta=1.0),
    )
    assert receipt["status"] == "pass" and receipt["filter"] == "letkf"
    # Everything was named (the beta named here is the ensemble alone: the
    # smoke ladder has no static table), so nothing was defaulted.
    assert receipt["defaults"]["defaulted"] == [] and receipt["defaults"]["hybrid_beta"] == 1.0
    assert receipt["information_cutoff_utc"] == (START + dt.timedelta(minutes=30)).isoformat(timespec="seconds")
    assert receipt["causal"]["information_cutoff_utc"] == receipt["information_cutoff_utc"]
    assert receipt["settings"]["observation_bin_s"] == 10.0
    handed = receipt["analysis_checkpoint"]
    assert handed["step"] == 2 and Path(handed["path"]).is_file()
    manifest = read_ensemble_manifest(out / ENSEMBLE_MANIFEST_NAME)
    assert manifest["filter"] == "letkf" and len(manifest["members"]) == 3
    report = json.loads((out / "assimilation-report-step00000002.json").read_text())
    assert report["observation_times"]["rows_at_own_time"] == 304
    assert report["control"]["increment_source"] == "control"
    forecast = da_door.forecast(load_config(receipt["config"]), out / "forecast", analysis=handed["path"])
    assert forecast["status"] == "pass" and forecast["final_step"] == 4
    # A cutoff in the future is refused by name.
    with pytest.raises(ValueError, match="future"):
        da_door.fresh(
            CONFIG, tmp_path / "future", stream_specs=["local-tables:paths=x.csv"], start_utc=START_TEXT,
            until_utc=until.isoformat(), interval_s=20.0, now=START, cutoff_utc=START + dt.timedelta(hours=2),
        )


def test_an_analysed_state_above_the_radiation_ceiling_is_named_and_not_handed_back(spun_up):
    """The T127 over T63 twin's noisy arm died inside the radiation tables
    with a 1,096.98 hPa column after a noisy pressure increment; the door
    reads the analysed surface pressure against the radiation tables'
    ceiling (109,663 Pa) and carries the background when it exceeds it.
    The observation vocabulary's 108,000 Pa gross bound is a bound on a
    report, not on a state: a T127 control carries 107.7 kPa at the Andes'
    Pacific foot by construction, so a state between the two is within."""
    from woof.globe.da.analysis import RADIATION_SURFACE_PRESSURE_CEILING_PA
    from woof.globe.da_filter import ANALYSED_STATE_BOUNDS_BREAKAGE, analysed_state_within_bounds
    from woof.globe.state import ArwenGlobalState

    cfg, checkpoint = spun_up
    transform = build_transform(cfg)
    model, _cold = build_model_and_cold_state(cfg, transform)
    metadata, arrays = read_checkpoint(checkpoint, expected_config_hash=cfg.config_hash)
    state = state_from_checkpoint(metadata, arrays, transform.backend)
    model.enforce(state)
    inside = analysed_state_within_bounds(model, transform, state)
    assert inside["within"] is True and inside["columns_outside"] == 0
    assert inside["ceiling_pa"] == RADIATION_SURFACE_PRESSURE_CEILING_PA == 109_663.0
    assert 45_000.0 < inside["surface_pressure_min_pa"] <= inside["surface_pressure_max_pa"] < 108_000.0
    field_index = SPECTRAL_FIELDS.index("log_surface_pressure")

    def lifted_by(factor):
        fields = list(state.atmosphere.fields())
        fields[field_index] = transform.add_grid_constant(fields[field_index], math.log(factor))
        return ArwenGlobalState(state.atmosphere.with_fields(fields), state.surface, state.physics_state)

    # A state whose maximum sits above the report bound and under the
    # ceiling is within: nothing refuses it.
    between = analysed_state_within_bounds(model, transform, lifted_by(108_800.0 / inside["surface_pressure_max_pa"]))
    assert between["within"] is True and 108_000.0 < between["surface_pressure_max_pa"] < 109_663.0
    # Lift the whole surface pressure by ten percent (ln ps + ln 1.1): above
    # the ceiling, named and not handed back.
    outside = analysed_state_within_bounds(model, transform, lifted_by(1.1))
    assert outside["within"] is False and outside["columns_outside"] > 0
    assert outside["surface_pressure_max_pa"] > 109_663.0
    assert outside["breakage"] == ANALYSED_STATE_BOUNDS_BREAKAGE and "1,096.98 hPa" in outside["breakage"]
