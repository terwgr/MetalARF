"""The control twin: perfect-model synthetic cycling of the door's control
path at two resolutions (T7 control, T3 ensemble, six members) on the CPU.

Four families both directions (agreeing reports move the control by
rounding only; perfect reports pull it to the truth at every analysis;
noisy reports recover; reports drawn from the mirror of the truth about
the control push it away at every analysis), and the comparison arm of amendment A (the
ensemble-mean transfer) run on the same truth, network and seed so its
curve lies beside the control path's in the receipt.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import json

import pytest

from woof.globe.da_twin import (
    AGREE_RELATIVE_BOUND,
    ControlTwinSetup,
    main,
    run_control_twin,
)

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


def _setup(**kwargs) -> ControlTwinSetup:
    # The independent perturbation family at the smoke truncation: the
    # balanced family scales a T3 planetary wind to a fraction of a metre
    # per second to balance the stated pressure, and the twin's families
    # need wind spread to pull.  Its amplitudes by name too (the defaults
    # are the balanced family's calibrated 0.8 hPa and 3.0 m/s; the
    # readings below were taken at 1.5 hPa and 2.5 m/s).
    base = dict(config=CONFIG, control_truncation=7, ensemble_truncation=3, members=6, cycles=2,
                perturbation_balance="none", perturbation_ln_surface_pressure=0.0015, perturbation_wind_m_s=2.5)
    base.update(kwargs)
    return ControlTwinSetup(**base)


def test_agreeing_reports_move_the_control_by_rounding_only(tmp_path):
    report = run_control_twin(_setup(family="agree"), tmp_path, progress=lambda *_: None)
    verdict = report["verdict"]
    assert verdict["passed"] is True
    assert verdict["largest_relative_control_change"] <= AGREE_RELATIVE_BOUND
    assert verdict["largest_relative_control_change_fields"] == ["temperature_k", "u", "v", "surface_pressure_pa"]
    # On the float64 smoke core the repair leaves the vapor alone too.
    assert verdict["largest_relative_vapor_change"] <= 1.0e-10
    assert report["control"]["truncation"] == 7 and report["ensemble"]["truncation"] == 3
    assert report["ensemble"]["members"] == 6
    for cycle in report["cycles"]:
        assert cycle["status"] == "pass"
        assert cycle["control"]["increment_source"] == "control"
        # The package's default taper at T3: one to 0.6 T (degree 2), zero at T.
        assert cycle["control"]["taper"]["start_degree"] == 2 and cycle["control"]["taper"]["end_degree"] == 3
    assert (tmp_path / "control-twin-agree-control.json").is_file()


def test_perfect_reports_pull_the_control_to_the_truth_at_every_analysis(tmp_path):
    report = run_control_twin(_setup(family="perfect", cycles=3), tmp_path, progress=lambda *_: None)
    verdict = report["verdict"]
    assert verdict["passed"] is True
    assert verdict["every_analysis_improved_temperature"] and verdict["every_analysis_improved_wind"]
    first, last = verdict["temperature_rmse_first_to_last"]
    assert last < 0.85 * first
    for cycle in report["cycles"]:
        assert cycle["after"]["surface_pressure_pa"] < cycle["before"]["surface_pressure_pa"]
        # The ensemble-mean increment is recorded beside the control's and
        # differs from it: two paths, one applied.
        comparison = cycle["mean_increment_transfer"]
        assert comparison["difference_rms"]["temperature_k_rms"] > 0.0
        assert cycle["recentre"]["fraction"] == 1.0


def test_mirror_reports_push_the_control_away_from_the_truth_at_every_analysis(tmp_path):
    """The fourth family (the 2026-09-06 refutation): the same network and
    errors as the perfect family with the reports drawn from ``2 c - t``,
    so they point the other way.  The filter has to follow them: the rmse
    against the truth rises at every analysis on temperature, wind and
    surface pressure, and by more over three analyses than the perfect
    family takes off (the square of the increment adds in both
    directions and the cross term flips sign).  A filter that damped
    every increment would pass the perfect family weakly and fail here."""
    report = run_control_twin(_setup(family="mirror", cycles=3), tmp_path, progress=lambda *_: None)
    verdict = report["verdict"]
    assert verdict["passed"] is True
    assert verdict["every_analysis_worsened_temperature"] and verdict["every_analysis_worsened_wind"]
    first, last = verdict["temperature_rmse_first_to_last"]
    # Measured 1.317 to 2.695 K on the smoke core under the cosine-mode
    # displacement; 2.485 to 3.496 K once the displacement carries the
    # internal modes (the same 1.5 hPa displaces twice the temperature, so
    # the same push is a smaller ratio); the claim is a rise of a quarter.
    assert last > 1.25 * first, (first, last)
    for cycle in report["cycles"]:
        assert cycle["status"] == "pass"
        assert cycle["after"]["surface_pressure_pa"] > cycle["before"]["surface_pressure_pa"]
        assert cycle["reports"] > 0 and cycle["assimilated"] > 0


def test_the_comparison_arm_runs_on_the_same_truth_and_its_curve_is_recorded(tmp_path):
    control = run_control_twin(_setup(family="recovery", cycles=3), tmp_path / "control", progress=lambda *_: None)
    mean = run_control_twin(
        _setup(family="recovery", cycles=3, increment_source="ensemble-mean"), tmp_path / "mean",
        progress=lambda *_: None,
    )
    assert control["verdict"]["passed"] is True and mean["verdict"]["passed"] is True
    # The same displaced start (same seed, same truth).
    assert control["initial_score"] == mean["initial_score"]
    assert mean["cycles"][0]["control"]["increment_source"] == "ensemble-mean"
    # Two different analysis paths give two different curves, both recorded.
    assert control["cycles"][-1]["after"] != mean["cycles"][-1]["after"]
    assert (tmp_path / "mean" / "control-twin-recovery-ensemble-mean.json").is_file()


def test_the_twin_door_refuses_an_ensemble_above_the_control(tmp_path):
    with pytest.raises(ValueError, match="exceeds the control"):
        run_control_twin(_setup(control_truncation=3, ensemble_truncation=7), tmp_path, progress=lambda *_: None)
    with pytest.raises(ValueError, match="family"):
        ControlTwinSetup(config=CONFIG, family="drift")


def test_the_twin_cli_writes_the_report_and_exits_on_the_verdict(tmp_path, capsys):
    code = main([
        "--config", CONFIG, "--outdir", str(tmp_path), "--control-truncation", "7",
        "--ensemble-truncation", "3", "--members", "6", "--cycles", "2", "--family", "agree",
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "twin: verdict" in printed
    report = json.loads((tmp_path / "control-twin-agree-control.json").read_text())
    assert report["verdict"]["passed"] is True
