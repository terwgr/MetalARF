from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from woof.globe.checkpoint import read_checkpoint
from woof.globe.config import load_config
from woof.globe.dynamics import MoistHybridModel
from woof.globe.insitu import (
    CAPTURE_NAMES,
    LEDGER_NAMES,
    ComponentCapture,
    InsituLedger,
    InsituOptions,
    THRESHOLDS,
    attach_capture,
)
from woof.globe.insitu.ledger import LEDGER_NAME
from woof.globe.physics.native_suite import ArwenCudaColumnSuite
from woof.globe.receipt import check_receipt
from woof.globe.runner import build_model_and_cold_state, run

from test_arwen_global_level5_native import _fake_modules, _options

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
REFERENCE_COMPONENTS = (
    "surface_radiation_fluxes", "turbulence", "convective_adjust",
    "betts_miller", "saturation_adjust", "microphysics", "negative_clamp",
)


def _smoke(steps: int = 24, **overrides):
    cfg = load_config(CONFIG)
    return replace(
        cfg, duration_s=cfg.dt_s * steps, output_interval_s=cfg.dt_s * steps,
        **overrides,
    )


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# -- component capture at the physics seams ------------------------------

def test_native_runtime_marks_every_scheme():
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    exchange = model._physics_exchange(state, 5.0)
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=_fake_modules())
    capture = ComponentCapture(model.transform.grid.quadrature_weights)
    assert attach_capture(suite, capture) == ["ArwenCudaColumnSuite"]
    capture.begin_step()
    suite.step(exchange)
    records = capture.drain()
    assert [name for _, name, _ in records] == ["rrtmgp", "sfclay", "noah", "ysu", "gf", "morrison"]
    by_name = {name: dict(zip(CAPTURE_NAMES, np.asarray(vec))) for _, name, vec in records}
    # The fake Morrison bumps one theta cell by 0.125 in place; everything
    # else in the fakes is a zero tendency.
    assert by_name["morrison"]["theta_max_abs_change"] == pytest.approx(0.125, abs=1e-6)
    assert by_name["ysu"]["theta_max_abs_change"] == 0.0
    assert by_name["rrtmgp"]["qv_max_abs_change"] == 0.0



# -- the runtime ------------------------------------------------------------

def test_ledger_on_and_off_are_bit_identical(tmp_path):
    on = _smoke(24)
    off = replace(on, insitu=InsituOptions(enabled=False))
    assert on.config_hash == off.config_hash
    hashes = {}
    for name, cfg in (("on", on), ("off", off)):
        result = run(cfg, tmp_path / name)
        assert result["status"] == "pass"
        metadata, _ = read_checkpoint(tmp_path / name / "arwen_global_step00000024.npz")
        hashes[name] = {key: value["sha256"] for key, value in metadata["arrays"].items()}
    assert hashes["on"] == hashes["off"]
    assert len(hashes["on"]) >= 27
    assert (tmp_path / "on" / LEDGER_NAME).exists()
    assert not (tmp_path / "off" / LEDGER_NAME).exists()


def test_ledger_rows_are_batched_and_summarised_in_the_receipt(tmp_path):
    cfg = _smoke(24)
    result = run(cfg, tmp_path / "run")
    rows = _rows(tmp_path / "run" / LEDGER_NAME)
    assert rows[0]["kind"] == "header"
    assert [term["name"] for term in rows[0]["terms"]] == list(LEDGER_NAMES)
    assert set(rows[0]["tripwires"]) == set(THRESHOLDS)
    steps = [row for row in rows if row["kind"] == "step"]
    spectra = [row for row in rows if row["kind"] == "spectra"]
    assert [row["step"] for row in steps] == list(range(1, 25))
    assert [row["step"] for row in spectra] == [10, 20]
    assert len(spectra[0]["rot_by_degree"]) == cfg.truncation + 1
    assert len(spectra[0]["rot_top_decile_by_level"]) == cfg.vertical.nlev
    for row in steps:
        assert set(row["terms"]) == set(LEDGER_NAMES)
        # Both Strang halves, every component, in the order they ran.
        assert [record["call"] for record in row["physics"]] == [0] * 7 + [1] * 7
        assert tuple(record["component"] for record in row["physics"][:7]) == REFERENCE_COMPONENTS
        assert tuple(record["component"] for record in row["physics"][7:]) == REFERENCE_COMPONENTS
        for record in row["physics"]:
            assert set(record) == {"call", "component", *CAPTURE_NAMES}
        assert "levy_kg_m2" in row["metrics"]
    # The smoke run's physics does something every step; the capture must
    # not read zero everywhere.
    assert max(
        row["physics"][0]["theta_max_abs_change"] for row in steps
    ) > 0.0
    summary = result["insitu"]
    assert summary["step_rows"] == 24 and summary["spectra_rows"] == 2
    assert summary["flushes"] == 3
    assert summary["attached_physics_capture"] == ["ReferencePhysics"]
    assert set(summary["drift_per_hour"]) == {
        "mass_pa", "water_total_kg_m2", "total_dry_energy_j_m2",
        "moist_total_energy_j_m2", "axial_angular_momentum_kg_s",
    }
    assert summary["drift_per_hour"]["mass_pa"]["relative_per_hour"] == pytest.approx(0.0, abs=1e-9)
    assert summary["first_trip"] is None and summary["trip_count"] == 0
    assert summary["max_abs_mass_fixer_log_offset"] >= 0.0
    assert summary["observer_wall_seconds"] > 0.0
    assert check_receipt(tmp_path / "run" / "arwen-global-receipt.json")["insitu"] == summary
    # A second run with --overwrite sweeps the ledger with the other owned files.
    run(cfg, tmp_path / "run", overwrite=True)
    assert sum(row["kind"] == "header" for row in _rows(tmp_path / "run" / LEDGER_NAME)) == 1


def test_tripwire_trip_snapshots_the_state_and_records_the_term(tmp_path):
    cfg = replace(_smoke(24), insitu=InsituOptions(max_snapshots=1))
    model, state = build_model_and_cold_state(cfg)
    ledger = InsituLedger(
        cfg, model, tmp_path,
        tripwire_overrides={
            "kinetic_energy_step_change": 0.0,
            "temperature_ceiling_pre_warning": 0.0,
        },
    ).attach()
    for _ in range(24):
        state, _ = model.step(state, cfg.dt_s)
    summary = ledger.close()
    # The absolute ceiling wire fires on every flush (steps 10, 20, 24);
    # the KE envelope wire arms after the 20-row window and fires once, in
    # the last batch.  Each flush records one trip per wire, but a wire
    # snapshots once and the cap is one.
    assert summary["trip_count"] == 4
    assert summary["trips_by_tripwire"] == {
        "kinetic_energy_step_change": 1, "temperature_ceiling_pre_warning": 3,
    }
    assert len(summary["snapshots"]) == 1
    # The earliest onset owns the one snapshot.
    first = summary["first_trip"]
    assert first["tripwire"] == "temperature_ceiling_pre_warning"
    assert first["step"] == 1 and first["term"] == "temperature_max_k"
    assert first["value"] > 0.0 and first["threshold"] == 0.0
    snapshot = Path(summary["snapshots"][0])
    assert snapshot.name == "insitu_trip_temperature_ceiling_pre_warning_step00000010.npz"
    metadata, _ = read_checkpoint(snapshot, expected_config_hash=cfg.config_hash)
    assert metadata["step"] == 10
    record = json.loads(snapshot.with_suffix(".json").read_text(encoding="utf-8"))
    assert record["trip"] == first
    assert record["snapshot_step"] == 10 and record["snapshot_lag_steps"] == 9
    assert [row["step"] for row in record["history"]] == list(range(1, 11))
    trips = [row for row in _rows(tmp_path / LEDGER_NAME) if row["kind"] == "trip"]
    assert sorted((row["tripwire"], row["step"]) for row in trips) == [
        ("kinetic_energy_step_change", 21),
        ("temperature_ceiling_pre_warning", 1), ("temperature_ceiling_pre_warning", 11),
        ("temperature_ceiling_pre_warning", 21),
    ]
    kinetic = [row for row in trips if row["tripwire"] == "kinetic_energy_step_change"][0]
    assert kinetic["term"] == "kinetic_energy_j_m2" and kinetic["value"] > 0.0
    assert "breakage" in trips[0] and trips[0]["direction"] == "max"


def test_failure_receipt_carries_the_ledger_and_the_refused_state(tmp_path, monkeypatch):
    cfg = _smoke(12)
    original = MoistHybridModel.enforce

    def refusing_enforce(self, bundle):
        original(self, bundle)
        if bundle.step >= 3:
            raise FloatingPointError("synthetic research-bound refusal")

    monkeypatch.setattr(MoistHybridModel, "enforce", refusing_enforce)
    with pytest.raises(FloatingPointError, match="synthetic"):
        run(cfg, tmp_path / "fail")
    receipt = check_receipt(tmp_path / "fail" / "arwen-global-receipt.json")
    assert receipt["status"] == "error" and receipt["completed_step"] == 2
    summary = receipt["insitu"]
    # The observer saw step 3 before the refusal: the rows up to it flushed
    # and the refused state itself is archived as a checkpoint.
    assert summary["step_rows"] == 3
    refused = Path(summary["refused_snapshot"])
    assert refused.name == "insitu_refused_step00000003.npz" and refused.exists()
    metadata, _ = read_checkpoint(refused, expected_config_hash=cfg.config_hash)
    assert metadata["step"] == 3
    steps = [row["step"] for row in _rows(tmp_path / "fail" / LEDGER_NAME) if row["kind"] == "step"]
    assert steps == [1, 2, 3]


# -- config -----------------------------------------------------------------

def test_insitu_table_is_parsed_and_outside_the_identity(tmp_path):
    source = Path(CONFIG).read_text(encoding="utf-8")
    plain = load_config(CONFIG)
    assert plain.insitu == InsituOptions()
    tuned = tmp_path / "tuned.toml"
    tuned.write_text(source + "\n[insitu]\nflush_every = 3\nspectra_every = 2\nsnapshot_on_trip = false\n", encoding="utf-8")
    cfg = load_config(tuned)
    assert cfg.insitu.flush_every == 3 and cfg.insitu.spectra_every == 2
    assert cfg.insitu.snapshot_on_trip is False and cfg.insitu.enabled is True
    assert cfg.config_hash == plain.config_hash
    assert "insitu" not in cfg.config_identity
    off = tmp_path / "off.toml"
    off.write_text(source + "\n[insitu]\nenabled = false\n", encoding="utf-8")
    assert load_config(off).insitu.enabled is False
    assert load_config(off).config_hash == plain.config_hash
    bad = tmp_path / "bad.toml"
    bad.write_text(source + "\n[insitu]\ncadence = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys in \\[insitu\\]: cadence"):
        load_config(bad)
    zero = tmp_path / "zero.toml"
    zero.write_text(source + "\n[insitu]\nflush_every = 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="insitu.flush_every must be a positive integer"):
        load_config(zero)
    with pytest.raises(ValueError, match="insitu.enabled must be true or false"):
        load_config(_write(tmp_path / "notbool.toml", source + "\n[insitu]\nenabled = 1\n"))


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path
