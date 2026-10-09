from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import json
from pathlib import Path

import numpy as np
import pytest

from woof.globe.cli import EXIT_REFUSED, main
from woof.globe.config import load_config
from woof.globe.export import export_parent, read_parent_export
from woof.globe.physics.arwen_bridge import NativeArwenPhysicsBridge
from woof.globe.physics.exchange import PhysicsResult
from woof.globe.receipt import check_receipt, write_receipt
from woof.globe.runner import build_model_and_cold_state, run
from woof.globe.state import PhysicsState
from woof.globe.physics.registry import (
    global_physics_manifest,
    register_global_physics_adapter,
)


CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


def _rewrite_npz(path: Path, field: str):
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    values[field].flat[0] += 1.0
    with path.open("wb") as stream:
        np.savez_compressed(stream, **values)


def test_parent_export_is_neutral_hash_bound_and_tamper_evident(tmp_path):
    cfg = load_config(CONFIG)
    run_dir = tmp_path / "run"
    run(cfg, run_dir)
    checkpoint = run_dir / "arwen_global_step00000004.npz"
    target = tmp_path / "parent.npz"
    export_parent(cfg, checkpoint, target, nlat=9, nlon=18)
    metadata, arrays = read_parent_export(target)
    assert metadata["target_grid"]["includes_poles"] is False
    assert "not a WOOF regional initial-condition" in metadata["admission"]
    assert arrays["surface_pressure_pa"].shape == (9, 18)
    assert arrays["potential_temperature_k"].shape[1:] == (9, 18)

    _rewrite_npz(target, "surface_pressure_pa")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_parent_export(target)


def test_missing_native_adapter_refuses_before_step_zero():
    with pytest.raises(ValueError, match="is not admitted"):
        NativeArwenPhysicsBridge("definitely-missing-adapter", {})


def _v1_contract(**overrides):
    contract = {
        "schema": "gpuwm.arwen-global-native-physics-adapter/v1",
        "scheme_identity": {"radiation": "dummy"},
        "backend": "numpy",
        "precision": "float64",
        "required_fields": ["theta", "qv"],
        "pressure_convention": "top-to-bottom-pa",
        "vertical_coordinate": "hybrid-a-b",
        "surface_state": "grid-resident",
        "restart_contract": "all-state-explicit",
        "budget_contract": "water-energy-receipt",
        "evidence_receipt_sha256": "1" * 64,
        "arithmetic_sha256": "2" * 64,
    }
    contract.update(overrides)
    return contract


def _v2_contract(**overrides):
    contract = _v1_contract()
    contract.update(
        schema="gpuwm.arwen-global-native-physics-adapter/v2",
        admission_status="device-pending",
        device_evidence_sha256="0" * 64,
        limitations=["unit-test adapter"],
    )
    contract.update(overrides)
    return contract


class _Dummy:
    def step(self, exchange):
        raise RuntimeError("not called in registration test")


def test_registry_records_the_contract_and_never_promotes_on_digests_alone():
    # Registration validates contract shape and records hashes; it binds no
    # digest to an artifact, so a fabricated digest must not buy any status
    # above the one the registry assigns itself.
    name = "unit-test-adapter-v1"
    register_global_physics_adapter(
        name, lambda options: _Dummy(), _v1_contract(), replace=True
    )
    manifest = global_physics_manifest()
    assert name in manifest
    assert len(manifest[name]["contract_hash"]) == 64
    assert manifest[name]["admission_status"] == "experimental"
    assert manifest[name]["contract"]["device_evidence_sha256"] == "0" * 64
    assert any(
        "no Level-5 device qualification" in row
        for row in manifest[name]["contract"]["limitations"]
    )
    bridge = NativeArwenPhysicsBridge(name, {})
    assert bridge.identity["adapter_name"] == name
    assert bridge.identity["admission_status"] == "experimental"


def test_registry_refuses_validated_status_without_device_evidence():
    with pytest.raises(ValueError, match="nonzero device evidence"):
        register_global_physics_adapter(
            "unit-test-validated-adapter",
            lambda options: _Dummy(),
            _v2_contract(admission_status="validated"),
            replace=True,
        )


def test_run_receipt_from_another_pin_document_is_refused(tmp_path):
    cfg = load_config(CONFIG)
    out = tmp_path / "run"
    run(cfg, out)
    receipt = out / "arwen-global-receipt.json"
    assert check_receipt(receipt)["status"] == "pass"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload.pop("self_sha256")
    payload["pins_hash"] = "f" * 64
    write_receipt(receipt, payload)
    with pytest.raises(ValueError, match="arithmetic pins mismatch"):
        check_receipt(receipt)
    assert main(["check-receipt", str(receipt)]) == EXIT_REFUSED


def _exchange():
    model, state = build_model_and_cold_state(load_config(CONFIG))
    return model._physics_exchange(state, 5.0)


def _result(exchange, **overrides):
    prognostics = {
        name: np.array(value, copy=True)
        for name, value in exchange.prognostics().items()
    }
    prognostics.update(overrides)
    return PhysicsResult(
        **prognostics,
        surface=exchange.surface.copy(),
        physics_state=PhysicsState(),
        diagnostics={},
        adapter_receipt={"mode": "arwen-native"},
    )


def _bridge(name, step):
    class Adapter:
        def __init__(self, options):
            self.options = options

        def step(self, exchange):
            return step(exchange)

    register_global_physics_adapter(
        name, Adapter, _v2_contract(), replace=True
    )
    return NativeArwenPhysicsBridge(name, {})


def test_bridge_admits_a_conforming_adapter():
    exchange = _exchange()
    bridge = _bridge("unit-test-conforming-adapter", lambda ex: _result(ex))
    result = bridge.step(exchange)
    assert np.array_equal(result.theta, np.asarray(exchange.theta))


def test_bridge_refuses_an_adapter_that_writes_into_the_caller_state():
    exchange = _exchange()

    def mutate(ex):
        ex.theta[0, 0, 0] += 1.0
        return _result(ex)

    bridge = _bridge("unit-test-mutating-adapter", mutate)
    with pytest.raises(ValueError, match="wrote into the caller's state"):
        bridge.step(exchange)


def test_bridge_refuses_an_adapter_that_writes_into_the_caller_surface():
    exchange = _exchange()

    def mutate(ex):
        result = _result(ex)
        ex.surface.temperature_k[0, 0] += 5.0
        return result

    bridge = _bridge("unit-test-surface-mutating-adapter", mutate)
    with pytest.raises(ValueError, match="surface.surface_temperature_k"):
        bridge.step(exchange)


def test_bridge_refuses_non_finite_and_negative_results():
    exchange = _exchange()

    def nan_theta(ex):
        theta = np.array(ex.theta, copy=True)
        theta[0, 0, 0] = np.nan
        return _result(ex, theta=theta)

    bridge = _bridge("unit-test-nan-adapter", nan_theta)
    with pytest.raises(FloatingPointError, match="non-finite theta"):
        bridge.step(exchange)

    def negative_qv(ex):
        qv = np.array(ex.qv, copy=True)
        qv[0, 0, 0] = -1.0e-3
        return _result(ex, qv=qv)

    bridge = _bridge("unit-test-negative-adapter", negative_qv)
    with pytest.raises(FloatingPointError, match="returned negative qv"):
        bridge.step(exchange)


def test_cli_pins_manifest_run_and_native_refusal(tmp_path, capsys):
    assert main(["pins"]) == 0
    pins = json.loads(capsys.readouterr().out)
    assert len(pins["sha256"]) == 64

    assert main(["physics-manifest"]) == 0
    json.loads(capsys.readouterr().out)

    out = tmp_path / "run"
    assert main(["run", CONFIG, "--outdir", str(out)]) == 0
    assert (out / "arwen-global-receipt.json").exists()
    capsys.readouterr()

    native = str(_shipped_configs() / "arwen_global_native_physics_bridge_example.toml")
    assert main(["run", native, "--outdir", str(tmp_path / "native")]) == EXIT_REFUSED
    assert "is not admitted" in capsys.readouterr().err
