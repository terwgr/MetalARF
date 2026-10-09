from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from woof.globe import native_qualification
from woof.globe.checkpoint import normalize_trackers, write_checkpoint
from woof.globe.cli import EXIT_REFUSED, main
from woof.globe.config import load_config
from woof.globe.native_qualification import (
    qualify_native_adapter,
    read_native_contract_candidate,
    read_native_device_evidence,
)
from woof.globe.physics.builtin_adapters import ADAPTER_NAME
from woof.globe.pins import pins_hash
from woof.globe.receipt import write_receipt
from woof.globe.runner import (
    CHECKPOINT_PREFIX,
    RECEIPT_NAME,
    build_model_and_cold_state,
)


NATIVE_CONFIG = str(_shipped_configs() / "arwen_global_level5_native_smoke.toml")
STATE_CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


def _stub_cupy():
    """Stands in for a present CUDA device without touching a real one."""
    runtime = SimpleNamespace(
        getDevice=lambda: 0,
        getDeviceProperties=lambda ordinal: {
            "name": b"stub-device", "totalGlobalMem": 1 << 30,
        },
        driverGetVersion=lambda: 12040,
        runtimeGetVersion=lambda: 12040,
        deviceSynchronize=lambda: None,
    )
    return SimpleNamespace(
        __version__="stub", cuda=SimpleNamespace(runtime=runtime)
    )


def _gates(passed: bool) -> dict[str, object]:
    return {
        "mass_relative_drift": {
            "value": 0.0 if passed else 1.0, "limit": 1.0e-7, "passed": passed,
        },
        "total_water_relative_drift": {
            "value": 0.0, "limit": 1.0e-6, "passed": True,
        },
    }


def _campaign_runner(monkeypatch, *, status: str, config_hash: str | None = None):
    """A runner that writes real checkpoints and returns a real receipt shape.

    Both campaigns write byte-identical checkpoints, so a refusal in the
    battery can only come from the receipt gates, never from the restart
    comparison.
    """
    _, state = build_model_and_cold_state(load_config(STATE_CONFIG))
    trackers = normalize_trackers()
    written: list[str] = []

    def fake_run(cfg, outdir, *, restart=None, overwrite=False, progress=None):
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = cfg.config_hash if config_hash is None else config_hash
        steps = (0, 2, 4) if restart is None else (4,)
        for step in steps:
            state.atmosphere.step = step
            state.atmosphere.time_s = float(step) * cfg.dt_s
            write_checkpoint(
                out / f"{CHECKPOINT_PREFIX}{step:08d}.npz",
                state,
                config_hash=stamp,
                to_numpy=np.asarray,
                trackers=trackers,
                semi_implicit_scheme=cfg.semi_implicit_scheme,
            )
        write_receipt(out / RECEIPT_NAME, {
            "name": cfg.name,
            "status": status,
            "config_hash": cfg.config_hash,
            "pins_hash": pins_hash(cfg.semi_implicit_scheme),
            "gates": _gates(status == "pass"),
            "run_trackers": trackers,
        })
        written.append(str(out))
        return {
            "name": cfg.name,
            "status": status,
            "gates": _gates(status == "pass"),
            "run_trackers": trackers,
            "receipt_path": str(out / RECEIPT_NAME),
        }

    # These tests stub the device: the GPUWM_NO_LOCAL_GPU refusal guards the
    # real card (27531ebcd) and must not fire on a stubbed cupy.
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    monkeypatch.setitem(sys.modules, "cupy", _stub_cupy())
    monkeypatch.setattr(native_qualification, "run", fake_run)
    return written


def test_qualification_passes_and_emits_a_candidate_when_both_campaigns_pass(
    tmp_path, monkeypatch
):
    _campaign_runner(monkeypatch, status="pass")
    cfg = load_config(NATIVE_CONFIG)
    out = tmp_path / "qualification"
    evidence_path, candidate_path = qualify_native_adapter(cfg, out)
    evidence = read_native_device_evidence(evidence_path)
    assert evidence["status"] == "pass"
    assert evidence["restart_comparison"]["bit_exact"] is True
    assert evidence["campaign"]["receipt_gates"]["continuous"]["status"] == "pass"
    assert evidence["campaign"]["receipt_gates"]["resumed"]["status"] == "pass"
    candidate = read_native_contract_candidate(candidate_path)
    assert candidate["adapter_name"] == ADAPTER_NAME
    assert candidate["proposed_admission_status"] == "experimental"
    assert candidate["device_evidence_self_sha256"] == evidence["self_sha256"]


def test_qualification_evidence_hashes_every_adapter_source_file():
    """Every adapter source file is in the evidence, keyed by where it is.

    The keys used to be asserted as `woof/arwen_global/physics/<file>`,
    which is the path shape on the tree this package was carved from.  In an
    install the driver lives at `<site-packages>/arwen_global/physics/`, so
    the loop above the assertion globbed a directory that does not exist,
    ran zero times, and the one literal below it failed on every card host.
    The scope is read off the INSTALLED package rather than written out, and
    the two named files stay named because they are the two the evidence
    exists to identify.
    """

    hashed = set(native_qualification._source_hashes())
    driver = Path(native_qualification.__file__).resolve().parent / "physics"
    for path in sorted(driver.glob("*.py")):
        suffix = "woof/globe/physics/" + path.name
        assert any(key.endswith(suffix) for key in hashed), path
    assert any(key.endswith("woof/globe/physics/arwen_bridge.py")
               for key in hashed)
    assert any(key.endswith("woof/globe/physics/exchange.py")
               for key in hashed)
    assert any(key.endswith("woof/globe/native_qualification.py")
               for key in hashed)


def test_failed_campaign_gates_refuse_and_emit_no_candidate(tmp_path, monkeypatch):
    _campaign_runner(monkeypatch, status="fail")
    cfg = load_config(NATIVE_CONFIG)
    out = tmp_path / "qualification"
    with pytest.raises(RuntimeError, match="durable evidence"):
        qualify_native_adapter(cfg, out)
    evidence = read_native_device_evidence(out / "native-device-evidence.json")
    assert evidence["status"] == "error"
    assert evidence["error_type"] == "FloatingPointError"
    assert "receipt gates" in evidence["error_message"]
    assert "mass_relative_drift" in evidence["error_message"]
    assert evidence["candidate_emitted"] is False
    assert not (out / "native-contract-candidate.json").exists()


def test_checkpoints_from_another_config_are_refused(tmp_path, monkeypatch):
    _campaign_runner(monkeypatch, status="pass", config_hash="c" * 64)
    cfg = load_config(NATIVE_CONFIG)
    out = tmp_path / "qualification"
    with pytest.raises(RuntimeError, match="durable evidence"):
        qualify_native_adapter(cfg, out)
    evidence = read_native_device_evidence(out / "native-device-evidence.json")
    assert evidence["status"] == "error"
    assert "not the qualified config" in evidence["error_message"]


def test_overwrite_replaces_only_the_artifacts_this_door_owns(tmp_path, monkeypatch):
    _campaign_runner(monkeypatch, status="pass")
    cfg = load_config(NATIVE_CONFIG)
    out = tmp_path / "qualification"
    (out / "plots").mkdir(parents=True)
    (out / "plots" / "a.png").write_bytes(b"png")
    (out / "operator-notes.md").write_text("keep me\n", encoding="utf-8")
    qualify_native_adapter(cfg, out)
    qualify_native_adapter(cfg, out, overwrite=True)
    assert (out / "operator-notes.md").read_text(encoding="utf-8") == "keep me\n"
    assert (out / "plots" / "a.png").exists()
    assert (out / "native-device-evidence.json").exists()


def test_qualification_without_overwrite_refuses_its_own_prior_artifacts(
    tmp_path, monkeypatch
):
    _campaign_runner(monkeypatch, status="pass")
    cfg = load_config(NATIVE_CONFIG)
    out = tmp_path / "qualification"
    qualify_native_adapter(cfg, out)
    with pytest.raises(FileExistsError, match="native-device-evidence.json"):
        qualify_native_adapter(cfg, out)


def test_cli_manifest_contains_device_pending_builtin(capsys):
    assert main(["physics-manifest"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[ADAPTER_NAME]["admission_status"] == "device-pending"
    assert payload[ADAPTER_NAME]["contract"]["device_evidence_sha256"] == "0" * 64
    # No evidence receipt artifact exists yet, so the field says so instead of
    # carrying a well-formed digest of nothing.
    assert payload[ADAPTER_NAME]["contract"]["evidence_receipt_sha256"] == "0" * 64


def test_cli_native_qualify_reports_pass_and_checks_both_artifacts(
    tmp_path, monkeypatch, capsys
):
    _campaign_runner(monkeypatch, status="pass")
    out = tmp_path / "qualification"
    assert main(["native-qualify", NATIVE_CONFIG, "--outdir", str(out)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass"
    assert main(["check-native-evidence", str(out / "native-device-evidence.json")]) == 0
    capsys.readouterr()
    assert main([
        "check-native-candidate", str(out / "native-contract-candidate.json"),
    ]) == 0


def test_cli_native_qualify_returns_refusal_but_leaves_evidence(
    tmp_path, monkeypatch, capsys
):
    _campaign_runner(monkeypatch, status="fail")
    out = tmp_path / "qualification"
    # EXIT_REFUSED, this distribution's code for a refusal a human reads.
    # The number written here used to be 2, which is this package's
    # EXIT_ARGUMENTS and was the source tree's code for a refusal: a test
    # asserting an exit code by literal cannot see a table that moved under
    # it.  Read from the table, so it cannot drift again.
    assert main(["native-qualify", NATIVE_CONFIG, "--outdir", str(out)]) == EXIT_REFUSED
    assert "durable evidence" in capsys.readouterr().err
    assert (out / "native-device-evidence.json").exists()
    # The check command reports a failed evidence record as non-green.
    assert main(["check-native-evidence", str(out / "native-device-evidence.json")]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"


def test_native_qualification_on_this_host_without_cuda_is_a_durable_error(
    tmp_path, monkeypatch
):
    cfg = load_config(NATIVE_CONFIG)
    out = tmp_path / "qualification"
    try:
        import cupy  # noqa: F401
    except Exception:
        pass
    else:
        pytest.skip("a real CuPy is importable; this arm covers its absence")
    # This arm is about a host that HAS NO CUDA, not about a host that has
    # been told not to look: the refusal the switch produces is a different
    # sentence from the one under test, so a run carrying it read the wrong
    # refusal and failed.  Both CI jobs set the switch in the job
    # environment, so this stays; what it no longer works around is a
    # module deciding the switch for the session at import time, which
    # `pytest.mark.cpu_only` and `conftest._cpu_only_marked_tests` ended on
    # 2026-09-10.  Cleared for this test only, which is the condition it is
    # written about.
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    with pytest.raises(RuntimeError, match="durable evidence"):
        qualify_native_adapter(cfg, out)
    evidence = read_native_device_evidence(out / "native-device-evidence.json")
    assert evidence["status"] == "error"
    assert evidence["error_type"] == "ModuleNotFoundError"
    assert evidence["candidate_emitted"] is False
