from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import json
from pathlib import Path

import numpy as np
import pytest

from woof.globe.spectral.checkpoint import (
    read_checkpoint,
    state_from_checkpoint,
    write_checkpoint,
)
from woof.globe.spectral.config import load_config
from woof.globe.spectral.runner import build_model_and_state, build_transform


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / str(_shipped_configs() / "global_spectral_primitive_smoke.toml")


def test_checkpoint_roundtrip_preserves_complex_state_exactly(tmp_path):
    cfg = load_config(CONFIG)
    transform = build_transform(cfg)
    _model, state, _cold, _meta = build_model_and_state(cfg, transform)
    path = write_checkpoint(
        tmp_path / "state.npz",
        state,
        model=cfg.model,
        config_hash=cfg.config_hash,
        to_numpy=transform.backend.to_numpy,
    )
    metadata, arrays = read_checkpoint(path, expected_config_hash=cfg.config_hash)
    restored = state_from_checkpoint(metadata, arrays, transform.backend)
    for left, right in zip(state.fields(), restored.fields()):
        np.testing.assert_array_equal(
            transform.backend.to_numpy(left), transform.backend.to_numpy(right)
        )
    assert restored.step == state.step
    assert restored.time_s == state.time_s


def test_checkpoint_rejects_config_identity_mismatch(tmp_path):
    cfg = load_config(CONFIG)
    transform = build_transform(cfg)
    _model, state, _cold, _meta = build_model_and_state(cfg, transform)
    path = write_checkpoint(
        tmp_path / "state.npz",
        state,
        model=cfg.model,
        config_hash=cfg.config_hash,
        to_numpy=transform.backend.to_numpy,
    )
    with pytest.raises(ValueError, match="config identity"):
        read_checkpoint(path, expected_config_hash="0" * 64)


def test_checkpoint_rejects_array_tampering(tmp_path):
    cfg = load_config(CONFIG)
    transform = build_transform(cfg)
    _model, state, _cold, _meta = build_model_and_state(cfg, transform)
    path = write_checkpoint(
        tmp_path / "state.npz",
        state,
        model=cfg.model,
        config_hash=cfg.config_hash,
        to_numpy=transform.backend.to_numpy,
    )
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload["temperature"].flat[0] += 1.0
    tampered = tmp_path / "tampered.npz"
    with tampered.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    with pytest.raises(ValueError, match="hash mismatch"):
        read_checkpoint(tampered)


def test_restart_trajectory_matches_uninterrupted_bit_for_bit(tmp_path):
    cfg = load_config(CONFIG)
    transform = build_transform(cfg)
    full_model, full, _cold, _meta = build_model_and_state(cfg, transform)
    for _ in range(12):
        full, _ = full_model.step(full, cfg.dt_s)

    split_model, split, _cold, _meta = build_model_and_state(cfg, transform)
    for _ in range(5):
        split, _ = split_model.step(split, cfg.dt_s)
    checkpoint = write_checkpoint(
        tmp_path / "split.npz",
        split,
        model=cfg.model,
        config_hash=cfg.config_hash,
        to_numpy=transform.backend.to_numpy,
    )
    resumed_model, resumed, _cold, metadata = build_model_and_state(
        cfg, transform, restart=checkpoint
    )
    assert metadata is not None
    for _ in range(7):
        resumed, _ = resumed_model.step(resumed, cfg.dt_s)

    for uninterrupted, restarted in zip(full.fields(), resumed.fields()):
        np.testing.assert_array_equal(
            transform.backend.to_numpy(uninterrupted),
            transform.backend.to_numpy(restarted),
        )


def test_checkpoint_carries_validated_run_trackers(tmp_path):
    cfg = load_config(CONFIG)
    transform = build_transform(cfg)
    _model, state, _cold, _meta = build_model_and_state(cfg, transform)
    trackers = {
        "maximum_spectral_cfl": 0.5,
        "maximum_mass_fixer_log_offset": 2.5e-12,
    }
    path = write_checkpoint(
        tmp_path / "tracked.npz",
        state,
        model=cfg.model,
        config_hash=cfg.config_hash,
        to_numpy=transform.backend.to_numpy,
        run_trackers=trackers,
    )
    metadata, _arrays = read_checkpoint(path)
    assert metadata["run_trackers"] == trackers


def test_restart_receipt_inherits_pre_checkpoint_run_maxima(tmp_path):
    from woof.globe.spectral.runner import run

    cfg = load_config(CONFIG)
    transform = build_transform(cfg)
    _model, state, _cold, _meta = build_model_and_state(cfg, transform)
    checkpoint = write_checkpoint(
        tmp_path / "restart.npz",
        state,
        model=cfg.model,
        config_hash=cfg.config_hash,
        to_numpy=transform.backend.to_numpy,
        run_trackers={
            "maximum_spectral_cfl": 0.5,
            # Inside the config's mass budget: this test's subject is tracker
            # INHERITANCE, and the mass gate now really reads the inherited
            # maximum, so an over-budget fabricated value fails the run.
            "maximum_mass_fixer_log_offset": 1.0e-11,
        },
    )
    receipt = run(cfg, tmp_path / "resumed", restart=checkpoint)
    assert receipt["status"] == "pass"
    assert receipt["maximum_spectral_cfl"] == 0.5
    assert receipt["maximum_mass_fixer_log_offset"] >= 1.0e-11
    assert receipt["segment_maximum_spectral_cfl"] < 0.5
    assert receipt["run_trackers_from_restart"] == {
        "maximum_spectral_cfl": 0.5,
        "maximum_mass_fixer_log_offset": 1.0e-11,
    }


def test_cfl_failure_writes_a_self_hashed_failure_receipt(tmp_path):
    import dataclasses

    from woof.globe.spectral.config import load_config
    from woof.globe.spectral.receipt import check_receipt
    from woof.globe.spectral.runner import run

    source = (
        _shipped_configs()
        / "global_spectral_primitive_smoke.toml"
    )
    cfg = dataclasses.replace(load_config(source), maximum_cfl=1.0e-12)
    with pytest.raises(ValueError, match="spectral CFL"):
        run(cfg, tmp_path)

    receipt = check_receipt(tmp_path / "global-spectral-receipt.json")
    assert receipt["status"] == "error"
    assert receipt["error_type"] == "ValueError"
    assert "spectral CFL" in receipt["error_message"]
    assert receipt["completed_step"] == 0
    assert len(receipt["checkpoints"]) == 1
