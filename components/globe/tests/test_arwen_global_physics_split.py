"""The physics split: Strang's two half calls, or one merged call per step."""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from woof.globe.checkpoint import read_checkpoint
from woof.globe.config import PHYSICS_SPLITS, load_config
from woof.globe.dynamics import MoistHybridModel
from woof.globe.runner import build_model_and_cold_state, run

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


def _smoke(steps: int = 4, **overrides):
    cfg = load_config(CONFIG)
    return replace(
        cfg, duration_s=cfg.dt_s * steps, output_interval_s=cfg.dt_s * steps,
        **overrides,
    )


def test_the_config_door_names_the_split_and_keeps_every_strang_hash(tmp_path):
    cfg = load_config(CONFIG)
    assert cfg.physics_split == "strang"
    assert "physics_split" not in cfg.config_identity
    merged = replace(cfg, physics_split="merged")
    assert merged.config_identity["physics_split"] == "merged"
    assert merged.config_hash != cfg.config_hash
    assert PHYSICS_SPLITS == ("strang", "merged")
    base_text = Path(CONFIG).read_text(encoding="utf-8")
    path = tmp_path / "merged.toml"
    path.write_text(base_text.replace('mode = "reference"', 'mode = "reference"\nsplit = "merged"'), encoding="utf-8")
    assert load_config(path).physics_split == "merged"
    path.write_text(base_text.replace('mode = "reference"', 'mode = "reference"\nsplit = "lie"'), encoding="utf-8")
    with pytest.raises(ValueError, match="physics.split must be one of"):
        load_config(path)
    with pytest.raises(ValueError, match="physics_split must be"):
        model, _ = build_model_and_cold_state(cfg)
        MoistHybridModel(
            transform=model.transform, vertical=model.vertical,
            surface_geopotential=np.zeros(model.transform.grid.shape),
            physics=None, physics_split="lie",
        )


def test_merged_calls_the_suite_once_per_step_over_the_full_step(monkeypatch):
    calls = {}
    for split in PHYSICS_SPLITS:
        cfg = _smoke(2, physics_split=split)
        model, state = build_model_and_cold_state(cfg)
        seen = []
        original = model.apply_physics

        def recording(bundle, dt_s, _original=original, _seen=seen):
            _seen.append(float(dt_s))
            return _original(bundle, dt_s)

        monkeypatch.setattr(model, "apply_physics", recording)
        state, metrics = model.step(state, cfg.dt_s)
        calls[split] = list(seen)
        assert "first_half_physics" in metrics and "second_half_physics" in metrics
        assert model.physics_split == split
    assert calls["strang"] == [cfg.dt_s / 2.0, cfg.dt_s / 2.0]
    assert calls["merged"] == [cfg.dt_s]


def test_merged_runs_restart_bit_exact_and_write_the_split_in_the_receipt(tmp_path):
    cfg = _smoke(4, physics_split="merged")
    cfg = replace(cfg, output_interval_s=cfg.dt_s * 2)
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    receipt = run(cfg, full)
    assert receipt["status"] == "pass"
    assert receipt["physics_split"] == "merged"
    assert receipt["config"]["physics_split"] == "merged"
    run(cfg, resumed, restart=full / "arwen_global_step00000002.npz")
    meta_a, arrays_a = read_checkpoint(full / "arwen_global_step00000004.npz")
    meta_b, arrays_b = read_checkpoint(resumed / "arwen_global_step00000004.npz")
    assert meta_a["run_trackers"] == meta_b["run_trackers"]
    for name in arrays_a:
        assert np.array_equal(arrays_a[name], arrays_b[name]), name
    # The two splits are different integrations of the same case.
    strang = run(replace(cfg, physics_split="strang"), tmp_path / "strang")
    assert strang["physics_split"] == "strang"
    meta_s, arrays_s = read_checkpoint(tmp_path / "strang" / "arwen_global_step00000004.npz")
    assert meta_s["config_hash"] != meta_a["config_hash"]
    assert not np.array_equal(arrays_s["atmosphere__theta"], arrays_a["atmosphere__theta"])
