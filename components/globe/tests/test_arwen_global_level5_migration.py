from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from woof.globe.checkpoint import read_checkpoint
from woof.globe.cli import EXIT_REFUSED, main as cli_main
from woof.globe.config import load_config
from woof.globe.constants import LEVEL4_CHECKPOINT_SCHEMA, LEVEL4_SPECTRAL_FIELDS, NUMBER_MOMENTS
from woof.globe.migration import (
    LEVEL4_PINS_HASH,
    _SURFACE_MAP as LEVEL4_SURFACE_MAP,
    migrate_level4_checkpoint,
    read_level4_checkpoint,
    read_migration_receipt,
)
from woof.globe.runner import build_model_and_cold_state


CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
NATIVE_CONFIG = str(_shipped_configs() / "arwen_global_level5_native_smoke.toml")
OLD_TRACKERS = (
    "maximum_spectral_cfl",
    "maximum_mass_fixer_log_offset",
    "maximum_global_water_fixer_kg_m2",
    "maximum_repaired_negative_mixing_ratio",
    "maximum_semi_implicit_divergence_increment_s1",
    "maximum_physics_water_repair_kg_m2",
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _array_hash(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(str(value.shape).encode())
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _write_level4(path: Path, *, schema: str = LEVEL4_CHECKPOINT_SCHEMA):
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    transform = model.transform

    def level4_field(name):
        # Level 4 carried every water species as spectral coefficients;
        # the condensate species are grid tracers now, so the fixture
        # analyses them the way that era's model would have held them.
        value = getattr(state.atmosphere, name)
        if np.iscomplexobj(np.asarray(value)):
            return np.asarray(value)
        return np.asarray(transform.project(transform.forward(value)))

    arrays = {
        f"atmosphere__{name}": level4_field(name)
        for name in LEVEL4_SPECTRAL_FIELDS
    }
    # A Level-4 surface carried the twelve reservoir fields only; the
    # static fields the Level-5 state grew are what the migration seeds.
    arrays.update({
        f"surface__{name}": np.asarray(value)
        for name, value in state.surface.arrays().items()
        if name in set(LEVEL4_SURFACE_MAP.values())
    })
    metadata = {
        "schema": schema,
        "config_hash": "a" * 64,
        "pins_hash": LEVEL4_PINS_HASH,
        "time_s": 0.0,
        "step": 0,
        "run_trackers": {name: 0.0 for name in OLD_TRACKERS},
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": _array_hash(value),
            }
            for name, value in arrays.items()
        },
    }
    metadata["self_sha256"] = hashlib.sha256(_canonical(metadata)).hexdigest()
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            __metadata__=np.asarray(json.dumps(metadata, sort_keys=True)),
            **arrays,
        )


def test_level4_migration_is_explicit_hash_bound_and_zero_seeds_moments(tmp_path):
    source = tmp_path / "level4.npz"
    output = tmp_path / "level5.npz"
    _write_level4(source)
    old_meta, _ = read_level4_checkpoint(source)
    cfg = load_config(CONFIG)
    migrated, receipt = migrate_level4_checkpoint(
        source, output, target_config=cfg
    )
    metadata, arrays = read_checkpoint(migrated, expected_config_hash=cfg.config_hash)
    for name in NUMBER_MOMENTS:
        assert np.count_nonzero(arrays[f"atmosphere__{name}"]) == 0
    assert metadata["physics_metadata"]["migration"]["source_checkpoint_self_sha256"] == old_meta["self_sha256"]
    checked = read_migration_receipt(receipt)
    assert checked["status"] == "migrated"
    assert checked["output_checkpoint_self_sha256"] == metadata["self_sha256"]


def test_level4_native_migration_requires_zero_moment_acknowledgement(tmp_path):
    source = tmp_path / "level4.npz"
    _write_level4(source)
    with pytest.raises(ValueError, match="allow_native_zero_moments"):
        migrate_level4_checkpoint(
            source,
            tmp_path / "native.npz",
            target_config=load_config(NATIVE_CONFIG),
        )


def test_migration_identity_and_refusal_read_the_same_target_config(tmp_path):
    # The refusal and the hash the output is stamped with must come from one
    # object: a mode supplied beside an unrelated hash lets a caller stamp a
    # native run's identity onto a checkpoint the refusal never examined.
    source = tmp_path / "level4.npz"
    _write_level4(source)
    native = load_config(NATIVE_CONFIG)
    with pytest.raises(ValueError, match="allow_native_zero_moments"):
        migrate_level4_checkpoint(
            source, tmp_path / "native.npz", target_config=native
        )
    assert not (tmp_path / "native.npz").exists()
    cfg = load_config(CONFIG)
    _, receipt = migrate_level4_checkpoint(
        source, tmp_path / "reference.npz", target_config=cfg
    )
    row = read_migration_receipt(receipt)
    assert row["new_config_hash"] == cfg.config_hash
    assert row["target_physics_mode"] == cfg.physics_mode


def test_migration_refuses_a_checkpoint_from_another_geometry(tmp_path):
    source = tmp_path / "level4.npz"
    _write_level4(source)
    cfg = load_config(CONFIG)
    other = replace(cfg, truncation=cfg.truncation + 2)
    assert other.config_hash != cfg.config_hash
    with pytest.raises(ValueError, match="incompatible with the target config"):
        migrate_level4_checkpoint(
            source, tmp_path / "other.npz", target_config=other
        )
    assert not (tmp_path / "other.npz").exists()


def test_migration_receipt_records_the_target_geometry(tmp_path):
    source = tmp_path / "level4.npz"
    _write_level4(source)
    cfg = load_config(CONFIG)
    _, receipt = migrate_level4_checkpoint(
        source, tmp_path / "level5.npz", target_config=cfg
    )
    row = read_migration_receipt(receipt)
    assert row["target_truncation"] == cfg.truncation
    assert row["target_nlev"] == cfg.vertical.nlev
    assert row["target_surface_shape"] == [6, 12]


def test_level4_migration_detects_source_tampering(tmp_path):
    source = tmp_path / "level4.npz"
    _write_level4(source)
    with np.load(source, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    values["atmosphere__qv"].flat[0] += 1.0
    with source.open("wb") as stream:
        np.savez_compressed(stream, **values)
    with pytest.raises(ValueError, match="failed validation"):
        read_level4_checkpoint(source)


def test_level4_restart_refusal_names_the_migration_door(tmp_path, capsys):
    source = tmp_path / "level4.npz"
    _write_level4(source)
    rc = cli_main([
        "run", CONFIG, "--restart", str(source), "--outdir", str(tmp_path / "out"),
    ])
    captured = capsys.readouterr()
    assert rc == EXIT_REFUSED
    assert "migrate-level4-checkpoint" in captured.err


def test_level4_inspect_refusal_names_the_migration_door(tmp_path, capsys):
    source = tmp_path / "level4.npz"
    _write_level4(source)
    rc = cli_main(["inspect", str(source)])
    captured = capsys.readouterr()
    assert rc == EXIT_REFUSED
    assert "migrate-level4-checkpoint" in captured.err


def test_unknown_schema_checkpoint_keeps_the_inventory_corruption_refusal(tmp_path):
    source = tmp_path / "unknown.npz"
    _write_level4(source, schema="gpuwm.arwen-global-checkpoint/v0")
    with pytest.raises(ValueError, match="checkpoint metadata inventory mismatch"):
        read_checkpoint(source)
