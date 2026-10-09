from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from pathlib import Path

import numpy as np

from woof.globe.spectral.config import load_config
from woof.globe.spectral.export import export_checkpoint_latlon, read_latlon_export
from woof.globe.spectral.runner import build_model_and_state, build_transform
from woof.globe.spectral.checkpoint import write_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / str(_shipped_configs() / "global_spectral_primitive_smoke.toml")


def test_regular_latlon_export_is_hash_bound_and_complete(tmp_path):
    cfg = load_config(CONFIG)
    transform = build_transform(cfg)
    _model, state, _cold, _meta = build_model_and_state(cfg, transform)
    checkpoint = write_checkpoint(
        tmp_path / "state.npz",
        state,
        model=cfg.model,
        config_hash=cfg.config_hash,
        to_numpy=transform.backend.to_numpy,
    )
    export = export_checkpoint_latlon(
        cfg, transform, checkpoint, tmp_path / "parent.npz", nlat=24, nlon=48
    )
    metadata, arrays = read_latlon_export(export)
    assert metadata["target_grid"]["includes_poles"] is False
    assert arrays["temperature_k"].shape == (cfg.nlev, 24, 48)
    assert arrays["eastward_wind_m_s"].shape == (cfg.nlev, 24, 48)
    assert arrays["surface_pressure_pa"].shape == (24, 48)
    assert np.isfinite(arrays["geopotential_m2_s2"]).all()
