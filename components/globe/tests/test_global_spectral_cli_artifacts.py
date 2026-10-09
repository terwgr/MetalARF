from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from woof.globe.spectral.checkpoint import write_checkpoint
from woof.globe.spectral.cli import main
from woof.globe.spectral.compression import (
    read_compressed_scalar,
    read_compressed_wind,
)
from woof.globe.spectral.config import load_config
from woof.globe.spectral.export import read_latlon_export
from woof.globe.spectral import runner as runner_module
from woof.globe.spectral.runner import build_model_and_state, build_transform, run
from woof.globe.spectral.vector import VorticityDivergenceOperator


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / str(_shipped_configs() / "global_spectral_williamson2.toml")


def test_cli_compression_and_export_doors(tmp_path, capsys):
    cfg = load_config(CONFIG)
    transform = build_transform(cfg)
    _model, state, _cold, _meta = build_model_and_state(cfg, transform)
    u, v = VorticityDivergenceOperator(transform).wind_from_vordiv(
        state.vorticity, state.divergence
    )
    geopotential = transform.backend.to_numpy(transform.inverse(state.geopotential))
    source = tmp_path / "source.npz"
    with source.open("wb") as stream:
        np.savez_compressed(
            stream,
            geopotential=geopotential,
            u=transform.backend.to_numpy(u),
            v=transform.backend.to_numpy(v),
        )
    scalar = tmp_path / "scalar.shc.npz"
    scalar_decoded = tmp_path / "scalar-decoded.npz"
    wind = tmp_path / "wind.shc.npz"
    wind_decoded = tmp_path / "wind-decoded.npz"
    assert main(
        [
            "compress-scalar",
            str(CONFIG),
            str(source),
            str(scalar),
            "--field",
            "geopotential",
            "--space",
            "log",
            "--floor",
            "1.0",
        ]
    ) == 0
    assert main(
        [
            "decompress-scalar",
            str(CONFIG),
            str(scalar),
            str(scalar_decoded),
            "--field",
            "geopotential",
        ]
    ) == 0
    assert main(
        ["compress-wind", str(CONFIG), str(source), str(wind)]
    ) == 0
    assert main(
        ["decompress-wind", str(CONFIG), str(wind), str(wind_decoded)]
    ) == 0
    assert main(["inspect-compressed", str(scalar)]) == 0

    # An exit code says the door opened, not that anything came through it: a
    # decoder writing zeros, the wrong key, or the wrong shape also returns 0.
    scalar_header = read_compressed_scalar(scalar).header
    with np.load(scalar_decoded) as archive:
        decoded_geopotential = np.array(archive["geopotential"])
    assert decoded_geopotential.shape == geopotential.shape
    scale = np.max(np.abs(geopotential))
    assert (
        np.max(np.abs(decoded_geopotential - geopotential)) / scale
        <= scalar_header["metrics"]["relative_linf"]["maximum"] * (1.0 + 1.0e-9)
    )
    assert scalar_header["metrics"]["relative_linf"]["maximum"] < 1.0e-5

    wind_header = read_compressed_wind(wind).header
    with np.load(wind_decoded) as archive:
        decoded_u = np.array(archive["u"])
        decoded_v = np.array(archive["v"])
    source_u = transform.backend.to_numpy(u)
    source_v = transform.backend.to_numpy(v)
    assert decoded_u.shape == source_u.shape
    speed_error = np.max(
        np.sqrt((decoded_u - source_u) ** 2 + (decoded_v - source_v) ** 2)
    )
    assert speed_error <= wind_header["metrics"]["maximum_vector_error_m_s"] * (
        1.0 + 1.0e-9
    )
    assert speed_error < 1.0e-6 * np.max(np.sqrt(source_u**2 + source_v**2))

    checkpoint = write_checkpoint(
        tmp_path / "state.npz",
        state,
        model=cfg.model,
        config_hash=cfg.config_hash,
        to_numpy=transform.backend.to_numpy,
    )
    export = tmp_path / "parent.npz"
    assert main(
        [
            "export-latlon",
            str(CONFIG),
            str(checkpoint),
            str(export),
            "--nlat",
            "18",
            "--nlon",
            "36",
        ]
    ) == 0
    metadata, arrays = read_latlon_export(export)
    assert metadata["model"] == "shallow-water"
    assert arrays["eastward_wind_m_s"].shape == (18, 36)
    assert main(["inspect-export", str(export)]) == 0
    capsys.readouterr()


def _short_williamson2_config(tmp_path):
    text = CONFIG.read_text(encoding="utf-8").replace(
        "duration_s = 432000.0       # five days", "duration_s = 6000.0"
    ).replace("output_interval_s = 86400.0", "output_interval_s = 6000.0")
    path = tmp_path / "williamson2-short.toml"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_williamson2_gates_measure_the_closed_form_solution(tmp_path, monkeypatch):
    # The steadiness gates compare the run against its own cold reference and
    # the wind arm puts wind_from_vordiv on both sides, so an error the
    # initial state and the final state share is invisible to them.  Offset
    # the initial geopotential by a constant: the flow stays geostrophically
    # balanced and steady, every steadiness gate stays at roundoff, and only
    # the closed-form gate sees it.
    cfg_path = _short_williamson2_config(tmp_path)
    cfg = load_config(cfg_path)
    baseline = run(cfg, tmp_path / "baseline")
    baseline_gates = {row["name"]: row for row in baseline["gates"]}
    assert baseline["status"] == "pass"
    assert (
        baseline_gates["williamson2_geopotential_analytic_normalized_l2"]["value"]
        < 1.0e-12
    )

    original = runner_module.williamson2_state

    def offset_state(transform, **kwargs):
        state = original(transform, **kwargs)
        return state.with_fields(
            (
                state.vorticity,
                state.divergence,
                transform.add_grid_constant(state.geopotential, 200.0),
            )
        )

    monkeypatch.setattr(runner_module, "williamson2_state", offset_state)
    offset = run(cfg, tmp_path / "offset")
    gates = {row["name"]: row for row in offset["gates"]}
    assert gates["williamson2_geopotential_steadiness_l2"]["passed"] is True
    assert gates["williamson2_wind_vector_steadiness_l2"]["passed"] is True
    assert gates["williamson2_vorticity_steadiness_l2"]["passed"] is True
    assert gates["williamson2_geopotential_analytic_normalized_l2"]["passed"] is False
    assert gates["williamson2_geopotential_analytic_normalized_l2"]["value"] > 1.0e-3
    assert offset["status"] == "fail"


def test_closed_form_reference_refuses_the_unadmitted_tilted_axis(tmp_path):
    cfg_path = _short_williamson2_config(tmp_path)
    cfg = load_config(cfg_path)
    tilted = dataclasses.replace(cfg, williamson_alpha_rad=0.5)
    transform = build_transform(cfg)
    with pytest.raises(ValueError, match="alpha_rad=0"):
        runner_module.williamson2_analytic_fields(tilted, transform.grid)


def test_diffusion_pressure_strength_reaches_the_shallow_water_integration(tmp_path):
    # The key is parsed, stored, and hashed into config_hash and the receipt.
    # If the model never reads it, two runs that differ only in this setting
    # carry different run identities and bit-identical output.
    states = {}
    for strength in (0.0, 7.5):
        text = (
            CONFIG.read_text(encoding="utf-8")
            .replace("enabled = false", "enabled = true")
            .replace("pressure_strength = 0.0", f"pressure_strength = {strength}")
        )
        path = tmp_path / f"diffusion-{strength}.toml"
        path.write_text(text, encoding="utf-8", newline="\n")
        cfg = load_config(path)
        assert cfg.pressure_diffusion_strength == strength
        transform = build_transform(cfg)
        model, state, _cold, _meta = build_model_and_state(cfg, transform)
        for _ in range(5):
            state = model.step(state, cfg.dt_s)
        states[strength] = transform.backend.to_numpy(state.geopotential)
    assert not np.array_equal(states[0.0], states[7.5])
