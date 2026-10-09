from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from pathlib import Path

import pytest

from woof.globe.spectral.cli import main
from woof.globe.spectral.config import load_config
from woof.globe.spectral.constants import RESEARCH_ACKNOWLEDGEMENT, RUN_SCHEMA
from woof.globe.spectral.receipt import check_receipt


ROOT = Path(__file__).resolve().parents[1]
W2 = ROOT / str(_shipped_configs() / "global_spectral_williamson2.toml")
PRIMITIVE = ROOT / str(_shipped_configs() / "global_spectral_primitive_smoke.toml")


def _base_config(*, model: str = "shallow-water", extra: str = "") -> str:
    model_table = (
        "[shallow_water]\nalpha_rad = 0.0\n"
        if model == "shallow-water"
        else "[primitive]\nnlev = 4\n"
    )
    return f'''[global_spectral]
schema = "{RUN_SCHEMA}"
name = "tiny"
model = "{model}"
acknowledgement = "{RESEARCH_ACKNOWLEDGEMENT}"
backend = "numpy"
precision = "float64"

[grid]
truncation = 5

[time]
dt_s = 300.0
duration_s = 600.0
output_interval_s = 600.0
integrator = "rk4"
maximum_cfl = 0.95

[diffusion]
enabled = false

{model_table}
{extra}
'''


def test_shipped_configs_load_and_are_model_specific():
    w2 = load_config(W2)
    primitive = load_config(PRIMITIVE)
    assert w2.model == "shallow-water"
    assert primitive.model == "primitive-dry"
    assert primitive.nlev == 4


def test_missing_research_acknowledgement_refuses(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(_base_config().replace(RESEARCH_ACKNOWLEDGEMENT, "no"))
    with pytest.raises(ValueError, match="research-only"):
        load_config(path)


def test_tilted_williamson_arm_is_fail_closed(tmp_path):
    path = tmp_path / "tilted.toml"
    path.write_text(_base_config().replace("alpha_rad = 0.0", "alpha_rad = 0.5"))
    with pytest.raises(ValueError, match="alpha_rad=0"):
        load_config(path)


def test_irrelevant_model_table_refuses_instead_of_being_ignored(tmp_path):
    path = tmp_path / "ignored.toml"
    path.write_text(_base_config(extra="[primitive]\nnlev = 4\n"))
    with pytest.raises(ValueError, match="not read"):
        load_config(path)


def test_cli_pins_transform_run_receipt_and_safe_overwrite(tmp_path, capsys):
    assert main(["pins"]) == 0
    assert main(["transform-check", "--truncation", "5"]) == 0
    config = tmp_path / "tiny.toml"
    config.write_text(_base_config())
    out = tmp_path / "out"
    assert main(["run", str(config), "--outdir", str(out)]) == 0
    receipt = out / "global-spectral-receipt.json"
    assert check_receipt(receipt)["status"] == "pass"
    assert main(["run", str(config), "--outdir", str(out)]) == 2
    assert main(
        ["run", str(config), "--outdir", str(out), "--overwrite"]
    ) == 0
    assert main(["inspect", str(out / "global_spectral_step00000002.npz")]) == 0
    assert main(["check-receipt", str(receipt)]) == 0
    capsys.readouterr()


def test_explicit_grid_must_honor_dealias_factor_and_even_fft_layout(tmp_path):
    path = tmp_path / "underresolved.toml"
    path.write_text(
        _base_config().replace(
            "truncation = 5",
            "truncation = 5\nnlat = 6\nnlon = 12\ndealias_factor = 1.5",
        )
    )
    with pytest.raises(ValueError, match="dealias_factor"):
        load_config(path)

    odd = tmp_path / "odd.toml"
    odd.write_text(
        _base_config().replace(
            "truncation = 5",
            "truncation = 5\nnlat = 9\nnlon = 19\ndealias_factor = 1.5",
        )
    )
    with pytest.raises(ValueError, match="must be even"):
        load_config(odd)


def test_integer_fields_do_not_silently_truncate_toml_floats(tmp_path):
    path = tmp_path / "float-truncation.toml"
    path.write_text(_base_config().replace("truncation = 5", "truncation = 5.9"))
    with pytest.raises(ValueError, match="TOML integer"):
        load_config(path)


def test_empty_irrelevant_table_and_malformed_sigma_refuse(tmp_path):
    irrelevant = tmp_path / "irrelevant.toml"
    irrelevant.write_text(_base_config(extra="[primitive]\n"))
    with pytest.raises(ValueError, match="not read"):
        load_config(irrelevant)

    malformed = tmp_path / "malformed.toml"
    malformed.write_text(
        _base_config(model="primitive-dry").replace("nlev = 4", "sigma_half = 1.0")
    )
    with pytest.raises(ValueError, match="TOML array"):
        load_config(malformed)


def test_benchmark_reports_first_call_and_warmed_steady_state(tmp_path, capsys):
    config = tmp_path / "tiny.toml"
    config.write_text(_base_config())
    assert main(
        ["benchmark", str(config), "--iterations", "2", "--warmup", "1"]
    ) == 0
    output = capsys.readouterr().out
    assert '"transform_first_seconds"' in output
    assert '"transform_roundtrip_steady_seconds"' in output
    assert '"rhs_first_seconds"' in output
    assert '"rhs_steady_seconds"' in output
