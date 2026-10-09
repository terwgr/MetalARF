from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from woof.globe.checkpoint import normalize_trackers, read_checkpoint, write_checkpoint
from woof.globe.config import load_config
from woof.globe.dynamics import MoistHybridModel
from woof.globe.physics import ReferencePhysics
from woof.globe.receipt import check_receipt
from woof.globe.runner import _CheckpointWriter, build_model_and_cold_state, run


CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")

#: The external (barotropic proxy) scheme's pin under the grid tracers
#: (provenance in tests/test_arwen_global_pins.py).  The proxy's pin
#: STRING is the one every archive carried before the vertical-mode
#: scheme existed; its digest moved with the grid tracers (2026-09-02),
#: whose transport of the condensate is not the arithmetic those
#: archives were advanced with.
#: RE-PINNED for WOOF 1.0.1: the v3 pin document names WOOF where v2
#: named the engine's earlier name, and no arithmetic moved; the
#: _WOOF_1_0_0 digest below is the v2 pin 1.0.0 wrote, its legacy alias.
EXTERNAL_SCHEME_PINS_HASH = (
    "592f66ce34c981db3073ae62179fd302e419e6085814416c959e47025a9a712f"
)
EXTERNAL_SCHEME_PINS_HASH_WOOF_1_0_0 = (
    "f536e10061499732a08bd6e32cb45160820bb55519df5c0721be4c33fbf573a0"
)


def _rewrite_npz(path: Path, *, mutate):
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    mutate(values)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **values)


def _restamp_pins(path: Path, pins_digest: str) -> None:
    """Rewrite a checkpoint's pins_hash and re-sign its metadata, so it
    reads as an intact archive written by a build carrying that pin."""

    def mutate(values):
        metadata = json.loads(str(values["__metadata__"].item()))
        metadata.pop("self_sha256")
        metadata["pins_hash"] = pins_digest
        canonical = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        metadata["self_sha256"] = hashlib.sha256(canonical).hexdigest()
        values["__metadata__"] = np.asarray(
            json.dumps(metadata, sort_keys=True, allow_nan=False)
        )

    _rewrite_npz(path, mutate=mutate)


def test_checkpoint_roundtrip_and_array_tamper(tmp_path):
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    path = tmp_path / "state.npz"
    write_checkpoint(
        path,
        state,
        config_hash=cfg.config_hash,
        to_numpy=model.transform.backend.to_numpy,
    )
    metadata, arrays = read_checkpoint(path, expected_config_hash=cfg.config_hash)
    assert metadata["step"] == 0
    assert "atmosphere__qv" in arrays
    assert "surface__soil_temperature_k" in arrays

    _rewrite_npz(
        path,
        mutate=lambda values: values["atmosphere__qv"].__setitem__(
            (0, 0, 0), values["atmosphere__qv"][0, 0, 0] + 1.0e-6
        ),
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        read_checkpoint(path)


def test_the_off_thread_writer_publishes_the_synchronous_archive(tmp_path):
    """The runner's worker-thread checkpoint carries the metadata and the
    arrays a synchronous write_checkpoint carries for the same state (the
    zip container's own timestamp is the one byte that may differ), and a
    failed write surfaces on the model thread at the next join."""
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    trackers = normalize_trackers()
    trackers["maximum_spectral_cfl"] = 0.25
    sync = write_checkpoint(
        tmp_path / "sync.npz", state, config_hash=cfg.config_hash,
        to_numpy=model.transform.backend.to_numpy, trackers=trackers,
        semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator,
    )
    writer = _CheckpointWriter(cfg, model.transform.backend.to_numpy)
    off = writer.submit(tmp_path / "off.npz", state, trackers)
    writer.close()
    meta_sync, arrays_sync = read_checkpoint(sync, expected_config_hash=cfg.config_hash)
    meta_off, arrays_off = read_checkpoint(off, expected_config_hash=cfg.config_hash)
    assert meta_sync == meta_off
    assert arrays_sync.keys() == arrays_off.keys()
    for name in arrays_sync:
        assert np.array_equal(arrays_sync[name], arrays_off[name]), name
    # A write that cannot land raises where the model thread joins it.
    (tmp_path / "blocker").write_text("not a directory", encoding="utf-8")
    failing = _CheckpointWriter(cfg, model.transform.backend.to_numpy)
    failing.submit(tmp_path / "blocker" / "x.npz", state, trackers)
    with pytest.raises(OSError):
        failing.close()


def test_runner_passes_and_receipt_tamper_is_detected(tmp_path):
    cfg = load_config(CONFIG)
    result = run(cfg, tmp_path / "run")
    assert result["status"] == "pass"
    # The fixer-absorption gates must exist alongside the drift gates: the
    # drift gates compare post-fix means against the targets the fixers
    # reset, so only these rows can fail on a conservation leak.
    for row in (
        "mass_fixer_max_step_log_offset",
        "water_fixer_max_step_relative",
        "physics_water_repair_max_step_kg_m2",
        "positivity_fixer_max_step_relative",
    ):
        assert result["gates"][row]["passed"]
    assert (
        result["supplementary_trackers"][
            "maximum_repaired_negative_number_per_kg"
        ]
        >= 0.0
    )
    # The vapor fixer is a measured quantity of the run: its per-step
    # magnitude reaches the receipt in kg/m2, relative to the atmospheric
    # column, and as the largest fraction any column paid of its own
    # vapor.  On this smoke the vapor field does not ring below zero, so
    # every figure is an exact zero (the condensate, which used to ring,
    # is grid-point now and never enters the fixer).
    supplementary = result["supplementary_trackers"]
    for key in (
        "maximum_positivity_fixer_water_kg_m2",
        "maximum_positivity_fixer_relative",
        "maximum_positivity_fixer_rescale",
    ):
        assert math.isfinite(supplementary[key]) and supplementary[key] >= 0.0
    assert (
        result["gates"]["positivity_fixer_max_step_relative"]["value"]
        == supplementary["maximum_positivity_fixer_relative"]
    )
    # The receipt states the semi-implicit scheme the run integrated with
    # and, under the vertical-mode default, the reference operator's modes.
    semi = result["semi_implicit"]
    assert semi["scheme"] == "vertical_modes"
    assert semi["reference_temperature_k"] == 320.0
    assert semi["off_centring_weight"] == 0.5
    assert len(semi["phase_speeds_m_s"]) == cfg.vertical.nlev
    assert semi["phase_speeds_m_s"][0] > semi["phase_speeds_m_s"][-1] > 0.0
    receipt = Path(result["receipt_path"])
    checked = check_receipt(receipt)
    assert checked["status"] == "pass"

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["status"] = "fail"
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash"):
        check_receipt(receipt)


def test_restart_is_bit_exact_and_inherits_trackers(tmp_path):
    cfg = load_config(CONFIG)
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    run(cfg, full)
    midpoint = full / "arwen_global_step00000002.npz"
    run(cfg, resumed, restart=midpoint)

    meta_a, arrays_a = read_checkpoint(full / "arwen_global_step00000004.npz")
    meta_b, arrays_b = read_checkpoint(resumed / "arwen_global_step00000004.npz")
    assert meta_a["run_trackers"] == meta_b["run_trackers"]
    assert arrays_a.keys() == arrays_b.keys()
    for name in arrays_a:
        assert np.array_equal(arrays_a[name], arrays_b[name]), name


def test_an_external_scheme_archive_resumes_under_the_external_scheme(tmp_path):
    # The external scheme carries its own pin document (the proxy's pin
    # string), distinct from the default's, and the restart door resumes
    # an archive written under it to the same bytes; a build-wide pin
    # refused every such archive while claiming to keep them readable.
    external = replace(load_config(CONFIG), semi_implicit_scheme="external")
    full = tmp_path / "full"
    run(external, full)
    midpoint = full / "arwen_global_step00000002.npz"
    written, _ = read_checkpoint(midpoint)
    assert written["pins_hash"] == EXTERNAL_SCHEME_PINS_HASH

    resumed = tmp_path / "resumed"
    result = run(external, resumed, restart=midpoint)
    assert result["status"] == "pass"
    meta_a, arrays_a = read_checkpoint(
        full / "arwen_global_step00000004.npz", semi_implicit_scheme="external", integrator="ssprk3"
    )
    meta_b, arrays_b = read_checkpoint(
        resumed / "arwen_global_step00000004.npz", semi_implicit_scheme="external", integrator="ssprk3"
    )
    assert meta_a["pins_hash"] == meta_b["pins_hash"] == EXTERNAL_SCHEME_PINS_HASH
    assert meta_a["run_trackers"] == meta_b["run_trackers"]
    assert arrays_a.keys() == arrays_b.keys()
    for name in arrays_a:
        assert np.array_equal(arrays_a[name], arrays_b[name]), name
    # The run's receipt carries the era's pin and document and checks green.
    checked = check_receipt(full / "arwen-global-receipt.json")
    assert checked["pins_hash"] == EXTERNAL_SCHEME_PINS_HASH
    assert checked["pins"]["sha256"] == EXTERNAL_SCHEME_PINS_HASH
    assert checked["semi_implicit"]["scheme"] == "external"


def test_a_checkpoint_never_resumes_under_the_other_scheme(tmp_path):
    default = load_config(CONFIG)
    external = replace(default, semi_implicit_scheme="external")
    run(external, tmp_path / "external")
    run(default, tmp_path / "modes")
    external_midpoint = tmp_path / "external" / "arwen_global_step00000002.npz"
    modes_midpoint = tmp_path / "modes" / "arwen_global_step00000002.npz"
    external_pin = read_checkpoint(external_midpoint)[0]["pins_hash"]
    modes_pin = read_checkpoint(modes_midpoint)[0]["pins_hash"]
    assert external_pin != modes_pin

    # The two arithmetics advance the state differently: resuming one
    # scheme's archive under the other is refused by name, before the
    # config identity is even compared.
    with pytest.raises(
        ValueError, match="arithmetic pins mismatch.*'external'.*'vertical_modes'"
    ):
        run(default, tmp_path / "cross-a", restart=external_midpoint)
    with pytest.raises(
        ValueError, match="arithmetic pins mismatch.*'vertical_modes'.*'external'"
    ):
        run(external, tmp_path / "cross-b", restart=modes_midpoint)

    # A reader without a scheme (inspection) admits both eras; a pin no
    # shipped scheme carries is refused with or without one.
    _restamp_pins(modes_midpoint, "f" * 64)
    with pytest.raises(ValueError, match="arithmetic pins mismatch.*not the pin of any"):
        read_checkpoint(modes_midpoint)
    with pytest.raises(ValueError, match="arithmetic pins mismatch.*no shipped scheme"):
        read_checkpoint(modes_midpoint, semi_implicit_scheme="vertical_modes", integrator="ssprk3")


def test_restart_with_overwrite_in_place_survives_and_keeps_source(tmp_path):
    cfg = load_config(CONFIG)
    cycle = tmp_path / "cycle"
    run(cfg, cycle)
    midpoint = cycle / "arwen_global_step00000002.npz"
    assert midpoint.exists()

    result = run(cfg, cycle, restart=midpoint, overwrite=True)

    assert result["status"] == "pass"
    # The overwrite sweep owns every checkpoint in the directory; the
    # restart source must survive it, both to be readable and so the
    # receipt's restart row keeps pointing at an existing file.
    assert midpoint.exists()
    assert result["restart"]["path"] == str(midpoint)

    full = tmp_path / "full"
    run(cfg, full)
    meta_a, arrays_a = read_checkpoint(full / "arwen_global_step00000004.npz")
    meta_b, arrays_b = read_checkpoint(cycle / "arwen_global_step00000004.npz")
    assert meta_a["run_trackers"] == meta_b["run_trackers"]
    assert arrays_a.keys() == arrays_b.keys()
    for name in arrays_a:
        assert np.array_equal(arrays_a[name], arrays_b[name]), name


def test_receipt_fails_when_the_water_fixer_absorbs_a_physics_leak(
    tmp_path, monkeypatch
):
    cfg = load_config(CONFIG)
    original = ReferencePhysics.step

    def deletes_water(self, exchange):
        result = original(self, exchange)
        result.qv = result.qv * 0.99
        return result

    monkeypatch.setattr(ReferencePhysics, "step", deletes_water)
    result = run(cfg, tmp_path / "leak")

    # The default-on water fixer holds the post-fix mean at its target, so
    # the drift gate stays green while ~0.12 kg/m2 of water per step is
    # manufactured; only the absorption gate can name the leak.
    assert result["gates"]["total_water_relative_drift"]["passed"]
    assert not result["gates"]["water_fixer_max_step_relative"]["passed"]
    assert result["status"] == "fail"


def test_receipt_fails_when_the_positivity_fixer_carries_a_sign_defect(
    tmp_path, monkeypatch
):
    cfg = load_config(CONFIG)
    original = ReferencePhysics.step

    def flips_a_quarter_of_the_vapor(self, exchange):
        result = original(self, exchange)
        quarter = result.qv.shape[-1] // 4
        result.qv[..., :quarter] = -result.qv[..., :quarter]
        return result

    monkeypatch.setattr(ReferencePhysics, "step", flips_a_quarter_of_the_vapor)
    result = run(cfg, tmp_path / "sign")

    # A defect that drives whole columns negative every step leaves those
    # columns with no positive vapor to fill their holes from: the clip
    # creates water there and the column-local fixer counts it as
    # created (unfillable) while the drift gate stays green, because the
    # reservoir never moves for it.  The fixer's own magnitude gate names
    # it directly (a quarter of the planet's vapor per step against the
    # 5e-3 limit, where the healthy smoke runs at exactly zero).
    assert result["gates"]["total_water_relative_drift"]["passed"]
    assert not result["gates"]["positivity_fixer_max_step_relative"]["passed"]
    assert result["gates"]["positivity_fixer_max_step_relative"]["value"] > 0.1
    assert result["status"] == "fail"


def test_receipt_fails_when_the_mass_fixer_absorbs_a_pressure_sink(
    tmp_path, monkeypatch
):
    cfg = load_config(CONFIG)
    original = MoistHybridModel._apply_diffusion

    def pressure_sink(self, state, dt_s):
        out = original(self, state, dt_s)
        fields = list(out.fields())
        fields[3] = self.transform.add_grid_constant(
            fields[3], math.log(0.995)
        )
        return out.with_fields(fields)

    monkeypatch.setattr(MoistHybridModel, "_apply_diffusion", pressure_sink)
    result = run(cfg, tmp_path / "sink")

    assert result["gates"]["mass_relative_drift"]["passed"]
    assert not result["gates"]["mass_fixer_max_step_log_offset"]["passed"]
    assert result["status"] == "fail"


def test_receipt_drift_gates_measure_the_model_with_fixers_off(tmp_path):
    cfg = replace(load_config(CONFIG), mass_fixer=False, water_fixer=False)
    result = run(cfg, tmp_path / "free")

    # With the fixers off the drift gates measure free dynamics against the
    # configured limits (free 4-step drift measured 1.3e-13 relative mass,
    # 1.3e-11 relative water on this config), so a dycore destroying mass or
    # water fails the receipt instead of being absorbed.
    assert result["status"] == "pass"
    assert result["run_trackers"]["maximum_mass_fixer_log_offset"] == 0.0
    assert result["run_trackers"]["maximum_global_water_fixer_kg_m2"] == 0.0
    assert result["gates"]["mass_relative_drift"]["passed"]
    assert result["gates"]["total_water_relative_drift"]["passed"]


def test_runtime_failure_writes_error_receipt_without_turning_green(tmp_path):
    cfg = replace(load_config(CONFIG), maximum_cfl=1.0e-12)
    out = tmp_path / "failure"
    with pytest.raises(ValueError, match="spectral CFL"):
        run(cfg, out)
    receipt = check_receipt(out / "arwen-global-receipt.json")
    assert receipt["status"] == "error"
    assert receipt["completed_step"] == 0
    assert receipt["error_type"] == "ValueError"


def test_a_woof_1_0_0_checkpoint_resumes_under_its_v2_pin(tmp_path):
    """A checkpoint WOOF 1.0.0 wrote carries the v2 pin of its arithmetic.
    1.0.1 reworded the pin document and moved no arithmetic, so the restart
    door resumes it to the same bytes an uninterrupted run reaches, writes
    v3 from there, and still refuses it under the other scheme."""

    from woof.globe import pins

    cfg = load_config(CONFIG)
    full = tmp_path / "full"
    run(cfg, full)
    midpoint = full / "arwen_global_step00000002.npz"
    label = pins.arithmetic_label(cfg.semi_implicit_scheme, cfg.integrator)
    _restamp_pins(midpoint, pins.WOOF_1_0_0_PINS_HASH_BY_ARITHMETIC[label])

    resumed = tmp_path / "resumed"
    assert run(cfg, resumed, restart=midpoint)["status"] == "pass"
    meta_a, arrays_a = read_checkpoint(full / "arwen_global_step00000004.npz")
    meta_b, arrays_b = read_checkpoint(resumed / "arwen_global_step00000004.npz")
    assert meta_b["pins_hash"] == pins.pins_hash(cfg.semi_implicit_scheme, cfg.integrator)
    assert meta_a["run_trackers"] == meta_b["run_trackers"]
    assert arrays_a.keys() == arrays_b.keys()
    for name in arrays_a:
        assert np.array_equal(arrays_a[name], arrays_b[name]), name
    other = replace(cfg, semi_implicit_scheme="external")
    with pytest.raises(ValueError, match="arithmetic pins mismatch"):
        run(other, tmp_path / "cross", restart=midpoint)
