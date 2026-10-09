"""The step profiler: calibrated before it is read, bit-neutral when attached."""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from dataclasses import replace
import json
import time

import numpy as np
import pytest

from woof.globe.checkpoint import read_checkpoint
from woof.globe.config import load_config
from woof.globe.physics.native_suite import ArwenCudaColumnSuite
from woof.globe.profile import (
    NULL_PROFILER,
    PROFILE_NAME,
    StepProfiler,
    attach_profiler,
    calibrate,
    profiler_of,
)
from woof.globe.runner import build_model_and_cold_state, run
from woof.globe.spectral.backend import get_backend

from test_arwen_global_level5_native import _fake_modules, _options

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


def _smoke(steps: int = 6, **overrides):
    cfg = load_config(CONFIG)
    return replace(
        cfg, duration_s=cfg.dt_s * steps, output_interval_s=cfg.dt_s * steps,
        **overrides,
    )


# -- calibration ----------------------------------------------------------

def test_calibration_reads_a_planted_sleep_and_nothing_else():
    """The instrument's own proof on the CPU backend: a 100 ms sleep reads
    100 ms in both columns, the next section reads none of it, a known
    device workload reads its benchmarked cost, nested sections sum."""
    receipt = calibrate(get_backend("numpy", "float64"), sleep_s=0.1)
    assert receipt["passed"] is True
    checks = receipt["checks"]
    assert checks["sleep_host_ms"]["read_ms"] == pytest.approx(100.0, rel=0.1)
    assert checks["sleep_device_ms"]["read_ms"] == pytest.approx(100.0, rel=0.1)
    assert checks["no_bleed_host_ms"]["read_ms"] < 5.0
    assert checks["nesting"]["passed"]
    # The device-work checks gate only on cupy; numpy records them.
    assert checks["device_fma"]["gated"] is False
    assert checks["device_reduce"]["read_ms"] > 0.0
    # Every check ran its rounds with the reference interleaved, and the
    # cited reading is the best round; every round rides in the receipt.
    assert receipt["rounds"] == 5
    assert len(checks["sleep_host_ms"]["rounds_ms"]) == 5
    assert checks["sleep_host_ms"]["read_ms"] == min(checks["sleep_host_ms"]["rounds_ms"])
    assert len(checks["device_fma"]["reference_rounds_ms"]) == 5
    assert checks["device_fma"]["spread"] >= 1.0
    assert checks["nesting"]["gap_ms"] == min(checks["nesting"]["rounds_gap_ms"])
    assert receipt["device_memory"] is None


def test_calibration_refuses_when_the_clock_lies(monkeypatch):
    """A clock that does not advance fails the sleep check by name."""
    import woof.globe.profile as module

    frozen = time.perf_counter()
    monkeypatch.setattr(module.time, "perf_counter", lambda: frozen)
    with pytest.raises(ValueError, match="calibration failed.*sleep_host_ms"):
        calibrate(get_backend("numpy", "float64"), sleep_s=0.05)


def test_sections_are_attributed_to_their_own_path_and_summed_per_step():
    backend = get_backend("numpy", "float64")
    profiler = StepProfiler(backend, steps=2, warmup=1)
    # Warm-up step: nothing is recorded.
    profiler.begin_step(0)
    with profiler.section("a"):
        time.sleep(0.01)
    profiler.end_step()
    assert profiler.profiled_steps == 0
    for step in (1, 2):
        profiler.begin_step(step)
        with profiler.section("a"):
            time.sleep(0.02)
            with profiler.section("inner"):
                time.sleep(0.01)
        with profiler.section("b"):
            time.sleep(0.005)
        with profiler.section("b"):
            time.sleep(0.005)
        profiler.end_step()
    assert profiler.profiled_steps == 2
    rows = {row["section"]: row for row in profiler.summary()}
    assert rows["step.a"]["host_ms"] == pytest.approx(30.0, abs=8.0)
    assert rows["step.a.inner"]["host_ms"] == pytest.approx(10.0, abs=5.0)
    # Two calls of the same section in one step fold into one row.
    assert rows["step.b"]["host_ms"] == pytest.approx(10.0, abs=5.0)
    assert rows["step"]["host_ms"] >= rows["step.a"]["host_ms"] + rows["step.b"]["host_ms"]
    # Past the window the profiler is inert.
    profiler.begin_step(3)
    assert profiler.active is False
    receipt = profiler.receipt()
    assert receipt["profiled_steps"] == 2
    assert "device_ms" in receipt["measures"]
    assert receipt["backend"] == "numpy"


def test_out_of_order_close_is_refused():
    profiler = StepProfiler(get_backend("numpy", "float64"), steps=1, warmup=0)
    profiler.begin_step(0)
    profiler._push("outer")
    with pytest.raises(RuntimeError, match="closed out of order"):
        profiler._pop("inner")


def test_the_null_profiler_is_what_an_unattached_model_sees():
    cfg = _smoke(2)
    model, _ = build_model_and_cold_state(cfg)
    assert profiler_of(model) is NULL_PROFILER
    assert profiler_of(model.transport) is NULL_PROFILER
    with NULL_PROFILER.section("anything"):
        pass
    assert NULL_PROFILER.active is False


# -- the run door -----------------------------------------------------------

def test_a_profiled_run_is_bit_identical_and_writes_the_profile(tmp_path):
    cfg = _smoke(6)
    plain = run(cfg, tmp_path / "plain")
    profiled = run(cfg, tmp_path / "profiled", profile_steps=3, profile_warmup=1)
    assert plain["status"] == "pass" and profiled["status"] == "pass"
    hashes = {}
    for name in ("plain", "profiled"):
        metadata, _ = read_checkpoint(tmp_path / name / "arwen_global_step00000006.npz")
        hashes[name] = {key: value["sha256"] for key, value in metadata["arrays"].items()}
    assert hashes["plain"] == hashes["profiled"]
    assert not (tmp_path / "plain" / PROFILE_NAME).exists()
    profile = json.loads((tmp_path / "profiled" / PROFILE_NAME).read_text(encoding="utf-8"))
    assert profile["profiled_steps"] == 3
    assert profile["warmup_steps"] == 1
    assert profile["calibration"]["passed"] is True
    sections = {row["section"] for row in profile["summary"]}
    for expected in (
        "step", "step.physics_first", "step.physics_second",
        "step.positivity_first", "step.dynamics", "step.diffusion",
        "step.mass_fixer", "step.transport", "step.transport.advance",
        "step.transport.advance.sweep_x", "step.transport.advance.floor",
        "step.water_fixer", "step.enforce", "step.observer",
        "step.dynamics.rhs", "step.dynamics.rhs.grid_state",
        "step.dynamics.rhs.momentum", "step.dynamics.rhs.scalars",
        "step.dynamics.rhs.linear",
    ):
        assert expected in sections, expected
    # The smoke config integrates with ssprk3 (the split), so no IMEX stage
    # sections and no implicit solve section.
    assert not any("stage" in s for s in sections)
    assert [int(row["step"]) for row in profile["steps"]] == [2, 3, 4]
    # Every reading is a finite nonnegative number.
    for row in profile["summary"]:
        assert np.isfinite(row["host_ms"]) and row["host_ms"] >= 0.0
        assert np.isfinite(row["device_ms"]) and row["device_ms"] >= 0.0


def test_an_imex_run_profiles_its_stage_solves(tmp_path):
    cfg = _smoke(4, integrator="imex_ssp3", semi_implicit_scheme="vertical_modes")
    result = run(cfg, tmp_path / "imex", profile_steps=2, profile_warmup=1)
    assert result["status"] == "pass"
    profile = json.loads((tmp_path / "imex" / PROFILE_NAME).read_text(encoding="utf-8"))
    sections = {row["section"] for row in profile["summary"]}
    assert "step.dynamics.stage1_solve" in sections
    assert "step.dynamics.stage2_solve" in sections
    assert "step.dynamics.stage0_linear" in sections
    assert "step.dynamics.stage0_rhs.rhs.grid_state" in sections


def test_the_profiler_refuses_a_window_of_no_steps():
    with pytest.raises(ValueError, match="profile steps"):
        StepProfiler(get_backend("numpy", "float64"), steps=0)


# -- the native suite ------------------------------------------------------

def test_native_runtime_sections_name_every_scheme():
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    exchange = model._physics_exchange(state, 5.0)
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=_fake_modules())
    profiler = StepProfiler(model.transform.backend, steps=1, warmup=0)
    suite.profiler = profiler
    profiler.begin_step(0)
    with profiler.section("suite"):
        suite.step(exchange)
    profiler.end_step()
    sections = {row["section"] for row in profiler.summary()}
    for name in (
        "step.suite.batch_in", "step.suite.ledger_before", "step.suite.rrtmgp",
        "step.suite.sfclay", "step.suite.noah", "step.suite.ysu",
        "step.suite.cumulus", "step.suite.morrison", "step.suite.validate",
        "step.suite.ledger_after", "step.suite.batch_out",
    ):
        assert name in sections, name
    assert attach_profiler(model, None) == ["MoistHybridModel", "GridTracerTransport"]


# -- the synthesis memo ------------------------------------------------------

def _run_hashes(cfg, path):
    result = run(cfg, path)
    assert result["status"] == "pass"
    step = int(round(cfg.duration_s / cfg.dt_s))
    metadata, _ = read_checkpoint(path / f"arwen_global_step{step:08d}.npz")
    return {key: value["sha256"] for key, value in metadata["arrays"].items()}, result


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"integrator": "imex_ssp3", "semi_implicit_scheme": "vertical_modes"},
    ],
    ids=["ssprk3", "imex"],
)
def test_the_synthesis_memo_is_bit_neutral(tmp_path, monkeypatch, overrides):
    """The memo hands back the arrays a previous call computed from the
    same spectral arrays, so a run with it is the run without it, bit for
    bit, under both integrators; and it is actually consulted."""
    import woof.globe.dynamics as dynamics

    cfg = _smoke(6, **overrides)
    with_memo, _ = _run_hashes(cfg, tmp_path / "memo")
    # The same run with every synthesis recomputed.
    original = dynamics.MoistHybridModel.__post_init__

    def without(self):
        self.synthesis_memo = False
        original(self)

    monkeypatch.setattr(dynamics.MoistHybridModel, "__post_init__", without)
    without_memo, _ = _run_hashes(cfg, tmp_path / "plain")
    assert with_memo == without_memo
    assert len(with_memo) > 10


def test_the_synthesis_memo_serves_repeats_and_misses_new_arrays():
    cfg = _smoke(2)
    model, state = build_model_and_cold_state(cfg)
    memo = model._memo
    assert memo is not None
    model.release_syntheses()
    assert len(memo) == 0
    hits, misses = memo.hits, memo.misses
    u1, v1 = model._wind(state.atmosphere)
    u2, v2 = model._wind(state.atmosphere)
    assert u1 is u2 and v1 is v2
    assert (memo.hits - hits, memo.misses - misses) == (1, 1)
    g1 = model.grid_state(state.atmosphere, only=("qv", "dp", "temperature"))
    g2 = model.grid_state(state.atmosphere, only=("qv", "dp", "temperature"))
    # Stack rows are views of the one cached stack; the derived fields
    # are the cached objects themselves.
    assert np.shares_memory(g1["qv"], g2["qv"]) and g1["dp"] is g2["dp"]
    assert g1["temperature"] is g2["temperature"]
    # A state whose vapor was replaced misses on the stack and on nothing
    # else that did not read the vapor.
    fields = list(state.atmosphere.fields())
    fields[4] = fields[4] * 1.0
    changed = state.atmosphere.with_fields(fields)
    g3 = model.grid_state(changed, only=("qv", "dp", "temperature"))
    assert not np.shares_memory(g3["qv"], g1["qv"])
    assert g3["dp"] is g1["dp"]
    assert g3["temperature"] is g1["temperature"]
    u3, _ = model._wind(changed)
    assert u3 is u1
    model.release_syntheses()
    assert len(memo) == 0
    u4, _ = model._wind(state.atmosphere)
    assert u4 is not u1
    assert np.array_equal(np.asarray(u4), np.asarray(u1))
