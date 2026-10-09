"""The semi-Lagrangian core against the pinned host tier.

Two defaults met here for the first time on 2026-09-07 and did not run:
``sl_si`` became the shipped core, and ``host_spill = "auto"`` parks the
ten grid tracers whenever the card is tight.  MEASURED, RTX 5090, T533
L40 native, a bare ``woof global run`` with nothing on the command line
but the config and an output directory: the sizer chose one latitude band
with physics, surface and tracers in the tier, the run wrote its step-0
checkpoint and died in the first step with

    TypeError: Argument 'a' has incorrect type
               (expected cupy._core.core._ndarray_base, got SpilledArray)

because the core read the tracers off the state and handed them to the
gather, and a parked slot is not an operand.  The tier's contract is that
it raises rather than running the step on the host, so this was the
contract working; what was missing was the core's side of it.

The three reads are the gather (the sweep itself), the mass fixer's
"before" state, and the water reading beside it; a fourth is the physics
increment under the two non-default couplings.  Each goes through
``spill.resident``, which is the identity for an array already on the
card, so a run with nothing parked is the run it always was.

The tests below are the gate: every one fails on the tree that shipped
the merge.  The BIT rows are byte-identical to the resident run because
staging changes a place and not an arithmetic.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import hashlib
from dataclasses import replace

import numpy as np
import pytest

from woof.globe.config import load_config
from woof.globe.runner import build_model_and_cold_state
from woof.globe.sizing import SPILL_SLICES
from woof.globe.spill import HostTier, spilled

CONFIG = str(_shipped_configs() / "arwen_global_t21_baroclinic_ten_day.toml")


def _digest(host, value) -> str:
    array = np.asarray(host(value))
    return hashlib.sha256(
        array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()
    ).hexdigest()


def _inventory(model, bundle) -> dict[str, str]:
    host = model.transform.backend.to_numpy
    out = {}
    for name in ("vorticity", "divergence", "theta",
                 "log_surface_pressure", "qv"):
        out["atmosphere." + name] = _digest(host, getattr(bundle.atmosphere, name))
    for name, value in bundle.atmosphere.grid_tracers().items():
        out["tracer." + name] = _digest(host, value)
    for name, value in bundle.surface.arrays().items():
        out["surface." + name] = _digest(host, value)
    for name, value in sorted(bundle.physics_state.arrays.items()):
        out["physics." + name] = _digest(host, value)
    return out


def _scalars(metrics, prefix=""):
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (bool, str)):
            out[prefix + key] = value
        elif isinstance(value, (int, float)):
            out[prefix + key] = float(value)
        elif isinstance(value, dict):
            out.update(_scalars(value, prefix + key + "."))
    return out


def _stepped(slices, *, bands=1, steps=3, dry=False):
    """The shipped step on the semi-Lagrangian core, tier optional."""
    cfg = replace(
        load_config(CONFIG), integrator="sl_si",
        host_spill="off", latitude_bands=bands,
    )
    model, bundle = build_model_and_cold_state(cfg)
    if dry:
        model.physics = None
    if slices:
        model.host_tier = HostTier(model.transform.backend.xp)
        model.host_tier.slices = list(slices)
        model.spill_slices = tuple(slices)
        bundle = model.park_persistent(bundle)
        for name in slices:
            assert model._spills(name)
    trace = []
    for _ in range(steps):
        bundle, metrics = model.step(bundle, cfg.dt_s)
        trace.append(_scalars(metrics))
    return model, bundle, _inventory(model, bundle), trace


# ------------------------------------------------- the step runs at all


@pytest.mark.parametrize("slices", [
    ("tracers",),
    ("physics",),
    ("surface",),
    ("physics", "surface"),
    SPILL_SLICES,
])
def test_the_semilagrangian_step_runs_with_the_tier_holding_each_slice(slices):
    """The crash, in the form the CPU can hold it: on numpy the parked
    slot reaches ``sum`` and raises ``unsupported operand type(s) for +:
    'int' and 'SpilledArray'``, on the card it reaches a cupy kernel and
    raises the argument-type error the T533 forecast day died on."""
    _model, bundle, _inv, _trace = _stepped(slices)
    assert bundle.step == 3


def test_the_dry_semilagrangian_route_runs_with_the_tier_too():
    """No physics suite, so the sponge-only pass is what returns the
    tracers to their slots; the core's own reads are the same three."""
    _model, bundle, _inv, _trace = _stepped(("tracers",), dry=True)
    assert bundle.step == 3


# --------------------------------------------------------------- BIT-1


@pytest.mark.parametrize("slices", [
    ("tracers",),
    ("physics",),
    ("surface",),
    ("physics", "surface"),
    SPILL_SLICES,
])
def test_bit_1_the_semilagrangian_step_is_the_resident_step_at_every_slice(slices):
    """Every array of the advanced bundle and every scalar the step
    reports, tier off against tier on.  One slice at a time as well as all
    three, because a defect in one slice's path is invisible in a run
    where another slice's path already moved the answer."""
    _m, _b, resident_inventory, resident_trace = _stepped(())
    _m2, _b2, spilled_inventory, spilled_trace = _stepped(slices)
    assert set(resident_inventory) == set(spilled_inventory)
    differing = sorted(
        name for name in resident_inventory
        if resident_inventory[name] != spilled_inventory[name]
    )
    assert not differing, f"{slices} moved {differing}"
    assert spilled_trace == resident_trace


@pytest.mark.parametrize("bands", (2, 4))
def test_bit_1_holds_with_the_band_pipeline_under_the_semilagrangian_core(bands):
    """The tier and the band pipeline are independent knobs and the
    semi-Lagrangian step is the same answer with either, both, or
    neither."""
    _m, _b, resident_inventory, resident_trace = _stepped((), bands=1)
    _m2, _b2, spilled_inventory, spilled_trace = _stepped(
        SPILL_SLICES, bands=bands)
    differing = sorted(
        name for name in resident_inventory
        if resident_inventory[name] != spilled_inventory[name]
    )
    assert not differing, f"bands={bands} with spill moved {differing}"
    assert spilled_trace == resident_trace


def test_the_same_spilled_semilagrangian_run_twice_is_the_same_answer():
    """Catches a transfer that passed by luck: an unordered copy is a
    race, and a race that lands right once is not a gate."""
    _m, _b, first, first_trace = _stepped(SPILL_SLICES)
    _m2, _b2, second, second_trace = _stepped(SPILL_SLICES)
    assert first == second
    assert first_trace == second_trace


# ---------------------------------------------- positive evidence of work


def test_the_core_actually_reads_the_tier_rather_than_the_host_copy():
    """A parked array carries ``__array__``, so numpy would have taken
    the host copy and run the sweep on the CPU without a word.  The tier's
    own counter is what says the bytes came across."""
    model, _bundle, _inv, _trace = _stepped(("tracers",))
    assert model.host_tier.stage_calls > 0
    assert model.host_tier.staged_bytes > 0


def test_the_advanced_tracers_come_back_to_the_tiers_own_slots():
    """The sweep advances the ten on the card; the physics half returns
    them.  A run whose tracers went resident at step one would hold the
    card it was given the tier to save, and its pinned slots would go
    stale."""
    model, bundle, _inv, _trace = _stepped(("tracers",))
    for name, value in bundle.atmosphere.grid_tracers().items():
        assert spilled(value), name
        assert value.tier is model.host_tier
    assert set(model.host_tier.parked) == {
        "tracer__" + name for name in bundle.atmosphere.grid_tracers()
    }


def test_the_flux_form_arm_leaves_the_ten_parked_through_the_sweep():
    """Under the flux-form tracer scheme the ten are not gathered: they
    ride the solve by reference and the Eulerian sweep in
    ``dynamics.step`` is what stages and re-parks them.  Staging them in
    the core as well would take the tier off the tracers for the rest of
    the run, so the core's staging sits inside the riding branch."""
    cfg = replace(
        load_config(CONFIG), integrator="sl_si", host_spill="off",
    )
    cfg = replace(cfg, semilag=replace(cfg.semilag, tracer_scheme="flux_form"))
    model, bundle = build_model_and_cold_state(cfg)
    model.host_tier = HostTier(model.transform.backend.xp)
    model.host_tier.slices = ["tracers"]
    model.spill_slices = ("tracers",)
    bundle = model.park_persistent(bundle)
    bundle, _metrics = model.step(bundle, cfg.dt_s)
    for name, value in bundle.atmosphere.grid_tracers().items():
        assert spilled(value), name


# -------------------------------------------- what a dead run leaves behind


def test_a_failed_run_records_the_tier_and_the_sizers_plan_as_well(
        tmp_path, monkeypatch):
    """The other half of the schedule a dead run was on.

    MEASURED 2026-09-07, RTX 5090: the T533 forecast day raised on a
    parked slot in its first step and its receipt read ``host_spill:
    null`` and ``sizer: null``, so the one artifact of the death could
    not say whether the tier was even on, let alone what it was holding.
    The band count was there because that gap was closed once already;
    this closes the rest of the same decision.
    """
    import json

    from woof.globe import runner as runner_module
    from woof.globe.runner import RECEIPT_NAME, run

    cfg = replace(load_config(CONFIG), host_spill="on")
    original = runner_module.MoistHybridModel.step

    def explode(self, bundle, dt_s, **kwargs):
        raise RuntimeError("the card went away")

    monkeypatch.setattr(runner_module.MoistHybridModel, "step", explode)
    try:
        with pytest.raises(RuntimeError):
            run(cfg, tmp_path / "died")
    finally:
        monkeypatch.setattr(runner_module.MoistHybridModel, "step", original)

    receipt = json.loads(
        (tmp_path / "died" / RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["status"] == "error"
    assert "host_spill" in receipt
    spill = receipt["host_spill"]
    assert spill is not None
    assert spill["mode"] == "on"
    assert spill["enabled"] is True
    # This config runs the reference suite, which keeps no native
    # namespace, so the physics slice is empty and the tier holds the
    # other two.  What the row has to say is which ones, not how many.
    assert set(spill["slices"]) <= set(SPILL_SLICES)
    assert "tracers" in spill["slices"]
    assert spill["parked_arrays"] > 0
    assert "sizer" in receipt
