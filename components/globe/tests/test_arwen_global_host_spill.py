"""The pinned host tier: a place, not an arithmetic.

The persistent grid state -- the ten grid tracers, the surface reservoirs
and the native physics namespace -- is alive from one step to the next
whatever the latitude band count is, so a band count does not divide it
and it sits on the card for the whole run.  MEASURED 2026-09-06, RTX
5090, T533 L40 native at eight bands: 4.388 GiB of a 20.433 GiB live
peak, and the section profile puts that peak in the physics half-step,
which holds all three slices.

The tier parks those arrays in pinned host memory and hands the card a
copy only where a copy is made anyway.  Everything below is the claim
that this changes NOTHING an answer depends on:

BIT-1 with spill  the shipped step, tier off against tier on, at every
                  subset of the three slices and at several band counts:
                  every array of the advanced bundle and every scalar
                  metric byte-identical
the door          a whole ``woof global run`` either way: the same
                  checkpoints, array for array, and the same config hash
the contract      a parked handle is not an array, the tier's slot is
                  written once per replacement and counted, and a
                  checkpoint takes its own host copy because the writer
                  thread outlives the step
the policy        minimum spill sheds the coldest slice first and stops
                  the moment the prediction fits
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import hashlib
from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_engine_module  # noqa: E402

from woof.globe.checkpoint import bundle_arrays
from woof.globe.config import HOST_SPILL_MODES, load_config
from woof.globe.runner import build_model_and_cold_state
from woof.globe.sizing import (
    NATIVE_NAMESPACE_FIXED_PLANES,
    NATIVE_NAMESPACE_LEVELLED_ARRAYS,
    SPILL_SLICES,
    estimate_global_memory,
    native_namespace_bytes,
    spill_census,
    spill_slices_for,
)
from woof.globe.spill import (
    SLICE_READS_PER_STEP,
    HostTier,
    SpilledArray,
    pinned_host_array,
    resident,
    spilled,
)

CONFIG = str(_shipped_configs() / "arwen_global_t21_baroclinic_ten_day.toml")


# ------------------------------------------------------------ the object


def test_a_parked_array_carries_the_interface_a_reader_needs_and_no_more():
    """It answers shape, dtype and nbytes, and it is NOT an array.

    A transparent proxy would let an operator run the step on the host
    and never say so; the handle refuses instead, which is how every
    reader of the persistent grid state was found (the in-situ ledger's
    reservoir minimum, the DA door's soil subscript, the export's).
    """
    tier = HostTier(np)
    value = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    handle = tier.park("x", value)
    assert spilled(handle)
    assert handle.shape == (2, 3, 4)
    assert handle.dtype == np.dtype(np.float32)
    assert handle.ndim == 3 and handle.nbytes == value.nbytes
    for attribute in ("min", "max", "sum", "astype", "reshape", "__getitem__"):
        assert not hasattr(handle, attribute), attribute


def test_a_parked_array_round_trips_whole_and_by_band():
    tier = HostTier(np)
    value = np.arange(120, dtype=np.float32).reshape(2, 6, 10)
    handle = tier.park("x", value)
    assert np.array_equal(resident(np, handle), value)
    for rows in (slice(0, 6), slice(0, 3), slice(2, 5), slice(5, 6)):
        assert np.array_equal(resident(np, handle, rows), value[:, rows, :])


def test_a_slot_is_replaced_whole_and_every_write_is_counted():
    """The version stamp is the physics boundary's fingerprint of a
    parked array, so it must move on every write and only on a write."""
    tier = HostTier(np)
    handle = tier.park("x", np.zeros((2, 3, 4), dtype=np.float32))
    assert handle.version == 1
    resident(np, handle)
    resident(np, handle, slice(1, 2))
    assert handle.version == 1
    handle.store(np.ones((2, 3, 4), dtype=np.float32))
    assert handle.version == 2
    assert float(np.asarray(handle).min()) == 1.0
    handle.store(np.full((2, 2, 4), 5.0, dtype=np.float32), slice(1, 3))
    assert handle.version == 3
    got = np.asarray(handle)
    assert float(got[:, 0, :].max()) == 1.0
    assert float(got[:, 1:, :].min()) == 5.0


def test_a_parked_arrays_copy_is_the_same_slot_and_the_reason_is_stated():
    """``SurfaceState.copy`` and ``PhysicsState.copy`` copy every array
    they hold; duplicating a parked one would double the tier and copy
    2.3 GiB of host memory four times a step at T533 for a value nothing
    distinguishes."""
    tier = HostTier(np)
    handle = tier.park("x", np.zeros((2, 3), dtype=np.float32))
    assert handle.copy() is handle
    assert "same parked array" in SpilledArray.copy.__doc__.lower()


def test_parking_reuses_the_slot_rather_than_allocating_a_new_one():
    tier = HostTier(np)
    first = tier.park("x", np.zeros((2, 3), dtype=np.float32))
    before = tier.parked_bytes
    for value in (1.0, 2.0, 3.0):
        again = tier.park("x", np.full((2, 3), value, dtype=np.float32))
        assert again is first
    assert tier.parked_bytes == before
    assert float(np.asarray(first).max()) == 3.0


def test_a_pinned_slot_is_a_plain_host_array_on_the_numpy_backend():
    """One code path either way, so the CPU gates exercise the object
    the card runs."""
    host = pinned_host_array((3, 4), np.float64, xp=np)
    assert host.shape == (3, 4) and host.dtype == np.float64
    host[...] = 7.0
    assert float(host.min()) == 7.0


def test_a_host_array_written_into_a_slot_takes_the_host_path_on_a_card():
    """The device branch is chosen by the method it calls, not by an
    attribute every numpy array has.

    ``numpy.ndarray.device`` exists from numpy 2.0 and answers "cpu", so
    a tier that asked ``hasattr(value, "device")`` sent a host array down
    the device branch on the cupy backend and died in ``source.get`` with
    an AttributeError rather than writing the slot.  The branch is taken
    here with the tier flagged as a card's, which is the only way to
    reach it without one.
    """
    assert hasattr(np.empty(2), "device"), (
        "this test exists because numpy arrays carry a device attribute"
    )
    tier = HostTier(np, enabled=True)
    handle = tier.park("plane", np.zeros((2, 3), dtype=np.float32))
    tier._cupy = True
    try:
        tier.store(handle, np.full((2, 3), 4.5, dtype=np.float32))
    finally:
        tier._cupy = False
    assert np.array_equal(handle.host, np.full((2, 3), 4.5, dtype=np.float32))


def test_the_tier_refuses_to_park_when_it_is_off():
    tier = HostTier(np, enabled=False)
    with pytest.raises(RuntimeError, match="host tier is off"):
        tier.park("x", np.zeros(3))


# ------------------------------------------------------------ the policy


def test_the_slice_order_is_the_cold_order_and_it_is_counted():
    """The namespace and the surface are read by the two physics halves;
    the tracers by those two and by the tracer transport."""
    assert SPILL_SLICES == tuple(SLICE_READS_PER_STEP)
    reads = [SLICE_READS_PER_STEP[name] for name in SPILL_SLICES]
    assert reads == sorted(reads), reads
    assert SLICE_READS_PER_STEP == {"physics": 2, "surface": 2, "tracers": 3}


def test_minimum_spill_sheds_the_coldest_slice_first_and_stops():
    census = {"physics": 4, "surface": 1, "tracers": 5}
    # Nothing to shed: the prediction already fits.
    assert spill_slices_for("auto", census, peak_bytes=10, budget_bytes=100) == ()
    # One slice is enough.
    taken = spill_slices_for("auto", census, peak_bytes=10, budget_bytes=8)
    assert taken == ("physics",)
    # One is enough here too, and the second is not taken: 10 - 4 = 6
    # against 6.  The model's margin is no longer applied here, because it
    # multiplies the LIVE figure inside sizing.card_required_bytes and
    # applying it again to a budget comparison charged it twice.
    taken = spill_slices_for("auto", census, peak_bytes=10, budget_bytes=6)
    assert taken == ("physics",)
    taken = spill_slices_for("auto", census, peak_bytes=10, budget_bytes=5)
    assert taken == ("physics", "surface")
    # Everything, and it stops there rather than looping.
    taken = spill_slices_for("auto", census, peak_bytes=100, budget_bytes=1)
    assert taken == SPILL_SLICES


def test_the_two_explicit_modes_do_what_they_say():
    census = {"physics": 4, "surface": 1, "tracers": 5}
    assert spill_slices_for("off", census, peak_bytes=100, budget_bytes=1) == ()
    assert spill_slices_for("on", census, peak_bytes=1, budget_bytes=100) == SPILL_SLICES
    with pytest.raises(ValueError, match="host_spill must be one of"):
        spill_slices_for("sometimes", census, peak_bytes=1, budget_bytes=1)


def test_the_policy_prices_a_namespace_that_does_not_exist_yet():
    """The suite seeds its namespace on its FIRST call, after the tier
    has had to decide what it holds, so a policy that read the cold state
    would never park the largest slice there is.  MEASURED 2026-09-06:
    86 of the 120 persistent arrays are the namespace, 88 fixed planes
    plus ten levelled volumes, 2.333 GiB at T533 L40."""
    cfg = load_config(CONFIG)
    assert native_namespace_bytes(cfg) == 0  # the reference suite has none
    native = replace(cfg, physics_mode="arwen-native")
    estimate = estimate_global_memory(native)
    planes = (
        NATIVE_NAMESPACE_FIXED_PLANES
        + NATIVE_NAMESPACE_LEVELLED_ARRAYS * estimate.nlev
    )
    expected = planes * estimate.nlat * estimate.nlon * estimate.float_itemsize
    assert native_namespace_bytes(native) == expected
    assert expected > 0


def test_the_census_reads_the_arrays_rather_than_a_fit():
    cfg = load_config(CONFIG)
    model, bundle = build_model_and_cold_state(replace(cfg, host_spill="off"))
    census = spill_census(bundle, cfg)
    assert set(census) == set(SPILL_SLICES)
    tracers = sum(
        value.nbytes for value in bundle.atmosphere.grid_tracers().values()
    )
    surface = sum(value.nbytes for value in bundle.surface.arrays().values())
    assert census["tracers"] == tracers
    assert census["surface"] == surface


def test_a_bare_run_parks_the_tier_exactly_where_the_card_needs_it():
    """FIXED MEANS DEFAULT, on the shapes the design names.

    A bare ``woof global run`` sets no flag, so the tier is reached
    through the door's plan (``sizing.plan_run_memory``, band count and
    slices together, priced with the measured fragmentation of the plan's
    allocation pattern) or it is not reached at all.  The rows below are
    the design's own capacity table read through that plan on each card:
    T383 fits a 32 GiB card and parks nothing, does not fit a 16 GiB one
    and parks all three slices; T533 fits a 32 GiB card at more than one
    band with nothing parked, and on a 16 GiB card parks all three
    slices.  MEASURED beside it: T383 at eight bands runs on the 16 GB
    card with the tier on and dies with ``OutOfMemoryError`` with it off
    (2026-09-06); T533 on the same card runs at sixteen bands with all
    three slices parked, 9.59 GiB live and 13.57 held (2026-09-07, the
    IMEX core).

    The configs are the SHIPPED T383 and T533 native experiments, not the
    band-gate configs of the campaign this test was written in: those
    lived under a working directory of one machine and reach no artefact,
    so against an installed wheel the test opened a path that does not
    exist.  What the plan prices is the grid, and the shipped pair
    carries the same one -- T383 and T533, dealias 1.5, forty
    surface-stretched levels, float32 -- so these are the same rows.
    """
    from woof.globe.sizing import plan_run_memory

    MIB = 1024 ** 2
    rows = []
    # What each card READS free with nothing else on it (nvidia-smi,
    # MEASURED 2026-09-07: the 16 GB card 15,880 MiB, the 32 GB card
    # 32,146 MiB), which is the figure the door prices against.
    # NEITHER T533 ROW IS A FIXED ANSWER HERE, and the absence is a
    # measurement rather than an omission.  The source tree's two T533 rows
    # were taken on the campaign's band-gate configuration; the SHIPPED T533
    # 24-hour experiment is a heavier one.  Against it the door parks the whole
    # tier and still refuses the 16 GB card (22.21 GiB of card against 15.51
    # free, MEASURED 2026-09-07), which is the door doing exactly its job.
    # Asserting the campaign's answer against a configuration it was never
    # measured on would be asserting a number rather than reading one, so T533
    # is judged by the RULE below on both cards -- which is the thing a bare
    # run relies on anyway.
    for name, expected in (
        ("arwen_global_gdas_t383_native_24h.toml",
         {15880: (True, SPILL_SLICES), 32146: (None, ())}),
        ("arwen_global_gdas_t533_native_24h.toml", {}),
    ):
        config = str(_shipped_configs() / name)
        cfg = load_config(config)
        estimate = estimate_global_memory(cfg)
        census = {
            "physics": native_namespace_bytes(cfg),
            "surface": estimate.surface_grid_bytes,
            "tracers": int(
                10 * estimate.nlev * estimate.nlat * estimate.nlon
                * estimate.float_itemsize
            ),
        }
        assert all(value > 0 for value in census.values()), census
        for free_mib in (15880, 32146):
            plan = plan_run_memory(cfg, free_mib * MIB, estimate, census=census)
            got = (plan.bands, plan.spill_slices)

            # THE RULE, on every combination and not only the named rows.  A
            # capacity table is a fact about two configurations; this is the
            # contract a bare run relies on on ANY configuration, and a table
            # that passed while the rule was broken would be a gate reading
            # green on the wrong tree.
            #
            #   the tier is taken COLDEST FIRST, a prefix and never a hole;
            #   a plan that names no band count has tried the WHOLE tier and
            #   says why, rather than refusing with something still in hand.
            assert plan.spill_slices == tuple(
                SPILL_SLICES[:len(plan.spill_slices)]), (config, free_mib, got)
            if plan.bands is None:
                assert plan.spill_slices == tuple(SPILL_SLICES), (
                    config, free_mib, got, "refused with slices still unparked")
                assert plan.reason, (config, free_mib, "refused without a reason")
            else:
                assert plan.bands >= 1, (config, free_mib, got)

            if free_mib in expected:
                banded, want = expected[free_mib]
                ok = plan.spill_slices == want and (
                    banded is None or (plan.bands > 1) == banded)
                rows.append((config, free_mib, got, want, ok))
    assert len(rows) == 2, rows
    wrong = [row for row in rows if not row[4]]
    assert not wrong, wrong


def test_host_spill_is_outside_the_identity_at_every_value():
    """A place is not an arithmetic: a spilled run shares a config hash,
    a checkpoint lineage and a receipt with a resident one."""
    cfg = load_config(CONFIG)
    hashes = {
        mode: replace(cfg, host_spill=mode).config_hash
        for mode in HOST_SPILL_MODES
    }
    assert len(set(hashes.values())) == 1, hashes
    assert "host_spill" not in replace(cfg, host_spill="on").config_identity


# ------------------------------------------------------------- BIT-1


def _digest(host, value) -> str:
    array = np.asarray(host(value))
    return hashlib.sha256(
        array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()
    ).hexdigest()


def _inventory(model, bundle) -> dict[str, str]:
    host = model.transform.backend.to_numpy
    out = {}
    for name in ("vorticity", "divergence", "theta", "log_surface_pressure", "qv"):
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


def _stepped(slices, bands=1, steps=2, integrator=None):
    cfg = replace(
        load_config(CONFIG), latitude_bands=bands, host_spill="off",
        **({"integrator": integrator} if integrator else {}),
    )
    model, bundle = build_model_and_cold_state(cfg)
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


@pytest.mark.parametrize("slices", [
    ("physics",),
    ("surface",),
    ("tracers",),
    ("physics", "surface"),
    ("physics", "surface", "tracers"),
])
def test_bit_1_with_spill_the_step_is_the_resident_step_at_every_slice(slices):
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
def test_bit_1_with_spill_holds_at_a_band_count_as_well(bands):
    """The tier and the band pipeline are independent knobs and the run
    is the same answer with either, both, or neither."""
    _m, _b, resident_inventory, resident_trace = _stepped((), bands=1)
    _m2, _b2, spilled_inventory, spilled_trace = _stepped(
        SPILL_SLICES, bands=bands)
    differing = sorted(
        name for name in resident_inventory
        if resident_inventory[name] != spilled_inventory[name]
    )
    assert not differing, f"bands={bands} with spill moved {differing}"
    assert spilled_trace == resident_trace


def test_bit_3_with_spill_the_same_run_twice_is_the_same_answer():
    """Catches a transfer that passed by luck: an unordered copy is a
    race, and a race that lands right once is not a gate."""
    _m, _b, first, first_trace = _stepped(SPILL_SLICES)
    _m2, _b2, second, second_trace = _stepped(SPILL_SLICES)
    assert first == second
    assert first_trace == second_trace


def test_the_tier_actually_holds_the_state_it_says_it_holds():
    """Positive evidence of work: a tier that parked nothing would pass
    every compare above by doing nothing at all."""
    model, bundle, _inv, _trace = _stepped(SPILL_SLICES)
    for value in bundle.atmosphere.grid_tracers().values():
        assert spilled(value)
    for value in bundle.surface.arrays().values():
        assert spilled(value)
    for value in bundle.physics_state.arrays.values():
        assert spilled(value)
    receipt = model.host_tier.receipt()
    assert receipt["parked_arrays"] >= 10
    assert receipt["stage_calls"] > 0
    assert receipt["store_calls"] > 0
    assert receipt["steps"] == 2


# --------------------------------------------------------- the checkpoint


def test_the_checkpoint_takes_its_own_host_copy_of_every_parked_array():
    """The writer thread outlives the step that produced these values
    while the tier's slot is written again by the next physics call, so
    handing the writer the slot itself would hash a state that is half
    this step and half the next."""
    model, bundle, _inv, _trace = _stepped(SPILL_SLICES)
    arrays = bundle_arrays(bundle, model.transform.backend.to_numpy)
    handle = bundle.atmosphere.qc
    assert spilled(handle)
    written = arrays["atmosphere__qc"]
    assert np.array_equal(written, np.asarray(handle))
    assert written.base is not handle.host
    before = written.copy()
    handle.store(np.full(handle.shape, 3.5, dtype=handle.dtype))
    assert np.array_equal(written, before)


def test_the_checkpoint_of_a_spilled_run_is_the_resident_runs_checkpoint():
    resident_model, resident_bundle, _i, _t = _stepped(())
    spilled_model, spilled_bundle, _i2, _t2 = _stepped(SPILL_SLICES)
    left = bundle_arrays(
        resident_bundle, resident_model.transform.backend.to_numpy)
    right = bundle_arrays(
        spilled_bundle, spilled_model.transform.backend.to_numpy)
    assert set(left) == set(right)
    differing = sorted(
        name for name in left
        if left[name].dtype != right[name].dtype
        or left[name].shape != right[name].shape
        or not np.array_equal(left[name], right[name])
    )
    assert not differing, differing
    assert len(left) >= 40


@requires_engine_module("woof.verify.harness", "04")
def test_the_radiation_instrument_shapes_a_scenario_on_a_parked_exchange():
    """A tool that OVERWRITES the exchange it is handed still works.

    ``build_model_and_cold_state`` attaches the tier, and every tool that
    builds a model through it is handed a state whose grid tracers and
    surface reservoirs may be parked handles.  The radiation instrument
    shapes its scenarios by writing into the exchange
    (``verify/harness/radiation_balance._shape_scenario``), which raised
    ``'SpilledArray' object does not support item assignment`` and failed
    the whole instrument rather than grading.  The scenario now owns the
    fields it writes, and the model's own slots are left as they were.
    """
    from woof.verify.harness.radiation_balance import _shape_scenario

    cfg = replace(load_config(CONFIG), host_spill="off")
    model, bundle = build_model_and_cold_state(cfg)
    model.host_tier = HostTier(model.transform.backend.xp)
    model.host_tier.slices = list(SPILL_SLICES)
    model.spill_slices = tuple(SPILL_SLICES)
    bundle = model.park_persistent(bundle)

    exchange = model._physics_exchange(bundle, cfg.dt_s)
    # The tier holds the model's grid tracers; the exchange carries the
    # STAGED band of them (the whole grid here), on the card, and never the
    # slot itself: the physics runs a band at a time and a band of a parked
    # array reaches the card as its own copy (dynamics._physics_exchange_band).
    assert spilled(bundle.atmosphere.grid_tracers()["qc"]), "the tier holds the grid tracers"
    assert not spilled(exchange.qc), "the exchange carries the staged band"
    before = np.array(np.asarray(bundle.atmosphere.grid_tracers()["qc"]), copy=True)
    before_skin = np.array(np.asarray(bundle.surface.temperature_k), copy=True)

    shaped = _shape_scenario(
        model.transform.backend, exchange,
        air_top_k=260.0, air_bottom_k=260.0, surface_temperature_k=288.0,
        longitude_deg=-1.25,
    )
    assert not spilled(shaped.qc)
    assert float(np.max(np.abs(np.asarray(shaped.qc)))) == 0.0
    assert float(np.max(np.abs(np.asarray(shaped.surface.temperature_k) - 288.0))) == 0.0
    # The model's own state is untouched: the scenario wrote its copies.
    assert np.array_equal(
        np.asarray(bundle.atmosphere.grid_tracers()["qc"]), before)
    assert np.array_equal(np.asarray(bundle.surface.temperature_k), before_skin)


@requires_engine_module("woof.verify.harness", "04")
def test_energy_attribution_refuses_a_model_whose_state_the_tier_holds():
    """Two arms from one state cannot share the tier's slots.

    The tier keeps ONE named slot per array for the life of a run, and a
    physics half-step writes its result back into those slots.  A
    stepping loop never wants the previous state again, so that is safe;
    the energy attribution does want it -- it runs ``apply_physics`` with
    the lid absorber and without it from the same bundle and books the
    difference.  MEASURED 2026-09-06 on the T3 smoke state with the tier
    forced on: the lid absorber's theta attribution, which is exactly
    zero for a momentum-only absorber, read 1.7e-08.  The instrument
    builds its model resident now, and refuses one that is not.
    """
    from woof.verify.harness.energy_tendency import attribute_checkpoint
    from woof.verify.harness.subjects import instrument_config

    cfg = replace(load_config(CONFIG), host_spill="on")
    assert instrument_config(cfg).host_spill == "off"

    model, bundle = build_model_and_cold_state(replace(cfg, host_spill="off"))
    model.host_tier = HostTier(model.transform.backend.xp)
    model.host_tier.slices = list(SPILL_SLICES)
    model.spill_slices = tuple(SPILL_SLICES)
    bundle = model.park_persistent(bundle)
    with pytest.raises(ValueError, match="TWO arms from one state"):
        attribute_checkpoint(model, bundle, cfg.dt_s, [0])


# ------------------------------------------- the semi-Lagrangian core


def test_the_runners_free_figure_counts_this_processs_own_pool(monkeypatch):
    """The subprocess probe cannot see the blocks this process's pool
    already holds; a launcher that pre-grows the pool to claim a shared
    card leaves them counted as used on the card and free to this run
    alone.  The door's gate credited them (sizing.pool_bytes_held_unused)
    and the RUNNER's parking decision did not: MEASURED 2026-09-07, a DA
    run launched with 30.2 GiB free and a 21 GiB pre-grown pool read 9
    GiB at the runner, parked the tracers to fit, and died in the
    semi-Lagrangian gather.  One figure now, at both places; a process
    without cupy loaded holds nothing and adds nothing."""
    import sys
    import types

    import woof.core.preflight as preflight
    from woof.globe import sizing

    gib = 1024 ** 3
    monkeypatch.setattr(
        preflight, "device_memory_probe_subprocess",
        lambda **kwargs: {"free_bytes": 5 * gib, "total_bytes": 32 * gib})
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    assert sizing._free_device_bytes() == 5 * gib

    class _Pool:
        def total_bytes(self):
            return 24 * gib

        def used_bytes(self):
            return 3 * gib

    monkeypatch.setitem(
        sys.modules, "cupy", types.SimpleNamespace(get_default_memory_pool=_Pool))
    assert sizing.pool_bytes_held_unused() == 21 * gib
    assert sizing._free_device_bytes() == 26 * gib


@pytest.mark.parametrize("slices", [
    ("physics", "surface"),
    SPILL_SLICES,
])
def test_bit_1_with_spill_holds_under_the_semilagrangian_core(slices):
    """The semi-Lagrangian step with the tier holding two slices, and all
    three, is the resident step array for array and scalar for scalar:
    a parked tracer reaches the gather as its handle and is staged one
    batch at a time (interpolate.gather_batch), the mass fixer stages the
    before-state one species at a time, and the fixer's water column
    reads each species through the tier's door.  MEASURED 2026-09-07
    before this: a parked handle in the CUDA gather killed the balance
    lane's bare DA run under sl_si in its first step."""
    _m, _b, resident_inventory, resident_trace = _stepped((), integrator="sl_si")
    _m2, _b2, spilled_inventory, spilled_trace = _stepped(slices, integrator="sl_si")
    assert set(resident_inventory) == set(spilled_inventory)
    differing = sorted(
        name for name in resident_inventory
        if resident_inventory[name] != spilled_inventory[name]
    )
    assert not differing, f"{slices} moved {differing}"
    assert spilled_trace == resident_trace
