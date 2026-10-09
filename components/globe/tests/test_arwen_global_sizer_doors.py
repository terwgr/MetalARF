"""The sizer's card model, its refusals, and the three doors it sizes.

What this file holds to, in one line each:

* the door refuses on what the CARD must hold, not on what the pool holds
  (gate MEM-1's arithmetic);
* fragmentation and the out-of-pool tax are MEASURED terms, each with its
  own table, and the planning value of each is the largest measurement
  rather than a typed constant;
* a shape above the largest measured truncation is refused BY NAME, and
  the sentence says which measurement lifts the refusal;
* the band count and the pinned host tier are chosen together, by one
  plan, and the plan the door prices is the plan the run takes;
* ``run``, ``cycle`` and ``assimilate`` all reach that plan with no flag
  set, and all three carry the memory levers (DOOR-1).
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import dataclasses
import json
import math
import os
from pathlib import Path

import pytest

from woof.globe import sizing
from woof.globe.cli import build_parser
from woof.globe.config import load_config

GIB = 1024 ** 3
QUICKSTART = Path(str(_shipped_configs() / "arwen_global_t255_quickstart.toml"))


def _native_cfg(**changes):
    cfg = dataclasses.replace(
        load_config(QUICKSTART), backend="cupy", precision="float32",
        physics_mode="arwen-native", **changes)
    return cfg


# --- the card model --------------------------------------------------------

def test_the_card_figure_is_the_pool_figure_plus_two_measured_terms():
    """``card = live x fragmentation + out-of-pool``, and both terms are
    read off measurements rather than typed."""

    live = 10 * GIB
    # THE MARGIN IS ADDED ON THE LIVE FIGURE AND NOTHING ELSE.  It is the
    # error bar of the model, which predicts the pool's live peak; the
    # fragmentation and the out-of-pool tax are separate measurements with
    # their own conservatism in them already, and charging the residual
    # against those too refused a T383 forecast this tree completed, and
    # the multiplicative form refused a T533 forecast the 16 GB card held.
    assert sizing.card_required_bytes(live) == math.ceil(
        live * sizing.POOL_FRAGMENTATION + live * (sizing.PREDICTION_MARGIN - 1.0)
    ) + sizing.OUT_OF_POOL_BYTES
    assert sizing.card_required_bytes(live, margin=1.0) == math.ceil(
        live * sizing.POOL_FRAGMENTATION) + sizing.OUT_OF_POOL_BYTES
    # Zero live bytes is zero card bytes: the numpy backend opens no
    # context and pays neither term.
    assert sizing.card_required_bytes(0) == 0
    # THE PLANNING VALUE OF EACH CLASS IS THE LARGEST MEASUREMENT OF THAT
    # CLASS, and the class is the allocation pattern: a band loop and a
    # host tier each change how the pool is asked for memory, MEASURED
    # x1.0296 to x1.3360 across the four combinations.
    for spilled in (False, True):
        for banded in (False, True):
            ratio, evidence = sizing.pool_fragmentation_for(spilled, banded)
            rows = [r for r in sizing.POOL_FRAGMENTATION_MEASURES
                    if not r.get("retired")
                    and bool(r.get("spilled", False)) is spilled
                    and bool(r.get("banded", False)) is banded
                    and not sizing._reduced_chunk(r.get("chunk"))]
            assert evidence
            if rows:
                assert ratio == pytest.approx(
                    max(r["held_bytes"] / r["live_bytes"] for r in rows),
                    rel=1e-9)
                assert "no measured" not in evidence
            else:
                # A class with no measurement borrows the widest
                # population's largest row and SAYS SO, rather than
                # quietly wearing another class's evidence.
                assert "no measured" in evidence
    assert sizing.POOL_FRAGMENTATION == pytest.approx(
        sizing.pool_fragmentation_for()[0], rel=1e-12)
    assert sizing.OUT_OF_POOL_BYTES == max(
        int(row["tax_bytes"]) for row in sizing.OUT_OF_POOL_MEASURES)
    # Every fragmentation row is a real pair of readings off one run.
    for row in sizing.POOL_FRAGMENTATION_MEASURES:
        assert row["held_bytes"] > row["live_bytes"] > 0
        assert row["where"]
    # WHAT THE TIER PARKS IS CALIBRATED TOO, and in the other direction:
    # the census estimate is SUBTRACTED from the peak, so reading it high
    # credits relief the run does not get.  MEASURED 2026-09-06 at T383 on
    # an RTX 5070 Ti, the tier parked 3.6 percent less than the terms
    # price, and the smallest measured ratio is the factor.
    assert 0.5 < sizing.SPILL_CENSUS_FACTOR <= 1.0
    assert sizing.SPILL_CENSUS_FACTOR == pytest.approx(min(
        row["parked_bytes"] / row["estimated_bytes"]
        for row in sizing.SPILL_CENSUS_MEASURES), rel=1e-12)

    # A RETIRED ROW IS HISTORY.  It measured a tree whose defect has been
    # fixed, so it may not set a planning value; it stays in the table
    # because what the defect was worth is worth keeping.
    retired = [r for r in sizing.POOL_FRAGMENTATION_MEASURES
               if r.get("retired")]
    assert retired, "the pre-fix fragmentation row is the record of what "                    "the transport defect cost, and it is not in the table"
    for row in retired:
        ratio = row["held_bytes"] / row["live_bytes"]
        assert "RETIRED" in row["where"]
        for spilled in (False, True):
            for banded in (False, True):
                got, _evidence = sizing.pool_fragmentation_for(spilled, banded)
                assert got != pytest.approx(ratio, rel=1e-12)
    # With no measurement at all the model refuses to hand out a planning
    # value rather than inventing one.
    original = sizing.POOL_FRAGMENTATION_MEASURES
    try:
        sizing.POOL_FRAGMENTATION_MEASURES = ()
        with pytest.raises(ValueError, match="no measured fragmentation row"):
            sizing.pool_fragmentation_for(True, True, True)
    finally:
        sizing.POOL_FRAGMENTATION_MEASURES = original


def test_the_card_figure_reproduces_the_measured_card_reading():
    """MEASURED 2026-09-06, RTX 5090, T383 L40 at the 12,500-column chunk,
    the ten-step probe of record, one band, spill off: the pool's live
    peak read 16,706,958,848 B and ``nvidia-smi`` read 18,300 MiB against
    this process for the life of the run.  The model's card figure for
    that live peak must not sit below the card reading, because sitting
    below it is what admits a run the card cannot hold."""

    live = 16_706_958_848
    measured_card = 18_300 * 1024 ** 2
    assert sizing.card_required_bytes(live) >= measured_card


# --- the fitted domain -----------------------------------------------------

def test_the_ceiling_is_read_from_the_calibration_table():
    """Adding a measured row raises the ceiling; no constant is typed."""

    assert sizing.fitted_truncation_ceiling() == max(
        int(row["truncation"]) for row in sizing.DEVICE_PEAK_CALIBRATION)
    assert sizing.fitted_truncation_ceiling() == max(
        sizing.CALIBRATED_DOMAIN["truncation"])


def test_a_shape_above_the_fitted_domain_is_refused_by_name():
    """THE BREAKAGE: above the largest measured truncation the model has
    only erred in the admitting direction, and the run it admitted on
    2026-09-06 died in the allocator several minutes in.  The refusal
    names the ceiling, the measurement that convicts the extrapolation,
    and the measurement that lifts the refusal."""

    ceiling = sizing.fitted_truncation_ceiling()
    inside = _native_cfg(truncation=ceiling, nlat=None, nlon=None)
    assert sizing.above_fitted_domain(inside, 40) is None

    above = _native_cfg(truncation=ceiling + 128, nlat=None, nlon=None,
                        zonal_wavenumber=5, diffusion_preserve_degree=1)
    sentence = sizing.above_fitted_domain(above, 40)
    assert sentence is not None
    assert f"T{ceiling}" in sentence
    assert "27.57" in sentence          # the run that was admitted and died
    assert "DEVICE_PEAK_CALIBRATION" in sentence
    assert "woof check" in sentence    # the door that still prices it

    # And the run door refuses on it however roomy the card is: the
    # figure itself is unweighed, so a bigger card does not make it true.
    gate = sizing.run_memory_gate(above, probe={"free_bytes": 400 * GIB})
    assert gate["refuse"] is True
    assert gate["unfitted"] == sentence


# --- the plan --------------------------------------------------------------

def test_the_plan_pays_the_tier_before_it_widens_the_band_count():
    """MEASURED both sides, 2026-09-06: banding costs 18 to 48 percent of
    a T255 step and 26 percent of a T533 step, while the tier's exposed
    transfer time read 5.9 percent of a T383 step.  Cheaper relief first,
    so a card that needs a little help parks a slice rather than cutting
    the globe into eight."""

    cfg = _native_cfg(truncation=255, nlat=None, nlon=None)
    estimate = sizing.estimate_global_memory(cfg)
    census = sizing._spill_census_estimate(cfg, estimate)
    assert census["physics"] > 0 and census["tracers"] > 0

    # A card that holds the resident run parks nothing and bands nothing.
    roomy = sizing.plan_run_memory(
        cfg, int(4 * sizing.card_required_bytes(estimate.device_peak_bytes)),
        estimate)
    assert roomy.bands == 1 and roomy.spill_slices == ()
    assert roomy.bands_chosen_by == "sizer" and roomy.spill_chosen_by == "sizer"

    # THE SEARCH ORDER, swept rather than sampled.  The plan is the FIRST
    # candidate that fits in a fixed cost-ordered enumeration -- band
    # counts ascending, and at each count the parked slices coldest-first
    # from none to all -- so over every card size the plan is taken, every
    # candidate ahead of it in that enumeration must not fit.  That is the
    # whole contract: cheaper relief first, and no cheaper plan skipped.
    def enumeration():
        present = [n for n, v in census.items() if v > 0]
        present.sort(key=sizing.SPILL_SLICES.index)
        for bands in sizing.LATITUDE_BAND_LADDER:
            for k in range(len(present) + 1):
                yield bands, tuple(present[:k])

    full = sizing.card_required_bytes(estimate.device_peak_bytes)
    saw_spill_at_one_band = 0
    saw_a_band_count_above_one = False
    for step in range(1, 61):
        free = int(full * 2.0 * (1.0 - step / 61.0))
        plan = sizing.plan_run_memory(cfg, free, estimate)
        if plan.bands is None:
            continue
        chosen = (plan.bands, tuple(plan.spill_slices))
        ceiling = free
        for candidate in enumeration():
            if candidate == chosen:
                break
            bands, slices = candidate
            spilled = sum(census[n] for n in slices)
            live = sizing.banded_device_peak_bytes(
                estimate, bands, spilled_bytes=spilled)
            card = sizing.card_required_bytes(
                live, spilled=bool(slices), banded=bands > 1,
                reduced_chunk=sizing._reduced_chunk(
                    estimate.radiation_column_chunk),
                bands=bands)
            assert card > ceiling, (
                f"{free / GIB:.2f} GiB free took {chosen} while the cheaper "
                f"{candidate} fitted")
        else:
            raise AssertionError(f"{chosen} is not in the enumeration")
        if plan.spill_slices:
            saw_spill_at_one_band = (
                plan.bands if not saw_spill_at_one_band
                else min(saw_spill_at_one_band, plan.bands))
        if plan.bands > 1:
            saw_a_band_count_above_one = True
    # The tier is reached while the band count is still small.  It is not
    # always reached at ONE band, and the measurement says why: a resident
    # run WITH the tier is the hardest-fragmenting pattern measured
    # (x1.3360), so two bands and a tier can price below one band and a
    # tier; and since the 34.7 km day of 2026-09-07 (two bands with the
    # tier on the 16 GB card, x1.1484 held over live) a two-band tier plan
    # at any truncation up to T383 is charged that ratio, so at this shape
    # the tier first arrives at four bands.  What matters is that the tier
    # arrives long before the ladder is exhausted.
    assert saw_spill_at_one_band, "no card size parked a slice at all"
    assert saw_spill_at_one_band <= 4, (
        f"the tier is first reached at {saw_spill_at_one_band} bands: it is "
        "meant to be the cheap relief, taken before the ladder is walked")
    assert saw_a_band_count_above_one

    # AND IT IS MONOTONE IN THAT ORDER.  More free VRAM may only ever buy
    # a cheaper plan.  The rule this replaced -- cheapest inside three
    # quarters of free, falling back to the whole card -- was not: just
    # below a threshold the fallback handed back a light plan and just
    # above it the budget pass handed back a heavier one, so freeing
    # memory made the model start spilling.
    order = {candidate: k for k, candidate in enumerate(enumeration())}
    previous = None
    for step in range(1, 121):
        free = int(full * 2.4 * (1.0 - step / 121.0))
        plan = sizing.plan_run_memory(cfg, free, estimate)
        if plan.bands is None:
            continue
        index = order[(plan.bands, tuple(plan.spill_slices))]
        if previous is not None:
            # The sweep walks DOWN in free VRAM, so the cost index may
            # only rise: a smaller card never gets a cheaper plan.
            assert index >= previous, (
                f"{free / GIB:.2f} GiB free took a CHEAPER plan than a "
                "larger card did")
        previous = index

    # A card that cannot hold it at any count and any spill says so, and
    # says what it tried.
    hopeless = sizing.plan_run_memory(cfg, GIB, estimate)
    assert hopeless.bands is None and hopeless.fits is False
    assert "nothing fits" in hopeless.reason


def test_the_config_is_taken_at_its_word_and_the_plan_says_who_chose():
    cfg = _native_cfg(truncation=255, nlat=None, nlon=None,
                      latitude_bands=8, host_spill="off")
    estimate = sizing.estimate_global_memory(cfg)
    plan = sizing.plan_run_memory(cfg, 32 * GIB, estimate)
    assert plan.bands == 8 and plan.spill_slices == ()
    assert plan.bands_chosen_by == "config" and plan.spill_chosen_by == "config"
    row = plan.receipt()
    assert row["latitude_bands"] == 8
    assert row["pool_fragmentation"] == pytest.approx(
        sizing.pool_fragmentation_for(False, True)[0], rel=1e-12)
    assert "banded" in row["pool_fragmentation_evidence"]
    assert row["out_of_pool_bytes"] == sizing.OUT_OF_POOL_BYTES
    assert row["predicted_card_bytes"] >= row["predicted_live_peak_bytes"]


def test_an_unknown_spill_mode_is_refused_rather_than_defaulted():
    cfg = _native_cfg(truncation=63, nlat=None, nlon=None,
                      host_spill="sometimes")
    with pytest.raises(ValueError, match="host_spill must be one of"):
        sizing.plan_run_memory(cfg, 32 * GIB)


# --- the gate --------------------------------------------------------------

def test_the_gate_weighs_card_bytes_and_says_so():
    """A live peak that fits the free VRAM but whose card requirement does
    not is refused.  This is the 2026-09-06 defect in one assertion."""

    cfg = _native_cfg(truncation=255, nlat=None, nlon=None, host_spill="off")
    live = sizing.estimate_global_memory(cfg).device_peak_bytes
    card = sizing.card_required_bytes(live)
    assert card > live

    between = sizing.run_memory_gate(
        cfg, probe={"free_bytes": (live + card) // 2})
    assert between["refuse"] is True
    assert "of card required" in between["verdict"]
    assert "fragmentation" in between["verdict"]

    roomy = sizing.run_memory_gate(
        cfg, probe={"free_bytes": int(card * 2)})
    assert roomy["refuse"] is False
    assert roomy["card_required_bytes"] == roomy["plan"].card_bytes


def test_the_gate_hands_the_plan_on_and_the_plan_is_what_the_run_takes():
    cfg = _native_cfg(truncation=255, nlat=None, nlon=None)
    gate = sizing.run_memory_gate(cfg, probe={"free_bytes": 40 * GIB})
    plan = gate["plan"]
    assert isinstance(plan, sizing.RunMemoryPlan)
    assert gate["latitude_bands"] == plan.bands
    assert gate["banded_peak_bytes"] == plan.live_peak_bytes
    assert gate["host_spill_slices"] == plan.spill_slices


# --- the doors -------------------------------------------------------------

@pytest.mark.parametrize("door", ["run", "assimilate", "cycle"])
def test_every_door_carries_the_memory_levers(door):
    """DOOR-1's front half: the levers are on all three doors, so a shape
    that needs one can be run through any of them.  Before this only
    ``run`` had them, and a T533 cycle had no way to name a band count."""

    parser = build_parser()
    args = parser.parse_args(_door_argv(door))
    for flag in ("latitude_bands", "host_spill", "spectral_chunk",
                 "device_allocator"):
        assert hasattr(args, flag), f"{door} is missing --{flag}"


def _door_argv(door: str) -> list[str]:
    cfg = str(QUICKSTART)
    return {
        "run": ["run", cfg, "--outdir", "out"],
        "assimilate": ["assimilate", cfg, "ckpt.npz", "--obs", "o.json",
                       "--out", "a.npz"],
        "cycle": ["cycle", cfg, "--outdir", "out", "--cycles", "1",
                  "--interval-s", "3600", "--obs", "o.json"],
    }[door]


@pytest.mark.parametrize("door", ["run", "assimilate", "cycle"])
def test_every_door_sizes_the_run_through_one_path(door, monkeypatch):
    """DOOR-1's back half: all three doors price the card through
    ``_size_the_run``, so the band count and the tier a bare run takes are
    the door's and not the builder's.  ``cycle`` used to price the card
    and throw the answer away; ``assimilate`` never asked."""

    from woof.globe import cli

    seen = {}

    def fake_size(args, cfg, door_name):
        seen["door"] = door_name
        raise SystemExit(0)

    monkeypatch.setattr(cli, "_size_the_run", fake_size)
    parser = build_parser()
    args = parser.parse_args(_door_argv(door))
    handler = {"run": cli._run, "assimilate": cli._assimilate,
               "cycle": cli._cycle}[door]
    with pytest.raises(SystemExit):
        handler(args)
    assert seen["door"]


# --- MEM-1 -----------------------------------------------------------------

def test_mem1_the_model_does_not_under_read_a_measured_run():
    """Gate MEM-1: on every shape the table has measured, the door's
    figure -- the prediction times its margin -- is at or above the
    measurement.  Under-reading is the direction that admits a run the
    card cannot hold, and it is the direction the model erred in on
    2026-09-06 at T383 (3.5 percent) and T533 (10 to 11)."""

    worst = None
    for row in sizing.calibration_residuals():
        weighed = sizing.PREDICTION_MARGIN * row["predicted_bytes"]
        ratio = weighed / row["measured_bytes"]
        worst = ratio if worst is None else min(worst, ratio)
        assert ratio >= 1.0, (
            f"{row['label']}: the door would weigh "
            f"{weighed / GIB:.2f} GiB against a measured "
            f"{row['measured_bytes'] / GIB:.2f} GiB")
    assert worst is not None


def test_the_plan_is_priced_at_its_own_radiation_chunks_fragmentation():
    """A run at a radiation chunk below the default is charged the
    fragmentation of THAT pattern, everywhere the figure is printed.

    THE BREAKAGE THIS PREVENTS, MEASURED 2026-09-06.  A smaller radiation
    chunk takes about 1.5 GiB off the LIVE peak while the pool still grows
    for the same allocations, so the ratio RISES on a run that is smaller:
    T383 resident reads x1.1011 at the default 12,500-column chunk and
    x1.2020 at 5,000, and T383 at eight bands with the tier reads x1.0296
    at the default and x1.1626 at 5,000.  A door that priced the plan at
    the default chunk's class asked 18.43 GiB of card for the 5,000-column
    T383 resident run that HELD 17.76, where the measured class asks
    18.96: the whole clearance of that leg came from a ratio measured on a
    different allocation pattern.
    """

    default = _native_cfg(truncation=383)
    reduced = dataclasses.replace(
        default,
        native_adapter_options={**dict(default.native_adapter_options),
                                "radiation_column_chunk": 5_000})
    assert not sizing._reduced_chunk(
        sizing.estimate_global_memory(default).radiation_column_chunk)
    assert sizing._reduced_chunk(
        sizing.estimate_global_memory(reduced).radiation_column_chunk)

    for cfg, is_reduced in ((default, False), (reduced, True)):
        estimate = sizing.estimate_global_memory(cfg)
        # Free enough that the resident, nothing-parked plan is the one
        # taken, so the class under test is the one being priced.
        free = 4 * sizing.estimate_card_required_bytes(estimate)
        plan = sizing.plan_run_memory(cfg, free, estimate)
        assert plan.reduced_chunk is is_reduced
        assert plan.fragmentation == sizing.pool_fragmentation_for(
            bool(plan.spill_slices), bool(plan.bands > 1), is_reduced,
            bands=plan.bands, truncation=plan.truncation)
        # The price, the receipt and the gate's verdict all read the same
        # ratio: a sentence whose two halves do not multiply out is a
        # sentence a reader cannot check.
        assert plan.card_bytes == sizing.card_required_bytes(
            plan.live_peak_bytes, spilled=bool(plan.spill_slices),
            banded=plan.bands > 1, reduced_chunk=is_reduced,
            bands=plan.bands, truncation=plan.truncation)
        receipt = plan.receipt()
        assert receipt["pool_fragmentation"] == plan.fragmentation[0]
        assert receipt["reduced_radiation_chunk"] is is_reduced
        # And the estimate route -- woof check, the refusal, the gate
        # list, the truncation bisection -- charges the same class.
        assert sizing.estimate_card_required_bytes(estimate) == (
            sizing.card_required_bytes(
                estimate.device_peak_bytes, reduced_chunk=is_reduced))

    # The reduced-chunk run is the SMALLER run and still asks more of the
    # card per live byte, which is the whole point of the class.
    lean = sizing.estimate_global_memory(reduced)
    fat = sizing.estimate_global_memory(default)
    assert lean.device_peak_bytes < fat.device_peak_bytes
    assert (sizing.estimate_card_required_bytes(lean)
            / lean.device_peak_bytes) > (
        sizing.estimate_card_required_bytes(fat) / fat.device_peak_bytes)


def test_no_measured_card_leg_fragments_harder_than_the_class_it_is_priced_at():
    """Every card leg this lane recorded is at or below the planning value
    of its own allocation pattern.

    THE BREAKAGE THIS PREVENTS, MEASURED 2026-09-06.  The rule the module
    states is "each class takes the largest of its own measured rows", and
    a class is only as conservative as the rows it was given: the BARE
    T383 run on the 16 GB card -- the plan the sizer actually chooses
    there, two bands with all three slices parked -- fragmented at x1.0877
    while its class was set to x1.0748 from three flagged runs at eight
    and sixteen bands.  The row that mattered most was the one not in the
    table.  This reads the recorded legs and fails when the table is
    behind them, so a new card leg cannot be filed on the branch without
    the planning value it moves.
    """

    # WHERE THE LEGS ARE is asked of the environment rather than written
    # here: a recorded card leg is a run's own receipt on the machine that
    # ran it, and a path into one tree resolves for nobody else.  Point
    # ARWEN_GLOBAL_CARD_LEGS at a directory holding `results/*.json` and
    # the configs those rows name; without it there is nothing to read and
    # this skips.
    legs_root = Path(os.environ.get("ARWEN_GLOBAL_CARD_LEGS", "card-legs"))
    results = legs_root / "results"
    if not results.is_dir():
        pytest.skip("the recorded card legs are not in this checkout")
    legs = 0
    for path in sorted(results.rglob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(row, dict) or "config" not in row:
            continue
        if row.get("returncode") != 0 or not row.get("live_peak_bytes"):
            continue
        if not row.get("pool_held_bytes"):
            continue
        config = legs_root / Path(row["config"]).name
        if not config.exists():
            continue
        estimate = sizing.estimate_global_memory(load_config(config))
        ratio = float(row["pool_held_bytes"]) / float(row["live_peak_bytes"])
        planning, evidence = sizing.pool_fragmentation_for(
            bool(row.get("spill_slices")),
            int(row.get("bands") or 1) > 1,
            sizing._reduced_chunk(estimate.radiation_column_chunk))
        legs += 1
        assert ratio <= planning + 1e-9, (
            f"{path.name} held {ratio:.4f} of its live bytes, above the "
            f"{planning:.4f} its class is priced at ({evidence}): the "
            "planning value is meant to be the largest measured row of "
            "the class, so this leg belongs in "
            "POOL_FRAGMENTATION_MEASURES")
    # A floor, so a reader knows the loop found legs rather than an empty
    # directory.  It is deliberately well below the twelve on the branch:
    # a checkout synced to a node can carry the code without the last of
    # the receipts, and a gate that fails on that is failing on the sync
    # rather than on the tree.
    assert legs >= 5, f"only {legs} recorded legs were read"


@pytest.mark.parametrize("door", ["run", "assimilate", "cycle"])
def test_every_door_hands_the_plan_to_what_writes_the_receipt(door,
                                                              monkeypatch):
    """DOOR-1 finishes at the file: the plan the door priced reaches the
    call that writes the run's own record, on all three doors.

    A door that sizes and then drops the plan leaves a receipt that
    cannot say what was chosen or who chose it, which is the state
    ``cycle`` and ``assimilate`` were in: ``cycle`` priced the card and
    threw the answer away, and ``assimilate`` never asked and still
    writes the one report of the three with no sizing block in it.
    """

    import inspect

    from woof.globe import assimilate as assimilate_module
    from woof.globe import cli
    from woof.globe import cycle as cycle_module
    from woof.globe import runner as runner_module

    target = {"run": runner_module.run,
              "assimilate": assimilate_module.assimilate,
              "cycle": cycle_module.cycle}[door]
    assert "door_plan" in inspect.signature(target).parameters, (
        f"{door} cannot be handed the plan its door priced")

    sentinel = object()
    seen = {}

    def fake_size(args, cfg, door_name):
        return cfg, sentinel

    def capture(*args, **kwargs):
        seen["plan"] = kwargs.get("door_plan")
        raise SystemExit(0)

    monkeypatch.setattr(cli, "_size_the_run", fake_size)
    monkeypatch.setattr(cli, "run", capture, raising=False)
    monkeypatch.setattr(assimilate_module, "assimilate", capture)
    monkeypatch.setattr(cycle_module, "cycle", capture)
    parser = build_parser()
    args = parser.parse_args(_door_argv(door))
    handler = {"run": cli._run, "assimilate": cli._assimilate,
               "cycle": cli._cycle}[door]
    with pytest.raises(SystemExit):
        handler(args)
    assert seen.get("plan") is sentinel, (
        f"{door} sized the run and did not hand the plan on")
