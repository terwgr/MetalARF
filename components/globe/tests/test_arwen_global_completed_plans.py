"""A completed run of a lighter plan bounds a heavier plan's live figure
from above (sizing.COMPLETED_PLAN_PEAKS, sizing.completed_plan_ceiling).

The breakage these hold: MEASURED 2026-09-07, the model read 15.08 GiB
live for T383 L40 at two bands with every slice parked and refused the
34.7 km day on a 16 GB card that had just completed it at 12.41 GiB live
(RTX 5070 Ti, the 24 h day run); and read 22.81 GiB for T533 L40 at
one band with every slice parked, at a pool ratio borrowed from another
class, where the 25 km day completed on the RTX 5090 at 21.12 GiB live
(the 25 km day run).
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from woof.globe.configs_dir import config_root as _shipped_configs
from woof.globe import sizing
from woof.globe.config import load_config


GIB = 2 ** 30
T383_DAY = str(_shipped_configs() / "arwen_global_gdas_t383_native_sl_si_24h.toml")
T533_DAY = str(_shipped_configs() / "arwen_global_gdas_t533_native_sl_si_24h.toml")
T383_EULERIAN_DAY = str(_shipped_configs() / "arwen_global_gdas_t383_native_imex_24h.toml")
ALL = ("physics", "surface", "tracers")


def _plan(config, free_gib):
    cfg = load_config(config)
    estimate = sizing.estimate_global_memory(cfg)
    estimate = dataclasses.replace(estimate, backend="cupy")
    return sizing.plan_run_memory(cfg, int(free_gib * GIB), estimate)


def test_the_34_7_km_day_is_admitted_on_the_16_gb_card_at_the_plan_that_completed():
    plan = _plan(T383_DAY, 15.28)
    assert plan.fits
    assert plan.bands == 2 and plan.spill_slices == ALL
    assert plan.live_peak_bytes == 13_322_662_912
    assert plan.card_bytes <= int(15.28 * GIB)
    assert plan.ceiling and "34.7 km day" in plan.ceiling
    assert "a completed run of this shape held" in plan.reason
    assert plan.receipt()["prediction_margin"] == 1.0
    assert plan.receipt()["live_peak_ceiling"] == plan.ceiling


def test_the_25_km_day_is_admitted_on_the_32_gb_card_at_the_plan_that_completed():
    plan = _plan(T533_DAY, 30.9)
    assert plan.fits
    assert plan.bands == 1 and plan.spill_slices == ALL
    # This tree's own day: 23.23 GiB live where the model reads 22.81, and
    # a row at exactly the plan is the figure, above or below the model.
    assert plan.live_peak_bytes == 24_940_350_464
    assert plan.ceiling and "25 km day" in plan.ceiling
    assert plan.receipt()["prediction_margin"] == 1.0
    # The pool ratio is the plan's own class, measured on this tree's
    # runs of it (x1.2102), not the largest of every class.
    assert plan.fragmentation[0] == pytest.approx(30_183_100_928 / 24_940_278_784)


def test_a_ceiling_never_raises_the_model_and_never_crosses_a_core():
    # T533 at 32 bands with every slice parked: the model reads under the
    # two-band completion, and the model stands.
    cfg = load_config(T533_DAY)
    estimate = dataclasses.replace(sizing.estimate_global_memory(cfg), backend="cupy")
    sizes = sizing._spill_census_estimate(cfg, estimate)
    model = sizing.banded_device_peak_bytes(estimate, 32, spilled_bytes=sum(sizes.values()))
    row = sizing.completed_plan_ceiling(estimate, 32, ALL)
    assert row is not None and int(row["peak_used_bytes"]) == 21_152_070_656
    assert model < int(row["peak_used_bytes"])
    # The Eulerian core has no row: it holds neither the second time level
    # nor the gather, so the semi-Lagrangian rows do not speak for it.
    eulerian = dataclasses.replace(sizing.estimate_global_memory(load_config(T383_EULERIAN_DAY)), backend="cupy")
    assert eulerian.trajectory_bytes == 0
    assert sizing.completed_plan_ceiling(eulerian, 8, ALL) is None


@pytest.mark.parametrize("bands, slices, chunk, expected", [
    (1, ALL, 5_000, 24_940_350_464),           # the one-band day IS one band's figure
    (2, ALL, 5_000, 21_152_070_656),           # the two-band probe is the smaller bound at two
    (4, ALL, 5_000, 21_152_070_656),           # and the lightest applicable row bounds every count above
    (2, ("physics",), 5_000, None),            # a plan that parks less is not bounded
    (2, ALL, 12_500, None),                    # a larger radiation chunk is not bounded by a smaller one's run
    (1, ("physics", "surface"), 5_000, None),  # the surface alone is not every slice
])
def test_a_row_applies_at_fewer_or_equal_bands_a_subset_of_the_slices_and_a_chunk_at_or_above(
        bands, slices, chunk, expected):
    estimate = SimpleNamespace(truncation=533, nlev=40, precision="float32",
                               radiation_column_chunk=chunk, trajectory_bytes=1)
    row = sizing.completed_plan_ceiling(estimate, bands, slices)
    if expected is None:
        assert row is None
    else:
        assert row is not None and int(row["peak_used_bytes"]) == expected


def test_every_row_is_a_completed_run_with_its_plan_and_its_pool_ratio_on_record():
    for row in sizing.COMPLETED_PLAN_PEAKS:
        assert row["core"] in ("semilag", "eulerian")
        assert int(row["bands"]) >= 1 and set(row["spill_slices"]) <= set(sizing.SPILL_SLICES)
        assert int(row["held_bytes"]) >= int(row["peak_used_bytes"]) > 0
        assert row["measured_on"] and row["device"] and "status pass" in row["run"] or "completions" in row["run"]


def test_an_exact_row_is_the_figure_even_when_the_model_reads_under_it():
    cfg = load_config(T533_DAY)
    estimate = dataclasses.replace(sizing.estimate_global_memory(cfg), backend="cupy")
    sizes = sizing._spill_census_estimate(cfg, estimate)
    model = sizing.banded_device_peak_bytes(estimate, 1, spilled_bytes=sum(sizes.values()))
    row = sizing.completed_plan_ceiling(estimate, 1, ALL)
    assert sizing.completed_plan_is_exact(estimate, 1, ALL, row)
    assert model < int(row["peak_used_bytes"]) == 24_940_350_464
    plan = sizing.plan_run_memory(cfg, int(30.9 * GIB), estimate)
    assert plan.live_peak_bytes == 24_940_350_464 and plan.bands == 1


def test_the_sixteen_band_t383_plan_is_priced_at_its_own_rows_not_a_larger_shapes_ratio():
    # MEASURED 2026-09-07 on the RTX 5070 Ti: T383 L40 at sixteen bands with
    # every slice parked completes at 9.73 GiB live and 11.87 held
    # (x1.2201), while the door priced it at 18.30 GiB of card by charging
    # the T533 sixteen-band row's x1.4149 on the two-band day's 12.41 GiB
    # live, and refused rank 1 of a two-card T383 run for it.
    cfg = dataclasses.replace(load_config(T383_DAY), latitude_bands=16)
    estimate = dataclasses.replace(sizing.estimate_global_memory(cfg), backend="cupy")
    ratio, phrase = sizing.pool_fragmentation_for(True, True, False, bands=16, truncation=383)
    assert abs(ratio - 12_746_315_776 / 10_446_974_464) < 1e-9
    assert "exactly T383 and 16 bands" in phrase
    larger, _ = sizing.pool_fragmentation_for(True, True, False, bands=16, truncation=533)
    assert larger > 1.4
    row = sizing.completed_plan_ceiling(estimate, 16, ALL)
    assert sizing.completed_plan_is_exact(estimate, 16, ALL, row)
    assert int(row["peak_used_bytes"]) == 10_446_974_464
    plan = sizing.plan_run_memory(cfg, int(15.28 * GIB), estimate)
    assert plan.fits and plan.bands == 16 and plan.spill_slices == ALL
    assert plan.live_peak_bytes == 10_446_974_464
    assert plan.card_bytes <= int(15.28 * GIB)
    assert plan.card_bytes < int(13.0 * GIB)
