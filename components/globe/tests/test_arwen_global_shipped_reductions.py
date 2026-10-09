"""Gate RED-1 on the SHIPPED call sites, not on a copy of them.

``tools/arwen_global_reduction_probe.py`` and
``tests/test_arwen_global_bands.py`` both build their own whole and banded
forms out of the accumulators, so between them they prove that
:mod:`woof.globe.bands` is band-invariant.  Neither of them calls
the model.  A reduction whose accumulator is perfect and whose CALL SITE
folds an extra axis inside the band is exactly the defect RED-1 refused
once already, and it would pass both of those instruments.

So these tests drive the model's own methods: the whole-grid method the
resident path runs today against the same method's band-local stage fed
band by band, on a real model built from a shipped config.  They are the
tests Lane 3 inherits, because Lane 3 is the caller that will hand these
seams real bands.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from pathlib import Path

import dataclasses
import numpy as np
import pytest

from woof.globe.bands import LatitudeAccumulator, band_slices
from woof.globe.config import load_config
from woof.globe.constants import GRAVITY_M_S2
from woof.globe.runner import build_model_and_cold_state

ROOT = Path(__file__).resolve().parents[1]
CONFIG = _shipped_configs() / "arwen_global_t21_baroclinic_ten_day.toml"

BAND_COUNTS = (1, 2, 3, 5, 8, 16, "nlat")


@pytest.fixture(scope="module")
def model_and_state():
    cfg = load_config(CONFIG)
    cfg = dataclasses.replace(cfg, duration_s=cfg.dt_s, output_interval_s=cfg.dt_s)
    model, bundle = build_model_and_cold_state(cfg)
    return model, bundle


def _counts(nlat):
    return [nlat if c == "nlat" else c for c in BAND_COUNTS if c == "nlat" or c <= nlat]


def test_the_shipped_level_water_mass_is_the_same_at_every_band_count(model_and_state):
    """``_level_water_mass`` band by band through its own two stages."""
    model, bundle = model_and_state
    xp = model.transform.backend.xp
    g = model.grid_state(bundle.atmosphere, only=("qv", "dp"))
    stacked = xp.stack([g["qv"], g["qv"] * 0.5 + 1.0e-6])
    dp = g["dp"]
    nlat = int(stacked.shape[-2])
    whole = model._level_water_mass(stacked, dp)
    rows_whole = model._level_water_mass_rows(stacked, dp)
    for count in _counts(nlat):
        accumulator = LatitudeAccumulator(
            xp, rows_whole.shape, rows_whole.dtype, name="water",
        )
        for rows in band_slices(nlat, count):
            accumulator.add_band(
                rows,
                model._level_water_mass_rows(
                    stacked[..., rows, :], dp[..., rows, :]
                ),
            )
        banded = model._level_water_mass_total(accumulator)
        assert np.array_equal(
            np.asarray(banded), np.asarray(whole)
        ), f"_level_water_mass moved at B={count}"


def test_the_shipped_column_mass_is_the_same_at_every_band_count(model_and_state):
    """``GridTracerTransport._global_mean_columns`` band by band."""
    model, bundle = model_and_state
    xp = model.transform.backend.xp
    transport = model.transport
    g = model.grid_state(bundle.atmosphere, only=("qv", "dp"))
    value = xp.stack([g["qv"], g["qv"] * 0.25])
    density = g["dp"]
    nlat = int(value.shape[-2])
    whole = transport._global_mean_columns(value, density)
    rows_whole = transport._global_mean_columns_rows(value, density)
    for count in _counts(nlat):
        accumulator = LatitudeAccumulator(
            xp, rows_whole.shape, rows_whole.dtype, name="columns",
        )
        for rows in band_slices(nlat, count):
            accumulator.add_band(
                rows,
                transport._global_mean_columns_rows(
                    value[..., rows, :], density[..., rows, :], rows
                ),
            )
        banded = accumulator.total() / GRAVITY_M_S2
        assert np.array_equal(
            np.asarray(banded), np.asarray(whole)
        ), f"_global_mean_columns moved at B={count}"


def test_the_transport_gap_mean_is_the_step_s_second_flat_three_d_sum(
    model_and_state,
):
    """``_transport_grid_tracers``' pseudo-density gap mean.

    It was NOT in the eleven-reduction inventory and it is the same class
    as the vapor mass readings: a flat sum over (level, latitude,
    longitude).  Two-stage here, and band-invariant at every count down to
    one row a band; the flat form it replaces is not, which is why the
    reading moves by a few ulp and is published as moving.
    """
    model, bundle = model_and_state
    xp = model.transform.backend.xp
    g = model.grid_state(bundle.atmosphere, only=("dp",))
    gap = xp.abs(g["dp"] - float(np.asarray(g["dp"]).mean())) / g["dp"]
    nlev, nlat, nlon = gap.shape
    cell = model.transform.backend.asarray(
        model.transform.grid.quadrature_weights,
        dtype=model.transform.backend.float_dtype,
    )[None, :, None] / (2.0 * nlon)
    reference = None
    for count in _counts(nlat):
        accumulator = LatitudeAccumulator(
            xp, (nlev, nlat), gap.dtype, name="gap",
        )
        for rows in band_slices(nlat, count):
            accumulator.add_band(
                rows, xp.sum(gap[..., rows, :] * cell[..., rows, :], axis=-1)
            )
        banded = accumulator.total(axis=None) / nlev
        if reference is None:
            reference = banded
        assert np.array_equal(
            np.asarray(banded), np.asarray(reference)
        ), f"the transport gap mean moved at B={count}"
