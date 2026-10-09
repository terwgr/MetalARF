"""An inland lake's skin follows its own surface energy budget.

The breakage (GDAS 2026-09-30 12Z, T255 L40, 240 h): open water held the
analysis skin for the whole run.  On the ocean that is the analysis SST, the
usual medium-range choice.  On an inland lake it is the analysis's lake
temperature, which GDAS carried 9.2 K above the 2 m air over Lake Tana
(11.9 N, 37.5 E) at all eight cycles of 2026-09-29 and 2026-09-30.  Held,
that column evaporated about 22 kg/m2 a day for 240 h (published estimates
of the lake's open-water evaporation are about 4 to 5 mm a day), and the
warm lakes and coasts were the columns whose surface reservoirs would next
reach the refusal "native physics water closure exceeds the explicit surface
reservoir", about 23 days out.

A lake column (WRF's LAKEMASK rule on the statics' lake share) now integrates
its skin on the land cadence from the energy it exchanges, with the water
heat capacity the state carries; the ocean, land and frozen columns are
untouched.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from woof.globe.constants import LATENT_HEAT_VAPORIZATION, STEFAN_BOLTZMANN
from woof.globe.physics.native_runtime import LAKE_FREEZING_K
from woof.globe.physics.native_suite import ArwenCudaColumnSuite
from woof.globe.statics import (
    SURFACE_STATIC_FIELDS, lake_columns, synthetic_surface_statics, water_columns,
)
from woof.globe.state import SEEDED_SURFACE_MEMBERS

from test_arwen_global_level5_native import (
    _authority_sfclay_modules,
    _exchange,
    _options,
    _warm_ocean,
)


def test_a_lake_is_a_water_column_whose_water_is_mostly_lake():
    # (land, lake, ocean) shares of one column each, and the analysed ice.
    columns = [
        ((0.445, 0.555, 0.000), 0.0, True),   # Lake Tana's T255 column
        ((0.485, 0.515, 0.000), 0.0, True),   # Lake Turkana's north basin
        ((0.300, 0.400, 0.300), 0.0, True),   # a lake beside a little sea
        ((0.400, 0.200, 0.400), 0.0, False),  # a coast: the sea wins
        ((0.000, 0.000, 1.000), 0.0, False),  # the open ocean
        ((0.600, 0.400, 0.000), 0.0, False),  # land with a lake in it
        ((0.000, 1.000, 0.000), 0.9, False),  # a lake the analysis covers with ice
    ]
    land = np.array([[c[0][0] for c in columns]], np.float32)
    lake = np.array([[c[0][1] for c in columns]], np.float32)
    ice = np.array([[c[1] for c in columns]], np.float32)
    expected = np.array([[c[2] for c in columns]])
    found = lake_columns(land, lake, sea_ice_fraction=ice)
    assert np.array_equal(found, expected)
    # A lake is always one of the runtime's open-water columns.
    assert not np.any(found & ~water_columns(land, sea_ice_fraction=ice))


def test_the_statics_seed_a_lake_share_and_the_planet_without_lakes_has_none():
    assert "lake_fraction" in SURFACE_STATIC_FIELDS
    # A checkpoint from before the plane reads as a planet without lakes
    # (checkpoint.py fills the seeded members it lacks with zero).
    assert "lake_fraction" in SEEDED_SURFACE_MEMBERS
    land = np.array([[0.0, 0.2, 0.9]])
    planet = synthetic_surface_statics(land, np.full((4, 1, 3), 290.0))
    assert not np.any(planet["lake_fraction"])


def _lake_exchange(*, skin_k=305.0, air_theta_k=296.0, qv=0.012):
    """The smoke exchange with every water column a warm lake or sea under
    cooler, drier air (the Lake Tana shape), and every other water column
    made a lake: the two halves run the same surface layer and differ only
    in the statics' lake share."""
    exchange = _warm_ocean(
        _exchange(), air_theta_k=air_theta_k, skin_k=skin_k, qv=qv, wind_m_s=6.0,
    )
    surface = exchange.surface
    land = np.asarray(surface.land_fraction)
    water = water_columns(land)
    rows, cols = np.nonzero(water)
    lake = np.zeros_like(water)
    lake[rows[::2], cols[::2]] = True
    surface.lake_fraction[...] = np.where(lake, 1.0 - land, 0.0).astype(
        surface.lake_fraction.dtype
    )
    assert lake.any() and (water & ~lake).any()
    return exchange, water, lake


def test_a_lake_skin_follows_its_energy_budget_and_the_sea_is_held():
    exchange, water, lake = _lake_exchange()
    surface = exchange.surface
    skin = np.array(surface.temperature_k, dtype=np.float32, copy=True)
    calls = []
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_authority_sfclay_modules(calls)
    )
    result = suite.step(exchange)
    after = np.asarray(result.surface.temperature_k, dtype=np.float32)
    f = result.physics_state.arrays

    # The sea keeps the skin it started with, bit for bit.
    sea = water & ~lake
    assert np.array_equal(after[sea], skin[sea])

    # A lake's skin moved by the energy it exchanged over the land interval
    # (the first due land call integrates the configured interval), with
    # the surface layer's own fluxes of this call.
    elapsed = float(_options()["land_surface_interval_s"])
    emissivity = np.asarray(surface.emissivity, np.float64)
    albedo = np.asarray(surface.albedo, np.float64)
    net = (
        np.asarray(f["swdown"], np.float64) * (1.0 - albedo)
        + emissivity * np.asarray(f["glw"], np.float64)
        - emissivity * STEFAN_BOLTZMANN * np.asarray(skin, np.float64) ** 4
        - np.asarray(f["hfx"], np.float64)
        - LATENT_HEAT_VAPORIZATION * np.asarray(f["qfx"], np.float64)
    )
    expected = skin + elapsed * net / np.asarray(surface.heat_capacity_j_m2_k, np.float64)
    assert np.allclose(after[lake], expected[lake], rtol=0.0, atol=2.0e-5)

    # The surface layer saw a warm, wet surface under cooler, drier air: it
    # evaporates and heats the air, so the lake loses energy and cools.
    assert float(np.asarray(f["qfx"])[lake].min()) > 0.0
    assert float(np.asarray(f["hfx"])[lake].min()) > 0.0
    assert float((after - skin)[lake].max()) < 0.0
    assert result.diagnostics["lake_surface_columns"] == int(lake.sum())
    assert int(result.physics_state.metadata["lake_surface_calls"]) == 1


def test_a_warm_lake_keeps_cooling_and_its_evaporation_falls():
    exchange, water, lake = _lake_exchange()
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_authority_sfclay_modules([])
    )
    first = suite.step(exchange)
    qfx_first = np.asarray(first.physics_state.arrays["qfx"], np.float64)[lake]
    skin_first = np.asarray(first.surface.temperature_k, np.float64)[lake]
    # Lower the skin by what a few days of that loss would take off and
    # call again: the same lake evaporates less from a cooler surface.
    cooled = exchange.surface.copy()
    cooled.temperature_k[...] = np.where(
        lake, cooled.temperature_k - cooled.temperature_k.dtype.type(6.0),
        cooled.temperature_k,
    )
    start = np.asarray(cooled.temperature_k, np.float64)[lake]
    later = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_authority_sfclay_modules([])
    ).step(replace(exchange, surface=cooled))
    qfx_later = np.asarray(later.physics_state.arrays["qfx"], np.float64)[lake]
    assert float(np.median(qfx_later / qfx_first)) < 0.8
    # Still warmer than the air above it, so still cooling, more slowly.
    step_later = np.asarray(later.surface.temperature_k, np.float64)[lake] - start
    step_first = skin_first - np.asarray(exchange.surface.temperature_k, np.float64)[lake]
    assert np.all(step_later < 0.0)
    assert float(np.median(step_later / step_first)) < 1.0


def test_a_lake_is_not_cooled_below_freshwater_freezing():
    exchange, water, lake = _lake_exchange(skin_k=273.155, air_theta_k=250.0, qv=0.0005)
    surface = exchange.surface
    below = np.zeros_like(lake)
    rows, cols = np.nonzero(lake)
    below[rows[:1], cols[:1]] = True
    # One lake starts below the floor (an analysis can carry that): the
    # floor never warms it, and it is not cooled further either.
    surface.temperature_k[...] = np.where(
        below, surface.temperature_k.dtype.type(272.5), surface.temperature_k
    )
    result = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_authority_sfclay_modules([])
    ).step(exchange)
    after = np.asarray(result.surface.temperature_k, np.float64)
    assert float(after[lake & ~below].min()) >= LAKE_FREEZING_K - 1.0e-4
    assert float(after[lake & ~below].max()) <= 273.155 + 1.0e-4
    assert np.allclose(after[below], 272.5, atol=1.0e-4)


def test_a_planet_without_lakes_runs_as_it_did():
    exchange = _warm_ocean(_exchange(), wind_m_s=6.0)
    assert not np.any(np.asarray(exchange.surface.lake_fraction))
    skin = np.array(exchange.surface.temperature_k, copy=True)
    water = water_columns(np.asarray(exchange.surface.land_fraction))
    result = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_authority_sfclay_modules([])
    ).step(exchange)
    assert np.array_equal(np.asarray(result.surface.temperature_k)[water], skin[water])
    assert result.diagnostics["lake_surface_columns"] == 0
    assert "lake_surface_calls" not in result.physics_state.metadata
