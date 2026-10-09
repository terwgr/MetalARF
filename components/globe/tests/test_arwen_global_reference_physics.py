from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import math

import numpy as np
import pytest

from woof.globe.config import load_config
from woof.globe.constants import (
    DRY_AIR_CP,
    GRAVITY_M_S2,
    LATENT_HEAT_FUSION,
    LATENT_HEAT_VAPORIZATION,
    WATER_SPECIES,
)
from woof.globe.physics.reference import (
    ReferencePhysics,
    ReferencePhysicsOptions,
    _implicit_diffuse,
    saturation_mixing_ratio,
)
from woof.globe.runner import build_model_and_cold_state
from woof.globe.water import native_store_column, soil_water_column


CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


def _column_water(exchange, fields):
    return np.sum(sum(fields[name] for name in WATER_SPECIES) * exchange.dp / GRAVITY_M_S2, axis=0)


def _model_and_exchange(dt_s):
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    return cfg, model, state, model._physics_exchange(state, dt_s)


def test_reference_physics_water_repair_is_roundoff_and_ledger_closes():
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    exchange = model._physics_exchange(state, 0.5 * cfg.dt_s)
    before = _column_water(exchange, exchange.water()) + exchange.surface.water_kg_m2
    result = model.physics.step(exchange)
    # The operators must conserve local water on their own; the closing
    # repair only absorbs roundoff. Bound: ~100 ulp of the ~5e2 kg/m2 column
    # ledger (ulp 5.7e-14); measured 2026-08-30 with conservative diffusion:
    # 1.14e-13 kg/m2. The unfixed diffusion operator booked 8.2e-4 kg/m2.
    assert result.diagnostics["maximum_local_water_repair_kg_m2"] < 1.0e-11
    # And the repair arithmetic itself must leave the books closed: a few
    # ulp of the ~5e2 kg/m2 column total from re-associating the closing
    # subtraction (2e-12 is ~35 ulp).
    after = _column_water(exchange, result.water()) + result.surface.water_kg_m2
    assert np.max(np.abs(after - before)) < 2.0e-12
    assert result.adapter_receipt["mode"] == "reference"
    assert np.all(result.qv >= 0.0)
    assert np.all(result.qc >= 0.0)


def test_longwave_only_cools_upper_levels_and_olr_is_positive():
    _, model, _, exchange = _model_and_exchange(600.0)
    # Dry resting isothermal atmosphere, longwave only: no shortwave
    # absorption, no surface exchange, no moisture.
    exchange.theta[...] = 260.0 / exchange.exner
    exchange.u[...] = 0.0
    exchange.v[...] = 0.0
    for name in WATER_SPECIES:
        getattr(exchange, name)[...] = 0.0
    options = ReferencePhysicsOptions(
        radiation=True,
        surface_fluxes=False,
        turbulence=False,
        convection=False,
        saturation_adjustment=False,
        microphysics=False,
        # Zero absorptivity routes all shortwave to the SURFACE, and with
        # surface fluxes off the surface cannot pass it to the atmosphere
        # within the step, so the one-step atmospheric tendency measured
        # below is pure longwave; the options gate refuses an actual zero
        # solar constant.
        atmospheric_shortwave_absorptivity=0.0,
    )
    physics = ReferencePhysics(model.transform.backend, options)
    result = physics.step(exchange)
    dtemp = (result.theta - exchange.theta) * exchange.exner
    # Each layer emits both up and down; an isothermal column above a
    # <=285 K surface absorbs less than 2*eps*sigma*260^4 everywhere, so
    # the upper levels must lose energy to space.
    assert np.all(dtemp[0] < 0.0)
    assert np.all(dtemp[1] < 0.0)
    assert result.diagnostics["mean_outgoing_longwave_w_m2"] > 0.0


def test_warm_supersaturated_parcel_condenses_to_analytic_adjustment():
    _, model, _, _ = _model_and_exchange(600.0)
    pressure = np.full((1, 1, 1), 1.0e5)
    exner = np.ones((1, 1, 1))
    t0 = 300.0
    qsat0 = float(saturation_mixing_ratio(np.full((1, 1, 1), t0), pressure, np)[0, 0, 0])
    supersaturation = 2.0e-3
    q = {name: np.zeros((1, 1, 1)) for name in WATER_SPECIES}
    q["qv"][:] = qsat0 + supersaturation
    physics = ReferencePhysics(model.transform.backend)
    theta, condensed = physics._saturation_adjust(
        np.full((1, 1, 1), t0), q, exner, pressure, np
    )
    t_final = float(theta[0, 0, 0])
    qc_final = float(q["qc"][0, 0, 0])
    qv_final = float(q["qv"][0, 0, 0])
    assert qc_final > 0.0
    assert float(condensed[0, 0]) > 0.0
    # Analytic isobaric adjustment: bisect the moist adiabat
    # qv0 - (cp/L)(T - T0) = qsat(T) for the equilibrium temperature.
    cp_over_l = DRY_AIR_CP / LATENT_HEAT_VAPORIZATION

    def excess(t):
        qsat = float(saturation_mixing_ratio(np.full((1, 1, 1), t), pressure, np)[0, 0, 0])
        return (qsat0 + supersaturation) - cp_over_l * (t - t0) - qsat

    low, high = t0, t0 + 10.0
    assert excess(low) > 0.0 > excess(high)
    for _ in range(200):
        mid = 0.5 * (low + high)
        if excess(mid) > 0.0:
            low = mid
        else:
            high = mid
    t_star = 0.5 * (low + high)
    qc_star = cp_over_l * (t_star - t0)
    assert abs(qc_final - qc_star) <= 0.01 * qc_star
    assert abs(t_final - t_star) <= 0.01 * (t_star - t0)
    qsat_final = float(
        saturation_mixing_ratio(np.full((1, 1, 1), t_final), pressure, np)[0, 0, 0]
    )
    assert abs(qv_final - qsat_final) < 1.0e-10


def test_warm_column_melts_falling_snow_and_books_fusion_both_directions():
    _, model, _, exchange = _model_and_exchange(600.0)
    exchange.theta[...] = 290.0 / exchange.exner
    for name in WATER_SPECIES:
        getattr(exchange, name)[...] = 0.0
    exchange.qs[1] = 5.0e-4
    options = ReferencePhysicsOptions(
        radiation=False,
        surface_fluxes=False,
        turbulence=False,
        convection=False,
        saturation_adjustment=False,
        microphysics=True,
    )
    physics = ReferencePhysics(model.transform.backend, options)
    result = physics.step(exchange)
    # Snow at +17 C must melt into rain, and the melt must debit the fusion
    # heat from the melting layer (Lf/cp * melted ~ 0.1 K here).
    assert float(np.max(result.qr)) > 1.0e-4
    layer_dtemp = (result.theta[1] - exchange.theta[1]) * exchange.exner[1]
    assert np.all(layer_dtemp < -0.05)
    # Frozen mass reaching the liquid reservoir pays its fusion heat out of
    # the surface energy budget ONLY where the surface is above freezing;
    # a frozen surface cannot melt it and takes no debit. Tolerance: a few
    # ulp of the ~280 K surface temperature (1e-12 is ~35 ulp).
    snow_at_surface = model.transform.backend.to_numpy(
        result.surface.accumulated_snow_kg_m2
    )
    assert float(np.max(snow_at_surface)) > 0.0
    surface_before = model.transform.backend.to_numpy(
        exchange.surface.temperature_k
    )
    surface_dtemp = model.transform.backend.to_numpy(
        result.surface.temperature_k
    ) - surface_before
    heat_capacity = model.transform.backend.to_numpy(
        exchange.surface.heat_capacity_j_m2_k
    )
    expected = np.where(
        surface_before > 273.15,
        -LATENT_HEAT_FUSION * snow_at_surface / np.maximum(heat_capacity, 1.0e4),
        0.0,
    )
    assert np.any(surface_before > 273.15)
    assert np.any(surface_before <= 273.15)
    assert float(np.max(np.abs(surface_dtemp - expected))) < 1.0e-12


def test_sublimation_cannot_supersaturate_or_convert_ice_to_liquid():
    # An uncorrected deficit cap let sublimation consume the full deficit
    # while its own Ls/cp cooling dropped qsat underneath it, leaving layers
    # at RH 1.06..1.11 in one step; with the default saturation adjustment
    # that excess condensed as LIQUID cloud 9 K below freezing.
    for adjust in (False, True):
        _, model, _, exchange = _model_and_exchange(600.0)
        exchange.theta[...] = 265.0 / exchange.exner
        for name in WATER_SPECIES:
            getattr(exchange, name)[...] = 0.0
        qsat0 = saturation_mixing_ratio(
            exchange.theta * exchange.exner, exchange.p_full, np
        )
        # Capped: near the model top qsat is O(0.1) kg/kg and 90% of it is
        # not an atmosphere; the feedback bound is asserted in the regime
        # where the defect was measured (RH 0.9 at real pressures).
        exchange.qv[...] = np.minimum(0.9 * qsat0, 0.02)
        exchange.qi[...] = 2.0e-3
        options = ReferencePhysicsOptions(
            radiation=False,
            surface_fluxes=False,
            turbulence=False,
            convection=False,
            saturation_adjustment=adjust,
            microphysics=True,
        )
        physics = ReferencePhysics(model.transform.backend, options)
        result = physics.step(exchange)
        t_final = result.theta * exchange.exner
        qsat_final = saturation_mixing_ratio(t_final, exchange.p_full, np)
        if adjust:
            # No spurious ice-to-liquid conversion below freezing.
            assert float(np.max(result.qc)) < 1.0e-9
        else:
            # The linearized feedback leaves at most second-order residual
            # supersaturation (0.5% covers it; the defect measured 6-11%).
            assert np.all(result.qv <= qsat_final * 1.005 + 1.0e-12)


def test_vertical_diffusion_conserves_mass_weighted_column_on_nonuniform_grid():
    # Direct operator check: strongly non-uniform dp, sum(q*dp) invariant to
    # solver roundoff (measured 2026-08-30: 2e-16 relative; the unfixed
    # operator conserved the unweighted sum instead).
    rng = np.random.default_rng(7)
    dp = np.array([600.0, 2000.0, 5000.0, 9000.0, 30000.0, 53400.0])[:, None, None] * np.ones((1, 3, 4))
    height = np.array([30000.0, 18000.0, 12000.0, 8000.0, 4000.0, 500.0])[:, None, None] * np.ones((1, 3, 4))
    diffusivity = 5.0 + 25.0 * np.exp(-(height - 500.0) / 1800.0)
    field = rng.uniform(0.0, 0.01, size=(6, 3, 4))
    before = np.sum(field * dp, axis=0)
    after = np.sum(_implicit_diffuse(field, height, dp, diffusivity, 600.0, np) * dp, axis=0)
    assert float(np.max(np.abs(after - before) / before)) < 1.0e-13
    # Step-level: a turbulence-only step on the model's hybrid grid must not
    # need the water-repair ledger (unfixed: 8.2e-4 kg/m2 at dt=600 s).
    _, model, _, exchange = _model_and_exchange(600.0)
    options = ReferencePhysicsOptions(
        radiation=False,
        surface_fluxes=False,
        turbulence=True,
        convection=False,
        saturation_adjustment=False,
        microphysics=False,
    )
    physics = ReferencePhysics(model.transform.backend, options)
    result = physics.step(exchange)
    assert result.diagnostics["maximum_local_water_repair_kg_m2"] < 1.0e-11


def test_driven_negative_stores_refuse_and_roundoff_ring_clamps():
    _, model, state, _ = _model_and_exchange(600.0)
    surface = state.surface.copy()
    surface.soil_water_fraction[0, 0, 0] = -5.0
    with pytest.raises(FloatingPointError, match="soil_water_fraction driven negative"):
        soil_water_column(surface, np)
    ringing = state.surface.copy()
    ringing.soil_water_fraction[0, 0, 0] = -1.0e-12
    assert float(np.min(soil_water_column(ringing, np))) >= 0.0
    physics_state = state.physics_state.copy()
    physics_state.arrays["snow"] = np.zeros_like(
        model.transform.backend.to_numpy(state.surface.water_kg_m2)
    )
    physics_state.arrays["snow"][0, 0] = -1.0
    with pytest.raises(FloatingPointError, match="snow driven negative"):
        native_store_column(physics_state, state.surface.water_kg_m2, np)


def test_disabled_radiation_does_not_heat_surface_by_itself():
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    exchange = model._physics_exchange(state, 5.0)
    options = ReferencePhysicsOptions(
        radiation=False,
        surface_fluxes=False,
        turbulence=False,
        convection=False,
        moist_convection=False,
        saturation_adjustment=False,
        microphysics=False,
    )
    physics = ReferencePhysics(model.transform.backend, options)
    result = physics.step(exchange)
    assert np.array_equal(result.theta, exchange.theta)
    assert np.array_equal(result.qv, exchange.qv)
    assert np.array_equal(result.surface.temperature_k, exchange.surface.temperature_k)


def test_reference_physics_transaction_does_not_mutate_exchange():
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    exchange = model._physics_exchange(state, 5.0)
    theta = np.array(exchange.theta, copy=True)
    water = {name: np.array(getattr(exchange, name), copy=True) for name in WATER_SPECIES}
    surface = np.array(exchange.surface.water_kg_m2, copy=True)
    model.physics.step(exchange)
    assert np.array_equal(exchange.theta, theta)
    for name in WATER_SPECIES:
        assert np.array_equal(getattr(exchange, name), water[name])
    assert np.array_equal(exchange.surface.water_kg_m2, surface)


def _tropical_test_column(qv_surface):
    """A deep conditionally unstable column: stable to dry ascent, buoyant
    to a saturated parcel carrying qv_surface, dry aloft (the
    non-precipitating regime)."""
    from types import SimpleNamespace

    nlev = 20
    p_half = np.linspace(5_000.0, 100_000.0, nlev + 1)[:, None, None]
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    dp = p_half[1:] - p_half[:-1]
    exner = (p_full / 100_000.0) ** (2.0 / 7.0)
    temperature = 302.0 * (p_full / p_full[-1]) ** 0.23
    qv = np.full_like(temperature, 1.0e-4)
    qv[-4:] = qv_surface
    q = {name: np.zeros_like(temperature) for name in WATER_SPECIES}
    q["qv"] = qv
    exchange = SimpleNamespace(
        dt_s=300.0,
        exner=exner,
        p_full=p_full,
        dp=dp,
        virtual_temperature=temperature * (1.0 + 0.61 * qv),
    )
    surface = SimpleNamespace(water_kg_m2=np.full((1, 1), 500.0))
    return exchange, temperature / exner, q, surface


def _near_saturated_low_cape_column():
    """Small CAPE (~2 K) over a near-saturated column: the precipitating
    regime, the state grid-point storms grow from."""
    from types import SimpleNamespace
    from woof.globe.constants import (
        DRY_AIR_GAS_CONSTANT,
        EPSILON,
        LATENT_HEAT_VAPORIZATION,
    )
    from woof.globe.physics.reference import saturation_mixing_ratio

    nlev = 20
    p_half = np.linspace(5_000.0, 100_000.0, nlev + 1)[:, None, None]
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    dp = p_half[1:] - p_half[:-1]
    exner = (p_full / 100_000.0) ** (2.0 / 7.0)
    # Environment: a moist adiabat from 300 K cooled 2 K aloft.
    temperature = np.empty_like(p_full)
    temperature[-1] = 300.0
    for k in range(nlev - 2, -1, -1):
        qs = saturation_mixing_ratio(temperature[k + 1], p_full[k + 1], np)
        rate = (
            DRY_AIR_GAS_CONSTANT * temperature[k + 1]
            + LATENT_HEAT_VAPORIZATION * qs
        ) / (
            DRY_AIR_CP
            + LATENT_HEAT_VAPORIZATION**2
            * qs
            * EPSILON
            / (DRY_AIR_GAS_CONSTANT * temperature[k + 1] ** 2)
        )
        temperature[k] = temperature[k + 1] + rate * np.log(
            p_full[k] / p_full[k + 1]
        )
    temperature[:-1] -= 2.0
    qv = 0.95 * saturation_mixing_ratio(temperature, p_full, np)
    q = {name: np.zeros_like(temperature) for name in WATER_SPECIES}
    q["qv"] = qv
    exchange = SimpleNamespace(
        dt_s=300.0,
        exner=exner,
        p_full=p_full,
        dp=dp,
        virtual_temperature=temperature * (1.0 + 0.61 * qv),
    )
    surface = SimpleNamespace(water_kg_m2=np.full((1, 1), 500.0))
    return exchange, temperature / exner, q, surface


def test_betts_miller_precipitating_regime_closes_water_and_enthalpy():
    _, model, _, _ = _model_and_exchange(300.0)
    exchange, theta, q, surface = _near_saturated_low_cape_column()
    qv_before = q["qv"].copy()
    theta_before = theta.copy()
    water_before = surface.water_kg_m2.copy()
    theta_after, q, surface, precip, deep_columns = (
        model.physics._betts_miller_adjust(exchange, theta, q, surface, np)
    )
    assert deep_columns == 1
    assert float(precip[0, 0]) > 0.0
    mass = exchange.dp / GRAVITY_M_S2
    # Water: every kg the column dries lands on the surface reservoir.
    column_change = np.sum((q["qv"] - qv_before) * mass, axis=0)
    reservoir_change = surface.water_kg_m2 - water_before
    np.testing.assert_allclose(column_change + reservoir_change, 0.0, atol=1e-10)
    np.testing.assert_allclose(reservoir_change, precip, atol=1e-12)
    assert np.all(q["qv"] >= 0.0)
    # Enthalpy: cp * integral(dT dm) == L * P exactly by the closure.
    heating = np.sum(
        (theta_after - theta_before) * exchange.exner * mass, axis=0
    )
    np.testing.assert_allclose(
        DRY_AIR_CP * heating,
        LATENT_HEAT_VAPORIZATION * precip,
        rtol=1e-9,
    )


def test_betts_miller_nonprecipitating_regime_is_water_and_enthalpy_neutral():
    _, model, _, _ = _model_and_exchange(300.0)
    exchange, theta, q, surface = _tropical_test_column(qv_surface=0.018)
    qv_before = q["qv"].copy()
    theta_before = theta.copy()
    water_before = surface.water_kg_m2.copy()
    theta_after, q, surface, precip, deep_columns = (
        model.physics._betts_miller_adjust(exchange, theta, q, surface, np)
    )
    # Deep instability is still adjusted, but with no water source the
    # column books to zero: no rain, no net water, no net enthalpy.
    assert deep_columns == 1
    assert float(np.max(np.abs(precip))) == 0.0
    np.testing.assert_array_equal(surface.water_kg_m2, water_before)
    mass = exchange.dp / GRAVITY_M_S2
    column_change = np.sum((q["qv"] - qv_before) * mass, axis=0)
    np.testing.assert_allclose(column_change, 0.0, atol=1e-10)
    assert np.all(q["qv"] >= 0.0)
    heating = np.sum(
        (theta_after - theta_before) * exchange.exner * mass, axis=0
    )
    np.testing.assert_allclose(DRY_AIR_CP * heating, 0.0, atol=1e-6)
    # And it genuinely moved the profile toward the reference.
    assert float(np.max(np.abs(theta_after - theta_before))) > 0.0


def test_betts_miller_leaves_dry_stable_column_untouched():
    _, model, _, _ = _model_and_exchange(300.0)
    exchange, theta, q, surface = _tropical_test_column(qv_surface=1.0e-4)
    theta_before = theta.copy()
    qv_before = q["qv"].copy()
    theta_after, q, surface, precip, deep_columns = (
        model.physics._betts_miller_adjust(exchange, theta, q, surface, np)
    )
    assert deep_columns == 0
    assert float(np.max(np.abs(precip))) == 0.0
    np.testing.assert_array_equal(theta_after, theta_before)
    np.testing.assert_array_equal(q["qv"], qv_before)


def test_betts_miller_disabled_is_identity():
    from woof.globe.physics.reference import (
        ReferencePhysics,
        ReferencePhysicsOptions,
    )

    _, model, _, _ = _model_and_exchange(300.0)
    physics = ReferencePhysics(
        model.physics.backend,
        ReferencePhysicsOptions(moist_convection=False),
    )
    exchange, theta, q, surface = _tropical_test_column(qv_surface=0.018)
    theta_before = theta.copy()
    theta_after, q, surface, precip, deep_columns = (
        physics._betts_miller_adjust(exchange, theta, q, surface, np)
    )
    assert deep_columns == 0
    assert float(np.max(np.abs(precip))) == 0.0
    np.testing.assert_array_equal(theta_after, theta_before)


def test_dry_land_evaporates_at_soil_beta_while_ocean_runs_potential():
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    exchange = model._physics_exchange(state, 60.0)
    xp = np
    # Force a strongly evaporative state: warm wet skin, dry air above.
    exchange.surface.temperature_k[...] = 300.0
    exchange.qv[-1][...] = 1.0e-3
    exchange.surface.water_kg_m2[...] = 500.0
    desert = exchange.surface.soil_water_fraction.copy()
    desert[...] = 0.03
    exchange.surface.soil_water_fraction[...] = desert
    exchange.surface.land_fraction[...] = 1.0
    physics = model.physics
    result_land = physics.step(exchange)
    evap_land = result_land.diagnostics["mean_surface_evaporation_kg_m2_s"]

    exchange2 = model._physics_exchange(state, 60.0)
    exchange2.surface.temperature_k[...] = 300.0
    exchange2.qv[-1][...] = 1.0e-3
    exchange2.surface.water_kg_m2[...] = 500.0
    exchange2.surface.soil_water_fraction[...] = 0.03
    exchange2.surface.land_fraction[...] = 0.0
    result_ocean = physics.step(exchange2)
    evap_ocean = result_ocean.diagnostics["mean_surface_evaporation_kg_m2_s"]

    # beta = 0.03 / 0.30 = 0.1: desert land evaporates at ~10% of the
    # ocean's potential rate under identical forcing.
    assert evap_ocean > 0.0
    ratio = evap_land / evap_ocean
    assert 0.05 < ratio < 0.2, f"desert/ocean evaporation ratio {ratio}"


def test_reference_suite_takes_albedo_and_roughness_from_the_statics():
    """The reference suite's surface radiation and bulk fluxes read the
    surface state's albedo and roughness, which woof.globe.statics
    seeds (here the smoke config's declared-by-default synthetic planet:
    the land-fraction formulas), so a different planet changes the
    fluxes and nothing else supplies them."""
    from woof.globe.statics import synthetic_surface_statics

    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    exchange = model._physics_exchange(state, 60.0)
    planet = synthetic_surface_statics(
        np.asarray(exchange.surface.land_fraction),
        np.asarray(exchange.surface.soil_temperature_k),
    )
    assert np.array_equal(np.asarray(exchange.surface.albedo), planet["albedo"])
    assert np.array_equal(np.asarray(exchange.surface.roughness_m), planet["roughness_m"])

    def arm(**surface_overrides):
        arm_exchange = model._physics_exchange(state, 60.0)
        arm_exchange.surface.temperature_k[...] = 300.0
        arm_exchange.qv[-1][...] = 1.0e-3
        arm_exchange.surface.water_kg_m2[...] = 500.0
        for name, value in surface_overrides.items():
            getattr(arm_exchange.surface, name)[...] = value
        return model.physics.step(arm_exchange)

    # Albedo: only the sunlit columns' skin temperature moves with it.
    dark, bright = arm(albedo=0.1), arm(albedo=0.9)
    latitude = np.deg2rad(exchange.latitude_deg)
    longitude = np.deg2rad(exchange.longitude_deg)
    hour_angle = 2.0 * math.pi * ((exchange.time_s + 30.0) % 86_400.0) / 86_400.0 + longitude - math.pi
    sunlit = np.cos(latitude) * np.cos(hour_angle) > 0.0
    assert sunlit.any() and (~sunlit).any()
    skin_dark = np.asarray(dark.surface.temperature_k)
    skin_bright = np.asarray(bright.surface.temperature_k)
    assert np.all(skin_dark[sunlit] > skin_bright[sunlit])
    assert np.array_equal(skin_dark[~sunlit], skin_bright[~sunlit])

    # Roughness: the bulk exchange coefficient is
    # C (1 + 0.08 ln(1 + z0 / 1e-4)), so a uniform roughness scales every
    # column's evaporation by the same closed-form factor.
    smooth, rough = arm(roughness_m=1.0e-4), arm(roughness_m=1.0)
    evap_smooth = smooth.diagnostics["mean_surface_evaporation_kg_m2_s"]
    evap_rough = rough.diagnostics["mean_surface_evaporation_kg_m2_s"]
    assert evap_smooth > 0.0
    expected = (1.0 + 0.08 * math.log1p(1.0 / 1.0e-4)) / (1.0 + 0.08 * math.log1p(1.0))
    assert evap_rough / evap_smooth == pytest.approx(expected, rel=1.0e-6)


def test_cold_top_relaxation_warms_only_cold_stratospheric_levels():
    _, model, _, exchange = _model_and_exchange(600.0)
    exchange.theta[...] = 150.0 / exchange.exner  # everywhere below the floor
    for name in WATER_SPECIES:
        getattr(exchange, name)[...] = 0.0
    options = ReferencePhysicsOptions(
        radiation=True,
        surface_fluxes=False,
        turbulence=False,
        convection=False,
        moist_convection=False,
        saturation_adjustment=False,
        microphysics=False,
        atmospheric_shortwave_absorptivity=0.0,
    )
    physics = ReferencePhysics(model.transform.backend, options)
    result = physics.step(exchange)
    dtemp = (result.theta - exchange.theta) * exchange.exner
    strat = exchange.p_full < options.stratospheric_floor_pa
    # Above the configured pressure a 150 K column warms toward the floor;
    # the expected pull is (195-150) * dt/tau plus the small longwave term.
    expected = 45.0 * 600.0 / options.stratospheric_relaxation_time_s
    assert np.all(dtemp[strat] > 0.5 * expected)
    # Below it the only tendency is longwave (cooling or tiny): no floor pull.
    troposphere = exchange.p_full > 50_000.0
    assert np.all(dtemp[troposphere] < 0.5 * expected)


def test_cold_top_relaxation_disabled_by_zero_floor():
    _, model, _, exchange = _model_and_exchange(600.0)
    exchange.theta[...] = 150.0 / exchange.exner
    for name in WATER_SPECIES:
        getattr(exchange, name)[...] = 0.0
    options = ReferencePhysicsOptions(
        radiation=True,
        surface_fluxes=False,
        turbulence=False,
        convection=False,
        moist_convection=False,
        saturation_adjustment=False,
        microphysics=False,
        atmospheric_shortwave_absorptivity=0.0,
        stratospheric_floor_k=0.0,
    )
    physics = ReferencePhysics(model.transform.backend, options)
    result = physics.step(exchange)
    dtemp = (result.theta - exchange.theta) * exchange.exner
    strat = exchange.p_full < 2_000.0
    # A 150 K atmosphere over a warm surface warms radiatively, but with
    # the floor disabled the pull is longwave-sized, far below the
    # (195-150) * dt/tau = 1.25 K the relaxation would add.
    expected_pull = 45.0 * 600.0 / 21_600.0
    assert np.all(dtemp[strat] < 0.1 * expected_pull)


def _inert_options(**overrides):
    """Every category toggle off, so a physics step moves nothing."""
    return ReferencePhysicsOptions(
        radiation=False,
        surface_fluxes=False,
        turbulence=False,
        convection=False,
        moist_convection=False,
        saturation_adjustment=False,
        microphysics=False,
        **overrides,
    )


def test_v8_suite_carries_no_sponge_and_leaves_winds_to_the_dycore():
    # The v7 absorber moved to the dycore (it compensates the dycore's
    # rigid lid, so it belongs to the model, not to one suite): v8 must
    # neither expose the options nor damp anything itself.
    _, model, _, exchange = _model_and_exchange(600.0)
    identity = ReferencePhysics(model.transform.backend).identity
    assert identity["suite"].endswith("-v8")
    assert "-sponge-" not in identity["suite"]
    assert "sponge_base_pa" not in identity["options"]
    assert "sponge_lid_relaxation_time_s" not in identity["options"]
    with pytest.raises(TypeError):
        ReferencePhysicsOptions(sponge_base_pa=5000.0)
    # An inert v8 step passes a top-level wave anomaly through bit-exactly
    # where the v7 step damped ~46% of it (the smoke grid's ~978 Pa top
    # ring sat inside the retired suite absorber's 5000 Pa base).
    lon = np.deg2rad(np.asarray(exchange.longitude_deg))
    exchange.u[...] = 12.0 + 5.0 * np.sin(2.0 * lon)[None]
    exchange.v[...] = 3.0 * np.cos(3.0 * lon)[None]
    physics = ReferencePhysics(model.transform.backend, _inert_options())
    result = physics.step(exchange)
    np.testing.assert_array_equal(result.u, exchange.u)
    np.testing.assert_array_equal(result.v, exchange.v)


def _v7_sponge(u, v, exchange, base_pa=5000.0, tau_lid=900.0):
    """The retired suite-v7 absorber arithmetic, verbatim, as the oracle."""
    ring_p = np.mean(np.asarray(exchange.p_full), axis=-1, keepdims=True)
    ring_lid = np.mean(np.asarray(exchange.p_half[0]), axis=-1, keepdims=True)
    in_sponge = ring_p < base_pa
    ramp = np.clip(
        (ring_p - ring_lid) / np.maximum(base_pa - ring_lid, 1.0e-3), 0.0, 1.0
    )
    rate = np.cos(0.5 * math.pi * ramp) ** 2 / tau_lid
    decay = np.exp(-float(exchange.dt_s) * rate)
    u_mean = np.mean(u, axis=-1, keepdims=True)
    v_mean = np.mean(v, axis=-1, keepdims=True)
    return (
        np.where(in_sponge, u_mean + (u - u_mean) * decay, u),
        np.where(in_sponge, v_mean + (v - v_mean) * decay, v),
    )


def test_dycore_absorber_reproduces_the_v7_physics_placement_bit_for_bit():
    # The v7 suite applied the absorber after turbulence, and no later
    # reference stage touched winds, so v7's returned winds are exactly
    # absorber(v8's returned winds).  The dycore applies the identical
    # arithmetic to the physics result's winds before their analysis, so
    # the placement difference on a reference-mode half-step is ZERO -
    # asserted bit-for-bit on the analyzed vorticity and divergence.
    from woof.globe.state import ArwenGlobalState

    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    # A resolvable top-level wave anomaly in the SPECTRAL state, so the
    # absorber has structure to remove on both paths.
    g = model.grid_state(state.atmosphere, only=("u", "v"))
    lon = np.deg2rad(model.transform.grid.longitude_deg)[None, :]
    u = np.array(g["u"], copy=True)
    v = np.array(g["v"], copy=True)
    u[0] += 5.0 * np.sin(2.0 * lon)
    v[0] += 3.0 * np.cos(3.0 * lon)
    zeta, div = model.vector.vordiv_from_wind(u, v)
    fields = list(state.atmosphere.fields())
    fields[0] = zeta
    fields[1] = div
    bundle = ArwenGlobalState(
        state.atmosphere.with_fields(fields), state.surface,
        state.physics_state,
    )
    dt = 0.5 * cfg.dt_s

    exchange = model._physics_exchange(bundle, dt)
    result = model.physics.step(exchange)
    u7, v7 = _v7_sponge(np.asarray(result.u), np.asarray(result.v), exchange)
    # Treatment proof: the oracle really damped the top level.
    assert float(np.max(np.abs(u7[0] - np.asarray(result.u)[0]))) > 1.0e-4
    np.testing.assert_array_equal(u7[1:], np.asarray(result.u)[1:])
    zeta7, div7 = model.vector.vordiv_from_wind(u7, v7)

    out, info = model.apply_physics(bundle, dt)
    assert info["physics_mode"] == "reference"
    np.testing.assert_array_equal(
        np.asarray(out.atmosphere.vorticity), np.asarray(zeta7)
    )
    np.testing.assert_array_equal(
        np.asarray(out.atmosphere.divergence), np.asarray(div7)
    )
