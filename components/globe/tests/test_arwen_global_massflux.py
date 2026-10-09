"""arwen-massflux-v1: the native suite's own convection closure.

Ledger gates run on a synthetic 40-level column in float64 (1e-6
relative), the scale-aware shutdown on the same column across dx, and the
runtime plumbing (cumulus='own') on the moist smoke configuration with the
real scheme behind the suite's fakes.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from types import SimpleNamespace

import numpy as np
import pytest

from woof.globe.constants import (
    DRY_AIR_CP,
    DRY_AIR_GAS_CONSTANT,
    EPSILON,
    GRAVITY_M_S2,
    LATENT_HEAT_FUSION,
    LATENT_HEAT_VAPORIZATION,
    NATIVE_PHYSICS_ACKNOWLEDGEMENT,
    WATER_SPECIES,
)
from woof.globe.physics.arwen_massflux import (
    SCHEME_NAME,
    ArwenMassFluxV1,
    MassFluxParameters,
    massflux_columns,
    saturation_mixing_ratio,
    scale_factors,
)

NLEV = 40
DT_S = 100.0
DX_M = 52_100.0


def _column(*, unstable=True, ncol=1, dtype=np.float64):
    """A 40-level column with the native split's first level at 22.9 m.

    ``unstable``: a moist tropical sounding (300 K, 85% RH in the lowest
    2 km, conditionally unstable through 12 km, rising motion, 150 W/m2 of
    sensible and latent heat flux).  ``unstable=False``: the same shape
    with a dry, stable troposphere and no surface flux.
    """
    g = GRAVITY_M_S2
    dz = 45.8 * 1.10 ** np.arange(NLEV)
    z_half = np.concatenate([[0.0], np.cumsum(dz)])
    z = z_half[:-1] + 0.5 * dz
    if unstable:
        t = np.maximum(300.0 - 6.5e-3 * z, 200.0)
        rh = np.where(z < 2_000.0, 0.85, np.clip(0.85 - 0.55 * (z - 2_000.0) / 8_000.0, 0.05, 0.85))
    else:
        t = np.maximum(285.0 - 4.0e-3 * z, 215.0)
        rh = np.full(NLEV, 0.3)
    # Hydrostatic pressure on the half levels, then full levels at the
    # layer-mean pressure.
    p_half = np.empty(NLEV + 1)
    p_half[0] = 100_000.0
    for k in range(NLEV):
        p_half[k + 1] = p_half[k] * np.exp(-g * dz[k] / (DRY_AIR_GAS_CONSTANT * t[k]))
    p = 0.5 * (p_half[:-1] + p_half[1:])
    dp = p_half[:-1] - p_half[1:]
    qv = rh * saturation_mixing_ratio(np, t, p)
    frac = (p_half[0] - p_half) / (p_half[0] - p_half[-1])
    omega = -0.15 * np.sin(np.pi * np.clip(frac, 0.0, 1.0)) if unstable else np.zeros(NLEV + 1)
    u = 5.0 + 2.0e-3 * z
    v = -2.0 + 1.0e-3 * z

    def level(a):
        return np.ascontiguousarray(np.repeat(np.asarray(a, dtype)[:, None], ncol, axis=1))

    def plane(value):
        return np.full(ncol, value, dtype)

    return {
        "p": level(p), "dp": level(dp), "dz": level(dz), "t": level(t),
        "qv": level(qv), "qc": level(np.zeros(NLEV)), "qi": level(np.zeros(NLEV)),
        "u": level(u), "v": level(v), "omega_half": level(omega),
        "hfx": plane(150.0 if unstable else 0.0),
        "qfx": plane(150.0 / LATENT_HEAT_VAPORIZATION if unstable else 0.0),
        "pblh": plane(900.0),
    }


def _apply(columns, out, dt=DT_S):
    return {
        "t": columns["t"] + dt * out["dt"],
        "qv": columns["qv"] + dt * out["dqv"],
        "qc": columns["qc"] + dt * out["dqc"],
        "qi": columns["qi"] + dt * out["dqi"],
        "u": columns["u"] + dt * out["du"],
        "v": columns["v"] + dt * out["dv"],
    }


def _integral(columns, field):
    return np.sum(field * columns["dp"] / GRAVITY_M_S2, axis=0)


def test_the_first_full_level_of_the_synthetic_column_is_the_native_split():
    columns = _column()
    assert float(columns["dz"][0, 0] / 2.0) == pytest.approx(22.9, abs=0.05)
    assert columns["p"].shape == (NLEV, 1)


def test_deep_convection_triggers_on_the_unstable_column_and_not_on_the_stable_one():
    unstable = massflux_columns(np, _column(), dt_s=DT_S, dx_m=DX_M)
    assert bool(unstable["deep"][0]) and not bool(unstable["shallow"][0])
    assert float(unstable["cape"][0]) > 100.0
    assert float(unstable["rain"][0]) > 0.0
    assert float(unstable["mass_flux"][0]) > 0.0
    # Cloud through more than the deep threshold, top in the upper troposphere.
    base, top = int(unstable["cloud_base"][0]), int(unstable["cloud_top"][0])
    assert 0 < base < top < NLEV - 3
    stable = massflux_columns(np, _column(unstable=False), dt_s=DT_S, dx_m=DX_M)
    assert not bool(stable["deep"][0]) and not bool(stable["shallow"][0])
    assert float(stable["rain"][0]) == 0.0
    for name in ("dt", "dqv", "dqc", "dqi", "du", "dv"):
        assert np.all(stable[name] == 0.0), name


def test_column_water_removed_equals_the_raincv_booked_to_1e_6():
    columns = _column()
    out = massflux_columns(np, columns, dt_s=DT_S, dx_m=DX_M)
    after = _apply(columns, out)
    before_water = _integral(columns, columns["qv"] + columns["qc"] + columns["qi"])
    after_water = _integral(columns, after["qv"] + after["qc"] + after["qi"])
    removed = float((before_water - after_water)[0])
    rain = float(out["rain"][0])
    assert rain > 1.0e-3
    assert removed == pytest.approx(rain, rel=1.0e-6)


def test_enthalpy_change_equals_the_latent_heating_of_the_precipitation_to_1e_6():
    columns = _column()
    out = massflux_columns(np, columns, dt_s=DT_S, dx_m=DX_M)
    after = _apply(columns, out)
    sensible = _integral(columns, DRY_AIR_CP * (after["t"] - columns["t"]))
    vapour = _integral(columns, after["qv"] - columns["qv"])
    ice = _integral(columns, after["qi"] - columns["qi"])
    # cp dT + Lv dqv - Lf dqi closes: the sensible heating is the latent heat
    # of the vapour condensed (precipitation plus detrained condensate,
    # net of evaporation and melting), with fusion booked on the ice.
    latent = -LATENT_HEAT_VAPORIZATION * vapour + LATENT_HEAT_FUSION * ice
    assert float(sensible[0]) > 0.0
    assert float(sensible[0]) == pytest.approx(float(latent[0]), rel=1.0e-6)
    # And the precipitation's own share: the vapour lost is the rain plus
    # what was detrained as cloud water and ice.
    detrained = _integral(columns, (after["qc"] - columns["qc"]) + (after["qi"] - columns["qi"]))
    assert float(-vapour[0]) == pytest.approx(float(out["rain"][0]) + float(detrained[0]), rel=1.0e-6)
    assert float(detrained[0]) > 0.0


def test_convective_momentum_transport_conserves_column_momentum_and_acts():
    columns = _column()
    out = massflux_columns(np, columns, dt_s=DT_S, dx_m=DX_M)
    after = _apply(columns, out)
    scale = float(_integral(columns, np.abs(columns["u"]))[0])
    assert abs(float(_integral(columns, after["u"] - columns["u"])[0])) < 1.0e-9 * scale
    assert abs(float(_integral(columns, after["v"] - columns["v"])[0])) < 1.0e-9 * scale
    assert float(np.max(np.abs(out["du"]))) > 0.0
    # Sheared flow: the cloud layer mixes momentum downward, so the
    # lower cloud speeds up and the upper cloud slows (u increases with z).
    top = int(out["cloud_top"][0])
    assert float(out["du"][top, 0]) < 0.0


def test_every_species_stays_nonnegative_after_a_step_at_the_longest_dt():
    columns = _column()
    for dt in (50.0, 100.0, 200.0):
        out = massflux_columns(np, columns, dt_s=dt, dx_m=DX_M)
        after = _apply(columns, out, dt=dt)
        for name in ("qv", "qc", "qi"):
            assert float(after[name].min()) >= 0.0, (name, dt)
        assert float(after["t"].min()) > 150.0


def test_the_deep_branch_shuts_as_the_grid_resolves_convection():
    params = MassFluxParameters()
    sigma, beta, eps = scale_factors(52_100.0, params)
    assert sigma == pytest.approx(0.0059, abs=1.0e-3) and beta == pytest.approx(0.988, abs=1.0e-3)
    assert scale_factors(25_000.0, params)[1] == pytest.approx(0.949, abs=1.0e-3)
    assert scale_factors(4_000.0, params)[1] == 0.0
    assert scale_factors(1_000.0, params)[1] == 0.0
    assert scale_factors(10_000.0, params)[2] > eps
    fluxes = []
    for dx in (52_100.0, 25_000.0, 10_000.0, 5_000.0, 4_000.0):
        out = massflux_columns(np, _column(), dt_s=DT_S, dx_m=dx)
        fluxes.append(float(out["mass_flux"][0]))
        if dx > 4_000.0:
            assert bool(out["deep"][0]), dx
    assert fluxes == sorted(fluxes, reverse=True)
    assert fluxes[-1] == 0.0 or not bool(massflux_columns(np, _column(), dt_s=DT_S, dx_m=4_000.0)["deep"][0])
    with pytest.raises(ValueError, match="dx_m must be positive"):
        scale_factors(0.0, params)


def test_the_closure_is_bounded_by_the_positivity_transfer_bound():
    # Force a huge closure by handing enormous forcing: the transfer bound
    # must still hold every level's loss under transfer_bound of itself.
    columns = _column()
    params = MassFluxParameters()
    forcing_s = np.full_like(columns["t"], 5.0)        # J/kg/s, absurd
    forcing_q = np.full_like(columns["t"], 1.0e-3)
    out = massflux_columns(
        np, columns, dt_s=200.0, dx_m=DX_M, forcing_s=forcing_s, forcing_q=forcing_q,
        params=params,
    )
    after = _apply(columns, out, dt=200.0)
    assert float(after["qv"].min()) >= 0.0
    assert float(out["mass_flux"][0]) > 0.0
    loss = -200.0 * out["dqv"] / np.maximum(columns["qv"], 1.0e-12)
    assert float(loss.max()) <= params.transfer_bound + 1.0e-9


def test_many_columns_run_as_one_batch_the_same_as_singles():
    # Columns never read each other; the only difference between a batch
    # and a single column is numpy's level-axis summation order (pairwise
    # on the contiguous single column, sequential on the strided batch).
    single = massflux_columns(np, _column(), dt_s=DT_S, dx_m=DX_M)
    stable = massflux_columns(np, _column(unstable=False), dt_s=DT_S, dx_m=DX_M)
    batch = {
        name: np.concatenate([_column()[name], _column(unstable=False)[name]], axis=-1)
        for name in _column()
    }
    both = massflux_columns(np, batch, dt_s=DT_S, dx_m=DX_M)
    for name in ("dt", "dqv", "dqc", "dqi", "du", "dv"):
        assert np.allclose(both[name][:, 0], single[name][:, 0], rtol=1.0e-10, atol=0.0), name
        assert np.array_equal(both[name][:, 1], stable[name][:, 0]), name
    assert np.allclose(both["rain"], np.concatenate([single["rain"], stable["rain"]]), rtol=1.0e-10)


def test_the_callable_runs_float32_columns_chunked_and_returns_the_cumulus_contract():
    ny, nx = 3, 4
    columns = _column(ncol=ny * nx, dtype=np.float32)
    shape = (NLEV, ny, nx)

    def grid(a):
        return np.ascontiguousarray(a.reshape(shape))

    p = grid(columns["p"])
    dp = grid(columns["dp"])
    p_half = np.empty((NLEV + 1, ny, nx), np.float32)
    p_half[0] = p[0] + 0.5 * dp[0]
    p_half[1:] = p - 0.5 * dp
    atmosphere = {
        "u": grid(columns["u"]), "v": grid(columns["v"]),
        "temperature": grid(columns["t"]), "qv": grid(columns["qv"]),
        "qc": grid(columns["qc"]), "qi": grid(columns["qi"]),
        "pressure": p, "exner": (p / 1.0e5).astype(np.float32) ** 0.2857,
        "rho": (p / (287.0 * grid(columns["t"]))).astype(np.float32),
        "dz": grid(columns["dz"]), "p_interface": p_half,
    }
    fields = {
        "hfx": columns["hfx"].reshape(ny, nx), "qfx": columns["qfx"].reshape(ny, nx),
        "xland": np.ones((ny, nx), np.float32), "kpbl": np.full((ny, nx), 4, np.int32),
        "pblh": columns["pblh"].reshape(ny, nx),
    }
    state = SimpleNamespace(
        p=p, w=np.zeros((NLEV + 1, ny, nx), np.float32),
        ht=np.zeros((ny, nx), np.float32), qi=atmosphere["qi"],
        omega_half=np.ascontiguousarray(columns["omega_half"].reshape(NLEV + 1, ny, nx)),
    )
    cfg = SimpleNamespace(dx=DX_M, dt=DT_S, clock_dt=DT_S)
    scheme = ArwenMassFluxV1(column_chunk=5)
    # No PBL lanes handed (None): the quasi-equilibrium closure falls back
    # to the surface fluxes, the same arm the column function ran.
    scheme.bind_driver(SimpleNamespace(
        rthratenlw=np.zeros(shape, np.float32), rthratensw=np.zeros(shape, np.float32),
        gf_rthblten=None, gf_rqvblten=None,
    ))
    result = scheme(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten"):
        value = getattr(result, name)
        assert value.shape == shape and value.dtype == np.float32, name
    assert result.rainc.shape == (ny, nx) and result.rainc.dtype == np.float32
    assert float(result.rainc.min()) > 0.0
    assert np.allclose(result.rainc, result.rainc[0, 0], rtol=1.0e-5)
    assert result.diagnostics["deep_column_fraction"] == 1.0
    assert result.diagnostics["scale_beta"] == pytest.approx(0.988, abs=1.0e-3)
    # Heating in theta = dT / exner, the column function's own float32
    # arithmetic on the same packed columns.
    reference = massflux_columns(np, columns, dt_s=DT_S, dx_m=DX_M)
    assert np.allclose(
        result.rthcuten, grid(reference["dt"]) / atmosphere["exner"], rtol=1.0e-3, atol=1.0e-12
    )
    assert float(np.max(np.abs(result.rthcuten))) > 1.0e-6
    assert scheme.identity["scheme"] == SCHEME_NAME
    with pytest.raises(ValueError, match="column_chunk"):
        ArwenMassFluxV1(column_chunk=0)


# --- runtime plumbing ---------------------------------------------------

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


def _options(**extra):
    return {
        "acknowledgement": NATIVE_PHYSICS_ACKNOWLEDGEMENT,
        "start_time_utc": "2024-05-21T00:00:00Z",
        "radiation_interval_s": 10.0,
        "land_surface_interval_s": 10.0,
        "cumulus": "own",
        **extra,
    }


def _native_fakes():
    """The native suite test's fakes (that file is not a package member)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).with_name("test_arwen_global_level5_native.py")
    spec = importlib.util.spec_from_file_location("_level5_native_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._fake_modules


def _suite_and_exchange(modules_override=None, **extra):
    from woof.globe.config import load_config
    from woof.globe.runner import build_model_and_cold_state
    from woof.globe.physics.native_suite import ArwenCudaColumnSuite

    _fake_modules = _native_fakes()
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    exchange = model._physics_exchange(state, 5.0)
    modules = _fake_modules()
    modules.update(modules_override or {})
    suite = ArwenCudaColumnSuite(_options(**extra), array_module=np, modules=modules)
    return suite, exchange


def test_cumulus_own_is_admitted_and_names_the_scheme_in_the_order_and_identity():
    from woof.globe.physics.native_options import NativePhysicsOptions
    from woof.globe.physics.native_suite import (
        NATIVE_COMPONENT_ORDER, native_component_order,
    )

    options = NativePhysicsOptions.from_mapping(_options())
    assert options.cumulus_enabled and not options.uses_grell_freitas
    assert options.cumulus_scheme == "own"
    assert options.cumulus_component == SCHEME_NAME
    assert native_component_order(options) == (
        "rrtmgp", "sfclay", "noah", "ysu", SCHEME_NAME, "morrison"
    )
    assert NATIVE_COMPONENT_ORDER == ("rrtmgp", "sfclay", "noah", "ysu", "gf", "morrison")
    gf = NativePhysicsOptions.from_mapping({**_options(), "cumulus": "gf"})
    assert gf.uses_grell_freitas and native_component_order(gf) == NATIVE_COMPONENT_ORDER
    none = NativePhysicsOptions.from_mapping({**_options(), "cumulus": "none"})
    assert none.cumulus_component is None and not none.cumulus_enabled
    with pytest.raises(ValueError, match="cumulus='own'"):
        NativePhysicsOptions.from_mapping({**_options(), "cumulus": "mine"})
    assert options.identity["cumulus"] == "own"


def test_the_suite_runs_the_real_scheme_in_the_cumulus_slot_and_the_water_closes():
    suite, exchange = _suite_and_exchange()
    result = suite.step(exchange)
    assert result.diagnostics["cumulus_active"] is True
    assert result.physics_state.metadata["cumulus_updates"] == 1
    assert suite.identity["component_order"][4] == SCHEME_NAME
    assert suite.identity["options"]["cumulus"] == "own"
    assert type(suite._runtime._cumulus).__name__ == "ArwenMassFluxV1"
    assert suite._runtime._cumulus.column_chunk == 131_072
    for name in ("deep_column_fraction", "shallow_column_fraction",
                 "mean_base_mass_flux_kg_m2_s", "mean_convective_rain_kg_m2"):
        assert f"cumulus_{name}" in result.diagnostics
    assert result.diagnostics["maximum_native_water_residual_kg_m2"] < 1.0e-5
    assert result.diagnostics["native_water_residual_exceeds_tolerance"] == 0.0
    for name in ("u", "v", "theta", *WATER_SPECIES):
        assert np.all(np.isfinite(getattr(result, name))), name
    for name in WATER_SPECIES:
        assert float(np.min(getattr(result, name))) >= 0.0, name


def test_momentum_rates_from_the_cumulus_slot_integrate_into_u_and_v():
    from woof.globe.physics.arwen_massflux import MassFluxResult

    class Pushing(ArwenMassFluxV1):
        def __call__(self, **kwargs):
            shape = kwargs["atmosphere"]["temperature"].shape
            zeros = np.zeros(shape, np.float32)
            du = zeros.copy()
            du[2] = 1.0e-3
            dv = zeros.copy()
            dv[3] = -2.0e-3
            return MassFluxResult(
                rthcuten=zeros.copy(), rqvcuten=zeros.copy(), rqccuten=zeros.copy(),
                rqicuten=zeros.copy(), rucuten=du, rvcuten=dv,
                rainc=np.zeros(shape[1:], np.float32), diagnostics={},
            )

    suite, exchange = _suite_and_exchange({
        "woof.globe.physics.arwen_massflux": SimpleNamespace(ArwenMassFluxV1=Pushing),
    })
    result = suite.step(exchange)
    dt = float(exchange.dt_s)
    du = np.asarray(result.u, np.float64) - np.asarray(exchange.u, np.float64)
    dv = np.asarray(result.v, np.float64) - np.asarray(exchange.v, np.float64)
    # Native level 2 is global level -3 (bottom-to-top reversal).
    assert np.allclose(du[-3], dt * 1.0e-3, atol=1.0e-6)
    assert np.allclose(dv[-4], -dt * 2.0e-3, atol=1.0e-6)
    other = [k for k in range(du.shape[0]) if k != du.shape[0] - 3]
    # Untouched levels move only by the float32 round trip of u (~1e-8).
    assert np.allclose(du[other], 0.0, atol=1.0e-6)


def test_cumulus_own_refuses_an_exchange_without_vertical_velocity():
    from dataclasses import replace

    suite, exchange = _suite_and_exchange()
    with pytest.raises(ValueError, match="cumulus='own'.*w = 0"):
        suite.step(replace(exchange, omega_half_pa_s=None))


def test_the_registration_names_the_own_scheme():
    from woof.globe.physics.arwen_bridge import NativeArwenPhysicsBridge
    from woof.globe.physics.builtin_adapters import ADAPTER_NAME

    bridge = NativeArwenPhysicsBridge(ADAPTER_NAME, _options())
    contract = bridge.registration.contract
    assert SCHEME_NAME in contract["scheme_identity"]["cumulus"]
    assert bridge.identity["options"]["cumulus"] == "own"
