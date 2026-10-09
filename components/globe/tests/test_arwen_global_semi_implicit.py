"""The external-only (barotropic) semi-implicit proxy, selectable as
[semi_implicit] scheme = "external" for identity-locked archives and A/B.

Every test here pins the config to that scheme: the default is the
vertical-mode scheme (tests/test_arwen_global_vertical_modes.py), and the
proxy must stay bit-identical to its era.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import dataclasses

import numpy as np
import pytest

from woof.globe.config import load_config
from woof.globe.runner import build_model_and_cold_state
from woof.globe.semi_implicit import (
    BarotropicSemiImplicit,
    VerticalModeSemiImplicit,
)

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


def _quiet_config(**overrides):
    overrides.setdefault("semi_implicit_scheme", "external")
    return dataclasses.replace(load_config(CONFIG), **overrides)


def test_rhs_subtracts_exactly_the_weighted_linear_pair():
    # The split is only real if the explicit right-hand side stops carrying
    # the wave: with the semi-implicit active, rhs must differ from the
    # inactive rhs by exactly w*c^2*k^2*p in every divergence level and
    # -(-w*Dbar) in log surface pressure. A corrector that integrates the
    # pair while this subtraction is missing double-steps the external mode
    # (the defect that amplified the first real-data run to 1256 hPa).
    cfg = _quiet_config()
    model, state = build_model_and_cold_state(cfg)
    assert isinstance(model.semi_implicit, BarotropicSemiImplicit)
    transform = model.transform
    on = model.rhs(state.atmosphere)

    off_model = dataclasses.replace(
        model.semi_implicit, enabled=False
    )
    model.semi_implicit = off_model
    off = model.rhs(state.atmosphere)
    model.semi_implicit = BarotropicSemiImplicit(
        enabled=True,
        external_wave_speed_m_s=cfg.external_wave_speed_m_s,
        divergence_weight=cfg.semi_implicit_weight,
    )

    n = np.arange(transform.truncation + 1, dtype=np.float64)
    k2 = n * (n + 1.0) / transform.grid.radius_m ** 2
    w = cfg.semi_implicit_weight
    c2 = cfg.external_wave_speed_m_s ** 2
    p = transform.backend.to_numpy(state.atmosphere.log_surface_pressure)
    expected_divergence = transform.project(
        np.asarray(w * c2 * k2[:, None] * p)[None]
    )[0]
    delta_divergence = transform.backend.to_numpy(
        off.divergence - on.divergence
    )
    for level in range(model.nlev):
        np.testing.assert_allclose(
            delta_divergence[level], expected_divergence,
            rtol=1.0e-12, atol=1.0e-18,
        )
    # The proxy never touches theta.
    np.testing.assert_array_equal(
        transform.backend.to_numpy(off.theta), transform.backend.to_numpy(on.theta)
    )
    weights = model.vertical.delta_b / np.sum(model.vertical.delta_b)
    barotropic = np.sum(
        weights[:, None, None]
        * transform.backend.to_numpy(state.atmosphere.divergence),
        axis=0,
    )
    delta_logps = transform.backend.to_numpy(
        off.log_surface_pressure - on.log_surface_pressure
    )
    np.testing.assert_allclose(
        delta_logps, transform.project(-w * barotropic),
        rtol=1.0e-12, atol=1.0e-20,
    )


def test_apply_is_neutral_crank_nicolson_and_actually_acts():
    # Crank-Nicolson of the weighted pair has |amplification| = 1: the wave
    # energy w^2*c^2*k^2|p|^2-ish invariant E = c^2 k^2 |p|^2 + |Dbar|^2 is
    # conserved by apply() at any dt, which is the whole stability claim.
    cfg = _quiet_config()
    model, state = build_model_and_cold_state(cfg)
    transform = model.transform
    semi = model.semi_implicit
    atmosphere = state.atmosphere

    n = np.arange(transform.truncation + 1, dtype=np.float64)
    k2 = n * (n + 1.0) / transform.grid.radius_m ** 2
    weights = model.vertical.delta_b / np.sum(model.vertical.delta_b)

    def energy(s):
        barotropic = np.sum(
            weights[:, None, None] * transform.backend.to_numpy(s.divergence),
            axis=0,
        )
        p = transform.backend.to_numpy(s.log_surface_pressure)
        c2 = semi.external_wave_speed_m_s ** 2
        total = np.abs(barotropic) ** 2 + c2 * k2[:, None] * np.abs(p) ** 2
        total[0, 0] = 0.0
        return float(np.sum(total))

    before = energy(atmosphere)
    assert before > 0.0
    stepped, metrics = semi.apply(
        atmosphere, transform, model.vertical, 7200.0
    )
    after = energy(stepped)
    assert metrics["semi_implicit_max_divergence_increment_s1"] > 0.0
    assert abs(after - before) <= 1.0e-9 * before


def test_weight_scales_the_split_measurably():
    # divergence_weight is pinned by measurement: halving it must halve the
    # subtracted linear tendency and change the corrector's increment, so
    # the mutation that replaces the weight with 1.0 dies here.
    cfg = _quiet_config()
    model, state = build_model_and_cold_state(cfg)
    transform = model.transform
    full = BarotropicSemiImplicit(
        enabled=True,
        external_wave_speed_m_s=cfg.external_wave_speed_m_s,
        divergence_weight=1.0,
    )
    half = BarotropicSemiImplicit(
        enabled=True,
        external_wave_speed_m_s=cfg.external_wave_speed_m_s,
        divergence_weight=0.5,
    )
    lin_full = full.linear_tendencies(
        state.atmosphere, transform, model.vertical
    )
    lin_half = half.linear_tendencies(
        state.atmosphere, transform, model.vertical
    )
    np.testing.assert_allclose(
        transform.backend.to_numpy(lin_half.divergence),
        0.5 * transform.backend.to_numpy(lin_full.divergence),
        rtol=1.0e-12, atol=0.0,
    )
    _, metric_full = full.apply(
        state.atmosphere, transform, model.vertical, 3600.0
    )
    _, metric_half = half.apply(
        state.atmosphere, transform, model.vertical, 3600.0
    )
    ratio = (
        metric_full["semi_implicit_max_divergence_increment_s1"]
        / max(metric_half["semi_implicit_max_divergence_increment_s1"], 1.0e-300)
    )
    assert 1.2 < ratio < 4.0


def _run_arm(scheme, dt, steps, **overrides):
    cfg = _quiet_config(
        physics_mode="none",
        diffusion_enabled=False,
        mass_fixer=False,
        water_fixer=False,
        positivity_repair=False,
        truncation=10,
        semi_implicit_scheme=scheme,
        # The smoke config's 200 m/s is BELOW the measured effective
        # barotropic coupling (331 m/s on this atmosphere; a 300 m/s
        # draft of this test still blew up on the residual), which is
        # the module header's stability constraint demonstrated; the
        # derived default covers it.
        external_wave_speed_m_s=450.0,
        semi_implicit_weight=1.0,
        **overrides,
    )
    model, state = build_model_and_cold_state(cfg)
    model.maximum_cfl = 1.0e9
    for _ in range(steps):
        state, _metrics = model.step(state, dt)
    divergence = model.transform.backend.to_numpy(state.atmosphere.divergence)
    return float(np.max(np.abs(divergence)))


def test_external_proxy_is_stable_beyond_the_explicit_external_limit():
    # The operating point the proxy exists for, with the budget the audit
    # corrected (DN-5): the 4-level smoke stack's linearized modes are
    # 328.8 / 210.7 / 72.0 / 26.5 m/s (286/220 K column), so at T10
    # (k = 1.646e-6 1/m) the SSPRK3 imaginary-axis limit sqrt(3)/(c k) is
    # 3200 s for the external mode and 4994 s for the first INTERNAL mode,
    # which the proxy leaves explicit.  The former dt = 5500 s arm sat
    # past that internal limit and passed only because 12 steps of growth
    # from a smooth state stayed under its threshold (step 27: temperature
    # -15 K).  dt = 4000 s is beyond the external limit and inside the
    # internal one: ON must stay bounded over 60 steps (5x the former
    # count, past where the 5500 s arm died); OFF must leave the research
    # bounds on the external mode this proxy owns.
    dt = 4000.0
    stable = _run_arm("external", dt, 60)
    with pytest.raises(FloatingPointError):
        _run_arm("external", dt, 60, semi_implicit_enabled=False)
    assert np.isfinite(stable)
    assert stable < 1.0e-3


def test_vertical_modes_are_stable_where_the_proxy_left_internal_modes_explicit():
    # dt = 5500 s is past the first internal mode's explicit limit (4994 s
    # on this stack): the audit's v5 run of the proxy's ON arm reached
    # 2.4e-4 at step 24 and died at step 27.  With every mode implicit
    # the same arm stays bounded through 60 steps.
    dt = 5500.0
    stable = _run_arm("vertical_modes", dt, 60)
    assert np.isfinite(stable)
    assert stable < 1.0e-3


def test_scheme_door_identity_and_refusals(tmp_path):
    # The default is the vertical-mode scheme; selecting "external" keeps
    # the hash the config had when that was the only scheme (measured on
    # the base tree before the scheme key existed), and the two never
    # share a hash.
    cfg = load_config(CONFIG)
    assert cfg.semi_implicit_scheme == "vertical_modes"
    assert "semi_implicit_scheme" in cfg.config_identity
    external = dataclasses.replace(cfg, semi_implicit_scheme="external")
    assert "semi_implicit_scheme" not in external.config_identity
    assert external.config_hash == (
        "7cebe340c371837416aa3aa5fd488da8cc4d630823d5e81c131098b0ea61cdc5"
    )
    assert cfg.config_hash != external.config_hash
    model, _ = build_model_and_cold_state(external)
    assert isinstance(model.semi_implicit, BarotropicSemiImplicit)
    model, _ = build_model_and_cold_state(cfg)
    assert isinstance(model.semi_implicit, VerticalModeSemiImplicit)
    assert model.semi_implicit.reference_temperature_k == 320.0
    assert model.semi_implicit.off_centring_weight == 0.5

    from pathlib import Path

    base_text = Path(CONFIG).read_text(encoding="utf-8")

    def write(name, table):
        path = tmp_path / name
        path.write_text(
            base_text.replace("[semi_implicit]\n", "[semi_implicit]\n" + table),
            encoding="utf-8",
        )
        return path

    tuned = load_config(write("tuned.toml", "off_centring_weight = 0.55\nreference_temperature_k = 350.0\n"))
    assert tuned.semi_implicit_off_centring_weight == 0.55
    assert tuned.semi_implicit_reference_temperature_k == 350.0
    assert tuned.config_hash != cfg.config_hash
    with pytest.raises(ValueError, match="amplifies"):
        load_config(write("amplify.toml", "off_centring_weight = 0.4\n"))
    with pytest.raises(ValueError, match="scheme must be one of"):
        load_config(write("scheme.toml", 'scheme = "leapfrog"\n'))
    with pytest.raises(ValueError, match="reference_temperature_k"):
        load_config(write("cold.toml", "reference_temperature_k = 100.0\n"))
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(write("unknown.toml", "alpha = 0.5\n"))
