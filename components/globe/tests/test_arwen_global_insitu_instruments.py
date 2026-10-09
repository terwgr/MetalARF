from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from dataclasses import replace
import json
import math
from pathlib import Path

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
from woof.globe.insitu import (
    CAPTURE_NAMES,
    LEDGER_NAMES,
    LEDGER_TERMS,
    ComponentCapture,
    SpectralKineticEnergy,
    THRESHOLDS,
    TripwireSet,
)
from woof.globe.insitu.budgets import LEDGER_INDEX, LedgerGeometry, ledger_row
from woof.globe.runner import build_model_and_cold_state
from woof.globe.spectral.transform import SphericalHarmonicTransform
from woof.globe.spectral.vector import VorticityDivergenceOperator

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
REFERENCE_COMPONENTS = (
    "surface_radiation_fluxes", "turbulence", "convective_adjust",
    "betts_miller", "saturation_adjust", "microphysics", "negative_clamp",
)


def _smoke(steps: int = 24, **overrides):
    cfg = load_config(CONFIG)
    return replace(
        cfg, duration_s=cfg.dt_s * steps, output_interval_s=cfg.dt_s * steps,
        **overrides,
    )


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# -- budgets ----------------------------------------------------------------

def test_ledger_row_matches_host_budgets():
    cfg = _smoke()
    model, state = build_model_and_cold_state(cfg)
    state, _ = model.step(state, cfg.dt_s)
    geometry = LedgerGeometry(model.transform, model.rotation_rate_s)
    row = np.asarray(ledger_row(model, state, geometry), dtype=np.float64)
    assert row.shape == (len(LEDGER_TERMS),)
    assert LEDGER_NAMES == tuple(name for name, _ in LEDGER_TERMS)

    g = model.grid_state(state.atmosphere)
    grid = model.transform.grid
    dp_g = g["dp"] / GRAVITY_M_S2
    term = lambda name: row[LEDGER_INDEX[name]]  # noqa: E731

    assert term("mass_pa") == pytest.approx(grid.global_mean(g["ps"]), rel=1.0e-13)
    assert term("water_total_kg_m2") == pytest.approx(
        model._global_mean_total_water(state), rel=1.0e-12
    )
    for name in WATER_SPECIES:
        expected = grid.global_mean(np.sum(g[name] * dp_g, axis=0))
        assert term(f"water_{name}_kg_m2") == pytest.approx(expected, rel=1.0e-12, abs=1e-300)
    kinetic = grid.global_mean(np.sum(0.5 * (g["u"] ** 2 + g["v"] ** 2) * dp_g, axis=0))
    dry = grid.global_mean(np.sum(DRY_AIR_CP * g["temperature"] * dp_g, axis=0))
    surface_potential = grid.global_mean(
        np.asarray(model.surface_geopotential) * g["ps"] / GRAVITY_M_S2
    )
    potential = grid.global_mean(np.sum(g["geopotential"] * dp_g, axis=0))
    assert term("kinetic_energy_j_m2") == pytest.approx(kinetic, rel=1.0e-12)
    assert term("dry_enthalpy_j_m2") == pytest.approx(dry, rel=1.0e-12)
    assert term("surface_potential_energy_j_m2") == pytest.approx(surface_potential, rel=1.0e-12)
    assert term("potential_energy_j_m2") == pytest.approx(potential, rel=1.0e-12)
    # The stated identities hold in the row itself.
    assert term("total_dry_energy_j_m2") == pytest.approx(
        term("dry_enthalpy_j_m2") + term("kinetic_energy_j_m2")
        + term("surface_potential_energy_j_m2"), rel=1.0e-14,
    )
    assert term("latent_vapor_j_m2") == pytest.approx(
        LATENT_HEAT_VAPORIZATION * term("water_qv_kg_m2"), rel=1.0e-12
    )
    assert term("latent_frozen_j_m2") == pytest.approx(
        LATENT_HEAT_FUSION * (
            term("water_qi_kg_m2") + term("water_qs_kg_m2") + term("water_qg_kg_m2")
        ), rel=1.0e-12, abs=1e-300,
    )
    assert term("moist_total_energy_j_m2") == pytest.approx(
        term("total_dry_energy_j_m2") + term("latent_vapor_j_m2")
        - term("latent_frozen_j_m2"), rel=1.0e-14,
    )
    a_cos = grid.radius_m * grid.cos_lat[:, None]
    relative = grid.global_mean(np.sum(g["u"] * dp_g, axis=0) * a_cos)
    planetary = grid.global_mean(
        np.sum(dp_g, axis=0) * model.rotation_rate_s * a_cos * a_cos
    )
    assert term("relative_angular_momentum_kg_s") == pytest.approx(relative, rel=1.0e-12)
    assert term("axial_angular_momentum_kg_s") == pytest.approx(relative + planetary, rel=1.0e-12)
    assert term("temperature_min_k") == g["temperature"].min()
    assert term("temperature_max_k") == g["temperature"].max()
    assert term("surface_pressure_min_pa") == g["ps"].min()
    assert term("wind_max_m_s") == pytest.approx(
        math.sqrt(float(np.max(g["u"] ** 2 + g["v"] ** 2))), rel=1.0e-15
    )
    assert term("surface_water_min_kg_m2") == state.surface.water_kg_m2.min()
    for name in WATER_SPECIES:
        assert term(f"min_{name}_kg_kg") == g[name].min()


# -- spectra ----------------------------------------------------------------

def _random_spectral(transform, nlev, rng, scale):
    t = transform.truncation
    coeff = np.zeros((nlev, t + 1, t + 1), dtype=np.complex128)
    for n in range(1, t + 1):
        for m in range(n + 1):
            coeff[:, n, m] = rng.normal(size=nlev) + (
                0.0 if m == 0 else 1j * rng.normal(size=nlev)
            )
    return transform.project(coeff * scale)


def test_spectral_kinetic_energy_is_parseval_anchored():
    transform = SphericalHarmonicTransform.create(21)
    rng = np.random.default_rng(7)
    vorticity = _random_spectral(transform, 3, rng, 1.0e-5)
    divergence = _random_spectral(transform, 3, rng, 3.0e-6)
    vector = VorticityDivergenceOperator(transform)
    spectra = SpectralKineticEnergy(transform)

    u, v = vector.wind_from_vordiv(vorticity, divergence)
    grid_ke = np.array([
        transform.grid.global_mean(0.5 * (u[k] ** 2 + v[k] ** 2))
        for k in range(3)
    ])
    by_degree = spectra.by_degree(vorticity) + spectra.by_degree(divergence)
    assert by_degree.shape == (3, 22)
    assert np.all(by_degree[:, 0] == 0.0)
    np.testing.assert_allclose(by_degree.sum(axis=-1), grid_ke, rtol=1.0e-12)

    # Rotational and divergent parts are separately Parseval-anchored: the
    # cross term integrates to zero on the sphere.
    zero = np.zeros_like(divergence)
    u_rot, v_rot = vector.wind_from_vordiv(vorticity, zero)
    rot_ke = transform.grid.global_mean(0.5 * (u_rot[0] ** 2 + v_rot[0] ** 2))
    assert spectra.by_degree(vorticity)[0].sum() == pytest.approx(rot_ke, rel=1.0e-12)

    sample = spectra.unpack(spectra.sample(vorticity, divergence), 3)
    np.testing.assert_allclose(
        sample["rot_total_by_level"], spectra.by_degree(vorticity).sum(axis=-1), rtol=1e-14
    )
    band = spectra.band_start
    assert band == 21 - 2
    np.testing.assert_allclose(
        sample["div_top_decile_by_level"],
        spectra.by_degree(divergence)[:, band:].sum(axis=-1), rtol=1e-14,
    )
    np.testing.assert_allclose(
        sample["rot_by_degree"], spectra.by_degree(vorticity).sum(axis=0), rtol=1e-14
    )


# -- component capture ------------------------------------------------------

def test_component_capture_measures_in_place_updates():
    weights = np.array([0.5, 1.0, 0.5])
    capture = ComponentCapture(weights)
    capture.begin_step()
    fields = {
        name: np.zeros((2, 3, 4)) for name in ("theta", "qv", "u", "v")
    }
    with pytest.raises(ValueError, match="before a 'start' mark"):
        capture.mark("early", fields, np)
    capture.mark("start", fields, np)
    fields["theta"] += 2.0          # in place: the reference suite does this
    fields["qv"][0, 1, 2] = -3.0    # one cell
    capture.mark("a", fields, np)
    fields["u"] = fields["u"] + 1.0  # replaced array: also seen
    capture.mark("b", fields, np)
    capture.mark("start", fields, np)
    capture.mark("c", fields, np)
    records = capture.drain()
    assert [(call, name) for call, name, _ in records] == [(0, "a"), (0, "b"), (1, "c")]
    stats = dict(zip(CAPTURE_NAMES, np.asarray(records[0][2])))
    assert stats["theta_mean_change"] == pytest.approx(2.0)
    assert stats["theta_max_abs_change"] == 2.0
    # qv: one cell of -3 at level 0, latitude 1: zonal mean -0.75, level
    # mean -0.375, weight 1.0 * 0.5 -> -0.1875.
    assert stats["qv_mean_change"] == pytest.approx(-0.1875)
    assert stats["qv_max_abs_change"] == 3.0
    assert stats["u_max_abs_change"] == 0.0
    stats_b = dict(zip(CAPTURE_NAMES, np.asarray(records[1][2])))
    assert stats_b["u_mean_change"] == pytest.approx(1.0)
    assert stats_b["theta_max_abs_change"] == 0.0
    stats_c = dict(zip(CAPTURE_NAMES, np.asarray(records[2][2])))
    assert all(value == 0.0 for value in stats_c.values())
    assert capture.drain() == []
    capture.active = False
    capture.mark("start", fields, np)
    capture.mark("d", fields, np)
    assert capture.drain() == []


# -- tripwire evaluator -----------------------------------------------------

def _row(step, **terms):
    metrics = {
        key: terms.pop(key)
        for key in list(terms)
        if key in ("mass_fixer_log_offset", "global_water_fixer_kg_m2", "levy_kg_m2", "spectral_cfl")
    }
    base = {
        "water_total_kg_m2": 700.0, "kinetic_energy_j_m2": 1.0e6,
        "temperature_min_k": 200.0, "temperature_max_k": 300.0,
        "surface_pressure_min_pa": 60_000.0, "surface_pressure_max_pa": 104_000.0,
        "surface_water_min_kg_m2": 500.0, "water_surface_kg_m2": 500.0,
        **{f"min_{name}_kg_kg": 0.0 for name in WATER_SPECIES},
    }
    base.update(terms)
    return {"kind": "step", "step": step, "time_s": 10.0 * step, "terms": base, "metrics": metrics}


def test_tripwire_set_evaluates_ratios_directions_and_history():
    wires = TripwireSet(
        precision="float64", maximum_cfl=0.75,
        overrides={
            "mass_fixer_jump": 4.0, "temperature_floor_pre_warning": 150.0,
            "spectral_top_decile_growth": 2.0, "surface_reservoir_pre_warning": 0.1,
            "kinetic_energy_step_change": None,
        },
    )
    assert not wires.specs["kinetic_energy_step_change"].enabled
    # The envelope wires arm only once a full trailing window exists: a
    # 100x jump inside the first window is the spin-up, not a trip.
    assert not wires.armed
    assert wires.evaluate_row(_row(1, mass_fixer_log_offset=1.0e-6)) == []
    assert wires.evaluate_row(_row(2, mass_fixer_log_offset=1.0e-4)) == []
    trips = []
    for step in range(3, 21):
        trips += wires.evaluate_row(_row(step, mass_fixer_log_offset=1.0e-6))
    assert trips == [] and wires.armed
    jump = wires.evaluate_row(_row(21, mass_fixer_log_offset=1.0e-4))
    assert [t["tripwire"] for t in jump] == ["mass_fixer_jump"]
    # Trailing mean over the 20-row window: (18 * 1e-6 + 1e-4 + 1e-6) / 20.
    assert jump[0]["value"] == pytest.approx(1.0e-4 / ((19 * 1.0e-6 + 1.0e-4) / 20.0))
    assert jump[0]["term"] == "mass_fixer_log_offset"
    assert jump[0]["threshold"] == 4.0 and jump[0]["step"] == 21
    # A trailing mean below the precision floor is roundoff against roundoff.
    quiet = TripwireSet(precision="float64", maximum_cfl=0.75, overrides={"mass_fixer_jump": 4.0})
    for step in range(1, 21):
        quiet.evaluate_row(_row(step, mass_fixer_log_offset=1.0e-12))
    assert quiet.evaluate_row(_row(21, mass_fixer_log_offset=1.0e-9)) == []
    loud = quiet.evaluate_row(_row(22, mass_fixer_log_offset=1.0e-6))
    assert [t["tripwire"] for t in loud] == ["mass_fixer_jump"]
    # Pre-warnings are absolute and act from the first row.
    fresh = TripwireSet(precision="float64", maximum_cfl=0.75,
                        overrides={"temperature_floor_pre_warning": 150.0})
    assert [t["tripwire"] for t in fresh.evaluate_row(_row(1, temperature_min_k=149.0))] == [
        "temperature_floor_pre_warning"
    ]
    assert wires.evaluate_row(_row(22, temperature_min_k=151.0)) == []
    cold = wires.evaluate_row(_row(23, temperature_min_k=149.0))
    assert [t["tripwire"] for t in cold] == ["temperature_floor_pre_warning"]
    assert cold[0]["direction"] == "min" and cold[0]["value"] == 149.0
    dry = wires.evaluate_row(_row(24, surface_water_min_kg_m2=40.0))
    assert [t["tripwire"] for t in dry] == ["surface_reservoir_pre_warning"]
    assert dry[0]["value"] == pytest.approx(0.08)

    def sample(step, band):
        return {
            "kind": "spectra", "step": step, "time_s": 10.0 * step,
            "rot_top_decile_by_level": band, "div_top_decile_by_level": [0.0] * len(band),
        }

    assert wires.evaluate_spectra(sample(10, [1.0, 1.0])) == []
    assert wires.evaluate_spectra(sample(20, [1.0, 1.0])) == []
    assert wires.evaluate_spectra(sample(30, [1.0, 1.5])) == []
    grown = wires.evaluate_spectra(sample(40, [1.0, 3.0]))
    assert [t["tripwire"] for t in grown] == ["spectral_top_decile_growth"]
    assert grown[0]["term"] == "top_decile_kinetic_energy_level_1"
    assert grown[0]["value"] == pytest.approx(3.0 / ((1.0 + 1.0 + 1.5) / 3.0))
    disabled = TripwireSet(precision="float64", maximum_cfl=0.75, enabled=False,
                           overrides={"temperature_floor_pre_warning": 150.0})
    assert disabled.evaluate_row(_row(1, temperature_min_k=100.0)) == []
    with pytest.raises(ValueError, match="no measured fixer floor"):
        TripwireSet(precision="float16", maximum_cfl=0.75)


def test_every_threshold_cites_a_measurement_or_is_disabled():
    for name, (direction, threshold, measurement) in THRESHOLDS.items():
        assert direction in ("max", "min"), name
        if threshold is None:
            assert "TO BE MEASURED" in measurement, name
        else:
            assert "TO BE MEASURED" not in measurement, name
            assert len(measurement) > 40, name


