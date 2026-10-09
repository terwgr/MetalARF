"""The per-operator total-energy ledger (woof.globe.insitu.energy).

An instrument: per-level signed net of kinetic + internal + potential
energy around every operator of the step, booked from device reductions
and flushed with the ledger batch.  The two properties the Level 7 review
found a pointwise-rectified kinetic-energy ledger cannot have:

* a reversible operator nets ~0 over a period even though every step of
  it moves energy between forms (the semi-implicit map over one linear
  gravity-wave period);
* a dissipative operator nets its analytic decay (hyperdiffusion on a
  single-degree rotational wind, to 1e-6 relative).
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

# CPU only: the device switch is set for each test in this module by
# `conftest._cpu_only_marked_tests` and put back afterwards, because a
# module that set it at import time decided it for the whole session.
pytestmark = pytest.mark.cpu_only

from woof.globe.checkpoint import read_checkpoint  # noqa: E402
from woof.globe.config import DEFAULT_DIFFUSION_EFOLD_S, DEFAULT_DIFFUSION_ORDER, load_config  # noqa: E402
from woof.globe.constants import KAPPA, REFERENCE_PRESSURE_PA, SPECTRAL_FIELDS  # noqa: E402
from woof.globe.insitu import (  # noqa: E402
    ENERGY_COMPONENTS,
    InsituOptions,
    OperatorEnergyLedger,
    STEP_OPERATORS,
)
from woof.globe.insitu.ledger import LEDGER_NAME  # noqa: E402
from woof.globe.runner import build_model_and_cold_state, run  # noqa: E402
from woof.globe.semi_implicit import BarotropicSemiImplicit, VerticalModeSemiImplicit  # noqa: E402
from woof.globe.state import MoistHybridState  # noqa: E402
from woof.globe.vertical import HybridCoordinate  # noqa: E402
from woof.globe.spectral.diffusion import ExponentialHyperdiffusion  # noqa: E402
from woof.globe.spectral.transform import SphericalHarmonicTransform  # noqa: E402

from test_arwen_global_vertical_modes import _column_state, _quiet_model  # noqa: E402

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


def _measure(ledger, atmosphere):
    """The per-level rows of one mark (the band block that follows them
    is read by tests/test_arwen_global_insitu_energy_bands.py)."""
    vector = np.asarray(ledger.measure(atmosphere))
    return vector[:ledger.base_width].reshape(ledger.rows, len(ENERGY_COMPONENTS))


def test_semi_implicit_operator_nets_zero_over_a_linear_wave_period():
    """A standing external-mode gravity wave (degree 6, order 2, about
    20 m/s) on an isothermal rest column; the Crank-Nicolson map at
    alpha = 0.5 rotates it by exactly 2 pi / N per step when omega dt =
    2 tan(pi / N), so after N steps the state is back.  Every step of the
    map moves energy between kinetic and internal + potential, the column
    total moves too within the period at twice the wave frequency (the discrete energy is not an
    invariant of the linearized dynamics) and nets to roundoff over it; a
    ledger that
    rectified the per-step or per-column changes would book the exchange
    as loss (the Level 7 finding: 165x over-credit for this operator)."""
    transform = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    semi = VerticalModeSemiImplicit()
    model = _quiet_model(transform, vertical, semi)
    ledger = OperatorEnergyLedger(model)
    rest = _column_state(
        transform, vertical, lambda n: np.full_like(n, semi.reference_temperature_k),
        ps0=semi.reference_surface_pressure_pa,
    )
    operator = semi.operator(vertical)
    n, m = 6, 2
    k = math.sqrt(n * (n + 1.0)) / transform.grid.radius_m
    # Divergence in the external mode's structure and nothing in the
    # thermodynamic pair: a STANDING wave (two counter-propagating modes),
    # all kinetic at the start, whose global kinetic energy oscillates
    # against the thermodynamic pair over the period.  (A single
    # travelling mode keeps its global kinetic energy constant and would
    # show no exchange at all.)
    v = operator.eigenvectors[:, 0]
    amplitude = 2.0e-5
    state = rest.with_fields([f.copy() for f in rest.fields()])
    state.divergence[:, n, m] = amplitude * v
    steps = 24
    omega = k * operator.phase_speeds_m_s[0]
    dt = 2.0 * math.tan(math.pi / steps) / omega
    start = _measure(ledger, state)
    nets = []
    for _ in range(steps):
        stepped, _ = semi.apply(state, transform, vertical, dt)
        after = _measure(ledger, stepped)
        nets.append(after - _measure(ledger, state))
        state = stepped
    nets = np.asarray(nets)
    column_total = nets[:, :, -1].sum(axis=1)
    # Within the period the operator exchanges forms: kinetic and the
    # thermodynamic pair move in opposite directions step by step, and
    # that traffic is the scale the nets are read against.
    kinetic = nets[:, :-1, 0].sum(axis=1)
    thermal = nets[:, :, 1].sum(axis=1) + nets[:, :, 2].sum(axis=1)
    traffic = float(np.sum(np.abs(kinetic)))
    assert traffic > 1.0e3, traffic
    assert np.corrcoef(kinetic, thermal)[0, 1] < -0.99
    # Over the period the column-total net is roundoff of a 2.8e9 J/m2
    # total against the 3.6e4 J/m2 kinetic traffic, and the map rotated
    # the state back to where it started.  Within the period the total
    # is NOT constant (per-step nets up to 7.3e3 J/m2, antisymmetric over
    # the half period): the discrete total energy is not an invariant of
    # the model's own linearized dynamics (audit 2026-09-01 DN-8 found no
    # discrete energy identity), and the ledger reports that as the signed
    # per-step figure it is, not as a loss.
    assert abs(float(column_total.sum())) < 1.0e-6 * traffic, (column_total.sum(), traffic)
    assert float(np.max(np.abs(column_total))) > 1.0e-2 * traffic
    quarter = steps // 4
    np.testing.assert_allclose(column_total[quarter:2 * quarter], -column_total[:quarter], rtol=1e-6, atol=1e-6 * traffic)
    np.testing.assert_allclose(column_total[2 * quarter:], column_total[:2 * quarter], rtol=1e-6, atol=1e-6 * traffic)
    end = _measure(ledger, state)
    assert abs(float((end - start)[:, -1].sum())) < 1.0e-6 * traffic
    assert float(np.max(np.abs((end - start)[:, 0]))) < 1.0e-6 * traffic


@pytest.mark.parametrize("order,tau", [(4, 14400.0), (DEFAULT_DIFFUSION_ORDER, DEFAULT_DIFFUSION_EFOLD_S)])
@pytest.mark.parametrize("field,strength_name", [("vorticity", None), ("divergence", "divergence_diffusion_strength")])
def test_hyperdiffusion_nets_the_analytic_decay(field, strength_name, order, tau):
    """A single-degree wind (degree 17 of T21, all orders) on a uniform column:
    the exponential hyperdiffusion multiplies the coefficient by f^s, so
    the kinetic energy of every level decays by exactly f^(2 s) - 1 and
    nothing else moves.  The ledger's per-level kinetic net matches that
    to 1e-6 relative; internal, potential and the surface row are
    unchanged to roundoff."""
    transform = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    diffusion = ExponentialHyperdiffusion(order=order, e_folding_time_s_at_truncation=tau, preserve_degree=1)
    model = _quiet_model(transform, vertical, BarotropicSemiImplicit(enabled=False), diffusion=diffusion)
    ledger = OperatorEnergyLedger(model)
    state = _column_state(transform, vertical, lambda n: 250.0 + 30.0 * n)
    rng = np.random.default_rng(3)
    n = 17
    coefficients = np.zeros((vertical.nlev, *transform.spectral_shape), dtype=np.complex128)
    coefficients[:, n, : n + 1] = 3.0e-5 * (rng.standard_normal((vertical.nlev, n + 1)) + 1j * rng.standard_normal((vertical.nlev, n + 1)))
    coefficients[:, n, 0] = coefficients[:, n, 0].real
    setattr(state, field, transform.project(coefficients))
    dt = 1800.0
    before = _measure(ledger, state)
    after = _measure(ledger, model._apply_diffusion(state, dt))
    net = after - before
    strength = 1.0 if strength_name is None else getattr(model, strength_name)
    factor = float(np.asarray(diffusion.factors(transform, dt))[n]) ** strength
    expected = (factor * factor - 1.0) * before[:-1, 0]
    assert factor < 0.99
    np.testing.assert_allclose(net[:-1, 0], expected, rtol=1.0e-6, atol=0.0)
    # Nothing but kinetic energy moves.
    assert float(np.max(np.abs(net[:, 1]))) < 1.0e-9 * float(np.max(np.abs(before[:, 1])))
    assert float(np.max(np.abs(net[:, 2]))) < 1.0e-9 * float(np.max(np.abs(before[:, 2])))
    np.testing.assert_allclose(net[:-1, 3], expected, rtol=1.0e-6, atol=0.0)
    # The column total is the Kasahara invariant: kinetic + cp T + Phi_s ps.
    assert abs(float(before[:, 3].sum()) - float(before[:-1, 0].sum() + before[:-1, 1].sum() + before[:, 2].sum())) < 1e-6 * float(before[:, 3].sum())


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_ledger_books_every_operator_of_the_step_and_stays_bit_identical(tmp_path):
    """The runtime marks every operator of the step at the energy cadence
    (both split and IMEX dynamics), the rows carry the per-level nets and
    the receipt sums the column totals; and a run with the marks on is
    bit-identical to one with them off (the instrument reads only)."""
    cfg = load_config(CONFIG)
    steps = 6
    hashes = {}
    spectra_by_run = {}
    for name, integrator, every in (("split", "ssprk3", 2), ("imex", "imex_ssp3", 3), ("off", "ssprk3", 1000), ("imex_off", "imex_ssp3", 1000)):
        this = dataclasses.replace(
            cfg, integrator=integrator, duration_s=cfg.dt_s * steps, output_interval_s=cfg.dt_s * steps,
            # spectra_every = 2 makes the spectra sample and the energy
            # marks share flushed steps: the buffer is consumed in the
            # order it was packed (a 2026-09-02 T255 arm read 7e9 J/m2
            # energy marks as top-decile kinetic energy and tripped the
            # spectral growth tripwire at step 50).
            insitu=InsituOptions(energy_every=every, flush_every=4, spectra_every=2),
        )
        result = run(this, tmp_path / name)
        assert result["status"] == "pass"
        metadata, _ = read_checkpoint(tmp_path / name / f"arwen_global_step{steps:08d}.npz")
        hashes[name] = {key: value["sha256"] for key, value in metadata["arrays"].items()}
        all_rows = _rows(tmp_path / name / LEDGER_NAME)
        rows = [row for row in all_rows if row["kind"] == "step"]
        sampled = [row for row in rows if "energy" in row]
        spectra = {row["step"]: row for row in all_rows if row["kind"] == "spectra"}
        assert sorted(spectra) == [2, 4, 6]
        for row in spectra.values():
            assert all(0.0 <= v < 1.0e6 for v in row["rot_top_decile_by_level"])
            assert all(0.0 <= v < 1.0e6 for v in row["div_top_decile_by_level"])
        spectra_by_run[name] = {step: row["rot_top_decile_by_level"] for step, row in spectra.items()}
        assert result["insitu"]["trip_count"] == 0
        summary = result["insitu"]["energy"]
        assert summary["sampled_every"] == every
        if every > steps:
            assert not sampled and summary["sampled_steps"] == 0
            continue
        assert [row["step"] for row in sampled] == [s for s in range(1, steps + 1) if s % every == 0]
        expected = ["physics_first", "positivity_first"]
        # The IMEX pair marks its explicit sum and its implicit sum apart (2026-09-04).
        expected += ["dynamics_explicit", "dynamics_implicit"] if integrator == "imex_ssp3" else ["semi_implicit_pre", "explicit_dynamics", "semi_implicit_post"]
        # The grid tracers' transport marks after the mass fixer (2026-09-02).
        expected += ["diffusion", "mass_fixer", "tracer_transport", "physics_second", "positivity_second"]
        assert set(expected) <= set(STEP_OPERATORS)
        for row in sampled:
            energy = row["energy"]
            assert energy["operators"] == expected
            assert energy["components"] == list(ENERGY_COMPONENTS)
            assert energy["rows"] == cfg.vertical.nlev + 1
            for op in expected:
                assert len(energy["net"][op]) == cfg.vertical.nlev + 1
                assert len(energy["net"][op][0]) == len(ENERGY_COMPONENTS)
            # The dry column total is the sum of the rows' last component.
            assert abs(energy["column_total_net"]["diffusion"] - sum(r[-1] for r in energy["net"]["diffusion"])) < 1e-6
            # Diffusion removes energy; the physics moves it (nonzero).
            assert energy["column_total_net"]["diffusion"] < 0.0
            assert energy["column_total_net"]["physics_first"] != 0.0
            # The per-band, per-hemisphere kinetic booking (2026-09-02):
            # one [northern, southern] pair per band per operator, the
            # opening measure beside it, and hyperdiffusion removes
            # kinetic energy from every band it reaches (the top band
            # by the largest fraction of what it holds).
            bands = energy["bands"]
            assert bands["labels"] == [f"n{lo:03d}-{hi:03d}" for lo, hi in bands["edges"]]
            assert bands["edges"][-1][1] == cfg.truncation
            start = energy["band_start"]["column_kinetic_j_m2"]
            assert len(start["bands"]) == len(bands["labels"])
            assert all(v > 0.0 for pair in start["bands"] for v in pair)
            for op in expected:
                entry = energy["band_net"][op]
                assert len(entry["column_kinetic_j_m2"]["bands"]) == len(bands["labels"])
                assert len(entry["level_kinetic_m2_s2"]["bands"]) == len(bands["labels"])
                assert all(len(pair) == 2 for pair in entry["column_kinetic_j_m2"]["bands"])
            diffusion = energy["band_net"]["diffusion"]["column_kinetic_j_m2"]["bands"]
            fractions = [
                -(pair[0] + pair[1]) / (held[0] + held[1])
                for pair, held in zip(diffusion, start["bands"])
            ]
            assert all(f >= 0.0 for f in fractions), fractions
            assert fractions[-1] == max(fractions)
        assert summary["sampled_steps"] == len(sampled)
        assert set(summary["column_total_net_j_m2"]) == set(expected)
        assert summary["bands"]["labels"] == sampled[0]["energy"]["bands"]["labels"]
        assert set(summary["band_column_kinetic_net_j_m2"]) == set(expected)
        labels = summary["bands"]["labels"]
        for op in expected:
            per_band = summary["band_column_kinetic_net_j_m2"][op]
            assert set(per_band) == set(labels)
            for index, label in enumerate(labels):
                value = per_band[label]
                assert value["global"] == pytest.approx(value["northern"] + value["southern"], abs=1e-9)
                booked = sum(
                    sum(row["energy"]["band_net"][op]["column_kinetic_j_m2"]["bands"][index])
                    for row in sampled
                )
                assert value["global"] == pytest.approx(booked, rel=1e-9, abs=1e-9)
        assert set(summary["mean_band_column_kinetic_j_m2"]) == set(labels)
        assert summary["column_total_net_j_m2"]["diffusion"] == pytest.approx(
            sum(row["energy"]["column_total_net"]["diffusion"] for row in sampled), rel=1e-9,
        )
        assert summary["wall_seconds"] > 0.0
    assert hashes["split"] == hashes["off"]
    # The IMEX marks (the explicit-sum accumulator of imex_step, 2026-09-04)
    # read only too: the IMEX run with marks every third step is
    # bit-identical to the IMEX run with none.
    assert hashes["imex"] == hashes["imex_off"]
    assert hashes["imex"] != hashes["split"]
    # The spectra samples of the sampled and unsampled split runs agree
    # exactly: the energy marks never displace them in the flush.
    assert spectra_by_run["split"] == spectra_by_run["off"]
