"""The column water-budget instrument: calibration families in both
directions, the checkpoint reader against the model's own synthesis and
the in-situ ledger row, and the CLI over a smoke run.

Every planted number is read back to float64 rounding except the band
transport term, whose stated tolerance is the quadrature error of the
band area at the synthetic grid's resolution (recorded in the module's
CALIBRATION table).
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from woof.globe import water_budget as wb
from woof.globe.checkpoint import write_checkpoint
from woof.globe.config import load_config
from woof.globe.constants import GRAVITY_M_S2, WATER_SPECIES
from woof.globe.insitu.budgets import LEDGER_INDEX, LedgerGeometry, ledger_row
from woof.globe.runner import build_model_and_cold_state, run

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
ROUNDOFF = 1.0e-12


@pytest.fixture(scope="module")
def grid():
    return wb.synthetic_grid(21)


# -- family A: a planted precipitation sink, no evaporation ------------------


@pytest.mark.parametrize("magnitude", [0.1, 1.0, 10.0])
def test_planted_precipitation_reads_back_and_evaporation_reads_zero(grid, magnitude):
    before, after, planted = wb.synthetic_pair(
        grid, p_conv_kg_m2=0.3 * magnitude, p_grid_kg_m2=0.7 * magnitude
    )
    g = wb.interval_budget(before, after, grid)["regions"][wb.GLOBAL]
    p_true = grid.area_mean(planted["P_conv"] + planted["P_grid"])
    conv_true = grid.area_mean(planted["P_conv"])
    assert g["P"] == pytest.approx(p_true, rel=ROUNDOFF)
    assert g["P_conv"] == pytest.approx(conv_true, rel=ROUNDOFF)
    assert g["P_grid"] == pytest.approx(p_true - conv_true, rel=ROUNDOFF)
    assert abs(g["E_books"]) <= ROUNDOFF * p_true
    assert abs(g["E_flux"]) <= ROUNDOFF * p_true
    assert abs(g["residual"]) <= ROUNDOFF * p_true
    assert g["dW_condensate"] == pytest.approx(-p_true, rel=ROUNDOFF)
    assert g["dW_condensate_by_species"]["qr"] == pytest.approx(-p_true, rel=ROUNDOFF)
    assert abs(g["dW_vapor"]) <= ROUNDOFF * p_true
    assert g["rates_mm_day"]["P"] == pytest.approx(p_true * 8.0, rel=ROUNDOFF)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("magnitude", [0.1, 1.0, 10.0])
def test_unbooked_leak_reads_back_as_the_residual_with_its_sign(grid, sign, magnitude):
    before, after, planted = wb.synthetic_pair(
        grid, p_conv_kg_m2=0.3, p_grid_kg_m2=0.7, leak_kg_m2=sign * magnitude
    )
    g = wb.interval_budget(before, after, grid)["regions"][wb.GLOBAL]
    leak = grid.area_mean(planted["leak"])
    assert g["residual"] == pytest.approx(leak, rel=ROUNDOFF)
    assert g["residual_fraction"] == pytest.approx(leak / max(abs(g["E_books"]), abs(g["P"])), rel=ROUNDOFF)
    # The leak touches neither book: P and E are what was planted.
    assert g["P"] == pytest.approx(grid.area_mean(planted["P_conv"] + planted["P_grid"]), rel=ROUNDOFF)
    assert abs(g["E_books"]) <= ROUNDOFF


# -- family B: a planted evaporation source, no precipitation ----------------


@pytest.mark.parametrize("magnitude", [0.1, 1.0, 10.0])
def test_planted_evaporation_reads_back_from_the_books_and_the_flux(grid, magnitude):
    before, after, planted = wb.synthetic_pair(grid, e_kg_m2=magnitude)
    g = wb.interval_budget(before, after, grid)["regions"][wb.GLOBAL]
    e_true = grid.area_mean(planted["E"])
    assert g["E_books"] == pytest.approx(e_true, rel=ROUNDOFF)
    assert g["E_flux"] == pytest.approx(e_true, rel=ROUNDOFF)
    assert abs(g["P"]) <= ROUNDOFF * e_true
    assert abs(g["residual"]) <= ROUNDOFF * e_true
    assert g["dW_vapor"] == pytest.approx(e_true, rel=ROUNDOFF)
    assert abs(g["dW_condensate"]) <= ROUNDOFF * e_true


def test_dew_reads_back_as_negative_evaporation(grid):
    before, after, planted = wb.synthetic_pair(grid, e_kg_m2=-0.5)
    g = wb.interval_budget(before, after, grid)["regions"][wb.GLOBAL]
    e_true = grid.area_mean(planted["E"])
    assert e_true < 0.0
    assert g["E_books"] == pytest.approx(e_true, rel=ROUNDOFF)
    assert g["E_flux"] == pytest.approx(e_true, rel=ROUNDOFF)
    assert abs(g["residual"]) <= ROUNDOFF * abs(e_true)


def test_the_fixer_correction_is_kept_out_of_evaporation(grid):
    before, after, planted = wb.synthetic_pair(grid, e_kg_m2=1.0, fixer_kg_m2=0.05)
    e_true = grid.area_mean(planted["E"])
    with_fixer = wb.interval_budget(before, after, grid, fixer_kg_m2=0.05)["regions"][wb.GLOBAL]
    assert with_fixer["E_books"] == pytest.approx(e_true, rel=ROUNDOFF)
    assert abs(with_fixer["residual"]) <= ROUNDOFF
    # Without the ledger the fixer's credit to the reservoir reads as
    # less evaporation and the atmosphere's true gain surfaces as an
    # unbooked source of the same size; budget() states the assumption.
    without = wb.interval_budget(before, after, grid)["regions"][wb.GLOBAL]
    assert without["E_books"] == pytest.approx(e_true - 0.05, rel=ROUNDOFF)
    assert without["residual"] == pytest.approx(0.05, rel=ROUNDOFF)


# -- family C: the band decomposition ----------------------------------------


def test_hemispheric_split_lands_in_its_bands(grid):
    # >= 0 so an equatorial row (odd nlat) sits with the 0-30N band.
    north = grid.latitude_deg >= 0.0
    before, after, planted = wb.synthetic_pair(
        grid, p_conv_kg_m2=0.6, p_grid_kg_m2=1.4, e_kg_m2=2.0, p_rows=north, e_rows=~north
    )
    regions = wb.interval_budget(before, after, grid)["regions"]
    masks = wb.region_masks(grid, None)
    for name, south, _north in wb.BANDS:
        r = regions[name]
        p_true = grid.area_mean(planted["P_conv"] + planted["P_grid"], masks[name])
        e_true = grid.area_mean(planted["E"], masks[name])
        assert r["P"] == pytest.approx(p_true, rel=ROUNDOFF, abs=ROUNDOFF)
        assert r["E_books"] == pytest.approx(e_true, rel=ROUNDOFF, abs=ROUNDOFF)
        assert abs(r["residual"]) <= ROUNDOFF * max(abs(e_true), abs(p_true))
        if south >= 0.0:
            assert p_true > 0.0 and e_true == 0.0
        else:
            assert e_true > 0.0 and p_true == 0.0
    # Land and ocean carry the exchanges but no residual (no closed edges).
    assert regions[wb.LAND]["residual"] is None
    assert regions[wb.LAND]["residual_status"] == "not separable from transport"
    assert regions[wb.OCEAN]["E_books"] > 0.0


# -- family D: a planted transport -------------------------------------------


@pytest.mark.parametrize("truncation,bar", [(21, 3.0e-3), (63, 4.0e-4)])
@pytest.mark.parametrize("v", [1.0, -1.0])
def test_planted_transport_reads_back_within_the_stated_bar(truncation, bar, v):
    tgrid = wb.synthetic_grid(truncation)
    before, after, planted = wb.synthetic_pair(tgrid, transport_v_m_s=v)
    regions = wb.interval_budget(before, after, tgrid)["regions"]
    masks = wb.region_masks(tgrid, None)
    for name, _s, _n in wb.BANDS:
        r = regions[name]
        t_true = tgrid.area_mean(planted["transport"], masks[name])
        assert t_true != 0.0
        # Without the transport term the band residual IS the planted
        # convergence; with it the residual drops to the quadrature error.
        without = r["dW_atmosphere"] - r["physics_net_e_minus_p"]
        assert without == pytest.approx(t_true, rel=ROUNDOFF)
        assert abs(r["transport_convergence"] - t_true) <= bar * abs(t_true)
        assert abs(r["residual"]) <= bar * abs(t_true)
    assert regions[wb.GLOBAL]["transport_convergence"] == 0.0
    # The planted band convergences use analytic band areas, so their
    # area-weighted sum on the quadrature is not exactly zero; the global
    # residual is that family mismatch (4e-11 kg/m2 at T21 on 25 kg/m2),
    # not an instrument error.
    assert abs(regions[wb.GLOBAL]["residual"]) <= 1.0e-9


def test_edge_flux_of_a_uniform_flux_is_the_circle_length_times_it(grid):
    northward = np.full(grid.shape[0], 3.0)
    for edge in (-45.0, 0.0, 30.0):
        expected = 2.0 * np.pi * grid.radius_m * np.cos(np.deg2rad(edge)) * 3.0
        assert wb.edge_flux_kg_s(grid, northward, edge) == pytest.approx(expected, rel=ROUNDOFF)
    assert wb.edge_flux_kg_s(grid, northward, 90.0) == 0.0
    assert wb.edge_flux_kg_s(grid, northward, -90.0) == 0.0


# -- the calibration table ----------------------------------------------------


def test_every_calibration_row_meets_its_bar():
    payload = wb.calibrate()
    rows = payload["rows"]
    assert {row["family"] for row in rows} == {"A", "A'", "B", "B'", "B''", "C", "D"}
    for row in rows:
        family = row["family"]
        if family == "A":
            assert row["P_relative_error"] <= ROUNDOFF
            assert row["P_conv_relative_error"] <= ROUNDOFF
            assert row["E_over_P"] <= ROUNDOFF
            assert row["residual_over_P"] <= ROUNDOFF
        elif family == "A'":
            assert row["leak_relative_error"] <= ROUNDOFF
            assert np.sign(row["residual_read"]) == np.sign(row["leak_planted"])
        elif family in ("B", "B'", "B''"):
            assert row["E_relative_error"] <= ROUNDOFF
            if "E_flux_relative_error" in row:
                assert row["E_flux_relative_error"] <= ROUNDOFF
        elif family == "C":
            assert row["residual_over_max"] <= ROUNDOFF
        elif family == "D":
            bar = 3.0e-3 if row["truncation"] == 21 else 4.0e-4
            assert row["transport_relative_error"] <= bar
    assert wb.main(["--calibrate"]) == 0


# -- refusals -----------------------------------------------------------------


def test_budget_refuses_a_series_it_cannot_difference(grid):
    before, after, _ = wb.synthetic_pair(grid, e_kg_m2=1.0)
    with pytest.raises(ValueError, match="at least 2"):
        wb.budget([before], grid)
    with pytest.raises(ValueError, match="strictly increasing"):
        wb.budget([after, before], grid)
    with pytest.raises(ValueError, match="one value per interval"):
        wb.budget([before, after], grid, fixer_kg_m2=[0.0, 0.0])
    with pytest.raises(ValueError, match="exactly"):
        wb.ColumnSample(
            time_s=0.0, step=0, vapor=before.vapor, condensate={"qc": before.vapor},
            reservoir=before.reservoir, convective=before.convective, grid_scale=before.grid_scale,
        )


# -- the checkpoint reader against the model and the ledger --------------------


def _receipt_for(cfg, model) -> dict:
    grid = model.transform.grid
    return {
        "config_hash": cfg.config_hash,
        "config": {
            "truncation": cfg.truncation,
            "dealias_factor": cfg.dealias_factor,
            "a_half_pa": [float(v) for v in cfg.vertical.a_half_pa],
            "b_half": [float(v) for v in cfg.vertical.b_half],
        },
        "transform": {"nlat": grid.nlat, "nlon": grid.nlon, "radius_m": grid.radius_m},
    }


def test_checkpoint_reader_matches_the_model_synthesis_and_the_ledger_row(tmp_path):
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    for _ in range(2):
        state, _metrics = model.step(state, cfg.dt_s)
    path = write_checkpoint(
        tmp_path / "arwen_global_step00000002.npz", state,
        config_hash=cfg.config_hash, to_numpy=model.transform.backend.to_numpy,
        semi_implicit_scheme=cfg.semi_implicit_scheme,
    )
    reader = wb.CheckpointReader(_receipt_for(cfg, model))
    sample = reader.sample(path)
    assert sample.step == 2 and sample.time_s == pytest.approx(2.0 * cfg.dt_s)

    g = model.grid_state(state.atmosphere, only=("dp", "v", *WATER_SPECIES))
    dp_g = g["dp"] / GRAVITY_M_S2
    np.testing.assert_allclose(sample.vapor, np.sum(g["qv"] * dp_g, axis=0), rtol=1.0e-10, atol=1e-300)
    for species in wb.CONDENSATE_SPECIES:
        np.testing.assert_allclose(
            sample.condensate[species], np.sum(g[species] * dp_g, axis=0), rtol=1.0e-10, atol=1.0e-13
        )
    total = sum(g[s] for s in WATER_SPECIES)
    expected_transport = np.mean(np.sum(g["v"] * total * dp_g, axis=0), axis=-1)
    np.testing.assert_allclose(sample.northward_transport, expected_transport, rtol=1.0e-9, atol=1.0e-12)

    row = np.asarray(ledger_row(model, state, LedgerGeometry(model.transform, model.rotation_rate_s)), dtype=np.float64)
    assert reader.grid.area_mean(sample.vapor) == pytest.approx(row[LEDGER_INDEX["water_qv_kg_m2"]], rel=1.0e-10)
    assert reader.grid.area_mean(sample.atmosphere) == pytest.approx(row[LEDGER_INDEX["water_atmosphere_kg_m2"]], rel=1.0e-10)
    reservoir = sum(
        row[LEDGER_INDEX[name]]
        for name in ("water_surface_kg_m2", "water_soil_kg_m2", "water_native_kg_m2", "water_outflow_kg_m2")
    )
    assert reader.grid.area_mean(sample.reservoir) == pytest.approx(reservoir, rel=1.0e-10)
    assert reader.grid.area_mean(sample.atmosphere + sample.reservoir) == pytest.approx(
        row[LEDGER_INDEX["water_total_kg_m2"]], rel=1.0e-10
    )


def test_cli_over_a_smoke_run_closes_the_global_budget_on_the_fixer(tmp_path):
    cfg = load_config(CONFIG)
    cfg = replace(cfg, duration_s=4.0 * cfg.dt_s, output_interval_s=2.0 * cfg.dt_s)
    outdir = tmp_path / "run"
    result = run(cfg, outdir)
    assert result["status"] == "pass"
    out = tmp_path / "budget.json"
    assert wb.main(["--run-dir", str(outdir), "--out", str(out), "--label", "smoke", "--quiet"]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == wb.SCHEMA
    assert payload["sampling"]["steps"] == [0, 2, 4]
    assert payload["ledger"]["path"].endswith("insitu.ndjson")
    # The checkpoint reads and the ledger rows are the same numbers.
    for check in payload["ledger"]["checks"]:
        assert abs(check["atmosphere_relative_gap"]) < 1.0e-9
        assert abs(check["reservoir_relative_gap"]) < 1.0e-9
    # With the fixer pinning the total, the global residual of every
    # interval is minus the fixer's correction: the water the dynamics
    # moved in or out of the atmosphere without a booking.
    for interval, ledger_interval in zip(payload["intervals"], payload["ledger"]["intervals"]):
        g = interval["regions"][wb.GLOBAL]
        assert g["transport_convergence"] == 0.0
        assert g["residual"] == pytest.approx(-interval["fixer_kg_m2"], abs=1.0e-9)
        if "residual" in ledger_interval:
            assert ledger_interval["residual"] == pytest.approx(g["residual"], abs=1.0e-9)
            assert ledger_interval["physics_net_e_minus_p"] == pytest.approx(g["physics_net_e_minus_p"], abs=1.0e-9)
    cumulative = payload["cumulative"][wb.GLOBAL]
    assert cumulative["residual_status"] == "measured"
    assert cumulative["hours"] == pytest.approx(4.0 * cfg.dt_s / 3600.0)
    assert "water budget smoke" in payload["summary"]
    assert set(payload["cumulative"]) >= {wb.GLOBAL, wb.LAND, wb.OCEAN, *[b[0] for b in wb.BANDS]}
    # The cold-start checkpoint is named in the record, not silently zeroed.
    assert any("step 0" in note for note in payload["sampling"]["notes"])


def test_no_transport_leaves_band_residuals_unformed(tmp_path):
    cfg = load_config(CONFIG)
    cfg = replace(cfg, duration_s=2.0 * cfg.dt_s, output_interval_s=cfg.dt_s)
    outdir = tmp_path / "run"
    run(cfg, outdir)
    out = tmp_path / "budget.json"
    assert wb.main(["--run-dir", str(outdir), "--out", str(out), "--no-transport", "--ledger", "none", "--quiet"]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ledger"] == {"status": "absent"}
    assert payload["fixer"].startswith("no ledger")
    band = payload["intervals"][0]["regions"][wb.BANDS[0][0]]
    assert band["residual"] is None
    assert band["transport_status"] == "no transport field in the samples"
    assert payload["intervals"][0]["regions"][wb.GLOBAL]["residual"] is not None
