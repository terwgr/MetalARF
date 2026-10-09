"""2 m / 10 m diagnostics: similarity formulation, reference suite, export.

Audit 2026-09-01 (task 1b): the render tape wrote T2 = temperature[-1],
the lowest full level (378 m up on the 20-level grid, 23 m on the 40-level
one), and Q2/U10/V10 likewise.  Now the suites persist screen-level fields
in the physics state, the export reads them, and a tape that had to be
diagnosed at export says so.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_netcdf_writer  # noqa: E402

from woof.globe.config import load_config
from woof.globe.constants import KAPPA
from woof.globe.physics.surface_diagnostics import (
    ANEMOMETER_HEIGHT_M,
    SIMILARITY_SOURCE,
    SOURCE_METADATA_KEY,
    STABLE_ZETA_CAP,
    UNSTABLE_ZETA_FLOOR,
    effective_surface_humidity,
    similarity_surface_diagnostics,
)
from woof.globe.runner import build_model_and_cold_state, run
from woof.globe.wrfout_export import (
    EXPORT_FALLBACK_SOURCE,
    EXPORT_RECEIPT_NAME,
    MODEL_LABEL,
    MODEL_LABEL_ATTR,
    SURFACE_DIAGNOSTICS_ATTR,
    export_wrfout,
)

SMOKE_CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


def _column(*, skin_k, air_k=288.0, wind=5.0, z1=23.0, z0=0.1, qv=0.008, q_s=None):
    """One column with the lowest full level z1 m up in an isothermal layer."""
    ps = np.array([101_325.0])
    p_full = ps * np.exp(-9.80665 * z1 / (287.05 * air_k * (1.0 + 0.61 * qv)))
    return dict(
        u_lowest=np.array([wind]), v_lowest=np.array([0.0]),
        temperature_lowest=np.array([air_k]), qv_lowest=np.array([qv]),
        p_full_lowest_pa=p_full, p_surface_pa=ps,
        skin_temperature_k=np.array([skin_k]),
        surface_humidity=np.array([qv if q_s is None else q_s]),
        roughness_m=np.array([z0]),
    )


def _neutral_skin(air_k=288.0, z1=23.0, qv=0.008):
    """The skin temperature whose potential temperature equals the air's."""
    ps = 101_325.0
    p_full = ps * np.exp(-9.80665 * z1 / (287.05 * air_k * (1.0 + 0.61 * qv)))
    theta = air_k * (1.0e5 / p_full) ** KAPPA
    return theta * (ps / 1.0e5) ** KAPPA


def test_neutral_layer_reduces_to_the_log_law():
    out = similarity_surface_diagnostics(**_column(skin_k=_neutral_skin()))
    assert out["zeta"] == pytest.approx(0.0, abs=1.0e-9)
    z1 = float(out["z_lowest_m"][0])
    assert 20.0 < z1 < 26.0
    # U10 = U1 ln(10/z0) / ln(z1/z0), exactly, with no stability term.
    expected = 5.0 * np.log(10.0 / 0.1) / np.log(z1 / 0.1)
    assert out["u10"][0] == pytest.approx(expected, rel=1.0e-9)
    assert out["v10"][0] == 0.0
    # theta is uniform, so th2 is the air's theta and T2 sits on the dry
    # adiabat 2 m above the surface: warmer than the 23 m level by ~0.2 K.
    theta = 288.0 * (1.0e5 / _column(skin_k=288.0)["p_full_lowest_pa"][0]) ** KAPPA
    assert out["th2"][0] == pytest.approx(theta, rel=1.0e-9)
    assert 288.15 < out["t2"][0] < 288.30
    assert out["q2"][0] == pytest.approx(0.008)


def test_stable_and_unstable_layers_bracket_the_2m_value_between_skin_and_air():
    stable = similarity_surface_diagnostics(**_column(skin_k=280.0))
    unstable = similarity_surface_diagnostics(**_column(skin_k=298.0))
    assert 0.0 < stable["zeta"][0] <= STABLE_ZETA_CAP
    assert UNSTABLE_ZETA_FLOOR <= unstable["zeta"][0] < 0.0
    assert 280.0 < stable["t2"][0] < 288.0
    assert 288.0 < unstable["t2"][0] < 298.0
    # Stable stratification suppresses the 10 m wind relative to neutral,
    # unstable mixing raises it.
    neutral = similarity_surface_diagnostics(**_column(skin_k=_neutral_skin()))
    assert stable["u10"][0] < neutral["u10"][0] < unstable["u10"][0]
    assert 0.0 < stable["u10"][0] and unstable["u10"][0] < 5.0


def test_a_lowest_level_at_ten_metres_returns_its_own_wind():
    out = similarity_surface_diagnostics(**_column(skin_k=_neutral_skin(z1=10.0), z1=10.0))
    assert out["z_lowest_m"][0] == pytest.approx(ANEMOMETER_HEIGHT_M, abs=0.05)
    assert out["u10"][0] == pytest.approx(5.0, abs=0.01)


def test_the_378m_lowest_level_of_the_legacy_grid_no_longer_leaks_into_t2():
    # The audit's case: lowest level ~378 m up, a 6.5 K/km atmosphere puts
    # it 2.5 K below the surface air; the old export wrote that as T2.
    skin = 300.0
    air = skin - 0.0065 * 378.0
    out = similarity_surface_diagnostics(**_column(skin_k=skin, air_k=air, z1=378.0))
    # theta rises 1.3 K over the layer (a 6.5 K/km lapse is statically
    # stable), so the 2 m value sits within half a kelvin of the skin and
    # well above the 297.5 K the old export wrote.
    assert out["t2"][0] > air + 1.0
    assert abs(out["t2"][0] - skin) < 0.5


def test_surface_humidity_follows_the_bucket_beta():
    skin = np.array([300.0, 300.0, 300.0])
    ps = np.full(3, 101_325.0)
    land = np.array([0.0, 1.0, 1.0])
    wetness = np.array([0.0, 1.0, 0.25])
    qv = np.full(3, 0.010)
    q_s = effective_surface_humidity(skin, ps, land, wetness, qv, np)
    from woof.globe.physics.reference import saturation_mixing_ratio
    qsat = float(saturation_mixing_ratio(np.array([300.0]), np.array([101_325.0]), np)[0])
    assert q_s[0] == pytest.approx(qsat)          # ocean: saturated
    assert q_s[1] == pytest.approx(qsat)          # wet land: saturated
    assert q_s[2] == pytest.approx(0.010 + 0.25 * (qsat - 0.010))


def test_calm_air_and_rough_surfaces_stay_finite():
    out = similarity_surface_diagnostics(**_column(skin_k=310.0, wind=0.0, z0=5.0))
    for name in ("t2", "th2", "q2", "u10", "v10"):
        assert np.isfinite(out[name]).all()
    assert out["u10"][0] == 0.0


def test_the_reference_suite_persists_screen_level_fields():
    cfg = load_config(SMOKE_CONFIG)
    model, state = build_model_and_cold_state(cfg)
    exchange = model._physics_exchange(state, cfg.dt_s)
    result = model.physics.step(exchange)
    arrays = result.physics_state.arrays
    for name in ("t2", "th2", "q2", "u10", "v10"):
        assert name in arrays, name
        assert arrays[name].shape == exchange.grid_shape
        assert np.isfinite(arrays[name]).all()
    assert result.physics_state.metadata[SOURCE_METADATA_KEY] == SIMILARITY_SOURCE
    # Not the lowest level: on this 4-level smoke grid the lowest full
    # level is ~2 km up, so a 2 m value must differ from it everywhere.
    lowest_t = np.asarray(result.theta[-1]) * np.asarray(exchange.exner[-1])
    assert np.all(np.abs(arrays["t2"] - lowest_t) > 0.1)
    assert np.all(np.abs(arrays["u10"]) <= np.abs(np.asarray(result.u[-1])) + 1.0e-9)


@requires_netcdf_writer
def test_export_writes_screen_fields_from_the_physics_state_and_labels_the_fallback(tmp_path):
    cfg = load_config(SMOKE_CONFIG)
    receipt = run(cfg, tmp_path / "run")
    checkpoints = [Path(p) for p in receipt["checkpoints"]]
    assert len(checkpoints) == 3
    tapes = export_wrfout(
        cfg, checkpoints, tmp_path / "tapes", nlat=18, nlon=36,
        start_date="2026-08-30_18:00:00",
    )
    assert len(tapes) == 3
    export_receipt = json.loads(
        (tmp_path / "tapes" / EXPORT_RECEIPT_NAME).read_text(encoding="utf-8")
    )
    rows = export_receipt["tapes"]
    assert [row["tape"] for row in rows] == [str(p) for p in tapes]
    # The cold checkpoint (step 0) has no physics state yet: diagnosed at
    # export and labelled so.  Every later checkpoint carries the suite's.
    assert rows[0]["surface_diagnostics"] == EXPORT_FALLBACK_SOURCE
    assert all(row["surface_diagnostics"] == SIMILARITY_SOURCE for row in rows[1:])
    for label in (EXPORT_FALLBACK_SOURCE, SIMILARITY_SOURCE):
        assert label in export_receipt["screen_level_sources"]
    assert export_receipt["vertical"]["coordinate"] == cfg.vertical_coordinate

    from netCDF4 import Dataset

    for tape, row in zip(tapes, rows):
        with Dataset(tape) as ds:
            assert ds.getncattr(SURFACE_DIAGNOSTICS_ATTR) == row["surface_diagnostics"]
            # The renderer names this model, not "WRF", in the metadata row.
            assert ds.getncattr(MODEL_LABEL_ATTR) == MODEL_LABEL
            t2 = np.asarray(ds["T2"][0])
            lowest_theta = np.asarray(ds["T"][0][0]) + 300.0
            lowest_t = lowest_theta * (np.asarray(ds["PB"][0][0]) / 1.0e5) ** KAPPA
            assert np.isfinite(t2).all()
            # A 2 m value, not the (2 km) lowest level.
            assert np.all(np.abs(t2 - lowest_t) > 0.1)
            assert np.isfinite(np.asarray(ds["Q2"][0])).all()
            assert np.isfinite(np.asarray(ds["U10"][0])).all()
