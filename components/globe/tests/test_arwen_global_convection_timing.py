"""The convection-timing instruments: the trigger diagnostic and the storm reader.

Both are calibrated on synthetic families in both directions before any
number of theirs is cited: a planted reading is read back exactly, and
the absence of the thing (no inactive columns, no storm, an atmosphere
at rest) reads as zero.  The families are the instruments' own
``calibrate()`` rows plus the contracts a run reading relies on (the
regional binning, the config fallback for a run still writing, the
refusal of a checkpoint set without physics state).
"""
from __future__ import annotations

import math

import numpy as np
import pytest


from woof.globe import column_sounding as cs
from woof.globe import storm_reader as sr
from woof.globe import trigger_diagnostic as td


# --------------------------------------------------------------------------
# calibration rows, both instruments
# --------------------------------------------------------------------------


def test_trigger_diagnostic_calibration_rows_all_meet_their_bars():
    payload = td.calibrate()
    failed = [r for r in payload["rows"] if not r["ok"]]
    assert payload["all_ok"], failed
    families = {r["family"] for r in payload["rows"]}
    assert {"inactive fraction", "lead of first grid rain", "analytic CAPE", "analytic CIN",
            "column state, cloudy levels", "column state, no cloud"} <= families
    # The planted inactive fractions span both directions: none, a part, all.
    planted = sorted(r["planted"] for r in payload["rows"] if r["family"] == "inactive fraction")
    assert planted[0] == 0.0 and planted[-1] == 1.0 and 0.0 < planted[1] < 1.0


def test_storm_reader_calibration_rows_all_meet_their_bars():
    payload = sr.calibrate()
    failed = [r for r in payload["rows"] if not r["ok"]]
    assert payload["all_ok"], failed
    families = {r["family"] for r in payload["rows"]}
    assert {"storm cell accumulation", "storm cell peak rate", "no storm: uniform rain",
            "omega from a two-layer divergence", "omega at rest"} <= families


# --------------------------------------------------------------------------
# the trigger reading is a number: planted fractions read back exactly
# --------------------------------------------------------------------------


@pytest.mark.parametrize("planted", [0.0, 0.25, 0.6, 1.0])
def test_noon_inactive_fraction_reads_the_planted_area_fraction(planted):
    lat, lon, weights = td.synthetic_geometry()
    samples, land, exact = td.synthetic_samples(lat=lat, lon=lon, weights=weights, inactive_area_fraction=planted)
    result = td.measure(samples, lat, lon, land, weights, regions=("conus_land",))
    noon = result["regions"]["conus_land"]["noon_window"]
    assert noon["fraction_of_grid_scale_rain_in_inactive_columns"] == pytest.approx(exact, abs=1e-12)
    # All-hours fraction equals the noon one when all the rain fell at noon.
    assert result["regions"]["conus_land"]["all_hours"]["fraction_of_grid_scale_rain_in_inactive_columns"] == pytest.approx(exact, abs=1e-12)


def test_grid_first_and_convective_first_orders_are_told_apart():
    lat, lon, weights = td.synthetic_geometry()
    samples, land, _ = td.synthetic_samples(lat=lat, lon=lon, weights=weights, inactive_area_fraction=0.0, grid_lst_h=9.0, convective_lst_h=15.0)
    order = td.measure(samples, lat, lon, land, weights, regions=("conus_land",))["regions"]["conus_land"]["order_of_first_rain"]
    assert order["grid_first_fraction"] == pytest.approx(1.0) and order["convective_first_fraction"] == pytest.approx(0.0)
    assert order["lead_hours"]["median"] == pytest.approx(6.0)
    samples, land, _ = td.synthetic_samples(lat=lat, lon=lon, weights=weights, inactive_area_fraction=0.0, grid_lst_h=15.0, convective_lst_h=9.0)
    order = td.measure(samples, lat, lon, land, weights, regions=("conus_land",))["regions"]["conus_land"]["order_of_first_rain"]
    assert order["convective_first_fraction"] == pytest.approx(1.0)
    assert order["lead_hours"]["median"] == pytest.approx(-6.0)


def test_local_solar_hour_wraps_and_follows_longitude():
    lon = np.asarray([0.0, 90.0, 180.0, 270.0])
    lst = td.local_solar_hour(12.0 * 3600.0, lon)
    assert np.allclose(lst, [12.0, 18.0, 0.0, 6.0])


def test_column_state_reads_saturated_cloudy_levels_and_dry_columns():
    sounding = td.synthetic_sounding(cloudy_levels=(5, 9), rh_elsewhere=0.4)
    state = td.column_state(sounding)
    assert int(state["cloudy_levels"][0, 0]) == 5
    assert int(state["saturated_levels"][0, 0]) == 5
    assert state["rh_cloudy_mean"][0, 0] == pytest.approx(1.0)
    assert state["cloud_top_pa"][0, 0] == sounding.p_full[5, 0, 0]
    dry = td.column_state(td.synthetic_sounding(cloudy_levels=None, rh_elsewhere=0.4))
    assert int(dry["cloudy_levels"][0, 0]) == 0 and math.isnan(dry["rh_cloudy_mean"][0, 0])
    assert dry["rh_max"][0, 0] == pytest.approx(0.4)


def test_relative_humidity_is_one_at_saturation_and_bolton_matches_dewpoint_inversion():
    t = np.asarray([250.0, 273.15, 290.0, 305.0])
    p = np.asarray([50000.0, 85000.0, 95000.0, 100000.0])
    qs = cs.saturation_mixing_ratio(t, p)
    assert np.allclose(cs.relative_humidity(t, p, qs), 1.0)
    assert np.allclose(cs.dewpoint_k(p, qs), t, atol=1e-6)


def test_moist_parcel_is_colder_than_dry_and_a_dry_neutral_column_has_no_cape():
    nlev = 20
    ph = np.exp(np.linspace(np.log(10000.0), np.log(100000.0), nlev + 1))
    pf = np.sqrt(ph[:-1] * ph[1:])
    dry = 300.0 * (pf / pf[-1]) ** cs.KAPPA
    cape = cs.parcel_cape(dry[:, None], np.zeros((nlev, 1)), pf[:, None], ph[:, None])
    # A parcel on its own adiabat: the buoyancy is float roundoff only
    # (measured 8.4e-8 J/kg over twenty layers).
    assert cape["cape_sb_j_kg"][0] == pytest.approx(0.0, abs=1e-6)
    assert cape["cin_sb_j_kg"][0] == pytest.approx(0.0, abs=1e-6)
    # A moist parcel in the same dry-neutral column is buoyant above its LCL.
    qv = np.zeros((nlev, 1))
    qv[-1, 0] = 0.9 * cs.saturation_mixing_ratio(dry[-1], pf[-1])
    moist = cs.parcel_cape(dry[:, None], qv, pf[:, None], ph[:, None])
    assert moist["cape_sb_j_kg"][0] > 1000.0
    assert moist["lcl_sb_pa"][0] < pf[-1]


# --------------------------------------------------------------------------
# storm statistics contracts
# --------------------------------------------------------------------------


def test_storm_statistics_tail_share_and_top_cells():
    lat, lon, weights = sr.synthetic_geometry()
    intervals = sr.synthetic_storm(lat=lat, lon=lon, cell=(12, 30), rates_mm_h=(100.0, 150.0), background_mm_h=1.0)
    intervals2 = sr.synthetic_storm(lat=lat, lon=lon, cell=(20, 5), rates_mm_h=(60.0,), background_mm_h=0.0)
    merged = [sr.IntervalRain(a.start_utc_s, a.end_utc_s, a.grid_kg_m2 + b.grid_kg_m2, a.convective_kg_m2) for a, b in zip(intervals, intervals2)]
    stats = sr.storm_statistics(merged, lat, lon, weights, top_n=2)
    top = stats["top_cells_by_grid_scale"]
    assert [(c["j"], c["i"]) for c in top] == [(12, 30), (20, 5)]
    assert top[0]["grid_scale_mm"] == pytest.approx(274.0)
    assert top[1]["grid_scale_mm"] == pytest.approx(84.0)
    above = stats["cells_above_total"]
    assert above["200"]["grid_scale_count"] == 1 and above["50"]["grid_scale_count"] == 2
    total_area = float(np.sum(weights))
    expect_share = (weights[12, 30] * 274.0 + weights[20, 5] * 84.0) / total_area / stats["global_mean_mm"]["grid_scale"]
    assert above["50"]["grid_scale_share_of_global_mean"] == pytest.approx(expect_share, abs=1e-12)
    assert stats["per_interval"][1]["max_grid_scale_rate_mm_h"] == pytest.approx(151.0)
    assert stats["per_interval"][1]["max_grid_scale_cell"]["j"] == 12


def test_a_seeded_cold_start_checkpoint_reads_zero_convective_rain_and_a_bucketed_one_is_refused():
    """The cold-start seeding writes Noah's snow store into the step-0
    namespace; both readers took that for a physics call and refused the
    whole run (measured 2026-09-05 on the merged tip's combined arm).  A
    step-0 checkpoint whose physics arrays are all seeded surface stores is
    a cumulus-bearing run before its first call (zero convective rain by
    construction); one carrying any other physics bucket without the
    convective accumulator, or any later checkpoint, is refused by name."""
    seeded = {"atmosphere__theta", "surface__land_fraction", "surface__accumulated_rain_kg_m2",
              "physics__noah_snow", "physics__noah_snowc", "physics__noah_snowh"}
    assert cs.convective_accumulator_is_absent_by_construction(seeded, 0)
    assert cs.convective_accumulator_is_absent_by_construction(seeded - {"physics__noah_snow", "physics__noah_snowc", "physics__noah_snowh"}, 0)
    assert not cs.convective_accumulator_is_absent_by_construction(seeded, 72)
    bucketed = seeded | {"physics__rainnc"}
    assert not cs.convective_accumulator_is_absent_by_construction(bucketed, 0)
    message = cs.convective_accumulator_refusal("ck.npz", bucketed, 0)
    assert "physics__rainc" in message and "physics__rainnc" in message and "physics__noah_snow" not in message
    assert "step 72" in cs.convective_accumulator_refusal("ck.npz", seeded, 72)


def test_storm_statistics_refuse_a_misshaped_interval_and_an_empty_run():
    lat, lon, weights = sr.synthetic_geometry()
    with pytest.raises(ValueError, match="at least one"):
        sr.storm_statistics([], lat, lon, weights)
    bad = sr.IntervalRain(0.0, 3600.0, np.zeros((3, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="grid"):
        sr.storm_statistics([bad], lat, lon, weights)


def test_omega_reader_matches_the_continuity_for_a_two_layer_divergence_and_is_zero_at_rest():
    for d in (2.0e-5, 0.0):
        receipt, arrays, expected = sr.synthetic_omega_arrays(divergence_s1=d)
        reader = cs.SoundingReader(receipt, omega=True)
        ps = np.full(reader.shape, 1.0e5)
        pressure = reader.vertical.pressure(ps, reader.transform.backend)
        omega = reader.omega_from_arrays(arrays, ps, pressure)
        assert np.max(np.abs(omega - expected[:, None, None])) <= 1e-9 * max(float(np.max(np.abs(expected))), 1.0)
        if d == 0.0:
            assert np.all(omega == 0.0)
        else:
            assert np.min(expected) < 0.0 < np.max(expected) or np.max(np.abs(expected)) > 0.0


def test_column_readings_report_the_strongest_ascent_and_its_pressure():
    receipt, arrays, expected = sr.synthetic_omega_arrays(divergence_s1=-2.0e-5)
    reader = cs.SoundingReader(receipt, omega=True)
    ny, nx = reader.shape
    nlev = reader.vertical.nlev
    ps = np.full((ny, nx), 1.0e5)
    pressure = reader.vertical.pressure(ps, reader.transform.backend)
    p_full = np.asarray(pressure["p_full"])
    p_half = np.asarray(pressure["p_half"])
    omega = reader.omega_from_arrays(arrays, ps, pressure)
    t = 280.0 * np.ones((nlev, ny, nx))
    qv = 0.5 * cs.saturation_mixing_ratio(t, p_full)
    zeros = np.zeros((nlev, ny, nx))
    sounding = cs.Sounding(
        time_s=0.0, step=0, p_full=p_full, p_half=p_half, temperature_k=t, qv=qv,
        condensate={n: zeros.copy() for n in cs.CONDENSATE_SPECIES}, relative_humidity=cs.relative_humidity(t, p_full, qv),
        omega_half_pa_s=omega, convective_kg_m2=np.zeros((ny, nx)), grid_scale_kg_m2=np.zeros((ny, nx)),
        land=np.ones((ny, nx), dtype=bool), latitude_deg=reader.latitude_deg, longitude_deg=reader.longitude_deg,
    )
    rows = sr.column_readings(sounding, [(3, 4)])
    full = 0.5 * (expected[:-1] + expected[1:])
    k = int(np.argmin(full))
    assert rows[0]["omega_min_pa_s"] == pytest.approx(float(full[k]), rel=1e-9)
    assert rows[0]["omega_min_pressure_pa"] == pytest.approx(float(p_full[k, 3, 4]))
    assert rows[0]["rh_max"] == pytest.approx(0.5)
    assert rows[0]["cloud_top_pa"] is None


# --------------------------------------------------------------------------
# run-reading contracts
# --------------------------------------------------------------------------


def test_receipt_fallback_refuses_a_run_without_receipt_or_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="no receipt yet"):
        cs.read_receipt_or_config(tmp_path)


def test_receipt_fallback_builds_the_reader_geometry_from_the_config(tmp_path):
    config = tmp_path / "run.toml"
    config.write_text(
        '[arwen_global]\nschema = "gpuwm.arwen-global-run/v1"\nname = "fallback"\n'
        'acknowledgement = "research-only-arwen-global-v1"\nbackend = "numpy"\nprecision = "float64"\n'
        '[grid]\ntruncation = 21\n[time]\ndt_s = 600.0\nduration_s = 600.0\n'
        '[vertical]\ncoordinate = "surface_stretched"\nnlev = 40\np_top_pa = 100.0\n'
    )
    receipt = cs.read_receipt_or_config(tmp_path, config)
    assert receipt["config"]["truncation"] == 21
    assert len(receipt["config"]["a_half_pa"]) == 41
    reader = cs.SoundingReader(receipt, omega=False)
    assert reader.shape[0] >= 32 and reader.shape[1] >= 64
    assert reader.vertical.nlev == 40


def test_gf_driver_integer_readings_agree_with_the_kernel_and_the_oracle_field_list():
    """The per-column readings the storm census reads (the deep exit code
    and the downdraft exit the updraft-only switch overrode) are one word
    each in the driver kernel's integer output row: the driver's slot list,
    the oracle field list and the kernel's enum name them in one order."""
    import re
    from pathlib import Path

    from woof.globe.core import gf
    from tools.gf_wrf461_oracle.gf_field_lists import DRV_ISCA_FIELDS

    assert tuple(DRV_ISCA_FIELDS) == gf._OUT_ISCA
    assert gf._OUT_ISCA[6:8] == ("ierr_deep", "downdraft_dry_exit")
    source = (Path(gf.__file__).parent / "kernels" / "gf.cu").read_text(encoding="utf-8")
    enum = source[source.index("DI_ktop_deep"):source.index("GF_DRV_NISCA")]
    names = re.findall(r"\bDI_([a-z0-9_]+)", enum)
    assert names == list(gf._OUT_ISCA)
    assert "ISCB[DI_downdraft_dry_exit] = dd_d;" in source
    # The override covers both downdraft exits and remembers which one.
    assert "(ierr == 7 || ierr == 51)" in source and "downdraft_dry = ierr;" in source


def test_run_start_reads_the_native_options_start_time():
    receipt = {"config": {"native_adapter_options": {"start_time_utc": "2026-09-01T00:00:00Z"}}}
    from woof.globe.diurnal_phase import parse_utc

    assert cs.run_start_utc_s(receipt) == parse_utc("2026-09-01T00:00:00Z")
    assert cs.run_start_utc_s({"config": {}}) is None
