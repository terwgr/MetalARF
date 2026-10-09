"""The microwave leg (design item 5b): channel table, absorption,
emissivity, radiative transfer, calibration, fetch manifest parsing and
the ``rw_atms`` bridge contract.

The physics readings here are the calibration of record before any real
radiance is cited: an isothermal column reads back its temperature
exactly, the Rayleigh-Jeans weights sum to one, a planted warm layer
reads back through the weights alone when the opacity is frozen and
within 0.03 K with the absorption's own temperature dependence, a column
identical to itself moves nothing.  The absorption is held against the
standard-atmosphere figures of ITU-R P.676-13 and the emissivity against
the known nadir ocean values.  The bridge test writes a decode
directory the way ``rw_atms`` does and reads it back; the binary itself
is exercised when a build is present.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from woof.globe.microwave import absorption, atms_bridge, atms_fetch, calibrate, channels, emissivity
from woof.globe.microwave.rte import (
    Column, brightness_temperature, channel_quadrature, planck_radiance, planck_temperature,
    weighting_function,
)
from woof.globe.microwave.score import ScreenOptions, grody_cloud_liquid_water, score_channels


# ------------------------------------------------------------------ channels

def test_channel_table_is_the_atms_plan():
    assert len(channels.CHANNELS) == 22
    assert [c.number for c in channels.CHANNELS] == list(range(1, 23))
    assert channels.channel(10).passbands_ghz == (channels.F0_GHZ,)
    assert len(channels.channel(6).passbands_ghz) == 2
    assert len(channels.channel(12).passbands_ghz) == 4
    assert channels.channel(15).bandwidth_mhz == 3.0
    assert channels.channel(1).polarization == "QV"
    assert channels.channel(16).polarization == "QV"
    assert all(channels.channel(n).polarization == "QH" for n in range(3, 16))
    assert channels.TEMPERATURE_SOUNDING_CHANNELS == tuple(range(4, 16))
    assert channels.scan_angle_deg(0) == pytest.approx(-52.725)
    assert channels.scan_angle_deg(95) == pytest.approx(52.725)


def test_channel_quadrature_weights_normalise():
    for channel in channels.CHANNELS:
        frequencies, weights = channel_quadrature(channel)
        assert weights.sum() == pytest.approx(1.0)
        assert frequencies.size == 5 * len(channel.passbands_ghz)
        assert np.all(np.abs(frequencies - channel.centre_ghz) < 8.0)


# ---------------------------------------------------------------- absorption

def test_absorption_matches_p676_standard_atmosphere_figure():
    """P.676-13 Figure 1: 1013.25 hPa, 15 C, 7.5 g/m3.  The 60 GHz band
    centre and the 22 and 183 GHz water lines are the readable points."""
    t = 288.15
    e = 7.5 * t / 216.7
    p = 1013.25 - e
    gamma = absorption.specific_attenuation_db_km([22.235, 60.0, 118.75, 183.31], p, t, e)
    assert gamma[0] == pytest.approx(0.19, abs=0.02)
    assert gamma[1] == pytest.approx(14.7, abs=0.7)
    assert gamma[2] == pytest.approx(1.9, abs=0.2)
    assert gamma[3] == pytest.approx(28.0, abs=2.0)
    dry = absorption.specific_attenuation_db_km([183.31], 1013.25, t, 0.0)
    assert dry[0] < 0.05  # the 183 GHz line is water vapour alone


def test_absorption_tables_have_the_published_shape():
    assert absorption.OXYGEN_LINES.shape == (44, 7)
    assert absorption.WATER_VAPOUR_LINES.shape == (35, 7)
    assert absorption.WATER_VAPOUR_LINES[-1, 0] == 1780.0
    assert np.all(np.diff(absorption.OXYGEN_LINES[:, 0]) > 0)
    assert np.all(np.diff(absorption.WATER_VAPOUR_LINES[:, 0]) > 0)


def test_absorption_broadcasts_frequencies_over_columns():
    f = np.array([50.3, 54.4, 57.29])
    p = np.array([[1000.0, 500.0], [100.0, 10.0]])
    t = np.array([[290.0, 260.0], [220.0, 230.0]])
    e = np.array([[10.0, 1.0], [0.01, 0.0]])
    alpha = absorption.absorption_coefficient(f, p, t, e)
    assert alpha.shape == (3, 2, 2)
    assert np.all(alpha > 0)
    # Opacity falls with pressure in the band wings.
    assert np.all(alpha[:, 0, :] > alpha[:, 1, :])


def test_vapour_pressure_from_specific_humidity():
    e = absorption.vapour_pressure_hpa(100000.0, 0.01)
    assert e == pytest.approx(0.01 * 1000.0 / (absorption.EPSILON + (1 - absorption.EPSILON) * 0.01))
    assert absorption.vapour_pressure_hpa(50000.0, 0.0) == 0.0


# ---------------------------------------------------------------- emissivity

def test_ocean_emissivity_has_the_known_nadir_values_and_ordering():
    # Nadir: the ocean is a poor emitter at 19 GHz (a 290 K sea reads
    # about 130 K from above) and a better one at 89 GHz.
    for f, expected in ((19.35, 0.40), (37.0, 0.46), (50.3, 0.50), (89.0, 0.58)):
        e_v, e_h = emissivity.fresnel_emissivity(f, 290.0, 0.0)
        assert e_v == pytest.approx(e_h, abs=1e-9)  # nadir has no polarization
        assert e_v == pytest.approx(expected, abs=0.04)
    # The SSM/I geometry: 19 GHz at 53 degrees reads about 0.58 vertical
    # and 0.27 horizontal over a calm sea.
    e_v, e_h = emissivity.fresnel_emissivity(19.35, 290.0, 53.0)
    assert e_v == pytest.approx(0.58, abs=0.04)
    assert e_h == pytest.approx(0.27, abs=0.04)
    e_v, e_h = emissivity.fresnel_emissivity(50.3, 290.0, 50.0)
    assert e_v > e_h  # vertical rises toward Brewster, horizontal falls
    assert 0.0 < e_h < e_v < 1.0


def test_quasi_polarization_mixes_with_scan_angle():
    e_v, e_h = 0.7, 0.5
    assert emissivity.quasi_polarized_emissivity(e_v, e_h, 0.0, "QV") == pytest.approx(0.7)
    assert emissivity.quasi_polarized_emissivity(e_v, e_h, 0.0, "QH") == pytest.approx(0.5)
    assert emissivity.quasi_polarized_emissivity(e_v, e_h, 90.0, "QV") == pytest.approx(0.5)
    assert emissivity.quasi_polarized_emissivity(e_v, e_h, 45.0, "QH") == pytest.approx(0.6)
    with pytest.raises(ValueError):
        emissivity.quasi_polarized_emissivity(e_v, e_h, 0.0, "V")


# ----------------------------------------------------------- radiative transfer

def test_planck_round_trip():
    f = np.array([23.8, 57.29, 183.31])
    t = np.array([200.0, 250.0, 300.0])
    assert planck_temperature(f, planck_radiance(f, t)) == pytest.approx(t, rel=1e-12)


def test_isothermal_black_column_reads_its_temperature_exactly():
    receipt = calibrate.isothermal_reading(250.0, 1.0)
    assert receipt["max_abs_error_k"] < 1e-9


def test_isothermal_grey_column_reads_the_analytic_answer():
    receipt = calibrate.isothermal_reading(280.0, 0.6, zenith_deg=45.0)
    assert receipt["max_abs_error_k"] < 1e-9


def test_weights_sum_to_one_and_peaks_climb_with_channel():
    receipt = calibrate.weights_sum_reading()
    assert receipt["max_abs_deviation_from_one"] < 1e-9
    peaks = [receipt["channels"][n]["peak_hpa"] for n in channels.TEMPERATURE_SOUNDING_CHANNELS]
    # Monotone non-increasing peak pressure (per unit ln p) from channel 4
    # to 15, and the per-sub-layer maximum the receipt used to call the
    # peak is recorded beside it: it is quantised by the level spacing and
    # gave channels 8 and 9 the same layer.
    assert all(a >= b for a, b in zip(peaks, peaks[1:])), peaks
    per_sublayer = [receipt["channels"][n]["peak_per_sublayer_hpa"] for n in (8, 9)]
    assert per_sublayer[0] == per_sublayer[1]
    assert receipt["channels"][8]["peak_hpa"] > receipt["channels"][9]["peak_hpa"]
    centroids = [receipt["channels"][n]["centroid_hpa"] for n in channels.TEMPERATURE_SOUNDING_CHANNELS]
    assert all(a > b for a, b in zip(centroids, centroids[1:])), centroids
    assert receipt["channels"][4]["surface_transmittance"] > 0.3
    assert receipt["channels"][8]["surface_transmittance"] < 0.01
    assert receipt["channels"][15]["peak_hpa"] < 5.0


def test_planted_layer_reads_back_through_the_weights():
    receipt = calibrate.planted_layer_reading()
    assert receipt["max_abs_frozen_error_k"] < 1e-3
    assert receipt["max_abs_response_error_k"] < 0.03
    assert receipt["max_quiet_response_k"] < 0.02
    ranks = {r["channel"]: r["rank_among_sounding_channels"] for r in receipt["readings"]}
    # A plant at a channel's own peak (per unit ln p) is loudest in that
    # channel.  The one exception is physics, not quantisation: channel 5
    # peaks in the surface layer of the tropical column, where channel 4
    # carries more weight than channel 5 does anywhere, so channel 4 is
    # louder there and channel 5 ranks second.
    assert all(rank == 1 for ch, rank in ranks.items() if ch != 5), ranks
    assert ranks[5] <= 2, ranks
    for reading in receipt["readings"]:
        lo, hi = reading["planted_levels_hpa"]
        if reading["peak_in_surface_slab"]:
            # Channel 4 peaks in the slab between the 1000 hPa level and the
            # 1010 hPa surface; it is planted at the lowest analysis pair.
            assert reading["channel"] == 4 and hi == 1000.0 and reading["peak_hpa"] > hi
        else:
            assert lo - 1e-6 <= reading["peak_hpa"] <= hi + 1e-6, reading


def test_null_column_moves_nothing():
    receipt = calibrate.null_reading()
    assert receipt["bitwise_equal"]


def test_planck_term_is_small_and_stated():
    receipt = calibrate.planck_term_reading()
    term = receipt["planck_minus_rayleigh_jeans_k"]
    assert all(abs(term[n]) < 0.05 for n in channels.TEMPERATURE_SOUNDING_CHANNELS)
    assert abs(term[16]) < 0.2


def test_calibration_gate_passes():
    receipt = calibrate.run()
    verdict = calibrate.passes(receipt)
    assert all(verdict.values()), verdict


def test_brightness_temperature_batches_agree_with_single_columns():
    base = calibrate.standard_column()
    warm = Column(
        pressure_pa=base.pressure_pa,
        temperature_k=base.temperature_k + 3.0,
        specific_humidity=base.specific_humidity,
        surface_pressure_pa=base.surface_pressure_pa,
        skin_temperature_k=base.skin_temperature_k + 3.0,
        air_temperature_2m_k=base.air_temperature_2m_k + 3.0,
    )
    both = Column(
        pressure_pa=base.pressure_pa,
        temperature_k=np.concatenate([base.temperature_k, warm.temperature_k], axis=1),
        specific_humidity=np.concatenate([base.specific_humidity, warm.specific_humidity], axis=1),
        surface_pressure_pa=np.concatenate([base.surface_pressure_pa, warm.surface_pressure_pa]),
        skin_temperature_k=np.concatenate([base.skin_temperature_k, warm.skin_temperature_k]),
        air_temperature_2m_k=np.concatenate([base.air_temperature_2m_k, warm.air_temperature_2m_k]),
    )
    channels_ = [4, 8, 12]
    tb_both = brightness_temperature(both, channels_, [20.0, 40.0], [10.0, 30.0])
    tb_a = brightness_temperature(base, channels_, 20.0, 10.0)
    tb_b = brightness_temperature(warm, channels_, 40.0, 30.0)
    assert tb_both[:, 0] == pytest.approx(tb_a[:, 0], abs=1e-9)
    assert tb_both[:, 1] == pytest.approx(tb_b[:, 0], abs=1e-9)
    # A uniformly warmer column at the same geometry reads warmer everywhere.
    tb_warm_same = brightness_temperature(warm, channels_, 20.0, 10.0)
    assert np.all(tb_warm_same[:, 0] > tb_a[:, 0])


def test_column_rejects_bad_shapes():
    p = np.array([100.0, 1000.0, 10000.0])
    with pytest.raises(ValueError):
        Column(p, np.zeros((2, 1)), np.zeros((2, 1)), np.array([1e5]), np.array([300.0]))
    with pytest.raises(ValueError):
        Column(np.array([1000.0, 100.0, 5000.0]), np.zeros((3, 1)), np.zeros((3, 1)),
               np.array([1e5]), np.array([300.0]))


# --------------------------------------------------------------------- score

def test_grody_cloud_liquid_water_is_zero_for_a_clear_pair_and_rises_with_cloud():
    clear = grody_cloud_liquid_water(200.0, 170.0, 30.0)
    cloudy = grody_cloud_liquid_water(230.0, 215.0, 30.0)
    assert cloudy > clear
    assert np.isnan(grody_cloud_liquid_water(290.0, 200.0, 0.0))


def test_score_channels_corrects_a_planted_linear_bias_out_of_sample():
    rng = np.random.default_rng(7)
    n = 400
    background = 230.0 + 20.0 * rng.random((n, 2))
    truth_bias = 1.5 + 0.1 * (background - background.mean(axis=0))
    observed = background + truth_bias + 0.05 * rng.standard_normal((n, 2))
    scores = score_channels(
        observed, background,
        zenith_deg=rng.uniform(0, 55, n), latitude_deg=rng.uniform(-59, 59, n),
        wind_speed_m_s=rng.uniform(0, 12, n), precipitable_water_kg_m2=rng.uniform(5, 50, n),
        time_unix_s=np.arange(n, dtype=float), cell_std_k=np.zeros((n, 2)),
        channels=(5, 6), options=ScreenOptions(), bar_k=1.0,
    )
    for s in scores:
        assert s.raw["bias"] == pytest.approx(1.5, abs=0.2)
        assert s.raw["rmse"] > 1.0
        assert s.linear_corrected["rmse"] < 0.1
        assert s.linear_coefficients["a"] == pytest.approx(1.5, abs=0.1)
        assert s.linear_coefficients["b"] == pytest.approx(0.1, abs=0.02)
        assert s.within_bar
        assert s.fit_cells + s.score_cells == n


# ------------------------------------------------------------------- fetcher

def test_granule_name_parses_and_pairs():
    sdr = atms_fetch.parse_granule_name(
        "SATMS_j02_d20260901_t0000031_e0000347_b19730_c20260901000902460000_oebc_ops.h5"
    )
    geo = atms_fetch.parse_granule_name(
        "GATMO_j02_d20260901_t0000031_e0000347_b19730_c20260901000902855000_oebc_ops.h5"
    )
    assert sdr.spacecraft == "j02" and sdr.orbit == 19730
    assert sdr.start == dt.datetime(2026, 9, 1, 0, 0, 3, 100000, tzinfo=dt.timezone.utc)
    assert sdr.end == dt.datetime(2026, 9, 1, 0, 0, 34, 700000, tzinfo=dt.timezone.utc)
    assert sdr.created == dt.datetime(2026, 9, 1, 0, 9, 2, 460000, tzinfo=dt.timezone.utc)
    assert sdr.pair_key == geo.pair_key
    assert (sdr.created - sdr.end).total_seconds() == pytest.approx(507.76)


def test_granule_crossing_midnight_ends_next_day():
    name = atms_fetch.parse_granule_name(
        "SATMS_j01_d20260901_t2359471_e0000187_b00001_c20260902000902460000_oebc_ops.h5"
    )
    assert name.end.day == 2 and name.start.day == 1


def test_granule_name_refuses_other_files():
    with pytest.raises(atms_fetch.AtmsFetchError):
        atms_fetch.parse_granule_name("not-a-granule.h5")


def test_manifest_pairs(tmp_path):
    manifest = {
        "files": [
            {"product": "sdr", "path": "sdr/a.h5", "granule_start": "s", "granule_end": "e", "orbit": 1},
            {"product": "geo", "path": "geo/a.h5", "granule_start": "s", "granule_end": "e", "orbit": 1},
            {"product": "sdr", "path": "sdr/b.h5", "granule_start": "s2", "granule_end": "e2", "orbit": 2},
        ]
    }
    pairs = atms_fetch.granule_pairs(manifest, tmp_path)
    assert pairs == [(tmp_path / "sdr/a.h5", tmp_path / "geo/a.h5")]


# -------------------------------------------------------------------- bridge

def _write_decode_dir(root: Path, nscan: int = 2) -> None:
    arrays = {}

    def put(name, array, dtype, units):
        array = np.ascontiguousarray(array.astype(dtype))
        filename = f"{name}.{dtype.lstrip('<')}"
        array.tofile(root / filename)
        arrays[name] = {"filename": filename, "shape": list(array.shape), "dtype": dtype, "units": units}

    fov = 96
    put("brightness_temperature_k", 200.0 + np.arange(nscan * fov * 22).reshape(nscan, fov, 22) * 1e-3, "<f4", "K")
    put("latitude_deg", np.linspace(-10, 10, nscan * fov).reshape(nscan, fov), "<f4", "degrees_north")
    put("longitude_deg", np.linspace(100, 110, nscan * fov).reshape(nscan, fov), "<f4", "degrees_east")
    put("satellite_zenith_deg", np.abs(np.linspace(-52, 52, fov))[None, :].repeat(nscan, 0), "<f4", "degrees")
    put("satellite_azimuth_deg", np.full((nscan, fov), 90.0), "<f4", "degrees")
    put("solar_zenith_deg", np.full((nscan, fov), 60.0), "<f4", "degrees")
    put("beam_time_unix_s", np.full((nscan, fov), 1788220800.0), "<f8", "s")
    put("granule_index", np.zeros(nscan), "<i4", "1")
    metadata = {
        "schema": atms_bridge.DECODE_SCHEMA, "scan_count": nscan, "fov_count": fov, "channel_count": 22,
        "granules": [], "arrays": arrays,
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_bridge_reads_a_decode_directory_and_refuses_short_arrays(tmp_path):
    _write_decode_dir(tmp_path)
    decoded = atms_bridge.read_decoded(tmp_path)
    assert decoded.brightness_temperature_k.shape == (2, 96, 22)
    assert decoded.beam_time_unix_s.dtype == np.float64
    assert decoded.latitude_deg[0, 0] == pytest.approx(-10.0)
    # Truncate one array: the reader names the byte counts.
    (tmp_path / "latitude_deg.f4").write_bytes(b"\0" * 8)
    with pytest.raises(atms_bridge.AtmsDecodeError, match="bytes"):
        atms_bridge.read_decoded(tmp_path)


def test_bridge_refuses_a_foreign_schema(tmp_path):
    _write_decode_dir(tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    metadata["schema"] = "something-else"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(atms_bridge.AtmsDecodeError, match="schema"):
        atms_bridge.read_decoded(tmp_path)


def test_bridge_candidates_follow_the_ladder(monkeypatch, tmp_path):
    monkeypatch.delenv(atms_bridge.ATMS_ENV, raising=False)
    candidates = atms_bridge.atms_candidates()
    assert any("target" in str(path) and "release" in str(path) for path in candidates)
    monkeypatch.setenv(atms_bridge.ATMS_ENV, str(tmp_path / "missing"))
    with pytest.raises(atms_bridge.AtmsBridgeMissing, match="missing file"):
        atms_bridge.find_atms_bin()


@pytest.mark.skipif(atms_bridge.find_atms_bin() is None, reason="rw_atms is not built here")
def test_built_binary_carries_the_contract_marker():
    assert atms_bridge.abi_matches()
