"""The reference leg's arithmetic and contracts, without CRTM, without a
tape and without the card.

Proven here: the reference output stream round-trips and refuses a
truncated one by its numbers; the paired statistics read a planted bias
and a planted slope both ways and the linear correction removes exactly
what it was given; the Jacobian summary places a planted weighting
function's peak and its half-sensitivity band; the block tables of two
bands join on the block index and refuse two clear-sky censuses that
disagree; the climatology codes follow latitude and season; the ozone
fill is the US standard between its ends; the superobservation batch of
the radiance stream carries the block means, thins the thin blocks,
places every row inside the scan window from north to south, and the
forward door names its remedy when the front door is missing.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
import struct

import numpy as np
import pytest

from woof.globe import abi_reference as ref
from woof.globe import abi_radiance_operator as rad
from woof.globe.abi_operator import AbiOperatorError


def _abir(path: Path, n=4, nlay=3, channels=(8, 13), truncate=0):
    rng = np.random.default_rng(0)
    nchan = len(channels)
    payload = bytearray(struct.pack("<7i", ref.OUTPUT_MAGIC, 1, n, nlay, nchan, 0, 0))
    payload += np.asarray(channels, dtype="<i4").tobytes()
    arrays = {}
    for name in ("bt", "emissivity", "radiance", "jac_tskin", "surface_planck"):
        arr = rng.uniform(200, 300, (nchan, n))
        arrays[name] = arr
        payload += arr.astype("<f8").tobytes()
    for name in ("jac_t", "jac_q", "layer_od"):
        arr = rng.uniform(0, 1, (nchan, n, nlay))
        arrays[name] = arr
        payload += arr.astype("<f8").tobytes()
    if truncate:
        payload = payload[:-truncate]
    path.write_bytes(bytes(payload))
    return arrays


def test_reference_stream_round_trips_and_refuses_a_short_one(tmp_path):
    arrays = _abir(tmp_path / "ok.bin")
    out = ref.read_crtm_output(tmp_path / "ok.bin")
    assert out["channels"] == [8, 13] and out["n"] == 4 and out["layers"] == 3
    assert np.array_equal(out["bt"][13], arrays["bt"][1])
    assert np.array_equal(out["layer_od"][8], arrays["layer_od"][0])
    _abir(tmp_path / "short.bin", truncate=8)
    with pytest.raises(ref.AbiReferenceError, match="holds .* bytes where its header"):
        ref.read_crtm_output(tmp_path / "short.bin")


def test_paired_stats_read_a_planted_bias_and_slope_both_ways():
    rng = np.random.default_rng(1)
    obs = rng.uniform(250, 310, 5000)
    sim = 1.1 * obs - 20.0 + rng.normal(0, 0.05, obs.size)          # obs = (sim + 20) / 1.1
    row = ref.paired_stats(obs, sim)
    assert row["n"] == 5000
    assert abs(row["bias_k"] - float(np.mean(sim - obs))) < 1e-9
    assert abs(row["fit_slope"] - 1 / 1.1) < 1e-3 and abs(row["fit_intercept_k"] - 20 / 1.1) < 0.3
    assert row["rmse_after_k"] < 0.06 and row["rmse_k"] > 2.0
    identical = ref.paired_stats(obs, obs)
    assert identical["bias_k"] == 0.0 and identical["rmse_k"] == 0.0
    assert ref.paired_stats(np.array([]), np.array([]))["n"] == 0
    weighted = ref.paired_stats(np.array([1.0, 3.0]), np.array([2.0, 2.0]), w=np.array([3.0, 1.0]))
    assert abs(weighted["bias_k"] - (3 * 1.0 + 1 * (-1.0)) / 4) < 1e-12


def test_jacobian_summary_places_a_planted_peak():
    n, nlay = 5, 30
    p_half = np.tile(np.geomspace(1.0, 1000.0, nlay + 1), (n, 1))
    p_full = np.sqrt(p_half[:, :-1] * p_half[:, 1:])
    dlnp = np.log(p_half[:, 1:] / p_half[:, :-1])
    jac_t = np.exp(-0.5 * ((np.log(p_full) - np.log(300.0)) / 0.4) ** 2) * dlnp
    jac_q = -jac_t
    summary = ref.jacobian_summary(jac_t, jac_q, np.full(n, 0.01), p_half, p_full)
    peak = summary["peak_pressure_hpa_percentiles_5_25_50_75_95"][2]
    assert 250.0 < peak < 360.0, peak
    lo, hi = summary["half_sensitivity_band_hpa_median"]
    assert lo < 300.0 < hi
    assert summary["sensitivity_below_500hpa_fraction_mean"] < 0.15
    assert abs(summary["skin_jacobian_mean"] - 0.01) < 1e-12


def _blocks_csv(path: Path, rows):
    header = ("block_x,block_y,lat_mean_deg,lon_mean_deg,zenith_mean_deg,n_sim,n_obs,n_pair,obs_mean_k,sim_mean_k,bias_k,"
              "rmse_k,obs_clear,obs_cloudy,sim_clear,sim_cloudy,n_both_clear,obs_mean_both_clear_k,sim_mean_both_clear_k")
    lines = [header] + [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def test_block_tables_join_on_the_index_and_refuse_two_censuses(tmp_path):
    rows13 = [(1, 1, 10.0, -80.0, 30.0, 500, 500, 500, 290.0, 292.0, 2.0, 2.1, 400, 100, 500, 0, 400, 291.0, 293.0),
              (2, 1, 10.5, -79.5, 31.0, 500, 500, 500, 280.0, 281.0, 1.0, 1.1, 0, 500, 500, 0, 0, 0.0, 0.0),
              (3, 1, 11.0, -79.0, 75.0, 500, 500, 500, 285.0, 286.0, 1.0, 1.1, 500, 0, 500, 0, 500, 285.0, 286.0)]
    rows08 = [(1, 1, 10.0, -80.0, 30.0, 500, 500, 500, 240.0, 247.0, 7.0, 7.1, 400, 100, 500, 0, 400, 241.0, 248.0),
              (2, 1, 10.5, -79.5, 31.0, 500, 500, 500, 235.0, 236.0, 1.0, 1.1, 0, 500, 500, 0, 0, 0.0, 0.0),
              (3, 1, 11.0, -79.0, 75.0, 500, 500, 500, 238.0, 239.0, 1.0, 1.1, 500, 0, 500, 0, 500, 238.0, 239.0)]
    _blocks_csv(tmp_path / "b13.csv", rows13)
    _blocks_csv(tmp_path / "b08.csv", rows08)
    blocks = ref.read_block_tables({13: tmp_path / "b13.csv", 8: tmp_path / "b08.csv"}, zenith_max_deg=70.0)
    assert blocks["bands"] == [8, 13]
    assert blocks["lat"].tolist() == [10.0]                       # block 2 has no clear pixel, block 3 is beyond 70
    assert blocks["obs_13"].tolist() == [291.0] and blocks["obs_8"].tolist() == [241.0]
    rows08[0] = rows08[0][:16] + (399,) + rows08[0][17:]
    _blocks_csv(tmp_path / "b08bad.csv", rows08)
    with pytest.raises(ref.AbiReferenceError, match="disagree on the both-clear count"):
        ref.read_block_tables({13: tmp_path / "b13.csv", 8: tmp_path / "b08bad.csv"})


def test_climatology_and_ozone_fill():
    codes = ref.climatology_for(np.array([-70.0, -45.0, -10.0, 10.0, 45.0, 70.0]), 9)
    assert codes.tolist() == [ref.SUBARCTIC_WINTER, ref.MIDLATITUDE_WINTER, ref.TROPICAL, ref.TROPICAL,
                              ref.MIDLATITUDE_SUMMER, ref.SUBARCTIC_SUMMER]
    codes = ref.climatology_for(np.array([45.0, -45.0]), 1)
    assert codes.tolist() == [ref.MIDLATITUDE_WINTER, ref.MIDLATITUDE_SUMMER]
    o3 = ref.ozone_ppmv(np.array([0.5, 10.0, 300.0, 1000.0, 1200.0]))
    assert o3[0] == pytest.approx(3.035) and o3[-1] == pytest.approx(0.01428)
    assert o3[1] > o3[2] > o3[3]


def test_superobs_batch_carries_the_blocks_and_their_scan_times(tmp_path):
    pytest.importorskip("woof.globe.da.observations",
                        reason="the PointObs contract lives on the ensemble lane; the batch is proven where it is merged")
    # block_y is the block INDEX from the south ((y_index + half) div 24 in
    # rw_goes), 0 to 225 on the 5,424-pixel axis, not a pixel row.
    rows = [(1, 108, 10.0, -80.0, 30.0, 500, 500, 500, 290.0, 292.0, 2.0, 2.1, 400, 100, 500, 0, 400, 291.0, 293.0),
            (2, 4, -60.0, -79.5, 65.0, 500, 500, 500, 280.0, 281.0, 1.0, 1.1, 5, 495, 500, 0, 5, 279.0, 280.0),
            (3, 208, 60.0, -79.0, 55.0, 500, 500, 500, 285.0, 286.0, 1.0, 1.1, 500, 0, 500, 0, 500, 285.0, 286.0)]
    _blocks_csv(tmp_path / "b13.csv", rows)
    start = dt.datetime(2026, 9, 1, 18, 0, 20, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 9, 1, 18, 9, 52, tzinfo=dt.timezone.utc)
    batch, extras = rad.superobs_from_blocks(tmp_path / "b13.csv", band=13, observation_error_k=1.2, scan_start=start,
                                             scan_end=end, zenith_max_deg=60.0, minimum_pixels=12, ln_pressure_hpa=947.0)
    assert batch.stream == rad.STREAM and batch.variable == rad.VARIABLE
    assert batch.count == 2 and batch.value.tolist() == [291.0, 285.0]
    assert batch.rejections == {"thinned_below_minimum_pixels": 1, "below_minimum_clear_fraction": 0, "beyond_zenith": 0}
    assert np.allclose(batch.error, 1.2) and np.allclose(np.exp(batch.ln_pressure), 94700.0)
    assert extras["zenith_deg"].tolist() == [30.0, 55.0]
    times = batch.valid_time
    assert start <= times[0] <= end and start <= times[1] <= end
    assert times[1] < times[0]                                    # the northern block (block_y 208) is scanned first
    assert batch.identity[0] == "abi13:1:108"


def test_forward_door_names_its_remedy_without_the_front_door(tmp_path, monkeypatch):
    monkeypatch.delenv("WOOF_RW_GOES", raising=False)
    monkeypatch.setattr("woof.globe.abi_operator.shutil.which", lambda name: None)
    monkeypatch.setattr("woof.globe.abi_operator.Path.is_file", lambda self: False)
    with pytest.raises(AbiOperatorError, match="cargo build"):
        rad.run_forward(tmp_path / "c.bin", tmp_path / "t.json", tmp_path / "o.bin")


def _provenance():
    return {"dataset_name": "OR_ABI-L1b-RadF-M6C13_G19_s20262441800203_e20262441809523_c20262441809550.nc",
            "id": "b3cc4740-5f21-464f-904c-c3d6a898f8a6", "date_created": "2026-09-01T18:09:55.0Z",
            "time_coverage_start": "2026-09-01T18:00:20.3Z", "time_coverage_end": "2026-09-01T18:09:52.3Z",
            "production_site": "GCCS", "production_environment": "OE", "production_data_source": "Realtime",
            "platform_id": "G19", "processing_level": "L1b", "received_utc": "2026-09-06T00:29:18Z",
            "row_time_model": "linear from the north inside the scan window"}


def test_the_pack_carries_the_bookkeeping_row_and_the_batch_reads_the_scan_from_it(tmp_path):
    from woof.globe.obs_pack import read_goes_pack
    from goes_pack_fixtures import write_bt_pack

    bt = np.full((4, 4), 290.0, dtype="<f4")
    pack = write_bt_pack(tmp_path / "b13.goespack", bt=bt, rad=bt, lat=bt, lon=bt, provenance=_provenance())
    meta = read_goes_pack(pack).meta
    assert meta["provenance"]["date_created"] == "2026-09-01T18:09:55.0Z"
    assert meta["provenance"]["received_utc"] == "2026-09-06T00:29:18Z"
    assert meta["provenance"]["id"].count("-") == 4
    pytest.importorskip("woof.globe.da.observations",
                        reason="the PointObs contract lives on the ensemble lane; the batch is proven where it is merged")
    rows = [(1, 2600, 10.0, -80.0, 30.0, 500, 500, 500, 290.0, 292.0, 2.0, 2.1, 400, 100, 500, 0, 400, 291.0, 293.0)]
    _blocks_csv(tmp_path / "b13.csv", rows)
    batch, extras = rad.superobs_from_pack(pack, tmp_path / "b13.csv", observation_error_k=0.61)
    assert batch.count == 1 and extras["band"] == 13
    assert extras["latency_class"] == "first receipt recorded"
    assert extras["provenance"]["production_site"] == "GCCS"
    assert dt.datetime(2026, 9, 1, 18, 0, 20, tzinfo=dt.timezone.utc) <= batch.valid_time[0] <= dt.datetime(2026, 9, 1, 18, 9, 53, tzinfo=dt.timezone.utc)
