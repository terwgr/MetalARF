"""The operator entries and the radiance stream's block quality control,
without CRTM, without a granule and without the card.

Proven here: the entries compare the operator's Jacobians with the NAMED
reference run and refuse to be built without one (an entry built against
the first other run compared the operator with itself under another
emissivity); a class is admitted on the population the stream hands the
filter (the stream QC, one block one row, unweighted), not on the
pixel-weighted statistics over every both-clear block, and the two are
carried side by side; the filter-facing statistics reject a cloud-edge
block (few clear pixels among many paired ones) and read a planted bias
and slope; the residual shape reads a planted heavy tail; the stream
batch drops cloud-edge blocks, keeps only the surface classes the entry
admits with the entry's error, carries the entry's correction, refuses a
class-admitting entry without a land fraction and an entry graded under
another QC; and the operator applies the registered correction so the
filter's O-B is in the frame the entry was graded in.
"""
from __future__ import annotations

import datetime as dt
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from woof.globe import abi_operator as op
from woof.globe import abi_radiance_operator as rad
from woof.globe import abi_reference as ref
from woof.globe.abi_operator import AbiOperatorError


# ---------------------------------------------------------------------------
# a stand-in for the ensemble lane's PointObs where that package is not merged
# ---------------------------------------------------------------------------

@dataclass
class _PointObs:
    stream: str
    variable: str
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    ln_pressure: np.ndarray
    surface: np.ndarray
    value: np.ndarray
    error: np.ndarray
    simulated: np.ndarray | None = None
    control_simulated: np.ndarray | None = None
    elevation_m: np.ndarray | None = None
    identity: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    valid_time: list | None = None
    horizontal_cutoff_km: float | None = None
    vertical_cutoff_lnp: float | None = None
    operator: object = None
    rejections: dict = field(default_factory=dict)

    def __post_init__(self):
        n = int(np.asarray(self.value).size)
        if not np.all(np.isfinite(self.error)) or not np.all(np.asarray(self.error) > 0):
            raise ValueError("error must be finite and positive")
        if np.asarray(self.error).size != n:
            raise ValueError("one error per row")

    @property
    def count(self):
        return int(np.asarray(self.value).size)


@pytest.fixture
def point_obs(monkeypatch):
    """The real contract where it is merged, the stand-in elsewhere."""
    try:
        from woof.globe.da.observations import PointObs  # noqa: F401
        return PointObs
    except ImportError:
        pkg = types.ModuleType("woof.globe.da")
        pkg.__path__ = []
        mod = types.ModuleType("woof.globe.da.observations")
        mod.PointObs = _PointObs
        monkeypatch.setitem(sys.modules, "woof.globe.da", pkg)
        monkeypatch.setitem(sys.modules, "woof.globe.da.observations", mod)
        return _PointObs


# ---------------------------------------------------------------------------
# the filter-facing statistics and the residual shape
# ---------------------------------------------------------------------------

def test_filter_facing_stats_reject_cloud_edges_and_read_a_planted_correction():
    rng = np.random.default_rng(4)
    n = 3000
    sim = rng.uniform(270, 300, n)
    obs = 0.5 + 1.02 * sim + rng.normal(0, 0.3, n)          # obs = 0.5 + 1.02 sim
    n_pair = np.full(n, 576)
    n_both = np.full(n, 500)
    n_both[:300] = 5                                          # cloud edges: 5 of 576 clear
    obs[:300] -= 10.0                                          # and they read the cloud the mask missed
    n_both[300:400] = 30                                       # 30 clear pixels of 576: enough pixels, not enough fraction
    obs[300:400] -= 6.0
    zen = rng.uniform(0, 70, n)
    row = ref.filter_facing_stats(obs, sim, n_both_clear=n_both, n_pair=n_pair, zenith=zen)
    assert row["blocks_rejected"]["below_minimum_pixels"] == 300
    assert row["blocks_rejected"]["below_minimum_clear_fraction"] == 100
    assert row["blocks_rejected"]["beyond_zenith"] == int(np.sum(zen[400:] > 60.0))
    assert row["n"] == int(np.sum(zen[400:] <= 60.0))
    assert abs(row["fit_slope"] - 1.02) < 0.01 and abs(row["fit_intercept_k"] - 0.5) < 3.0
    assert row["rmse_after_k"] < 0.35
    shape = row["residual_shape_after_correction"]
    assert abs(shape["excess_kurtosis"]) < 1.0 and shape["fraction_beyond_3_sigma"] < 0.01
    # the pixel-weighted statistics over every block carry the cloud edges in
    pixel = ref.paired_stats(obs, sim, n_both.astype(float))
    unweighted = ref.paired_stats(obs, sim)
    assert unweighted["rmse_after_k"] > 1.5 and pixel["rmse_after_k"] < unweighted["rmse_after_k"]


def test_residual_shape_reads_a_planted_heavy_tail():
    rng = np.random.default_rng(7)
    gauss = rng.normal(0, 1, 20000)
    clean = ref.residual_shape(gauss)
    assert abs(clean["excess_kurtosis"]) < 0.2 and abs(clean["fraction_beyond_3_sigma"] - 0.0027) < 0.002
    heavy = gauss.copy()
    heavy[:400] = -12.0
    tail = ref.residual_shape(heavy, zenith=np.linspace(0, 60, heavy.size))
    assert tail["excess_kurtosis"] > 20 and tail["skew"] < -3 and tail["fraction_beyond_3_sigma"] > 0.015
    assert "correlation_with_zenith" in tail


# ---------------------------------------------------------------------------
# the entries
# ---------------------------------------------------------------------------

def _score(reference_run="crtm", with_filter_facing=True, admitted_after=0.7, land_after=3.1):
    def cls_row(after, n=5000):
        pixel = {"n": n, "bias_k": 0.05, "rmse_k": after * 0.9, "correlation": 0.99, "fit_intercept_k": -3.7, "fit_slope": 1.012,
                 "rmse_after_k": after * 0.85}
        ff = {"n": n - 800, "bias_k": 0.2, "rmse_k": after * 1.05, "correlation": 0.98, "fit_intercept_k": -4.7, "fit_slope": 1.015,
              "rmse_after_k": after, "qc": dict(ref.STREAM_QC), "blocks_rejected": {"below_minimum_pixels": 500, "below_minimum_clear_fraction": 300, "beyond_zenith": 0},
              "residual_shape_after_correction": {"n": n - 800, "excess_kurtosis": 30.0, "fraction_beyond_3_sigma": 0.012}}
        row = {"reference_vs_obs": pixel, "crtm_minus_fast": {"n": n, "bias_k": 0.05, "rmse_k": 0.14, "rmse_after_k": 0.13},
               "other_minus_fast": {"n": n, "bias_k": 0.5, "rmse_k": 0.7, "rmse_after_k": 0.4}}
        if with_filter_facing:
            row["filter_facing"] = ff
        else:
            row["filter_facing"] = {"n": 0, "verdict": "INCOMPLETE", "reason": "the block table carries no n_pair"}
        return row
    agreement = {"crtm": {"columns": 5000, "skin": {"rms_difference": 0.004, "rms_primary": 0.6, "mean_primary": 0.6, "mean_other": 0.6}},
                 "other": {"columns": 5000, "skin": {"rms_difference": 0.008, "rms_primary": 0.6, "mean_primary": 0.6, "mean_other": 0.61}}}
    return {"zenith_gate_deg": 60.0, "primary_run": "fast", "reference_run": reference_run, "stream_qc": dict(ref.STREAM_QC),
            "bands": {"13": {"classes": {"water": cls_row(admitted_after), "land": cls_row(land_after), "all": cls_row(2.0)},
                             "jacobian_agreement_with_primary": agreement}}}


def _table():
    return {"schema": "gpuwm-da.abi-fast-model.v1", "vertical": {"sha256": "x" * 64}, "written_utc": "2026-09-06T00:00:00+00:00",
            "bands": {"13": {"form": "linear", "planck": {"fk1": 10860.4, "fk2": 1395.19, "bc1": 0.0748, "bc2": 0.99975},
                             "validation": {"test_rms_k": 0.08, "test_bias_k": -0.01, "test_p99_abs_k": 0.25, "test_columns": 14000},
                             "reference_jacobians": {"peak_pressure_hpa_percentiles_5_25_50_75_95": [781.0, 915.0, 947.0, 973.0, 1012.0],
                                                     "half_sensitivity_band_hpa_median": [747.0, 936.0], "skin_jacobian_mean": 0.6}}}}


def test_entries_compare_with_the_named_reference_and_refuse_without_one():
    entries = op.fast_operator_entries(_score(), _table(), operator_run="fast")
    band = entries["bands"]["13"]
    assert entries["reference_run"] == "crtm"
    assert band["jacobians_vs_reference"]["reference_run"] == "crtm"
    assert band["jacobians_vs_reference"]["skin"]["rms_difference"] == 0.004      # not the other run's 0.008
    assert band["classes"]["water"]["reference_minus_operator_pixel_weighted"]["rmse_k"] == 0.14
    # an explicit name overrides the score's
    other = op.fast_operator_entries(_score(), _table(), operator_run="fast", reference_run="other")
    assert other["bands"]["13"]["jacobians_vs_reference"]["skin"]["rms_difference"] == 0.008
    with pytest.raises(AbiOperatorError, match="reference run.*none is named"):
        op.fast_operator_entries(_score(reference_run=None), _table(), operator_run="fast")
    with pytest.raises(AbiOperatorError, match="arbitrary other run"):
        op.fast_operator_entries(_score(), _table(), operator_run="fast", reference_run="missing")


def test_a_class_is_admitted_on_the_filter_facing_population_and_both_numbers_ride():
    entries = op.fast_operator_entries(_score(admitted_after=0.7, land_after=3.1), _table(), operator_run="fast")
    band = entries["bands"]["13"]
    water = band["classes"]["water"]
    assert water["verdict"] == "ADMITTED" and band["admitted_classes"] == ["water"]
    assert water["observation_error_k"] == 0.7                                   # the filter-facing residual, not 0.595
    assert water["bias_correction"] == {"form": "obs = intercept + slope * simulated", "intercept_k": -4.7, "slope": 1.015}
    assert water["n_blocks"] == 4200 and water["filter_facing"]["rmse_after_k"] == 0.7
    assert water["pixel_weighted_all_blocks"]["rmse_after_k"] == pytest.approx(0.595)
    assert water["filter_facing"]["residual_shape_after_correction"]["excess_kurtosis"] == 30.0
    assert water["filter_facing"]["blocks_rejected"]["below_minimum_clear_fraction"] == 300
    land = band["classes"]["land"]
    assert land["verdict"] == "NOT ADMITTED" and land["observation_error_k"] is None
    assert entries["gate"]["population"].startswith("the blocks the stream hands the filter")
    assert entries["stream_qc"] == ref.STREAM_QC
    contract = band["acceptance_contract"]
    assert "50%" in contract["error_correlations"] and "no further thinning" in contract["error_correlations"]
    assert "one block per model cell" not in contract["error_correlations"]
    # a filter-facing population inside the gate but too thin reads INCOMPLETE, never PASS
    thin = _score(admitted_after=0.7)
    thin["bands"]["13"]["classes"]["water"]["filter_facing"]["n"] = 400
    assert op.fast_operator_entries(thin, _table(), operator_run="fast")["bands"]["13"]["classes"]["water"]["verdict"] == "INCOMPLETE"
    # a filter-facing row that measured nothing reads INCOMPLETE even when the pixel-weighted number is inside the gate
    none = op.fast_operator_entries(_score(with_filter_facing=False), _table(), operator_run="fast")
    assert none["bands"]["13"]["classes"]["water"]["verdict"] == "INCOMPLETE"
    assert "n_pair" in none["bands"]["13"]["classes"]["water"]["reason"]


# ---------------------------------------------------------------------------
# the stream
# ---------------------------------------------------------------------------

_HEADER = ("block_x,block_y,lat_mean_deg,lon_mean_deg,zenith_mean_deg,n_sim,n_obs,n_pair,obs_mean_k,sim_mean_k,bias_k,"
           "rmse_k,obs_clear,obs_cloudy,sim_clear,sim_cloudy,n_both_clear,obs_mean_both_clear_k,sim_mean_both_clear_k")


def _blocks_csv(path: Path, rows):
    path.write_text("\n".join([_HEADER] + [",".join(str(v) for v in r) for r in rows]) + "\n")


def _rows():
    #      x   y     lat    lon    zen   nsim nobs npair obs   sim   bias rmse oc  ocl sc scl nboth obs_clear sim_clear
    return [(1, 2600, 10.0, -80.0, 30.0, 576, 576, 576, 290.0, 292.0, 2.0, 2.1, 500, 76, 576, 0, 500, 291.0, 293.0),   # water, clear
            (2, 2601, 10.5, -79.5, 31.0, 576, 576, 576, 280.0, 281.0, 1.0, 1.1, 30, 546, 576, 0, 30, 279.0, 290.0),    # water, cloud edge (5 %)
            (3, 2602, 11.0, -79.0, 32.0, 576, 576, 576, 285.0, 286.0, 1.0, 1.1, 5, 571, 576, 0, 5, 284.0, 286.0),      # water, too few pixels
            (4, 2603, 11.5, -78.5, 33.0, 576, 576, 576, 300.0, 303.0, 3.0, 3.1, 560, 16, 576, 0, 560, 301.0, 304.0),   # land, clear
            (5, 2604, 12.0, -78.0, 65.0, 576, 576, 576, 288.0, 289.0, 1.0, 1.1, 576, 0, 576, 0, 576, 288.0, 289.0)]    # water, beyond zenith


def _entry(admitted=("water",)):
    def cls(verdict, err, a, b):
        return {"verdict": verdict, "observation_error_k": err if verdict == "ADMITTED" else None,
                "bias_correction": ({"form": "obs = intercept + slope * simulated", "intercept_k": a, "slope": b} if verdict == "ADMITTED" else None),
                "filter_facing": {"qc": dict(ref.STREAM_QC)}}
    return {"band": 13, "admitted_classes": list(admitted),
            "classes": {"water": cls("ADMITTED" if "water" in admitted else "NOT ADMITTED", 0.7, -4.7, 1.015),
                        "land": cls("ADMITTED" if "land" in admitted else "NOT ADMITTED", 3.0, 70.0, 0.76)}}


def test_stream_batch_drops_cloud_edges_and_keeps_the_admitted_class_with_its_correction(tmp_path, point_obs):
    _blocks_csv(tmp_path / "b13.csv", _rows())
    start = dt.datetime(2026, 9, 1, 18, 0, 20, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 9, 1, 18, 9, 52, tzinfo=dt.timezone.utc)
    land = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    # without an entry: the QC alone, one error for every row
    batch, extras = rad.superobs_from_blocks(tmp_path / "b13.csv", band=13, observation_error_k=1.0, scan_start=start, scan_end=end)
    assert batch.count == 2 and batch.value.tolist() == [291.0, 301.0]
    assert batch.rejections == {"thinned_below_minimum_pixels": 1, "below_minimum_clear_fraction": 1, "beyond_zenith": 1}
    assert extras["qc"] == ref.STREAM_QC and np.allclose(extras["clear_fraction"], [500 / 576, 560 / 576])
    assert np.allclose(extras["correction_slope"], 1.0) and np.allclose(extras["correction_intercept_k"], 0.0)
    # with the entry: the admitted class only, the entry's error and correction per row
    batch, extras = rad.superobs_from_blocks(tmp_path / "b13.csv", band=13, entry=_entry(), land_fraction=land,
                                             scan_start=start, scan_end=end)
    assert batch.count == 1 and batch.value.tolist() == [291.0] and batch.error.tolist() == [0.7]
    assert batch.rejections["class_not_admitted"] == 1
    assert extras["surface_class"].tolist() == ["water"] and extras["admitted_classes"] == ["water"]
    assert extras["correction_intercept_k"].tolist() == [-4.7] and extras["correction_slope"].tolist() == [1.015]
    # a callable land fraction serves the same
    batch2, _ = rad.superobs_from_blocks(tmp_path / "b13.csv", band=13, entry=_entry(), scan_start=start, scan_end=end,
                                         land_fraction=lambda lat, lon: (lat > 11.2).astype(float))
    assert batch2.count == 1
    # both classes admitted: two rows, each with its own class's error
    batch3, extras3 = rad.superobs_from_blocks(tmp_path / "b13.csv", band=13, entry=_entry(("water", "land")), land_fraction=land,
                                               scan_start=start, scan_end=end)
    assert batch3.error.tolist() == [0.7, 3.0] and extras3["surface_class"].tolist() == ["water", "land"]


def test_stream_batch_refuses_an_entry_without_a_land_fraction_and_one_graded_under_another_qc(tmp_path, point_obs):
    _blocks_csv(tmp_path / "b13.csv", _rows())
    start = dt.datetime(2026, 9, 1, 18, 0, 20, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 9, 1, 18, 9, 52, tzinfo=dt.timezone.utc)
    with pytest.raises(AbiOperatorError, match="land_fraction"):
        rad.superobs_from_blocks(tmp_path / "b13.csv", band=13, entry=_entry(), scan_start=start, scan_end=end)
    with pytest.raises(AbiOperatorError, match="graded with minimum_clear_fraction"):
        rad.superobs_from_blocks(tmp_path / "b13.csv", band=13, entry=_entry(), land_fraction=np.zeros(5), scan_start=start,
                                 scan_end=end, minimum_clear_fraction=0.1)
    with pytest.raises(AbiOperatorError, match="observation_error_k"):
        rad.superobs_from_blocks(tmp_path / "b13.csv", band=13, scan_start=start, scan_end=end)
    with pytest.raises(AbiOperatorError, match="band 8"):
        rad.superobs_from_blocks(tmp_path / "b13.csv", band=13, entry=dict(_entry(), band=8), land_fraction=np.zeros(5),
                                 scan_start=start, scan_end=end)


def test_the_operator_applies_the_registered_correction(tmp_path, point_obs, monkeypatch):
    _blocks_csv(tmp_path / "b13.csv", _rows())
    start = dt.datetime(2026, 9, 1, 18, 0, 20, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 9, 1, 18, 9, 52, tzinfo=dt.timezone.utc)
    batch, extras = rad.superobs_from_blocks(tmp_path / "b13.csv", band=13, entry=_entry(("water", "land")),
                                             land_fraction=np.array([0.0, 0.0, 0.0, 1.0, 0.0]), scan_start=start, scan_end=end)
    operator = rad.AbiRadianceOperator.__new__(rad.AbiRadianceOperator)
    operator.band = 13
    operator.zenith_by_identity = {}
    operator.correction_by_identity = {}
    operator.work_dir = tmp_path
    operator.table_path = tmp_path / "table.json"
    operator.rw_goes = tmp_path / "rw_goes"
    operator.threads = None
    operator.keep_files = False
    operator.receipts = []
    rad.register_batch(operator, batch, extras)
    assert batch.operator is operator
    assert operator.correction_by_identity == {"abi13:1:2600": (-4.7, 1.015), "abi13:4:2603": (70.0, 0.76)}
    raw = np.array([[293.0, 304.0], [292.0, 303.0]])
    monkeypatch.setattr(operator, "member_columns", lambda members, lat, lon, zenith: {"stub": True})
    monkeypatch.setattr(rad.ref, "write_columns", lambda path, columns, provenance=None: path)
    monkeypatch.setattr(rad, "run_forward", lambda *a, **k: {"_wall_s": 0.0, "columns_sha256": "", "table_sha256": "", "coordinate_sha256": "", "threads": 1})
    monkeypatch.setattr(rad.ref, "read_crtm_output", lambda path: {"bt": {13: raw.reshape(-1)}})
    out = operator(["m1", "m2"], batch)
    expected = np.array([[-4.7 + 1.015 * 293.0, 70.0 + 0.76 * 304.0], [-4.7 + 1.015 * 292.0, 70.0 + 0.76 * 303.0]])
    assert np.allclose(out, expected)
    assert operator.receipts[-1]["rows_corrected"] == 2
    # a batch registered without a correction passes the operator's output through
    plain, plain_extras = rad.superobs_from_blocks(tmp_path / "b13.csv", band=13, observation_error_k=1.0, scan_start=start, scan_end=end)
    operator.correction_by_identity = {}
    rad.register_batch(operator, plain, plain_extras)
    assert operator.correction_by_identity == {}
    assert np.allclose(operator(["m1", "m2"], plain), raw)


def test_the_tangent_linear_vapor_term_is_the_relative_form(tmp_path, point_obs, monkeypatch):
    """The members and the hybrid's static draws pass through the forward's
    Jacobians about the reference column.  On the water-vapor band the
    per-g/kg vapor Jacobian is enormous where the reference is nearly dry
    (the model's upper layers), so an absolute departure of 1e-4 g/kg over a
    1e-7 g/kg reference read as millions of kelvin (the completed system's
    first analysis, 2026-09-07: band 8 O-A rms 2.9e6 K, the static draws'
    spread 7.0e6 K against the members' 0.84 K).  The relative form is the
    same to first order and bounded by the logarithm of the ratio."""
    from types import SimpleNamespace

    nlay = 6
    jac_q = np.array([[1.0e7, 1.0e5, -2.0, -1.5, -0.6, -0.1]])           # K per g/kg, top first
    q_ref = np.array([[1.0e-7, 1.0e-5, 0.05, 0.4, 2.0, 8.0]])
    jac_t = np.zeros((1, nlay))
    t_ref = np.full((1, nlay), 250.0)
    # small relative departures: the relative form equals the per-g/kg form to first order
    q_small = q_ref * (1.0 + np.array([[0.0, 0.0, 0.01, -0.01, 0.02, -0.005]]))
    first_order = np.sum(jac_q * (q_small - q_ref), axis=1)
    got = rad.vapor_tangent_linear(jac_q, q_ref, q_small)
    assert np.allclose(got, first_order, rtol=2e-2, atol=1e-9)
    # a static-draw-sized absolute departure on the dry layers: bounded, not millions of kelvin
    q_draw = q_ref + np.array([[1.0e-4, 1.0e-4, 0.0, 0.0, 0.0, 0.0]])
    per_gkg = np.sum(jac_q * (q_draw - q_ref), axis=1)
    assert per_gkg[0] > 1.0e3
    bounded = rad.vapor_tangent_linear(jac_q, q_ref, q_draw)
    # jac_q * q_ref is 1 K per ln q on both dry layers: ln(1e-4 / 1e-6) + ln(1.1e-4 / 1e-5), about 7 K
    assert abs(bounded[0]) < 10.0 and per_gkg[0] / abs(bounded[0]) > 100.0 and np.isfinite(bounded).all()
    # a reference layer below the floor contributes nothing, a zero member layer is floored, never -inf
    assert np.isfinite(rad.vapor_tangent_linear(jac_q, np.zeros((1, nlay)), q_draw)).all()
    assert np.isfinite(rad.vapor_tangent_linear(jac_q, q_ref, np.zeros((1, nlay)))).all()
    # and the operator uses it: the members' mean is the reference (tiny vapor aloft, the forward's huge
    # per-g/kg Jacobian there); the static draws then pass through the cached reference with a
    # draw-sized departure, and read bounded where the per-g/kg form read thousands of kelvin
    _blocks_csv(tmp_path / "b8.csv", _rows())
    start = dt.datetime(2026, 9, 1, 18, 0, 20, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 9, 1, 18, 9, 52, tzinfo=dt.timezone.utc)
    batch, extras = rad.superobs_from_blocks(tmp_path / "b8.csv", band=13, observation_error_k=1.0, scan_start=start, scan_end=end)
    n = batch.count
    operator = rad.AbiRadianceOperator.__new__(rad.AbiRadianceOperator)
    operator.band = 13
    operator.zenith_by_identity = {}
    operator.correction_by_identity = {}
    operator.work_dir = tmp_path
    operator.table_path = tmp_path / "table.json"
    operator.rw_goes = tmp_path / "rw_goes"
    operator.threads = None
    operator.keep_files = False
    operator.receipts = []
    operator.references = {}
    operator.linearise_members = True
    operator.check_rows = 1
    operator.check_members = 1
    operator.last_linearisation_check = None
    operator.linear_regime_multiple = float("inf")      # this test reads the linear form itself, not the guard
    operator.linear_regime_floor_k = float("inf")
    rad.register_batch(operator, batch, extras)
    r = 2
    temperature = np.tile(t_ref, (r * n, 1))
    members_q = np.concatenate([np.tile(q_ref * 0.95, (n, 1)), np.tile(q_ref * 1.05, (n, 1))])   # a 10 percent spread
    draws_q = np.concatenate([np.tile(q_ref, (n, 1)), np.tile(q_draw, (n, 1))])                  # one draw-sized departure
    columns = {"which": "members"}

    def member_columns(members, lat, lon, zenith):
        q = members_q if columns["which"] == "members" else draws_q
        return {"temperature_k": temperature, "q_gkg": q, "lat": np.zeros(r * n), "lon": np.zeros(r * n),
                "zenith": np.zeros(r * n), "checks": {"members": r, "points": n}}

    asked = {}

    def write_columns(path, columns_written, provenance=None):
        asked["size"] = int(provenance["members"]) * int(provenance["points"])
        return path

    monkeypatch.setattr(operator, "member_columns", member_columns)
    monkeypatch.setattr(operator, "transform_for", lambda members: SimpleNamespace(truncation=127))
    monkeypatch.setattr(rad.ref, "write_columns", write_columns)
    monkeypatch.setattr(rad, "run_forward", lambda *a, **k: {"_wall_s": 0.0})
    monkeypatch.setattr(rad.ref, "read_crtm_output", lambda path: {
        "bt": {13: np.full(asked["size"], 240.0)}, "jac_t": {13: np.tile(jac_t, (asked["size"], 1))},
        "jac_q": {13: np.tile(jac_q, (asked["size"], 1))}})
    out = operator(["m1", "m2"], batch)
    assert out.shape == (r, n) and np.all(np.abs(out - 240.0) < 0.2), out          # the members: a first-order spread
    assert operator.receipts[-1]["mode"] == "linearised" and operator.last_linearisation_check["rows_checked"] == 1
    columns["which"] = "draws"
    draws = operator(["s1", "s2"], batch)
    assert np.allclose(draws[0], 240.0, atol=1e-9)
    assert np.all(np.abs(draws[1] - 240.0) < 10.0) and np.all(np.abs(draws[1] - 240.0) > 1.0), draws[1]
    assert operator.receipts[-1]["rows_corrected"] == 0


def test_a_pair_beyond_the_linear_regime_takes_the_full_forward_and_the_window_record_counts_it(tmp_path, point_obs, monkeypatch):
    """The run of record's band-8 linearisation residual read 1.22 K rms
    (largest 10.2 K) at 23Z and 2.02 K rms (largest 17.5 K) at 00Z against a
    1.21 K error: a few members' vapor sat e-folds from the reference and the
    linear form, summed over the layers, read what the forward never does.
    A pair whose linear departure exceeds the guard's limit is evaluated
    with the full forward, the receipt counts the pairs, and the window's
    record accumulates over every call instead of keeping the last one."""
    from types import SimpleNamespace

    nlay = 6
    jac_q = np.array([[1.0e7, 1.0e5, -2.0, -1.5, -0.6, -0.1]])
    q_ref = np.array([[1.0e-7, 1.0e-5, 0.05, 0.4, 2.0, 8.0]])
    jac_t = np.zeros((1, nlay))
    t_ref = np.full((1, nlay), 250.0)
    q_far = q_ref + np.array([[1.0e-4, 1.0e-4, 0.0, 0.0, 0.0, 0.0]])          # about 7 K in the linear form
    assert abs(rad.vapor_tangent_linear(jac_q, q_ref, q_far)[0]) > 3.0
    _blocks_csv(tmp_path / "b8.csv", _rows())
    start = dt.datetime(2026, 9, 1, 18, 0, 20, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 9, 1, 18, 9, 52, tzinfo=dt.timezone.utc)
    batch, extras = rad.superobs_from_blocks(tmp_path / "b8.csv", band=13, observation_error_k=1.0, scan_start=start, scan_end=end)
    n = batch.count
    operator = rad.AbiRadianceOperator.__new__(rad.AbiRadianceOperator)
    operator.band = 13
    operator.zenith_by_identity = {}
    operator.correction_by_identity = {}
    operator.work_dir = tmp_path
    operator.table_path = tmp_path / "table.json"
    operator.rw_goes = tmp_path / "rw_goes"
    operator.threads = None
    operator.keep_files = False
    operator.receipts = []
    operator.references = {}
    operator.linearise_members = True
    operator.check_rows = 1
    operator.check_members = 1
    operator.last_linearisation_check = None
    operator._window_checks = []
    operator.linear_regime_multiple = rad.LINEAR_REGIME_ERROR_MULTIPLE      # 3 assigned errors: 3 K on this batch
    operator.linear_regime_floor_k = rad.LINEAR_REGIME_FLOOR_K
    rad.register_batch(operator, batch, extras)
    r = 2
    which = {"states": "single"}
    written = {}

    def member_columns(members, lat, lon, zenith):
        if which["states"] == "single":                   # the control background: the reference column itself
            q = np.tile(q_ref, (n, 1))
            return {"temperature_k": np.tile(t_ref, (n, 1)), "q_gkg": q, "lat": np.zeros(n), "lon": np.zeros(n),
                    "zenith": np.zeros(n), "checks": {"members": 1, "points": n}}
        q = np.concatenate([np.tile(q_ref * 0.95, (n, 1)), np.tile(q_far, (n, 1))])   # member 2 is out of the regime
        return {"temperature_k": np.tile(t_ref, (r * n, 1)), "q_gkg": q, "lat": np.zeros(r * n), "lon": np.zeros(r * n),
                "zenith": np.zeros(r * n), "checks": {"members": r, "points": n}}

    def write_columns(path, columns_written, provenance=None):
        written["q"] = np.asarray(columns_written["q_gkg"])
        return path

    def read_crtm_output(path):
        q = written["q"]
        # the "full forward": 240 K on the reference column, 236 K on the far column (nonlinear, not 247)
        bt = np.where(np.isclose(q[:, 0], q_far[0, 0]), 236.0, 240.0)
        return {"bt": {13: bt}, "jac_t": {13: np.tile(jac_t, (q.shape[0], 1))}, "jac_q": {13: np.tile(jac_q, (q.shape[0], 1))}}

    monkeypatch.setattr(operator, "member_columns", member_columns)
    monkeypatch.setattr(operator, "transform_for", lambda members: SimpleNamespace(truncation=127))
    monkeypatch.setattr(rad.ref, "write_columns", write_columns)
    monkeypatch.setattr(rad, "run_forward", lambda *a, **k: {"_wall_s": 0.0})
    monkeypatch.setattr(rad.ref, "read_crtm_output", read_crtm_output)
    single = operator(["s1"], batch)                                            # the reference: the full forward, cached
    assert np.allclose(single[0], 240.0) and operator.receipts[-1]["mode"] == "full"
    which["states"] = "members"
    out = operator(["m1", "m2"], batch)                                         # the members about the cached reference
    assert np.all(np.abs(out[0] - 240.0) < 0.2), out[0]                         # inside the regime: the linear form
    assert np.allclose(out[1], 236.0), out[1]                                    # beyond it: the full forward's own reading
    assert operator.receipts[-1]["mode"] == "linearised" and operator.receipts[-1]["pairs_full_forward"] == n
    record = operator.last_linearisation_check
    assert record["calls"] == 1 and record["pairs_full_forward"] == n and record["pairs_evaluated"] == r * n
    assert record["max_abs_departure_k"] > 3.0 and record["linear_regime"] == {"multiple_of_error": 3.0, "floor_k": 1.0}
    # the second call of the window (the analysed members through the cached references) accumulates
    operator(["m1", "m2"], batch)
    assert operator.last_linearisation_check["calls"] == 2
    assert operator.last_linearisation_check["pairs_full_forward"] == 2 * n
    # the reset keeps the finished window's record readable and starts the next window's
    operator.reset()
    assert operator.last_linearisation_check["calls"] == 2 and operator._window_checks == []
    which["states"] = "single"
    operator(["s1"], batch)
    which["states"] = "members"
    operator(["m1", "m2"], batch)
    assert operator.last_linearisation_check["calls"] == 1
    # with the guard off the far member reads the linear form, out of the regime
    operator.linear_regime_multiple = float("inf")
    operator.linear_regime_floor_k = float("inf")
    loose = operator(["m1", "m2"], batch)
    assert np.all(np.abs(loose[1] - 240.0) > 3.0) and operator.receipts[-1]["pairs_full_forward"] == 0
