"""Radiances in the ensemble filter: the model-space vertical localisation,
the radiance streams' batch path through the door, and the day update.

Held here, without a card, a granule or a Rust door:

* a point row's profile reproduces the Gaspari-Cohn rule on the axis, and
  a profile row's weight is read at every level by interpolation (the
  filter's lookup against the analytic kernel);
* a channel profile is the weighting function convolved with the kernel:
  it peaks where the channel sounds and is small at the surface, and the
  measured correlation half width maps to a bounded cutoff;
* a single radiance report with a profile increments the levels the
  profile names (the increment across levels follows the profile, not the
  centroid's Gaspari-Cohn), against the analytic localised gain;
* the ATMS member operator through the door's contract: a batch built
  from cells fills ``(R, n)`` for the members through the tangent-linear
  transfer and for a single state through the full transfer, subsets
  agree, the transform is chosen by truncation and an unbound truncation
  is refused by name;
* a synthetic radiance stream (a toy operator that reads the members'
  lowest-level temperature, a profile at the surface) runs through
  ``woof global da cycle`` under the letkf filter beside the point
  streams: the report carries the stream with O-B and O-A, its Desroziers
  reading, the localisation profile record and the window's radiance
  receipt; the successive-correction filter refuses it by name; the bias
  ledger moves by the memory rule and reads back.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.globe.configs_dir import config_root as _shipped_configs
from woof.globe import da_door, da_streams
from woof.globe.config import load_config
from woof.globe.da import EnsembleOptions, FilterOptions, GlobalEnsemble, PointObs, ensemble_config
from woof.globe.da.analysis import analyze_ensemble
from woof.globe.da.letkf_point import profile_weight, single_observation_increment
from woof.globe.da.localisation import (
    RADIANCE_CUTOFF_BOUNDS_LNP,
    cutoff_from_half_width,
    point_profile,
    profile_from_weighting_function,
    vertical_correlation_length,
)
from woof.globe.da.observations import LOCALISATION_AXIS_LNP
from woof.globe.da.operators import evaluate_batches
from woof.globe.da_filter import ENSEMBLE_MANIFEST_NAME
from woof.globe.microwave.entry import (
    AtmsBatchOperator,
    channel_profile,
    point_obs_from_cells,
    read_entry,
    stream_name,
)
from woof.globe.radiance_streams import BiasLedger, RadianceContext, TABLES_DIR
from woof.globe.runner import build_model_and_cold_state, build_transform
from woof.da.letkf import gaspari_cohn

from test_arwen_global_assimilate import (  # noqa: F401 - the fixture rides the import
    CONFIG, _synthetic_obs_files, spun_up,
)
from test_arwen_global_cycle import START_TEXT, OPTIONS

SMOKE = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
ENTRY = TABLES_DIR / "atms-noaa-21-2026-09-01.entry.json"


# ---------------------------------------------------------------------------
# the profile and its lookup
# ---------------------------------------------------------------------------

def test_a_point_profile_is_the_gaspari_cohn_rule_and_the_lookup_reads_it():
    prof = point_profile(np.log(50_000.0), 0.6)
    assert prof.max() > 0.99 and prof[0] == 0.0 and prof[-1] == 0.0
    levels = np.log(np.array([1_000.0, 20_000.0, 50_000.0, 70_000.0, 100_000.0]))
    got = profile_weight(np.broadcast_to(prof, (1, levels.size, prof.size)), levels[None, :], np)[0]
    want = gaspari_cohn(np.abs(levels - np.log(50_000.0)) / 0.6, 1.0)
    assert np.allclose(got, want, atol=0.02)   # the 0.05 ln p tabulation reads the peak within a percent
    assert got[2] > 0.98


def test_a_channel_profile_peaks_where_the_channel_sounds_and_the_cutoff_is_bounded():
    prof7 = channel_profile(7, 30.0, 1.2)
    prof12 = channel_profile(12, 30.0, 1.2)
    peak7 = np.exp(LOCALISATION_AXIS_LNP[prof7.argmax()])
    peak12 = np.exp(LOCALISATION_AXIS_LNP[prof12.argmax()])
    assert 15_000.0 < peak7 < 60_000.0 and peak12 < 5_000.0
    assert prof7[-1] < 0.4 and prof12[-1] < 0.01
    assert prof7.max() == pytest.approx(1.0) and prof12.max() == pytest.approx(1.0)
    # A weighting function of one layer convolved with the kernel is the kernel.
    single = profile_from_weighting_function([np.log(30_000.0)], [1.0], 1.0)
    assert np.allclose(single, point_profile(np.log(30_000.0), 1.0), atol=2e-3)
    assert cutoff_from_half_width(0.1) == RADIANCE_CUTOFF_BOUNDS_LNP[0]
    assert cutoff_from_half_width(5.0) == RADIANCE_CUTOFF_BOUNDS_LNP[1]
    assert cutoff_from_half_width(0.4) == pytest.approx(1.4)
    assert cutoff_from_half_width(float("nan")) == RADIANCE_CUTOFF_BOUNDS_LNP[1]
    with pytest.raises(ValueError, match="no positive weight"):
        profile_from_weighting_function([1.0, 2.0], [0.0, 0.0], 1.0)


def test_the_correlation_half_width_reads_a_planted_vertical_structure():
    rng = np.random.default_rng(2)
    nlev, nlat, nlon, members = 12, 6, 8, 40
    lnp = np.linspace(np.log(2_000.0), np.log(100_000.0), nlev)
    ln_p_full = np.broadcast_to(lnp[:, None, None], (nlev, nlat, nlon)).copy()
    # Perturbations correlated over one unit of ln p: a Gaussian kernel of that scale.
    scale = 0.5
    kernel = np.exp(-0.5 * ((lnp[:, None] - lnp[None, :]) / scale) ** 2)
    chol = np.linalg.cholesky(kernel + 1e-9 * np.eye(nlev))
    theta = np.einsum("kl,rlji->rkji", chol, rng.standard_normal((members, nlev, nlat, nlon)))
    read = vertical_correlation_length(theta, ln_p_full, np.log(30_000.0), latitude_deg=np.linspace(60, -60, nlat),
                                       ring_stride=1, lon_stride=1)
    # exp(-0.5 (d / 0.5)^2) = 0.5 at d = 0.59.
    assert 0.4 < read["half_width_lnp"] < 0.9
    assert read["columns_sampled"] == nlat * nlon and read["members"] == members
    assert RADIANCE_CUTOFF_BOUNDS_LNP[0] <= read["cutoff_lnp"] <= RADIANCE_CUTOFF_BOUNDS_LNP[1]


# ---------------------------------------------------------------------------
# the single report with a profile
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def world():
    cfg = load_config(SMOKE)
    options = EnsembleOptions(members=6, truncation=cfg.truncation, seed=3, additive_inflation_fraction=0.0)
    ecfg = ensemble_config(cfg, options)
    transform = build_transform(ecfg)
    model, cold = build_model_and_cold_state(ecfg, transform)
    return cfg, ecfg, options, transform, model, cold


def _ensemble(world):
    cfg, ecfg, options, transform, model, cold = world
    return GlobalEnsemble.from_state(ecfg, model, transform, cold, options)


class _LevelOperator:
    """A toy radiance: the state's temperature at level ``k`` at the rows,
    the transform chosen by the states' truncation."""

    evaluates_states = True

    def __init__(self, transform, model, level: int):
        self.transforms = {int(transform.truncation): transform}
        self.model = model
        self.level = int(level)

    def bind(self, transform):
        self.transforms[int(transform.truncation)] = transform
        return self

    def __call__(self, states, batch):
        from woof.globe.microwave.entry import truncation_of

        t = truncation_of(states)
        if t not in self.transforms:
            raise ValueError(f"unbound truncation T{t}")
        transform = self.transforms[t]
        from woof.globe.assimilate import _sample_grid

        out = np.empty((len(states), batch.count))
        for k, state in enumerate(states):
            g = self.model.grid_state(state.atmosphere, only=("temperature",))
            field = np.asarray(transform.backend.to_numpy(g["temperature"][self.level]), dtype=np.float64)
            self.model.release_syntheses()
            out[k] = _sample_grid(field, transform.grid, batch.latitude_deg, batch.longitude_deg)
        return out


def test_a_profile_row_increments_the_levels_its_profile_names(world):
    cfg, ecfg, options, transform, model, cold = world
    ensemble = _ensemble(world)
    fields, ln_p_full, _ = ensemble.grid_fields(("theta",))
    theta = np.asarray(fields["theta"])
    nlev = theta.shape[1]
    lat, lon = 10.0, 100.0
    # A profile with its weight on the two highest levels that carry
    # ensemble spread only (the balanced perturbation family's cosine modes
    # have no temperature draw at the very top level, where a level with no
    # spread takes no increment whatever the profile says).
    spread = theta.std(axis=0).mean(axis=(1, 2))
    with_spread = [k for k in range(nlev) if spread[k] > 1e-9]
    assert len(with_spread) >= 4, f"the smoke ensemble has spread on {len(with_spread)} levels only"
    k_top, k_second = with_spread[0], with_spread[1]
    lnp_top = float(np.asarray(ln_p_full)[k_top].mean())
    lnp_second = float(np.asarray(ln_p_full)[k_second].mean())
    prof = np.zeros(LOCALISATION_AXIS_LNP.size)
    for z in (lnp_top, lnp_second):
        prof = np.maximum(prof, point_profile(z, 0.3))
    op = _LevelOperator(transform, model, k_second)
    batch = PointObs(
        stream="toy-radiance", variable="brightness_temperature_k", latitude_deg=[lat], longitude_deg=[lon],
        ln_pressure=[lnp_second], surface=[False], value=[250.0], error=[1.0],
        identity=np.array(["toy:1"], dtype=object), valid_time=[dt.datetime(2026, 8, 31, 12, tzinfo=dt.timezone.utc)],
        localisation_profile=prof[None, :], operator=op, horizontal_cutoff_km=20_000.0,
    )
    evaluate_batches(None, ensemble.members, [batch], target="simulated")
    assert batch.simulated.shape == (6, 1) and np.all(np.isfinite(batch.simulated))
    before = np.asarray(ensemble.grid_fields(("theta",))[0]["theta"]).mean(axis=0)
    # The increment is read off the members directly: this test measures
    # where the profile let the row act, not how the increment is applied
    # (the package applies it through the IAU by default, over the window).
    opts = FilterOptions(thinning=False, withheld_fraction=0.01, gate_minimum_count=1000, rtps_alpha=0.0,
                         background_check_sigmas=1e6, horizontal_cutoff_km=20_000.0,
                         increment_application="direct")
    result = analyze_ensemble(ensemble, [batch], opts, analysis_time=batch.valid_time[0], additive_inflation=False)
    assert result.status == "pass"
    after = np.asarray(ensemble.grid_fields(("theta",))[0]["theta"]).mean(axis=0)
    inc = after - before
    rms = np.sqrt((inc ** 2).mean(axis=(1, 2)))
    # The two named levels moved, the lowest levels did not.
    assert rms[k_top] > 0.0 and rms[k_second] > 0.0
    assert rms[-1] == 0.0 and rms[nlev // 2] == 0.0 if nlev > 3 else True
    record = result.report["localisation"]["profile_rows"]["toy-radiance/brightness_temperature_k"]
    assert record["rows"] == 1 and record["peak_pressure_pa"] < np.exp(lnp_second) * 1.5
    assert "not Gaspari-Cohn on a centroid" in record["rule"]


def test_the_analytic_gain_of_a_profile_row_matches_the_filter(world):
    cfg, ecfg, options, transform, model, cold = world
    ensemble = _ensemble(world)
    fields, ln_p_full, _ = ensemble.grid_fields(("theta",))
    theta = np.asarray(fields["theta"])
    lnp_levels = np.asarray(ln_p_full)
    lat, lon = -20.0, 200.0
    z0 = float(lnp_levels[2].mean())
    prof = point_profile(z0, 0.8)
    op = _LevelOperator(transform, model, 2)
    batch = PointObs(
        stream="toy-radiance", variable="brightness_temperature_k", latitude_deg=[lat], longitude_deg=[lon],
        ln_pressure=[z0], surface=[False], value=[0.0], error=[0.7],
        identity=np.array(["toy:2"], dtype=object), valid_time=[dt.datetime(2026, 8, 31, 12, tzinfo=dt.timezone.utc)],
        localisation_profile=prof[None, :], operator=op, horizontal_cutoff_km=3_000.0,
    )
    evaluate_batches(None, ensemble.members, [batch], target="simulated")
    batch.value[:] = batch.simulated.mean() + 1.0
    from woof.globe.constants import EARTH_RADIUS_M

    grid = transform.grid
    glat = np.deg2rad(np.asarray(grid.latitude_deg))[:, None]
    glon = np.deg2rad(np.asarray(grid.longitude_deg))[None, :]
    plat, plon = np.deg2rad(lat), np.deg2rad(lon)
    a = np.sin((glat - plat) / 2) ** 2 + np.cos(glat) * np.cos(plat) * np.sin((glon - plon) / 2) ** 2
    dist = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.minimum(a, 1.0)))
    wh = gaspari_cohn(dist / 3.0e6, 1.0)
    wv = np.stack([profile_weight(np.broadcast_to(prof, (1, wh.size, prof.size)),
                                  lnp_levels[k].reshape(1, -1), np)[0].reshape(wh.shape) for k in range(theta.shape[1])])
    weight = wh[None] * wv
    analytic = single_observation_increment(theta, batch.simulated[:, 0], batch.value[0], batch.error[0], weight)
    # The local solve's own grid increment (before the spectral projection
    # that smooths it into the state): the profile's weight at every level.
    from woof.globe.da.letkf_point import (
        ColumnGeometry, PointLetkfConfig, analyze_points_with_control, flatten_batches,
    )

    geometry = ColumnGeometry(latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg,
                              ln_p_full=lnp_levels, ln_ps=np.asarray(ensemble.grid_fields(("theta",))[2]),
                              radius_m=float(grid.radius_m))
    flat = flatten_batches([batch], np, horizontal_cutoff_m=3.0e6,
                           vertical_cutoff_for=lambda b, surface: np.full(b.count, 1.0), solve_dtype="float64")
    solved = analyze_points_with_control({"theta": theta}, flat, geometry, PointLetkfConfig(rtps_alpha=0.0))
    got = np.asarray(solved.increments["theta"]).mean(axis=0)
    scale = np.abs(analytic).max()
    assert scale > 0.0
    assert np.abs(got - analytic).max() < 1e-6 * scale
    # And without the profile the same row would have reached every level through the centroid's cutoff alone.
    assert (np.abs(got).max(axis=(1, 2)) > 0.0).sum() < theta.shape[1] or theta.shape[1] <= 2


# ---------------------------------------------------------------------------
# the ATMS operator through the contract
# ---------------------------------------------------------------------------

def _cells(rng, n):
    return SimpleNamespace(
        lat_mean_deg=rng.uniform(-50.0, 50.0, n), lon_mean_deg=rng.uniform(0.0, 360.0, n),
        zenith_mean_deg=rng.uniform(0.0, 55.0, n), scan_angle_abs_mean_deg=rng.uniform(0.0, 45.0, n),
        time_mean_unix_s=np.full(n, 1.7882e9), tb_mean_k=np.full((n, 22), 230.0),
        cell_bin=np.zeros(n, dtype=np.int64), cell_j=np.arange(n), cell_i=np.zeros(n, dtype=int),
    )


def test_the_atms_operator_answers_for_members_and_a_single_state_by_truncation(world):
    cfg, ecfg, options, transform, model, cold = world
    ensemble = _ensemble(world)
    entry = read_entry(ENTRY)
    cells = _cells(np.random.default_rng(1), 10)
    batches = point_obs_from_cells(entry, cells, transforms=[transform], vertical=model.vertical, channels=(5, 7, 9),
                                   satellite="noaa-21", cutoffs_lnp={5: 1.2, 7: 1.0, 9: 1.4})
    assert [b.stream for b in batches] == [stream_name("noaa-21", c) for c in (5, 7, 9)]
    assert all(b.localisation_profile.shape == (10, LOCALISATION_AXIS_LNP.size) for b in batches)
    op = batches[0].operator
    assert isinstance(op, AtmsBatchOperator) and op.evaluates_states
    linear = op(ensemble.members, batches[1])
    assert linear.shape == (6, 10) and np.all(np.isfinite(linear))
    check = op.last_linearisation_check
    assert check["rows_checked"] == 10 and check["rms_k"] < 0.3
    op.linearise_members = False
    full = op(ensemble.members, batches[1])
    assert np.abs(full - linear).max() < 0.5
    assert np.std(full, axis=0).mean() > 0.05
    single = op([ensemble.members[2]], batches[1])
    assert single.shape == (1, 10) and np.allclose(single[0], full[2])
    part = op([ensemble.members[2]], batches[1].subset([3, 7]))
    assert np.allclose(part[0], full[2, [3, 7]], atol=1e-9)
    # evaluate_batches routes the batch through the operator (members and control alike).
    evaluate_batches(None, ensemble.members, batches, target="simulated")
    evaluate_batches(None, [ensemble.members[0]], batches, target="control_simulated")
    assert batches[2].simulated.shape == (6, 10) and batches[2].control_simulated.shape == (1, 10)
    # An unbound truncation is refused by name.
    bare = AtmsBatchOperator(entry=entry, vertical=model.vertical, geometry=dict(op.geometry))
    with pytest.raises(ValueError, match="holds no transform at T"):
        bare(ensemble.members, batches[1])
    # The bias correction is the live coefficients.
    op.coefficients[7]["a"] += 1.0
    shifted = op([ensemble.members[2]], batches[1])
    assert np.allclose(shifted[0] - single[0], 1.0)


def test_the_atms_linearisation_record_carries_every_channel_of_the_window(world):
    """One record per call, overwritten by the next, left the case day's
    receipts with channel 14 alone on every window while the check had run
    on all eleven channels; the record accumulates over the window's calls
    and the reset keeps the finished window's record for the stream to read."""
    cfg, ecfg, options, transform, model, cold = world
    ensemble = _ensemble(world)
    entry = read_entry(ENTRY)
    cells = _cells(np.random.default_rng(3), 8)
    batches = point_obs_from_cells(entry, cells, transforms=[transform], vertical=model.vertical, channels=(5, 7, 9),
                                   satellite="noaa-21", cutoffs_lnp={5: 1.2, 7: 1.0, 9: 1.4})
    op = batches[0].operator
    for batch in batches:
        op(ensemble.members, batch)
    record = op.last_linearisation_check
    assert record["calls"] == 3 and sorted(record["by_channel"]) == [5, 7, 9] and record["rows_checked"] == 24
    assert all(v["count"] == 16 and v["rms_k"] is not None for v in record["by_channel"].values())
    assert record["rms_k"] is not None and record["max_abs_k"] >= max(v["max_abs_k"] for v in record["by_channel"].values())
    op.reset()
    assert op.last_linearisation_check["calls"] == 3                     # the stream reads it after the reset
    op(ensemble.members, batches[1])
    assert op.last_linearisation_check["calls"] == 1 and sorted(op.last_linearisation_check["by_channel"]) == [7]


# ---------------------------------------------------------------------------
# the stream through the door
# ---------------------------------------------------------------------------

class _ToyRadianceStream:
    """A radiance stream for the door test: one batch of the members'
    lowest-level temperature at synthetic ocean points with a surface
    profile, a fetch that records nothing but a manifest line."""

    name = "toy-radiance"
    description = "the lowest-level temperature at synthetic points, as a radiance stream"

    def __init__(self, count: int = 24):
        self.count = int(count)
        self.receipts = []
        self.operator = None

    def fetch(self, window_start, window_end, out_dir):
        path = Path(out_dir) / f"toy-{da_streams._utc(window_end):%Y%m%dT%H%M%SZ}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"window_end": da_streams._stamp(window_end)}), encoding="utf-8")
        size, sha = da_streams._digest(path)
        return [da_streams.FetchRecord(
            stream=self.name, location="synthetic", path=str(path), bytes=size, sha256=sha,
            window_start_utc=da_streams._stamp(window_start), window_end_utc=da_streams._stamp(window_end),
            fetched_utc=da_streams._stamp(dt.datetime.now(dt.timezone.utc)), wall_s=0.0,
            latency_behind_real_time_s=None, decoder="toy")]

    def batches(self, context: RadianceContext):
        ensemble = context.ensemble
        model = ensemble.model
        nlev = int(np.asarray(model.vertical.a_half_pa).size) - 1
        if self.operator is None:
            self.operator = _LevelOperator(context.ensemble_transform, model, nlev - 1)
        self.operator.bind(context.control_transform)
        rng = np.random.default_rng(int(context.window_end.timestamp()) % 1000)
        n = self.count
        lat = rng.uniform(-40.0, 40.0, n)
        lon = rng.uniform(0.0, 360.0, n)
        lnp = np.log(95_000.0)
        prof = point_profile(lnp, 0.6)
        stamp = context.window_end.strftime("%Y%m%dT%H%M%S")
        # The truth: the control's own reading plus noise.
        probe = PointObs(stream=self.name, variable="brightness_temperature_k", latitude_deg=lat, longitude_deg=lon,
                         ln_pressure=np.full(n, lnp), surface=np.zeros(n, dtype=bool), value=np.full(n, 280.0),
                         error=np.full(n, 0.5), identity=np.array([f"toy:{stamp}:{i}" for i in range(n)], dtype=object),
                         valid_time=[context.window_end - dt.timedelta(seconds=5)] * n, operator=self.operator)
        truth = self.operator([context.control_state], probe)[0] + rng.normal(0.0, 0.3, n) + 0.8
        batch = PointObs(stream=self.name, variable="brightness_temperature_k", latitude_deg=lat, longitude_deg=lon,
                         ln_pressure=np.full(n, lnp), surface=np.zeros(n, dtype=bool), value=truth,
                         error=np.full(n, 0.5), identity=probe.identity, valid_time=list(probe.valid_time),
                         localisation_profile=np.broadcast_to(prof, (n, prof.size)).copy(), operator=self.operator,
                         vertical_cutoff_lnp=0.6)
        receipt = {"stream": self.name, "rows": n, "window_end_utc": da_streams._stamp(context.window_end)}
        self.receipts.append(receipt)
        return [batch], receipt


@pytest.fixture
def toy_stream(monkeypatch):
    stream = _ToyRadianceStream()
    monkeypatch.setitem(da_streams.STREAM_TABLE, "toy-radiance", lambda options: stream)
    return stream


def test_a_radiance_stream_runs_through_the_letkf_door_and_the_report_carries_it(spun_up, tmp_path, toy_stream):
    cfg, checkpoint = spun_up
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    out = tmp_path / "letkf-radiance"
    da_door.init(cfg, out, filter_name="letkf", members=3, ensemble_truncation=3,
                 analysis_time_utc=START_TEXT, options=OPTIONS)
    receipt = da_door.cycle(
        cfg, out, stream_specs=["local-tables:paths=" + ",".join(str(p) for p in obs), "toy-radiance"],
        cycles=2, start_utc=START_TEXT, interval_s=20.0, ensemble=out / ENSEMBLE_MANIFEST_NAME,
        options=OPTIONS, observation_bin_s=10.0,
    )
    assert receipt["cycles"]["applied"] == 2 and receipt["status"] == "pass"
    assert [s["name"] for s in receipt["streams"]] == ["local-tables", "toy-radiance"]
    assert len(toy_stream.receipts) == 2
    report = json.loads((out / "assimilation-report-step00000002.json").read_text())
    streams = report["ensemble_report"]["streams"]
    assert "toy-radiance" in streams
    row = streams["toy-radiance"]["brightness_temperature_k"]
    rejected = report["ensemble_report"]["rejections"].get("toy-radiance", {})
    assert row["count"] + row["withheld_count"] + sum(rejected.values()) == 24
    assert set(rejected) <= {"thinned", "background_check"}
    assert row["desroziers"]["count"] == row["count"] and row["regions"]["global"]["assimilated"]["o_minus_a"] is not None
    assert row["regions"]["global"]["control"]["assimilated"]["o_minus_b"]["count"] == row["count"]
    # The window record carries the stream's receipt and the row counts.
    times = report["observation_times"]
    assert times["radiance_rows"] == 24 and times["point_rows"] > 0
    assert times["radiance"]["toy-radiance"]["rows"] == 24
    # The localisation record names the profile rows.
    prof = report["localisation"]["profile_rows"]["toy-radiance/brightness_temperature_k"]
    assert prof["rows"] == row["count"] and abs(prof["peak_pressure_pa"] - 95_000.0) < 3_000.0
    # The door's card carries the stream beside the point streams.
    assert "toy-radiance" in report["scorecard"]["streams"]
    assert "toy-radiance" in report["variables"]
    second = json.loads((out / "assimilation-report-step00000004.json").read_text())
    assert second["status"] == "pass" and second["rejections"].get("already_assimilated", 0) >= 0
    assert "toy-radiance" in second["ensemble_report"]["streams"]
    # The chain names the stream.
    assert "toy-radiance" in report["lineage"]["streams"]


def test_the_successive_correction_filter_refuses_a_radiance_stream_by_name(spun_up, tmp_path, toy_stream):
    cfg, checkpoint = spun_up
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    out = tmp_path / "sc-radiance"
    da_door.init(cfg, out, analysis_time_utc=START_TEXT, options=OPTIONS)
    with pytest.raises(ValueError, match="no ensemble and no radiance operator"):
        da_door.cycle(
            cfg, out, stream_specs=["local-tables:paths=" + ",".join(str(p) for p in obs), "toy-radiance"],
            cycles=1, start_utc=START_TEXT, interval_s=20.0, ensemble=out / ENSEMBLE_MANIFEST_NAME, options=OPTIONS,
        )


def test_the_bias_ledger_moves_by_the_memory_rule_and_reads_back(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = BiasLedger.load("atms:noaa-21", path, {"7": {"a": 0.2, "b": 0.01, "c": 0.0, "mean_background_k": 241.0}},
                             memory_rows=1000)
    record = ledger.move("7", {"a": 1.0, "b": 0.0, "c": 0.5}, 1000, label="atms-clear-ocean:noaa-21:ch07",
                         before={"count": 1000, "mean": 1.0, "rms": 1.2}, after={"count": 1000, "mean": 0.0, "rms": 0.6})
    assert record["weight"] == pytest.approx(0.5)
    # The scan term moves by the memory rule; the constant is anchored to the
    # entry's fit (its window fit is recorded, not applied).
    assert ledger.coefficients["7"]["a"] == pytest.approx(0.2) and ledger.coefficients["7"]["c"] == pytest.approx(0.25)
    assert record["anchored"] == ["a"] and record["applied"] == {"b": pytest.approx(0.0), "c": pytest.approx(0.25)}
    assert record["fit"]["a"] == 1.0
    assert ledger.coefficients["7"]["mean_background_k"] == 241.0
    ledger.write()
    again = BiasLedger.load("atms:noaa-21", path, {"7": {"a": 0.2, "b": 0.01, "c": 0.0, "mean_background_k": 241.0}},
                            memory_rows=1000)
    assert again.coefficients["7"]["a"] == pytest.approx(0.2) and again.coefficients["7"]["c"] == pytest.approx(0.25)
    assert len(again.history) == 1
    abi = BiasLedger.load("goes-abi:G19", tmp_path / "abi.json", {"13": {"intercept_shift_k": 0.0}}, memory_rows=1000)
    abi_record = abi.move("13", {"intercept_shift_k": 0.4}, 3000, label="goes-abi-l1b-bt:G19:band13",
                          before={"count": 3000, "mean": 0.4, "rms": 0.5}, after={"count": 3000, "mean": 0.0, "rms": 0.3})
    assert abi.coefficients["13"]["intercept_shift_k"] == 0.0 and abi_record["anchored"] == ["intercept_shift_k"]
    # Another stream's ledger at the same path is not read.
    other = BiasLedger.load("goes-abi:G19", path, {"13": {"intercept_shift_k": 0.0}}, memory_rows=1000)
    assert other.coefficients["13"]["intercept_shift_k"] == 0.0 and other.history == []


def test_the_stream_table_carries_the_radiance_streams_with_their_options():
    atms = da_streams.resolve_stream("atms:satellites=noaa-21;thin_deg=0.25;channels=7,9;day_update=off")
    assert atms.satellites == ("noaa-21",) and atms.thin_deg == 0.25 and atms.channels == (7, 9)
    assert atms.day_update is False and atms.entry("noaa-21")[0].name.startswith("atms-clear-ocean:noaa-21")
    with pytest.raises(FileNotFoundError, match="no measured ATMS operator entry"):
        da_streams.resolve_stream("atms:satellites=noaa-99").entry("noaa-99")
    abi = da_streams.resolve_stream("goes-abi:satellites=G19;bands=13")
    assert abi.bands == (13,) and abi.entries()["bands"]["13"]["admitted_classes"] == ["water"]
    # the day update's memory is one number per stream, overridable in the spelling
    assert atms.bias_memory_rows == 2000 and abi.bias_memory_rows == 2000
    assert da_streams.resolve_stream("goes-abi:bias_memory_rows=5000").bias_memory_rows == 5000
    assert abi.table()["vertical"]["nlev"] == 40
