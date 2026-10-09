"""The microwave leg's scoring pass and its operator entry.

Held here: the two-stage scoring pass (surface under every candidate,
columns under the survivors) counts every screen stage and produces the
same columns as sampling the survivors directly; the predictor family
separates a wind-shaped residual (caught by the wind model, not by the
geometry model, both directions); the noise floor reads the planted
per-beam noise; the channel verticals climb with channel number; an
entry is promoted from a scorecard only for admitted channels and reads
back from its file; the member operator on a three-member CPU ensemble is
deterministic (identical members read bitwise the same), reads a scaled
temperature back in an opaque channel, applies the bias model it was
handed, and fills a PointObs batch built from cells.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import datetime as dt
import json
from types import SimpleNamespace

import numpy as np
import pytest

from woof.globe.microwave import entry as entry_module
from woof.globe.microwave.calibrate import STANDARD_LEVELS_PA, standard_column
from woof.globe.microwave.channels import TEMPERATURE_SOUNDING_CHANNELS
from woof.globe.microwave.columns import SURFACE_FIELDS, Analysis, sample_columns
from woof.globe.microwave.rte import Column
from woof.globe.microwave.entry import (
    AtmsBatchOperator,
    ChannelEntry,
    OperatorEntry,
    channel_error_k,
    channel_vertical,
    entry_from_scorecard,
    make_identity,
    member_columns,
    point_obs_from_cells,
    read_entry,
    write_entry,
)
from woof.globe.microwave.score import (
    ScreenOptions,
    candidate_mask,
    score_channels,
    score_day,
    write_scorecard,
)

T0 = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)


def _analysis(when, rng, *, ny=46, nx=90):
    lat = 90.0 - 4.0 * np.arange(ny)
    lon = 4.0 * np.arange(nx)
    col = standard_column()
    t = np.broadcast_to(col.temperature_k[:, :, None], (41, ny, nx)).copy()
    t += rng.normal(0.0, 1.0, (41, ny, nx))
    q = np.broadcast_to(col.specific_humidity[:, :, None], (41, ny, nx)).copy()
    surface = {name: np.zeros((ny, nx)) for name in SURFACE_FIELDS}
    surface["surface_pressure"][:] = 101000.0
    surface["skin_temperature"][:] = 300.0
    surface["air_temperature_2m"][:] = 299.0
    surface["eastward_wind_10m"][:] = rng.uniform(0.0, 12.0, (ny, nx))
    surface["precipitable_water"][:] = 30.0
    surface["land_fraction"][:, : nx // 2] = 1.0
    return Analysis(
        valid_time=when, latitude=lat, longitude=lon, pressure_pa=STANDARD_LEVELS_PA,
        temperature_k=t, specific_humidity=q, surface=surface, provenance={"grib": "synthetic"},
    )


def _cells(rng, n=800):
    cells = SimpleNamespace(
        ncell=n,
        lat_mean_deg=rng.uniform(-70.0, 70.0, n).astype(np.float32),
        lon_mean_deg=rng.uniform(0.0, 360.0, n).astype(np.float32),
        zenith_mean_deg=rng.uniform(0.0, 65.0, n).astype(np.float32),
        scan_angle_abs_mean_deg=rng.uniform(0.0, 50.0, n).astype(np.float32),
        time_mean_unix_s=T0.timestamp() + rng.uniform(0.0, 6 * 3600.0, n),
        count=rng.integers(1, 8, n).astype(np.int32),
        tb_mean_k=np.full((n, 22), 220.0, dtype=np.float32),
        tb_std_k=np.full((n, 22), 0.3, dtype=np.float32),
        tb_count=np.full((n, 22), 4, dtype=np.int32),
        cell_bin=np.zeros(n, dtype=np.int32),
        cell_j=np.arange(n, dtype=np.int32),
        cell_i=np.zeros(n, dtype=np.int32),
    )
    cells.tb_mean_k[:, 0] = 200.0
    cells.tb_mean_k[:, 1] = 170.0
    return cells


@pytest.fixture(scope="module")
def synthetic_day():
    rng = np.random.default_rng(11)
    analyses = [_analysis(T0, rng), _analysis(T0 + dt.timedelta(hours=6), rng)]
    cells = _cells(rng)
    day = score_day(cells, analyses, options=ScreenOptions(), bar_k=1.0, channels=(4, 5, 7, 9))
    return analyses, cells, day


def test_score_day_counts_every_stage_and_scores_only_survivors(synthetic_day):
    analyses, cells, day = synthetic_day
    stages = day.screen.stages
    assert stages["cells_total"] == cells.ncell
    assert stages["cells_candidate"] == int(candidate_mask(cells, ScreenOptions()).sum())
    assert stages["cells_candidate"] < stages["cells_total"]
    assert stages["ocean"] < stages["cells_candidate"]
    assert stages["finite"] == day.index.size == day.background.shape[0]
    assert np.all(np.isfinite(day.background))
    assert day.background.shape == (day.index.size, 4)
    assert sum(day.cells_by_hour.values()) == day.index.size


def test_two_stage_columns_equal_direct_sampling_of_the_survivors(synthetic_day):
    analyses, cells, day = synthetic_day
    direct = sample_columns(
        analyses, cells.lat_mean_deg[day.index].astype(np.float64),
        cells.lon_mean_deg[day.index].astype(np.float64), cells.time_mean_unix_s[day.index],
    )
    assert np.array_equal(direct.column.temperature_k, day.column.temperature_k)
    assert np.array_equal(direct.column.specific_humidity, day.column.specific_humidity)
    assert np.array_equal(direct.surface["skin_temperature"], day.column.skin_temperature_k)


def test_max_cells_subsample_is_recorded(synthetic_day):
    analyses, cells, _ = synthetic_day
    day = score_day(cells, analyses, options=ScreenOptions(), max_cells=50, channels=(7,))
    assert day.screen.stages["cells_subsampled"] == 50
    assert day.index.size <= 50


def _scores(observed, background, wind, n, **kwargs):
    rng = np.random.default_rng(5)
    return score_channels(
        observed, background,
        zenith_deg=rng.uniform(0.0, 55.0, n), latitude_deg=rng.uniform(-59.0, 59.0, n),
        wind_speed_m_s=wind, precipitable_water_kg_m2=rng.uniform(5.0, 50.0, n),
        time_unix_s=np.arange(n, dtype=float), channels=(5,), options=ScreenOptions(), bar_k=1.0,
        **kwargs,
    )


def test_wind_shaped_residual_is_caught_by_the_wind_model_only():
    rng = np.random.default_rng(3)
    n = 2000
    background = 230.0 + 10.0 * rng.random((n, 1))
    wind = rng.uniform(0.0, 15.0, n)
    observed = background + 0.25 * wind[:, None] + 0.02 * rng.standard_normal((n, 1))
    (s,) = _scores(observed, background, wind, n, cell_std_k=np.zeros((n, 1)))
    assert s.geometry_corrected["rmse"] > 1.0
    assert s.wind_corrected["rmse"] < 0.1
    assert s.wind_coefficients["d"] == pytest.approx(0.25, abs=0.01)
    assert not s.within_bar and s.within_bar_with_wind


def test_geometry_shaped_residual_is_removed_by_the_geometry_model():
    rng = np.random.default_rng(4)
    n = 2000
    background = 230.0 + 10.0 * rng.random((n, 1))
    zenith = rng.uniform(0.0, 55.0, n)
    sec = 1.0 / np.cos(np.deg2rad(zenith))
    observed = background + 2.0 + 0.5 * (sec - 1.0)[:, None] + 0.02 * rng.standard_normal((n, 1))
    s = score_channels(
        observed, background, zenith_deg=zenith, latitude_deg=np.zeros(n),
        wind_speed_m_s=rng.uniform(0.0, 12.0, n), precipitable_water_kg_m2=np.full(n, 30.0),
        time_unix_s=np.arange(n, dtype=float), cell_std_k=np.zeros((n, 1)),
        channels=(5,), options=ScreenOptions(), bar_k=1.0,
    )[0]
    assert s.raw["rmse"] > 1.5
    assert s.geometry_corrected["rmse"] < 0.1
    assert s.geometry_coefficients["c"] == pytest.approx(0.5, abs=0.05)
    assert s.within_bar


def test_noise_floor_reads_the_planted_beam_noise():
    rng = np.random.default_rng(6)
    n = 4000
    background = np.full((n, 1), 230.0)
    sigma_beam, beams = 0.8, 4
    observed = background + rng.normal(0.0, sigma_beam / np.sqrt(beams), (n, 1))
    (s,) = _scores(observed, background, np.zeros(n), n,
                   cell_std_k=np.full((n, 1), sigma_beam), beam_count=np.full((n, 1), beams))
    assert s.beam_noise_k == pytest.approx(sigma_beam)
    assert s.noise_floor_k == pytest.approx(sigma_beam / np.sqrt(beams))
    assert s.geometry_corrected["rmse"] == pytest.approx(s.noise_floor_k, rel=0.1)
    assert s.rmse_above_noise_k < 0.15


def test_channel_verticals_climb_with_channel_number():
    peaks = [channel_vertical(c)[0] for c in TEMPERATURE_SOUNDING_CHANNELS]
    cutoffs = [channel_vertical(c)[1] for c in TEMPERATURE_SOUNDING_CHANNELS]
    assert all(p2 < p1 for p1, p2 in zip(peaks[:-1], peaks[1:])), peaks
    assert peaks[0] > 50_000.0 and peaks[-1] < 1_000.0
    assert all(0.05 <= c < 5.0 for c in cutoffs), cutoffs


def test_channel_error_is_the_cell_residual_with_the_cells_own_noise():
    # the rule the 2026-09-06 entries were built with, kept only where no noise floor is known
    assert channel_error_k(0.5, 0.0) == 0.5
    assert channel_error_k(0.5, 0.5) == pytest.approx(np.sqrt(0.5))
    assert channel_error_k(0.0, 0.3) == pytest.approx(0.3)
    # with the scorecard's noise floor: the residual above the noise, with the cell noise put back
    assert channel_error_k(0.5, 0.195, 0.107) == pytest.approx(0.195)          # the scored residual itself
    assert channel_error_k(0.5, 0.05, 0.107) == pytest.approx(0.107)           # never under the noise floor
    # per cell, the cell's own noise NEdT / sqrt(beams): three beams read more than twenty
    per_cell = channel_error_k(0.5, 0.195, 0.107, beams=np.array([3.0, 22.0]))
    above = np.sqrt(0.195 ** 2 - 0.107 ** 2)
    assert per_cell == pytest.approx(np.sqrt(above ** 2 + np.array([0.5 / np.sqrt(3.0), 0.5 / np.sqrt(22.0)]) ** 2))
    assert per_cell[0] > per_cell[1] > 0.16
    # the single-beam NEdT never enters a cell mean's error whole
    assert channel_error_k(0.5, 0.195, 0.107) < 0.5 and per_cell.max() < 0.5


def _document(day, analyses, bar_k=1.0, tmp_path=None):
    provenance = {
        "thinned": {"decoded_dir": "/x/atms-decoded/noaa-21-20260901"},
        "analyses": [{"grib": "synthetic", "valid_time": a.valid_time.isoformat()} for a in analyses],
        "analysis_span": [analyses[0].valid_time.isoformat(), analyses[-1].valid_time.isoformat()],
    }
    path = tmp_path / "microwave-scorecard.json"
    return write_scorecard(path, scores=day.scores, screen=day.screen, options=ScreenOptions(),
                           provenance=provenance, bar_k=bar_k)


def test_entry_is_promoted_only_for_admitted_channels(synthetic_day, tmp_path):
    analyses, cells, day = synthetic_day
    document = _document(day, analyses, tmp_path=tmp_path)
    admitted = document["verdict"]["admitted_channels"]
    entry = entry_from_scorecard(document, {"passes": {"isothermal": True}})
    if not admitted:
        assert entry is None
        return
    assert entry.admitted_channels == admitted
    assert entry.satellite == "noaa-21" and entry.day == "2026-09-01"
    assert entry.name == "atms-clear-ocean:noaa-21:2026-09-01"
    for channel in entry.channels:
        assert channel.channel in admitted
        assert channel.error_k >= channel.noise_floor_k and channel.error_k == pytest.approx(
            max(channel.rmse_after_k, channel.noise_floor_k), abs=1e-9)
        assert set(channel.bias_coefficients) == {"a", "b", "c", "mean_background_k"}
        assert channel.vertical_cutoff_lnp > 0.0
    assert entry.contract.assimilation_status.startswith("assimilated by the ensemble filter")


def test_entry_refused_when_nothing_is_admitted(tmp_path):
    document = {"verdict": {"admitted_channels": [], "nearest_term": {"15": "noise"}}, "channels": [],
                "provenance": {}}
    assert entry_from_scorecard(document, {}) is None
    write_entry(tmp_path / "entry.json", None)
    payload = json.loads((tmp_path / "entry.json").read_text(encoding="utf-8"))
    assert payload["operator_entry_ships"] is False
    assert read_entry(tmp_path / "entry.json") is None


def _toy_entry(channels=(5, 7), bias=None):
    bias = bias or {"a": 0.0, "b": 0.0, "c": 0.0, "mean_background_k": 230.0}
    entries = []
    for number in channels:
        peak, cutoff = channel_vertical(number)
        entries.append(ChannelEntry(
            channel=number, centre_ghz=54.0, nedt_k=0.5, error_k=0.6, bias_coefficients=dict(bias),
            peak_pressure_pa=peak, vertical_cutoff_lnp=cutoff, rmse_after_k=0.3, noise_floor_k=0.2,
            score_cells=100,
        ))
    return OperatorEntry(
        name="atms-clear-ocean:noaa-21:2026-09-01", satellite="noaa-21", day="2026-09-01",
        admitted_channels=list(channels), channels=entries, refused={"15": "noise"},
        calibration_passes={"isothermal": True}, analyses=[], fitted_utc="2026-09-06T00:00:00Z",
    )


def test_entry_round_trips_through_its_file(tmp_path):
    entry = _toy_entry()
    write_entry(tmp_path / "entry.json", entry)
    back = read_entry(tmp_path / "entry.json")
    assert back.admitted_channels == [5, 7]
    assert back.channel_entry(7).peak_pressure_pa == pytest.approx(entry.channel_entry(7).peak_pressure_pa)
    assert back.contract == entry.contract
    assert back.levels_pa == [float(p) for p in STANDARD_LEVELS_PA]
    with pytest.raises(KeyError):
        back.channel_entry(15)


# ---------------------------------------------------------------- members

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


@pytest.fixture(scope="module")
def members():
    from woof.globe.config import load_config
    from woof.globe.da import EnsembleOptions, GlobalEnsemble, ensemble_config
    from woof.globe.runner import build_model_and_cold_state, build_transform

    cfg = load_config(CONFIG)
    options = EnsembleOptions(members=3, truncation=cfg.truncation, seed=3, additive_inflation_fraction=0.0)
    ecfg = ensemble_config(cfg, options)
    transform = build_transform(ecfg)
    model, cold = build_model_and_cold_state(ecfg, transform)
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    return transform, model, ensemble.members


def _ocean_points():
    return np.array([-30.0, 0.0, 20.0]), np.array([200.0, 180.0, 330.0])


def test_member_columns_are_on_the_operator_levels_and_finite(members):
    transform, model, states = members
    lat, lon = _ocean_points()
    columns = member_columns(states, transform, model.vertical, lat, lon)
    assert len(columns) == len(states)
    for column in columns:
        assert np.array_equal(column.pressure_pa, STANDARD_LEVELS_PA)
        assert column.temperature_k.shape == (STANDARD_LEVELS_PA.size, 3)
        assert np.all(np.isfinite(column.temperature_k)) and np.all(column.temperature_k > 150.0)
        assert np.all(column.specific_humidity >= 0.0)
        assert np.all(column.surface_pressure_pa > 50_000.0)
        assert np.all(np.isfinite(column.skin_temperature_k))


def test_identical_members_read_bitwise_the_same_and_a_scaled_member_reads_warmer(members):
    transform, model, states = members
    lat, lon = _ocean_points()
    entry = _toy_entry()
    n = lat.size
    identities = [make_identity("noaa-21", 0, i, 0, 7) for i in range(n)]
    operator = AtmsBatchOperator(
        entry=entry, vertical=model.vertical,
        geometry={i: (7, 20.0, 15.0) for i in identities},
    ).bind(transform)
    batch = SimpleNamespace(identity=np.array(identities, dtype=object), latitude_deg=lat,
                            longitude_deg=lon, count=n)
    twin = [states[0], states[0]]
    same = operator(twin, batch)
    assert same.shape == (2, n)
    assert np.array_equal(same[0], same[1])
    assert np.all((same > 180.0) & (same < 300.0))

    # Direction one: theta scaled by 1 percent everywhere reads back as a
    # warmer channel 7 by close to 1 percent of its brightness temperature.
    import copy

    warm = copy.copy(states[0])
    atmosphere = copy.copy(states[0].atmosphere)
    atmosphere.theta = states[0].atmosphere.theta * 1.01
    warm.atmosphere = atmosphere
    warmer = operator([warm], batch)[0]
    expected = 0.01 * same[0]
    assert np.all(warmer - same[0] > 0.7 * expected)
    assert np.all(warmer - same[0] < 1.3 * expected)


def test_operator_applies_the_entry_bias_model(members):
    transform, model, states = members
    lat, lon = _ocean_points()
    n = lat.size
    identities = [make_identity("noaa-21", 0, i, 0, 5) for i in range(n)]
    geometry = {i: (5, 45.0, 30.0) for i in identities}
    batch = SimpleNamespace(identity=np.array(identities, dtype=object), latitude_deg=lat,
                            longitude_deg=lon, count=n)
    plain = AtmsBatchOperator(entry=_toy_entry(), vertical=model.vertical,
                              geometry=geometry).bind(transform)([states[0]], batch)[0]
    biased = AtmsBatchOperator(
        entry=_toy_entry(bias={"a": 1.0, "b": 0.0, "c": 2.0, "mean_background_k": 230.0}),
        vertical=model.vertical, geometry=geometry,
    ).bind(transform)([states[0]], batch)[0]
    sec = 1.0 / np.cos(np.deg2rad(45.0))
    assert biased - plain == pytest.approx(1.0 + 2.0 * (sec - 1.0), abs=1e-9)


def test_point_obs_from_cells_builds_one_batch_per_admitted_channel(members):
    transform, model, states = members
    rng = np.random.default_rng(2)
    cells = _cells(rng, n=5)
    cells.lat_mean_deg[:] = np.array([-30.0, 0.0, 20.0, -10.0, 5.0], dtype=np.float32)
    cells.lon_mean_deg[:] = np.array([200.0, 180.0, 330.0, 250.0, 190.0], dtype=np.float32)
    cells.tb_mean_k[2, 4] = np.nan
    entry = _toy_entry()
    batches = point_obs_from_cells(entry, cells, transforms=[transform], vertical=model.vertical)
    assert [b.count for b in batches] == [4, 5]
    for batch in batches:
        assert batch.stream.startswith("atms-clear-ocean:noaa-21:ch") and batch.variable == "brightness_temperature_k"
        assert batch.localisation_profile.shape == (batch.count, 188)
        # the row's error is the entry's residual above the noise with this cell's own noise (its beam count)
        channel = entry.channel_entry(int(batch.identity[0][-2:]))
        finite = np.isfinite(cells.tb_mean_k[:, channel.channel - 1])
        expected = channel_error_k(channel.nedt_k, channel.rmse_after_k, channel.noise_floor_k, beams=cells.count[finite])
        assert batch.error == pytest.approx(expected) and np.all(batch.error < channel.nedt_k)
        assert batch.vertical_cutoff_lnp > 0.0
        assert not batch.surface.any()
        simulated = batch.operator(states, batch)
        assert simulated.shape == (3, batch.count)
        assert np.all(np.isfinite(simulated))
        sub = batch.subset(np.array([0, 2]))
        assert np.allclose(batch.operator(states, sub), simulated[:, [0, 2]], rtol=0.0, atol=1e-9)


def test_entry_module_names_its_contract_fields():
    fields = set(entry_module.AcceptanceContract.__dataclass_fields__)
    assert fields == {"measurement", "time", "location", "vertical_coordinate", "representativeness",
                      "bias_treatment", "error_correlations", "assimilation_status"}


def test_surface_slab_below_the_lowest_level_is_integrated_and_batch_independent():
    """Over the ocean the surface sits below the 1000 hPa analysis level; the
    slab down to it carries emission the surface-seeing channels read, and
    the layer set must not depend on which columns share a batch."""
    from woof.globe.microwave.rte import brightness_temperature

    col = standard_column()

    def at(ps):
        return Column(
            pressure_pa=col.pressure_pa, temperature_k=col.temperature_k,
            specific_humidity=col.specific_humidity, surface_pressure_pa=np.array([ps]),
            skin_temperature_k=col.skin_temperature_k, air_temperature_2m_k=None,
        )

    channels = (1, 4, 5, 7, 9)
    flat = brightness_temperature(at(100000.0), channels, 30.0, 20.0)[:, 0]
    deep = brightness_temperature(at(101000.0), channels, 30.0, 20.0)[:, 0]
    assert deep[1] - flat[1] > 1.0          # channel 4 sees the 10 hPa slab
    assert deep[0] - flat[0] > 1.0          # channel 1 too
    assert abs(deep[3] - flat[3]) < 0.02    # channel 7 does not
    assert abs(deep[4] - flat[4]) < 1e-3    # channel 9 does not
    both = Column(
        pressure_pa=col.pressure_pa,
        temperature_k=np.repeat(col.temperature_k, 2, axis=1),
        specific_humidity=np.repeat(col.specific_humidity, 2, axis=1),
        surface_pressure_pa=np.array([101000.0, 100000.0]),
        skin_temperature_k=np.repeat(col.skin_temperature_k, 2),
        air_temperature_2m_k=None,
    )
    batch = brightness_temperature(both, channels, 30.0, 20.0)
    # The layer set is the same in any company; numpy's reduction order over
    # a different trailing dimension moves the last bit, nothing more.
    assert np.allclose(batch[:, 0], deep, rtol=0.0, atol=1e-9)
    assert np.allclose(batch[:, 1], flat, rtol=0.0, atol=1e-9)
