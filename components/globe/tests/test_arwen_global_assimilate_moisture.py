"""The moisture update of the assimilation door: dewpoint reports into
specific humidity (assimilate.MOISTURE_UPDATE_DIVERGENCE).

Calibrated both ways on synthetic reports against the spun-up smoke
background: a planted dry bias is removed at the reports and its increment
decays with distance and height as the localization says; a report set
that agrees with the background moves no vapor at all; a planted
saturation excess is capped and a planted drying is floored; the update
switched off leaves the vapor as the v1 door did; the IEM ASOS record the
scorecard scores against decodes through its table entry.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pytest

from woof.globe.assimilate import (
    DEFAULT_HUMIDITY_DECAY_HEIGHT_M,
    LOCALIZATION_SCALE_HEIGHT_M,
    AssimilationOptions,
    _Family,
    _spread_column,
    _to_numpy_spectral,
    assimilate,
    bounded_vapor_increment,
    specific_humidity_from_dewpoint,
)
from woof.globe.checkpoint import read_checkpoint, state_from_checkpoint
from woof.globe.cli import main as cli_main
from woof.globe.config import load_config
from woof.globe.obs_table import (
    INHG_TO_PA, ObsRow, VARIABLE_TABLE, decode_obs_csv,
)
from woof.globe.runner import build_transform
from woof.globe.surface_energy import dewpoint_from_specific_humidity

from test_arwen_global_assimilate import (  # noqa: F401 - the fixture rides the import
    CONFIG, _forty_level_columns, _model_space, _rotational_wind_signal,
    _wind_to_dir_speed, spun_up,
)

IEM_HEADER = "station,valid,lon,lat,elevation,tmpf,dwpf,drct,sknt,gust,alti,mslp,p01i"
TIME = "2026-08-31 12:00"


def _kelvin_to_f(k: float) -> float:
    return (k - 273.15) * 9.0 / 5.0 + 32.0


def _probe(variable, lats, lons, level_pa=None, elev=0.0) -> _Family:
    return _Family([
        ObsRow(
            source="probe", station_id=f"P{k}", latitude_deg=lats[k],
            longitude_deg=lons[k], elevation_m=elev, level_pa=level_pa,
            valid_time=dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc),
            variable=variable, value=0.0, error=1.0,
        )
        for k in range(len(lats))
    ])


def _iem_reports(cfg, checkpoint: Path, path: Path, *, dewpoint_offset_k,
                 others_signal: bool = True) -> Path:
    """An IEM ASOS CSV at 64 stations whose dewpoint is the background's
    plus ``dewpoint_offset_k`` (a callable of the station index or a
    number).  With ``others_signal`` the pressure, temperature and wind
    carry the smooth wavenumber-one innovation of the door's own
    end-to-end test (so their gates have something to judge); without it
    they are exactly the background's (to the CSV's Fahrenheit rounding)."""
    space, state = _model_space(cfg, checkpoint)
    lats = np.repeat(np.linspace(-60.0, 60.0, 8), 8)
    lons = np.tile(np.arange(0.0, 360.0, 45.0), 8)
    pattern = np.sin(np.deg2rad(lons)) * np.cos(np.deg2rad(lats))
    ps = space.hx_surface_pressure(state.atmosphere, _probe("surface_pressure_pa", lats, lons))
    t = space.hx_temperature(state.atmosphere, _probe("temperature_k", lats, lons))
    td = space.hx_dewpoint(state.atmosphere, _probe("dewpoint_k", lats, lons))
    u = space.hx_wind(state.atmosphere, _probe("wind_u_m_s", lats, lons), "u")
    v = space.hx_wind(state.atmosphere, _probe("wind_v_m_s", lats, lons), "v")
    if others_signal:
        du, dv = _rotational_wind_signal(lats, lons, 3.0)
        ps = ps + 500.0 * pattern
        t = t + 2.0 * pattern
        u = u + du
        v = v + dv
    lines = [IEM_HEADER]
    for k in range(lats.size):
        offset = dewpoint_offset_k(k) if callable(dewpoint_offset_k) else dewpoint_offset_k
        direction, speed_kt = _wind_to_dir_speed(u[k], v[k])
        lon = lons[k] if lons[k] <= 180.0 else lons[k] - 360.0
        lines.append(
            f"S{k:03d},{TIME},{lon:.4f},{lats[k]:.4f},0.00,"
            f"{_kelvin_to_f(t[k]):.6f},{_kelvin_to_f(td[k] + offset):.6f},"
            f"{direction:.3f},{speed_kt:.5f},,{ps[k] / INHG_TO_PA:.7f},,0.00"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _grid_vapor(cfg, checkpoint: Path):
    transform = build_transform(cfg)
    metadata, arrays = read_checkpoint(checkpoint)
    state = state_from_checkpoint(metadata, arrays, transform.backend)
    return np.asarray(transform.backend.to_numpy(state.atmosphere.qv)), state, transform


def test_iem_asos_record_decodes_through_its_table_entry():
    text = (
        IEM_HEADER + "\n"
        "APF,2026-09-01 23:00,-81.7753,26.1525,3.00,77.00,75.00,60.00,9.00,,29.99,,0.02\n"
        "ASH,2026-09-01 23:03,-71.5148,42.7818,61.00,62.60,,150.00,4.00,,30.09,,0.00\n"
    )
    source, rows, counters = decode_obs_csv(text)
    assert source == "iem-asos-csv"
    by = {(r.station_id, r.variable): r for r in rows}
    assert by[("APF", "temperature_k")].value == pytest.approx(298.15)
    assert by[("APF", "dewpoint_k")].value == pytest.approx(297.0389, abs=1e-3)
    assert by[("APF", "dewpoint_k")].error == 1.5
    assert by[("APF", "surface_pressure_pa")].value == pytest.approx(101521.7, abs=0.5)
    assert ("ASH", "dewpoint_k") not in by
    assert counters["values_not_derivable"] == 1
    assert by[("ASH", "temperature_k")].valid_time == dt.datetime(2026, 9, 1, 23, 3, tzinfo=dt.timezone.utc)
    assert "dewpoint_k" in VARIABLE_TABLE
    assert dict(AssimilationOptions().background_errors)["dewpoint_k"] == 3.0


def test_bolton_round_trip_is_the_identity_and_saturation_is_the_temperature():
    q = np.array([1.0e-6, 1.0e-4, 2.0e-3, 1.0e-2, 2.5e-2])
    p = np.array([2.0e4, 5.0e4, 8.0e4, 9.5e4, 1.01e5])
    td = dewpoint_from_specific_humidity(q, p)
    assert np.allclose(specific_humidity_from_dewpoint(td, p), q, rtol=1e-12, atol=0.0)
    t = np.array([300.0, 273.15, 250.0])
    qs = specific_humidity_from_dewpoint(t, np.array([1.0e5, 1.0e5, 1.0e5]))
    # The dewpoint of saturated air is the temperature itself.
    assert np.allclose(dewpoint_from_specific_humidity(qs, 1.0e5), t, atol=1e-9)
    # A colder dewpoint is a drier air: monotone.
    assert np.all(np.diff(specific_humidity_from_dewpoint(np.linspace(240.0, 305.0, 30), 1.0e5)) > 0.0)


def test_surface_dewpoint_increment_decays_as_the_humidity_localization_says():
    """One 2 m dewpoint report on the forty-level stack: horizontally the
    Gaussian of the length scale, vertically exp(-z / humidity decay
    height) with z the ln p separation on the 7.6 km scale height; the
    temperature's own 2 km height is not what the moisture update uses."""
    options = AssimilationOptions(length_scale_km=500.0)
    moisture = dataclasses.replace(
        options, surface_decay_height_m=options.humidity_decay_height_m
    )
    assert moisture.surface_decay_height_m == DEFAULT_HUMIDITY_DECAY_HEIGHT_M == 1500.0
    p_full, p_full_columns, ps_columns = _forty_level_columns(4)
    family = _Family([ObsRow(
        source="iem-asos-csv", station_id="S0", latitude_deg=0.0, longitude_deg=0.0,
        elevation_m=0.0, level_pa=None,
        valid_time=dt.datetime(2026, 8, 31, 12, tzinfo=dt.timezone.utc),
        variable="dewpoint_k", value=0.0, error=1.5,
    )])
    gains = np.array([(3.0 / 1.5) ** 2])
    distances_km = np.array([0.0, 500.0, 1000.0, 1500.0])
    lon_rad = distances_km * 1000.0 / 6_371_220.0
    out = _spread_column(
        family, np.array([3.0]), gains, np.zeros(4), lon_rad, moisture,
        ps_columns=ps_columns, p_full_columns=p_full_columns,
    )
    assert out.shape == (40, 4)
    # Every level and column: the single-report successive correction
    # g h w d / (1 + g h w) with g = 4 the error ratio squared, h the
    # Gaussian of the great-circle distance, w = exp(-z / 1500 m) with z
    # the ln p height of the level above the surface on the 7.6 km scale
    # height (the lowest full level sits 23 m up, so even the report's own
    # column reads slightly under the OI limit 3 g / (1 + g) = 2.4 K).
    z = LOCALIZATION_SCALE_HEIGHT_M * (np.log(ps_columns[0]) - np.log(p_full))
    w = np.exp(-z / 1500.0)[:, None]
    h = np.exp(-0.5 * (distances_km / 500.0) ** 2)[None, :]
    expected = 3.0 * 4.0 * h * w / (1.0 + 4.0 * h * w)
    assert np.allclose(out, expected, rtol=1e-9, atol=1e-12)
    assert out[-1, 0] == pytest.approx(2.4, abs=0.01)
    assert out[-1, 0] < 2.4
    # The same report through the temperature's 2 km height reaches higher.
    out_t = _spread_column(
        family, np.array([3.0]), gains, np.zeros(4), lon_rad, options,
        ps_columns=ps_columns, p_full_columns=p_full_columns,
    )
    assert np.all(out_t[:-1, 0] > out[:-1, 0])


def test_planted_dry_bias_is_removed_at_the_reports_and_the_vapor_alone_moves(spun_up, tmp_path):
    cfg, checkpoint = spun_up
    obs = _iem_reports(cfg, checkpoint, tmp_path / "dry.csv", dewpoint_offset_k=3.0)
    outdir = tmp_path / "analysis"
    code = cli_main([
        "assimilate", CONFIG, str(checkpoint), "--obs", str(obs),
        "--out", str(outdir), "--length-scale-km", "4000",
        "--moisture-update", "on",
    ])
    assert code == 0
    report = json.loads((outdir / "assimilation-report.json").read_text())
    assert report["status"] == "pass"
    assert report["options"]["moisture_update"] is True
    # Selectable, not the default: the bare door leaves water untouched.
    assert AssimilationOptions().moisture_update is False
    row = report["variables"]["dewpoint_k"]
    assert row["analysed"] is True and row["gated"] is True and row["gate_passed"] is True
    # The background is 3 K dry against every report; the analysis
    # removes most of it at the reports (the OI single-report limit keeps
    # 1/(1+g) = 0.2 of the innovation; the dense set saturates further)
    # and predicts the withheld tenth better than the background did.
    assert row["o_minus_b"]["mean"] == pytest.approx(3.0, abs=0.05)
    assert row["o_minus_a"]["rms"] < 0.25 * row["o_minus_b"]["rms"]
    assert row["withheld"]["o_minus_a"]["rms"] < row["withheld"]["o_minus_b"]["rms"]
    # The other four variables carry the door's own end-to-end signal and
    # pass their gates beside the moisture update.
    for variable in ("surface_pressure_pa", "temperature_k", "wind_u_m_s", "wind_v_m_s"):
        assert report["variables"][variable]["gate_passed"] is True, variable
    moisture = report["moisture_update"]
    assert moisture["enabled"] is True and moisture["reports"] == row["count"]
    assert moisture["increment"]["dewpoint_k_maxabs"] > 2.0
    assert moisture["increment"]["specific_humidity_kg_kg_maxabs"] > 0.0
    assert moisture["increment"]["column_water_kg_m2_global_mean"] > 0.0
    assert "1500 m" in moisture["vertical_localization"]
    assert "positivity_repair" in moisture
    qv_bg, state_bg, transform = _grid_vapor(cfg, checkpoint)
    qv_an, state_an, _ = _grid_vapor(cfg, Path(report["analysis"]["path"]))
    assert np.any(qv_an != qv_bg)
    host = transform.backend.to_numpy
    # The vapor moved toward the reports: the global-mean column water
    # rose, and the grid vapor stays inside its bounds after the repair.
    grid_bg = transform.inverse(state_bg.atmosphere.qv)
    grid_an = transform.inverse(state_an.atmosphere.qv)
    assert float(np.mean(host(grid_an))) > float(np.mean(host(grid_bg)))
    assert float(np.min(host(grid_an))) >= -1e-12
    # The moistened analysis restarts and passes every gate: the water
    # target is re-based to the analysis (a chain opens a new epoch), so
    # the fixer does not spend the first step removing the increment.
    from woof.globe.runner import run
    result = run(cfg, tmp_path / "resumed", restart=Path(report["analysis"]["path"]))
    assert result["status"] == "pass", result["gates"]
    assert result["restart_targets"]["total_water_target_kg_m2"] > result["restart_targets"]["cold_total_water_target_kg_m2"]
    assert result["gates"]["water_fixer_max_step_relative"]["value"] <= result["gates"]["water_fixer_max_step_relative"]["limit"]


def test_reports_that_agree_with_the_background_move_no_vapor(spun_up, tmp_path):
    cfg, checkpoint = spun_up
    obs = _iem_reports(cfg, checkpoint, tmp_path / "agree.csv", dewpoint_offset_k=0.0)
    outdir = tmp_path / "analysis"
    result = assimilate(cfg, checkpoint, [str(obs)], outdir, options=AssimilationOptions(length_scale_km=4000.0, moisture_update=True))
    row = result["variables"]["dewpoint_k"]
    # The Fahrenheit round trip of the CSV is the only residual: under a
    # millikelvin, and no increment is manufactured from it beyond that.
    assert row["o_minus_b"]["rms"] < 2e-3
    assert result["moisture_update"]["increment"]["dewpoint_k_maxabs"] < 2e-3
    qv_bg, _, transform = _grid_vapor(cfg, checkpoint)
    qv_an, _, _ = _grid_vapor(cfg, Path(result["analysis"]["path"]))
    host = transform.backend.to_numpy
    grid_bg = host(transform.inverse(transform.backend.asarray(qv_bg)))
    grid_an = host(transform.inverse(transform.backend.asarray(qv_an)))
    assert float(np.max(np.abs(grid_an - grid_bg))) < 1e-6 * float(np.max(grid_bg))


def test_exact_agreement_is_exactly_zero_increment(spun_up, tmp_path):
    """Bit for bit: the increment is q(Td + 0) - q(Td) through one relation,
    so a zero dewpoint innovation writes no vapor at all (the CSV route
    above carries a Fahrenheit rounding; this drives the door's own rows)."""
    cfg, checkpoint = spun_up
    from woof.globe import assimilate as door

    space, state = _model_space(cfg, checkpoint)
    lats = np.repeat(np.linspace(-60.0, 60.0, 8), 8)
    lons = np.tile(np.arange(0.0, 360.0, 45.0), 8)
    td = space.hx_dewpoint(state.atmosphere, _probe("dewpoint_k", lats, lons))
    rows = [ObsRow(
        source="probe", station_id=f"P{k}", latitude_deg=lats[k], longitude_deg=lons[k],
        elevation_m=0.0, level_pa=None,
        valid_time=dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc),
        variable="dewpoint_k", value=float(td[k]), error=1.5,
    ) for k in range(lats.size)]
    original = door.load_obs

    def fake_load(location):
        return "probe", list(rows), {"location": location, "source": "probe", "counters": {}}

    door.load_obs = fake_load
    try:
        result = assimilate(cfg, checkpoint, ["probe"], tmp_path / "analysis",
                            options=AssimilationOptions(length_scale_km=4000.0, moisture_update=True))
    finally:
        door.load_obs = original
    assert result["variables"]["dewpoint_k"]["o_minus_b"]["rms"] == 0.0
    assert result["moisture_update"]["increment"]["specific_humidity_kg_kg_maxabs"] == 0.0
    qv_bg, state_bg, transform = _grid_vapor(cfg, checkpoint)
    qv_an, _, _ = _grid_vapor(cfg, Path(result["analysis"]["path"]))
    assert np.array_equal(qv_an, qv_bg)

    # The same rows 3 K moist: a dewpoint-only report set moves the vapor
    # and nothing else (theta, vorticity, divergence bit-identical).
    rows = [dataclasses.replace(row, value=row.value + 3.0) for row in rows]
    door.load_obs = fake_load
    try:
        result = assimilate(cfg, checkpoint, ["probe"], tmp_path / "dry-only",
                            options=AssimilationOptions(length_scale_km=4000.0, moisture_update=True))
    finally:
        door.load_obs = original
    qv_an, state_an, _ = _grid_vapor(cfg, Path(result["analysis"]["path"]))
    host = transform.backend.to_numpy
    assert np.any(qv_an != qv_bg)
    for name in ("vorticity", "divergence", "theta"):
        assert np.array_equal(host(getattr(state_an.atmosphere, name)), host(getattr(state_bg.atmosphere, name))), name
    assert result["variables"]["dewpoint_k"]["o_minus_a"]["rms"] < 0.25 * 3.0


def test_bounds_cap_at_saturation_floor_at_zero_and_leave_zero_exactly_zero():
    """The bounded increment on synthetic columns, both directions: a
    planted excess lands exactly on saturation where it would have crossed
    it and nowhere else; a planted drying lands exactly on zero; a zero
    dewpoint change is exactly zero vapor; a drying at a point already
    above saturation is not blocked; a moistening there is refused."""
    rng = np.random.default_rng(3)
    p = np.linspace(3.0e4, 1.0e5, 12)[:, None] * np.ones((1, 9))
    t = 220.0 + 80.0 * (p / 1.0e5) + rng.normal(0.0, 2.0, p.shape)
    q_sat = specific_humidity_from_dewpoint(t, p)
    rh = rng.uniform(0.2, 0.95, p.shape)
    q_bg = rh * q_sat
    # (1) a large moistening: capped where the unbounded increment crosses
    # saturation, exactly at saturation there, untouched elsewhere.
    dq, capped, floored = bounded_vapor_increment(q_bg, t, p, np.full(p.shape, 25.0))
    unbounded = specific_humidity_from_dewpoint(
        dewpoint_from_specific_humidity(q_bg, p) + 25.0, p
    ) - specific_humidity_from_dewpoint(dewpoint_from_specific_humidity(q_bg, p), p)
    assert capped.any() and not floored.any()
    assert np.array_equal(capped, q_bg + unbounded > q_sat)
    assert np.allclose(q_bg[capped] + dq[capped], q_sat[capped], rtol=1e-12)
    assert np.array_equal(dq[~capped], unbounded[~capped])
    assert np.all(q_bg + dq <= q_sat * (1.0 + 1e-12))
    # (2) a large drying: the dewpoint form cannot cross zero (a colder
    # dewpoint is a smaller positive vapor), so nothing is floored and the
    # analysed vapor stays positive everywhere.
    dq, capped, floored = bounded_vapor_increment(q_bg, t, p, np.full(p.shape, -60.0))
    assert not floored.any() and not capped.any()
    assert np.all(q_bg + dq > 0.0) and np.all(dq < 0.0)
    # The floor acts on a background the truncation left negative: such a
    # point is held (zero increment) and named, never dried further.
    q_neg = q_bg.copy()
    q_neg[0, :3] = -1.0e-7
    dq, capped, floored = bounded_vapor_increment(q_neg, t, p, np.full(p.shape, -5.0))
    assert np.array_equal(floored, q_neg < 0.0)
    assert np.all(dq[floored] == 0.0)
    assert np.all(q_neg[~floored] + dq[~floored] > 0.0)
    # (3) zero change: bit-for-bit zero.
    dq, capped, floored = bounded_vapor_increment(q_bg, t, p, np.zeros(p.shape))
    assert np.all(dq == 0.0) and not capped.any() and not floored.any()
    # (4) a small change is the dewpoint change through the local slope:
    # first order in delta_td, the Clausius-Clapeyron derivative.
    dq, _c, _f = bounded_vapor_increment(q_bg, t, p, np.full(p.shape, 0.01))
    td = dewpoint_from_specific_humidity(q_bg, p)
    slope = (specific_humidity_from_dewpoint(td + 1e-4, p) - specific_humidity_from_dewpoint(td - 1e-4, p)) / 2e-4
    assert np.allclose(dq, 0.01 * slope, rtol=1e-3)
    # (5) a point already above saturation: a moistening is refused (zero
    # increment), a drying passes untouched.
    q_over = 1.2 * q_sat
    dq, capped, floored = bounded_vapor_increment(q_over, t, p, np.full(p.shape, 5.0))
    assert capped.all() and np.all(dq == 0.0)
    dq, capped, floored = bounded_vapor_increment(q_over, t, p, np.full(p.shape, -2.0))
    assert not capped.any() and not floored.any() and np.all(dq < 0.0)


def test_the_door_reports_its_bounds_and_the_analysis_carries_no_negative_vapor(spun_up, tmp_path):
    cfg, checkpoint = spun_up
    wet = _iem_reports(cfg, checkpoint, tmp_path / "wet.csv", dewpoint_offset_k=40.0)
    result = assimilate(cfg, checkpoint, [str(wet)], tmp_path / "wet",
                        options=AssimilationOptions(length_scale_km=4000.0, moisture_update=True))
    bounds = result["moisture_update"]["bounds"]
    assert bounds["saturation_capped_points"] > 0
    assert bounds["zero_floored_points"] == 0
    after = result["moisture_update"]["after_repair"]
    assert after["negative_points"] == 0 and after["min_specific_humidity_kg_kg"] >= 0.0
    assert "supersaturated_points" in after and "newly_supersaturated_points" in result["moisture_update"]["after_truncation"]
    dry = _iem_reports(cfg, checkpoint, tmp_path / "dry.csv", dewpoint_offset_k=-80.0)
    result = assimilate(cfg, checkpoint, [str(dry)], tmp_path / "dry",
                        options=AssimilationOptions(length_scale_km=4000.0, moisture_update=True))
    bounds = result["moisture_update"]["bounds"]
    # An 80 K planted drying crosses nothing: the dewpoint form keeps the
    # vapor positive without the floor, and the cap has nothing to do.
    assert bounds["zero_floored_points"] == 0
    assert bounds["saturation_capped_points"] == 0
    assert result["moisture_update"]["increment"]["column_water_kg_m2_global_mean"] < 0.0
    qv_an, state_an, transform = _grid_vapor(cfg, Path(result["analysis"]["path"]))
    from woof.globe.runner import build_model_and_cold_state
    model, _ = build_model_and_cold_state(cfg, transform)
    grid = transform.backend.to_numpy(model.grid_state(state_an.atmosphere, only=("qv",))["qv"])
    assert float(np.min(grid)) >= 0.0
    assert result["moisture_update"]["after_repair"]["negative_points"] == 0
    assert result["moisture_update"]["positivity_repair"]["largest_negative_vapor_kg_kg"] >= 0.0


def test_moisture_update_off_leaves_the_vapor_as_v1_did(spun_up, tmp_path):
    cfg, checkpoint = spun_up
    obs = _iem_reports(cfg, checkpoint, tmp_path / "dry.csv", dewpoint_offset_k=3.0)
    outdir = tmp_path / "off"
    code = cli_main([
        "assimilate", CONFIG, str(checkpoint), "--obs", str(obs),
        "--out", str(outdir), "--length-scale-km", "4000", "--moisture-update", "off",
    ])
    assert code == 0
    report = json.loads((outdir / "assimilation-report.json").read_text())
    assert report["options"]["moisture_update"] is False
    row = report["variables"]["dewpoint_k"]
    assert row["analysed"] is False and row["gated"] is False
    assert "gate_passed" not in row
    assert report["moisture_update"]["enabled"] is False
    assert "bounds" not in report["moisture_update"]
    qv_bg, _, _ = _grid_vapor(cfg, checkpoint)
    qv_an, _, _ = _grid_vapor(cfg, Path(report["analysis"]["path"]))
    assert np.array_equal(qv_an, qv_bg)
    assert "v1 of this door left water untouched" in report["moisture_update"]["divergence_from_v1"]


def test_repeated_points_read_the_same_bits_as_single_rows(spun_up):
    """The operators synthesize once per distinct point and the rows gather:
    a station's five variable rows and a site's seven levels must read
    exactly what a lone row at that point reads, surface and aloft."""
    cfg, checkpoint = spun_up
    space, state = _model_space(cfg, checkpoint)
    lats = np.array([10.0, -35.0, 52.5])
    lons = np.array([20.0, 150.0, 300.0])
    single = {
        variable: getattr(space, method)(state.atmosphere, _probe(variable, lats, lons))
        if method != "hx_wind" else space.hx_wind(state.atmosphere, _probe(variable, lats, lons), variable[5])
        for variable, method in (
            ("surface_pressure_pa", "hx_surface_pressure"), ("temperature_k", "hx_temperature"),
            ("dewpoint_k", "hx_dewpoint"), ("wind_u_m_s", "hx_wind"), ("wind_v_m_s", "hx_wind"),
        )
    }
    repeated_lats = np.repeat(lats, 4)
    repeated_lons = np.repeat(lons, 4)
    rows = []
    for variable in single:
        rows.extend(_probe(variable, repeated_lats, repeated_lons).rows)
    union = _Family(rows)
    values = space.evaluate(state.atmosphere, union)
    start = 0
    for variable, expected in single.items():
        got = values[variable][start:start + repeated_lats.size]
        assert np.array_equal(got, np.repeat(expected, 4)), variable
        start += repeated_lats.size
    # Aloft: seven levels at one site read the profile interpolated to each
    # level, the same bits whether the levels ride together or alone.
    levels = [85000.0, 70000.0, 50000.0, 40000.0, 30000.0, 25000.0, 20000.0]
    together = _Family([
        row for level in levels
        for row in _probe("temperature_k", lats[:1], lons[:1], level_pa=level, elev=1500.0).rows
    ])
    together_values = space.evaluate(state.atmosphere, together)["temperature_k"]
    for k, level in enumerate(levels):
        alone = space.hx_temperature(
            state.atmosphere, _probe("temperature_k", lats[:1], lons[:1], level_pa=level, elev=1500.0))
        assert together_values[k] == alone[0], level
