from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pytest

from woof.globe.assimilate import (
    ASSIMILATION_HISTORY_KEY,
    ASSIMILATION_HISTORY_SCHEMA,
    LOCALIZATION_SCALE_HEIGHT_M,
    REJECTION_BREAKAGE,
    AssimilationOptions,
    _Family,
    _ModelSpace,
    _anemometer_wind,
    _balanced_wind_increment,
    _chain_for_analysis,
    _increment_kinetic_energy,
    _sponge_columns,
    _spread_column,
    _table_quality_control,
    _to_numpy_spectral,
    _withhold,
    assimilate,
)
from woof.globe.checkpoint import read_checkpoint, state_from_checkpoint
from woof.globe.cli import main as cli_main
from woof.globe.constants import (
    DRY_AIR_GAS_CONSTANT,
    EARTH_RADIUS_M,
    GRAVITY_M_S2,
    KAPPA,
    REFERENCE_PRESSURE_PA,
)
from woof.globe.config import load_config
from woof.globe.obs_table import (
    INHG_TO_PA,
    KNOTS_TO_M_S,
    ObsRow,
    decode_obs_csv,
    isa_pressure_pa,
    load_obs,
)
from woof.globe.runner import build_model_and_cold_state, build_transform, run
from woof.globe.vertical import HybridCoordinate

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")

METAR_HEADER = (
    "raw_text,station_id,observation_time,latitude,longitude,temp_c,"
    "wind_dir_degrees,wind_speed_kt,altim_in_hg,elevation_m"
)
AIRCRAFT_HEADER = (
    "receipt_time,observation_time,no_time_stamp,above_ground_level_indicated,"
    "no_flt_lvl,bad_location,aircraft_ref,latitude,longitude,altitude_ft_msl,"
    "temp_c,wind_dir_degrees,wind_speed_kt,report_type"
)


def _metar_line(
    station: str, time: str, lat: float, lon: float, *,
    temp_c="", direction="", speed_kt="", altim_in_hg="", elevation_m="0",
) -> str:
    return (
        f"RAW,{station},{time},{lat},{lon},{temp_c},"
        f"{direction},{speed_kt},{altim_in_hg},{elevation_m}"
    )


def _aircraft_line(
    ident: str, time: str, lat: float, lon: float, *,
    altitude_ft="35000", temp_c="", direction="", speed_kt="",
    bad_location="", no_flt_lvl="",
) -> str:
    return (
        f"{time},{time},,,{no_flt_lvl},{bad_location},{ident},"
        f"{lat},{lon},{altitude_ft},{temp_c},{direction},{speed_kt},AIREP"
    )


def test_metar_decoder_units_and_variable_wind():
    text = "\n".join([
        METAR_HEADER,
        _metar_line(
            "AAAA", "2026-08-31T12:00:00Z", 10.0, 20.0,
            temp_c="15", direction="90", speed_kt="10",
            altim_in_hg="29.92", elevation_m="0",
        ),
        _metar_line(
            "BBBB", "2026-08-31T12:05:00Z", -10.0, 200.0,
            temp_c="-5", direction="0", speed_kt="7",
            altim_in_hg="30.10", elevation_m="1000",
        ),
    ]) + "\n"
    source, rows, counters = decode_obs_csv(text)
    assert source == "awc-metar-cache"
    by = {(r.station_id, r.variable): r for r in rows}

    pressure = by[("AAAA", "surface_pressure_pa")]
    assert pressure.value == pytest.approx(29.92 * INHG_TO_PA, rel=1e-9)
    assert by[("AAAA", "temperature_k")].value == pytest.approx(288.15)
    # 90 degrees at 10 kt blows toward the west: u negative, v ~ 0.
    assert by[("AAAA", "wind_u_m_s")].value == pytest.approx(-10 * KNOTS_TO_M_S)
    assert by[("AAAA", "wind_v_m_s")].value == pytest.approx(0.0, abs=1e-9)

    # Elevated station: the altimeter inversion applies the ISA column.
    reduced = by[("BBBB", "surface_pressure_pa")]
    expected = (30.10 * INHG_TO_PA) * (
        1.0 - 0.0065 * 1000.0 / 288.15
    ) ** 5.255877
    assert reduced.value == pytest.approx(expected, rel=1e-9)
    assert reduced.elevation_m == 1000.0
    # East-of-antimeridian longitude normalizes into -180..180.
    assert reduced.longitude_deg == pytest.approx(-160.0)
    # Direction 0 with nonzero speed encodes a variable wind: no wind rows.
    assert ("BBBB", "wind_u_m_s") not in by
    assert ("BBBB", "wind_v_m_s") not in by
    assert counters["values_not_derivable"] == 2


def test_aircraft_decoder_flags_and_flight_level():
    text = "\n".join([
        AIRCRAFT_HEADER,
        _aircraft_line(
            "GOOD1", "2026-08-31T12:00:00Z", 45.0, -60.0,
            altitude_ft="35000", temp_c="-40", direction="270", speed_kt="80",
        ),
        _aircraft_line(
            "BAD1", "2026-08-31T12:00:00Z", 45.0, -60.0,
            altitude_ft="35000", temp_c="-40", bad_location="TRUE",
        ),
    ]) + "\n"
    source, rows, counters = decode_obs_csv(text)
    assert source == "awc-aircraft-cache"
    assert counters["rows_flag_rejected"] == 1
    assert {r.station_id for r in rows} == {"GOOD1"}
    temp = next(r for r in rows if r.variable == "temperature_k")
    assert temp.value == pytest.approx(233.15)
    assert temp.elevation_m == pytest.approx(35000 * 0.3048)
    assert temp.level_pa == pytest.approx(isa_pressure_pa(35000 * 0.3048))
    u = next(r for r in rows if r.variable == "wind_u_m_s")
    assert u.value == pytest.approx(80 * KNOTS_TO_M_S, rel=1e-9)


def test_unknown_header_is_refused_with_declared_sources():
    with pytest.raises(ValueError, match="no declared obs decoder"):
        decode_obs_csv("alpha,beta,gamma\n1,2,3\n")


def test_third_source_joins_by_table_entry_alone():
    """The arbitrary acceptance test: a new stream is a decoder-table entry,
    never a code path.  A synthetic buoy-like stream decodes through the
    unchanged engine, and the two shipped streams still decode beside it."""
    from woof.globe.obs_table import (
        DECODER_TABLES, SourceDecoder, VariableRule,
    )

    buoy = SourceDecoder(
        source="synthetic-buoy-cache",
        detect_columns=(
            "buoy_id", "obs_time", "lat_deg", "lon_deg", "hull_altim_in_hg",
        ),
        id_column="buoy_id",
        time_column="obs_time",
        latitude_column="lat_deg",
        longitude_column="lon_deg",
        elevation_column="hull_elev_m",
        elevation_to_m=1.0,
        level_mode="surface",
        reject_when_true=(),
        variables=(
            VariableRule(
                variable="surface_pressure_pa",
                columns=("hull_altim_in_hg", "hull_elev_m"),
                conversion="altimeter_inhg_to_station_pa",
                error=120.0,
            ),
            VariableRule(
                variable="temperature_k",
                columns=("sst_c",),
                conversion="celsius_to_kelvin",
                error=1.2,
            ),
        ),
    )
    tables = DECODER_TABLES + (buoy,)
    text = (
        "buoy_id,obs_time,lat_deg,lon_deg,hull_elev_m,hull_altim_in_hg,sst_c\n"
        "B001,2026-08-31T12:00:00Z,10.5,-40.25,0,29.92,21.5\n"
    )
    source, rows, counters = decode_obs_csv(text, tables=tables)
    assert source == "synthetic-buoy-cache"
    assert counters["observations"] == 2
    by = {r.variable: r for r in rows}
    assert by["surface_pressure_pa"].value == pytest.approx(29.92 * INHG_TO_PA)
    assert by["surface_pressure_pa"].error == 120.0
    assert by["temperature_k"].value == pytest.approx(294.65)
    metar_text = "\n".join([
        METAR_HEADER,
        _metar_line("AAAA", "2026-08-31T12:00:00Z", 1.0, 2.0, temp_c="10"),
    ]) + "\n"
    source, rows, _ = decode_obs_csv(metar_text, tables=tables)
    assert source == "awc-metar-cache"
    assert rows and rows[0].variable == "temperature_k"


def _row(
    variable="surface_pressure_pa", value=100_000.0, *, station="S",
    minutes_old=10.0, level_pa=None, source="awc-metar-cache",
    longitude_deg=0.0, latitude_deg=0.0,
):
    return ObsRow(
        source=source, station_id=station, latitude_deg=latitude_deg,
        longitude_deg=longitude_deg, elevation_m=0.0, level_pa=level_pa,
        valid_time=dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc)
        - dt.timedelta(minutes=minutes_old),
        variable=variable, value=value, error=100.0,
    )


def test_table_quality_control_gates():
    moment = dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc)
    options = AssimilationOptions()
    rows = [
        _row(value=100_000.0, station="KEEP", minutes_old=10.0),
        _row(value=200_000.0, station="GROSS", minutes_old=10.0),
        _row(value=100_000.0, station="OLD", minutes_old=120.0),
        _row(value=100_000.0, station="FUTURE", minutes_old=-30.0),
        _row(value=99_000.0, station="DUP", minutes_old=40.0),
        _row(value=99_500.0, station="DUP", minutes_old=5.0),
    ]
    kept, rejections = _table_quality_control(rows, moment, options)
    assert rejections["gross_bounds"] == 1
    assert rejections["age_window"] == 1
    assert rejections["future_time"] == 1
    assert rejections["duplicate_superseded"] == 1
    by_station = {row.station_id: row for row in kept}
    assert set(by_station) == {"KEEP", "DUP"}
    # Latest wins for the duplicated surface station.
    assert by_station["DUP"].value == 99_500.0
    # Aloft rows collapse only exact repeats: a moving platform is a track.
    track = [
        _row(level_pa=25_000.0, station="AC1", minutes_old=10.0,
             source="awc-aircraft-cache"),
        _row(level_pa=25_000.0, station="AC1", minutes_old=20.0,
             source="awc-aircraft-cache"),
        _row(level_pa=25_000.0, station="AC1", minutes_old=10.0,
             source="awc-aircraft-cache"),
    ]
    kept, rejections = _table_quality_control(track, moment, options)
    assert len(kept) == 2
    assert rejections["duplicate_superseded"] == 1


def test_table_quality_control_rejects_a_report_at_the_exact_pole():
    """One row at latitude -90 must lose itself, not the analysis.

    Every surface row is evaluated through the lowest level's wind, and the
    spectral sampler refuses a lat-lon vector at the exact pole.  The public
    surface stream carries a station at latitude -90.0000, and before this
    gate that one row refused the whole cycle (measured 2026-09-07).
    """
    moment = dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc)
    options = AssimilationOptions()
    rows = [
        _row(station="SOUTH", latitude_deg=-90.0),
        _row(station="NORTH", latitude_deg=90.0),
        _row(station="NEAR", latitude_deg=-89.9),
        _row(station="KEEP", latitude_deg=12.5),
    ]
    kept, rejections = _table_quality_control(rows, moment, options)
    assert rejections["pole_singularity"] == 2
    # A report the sampler CAN evaluate is not thrown away for being far
    # south: the gate is the operator's own domain, not a latitude band.
    assert {row.station_id for row in kept} == {"NEAR", "KEEP"}
    assert "pole_singularity" in REJECTION_BREAKAGE


def _lat_lon_grid():
    lat = np.linspace(-85.0, 85.0, 35)
    lon = np.arange(0.0, 360.0, 5.0)
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    return lat_mesh.ravel(), lon_mesh.ravel()


def test_scheme_is_successive_correction_with_the_oi_single_report_limit():
    """Audit 2026-09-01 DA-1: the scheme is a data-density-normalised
    successive correction, and the docstring says so.  Where it coincides
    with optimal interpolation it must (one report: g/(1+g); two equal
    same-sign reports at any separation), and where it departs the
    departure is the documented under-fit of disagreeing neighbours (at
    the first of two opposite-sign reports: 0.052 against OI's 0.246 at
    half a length scale, 0.200 against 0.522 at one), not something
    else."""
    lat = np.linspace(-15.0, 15.0, 301)
    lon_mesh, lat_mesh = np.meshgrid(lat, lat)
    grid_lat, grid_lon = np.deg2rad(lat_mesh.ravel()), np.deg2rad(lon_mesh.ravel())
    options = AssimilationOptions()
    g = (2.5 / 1.5) ** 2
    centre = int(np.argmin(np.abs(grid_lat) + np.abs(grid_lon)))

    def code(rows, innovations):
        return _spread_column(
            _Family(rows), np.array(innovations), np.full(len(rows), g),
            grid_lat, grid_lon, options,
        )[centre]

    def oi_at_first(d1, d2, separation_scales):
        rho = math.exp(-0.5 * separation_scales ** 2)
        matrix = g * np.array([[1.0, rho], [rho, 1.0]]) + np.eye(2)
        weights = np.linalg.solve(matrix, np.array([d1, d2]))
        return g * np.array([1.0, rho]) @ weights

    single = code([_row(variable="temperature_k")], [1.0])
    assert single == pytest.approx(g / (1.0 + g), abs=1e-6)
    expected_under_fit = {0.5: (0.0524, 0.2461), 1.0: (0.2001, 0.5222), 2.0: (0.5782, 0.7060)}
    for separation, (scheme, textbook) in expected_under_fit.items():
        dlon = math.degrees(separation * options.length_scale_km * 1e3 / EARTH_RADIUS_M)
        rows = [
            _row(variable="temperature_k", station="A"),
            _row(variable="temperature_k", station="B", longitude_deg=dlon),
        ]
        same = code(rows, [1.0, 1.0])
        assert same == pytest.approx(oi_at_first(1.0, 1.0, separation), abs=2e-3)
        opposite = code(rows, [1.0, -1.0])
        assert opposite == pytest.approx(scheme, abs=2e-3)
        assert oi_at_first(1.0, -1.0, separation) == pytest.approx(textbook, abs=2e-3)
        assert opposite < textbook


def test_single_observation_increment_is_a_local_bump():
    lat_deg, lon_deg = _lat_lon_grid()
    family = _Family([_row(value=1.0)])  # observation at 0N 0E
    gains = np.array([9.0])
    innovation = np.array([1.0])
    options = AssimilationOptions(length_scale_km=300.0)
    delta = _spread_column(
        family, innovation, gains,
        np.deg2rad(lat_deg), np.deg2rad(lon_deg), options,
    )
    peak = int(np.argmax(np.abs(delta)))
    # The bump sits at the column nearest the observation...
    assert abs(lat_deg[peak]) <= 3.0
    assert min(lon_deg[peak], 360.0 - lon_deg[peak]) <= 3.0
    # ...with the single-observation optimal-interpolation amplitude...
    assert 0.5 * 9.0 / 10.0 <= delta[peak] <= 9.0 / 10.0 + 1e-9
    # ...and is exactly zero beyond the localization cutoff.
    distance = 6.371e6 * np.arccos(np.clip(
        np.cos(np.deg2rad(lat_deg)) * np.cos(np.deg2rad(lon_deg)), -1, 1
    ))
    assert np.all(delta[distance > 3.5 * 300e3] == 0.0)


def test_vertical_structure_of_volume_increments():
    lat_deg, lon_deg = _lat_lon_grid()
    ncol = lat_deg.size
    ps = np.full(ncol, 100_000.0)
    p_full = np.array([20_000.0, 50_000.0, 85_000.0, 97_000.0])[:, None] * np.ones(ncol)
    options = AssimilationOptions(length_scale_km=300.0)
    surface = _Family([_row(variable="temperature_k", value=1.0)])
    delta = _spread_column(
        surface, np.array([1.0]), np.array([4.0]),
        np.deg2rad(lat_deg), np.deg2rad(lon_deg), options,
        ps_columns=ps, p_full_columns=p_full,
    )
    column = delta[:, int(np.argmax(np.abs(delta[-1])))]
    # A surface observation decays upward: monotone with height.
    assert column[3] > column[2] > column[1] > column[0] >= 0.0

    aloft = _Family([_row(
        variable="temperature_k", value=1.0, level_pa=20_000.0,
        source="awc-aircraft-cache",
    )])
    delta = _spread_column(
        aloft, np.array([1.0]), np.array([4.0]),
        np.deg2rad(lat_deg), np.deg2rad(lon_deg), options,
        ps_columns=ps, p_full_columns=p_full,
    )
    column = delta[:, int(np.argmax(np.abs(delta[0])))]
    # An aloft observation localizes around its own level.
    assert column[0] == np.max(column)
    assert column[0] > column[2]


def _forty_level_columns(ncol: int, ps_pa: float = 100_000.0):
    """The default 40-level surface-stretched stack at one surface
    pressure, replicated over ``ncol`` columns (top to bottom)."""
    coordinate = HybridCoordinate.surface_stretched(40, 100.0)
    a = np.asarray(coordinate.a_half_pa)
    b = np.asarray(coordinate.b_half)
    p_half = a + b * ps_pa
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    return p_full, np.repeat(p_full[:, None], ncol, axis=1), np.full(ncol, ps_pa)


def _one_aircraft_report(level_pa: float, lat=45.0, lon=0.0) -> _Family:
    return _Family([ObsRow(
        source="awc-aircraft-cache", station_id="AC0", latitude_deg=lat,
        longitude_deg=lon, elevation_m=9000.0, level_pa=level_pa,
        valid_time=dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc),
        variable="temperature_k", value=0.0, error=1.0,
    )])


def test_aircraft_increment_stays_near_its_level_on_the_forty_level_stack():
    """Audit 2026-09-01 DA-2, reproduced on the real default stack: a 250
    hPa aircraft temperature innovation used to land at 0.616 of itself on
    the model lid (Gaussian in pressure, sigma 15 kPa).  The ln p
    localization with the ADAS 1 km zrange and its 1e-3 weight floor
    leaves exactly nothing above 177 hPa."""
    p_full, p_full_columns, ps = _forty_level_columns(5)
    lat = np.full(5, 45.0)
    lon = np.array([0.0, 1.0, 2.0, 5.0, 20.0])
    options = AssimilationOptions()
    increment = _spread_column(
        _one_aircraft_report(25_000.0), np.array([1.0]),
        np.array([(2.5 / 1.0) ** 2]), np.deg2rad(lat), np.deg2rad(lon),
        options, ps_columns=ps, p_full_columns=p_full_columns,
    )
    column = increment[:, 0]
    obs_level = int(np.argmin(np.abs(p_full - 25_000.0)))
    # The obs level takes the near-full single-observation gain.
    assert column[obs_level] >= 0.8
    assert column[obs_level] == np.max(column)
    # Lid and every level at or above 100 hPa: exactly zero, not merely
    # small (the predecessor left 0.616 at the lid).
    assert column[0] == 0.0
    assert np.all(column[p_full <= 10_000.0] == 0.0)
    # The reach is the ADAS cut: sqrt(ln 1000) * zrange in ln p metres.
    reach_m = options.aircraft_level_scale_m * math.sqrt(
        -math.log(options.vertical_weight_floor)
    )
    touched = p_full[column > 0.0]
    assert np.all(
        LOCALIZATION_SCALE_HEIGHT_M * np.abs(np.log(touched / 25_000.0))
        <= reach_m + 1e-9
    )
    assert touched.min() > 17_000.0 and touched.max() < 36_000.0
    # The weight the scheme assigns at 100/50/10/1 hPa is zero by the cut.
    for target_pa in (10_000.0, 5_000.0, 1_000.0, 100.0):
        dz = LOCALIZATION_SCALE_HEIGHT_M * abs(math.log(target_pa / 25_000.0))
        assert dz > reach_m
    # Beyond the horizontal cut (20 degrees at 45N is ~1570 km > 1050 km)
    # the column is untouched at every level.
    assert np.all(increment[:, -1] == 0.0)


def test_sponged_levels_refuse_increments_from_reports_below_the_absorber():
    """A report just under the absorber base (60 hPa) would reach the
    47 hPa ring with weight 0.04 by the vertical model alone; the sponge
    rule masks it to exactly zero.  A report inside the region (30 hPa)
    still writes there: the rule is about where the observation is."""
    p_full, p_full_columns, ps = _forty_level_columns(3)
    lat = np.full(3, 45.0)
    lon = np.array([0.0, 1.0, 2.0])
    grid_shape = p_full_columns.reshape(40, 1, 3)
    sponge = _sponge_columns(grid_shape, 5000.0)
    sponged = p_full < 5000.0
    assert sponged.sum() == 11 and np.all(sponge[sponged]) and not np.any(sponge[~sponged])
    options = AssimilationOptions()
    args = (np.array([1.0]), np.array([6.25]), np.deg2rad(lat), np.deg2rad(lon), options)
    unmasked = _spread_column(
        _one_aircraft_report(6_000.0), *args,
        ps_columns=ps, p_full_columns=p_full_columns,
    )
    masked = _spread_column(
        _one_aircraft_report(6_000.0), *args,
        ps_columns=ps, p_full_columns=p_full_columns,
        sponge_columns=sponge, sponge_base_pa=5000.0,
    )
    assert unmasked[sponged, 0].max() > 0.05
    assert np.all(masked[sponged, 0] == 0.0)
    assert np.array_equal(masked[~sponged], unmasked[~sponged])
    inside = _spread_column(
        _one_aircraft_report(3_000.0), *args,
        ps_columns=ps, p_full_columns=p_full_columns,
        sponge_columns=sponge, sponge_base_pa=5000.0,
    )
    # 30 hPa sits 680 ln-p metres from the 32.8 hPa ring: weight 0.63,
    # applied fraction 0.63 g / (1 + 0.63 g) = 0.80 at g = 6.25.
    assert inside[sponged, 0].max() > 0.75


@pytest.mark.parametrize("z1_m", [23.0, 367.0])
@pytest.mark.parametrize("roughness_m", [0.03, 0.1])
def test_surface_wind_operator_reduces_a_log_profile_to_the_anemometer(z1_m, roughness_m):
    """Audit 2026-09-01 DA-4: a 5 m/s 10 m report used to be compared to
    the lowest full level raw (8.1-8.9 m/s at 367 m, 5.7-6.2 m/s at 23 m
    for a neutral log profile), an innovation of fixed sign, larger than
    the report's 2.5 m/s error on the 20-level stack.  Against a neutral
    log-law background whose 10 m wind is 5 m/s the operator must return
    5 m/s."""
    ps = np.array([100_000.0])
    t_low = np.array([288.0])
    qv_low = np.zeros(1)
    p_full_low = ps * np.exp(-GRAVITY_M_S2 * z1_m / (DRY_AIR_GAS_CONSTANT * t_low))
    # Neutral: the skin potential temperature equals the lowest level's.
    theta_low = t_low * (REFERENCE_PRESSURE_PA / p_full_low) ** KAPPA
    skin = theta_low * (ps / REFERENCE_PRESSURE_PA) ** KAPPA
    z0 = np.array([roughness_m])
    log_ratio = math.log(z1_m / roughness_m) / math.log(10.0 / roughness_m)
    u_low = np.array([5.0 * log_ratio])
    v_low = np.array([-2.0 * log_ratio])
    assert u_low[0] > 5.5  # the raw comparison's innovation, > 0.5 m/s
    # Dry land with dry soil: the surface humidity is the air's, so the
    # virtual potential temperatures match and the layer is neutral.
    u10, v10 = _anemometer_wind(
        u_low, v_low, t_low, qv_low, p_full_low, ps, skin,
        land_fraction=np.ones(1), soil_wetness=np.zeros(1), roughness_m=z0,
    )
    assert u10[0] == pytest.approx(5.0, abs=1e-3)
    assert v10[0] == pytest.approx(-2.0, abs=1e-3)


def test_surface_wind_operator_runs_on_the_spun_up_state(spun_up):
    """Through hx_wind on a real checkpoint: surface rows come back
    reduced below the lowest level's speed (the smoke state's layer is
    stable), aloft rows are untouched by the reduction."""
    cfg, checkpoint = spun_up
    transform = build_transform(cfg)
    model, _ = build_model_and_cold_state(cfg, transform)
    metadata, arrays = read_checkpoint(checkpoint)
    state = state_from_checkpoint(metadata, arrays, transform.backend)
    terrain = _to_numpy_spectral(
        transform.backend, transform.forward(model.surface_geopotential)
    )
    space = _ModelSpace(
        transform, cfg.vertical, terrain, state.surface,
        cfg.reference_physics.soil_wetness_capacity,
    )
    lats = np.linspace(-50.0, 50.0, 6)
    lons = np.linspace(0.0, 300.0, 6)
    rows = [
        ObsRow(
            source="probe", station_id=f"P{k}", latitude_deg=lats[k],
            longitude_deg=lons[k], elevation_m=0.0, level_pa=None,
            valid_time=dt.datetime(2026, 8, 31, 12, tzinfo=dt.timezone.utc),
            variable="wind_u_m_s", value=0.0, error=1.0,
        )
        for k in range(6)
    ]
    family = _Family(rows)
    u10 = space.hx_wind(state.atmosphere, family, "u")
    v10 = space.hx_wind(state.atmosphere, family, "v")
    from woof.globe.spectral.sampling import sample_wind
    u_low, v_low = sample_wind(
        transform,
        _to_numpy_spectral(transform.backend, state.atmosphere.vorticity),
        _to_numpy_spectral(transform.backend, state.atmosphere.divergence),
        lats, lons,
    )
    speed_10 = np.hypot(u10, v10)
    speed_low = np.hypot(u_low[-1], v_low[-1])
    assert np.all(speed_10 < speed_low)
    assert np.all(speed_10 > 0.0)
    # The reduction keeps the direction: u10/v10 is a scalar multiple.
    assert np.allclose(u10 * v_low[-1], v10 * u_low[-1], atol=1e-12)


def _spectral_grid(truncation: int):
    from woof.globe.spectral.transform import SphericalHarmonicTransform
    from woof.globe.spectral.vector import VorticityDivergenceOperator

    transform = SphericalHarmonicTransform.create(
        truncation, dealias_factor=1.5, backend="numpy", precision="float64"
    )
    grid = transform.grid
    lon_mesh, lat_mesh = np.meshgrid(grid.longitude_deg, grid.latitude_deg)
    return (
        transform, VorticityDivergenceOperator(transform),
        np.deg2rad(lat_mesh.ravel()), np.deg2rad(lon_mesh.ravel()),
    )


def _wind_row(station, lat, lon, variable="wind_u_m_s"):
    return ObsRow(
        source="awc-metar-cache", station_id=station, latitude_deg=lat,
        longitude_deg=lon, elevation_m=0.0, level_pa=None,
        valid_time=dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc),
        variable=variable, value=0.0, error=2.5,
    )


def test_increment_kinetic_energy_matches_the_grid_mean():
    """Parseval: the spectral KE sum equals the grid mean of 0.5 |v|^2
    for a wind made from a random vorticity/divergence pair."""
    transform, vector, _, _ = _spectral_grid(21)
    rng = np.random.default_rng(3)
    shape = (2, 22, 22)
    zeta = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    div = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    zeta = transform.project(zeta * 1e-6)
    div = transform.project(div * 1e-6)
    zeta[..., 0, :] = 0.0
    div[..., 0, :] = 0.0
    u, v = vector.wind_from_vordiv(zeta, div)
    grid_ke = sum(
        transform.grid.global_mean(0.5 * (u[k] ** 2 + v[k] ** 2))
        for k in range(shape[0])
    )
    radius = transform.grid.radius_m
    spectral = (
        _increment_kinetic_energy(zeta, radius)
        + _increment_kinetic_energy(div, radius)
    )
    assert spectral == pytest.approx(grid_ke, rel=1e-9)


def test_single_u_only_report_applies_no_divergence():
    """Audit 2026-09-01 DA-3: one u-only report spread as a scalar bump
    analyses to an increment whose kinetic energy is half divergent
    (measured 0.499 at T63; 0.486 at T31 with a 1000 km scale).  The
    bare default applies the rotational part only: the divergence
    applied is exactly zero, below the audit's lowest measured fraction
    of 0.403, and the applied wind is the rotational half of the bump."""
    transform, vector, grid_lat, grid_lon = _spectral_grid(31)
    grid = transform.grid
    options = AssimilationOptions(length_scale_km=1000.0)
    assert options.wind_balance == "rotational"
    family = _Family([_wind_row("S", 45.0, 0.0)])
    gain = np.array([(4.0 / 2.5) ** 2])
    du = _spread_column(family, np.array([1.0]), gain, grid_lat, grid_lon, options)
    du = du.reshape(1, grid.nlat, grid.nlon)
    dv = np.zeros_like(du)
    zeta, div, receipt = _balanced_wind_increment(
        vector, transform, du, dv, options.wind_balance
    )
    assert 0.40 <= receipt["divergent_fraction_analysed"] <= 0.60
    assert receipt["divergent_fraction_applied"] == 0.0
    assert receipt["divergent_fraction_applied"] < 0.403
    assert np.all(np.asarray(div) == 0.0)
    assert receipt["rotational_ke_j_kg"] > 0.0
    assert receipt["mode"] == "rotational"
    u_applied, _ = vector.wind_from_vordiv(zeta, div)
    assert 0.4 * du.max() < np.abs(u_applied).max() < 0.6 * du.max()
    # The predecessor, kept for measurement, applies the whole split.
    _, div_kept, kept = _balanced_wind_increment(
        vector, transform, du, dv, "unconstrained"
    )
    assert kept["divergent_fraction_applied"] == receipt["divergent_fraction_analysed"]
    assert np.abs(np.asarray(div_kept)).max() > 0.0


def test_audit_scenario_b_random_wind_reports_at_t63():
    """The audit's extra_check.py section B, verbatim geometry: T63, 50
    and 400 random u-only reports over 25-60N 130-60W, innovations
    N(0, 2 m/s), seed 1.  Analysed divergent fractions reproduce the
    audit (0.403 and 0.512); the default applies zero."""
    transform, vector, grid_lat, grid_lon = _spectral_grid(63)
    grid = transform.grid
    options = AssimilationOptions()
    rng = np.random.default_rng(1)
    gain = (4.0 / 2.5) ** 2
    expected = {50: 0.403, 400: 0.512}
    for count in (50, 400):
        lat = rng.uniform(25, 60, count)
        lon = rng.uniform(-130, -60, count)
        family = _Family([_wind_row(f"s{i}", lat[i], lon[i]) for i in range(count)])
        du = _spread_column(
            family, rng.normal(0, 2.0, count), np.full(count, gain),
            grid_lat, grid_lon, options,
        ).reshape(1, grid.nlat, grid.nlon)
        dv = _spread_column(
            family, rng.normal(0, 2.0, count), np.full(count, gain),
            grid_lat, grid_lon, options,
        ).reshape(1, grid.nlat, grid.nlon)
        _, div, receipt = _balanced_wind_increment(
            vector, transform, du, dv, options.wind_balance
        )
        assert receipt["divergent_fraction_analysed"] == pytest.approx(
            expected[count], abs=0.002
        )
        assert receipt["divergent_fraction_applied"] == 0.0
        assert np.all(np.asarray(div) == 0.0)


def test_wind_balance_option_names_its_modes():
    with pytest.raises(ValueError, match="wind_balance"):
        AssimilationOptions(wind_balance="geostrophic")
    assert AssimilationOptions().identity()["wind_balance"] == "rotational"


def test_withheld_fraction_is_default_on_and_bounded():
    """The bare default withholds a tenth; a fraction that leaves the gate
    nothing to judge (0) or judges more than it analyses (>= 0.5) is
    refused with the breakage named."""
    identity = AssimilationOptions().identity()
    assert identity["withheld_fraction"] == 0.1
    assert identity["withheld_seed"] == 0
    for fraction in (0.0, 0.5, 1.0):
        with pytest.raises(ValueError, match="withheld_fraction"):
            AssimilationOptions(withheld_fraction=fraction)
    with pytest.raises(ValueError, match="withheld_seed"):
        AssimilationOptions(withheld_seed=-1)


def _station_rows(count: int, variable="temperature_k") -> list[ObsRow]:
    return [
        _row(variable=variable, value=280.0 + k, station=f"S{k:03d}")
        for k in range(count)
    ]


def test_withheld_split_is_seeded_and_independent_of_arrival_order():
    options = AssimilationOptions()
    rows = _station_rows(60)
    used, held = _withhold(rows, 1, options)
    assert len(held) == 6 and len(used) == 54
    ids = {row.identity_hash() for row in rows}
    assert {r.identity_hash() for r in used} | {r.identity_hash() for r in held} == ids
    assert not ({r.identity_hash() for r in used} & {r.identity_hash() for r in held})
    # The same reports in another order give the same withheld set.
    shuffled = list(rows)
    np.random.default_rng(9).shuffle(shuffled)
    _, held_again = _withhold(shuffled, 1, options)
    assert {r.identity_hash() for r in held_again} == {r.identity_hash() for r in held}
    # Another seed, or another variable slot, chooses differently.
    _, other_seed = _withhold(rows, 1, AssimilationOptions(withheld_seed=1))
    assert {r.identity_hash() for r in other_seed} != {r.identity_hash() for r in held}
    _, other_variable = _withhold(rows, 2, options)
    assert {r.identity_hash() for r in other_variable} != {r.identity_hash() for r in held}
    # Below the gate's minimum count nothing is gated, so nothing is withheld.
    used, held = _withhold(_station_rows(49), 1, options)
    assert len(used) == 49 and held == []


def test_report_identity_is_the_measurement_not_the_row_order():
    base = _row(variable="temperature_k", station="KABC")
    assert base.identity_hash() == _row(variable="temperature_k", station="KABC").identity_hash()
    assert len(base.identity_hash()) == 16
    later = _row(variable="temperature_k", station="KABC", minutes_old=5.0)
    assert later.identity_hash() != base.identity_hash()
    other_variable = _row(variable="wind_u_m_s", station="KABC")
    assert other_variable.identity_hash() != base.identity_hash()
    moved = _row(variable="temperature_k", station="KABC", longitude_deg=0.01)
    assert moved.identity_hash() != base.identity_hash()
    aloft = _row(variable="temperature_k", station="KABC", level_pa=25_000.0)
    assert aloft.identity_hash() != base.identity_hash()
    # The value is not part of the identity: the same report re-decoded
    # with a different rounding is the same report.
    assert base.identity_hash() == _row(
        variable="temperature_k", value=1.0, station="KABC"
    ).identity_hash()


def test_analysis_chain_keeps_only_what_a_later_cycle_could_be_offered():
    options = AssimilationOptions()
    moment = dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc)
    stale = (moment - dt.timedelta(seconds=options.maximum_age_s + 1)).isoformat(timespec="seconds")
    fresh = (moment - dt.timedelta(seconds=options.maximum_age_s)).isoformat(timespec="seconds")
    # A cycle stays as long as a report it accepted from up to
    # future_tolerance_s ahead of itself could still be in the window.
    dead_cycle = (moment - dt.timedelta(
        seconds=options.maximum_age_s + options.future_tolerance_s + 1
    )).isoformat(timespec="seconds")
    chain = {
        "schema": ASSIMILATION_HISTORY_SCHEMA,
        "reports": {"stale000000000000": stale, "fresh000000000000": fresh},
        "cycles": [
            {"analysis_time_utc": dead_cycle, "assimilated": 1, "withheld": 0},
            {"analysis_time_utc": stale, "assimilated": 1, "withheld": 0},
            {"analysis_time_utc": fresh, "assimilated": 1, "withheld": 0},
        ],
    }
    rows = _station_rows(3)
    result = _chain_for_analysis(
        chain, rows, moment, options, background_sha256="ab" * 32, withheld_count=2,
    )
    assert result["schema"] == ASSIMILATION_HISTORY_SCHEMA
    assert "stale000000000000" not in result["reports"]
    assert result["reports"]["fresh000000000000"] == fresh
    for row in rows:
        assert result["reports"][row.identity_hash()] == row.valid_time.isoformat(timespec="seconds")
    assert [c["analysis_time_utc"] for c in result["cycles"]] == [
        stale, fresh, moment.isoformat(timespec="seconds")
    ]
    # The entry is one link of the lineage: it names the filter that
    # formed the increment and the streams (obs-table sources) that fed it.
    assert result["cycles"][-1] == {
        "analysis_time_utc": moment.isoformat(timespec="seconds"),
        "background_self_sha256": "ab" * 32,
        "assimilated": 3,
        "withheld": 2,
        "filter": "successive-correction",
        "streams": ["awc-metar-cache"],
    }


def _wind_to_dir_speed(u: float, v: float) -> tuple[float, float]:
    speed = math.hypot(u, v)
    direction = math.degrees(math.atan2(-u, -v)) % 360.0
    if direction == 0.0 and speed > 0.0:
        direction = 360.0
    return direction, speed / KNOTS_TO_M_S


@pytest.fixture(scope="module")
def spun_up(tmp_path_factory):
    cfg = load_config(CONFIG)
    outdir = tmp_path_factory.mktemp("background")
    run(cfg, outdir)
    return cfg, outdir / "arwen_global_step00000002.npz"


def _rotational_wind_signal(lat_deg, lon_deg, amplitude_m_s):
    """A nondivergent wind perturbation: the streamfunction
    ``psi = a A sin(lon) cos(lat)`` (rigid rotation about an equatorial
    axis, the simplest degree-1 mode) gives ``u = A sin(lon) sin(lat)``,
    ``v = A cos(lon)``.  The predecessor signal, ``(3P, -3P + 1)`` with
    ``P = sin(lon) cos(lat)``, was 0.63 divergent by kinetic energy at T3
    (a uniform northward wind is the gradient of ``chi ~ lat`` and
    converges at the pole); a rotational analysis cannot fit that and
    should not, so the fixture carries the wind error the scheme is
    built for."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return (
        amplitude_m_s * np.sin(lon) * np.sin(lat),
        amplitude_m_s * np.cos(lon),
    )


def _model_space(cfg, checkpoint: Path):
    """The observation operators against one checkpoint's state."""
    transform = build_transform(cfg)
    model, _ = build_model_and_cold_state(cfg, transform)
    metadata, arrays = read_checkpoint(checkpoint)
    state = state_from_checkpoint(metadata, arrays, transform.backend)
    terrain = _to_numpy_spectral(
        transform.backend, transform.forward(model.surface_geopotential)
    )
    space = _ModelSpace(
        transform, cfg.vertical, terrain, state.surface,
        cfg.reference_physics.soil_wetness_capacity,
    )
    return space, state


def _synthetic_obs_files(cfg, checkpoint: Path, directory: Path) -> list[Path]:
    """Write METAR/aircraft CSVs whose values are H(background) plus a
    smooth zero-mean wavenumber-one pattern (wind: a nondivergent one,
    ``_rotational_wind_signal``)."""
    space, state = _model_space(cfg, checkpoint)
    time = "2026-08-31T12:00:00Z"

    lats = np.repeat(np.linspace(-60.0, 60.0, 8), 8)
    lons = np.tile(np.arange(0.0, 360.0, 45.0), 8)
    pattern = np.sin(np.deg2rad(lons)) * np.cos(np.deg2rad(lats))

    def probe(variable, level_pa=None, elev=0.0, lat=lats, lon=lons):
        return _Family([
            ObsRow(
                source="probe", station_id=f"P{k}", latitude_deg=lat[k],
                longitude_deg=lon[k], elevation_m=elev, level_pa=level_pa,
                valid_time=dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc),
                variable=variable, value=0.0, error=1.0,
            )
            for k in range(len(lat))
        ])

    ps_bg = space.hx_surface_pressure(state.atmosphere, probe("surface_pressure_pa"))
    t_bg = space.hx_temperature(state.atmosphere, probe("temperature_k"))
    u_bg = space.hx_wind(state.atmosphere, probe("wind_u_m_s"), "u")
    v_bg = space.hx_wind(state.atmosphere, probe("wind_v_m_s"), "v")

    du, dv = _rotational_wind_signal(lats, lons, 3.0)
    metar_lines = [METAR_HEADER]
    for k in range(lats.size):
        altimeter = (ps_bg[k] + 500.0 * pattern[k]) / INHG_TO_PA
        direction, speed_kt = _wind_to_dir_speed(u_bg[k] + du[k], v_bg[k] + dv[k])
        metar_lines.append(_metar_line(
            f"ST{k:03d}", time, lats[k], lons[k],
            temp_c=f"{t_bg[k] + 2.0 * pattern[k] - 273.15:.4f}",
            direction=f"{direction:.3f}", speed_kt=f"{speed_kt:.4f}",
            altim_in_hg=f"{altimeter:.6f}", elevation_m="0",
        ))

    altitude_ft = 30_000.0
    level = isa_pressure_pa(altitude_ft * 0.3048)
    air_lat = np.linspace(-50.0, 50.0, 16)
    air_lon = np.arange(10.0, 330.0, 20.0)
    air_pattern = np.sin(np.deg2rad(air_lon))
    fam_kwargs = dict(level_pa=level, elev=altitude_ft * 0.3048,
                      lat=air_lat, lon=air_lon)
    t_air = space.hx_temperature(state.atmosphere, probe("temperature_k", **fam_kwargs))
    u_air = space.hx_wind(state.atmosphere, probe("wind_u_m_s", **fam_kwargs), "u")
    v_air = space.hx_wind(state.atmosphere, probe("wind_v_m_s", **fam_kwargs), "v")
    du_air, dv_air = _rotational_wind_signal(air_lat, air_lon, 4.0)
    aircraft_lines = [AIRCRAFT_HEADER]
    for k in range(air_lat.size):
        direction, speed_kt = _wind_to_dir_speed(
            u_air[k] + du_air[k], v_air[k] + dv_air[k]
        )
        aircraft_lines.append(_aircraft_line(
            f"AC{k:02d}", time, air_lat[k], air_lon[k],
            altitude_ft=f"{altitude_ft:.0f}",
            temp_c=f"{t_air[k] + 2.0 * air_pattern[k] - 273.15:.4f}",
            direction=f"{direction:.3f}", speed_kt=f"{speed_kt:.4f}",
        ))

    metar_path = directory / "synthetic_metars.csv"
    aircraft_path = directory / "synthetic_aircraft.csv"
    metar_path.write_text("\n".join(metar_lines) + "\n", encoding="utf-8")
    aircraft_path.write_text("\n".join(aircraft_lines) + "\n", encoding="utf-8")
    return [metar_path, aircraft_path]


def test_end_to_end_analysis_beats_background_and_restarts(spun_up, tmp_path):
    cfg, checkpoint = spun_up
    obs_files = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    outdir = tmp_path / "analysis"
    code = cli_main([
        "assimilate", CONFIG, str(checkpoint),
        "--obs", str(obs_files[0]), "--obs", str(obs_files[1]),
        "--out", str(outdir), "--length-scale-km", "4000",
    ])
    assert code == 0
    report = json.loads((outdir / "assimilation-report.json").read_text())
    assert report["status"] == "pass"
    assert report["schema"] == "gpuwm.arwen-global-assimilation/v1"
    assert report["options"]["withheld_fraction"] == 0.1
    assimilated_ids: set[str] = set()
    for variable in (
        "surface_pressure_pa", "temperature_k", "wind_u_m_s", "wind_v_m_s"
    ):
        row = report["variables"][variable]
        held = row["withheld"]
        assert row["count"] >= 50
        assert row["gated"] is True
        # The gate of record is the withheld tenth: judged, never analysed.
        assert held["count"] == round(0.1 * (row["count"] + held["count"]))
        assert len(held["ids"]) == held["count"]
        assert held["o_minus_a"]["rms"] < held["o_minus_b"]["rms"], variable
        assert row["gate_passed"] is True
        # The assimilated-row fit is reported beside it as a diagnostic.
        assert row["o_minus_a"]["rms"] < row["o_minus_b"]["rms"], variable
        ids = report["assimilated_report_ids"][variable]
        assert len(ids) == row["count"]
        assert not set(ids) & set(held["ids"])
        assimilated_ids |= set(ids)
    assert report["assimilated_total"] == len(assimilated_ids)
    assert report["withheld_total"] == sum(
        r["withheld"]["count"] for r in report["variables"].values()
    )
    assert "withheld" in report["gate_of_record"]["rule"]
    assert report["rejections"]["gross_bounds"] == 0
    assert report["rejections"]["already_assimilated"] == 0
    # The bare default applies a rotational wind increment: the report
    # says so, and the divergence it applied is exactly none.
    balance = report["wind_balance"]
    assert report["options"]["wind_balance"] == "rotational"
    assert balance["mode"] == "rotational"
    assert balance["divergent_fraction_applied"] == 0.0
    assert balance["rotational_ke_j_kg"] > 0.0

    # Identity: the analysis carries the run's config hash and trackers and
    # is accepted by the restart door on the same configuration.
    analysis_path = Path(report["analysis"]["path"])
    metadata, _ = read_checkpoint(analysis_path, expected_config_hash=cfg.config_hash)
    source_metadata, _ = read_checkpoint(checkpoint)
    assert metadata["run_trackers"] == source_metadata["run_trackers"]
    assert metadata["step"] == source_metadata["step"]
    assert metadata["time_s"] == source_metadata["time_s"]
    # The analysis carries its assimilation chain: every assimilated
    # report's identity, none of the withheld ones, and this cycle.
    chain = metadata["physics_metadata"][ASSIMILATION_HISTORY_KEY]
    assert chain["schema"] == ASSIMILATION_HISTORY_SCHEMA
    assert set(chain["reports"]) == assimilated_ids
    assert ASSIMILATION_HISTORY_KEY not in source_metadata["physics_metadata"]
    assert chain["cycles"] == [{
        "analysis_time_utc": report["analysis_time_utc"],
        "background_self_sha256": source_metadata["self_sha256"],
        "assimilated": report["assimilated_total"],
        "withheld": report["withheld_total"],
        "step": source_metadata["step"],
        # The lineage link: the filter and the streams that fed the increment.
        "filter": "successive-correction",
        "streams": ["awc-aircraft-cache", "awc-metar-cache"],
    }]
    assert report["lineage"]["filter"] == "successive-correction"
    assert report["lineage"]["streams"] == ["awc-aircraft-cache", "awc-metar-cache"]
    assert report["lineage"]["chain_length"] == 1
    # The DA scorecard rides in the report: both streams reached the
    # operators (engineering complete) and O-A fell below O-B on the
    # station pressure (a statistical reading under amendment G).
    card = report["scorecard"]
    assert card["verdict"] == "complete"
    assert set(card["streams"]) == {"awc-aircraft-cache", "awc-metar-cache"}
    pressure = card["streams"]["awc-metar-cache"]["variables"]["surface_pressure_pa"]["assessments"]
    assert pressure["engineering"]["verdict"] == "pass"
    assert pressure["statistical_consistency"]["o_a_rms_below_o_b_rms"] is True
    assert card["assessments"]["engineering"]["verdict"] == "pass"
    assert report["assimilation_history"]["reports_in_analysis_chain"] == len(assimilated_ids)

    resumed = tmp_path / "resumed"
    result = run(cfg, resumed, restart=analysis_path)
    assert result["status"] == "pass"
    # A restart from an analysis opens a new conservation epoch: the
    # targets are the analysis's own global means, recorded beside the
    # cold start's.
    rebased = result["restart_targets"]
    assert rebased is not None
    assert rebased["mass_target_pa"] == result["mass_target_pa"]
    assert rebased["total_water_target_kg_m2"] == result["total_water_target_kg_m2"]
    assert "cold_total_water_target_kg_m2" in rebased
    # Mass preservation held: the restart's first-step mass fixer saw only
    # roundoff, not a re-absorbed analysis mean shift.
    assert result["run_trackers"]["maximum_mass_fixer_log_offset"] <= 1.0e-6
    # The chain rides the forecast: the checkpoint the run writes after
    # integrating from the analysis still carries it unchanged.
    forecast = sorted(resumed.glob("arwen_global_step*.npz"))[-1]
    forecast_metadata, _ = read_checkpoint(forecast)
    assert forecast_metadata["step"] > metadata["step"]
    assert forecast_metadata["physics_metadata"][ASSIMILATION_HISTORY_KEY] == chain


def test_gate_of_record_fails_when_obs_disagree_with_themselves(spun_up, tmp_path):
    """Co-located contradictory pairs: whichever twin is withheld, the
    analysis follows the other one and predicts the withheld report
    worse than the background did (withheld O-B 800 Pa -> O-A 852 Pa on
    the smoke state), and the gate must report failure.  The assimilated
    rows alone would now pass (their twins' absence leaves a small
    increment that fits them, 800 -> 796 Pa), which is the diagnostic's
    weakness the withheld set exists for."""
    cfg, checkpoint = spun_up
    space, state = _model_space(cfg, checkpoint)
    time = "2026-08-31T12:00:00Z"
    rng = np.random.default_rng(7)
    lats = rng.uniform(-60, 60, 60)
    lons = rng.uniform(0, 360, 60)
    hx = space.hx_surface_pressure(state.atmosphere, _Family([
        ObsRow(
            source="probe", station_id=f"P{k}", latitude_deg=lats[k],
            longitude_deg=lons[k], elevation_m=0.0, level_pa=None,
            valid_time=dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc),
            variable="surface_pressure_pa", value=0.0, error=1.0,
        )
        for k in range(60)
    ]))
    lines = [METAR_HEADER]
    for k in range(60):
        for tag, offset in (("A", 800.0), ("B", -800.0)):
            lines.append(_metar_line(
                f"X{k:03d}{tag}", time, lats[k], lons[k],
                altim_in_hg=f"{(hx[k] + offset) / INHG_TO_PA:.8f}",
                elevation_m="0",
            ))
    path = tmp_path / "contradictory.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    code = cli_main([
        "assimilate", CONFIG, str(checkpoint),
        "--obs", str(path), "--out", str(tmp_path / "out"),
        "--length-scale-km", "4000",
    ])
    assert code == 1
    report = json.loads((tmp_path / "out" / "assimilation-report.json").read_text())
    assert report["status"] == "fail"
    assert "surface_pressure_pa" in report["gate_of_record"]["failed_variables"]
    entry = report["variables"]["surface_pressure_pa"]
    assert entry["withheld"]["o_minus_a"]["rms"] > entry["withheld"]["o_minus_b"]["rms"]


def _background_metar_values(cfg, checkpoint: Path, lats, lons):
    """H(background) at surface stations: ps, T, u, v."""
    space, state = _model_space(cfg, checkpoint)

    def family(variable):
        return _Family([
            ObsRow(
                source="probe", station_id=f"P{k}", latitude_deg=lats[k],
                longitude_deg=lons[k], elevation_m=0.0, level_pa=None,
                valid_time=dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc),
                variable=variable, value=0.0, error=1.0,
            )
            for k in range(len(lats))
        ])

    return (
        space.hx_surface_pressure(state.atmosphere, family("surface_pressure_pa")),
        space.hx_temperature(state.atmosphere, family("temperature_k")),
        space.hx_wind(state.atmosphere, family("wind_u_m_s"), "u"),
        space.hx_wind(state.atmosphere, family("wind_v_m_s"), "v"),
    )


def _pure_noise_metars(cfg, checkpoint: Path, path: Path, noise_scale: float) -> Path:
    """The audit's DA-5b reports: H(background) plus white noise at
    ``noise_scale`` times the decoder table's errors (1.5 K, 100 Pa,
    2.5 m/s per component) at the fixture's 64 stations.  Nothing real
    to fit."""
    lats = np.repeat(np.linspace(-60.0, 60.0, 8), 8)
    lons = np.tile(np.arange(0.0, 360.0, 45.0), 8)
    ps_bg, t_bg, u_bg, v_bg = _background_metar_values(cfg, checkpoint, lats, lons)
    rng = np.random.default_rng(11)
    lines = [METAR_HEADER]
    for k in range(lats.size):
        direction, speed_kt = _wind_to_dir_speed(
            u_bg[k] + rng.normal(0.0, 2.5 * noise_scale),
            v_bg[k] + rng.normal(0.0, 2.5 * noise_scale),
        )
        lines.append(_metar_line(
            f"NZ{k:03d}", "2026-08-31T12:00:00Z", lats[k], lons[k],
            temp_c=f"{t_bg[k] + rng.normal(0.0, 1.5 * noise_scale) - 273.15:.4f}",
            direction=f"{direction:.3f}", speed_kt=f"{speed_kt:.4f}",
            altim_in_hg=f"{(ps_bg[k] + rng.normal(0.0, 100.0 * noise_scale)) / INHG_TO_PA:.6f}",
            elevation_m="0",
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("background_error_multiplier", [1.0, 1000.0])
def test_gate_of_record_fails_pure_noise_that_the_assimilated_rows_pass(
    spun_up, tmp_path, background_error_multiplier
):
    """Audit 2026-09-01 DA-5: judged on the rows that built the
    increment, O-A < O-B holds for any positive gain - reports made of
    white noise at ten times the table errors passed on every variable
    while 3.1 K and 10.4 m/s of that noise went into the state, and the
    audit's sigma_b x1000 passed the same way.  Judged on the withheld
    tenth the analysis never saw, the same reports fit worse than the
    background and the door fails, at x1 and at x1000 alike."""
    cfg, checkpoint = spun_up
    noise = _pure_noise_metars(cfg, checkpoint, tmp_path / "noise.csv", 10.0)
    defaults = dict(AssimilationOptions().background_errors)
    options = AssimilationOptions(
        length_scale_km=4000.0,
        background_errors=tuple(
            (name, value * background_error_multiplier)
            for name, value in defaults.items()
        ),
    )
    report = assimilate(cfg, checkpoint, [str(noise)], tmp_path / "out", options=options)
    assert report["status"] == "fail"
    failed = report["gate_of_record"]["failed_variables"]
    # Which variables' withheld sixths land on the wrong side is the
    # noise draw's business (two here); that at least one does is not.
    assert len(failed) >= 1
    assert report["gate_of_record"]["breakage"]
    vacuous_passes = 0
    for variable, entry in report["variables"].items():
        assert entry["gated"] is True
        # The assimilated-row diagnostic still "improves" on noise...
        if entry["o_minus_a"]["rms"] < entry["o_minus_b"]["rms"]:
            vacuous_passes += 1
        # ...and the withheld verdict is what the door reports.
        held = entry["withheld"]
        assert entry["gate_passed"] == (held["o_minus_a"]["rms"] < held["o_minus_b"]["rms"])
        assert (variable in failed) == (not entry["gate_passed"])
    assert vacuous_passes == len(report["variables"])
    assert report["increment_maxabs"]["temperature_k"] > 1.0


def test_noise_free_reports_pass_the_withheld_gate_at_any_gain(spun_up, tmp_path):
    """The fixture's reports carry no noise, so there is nothing to
    over-fit and sigma_b x1000 is not an over-fit: in a data-density-
    normalised scheme the denominator 1 + sum(w g) is already dominated
    by sum(w g) at x1 wherever reports are dense, and the x1000 analysis
    predicts the withheld reports better, not worse (withheld T O-A
    0.463 K at x1, 0.256 K at x1000 on the smoke state).  The gate must
    pass both: over-fitting is a property of the data, which the pure-
    noise test above supplies."""
    cfg, checkpoint = spun_up
    obs_files = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    defaults = dict(AssimilationOptions().background_errors)
    withheld_t = {}
    for multiplier in (1.0, 1000.0):
        options = AssimilationOptions(
            length_scale_km=4000.0,
            background_errors=tuple(
                (name, value * multiplier) for name, value in defaults.items()
            ),
        )
        report = assimilate(
            cfg, checkpoint, [str(p) for p in obs_files],
            tmp_path / f"out-{multiplier:g}", options=options,
        )
        assert report["status"] == "pass"
        withheld_t[multiplier] = report["variables"]["temperature_k"]["withheld"]
    assert withheld_t[1.0]["ids"] == withheld_t[1000.0]["ids"]
    assert withheld_t[1000.0]["o_minus_a"]["rms"] < withheld_t[1.0]["o_minus_a"]["rms"]


def test_cycling_refuses_reports_already_in_the_background_chain(spun_up, tmp_path):
    """Audit 2026-09-01 DA-6: the same two obs files offered every 900 s
    were re-accepted at full gain by every cycle inside the 5400 s age
    window, and the temperature O-B rms against reports that never
    changed fell 1.211 -> 0.508 -> 0.420 -> 0.411 -> 0.410 K over the
    first four cycles on this tree (1.211 -> 0.030 K over seven on the
    predecessor's vertical localization).  Now the analysis carries the
    identity of every report it used; the next cycle refuses those,
    analyses only the reports it has never seen (the previous cycle's
    withheld tenth), and the cycle after that has nothing new and says
    so instead of analysing the background's own contents again."""
    cfg, checkpoint = spun_up
    obs_files = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    all_t = _Family([
        row for path in obs_files for row in load_obs(str(path))[1]
        if row.variable == "temperature_k"
    ])
    assert all_t.count == 80

    def o_minus_b_all_rows(path: Path) -> float:
        space, state = _model_space(cfg, path)
        return float(np.sqrt(np.mean(
            (all_t.value - space.hx_temperature(state.atmosphere, all_t)) ** 2
        )))

    start = dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc)
    options = AssimilationOptions(length_scale_km=4000.0)
    background = checkpoint
    reports = []
    fits = [o_minus_b_all_rows(background)]
    for cycle in range(2):
        report = assimilate(
            cfg, background, [str(p) for p in obs_files],
            tmp_path / f"cycle{cycle}", options=options,
            analysis_time=start + dt.timedelta(seconds=900 * cycle),
        )
        reports.append(report)
        background = Path(report["analysis"]["path"])
        fits.append(o_minus_b_all_rows(background))
    first, second = reports
    assert first["rejections"]["already_assimilated"] == 0
    assert first["variables"]["temperature_k"]["count"] == 72
    assert first["variables"]["temperature_k"]["withheld"]["count"] == 8
    # Cycle 1 refuses everything cycle 0 used and analyses only the
    # withheld eight, which are below the gate's minimum: no gate, and
    # nothing more is withheld.
    assert second["rejections"]["already_assimilated"] == first["assimilated_total"]
    assert second["assimilation_history"]["refused_from_chain"] == first["assimilated_total"]
    assert second["assimilation_history"]["reports_in_background_chain"] == first["assimilated_total"]
    second_t = second["variables"]["temperature_k"]
    assert second_t["count"] == 8 and second_t["gated"] is False
    assert second_t["withheld"]["count"] == 0
    assert sorted(second["assimilated_report_ids"]["temperature_k"]) == sorted(
        first["variables"]["temperature_k"]["withheld"]["ids"]
    )
    assert second["status"] == "pass"
    assert second["assimilation_history"]["cycles_in_analysis_chain"] == 2
    # The full 80-row fit: the first analysis moves it, the second only
    # by what its eight new reports say, and the third cycle is refused
    # outright, so the fit cannot collapse any further.
    assert fits[0] == pytest.approx(1.211, abs=0.01)
    assert fits[1] < 0.6 * fits[0]
    assert fits[2] < fits[1]
    with pytest.raises(ValueError, match="already in the background's assimilation chain"):
        assimilate(
            cfg, background, [str(p) for p in obs_files], tmp_path / "cycle2",
            options=options, analysis_time=start + dt.timedelta(seconds=1800),
        )
    assert not (tmp_path / "cycle2" / "assimilation-report.json").exists()
    assert o_minus_b_all_rows(background) == fits[2]


def test_wrong_config_identity_is_refused(spun_up, tmp_path):
    import dataclasses

    cfg, checkpoint = spun_up
    other = dataclasses.replace(cfg, name="a-different-experiment")
    with pytest.raises(ValueError, match="config identity"):
        assimilate(
            other, checkpoint, ["unused.csv"], tmp_path / "out",
        )
