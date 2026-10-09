from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
import hashlib
import json
import sys

import numpy as np
import pytest

from woof.globe.config import load_config
from woof.globe.constants import NUMBER_MOMENTS, WATER_SPECIES
from woof.globe.export import export_parent
from woof.globe.spectral.sampling import regular_latlon_coordinates
from woof.globe.regional.artifact import (
    canonical,
    file_hash,
    read_parent_series,
    read_regional_frame,
    read_regional_target,
    write_parent_series,
    write_regional_target,
)
from woof.globe.regional.interpolation import (
    log_pressure_interpolate,
    periodic_bilinear,
    side_tables,
    standard_lapse_theta_below,
)
from woof.globe.regional.runtime import (
    attach_parent_series,
    build_lateral_boundaries_from_parent_series,
    install_regional_initial_frame,
)
from woof.globe.regional.translate import translate_parent_to_regional_frame
from woof.globe.runner import run


CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


_KAPPA = 287.0 / 1004.0
_BASE_SURFACE_PRESSURE_PA = 100_000.0


def _base_state(cfg):
    """Hydrostatic base state for the fixture target, surface-to-top.

    ``thb`` must be a real profile, not the WRF ``T0`` offset: with
    ``thb == 300`` the coupled-theta reference ``chm*(theta - T0)`` and the
    base-state form ``chm*(theta - thb)`` are numerically identical and no
    assertion can separate them.
    """
    a = np.asarray(cfg.a_half_pa, np.float64)
    b = np.asarray(cfg.b_half, np.float64)
    p_top = float(a[0])
    mub = _BASE_SURFACE_PRESSURE_PA - p_top
    pb_half = a + b * _BASE_SURFACE_PRESSURE_PA          # top-to-surface
    pb_full = np.sqrt(pb_half[:-1] * pb_half[1:])
    # ICAO standard atmosphere, troposphere then the 216.65 K isothermal layer.
    tb = np.maximum(
        288.15 * (pb_full / 101_325.0) ** (287.0 * 0.0065 / 9.80665), 216.65
    )
    thb = tb * (100_000.0 / pb_full) ** _KAPPA
    alb = 287.0 * thb * (pb_full / 100_000.0) ** _KAPPA / pb_full
    dpb = np.diff(pb_half)                                # top-to-surface, > 0
    phb = np.zeros(a.size)
    for k in range(dpb.size):                             # surface-to-top
        phb[k + 1] = phb[k] + alb[::-1][k] * dpb[::-1][k]
    return mub, thb[::-1], phb


def _target_arrays(cfg, ny=8, nx=10):
    lat_1d = np.linspace(28.0, 44.0, ny)
    lon_1d = np.linspace(-112.0, -88.0, nx)
    lon, lat = np.meshgrid(lon_1d, lat_1d)
    nz = len(cfg.a_half_pa) - 1
    mub, thb, phb = _base_state(cfg)
    return {
        "latitude_deg": lat,
        "longitude_deg": lon,
        "terrain_height_m": np.zeros((ny, nx)),
        "cosa": np.ones((ny, nx)),
        "sina": np.zeros((ny, nx)),
        "a_half_pa": np.asarray(cfg.a_half_pa),
        "b_half": np.asarray(cfg.b_half),
        "mub2d": np.full((ny, nx), mub),
        "c1h": np.ones(nz),
        "c2h": np.zeros(nz),
        "c1f": np.ones(nz + 1),
        "c2f": np.zeros(nz + 1),
        "thb": thb,
        "phb": phb,
        "msft": np.ones((ny, nx)),
        "msfu": np.ones((ny, nx + 1)),
        "msfv": np.ones((ny + 1, nx)),
    }


def _build_artifacts(tmp_path):
    cfg = load_config(CONFIG)
    run_dir = tmp_path / "global"
    run(cfg, run_dir)
    target = tmp_path / "target.npz"
    write_regional_target(
        target, _target_arrays(cfg), name="unit-target", grid_id="unit-grid"
    )
    exports = []
    frames = []
    for step in (0, 2):
        parent = tmp_path / f"parent-{step}.npz"
        export_parent(
            cfg,
            run_dir / f"arwen_global_step{step:08d}.npz",
            parent,
            nlat=17,
            nlon=36,
        )
        frame = tmp_path / f"frame-{step}.npz"
        translate_parent_to_regional_frame(parent, target, frame)
        meta, _ = read_regional_frame(frame)
        exports.append(parent)
        frames.append((frame, meta))
    series = tmp_path / "series.json"
    target_meta, _ = read_regional_target(target)
    write_parent_series(
        series,
        frames,
        target_path=target,
        target_self_sha256=target_meta["self_sha256"],
    )
    return cfg, target, exports, frames, series


def _install_fake_lbc_module(monkeypatch):
    @dataclass(frozen=True)
    class SideBoundary:
        value: object
        tendency: object

    @dataclass(frozen=True)
    class FieldBoundary:
        west: object
        east: object
        south: object
        north: object

    @dataclass(frozen=True)
    class BoundaryInterval:
        start_seconds: float
        end_seconds: float
        fields: object

    @dataclass(frozen=True)
    class LateralBoundaries:
        intervals: tuple
        spec_bdy_width: int = 5
        spec_zone: int = 1
        relax_zone: int = 4

    def attach(state, boundaries):
        state.attached_boundaries = boundaries
        state.elapsed_seconds = 0.0

    ingest = ModuleType("woof.ingest")
    ingest.__path__ = []
    lbc = ModuleType("woof.ingest.lateral_bc")
    for name, value in locals().copy().items():
        if name in {
            "SideBoundary", "FieldBoundary", "BoundaryInterval",
            "LateralBoundaries",
        }:
            setattr(lbc, name, value)
    lbc.attach_lateral_boundaries = attach
    lbc.attach_streaming_lateral_boundaries = attach
    monkeypatch.setitem(sys.modules, "woof.ingest", ingest)
    monkeypatch.setitem(sys.modules, "woof.ingest.lateral_bc", lbc)


def test_periodic_interpolation_and_boundary_outermost_order():
    lat = np.array([-10.0, 10.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    field = np.cos(np.deg2rad(lon))[None, :] * np.ones((2, 1))
    value = periodic_bilinear(lat, lon, field, np.array([[0.0]]), np.array([[359.0]]))
    assert value[0, 0] > 0.98

    source = np.arange(3 * 6 * 8).reshape(3, 6, 8)
    sides = side_tables(source, 2)
    assert np.array_equal(sides["west"], source[:, :, :2])
    assert np.array_equal(sides["east"], source[:, :, -2:][:, :, ::-1])
    assert np.array_equal(sides["north"], source[:, -2:, :][:, ::-1, :])


def test_global_export_translates_to_exact_arwen_coupled_units(tmp_path):
    cfg, target, _exports, frames, _series = _build_artifacts(tmp_path)
    target_meta, target_arrays = read_regional_target(target)
    metadata, arrays = read_regional_frame(frames[0][0])
    nz = len(cfg.a_half_pa) - 1
    assert arrays["u"].shape == (nz, 8, 11)
    assert arrays["v"].shape == (nz, 9, 10)
    assert arrays["w"].shape == (nz + 1, 8, 10)
    chm = target_arrays["c1h"][:, None, None] * arrays["dry_mu"][None] + target_arrays["c2h"][:, None, None]
    thb = target_arrays["thb"][:, None, None]
    # The fixture's base-state theta is a profile, so the WRF T0 offset and the
    # base state are separately observable here.
    assert np.max(np.abs(thb - 300.0)) > 1.0
    assert np.allclose(arrays["coupled__theta"], chm * (arrays["theta"] - 300.0))
    assert not np.allclose(arrays["coupled__theta"], chm * (arrays["theta"] - thb))
    assert np.allclose(arrays["thp"], arrays["theta"] - thb)
    assert np.allclose(arrays["coupled__qv"], chm * arrays["qv"])
    assert metadata["target_self_sha256"] == target_meta["self_sha256"]
    assert "standard-lapse" in metadata["methods"]["vertical"]
    assert "zeroth-order-outer-faces" in metadata["methods"]["wind"]
    assert set(metadata["extrapolated_fraction"]) == {
        "below_parent_bottom", "above_parent_top"
    }


def test_parent_series_builds_existing_lbc_contract_and_installs_transactionally(tmp_path, monkeypatch):
    _cfg, target, _exports, frames, series = _build_artifacts(tmp_path)
    _install_fake_lbc_module(monkeypatch)
    boundaries, payload, resolved = build_lateral_boundaries_from_parent_series(
        series, spec_bdy_width=3, spec_zone=1, relax_zone=2
    )
    assert len(boundaries.intervals) == 1
    assert set(boundaries.intervals[0].fields) == {"u", "v", "theta", "phi", "mu", "qv"}
    assert boundaries.intervals[0].fields["u"].west.value.shape[-1] == 3
    assert payload["self_sha256"]
    assert len(resolved) == 2

    _meta, frame = read_regional_frame(frames[0][0])
    state = SimpleNamespace(elapsed_seconds=-1.0)
    for name in ("u", "v", "w", "thp", "php", "mup", *WATER_SPECIES, *NUMBER_MOMENTS):
        state_value = frame[name if name in frame else name]
        setattr(state, name, np.zeros(state_value.shape, np.float32))
    state.p = np.zeros(frame["pressure"].shape, np.float32)
    state.alt = np.zeros(frame["density"].shape, np.float32)
    install = install_regional_initial_frame(state, frames[0][0], target)
    assert install["status"] == "installed"
    assert np.allclose(state.thp, frame["thp"])
    assert np.allclose(state.p, frame["pressure"])
    # state.alt is dry alpha_d, not the moist inverse density.
    assert np.allclose(state.alt, frame["dry_inverse_density"])
    total_water = sum(frame[name] for name in WATER_SPECIES)
    assert np.allclose(
        frame["dry_inverse_density"], (1.0 + total_water) / frame["density"]
    )
    assert total_water.max() > 1.0e-4
    assert not np.allclose(state.alt, 1.0 / frame["density"])
    # The receipt describes the state that exists, before and after attach.
    assert install["model_time_s"] == state.elapsed_seconds
    assert install["parent_time_s"] == float(frames[0][1]["time_s"])
    attach = attach_parent_series(
        state, series, spec_bdy_width=3, spec_zone=1, relax_zone=2
    )
    assert attach["status"] == "attached"
    assert state.attached_boundaries is not None
    assert install["model_time_s"] == state.elapsed_seconds


def test_install_receipt_model_time_agrees_with_the_state_clock(tmp_path, monkeypatch):
    _cfg, target, _exports, frames, series = _build_artifacts(tmp_path)
    _install_fake_lbc_module(monkeypatch)
    later_path, later_meta = frames[1]
    assert float(later_meta["time_s"]) > 0.0
    _meta, frame = read_regional_frame(later_path)
    state = SimpleNamespace(elapsed_seconds=-1.0)
    for name in ("u", "v", "w", "thp", "php", "mup", *WATER_SPECIES, *NUMBER_MOMENTS):
        setattr(state, name, np.zeros(frame[name].shape, np.float32))
    receipt = tmp_path / "install.json"
    install = install_regional_initial_frame(
        state, later_path, target, receipt_path=receipt
    )
    assert install["parent_time_s"] == float(later_meta["time_s"])
    assert install["model_time_s"] == state.elapsed_seconds
    attach_parent_series(state, series, spec_bdy_width=3, spec_zone=1, relax_zone=2)
    assert install["model_time_s"] == state.elapsed_seconds


def test_parent_series_binds_every_frame_to_its_declared_target(tmp_path):
    cfg, target, _exports, frames, series = _build_artifacts(tmp_path)
    other = tmp_path / "other-target.npz"
    write_regional_target(
        other, _target_arrays(cfg), name="other-target", grid_id="unit-grid"
    )
    other_frame = tmp_path / "other-frame.npz"
    translate_parent_to_regional_frame(_exports[1], other, other_frame)
    other_meta, _ = read_regional_frame(other_frame)
    target_meta, _ = read_regional_target(target)
    assert other_meta["target_self_sha256"] != target_meta["self_sha256"]

    with pytest.raises(ValueError, match="different target"):
        write_parent_series(
            tmp_path / "mixed.json",
            [frames[0], (other_frame, other_meta)],
            target_path=target,
            target_self_sha256=target_meta["self_sha256"],
        )

    # A series written without the binding must still be refused at read time.
    payload = json.loads(series.read_text(encoding="utf-8"))
    payload.pop("self_sha256")
    row = payload["frames"][1]
    row["path"] = other_frame.name
    row["file_sha256"] = file_hash(other_frame)
    row["frame_self_sha256"] = other_meta["self_sha256"]
    row["source_parent_self_sha256"] = other_meta["source_parent_self_sha256"]
    row["target_self_sha256"] = payload["target_self_sha256"]
    payload["self_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    mixed = tmp_path / "mixed-read.json"
    mixed.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="different target"):
        read_parent_series(mixed)


def test_target_refuses_a_mub2d_its_own_coefficients_cannot_carry(tmp_path):
    cfg = load_config(CONFIG)
    arrays = _target_arrays(cfg)
    arrays["mub2d"] = np.full_like(arrays["mub2d"], 5_000.0)
    path = tmp_path / "bad-mub.npz"
    write_regional_target(path, arrays, name="unit-target", grid_id="unit-grid")
    with pytest.raises(ValueError, match="base surface pressure outside"):
        read_regional_target(path)


def test_translate_refuses_a_base_column_mass_that_is_not_this_column(tmp_path):
    cfg = load_config(CONFIG)
    run_dir = tmp_path / "global"
    run(cfg, run_dir)
    parent = tmp_path / "parent.npz"
    export_parent(
        cfg, run_dir / "arwen_global_step00000000.npz", parent, nlat=17, nlon=36
    )
    arrays = _target_arrays(cfg)
    # 60.1 kPa base surface pressure: admissible on its own, but 39.8 kPa away
    # from this column's dry mass, far past any observed pressure anomaly.
    arrays["mub2d"] = np.full_like(arrays["mub2d"], 60_000.0)
    path = tmp_path / "shifted-mub.npz"
    write_regional_target(path, arrays, name="unit-target", grid_id="unit-grid")
    with pytest.raises(ValueError, match="mu' exceeds"):
        translate_parent_to_regional_frame(parent, path, tmp_path / "frame.npz")


def test_poleward_target_latitude_is_refused_not_clamped():
    lat = regular_latlon_coordinates(17, 36, include_poles=False)[0]
    lon = regular_latlon_coordinates(17, 36, include_poles=False)[1]
    field = np.broadcast_to(np.asarray(lat)[:, None], (len(lat), len(lon)))
    inside = periodic_bilinear(
        lat, lon, field, np.array([[80.0]]), np.array([[10.0]])
    )
    assert abs(float(inside[0, 0]) - 80.0) < 1.0e-9
    with pytest.raises(ValueError, match="outside the parent grid"):
        periodic_bilinear(lat, lon, field, np.array([[89.9]]), np.array([[10.0]]))


def test_non_monotonic_parent_latitude_is_refused():
    lat = np.array([-80.0, -10.0, -40.0, 80.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    field = np.array([0.0, 1.0, 0.0, 1.0])[:, None] * np.ones((1, 4))
    with pytest.raises(ValueError, match="strictly monotonic"):
        periodic_bilinear(lat, lon, field, np.array([[-30.0]]), np.array([[0.0]]))


def test_theta_below_the_parent_bottom_follows_the_standard_lapse():
    source_p = np.array([200.0, 500.0, 850.0, 985.0])[:, None, None] * 100.0
    theta = np.array([400.0, 330.0, 300.0, 289.0])[:, None, None]
    target_p = np.array([1000.0, 1010.0, 1030.0])[:, None, None] * 100.0
    continued = standard_lapse_theta_below(
        source_p, theta, target_p,
        reference_pressure_pa=100_000.0, kappa=287.0 / 1004.0,
        gas_constant=287.0, gravity=9.81,
    )
    values = log_pressure_interpolate(
        source_p, theta, target_p, bottom_values=continued
    )
    kappa = 287.0 / 1004.0
    temperature = values[:, 0, 0] * (target_p[:, 0, 0] / 100_000.0) ** kappa
    bottom_t = 289.0 * (98_500.0 / 100_000.0) ** kappa
    depth_m = (287.0 * bottom_t / 9.81) * np.log(target_p[:, 0, 0] / 98_500.0)
    lapse = (temperature - bottom_t) / depth_m
    assert np.allclose(lapse, 6.5e-3, rtol=1.0e-6)
    held = log_pressure_interpolate(source_p, theta, target_p)
    held_t = held[:, 0, 0] * (target_p[:, 0, 0] / 100_000.0) ** kappa
    assert (held_t[-1] - bottom_t) / depth_m[-1] > 9.0e-3


def test_extrapolation_past_one_parent_bottom_layer_is_refused():
    source_p = np.array([200.0, 500.0, 850.0, 985.0])[:, None, None] * 100.0
    theta = np.array([400.0, 330.0, 300.0, 289.0])[:, None, None]
    limit = float(np.log(98_500.0 / 85_000.0))
    inside = np.array([98_500.0 * np.exp(0.9 * limit)])[:, None, None]
    log_pressure_interpolate(source_p, theta, inside)
    outside = np.array([98_500.0 * np.exp(1.1 * limit)])[:, None, None]
    with pytest.raises(ValueError, match="parent bottom layer"):
        log_pressure_interpolate(source_p, theta, outside)


def test_rotation_gate_sits_at_the_float32_round_trip_bound(tmp_path):
    cfg = load_config(CONFIG)
    arrays = _target_arrays(cfg)
    angle = np.linspace(-0.6, 0.6, arrays["cosa"].size).reshape(arrays["cosa"].shape)
    arrays["cosa"] = np.cos(angle).astype(np.float32).astype(np.float64)
    arrays["sina"] = np.sin(angle).astype(np.float32).astype(np.float64)
    good = tmp_path / "f32-rotation.npz"
    write_regional_target(good, arrays, name="unit-target", grid_id="unit-grid")
    read_regional_target(good)

    arrays["cosa"] = np.cos(angle) * (1.0 + 1.0e-5)
    arrays["sina"] = np.sin(angle) * (1.0 + 1.0e-5)
    bad = tmp_path / "scaled-rotation.npz"
    write_regional_target(bad, arrays, name="unit-target", grid_id="unit-grid")
    with pytest.raises(ValueError, match="unit-normalized"):
        read_regional_target(bad)


def test_target_and_series_tampering_is_detected(tmp_path):
    _cfg, target, _exports, _frames, series = _build_artifacts(tmp_path)
    with np.load(target, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    values["latitude_deg"].flat[0] += 1.0
    with target.open("wb") as stream:
        np.savez_compressed(stream, **values)
    with pytest.raises(ValueError, match="hash/shape/dtype"):
        read_regional_target(target)
    with pytest.raises(ValueError, match="target file hash"):
        read_parent_series(series)
