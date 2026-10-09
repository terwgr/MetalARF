"""New Tiedtke in WOOF global's native suite: the cumulus slot's second scheme.

``cumulus = "ntiedtke"`` puts the WRF v4.6.1 New Tiedtke port
(woof.globe.core.ntiedtke) where Grell-Freitas runs by default, fed through the
same step: the same entry state and forcing lanes in, the same four rates
integrated the same way, the same rain into the same accumulators.  Two
things differ and both are checked here: the scheme is handed each
column's own Gaussian spacing instead of the scalar ``dx_m`` option, and
its convective momentum pair reaches the wind on the lane YSU's du/dv
already use.  The Grell-Freitas arm is untouched by the addition, and the
closure flag stays out of a Grell-Freitas configuration's identity, so the
restart pin of a Grell-Freitas checkpoint does not move.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import io
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_engine_module, requires_gpu
from woof.globe.config import load_config
from woof.globe.constants import (
    CONVECTIVE_RAIN_ACCUMULATOR,
    EARTH_RADIUS_M,
    WATER_SPECIES,
)
from woof.globe.physics.native_options import NativePhysicsOptions
from woof.globe.physics.native_suite import (
    NATIVE_COMPONENT_ORDER,
    ArwenCudaColumnSuite,
    native_component_order,
)
from woof.globe.state import PhysicsState
from test_arwen_global_level5_native import (
    CONFIG,
    _advance,
    _exchange,
    _fake_modules,
    _options,
)
from test_arwen_global_vertical_surface_stretched import PRESSURE_BLEND_HASHES

NATIVE_SMOKE_CONFIG = str(_shipped_configs() / "arwen_global_level5_native_smoke.toml")
NTIEDTKE_ORDER = ("rrtmgp", "sfclay", "noah", "ysu", "ntiedtke", "morrison")


def _nt_options(**overrides):
    return {**_options(), "cumulus": "ntiedtke", **overrides}


class _Marks:
    """An in-situ observer that records the component names it is shown."""

    def __init__(self):
        self.names = []

    def mark(self, name, arrays, xp):
        self.names.append(name)


def test_ntiedtke_is_constructed_in_the_cumulus_slot():
    from woof.globe.physics.arwen_bridge import NativeArwenPhysicsBridge
    from woof.globe.physics.builtin_adapters import ADAPTER_NAME

    calls = []
    modules = _fake_modules(cumulus_calls=calls)

    class Refusing:
        def __init__(self, **kwargs):
            raise AssertionError("cumulus='ntiedtke' must never construct Grell-Freitas")

    modules["woof.globe.core.gf"] = SimpleNamespace(GrellFreitas=Refusing)
    exchange = _exchange()
    suite = ArwenCudaColumnSuite(
        _nt_options(cumulus_column_chunk=4_096, ntiedtke_tiedtke_closure=True),
        array_module=np, modules=modules,
    )
    marks = _Marks()
    suite.observer = marks
    result = suite.step(exchange)
    scheme = suite._runtime._cumulus
    assert type(scheme).__name__ == "NewTiedtke"
    # The one memory option bounds both schemes' column packing.
    assert scheme.column_chunk == 4_096
    assert scheme.driver is not None
    assert len(calls) == 1
    cfg = calls[0]["cfg"]
    assert cfg.cu_physics == 16
    assert cfg.ntiedtke_tiedtke_closure is True
    assert cfg.clock_dt == float(exchange.dt_s)
    assert result.diagnostics["cumulus_active"] is True
    assert result.physics_state.metadata["cumulus_updates"] == 1
    # The slot is named by the scheme that fills it, everywhere the order
    # is reported: the identity, the helper, and the in-situ marks.
    assert suite.identity["component_order"] == list(NTIEDTKE_ORDER)
    assert native_component_order(suite.options) == NTIEDTKE_ORDER
    assert marks.names == ["start", *NTIEDTKE_ORDER]
    assert NATIVE_COMPONENT_ORDER == ("rrtmgp", "sfclay", "noah", "ysu", "gf", "morrison")
    # The registration names the scheme, its spacing and its momentum.
    contract = NativeArwenPhysicsBridge(ADAPTER_NAME, _nt_options()).registration.contract
    assert "New Tiedtke" in contract["scheme_identity"]["cumulus"]
    assert "ntiedtke_tiedtke_closure" in contract["scheme_identity"]["cumulus"]
    assert any("cumulus='ntiedtke'" in row and "momentum" in row for row in contract["limitations"])


def test_the_real_adapter_takes_the_chunk_cap_and_refuses_a_bad_one():
    from woof.globe.core.ntiedtke import NewTiedtke

    assert NewTiedtke().column_chunk is None
    assert NewTiedtke(column_chunk=4_096).column_chunk == 4_096
    for bad in (0, -1, True):
        with pytest.raises(ValueError, match="column_chunk must be positive"):
            NewTiedtke(column_chunk=bad)


def test_ntiedtke_admission_is_fail_closed_and_scoped_out_of_the_gf_identity():
    with pytest.raises(
        ValueError, match="cumulus='gf'.*cumulus='ntiedtke'.*cumulus='none'"
    ):
        NativePhysicsOptions.from_mapping({**_options(), "cumulus": "kf"})
    with pytest.raises(ValueError, match="ntiedtke_tiedtke_closure must be true or false"):
        NativePhysicsOptions.from_mapping(_nt_options(ntiedtke_tiedtke_closure=1))
    options = NativePhysicsOptions.from_mapping(_nt_options())
    assert options.cumulus_enabled
    assert options.cumulus_scheme == "ntiedtke"
    assert options.ntiedtke_tiedtke_closure is False
    assert options.identity["ntiedtke_tiedtke_closure"] is False
    # The two closures are different trajectories and never share an identity.
    classic = NativePhysicsOptions.from_mapping(_nt_options(ntiedtke_tiedtke_closure=True))
    assert classic.identity != options.identity
    assert classic.identity["ntiedtke_tiedtke_closure"] is True
    # Outside the scheme the flag changes no arithmetic and is not part of
    # the identity: a Grell-Freitas (or scheme-less) configuration keeps the
    # identity it had before the flag existed, flag set or not.
    for scheme in ("gf", "none"):
        base = NativePhysicsOptions.from_mapping({**_options(), "cumulus": scheme}).identity
        assert "ntiedtke_tiedtke_closure" not in base
        flagged = NativePhysicsOptions.from_mapping(
            {**_options(), "cumulus": scheme, "ntiedtke_tiedtke_closure": True}
        ).identity
        assert flagged == base


def test_the_gf_restart_pin_holds_and_ntiedtke_never_shares_it(tmp_path):
    # The pin recorded for the Grell-Freitas native smoke config (the same
    # value tests/test_arwen_global_vertical_surface_stretched.py asserts):
    # a checkpoint written under Grell-Freitas must keep restarting after
    # New Tiedtke joined the slot.
    cfg = load_config(NATIVE_SMOKE_CONFIG)
    assert cfg.native_adapter_options["cumulus"] == "gf"
    # The build payload carries the flag at its default; the hash payload
    # (the adapter's identity of the options) does not.
    assert cfg.native_adapter_options["ntiedtke_tiedtke_closure"] is False
    assert "ntiedtke_tiedtke_closure" not in cfg.config_identity["native_adapter_options"]
    archived = replace(cfg, semi_implicit_scheme="external")
    assert archived.config_hash == PRESSURE_BLEND_HASHES[NATIVE_SMOKE_CONFIG]
    # The same config under New Tiedtke is a different trajectory, and so
    # is the classic-closure arm; neither can resume from the other's
    # checkpoint (runner.read_checkpoint compares config hashes).
    base = open(NATIVE_SMOKE_CONFIG, encoding="utf-8").read()
    marker = 'dx_m = 50000.0 }'
    assert marker in base
    hashes = {"gf": archived.config_hash}
    for label, extra in (
        ("ntiedtke", ', cumulus = "ntiedtke" }'),
        ("classic", ', cumulus = "ntiedtke", ntiedtke_tiedtke_closure = true }'),
    ):
        path = tmp_path / f"{label}.toml"
        path.write_text(base.replace(marker, "dx_m = 50000.0" + extra), encoding="utf-8")
        loaded = load_config(path)
        assert loaded.native_adapter_options["cumulus"] == "ntiedtke"
        hashes[label] = replace(loaded, semi_implicit_scheme="external").config_hash
    assert hashes["ntiedtke"] != hashes["gf"]
    assert hashes["classic"] != hashes["ntiedtke"]
    assert len(set(hashes.values())) == 3


def test_ntiedtke_momentum_pair_reaches_the_wind_on_the_pbl_lane():
    du = 3.0e-4       # m/s/s on the lowest level
    dv = -2.0e-4
    heating = 1.0e-3  # K/s theta on the lowest level

    def convection(result, atmosphere):
        result.rucuten[0] = du
        result.rvcuten[0] = dv
        result.rthcuten[0] = heating

    exchange = _exchange()
    dt = float(exchange.dt_s)
    suite = ArwenCudaColumnSuite(
        _nt_options(), array_module=np, modules=_fake_modules(cumulus=convection)
    )
    result = suite.step(exchange)
    u_in = np.asarray(exchange.u, np.float32)
    v_in = np.asarray(exchange.v, np.float32)
    # The exchange runs top-to-bottom; the scheme's level 0 is the surface.
    assert np.allclose(
        np.asarray(result.u[-1], np.float64) - u_in[-1], dt * du, atol=2.0e-6
    )
    assert np.allclose(
        np.asarray(result.v[-1], np.float64) - v_in[-1], dt * dv, atol=2.0e-6
    )
    # Nothing else moved the wind (the YSU fake's du/dv are zero).
    assert np.array_equal(np.asarray(result.u[:-1], np.float32), u_in[:-1])
    assert np.array_equal(np.asarray(result.v[:-1], np.float32), v_in[:-1])
    # The thermodynamic rates couple exactly as Grell-Freitas's do: dt *
    # rate, plus the fake Morrison's 0.125 K at column (0, 0).
    dtheta = np.asarray(result.theta[-1], np.float64) - np.asarray(exchange.theta[-1], np.float64)
    expected = np.full(dtheta.shape, dt * heating)
    expected[0, 0] += 0.125
    assert np.allclose(dtheta, expected, atol=2.0e-5)
    # A scheme returning no momentum (Grell-Freitas) leaves the wind alone.
    gf = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules()
    ).step(exchange)
    assert np.array_equal(np.asarray(gf.u, np.float32), u_in)
    assert np.array_equal(np.asarray(gf.v, np.float32), v_in)


def test_half_a_momentum_pair_is_refused_by_name():
    modules = _fake_modules()

    class Half(modules["woof.globe.core.ntiedtke"].NewTiedtke):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            result.rvcuten = None
            return result

    modules["woof.globe.core.ntiedtke"] = SimpleNamespace(NewTiedtke=Half)
    suite = ArwenCudaColumnSuite(_nt_options(), array_module=np, modules=modules)
    with pytest.raises(ValueError, match="rucuten without its partner"):
        suite.step(_exchange())


def _ring_spacing(latitude_deg):
    """The instrument's own construction of the per-column spacing."""
    lat = np.asarray(latitude_deg, np.float64)
    ny, nx = lat.shape
    rows = np.deg2rad(lat[:, 0])
    gap = np.abs(np.diff(rows))
    extent = np.empty(ny)
    extent[1:-1] = 0.5 * (gap[:-1] + gap[1:])
    extent[0] = 0.5 * gap[0] + (0.5 * np.pi - abs(rows[0]))
    extent[-1] = 0.5 * gap[-1] + (0.5 * np.pi - abs(rows[-1]))
    dy = EARTH_RADIUS_M * extent
    dx = EARTH_RADIUS_M * np.cos(rows) * (2.0 * np.pi / nx)
    return np.sqrt(dx * dy), extent


def test_ntiedtke_receives_each_columns_gaussian_spacing_and_gf_keeps_the_scalar():
    exchange = _exchange()
    calls = []
    suite = ArwenCudaColumnSuite(
        _nt_options(dx_m=41_000.0), array_module=np,
        modules=_fake_modules(cumulus_calls=calls),
    )
    suite.step(exchange)
    call = calls[0]
    lane = call["driver"]["gf_dx_column"]
    lat = np.asarray(exchange.latitude_deg, np.float64)
    ny, nx = lat.shape
    assert lane is not None
    assert lane.shape == (ny, nx) and lane.dtype == np.float32
    expected, extent = _ring_spacing(lat)
    assert np.allclose(lane[:, 0], expected, rtol=1.0e-6)
    assert np.all(lane == lane[:, :1])
    # The ring extents tile the pole-to-pole meridian exactly.
    assert math.isclose(float(extent.sum()), math.pi, rel_tol=1.0e-9)
    # It is the grid's spacing, not the option: equatorward rings are wider
    # than polar ones and no column carries the scalar.
    assert lane[ny // 2, 0] > lane[0, 0]
    assert not np.any(np.isclose(lane, 41_000.0))
    assert call["cfg"].dx == 41_000.0
    # New Tiedtke also reads the resolved condensate.
    assert {"qc", "qi", "u", "v", "temperature", "qv", "pressure", "exner",
            "rho", "dz", "p_interface"} <= set(call["atmosphere"])
    # Built once and reused: the second call hands over the same plane.
    suite.step(_advance(exchange, suite.step(exchange), 10.0))
    assert np.array_equal(calls[-1]["driver"]["gf_dx_column"], lane)
    # Grell-Freitas keeps the scalar option it was measured with.
    calls.clear()
    ArwenCudaColumnSuite(
        {**_options(), "dx_m": 41_000.0}, array_module=np,
        modules=_fake_modules(cumulus_calls=calls),
    ).step(exchange)
    assert calls[0]["driver"]["gf_dx_column"] is None
    assert calls[0]["cfg"].dx == 41_000.0


def test_a_grid_that_is_not_latitude_rings_or_reaches_a_pole_is_refused():
    exchange = _exchange()
    lat = np.asarray(exchange.latitude_deg, np.float64)
    ny, nx = lat.shape
    sideways = np.tile(np.linspace(-80.0, 80.0, nx)[None, :], (ny, 1))
    suite = ArwenCudaColumnSuite(_nt_options(), array_module=np, modules=_fake_modules())
    with pytest.raises(ValueError, match="latitude rings"):
        suite.step(replace(exchange, latitude_deg=sideways))
    polar = lat.copy()
    polar[0, :] = 90.0
    with pytest.raises(ValueError, match="sits at a pole"):
        ArwenCudaColumnSuite(
            _nt_options(), array_module=np, modules=_fake_modules()
        ).step(replace(exchange, latitude_deg=polar))
    # Grell-Freitas never builds the plane, so the same grids run under it.
    ArwenCudaColumnSuite(_options(), array_module=np, modules=_fake_modules()).step(
        replace(exchange, latitude_deg=sideways)
    )


def test_ntiedtke_state_resumes_from_a_checkpoint_bit_for_bit():
    rain = 0.02

    def convection(result, atmosphere):
        result.rthcuten[0] = 5.0e-4
        result.rqvcuten[0] = -1.0e-6
        result.rqccuten[1] = 2.0e-7
        result.rucuten[0] = 2.0e-4
        result.rvcuten[0] = -1.0e-4
        result.rainc[...] = rain

    def fresh_suite():
        return ArwenCudaColumnSuite(
            _nt_options(), array_module=np, modules=_fake_modules(cumulus=convection)
        )

    exchange = _exchange()
    # Continuous: three calls through one suite.
    continuous = fresh_suite()
    c1 = continuous.step(exchange)
    c2 = continuous.step(_advance(exchange, c1, 10.0))
    c3 = continuous.step(_advance(exchange, c2, 20.0))
    # Interrupted: two calls, the physics state through the checkpoint's
    # array-and-JSON shape (checkpoint.bundle_arrays stores every physics
    # array under physics__ and the metadata as JSON), then a fresh suite,
    # fresh runtime and fresh scheme for the third.
    interrupted = fresh_suite()
    i1 = interrupted.step(exchange)
    i2 = interrupted.step(_advance(exchange, i1, 10.0))
    buffer = io.BytesIO()
    np.savez(buffer, **{
        f"physics__{name}": np.asarray(value)
        for name, value in i2.physics_state.arrays.items()
    })
    buffer.seek(0)
    stored = np.load(buffer)
    resumed_state = PhysicsState(
        schema=i2.physics_state.schema,
        arrays={name.removeprefix("physics__"): stored[name] for name in stored.files},
        metadata=json.loads(json.dumps(i2.physics_state.metadata)),
    )
    i3 = fresh_suite().step(
        replace(_advance(exchange, i2, 20.0), physics_state=resumed_state)
    )
    for name in ("u", "v", "theta", *WATER_SPECIES):
        assert np.array_equal(np.asarray(getattr(c3, name)), np.asarray(getattr(i3, name))), name
    resumed_surface = i3.surface.arrays()
    for name, value in c3.surface.arrays().items():
        assert np.array_equal(np.asarray(value), np.asarray(resumed_surface[name])), name
    assert c3.physics_state.metadata["cumulus_updates"] == 3
    assert i3.physics_state.metadata["cumulus_updates"] == 3
    assert c3.physics_state.arrays.keys() == i3.physics_state.arrays.keys()
    for name in c3.physics_state.arrays:
        assert np.array_equal(
            np.asarray(c3.physics_state.arrays[name]),
            np.asarray(i3.physics_state.arrays[name]),
        ), name
    # The accumulator carried its two calls across the checkpoint.
    assert np.allclose(np.asarray(i3.physics_state.arrays[CONVECTIVE_RAIN_ACCUMULATOR]), 3.0 * rain, rtol=1.0e-6)
    # And the momentum reached the wind on every call, including the resumed one.
    u_in = np.asarray(exchange.u, np.float32)
    assert np.allclose(
        np.asarray(i3.u[-1], np.float64) - u_in[-1],
        3.0 * float(exchange.dt_s) * 2.0e-4, atol=5.0e-6,
    )


class _Tagged(np.ndarray):
    """An ndarray subclass that marks an array as having been converted."""


# The harness lives in `woof/verify/harness/`, which a published 2.7 does
# not carry (patch item 04).  Without the mark the import raises inside the
# body and the test reports RED for an engine that is behind rather than
# for anything this package did: measured on a clean venv against a 2.7.0
# wheel on 2026-09-09, two failures here, on a Windows host and on a Linux host.
@requires_engine_module("woof.verify.harness", "04")
def test_the_device_hosting_wrapper_moves_every_array_and_nothing_else():
    from woof.verify.harness.subjects import DeviceHostedSubject, _move

    exchange = _exchange()
    moved = _move(exchange, lambda a: np.asarray(a).view(_Tagged))
    untouched = []
    for name in ("u", "theta", "qv", "p_half", "omega_half_pa_s", "latitude_deg"):
        assert isinstance(getattr(moved, name), _Tagged), name
    for name, value in moved.surface.arrays().items():
        assert isinstance(value, _Tagged), name
    for name, value in moved.physics_state.arrays.items():
        assert isinstance(value, _Tagged), name
    assert moved.physics_state.metadata == exchange.physics_state.metadata
    assert moved.time_s == exchange.time_s and moved.dt_s == exchange.dt_s
    assert not untouched
    # Round trip through a stepped subject: the result comes back as
    # plain host arrays whatever the subject returned.
    suite = ArwenCudaColumnSuite(_nt_options(), array_module=np, modules=_fake_modules())
    seen = []

    class Recording:
        identity = suite.identity

        def step(self, ex):
            seen.append(type(ex.surface.water_kg_m2))
            # Floating arrays arrive at the native suite's admitted
            # precision; the category carriers are floats too and follow.
            assert ex.surface.water_kg_m2.dtype == np.float32
            assert ex.theta.dtype == np.float32
            assert ex.physics_state.arrays == {} or all(
                v.dtype == np.float32 for v in ex.physics_state.arrays.values()
                if np.issubdtype(v.dtype, np.floating)
            )
            return suite.step(ex)

    # A stand-in "device" module: its asarray tags what it is handed, so
    # the subject can be seen to receive device-side arrays, while the
    # host conversion hands the instruments plain ndarrays back.
    device = SimpleNamespace(
        asarray=lambda a, dtype=None: np.asarray(a, dtype=dtype).view(_Tagged)
    )
    hosted = DeviceHostedSubject(Recording(), array_module=device, to_host=np.asarray)
    result = hosted.step(exchange)
    assert seen == [_Tagged]
    assert hosted.identity["mode"] == "arwen-native"
    assert type(result.theta) is np.ndarray
    assert type(result.surface.water_kg_m2) is np.ndarray
    assert all(type(v) is np.ndarray for v in result.physics_state.arrays.values())
    assert result.diagnostics["cumulus_active"] is True


@requires_engine_module("woof.verify.harness", "04")
def test_ntiedtke_column_battery_runs_through_the_harness_seam():
    from woof.verify.harness.convection_closure import (
        PRECIPITATING,
        measure_component,
    )
    from woof.verify.harness.subjects import DeviceHostedSubject

    calls = []

    def convection(result, atmosphere):
        # Small enough that the battery's dry columns (1e-4 kg/kg at the
        # surface) stay non-negative through the two 600 s steps.
        result.rthcuten[0] = 1.0e-4
        result.rqvcuten[0] = -1.0e-8
        result.rucuten[1] = 1.0e-5
        result.rvcuten[1] = -1.0e-5
        result.rainc[...] = 1.0e-3

    suite = DeviceHostedSubject(
        ArwenCudaColumnSuite(
            _nt_options(), array_module=np,
            modules=_fake_modules(cumulus=convection, cumulus_calls=calls),
        ),
        array_module=np, to_host=np.asarray,
    )
    report = measure_component(CONFIG, physics=suite, steps=2, dt_s=600.0)
    assert report.instrument == "convection.closure"
    assert "arwen-native" in report.subject
    # One cumulus call per battery step, each fed the battery at rest
    # (zero omega on every half level) and the battery grid's own spacing.
    assert len(calls) == 2
    for call in calls:
        assert call["w"].shape[0] == call["atmosphere"]["temperature"].shape[0] + 1
        assert np.all(call["w"] == 0.0)
        assert call["driver"]["gf_dx_column"].shape == call["fields"]["xland"].shape
        assert np.all(np.isfinite(call["driver"]["gf_dx_column"]))
        assert call["cfg"].cu_physics == 16
    # The deep columns reach the scheme with their instability intact.
    assert min(report.measurements["cape_proxy_initial_j_kg"]) > 100.0
    assert math.isfinite(report.measurements["water_residual_max_kg_m2"])
    assert report.measurements["run_precip_by_latitude_kg_m2"][PRECIPITATING] > 0.0


def _recording_ntiedtke(cp, recorded):
    from woof.globe.core import ntiedtke as real

    names = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
             "rucuten", "rvcuten", "rainc")

    class Recording(real.NewTiedtke):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            recorded.append({
                **{name: cp.asnumpy(getattr(result, name)) for name in names},
                "dx": cp.asnumpy(self._driver.gf_dx_column),
            })
            return result

    return SimpleNamespace(NewTiedtke=Recording)


@pytest.mark.gpu
@requires_gpu
@requires_engine_module("woof.verify.harness", "04")
def test_the_real_scheme_returns_finite_rates_on_the_battery_deep_columns():
    cp = pytest.importorskip("cupy")
    from woof.verify.harness.convection_closure import (
        PRECIPITATING,
        measure_component,
    )
    from woof.verify.harness.subjects import DeviceHostedSubject

    def battery(column_chunk):
        recorded = []
        options = _nt_options(
            radiation_interval_s=1800.0, land_surface_interval_s=1800.0,
            **({} if column_chunk is None else {"cumulus_column_chunk": column_chunk}),
        )
        suite = DeviceHostedSubject(ArwenCudaColumnSuite(
            options, modules={"woof.globe.core.ntiedtke": _recording_ntiedtke(cp, recorded)}
        ))
        report = measure_component(CONFIG, physics=suite, steps=2)
        return report, recorded

    report, recorded = battery(None)
    assert len(recorded) == 2
    for call in recorded:
        for name, value in call.items():
            assert np.all(np.isfinite(value)), name
        assert call["dx"].shape == (5, 3)
        assert np.all(call["dx"] > 0.0)
    deep = np.stack([call["rthcuten"][:, :, PRECIPITATING] for call in recorded])
    assert np.all(np.isfinite(deep))
    assert math.isfinite(report.measurements["water_residual_max_kg_m2"])
    assert math.isfinite(report.measurements["enthalpy_residual_max_j_m2"])
    # The chunk cap moves memory and nothing else: fifteen battery columns
    # walked seven at a time (7, 7, 1 with the padded tail) reproduce the
    # single-chunk rates bit for bit.
    _, chunked = battery(7)
    assert len(chunked) == len(recorded)
    for whole, part in zip(recorded, chunked):
        for name in whole:
            assert np.array_equal(whole[name], part[name]), name
