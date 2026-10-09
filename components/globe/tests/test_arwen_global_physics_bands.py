"""The physics suite runs a latitude band at a time, and nothing moves.

The physics half-step is the band loop's outer loop (dynamics.apply_physics):
every band's exchange is built for its rows, the suite runs on it, and its
result goes where the globe's result goes.  Every claim below is on the
real step with the real suites -- the reference suite as it ships, the
native suite behind its fakes on the numpy backend -- and every one is a
byte-for-byte equality, no tolerance:

PHYS-1  the native suite through the whole shipped step, resident against
        banded, every checkpointed array and every scalar metric
PHYS-2  the same with the pinned host tier holding all three slices, so
        the band's results are written into the tier's slots band by band
PHYS-3  the component capture's records are the same vectors whatever
        the band count
PHYS-4  a whole-grid exchange handed to a suite by hand finishes itself:
        the diagnostics a harness reads are complete
PHYS-5  the merge rules refuse what they must and fold what they may
PHYS-6  the radiation size-bounding record assembled from planes is the
        same record at every band count, counts and fractions
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from woof.globe.configs_dir import config_root as _shipped_configs
from woof.globe.checkpoint import bundle_arrays
from woof.globe.config import load_config
from woof.globe.insitu.capture import CAPTURE_NAMES, ComponentCapture, attach_capture
from woof.globe.physics import banding
from woof.globe.physics.native_suite import ArwenCudaColumnSuite
from woof.globe.runner import build_model_and_cold_state
from test_arwen_global_level5_native import _fake_modules, _options

CONFIG = str(_shipped_configs() / "arwen_global_t21_baroclinic_ten_day.toml")


def _digest(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(
        array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()
    ).hexdigest()


def _inventory(model, bundle) -> dict[str, str]:
    return {
        name: _digest(value)
        for name, value in bundle_arrays(
            bundle, model.transform.backend.to_numpy
        ).items()
    }


def _scalars(metrics, prefix=""):
    out = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            out.update(_scalars(value, f"{prefix}{key}."))
        elif isinstance(value, (int, float, bool, str)) or value is None:
            out[f"{prefix}{key}"] = value
    return out


def _band_fakes():
    """The level-5 fakes with a COLUMN-LOCAL microphysics.

    The stock fake bumps one cell of whatever batch it is handed
    (``theta[0, 0, 0] += 0.125``), which is a per-batch effect no real
    scheme has: the kernels are column-local, and that is the property
    the band loop rests on.  This one warms the bottom level of every
    column by the same amount, so a band's columns get what the globe's
    columns get.
    """
    modules = _fake_modules()

    def launch_morrison(theta, *args, **kwargs):
        theta[0] += np.float32(0.125)

    modules["woof.globe.core.morrison"].launch_morrison = launch_morrison
    return modules


def _native_model(bands: int, *, park: bool = False, capture=None, integrator=None):
    cfg = replace(load_config(CONFIG), latitude_bands=bands, host_spill="off")
    if integrator is not None:
        cfg = replace(cfg, integrator=integrator)
    model, state = build_model_and_cold_state(cfg)
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=_band_fakes())
    model.physics = suite
    if capture is not None:
        attach_capture(suite, capture)
    if park:
        # The tier with every slice, attached by hand: the config's
        # physics mode is the reference suite's, whose namespace the
        # sizer prices at nothing, so the door would park nothing under
        # the native suite swapped in above.
        from woof.globe.sizing import SPILL_SLICES
        from woof.globe.spill import HostTier

        model.host_tier = HostTier(model.transform.backend.xp)
        model.host_tier.slices = list(SPILL_SLICES)
        model.spill_slices = tuple(SPILL_SLICES)
        state = model.park_persistent(state)
    return cfg, model, state


def _stepped_native(bands: int, steps: int = 2, **kwargs):
    cfg, model, state = _native_model(bands, **kwargs)
    assert model.latitude_bands == bands
    metrics = None
    for _ in range(steps):
        state, metrics = model.step(state, cfg.dt_s)
    return model, state, metrics


# ---------------------------------------------------------------- PHYS-1


@pytest.mark.parametrize("bands", [2, 4, 8])
def test_phys_1_the_native_suite_banded_is_the_native_suite_resident(bands):
    """The whole shipped step with the native suite, two steps, every
    array of the advanced bundle and every scalar metric byte-identical
    between one band and ``bands``: the tracers, the surface, the whole
    namespace (the seeded planes included), the spectral state."""
    resident_model, resident, resident_metrics = _stepped_native(1)
    banded_model, banded, banded_metrics = _stepped_native(bands)
    a = _inventory(resident_model, resident)
    b = _inventory(banded_model, banded)
    assert sorted(a) == sorted(b)
    differing = [name for name in a if a[name] != b[name]]
    assert differing == [], f"bands={bands}: {differing}"
    assert len(a) > 100
    assert _scalars(resident_metrics) == _scalars(banded_metrics)
    # The namespace metadata, which the checkpoint hashes, is the same
    # record: the size-bounding record and the call counters included.
    assert resident.physics_state.metadata == banded.physics_state.metadata


def test_phys_1_the_band_loop_actually_ran_the_bands():
    cfg, model, state = _native_model(4)
    seen = []
    original = model.physics.step

    def spy(exchange):
        seen.append(exchange.band)
        return original(exchange)

    model.physics.step = spy
    model.step(state, cfg.dt_s)
    nlat = model.transform.grid.nlat
    assert len(seen) == 8  # two Strang halves, four bands each
    assert seen[:4] == [(r.start, r.stop) for r in model.pipeline.slices()]
    assert seen[0][0] == 0 and seen[3][1] == nlat


# ---------------------------------------------------------------- PHYS-2


@pytest.mark.parametrize("bands", [1, 4])
def test_phys_2_the_band_loop_writes_the_tier_slot_by_slot(bands):
    """With every slice parked, a band's tracers, surface and namespace go
    into the tier's slots band by band, and the checkpoint the tier's
    run writes is the resident run's."""
    from woof.globe.spill import spilled

    resident_model, resident, resident_metrics = _stepped_native(1)
    parked_model, parked, parked_metrics = _stepped_native(bands, park=True)
    assert parked_model.host_tier is not None
    assert all(spilled(v) for v in parked.physics_state.arrays.values())
    assert all(spilled(v) for v in parked.atmosphere.grid_tracers().values())
    assert spilled(parked.surface.temperature_k)
    a = _inventory(resident_model, resident)
    b = _inventory(parked_model, parked)
    differing = [name for name in a if a[name] != b[name]]
    assert differing == [], f"bands={bands}, parked: {differing}"
    assert _scalars(resident_metrics) == _scalars(parked_metrics)
    if bands > 1:
        # The tier saw the bands as band stores, not whole ones.
        assert parked_model.host_tier.store_calls > 3 * (
            10 + 24 + len(parked.physics_state.arrays))


@pytest.mark.parametrize("bands", [1, 4])
def test_phys_2_the_semi_lagrangian_core_reads_the_tier_too(bands):
    """The default core gathers the grid tracers on its stencil and the
    mass fixer measures against the originals; both read the tier's slot
    as its staged copy, and the parked run is the resident run."""
    from woof.globe.spill import spilled

    resident_model, resident, resident_metrics = _stepped_native(1, integrator="sl_si")
    parked_model, parked, parked_metrics = _stepped_native(bands, park=True, integrator="sl_si")
    assert all(spilled(v) for v in parked.atmosphere.grid_tracers().values())
    a = _inventory(resident_model, resident)
    b = _inventory(parked_model, parked)
    differing = [name for name in a if a[name] != b[name]]
    assert differing == [], f"bands={bands}, parked, sl_si: {differing}"
    assert _scalars(resident_metrics) == _scalars(parked_metrics)


# ---------------------------------------------------------------- PHYS-3


@pytest.mark.parametrize("bands", [2, 8])
def test_phys_3_the_component_capture_records_the_same_vectors(bands):
    def records(count):
        cfg, model, state = _native_model(1)
        capture = ComponentCapture(model.transform.grid.quadrature_weights)
        cfg, model, state = _native_model(count, capture=capture)
        capture.begin_step()
        model.step(state, cfg.dt_s)
        return [(call, name, np.asarray(vector)) for call, name, vector in capture.drain()]

    resident = records(1)
    banded = records(bands)
    assert [(c, n) for c, n, _ in resident] == [(c, n) for c, n, _ in banded]
    assert [c for c, _, _ in resident] == [0] * 6 + [1] * 6
    for (_, name, left), (_, _, right) in zip(resident, banded):
        assert np.array_equal(left, right), name
    assert len(CAPTURE_NAMES) == resident[0][2].shape[0]


# ---------------------------------------------------------------- PHYS-4


def test_phys_4_a_whole_grid_exchange_finishes_itself():
    cfg, model, state = _native_model(1)
    exchange = model._physics_exchange(state, 5.0)
    assert exchange.band is None
    assert exchange.model_top_pa is not None and exchange.model_top_pa > 0.0
    result = model.physics.step(exchange)
    for name in (
        "mean_outgoing_longwave_w_m2", "mean_net_surface_radiation_w_m2",
        "mean_stratospheric_floor_heating_j_m2", "radiation_calls",
        "frozen_surface_columns", "maximum_native_water_residual_kg_m2",
    ):
        assert name in result.diagnostics, name
    assert result.planes == {}
    assert "radiation_size_bounding_last" in result.physics_state.metadata


def test_phys_4_the_model_top_pressure_is_read_once_over_the_globe():
    cfg, model, state = _native_model(4)
    tops = []
    original = model.physics.step

    def spy(exchange):
        tops.append(exchange.model_top_pa)
        return original(exchange)

    model.physics.step = spy
    model.step(state, cfg.dt_s)
    assert len(set(tops)) == 1
    p_top = float(np.asarray(model.grid_state(state.atmosphere, only=("p_half",))["p_half"][0], dtype=np.float32).mean())
    assert tops[0] == p_top


# ---------------------------------------------------------------- PHYS-5


def test_phys_5_the_merge_rules():
    same = [{"a": 1, "m": 2.0, "c": 3.0}, {"a": 1, "m": 5.0, "c": 4.0}]
    rules = {"m": banding.MAX, "c": banding.COUNT}
    assert banding.merge_band_scalars(same, rules) == {"a": 1, "m": 5.0, "c": 7.0}
    with pytest.raises(ValueError, match="differs between the physics bands"):
        banding.merge_band_scalars([{"a": 1}, {"a": 2}], {})
    with pytest.raises(ValueError, match="not an integer"):
        banding.merge_band_scalars([{"c": 1.5}, {"c": 2.0}], {"c": banding.COUNT})
    weighted = banding.merge_band_scalars(
        [{"g": 1.0}, {"g": 3.0}], {"g": banding.COLUMN_MEAN}, columns=[1, 3])
    assert weighted == {"g": 2.5}
    assert banding.merge_band_metadata(
        [{"k": 1, "n": 0, "s": {"x": 1}}, {"k": 1, "n": 2, "s": {"x": 9}}],
        {"n": banding.MAX, "s": banding.SKIP},
    ) == {"k": 1, "n": 2}
    with pytest.raises(ValueError, match="physics namespace metadata"):
        banding.merge_band_metadata([{"k": 1}, {"k": 2}], {})


# ---------------------------------------------------------------- PHYS-6


def test_phys_6_the_size_bounding_record_is_assembled_from_planes():
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=_fake_modules())
    rng = np.random.default_rng(7)
    nlat, nlon = 8, 6
    names = (
        "total_liquid", "total_ice", "liquid_carried", "ice_carried",
        "liquid_radiative", "ice_radiative", "liquid_carried_radiative",
        "ice_carried_radiative", "liquid_sentinel_radiative",
        "ice_sentinel_radiative",
    )
    planes = {
        suite._SIZE_BOUNDING_PLANE + name: rng.random((nlat, nlon), dtype=np.float32)
        for name in names
    }
    counts = {"liquid_cells": 3, "ice_cells": 5, "liquid_above_path_fraction": 0.1,
              "ice_above_path_fraction": 0.2, "liquid_above_radiative_fraction": 0.0,
              "ice_above_radiative_fraction": 0.0,
              "liquid_sentinel_radiative_fraction": 0.0,
              "ice_sentinel_radiative_fraction": 0.0}
    one = [{"radiation_size_bounding_last": dict(counts)}]
    two = [
        {"radiation_size_bounding_last": {**counts, "liquid_cells": 1, "ice_cells": 2}},
        {"radiation_size_bounding_last": {**counts, "liquid_cells": 2, "ice_cells": 3}},
    ]
    metadata_in = {"radiation_size_bounding_sum": None}
    last_one, sum_one = suite._size_bounding_record(one, planes, metadata_in)
    last_two, sum_two = suite._size_bounding_record(two, planes, metadata_in)
    assert last_one == last_two and sum_one == sum_two
    assert last_one["liquid_cells"] == 3 and last_one["ice_cells"] == 5
    part = np.float32(np.sum(planes[suite._SIZE_BOUNDING_PLANE + "liquid_carried"], dtype=np.float32))
    whole = np.float32(np.sum(planes[suite._SIZE_BOUNDING_PLANE + "total_liquid"], dtype=np.float32))
    assert last_one["liquid_above_path_fraction"] == float(part / np.maximum(whole, np.float32(1e-30)))
    running = {k: 2 * v for k, v in last_one.items()}
    _, total = suite._size_bounding_record(two, planes, {"radiation_size_bounding_sum": running})
    assert total["liquid_cells"] == 3 * last_one["liquid_cells"]
