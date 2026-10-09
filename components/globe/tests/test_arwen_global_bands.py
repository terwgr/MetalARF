"""Gates RED-1, SLAB-1 and MEM-2: the determinism spine and the slab.

RED-1: every global reduction of the step, computed whole and computed
band by band through the accumulators, is byte-identical at every band
count down to one latitude row per band.  That is the property the whole
scale-out rests on -- a banded, spilled or two-card run reproduces a
resident run's checkpoints with no new pin and no tolerance -- and it is
proved here on the CPU reference before a card is touched.

RED-1 refused something real on 2026-09-06 and the refusal is pinned
below: a reduction's own algorithm moves with the SHAPE it is handed
(numpy sums a ``(nlev, 1)`` column pairwise and a ``(nlev, n > 1)`` block
sequentially), so folding the level axis inside the band-local stage made
a one-row band compute a different number from the same row inside a
wider band.  The buffer keeps the level axis for exactly that reason.

SLAB-1: the two-phase allocator hands no live buffer out twice, its
blocks cover the arena exactly, and its peak is the arena rather than the
sum of what was asked for.

MEM-2: the receipt carries the fragmentation gap -- what the allocator
holds from the device over what the run has live -- so a run reports it
rather than a reader reconstructing it from a pool total taken after the
last free.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from woof.globe import device_memory
from woof.globe.bands import (
    AssociativeAccumulator,
    LatitudeAccumulator,
    PlaneAccumulator,
    SlabAllocator,
    associative_over,
    band_edges,
    band_slices,
)
from woof.globe.config import DEFAULT_DEVICE_ALLOCATOR, load_config

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = _shipped_configs() / "arwen_global_moist_smoke.toml"


# ---------------------------------------------------------------------
# The band schedule
# ---------------------------------------------------------------------


def test_band_edges_are_a_pure_function_of_nlat_and_the_count():
    assert band_edges(96, 1) == [0, 96]
    assert band_edges(96, 4) == [0, 24, 48, 72, 96]
    # Uneven: floor(k nlat / B), and the last edge is nlat exactly.
    assert band_edges(97, 4) == [0, 24, 48, 72, 97]
    for nlat in (5, 31, 96, 129, 801):
        for bands in {1, 2, 3, min(7, nlat), min(16, nlat), nlat}:
            edges = band_edges(nlat, bands)
            assert edges[0] == 0 and edges[-1] == nlat
            assert all(b > a for a, b in zip(edges, edges[1:]))
            assert sum(b - a for a, b in zip(edges, edges[1:])) == nlat


def test_more_bands_than_rows_is_refused_by_name():
    with pytest.raises(ValueError, match="would leave a band empty"):
        band_edges(8, 9)
    with pytest.raises(ValueError, match="bands must be >= 1"):
        band_edges(8, 0)


def test_a_band_that_claims_a_row_twice_is_refused():
    accumulator = LatitudeAccumulator(np, (4,), np.float64, name="probe")
    accumulator.add_band(slice(0, 3), np.zeros(3))
    with pytest.raises(ValueError, match="already"):
        accumulator.add_band(slice(2, 4), np.zeros(2))


def test_a_reduction_over_an_incomplete_buffer_is_refused():
    """An unwritten row would put whatever the allocator handed back into
    a global reduction; the reduce names the missing rows instead."""
    accumulator = LatitudeAccumulator(np, (4,), np.float64, name="probe")
    accumulator.add_band(slice(0, 3), np.zeros(3))
    with pytest.raises(ValueError, match="never written"):
        accumulator.total()
    plane = PlaneAccumulator(np, (4, 2), np.float64, name="probe")
    plane.add_band(slice(0, 2), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="never written"):
        plane.plane


def test_a_sum_may_not_be_folded_band_by_band():
    """The refusal that keeps the answer off the band count: min, max, all
    and any fold in any order; a floating-point sum does not."""
    for op in ("sum", "add", "mean", "prod"):
        with pytest.raises(ValueError, match="not exactly associative"):
            AssociativeAccumulator(np, op)
    with pytest.raises(ValueError, match="op must be one of"):
        AssociativeAccumulator(np, "median")


# ---------------------------------------------------------------------
# RED-1
# ---------------------------------------------------------------------


def _reduction_fields(nlev, nlat, nlon, dtype, seed=20260906):
    rng = np.random.default_rng(seed)

    def field(shape, scale, offset=0.0):
        return (rng.standard_normal(shape) * scale + offset).astype(dtype)

    return {
        "qv": field((nlev, nlat, nlon), 3.0e-3, 6.0e-3),
        "dp": np.abs(field((nlev, nlat, nlon), 2.0e3, 2.5e4)),
        "temperature": field((nlev, nlat, nlon), 20.0, 250.0),
        "ps": field((nlat, nlon), 2.0e3, 1.0e5),
        "speed": np.abs(field((nlev, nlat, nlon), 15.0, 20.0)),
        "weights": np.abs(field((nlat,), 0.1, 1.0)),
    }


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_red1_every_reduction_is_the_same_at_every_band_count(dtype):
    nlev, nlat, nlon = 6, 24, 48
    f = _reduction_fields(nlev, nlat, nlon, dtype)
    weighted = f["qv"] * f["dp"]
    cell = (f["weights"][:, None] / (2.0 * nlon)).astype(dtype)
    plane = np.sum(np.minimum(weighted, 0.0), axis=0)

    whole = {
        "level_water_mass": 0.5 * np.sum(
            np.mean(weighted, axis=-1) * f["weights"].astype(dtype), axis=-1
        ),
        "column_holes_created": -np.sum(plane * cell),
        "qv_negative": -np.sum(np.sum(np.minimum(weighted, 0.0), axis=-1)),
        "qv_positive": np.sum(np.sum(np.maximum(weighted, 0.0), axis=-1)),
        "temperature_min": np.min(f["temperature"]),
        "temperature_max": np.max(f["temperature"]),
        "ps_min": np.min(f["ps"]),
        "cfl_max": np.max(f["speed"]),
        "row_courant": np.max(f["speed"], axis=(0, 2)),
        "surface_plane": np.exp(f["ps"] * 1.0e-5),
    }

    # Every band count from one band to one row per band.
    for count in range(1, nlat + 1):
        water = LatitudeAccumulator(np, (nlev, nlat), dtype, name="water")
        created = PlaneAccumulator(np, (nlat, nlon), plane.dtype, name="created")
        negative = LatitudeAccumulator(np, (nlev, nlat), dtype, name="neg")
        positive = LatitudeAccumulator(np, (nlev, nlat), dtype, name="pos")
        t_min = AssociativeAccumulator(np, "min")
        t_max = AssociativeAccumulator(np, "max")
        ps_min = AssociativeAccumulator(np, "min")
        cfl = AssociativeAccumulator(np, "max")
        courant = LatitudeAccumulator(np, (nlat,), dtype, name="courant")
        surface = PlaneAccumulator(np, (nlat, nlon), dtype, name="surface")
        for rows in band_slices(nlat, count):
            band = weighted[:, rows, :]
            water.add_band(rows, np.mean(band, axis=-1))
            created.add_band(rows, plane[rows])
            negative.add_band(rows, np.sum(np.minimum(band, 0.0), axis=-1))
            positive.add_band(rows, np.sum(np.maximum(band, 0.0), axis=-1))
            t_min.add_band(f["temperature"][:, rows, :])
            t_max.add_band(f["temperature"][:, rows, :])
            ps_min.add_band(f["ps"][rows])
            cfl.add_band(f["speed"][:, rows, :])
            courant.add_band(rows, np.max(f["speed"][:, rows, :], axis=(0, 2)))
            surface.add_band(rows, np.exp(f["ps"][rows] * 1.0e-5))
        got = {
            "level_water_mass": 0.5 * water.total(f["weights"].astype(dtype)),
            "column_holes_created": -created.total(cell),
            "qv_negative": -negative.total(axis=None),
            "qv_positive": positive.total(axis=None),
            "temperature_min": t_min.total(),
            "temperature_max": t_max.total(),
            "ps_min": ps_min.total(),
            "cfl_max": cfl.total(),
            "row_courant": courant.complete(),
            "surface_plane": surface.plane,
        }
        for name, reference in whole.items():
            assert np.array_equal(reference, got[name]), (
                f"{name} moved at B={count} ({dtype.__name__})"
            )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_red1_the_level_axis_stays_in_the_buffer(dtype):
    """The refusal RED-1 actually made, pinned.

    Folding the level axis inside the band-local stage -- ``sum(axis=(0,
    2))`` per band into an ``nlat`` vector -- makes a one-row band reduce
    a ``(nlev, 1)`` column, which numpy sums pairwise where it sums a
    wider block sequentially.  MEASURED 2026-09-06: the two disagree.  The
    shipped form keeps the level axis in the resident buffer so the only
    stage whose shape moves with the band count is the row-local one.
    """
    nlev, nlat, nlon = 40, 96, 192
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((nlev, nlat, nlon)) * 1.0e-3).astype(dtype)

    folded_whole = np.sum(w, axis=(0, 2))
    folded_banded = np.empty(nlat, dtype=dtype)
    for rows in band_slices(nlat, nlat):
        folded_banded[rows] = np.sum(w[:, rows, :], axis=(0, 2))
    assert not np.array_equal(folded_whole, folded_banded), (
        "the shape-dependent fold this test pins has stopped reproducing; "
        "re-measure before relaxing the buffer's level axis"
    )

    kept_whole = np.sum(w, axis=-1)
    kept = LatitudeAccumulator(np, (nlev, nlat), dtype, name="kept")
    for rows in band_slices(nlat, nlat):
        kept.add_band(rows, np.sum(w[:, rows, :], axis=-1))
    assert np.array_equal(kept_whole, kept.complete())
    assert np.sum(kept_whole) == kept.total(axis=None)


def test_associative_over_is_the_single_band_case_of_the_fold():
    values = np.arange(12.0).reshape(3, 4)
    assert associative_over(np, "max", values) == np.max(values)
    assert associative_over(np, "min", values) == np.min(values)
    assert bool(associative_over(np, "all", np.isfinite(values)))


# ---------------------------------------------------------------------
# SLAB-1
# ---------------------------------------------------------------------


class _FakeArena:
    _next_base = 1 << 40

    def __init__(self, size):
        self.size = int(size)
        _FakeArena._next_base += 1 << 34
        self.ptr = _FakeArena._next_base


class _FakeUnowned:
    def __init__(self, ptr, size, owner, device_id):
        self.ptr = int(ptr)
        self.size = int(size)
        self.owner = owner
        self.device_id = device_id


class _FakePointer:
    def __init__(self, mem, offset):
        self.mem = mem
        self.ptr = mem.ptr + offset


class _FakeFallback:
    def __init__(self):
        self.live = 0
        self.held = 0
        self.requests = []

    def malloc(self, size):
        self.requests.append(int(size))
        self.live += int(size)
        self.held = max(self.held, self.live)
        return _FakePointer(_FakeArena(size), 0)

    def used_bytes(self):
        return self.live

    def total_bytes(self):
        return self.held

    def free_all_blocks(self):
        self.held = self.live


@pytest.fixture()
def fake_cupy(monkeypatch):
    slot = {"allocator": None}
    module = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            memory=types.SimpleNamespace(Memory=_FakeArena),
            UnownedMemory=_FakeUnowned,
            MemoryPointer=_FakePointer,
            Device=lambda: types.SimpleNamespace(id=0),
            get_allocator=lambda: slot["allocator"],
            set_allocator=lambda fn: slot.__setitem__("allocator", fn),
            runtime=types.SimpleNamespace(memGetInfo=lambda: (16 << 30, 32 << 30)),
        ),
        get_default_memory_pool=lambda: _FakeFallback(),
    )
    monkeypatch.setitem(sys.modules, "cupy", module)
    return module, slot


def _slab(arena_bytes, fallback=None, ceiling=None):
    return SlabAllocator(
        arena_bytes, ceiling_bytes=arena_bytes if ceiling is None else ceiling,
        fallback=fallback or _FakeFallback(),
    )


def test_slab1_no_live_buffer_is_handed_out_twice(fake_cupy):
    """A randomized allocate/free workload, checked for overlap at every
    point: two live blocks sharing a byte is the one failure a slab can
    have that a pool cannot."""
    slab = _slab(4 << 20)
    rng = np.random.default_rng(11)
    live = {}
    for step in range(600):
        if live and rng.random() < 0.45:
            key = list(live)[int(rng.integers(len(live)))]
            del live[key]
        else:
            size = int(rng.integers(1, 40_000))
            pointer = slab.malloc(size)
            if isinstance(pointer.mem, _FakeUnowned):
                live[step] = (pointer, pointer.ptr, pointer.mem.size)
        spans = sorted(
            (start, start + length) for _p, start, length in live.values()
        )
        for (_a0, a1), (b0, _b1) in zip(spans, spans[1:]):
            assert a1 <= b0, "two live slab blocks overlap"
        slab.audit()
    audit = slab.audit()
    assert audit["live_blocks"] == len(live)
    assert audit["arena_bytes"] == slab.total_bytes()
    assert audit["segments"] == 1


def test_slab1_the_peak_is_the_arena_not_the_sum(fake_cupy):
    slab = _slab(1 << 20)
    held = []
    for _ in range(64):
        held.append(slab.malloc(4096))
    assert slab.used_bytes() == 64 * 4096
    # Held from the device is the arena, once, however many blocks are out.
    assert slab.total_bytes() == 1 << 20
    held.clear()
    assert slab.used_bytes() == 0
    audit = slab.audit()
    # Every block coalesced back into one free run.
    assert audit["blocks"] == 1
    assert audit["free_blocks"] == 1
    assert audit["adjacent_free_pairs"] == 0


def test_slab1_a_released_block_is_reused_exactly(fake_cupy):
    slab = _slab(1 << 20)
    first = slab.malloc(8192)
    address = first.ptr
    del first
    second = slab.malloc(8192)
    assert second.ptr == address, "the step asks for the same shapes again"
    assert slab.audit()["live_blocks"] == 1


def test_slab1_neighbours_coalesce_so_the_arena_does_not_saw(fake_cupy):
    slab = _slab(1 << 20)
    blocks = [slab.malloc(16384) for _ in range(16)]
    for index in range(0, 16, 2):
        blocks[index] = None
    assert slab.audit()["adjacent_free_pairs"] == 0
    blocks = None
    audit = slab.audit()
    assert audit["blocks"] == 1 and audit["live_bytes"] == 0
    # The whole arena is one free run again, so a request for all of it fits.
    whole = slab.malloc(1 << 20)
    assert whole.mem.size == 1 << 20


def test_slab1_an_overflow_past_the_ceiling_is_counted_and_named(fake_cupy):
    fallback = _FakeFallback()
    slab = _slab(1 << 16, fallback=fallback, ceiling=1 << 16)
    inside = slab.malloc(1 << 15)
    outside = slab.malloc(1 << 20)
    assert isinstance(outside.mem, _FakeArena)  # served by the fallback
    assert slab.overflow_allocations == 1
    assert fallback.requests == [1 << 20]
    row = slab.receipt()
    assert row["allocator"] == "slab"
    assert row["overflow_allocations"] == 1
    assert row["arena_bytes"] == 1 << 16
    assert "internal fragmentation" in row["measures"]
    assert inside is not None


def test_slab1_the_arena_extends_rather_than_being_guessed(fake_cupy):
    """The door's prediction over-read a T85 run's live peak by 6.2x
    (MEASURED 2026-09-06), so the arena starts at a slice of it and grows
    to what the run asks for.  What the card holds then tracks the run."""
    fallback = _FakeFallback()
    slab = SlabAllocator(
        1 << 20, ceiling_bytes=1 << 30, fallback=fallback,
    )
    held = [slab.malloc(1 << 18) for _ in range(64)]
    assert slab.extensions >= 1
    assert slab.audit()["segments"] == slab.extensions + 1
    assert fallback.requests == [], "no request fell through while the card had room"
    assert slab.total_bytes() >= slab.used_bytes()
    row = slab.receipt()
    assert row["arena_over_slab_live_peak"] >= 1.0
    assert row["segments"] == slab.extensions + 1
    held.clear()
    assert slab.used_bytes() == 0


def test_slab1_the_arena_outlives_the_allocator_while_a_block_is_out(fake_cupy):
    """close() drops the allocator's hold; the bytes go back with the last
    block, so a run that still holds an array at teardown keeps valid
    memory under it."""
    slab = _slab(1 << 20)
    block = slab.malloc(4096)
    segment = block.mem.owner._segment
    slab.close()
    assert segment is not None and segment.size == 1 << 20
    assert block.ptr >= segment.ptr


def test_slab1_the_hook_measures_a_slab_run_as_it_measures_a_pool_run(fake_cupy):
    """The device-peak hook finds the slab through the installed
    allocator's owner, so a slab run's receipt carries a peak instead of
    the 0.00 GiB a default-pool-by-name hook reported under a swapped
    allocator."""
    _module, slot = fake_cupy
    slab = _slab(1 << 20).install()
    assert slot["allocator"] == slab.malloc
    import cupy as cp  # the fake

    assert device_memory.installed_pool(cp) is slab
    tracker = device_memory.DevicePeakTracker(device_memory.installed_pool(cp))
    held = [tracker(4096) for _ in range(8)]
    assert tracker.peak_used_bytes == 8 * 4096
    assert tracker.peak_total_bytes == 1 << 20
    row = tracker.receipt()
    assert row["held_over_live_peak"] == round((1 << 20) / (8 * 4096), 4)
    held.clear()
    slab.uninstall()


def test_the_hook_does_not_hide_the_run_s_allocator_from_a_mid_run_reader(
    fake_cupy,
):
    """``installed_pool`` resolves THROUGH the peak hook.

    The hook is installed on top of the run's allocator and is a callable
    object, so it carries no ``__self__``.  MEASURED 2026-09-06 on an RTX
    5090: with the driver async pool holding 16,000,000 live bytes under
    the hook, ``installed_pool`` returned the default pool reading 0, so
    every mid-run reader -- the block release between ingests, the door's
    reusable-pool credit, the mass flux scheme's inter-chunk release --
    named the default pool while the run spent through another.
    """
    _module, slot = fake_cupy
    slab = _slab(1 << 20).install()
    import cupy as cp  # the fake

    tracker = device_memory.DevicePeakTracker(device_memory.installed_pool(cp))
    slot["allocator"] = tracker
    assert getattr(tracker, "__self__", None) is None
    assert device_memory.installed_pool(cp) is slab
    held = [tracker(4096) for _ in range(4)]
    assert device_memory.installed_pool(cp).used_bytes() == 4 * 4096
    held.clear()
    slot["allocator"] = slab.malloc
    slab.uninstall()


# ---------------------------------------------------------------------
# MEM-2 and the door
# ---------------------------------------------------------------------


def test_mem2_the_receipt_carries_the_fragmentation_gap():
    class _Pool:
        def __init__(self):
            self.live = 0
            self.held = 0

        def malloc(self, size):
            self.live += size
            self.held = max(self.held, int(self.live * 1.25))
            return object()

        def used_bytes(self):
            return self.live

        def total_bytes(self):
            return self.held

    pool = _Pool()
    tracker = device_memory.DevicePeakTracker(pool)
    for _ in range(4):
        tracker(device_memory.GIB)
    row = device_memory.device_memory_receipt(
        tracker, "cupy", sizing_model_peak_bytes=None,
        allocator=device_memory.PoolAllocatorChoice("async"),
    )
    assert row["held_over_live_peak"] == 1.25
    assert row["held_at_live_peak_bytes"] == int(4 * device_memory.GIB * 1.25)
    assert row["allocator"]["allocator"] == "async"
    sentence = device_memory.device_peak_sentence(row)
    assert "through the async allocator" in sentence
    assert "held/live 1.250" in sentence


def test_the_first_segment_is_a_slice_of_the_prediction_under_a_card_ceiling():
    from woof.globe.bands import INITIAL_FRACTION, MIN_SEGMENT_BYTES

    first, ceiling, reason = device_memory.slab_arena_bytes(
        10 * device_memory.GIB, None
    )
    assert first == int(10 * device_memory.GIB * INITIAL_FRACTION)
    assert ceiling is None and "first segment" in reason
    first, ceiling, reason = device_memory.slab_arena_bytes(
        30 * device_memory.GIB, 12 * device_memory.GIB
    )
    assert ceiling == 12 * device_memory.GIB - device_memory.OUT_OF_POOL_RESERVE_BYTES
    assert first == int(30 * device_memory.GIB * INITIAL_FRACTION)
    assert "ceiling" in reason
    # A card too small for even the first slice clamps it to what is there.
    first, ceiling, reason = device_memory.slab_arena_bytes(
        30 * device_memory.GIB, 2 * device_memory.GIB
    )
    assert first == ceiling == (
        2 * device_memory.GIB - device_memory.OUT_OF_POOL_RESERVE_BYTES
    )
    assert "out-of-pool reserve" in reason
    # No prediction is no longer a reason to give up the slab: it starts
    # at the minimum segment and extends.
    first, ceiling, reason = device_memory.slab_arena_bytes(
        None, 12 * device_memory.GIB
    )
    assert first == MIN_SEGMENT_BYTES and "no device-peak prediction" in reason
    none, ceiling, reason = device_memory.slab_arena_bytes(
        10 * device_memory.GIB, device_memory.GIB // 2
    )
    assert none == 0 and ceiling is None and "outside any pool" in reason


def test_a_run_that_could_not_have_the_slab_says_so_rather_than_dropping_it(
    monkeypatch, fake_cupy,
):
    module, _slot = fake_cupy
    # A card with less free than the out-of-pool reserve: no arena can be
    # taken at all, so the run falls back and the receipt says why.
    monkeypatch.setattr(
        module.cuda, "runtime",
        types.SimpleNamespace(memGetInfo=lambda: (1 << 28, 32 << 30)),
    )
    chosen = device_memory.select_device_allocator(
        "slab", "cupy", predicted_peak_bytes=None
    )
    row = chosen.receipt()
    assert row["allocator"] == "default"
    assert row["requested_allocator"] == "slab"
    assert "outside any pool" in row["fallback_reason"]


def test_the_numpy_backend_selects_no_allocator():
    assert device_memory.select_device_allocator("slab", "numpy") is None
    with pytest.raises(ValueError, match="device_allocator must be one of"):
        device_memory.select_device_allocator("jemalloc", "cupy")


def test_the_allocator_is_a_door_and_leaves_the_identity(tmp_path):
    import dataclasses

    cfg = load_config(SMOKE_CONFIG)
    # The default is the CuPy pool the process already carries: MEASURED
    # 2026-09-06 on an RTX 5070 Ti it held the least over what the run had
    # live and cost the least wall at both shapes measured, so a bare run
    # is unchanged and the other two are selectable.
    assert cfg.device_allocator == DEFAULT_DEVICE_ALLOCATOR == "default"
    for name in ("default", "async", "slab"):
        moved = dataclasses.replace(cfg, device_allocator=name)
        assert moved.config_hash == cfg.config_hash, (
            "an allocator is memory, not arithmetic: a run that moves it "
            "must share a config hash, a checkpoint lineage and a receipt"
        )
    text = SMOKE_CONFIG.read_text(encoding="utf-8")
    path = tmp_path / "allocator.toml"
    path.write_text(text + '\n[memory]\ndevice_allocator = "async"\n',
                    encoding="utf-8")
    assert load_config(path).device_allocator == "async"
    assert load_config(path).config_hash == cfg.config_hash
    bad = tmp_path / "bad.toml"
    bad.write_text(text + '\n[memory]\ndevice_allocator = "jemalloc"\n',
                   encoding="utf-8")
    with pytest.raises(ValueError, match="device_allocator must be one of"):
        load_config(bad)


def test_the_cli_flag_reaches_the_config():
    from woof.globe import cli

    parser = cli.build_parser()
    args = parser.parse_args(
        ["run", str(SMOKE_CONFIG), "--outdir", "out", "--device-allocator", "async"]
    )
    assert cli._memory_lever_overrides(args)["device_allocator"] == "async"
    bare = parser.parse_args(["run", str(SMOKE_CONFIG), "--outdir", "out"])
    assert "device_allocator" not in cli._memory_lever_overrides(bare)
