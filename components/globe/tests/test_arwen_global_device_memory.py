"""The run receipt's device peak is measured at the allocator, not sampled.

The sizing model's pre-run prediction printed as "10.41 GiB device peak"
on a T533 forty-level native run whose CuPy pool reached 28.50 GiB live
(RTX 5090, 2026-09-02).  These tests pin the instrument's CPU-provable
half: the hook folds the pool's live bytes after every allocation into a
running maximum, a numpy run installs nothing and says so in its
receipt, and the prediction never masquerades as the measurement.  The
cupy number itself is proved on the card (the run's own receipt).
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from pathlib import Path
import sys

import pytest

from woof.globe import device_memory
from woof.globe.config import load_config
from woof.globe.receipt import check_receipt
from woof.globe.runner import run

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = _shipped_configs() / "arwen_global_moist_smoke.toml"


class _Block:
    def __init__(self, pool, size):
        self.pool = pool
        self.size = size

    def free(self):
        self.pool.live -= self.size


class _FakePool:
    """malloc/used_bytes/total_bytes as the CuPy default pool exposes them."""

    def __init__(self):
        self.live = 0
        self.held = 0

    def malloc(self, size):
        self.live += size
        self.held = max(self.held, self.live)
        return _Block(self, size)

    def used_bytes(self):
        return self.live

    def total_bytes(self):
        return self.held


def test_the_hook_folds_the_live_bytes_after_every_allocation():
    pool = _FakePool()
    tracker = device_memory.DevicePeakTracker(pool)
    a = tracker(4 * device_memory.GIB)
    b = tracker(3 * device_memory.GIB)
    a.free()  # the peak stood at 7 GiB between the two mallocs and the free
    c = tracker(1 * device_memory.GIB)
    b.free()
    c.free()
    assert tracker.peak_used_bytes == 7 * device_memory.GIB
    assert tracker.allocations == 3
    row = tracker.receipt()
    assert row["peak_used_bytes"] == 7 * device_memory.GIB
    assert row["peak_used_gib"] == 7.0
    assert row["used_bytes_at_end"] == 0
    assert row["mechanism"] == device_memory.MECHANISM
    assert row["allocations"] == 3


def test_a_sampled_reading_between_allocations_would_have_missed_the_peak():
    """The failure mode the hook exists for: a probe that reads the pool at
    its own points sees the exits, not the transient inside."""
    pool = _FakePool()
    tracker = device_memory.DevicePeakTracker(pool)
    before = pool.used_bytes()
    big = tracker(20 * device_memory.GIB)
    big.free()
    after = pool.used_bytes()
    assert before == after == 0
    assert tracker.peak_used_bytes == 20 * device_memory.GIB


def test_the_numpy_backend_installs_no_hook_and_imports_no_cupy(tmp_path):
    assert device_memory.start_device_peak_tracking("numpy") is None
    cupy_loaded_before = "cupy" in sys.modules
    receipt = run(load_config(SMOKE_CONFIG), tmp_path / "out")
    assert ("cupy" in sys.modules) == cupy_loaded_before
    row = receipt["device_memory"]
    assert row["backend"] == "numpy"
    assert row["mechanism"] == "not-installed"
    assert row["peak_used_bytes"] is None
    assert row["peak_used_gib"] is None
    assert row["sizing_model_device_peak_bytes"] is None
    assert "not a measurement" in row["sizing_model_note"]
    # The row survives the receipt's self-hash round trip.
    checked = check_receipt(tmp_path / "out" / "arwen-global-receipt.json")
    assert checked["device_memory"] == row


def test_the_prediction_never_reads_as_the_measurement():
    measured = device_memory.device_memory_receipt(
        None, "numpy", sizing_model_peak_bytes=None)
    assert measured["peak_used_bytes"] is None
    sentence = device_memory.device_peak_sentence(measured)
    assert "no device peak measured" in sentence
    assert "backend='numpy'" in sentence

    pool = _FakePool()
    tracker = device_memory.DevicePeakTracker(pool)
    tracker(28 * device_memory.GIB)
    row = device_memory.device_memory_receipt(
        tracker, "cupy", sizing_model_peak_bytes=10 * device_memory.GIB)
    assert row["peak_used_bytes"] == 28 * device_memory.GIB
    assert row["sizing_model_device_peak_bytes"] == 10 * device_memory.GIB
    sentence = device_memory.device_peak_sentence(row)
    assert sentence.startswith("memory: 28.00 GiB device peak measured at the allocator")
    assert "calibrated model predicted 10.00 GiB" in sentence
    assert "calibrated model" in row["sizing_model_note"]


def test_uninstall_without_install_is_a_no_op():
    tracker = device_memory.DevicePeakTracker(_FakePool())
    tracker.uninstall()  # never installed: nothing to restore, nothing raised
    assert tracker.allocations == 0


@pytest.mark.parametrize("backend", ["numpy"])
def test_only_cupy_gets_a_tracker(backend):
    assert device_memory.start_device_peak_tracking(backend) is None


def test_the_numpy_backend_gets_a_no_op_plan_cache_restore():
    cupy_loaded_before = "cupy" in sys.modules
    restore = device_memory.disable_fft_plan_cache("numpy")
    assert restore() is None
    assert ("cupy" in sys.modules) == cupy_loaded_before


def test_the_hook_chains_onto_the_allocator_it_found_installed(monkeypatch):
    """An outer probe hooked the same way keeps seeing every allocation
    while the tracker is installed, and gets the allocator back after.
    (A tracker that called the pool directly read a 0.00 GiB peak on such
    a probe for a run that allocated tens of GiB through it.)"""
    import types

    pool = _FakePool()
    seen = []

    def outer(size):
        seen.append(size)
        return pool.malloc(size)

    slot = {"allocator": outer}
    fake_cuda = types.SimpleNamespace(
        get_allocator=lambda: slot["allocator"],
        set_allocator=lambda fn: slot.__setitem__("allocator", fn),
    )
    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(cuda=fake_cuda))
    tracker = device_memory.DevicePeakTracker(pool).install()
    assert slot["allocator"] is tracker
    block = slot["allocator"](2 * device_memory.GIB)
    assert seen == [2 * device_memory.GIB]
    assert tracker.peak_used_bytes == 2 * device_memory.GIB
    block.free()
    tracker.uninstall()
    assert slot["allocator"] is outer
    with pytest.raises(RuntimeError):
        device_memory.DevicePeakTracker(pool).install().install()
