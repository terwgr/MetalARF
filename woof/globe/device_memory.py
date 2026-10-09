"""The true device peak of a WOOF global run, measured at the allocator.

The number a run used to print as its "device peak" was the sizing
model's PREDICTION (:mod:`woof.globe.sizing`), and a probe that
sampled the CuPy pool at its own points read 10.41 GiB on a T533
forty-level native run whose pool reached 28.50 GiB live between those
points (2026-09-02, RTX 5090).  The prediction printed beside this
measurement is now the calibrated model's -- fitted to receipts written
by this very hook (``sizing.DEVICE_PEAK_CALIBRATION``) -- and it is
labelled as a prediction wherever it appears.  A peak can only be read where memory is
taken, so this hook sits in the allocator: it wraps the default pool's
``malloc`` and folds ``pool.used_bytes()`` after every successful
allocation into a running maximum.  Every byte the process puts on the
device through CuPy passes through here, including the Legendre tables
and the initial state, so the maximum is the run's true high-water mark
of live pool bytes.

What it measures: live bytes of the in-process CuPy default memory pool
(``used_bytes``), not the pool's held total (which includes cached free
blocks) and not the card's device-wide residency.  A numpy run installs
nothing and says so in its receipt.
"""
from __future__ import annotations

import time

GIB = 2**30

MECHANISM = "cupy-pool-malloc-hook"


def installed_pool(cp):
    """The pool whose live bytes the CURRENTLY installed allocator spends.

    ``cp.cuda.set_allocator(MemoryAsyncPool().malloc)`` leaves the default
    pool untouched, so a hook that reads ``get_default_memory_pool()``
    reports 0.00 GiB for a run that allocated tens of GiB through the
    driver pool -- the measurement reads zero exactly where the peak is
    the reason the pool was swapped.  The installed allocator is a bound
    ``malloc``, so its owner is the pool that holds the bytes.

    A run also installs the peak hook ON TOP of its allocator, and the
    hook is a callable OBJECT rather than a bound method, so it carries no
    ``__self__``.  MEASURED 2026-09-06 on an RTX 5090: with the async pool
    holding 16,000,000 live bytes under the hook, this returned the
    default ``MemoryPool`` reading 0.  Every caller mid-run -- the block
    release between ingests, the door's reusable-pool credit, the mass
    flux scheme's inter-chunk release -- was therefore still naming the
    default pool while the run spent through another.  A wrapper that
    publishes the pool it wraps is followed, so the chain resolves to the
    allocator that actually holds the bytes.
    """
    try:
        allocator = cp.cuda.get_allocator()
    except AttributeError:
        # A caller holding something that is not a full cupy module (the
        # sizing gate reads whatever `sys.modules` has, deliberately, so
        # that it never stands up a CUDA context to answer a question
        # about bytes) still gets the default pool rather than nothing.
        return cp.get_default_memory_pool()
    for _ in range(8):
        if allocator is None:
            break
        owner = getattr(allocator, "__self__", None)
        if owner is not None and hasattr(owner, "used_bytes"):
            return owner
        if hasattr(allocator, "used_bytes"):
            return allocator
        wrapped = getattr(allocator, "pool", None)
        if wrapped is not None and hasattr(wrapped, "used_bytes"):
            return wrapped
        allocator = getattr(allocator, "wrapped_allocator", None)
    return cp.get_default_memory_pool()


# The three allocators a run can spend its device bytes through.  Every
# one of them is bit-neutral: an address does not change a floating-point
# result, and the ten-step T255 checkpoint gate is run under each.
DEVICE_ALLOCATORS = ("default", "async", "slab")

# The out-of-pool tax: bytes the context, the driver and cuFFT hold on the
# card outside any pool.  MEASURED 2026-09-06, card held minus pool held:
# 0.42 / 0.44 / 0.37 / 0.73 GiB at T63 / T127 / T255 / T383.  The arena is
# sized under it so that taking the arena does not itself run the card out.
OUT_OF_POOL_RESERVE_BYTES = int(0.75 * GIB)


class PoolAllocatorChoice:
    """A CuPy pool selected by name, with the receipt row that says which.

    ``default`` leaves the process's allocator exactly as it was found
    (today's behaviour); ``async`` installs ``MemoryAsyncPool``, the
    driver's own ``cudaMallocAsync`` pool, which is what made a T533 step
    profile run at all on 2026-09-06.
    """

    def __init__(self, name: str, *, fallback_reason: str = ""):
        self.name = str(name)
        self.fallback_reason = str(fallback_reason)
        self._pool = None
        self._previous_allocator = None
        self._installed = False

    def install(self) -> "PoolAllocatorChoice":
        import cupy as cp

        if self.name == "default":
            self._pool = cp.get_default_memory_pool()
            return self
        if self.name != "async":
            raise ValueError(f"unknown pool allocator {self.name!r}")
        self._pool = cp.cuda.MemoryAsyncPool()
        self._previous_allocator = cp.cuda.get_allocator()
        cp.cuda.set_allocator(self._pool.malloc)
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        import cupy as cp

        cp.cuda.set_allocator(self._previous_allocator)
        self._installed = False

    def receipt(self) -> dict[str, object]:
        row: dict[str, object] = {
            "allocator": self.name,
            "measures": (
                "the CuPy default memory pool, the process allocator left as "
                "found" if self.name == "default"
                else "the CUDA driver's cudaMallocAsync pool (MemoryAsyncPool)"
            ),
        }
        if self.fallback_reason:
            # The run asked for an allocator it could not have.  It is
            # recorded here rather than dropped, because a run that
            # silently spent through a different allocator than the one
            # its receipt names is a measurement defect.
            row["requested_allocator"] = "slab"
            row["fallback_reason"] = self.fallback_reason
        return row


def slab_arena_bytes(
    predicted_peak_bytes: int | None, free_bytes: int | None
) -> tuple[int, int | None, str]:
    """The slab's first segment, its ceiling, and the sentence that says why.

    The first segment is a SLICE of the door's prediction, not the whole
    of it: MEASURED 2026-09-06 on an RTX 5070 Ti, the prediction over-read
    a T85 reference run's live peak by 6.2x (3.57 GiB predicted, 0.572 GiB
    live), so an arena taken at the prediction held seven times what the
    run used and the fragmentation figure the receipt reports would have
    been a sizing error wearing a fragmentation's name.  The allocator
    extends to what the run asks for instead.

    ``(0, None, reason)`` means no arena can be taken at all and the run
    falls back to a pool, with ``reason`` printed in the receipt rather
    than dropped.
    """
    from .bands import INITIAL_FRACTION, MIN_SEGMENT_BYTES

    ceiling = None
    if free_bytes is not None and int(free_bytes) > 0:
        ceiling = int(free_bytes) - OUT_OF_POOL_RESERVE_BYTES
        if ceiling < MIN_SEGMENT_BYTES:
            return 0, None, (
                f"the card reports {int(free_bytes) / GIB:.2f} GiB free, which "
                f"leaves under {MIN_SEGMENT_BYTES / GIB:.2f} GiB once the "
                f"{OUT_OF_POOL_RESERVE_BYTES / GIB:.2f} GiB the context and "
                "cuFFT hold outside any pool is set aside, so no arena could "
                "be taken"
            )
    if predicted_peak_bytes is None or int(predicted_peak_bytes) <= 0:
        first = MIN_SEGMENT_BYTES
        reason = (
            f"{first / GIB:.2f} GiB first segment: no device-peak prediction "
            "was available for this configuration, so the arena starts at the "
            "minimum segment and extends to what the run asks for"
        )
    else:
        first = max(
            MIN_SEGMENT_BYTES, int(int(predicted_peak_bytes) * INITIAL_FRACTION)
        )
        reason = (
            f"{first / GIB:.2f} GiB first segment = {INITIAL_FRACTION:.3f} x the "
            f"door's {int(predicted_peak_bytes) / GIB:.2f} GiB device-peak "
            "prediction; the arena extends to what the run asks for"
        )
    if ceiling is not None:
        if first > ceiling:
            first = ceiling
            reason = (
                f"{first / GIB:.2f} GiB first segment = the card's "
                f"{int(free_bytes) / GIB:.2f} GiB free less the "
                f"{OUT_OF_POOL_RESERVE_BYTES / GIB:.2f} GiB out-of-pool reserve"
            )
        reason += (
            f", up to a {ceiling / GIB:.2f} GiB ceiling (the card's free VRAM "
            "less the out-of-pool reserve)"
        )
    return int(first), ceiling, reason


def _free_device_bytes(cp) -> int | None:
    try:
        free, _total = cp.cuda.runtime.memGetInfo()
    except Exception:  # noqa: BLE001 - a card that will not report is not a run failure
        return None
    return int(free)


def select_device_allocator(
    name: str, backend_name: str, *, predicted_peak_bytes: int | None = None
):
    """Install the run's allocator and return it, or ``None`` for numpy."""
    key = str(name).strip().lower()
    if key not in DEVICE_ALLOCATORS:
        raise ValueError(
            f"device_allocator must be one of {DEVICE_ALLOCATORS}, got {name!r}"
        )
    if str(backend_name) != "cupy":
        return None
    import cupy as cp

    if key != "slab":
        return PoolAllocatorChoice(key).install()
    from .bands import SlabAllocator

    arena, ceiling, reason = slab_arena_bytes(
        predicted_peak_bytes, _free_device_bytes(cp)
    )
    if arena <= 0:
        return PoolAllocatorChoice("default", fallback_reason=reason).install()
    return SlabAllocator(
        arena, ceiling_bytes=ceiling, arena_reason=reason
    ).install()


class DevicePeakTracker:
    """Allocator hook: running maximum of ``pool.used_bytes()`` after each
    successful allocation through ``pool.malloc``.

    ``pool`` needs ``malloc(size)`` and ``used_bytes()``; the CuPy default
    pool is the production one, and a stand-in with the same two methods
    lets the folding logic be proved on the CPU.
    """

    def __init__(self, pool):
        self._pool = pool
        self._malloc = pool.malloc
        self.peak_used_bytes = int(pool.used_bytes())
        self.peak_total_bytes = int(pool.total_bytes())
        self.held_at_peak_bytes = int(pool.total_bytes())
        self.allocations = 0
        self.hook_seconds = 0.0
        self._previous_allocator = None
        self._installed = False

    def __call__(self, size):
        memory = self._malloc(size)
        started = time.perf_counter()
        used = self._pool.used_bytes()
        # The held total is folded beside the live total at every
        # allocation, so the receipt carries the fragmentation gap (held
        # over live) the memory gate MEM-2 reads.  Reading it only at the
        # end reports whatever the last free left behind, which at T383
        # was 4 GiB under the true maximum.
        total = self._pool.total_bytes()
        if used > self.peak_used_bytes:
            self.peak_used_bytes = int(used)
            self.held_at_peak_bytes = int(total)
        if total > self.peak_total_bytes:
            self.peak_total_bytes = int(total)
        self.allocations += 1
        self.hook_seconds += time.perf_counter() - started
        return memory

    @property
    def pool(self):
        """The allocator this hook wraps, so ``installed_pool`` resolves
        through the hook to the pool that actually holds the bytes."""
        return self._pool

    def install(self) -> "DevicePeakTracker":
        import cupy as cp

        if self._installed:
            raise RuntimeError("the device peak tracker is already installed")
        self._previous_allocator = cp.cuda.get_allocator()
        # The hook sits ON TOP of whatever allocator was installed, not
        # beside it: an external probe that wrapped the pool before the
        # run (a per-method tracer hooked the same way) keeps seeing
        # every allocation, and the tracker still folds the pool's live
        # bytes after each one.  Calling pool.malloc directly here read
        # a 0.00 GiB global peak on such a probe for a T533 run that
        # allocated tens of GiB through this hook.
        if self._previous_allocator is not None:
            self._malloc = self._previous_allocator
        cp.cuda.set_allocator(self)
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        import cupy as cp

        cp.cuda.set_allocator(self._previous_allocator)
        self._malloc = self._pool.malloc
        self._installed = False

    def receipt(self) -> dict[str, object]:
        return {
            "backend": "cupy",
            "mechanism": MECHANISM,
            "pool": type(self._pool).__name__,
            "measures": (
                "maximum of the live bytes (used_bytes) of the CuPy pool "
                f"this run's allocator spends ({type(self._pool).__name__}) "
                "after each allocation, over the whole run: tables, initial "
                "state and every step"
            ),
            "peak_used_bytes": int(self.peak_used_bytes),
            "peak_used_gib": round(self.peak_used_bytes / GIB, 3),
            "used_bytes_at_end": int(self._pool.used_bytes()),
            "pool_total_bytes_at_end": int(self._pool.total_bytes()),
            # The fragmentation gap: what the allocator holds from the
            # device over what the run has live.  MEM-2 reads
            # held_over_live_peak.
            "peak_total_bytes": int(self.peak_total_bytes),
            "peak_total_gib": round(self.peak_total_bytes / GIB, 3),
            "held_at_live_peak_bytes": int(self.held_at_peak_bytes),
            "held_over_live_peak": (
                None if self.peak_used_bytes <= 0
                else round(self.peak_total_bytes / self.peak_used_bytes, 4)
            ),
            "allocations": int(self.allocations),
            "hook_seconds": float(self.hook_seconds),
        }


def start_device_peak_tracking(backend_name: str) -> DevicePeakTracker | None:
    """Install the hook for a cupy run; ``None`` (nothing touched) otherwise."""
    if str(backend_name) != "cupy":
        return None
    import cupy as cp

    return DevicePeakTracker(installed_pool(cp)).install()


def disable_fft_plan_cache(backend_name: str):
    """Turn the CuPy cuFFT plan cache off for a cupy run; returns the restore.

    Memory only.  CuPy keeps up to 16 cuFFT plans per device and every
    cached plan keeps its work area allocated from the pool: measured
    1.15 GiB for the six-field rfft AND 1.15 GiB for its irfft at T533
    float32 (the work area equals the data; allocator probe,
    2026-09-02), so the step's distinct transform shapes left several
    GiB resident between calls, counted as live pool bytes.  With the
    cache off every call plans afresh and its work area lives only for
    the call.  cuFFT planning is deterministic -- the same parameters
    select the same kernels -- so a re-planned transform returns the
    cached plan's bits.  ``restore()`` puts the previous cache size back;
    a numpy run touches nothing and gets a no-op.
    """
    if str(backend_name) != "cupy":
        return lambda: None
    import cupy as cp

    cache = cp.fft.config.get_plan_cache()
    previous = cache.get_size()
    cache.set_size(0)

    def restore() -> None:
        cache.set_size(previous)

    return restore


def device_memory_receipt(
    tracker: DevicePeakTracker | None, backend_name: str, *,
    sizing_model_peak_bytes: int | None,
    allocator=None,
) -> dict[str, object]:
    """The receipt's ``device_memory`` row.

    ``sizing_model_peak_bytes`` is the pre-run PREDICTION the door printed
    before anything was allocated (the calibrated model of
    :mod:`woof.globe.sizing`); it rides along under its own name
    so a reader can weigh the model against the measurement, and it is
    never the number reported as the peak.
    """
    if tracker is None:
        row: dict[str, object] = {
            "backend": str(backend_name),
            "mechanism": "not-installed",
            "measures": (
                f"nothing: backend={backend_name!r} allocates no device "
                "memory, so no allocator hook was installed"
            ),
            "peak_used_bytes": None,
            "peak_used_gib": None,
        }
    else:
        row = tracker.receipt()
    row["sizing_model_device_peak_bytes"] = (
        None if sizing_model_peak_bytes is None else int(sizing_model_peak_bytes)
    )
    if allocator is not None:
        # Which allocator spent the bytes this row measures, and its own
        # arithmetic.  A peak has no meaning without it: the same run
        # under two allocators holds two different totals.
        row["allocator"] = allocator.receipt()
    row["sizing_model_note"] = (
        "prediction of woof.globe.sizing made before the run, not a "
        "measurement: the calibrated model of the allocator-measured pool "
        "peak (fitted to DEVICE_PEAK_CALIBRATION), labelled extrapolated by "
        "the door when the configuration is outside the measured domain"
    )
    return row


def device_peak_sentence(row: dict[str, object]) -> str:
    """One line for the door: the measured peak, then the prediction."""
    predicted = row.get("sizing_model_device_peak_bytes")
    predicted_text = (
        "" if predicted is None
        else f"; calibrated model predicted {predicted / GIB:.2f} GiB"
    )
    if row.get("peak_used_bytes") is None:
        return (
            f"memory: no device peak measured (backend={row.get('backend')!r}"
            f", allocator hook not installed){predicted_text}"
        )
    allocator = row.get("allocator") or {}
    named = allocator.get("allocator")
    gap = row.get("held_over_live_peak")
    gap_text = "" if gap is None else f", held/live {gap:.3f}"
    allocator_text = "" if not named else f" through the {named} allocator"
    return (
        f"memory: {row['peak_used_bytes'] / GIB:.2f} GiB device peak "
        f"measured at the allocator{allocator_text} "
        f"({row['allocations']} allocations{gap_text}, "
        f"hook {row['hook_seconds']:.2f} s){predicted_text}"
    )


__all__ = [
    "DEVICE_ALLOCATORS",
    "DevicePeakTracker",
    "MECHANISM",
    "OUT_OF_POOL_RESERVE_BYTES",
    "PoolAllocatorChoice",
    "installed_pool",
    "select_device_allocator",
    "slab_arena_bytes",
    "device_memory_receipt",
    "device_peak_sentence",
    "disable_fft_plan_cache",
    "start_device_peak_tracking",
]
