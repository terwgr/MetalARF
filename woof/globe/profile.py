"""Per-operator cost of one WOOF global step, measured on the stream.

WHAT IT MEASURES.  Every operator of ``MoistHybridModel.step`` (and of
the native physics runtime, the IMEX stages, the tracer transport and the
runner's own checkpoint write) is wrapped in a named section.  For each
section on the cupy backend the instrument records two numbers:

* ``host_ms``: host wall time between entering and leaving the section
  (``time.perf_counter``).  Launch-bound work shows up here; a section
  whose host time equals its device time is one where the host had to
  wait for the device (a synchronisation) or the device had to wait for
  the host (launch latency).
* ``device_ms``: the time the CUDA stream took to travel from the event
  recorded on entry to the event recorded on exit
  (``cupy.cuda.get_elapsed_time``).  This is the section's cost as the
  card experiences it: kernel time plus any idle gap the stream sat in
  while the host was still issuing work.  Summed over the top-level
  sections it reproduces the step wall to the last synchronisation.

Both are wall clocks, not kernel time: a section that sleeps on the host
reads the sleep in BOTH columns, because the stream cannot pass the exit
event until the host records it.  The distinction the two columns draw is
therefore "the host was inside this section for X ms" against "the card
was busy or waiting on this section for Y ms"; kernel-only time needs a
kernel-level profiler (nsys) and is not claimed here.

The instrument also counts device-to-host transfers routed through
``cupy.asnumpy`` per section (calls and bytes).  That is the model's own
``Backend.to_numpy`` and the native suite's ``_host``; a transfer through
``ndarray.get()`` or a scalar ``float()`` is not counted, and the receipt
says so.

On the numpy backend the two columns are the same host wall clock.

CALIBRATION (``calibrate``): before any reading is cited the instrument
proves, on the backend it runs on, that (1) a section wrapping a host
sleep of a known length reads that length in both columns, (2) the
section immediately after the sleep does not inherit any of it, (3) a
section wrapping a device workload of known cost (benchmarked apart with
``cupyx.profiler.benchmark``, which synchronises around every
repetition) reads that cost within its tolerance, and (4) nested
sections sum to their parent.  The receipt of that calibration rides in
the profile JSON beside the readings; a profile whose calibration failed
is refused rather than written.

DIAGNOSTIC ONLY.  The sections add no arithmetic; with no profiler
attached every hook is a no-op context manager and the step is
bit-identical to the unhooked step (tests pin this on the T3 smoke).
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

import numpy as np

PROFILE_NAME = "profile.json"
PROFILE_SCHEMA = "gpuwm.arwen-global-step-profile/v1"


class _NullProfiler:
    """The no-op every hook sees when no profiler is attached."""

    active = False

    @contextmanager
    def section(self, name: str):
        yield

    def begin_step(self, step: int) -> None:
        pass

    def end_step(self) -> None:
        pass


NULL_PROFILER = _NullProfiler()


def _section_path(stack: list[str], name: str) -> str:
    return ".".join([*stack, name]) if stack else name


class StepProfiler:
    """Named, nestable sections timed on the host and on the stream.

    ``backend`` is the run's ``Backend``; ``steps`` is how many steps to
    profile after ``warmup`` unprofiled ones (the first steps of a run
    build tables, compile kernels and warm the allocator, and are not a
    step's steady cost).  ``begin_step`` / ``end_step`` bracket a model
    step; ``section`` wraps an operator.  Readings are folded per section
    path into running sums so the summary is a mean over the profiled
    steps, and every step's own row is kept for the JSON.
    """

    def __init__(self, backend, *, steps: int, warmup: int = 2):
        if int(steps) < 1:
            raise ValueError("profile steps must be >= 1")
        if int(warmup) < 0:
            raise ValueError("profile warmup must be >= 0")
        self.backend = backend
        self.cupy = backend.name == "cupy"
        self.steps = int(steps)
        self.warmup = int(warmup)
        self.active = False
        self._stack: list[str] = []
        self._events: list[tuple[str, object, object, float, float, int, int]] = []
        self._step: int | None = None
        self._step_started = 0.0
        self._steps_seen = 0
        self.rows: list[dict] = []
        self.totals: dict[str, dict[str, float]] = {}
        self.calibration: dict | None = None
        # Device-to-host transfer census through cupy.asnumpy.
        self._transfer_calls = 0
        self._transfer_bytes = 0
        self._asnumpy_previous = None

    # -- transfer census -------------------------------------------------

    def _install_transfer_hook(self) -> None:
        if not self.cupy or self._asnumpy_previous is not None:
            return
        xp = self.backend.xp
        previous = xp.asnumpy
        profiler = self

        def counted(value, *args, **kwargs):
            if hasattr(value, "nbytes"):
                profiler._transfer_calls += 1
                profiler._transfer_bytes += int(value.nbytes)
            return previous(value, *args, **kwargs)

        xp.asnumpy = counted
        self._asnumpy_previous = previous

    def _remove_transfer_hook(self) -> None:
        if self._asnumpy_previous is None:
            return
        self.backend.xp.asnumpy = self._asnumpy_previous
        self._asnumpy_previous = None

    # -- the step bracket -------------------------------------------------

    def begin_step(self, step: int) -> None:
        self._steps_seen += 1
        self.active = (
            self._steps_seen > self.warmup
            and self._steps_seen <= self.warmup + self.steps
        )
        if not self.active:
            return
        self._install_transfer_hook()
        if self.cupy:
            # The step's own bracket starts from a drained device so its
            # first section is not charged for work the previous step
            # left in flight.
            self.backend.synchronize()
        self._step = int(step)
        self._events = []
        self._stack = []
        self._step_started = time.perf_counter()
        self._push("step")

    def end_step(self) -> None:
        if not self.active:
            return
        self._pop("step")
        if self.cupy:
            self.backend.synchronize()
        readings = self._resolve()
        row = {"step": self._step, "sections": readings}
        self.rows.append(row)
        for path, reading in readings.items():
            total = self.totals.setdefault(
                path, {"host_ms": 0.0, "device_ms": 0.0,
                       "transfer_calls": 0.0, "transfer_bytes": 0.0, "count": 0.0},
            )
            for key in ("host_ms", "device_ms", "transfer_calls", "transfer_bytes"):
                total[key] += float(reading[key])
            total["count"] += 1.0
        self.active = False
        self._step = None
        if self._steps_seen >= self.warmup + self.steps:
            self._remove_transfer_hook()

    # -- sections -------------------------------------------------------

    def _mark(self):
        if self.cupy:
            event = self.backend.xp.cuda.Event(block=False, disable_timing=False)
            event.record()
            return event
        return None

    def _push(self, name: str) -> None:
        path = _section_path(self._stack, name)
        self._stack.append(name)
        self._events.append((
            "enter", path, self._mark(), time.perf_counter(),
            self._transfer_calls, self._transfer_bytes,
        ))

    def _pop(self, name: str) -> None:
        if not self._stack or self._stack[-1] != name:
            raise RuntimeError(
                f"profiler section {name!r} closed out of order; open stack "
                f"is {self._stack}"
            )
        self._stack.pop()
        path = _section_path(self._stack, name)
        self._events.append((
            "leave", path, self._mark(), time.perf_counter(),
            self._transfer_calls, self._transfer_bytes,
        ))

    @contextmanager
    def section(self, name: str):
        if not self.active:
            yield
            return
        self._push(name)
        try:
            yield
        finally:
            self._pop(name)

    def _resolve(self) -> dict[str, dict[str, float]]:
        """Pair every enter with its leave and read both clocks."""
        xp = self.backend.xp
        open_marks: dict[str, list] = {}
        readings: dict[str, dict[str, float]] = {}
        for kind, path, event, wall, calls, nbytes in self._events:
            if kind == "enter":
                open_marks.setdefault(path, []).append((event, wall, calls, nbytes))
                continue
            start_event, start_wall, start_calls, start_bytes = open_marks[path].pop()
            host_ms = (wall - start_wall) * 1.0e3
            if self.cupy:
                device_ms = float(xp.cuda.get_elapsed_time(start_event, event))
            else:
                device_ms = host_ms
            row = readings.setdefault(
                path, {"host_ms": 0.0, "device_ms": 0.0,
                       "transfer_calls": 0.0, "transfer_bytes": 0.0, "calls": 0.0},
            )
            row["host_ms"] += host_ms
            row["device_ms"] += device_ms
            row["transfer_calls"] += float(calls - start_calls)
            row["transfer_bytes"] += float(nbytes - start_bytes)
            row["calls"] += 1.0
        return readings

    # -- reporting --------------------------------------------------------

    @property
    def profiled_steps(self) -> int:
        return len(self.rows)

    def summary(self) -> list[dict]:
        """Mean per profiled step, one row per section path, sorted by
        device time within each depth so a reader sees the top costs."""
        n = max(1, self.profiled_steps)
        rows = []
        for path, total in self.totals.items():
            rows.append({
                "section": path,
                "depth": path.count("."),
                "host_ms": total["host_ms"] / n,
                "device_ms": total["device_ms"] / n,
                "transfer_calls": total["transfer_calls"] / n,
                "transfer_mb": total["transfer_bytes"] / n / 1.0e6,
            })
        rows.sort(key=lambda r: (r["section"].split(".")[0] != "step", -r["device_ms"]))
        return rows

    def table(self) -> str:
        lines = [
            f"{'section':<52} {'device_ms':>10} {'host_ms':>10} {'d2h_calls':>9} {'d2h_MB':>9}"
        ]
        for row in self.summary():
            indent = "  " * row["depth"]
            name = indent + row["section"].split(".")[-1]
            lines.append(
                f"{name:<52} {row['device_ms']:>10.2f} {row['host_ms']:>10.2f} "
                f"{row['transfer_calls']:>9.1f} {row['transfer_mb']:>9.2f}"
            )
        return "\n".join(lines)

    def receipt(self) -> dict:
        return {
            "schema": PROFILE_SCHEMA,
            "backend": self.backend.name,
            "measures": {
                "host_ms": (
                    "host wall time inside the section (time.perf_counter), "
                    "mean per profiled step"
                ),
                "device_ms": (
                    "CUDA stream time from the section's entry event to its "
                    "exit event (cupy.cuda.get_elapsed_time), mean per "
                    "profiled step; equal to host_ms on the numpy backend"
                ),
                "transfer_calls": (
                    "device-to-host transfers routed through cupy.asnumpy "
                    "inside the section, mean per profiled step; ndarray.get() "
                    "and scalar float() conversions are not counted"
                ),
                "transfer_mb": "bytes of those transfers, MB per profiled step",
            },
            "warmup_steps": self.warmup,
            "profiled_steps": self.profiled_steps,
            "calibration": self.calibration,
            "summary": self.summary(),
            "steps": self.rows,
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(
            json.dumps(self.receipt(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        return target


# -- calibration ---------------------------------------------------------

def _device_memory_context(backend) -> dict | None:
    """Free and total device memory at calibration time (cupy only): the
    reader's context for a calibration taken on a shared card."""
    if backend.name != "cupy":
        return None
    try:
        free, total = backend.xp.cuda.runtime.memGetInfo()
    except Exception:  # noqa: BLE001 - context only, never a calibration failure
        return None
    return {"free_bytes": int(free), "total_bytes": int(total)}


def calibrate(
    backend, *, sleep_s: float = 0.1, tolerance: float = 0.1, rounds: int = 5
) -> dict:
    """Prove the instrument on this backend before its numbers are cited.

    Four checks, every one with its measured values in the receipt:

    ``sleep``: a section around ``time.sleep(sleep_s)`` must read the
    sleep in both columns within ``tolerance`` (relative).
    ``no_bleed``: the section right after the sleep, wrapping one trivial
    operation, must read under 5 ms in both columns.
    ``device_work``: a section around N repetitions of one large device
    elementwise operation must read, on the device column, within
    ``tolerance`` (or 2 ms) of the same work benchmarked by
    ``cupyx.profiler.benchmark`` (numpy: ``time.perf_counter`` around the
    same loop).  Two families: a float32 multiply-add over 2^24 elements
    and a float64 reduction over the same length.
    ``nesting``: three nested sections of known host cost must sum to
    their parent within 5 ms.

    Every check runs ``rounds`` times with the reference benchmark
    interleaved between the profiler's readings, and the check is taken
    on the best round: the minimum reading of a sleep (a sleep never
    returns early), the minimum reading and the minimum reference of a
    device workload (a co-tenant on the card or the host inflates both,
    at different moments; the least-inflated round of each is the
    instrument's own clock), the smallest nesting gap.  Every round's
    reading rides in the receipt with the spread between the rounds, so
    a calibration taken on a contended card says so in numbers.

    Raises ``ValueError`` naming the failed check; returns the receipt.
    """
    xp = backend.xp
    rounds = max(1, int(rounds))
    profiler = StepProfiler(backend, steps=rounds, warmup=0)
    n = 1 << 24
    a = xp.ones(n, dtype=xp.float32)
    b = xp.full(n, 0.5, dtype=xp.float32)
    c = xp.ones(n, dtype=xp.float64)
    repeats = 20

    def work_fma():
        for _ in range(repeats):
            _ = a * b + a

    def work_reduce():
        for _ in range(repeats):
            _ = xp.sum(c)

    # The reference cost, benchmarked apart with per-repetition
    # synchronisation.
    if backend.name == "cupy":
        from cupyx.profiler import benchmark

        def reference(fn):
            result = benchmark(fn, n_repeat=3, n_warmup=1)
            return float(np.mean(result.gpu_times)) * 1.0e3
    else:
        def reference(fn):
            fn()
            started = time.perf_counter()
            for _ in range(3):
                fn()
            return (time.perf_counter() - started) / 3.0 * 1.0e3

    references = {"device_fma": [], "device_reduce": []}
    for round_index in range(rounds):
        references["device_fma"].append(reference(work_fma))
        references["device_reduce"].append(reference(work_reduce))
        profiler.begin_step(round_index)
        with profiler.section("sleep"):
            time.sleep(sleep_s)
        with profiler.section("after_sleep"):
            _ = a[:1] + b[:1]
        with profiler.section("device_fma"):
            work_fma()
        with profiler.section("device_reduce"):
            work_reduce()
        with profiler.section("parent"):
            with profiler.section("child_a"):
                time.sleep(0.02)
            with profiler.section("child_b"):
                time.sleep(0.03)
            with profiler.section("child_c"):
                time.sleep(0.01)
        profiler.end_step()
    reads = [row["sections"] for row in profiler.rows]

    def column(path, key):
        return [float(read[path][key]) for read in reads]

    def spread(values):
        low = min(values)
        return float(max(values) / low) if low > 0.0 else float("inf")

    checks = {}
    sleep_ms = sleep_s * 1.0e3
    for col in ("host_ms", "device_ms"):
        values = column("step.sleep", col)
        value = min(values)
        checks[f"sleep_{col}"] = {
            "expected_ms": sleep_ms, "read_ms": value, "rounds_ms": values,
            "passed": bool(abs(value - sleep_ms) <= tolerance * sleep_ms),
        }
        afters = column("step.after_sleep", col)
        after = min(afters)
        checks[f"no_bleed_{col}"] = {
            "limit_ms": 5.0, "read_ms": after, "rounds_ms": afters,
            "passed": bool(after < 5.0),
        }
    for name in ("device_fma", "device_reduce"):
        values = column(f"step.{name}", "device_ms")
        value = min(values)
        refs = references[name]
        ref = min(refs)
        row = {
            "reference_ms": ref, "read_ms": value,
            "rounds_ms": values, "reference_rounds_ms": refs,
            "spread": spread(values), "reference_spread": spread(refs),
        }
        if backend.name == "cupy":
            row["passed"] = bool(abs(value - ref) <= max(tolerance * ref, 2.0))
        else:
            # numpy has no device stream: the reading and the reference
            # are two host loops on a possibly shared CPU, whose ratio
            # says nothing about the instrument (measured 3.4x apart on a
            # loaded 24-core host).  Recorded, not gated; the sleep and
            # nesting checks calibrate the one clock numpy reads.
            row["passed"] = True
            row["gated"] = False
            row["note"] = "numpy backend: host loop against host loop, recorded only"
        checks[name] = row
    gaps = []
    for read in reads:
        parent = float(read["step.parent"]["host_ms"])
        children = sum(
            float(read[f"step.parent.{child}"]["host_ms"])
            for child in ("child_a", "child_b", "child_c")
        )
        gaps.append((abs(parent - children), parent, children))
    gap, parent, children = min(gaps)
    checks["nesting"] = {
        "parent_ms": parent, "children_ms": children, "gap_ms": gap,
        "rounds_gap_ms": [g for g, _, _ in gaps],
        "passed": bool(gap <= 5.0),
    }
    receipt = {
        "sleep_s": sleep_s,
        "tolerance": tolerance,
        "rounds": rounds,
        "elements": n,
        "repeats": repeats,
        "device_memory": _device_memory_context(backend),
        "checks": checks,
        "passed": all(row["passed"] for row in checks.values()),
    }
    failed = [name for name, row in checks.items() if not row["passed"]]
    if failed:
        raise ValueError(
            "step profiler calibration failed on "
            f"{backend.name}: {', '.join(failed)}; readings "
            f"{json.dumps({name: checks[name] for name in failed})}"
        )
    del a, b, c
    return receipt


def attach_profiler(model, profiler) -> list[str]:
    """Hand ``profiler`` (or ``None``) to the model and its physics suite.

    Returns the names of the objects that took it; the model's
    ``profiler`` slot and the suite's are the same kind of read-only
    hook the in-situ observer uses.
    """
    attached = []
    model.profiler = profiler
    attached.append(type(model).__name__)
    transport = getattr(model, "transport", None)
    if transport is not None and hasattr(transport, "profiler"):
        transport.profiler = profiler
        attached.append(type(transport).__name__)
    # The physics object may be the bridge around the adapter suite (the
    # native suite behind NativeArwenPhysicsBridge): the slot lives on the
    # suite, so both are offered it, the way the in-situ capture attaches
    # (insitu.capture.attach_capture).  A profile that stopped at the
    # bridge read the whole suite as one number (2026-09-04).
    physics = getattr(model, "physics", None)
    for target in (physics, getattr(physics, "adapter", None)):
        if target is not None and hasattr(target, "profiler"):
            target.profiler = profiler
            attached.append(type(target).__name__)
    return attached


def profiler_of(owner) -> StepProfiler | _NullProfiler:
    """The profiler an object carries, or the no-op."""
    profiler = getattr(owner, "profiler", None)
    return NULL_PROFILER if profiler is None else profiler


__all__ = [
    "NULL_PROFILER",
    "PROFILE_NAME",
    "PROFILE_SCHEMA",
    "StepProfiler",
    "attach_profiler",
    "calibrate",
    "profiler_of",
]
