"""What does a WOOF global run hold on the card BETWEEN steps?  (R4)

``python tools/arwen_global_resident_floor_probe.py CONFIG [--steps 10]
[--pool async|default] [--json out.json]``

The capacity model of the scale-out design splits a run's device memory
into a floor ``R`` that does not divide with the band count and a
band-local total ``L`` that does.  ``R`` is read here, at the allocator,
in four stages that name what joins it:

1. ``import`` -- the pool before anything of the model exists.
2. ``transform`` -- plus the three Legendre tables and the grid
   geometry.  At high truncation this stage IS the floor: the tables are
   packed by ORDER and do not band by latitude, so they never divide.
3. ``model_cold_state`` -- plus the spectral state, the surface and soil
   state, the transport geometry and the physics runtime with its
   g-point tables and column workspace.
4. ``between_steps`` -- the low-water instant after each step, taken
   after the pool's free blocks are released, which is the resident set
   the next step starts from.  The high water INSIDE the step comes from
   the allocator hook (``device_memory.DevicePeakTracker``), which folds
   the pool's live bytes after every allocation, because a reading taken
   at the end of the step has already lost every transient.

Two instruments per reading, because one of them under-reads.
``used_bytes`` is taken from the pool that is actually installed as the
allocator, so a ``--device-allocator async`` run reports the driver pool's live
bytes rather than an untouched default pool's zero.  Beside it the
card's own ``memGetInfo`` free bytes are differenced against the
reading taken before anything was built.  MEASURED 2026-09-06 on an RTX
5070 Ti: with the DEFAULT pool the two agree with the transform's own
``legendre_table_nbytes`` to 0.002 GiB at T255, T383, T533 and T799,
and the card reads 0.02 to 0.08 GiB higher for the context and the
allocation granularity.  **The ASYNC pool reads the same.**  A paired
measurement on one RTX 5090 (2026-09-06) built the same three tables
under each pool and read 0.1935 GiB from both, against 0.1934 GiB of
tables and scratch by the packed-table formula, and the T799 transform
stage reads 4.4644 GiB under either pool against 4.4632 analytic.  An
earlier note here said the async pool's ``used_bytes`` under-reports by
about a tenth; no run in either node's results carries the reading it
cited and the paired measurement refutes it.  The card delta is worth
keeping because it prices the context and the granularity the pool does
not see, not because the pool is untrustworthy -- and on a SHARED card
it is the unreliable one of the two, since another process freeing
memory during the stage makes it read negative.  The run is the real
one -- ``build_transform``,
``build_model_and_cold_state`` and ``MoistHybridModel.step`` off the
config given -- so a truncation that cannot allocate says so at the
stage it died in, and the stages that completed still carry their
readings.
"""
from __future__ import annotations

import argparse
import json
import platform
import time

GIB = 2 ** 30


from woof.globe.config import DEFAULT_DEVICE_ALLOCATOR


def _pool_of(allocator, cp):
    """The pool whose live bytes the installed allocator spends.

    One resolver, the run door's own, so the probe and the run cannot
    disagree about which allocator they are reading.
    """
    del allocator
    from woof.globe.device_memory import installed_pool

    pool = installed_pool(cp)
    return pool, type(pool).__name__


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--device-allocator", choices=("slab", "default", "async"),
        default=DEFAULT_DEVICE_ALLOCATOR,
        help="the run door's own allocator (woof.globe.device_memory."
             "select_device_allocator).  This probe used to install "
             "MemoryAsyncPool by hand because a T533 step profile would not "
             "run without it; the allocator is a door now, so the probe "
             "selects one the same way a run does and no private "
             "substitution survives here.  It DEFAULTS to the shipped run "
             "default, so a floor read here is the floor a bare run has; "
             "reading it under another allocator is a deliberate flag",
    )
    parser.add_argument("--spectral-chunk", type=int, default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    import cupy as cp

    from woof.globe.device_memory import select_device_allocator

    allocator = select_device_allocator(args.device_allocator, "cupy")
    pool, pool_name = _pool_of(cp.cuda.get_allocator(), cp)

    def live() -> int:
        return int(pool.used_bytes())

    def release() -> None:
        cp.cuda.Device().synchronize()
        try:
            pool.free_all_blocks()
        except Exception:
            pass

    import dataclasses

    from woof.globe.device_memory import DevicePeakTracker, installed_pool
    from woof.globe.config import load_config
    from woof.globe.runner import (
        build_model_and_cold_state, build_transform,
    )

    cfg = load_config(args.config)
    if args.spectral_chunk is not None:
        cfg = dataclasses.replace(cfg, spectral_chunk=int(args.spectral_chunk))
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    report = {
        "probe": "R4-resident-floor",
        "config": str(args.config),
        "host": f"{platform.system().lower()}-{platform.machine().lower()}",
        "card": props["name"].decode(),
        "card_total_gib": round(total_bytes / GIB, 3),
        "card_free_at_start_gib": round(free_bytes / GIB, 3),
        "pool": pool_name,
        "truncation": int(cfg.truncation),
        "nlev": int(cfg.vertical.nlev),
        "precision": cfg.precision,
        "spectral_chunk": int(cfg.spectral_chunk),
        "stages": [],
        "steps": [],
    }

    def card_used() -> int:
        return int(free_bytes - cp.cuda.runtime.memGetInfo()[0])

    def stage(name: str, seconds: float) -> None:
        release()
        row = {
            "stage": name,
            "live_gib": round(live() / GIB, 4),
            "card_gib": round(card_used() / GIB, 4),
            "seconds": round(seconds, 2),
        }
        report["stages"].append(row)
        print("stage {0}: {1:.4f} GiB in the pool, {2:.4f} GiB on the card, "
              "{3:.1f} s".format(
                  name, row["live_gib"], row["card_gib"], row["seconds"]))

    stage("import", 0.0)
    try:
        started = time.perf_counter()
        transform = build_transform(cfg)
        # The meridional-derivative table is built lazily on the first
        # gradient, so a stage reading taken before it under-reports the
        # table floor by a third.  The dycore builds it on step one.
        _ = transform._derivative_basis
        cp.cuda.Device().synchronize()
        report["legendre_table_bytes"] = {
            name: int(value)
            for name, value in transform.legendre_table_nbytes.items()
        }
        stage("transform", time.perf_counter() - started)

        started = time.perf_counter()
        model, state = build_model_and_cold_state(cfg, transform)
        cp.cuda.Device().synchronize()
        stage("model_cold_state", time.perf_counter() - started)

        tracker = DevicePeakTracker(installed_pool(cp)).install()
        for index in range(int(args.steps)):
            tracker.peak_used_bytes = live()
            started = time.perf_counter()
            state, _ = model.step(state, cfg.dt_s)
            cp.cuda.Device().synchronize()
            wall = time.perf_counter() - started
            peak = tracker.peak_used_bytes
            held = card_used()
            release()
            row = {
                "step": index + 1,
                "in_step_peak_gib": round(peak / GIB, 4),
                "card_held_after_gib": round(held / GIB, 4),
                "low_water_gib": round(live() / GIB, 4),
                "low_water_card_gib": round(card_used() / GIB, 4),
                "wall_s": round(wall, 4),
            }
            report["steps"].append(row)
            print(
                "step {0}: in-step peak {1:.4f} GiB (allocator hook), card "
                "held {2:.4f}, low water {3:.4f} pool / {4:.4f} card, "
                "{5:.3f} s".format(
                    row["step"], row["in_step_peak_gib"],
                    row["card_held_after_gib"], row["low_water_gib"],
                    row["low_water_card_gib"], row["wall_s"]))
        tracker.uninstall()
        report["status"] = "completed"
    except Exception as error:  # the reading up to the failure is the answer
        report["status"] = "failed"
        report["error"] = "{0}: {1}".format(type(error).__name__, error)
        print("FAILED after stage {0}: {1}".format(
            report["stages"][-1]["stage"], report["error"]))

    if report["steps"]:
        lows = [row["low_water_gib"] for row in report["steps"]]
        report["resident_floor_gib"] = min(lows)
        report["resident_floor_card_gib"] = min(
            row["low_water_card_gib"] for row in report["steps"])
        report["in_step_peak_gib"] = max(
            row["in_step_peak_gib"] for row in report["steps"])
        report["card_held_gib"] = max(
            row["card_held_after_gib"] for row in report["steps"])
        report["resident_floor_note"] = (
            "the smallest between-step low water over "
            "{0} steps; the model's R plus the persistent grid state "
            "G_persist, since nothing is banded on this tip".format(len(lows)))
        print("resident between steps: {0:.4f} GiB (minimum low water)".format(
            report["resident_floor_gib"]))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    return 0 if report.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
