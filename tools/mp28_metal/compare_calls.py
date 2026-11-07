#!/usr/bin/env python3
"""Compare repeated production MP28 calls without stage resynchronization.

The host side uses the retained CUDA-source host backend. Both sides preserve
their own prognostics and accumulators across calls, including output-due radar
reflectivity. Every physical field, returned diagnostic and live scratch word
is compared; dead scratch differences are reported separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"),
                str(ROOT / "tools/thompson_real_column_parity")]

from metal_thompson_aerosol_call_probe import State, SURFACE, config, load_case

# Neither slot has a reader after its final write in the call. Metal prepare
# forms dz directly, while the host also materializes the full-level heights.
# The terminal cloud apply leaves its sum on Metal and its product on the host.
DEAD_SCRATCH = {"scratch/mp_z8w", "scratch/mp_thompson_aero_condensation_rate"}
WORKING_DT = "scratch/mp_thompson_working_dt"


def category(key):
    if key in DEAD_SCRATCH:
        return "dead_scratch"
    if key == WORKING_DT:
        return "backend_constant"
    return "live_scratch" if key.startswith("scratch/") else "physical"


def options(args):
    return {name: getattr(args, name) for name in (
        "version", "calls", "dt", "columns", "seed_condensate", "reflectivity")}


def frames(args, metal):
    if metal:
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.runtime import MetalContext
        context = MetalContext()
        xp = MetalArrayModule(context)
        is_array = lambda value: isinstance(value, MetalArray)
        host = lambda value: value.to_numpy()
    else:
        import host_backend
        host_backend.install()
        context = None
        xp = np
        is_array = lambda value: isinstance(value, np.ndarray)
        host = np.asarray
    try:
        from woof.core.microphysics_aerosol import _apply_thompson_aerosol
        from woof.core.refl import consume_refl_10cm

        arrays = load_case(args.prepared, args.restart,
                           tuple(map(int, args.columns.split(","))),
                           seed=args.seed_condensate)
        state = State(xp, arrays, metal)
        state.physics = SimpleNamespace(refl_10cm=None)
        cfg = config(args.version)
        for _ in range(args.calls):
            diagnostics = _apply_thompson_aerosol(
                state, cfg, args.dt, refl_10cm_due=args.reflectivity)
            frame = {f"state/{name}": host(value).copy()
                     for name, value in vars(state).items() if is_array(value)}
            frame.update({f"scratch/{name}": host(value).copy()
                          for name, value in state._scratch.items()})
            frame.update({f"diagnostic/{name}": host(getattr(diagnostics, name)).copy()
                          for name in SURFACE})
            if args.reflectivity:
                frame["diagnostic/refl_10cm"] = host(consume_refl_10cm(state)).copy()
            yield frame
            state.elapsed_seconds += args.dt
    finally:
        if context is not None:
            context.close()


def words(array):
    array = np.ascontiguousarray(array)
    dtype = {4: np.uint32, 8: np.uint64}.get(array.dtype.itemsize)
    if dtype is None:
        raise TypeError(f"unsupported comparison dtype {array.dtype}")
    return array.view(dtype).reshape(-1)


def compare(args):
    reference = np.load(args.reference)
    if json.loads(str(reference["options"])) != options(args):
        raise ValueError("reference options do not match this trajectory")
    counts = {name: {"words": 0, "mismatches": 0}
              for name in ("physical", "live_scratch", "dead_scratch", "backend_constant")}
    differences = []
    missing = []
    for call, frame in enumerate(frames(args, True)):
        expected_keys = {key.split(":", 1)[1] for key in reference.files
                         if key.startswith(f"{call}:")}
        for key in sorted(set(frame) | expected_keys):
            group = category(key)
            if key not in frame or key not in expected_keys:
                if key == WORKING_DT and key in frame:
                    actual = words(frame[key])
                    expected = np.full(frame[key].shape, args.dt, np.float32)
                    bad = int(np.count_nonzero(actual != words(expected)))
                    counts[group]["words"] += actual.size
                    counts[group]["mismatches"] += bad
                    if bad:
                        differences.append({"call": call, "field": key,
                                            "category": group, "mismatches": bad})
                    continue
                missing.append({"call": call, "field": key,
                                "side": "metal" if key not in frame else "host"})
                continue
            got = frame[key]
            expected = reference[f"{call}:{key}"]
            if got.shape != expected.shape or got.dtype != expected.dtype:
                missing.append({"call": call, "field": key, "reason": "shape or dtype"})
                continue
            actual_words, expected_words = words(got), words(expected)
            ids = np.flatnonzero(actual_words != expected_words)
            counts[group]["words"] += actual_words.size
            counts[group]["mismatches"] += ids.size
            if ids.size:
                differences.append({"call": call, "field": key, "category": group,
                                    "mismatches": int(ids.size), "first_index": int(ids[0]),
                                    "actual_word": int(actual_words[ids[0]]),
                                    "expected_word": int(expected_words[ids[0]])})
    failed = missing or any(counts[name]["mismatches"] for name in (
        "physical", "live_scratch", "backend_constant"))
    report = {"status": "FAIL" if failed else "PASS", **options(args),
              "resync": False, "counts": counts, "missing": missing,
              "differences": differences}
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return int(bool(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("host", "metal", "compare"))
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--restart", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--version", choices=("wrf_39_noaa", "wrf_461"), required=True)
    parser.add_argument("--columns", default="0,16,0,16")
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--dt", type=float, default=20.0)
    parser.add_argument("--seed-condensate", action="store_true")
    parser.add_argument("--reflectivity", action="store_true")
    args = parser.parse_args()
    if args.calls < 1 or not np.isfinite(args.dt) or args.dt <= 0:
        parser.error("calls and dt must be positive")
    if args.mode == "compare":
        # Keep the host's NumPy-backed CuPy adapter out of the Metal process.
        for mode in ("host", "metal"):
            done = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                   mode, *sys.argv[2:]], check=False)
            if done.returncode:
                return done.returncode
        return 0
    if args.mode == "host":
        payload = {f"{call}:{key}": value
                   for call, frame in enumerate(frames(args, False))
                   for key, value in frame.items()}
        payload["options"] = np.asarray(json.dumps(options(args), sort_keys=True))
        np.savez(args.reference, **payload)
        print(json.dumps({"status": "REFERENCE_SAVED", **options(args)}))
        return 0
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
