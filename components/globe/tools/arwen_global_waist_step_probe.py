"""Does the Fourier waist hold inside the dycore's own call sequence?

``python tools/arwen_global_waist_step_probe.py <config.toml>
[--bands 2,4,8,16] [--steps 5] [--json out.json]``

The isolated gate (``tools/arwen_global_waist_probe.py``) holds the
waist against the resident transform one call at a time, at every
leading shape the dycore transforms.  This one runs the REAL MODEL:
``build_model_and_cold_state`` builds the same model twice, once on a
one-band transform and once on a banded one, and both are stepped
through ``MoistHybridModel.step`` -- the same physics suite, the same
IMEX stages, the same transport, the same repairs -- and every array of
the resulting state is compared byte for byte.

What it catches that the isolated gate cannot: a transform call the
dycore makes at a shape or in an order the isolated gate does not
reproduce, a waist consumed twice inside one step, and any state the
transform keeps between calls that a band count could reach.

Nothing about the band count enters the config, so the two arms carry
the same config hash by construction; the comparison is of the states,
not of the receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np


def array_hash(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(arr.dtype.str.encode())
    digest.update(str(arr.shape).encode())
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def _ulp(a: np.ndarray, b: np.ndarray) -> float:
    if a.dtype.kind == "c":
        return max(_ulp(np.ascontiguousarray(a.real), np.ascontiguousarray(b.real)),
                   _ulp(np.ascontiguousarray(a.imag), np.ascontiguousarray(b.imag)))
    if a.dtype.kind not in "fc":
        return 0.0
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    kind = np.int32 if a.dtype == np.float32 else np.int64
    ia = a.view(kind).astype(np.int64)
    ib = b.view(kind).astype(np.int64)
    top = np.int64(np.iinfo(kind).min)
    ia = np.where(ia < 0, top - ia, ia)
    ib = np.where(ib < 0, top - ib, ib)
    return float(np.abs(ia - ib).max())


def _state_arrays(model, state) -> dict[str, np.ndarray]:
    """Every CHECKPOINTED array, on the host.

    The checkpoint's own inventory, not a hand-picked subset: the same
    five spectral fields, ten grid tracers, surface namespace and
    physics namespace the ten-step gate of record compares, through the
    same function that builds them.
    """
    from woof.globe.checkpoint import bundle_arrays

    return bundle_arrays(state, model.transform.backend.to_numpy)


def _step_arm(cfg, bands: int, steps: int):
    from woof.globe.runner import build_model_and_cold_state, build_transform
    from woof.globe.spectral.transform import SphericalHarmonicTransform

    base = build_transform(cfg)
    transform = SphericalHarmonicTransform(
        base.grid,
        base.backend,
        tensor_core_contractions=cfg.tensor_core_contractions,
        legendre_band=cfg.legendre_band,
        streaming=cfg.streaming,
        latitude_bands=bands,
    )
    del base
    model, state, *_rest = build_model_and_cold_state(cfg, transform=transform)
    started = time.perf_counter()
    for _ in range(steps):
        state, _metrics = model.step(state, cfg.dt_s)
    seconds = time.perf_counter() - started
    arrays = _state_arrays(model, state)
    identity = transform.identity_hash
    del model, state, transform
    return arrays, identity, seconds


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config")
    parser.add_argument("--bands", default="2,4,8,16")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    from woof.globe.config import load_config

    cfg = load_config(args.config)
    counts = [int(x) for x in args.bands.split(",") if x]
    report = {
        "gate": "WAIST-1, through MoistHybridModel.step",
        "config": args.config,
        "truncation": int(cfg.truncation),
        "nlev": int(cfg.vertical.nlev),
        "backend": cfg.backend,
        "precision": cfg.precision,
        "steps": args.steps,
        "arms": [],
    }
    reference, identity, seconds = _step_arm(cfg, 1, args.steps)
    report["reference"] = {
        "bands": 1,
        "arrays": len(reference),
        "identity_hash": identity,
        "seconds": round(seconds, 3),
        "hashes": {k: array_hash(v) for k, v in sorted(reference.items())},
    }
    print(
        f"reference: {len(reference)} state arrays after {args.steps} steps, "
        f"{seconds:.2f} s",
        flush=True,
    )
    worst = 0
    for bands in counts:
        arrays, banded_identity, seconds = _step_arm(cfg, bands, args.steps)
        differing = []
        for name, value in sorted(arrays.items()):
            ref = reference.get(name)
            if ref is None:
                differing.append({"array": name, "reason": "absent from the reference"})
                continue
            if array_hash(ref) == array_hash(value):
                continue
            differing.append({
                "array": name,
                "max_ulp": _ulp(ref, value),
                "max_abs": float(np.abs(np.asarray(ref, dtype=np.float64)
                                        - np.asarray(value, dtype=np.float64)).max()),
                "differing_points": int(np.count_nonzero(ref != value)),
            })
        worst = max(worst, len(differing))
        report["arms"].append({
            "bands": bands,
            "arrays": len(arrays),
            "identical": len(arrays) - len(differing),
            "differing": differing,
            "identity_unchanged": banded_identity == identity,
            "seconds": round(seconds, 3),
        })
        print(
            f"  bands={bands:<3} {len(arrays) - len(differing)}/{len(arrays)} "
            f"state arrays byte-identical, identity hash unchanged "
            f"{banded_identity == identity}, {seconds:.2f} s",
            flush=True,
        )
        for row in differing:
            print(f"    MOVED {row}")
    report["verdict"] = (
        "WAIST-1 (step) PASSES" if worst == 0 else "WAIST-1 (step) FAILS"
    )
    print(f"\n{report['verdict']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"wrote {args.json}")
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
