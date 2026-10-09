"""Compare two WOOF global checkpoint archives array by array, bit for bit,
and state the three digests a reader can check: the BIT-IMEX gate's
instrument, in the tree.

The gate: the untouched Eulerian path (``imex_ssp3``) on the shipped T255
native config, ten steps, on two trees, same card,
same decoder, same statics.  Every array of the step-0 and step-10
checkpoints has to be identical and so do the archives' ``self_sha256``.
The ``config_hash`` and ``pins_hash`` are printed because a step change
moves the config hash by design (the step is part of the config identity),
so a re-pin at a new step states the new hashes rather than expecting the
old ones.

The metadata blob is dropped from the array count (42 arrays at step 0
and 125 at step 10 on the shipped T255 native config), which is the
convention the merged tree's receipts use.

usage:
  python tools/arwen_global_checkpoint_identity.py A.npz B.npz [--out-json R.json]
exit status 0 when identical, 1 when anything differs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["__metadata__"].item()))
        arrays = {name: np.array(archive[name]) for name in archive.files if name != "__metadata__"}
    return meta, arrays


def compare(a: Path, b: Path) -> dict:
    meta_a, arr_a = load(a)
    meta_b, arr_b = load(b)
    missing = sorted(set(arr_a) ^ set(arr_b))
    differing = []
    for name in sorted(set(arr_a) & set(arr_b)):
        x, y = arr_a[name], arr_b[name]
        if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(
                x.view(np.uint8), y.view(np.uint8)):
            entry = {"array": name, "dtype": (str(x.dtype), str(y.dtype)), "shape": (list(x.shape), list(y.shape))}
            if x.dtype == y.dtype and x.shape == y.shape and np.issubdtype(x.dtype, np.number):
                entry["max_abs_difference"] = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
            differing.append(entry)
    identical = not missing and not differing and meta_a.get("self_sha256") == meta_b.get("self_sha256")
    return {
        "a": {"path": str(a), "schema": meta_a.get("schema"), "step": meta_a.get("step"),
              "arrays": len(arr_a), "config_hash": meta_a.get("config_hash"),
              "pins_hash": meta_a.get("pins_hash"), "self_sha256": meta_a.get("self_sha256")},
        "b": {"path": str(b), "schema": meta_b.get("schema"), "step": meta_b.get("step"),
              "arrays": len(arr_b), "config_hash": meta_b.get("config_hash"),
              "pins_hash": meta_b.get("pins_hash"), "self_sha256": meta_b.get("self_sha256")},
        "arrays_compared": len(set(arr_a) & set(arr_b)),
        "inventory_mismatch": missing,
        "differing": differing,
        "verdict": "IDENTICAL" if identical else "DIFFERS",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()
    result = compare(args.a, args.b)
    for side in ("a", "b"):
        s = result[side]
        print(f"{side.upper()} {s['path']}")
        print(f"  schema {s['schema']} step {s['step']} arrays {s['arrays']}")
        print(f"  config_hash {s['config_hash']}")
        print(f"  pins_hash   {s['pins_hash']}")
        print(f"  self_sha256 {s['self_sha256']}")
    print("inventory mismatch:", result["inventory_mismatch"])
    print("arrays compared:", result["arrays_compared"], "differing:", len(result["differing"]))
    for entry in result["differing"][:12]:
        print("  ", entry)
    print("VERDICT:", result["verdict"])
    if args.out_json:
        args.out_json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0 if result["verdict"] == "IDENTICAL" else 1


if __name__ == "__main__":
    sys.exit(main())
