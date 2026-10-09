"""Bit compare two WOOF global checkpoints, array by array.

``python tools/arwen_global_checkpoint_diff.py A.npz B.npz [--json out.json]``

The checkpoint format hashes every array as
``sha256(dtype.str + str(shape) + bytes)`` and hashes its own metadata, so
two runs that computed the same state carry the same hashes.  This reads
the two files with numpy directly (never through the config-hash
admission, because an A/B whose whole point is a config knob writes two
different config hashes by design) and reports, per array, whether the
bytes are equal and, where they are not, the maximum absolute and ULP
distance and the count of differing elements.
"""
from __future__ import annotations

import argparse
import hashlib
import json

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
    if a.dtype.kind != "f":
        return float("nan")
    kind = np.int32 if a.dtype == np.float32 else np.int64
    ia = np.ascontiguousarray(a).view(kind).astype(np.int64)
    ib = np.ascontiguousarray(b).view(kind).astype(np.int64)
    top = np.int64(np.iinfo(kind).min)
    ia = np.where(ia < 0, top - ia, ia)
    ib = np.where(ib < 0, top - ib, ib)
    return float(np.abs(ia - ib).max())


def compare(path_a: str, path_b: str) -> dict:
    a = np.load(path_a, allow_pickle=False)
    b = np.load(path_b, allow_pickle=False)
    names = sorted(set(a.files) | set(b.files))
    rows = []
    differing = 0
    for name in names:
        if name not in a.files or name not in b.files:
            rows.append({"array": name, "equal": False,
                         "reason": "present in one file only"})
            differing += 1
            continue
        va, vb = a[name], b[name]
        if va.dtype.kind in "SU" or va.dtype == object:
            equal = bool(np.array_equal(va, vb))
            rows.append({"array": name, "equal": equal, "dtype": str(va.dtype)})
            differing += 0 if equal else 1
            continue
        if va.shape != vb.shape or va.dtype != vb.dtype:
            rows.append({"array": name, "equal": False,
                         "reason": f"{va.dtype}{va.shape} vs {vb.dtype}{vb.shape}"})
            differing += 1
            continue
        equal = array_hash(va) == array_hash(vb)
        row = {"array": name, "equal": equal, "dtype": str(va.dtype),
               "shape": [int(v) for v in va.shape]}
        if not equal:
            differing += 1
            # A complex array cast to float64 DISCARDS the imaginary part,
            # so max_abs on the spectral state read the real half of the
            # difference and called it the difference.  Every checkpointed
            # atmosphere field is complex, which is where an A/B over a
            # transform knob does its diverging.
            if np.iscomplexobj(va) or np.iscomplexobj(vb):
                diff = np.abs(va.astype(np.complex128)
                              - vb.astype(np.complex128))
            else:
                diff = np.abs(va.astype(np.float64) - vb.astype(np.float64))
            row["differing_elements"] = int(np.count_nonzero(va != vb))
            row["max_abs"] = float(diff.max())
            row["max_ulp"] = _ulp(va, vb)
        rows.append(row)
    return {
        "a": path_a, "b": path_b,
        "arrays": len(names), "differing": differing,
        "bit_identical": differing == 0,
        "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    report = compare(args.a, args.b)
    for row in report["rows"]:
        if not row["equal"]:
            print("MOVED {0}: {1}".format(
                row["array"],
                {k: v for k, v in row.items() if k not in ("array", "equal")}))
    print("{0} arrays, {1} differing: {2}".format(
        report["arrays"], report["differing"],
        "BIT IDENTICAL" if report["bit_identical"] else "NOT IDENTICAL"))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    return 0 if report["bit_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
