#!/usr/bin/env python3
"""Compare actual Metal aerosol graupel entry-number words with the current pinned source.

Run device work through the project's GPU queue. The reference compiles the
retained source kernel and its libm bodies with explicit RN operations; the
platform power function is not used. This bounded probe is not a full
microphysics or strict WRF parity claim.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_reference(directory):
    kernel_path = ROOT / "woof/core/kernels/thompson_aerosol_state.cu"
    text = kernel_path.read_text()
    start = text.index('extern "C" __global__ void thompson_aa_graupel_number_init(')
    brace = text.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    shim = ROOT / "tools/thompson_real_column_parity/cuda_host_shim.h"
    headers = [ROOT / "woof/core/kernels" / name for name in
               ("glibc_flt32.cuh", "glibc_flt64.cuh", "thompson_aerosol_libm.cuh",
                "thompson_aerosol_common.cuh")]
    source = directory / "graupel_reference.cpp"
    source.write_text(f'#include "{shim}"\n'
                      + "".join(f'#include "{header}"\n' for header in headers)
                      + text[start:end] + r"""
extern "C" void graupel_reference(int n, const float *qg, const float *temperature,
        const float *pressure, const float *qv, float *out) {
    for (int i = 0; i < n; ++i) {
        threadIdx.x = static_cast<unsigned>(i);
        thompson_aa_graupel_number_init(qg, temperature, pressure, qv, out, n);
    }
}
""")
    library = directory / "graupel_reference.dylib"
    subprocess.run(["clang++", "-std=c++17", "-O2", "-ffp-contract=off",
                    "-fno-fast-math", "-shared", "-fPIC", str(source),
                    "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    fn = lib.graupel_reference
    fn.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 5
    return lib, fn


def inputs():
    n = 8193
    ordinal = np.arange(n, dtype=np.float32)
    qg = np.geomspace(1.e-14, 1.e4, n).astype(np.float32)
    temperature = (195. + ordinal % 120).astype(np.float32)
    pressure = (10000. + ordinal % 80000).astype(np.float32)
    qv = (-.01 + (ordinal % 67) * .001).astype(np.float32)
    # The gate, signed zeros/subnormals, and neighbours of the R1 word.
    r1 = np.asarray(1.e-12, np.float32).view(np.uint32).item()
    edges = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff,
                      0x807fffff, r1-1, r1, r1+1, 0xbf800000,
                      0x7fc00000, 0x7f7fffff, 0x7f800000], np.uint32).view(np.float32)
    qg[:edges.size] = edges
    return [("graupel_range_raw_vapour", tuple(np.ascontiguousarray(value)
            for value in (qg, temperature, pressure, qv)))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=Path,
                        default=Path(tempfile.gettempdir()) / "woof-thompson-graupel-word-probe-result.json")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 3:
        parser.error("repeat must be 1 through 3")
    from woof.core.thompson_aerosol_state import launch_aa_graupel_number_init
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext

    records = []
    with tempfile.TemporaryDirectory(prefix="woof-graupel-word-") as temp:
        library, reference = build_reference(Path(temp))
        with MetalContext() as context:
            xp = MetalArrayModule(context)
            for label, values in inputs():
                expected = np.empty_like(values[0])
                reference(expected.size, *(value.ctypes.data for value in values),
                          expected.ctypes.data)
                device_values = tuple(xp.asarray(value) for value in values)
                for repetition in range(args.repeat):
                    actual = xp.empty(expected.shape, dtype=np.float32)
                    launch_aa_graupel_number_init(*device_values, actual)
                    words = actual.to_numpy().view(np.uint32)
                    wanted = expected.view(np.uint32)
                    differing = np.flatnonzero(words != wanted)
                    record = {"case": label, "repetition": repetition,
                              "words": expected.size, "mismatches": differing.size}
                    if differing.size:
                        i = int(differing[0])
                        record["first_difference"] = {"index": i,
                            "input_words": [int(value.view(np.uint32)[i]) for value in values],
                            "expected_word": int(wanted[i]), "actual_word": int(words[i])}
                    records.append(record)
    receipt = {"status": "PASS" if not any(r["mismatches"] for r in records) else "FAIL",
               "reference": "current source kernel and pinned libm bodies with explicit RN operations",
               "repeat": args.repeat, "comparisons": len(records),
               "output_words": sum(r["words"] for r in records),
               "mismatches": sum(r["mismatches"] for r in records), "cases": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
