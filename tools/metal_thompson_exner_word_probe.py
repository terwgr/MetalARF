#!/usr/bin/env python3
"""Compare actual Metal aerosol Exner words with the current pinned source.

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
    start = text.index('extern "C" __global__ void thompson_aa_exner(')
    brace = text.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    shim = ROOT / "tools/thompson_real_column_parity/cuda_host_shim.h"
    headers = [ROOT / "woof/core/kernels" / name for name in
               ("glibc_flt32.cuh", "glibc_flt64.cuh", "thompson_aerosol_libm.cuh",
                "thompson_aerosol_common.cuh")]
    source = directory / "exner_reference.cpp"
    source.write_text(f'#include "{shim}"\n'
                      + "".join(f'#include "{header}"\n' for header in headers)
                      + text[start:end] + r"""
extern "C" void exner_reference(int n, const float *pressure, float *out) {
    for (int i = 0; i < n; ++i) {
        threadIdx.x = static_cast<unsigned>(i);
        thompson_aa_exner(pressure, out, n);
    }
}
""")
    library = directory / "exner_reference.dylib"
    subprocess.run(["clang++", "-std=c++17", "-O2", "-ffp-contract=off",
                    "-fno-fast-math", "-shared", "-fPIC", str(source),
                    "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    fn = lib.exner_reference
    fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    return lib, fn


def inputs(prepared):
    cases = {"atmospheric_pressure_band": np.linspace(
        100., 120000., 4097, dtype=np.float32)}
    cases["float32_boundaries"] = np.array([
        0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x00800001,
        0x3f7fffff, 0x3f800000, 0x3f800001, 0x47c34fff,
        0x47c35000, 0x47c35001, 0x7f7fffff, 0x7f800000,
        0xbf800000, 0xff800000, 0x7fc00000], dtype=np.uint32).view(np.float32)
    if prepared is not None:
        cache = prepared / "native/prepared-cache"
        header = json.loads((cache / "header.json").read_text())
        cases["prepared_pressure"] = np.load(
            cache / header["arrays"]["state/p"]["file"]).astype(np.float32).reshape(-1)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--out", type=Path,
                        default=Path(tempfile.gettempdir()) / "woof-thompson-exner-word-probe-result.json")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 3:
        parser.error("repeat must be 1 through 3")
    from woof.core.thompson_aerosol_state import launch_aerosol_exner
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext

    records = []
    with tempfile.TemporaryDirectory(prefix="woof-exner-word-") as temp:
        library, reference = build_reference(Path(temp))
        with MetalContext() as context:
            xp = MetalArrayModule(context)
            for label, pressure in inputs(args.prepared_root).items():
                pressure = np.ascontiguousarray(pressure)
                expected = np.empty_like(pressure)
                reference(pressure.size, pressure.ctypes.data, expected.ctypes.data)
                for repetition in range(args.repeat):
                    actual = xp.empty(pressure.shape, dtype=np.float32)
                    launch_aerosol_exner(xp.asarray(pressure), actual)
                    words = actual.to_numpy().view(np.uint32)
                    wanted = expected.view(np.uint32)
                    differing = np.flatnonzero(words != wanted)
                    record = {"case": label, "repetition": repetition,
                              "words": pressure.size, "mismatches": differing.size}
                    if differing.size:
                        i = int(differing[0])
                        record["first_difference"] = {"index": i,
                            "pressure_word": int(pressure.view(np.uint32)[i]),
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
