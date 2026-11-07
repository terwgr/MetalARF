#!/usr/bin/env python3
"""Compare Metal solar output words to the authoritative pinned source.

Run GPU work through the project's GPU queue. This is a bounded device-word
probe, not an exhaustive proof of all trig inputs or full radiation parity.
The host reference compiles the retained CUDA arithmetic bodies with their
explicit RN operations and pinned trig; it does not use the platform's trig.
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
    source = directory / "solar_reference.cpp"
    shim = ROOT / "tools/glibc_trig_flt32_proof/host_shim.hpp"
    trig = ROOT / "woof/core/kernels/glibc_trig_flt32.cuh"
    solar = ROOT / "woof/core/kernels/rrtmg_legacy_solar.cu"
    source.write_text(
        f'#include "{shim}"\n#include "{trig}"\n'
        + r"""
#define __global__
#define __forceinline__ inline
#define FADD __fadd_rn
#define FSUB __fsub_rn
#define FMUL __fmul_rn
#define FDIV __fdiv_rn
struct SolarIndex { int x; };
static SolarIndex blockIdx = {0}, threadIdx = {0}, blockDim = {1};
"""
        + f'#include "{solar}"\n'
        + r"""
extern "C" void solar_reference(long long n, const float *lat,
        const float *lon, float julian, float xtime, float gmt,
        float degrad, float dpd, int generation, int override_declin,
        float declin, float *pair, float *cosine, float *hrang) {
    rls_radconst_values(julian, degrad, dpd, pair, nullptr);
    if (override_declin) pair[0] = declin;
    for (long long i = 0; i < n; ++i)
        rls_coszen_point(i, lat, lon, julian, xtime, gmt, degrad,
                         generation, pair, cosine, hrang);
}
""")
    library = directory / "solar_reference.dylib"
    subprocess.run(["clang++", "-std=c++17", "-O2", "-ffp-contract=off",
                    "-fno-fast-math", "-shared", "-fPIC", str(source),
                    "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    fn = lib.solar_reference
    fn.argtypes = ([ctypes.c_int64] + [ctypes.c_void_p] * 2
                   + [ctypes.c_float] * 5 + [ctypes.c_int] * 2
                   + [ctypes.c_float] + [ctypes.c_void_p] * 3)
    return lib, fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeat <= 3:
        parser.error("repeat must be 1 through 3")
    from woof.core.rrtmg_legacy import _DEGRAD, _DPD
    from woof.core.solar import solar_geometry_device
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext

    # Include both seasonal branches, midnight MOD boundaries, signed
    # subnormals/zero and large finite reduction arguments in 257 columns.
    lat = np.linspace(-90, 90, 257, dtype=np.float32)
    lon = np.linspace(-180, 180, 257, dtype=np.float32)
    extras = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff,
                       0x807fffff, 0x00800000, 0x80800000,
                       0x7f7fffff, 0xff7fffff, 0x3f800000,
                       0xbf800000], np.uint32).view(np.float32)
    lat[:extras.size] = extras
    lon[-extras.size:] = extras
    tiny = np.array([1], np.uint32).view(np.float32)[0]
    cases = [
        (0., 0., 0., None), (.5, -0., 12., None),
        (79.99999, 1439.9999, 23.999, None),
        (80., 1440., 0., None), (80.00001, -1440.0001, 7., None),
        (182.25, 16777216., 12.5, None),
        (364.99997, -16777216., 23., None),
        (365.25, np.finfo(np.float32).max, 4., None),
        (366.75, -np.finfo(np.float32).max, 18., None),
        (182.25, 30.5, 6., tiny), (182.25, 30.5, 6., -tiny),
        (182.25, 30.5, 6., np.float32(.49999997)),
    ]
    comparisons = words = 0
    failures = []
    with tempfile.TemporaryDirectory(prefix="woof-solar-reference-") as temp:
        library, reference = build_reference(Path(temp))
        with MetalContext() as context:
            arrays = MetalArrayModule(context)
            dlat, dlon = arrays.asarray(lat), arrays.asarray(lon)
            for repeat in range(args.repeat):
                for generation in ("wrf_461", "wrf_480"):
                    for index, (julian, xtime, gmt, declin) in enumerate(cases):
                        pair = np.empty(2, np.float32)
                        cosine = np.empty_like(lat)
                        hrang = np.empty_like(lat)
                        reference(lat.size, lat.ctypes.data, lon.ctypes.data,
                                  julian, xtime, gmt, _DEGRAD, _DPD,
                                  int(generation == "wrf_480"),
                                  int(declin is not None), 0. if declin is None else declin,
                                  pair.ctypes.data, cosine.ctypes.data, hrang.ctypes.data)
                        actual = solar_geometry_device(
                            julian, xtime, gmt, dlat, dlon,
                            generation=generation, declination=declin)
                        for field, expected, received in zip(
                                ("declin_solcon", "coszen", "hrang"),
                                (pair, cosine, hrang), actual):
                            expected_bits = expected.view(np.uint32)
                            received_bits = received.to_numpy().view(np.uint32)
                            mismatch = np.flatnonzero(expected_bits != received_bits)
                            comparisons += 1
                            words += expected.size
                            if mismatch.size:
                                failures.append({
                                    "repeat": repeat, "generation": generation,
                                    "case": index, "field": field,
                                    "mismatches": int(mismatch.size),
                                    "examples": [
                                        {"column": int(i),
                                         "expected": hex(int(expected_bits[i])),
                                         "actual": hex(int(received_bits[i]))}
                                        for i in mismatch[:4]]})
    print(json.dumps({"comparisons": comparisons, "words": words,
                      "failed_comparisons": len(failures),
                      "failures": failures[:8]}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
