#!/usr/bin/env python3
"""Compare Metal cloud-tendency output words with the retained source column.

Run device work through the project's GPU queue. This grades the dedicated
cloud fallout entry, including its read-only cloud input, not a complete
microphysics call or a strict WRF forecast.
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


def reference(directory):
    source = ROOT / "woof/core/kernels/thompson_aerosol_sed.cu"
    text = source.read_text()
    start = text.index("template <int KMAX>\n__device__ __forceinline__ void thompson_aa_cloud_sediment_impl(")
    brace = text.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    prefix = text[:start]
    shim = ROOT / "tools/thompson_real_column_parity/cuda_host_shim.h"
    headers = [ROOT / "woof/core/kernels" / name for name in
               ("glibc_flt32.cuh", "glibc_flt64.cuh", "thompson_aerosol_libm.cuh",
                "thompson_aerosol_common.cuh")]
    cpp = directory / "cloud_reference.cpp"
    cpp.write_text(f'#include "{shim}"\n'
                  + "".join(f'#include "{p}"\n' for p in headers)
                  + "#define IDX3(k,j,i) ((size_t(k)*ny+(j))*nx+(i))\n"
                  + prefix + text[start:end] + r"""
extern "C" void cloud_reference(float *qc, const float *nc, float *ncten,
        const float *temperature, const float *pressure, const float *qv,
        const float *held, const float *rain, const float *mask,
        const float *w, const float *dz, float *qcten,
        float dt, int nz, int ny, int nx) {
    for (int column = 0; column < ny * nx; ++column) {
        threadIdx.x = static_cast<unsigned>(column);
        thompson_aa_cloud_sediment_impl<256>(qc, nc, ncten, temperature,
            pressure, qv, held, rain, mask, w, dz, nullptr, nullptr,
            nullptr, nullptr, qcten, dt, nz, ny, nx);
    }
}
""")
    path = directory / "cloud_reference.dylib"
    subprocess.run(["clang++", "-std=c++17", "-O2", "-ffp-contract=off",
                    "-fno-fast-math", "-shared", "-fPIC", str(cpp), "-o", str(path)], check=True)
    library = ctypes.CDLL(str(path))
    fn = library.cloud_reference
    fn.argtypes = [ctypes.c_void_p] * 12 + [ctypes.c_float] + [ctypes.c_int] * 3
    return library, fn


def case(nz, clear=False):
    rng = np.random.default_rng(6281 + nz)
    shape = (nz, 2, 3)
    def values(low, high):
        return rng.uniform(low, high, shape).astype(np.float32)
    fields = {"qc": values(0, 2e-5), "nc": values(1e7, 4e8),
              "ncten": values(-1e4, 1e4), "temperature": values(260, 295),
              "pressure": values(50000, 100000), "qv": values(0, .015),
              "held": values(.6, 1.3), "rain": np.array([[0, 1, 0], [1, 0, 1]], np.float32),
              "mask": np.array([[1, 1, 1], [1, 1, 0]], np.float32),
              "w": values(-.1, .2), "dz": values(40, 90),
              "qcten": values(-1e-8, 1e-8)}
    if clear:
        fields["qc"].fill(0)
        fields["qcten"].fill(np.float32(-0.0))
    else:
        fields["qcten"][0, 0, 0] = np.array([1], np.uint32).view(np.float32)[0]
        fields["qc"][0, 1, 0] = np.float32(-0.0)
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=Path,
                        default=Path(tempfile.gettempdir()) / "woof-cloud-tendency-word-probe-result.json")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 3:
        parser.error("repeat must be 1 through 3")
    from woof.core.thompson_aerosol_runtime import load_metal_gamma_moments
    from woof.core.thompson_aerosol_sed import launch_aa_cloud_sedimentation
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    records = []
    names = ("qc", "nc", "ncten", "temperature", "pressure", "qv", "held",
             "rain", "mask", "w", "dz", "qcten")
    with tempfile.TemporaryDirectory(prefix="woof-cloud-word-") as temp:
        library, fn = reference(Path(temp))
        with MetalContext() as context:
            xp = MetalArrayModule(context)
            gamma = load_metal_gamma_moments(xp)
            for nz, clear in ((4, False), (49, False), (65, False), (129, False), (49, True)):
                original = case(nz, clear)
                expected = {k: v.copy() for k, v in original.items()}
                fn(*(expected[k].ctypes.data for k in names), 20., nz, 2, 3)
                for repetition in range(args.repeat):
                    device = {k: xp.asarray(v) for k, v in original.items()}
                    launch_aa_cloud_sedimentation(
                        device["qc"], device["nc"], device["ncten"],
                        device["temperature"], device["pressure"], device["qv"],
                        device["w"], device["dz"], 20., reference_density=device["held"],
                        rain_active_columns=device["rain"], cloud_active_columns=device["mask"],
                        qcten=device["qcten"], gamma_moments=gamma)
                    for name in ("qc", "qcten", "ncten"):
                        actual = device[name].to_numpy().view(np.uint32).reshape(-1)
                        wanted = expected[name].view(np.uint32).reshape(-1)
                        mismatch = np.flatnonzero(actual != wanted)
                        record = {"nz": nz, "clear": clear, "repetition": repetition,
                                  "field": name, "words": actual.size,
                                  "mismatches": mismatch.size}
                        if mismatch.size:
                            i = int(mismatch[0])
                            record["first_difference"] = {"index": i,
                                "expected": int(wanted[i]), "actual": int(actual[i])}
                        records.append(record)
    receipt = {"status": "FAIL" if any(r["mismatches"] for r in records) else "PASS",
               "reference": "retained current serial-column source and pinned arithmetic bodies",
               "comparisons": len(records), "output_words": sum(r["words"] for r in records),
               "mismatches": sum(r["mismatches"] for r in records), "cases": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
