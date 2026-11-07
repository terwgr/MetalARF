#!/usr/bin/env python3
"""Compare the Metal FAST_SBM call with the CUDA unit built for the host.

    python tools/fast_sbm_metal/column_probe.py --levels 8 20 34 --size 4

Each level count builds a small tile of physical columns (warm rain, mixed
phase, ice, supersaturated clear air, empty), spectra from the bulk start, and
runs three consecutive calls (the first initialises the aerosol spectra). Every
packed field word is compared literally after each call; each backend feeds its own result
into its next call. This is CUDA source executed on the host, not CUDA-device evidence.
"""
import math
import os
import re
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/thompson_real_column_parity"))
import host_backend  # noqa: E402

host_backend.CXX_FLAGS += ("-w",)
from woof.core.fast_sbm_bulk_start import bins_from_bulk  # noqa: E402
from woof.core.fast_sbm_contract import FLAG_FIELDS, STATE_FIELDS  # noqa: E402
from woof.metal import fast_sbm_family  # noqa: E402
from woof.metal.array import MetalArrayModule  # noqa: E402

SURFACE = ("xland", "rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv")


def unit_text():
    """fast_sbm.cu with its quoted includes inlined, as the host backend takes a unit."""
    kernels = ROOT / "woof/core/kernels"
    seen = set()

    def inline(name):
        if name in seen:
            return ""
        seen.add(name)
        out = []
        for line in (kernels / name).read_text().splitlines():
            if line.startswith('#include "'):
                out.append(inline(line.split('"')[1]))
            else:
                out.append(line)
        return "\n".join(out)
    # the host shim's launch loop owns the name kz
    text = inline("fast_sbm.cu")
    # Match the production unit's device branch, including its pinned libm.
    # The non-CUDA branch deliberately uses the host platform libm for the
    # separate Fortran oracle and is a different numerical reference.
    return "#define __CUDACC_RTC__ 1\n" + re.sub(r"\bkz\b", "kzcol", text)


def saturation(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / np.maximum(p - es, 1.0)


ZERO = ()
SCALE = 1.
DT = 5.


def tile(nz, size, seed):
    rng = np.random.default_rng(seed)
    ny = nx = size
    shape = (nz, ny, nx)
    dz = rng.uniform(150., 450., (1, ny, nx)) * np.ones((nz, 1, 1))
    zi = np.concatenate([np.zeros((1, ny, nx)), np.cumsum(dz, axis=0)])
    z = 0.5 * (zi[1:] + zi[:-1])
    t = rng.uniform(268., 303., (1, ny, nx)) - rng.uniform(5., 8., (1, ny, nx)) * 1e-3 * z
    t = np.maximum(t + rng.normal(0., 0.3, shape), 190.)
    p = np.empty(shape)
    pm = rng.uniform(95000., 102000., (ny, nx))
    for k in range(nz):
        pm = pm * np.exp(-9.81 * (zi[k + 1] - zi[k]) / (287. * t[k]))
        p[k] = pm
    pii = (p / 1e5) ** (287. / 1004.)
    th = t / pii
    rh = rng.uniform(0.4, 1.0, (1, ny, nx)) * np.ones((nz, 1, 1))
    rh = np.where(rng.random(shape) < 0.25, rng.uniform(1.0, 1.04, shape), rh)
    qv = rh * saturation(t, p)
    rho = p / (287. * t * (1. + 0.61 * qv))
    regime = rng.integers(0, 5, (1, ny, nx)) * np.ones((nz, 1, 1), int)
    layer = rng.random(shape) < rng.uniform(0.2, 0.7, (1, ny, nx))
    mag = lambda lo, hi: 10. ** rng.uniform(lo, hi, shape)
    warm = (t > 273.15) & layer
    cold = (t < 270.) & layer
    bulk = {"qc": np.where(warm & (regime != 3), mag(-6., -3.2), 0.),
            "qr": np.where(warm & (regime != 3), mag(-7., -3.2), 0.),
            "qi": np.where(cold & (regime >= 1) & (regime != 0), mag(-7., -3.5), 0.),
            "qs": np.where(cold & (regime >= 1), mag(-7., -3.2), 0.),
            "qg": np.where((t < 280.) & layer & (regime == 2), mag(-7., -3.2), 0.)}
    bulk = {k: np.where(regime == 4, 0., v) for k, v in bulk.items()}
    for k in ZERO:
        bulk[k] = 0. * bulk[k]
    if SCALE != 1.:
        bulk["qr"] = bulk["qr"] * SCALE
    xland = rng.choice([1., 2.], (ny, nx))
    planes = bins_from_bulk(np, {k: v for k, v in bulk.items()}, rho, xland[None], erf=np.vectorize(math.erf, otypes=[np.float64]))
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    values = {"w": rng.normal(0., 2., (nz + 1, ny, nx)), "u": rng.normal(5., 3., shape),
              "v": rng.normal(0., 3., shape), "th_old": th + rng.normal(0., 0.02, shape),
              "dz8w": dz, "rho": rho, "p": p, "pii": pii, "th": th, "qv": qv,
              "qv_old": qv * rng.uniform(0.98, 1.02, shape),
              "qnc": rng.uniform(0., 1e8, shape) * (bulk["qc"] > 0),
              "qnr": rng.uniform(0., 1e5, shape) * (bulk["qr"] > 0),
              "qni": rng.uniform(0., 1e5, shape) * (bulk["qi"] > 0),
              "qns": rng.uniform(0., 1e5, shape) * (bulk["qs"] > 0),
              "qng": rng.uniform(0., 1e4, shape) * (bulk["qg"] > 0),
              "qna": np.zeros(shape), **bulk}
    packed = {}
    for name in STATE_FIELDS:
        if name == "chem":
            raw = np.zeros((143, ny + 2, nz + 1, nx + 2), np.float32)
            raw[1:100, 1:ny + 1, :nz, 1:nx + 1] = f32(planes).transpose(0, 2, 1, 3)
        elif name in SURFACE:
            raw = np.zeros((ny + 2, nx + 2), np.float32)
            raw[1:ny + 1, 1:nx + 1] = xland if name == "xland" else (
                f32(rng.uniform(0., 3., (ny, nx))) if name.startswith(("rainnc", "snownc", "graupelnc")) and not name.endswith("ncv") else 0.)
        else:
            a = f32(values[name])
            raw = np.zeros((ny + 2, nz + 1, nx + 2), np.float32)
            raw[1:ny + 1, :a.shape[0], 1:nx + 1] = a.transpose(1, 0, 2)
        packed[name] = raw
    dims = (1, nx + 1, 1, ny + 1, 1, nz + 1, 0, nx + 1, 0, ny + 1, 1, nz + 1, 1, nx, 1, ny, 1, nz)
    return packed, dims, bulk


def host_run(module, step, dt, dx, dy, dims, packed):
    arrays, ikr, flags = fast_sbm_family.table_arrays()
    fields = {k: np.ascontiguousarray(v.copy()) for k, v in packed.items()}
    d = np.asarray(dims, np.int32)
    n = int((d[7] - d[6] + 1) * (d[11] - d[10] + 1) * (d[9] - d[8] + 1))
    nz = int(d[17] - d[16] + 1)
    nc = int((d[13] - d[12] + 1) * (d[15] - d[14] + 1))
    tp = np.asarray([a.ctypes.data for a in arrays], np.uint64)
    ap = np.asarray([fields[name].ctypes.data for name in STATE_FIELDS], np.uint64)
    work = np.zeros(7 * n + 2 * nc * (nz + 2), np.float32)
    kz = np.zeros(nc, np.int32)
    tally = np.zeros((nc, 17))
    status = np.zeros(nc, np.int32)
    args = (np.int32(step), np.float32(dt), np.float32(dx), np.float32(dy), d, tp, np.int32(ikr),
            flags, ap, work, kz, tally, status)
    halo = ((min(int(d[1]) - 1, int(d[13]) + 1) - max(int(d[12]) - 1, 1) + 1)
            * (min(int(d[3]) - 1, int(d[15]) + 1) - max(int(d[14]) - 1, 1) + 1))
    for kernel, count in (("fsbm_grid_fields_f32", halo), ("fsbm_prepare_columns_f32", nc),
                          ("fsbm_columns_f32", nc)):
        module.get_function(kernel)(((count + 63) // 64,), (64,), args)
    if status.any():
        raise RuntimeError("host FAST_SBM stopped at line %d" % int(status[np.flatnonzero(status)[0]]))
    return fields, tally


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 20, 34])
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--zero", nargs="*", default=[])
    parser.add_argument("--scale", type=float, default=1.)
    parser.add_argument("--flag", nargs="*", default=[])
    parser.add_argument("--dt", type=float, default=5.)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    global ZERO, DT, SCALE
    ZERO, DT, SCALE = tuple(args.zero), args.dt, args.scale
    if args.flag:
        base = fast_sbm_family.table_arrays
        over = dict(f.split("=") for f in args.flag)

        def patched():
            arrays, ikr, flags = base()
            flags = np.array(flags, np.int32)
            for k, v in over.items():
                flags[FLAG_FIELDS.index(k)] = int(v)
            return arrays, ikr, flags
        fast_sbm_family.table_arrays = patched
    module = host_backend.HostModule("fast_sbm", unit_text())
    xp = MetalArrayModule()
    report, bad = [], 0
    try:
        for nz in args.levels:
            packed, dims, _ = tile(nz, args.size, args.seed + nz)
            before = {k: v.copy() for k, v in packed.items()}
            row = {"nz": nz, "columns": args.size ** 2, "mismatched_words": 0, "words_compared": 0}
            host_state = packed
            metal_state = {k: v.copy() for k, v in packed.items()}
            for call in range(args.calls):
                step = call + 1
                try:
                    a, ta = host_run(module, step, DT, 3000., 3000., dims, host_state)
                except RuntimeError as err:
                    a, ta = None, str(err)
                try:
                    b, tb = fast_sbm_family.run(xp.context, step, DT, 3000., 3000., dims, metal_state)
                except RuntimeError as err:
                    b, tb = None, str(err)
                if a is None or b is None:
                    print(json.dumps({"nz": nz, "host": ta if a is None else "ok", "metal": tb if b is None else "ok"}))
                    return 1
                tb = np.asarray(tb).reshape(ta.shape)
                row["tally_diff"] = [[int(c), ta[c].tolist(), tb[c].tolist()]
                                     for c in range(len(ta)) if (ta[c].view(np.uint64) != tb[c].view(np.uint64)).any()]
                tally_bad = int(np.count_nonzero(ta.view(np.uint32) != tb.view(np.uint32)))
                row["mismatched_words"] += tally_bad
                row["words_compared"] += int(ta.size * 2)
                for name in STATE_FIELDS:
                    mism = int(np.count_nonzero(a[name].view(np.uint32) != b[name].view(np.uint32)))
                    row["mismatched_words"] += mism
                    row["words_compared"] += int(a[name].size)
                    if mism:
                        bad_at = np.argwhere(a[name].view(np.uint32) != b[name].view(np.uint32))[:3]
                        if name == "chem":
                            dif = np.argwhere(a[name].view(np.uint32) != b[name].view(np.uint32))
                            row["chem_planes"] = sorted(set(int(i[0]) for i in dif))
                            row["chem_levels"] = sorted(set(int(i[2]) for i in dif))
                            row["chem_cols"] = sorted(set((int(i[1]), int(i[3])) for i in dif))
                        row.setdefault("first_mismatches", []).append(
                            {"call": call, "field": name, "words": mism,
                             "at": [[tuple(int(i) for i in ix), float(a[name][tuple(ix)]),
                                     float(b[name][tuple(ix)])] for ix in bad_at]})
                host_state, metal_state = a, b
            row["changed_words"] = {k: int(np.count_nonzero(host_state[k].view(np.uint32) != before[k].view(np.uint32)))
                                    for k in STATE_FIELDS}
            bad += row["mismatched_words"]
            report.append(row)
            print(json.dumps(row))
    finally:
        xp.context.close()
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return int(bad != 0)


if __name__ == "__main__":
    raise SystemExit(main())
