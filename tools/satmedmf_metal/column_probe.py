#!/usr/bin/env python3
"""Compare the Metal TKE-EDMF column with the CUDA unit built for the host.

    python tools/satmedmf_metal/column_probe.py --levels 8 20 48 100 --tracers 5 9

The saturation table and the Exner kernel are compared first; each level count
then runs the column over perturbed copies of the oracle regimes (convective,
stable, stratocumulus, windy, shallow, coarse and fine grid), twice, the second
call on the first call's advanced state. Every output word is compared literally.
"""
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
from tools.satmedmf_ccpp_oracle.check import columns as oracle_columns  # noqa: E402
from woof.core.satmedmf import _INPUTS  # noqa: E402
from woof.metal import satmedmf_family  # noqa: E402
from woof.metal.array import MetalArrayModule  # noqa: E402

OUTPUTS = ("du", "dv", "tdt", "rtg", "hpbl", "dkt", "dku", "sflux", "kpbl")


def columns(nz, nt, seed):
    """Oracle regimes with every field perturbed, tracers beyond the TKE added."""
    rng = np.random.default_rng(seed)
    x = oracle_columns(64, nz, nt)
    n = 64
    shape = (n, nz)
    x["t"] = x["t"] + rng.normal(0., 0.4, shape)
    x["u"] = x["u"] + rng.normal(0., 1.5, shape)
    x["v"] = x["v"] + rng.normal(0., 1.5, shape)
    q = x["q"]
    q[:, :, 0] *= rng.uniform(0.4, 1.3, (n, 1))
    layer = lambda p: rng.random(shape) < p
    q[:, :, 1] = np.where(layer(0.15), 10. ** rng.uniform(-6., -3., shape), q[:, :, 1])
    q[:, :, 2] = np.where(layer(0.1), 10. ** rng.uniform(-7., -3.5, shape), 0.)
    q[:, :, 3] = np.where(layer(0.1), 10. ** rng.uniform(-7., -3.5, shape), 0.)
    for j in range(4, nt - 1):
        q[:, :, j] = np.where(layer(0.5), 10. ** rng.uniform(-10., -5., shape), 0.)
    q[:, :, -1] = np.where(rng.random((n, 1)) < 0.15, 0., rng.uniform(0., 2.5, shape) ** 2)
    x["swh"] = rng.normal(0., 1e-5, shape)
    x["hlw"] = rng.normal(-1e-5, 1e-5, shape)
    s = x["sfc"]
    s[:, 1] = rng.uniform(-0.4, 0.4, n)
    s[:, 2] = 10. ** rng.uniform(-1., 2., n)
    s[:, 3] = x["t"][:, 0] + rng.uniform(-4., 4., n)
    s[:, 4] = rng.uniform(-0.15, 0.4, n) * (rng.random(n) < 0.9)
    s[:, 5] = rng.uniform(0., 1e-4, n)
    s[:, 6] = rng.uniform(0.005, 0.5, n)
    s[:, 7] = np.hypot(x["u"][:, 0], x["v"][:, 0]) + rng.uniform(0., 2., n)
    s[:, 8:10] = rng.normal(0., 4., (n, 2))
    s[:, 10] = rng.uniform(0.5, 5., n)
    s[:, 11] = rng.uniform(0.5, 5., n)
    s[:, 13] = rng.choice([250., 3000., 25000.], n) ** 2
    s[:, 14] = rng.uniform(0., 1., n)
    s[:, 15] = rng.uniform(0., 1., n)
    return {k: np.ascontiguousarray(x[k]) for k in _INPUTS}


def host_run(module, inputs, dt, table):
    n, k = inputs["u"].shape
    nt = inputs["q"].shape[2]
    capacity = (220 + 16 * nt) * k + 4096
    packed = np.ascontiguousarray(np.concatenate(
        [inputs[key].reshape(n, -1, order="F") for key in _INPUTS], axis=1).T)
    width = (5 + nt) * k + 5
    out = np.zeros(n * width)
    kpbl = np.zeros(n, np.int32)
    highwater = np.zeros(n, np.int32)
    module.get_function("satmedmf_column")(
        ((n + 31) // 32,), (32,),
        (np.int32(n), np.int32(k), np.int32(nt), np.float64(dt), packed, out, kpbl,
         np.zeros(n * capacity), table, highwater))
    result = out.reshape(width, n).T
    shapes = {"du": (n, k), "dv": (n, k), "tdt": (n, k), "rtg": (n, k, nt),
              "hpbl": (n,), "dkt": (n, k), "dku": (n, k), "sflux": (n, 4)}
    done, offset = {}, 0
    for key, shape in shapes.items():
        size = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        done[key] = result[:, offset:offset + size].reshape((n, *shape[1:]), order="F")
        offset += size
    done["kpbl"] = kpbl
    return done


def advance(inputs, out, dt):
    """The next call's inputs: the tendencies applied to the state."""
    x = {k: v.copy() for k, v in inputs.items()}
    x["u"] = x["u"] + dt * out["du"]
    x["v"] = x["v"] + dt * out["dv"]
    x["t"] = x["t"] + dt * out["tdt"]
    x["q"] = np.maximum(x["q"] + dt * out["rtg"], 0.)
    return x


def words(a):
    return np.ascontiguousarray(a).view(np.uint64 if a.dtype == np.float64 else np.uint32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 20, 48, 100])
    parser.add_argument("--tracers", type=int, nargs="+", default=[5, 9])
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    module = host_backend.build_module("satmedmf")
    xp = MetalArrayModule()
    context = xp.context
    report, bad = [], 0
    try:
        host_table = np.empty(7501)
        module.get_function("satmedmf_fpvs_table")((30,), (256,), (host_table,))
        metal_table = satmedmf_family.table(context).to_numpy()
        row = {"table_words": 7501, "table_mismatched_words":
               int(np.count_nonzero(words(host_table) != words(metal_table)))}
        bad += row["table_mismatched_words"]
        print(json.dumps(row))
        report.append(row)
        for nz in args.levels:
            for nt in args.tracers:
                x = columns(nz, nt, args.seed + 100 * nz + nt)
                n = x["u"].shape[0]
                p = np.asfortranarray(x["prsl"])
                host_exner = np.empty(n * nz)
                host_psk = np.empty(n)
                module.get_function("satmedmf_exner")(
                    ((n * nz + 255) // 256,), (256,),
                    (p.reshape(-1, order="F"), np.ascontiguousarray(x["prsi"][:, 0]),
                     host_exner, host_psk, np.int32(n), np.int32(nz)))
                m_exner, m_psk = satmedmf_family.exner(context, x["prsl"], x["prsi"][:, 0].copy())
                row = {"nz": nz, "tracers": nt, "columns": n, "mismatched_words": int(
                    np.count_nonzero(words(host_exner.reshape((n, nz), order="F")) != words(m_exner))
                    + np.count_nonzero(words(host_psk) != words(m_psk))),
                    "words_compared": int(n * nz + n)}
                state = x
                for call in range(2):
                    a = host_run(module, state, 60., host_table)
                    b = satmedmf_family.run_columns(context, state, dt=60.)
                    for key in OUTPUTS:
                        mism = int(np.count_nonzero(words(a[key]) != words(b[key])))
                        row["mismatched_words"] += mism
                        row["words_compared"] += int(a[key].size)
                        if mism:
                            row.setdefault("first_mismatches", []).append(
                                {"call": call, "field": key, "words": mism})
                    if call == 0:
                        row["changed_words"] = {k: int(np.count_nonzero(a[k])) for k in OUTPUTS}
                        row["pbl_columns_above_surface"] = int(np.count_nonzero(a["kpbl"] > 1))
                        row["tke_tendency_words"] = int(np.count_nonzero(a["rtg"][:, :, -1]))
                    state = advance(state, a, 60.)
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
