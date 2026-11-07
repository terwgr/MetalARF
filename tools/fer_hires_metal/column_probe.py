#!/usr/bin/env python3
"""Compare the production Ferrier-Aligo launcher on Metal with the CUDA unit built for the host.

    python tools/fer_hires_metal/column_probe.py --levels 2 12 40 100 200 --columns 64

The init tables are built by fer_hires_init's kernels on both sides and compared
word for word; each level count then runs consecutive calls (reflectivity off,
on, off, and the analysis-only moments) over randomized physical columns.
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
host_backend.install()
from woof.core import fer_hires  # noqa: E402
from woof.core.kernels import get_kernel  # noqa: E402
from woof.metal.array import MetalArrayModule  # noqa: E402
from woof.metal.fer_hires_family import device_tables  # noqa: E402

VOLUMES = ("th", "qv", "qc", "qr", "qi", "qt", "f_ice", "f_rain", "f_rimef")
SURFACES = ("rainnc", "rainncv", "sr")


def saturation(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / np.maximum(p - es, 1.0)


def columns(nz, nc, seed):
    rng = np.random.default_rng(seed)
    ht = rng.uniform(0., 3000., nc)
    edges = np.linspace(0., 1., nz + 1) ** rng.uniform(1.0, 1.4, (nc, 1))
    zi = (edges * rng.uniform(14000., 22000., nc)[:, None]).T + ht
    z = 0.5 * (zi[1:] + zi[:-1])
    dz = np.diff(zi, axis=0)
    t = np.maximum(rng.uniform(235., 310., nc) - rng.uniform(4., 8., nc) * 1e-3 * z, 195.)
    t = t + rng.normal(0., 0.4, z.shape)
    p = np.empty_like(t)
    pm = rng.uniform(85000., 103000., nc)
    for k in range(nz):
        below = zi[0] if k == 0 else z[k - 1]
        pm = pm * np.exp(-9.81 * (z[k] - below) / (287. * t[k]))
        p[k] = pm
    qs = saturation(t, p)
    qv = np.clip(rng.uniform(0.2, 1.1, (1, nc)) + rng.normal(0., .2, (nz, nc)), 0., 1.4) * qs
    rho = p / (287. * t * (1. + 0.61 * qv))
    pii = (p / 1e5) ** (287. / 1004.)
    kind = rng.integers(0, 6, nc)  # zero, warm, cold, all, tiny, sparse
    hydro = {}
    for name in ("qc", "qr", "qi"):
        mag = 10. ** rng.uniform(-9., -2.6, (nz, nc))
        layer = rng.random((nz, nc)) < rng.uniform(.2, .8, (1, nc))
        if name == "qi":
            layer &= (t < 290.) | (kind == 3)
        value = np.where(layer, mag, 0.)
        value = np.where(kind == 0, 0., value)
        if name == "qi":
            value = np.where(kind == 1, 0., value)
        value = np.where(kind == 4, value * 10. ** rng.uniform(-34., -29., (nz, nc)), value)
        hydro[name] = np.where(kind == 5, np.where(rng.random((nz, nc)) < .1, value, 0.), value)
    f = lambda a: np.ascontiguousarray(a.reshape(a.shape[0], 1, nc), dtype=np.float32)
    zero = f(np.zeros((nz, nc)))
    state = {"th": f(t / pii), "qv": f(qv), **{k: f(v) for k, v in hydro.items()},
             "qt": zero.copy(), "f_ice": zero.copy(), "f_rain": zero.copy(),
             "f_rimef": f(np.where(rng.random((nz, nc)) < .5, 1., rng.uniform(1., 45., (nz, nc))))}
    fixed = {"p": f(p), "pii": f(pii), "rho": f(rho), "dz8w": f(dz)}
    surf = {"rainnc": np.ascontiguousarray(rng.uniform(0., 5., (1, nc)), dtype=np.float32),
            "rainncv": np.zeros((1, nc), np.float32), "sr": np.zeros((1, nc), np.float32)}
    return state, fixed, surf


def tables(xp, dtph):
    host = fer_hires.host_table_words(fer_hires.load_table())
    get_kernel("fer_hires", "fer_hires_init_tables")(((fer_hires.NX + 255) // 256,), (256,), (host,))
    get_kernel("fer_hires", "fer_hires_init_scalars")((1,), (1,), (host, np.float32(dtph)))
    return host, device_tables(xp.context, dtph)


def launch(wrap, state, fixed, surf, tab, dt, gid, refl, analysis):
    a = {k: wrap(v) for k, v in {**state, **fixed, **surf}.items()}
    r = None if refl is None else wrap(np.zeros_like(state["th"]))
    fer_hires.launch_fer_hires(*(a[k] for k in VOLUMES), a["p"], a["pii"], a["rho"], a["dz8w"],
                               *(a[k] for k in SURFACES), tab, dt, gid, r, analysis_only=analysis)
    out = {k: a[k] for k in (*VOLUMES, *SURFACES)}
    if r is not None:
        out["refl"] = r
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 12, 40, 100, 200])
    parser.add_argument("--columns", type=int, default=64)
    parser.add_argument("--seed", type=int, default=5110)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    xp = MetalArrayModule()
    report, bad = [], 0
    try:
        host_tab, metal_tab = tables(xp, 60.)
        table_bad = int(np.count_nonzero(host_tab.view(np.uint32) != metal_tab.to_numpy().view(np.uint32)))
        print(json.dumps({"table_words": int(host_tab.size), "table_mismatched_words": table_bad}))
        bad += table_bad
        report.append({"table_words": int(host_tab.size), "table_mismatched_words": table_bad})
        for nz in args.levels:
            nc = args.columns
            state, fixed, surf = columns(nz, nc, args.seed + nz)
            before = {k: v.copy() for k, v in {**state, **surf}.items()}
            h, hs = {k: v.copy() for k, v in state.items()}, {k: v.copy() for k, v in surf.items()}
            m, ms = {k: v.copy() for k, v in state.items()}, {k: v.copy() for k, v in surf.items()}
            row = {"nz": nz, "columns": nc, "mismatched_words": 0, "words_compared": 0}
            for dt, gid, refl, analysis in ((60., 1, None, False), (60., 1, True, False),
                                            (30., 2, None, False), (60., 1, True, True)):
                a = launch(lambda v: v, h, fixed, hs, host_tab, dt, gid, refl, analysis)
                b = launch(xp.asarray, m, fixed, ms, metal_tab, dt, gid, refl, analysis)
                for k in a:
                    got = b[k].to_numpy()
                    mism = int(np.count_nonzero(a[k].view(np.uint32) != got.view(np.uint32)))
                    row["mismatched_words"] += mism
                    row["words_compared"] += int(a[k].size)
                    if mism:
                        row.setdefault("first_mismatches", []).append(
                            {"call": [dt, gid, refl, analysis], "field": k, "words": mism})
                    if k == "refl":
                        row["refl_nontrivial_words"] = row.get("refl_nontrivial_words", 0) + int(
                            np.count_nonzero(a[k] != np.float32(-20.)))
                h.update({k: a[k] for k in state})
                hs.update({k: a[k] for k in surf})
                m.update({k: b[k].to_numpy() for k in state})
                ms.update({k: b[k].to_numpy() for k in surf})
            row["changed_words"] = {k: int(np.count_nonzero({**h, **hs}[k].view(np.uint32) != before[k].view(np.uint32)))
                                    for k in {**state, **surf}}
            row["table_mismatched_words_after"] = int(np.count_nonzero(
                host_tab.view(np.uint32) != metal_tab.to_numpy().view(np.uint32)))
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
