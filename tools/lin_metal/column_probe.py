#!/usr/bin/env python3
"""Compare the production Lin launcher on Metal with the CUDA unit built for the host.

    python tools/lin_metal/column_probe.py --levels 2 12 40 100 128 --columns 48

Each level count runs two consecutive calls on randomized physical columns. The
host side is woof.core.lin.launch_lin over NumPy arrays through the host CUDA
backend; the Metal side is the same launcher over MetalArrays. Every output word
is compared as a uint32.
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
from woof.core.lin import launch_lin, launch_refl10cm_lin  # noqa: E402
from woof.metal.array import MetalArrayModule  # noqa: E402

VOLUMES = ("th", "qv", "qc", "qr", "qi", "qs", "qg")
SURFACES = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv", "sr")


def saturation(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / np.maximum(p - es, 1.0)


def columns(nz, nc, seed):
    rng = np.random.default_rng(seed)
    top = rng.uniform(14000., 22000., nc)
    edges = np.linspace(0., 1., nz + 1) ** rng.uniform(1.0, 1.4, (nc, 1))
    ht = rng.uniform(0., 3000., nc)
    zi = (edges * top[:, None]).T + ht
    z = 0.5 * (zi[1:] + zi[:-1])
    dz = np.diff(zi, axis=0)
    t0 = rng.uniform(235., 310., nc)
    lapse = rng.uniform(4., 8., nc) * 1e-3
    t = np.maximum(t0 - lapse * z, 195.) + rng.normal(0., 0.4, z.shape)
    p = np.empty_like(t)
    pm = rng.uniform(85000., 103000., nc)
    for k in range(nz):
        below = zi[k] if k == 0 else z[k - 1]
        tm = t[k]
        pm = pm * np.exp(-9.81 * (z[k] - below) / (287. * tm))
        p[k] = pm
    qs = saturation(t, p)
    rh = np.clip(rng.uniform(0.2, 1.1, (1, nc)) + rng.normal(0., .2, (nz, nc)), 0., 1.4)
    qv = rh * qs
    rho = p / (287. * t * (1. + 0.61 * qv))
    pii = (p / 1e5) ** (287. / 1004.)
    f = lambda a: np.ascontiguousarray(a, dtype=np.float32)
    kind = rng.integers(0, 6, nc)  # 0 zero, 1 warm, 2 cold, 3 all, 4 tiny, 5 sparse all
    hydro = {}
    for name in ("qc", "qr", "qi", "qs", "qg"):
        mag = 10. ** rng.uniform(-8.5, -2.6, (nz, nc))
        layer = rng.random((nz, nc)) < rng.uniform(.2, .8, (1, nc))
        if name in ("qi", "qs", "qg"):
            layer &= (t < 290.) | (kind == 3)
        value = np.where(layer, mag, 0.)
        value = np.where(kind == 0, 0., value)
        if name in ("qi", "qs", "qg"):
            value = np.where(kind == 1, 0., value)
        value = np.where(kind == 4, value * 10. ** rng.uniform(-34., -29., (nz, nc)), value)
        value = np.where(kind == 5, np.where(rng.random((nz, nc)) < .1, value, 0.), value)
        hydro[name] = value
    out = {"th": f(t / pii), "qv": f(qv), **{k: f(v) for k, v in hydro.items()}}
    fixed = {"rho": f(rho), "pii": f(pii), "pressure": f(p), "z": f(z), "dz": f(dz)}
    surf = {"ht": f(ht)}
    for name in SURFACES:
        surf[name] = f(rng.uniform(0., 5., nc)) if name.endswith("nc") else f(np.zeros(nc))
    return out, fixed, surf


def call(xp, state, fixed, surf, dt, nc):
    wrap = (lambda a: a) if xp is None else xp.asarray
    arrays = {k: wrap(v.reshape(v.shape[0], 1, nc) if v.ndim == 2 else v.reshape(1, nc))
              for k, v in {**state, **fixed, **surf}.items()}
    launch_lin(*(arrays[k] for k in VOLUMES), arrays["rho"], arrays["pii"], arrays["pressure"],
               arrays["z"], arrays["dz"], arrays["ht"], *(arrays[k] for k in SURFACES), dt)
    return {k: (a if xp is None else a.to_numpy()).reshape(state[k].shape if k in state else surf[k].shape)
            for k, a in arrays.items() if k in state or k in SURFACES}


def reflectivity(xp, nz, nc, seed):
    """Radar columns whose melting level, rain and ice layers vary."""
    rng = np.random.default_rng(seed)
    z = np.linspace(100., 16000., nz)[:, None]
    t = 273.15 + rng.uniform(-8., 14., (1, nc)) - 0.0065 * (z - 100.) * rng.uniform(.6, 1.2, (1, nc))
    p = 100000. * np.exp(-z / 7400.) * np.ones((1, nc))
    qv = 0.9 * saturation(t, p) * rng.uniform(.2, 1., (nz, nc))
    fields = []
    for name in ("qr", "qs", "qg"):
        mag = 10. ** rng.uniform(-10.5, -2.4, (nz, nc))
        on = rng.random((nz, nc)) < rng.uniform(.1, .9, (1, nc))
        fields.append(np.where(on, mag, 0.))
    arrays = [np.ascontiguousarray((a.reshape(nz, 1, nc)), dtype=np.float32)
              for a in (qv, *fields, t, p)]
    out = {}
    for name, wrap in (("host", lambda a: a), ("metal", xp.asarray)):
        refl = wrap(np.zeros((nz, 1, nc), np.float32))
        launch_refl10cm_lin(*(wrap(a.copy()) for a in arrays), refl)
        out[name] = refl if name == "host" else refl.to_numpy()
    return out["host"], out["metal"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 12, 40, 100, 128])
    parser.add_argument("--columns", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2110)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    xp = MetalArrayModule()
    report, bad = [], 0
    try:
        for nz in args.levels:
            nc = args.columns
            state, fixed, surf = columns(nz, nc, args.seed + nz)
            before = {k: v.copy() for k, v in {**state, **surf}.items()}
            host = {k: v.copy() for k, v in state.items()}
            hsurf = {k: v.copy() for k, v in surf.items()}
            metal, msurf = {k: v.copy() for k, v in state.items()}, {k: v.copy() for k, v in surf.items()}
            row = {"nz": nz, "columns": nc, "mismatched_words": 0, "changed_words": {}}
            for dt in (60., 30.):
                a = call(None, host, fixed, hsurf, dt, nc)
                b = call(xp, metal, fixed, msurf, dt, nc)
                for k in a:
                    mism = int(np.count_nonzero(a[k].view(np.uint32) != b[k].view(np.uint32)))
                    row["mismatched_words"] += mism
                    if mism:
                        w = np.flatnonzero(a[k].view(np.uint32).reshape(-1) != b[k].view(np.uint32).reshape(-1))
                        row.setdefault("first_mismatches", []).append(
                            {"dt": dt, "field": k, "words": mism, "index": [int(i) for i in np.unravel_index(w[0], a[k].shape)],
                             "host": float(a[k].reshape(-1)[w[0]]), "metal": float(b[k].reshape(-1)[w[0]])})
                host.update({k: a[k] for k in state})
                hsurf.update({k: a[k] for k in SURFACES})
                metal.update({k: b[k] for k in state})
                msurf.update({k: b[k] for k in SURFACES})
            for k in {**state, **surf}:
                now = {**host, **hsurf}[k]
                row["changed_words"][k] = int(np.count_nonzero(now.view(np.uint32) != before[k].view(np.uint32)))
            row["words_compared"] = 2 * sum(v.size for v in {**state, **surf}.values())
            hr, mr = reflectivity(xp, nz, nc, args.seed + 7 * nz)
            row["refl_mismatched_words"] = int(np.count_nonzero(hr.view(np.uint32) != mr.view(np.uint32)))
            row["refl_nontrivial_words"] = int(np.count_nonzero(hr > np.float32(-35.)))
            row["refl_words"] = int(hr.size)
            bad += row["mismatched_words"] + row["refl_mismatched_words"]
            report.append(row)
            print(json.dumps(row))
    finally:
        xp.context.close()
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return int(bad != 0)


if __name__ == "__main__":
    raise SystemExit(main())
