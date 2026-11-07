#!/usr/bin/env python3
"""Compare Metal YSU words with the CUDA source built as host C++.

ysu_480.cu (ysu_version="wrf_480") is built through
tools/thompson_real_column_parity/host_backend.py (g++/clang++, no multiply-add
contraction, the module's own gfk powf/expf) and run column by column in the
launcher's argument order. The Metal side is submit_ysu_column itself. The
probe reports words and mismatches, never a tolerance. It also reruns the host
kernel with each changed branch switched back to the 4.6.1 form (the top-down
entrainment cap, the entrainment Courant bound) and with the 4.6.1 module, so
the output shows the branches were reached.

Run device work through the project's GPU queue.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "thompson_real_column_parity"))

VOLUME = ("du", "dv", "dtheta", "dqv", "dqc", "dqi", "exch_h", "exch_m")
SURFACE = ("hpbl", "kpbl", "wstar", "delta", "topdown_radsum", "wstar3_2",
           "cloudflg")
BEP_TERMS = ("a_u_bep", "a_v_bep", "a_t_bep", "a_q_bep", "b_u_bep",
             "b_v_bep", "b_t_bep", "b_q_bep", "sf_bep", "vl_bep")


def make_case(seed, nz, ny, nx, *, thin=False, bep=False, topo=False,
              stable_only=False, gmax=0.12):
    rng = np.random.default_rng(seed)
    n = ny * nx
    f32 = np.float32
    psfc = rng.uniform(85000., 102000., n)
    dz0 = rng.uniform(8., 60., n) if thin else rng.uniform(15., 70., n)
    growth = rng.uniform(1.0, 1.0 + (0.05 if thin else gmax), n)
    dz = dz0[None] * growth[None] ** np.arange(nz)[:, None]
    zi = np.concatenate([np.zeros((1, n)), np.cumsum(dz, 0)])
    zc = .5 * (zi[1:] + zi[:-1])
    scale_h = 8000.
    p_if = psfc[None] * np.exp(-zi / scale_h)
    p = np.sqrt(p_if[:-1] * p_if[1:])
    exner = (p / 1.0e5) ** 0.2857142857
    unstable = (rng.random(n) < (0.0 if stable_only else 0.6))
    th_sfc = rng.uniform(278., 312., n)
    hmix = rng.uniform(150., 2200., n)
    inv = rng.uniform(0.2, 5.0, n) if not thin else rng.uniform(0.02, 0.6, n)
    gam_above = rng.uniform(2e-3, 6e-3, n)
    gam_stable = rng.uniform(4e-3, 1.5e-2, n)
    theta = np.where(
        unstable[None],
        th_sfc[None] + np.where(zc < hmix[None], 1e-4 * zc,
                                inv[None] + 1e-4 * hmix[None]
                                + gam_above[None] * (zc - hmix[None])),
        th_sfc[None] + gam_stable[None] * zc)
    theta = theta + rng.normal(0., 0.03, theta.shape)
    qs = rng.uniform(0.002, 0.018, n)
    qv = qs[None] * np.exp(-zc / rng.uniform(1500., 3500., n)[None]) \
        * rng.uniform(0.9, 1.05, theta.shape)
    qc = np.zeros_like(theta)
    qi = np.zeros_like(theta)
    rth = rng.normal(0., 2e-6, theta.shape)
    cloudy = rng.random(n) < 0.55
    for c in np.flatnonzero(cloudy):
        top = int(np.searchsorted(zc[:, c], hmix[c]))
        top = min(max(top, 2), nz - 3)
        span = rng.integers(1, 4)
        for k in range(max(top - span, 0), min(top + 2, nz - 2)):
            qc[k, c] = rng.uniform(1e-5, 8e-4)
            rth[k, c] = -rng.uniform(1e-5, 6e-4)
        if rng.random() < .3 and top + 3 < nz:
            qi[top + 2, c] = rng.uniform(1e-5, 2e-4)
    ush = rng.uniform(-3e-3, 3e-3, n)
    u = rng.uniform(-14., 14., n)[None] + ush[None] * zc
    v = rng.uniform(-14., 14., n)[None] + rng.uniform(-3e-3, 3e-3, n)[None] * zc
    ust = np.exp(rng.uniform(np.log(0.03), np.log(1.1), n))
    hfx = np.where(unstable, rng.uniform(5., 450., n), -rng.uniform(0., 70., n))
    qfx = rng.uniform(0., 3e-4, n)
    br = np.where(unstable, -rng.uniform(0.005, 2.0, n), rng.uniform(0.005, 1.2, n))
    psim = rng.uniform(2., 9., n)
    psih = rng.uniform(2., 9., n)
    xland = np.where(rng.random(n) < .35, 2., 1.)
    znt = np.exp(rng.uniform(np.log(1e-4), np.log(1.), n))
    u0, v0 = u[0], v[0]
    wspd = np.sqrt(u0 ** 2 + v0 ** 2) + rng.uniform(0., 1., n)
    u10 = .7 * u0
    v10 = .7 * v0
    # Edge columns: calm coupling, a tiny wspd, a tiny ust (prfac2 rescue).
    edge = rng.permutation(n)[:12]
    ust[edge[0]] = 0.; hfx[edge[0]] = 0.; qfx[edge[0]] = 0.
    ust[edge[1]] = 0.2; hfx[edge[1]] = 0.; qfx[edge[1]] = 0.
    wspd[edge[2]] = 1e-10
    ust[edge[3]] = 1e-16; br[edge[3]] = -0.3; hfx[edge[3]] = 80.
    ust[edge[4]] = 0.; hfx[edge[4]] = 40.; br[edge[4]] = -0.2
    case = {name: np.ascontiguousarray(a, f32).reshape(shape) for name, a, shape in (
        ("u", u, (nz, ny, nx)), ("v", v, (nz, ny, nx)),
        ("theta", theta, (nz, ny, nx)), ("qv", qv, (nz, ny, nx)),
        ("qc", qc, (nz, ny, nx)), ("qi", qi, (nz, ny, nx)),
        ("p", p, (nz, ny, nx)), ("p_interface", p_if, (nz + 1, ny, nx)),
        ("exner", exner, (nz, ny, nx)), ("dz", dz, (nz, ny, nx)),
        ("rthraten", rth, (nz, ny, nx)),
        ("psfc", psfc, (ny, nx)), ("znt", znt, (ny, nx)), ("ust", ust, (ny, nx)),
        ("hfx", hfx, (ny, nx)), ("qfx", qfx, (ny, nx)), ("wspd", wspd, (ny, nx)),
        ("br", br, (ny, nx)), ("psim", psim, (ny, nx)), ("psih", psih, (ny, nx)),
        ("xland", xland, (ny, nx)), ("u10", u10, (ny, nx)), ("v10", v10, (ny, nx)))}
    if bep:
        shp = (nz, ny, nx)
        bep_terms = {
            "a_u_bep": rng.uniform(0., 4e-3, shp), "a_v_bep": rng.uniform(0., 4e-3, shp),
            "a_t_bep": rng.uniform(0., 2e-3, shp), "a_q_bep": rng.uniform(0., 2e-3, shp),
            "b_u_bep": rng.normal(0., 2e-4, shp), "b_v_bep": rng.normal(0., 2e-4, shp),
            "b_t_bep": rng.normal(0., 1e-3, shp), "b_q_bep": rng.normal(0., 1e-7, shp),
            "sf_bep": rng.uniform(0.3, 1., (nz + 1, ny, nx)),
            "vl_bep": rng.uniform(0.4, 1., shp)}
        case["bep"] = {k: np.ascontiguousarray(v_, f32) for k, v_ in bep_terms.items()}
        case["frc_urb2d"] = np.ascontiguousarray(rng.uniform(0.1, 1., (ny, nx)), f32)
    if topo:
        case["ctopo"] = np.ascontiguousarray(rng.uniform(0.3, 1., (ny, nx)), f32)
        case["ctopo2"] = np.ascontiguousarray(rng.uniform(0.5, 1., (ny, nx)), f32)
    return case


def host_module(label, define=""):
    from host_backend import HostModule
    from woof.core.kernels import module_source
    text = module_source(label.split(":")[0]).replace("__launch_bounds__(32, 16)", "")
    return HostModule(label, define + text)


def run_host(module, case, dt, pblmix):
    nz, ny, nx = case["theta"].shape
    n = ny * nx
    outs = {name: np.zeros((nz, ny, nx), np.float32) for name in VOLUME}
    for name in SURFACE:
        outs[name] = np.zeros((ny, nx), np.int32 if name in ("kpbl", "cloudflg")
                              else np.float32)
    ws = np.zeros(((n + 31) // 32) * 18 * (nz + 1) * 32, np.float32)
    args = [case[k] for k in ("u", "v", "theta", "qv", "qc", "qi", "p",
                              "p_interface", "exner", "dz", "rthraten", "psfc",
                              "znt", "ust", "hfx", "qfx", "wspd", "br", "psim",
                              "psih", "xland", "u10", "v10")]
    args += [outs["du"], outs["dv"], outs["dtheta"], outs["dqv"], outs["dqc"],
             outs["dqi"], outs["hpbl"], outs["kpbl"], outs["exch_h"],
             outs["exch_m"], outs["wstar"], outs["delta"], np.float32(dt),
             outs["topdown_radsum"], outs["wstar3_2"], outs["cloudflg"],
             np.int32(pblmix), np.int32(nz), np.int32(ny), np.int32(nx), ws,
             np.int32(nz + 1), np.int32(0)]
    if "bep" in case:
        entry = "ysu_column_bep"
        args += [case["bep"][k] for k in BEP_TERMS] + [case["frc_urb2d"]]
    elif "ctopo" in case:
        entry = "ysu_column_topo"
        outs["u10o"] = np.zeros((ny, nx), np.float32)
        outs["v10o"] = np.zeros((ny, nx), np.float32)
        args += [case["ctopo"], case["ctopo2"], outs["u10o"], outs["v10o"]]
    else:
        entry = "ysu_column"
    module.get_function(entry)(((n + 31) // 32,), (32,), tuple(args))
    return outs


def run_metal(context, xp, case, dt, pblmix):
    from woof.metal.ysu import submit_ysu_column
    nz, ny, nx = case["theta"].shape
    dev = {k: xp.asarray(case[k]) for k in (
        "u", "v", "theta", "qv", "qc", "qi", "p", "p_interface", "exner", "dz",
        "rthraten", "psfc", "znt", "ust", "hfx", "qfx", "wspd", "br", "psim",
        "psih", "xland", "u10", "v10")}
    extra = {}
    outs_extra = {}
    if "bep" in case:
        extra["bep"] = {k: xp.asarray(v) for k, v in case["bep"].items()}
        extra["frc_urb2d"] = xp.asarray(case["frc_urb2d"])
    elif "ctopo" in case:
        extra["topo"] = (xp.asarray(case["ctopo"]), xp.asarray(case["ctopo2"]))
        outs_extra["u10o"] = xp.zeros((ny, nx), np.float32)
        outs_extra["v10o"] = xp.zeros((ny, nx), np.float32)
        extra["u10_out"] = outs_extra["u10o"]
        extra["v10_out"] = outs_extra["v10o"]
    out = submit_ysu_column(
        context, dev["u"], dev["v"], dev["theta"], dev["qv"], dev["qc"],
        dev["qi"], dev["p"], dev["p_interface"], dev["exner"], dev["dz"],
        dev["rthraten"], psfc=dev["psfc"], znt=dev["znt"], ust=dev["ust"],
        hfx=dev["hfx"], qfx=dev["qfx"], wspd=dev["wspd"], br=dev["br"],
        psim=dev["psim"], psih=dev["psih"], xland=dev["xland"], u10=dev["u10"],
        v10=dev["v10"], dt=dt, ysu_topdown_pblmix=pblmix,
        ysu_version="wrf_480", **extra)
    result = {k: v.to_numpy() for k, v in out.items()}
    for k, v in outs_extra.items():
        result[k] = v.to_numpy()
    return result


def words(a):
    return np.ascontiguousarray(a).view(np.uint32).reshape(-1)


def count(expected, actual, fields):
    bad = total = 0
    per = {}
    for f in fields:
        if f not in expected:
            continue
        e, a = words(expected[f]), words(actual[f])
        m = int((e != a).sum())
        per[f] = m
        bad += m
        total += e.size
    return total, bad, {k: v for k, v in per.items() if v}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/woof-ysu-word-probe.json"))
    parser.add_argument("--cases", default="")
    args = parser.parse_args()
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    cases = [
        ("default_nz49", dict(seed=1, nz=49, ny=24, nx=32), 24., 1),
        ("default_nz49_pblmix0", dict(seed=2, nz=49, ny=16, nx=32), 24., 0),
        ("courant_thin_nz40", dict(seed=3, nz=40, ny=24, nx=32, thin=True), 300., 1),
        ("shallow_nz4", dict(seed=4, nz=4, ny=16, nx=32), 24., 1),
        ("stable_nz30", dict(seed=5, nz=30, ny=16, nx=32, stable_only=True), 60., 1),
        ("topo_nz49", dict(seed=6, nz=49, ny=24, nx=32, topo=True), 24., 1),
        ("bep_nz49", dict(seed=7, nz=49, ny=24, nx=32, bep=True), 24., 1),
        ("bep_thin_nz40", dict(seed=8, nz=40, ny=24, nx=32, bep=True, thin=True), 300., 1),
        ("default_nz80", dict(seed=9, nz=80, ny=16, nx=32, gmax=0.03), 36., 1),
        ("topo_thin_nz40", dict(seed=10, nz=40, ny=24, nx=32, topo=True, thin=True), 240., 1),
        ("bep_stable_nz30", dict(seed=11, nz=30, ny=16, nx=32, bep=True, stable_only=True), 60., 1),
    ]
    wanted = set(filter(None, args.cases.split(",")))
    new = host_module("ysu_480")
    cap_off = host_module("ysu_480:legacy_topdown", "#define YSU_LEGACY_TOPDOWN 1\n")
    courant_off = host_module("ysu_480:no_courant", "#define GPUWM_WRF_EXACT 1\n")
    old = host_module("ysu")
    records = []
    with MetalContext() as context:
        xp = MetalArrayModule(context)
        for name, kw, dt, pblmix in cases:
            if wanted and name not in wanted:
                continue
            case = make_case(**kw)
            expected = run_host(new, case, dt, pblmix)
            actual = run_metal(context, xp, case, dt, pblmix)
            fields = VOLUME + SURFACE + ("u10o", "v10o")
            total, bad, per = count(expected, actual, fields)
            nonfinite = {
                f: {"expected": int(np.count_nonzero(~np.isfinite(expected[f]))),
                    "actual": int(np.count_nonzero(~np.isfinite(actual[f])))}
                for f in fields if f in expected and expected[f].dtype.kind == "f"
                and (not np.isfinite(expected[f]).all() or not np.isfinite(actual[f]).all())
            }
            record = {"case": name, "words": total, "mismatches": bad,
                      "by_field": per, "nonfinite_by_field": nonfinite}
            for label, module in (("cap_removed", cap_off),
                                  ("courant_removed", courant_off),
                                  ("wrf_461_module", old)):
                if label == "wrf_461_module" and "bep" in case:
                    continue
                other = run_host(module, case, dt, pblmix)
                _, differ, where = count(expected, other, fields)
                record["host_words_differing_" + label] = differ
            records.append(record)
            print(json.dumps(record), flush=True)
    args.out.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"words": sum(r["words"] for r in records),
                      "mismatches": sum(r["mismatches"] for r in records)}))
    return 0 if all(r["mismatches"] == 0 for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
