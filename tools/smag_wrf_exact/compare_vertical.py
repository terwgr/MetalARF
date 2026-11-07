"""Compare the PBL-off vertical and surface production caller as literal words.

The reference compiles the retained strict CUDA source closure with separate
IEEE binary32 operations. Cases isolate terrain, staggering, physical and
periodic boundaries, time levels, surface forcing and gradual underflow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.smag_wrf_exact.compare import compare_words
from tools.smag_wrf_exact.compare_tendencies import build_state, seed_tendency, synthetic
from tools.smag_wrf_exact.generate import VERTICAL_KERNELS, retained_source


def cases():
    rng = np.random.default_rng(21)
    for index, (bx, by) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        name, arrays, ops, meta, _ = synthetic(
            f"boundary_{bx}{by}", rng, bx=bx, by=by, phb3d=index % 2)
        meta.update(isfflx=(0, 1, 2, 2)[index], active=index != 3,
                    moist=index != 3, tiny=False, w_override=index != 0)
        yield name, arrays, ops, meta
    for mode in ("stress", "constant", "wind"):
        name, arrays, ops, meta, _ = synthetic(
            "underflow_" + mode, rng, bx=1, by=1, phb3d=True,
            scale_km=2. ** -138)
        meta.update(isfflx=0 if mode == "constant" else 1,
                    active=True, moist=True, tiny=True, underflow=mode)
        if mode == "wind":
            for field in ("u", "v", "w"):
                arrays[field] *= np.float32(2. ** -65)
        yield name, arrays, ops, meta


    # Real exact-mode preparation adopts theta-300 and a constant 300 K
    # base. Nonconstant bases above exercise leaf arithmetic only.
    from woof.core.state import adopt_wrf_theta_carrier
    for phb3d in (False, True):
        name, arrays, ops, meta, _ = synthetic(
            "adopted_carrier_" + ("3d" if phb3d else "1d"), rng,
            bx=1, by=0, phb3d=phb3d)
        carrier = SimpleNamespace(thb=arrays["thb"], thp=arrays["thp"])
        adopt_wrf_theta_carrier(carrier)
        assert np.all(carrier.thb == np.float32(300.))
        arrays.update(thb=carrier.thb, thp=carrier.thp)
        meta.update(isfflx=2, active=True, moist=True, tiny=False,
                    w_override=phb3d)
        yield name, arrays, ops, meta


def run(arrays, ops, meta, xp, metal, *, time_t, surface_only):
    from woof.core.dycore import launch_wrf_smag2d_vertical
    state = build_state(arrays, meta, xp, metal, time_t)
    nz, ny, nx = arrays["alt"].shape
    plane = (ny, nx)
    if not meta["moist"]:
        state.qv = state.qv0 = None
    state.mup0 = xp.zeros(plane, dtype=np.float32)
    tiny = meta["tiny"]
    ust = np.float32(2. ** -70 if meta.get("underflow") == "stress" else .31)
    heat = np.float32(2. ** -140 if tiny else .125)
    cd0 = np.float32(2. ** -140 if tiny else .007)
    hfx = np.arange(ny * nx, dtype=np.float32).reshape(plane) + np.float32(.23)
    qfx = (hfx - np.float32(3.1)) * np.float32(1e-5)
    if tiny:
        hfx *= np.float32(2. ** -130)
        qfx *= np.float32(2. ** -120)
    fields = {"ustm": xp.asarray(np.full(plane, ust, np.float32)),
              "hfx": xp.asarray(hfx), "qfx": xp.asarray(qfx)}
    # The inactive case has no surface writers, matching the caller's dummy
    # buffers and write_hfx=False branch.
    state.physics = SimpleNamespace(fields=fields if meta["active"] else {})
    cfg = SimpleNamespace(dx=meta["dx"], dy=meta["dy"], nx=nx, ny=ny,
                          open_x=bool(meta["bx"]), open_y=bool(meta["by"]),
                          specified=False, nested=False, isfflx=meta["isfflx"],
                          sf_sfclay_physics=91 if meta["active"] else 0,
                          tke_drag_coefficient=cd0, tke_heat_flux=heat)
    shapes = {"ru": arrays["u"].shape, "rv": arrays["v"].shape,
              "rw": arrays["w"].shape, "rth": arrays["alt"].shape,
              "rqv": arrays["alt"].shape, "rscalar": arrays["alt"].shape}
    targets = {name: xp.asarray(seed_tendency(shape, not tiny))
               for name, shape in shapes.items()}
    coefficients = {name: xp.asarray(ops[name]) for name in ("kmh", "kmv", "khh")}
    theta = xp.asarray(arrays["thp"] * np.float32(1.125 if time_t else 1.))
    passive = xp.asarray(arrays["qv"] * np.float32(.75 if time_t else 1.))
    scalar_rows = ((theta, targets["rth"], True),
                   (passive, targets["rqv"], False),
                   (passive, targets["rscalar"], False))
    launch_wrf_smag2d_vertical(
        state, cfg, coefficients["kmh"],
        **{name: targets[name] for name in ("ru", "rv", "rw", "rth", "rqv")},
        time_t=time_t, kmv=coefficients["kmv"], khv=coefficients["khh"],
        scalar_rows=scalar_rows,
        w_km=coefficients["kmh"] if meta.get("w_override", True) else None,
        interior=not surface_only)
    host = (lambda a: a.to_numpy()) if metal else np.asarray
    out = {name: host(value).copy() for name, value in targets.items()}
    out["hfx"] = host(state.physics.fields.get("hfx", state.mup0)).copy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cases", default="", help="comma list of arithmetic cases; empty selects all")
    args = ap.parse_args()
    from woof.wrf_exact import ENABLED, DIFFUSION_ENABLED
    if not (ENABLED and DIFFUSION_ENABLED):
        raise ValueError("select GPUWM_WRF_EXACT=1 before import")
    from tools.thompson_real_column_parity import host_backend
    host_backend.install()
    source = "\n".join(retained_source(VERTICAL_KERNELS))
    module = host_backend.HostModule("wrf_exact_smag_vertical", source)
    from woof.core import dycore
    original = dycore.get_kernel
    dycore.get_kernel = lambda unit, name: (
        module.get_function(name) if unit == "smag2d" and name in VERTICAL_KERNELS
        else original(unit, name))
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    selected = set(filter(None, args.cases.split(",")))
    report = {"cases": {}}
    with MetalContext() as context:
        xp = MetalArrayModule(context)
        for name, arrays, ops, meta in cases():
            if selected and name not in selected:
                continue
            for time_t in (False, True):
                for surface_only in (False, True):
                    expected = run(arrays, ops, meta, np, False,
                                   time_t=time_t, surface_only=surface_only)
                    actual = run(arrays, ops, meta, xp, True,
                                 time_t=time_t, surface_only=surface_only)
                    row = {key: compare_words(actual[key], expected[key]) for key in expected}
                    label = f"{name}:t{int(time_t)}:surface{int(surface_only)}"
                    report["cases"][label] = row
                    print(label, json.dumps({
                        "words": sum(r["words"] for r in row.values()),
                        "different_words": sum(r["different_words"] for r in row.values()),
                        "subnormal_words": sum(r["subnormal_words"] for r in row.values())}), flush=True)
    checks = [r for case in report["cases"].values() for r in case.values()]
    for key in ("words", "different_words", "subnormal_words"):
        report[key] = sum(r[key] for r in checks)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("words", "different_words", "subnormal_words")}))
    if report["different_words"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
