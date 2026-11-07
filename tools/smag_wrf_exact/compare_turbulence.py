"""Compare production turbulence callers with the host-compiled strict source.

This reference is a source-arithmetic authority, not NVIDIA GPU execution.
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
from tools.smag_wrf_exact.compare_tendencies import build_state, synthetic
from tools.smag_wrf_exact.generate import KERNELS, TURBULENCE_KERNELS, retained_source


def cases():
    from woof.core.state import adopt_wrf_theta_carrier
    rng = np.random.default_rng(31)
    for index, (bx, by) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        name, arrays, _, meta, _ = synthetic(
            f"boundary_{bx}{by}", rng, bx=bx, by=by, phb3d=index % 2)
        carrier = SimpleNamespace(thb=arrays["thb"], thp=arrays["thp"])
        adopt_wrf_theta_carrier(carrier)
        assert np.all(carrier.thb == np.float32(300.))
        nz, ny, nx = arrays["alt"].shape
        arrays["p"] = np.broadcast_to(np.linspace(98000., 66000., nz,
            dtype=np.float32)[:, None, None], (nz, ny, nx)).copy()
        arrays["qc"] = np.zeros_like(arrays["alt"])
        arrays["qc"][::2] = np.float32(1e-4)
        arrays["qi"] = np.full_like(arrays["alt"], 2e-5)
        arrays["tke"] = rng.uniform(-1e-5, 5., (nz, ny, nx)).astype(np.float32)
        arrays["tke"].flat[:4] = [-0., 0., 2. ** -140, -2. ** -140]
        meta.update(isfflx=index % 3, moist=index != 3, fluxes=index != 2,
                    qc=index != 1, qi=index == 0, budget=index % 2 == 0)
        yield name, arrays, meta


def run(arrays, meta, xp, metal, *, km_opt, isotropic):
    from woof.core.dycore import (launch_wrf_smag3d_km, launch_wrf_tke_km,
                                  _wrf_smag_prefix_view)
    from woof.core import tke_budget
    state = build_state(arrays, meta, xp, metal, bool(isotropic))
    nz, ny, nx = arrays["alt"].shape
    state.thp0 = xp.asarray(arrays["thp"] * np.float32(1.0625))
    for key, factor in (("tke", .9375), ("qc", .875), ("qi", .75)):
        setattr(state, key + "0", xp.asarray(arrays[key] * np.float32(factor)))
    for key, present in (("qv", meta["moist"]), ("qc", meta["qc"]), ("qi", meta["qi"])):
        if not present:
            setattr(state, key, None)
            setattr(state, key + "0", None)
    for key, value in (("mub2d", 9000.), ("mup", .125), ("mup0", -.25)):
        setattr(state, key, xp.asarray(np.full((ny, nx), value, np.float32)))
    state.c1h = xp.asarray(np.linspace(.125, 1., nz, dtype=np.float32))
    state.c2h = xp.asarray(np.linspace(3., .25, nz, dtype=np.float32))
    fields = {"ustm": xp.asarray(np.full((ny, nx), .23, np.float32)),
              "hfx": xp.asarray(np.full((ny, nx), 12.5, np.float32))}
    state.physics = SimpleNamespace(fields=fields if meta["fluxes"] else {})
    cfg = SimpleNamespace(dx=meta["dx"], dy=meta["dy"], nx=nx, ny=ny,
        open_x=bool(meta["bx"]), open_y=bool(meta["by"]), specified=False,
        nested=False, diff_opt=2, bl_pbl_physics=0, km_opt=km_opt, c_s=.25,
        c_k=.1, dt=12., mix_upper_bound=.5, mix_isotropic=isotropic,
        isfflx=meta["isfflx"], tke_drag_coefficient=0., tke_heat_flux=0.,
        tke_budget=int(meta["budget"]))
    rtke = state.scratch((nz, ny, nx), "smag_rtke")
    # Nonzero incoming tendencies distinguish sequential RHS increments.
    rtke[...] = xp.asarray(np.linspace(-.25, .125, nz * ny * nx,
        dtype=np.float32).reshape(nz, ny, nx))
    kmh, khh = (xp.zeros((nz, ny, nx), dtype=np.float32) for _ in range(2))
    launch = launch_wrf_tke_km if km_opt == 2 else launch_wrf_smag3d_km
    deformation = launch(state, cfg, kmh, khh, time_t=bool(isotropic))
    out = dict(zip(("d11", "d22", "d12"), deformation))
    bn2 = _wrf_smag_prefix_view(state, (nz, ny, nx + 1), (nz, ny, nx), "diff6_x")
    out.update(kmh=kmh, khh=khh, bn2=bn2, rtke=rtke,
        kmv=state.scratch((nz, ny, nx), "smag_kmv"),
        khv=state.scratch((nz, ny, nx), "smag_khv"))
    if km_opt == 2 and meta["budget"]:
        out.update({name: tke_budget.term(state, cfg, name)
                    for name in ("shear", "buoyancy", "dissipation", "limiter")})
    host = (lambda a: a.to_numpy()) if metal else np.asarray
    return {name: host(value).copy() for name, value in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from woof.wrf_exact import ENABLED, DIFFUSION_ENABLED
    if not (ENABLED and DIFFUSION_ENABLED):
        raise ValueError("select GPUWM_WRF_EXACT=1 before import")
    from tools.thompson_real_column_parity import host_backend
    host_backend.install()
    module = host_backend.HostModule("wrf_exact_smag_turbulence",
        "\n".join(retained_source(KERNELS + TURBULENCE_KERNELS)))
    from woof.core import dycore
    original = dycore.get_kernel
    dycore.get_kernel = lambda unit, name: (
        module.get_function(name) if unit == "smag2d" and name in KERNELS + TURBULENCE_KERNELS
        else original(unit, name))
    dycore.calc_n2_kernel = lambda: module.get_function("wrf_calc_n2")
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    report = {"reference": "host-compiled strict CUDA source", "cases": {}}
    with MetalContext() as context:
        xp = MetalArrayModule(context)
        for name, arrays, meta in cases():
            for km_opt in (2, 3):
                for isotropic in (0, 1):
                    expected = run(arrays, meta, np, False, km_opt=km_opt, isotropic=isotropic)
                    actual = run(arrays, meta, xp, True, km_opt=km_opt, isotropic=isotropic)
                    row = {key: compare_words(actual[key], expected[key]) for key in expected}
                    label = f"{name}:km{km_opt}:iso{isotropic}"
                    report["cases"][label] = row
                    print(label, json.dumps({key: sum(r[key] for r in row.values())
                        for key in ("words", "different_words", "subnormal_words")}), flush=True)
        from woof.core.moist import bound_tke
        words = np.array([0, 0x80000000, 1, 0x80000001, 0x7f800000,
            0xff800000, 0x7fc12345, 0x3f800000, 0x42800000], np.uint32)
        for upper in (50., -0., -1., float("nan")):
            values, bound = words.view(np.float32), np.float32(upper)
            expected = np.where(np.float32(0.) > bound, bound,
                np.where(values < 0., np.float32(0.), np.where(values > bound, bound, values)))
            actual = xp.asarray(values)
            bound_tke(actual, upper)
            report["cases"][f"bound:{upper}"] = {"tke": compare_words(actual.to_numpy(), expected)}
    checks = [r for case in report["cases"].values() for r in case.values()]
    for key in ("words", "different_words", "subnormal_words"):
        report[key] = sum(r[key] for r in checks)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("words", "different_words", "subnormal_words")}))
    if report["different_words"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
