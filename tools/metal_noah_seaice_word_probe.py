#!/usr/bin/env python3
"""Compare Metal's Noah sea-ice column with the CUDA source built as host C++.

noah_seaice_column is compiled from woof.core.kernels.module_source("noah")
through tools/thompson_real_column_parity/host_backend.py (no FMA contraction)
and run on randomized columns that reach every branch (skipped open water,
snow floor, snow depth clamp, precipitation phase, snow melt, frozen-rain
heat, snow density fatal). The Metal side runs submit_noah_seaice on the same
arrays. Words are compared bit for bit; no tolerance.

Run device work through the project's GPU queue.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/thompson_real_column_parity"))

SHIM_PREFIX = ("#define __launch_bounds__(...)\n"
               "static inline int atomicExch(int* a, int v)"
               " { int o = *a; *a = v; return o; }\n")


def columns(rng, ny, nx):
    shape = (ny, nx)
    u = lambda lo, hi: rng.uniform(lo, hi, shape).astype(np.float32)
    f = {}
    f["xice"] = np.where(rng.random(shape) < .8, u(.5, 1.), u(0., .5)).astype(np.float32)
    f["psfc"] = u(8.0e4, 1.05e5)
    f["sfcprs"] = (f["psfc"] * u(.97, 1.)).astype(np.float32)
    f["sfctmp"] = u(235., 285.)
    f["qv1"] = u(0., .012)
    f["qgh"] = u(1e-4, .02)
    f["glw"] = u(100., 400.)
    f["swdown"] = np.where(rng.random(shape) < .3, 0., u(0., 900.)).astype(np.float32)
    f["rainbl"] = np.where(rng.random(shape) < .5, 0., u(0., 3.)).astype(np.float32)
    f["sr"] = np.where(rng.random(shape) < .3, 1., u(0., 1.)).astype(np.float32)
    f["rib"] = u(-1., 1.)
    f["chs"] = u(1e-4, .05)
    f["cqs2"] = u(1e-4, .05)
    f["chs2"] = u(1e-4, .05)
    f["tsk"] = np.where(rng.random(shape) < .3, u(273.0, 274.5), u(235., 273.)).astype(np.float32)
    f["emiss"] = u(.9, 1.)
    f["albedo"] = u(.2, .9)
    f["z0"] = u(1e-4, .05)
    snow = np.where(rng.random(shape) < .25, 0., u(0., 500.)).astype(np.float32)
    snow = np.where(rng.random(shape) < .1, u(0., 10.), snow).astype(np.float32)
    snow[rng.random(shape) < .02] = np.float32(1e-42)       # positive subnormal
    f["snow"] = snow
    f["snowc"] = np.where(rng.random(shape) < .3, 1., u(0., 1.)).astype(np.float32)
    snowh = np.maximum(snow * 1e-3 / u(.05, .4), 1e-3).astype(np.float32)
    bad = rng.random(shape) < .03                         # density above 1
    snowh = np.where(bad, snow * 1e-3 / u(1.1, 3.), snowh).astype(np.float32)
    snowh = np.where(rng.random(shape) < .02, u(0., 2e-3), snowh).astype(np.float32)
    snowh = np.where(rng.random(shape) < .02, u(0.5, 4.), snowh).astype(np.float32)
    f["snowh"] = np.maximum(snowh, np.float32(1e-6))
    for name, lo, hi in (("znt", 1e-4, .05), ("lh", -50, 50), ("hfx", -50, 50),
                         ("qfx", -1e-4, 1e-4), ("potevp", 0, 1), ("grdflx", -50, 50),
                         ("qsfc", 0, .01), ("acsnow", 0, 10), ("acsnom", 0, 10),
                         ("snopcx", -10, 0), ("sfcrunoff", 0, 10), ("noahres", -5, 5)):
        f[name] = u(lo, hi)
    f["tslb"] = rng.uniform(245., 275., (4, ny, nx)).astype(np.float32)
    f["tslb"][0] = np.where(rng.random(shape) < .4, rng.uniform(272.5, 273.15, shape),
                            f["tslb"][0]).astype(np.float32)
    return f


def main():
    from woof.core import kernels
    from woof.core.noah import (SEAICE_F2D, SEAICE_SNOWDEPTH_MAX_M,
                                SEAICE_SNOWDEPTH_MIN_M, SEAICE_THICKNESS_DEFAULT_M)
    from woof.metal.array import MetalArrayModule
    from woof.metal.noah_family import submit_noah_seaice
    from woof.metal.runtime import MetalContext
    import host_backend

    host = host_backend.HostModule(
        "noah_seaice_probe", SHIM_PREFIX + kernels.module_source("noah"))
    kernel = host.get_function("noah_seaice_column")
    ny, nx, total_active, bad_words = 160, 160, 0, 0
    with MetalContext() as context:
        xp = MetalArrayModule(context)
        for seed in range(6):
            frpcpn = bool(seed & 1)
            dt = np.float32((20., 60., 12.)[seed % 3])
            albedo_default = np.float32((.65, .72)[seed % 2])
            ref = columns(np.random.default_rng(seed), ny, nx)
            dev = {k: xp.asarray(v.copy()) for k, v in ref.items()}
            fatal = np.zeros(1, np.int32)
            args = [ref[n] for n in SEAICE_F2D] + [
                ref["tslb"], fatal, dt, np.float32(.5), np.int32(frpcpn),
                albedo_default, np.float32(SEAICE_THICKNESS_DEFAULT_M),
                np.float32(SEAICE_SNOWDEPTH_MIN_M), np.float32(SEAICE_SNOWDEPTH_MAX_M),
                np.int32(ny), np.int32(nx)]
            kernel(((ny * nx + 63) // 64,), (64,), args)
            submit_noah_seaice(dev, dt, frpcpn=frpcpn, xice_threshold=.5,
                               albedo_default=albedo_default,
                               check_snow_density=False)
            mismatched = 0
            for name in (*SEAICE_F2D, "tslb"):
                a = ref[name].view(np.uint32)
                b = dev[name].to_numpy().view(np.uint32)
                mismatched += int(np.count_nonzero(a != b))
            total_active += int(np.count_nonzero(ref["xice"] >= .5))
            bad_words += mismatched
            print(f"seed {seed} frpcpn={int(frpcpn)} dt={float(dt):g}: "
                  f"{mismatched} mismatched words, host fatal={int(fatal[0])}")
        # Fatal readback: one inconsistent column raises, a consistent one does not.
        one = {k: np.full((1, 1), 0.5 if k == "xice" else 1., np.float32)
               for k in SEAICE_F2D}
        one["tslb"] = np.full((4, 1, 1), 270., np.float32)
        one["snow"][...] = 500.
        one["snowh"][...] = .1
        one["sfctmp"][...] = 270.
        for snowh, raises in ((.1, True), (2., False)):
            one["snowh"][...] = snowh
            dev = {k: xp.asarray(v.copy()) for k, v in one.items()}
            try:
                submit_noah_seaice(dev, 20., frpcpn=False, xice_threshold=.5,
                                   albedo_default=.65, check_snow_density=True)
                got = False
            except ValueError:
                got = True
            print(f"snowh={snowh}: raised={got} expected={raises}")
            bad_words += int(got != raises)
    print(f"active columns {total_active}; total mismatched words {bad_words}")
    return int(bad_words != 0)


if __name__ == "__main__":
    raise SystemExit(main())
