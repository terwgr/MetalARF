"""Run the production strict Smagorinsky tendency launches word for word.

The independent reference compiles the retained strict CUDA source closure
(`GPUWM_WRF_EXACT`, `GPUWM_WRF_EXACT_C_DIFFUSION`) as IEEE C++ with
contraction disabled and drives it through the same `launch_wrf_smag2d_hd`
and `_zero_open_strips` code. Unmodified compiled-WRF tendency words are also
compared for the retained horizontal fixtures. Every output tendency, the
scalar face fluxes and the boundary strips are compared as literal words.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.smag_wrf_exact.compare import compare_words
from tools.smag_wrf_exact.generate import KERNELS, TENDENCY_KERNELS, retained_source

G = np.float32(9.81)
STAGGERS = (("u", "x", "kmh"), ("v", "y", "kmh"), ("w", "z", "kmv"), ("s", "", "khh"))


def fixture_cases(directory, selected):
    manifest = json.loads((directory / "manifest.json").read_text())
    for row in manifest["cases"]:
        if selected and row["name"] not in selected:
            continue
        path = directory / row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][row["file"]]:
            raise ValueError("compiled-WRF fixture changed: " + row["file"])
        with np.load(path) as data:
            arrays = {key.removeprefix("input_"): data[key].copy()
                      for key in data.files if key.startswith("input_")}
            ops = {key.removeprefix("operator_"): data[key].copy()
                   for key in data.files if key.startswith("operator_")}
            refs = {key.removeprefix("wrf_"): data[key].copy()
                    for key in data.files if key.startswith("wrf_")}
        meta = {key: row[key] for key in ("nz", "ny", "nx", "bx", "by", "dx", "dy",
                                          "cf1", "cf2", "cf3")}
        yield row["name"], arrays, ops, meta, refs


def synthetic(name, rng, *, bx, by, phb3d, scale_km=1., scale_d=1., seeded=True,
              uniform=False):
    """A small terrain-following state with stretched levels and varying maps."""
    nz, ny, nx = 6, 5, 7
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
    uni = lambda lo, hi, *shape: f32(rng.uniform(lo, hi, shape))
    k = np.arange(nz + 1, dtype=np.float32)[:, None, None]
    terrain = uni(0., 300., 1, ny, nx)
    base = np.float32(9.81) * np.float32(500.) * k
    lift = np.float32(9.81) * terrain * (1 - k / np.float32(nz))
    if phb3d:
        phb = f32(base + lift)
        php = uni(-60., 60., nz + 1, ny, nx)
    else:
        phb = f32(base[:, 0, 0])
        php = f32(lift + uni(-60., 60., nz + 1, ny, nx))
    fnm = uni(.4, .6, nz)
    arrays = dict(
        u=uni(-10., 10., nz, ny, nx + 1), v=uni(-10., 10., nz, ny + 1, nx),
        w=uni(-1., 1., nz + 1, ny, nx), php=php, phb=phb,
        alt=uni(.8, 1.2, nz, ny, nx), qv=uni(0., .02, nz, ny, nx),
        msft=uni(.7, 1.3, ny, nx), msfu=uni(.7, 1.3, ny, nx + 1),
        msfv=uni(.7, 1.3, ny + 1, nx), fnm=fnm, fnp=f32(1 - fnm),
        dn=uni(-.03, -.01, nz), dnw=uni(-.03, -.01, nz),
        thb=f32(uni(290., 320., nz, ny, nx) if phb3d else uni(290., 320., nz)),
        thp=uni(-5., 5., nz, ny, nx))
    ops = dict(kmh=uni(0., 50., nz, ny, nx), kmv=uni(0., 50., nz, ny, nx),
               khh=uni(0., 150., nz, ny, nx), d11=uni(-1e-3, 1e-3, nz, ny, nx),
               d22=uni(-1e-3, 1e-3, nz, ny, nx), d12=uni(-1e-3, 1e-3, nz, ny, nx))
    if uniform:
        # Identical neighbours: stress and flux differences cancel to signed zeros.
        for key in ("u", "v", "w", "alt", "qv", "thp"):
            arrays[key][...] = arrays[key].flat[0]
        for key in ("kmh", "kmv", "khh", "d11", "d22", "d12"):
            ops[key][...] = ops[key].flat[0]
        for key in ("msft", "msfu", "msfv"):
            arrays[key][...] = 1
        arrays["php"][...] = 0
        arrays["phb"] = f32(base[:, 0, 0])
        arrays["thb"] = f32(uni(290., 320., nz))
    for key in ("kmh", "kmv", "khh"):
        ops[key] = f32(ops[key] * np.float32(scale_km))
    for key in ("d11", "d22", "d12"):
        ops[key] = f32(ops[key] * np.float32(scale_d))
    meta = dict(nz=nz, ny=ny, nx=nx, bx=bx, by=by, dx=3000., dy=3000.,
                cf1=1.8286585807800293, cf2=-1.135213851928711,
                cf3=.30655530095100403, seeded=seeded)
    return name, arrays, ops, meta, None


def synthetic_cases():
    rng = np.random.default_rng(20261024)
    for bx, by in ((1, 1), (1, 0), (0, 1), (0, 0)):
        for phb3d in (0, 1):
            yield synthetic(f"terrain_maps_b{bx}{by}_phb{'3d' if phb3d else '1d'}",
                            rng, bx=bx, by=by, phb3d=phb3d)
    yield synthetic("subnormal_products", rng, bx=1, by=1, phb3d=1,
                    scale_km=2. ** -100, scale_d=2. ** -40)
    yield synthetic("subnormal_underflow", rng, bx=0, by=0, phb3d=0,
                    scale_km=2. ** -124, scale_d=2. ** -20)
    yield synthetic("cancellation_open", rng, bx=1, by=1, phb3d=0, uniform=True)
    yield synthetic("cancellation_periodic", rng, bx=0, by=0, phb3d=0, uniform=True)


def build_state(arrays, meta, xp, metal, time_t):
    """Both time levels are distinct so the time-t selector is observable."""
    values = {key: xp.asarray(value) for key, value in arrays.items()}
    for key, factor in (("u", 1.0625), ("v", -.9375), ("w", -1.), ("php", 1.03125),
                        ("qv", .75)):
        values[key + "0"] = xp.asarray(np.ascontiguousarray(
            arrays[key] * np.float32(factor), dtype=np.float32))
    values.update({key: np.float32(meta[key]) for key in ("cf1", "cf2", "cf3")})
    cache = {}

    def scratch(shape, slot):
        key = (tuple(shape), slot)
        if key not in cache:
            cache[key] = xp.zeros(shape, dtype=np.float32)
        return cache[key]
    values.update(scratch=scratch, _scratch_cache=cache, _metal_setup_state=metal)
    return SimpleNamespace(**values)


def seed_tendency(shape, seeded):
    """Production starts from literal zero; the seeded runs also add onto data."""
    if not seeded:
        return np.zeros(shape, np.float32)
    rng = np.random.default_rng(shape[0] * 1000003 + shape[1] * 1009 + shape[2])
    out = rng.uniform(-1e-2, 1e-2, shape).astype(np.float32)
    out.flat[0] = np.float32(-0.)
    out.flat[1] = np.float32(2. ** -140)
    return out


def run(arrays, ops, meta, xp, metal, *, time_t, seeded):
    from woof.core.dycore import launch_wrf_smag2d_hd, _zero_open_strips
    state = build_state(arrays, meta, xp, metal, time_t)
    cfg = SimpleNamespace(dx=meta["dx"], dy=meta["dy"], nx=meta["nx"], ny=meta["ny"],
                          open_x=bool(meta["bx"]), open_y=bool(meta["by"]),
                          specified=False, nested=False)
    kms = {key: xp.asarray(value) for key, value in ops.items()}
    deformation = tuple(kms[key] for key in ("d11", "d22", "d12"))
    host = (lambda a: a.to_numpy()) if metal else np.asarray
    out = {}
    jobs = [(name, stag, key, arrays["thp"], False) for name, stag, key in STAGGERS]
    jobs.append(("s_theta", "", "khh", arrays["thp"], True))
    jobs.append(("s_vapor", "", "khh", arrays["qv"], False))
    for name, stag, key, field, full_theta in jobs:
        field = {"u": state.u, "v": state.v, "w": state.w}.get(name, xp.asarray(field))
        shape = {"u": state.u.shape, "v": state.v.shape, "w": state.w.shape}.get(
            name, kms["khh"].shape)
        tend = xp.asarray(seed_tendency(shape, seeded))
        launch_wrf_smag2d_hd(state, cfg, field, kms[key], tend, stagger=stag,
                             time_t=time_t, full_theta=full_theta,
                             deformation=deformation)
        _zero_open_strips(tend, cfg, 1)
        out[name] = host(tend).copy()
        if stag == "":
            for flux in ("diff6_x", "diff6_y"):
                (array,) = [v for (shape_, slot), v in state._scratch_cache.items()
                            if slot == flux]
                out[name + "_" + flux] = host(array).copy()
    return out


def strip_cases(xp, metal):
    """The width-1 production strips and the wider historical modes, word for word."""
    from woof.core.dycore import _zero_open_strips
    rng = np.random.default_rng(7)
    host = (lambda a: a.to_numpy()) if metal else np.asarray
    shapes = ((4, 5, 7), (4, 5, 8), (4, 6, 7), (5, 5, 7), (5, 7))
    results = {}
    for bx, by in ((1, 1), (1, 0), (0, 1)):
        cfg = SimpleNamespace(nx=7, ny=5, open_x=bool(bx), open_y=bool(by),
                              specified=False, nested=False)
        for shape in shapes:
            for width, extra in ((1, 0), (3, 0), (1, 1)):
                data = rng.uniform(-1, 1, shape).astype(np.float32)
                data.flat[0] = np.float32(-0.)
                data.flat[-1] = np.float32(2. ** -140)
                buf = xp.asarray(data)
                _zero_open_strips(buf, cfg, width, extra)
                results[f"b{bx}{by}_{'x'.join(map(str, shape))}_w{width}e{extra}"] = host(buf).copy()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path,
                    default=ROOT / "tests/data/wrf471_diffusion/horizontal")
    ap.add_argument("--cases", default="", help="comma list of fixture names; empty selects all")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from woof.wrf_exact import ENABLED, DIFFUSION_ENABLED
    if not (ENABLED and DIFFUSION_ENABLED):
        raise ValueError("select GPUWM_WRF_EXACT=1 with the diffusion control enabled before import")
    from tools.thompson_real_column_parity import host_backend
    host_backend.install()
    source = "\n".join(retained_source(KERNELS + TENDENCY_KERNELS))
    module = host_backend.HostModule("wrf_exact_smag_tendencies", source)
    from woof.core import dycore
    original = dycore.get_kernel
    names = KERNELS + TENDENCY_KERNELS
    dycore.get_kernel = lambda unit, name: (module.get_function(name) if unit == "smag2d" and name in names
                                            else original(unit, name))
    # Narrow addressing changes indices only; the strict source has one definition.
    dycore.get_kernel_int_defines = lambda unit, name, defines: dycore.get_kernel(unit, name)
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    selected = [name for name in args.cases.split(",") if name]
    report = {"source_sha256": hashlib.sha256(source.encode()).hexdigest(), "cases": {},
              "strips": {}}
    rows = []
    with MetalContext() as context:
        xp = MetalArrayModule(context)
        stream = list(fixture_cases(args.fixtures, selected)) if args.fixtures else []
        stream += list(synthetic_cases())
        for name, arrays, ops, meta, refs in stream:
            for time_t in (False, True):
                seeded = bool(meta.get("seeded", False))
                expected = run(arrays, ops, meta, np, False, time_t=time_t, seeded=seeded)
                actual = run(arrays, ops, meta, xp, True, time_t=time_t, seeded=seeded)
                row = {"time_t": time_t, "phb3d": arrays["phb"].ndim == 3,
                       "seeded": seeded,
                       "source": {key: compare_words(actual[key], expected[key])
                                  for key in expected}}
                if refs is not None and not time_t:
                    pairs = {"u": "thp_u", "v": "thp_v", "w": "thp_w", "s": "thp_s",
                             "s_vapor": "qv_s"}
                    row["wrf"] = {key: compare_words(actual[key], refs[ref])
                                  for key, ref in pairs.items()}
                report["cases"][f"{name}:t{int(time_t)}"] = row
                rows.append(row)
                print(name, int(time_t), json.dumps({
                    "source_words": sum(m["words"] for m in row["source"].values()),
                    "source_diff": sum(m["different_words"] for m in row["source"].values()),
                    "wrf_diff": {k: m["different_words"] for k, m in row.get("wrf", {}).items()}}),
                    flush=True)
        reference = strip_cases(np, False)
        measured = strip_cases(xp, True)
        report["strips"] = {key: compare_words(measured[key], reference[key]) for key in reference}
    checks = [m for row in rows for m in row["source"].values()]
    checks += list(report["strips"].values())
    report["source_words"] = sum(m["words"] for m in checks)
    report["source_different_words"] = sum(m["different_words"] for m in checks)
    wrf = [m for row in rows for m in row.get("wrf", {}).values()]
    report["wrf_words"] = sum(m["words"] for m in wrf)
    report["wrf_different_words"] = sum(m["different_words"] for m in wrf)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "source_words", "source_different_words", "wrf_words", "wrf_different_words")}))
    if report["source_different_words"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
