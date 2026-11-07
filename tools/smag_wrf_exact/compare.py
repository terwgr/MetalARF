"""Run the production regional km_opt=4 coefficient launcher word for word.

The independent reference compiles the retained strict CUDA source closure
as IEEE C++ with contraction disabled. Unmodified compiled-WRF coefficient
and deformation words are also compared for the selected native fixtures.
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
from tools.smag_wrf_exact.generate import KERNELS, retained_source

FIELDS = ("d11", "d22", "d12", "kmh", "khh")


def compare_words(actual, reference):
    a, b = (np.ascontiguousarray(v, dtype=np.float32) for v in (actual, reference))
    if a.shape != b.shape:
        raise ValueError(f"comparison shape mismatch: {a.shape} != {b.shape}")
    unequal = a.view(np.uint32) != b.view(np.uint32)
    magnitude = b.view(np.uint32) & 0x7fffffff
    first = np.argwhere(unequal)
    return {"words": int(a.size), "different_words": int(unequal.sum()),
            "subnormal_words": int(np.count_nonzero(
                (magnitude > 0) & (magnitude < 0x00800000))),
            "first": first[:3].tolist()}


def cases(directory, selected):
    manifest = json.loads((directory / "deformation-manifest.json").read_text())
    pins = {row["name"]: row for row in manifest["cases"]}
    for name in selected:
        pin = pins[name]
        path = directory / pin["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]:
            raise ValueError("compiled-WRF fixture changed: " + pin["file"])
        with np.load(path) as data:
            arrays = {key.removeprefix("input__"): data[key].copy()
                      for key in data.files if key.startswith("input__")}
            meta = json.loads(str(data["meta_json"]))
            refs = {field: data["ref__km4_iso0__" + field].copy() for field in FIELDS}
        refs["d12"] = np.ascontiguousarray(refs["d12"][:, :meta["ny"], :meta["nx"]])
        yield name, arrays, meta, refs
    yield from arithmetic_cases()


def arithmetic_cases():
    nz, ny, nx = 5, 4, 5
    u = np.zeros((nz, ny, nx + 1), np.float32)
    v = np.zeros((nz, ny + 1, nx), np.float32)
    base = dict(u=u, v=v, w=np.zeros((nz + 1, ny, nx), np.float32),
                php=np.zeros((nz + 1, ny, nx), np.float32),
                phb=np.arange(nz + 1, dtype=np.float32) * np.float32(981.),
                alt=np.ones((nz, ny, nx), np.float32),
                qv=np.zeros((nz, ny, nx), np.float32),
                msft=np.ones((ny, nx), np.float32),
                msfu=np.ones((ny, nx + 1), np.float32),
                msfv=np.ones((ny + 1, nx), np.float32),
                fnm=np.full(nz, .5, np.float32), fnp=np.full(nz, .5, np.float32),
                dn=np.full(nz, -.2, np.float32), dnw=np.full(nz, -.2, np.float32))
    meta = dict(nz=nz, ny=ny, nx=nx, bx=1, by=1, dx=1., dy=1., c_s=.25,
                cf1=1.8286585807800293, cf2=-1.135213851928711,
                cf3=.30655530095100403)
    for name, scale in (("subnormal_deformation", np.float32(2.0 ** -128)),
                        ("subnormal_strain_square", np.float32(2.0 ** -65))):
        arrays = {key: value.copy() for key, value in base.items()}
        arrays["u"][:] = np.arange(nx + 1, dtype=np.float32)[None, None, :] * scale
        arrays["v"][:] = -np.arange(ny + 1, dtype=np.float32)[None, :, None] * scale
        yield name, arrays, meta.copy(), None
    arrays = {key: value.copy() for key, value in base.items()}
    arrays["u"][:] = np.arange(ny, dtype=np.float32)[None, :, None] * np.float32(.125)
    arrays["v"][:] = -np.arange(nx, dtype=np.float32)[None, None, :] * np.float32(.125)
    yield "rotation_cancellation", arrays, {**meta, "dx": 3., "dy": 3.}, None


def state(arrays, meta, xp, metal):
    values = {key: xp.asarray(value) for key, value in arrays.items()}
    values.update({key + "0": values[key] for key in ("u", "v", "w", "php", "qv")})
    values.update({key: meta[key] for key in ("cf1", "cf2", "cf3")})
    cache = {}
    def scratch(shape, slot):
        key = (tuple(shape), slot)
        if key not in cache:
            cache[key] = xp.zeros(shape, dtype=np.float32)
        return cache[key]
    values.update(scratch=scratch, _metal_setup_state=metal)
    return SimpleNamespace(**values)


def launch(arrays, meta, xp, metal, *, time_t):
    from woof.core.dycore import launch_wrf_smag2d_km
    s = state(arrays, meta, xp, metal)
    cfg = SimpleNamespace(dx=meta["dx"], dy=meta["dy"], c_s=meta["c_s"],
                          diff_opt=2, open_x=bool(meta["bx"]), open_y=bool(meta["by"]),
                          specified=False, nested=False)
    outputs = [xp.zeros(arrays["alt"].shape, dtype=np.float32) for _ in range(2)]
    deform = launch_wrf_smag2d_km(s, cfg, *outputs, time_t=time_t)
    host = (lambda a: a.to_numpy()) if metal else np.asarray
    return dict(zip(FIELDS, (host(a).copy() for a in (*deform, *outputs))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path, default=ROOT / "tests/data/wrf471_diffusion")
    ap.add_argument("--cases", default="real_open,real_periodic,steep_open,map_extremes,near_zero_flow,zero_flow")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from woof.wrf_exact import ENABLED, DIFFUSION_ENABLED
    if not (ENABLED and DIFFUSION_ENABLED):
        raise ValueError("select GPUWM_WRF_EXACT=1 with the diffusion control enabled before import")
    from tools.thompson_real_column_parity import host_backend
    host_backend.install()
    source = "\n".join(retained_source())
    module = host_backend.HostModule("wrf_exact_smag_coefficients", source)
    from woof.core import dycore
    original = dycore.get_kernel
    dycore.get_kernel = lambda unit, name: (module.get_function(name) if unit == "smag2d" and name in KERNELS
                                            else original(unit, name))
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    report = {"source_sha256": hashlib.sha256(source.encode()).hexdigest(), "cases": {}}
    with MetalContext() as context:
        xp = MetalArrayModule(context)
        for index, (name, arrays, meta, refs) in enumerate(cases(args.fixtures, args.cases.split(","))):
            time_t = bool(index % 2)
            expected = launch(arrays, meta, np, False, time_t=time_t)
            actual = launch(arrays, meta, xp, True, time_t=time_t)
            row = {"source": {field: compare_words(actual[field], expected[field]) for field in FIELDS},
                   "time_t": time_t, "phb3d": arrays["phb"].ndim == 3}
            if refs is not None:
                row["wrf"] = {field: compare_words(actual[field], refs[field]) for field in FIELDS}
            report["cases"][name] = row
            print(name, json.dumps(row), flush=True)
    checks = [metric for case in report["cases"].values() for group in ("source", "wrf")
              for metric in case.get(group, {}).values()]
    report["words"] = sum(m["words"] for m in checks)
    report["different_words"] = sum(m["different_words"] for m in checks)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("words", "different_words")}))
    if report["different_words"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
