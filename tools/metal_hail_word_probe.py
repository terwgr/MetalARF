#!/usr/bin/env python3
"""Compare Metal hail diagnostics words with the CUDA source built as host C++.

hail.cu, hailcast_support.cuh and the two HAILCAST translations are compiled through
tools/thompson_real_column_parity/cuda_host_shim.h with -ffp-contract=off and
run serially in woof/core/hail.py's launch order. The Metal side runs
hail.update() itself on a minimal Metal state. The host build uses the host
libm for powf/expf/sinf/pow/exp/log, so a mismatch is a transcription fault
or a libm difference; the probe reports words, never a tolerance.

Run device work through the project's GPU queue.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KERNELS = ROOT / "woof/core/kernels"
FIELDS = ("hail_temperature", "hail_height", "hail_density", "hail_new_mask",
          "hail_new_duration", "hailcast_wup_mask", "hailcast_wdur",
          "hail_maxk1", "hail_max2d", "hail_maxk1_run", "hail_max2d_run",
          "hailcast_dhail1", "hailcast_dhail2", "hailcast_dhail3",
          "hailcast_dhail4", "hailcast_dhail5", "hailcast_diam_max",
          "hailcast_diam_mean", "hailcast_diam_std", "hailcast_runmax")
LAUNCHERS = r"""
#define L(c) blockIdx.x = c; blockDim.x = 1; threadIdx.x = 0;
extern "C" {
void ref_prepare(const float*a,const float*b,const float*c,const float*d,const float*e,
    const float*f,const float*g,float*h,float*i,float*j,float*k,int nx,int ny,int nz,int bp) {
  for(int c0=0;c0<nx*ny;c0++){L(c0) hail_prepare(a,b,c,d,e,f,g,h,i,j,k,nx,ny,nz,bp);} }
void ref_duration(const float*a,const float*b,const float*c,float*d,int nx,int ny,float dt,int px,int py) {
  for(int c0=0;c0<nx*ny;c0++){L(c0) hail_duration(a,b,c,d,nx,ny,dt,px,py);} }
void ref_psd(const float*a,const float*b,const float*c,const float*d,const float*e,const float*f,
    const float*g,float*h,float*i,float*j,float*k,int nx,int ny,int nz,int ng,int vg,int nh,int vh,
    int th,float dg,float dh,float ag,float ah,float ic,float ich) {
  for(int c0=0;c0<nx*ny;c0++){L(c0) hail_psd(a,b,c,d,e,f,g,h,i,j,k,nx,ny,nz,ng,vg,nh,vh,th,dg,dh,ag,ah,ic,ich);} }
void ref_columns(float*t,float*z,float*p,float*rho,float*qv,float*qi,float*qc,float*qr,float*qs,
    float*qg,float*w,const float*ht,const float*dur,float*e1,float*e2,float*e3,float*e4,float*e5,
    float*mx,float*mean,float*sd,float*runmax,int nx,int ny,int nz) {
  for(int c0=0;c0<nx*ny;c0++){L(c0) hailcast_columns(t,z,p,rho,qv,qi,qc,qr,qs,qg,w,ht,dur,
      e1,e2,e3,e4,e5,mx,mean,sd,runmax,nx,ny,nz);} }
}
"""


def build_reference(directory, version):
    shim = ROOT / "tools/thompson_real_column_parity/cuda_host_shim.h"
    cpp = directory / "hail_reference.cpp"
    cpp.write_text(f'#include "{shim}"\n'
                   + "".join((KERNELS / name).read_text() + "\n" for name in
                             ("hailcast_support.cuh", "hailcast_wrf.cuh", "hailcast_wrf39.cuh", "hail.cu"))
                   + LAUNCHERS.replace("hailcast_columns(t,z,p",
                       ("hailcast_columns_wrf39" if version == "wrf_39_noaa" else "hailcast_columns") + "(t,z,p"))
    path = directory / "hail_reference.dylib"
    subprocess.run(["clang++", "-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math",
                    "-w", "-shared", "-fPIC", str(cpp), "-o", str(path)], check=True)
    return ctypes.CDLL(str(path))


def case(name, mp, nz, nx, ny, *, species, periodic, base_profile, seed):
    """A convective column set: strong, marginal, weak and clear columns."""
    rng = np.random.default_rng(seed)
    z_face = (16000. * (np.linspace(0, 1, nz + 1) ** 1.3)).astype(np.float64)
    zc = .5 * (z_face[1:] + z_face[:-1])
    temp = 300. - 6.5e-3 * zc
    temp = np.maximum(temp, 215.)
    p = (100000. * (1. - 2.25577e-5 * zc) ** 5.25588).astype(np.float32)
    shape = (nz, ny, nx)
    ht = rng.uniform(0., 500., (ny, nx)).astype(np.float32)
    kind = rng.integers(0, 4, (ny, nx))          # 0 clear 1 weak 2 strong 3 very strong
    wmax = np.array([0., 12., 28., 45.])[kind] * rng.uniform(.85, 1.1, (ny, nx))
    shape_w = np.sin(np.pi * np.clip(z_face / 13000., 0, 1)) ** 1.2
    w = (shape_w[:, None, None] * wmax[None] + rng.normal(0, .3, (nz + 1, ny, nx))).astype(np.float32)
    qv_profile = .016 * np.exp(-zc / 2500.)
    qv = (qv_profile[:, None, None] * rng.uniform(.9, 1.05, shape)).astype(np.float32)
    theta = (temp * (100000. / p) ** (287. / 1004.)).astype(np.float32)
    thp = (rng.normal(0, .5, shape)).astype(np.float32)
    thb = (theta).astype(np.float32) if base_profile else np.broadcast_to(
        theta[:, None, None], shape).astype(np.float32).copy()
    alt = (287. * temp * (1 + .61 * qv_profile) / p).astype(np.float32)
    alt = np.broadcast_to(alt[:, None, None], shape).astype(np.float32) * rng.uniform(.99, 1.01, shape).astype(np.float32)
    phb = (9.81 * z_face).astype(np.float32)
    php = rng.normal(0, 20., (nz + 1, ny, nx)).astype(np.float32)
    if base_profile:
        phb_field = phb
    else:
        phb_field = (np.broadcast_to(phb[:, None, None], (nz + 1, ny, nx))
                     + 9.81 * ht[None]).astype(np.float32)
    cloud = (kind > 0)[None] & (zc[:, None, None] > 800.) & (zc[:, None, None] < 12500.)
    def hydro(low, high, lo_z, hi_z, present):
        field = rng.uniform(low, high, shape) * (rng.random(shape) < .85)
        window = (zc[:, None, None] > lo_z) & (zc[:, None, None] < hi_z)
        return (field * cloud * window * present).astype(np.float32)
    on = 1.0
    fields = {
        "qc": hydro(1e-5, 2.5e-3, 800., 9500., on), "qi": hydro(1e-6, 8e-4, 5500., 12500., on),
        "qr": hydro(1e-6, 1.5e-3, 800., 5000., on), "qs": hydro(1e-6, 1.2e-3, 4000., 12500., on),
        "qg": hydro(1e-6, 3e-3, 1500., 9000., species["graupel"]),
        "qh": hydro(1e-6, 2e-3, 1500., 8000., species["hail"]),
    }
    for name in ("qg", "qh"):
        mass = fields[name]
        fields["qn" + name[1]] = (mass * rng.uniform(2e3, 4e5, shape)).astype(np.float32)
        fields["qvol" + name[1]] = (mass / rng.uniform(380., 920., shape)).astype(np.float32)
    # Rare mass just under the PSD threshold and one exact zero column.
    fields["qg"][0, :, :] = np.where(rng.random((ny, nx)) < .3, 5e-7, fields["qg"][0])
    wdur = (rng.uniform(0., 1500., (ny, nx)) * (kind > 1)).astype(np.float32)
    wup = (kind > 1).astype(np.float32)
    # Seed some old windows just short of the 900 s threshold.
    wdur[(kind == 3)] = rng.uniform(870., 1100., int((kind == 3).sum())).astype(np.float32)
    return {"name": name, "mp": mp, "nz": nz, "nx": nx, "ny": ny, "periodic": periodic,
            "base_profile": base_profile, "ht": ht, "w": w, "p": np.broadcast_to(
                p[:, None, None], shape).astype(np.float32).copy(),
            "qv": qv, "thb": thb, "thp": thp, "alt": alt, "phb": phb_field, "php": php,
            "fields": fields, "wdur": wdur, "wup": wup, "species": species}


def make_cfg(c, version):
    row = SimpleNamespace(
        mp_physics=c["mp"], gsd_diagnostics=0, hailcast_opt=1, hailcast_version=version, dx=3000., dy=3000.,
        dt=20., haildt=0, open_x=not c["periodic"], open_y=not c["periodic"],
        specified=False, nested=False, morr_rimed_ice=0, wsm6_hail_opt=0, wdm6_hail_opt=0,
        nz=c["nz"], ny=c["ny"], nx=c["nx"])
    return row


class State:
    """The slice of DomainState hail.update() reads, on Metal."""
    _metal_setup_state = True

    def __init__(self, xp, c, names):
        self._array_module = xp
        self._scratch = {}
        self.elapsed_seconds = 0.
        for key in ("p", "alt", "phb", "php", "w", "ht", "thb", "thp", "qv"):
            setattr(self, key, xp.asarray(np.ascontiguousarray(c[key])))
        for key, value in c["fields"].items():
            if key in names:
                setattr(self, key, xp.asarray(value))

    def scratch(self, shape, slot, dtype=None):
        shape = tuple(shape)
        if slot not in self._scratch:
            self._scratch[slot] = self._array_module.zeros(shape, dtype=dtype or np.float32)
        return self._scratch[slot]

    def existing_scratch(self, slot):
        return self._scratch.get(slot)


def species_names(mp, c):
    from woof.core import hail
    row = hail.PSD_ROWS[mp]
    names = {"qv", "qi", "qc", "qr", "qs", "qg"}
    if row.get("density_h") is not None or c["species"]["hail"]:
        names.add("qh")
    if mp == 18:
        names |= {"qng", "qnh"} if c["species"]["number"] else set()
        names |= {"qvolg", "qvolh"} if c["species"]["volume"] else set()
    return names


def run_reference(lib, c, cfg, row):
    nz, ny, nx = c["p"].shape
    f = c["fields"]
    zero = np.zeros((nz, ny, nx), np.float32)
    out = {name: np.zeros((nz, ny, nx), np.float32) for name in
           ("hail_temperature", "hail_height", "hail_density")}
    plane = lambda: np.zeros((ny, nx), np.float32)
    for name in FIELDS[3:]:
        out[name] = plane()
    out["hailcast_wup_mask"] = c["wup"].copy()
    out["hailcast_wdur"] = c["wdur"].copy()
    names = species_names(c["mp"], c)
    sp = lambda key: f[key] if key in names else zero
    theta = (np.broadcast_to(c["thb"][:, None, None], c["p"].shape) if c["thb"].ndim == 1
             else c["thb"]) + c["thp"]
    ptr = lambda a: a.ctypes.data
    arrays = {k: np.ascontiguousarray(c[k], np.float32) for k in ("p", "alt", "qv", "phb", "php", "w", "ht")}
    theta = np.ascontiguousarray(theta, np.float32)
    lib.ref_prepare.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int] * 4
    lib.ref_prepare(ptr(theta), ptr(arrays["p"]), ptr(arrays["alt"]), ptr(arrays["qv"]),
                    ptr(arrays["phb"]), ptr(arrays["php"]), ptr(arrays["w"]),
                    ptr(out["hail_temperature"]), ptr(out["hail_height"]),
                    ptr(out["hail_density"]), ptr(out["hail_new_mask"]),
                    nx, ny, nz, int(c["phb"].ndim == 1))
    lib.ref_duration.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2 + [ctypes.c_float] + [ctypes.c_int] * 2
    lib.ref_duration(ptr(out["hail_new_mask"]), ptr(out["hailcast_wup_mask"]),
                     ptr(out["hailcast_wdur"]), ptr(out["hail_new_duration"]), nx, ny, 20.,
                     int(not (cfg.open_x or cfg.specified or cfg.nested)),
                     int(not (cfg.open_y or cfg.specified or cfg.nested)))
    out["hailcast_wup_mask"][...] = out["hail_new_mask"]
    out["hailcast_wdur"][...] = out["hail_new_duration"]
    if row is not None:
        ng = row.get("number_g", "ng"); nh = row.get("number_h", "nh")
        lib.ref_psd.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int] * 8 + [ctypes.c_float] * 6
        lib.ref_psd(ptr(out["hail_density"]), ptr(sp("qg")), ptr(sp(ng)), ptr(sp("qvolg")),
                    ptr(sp("qh")), ptr(sp(nh)), ptr(sp("qvolh")), ptr(out["hail_maxk1"]),
                    ptr(out["hail_max2d"]), ptr(out["hail_maxk1_run"]), ptr(out["hail_max2d_run"]),
                    nx, ny, nz, int(ng in names), int("qvolg" in names), int(nh in names),
                    int("qvolh" in names), int(row.get("thompson", False)),
                    *(np.float32(row.get(key, default)).item() for key, default in (
                        ("density_g", 500.), ("density_h", 900.), ("alpha_g", 0.),
                        ("alpha_h", 0.), ("intercept", 0.), ("intercept_h", 0.))))
    lib.ref_columns.argtypes = [ctypes.c_void_p] * 22 + [ctypes.c_int] * 3
    lib.ref_columns(ptr(out["hail_temperature"]), ptr(out["hail_height"]), ptr(arrays["p"]),
                    ptr(out["hail_density"]), ptr(arrays["qv"]), ptr(sp("qi")), ptr(sp("qc")),
                    ptr(sp("qr")), ptr(sp("qs")), ptr(sp("qg")), ptr(arrays["w"]), ptr(arrays["ht"]),
                    ptr(out["hail_new_duration"]),
                    *(ptr(out[f"hailcast_dhail{i}"]) for i in range(1, 6)),
                    ptr(out["hailcast_diam_max"]), ptr(out["hailcast_diam_mean"]),
                    ptr(out["hailcast_diam_std"]), ptr(out["hailcast_runmax"]), nx, ny, nz)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path(tempfile.gettempdir()) / "woof-hail-word-probe-result.json")
    parser.add_argument("--cases", default="", help="comma list of case names")
    parser.add_argument("--version", choices=("wrf_39_noaa", "wrf_461"), default="wrf_39_noaa")
    args = parser.parse_args()
    from woof.core import hail
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    both = {"graupel": 1., "hail": 1., "number": True, "volume": True}
    none = {"graupel": 0., "hail": 0., "number": False, "volume": False}
    cases = [
        ("mp8_nz40", 8, 40, 7, 5, both, False, True, 11),
        ("mp8_nz256", 8, 256, 6, 4, both, False, False, 12),
        ("mp28_graupel_absent", 28, 48, 7, 5, none, False, False, 13),
        ("mp28_periodic", 28, 57, 6, 6, both, True, True, 14),
        ("mp10_rimed0", 10, 64, 7, 5, both, False, False, 15),
        ("mp10_rimed1", 10, 64, 7, 5, both, False, True, 16),
        ("mp18_number_volume", 18, 72, 7, 5, both, False, False, 17),
        ("mp18_number_only", 18, 72, 7, 5, {**both, "volume": False}, False, True, 18),
        ("mp18_bulk_only", 18, 72, 7, 5, {**both, "number": False, "volume": False}, False, False, 19),
        ("mp24_hail_intercept", 24, 60, 7, 5, both, False, False, 20),
        ("mp6_graupel", 6, 52, 7, 5, both, False, True, 21),
    ]
    wanted = set(filter(None, args.cases.split(",")))
    records = []
    with tempfile.TemporaryDirectory(prefix="woof-hail-word-") as temp:
        lib = build_reference(Path(temp), args.version)
        with MetalContext() as context:
            xp = MetalArrayModule(context)
            for name, mp, nz, nx, ny, species, periodic, base_profile, seed in cases:
                if wanted and name not in wanted:
                    continue
                c = case(name, mp, nz, nx, ny, species=species, periodic=periodic,
                         base_profile=base_profile, seed=seed)
                cfg = make_cfg(c, args.version)
                if mp == 10:
                    cfg.morr_rimed_ice = 1 if name.endswith("rimed1") else 0
                row = hail.psd_parameters(cfg)
                expected = run_reference(lib, c, cfg, row)
                names = species_names(mp, c)
                state = State(xp, c, names)
                hail.initialize(state, cfg)
                state.scratch((ny, nx), "hailcast_wup_mask")[...] = c["wup"]
                state.scratch((ny, nx), "hailcast_wdur")[...] = c["wdur"]
                hail.update(state, cfg)
                active = int((expected["hail_new_duration"] > 900).sum())
                for field in FIELDS:
                    if row is None and field.startswith(("hail_max",)):
                        continue
                    buf = state.existing_scratch(field)
                    if buf is None:
                        continue
                    actual = buf.to_numpy().view(np.uint32).reshape(-1)
                    wanted_words = expected[field].view(np.uint32).reshape(-1)
                    bad = np.flatnonzero(actual != wanted_words)
                    record = {"case": name, "field": field, "words": int(actual.size),
                              "mismatches": int(bad.size), "active_columns": active,
                              "positive_finite_reference": int(np.count_nonzero(
                                  np.isfinite(expected[field]) & (expected[field] > 0)))}
                    if bad.size:
                        i = int(bad[0])
                        a = np.array([actual[i]], np.uint32).view(np.float32)[0]
                        b = np.array([wanted_words[i]], np.uint32).view(np.float32)[0]
                        record["first_difference"] = {"index": i, "metal": float(a),
                                                      "reference": float(b),
                                                      "metal_word": int(actual[i]),
                                                      "reference_word": int(wanted_words[i])}
                    records.append(record)
                print(name, "active", active, "mismatches",
                      sum(r["mismatches"] for r in records if r["case"] == name), flush=True)
    if not any(r["field"] == "hailcast_diam_max" and r["positive_finite_reference"] for r in records):
        raise RuntimeError("probe did not exercise a nonzero hailstone result")
    receipt = {"version": args.version, "status": "FAIL" if any(r["mismatches"] for r in records) else "PASS",
               "comparisons": len(records), "output_words": sum(r["words"] for r in records),
               "mismatches": sum(r["mismatches"] for r in records), "cases": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "cases"}, indent=2))
    for record in records:
        if record["mismatches"]:
            print(json.dumps(record))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
