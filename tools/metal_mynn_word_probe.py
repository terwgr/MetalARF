#!/usr/bin/env python3
"""Compare Metal MYNN driver words with the CUDA driver built as host C++.

``mynn_bl_driver_cuda`` runs unmodified on the host through
tools/thompson_real_column_parity/host_backend.py (no multiply-add
contraction, the module's own pinned powf/expf/tanhf); the Metal side is
``mynn_bl_driver_metal`` itself. Both take the same specific-humidity
columns, the same carried state, and run a cold step followed by warm steps
that feed their own state back, so every output and every carried field is
compared as literal uint32 words. The probe reports words and mismatches,
never a tolerance.

Run through the project's GPU queue, one job per option group:

    --group length        mixing length 1/2 with both qtke conversions
    --group cloud         both cloud tendency forms
    --group baseline      the wrf_461 chain

Cases mix stable, convective and cloudy columns over several nz. Use
``--entry caller`` to include the public coupling layer and
``--reload-scratch`` to resume from carried fields in a fresh scratch arena.
This compares CUDA source executed on the host; it is not CUDA-device evidence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "thompson_real_column_parity"))

STATE = ("qke", "tsq", "qsq", "cov", "el", "sh", "sm",
         "qc_bl", "qi_bl", "cldfra_bl")
LAYER_OUT = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
             "rqiblten", "rqsblten", "dozone", "exch_h", "exch_m", *STATE)
COLUMN_OUT = ("pblh", "rmol", "maxwidth", "maxmf", "ztop_plume")
INT_OUT = ("kpbl", "ktop_plume")


def plain_arithmetic(text):
    """The unit's inline-PTX rounded operators as plain host operators.

    ``add.rn.f32`` and its siblings are the IEEE operation with
    round-to-nearest and no flush; the host build (no contraction, SSE
    arithmetic) performs exactly that, and ``setp.gt.f32`` is ``a > b``.
    """
    import re
    for op, symbol in (("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")):
        text = text.replace(
            f'asm("{op}.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));',
            f"r = a {symbol} b;")
    text = re.sub(
        r'asm\("\{ \.reg \.pred q; setp\.gt\.f32 q, %1, %2; selp\.u32 %0, 1, 0, q; \}"\s*'
        r': "=r"\(p\) : "f"\(a\), "f"\(b\)\);',
        "p = a > b ? 1u : 0u;", text)
    if "asm(" in text.replace("asm volatile and", ""):
        raise RuntimeError("the MYNN unit carries inline PTX this probe does not translate")
    return text


def make_case(seed, nz, ncol):
    """Stable, convective and cloudy columns, level-major (nz, ncol)."""
    rng = np.random.default_rng(seed)
    f32 = np.float32
    ps = rng.uniform(88000., 102000., ncol)
    dz0 = rng.uniform(12., 60., ncol)
    growth = rng.uniform(1.0, 1.12 if nz <= 40 else 1.04, ncol)
    dz = dz0[None] * growth[None] ** np.arange(nz)[:, None]
    zi = np.concatenate([np.zeros((1, ncol)), np.cumsum(dz, 0)])
    zc = 0.5 * (zi[1:] + zi[:-1])
    p_if = ps[None] * np.exp(-zi / 8000.)
    p = np.sqrt(p_if[:-1] * p_if[1:])
    exner = (p / 1.0e5) ** 0.2857142857
    regime = rng.integers(0, 3, ncol)          # 0 stable, 1 convective, 2 cloudy
    unstable = regime >= 1
    th_sfc = rng.uniform(278., 308., ncol)
    hmix = rng.uniform(250., 2000., ncol)
    inv = rng.uniform(0.3, 4.0, ncol)
    gam_above = rng.uniform(2e-3, 6e-3, ncol)
    gam_stable = rng.uniform(3e-3, 1.2e-2, ncol)
    th = np.where(unstable[None],
                  th_sfc[None] + np.where(zc < hmix[None], 1e-4 * zc,
                                          inv[None] + 1e-4 * hmix[None]
                                          + gam_above[None] * (zc - hmix[None])),
                  th_sfc[None] + gam_stable[None] * zc)
    th = th + rng.normal(0., 0.02, th.shape)
    tk = th * exner
    qs0 = rng.uniform(0.003, 0.016, ncol)
    qv = qs0[None] * np.exp(-zc / rng.uniform(1500., 3500., ncol)[None])
    qc = np.zeros_like(th)
    qi = np.zeros_like(th)
    qsn = np.zeros_like(th)
    for c in np.flatnonzero(regime == 2):
        top = int(np.searchsorted(zc[:, c], hmix[c]))
        top = min(max(top, 3), nz - 3)
        for k in range(max(top - rng.integers(1, 4), 1), min(top + 2, nz - 2)):
            qc[k, c] = rng.uniform(2e-5, 6e-4)
            qv[k, c] = max(qv[k, c], 0.012)
        if rng.random() < 0.4 and top + 3 < nz:
            qi[top + 2, c] = rng.uniform(1e-5, 1.5e-4)
            qsn[top + 3, c] = rng.uniform(0., 5e-5)
    tv = tk * (1. + 0.608 * qv)
    rho = p / (287.04 * tv)
    u = rng.uniform(-12., 12., ncol)[None] + rng.uniform(-3e-3, 3e-3, ncol)[None] * zc
    v = rng.uniform(-12., 12., ncol)[None] + rng.uniform(-3e-3, 3e-3, ncol)[None] * zc
    w = rng.normal(0., 0.04, th.shape)
    ust = np.exp(rng.uniform(np.log(0.05), np.log(0.9), ncol))
    hfx = np.where(unstable, rng.uniform(10., 380., ncol), -rng.uniform(1., 70., ncol))
    qfx = rng.uniform(0., 2.5e-4, ncol)
    wspd = np.sqrt(u[0] ** 2 + v[0] ** 2) + rng.uniform(0.2, 1.5, ncol)
    ts = th_sfc * (ps / 1.0e5) ** 0.2857142857 + np.where(unstable, 1.5, -1.0)
    xland = np.where(rng.random(ncol) < 0.3, 2., 1.)
    case = {name: np.ascontiguousarray(a, f32) for name, a in (
        ("dz", dz), ("u", u), ("v", v), ("w", w), ("th", th), ("p", p),
        ("exner", exner), ("rho", rho), ("tk", tk), ("qv", qv), ("qc", qc),
        ("qi", qi), ("qs", qsn), ("dx", np.full(ncol, 3000.)), ("xland", xland),
        ("ts", ts), ("ps", ps), ("ust", ust), ("hfx", hfx), ("qfx", qfx),
        ("wspd", wspd), ("pblh", np.full(ncol, 500.)),
        ("rmol", np.zeros(ncol)))}
    return case


def specific(case):
    """The wrapper's mixing-ratio to specific-humidity conversion."""
    f = np.float32
    denom = f(1.) + case["qv"]
    return {"sqv": case["qv"] / denom, "sqc": case["qc"] / denom,
            "sqi": case["qi"] / denom, "sqs": case["qs"] / denom}


def host_values(case, sq, state, carried, nz, ncol):
    """Driver inputs in the CUDA layout, (ncol, nz) Fortran order."""
    f32 = np.float32

    def pair(a):
        return np.asfortranarray(np.ascontiguousarray(a).T, dtype=f32)

    values = {n: pair(case[n]) for n in ("dz", "u", "v", "w", "th", "p",
                                         "exner", "rho", "tk", "qv", "qc", "qi")}
    values.update({n: pair(sq[n]) for n in sq})
    values.update({n: pair(state[n]) for n in STATE})
    for n in ("dx", "xland", "ts", "ps", "ust", "hfx", "qfx", "wspd"):
        values[n] = np.ascontiguousarray(case[n], f32)
    values["uoce"] = np.zeros(ncol, f32)
    values["voce"] = np.zeros(ncol, f32)
    values["pblh"] = np.ascontiguousarray(carried["pblh"], f32)
    values["rmol"] = np.ascontiguousarray(carried["rmol"], f32)
    values["kpbl"] = np.ascontiguousarray(carried["kpbl"], np.int32)
    return values


class Scratch:
    """DomainState.scratch for a probe: named persistent Metal buffers."""

    def __init__(self, xp):
        self.xp = xp
        self.slots = {}

    def scratch(self, shape, slot, dtype=None):
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        dtype = np.dtype(np.float32 if dtype is None else dtype)
        buf = self.slots.get(slot)
        if buf is None:
            buf = self.xp.zeros(shape, dtype=dtype)
            self.slots[slot] = buf
        return buf


def words(a):
    return np.ascontiguousarray(a).view(np.uint32).reshape(-1)


def step(run_host, run_metal, case, sq, nz, ncol, steps, initial,
         layer_out=LAYER_OUT):
    """Cold step then warm steps; compare every output; feed each side its own."""
    host_state = {n: np.zeros((nz, ncol), np.float32) for n in STATE}
    metal_state = {n: np.zeros((nz, ncol), np.float32) for n in STATE}
    host_carried = {"pblh": case["pblh"].copy(), "rmol": case["rmol"].copy(),
                    "kpbl": np.zeros(ncol, np.int32)}
    metal_carried = {"pblh": case["pblh"].copy(), "rmol": case["rmol"].copy()}
    if initial:
        for n in STATE:
            host_state[n][...] = metal_state[n][...] = initial[n]
    records = []
    for index in range(steps):
        hv = host_values(case, sq, host_state, host_carried, nz, ncol)
        expected = run_host(hv, index == 0)
        actual = run_metal(case, sq, metal_state, metal_carried, index == 0)
        total = bad = 0
        per = {}
        first = {}
        nonfinite = 0
        for name in (*layer_out, *COLUMN_OUT, *INT_OUT):
            e = expected[name]
            if name in LAYER_OUT:
                e = np.ascontiguousarray(np.asarray(e).T)
            e = np.ascontiguousarray(e)
            a = np.ascontiguousarray(actual[name])
            assert e.shape == a.shape and e.dtype == a.dtype, (name, e.shape, a.shape, e.dtype, a.dtype)
            m = int(np.count_nonzero(words(e) != words(a)))
            total += e.size
            bad += m
            if e.dtype.kind == "f":
                nonfinite += int(np.count_nonzero(~np.isfinite(e)))
            if m:
                per[name] = m
                where = np.flatnonzero(words(e) != words(a))[:2]
                first[name] = [[int(i), f"{int(words(e)[i]):08x}",
                                f"{int(words(a)[i]):08x}"] for i in where]
        records.append({"step": index + 1, "words": total, "mismatches": bad,
                        "by_field": per, "first": first,
                        "host_nonfinite": nonfinite})
        # each side carries its own result
        for n in STATE:
            host_state[n] = np.ascontiguousarray(np.asarray(expected[n]).T)
            metal_state[n] = np.ascontiguousarray(actual[n])
        host_carried = {"pblh": np.asarray(expected["pblh"]), "rmol": np.asarray(expected["rmol"]),
                        "kpbl": np.asarray(expected["kpbl"])}
        metal_carried = {"pblh": np.asarray(actual["pblh"]), "rmol": np.asarray(actual["rmol"])}
    return records


def install_backends():
    """The CUDA driver on the host, and both drivers."""
    build = os.environ.setdefault(
        "WOOF_HOST_KERNEL_BUILD_DIR",
        str(Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "woof" / "host-kernels"))
    Path(build).mkdir(parents=True, exist_ok=True)
    import host_backend
    host_backend.install()
    host_backend.set_transform("mynn_pbl", plain_arithmetic)
    import woof.core.mynn_pbl_gpu as gpu_module
    from woof.core.mynn_pbl_gpu import mynn_bl_driver_cuda, mynn_bl_driver_metal

    # The MYNN launchers pass a trailing shape-key argument that the host
    # loaders' drop-ins do not take; the key only names a device build.
    plain, defined = gpu_module.get_kernel, gpu_module.get_kernel_int_defines
    gpu_module.get_kernel = lambda name, func, shape=(): plain(name, func)
    gpu_module.get_kernel_int_defines = (
        lambda name, func, defines, shape=(): defined(name, func, defines))
    # The input-validation scans are device reductions with no host
    # spelling; they only refuse, never touch a value.
    predicates = {
        "nonfinite": lambda a: not np.isfinite(a).all(),
        "nonpositive": lambda a: bool((a <= 0).any()),
        "nonzero": lambda a: bool((a != 0).any()),
    }
    for name in predicates:
        setattr(gpu_module, f"_{name}", lambda name=name: name)
    gpu_module._flag_mask = lambda kernel, arrays, flags: [
        predicates[kernel](a) for a in arrays]
    gpu_module._tripped = lambda kernel, arrays, flags: any(
        predicates[kernel](a) for a in arrays)

    def first_refusal(checks, flags, site=None):
        for kernel, arrays, message in checks:
            if any(predicates[kernel](a) for a in arrays):
                return message
        return None

    gpu_module._first_refusal = first_refusal
    return gpu_module, mynn_bl_driver_cuda, mynn_bl_driver_metal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("length", "cloud", "baseline", "all"),
                        default="all")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--entry", choices=("driver", "caller"), default="driver",
                        help="compare the column driver or public MYNN coupling caller")
    parser.add_argument("--reload-scratch", action="store_true",
                        help="recreate the scratch arena on warm calls, retaining output fields")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    gpu_module, mynn_bl_driver_cuda, mynn_bl_driver_metal = install_backends()
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    from woof.core import mynn_pbl_runtime as runtime

    cases = [("nz12", 11, 12, 48), ("nz30", 12, 30, 64), ("nz49", 13, 49, 64),
             ("nz64", 14, 64, 48)]
    groups = {
        "length": [dict(bl_mynn_version="gsd_41", bl_mynn_mixlength=m,
                        bl_mynn_gsd41_unsquared_qtke=u)
                   for m in (1, 2) for u in (False, True)],
        "cloud": [dict(bl_mynn_version="gsd_41", bl_mynn_mixlength=m,
                       bl_mynn_cloud_tendency_form="gsd_41",
                       bl_mynn_gsd41_unsquared_qtke=u)
                  for m in (1, 2) for u in (False, True)],
        "baseline": [dict(bl_mynn_version="wrf_461", bl_mynn_mixlength=m)
                     for m in (1, 2)],
    }
    wanted = groups if args.group == "all" else {args.group: groups[args.group]}
    records = []
    with MetalContext() as context:
        xp = MetalArrayModule(context)
        for group, options in wanted.items():
            for option in options:
                flag_qs = True
                for label, seed, nz, ncol in cases:
                    scratch = Scratch(xp)
                    case = make_case(seed, nz, ncol)
                    sq = specific(case)
                    delt = np.float32(30.)

                    def run_host(values, cold, option=option, flag_qs=flag_qs):
                        out = mynn_bl_driver_cuda(
                            dict(values), initflag=1 if cold else 0,
                            delt=delt, flag_qs=flag_qs, **option)
                        if args.entry == "caller":
                            gpu_module.mynn_pbl_kernel("mynn_wrapper_from_specific",
                                option["bl_mynn_version"])(((nz * ncol + 127) // 128,), (128,),
                                (values["sqv"], out["rqvblten"], out["rqcblten"],
                                 out["rqiblten"], out["qc_bl"], out["qi_bl"],
                                 np.int32(nz * ncol)))
                        return out

                    def run_metal(case, sq, state, carried, cold,
                                  option=option, flag_qs=flag_qs):
                        if args.reload_scratch and not cold:
                            scratch.slots.clear()
                        def lay(a):
                            return xp.asarray(np.ascontiguousarray(a, np.float32))
                        values = {n: lay(case[n]) for n in (
                            "dz", "u", "v", "w", "th", "p", "exner", "rho", "tk")}
                        values.update({n: lay(sq[n]) for n in sq})
                        values.update({n: lay(case[n]) for n in ("qv", "qc", "qi", "qs")})
                        values.update({n: lay(state[n]) for n in STATE})
                        for n in ("dx", "xland", "ts", "ps", "ust", "hfx", "qfx", "wspd"):
                            values[n] = lay(case[n])
                        values["uoce"] = lay(np.zeros(ncol))
                        values["voce"] = lay(np.zeros(ncol))
                        values["pblh"] = lay(carried["pblh"])
                        values["rmol"] = lay(carried["rmol"])
                        if args.entry == "caller":
                            shape = (nz, 1, ncol)
                            atmosphere = {key: values[name].view(shape)
                                for name, key in runtime._ATMOSPHERE_LAYERS}
                            fields = {field: values[name].view(shape)
                                for name, field in runtime._STATE_FIELD.items()}
                            fields.update({key: values[name].view((1, ncol))
                                for name, key in runtime._COLUMN_FIELDS})
                            for name in (*runtime.MYNN_PBL_DIAGNOSTICS_2D,
                                         *runtime.MYNN_PBL_DIAGNOSTICS_INT_2D,
                                         "kpbl"):
                                dtype = np.int32 if name in INT_OUT else np.float32
                                fields.setdefault(name, xp.zeros((1, ncol), dtype))
                            for name in ("exch_h", "exch_m"):
                                fields[name] = xp.zeros(shape, np.float32)
                            rates = runtime.mynn_pbl_step(atmosphere, fields,
                                w=values["w"].view(shape), dx=3000., delt=delt,
                                itimestep=1 if cold else 2, mp_physics=8,
                                state=scratch, **option)
                            out = {solver: rates[target].view((nz, ncol))
                                for solver, target in runtime._TENDENCY_OUT}
                            out.update({name: fields[field].view((nz, ncol))
                                for name, field in runtime._STATE_FIELD.items()})
                            out.update({name: fields[name].view((nz, ncol))
                                for name in ("exch_h", "exch_m")})
                            out.update({name: fields[name].view((ncol,))
                                for name in (*COLUMN_OUT, *INT_OUT)})
                        else:
                            out = mynn_bl_driver_metal(
                                values, initflag=1 if cold else 0, delt=delt,
                                state=scratch, flag_qs=flag_qs, **option)
                        result = {}
                        for name, value in out.items():
                            result[name] = value.to_numpy()
                        return result

                    recs = step(run_host, run_metal, case, sq, nz, ncol,
                                args.steps, None, layer_out=tuple(n for n in LAYER_OUT
                                    if args.entry == "driver" or n not in ("rqsblten", "dozone")))
                    row = {"entry": args.entry, "reload_scratch": args.reload_scratch,
                           "group": group, "option": {k: v for k, v in option.items()},
                           "case": label, "nz": nz, "ncol": ncol,
                           "words": sum(r["words"] for r in recs),
                           "mismatches": sum(r["mismatches"] for r in recs),
                           "by_step": recs}
                    records.append(row)
                    print(json.dumps({k: row[k] for k in (
                        "group", "option", "case", "words", "mismatches")}
                        | {"by_field": {r["step"]: r["by_field"] for r in recs if r["by_field"]},
                           "first": {r["step"]: r["first"] for r in recs if r["first"]},
                           "host_nonfinite": sum(r["host_nonfinite"] for r in recs)}),
                        flush=True)
    summary = {"words": sum(r["words"] for r in records),
               "mismatches": sum(r["mismatches"] for r in records)}
    print(json.dumps(summary))
    if args.out:
        args.out.write_text(json.dumps(records, indent=2) + "\n")
    return 0 if summary["mismatches"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
