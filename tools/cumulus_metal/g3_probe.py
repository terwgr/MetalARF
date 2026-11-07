#!/usr/bin/env python3
"""Word probe for the Metal Grell 3D launcher.

``host`` runs the CUDA unit, built for the host without contraction, through
the unchanged CUDA launcher (woof.core.g3.run_g3); ``metal`` runs the
production Metal launcher on the same inputs and compares literal words.
"""
from pathlib import Path
import argparse
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/thompson_real_column_parity"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# (nz, ny, nx, seed, ishallow, avedx, clos_choice, qi present, periodic)
CASES = ((30, 16, 18, 21, 0, 1, 0, 1, 0), (40, 17, 16, 22, 1, 1, 0, 1, 0),
         (50, 16, 16, 23, 1, 3, 3, 0, 0), (36, 16, 17, 24, 1, 1, 9, 1, 1),
         (64, 16, 16, 25, 1, 3, 0, 1, 0))
DX, DT = 12000.0, 30.0
F = np.float32


def inputs(nz, ny, nx, seed):
    from column_fixture import make_columns
    ncol = ny * nx
    a = make_columns(nz, ncol, seed)
    rng = np.random.default_rng(seed + 500)
    shape = (nz, ny, nx)
    out = {k: a[k].reshape(shape) for k in
           ("u3d", "v3d", "t3d", "qv3d", "pcps", "pi3d", "rho3d")}
    out["qc3d"], out["qi3d"] = a["qc3d"].reshape(shape), a["qi3d"].reshape(shape)
    out["w"] = a["w"].reshape(nz + 1, ny, nx)
    out["p8w"] = a["p8w"].reshape(nz + 1, ny, nx)
    z = np.linspace(0., 1., nz, dtype=np.float64)[:, None]
    out["thdyn"] = (rng.uniform(-1e-4, 7e-4, ncol)[None] * (1. - z) * 1e-1
                    ).astype(F).reshape(shape)
    out["qvdyn"] = (a["qvften"] * 4.).astype(F).reshape(shape)
    out["lw"] = rng.normal(0., 1e-5, shape).astype(F)
    out["sw"] = rng.normal(0., 1e-5, shape).astype(F)
    out["thbl"] = rng.normal(0., 2e-5, shape).astype(F)
    out["qvbl"] = a["qvpblten"].reshape(shape)
    out["xland"] = a["xland"].reshape(ny, nx)
    out["ht"] = rng.uniform(0., 400., (ny, nx)).astype(F)
    out["gsw"] = rng.uniform(0., 700., (ny, nx)).astype(F)
    out["kpbl"] = rng.integers(1, 12, (ny, nx)).astype(np.int32)
    out["apr"] = rng.uniform(0., 3., (8, ny, nx)).astype(F)
    out["xf_ens"] = rng.uniform(0., 1., (144, ny, nx)).astype(F)
    out["pr_ens"] = rng.uniform(0., 1., (144, ny, nx)).astype(F)
    return out


def fold(a):
    """The driver's forcing fold, in the driver's operation order."""
    exner = a["pi3d"]
    th = (a["thdyn"] + (a["lw"] + a["sw"]) + a["thbl"]) * exner
    qv = a["qvdyn"] + a["qvbl"]
    return th.astype(F), qv.astype(F)


DIAG = ("gd_cloud_a", "qc_cu", "gd_cloud2_a", "qi_cu", "gd_cldfr", "raincv_a",
        "raincv_b")
CALLS = 3
AVG_MINUTES = 1.0
NAMES = ("lev", "sfc", "isfc", "xf_ens", "pr_ens", "apr", "rthcuten",
         "rqvcuten", "rqccuten", "rqicuten", "raincv")


def host(args):
    import host_backend
    host_backend.CXX_FLAGS += ("-w",)
    host_backend.install()
    import woof.core.g3 as g3
    # numpy has no .get(): read the kernel's own slot count here instead.
    slots = np.zeros(1, np.int32)
    g3._module().get_function("g3_slot_count")((1,), (1,), (slots,))
    assert int(slots[0]) == g3.G3_SLOT_COUNT
    g3.g3_slot_count = lambda module=None: int(slots[0])
    run_g3 = g3.run_g3
    from woof.core.g3 import run_g3_diag
    store = {}
    for case in CASES:
        nz, ny, nx, seed, ish, avedx, ichoice, fqi, periodic = case
        a = inputs(nz, ny, nx, seed)
        th, qv = fold(a)
        lev_in = np.stack([a["u3d"], a["v3d"], a["t3d"], a["qv3d"], a["pcps"],
                           a["pi3d"], a["rho3d"], th, qv, a["thbl"], a["qvbl"]]).astype(F)
        sfc_in = np.stack([a["p8w"][0], a["ht"], a["xland"], a["gsw"]]).astype(F)
        apr, xf, pr = a["apr"].copy(), a["xf_ens"].copy(), a["pr_ens"].copy()
        step = run_g3(lev_in=lev_in, w=a["w"][:nz], sfc_in=sfc_in, kpbl=a["kpbl"],
                      apr=apr, dt=DT, dx=DX, ishallow=ish, avedx=avedx,
                      ichoice=ichoice, fqi=fqi, periodic_x=bool(periodic),
                      periodic_y=bool(periodic), xf_ens=xf, pr_ens=pr)
        tag = "_".join(map(str, case))
        for k, v in a.items():
            store[f"in_{tag}_{k}"] = v
        for k in NAMES:
            store[f"out_{tag}_{k}"] = np.asarray(getattr(step, k))
        # The convective-diagnostic window over CALLS consecutive calls.
        adaptive = bool(seed % 2)
        averages = {}
        for call in range(CALLS):
            if call:
                step = run_g3(lev_in=lev_in, w=a["w"][:nz], sfc_in=sfc_in,
                              kpbl=a["kpbl"], apr=apr, dt=DT, dx=DX, ishallow=ish,
                              avedx=avedx, ichoice=ichoice, fqi=fqi,
                              periodic_x=bool(periodic), periodic_y=bool(periodic),
                              xf_ens=xf, pr_ens=pr)
            run_g3_diag(averages, cloud=step.lev[7], ice=step.lev[8],
                        rain=step.raincv, temperature=a["t3d"], pressure=a["pcps"],
                        qv=a["qv3d"], qc=a["qc3d"], qi=a["qi3d"] if fqi else
                        np.zeros_like(a["qi3d"]), fqi=fqi, dt=DT,
                        curr_secs=DT * call, avg_minutes=AVG_MINUTES,
                        adaptive=adaptive)
            for k in DIAG + ("clock",):
                store[f"diag_{tag}_{call}_{k}"] = np.array(averages[k])
            store[f"diag_{tag}_{call}_ktop"] = np.array(step.isfc[3])
    np.savez(args.out, **store)


def metal(args):
    from types import SimpleNamespace
    from woof.metal.array import MetalArrayModule
    from woof.core.g3 import Grell3D
    ref = np.load(args.ref)
    xp = MetalArrayModule()
    total = 0
    try:
        for case in CASES:
            nz, ny, nx, seed, ish, avedx, ichoice, fqi, periodic = case
            tag = "_".join(map(str, case))
            g = lambda k: ref[f"in_{tag}_{k}"]
            up = lambda a: xp.asarray(np.ascontiguousarray(a))
            atmosphere = {
                "u": up(g("u3d")), "v": up(g("v3d")), "temperature": up(g("t3d")),
                "qv": up(g("qv3d")), "exner": up(g("pi3d")), "rho": up(g("rho3d")),
                "pressure": up(g("pcps")), "p_interface": up(g("p8w")),
                "qc": up(g("qc3d"))}
            if fqi:
                atmosphere["qi"] = up(g("qi3d"))
            state = SimpleNamespace(
                _metal_setup_state=True, _array_module=xp, p=np.zeros((nz, ny, nx)),
                w=up(g("w")), ht=up(g("ht")), qi=object() if fqi else None)
            fields = {"xland": up(g("xland")), "gsw": up(g("gsw")),
                      "kpbl": up(g("kpbl")), "g3_apr": up(g("apr")),
                      "g3_xf_ens": up(g("xf_ens")), "g3_pr_ens": up(g("pr_ens")),
                      "g3_lev": xp.zeros((9, nz, ny, nx), np.float32),
                      "g3_sfc": xp.zeros((7, ny, nx), np.float32),
                      "g3_isfc": xp.zeros((4, ny, nx), np.int32)}
            driver = SimpleNamespace(
                gf_rthdynten=up(g("thdyn")), gf_rqvdynten=up(g("qvdyn")),
                rthratenlw=up(g("lw")), rthratensw=up(g("sw")),
                gf_rthblten=up(g("thbl")), gf_rqvblten=up(g("qvbl")))
            cfg = SimpleNamespace(
                dt=DT, dx=DX, ishallow=ish, cugd_avedx=avedx, clos_choice=ichoice,
                maxiens=1, maxens=3, maxens2=3, maxens3=16, ensdim=144,
                open_x=not periodic, open_y=not periodic, specified=False,
                nested=False, lightning_option=11, cu_diag=1,
                convtrans_avglen_m=AVG_MINUTES, use_adaptive_time_step=bool(seed % 2))
            state.elapsed_seconds = 0.0
            state.scratch = lambda shape, slot, dtype=None: scratch.setdefault(
                slot, xp.zeros(shape, dtype or np.float32))
            scratch = {}
            for k in DIAG:
                fields[k] = xp.zeros(
                    (ny, nx) if k.startswith("raincv") else (nz, ny, nx), np.float32)
            fields["g3_diag_clock"] = xp.zeros((4,), np.int32)
            owner = Grell3D()
            owner.bind_driver(driver)
            result = owner(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
            got = {"lev": fields["g3_lev"], "sfc": fields["g3_sfc"],
                   "isfc": fields["g3_isfc"], "xf_ens": fields["g3_xf_ens"],
                   "pr_ens": fields["g3_pr_ens"], "apr": fields["g3_apr"],
                   "rthcuten": result.rthcuten, "rqvcuten": result.rqvcuten,
                   "rqccuten": result.rqccuten, "raincv": result.rainc,
                   "rqicuten": result.rqicuten}
            def compare(name, exp, value):
                nonlocal total
                bad = int(np.count_nonzero(
                    exp.reshape(-1).view(np.uint32)
                    != value.to_numpy().reshape(-1).view(np.uint32)))
                total += bad
                print(tag, name, "mismatch", bad, "nonzero",
                      int(np.count_nonzero(exp)), "of", exp.size)
            compare("ktop_deep scratch", ref[f"out_{tag}_isfc"][3],
                    scratch["ltng_ktop_deep"])
            for call in range(CALLS):
                if call:
                    state.elapsed_seconds = DT * call
                    result = owner(atmosphere=atmosphere, fields=fields,
                                   state=state, cfg=cfg)
                for k in DIAG + ("clock",):
                    compare(f"diag{call} {k}", ref[f"diag_{tag}_{call}_{k}"],
                            fields["g3_diag_clock" if k == "clock" else k])
            for k in NAMES:
                exp = ref[f"out_{tag}_{k}"]
                if k == "rqicuten" and not fqi:
                    continue
                value = got[k].to_numpy()
                bad = int(np.count_nonzero(exp.reshape(-1).view(np.uint32)
                                           != value.reshape(-1).view(np.uint32)))
                total += bad
                print(tag, k, "mismatch", bad, "nonzero", int(np.count_nonzero(exp)),
                      "of", exp.size)
        print("TOTAL MISMATCHED WORDS", total)
        return int(total != 0)
    finally:
        xp.context.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("host"); h.add_argument("--out", type=Path)
    m = sub.add_parser("metal"); m.add_argument("--ref", type=Path)
    a = p.parse_args()
    raise SystemExit(host(a) if a.cmd == "host" else metal(a))
