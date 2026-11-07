#!/usr/bin/env python3
"""Word probe for the Metal New SAS launcher.

``host`` runs the CUDA unit, built for the host without contraction, through
the unchanged table-driven CUDA launcher; ``metal`` runs the production Metal
launcher on the same inputs and compares literal words.
"""
from pathlib import Path
import argparse
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/thompson_real_column_parity"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

LEVEL = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten")
# (nz, ny, nx, seed, mp_physics, nsas_dx_factor, stepcu)
CASES = ((30, 8, 12, 11, 8, 0, 1), (30, 8, 12, 12, 8, 1, 3),
         (64, 6, 10, 13, 0, 0, 1), (72, 5, 9, 14, 3, 1, 2))
DX, DT, PGCON = 12000.0, 30.0, 0.55


def columns(nz, ny, nx, seed):
    from column_fixture import make_columns
    ncol = ny * nx
    a = make_columns(nz, ncol, seed)
    rng = np.random.default_rng(seed + 100)
    a["hpbl"] = rng.uniform(100., 2500., ncol).astype(np.float32)
    a["hfx"] = rng.uniform(-20., 400., ncol).astype(np.float32)
    # NSAS reads the model-top-down column in kPa-scale internally; it takes
    # the same level-major mass and interface planes as the fixture.
    return a


def host(args):
    import host_backend
    host_backend.CXX_FLAGS += ("-w",)
    host_backend.install()
    from woof.core import column_cumulus as cc
    import cupy as cp
    # The host device has no SM count; the tile only sets the workspace layout.
    type(cp.cuda.Device()).attributes = {"MultiProcessorCount": 1}
    store = {}
    for nz, ny, nx, seed, mp, dxf, stepcu in CASES:
        a = columns(nz, ny, nx, seed)
        ncol = ny * nx
        scheme = cc.COLUMN_CUMULUS_SCHEMES[96]
        mass = {"t": a["t3d"], "qv": a["qv3d"], "qc": a["qc3d"], "qi": a["qi3d"],
                "p": a["pcps"], "pi": a["pi3d"], "rho": a["rho3d"],
                "dz8w": a["dz8w"], "u": a["u3d"], "v": a["v3d"]}
        iface = {"p8w": a["p8w"], "w": a["w"]}
        sc = np.empty((ncol, len(cc.SC_FIELDS)), np.float32)
        for name, key in (("xland", "xland"), ("hpbl", "hpbl"), ("hfx", "hfx"), ("qfx", "qfx")):
            sc[:, cc.SC_FIELDS.index(name)] = a[key]
        sc[:, cc.SC_FIELDS.index("dx")] = np.float32(DX)
        sc[:, cc.SC_FIELDS.index("dt")] = np.float32(DT)
        sc[:, cc.SC_FIELDS.index("pgcon")] = np.float32(PGCON)
        ic = np.empty((ncol, len(cc.IC_FIELDS)), np.int32)
        ic[:, cc.IC_FIELDS.index("stepcu")] = stepcu
        ic[:, cc.IC_FIELDS.index("ncloud")] = cc.wrf_sas_ncloud(mp)
        ic[:, cc.IC_FIELDS.index("dx_factor")] = dxf
        out = {n: np.empty((nz, ncol), np.float32) for n in LEVEL}
        sfc = np.empty((4, ncol), np.float32)
        cc.launch_columns(scheme, mass=mass, iface=iface, sc=sc, ic=ic, outputs=out,
                          sfc=sfc, nz=nz, ncol=ncol, row_width=nx)
        tag = f"{nz}_{ny}_{nx}_{seed}_{mp}_{dxf}_{stepcu}"
        for k, v in a.items():
            store[f"in_{tag}_{k}"] = v
        for k in LEVEL:
            store[f"out_{tag}_{k}"] = out[k]
        store[f"out_{tag}_sfc"] = sfc
    np.savez(args.out, **store)


def metal(args):
    from types import SimpleNamespace
    from woof.metal.array import MetalArrayModule
    from woof.core.column_cumulus import ColumnCumulus
    ref = np.load(args.ref)
    xp = MetalArrayModule()
    total = 0
    try:
        for nz, ny, nx, seed, mp, dxf, stepcu in CASES:
            tag = f"{nz}_{ny}_{nx}_{seed}_{mp}_{dxf}_{stepcu}"
            g = lambda k: ref[f"in_{tag}_{k}"]
            vol = lambda a: xp.asarray(a.reshape(a.shape[0], ny, nx))
            surf = lambda a: xp.asarray(a.reshape(ny, nx))
            atmosphere = {
                "u": vol(g("u3d")), "v": vol(g("v3d")), "temperature": vol(g("t3d")),
                "qv": vol(g("qv3d")), "qc": vol(g("qc3d")), "qi": vol(g("qi3d")),
                "exner": vol(g("pi3d")), "rho": vol(g("rho3d")), "dz": vol(g("dz8w")),
                "pressure": vol(g("pcps")), "p_interface": vol(g("p8w"))}
            state = SimpleNamespace(
                _metal_setup_state=True, _array_module=xp, p=np.zeros((nz, ny, nx)),
                w=vol(g("w")), qi=object(), qc=object())
            fields = {"xland": surf(g("xland")), "pblh": surf(g("hpbl")),
                      "hfx": surf(g("hfx")), "qfx": surf(g("qfx"))}
            cfg = SimpleNamespace(mp_physics=mp, dx=DX, dt=DT, sas_pgcon=PGCON,
                                  nsas_dx_factor=dxf)
            owner = ColumnCumulus(96)
            owner.bind_driver(SimpleNamespace(stepcu=stepcu))
            result = owner(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
            got = {k: getattr(result, k).to_numpy().reshape(nz, ny * nx) for k in LEVEL}
            got["pratec"] = result.pratec.to_numpy().reshape(-1)
            got["clock"] = result.clock_raincv.to_numpy().reshape(-1)
            for k in LEVEL + ("pratec",):
                exp = (ref[f"out_{tag}_sfc"][1] if k == "pratec"
                       else ref[f"out_{tag}_{k}"]).reshape(-1)
                bad = int(np.count_nonzero(exp.view(np.uint32) != got[k].reshape(-1).view(np.uint32)))
                total += bad
                print(tag, k, "mismatch", bad, "nonzero", int(np.count_nonzero(exp)), "of", exp.size)
            clock = (ref[f"out_{tag}_sfc"][1] * np.float32(DT)).reshape(-1)
            bad = int(np.count_nonzero(clock.view(np.uint32) != got["clock"].view(np.uint32)))
            total += bad
            print(tag, "clock_raincv mismatch", bad)
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
