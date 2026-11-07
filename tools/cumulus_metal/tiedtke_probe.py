#!/usr/bin/env python3
"""Word probe for the Metal original-Tiedtke launcher.

``host`` runs the CUDA unit, built for the host without contraction, through
the unchanged CUDA launcher and stores inputs and outputs; ``metal`` runs the
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

NAMES = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten",
         "raincv", "pratec")


def host(args):
    import host_backend
    host_backend.CXX_FLAGS += ("-w",)
    host_backend.install()
    from woof.core import tiedtke
    from column_fixture import make_columns
    store = {}
    for nz, ny, nx, seed in ((30, 16, 25, 1), (64, 12, 20, 2), (72, 10, 16, 3)):
        inputs = make_columns(nz, ny * nx, seed)
        refused = np.zeros((), dtype=bool)
        out = tiedtke.column_step(
            {k: v.copy() for k, v in inputs.items()}, dt=30.0, itimestep=2,
            stepcu=1, f_qc=True, f_qi=True, row_width=nx, refused=refused)
        store[f"refused_{nz}_{ny}_{nx}"] = np.asarray(out["cu_act_flag"])
        tag = f"{nz}_{ny}_{nx}"
        for k, v in inputs.items():
            store[f"in_{tag}_{k}"] = v
        for k in NAMES:
            store[f"out_{tag}_{k}"] = np.asarray(out[k])
    np.savez(args.out, **store)


def metal(args):
    from woof.metal.array import MetalArrayModule
    from woof.metal import cumulus_launch, tiedtke_family
    ref = np.load(args.ref)
    xp = MetalArrayModule()
    total = 0
    active = {}
    try:
        for tag in sorted({"_".join(k.split("_")[1:4]) for k in ref.files
                           if k.startswith("in_")}):
            nz, ny, nx = map(int, tag.split("_"))
            ncol = ny * nx
            for limit in (None, nx * 2):
                if limit:
                    cumulus_launch.COLUMNS_PER_SUBMIT = limit
                else:
                    cumulus_launch.COLUMNS_PER_SUBMIT = 2048
                tiedtke_family.COLUMNS_PER_SUBMIT = cumulus_launch.COLUMNS_PER_SUBMIT
                got = run_production(xp, ref, tag, nz, ny, nx)
                got["pratec"], flags = run_direct_pratec(xp, ref, tag, nz, ny, nx)
                refused = int(np.count_nonzero(
                    flags.view(np.uint32) != ref[f"refused_{tag}"].view(np.uint32)))
                total += refused
                print(tag, "flag mismatches", refused, "refused columns",
                      int(np.count_nonzero(flags < 0)))
                for k in NAMES:
                    exp = ref[f"out_{tag}_{k}"].reshape(-1)
                    have = got[k].reshape(-1)
                    bad = int(np.count_nonzero(exp.view(np.uint32) != have.view(np.uint32)))
                    total += bad
                    changed = int(np.count_nonzero(exp != 0))
                    active[(tag, bool(limit), k)] = (bad, changed, exp.size)
                    print(tag, "tiled" if limit else "whole", k, "mismatch", bad, "nonzero", changed, "of", exp.size)
        print("TOTAL MISMATCHED WORDS", total)
        return int(total != 0)
    finally:
        xp.context.close()


def run_direct_pratec(xp, ref, tag, nz, ny, nx):
    """The unit's second surface output, from the direct launcher."""
    from woof.metal.tiedtke_family import column_step
    ncol = ny * nx
    names = ("u3d", "v3d", "w", "t3d", "qv3d", "qc3d", "qi3d", "pi3d", "rho3d",
             "qvften", "qvpblten", "dz8w", "pcps", "p8w", "xland", "qfx", "znu")
    inputs = {k: xp.asarray(ref[f"in_{tag}_{k}"].reshape(-1)) for k in names}
    outputs = {k: xp.zeros((ncol * (nz if k not in ("raincv", "pratec") else 1),), np.float32)
               for k in NAMES}
    flag = column_step(xp, inputs, outputs, dt=30.0, itimestep=2, stepcu=1,
                       nz=nz, ncol=ncol, width=nx, f_qc=1, f_qi=1)
    return outputs["pratec"].to_numpy(), flag.to_numpy()


def run_production(xp, ref, tag, nz, ny, nx):
    from types import SimpleNamespace
    from woof.core.tiedtke import Tiedtke
    ncol = ny * nx
    g = lambda k: ref[f"in_{tag}_{k}"]
    vol = lambda a: xp.asarray(a.reshape(a.shape[0], ny, nx))
    surf = lambda a: xp.asarray(a.reshape(ny, nx))
    atmosphere = {"u": vol(g("u3d")), "v": vol(g("v3d")), "temperature": vol(g("t3d")),
                  "qv": vol(g("qv3d")), "qc": vol(g("qc3d")), "qi": vol(g("qi3d")),
                  "exner": vol(g("pi3d")), "rho": vol(g("rho3d")), "dz": vol(g("dz8w")),
                  "pressure": vol(g("pcps")), "p_interface": vol(g("p8w"))}
    state = SimpleNamespace(
        _metal_setup_state=True, _array_module=xp, p=np.zeros((nz, ny, nx)),
        w=vol(g("w")), znu=xp.asarray(g("znu")), qi=object(), elapsed_seconds=30.0,
        domain_start_offset=0.0)
    driver = SimpleNamespace(gf_rqvdynten=vol(g("qvften")), gf_rqvblten=vol(g("qvpblten")))
    cfg = SimpleNamespace(mp_physics=8, dt=30.0, cudt_minutes=0.0)
    cfg.cu_physics = 6
    owner = Tiedtke()
    owner.bind_driver(driver)
    fields = {"xland": surf(g("xland")), "qfx": surf(g("qfx"))}
    result = owner(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    out = {k: getattr(result, k).to_numpy() for k in
           ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten")}
    out["raincv"] = result.rainc.to_numpy()
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("host"); h.add_argument("--out", type=Path)
    m = sub.add_parser("metal"); m.add_argument("--ref", type=Path)
    a = p.parse_args()
    raise SystemExit(host(a) if a.cmd == "host" else metal(a))
