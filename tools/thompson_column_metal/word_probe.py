"""Word probe: the generated wrf_44 Thompson column, host-built CUDA vs Metal, on deterministic synthetic coefficient tables."""
import sys, json, argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools/thompson_real_column_parity"), str(Path(__file__).resolve().parent)]
import host_backend
host_backend.CXX_FLAGS += ("-w",)
from host_backend import HostModule
from woof.core import kernels
from woof.metal.array import MetalArrayModule
from woof.metal import thompson_column_family as fam
import synthetic_tables as column44_tables

STATE = ("qv", "qc", "qi", "qr", "qs", "qg", "ni", "nr", "temperature")
SURF = fam.SURFACE_NAMES


def fixture(nz, nc, seed):
    rng = np.random.default_rng(seed)
    zi = np.linspace(0., 17000., nz + 1)
    zm = 0.5 * (zi[1:] + zi[:-1])
    dzv = np.diff(zi)
    shift = rng.uniform(-25., 20., nc)
    p = (100000. * np.exp(-zm / 7400.))[:, None] * np.ones(nc)
    t = np.maximum(205., 300. - 0.0066 * zm)[:, None] + shift[None, :]
    es = 611.2 * np.exp(17.62 * (t - 273.15) / (t - 30.04))
    qs_sat = 0.622 * es / np.maximum(p - es, 1000.)
    rh = rng.uniform(0.2, 1.15, (nz, nc))
    qv = qs_sat * rh
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)

    def species(prob, hi):
        a = rng.uniform(0., hi, (nz, nc)) * (rng.random((nz, nc)) < prob)
        return a * (rng.random((nz, nc)) > 0.05)
    qc = species(0.35, 1.5e-3); qi = species(0.25, 5e-4); qr = species(0.25, 2e-3)
    qs = species(0.3, 2e-3); qg = species(0.2, 2e-3)
    # zero columns and zero fields
    qc[:, 0] = qi[:, 0] = qr[:, 0] = qs[:, 0] = qg[:, 0] = 0.
    ni = np.where(qi > 0, rng.uniform(1e3, 1e6, (nz, nc)), rng.choice([0., 5e2], (nz, nc)))
    nr = np.where(qr > 0, rng.uniform(1e2, 1e5, (nz, nc)), rng.choice([0., 3e1], (nz, nc)))
    w = rng.normal(0., 1.5, (nz, nc))
    out = {"qv": qv, "qc": qc, "qi": qi, "qr": qr, "qs": qs, "qg": qg, "ni": ni, "nr": nr,
           "temperature": t, "pressure": p, "w": w,
           "dz": np.broadcast_to(dzv[:, None], (nz, nc))}
    out = {k: f32(v).reshape(nz * nc) for k, v in out.items()}
    surf = {n: f32(rng.uniform(0., 5., nc) if n.endswith("nc") else np.zeros(nc)) for n in SURF}
    return out, surf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nz", type=int, nargs="+", default=[24, 64, 65, 100])
    ap.add_argument("--columns", type=int, default=48)
    ap.add_argument("--dt", type=float, default=30.)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    tabs = column44_tables.tables()
    xp = MetalArrayModule()
    mtabs = {n: xp.asarray(v) for n, v in tabs.items()}
    report = []
    bad = 0
    try:
        for nz in a.nz:
            cap = next(k for k in fam.TIERS if nz <= k)
            src = (kernels.module_source("thompson_column") if cap == 64 else
                   kernels.module_source_int_defines("thompson_column", (("THOMPSON_COLUMN_KMAX", cap),)))
            host = HostModule(f"thompson_column_{cap}", src)
            nc = a.columns
            for diag in (0, 1):
                f, surf = fixture(nz, nc, 7000 + nz)
                hf = {k: v.copy() for k, v in f.items()}
                hs = {k: v.copy() for k, v in surf.items()}
                eff = [np.full(nz * nc, 7., np.float32) for _ in range(3)]
                hrefl = np.full(nz * nc, 3., np.float32)
                before = {k: v.copy() for k, v in hf.items()}
                names = [p.name for p in host.kernels["thompson_column_step"]]
                vals = dict(hf); vals.update(hs)
                vals.update(effc=eff[0], effi=eff[1], effs=eff[2], refl=hrefl,
                            diagnostic_due=diag, nz=nz, columns=nc, dt=a.dt)
                vals.update(tabs)
                host.get_function("thompson_column_step")(((nc + 31) // 32,), (32,), tuple(vals[n] for n in names))
                mf = {k: xp.asarray(v.reshape(nz, nc).reshape(nz, 1, nc)) for k, v in f.items()}
                ms = {k: xp.asarray(v.reshape(1, nc)) for k, v in surf.items()}
                meff = [xp.asarray(np.full((nz, 1, nc), 7., np.float32)) for _ in range(3)]
                mrefl = xp.asarray(np.full((nz, 1, nc), 3., np.float32))
                fam.submit_thompson_column(mf, ms, meff, mrefl, mtabs, nz=nz, dt=a.dt, diagnostic_due=diag)
                xp.context.synchronize() if hasattr(xp.context, "synchronize") else None
                row = {"nz": nz, "diag": diag, "mismatch": {}, "changed": {}}
                pairs = [(k, hf[k], mf[k]) for k in STATE] + [(k, hs[k], ms[k]) for k in SURF] + \
                        [(n, e, m) for n, e, m in zip(("effc", "effi", "effs"), eff, meff)] + [("refl", hrefl, mrefl)]
                for n, h, m in pairs:
                    g = m.to_numpy().reshape(-1)
                    diff = int(np.count_nonzero(g.view(np.uint32) != h.reshape(-1).view(np.uint32)))
                    row["mismatch"][n] = diff
                    base = before.get(n)
                    if base is not None:
                        row["changed"][n] = int(np.count_nonzero(h.view(np.uint32) != base.view(np.uint32)))
                    bad += diff
                row["active"] = {"refl_written": int(np.count_nonzero(hrefl != 3.0)),
                                 "refl_gt_0dbz": int(np.count_nonzero(hrefl > 0.0)),
                                 "rainncv_nonzero": int(np.count_nonzero(hs["rainncv"])),
                                 "snowncv_nonzero": int(np.count_nonzero(hs["snowncv"])),
                                 "graupelncv_nonzero": int(np.count_nonzero(hs["graupelncv"])),
                                 "effc_written": int(np.count_nonzero(eff[0] != 7.0)),
                                 "melting_columns_sr_between": int(np.count_nonzero((hs["sr"] > 0) & (hs["sr"] < 1)))}
                report.append(row)
                print(json.dumps({k: row[k] for k in ('nz','diag','active')}), 'MISMATCH_SUM', sum(row['mismatch'].values()), flush=True)
    finally:
        xp.context.close()
    print("TOTAL_MISMATCHED_WORDS", bad)
    if a.out:
        a.out.write_text(json.dumps(report, indent=1))
    return int(bad > 0)


if __name__ == "__main__":
    raise SystemExit(main())
