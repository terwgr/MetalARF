"""Word-for-word check of a Metal ensemble run against host transcriptions of the CUDA kernels.

usage: compare_products.py RUN_DIR DUMP_DIR [--domain d01]

DUMP_DIR holds, per member and frame, ``raw_*.npz`` (the member planes the
collector received) and ``diag_*.npz`` (the diagnostics it handed the spool).
Three things are compared, every float32 as its bit pattern:

* the diagnostics against scalar transcriptions of ``_HEADLINE_SOURCE``,
  ``visibility_column`` and ``saved_frame_ice``, one cell at a time;
* the published product file against ``ensemble_field_products`` and
  ``ensemble_spaghetti_cells`` (batch_products._SOURCE) run on those diagnostics;
* the NumPy joint-event field against ``_COMPOUND_SOURCE``, cell by cell.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

F32 = np.float32
NAN = float("nan")
COMPARISON = {"visibility": "lt", "temperature2_below_freezing": "lt",
              "ice_accretion": "gt", "ice_accretion_horizontal": "gt"}


def fin(x):
    return math.isfinite(float(x))


def bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def same(a, b):
    return np.array_equal(bits(a), bits(b)) or (
        np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(bits(a)[~np.isnan(a)], bits(b)[~np.isnan(b)]))


# --- scalar transcriptions of the CUDA diagnostics -------------------------------------------

def dewpoint_humidity(t2, q2, psfc):
    q, p, t = float(q2), float(psfc), float(t2)
    vapor = (q * p) / (0.622 + q)
    if not (fin(q2) and fin(psfc)) or q2 <= 0 or psfc <= 0:
        dew = F32(NAN)
    else:
        lg = math.log(max(vapor, 1.0) / 611.2)
        dew = F32(((243.5 * lg) / (17.67 - lg)) + 273.15)
    if not (fin(t2) and fin(q2) and fin(psfc)) or t2 <= 0:
        return dew, F32(NAN)
    temperature = t - 273.15
    rh = (100.0 * vapor) / (611.2 * math.exp((17.67 * temperature) / (temperature + 243.5)))
    return dew, (F32(NAN) if math.isnan(rh) else F32(min(max(rh, 0.0), 100.0)))


def visibility(p, theta, base, qv, species, mode):
    a = (.99999683, -.0090826951, .000078736169, -.00000061117958, 4.3884187e-9,
         -2.9883885e-11, 2.1874425e-13, -1.7892321e-15, 1.1112018e-17, -3.0994571e-20)

    def good(x):
        return math.isfinite(x) and abs(x) < 1.e30

    def rh(pr, t, r):
        x, pol = t - 273.15, 0.0
        for j in range(9, -1, -1):
            pol = pol * x + a[j]
        es = 6.1078 / math.pow(pol, 8.0) if pol > 0 else NAN
        v = 100.0 * (((pr / 100.0) * r) / (.62197 + r)) / es
        return min(100.0, max(0.0, v)) if not math.isnan(v) else NAN
    mq, maxrh, ok = [0.0] * 5, 0.0, True
    for k in range(len(p)):
        pr = float(p[k])
        th = float(theta[k]) + (300.0 if mode in (0, 2) else float(base[k]))
        t = th * math.pow(pr / 100000.0, 0.2857142857) if pr > 0 else NAN
        r = float(qv[k])
        if not (good(pr) and good(t) and good(r)) or pr <= 0 or t < 150 or t > 350 or r < 0:
            ok = False
        if k == 0:
            p0, t0, r0 = pr, t, r
        if k < 2:
            v = rh(pr, t, r)
            maxrh = maxrh if math.isnan(v) else max(maxrh, v)
        for s, plane in enumerate(species):
            if plane is not None:
                q = float(plane[k])
                ok = ok and good(q)
                mq[s] = max(mq[s], max(0.0, q))
    if not ok:
        return F32(NAN)
    tv = t0 * (1.0 + r0 / .62197) / (1.0 + r0)
    rho = p0 / (287.0 * tv)
    volume = (1.0 + r0) / rho + (mq[0] + mq[1]) / 1000.0 + (mq[2] + mq[3] + mq[4]) / 917.0
    c = [1000.0 * m / volume for m in mq]
    beta = (144.7 * c[0] ** .88 + 2.24 * c[1] ** .75 + 327.8 * c[2]
            + 10.36 * c[3] ** .7776 + 8. * c[4] ** .75 + 1.e-10)
    haze = 90000. * math.exp(-2.5 * min(.8, max(0., maxrh / 100. - .15)))
    return F32(min(90000., min(1000. * 3.912023005428146 / beta, haze)))


def ice(rain, prev_rain, prev, t2, skin, q2, psfc, u10, v10, qr, qs, qg, hours, have_skin):
    t, pressure = float(t2) - 273.15, float(psfc) / 100.0
    q, r, s, g = float(q2), float(qr), float(qs), float(qg)
    delta = float(rain) - float(prev_rain)
    if (not math.isfinite(t) or t2 <= 0 or not math.isfinite(pressure) or pressure <= 0
            or not math.isfinite(q) or q < 0 or not all(math.isfinite(x) for x in (r, s, g))
            or not math.isfinite(delta) or delta < 0 or not fin(u10) or not fin(v10)
            or (have_skin and (not fin(skin) or skin <= 0)) or not all(fin(x) for x in prev)):
        return [F32(NAN)] * 5
    vapor = pressure * q / (0.622 + q)
    lo, hi = -100.0, t
    for _ in range(40):
        tw = 0.5 * (lo + hi)
        es = 6.112 * math.exp(17.67 * tw / (tw + 243.5))
        e = es - 0.00066 * (1.0 + 0.00115 * tw) * pressure * (t - tw)
        if e > vapor:
            hi = tw
        else:
            lo = tw
    tw = 0.5 * (lo + hi)
    r, s, g = max(r, 0.0) * 5.0, max(s, 0.0), max(g, 0.0) * 3.0
    total = r + s + g
    flux = float(psfc) / (287.04 * float(t2)) * total * 3600.0
    cold = t2 < F32(273.15) or (have_skin and skin < F32(273.15))
    freezing = total > 0 and flux >= 0.1 and (s + g) / total <= 0.3 and cold
    active = freezing and delta > 0 and hours > 0
    dz, flat = (delta if active else 0.0), 0.0
    if active and tw < 1.0:
        p = (delta / hours) / 25.4
        v = math.hypot(float(u10), float(v10)) / (1852.0 / 3600.0)
        w = max(tw, -7.0)
        ip = 0.1395 * math.pow(p, -0.541)
        it = ((-0.0071 * w - 0.1039) * w - 0.3904) * w + 0.5545
        iv = (0.0014 * v + 0.0027) * v + 0.7574
        ilr = (0.70 * ip + 0.29 * it + 0.01 * iv if w > -0.35 else
               0.73 * ip + 0.01 * it + 0.26 * iv if v > 12.0 else 0.79 * ip + 0.20 * it + 0.01 * iv)
        flat = delta * max(ilr, 0.0)
    return [F32(float(prev[0]) + dz), F32(float(prev[1]) + flat), F32(float(prev[2]) + flat * 0.394),
            F32(tw), F32(dz / hours) if hours > 0 else F32(0)]


# --- scalar transcription of ensemble_field_products / ensemble_spaghetti_cells ----------------

def reduce_cell(values, thresholds, relation):
    """One cell's mean, spread, min, max, finite count, probabilities and paintball words."""
    sel = {"ge": lambda x, t: x >= t, "gt": lambda x, t: x > t,
           "le": lambda x, t: x <= t, "lt": lambda x, t: x < t}[relation]
    members = len(values)
    total, finite = 0.0, 0
    lo = hi = values[0]
    for x in values:
        if math.isfinite(x):
            finite += 1
            total = total + float(x)
            lo, hi = (x if x < lo else lo), (x if x > hi else hi)
    average = total / members
    deviations = 0.0
    for x in values:
        delta = float(x) - average
        deviations = deviations + delta * delta
    complete = finite == members
    nan = F32(NAN)
    mean = F32(average) if complete else nan
    spread = (F32(0) if members == 1 else F32(math.sqrt(deviations / (members - 1)))) if complete else nan
    probability, words = [], []
    for t in thresholds:
        count, mask = 0, 0
        for m, x in enumerate(values):
            if math.isfinite(x) and sel(x, t):
                count += 1
                mask |= 1 << m
        probability.append(F32(count / members) if complete else nan)
        words.append(mask)
    return mean, spread, (lo if complete else nan), (hi if complete else nan), finite, probability, words


def spaghetti_code(a, b, c, d, t, relation):
    sel = {"ge": lambda x, t: x >= t, "gt": lambda x, t: x > t,
           "le": lambda x, t: x <= t, "lt": lambda x, t: x < t}[relation]
    if not all(math.isfinite(x) for x in (a, b, c, d)):
        return 255
    return (1 if sel(a, t) else 0) | (2 if sel(b, t) else 0) | (4 if sel(c, t) else 0) | (8 if sel(d, t) else 0)


def compound_cell(values, thresholds, relations):
    """_COMPOUND_SOURCE for one point."""
    event, valid = True, True
    for x, t, relation in zip(values, thresholds, relations):
        valid = valid and math.isfinite(x)
        term = x >= t if relation == 0 else x > t if relation == 1 else x <= t if relation == 2 else x < t
        event = event and term
    return F32(1.0 if event else 0.0) if valid else F32(NAN)


def check(name, got, want, failures, counts):
    got, want = np.asarray(got), np.asarray(want)
    equal = got.shape == want.shape and (
        np.array_equal(got, want) if got.dtype.kind in "iu" else same(got, want))
    counts[0] += int(want.size)
    if not equal:
        counts[1] += int(np.count_nonzero(bits(got) != bits(want))) if got.shape == want.shape and got.dtype.kind == "f" else 1
        failures.append(name)
    return equal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run")
    parser.add_argument("dump")
    parser.add_argument("--domain", default="d01")
    args = parser.parse_args(argv)
    from woof.netcdf_bridge import open_dataset
    run, dump = Path(args.run), Path(args.dump)
    failures, counts = [], [0, 0]
    diag = {}
    for path in sorted(dump.glob("diag_m*_*.npz")):
        member, _, valid = path.stem[len("diag_m"):].partition("_")
        diag[(int(member), valid)] = dict(np.load(path))
    raw = {}
    for path in sorted(dump.glob("raw_m*_*.npz")):
        member, _, valid = path.stem[len("raw_m"):].partition("_")
        raw[(int(member), valid)] = dict(np.load(path))
    members = sorted({m for m, _ in diag})
    frames = sorted({v for _, v in diag})
    print(f"{len(members)} members, {len(frames)} frames, fields: {sorted(next(iter(diag.values())))}")
    # 1. diagnostics against the scalar kernels
    for member in members:
        state = None
        for valid in frames:
            d = diag[(member, valid)]
            r = raw[(member, valid)]
            ny, nx = d["temperature2"].shape
            cells = [(j, i) for j in range(ny) for i in range(nx)]
            if "wind10" in d:
                want = np.sqrt(r["U10"] * r["U10"] + r["V10"] * r["V10"])
                check(f"wind10 m{member} {valid}", d["wind10"], want, failures, counts)
            if "dewpoint2" in d:
                pairs = [dewpoint_humidity(r["T2"][c], r["Q2"][c], r["PSFC"][c]) for c in cells]
                check(f"dewpoint2 m{member} {valid}", d["dewpoint2"],
                      np.array([p[0] for p in pairs], np.float32).reshape(ny, nx), failures, counts)
                check(f"humidity2 m{member} {valid}", d["humidity2"],
                      np.array([p[1] for p in pairs], np.float32).reshape(ny, nx), failures, counts)
            if "refl" in d and "REFL_10CM" in r:
                vol = r["REFL_10CM"]
                want = np.full((ny, nx), np.nan, np.float32)
                for lev in vol:
                    for c in cells:
                        x = lev[c]
                        if math.isfinite(x):
                            want[c] = x if math.isnan(want[c]) else max(want[c], x)
                check(f"refl m{member} {valid}", d["refl"], want, failures, counts)
            if "visibility" in d:
                species = [r.get("S_qc"), r.get("S_qr"), r.get("S_qi"), r.get("S_qs"), r.get("S_qg")]
                base = r["S_thb"]
                want = np.array([visibility(r["S_p"][:, c[0], c[1]], r["S_thp"][:, c[0], c[1]],
                                            base if base.ndim == 1 else base[:, c[0], c[1]],
                                            r["S_qv"][:, c[0], c[1]],
                                            [None if s is None else s[:, c[0], c[1]] for s in species], 1)
                                 for c in cells], np.float32).reshape(ny, nx)
                check(f"visibility m{member} {valid}", d["visibility"], want, failures, counts)
            if "RAINNC" in r:
                cum = r["RAINNC"] + (r["RAINC"] if "RAINC" in r else F32(0))
                cum = cum + (r["RAINSH"] if "RAINSH" in r else F32(0))
                if state is None:
                    state = {"initial": cum.copy(), "ice": None, "tick": None}
                if "rain_total" in d:
                    check(f"rain_total m{member} {valid}", d["rain_total"],
                          np.fmax(F32(0), cum - state["initial"]), failures, counts)
                if "ice_accretion" in d:
                    tick = frames.index(valid)
                    prev = state["ice"]
                    hours = 0.0 if prev is None else (tick - prev["tick"]) / 3600.0
                    want = [ice(cum[c], cum[c] if prev is None else prev["rain"][c],
                                [F32(0)] * 3 if prev is None else [prev["f"][k][c] for k in range(3)],
                                r["T2"][c], r["T2"][c], r["Q2"][c], r["PSFC"][c], r["U10"][c],
                                r["V10"][c], r["QRAIN"][c], r["QSNOW"][c], r["QGRAUP"][c] if "QGRAUP" in r else F32(0),
                                hours, False) for c in cells]
                    names = ("freezing_rain_total", "ice_accretion_horizontal", "ice_accretion")
                    got = [np.array([w[k] for w in want], np.float32).reshape(ny, nx) for k in range(3)]
                    for k, name in enumerate(names):
                        check(f"{name} m{member} {valid}", d[name], got[k], failures, counts)
                    state["ice"] = {"tick": tick, "rain": cum.copy(), "f": got}
    # 2. the published product against the kernel
    for valid in frames:
        stacks = {f: np.stack([diag[(m, valid)][f] for m in members]) for f in diag[(members[0], valid)]}
        path = run / args.domain / "ensemble" / "products" / f"ensemble_{valid}.nc"
        ds = open_dataset(str(path))
        for field, stack in stacks.items():
            m, ny, nx = stack.shape
            thresholds = np.asarray(ds.variables[f"{field}_thresholds"][:], np.float32) if f"{field}_thresholds" in ds.variables else np.zeros(0, np.float32)
            relation = COMPARISON.get(field, "ge")
            want = [[reduce_cell(stack[:, j, i].tolist(), thresholds.tolist(), relation)
                     for i in range(nx)] for j in range(ny)]
            planes = {"mean": 0, "spread": 1, "min": 2, "max": 3}
            for name, k in planes.items():
                check(f"{field}_{name} {valid}", np.asarray(ds.variables[f"{field}_{name}"][:]),
                      np.array([[w[k] for w in row] for row in want], np.float32), failures, counts)
            check(f"{field}_finite_count {valid}", np.asarray(ds.variables[f"{field}_finite_count"][:]),
                  np.array([[w[4] for w in row] for row in want], np.uint32), failures, counts)
            for t in range(len(thresholds)):
                check(f"{field}_probability[{t}] {valid}", np.asarray(ds.variables[f"{field}_probability"][:])[t],
                      np.array([[w[5][t] for w in row] for row in want], np.float32), failures, counts)
                words = np.asarray(ds.variables[f"{field}_paintball"][:])[t, 0]
                check(f"{field}_paintball[{t}] {valid}", words,
                      np.array([[w[6][t] for w in row] for row in want], np.uint64), failures, counts)
                crossings = np.asarray(ds.variables[f"{field}_spaghetti"][:])[t]
                code = np.zeros((m, ny - 1, nx - 1), np.uint8)
                for mm in range(m):
                    for j in range(ny - 1):
                        for i in range(nx - 1):
                            code[mm, j, i] = spaghetti_code(stack[mm, j, i], stack[mm, j, i + 1],
                                                            stack[mm, j + 1, i + 1], stack[mm, j + 1, i],
                                                            thresholds[t], relation)
                check(f"{field}_spaghetti[{t}] {valid}", crossings, code, failures, counts)
        print(f"{path.name}: {len(stacks)} fields: {sorted(stacks)}")
        print(f"  {len(ds.variables)} variables")
    # 3. the joint-event field against _COMPOUND_SOURCE
    from woof.ensemble.batch_products import ThresholdCondition
    from woof.ensemble.metal_output import compound_event
    rng = np.random.default_rng(7)
    planes = {"wind10": rng.uniform(0, 30, (7, 9)).astype(F32), "temperature2": rng.uniform(260, 310, (7, 9)).astype(F32)}
    planes["wind10"][0, 0] = np.nan
    conditions = [ThresholdCondition("wind10", "m s-1", 15.0, "gt"), ThresholdCondition("temperature2", "K", 290.0, "le")]
    want = np.array([[compound_cell([planes[c.field][j, i] for c in conditions], [c.threshold for c in conditions],
                                    [("ge", "gt", "le", "lt").index(c.comparison) for c in conditions])
                      for i in range(9)] for j in range(7)], np.float32)
    check("compound_event", compound_event(planes, conditions), want, failures, counts)
    print(f"compared {counts[0]} words; {counts[1]} differ; failing checks: {failures or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
