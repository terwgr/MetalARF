"""Power by total wavenumber of two WOOF global checkpoints, banded, and the
ratio between them: the reading the observation scorecard cannot take.

The scorecard reads the largest scales of the flow at stations and sounding
sites.  Two integrators can agree there and differ by a factor of six in the
vorticity power above n = 180 (the grade refutation of 2026-09-06 measured
0.17 at 500 hPa), so a grade of a core carries this table beside the rows.

The truncation is read from each checkpoint's own spectral arrays, so a T383
arm can be set beside a T255 arm: the comparison is over the degrees both
carry (0..min(T)), and the sharper arm's power above that is reported on its
own so it is not mistaken for a loss.  The level is chosen by pressure from
the receipt's vertical description (the full level nearest the requested
pressure at the reference surface pressure), and named in the output.

usage:
  python tools/arwen_global_spectrum_bands.py --candidate RUN_DIR/ckpt.npz \
      --reference RUN_DIR/ckpt.npz [--levels-hpa 500,250] [--out-json S.json] [--out-md S.md]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

BANDS = ((1, 20), (21, 60), (61, 120), (121, 180), (181, 230), (231, 255),
         (256, 383), (384, 533))


def _receipt_for(checkpoint: Path) -> dict | None:
    r = checkpoint.parent / "arwen-global-receipt.json"
    if r.exists():
        return json.loads(r.read_text(encoding="utf-8"))
    return None


def _level_pressures_pa(receipt: dict | None, nlev: int) -> np.ndarray | None:
    """Full-level reference pressures at the receipt's reference surface
    pressure, from the half-level coefficients the receipt carries.

    The coefficients live under the receipt's ``config`` (``a_half_pa``,
    ``b_half``, the parsed ArwenGlobalConfig); the ``vertical`` block carries
    the summary (``reference_surface_pa``, ``nlev``) and not the arrays.
    Until 2026-09-07 this looked for the arrays under ``vertical`` only,
    never found them, and the caller fell back to hard-coded levels 26 and
    16 as "500" and "250 hPa", which on the 40-level surface-stretched stack
    of record are 793 and 295 hPa: every banded ratio the equal-cost grade
    published under those labels was read there."""
    if not receipt:
        return None
    v = receipt.get("vertical") or {}
    c = receipt.get("config") or {}
    a = c.get("a_half_pa", v.get("a_half_pa"))
    b = c.get("b_half", v.get("b_half"))
    ps = v.get("reference_surface_pa") or c.get("reference_surface_pa") or 101325.0
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    half = a + b * float(ps)
    full = 0.5 * (half[:-1] + half[1:])
    if full.shape[0] != nlev:
        return None
    return full


def power_by_degree(coeff2d: np.ndarray) -> np.ndarray:
    """Sum |a_nm|^2 over orders m per total degree n, from the packed (m, n)
    layout the transform uses, weighting m > 0 twice for the conjugate."""
    a = np.asarray(coeff2d)
    nm, nn = a.shape
    T = nn - 1
    p = np.zeros(T + 1)
    mags = np.abs(a) ** 2
    for m in range(min(nm, T + 1)):
        w = 1.0 if m == 0 else 2.0
        p[m:T + 1] += w * mags[m, m:T + 1]
    return p


def load(path: Path, name: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as f:
        return np.array(f[name])


def spectrum(path: Path, levels_hpa: list[float]) -> dict:
    vort = load(path, "atmosphere__vorticity")
    lnps = load(path, "atmosphere__log_surface_pressure")
    nlev = vort.shape[0]
    T = vort.shape[-1] - 1
    receipt = _receipt_for(path)
    pres = _level_pressures_pa(receipt, nlev)
    out = {"checkpoint": str(path), "truncation": T, "nlev": nlev, "levels": {}}
    for hpa in levels_hpa:
        if pres is not None:
            k = int(np.argmin(np.abs(pres - hpa * 100.0)))
            out["levels"][str(hpa)] = {"index": k, "reference_pressure_hpa": float(pres[k] / 100.0),
                                       "power": power_by_degree(vort[k]).tolist()}
        else:
            # No guessed level: the fallback this tool carried until
            # 2026-09-07 (levels 26 and 16 for 500 and 250 hPa) read 793 and
            # 295 hPa on the 40-level surface-stretched stack of record, and
            # the ratios it published under the wrong labels disagreed with
            # the pressure-surface instrument (tools/semilag_scale_diagnostics.py)
            # by 1.6x to 2.5x on identical pairs.  A level has to come from
            # the receipt beside the checkpoint.
            raise SystemExit(
                f"no half-level coefficients found beside {path}: the receipt's config must "
                f"carry a_half_pa and b_half ({nlev} levels) for the level nearest {hpa:g} hPa "
                "to be chosen; this tool no longer guesses a level"
            )
    out["surface_pressure_power"] = power_by_degree(lnps).tolist()
    return out


def banded(power: list[float], bands=BANDS) -> dict[str, float]:
    p = np.asarray(power)
    T = p.shape[0] - 1
    rows = {}
    for lo, hi in bands:
        if lo > T:
            continue
        rows[f"n {lo}-{min(hi, T)}"] = float(p[lo:min(hi, T) + 1].sum())
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--levels-hpa", default="500,250")
    ap.add_argument("--label-candidate", default="candidate")
    ap.add_argument("--label-reference", default="reference")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()
    levels = [float(x) for x in args.levels_hpa.split(",") if x.strip()]
    cand = spectrum(args.candidate, levels)
    ref = spectrum(args.reference, levels)
    common = min(cand["truncation"], ref["truncation"])
    md = [f"candidate {args.label_candidate}: T{cand['truncation']} `{args.candidate}`",
          f"reference {args.label_reference}: T{ref['truncation']} `{args.reference}`",
          f"bands compared over the degrees both carry (n <= {common}); a sharper arm's power above that is listed on its own", ""]
    result = {"candidate": cand["checkpoint"], "reference": ref["checkpoint"],
              "candidate_truncation": cand["truncation"], "reference_truncation": ref["truncation"],
              "common_truncation": common, "levels": {}, "surface_pressure": {}}
    for hpa in levels:
        key = str(hpa)
        pc = cand["levels"][key]["power"]
        pr = ref["levels"][key]["power"]
        bc, br = banded(pc), banded(pr)
        rows = {}
        md.append(f"vorticity power at {hpa:g} hPa (candidate level {cand['levels'][key]['index']}"
                  f"{'' if cand['levels'][key]['reference_pressure_hpa'] is None else ' at %.0f hPa' % cand['levels'][key]['reference_pressure_hpa']}, "
                  f"reference level {ref['levels'][key]['index']})")
        md.append("| band | reference | candidate | ratio candidate/reference |")
        md.append("|---|---|---|---|")
        for band in bc:
            lo = int(band.split()[1].split("-")[0])
            if lo > common:
                rows[band] = {"candidate": bc[band], "reference": None, "ratio": None}
                md.append(f"| {band} | (beyond the reference's truncation) | {bc[band]:.6e} | |")
            elif band in br and br[band] > 0.0:
                rows[band] = {"candidate": bc[band], "reference": br[band], "ratio": bc[band] / br[band]}
                md.append(f"| {band} | {br[band]:.6e} | {bc[band]:.6e} | {bc[band] / br[band]:.4f} |")
        # the whole-spectrum reading up to the common truncation
        tot_c = float(np.asarray(pc)[1:common + 1].sum())
        tot_r = float(np.asarray(pr)[1:common + 1].sum())
        rows["n 1-%d" % common] = {"candidate": tot_c, "reference": tot_r, "ratio": tot_c / tot_r}
        md.append(f"| all n 1-{common} | {tot_r:.6e} | {tot_c:.6e} | {tot_c / tot_r:.4f} |")
        md.append("")
        result["levels"][key] = {"candidate_level": cand["levels"][key]["index"],
                                 "reference_level": ref["levels"][key]["index"],
                                 "bands": rows,
                                 "candidate_power_by_degree": pc, "reference_power_by_degree": pr}
    bc, br = banded(cand["surface_pressure_power"]), banded(ref["surface_pressure_power"])
    md.append("log surface pressure power")
    md.append("| band | reference | candidate | ratio |")
    md.append("|---|---|---|---|")
    for band in bc:
        lo = int(band.split()[1].split("-")[0])
        if lo <= common and band in br and br[band] > 0.0:
            result["surface_pressure"][band] = {"candidate": bc[band], "reference": br[band], "ratio": bc[band] / br[band]}
            md.append(f"| {band} | {br[band]:.6e} | {bc[band]:.6e} | {bc[band] / br[band]:.4f} |")
    text = "\n".join(md) + "\n"
    print(text)
    if args.out_md:
        args.out_md.write_text(text, encoding="utf-8")
    if args.out_json:
        args.out_json.write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
