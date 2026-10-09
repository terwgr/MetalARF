"""Negative-ringing mass fraction per water species and level in checkpoints.

Every water species of the truncated spectral state rings below zero in
grid space.  The positivity repair (dynamics._repair_positivity) and the
physics-exchange clamp (dynamics._physics_exchange_with_closure) clip the
ringing and then rescale the level's positive part by
``scale = mass_before / mass_after`` (dynamics._fill_water_holes), where
mass_before is the pre-clip level integral (pos - neg) and mass_after the
clipped level integral (pos).  ``1 - scale = neg / pos`` is therefore the
fraction of a level's positive water the rescale moves out of the columns
that hold it and into the clipped lobes on every pass, and there are four
passes per model step.  This reads neg / pos per species and level from a
checkpoint, in float64 on the run's own grid, plus the distribution of the
positive mass by mixing-ratio magnitude (how much of it sits below the
Morrison kernel's 1e-8 kg/kg instant-evaporation threshold).

Usage:
  python tools/arwen_global_ringing_fraction.py RUN_DIR CKPT [CKPT ...] --out JSON
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from woof.globe.checkpoint import read_checkpoint
from woof.globe.constants import GRAVITY_M_S2, WATER_SPECIES
from woof.globe.water_budget import CheckpointReader, read_receipt

BINS = ((0.0, 1.0e-8), (1.0e-8, 1.0e-6), (1.0e-6, 1.0e-5), (1.0e-5, np.inf))


def analyse(reader: CheckpointReader, path: Path) -> dict:
    metadata, arrays = read_checkpoint(path)
    transform = reader.transform
    logps = transform.inverse(arrays["atmosphere__log_surface_pressure"].astype(np.complex128))
    ps = np.exp(logps)
    dp = reader.vertical.pressure(ps, transform.backend)["dp"]
    nlat, nlon = ps.shape
    w = np.asarray(transform.grid.quadrature_weights, dtype=np.float64)
    cell = (w / (2.0 * nlon))[:, None]
    dp_g_w = dp / GRAVITY_M_S2 * cell[None]
    out = {"step": int(metadata["step"]), "time_s": float(metadata["time_s"]), "species": {}}
    for species in WATER_SPECIES:
        q = reader._grid_field(arrays[f"atmosphere__{species}"])
        mass = q * dp_g_w
        pos = np.sum(np.maximum(mass, 0.0), axis=(1, 2))
        neg = -np.sum(np.minimum(mass, 0.0), axis=(1, 2))
        ratio = np.where(pos > 0.0, neg / np.where(pos > 0.0, pos, 1.0), 0.0)
        bins = []
        for lo, hi in BINS:
            sel = (q > lo) & (q <= hi)
            bins.append(float(np.sum(np.where(sel, mass, 0.0))))
        cells_positive = int(np.sum(q > 0.0))
        cells_above_1e5 = int(np.sum(q > 1.0e-5))
        total_pos = float(np.sum(pos))
        total_neg = float(np.sum(neg))
        out["species"][species] = {
            "column_positive_kg_m2": total_pos,
            "column_negative_kg_m2": total_neg,
            "column_neg_over_pos": (total_neg / total_pos) if total_pos > 0 else 0.0,
            "level_positive_kg_m2": [float(v) for v in pos],
            "level_negative_kg_m2": [float(v) for v in neg],
            "level_neg_over_pos": [float(v) for v in ratio],
            "positive_mass_by_q_bin": {f"{lo:g}..{hi:g}": b for (lo, hi), b in zip(BINS, bins)},
            "cells_positive": cells_positive,
            "cells_above_1e-5": cells_above_1e5,
            "grid_min": float(np.min(q)),
            "grid_max": float(np.max(q)),
        }
    return out


def report(result: dict) -> str:
    lines = [f"step {result['step']} t={result['time_s']/3600:.1f} h"]
    lines.append(f"  {'species':8s} {'pos kg/m2':>11s} {'neg kg/m2':>11s} {'neg/pos':>8s} {'worst level (ratio, pos)':>28s}  "
                 f"{'mass<1e-8':>10s} {'1e-8..1e-6':>10s} {'1e-6..1e-5':>10s} {'>1e-5':>10s}  cells>0  cells>1e-5  min")
    for species, s in result["species"].items():
        ratio = np.asarray(s["level_neg_over_pos"]); pos = np.asarray(s["level_positive_kg_m2"])
        k = int(np.argmax(ratio))
        b = s["positive_mass_by_q_bin"]
        vals = list(b.values())
        lines.append(
            f"  {species:8s} {s['column_positive_kg_m2']:11.5f} {s['column_negative_kg_m2']:11.5f} "
            f"{s['column_neg_over_pos']:8.4f} {f'k={k} ({ratio[k]:.3f}, {pos[k]:.2e})':>28s}  "
            f"{vals[0]:10.2e} {vals[1]:10.2e} {vals[2]:10.2e} {vals[3]:10.2e}  {s['cells_positive']:7d}  {s['cells_above_1e-5']:10d}  {s['grid_min']:.2e}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = read_receipt(args.run_dir)
    reader = CheckpointReader(receipt, transport=False)
    results = []
    for path in args.checkpoints:
        started = time.perf_counter()
        result = analyse(reader, path)
        result["path"] = str(path)
        result["wall_s"] = time.perf_counter() - started
        results.append(result)
        print(report(result), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"run_dir": str(args.run_dir), "results": results}, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
