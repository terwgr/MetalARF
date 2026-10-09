"""The shipped step of the Eulerian core, chosen from the strongest jet on disk.

The runner refuses an Eulerian step whose spectral CFL, dt |V|max sqrt(N(N+1))/a,
exceeds ``[time] maximum_cfl`` (0.75).  A shipped step is chosen against that
refusal with a margin: the largest whole step that keeps EVERY analysis day on
disk under a fraction of the gate (0.70), so the default survives a stronger day
than the one it was tuned on and a refusal on a still stronger one is explained
by the receipt's headroom block rather than by a traceback.

Two verbs.

``plan``   writes one measurement config per (analysis day, truncation) from a
           template config, holding the suite and every bucket and moving only
           the analysis, its start time, the truncation, the step and the
           grid-spacing-dependent settings; the step is short enough that the
           measurement is not itself refused (the gate is set to 1.0 for the
           measurement, which the runner admits, and the receipt's maximum is
           what is read).

``table``  reads the receipts those runs wrote (``run_trackers.maximum_spectral_cfl``
           at the run's own ``dt_s`` and truncation), turns each into a rate per
           second of step and an implied maximum wind, and derives, per
           truncation, the largest admissible step under the rule from the
           strongest day.  A truncation measured on fewer days than the
           anchor (the truncation with the most days) is set against the
           anchor's strongest day through the ratio measured on the days both
           share (or sqrt(N(N+1)) with no shared day) and says so in its row.

usage:
  python tools/arwen_global_cfl_sweep.py plan --template configs/verify/X.toml \
      --out-dir DIR --truncation 255 --dt 100 \
      --case 2026-09-01T00=cases/baseline-2026090100/gdas.t00z.pgrb2.0p25.f000 ...
  python tools/arwen_global_cfl_sweep.py table --rule 0.70 --gate 0.75 \
      --receipt 2026-09-01T00@255=runs/cfl-2026-09-01T00-T255/arwen-global-receipt.json ... \
      --truncations 255,383,533 --out-json T.json --out-md T.md
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

EARTH_RADIUS_M = 6371220.0

#: Candidate whole steps: multiples of five seconds that divide the hour, so
#: the six-hour output cadence, the hourly diagnostics and the radiation
#: bucket all land on whole steps.
CANDIDATE_STEPS_S = tuple(
    s for s in range(5, 3601, 5) if 3600 % s == 0
)

#: Grid-spacing-dependent settings per truncation at dealias 1.5: the
#: equatorial spacing 2 pi a / nlon and the RRTMGP column chunk the tree's
#: configs of record use at that truncation.
GRID_SETTINGS = {
    255: {"dx_m": 52100.0, "radiation_column_chunk": 12500},
    383: {"dx_m": 34700.0, "radiation_column_chunk": 12500},
    533: {"dx_m": 25000.0, "radiation_column_chunk": 5000},
    799: {"dx_m": 16700.0, "radiation_column_chunk": 5000},
}


def spectral_factor(truncation: int) -> float:
    """sqrt(N(N+1)) / a: the CFL per metre per second of wind per second of step."""
    return math.sqrt(truncation * (truncation + 1)) / EARTH_RADIUS_M


def radiation_bucket_s(dx_m: float, dt_s: float) -> float:
    """The whole multiple of the step nearest the one-minute-per-km convention."""
    target = dx_m / 1000.0 * 60.0
    return max(1.0, round(target / dt_s)) * dt_s


def _sub(text: str, pattern: str, replacement: str, *, count: int = 1) -> str:
    new, n = re.subn(pattern, replacement, text, count=count, flags=re.M)
    if n == 0:
        raise SystemExit(f"template has no line matching {pattern!r}")
    return new


def plan(args) -> int:
    template = Path(args.template).read_text(encoding="utf-8")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    truncation = int(args.truncation)
    dt = float(args.dt)
    grid = GRID_SETTINGS.get(truncation)
    if grid is None:
        raise SystemExit(f"no grid settings for T{truncation}; add them to GRID_SETTINGS")
    rad = radiation_bucket_s(grid["dx_m"], dt)
    written = []
    for spec in args.case:
        label, path = spec.split("=", 1)
        start = args.start or (label + ":00:00Z")
        # label form 2026-09-01T00 -> start 2026-09-01T00:00:00Z unless given
        if "@" in path:
            path, start = path.split("@", 1)
        text = template
        text = _sub(text, r'^name = ".*"$', f'name = "arwen-global-cfl-{label}-T{truncation}"')
        text = _sub(text, r"^truncation = \d+$", f"truncation = {truncation}")
        text = _sub(text, r"^dt_s = [0-9.]+$", f"dt_s = {dt}")
        text = _sub(text, r"^output_interval_s = [0-9.]+$", f"output_interval_s = {float(args.output_interval_s)}")
        text = _sub(text, r"^duration_s = [0-9.]+$", f"duration_s = {float(args.duration_s)}")
        text = _sub(text, r"^maximum_cfl = [0-9.]+$", f"maximum_cfl = {float(args.measure_gate)}")
        text = _sub(text, r'^analysis_grib = ".*"$', f'analysis_grib = "{path}"')
        text = _sub(text, r'start_time_utc = "[^"]*"', f'start_time_utc = "{start}"')
        text = _sub(text, r"radiation_interval_s = [0-9.]+", f"radiation_interval_s = {rad}")
        text = _sub(text, r"radiation_column_chunk = \d+", f"radiation_column_chunk = {grid['radiation_column_chunk']}")
        text = _sub(text, r"land_surface_interval_s = [0-9.]+", f"land_surface_interval_s = {dt}")
        text = _sub(text, r"dx_m = [0-9.]+", f"dx_m = {grid['dx_m']}")
        head = (
            f"# CFL measurement: analysis {label} at T{truncation}, dt {dt:g} s, the gate\n"
            f"# raised to {float(args.measure_gate):g} so the day is MEASURED and not refused; the\n"
            f"# receipt's run_trackers.maximum_spectral_cfl is what tools/arwen_global_cfl_sweep.py\n"
            f"# reads.  Written by that tool from {Path(args.template).name}.\n"
        )
        out = out_dir / f"cfl-{label}-T{truncation}.toml"
        out.write_text(head + text, encoding="utf-8")
        written.append(out)
        print(out)
    return 0 if written else 1


def _read_receipt(path: Path) -> dict:
    r = json.loads(path.read_text(encoding="utf-8"))
    cfg = r.get("config") or {}
    return {
        "status": r.get("status"),
        "dt_s": float(cfg.get("dt_s")),
        "truncation": int(cfg.get("truncation")),
        "integrator": cfg.get("integrator"),
        "maximum_spectral_cfl": float(r["run_trackers"]["maximum_spectral_cfl"]),
        "completed_time_s": float(
            r.get("completed_time_s", (r.get("final_diagnostics") or {}).get("time_s", float("nan")))
        ),
        "maximum_wind_m_s_final": float((r.get("final_diagnostics") or {}).get("maximum_wind_m_s", float("nan"))),
        "maximum_wind_m_s_cold": float((r.get("cold_start_diagnostics") or {}).get("maximum_wind_m_s", float("nan"))),
        "config_hash": r.get("config_hash"),
        "wall_seconds": r.get("wall_seconds"),
        "receipt": str(path),
    }


def largest_step(rate_per_s: float, bound: float) -> int | None:
    """The largest candidate step with dt * rate <= bound."""
    admitted = [s for s in CANDIDATE_STEPS_S if s * rate_per_s <= bound]
    return max(admitted) if admitted else None


def table(args) -> int:
    rule = float(args.rule)
    gate = float(args.gate)
    truncations = [int(t) for t in str(args.truncations).split(",") if t.strip()]
    rows = []
    for spec in args.receipt:
        key, path = spec.split("=", 1)
        label, _, trunc = key.partition("@")
        rec = _read_receipt(Path(path))
        if trunc and int(trunc) != rec["truncation"]:
            raise SystemExit(f"{key}: receipt is T{rec['truncation']}, not T{trunc}")
        rate = rec["maximum_spectral_cfl"] / rec["dt_s"]
        rows.append({
            "day": label,
            "truncation": rec["truncation"],
            "measured_dt_s": rec["dt_s"],
            "integrator": rec["integrator"],
            "status": rec["status"],
            "completed_time_s": rec["completed_time_s"],
            "maximum_spectral_cfl": rec["maximum_spectral_cfl"],
            "rate_per_s": rate,
            "implied_maximum_wind_m_s": rate / spectral_factor(rec["truncation"]),
            "cold_start_maximum_wind_m_s": rec["maximum_wind_m_s_cold"],
            "receipt": rec["receipt"],
        })
    if not rows:
        raise SystemExit("no receipts")
    measured = sorted({r["truncation"] for r in rows})
    # the truncation with the most measured days anchors every other one:
    # a truncation probed on fewer days is set against the anchor's
    # strongest day through the ratio MEASURED on the days both share (the
    # same analysis wind read on the sharper grid), never against its own
    # few days alone; with no shared day the ratio falls back to
    # sqrt(N(N+1)), the same wind on a sharper grid, and the row says so.
    by_truncation = {m: {r["day"]: r["rate_per_s"] for r in rows if r["truncation"] == m} for m in measured}
    anchor = max(measured, key=lambda m: len(by_truncation[m]))
    anchor_days = by_truncation[anchor]
    per_truncation = {}
    for N in truncations:
        own = by_truncation.get(N, {})
        if N == anchor or len(own) >= len(anchor_days):
            worst_day = max(own, key=own.get)
            rate = own[worst_day]
            basis = {"kind": "measured", "days": len(own), "worst_day": worst_day}
            per_day = dict(own)
        else:
            shared = sorted(set(own) & set(anchor_days))
            if shared:
                ratios = {d: own[d] / anchor_days[d] for d in shared}
                scale = max(ratios.values())
                kind = "measured ratio"
            else:
                ratios = {}
                scale = spectral_factor(N) / spectral_factor(anchor)
                kind = "sqrt(N(N+1)) ratio"
            per_day = {d: v * scale for d, v in anchor_days.items()}
            for d, v in own.items():
                per_day[d] = max(per_day.get(d, 0.0), v)  # a measured day is never scaled below itself
            worst_day = max(per_day, key=per_day.get)
            rate = per_day[worst_day]
            basis = {"kind": "derived", "from_truncation": anchor, "scale": scale,
                     "ratio_kind": kind, "shared_days": shared, "ratios_on_shared_days": ratios,
                     "own_days": len(own), "days": len(per_day), "worst_day": worst_day}
        default = largest_step(rate, rule * gate)
        at_gate = largest_step(rate, gate)
        per_truncation[str(N)] = {
            "rule_fraction_of_gate": rule,
            "gate": gate,
            "worst_rate_per_s": rate,
            "worst_implied_wind_m_s": rate / spectral_factor(N),
            "basis": basis,
            "default_step_s": default,
            "default_step_cfl_worst_day": None if default is None else default * rate,
            "largest_step_at_gate_s": at_gate,
            "cfl_per_day_at_default": {} if default is None else {
                d: default * v for d, v in sorted(per_day.items())},
        }
    out = {
        "schema": "gpuwm.arwen-global-cfl-sweep/v1",
        "rule": {"fraction_of_gate": rule, "gate": gate,
                 "candidate_steps_s": "multiples of 5 s dividing 3600 s",
                 "statement": "the default is the largest candidate step whose spectral CFL on the strongest analysis day on disk stays at or under rule x gate"},
        "days": rows,
        "per_truncation": per_truncation,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = []
    md.append(f"| day | T | measured dt (s) | max spectral CFL | rate per s of step | implied max wind (m/s) | analysis max wind (m/s) | status |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["truncation"], r["day"])):
        md.append(f"| {r['day']} | {r['truncation']} | {r['measured_dt_s']:g} | {r['maximum_spectral_cfl']:.4f} | "
                  f"{r['rate_per_s']:.6f} | {r['implied_maximum_wind_m_s']:.1f} | {r['cold_start_maximum_wind_m_s']:.1f} | {r['status']} |")
    md.append("")
    md.append(f"| T | basis | strongest day | rate per s | implied wind (m/s) | default step (s) | CFL on the strongest day at it | largest step at the {gate:g} gate |")
    md.append("|---|---|---|---|---|---|---|---|")
    for N in truncations:
        p = per_truncation[str(N)]
        b = p["basis"]
        basis = ("measured over %d days" % b["days"]) if b["kind"] == "measured" else (
            "derived from T%d over %d days through the %s x%.4f (measured on %d of them)"
            % (b["from_truncation"], b["days"], b["ratio_kind"], b["scale"], b["own_days"]))
        md.append(f"| {N} | {basis} | {b['worst_day']} | {p['worst_rate_per_s']:.6f} | {p['worst_implied_wind_m_s']:.1f} | "
                  f"**{p['default_step_s']}** | {p['default_step_cfl_worst_day']:.4f} | {p['largest_step_at_gate_s']} |")
    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="verb", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--template", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--truncation", type=int, required=True)
    p.add_argument("--dt", type=float, required=True)
    p.add_argument("--duration-s", type=float, default=86400.0)
    p.add_argument("--output-interval-s", type=float, default=21600.0)
    p.add_argument("--measure-gate", type=float, default=1.0,
                   help="[time] maximum_cfl for the measurement run (default 1.0, the largest the door admits)")
    p.add_argument("--start", default=None, help="start_time_utc for every case (default: the label + ':00:00Z')")
    p.add_argument("--case", action="append", required=True,
                   help="LABEL=PATH[@START]; LABEL like 2026-09-01T00")
    p.set_defaults(func=plan)
    t = sub.add_parser("table")
    t.add_argument("--receipt", action="append", required=True, help="LABEL[@T]=receipt.json")
    t.add_argument("--truncations", default="255,383,533")
    t.add_argument("--rule", type=float, default=0.70)
    t.add_argument("--gate", type=float, default=0.75)
    t.add_argument("--out-json", required=True)
    t.add_argument("--out-md", required=True)
    t.set_defaults(func=table)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
