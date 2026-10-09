"""Measure the resident ensemble on a card: bytes per member, wall per
member-step, and the analysis wall at a real station density.

``python -m woof.globe.da.measure --config <run toml> --truncation
127 --members 8 --steps 3 --outdir <dir> [--obs <iem-asos csv> --obs-hour
2026-08-31T12:00:00Z]``

The config is re-cut at ``--truncation`` with :func:`ensemble_config`
(the same physics, statics and analysis source); the model and its cold
state are built once; the ensemble is built around the cold state; the
pool's live bytes are read before and after the members exist (the
difference over N is the resident cost of one member, the allocator's
own reading beside the arrays' nbytes); ``--steps`` ensemble steps are
timed per member with the device synchronised; with ``--obs`` the table's
rows at ``--obs-hour`` are replaced by H(member 0) plus noise (synthetic
values at the REAL positions) and one analysis is timed by phase.  The
receipt is ``measure.json`` in ``--outdir``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from ..config import load_config
from ..obs_table import ObsRow, load_obs, parse_valid_time
from ..runner import build_model_and_cold_state, build_transform
from .analysis import analyze_ensemble
from .ensemble import GlobalEnsemble, ensemble_config, state_bytes
from .operators import MemberOperators, batches_from_rows
from .options import EnsembleOptions, FilterOptions

GIB = 2 ** 30


def _pool_used(transform) -> int | None:
    backend = transform.backend
    if getattr(backend, "name", "numpy") != "cupy":
        return None
    xp = backend.xp
    xp.cuda.runtime.deviceSynchronize()
    return int(xp.get_default_memory_pool().used_bytes())


def _synthetic_rows_at_real_positions(path: str, hour: dt.datetime, operators, truth, rng, window_s: float = 1800.0):
    _source, rows, _prov = load_obs(str(path))
    chosen = [r for r in rows if abs((r.valid_time - hour).total_seconds()) <= window_s]
    if not chosen:
        raise ValueError(f"no rows of {path} within {window_s:g} s of {hour.isoformat()}")
    # One row per (station, variable): the latest.
    latest: dict[tuple, ObsRow] = {}
    for r in chosen:
        key = (r.station_id, r.variable, r.level_pa)
        if key not in latest or r.valid_time > latest[key].valid_time:
            latest[key] = r
    chosen = list(latest.values())
    lat = np.array([r.latitude_deg for r in chosen])
    lon = np.array([r.longitude_deg for r in chosen])
    elev = np.array([r.elevation_m for r in chosen])
    level = np.array([np.nan if r.level_pa is None else r.level_pa for r in chosen])
    values, _ = operators.evaluate([truth], lat, lon, elev, level)
    out = []
    for k, r in enumerate(chosen):
        v = values.get(r.variable)
        if v is None or not np.isfinite(v[0, k]):
            continue
        out.append(ObsRow(r.source, r.station_id, r.latitude_deg, r.longitude_deg, r.elevation_m,
                          r.level_pa, hour, r.variable, float(v[0, k] + rng.normal(0.0, r.error)), r.error))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m woof.globe.da.measure")
    p.add_argument("--config", required=True)
    p.add_argument("--truncation", type=int, required=True)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--dt-s", type=float, default=None)
    p.add_argument("--outdir", required=True)
    p.add_argument("--obs", action="append", default=[])
    p.add_argument("--obs-hour", default=None)
    p.add_argument("--horizontal-cutoff-km", type=float, default=1200.0)
    p.add_argument("--max-local-obs", type=int, default=400)
    p.add_argument("--seed", type=int, default=20260906)
    a = p.parse_args(argv)
    output = Path(a.outdir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config)
    options = EnsembleOptions(members=int(a.members), truncation=int(a.truncation), seed=int(a.seed))
    ecfg = ensemble_config(cfg, options, dt_s=a.dt_s)
    record: dict[str, object] = {
        "config": str(a.config), "truncation": int(a.truncation), "nlev": int(ecfg.vertical.nlev),
        "members": int(a.members), "dt_s": float(ecfg.dt_s), "backend": ecfg.backend,
        "precision": ecfg.precision, "physics_mode": ecfg.physics_mode,
    }
    t0 = time.perf_counter()
    transform = build_transform(ecfg)
    model, cold = build_model_and_cold_state(ecfg, transform, scratch_destination=output)
    record["model_build_s"] = time.perf_counter() - t0
    used_before = _pool_used(transform)
    record["pool_used_before_members_bytes"] = used_before
    t0 = time.perf_counter()
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    record["ensemble_build_s"] = time.perf_counter() - t0
    del cold
    used_after = _pool_used(transform)
    record["pool_used_with_members_bytes"] = used_after
    record["member_arrays"] = state_bytes(ensemble.members[0])
    if used_before is not None:
        record["pool_per_member_bytes"] = (used_after - used_before) / max(1, a.members)
        record["pool_per_member_gib"] = record["pool_per_member_bytes"] / GIB
        record["shared_gib"] = used_before / GIB
    timings = []
    for _ in range(int(a.steps)):
        timing = ensemble.step_all(ecfg.dt_s)
        timings.append(timing.as_record())
    record["steps"] = timings
    if timings:
        record["wall_per_member_step_s_mean_after_first"] = float(np.mean(
            [t["mean_per_member_s"] for t in timings[1:]] or [timings[0]["mean_per_member_s"]]))
    used_after_steps = _pool_used(transform)
    record["pool_used_after_steps_bytes"] = used_after_steps
    if used_after_steps is not None:
        pool = transform.backend.xp.get_default_memory_pool()
        record["pool_total_after_steps_bytes"] = int(pool.total_bytes())
    record["resident_bytes"] = ensemble.resident_bytes()
    if a.obs:
        hour = parse_valid_time(a.obs_hour) if a.obs_hour else None
        if hour is None:
            raise ValueError("--obs needs --obs-hour")
        operators = MemberOperators.for_model(model, transform, ecfg)
        rng = np.random.default_rng(a.seed + 1)
        rows = []
        for path in a.obs:
            rows.extend(_synthetic_rows_at_real_positions(path, hour, operators, ensemble.members[0], rng))
        t0 = time.perf_counter()
        batches = batches_from_rows(rows, operators, ensemble.members)
        record["operators_background_s"] = time.perf_counter() - t0
        record["reports"] = len(rows)
        record["batches"] = [(b.stream, b.variable, b.count) for b in batches]
        filter_options = FilterOptions(horizontal_cutoff_km=a.horizontal_cutoff_km, max_local_obs=a.max_local_obs)
        t0 = time.perf_counter()
        result = analyze_ensemble(ensemble, batches, filter_options, analysis_time=hour)
        record["analysis_wall_s"] = time.perf_counter() - t0
        record["analysis_timings_s"] = result.timings_s
        record["analysis_status"] = result.status
        record["analysis_letkf"] = result.report["letkf"]
        record["analysis_rejections"] = result.report["rejections"]
        record["analysis_streams"] = {
            s: {v: {"count": e["count"], "verdict": e["verdict"],
                    "o_minus_b": e["regions"]["global"]["assimilated"]["o_minus_b"]["rms"],
                    "o_minus_a": (e["regions"]["global"]["assimilated"]["o_minus_a"] or {}).get("rms")}
                for v, e in vs.items()}
            for s, vs in result.report["streams"].items()}
        record["pool_used_after_analysis_bytes"] = _pool_used(transform)
    (output / "measure.json").write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k not in ("analysis_letkf", "analysis_streams", "batches")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
