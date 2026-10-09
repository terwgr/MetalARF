"""The microwave leg's door: ``woof global microwave``.

Subcommands, each a stage with its own receipt:

* ``fetch``      one day of ATMS SDR granule pairs (manifest, SHA-256, per-hour volume, latency)
* ``decode``     the fetched day through ``rw_atms`` into flat arrays
* ``thin``       the decoded beams onto latitude rings x longitudes x time bins (``rw_atms thin``)
* ``columns``    GDAS pgrb2 analyses through the mapped engine into cached column sets
* ``score``      the O-B scorecard of the thinned cells against the columns, clear sky over ocean
* ``calibrate``  the two-direction synthetic calibration receipt

``score`` is the reading of record for the design's item 5b; it writes
``microwave-scorecard.json`` (per-channel bias and rmse before and after
the linear bias corrections, the noise floor, the screening counts, the
diagnostics by scan angle, latitude band, wind, precipitable water and
hour), the calibration receipt and the operator entry
(``microwave-operator-entry.json``, the admitted channels with their
errors and bias coefficients, or the refusal) beside it; its exit status
is 0 only when every sounding channel is inside the 1 K bar.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np


def _cmd_fetch(args) -> int:
    from .atms_fetch import _main as fetch_main

    argv = ["--satellite", args.satellite, "--day", args.day, "--out", args.out,
            "--workers", str(args.workers)]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    return fetch_main(argv)


def _cmd_decode(args) -> int:
    from .atms_bridge import decode
    from .atms_fetch import granule_pairs

    manifest = json.loads((Path(args.fetched) / "fetch-manifest.json").read_text(encoding="utf-8"))
    pairs = granule_pairs(manifest, args.fetched)
    if args.limit is not None:
        pairs = pairs[:args.limit]
    started = time.monotonic()
    decoded = decode(pairs, args.out)
    print(json.dumps({
        "granules": len(decoded.metadata["granules"]),
        "scan_count": decoded.metadata["scan_count"],
        "wall_seconds": round(time.monotonic() - started, 2),
        "outdir": str(args.out),
    }))
    return 0


def _latitudes(spec: str) -> np.ndarray:
    """``0.25`` for a regular grid from 90 to -90, or a file of ring centres."""
    try:
        step = float(spec)
    except ValueError:
        return np.loadtxt(spec, dtype=np.float64)
    n = int(round(180.0 / step)) + 1
    return 90.0 - step * np.arange(n)


def _cmd_thin(args) -> int:
    from .atms_bridge import thin

    rings = _latitudes(args.latitudes)
    origin = dt.datetime.fromisoformat(args.origin.replace("Z", "+00:00"))
    if origin.tzinfo is None:
        origin = origin.replace(tzinfo=dt.timezone.utc)
    started = time.monotonic()
    thinned = thin(
        args.decoded, args.out, latitudes_deg=rings, nlon=args.nlon,
        origin_unix_s=origin.timestamp(), bin_s=args.bin_s, max_zenith_deg=args.max_zenith,
    )
    print(json.dumps({
        "cells": thinned.ncell,
        "beams_placed": thinned.metadata["beams_placed"],
        "wall_seconds": round(time.monotonic() - started, 2),
        "outdir": str(args.out),
    }))
    return 0


def _cmd_columns(args) -> int:
    from .columns import decode_analysis, save_analysis

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for grib in args.grib:
        started = time.monotonic()
        analysis = decode_analysis(grib, scratch_destination=args.scratch)
        target = out / f"gdas-{analysis.valid_time:%Y-%m-%dT%H}.npz"
        save_analysis(analysis, target)
        written.append({"grib": str(grib), "valid_time": analysis.valid_time.isoformat(),
                        "npz": str(target), "wall_seconds": round(time.monotonic() - started, 1)})
        print(json.dumps(written[-1]), flush=True)
    return 0


def _cmd_calibrate(args) -> int:
    from . import calibrate

    receipt = calibrate.run()
    verdict = calibrate.passes(receipt)
    receipt["passes"] = verdict
    Path(args.out).write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps(verdict))
    return 0 if all(verdict.values()) else 1


def _cmd_score(args) -> int:
    from . import calibrate
    from .atms_bridge import read_thinned
    from .channels import TEMPERATURE_SOUNDING_CHANNELS
    from .columns import load_analysis
    from .entry import entry_from_scorecard, write_entry
    from .score import ALL_CHANNELS, ScreenOptions, score_day, write_scorecard

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    options = ScreenOptions()

    thinned = read_thinned(args.thinned)
    analyses = [load_analysis(path) for path in args.columns]
    times = sorted(a.valid_time for a in analyses)

    day = score_day(thinned, analyses, options=options, bar_k=args.bar_k, max_cells=args.max_cells,
                    channels=ALL_CHANNELS)
    scores = day.scores

    calibration = calibrate.run()
    calibration["passes"] = calibrate.passes(calibration)
    (out / "microwave-calibration.json").write_text(json.dumps(calibration, indent=1), encoding="utf-8")

    provenance = {
        "thinned": thinned.metadata,
        "analyses": [a.provenance | {"valid_time": a.valid_time.isoformat()} for a in analyses],
        "analysis_span": [times[0].isoformat(), times[-1].isoformat()],
        "operator": {
            "absorption": "ITU-R P.676-13 Annex 1 (oxygen, water vapour, dry continuum)",
            "emissivity": "specular Fresnel, Meissner-Wentz 2004 dielectric, 35 psu, no roughness or foam",
            "planck": "exact, Rayleigh-Jeans difference in the calibration receipt",
            "geometry": "plane-parallel, sec(zenith), quasi-polarization by scan angle",
            "layers": "GDAS 41 levels refined 4x in ln p, isothermal extension to 1 Pa",
            "time_interpolation": "linear between bracketing six-hourly analyses, at the cell's mean time",
        },
        "calibration_passes": calibration["passes"],
        "wall_seconds": round(time.monotonic() - started, 1),
        "sampling_seconds": round(day.sampling_seconds, 1),
        "operator_seconds": round(day.operator_seconds, 1),
        "cells_scored": int(day.index.size),
        "cells_scored_by_hour": day.cells_by_hour,
        "subsampled": args.max_cells is not None,
    }
    document = write_scorecard(
        out / "microwave-scorecard.json", scores=scores, screen=day.screen, options=options,
        provenance=provenance, bar_k=args.bar_k,
    )
    entry = entry_from_scorecard(document, calibration)
    write_entry(out / "microwave-operator-entry.json", entry)
    # Per-cell record for the evidence plots and any re-analysis.
    channel_index = np.asarray([c - 1 for c in ALL_CHANNELS])
    np.savez_compressed(
        out / "microwave-cells.npz",
        observed=day.observed.astype(np.float32), background=day.background.astype(np.float32),
        latitude=np.asarray(thinned.lat_mean_deg)[day.index],
        longitude=np.asarray(thinned.lon_mean_deg)[day.index],
        zenith=day.zenith_deg.astype(np.float32), scan_angle=day.scan_angle_deg.astype(np.float32),
        time_unix_s=np.asarray(thinned.time_mean_unix_s)[day.index],
        wind_speed=day.wind_speed_m_s.astype(np.float32),
        precipitable_water=day.surface["precipitable_water"].astype(np.float32),
        skin_temperature=day.column.skin_temperature_k.astype(np.float32),
        cell_std=np.asarray(thinned.tb_std_k)[day.index][:, channel_index],
        beam_count=np.asarray(thinned.tb_count)[day.index][:, channel_index],
        cell_index=day.index,
    )
    summary = {
        "screen": day.screen.stages,
        "verdict": document["verdict"],
        "entry": None if entry is None else {"name": entry.name, "admitted_channels": entry.admitted_channels},
        "channels": {
            s.channel: {
                "n": s.score_cells,
                "bias": round(s.raw["bias"], 3), "rmse": round(s.raw["rmse"], 3),
                "rmse_const": round(s.constant_corrected["rmse"], 3),
                "rmse_linear": round(s.linear_corrected["rmse"], 3),
                "rmse_geometry": round(s.geometry_corrected["rmse"], 3),
                "rmse_wind": round(s.wind_corrected["rmse"], 3),
                "noise_floor": round(s.noise_floor_k, 3),
            }
            for s in scores
        },
        "wall_seconds": provenance["wall_seconds"],
        "operator_seconds": provenance["operator_seconds"],
    }
    print(json.dumps(summary, indent=1))
    sounding = [s for s in scores if s.channel in TEMPERATURE_SOUNDING_CHANNELS]
    return 0 if all(s.within_bar for s in sounding) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="woof global microwave", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="fetch one day of ATMS SDR granule pairs")
    fetch.add_argument("--satellite", required=True, choices=("noaa-20", "noaa-21"))
    fetch.add_argument("--day", required=True)
    fetch.add_argument("--out", required=True)
    fetch.add_argument("--workers", type=int, default=8)
    fetch.add_argument("--limit", type=int, default=None)
    fetch.set_defaults(func=_cmd_fetch)

    decode = sub.add_parser("decode", help="decode a fetched day through rw_atms")
    decode.add_argument("--fetched", required=True, help="the fetch output directory")
    decode.add_argument("--out", required=True)
    decode.add_argument("--limit", type=int, default=None)
    decode.set_defaults(func=_cmd_decode)

    thin = sub.add_parser("thin", help="colocate decoded beams onto rings x longitudes x time bins")
    thin.add_argument("--decoded", required=True)
    thin.add_argument("--out", required=True)
    thin.add_argument("--latitudes", default="0.25", help="grid step in degrees, or a file of ring centres")
    thin.add_argument("--nlon", type=int, default=1440)
    thin.add_argument("--origin", required=True, help="time-bin origin, ISO 8601 UTC")
    thin.add_argument("--bin-s", type=float, default=3600.0)
    thin.add_argument("--max-zenith", type=float, default=90.0)
    thin.set_defaults(func=_cmd_thin)

    columns = sub.add_parser("columns", help="decode GDAS pgrb2 analyses into cached column sets")
    columns.add_argument("--out", required=True)
    columns.add_argument("--scratch", default=None)
    columns.add_argument("grib", nargs="+")
    columns.set_defaults(func=_cmd_columns)

    calibrate = sub.add_parser("calibrate", help="write the synthetic calibration receipt")
    calibrate.add_argument("--out", required=True)
    calibrate.set_defaults(func=_cmd_calibrate)

    score = sub.add_parser("score", help="O-B scorecard of thinned cells against analysis columns")
    score.add_argument("--thinned", required=True)
    score.add_argument("--columns", nargs="+", required=True, help="cached analysis npz files")
    score.add_argument("--out", required=True)
    score.add_argument("--bar-k", type=float, default=1.0)
    score.add_argument("--max-cells", type=int, default=None,
                       help="random subsample of candidate cells (recorded); a probe, not the reading of record")
    score.set_defaults(func=_cmd_score)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
