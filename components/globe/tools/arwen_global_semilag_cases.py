"""Drive and score the two analytic cases the semi-Lagrangian core is
graded on: Jablonowski and Williamson 2006, and Held and Suarez 1994.

The baroclinic wave runs through the ordinary ``arwen_global run``
door (``[initial] mode = "baroclinic_wave"``), so this tool only SCORES
it: it reads the checkpoints of a steady arm and a perturbed arm and
reports the norms the case is judged on.

The Held and Suarez climate has no door, because its forcing is not a
physics suite: it relaxes temperature and damps wind and touches nothing
else, so a suite's exchange, closure ledger and water reservoir would all
be machinery with nothing to do.  It is driven here instead, one model
step and one forcing step at a time, with the same dycore, the same
fixers and the same hyperdiffusion the door would use.

Usage::

    python tools/arwen_global_semilag_cases.py score-baroclinic \\
        --steady OUT_STEADY --perturbed OUT_PERTURBED --out report.json
    python tools/arwen_global_semilag_cases.py held-suarez CONFIG \\
        --outdir OUT --spin-up-days 200 --statistics-days 1000
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from woof.globe.checkpoint import read_checkpoint  # noqa: E402
from woof.globe.config import load_config  # noqa: E402
from woof.globe.constants import KAPPA, REFERENCE_PRESSURE_PA  # noqa: E402
from woof.globe.runner import (  # noqa: E402
    build_model_and_cold_state, build_transform,
)
from woof.globe.testcases import apply_held_suarez  # noqa: E402


# ------------------------------------------------------------- baroclinic

def _checkpoints(outdir: Path) -> dict[int, Path]:
    out = {}
    for path in sorted(Path(outdir).glob("arwen_global_step*.npz")):
        out[int(path.stem.rsplit("step", 1)[1])] = path
    return out


def _surface_pressure(path: Path, transform):
    metadata, arrays = read_checkpoint(path)
    coefficients = transform.backend.asarray(
        arrays["atmosphere__log_surface_pressure"],
        dtype=transform.backend.complex_dtype,
    )
    grid = transform.backend.to_numpy(transform.inverse(coefficients))
    return metadata, np.exp(np.asarray(grid, dtype=np.float64))


def _area_norms(field, transform):
    """Area-weighted l2 and linf of a two-dimensional field."""
    weights = np.asarray(transform.grid.quadrature_weights, dtype=np.float64)
    cell = weights[:, None] / (2.0 * transform.grid.nlon)
    return (
        float(math.sqrt(float(np.sum(field ** 2 * cell)))),
        float(np.max(np.abs(field))),
    )


def score_baroclinic(args) -> int:
    cfg = load_config(args.config)
    transform = build_transform(cfg)
    steady = _checkpoints(Path(args.steady))
    perturbed = _checkpoints(Path(args.perturbed))
    shared = sorted(set(steady) & set(perturbed))
    if not shared:
        raise SystemExit("the two arms share no checkpoint step")
    _meta0, ps0 = _surface_pressure(steady[shared[0]], transform)
    rows = []
    for step in shared:
        meta_s, ps_s = _surface_pressure(steady[step], transform)
        meta_p, ps_p = _surface_pressure(perturbed[step], transform)
        l2, linf = _area_norms(ps_p - ps_s, transform)
        drift_l2, drift_linf = _area_norms(ps_s - ps0, transform)
        rows.append({
            "step": step,
            "day": float(meta_s["time_s"]) / 86400.0,
            "steady_max_abs_ps_change_pa": drift_linf,
            "steady_l2_ps_change_pa": drift_l2,
            "perturbation_l2_ps_pa": l2,
            "perturbation_linf_ps_pa": linf,
            "perturbed_min_ps_pa": float(np.min(ps_p)),
            "perturbed_max_ps_pa": float(np.max(ps_p)),
        })
    # The e-folding time of the perturbation between two days, which is
    # the case's own growth-rate instrument.
    growth = None
    early = [r for r in rows if abs(r["day"] - args.growth_from) < 1.0e-6]
    late = [r for r in rows if abs(r["day"] - args.growth_to) < 1.0e-6]
    if early and late and early[0]["perturbation_l2_ps_pa"] > 0.0:
        span = late[0]["day"] - early[0]["day"]
        ratio = late[0]["perturbation_l2_ps_pa"] / early[0]["perturbation_l2_ps_pa"]
        if ratio > 1.0:
            growth = span / math.log(ratio)
    report = {
        "case": "jablonowski-williamson-2006",
        "config": str(args.config),
        "steady": str(args.steady),
        "perturbed": str(args.perturbed),
        "rows": rows,
        "e_folding_days": growth,
        "growth_window_days": [args.growth_from, args.growth_to],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


# ------------------------------------------------------------ held-suarez

def _zonal_statistics(model, bundle, accumulator):
    """Accumulate the zonal-mean and eddy statistics the case reports."""
    transform = model.transform
    host = transform.backend.to_numpy
    g = model.grid_state(bundle.atmosphere, only=("u", "v", "temperature",
                                                  "p_full"))
    u = np.asarray(host(g["u"]), dtype=np.float64)
    v = np.asarray(host(g["v"]), dtype=np.float64)
    t = np.asarray(host(g["temperature"]), dtype=np.float64)
    p = np.asarray(host(g["p_full"]), dtype=np.float64)
    u_bar = u.mean(axis=2)
    v_bar = v.mean(axis=2)
    t_bar = t.mean(axis=2)
    u_prime = u - u_bar[..., None]
    v_prime = v - v_bar[..., None]
    accumulator["count"] += 1
    accumulator["u"] += u_bar
    accumulator["t"] += t_bar
    accumulator["p"] += p.mean(axis=2)
    accumulator["eke"] += 0.5 * (u_prime ** 2 + v_prime ** 2).mean(axis=2)
    accumulator["uv"] += (u_prime * v_prime).mean(axis=2)


def held_suarez(args) -> int:
    cfg = load_config(args.config)
    transform = build_transform(cfg)
    model, bundle = build_model_and_cold_state(cfg)
    dt = float(cfg.dt_s)
    spin_up = int(round(args.spin_up_days * 86400.0 / dt))
    statistics = int(round(args.statistics_days * 86400.0 / dt))
    sample_every = max(1, int(round(args.sample_hours * 3600.0 / dt)))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nlev = int(model.nlev)
    nlat = int(transform.grid.nlat)
    accumulator = {
        "count": 0,
        "u": np.zeros((nlev, nlat)),
        "t": np.zeros((nlev, nlat)),
        "p": np.zeros((nlev, nlat)),
        "eke": np.zeros((nlev, nlat)),
        "uv": np.zeros((nlev, nlat)),
    }
    start_mass = model.diagnostics(bundle)["global_mean_surface_pressure_pa"]
    started = time.perf_counter()
    trips = []
    for step in range(spin_up + statistics):
        bundle, metrics = model.step(bundle, dt)
        bundle = apply_held_suarez(model, bundle, dt)
        if step >= spin_up and (step - spin_up) % sample_every == 0:
            _zonal_statistics(model, bundle, accumulator)
        if step % max(1, (spin_up + statistics) // 40) == 0:
            diagnostic = model.diagnostics(bundle)
            trips.append({
                "step": step,
                "day": step * dt / 86400.0,
                "global_mean_surface_pressure_pa":
                    diagnostic["global_mean_surface_pressure_pa"],
                "semilag_lipschitz": float(metrics.get("semilag_lipschitz", 0.0)),
                "spectral_cfl": float(metrics["spectral_cfl"]),
            })
            print(f"  day {trips[-1]['day']:7.1f}  ps "
                  f"{trips[-1]['global_mean_surface_pressure_pa']:.4f}  "
                  f"lipschitz {trips[-1]['semilag_lipschitz']:.4f}", flush=True)
    count = max(1, accumulator["count"])
    for key in ("u", "t", "p", "eke", "uv"):
        accumulator[key] /= count
    end_mass = model.diagnostics(bundle)["global_mean_surface_pressure_pa"]

    latitude = np.asarray(transform.grid.latitude_deg, dtype=np.float64)
    pressure = accumulator["p"]
    jet_index = np.unravel_index(np.argmax(accumulator["u"]),
                                 accumulator["u"].shape)
    eke_index = np.unravel_index(np.argmax(accumulator["eke"]),
                                 accumulator["eke"].shape)
    north = latitude > 0.0
    south = latitude < 0.0
    summary = {
        "case": "held-suarez-1994",
        "config": str(args.config),
        "integrator": cfg.integrator,
        "dt_s": dt,
        "truncation": cfg.truncation,
        "nlev": nlev,
        "spin_up_days": args.spin_up_days,
        "statistics_days": args.statistics_days,
        "samples": int(count),
        "wall_seconds": float(time.perf_counter() - started),
        "jet_maximum_m_s": float(accumulator["u"][jet_index]),
        "jet_pressure_pa": float(pressure[jet_index]),
        "jet_latitude_deg": float(latitude[jet_index[1]]),
        "jet_maximum_northern_m_s": float(np.max(accumulator["u"][:, north])),
        "jet_maximum_southern_m_s": float(np.max(accumulator["u"][:, south])),
        "eddy_kinetic_energy_maximum_m2_s2": float(accumulator["eke"][eke_index]),
        "eddy_kinetic_energy_pressure_pa": float(pressure[eke_index]),
        "eddy_kinetic_energy_latitude_deg": float(latitude[eke_index[1]]),
        "eddy_momentum_flux_maximum_m2_s2": float(np.max(accumulator["uv"])),
        "eddy_momentum_flux_minimum_m2_s2": float(np.min(accumulator["uv"])),
        "surface_equatorial_temperature_k": float(
            accumulator["t"][-1][np.argmin(np.abs(latitude))]
        ),
        "global_mean_surface_pressure_start_pa": float(start_mass),
        "global_mean_surface_pressure_end_pa": float(end_mass),
        "surface_pressure_relative_drift": float(
            abs(end_mass - start_mass) / max(abs(start_mass), 1.0e-30)
        ),
        "progress": trips,
    }
    np.savez_compressed(
        outdir / "held_suarez_zonal_means.npz",
        latitude_deg=latitude, **{k: accumulator[k]
                                  for k in ("u", "t", "p", "eke", "uv")},
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    (outdir / "held_suarez.json").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    score = commands.add_parser("score-baroclinic")
    score.add_argument("config")
    score.add_argument("--steady", required=True)
    score.add_argument("--perturbed", required=True)
    score.add_argument("--out", default=None)
    score.add_argument("--growth-from", type=float, default=4.0)
    score.add_argument("--growth-to", type=float, default=8.0)
    score.set_defaults(handler=score_baroclinic)

    hs = commands.add_parser("held-suarez")
    hs.add_argument("config")
    hs.add_argument("--outdir", required=True)
    hs.add_argument("--spin-up-days", type=float, default=200.0)
    hs.add_argument("--statistics-days", type=float, default=1000.0)
    hs.add_argument("--sample-hours", type=float, default=24.0)
    hs.set_defaults(handler=held_suarez)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
