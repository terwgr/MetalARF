"""Gate RED-1: does a global reduction move when the grid is banded?

``python tools/arwen_global_reduction_probe.py [--config CFG]
[--truncation 63] [--nlev 40] [--bands 1,2,3,4,8,16,32] [--backend numpy]
[--precision float64] [--json out.json]``

Eleven global reductions run in one WOOF global step.  This probe
computes each of them TWICE on the same arrays: once whole, and once band
by band through the accumulators of
:mod:`woof.globe.bands`, at every band count asked for.  A
reduction that returns a different bit at any band count is printed by
name; the gate passes only when every one of them is byte-identical at
every band count.

What it measures: whether the two-stage form each reduction now takes on
the resident path is invariant under the band schedule -- which is the
property that lets a banded or a two-card run reproduce a resident run's
checkpoint bit for bit, with no new pin and no tolerance.  It does not
measure the transforms (gate FFT-1) or the contraction (gate WAIST-1).

With ``--config`` the arrays are a real model's: the config's own initial
state is built and its vapor, layer thickness, temperature, surface
pressure, winds and grid tracers are the operands, so the numbers are the
ones the step actually reduces.  Without it the arrays are a seeded RNG's
at the requested shape, which sweeps band counts and truncations the
model would take minutes to build.

``--flat-sum-ulp`` additionally reports what the ONE rewritten reduction
cost: ``enforce``'s vapor negative- and positive-mass readings were a flat
three-dimensional sum and are now a per-row sum reduced over latitude.
The probe prints the difference between the two forms in ulp of the
result, which is the number published beside the unchanged checkpoint
hashes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

# The probe is run as a script from the tree root; make the tree's own
# package the one it measures rather than whatever is installed.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))


def _ulp_delta(a, b, dtype) -> float:
    """|a - b| in ulp OF THE WORKING DTYPE.

    Counting a float32 difference in float64 ulp reads eight orders of
    magnitude too large, which is the difference between "the last bit
    moved" and "the answer changed".
    """
    left = np.asarray(a, dtype=dtype)
    right = np.asarray(b, dtype=dtype)
    if left == right:
        return 0.0
    scale = np.maximum(np.abs(left), np.abs(right))
    if not np.isfinite(scale) or scale == 0:
        return float("inf")
    return float(np.abs(left.astype(np.float64) - right.astype(np.float64))
                 / float(np.spacing(scale)))


def _equal(xp, left, right) -> bool:
    return bool(np.array_equal(_host(xp, left), _host(xp, right)))


def _host(xp, value):
    if xp.__name__ == "numpy":
        return np.asarray(value)
    return xp.asnumpy(value)


def _fields_from_config(path, backend_name, precision):
    from woof.globe.config import load_config
    from woof.globe.runner import build_model_and_cold_state, build_transform
    import dataclasses

    cfg = load_config(path)
    cfg = dataclasses.replace(cfg, backend=backend_name, precision=precision)
    transform = build_transform(cfg)
    model, bundle = build_model_and_cold_state(cfg, transform)
    g = model.grid_state(
        bundle.atmosphere, only=("qv", "dp", "temperature", "ps", "u", "v"),
    )
    tracers = bundle.atmosphere.grid_tracers()
    return {
        "xp": transform.backend.xp,
        "dtype": transform.backend.float_dtype,
        "grid": transform.grid,
        "qv": g["qv"], "dp": g["dp"], "temperature": g["temperature"],
        "ps": g["ps"], "u": g["u"], "v": g["v"],
        "tracers": tracers,
        "source": f"model built from {path}",
    }


def _fields_synthetic(truncation, nlev, backend_name, precision, seed):
    from woof.globe.spectral.backend import get_backend
    from woof.globe.spectral.grid import GaussianGrid

    backend = get_backend(backend_name, precision)
    xp = backend.xp
    grid = GaussianGrid.create(int(truncation))
    nlat, nlon = grid.shape
    rng = np.random.default_rng(int(seed))
    dtype = backend.float_dtype

    def field(shape, scale, offset=0.0):
        return backend.asarray(
            (rng.standard_normal(shape) * scale + offset), dtype=dtype
        )

    qv = field((nlev, nlat, nlon), 3.0e-3, 6.0e-3)
    dp = field((nlev, nlat, nlon), 2.0e3, 2.5e4)
    tracers = {
        name: xp.maximum(field((nlev, nlat, nlon), 2.0e-5, 4.0e-5), 0.0)
        for name in ("qc", "qr", "qi")
    }
    return {
        "xp": xp,
        "dtype": dtype,
        "grid": grid,
        "qv": qv,
        "dp": xp.abs(dp),
        "temperature": field((nlev, nlat, nlon), 20.0, 250.0),
        "ps": field((nlat, nlon), 2.0e3, 1.0e5),
        "u": field((nlev, nlat, nlon), 15.0, 5.0),
        "v": field((nlev, nlat, nlon), 12.0, 0.0),
        "tracers": tracers,
        "source": (
            f"seeded RNG {seed} at T{truncation} L{nlev} on the "
            f"{backend_name}/{precision} backend"
        ),
    }


def _reductions(fields, bands):
    """Every one of the eleven, whole against banded, at each band count."""
    from woof.globe.bands import (
        AssociativeAccumulator,
        LatitudeAccumulator,
        PlaneAccumulator,
        band_slices,
    )
    from woof.globe.constants import GRAVITY_M_S2

    xp = fields["xp"]
    grid = fields["grid"]
    nlat, nlon = grid.shape
    dtype = fields["dtype"]
    qv = fields["qv"]
    dp = fields["dp"]
    weights = xp.asarray(np.asarray(grid.quadrature_weights), dtype=dtype)
    cell = weights[:, None] / (2.0 * nlon)
    cell_weight = xp.asarray(
        (np.asarray(grid.quadrature_weights) / (2.0 * nlon))[:, None], dtype=dtype
    )
    weighted = qv * dp
    stacked = xp.stack([qv, sum(fields["tracers"].values())])
    density = dp
    negative_plane = xp.sum(xp.minimum(weighted, 0.0), axis=0)
    positive_plane = xp.sum(xp.maximum(weighted, 0.0), axis=0)
    fillable = (positive_plane > 0.0) & (positive_plane + negative_plane > 0.0)
    scale = xp.where(
        fillable,
        (positive_plane + negative_plane) / xp.where(fillable, positive_plane, 1.0),
        1.0,
    )
    speed = xp.sqrt(fields["u"] ** 2 + fields["v"] ** 2)
    courant = xp.abs(fields["u"]) * 1.0e-4
    # The tracer transport's pseudo-density gap: the step's SECOND flat
    # three-dimensional sum, and a maximum beside it.
    gap = xp.abs(fields["temperature"] - 250.0) / 250.0
    gap_cell = weights[None, :, None] / (2.0 * nlon)

    def whole():
        out = {}
        rows = xp.mean(stacked * dp, axis=-1)
        out["level_water_mass"] = 0.5 * xp.sum(rows * weights, axis=-1) / GRAVITY_M_S2
        out["column_holes_created"] = -xp.sum(negative_plane * cell)
        out["column_holes_unfillable"] = -xp.sum(
            xp.where(fillable, 0.0, negative_plane) * cell
        )
        out["column_holes_rescale"] = xp.max(1.0 - scale)
        out["floor_grid_tracers_min"] = xp.stack(
            [xp.min(v) for v in fields["tracers"].values()]
        )
        out["cfl_wind_max"] = xp.max(speed)
        out["enforce_temperature_min"] = xp.min(fields["temperature"])
        out["enforce_temperature_max"] = xp.max(fields["temperature"])
        out["enforce_ps_min"] = xp.min(fields["ps"])
        out["enforce_ps_max"] = xp.max(fields["ps"])
        out["enforce_qv_negative"] = -xp.sum(
            xp.sum(xp.minimum(weighted, 0.0), axis=-1)
        )
        out["enforce_qv_positive"] = xp.sum(
            xp.sum(xp.maximum(weighted, 0.0), axis=-1)
        )
        column = xp.sum(stacked * density, axis=1)
        out["transport_global_mean_columns"] = xp.sum(
            xp.sum(column * cell_weight, axis=-1), axis=-1
        ) / GRAVITY_M_S2
        out["transport_substeps_max"] = xp.max(courant)
        out["pseudo_density_gap_max"] = xp.max(gap)
        out["pseudo_density_gap_mean"] = xp.sum(
            xp.sum(gap * gap_cell, axis=-1)
        ) / gap.shape[0]
        out["sponge_active_any"] = xp.any(gap > 0.05)
        # enforce's finiteness fold, the one exactly-associative reduction
        # of the inventory the first probe did not carry.
        out["enforce_finite_all"] = xp.all(xp.isfinite(fields["temperature"]))
        out["transport_row_courant"] = xp.max(courant, axis=(0, 2))
        out["fix_mass_plane_mean"] = xp.asarray(
            grid.global_mean(_host(xp, xp.exp(fields["ps"] * 1.0e-5)))
        )
        return out

    def banded(count):
        slices = band_slices(nlat, count)
        out = {}
        water = LatitudeAccumulator(
            xp, (stacked.shape[0], stacked.shape[1], nlat), dtype, name="water",
        )
        created = PlaneAccumulator(xp, (nlat, nlon), negative_plane.dtype, name="cr")
        unfill = PlaneAccumulator(xp, (nlat, nlon), negative_plane.dtype, name="un")
        rescale = AssociativeAccumulator(xp, "max", name="rescale")
        tracer_minima = {
            name: AssociativeAccumulator(xp, "min", name=name)
            for name in fields["tracers"]
        }
        wind = AssociativeAccumulator(xp, "max", name="cfl")
        t_min = AssociativeAccumulator(xp, "min", name="tmin")
        t_max = AssociativeAccumulator(xp, "max", name="tmax")
        ps_min = AssociativeAccumulator(xp, "min", name="psmin")
        ps_max = AssociativeAccumulator(xp, "max", name="psmax")
        row_shape = (weighted.shape[0], nlat)
        qv_negative = LatitudeAccumulator(xp, row_shape, weighted.dtype, name="neg")
        qv_positive = LatitudeAccumulator(xp, row_shape, weighted.dtype, name="pos")
        columns = LatitudeAccumulator(
            xp, (stacked.shape[0], nlat), dtype, name="columns",
        )
        substeps = AssociativeAccumulator(xp, "max", name="substeps")
        gap_max = AssociativeAccumulator(xp, "max", name="gapmax")
        gap_rows = LatitudeAccumulator(
            xp, (gap.shape[0], nlat), gap.dtype, name="gaprows",
        )
        sponge = AssociativeAccumulator(xp, "any", name="sponge")
        finite = AssociativeAccumulator(xp, "all", name="finite")
        row_courant = LatitudeAccumulator(xp, (nlat,), courant.dtype, name="rows")
        surface = PlaneAccumulator(xp, (nlat, nlon), dtype, name="ps")
        for rows in slices:
            water.add_band(rows, xp.mean(stacked[..., rows, :] * dp[..., rows, :], axis=-1))
            created.add_band(rows, negative_plane[rows])
            unfill.add_band(
                rows, xp.where(fillable[rows], 0.0, negative_plane[rows])
            )
            rescale.add_band(1.0 - scale[rows])
            for name, value in fields["tracers"].items():
                tracer_minima[name].add_band(value[..., rows, :])
            wind.add_band(speed[..., rows, :])
            t_min.add_band(fields["temperature"][..., rows, :])
            t_max.add_band(fields["temperature"][..., rows, :])
            ps_min.add_band(fields["ps"][rows])
            ps_max.add_band(fields["ps"][rows])
            band_weighted = weighted[..., rows, :]
            qv_negative.add_band(
                rows, xp.sum(xp.minimum(band_weighted, 0.0), axis=-1)
            )
            qv_positive.add_band(
                rows, xp.sum(xp.maximum(band_weighted, 0.0), axis=-1)
            )
            band_column = xp.sum(
                stacked[..., rows, :] * density[..., rows, :], axis=1
            )
            columns.add_band(
                rows, xp.sum(band_column * cell_weight[rows], axis=-1)
            )
            substeps.add_band(courant[..., rows, :])
            gap_max.add_band(gap[..., rows, :])
            gap_rows.add_band(
                rows, xp.sum(gap[..., rows, :] * gap_cell[..., rows, :], axis=-1)
            )
            sponge.add_band(gap[..., rows, :] > 0.05)
            finite.add_band(xp.isfinite(fields["temperature"][..., rows, :]))
            row_courant.add_band(rows, xp.max(courant[..., rows, :], axis=(0, 2)))
            surface.add_band(rows, xp.exp(fields["ps"][rows] * 1.0e-5))
        out["level_water_mass"] = 0.5 * water.total(weights) / GRAVITY_M_S2
        out["column_holes_created"] = -created.total(cell)
        out["column_holes_unfillable"] = -unfill.total(cell)
        out["column_holes_rescale"] = rescale.total()
        out["floor_grid_tracers_min"] = xp.stack(
            [tracer_minima[name].total() for name in fields["tracers"]]
        )
        out["cfl_wind_max"] = wind.total()
        out["enforce_temperature_min"] = t_min.total()
        out["enforce_temperature_max"] = t_max.total()
        out["enforce_ps_min"] = ps_min.total()
        out["enforce_ps_max"] = ps_max.total()
        out["enforce_qv_negative"] = -qv_negative.total(axis=None)
        out["enforce_qv_positive"] = qv_positive.total(axis=None)
        out["transport_global_mean_columns"] = columns.total() / GRAVITY_M_S2
        out["transport_substeps_max"] = substeps.total()
        out["pseudo_density_gap_max"] = gap_max.total()
        out["pseudo_density_gap_mean"] = gap_rows.total(axis=None) / gap.shape[0]
        out["sponge_active_any"] = sponge.total()
        out["enforce_finite_all"] = finite.total()
        out["transport_row_courant"] = row_courant.complete()
        out["fix_mass_plane_mean"] = xp.asarray(
            grid.global_mean(_host(xp, surface.plane))
        )
        return out

    reference = whole()
    results = {}
    for count in bands:
        got = banded(count)
        results[int(count)] = {
            key: _equal(xp, reference[key], got[key]) for key in reference
        }
    flat = {
        "pseudo_density_gap_mean_flat": float(
            _host(xp, xp.sum(gap * gap_cell) / gap.shape[0])
        ),
        "pseudo_density_gap_mean_two_stage": float(
            _host(xp, reference["pseudo_density_gap_mean"])
        ),
        "enforce_qv_negative_flat": float(_host(xp, -xp.sum(xp.minimum(weighted, 0.0)))),
        "enforce_qv_positive_flat": float(_host(xp, xp.sum(xp.maximum(weighted, 0.0)))),
        "enforce_qv_negative_two_stage": float(
            _host(xp, reference["enforce_qv_negative"])
        ),
        "enforce_qv_positive_two_stage": float(
            _host(xp, reference["enforce_qv_positive"])
        ),
    }
    dtype = np.dtype(_host(xp, weighted).dtype)
    flat["dtype"] = str(dtype)
    flat["enforce_qv_negative_ulp"] = _ulp_delta(
        flat["enforce_qv_negative_flat"], flat["enforce_qv_negative_two_stage"], dtype
    )
    flat["enforce_qv_positive_ulp"] = _ulp_delta(
        flat["enforce_qv_positive_flat"], flat["enforce_qv_positive_two_stage"], dtype
    )
    flat["pseudo_density_gap_mean_ulp"] = _ulp_delta(
        flat["pseudo_density_gap_mean_flat"],
        flat["pseudo_density_gap_mean_two_stage"],
        np.dtype(_host(xp, gap).dtype),
    )
    return len(reference), results, flat


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--truncation", type=int, default=63)
    parser.add_argument("--nlev", type=int, default=40)
    parser.add_argument("--bands", default="1,2,3,4,8,16,32")
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--precision", default="float64")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    band_tokens = [v.strip() for v in str(args.bands).split(",") if v.strip()]
    if args.config:
        fields = _fields_from_config(args.config, args.backend, args.precision)
    else:
        fields = _fields_synthetic(
            args.truncation, args.nlev, args.backend, args.precision, args.seed
        )
    nlat = fields["grid"].shape[0]
    # "nlat" asks for one latitude row per band, the strongest case the
    # gate has: it is where a reduction's own algorithm changes with the
    # shape it is handed.
    bands = sorted({
        min(nlat, nlat if token == "nlat" else int(token))
        for token in band_tokens
    })
    count, results, flat = _reductions(fields, bands)

    failures = []
    for band_count, row in sorted(results.items()):
        for name, ok in sorted(row.items()):
            if not ok:
                failures.append((band_count, name))
    print(f"RED-1  source: {fields['source']}")
    print(f"RED-1  {count} reductions x {len(bands)} band counts "
          f"({','.join(str(b) for b in bands)})")
    for band_count, row in sorted(results.items()):
        bad = sorted(name for name, ok in row.items() if not ok)
        print(f"  B={band_count:<3d} {sum(row.values())}/{len(row)} exact"
              + ("" if not bad else "   MOVED: " + ", ".join(bad)))
    print(
        "RED-1  the rewritten flat sum: negative "
        f"{flat['enforce_qv_negative_flat']:.17g} flat against "
        f"{flat['enforce_qv_negative_two_stage']:.17g} two-stage, "
        f"{flat['enforce_qv_negative_ulp']:.1f} ulp; positive "
        f"{flat['enforce_qv_positive_ulp']:.1f} ulp"
    )
    print(
        "RED-1  the transport gap mean: "
        f"{flat['pseudo_density_gap_mean_flat']:.17g} flat against "
        f"{flat['pseudo_density_gap_mean_two_stage']:.17g} two-stage, "
        f"{flat['pseudo_density_gap_mean_ulp']:.1f} ulp"
    )
    print("RED-1  " + ("PASS" if not failures else f"FAIL ({len(failures)})"))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "gate": "RED-1",
                    "source": fields["source"],
                    "backend": args.backend,
                    "precision": args.precision,
                    "bands": bands,
                    "reductions": count,
                    "results": {str(k): v for k, v in results.items()},
                    "flat_sum_rewrite": flat,
                    "pass": not failures,
                },
                handle, indent=1,
            )
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
