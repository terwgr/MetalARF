#!/usr/bin/env python3
"""Exercise real SW solver routing above the local frame's layer capacity.

The unchanged 128-layer solver supplies nontrivial optics/flux words.
Inserting transparent layers at TOA preserves those words at every mapped
level, so taller workspace solves can be compared without substituting an
approximate host transcendental. Calls include public sw_rte, the driver's
paired fused-cloud/clear route, and multiple workspace slabs/g-point tiles.
Run GPU work through Anvil with synchronous verification enabled.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from woof.core.rrtmgp import (
        CloudOpticsResult, FusedCloudOptics, _sw_rte, sw_rte)
    from woof.metal.array import MetalArrayModule
    from woof.metal.rrtmgp_rte_family import SW2StreamWorkspace
    from woof.metal.runtime import MetalContext

    ncol, base_nlay, ngpt, nband = 5, 128, 35, 2
    index = np.arange(ncol * base_nlay * ngpt).reshape(ncol, base_nlay, ngpt)
    tau = (.0001 * (1 + index % 31)).astype(np.float32)
    ssa = (.2 + .01 * (index % 61)).astype(np.float32)
    asym = (.01 * (index % 71)).astype(np.float32)
    mu = np.broadcast_to(np.array([.3, .5, .8, 0., 1.], np.float32)[:, None],
                         (ncol, base_nlay)).copy()
    incident = np.broadcast_to(np.linspace(.1, 2., ngpt, dtype=np.float32),
                               (ncol, ngpt)).copy()
    albedo = np.full((ncol, ngpt), .17, np.float32)
    diffuse = np.full_like(albedo, .23)
    cloud_index = np.arange(ncol * base_nlay * nband).reshape(ncol, base_nlay, nband)
    cloud_tau = (.0003 * (cloud_index % 17)).astype(np.float32)
    cloud_ssa = np.full_like(cloud_tau, .8)
    cloud_g = np.full_like(cloud_tau, .7)
    mask = (index % 3 != 0).astype(np.float32)
    comparisons = words = calls = 0
    failures = []
    workspaces = []

    def compare(label, expected, actual):
        nonlocal comparisons, words
        expected = np.ascontiguousarray(expected)
        actual = np.ascontiguousarray(actual)
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            failures.append({"case": label, "nonfinite": True})
        mismatch = np.flatnonzero(expected.view(np.uint32).ravel()
                                  != actual.view(np.uint32).ravel())
        comparisons += 1
        words += expected.size
        if mismatch.size:
            failures.append({"case": label, "mismatches": int(mismatch.size),
                             "max_abs": float(np.max(np.abs(actual - expected)))})

    with MetalContext() as context:
        arrays = MetalArrayModule(context)
        tables = SimpleNamespace(ngpt=ngpt, nband=nband,
                                 gpoint_bands=np.arange(ngpt, dtype=np.int32) % nband)
        alb, dif, inc = map(arrays.asarray, (albedo, diffuse, incident))
        for top_at_1 in (False, True):
            for fused in (False, True):
                def solve(nlay, *, workspace=None):
                    nonlocal calls
                    padding = nlay - base_nlay
                    def padded(value, fill=0.):
                        extra = np.full((ncol, padding, *value.shape[2:]), fill, np.float32)
                        return arrays.asarray(np.concatenate(
                            (extra, value) if top_at_1 else (value, extra), axis=1))
                    dtau, dssa, dasym = map(padded, (tau, ssa, asym))
                    # A constant per-column solar direction is unchanged by
                    # transparent top layers, including an unlit column.
                    dmu = arrays.asarray(np.repeat(mu[:, :1], nlay, axis=1))
                    if fused:
                        cloud = FusedCloudOptics(tables,
                            CloudOpticsResult(padded(cloud_tau), padded(cloud_ssa),
                                              padded(cloud_g)), padded(mask))
                        result = _sw_rte(dtau, dssa, None, dmu, alb, dif, inc,
                            top_at_1=top_at_1, out=None, fused_cloud=cloud,
                            with_clear=True, _metal_workspace=workspace)
                    elif workspace is None:
                        result = sw_rte(dtau, dssa, dasym, dmu, alb, dif, inc,
                                        top_at_1=top_at_1)
                    else:
                        result = _sw_rte(dtau, dssa, dasym, dmu, alb, dif, inc,
                            top_at_1=top_at_1, out=None, _metal_workspace=workspace)
                    calls += 1
                    fluxes = result if fused else (result,)
                    return tuple(tuple(value.to_numpy() for value in
                        (flux.flux_up, flux.flux_dn, flux.flux_dir)
                        if value is not None) for flux in fluxes)

                baseline = solve(base_nlay)
                for nlay in (129, 160, 257):
                    full = solve(nlay)
                    frame = SW2StreamWorkspace(context, 2, nlay, ngpt, gpt_tile=6)
                    chunked = solve(nlay, workspace=frame)
                    workspaces.append({"nlay": nlay, "column_capacity": frame.columns,
                        "tile": frame.gpt_tile, "allocated_bytes": frame.allocated_bytes,
                        "budget_bytes": frame.budget_bytes})
                    pad = nlay - base_nlay
                    label = f"layers={nlay},top_at_1={top_at_1},fused={fused}"
                    for sky, (base_fields, full_fields, chunk_fields) in enumerate(
                            zip(baseline, full, chunked)):
                        for field, (expected, value, slabbed) in enumerate(
                                zip(base_fields, full_fields, chunk_fields)):
                            mapped = value[:, pad:] if top_at_1 else value[:, :base_nlay+1]
                            compare(label + f",sky={sky},field={field},mapped", expected, mapped)
                            upper = value[:, :pad] if top_at_1 else value[:, base_nlay+1:]
                            toa = expected[:, :1] if top_at_1 else expected[:, -1:]
                            compare(label + f",sky={sky},field={field},transparent",
                                    np.repeat(toa, pad, axis=1), upper)
                            compare(label + f",sky={sky},field={field},slabs", value, slabbed)
    print(json.dumps({"solver_calls": calls, "comparisons": comparisons,
        "words": words, "failed_comparisons": len(failures), "failures": failures[:12],
        "workspace_plans": workspaces}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
