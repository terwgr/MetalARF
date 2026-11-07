#!/usr/bin/env python3
"""Compare Metal's lake source-state binding with the CUDA source built as host C++.

Part 1 runs lake_bind_source_columns from woof.core.kernels.module_source("lake")
through tools/thompson_real_column_parity/host_backend.py (no FMA contraction)
and the Metal kernel on randomized source fields that reach every branch
(missing or out-of-range water, bottom, mixed-layer and shape values, ice
with and without a fraction, zero fraction, missing ice temperature). CUDA's
own powf is device libm, so the host build maps powf to the glibc float32
word the Metal kernel uses (gfk_pow). Words are compared bit for bit.

Part 2 runs lake_state.initialize_surface on NumPy planes and on Metal planes
with the same synthetic source rows and compares the planes it writes.

Run device work through the project's GPU queue.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "woof/core/kernels"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/thompson_real_column_parity"))
os.environ["CPLUS_INCLUDE_PATH"] = str(KERNELS)

PREFIX = ("#define __launch_bounds__(...)\n"
          "#define powf(a, b) gfk_pow((a), (b))\n")


def source_rows(rng, n):
    def plane(lo, hi, bad_lo, bad_hi, miss=.15):
        v = rng.uniform(lo, hi, n).astype(np.float32)
        r = rng.random(n)
        v[r < miss] = np.nan
        v[(r >= miss) & (r < miss + .08)] = bad_lo
        v[(r >= miss + .08) & (r < miss + .16)] = bad_hi
        return v
    src = np.stack([
        plane(272., 300., 260., 340.),            # water
        plane(272., 290., 260., 340.),            # bottom
        plane(0., 30., -1., np.inf),              # mixed layer depth
        plane(.05, .95, 0., 1.),                  # shape factor
        plane(0., 3., -1., np.inf),               # ice thickness
        plane(0., 1., -.1, 1.2),                  # fraction
        plane(250., 280., np.nan, np.nan, .1),    # ice temperature
    ])
    src[5, rng.random(n) < .1] = 0.
    src[5, rng.random(n) < .1] = 1.
    return src.astype(np.float32)


def lake_layers(rng, n):
    depth = rng.uniform(2., 60., n).astype(np.float32)
    statics = np.zeros((71, n), np.float32)
    statics[0] = depth
    edges = np.sort(rng.uniform(0., 1., (11, n)), axis=0)
    dz = np.diff(np.concatenate([np.zeros((1, n)), edges * depth], axis=0), axis=0)
    dz = np.maximum(dz[:10], np.float32(.05)).astype(np.float32)
    statics[11:21] = dz
    statics[1:11] = (np.cumsum(dz, axis=0) - .5 * dz).astype(np.float32)
    columns = np.zeros((131, n), np.float32)
    columns[5:15] = rng.uniform(272., 295., (10, n))
    columns[15:25] = rng.uniform(0., 1., (10, n))
    columns[4] = rng.uniform(260., 275., n)
    return columns, statics


def part1():
    from woof.core import kernels
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    import host_backend

    host = host_backend.HostModule(
        "lake_bind_probe", PREFIX + kernels.module_source("lake"))
    kernel = host.get_function("lake_bind_source_columns")
    ny, nx = 150, 200
    n = ny * nx
    mismatched = 0
    with MetalContext() as context:
        xp = MetalArrayModule(context)
        for seed in range(4):
            rng = np.random.default_rng(seed)
            src = source_rows(rng, n)
            columns, statics = lake_layers(rng, n)
            ref = columns.copy()
            kernel(((n + 127) // 128,), (128,), (np.int32(n), src, ref, statics))
            mask = np.ones(n, np.float32)
            mask[:7] = 0.
            dev_columns = xp.asarray(columns.reshape(131, ny, nx).copy())
            context.submit_kernel(
                "lake_bind_source_columns",
                {"lakemask": xp.asarray(mask.reshape(ny, nx)),
                 "src": xp.asarray(src.reshape(7, ny, nx).copy()),
                 "lake_columns": dev_columns,
                 "lake_static": xp.asarray(statics.reshape(71, ny, nx).copy())},
                {"n": n, "ny": ny, "nx": nx, "state_words": 131, "static_words": 71},
                exact_policy=False, grid_size=n, threads_per_group=32)
            got = dev_columns.to_numpy().reshape(131, n)
            ref[:, :7] = columns[:, :7]          # masked cells stay untouched
            bad = int(np.count_nonzero(ref.view(np.uint32) != got.view(np.uint32)))
            mismatched += bad
            print(f"seed {seed}: {bad} mismatched words of {ref.size}")
    return mismatched


def part2():
    import woof.ingest.lake_state as obs
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext

    shape = (3, 4)
    wet = np.zeros(shape, np.float32)
    wet[:2, :3] = 1.
    supplied = dict(
        water_temperature=np.array([280., 273.15, 285., 283., 281., 279.]),
        bottom_temperature=np.array([277., 277., 280., np.nan, 276., 276.]),
        mixed_layer_depth=np.array([2., 1., 3., 1., 2., 2.]),
        temperature_shape_factor=np.array([.5, .5, .4, .5, .5, .5]),
        ice_thickness=np.array([0., .6, 0., .2, np.nan, 0.]),
        ice_fraction=np.array([0., .5, 0., .9, .3, 0.]),
        ice_temperature=np.array([273.15, 269., 270., np.nan, 268., 272.]))
    obs.load_state = lambda *a, **kw: (
        {k: v.copy() for k, v in supplied.items()},
        dict(schema="gpuwm.lake-init.v1", status="bound"))
    cfg = SimpleNamespace(lake_state_source="auto")
    valid = datetime(2022, 11, 17, tzinfo=timezone.utc)

    def planes():
        plane = lambda v: np.full(shape, v, np.float32)
        return dict(tsk=plane(290), sst=plane(290), xice=plane(0), xicem=plane(0),
                    lakemask=wet.copy())
    lat, lon = (np.full(shape, v, np.float32) for v in (42., -80.))
    host = planes()
    host_initial, _ = obs.initialize_surface(host, cfg, valid_time=valid,
                                             latitude=lat, longitude=lon)
    with MetalContext() as context:
        xp = MetalArrayModule(context)
        dev = {k: xp.asarray(v) for k, v in planes().items()}
        dev_initial, _ = obs.initialize_surface(dev, cfg, valid_time=valid,
                                                latitude=xp.asarray(lat),
                                                longitude=xp.asarray(lon))
        bad = 0
        for name in host:
            bad += int(np.count_nonzero(
                host[name].view(np.uint32) != dev[name].to_numpy().view(np.uint32)))
        for name in host_initial:
            bad += int(np.count_nonzero(
                host_initial[name].view(np.uint32) != dev_initial[name].view(np.uint32)))
        # End to end: bind through initialize_lake on Metal arrays.
        from woof.core.lake import initialize_lake
        full = dict(dev)
        full.update(isltyp=xp.asarray(np.full(shape, 3, np.int32)),
                    snow=xp.zeros(shape, np.float32),
                    ivgtyp=xp.asarray(np.full(shape, 17, np.int32)),
                    xland=xp.asarray(np.full(shape, 2, np.float32)))
        initialize_lake(full, latitude=xp.asarray(lat), lake_depth=xp.asarray(
            np.full(shape, 20, np.float32)), initial_state=dev_initial)
        columns = full["lake_columns"].to_numpy().reshape(131, -1)
        static = full["lake_static"].to_numpy().reshape(71, -1)
        wet_cells = np.flatnonzero(wet.reshape(-1) == 1)
        ice = (columns[15:25] * static[11:21]).sum(axis=0)[wet_cells]
        print("bound ice volume (m):", np.round(ice, 4))
        print("top-layer water (K):", np.round(columns[5, wet_cells], 3))
    print(f"initialize_surface host-vs-Metal differing words: {bad}")
    return bad


def main():
    total = part1() + part2()
    print(f"total mismatched words {total}")
    return int(total != 0)


if __name__ == "__main__":
    raise SystemExit(main())
