"""What the Grell-Freitas column packing costs on this card, and whether a
one-launch gather returns the same bits.

``python tools/arwen_global_pack_probe.py [--nz 40] [--ny 384] [--nx 768]
[--chunk 131072] [--lanes 15] [--repeats 20] [--json out.json]``

woof.globe.core.gf packs every level lane of a column chunk with two strided
copies per lane (``cols``: the (nz, ncol) slice transposed into a
contiguous (n, nz) block, then written into ``lvin[:, j, :]``) and
unpacks four output lanes the same way.  This probe times, on the cupy
backend, (a) that two-copy packing of ``lanes`` lanes for one chunk,
(b) one ElementwiseKernel gather that writes ``lvin`` straight from the
(nz, ny, nx) lanes, and (c) the output unpacking both ways, and checks
with numpy.array_equal that the gathers return the packed bits.  Copies
move no arithmetic, so equality is the whole correctness claim.

What it measures: cupyx.profiler.benchmark kernel time per repetition
(minimum and mean over the repeats) of each packing on the running card.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

_GATHER = r"""
    // i runs over (n, lanes, nz): column c of the chunk, lane j, level k.
    const long long k = i % nz;
    const long long j = (i / nz) % lanes;
    const long long c = i / (nz * lanes);
    const long long src = k * ncol + lo + c;
    T value;
    switch (j) {
        case 0: value = l0[src]; break;  case 1: value = l1[src]; break;
        case 2: value = l2[src]; break;  case 3: value = l3[src]; break;
        case 4: value = l4[src]; break;  case 5: value = l5[src]; break;
        case 6: value = l6[src]; break;  case 7: value = l7[src]; break;
        case 8: value = l8[src]; break;  case 9: value = l9[src]; break;
        case 10: value = l10[src]; break; case 11: value = l11[src]; break;
        case 12: value = l12[src]; break; case 13: value = l13[src]; break;
        case 14: value = l14[src]; break;
        default: value = T(0);
    }
    out = value;
"""

_SCATTER = r"""
    // i runs over (nout, nz, n): output lane j, level k, column c -> out[j][k, lo + c].
    const long long c = i % n;
    const long long k = (i / n) % nz;
    const long long j = i / (n * nz);
    // lev is (n, nlev_lanes, nz); the output lanes are the first nout rows.
    out = lev[(c * nlev_lanes + j) * nz + k];
"""


def gather_kernel(cp):
    params = ", ".join(f"raw T l{j}" for j in range(15))
    return cp.ElementwiseKernel(
        params + ", int64 nz, int64 lanes, int64 ncol, int64 lo",
        "T out", _GATHER, "arwen_gf_gather",
    )


def scatter_kernel(cp):
    return cp.ElementwiseKernel(
        "raw T lev, int64 n, int64 nz, int64 nlev_lanes", "T out", _SCATTER, "arwen_gf_scatter",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nz", type=int, default=40)
    parser.add_argument("--ny", type=int, default=384)
    parser.add_argument("--nx", type=int, default=768)
    parser.add_argument("--chunk", type=int, default=131072)
    parser.add_argument("--lanes", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    import cupy as cp
    from cupyx.profiler import benchmark

    nz, ny, nx = args.nz, args.ny, args.nx
    ncol = ny * nx
    n = min(args.chunk, ncol)
    lo = 0
    rng = np.random.default_rng(7)
    lanes = [
        cp.asarray(rng.standard_normal((nz, ny, nx)).astype(np.float32))
        for _ in range(args.lanes)
    ]
    lane_args = lanes + [lanes[0]] * (15 - len(lanes))

    def cols(a):
        return cp.ascontiguousarray(a.reshape(nz, ncol)[:, lo:lo + n].T, dtype=cp.float32)

    def pack_two_copies():
        lvin = cp.empty((n, args.lanes, nz), dtype=cp.float32)
        for j in range(args.lanes):
            lvin[:, j, :] = cols(lanes[j])
        return lvin

    gather = gather_kernel(cp)

    def pack_gather():
        lvin = cp.empty((n, args.lanes, nz), dtype=cp.float32)
        gather(*lane_args, np.int64(nz), np.int64(args.lanes), np.int64(ncol), np.int64(lo), lvin)
        return lvin

    a = pack_two_copies()
    b = pack_gather()
    same_in = bool(np.array_equal(cp.asnumpy(a), cp.asnumpy(b)))
    print(f"pack: gather {'returns the two-copy bits' if same_in else 'MOVES BITS'}")

    # Outputs: lev (n, 16, nz) -> four (nz, ncol) lanes.
    nout, nlev_lanes = 4, 16
    lev = cp.asarray(rng.standard_normal((n, nlev_lanes, nz)).astype(np.float32))

    def unpack_two_copies():
        out = [cp.empty((nz, ncol), dtype=cp.float32) for _ in range(nout)]
        for j in range(nout):
            out[j][:, lo:lo + n] = lev[:, j, :].T
        return out

    scatter = scatter_kernel(cp)

    def unpack_scatter():
        block = cp.empty((nout, nz, n), dtype=cp.float32)
        scatter(lev, np.int64(n), np.int64(nz), np.int64(nlev_lanes), block)
        out = [cp.empty((nz, ncol), dtype=cp.float32) for _ in range(nout)]
        for j in range(nout):
            out[j][:, lo:lo + n] = block[j]
        return out

    c = unpack_two_copies()
    d = unpack_scatter()
    same_out = all(bool(np.array_equal(cp.asnumpy(x), cp.asnumpy(y))) for x, y in zip(c, d))
    print(f"unpack: scatter {'returns the two-copy bits' if same_out else 'MOVES BITS'}")

    report = {"nz": nz, "ny": ny, "nx": nx, "chunk": n, "lanes": args.lanes,
              "pack_same_bits": same_in, "unpack_same_bits": same_out, "timing_ms": {}}
    for name, fn in (("pack_two_copies", pack_two_copies), ("pack_gather", pack_gather),
                     ("unpack_two_copies", unpack_two_copies), ("unpack_scatter", unpack_scatter)):
        result = benchmark(fn, n_repeat=args.repeats, n_warmup=3)
        gpu = np.asarray(result.gpu_times).reshape(-1) * 1.0e3
        report["timing_ms"][name] = {"gpu_ms_min": float(gpu.min()), "gpu_ms_mean": float(gpu.mean())}
        print(f"{name}: {gpu.min():.2f} / {gpu.mean():.2f} ms (min / mean of {args.repeats})")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    return 0 if (same_in and same_out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
