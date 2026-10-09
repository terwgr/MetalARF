"""Measure the streamed spherical-harmonic analysis on a native Gaussian grid.

Builds a synthetic field on an ``nlat x nlon`` Gaussian grid, analyses it
at the requested truncation through ``forward_streaming`` (no resident
Legendre table) and prints one JSON line with the peak resident set,
wall time and a correctness check: the synthetic field is a sum of a few
known harmonics, so the coefficients that come back must carry all of
its energy in the orders it was built from, and the round trip through
``inverse_streaming`` must reproduce the field.

Usage::

    GPUWM_NO_LOCAL_GPU=1 python tools/global_spectral_streaming_probe.py \
        --truncation 1534 --nlat 1536 --nlon 3072 --order-chunk 32

``--backend cupy --precision float32`` runs the same probe on a node's
card (the process must not carry GPUWM_NO_LOCAL_GPU there); the device
peak is then reported from the cupy memory pool.  Peak RSS is the OS's
figure for the whole process (psutil), the number the OOM killer acts
on, not a tracemalloc figure.

Measured 2026-09-01 (numpy 2.2.6, 16-core Zen 5, 96 GB host, chunk 32,
float64, run from the package root with PYTHONPATH set to it):

    T767  on 1536x3072: forward_streaming 78.7 s, peak RSS 0.986 GiB,
          round trip 1.7e-15 (a standalone re-run: 38.6 s)
    T1534 on 1536x3072: forward_streaming 159.0 s, peak RSS 1.892 GiB,
          round trip 1.6e-15

against the resident T767 build's 7.03 GiB and the dense T1534 build
that could not be held at all.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from woof.globe.spectral.grid import GaussianGrid
from woof.globe.spectral.transform import SphericalHarmonicTransform

#: (degree, order, amplitude) of the harmonics the synthetic field carries.
#: Low orders and a high one, so the probe reads chunks at both ends.
SYNTHETIC_MODES = ((3, 0, 1.0), (7, 5, 0.5), (40, 29, 0.25))


def peak_rss_bytes() -> int:
    import psutil

    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", None) or getattr(info, "rss"))


def synthetic_field(transform: SphericalHarmonicTransform, modes) -> np.ndarray:
    """A real field made of ``modes``, synthesised by the streamed inverse."""
    coeff = transform.zeros()
    for n, m, amplitude in modes:
        if n > transform.truncation:
            continue
        coeff[n, m] = amplitude * (1.0 + (0.0 if m == 0 else 0.5j))
    return transform.backend.to_numpy(transform.inverse_streaming(coeff))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--truncation", type=int, required=True)
    parser.add_argument("--nlat", type=int, default=1536)
    parser.add_argument("--nlon", type=int, default=3072)
    parser.add_argument("--order-chunk", type=int, default=32)
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--precision", default="float64")
    parser.add_argument(
        "--roundtrip", action="store_true",
        help="also synthesise the coefficients back and report the field error",
    )
    args = parser.parse_args(argv)

    t_start = time.perf_counter()
    grid = GaussianGrid.for_shape(args.nlat, args.nlon)
    transform = SphericalHarmonicTransform.create(
        args.truncation, grid=grid, backend=args.backend,
        precision=args.precision, legendre_band=args.order_chunk, streaming=True,
    )
    field = synthetic_field(transform, SYNTHETIC_MODES)
    t_synth = time.perf_counter() - t_start
    rss_before = peak_rss_bytes()

    t0 = time.perf_counter()
    coeff = transform.forward_streaming(field, order_chunk=args.order_chunk)
    transform.backend.synchronize()
    forward_s = time.perf_counter() - t0
    rss_after = peak_rss_bytes()

    c = transform.backend.to_numpy(coeff)
    energy = np.abs(c) ** 2
    orders = sorted({m for _, m, _ in SYNTHETIC_MODES})
    in_modes = sum(float(energy[:, m].sum()) for m in orders)
    recovered = {
        f"{n},{m}": [float(c[n, m].real), float(c[n, m].imag)]
        for n, m, _ in SYNTHETIC_MODES if n <= transform.truncation
    }
    report = {
        "truncation": transform.truncation,
        "nlat": grid.nlat,
        "nlon": grid.nlon,
        "order_chunk": args.order_chunk,
        "backend": args.backend,
        "precision": args.precision,
        "synthesis_s": round(t_synth, 3),
        "forward_streaming_s": round(forward_s, 3),
        "peak_rss_bytes_after_synthesis": rss_before,
        "peak_rss_bytes_after_forward": rss_after,
        "peak_rss_gib": round(rss_after / 2 ** 30, 3),
        "streaming_working_bytes": transform.streaming_working_bytes(args.order_chunk),
        "resident_table_bytes": transform.legendre_table_nbytes,
        "energy_fraction_in_synthetic_orders": in_modes / float(energy.sum()),
        "recovered_coefficients": recovered,
    }
    if args.backend == "cupy":
        pool = transform.backend.xp.get_default_memory_pool()
        report["device_pool_total_bytes"] = int(pool.total_bytes())
    if args.roundtrip:
        t0 = time.perf_counter()
        back = transform.backend.to_numpy(
            transform.inverse_streaming(coeff, order_chunk=args.order_chunk)
        )
        report["inverse_streaming_s"] = round(time.perf_counter() - t0, 3)
        report["roundtrip_relative_linf"] = float(
            np.max(np.abs(back - field)) / np.max(np.abs(field))
        )
        report["peak_rss_gib"] = round(peak_rss_bytes() / 2 ** 30, 3)
    json.dump(report, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
