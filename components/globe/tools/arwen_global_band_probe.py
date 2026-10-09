"""Does the Legendre band move bits or time on this card?

``python tools/arwen_global_band_probe.py [--truncation 255] [--nlev 40]
[--bands 32,64,128,256] [--repeats 20] [--json out.json]``

Builds one spherical-harmonic transform per band on the cupy float32
backend and, on the same random fields, compares every operation the
dycore uses (analysis, synthesis, gradient, the vector analysis and the
wind synthesis) between the first band and each other band with
numpy.array_equal, then times a stacked ``nlev``-level synthesis and
analysis per band with cupyx.profiler.benchmark (the kernel time the
stream reports, per repetition, minimum and mean over the repeats).

What it measures: whether cuBLAS's strided-batched GEMM returns the same
bits for the same per-order operands when the batch count is the band,
on the card and library this process runs on, and what one contraction
costs per band on that card.  A band that moves a bit is printed by
operation.  A band that moves no bit AT THE STACK WIDTH PROBED is not
thereby a layout choice: MEASURED 2026-09-06, the shipped stacked probe
read every band as neutral because it probed forty levels, and the same
bands move the analysis of a SINGLE plane, which is what the dycore does
to the surface pressure.  Use ``--m-sweep`` before concluding anything
about a band.  A band other than the shipped 32 carries its own identity
hash (transform.py).

``--spectral-chunks 1,2,6`` adds gate CHUNK-1, the other axis of the
same question.  The Legendre band is the contraction's BATCH count; the
spectral chunk (``dynamics.MoistHybridModel.spectral_chunk``, the width
of the field stack one transform call carries) is its M dimension, and
the measurement of record for the packed table covers N and K only.
CHUNK-1 runs the real ``MoistHybridModel._chunked`` over a stack at each
chunk width and compares every chunk against the one-field-at-a-time
reference with ``array_equal``, for the analysis and the synthesis.  A
chunk width that moves no bit makes the chunk a layout knob, which is
what lets it leave the config identity; a width that moves a bit keeps
it arithmetic-visible and every archive written under another width
keeps its own hash.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--truncation", type=int, default=255)
    parser.add_argument("--nlev", type=int, default=40)
    parser.add_argument("--bands", default="32,64,128,256")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--backend", default="cupy")
    parser.add_argument("--precision", default="float32")
    parser.add_argument(
        "--spectral-chunks", default=None,
        help="gate CHUNK-1: field-stack widths to bit compare against the "
             "one-at-a-time reference (e.g. 1,2,6)")
    parser.add_argument(
        "--stack-fields", type=int, default=12,
        help="fields in the CHUNK-1 stack (default 12: the widest stack "
             "the dycore synthesises in one call)")
    parser.add_argument(
        "--m-sweep", default=None,
        help="leading sizes to compare every band at (e.g. 1,2,4,8,40): the "
             "Legendre GEMM's M dimension, which is 1 for the one field "
             "the dycore transforms as a single plane (ln ps) and nlev for "
             "every volume")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    from woof.globe.spectral.transform import SphericalHarmonicTransform
    from woof.globe.spectral.vector import VorticityDivergenceOperator

    bands = [int(b) for b in args.bands.split(",")]
    bands = [min(b, args.truncation + 1) for b in bands]
    rng = np.random.default_rng(2026)
    transforms = {}
    for band in bands:
        transforms[band] = SphericalHarmonicTransform.create(
            args.truncation, backend=args.backend, precision=args.precision,
            legendre_band=band,
        )
    first = transforms[bands[0]]
    xp = first.backend.xp
    nlat, nlon = first.grid.shape
    t1 = args.truncation + 1
    field = xp.asarray(rng.standard_normal((args.nlev, nlat, nlon)).astype(np.float32))
    # A resolved spectral state: analysed from a random grid field, then projected.
    coeff = first.project(first.forward(field))
    u = xp.asarray(rng.standard_normal((args.nlev, nlat, nlon)).astype(np.float32))
    v = xp.asarray(rng.standard_normal((args.nlev, nlat, nlon)).astype(np.float32))

    def operations(transform):
        vector = VorticityDivergenceOperator(transform)
        east, north = transform.gradient(coeff)
        zeta, div = vector.vordiv_from_wind(u, v)
        uu, vv = vector.wind_from_vordiv(zeta, div)
        return {
            "forward": transform.forward(field),
            "inverse": transform.inverse(coeff),
            "gradient_east": east,
            "gradient_north": north,
            "vordiv_zeta": zeta,
            "vordiv_div": div,
            "wind_u": uu,
            "wind_v": vv,
        }

    host = first.backend.to_numpy
    reference = {name: host(value) for name, value in operations(first).items()}
    report = {
        "truncation": args.truncation, "nlev": args.nlev, "bands": bands,
        "backend": args.backend, "grid": [int(nlat), int(nlon)],
        "identity": {}, "timing_ms": {},
    }
    for band, transform in transforms.items():
        got = {name: host(value) for name, value in operations(transform).items()}
        moved = [name for name in reference if not np.array_equal(reference[name], got[name])]
        report["identity"][str(band)] = {"same_bits_as_first_band": not moved, "moved": moved}
        print(f"band {band}: {'same bits' if not moved else 'MOVED ' + ', '.join(moved)} against band {bands[0]}")
    if args.backend == "cupy":
        from cupyx.profiler import benchmark

        for band, transform in transforms.items():
            row = {}
            for name, fn in (
                ("synthesis", lambda t=transform: t.inverse(coeff)),
                ("analysis", lambda t=transform: t.forward(field)),
                ("wind", lambda t=transform: VorticityDivergenceOperator(t).wind_from_vordiv(coeff, coeff)),
                ("vordiv", lambda t=transform: VorticityDivergenceOperator(t).vordiv_from_wind(u, v)),
            ):
                result = benchmark(fn, n_repeat=args.repeats, n_warmup=3)
                gpu = np.asarray(result.gpu_times).reshape(-1) * 1.0e3
                cpu = np.asarray(result.cpu_times).reshape(-1) * 1.0e3
                row[name] = {
                    "gpu_ms_min": float(gpu.min()), "gpu_ms_mean": float(gpu.mean()),
                    "cpu_ms_mean": float(cpu.mean()),
                }
            report["timing_ms"][str(band)] = row
            print(
                f"band {band}: " + ", ".join(
                    f"{name} {row[name]['gpu_ms_min']:.2f}/{row[name]['gpu_ms_mean']:.2f} ms (min/mean)"
                    for name in row
                )
            )
            scratch = int(transform._basis.scratch_nbytes) if hasattr(transform._basis, "scratch_nbytes") else None
            report["timing_ms"][str(band)]["basis_scratch_bytes"] = scratch
    else:
        for band, transform in transforms.items():
            started = time.perf_counter()
            for _ in range(3):
                transform.inverse(coeff)
            report["timing_ms"][str(band)] = {"synthesis": {"cpu_ms_mean": (time.perf_counter() - started) / 3 * 1e3}}
    if args.m_sweep:
        report["m_sweep"] = _m_sweep_gate(transforms, bands, args, rng)
    if args.spectral_chunks:
        report["chunk_identity"] = _chunk_gate(
            first, args, rng, report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    moved_any = any(
        not row["same_bits_as_first_band"]
        for row in report["identity"].values()
    ) or any(
        not row["same_bits_as_chunk_one"]
        for row in report.get("chunk_identity", {}).get("chunks", {}).values()
    )
    return 0 if not moved_any else 1


def _m_sweep_gate(transforms, bands, args, rng) -> dict:
    """Is the Legendre band a layout choice at every stack width?

    The band changes the packed table's N and K and, under cupy, the
    strided-batched GEMM's batch count.  M is the leading size of the
    stack being transformed, and it is 1 for the ONE field the dycore
    analyses as a single plane, the surface pressure.  This sweeps M and
    reports, per band and per M, whether the analysis and the synthesis
    return the reference band's bits.
    """
    reference = transforms[bands[0]]
    xp = reference.backend.xp
    host = reference.backend.to_numpy
    nlat, nlon = reference.grid.shape
    leads = [int(m) for m in str(args.m_sweep).split(",") if m]
    plane = (11.5 + 0.01 * rng.standard_normal((nlat, nlon))).astype(
        reference.backend.float_dtype)
    out = {
        "gate": "BAND-M",
        "reference_band": bands[0],
        "truncation": args.truncation,
        "precision": args.precision,
        "leads": leads,
        "rows": [],
    }
    for lead in leads:
        field = xp.asarray(
            np.broadcast_to(plane, (lead, nlat, nlon)).copy())
        ref_f = host(reference.forward(field))
        ref_i = host(reference.inverse(reference.project(reference.forward(field))))
        for band, transform in transforms.items():
            if band == bands[0]:
                continue
            got_f = host(transform.forward(field))
            got_i = host(transform.inverse(
                transform.project(transform.forward(field))))
            row = {
                "M": lead,
                "band": band,
                "analysis_bit_exact": bool(np.array_equal(ref_f, got_f)),
                "synthesis_bit_exact": bool(np.array_equal(ref_i, got_i)),
                "analysis_max_abs": float(np.abs(ref_f - got_f).max()),
                "synthesis_max_abs": float(np.abs(ref_i - got_i).max()),
            }
            out["rows"].append(row)
            print("M={0} band {1}: analysis {2} ({3:.3e}), synthesis {4} "
                  "({5:.3e}) against band {6}".format(
                      lead, band,
                      "same bits" if row["analysis_bit_exact"] else "MOVED",
                      row["analysis_max_abs"],
                      "same bits" if row["synthesis_bit_exact"] else "MOVED",
                      row["synthesis_max_abs"], bands[0]))
        del field
    out["passed"] = all(
        row["analysis_bit_exact"] and row["synthesis_bit_exact"]
        for row in out["rows"])
    return out



def _chunk_gate(transform, args, rng, report) -> dict:
    """Gate CHUNK-1: the field-stack width against one field at a time.

    Runs the REAL ``MoistHybridModel._chunked`` (the unbound method on a
    shim carrying only the two attributes it reads) so the loop under
    test is the loop the dycore runs, not a transcription of it.
    """
    from woof.globe.dynamics import MoistHybridModel

    class _Shim:
        def __init__(self, transform, chunk):
            self.transform = transform
            self.spectral_chunk = chunk

    xp = transform.backend.xp
    host = transform.backend.to_numpy
    nlat, nlon = transform.grid.shape
    nfields = int(args.stack_fields)
    grid_stack = xp.asarray(
        rng.standard_normal((nfields, args.nlev, nlat, nlon)).astype(
            transform.backend.float_dtype))
    spectral_stack = transform.project(
        MoistHybridModel._chunked(_Shim(transform, 1), transform.forward,
                                  grid_stack))
    chunks = [int(c) for c in str(args.spectral_chunks).split(",") if c]
    if 1 not in chunks:
        chunks = [1] + chunks
    out = {
        "gate": "CHUNK-1",
        "truncation": args.truncation,
        "nlev": args.nlev,
        "stack_fields": nfields,
        "grid": [int(nlat), int(nlon)],
        "precision": args.precision,
        "chunks": {},
    }
    reference = None
    for chunk in chunks:
        shim = _Shim(transform, chunk)
        got = {
            "forward": host(MoistHybridModel._chunked(
                shim, transform.forward, grid_stack)),
            "inverse": host(MoistHybridModel._chunked(
                shim, transform.inverse, spectral_stack)),
        }
        if reference is None:
            reference = got
            out["chunks"][str(chunk)] = {
                "same_bits_as_chunk_one": True, "moved": []}
            print(f"spectral_chunk {chunk}: reference (one field at a time)")
            continue
        moved = [name for name in reference
                 if not np.array_equal(reference[name], got[name])]
        row = {"same_bits_as_chunk_one": not moved, "moved": moved}
        for name in moved:
            row[f"max_abs_{name}"] = float(
                np.abs(reference[name] - got[name]).max())
        out["chunks"][str(chunk)] = row
        print(f"spectral_chunk {chunk}: "
              + ("same bits" if not moved else "MOVED " + ", ".join(moved))
              + " against chunk 1")
    out["passed"] = all(
        row["same_bits_as_chunk_one"] for row in out["chunks"].values())
    return out



if __name__ == "__main__":
    raise SystemExit(main())
