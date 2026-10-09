"""Does feeding the longitude FFT band by band move a bit?  (Gate FFT-1)

``python tools/arwen_global_fft_band_probe.py [--truncations 255,383,533]
[--bands 2,4,8,16,32] [--levels 40] [--fields 1,4] [--json out.json]``

The whole scale-out claim rests on one assumption: the real FFT along
longitude is row-local, so a field cut into latitude bands and
transformed band by band returns the bits the whole-field call returns.
The transform LENGTH (``nlon``) never moves; only the batch count does,
and cuFFT is free to pick a different plan for a different batch.  This
probe measures that, on the card, before any pipeline code exists.

Four comparisons per (truncation, band count, stack shape), each an
exact ``array_equal`` and, when it fails, a maximum ULP distance:

* ``rfft`` -- ``xp.fft.rfft(field, axis=-1)`` whole against the
  concatenation over latitude bands.
* ``analysis_head`` -- the engine's own analysis expression,
  ``(rfft(f) / nlon)[..., :T+1]`` (``transform._analyze``), whole
  against banded.
* ``irfft`` -- the engine's synthesis buffer (zeros of
  ``(..., nlat, nlon//2+1)`` with ``nlon * coeff`` written into the
  retained orders) inverted whole against band by band.
* ``irfft_overwrite`` -- the same through
  ``cupyx.scipy.fft.irfft(..., overwrite_x=True)``, which is the call
  ``transform._synthesize`` actually makes on the cupy backend.

Band edges are ``floor(k * nlat / B)``, the design's band schedule, so
the band widths this probe measures are the band widths the pipeline
will use.  A band count that moves a bit is printed by operation and the
report says so; the fallback then keeps the real field whole for the FFT
only.
"""
from __future__ import annotations

import argparse
import json
import platform

import numpy as np

# The band schedule is the transform's own: two copies that drift apart
# is a divergence no bit-identity gate would catch, because each half
# would be self-consistent.
from woof.globe.spectral.transform import latitude_band_edges as band_edges


def _ulp(a: np.ndarray, b: np.ndarray) -> float:
    """Maximum ULP distance between two same-dtype float arrays."""
    if a.dtype.kind == "c":
        return max(_ulp(a.real.copy(), b.real.copy()),
                   _ulp(a.imag.copy(), b.imag.copy()))
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    kind = np.int32 if a.dtype == np.float32 else np.int64
    ia = a.view(kind).astype(np.int64)
    ib = b.view(kind).astype(np.int64)
    top = np.int64(np.iinfo(kind).min)
    ia = np.where(ia < 0, top - ia, ia)
    ib = np.where(ib < 0, top - ib, ib)
    return float(np.abs(ia - ib).max())


def _compare(whole, banded, host) -> dict:
    w = host(whole)
    b = host(banded)
    same = bool(np.array_equal(w, b))
    row = {"bit_exact": same}
    if not same:
        row["max_ulp"] = _ulp(w, b)
        row["max_abs"] = float(np.abs(w - b).max())
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--truncations", default="255,383,533")
    parser.add_argument("--bands", default="2,4,8,16,32")
    parser.add_argument("--levels", type=int, default=40)
    parser.add_argument("--fields", default="1,4")
    parser.add_argument("--backend", default="cupy")
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    from woof.globe.spectral.backend import get_backend
    from woof.globe.spectral.grid import GaussianGrid

    backend = get_backend(args.backend, args.precision)
    xp = backend.xp
    host = backend.to_numpy
    device_fft = None
    if args.backend == "cupy":
        from cupyx.scipy import fft as device_fft

    truncations = [int(t) for t in args.truncations.split(",") if t]
    bands = [int(b) for b in args.bands.split(",") if b]
    fields = [int(f) for f in args.fields.split(",") if f]
    report = {
        "gate": "FFT-1",
        "backend": args.backend,
        "precision": args.precision,
        "host": f"{platform.system().lower()}-{platform.machine().lower()}",
        "levels": args.levels,
        "fields": fields,
        "bands": bands,
        "cases": [],
    }
    if args.backend == "cupy":
        import cupy as cp

        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        report["card"] = props["name"].decode()
        report["cupy_version"] = cp.__version__
        report["cuda_runtime"] = int(cp.cuda.runtime.runtimeGetVersion())

    rng = np.random.default_rng(20260906)
    failures = []
    for truncation in truncations:
        nlat, nlon = GaussianGrid.shape_for(truncation, dealias_factor=1.5)
        t1 = truncation + 1
        for nfields in fields:
            if nfields == 1:
                shape = (args.levels, nlat, nlon)
            else:
                shape = (nfields, args.levels, nlat, nlon)
            field = xp.asarray(
                rng.standard_normal(shape).astype(backend.float_dtype))
            coeff = xp.asarray(
                (rng.standard_normal((*shape[:-2], nlat, t1))
                 + 1j * rng.standard_normal((*shape[:-2], nlat, t1))
                 ).astype(backend.complex_dtype))

            whole_rfft = xp.fft.rfft(field, axis=-1)
            whole_head = (whole_rfft / nlon)[..., :t1]

            spectrum = xp.zeros(
                (*shape[:-2], nlat, nlon // 2 + 1),
                dtype=backend.complex_dtype)
            xp.multiply(nlon, coeff, out=spectrum[..., :t1])
            whole_irfft = xp.fft.irfft(spectrum, n=nlon, axis=-1)
            whole_over = None
            if device_fft is not None:
                over_in = spectrum.copy()
                whole_over = device_fft.irfft(
                    over_in, n=nlon, axis=-1, overwrite_x=True)
                del over_in

            for band_count in bands:
                edges = band_edges(nlat, band_count)
                if len(edges) < band_count:
                    continue
                rows = [b - a for a, b in edges]
                banded_rfft = xp.empty_like(whole_rfft)
                banded_head = xp.empty_like(whole_head)
                banded_irfft = xp.empty_like(whole_irfft)
                banded_over = (None if whole_over is None
                               else xp.empty_like(whole_over))
                for a, b in edges:
                    slab = field[..., a:b, :]
                    piece = xp.fft.rfft(slab, axis=-1)
                    banded_rfft[..., a:b, :] = piece
                    banded_head[..., a:b, :] = (piece / nlon)[..., :t1]
                    del piece, slab
                    sub = xp.zeros(
                        (*shape[:-2], b - a, nlon // 2 + 1),
                        dtype=backend.complex_dtype)
                    xp.multiply(nlon, coeff[..., a:b, :], out=sub[..., :t1])
                    banded_irfft[..., a:b, :] = xp.fft.irfft(
                        sub, n=nlon, axis=-1)
                    if device_fft is not None:
                        banded_over[..., a:b, :] = device_fft.irfft(
                            sub, n=nlon, axis=-1, overwrite_x=True)
                    del sub
                case = {
                    "truncation": truncation,
                    "grid": [int(nlat), int(nlon)],
                    "stack": [int(v) for v in shape],
                    "bands": band_count,
                    "band_rows_min": int(min(rows)),
                    "band_rows_max": int(max(rows)),
                    "rfft": _compare(whole_rfft, banded_rfft, host),
                    "analysis_head": _compare(whole_head, banded_head, host),
                    "irfft": _compare(whole_irfft, banded_irfft, host),
                }
                if banded_over is not None:
                    case["irfft_overwrite"] = _compare(
                        whole_over, banded_over, host)
                moved = [
                    name for name, row in case.items()
                    if isinstance(row, dict) and not row["bit_exact"]
                ]
                case["moved"] = moved
                if moved:
                    failures.append(case)
                report["cases"].append(case)
                print(
                    "T{0} stack{1} B={2} rows {3}..{4}: {5}".format(
                        truncation, tuple(int(v) for v in shape), band_count,
                        min(rows), max(rows),
                        "same bits" if not moved
                        else "MOVED " + ",".join(moved)))
                del banded_rfft, banded_head, banded_irfft, banded_over
            del field, coeff, spectrum, whole_rfft, whole_head
            del whole_irfft, whole_over
            if args.backend == "cupy":
                import cupy as cp

                cp.get_default_memory_pool().free_all_blocks()

    report["passed"] = not failures
    report["failures"] = failures
    print("FFT-1: {0} cases, {1}".format(
        len(report["cases"]),
        "ALL BIT EXACT" if not failures
        else "{0} MOVED BITS".format(len(failures))))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
