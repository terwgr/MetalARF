"""Does the Fourier waist return the resident transform's bits?  (WAIST-1, FFT-1)

``python tools/arwen_global_waist_probe.py [--truncations 255,383,533]
[--bands 1,2,4,8,16,32] [--levels 40] [--backend cupy]
[--precision float32] [--json out.json]``

The scale-out design cuts grid space into latitude bands and joins it to
a whole spectral space at a full-latitude Fourier buffer -- the waist.
Every capacity claim downstream of that (the band pipeline, the host
spill, the second card) inherits its bit-identity from this one seam, so
this probe holds the seam against the resident transform ON THE CARD, at
the truncations the model runs and at every leading shape the dycore
carries into a transform.

Four comparisons per (truncation, stack shape, band count), each an
exact ``array_equal`` and, when it fails, a maximum ULP distance and the
worst absolute difference:

* ``analysis`` -- ``contract_waist(fourier_waist(f, bands))`` against
  ``transform.forward(f)``.  Gate WAIST-1's forward half.
* ``synthesis`` -- ``waist_to_grid(contract_to_waist(c, bands))``
  against ``transform.inverse(c)``.  Gate WAIST-1's inverse half.
* ``band_drain`` -- the same waist drained one band at a time through
  ``waist_band_to_grid`` and concatenated, which is the call the band
  pipeline makes and the one that would show a stale half spectrum
  between bands (cuFFT's complex-to-real transform destroys its input).
* ``vector`` -- ``vordiv_from_wind`` through the shared waist against
  the one-band answer, for the stacked wind pairs the momentum block
  and the scalar advection transform.
* ``legacy_head`` / ``legacy_tail`` -- the analysis head and the
  synthesis tail AS THEY WERE WRITTEN BEFORE THE SPLIT, transcribed
  into this file, against the waist that replaced them.  The four
  comparisons above hold the band counts against each other and
  against the transform as it is now; these two hold the transform as
  it is now against the transform as it was, which is the claim a
  reader of the design actually wants.

The band schedule is the transform's own ``latitude_band_edges``, so the
band widths measured here are the band widths the pipeline will use.

Peak device bytes are recorded per leg through the model's own allocator
hook, so the memory the waist saves is a number and not a claim.  The
cuFFT plan cache is CLEARED before every measured leg: CuPy keeps up to
sixteen plans per device and each cached plan holds its work area from
the same pool, 1.15 GiB per six-field shape at T533 float32
(``device_memory.disable_fft_plan_cache`` records the measurement), so
a probe that sweeps band counts without clearing it reads the cache
filling up rather than the transform.  It read exactly that at T533
before the clear was added: the peak ROSE from 2.243 to 2.995 GiB
between one band and sixteen and then fell again when the cache
evicted.  Each leg therefore measures a cold plan cache, which is
comparable across band counts and still charges the plans that band
count actually needs.
"""
from __future__ import annotations

import argparse
import json
import platform

import numpy as np

from woof.globe.spectral.transform import (
    SphericalHarmonicTransform,
    latitude_band_edges,
)
from woof.globe.spectral.vector import VorticityDivergenceOperator

GIB = 2**30

#: Leading shapes a grid field carries into the transform in
#: ``MoistHybridModel.step``, read off the call sites: a single plane
#: (``forward(log ps)``), a levelled field (``inverse(divergence)``,
#: ``forward(bernoulli)``), a spectral chunk of the stacked synthesis
#: (``_chunked``, up to ``spectral_chunk`` rows), the vector pair
#: (``vordiv_from_wind``, ``gradient(stack([psi, chi]))``) and the
#: advection's stacked flux pair (``_scalar_tendency``).  ``F`` is the
#: field count, ``L`` the level count.
STACKS = ("plane", "L", "1L", "2L", "6L", "2x2L", "2x6L")

#: What fits beside a resident Legendre table set at each truncation.
#: T533's three float32 tables are 1.352 GiB (MEASURED 2026-09-06), and a
#: six-field forty-level analysis at T533 carries about 4 GiB of operands
#: and results, so the widest stacks are run where the card has room.
DEFAULT_STACKS = {
    255: ("plane", "L", "1L", "2L", "6L", "2x2L", "2x6L"),
    383: ("plane", "L", "2L", "6L", "2x2L"),
    533: ("plane", "L", "2L", "6L"),
    799: ("plane", "L", "2L"),
}


def stack_lead(name: str, levels: int) -> tuple[int, ...]:
    table = {
        "plane": (),
        "L": (levels,),
        "1L": (1, levels),
        "2L": (2, levels),
        "6L": (6, levels),
        "2x2L": (2, 2, levels),
        "2x6L": (2, 6, levels),
    }
    if name not in table:
        raise ValueError(f"unknown stack {name!r}; known are {sorted(table)}")
    return table[name]


def _ulp(a: np.ndarray, b: np.ndarray) -> float:
    """Maximum ULP distance between two same-dtype float arrays."""
    if a.dtype.kind == "c":
        return max(_ulp(np.ascontiguousarray(a.real), np.ascontiguousarray(b.real)),
                   _ulp(np.ascontiguousarray(a.imag), np.ascontiguousarray(b.imag)))
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


def legacy_analysis_head(transform, field):
    """``(rfft(f) / nlon)[..., :T+1]``: the analysis head before the split.

    Transcribed from ``SphericalHarmonicTransform._analyze`` at tip
    ca57bff43, the expression the contraction was handed.  It is here so
    the probe can hold the waist against the code it replaced without a
    second checkout.
    """
    xp = transform.backend.xp
    f = xp.asarray(field, dtype=transform.backend.float_dtype)
    return (xp.fft.rfft(f, axis=-1) / transform.grid.nlon)[
        ..., : transform.truncation + 1
    ]


def legacy_synthesis_tail(transform, values):
    """The synthesis tail before the split, from the contraction's output.

    Transcribed from ``SphericalHarmonicTransform._synthesize`` at tip
    ca57bff43: a full-width zero half spectrum, the nlon scaling written
    into its retained orders, one whole inverse real FFT, one cast.
    """
    xp = transform.backend.xp
    spectrum = xp.zeros(
        (*values.shape[:-2], transform.grid.nlat, transform.grid.nlon // 2 + 1),
        dtype=transform.backend.complex_dtype,
    )
    xp.multiply(
        transform.grid.nlon, values, out=spectrum[..., : transform.truncation + 1]
    )
    if transform.backend.name == "cupy":
        from cupyx.scipy import fft as device_fft

        grid = device_fft.irfft(
            spectrum, n=transform.grid.nlon, axis=-1, overwrite_x=True
        )
    else:
        grid = xp.fft.irfft(spectrum, n=transform.grid.nlon, axis=-1)
    del spectrum
    return grid.astype(transform.backend.float_dtype, copy=False)


def _verdict(vector) -> str:
    """The vector seam's verdict, or ``n/a`` where the stack is not a pair."""
    if vector is None:
        return "n/a"
    if vector["zeta"]["bit_exact"] and vector["divergence"]["bit_exact"]:
        return "exact"
    return "MOVED"


def _legacy_verdict(row) -> str:
    """The pre-split comparison's verdict, or ``n/a`` where it did not run."""
    if "legacy_head" not in row:
        return "n/a"
    if row["legacy_head"]["bit_exact"] and row["legacy_tail"]["bit_exact"]:
        return "exact"
    return "MOVED"


class _Peak:
    """Per-leg peak of the live bytes of the pool this run's allocator spends."""

    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        self.tracker = None

    def __enter__(self):
        if self.backend_name == "cupy":
            import cupy as cp

            from woof.globe.device_memory import (
                DevicePeakTracker,
                installed_pool,
            )

            # A cached cuFFT plan holds its work area from this pool, so
            # a sweep that leaves the cache filling up measures the cache.
            cp.fft.config.get_plan_cache().clear()
            cp.get_default_memory_pool().free_all_blocks()
            self.tracker = DevicePeakTracker(installed_pool(cp)).install()
        return self

    def __exit__(self, *exc):
        if self.tracker is not None:
            self.tracker.uninstall()
        return False

    @property
    def gib(self):
        if self.tracker is None:
            return None
        return round(self.tracker.peak_used_bytes / GIB, 3)


def run_truncation(
    truncation: int,
    *,
    bands: list[int],
    levels: int,
    stacks: tuple[str, ...],
    backend: str,
    precision: str,
    seed: int = 0,
) -> dict:
    resident = SphericalHarmonicTransform.create(
        truncation, backend=backend, precision=precision
    )
    host = resident.backend.to_numpy
    xp = resident.backend.xp
    grid = resident.grid
    rng = np.random.default_rng(seed)
    report = {
        "truncation": truncation,
        "nlat": grid.nlat,
        "nlon": grid.nlon,
        "backend": backend,
        "precision": precision,
        "levels": levels,
        "legendre_table_gib": round(
            sum(resident.legendre_table_nbytes.values()) / GIB, 4
        ),
        "identity_hash": resident.identity_hash,
        "cases": [],
    }
    # The band count reaches no table and no contraction, so the
    # comparisons run it through the resident transform's own per-call
    # argument -- the same code path the transform field selects.
    # Building a transform per band count here would re-run the float64
    # Legendre recurrence and its per-order Gram solve on the HOST for
    # every one, which on a loaded CPU host is most of the wall clock and
    # none of the measurement, and holding them would put a table set per
    # band count beside the peak being read.  That the identity hash is
    # the same at every band count is a property of the identity payload
    # rather than of a truncation, and it is asserted at ten band counts
    # in tests/test_global_spectral_fourier_waist.py; recorded here is
    # that the payload does not carry the band count at all.
    identity_carries_the_band_count = "latitude_bands" in resident.identity
    for name in stacks:
        lead = stack_lead(name, levels)
        field = resident.backend.asarray(
            rng.normal(size=(*lead, grid.nlat, grid.nlon)),
            dtype=resident.backend.float_dtype,
        )
        reference_forward = resident.forward(field)
        reference_inverse = resident.inverse(reference_forward)
        pair = None
        reference_vector = None
        if lead and lead[0] == 2:
            pair = field
            operator = VorticityDivergenceOperator(resident)
            reference_vector = operator.vordiv_from_wind(pair[0], pair[1])
        waist_bytes = int(
            np.prod((*lead, grid.nlat, truncation + 1))
            * np.dtype(resident.backend.complex_dtype).itemsize
        )
        grid_bytes = int(
            np.prod((*lead, grid.nlat, grid.nlon))
            * np.dtype(resident.backend.float_dtype).itemsize
        )
        for count in bands:
            edges = latitude_band_edges(grid.nlat, count)
            banded = resident
            row = {
                "stack": name,
                "lead": list(lead),
                "bands": count,
                "band_rows": [b - a for a, b in edges],
                "waist_gib": round(waist_bytes / GIB, 4),
                "grid_gib": round(grid_bytes / GIB, 4),
                "identity_unchanged": not identity_carries_the_band_count,
            }
            with _Peak(backend) as peak:
                got = banded.contract_waist(
                    banded.fourier_waist(field, bands=count)
                )
            row["analysis"] = _compare(reference_forward, got, host)
            row["analysis_peak_gib"] = peak.gib
            del got

            with _Peak(backend) as peak:
                got = banded.waist_to_grid(
                    banded.contract_to_waist(reference_forward, bands=count)
                )
            row["synthesis"] = _compare(reference_inverse, got, host)
            row["synthesis_peak_gib"] = peak.gib
            del got

            waist = banded.contract_to_waist(reference_forward, bands=count)
            pieces = [
                banded.waist_band_to_grid(waist, r0, r1) for r0, r1 in edges
            ]
            drained = xp.concatenate(pieces, axis=-2)
            del pieces, waist
            row["band_drain"] = _compare(reference_inverse, drained, host)
            del drained

            if count == bands[0]:
                # The transform as it is now against the transform as it
                # was.  Once per stack: neither side reads a band count.
                row["legacy_head"] = _compare(
                    legacy_analysis_head(resident, field),
                    banded.fourier_waist(field, bands=1).values,
                    host,
                )
                waist = banded.contract_to_waist(reference_forward, bands=1)
                legacy = legacy_synthesis_tail(resident, waist.values)
                row["legacy_tail"] = _compare(
                    legacy, banded.waist_to_grid(waist), host
                )
                del waist, legacy
            if reference_vector is not None:
                operator = VorticityDivergenceOperator(banded)
                # vordiv_from_wind's own two halves, with the band count
                # on the fill: the call the dycore's scalar advection
                # makes at dynamics.py:692.
                zeta, divergence = operator._vordiv_from_fourier(
                    operator._wind_fourier(
                        xp.stack([pair[0], pair[1]]), bands=count
                    )
                )
                row["vector"] = {
                    "zeta": _compare(reference_vector[0], zeta, host),
                    "divergence": _compare(
                        reference_vector[1], divergence, host
                    ),
                }
                del zeta, divergence
            report["cases"].append(row)
            print(
                f"  T{truncation} {name:>5} B={count:<4} "
                f"analysis={'exact' if row['analysis']['bit_exact'] else 'MOVED'} "
                f"synthesis={'exact' if row['synthesis']['bit_exact'] else 'MOVED'} "
                f"drain={'exact' if row['band_drain']['bit_exact'] else 'MOVED'} "
                f"legacy={_legacy_verdict(row)} "
                f"vector={_verdict(row.get('vector'))} "
                f"peak={row['analysis_peak_gib']}/{row['synthesis_peak_gib']} GiB",
                flush=True,
            )
            if backend == "cupy":
                import cupy as cp

                cp.get_default_memory_pool().free_all_blocks()
        del field, reference_forward, reference_inverse, pair, reference_vector
        if backend == "cupy":
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()
    return report


def _flatten(report: dict) -> list[dict]:
    out = []
    for truncation in report["truncations"]:
        for case in truncation["cases"]:
            for key in ("analysis", "synthesis", "band_drain",
                        "legacy_head", "legacy_tail"):
                if key in case:
                    out.append({"op": key, **case[key]})
            if "vector" in case:
                out.append({"op": "vector_zeta", **case["vector"]["zeta"]})
                out.append(
                    {"op": "vector_divergence", **case["vector"]["divergence"]}
                )
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--truncations", default="255,383,533")
    parser.add_argument("--bands", default="1,2,4,8,16,32")
    parser.add_argument("--levels", type=int, default=40)
    parser.add_argument(
        "--stacks",
        default=None,
        help="comma-separated subset of " + ",".join(STACKS),
    )
    parser.add_argument("--backend", default="cupy")
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    truncations = [int(x) for x in args.truncations.split(",") if x]
    bands = [int(x) for x in args.bands.split(",") if x]
    report = {
        "gate": "WAIST-1 / FFT-1",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "bands": bands,
        "levels": args.levels,
        "truncations": [],
    }
    if args.backend == "cupy":
        import cupy as cp

        device = cp.cuda.Device()
        free, total = cp.cuda.runtime.memGetInfo()
        report["device"] = {
            "name": cp.cuda.runtime.getDeviceProperties(device.id)["name"].decode(),
            "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "free_mib_at_start": free // (1024 * 1024),
            "total_mib": total // (1024 * 1024),
        }
        print(
            f"device: {report['device']['name']}, "
            f"{report['device']['free_mib_at_start']} of "
            f"{report['device']['total_mib']} MiB free",
            flush=True,
        )
    for truncation in truncations:
        stacks = (
            tuple(x for x in args.stacks.split(",") if x)
            if args.stacks
            else DEFAULT_STACKS.get(truncation, STACKS)
        )
        print(f"T{truncation}: stacks {','.join(stacks)}", flush=True)
        report["truncations"].append(
            run_truncation(
                truncation,
                bands=bands,
                levels=args.levels,
                stacks=stacks,
                backend=args.backend,
                precision=args.precision,
                seed=args.seed,
            )
        )
    rows = _flatten(report)
    exact = sum(1 for r in rows if r["bit_exact"])
    report["comparisons"] = len(rows)
    report["bit_exact"] = exact
    report["identity_unchanged"] = all(
        case["identity_unchanged"]
        for t in report["truncations"]
        for case in t["cases"]
    )
    report["verdict"] = (
        "WAIST-1 PASSES" if exact == len(rows) else "WAIST-1 FAILS"
    )
    print(
        f"\n{report['verdict']}: {exact} of {len(rows)} comparisons bit-exact; "
        f"identity hash unchanged at every band count: "
        f"{report['identity_unchanged']}"
    )
    if exact != len(rows):
        for t in report["truncations"]:
            for case in t["cases"]:
                for key in ("analysis", "synthesis", "band_drain",
                            "legacy_head", "legacy_tail"):
                    if key in case and not case[key]["bit_exact"]:
                        print(
                            f"  MOVED T{t['truncation']} {case['stack']} "
                            f"B={case['bands']} {key}: "
                            f"{case[key].get('max_ulp')} ulp, "
                            f"{case[key].get('max_abs')} abs"
                        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"wrote {args.json}")
    return 0 if exact == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
