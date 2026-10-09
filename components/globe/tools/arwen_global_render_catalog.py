"""Measure which render products a global tape can draw, and write the table.

``woof global run-plan --catalog`` has to answer "what may I put in
``render_products``" without a plan, which means without a tape.  The
renderer answers that question only against a file: its catalog is a
property of what the importer found in the store, so the same slug is
renderable on one tape and ``missing-fields`` on another.

So the answer is MEASURED here, once, against a real global tape written
by this package's own exporter, and shipped as
``woof/globe/data/render-catalog-global.json``.  The catalog door reads
that table beside the renderer's live slug list and reports the status of
every slug the table carries, ``unmeasured`` for one it does not, and the
host, date and renderer digest the table was measured on.

WHAT IT MEASURES, exactly: ``woof.rustwx.list_products`` against one
exported tape (and ``list_products_series`` across every tape given), which
is the renderer's own import-time catalog and the same call the render door
already makes before it draws.  Nothing is inferred and no slug list is
transcribed.

WHY THE TABLE TRAVELS RATHER THAN THE TAPE.  The field set a tape carries is
fixed by ``woof.globe.wrfout_export``, not by the truncation, so the answer
is a property of this package's exporter and moves only when the exporter
does.  The frame COUNT is not: windowed accumulations need more than one
stored whole-hour frame, so the series measurement is recorded beside the
single-frame one and the door reports both.

    python -m tools.arwen_global_render_catalog TAPE [TAPE ...] \
        --experiment NAME --truncation 21 --out src/arwen_global/data/render-catalog-global.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import sys


def _machine_class() -> str:
    """The KIND of machine this was measured on, never its name.

    THE BREAKAGE THIS PREVENTS, measured 2026-09-10 inside the built wheel:
    this field carried the hostname the operating system reports and the
    table travels as package data, so the shipped catalog named one machine
    to every reader of the package, beside the exact operating-system build
    it runs.  What a reader can act on is the kind of machine, because that
    is what says whether a stored measurement applies to theirs; the name
    says nothing they can use and is not theirs to have.  The operating
    system beside it is recorded as family and release for the same reason:
    a build number is a fingerprint and the release is the fact.
    """

    return f"{platform.system().lower()}-{platform.machine().lower()}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(renderer: Path, tapes: list[Path], store: Path, series: bool):
    from woof import rustwx

    if series:
        rows, summary = rustwx.list_products_series(
            renderer, tapes, store_root=store, heavy=False)
    else:
        rows, summary = rustwx.list_products(
            renderer, tapes[0], store_root=store, heavy=False)
    return ([{"name": slug, "kind": kind, "status": status, "detail": detail}
             for slug, kind, status, detail in rows], summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tapes", nargs="+", type=Path,
                        help="exported wrfout tapes, in time order")
    parser.add_argument("--experiment", required=True,
                        help="the shipped experiment the tapes were run from")
    parser.add_argument("--truncation", type=int, required=True)
    parser.add_argument("--nlat", type=int, required=True,
                        help="the export grid's latitude points")
    parser.add_argument("--nlon", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from woof import render as engine_render
    from woof import rustwx

    renderer = rustwx.find_renderer()
    if renderer is None:
        print("no rw_wrfbatch is staged; nothing can be measured",
              file=sys.stderr)
        return 1
    tapes = [Path(tape) for tape in args.tapes]
    with engine_render.scratch_store(args.out.parent / "catalog-probe") as store:
        single, single_summary = _rows(renderer, tapes, store, series=False)
    series = series_summary = None
    if len(tapes) > 1:
        with engine_render.scratch_store(
                args.out.parent / "catalog-probe") as store:
            series, series_summary = _rows(renderer, tapes, store, series=True)

    document = {
        "schema": "arwen-global-render-catalog-measurement-v1",
        "measured_utc": dt.datetime.now(dt.timezone.utc).replace(
            microsecond=0).isoformat(),
        "host": _machine_class(),
        "platform": f"{platform.system()}-{platform.release()}",
        "instrument": "woof.rustwx.list_products / list_products_series, "
                      "the renderer's own import-time catalog",
        "renderer": {
            "name": renderer.name,
            "sha256": _sha256(renderer),
            "bytes": renderer.stat().st_size,
        },
        "tape": {
            "experiment": args.experiment,
            "truncation": args.truncation,
            "export_nlat": args.nlat,
            "export_nlon": args.nlon,
            "exporter": "woof.globe.wrfout_export.export_wrfout",
            "frames_single": 1,
            "frames_series": len(tapes),
        },
        "single_frame": {"summary": single_summary, "products": single},
        "series": (None if series is None else
                   {"summary": series_summary, "products": series}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"{args.out}: {single_summary}")
    if series_summary is not None:
        print(f"{args.out}: series {series_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
