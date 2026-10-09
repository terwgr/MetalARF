"""``woof global fetch-analysis``: the one object a global cold start needs.

The route a reader takes is fetch, run, export, render, and the FIRST of the
four used to belong to a different program: `woof fetch --source gdas
--hours 0 --mode full-file`, with two flags that are not that door's defaults
and both of which decide whether the run works at all.  The engine's own
acquisition door prints what comes next after a whole-globe GDAS fetch --
`GDAS_GLOBAL_ANALYSIS_MAPPING_ID` and `gdas_fetch_epilogue` in
`woof.fetch` -- and a PUBLISHED engine carries neither, so on a plain `pip
install woof woof global` the first command of the route ended in a full
stop that says this WOOF has no regional GDAS route, and nothing at all
about the model the reader installed.

So the route's first step is a door of this package's own.  It is the same
transport underneath -- the engine's Rust fetch route, through
`woof.globe.da_streams.fetch_analysis`, which is what `woof global da
fresh` has always used -- with the two flags bound rather than typed, and
the next step printed.

THE ENGINE IS ASKED FIRST for both names.  When a published engine carries
`GDAS_GLOBAL_ANALYSIS_MAPPING_ID` this module returns the engine's value,
and the `--print-next-step` text is the engine's `gdas_fetch_epilogue` when
the engine has one.  What this module never does is print the engine's
regional full stop as though it were this model's answer.

WHY THE NEXT STEP NAMES A SHIPPED EXPERIMENT BY NAME.  A repository-relative
config path is typed by someone who has the repository; a reader who ran `pip
install` has the experiments as package data, and a bare name resolves to
them (`woof.globe.configs_dir`).  The name is the spelling that works in
both places.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

__all__ = [
    "GDAS_GLOBAL_ANALYSIS_MAPPING_ID",
    "GDAS_GLOBAL_QUICKSTART_CONFIG",
    "GDAS_GLOBAL_QUICKSTART_DOC",
    "add_fetch_analysis_arguments",
    "analysis_mapping_id",
    "fetch_analysis_main",
    "gdas_fetch_epilogue",
]

#: The source id the shipped GDAS experiments name.  The engine's value when
#: a published engine carries it, this copy otherwise; the two are held
#: together by tests/test_arwen_global_quickstart.py.
GDAS_GLOBAL_ANALYSIS_MAPPING_ID = "gdas-global"

#: The shipped experiment the next step names, by the spelling that resolves
#: from an install as well as from a checkout.
GDAS_GLOBAL_QUICKSTART_CONFIG = "arwen_global_t255_quickstart"

#: The page that runs the whole route, in this repository.
GDAS_GLOBAL_QUICKSTART_DOC = "docs/ARWEN_GLOBAL_QUICKSTART.md"

#: The object a whole-globe GDAS fetch lands, by the transport's own format
#: string (`woof.fetch._fetch_gfs_fullfile_locked` names each object
#: `f"{prefix}.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}"`).
ANALYSIS_OBJECT = "gdas.t{hour:02d}z.pgrb2.0p25.f000"


def analysis_mapping_id() -> str:
    """The GDAS global analysis source id: the engine's, or the carried one."""

    try:
        from woof import fetch as engine_fetch
    except Exception:  # pragma: no cover - a broken engine install
        return GDAS_GLOBAL_ANALYSIS_MAPPING_ID
    value = getattr(engine_fetch, "GDAS_GLOBAL_ANALYSIS_MAPPING_ID", None)
    return str(value) if value else GDAS_GLOBAL_ANALYSIS_MAPPING_ID


def gdas_fetch_epilogue(*, out, cycle: dt.datetime,
                        whole_globe_analysis: bool) -> list[str]:
    """What a completed GDAS fetch says about what comes next.

    The engine's own, when a published engine carries it; this text
    otherwise.  Returned as lines rather than printed, so the hint and the
    shipped quickstart config cannot drift apart unnoticed --
    tests/test_arwen_global_quickstart.py holds the ``[initial]`` keys below
    against the ones that config really carries.

    A CROPPED fetch gets no next step, and that is not an oversight.  The
    default GDAS transport is a NOMADS area crop, and a crop cannot
    initialize a global model: `analysis_initial` measures the longitude ring
    and refuses one that does not close.  Printing `woof global run` there
    would send a reader to a refusal, which is worse than a stop.
    """

    try:
        from woof import fetch as engine_fetch
    except Exception:  # pragma: no cover - a broken engine install
        engine_fetch = None
    engine_epilogue = getattr(engine_fetch, "gdas_fetch_epilogue", None)
    if engine_epilogue is not None:
        return list(engine_epilogue(
            out=Path(out), cycle=cycle,
            whole_globe_analysis=whole_globe_analysis))

    lines: list[str] = []
    if whole_globe_analysis:
        # as_posix, because this is printed INSIDE a TOML basic string a
        # reader pastes: a Windows `data\gdas-analysis\...` carries `\g` and
        # `\.`, which TOML rejects as invalid escapes.
        analysis = (Path(out) / ANALYSIS_OBJECT.format(
            hour=cycle.hour)).as_posix()
        lines.append(
            "fetch-analysis: next: cold-start the global model from the f000 "
            "analysis.  This half is bound:")
        lines.append(
            f"  woof global run {GDAS_GLOBAL_QUICKSTART_CONFIG} "
            "--outdir out/arwen-global")
        lines.append(
            "  # CONFIG is yours, and one key of it is this fetch's "
            "output:\n"
            "  #   [initial]\n"
            "  #   mode = \"analysis\"\n"
            f"  #   analysis_grib = \"{analysis}\"\n"
            f"  #   analysis_mapping = \"{analysis_mapping_id()}\"\n"
            f"  # {GDAS_GLOBAL_QUICKSTART_CONFIG} is a ready one; the "
            "whole\n"
            f"  # route through render is {GDAS_GLOBAL_QUICKSTART_DOC}.")
    else:
        lines.append(
            "fetch-analysis: this object is a NOMADS AREA CROP and cannot "
            "cold-start a global model: the initializer measures the "
            "longitude ring and refuses one that does not close.  Re-fetch "
            "without --area for a whole-globe analysis.")
    return lines


# ------------------------------------------------------------------- the door

def add_fetch_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out", type=Path, default=Path("data/gdas-analysis"),
        help="directory the analysis object and its manifest land in")
    parser.add_argument(
        "--cycle", default=None, metavar="YYYY-MM-DDTHH",
        help="the GDAS cycle to fetch; without it, the newest published one")
    parser.add_argument(
        "--engine", default="auto", choices=("auto", "rust", "python"),
        help="which transport the engine's fetch route uses")
    parser.add_argument(
        "--quiet", action="store_true",
        help="do not print the transport's own progress lines")


def _parse_cycle(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "")
    for shape in ("%Y-%m-%dT%H", "%Y-%m-%d_%H", "%Y-%m-%dT%H:%M",
                  "%Y%m%d%H"):
        try:
            return dt.datetime.strptime(text, shape)
        except ValueError:
            continue
    raise ValueError(
        f"--cycle {value!r} is not a cycle instant; spell it "
        "YYYY-MM-DDTHH (GDAS runs at 00, 06, 12 and 18Z)")


def fetch_analysis_main(args: argparse.Namespace) -> int:
    """Fetch one whole-globe GDAS analysis and say what comes next."""

    from .da_streams import fetch_analysis

    cycle = _parse_cycle(getattr(args, "cycle", None))
    progress = (lambda *_a, **_k: None) if getattr(args, "quiet", False) else print
    record = fetch_analysis(
        Path(args.out), source="gdas", cycle=cycle,
        engine=getattr(args, "engine", "auto"), progress=progress)
    print(f"fetch-analysis: {record.path}")
    print(f"fetch-analysis: {record.bytes:,} bytes, sha256 {record.sha256}")
    print(f"fetch-analysis: cycle {record.cycle_utc}, mapping {record.mapping}")
    if record.manifest:
        print(f"fetch-analysis: manifest {record.manifest}")
    fetched = dt.datetime.fromisoformat(record.cycle_utc.replace("Z", "+00:00"))
    for line in gdas_fetch_epilogue(
            out=Path(args.out), cycle=fetched.replace(tzinfo=None),
            whole_globe_analysis=True):
        print(line)
    return 0
