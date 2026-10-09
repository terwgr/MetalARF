"""``woof global doctor``: what is installed, what is staged, what is missing.

A user looks here before burning a run.  The report is therefore written to
answer one question per line -- can this machine execute the command I am
about to type -- and every line that says no names the command it stops and
the exact remedy, never a bare "not found".

Nothing here executes a binary.  Each staged door is checked the three ways
the engine's own staging checks one -- the exact byte count, the SHA-256 pin
this release published, and a contract literal the current record shape
compiles in -- and all three are properties of the bytes, read without
running anything.  So a stale door is caught statically: no subprocess, no
new command surface, and it works on the binaries a user already has on disk.

Which pin is asked depends on who publishes the door.  A door from the
engine's bundle is checked against the pins inside the installed woof wheel;
a door this package publishes is checked against
``woof/globe/data/door-pins.json``.  A pin that does not exist yet is
reported as an unchecked hash rather than a pass, because a check that says
"verified" without hashing anything is worse than no check.

EXIT CODE.  ``0`` when every documented command on this machine can run.
``1`` when at least one cannot, and the report says which and why.  A missing
optional (no CUDA device, no geography archive) is reported as a note, not a
gap, because it stops nothing a user asked for until they ask for it.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version as _dist_version
import platform
import sys

from ._version import CONSOLE_SCRIPT, DISTRIBUTION_NAME, __version__
from .doors import (
    COMPANION_BUNDLE,
    DOORS,
    ENGINE_BUNDLE,
    artifact_filename,
    companion_door_dir,
    companion_pins,
    current_platform,
    find_door,
    publisher,
    sha256_file,
    verify_staged,
)

__all__ = ["Row", "Report", "build_report", "doctor", "add_doctor_arguments"]

#: How wide the label column is printed.
_LABEL = 26


@dataclass
class Row:
    """One reported fact: a label, a finding, and whether it stops anything."""

    label: str
    finding: str
    #: "ok", "gap" or "note".  Only "gap" moves the exit code.
    verdict: str = "ok"
    #: Printed under the finding, indented, when present.
    detail: tuple[str, ...] = ()


@dataclass
class Report:
    """The whole report, section by section."""

    sections: list[tuple[str, list[Row]]] = field(default_factory=list)

    def section(self, title: str) -> list[Row]:
        rows: list[Row] = []
        self.sections.append((title, rows))
        return rows

    @property
    def gaps(self) -> list[Row]:
        return [row for _, rows in self.sections for row in rows
                if row.verdict == "gap"]

    def render(self) -> str:
        out: list[str] = []
        for title, rows in self.sections:
            out.append(title)
            out.append("-" * len(title))
            for row in rows:
                mark = {"ok": "ok  ", "gap": "GAP ", "note": "note"}[row.verdict]
                out.append(f"  {mark} {row.label:<{_LABEL}} {row.finding}")
                for line in row.detail:
                    out.append(f"       {'':<{_LABEL}} {line}")
            out.append("")
        gaps = self.gaps
        if gaps:
            out.append(f"{len(gaps)} gap(s).  Each one above names the command it stops.")
        else:
            out.append("No gaps.  Every documented command can run on this machine.")
        return "\n".join(out)


def _installed(name: str) -> str | None:
    try:
        return _dist_version(name)
    except PackageNotFoundError:
        return None


def _engine_requirement() -> str:
    """The woof range this installed distribution declares, read back."""

    try:
        from importlib.metadata import requires
    except ImportError:  # pragma: no cover - stdlib on every supported version
        return "woof (requirement unreadable)"
    try:
        declared = requires(DISTRIBUTION_NAME) or ()
    except PackageNotFoundError:
        return "woof (this package is not installed; running from a source tree)"
    for entry in declared:
        if entry.split(";")[0].strip().startswith("woof") and "gpuwm-" not in entry:
            return entry.split(";")[0].strip()
    return f"recast-woof=={__version__}"  # the engine ships in this distribution


def _in_range(installed: str, requirement: str) -> bool | None:
    """Whether ``installed`` satisfies ``requirement``; None when unknowable."""

    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
    except ImportError:
        return None
    try:
        return Version(installed) in Requirement(requirement).specifier
    except Exception:  # pragma: no cover - a malformed version string
        return None


def _distribution_section(report: Report) -> None:
    rows = report.section("distribution")
    rows.append(Row("name", DISTRIBUTION_NAME))
    rows.append(Row("version", __version__,
                    verdict="note" if __version__ == "0+unknown" else "ok",
                    detail=(("running from a source tree that is not installed; "
                             "`pip install -e .` to read a real version",)
                            if __version__ == "0+unknown" else ())))
    rows.append(Row("import name", "woof.globe"))
    rows.append(Row("console script", CONSOLE_SCRIPT))
    rows.append(Row("python", f"{platform.python_version()} ({sys.executable})"))
    rows.append(Row("platform", f"{platform.system()} {platform.machine()}"))


def _engine_section(report: Report) -> None:
    rows = report.section("engine")
    requirement = _engine_requirement()
    installed = _installed("recast-woof")
    if installed is None:
        rows.append(Row(
            "woof", "not installed", verdict="gap",
            detail=(f"every command in this package imports it; install {requirement}",)))
    else:
        verdict = _in_range(installed, requirement)
        if verdict is False:
            rows.append(Row(
                "woof", f"{installed} (declared {requirement})", verdict="gap",
                detail=("the engine outside the measured range: this package imports 84 "
                        "symbols across that boundary and a moved symbol fails at the "
                        "first import of a forecast, after the install reported success",
                        f"install an engine inside {requirement}")))
        elif verdict is None:
            rows.append(Row("woof", f"{installed} (declared {requirement})",
                            verdict="note",
                            detail=("`packaging` is not installed, so the range was not "
                                    "checked; the version above is what is present",)))
        else:
            rows.append(Row("woof", f"{installed}, inside {requirement}"))
    data = _installed("recast-woof-data")
    if data is None:
        rows.append(Row(
            "recast-woof-data", "not installed", verdict="gap",
            detail=("the physics tables live there: the convection, microphysics, "
                    "radiation, PBL and land-surface suites this model runs have no "
                    "tables without it",
                    "the engine declares it; reinstalling woof brings it")))
    else:
        rows.append(Row("recast-woof-data", data))
    for name, why in (("numpy", "every module in this package"),
                      ("netCDF4", "checkpoint and tape input/output"),
                      ("matplotlib", "the scorecard analysis charts (never a weather field)"),
                      ("scipy", "the radiation scorecard's interpolation")):
        found = _installed(name)
        if found is None:
            rows.append(Row(name, "not installed",
                            verdict="gap" if name == "numpy" else "note",
                            detail=(f"needed by {why}",)))
        else:
            rows.append(Row(name, found))


def _boundary_section(report: Report) -> None:
    """What this package asks of the engine, and what the engine answers.

    THE BREAKAGE THIS SECTION EXISTS FOR.  Before it, this report ended with
    "No gaps.  Every documented command can run on this machine" on a box
    where `sizing` could not price a card, the radiation scorecard could not
    load its float64 reference and the surface-energy scorecard could not
    regrid a tape -- because those are engine symbols the installed engine
    does not carry, and the report only looked at Rust doors and versions.

    TWO KINDS OF ANSWER, and the difference decides the verdict:

    * A NAME that does not resolve.  A carried CONTRACT is a note (the
      package supplies it and nothing stops).  A refused COMPUTATION that a
      documented command reaches is a gap (that command stops, and this
      package will not answer with a second instrument).  A refused
      computation no documented command reaches is a note marked optional:
      calling it still refuses by name, but the exit code says whether every
      documented command can run, and on a correct install against the
      published 2.8.0 both absent symbols are of this kind, so grading them
      gaps made a correct install exit 1.
    * A SIGNATURE that does not take an argument this package passes.  Always
      a gap, and the worst-placed kind: the name resolves, so the install and
      every version check report success, and the call fails inside the run.

    A row the machine cannot answer -- a module that imports cupy at module
    scope cannot be read on a host with no CUDA runtime -- is printed as a
    note saying so, never as `ok`.

    THE SECTION SHRANK ON 2026-09-09 and that is the point of it.  It carried
    two contract rows and five signature rows for the physics; the carve took
    that physics into `woof.globe.core`, so those rows describe nothing an
    installed engine can still get wrong.  What the engine still supplies
    underneath the carried code moved to the `engine seam` section below,
    which is a different question with a different verdict.
    """

    from .engine_compat import (
        GAPS, SIGNATURE_GAPS, engine_gaps, engine_signature_gaps,
    )

    rows = report.section("engine boundary")
    missing = engine_gaps()
    for gap in missing:
        label = f"{gap.module.split('.')[-1]}.{gap.symbol}"
        stops = f"the installed engine does not carry it; it stops {gap.stops}"
        if gap.handling == "carried":
            rows.append(Row(label, "carried by this package", verdict="note",
                            detail=(stops, "carried here as a contract, so "
                                           "nothing stops")))
            continue
        refused = ("this package does not reimplement it: a second "
                   "instrument answering the same question is how two "
                   "numbers get reported with equal confidence and one "
                   "of them is wrong")
        if gap.stops_a_documented_command:
            rows.append(Row(label, "absent", verdict="gap",
                            detail=(stops, refused)))
            continue
        rows.append(Row(
            label, "absent (optional)", verdict="note",
            detail=(stops,
                    "no documented command reaches it, so it does not move "
                    "the exit code; a call to it still refuses by name",
                    refused,
                    f"an engine that carries {gap.symbol} closes this row")))
    signature_gaps = engine_signature_gaps()
    for gap, lacking in signature_gaps:
        rows.append(Row(
            f"{gap.module.split('.')[-1]}.{gap.name}",
            f"does not take {', '.join(lacking)}",
            verdict="gap",
            detail=(f"the name resolves and the callable is a different one; it "
                    f"stops {gap.stops}",
                    "refused at the door -- where the options are frozen, or "
                    "before the decode opens a file -- rather than inside the "
                    "run")))
    unanswerable = [gap for gap in SIGNATURE_GAPS if gap.missing() is None]
    for gap in unanswerable:
        rows.append(Row(
            f"{gap.module.split('.')[-1]}.{gap.name}", "not readable here",
            verdict="note",
            detail=(f"{gap.module} needs a CUDA runtime to import, so this host "
                    "cannot read its signature and does not answer for one that "
                    "can",)))
    if not missing and not signature_gaps and not unanswerable:
        rows.append(Row(
            "boundary", f"{len(GAPS)} symbols and {len(SIGNATURE_GAPS)} "
                        "signatures resolve"))


def _engine_seam_section(report: Report) -> None:
    """The engine files this package reaches and does not carry.

    THE BREAKAGE THIS SECTION EXISTS FOR.  Most of the physics is carried, so
    it cannot move.  The rest is reached on the installed engine, and the
    decision to leave it there was measured against ONE published engine.  A
    resolution inside `woof>=2.8.0,<2.9` can put a different one underneath
    and print nothing: `woof.core.constants` supplies CUDA_DEFINES to the
    preamble of every carried kernel, so moving it moves every kernel's
    assembled source, its digest, its PTX and its contraction, with no other
    signal anywhere.

    A moved file is a NOTE naming the file, never a gap.  The version ceiling
    in the dependency pin is the refusal; "these bytes are not the ones I
    measured" does not name a breakage, and a package that stopped running
    over a changed comment would be worse than one that says what it no
    longer recognises.
    """

    from .engine_seam import check_seam, load_manifest

    rows = report.section("engine seam")
    try:
        manifest = load_manifest()
        seam = check_seam()
    except Exception as exc:
        rows.append(Row("seam manifest", f"unreadable ({exc})", verdict="gap",
                        detail=("without it nothing can say whether the "
                                "engine underneath the carried physics is "
                                "the one this package was measured against",)))
        return
    pinned_against = manifest.get("engine", {}).get("version", "an "
                                                    "unrecorded version")
    unproven = [row for row in seam if row.verdict != "proven"]
    for row in unproven:
        rows.append(Row(
            row.path.rsplit("/", 1)[-1],
            "absent" if row.verdict == "absent" else "moved",
            verdict="note",
            detail=(f"{row.path}",
                    f"pinned {row.pinned_sha256[:12]} at woof "
                    f"{pinned_against}"
                    + ("" if row.found_sha256 is None
                       else f", installed {row.found_sha256[:12]}"),
                    f"this package reaches {row.reached}",
                    "unproven, not refused: the dependency ceiling is the "
                    "refusal")))
    proven = len(seam) - len(unproven)
    # THE TOKEN IS THE SUMMARY, and a reader scans tokens.  Measured on
    # the desktop 2026-09-10 with a comment appended to the installed
    # woof/core/constants.py: the note row above printed "moved" with
    # both digests and this row still printed "ok  seam  45/46 files
    # proven".  Green on the row that summarises a moved file, and
    # constants.py is the one that reaches every carried kernel's
    # preamble.  The refusal stays the version ceiling; only the token
    # moves.
    rows.append(Row(
        "seam", f"{proven}/{len(seam)} files proven",
        verdict="note" if unproven else "ok",
        detail=(f"pinned against woof {pinned_against}",
                "the scope is the DIRECT engine imports of the carried "
                "physics, plus the assimilation's filter, the local-GPU "
                "switch and two files those reach; it is not the import "
                "closure, which nothing on the run path enters",
                "the engine's DOORS are not pinned here: they are measured "
                "by symbol in the boundary section above")))


def _device_section(report: Report) -> None:
    rows = report.section("device")
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        rows.append(Row(
            "cupy", "not importable", verdict="note",
            detail=(f"{type(exc).__name__}: {exc}",
                    "the CPU backend runs every door; the GPU backend needs "
                    f"`pip install \"{DISTRIBUTION_NAME}[gpu]\"` matched to the "
                    "CUDA major the driver reports")))
        return
    import cupy as cp
    rows.append(Row("cupy", cp.__version__))
    try:
        runtime = int(cp.cuda.runtime.runtimeGetVersion())
        rows.append(Row("cuda runtime", f"{runtime // 1000}.{(runtime % 1000) // 10}"))
    except Exception as exc:
        rows.append(Row("cuda runtime", f"unreadable ({exc})", verdict="note"))
        return
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        rows.append(Row("devices", f"none reachable ({exc})", verdict="note"))
        return
    for index in range(count):
        props = cp.cuda.runtime.getDeviceProperties(index)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        free, total = _device_memory(cp, index)
        rows.append(Row(
            f"device {index}", name,
            detail=(f"compute capability {props['major']}.{props['minor']}, "
                    f"{free / 2**20:,.0f} MiB free of {total / 2**20:,.0f} MiB",)))


def _device_memory(cp, index: int) -> tuple[int, int]:
    with cp.cuda.Device(index):
        return cp.cuda.runtime.memGetInfo()


def _doors_section(report: Report) -> None:
    rows = report.section("rust doors")
    pins = companion_pins()
    platform_key = current_platform()
    published = bool(pins.get("platforms", {}).get(platform_key or ""))
    companion_dir = companion_door_dir()
    for door in DOORS:
        staged = find_door(door.name)
        # Who publishes it ON THIS INSTALL, not only what the table says:
        # a door the installed engine's own bundle declares is the engine's,
        # graded against the engine's pins and restaged by the engine's
        # command (doors.publisher says why).
        bundle = publisher(door.name)
        origin = ("`woof fetch-bridges`" if bundle == ENGINE_BUNDLE
                  else f"`{CONSOLE_SCRIPT} fetch-doors`")
        used = ", ".join(door.used_by)
        if staged is None:
            detail = [f"{door.role}",
                      (f"does not stop {used}: {door.fallback}" if door.fallback
                       else f"stops: {used}"),
                      f"published by the {bundle} bundle; stage it with {origin}"]
            if bundle == COMPANION_BUNDLE and not published:
                detail.append(
                    "no companion bundle has been published for this platform yet, so "
                    "there is nothing to stage: build the crate from the engine's "
                    "tools/rustwx workspace, or wait for the first release of this "
                    "package")
            rows.append(Row(door.name, "not staged", verdict="gap",
                            detail=tuple(detail)))
            continue
        size = staged.stat().st_size
        verdict, checked = verify_staged(door.name, staged)
        detail = [f"published by the {bundle} bundle, resolved from "
                  f"{_origin(door, staged)}: {staged}", checked]
        detail.extend(_shadow_notes(door, staged, companion_dir))
        if verdict == "gap":
            detail.append(f"breakage: {door.marker_breakage or door.role}")
            detail.append(f"does not stop {used}: {door.fallback}" if door.fallback
                          else f"stops: {used}")
            detail.append(f"restage it from the {bundle} bundle with {origin}")
            rows.append(Row(door.name, f"staged but wrong: {checked}",
                            verdict="gap", detail=tuple(detail)))
            continue
        rows.append(Row(door.name, f"{size:,} B", verdict=verdict,
                        detail=tuple(detail)))


def _origin(door, staged) -> str:
    """Which directory these exact bytes were resolved from.

    The publisher and the directory are different questions and printing
    only the first is how a door staged in one place gets reported as
    coming from another.  An operator reading a gap needs the directory,
    because that is where the wrong file is.
    """

    from woof.bridges import default_bridge_dir, packaged_bridge_dir
    import os

    parent = staged.parent
    override = os.environ.get(door.env_var)
    if override and Path(override).resolve() == staged.resolve():
        # The console script binds one variable per companion door into its
        # own environment before any command runs, pointing at the file in
        # the companion directory.  That binding is this program's, not the
        # operator's, and printing it as an override sends a reader to unset
        # something the program sets again on every run.  An override is the
        # operator's only when it points somewhere other than the file this
        # package's own binding would have produced, the same rule
        # fetch-doors applies when it says what outranks a staging.
        own = companion_door_dir() / artifact_filename(door.name)
        if (publisher(door.name) == COMPANION_BUNDLE
                and own.resolve() == staged.resolve()):
            return ("the companion door directory, named to the engine as "
                    f"{door.env_var}")
        return f"the {door.env_var} override, set outside this program"
    if parent == companion_door_dir():
        return "the companion door directory"
    if parent == default_bridge_dir():
        return "the engine's staged bridges"
    if parent == packaged_bridge_dir():
        return "the engine's wheel"
    return "an unrecognised directory"


def _shadow_notes(door, staged, companion_dir) -> list[str]:
    """Say when a second copy of this door exists somewhere else on the box.

    A door resolving correctly while a different build of the same name sits
    one rung further down the ladder is the shadowing that made an earlier
    line of releases fail with a message blaming the file that was right.
    Naming it here costs one line and makes it impossible to miss.
    """

    from woof.bridges import default_bridge_dir

    notes: list[str] = []
    engine_copy = default_bridge_dir() / artifact_filename(door.name)
    if engine_copy.is_file() and engine_copy.resolve() != staged.resolve():
        same = (engine_copy.stat().st_size == staged.stat().st_size
                and sha256_file(engine_copy) == sha256_file(staged))
        notes.append(
            f"the engine's bundle holds {'the same' if same else 'a DIFFERENT'} "
            f"build at {engine_copy}"
            + ("" if same else "; this package resolves its own copy first and "
                               "the engine keeps resolving that one"))
    return notes


def _configs_section(report: Report) -> None:
    from .configs_dir import config_root, list_configs

    rows = report.section("configs")
    root = config_root()
    if not root.is_dir():
        rows.append(Row("shipped configs", f"missing: {root}", verdict="gap",
                        detail=("the wheel ships the experiment TOMLs as package data; "
                                "an install without them cannot run a named experiment",)))
        return
    names = list_configs()
    rows.append(Row("shipped configs", f"{len(names)} experiments in {root}"))


#: The mapping ids this package names in its own source, beside the ones its
#: experiments name.  Each is a source the authority tables are asked for by
#: NAME, so a spec neither table answers refuses the command that reads it.
#:
#: EVERY ENTRY IS AN ID, never a file name.  Two of these rows are opened by
#: their readers as files -- `radiation_scorecard.REFERENCE_MAPPING` and
#: `microwave.columns.MAPPING_NAME` both spell `<id>.mapping.json` -- and this
#: table used to carry that spelling so the label matched the reader.  The
#: result was one registry field holding two kinds of thing across six rows,
#: an id on four and a file name on two, which a picker cannot key.  The
#: resolver answers both spellings for every row (`_authority_matches` asks
#: for the file whose name ends at the id and then for the family glob), so
#: the label is the id everywhere and the file name stays reachable.
_NAMED_MAPPINGS = (
    ("gfs-surface-state", "the surface energy scorecard's state fields"),
    ("gfs-surface-flux", "the surface energy scorecard's fluxes"),
    ("rw-wps-gfs-pgrb2-0p25-cloud-cover",
     "the radiation scorecard's reference cloud cover"),
    ("rw-wps-gdas-pgrb2-0p25-microwave-columns",
     "the ATMS forward operator's columns"),
    ("ecmwf-open-data-global-forecast",
     "the obs scorecard's IFS open-data reference columns"),
)


def _config_mapping_ids() -> dict[str, list[str]]:
    """Every analysis mapping id the shipped experiments name, and who names it.

    Read out of the shipped TOMLs rather than listed here, so an experiment
    added or a source changed moves this report without anyone remembering to.
    """

    import tomllib

    from .configs_dir import config_root

    found: dict[str, list[str]] = {}
    root = config_root()
    if not root.is_dir():
        return found
    for path in sorted(root.glob("*.toml")):
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for table in payload.values():
            if not isinstance(table, dict):
                continue
            spec = table.get("analysis_mapping")
            if isinstance(spec, str) and spec and not spec.endswith(".json"):
                found.setdefault(spec, []).append(path.stem)
    return found


def _mapping_digest(path) -> str:
    """The SHA-256 of the mapping file that answered a row.

    WHAT IT MEASURES: the bytes of that one file, on this machine, now.  It
    compares them to nothing.  A digest a reader can hold against the
    ``mapping_sha256`` a run receipt records is a fact this instrument can
    produce; "identical to the tree this model was graded in" is not, because
    doctor never opens that tree.  A file it cannot read says so rather than
    coming back as a digest of nothing.
    """

    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as failure:
        return f"unreadable ({failure.__class__.__name__})"


def _source_mappings_section(report: Report) -> None:
    """Which authority table answers each source this package names.

    A source mapping is metadata the engine owns and this package reads by
    bare id, which is what keeps adding a source declarative.  The engine's
    2.7 table carries none of the six global ones, so every shipped GDAS
    experiment used to refuse at the first door with `found 0 (none)`.  The
    package now carries those six itself and the resolver asks the engine
    first, so the row that matters here is no longer present/absent but
    WHICH COPY ANSWERED: an engine that has taken the row wins the moment it
    is installed, and a reader who cannot tell the two apart cannot tell
    which source table a forecast was initialized from.

    A spec neither table answers is still a gap, and so is a spec both carry
    with different bytes -- that one is the resolver's refusal, printed here
    before a run meets it.
    """

    from .analysis_initial import (
        MappingTablesDisagree, PACKAGE_AUTHORITIES_DIR,
        resolve_analysis_mapping_row,
    )

    rows = report.section("source mappings")
    wanted: list[tuple[str, str]] = [
        (spec, f"{len(users)} shipped experiment{'' if len(users) == 1 else 's'}, "
               f"including {users[0]}")
        for spec, users in sorted(_config_mapping_ids().items())
    ]
    wanted.extend(_NAMED_MAPPINGS)
    carried = 0
    for spec, used_for in wanted:
        try:
            row = resolve_analysis_mapping_row(spec)
        except MappingTablesDisagree as exc:
            # BOTH tables answered.  Saying "no authority table answers it"
            # here sends a reader to look for a file that is missing when
            # the file is present twice and the two copies decode different
            # records under one name.
            rows.append(Row(
                spec, "the two tables disagree",
                verdict="gap",
                detail=(f"it stops: {used_for}",
                        "the engine publishes the source table and its row "
                        "wins, but a row that has MOVED means a run decoded "
                        "through it is not the run this model was graded "
                        "against",
                        str(exc).splitlines()[0])))
            continue
        except Exception as exc:  # the resolver's refusal IS the finding
            rows.append(Row(
                spec, "no authority table answers it",
                verdict="gap",
                detail=(f"it stops: {used_for}",
                        "the engine publishes the source table and this "
                        "package carries the six rows the engine has not "
                        "taken yet; a spec neither answers is a source "
                        "nobody has",
                        str(exc).split(";")[0])))
            continue
        digest = _mapping_digest(row.path)
        if row.origin == "package":
            carried += 1
            rows.append(Row(
                spec, f"{row.path.name} (carried by this package)",
                detail=("the installed woof's authority table has no row "
                        "for it; the engine is still asked first and wins "
                        "the day it publishes one",
                        f"sha256 {digest}")))
        else:
            rows.append(Row(spec, f"{row.path.name} (from the engine)",
                            detail=(f"sha256 {digest}",)))
    if carried:
        rows.append(Row(
            "carried copies", f"{carried} of {len(wanted)} answered from "
            f"{PACKAGE_AUTHORITIES_DIR}",
            detail=("each row above prints the SHA-256 of the file that "
                    "answered it, which is the digest a run receipt records "
                    "as mapping_sha256; this reads the answering file and "
                    "opens no other tree",
                    "a spec both tables carry with different bytes is "
                    "refused by name rather than chosen between",)))


def build_report() -> Report:
    """The whole report, without printing it."""

    report = Report()
    _distribution_section(report)
    _engine_section(report)
    _boundary_section(report)
    _engine_seam_section(report)
    _device_section(report)
    _doors_section(report)
    _source_mappings_section(report)
    _configs_section(report)
    return report


def add_doctor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json", action="store_true",
        help="machine-readable report on stdout instead of the human table")


def doctor(args: argparse.Namespace) -> int:
    report = build_report()
    if getattr(args, "json", False):
        import json

        print(json.dumps({
            "schema": "gpuwm-global-doctor-v1",
            "version": __version__,
            "gaps": len(report.gaps),
            "sections": [
                {"title": title,
                 "rows": [{"label": row.label, "finding": row.finding,
                           "verdict": row.verdict, "detail": list(row.detail)}
                          for row in rows]}
                for title, rows in report.sections
            ],
        }, indent=2, sort_keys=True))
    else:
        print(report.render())
    return 1 if report.gaps else 0
