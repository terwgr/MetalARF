#!/usr/bin/env python3
"""Fingerprint every difference between the carried physics and the engine's.

WHAT THIS IS FOR.  ``src/arwen_global/core`` carries the physics this model
was graded with, cut from the model's own source tree.  The engine on the
index carries its own copies of the same files and the two lines move
independently.  ``docs/CARRIED-PHYSICS-DIVERGENCE.md`` says, for every place
they differ, what differs and whether a future engine change to that code
should be pulled, refused or offered back.  That document is prose and its
line numbers go stale the first time either side inserts a line above them.

So every row also carries a FINGERPRINT that survives line shifts, and this
tool is what computes it.  ``src/arwen_global/data/engine-divergence.json``
is this tool's output with the class and decision columns of the document
joined onto it by fingerprint, and ``tests/test_engine_divergence.py``
re-computes the fingerprints against whatever engine is installed and fails
by name on a difference no row covers or a row no difference matches.

THE NORMALISATION, exactly, because the engine line keys its own change
lists by the engine-side hash and the two sides have to agree:

1.  For each carried file, take the engine's copy of the same file.  The
    mapping is ``CORE_CARVE`` in ``tools/resync_from_owner.py``: a source
    path ``gpuwm/<rest>`` lands at ``src/arwen_global/<carried>``, and the
    engine's copy of it is ``<woof package root>/<rest>``.
2.  Apply the carve's rewiring rules to the ENGINE side only: the module and
    command rewrites, the two words this project does not publish, the dash
    rule on Python, the slash-spelled source paths, the named exact blocks,
    the markdown title rule and the one preserve rule.  These are the rule
    tables of ``tools/resync_from_owner.py``, applied here without its
    post-conditions: a rule that matches nothing is skipped rather than
    raising, because this tool measures a difference and does not police a
    cut.  A file the carve does not treat as text -- its suffix is outside
    the carve's own ``TEXT_SUFFIXES``, which is what the four WRF parameter
    tables are -- is rewired by nothing on either side, exactly as the carve
    copies it.  After this step a hunk that is nothing but rewiring has
    identical text on both sides and produces no hunk at all.
3.  LF-normalise both sides (a carriage return and newline pair, and a lone
    carriage return, both become one newline) and split into lines keeping
    no line ending.
4.  ``difflib.unified_diff(engine_lines, carried_lines, n=0)``: zero context
    lines, so every hunk is exactly the lines that differ.
5.  Per hunk, ``sha256`` over the engine-side lines and ``sha256`` over the
    carried-side lines.  Each side is joined with a newline and, when the
    side is not empty, given a trailing newline; an EMPTY side (a pure
    insertion or a pure deletion) hashes the empty byte string, whose digest
    is the well-known ``e3b0c442...``.  The hash is of the text alone: no
    file name, no line numbers, nothing that moves when a line is inserted
    above.
6.  A file present on one side and absent on the other is one hunk whose
    absent side is empty and whose present side is the whole file.
7.  EVERY carried file is measured, whatever its suffix.  A pair either side
    of which does not decode as UTF-8 text has no lines to diff, so it is
    compared byte for byte and, when the bytes differ, produces one whole
    file hunk whose hashes are over the raw bytes.  Nothing carried is
    exempt: a difference that is never measured is a difference no row can
    be owed for, which is the one failure this tool exists to prevent.

A row is keyed by the file plus the PAIR of hashes, and the engine-side hash
is printed first.

WHAT THE ENGINE-SIDE HASH IS AND IS NOT.  It digests engine lines, but WHICH
lines is decided by the diff against the carried copy, so it is not
computable from the engine tree alone: a hunk boundary is a property of the
pair, not of one side.  What the engine line can do with a row is VERIFY it,
and that is exact: take the row's engine line range, read those lines out of
the published engine, apply the rewiring step 2 names, hash them as step 5
says, and compare.  Recomputing the whole set needs both copies present and
this tool, which the sdist ships beside the package for that reason.

USAGE

    python tools/fingerprint_engine_divergence.py            # a table
    python tools/fingerprint_engine_divergence.py --json     # the raw rows
    python tools/fingerprint_engine_divergence.py --rewrite  # rejoin and
                                                             # write the JSON

``--rewrite`` keeps the class, decision, row and note of every row whose
fingerprint pair is unchanged, drops rows whose hunk is gone, and writes any
new hunk with class ``unknown`` so the gate fails until somebody classifies
it.  It never invents a classification.

THE ENGINE RANGE.  The tool refuses to measure an engine outside the range
this package declares in ``pyproject.toml``, because a hunk measured against
an engine this package will not install with is not evidence about anything
it ships.  The version it did measure is recorded in the JSON as
``engine_version``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import tomllib
from difflib import unified_diff
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "src" / "arwen_global"
SHIPPED_JSON = PACKAGE / "data" / "engine-divergence.json"

sys.path.insert(0, str(REPO / "tools"))
import resync_from_owner as carve  # noqa: E402

#: The digest of the empty byte string, which is what an absent side hashes
#: to.  Written out so a reader who meets it in the JSON knows it is not a
#: mistake: it means the hunk is a pure insertion or a pure deletion.
EMPTY_SHA = hashlib.sha256(b"").hexdigest()



def engine_root() -> Path:
    """Where the installed engine's package directory is, or a refusal."""

    spec = importlib.util.find_spec("woof")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(
            "no engine is installed: importlib cannot find a woof package. "
            "Install one inside this package's declared range and run again")
    return Path(list(spec.submodule_search_locations)[0])


def engine_version() -> str:
    from importlib.metadata import version

    return version("woof")


def declared_range() -> tuple[str, str]:
    """The floor and the ceiling this package declares for the engine."""

    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    for spec in data["project"]["dependencies"]:
        if not re.match(r"^woof(?![-\w])", spec):
            continue
        floor = re.search(r">=\s*([0-9][0-9.]*)", spec)
        ceiling = re.search(r"<\s*([0-9][0-9.]*)", spec)
        if floor and ceiling:
            return floor.group(1), ceiling.group(1)
    raise SystemExit(
        "pyproject.toml declares no woof requirement with both a floor and a "
        "ceiling, so there is no range to hold an engine to")


def _parts(version_string: str) -> tuple[int, ...]:
    return tuple(int(piece) for piece in re.findall(r"\d+", version_string))


def check_range(measured: str) -> None:
    floor, ceiling = declared_range()
    if not (_parts(floor) <= _parts(measured) < _parts(ceiling)):
        raise SystemExit(
            f"the installed engine is woof {measured} and this package "
            f"declares woof>={floor},<{ceiling}. A difference measured "
            "against an engine this package will not install with is not "
            "evidence about what it ships, so nothing is measured")


def carried_pairs() -> list[tuple[str, str]]:
    """(carried path under src/arwen_global, engine path under the package).

    Directories in the carve mapping are walked file by file, so the Noah
    tables come out as six entries rather than one.  EVERY file under such a
    directory is returned, whatever its suffix: the four ``.TBL`` parameter
    tables are numbers Noah reads at run time, and a filter that dropped them
    would leave an engine change to one of them producing no hunk, no row and
    no failure.
    """

    pairs: list[tuple[str, str]] = []
    for source, carried in carve.CORE_CARVE:
        if not source.startswith("gpuwm/"):
            raise SystemExit(
                f"the carve maps {source}, which is not under the engine's "
                "package root, so this tool cannot find the engine's copy")
        inside = source[len("gpuwm/"):]
        here = PACKAGE / carried
        if here.is_dir():
            for path in sorted(here.rglob("*")):
                if path.is_file():
                    tail = path.relative_to(here).as_posix()
                    pairs.append((f"{carried}/{tail}", f"{inside}/{tail}"))
        else:
            pairs.append((carried, inside))
    return pairs


def rewire_engine_text(carried_rel: str, text: str) -> str:
    """The carve's rewiring rules, applied to the engine's own text.

    Mirrors ``resync_from_owner.rewire`` and ``device_prose`` rule for rule
    and in the same order, minus every post-condition: a rule that matches
    nothing is simply skipped here.
    """

    suffix = Path(carried_rel).suffix
    if any(carried_rel.startswith(p) for p in carve.VERBATIM_PREFIXES):
        return text
    if suffix not in carve.TEXT_SUFFIXES:
        # The carve rewires nothing it does not treat as text, so neither
        # does this.  The parameter tables come through here.
        return text
    if suffix in carve.VERBATIM_SUFFIXES:
        for pattern, replacement in carve.CORE_PROSE:
            text = re.sub(pattern, replacement, text)
        for name, old, new in carve.DEVICE_EXACT:
            if Path(carried_rel).name == name:
                text = text.replace(old, new)
        return text
    for pattern, replacement in (carve.REWIRE + carve.COMMANDS
                                 + carve.CORE_REWIRE + carve.CORE_PROSE):
        text = re.sub(pattern, replacement, text)
    if suffix == ".py":
        for pattern, replacement in carve.PROSE_DASH:
            text = re.sub(pattern, replacement, text)
    for named, old_lines, new_lines in (carve.CORE_EXACT
                                        + carve.CORE_TRACE_CLIMATOLOGY
                                        + carve.KERNEL_NOTICE_SCOPE):
        if carried_rel == named:
            text = text.replace("\n".join(old_lines), "\n".join(new_lines))
    for pattern, replacement in carve.SLASH_PATHS:
        text = re.sub(pattern, replacement, text)
    for named, old, new in carve.CORE_MD_PROSE:
        if carried_rel == named:
            text = text.replace(old, new)
    for named, pattern, replacement in carve.CORE_PRESERVE:
        if carried_rel == named:
            text = re.sub(pattern, replacement, text)
    return text


def lines_of(text: str) -> list[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def side_sha(lines: list[str]) -> str:
    if not lines:
        return EMPTY_SHA
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def read_bytes(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def decode(raw: bytes | None) -> str | None:
    """The text of a file, or ``None`` when it is absent or is not text."""

    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def byte_sha(raw: bytes | None) -> str:
    return hashlib.sha256(raw if raw is not None else b"").hexdigest()


def hunks_for(carried_rel: str, engine_rel: str, root: Path) -> list[dict]:
    here_raw = read_bytes(PACKAGE / carried_rel)
    there_raw = read_bytes(root / engine_rel)
    if here_raw is None and there_raw is None:
        return []
    here = decode(here_raw)
    there = decode(there_raw)
    if (here_raw is not None and here is None) or (
            there_raw is not None and there is None):
        # AT LEAST ONE SIDE IS NOT TEXT.  There are no lines to diff, so the
        # pair is compared whole: equal bytes are no hunk, and different
        # bytes are one hunk hashed over the raw bytes.
        if here_raw == there_raw:
            return []
        return [{
            "file": carried_rel,
            "engine_file": engine_rel,
            "engine_lines": "whole file" if there_raw is not None else "absent",
            "carried_lines": "whole file" if here_raw is not None else "absent",
            "engine_sha256": byte_sha(there_raw),
            "carried_sha256": byte_sha(here_raw),
        }]
    if here is None or there is None:
        present_here = here is not None
        text = here if present_here else rewire_engine_text(carried_rel, there)
        body = lines_of(text)
        return [{
            "file": carried_rel,
            "engine_file": engine_rel,
            "engine_lines": "absent" if present_here else f"1-{len(body)}",
            "carried_lines": f"1-{len(body)}" if present_here else "absent",
            "engine_sha256": EMPTY_SHA if present_here else side_sha(body),
            "carried_sha256": side_sha(body) if present_here else EMPTY_SHA,
        }]
    engine_lines = lines_of(rewire_engine_text(carried_rel, there))
    carried_lines = lines_of(here)
    open_hunks: list[dict] = []
    for piece in unified_diff(engine_lines, carried_lines, n=0, lineterm=""):
        head = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", piece)
        if head:
            open_hunks.append({
                "file": carried_rel,
                "engine_file": engine_rel,
                "_e": (int(head.group(1)), int(head.group(2) or 1)),
                "_c": (int(head.group(3)), int(head.group(4) or 1)),
                "_elines": [], "_clines": [],
            })
        elif open_hunks and piece.startswith("-"):
            open_hunks[-1]["_elines"].append(piece[1:])
        elif open_hunks and piece.startswith("+"):
            open_hunks[-1]["_clines"].append(piece[1:])
    rows = []
    for hunk in open_hunks:
        e_start, e_count = hunk.pop("_e")
        c_start, c_count = hunk.pop("_c")
        e_body = hunk.pop("_elines")
        c_body = hunk.pop("_clines")
        hunk["engine_lines"] = (f"{e_start}-{e_start + e_count - 1}"
                                if e_count else f"after {e_start}")
        hunk["carried_lines"] = (f"{c_start}-{c_start + c_count - 1}"
                                 if c_count else f"after {c_start}")
        hunk["engine_sha256"] = side_sha(e_body)
        hunk["carried_sha256"] = side_sha(c_body)
        rows.append(hunk)
    return rows


def measure() -> tuple[str, list[dict]]:
    measured = engine_version()
    check_range(measured)
    root = engine_root()
    rows: list[dict] = []
    for carried_rel, engine_rel in carried_pairs():
        rows.extend(hunks_for(carried_rel, engine_rel, root))
    return measured, rows


def key(row: dict) -> tuple[str, str, str]:
    return row["file"], row["engine_sha256"], row["carried_sha256"]


def rewrite() -> int:
    measured, rows = measure()
    # A POOL PER KEY, not one entry per key.  Two hunks in the same file can
    # carry the same text on both sides -- the same one-line change made at
    # two call sites -- and they are two rows, not one, so a fingerprint is
    # consumed rather than looked up.
    old: dict[tuple[str, str, str], list[dict]] = {}
    if SHIPPED_JSON.is_file():
        previous = json.loads(SHIPPED_JSON.read_text(encoding="utf-8"))
        for r in previous["rows"]:
            pool = old.setdefault(
                (r["file"], r["engine_sha256"], r["carried_sha256"]), [])
            pool.append(r)
    kept, fresh = 0, 0
    out = []
    for row in rows:
        pool = old.get(key(row)) or []
        if not pool:
            fresh += 1
            row.update({"row": "", "class": "unknown", "decision": "",
                        "note": ""})
        else:
            kept += 1
            carry = pool.pop(0)
            for field in ("row", "class", "decision", "note"):
                row[field] = carry.get(field, "")
        out.append(row)
    gone = sum(len(pool) for pool in old.values())
    document = {
        "schema": "arwen-global.engine-divergence.v1",
        "engine_version": measured,
        "document": "docs/CARRIED-PHYSICS-DIVERGENCE.md",
        "normalisation": (
            "per carried file, unified diff of the engine's copy with the "
            "carve's rewiring rules applied against the carried copy, zero "
            "context lines, both sides LF-normalised; per hunk, sha256 over "
            "the engine-side lines and sha256 over the carried-side lines, "
            "each joined with a newline and given a trailing newline, an "
            "empty side hashing the empty string; a row is keyed by file plus "
            "that pair"),
        "empty_side_sha256": EMPTY_SHA,
        "rows": out,
    }
    SHIPPED_JSON.write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"engine {measured}: {len(out)} hunks, {kept} carried forward, "
          f"{fresh} new (class unknown), {gone} stale rows dropped")
    return 1 if (fresh or gone) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true",
                        help="print the measured hunks as JSON")
    parser.add_argument("--rewrite", action="store_true",
                        help="rejoin onto the shipped JSON and write it")
    args = parser.parse_args(argv)
    if args.rewrite:
        return rewrite()
    measured, rows = measure()
    if args.json:
        print(json.dumps({"engine_version": measured, "rows": rows}, indent=2))
        return 0
    print(f"engine woof {measured}, {len(rows)} hunks")
    width = max((len(r["file"]) for r in rows), default=4)
    for row in rows:
        print(f"{row['file']:<{width}}  engine {row['engine_lines']:>12}  "
              f"carried {row['carried_lines']:>12}  "
              f"{row['engine_sha256'][:12]} {row['carried_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
