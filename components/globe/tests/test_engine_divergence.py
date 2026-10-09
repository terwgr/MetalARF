"""The carried physics differs from the engine's only where a row says so.

WHAT THIS GATE PREVENTS.  ``src/arwen_global/core`` carries the physics this
model was graded with.  The engine on the index carries its own copies and
the two lines move independently, which is fine and is the point; what is
not fine is a difference nobody has decided about.  Three ways that happens:

* the engine publishes a minor that changes one of the carried files, and
  nobody notices that a fix, or a defect, now exists on one side only;
* a re-cut brings a change across from the model's own source tree and no
  entry in ``docs/CARRIED-PHYSICS-DIVERGENCE.md`` says whether it is
  deliberate;
* a row in that document describes a difference that is no longer there,
  so a reader following it re-applies something already applied.

``tools/fingerprint_engine_divergence.py`` measures the differences and
hashes each one on both sides; ``src/arwen_global/data/engine-divergence.json``
is that measurement with the document's class and decision joined on.  This
file re-measures against the installed engine and holds the two to each
other.

WHICH ENGINE.  The rows are measured against ONE published engine, named in
the JSON, and four of the carried files moved between versions that are all
inside the range this package declares.  So the two nodes that compare a
measurement against the rows run on that version and skip, naming both
versions, on any other; the nodes that hold the JSON and the document to each
other need no engine and always run.  Re-baselining is one command and is the
workflow the document describes.

WHAT IT DOES NOT DO.  It does not check that a classification is RIGHT.  A
row saying ``refuse`` where the answer should be ``pull`` passes here and is
caught by a person reading the document.  What cannot pass is a difference
with no row, a row with no difference, a row whose table in the document and
whose entry in the JSON disagree, or a carried file that neither of them
measures.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys

import pytest

from conftest import REPO_ROOT, TOOLS

DOCUMENT = REPO_ROOT / "docs" / "CARRIED-PHYSICS-DIVERGENCE.md"
SHIPPED = REPO_ROOT / "src" / "arwen_global" / "data" / "engine-divergence.json"

#: A class of ``unknown`` is what the tool writes for a hunk nobody has
#: classified, so it is the one value the shipped file may not carry.
UNCLASSIFIED = "unknown"


def _load_tool():
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "fingerprint_engine_divergence",
        TOOLS / "fingerprint_engine_divergence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _engine_version() -> str | None:
    try:
        if importlib.util.find_spec("woof") is None:
            return None
    except (ImportError, ValueError):
        return None
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("woof")
    except PackageNotFoundError:
        return None


_INSTALLED = _engine_version()
_BASELINE = json.loads(SHIPPED.read_text(encoding="utf-8"))["engine_version"]

#: THE VERSION THIS GATE CAN SPEAK ABOUT.  The rows were measured against one
#: published engine, and four of the carried files moved between the versions
#: inside the range this package declares, so on a different in-range engine
#: the measurement legitimately disagrees with the rows and a failure would be
#: saying nothing about this package.  So the engine-dependent nodes run
#: against the baseline version and skip, by name and with both versions in
#: the reason, against any other.  The document-and-JSON nodes below need no
#: engine at all and always run.
needs_baseline_engine = pytest.mark.skipif(
    _INSTALLED != _BASELINE,
    reason=(f"the rows were measured against woof {_BASELINE} and the "
            f"installed engine is {_INSTALLED or 'absent'}; re-baseline with "
            "`python tools/fingerprint_engine_divergence.py --rewrite` to "
            "hold the document to this one"))


@pytest.fixture(scope="module")
def measured():
    tool = _load_tool()
    version, rows = tool.measure()
    return tool, version, rows


@pytest.fixture(scope="module")
def shipped():
    return json.loads(SHIPPED.read_text(encoding="utf-8"))


def _key(row):
    return row["file"], row["engine_sha256"], row["carried_sha256"]


def _pool(rows):
    out = {}
    for row in rows:
        out.setdefault(_key(row), []).append(row)
    return out


def test_the_shipped_json_names_the_engine_it_was_measured_against(shipped):
    assert shipped["schema"] == "arwen-global.engine-divergence.v1"
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", shipped["engine_version"])
    assert shipped["document"] == "docs/CARRIED-PHYSICS-DIVERGENCE.md"
    assert shipped["normalisation"]


@needs_baseline_engine
def test_every_difference_the_engine_shows_has_a_row(measured, shipped):
    """(a) An unclassified difference fails by name, with its lines."""

    _tool, version, rows = measured
    have = _pool(shipped["rows"])
    orphans = []
    for row in rows:
        pool = have.get(_key(row))
        if not pool:
            orphans.append(
                f"{row['file']} engine {row['engine_lines']} "
                f"carried {row['carried_lines']} "
                f"(engine sha {row['engine_sha256'][:16]})")
        else:
            pool.pop()
    assert not orphans, (
        f"against the installed woof {version}, the carried physics differs "
        "in places no row of docs/CARRIED-PHYSICS-DIVERGENCE.md covers. Read "
        "each one, add its row to the document, then run "
        "`python tools/fingerprint_engine_divergence.py --rewrite` and fill "
        "in the class and the decision:\n  " + "\n  ".join(orphans))


@needs_baseline_engine
def test_every_row_still_describes_a_difference(measured, shipped):
    """(b) A stale row fails by name: one side moved under the document."""

    _tool, version, rows = measured
    seen = _pool(rows)
    stale = []
    for row in shipped["rows"]:
        pool = seen.get(_key(row))
        if not pool:
            stale.append(
                f"{row['row'] or '(unnamed row)'} in {row['file']} "
                f"(engine {row['engine_lines']}, carried "
                f"{row['carried_lines']})")
        else:
            pool.pop()
    assert not stale, (
        f"against the installed woof {version}, these rows describe a "
        "difference that is no longer there. Either the engine moved, or this "
        "package did, or the difference was resolved; in every case the "
        "document has to say so before the row goes:\n  " + "\n  ".join(stale))


def test_no_row_is_left_unclassified(shipped):
    """(c) A row must carry a class, and `unknown` is not one."""

    blank = [f"{r['file']} engine {r['engine_lines']}"
             for r in shipped["rows"]
             if not r.get("class") or r["class"] == UNCLASSIFIED]
    assert not blank, (
        "these rows carry no classification. `unknown` is what the "
        "fingerprint tool writes for a hunk it has just found; it is not a "
        "verdict, and a row keeping it means nobody has read the "
        "difference:\n  " + "\n  ".join(blank))
    nameless = [f"{r['file']} engine {r['engine_lines']}"
                for r in shipped["rows"] if not r.get("row")]
    assert not nameless, (
        "these rows name no section of the document, so a reader who finds "
        "the fingerprint cannot find the paragraph that explains "
        "it:\n  " + "\n  ".join(nameless))


#: The line inside the PULL section that lists, and only lists, the rows
#: still owed.  It is a line of its own rather than every backticked name in
#: the section because the prose there names rows a pull has to ride WITH,
#: which are not themselves pulls.
PULL_ROLL = "Rows owed:"


def _pull_list() -> set[str]:
    """The row names on the document's PULL roll."""

    text = DOCUMENT.read_text(encoding="utf-8")
    start = text.index("## PULL")
    end = text.index("## OFFER", start)
    section = text[start:end]
    assert PULL_ROLL in section, (
        f"the PULL section of {DOCUMENT.name} has no line beginning "
        f"{PULL_ROLL!r}, so there is nothing to hold the rows to")
    roll = section[section.index(PULL_ROLL):].split("\n\n", 1)[0]
    return set(re.findall(r"`([A-Z][A-Z0-9-]*-\d+)`", roll))


def test_every_open_pull_is_listed_in_the_document(shipped):
    """(d) An engine fix this package has decided to take is listed.

    A row of class ``engine-fix`` whose decision is ``pull`` and which is
    still present is work owed.  The document's PULL list is where that work
    is read off, so a row that is not in it is a decision made in a table and
    lost.  A pull that has been TAKEN stops being a row at all, because the
    difference is gone, which is why this holds only over present rows.
    """

    listed = _pull_list()
    owed = {r["row"] for r in shipped["rows"]
            if r.get("class") == "engine-fix"
            and r.get("decision", "").startswith("pull")}
    missing = sorted(owed - listed)
    assert not missing, (
        "these rows say the engine fixed something this package should take, "
        "and the document's PULL list does not carry them:\n  "
        + "\n  ".join(missing))


def test_the_pull_list_names_no_row_that_is_not_owed(shipped):
    """The reverse: the list does not promise work no row asks for.

    A name on the roll must be a row that is still measured and still says
    ``pull``.  A pull that has been TAKEN leaves the roll at the same moment
    it leaves the rows, and is recorded in the prose above the roll instead;
    letting an unmatched name pass would make a mistyped row name
    indistinguishable from finished work.
    """

    rows = {r["row"]: r for r in shipped["rows"] if r.get("row")}
    spurious = []
    for name in sorted(_pull_list()):
        row = rows.get(name)
        if row is None:
            spurious.append(
                f"{name} is on the PULL roll and no measured row carries that "
                "name. Either the name is wrong, or the pull was taken and "
                "the roll still names it; a taken pull is recorded in the "
                "prose above the roll, never on it")
            continue
        if not row.get("decision", "").startswith("pull"):
            spurious.append(f"{name} is listed under PULL and its rows say "
                            f"{row['decision']!r}")
    assert not spurious, "\n  ".join(spurious)


#: A row line inside a file's table, and the heading that says which carried
#: file the table belongs to.
_ROW_LINE = re.compile(r"^\| `([A-Z][A-Z0-9-]*-\d+)` \|")
_SECTION = re.compile(r"^### `([^`]+)`$")


def _document_rows() -> dict[str, dict[str, str]]:
    """Every row table in the document, keyed by row name.

    The section heading a row sits under is the carried file the row is
    about, so a row that drifts into the wrong section is caught here too.
    """

    out: dict[str, dict[str, str]] = {}
    section = None
    for line in DOCUMENT.read_text(encoding="utf-8").split("\n"):
        heading = _SECTION.fullmatch(line.strip())
        if heading:
            section = heading.group(1)
            continue
        named = _ROW_LINE.match(line)
        if not named or section is None:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        out[named.group(1)] = {
            "file": section,
            "carried_lines": cells[1],
            "engine_lines": cells[2],
            "class": cells[4],
            "decision": cells[7].replace("*", "").strip(),
        }
    return out


def test_the_document_and_the_shipped_rows_say_the_same_thing(shipped):
    """The table a person acts on and the JSON a tool acts on are one thing.

    The document is what somebody reads to decide whether to take an engine
    change; the JSON is what the fingerprint join and every other node here
    reads.  Nothing else holds the two together, so a row whose table says
    ``refuse`` while its rows say ``pull`` would ship, and the reader and the
    tool would give different answers about the same code.
    """

    here = _document_rows()
    there: dict[str, list[dict]] = {}
    for row in shipped["rows"]:
        if row.get("row"):
            there.setdefault(row["row"], []).append(row)

    only_document = sorted(set(here) - set(there))
    only_json = sorted(set(there) - set(here))
    assert not only_document, (
        "these rows are tabled in the document and no measured hunk in the "
        "shipped JSON carries their name:\n  " + "\n  ".join(only_document))
    assert not only_json, (
        "these rows are in the shipped JSON and the document tables none of "
        "them, so a reader handed the name has nothing to read:\n  "
        + "\n  ".join(only_json))

    disagreements = []
    for name in sorted(here):
        said = here[name]
        hunks = there[name]
        for field in ("file", "class", "decision"):
            values = {hunk.get(field, "") for hunk in hunks}
            if values != {said[field]}:
                disagreements.append(
                    f"{name}: the document says {field} {said[field]!r} and "
                    f"the shipped rows say {sorted(values)!r}")
        for field in ("carried_lines", "engine_lines"):
            joined = ", ".join(hunk[field] for hunk in hunks)
            if joined != said[field]:
                disagreements.append(
                    f"{name}: the document's {field} read {said[field]!r} and "
                    f"the shipped rows read {joined!r}")
    assert not disagreements, (
        "the document and the shipped measurement disagree about rows they "
        "both describe. Whichever is right, they cannot both ship:\n  "
        + "\n  ".join(disagreements))


def test_every_carried_file_is_measured():
    """No carried file sits outside the measurement, whatever its suffix.

    A file the fingerprint tool does not walk produces no hunk, so it
    produces no row and no failure however far the two lines drift.  The
    four parameter tables are the case that matters: they are numbers Noah
    reads at run time, and a suffix filter once left all four unmeasured
    while the document asserted they were identical.
    """

    tool = _load_tool()
    walked = {carried for carried, _engine in tool.carried_pairs()}
    root = REPO_ROOT / "src" / "arwen_global"
    carried_files = set()
    for _source, carried in tool.carve.CORE_CARVE:
        here = root / carried
        if here.is_dir():
            carried_files.update(
                f"{carried}/{path.relative_to(here).as_posix()}"
                for path in here.rglob("*") if path.is_file())
        else:
            carried_files.add(carried)
    missed = sorted(carried_files - walked)
    assert not missed, (
        "these files are carried and the fingerprint tool does not measure "
        "them, so an engine change to one would produce no row and no "
        "failure:\n  " + "\n  ".join(missed))
    tables = sorted(name for name in walked if name.endswith(".TBL"))
    assert len(tables) == 4, (
        "the four WRF parameter tables are what a suffix filter drops first; "
        f"the tool walks {tables}")


def test_the_document_has_a_section_for_every_file_that_differs(shipped):
    text = DOCUMENT.read_text(encoding="utf-8")
    missing = sorted({r["file"] for r in shipped["rows"]
                      if f"`{r['file']}`" not in text})
    assert not missing, (
        "these carried files differ from the engine's and the document has "
        "no section naming them:\n  " + "\n  ".join(missing))
