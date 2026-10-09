"""The seam manifest, against the engine actually installed.

`woof.globe.core` is what this package carries.  The seam is what it does
not: the engine files it still reaches, pinned by path, size and SHA-256 at
the engine version the decision to leave them was measured against.

WHY THIS FILE EXISTS.  A pip resolution inside `woof>=2.8.0,<2.9` can put a
different engine underneath and print nothing about a file's contents.
`woof/core/constants.py` alone supplies CUDA_DEFINES to the preamble of every
carried kernel, so moving it moves every kernel's assembled source, its
digest, its PTX and its floating-point contraction with no other signal.

WHAT A FAILURE HERE MEANS depends on which test fails, and the split is the
point.  A manifest that does not describe this repository is a defect in this
package.  A manifest that no longer matches the INSTALLED engine is a fact
about the engine, reported by name and never refused, so that row is a note
rather than an assertion about somebody else's release.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("woof", reason="the engine the seam is measured against")

from woof.globe.engine_seam import (  # noqa: E402
    SEAM_SCHEMA, check_seam, engine_root, load_manifest, manifest_path,
)

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "arwen_global"

def _carve_table():
    """`tools/resync_from_owner.py`, loaded by path rather than restated.

    THE BREAKAGE THIS PREVENTS.  What this package carries was written out
    here as a literal beside the same list in the carve table.  Drop a module
    from `CORE_CARVE` and leave it in the literal and the closure test below
    treats it as carried, so it is neither carried nor pinned and nothing
    reports it when the engine underneath moves.  The other direction, added
    to the carve and forgotten here, fails loudly, which is the safe half; a
    table read from the tool has no halves.
    """

    tool = REPO / "tools" / "resync_from_owner.py"
    spec = importlib.util.spec_from_file_location("resync_from_owner", tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: What the package carries, so a `woof.` name that resolves to one of these
#: is neither pinned nor a hole: the bytes travel in the wheel.  The Python
#: rows of the carve table, which is where the decision is made.
CARRIED = {src for src, _ in _carve_table().CORE_CARVE if src.endswith(".py")}

#: Pinned files no `import woof...` statement in this tree names.  Each is
#: reached THROUGH another pinned file, which is why a walk of this package's
#: own imports cannot see it, and each is stated here rather than allowed by
#: a pattern that would also allow a rationale nobody checked.
INDIRECT = {
    # staying woof/core/state.py binds both at ITS module scope
    "woof/core/sase_limits.py",
    "woof/core/wdm6_constants.py",
    # the engine's LETKF compiles it; the assimilation reaches it through
    # woof/da/letkf.py
    "woof/core/jacobi_eigh.py",
}

#: Content this package depends on outside the carried physics: the filter
#: the assimilation runs ON the engine, and the switch the native suite reads.
OUTSIDE_THE_CORE = {"woof/da/letkf.py", "woof/local_gpu.py"}


def _installed_engine_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("recast-woof")
    except PackageNotFoundError:
        return None


_INSTALLED = _installed_engine_version()
_PINNED = load_manifest()["engine"]["version"]

#: THE VERSION THIS TABLE CAN SPEAK ABOUT, and the rule is the one
#: `tests/test_engine_divergence.py` already follows.  Every row is a SHA-256
#: read off one published engine, and the declared range `woof>=2.8.0,<2.9`
#: legitimately resolves others: 2.7.3 moved eight of these files against
#: 2.7.0 with nothing the package reaches changing behaviour.  A hash
#: mismatch there is a fact about somebody else's release, not a defect in
#: this package, and the doctor already prints it file by file as "moved".
#: So the two nodes that COMPARE hashes run against the pinned version and
#: skip, naming both versions, on any other; every other node here is about
#: this repository and always runs.
#:
#: The refusal is elsewhere, on purpose: the dependency ceiling.
needs_pinned_engine = pytest.mark.skipif(
    _INSTALLED != _PINNED,
    reason=(f"the seam rows were read off woof {_PINNED} and the installed "
            f"engine is {_INSTALLED or 'absent'}; re-pin with "
            "`python tools/pin_engine_seam.py` to hold the table to this "
            "one, and read `woof global doctor` for the per-file rows"))


def _boundary_instrument():
    """`tools/measure_boundary.py`, loaded by path rather than reimplemented.

    Its `collect` is a pure AST walk of this package's source: it never
    imports the package, which is what makes it usable from a test that is
    asking what the package imports.
    """

    tool = REPO / "tools" / "measure_boundary.py"
    spec = importlib.util.spec_from_file_location("measure_boundary", tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _engine_relative(module: str, root: Path) -> str | None:
    """The engine file a dotted `woof...` name resolves to, or None."""

    stem = module.replace(".", "/")
    for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
        if (root / candidate).is_file():
            return candidate
    return None


def _imports_by_file() -> dict[str, set[str]]:
    """{engine module: {files in this package that import it}}."""

    collected = _boundary_instrument().collect(SRC)
    out: dict[str, set[str]] = {}
    for module, symbols in collected.items():
        for symbol, where in symbols.items():
            out.setdefault(module, set()).update(where)
            # `from woof.core import health_ledger` is recorded as the
            # PACKAGE plus a symbol, and the file it actually reaches is
            # `woof/core/health_ledger.py`.  A closure that only resolved
            # the module half would miss every submodule imported that way,
            # which on this tree is the ledger, the constants and the
            # companion-data resolver.
            out.setdefault(f"{module}.{symbol}", set()).update(where)
    return out


def test_the_manifest_ships_inside_the_package():
    path = manifest_path()
    assert path.is_file(), (
        f"{path} is missing from this install; without it the doctor cannot "
        "say whether the engine underneath the carried physics is the one "
        "this package was measured against")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == SEAM_SCHEMA
    assert payload["engine"]["distribution"] == "woof"
    assert payload["engine"]["version"]
    assert len(payload["files"]) >= 20


def test_every_row_says_what_this_package_reaches_in_it():
    """A "moved" row a reader cannot act on is a row that wastes the trip."""

    for entry in load_manifest()["files"]:
        assert entry["path"].startswith("woof/"), entry["path"]
        assert entry["path"].endswith(".py"), entry["path"]
        assert len(entry["sha256"]) == 64, entry["path"]
        # `>= 0`, not `> 0`: `woof/core/__init__.py` is EMPTY at the pinned
        # version, and a pin on an empty file is the thing that catches code
        # arriving in it -- which would then execute on every import of the
        # carried physics.
        assert entry["size"] >= 0, entry["path"]
        assert len(entry.get("reached", "")) > 30, entry["path"]


def test_the_manifest_names_no_file_this_package_carries():
    """A carried file cannot move underneath us; a pin on one is noise."""

    # Non-empty is asserted first: a carve table this test could not
    # read would make the intersection empty and this row green.
    assert len(CARRIED) >= 13, sorted(CARRIED)
    named = {entry["path"] for entry in load_manifest()["files"]}
    assert not (named & CARRIED), sorted(named & CARRIED)


@needs_pinned_engine
def test_the_pins_match_the_installed_engine():
    """The measurement, on this machine, against this wheel.

    Not an assertion about which engine is installed: it reports what it
    found.  A moved row fails here rather than in a run, and the message
    names the file and what the package reaches in it.
    """

    root = engine_root()
    assert root is not None, "the engine is installed but has no file path"
    rows = check_seam(root)
    moved = [row for row in rows if row.verdict != "proven"]
    assert not moved, "\n".join(
        f"{row.path}: {row.verdict}; pinned {row.pinned_sha256[:12]}, "
        f"installed {(row.found_sha256 or 'absent')[:12]}; this package "
        f"reaches {row.reached}" for row in moved)


def test_the_doctor_reports_the_seam_and_never_refuses_on_it():
    from woof.globe.doctor import build_report

    report = build_report()
    rows = None
    for title, section in report.sections:
        if title == "engine seam":
            rows = section
    assert rows is not None, "the doctor has no engine seam section"
    assert any(row.label == "seam" for row in rows)
    for row in rows:
        assert row.verdict != "gap", (
            f"{row.label} is reported as a gap; a moved seam file is unproven, "
            "not a refusal -- the dependency ceiling is the refusal")


@needs_pinned_engine
def test_the_pin_tool_reads_the_engine_rather_than_the_manifest():
    """`--check` must be able to disagree, or it proves nothing."""

    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "pin_engine_seam.py"), "--check"],
        capture_output=True, text=True, cwd=str(REPO))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "proven against woof" in proc.stdout, proc.stdout


# --------------------------------------------------------- the closure

def test_every_engine_file_the_carried_physics_imports_is_carried_or_pinned():
    """The table covers what the carved physics reaches, or it says so falsely.

    THE BREAKAGE THIS PREVENTS, and it had already happened when this test was
    written.  The first seam table was assembled from the engine names the
    carved SCHEMES bind, which is not the closure: twenty-four further engine
    files were imported by code under `src/arwen_global/core/` and pinned
    nowhere, twelve of them at module scope, so they executed on every import
    of the carried physics driver.  Six of the twenty-four differed between
    the published engine and the tree this model was graded in.  Nothing was
    measurably wrong -- the six differ in code no door of this package enters
    -- but `woof global doctor` printed "N/N files proven" over a table that
    covered half of what the carried physics imports, and a 2.7.1 that moved
    one of them would have changed what the driver imports with no row
    anywhere.

    The closure is computed here from the same AST walk the boundary
    instrument uses, so this is a set difference rather than a second
    instrument with its own opinion.
    """

    root = engine_root()
    assert root is not None, "the engine is installed but has no file path"
    pinned = {entry["path"] for entry in load_manifest()["files"]}
    missing: list[str] = []
    for module, files in sorted(_imports_by_file().items()):
        inside = sorted(f for f in files if f.startswith("src/arwen_global/core/"))
        if not inside:
            continue
        relative = _engine_relative(module, root)
        if relative is None:
            # Absent from the installed engine entirely.  That is a boundary
            # GAP, which `tools/measure_boundary.py` reports by name and the
            # doctor prints; it is not a file this table can pin.
            continue
        if relative in CARRIED or relative in pinned:
            continue
        missing.append(f"{relative} ({module}), imported by {', '.join(inside)}")
    assert not missing, (
        "an engine file the carried physics imports is neither carried nor "
        "pinned, so nothing reports it when the engine underneath moves:\n  "
        + "\n  ".join(missing))


def test_the_table_stays_inside_the_scope_it_states():
    """Scope drift the other way: a pin nobody can justify is not free.

    Every pinned row is reached by the carried physics, reached THROUGH
    another pinned file, or one of the two content dependencies outside the
    carved core.  The door modules -- the run plan, the fetchers, the obs
    readers, the writers -- are deliberately not here: what this package
    depends on there is an API, measured by symbol and by signature, and
    pinning their bytes would print a moved row on every engine release while
    saying nothing about the physics.
    """

    root = engine_root()
    assert root is not None
    reached_by_core: set[str] = set()
    for module, files in _imports_by_file().items():
        if not any(f.startswith("src/arwen_global/core/") for f in files):
            continue
        relative = _engine_relative(module, root)
        if relative:
            reached_by_core.add(relative)
    allowed = reached_by_core | INDIRECT | OUTSIDE_THE_CORE
    pinned = {entry["path"] for entry in load_manifest()["files"]}
    stray = sorted(pinned - allowed)
    assert not stray, (
        "a pinned row is outside the scope this table states: it is not "
        "imported by the carried physics, not one of the files reached "
        "through another pinned file, and not one of the two content "
        "dependencies outside the carved core:\n  " + "\n  ".join(stray))


def test_the_closure_walk_actually_sees_this_package():
    """A walk that resolved to nothing would make both tests above green."""

    imports = _imports_by_file()
    assert len(imports) > 40, len(imports)
    core = {module for module, files in imports.items()
            if any(f.startswith("src/arwen_global/core/") for f in files)}
    assert len(core) > 20, sorted(core)
    assert "woof.core.constants" in core
