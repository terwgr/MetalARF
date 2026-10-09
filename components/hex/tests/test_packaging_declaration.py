"""What the distribution declares must match what the package actually is.

Every assertion here exists because the opposite has cost somebody a cut:
a version restated in two files and shipped wrong for four releases, a loose
package glob that swallowed a sibling distribution, a dependency table that
did not name a module the code imports at line one, a data file that never
reached the wheel.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import sys
import tomllib

import pytest

import woof.hex


#: Where this file finds what it checks.  Two layouts carry these tests:
#:
#: * the standalone woof hex tree, where this component IS the project:
#:   ``pyproject.toml``, ``MANIFEST.in`` and ``src/hexcore`` sit beside
#:   ``tests/``;
#: * recast-woof, where the fold puts the package at ``woof/hex``, these
#:   tests, the docs and the tools under ``components/hex/``, and the one
#:   declaration (``pyproject.toml``, ``MANIFEST.in``, ``LICENSE``) at the
#:   repository root.
#:
#: THE BREAKAGE THE SPLIT PREVENTS: with the standalone paths hard-wired,
#: every test here errored or failed in recast-woof on a file that layout
#: does not have, so nothing held the folded package to its declaration.
FOLDED = woof.hex.__name__ != "hexcore"
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1] if FOLDED else ROOT
PYPROJECT = PROJECT / "pyproject.toml"
PACKAGE = (
    PROJECT.joinpath(*woof.hex.__name__.split(".")) if FOLDED else ROOT / "src" / "hexcore"
)
#: The top-level import name of this tree's own code.
OWN_TOP = woof.hex.__name__.split(".")[0]
#: The hex release this tree's documents describe.  Standalone it is also the
#: declared version; folded, the project declares its own version and the hex
#: manual still names the hex release it documents.
HEX_RELEASE = "0.3.2"

#: Distribution names on PyPI vs the module they install.  Only the entries
#: that actually differ; anything else is assumed to match.
_DISTRIBUTION_OF_MODULE = {"netCDF4": "netCDF4", "cupy": "cupy"}


@pytest.fixture(scope="module")
def declaration() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the version, stated once
# ---------------------------------------------------------------------------
def test_the_version_is_stated_in_exactly_one_place(declaration: dict) -> None:
    declared = declaration["project"]["version"]
    assert FOLDED or declared == HEX_RELEASE, (
        "0.3.2 is the declared cut: the release forced by woof 2.8.0 (its "
        "pin window excludes every earlier engine, 2.7.4 included), carrying "
        "the forecast from an install; moving it is a release decision, not "
        "an edit"
    )

    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    literals = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    ]
    # "0+unknown" is allowed and is the point: it is what the package says
    # when it is NOT installed, and it says so rather than inventing a number.
    # Anything shaped like a release is a second source of truth.
    literals = [text for text in literals if re.match(r"^\d+\.\d+", text)]
    assert literals == [], (
        "__version__ is assigned a release-shaped string literal in "
        "woof/hex/__init__.py.  "
        "That makes the number a promise to update two files at every cut, "
        "and woof broke exactly that promise: its wheel shipped 0.1.1 for "
        f"four releases.  Read it from installed metadata instead.  Found: {literals}"
    )
    assert "importlib.metadata" in source or "_distribution_version" in source, (
        "the package must read its version back out of installed metadata"
    )


def test_the_manual_names_the_declared_version_and_no_other(
    declaration: dict,
) -> None:
    """The manual's own version statements must be the declared version.

    THE BREAKAGE THIS PREVENTS, MEASURED (2026-08-27).  ``docs/manual/``
    was brought current for 0.2.0 chapter by chapter, and
    ``docs/manual/index.md`` -- the page a new reader opens first -- was
    left saying "This manual describes woof hex 0.1.0" and "the installed
    0.1.0 wheel for the doors".  A fresh-user walk of the printed commands
    found it in the first minute.  A reader who believes that line has no
    reason to trust the chapter numbers, the flags or the transcripts,
    because the document has just told them it describes a different
    release.  Nothing else in the packaging surface could see it: the
    declaration is right, the wheel is right, and only the prose is wrong.
    """
    declared = HEX_RELEASE if FOLDED else declaration["project"]["version"]
    index = ROOT / "docs" / "manual" / "index.md"
    assert index.is_file(), f"the manual's index is missing: {index}"
    text = index.read_text(encoding="utf-8")

    stated = re.findall(r"woof hex (\d+\.\d+\.\d+)", text)
    assert stated, (
        "the manual index no longer states which release it describes; "
        "state it, so this gate has something to check"
    )
    wrong = sorted({found for found in stated if found != declared})
    assert wrong == [], (
        f"docs/manual/index.md names woof hex {wrong} while the "
        f"declaration is {declared}.  The index is the page a new reader "
        "opens first; a stale release number there discredits every chapter "
        "behind it."
    )

    wheel = re.findall(r"installed (\d+\.\d+\.\d+) wheel", text)
    wrong_wheel = sorted({found for found in wheel if found != declared})
    assert wrong_wheel == [], (
        f"docs/manual/index.md says the commands were proven against the "
        f"installed {wrong_wheel} wheel while this cut is {declared}.  That "
        "sentence is the manual's whole warranty; it has to name this cut."
    )


def test_the_runtime_version_agrees_with_the_declaration(declaration: dict) -> None:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as distribution_version

    import woof.hex

    try:
        installed = distribution_version(woof.hex.DISTRIBUTION_NAME)
    except PackageNotFoundError:
        assert woof.hex.__version__ == "0+unknown", (
            "with no installed distribution the package must say it does not "
            "know its version, not invent one"
        )
        pytest.skip(
            "woof hex is not installed in this interpreter (a bare "
            "PYTHONPATH=src checkout); the agreement is proven by the "
            "packaging tier, which installs the built wheel first"
        )
    assert installed == declaration["project"]["version"]
    assert woof.hex.__version__ == installed


# ---------------------------------------------------------------------------
# the front doors
# ---------------------------------------------------------------------------
def test_the_console_script_reaches_a_real_entry_point(declaration: dict) -> None:
    scripts = declaration["project"]["scripts"]
    if FOLDED:
        # One distribution, one command: the hex model is a subcommand of the
        # project's own front door, which hands it the rest of the line.
        import importlib

        engine = importlib.import_module(woof.hex.__name__.rpartition(".")[0] + ".components")
        assert tuple(engine.COMPONENTS["hex"][:2]) == (f"{woof.hex.__name__}.cli", "main")
        assert not [name for name in scripts if "hex" in name], (
            f"a second console script for the hex model: {scripts}"
        )
    else:
        assert scripts == {"woof hex": "woof.hex.cli:main"}, (
            "one console script with subcommands, mirroring woof's shape; a "
            f"script per door was not the decision.  Found: {scripts}"
        )

    from woof.hex.cli import main

    assert callable(main)


def test_the_user_facing_names_carry_no_mpas_token(declaration: dict) -> None:
    """MPAS may be a trademark, so the distribution is spelled woof hex.

    The brand and the spellings a user types never use the MPAS name;
    nominative reference in prose documentation is fine and is used freely in
    README.md and docs/.
    """

    project = declaration["project"]
    assert "mpas" not in project["name"].lower()
    for script in project["scripts"]:
        assert "mpas" not in script.lower(), script

    from woof.hex.cli import build_parser

    parser = build_parser()
    assert parser.prog == "woof hex"
    for action in parser._subparsers._group_actions:  # type: ignore[union-attr]
        for name in getattr(action, "choices", {}):
            assert "mpas" not in name.lower(), name


def test_the_preferred_environment_spellings_carry_the_distribution_name() -> None:
    """...and the older ones still resolve.

    A rename that quietly stops reading a variable is the worst kind: the
    install line still looks right, the door still runs, and it picks up a
    different binary off PATH.  The legacy names stay in the ladder behind the
    preferred ones, permanently.
    """

    from woof.hex import render_door

    assert render_door.CONVERT_ENV == "WOOF_HEX_RW_MPAS_CONVERT"
    assert render_door.RENDERER_ENV == "WOOF_HEX_RW_WRFBATCH"
    for preferred in (render_door.CONVERT_ENV, render_door.RENDERER_ENV):
        assert "MPAS_PORT" not in preferred

    assert render_door.CONVERT_ENV_LEGACY == "MPAS_PORT_RW_MPAS_CONVERT"
    assert render_door.RENDERER_ENV_LEGACY == "MPAS_PORT_RW_WRFBATCH"

    source = (PACKAGE / "render_door.py").read_text(encoding="utf-8")
    for legacy in ("CONVERT_ENV_LEGACY", "RENDERER_ENV_LEGACY"):
        # Twice at least: the constant's definition, and its use in the
        # resolution ladder.  A constant that exists but is never consulted
        # is a promise the code does not keep.
        assert source.count(legacy) >= 3, (
            f"{legacy} is defined but does not appear in the resolution "
            "ladder and the refusal message"
        )


def test_the_import_namespace_carries_no_other_project_name() -> None:
    """The import name is ``woof.hex`` and no MPAS token reaches it.

    THE BREAKAGE THIS PREVENTS.  Through 0.1.1 the package installed as
    ``mpas_port``, so every user's import line carried another project's
    name for a relationship this port holds over only half the model: the
    DYCORE and the MESH are kept byte-identical to MPAS-A v8.4.1 and are
    pinned as a specification, while the physics is deliberately not MPAS's
    at all -- it is WOOF's column-batch seam.  A name is a claim, and that
    one was wrong in the direction that matters, towards somebody else's
    trademarkable project.  ``woof.hex`` names what IS pinned, the hexagonal
    Voronoi mesh and the dycore, and it matches the distribution.

    There is deliberately no ``mpas_port`` alias, and this test refuses one:
    a shim keeps the overclaiming spelling resolvable in an import line,
    which is the single thing the rename removes.
    """

    if FOLDED:
        declared = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        assert woof.hex.__name__ == "woof.hex"
        assert woof.hex.DISTRIBUTION_NAME == declared["project"]["name"]
    else:
        assert woof.hex.__name__ == "hexcore"
        assert woof.hex.DISTRIBUTION_NAME == "woof hex"

    assert not (PACKAGE.parent / "mpas_port").exists(), (
        "src/mpas_port is back.  The 0.2.0 rename removed it; a directory "
        "under the old name means either an unfinished move or a shim, and "
        "a shim keeps another project's name in the user's import line"
    )
    import importlib.util

    assert importlib.util.find_spec("mpas_port") is None, (
        "mpas_port is importable in this interpreter.  Either a stale "
        "installed copy of the pre-0.2.0 package is on the path -- which "
        "will shadow or be shadowed by hexcore depending on cwd -- or an "
        "alias shim was added.  Neither may ship"
    )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "import namespace" in readme.lower(), (
        "README must state what the installed import name is, because the "
        "0.2.0 rename breaks every script that imported the old one"
    )


# ---------------------------------------------------------------------------
# the dependency table
# ---------------------------------------------------------------------------
def _module_scope_third_party_imports() -> set[str]:
    found: set[str] = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import, always our own package
                    continue
                names = [node.module or ""]
            for name in names:
                top = name.split(".")[0]
                if not top or top in sys.stdlib_module_names or top == OWN_TOP:
                    continue
                found.add(top)
    return found


def test_every_module_scope_import_is_a_declared_dependency(
    declaration: dict,
) -> None:
    declared = {
        entry.split(">")[0].split("=")[0].split("[")[0].split("<")[0].strip().lower()
        for entry in declaration["project"]["dependencies"]
    }
    for module in sorted(_module_scope_third_party_imports()):
        distribution = _DISTRIBUTION_OF_MODULE.get(module, module).lower()
        assert distribution in declared, (
            f"{module} is imported at module scope somewhere under "
            f"src/hexcore -- so the package does not import without it -- "
            f"but {distribution} is not in [project].dependencies.  A "
            "dependency that is real at line one and optional in the table "
            "produces a green install and an ImportError on first use"
        )


def test_the_declared_engine_range_is_the_one_engine_identity_names(
    declaration: dict,
) -> None:
    """The woof specifier is stated once, in ``woof.hex.engine_identity``.

    Through 0.3.1 the specifier was derived from a table of measured
    verdicts, one exact engine wide, because a separately published engine
    could carry seam bytes this port was never run with.  The engine now
    ships with this port and that pin retired; the standalone distribution
    declares the engine minor its CPU suite and seam contract test ran
    against, and a dependency specifier has to be static metadata, so this
    test holds the literal and the constant equal.
    """

    from woof.hex import engine_identity

    if FOLDED:
        # The engine is this distribution: there is no range to declare, and
        # the remedy names the one distribution that carries both.
        assert engine_identity.ENGINE_REQUIREMENT == declaration["project"]["name"]
        assert len(engine_identity.SEAM_PATHS) == 16
        assert "docs/mpas-seam.md" in engine_identity.SEAM_PATHS
        return
    pins = [
        entry
        for entry in declaration["project"]["dependencies"]
        if re.split(r"[<>=!~\[]", entry, maxsplit=1)[0].strip() == "woof"
    ]
    assert pins == [engine_identity.ENGINE_REQUIREMENT], (
        "the declared woof specifier must be exactly "
        f"woof.hex.engine_identity.ENGINE_REQUIREMENT "
        f"({engine_identity.ENGINE_REQUIREMENT!r}); declared: {pins}"
    )
    assert len(engine_identity.SEAM_PATHS) == 16
    assert "docs/mpas-seam.md" in engine_identity.SEAM_PATHS, (
        "the seam contract document is half of the contract surface the "
        "seam contract test holds, and the engine ships it in site-packages"
    )


# ---------------------------------------------------------------------------
# what reaches the wheel
# ---------------------------------------------------------------------------
def test_package_discovery_is_anchored(declaration: dict) -> None:
    find = declaration["tool"]["setuptools"]["packages"]["find"]
    if FOLDED:
        # The project's own anchored pair reaches this package, and the
        # held-out ``mod`` subpackage is excluded by exact name.
        assert {OWN_TOP, f"{OWN_TOP}.*"} <= set(find["include"])
        assert {f"{woof.hex.__name__}.mod", f"{woof.hex.__name__}.mod.*"} <= set(
            find.get("exclude", [])
        )
        return
    assert find["where"] == ["src"]
    assert find["include"] == ["hexcore", "hexcore.*"], (
        "anchored patterns, never a loose glob: these are fnmatch over "
        "directory names, and woof measured a loose glob swallowing a whole "
        "sibling distribution into its wheel"
    )


def test_no_package_data_file_lands_outside_the_declaration(
    declaration: dict,
) -> None:
    from fnmatch import fnmatch

    patterns = declaration["tool"]["setuptools"]["package-data"][woof.hex.__name__]
    stray: list[str] = []
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_dir() or path.suffix == ".py":
            continue
        if "__pycache__" in path.parts:
            continue  # a build product of the sources beside it, never input
        relative = path.relative_to(PACKAGE).as_posix()
        if not any(fnmatch(relative, pattern) for pattern in patterns):
            stray.append(relative)
    assert stray == [], (
        "these files live under the package but match no package-data "
        f"pattern, so the wheel will not carry them: {stray}"
    )


# ---------------------------------------------------------------------------
# the blockers a public cut has to clear
# ---------------------------------------------------------------------------
def test_the_licence_blocker_is_declared_until_a_licence_exists(
    declaration: dict,
) -> None:
    """No LICENSE file, therefore no licence claim in the metadata.

    This port derives from MPAS-Atmosphere v8.4.1 (NCAR/LANL, BSD-3-Clause)
    with source-line citations pinned by tools/check_mpas_citations.py.  What
    licence a derived work ships under is a release decision.  Until a LICENSE and
    NOTICE exist in this tree, the metadata must stay silent rather than
    guess -- a wrong SPDX expression in published metadata is worse than an
    absent one, because it is a claim.
    """

    has_licence_file = (PROJECT / "LICENSE").is_file() or (PROJECT / "LICENCE").is_file()
    project = declaration["project"]
    if has_licence_file:
        assert "license" in project, (
            "a LICENSE now exists in the tree, so [project].license must "
            "carry its SPDX expression and [project].license-files must ship "
            "it in the wheel"
        )
        return
    assert "license" not in project and "license-files" not in project, (
        "metadata declares a licence that no file in this tree backs"
    )


# ---------------------------------------------------------------------------
# what the sdist template actually resolves to
#
# `MANIFEST.in` is a template language, not a list, and it fails SILENTLY in
# both directions.  Both directions have already happened here:
#
#   * `include evidence/EVIDENCE.md` named a file that did not exist in this
#     tree.  setuptools warns and carries on, so the line read as protection
#     for two years' worth of `evidence/...` links while packing nothing, and
#     `docs/source-matrix.md`'s link into it was dead in every sdist built
#     from this tree.  The file existed only in the assembled public tree,
#     which is why the audit there passed.
#
#   * `graft docs` is a wildcard, and a wildcard is how the thing nobody
#     listed gets shipped.  `docs/LANE-BRIEFING.md` is the internal working
#     brief: a private LAN address, a home-directory path, the maintainer's
#     name against the machines he owns, the card reservation protocol, and
#     the existence of a second private effort.  It travelled in the sdist
#     from the day it landed.
#
# These tests drive the REAL template engine over the REAL tree, which is the
# only way to read a MANIFEST.in without guessing.
# ---------------------------------------------------------------------------
def _sdist_template_files() -> set[str]:
    # setuptools is a TEST dependency of this file and is declared as one in
    # [project.optional-dependencies].dev.  THE BREAKAGE THIS SKIP PREVENTS
    # BEING SILENT, measured on the proving RTX 5090 (evidence/xmachine-20260827/): since
    # Python 3.12 `ensurepip` no longer seeds setuptools, so the three tests
    # below raised ModuleNotFoundError in the sdist a user unpacks -- on 3.13
    # and not on 3.11, which is how it survived every run on the desk it was
    # written on.  Absent is a different finding from wrong, so it says which
    # it is; and the dev extra plus ci.yml's install list are what stop the
    # skip from ever being the answer where the gate matters.
    try:
        from setuptools._distutils.filelist import FileList
    except ModuleNotFoundError:  # pragma: no cover - depends on the venv
        pytest.skip(
            "setuptools is not installed in this interpreter, and reading a "
            "MANIFEST.in means driving its real template engine -- guessing "
            "what the template resolves to is exactly what these tests exist "
            "to stop.  Install the test dependency: pip install "
            '"woof hex[dev]" (or setuptools>=77)'
        )

    file_list = FileList()
    cwd = Path.cwd()
    try:
        import os

        os.chdir(PROJECT)
        file_list.findall(".")
        for raw in (PROJECT / "MANIFEST.in").read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                file_list.process_template_line(line)
    finally:
        import os

        os.chdir(cwd)
    return {name.replace("\\", "/").lstrip("./") for name in file_list.files}


def test_every_path_the_manifest_includes_exists() -> None:
    """An `include` naming a missing file is a silent no-op, and packs nothing.

    THE ASYMMETRY, and it is the whole point of this test's shape.  The two
    directives fail in opposite ways:

    * an `include` that names nothing PACKS nothing, and every document
      linking to that path then points at a file the reader does not have.
      That has already happened here -- `include evidence/EVIDENCE.md` named
      a file this tree did not carry, so `docs/source-matrix.md`'s link into
      it was dead in every sdist built from this tree.
    * an `exclude` that names nothing LEAKS nothing.  Absence is the
      strongest form the exclusion can take.

    So requiring every excluded path to exist -- which this test used to do
    -- makes the published tree carry the very documents it excludes.  It
    did exactly that: the assembled public tree drops `docs/LANE-BRIEFING.md`
    on purpose, and this assertion then failed on both CI matrix legs, on the
    release commit, with `ci.yml` firing on push
    (`evidence/xmachine-20260827/`, `evidence/assembly-rehearsal-20260827/`).

    What replaces it for the exclusions is an OUTCOME check, below: the thing
    that must not travel is asserted absent from the real packed set, whether
    it is absent because it was excluded or because it is not in the tree at
    all.  A misspelled `exclude` is caught there, by the path that matters,
    rather than here by a path that does not.
    """

    named: list[str] = []
    for raw in (PROJECT / "MANIFEST.in").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("include "):
            named.extend(line.split(None, 1)[1].split())

    missing = [
        path
        for path in named
        if "*" not in path and "?" not in path and not (PROJECT / path).exists()
    ]
    assert missing == [], (
        f"MANIFEST.in includes {missing}, which do not exist in this tree.  "
        "setuptools warns and continues, so the directive packs nothing "
        "and any document linking to it is pointing at a file the reader "
        "does not have -- the exact defect the grafts were written to prevent"
    )


def test_the_evidence_pointer_reaches_the_sdist() -> None:
    """README and docs/source-matrix.md both link it; it has to be there."""

    if FOLDED:
        pytest.skip(
            "recast-woof folds the hex package, tests, tools, docs and "
            "verification but no evidence/ directory, so its sdist has no "
            "hex receipt pointer to carry; the pointer ships with the "
            "standalone hex source tree, where this test holds it"
        )
    assert "evidence/EVIDENCE.md" in _sdist_template_files(), (
        "docs/source-matrix.md links `../evidence/EVIDENCE.md`, which is what "
        "tells a reader that an `evidence/...` reference names the IDENTITY "
        "of a measurement rather than a file they are missing.  Without it "
        "in the sdist that link resolves to nothing"
    )


#: Documents that are internal working material and must never be inside a
#: published artefact.  Held as data because the assertion below is the same
#: three times over and a fourth document should be a row, not a new test.
#: The release checklist is covered by the pattern check below rather than
#: named here, because its filename carries a version and a named row would
#: stop covering the next one.
_MUST_NOT_TRAVEL = ("docs/LANE-BRIEFING.md",)


def test_the_internal_working_brief_does_not_travel() -> None:
    """`docs/LANE-BRIEFING.md` is not a user document and must not ship.

    THE INVARIANT IS THE OUTCOME, not the mechanism, and that distinction is
    what this test got wrong.  It used to open by asserting the briefing
    EXISTS -- so that an exclusion protecting nothing could not read as a
    pass -- and the published tree, which deliberately does not carry the
    briefing at all, then failed its own packaging gate on both CI matrix
    legs (`evidence/xmachine-20260827/` §4).  An excluded internal document
    is not something a published tree should be required to carry.

    So: the document is asserted absent from the real packed set either way,
    and where it IS present the exclusion is additionally proved to be doing
    the work -- `graft docs` would otherwise pack it, so a `graft` that no
    longer reaches it, or an `exclude` misspelled, is caught by the positive
    control rather than by an existence assertion.
    """

    packed = _sdist_template_files()
    present = [
        _in_project(relative)
        for relative in _MUST_NOT_TRAVEL
        if (ROOT / relative).is_file()
    ]
    if present:
        # Positive control: the wildcard that would otherwise carry them is
        # live.  Without this, a `graft docs` that had stopped reaching docs/
        # would make the exclusions pass while proving nothing.
        assert any(name.startswith(_in_project("docs") + "/") for name in packed), (
            f"{present} are in this tree but no docs/ file reaches the sdist "
            "at all, so the exclusions are not being tested against the "
            "wildcard that would carry them; check `graft docs` in MANIFEST.in"
        )
    for relative in _MUST_NOT_TRAVEL:
        assert _in_project(relative) not in packed, (
            f"{relative} is in the sdist.  It is internal working material, "
            "not a user document, and an sdist on PyPI cannot be recalled"
        )
    leaked = sorted(name for name in packed if "release-checklist" in name)
    assert leaked == [], (
        f"the release checklist is in the sdist: {leaked}.  It names the "
        "private assembly tree, the scrub policy, and the ordered commands "
        "for acts that need explicit approval; shipping it inside the "
        "artefact it describes publishes the recipe for its own publication"
    )


#: The two `mod` directories the packaging holds out, in the spelling the
#: sdist template reports.  A PREFIX match, never a substring
#: match: the directory is now named ``mod``, and ``"mod" in name`` is true of
#: half this tree ("model", "module", "modified").  The old spelling of this
#: gate matched a substring, which only worked because the substring happened
#: to be a rare token; a rename to a common one turns that into a gate that
#: fires on everything or, if inverted, on nothing.
def _in_project(relative: str) -> str:
    """A path under this component, spelled as the sdist template reports it."""

    return (ROOT / relative).relative_to(PROJECT).as_posix()


_PRIVATE_PROVIDER_PREFIXES = (
    (PACKAGE / "mod").relative_to(PROJECT).as_posix() + "/",
    _in_project("tests/mod") + "/",
)


def test_the_sibling_seam_reaches_neither_half_of_the_sdist() -> None:
    """The package exclusion covers the wheel; this covers the sdist."""

    leaked = sorted(
        name
        for name in _sdist_template_files()
        if name.replace("\\", "/").startswith(_PRIVATE_PROVIDER_PREFIXES)
    )
    assert leaked == [], (
        f"a held-out `mod` directory is in the sdist: {leaked}.  It is not "
        "part of this distribution, and an sdist on PyPI cannot be recalled"
    )
