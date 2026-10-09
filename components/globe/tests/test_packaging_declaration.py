"""The packaging declaration is a contract, so it is tested like one.

Every gate here exists because the same mistake was made once already on a
neighbouring distribution and shipped:

  - a runtime version constant drifted four releases behind the wheel;
  - a `find` glob without an anchor swallowed a sibling project directory;
  - a per-file package-data enumeration fell 52 files behind its own tree;
  - an sdist shipped tests without the conftest that makes them skip.

None of those failed loudly.  Each produced an artefact that installed, ran,
and was wrong.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re
import sys
import tomllib

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "arwen_global"
PYPROJECT = REPO / "pyproject.toml"


@pytest.fixture(scope="module")
def declaration() -> dict:
    with PYPROJECT.open("rb") as stream:
        return tomllib.load(stream)


# ---------------------------------------------------------------- identity

def test_distribution_and_import_names(declaration):
    """The three names are what the carve settled on, stated once each."""

    assert declaration["project"]["name"] == "woof global"
    assert list(declaration["project"]["scripts"]) == ["woof global"]
    assert (declaration["project"]["scripts"]["woof global"]
            == "woof.globe.cli:main")
    assert (declaration["tool"]["setuptools"]["packages"]["find"]["include"]
            == ["arwen_global", "arwen_global.*"])


def test_version_is_stated_exactly_once(declaration):
    """pyproject states the number; nothing under src/ restates it.

    THE BREAKAGE THIS PREVENTS, and it is measured rather than imagined: the
    engine shipped a wheel whose metadata said one version while a
    hand-maintained runtime constant beside it said another, for four
    releases, and the refusal messages users quoted named the wrong release.
    A second literal is a promise to update two files at every cut.
    """

    version = declaration["project"]["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), version

    pattern = re.compile(r"""["']%s["']""" % re.escape(version))
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(REPO)}:{number}: {line.strip()}")
    assert not offenders, (
        "the version literal appears under src/ as well as in pyproject:\n  "
        + "\n  ".join(offenders))


def test_version_is_read_back_from_metadata():
    """`woof.globe._version` asks importlib.metadata, it does not restate."""

    source = (SRC / "_version.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "__version__" in assigned
    assert "importlib.metadata" in source or "from importlib.metadata" in source
    assert "PackageNotFoundError" in source, (
        "an uninstalled source tree must say so, not invent a number")


def test_licence_is_an_spdx_expression_with_both_files(declaration):
    """The distribution's own licence, its NOTICE, and the texts it owes.

    `licenses/*` joined the declaration at 0.1.1 and is not decoration.
    Four of the works the carried physics transcribes condition
    redistribution on their TEXT travelling with the copy rather than on
    being named: MIT over Arm's libm cores, the FDLIBM/SunPro notice,
    BSD-3-Clause clause 1 over RTE+RRTMGP, and UCAR's request over the four
    WRF tables.  0.1.0 shipped all four transcriptions with LICENSE and
    NOTICE alone and performed none of them.  The glob is asserted AS a
    glob: the set of texts grows with what the package carries, and the pin
    belongs on the declaration, with tests/test_licence_notices_ship.py
    holding the inventory.
    """

    project = declaration["project"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE", "NOTICE", "licenses/*"]
    for name in project["license-files"]:
        if name.endswith("/*"):
            directory = REPO / name[:-2]
            assert directory.is_dir(), f"{name} globs a missing directory"
            assert sorted(directory.glob("*.txt")), f"{name} globs nothing"
            continue
        assert (REPO / name).is_file(), f"{name} is declared and missing"
    classifiers = project.get("classifiers", [])
    assert not [row for row in classifiers if row.startswith("License ::")], (
        "PEP 639 replaced the trove classifier with the expression above, and "
        "setuptools>=77 refuses a build that states both")
    assert declaration["build-system"]["requires"] == ["setuptools>=77"], (
        "below 77 the SPDX expression silently becomes the old License: field "
        "and the licence files are dropped from the wheel")


def test_engine_dependency_is_bounded_at_both_ends(declaration):
    """The floor AND the ceiling, because a ceiling is the whole point.

    An unbounded floor let pip resolve an engine whose bytes a neighbouring
    distribution then refused at launch, while its doctor reported the estate
    healthy and exited 0.
    """

    deps = declaration["project"]["dependencies"]
    engine = [row for row in deps if row.split(">")[0].strip() == "woof"]
    assert len(engine) == 1, deps
    assert ">=2.8.0" in engine[0] and "<2.9" in engine[0], engine[0]


def test_the_companion_data_distribution_is_declared(declaration):
    """It became an IMPORT requirement the day the physics was carried in.

    `woof/globe/core/rrtmgp.py` calls `data_assets.rrtmgp_data_dir()` at
    module scope and resolves a member of the companion wheel before its
    first function is defined, so `import woof.globe.core.rrtmgp` fails
    without it.  Before the carve that call lived in the engine and the
    companion was the engine's business.  It arrives today only through the
    engine's own `recast-woof-data==2.8.0`, which is exactly the transitive route
    this file refuses for netCDF4: a dependency the owner can drop without
    anything here changing, and the failure is an ImportError after an
    install that reported success.
    """

    deps = declaration["project"]["dependencies"]
    companion = [row for row in deps
                 if re.split(r"[<>=!~\[]", row, maxsplit=1)[0].strip()
                 == "recast-woof-data"]
    assert len(companion) == 1, deps
    assert ">=2.8.0" in companion[0] and "<2.9" in companion[0], companion[0]

    source = (SRC / "core" / "rrtmgp.py").read_text(encoding="utf-8")
    module_scope = [line for line in source.splitlines()
                    if line.startswith("DATA_DIR = data_assets.")]
    assert module_scope, (
        "the reason for the row above is a module-scope call in the carried "
        "radiation driver; it is no longer there, so re-take the decision "
        "rather than leaving a dependency nobody can trace")


# ------------------------------------------------------------ package data

def _globs(declaration) -> list[str]:
    return declaration["tool"]["setuptools"]["package-data"]["arwen_global"]


def test_package_data_globs_are_recursive_not_an_enumeration(declaration):
    """No row may name one file.

    The engine's enumeration drifted 52 files behind its tree before it was
    rewritten this way.  A glob cannot drift.
    """

    for glob in _globs(declaration):
        assert "*" in glob, f"{glob!r} names a file rather than a pattern"


def test_every_shipped_data_file_matches_a_glob(declaration):
    """Every non-Python file under the package is selected by some row.

    This is the gate that fails the day somebody adds a table, a kernel or a
    config and does not notice that the wheel left it behind.  A package that
    imports and then cannot find its own data is the shape a user reports as
    "it installed and then said the config does not exist".
    """

    import fnmatch

    globs = _globs(declaration)
    missed = []
    for path in sorted(SRC.rglob("*")):
        if not path.is_file() or path.suffix == ".py":
            continue
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(SRC).as_posix()
        if not any(_matches(relative, glob) for glob in globs):
            missed.append(relative)
    assert not missed, (
        "these files live under the package and no package-data row selects "
        "them, so the wheel would not carry them:\n  " + "\n  ".join(missed))


def _matches(relative: str, glob: str) -> bool:
    import fnmatch

    if "**" in glob:
        # setuptools treats ** as "any number of path segments"; fnmatch does
        # not, so the equivalent is checked directly.
        head, _, tail = glob.partition("**/")
        if not relative.startswith(head):
            return False
        rest = relative[len(head):]
        return fnmatch.fnmatch(rest.rsplit("/", 1)[-1], tail) or bool(rest)
    if "/" in glob:
        return fnmatch.fnmatch(relative, glob)
    return "/" not in relative and fnmatch.fnmatch(relative, glob)


def test_the_configs_that_ship_are_the_ones_that_were_carved():
    """55 experiments, and every one of them parses as TOML.

    Fifty-three came out of the engine's `configs/verify/` at the first cut;
    the fifty-fourth is the quickstart, which lived one directory up and was
    missed by the first count.  Two of the carved tests name it, so a package
    without it ships a suite that cannot run and a document that points at a
    file the reader does not have.  The fifty-fifth is the closure door,
    carried at the 857cb277c re-cut: the T255 record config with one table
    changed, which its own test holds against the record.
    """

    configs = sorted((SRC / "configs").glob("*.toml"))
    assert len(configs) == 55, [path.name for path in configs]
    families = {path.name.split("_")[0] for path in configs}
    assert families == {"arwen", "global"}, families
    for path in configs:
        with path.open("rb") as stream:
            tomllib.load(stream)


def test_every_kernel_source_ships(declaration):
    """Two kernel sets travel in this package, and both are declared.

    THE BREAKAGE.  The loader in `core/kernels/` binds its own directory and
    has no fallback to the engine's, so a wheel that carries the carried
    physics' Python and not its `.cu` files raises FileNotFoundError at the
    first radiation or surface-layer call -- on a card, minutes into a run,
    with an install that reported success.  The semi-Lagrangian core's kernel
    was the only one here until 2026-09-09 and this gate said so; it now names
    both sets.
    """

    sources = sorted(path.relative_to(SRC).as_posix()
                     for path in SRC.rglob("*.cu"))
    assert "semilag/kernels.cu" in sources
    carried = [name for name in sources if name.startswith("core/kernels/")]
    assert len(carried) == 12, carried
    assert sources == sorted(["semilag/kernels.cu"] + carried)
    headers = sorted(path.relative_to(SRC).as_posix()
                     for path in SRC.rglob("*.cuh"))
    assert headers == ["core/kernels/common.cuh",
                       "core/kernels/glibc_flt32.cuh",
                       "core/kernels/rrtmgp_planck_common.cuh"]
    # Declared, not merely present: a source the wheel writer never selects
    # is a source the reader does not get.
    globs = _globs(declaration)
    for name in sources + headers:
        assert any(_matches(name, glob) for glob in globs), name


# ------------------------------------------------------------------ sdist

@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib")
def test_manifest_carries_the_directories_the_documents_point_at():
    """An sdist whose README links resolve to nothing is a broken sdist."""

    manifest = (REPO / "MANIFEST.in").read_text(encoding="utf-8")
    # `licenses` is grafted for the same reason as the rest and for one
    # more: NOTICE points at every file in it by path, so an sdist without
    # the graft carries a notice whose every reference resolves to nothing.
    for directory in ("tests", "tools", "docs", "licenses"):
        assert f"graft {directory}" in manifest, directory
    for name in ("LICENSE", "NOTICE", "README.md"):
        assert f"include {name}" in manifest, name
    assert "prune **/__pycache__" in manifest


# --------------------------------------------------------------- the CI seam
#
# The gates below are the ones the CI lane needs true, and each names a
# breakage the workflows cannot catch on their own.  They are source-level
# like the rest of this file: a packaging gate that needs the package
# installed before it can speak is a gate that says nothing about the tree
# it is run in.


def test_package_discovery_is_anchored(declaration):
    """Anchored include patterns, never a loose glob.

    These are fnmatch over DIRECTORY names walked from `src/`, `find`
    defaults to `namespaces = true`, and a namespace finder does not need a
    directory name to be a valid identifier.  The engine measured what that
    costs: its 2.5.0 wheel swallowed a whole sibling project directory.
    There is no sibling under `src/` today, and the anchor is what keeps
    that true when one appears.
    """

    find = declaration["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["src"]
    assert find["include"] == ["arwen_global", "arwen_global.*"], find


def test_the_console_script_is_one_script_naming_a_real_target(declaration):
    """One script with subcommands, pointing at a module that exists.

    An entry point that resolves to nothing is a green install and a
    command that dies on its first use.  The target is checked as SOURCE
    (the module file is there and defines the attribute) so this gate holds
    in a checkout; the CI job additionally runs the installed script's
    --help before it runs the suite, which is the half a source check
    cannot do.
    """

    scripts = declaration["project"]["scripts"]
    assert list(scripts) == ["woof global"], (
        "one console script with subcommands, mirroring the engine and the "
        f"hex line rather than a script per door: {scripts}")

    module_name, _, attribute = scripts["woof global"].partition(":")
    assert module_name.startswith("arwen_global."), module_name
    module_path = REPO / "src" / Path(module_name.replace(".", "/") + ".py")
    assert module_path.is_file(), (
        f"the console script names {module_name}, and "
        f"{module_path.relative_to(REPO)} does not exist")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    defined = {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert attribute in defined, (
        f"the console script names {scripts['woof global']} and "
        f"{module_name} defines no {attribute}()")


def test_every_module_scope_import_is_a_declared_dependency(declaration):
    """A dependency real at line one and absent from the table.

    That combination installs green and raises ImportError on first use.
    Module scope is the distinction that matters: an import inside a
    function is one path's runtime requirement, while an import at module
    scope means the package does not import at all without it.
    """

    declared = {
        re.split(r"[<>=!~\[]", entry, maxsplit=1)[0].strip().lower()
        for entry in declaration["project"]["dependencies"]
    }
    found: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        if path.relative_to(SRC).parts[0] == "core":
            # THE CARRIED ENGINE PHYSICS IS A DIFFERENT CONTRACT, and the
            # test below is the one that holds it.  These modules drive CUDA
            # kernels: reaching cupy at line one is correct there, and it is
            # an EXTRA (`woof global[gpu]`) because a pip extra cannot detect
            # a CUDA major.  This walk's premise -- module scope means the
            # package does not import at all without it -- is false for them:
            # `import woof.globe` succeeds on a host with no CUDA, and only
            # the card path raises.
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative, always our own package
                    continue
                names = [node.module or ""]
            for name in names:
                top = name.split(".")[0]
                if top and top not in sys.stdlib_module_names \
                        and top != "arwen_global":
                    found.add(top)
    missing = sorted(module for module in found if module.lower() not in declared)
    assert missing == [], (
        "imported at module scope under src/arwen_global, so the package "
        "does not import without them, and absent from "
        f"[project].dependencies: {missing}")


def test_the_carried_core_imports_only_declared_extras(declaration):
    """The other half of the rule above, for the code it steps over.

    The carried physics may reach cupy and netCDF4 at module scope, because
    it drives kernels and reads k-distribution tables.  It may not reach
    anything a reader cannot install: every top-level name it imports must be
    a declared dependency, a declared extra, or the engine itself.
    """

    declared = {
        re.split(r"[<>=!~\[]", entry, maxsplit=1)[0].strip().lower()
        for entry in declaration["project"]["dependencies"]
    }
    for entries in declaration["project"]["optional-dependencies"].values():
        for entry in entries:
            name = re.split(r"[<>=!~\[]", entry, maxsplit=1)[0].strip().lower()
            declared.add(name)
            # `cupy-cuda13x` is what pip installs and `cupy` is what the
            # source imports; the extra is what makes the import reachable.
            if name.startswith("cupy"):
                declared.add("cupy")

    core = SRC / "core"
    found: set[str] = set()
    for path in sorted(core.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                names = [node.module or ""]
            for name in names:
                top = name.split(".")[0]
                if top and top not in sys.stdlib_module_names \
                        and top != "arwen_global":
                    found.add(top)
    missing = sorted(module for module in found
                     if module.lower() not in declared)
    assert missing == [], (
        "imported at module scope by the carried engine physics under "
        f"src/arwen_global/core, and installable from no declared "
        f"dependency or extra: {missing}")


def test_the_build_backend_floor_is_what_the_licence_field_needs(declaration):
    """setuptools below 77 misstates this tree's licence in its own wheel.

    `license = "Apache-2.0"` is a PEP 639 SPDX expression.  Only
    setuptools>=77 turns it into a License-Expression metadata field and
    places LICENSE and NOTICE under `dist-info/licenses/`.  Below that
    floor the same correct source tree emits the old `License:` field and
    drops the licence files entirely.
    """

    requires = declaration["build-system"]["requires"]
    setuptools = [r for r in requires if re.split(r"[<>=!~\[]", r, maxsplit=1)[0].strip() == "setuptools"]
    assert len(setuptools) == 1, requires
    floor = re.search(r">=\s*(\d+)", setuptools[0])
    assert floor is not None and int(floor.group(1)) >= 77, requires


def test_the_files_the_declaration_names_exist(declaration):
    """A declaration naming a file that is not there fails at build time.

    `readme` and `license-files` are read by setuptools while building, and
    a missing one is a build error rather than a warning.  Naming them here
    means the failure arrives as a test that says which file, rather than a
    traceback out of the build backend on a release runner.
    """

    project = declaration["project"]
    missing = []
    readme = project.get("readme")
    if isinstance(readme, str) and not (REPO / readme).is_file():
        missing.append(readme)
    for name in project.get("license-files", []):
        if "*" not in name and "?" not in name and not (REPO / name).is_file():
            missing.append(name)
    assert missing == [], (
        "pyproject.toml names these files and this tree does not carry "
        f"them, so `python -m build` fails before it starts: {missing}")


def test_the_markers_the_ci_selection_deselects_are_registered(declaration):
    """`-m "not gpu and not slow and not network"` has to mean something.

    A marker expression naming a marker pytest does not know is not an
    error: the expression matches nothing to deselect, so the job runs the
    tests it was written to skip, on a runner with no card and no network.
    The selection lives in .github/workflows/test.yml and the registration
    lives in the declaration, so the two are checked against each other
    rather than each trusted on its own.
    """

    registered = {
        row.split(":", 1)[0].strip()
        for row in declaration["tool"]["pytest"]["ini_options"]["markers"]
    }
    workflow = (REPO / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8")
    match = re.search(r'-m "([^"]+)"', workflow)
    assert match is not None, "no marker expression in test.yml"
    used = set(re.findall(r"\bnot\s+([a-z_]+)", match.group(1)))
    assert used, "the marker expression deselects nothing"
    assert used <= registered, (
        f"test.yml deselects {sorted(used - registered)}, which "
        "[tool.pytest.ini_options].markers does not register")


def test_there_is_no_addopts_line(declaration):
    """A bare `pytest` must not be quietly shorter than it looks."""

    options = declaration["tool"]["pytest"]["ini_options"]
    assert "addopts" not in options, options.get("addopts")


def test_every_shell_script_that_ships_has_unix_line_endings():
    """A CRLF shebang is not a shebang.

    THE BREAKAGE THIS PREVENTS, and it happened here: a driver in `tools/`
    was edited on Windows through a text write that translated every `\n`
    to `\r\n`, so the first line of the shipped file became
    `#!/bin/bash\r`.  Linux resolves the interpreter as the literal string
    `/bin/bash\r`, which does not exist, and the script dies with
    "bad interpreter: No such file or directory" before a line of it runs.
    The file still opens correctly in every editor, `bash -n` on Windows
    still reports clean syntax, and the sdist still builds, so nothing
    upstream of a Linux user notices.

    Markdown and Python are deliberately not held to this: the repository
    already stores both with either ending, and a rule that demanded one
    would rewrite 56 files to say nothing about correctness.  A shell
    script is different because the ending is executable content.
    """

    scripts = sorted(REPO.glob("tools/**/*.sh")) + sorted(REPO.glob("*.sh"))
    assert scripts, "no shell script found; this gate would pass vacuously"
    carriage_return = chr(13).encode()
    bad = [
        path.relative_to(REPO).as_posix()
        for path in scripts
        if carriage_return in path.read_bytes()
    ]
    assert bad == [], (
        "a shell script that ships carries a carriage return, so its "
        "shebang line names an interpreter that does not exist on the "
        "platform it is written for:\n  " + "\n  ".join(bad)
    )


def test_the_published_boundary_count_is_the_measured_one():
    """The number two published pages state is the number the tool prints.

    THE BREAKAGE THIS PREVENTS, and it was in the tree when this gate was
    written: the README and the declaration both said this package imports
    "84 symbols across that boundary", in a sentence whose next line points
    the reader at `tools/measure_boundary.py` and says the measurement is
    regenerated rather than trusted.  The tool printed 76.  A page that
    names its own instrument and then disagrees with it is worse than a
    page with no number on it, because a reader has no way to tell which
    of the two is stale.

    The count is engine-independent on purpose: it is how many `from
    woof ...` targets the package's own source names, read out of the
    source with `ast`, so this gate runs the same on an engine that is
    behind and on one that is ahead.  It moves whenever an import is added
    or dropped, which is the point: the two sentences move with it.
    """

    import importlib.util

    tool = REPO / "tools" / "measure_boundary.py"
    spec = importlib.util.spec_from_file_location("_measure_boundary", tool)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    measured = sum(len(symbols) for symbols in module.collect(SRC).values())

    claim = re.compile(r"imports (\d+) symbols across that boundary")
    for path in (REPO / "README.md", PYPROJECT):
        text = path.read_text(encoding="utf-8")
        stated = claim.findall(text)
        assert stated, f"{path.name} no longer states a boundary count"
        assert [int(value) for value in stated] == [measured] * len(stated), (
            f"{path.name} states {stated} symbols across the engine "
            f"boundary; tools/measure_boundary.py counts {measured}"
        )


def test_every_root_document_is_in_the_source_distribution():
    """A test that ships in the sdist must be able to read what it reads.

    THE BREAKAGE THIS PREVENTS, measured 2026-09-07: RELEASE-NOTES-0.1.0.md
    was the one root document MANIFEST.in did not name, and
    tests/test_doors_bundle.py reads it to hold the published Rust-door
    count to the door table.  Run from the repository the suite was green;
    run from an unpacked sdist, which is the environment MANIFEST.in's own
    first paragraph promises, that test died on a FileNotFoundError for a
    file the reader had never been given.  A missing manifest line reads as
    a broken package, which is the most expensive way for it to be found.

    The rule is every root markdown file rather than that one file, because
    the next release surface will be added the same way this one was.
    """

    manifest = (REPO / "MANIFEST.in").read_text(encoding="utf-8")
    included = {
        line.split(None, 1)[1].strip()
        for line in manifest.splitlines()
        if line.strip().startswith("include ")
    }
    roots = sorted(path.name for path in REPO.glob("*.md"))
    assert roots, "no root markdown found; this gate would pass vacuously"
    missing = [name for name in roots if name not in included]
    assert missing == [], (
        "a root document is not in the source distribution, so a reader of "
        "the sdist does not have it and any test that reads it fails "
        f"there: {missing}")
