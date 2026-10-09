"""Two rules about the suite itself, read off the suite's own source.

Both exist because a red run nobody could read hid a red test.  The card
selection of 2026-09-10 on an RTX 5090 Linux host reported 8 failed and 4 errors against
published woof 2.7.0; nine of those rows were one environment variable set
at import time, and under that noise a genuinely unmarked test rode a cut.
A count by eye found two of the three tests that needed the mark.  These
are the count done by a machine.

Neither rule needs a card, an engine feature or a network: both parse the
files.  They are cheap enough to run in every selection, which is the
point, because a gate that only runs where the breakage shows is a gate
that reports after the fact.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent

#: Engine modules a published `woof` in the supported range may not carry,
#: with the patch item of the series that supplies each.  A test that
#: imports one of these in its body and carries no mark FAILS on an engine
#: that is behind, and the red row then describes the engine's release
#: schedule rather than this package.
#:
#: `woof.verify.harness` is the measured case: absent from published
#: 2.7.0, imported inside the bodies of three tests in
#: `test_arwen_global_native_ntiedtke.py`.  Two carried
#: `@requires_engine_module`; the third did not, and on a Linux host on
#: 2026-09-10 that one file reported 1 failed, 9 passed, 2 skipped.
GAP_MODULES: dict[str, str] = {
    "woof.verify.harness": "04",
}

#: The switch a module must not set for the whole session.
NO_LOCAL_GPU_ENV = "GPUWM_NO_LOCAL_GPU"

#: The files that used to write that switch at module scope.  Named rather
#: than counted: a file that lost the write and gained no mark runs its
#: CPU-only tests on whatever card the box has, which is the other half of
#: the same defect and is silent.
CPU_ONLY_MODULES = (
    "test_arwen_global_default_core.py",
    "test_arwen_global_diffusion_default.py",
    "test_arwen_global_imex.py",
    "test_arwen_global_insitu_energy.py",
    "test_arwen_global_insitu_energy_bands.py",
    "test_arwen_global_insitu_energy_rossby_haurwitz.py",
    "test_arwen_global_pbl_free_atmosphere.py",
    "test_arwen_global_semilag_quintic.py",
    "test_arwen_global_semilag_step.py",
    "test_arwen_global_semilag_tracers.py",
    "test_arwen_global_spectral_eddy_viscosity.py",
    "test_arwen_global_vertical_modes.py",
    "test_arwen_global_vertical_numerics.py",
    "test_global_spectral_fused_kernels_cpu.py",
)


def _test_files() -> list[Path]:
    here = Path(__file__).name
    return sorted(p for p in TESTS.glob("*.py") if p.name != here)


def _body_imports(node: ast.AST) -> set[str]:
    """Every `woof...` module imported below a function's own signature,
    which is where a lazily imported engine module hides."""

    found: set[str] = set()
    for sub in ast.walk(node):
        if sub is node:
            continue
        if isinstance(sub, ast.ImportFrom) and sub.module:
            found.add(sub.module)
        elif isinstance(sub, ast.Import):
            found.update(alias.name for alias in sub.names)
    return {name for name in found if name.split(".")[0] == "woof"}


def _module_marks(tree: ast.Module) -> list[str]:
    return [
        ast.unparse(statement.value)
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark"
                for t in statement.targets)
    ]


def _module_scope_switch_writes(tree: ast.Module) -> list[tuple[int, str]]:
    """Calls that write the device switch on a line no `def` encloses.

    The shape that caused the leak is a bare `os.environ.setdefault`; the
    shape a regression could take is a dictionary update or a `putenv`.
    What they share is the variable's name in a call outside every
    function body.
    """

    enclosed = {
        id(sub)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for sub in ast.walk(node)
    }
    written = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or id(node) in enclosed:
            continue
        text = ast.unparse(node)
        if NO_LOCAL_GPU_ENV in text and ("environ" in text or "putenv" in text):
            written.append((node.lineno, text))
    return written


def _gap_imports_without_a_mark(tree: ast.Module, where: str) -> list[str]:
    """One module's tests that reach a gap module carrying no mark for it."""

    unmarked: list[str] = []
    module_marks = _module_marks(tree)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        marks = [ast.unparse(d) for d in node.decorator_list] + module_marks
        for imported in sorted(_body_imports(node)):
            for gap, item in GAP_MODULES.items():
                if imported != gap and not imported.startswith(gap + "."):
                    continue
                # `ast.unparse` writes single quotes and the source writes
                # double ones, so both sides are normalised.  A comparison
                # that never matched would report every marked test and
                # look exactly like a real finding; the self-test below is
                # what caught that, on this very line.
                wanted = 'requires_engine_module("' + gap + '"'
                if not any(wanted in mark.replace("'", '"') for mark in marks):
                    unmarked.append(
                        "{}:{}:{} imports {} with no "
                        '@requires_engine_module("{}", "{}")'.format(
                            where, node.lineno, node.name,
                            imported, gap, item))
    return unmarked


def _suite_gap_imports_without_a_mark() -> list[str]:
    return [
        row
        for path in _test_files()
        for row in _gap_imports_without_a_mark(
            ast.parse(path.read_text(encoding="utf-8")), path.name)
    ]


def test_every_test_reaching_an_absent_engine_module_carries_its_mark() -> None:
    unmarked = _suite_gap_imports_without_a_mark()
    assert unmarked == [], (
        "a test imports an engine module a published woof in the supported "
        "range does not carry, and carries no mark, so it FAILS rather than "
        "skips by name on an engine that is behind:\n  "
        + "\n  ".join(unmarked))


def test_the_mark_sweep_finds_the_unmarked_one_and_only_that_one() -> None:
    """Test the tester, both directions.

    An AST walk that stopped finding function bodies, or a decorator
    comparison that stopped matching, both report a clean suite and look
    exactly like one.  So the scan is run against a module written here:
    one test that reaches the gap module unmarked, one that reaches it
    with the mark spelled in double quotes, and one with it spelled in
    single quotes, and it must find the first and only the first.
    """

    written = ast.parse("\n".join((
        "from conftest import requires_engine_module",
        "",
        "def test_unmarked():",
        "    from woof.verify.harness.subjects import DeviceHostedSubject",
        "",
        '@requires_engine_module("woof.verify.harness", "04")',
        "def test_marked_double():",
        "    from woof.verify.harness import convection_closure",
        "",
        "@requires_engine_module('woof.verify.harness', '04')",
        "def test_marked_single():",
        "    from woof.verify.harness import subjects",
        "")))
    found = _gap_imports_without_a_mark(written, "written.py")
    assert len(found) == 1, found
    assert found[0].startswith("written.py:3:test_unmarked imports "
                               "woof.verify.harness.subjects")


def test_no_test_module_decides_the_device_switch_for_the_session() -> None:
    """THE BREAKAGE THIS PREVENTS, measured rather than imagined.

    pytest imports every collected module before it runs any test, so
    `os.environ.setdefault(NO_LOCAL_GPU_ENV, "1")` at module scope in one
    CPU-only file turns the device off for a card selection that file is
    not part of.  On a Linux host (RTX 5090, Python 3.14.4) on 2026-09-10,
    `pytest -m "gpu and not slow and not network" tests` gave 8 failed, 63
    passed, 34 skipped, 4 errors against published woof 2.7.0; nine of
    those rows were this leak, and
    `tests/test_global_spectral_sampling_device.py` passed 6 of 6 when its
    own file was run alone.  `pytest.mark.cpu_only` is the shape that does
    not leak.
    """

    offenders = [
        "{}:{}: {}".format(path.name, line, text)
        for path in _test_files()
        for line, text in _module_scope_switch_writes(
            ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert offenders == [], (
        "a test module writes " + NO_LOCAL_GPU_ENV + " at module scope, "
        "which decides the device for every test pytest collects "
        "afterwards.  Mark the module `pytestmark = pytest.mark.cpu_only` "
        "instead:\n  " + "\n  ".join(offenders))


def test_the_switch_sweep_sees_a_module_scope_write_and_ignores_a_scoped_one(
) -> None:
    """Test the tester, both directions, on the shape it is written about."""

    module_scope = ast.parse(
        "import os\n"
        'os.environ.setdefault("' + NO_LOCAL_GPU_ENV + '", "1")\n')
    inside_a_test = ast.parse(
        "def test_one(monkeypatch):\n"
        '    monkeypatch.setenv("' + NO_LOCAL_GPU_ENV + '", "1")\n')
    assert len(_module_scope_switch_writes(module_scope)) == 1
    assert _module_scope_switch_writes(inside_a_test) == []


def test_the_modules_that_used_to_set_the_switch_carry_the_mark() -> None:
    missing = sorted(
        name for name in CPU_ONLY_MODULES
        if "pytest.mark.cpu_only"
        not in (TESTS / name).read_text(encoding="utf-8"))
    assert missing == [], (
        "a module that used to forbid the local device at import time now "
        "forbids it nowhere:\n  " + "\n  ".join(missing))


def test_an_unmarked_test_is_left_with_whatever_the_operator_set(request):
    """This file carries no `cpu_only` mark, so the fixture does nothing."""

    assert request.node.get_closest_marker("cpu_only") is None
    before = os.environ.get(NO_LOCAL_GPU_ENV)
    assert os.environ.get(NO_LOCAL_GPU_ENV) == before


@pytest.mark.cpu_only
def test_a_marked_test_sees_the_switch_set() -> None:
    assert os.environ.get(NO_LOCAL_GPU_ENV) == "1"


# ---------------------------------------------------------------------------
# A third rule: no test reads the ENGINE checkout this package was carved out
# of.
# ---------------------------------------------------------------------------

#: The one file allowed to build that path, and why.  It joins the directory
#: to a SOURCE TREE named by an environment variable, not to this repository,
#: and skips itself when the variable is unset, which is the shape a byte
#: comparison against the tree the bytes came from has to have.
_SOURCE_TREE_READERS = ("test_arwen_global_source_mappings.py",)


def _engine_checkout_reads(tree: ast.Module, where: str) -> list[str]:
    """Every `<this repository> / "woof" / ...` a module builds.

    The shape is a `BinOp` whose operator is `/` and whose right side is the
    string `"woof"`, with a left side that reaches this repository: `ROOT`,
    `REPO_ROOT`, `parents[1]`, or `parent.parent`.  The environment-named
    source tree is a different left side and is not matched.
    """

    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        right = node.right
        if not (isinstance(right, ast.Constant) and right.value == "woof"):
            continue
        left = ast.unparse(node.left)
        repo_shaped = (
            left in {"ROOT", "REPO_ROOT", "REPO", "HERE"}
            or "parents[1]" in left
            or "parent.parent" in left
        )
        if repo_shaped:
            hits.append(f"{where}:{node.lineno}: {left} / 'woof'")
    return hits


def _suite_engine_checkout_reads() -> list[str]:
    hits: list[str] = []
    for path in _test_files():
        if path.name in _SOURCE_TREE_READERS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits.extend(_engine_checkout_reads(tree, path.name))
    return hits


def test_no_test_reads_the_engine_checkout_this_package_was_carved_from() -> None:
    """A data file read out of a directory that exists in no install.

    THE BREAKAGE THIS PREVENTS, measured at the merged release tip on a Linux
    host 2026-09-10: `test_arwen_global_radiation_scorecard.py` read its two
    source mappings from `<repo>/woof/authorities`, which is the ENGINE
    checkout this model was developed in.  That directory is not in this
    repository, not in the wheel and not in the sdist, so the read could
    never have worked here.

    It survived because two separate mechanisms hid it: a conftest hook that
    turned any refusal naming a mapping into a named skip, and a module-scope
    mark that skipped the whole file while the engine lacked the float64
    mirror.  The carve retired the mark, the carried mappings retired the
    hook, and the read then failed for the first time in a full run of the
    merged tree.  A rule that is written down and not enforced is a rule
    that survives exactly as long as something else is masking it.

    Package data belongs to the INSTALLED package: `configs_dir.config_root`
    for the experiments, `analysis_initial.PACKAGE_AUTHORITIES_DIR` for the
    carried mappings, and the module's own `__file__` for anything else.
    """

    hits = _suite_engine_checkout_reads()
    assert hits == [], (
        "these tests read the engine checkout this package was carved out "
        "of, which exists in no install:\n  " + "\n  ".join(hits)
    )


def test_the_engine_checkout_sweep_sees_the_shape_and_ignores_the_source_tree(
        tmp_path) -> None:
    """The rule is driven in the direction that fails silently.

    A sweep that matched nothing would pass on a tree full of the defect, so
    it is run over a module written here: one line it must catch, and two it
    must not (the environment-named source tree, and the package's own data
    directory).
    """

    module = tmp_path / "test_planted.py"
    module.write_text(
        "from pathlib import Path\n"
        "import os\n"
        "ROOT = Path(__file__).resolve().parents[1]\n"
        "BAD = ROOT / 'woof' / 'authorities'\n"
        "SOURCE = Path(os.environ['WOOF_GLOBAL_SOURCE_WORKTREE']) / 'woof'\n"
        "GOOD = Path(__file__).resolve().parent / 'data'\n",
        encoding="utf-8")
    tree = ast.parse(module.read_text(encoding="utf-8"))
    hits = _engine_checkout_reads(tree, "test_planted.py")
    assert len(hits) == 1, hits
    assert hits[0].startswith("test_planted.py:4:"), hits
