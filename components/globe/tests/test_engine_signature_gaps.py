"""A name that resolves can still be a different callable.

`tests/test_engine_compat.py` holds the seam for symbols the engine does not
carry.  This file holds the other half of the same boundary: engine callables
that DO resolve and do not accept an argument this package hands them.

Why it needed its own instrument.  The carve's boundary table asked, symbol by
symbol, whether the engine had the name.  Eighty-two of eighty-six resolved and
the table called the rest of the boundary clean.  Then
`tools/measure_engine_signatures.py` compared the parsed signatures of the two
engine trees and found eighteen callables that differ, eight of which do not
take an argument this package passes.

ALL EIGHT ROWS WERE RETIRED ON 2026-09-09 AND 2026-09-10, and that is the
healthy outcome for a gate.  Seven were physics: the surface layer's `vegfra`,
radiation's `column_size_bounding`, both cumulus constructors' `column_chunk`,
the YSU launcher's mixing-length flag and the two float64 mirrors' keywords.
The carve took those callables into `woof.globe.core`, so their signatures
are this package's own and no installed engine can present a different one.
The eighth was `decode_mapped_source(scratch_destination=)`: the published
engine places the same scratch by `WOOF_COMPOSE_SCRATCH`, so
`woof.globe.mapped_source_compat` translates between the two spellings and
says in the receipt which one placed the directory.  A gate is retired when
the breakage it names cannot happen, not weakened into a warning.

SO THE TABLE IS EMPTY, and this file proves the retirement from both sides:
that the carried callables take the keywords their rows used to refuse, that
the scratch placement is translated rather than dropped, and that the
mechanism still refuses by name for a row the next measurement adds.  The
mechanism is kept because the question does not go away: an engine inside
`woof>=2.8.0,<2.9` can move a signature underneath the seam.
"""
from __future__ import annotations

import inspect
import sys
import types

import pytest

from woof.globe import engine_compat
from woof.globe.engine_compat import (
    SIGNATURE_GAPS,
    MissingEngineSymbol,
    SignatureGap,
    engine_signature_gaps,
    require_engine_signature,
)


def test_the_table_is_empty_because_every_measured_row_was_retired():
    """Emptiness is the measurement, so it is asserted rather than assumed."""

    assert SIGNATURE_GAPS == ()
    assert engine_signature_gaps() == ()


def test_an_unmeasured_call_is_a_key_error_not_a_silent_pass():
    """Asking about a call nobody measured must not answer "fine"."""

    with pytest.raises(KeyError):
        require_engine_signature("woof.core.noah", "load_tables")
    with pytest.raises(KeyError):
        require_engine_signature("woof.mapped_source", "decode_mapped_source")


def test_the_seven_physics_rows_are_gone_and_the_callables_are_carried():
    """The retirement, measured on the objects rather than asserted.

    Each takes the keyword its row used to say a published engine refused,
    because the module that defines it is now this package's.
    """

    from woof.globe.core import gf, ntiedtke, rrtmgp

    def takes(target, keyword):
        return keyword in inspect.signature(target).parameters

    assert takes(gf.GrellFreitas, "column_chunk")
    assert takes(gf.GrellFreitas, "updraft_only_when_downdraft_dry")
    assert takes(gf.GrellFreitas, "resolved_convergence_closure")
    assert takes(ntiedtke.NewTiedtke, "column_chunk")
    assert takes(rrtmgp.RRTMGPRadiation, "column_size_bounding")
    for module in (gf, ntiedtke, rrtmgp):
        assert module.__name__.startswith("woof.globe.core.")
    # sfclay, ysu and the float64 mirror reach cupy at module scope, so their
    # signatures are read where a card is, in
    # tests/test_arwen_global_carried_core.py.  The keywords are asserted on
    # the SOURCE here rather than not at all, because the point of this test
    # is that the rows were retired for a reason.
    from pathlib import Path

    import woof.globe.core as core

    carried = Path(core.__file__).parent
    assert "vegfra" in (carried / "sfclay.py").read_text(encoding="utf-8")
    assert "free_atmosphere_mixing_length" in (
        carried / "ysu.py").read_text(encoding="utf-8")


def test_the_scratch_row_is_gone_because_the_placement_is_translated():
    """The eighth retirement: one placement, two spellings, one receipt."""

    from woof.globe import mapped_source_compat

    assert mapped_source_compat.COMPOSE_SCRATCH_ENV == "WOOF_COMPOSE_SCRATCH"
    assert hasattr(mapped_source_compat, "engine_scratch")
    # What the translation records is proven in
    # tests/test_mapped_source_compat.py, which drives the door itself from
    # two threads and reads each receipt.


# --------------------------------------- the mechanism, on a row it is given

def _install(monkeypatch, module_name: str, name: str, target) -> None:
    """Put one callable in sys.modules under a name the gate reads."""

    module = types.ModuleType(module_name)
    setattr(module, name, target)
    monkeypatch.setitem(sys.modules, module_name, module)


_ROW = SignatureGap(
    module="gpuwm.example_seam",
    name="example_call",
    keywords=("staging_destination",),
    stops="nothing: this row exists only where a test puts it, so the gate "
          "that refuses the next real one is exercised rather than trusted")


def _with_row(monkeypatch) -> None:
    monkeypatch.setattr(engine_compat, "SIGNATURE_GAPS", (_ROW,))


def test_a_signature_that_takes_the_keyword_is_not_a_gap(monkeypatch):
    _with_row(monkeypatch)

    def example_call(mapping, files, *, staging_destination=None):
        return []

    _install(monkeypatch, "gpuwm.example_seam", "example_call", example_call)
    require_engine_signature("gpuwm.example_seam", "example_call")
    assert engine_compat.engine_signature_gaps() == ()


def test_a_signature_without_the_keyword_is_refused_by_name(monkeypatch):
    _with_row(monkeypatch)

    def example_call(mapping, files, *, destination=None):
        return []

    _install(monkeypatch, "gpuwm.example_seam", "example_call", example_call)
    with pytest.raises(MissingEngineSymbol) as raised:
        require_engine_signature("gpuwm.example_seam", "example_call")
    text = str(raised.value)
    assert "staging_destination" in text
    assert "gpuwm.example_seam.example_call" in text
    # And it says why the argument is not simply dropped.
    assert "the symbol resolves" in text


def test_a_module_that_cannot_be_imported_is_unanswerable_not_clean(monkeypatch):
    """A host that cannot import the module must not publish a verdict."""

    _with_row(monkeypatch)
    monkeypatch.setitem(sys.modules, "gpuwm.example_seam", None)
    assert _ROW.missing() is None
    require_engine_signature("gpuwm.example_seam", "example_call")


def test_a_star_kwargs_signature_is_unanswerable(monkeypatch):
    """`**kwargs` accepts anything at the call and refuses inside it."""

    _with_row(monkeypatch)

    def example_call(mapping, files, **kwargs):
        return []

    _install(monkeypatch, "gpuwm.example_seam", "example_call", example_call)
    assert _ROW.missing() is None


def test_building_native_physics_no_longer_refuses_on_a_published_engine():
    """The gate that stood here, and why nothing stands in its place.

    THE BREAKAGE IT NAMED: a native run against a published engine was
    accepted, priced the card, allocated it, integrated its first steps and
    died inside the surface layer on `vegfra`.  The gate moved that to the
    point where the forecast is built.

    The carve moved the surface layer itself.  `build_physics` now constructs
    the bridge and returns it, and this test is what says the refusal is gone
    rather than merely untested: a native config builds against the published
    engine this suite runs on.
    """

    from woof.globe.config import load_config
    from woof.globe.configs_dir import config_root
    from woof.globe.runner import build_physics

    cfg = load_config(str(config_root()
                          / "arwen_global_level5_native_smoke.toml"))
    assert cfg.physics_mode == "arwen-native"
    bridge = build_physics(cfg, "numpy")
    assert bridge is not None


def test_the_engine_this_suite_runs_against_reports_its_own_gaps():
    """Whatever the installed engine is, the report is about THAT engine."""

    for gap, missing in engine_signature_gaps():
        assert missing, (gap.module, gap.name)
        assert set(missing) <= set(gap.keywords)
        loaded = sys.modules.get(gap.module)
        if loaded is not None and getattr(loaded, gap.name, None) is not None:
            parameters = inspect.signature(getattr(loaded, gap.name)).parameters
            assert all(name not in parameters for name in missing)


# ------------------------------------------------- and the doctor says so

def _boundary_rows():
    from woof.globe.doctor import build_report

    report = build_report()
    for title, rows in report.sections:
        if title == "engine boundary":
            return rows
    raise AssertionError("the doctor has no engine boundary section")


def test_the_doctor_carries_the_boundary_and_grades_it_the_way_the_seam_does():
    """Every standing gap is printed; a gap only where a documented command stops.

    THE BREAKAGE THIS PREVENTS.  The doctor's last line used to read "No gaps.
    Every documented command can run on this machine" on a box where `sizing`
    could not price a card and a scorecard could not regrid its reference,
    because the report looked at Rust doors and versions and never at the
    symbols the package imports out of the engine.  So every standing row is
    printed, with what it stops.

    THE VERDICT IS THE EXIT CODE'S QUESTION.  A carried contract is a note.
    A refused computation is a gap when a documented command reaches it and
    an optional note when none does: until 0.1.3 every refused row was a gap,
    and a correct 0.1.2 install against the published 2.8.0 exited 1 on two
    rows that stop no documented command.
    """

    from woof.globe.engine_compat import engine_gaps

    rows = {row.label: row for row in _boundary_rows()}
    for gap in engine_gaps():
        label = f"{gap.module.split('.')[-1]}.{gap.symbol}"
        assert label in rows, (label, sorted(rows))
        refused_and_stopping = (gap.handling == "refused"
                                and gap.stops_a_documented_command)
        expected = "gap" if refused_and_stopping else "note"
        assert rows[label].verdict == expected, (label, rows[label].verdict)
        assert rows[label].detail, label
        assert any(gap.stops in line for line in rows[label].detail), label


def _symbol_gap(symbol: str, *, documented: bool) -> engine_compat.EngineGap:
    return engine_compat.EngineGap(
        module="woof.core.preflight",
        symbol=symbol,
        stops="nothing: this row exists only where a test puts it",
        handling="refused",
        stops_a_native_forecast=False,
        stops_a_documented_command=documented)


def test_a_refused_symbol_is_a_gap_exactly_when_a_documented_command_stops(
        monkeypatch):
    """Both directions of the verdict, on rows the engine cannot carry.

    A verdict that can only answer one way is not a measurement: the row a
    documented command reaches must still move the exit code, and the row
    none reaches must not.
    """

    from woof.globe.doctor import build_report

    stopping = _symbol_gap("a_symbol_no_engine_carries_" + "stopping",
                           documented=True)
    optional = _symbol_gap("a_symbol_no_engine_carries_" + "optional",
                           documented=False)
    monkeypatch.setattr(engine_compat, "GAPS", (stopping, optional))
    report = build_report()
    rows = {row.label: row for row in _boundary_rows()}
    stop_row = rows[f"preflight.{stopping.symbol}"]
    optional_row = rows[f"preflight.{optional.symbol}"]
    assert stop_row.verdict == "gap"
    assert stop_row.finding == "absent"
    assert optional_row.verdict == "note"
    assert optional_row.finding == "absent (optional)"
    assert any("no documented command reaches it" in line
               for line in optional_row.detail)
    gap_labels = {row.label for row in report.gaps}
    assert stop_row.label in gap_labels
    assert optional_row.label not in gap_labels


def test_the_absent_symbols_of_a_published_engine_leave_the_boundary_clean(
        monkeypatch):
    """The A124 regression: a correct install is not failed by optional rows.

    Measured 2026-09-29: `woof global doctor` on a correct 0.1.2 install
    against the published 2.8.0 printed both rows of `GAPS` as gaps and
    exited 1.  Every row is made to stand here whatever engine this suite
    runs on, so the assertion does not depend on which one is installed.
    """

    monkeypatch.setattr(engine_compat.EngineGap, "present",
                        lambda self: False)
    assert set(engine_compat.engine_gaps()) == set(engine_compat.GAPS)
    rows = _boundary_rows()
    for gap in engine_compat.GAPS:
        label = f"{gap.module.split('.')[-1]}.{gap.symbol}"
        matching = [row for row in rows if row.label == label]
        assert len(matching) == 1, (label, [row.label for row in rows])
        if not gap.stops_a_documented_command:
            assert matching[0].verdict == "note", label
            assert matching[0].finding == "absent (optional)", label
    documented = [gap for gap in engine_compat.GAPS
                  if gap.stops_a_documented_command]
    assert not documented, (
        "a row now stops a documented command, so the published engine's "
        "doctor exits 1 on it; carry the symbol in the engine")
    assert not [row for row in rows if row.verdict == "gap"]


# ------------------------- and what a row says it stops is measured, not said

#: The modules whose functions are the console script: `woof global`
#: (`cli.main`), the engine's `woof global` (`cli.register_cli`), the
#: `python -m arwen_global` door and the detached terminal job's worker.
_ENTRY_MODULES = frozenset({
    "woof.globe.cli", "woof.globe.__main__", "woof.globe.tui_worker",
})


def _package_nodes():
    """Each top-level definition of the package and the names it references.

    Returns ``{dotted node: set of dotted references}``.  A module's
    statements outside any definition are one node, ``<module>``, because
    they run at import.  Imports are resolved through every binding a module
    makes, at any depth, which over-approximates reachability rather than
    missing an edge; an attribute on an object that is not an imported
    module is not followed, and a caller that constructs the object names
    its class, which is followed.
    """

    import ast
    from pathlib import Path

    import woof.globe

    root = Path(woof.globe.__file__).resolve().parent
    nodes: dict[str, set[str]] = {}
    for path in sorted(root.rglob("*.py")):
        # woof.globe sits one folder below the top-level package
        parts = list(path.relative_to(root.parents[1]).with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts = parts[:-1]
        module = ".".join(parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package = module if is_package else module.rpartition(".")[0]
        bindings: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        bindings.setdefault(alias.asname, set()).add(alias.name)
                    else:
                        head = alias.name.split(".")[0]
                        bindings.setdefault(head, set()).add(head)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    anchor = package
                    for _ in range(node.level - 1):
                        anchor = anchor.rpartition(".")[0]
                    base = f"{anchor}.{base}" if base else anchor
                for alias in node.names:
                    bindings.setdefault(alias.asname or alias.name, set()).add(
                        f"{base}.{alias.name}")
        defined = {statement.name for statement in tree.body
                   if isinstance(statement, (ast.FunctionDef,
                                             ast.AsyncFunctionDef,
                                             ast.ClassDef))}

        def resolve(name: str) -> set[str]:
            found = set(bindings.get(name, ()))
            if name in defined:
                found.add(f"{module}.{name}")
            return found

        def references(statement) -> set[str]:
            out: set[str] = set()
            for node in ast.walk(statement):
                if isinstance(node, ast.Attribute):
                    chain = []
                    head = node
                    while isinstance(head, ast.Attribute):
                        chain.append(head.attr)
                        head = head.value
                    if isinstance(head, ast.Name):
                        tail = ".".join(reversed(chain))
                        out.update(f"{target}.{tail}"
                                   for target in resolve(head.id))
                elif isinstance(node, ast.Name):
                    out.update(resolve(node.id))
            return out

        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
                key = f"{module}.{statement.name}"
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                continue
            elif _is_main_guard(statement):
                key = f"{module}.<main>"
            else:
                key = f"{module}.<module>"
            nodes.setdefault(key, set()).update(references(statement))
    return nodes


def _is_main_guard(statement) -> bool:
    """``if __name__ == "__main__":``, which runs only as ``python -m``."""

    import ast

    test = getattr(statement, "test", None)
    return (isinstance(statement, ast.If)
            and isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__")


def _documented_module_entries() -> set[str]:
    """Every ``python -m arwen_global...`` a shipped page tells a reader to run."""

    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pages = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    found: set[str] = set()
    for page in pages:
        if page.is_file():
            found.update(re.findall(r"python -m (woof\.globe[\w.]*)",
                                    page.read_text(encoding="utf-8")))
    return found


def _callers_of(seed: str, nodes) -> set[str]:
    """Every node that reaches ``seed`` through the references above."""

    reached = {seed}
    changed = True
    while changed:
        changed = False
        for key, refs in nodes.items():
            if key in reached:
                continue
            if any(ref == hit or ref.startswith(hit + ".")
                   for ref in refs for hit in reached):
                reached.add(key)
                changed = True
    return reached


def _reaches_an_entry(closure, documented_modules) -> list[str]:
    """The documented entries in ``closure``.

    The console script's modules always count.  Code a module runs at
    import counts, because every importer runs it.  A ``python -m`` block
    counts when a shipped page documents that module: an undocumented
    research entry such as `python -m woof.globe.surface_energy` is not a
    command the doctor answers for, and its row names it instead.
    """

    def module_of(key: str) -> str:
        module = key.rsplit(".", 1)[0]
        return module.removesuffix(".__main__")

    return sorted(
        key for key in closure
        if key.rsplit(".", 1)[0] in _ENTRY_MODULES
        or key.endswith(".<module>")
        or (key.endswith(".<main>") and module_of(key) in documented_modules))


def test_the_call_graph_walk_finds_a_path_the_console_script_takes():
    """The instrument, on a path known to be reached, must answer yes.

    `run` and `go` price the card through `sizing.run_memory_gate`, which
    the console script imports inside its handlers.
    """

    nodes = _package_nodes()
    documented = _documented_module_entries()
    closure = _callers_of("woof.globe.sizing.run_memory_gate", nodes)
    assert any(key.startswith("woof.globe.cli.") for key in closure), (
        sorted(closure))
    assert _reaches_an_entry(closure, documented)
    # And a documented `python -m` scorecard: docs/ARWEN_GLOBAL_LEVEL5.md
    # tells a reader to run `python -m woof.globe.radiation_scorecard`,
    # whose regrid is its own function.
    assert "woof.globe.radiation_scorecard" in documented
    closure = _callers_of("woof.globe.radiation_scorecard.regrid_reference",
                          nodes)
    assert "woof.globe.radiation_scorecard.<main>" in _reaches_an_entry(
        closure, documented), sorted(closure)


def test_a_row_that_stops_no_documented_command_is_reached_by_none():
    """`stops_a_documented_command=False` is what the doctor grades on.

    THE BREAKAGE THIS PREVENTS.  The doctor reports a refused row that no
    documented command reaches as optional, and exits 0 over it.  Wiring a
    subcommand to the check it stops, without flipping the row, would ship
    a doctor that says "every documented command can run" above a command
    that refuses.  Each row must also still have a caller in this package,
    because a row nothing calls is a guard with no defect.
    """

    nodes = _package_nodes()
    documented = _documented_module_entries()
    for gap in engine_compat.GAPS:
        seed = f"woof.globe.engine_compat.{gap.symbol}"
        assert seed in nodes, f"{seed} is no longer defined"
        closure = _callers_of(seed, nodes)
        callers = sorted(closure - {seed})
        assert callers, (
            f"nothing in this package calls {seed}; retire the row")
        if gap.stops_a_documented_command:
            continue
        entries = _reaches_an_entry(closure, documented)
        assert not entries, (
            f"{gap.module}.{gap.symbol} says it stops no documented command, "
            f"and {entries} reach it through {callers}; set "
            "stops_a_documented_command=True so the doctor reports it as a gap")


def test_a_signature_gap_reaches_the_doctor_as_a_gap(monkeypatch):
    _with_row(monkeypatch)

    def example_call(mapping, files, *, destination=None):
        return []

    _install(monkeypatch, "gpuwm.example_seam", "example_call", example_call)
    rows = {row.label: row for row in _boundary_rows()}
    assert "example_seam.example_call" in rows
    assert rows["example_seam.example_call"].verdict == "gap"
    assert "staging_destination" in rows["example_seam.example_call"].finding


def test_an_engine_whose_signatures_are_whole_leaves_no_signature_row(monkeypatch):
    """The section retires itself, row by row, as the engine catches up."""

    _with_row(monkeypatch)

    def example_call(mapping, files, *, staging_destination=None):
        return []

    _install(monkeypatch, "gpuwm.example_seam", "example_call", example_call)
    assert engine_compat.engine_signature_gaps() == ()
    labels = {row.label for row in _boundary_rows()}
    assert "example_seam.example_call" not in labels
