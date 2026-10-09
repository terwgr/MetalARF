"""The console script's surface: every command reachable, exit codes fixed.

`woof global` inside the engine exposed eleven leaf commands and the module
door exposed twenty-one, ten of which the product door never got.  This
distribution's script carries all of them plus the eight it adds, because
ship-only-what-a-user-can-reach applies to the ten as much as to the eleven.
The eight are `doctor`, `fetch-doors`, `obs`, `go`, `render`, `configs`, and
the machine seam `run-plan` with its human view `sources`.
"""
from __future__ import annotations

import argparse

import pytest

pytest.importorskip("arwen_global", reason="the package under test")

from woof.globe import cli  # noqa: E402

#: Every command the door promises.  A name removed from this list is a name
#: removed from written instructions somewhere, so the list is the contract.
EXPECTED = {
    "abi-fast-model", "abi-reference", "abi-score", "assimilate",
    "check-migration", "check-native-candidate", "check-native-evidence",
    "check-receipt", "configs", "cycle", "da", "doctor", "export",
    "export-parent", "export-wrfout", "fetch-analysis", "fetch-doors", "go",
    "inspect",
    "inspect-export", "inspect-parent-series", "inspect-regional-frame",
    "inspect-regional-target", "make-parent-series", "make-regional-target",
    "microwave", "migrate-level4-checkpoint", "native-qualify", "obs",
    "physics-manifest", "pins", "render", "run", "run-plan", "sources",
    "statics", "transform-check", "translate-regional-frame",
}


def _subcommands() -> set[str]:
    parser = cli.build_parser()
    actions = [row for row in parser._actions
               if isinstance(row, argparse._SubParsersAction)]
    assert len(actions) == 1
    return set(actions[0].choices)


def test_every_documented_command_is_reachable():
    missing = EXPECTED - _subcommands()
    assert not missing, f"documented and unreachable: {sorted(missing)}"


def test_no_command_exists_that_nothing_documents():
    extra = _subcommands() - EXPECTED
    assert not extra, (
        f"reachable and undocumented: {sorted(extra)}; add it to the door "
        "table and its page, or take it off the parser")


def test_export_keeps_its_old_spelling_as_an_alias():
    """A name that vanishes turns a working instruction into an error."""

    choices = _subcommands()
    assert "export" in choices and "export-wrfout" in choices
    assert choices == _subcommands()


def test_the_prog_name_is_the_console_script():
    parser = cli.build_parser()
    assert parser.prog == "woof global"


@pytest.mark.parametrize("command", sorted(EXPECTED))
def test_every_command_has_help(command):
    """`--help` must not need a GPU, an engine door, or a network."""

    parser = cli.build_parser()
    with pytest.raises(SystemExit) as caught:
        parser.parse_args([command, "--help"])
    assert caught.value.code == 0


def test_exit_codes_are_the_contract():
    assert (cli.EXIT_OK, cli.EXIT_REFUSED, cli.EXIT_ARGUMENTS,
            cli.EXIT_DOOR, cli.EXIT_DEVICE) == (0, 1, 2, 3, 4)


def test_a_missing_rust_door_earns_its_own_exit_code():
    """3, not 1: a caller that sees 3 knows to offer `fetch-doors`."""

    engine_shape = RuntimeError(
        "the radiosonde front door (rw_igra2) is not built or not found.\n"
        "stage it with woof fetch-bridges")
    ours = RuntimeError("rw_wrfbatch is not staged, so there is nothing that "
                        "can draw a weather field here.")
    assert cli.exit_code_for(engine_shape) == cli.EXIT_DOOR
    assert cli.exit_code_for(ours) == cli.EXIT_DOOR


def test_a_plain_refusal_is_exit_one():
    assert cli.exit_code_for(ValueError("that config selects a synthetic planet")) == 1


def test_a_device_refusal_earns_its_own_exit_code():
    """4, not 1: a caller that sees 4 knows to offer a smaller truncation."""

    from woof.globe.sizing import GlobalMemoryRefusal

    refusal = GlobalMemoryRefusal("free 8,000,000,000 B, required 20,000,000,000 B")
    assert cli.exit_code_for(refusal) == cli.EXIT_DEVICE


def test_an_argument_error_is_argparse_own_code():
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["run"])
    assert caught.value.code == cli.EXIT_ARGUMENTS


def test_a_mistyped_experiment_name_is_an_argument_error_not_a_late_crash(tmp_path, monkeypatch):
    """The refusal lands at parse time with the shipped list beside it."""

    monkeypatch.chdir(tmp_path)
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["run", "no_such_experiment", "--outdir", str(tmp_path)])
    assert caught.value.code == cli.EXIT_ARGUMENTS


def test_a_missing_rust_door_earns_exit_three_and_names_the_bundle():
    """3, not 1: a caller that sees 3 knows to stage a bundle.

    The engine's own resolvers raise their own errors in their own words and
    end them with a `cargo build` inside a checkout.  A user of two wheels has
    no checkout, so the refusal this package prints names the door, what it
    stops, the bundle that publishes it and the command that stages it, and
    the build recipe is dropped rather than passed on.
    """

    from woof.globe.doors import DoorMissing, missing_door_refusal

    refusal = missing_door_refusal(
        "static_fields",
        "the Rust static-field library was not found; searched:\n"
        "  /somewhere/libstatic_fields.so\n"
        "  # build it from a checkout:\n"
        "  cd tools/rustwx && cargo build --release -p static-fields",
    )
    assert isinstance(refusal, DoorMissing)
    assert cli.exit_code_for(refusal) == cli.EXIT_DOOR
    text = str(refusal)
    assert "static_fields" in text
    assert "woof fetch-bridges" in text
    assert "statics" in text
    assert "/somewhere/libstatic_fields.so" in text
    assert "cargo" not in text


def test_the_doctor_says_which_table_answers_each_source_it_names():
    """The third kind of thing a command needs, beside a symbol and a binary.

    Every shipped GDAS experiment names a source id.  woof 2.7.0's authority
    table carries none of them, so this package carries those six itself and
    the resolver asks the engine first.  What the report owes a reader is
    therefore no longer present/absent but WHICH COPY ANSWERED: two copies of
    one source on one machine decide between them what a forecast was
    initialized from, and a reader who cannot see which one answered cannot
    tell the two runs apart.
    """

    from woof.globe import doctor as doctor_module

    ids = doctor_module._config_mapping_ids()
    assert "gdas-global" in ids
    assert len(ids["gdas-global"]) > 1
    report = doctor_module.build_report()
    titles = [title for title, _ in report.sections]
    assert "source mappings" in titles
    rows = dict(report.sections)["source mappings"]
    labels = [row.label for row in rows]
    assert "gdas-global" in labels
    for spec, _ in doctor_module._NAMED_MAPPINGS:
        assert spec in labels
        # EVERY LABEL IS AN ID.  This table used to spell two of its four
        # entries as file names, because their readers open the file by name,
        # and the registry field built from it then held an id on four rows
        # and a file name on two.
        assert not spec.endswith(".mapping.json"), spec

    # Both spellings reach the same file, which is what lets the label be the
    # id while `radiation_scorecard.REFERENCE_MAPPING` and
    # `microwave.columns.MAPPING_NAME` keep opening theirs by name.  The
    # resolver is the carried one, so the answer also says which table it
    # came from.
    from woof.globe.analysis_initial import resolve_analysis_mapping_row
    from woof.globe.sources import MAPPING_SUFFIX

    for spec, _ in doctor_module._NAMED_MAPPINGS:
        try:
            bare = resolve_analysis_mapping_row(spec)
        except Exception:
            with pytest.raises(Exception):
                resolve_analysis_mapping_row(spec + MAPPING_SUFFIX)
            continue
        suffixed = resolve_analysis_mapping_row(spec + MAPPING_SUFFIX)
        assert suffixed.path == bare.path
        assert suffixed.origin == bare.origin
    # Every named source resolves here, and every row says where from.
    for row in rows:
        if row.label == "carried copies":
            continue
        assert row.verdict != "gap", row.label
        assert ("carried by this package" in row.finding
                or "from the engine" in row.finding), row.label


def test_both_spellings_of_a_mapping_reach_the_same_file(tmp_path, monkeypatch):
    """POSITIVE evidence, because two matching refusals prove nothing.

    On the published 2.7.0 none of the six global mappings is in the
    authority table, so the loop above only ever compares two
    FileNotFoundErrors.  This builds a table that carries both shapes of name
    -- the file whose name ends at the id, and the file the bare-id glob finds
    -- and asks for each row both ways.  Four questions, two files, and the
    answers have to pair up.
    """

    from woof.globe.analysis_initial import resolve_analysis_mapping_row
    from woof.globe.sources import MAPPING_SUFFIX

    authorities = tmp_path / "authorities"
    authorities.mkdir()
    ends_at_id = authorities / f"rw-wps-cloud-cover{MAPPING_SUFFIX}"
    ends_at_id.write_text("{}", encoding="utf-8")
    globbed = authorities / f"rw-wps-surface-state-grib2{MAPPING_SUFFIX}"
    globbed.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("woof.globe.analysis_initial._engine_authorities_dir",
                        lambda: authorities)

    for spelling in ("rw-wps-cloud-cover", "rw-wps-cloud-cover" + MAPPING_SUFFIX):
        row = resolve_analysis_mapping_row(spelling)
        assert row.path == ends_at_id
        assert row.origin == "engine"
    for spelling in ("surface-state", "surface-state" + MAPPING_SUFFIX):
        row = resolve_analysis_mapping_row(spelling)
        assert row.path == globbed
        assert row.origin == "engine"
    # ...and a name neither question answers refuses, naming both tables.
    with pytest.raises(ValueError) as caught:
        resolve_analysis_mapping_row("no-such-source")
    message = str(caught.value)
    assert "no-such-source" in message
    assert "found 0 (none)" in message
    assert "carried copies" in message


def test_a_door_with_a_measured_fallback_does_not_claim_to_stop_a_command():
    """The gate law from the other side: a report names what actually breaks.

    `rw_fetch` is absent on any machine that has not staged the engine's
    bundle, and the engine resolves its downloader at the call: measured, the
    analysis came down through its own byte-range transport and the command
    went on. A row saying the command stops sends a reader to stage a bundle
    for a breakage that is not happening, and hides the one that is -- the
    fetch leaving the Rust data path.
    """

    from woof.globe.doors import door_by_name

    fetch = door_by_name("rw_fetch")
    assert fetch.fallback and "Python" in fetch.fallback
    for name in ("gpuwm_mapped_engine", "static_fields", "rw_wrfbatch", "rw_atms"):
        assert door_by_name(name).fallback is None


def test_the_tape_writer_is_a_door_and_refuses_at_exit_three(monkeypatch):
    """The wrfout writer is its own artifact with its own pin.

    It is not `rw_netcdf`: `WrfoutWriter` drives the netcdf-writer cdylib
    through a different bridge, and the engine's bundle pins both. While the
    row was missing, the report counted every door present on a machine where
    `render` regridded a whole day and then died at the first tape.
    """

    from woof.globe.doors import DoorMissing, door_by_name
    from woof.globe import wrfout_export

    door = door_by_name("netcdf_writer")
    assert door.library and door.bundle == "woof"
    assert "export" in door.used_by and "render" in door.used_by

    from woof.io import nc_writer_bridge

    monkeypatch.setattr(nc_writer_bridge, "unavailable_reason",
                        lambda: "the Rust NetCDF writer library was not found")
    with pytest.raises(DoorMissing) as caught:
        wrfout_export._require_tape_writer()
    assert cli.exit_code_for(caught.value) == cli.EXIT_DOOR
    assert "woof fetch-bridges" in str(caught.value)

    # The engine's own netCDF4 route is somebody's deliberate choice.
    monkeypatch.setenv("WOOF_WRFOUT_WRITER", "python")
    wrfout_export._require_tape_writer()


def test_no_written_command_names_the_console_script_with_a_dotted_tail():
    """A `woof global.<module>` spelling is a command nobody can type.

    THE BREAKAGE THIS NAMES.  The carve rewrote `python -m
    gpuwm.arwen_global.X` in two steps -- the package prefix first, then
    `python -m arwen_global` into the console script -- and the two rules
    composed into `woof global.X`.  That is a console script with a
    subcommand called `.X`, which does not exist: the shell reports "command
    not found" for the whole thing, and the reader has no way to guess that
    the real spelling is `python -m arwen_global.X`.  21 sites shipped in the
    package's own source and 18 more in its published pages, across thirteen
    modules that every one of them told the reader to run.

    The dot is the whole tell, so the gate is the dot.  `woof global run`,
    `woof global da fresh` and every other real spelling separates the
    script from its subcommand with a space.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    pattern = re.compile(r"woof global\.[a-z_][a-z_0-9]*")
    hits = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".md", ".toml", ".sh", ".txt"}:
            continue
        if any(part in {".git", "build", "dist", "__pycache__", "out", "runs"}
               for part in path.parts):
            continue
        if path.name == Path(__file__).name:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            hits.append(f"{path.relative_to(root)}:{line}: {match.group(0)}")
    assert not hits, (
        "these name the console script with a dotted tail, which is not a "
        "command:\n  " + "\n  ".join(hits))


def _distinct_leaf_parsers(parser: argparse.ArgumentParser, seen: set) -> None:
    """Every leaf command, counted ONCE however many names reach it.

    `export` and `export-wrfout` are two spellings of one command -- the old
    name is kept so a written instruction that predates the rename still runs
    -- and they share a parser object, so identity is what separates a command
    from an alias.  Counting `choices` would count the alias as a command and
    the README's number would be one higher than the surface a reader meets.
    """

    subparsers = [row for row in parser._actions
                  if isinstance(row, argparse._SubParsersAction)]
    if not subparsers:
        seen.add(id(parser))
        return
    for child in subparsers[0].choices.values():
        _distinct_leaf_parsers(child, seen)


def _shipped_markdown() -> list:
    """Every markdown page this cut publishes, in a stable order."""

    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    pages = [root / "README.md", root / "CHANGELOG.md"]
    pages += sorted(root.glob("RELEASE-NOTES-*.md"))
    pages += sorted(root.glob("docs/*.md"))
    return [path for path in pages if path.is_file()]


def _stated_numbers(pattern_text: str) -> list:
    """Every `<number> <thing>` a shipped page states, with where it says it."""

    import re

    pattern = re.compile(pattern_text)
    found = []
    for path in _shipped_markdown():
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            line = text.count(chr(10), 0, match.start()) + 1
            found.append((path.name, line, match.group(0),
                          int(match.group(1))))
    return found


def test_every_shipped_page_states_the_command_count_the_parser_has():
    """A page that counts its own product's doors and gets a different answer.

    THE BREAKAGE THIS PREVENTS, found 2026-09-09 and MEASURED AGAIN
    2026-09-10: "One console script, N commands" is a hand-maintained number
    in a sentence whose next clause points the reader at `--help` and at a
    page generated from the parser.  Two lanes adding a door each move it,
    neither notices, and the one sentence a reader uses to decide whether they
    have found everything is the sentence that is wrong.

    The first form of this gate read README.md alone.  It passed at a tip
    where CHANGELOG.md and RELEASE-NOTES-0.1.0.md both said 45 while the
    parser had 47 and the README had been corrected: the release surface for
    this cut was wrong in the two documents a reader of a release reads first,
    and the gate built for exactly that breakage could not see them.  So the
    referent is every shipped page and the pattern is any count of commands,
    however the sentence around it is built.  A page that means a SUBSET of
    the doors spells its number in words, as these pages already do ("the
    engine's `woof global` reached eleven of them").
    """

    seen: set = set()
    _distinct_leaf_parsers(cli.build_parser(), seen)
    expected = len(seen)
    stated = _stated_numbers(r"(\d+) commands")
    assert stated, "no shipped page states a command count any more"
    wrong = [f"{name}:{line}: {text}" for name, line, text, value in stated
             if value != expected]
    assert not wrong, (
        f"the parser has {expected} distinct leaf commands (aliases counted "
        "once) and these pages say otherwise:" + chr(10) + "  "
        + (chr(10) + "  ").join(wrong))


def test_every_shipped_page_states_the_experiment_count_that_ships():
    """The same failure in the same two documents, on the other number.

    THE BREAKAGE THIS PREVENTS, measured 2026-09-10: CHANGELOG.md and
    RELEASE-NOTES-0.1.0.md both said 54 configured experiments while 55 TOMLs
    shipped -- the same off-by-one a hard-coded 54 in
    `tests/test_package_data_coverage.py` had already been removed for.  A
    reader who counts what they installed and gets a different number from the
    release note has to decide which of the two is lying.
    """

    from woof.globe import configs_dir

    expected = len(configs_dir.list_configs())
    stated = _stated_numbers(r"(\d+) (?:configured )?experiments")
    wrong = [f"{name}:{line}: {text}" for name, line, text, value in stated
             if value != expected]
    assert not wrong, (
        f"{expected} experiments ship in this install and these pages say "
        "otherwise:" + chr(10) + "  " + (chr(10) + "  ").join(wrong))


#: The number words the shipped pages use where they mean a SUBSET of the
#: doors, which is the form the numeric gates above are built to skip.  A
#: sentence that names its members and then counts them in words is still a
#: count, and it was wrong at exactly the tip the numeric gate was added.
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


def _spelled_numbers(pattern_text: str) -> list:
    """Every `<number word> <thing>` a shipped page states, with where."""

    import re

    pattern = re.compile(pattern_text, re.IGNORECASE)
    found = []
    for path in _shipped_markdown():
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            word = match.group(1).lower()
            if word not in _NUMBER_WORDS:
                continue
            line = text.count(chr(10), 0, match.start()) + 1
            found.append((path.name, line, match.group(0),
                          _NUMBER_WORDS[word]))
    return found


def _status_writing_doors() -> set:
    """The commands that construct a ``StatusWriter``, read off the source.

    Not a transcription: every door that writes ``status.json`` names itself
    in the constructor call, so the set is the second positional argument of
    every ``StatusWriter(...)`` in the package.  Read with ``ast`` rather than
    by importing, because the doors' modules pull in the engine.
    """

    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "arwen_global"
    doors = set()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.id if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute) else "")
            if name != "StatusWriter" or len(node.args) < 2:
                continue
            second = node.args[1]
            if isinstance(second, ast.Constant) and isinstance(second.value, str):
                doors.add(second.value)
    return doors


def test_every_shipped_page_counts_the_long_running_doors_in_words_right():
    r"""The numeric gates' blind spot, on the sentence that has the subset.

    THE BREAKAGE THIS PREVENTS, measured on the Windows desktop 2026-09-10:
    README.md listed `run`, `go`, `render` and `run-plan` and then called them
    "the three long-running doors" in the same sentence, one commit after this
    file's own docstring was corrected for saying seven and listing eight.
    `test_every_shipped_page_states_the_command_count_the_parser_has` matches
    `(\d+) commands` and cannot see a number spelled in words, which is the
    form these subset sentences use by convention.  A reader counting the
    doors that leave a `status.json` gets a different answer from the page.
    """

    expected = len(_status_writing_doors())
    assert expected, "no door in this package constructs a StatusWriter"
    stated = _spelled_numbers(r"(\w+) long-running doors")
    assert stated, "no shipped page counts the long-running doors any more"
    wrong = [f"{name}:{line}: {text}" for name, line, text, value in stated
             if value != expected]
    assert not wrong, (
        f"{expected} commands write a status.json ("
        + ", ".join(sorted(_status_writing_doors()))
        + ") and these pages say otherwise:" + chr(10) + "  "
        + (chr(10) + "  ").join(wrong))
