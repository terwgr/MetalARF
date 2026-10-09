"""How an experiment name resolves, what a refusal may say, and two page claims.

Every command of this package takes a config, and after the carve there are
two spellings for one: a path the reader has on disk, and the bare name of an
experiment that ships inside the wheel.  The pages were written when only the
first existed, so the failure this file guards is not hypothetical: seven
command lines on the quickstart page and six more across the other pages read
`configs/<name>.toml`, which is a directory in a checkout the reader of a
wheel does not have.

The last two tests hold claims the README makes about the doors, for the
same reason: both were measured against the installed wheel and both were
wrong in the direction that sends a reader somewhere the software is not.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from woof.globe import configs_dir


def test_a_bare_shipped_name_resolves_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved = configs_dir.resolve_config("arwen_global_moist_smoke")
    assert resolved.is_file()
    assert resolved.parent == configs_dir.config_root()


def test_a_path_that_exists_wins_over_a_shipped_name_of_the_same_spelling(
        tmp_path, monkeypatch):
    """A reader who copies and edits a config runs their copy."""

    monkeypatch.chdir(tmp_path)
    mine = tmp_path / "arwen_global_moist_smoke.toml"
    mine.write_text("# mine\n", encoding="utf-8")
    assert configs_dir.resolve_config(
        "arwen_global_moist_smoke.toml").resolve() == mine.resolve()


def test_a_shipped_name_written_as_a_path_is_refused_without_contradicting_itself(
        tmp_path, monkeypatch):
    """The refusal used to deny and offer the same string, two lines apart.

    THE BREAKAGE THIS PREVENTS, measured 2026-09-07 from the installed
    wheel: `woof global run configs/arwen_global_moist_smoke.toml` from a
    directory with no `configs/` answered

        configs\\arwen_global_moist_smoke.toml does not exist, and no
        shipped experiment is named 'arwen_global_moist_smoke'.
          did you mean: arwen_global_moist_smoke

    The suggestion list was computed from the stem whether or not the token
    carried a directory, while the lookup was gated on it carrying none, so
    the sentence named the same string as absent and as the remedy.
    """

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as caught:
        configs_dir.resolve_config("configs/arwen_global_moist_smoke.toml")
    message = str(caught.value)
    assert "no shipped experiment is named" not in message, message
    assert "did you mean" not in message, message
    assert "arwen_global_moist_smoke" in message
    assert "with no directory" in message


def test_a_real_typo_still_gets_the_shipped_list_and_a_suggestion(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as caught:
        configs_dir.resolve_config("arwen_global_moist_smok")
    message = str(caught.value)
    assert "no shipped experiment is named" in message
    assert "did you mean: arwen_global_moist_smoke" in message


def test_the_module_docstring_counts_the_configs_that_ship():
    """A count in a shipped docstring is a claim about the artefact.

    It said fifty-three (fifty plus three) while fifty-four ship.
    """

    words = {50: "fifty", 51: "fifty-one", 52: "fifty-two",
             53: "fifty-three", 54: "fifty-four", 55: "fifty-five",
             56: "fifty-six", 57: "fifty-seven"}
    shipped = configs_dir.list_configs()
    spectral = [name for name in shipped if name.startswith("global_spectral_")]
    globals_ = [name for name in shipped if name.startswith("arwen_global_")]
    assert len(spectral) + len(globals_) == len(shipped), sorted(shipped)

    text = configs_dir.__doc__ or ""
    assert words[len(shipped)] in text.lower(), (
        f"{len(shipped)} configs ship and the docstring does not say so")
    assert words[len(globals_)] in text.lower(), (
        f"{len(globals_)} arwen_global experiments ship and the docstring "
        "does not say so")


def test_no_shipped_config_is_named_with_a_directory_anywhere_in_the_pages():
    """The instrument that missed this class, run as a test.

    `tools/check_doc_examples.py` asked only whether argparse accepted the
    SHAPE of a documented line, so thirteen lines that exit 2 from an
    installed wheel were reported as accepted.
    """

    root = Path(__file__).resolve().parents[1]
    shipped = set(configs_dir.list_configs())
    offenders: list[str] = []
    pages = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    for page in pages:
        for number, line in enumerate(
                page.read_text(encoding="utf-8").splitlines(), start=1):
            for name in shipped:
                if f"configs/{name}" in line or f"configs\\{name}" in line:
                    offenders.append(
                        f"{page.relative_to(root).as_posix()}:{number}: {line.strip()}")
    assert offenders == [], (
        "a page names a shipped experiment as a path into a checkout the "
        "reader of a wheel does not have:\n  " + "\n  ".join(offenders))


def test_the_page_names_exactly_the_doors_that_write_a_status_file():
    """A claim about which commands a workspace can poll is checked.

    THE BREAKAGE THIS PREVENTS, measured 2026-09-07 from the installed
    wheel: the README said "Every long command writes `status.json` into
    its output directory ... and an append-only log whose path is the first
    line of stdout".  Two of the forty-three do.  `run`, the door the whole
    page is about, writes neither: its output directory after a completed
    T21 forecast holds the checkpoints, the receipt, the config copy and
    the diagnostics, and no status file and no log, and the first line of
    its stdout is a memory note.  A terminal workspace built on that
    sentence would poll a file that is never written for the longest
    command in the package.
    """

    import ast

    root = Path(__file__).resolve().parents[1]
    src = root / "src" / "arwen_global"
    writers = set()
    for module in sorted(src.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "StatusWriter"
                    and node.args
                    and len(node.args) > 1
                    and isinstance(node.args[1], ast.Constant)):
                writers.add(node.args[1].value)
    assert writers, "no StatusWriter call site found; this gate cannot see"

    readme = (root / "README.md").read_text(encoding="utf-8")
    claim = [line for line in readme.splitlines() if "`status.json`" in line]
    assert claim, "the README no longer says anything about status.json"
    sentence = claim[0]
    assert "Every long command" not in sentence, sentence
    for name in sorted(writers):
        assert f"`{name}`" in sentence, (
            f"`{name}` writes status.json and the README's sentence does not "
            f"name it: {sentence}")


def test_every_door_that_takes_a_config_resolves_a_shipped_bare_name():
    """A bare experiment name is either the spelling everywhere or nowhere.

    THE BREAKAGE THIS PREVENTS, measured 2026-09-10 from the installed wheel
    in an empty directory against published woof 2.7.2:

        woof global export arwen_global_moist_smoke out/smoke/*.npz \
          --outdir out/smoke-tapes --nlat 90 --nlon 180 \
          --start-date 2026-08-30_18:00:00
        woof global: [Errno 2] No such file or directory:
        'arwen_global_moist_smoke'

    That is the quickstart's own no-card rehearsal, line two of three, and it
    is the line a reader with no card runs first.  Eight doors declared their
    config as a bare `Path` while every other door used
    `configs_dir.config_argument`, so `run` resolved the shipped name and
    `export`, `assimilate`, `da init`, `da analyze`, `da static-covariance`,
    `da localisation`, `abi-score` and `abi-reference --config` opened it as a
    file in the working directory.  `configs_dir.config_argument`'s own
    docstring says it is the type for EVERY config argument, which is what
    made the eight invisible: the rule was written down and not enforced.

    The parser is the referent rather than a list of door names, so a door
    added later is covered by existing.  A `--config` that is optional is in
    scope too: it resolves the same way when it is given.
    """

    import argparse

    from woof.globe import cli

    def leaves(parser, path=()):
        subs = [row for row in parser._actions
                if isinstance(row, argparse._SubParsersAction)]
        if not subs:
            yield " ".join(path), parser
            return
        for name, child in subs[0].choices.items():
            yield from leaves(child, path + (name,))

    wrong = []
    for name, parser in leaves(cli.build_parser()):
        for action in parser._actions:
            names = [action.dest] + [option.lstrip("-").replace("-", "_")
                                     for option in action.option_strings]
            if "config" not in names:
                continue
            if action.dest in {"config_hash", "static_covariance"}:
                continue
            if action.type is not configs_dir.config_argument:
                wrong.append(f"{name}: {action.dest} has type {action.type!r}")

    assert not wrong, (
        "these doors take a config that a shipped experiment's bare name "
        "cannot spell, so a documented command line fails from any directory "
        "that has no such file:\n  " + "\n  ".join(wrong))
