"""No commit here rewrites a file's line endings, and the gate is measured.

THE BREAKAGE.  A file stored CRLF that a tool reads and writes back LF (or
the reverse) diffs whole: every line removed, every line re-added, the real
change buried inside.  Measured against the cut base `b1643e0` on the merged
tree: ten test files had flipped, nine CRLF to LF and one LF to CRLF, giving
7,660 added and 7,660 deleted lines of diff which carried 465 real changed
lines.  Those 465 were assertions, in the lanes whose whole change was
assertions, and the only way anyone read them was to normalise both sides
and re-diff.

They were then REPAIRED rather than recorded: each file was rewritten to the
convention `b1643e0` stores it under, so `git diff b1643e0..HEAD` shows the
465 lines and nothing else.  `BASELINE` is empty because of that repair, and
`test_the_cut_carries_no_line_ending_rewrite` holds the range at zero.

Two halves hold it now.  `.gitattributes` sets `* -text`, so git itself
converts nothing on checkout or commit on any platform.  This file drives
`tools/check_line_endings.py`, which measures what a tool did anyway.

THE INSTRUMENT IS TESTED IN BOTH DIRECTIONS, because a checker that never
fires and a checker that always fires read the same from the green side.
The two tests at the bottom build a throwaway repository, flip one file in
it, and require the report to name it; then change the same file without
flipping it, and require silence.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.check_line_endings import (  # noqa: E402
    BASELINE, convention, flipped, main, mixed_in_tree)

REPO = Path(__file__).resolve().parents[1]
GIT = shutil.which("git")

#: The revision this 0.1.0 cut was taken from.  The gate's other range,
#: `main...HEAD`, is empty on `main` and measures nothing there.
CUT_BASE = "b1643e0"

#: Written once, so a refusal message never has to escape one.
NL = chr(10)


def _base_ref() -> str | None:
    """`main`, `origin/main`, or None when neither is fetched here.

    A checkout that fetched one branch has no `main` to compare against, and
    an instrument that reported "no flips" from a failed comparison would be
    the flawed kind.  None means the comparison cannot run, and the caller
    says so rather than passing.
    """

    for ref in ("main", "origin/main"):
        done = subprocess.run(
            ("git", "-C", str(REPO), "rev-parse", "--verify", "--quiet", ref),
            capture_output=True)
        if done.returncode == 0:
            return ref
    return None


def _run(repo: Path, *args: str) -> None:
    done = subprocess.run(("git", "-C", str(repo)) + args,
                          capture_output=True)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")


def _repo_with_two_commits(root: Path, first: bytes,
                           second: bytes) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    _run(root, "init", "-q", "-b", "main")
    _run(root, "config", "user.email", "gate@localhost")
    _run(root, "config", "user.name", "gate")
    # THE PIN IS PART OF THE INSTRUMENT.  Without it a machine whose global
    # config converts on commit stores the same blob for both sides, the
    # second commit reports nothing to commit, and a flip becomes
    # unmeasurable.  This repository carries the same two lines.
    _run(root, "config", "core.autocrlf", "false")
    (root / ".gitattributes").write_text("* -text\n", encoding="utf-8",
                                         newline="\n")
    _run(root, "add", ".gitattributes")
    target = root / "module.py"
    target.write_bytes(first)
    _run(root, "add", "module.py")
    _run(root, "commit", "-q", "-m", "first")
    base = subprocess.run(("git", "-C", str(root), "rev-parse", "HEAD"),
                          capture_output=True).stdout.decode().strip()
    _run(root, "checkout", "-q", "-b", "work")
    target.write_bytes(second)
    _run(root, "add", "module.py")
    _run(root, "commit", "-q", "-m", "second")
    return root, base


def test_no_text_file_in_this_tree_mixes_line_endings() -> None:
    """A half-rewritten file is what neither side of a flip check can see."""

    mixed = mixed_in_tree(REPO)
    assert mixed == [], (
        "text files hold both CRLF and bare LF at once:\n  "
        + "\n  ".join(f"{name}: {crlf} CRLF and {lf} LF"
                      for name, crlf, lf in mixed))


@pytest.mark.skipif(GIT is None, reason="git is not on PATH here")
def test_this_branch_adds_no_line_ending_rewrite() -> None:
    """Green means every flip in `main...HEAD` is one of the recorded ten.

    The recorded ten are in `BASELINE` with their counts and what each one
    hid.  History here is forward-only, so they cannot be undone; an
    eleventh is a new one and fails this.
    """

    if not (REPO / ".git").exists():
        pytest.skip("this tree is an unpacked sdist, not a checkout")
    base = _base_ref()
    if base is None:
        pytest.skip("neither main nor origin/main is fetched in this checkout")
    rows = flipped(REPO, base, "HEAD")
    unrecorded = [row for row in rows if (row[0], row[1]) not in BASELINE]
    assert unrecorded == [], (
        "a commit on this branch rewrote a file's line endings:\n  "
        + "\n  ".join(f"{name} {direction}: {a} CRLF / {b} LF -> "
                      f"{c} CRLF / {d} LF"
                      for name, direction, a, b, c, d in unrecorded))


@pytest.mark.skipif(GIT is None, reason="git is not on PATH here")
def test_the_cut_carries_no_line_ending_rewrite() -> None:
    """The range that is not empty once this work is on `main`.

    The test above compares against `main`, which is the right base while a
    lane is a branch and is EMPTY the moment that branch is merged: on `main`
    it passes without measuring anything.  This one pins the base the 0.1.0
    cut was taken from, so the release's own diff is what is held at zero.

    Not a suppression list.  The ten flips that happened were rewritten back
    to the convention `CUT_BASE` stores them under, which is why `BASELINE`
    is empty; an eleventh flip anywhere in the cut fails here by name.
    """

    if not (REPO / ".git").exists():
        pytest.skip("this tree is an unpacked sdist, not a checkout")
    done = subprocess.run(
        ("git", "-C", str(REPO), "rev-parse", "--verify", "--quiet",
         CUT_BASE + "^{commit}"), capture_output=True)
    if done.returncode != 0:
        pytest.skip(f"the cut base {CUT_BASE} is not in this checkout")
    rows = flipped(REPO, CUT_BASE, "HEAD")
    unrecorded = [row for row in rows if (row[0], row[1]) not in BASELINE]
    assert unrecorded == [], (
        "a file's line endings were rewritten between the cut base and the "
        "tip, so its real change is buried in a whole-file diff:" + NL
        + NL.join(f"  {name} {direction}: {a} CRLF / {b} LF -> "
                      f"{c} CRLF / {d} LF"
                      for name, direction, a, b, c, d in unrecorded))
    assert BASELINE == {}, (
        "the repair retired this table; a row here is a flip that was "
        "recorded instead of repaired: " + repr(sorted(BASELINE)))


@pytest.mark.skipif(GIT is None, reason="git is not on PATH here")
def test_the_gate_names_a_file_that_was_flipped(tmp_path: Path) -> None:
    """One direction of the instrument: a flip is reported and refused."""

    body = b"import sys\nvalue = 1\n"
    root, base = _repo_with_two_commits(
        tmp_path / "flip", body.replace(b"\n", b"\r\n"), body)
    rows = flipped(root, base, "HEAD")
    assert rows == [("module.py", "crlf->lf", 2, 0, 0, 2)]
    assert main(["--repo", str(root), "--since", base]) == 1


@pytest.mark.skipif(GIT is None, reason="git is not on PATH here")
def test_the_gate_is_silent_when_a_file_only_changes(tmp_path: Path) -> None:
    """The other direction: a real edit that keeps the endings passes."""

    first = b"import sys\r\nvalue = 1\r\n"
    second = b"import sys\r\nvalue = 2\r\nextra = 3\r\n"
    root, base = _repo_with_two_commits(tmp_path / "keep", first, second)
    assert flipped(root, base, "HEAD") == []
    assert main(["--repo", str(root), "--since", base]) == 0
    assert convention(second) == "crlf"
