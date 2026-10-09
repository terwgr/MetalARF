"""Refuse a change that rewrites a file's line endings.

THE BREAKAGE.  A file stored CRLF that some tool reads and rewrites LF (or
the reverse) diffs whole.  Every line is removed and re-added, the real
change sits inside that, and no reviewer finds it.  Measured on this branch,
`main..734e7e0`: ten test files flipped, one LF to CRLF and nine CRLF to LF,
for 7,639 added and 7,591 deleted lines of diff over 428 real changed lines
counted by normalising both sides and re-diffing.  The lane that produced it
was retiring test skips, so what the churn hid was assertions.

`.gitattributes` pins the other half of this: `* -text` stops git itself from
converting anything on checkout or on commit.  Nothing in git stops a tool
that reads a file as text and writes it back, which is what happened, so this
gate measures the result.

WHAT IT MEASURES, because a gate whose scope nobody wrote down is a claim
nobody can check.

  * `--since REF` compares the stored bytes of every modified text file
    between `REF` and the tip, and refuses a file whose convention changed.
    It reports the CRLF and bare-LF counts on both sides, so the number in
    the refusal is the measurement, not a verdict word.
  * The tree walk refuses any text file that MIXES the two conventions,
    which is what a half-rewrite leaves behind and what neither half of a
    flip check can see.

It does not judge WHICH convention a file should use.  This tree has both on
purpose: the four Noah tables and thirty other files are CRLF because the
tree they were carved from stores them that way, and `match_eol` in
`tools/resync_from_owner.py` keeps a re-cut on whichever side each file
already has.  A gate that imposed one convention would rewrite all of them
and produce, once, exactly the diff it exists to prevent.

Usage::

    python tools/check_line_endings.py                  # main..HEAD + walk
    python tools/check_line_endings.py --since v0.1.0
    python tools/check_line_endings.py --tree-only      # no git needed

Exit status is 1 when anything is refused, 0 otherwise.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parent.parent

#: Extensions whose bytes a person reads and a reviewer diffs.  A suffix list
#: is the scope, and the scope is written down: `.npz` and `.png` hold byte
#: pairs that look like line endings and are not.
TEXT_SUFFIXES = (
    ".py", ".pyi", ".md", ".txt", ".toml", ".cfg", ".ini", ".json", ".yml",
    ".yaml", ".cu", ".cuh", ".c", ".h", ".rs", ".sh", ".in", ".TBL", ".csv",
)

#: THE FLIPS THAT HAPPENED, AND WERE REPAIRED.  Empty, and that is the
#: measurement rather than an absence of one.
#:
#: Ten test files flipped while the carried physics was being repointed on
#: the lane branches, nine CRLF to LF and one LF to CRLF, for 7,660 lines of
#: whole-file diff over 465 real changed lines.  Every one of the 465 was
#: re-read by normalising both sides and re-diffing before anything was
#: touched, and each file was then rewritten to the convention the cut base
#: `b1643e0` stores it under, in one named commit on `main`.  So the cut's
#: own diff, `git diff b1643e0..HEAD`, now shows those 465 lines and nothing
#: else, which is the diff a reader of this release actually opens.
#:
#: A row here suppresses a flip the tip still shows.  There are none: the
#: repair is what retired the table, and `tests/test_line_endings.py` holds
#: the range at zero rather than at ten.  A new flip is an eleventh, has no
#: row, and is refused.
BASELINE: dict[tuple[str, str], str] = {}


#: Directories that are never source.
SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist",
    ".mypy_cache", ".ruff_cache", "node_modules",
})


def is_text_path(name: str) -> bool:
    """True when `name` ends in one of the extensions this gate covers."""

    return name.endswith(TEXT_SUFFIXES) or name.rsplit("/", 1)[-1] in {
        ".gitignore", ".gitattributes"}


def counts(data: bytes) -> tuple[int, int]:
    """(CRLF count, bare-LF count) for `data`."""

    crlf = data.count(b"\r\n")
    return crlf, data.count(b"\n") - crlf


def convention(data: bytes) -> str:
    """`crlf`, `lf`, `mixed` or `none` for `data`."""

    crlf, lf = counts(data)
    if crlf and lf:
        return "mixed"
    if crlf:
        return "crlf"
    if lf:
        return "lf"
    return "none"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(("git", "-C", str(repo)) + args,
                          capture_output=True)


def _merge_base(repo: Path, since: str, tip: str) -> str:
    """Where `tip` left `since`, so the before side matches the diff range."""

    done = _git(repo, "merge-base", since, tip)
    if done.returncode != 0:
        return since
    return done.stdout.decode("utf-8").strip() or since


def _blob(repo: Path, rev: str, path: str) -> bytes | None:
    done = _git(repo, "show", f"{rev}:{path}")
    return done.stdout if done.returncode == 0 else None


def flipped(repo: Path, since: str, tip: str) -> list[tuple[str, str, int,
                                                            int, int, int]]:
    """Every text file whose stored convention differs between two revisions.

    Returns rows of (path, direction, CRLF before, LF before, CRLF after,
    LF after), BASELINE included: the caller decides what a baselined row
    means, so the measurement and the judgement stay separable.
    Raises `RuntimeError` when git cannot answer, because a gate that
    returns "nothing found" from a failed measurement is worse than no gate.
    """

    # Three dots: the comparison starts at the merge base, so a `since` that
    # has moved on does not report its own commits as this branch's.
    done = _git(repo, "diff", "--name-only", f"{since}...{tip}")
    if done.returncode != 0:
        raise RuntimeError(
            f"git diff {since}...{tip} failed in {repo}: "
            f"{done.stderr.decode('utf-8', 'replace').strip()}")
    rows = []
    for name in done.stdout.decode("utf-8").split("\n"):
        name = name.strip()
        if not name or not is_text_path(name):
            continue
        before = _blob(repo, _merge_base(repo, since, tip), name)
        after = _blob(repo, tip, name)
        if before is None or after is None:
            continue          # added or deleted: there is nothing to flip
        if convention(before) == convention(after):
            continue
        if convention(before) == "none" or convention(after) == "none":
            continue          # an emptied or newly-lined file, not a rewrite
        direction = f"{convention(before)}->{convention(after)}"
        rows.append((name, direction, *counts(before), *counts(after)))
    return rows


def mixed_in_tree(root: Path) -> list[tuple[str, int, int]]:
    """Every text file under `root` that holds both conventions at once."""

    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if SKIP_DIRS.intersection(path.relative_to(root).parts):
            continue
        name = path.relative_to(root).as_posix()
        if not is_text_path(name):
            continue
        crlf, lf = counts(path.read_bytes())
        if crlf and lf:
            rows.append((name, crlf, lf))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo", type=Path, default=REPO,
                        help="the tree to measure (default: this one)")
    parser.add_argument("--since", default="main",
                        help="the revision to compare against")
    parser.add_argument("--tip", default="HEAD",
                        help="the revision to compare")
    parser.add_argument("--tree-only", action="store_true",
                        help="skip the revision comparison")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    bad = 0

    if args.tree_only:
        print("revisions   not compared: --tree-only was given")
    elif _git(repo, "rev-parse", "--git-dir").returncode != 0:
        # NEVER SILENT.  An sdist is not a repository, and a reader who is
        # told nothing assumes the comparison ran and passed.
        print(f"revisions   not compared: {repo} is not a git checkout")
    else:
        try:
            rows = flipped(repo, args.since, args.tip)
        except RuntimeError as exc:
            # A failed measurement is not a pass.  The usual cause is a
            # checkout that fetched one branch, so `--since` names a ref that
            # is not there.
            print(f"revisions   NOT MEASURED: {exc}")
            return 1
        new = [row for row in rows if (row[0], row[1]) not in BASELINE]
        known = len(rows) - len(new)
        for name, direction, was_crlf, was_lf, now_crlf, now_lf in rows:
            token = "recorded  " if (name, direction) in BASELINE                 else "REWRITTEN "
            print(f"{token}  {name}: {was_crlf} CRLF / {was_lf} LF -> "
                  f"{now_crlf} CRLF / {now_lf} LF")
        for key in sorted(set(BASELINE) - {(row[0], row[1]) for row in rows}):
            # A recorded flip that no longer shows is not a breakage, and
            # refusing on it would turn every merge into a red gate.  It is
            # printed because a record nobody prunes is a record nobody
            # reads.
            print(f"stale       {key[0]} ({key[1]}) is recorded and not "
                  f"present in {args.since}...{args.tip}")
        bad += len(new)
        print(f"revisions   {args.since}...{args.tip}: {len(new)} file(s) "
              f"changed line endings, {known} recorded")

    mixed = mixed_in_tree(repo)
    for name, crlf, lf in mixed:
        print(f"MIXED       {name}: {crlf} CRLF and {lf} bare LF")
    bad += len(mixed)
    print(f"tree        {repo}: {len(mixed)} file(s) mix line endings")

    if bad:
        print(f"\nREFUSED: {bad} file(s).  A whole-file rewrite buries the "
              "real change inside it, and the reviewer reads the churn "
              "instead of the diff.  Restore the endings the file is stored "
              "with rather than committing the rewrite.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
