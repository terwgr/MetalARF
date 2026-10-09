"""Published text carries no provenance, and this is the gate.

Everything in this repository is bound for a public GitHub repository and
a PyPI page.  An sdist on PyPI cannot be recalled, and a git history
cannot be rewritten here (forward commits only), so a leak is permanent
from the moment it is pushed.  Three separate classes of leak have
already cost this family real work:

* **Internal working material inside the artefact.**  The hex line
  shipped a document naming a private LAN address, a home-directory
  path, the maintainer's name against the machines he owns and the
  existence of a second private effort, inside every sdist built from
  its tree, because `graft docs` is a wildcard and a wildcard is how the
  thing nobody listed gets shipped.
* **A maintainer's name in a module docstring.**  Three lines of this
  package said whose Rust the data path is.  That is provenance about a
  person, in code, on a public index.
* **A private address in help text.**  A `--help` line named the subnet
  of a private network, which every user of the distribution would have
  read and none could use.

The gate has two scopes, because the two failures are different sizes.

`PROSE_SCOPE` is the text a reader reads: the README, the changelog, the
NOTICE, everything under `docs/`, the declaration and the workflows.  It
is held to the full standard, which includes the house style: no
em-dashes, and two words that are never used.

`SHIPPED_SCOPE` is every file that reaches an artefact and carries text
a person wrote: the Python, the CUDA device sources, and the markdown
that travels as package data.  It is held to the identity standard AND
to the house style, and that second half arrived in three widenings, each
one measured rather than reasoned about.  The two banned words came
first: measured 2026-09-07 with the word rule on the prose alone, every
page was clean while five comments and one test function name carried
one.  The em-dash followed: measured 2026-09-10, every page clean while
27 sat inside `src/arwen_global/core/npref.py`, which is package data in
every wheel built from this branch.  The process rule came last, in a
shipped form that drops one token and adds another (see
`_SHIPPED_PROCESS_REFERENCE`): three measurement rows named the scratch
directory a run happened to sit in.  A comment in a module is read by the
same people who read the README, so "a dash in a comment is a
typographic choice" was a distinction the reader does not make.

THE DEVICE SOURCES ARE IN SCOPE, and that is the second time this scope
was measured rather than reasoned about.  The carve that brought the
engine physics into this package moved fifteen `.cu`/`.cuh` files and
one markdown page into `src/`, all of them outside a glob that read
`src/**/*.py`, so every rule in this file passed over them.  Measured on
the tree 2026-09-09: one carried header cited the file name of the tool
a comment was written with, two carried kernels used a banned word four
times between them, and the shipped Noah table page opened on an
em-dash.  All of it was inside the wheel.  A glob that names one
extension is not a scope; it is the extension somebody happened to think
of.

The remedy for a carried device source is never a hand edit: the bytes
are reproduced from the source tree by `tools/resync_from_owner.py`, so
the rule that cleans them lives there (`CORE_PROSE`, `DEVICE_EXACT`) and
this file is what proves the rule ran.

WHAT THIS GATE DELIBERATELY DOES NOT COVER, so the next reader does not
mistake green for finished.

Machine names used to be on this list: measurement docstrings across
`src/` and `tests/` named the machines figures were taken on, and they were
reported by `tools/provenance_sweep.py` rather than asserted, as another
author's measurement text.  Since 0.1.2 they are asserted over everything
that ships (`test_no_shipped_file_names_a_private_machine`), because the
0.1.1 wheel carried six of them in carried physics comments on a public
index.  The card a figure was taken on stays: it is a measurement condition.

Lines that say "the <name> lane" or "the <name> lane's" name how the work
was divided rather than what the software does.  THE TALLY IS NOT IN THIS
PARAGRAPH, and that is the fix for how it went wrong twice: it read 67
over a file set nobody wrote down, then 71 over a file set it was stated
on and did not reproduce on.  A number in prose is a number a reader has
to take on trust.

The instrument is `lane_vocabulary_tally()` below.  It runs
`_LANE_VOCABULARY` over `shipped_scope()`, counts LINES per top-level
directory, and `test_lane_vocabulary_tally_is_the_recorded_one` asserts
what it returns against `_RECORDED_LANE_TALLY`, so the number is run
rather than quoted and a reword moves it in exactly one place.  Read the
tally there.

These sites are the same class as the rule above and are not in it,
because a pattern added before they are reworded turns the gate red on
dense measurement prose whose wording is another author's.  Rewording
them is named work, not a silent omission: until it is done, that
constant and that test are the record that the gate is quiet about them.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: This file states every banned pattern as a literal, so it necessarily
#: matches itself.  It is excluded by name rather than by a cleverness
#: that could also exclude something real.
SELF = Path(__file__).resolve()

_SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", "build", "dist",
               ".venv", "venv", "node_modules"}

#: SOMEBODY ELSE'S WORDS, REPRODUCED BECAUSE A LICENCE ASKS FOR THEM.
#:
#: These files are not this project's prose and this project may not edit
#: them.  MIT's substantive condition is that Arm's copyright notice and its
#: permission notice appear in every copy; FDLIBM's only condition is that
#: its notice be preserved; BSD-3-Clause clause 1 asks for the notice, the
#: conditions and the disclaimer retained unchanged; UCAR asks that its WRF
#: notice travel with any copy.  A house-style rule applied to one of them
#: would be a modification of the thing the licence says must be reproduced,
#: which is the opposite of performing it.
#:
#: So they are exempt from the rules below BY NAME, never by a glob: a new
#: file under licenses/ is in scope until somebody adds it here and says
#: whose words it is.  `test_the_third_party_texts_are_exempt_by_name`
#: holds the list to files that exist.
THIRD_PARTY_VERBATIM: tuple[str, ...] = (
    "licenses/LICENSE-Arm-optimized-routines-MIT.txt",
    "licenses/LICENSE-FDLIBM-SunPro.txt",
    "licenses/LICENSE-RTE-RRTMGP-BSD-3-Clause.txt",
    "licenses/LICENSE-WRF-public-domain.txt",
    "licenses/LICENSE-AER-RRTMG-BSD-3-Clause.txt",
    "licenses/NOTICE-AER-RRTMG-as-distributed-with-WRF.txt",
    "src/arwen_global/data/noah_tables/LICENSE-WRF.txt",
)


def _walk(patterns: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            if not path.is_file():
                continue
            if set(path.relative_to(ROOT).parts) & _SKIP_PARTS:
                continue
            if path.resolve() == SELF:
                continue
            if path.relative_to(ROOT).as_posix() in THIRD_PARTY_VERBATIM:
                continue
            found.append(path)
    return sorted(set(found))


def prose_scope() -> list[Path]:
    """The text a reader reads.

    `src/**/*.md` is here rather than in the shipped scope alone because a
    page that travels inside the wheel as package data is read by whoever
    opens the directory it documents.  The Noah table provenance page is the
    measured case: it ships beside the four tables and opened on an em-dash.
    """

    return _walk((
        "*.md", "NOTICE", "docs/**/*.md", "docs/*.md", "src/**/*.md",
        "pyproject.toml", ".github/workflows/*.yml", ".github/*.md",
    ))


def shipped_scope() -> list[Path]:
    """Every file that reaches a wheel or an sdist and carries written text.

    The device sources are here for the reason the docstring at the top of
    this file gives: they are inside the wheel, they are read by whoever
    reads the rest of the source, and a glob naming one extension is not a
    scope.

    `src/**/*.toml` is the fifty-five shipped experiments.  They travel as
    package data and carry comments a reader of `woof global configs`
    opens, and they were outside every glob in this file until the tally
    below was measured both ways and read four lines higher over the tree
    than over this scope; those four lines are in four of those
    experiments.

    `src/**/*.json` is the shipped data: the static covariance receipt and
    the radiance tables.  They were outside this scope until 2026-09-10,
    when four of them reached a public push carrying the input paths they
    were built from, and those paths began with a home directory that names
    the maintainer.  A record of where an input sat on one machine is not
    the input, so the path keeps its file name and loses its root.

    `src/**/*.npz` is the shipped data a text tool cannot read, and it is
    the fourth widening.  MEASURED 2026-09-10, hours after the JSON above
    was scrubbed: the covariance table's own `__metadata__` member is a
    UTF-32 unicode array inside a zip carrying the SAME twenty-three paths
    that had just been taken out of its JSON twin, and it went through
    every rule in this file, through `git grep`, and through `strings`,
    untouched.  A scope that stops at what opens as text reports green on
    the copy nobody can read.  `_archive_documents` below opens the members
    and hands their decoded text to every rule here.

    `src/**/*.txt` and `licenses/*.txt` are the fifth widening, and they
    arrived with the third-party notices in 0.1.1.  The beside-the-code
    notice in `src/arwen_global/core/kernels/` is this project's own prose
    about somebody else's code: it ships inside the wheel, a reader who
    opens the kernel directory reads it first, and every rule here applies
    to it.  The verbatim licence texts are the exception and they are
    excluded by name in `THIRD_PARTY_VERBATIM`, not by extension, so a new
    file under `licenses/` is in scope until somebody says whose words it
    is.
    """

    return sorted(set(_walk((
        "src/**/*.py", "tools/**/*.py", "tests/**/*.py", "*.py",
        "src/**/*.cu", "src/**/*.cuh", "src/**/*.toml", "src/**/*.json",
        "src/**/*.npz", "src/**/*.txt", "licenses/*.txt",
    ))) | set(prose_scope()))


def _decode_member(descr: object, payload: bytes) -> str | None:
    """The text inside one stored array, or None when it holds no text.

    A numpy array declares its own type in its header, so this asks rather
    than guesses, and it reads only the members that CAN carry written
    text.  A float array is skipped by name: NUL-stripping eight bytes of
    mantissa and calling the result UTF-8 produces text nobody wrote, and a
    gate that reports a leak inside a covariance matrix is an instrument
    nobody will keep.

      * `U` is UCS-4, which is UTF-32 in the byte order the descriptor
        states, and every character of ASCII text in it is one byte
        followed by three NULs.  It is decoded as UTF-32 and the padding
        stripped.
      * `S` and `a` are bytes, NUL-padded to the widest element.
      * anything else that is not numeric (an object member, a structured
        record, a two-byte text member some other writer produced) falls
        back to stripping NULs and reading UTF-8, which is what recovers
        ASCII out of UTF-16 as well.
    """

    if not isinstance(descr, str):
        return payload.replace(b"\x00", b"").decode("utf-8", "replace")
    if descr[:1] in ("<", ">", "|", "="):
        order, kind = descr[0], descr[1:2]
    else:
        order, kind = "|", descr[:1]
    if kind in "biufc":
        return None
    if kind == "U":
        codec = "utf-32-be" if order == ">" else "utf-32-le"
        return payload.decode(codec, "replace").replace("\x00", "")
    if kind in ("S", "a"):
        return payload.decode("utf-8", "replace").replace("\x00", "")
    return payload.replace(b"\x00", b"").decode("utf-8", "replace")


def _archive_documents(path: Path, label: str) -> list[tuple[str, str]]:
    """Every text-carrying member of one stored archive, `file::member`."""

    out: list[tuple[str, str]] = []
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                raw = archive.read(name)
                if not raw.startswith(b"\x93NUMPY"):
                    out.append((f"{label}::{name}",
                                raw.replace(b"\x00", b"").decode(
                                    "utf-8", "replace")))
                    continue
                if raw[6] == 1:
                    start = 10 + int.from_bytes(raw[8:10], "little")
                    header = raw[10:start]
                else:
                    start = 12 + int.from_bytes(raw[8:12], "little")
                    header = raw[12:start]
                try:
                    descr = ast.literal_eval(
                        header.decode("latin-1").strip())["descr"]
                except (SyntaxError, ValueError, KeyError, TypeError):
                    descr = None
                text = _decode_member(descr, raw[start:])
                if text:
                    out.append((f"{label}::{name}", text))
    except (OSError, zipfile.BadZipFile):
        return []
    return out


def _documents(paths: list[Path]) -> list[tuple[str, str]]:
    """Every file in a scope as `(label, text)`, archives opened.

    A label is the repository-relative path, and for a member of a stored
    archive it is `path::member`, so a refusal names the thing to edit.
    """

    out: list[tuple[str, str]] = []
    for path in paths:
        label = path.relative_to(ROOT).as_posix()
        if path.suffix == ".npz":
            out.extend(_archive_documents(path, label))
            continue
        try:
            out.append((label, path.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue
    return out


def _hits(paths: list[Path], expression: re.Pattern[str]) -> list[str]:
    out: list[str] = []
    for label, text in _documents(paths):
        lines = text.splitlines()
        for match in expression.finditer(text):
            number = text[:match.start()].count("\n") + 1
            body = lines[number - 1].strip() if number <= len(lines) else ""
            out.append(f"{label}:{number}: {body[:160]}")
    return out


#: HOW THE WORK WAS DIVIDED.  Not asserted anywhere: the docstring at the
#: top of this file says why, and `lane_vocabulary_tally()` is what makes
#: that a measurement rather than a claim.
#:
#: Case-sensitive on purpose.  A capitalised "Lane" opening a sentence is a
#: different site and there are none of those; matching it would move the
#: tally without moving any prose.
_LANE_VOCABULARY = re.compile(r"the [a-z][a-z-]* lane('s)?")


def lane_vocabulary_tally() -> dict[str, int]:
    """Lines matching `_LANE_VOCABULARY`, per top-level directory.

    LINES, not matches: two on one line is one site to reword.  The scope
    is `shipped_scope()`, every file that reaches a wheel or an sdist and
    carries written text, so the tally covers what a reader of the
    artefact can find.  This file is outside that scope by `SELF`, so the
    paragraph and the test that record the tally do not enter it.

    `total` is the sum of the rest.  Re-take it from the repository root::

        python -c "import sys; sys.path.insert(0, 'tests'); import
        test_no_provenance as t; print(t.lane_vocabulary_tally())"
    """

    tally: dict[str, int] = {}
    for label, text in _documents(shipped_scope()):
        hits = sum(1 for line in text.splitlines()
                   if _LANE_VOCABULARY.search(line))
        if hits:
            top = label.split("/")[0]
            tally[top] = tally.get(top, 0) + hits
    tally["total"] = sum(tally.values())
    return tally


#: What `lane_vocabulary_tally()` returned at this tip, on the Windows
#: desktop 2026-09-10.  Asserted rather than quoted in prose: a reword that
#: moves the count turns this red and moves the number here, which is the
#: failure mode both earlier tallies had.
_RECORDED_LANE_TALLY: dict[str, int] = {
    "docs": 8, "src": 57, "tests": 16, "tools": 1, "total": 82,
}


def test_lane_vocabulary_tally_is_the_recorded_one() -> None:
    measured = lane_vocabulary_tally()
    assert measured == _RECORDED_LANE_TALLY, (
        "the count of lines naming how the work was divided has moved.  "
        "The docstring at the top of this file sends a reader here for the "
        "number, so re-take it and update _RECORDED_LANE_TALLY:\n"
        f"  measured {measured}\n  recorded {_RECORDED_LANE_TALLY}"
    )


# ---------------------------------------------------------------------------
# identity: what must not escape, anywhere in the shipped tree
# ---------------------------------------------------------------------------

#: Names of the tools a file might have been written with, the commit
#: trailer that carries one, and the DIRECTORY one of them keeps its
#: working notes in.  A published artefact says what it does, not what
#: produced it.
#:
#: THE BREAKAGE THE LAST ALTERNATIVE PREVENTS, measured inside the built
#: wheel on 2026-09-10: two comment lines in the carried physics driver
#: cited files under that directory on the private source tree, one as
#: ``docs/<name>/plans/...`` and one as ``.<name>/sdd/...``.  Neither path
#: resolves in any install.  The carve HAS machinery for source-tree paths
#: -- `SLASH_PATHS` in `tools/resync_from_owner.py` rewrites them and
#: `CARRIED_PATH_FORBIDDEN` is its post-condition -- but that machinery
#: only reaches files the carve TAKES, and it takes nothing from there, so
#: both lines went through every rule in this file untouched.  The
#: alternative is the directory NAME rather than a path shape, because the
#: two spellings measured share nothing else.
_TOOLING = re.compile(
    r"\b(claude|anthropic|chatgpt|openai|copilot|"
    r"co-authored-by|subagents?|sub-agents?|superpowers)\b", re.IGNORECASE)

#: A personal name, case-sensitive on purpose: its lower-case spelling is
#: an ordinary English verb this package uses (an ice pack drawing energy
#: up through snow), so a case-insensitive rule here would fire on physics
#: and catch nothing extra.  The name is built from its code points, so
#: this file does not carry the name it screens for.
_NAME = "".join(map(chr, (0x44, 0x72, 0x65, 0x77)))
_PERSONAL_NAME = re.compile(r"\b" + _NAME + r"\b")

#: Home directories and private address space.  Both are the hex line's
#: measured leak: a path that resolves on one machine, and a subnet
#: nobody outside one building can route to.
#: A backslash is built from its code point rather than written as an
#: escape: this pattern has to survive being copied through shells and
#: editors that collapse a doubled backslash, and a pattern that has
#: quietly become "literal plus sign" matches nothing while looking
#: exactly like a clean tree.  The self-test below is what caught it.
_BACKSLASH = chr(92)
_MACHINE_LOCATION = re.compile(
    "(?:[A-Za-z]:" + _BACKSLASH * 2 + "+Users" + _BACKSLASH * 2 + "+"
    r"|/home/[a-z][a-z0-9_-]*/"
    r"|\b10\.250\.\d|\b192\.168\.\d|\b172\.(?:1[6-9]|2\d|3[01])\.\d)")

#: A HOME DIRECTORY IN THE OTHER SPELLING, and what one is allowed to name.
#:
#: THE BREAKAGE THIS PREVENTS, measured 2026-09-10 in the built sdist: a
#: tool's usage docstring pointed at the copy of itself that runs on a node,
#: under a working directory named after a private revision and a directory
#: of scoring tooling that exists on no published tree.  `_MACHINE_LOCATION`
#: above is written for the `/home/<user>/` spelling, and a tilde is the
#: same path with its root written portably, so every alternative in that
#: rule passed over it.
#:
#: Stated as what a `~/` path MAY name rather than as a pattern for the
#: private ones, for the reason `_ALLOWED_HOST_VALUE` is stated that way:
#: the private shapes a rule lists are the ones somebody thought of.  The
#: three allowed are the locations this distribution's own pages send a
#: user to ON THEIR OWN MACHINE, the geography archive, this family's
#: configuration directory and the package cache; a tilde path naming
#: anything else is a path on somebody's machine, which is the only kind
#: this rule has met.
_HOME_PATH = re.compile("~[/" + _BACKSLASH * 2 + "]([A-Za-z0-9_.-]+)")
_ALLOWED_HOME_TARGET = re.compile(r"\A(?:WPS_GEOG|\.woof|\.cache)\Z")

#: A DIRECTORY OF TOOLING THAT IS NOT THIS REPOSITORY'S.
#:
#: THE BREAKAGE THIS PREVENTS, measured 2026-09-10 in the built sdist:
#: three files named the same directory of scoring tooling on the control
#: chain's own tree, twice inside a path and once in prose as a bare word.
#: A path rule reaches the first two.  The third is an ordinary English
#: sentence with one word in it that resolves nowhere, and nothing here
#: could see it: it carries no separator, no root and no branch spelling.
#:
#: So the rule is the SHAPE such a directory takes, `<what it does>tools`,
#: which is what the measured one is called.  This repository's own tooling
#: directory is `tools/`, one word with no prefix, and the leading letter
#: class is what excludes it.  The allowed compounds are the two libraries
#: this package imports plus the standard-library sibling of the first, so
#: an import line is not a refusal; anything else with that ending is a
#: directory on a tree the reader cannot open.
_PRIVATE_TOOLING_DIRECTORY = re.compile(
    r"\b(?!functools\b|itertools\b|setuptools\b)[a-z][a-z0-9]*tools\b")

#: How the work was ORGANISED, which is not part of what the software is.
#:
#: THE BREAKAGE THIS PREVENTS, and it had already happened when this rule
#: was written (measured 2026-09-07, twenty-four sites across eleven
#: files): six pages and eight modules of this package were carried across
#: from a private development tree still speaking that tree's own
#: vocabulary.  Two documentation pages opened with the git branch they
#: were written on, one closed with a section headed "Commits on
#: <branch>", four sites pointed a reader at a folder inside the
#: maintainer's Downloads directory, three shipped modules named a branch
#: in a comment, and two shipped constants carried the name of a private
#: branch's tip in their own identifiers.  None of it resolves for a
#: reader, all of it describes the process rather than the product, and
#: every earlier rule in this file passed over it.
#:
#: Each alternative is a thing a reader outside the project cannot
#: resolve:
#:   lane/... and integration/... -- branch names on a tree that has no
#:       public remote.  The boundary is on the left only: "plane/" has a
#:       word character before the l and does not match.
#:   Downloads/ ...              -- the maintainer's own delivery folder.
#:   "the refuter", "the owner tip", "the lane report" -- roles and
#:       artefacts of the development process, not of the software.
#: No global IGNORECASE, and the reason is a measured false positive:
#: `Downloads` is the maintainer's own delivery folder while `downloads`
#: is a verb these very pages use ("the door downloads the station
#: list"), so that alternative is case-sensitive and the rest carry
#: their own scoped `(?i:...)`.
#: THE SECOND WIDENING, and the same failure it was written for.  Measured
#: 2026-09-10 inside the built wheel: thirteen measurement rows in
#: `sizing.py` and four lines in two test files named the run directories
#: the figures came off, on a tree that has no public remote.  Every one
#: of them read as a folder rather than as a branch, so the alternatives
#: above passed over all seventeen: `lane/` was required literally and the
#: sites spelled it `<word>-lane<digit>/`.  The directory shapes are now
#: refused as well:
#:   <word>-lane<digits>/  -- how the work was split, as a folder name.
#:   <word>-merge/         -- where two splits were brought together.
#:   work/<word>-<word>/   -- the working root those sat under.
#:   refutation/           -- a stage of the process, not of the software.
#:   <word>-<word>/out|results|runs/ -- a run directory on a private tree,
#:       which is the shape all thirteen sizer rows carried.
#: Bounded on the left so an ordinary hyphenated word inside a longer path
#: does not match, and the trailing separator is required so the words
#: themselves stay usable in prose.
#: THE WORKING ROOT IS NOW ANYTHING UNDER IT, and that is the third round
#: on the same rule.  Measured 2026-09-10 inside the wheel, after the
#: seventeen above were cleared: one comment in the carried physics driver
#: cited the script a measurement had been taken with, and the alternative
#: written for that root required a HYPHENATED directory beneath it while
#: the site was an UNDERSCORED file name directly under it.  A rule that
#: reads only the shapes already met reports green on the next one, and
#: nobody writes that root with a separator after it except as that root,
#: so the whole path is refused, file or directory.
#: THE FOURTH WIDENING: a planning document and its review.  Measured on the
#: 0.1.2 source tree before it was published: seven comment and docstring
#: lines across two tools and two test files cited a plan's audit and the
#: audit's finding numbers ("finding" followed by a letter and a number) as
#: the source of a measurement.  Neither resolves for a reader, and every
#: alternative above passed over all seven, because they read as ordinary
#: prose.  Refused now, and each of the seven carried both:
#:   "plan audit" / "plan's audit", across a line break too;
#:   "finding" or "findings" followed by a letter-and-number label.
#: A label a page defines for its own reader is written without the word
#: ("F1 of the calibration, below"), which is how the one such page in
#: docs/ now spells it.
_DEVELOPMENT_ORGANISATION = re.compile(
    r"(?<![A-Za-z])(?i:lane|integration)/[a-z0-9][a-z0-9._-]*"
    r"|\bDownloads[/\\]"
    r"|\bDownloads evidence"
    r"|(?i:\brefuters?\b)"
    r"|(?i:\bowner (?:tip|branch)\b)"
    r"|(?i:\blane reports?\b)"
    r"|(?<![A-Za-z])[a-z][a-z0-9]*-lane\d+[/\\]"
    r"|(?<![A-Za-z])[a-z][a-z0-9]*-merge[/\\]"
    r"|(?<![A-Za-z])work[/\\][A-Za-z0-9_.-]+"
    r"|(?<![A-Za-z])refutation[/\\]"
    r"|(?<![A-Za-z])[a-z][a-z0-9]*-[a-z][a-z0-9]*[/\\](?:out|results|runs)"
    r"[/\\]"
    r"|(?i:\bplan(?:'s)?\s+audit\b)"
    r"|(?i:\bfindings?)\s+[A-Z]\d+\b")

#: A MACHINE BY NAME, in anything that ships.
#:
#: THE BREAKAGE THIS PREVENTS, measured 2026-09-10 inside the built wheel:
#: the shipped render catalog opened with the hostname of the desktop it
#: was measured on and the exact operating-system build that desktop runs.
#: `_MACHINE_NAME` above did not see it and could not: that rule lists the
#: machines whose names are ALLOWED, so a name outside the list is exactly
#: what it passes.  This is the other direction, and it is the direction a
#: leak arrives from.
#:
#: The two shapes are the ones an operating system hands out by default: a
#: Windows installation names itself DESKTOP-<random> and a name ending
#: `-PC` is the other default.  `-PC` needs the boundary on the right, so
#: `PCIe` and `-PCI` do not match.
_HOSTNAME = re.compile(r"\bDESKTOP-[A-Z0-9]+\b|\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*-PC\b")

#: What a `"host"` field in shipped data is allowed to say.  A stated set
#: rather than a pattern for "looks private": a CLASS of machine and, since
#: 0.1.2, nothing else (the node names this family allowed here before
#: 0.1.2 are gone).  A class of machine is what a reader can act on ("does
#: this stored measurement apply to mine?").  A hostname answers a question
#: nobody outside one building can ask.  The class is the operating-system
#: family with the architecture, which is what
#: `tools/arwen_global_render_catalog._machine_class` writes.
_HOST_FIELD = re.compile(r'"host"\s*:\s*"([^"]*)"')
_ALLOWED_HOST_VALUE = re.compile(
    r"\A(?:windows|linux|darwin|macos)-[a-z0-9_]+\Z")


def test_no_tooling_names_reach_the_shipped_tree() -> None:
    hits = _hits(shipped_scope(), _TOOLING)
    assert hits == [], (
        "the name of a tool a file was written with, or a commit trailer "
        "carrying one, is inside a file that reaches the artefact:\n  "
        + "\n  ".join(hits)
    )


def test_no_personal_name_reaches_the_shipped_tree() -> None:
    hits = _hits(shipped_scope(), _PERSONAL_NAME)
    assert hits == [], (
        "a personal name is inside a file that reaches the artefact.  "
        "Three module docstrings in this package said whose Rust the data "
        "path is; a published module describes the code, not its "
        "author:\n  " + "\n  ".join(hits)
    )


def test_no_development_organisation_reaches_the_shipped_tree() -> None:
    """A branch name, a delivery folder or a development role is not product.

    These read as ordinary prose, which is why they survived every other
    rule here: "Lane `lane/da-ensemble` (2026-09-06)" is a grammatical
    English sentence and names nothing this file used to look for.  It is
    also the first line a reader of that page sees, and it points at a
    branch on a tree with no public remote.
    """

    hits = _hits(shipped_scope(), _DEVELOPMENT_ORGANISATION)
    assert hits == [], (
        "a file that reaches the artefact names how the work was "
        "organised rather than what the software does: a development "
        "branch, the maintainer's delivery folder, or a role or report "
        "that exists only inside the project:\n  " + "\n  ".join(hits)
    )


def test_no_shipped_file_names_a_machine_by_hostname() -> None:
    """A default operating-system hostname is a person's machine.

    THE BREAKAGE THIS PREVENTS: the render catalog that ships inside every
    wheel carried `DESKTOP-<random>` in its first field and the exact
    operating-system build beside it, because the tool that wrote it asked
    the operating system for a name.  It writes the KIND of machine now,
    and this refuses the shape a name comes in.
    """

    hits = _hits(shipped_scope(), _HOSTNAME)
    assert hits == [], (
        "a file that reaches the artefact names a machine by its "
        "hostname.  A reader cannot reach it and does not need it; record "
        "the kind of machine instead:\n  " + "\n  ".join(hits)
    )


def test_every_shipped_host_field_names_a_machine_class() -> None:
    """The other direction: what a `host` field IS, not what it is not.

    A pattern for names that look private catches the names somebody
    thought of.  This states what the field may say and refuses the rest,
    so a hostname in a spelling nobody predicted is red on arrival.
    """

    hits = []
    for label, text in _documents(shipped_scope()):
        lines = text.splitlines()
        for match in _HOST_FIELD.finditer(text):
            value = match.group(1)
            if _ALLOWED_HOST_VALUE.match(value):
                continue
            number = text[:match.start()].count("\n") + 1
            body = lines[number - 1].strip() if number <= len(lines) else ""
            hits.append(f"{label}:{number}: {body[:160]}")
    assert hits == [], (
        "a `host` field in shipped data says something that is not a "
        "machine class (an operating-system family with an architecture, "
        "which is what a reader can act on):\n  " + "\n  ".join(hits)
    )


def test_no_home_directory_or_private_address_is_shipped() -> None:
    hits = _hits(shipped_scope(), _MACHINE_LOCATION)
    assert hits == [], (
        "a home-directory path or a private network address is inside a "
        "file that reaches the artefact.  A path that resolves on one "
        "machine is not an instruction to anybody else, and a private "
        "subnet in a --help line is an address no reader can route "
        "to:\n  " + "\n  ".join(hits)
    )


def test_every_home_path_in_shipped_text_is_a_user_side_location() -> None:
    """A `~/` path is either the reader's own or somebody else's.

    THE BREAKAGE THIS PREVENTS: a scoring tool that ships in the sdist
    opened by naming the copy of itself that runs on a node, under a
    working directory named for a private revision.  The reader cannot
    reach it and is not meant to, and the sentence around it is about what
    the tool scores, so the pointer is what goes.
    """

    hits = []
    for label, text in _documents(shipped_scope()):
        lines = text.splitlines()
        for match in _HOME_PATH.finditer(text):
            if _ALLOWED_HOME_TARGET.match(match.group(1)):
                continue
            number = text[:match.start()].count("\n") + 1
            body = lines[number - 1].strip() if number <= len(lines) else ""
            hits.append(f"{label}:{number}: {body[:160]}")
    assert hits == [], (
        "a file that reaches the artefact names a path under somebody's "
        "home directory.  A tilde path in shipped text may only name a "
        "location on the READER's machine that this distribution's own "
        "pages send them to:\n  " + "\n  ".join(hits)
    )


def test_no_private_tooling_directory_is_named_in_the_shipped_tree() -> None:
    """A directory of tooling on another tree is not part of this one.

    THE BREAKAGE THIS PREVENTS: two tools and one test named the control
    chain's own directory of scoring scripts, twice as a path and once as a
    bare word in a sentence.  The bare word is why this rule is written on
    the NAME rather than on the path: it carried no separator, no root and
    no branch spelling, so every path rule in this file passed over it
    while the word itself resolves for nobody.
    """

    hits = _hits(shipped_scope(), _PRIVATE_TOOLING_DIRECTORY)
    assert hits == [], (
        "a file that reaches the artefact names a directory of tooling "
        "that is not in this repository.  This repository's own is "
        "`tools/`; a compound ending in the same word is a folder on "
        "somebody else's tree:\n  " + "\n  ".join(hits)
    )


# ---------------------------------------------------------------------------
# house style: the prose a reader reads
# ---------------------------------------------------------------------------

#: Two words this project does not use.  Stated as a rule rather than a
#: preference: they are the two that kept coming back after being struck
#: out, which is what makes a gate the right instrument instead of an
#: intention.
#:
#: The leading boundary is `(?<![\w])` rather than `\b` and the trailing one
#: allows an underscore, because an underscore is a word character: `\b` finds
#: no boundary in `..._honest_full_stop`, so a test FUNCTION NAME carrying the
#: word passed this gate while the same word in the sentence above it would
#: not have (measured 2026-09-07, one name and five comments).
_BANNED_WORDS = re.compile(
    r"(?<![A-Za-z])(honest\w*|load[-_ ]bearing)(?![A-Za-z])", re.IGNORECASE)

#: An em-dash, named by code point so this line survives a re-encode.
_EM_DASH = re.compile("\u2014")

#: Where the text came from is never part of the text.  A published page
#: describes the software; it does not describe the process that wrote it
#: or the discussion that decided it.
_PROCESS_REFERENCE = re.compile(
    r"\b(conversations?|chats?|this session|the session|"
    r"a previous session|the agent|an agent)\b", re.IGNORECASE)

#: THE SAME RULE FOR THE SHIPPED SCOPE, one token out and one in.  Measured
#: 2026-09-10 with the prose pattern turned on the shipped tree.
#:
#: OUT: "the session".  This package has a `CardSession` type and a section
#: header that reads "The session: what the model asks", so on shipped code
#: that token fires on domain vocabulary.  An instrument that reports a real
#: noun as a leak is a flawed instrument, and the two conversational forms
#: ("this session", "a previous session") carry the meaning the rule is for.
#:
#: IN: `agent[-_ ]scratch`.  Three measurement rows in `sizing.py` named the
#: scratch directory the run happened to sit in, inside the wheel, and the
#: prose pattern would not have matched them even where it ran.  They now
#: name the host shape and the probe, which is what a reader can use.
_SHIPPED_PROCESS_REFERENCE = re.compile(
    r"\b(conversations?|chats?|this session|a previous session|"
    r"the agent|an agent)\b|agent[-_ ]scratch", re.IGNORECASE)

#: The machines the maintainer owns, by hostname.  Enforced in prose and,
#: since 0.1.2, in everything that ships (see the test that names it).
_MACHINE_NAME = re.compile(r"\b(weather-node-\d+|node-[1-9]\b)")

#: The private case whose working name reached eight docstrings in the
#: carried physics of 0.1.1.  Spelled in pieces so the detector that lists
#: case tokens (tools/check_case_token_leakage.py, exempt by name below)
#: is the only other file that spells it whole.
_PRIVATE_CASE = re.compile("(?i)" + "real" + "74")

#: The one file that must spell the case token: it is the detector that
#: refuses the token in generic code positions.
_CASE_TOKEN_DETECTOR = ROOT / "tools" / "check_case_token_leakage.py"


def test_published_prose_uses_neither_banned_word() -> None:
    """The scope is everything that ships, not only the markdown.

    The repository is public and the source goes with it, so a comment in a
    module is read by the same people who read the README.  Measured
    2026-09-07 with the scope at prose only: five comments and one test
    function name carried a banned word while every page was clean.
    """

    hits = _hits(shipped_scope(), _BANNED_WORDS)
    assert hits == [], (
        "text that ships uses one of the two words this project does not "
        "use:\n  " + "\n  ".join(hits)
    )


def test_published_prose_carries_no_em_dash() -> None:
    """The scope is everything that ships, and that is the third widening.

    Measured on the tree 2026-09-10 with the rule on the prose alone:
    every page was clean while 27 em-dashes sat inside
    `src/arwen_global/core/npref.py`, which is package data in every wheel
    built from this branch.  The remedy for a carried file is never a hand
    edit: `tools/resync_from_owner.PROSE_DASH` rewrites them when the file
    is cut, and this is what proves the rule ran.  The device sources are
    in this scope and carry none: their bytes are digested, so a rule that
    could rewrite one would move a receipt's kernel digest, and
    `DEVICE_FORBIDDEN` stops the cut instead.
    """

    hits = _hits(shipped_scope(), _EM_DASH)
    assert hits == [], (
        "text that ships carries an em-dash (U+2014).  The house style "
        "uses a comma, a colon or a full stop instead:\n  "
        + "\n  ".join(hits)
    )


def test_published_prose_references_no_process() -> None:
    hits = _hits(prose_scope(), _PROCESS_REFERENCE)
    assert hits == [], (
        "published prose refers to the process that produced it rather "
        "than to the software it describes:\n  " + "\n  ".join(hits)
    )


def test_shipped_text_refers_to_no_process() -> None:
    """The scope is everything that ships, for the reason the words test gives.

    Measured 2026-09-10 with this rule on the prose alone: every page was
    clean while three measurement rows inside `sizing.py` named an agent
    scratch directory, in a module that is inside every wheel.
    """

    hits = _hits(shipped_scope(), _SHIPPED_PROCESS_REFERENCE)
    assert hits == [], (
        "text that ships refers to how the work was organised rather than "
        "to what the software does:" + chr(10) + "  " + (chr(10) + "  ").join(hits))


def test_no_shipped_file_names_a_private_machine() -> None:
    """A private hostname in anything that reaches a wheel or an sdist.

    THE BREAKAGE THIS PREVENTS, measured on the published 0.1.1 wheel: six
    comment lines in the carried physics (`core/gf.py`, `kernels/gf.cu`,
    `kernels/ntiedtke.cu`, `kernels/ysu.cu`)
    and the two shipped receipts named the maintainer's machines, which no
    reader can reach.  The carried files are cleaned by
    `tools/resync_from_owner.CORE_IDENTITY` at the cut; this is the proof
    that the rule ran, on the tree and on the npz members.
    """

    hits = _hits(shipped_scope(), _MACHINE_NAME)
    assert hits == [], (
        "a file that reaches the artefact names one of the maintainer's "
        "machines.  Name the card or the operating system instead:\n  "
        + "\n  ".join(hits))


def test_no_shipped_file_names_the_private_case() -> None:
    """A private case's working name in generic shipped text.

    THE BREAKAGE THIS PREVENTS, measured on the published 0.1.1 wheel:
    eight docstring lines across `core/landuse.py`, `core/npref.py`,
    `core/physics.py` and `core/rrtmgp.py` named it.  The
    case-token gate reads code positions only and skips docstrings on
    purpose, so nothing else in this repository saw them.
    """

    scope = [p for p in shipped_scope() if p.resolve() != _CASE_TOKEN_DETECTOR]
    hits = _hits(scope, _PRIVATE_CASE)
    assert hits == [], (
        "a file that reaches the artefact names a private case.  Say what "
        "the reference configuration is instead:\n  " + "\n  ".join(hits))


def test_published_prose_names_no_machine() -> None:
    hits = _hits(prose_scope(), _MACHINE_NAME)
    assert hits == [], (
        "published prose names one of the maintainer's machines by "
        "hostname.  A reader cannot reach it and does not need it; name "
        "the card or the operating system instead:\n  "
        + "\n  ".join(hits)
    )


def test_no_published_link_names_a_repository_a_reader_cannot_open() -> None:
    """Every GitHub link in the published pages names a repository that exists.

    THE BREAKAGE THIS NAMES.  Three links in the door pages sent the reader to
    a repository named after the PRIVATE tree.  That is not a stale link: it
    is a 404 that also tells every reader a repository by that name exists and
    that they cannot see it.  The engine's public repository is `arwen`, which
    is what its own wheel metadata states, and that is where those pages live.

    The allowed set is small and stated, rather than the pattern being a guess
    at what looks private: the repositories this family publishes are named
    here, and a link to anything else has to be argued for by adding it.

    THE SECOND SET IS SOMEBODY ELSE'S, and it exists because the rule above
    reads every link as one of ours.  A third-party notice has to say where
    the transcribed work was published, by name and by URL, or the notice
    does not identify what it is a notice for.  Those repositories are
    public and a reader opens them; the failure this gate exists to prevent,
    a 404 that advertises a private repository of ours, cannot happen with
    one.  Each is listed with the section that cites it, so an upstream
    arriving here has to be an upstream somebody wrote a notice for.
    """

    allowed = {
        "FahrenheitResearch/arwen",
        # WOOF's own repository, which its pages link now
        "recastsystems/woof",
        "FahrenheitResearch/gpuwm-global",
        "FahrenheitResearch/gpuwm-hex",
        # NOTICE, "FP32 libm transcriptions -- Arm optimized-routines":
        # where Arm published the logf/expf/exp2f/powf cores, which MIT
        # requires this distribution to identify and attribute.
        "ARM-software/optimized-routines",
    }
    link = re.compile(r"github[.]com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")
    hits = []
    for path in prose_scope():
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in link.finditer(text):
            name = match.group(1)
            if name in allowed:
                continue
            number = text[:match.start()].count(chr(10)) + 1
            hits.append(f"{path.relative_to(ROOT).as_posix()}:{number}: {name}")
    assert hits == [], (
        "published prose links to a repository this family does not publish; "
        "a reader gets a 404 and the name of something they cannot see:"
        + chr(10) + "  " + (chr(10) + "  ").join(hits)
    )


# ---------------------------------------------------------------------------
# the gate has to be able to see
# ---------------------------------------------------------------------------
def test_both_scopes_actually_resolve_to_files() -> None:
    """A gate that walks an empty file list reports green and proves nothing.

    THE BREAKAGE THIS PREVENTS: the globs above are relative to a resolved
    repository root, and every one of them silently yields nothing if that
    root moves, if the tests are run from an installed copy, or if a
    directory is renamed.  Every test in this file would then pass, which
    is the failure mode a provenance gate can least afford.
    """

    prose = prose_scope()
    shipped = shipped_scope()
    assert len(prose) >= 2, (
        f"the prose scope resolved to {len(prose)} files under {ROOT}; at "
        "minimum the declaration and the workflows are there")
    assert len(shipped) >= 100, (
        f"the shipped scope resolved to {len(shipped)} files under {ROOT}; "
        "this package alone carries over a hundred modules")
    assert any(path.name == "pyproject.toml" for path in prose)
    assert any("src" in path.parts for path in shipped)
    # NAMED, not counted: the carve put the physics that runs on the card
    # into `src/` as device sources, and they were outside this gate until
    # they were measured to be carrying what it exists to stop.  A glob that
    # silently stops matching them would restore exactly that state.
    kernels = [path for path in shipped if path.suffix in (".cu", ".cuh")]
    assert len(kernels) >= 15, (
        f"the shipped scope resolved to {len(kernels)} device sources; the "
        "carried physics alone ships fifteen")
    assert any(path.name == "gf.cu" for path in kernels), sorted(
        path.name for path in kernels)
    assert any(path.suffix == ".md" and "src" in path.parts
               for path in prose), (
        "the shipped markdown that travels as package data is outside the "
        "prose scope; the Noah table page is the one this was measured on")
    experiments = [path for path in shipped
                   if path.suffix == ".toml" and "src" in path.parts]
    assert len(experiments) >= 50, (
        f"the shipped scope resolved to {len(experiments)} experiment files "
        "under src/; the package ships fifty-five")
    # THE ARCHIVE IS OPENED, and this is the assertion that says so.  A
    # scope that lists an npz and a reader that hands back nothing report
    # exactly what a clean table reports, which is the state this widening
    # was written out of: the leak sat in a member no text tool opened.
    # NAMED rather than counted: the covariance table is the one the
    # measurement was taken on, and its metadata member is where the paths
    # were.
    archives = [path for path in shipped if path.suffix == ".npz"]
    assert archives, (
        "the shipped scope resolved to no stored archives; the covariance "
        "table is package data and its receipt lives inside it")
    members = [(label, text) for label, text in _documents(archives)]
    assert any("__metadata__" in label for label, _ in members), sorted(
        label for label, _ in members)
    metadata = next(text for label, text in members if "__metadata__" in label)
    assert "arwen-global-static-covariance" in metadata, metadata[:200]
    assert len(metadata) > 10_000, len(metadata)
    # ...and the numeric members are NOT read as text, because a rule that
    # fires inside a covariance matrix is an instrument nobody will keep.
    assert len(members) == 1, sorted(label for label, _ in members)


def test_the_patterns_still_match_what_they_are_written_to_match() -> None:
    """Test the tester, in the direction that fails silently.

    A regular expression that has stopped matching is indistinguishable
    from a clean tree, and five instruments in this family gave confident
    wrong answers in one night.  Each pattern is exercised here against a
    string it must catch and one it must not.

    THE STRINGS ARE SHAPES, NOT THE THINGS THEMSELVES.  This file ships
    inside the sdist, so a rule tested against the real hostname or the
    real directory it was written to refuse would publish exactly what it
    screens for.  MEASURED 2026-09-10: the first draft of the three rules
    below did that, and the artefact scan found the name in the test that
    exists to stop the name.
    """

    assert _TOOLING.search("Co-Authored-By: someone")
    assert _TOOLING.search("docs/superpowers/plans/2026-07-14-x.md")
    assert _TOOLING.search(".superpowers/sdd/a-root-cause.md")
    assert not _TOOLING.search("the agent-based ensemble")
    assert not _TOOLING.search("the super-cooled liquid powers the rime")
    assert _PERSONAL_NAME.search("the name " + _NAME + " appears here")
    assert not _PERSONAL_NAME.search("the pack drew energy upward")
    assert _MACHINE_LOCATION.search("/home/user/runs")
    assert _MACHINE_LOCATION.search("C:" + chr(92) + "Users" + chr(92) + "x")
    assert _MACHINE_LOCATION.search("the 25 GbE at 10.250.0.4")
    assert not _MACHINE_LOCATION.search("the 10.25 micron window")
    assert _BANNED_WORDS.search("an honest answer")
    assert _BANNED_WORDS.search("a load-bearing wall")
    assert not _BANNED_WORDS.search("the load balancer")
    # The underscore case: a word character on the left, so `` finds no
    # boundary and a function name carrying the word walked past this gate.
    assert _BANNED_WORDS.search("def test_the_honest_full_stop() -> None:")
    assert _BANNED_WORDS.search("a load_bearing_wall variable")
    assert _EM_DASH.search("a clause \u2014 and another")
    assert not _EM_DASH.search("a clause - and another")
    assert _MACHINE_NAME.search("measured on " + "node" + "-2")
    assert not _MACHINE_NAME.search("the node count")
    assert _DEVELOPMENT_ORGANISATION.search("Lane `lane/da-ensemble` (2026-09-06)")
    assert _DEVELOPMENT_ORGANISATION.search("a tree at or after integration/2.7.0")
    assert _DEVELOPMENT_ORGANISATION.search("charts under Downloads/evidence-gallery/x")
    assert _DEVELOPMENT_ORGANISATION.search("recomputed by the refuter")
    assert _DEVELOPMENT_ORGANISATION.search("0.670 (the owner tip)")
    assert _DEVELOPMENT_ORGANISATION.search("recorded in the lane report")
    # The word "lane" itself is ordinary technical English in this model
    # (a forcing lane, a momentum lane) and is deliberately not matched:
    # only the branch spelling is.
    assert not _DEVELOPMENT_ORGANISATION.search("the advective forcing lanes")
    assert not _DEVELOPMENT_ORGANISATION.search("the airplane/glider case")
    assert not _DEVELOPMENT_ORGANISATION.search("the reader downloads the list")
    assert _LANE_VOCABULARY.search("measured by the scorecard lane")
    assert _LANE_VOCABULARY.search("the docs lane's own tally")
    assert not _LANE_VOCABULARY.search("Lane names are capitalised here")
    assert not _LANE_VOCABULARY.search("the advective forcing lanes")
    # The directory shapes, which are the seventeen sites measured inside
    # the wheel on 2026-09-10.
    assert _DEVELOPMENT_ORGANISATION.search("the 32 GB host probe-lane5/out/runs/t383")
    assert _DEVELOPMENT_ORGANISATION.search("probe-lane3/results/node2/t533-b1")
    assert _DEVELOPMENT_ORGANISATION.search("the 16 GB host probe-merge/out/t383day/run")
    assert _DEVELOPMENT_ORGANISATION.search("probe-physics/out/capacity/t533_bare")
    assert _DEVELOPMENT_ORGANISATION.search("that config (work/probe-physics/t255-gate.toml)")
    assert _DEVELOPMENT_ORGANISATION.search("the legs are at work/probe-lane5/refutation/")
    assert _DEVELOPMENT_ORGANISATION.search("filed under refutation/two-legs")
    # ...and the ordinary prose the same words appear in.  A hyphenated
    # word in a sentence, a package path, and the words on their own are
    # all left alone: the separator is what makes it a directory.
    assert not _DEVELOPMENT_ORGANISATION.search("a two-lane road")
    assert not _DEVELOPMENT_ORGANISATION.search("the merge/split pair of tests")
    assert not _DEVELOPMENT_ORGANISATION.search("woof/globe/data/noah_tables/")
    assert not _DEVELOPMENT_ORGANISATION.search("the refutation of the claim")
    assert not _DEVELOPMENT_ORGANISATION.search("configs/verify/out")
    assert not _DEVELOPMENT_ORGANISATION.search("the semi-lagrangian core")
    # The working root in the spelling that walked through, measured
    # 2026-09-10: the alternative required a hyphenated directory under it
    # and the site was an underscored file name sitting directly under it.
    assert _DEVELOPMENT_ORGANISATION.search("(work/probe_big_ensemble.py, 2026-08-16)")
    assert _DEVELOPMENT_ORGANISATION.search("kept at work/t255-gate.toml")
    assert not _DEVELOPMENT_ORGANISATION.search("the .github/workflows page")
    assert not _DEVELOPMENT_ORGANISATION.search("the framework/library split")
    # The planning document and its review, in the shapes the seven sites
    # measured on the 0.1.2 source tree had (the labels here are made up).
    assert _DEVELOPMENT_ORGANISATION.search(
        "measured on the published wheel (the design plan audit, finding Q7)")
    assert _DEVELOPMENT_ORGANISATION.search("with finding Q7 of the plan's audit")
    assert _DEVELOPMENT_ORGANISATION.search("the plan's audit measured them")
    assert _DEVELOPMENT_ORGANISATION.search("(the design plan\n    audit)")
    assert _DEVELOPMENT_ORGANISATION.search("(findings Q2 and Q3)")
    assert _DEVELOPMENT_ORGANISATION.search("per finding\nB12")
    # ...and the same words where they are ordinary: a flight plan, an audit
    # of the arrays, and the finding of a result.
    assert not _DEVELOPMENT_ORGANISATION.search("the flight plan for the run")
    assert not _DEVELOPMENT_ORGANISATION.search("an audit of every array")
    assert not _DEVELOPMENT_ORGANISATION.search("the finding that T255 fits")
    assert not _DEVELOPMENT_ORGANISATION.search("finding 3 of 5 rows")
    # The tilde spelling of a home directory: what the reader is sent to on
    # their own machine passes, a working directory on a node does not.
    assert _ALLOWED_HOME_TARGET.match("WPS_GEOG")
    assert _ALLOWED_HOME_TARGET.match(".woof")
    assert _ALLOWED_HOME_TARGET.match(".cache")
    assert not _ALLOWED_HOME_TARGET.match("r7f3a1c2")
    assert _HOME_PATH.search("a copy at ~/r7f3a1c2/x.py").group(1) == "r7f3a1c2"
    assert _HOME_PATH.search("under ~/WPS_GEOG").group(1) == "WPS_GEOG"
    assert not _HOME_PATH.search("about 7~8 K of spread")
    # A directory of tooling on another tree, as a path and as a bare word,
    # against the two library names that end the same way.
    assert _PRIVATE_TOOLING_DIRECTORY.search("the chain's chaintools copy")
    assert _PRIVATE_TOOLING_DIRECTORY.search("chaintools/summarize.py")
    assert not _PRIVATE_TOOLING_DIRECTORY.search("import functools")
    assert not _PRIVATE_TOOLING_DIRECTORY.search("from setuptools import setup")
    assert not _PRIVATE_TOOLING_DIRECTORY.search("under tools/ in this repository")
    assert not _PRIVATE_TOOLING_DIRECTORY.search("the array's subscripts")
    # Hostnames: the two default shapes, and the words they are built from
    # used ordinarily.
    assert _HOSTNAME.search('"host": "DESKTOP-A1B2C3D4"')
    assert _HOSTNAME.search("measured on WEATHER-PC")
    assert not _HOSTNAME.search("the desktop it was measured on")
    assert not _HOSTNAME.search("over PCIe, not NVLINK")
    assert not _HOSTNAME.search("a PCI address")
    # The host field: a class passes, a name does not, and since 0.1.2 that
    # includes the node names.
    assert _ALLOWED_HOST_VALUE.match("windows-amd64")
    assert _ALLOWED_HOST_VALUE.match("linux-x86_64")
    assert not _ALLOWED_HOST_VALUE.match("weather-node" + "-4")
    assert not _ALLOWED_HOST_VALUE.match("node" + "-2")
    assert _PRIVATE_CASE.search("matching " + "REAL" + "74.")
    assert not _PRIVATE_CASE.search("the reference WRF configuration")
    assert not _ALLOWED_HOST_VALUE.match("DESKTOP-A1B2C3D4")
    assert not _ALLOWED_HOST_VALUE.match("windows-11-0.0.00000-sp0")
    assert not _ALLOWED_HOST_VALUE.match("someones-laptop")
    assert _HOST_FIELD.search('  "host": "windows-amd64",').group(1) == "windows-amd64"
    # The archive reader, in the direction that fails silently: a member
    # that carries text must come back as that text, and a numeric member
    # must not come back as anything at all.
    assert _decode_member("<U6", "hello".encode("utf-32-le")) == "hello"
    assert _decode_member("|S8", b"hello" + bytes(3)) == "hello"
    assert _decode_member("<f8", bytes(8)) is None
    assert _decode_member("<i4", bytes(4)) is None
    assert "leak" in (_decode_member(None, "leak".encode("utf-16-le")) or "")


def test_the_third_party_texts_are_exempt_by_name() -> None:
    """An exemption naming a file that is not there exempts nothing.

    The list is what keeps a licence text out of the house-style rules, so
    a stale entry is an exemption a future file could inherit by being
    given the same name, and a missing file is a licence this distribution
    stopped shipping while NOTICE still points at it.  Both are caught
    here rather than at a reader.
    """

    for name in THIRD_PARTY_VERBATIM:
        path = ROOT / name
        assert path.is_file(), f"exempt by name and not in the tree: {name}"
        assert path.read_bytes().strip(), f"exempt and empty: {name}"
        assert path not in shipped_scope(), name
        assert path not in prose_scope(), name


def test_the_declaration_is_inside_the_prose_scope() -> None:
    """The declaration is published text and is held to the prose rules.

    It is asserted here rather than left to a glob because it is the one
    published file that is easy to mistake for configuration: it carries
    the description PyPI renders and the comments a packager reads.
    """

    assert (ROOT / "pyproject.toml") in prose_scope()
