"""The shipped tree parses on the oldest Python the distribution declares.

THE BREAKAGE THIS NAMES.  `pyproject.toml` declares `requires-python =
">=3.11"`, and the continuous-integration matrix tests 3.11 and 3.12, but the
whole development of this package ran on 3.13 and 3.14.  Python 3.12 (PEP 701)
started accepting a quote inside an f-string that matches the quote around it,
and one line of `obs_scorecard.py` used exactly that: three quote levels in one
expression.  Every suite on 3.13 and 3.14 was green; on 3.11 the module did not
compile, so `woof global` was broken for anyone on the floor the metadata
promised.  A test that only runs on the interpreter it happens to be given
cannot see that, so this one asks for the floor interpreter by name.

Two arms, and the second exists because the first can be silent:

* `test_shipped_tree_compiles_on_the_declared_floor` finds a 3.11 interpreter
  (`py -3.11` on Windows, `python3.11` on a PATH) and runs `compileall` over
  the shipped package.  When no floor interpreter is installed it SKIPS BY
  NAME, printing what to install; the CI matrix's 3.11 job is the arm that
  never skips.
* `test_no_nested_same_quote_fstrings` needs no interpreter: it tokenises every
  shipped module with the running interpreter and refuses an f-string whose
  replacement field reuses the f-string's own quote character, which is the
  one PEP 701 construct this tree has produced.  It is a narrower net than the
  compiler, and it says so; it is here so the desktop suite catches the
  recurrence the day it is written rather than the day CI runs.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "arwen_global"
FLOOR = (3, 11)


def _floor_interpreter() -> list[str] | None:
    """A command that runs the floor interpreter, or None."""
    version = f"{FLOOR[0]}.{FLOOR[1]}"
    if sys.version_info[:2] == FLOOR:
        return [sys.executable]
    if sys.platform == "win32" and shutil.which("py"):
        probe = subprocess.run(["py", f"-{version}", "-c", "import sys; print(sys.version_info[:2])"],
                               capture_output=True, text=True)
        if probe.returncode == 0 and str(FLOOR) in probe.stdout:
            return ["py", f"-{version}"]
    candidate = shutil.which(f"python{version}")
    if candidate:
        return [candidate]
    return None


def test_the_declared_floor_is_the_one_this_test_checks():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'requires-python\s*=\s*">=(\d+)\.(\d+)"', text)
    assert match, "pyproject.toml declares no requires-python floor"
    assert (int(match.group(1)), int(match.group(2))) == FLOOR, (
        "the declared floor moved; move FLOOR in this test with it")


def test_shipped_tree_compiles_on_the_declared_floor():
    command = _floor_interpreter()
    if command is None:
        pytest.skip(f"no Python {FLOOR[0]}.{FLOOR[1]} interpreter on this host "
                    "(install one, or rely on the CI matrix's floor job)")
    result = subprocess.run(
        [*command, "-m", "compileall", "-q", "-f", str(SRC)],
        capture_output=True, text=True, cwd=str(ROOT))
    assert result.returncode == 0, (
        f"the shipped tree does not compile on Python {FLOOR[0]}.{FLOOR[1]}:\n"
        f"{result.stdout}\n{result.stderr}")


_SAME_QUOTE_FIELD = re.compile(r"""(?P<prefix>\b[fF][rR]?|\b[rR][fF])(?P<quote>['"])(?P<body>(?:[^\\\n]|\\.)*?)(?P=quote)""")


def _reuses_own_quote(body: str, quote: str) -> bool:
    depth = 0
    for char in body:
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        elif char == quote and depth > 0:
            return True
    return False


def test_no_nested_same_quote_fstrings():
    """A single-quoted f-string whose field contains a single quote (or the
    double-quoted mirror) is Python 3.12 syntax.  The regex stops at the
    first unescaped closing quote, so a field that reuses the quote is
    caught when the field's text survives the split; this is the shape the
    tree produced, and the compiler arm above is the complete check."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in _SAME_QUOTE_FIELD.finditer(text):
            if _reuses_own_quote(match.group("body"), match.group("quote")):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{line}")
    assert not offenders, "f-strings reusing their own quote inside a field (3.12 syntax):\n" + "\n".join(offenders)


def test_the_net_catches_the_shape_it_names():
    assert _reuses_own_quote("{k} {v['admitted']}", "'")
    assert not _reuses_own_quote("{k} {v[\"admitted\"]}", "'")
    assert not _reuses_own_quote("plain 'quote' outside a field", "'")
