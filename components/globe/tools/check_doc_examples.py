"""Check that every documented ``woof global`` line is a line the parser accepts.

A page that teaches a command the parser refuses teaches a failure: the reader
types it, argparse exits 2, and the page is the reason.  This tool reads the
markdown, pulls every ``woof global`` invocation out of it (fenced blocks and
inline code spans alike), and checks each one against the parser this package
ships:

* the subcommand chain exists (``da fresh`` is two tokens, ``fetch-doors`` is
  one);
* every option the line uses is declared on that subcommand;
* the number of positional arguments is between the parser's minimum and its
  maximum, and every required option is present.

It does NOT run anything and does not touch the filesystem, so an example
naming an output directory that does not exist yet is still checked.  A line
carrying a placeholder (``<config>``, ``TAPE``, ``...``) is reported as a
template and not failed: a placeholder is not a command line.

Exit 1 when a line would be refused, which is the form CI runs::

    python tools/check_doc_examples.py            # README.md and docs/*.md
    python tools/check_doc_examples.py README.md  # one page
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shlex
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROG = "woof global"

#: A token standing for something the reader supplies.  A line carrying one is
#: a template, not a command line, and is counted separately.
PLACEHOLDER = re.compile(r"[<>]|\.\.\.|^[A-Z][A-Z0-9_]{2,}$")

FENCE = re.compile(r"^```")
INLINE = re.compile(r"`([^`\n]+)`")


def _invocations(text: str) -> list[tuple[int, str, bool]]:
    """Every ``woof global`` line in one document: line number, text, is-fenced.

    A fenced line is a command line a reader copies and is checked whole.  An
    inline span is a MENTION of a command, usually a fragment naming one flag,
    so only its command chain and its option spellings are checked.
    """

    found: list[tuple[int, str, bool]] = []
    lines = text.splitlines()
    in_fence = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if FENCE.match(line.strip()):
            in_fence = not in_fence
            index += 1
            continue
        if in_fence:
            stripped = line.strip()
            if stripped.startswith("$ "):
                stripped = stripped[2:]
            if stripped.startswith(PROG + " ") or stripped == PROG:
                joined = stripped
                start = index
                while joined.rstrip().endswith("\\") and index + 1 < len(lines):
                    index += 1
                    joined = joined.rstrip()[:-1] + " " + lines[index].strip()
                found.append((start + 1, joined, True))
        else:
            for span in INLINE.findall(line):
                candidate = span.strip()
                if candidate.startswith(PROG + " ") or candidate == PROG:
                    found.append((index + 1, candidate, False))
        index += 1
    return found


def _subparsers(parser: argparse.ArgumentParser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _resolve(parser: argparse.ArgumentParser, tokens: list[str]):
    """Walk the subcommand chain; return (parser, remaining tokens, chain, error)."""

    chain: list[str] = []
    current = parser
    while tokens:
        action = _subparsers(current)
        if action is None:
            break
        head = tokens[0]
        if head.startswith("-"):
            break
        if head not in action.choices:
            if chain:
                return current, tokens, chain, (
                    f"{' '.join(chain)} has no subcommand {head!r}")
            return current, tokens, chain, f"no command {head!r}"
        current = action.choices[head]
        chain.append(head)
        tokens = tokens[1:]
    return current, tokens, chain, None


def _option_map(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    table: dict[str, argparse.Action] = {}
    for action in parser._actions:
        for spelling in action.option_strings:
            table[spelling] = action
    return table


def _takes_a_value(action: argparse.Action) -> bool:
    if action.nargs == 0:
        return False
    return not isinstance(action, (
        argparse._StoreTrueAction, argparse._StoreFalseAction,
        argparse._CountAction, argparse._HelpAction, argparse._VersionAction))


def _positional_bounds(parser: argparse.ArgumentParser) -> tuple[int, float]:
    low = 0
    high: float = 0
    for action in parser._actions:
        if action.option_strings:
            continue
        if isinstance(action, argparse._SubParsersAction):
            continue
        nargs = action.nargs
        if nargs is None:
            low, high = low + 1, high + 1
        elif nargs == "?":
            high += 1
        elif nargs in ("*", argparse.REMAINDER):
            high = float("inf")
        elif nargs == "+":
            low, high = low + 1, float("inf")
        elif isinstance(nargs, int):
            low, high = low + nargs, high + nargs
    return low, high


#: Commands that take a REMAINDER and hand it to a door with a parser of its
#: own.  This checker reads ONE parser; for these the outer parser knows the
#: command name and nothing after it, so an option of the inner door reads as
#: an option that does not exist.  Listing them here is narrower than
#: switching the checker off for a page.
FORWARDING_COMMANDS = frozenset({"microwave"})


def check_line(parser: argparse.ArgumentParser, line: str,
               *, whole: bool = True) -> tuple[str, str]:
    """Return ``(verdict, detail)``; the verdict is ``ok``, ``template`` or ``FAIL``.

    ``whole`` is False for an inline mention: the command chain and the option
    spellings are still checked, the argument count is not.
    """

    try:
        tokens = shlex.split(line, comments=True)
    except ValueError as exc:
        return "FAIL", f"cannot be tokenized: {exc}"
    tokens = tokens[1:]
    if any(PLACEHOLDER.search(token) for token in tokens):
        return "template", "carries a placeholder"
    if not tokens:
        return "ok", "the bare prog name"

    target, rest, chain, error = _resolve(parser, tokens)
    if error:
        return "FAIL", error
    where = " ".join([PROG, *chain])
    if chain and chain[0] in FORWARDING_COMMANDS:
        # A forwarding door owns its own parser.  Its outer command declares a
        # single REMAINDER positional and hands everything after it to the
        # module's parser, so reading the outer parser for option spellings
        # answers "no such option" for every one of the inner door's real
        # options.  Checked by running it: `microwave calibrate --out FILE`
        # exits 0 on `--help` and is refused only for the reasons the inner
        # parser gives.  The chain is checked; the tail is the inner door's.
        return "ok", where
    options = _option_map(target)
    positionals = 0
    index = 0
    while index < len(rest):
        token = rest[index]
        if token == "--":
            positionals += len(rest) - index - 1
            break
        if token.startswith("--"):
            name, equals, _ = token.partition("=")
            action = options.get(name)
            if action is None:
                return "FAIL", f"{where} has no option {name}"
            if not equals and _takes_a_value(action):
                index += action.nargs if isinstance(action.nargs, int) else 1
        elif token.startswith("-") and len(token) > 1:
            action = options.get(token)
            if action is None:
                return "FAIL", f"{where} has no option {token}"
            if _takes_a_value(action):
                index += action.nargs if isinstance(action.nargs, int) else 1
        else:
            positionals += 1
        index += 1

    if not whole or "--help" in rest or "-h" in rest:
        return "ok", where
    low, high = _positional_bounds(target)
    if positionals < low:
        missing = [action.dest for action in target._actions
                   if not action.option_strings
                   and not isinstance(action, argparse._SubParsersAction)]
        return "FAIL", (f"{where} needs {low} positional argument(s) "
                        f"({', '.join(missing)}) and the line gives {positionals}")
    if positionals > high:
        return "FAIL", (f"{where} takes at most {high} positional argument(s), "
                        f"the line gives {positionals}")
    for action in target._actions:
        if action.option_strings and action.required:
            if not any(token.split("=")[0] in action.option_strings for token in rest):
                return "FAIL", f"{where} requires {action.option_strings[0]} and the line omits it"
    stale = _shipped_experiment_written_as_a_path(rest)
    if stale:
        written, name = stale
        return "FAIL", (
            f"{where} names {written}, which is a shipped experiment written "
            f"as a path into a checkout: from an installed wheel that path "
            f"does not exist and the parser exits 2.  Write it as {name}")
    return "ok", where


def _shipped_experiment_written_as_a_path(tokens) -> tuple[str, str] | None:
    """A shipped experiment named with a directory or a `.toml`, or None.

    THE BREAKAGE THIS CATCHES, and it was on the quickstart page when this
    check was written: seven command lines read
    `woof global run configs/arwen_global_t255_quickstart.toml`.  Every one
    of them is a line the parser ACCEPTS in shape, which is all this tool
    used to ask, and every one of them exits 2 from an installed wheel,
    because `configs/` is a directory in a checkout the reader does not
    have.  The pages were written before the carve, when that path was the
    only way to name an experiment.

    The rule is deliberately narrow: only a token whose stem is a name this
    package actually ships is reported.  A reader's own `my-run.toml` is
    left alone, because a page may legitimately show a file the reader is
    expected to create.
    """

    try:
        from woof.globe.configs_dir import list_configs
    except Exception:                                   # pragma: no cover
        return None
    shipped = set(list_configs())
    if not shipped:
        return None
    for token in tokens:
        if token.startswith("-"):
            continue
        cleaned = token.replace("\\", "/")
        stem = cleaned.rsplit("/", 1)[-1]
        if stem.endswith(".toml"):
            stem = stem[: -len(".toml")]
        if stem not in shipped:
            continue
        if cleaned != stem:
            return token, stem
    return None


def main(argv: list[str] | None = None) -> int:
    from woof.globe.cli import build_parser

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        pages = [pathlib.Path(name) for name in argv]
    else:
        pages = [REPO_ROOT / "README.md",
                 *(page for page in sorted((REPO_ROOT / "docs").glob("*.md"))
                   if page.name != "CLI-REFERENCE.md")]

    parser = build_parser()
    failures = 0
    checked = 0
    templates = 0
    for page in pages:
        if not page.is_file():
            print(f"{page}: not a file")
            failures += 1
            continue
        for number, line, whole in _invocations(page.read_text(encoding="utf-8")):
            verdict, detail = check_line(parser, line, whole=whole)
            if verdict == "FAIL":
                failures += 1
                print(f"{page}:{number}: {detail}\n    {line}")
            elif verdict == "template":
                templates += 1
            else:
                checked += 1
    print(f"{checked} example(s) accepted by the parser, {templates} template(s), "
          f"{failures} refused")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
