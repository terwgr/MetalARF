"""Write ``docs/CLI-REFERENCE.md`` from the package's own argparse parser.

The complete option surface of every door, in one page, built by reading
the parser rather than by remembering to write a line.  Run it after
changing any option::

    python tools/build_cli_reference.py

``--check`` rewrites nothing and exits 1 if the committed page differs
from what the parser produces now, which is the form CI runs: the page
cannot fall behind the code and cannot name a flag the code dropped.

Two things are deliberately not scraped.  The prog name comes from the
parser itself, so a rename of the console script moves the whole page.
Help strings that interpolate a resolved filesystem path are rewritten to
a package-relative stand-in, because a page every reader shares must not
carry the path of the machine that generated it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "docs" / "CLI-REFERENCE.md"

HEADER = """# Command reference

Every door of `{prog}`, with every option, generated from the parser by
`tools/build_cli_reference.py`.  Nothing on this page is typed by hand; run
the tool after changing an option and commit what it writes.

`{prog} <command> --help` prints the same text at the terminal.

"""

NO_HELP = "_(the parser declares no help text for this option)_"


def _portable(text: str) -> str:
    """Replace an absolute path in a help string with a package-relative stand-in."""
    import woof.globe

    package = pathlib.Path(woof.globe.__file__).resolve().parent
    for root, stand_in in ((str(package), "<arwen_global package>"),
                           (str(REPO_ROOT), "<repository root>")):
        for spelling in (root, root.replace("\\", "/")):
            text = text.replace(spelling, stand_in)
    return text


def _cell(text: str) -> str:
    return _portable(" ".join(text.split())).replace("|", r"\|")


def _metavar(action: argparse.Action) -> str:
    if action.metavar:
        return str(action.metavar)
    if action.choices:
        return "{" + ",".join(str(c) for c in action.choices) + "}"
    if action.nargs == 0 or isinstance(action.const, bool):
        return ""
    return action.dest.upper()


def _spelling(action: argparse.Action) -> str:
    if action.option_strings:
        metavar = _metavar(action)
        joined = ", ".join(f"`{o}`" for o in action.option_strings)
        return f"{joined} `{metavar}`" if metavar else joined
    return f"`{action.dest}` (positional)"


def _rows(parser: argparse.ArgumentParser) -> list[tuple[str, str, str]]:
    rows = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            continue
        if isinstance(action, argparse._HelpAction):
            continue
        default = ""
        if action.default is not None and action.default is not argparse.SUPPRESS:
            if not isinstance(action.default, bool) or action.default:
                # A Path default prints with the separator of the machine that
                # ran this tool, so the same parser produced `out/arwen-global`
                # on Linux and `out\\arwen-global` on Windows.  The page is
                # published once and read on both, and the reader retypes what
                # it shows: posix separators, like every path in the help text.
                shown = action.default
                if isinstance(shown, pathlib.PurePath):
                    shown = shown.as_posix()
                default = f"`{shown}`"
        help_text = _cell(action.help) if action.help else NO_HELP
        rows.append((_spelling(action), help_text, default))
    return rows


def _subparsers(parser: argparse.ArgumentParser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            seen = set()
            for name, sub in action.choices.items():
                if id(sub) in seen:
                    continue
                seen.add(id(sub))
                yield name, sub


def _section(lines: list[str], name: str, parser: argparse.ArgumentParser,
             depth: int) -> None:
    lines.append(f"{'#' * min(depth, 6)} `{name}`")
    lines.append("")
    summary = parser.description or ""
    if summary:
        lines.append(_cell(summary))
        lines.append("")
    rows = _rows(parser)
    if rows:
        lines.append("| option | what it does | default |")
        lines.append("|---|---|---|")
        for spelling, help_text, default in rows:
            lines.append(f"| {spelling} | {help_text} | {default} |")
        lines.append("")
    for child_name, child in _subparsers(parser):
        _section(lines, f"{name} {child_name}", child, depth + 1)


def build() -> str:
    from woof.globe.cli import build_parser

    parser = build_parser()
    prog = parser.prog
    lines = [HEADER.format(prog=prog).rstrip(), ""]

    leaves = []
    for name, sub in _subparsers(parser):
        children = list(_subparsers(sub))
        if children:
            leaves.extend(f"{prog} {name} {child}" for child, _ in children)
        else:
            leaves.append(f"{prog} {name}")
    lines.append(f"**{len(leaves)} commands.**")
    lines.append("")
    for leaf in leaves:
        lines.append(f"- `{leaf}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    for name, sub in _subparsers(parser):
        _section(lines, f"{prog} {name}", sub, 2)

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--check", action="store_true",
                      help="compare the committed page against the parser and exit 1 on a difference")
    parsed = args.parse_args(argv)
    text = build()
    if parsed.check:
        if not PAGE.exists():
            print(f"{PAGE} does not exist; run this tool without --check", file=sys.stderr)
            return 1
        if PAGE.read_text(encoding="utf-8") != text:
            print(f"{PAGE} is behind the parser; run python tools/build_cli_reference.py",
                  file=sys.stderr)
            return 1
        print(f"{PAGE} agrees with the parser")
        return 0
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" and not the platform default: written from Windows
    # the page came back CRLF, so every regeneration on that machine rewrote
    # all 599 lines and the one changed option was invisible in the diff.  The
    # --check leg never caught it, because text-mode reading translates the
    # line endings back before it compares.
    PAGE.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {PAGE} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
