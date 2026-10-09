"""One door registry and one command reader, for every docs-to-code binding.

A page that tells a reader to RUN something makes a claim the parser is the
authority on, and this module is the mechanism that holds the two together:
the registry of doors, the reader that decides whether a fragment of a page
is a command at all, and the option set a door defines.

THE REGISTRY IS READ OFF THE PARSERS, never listed.  A hand-written list of
subcommands is a list that goes stale the first time a door is renamed, and
it goes stale silently: the page keeps naming a door that no longer exists
and the test keeps passing because its own list still has it.  Both programs
are asked for their own parser and descended:

* ``woof global`` and its subcommands, from ``woof.globe.cli``, and
* ``woof`` and its subcommands, from the INSTALLED engine's ``woof.cli``.

The engine's doors are in the registry because the documented route crosses
both programs -- a reader fetches with ``woof fetch`` and renders with
``woof render`` -- and a route that is only checked on one side is a route
whose first and last steps nobody grades.  When the engine cannot be
imported the registry holds this package's doors alone and says so by
returning them; a consumer that needs an engine door skips.
"""
from __future__ import annotations

import argparse
import re

#: A shell prompt a page may print before the command itself.
PROMPT = re.compile(r"^\s*(?:\$|>|PS[^>]*>)\s+")

#: The two programs a page in this repository may tell a reader to run.
#: Longest first: `woof global` must not be read as `woof` with a stray
#: argument, and sorting by length is what makes that true without a rule.
PROGRAMS = ("woof global", "woof")


def _descend(prefix: str, parser: argparse.ArgumentParser, out: dict) -> None:
    """Register a parser and every subcommand tree under it."""

    out.setdefault(prefix, parser)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                _descend(f"{prefix} {name}", sub, out)


def doors() -> dict:
    """Every documented door, name -> the parser that owns its options."""

    out: dict = {}
    from woof.globe.cli import build_parser

    _descend("woof global", build_parser(), out)
    try:
        from woof.cli import build_parser as engine_build_parser
    except Exception:
        return out
    try:
        _descend("woof", engine_build_parser(), out)
    except Exception:
        pass
    return out


def door_options(parser) -> set[str]:
    """Every long option a parser defines, ``--help`` included."""

    out: set[str] = set()
    for action in parser._actions:
        out.update(o for o in action.option_strings if o.startswith("--"))
    return out


def code_fragments(text: str):
    """``(lineno, fragment)`` for every fenced line and inline span."""

    inline = re.compile(r"`([^`\n]+)`")
    fence = re.compile(r"^\s*(?:```|~~~)")
    inside = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if fence.match(line):
            inside = not inside
            continue
        if inside:
            yield lineno, line
        else:
            for match in inline.finditer(line):
                yield lineno, match.group(1)


def resolve_door(body: str, known: dict):
    """The door a fragment invokes, or ``None`` if it is not one.

    A fragment counts as an invocation only when it STARTS with a program
    name, after an optional shell prompt.  Prose that merely contains the
    program name is not a command and is not checked, which is why this rule
    needs no allowlist of English words.

    The LONGEST known door wins, not the first token.  ``woof global da``
    owns a subcommand tree, and stopping at the first token resolved
    ``woof global da cycle`` to the ``da`` parser, which defines none of
    its flags.  The descent only follows tokens the registry knows, so an
    argument that happens to be a bare word still resolves to the door that
    takes it.  An UNKNOWN first token is returned unchanged rather than
    swallowed: a page naming a subcommand no parser has is exactly what the
    consumer exists to report.
    """

    body = PROMPT.sub("", body).strip()
    for program in PROGRAMS:
        if body != program and not body.startswith(program + " "):
            continue
        rest = body[len(program):].strip()
        words = rest.split()
        token = words[0] if words else ""
        # A placeholder (`woof <command>`, `woof SUBCOMMAND`) is not a
        # claim that a subcommand exists.
        if not token or not re.fullmatch(r"[a-z][a-z0-9-]*", token):
            return None
        door = f"{program} {token}"
        for word in (words[1:] if door in known else []):
            if not re.fullmatch(r"[a-z][a-z0-9-]*", word):
                break
            if f"{door} {word}" not in known:
                break
            door = f"{door} {word}"
        return door, body
    return None
