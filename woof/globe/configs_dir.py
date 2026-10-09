"""The experiment TOMLs that ship inside the wheel, and how a name resolves.

Fifty-five configurations travel with this package: the fifty-two
``arwen_global_*`` experiments and the three ``global_spectral_*`` dry-core
cases.  The count is stated here and nowhere else that a reader meets it;
`woof global doctor` and `woof global configs` read the directory.  Before the carve they lived in a repository directory, so every
command in every document began with a relative path into a checkout the
reader did not have.

Here they are package data, and a bare name resolves to the shipped file:

    woof global run arwen_global_gdas_t255_native_24h --outdir out/day

A path that exists still wins, always, so a reader who edits a copy runs
their copy.  ``woof global configs`` lists what shipped.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["config_root", "list_configs", "resolve_config", "config_argument"]

#: The suffix a shipped experiment carries.
SUFFIX = ".toml"


def config_root() -> Path:
    """The directory inside the installed package holding the experiments."""

    return Path(__file__).resolve().parent / "configs"


def list_configs() -> tuple[str, ...]:
    """Every shipped experiment name, without the suffix, sorted."""

    root = config_root()
    if not root.is_dir():
        return ()
    return tuple(sorted(path.stem for path in root.glob(f"*{SUFFIX}")))


def resolve_config(value: str | Path) -> Path:
    """A path the caller gave, or the shipped experiment of that name.

    A file on disk wins over a shipped name of the same spelling: a reader who
    copies a config next to their run directory and edits it must get their
    edit, not the wheel's copy.  Only when the path does not exist is the
    shipped set consulted, and a miss there names the shipped set rather than
    printing the path back.
    """

    path = Path(value)
    if path.exists():
        return path
    stem = path.name[:-len(SUFFIX)] if path.name.endswith(SUFFIX) else path.name
    bare = path.parent in (Path("."), Path(""))
    if bare:
        candidate = config_root() / f"{stem}{SUFFIX}"
        if candidate.is_file():
            return candidate
    shipped = list_configs()
    if not shipped:
        raise FileNotFoundError(
            f"{path} does not exist, and this install ships no experiment configs "
            f"(expected them under {config_root()})")

    # A PATH that does not exist whose STEM does.  This is what every command
    # line on a page written before the carve looks like:
    # `configs/arwen_global_moist_smoke.toml`, a relative path into a
    # checkout the reader of a wheel does not have.  The refusal used to say
    # "no shipped experiment is named 'arwen_global_moist_smoke'" and then,
    # from a suggestion list computed without the directory test, "did you
    # mean: arwen_global_moist_smoke" -- the same string, denied and offered
    # in consecutive lines.
    #
    # It still refuses rather than resolving.  A reader who typed a directory
    # meant a file in that directory, and running the wheel's copy of a
    # same-named experiment instead would run a configuration they did not
    # ask for and cannot see.  What changes is that the sentence now names
    # the one edit that runs: drop the directory.
    if not bare and (config_root() / f"{stem}{SUFFIX}").is_file():
        raise FileNotFoundError(
            f"{path} does not exist here.  A shipped experiment named "
            f"{stem!r} does: run it as `{stem}` with no directory and no "
            f"{SUFFIX}, from any directory.  The path is not resolved to it "
            "because a path names a file you can read and edit, and this "
            "one would silently become the copy inside the wheel.")

    near = [name for name in shipped if stem and stem in name]
    suggestion = ("\n  did you mean: " + ", ".join(near[:6])) if near else ""
    raise FileNotFoundError(
        f"{path} does not exist, and no shipped experiment is named {stem!r}.  "
        f"{len(shipped)} experiments ship with this package; "
        f"`woof global configs` lists them.{suggestion}")


def config_argument(value: str) -> Path:
    """argparse ``type=`` for every config argument.

    Resolution happens at parse time so a mistyped experiment name is an
    argument error with the shipped list beside it, rather than a
    FileNotFoundError several seconds into a run.
    """

    try:
        return resolve_config(value)
    except FileNotFoundError as exc:
        import argparse

        raise argparse.ArgumentTypeError(str(exc)) from None
