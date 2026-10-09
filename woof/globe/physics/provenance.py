"""Which physics module actually integrated, recorded as it happens.

THE BREAKAGE THIS PREVENTS.  This package now carries most of the physics it
runs (`woof.globe.core`) while the rest of what it calls stays on the
installed engine.  Both copies of several modules exist in one process: the
carried `sfclay`, `ysu` and `physics_inventory` this model calls, and the
engine's, which the engine's own regional drivers import.  A reader of a
receipt asking "whose physics produced these numbers" cannot answer it from a
version number, because the version names the ENGINE and the answer may be
this package; and cannot answer it from this package's version either,
because a `modules` override or a stale install can put the engine's module
in the slot.

So it is recorded rather than inferred: every physics module the runtime
actually resolves is noted at the moment it is resolved, with its dotted name,
whether it came from this package or the engine, and the SHA-256 of the file
that was imported.  The kernels get one digest over the whole carried
directory, because the device sources are half of what integrated and a
Python file's hash says nothing about them.

WHAT THIS IS NOT.  It is not the kernel manifest, which pins each compiled
image's assembled source and is the engine's mechanism.  It is not the config
identity: nothing here joins `NativePhysicsOptions.identity`, because that
hash is every checkpoint's restart admission and a provenance record must
never move it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType

__all__ = ["note", "integrated_physics_modules", "reset"]

#: What was resolved in THIS process, dotted name -> record.  A process
#: writes one run's receipts, so this is that run's answer.
_SEEN: dict[str, dict[str, str]] = {}

#: The prefix that makes a module this package's rather than the engine's.
_OURS = "arwen_global."

#: The carried NVRTC loader, recorded beside every carried module that has
#: one, because a carried module is only half the physics that ran.
_LOADER = "woof.globe.core.kernels"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _kernel_set_sha256(directory: Path) -> str:
    """One digest over every device source in a kernel directory.

    Over the NAMES as well as the bytes, so a kernel that disappears moves
    the digest rather than leaving it unchanged.
    """

    digest = hashlib.sha256()
    for path in sorted(directory.iterdir()):
        if path.suffix not in (".cu", ".cuh"):
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def note(module: ModuleType) -> ModuleType:
    """Record a resolved physics module.  Returns it, so it wraps a call."""

    name = getattr(module, "__name__", None)
    if not name or name in _SEEN:
        return module
    record: dict[str, str] = {
        "module": name,
        "origin": "package" if name.startswith(_OURS) else "engine",
    }
    source = getattr(module, "__file__", None)
    if source:
        path = Path(source)
        try:
            record["sha256"] = _file_sha256(path)
        except OSError:
            # A module with no readable file (a zip import, a namespace
            # stub) is recorded WITHOUT a hash rather than with a wrong one.
            record["sha256"] = "unreadable"
        if path.name == "__init__.py" and (path.parent / "common.cuh").is_file():
            # The kernel loader.  Its own bytes say nothing about the device
            # sources it binds, and those are what the card ran.
            record["kernels_sha256"] = _kernel_set_sha256(path.parent)
            record["kernels"] = str(sum(
                1 for p in path.parent.iterdir() if p.suffix in (".cu", ".cuh")))
    _SEEN[name] = record
    if name.startswith(_OURS + "core.") and name != _LOADER:
        # THE KERNELS ARE HALF OF WHAT INTEGRATED.  A carried module reaches
        # its loader by import, not through the runtime's resolver, so it
        # would never be recorded on its own -- and it is the row that names
        # the device sources the card actually ran.
        import sys

        loader = sys.modules.get(_LOADER)
        if loader is not None:
            note(loader)
    return module


def integrated_physics_modules() -> dict[str, dict[str, str]]:
    """Every physics module this process resolved, in name order.

    Empty when nothing ran physics, which is the correct answer for a
    receipt written by a command that did not: a row asserting a module that
    was never imported would be a claim about code that did not execute.
    """

    return {name: dict(record) for name, record in sorted(_SEEN.items())}


def reset() -> None:
    """Forget what this process resolved.  For tests, and for them only."""

    _SEEN.clear()
