"""What this package leaves on the engine, and whether it is what was measured.

`woof.globe.core` is the physics this package CARRIES.  This module is the
other half of that decision: the engine files the carried physics still
reaches, pinned by path, size and SHA-256 at the engine version the decision
was measured against.

THE SCOPE, because the doctor row this feeds says "N/N files proven" and a
scope nobody wrote down is a claim nobody can check.  It is the DIRECT
engine imports of code under `woof/globe/core/` -- 42 modules on this
tree, every one of them a row here, the kernel manifest the carried loader
files its images in among them -- plus four: the LETKF the assimilation
runs on the engine, the local-GPU switch the native suite reads, and
`woof/core/jacobi_eigh.py` and `woof/core/sase_limits.py`, which pinned
files reach in turn.  46 rows.

WHAT THE SCOPE IS NOT, said here because "46/46 files proven" reads like an
import closure and is not one.  Following module-scope imports out of those
46 against published woof 2.7.0 reaches 109 engine modules; 63 are
unpinned, and 25 of the 63 differ from the tree this model was graded in or
do not exist there at all (`woof/core/streaming.py` 1390 changed lines,
the engine's own copy of the physics driver 326, `woof/experiment.py` 267,
`woof/ingest/water_temperature.py` 182, and 21 more).  Not one of them is
on this model's run path: they are reached only through
`woof.core.mpas_column_batch`, the MYNN, Noah-MP and legacy-RRTMG
runtimes, and the regional experiment and streaming machinery, none of
which this package's door can select, because the native options admit the
carried schemes only.  Measured on the desktop, 2026-09-10.

The engine's DOORS are deliberately absent: the run plan, the fetchers, the
obs readers and the writers are an API dependency, measured by symbol in
`tools/measure_boundary.py` and by signature in
`tools/measure_engine_signatures.py`, and pinning their bytes would print a
moved row on every engine release while saying nothing about the physics.

WHY A PIN AND NOT A CARRY.  Each file here was measured and found safe to
leave: byte-identical to the tree the model was graded in, or differing only
in code this package never enters (`RunConfig`, which the package never
constructs; `DomainState`, likewise; the regional allocator).  That is a fact
about ONE published engine.  A pip resolution inside `woof>=2.8.0,<2.9` can
put a different one underneath without printing anything about a file's
contents, and `woof.core.constants` alone reaches the assembled source of
every carried kernel: move it and every kernel's `source_sha256`, PTX and
floating-point contraction move with it, with no other signal.

WHY A WARNING AND NOT A REFUSAL.  A moved seam file is reported by name as
unproven.  It is not refused: the version ceiling in the dependency pin is the
refusal, and a package that stopped running because a comment changed in a
file it reads two constants from would be worse than one that says which file
it no longer recognises.  A refusal has to name the breakage it prevents, and
"this file's bytes are not the ones I measured" does not name one.

The manifest is written by `tools/pin_engine_seam.py`, which reads the hashes
off the installed engine.  Nothing here is typed by hand.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

__all__ = ["SeamFile", "manifest_path", "load_manifest", "engine_root",
           "check_seam", "SEAM_SCHEMA"]

SEAM_SCHEMA = "gpuwm.arwen-global-engine-seam/v1"


@dataclass(frozen=True)
class SeamFile:
    """One pinned engine file and what the installed engine has there."""

    path: str
    #: What this package reaches in it.
    reached: str
    pinned_sha256: str
    pinned_size: int
    #: None when the installed engine does not have the file at all.
    found_sha256: str | None
    found_size: int | None

    @property
    def verdict(self) -> str:
        if self.found_sha256 is None:
            return "absent"
        return "proven" if self.found_sha256 == self.pinned_sha256 else "moved"


def manifest_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "engine-seam.json"


def load_manifest() -> dict:
    text = manifest_path().read_text(encoding="utf-8")
    payload = json.loads(text)
    if payload.get("schema") != SEAM_SCHEMA:
        raise ValueError(
            f"{manifest_path()} is schema {payload.get('schema')!r}, not "
            f"{SEAM_SCHEMA}")
    return payload


def engine_root() -> Path | None:
    """The directory the installed engine's package sits in, or None."""

    try:
        import woof
    except Exception:
        return None
    source = getattr(woof, "__file__", None)
    if not source:
        return None
    return Path(source).resolve().parent.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_seam(root: Path | None = None) -> tuple[SeamFile, ...]:
    """Every pinned file, hashed on the installed engine.

    Raises nothing on a missing file: that is a row, not an error.
    """

    if root is None:
        root = engine_root()
    payload = load_manifest()
    rows: list[SeamFile] = []
    for entry in payload["files"]:
        found_sha = found_size = None
        if root is not None:
            path = root / entry["path"]
            if path.is_file():
                found_sha = _sha256(path)
                found_size = path.stat().st_size
        rows.append(SeamFile(
            path=entry["path"],
            reached=entry.get("reached", ""),
            pinned_sha256=entry["sha256"],
            pinned_size=int(entry["size"]),
            found_sha256=found_sha,
            found_size=found_size,
        ))
    return tuple(rows)
