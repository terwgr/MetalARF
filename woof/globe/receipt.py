"""Atomic self-hashed WOOF global receipts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .constants import RECEIPT_SCHEMA
from .pins import ACCEPTED_PINS_HASHES, PINS_HASH_BY_SCHEME


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


#: Distributions whose version changes a run's numbers, in the order a reader
#: cares about them.  A name that is not installed is simply absent.
RECORDED_LIBRARIES = ("recast-woof", "recast-woof-data", "numpy", "scipy",
                      "cupy-cuda13x", "cupy-cuda12x", "cupy", "netCDF4")


def library_versions() -> dict[str, str]:
    """The installed versions of everything a run's arithmetic rides on.

    THE BREAKAGE THIS NAMES.  A receipt recorded the configuration, the
    arithmetic pins, the physics identity and the machine, and said nothing
    about the libraries.  Two runs of the same configuration on the same card
    can disagree, and this package already has a measured instance of exactly
    that: the Legendre quadrature this model's spectral tables ride on moved
    its bits between numpy 2.2.6 and 2.3.0, and a table's recorded digest
    reproduces under one and not the other.  A receipt that cannot say which
    numpy produced it cannot answer the first question anybody asks about two
    runs that differ, and the answer is not recoverable later: by the time the
    question is asked the environment has moved.

    The interpreter is recorded beside them for the same reason.
    """

    import sys
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {
        "python": ".".join(str(n) for n in sys.version_info[:3]),
    }
    for name in RECORDED_LIBRARIES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def finalize_receipt(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["schema"] = RECEIPT_SCHEMA
    # Beside the pins, never inside them: `pins_hash` is the identity of the
    # ARITHMETIC and two runs under different numpys share it, which is the
    # point of it.  The libraries are what tells a reader why two runs that
    # share it might still differ.
    result.setdefault("libraries", library_versions())
    # WHOSE PHYSICS INTEGRATED.  The libraries above name the engine's
    # version, and this package carries most of the physics it runs, so the
    # engine's version does not answer the question.  Every physics module
    # this process actually resolved is recorded with its origin and the
    # SHA-256 of the file that was imported (and, for the kernel loader, one
    # digest over the device sources it binds).  A command that ran no
    # physics records an empty table rather than a claim about code that did
    # not execute.
    from .physics.provenance import integrated_physics_modules

    result.setdefault("physics_modules", integrated_physics_modules())
    result.setdefault("engine_seam", engine_seam_verdict())
    result.pop("self_sha256", None)
    result["self_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def engine_seam_verdict() -> dict[str, object]:
    """Whether the engine under the carried physics is the one it was pinned to.

    THE BREAKAGE THIS NAMES.  `physics_modules` records the SHA-256 of every
    carried module that integrated, which answers "whose physics ran" for the
    half of the physics this package carries.  It cannot see the other half.
    `woof/core/constants.py` stays on the engine and supplies CUDA_DEFINES to
    the preamble of every carried kernel, so a run against a 2.8.x that moved
    it produces a receipt whose module hashes are identical to yesterday's
    while every compiled kernel's assembled source, its digest and its
    floating-point contraction have moved.  The same is true of the twelve
    engine modules the carried driver imports at module scope.

    So the seam verdict travels in the receipt beside the module hashes: the
    engine version the pins were taken against, the engine version that
    actually resolved, and every file whose bytes are not the pinned ones, by
    name.  A receipt is read long after the environment it was written in is
    gone, which is why this is recorded rather than left to the doctor.

    It never raises.  A receipt that failed to write because the seam manifest
    was unreadable would lose the run, so an unreadable seam is recorded AS
    unreadable.
    """

    verdict: dict[str, object] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            verdict["engine_resolved"] = version("woof")
        except PackageNotFoundError:
            verdict["engine_resolved"] = "not installed"

        from .engine_seam import check_seam, load_manifest

        manifest = load_manifest()
        verdict["pinned_against"] = str(
            manifest.get("engine", {}).get("version", "unrecorded"))
        rows = check_seam()
        unproven = [row for row in rows if row.verdict != "proven"]
        verdict["files"] = len(rows)
        verdict["proven"] = len(rows) - len(unproven)
        verdict["unproven"] = {row.path: row.verdict for row in unproven}
    except Exception as exc:  # noqa: BLE001 - a receipt is never lost to this
        verdict["unreadable"] = f"{type(exc).__name__}: {exc}"
    return verdict


def write_receipt(path: str | Path, payload: dict[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    result = finalize_receipt(payload)
    temporary = target.with_name(f".{target.name}.partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return target


def check_receipt(path: str | Path) -> dict[str, object]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("WOOF global receipt schema mismatch")
    # The self-hash only proves the file is intact, not that the numbers in it
    # were produced by this build's arithmetic.  Without the pin comparison a
    # receipt written under a superseded PIN_DOCUMENT validates green and its
    # gate values are read as evidence for the current one.  The build
    # integrates one arithmetic per [semi_implicit] scheme, each under its
    # own pin (the external proxy's is the pin of every receipt written
    # before the vertical-mode scheme existed), so any of those pins is
    # this build's.
    # this build's pins and the pins WOOF 1.0.0 wrote for the same arithmetics
    if payload.get("pins_hash") not in ACCEPTED_PINS_HASHES:
        raise ValueError(
            "WOOF global receipt arithmetic pins mismatch: receipt "
            f"{payload.get('pins_hash')!r} is not a pin of this build "
            + "("
            + ", ".join(
                f"{scheme}: {digest}" for scheme, digest in PINS_HASH_BY_SCHEME.items()
            )
            + ")"
        )
    self_hash = payload.pop("self_sha256", None)
    expected = hashlib.sha256(canonical(payload)).hexdigest()
    if self_hash != expected:
        raise ValueError("WOOF global receipt self-hash mismatch")
    payload["self_sha256"] = self_hash
    return payload


__all__ = ["check_receipt", "finalize_receipt", "write_receipt"]
