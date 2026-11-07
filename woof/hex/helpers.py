"""The hex model's compiled helpers: carried by the wheel, or fetched on first use.

The helpers are the hex release's own compiled programs (the libexec folder of
its manylinux wheel): the hex-vertical geometry bridge, the stream writer, the
binary64 static builder, the case-9 boundary builder, the RRFS programs, the
S2S renderer and reducers, the observation scorer, the table decoders and the
libraries hex loads.  An ops wheel carries them in ``woof/hex/libexec``.

THE BREAKAGE THIS PREVENTS (WOOF 1.1.0 r15, 2026-10-23): carried in the
platform wheel they are 39.9 MB compressed, and the wheel came out at
138,837,702 bytes against PyPI's 100,000,000-byte per-file limit, so 1.1.0
could not be published.  A PyPI wheel now carries their pins instead
(``data/hex-helpers-pins.json``): the release publishes them as one asset,
``woof-hex-helpers-<release>-<platform>.zip``, and the first hex door that
needs a helper downloads it, verifies the archive and every file against the
pins (exact size and SHA-256), and stages it into this release's own
``~/.woof/hex-helpers/<release>-<archive digest>``.  ``woof fetch-bridges`` and
``woof setup`` (``--from DIR`` included) stage it ahead of time.

Every door asks :func:`path` for the file it used to read beside this module,
so the bytes a door runs are the bytes the hex release proved, wherever they
were staged.  Doctors and receipt writers never download (``is_inspection``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import zipfile

PINS_SCHEMA = "woof.hex-helpers-pins/v1"
PINS_PATH = Path(__file__).resolve().parent / "data" / "hex-helpers-pins.json"
PACKAGED_DIR = Path(__file__).resolve().parent / "libexec"
#: Where to stage instead of ~/.woof/hex-helpers/<tag> (a read-only home, a shared image).
DIR_ENV = "WOOF_HEX_HELPERS_DIR"
#: The release-asset mirror; the same variable woof fetch-bridges reads.
URL_BASE_ENV = "WOOF_BRIDGE_ASSET_URL_BASE"
MARKER = ".complete"


class HelperError(RuntimeError):
    """The hex helper archive could not be staged."""


_LOCK = threading.Lock()
_ATTEMPT: dict[str, str | None] = {}


def load_pins(path: Path | None = None) -> dict | None:
    """This install's pins, or None (an ops wheel, a pure wheel, another platform)."""

    path = path or PINS_PATH
    try:
        pins = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if pins.get("schema") != PINS_SCHEMA or not isinstance(pins.get("files"), dict):
        return None
    try:
        from woof.bridge_assets import host_platform
        if pins.get("platform") != host_platform():
            return None
    except Exception:                                # noqa: BLE001
        return None
    return pins


def _tag(pins: dict) -> str:
    release = "".join(c if c.isalnum() or c in "._+-" else "_" for c in str(pins.get("release") or "release"))
    return f"{release}-{pins['bundle']['sha256'][:12]}"


def staged_dir(pins: dict) -> Path:
    """This release's staging directory for the helpers."""

    override = os.environ.get(DIR_ENV)
    if override and override.strip():
        return Path(override).expanduser()
    return Path.home() / ".woof" / "hex-helpers" / _tag(pins)


def bundle_url(pins: dict) -> str:
    base = os.environ.get(URL_BASE_ENV, "").strip().rstrip("/")
    if not base:
        from woof.bridges import REPOSITORY_URL
        base = f"{REPOSITORY_URL}/releases/download/{pins['release']}"
    return f"{base}/{pins['bundle']['filename']}"


def complete(directory: Path, pins: dict) -> bool:
    """Was ``directory`` staged, verified, from this release's archive?"""

    try:
        marker = json.loads((directory / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if marker.get("bundle_sha256") != pins["bundle"]["sha256"]:
        return False
    for name, pin in pins["files"].items():
        try:
            if (directory / name).stat().st_size != pin["bytes"]:
                return False
        except OSError:
            return False
    return True


def packaged() -> bool:
    """Does this wheel carry the helpers itself (an ops wheel)?"""

    return (PACKAGED_DIR / "HEX-HELPERS.json").is_file()


@contextlib.contextmanager
def _owner(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as handle:
        try:
            import fcntl
        except ImportError:                          # pragma: no cover - Windows has no helpers to stage
            yield
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage(pins: dict | None = None, *, from_dir: Path | str | None = None, progress=None) -> Path:
    """Stage the helpers (from ``from_dir`` when it holds the archive, else the release), verified."""

    say = progress or (lambda message: print(message, file=sys.stderr))
    pins = pins or load_pins()
    if pins is None:
        raise HelperError("this install carries no hex helper pins for this platform")
    dest = staged_dir(pins)
    if complete(dest, pins):
        return dest
    bundle = pins["bundle"]
    with _owner(dest.parent / f".{dest.name}.lock"):
        if complete(dest, pins):
            return dest
        archive = None
        if from_dir is not None and (Path(from_dir) / bundle["filename"]).is_file():
            archive = Path(from_dir) / bundle["filename"]
        download = dest.parent / f".{dest.name}.download"
        if archive is None:
            from woof.bridge_assets import download_bundle
            download.mkdir(parents=True, exist_ok=True)
            archive = download / bundle["filename"]
            url = bundle_url(pins)
            say(f"woof hex: fetching the hex helpers ({bundle['bytes'] / 1e6:.1f} MB) from {url}")
            download_bundle(url, archive, expected_bytes=bundle["bytes"], progress=say)
        size = archive.stat().st_size
        if size != bundle["bytes"] or _sha256(archive) != bundle["sha256"]:
            if archive.parent == download:
                archive.unlink(missing_ok=True)
            raise HelperError(f"{archive}: not the pinned {bundle['filename']} "
                              f"({bundle['bytes']} bytes, sha256 {bundle['sha256']})")
        work = Path(tempfile.mkdtemp(prefix=f".{dest.name}.stage-", dir=dest.parent))
        try:
            with zipfile.ZipFile(archive) as zf:
                held = set(zf.namelist())
                absent = sorted(set(pins["files"]) - held)
                if absent:
                    raise HelperError(f"{bundle['filename']} lacks {', '.join(absent)}")
                for name, pin in sorted(pins["files"].items()):
                    data = zf.read(name)
                    if len(data) != pin["bytes"] or hashlib.sha256(data).hexdigest() != pin["sha256"]:
                        raise HelperError(f"{bundle['filename']}: {name} is not its pinned bytes")
                    target = work / name
                    target.write_bytes(data)
                    os.chmod(target, 0o755)
            (work / MARKER).write_text(json.dumps({"bundle": bundle["filename"],
                                                   "bundle_sha256": bundle["sha256"],
                                                   "files": len(pins["files"])}) + "\n", encoding="utf-8")
            if dest.exists():
                shutil.rmtree(dest)
            os.replace(work, dest)
        finally:
            if work.exists():
                shutil.rmtree(work, ignore_errors=True)
            if download.exists():
                shutil.rmtree(download, ignore_errors=True)
    say(f"woof hex: {len(pins['files'])} hex helpers staged at {dest}, each verified against this release's pins")
    return dest


def _inspecting() -> bool:
    try:
        from .assets import is_inspection
        return bool(is_inspection())
    except Exception:                                # noqa: BLE001
        bridges = sys.modules.get("woof.bridges")
        return bool(getattr(bridges, "_INSPECTION_ONLY", False))


def failure() -> str | None:
    """Why this process's automatic staging failed, if it did."""

    return next((reason for reason in _ATTEMPT.values() if reason), None)


def ensure(pins: dict) -> bool:
    """Stage on first use, once per process; True when the helpers are staged."""

    dest = staged_dir(pins)
    if complete(dest, pins):
        return True
    if _inspecting():
        return False
    with _LOCK:
        key = pins["bundle"]["sha256"]
        if key in _ATTEMPT:
            return _ATTEMPT[key] is None and complete(dest, pins)
        try:
            from woof.explain import warn
            warn(f"the hex helpers are fetched on demand: this wheel does not carry them, so "
                 f"{pins['bundle']['filename']} is being staged into {dest} before this run continues")
        except Exception:                            # noqa: BLE001
            pass
        try:
            stage(pins)
        except Exception as error:                   # noqa: BLE001
            _ATTEMPT[key] = f"staging the hex helpers failed ({error})"
            print(f"woof hex: {_ATTEMPT[key]}; `woof fetch-bridges` retries it", file=sys.stderr)
            return False
        _ATTEMPT[key] = None
        return True


def path(name: str, *, fetch: bool = True) -> Path:
    """Where helper ``name`` lives on this install.

    ``woof/hex/libexec/<name>`` when the wheel carries it (or when the helper
    is not one this release pins); otherwise this release's staged copy,
    staged now when ``fetch`` and this is not an inspection.  The path is
    returned whether or not the file exists, so every caller keeps its own
    refusal for a helper it cannot find.
    """

    carried = PACKAGED_DIR / name
    if carried.is_file():
        return carried
    pins = load_pins()
    if pins is None or name not in pins["files"]:
        return carried
    if fetch:
        ensure(pins)
    return staged_dir(pins) / name


def on_demand(name: str) -> bool:
    """Is ``name`` a pinned helper this wheel leaves to the release archive?"""

    if (PACKAGED_DIR / name).is_file():
        return False
    pins = load_pins()
    return pins is not None and name in pins["files"]


def status() -> dict:
    """The helpers' state for the doctors: carried, staged, on demand, or absent."""

    if packaged():
        names = [p.name for p in PACKAGED_DIR.iterdir() if p.is_file() and p.name != "HEX-HELPERS.json"]
        return {"state": "packaged", "dir": str(PACKAGED_DIR), "files": len(names)}
    pins = load_pins()
    if pins is None:
        return {"state": "absent"}
    dest = staged_dir(pins)
    return {"state": "staged" if complete(dest, pins) else "on-demand", "dir": str(dest),
            "files": len(pins["files"]), "bundle": pins["bundle"]["filename"],
            "bundle_bytes": pins["bundle"]["bytes"], "url": bundle_url(pins)}


__all__ = ["DIR_ENV", "HelperError", "PINS_PATH", "complete", "ensure", "failure", "load_pins",
           "on_demand", "packaged", "path", "stage", "staged_dir", "status"]
