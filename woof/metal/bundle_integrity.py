"""Source fingerprint checks for the packaged Metal runtime bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


STALE_BUNDLE_MESSAGE = (
    "stale Metal bundle: run tools/build_metal_backend.py")


def metal_source_sha256(repo_root: Path) -> str:
    """Hash all registered shader inputs and the canonical kernel registry."""
    repo_root = Path(repo_root).resolve()
    metal_root = repo_root / "woof" / "metal"
    inputs = [metal_root / "kernel_signatures.json"]
    for folder in (metal_root / "shaders",
                   metal_root / "signature_fragments"):
        if folder.is_dir():
            inputs.extend(path for path in folder.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(set(inputs), key=lambda item: item.relative_to(repo_root).as_posix()):
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


def verify_metal_bundle(repo_root: Path) -> None:
    """Fail before a Metal run when shader/signature sources outpace bundle."""
    repo_root = Path(repo_root).resolve()
    manifest_path = repo_root / "woof" / "libexec" / "metal" / "METAL_BUNDLE.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest["metal_inputs_sha256"]
        current = metal_source_sha256(repo_root)
    except (OSError, ValueError, KeyError, TypeError):
        raise RuntimeError(STALE_BUNDLE_MESSAGE) from None
    if expected != current:
        raise RuntimeError(STALE_BUNDLE_MESSAGE)
