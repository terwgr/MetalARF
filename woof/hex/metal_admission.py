"""Named device admission and artifact identity for Hex's Metal preview.

This gate is intentionally narrower than the CUDA Hex admission path.  It
admits only the registered FP32 mass-flux-divergence preview slice.  It does
not imply that the rest of the Hex forecast, physics, or recovery path runs on
Metal.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from .errors import ConfigurationRefusal


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_ENTRYPOINT = "hex_mass_flux_divergence"
_EXPECTED_FILES = (
    "libwoof_metal.dylib",
    "woof_default.metallib",
    "woof_exact.metallib",
)


@dataclass(frozen=True, slots=True)
class HexMetalCompileRuntimeIdentity:
    """Verified build and live-device identity for one Hex Metal kernel."""

    source_inputs_sha256: str
    implementation_sha256: str
    bundle_abi_version: int
    runtime_sha256: str
    default_metallib_sha256: str
    exact_metallib_sha256: str
    entrypoint: str
    layout_sha256: str
    shader_sha256: str
    device_name: str
    device_registry_id: int
    identity_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "woof.hex-metal-compile-runtime-identity/v2",
            "source_inputs_sha256": self.source_inputs_sha256,
            "implementation_sha256": self.implementation_sha256,
            "bundle_abi_version": self.bundle_abi_version,
            "runtime_sha256": self.runtime_sha256,
            "default_metallib_sha256": self.default_metallib_sha256,
            "exact_metallib_sha256": self.exact_metallib_sha256,
            "entrypoint": self.entrypoint,
            "layout_sha256": self.layout_sha256,
            "shader_sha256": self.shader_sha256,
            "device_name": self.device_name,
            "device_registry_id": self.device_registry_id,
            "identity_sha256": self.identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class HexMetalCapability:
    """The admitted narrow Metal capability and its artifact identity."""

    capability: str
    precision: str
    device_name: str
    device_registry_id: int
    identity: HexMetalCompileRuntimeIdentity

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "woof.hex-metal-capability/v1",
            "capability": self.capability,
            "precision": self.precision,
            "device_name": self.device_name,
            "device_registry_id": self.device_registry_id,
            "identity": self.identity.as_dict(),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_paths() -> tuple[Path, Path, Path, Path]:
    """Resolve the same bundle paths used by MetalContext, plus its receipt."""
    package_root = Path(__file__).resolve().parents[1]
    packaged = package_root / "libexec" / "metal"
    runtime = Path(os.environ.get(
        "WOOF_METAL_RUNTIME_LIB", packaged / _EXPECTED_FILES[0])).expanduser()
    default = Path(os.environ.get(
        "WOOF_METAL_SHADER_LIBRARY", packaged / _EXPECTED_FILES[1])).expanduser()
    exact = Path(os.environ.get(
        "WOOF_METAL_EXACT_SHADER_LIBRARY", packaged / _EXPECTED_FILES[2])).expanduser()
    manifest_raw = os.environ.get("WOOF_METAL_BUNDLE_MANIFEST")
    if manifest_raw:
        manifest = Path(manifest_raw).expanduser()
    elif any(name in os.environ for name in (
            "WOOF_METAL_RUNTIME_LIB", "WOOF_METAL_SHADER_LIBRARY",
            "WOOF_METAL_EXACT_SHADER_LIBRARY")):
        parents = {runtime.resolve().parent, default.resolve().parent,
                   exact.resolve().parent}
        if len(parents) != 1:
            raise ConfigurationRefusal(
                "hex.metal_bundle", tuple(map(str, sorted(parents))),
                "the three active Metal artifacts do not share one bundle "
                "directory and no explicit bundle manifest was supplied",
                "one complete Metal bundle directory or "
                "WOOF_METAL_BUNDLE_MANIFEST",
            )
        manifest = next(iter(parents)) / "METAL_BUNDLE.json"
    else:
        manifest = packaged / "METAL_BUNDLE.json"
    return runtime.resolve(), default.resolve(), exact.resolve(), manifest.resolve()


@lru_cache(maxsize=8)
def _verified_bundle(
    runtime: str,
    default: str,
    exact: str,
    manifest_path: str,
    file_state: tuple[tuple[str, int, int], ...],
) -> tuple[str, str, int, str, str, str]:
    """Verify bundle receipt and file bytes; cache only for unchanged files."""
    del file_state  # included in the cache key to invalidate changed artifacts
    paths = {Path(runtime), Path(default), Path(exact)}
    manifest = Path(manifest_path)
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfigurationRefusal(
            "hex.metal_bundle_manifest", str(manifest),
            f"the active Metal bundle identity cannot be read: {error}",
            "a built Metal bundle with its METAL_BUNDLE.json receipt",
        ) from error
    if not isinstance(document, dict) or document.get("schema") != "woof-metal-bundle-v2":
        raise ConfigurationRefusal(
            "hex.metal_bundle_manifest", str(manifest),
            "the receipt is not a WOOF Metal bundle v2",
            "a WOOF Metal bundle v2",
        )
    files = document.get("files")
    if not isinstance(files, dict) or set(files) != set(_EXPECTED_FILES):
        raise ConfigurationRefusal(
            "hex.metal_bundle_files", sorted(files) if isinstance(files, dict) else files,
            "the receipt must identify exactly the runtime and two shader "
            "libraries loaded by MetalContext",
            "the complete WOOF Metal runtime/default/exact artifact set",
        )
    abi = document.get("abi_version")
    if (not isinstance(abi, int) or abi >> 16 != 1 or (abi & 0xFFFF) < 7
            or document.get("platform") != "darwin-arm64"):
        raise ConfigurationRefusal(
            "hex.metal_bundle_abi", abi,
            "the Hex preview currently requires the supported Darwin arm64 "
            "Metal ABI and platform",
            "a Darwin arm64 WOOF Metal bundle with ABI major 1/minor >= 7",
        )
    source_inputs_sha256 = document.get("metal_inputs_sha256")
    implementation = document.get("implementation_sha256")
    if (not isinstance(source_inputs_sha256, str) or not _SHA256.fullmatch(source_inputs_sha256)
            or not isinstance(implementation, str)
            or not _SHA256.fullmatch(implementation)):
        raise ConfigurationRefusal(
            "hex.metal_implementation_identity", source_inputs_sha256,
            "the bundle receipt lacks a build-input or implementation "
            "digest",
            "a bundle produced by tools/build_metal_backend.py",
        )
    actual_by_name = {
        _EXPECTED_FILES[0]: Path(runtime),
        _EXPECTED_FILES[1]: Path(default),
        _EXPECTED_FILES[2]: Path(exact),
    }
    hashes: dict[str, str] = {}
    for name, path in actual_by_name.items():
        row = files[name]
        if (not isinstance(row, dict) or row.get("size") != path.stat().st_size
                or not isinstance(row.get("sha256"), str)
                or not _SHA256.fullmatch(row["sha256"])):
            raise ConfigurationRefusal(
                "hex.metal_artifact_receipt", name,
                "the active file size or hash declaration does not match its "
                "bundle receipt",
                "the exact runtime and metallib bytes recorded in the bundle",
            )
        digest = _sha256_file(path)
        if digest != row["sha256"]:
            raise ConfigurationRefusal(
                "hex.metal_artifact_hash", name,
                f"active SHA-256 {digest} differs from receipt {row['sha256']}",
                "the unmodified artifact set produced with this receipt",
            )
        hashes[name] = digest
    return (source_inputs_sha256, implementation, abi,
            hashes[_EXPECTED_FILES[0]], hashes[_EXPECTED_FILES[1]],
            hashes[_EXPECTED_FILES[2]])


def require_hex_metal(context: object) -> HexMetalCapability:
    """Admit only the registered FP32 Hex mass-flux preview capability."""
    from woof.metal.runtime import MetalContext

    if not isinstance(context, MetalContext):
        raise TypeError("Hex Metal admission requires a MetalContext")
    if getattr(context, "_closed", True):
        raise ConfigurationRefusal(
            "hex.metal_context", "closed",
            "a closed context cannot submit the Hex Metal preview kernel",
            "an open MetalContext",
        )
    try:
        from woof.metal.generated import kernel_specs
        resources = getattr(kernel_specs, "HEX_MASS_FLUX_DIVERGENCE_RESOURCES")
        layout_hash = getattr(kernel_specs, "HEX_MASS_FLUX_DIVERGENCE_LAYOUT_HASH")
        shader_hash = getattr(kernel_specs, "HEX_MASS_FLUX_DIVERGENCE_SHADER_HASH")
    except (ImportError, AttributeError) as error:
        raise ConfigurationRefusal(
            "hex.metal_entrypoint", _ENTRYPOINT,
            f"the generated Metal registry does not contain the Hex preview "
            f"signature: {error}",
            "a built bundle generated from the Hex preview signature",
        ) from error
    if not resources or any(resource[1] not in ("float", "int32") for resource in resources):
        raise ConfigurationRefusal(
            "hex.metal_resource_types", resources,
            "the preview signature does not declare its complete typed resource "
            "layout",
            "the registered FP32/int32 Hex mass-flux signature",
        )
    if (not isinstance(layout_hash, str) or not _SHA256.fullmatch(layout_hash)
            or not isinstance(shader_hash, str) or not _SHA256.fullmatch(shader_hash)):
        raise ConfigurationRefusal(
            "hex.metal_compile_identity", (layout_hash, shader_hash),
            "the generated entrypoint is missing valid signature or shader "
            "digests",
            "generated signature and shader hashes",
        )
    registered = False
    try:
        generated_root = Path(__file__).resolve().parents[1] / "metal" / "generated"
        registry_document = json.loads(
            (generated_root / "registry.json").read_text(encoding="utf-8"))
        entries = registry_document["kernels"]
        for entry in entries:
            if entry.get("name") == _ENTRYPOINT:
                registered = (entry.get("layout_hash") == layout_hash
                              and entry.get("shader_hash") == shader_hash)
                break
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
        registered = False
    if not registered:
        raise ConfigurationRefusal(
            "hex.metal_registry_identity", _ENTRYPOINT,
            "the generated source registry and runtime signature digests "
            "disagree",
            "a regenerated Hex signature/runtime registry pair",
        )

    shader_source = Path(__file__).resolve().parents[1] / "metal" / "shaders" / \
        "hex_mass_flux_divergence.metal"
    if not shader_source.is_file() or _sha256_file(shader_source) != shader_hash:
        raise ConfigurationRefusal(
            "hex.metal_shader_identity", str(shader_source),
            "the checked-in Hex MSL source differs from the generated shader "
            "identity used by the runtime binding",
            "regenerated bindings from the current Hex Metal shader",
        )

    runtime, default, exact, manifest = _artifact_paths()
    paths = (runtime, default, exact, manifest)
    try:
        file_state = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns)
                           for path in paths)
    except OSError as error:
        raise ConfigurationRefusal(
            "hex.metal_artifact", str(error),
            "the runtime, both metallibs, and bundle receipt must all be "
            "available for identity verification",
            "a complete installed or development Metal bundle",
        ) from error
    source_inputs_sha256, implementation, abi, runtime_sha, default_sha, exact_sha = \
        _verified_bundle(*(str(path) for path in paths), file_state)

    loaded_runtime = Path(str(getattr(getattr(context, "_lib", None), "_name", "")))
    if loaded_runtime and loaded_runtime.is_absolute() and loaded_runtime.resolve() != runtime:
        raise ConfigurationRefusal(
            "hex.metal_runtime_identity", str(loaded_runtime),
            "the active MetalContext loaded a runtime outside the artifact path "
            "whose receipt was verified",
            "a context created with the same WOOF_METAL_RUNTIME_LIB bundle",
        )

    # The generated shader hash describes the MSL input.  Bind it alongside
    # the verified metallib digests; do not claim to reverse-engineer Apple's
    # compiled metallib back into source identity.
    identity_payload = {
        "schema": "woof.hex-metal-compile-runtime-identity/v2",
        "source_inputs_sha256": source_inputs_sha256,
        "implementation_sha256": implementation,
        "bundle_abi_version": abi,
        "runtime_sha256": runtime_sha,
        "default_metallib_sha256": default_sha,
        "exact_metallib_sha256": exact_sha,
        "entrypoint": _ENTRYPOINT,
        "layout_sha256": layout_hash,
        "shader_sha256": shader_hash,
        "device_name": str(context.device_name),
        "device_registry_id": int(context.device_registry_id),
    }
    encoded = json.dumps(identity_payload, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    identity = HexMetalCompileRuntimeIdentity(
        source_inputs_sha256=source_inputs_sha256,
        implementation_sha256=implementation,
        bundle_abi_version=abi,
        runtime_sha256=runtime_sha,
        default_metallib_sha256=default_sha,
        exact_metallib_sha256=exact_sha,
        entrypoint=_ENTRYPOINT,
        layout_sha256=layout_hash,
        shader_sha256=shader_hash,
        device_name=str(context.device_name),
        device_registry_id=int(context.device_registry_id),
        identity_sha256=hashlib.sha256(encoded).hexdigest(),
    )
    return HexMetalCapability(
        capability="hex.mass_flux_divergence.preview",
        precision="float32",
        device_name=identity.device_name,
        device_registry_id=identity.device_registry_id,
        identity=identity,
    )


__all__ = [
    "HexMetalCapability",
    "HexMetalCompileRuntimeIdentity",
    "require_hex_metal",
]
