#!/usr/bin/env python3
"""Stage a verified macOS arm64 Metal bundle into the WOOF package.

Build first with ``tools/build_metal_backend.py``, then pass its output
directory here. This copies only the declared dylib, metallib, and manifest;
it does not compile source or invent release pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from woof import bridge_assets  # noqa: E402
from tools.metal_abi import canonical_metal_abi_version  # noqa: E402


PACKAGE_DESTINATION = ROOT / "woof" / "libexec" / "metal"
EXPECTED = bridge_assets.METAL_FILES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage(bundle: Path, destination: Path) -> dict:
    bundle = bundle.resolve()
    manifest_path = bundle / "METAL_BUNDLE.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"invalid Metal bundle manifest {manifest_path}: {error}")
    expected_abi = canonical_metal_abi_version()
    abi = manifest.get("abi_version") if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict)
            or manifest.get("schema") != "woof-metal-bundle-v1"
            or not isinstance(abi, int)
            or isinstance(abi, bool)
            or abi != expected_abi
            or manifest.get("platform") != "darwin-arm64"
            or manifest.get("minimum_os") != "14.0"):
        raise SystemExit(
            "Metal bundle must declare schema woof-metal-bundle-v1 with the "
            f"canonical header ABI 0x{expected_abi:08x}, platform "
            "darwin-arm64, and minimum_os 14.0")
    source_commit = manifest.get("source_commit")
    implementation_sha256 = manifest.get("implementation_sha256")
    implementation_file_count = manifest.get("implementation_file_count")
    if (not isinstance(source_commit, str) or len(source_commit) != 40
            or any(c not in "0123456789abcdef" for c in source_commit)
            or not isinstance(implementation_sha256, str)
            or len(implementation_sha256) != 64
            or any(c not in "0123456789abcdef"
                   for c in implementation_sha256)
            or not isinstance(implementation_file_count, int)
            or isinstance(implementation_file_count, bool)
            or implementation_file_count <= 0):
        raise SystemExit(
            "Metal bundle manifest must record a full source commit and a "
            "non-empty implementation source fingerprint")
    records = manifest.get("files")
    if not isinstance(records, dict) or set(records) != set(EXPECTED):
        raise SystemExit(
            f"Metal bundle file list must be exactly {list(EXPECTED)}")
    for name in EXPECTED:
        path = bundle / name
        record = records[name]
        if (not isinstance(record, dict)
                or not path.is_file()
                or path.stat().st_size != record.get("size")
                or _sha256(path) != record.get("sha256")):
            raise SystemExit(f"Metal bundle artifact missing or hash mismatch: {path}")
    runtime = (bundle / "libwoof_metal.dylib").read_bytes()
    minimum_os = bridge_assets.macho_arm64_minimum_os(
        runtime, label="libwoof_metal.dylib")
    expected_minimum_os = (*bridge_assets.DARWIN_MINIMUM_OS, 0)
    if minimum_os != expected_minimum_os:
        raise SystemExit(
            "libwoof_metal.dylib declares minimum macOS "
            f"{minimum_os!r}; expected {expected_minimum_os!r}")
    install_ids = bridge_assets.macho_dylib_ids(
        runtime, label="libwoof_metal.dylib")
    if install_ids != ("@rpath/libwoof_metal.dylib",):
        raise SystemExit(
            "libwoof_metal.dylib must have the relocatable install name "
            "@rpath/libwoof_metal.dylib")

    destination = destination.resolve()
    if destination == PACKAGE_DESTINATION.resolve():
        shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)
    for name in (*EXPECTED, "METAL_BUNDLE.json"):
        shutil.copy2(bundle / name, destination / name)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path,
                        help="output directory from build_metal_backend.py")
    parser.add_argument("--dest", type=Path, default=PACKAGE_DESTINATION,
                        help="stage location (default: package libexec/metal)")
    parser.add_argument("--clean", action="store_true",
                        help="remove only the destination Metal staging directory")
    args = parser.parse_args()
    destination = args.dest.resolve()
    if args.clean:
        if destination == ROOT / "woof" / "libexec" / "metal":
            shutil.rmtree(destination, ignore_errors=True)
        else:
            raise SystemExit("--clean is allowed only for the package staging path")
        print(f"removed {destination}")
        return 0
    manifest = stage(args.bundle, destination)
    print(json.dumps({"destination": str(destination),
                      "platform": manifest["platform"],
                      "files": sorted(manifest["files"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
