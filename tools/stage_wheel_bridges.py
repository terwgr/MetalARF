"""Stage every declared native artifact into a platform wheel.

Run after the release Cargo builds and before ``python -m build``. The shared
rw_wrfbatch executable and its five native launchers carry the old tool names.
Both package staging and a user's bridge directory receive the full roster.
Basemap assets stay in recast-woof-data and the downloadable bridge bundle.
``tools.battery.wheel_size`` measures wheels before release gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
import shutil
import sys
import zipfile

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from woof import bridge_assets, bridges  # noqa: E402

#: Name of the record this tool writes beside the staged artifacts.  A
#: wheel that carries binaries must be able to SAY which platform they
#: are for and which commit built them, or a mis-tagged wheel is only
#: discoverable by running it.
MANIFEST_NAME = "BUNDLE.json"

#: Schema of that record.
MANIFEST_SCHEMA = "gpuwm-wheel-bridge-bundle-v1"

#: Where each workspace puts its release build, by crate-relative path.
_WORKSPACE_TARGET = {
    "tools/rw_icechunk": "target/release",
    "tools/nextgen_hydrofabric": "target/release",
    "tools/nextgen_verify": "target/release",
    "tools/nextgen_output": "target/release",
    "tools/nextgen_forcing": "target/release",
    "tools/hydro_world": "target/release",
    "tools/hydro_world_params": "target/release",
    "tools/nextgen_reservoir_prepare": "target/release",
    "tools/hydro_world_init": "target/release",
    "tools/ocean_prepare": "target/release",
    "tools/rw-bathy": "target/release",
    "tools/wind-table": "target/release",
    bridges.CRATE_RELATIVE: "target/release",
    bridges.RUSTWX_CRATE_RELATIVE: "target/release",
    "tools/region_global_dealias": "target/release",
    "tools/rw_wps": "target/release",
    "tools/zarr_bridge": "target/release",
}


def purge_build_staging(root: Path | None = None) -> list[Path]:
    """Remove staged artifacts from setuptools' ``build/`` copy tree.

    setuptools copies package data into ``build/lib.<plat>-<pyver>/`` and
    never deletes files that have gone away, so building a second
    platform wheel in the same checkout silently ships the FIRST
    platform's binaries alongside the second's.  Measured, not theorised:
    the linux wheel came out at 105.99 MB carrying 22 artifacts -- every
    Linux binary plus every Windows ``.exe`` and ``.dll`` -- which is
    both wrong and over PyPI's 100 MB cap.

    So every staging operation purges that tree first.  Putting it here
    rather than in a workflow step is deliberate: a release runs this
    tool once per platform, and a cleanup that lives beside the thing it
    cleans cannot be left out of a new caller.
    """

    root = root or _REPO_ROOT
    removed: list[Path] = []
    build = root / "build"
    if not build.is_dir():
        return removed
    for staged in build.glob("lib*/woof/libexec"):
        shutil.rmtree(staged, ignore_errors=True)
        removed.append(staged)
    return removed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_native_payload(platform: str,
                             artifact: bridge_assets.BundledArtifact,
                             filename: str, payload: bytes) -> None:
    """Refuse Darwin wheel inputs with wrong architecture or install IDs."""

    if platform != bridge_assets.DARWIN_PLATFORM:
        return
    minimum_os = bridge_assets.macho_arm64_minimum_os(
        payload, label=f"{platform}: {filename}")
    expected_minimum_os = (*bridge_assets.DARWIN_MINIMUM_OS, 0)
    if minimum_os != expected_minimum_os:
        raise SystemExit(
            f"stage_wheel_bridges: {filename} declares minimum macOS "
            f"{minimum_os!r}; expected {expected_minimum_os!r}")
    if artifact.kind == "library":
        names = bridge_assets.macho_dylib_ids(
            payload, label=f"{platform}: {filename}")
        expected = f"@rpath/{filename}"
        if names != (expected,):
            raise SystemExit(
                f"stage_wheel_bridges: {filename} install name(s) "
                f"{names or 'missing'}; expected {expected!r}; relink "
                "the Darwin library before wheel staging")


def source_path(artifact: bridge_assets.BundledArtifact, platform: str) -> Path:
    """Where ``artifact``'s built bytes live in this checkout."""

    target = _WORKSPACE_TARGET.get(artifact.crate)
    if target is None:
        raise SystemExit(
            f"{artifact.name}: crate {artifact.crate!r} has no declared "
            f"release-build directory in stage_wheel_bridges._WORKSPACE_TARGET; "
            f"add it there beside the other workspaces")
    filename = bridge_assets.artifact_filename(artifact, platform)
    return _REPO_ROOT / artifact.crate / target / filename


def _is_packaged_destination(destination: Path) -> bool:
    """Is ``destination`` the in-package directory a wheel carries?

    The two destinations this tool writes have opposite emptying rules,
    and the difference is not cosmetic.  The PACKAGE directory must be
    emptied first: setuptools never removes package data that has gone
    away, so a second platform's staging would ship both platforms'
    binaries in one wheel -- measured at 105.99 MB, over PyPI's cap.
    Any OTHER destination is somebody's directory: a
    ``~/.woof/bridges`` holds the renderer's ``assets/`` basemap tree
    and whatever kept-stale copies its owner parked there, none of which
    this tool staged and none of which it may delete.
    """

    try:
        return destination.resolve() == bridges.packaged_bridge_dir().resolve()
    except OSError:                              # pragma: no cover - exotic FS
        return False


def staged_artifacts(destination: Path) -> tuple[bridge_assets.BundledArtifact, ...]:
    """What ``destination`` takes: the in-package directory gets what the
    wheel carries (:func:`woof.bridge_assets.wheel_artifacts`, which leaves
    out the on-demand artifacts so the wheel stays under PyPI's per-file
    limit); any other destination, a user's bridge directory included, gets
    the whole declared roster, on-demand artifacts and all."""
    if _is_packaged_destination(destination):
        return bridge_assets.wheel_artifacts()
    return bridge_assets.BUNDLED_ARTIFACTS

def stage(platform: str, *, destination: Path | None = None) -> Path:
    """Copy every artifact ``destination`` takes into it, or refuse.

    Fail closed and fail COMPLETE: every missing artifact is collected
    and reported at once, because a build operator who has to re-run
    cargo wants the whole list, not the first name alphabetically.

    The declaration is the whole point.  A directory staged BY HAND has
    nothing to check itself against, which is how a
    ``~/.woof/bridges`` came to hold sixteen of the declared
    artifacts -- missing ``rw_netcdf`` and ``netcdf_writer``, so every
    NetCDF source refused to decode and every history write would have
    refused, on a home that looked staged.  Driven from
    :data:`woof.bridge_assets.BUNDLED_ARTIFACTS`, the artifact that
    joins the declaration next travels to every destination on the same
    commit.
    """

    destination = destination or bridges.packaged_bridge_dir()
    roster = staged_artifacts(destination)
    missing: list[str] = []
    plan: list[tuple[bridge_assets.BundledArtifact, Path, Path]] = []
    for artifact in roster:
        origin = source_path(artifact, platform)
        if not origin.is_file():
            missing.append(f"  {artifact.name:24s} expected at {origin}")
            continue
        plan.append((artifact, origin,
                     destination / bridge_assets.artifact_filename(artifact, platform)))
    if missing:
        raise SystemExit(
            f"stage_wheel_bridges: {len(missing)} of "
            f"{len(roster)} declared artifacts are "
            f"not built for {platform}:\n" + "\n".join(missing) +
            "\n\nBuild them first, from the checkout root:\n"
            + "\n".join(
                "  " + bridges.cargo_build_one_liner(crate)
                for crate in dict.fromkeys(
                    artifact.crate
                    for artifact in roster)))

    if platform == bridge_assets.DARWIN_PLATFORM:
        for artifact, origin, _target in plan:
            _validate_native_payload(
                platform, artifact, origin.name, origin.read_bytes())

    purge_build_staging()
    if _is_packaged_destination(destination) and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for artifact, origin, target in plan:
        shutil.copy2(origin, target)
        if artifact.kind == "executable" and os.name != "nt":
            target.chmod(target.stat().st_mode | 0o111)
        records.append({
            "artifact": artifact.name,
            "filename": target.name,
            "kind": artifact.kind,
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
        })

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "platform": platform,
        "artifacts": records,
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def stage_from_bundle(archive: Path, platform: str,
                      *, destination: Path | None = None) -> Path:
    """Stage from a packed release bundle instead of a build tree.

    The release builds each platform's binaries on its own target-native
    runner, then one job assembles the distributions.  That job cannot
    have a Windows cargo target directory and a Linux one, so the
    platform wheels are staged from the BUNDLES those runners uploaded --
    the identical bytes the release publishes and pins, which also means
    the wheel and the downloadable bundle can never disagree.

    Only the binaries are taken.  ``assets/`` members are the renderer's
    basemap shapefiles, deliberately not shipped in the wheel (see the
    module docstring).    """

    destination = destination or bridges.packaged_bridge_dir()
    wanted = {
        bridge_assets.artifact_filename(artifact, platform): artifact
        for artifact in staged_artifacts(destination)
    }
    purge_build_staging()
    if _is_packaged_destination(destination) and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(archive) as bundle:
        members = set(bundle.namelist())
        missing = sorted(name for name in wanted if name not in members)
        if missing:
            raise SystemExit(
                f"stage_wheel_bridges: {archive} does not carry "
                f"{len(missing)} declared artifact(s) for {platform}: "
                + ", ".join(missing))
        for filename, artifact in sorted(wanted.items()):
            payload = bundle.read(filename)
            _validate_native_payload(platform, artifact, filename, payload)
            target = destination / filename
            target.write_bytes(payload)
            if artifact.kind == "executable" and os.name != "nt":
                target.chmod(target.stat().st_mode | 0o111)
            records.append({
                "artifact": artifact.name,
                "filename": filename,
                "kind": artifact.kind,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "platform": platform,
        "artifacts": records,
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def stage_from_native_dir(native_root: Path, platform: str,
                          *, destination: Path | None = None, incremental: bool = False) -> Path:
    """Stage artifacts produced by ``tools/build_darwin_native.py``.

    Each artifact is built into ``<native-root>/<artifact>/`` and carries an
    ``artifact.json`` receipt.  This path is useful for local native builds;
    it does not create release pins or replace the uploaded, pinned bundle
    path used by release assembly.
    """

    if platform != bridge_assets.DARWIN_PLATFORM:
        raise SystemExit("stage_wheel_bridges: --from-native-dir is Darwin-only")
    native_root = native_root.resolve(strict=True)
    destination = destination or bridges.packaged_bridge_dir()
    wanted = {
        artifact.name: (artifact, bridge_assets.artifact_filename(
            artifact, platform))
        for artifact in bridge_assets.BUNDLED_ARTIFACTS
    }
    plan: list[tuple[bridge_assets.BundledArtifact, Path, bytes | None]] = []
    native_revisions: list[tuple[bridge_assets.BundledArtifact, str]] = []
    existing = {}
    existing_records = {}
    replacements = set()
    if incremental:
        manifest_path = destination / MANIFEST_NAME
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise SystemExit(f'stage_wheel_bridges: cannot read {manifest_path}: {error}') from None
        if (not isinstance(manifest, dict) or manifest.get('schema') != MANIFEST_SCHEMA or manifest.get('platform') != platform
                or not isinstance(manifest.get('artifacts'), list)):
            raise SystemExit('stage_wheel_bridges: existing manifest schema or platform differs')
        for record in manifest['artifacts']:
            if not isinstance(record, dict):
                raise SystemExit('stage_wheel_bridges: invalid staged artifact record')
            name = record.get('artifact')
            if not isinstance(name, str) or name not in wanted or name in existing:
                raise SystemExit(f'stage_wheel_bridges: unknown or repeated staged artifact {name!r}')
            artifact, filename = wanted[name]
            path = destination / filename
            if (record.get('filename') != filename or record.get('kind') != artifact.kind
                    or not path.is_file() or path.stat().st_size != record.get('bytes')
                    or _sha256(path) != record.get('sha256')):
                raise SystemExit(f'stage_wheel_bridges: staged {name} differs from its manifest')
            payload = path.read_bytes()
            _validate_native_payload(platform, artifact, filename, payload)
            if (native_root / name / 'artifact.json').is_file():
                replacements.add(name)
            else:
                try:
                    bridge_assets.verify_contract_marker(name, path)
                except bridge_assets.BridgeAssetError as error:
                    raise SystemExit(f'stage_wheel_bridges: {error}') from None
                if not artifact.vendored:
                    revisions = bridge_assets.embedded_source_revisions(payload)
                    if len(revisions) != 1:
                        raise SystemExit(f'stage_wheel_bridges: staged {name} lacks one source revision')
                    native_revisions.append((artifact, revisions[0]))
            existing[name] = (artifact, path, None)
            existing_records[name] = record
        staged_files = {p.name for p in destination.iterdir() if p.is_file() and p.name != MANIFEST_NAME}
        if staged_files != {record['filename'] for record in manifest['artifacts']}:
            raise SystemExit('stage_wheel_bridges: staged files differ from the existing manifest')
    for name, (artifact, filename) in wanted.items():
        if name in existing and name not in replacements:
            plan.append(existing[name])
            continue
        source_dir = native_root / name
        receipt_path = source_dir / "artifact.json"
        payload_path = source_dir / filename
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            payload = payload_path.read_bytes()
        except (OSError, ValueError) as error:
            raise SystemExit(
                f"stage_wheel_bridges: cannot read native build for {name} "
                f"({receipt_path}, {payload_path}): {error}") from None
        if (not isinstance(receipt, dict)
                or receipt.get("schema") != "woof-darwin-native-artifact-v1"
                or receipt.get("platform") != platform
                or receipt.get("target") != "aarch64-apple-darwin"
                or receipt.get("artifact") != name
                or receipt.get("filename") != filename
                or receipt.get("minimum_os") != "14.0"
                or receipt.get("bytes") != len(payload)
                or receipt.get("sha256") != hashlib.sha256(payload).hexdigest()):
            raise SystemExit(
                f"stage_wheel_bridges: {receipt_path} does not describe the "
                f"expected {platform} artifact {filename!r}")
        checkout_revision = receipt.get("checkout_revision")
        if (not isinstance(checkout_revision, str)
                or len(checkout_revision) != 40
                or any(c not in "0123456789abcdef" for c in checkout_revision)):
            raise SystemExit(
                f"stage_wheel_bridges: {receipt_path} has no full checkout "
                "revision")
        if artifact.vendored:
            from tools.build_darwin_native import VENDORED_SOURCE_COMMITS
            if (name not in VENDORED_SOURCE_COMMITS or receipt.get("vendored") is not True
                    or receipt.get("vendored_source_commit") != VENDORED_SOURCE_COMMITS.get(name)):
                raise SystemExit(
                    f"stage_wheel_bridges: {receipt_path} has invalid vendored "
                    "source provenance")
        elif receipt.get("source_revision") != checkout_revision:
            raise SystemExit(
                f"stage_wheel_bridges: {receipt_path} source revision differs "
                "from the checkout revision")
        if not artifact.vendored:
            source_marker = (
                b"GPUWM_BRIDGE_SOURCE_REV=" + checkout_revision.encode("ascii"))
            if source_marker not in payload:
                raise SystemExit(
                    f"stage_wheel_bridges: {name} payload does not carry its "
                    f"recorded checkout revision {checkout_revision}")
            native_revisions.append((artifact, checkout_revision))
        _validate_native_payload(platform, artifact, filename, payload)
        try:
            bridge_assets.verify_contract_marker(name, payload_path)
        except bridge_assets.BridgeAssetError as error:
            raise SystemExit(f'stage_wheel_bridges: {error}') from None
        plan.append((artifact, payload_path, payload))

    current_revision = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True)
    if current_revision.returncode:
        raise SystemExit(
            "stage_wheel_bridges: cannot identify the wheel checkout revision: "
            + current_revision.stderr.strip())
    current_revision = current_revision.stdout.strip()
    checked_inputs: dict[tuple[str, str], str | None] = {}
    def input_difference(artifact, built_revision):
        inputs = bridge_assets.NATIVE_BUILD_INPUTS.get(artifact.crate)
        if inputs is None:
            return f"no build inputs are declared for {artifact.crate}"
        dirty = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain",
             "--untracked-files=all", "--", *inputs],
            capture_output=True, text=True)
        if dirty.returncode:
            return "git could not check the wheel checkout inputs: " + dirty.stderr.strip()
        if dirty.stdout.strip():
            return f"the wheel checkout has modified native build inputs for {artifact.crate}"
        if built_revision == current_revision:
            return None
        common_base = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "merge-base", built_revision, current_revision],
            capture_output=True, text=True)
        if common_base.returncode:
            return f"{built_revision} and {current_revision} have no common Git ancestor"
        base = common_base.stdout.strip()
        return (bridge_assets.native_input_difference(_REPO_ROOT, artifact.crate, base, built_revision)
                or bridge_assets.native_input_difference(_REPO_ROOT, artifact.crate, base, current_revision))

    for artifact, built_revision in native_revisions:
        key = (artifact.crate, built_revision)
        if key not in checked_inputs:
            checked_inputs[key] = input_difference(artifact, built_revision)
        reason = checked_inputs[key]
        if reason is not None:
            raise SystemExit(
                f"stage_wheel_bridges: {artifact.name} built at "
                f"{built_revision} cannot be reused at {current_revision}: "
                f"{reason}")

    destination = destination.resolve()
    purge_build_staging()
    if not incremental:
        if _is_packaged_destination(destination) and destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
    records = []
    for artifact, _source, payload in plan:
        if artifact.name in existing and artifact.name not in replacements:
            records.append(existing_records[artifact.name])
            continue
        filename = bridge_assets.artifact_filename(artifact, platform)
        target = destination / filename
        target.write_bytes(payload)
        if artifact.kind == "executable" and os.name != "nt":
            target.chmod(target.stat().st_mode | 0o111)
        records.append({
            "artifact": artifact.name,
            "filename": filename,
            "kind": artifact.kind,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "platform": platform,
        "artifacts": records,
    }
    manifest_path = destination / MANIFEST_NAME
    if incremental:
        temporary = destination / (MANIFEST_NAME + '.pending')
        temporary.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
    else:
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return destination

def _staged_count(destination: Path) -> str:
    """What was staged, as the count the destination's own record holds."""

    staged = len(json.loads(
        (destination / MANIFEST_NAME).read_text(encoding="utf-8"))["artifacts"])
    declared = len(bridge_assets.BUNDLED_ARTIFACTS)
    if staged == declared:
        return f"{staged} artifacts"
    return f"{staged} of {declared} artifacts"

def clean(destination: Path | None = None) -> None:
    purge_build_staging()
    destination = destination or bridges.packaged_bridge_dir()
    if destination.exists():
        shutil.rmtree(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage built Rust artifacts into the package for a "
                    "platform wheel.")
    parser.add_argument(
        "--platform", choices=bridge_assets.SUPPORTED_PLATFORMS,
        default=bridge_assets.host_platform(),
        help="platform key the artifacts were built for (default: this host)")
    parser.add_argument(
        "--from-bundle", type=Path, default=None, metavar="ARCHIVE",
        help="stage from a packed release bundle rather than a build tree "
             "(requires --platform)")
    parser.add_argument(
        "--from-native-dir", type=Path, default=None, metavar="DIR",
        help="stage the complete set of build_darwin_native.py outputs from "
             "DIR/<artifact>/ (Darwin only; does not create release pins)")
    parser.add_argument('--incremental', action='store_true',
                        help='preserve validated staged artifacts and add or replace native builds')
    parser.add_argument(
        "--dest", type=Path, default=None, metavar="DIR",
        help="stage into DIR instead of the in-package directory.  "
             f"{bridges.default_bridge_dir()} is the one woof searches "
             "last, so staging there completes a checkout's own home "
             "from a build tree -- every declared artifact, driven from "
             "the same table the wheel and the release bundle use.  Only "
             "the in-package destination is emptied first; anywhere else "
             "keeps what this tool did not stage")
    parser.add_argument(
        "--clean", action="store_true",
        help="remove the staged directory and exit (restores a pure wheel)")
    args = parser.parse_args(argv)

    if args.incremental and (args.from_native_dir is None or args.clean):
        raise SystemExit('--incremental requires --from-native-dir and cannot be combined with --clean')

    if args.clean:
        if args.dest is not None:
            raise SystemExit(
                "--clean removes its destination entirely, so it is "
                "restricted to the in-package directory this tool owns.  "
                f"{args.dest} may hold the renderer's assets/ tree and "
                "copies nobody here staged; delete those deliberately, "
                "not as a flag combination")
        clean()
        print(f"removed {bridges.packaged_bridge_dir()}")
        return 0
    if args.from_bundle is not None and args.from_native_dir is not None:
        raise SystemExit("choose only one of --from-bundle and --from-native-dir")
    if args.from_native_dir is not None:
        if args.platform != bridge_assets.DARWIN_PLATFORM:
            raise SystemExit("--from-native-dir requires --platform darwin-arm64")
        destination = stage_from_native_dir(
            args.from_native_dir, args.platform, destination=args.dest, incremental=args.incremental)
        print(f"staged {len(bridge_assets.BUNDLED_ARTIFACTS)} Darwin artifacts "
              f"from native builds in {args.from_native_dir} into {destination}")
        return 0
    if args.from_bundle is not None:
        if args.platform is None:
            raise SystemExit(
                "--from-bundle needs --platform: a bundle's platform is a "
                "property of the release that built it, not of this host")
        destination = stage_from_bundle(args.from_bundle, args.platform,
                                        destination=args.dest)
        total = sum(path.stat().st_size for path in destination.iterdir()
                    if path.is_file())
        print(f"staged {_staged_count(destination)} for "
              f"{args.platform} from {args.from_bundle} into {destination} "
              f"({total / 1e6:.2f} MB)")
        return 0
    if args.platform is None:
        raise SystemExit(
            "no bundle is published for this host "
            f"({bridge_assets.host_platform_description()}); pass --platform "
            f"explicitly to cross-stage, or build from source.  Known: "
            f"{', '.join(bridge_assets.SUPPORTED_PLATFORMS)}")
    destination = stage(args.platform, destination=args.dest)
    total = sum(path.stat().st_size for path in destination.iterdir()
                if path.is_file())
    print(f"staged {_staged_count(destination)} for "
          f"{args.platform} into {destination} ({total / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
