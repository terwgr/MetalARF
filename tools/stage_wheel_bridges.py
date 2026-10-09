"""Stage the built Rust artifacts INTO the package, for a platform wheel.

``pip install recast-woof`` used to land a Python half with no compiled Rust at
all: every decoder, the fetch backbone, the renderer and the radar front
door were downloaded afterwards by ``woof setup`` / ``woof
fetch-bridges``.  That made the post-install step a failure surface on
the one command every user runs first, and it made "the decode runs in
Rust" unreachable on a bare install -- which is the whole point of doing
the decode in Rust.

This tool closes that by copying the artifacts
:data:`woof.bridge_assets.BUNDLED_ARTIFACTS` declares out of the two
Cargo workspaces and into :func:`woof.bridges.packaged_bridge_dir`
(``woof/libexec/bridges``), which
``[tool.setuptools.package-data]`` sweeps into the wheel.  Run it
between ``cargo build --release`` and ``python -m build``; the resulting
wheel is platform-tagged by ``setup.py`` because it now carries native
code.

It is deliberately NOT a build step: it copies bytes that already exist
and refuses when they do not, so a wheel can never be built with a
silently missing artifact and ship a door that dies at first use.

    python tools/stage_wheel_bridges.py --platform win-x86_64
    python tools/stage_wheel_bridges.py --clean

``--dest DIR`` points the same table-driven copy at a directory other
than the package, which is what completes a HOME from a build tree::

    python tools/stage_wheel_bridges.py --dest ~/.woof/bridges

That rung -- ``woof.bridges.default_bridge_dir`` -- is the last one
every resolver searches, and it is the rung a wheel-less checkout user
fills.  Filling it by hand is how a home came to carry sixteen of the
declared artifacts, missing the NetCDF decoder and the NetCDF writer
behind the default wrfout engine; a hand copy has no declaration to
check itself against and this tool does.  Only the in-package
destination is emptied first (a wheel carrying two platforms is both
wrong and over PyPI's cap); any other destination keeps everything this
tool did not stage.

What is NOT staged, and why: the renderer's basemap shapefiles.  They
are 21.1 MB compressed, and there is no room for them.  They are also
DATA (Natural Earth coastlines and county polygons), not a processing
path.  Since 2.8.0 they ship in the ``recast-woof-data`` companion, which
every install pulls, and :func:`woof.rustwx.renderer_env` hands that
copy to this wheel's renderer; before that they reached a wheel install
only through ``woof fetch-bridges``, which no install text ran, and
every picture came out with no coastlines, borders or state lines.

Sizes, measured 2026-08-17 at the 2.5.0 integration tip, against
PyPI's 100 MB per-file cap read strictly as 100,000,000 B::

    pure wheel      gpuwm-2.5.0-py3-none-any.whl          91,926,618 B
    pure sdist      gpuwm-2.5.0.tar.gz                    95,245,042 B
    win-x86_64      gpuwm-2.5.0-py3-none-win_amd64.whl   108,695,109 B
    linux-x86_64    ...-manylinux_2_28_x86_64.whl        111,697,820 B

The pure pair is what ``.github/workflows/publish.yml`` ships (it runs
``--clean`` first) and it fits, with 8.07 MB of headroom on the wheel
and 4.75 MB on the sdist -- the sdist, not the wheel, is the binding
constraint of the published pair.  The win and linux numbers are this
Windows host and a development machine respectively; the linux one is from the
2.5.0 Linux shakeout, everything else was measured together on one box.

BOTH PLATFORM WHEELS ARE OVER THE CAP AT THIS TIP -- by 8.70 MB and
11.70 MB.  The bridge payload has roughly doubled since these numbers
were last written (compressed, it is now 16.77 MB on Windows and 19.77
MB on Linux over the pure base), which is ``rw_netcdf``,
``netcdf_writer`` and ``region_global_dealias`` joining
:data:`woof.bridge_assets.BUNDLED_ARTIFACTS`.  Whatever this docstring
claimed before, do not read a headroom number off it again without
re-measuring: the figures it carried into this tip (96.66 MB win, 98.35
MB linux, "1.65 MB of headroom on Linux") were 12-13 MB low and had the
sign inverted -- the wheels were over the cap while the doc said they
were under it, and the whole documented rationale for excluding the
basemaps rested on that dead number.

Nothing published is broken by this: the per-platform wheel is not a
shape PyPI receives today.  The remedy for making it one is the
companion split -- the payload moves into a companion distribution
rather than a decoder being dropped -- and until that lands, a platform
wheel built here is a local artifact, not an uploadable one.
``tools/verify_release_artifacts.py`` measures every distribution file
in the dist directory against the cap and refuses the ones a cut is
about to publish, so this cannot be discovered at upload time with the
tag already public.

67 MB of the pure base is physics tables (Thompson and RRTMGP).  If the
PURE pair ever runs out of room, the remedy is externalising another
table through ``woof fetch-tables`` -- the route ``freezeH2O.dat`` and
``qr_acr_qg_V4.dat`` already take -- rather than dropping a decoder.
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


def stage(platform: str, *, destination: Path | None = None) -> Path:
    """Copy every declared artifact into ``destination``, or refuse.

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
    missing: list[str] = []
    plan: list[tuple[bridge_assets.BundledArtifact, Path, Path]] = []
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        origin = source_path(artifact, platform)
        if not origin.is_file():
            missing.append(f"  {artifact.name:24s} expected at {origin}")
            continue
        plan.append((artifact, origin,
                     destination / bridge_assets.artifact_filename(artifact, platform)))
    if missing:
        raise SystemExit(
            f"stage_wheel_bridges: {len(missing)} of "
            f"{len(bridge_assets.BUNDLED_ARTIFACTS)} declared artifacts are "
            f"not built for {platform}:\n" + "\n".join(missing) +
            "\n\nBuild them first, from the checkout root:\n"
            + "\n".join(
                "  " + bridges.cargo_build_one_liner(crate)
                for crate in dict.fromkeys(
                    artifact.crate
                    for artifact in bridge_assets.BUNDLED_ARTIFACTS)))

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
    module docstring).
    """

    destination = destination or bridges.packaged_bridge_dir()
    wanted = {
        bridge_assets.artifact_filename(artifact, platform): artifact
        for artifact in bridge_assets.BUNDLED_ARTIFACTS
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
                          *, destination: Path | None = None) -> Path:
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
    plan: list[tuple[bridge_assets.BundledArtifact, Path, bytes]] = []
    native_revisions: list[tuple[bridge_assets.BundledArtifact, str]] = []
    for name, (artifact, filename) in wanted.items():
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
            if (receipt.get("vendored") is not True
                    or not isinstance(receipt.get("vendored_source_commit"), str)
                    or len(receipt["vendored_source_commit"]) != 40):
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
    for artifact, built_revision in native_revisions:
        if built_revision == current_revision:
            continue
        key = (artifact.crate, built_revision)
        if key not in checked_inputs:
            inputs = bridge_assets.NATIVE_BUILD_INPUTS.get(artifact.crate)
            if inputs is None:
                checked_inputs[key] = (
                    f"no build inputs are declared for {artifact.crate}")
                continue
            dirty = subprocess.run(
                ["git", "-C", str(_REPO_ROOT), "status", "--porcelain",
                 "--untracked-files=all", "--", *inputs],
                capture_output=True, text=True)
            if dirty.returncode:
                checked_inputs[key] = (
                    "git could not check the wheel checkout inputs: "
                    + dirty.stderr.strip())
                continue
            if dirty.stdout.strip():
                checked_inputs[key] = (
                    f"the wheel checkout has modified native build inputs for "
                    f"{artifact.crate}")
                continue
            common_base = subprocess.run(
                ["git", "-C", str(_REPO_ROOT), "merge-base",
                 built_revision, current_revision],
                capture_output=True, text=True)
            if common_base.returncode:
                checked_inputs[key] = (
                    f"{built_revision} and {current_revision} have no "
                    "common Git ancestor")
                continue
            base = common_base.stdout.strip()
            build_difference = bridge_assets.native_input_difference(
                _REPO_ROOT, artifact.crate, base, built_revision)
            checkout_difference = bridge_assets.native_input_difference(
                _REPO_ROOT, artifact.crate, base, current_revision)
            checked_inputs[key] = build_difference or checkout_difference
        reason = checked_inputs[key]
        if reason is not None:
            raise SystemExit(
                f"stage_wheel_bridges: {artifact.name} built at "
                f"{built_revision} cannot be reused at {current_revision}: "
                f"{reason}")

    destination = destination.resolve()
    purge_build_staging()
    if _is_packaged_destination(destination) and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for artifact, _source, payload in plan:
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
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return destination


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
            args.from_native_dir, args.platform, destination=args.dest)
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
        print(f"staged {len(bridge_assets.BUNDLED_ARTIFACTS)} artifacts for "
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
    print(f"staged {len(bridge_assets.BUNDLED_ARTIFACTS)} artifacts for "
          f"{args.platform} into {destination} ({total / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
