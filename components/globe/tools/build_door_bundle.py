"""Build this package's door bundle and the pins the wheel carries.

Two steps, run in this order by the release workflow and reproducible by
hand from a checkout of the engine's source tree:

``pack``
    On each target platform, after ``cargo build --release --locked
    --offline`` in the engine's ``tools/rustwx`` workspace, collect every
    door of ``woof.globe.doors.doors_from_bundle("woof global")`` into one
    zip named for the release and the platform.  The archive is
    deterministic: binaries in the declared door order, fixed member
    timestamps, no directory entries, so two packs of the same bytes produce
    the same archive.

``pin``
    On one machine, from the bundles ``pack`` produced, compute the size and
    SHA-256 of every bundle and of every binary inside it and write them into
    ``src/arwen_global/data/door-pins.json`` BEFORE the wheel is built.

``notice``
    From the engine workspace's lockfile (``cargo metadata --locked
    --offline``, filtered to the platform's target triple), the licence
    notice the bundle carries as ``THIRD-PARTY-LICENSES.txt``.  ``pack``
    writes it into the zip when given ``--workspace``, and ``pin`` refuses a
    bundle without it.  THE BREAKAGE THIS PREVENTS, measured on 0.1.1 as
    published: both zips carried eight statically linked binaries and no
    licence text for any of the Rust crates inside them, while MIT, BSD,
    Apache-2.0, ISC and the Unicode licence each condition redistribution in
    binary form on the notice travelling with it.  The notice names the
    target triple the doors were built for, which for a cross-built bundle
    is the ``--triple`` given rather than the platform's native one.

``scan``
    Every member of a bundle, read as bytes with the NULs stripped (so a
    UTF-16 or padded string reads as the ASCII it spells), searched for a
    build machine's paths: a home directory on Linux, macOS or Windows and
    the CI runner's work tree.  ``pin`` runs the same scan and refuses a
    bundle it finds anything in.  THE BREAKAGE THIS PREVENTS, measured on
    0.1.1 as published: the door binaries embedded about 1,100 paths under
    the Linux build host's home directory and about 1,250 under the Windows
    one, because rustc writes the source path of every panic location and
    debug record into the binary.  A build with ``--remap-path-prefix`` for
    the engine checkout and the Cargo home writes neutral prefixes instead,
    and the release workflow sets it.

Nothing here invents a hash: every number written comes from hashing bytes
that exist on this disk at the moment it runs.  ``pin`` refuses a bundle
whose members are not exactly the door set this package expects for that
platform, so a half-built bundle cannot be pinned into a wheel.

``pin`` also refuses a STALE bundle, and does it two ways, because hashing
binds a wheel to exact bytes and says nothing about which source produced
them.  The engine's own cut nearly shipped bundles predating its source tip
once, with every check passing because every check hashed what it was handed.

*   Every door built from the engine's workspace embeds
    ``GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`` at build time.
    ``pin --source-rev COMMIT`` -- a required argument, so no cut can skip it
    -- extracts the stamp from every member and refuses a bundle whose stamp
    is absent, unparseable, ambiguous, or names any other commit.  String
    extraction, never execution: it works on the other platform's binaries
    and on a runner with no GPU.

*   Every door also carries the contract literal its ``Door`` row declares,
    which is what makes "this is the door the Python half was written
    against" a property of the bytes rather than of a filename.

``--allow-unstamped DOOR`` accepts a door whose stamp the build dropped,
and prints it as NOT CHECKED.  Through 0.1.1 every cut named ``rw_atms``
there: its crate declared the stamp as a ``pub static`` nothing read, so
the linker dropped it.  The engine's copy of the crate reads it once in
``main`` since the door crates were offered to its 2.8 line, so a bundle
built from that line needs no exemption and the release workflow passes
none.  The flag stays for a bundle rebuilt from an older revision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"


def _load_door_table():
    """Load the door table from source, without importing the package.

    ``import woof.globe.doors`` would run the package's ``__init__``, which
    pulls in the model and therefore the engine.  This tool runs on a build
    runner that has a Rust toolchain and no woof install, and it needs one
    table: the doors, their filenames, their contract literals.  Loading the
    module by path keeps the table the single source it already is without
    making the bundle build depend on the whole import graph.
    """

    import importlib.util

    path = SOURCE_ROOT / "arwen_global" / "doors.py"
    spec = importlib.util.spec_from_file_location(
        "_arwen_global_doors_table", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


door_table = _load_door_table()

#: Fixed member timestamp so two packs of identical bytes produce identical
#: archives (a zip stores an mtime per member).
_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)

#: The byte marker every door built from the engine's workspace embeds.
SOURCE_REV_MARKER = b"GPUWM_BRIDGE_SOURCE_REV="
_SOURCE_REV_LENGTH = 40
_STAMP = re.compile(
    SOURCE_REV_MARKER + b"([0-9a-f]{%d})" % _SOURCE_REV_LENGTH)

PINS_PATH = REPO_ROOT / "src" / "arwen_global" / "data" / "door-pins.json"

#: The member every bundle carries beside the doors: the licence notice for
#: every crate linked into them.  Written by ``notice``, required by ``pin``.
NOTICE_MEMBER = "THIRD-PARTY-LICENSES.txt"

#: The Rust target each published platform is built for.  The notice is
#: resolved per target because the dependency closure is: a Windows door
#: links windows-sys and its import libraries, a Linux door links neither.
TARGET_TRIPLES = {
    "linux-x86_64": "x86_64-unknown-linux-gnu",
    "win-x86_64": "x86_64-pc-windows-msvc",
}

#: A build machine's paths, as they land in a binary: rustc records the
#: source path of every panic location, and a dependency's build script can
#: record more.  Matched on the NUL-stripped bytes so a wide string reads as
#: the text it spells.  `/build/` is what the release workflow remaps to and
#: is not in this list; `/rustc/<commit>/` is the toolchain's own remapped
#: standard library and is not either.
BUILD_PATH_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("a Linux home directory", re.compile(rb"/home/[A-Za-z0-9._-]+/")),
    ("a macOS home directory", re.compile(rb"/Users/[A-Za-z0-9._-]+/")),
    ("a Windows home directory",
     re.compile(rb"(?i)[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9._ -]+[\\/]")),
    ("the root account's home", re.compile(rb"/root/\.(?:cargo|rustup)/")),
    ("a GitHub runner work tree",
     re.compile(rb"(?i)(?:[A-Z]:[\\/]+a[\\/]+|/ho" rb"me/runner/work/)")),
)


def scan_payload(payload: bytes) -> dict[str, tuple[int, str]]:
    """What build-machine paths a member carries: kind -> (count, first)."""

    text = payload.replace(b"\x00", b"")
    found: dict[str, tuple[int, str]] = {}
    for what, pattern in BUILD_PATH_PATTERNS:
        hits = pattern.findall(text)
        if hits:
            first = pattern.search(text)
            sample = text[first.start():first.start() + 80]
            found[what] = (len(hits), sample.decode("ascii", "replace"))
    return found


def scan(archives: list[Path]) -> int:
    """Print every member that carries a build path; 1 if any does."""

    dirty = 0
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                found = scan_payload(zf.read(name))
                if not found:
                    print(f"clean  {archive.name}: {name}")
                    continue
                dirty += 1
                for what, (count, sample) in sorted(found.items()):
                    print(f"DIRTY  {archive.name}: {name}: {count:,} x "
                          f"{what}, e.g. {sample!r}")
    return 1 if dirty else 0


#: File names that carry a licence, a copyright line or a notice a licence
#: asks to travel.  Matched case-insensitively on the name's start.
_LICENCE_FILE = re.compile(
    r"^(licen[cs]e|copying|notice|unlicense|copyright|authors)", re.I)


def _companion_doors():
    return door_table.doors_from_bundle(door_table.COMPANION_BUNDLE)


def _locate(name: str, search: list[Path]) -> Path:
    for directory in search:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"build_door_bundle: {name} is in none of the search directories: "
        + ", ".join(str(d) for d in search))


def pack(release: str, platform: str, search: list[Path],
         out_dir: Path, workspace: Path | None = None,
         triple: str | None = None) -> Path:
    if platform not in door_table.SUPPORTED_PLATFORMS:
        raise SystemExit(
            f"build_door_bundle: unknown platform {platform!r}; known: "
            + ", ".join(door_table.SUPPORTED_PLATFORMS))
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / door_table.bundle_filename(release, platform)
    sources = [
        (door_table.artifact_filename(door.name, platform),
         _locate(door_table.artifact_filename(door.name, platform), search))
        for door in _companion_doors()]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, source in sources:
            info = zipfile.ZipInfo(name, date_time=_FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            # 0o755 in the high half of external_attr: the unix mode a zip
            # can carry.  Staging chmods anyway, but an operator who unzips
            # by hand should get executables.
            info.external_attr = (0o100755 << 16)
            zf.writestr(info, source.read_bytes())
        if workspace is not None:
            info = zipfile.ZipInfo(NOTICE_MEMBER, date_time=_FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100644 << 16)
            zf.writestr(info, build_notice(workspace, platform, triple)
                        .encode("utf-8"))
    print(f"build_door_bundle: packed {archive} "
          f"({archive.stat().st_size:,} B) from {len(sources)} doors")
    for name, source in sources:
        print(f"  {name} <- {source}")
    return archive


def _cargo_metadata(workspace: Path, triple: str) -> dict:
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked",
         "--offline", "--filter-platform", triple],
        cwd=workspace, capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise SystemExit(
            f"build_door_bundle: cargo metadata failed in {workspace} for "
            f"{triple}:\n{out.stderr.strip()}")
    return json.loads(out.stdout)


def _licence_files(directory: Path) -> list[Path]:
    return sorted(
        (p for p in directory.iterdir()
         if p.is_file() and _LICENCE_FILE.match(p.name)),
        key=lambda p: p.name)


def _first_party_notices(package_dir: Path) -> list[Path]:
    """Notices a workspace crate carries for third-party data inside it.

    A first-party crate can carry somebody else's work (the BUFR master
    tables inside rw-obs are ECMWF's ecCodes transcription of the WMO
    tables, under Apache-2.0); that notice sits beside the data, not at the
    crate root, and no dependency walk finds it.
    """

    return [path for path in sorted(package_dir.rglob("*"))
            if path.is_file() and path.name.upper().startswith("NOTICE")
            and path.parent != package_dir and "target" not in path.parts]


def build_notice(workspace: Path, platform: str,
                 triple: str | None = None) -> str:
    """The third-party notice for this platform's doors, from the lockfile.

    Walks the resolved dependency graph from every crate that builds a door
    of this bundle, following normal dependencies only (a build dependency
    runs on the build machine and a dev dependency only in tests; neither
    is linked into a door), and emits each package's name, version, licence
    expression and every licence text found in its source directory,
    deduplicated by SHA-256 so no distinct copyright line is dropped.

    ``triple`` names the target the doors were actually built for when it
    is not the platform's default (a Windows bundle cross-built for
    ``x86_64-pc-windows-gnu`` links the gnu import crates, not the msvc
    ones), so the notice lists what is linked rather than what a native
    build would have linked.
    """

    if platform not in TARGET_TRIPLES:
        raise SystemExit(
            f"build_door_bundle: no target triple for platform {platform!r}")
    # The architecture AND the operating system must be the platform's: a
    # triple for another system resolves another dependency set (the
    # Windows import crates are not in a Linux graph), and the notice would
    # list crates the binaries beside it do not link.
    native = TARGET_TRIPLES[platform].split("-")
    if triple is not None and (triple.split("-")[0] != native[0]
                               or f"-{native[2]}" not in triple):
        raise SystemExit(
            f"build_door_bundle: {triple} is not a {platform} target")
    workspace = workspace.resolve()
    target = triple or TARGET_TRIPLES[platform]
    meta = _cargo_metadata(workspace, target)
    packages = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    engine_root = workspace.parent.parent
    crate_dirs = {
        (engine_root / door.crate).resolve()
        for door in _companion_doors()
        if door.crate.startswith("tools/rustwx/")}
    roots = [pid for pid, p in packages.items()
             if Path(p["manifest_path"]).resolve().parent in crate_dirs]
    if len(roots) != len(crate_dirs):
        raise SystemExit(
            "build_door_bundle: the workspace does not declare every crate "
            f"the doors are built from: {sorted(str(d) for d in crate_dirs)}")
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        for dep in nodes[pid]["deps"]:
            if any(kind.get("kind") is None for kind in dep["dep_kinds"]):
                stack.append(dep["pkg"])
    import tomllib
    workspace_licence = (
        tomllib.loads((workspace / "Cargo.toml").read_text(encoding="utf-8"))
        .get("workspace", {}).get("package", {}).get("license", ""))
    rows = []
    texts: dict[str, tuple[str, list[str]]] = {}
    unlicensed = []
    for pid in sorted(seen, key=lambda i: (packages[i]["name"],
                                           packages[i]["version"])):
        package = packages[pid]
        directory = Path(package["manifest_path"]).parent
        first_party = package.get("source") is None
        files = _licence_files(directory)
        if first_party:
            if not files:
                files = _licence_files(workspace)
            files = files + _first_party_notices(directory)
        expression = package.get("license") or ""
        if not expression and package.get("license_file"):
            expression = "see licence file"
        if not expression and first_party and workspace_licence:
            # A path crate built from the engine tree with no licence field
            # of its own (the GRIB decoder is exported from the same upstream
            # workspace and carries its declaration in a provenance note):
            # the workspace's own declared licence and text apply.
            expression = f"{workspace_licence} (workspace licence)"
        label = f"{package['name']} {package['version']}"
        if not expression and not files:
            unlicensed.append(label)
        for path in files:
            body = path.read_text(encoding="utf-8", errors="replace")
            body = body.replace("\r\n", "\n").strip() + "\n"
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            texts.setdefault(digest, (body, []))[1].append(
                f"{label}: {path.name}")
        rows.append((package["name"], package["version"],
                     expression or "(no expression)",
                     "engine tree" if first_party else "crates.io",
                     len(files)))
    if unlicensed:
        raise SystemExit(
            "build_door_bundle: these crates declare no licence and carry no "
            "licence file, so the notice cannot say on what terms they are "
            "redistributed: " + ", ".join(unlicensed))
    lines = [
        "THIRD-PARTY LICENCES FOR THE ARWEN GLOBAL DOOR BUNDLE",
        "=" * 72,
        "",
        f"Platform: {platform} ({target})",
        "",
        "The executables in this archive are statically linked Rust programs.",
        "Each links the crates listed below, resolved from the engine",
        "workspace's Cargo.lock for this target (normal dependencies only:",
        "build and test dependencies are not linked into a door).  Their",
        "licences condition redistribution in binary form on their notices",
        "travelling with the binaries; this file is that notice.  Section 1",
        "lists every crate with its licence expression; section 2 reproduces",
        "every distinct licence and notice text found in those crates, each",
        "headed by the crates it came from.  The crates marked 'engine",
        "tree' are built from source inside the engine repository (its own",
        "crates under the workspace's MIT licence, and the path dependencies",
        "it vendors under theirs), with any third-party data they carry",
        "listed under its own notice.",
        "",
        "1. CRATES",
        "-" * 72,
        "",
    ]
    for name, version, expression, origin, count in rows:
        plural = "" if count == 1 else "s"
        lines.append(f"{name} {version}  [{origin}]  {expression}  "
                     f"({count} text{plural})")
    bare = [(n, v, e) for n, v, e, _, c in rows if c == 0]
    lines += ["", f"{len(rows)} crates.", ""]
    if bare:
        lines += [
            "Crates published without a licence text in the package: the",
            "standard text of the licence their expression names applies, and",
            "for each licence named below that text is reproduced in section 2",
            "under another crate that ships it.",
            "",
        ]
        lines += [f"  {n} {v}  {e}" for n, v, e in bare]
        lines.append("")
    lines += ["2. TEXTS", "-" * 72, ""]
    for digest in sorted(texts, key=lambda d: (texts[d][1][0], d)):
        body, owners = texts[digest]
        lines.append("=" * 72)
        for owner in owners:
            lines.append(f"From {owner}")
        lines.append(f"SHA-256 {digest}")
        lines.append("=" * 72)
        lines.append("")
        lines.append(body.rstrip("\n"))
        lines.append("")
    return "\n".join(lines) + "\n"


def notice(workspace: Path, platform: str, out: Path,
           triple: str | None = None) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(build_notice(workspace, platform, triple).encode("utf-8"))
    print(f"build_door_bundle: wrote {out} ({out.stat().st_size:,} B)")
    return out


def _platform_of(archive: Path, release: str) -> str:
    for platform in door_table.SUPPORTED_PLATFORMS:
        if archive.name == door_table.bundle_filename(release, platform):
            return platform
    raise SystemExit(
        f"build_door_bundle: {archive.name} is not a bundle name for release "
        f"{release}; expected one of "
        + ", ".join(door_table.bundle_filename(release, p)
                    for p in door_table.SUPPORTED_PLATFORMS))


def _verify_source_revision(payload: bytes, expected: str, label: str) -> None:
    found = sorted({match.decode("ascii") for match in _STAMP.findall(payload)})
    if not found:
        raise SystemExit(
            f"build_door_bundle: {label} carries no "
            f"{SOURCE_REV_MARKER.decode()}<commit> stamp, so nothing in the "
            "bytes says which source produced them; rebuild it from the "
            "engine checkout with GPUWM_BRIDGE_SOURCE_REV set, or name it in "
            "--allow-unstamped if the crate is known not to keep its stamp")
    if len(found) > 1:
        raise SystemExit(
            f"build_door_bundle: {label} carries more than one source stamp "
            f"({', '.join(found)}); refusing to pin a bundle whose provenance "
            "is ambiguous")
    if found[0] != expected:
        raise SystemExit(
            f"build_door_bundle: {label} was built from {found[0]} and this "
            f"cut declares {expected}; refusing to pin a stale door")


def _pin_bundle(archive: Path, release: str, source_rev: str,
                unstamped: set[str]) -> tuple[str, dict]:
    platform = _platform_of(archive, release)
    expected = [(door, door_table.artifact_filename(door.name, platform))
                for door in _companion_doors()]
    with zipfile.ZipFile(archive) as zf:
        held = set(zf.namelist())
        missing = [name for _, name in expected if name not in held]
        if missing:
            raise SystemExit(
                f"build_door_bundle: {archive.name} is missing "
                f"{', '.join(missing)}; refusing to pin a partial bundle")
        if NOTICE_MEMBER not in held:
            raise SystemExit(
                f"build_door_bundle: {archive.name} carries no "
                f"{NOTICE_MEMBER}; refusing to pin statically linked "
                "binaries whose crates' licences travel with none of them "
                "(pack with --workspace writes it)")
        extra = sorted(held - {name for _, name in expected} - {NOTICE_MEMBER})
        if extra:
            raise SystemExit(
                f"build_door_bundle: {archive.name} carries members this "
                f"package does not publish ({', '.join(extra)}); refusing to "
                "pin a bundle whose contents are not the door set")
        binaries = []
        for door, name in expected:
            payload = zf.read(name)
            leaked = scan_payload(payload)
            if leaked:
                raise SystemExit(
                    f"build_door_bundle: {archive.name}: {name} carries "
                    + "; ".join(f"{count:,} x {what} (e.g. {sample!r})"
                                for what, (count, sample)
                                in sorted(leaked.items()))
                    + ".  Rebuild with --remap-path-prefix for the engine "
                    "checkout and the Cargo home, as the release workflow does")
            if door.name in unstamped:
                print(f"  {name}: source stamp NOT CHECKED "
                      f"(--allow-unstamped {door.name})")
            else:
                _verify_source_revision(
                    payload, source_rev, f"{archive.name}: {name}")
            if door.marker is not None and door.marker not in payload:
                raise SystemExit(
                    f"build_door_bundle: {archive.name}: {name} does not "
                    f"carry the contract literal "
                    f"{door.marker.decode(errors='replace')!r}, so it is not "
                    "the door this package's Python half was written against; "
                    f"rebuild it from {door.crate}")
            binaries.append({
                "artifact": door.name,
                "filename": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
        notice_bytes = zf.read(NOTICE_MEMBER)
    record = {
        "notice": {
            "filename": NOTICE_MEMBER,
            "bytes": len(notice_bytes),
            "sha256": hashlib.sha256(notice_bytes).hexdigest(),
        },
        "bundle": {
            "filename": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": door_table.sha256_file(archive),
        },
        "binaries": binaries,
    }
    return platform, record


def pin(release: str, archives: list[Path], out: Path, source_rev: str,
        unstamped: set[str], engine_rev: str | None) -> Path:
    platforms: dict[str, dict] = {}
    for archive in archives:
        platform, record = _pin_bundle(archive, release, source_rev, unstamped)
        if platform in platforms:
            raise SystemExit(f"build_door_bundle: two bundles for {platform}")
        platforms[platform] = record
    existing = json.loads(out.read_text(encoding="utf-8"))
    document = {
        "schema": door_table.DOOR_PINS_SCHEMA,
        "release": release,
        "engine_source_rev": engine_rev or source_rev,
        "note": existing.get("note"),
        "platforms": platforms,
    }
    if unstamped:
        document["unstamped"] = sorted(unstamped)
    out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"build_door_bundle: wrote {out} for "
          f"{', '.join(sorted(platforms))} at release {release}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_door_bundle",
        description="pack and pin the door bundle this package publishes")
    sub = parser.add_subparsers(dest="command", required=True)

    packer = sub.add_parser("pack", help="build one platform's bundle")
    packer.add_argument("--release", required=True,
                        help="the release tag, e.g. v0.1.0")
    packer.add_argument("--platform", required=True,
                        choices=door_table.SUPPORTED_PLATFORMS)
    packer.add_argument("--search", required=True, nargs="+", type=Path,
                        help="directories holding the built doors")
    packer.add_argument("--out", required=True, type=Path)
    packer.add_argument("--workspace", type=Path, default=None,
                        help="the engine's tools/rustwx workspace; writes "
                             f"{NOTICE_MEMBER} into the bundle from its "
                             "lockfile (pin refuses a bundle without it)")
    packer.add_argument("--triple", default=None,
                        help="the Rust target the doors were built for, when "
                             "it is not the platform's default; the notice "
                             "is resolved for it")

    scanner = sub.add_parser(
        "scan", help="report build-machine paths inside bundle members")
    scanner.add_argument("archives", nargs="+", type=Path)

    noticer = sub.add_parser(
        "notice", help="write the third-party notice for one platform")
    noticer.add_argument("--workspace", required=True, type=Path)
    noticer.add_argument("--platform", required=True,
                         choices=door_table.SUPPORTED_PLATFORMS)
    noticer.add_argument("--out", required=True, type=Path)
    noticer.add_argument("--triple", default=None,
                         help="the Rust target, when not the platform's default")

    pinner = sub.add_parser("pin", help="write the pins from packed bundles")
    pinner.add_argument("--release", required=True)
    pinner.add_argument("--source-rev", required=True,
                        help="the 40-hex engine commit every door was built "
                             "from; a bundle whose stamps disagree is refused")
    pinner.add_argument("--engine-rev", default=None,
                        help="recorded in the pins as the engine revision the "
                             "crates came from (defaults to --source-rev)")
    pinner.add_argument("--allow-unstamped", action="append", default=[],
                        metavar="DOOR",
                        help="doors whose build is known not to keep the "
                             "source-revision stamp; each one is printed as "
                             "NOT CHECKED")
    pinner.add_argument("--out", type=Path, default=PINS_PATH)
    pinner.add_argument("archives", nargs="+", type=Path)

    args = parser.parse_args(argv)
    if args.command == "pack":
        pack(args.release, args.platform, list(args.search), args.out,
             args.workspace, args.triple)
        return 0
    if args.command == "scan":
        return scan(list(args.archives))
    if args.command == "notice":
        notice(args.workspace, args.platform, args.out, args.triple)
        return 0
    unstamped = set(args.allow_unstamped or ())
    unknown = sorted(unstamped - {d.name for d in _companion_doors()})
    if unknown:
        raise SystemExit(
            f"build_door_bundle: --allow-unstamped names doors this package "
            f"does not publish: {', '.join(unknown)}")
    pin(args.release, list(args.archives), args.out, args.source_rev,
        unstamped, args.engine_rev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
