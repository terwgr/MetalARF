"""``woof global fetch-doors``: stage the Rust doors this package publishes.

The wheel carries no Rust.  Eight of the fourteen binaries this model runs on
are published by this package rather than by the engine, and before this
command exists the only way to obtain them is a Rust toolchain, a checkout of
the engine's source tree and a few minutes of compiling -- which by the rule
that a capability a user cannot reach does not exist means those eight doors
were not shipped at all.  This command is the same trade ``woof
fetch-bridges`` already makes: the binaries are published as versioned GitHub
release assets, their exact size and SHA-256 pins are packaged inside this
wheel at release time, and every byte is verified against those pins BEFORE
anything is installed.

WHAT IS STAGED, AND WHERE
One bundle per platform holding the doors of
``woof.globe.doors.companion_doors()``, staged into
:func:`woof.globe.doors.companion_door_dir` (``~/.woof/global-doors``).

Beside the engine's ``~/.woof/bridges``, not inside it, and that placement
is the whole reason this command is not simply a copy into woof's directory.
Two of the eight, ``rw_asos`` and ``rw_goes``, share a filename with a binary
the engine's own bundle publishes and the two builds are not the same door:
the engine's ``rw_asos`` has no ``networks``, ``table`` or ``awc``
subcommand, and the engine's ``rw_goes`` has no ``bt``, ``colocate``,
``quicklook`` or ``forward``, which is the entire ABI radiance leg.  Writing
this package's copies over the engine's would leave every ``woof`` command
resolving an executable whose ``--abi`` the engine's own doctor rejects.  So
each owner keeps its own copy, this package points its commands at its own
through the per-binary environment variables the engine's resolution ladder
already reads first, and ``woof global doctor`` prints which directory every
door came from -- so a second copy on the machine is a printed line rather
than a silent shadow.

THE STAGING CONTRACT
Every artifact is verified three ways before it is installed: the exact byte
count, the SHA-256 pin packaged with this release, and the contract literal
the door's current record shape compiles in.  A staged file that fails any of
the three is not written at all.  Members are read by their exact pinned
filename, so an archive cannot place a byte anywhere the pins did not name.

An existing file whose bytes do not match the pin is REPLACED rather than
refused.  A door is a versioned build, and yesterday's copy sitting in the
staging directory after a Python-half upgrade is exactly the skew this
command exists to end.  The replacement is still gated on the new bytes
passing all three checks first.

OFFLINE AND MIRRORS
``--from DIR`` stages from a local directory under identical verification:
either the bundle archive itself, or the binaries loose in that directory,
which is what an operator has after building them on a machine that does have
a toolchain.  ``ARWEN_GLOBAL_DOOR_URL_BASE`` overrides the download base URL;
the bundle filename is appended to it either way.

WHERE THE DOWNLOAD COMES FROM
The release assets of the repository this installed distribution's own
metadata names (its ``Repository`` project URL), never an address written
into the source.  A distribution that carries this package under another
name publishes from its own repository, and a literal here would send its
users to somebody else's releases.

WHEN THE ENGINE PUBLISHES EVERY DOOR
A door the installed engine's own bundle declares is the engine's door
(:func:`woof.globe.doors.publisher`).  When that is all of them this
package has nothing to stage, and the command says so and names the
engine's command instead of downloading a bundle nobody publishes.

PINS ARE GENERATED AT RELEASE TIME
The bundles are built by this repository's CI from the engine's ``tools/
rustwx`` workspace at the pinned engine revision, and ``tools/
build_door_bundle.py`` computes the pins from those exact bytes and writes
them into ``woof/globe/data/door-pins.json`` before the wheel is built.  A
tree that has not been through that step carries a pins document declaring no
platforms, and this command says so and refuses rather than inventing a hash
to check against.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ._version import CONSOLE_SCRIPT, DISTRIBUTION_NAME, __version__
from .doors import (
    DoorStagingError,
    artifact_filename,
    bundle_filename,
    companion_door_dir,
    companion_doors,
    companion_pins,
    current_platform,
    sha256_file,
    BUNDLE_NOTICE,
    stage_from_directory,
)

__all__ = ["add_fetch_doors_arguments", "fetch_doors", "asset_url_base"]

#: Override the download base URL (private mirrors, tests).  The bundle
#: filename is appended to it.
ASSET_URL_BASE_ENV = "ARWEN_GLOBAL_DOOR_URL_BASE"

#: The project-URL labels read, in order, for the repository whose release
#: assets carry the bundle.
_REPOSITORY_LABELS = ("repository", "source", "homepage")
_USER_AGENT = "gpuwm-global-fetch-doors/1"
_TIMEOUT_S = 120


def repository_url() -> str | None:
    """The repository this installed distribution's metadata names, or None.

    THE BREAKAGE THIS PREVENTS: the address used to be a literal in this
    file, so a distribution that carries this package under its own name
    and publishes from its own repository sent every ``fetch-doors`` to a
    different repository's releases.  The metadata is written from the
    same project file that names the repository on the package index page.
    """

    from importlib.metadata import PackageNotFoundError, metadata

    try:
        entries = metadata(DISTRIBUTION_NAME).get_all("Project-URL") or []
    except PackageNotFoundError:
        return None
    urls = {}
    for entry in entries:
        label, _, url = entry.partition(",")
        if url.strip():
            urls.setdefault(label.strip().lower(), url.strip().rstrip("/"))
    for label in _REPOSITORY_LABELS:
        if urls.get(label, "").startswith("https://"):
            return urls[label]
    return None


def asset_url_base(release: str) -> str:
    override = os.environ.get(ASSET_URL_BASE_ENV)
    if override:
        return override.rstrip("/")
    repository = repository_url()
    if repository is None:
        raise DoorStagingError(
            f"the installed {DISTRIBUTION_NAME} names no repository in its "
            "metadata, so there is no release to download the bundle from; "
            f"set {ASSET_URL_BASE_ENV} to the release asset base, or stage "
            "from a local bundle with --from")
    return f"{repository}/releases/download/{release}"


def _engine_publishes_everything() -> int:
    print(f"{CONSOLE_SCRIPT} fetch-doors: the installed engine's bundle "
          "publishes every Rust door this model runs on, so there is nothing "
          "for this command to stage.  Stage them with `woof fetch-bridges` "
          f"(an install whose wheel carries them needs neither); "
          f"`{CONSOLE_SCRIPT} doctor` grades each one against the engine's "
          "pins.")
    return 0


def add_fetch_doors_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--from", dest="source", metavar="DIR_OR_ZIP", type=Path,
        help="stage from a local bundle archive or a directory of built "
             "doors instead of downloading, under identical verification")
    parser.add_argument(
        "--dest", metavar="DIR", type=Path,
        # The resolved path is NOT interpolated here.  This help text is
        # published: tools/build_cli_reference.py renders it into
        # docs/CLI-REFERENCE.md, and on the machine that generated it the
        # interpolation expanded to the generating user's home directory,
        # which then travelled in the reference table as an instruction no
        # reader can follow.  `doctor` prints the resolved directory, which
        # is the right place for a machine-specific answer.
        help="stage somewhere other than the default companion door "
             "directory (woof global doctor prints the resolved path)")
    parser.add_argument(
        "--list", action="store_true",
        help="print what would be staged, and from where, without staging it")


def fetch_doors(args: argparse.Namespace) -> int:
    dest = args.dest or companion_door_dir()
    platform_key = current_platform()
    doors = companion_doors()
    if not doors:
        return _engine_publishes_everything()
    if platform_key is None:
        print(
            f"{CONSOLE_SCRIPT} fetch-doors: no door bundle is published for "
            f"{sys.platform} on this machine architecture.  The build-from-"
            "source route stays universal: build the crates below from the "
            "engine's tools/rustwx workspace and point the per-door "
            "environment variables at them.", file=sys.stderr)
        for door in doors:
            print(f"  {door.name:<12} {door.crate}  ({door.env_var})",
                  file=sys.stderr)
        return 3

    pins = companion_pins()
    record = pins.get("platforms", {}).get(platform_key)
    release = pins.get("release")

    if getattr(args, "list", False):
        print(f"platform      {platform_key}")
        print(f"destination   {dest}")
        print(f"release       {release or '(no bundle published yet)'}")
        for door in doors:
            filename = artifact_filename(door.name, platform_key)
            print(f"  {filename:<16} {door.crate}")
        return 0

    if not record:
        print(
            f"{CONSOLE_SCRIPT} fetch-doors: this build of the package carries "
            f"no pins for {platform_key}, so there is nothing to verify a "
            "download against and nothing will be downloaded.\n"
            "  A release cut writes the pins from the exact bytes it "
            "published; a tree that has not been through that step declares "
            "no platforms.\n"
            f"  Build the {len(doors)} doors from the engine's tools/rustwx "
            f"workspace and stage them with --from DIR, or install a release "
            "of this package.", file=sys.stderr)
        return 3

    if args.source is not None:
        source = args.source
        origin = f"local {source}"
        return _stage(source, dest, platform_key, origin, doors)

    filename = bundle_filename(release, platform_key)
    try:
        url = f"{asset_url_base(release)}/{filename}"
    except DoorStagingError as error:
        print(f"{CONSOLE_SCRIPT} fetch-doors: {error}", file=sys.stderr)
        return 3
    with tempfile.TemporaryDirectory(prefix="gpuwm-global-doors-") as scratch:
        archive = Path(scratch) / filename
        try:
            _download(url, archive)
        except DoorStagingError as error:
            print(f"{CONSOLE_SCRIPT} fetch-doors: {error}", file=sys.stderr)
            return 3
        expected = record.get("bundle", {})
        if expected:
            size = archive.stat().st_size
            if size != int(expected["bytes"]):
                print(f"{CONSOLE_SCRIPT} fetch-doors: {filename} downloaded "
                      f"{size:,} B and the pin says {int(expected['bytes']):,} "
                      "B; refusing it", file=sys.stderr)
                return 3
            digest = sha256_file(archive)
            if digest != expected["sha256"]:
                print(f"{CONSOLE_SCRIPT} fetch-doors: {filename} hashes to "
                      f"{digest} and the pin says {expected['sha256']}; "
                      "refusing it", file=sys.stderr)
                return 3
        return _stage(archive, dest, platform_key, url, doors)


def _stage(source: Path, dest: Path, platform_key: str, origin: str,
           doors) -> int:
    try:
        staged = stage_from_directory(source, dest, platform_key)
    except DoorStagingError as error:
        print(f"{CONSOLE_SCRIPT} fetch-doors: {error}", file=sys.stderr)
        return 3
    print(f"{CONSOLE_SCRIPT} fetch-doors: staged {len(staged)} doors into "
          f"{dest} from {origin}")
    for path in staged:
        print(f"  {path.name:<16} {path.stat().st_size:,} B")
    notice = dest / BUNDLE_NOTICE
    if notice.is_file():
        print(f"  licence notice for the crates inside them: {notice}")
    _report_what_outranks(dest, doors)
    return 0


def _report_what_outranks(dest: Path, doors) -> None:
    """Say what will resolve ahead of what was just staged, and only that.

    THE BREAKAGE THIS NAMES, measured 2026-09-07 on the Windows leg: the
    console script calls `doors.bind_companion_doors()` at the top of
    `main()`, which writes one `WOOF_RW_*` variable per companion door
    into this process's own environment.  So the second `fetch-doors` on
    any machine -- and every one after it -- read those back and told the
    operator that eight environment overrides they had never set were
    outranking the staging they had just performed.  The remedy that
    sentence implies, unset them, is not available: the program sets them
    itself, on every run, before the command starts.

    An override therefore counts as the OPERATOR'S only when it points
    somewhere other than the file this package's own binding would have
    produced.  That one is worth printing, because it really does win and
    the reader is the only one who can change it, so the line names the
    path it wins with.

    The other half is the destination.  `--dest` stages somewhere the
    resolution ladder does not look, so without ARWEN_GLOBAL_DOOR_DIR the
    copies just written are inert.  Saying so here is the difference
    between a flag that works and a flag that appears to.
    """

    resolved = companion_door_dir()
    foreign: list[str] = []
    for door in doors:
        value = os.environ.get(door.env_var)
        if not value:
            continue
        own = resolved / artifact_filename(door.name)
        try:
            same = Path(value).resolve() == own.resolve()
        except OSError:
            same = False
        if not same:
            foreign.append(f"{door.env_var}={value}")
    if foreign:
        print("  note: these overrides were set outside this program and "
              "resolve ahead of what was just staged:")
        for row in sorted(foreign):
            print(f"    {row}")
    try:
        elsewhere = dest.resolve() != resolved.resolve()
    except OSError:
        elsewhere = str(dest) != str(resolved)
    if elsewhere:
        print(f"  note: these doors are staged in {dest}, which is not the "
              f"directory this package resolves from ({resolved}).  Set "
              f"ARWEN_GLOBAL_DOOR_DIR={dest} to make them the copies that "
              "run, or re-stage without --dest.")


def _download(url: str, target: Path) -> None:
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=_TIMEOUT_S) as response, \
                open(target, "wb") as handle:
            shutil.copyfileobj(response, handle)
    except HTTPError as error:
        raise DoorStagingError(
            f"{url} returned HTTP {error.code}; the release may not carry "
            "this asset yet") from error
    except (URLError, OSError) as error:
        raise DoorStagingError(f"{url} could not be fetched: {error}") from error
