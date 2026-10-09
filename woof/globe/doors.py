"""The Rust doors this model runs on, named one row each.

Every data path in this package is Rust: observations are decoded by Rust
binaries, GRIB2 analyses by the mapped engine, statics by the static-field
library, NetCDF by the NetCDF bridge, and every weather-field picture by the
renderer.  Python here is the model, the orchestration and the CUDA driver
code; it decodes nothing.

That makes the set of binaries a contract rather than an implementation
detail, and this module is where it is written down once.  Two things read
it: :mod:`woof.globe.doctor`, which reports each row against what is
actually staged on the machine, and the release process, which builds the
rows this package's own bundle has to carry.

WHERE A BINARY COMES FROM.  Six of the fourteen ship in the engine's own
bridge bundle, which ``woof fetch-bridges`` stages into ``~/.woof/bridges``
and pins by size and SHA-256.  Eight do not, and the reason splits in two.

Six of the eight -- ``rw_atms``, ``rw_gnssro``, ``rw_ndbc``, ``rw_igra2``,
``rw_amv``, ``rw_wis2`` -- have no binary in any published bundle at all,
because the crates that build them live only on the branch this package was
carved from.

The other two, ``rw_asos`` and ``rw_goes``, are the ones worth reading
carefully, because a bundle that carries a binary of the right NAME is not
the same as a bundle that carries the door.  Both are in the engine's bundle
and neither can run this package's commands: the engine's ``rw_asos`` has
``stations``, ``fetch``, ``decode`` and ``verify`` and this package calls
``networks``, ``table`` and ``awc``; the engine's ``rw_goes`` has ``list``,
``fetch``, ``cwp``, ``cloud-top`` and ``verify`` and this package calls
``bt``, ``colocate``, ``quicklook`` and ``forward``, which is the whole ABI
radiance leg.  Measured 2026-09-07 against woof 2.7.0: the two ``--abi``
lines are proper prefixes of the ones this package was written against, so
the handshake in the table below catches both statically.  By the rule that
a capability a user cannot reach does not exist, those two are doors this
package publishes, not doors it inherits.

Those eight are this package's own bundle, staged by
``woof global fetch-doors`` into :func:`companion_door_dir`
(``~/.woof/global-doors``) and pinned by size and SHA-256 in
``woof/globe/data/door-pins.json``.

They stage into a directory of their own rather than into the engine's,
and this package points its own commands at them through the engine's
per-binary environment variables.  Writing a newer ``rw_asos`` into
``~/.woof/bridges`` would leave the engine resolving a binary whose
``--abi`` its own doctor rejects: one copy of a door per owner, each owner
resolving its own, and :mod:`woof.globe.doctor` printing which directory
every door came from so neither is shadowing the other in silence.

If the engine takes those crates onto its own line, this package's bundle
becomes empty and ``fetch-doors`` collapses to a pointer at
``woof fetch-bridges``.  That is the better end state and the table says so
per row, so the day it happens the change here is a column, not a redesign.

THE COLUMN IS READ FROM THE INSTALLED ENGINE, NOT ONLY FROM THIS TABLE.
``Door.bundle`` says who publishes a door when the engine does not: it is
what this package's own release builds and pins.  At run time the answer is
:func:`publisher`, and a door the installed engine's own bundle declares
(``woof.bridge_assets.BUNDLED_ARTIFACTS``, the one roster ``woof
fetch-bridges`` stages from and the engine's release cut builds, pins and
probes) is the engine's door, resolved from the engine's bundle and checked
against the engine's pins.  The contract literal still has to be in the
bytes, so an engine build that predates the contract is still a gap by name.

THE BREAKAGE THIS PREVENTS, measured 2026-09-29 on a clean install of a
distribution that carries this model and the engine together and ships all
fourteen doors in its one bundle: ``doctor`` graded the engine's own
``rw_asos`` and ``rw_goes`` against this package's pins for a different
build and called them wrong, reported the six others as not staged, and
told the reader to download a companion bundle that distribution never
publishes, so a correct install exited 1 with eight gaps and every
observation and DA command pointed at a remedy that could not help.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

__all__ = [
    "COMPANION_BUNDLE",
    "COMPANION_DIR_ENV",
    "DOORS",
    "DOOR_PINS_SCHEMA",
    "Door",
    "DoorStagingError",
    "ENGINE_BUNDLE",
    "SUPPORTED_PLATFORMS",
    "artifact_filename",
    "bind_companion_doors",
    "bundle_filename",
    "companion_door_dir",
    "companion_doors",
    "companion_pins",
    "companion_pins_path",
    "current_platform",
    "door_by_name",
    "door_environment",
    "doors_from_bundle",
    "engine_bundle_names",
    "engine_pin_for",
    "find_door",
    "pin_for",
    "publisher",
    "search_path",
    "sha256_file",
    "stage_from_directory",
    "verify_staged",
]

#: The engine's bundle: built by woof's CI, attached to a woof GitHub
#: release, staged by `woof fetch-bridges`.
ENGINE_BUNDLE = "woof"

#: This package's bundle: built by this repository's CI from the engine's
#: `tools/rustwx` workspace at the pinned engine revision, attached to a
#: woof global GitHub release, staged by `woof global fetch-doors`.
COMPANION_BUNDLE = "woof global"

#: Schema of `woof/globe/data/door-pins.json`.
DOOR_PINS_SCHEMA = "gpuwm-global-door-pins-v1"

#: Platform keys the companion bundle is published for.  A key names an
#: operating system and a machine architecture and nothing else: it answers
#: "can this box run those bytes", never "who is asking".
SUPPORTED_PLATFORMS = ("linux-x86_64", "win-x86_64")

#: The licence notice every companion bundle from 0.1.2 on carries beside
#: the doors (tools/build_door_bundle.py writes it from the engine's lockfile).
BUNDLE_NOTICE = "THIRD-PARTY-LICENSES.txt"

#: Override the directory the companion doors stage into.
COMPANION_DIR_ENV = "ARWEN_GLOBAL_DOOR_DIR"

_BLOCK_BYTES = 8 * 1024 * 1024


class DoorStagingError(RuntimeError):
    """A refusal: wrong bytes, an unusable source, or an unknown platform."""


class DoorMissing(RuntimeError):
    """A refusal: a Rust door a command needs is not staged on this machine.

    Its own class, rather than a sentence a caller has to match, because the
    exit code turns on it: `woof global` answers 3 for a missing door and 1
    for every other refusal, and 3 is the code a workspace or a CI job acts
    on without a human by staging a bundle.  The engine raises its own
    resolution errors in its own words, and the doors this package calls
    through the engine used to reach a user as exit 1 carrying the engine's
    developer remedy -- a `cargo build` in a checkout no user of a wheel has.
    """


def missing_door_refusal(name: str, detail: str | None = None) -> DoorMissing:
    """The refusal one unstaged door earns, in this package's own words.

    One composer for every call site, so the sentence a user reads at a
    refusal is the sentence `doctor` prints for the same door: the role, the
    commands it stops, the bundle that publishes it and the command that
    stages it.  `detail` carries the resolver's own account of where it
    looked, minus any instruction to build from a checkout: this package
    ships no Rust and its user never runs cargo, so an instruction to do so
    is a remedy nobody can take.
    """

    door = door_by_name(name)
    bundle = publisher(door.name)
    origin = ("`woof fetch-bridges`" if bundle == ENGINE_BUNDLE
              else "`woof global fetch-doors`")
    lines = [
        f"the Rust door {door.name} is not staged: {door.role}",
        f"it stops: {', '.join(door.used_by)}",
        f"published by the {bundle} bundle; stage it with {origin}",
    ]
    if detail:
        lines.append(_without_the_checkout_remedy(detail))
    return DoorMissing("\n".join(lines))


#: The first line of the engine's build recipe, in every spelling measured.
#:
#: THE BREAKAGE THIS PREVENTS: the cut used to be the single literal
#: "# build it from a checkout", and woof 2.7.0 does not write that line.
#: It opens with "# this install carries no Rust sources" and then prints
#: `git clone`, `cargo build --release --locked --offline` and a
#: `Copy-Item`, so the whole recipe came through this filter untouched
#: (measured 2026-09-07 from the installed wheel, `obs fetch --stream
#: ndbc` with nothing staged).  A user of a wheel has no checkout, this
#: package ships no Rust, and for five of the eight companion doors the
#: clone would not even contain the crate: the recipe is a remedy nobody
#: can take, printed instead of the one command that works.
#:
#: Matched as a set of markers rather than one sentence because the
#: engine's wording is the engine's to change, and the failure mode of
#: missing one is silent.
_BUILD_RECIPE_MARKERS = (
    "# build it from a checkout",
    "# this install carries no Rust sources",
    "git clone",
    "cargo build",
)


def _without_the_checkout_remedy(detail: str) -> str:
    """The resolver's account of where it looked, without its build recipe."""

    kept = []
    for line in detail.splitlines():
        if any(marker in line for marker in _BUILD_RECIPE_MARKERS):
            break
        kept.append(line)
    return "\n".join(kept).rstrip()


@dataclass(frozen=True)
class Door:
    """One Rust binary, and everything a reader needs to place it."""

    #: The executable's basename, without the platform suffix.
    name: str
    #: The crate that builds it, inside the engine's `tools/rustwx` workspace
    #: (or `tools/rw_wps` for the mapped engine).
    crate: str
    #: Which bundle publishes it: ENGINE_BUNDLE or COMPANION_BUNDLE.
    bundle: str
    #: What it does on the data path, in one line.
    role: str
    #: The commands of this package that cannot run without it.
    used_by: tuple[str, ...]
    #: True when the binary is a shared library rather than an executable.
    library: bool = False
    #: A byte literal the CURRENT contract compiles into the binary.
    #:
    #: A door is not "whatever executable has the right basename".  The wheel
    #: ships no Rust, so the binaries on a machine were built from some
    #: checkout at some time, and upgrading the Python half does not touch
    #: them.  Each marker is a literal this package's contract needs and an
    #: older build does not carry, so a stale binary fails the handshake
    #: statically: no execution, no new command surface, and it works on the
    #: binaries already on disk.  ``None`` means no literal has been pinned
    #: for this door yet, and the doctor says so instead of implying a check
    #: it did not run.
    marker: bytes | None = None
    #: What breaks when the marker is absent, in one sentence.  Never a
    #: version number: a rebuild bumps that whether or not anything changed.
    marker_breakage: str | None = None
    #: What happens when the door is ABSENT, when the answer is not "the
    #: commands above stop".  A door whose caller has a fallback does not
    #: stop anything, and a report that says it does names a breakage that
    #: is not happening -- which is the gate law read from the other side.
    #: Measured, never assumed: this field is filled in only for a fallback
    #: that has been watched happen.
    fallback: str | None = None
    #: The environment variable that overrides this door, spelled out
    #: whole on its row.  It used to be composed from a prefix and the
    #: door's name, and a composed name is invisible to anything that
    #: reads the table for the variables it names: a distribution that
    #: renames the engine's variables renamed the engine's copy of each
    #: one and not this table's, so an override the engine honoured was
    #: one this package's doctor and ladder never read.
    env: str = ""

    @property
    def env_var(self) -> str:
        """The environment variable the engine's resolution ladder reads first.

        The engine already honours one variable per binary and consults it
        ahead of every directory.  A companion door is pointed at through
        exactly that variable rather than a mechanism of this package's own,
        because a second ladder is how one copy of a binary comes to shadow
        another without anyone being told.
        """

        return self.env


#: The fourteen doors, in the order a reader meets them: the analysis and the
#: statics first, then the observations, then the pictures.
DOORS: tuple[Door, ...] = (
    Door(
        name="gpuwm_mapped_engine",
        env="WOOF_MAPPED_ENGINE_BIN",
        crate="tools/rw_wps",
        bundle=ENGINE_BUNDLE,
        role="decode a GRIB2 analysis or forecast through a source mapping",
        used_by=("run", "cycle", "da fresh", "da cycle", "go"),
        marker=b'gpuwm-mapped-frameset-v1',
        marker_breakage=(
            "an older mapped engine writes a frameset this package cannot read, so every analysis-initialised run dies after the decode instead of before it"),
    ),
    Door(
        name="rw_fetch",
        env="WOOF_RW_FETCH",
        crate="tools/rustwx/crates/rw-fetch",
        bundle=ENGINE_BUNDLE,
        role="fetch analyses, forecasts and observation archives",
        used_by=("da fresh", "da cycle", "microwave", "abi-score", "go"),
        # Measured 2026-09-07 on Linux with nothing staged: `da fresh` brought
        # a 477,409,262 B GDAS analysis down at 13.5 MiB/s and went on to the
        # decode.  The engine resolves its downloader at the call and falls
        # back to its own stdlib byte-range transport, so this door stops
        # none of the commands above; what it costs is the Rust data path.
        fallback=("the engine falls back to its own Python byte-range transport, so "
                  "these commands run and their fetch leaves the Rust data path"),
        marker=b'usage: rw_fetch <fetch|probe|latest>',
        marker_breakage=(
            "a build without all three of fetch, probe and latest cannot resolve a cycle's newest available analysis, so a fresh run starts from whatever happens to be cached or from nothing at all"),
    ),
    Door(
        name="rw_netcdf",
        env="WOOF_RW_NETCDF",
        crate="tools/rustwx/crates/rw-netcdf",
        bundle=ENGINE_BUNDLE,
        role="read and write NetCDF without a Python decoder on the data path",
        used_by=("export", "render", "go"),
        marker=b'gpuwm-rw-netcdf-dump-v1',
        marker_breakage=(
            "an older rw_netcdf writes a dump record this package's reader does not know, and the Python boundary forbids the fallback decoder that would otherwise hide it"),
    ),
    Door(
        # The tape writer, and NOT rw_netcdf: `woof.io.wrfout.WrfoutWriter`
        # drives the netcdf-writer cdylib through `woof.io.nc_writer_bridge`,
        # a different artifact with a different pin, and the engine's bundle
        # pins both.  While this row was missing, `doctor` reported fourteen
        # of fourteen doors on a machine where `woof global render` could not
        # write the tape it draws from, and the refusal arrived as exit 1
        # carrying the engine's `cargo build`.
        name="netcdf_writer",
        env="WOOF_NCWRITE_BRIDGE",
        crate="tools/rustwx/crates/netcdf-writer",
        bundle=ENGINE_BUNDLE,
        role="write the wrfout product tapes every picture is drawn from",
        used_by=("export", "render", "go"),
        library=True,
    ),
    Door(
        name="static_fields",
        env="WOOF_STATIC_BRIDGE",
        crate="tools/rustwx/crates/static-fields",
        bundle=ENGINE_BUNDLE,
        role="build land use, soil, vegetation, albedo and deep soil temperature from the geography archive",
        used_by=("statics", "run", "go"),
        library=True,
        marker=b'gpuwm_static_build_fields',
        marker_breakage=(
            "an older static-fields library builds a field set the statics door cannot bind, so a cold start comes up with no land use"),
    ),
    Door(
        name="obs_regrid",
        env="WOOF_OBSREGRID_BRIDGE",
        crate="tools/rustwx/crates/obs-regrid",
        bundle=ENGINE_BUNDLE,
        role="regrid observation fields onto the model grid",
        used_by=("assimilate", "da cycle", "da analyze"),
        library=True,
        marker=b'gpuwm_obsregrid_build_plan',
        marker_breakage=(
            "an older obs-regrid library cannot build the plan the filter hands it, so every analysis stops at the first observation batch"),
    ),
    Door(
        name="rw_asos",
        env="WOOF_RW_ASOS",
        crate="tools/rustwx/crates/rw-obs",
        bundle=COMPANION_BUNDLE,
        role="decode surface station observations",
        used_by=("obs fetch", "assimilate", "cycle", "da cycle", "da fresh"),
        marker=b'gpuwm-obs.table.v2',
        marker_breakage=(
            "the surface stream calls `rw_asos networks` and `rw_asos table`, which an older rw_asos does not have: the DA cycle loses its global surface network and the neutral observation table it grades against, and the failure lands mid-cycle as an unknown subcommand rather than at the door"),
    ),
    Door(
        name="rw_goes",
        env="WOOF_RW_GOES",
        crate="tools/rustwx/crates/rw-goes",
        bundle=COMPANION_BUNDLE,
        role="decode GOES-R ABI Level 1b radiances and run the clear-sky forward operator",
        used_by=("abi-score", "abi-reference", "abi-fast-model", "da cycle"),
        marker=b'gpuwm-da.abi-forward.v1',
        marker_breakage=(
            "the ABI leg calls `rw_goes bt`, `rw_goes colocate` and `rw_goes forward`, none of which an older rw_goes has: `abi-score`, `abi-reference` and `abi-fast-model` have no radiances to score and no forward operator to score them with"),
    ),
    Door(
        name="rw_wrfbatch",
        env="WOOF_RW_WRFBATCH",
        crate="tools/rustwx/crates/rw-wrfbatch",
        bundle=ENGINE_BUNDLE,
        role="draw every weather-field picture this package produces",
        used_by=("render", "go"),
        marker=b'usage: rw_wrfbatch --store-root DIR --out-dir DIR',
        marker_breakage=(
            "the render path drives the batch renderer by store root and output directory, and a build that does not take those is not the renderer this package calls; the render law leaves no second way to draw a weather field"),
    ),
    Door(
        name="rw_atms",
        env="WOOF_RW_ATMS",
        crate="tools/rustwx/crates/rw-atms",
        bundle=COMPANION_BUNDLE,
        role="decode and thin ATMS microwave brightness temperatures",
        used_by=("microwave", "da cycle"),
        marker=b'gpuwm-rw-atms-decode-v1',
        marker_breakage=(
            "the microwave leg cannot decode or thin an ATMS granule, so `microwave` and the ATMS stream of `da cycle` have no observations at all"),
    ),
    Door(
        name="rw_gnssro",
        env="WOOF_RW_GNSSRO",
        crate="tools/rustwx/crates/rw-obs",
        bundle=COMPANION_BUNDLE,
        role="decode GNSS radio-occultation bending angles and refractivity",
        used_by=("obs fetch", "da cycle", "da fresh"),
        marker=b'gpuwm-obs.gnssro-table.v1',
        marker_breakage=(
            "the radio-occultation stream cannot write the neutral table the filter reads"),
    ),
    Door(
        name="rw_ndbc",
        env="WOOF_RW_NDBC",
        crate="tools/rustwx/crates/rw-obs",
        bundle=COMPANION_BUNDLE,
        role="decode marine buoy and coastal station observations",
        used_by=("obs fetch", "da cycle", "da fresh"),
        marker=b'gpuwm-obs.ndbc-table.v1',
        marker_breakage=(
            "the marine stream cannot write the neutral table the filter reads"),
    ),
    Door(
        name="rw_igra2",
        env="WOOF_RW_IGRA2",
        crate="tools/rustwx/crates/rw-obs",
        bundle=COMPANION_BUNDLE,
        role="decode radiosonde soundings",
        used_by=("obs fetch", "da cycle", "da fresh", "upper-air scoring"),
        marker=b'gpuwm-obs.igra2-table.v1',
        marker_breakage=(
            "the radiosonde stream cannot write the neutral table the upper-air scorecard reads"),
    ),
    Door(
        name="rw_amv",
        env="WOOF_RW_AMV",
        crate="tools/rustwx/crates/rw-obs",
        bundle=COMPANION_BUNDLE,
        role="decode atmospheric motion vectors",
        used_by=("obs fetch", "da cycle", "da fresh"),
        marker=b'gpuwm-obs.amv-table.v1',
        marker_breakage=(
            "the motion-vector stream cannot write the neutral table the filter reads"),
    ),
    Door(
        name="rw_wis2",
        env="WOOF_RW_WIS2",
        crate="tools/rustwx/crates/rw-obs",
        bundle=COMPANION_BUNDLE,
        role="decode the WIS2 surface and upper-air feeds",
        used_by=("obs subscribe", "da cycle", "da fresh"),
        marker=b'gpuwm-obs.wis2-subscribe.v1',
        marker_breakage=(
            "the WIS2 stream cannot subscribe, so the feed is silent rather than refused"),
    ),
)

_BY_NAME = {door.name: door for door in DOORS}


def door_by_name(name: str) -> Door:
    """The row for ``name``, or a refusal naming the fourteen that exist."""

    try:
        return _BY_NAME[name]
    except KeyError:
        known = ", ".join(sorted(_BY_NAME))
        raise KeyError(
            f"{name} is not one of this package's doors; the doors are: {known}"
        ) from None


def doors_from_bundle(bundle: str) -> tuple[Door, ...]:
    """Every door this table assigns to ``bundle``.

    The static column: what this package's own release builds and pins.
    What a running install resolves is :func:`publisher`, which also asks
    the installed engine.
    """

    return tuple(door for door in DOORS if door.bundle == bundle)


def engine_bundle_names() -> frozenset[str]:
    """The artifacts the installed engine's own bundle declares, by name.

    Read from ``woof.bridge_assets.BUNDLED_ARTIFACTS``: the roster ``woof
    fetch-bridges`` stages from, and the one the engine's release cut
    builds, pins and probes, so a name on it is a door that engine
    publishes bytes and pins for.  Empty when the engine or its roster
    cannot be read, which leaves every door where this table puts it.
    """

    try:
        from woof import bridge_assets
    except Exception:
        return frozenset()
    roster = getattr(bridge_assets, "BUNDLED_ARTIFACTS", ())
    return frozenset(
        name for name in (getattr(entry, "name", None) for entry in roster)
        if isinstance(name, str))


def publisher(name: str) -> str:
    """Which bundle publishes one door on this install.

    The engine's, for a door this table gives the engine and for any door
    the installed engine's own bundle declares; this package's otherwise.
    One copy of a door per install, from the bundle that actually ships it:
    a door the engine publishes is resolved from the engine's bundle and
    checked against the engine's pins, and is never sent to ``fetch-doors``.
    """

    door = door_by_name(name)
    if door.bundle == ENGINE_BUNDLE or door.name in engine_bundle_names():
        return ENGINE_BUNDLE
    return COMPANION_BUNDLE


def companion_doors() -> tuple[Door, ...]:
    """The doors this package itself has to supply on this install.

    Empty when the installed engine's bundle publishes every door, which is
    the end state this module's docstring names: ``fetch-doors`` then has
    nothing to stage and points at ``woof fetch-bridges``.
    """

    return tuple(door for door in DOORS
                 if publisher(door.name) == COMPANION_BUNDLE)


def companion_pins_path() -> Path:
    """Where the companion bundle's pins live inside the installed package."""

    return Path(__file__).resolve().parent / "data" / "door-pins.json"


def companion_pins() -> dict:
    """The companion bundle's size and SHA-256 pins, as written.

    Returns the parsed document.  The file always exists in the wheel; before
    the first release cut its `platforms` table is empty, which is what
    `woof global fetch-doors` reports as "no companion bundle has been
    published yet" rather than pretending a download exists.
    """

    return json.loads(companion_pins_path().read_text(encoding="utf-8"))


def companion_door_dir() -> Path:
    """Where `woof global fetch-doors` stages the companion bundle.

    ``~/.woof/global-doors`` by default, overridden by
    :data:`COMPANION_DIR_ENV`.  Beside the engine's ``~/.woof/bridges``, not
    inside it: two of the eight companion doors share a filename with a
    binary the engine's own bundle publishes, and writing over those would
    leave the engine resolving an executable whose ``--abi`` its own doctor
    rejects.  One copy of a door per owner, each owner resolving its own.
    """

    override = os.environ.get(COMPANION_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".woof" / "global-doors"


def _library_filename(name: str, platform: str | None = None) -> str:
    if platform is None:
        platform = "win-x86_64" if sys.platform == "win32" else "linux-x86_64"
    if platform.startswith("win"):
        return f"{name}.dll"
    return f"lib{name}.so"


def artifact_filename(name: str, platform: str | None = None) -> str:
    """The filename one door has on ``platform``.

    The two spellings a platform gives a built artifact are why this cannot
    be derived from the logical name alone.
    """

    door = door_by_name(name)
    if platform is None:
        platform = "win-x86_64" if sys.platform == "win32" else "linux-x86_64"
    if door.library:
        return _library_filename(name, platform)
    return f"{name}.exe" if platform.startswith("win") else name


def bundle_filename(release: str, platform: str) -> str:
    """The release-asset name of one companion bundle."""

    if platform not in SUPPORTED_PLATFORMS:
        raise DoorStagingError(
            f"{platform} is not a platform this package publishes doors for; "
            f"the platforms are {', '.join(SUPPORTED_PLATFORMS)}")
    return f"arwen-global-doors-{release}-{platform}.zip"


def current_platform() -> str | None:
    """This machine's platform key, or ``None`` when no bundle fits it.

    A capability check on the operating system and the machine architecture,
    never an identity gate: a platform with no published bundle is told so by
    name and handed the build-from-source route, which stays universal.
    """

    import platform as platform_module

    machine = platform_module.machine().lower()
    if machine not in ("x86_64", "amd64"):
        return None
    if sys.platform.startswith("linux"):
        return "linux-x86_64"
    if sys.platform == "win32":
        return "win-x86_64"
    return None


def search_path(name: str) -> tuple[Path, ...]:
    """Where a door is looked for, in the order it is looked for.

    The engine's ladder, with one rung added ahead of its directories for a
    door this package publishes.  The environment override stays first,
    because it is what an operator reaches for and it must outrank anything
    either bundle staged.
    """

    from woof.bridges import default_bridge_dir, packaged_bridge_dir

    door = door_by_name(name)
    filename = artifact_filename(name)
    candidates: list[Path] = []
    override = os.environ.get(door.env_var)
    if override:
        candidates.append(Path(override))
    if publisher(door.name) == COMPANION_BUNDLE:
        candidates.append(companion_door_dir() / filename)
    candidates.extend((
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
    ))
    return tuple(candidates)


def find_door(name: str) -> Path | None:
    """The staged path of one door, or ``None`` when nothing is staged."""

    for candidate in search_path(name):
        if candidate.is_file():
            return candidate
    return None


def door_environment() -> dict[str, str]:
    """The per-door environment this package's commands run their doors under.

    The engine resolves ``rw_asos`` and ``rw_goes`` from its own bundle, and
    for the engine's own commands that is right.  This package needs the
    newer pair, so it names them the way the engine's ladder already accepts:
    one variable per binary, set only for the doors this package publishes,
    and only when the operator has not already set it -- an explicit override
    outranks a package that thinks it knows better.
    """

    environment: dict[str, str] = {}
    for door in companion_doors():
        if os.environ.get(door.env_var):
            continue
        staged = companion_door_dir() / artifact_filename(door.name)
        if staged.is_file():
            environment[door.env_var] = str(staged)
    return environment


def bind_companion_doors() -> dict[str, str]:
    """Point this process at the doors this package publishes, and say so.

    Called once at the top of the console script.  It writes
    :func:`door_environment` into ``os.environ``, which is what makes every
    consumer -- this package's own modules and the engine's front-door
    ladder alike -- resolve the same binary without any of them being taught
    a second lookup.

    It never overwrites a variable the operator set, and it returns exactly
    what it did set so the caller can print it.  Silently redirecting a
    binary is the shadowing this whole arrangement exists to prevent; doing
    it and saying which is the arrangement working.
    """

    bound = door_environment()
    os.environ.update(bound)
    return bound


# --------------------------------------------------------------------- pins

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def pin_for(name: str, platform: str | None = None) -> dict | None:
    """This package's pinned bytes for one companion door, or ``None``.

    ``None`` means no companion bundle has been published for this platform
    yet, which the doctor reports as exactly that rather than as a failure to
    match a hash it never had.
    """

    if platform is None:
        platform = current_platform()
    if platform is None:
        return None
    record = companion_pins().get("platforms", {}).get(platform)
    if not record:
        return None
    for entry in record.get("binaries", ()):
        if entry.get("artifact") == name:
            return entry
    return None


def engine_pin_for(name: str, platform: str | None = None) -> dict | None:
    """The ENGINE's pinned bytes for one door, read from the installed woof.

    The engine packages its own bundle pins inside its wheel, so a door that
    came from ``woof fetch-bridges`` can be checked against the numbers that
    release published without a network call and without trusting the file on
    disk to describe itself.
    """

    if platform is None:
        platform = current_platform()
    if platform is None:
        return None
    try:
        from importlib.resources import files

        import woof
        from woof import bridge_assets
    except Exception:
        return None
    try:
        resource = files(woof).joinpath(bridge_assets.PINS_RESOURCE)
        document = json.loads(resource.read_text(encoding="utf-8"))
    except Exception:
        return None
    record = document.get("platforms", {}).get(platform)
    if not record:
        return None
    for entry in record.get("binaries", ()):
        if entry.get("artifact") == name:
            return entry
    return None


def verify_staged(name: str, path: Path) -> tuple[str, str]:
    """Check one staged door three ways and say what was actually checked.

    Bytes, SHA-256, contract literal: the same three the engine's own staging
    applies, and the verdict names which of the three ran.  A pin that does
    not exist yet is reported as an unchecked hash, never as a pass, because
    claiming a verification that did not happen is the failure this whole
    path exists to make impossible.

    Returns ``(verdict, detail)`` where verdict is ``"ok"``, ``"gap"`` or
    ``"note"``.
    """

    door = door_by_name(name)
    checks: list[str] = []
    pin = (pin_for(name) if publisher(name) == COMPANION_BUNDLE
           else engine_pin_for(name))
    size = path.stat().st_size
    if pin is None:
        checks.append("size and hash unpinned")
    else:
        if int(pin["bytes"]) != size:
            return "gap", (
                f"{size:,} B on disk, the pin for this release says "
                f"{int(pin['bytes']):,} B")
        digest = sha256_file(path)
        if digest != pin["sha256"]:
            return "gap", (
                f"SHA-256 {digest[:16]}... does not match the pin "
                f"{pin['sha256'][:16]}... for these {size:,} bytes")
        checks.append("size and SHA-256 match the pin")
    if door.marker is None:
        checks.append("no contract literal pinned")
    elif door.marker in path.read_bytes():
        checks.append("contract literal present")
    else:
        return "gap", (
            f"the contract literal {door.marker.decode(errors='replace')!r} "
            "is absent from these bytes")
    verdict = "ok" if pin is not None and door.marker is not None else "note"
    return verdict, "; ".join(checks)


# ------------------------------------------------------------------ staging

def stage_from_directory(source: Path, dest: Path,
                         platform: str | None = None) -> list[Path]:
    """Stage every companion door from ``source`` into ``dest``.

    ``source`` is either the bundle archive itself or the binaries loose in a
    directory, which is what an operator has after building them on a machine
    that does have a toolchain.  Every file is verified before it is
    installed, and a file that fails is not written at all.
    """

    if platform is None:
        platform = current_platform()
    if platform is None:
        raise DoorStagingError(
            "no companion bundle is published for this operating system and "
            "machine architecture; build the doors from the engine's "
            "tools/rustwx workspace instead")
    wanted = {artifact_filename(door.name, platform): door
              for door in companion_doors()}
    if not wanted:
        raise DoorStagingError(
            "the installed engine's bundle publishes every door this package "
            "runs on, so there is nothing here to stage; stage them with "
            "`woof fetch-bridges`")
    dest.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            held = set(archive.namelist())
            missing = sorted(set(wanted) - held)
            if missing:
                raise DoorStagingError(
                    f"{source.name} does not carry {', '.join(missing)}; "
                    "refusing to stage a partial bundle")
            for filename, door in wanted.items():
                target = dest / filename
                _write_verified(archive.read(filename), target, door)
                staged.append(target)
            if BUNDLE_NOTICE in held:
                # The licence notice for the crates linked into these
                # binaries travels with them into the directory they run
                # from, which is where their binary-form conditions say it
                # has to be.  A bundle from before 0.1.2 carries none.
                notice = dest / BUNDLE_NOTICE
                notice.write_bytes(archive.read(BUNDLE_NOTICE))
    elif source.is_dir():
        missing = sorted(name for name in wanted
                         if not (source / name).is_file())
        if missing:
            raise DoorStagingError(
                f"{source} does not hold {', '.join(missing)}; refusing to "
                "stage a partial door set")
        for filename, door in wanted.items():
            target = dest / filename
            _write_verified((source / filename).read_bytes(), target, door)
            staged.append(target)
    else:
        raise DoorStagingError(
            f"{source} is neither a bundle archive nor a directory of doors")
    return staged


def _write_verified(payload: bytes, target: Path, door: Door) -> None:
    """Write one door only after its bytes pass every check that exists."""

    pin = pin_for(door.name)
    if pin is not None:
        if len(payload) != int(pin["bytes"]):
            raise DoorStagingError(
                f"{target.name} is {len(payload):,} B and the pin says "
                f"{int(pin['bytes']):,} B; refusing to install it")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != pin["sha256"]:
            raise DoorStagingError(
                f"{target.name} hashes to {digest} and the pin says "
                f"{pin['sha256']}; refusing to install it")
    if door.marker is not None and door.marker not in payload:
        raise DoorStagingError(
            f"{target.name} does not carry the contract literal "
            f"{door.marker.decode(errors='replace')!r}: {door.marker_breakage}")
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_bytes(payload)
    if not door.library:
        temporary.chmod(0o755)
    shutil.move(str(temporary), str(target))
