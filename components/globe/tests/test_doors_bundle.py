"""The door table, the companion bundle, and the checks that guard it.

These tests are about the Rust side of the package: which binary comes from
which bundle, what the pins document says, and whether a wrong byte is
actually refused rather than reported.  They run without a staged binary and
without a network, because the failures they guard are packaging failures and
those have to be catchable on a laptop.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from woof.globe import doors


@pytest.fixture(autouse=True)
def _the_table_as_this_package_releases_it(monkeypatch):
    """These tests are about this package's own bundle and the table it is
    built from.  An engine whose bundle declares a companion door takes that
    door over at run time (doors.publisher, tested on its own in
    test_doors_engine_publisher.py), so the engine's roster is held silent
    here and the result does not depend on which engine the suite runs on."""

    monkeypatch.setattr(doors, "engine_bundle_names", lambda: frozenset())


def test_every_door_names_a_bundle_and_a_command_it_stops():
    """A row with no bundle cannot be staged; a row with no consumer is not
    a door, it is a binary somebody remembered."""

    assert doors.DOORS, "the door table is empty"
    for door in doors.DOORS:
        assert door.bundle in (doors.ENGINE_BUNDLE, doors.COMPANION_BUNDLE), (
            f"{door.name} names bundle {door.bundle!r}, which is neither")
        assert door.used_by, f"{door.name} names no command it stops"
        assert door.role, f"{door.name} says nothing about what it does"


def test_a_marker_comes_with_the_breakage_it_prevents():
    """The gate law, applied to the contract handshake: a check that cannot
    say what it prevents is a check nobody can act on."""

    for door in doors.DOORS:
        if door.marker is None:
            assert door.marker_breakage is None, (
                f"{door.name} states a breakage for a check it does not run")
            continue
        assert door.marker_breakage, (
            f"{door.name} pins a contract literal and does not say what its "
            "absence breaks")
        assert isinstance(door.marker, bytes)


def test_the_two_shared_names_are_published_here_not_by_the_engine():
    """rw_asos and rw_goes exist in the engine's bundle under the same
    filename and are not the same door.

    Measured 2026-09-07 against woof 2.7.0: the engine's rw_asos has no
    `networks`, `table` or `awc`, and its rw_goes has no `bt`, `colocate`,
    `quicklook` or `forward`.  If either row ever moves back to the engine's
    bundle it must be because the engine took the crates, and then this test
    is the thing that says so out loud.
    """

    published_here = {d.name for d in doors.doors_from_bundle(doors.COMPANION_BUNDLE)}
    assert {"rw_asos", "rw_goes"} <= published_here
    for name in ("rw_asos", "rw_goes"):
        door = doors.door_by_name(name)
        assert door.marker is not None, (
            f"{name} shares a filename with an older engine binary, so it "
            "must carry a contract literal or nothing distinguishes the two")


def test_the_six_absent_doors_are_all_published_here():
    published_here = {d.name for d in doors.doors_from_bundle(doors.COMPANION_BUNDLE)}
    assert {"rw_atms", "rw_gnssro", "rw_ndbc", "rw_igra2", "rw_amv",
            "rw_wis2"} <= published_here


def test_filenames_differ_by_platform_and_by_kind():
    assert doors.artifact_filename("rw_atms", "linux-x86_64") == "rw_atms"
    assert doors.artifact_filename("rw_atms", "win-x86_64") == "rw_atms.exe"
    assert doors.artifact_filename("obs_regrid", "linux-x86_64") == "libobs_regrid.so"
    assert doors.artifact_filename("obs_regrid", "win-x86_64") == "obs_regrid.dll"


def test_an_unknown_door_refuses_by_naming_the_ones_that_exist():
    with pytest.raises(KeyError) as caught:
        doors.door_by_name("rw_nonesuch")
    assert "rw_atms" in str(caught.value)


def test_an_unknown_platform_refuses_by_naming_the_ones_published():
    with pytest.raises(doors.DoorStagingError) as caught:
        doors.bundle_filename("v0.1.0", "mac-arm64")
    assert "linux-x86_64" in str(caught.value)


# ------------------------------------------------------------------- pins

def test_the_pins_document_is_shaped_the_way_the_stager_reads_it():
    pins = doors.companion_pins()
    assert pins["schema"] == doors.DOOR_PINS_SCHEMA
    for platform, record in pins.get("platforms", {}).items():
        assert platform in doors.SUPPORTED_PLATFORMS
        assert set(record["bundle"]) >= {"filename", "bytes", "sha256"}
        pinned = {entry["artifact"] for entry in record["binaries"]}
        expected = {d.name for d in doors.doors_from_bundle(doors.COMPANION_BUNDLE)}
        assert pinned == expected, (
            f"{platform} pins {sorted(pinned)} and this package publishes "
            f"{sorted(expected)}; a bundle that is not the door set cannot "
            "be staged")
        for entry in record["binaries"]:
            assert len(entry["sha256"]) == 64
            assert entry["bytes"] > 0
            assert entry["filename"] == doors.artifact_filename(
                entry["artifact"], platform)


def test_an_unpinned_platform_reports_no_bundle_rather_than_a_pass():
    """A pin that does not exist is not a hash that matched.

    THE BREAKAGE THIS PREVENTS: a doctor that prints `ok` beside a binary it
    never hashed, which is how an unverified door gets shipped believing it
    was checked.
    """

    assert doors.pin_for("rw_atms", "mac-arm64") is None


# ---------------------------------------------------------------- staging

def _fake_bundle(tmp_path: Path, platform: str, payloads: dict[str, bytes]) -> Path:
    archive = tmp_path / doors.bundle_filename("v0.1.0", platform)
    with zipfile.ZipFile(archive, "w") as zf:
        for name, payload in payloads.items():
            zf.writestr(name, payload)
    return archive


def test_a_partial_bundle_is_refused_by_naming_what_is_missing(tmp_path):
    platform = "linux-x86_64"
    archive = _fake_bundle(tmp_path, platform, {"rw_atms": b"not a binary"})
    with pytest.raises(doors.DoorStagingError) as caught:
        doors.stage_from_directory(archive, tmp_path / "dest", platform)
    assert "rw_igra2" in str(caught.value)


def test_a_door_whose_bytes_miss_the_contract_literal_is_never_written(tmp_path):
    """The staging contract: a file that fails a check is not installed at
    all, rather than installed and then reported.

    THE BREAKAGE THIS PREVENTS: a half-staged directory whose next command
    resolves the wrong binary because it is the one that happens to be there.
    """

    platform = "linux-x86_64"
    published = doors.doors_from_bundle(doors.COMPANION_BUNDLE)
    payloads = {doors.artifact_filename(d.name, platform): b"\x00" * 64
                for d in published}
    archive = _fake_bundle(tmp_path, platform, payloads)
    dest = tmp_path / "dest"
    with pytest.raises(doors.DoorStagingError):
        doors.stage_from_directory(archive, dest, platform)
    staged = list(dest.glob("*")) if dest.exists() else []
    assert not [p for p in staged if not p.name.endswith(".partial")], (
        "a door that failed its checks was written anyway")


def test_verify_staged_refuses_a_size_that_is_not_the_pinned_one(tmp_path):
    platform = doors.current_platform()
    if platform is None:
        pytest.skip("no companion bundle is published for this platform")
    pin = doors.pin_for("rw_atms", platform)
    if pin is None:
        pytest.skip("no companion bundle has been pinned yet")
    wrong = tmp_path / doors.artifact_filename("rw_atms", platform)
    wrong.write_bytes(b"\x00" * (int(pin["bytes"]) + 1))
    verdict, detail = doors.verify_staged("rw_atms", wrong)
    assert verdict == "gap"
    assert "B on disk" in detail


def test_verify_staged_refuses_the_right_size_with_the_wrong_bytes(tmp_path):
    """Size alone is not identity.

    THE BREAKAGE THIS PREVENTS: a door replaced by a same-length file --
    a truncated download that happened to land on the pinned length, or a
    different build of the same source -- passing as the pinned one.
    """

    platform = doors.current_platform()
    if platform is None:
        pytest.skip("no companion bundle is published for this platform")
    pin = doors.pin_for("rw_atms", platform)
    if pin is None:
        pytest.skip("no companion bundle has been pinned yet")
    wrong = tmp_path / doors.artifact_filename("rw_atms", platform)
    wrong.write_bytes(b"\x11" * int(pin["bytes"]))
    assert hashlib.sha256(wrong.read_bytes()).hexdigest() != pin["sha256"]
    verdict, detail = doors.verify_staged("rw_atms", wrong)
    assert verdict == "gap"
    assert "SHA-256" in detail


def test_the_environment_never_overwrites_an_operator_override(monkeypatch, tmp_path):
    """An explicit override outranks a package that thinks it knows better."""

    monkeypatch.setenv("WOOF_RW_ATMS", str(tmp_path / "mine"))
    monkeypatch.setenv(doors.COMPANION_DIR_ENV, str(tmp_path / "staged"))
    (tmp_path / "staged").mkdir()
    (tmp_path / "staged" / doors.artifact_filename("rw_atms")).write_bytes(b"x")
    assert "WOOF_RW_ATMS" not in doors.door_environment()


def test_the_search_ladder_puts_the_override_first_then_this_package(monkeypatch, tmp_path):
    monkeypatch.setenv(doors.COMPANION_DIR_ENV, str(tmp_path / "staged"))
    monkeypatch.setenv("WOOF_RW_ATMS", str(tmp_path / "mine"))
    ladder = doors.search_path("rw_atms")
    assert ladder[0] == tmp_path / "mine"
    assert ladder[1] == tmp_path / "staged" / doors.artifact_filename("rw_atms")


def test_an_engine_door_does_not_search_this_packages_directory(monkeypatch, tmp_path):
    """The companion directory holds doors this package publishes and nothing
    else; an engine door found there would be a copy nobody staged."""

    monkeypatch.setenv(doors.COMPANION_DIR_ENV, str(tmp_path / "staged"))
    monkeypatch.delenv("WOOF_RW_FETCH", raising=False)
    ladder = doors.search_path("rw_fetch")
    assert all(tmp_path / "staged" != path.parent for path in ladder)


def test_the_pins_note_and_the_table_agree_on_the_count():
    """The note is read by a person deciding whether to trust the file."""

    pins = doors.companion_pins()
    published = doors.doors_from_bundle(doors.COMPANION_BUNDLE)
    note = pins.get("note") or ""
    assert "fetch-doors" in note
    for door in published:
        assert door.name in note, (
            f"{door.name} is published by this package and the pins note does "
            "not name it")


def test_the_pins_file_is_valid_json_on_disk():
    text = doors.companion_pins_path().read_text(encoding="utf-8")
    assert json.loads(text)["schema"] == doors.DOOR_PINS_SCHEMA


def test_the_stage_note_never_reports_this_programs_own_bindings_as_the_operators(
        monkeypatch, tmp_path, capsys):
    """The second fetch-doors told the operator to unset what it had set.

    THE BREAKAGE THIS PREVENTS, measured 2026-09-07 from the installed
    wheel: `main()` binds one `WOOF_RW_*` variable per companion door
    into its own environment before any command runs.  The staging note
    read those back, so every fetch-doors after the first one printed
    "these environment overrides are set and outrank what was just
    staged" and named all eight, on a machine where the operator had set
    none of them and where the paths named were the very bytes just
    written.  The action that sentence asks for cannot be taken.
    """

    from woof.globe import fetch_doors

    dest = tmp_path / "global-doors"
    dest.mkdir()
    monkeypatch.setenv(doors.COMPANION_DIR_ENV, str(dest))
    companion = list(doors.doors_from_bundle(doors.COMPANION_BUNDLE))

    # what bind_companion_doors() writes: this package pointing at its own
    # resolved directory, which is not an operator override.
    for door in companion:
        monkeypatch.setenv(
            door.env_var, str(dest / doors.artifact_filename(door.name)))
    fetch_doors._report_what_outranks(dest, companion)
    printed = capsys.readouterr().out
    # NOTHING, not merely a different sentence: the old note said something
    # here, so a test that only forbade a phrase would have passed on it.
    assert printed == "", printed

    # a real one still reports, and names the path that wins
    elsewhere = tmp_path / "mine" / doors.artifact_filename(companion[0].name)
    monkeypatch.setenv(companion[0].env_var, str(elsewhere))
    fetch_doors._report_what_outranks(dest, companion)
    printed = capsys.readouterr().out
    assert "set outside this program" in printed, printed
    assert companion[0].env_var in printed and str(elsewhere) in printed


def test_staging_somewhere_the_ladder_does_not_look_says_so(
        monkeypatch, tmp_path, capsys):
    """--dest stages doors nothing will resolve unless the reader is told.

    The flag's own help says the doctor prints the resolved path; it
    prints the companion directory, which is not where --dest put them.
    """

    from woof.globe import fetch_doors

    resolved = tmp_path / "resolved"
    resolved.mkdir()
    monkeypatch.setenv(doors.COMPANION_DIR_ENV, str(resolved))
    for door in doors.doors_from_bundle(doors.COMPANION_BUNDLE):
        monkeypatch.delenv(door.env_var, raising=False)

    fetch_doors._report_what_outranks(
        tmp_path / "elsewhere", list(doors.doors_from_bundle(doors.COMPANION_BUNDLE)))
    printed = capsys.readouterr().out
    assert "not the directory this package resolves from" in printed, printed
    assert doors.COMPANION_DIR_ENV in printed, printed


def test_every_door_spells_out_the_variable_that_overrides_it():
    """The override name is written whole on each row, never composed.

    THE BREAKAGE THIS PREVENTS: a composed name is invisible to anything
    that reads the table for the variables it names, so a distribution that
    renamed the engine's variables left this table reading the old ones and
    an override the engine honoured was one the doctor never saw.
    """

    import re

    for door in doors.DOORS:
        assert re.fullmatch(r"[A-Z][A-Z0-9_]+", door.env_var), (
            f"{door.name} names no override variable")
    assert len({door.env_var for door in doors.DOORS}) == len(doors.DOORS)


def test_the_published_door_count_is_the_number_in_the_table():
    """Two release surfaces state how many Rust doors this package publishes.

    THE BREAKAGE THIS PREVENTS, and it shipped: both said ELEVEN.  The
    changelog said "Eleven observation doors, all Rust" and then named
    eight things; the release notes said "eleven Rust observation doors"
    beside the two radiance operators, which would make it eleven on top
    of those.  The table publishes eight, `woof global doctor` checks
    eight, the bundle carries eight, and the README's own list names
    eight.  The number came across from a programme document describing a
    tree that is not this distribution.

    A count on a release surface is a claim about the artefact, so it is
    read out of the artefact.
    """

    import re

    root = Path(__file__).resolve().parents[1]
    published = len(list(doors.doors_from_bundle(doors.COMPANION_BUNDLE)))
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
        "twelve": 12, "thirteen": 13, "fourteen": 14,
    }
    # Whitespace is collapsed first: both sentences wrap, and a pattern
    # that stopped at a newline found nothing and reported a clean page.
    pattern = re.compile(
        r"\b(" + "|".join(words) + r"|\d+) [^.]{0,40}?\bdoors\b",
        re.IGNORECASE)

    # The 0.1.0 notes travel with this repository and not with a tree that
    # carries the model inside another distribution; the changelog travels
    # with both, so it is required and the notes are read where present.
    assert (root / "CHANGELOG.md").is_file()
    for name in ("CHANGELOG.md", "RELEASE-NOTES-0.1.0.md"):
        if not (root / name).is_file():
            continue
        text = " ".join((root / name).read_text(encoding="utf-8").split())
        stated = [
            words.get(value.lower(), None) or int(value)
            for value in pattern.findall(text)
            if value.lower() in words or value.isdigit()
        ]
        assert stated, f"{name} states no door count at all any more"
        assert set(stated) == {published}, (
            f"{name} states {stated} Rust doors; "
            f"woof.globe.doors publishes {published}"
        )


def test_every_stream_door_names_the_obs_command_that_reaches_it():
    """A door's `used_by` is the list a refusal prints, so it has to be whole.

    THE BREAKAGE THIS PREVENTS, measured 2026-09-07 from the installed
    wheel: `woof global obs fetch --stream ndbc` with nothing staged
    refused with "it stops: da cycle, da fresh", which does not include the
    command the reader had just run.  `obs fetch` drives six of the eight
    companion doors directly, and `obs subscribe` drives the seventh, and
    none of the eight rows said so.  Under the gate law a refusal names the
    breakage it prevents; a refusal that lists commands the reader is not
    running, and omits the one they are, names the wrong breakage.

    Computed from the stream table rather than transcribed, so a stream
    added later carries its door's row with it.
    """

    from woof.globe.obs_streams import STREAMS

    #: `obs fetch --stream wis2` refuses by name (no decoder writes the
    #: neutral table yet); the door is reached by `obs subscribe`.
    BY_SUBSCRIBE = {"wis2"}

    expected: dict[str, set[str]] = {}
    for stream, spec in STREAMS.items():
        if not spec.door:
            continue
        command = "obs subscribe" if stream in BY_SUBSCRIBE else "obs fetch"
        expected.setdefault(spec.door, set()).add(command)
    assert expected, "no stream names a door; this gate would pass vacuously"

    missing = []
    for name, commands in sorted(expected.items()):
        used = set(doors.door_by_name(name).used_by)
        for command in sorted(commands - used):
            missing.append(f"{name} is driven by `{command}` and its row does "
                           f"not say so: {doors.door_by_name(name).used_by}")
    assert missing == [], "\n  ".join([""] + missing)


def test_the_doctor_names_this_programs_own_binding_as_the_companion_directory(
        monkeypatch, tmp_path):
    """The origin line tells the program's own binding from an operator's.

    THE BREAKAGE THIS PREVENTS, read off a host whose companion directory
    held the previous release's doors: `doctor` printed every door as
    "resolved from the WOOF_RW_* override" when the operator had set none
    of those variables, because the console script binds one per companion
    door into its own environment before any command runs.  A reader
    upgrading is then sent to unset something the program sets again on
    every run.  The same rule fetch-doors applies to what outranks a
    staging applies here: an override is the operator's only when it
    points somewhere other than the file this package's own binding would
    have produced.
    """

    from woof.globe import doctor

    staged_dir = tmp_path / "global-doors"
    staged_dir.mkdir()
    monkeypatch.setenv(doors.COMPANION_DIR_ENV, str(staged_dir))
    door = doors.door_by_name("rw_atms")
    own = staged_dir / doors.artifact_filename(door.name)
    own.write_bytes(b"x")

    # what bind_companion_doors() writes: this program's own binding
    monkeypatch.setenv(door.env_var, str(own))
    origin = doctor._origin(door, own)
    assert "override" not in origin, origin
    assert "companion door directory" in origin and door.env_var in origin

    # the same variable set by the operator to somewhere else really is one
    elsewhere = tmp_path / "mine" / doors.artifact_filename(door.name)
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"y")
    monkeypatch.setenv(door.env_var, str(elsewhere))
    origin = doctor._origin(door, elsewhere)
    assert "override" in origin and "outside this program" in origin, origin

    # and no variable at all resolves to the directory by name
    monkeypatch.delenv(door.env_var, raising=False)
    assert doctor._origin(door, own) == "the companion door directory"
