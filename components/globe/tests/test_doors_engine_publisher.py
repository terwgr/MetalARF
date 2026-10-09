"""A door the installed engine's own bundle declares is the engine's door.

THE BREAKAGE THESE PREVENT, measured 2026-09-29 on a clean install of a
distribution that ships this model and the engine together, with all
fourteen doors in the engine's one bundle: ``doctor`` graded the engine's
``rw_asos`` and ``rw_goes`` against this package's pins for a different build
and called them wrong, reported the other six as not staged, and named
``fetch-doors`` as the remedy, which downloads a companion bundle that
distribution never publishes.  A correct install exited 1 with eight gaps.

The publisher of a door is read from the installed engine
(:func:`woof.globe.doors.publisher`); these tests drive that answer
directly, so they hold whichever engine the suite runs against.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pytest

from woof.globe import doctor, doors, fetch_doors

COMPANION = tuple(d.name for d in doors.doors_from_bundle(doors.COMPANION_BUNDLE))
EVERY = tuple(d.name for d in doors.DOORS)


def _engine_declares(monkeypatch, names) -> None:
    monkeypatch.setattr(doors, "engine_bundle_names", lambda: frozenset(names))


def test_a_door_the_engine_declares_is_published_by_the_engine(monkeypatch):
    _engine_declares(monkeypatch, {"rw_atms"})
    assert doors.publisher("rw_atms") == doors.ENGINE_BUNDLE
    assert "rw_atms" not in {d.name for d in doors.companion_doors()}
    # the static column is what this package's own release builds; it moves
    # only when the table does
    assert doors.door_by_name("rw_atms").bundle == doors.COMPANION_BUNDLE


def test_an_engine_that_declares_none_leaves_the_table_as_written(monkeypatch):
    _engine_declares(monkeypatch, ())
    assert {d.name for d in doors.companion_doors()} == set(COMPANION)
    for name in EVERY:
        assert doors.publisher(name) == doors.door_by_name(name).bundle


def test_the_engines_door_is_not_searched_for_in_this_packages_directory(
        monkeypatch, tmp_path):
    _engine_declares(monkeypatch, {"rw_atms"})
    monkeypatch.setenv(doors.COMPANION_DIR_ENV, str(tmp_path / "staged"))
    monkeypatch.delenv(doors.door_by_name("rw_atms").env_var, raising=False)
    (tmp_path / "staged").mkdir()
    (tmp_path / "staged" / doors.artifact_filename("rw_atms")).write_bytes(b"x")
    ladder = doors.search_path("rw_atms")
    assert all(path.parent != tmp_path / "staged" for path in ladder)
    assert doors.door_by_name("rw_atms").env_var not in doors.door_environment()


def test_the_engines_door_is_graded_against_the_engines_pin(monkeypatch, tmp_path):
    """Not this package's pin for a build it did not ship."""

    door = doors.door_by_name("rw_asos")
    payload = b"\x7fELF" + door.marker + b"\x00" * 32
    staged = tmp_path / doors.artifact_filename("rw_asos")
    staged.write_bytes(payload)
    engine_pin = {"artifact": "rw_asos", "bytes": len(payload),
                  "sha256": hashlib.sha256(payload).hexdigest()}
    other_pin = {"artifact": "rw_asos", "bytes": len(payload) + 7,
                 "sha256": "0" * 64}
    monkeypatch.setattr(doors, "engine_pin_for", lambda name, platform=None: engine_pin)
    monkeypatch.setattr(doors, "pin_for", lambda name, platform=None: other_pin)

    _engine_declares(monkeypatch, {"rw_asos"})
    verdict, detail = doors.verify_staged("rw_asos", staged)
    assert verdict == "ok", detail
    assert "match the pin" in detail

    _engine_declares(monkeypatch, ())
    verdict, detail = doors.verify_staged("rw_asos", staged)
    assert verdict == "gap", "with the engine silent this package's pin applies"


def test_the_contract_literal_still_decides_an_engine_door(monkeypatch, tmp_path):
    """An engine build that predates the contract is a gap by name."""

    payload = b"\x7fELF an older rw_goes" + b"\x00" * 32
    staged = tmp_path / doors.artifact_filename("rw_goes")
    staged.write_bytes(payload)
    pin = {"artifact": "rw_goes", "bytes": len(payload),
           "sha256": hashlib.sha256(payload).hexdigest()}
    monkeypatch.setattr(doors, "engine_pin_for", lambda name, platform=None: pin)
    _engine_declares(monkeypatch, {"rw_goes"})
    verdict, detail = doors.verify_staged("rw_goes", staged)
    assert verdict == "gap"
    assert "contract literal" in detail


def test_a_missing_engine_door_names_the_engines_command(monkeypatch):
    _engine_declares(monkeypatch, {"rw_igra2"})
    text = str(doors.missing_door_refusal("rw_igra2"))
    assert "woof fetch-bridges" in text
    assert "fetch-doors" not in text


def test_fetch_doors_points_at_the_engine_when_it_publishes_every_door(
        monkeypatch, tmp_path, capsys):
    """Nothing to stage is exit 0 and one sentence, never a download."""

    _engine_declares(monkeypatch, EVERY)

    def no_network(*_args, **_kwargs):
        raise AssertionError("fetch-doors reached for the network")

    monkeypatch.setattr(fetch_doors, "_download", no_network)
    for listing in (False, True):
        args = argparse.Namespace(dest=tmp_path / "dest", source=None, list=listing)
        assert fetch_doors.fetch_doors(args) == 0
        out = capsys.readouterr().out
        assert "woof fetch-bridges" in out
        assert "nothing for this command to stage" in out
    assert not (tmp_path / "dest").exists()
    with pytest.raises(doors.DoorStagingError) as caught:
        doors.stage_from_directory(tmp_path, tmp_path / "dest", "linux-x86_64")
    assert "woof fetch-bridges" in str(caught.value)


def test_the_download_comes_from_the_repository_the_metadata_names(monkeypatch):
    """No release address is written into the source.

    THE BREAKAGE THIS PREVENTS: the literal sent a distribution that carries
    this package under its own name to another repository's releases.
    """

    monkeypatch.delenv(fetch_doors.ASSET_URL_BASE_ENV, raising=False)
    monkeypatch.setattr(fetch_doors, "repository_url",
                        lambda: "https://example.invalid/owner/repo")
    assert (fetch_doors.asset_url_base("v9.9.9")
            == "https://example.invalid/owner/repo/releases/download/v9.9.9")

    monkeypatch.setattr(fetch_doors, "repository_url", lambda: None)
    with pytest.raises(doors.DoorStagingError) as caught:
        fetch_doors.asset_url_base("v9.9.9")
    assert fetch_doors.ASSET_URL_BASE_ENV in str(caught.value)

    monkeypatch.setenv(fetch_doors.ASSET_URL_BASE_ENV, "https://mirror.invalid/x/")
    assert fetch_doors.asset_url_base("v9.9.9") == "https://mirror.invalid/x"
    source = Path(fetch_doors.__file__).read_text(encoding="utf-8")
    assert "releases/download\")" not in source and "github.com/" not in source


def test_doctor_grades_every_door_from_the_bundle_that_ships_it(
        monkeypatch, tmp_path):
    """The whole rust-doors section on an install whose engine ships all
    fourteen: no gap, no `fetch-doors`, every row the engine's."""

    _engine_declares(monkeypatch, EVERY)
    staged: dict[str, Path] = {}
    pins: dict[str, dict] = {}
    for door in doors.DOORS:
        payload = b"\x7fELF " + (door.marker or b"") + door.name.encode()
        path = tmp_path / doors.artifact_filename(door.name)
        path.write_bytes(payload)
        staged[door.name] = path
        pins[door.name] = {"artifact": door.name, "bytes": len(payload),
                           "sha256": hashlib.sha256(payload).hexdigest()}
        monkeypatch.delenv(door.env_var, raising=False)
    monkeypatch.setattr(doctor, "find_door", lambda name: staged[name])
    monkeypatch.setattr(doors, "engine_pin_for",
                        lambda name, platform=None: pins.get(name))
    monkeypatch.setenv(doors.COMPANION_DIR_ENV, str(tmp_path / "companion"))

    report = doctor.Report()
    doctor._doors_section(report)
    rows = dict(report.sections)["rust doors"]
    assert [row.label for row in rows] == list(EVERY)
    assert not [row for row in rows if row.verdict == "gap"], [
        (row.label, row.finding) for row in rows if row.verdict == "gap"]
    text = "\n".join(" ".join((row.finding, *row.detail)) for row in rows)
    assert "fetch-doors" not in text
    assert f"published by the {doors.COMPANION_BUNDLE} bundle" not in text
