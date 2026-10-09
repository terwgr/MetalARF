"""The five fallback observation rows, and the rule that keeps them a
fallback rather than a fork.

An observation front door is a table row: a binary name, the environment
variable that overrides it, what it is a door onto, and the exact `--abi`
line the Python side was written against.  The engine owns the mechanism.
This package owns five rows the published engine does not have yet, and the
whole point is that they disappear the day it does.
"""
from __future__ import annotations

import pytest

pytest.importorskip("woof", reason="the engine supplies the FrontDoor class")

from woof.globe import obs_doors  # noqa: E402


def test_the_engine_table_is_always_consulted_first():
    """A row here must never shadow one the engine already carries.

    THE BREAKAGE THIS PREVENTS: two definitions of one door on one machine,
    with this package's copy winning.  The engine would then resolve a door
    one way and this package another, and the difference would show up as an
    observation stream that decodes here and refuses there -- the exact skew
    that made an earlier release's preparations fail.
    """

    assert obs_doors.shadowed_rows() == (), (
        "the installed engine now carries these rows and no bundle of this "
        "package publishes them; delete them from woof/globe/obs_doors.py "
        "and let the engine's table answer")


def test_every_fallback_row_is_a_stream_this_package_documents():
    from woof.globe.doors import DOORS

    named = {door.name for door in DOORS}
    for name in obs_doors.FALLBACK_ROWS:
        assert name in named, (
            f"{name} has a front-door row and no entry in the door table, so "
            "the doctor would never report it")


def test_a_fallback_row_resolves_to_a_real_front_door():
    from woof.obs.frontdoor import FrontDoor

    for name in obs_doors.FALLBACK_ROWS:
        door = obs_doors.front_door(name)
        assert isinstance(door, FrontDoor)
        assert door.name == name
        # The override is the one the door table names for this binary, so
        # the front door, the doctor and the search ladder read one variable.
        from woof.globe.doors import door_by_name

        assert door.env_var == door_by_name(name).env_var
        # The marker is the record contract, not a version number: it names
        # the fields the Python side reads.
        assert "\t" in door.abi_marker
        assert door.abi_marker.startswith("gpuwm-obs.")


def test_a_door_the_engine_publishes_comes_back_as_the_engine_object():
    """Identity, not equality: no copy of an engine row exists here.

    `rw_mrms` is a door only the engine publishes, so the engine's row must
    come back untouched.  The two rows that deliberately override an engine
    row (`rw_asos`, `rw_goes`) are a different case and are covered below.
    """

    from woof.obs import frontdoor

    engine = frontdoor.FRONT_DOORS["mrms"]
    assert obs_doors.front_door("rw_mrms") is engine


def test_an_unknown_door_refuses_by_naming_the_ones_that_exist():
    with pytest.raises(KeyError) as caught:
        obs_doors.front_door("rw_nothing")
    message = str(caught.value)
    assert "rw_asos" in message and "rw_igra2" in message


def test_the_whole_table_is_the_union_and_the_publisher_decides():
    """Every door reachable, each row from whoever publishes that binary."""

    from woof.obs import frontdoor

    table = obs_doors.front_doors()
    published_here = obs_doors._published_here()
    for door in frontdoor.FRONT_DOORS.values():
        if door.name in published_here:
            # This package publishes a newer binary under that name, so its
            # row names the newer record contract.  A probe that passed
            # against the older contract would be worse than no probe.
            assert table[door.name] is not door
            assert table[door.name].name == door.name
        else:
            assert table[door.name] is door
    for name in obs_doors.FALLBACK_ROWS:
        assert name in table


def test_an_override_is_only_ever_a_door_this_package_publishes():
    """An override without a bundle behind it is a fork.

    A row here may differ from the engine's ONLY when `doors.DOORS` names this
    package as that binary's publisher -- that is, only when a bundle of this
    package's own actually carries the newer binary the row describes.
    Otherwise the row would describe a contract no binary on the machine
    speaks.
    """

    from woof.obs import frontdoor

    engine = {door.name: door for door in frontdoor.FRONT_DOORS.values()}
    published_here = obs_doors._published_here()
    for name in obs_doors.FALLBACK_ROWS:
        if name in engine and obs_doors.front_door(name) is not engine[name]:
            assert name in published_here, (
                f"{name} overrides the engine's row and no bundle of this "
                "package publishes it")


@pytest.mark.parametrize("name", ["rw_asos", "rw_goes"])
def test_a_door_both_tables_carry_has_the_engines_line(name):
    """One binary built from the engine's commit answers both tables.

    THE BREAKAGE THIS PREVENTS: this package publishes `rw_asos` and
    `rw_goes` in its own bundle while the engine carries rows for the same
    binaries.  From 2.8.0 the engine's rows carry every subcommand this
    package calls, and the doors are built from the engine's source.  A
    marker here that differs from the engine's by one token makes the probe
    refuse the binary the engine itself accepts, so the surface or radiance
    stream is lost on a correct install.
    """

    from woof.obs import frontdoor

    engine = {door.name: door for door in frontdoor.FRONT_DOORS.values()}
    assert name in engine
    assert obs_doors._ROWS[name][2] == engine[name].abi_marker
