"""The compat seam, and the rule that makes it retire itself.

Two symbols this package needs live on the engine's side of the line and are
not in a published engine release yet.  `woof.globe.engine_compat` is what
stands between that fact and a package that cannot be imported.

Everything here is written so that the day the engine carries the symbols,
these tests tell somebody to delete the seam rather than letting it quietly
become a second implementation nobody compares against the first.

MOST OF THIS FILE WAS DELETED ON 2026-09-09, and the deletion is the point.
It held the YSU mixing-length contract's carried copy, the two float64 cloud
references, and the kernel refusal that stopped `"fixed"` on an engine whose
YSU launcher took no flag.  All of that is now CARRIED CODE in
`woof.globe.core`, cut from the tree the model was graded in: the contract,
the launcher, the kernel and the float64 mirror travel together.  A carried
callable's signature is this package's own, so the questions those tests asked
have one answer and no longer have a second.  What is left is the seam that is
still a seam.
"""
from __future__ import annotations

import pytest

pytest.importorskip("woof", reason="the engine under measurement")

from woof.globe import engine_compat  # noqa: E402


def test_a_computation_refuses_rather_than_being_reimplemented(monkeypatch):
    """Both remaining gaps are refused, and the refusal says why.

    Reimplementing a card measurement or a reference regrid here would put a
    SECOND instrument beside the engine's, answering the same question with
    equal confidence and no way to see which one is wrong.
    """

    monkeypatch.setattr(engine_compat, "_from_engine",
                        lambda module, symbol: None)
    for call in (engine_compat.measured_free_vram_bytes,
                 engine_compat.interpolate_to_tape):
        with pytest.raises(engine_compat.MissingEngineSymbol) as caught:
            call()
        message = str(caught.value)
        assert "does not carry" in message
        assert "stops" in message
        assert "does not reimplement" in message


def test_every_gap_names_what_it_stops():
    for gap in engine_compat.GAPS:
        assert gap.stops and gap.stops[0].islower() or gap.stops[0] == "`"
        assert gap.handling in ("carried", "refused")
        assert gap.module.startswith("woof.")
        assert isinstance(gap.stops_a_documented_command, bool)
        # A native forecast is `run` or `go`, both documented, so a row that
        # stops one and claims to stop no documented command contradicts
        # itself and the doctor would exit 0 over a refused forecast.
        assert gap.stops_a_documented_command or not gap.stops_a_native_forecast


def test_the_gap_list_is_measured_not_asserted():
    """`engine_gaps()` reports the gaps on THIS install, whatever they are."""

    gaps = engine_compat.engine_gaps()
    for gap in gaps:
        assert not gap.present()
    for gap in set(engine_compat.GAPS) - set(gaps):
        assert gap.present()


def test_the_seam_names_only_what_is_still_on_the_engine():
    """The retirement, stated as a test rather than as a comment.

    A guard that outlives its defect is worse than no guard: it teaches a
    reader to expect a failure that cannot happen, and it hides the one that
    can.  Every physics module this package calls is carried, so no row here
    may name one.
    """

    carried = {"rrtmgp", "gf", "ntiedtke", "sfclay", "ysu", "ysu_contract",
               "landuse", "physics_inventory", "physics", "noah", "morrison",
               "npref", "kernels"}
    for gap in engine_compat.GAPS:
        tail = gap.module.rsplit(".", 1)[-1]
        assert tail not in carried, (
            f"{gap.module}.{gap.symbol} names a module this package carries; "
            "a gap row for carried code is a guard with no defect")
    for gap in engine_compat.SIGNATURE_GAPS:
        tail = gap.module.rsplit(".", 1)[-1]
        assert tail not in carried, (
            f"{gap.module}.{gap.name} names a module this package carries")


def test_the_ysu_mixing_length_is_answered_by_carried_code():
    """The refusal that is gone, and what stands where it stood.

    `"fixed"` was never a Python branch: it is one more integer argument to
    the YSU kernel, and an engine built before that argument existed launched
    the old vector.  The kernel is carried now, so the option is answered by
    this package's own contract and its own `ysu.cu`.
    """

    from woof.globe.core.ysu_contract import (
        YSU_FIXED_ASYMPTOTIC_LENGTH_M,
        YSU_FREE_ATMOSPHERE_MIXING_LENGTHS,
        free_atmosphere_mixing_length_flag,
    )
    from woof.globe.physics.native_options import NativePhysicsOptions
    from woof.globe.constants import NATIVE_PHYSICS_ACKNOWLEDGEMENT

    assert YSU_FREE_ATMOSPHERE_MIXING_LENGTHS == {"wrf-layer": 0, "fixed": 1}
    assert YSU_FIXED_ASYMPTOTIC_LENGTH_M == 30.0
    assert free_atmosphere_mixing_length_flag("fixed") == 1
    assert not hasattr(engine_compat, "require_free_atmosphere_mixing_length")

    ack = dict(acknowledgement=NATIVE_PHYSICS_ACKNOWLEDGEMENT,
               start_time_utc="2026-09-01T00:00:00Z")
    # The option that used to be refused is accepted, on any engine.
    options = NativePhysicsOptions.from_mapping(
        dict(ack, ysu_free_atmosphere_mixing_length="fixed"))
    assert options.ysu_free_atmosphere_mixing_length == "fixed"
    with pytest.raises(ValueError, match="must be one of"):
        NativePhysicsOptions.from_mapping(
            dict(ack, ysu_free_atmosphere_mixing_length="30m"))
