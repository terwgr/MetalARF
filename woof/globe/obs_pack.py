"""Reading the observation packs this package's own Rust doors write.

The ABI radiance leg runs on `rw_goes bt`, a door this package publishes and
the engine's bundle does not: woof 2.7.0's `rw_goes` has no `bt` subcommand
at all (`woof.globe.doors` says so by name, and `woof global fetch-doors`
is what stages the build that does).  That door writes a
`gpuwm-obs.goes-bt.v1` pack in the engine's own `GPWMGOES` container, and the
engine's reader knows the container but not the family: `KNOWN_SCHEMAS` in a
published `woof.obs.goes_pack` lists the two cloud families and stops, so
every BT pack is refused with "declares schema 'gpuwm-obs.goes-bt.v1'; this
reader knows [...]" before a plane is read.

THE READER IS THE ENGINE'S, byte for byte.  Measured 2026-09-09 on the
desktop, the whole difference between the owner tree's `goes_pack.py` and
woof 2.7.0's is nine lines: the two constants below and their addition to
`KNOWN_SCHEMAS`.  The container parse, the header contract, the metadata
requirements, the projection agreement and the plane bounds are identical
text on both sides.  So this module carries the FAMILY NAME and widens the
engine's own table for the length of one read; it does not carry a second
decoder, because a second decoder is how two readers come to disagree about
what a pack says while both report success.

THE ENGINE IS ASKED FIRST.  When the installed engine's table already knows
the family -- which it will, the day it publishes the row -- the widening is
never reached and `read_goes_pack` here is the engine's function called
directly.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import threading

__all__ = [
    "BT_SCHEMA_V1",
    "BT_SCHEMAS",
    "GoesPackError",
    "bt_schemas",
    "engine_knows_bt_schema",
    "read_goes_pack",
]

#: The 2 km Level 1b brightness-temperature family (``rw_goes bt``): the
#: radiance inverted through the file's own Planck constants, DQF-gated, with
#: the scan's clear-sky mask and a CMIP cross-check plane when they rode
#: along.  Same container, same reader; the planes are ``bt``, ``rad``,
#: ``lat``, ``lon``, then ``bcm`` and ``cmip_bt`` when present, then the
#: per-source ``_dqf`` planes.
#:
#: The value is the ENGINE's when the engine has it, and this copy otherwise;
#: `tests/test_arwen_global_obs_pack.py` holds the two together the moment a
#: published engine carries the name.
BT_SCHEMA_V1 = "gpuwm-obs.goes-bt.v1"
BT_SCHEMAS = (BT_SCHEMA_V1,)

#: The table widening below swaps a module attribute, so two threads reading
#: packs at once could restore it out of order.  Nothing in this package
#: reads packs from more than one thread today; the lock is what keeps that
#: from being an assumption a future caller has to know about.
#:
#: REENTRANT, matching `mapped_source_compat._ENGINE_STATE_LOCK`, which
#: guards the same class of process-global swap in the other door.  A
#: nested read does not reach the lock today -- the enclosing read has
#: already widened the table, so `engine_knows_bt_schema` answers True and
#: the inner context yields without acquiring anything -- and that is a
#: property of the widening being idempotent, not a property anybody should
#: have to re-derive.  A plain lock would turn any future change that made
#: the inner path acquire (a second family, a narrower widening) into a
#: silent deadlock instead of a refusal; this makes it a no-op.
_TABLE_LOCK = threading.RLock()


def _engine_module():
    """``woof.obs.goes_pack``, resolved through ``sys.modules``."""

    return importlib.import_module("woof.obs.goes_pack")


def __getattr__(name):
    """`GoesPackError` is the ENGINE's class, resolved when it is asked for.

    Imported here rather than re-declared, so `except obs_pack.GoesPackError`
    catches what the engine's reader actually raises.  A second exception
    class with the same name is how a caller comes to write a handler that
    never fires.
    """

    if name == "GoesPackError":
        return _engine_module().GoesPackError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def bt_schemas() -> tuple[str, ...]:
    """The BT family names: the engine's when it has them, ours otherwise."""

    engine = getattr(_engine_module(), "BT_SCHEMAS", None)
    return tuple(engine) if engine else BT_SCHEMAS


def engine_knows_bt_schema() -> bool:
    """Whether the installed reader's own table lists the BT family.

    Read off `KNOWN_SCHEMAS`, which is what the reader actually checks, not
    off `BT_SCHEMA_V1`: an engine that defined the constant and forgot the
    table row would still refuse every pack, and this must not say it would
    not.
    """

    try:
        known = tuple(_engine_module().KNOWN_SCHEMAS)
    except Exception:  # pragma: no cover - a broken engine install
        return False
    return all(name in known for name in bt_schemas())


@contextmanager
def _bt_family_known():
    """Widen the engine's schema table for the length of one read.

    The table is restored on every exit, including a refused pack: a module
    attribute left widened would make the NEXT read accept a family this
    process never decided to accept.
    """

    if engine_knows_bt_schema():
        yield False
        return
    engine = _engine_module()
    with _TABLE_LOCK:
        original = engine.KNOWN_SCHEMAS
        engine.KNOWN_SCHEMAS = tuple(original) + tuple(
            name for name in bt_schemas() if name not in original)
        try:
            yield True
        finally:
            engine.KNOWN_SCHEMAS = original


def read_goes_pack(path, *, expected_schema=None):
    """Read one ``GPWMGOES`` pack, including the BT family.

    The engine's `read_goes_pack`, with its schema table widened for this
    call when the installed engine does not list the BT family.  Everything
    else about the read is the engine's: the container contract, the
    metadata requirements and the plane bounds all refuse in the engine's
    own words, and `expected_schema` fails closed on the family exactly as
    it does for the two cloud families.
    """

    with _bt_family_known():
        return _engine_module().read_goes_pack(
            path, expected_schema=expected_schema)
