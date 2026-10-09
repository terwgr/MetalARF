"""The one door between this package and the engine's mapped-source decoder.

Every command that turns a real analysis into a model state goes through
:func:`decode_through_engine`: the cold start, the ATMS column product, the
radiation reference cover, the surface-energy STATE product and the
upper-air scorecard's reference.  The surface-energy FLUX ladder is the one
route that runs no decode at all: its records are product-definition-template
8, which the mapping grammar cannot bind as a model input, so it reads them
through the engine's GRIB2 inventory and dump bridges and reaches this module
only for :func:`load_mapping`, which is where the validator adaptation below
lives; what it can record, and does, is how its document was read and which
of its records the published validator would have narrowed.

It exists because a PUBLISHED engine is not the tree this model was developed
in, and the two differences that stop a decode are both narrow enough to
adapt in place without a second decoder existing anywhere.  Each adaptation
is recorded in the receipt block the caller keeps -- the cold start's
``decode``, the ATMS columns' sidecar, the radiation reference's per-
checkpoint entry, the surface-energy state block and the upper-air
reference's ``source`` -- so a run says which mechanism it actually used
rather than leaving a reader to infer it from a version number.

THE ENGINE IS ASKED FIRST, every time.  Each adaptation below begins by
trying the installed engine's own call.  The day a published engine carries
the change, the adaptation is never reached and the receipt says so, which is
how it retires itself instead of quietly becoming a fork.

WHAT IS ADAPTED, AND WHY EACH IS SAFE

1. WHERE THE SCRATCH GOES.  The decoder stages the whole decoded frame
   stream on disk -- several GB of float64 for one 0.25-degree global
   analysis at full level count.  The tree this model was graded in takes
   the run directory as a `scratch_destination` keyword; woof 2.7.0 has no
   such keyword and places the same scratch by the `WOOF_COMPOSE_SCRATCH`
   environment variable (`woof.mapped_source._engine_scratch_directory`,
   `woof.mapped_composition._compose_scratch_base`).  Two spellings, one
   placement.  Named breakage, and it is measured rather than imagined: on a
   box whose `/tmp` is a quota-limited tmpfs the decode died with "cannot
   write the frame stream: Disk quota exceeded (os error 122)".  Dropping
   the argument to make the call go through would put the stream back on
   that filesystem, so it is translated instead.

   The USER'S value wins.  Someone who set `WOOF_COMPOSE_SCRATCH` steered
   the stream deliberately, and a package that overwrote it would move the
   stream to the disk the variable was set to avoid.  The receipt says which
   of the two placed it.

2. A SURFACE FIELD THAT KEEPS ITS MASK.  Four of the six source mappings
   this model reads declare `missing.kind = "preserve_mask"` on a surface
   record: the GDAS analysis's snow water and snow depth, whose GRIB bitmap
   is unset over open water, and the GFS vegetation record, whose bitmap is
   off over every water cell.  woof 2.7.0's `load_mapping` refuses that
   combination -- "preserve_mask is currently restricted to soil fields" --
   and the refusal lands before a single byte is read, so on a published
   engine every shipped GDAS experiment stopped at its first door.

   The engine's own Rust decoder does NOT refuse it: measured 2026-09-09 on
   the desktop, `gpuwm_mapped_engine inspect` on the carried GDAS mapping
   passes mapping validation and goes on to read the input bytes.  So the
   document is executable by the engine that decodes it and is refused only
   by the Python pre-check in front of it.

   This does not reimplement that pre-check.  It runs the engine's own
   validator over the whole document, and when the ONLY thing the engine
   objected to is that one rule on fields the document really does declare
   that way, it runs the engine's validator a second time over a copy whose
   sole difference is those fields' missing policy -- so every other rule in
   the document is still checked by the engine -- and hands back a document
   whose missing policies are the originals.  Any other refusal is re-raised
   untouched.  A mapping that is wrong is still refused, in the engine's own
   words.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import inspect
import json
import os
from pathlib import Path
import threading
from typing import NamedTuple, Sequence

__all__ = [
    "COMPOSE_SCRATCH_ENV",
    "DecodedSource",
    "decode_through_engine",
    "engine_scratch",
    "engine_takes_scratch_destination",
    "load_mapping",
    "surface_preserve_mask_fields",
]

#: The engine's own variable, spelled here because the published engine reads
#: it from a private module constant.  Held to the engine's spelling by
#: tests/test_mapped_source_compat.py, which reads it off the engine.
COMPOSE_SCRATCH_ENV = "WOOF_COMPOSE_SCRATCH"

#: Both adaptations below reach process-global state for the length of one
#: decode: the environment variable above, and `load_mapping` on the engine's
#: module.  Two threads decoding at once would restore them out of order and
#: leave the engine's validator permanently swapped or the variable
#: permanently set, so the state a LATER command reads would be one this
#: process never chose.  Nothing in this package decodes from more than one
#: thread today; the lock is what keeps that from being an assumption a
#: future caller has to know about.  REENTRANT, because a nested decode in
#: one thread is supported (`_ENGINE_LOAD_MAPPING` is a stack) and a plain
#: lock held across the yield would deadlock it.
_ENGINE_STATE_LOCK = threading.RLock()

#: The scratch directories THIS PACKAGE has placed and not yet restored, as
#: a stack, under `_ENGINE_STATE_LOCK`.  A nested decode finds the variable
#: already set and leaves it alone, and without this it would report the
#: value as the caller's when this package set it one frame up.  A receipt
#: that names the wrong placer is the one thing this block exists to say.
_PLACED_SCRATCH: list[str] = []

#: The exact sentence woof 2.7.0 refuses a surface preserve_mask with.  The
#: refusal is matched on the ENGINE's own words and then VERIFIED against the
#: document, never on the words alone: a message match by itself would be a
#: guess, and a guess here would wave through a mapping that is genuinely
#: malformed.
_SURFACE_PRESERVE_MASK_REFUSAL = "preserve_mask is currently restricted to"


class DecodedSource(NamedTuple):
    """The frames a decode produced, and how it was obtained."""

    frames: tuple
    #: What placed the scratch, what was adapted, and which engine did it.
    #: Goes into the run receipt beside the mapping digest.
    receipt: dict


def _engine_module():
    """``woof.mapped_source``, resolved through ``sys.modules``.

    ``from woof import mapped_source`` reads an attribute of the package
    object and would miss a module a caller has replaced, which is how the
    adaptation below could end up swapping a function on one module while
    the decode called another.
    """

    import importlib

    return importlib.import_module("woof.mapped_source")


def engine_takes_scratch_destination() -> bool:
    """Whether the installed decoder takes the scratch destination keyword.

    Read off the SIGNATURE, not off a version: a version is a promise and a
    signature is the thing that either accepts the keyword or raises
    TypeError after the GRIB has been opened.
    """

    try:
        from woof.mapped_source import decode_mapped_source
    except Exception:  # pragma: no cover - a broken engine install
        return False
    try:
        signature = inspect.signature(decode_mapped_source)
    except (TypeError, ValueError):  # pragma: no cover - a C callable
        return False
    return "scratch_destination" in signature.parameters


@contextmanager
def engine_scratch(destination):
    """Place the decoder's scratch where ``destination`` says, and say how.

    Yields the receipt block for the placement.  On exit the environment is
    exactly what it was, including the case where the variable was unset:
    a decode must not leave a variable behind that steers the next command.

    SERIALIZED from the read of the variable to the restore of it, on every
    route that touches it, so that a second thread waits for the first
    rather than staging its frame stream under the first one's directory
    and reporting the caller placed it.  Reentrant in one thread.
    """

    # BOTH SPELLINGS OF THE DESTINATION.  The typed value is what a caller
    # recognises as the thing they asked for; the resolved one is what a
    # reader of the receipt can follow.  A relative `--outdir out/tip`
    # reaches the receipt as "out/tip", which names a different directory
    # on every machine and every working directory that reads it -- the
    # same shape the mapping path already resolves out of, and the reason
    # `compose_scratch` below has always been absolute.
    block: dict[str, object] = {
        "scratch_destination": None if destination is None else str(destination),
        "scratch_destination_resolved": (
            None if destination is None else str(Path(destination).resolve())),
    }
    if engine_takes_scratch_destination():
        # The only route that reaches no process-global state at all: the
        # destination rides the call.  Nothing to serialize.
        block["mechanism"] = "scratch_destination keyword"
        block["placed_by"] = "the engine's own signature"
        yield block
        return

    # EVERY REMAINING ROUTE READS THE VARIABLE, so the read and the write
    # are ONE critical section, held across the decode.  Reading it outside
    # the lock is not a smaller version of the same thing: a second thread
    # then sees the first thread's directory as "the caller's environment",
    # stages its frame stream under a directory it did not name, and writes
    # a receipt saying the caller placed it.  Measured on the desktop
    # against woof 2.7.0, 2026-09-10: the second call did not block for a
    # measurable moment and its receipt read `placed_by = "the caller's
    # environment, which is left alone"` when this package had placed it.
    # REENTRANT, so a nested decode in one thread still passes through.
    with _ENGINE_STATE_LOCK:
        if destination is None:
            block["mechanism"] = "engine default"
            block["placed_by"] = (
                f"the engine's own resolution ({COMPOSE_SCRATCH_ENV} when "
                "set, the system temp otherwise)")
            yield block
            return

        existing = os.environ.get(COMPOSE_SCRATCH_ENV)
        if existing:
            block["mechanism"] = COMPOSE_SCRATCH_ENV
            block["compose_scratch"] = existing
            if _PLACED_SCRATCH and existing == _PLACED_SCRATCH[-1]:
                # A nested decode inside one this package already placed.
                # Saying "the caller's environment" here would be a receipt
                # naming someone else for this package's own directory.
                block["placed_by"] = (
                    "this package's enclosing placement, which is left "
                    "alone")
            else:
                block["placed_by"] = (
                    "the caller's environment, which is left alone")
            yield block
            return

        yield from _place_scratch(destination, block)


def _place_scratch(destination, block: dict):
    """Set the variable, yield the receipt block, and put it back.

    Called with ``_ENGINE_STATE_LOCK`` HELD, which is what makes the read of
    the prior value above and the write here one critical section.
    """

    # The owner tree's keyword creates the scratch in the destination's
    # PARENT -- the same disk-backed filesystem the run's own output lands
    # on -- while the destination itself stays untouched, so a create-only
    # refusal there keeps meaning something.  The variable names the
    # directory the scratch is created IN, so the parent is what it is set
    # to, and the engine refuses a directory that does not exist rather than
    # falling back to the temp this exists to avoid.
    base = Path(destination).resolve().parent
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as failure:
        # The engine refuses a scratch directory that does not exist, by
        # name, rather than falling back to the system temp -- and it is
        # right to, because the fallback is the filesystem this placement
        # exists to avoid.  This is the same refusal one step earlier, said
        # in terms of the run directory the caller actually named.
        raise NotADirectoryError(
            f"cannot create {base}, the directory the decoder's frame "
            f"stream would be staged in beside the run output "
            f"{destination}: {failure}.\n"
            "The stream is several GB of float64 for one global analysis, "
            "and the alternative is the system temp, which is a "
            "quota-limited tmpfs on common Linux boxes and is where this "
            "decode has died before.\n"
            f"Point --outdir at a writable disk, or set "
            f"{COMPOSE_SCRATCH_ENV} to a directory that exists.") from failure
    block["mechanism"] = COMPOSE_SCRATCH_ENV
    block["placed_by"] = (
        "this package, because the installed decoder takes no "
        "scratch_destination keyword")
    block["compose_scratch"] = str(base)
    # `existing` is what the variable held before this call, read under the
    # same lock, and reaching here it is unset or empty.  RESTORED rather
    # than deleted on the way out: a `del` raises KeyError out of the
    # finally when the body cleared the variable itself, and that KeyError
    # replaces whatever the decode actually failed with; and a caller whose
    # value was the empty string would end the call with the variable unset
    # instead of empty.
    existing = os.environ.get(COMPOSE_SCRATCH_ENV)
    os.environ[COMPOSE_SCRATCH_ENV] = str(base)
    _PLACED_SCRATCH.append(str(base))
    try:
        yield block
    finally:
        _PLACED_SCRATCH.pop()
        if existing is None:
            os.environ.pop(COMPOSE_SCRATCH_ENV, None)
        else:
            os.environ[COMPOSE_SCRATCH_ENV] = existing


def surface_preserve_mask_fields(mapping_path) -> tuple[str, ...]:
    """The fields a mapping declares as a masked SURFACE record, by name.

    Read out of the document, so the adaptation below is granted for what a
    mapping really says and never for a message that merely looks familiar.
    """

    try:
        document = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    fields = document.get("fields")
    if not isinstance(fields, dict):
        return ()
    found = []
    for name, field in fields.items():
        if not isinstance(field, dict):
            continue
        missing = field.get("missing")
        if not isinstance(missing, dict):
            continue
        if (missing.get("kind") == "preserve_mask"
                and field.get("location") == "surface"):
            found.append(str(name))
    return tuple(sorted(found))


#: The engine's OWN `load_mapping`, while this module's wrapper is standing
#: in its place on the engine module.  A stack, so a nested decode restores
#: in order.
#:
#: WHY IT IS SAVED RATHER THAN READ BACK.  The adaptation below has to swap
#: the attribute, because `woof.mapped_source._decode_through_engine` calls
#: its own module-global `load_mapping` and that call is the one that refuses.
#: A wrapper that then resolved "the engine's function" by reading that same
#: attribute would find ITSELF and recurse until the interpreter stopped it:
#: measured 2026-09-09 on a real GDAS cold start, which died with "maximum
#: recursion depth exceeded" after 987 frames of exactly that.
_ENGINE_LOAD_MAPPING: list = []


def _engine_load_mapping():
    """The engine's own validator, never this module's stand-in."""

    if _ENGINE_LOAD_MAPPING:
        return _ENGINE_LOAD_MAPPING[-1]
    return _engine_module().load_mapping


def _load_mapping(engine_load_mapping, path, *, _raw=None):
    """One mapping through ``engine_load_mapping``, the narrowing adapted.

    Returns what the engine returns.  The second validation pass differs
    from the first in exactly one way -- those fields' `missing` block is
    the `"reject"` policy instead of `"preserve_mask"` -- and that is inert
    for every other rule the validator applies: the engine reads
    `missing.kind` again only to check that a `landmask_water` field has a
    land mask to key from, which no field here declares.  The originals are
    put back before the document is returned, so nothing downstream sees the
    substitution.
    """

    try:
        if _raw is None:
            return engine_load_mapping(path)
        return engine_load_mapping(path, _raw=_raw)
    except ValueError as refusal:
        if _SURFACE_PRESERVE_MASK_REFUSAL not in str(refusal):
            raise
        masked = surface_preserve_mask_fields(path)
        if not masked:
            raise
        probe = (copy.deepcopy(_raw) if _raw is not None
                 else json.loads(Path(path).read_text(encoding="utf-8")))
        try:
            originals = {name: probe["fields"][name]["missing"]
                         for name in masked}
        except (KeyError, TypeError):  # pragma: no cover - not this refusal
            raise refusal from None
        for name in masked:
            probe["fields"][name]["missing"] = {"kind": "reject"}
        validated = engine_load_mapping(path, _raw=probe)
        for name, block in originals.items():
            validated["fields"][name]["missing"] = block
        return validated


def load_mapping(path, *, _raw=None):
    """The engine's `load_mapping`, with the one narrowing adapted.

    The package's door to the engine's mapping validator: every module in
    this package that reads a mapping document reads it through this, so a
    reader and a decode cannot disagree about whether a document is
    executable.  `surface_energy.decode_gfs_flux` read one document around
    it and was inert only because the flux mapping declares no masked
    surface record; four of the six carried mappings do, so the day a re-cut
    gave the flux document one, that site alone would have refused and the
    receipt would have said nothing about why.
    """

    return _load_mapping(_engine_load_mapping(), path, _raw=_raw)


@contextmanager
def _surface_preserve_mask_adaptation(mapping_path):
    """Adapt the engine's mapping validator for the length of one decode.

    Yields the receipt block.  The engine's own function is restored on
    every exit, including a failed decode: a module attribute left swapped
    would change how the NEXT command in the same process validates.
    """

    engine = _engine_module()

    block: dict[str, object] = {"surface_preserve_mask": "not needed"}
    masked = surface_preserve_mask_fields(mapping_path)
    if not masked:
        yield block
        return
    original = _engine_load_mapping()
    try:
        original(Path(mapping_path))
    except ValueError as refusal:
        if _SURFACE_PRESERVE_MASK_REFUSAL not in str(refusal):
            # A mapping this package carries is malformed, or the engine
            # objects for some other reason.  Not this adaptation's
            # business: let the decode raise it in the engine's own words.
            # The block still names the masked records and the reason,
            # because a receipt reading "not needed" about a document that
            # DOES declare them says the opposite of what the file says.
            block["surface_preserve_mask"] = (
                "the installed engine refused this document for another "
                "reason; the decode raises it in the engine's own words")
            block["masked_surface_fields"] = list(masked)
            yield block
            return
    except Exception as failure:
        # The validator could not be RUN (a broken engine install, an
        # unreadable document).  The decode below meets the same thing; the
        # block must not claim the adaptation was not needed when what
        # happened is that the question was never answered.
        block["surface_preserve_mask"] = (
            "the installed engine's validator could not be run "
            f"({failure.__class__.__name__}); no adaptation was attempted")
        block["masked_surface_fields"] = list(masked)
        yield block
        return
    else:
        block["surface_preserve_mask"] = (
            "accepted by the installed engine; no adaptation")
        block["masked_surface_fields"] = list(masked)
        yield block
        return

    block["surface_preserve_mask"] = (
        "adapted: the installed engine's load_mapping restricts preserve_mask "
        "to soil fields, and its own decoder accepts these records")
    block["masked_surface_fields"] = list(masked)

    def adapted(path, *, _raw=None):
        return _load_mapping(original, path, _raw=_raw)

    with _ENGINE_STATE_LOCK:
        _ENGINE_LOAD_MAPPING.append(original)
        engine.load_mapping = adapted
        try:
            yield block
        finally:
            engine.load_mapping = original
            _ENGINE_LOAD_MAPPING.pop()


def decode_through_engine(
    mapping_path, files: Sequence, *, scratch_destination=None,
) -> DecodedSource:
    """Decode one mapped source through the installed engine.

    The only route this package takes to `decode_mapped_source`.  The
    engine's decoder does the byte work; this places its scratch, adapts the
    one mapping rule a published engine narrowed, and records both in a
    receipt block the caller keeps beside the mapping digest.
    """

    from woof.mapped_source import decode_mapped_source

    inputs = [Path(item) for item in files]
    receipt: dict[str, object] = {
        "mapping": str(mapping_path),
        "inputs": [str(item) for item in inputs],
    }
    with engine_scratch(scratch_destination) as scratch, \
            _surface_preserve_mask_adaptation(mapping_path) as adaptation:
        receipt["scratch"] = dict(scratch)
        receipt.update(adaptation)
        if scratch["mechanism"] == "scratch_destination keyword":
            frames = decode_mapped_source(
                mapping_path, inputs, scratch_destination=scratch_destination)
        else:
            frames = decode_mapped_source(mapping_path, inputs)
    try:
        from woof import __version__ as engine_version
    except Exception:  # pragma: no cover - a broken engine install
        engine_version = None
    receipt["engine_version"] = engine_version
    return DecodedSource(tuple(frames), receipt)
