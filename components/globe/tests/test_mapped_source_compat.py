"""The seam between this package and the engine's mapped-source decoder.

Two published-engine differences stop a decode of a real analysis, and both
are adapted in `woof.globe.mapped_source_compat` rather than refused.  This
file holds the adaptations to the three things that make an adaptation
legitimate instead of a fork:

* THE ENGINE IS ASKED FIRST.  Each one begins by trying the installed
  engine's own call and is never reached when that works, so it retires
  itself the day the engine publishes the change.
* NOTHING IS RESTATED.  Neither adaptation contains a second copy of engine
  logic: the scratch translation moves one path into the engine's own
  variable, and the mapping adaptation runs the engine's own validator twice.
* THE RECEIPT SAYS WHAT HAPPENED.  A run records which mechanism placed its
  scratch and whether the mapping rule was adapted, so a reader is never left
  inferring it from a version number.

The environment tests matter more than they look.  A decode that left
`WOOF_COMPOSE_SCRATCH` set behind it would steer the NEXT command in the
same process onto a directory nobody chose, and one that overwrote a value
the caller set would move a multi-GB stream onto the disk that value exists
to avoid.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import types

import pytest

from woof.globe.analysis_initial import PACKAGE_AUTHORITIES_DIR
from woof.globe.mapped_source_compat import (
    COMPOSE_SCRATCH_ENV,
    decode_through_engine,
    engine_scratch,
    engine_takes_scratch_destination,
    load_mapping,
    surface_preserve_mask_fields,
)

GDAS = PACKAGE_AUTHORITIES_DIR / "rw-wps-gdas-global-analysis-grib2.mapping.json"
SURFACE_STATE = PACKAGE_AUTHORITIES_DIR / "rw-wps-gfs-surface-state-grib2.mapping.json"
FLUX = PACKAGE_AUTHORITIES_DIR / "rw-wps-gfs-surface-flux-grib2.mapping.json"


# ------------------------------------------------------------- the variable

def test_the_variable_is_the_engines_own_spelling() -> None:
    """Not a string this package invented: the engine's constant.

    The engine keeps it private, so it is spelled out in the package and
    checked against the engine here.  A package steering the scratch with a
    variable the engine does not read would place nothing and say it had.
    """

    from woof import mapped_composition

    assert COMPOSE_SCRATCH_ENV == mapped_composition._COMPOSE_SCRATCH_ENV


# --------------------------------------------------------------- the scratch

class _FakeDecoder:
    """An engine decoder with one signature or the other."""

    def __init__(self, *, takes_keyword: bool):
        self.calls: list[dict] = []
        self.takes_keyword = takes_keyword

    def install(self, monkeypatch):
        if self.takes_keyword:
            def decode_mapped_source(mapping_path, files, *,
                                     scratch_destination=None, **_rest):
                self.calls.append({
                    "scratch_destination": scratch_destination,
                    "env": os.environ.get(COMPOSE_SCRATCH_ENV),
                })
                return ()
        else:
            def decode_mapped_source(mapping_path, files, **_rest):
                self.calls.append({
                    "scratch_destination": None,
                    "env": os.environ.get(COMPOSE_SCRATCH_ENV),
                })
                return ()

        module = types.ModuleType("woof.mapped_source")
        module.decode_mapped_source = decode_mapped_source
        module.load_mapping = lambda path, **_kw: json.loads(
            Path(path).read_text(encoding="utf-8"))
        monkeypatch.setitem(sys.modules, "woof.mapped_source", module)
        return decode_mapped_source


def test_an_engine_with_the_keyword_is_handed_the_destination(
        monkeypatch, tmp_path) -> None:
    """The old spelling still wins when the engine has it."""

    fake = _FakeDecoder(takes_keyword=True)
    fake.install(monkeypatch)
    assert engine_takes_scratch_destination()
    outdir = tmp_path / "run"
    decoded = decode_through_engine(FLUX, [tmp_path / "x.grib2"],
                                    scratch_destination=outdir)
    assert fake.calls[0]["scratch_destination"] == outdir
    assert fake.calls[0]["env"] is None
    assert decoded.receipt["scratch"]["mechanism"] == "scratch_destination keyword"


def test_an_engine_without_the_keyword_gets_the_variable(
        monkeypatch, tmp_path) -> None:
    """The same placement, spelled the way the published engine reads it.

    The engine creates its scratch IN the directory the variable names, and
    the keyword version creates it in the destination's PARENT, so the
    variable is set to the parent and the two place the stream on the same
    filesystem.
    """

    monkeypatch.delenv(COMPOSE_SCRATCH_ENV, raising=False)
    fake = _FakeDecoder(takes_keyword=False)
    fake.install(monkeypatch)
    assert not engine_takes_scratch_destination()
    outdir = tmp_path / "runs" / "case"
    decoded = decode_through_engine(FLUX, [tmp_path / "x.grib2"],
                                    scratch_destination=outdir)
    assert fake.calls[0]["env"] == str(outdir.resolve().parent)
    assert decoded.receipt["scratch"]["mechanism"] == COMPOSE_SCRATCH_ENV
    assert "this package" in decoded.receipt["scratch"]["placed_by"]
    # The directory the engine will refuse if it does not exist.
    assert outdir.resolve().parent.is_dir()


def test_the_variable_is_restored_after_the_call(
        monkeypatch, tmp_path) -> None:
    """An unset variable comes back unset, not set to empty."""

    monkeypatch.delenv(COMPOSE_SCRATCH_ENV, raising=False)
    fake = _FakeDecoder(takes_keyword=False)
    fake.install(monkeypatch)
    decode_through_engine(FLUX, [tmp_path / "x.grib2"],
                          scratch_destination=tmp_path / "run")
    assert COMPOSE_SCRATCH_ENV not in os.environ


def test_the_variable_is_restored_after_a_failed_decode(
        monkeypatch, tmp_path) -> None:
    """A decode that raises must not leave the next command steered."""

    monkeypatch.delenv(COMPOSE_SCRATCH_ENV, raising=False)

    def decode_mapped_source(mapping_path, files, **_rest):
        raise RuntimeError("the engine refused these bytes")

    module = types.ModuleType("woof.mapped_source")
    module.decode_mapped_source = decode_mapped_source
    module.load_mapping = lambda path, **_kw: {}
    monkeypatch.setitem(sys.modules, "woof.mapped_source", module)

    with pytest.raises(RuntimeError):
        decode_through_engine(FLUX, [tmp_path / "x.grib2"],
                              scratch_destination=tmp_path / "run")
    assert COMPOSE_SCRATCH_ENV not in os.environ


def test_an_empty_prior_value_is_restored_as_empty(
        monkeypatch, tmp_path) -> None:
    """Cleanup RESTORES; it does not delete.

    An empty string is falsy, so the caller-set branch is skipped and the
    placement happens.  Deleting the variable on the way out then ended the
    call with it unset rather than empty, which is a different environment
    from the one the caller had.
    """

    monkeypatch.setenv(COMPOSE_SCRATCH_ENV, "")
    fake = _FakeDecoder(takes_keyword=False)
    fake.install(monkeypatch)
    decode_through_engine(FLUX, [tmp_path / "x.grib2"],
                          scratch_destination=tmp_path / "run")
    assert os.environ[COMPOSE_SCRATCH_ENV] == ""


def test_a_body_that_clears_the_variable_does_not_replace_the_failure(
        monkeypatch, tmp_path) -> None:
    """The decode's own error is what a reader has to see.

    Cleanup used to be `del os.environ[...]`, which raises KeyError out of
    the finally when the body cleared the variable itself, and that
    KeyError arrives in place of whatever the decode actually failed with.
    """

    monkeypatch.delenv(COMPOSE_SCRATCH_ENV, raising=False)
    with pytest.raises(RuntimeError, match="the engine refused these bytes"):
        with engine_scratch(tmp_path / "run"):
            del os.environ[COMPOSE_SCRATCH_ENV]
            raise RuntimeError("the engine refused these bytes")
    assert COMPOSE_SCRATCH_ENV not in os.environ


def test_the_process_global_state_is_taken_under_one_reentrant_lock(
        monkeypatch, tmp_path) -> None:
    """Nesting in one thread works; two threads take turns.

    Both adaptations reach process-global state -- the environment variable
    and `load_mapping` on the engine's module -- so two threads decoding at
    once could restore them out of order and leave the engine's validator
    swapped for a command that never asked for it.  A nested decode in ONE
    thread is supported, which is why the lock is reentrant.
    """

    import threading

    from woof.globe.mapped_source_compat import _ENGINE_STATE_LOCK

    assert isinstance(_ENGINE_STATE_LOCK, type(threading.RLock()))

    monkeypatch.delenv(COMPOSE_SCRATCH_ENV, raising=False)
    run = tmp_path / "run"
    with engine_scratch(run) as outer:
        with engine_scratch(run) as inner:  # reentrant: must not deadlock
            assert os.environ[COMPOSE_SCRATCH_ENV] == str(run.parent)
            # The INNER BLOCK is what a nested decode's receipt records, so
            # it is what this reads.  It found the variable already set and
            # left it alone, and it names who set it: this package one frame
            # up, not "the caller's environment", which would put someone
            # else's name on this package's own directory.
            assert inner["compose_scratch"] == str(run.parent)
            assert inner["placed_by"] == (
                "this package's enclosing placement, which is left alone")
        assert outer["placed_by"].startswith("this package, because")
    assert COMPOSE_SCRATCH_ENV not in os.environ

    order: list[str] = []
    holding = threading.Event()
    let_go = threading.Event()

    def first() -> None:
        with _ENGINE_STATE_LOCK:
            order.append("first in")
            holding.set()
            let_go.wait(5)
            order.append("first out")

    def second() -> None:
        holding.wait(5)
        let_go.set()
        with _ENGINE_STATE_LOCK:
            order.append("second in")

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    assert order == ["first in", "first out", "second in"]


def test_two_threads_placing_scratch_take_turns_and_each_receipt_is_true(
        monkeypatch, tmp_path) -> None:
    """The second decode gets its OWN directory, and says so.

    This drives `engine_scratch` itself, which is the thing a decode calls,
    rather than the lock object beside it.  With the prior value read
    outside the lock the second call did not wait at all: it saw the first
    thread's directory, took the "left alone" branch, staged its frame
    stream under a directory it never named, and wrote a receipt saying the
    caller's environment placed it.
    """

    import threading

    monkeypatch.delenv(COMPOSE_SCRATCH_ENV, raising=False)
    _FakeDecoder(takes_keyword=False).install(monkeypatch)

    first_in = threading.Event()
    let_first_go = threading.Event()
    second_out = threading.Event()
    receipts: dict[str, dict] = {}
    seen: dict[str, str] = {}
    failures: list[BaseException] = []

    def place(name: str, run: Path, entered, release) -> None:
        try:
            with engine_scratch(run) as block:
                receipts[name] = dict(block)
                seen[name] = os.environ.get(COMPOSE_SCRATCH_ENV)
                if entered is not None:
                    entered.set()
                if release is not None:
                    release.wait(10)
        except BaseException as failure:  # surfaced in the main thread
            failures.append(failure)

    first = threading.Thread(
        target=place, args=("A", tmp_path / "runA" / "out",
                            first_in, let_first_go))
    first.start()
    assert first_in.wait(10), "the first placement never happened"

    def second() -> None:
        place("B", tmp_path / "runB" / "out", None, None)
        second_out.set()

    worker = threading.Thread(target=second)
    worker.start()
    # THE MEASUREMENT.  The first thread still holds its placement, so a
    # serialized second call is still blocked here.  Reading the prior
    # value outside the lock let it through immediately.
    assert not second_out.wait(0.5), (
        "the second placement did not wait for the first")
    let_first_go.set()
    for thread in (first, worker):
        thread.join(15)
        assert not thread.is_alive()
    if failures:
        raise failures[0]

    assert seen["A"] == str((tmp_path / "runA").resolve())
    assert seen["B"] == str((tmp_path / "runB").resolve())
    for name in ("A", "B"):
        assert receipts[name]["compose_scratch"] == seen[name]
        assert receipts[name]["placed_by"].startswith("this package, because")
    assert COMPOSE_SCRATCH_ENV not in os.environ


def test_the_callers_own_value_wins_and_is_left_alone(
        monkeypatch, tmp_path) -> None:
    """Someone who set the variable steered the stream deliberately."""

    chosen = tmp_path / "big-disk"
    chosen.mkdir()
    monkeypatch.setenv(COMPOSE_SCRATCH_ENV, str(chosen))
    fake = _FakeDecoder(takes_keyword=False)
    fake.install(monkeypatch)
    decoded = decode_through_engine(FLUX, [tmp_path / "x.grib2"],
                                    scratch_destination=tmp_path / "run")
    assert fake.calls[0]["env"] == str(chosen)
    assert os.environ[COMPOSE_SCRATCH_ENV] == str(chosen)
    assert decoded.receipt["scratch"]["compose_scratch"] == str(chosen)
    assert "caller's environment" in decoded.receipt["scratch"]["placed_by"]


def test_no_destination_leaves_the_engines_own_resolution(
        monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(COMPOSE_SCRATCH_ENV, raising=False)
    fake = _FakeDecoder(takes_keyword=False)
    fake.install(monkeypatch)
    decoded = decode_through_engine(FLUX, [tmp_path / "x.grib2"])
    assert fake.calls[0]["env"] is None
    assert decoded.receipt["scratch"]["mechanism"] == "engine default"


def test_the_context_manager_reports_before_the_call(tmp_path) -> None:
    """The receipt block exists while the call is in flight, not after."""

    with engine_scratch(None) as block:
        assert "mechanism" in block


def test_a_scratch_directory_that_cannot_be_made_is_refused_by_name(
        monkeypatch, tmp_path) -> None:
    """The engine's refusal, one step earlier and in the caller's terms.

    Falling back to the system temp is what the placement exists to avoid,
    so a parent that cannot be created is a refusal naming the run directory
    and the two ways out, not a raw OSError several frames in.
    """

    monkeypatch.delenv(COMPOSE_SCRATCH_ENV, raising=False)
    fake = _FakeDecoder(takes_keyword=False)
    fake.install(monkeypatch)

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    with pytest.raises(NotADirectoryError) as caught:
        decode_through_engine(FLUX, [tmp_path / "x.grib2"],
                              scratch_destination=blocker / "run")
    text = str(caught.value)
    assert COMPOSE_SCRATCH_ENV in text
    assert "--outdir" in text
    assert not fake.calls
    assert COMPOSE_SCRATCH_ENV not in os.environ


# ------------------------------------------- the masked surface record

def test_the_masked_surface_fields_are_read_out_of_the_document() -> None:
    """Granted for what a mapping SAYS, never for a familiar message."""

    assert surface_preserve_mask_fields(GDAS) == ("snow_depth",
                                                  "snow_water_equivalent")
    assert surface_preserve_mask_fields(SURFACE_STATE) == (
        "snow_water_equivalent", "vegetation_fraction")
    # The flux product declares none, so it never reaches the adaptation.
    assert surface_preserve_mask_fields(FLUX) == ()
    assert surface_preserve_mask_fields(Path("nowhere.json")) == ()


def test_the_carried_mappings_load_on_the_installed_engine() -> None:
    """The whole point: the cold start's mapping validates here.

    woof 2.7.0's `load_mapping` refuses `preserve_mask` on a surface record
    -- "currently restricted to soil fields" -- and the refusal lands before
    a byte of GRIB is read, so on a published engine every shipped GDAS
    experiment stopped at its first door.  Through the seam it loads, with
    the missing policies the document really declares.
    """

    for path in (GDAS, SURFACE_STATE, FLUX):
        mapping = load_mapping(path)
        declared = json.loads(path.read_text(encoding="utf-8"))["fields"]
        for name in surface_preserve_mask_fields(path):
            assert mapping["fields"][name]["missing"] == declared[name]["missing"]
            assert mapping["fields"][name]["missing"]["kind"] == "preserve_mask"
        assert mapping["schema"] == "rw-wps.mapping.v1"
        assert isinstance(mapping["name"], str)


def test_a_mapping_that_is_wrong_is_still_refused(tmp_path) -> None:
    """The adaptation widens one rule, not the validator.

    A document broken in any other way earns the engine's own refusal, in
    the engine's own words.  Waving one through would make this a second
    validator, which is the thing it exists not to be.
    """

    document = json.loads(GDAS.read_text(encoding="utf-8"))
    name = surface_preserve_mask_fields(GDAS)[0]
    document["fields"][name]["units"]["target"] = 17  # not a string
    broken = tmp_path / GDAS.name
    broken.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        load_mapping(broken)
    assert "units.target" in str(caught.value)


def test_an_engine_that_accepts_it_is_not_adapted(monkeypatch, tmp_path) -> None:
    """The adaptation retires itself, and the receipt says it did."""

    fake = _FakeDecoder(takes_keyword=True)
    fake.install(monkeypatch)  # its load_mapping accepts everything
    decoded = decode_through_engine(GDAS, [tmp_path / "x.grib2"])
    assert decoded.receipt["surface_preserve_mask"] == (
        "accepted by the installed engine; no adaptation")
    assert decoded.receipt["masked_surface_fields"] == [
        "snow_depth", "snow_water_equivalent"]


def test_the_engines_own_function_is_restored_after_the_decode(
        tmp_path) -> None:
    """A module attribute left swapped changes the NEXT command."""

    from woof import mapped_source as engine

    before = engine.load_mapping
    with pytest.raises(Exception):
        decode_through_engine(GDAS, [tmp_path / "absent.grib2"])
    assert engine.load_mapping is before


def test_the_decoder_calling_its_own_load_mapping_does_not_recurse(
        monkeypatch, tmp_path) -> None:
    """The shape the real engine has, and the one that cost a cold start.

    `woof.mapped_source._decode_through_engine` calls its own module-global
    `load_mapping`, which is exactly why the adaptation has to swap that
    attribute.  A stand-in that then resolved "the engine's validator" by
    reading the same attribute would find ITSELF: measured 2026-09-09,
    when a real GDAS cold start died with "maximum recursion depth
    exceeded" after 987 frames of that, and every unit test passed because
    none of them had a decoder that called back into the module.

    This one does.  It also refuses a surface preserve_mask the way a
    published engine does, so the adaptation is really installed while the
    call-back happens.
    """

    import types

    module = types.ModuleType("woof.mapped_source")
    seen: list = []

    def engine_load_mapping(path, *, _raw=None):
        document = (_raw if _raw is not None
                    else json.loads(Path(path).read_text(encoding="utf-8")))
        for name, field in document["fields"].items():
            missing = field.get("missing") or {}
            if (missing.get("kind") == "preserve_mask"
                    and field.get("location") == "surface"):
                raise ValueError(
                    f"fields.{name} preserve_mask is currently restricted to "
                    "soil fields repaired by the land/water-aware initializer")
        return document

    def decode_mapped_source(mapping_path, files, **_rest):
        # What the engine does: its own module-global, mid-decode.
        seen.append(module.load_mapping(mapping_path))
        return ()

    module.load_mapping = engine_load_mapping
    module.decode_mapped_source = decode_mapped_source
    monkeypatch.setitem(sys.modules, "woof.mapped_source", module)

    decoded = decode_through_engine(GDAS, [tmp_path / "x.grib2"])
    assert "adapted" in decoded.receipt["surface_preserve_mask"]
    # The decode saw the document with its REAL missing policies.
    assert seen[0]["fields"]["snow_depth"]["missing"] == {
        "kind": "preserve_mask"}
    # And the engine's own function is back.
    assert module.load_mapping is engine_load_mapping


# ------------------------------------------------------- one door, checked


def _direct_engine_decode_calls(path: Path) -> list[int]:
    """Line numbers where a file reaches the engine's decoder itself.

    Parsed, not grepped: `analysis_initial` and `engine_compat` both NAME
    `woof.mapped_source.decode_mapped_source` in prose, saying that this
    package does not call it, and a text search cannot tell that sentence
    from a call.  An import of the name and an attribute call on the
    module both count; a mention in a docstring or a comment does not.
    """

    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "woof.mapped_source" and any(
                    alias.name == "decode_mapped_source"
                    for alias in node.names):
                hits.append(node.lineno)
        elif isinstance(node, ast.Attribute):
            if node.attr == "decode_mapped_source":
                hits.append(node.lineno)
    return sorted(set(hits))


def test_decode_through_engine_is_the_only_route_to_the_engines_decoder(
) -> None:
    """One door, held to it by the tree rather than by a sentence.

    THE BREAKAGE THIS PREVENTS, measured: `tools/arwen_global_bias_vs_gfs.py`
    called `decode_mapped_source` directly and so hit the published
    engine's surface `preserve_mask` refusal -- the exact refusal the shim
    exists to translate -- while every shipped route decoded the same
    analysis fine.  The scratch placement, the mask adaptation and the
    receipt all live behind `decode_through_engine`; a caller that walks
    around it gets none of them and says nothing about it.
    """

    import woof.globe

    package = Path(woof.globe.__file__).resolve().parent
    offenders: dict[str, list[int]] = {}
    for source in sorted(package.rglob("*.py")):
        if source.name == "mapped_source_compat.py":
            continue
        hits = _direct_engine_decode_calls(source)
        if hits:
            offenders[str(source.relative_to(package))] = hits
    assert offenders == {}, offenders

    # The instruments beside the package, which are not installed but are
    # shipped in the sdist and named by path in the door pages.
    tools = Path(__file__).resolve().parents[1] / "tools"
    if not tools.is_dir():  # pragma: no cover - a wheel-only checkout
        pytest.skip(f"{tools} is not in this checkout: the sdist grafts it, "
                    "so this half of the door check runs where tools/ is")
    tool_offenders: dict[str, list[int]] = {}
    for source in sorted(tools.rglob("*.py")):
        hits = _direct_engine_decode_calls(source)
        if hits:
            tool_offenders[str(source.relative_to(tools))] = hits
    assert tool_offenders == {}, tool_offenders


def test_a_relative_destination_reaches_the_receipt_absolute(
        monkeypatch, tmp_path) -> None:
    """A receipt a reader cannot follow is not a receipt.

    `--outdir out/tip` is the shape people actually type, and recorded as
    typed it names a different directory on every machine and every working
    directory that opens the receipt.  Both spellings go in: the typed one
    because it is what the caller recognises, the resolved one because it
    is the only one that places the run.
    """

    monkeypatch.delenv(COMPOSE_SCRATCH_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    fake = _FakeDecoder(takes_keyword=False)
    fake.install(monkeypatch)
    decoded = decode_through_engine(FLUX, [tmp_path / "x.grib2"],
                                    scratch_destination=Path("out") / "tip")
    scratch = decoded.receipt["scratch"]
    assert scratch["scratch_destination"] == str(Path("out") / "tip")
    resolved = Path(scratch["scratch_destination_resolved"])
    assert resolved.is_absolute()
    assert resolved == (tmp_path / "out" / "tip").resolve()
    # The absolute placement is the parent of the resolved destination,
    # never of the typed one.
    assert Path(scratch["compose_scratch"]) == resolved.parent


def test_the_keyword_route_records_both_spellings(monkeypatch, tmp_path) -> None:
    """The engine-signature route writes the same two rows.

    The day the engine takes `scratch_destination` this package stops
    placing anything, and a receipt that lost the resolved path on that
    day would be a receipt that got worse when the engine got better.
    """

    monkeypatch.chdir(tmp_path)
    fake = _FakeDecoder(takes_keyword=True)
    fake.install(monkeypatch)
    decoded = decode_through_engine(FLUX, [tmp_path / "x.grib2"],
                                    scratch_destination=Path("out") / "tip")
    scratch = decoded.receipt["scratch"]
    assert scratch["mechanism"] == "scratch_destination keyword"
    assert scratch["scratch_destination"] == str(Path("out") / "tip")
    assert Path(scratch["scratch_destination_resolved"]) == (
        tmp_path / "out" / "tip").resolve()


def test_the_receipt_names_the_engine_that_decoded() -> None:
    """A number with no host is not a measurement; nor is a receipt."""

    import woof

    with engine_scratch(None) as block:
        assert block["scratch_destination"] is None
        assert block["scratch_destination_resolved"] is None
    # The version is read from the installed engine, not restated.
    assert woof.__version__
