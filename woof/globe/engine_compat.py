"""The seam where this package meets an engine that is not quite there yet.

The carve measured this distribution's boundary symbol by symbol against the
engine it depends on.  `tools/measure_boundary.py` is that measurement and it
is re-run rather than transcribed, because a count in a docstring ages: this
one said eighty-two while the tool said seventy-six, and said four while the
list under it had five entries and `GAPS` below had six rows.

WHAT THIS MODULE IS NOW, AND WHAT IT STOPPED BEING.  Until 2026-09-09 it
carried a contract, forwarded to two float64 reference functions and refused
five signatures, because the physics this model was graded with sat in a
source tree and the published engine had a different one.  That is no longer
a gap to stand in: the modules and kernels the model executes are CARRIED, in
`woof.globe.core`, cut from the tree the model was graded in.  A carried
callable's signature is this package's own, and no installed engine can
present a different one, so the five signature refusals and the YSU
mixing-length contract are retired -- not weakened, retired, because a guard
whose defect is fixed teaches a reader to expect a failure that cannot
happen.

What remains is what the carve genuinely left on the engine's side:

    woof.core.preflight.measured_free_vram_bytes
    woof.verify.harness.surface_bias.interpolate_to_tape
    woof.mapped_source.decode_mapped_source(scratch_destination=...)

They are patch items the engine has to carry, and the carve wrote the series
out for the engine's owner.

THE RULE, and it is the same one `woof.globe.obs_doors` follows: the ENGINE
IS ASKED FIRST, every time, for every symbol.  Nothing here shadows a symbol
the engine has.  `tests/test_engine_compat.py` fails the day the engine grows
one, which is how the shim retires itself instead of quietly becoming a fork.

A COMPUTATION IS REFUSED, never reimplemented.  `measured_free_vram_bytes`
measures a card and `interpolate_to_tape` interpolates a reference field.
Reimplementing either here would produce a SECOND instrument answering the
same question, and a second instrument that disagrees with the first is worse
than no answer: the number would be reported with the same confidence and be
wrong in a way nobody could see.  So each one refuses by name, says which
command it stops, and names the engine change that fixes it.

A CARRY IS NOT A REFUSAL AND NOT A STAND-IN.  The distinction matters because
the three mechanisms fail differently.  A refusal stops a command.  A stand-in
answers with this package's copy of somebody else's value.  A carry means the
code is HERE and runs, and what the engine still supplies underneath it is
pinned in `woof/globe/data/engine-seam.json` and reported file by file by
the doctor.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect

__all__ = [
    "EngineGap",
    "MissingEngineSymbol",
    "SignatureGap",
    "GAPS",
    "SIGNATURE_GAPS",
    "engine_gaps",
    "engine_signature_gaps",
    "require_engine_signature",
    "interpolate_to_tape",
    "measured_free_vram_bytes",
]


class MissingEngineSymbol(RuntimeError):
    """The installed engine does not carry something a command needs."""


@dataclass(frozen=True)
class EngineGap:
    """One symbol the engine may or may not have, and what turns on it."""

    module: str
    symbol: str
    #: What this package cannot do without it.
    stops: str
    #: Carried here (a contract) or refused (a computation).
    handling: str
    #: Whether a native-physics FORECAST is one of the things it stops.
    #:
    #: NO DEFAULT, on purpose.  Before the carve every row here was a native
    #: column symbol, so "a symbol gap stands" and "the native suite cannot
    #: run" were the same sentence and the run-plan documents were built on
    #: that identity.  The carve took the column physics into this package
    #: and left two rows that stop neither a forecast nor any door on the
    #: forecast path, and the identity became false while the code still
    #: read it: `run-plan --physics-profiles` reported the native profile
    #: `admissible: false` with a `why_not` naming a card-pricing check and
    #: a scorecard regrid, in the same document whose own statement says
    #: each gap "stops what its row names and nothing else".
    #:
    #: A row without this field will not construct, so a symbol added later
    #: cannot inherit an answer nobody gave.
    stops_a_native_forecast: bool
    #: Whether a documented command reaches the refusal: a subcommand of the
    #: console script, or a `python -m` entry the README or a page under
    #: docs/ tells a reader to run.  It decides the doctor's verdict: a row that
    #: stops a documented command is a gap and moves the exit code, and one
    #: no documented command reaches is reported as optional.
    #:
    #: NO DEFAULT, for the reason the field above has none.  The doctor
    #: used to grade every refused row a gap, and its exit code is "every
    #: documented command can run"; neither row left here stops one, so a
    #: correct 0.1.2 install against the published 2.8.0 exited 1 and the
    #: README had to tell readers to ignore the exit code.
    #: `tests/test_engine_signature_gaps.py` walks the package's call graph
    #: and fails when a row that says False becomes reachable from a
    #: documented entry.
    stops_a_documented_command: bool

    def present(self) -> bool:
        try:
            loaded = importlib.import_module(self.module)
        except Exception:
            return False
        return hasattr(loaded, self.symbol)


#: Every symbol this module stands in for, in the order the patch series
#: numbers them.
#:
#: RETIRED 2026-09-09, when the carve took the physics into this package:
#: `woof.core.ysu_contract`'s two constants (carried outright, at
#: `woof.globe.core.ysu_contract`, kernel included) and
#: `tests.physics_reference.npref`'s `np_bound_cloud_sizes` and
#: `np_max_random_total_cloud_cover` (carried at `woof.globe.core.npref`,
#: which is the mirror of the carried kernels and so the only mirror that
#: grades them).  No stand-in was kept for any of the four: nothing outside
#: the carried core imports them from the engine any more, and a forwarder
#: that nothing calls is a second name for one thing.
GAPS: tuple[EngineGap, ...] = (
    EngineGap(
        module="woof.core.preflight",
        symbol="measured_free_vram_bytes",
        # NOT "`woof global sizing`", which is what this row said until
        # 2026-09-07 and is a command this package has never had: the parser
        # answers "invalid choice: 'sizing'".  A doctor line that sends a
        # reader to a command that does not exist is worse than one that
        # names nothing, because the reader spends the trip finding out.
        #
        # The only caller is `sizing.global_check_main`, the standalone
        # card-pricing check, and no subcommand is wired to it, so the
        # breakage is that it stops that check and that no door
        # reaches it today.  `run` and `go` price the card
        # through `run_memory_gate`, which does not read this symbol, so a
        # forecast is not stopped by this gap and the report must not imply
        # that it is.
        stops="the standalone card-pricing check "
              "(`sizing.global_check_main`), which no subcommand is wired "
              "to yet; `run` and `go` price the card by another path and "
              "are not stopped",
        handling="refused",
        # Measured on the Windows desktop 2026-09-10 against woof 2.7.2:
        # `run` and `go` price the card through `sizing.run_memory_gate`,
        # which does not read this symbol.  No forecast reaches it.
        stops_a_native_forecast=False,
        # No subcommand of `woof global` or of the engine's `woof global`
        # is wired to `global_check_main`, and the engine's `woof check`
        # does not call into this package.  Wiring a door to it makes this
        # True, and then the doctor reports the row as a gap.
        stops_a_documented_command=False),
    EngineGap(
        module="woof.verify.harness.surface_bias",
        symbol="interpolate_to_tape",
        stops="regridding a reference analysis onto the tape in the surface "
              "energy scorecard (`python -m woof.globe.surface_energy`, "
              "a research entry no page documents)",
        handling="refused",
        # The only caller is `surface_energy.interpolate_to_grid`, reached
        # from `surface_energy.regrid_reference`, which is scorecard code:
        # no subcommand and no stage of `run` or `go` enters it.  Measured
        # the same day, on the same install.
        stops_a_native_forecast=False,
        # `localize` and `bias_planes` reach it, and only the module's own
        # `python -m` entry calls them, which neither the README nor any
        # page under docs/ tells a reader to run.  The radiation scorecard,
        # which docs/ARWEN_GLOBAL_LEVEL5.md does document, regrids with its
        # own function and does not read this symbol.  Documenting the
        # surface-energy entry makes this True.
        stops_a_documented_command=False),
)


def engine_gaps() -> tuple[EngineGap, ...]:
    """The gaps that are still gaps on the INSTALLED engine."""

    return tuple(gap for gap in GAPS if not gap.present())


def _from_engine(module: str, symbol: str):
    """The engine's own symbol, or ``None`` when it does not have it."""

    try:
        loaded = importlib.import_module(module)
    except Exception:
        return None
    return getattr(loaded, symbol, None)


def _refuse(gap: EngineGap):
    raise MissingEngineSymbol(
        f"the installed engine does not carry {gap.module}.{gap.symbol}, "
        f"which stops {gap.stops}.\n"
        "This package does not reimplement it: a second instrument answering "
        "the same question is how two numbers get reported with equal "
        "confidence and one of them is wrong.\n"
        f"The remedy is an engine that carries {gap.symbol}; every other "
        "command in this package works without it.")


# ---------------------------------------------------------- the refusals

def measured_free_vram_bytes(*args, **kwargs):
    engine = _from_engine("woof.core.preflight", "measured_free_vram_bytes")
    if engine is None:
        _refuse(GAPS[0])
    return engine(*args, **kwargs)


def interpolate_to_tape(*args, **kwargs):
    engine = _from_engine("woof.verify.harness.surface_bias",
                          "interpolate_to_tape")
    if engine is None:
        _refuse(GAPS[1])
    return engine(*args, **kwargs)


# ------------------------------------------- signatures, not just symbols

#: An engine callable this package invokes with an argument the engine may
#: not accept.  A name that RESOLVES can still be a different callable, and
#: the difference is not visible until the call: the install reports success,
#: the run allocates, and the argument vector is refused several minutes in,
#: inside the physics or inside a decode.
#:
#: MEASURED 2026-09-07 with `tools/measure_engine_signatures.py` between the
#: tree this package was carved from and a published 2.7.0: eighteen callables
#: differ and eight do not accept an argument this package passes.  Seven of
#: the eight were physics, and the carve took all seven into
#: `woof.globe.core` on 2026-09-09 -- the surface layer's `vegfra`,
#: radiation's `column_size_bounding`, both cumulus constructors'
#: `column_chunk`, the YSU launcher's mixing-length flag and the two float64
#: mirrors' keywords -- so their rows are gone from this table rather than
#: standing as guards against a thing that can no longer happen.
#:
#: THE EIGHTH WAS THE SOURCES SEAM and it was translated rather than carried.
#: `woof.mapped_source.decode_mapped_source` used to be a row.  Its missing
#: `scratch_destination` keyword named WHERE the decoder stages its frame
#: stream, and a published engine places the same scratch by
#: `WOOF_COMPOSE_SCRATCH` -- two spellings of one placement, so
#: `woof.globe.mapped_source_compat` translates between them, under one
#: lock, and records in the receipt which of the two placed the directory.
#: A row that is adaptable without changing what a run does is not a refusal.
#:
#: SO THE TABLE IS EMPTY, and that is the measurement rather than an
#: omission: against a published 2.7.0 there is no call this package makes
#: that the installed engine refuses.  The mechanism stays because the
#: question does not go away -- an engine inside `>=2.8.0,<2.9` can still
#: move a signature underneath the seam -- and a row added here is refused at
#: the door by name, before a run prices a card.  `tools/measure_engine_signatures.py`
#: is what finds the next one.
@dataclass(frozen=True)
class SignatureGap:
    """One engine callable, the keywords this package hands it, and the cost."""

    module: str
    #: The module-level name.  For a class, the keywords are its constructor's.
    name: str
    keywords: tuple[str, ...]
    #: What cannot run when the engine does not accept them.
    stops: str

    def missing(self) -> tuple[str, ...] | None:
        """Keywords the installed engine's signature lacks.

        ``None`` means UNANSWERABLE on this machine rather than "none": a
        module that imports cupy at module scope cannot be read on a host
        with no CUDA runtime, and answering "no gaps" would let a CPU-only
        box report a card machine's verdict.
        """

        try:
            loaded = importlib.import_module(self.module)
        except Exception:
            return None
        target = getattr(loaded, self.name, None)
        if target is None:
            return tuple(self.keywords)
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):  # pragma: no cover - a C callable
            return None
        parameters = signature.parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            # `**kwargs` accepts anything at the call and refuses inside, which
            # this check cannot see through.  Not a gap it can report.
            return None
        return tuple(k for k in self.keywords if k not in parameters)


SIGNATURE_GAPS: tuple[SignatureGap, ...] = ()


def engine_signature_gaps() -> tuple[tuple[SignatureGap, tuple[str, ...]], ...]:
    """Every signature gap that is a gap on the INSTALLED engine.

    Rows the machine cannot answer (a module that needs a CUDA runtime on a
    host without one) are left out rather than reported as clean.
    """

    found = []
    for gap in SIGNATURE_GAPS:
        missing = gap.missing()
        if missing:
            found.append((gap, missing))
    return tuple(found)


def require_engine_signature(module: str, name: str) -> None:
    """Refuse a run the installed engine's signature cannot accept.

    THE BREAKAGE THIS PREVENTS, measured rather than imagined.  Against a
    published 2.7.0, before the carve and before the scratch translation,
    `sfclay(..., vegfra=)` raised TypeError after the run had been accepted,
    priced the card, allocated it and integrated its first steps, and
    `decode_mapped_source(..., scratch_destination=)` raised after the decode
    had opened a file, which is minutes into a fetch of several GB.  The same
    refusal, raised at the door, costs the reader seconds instead of the run
    and names the engine change that fixes it.

    `SIGNATURE_GAPS` is empty today, so this raises `KeyError` for every
    call: asking about a signature nobody measured must not answer "fine".
    The function stays because the question does not: an engine resolution
    inside `woof>=2.8.0,<2.9` can still move a signature under the seam, and
    `tools/measure_engine_signatures.py` is what finds it.  A row added to
    the table is refused here by name, and silently dropping the argument is
    never the alternative -- that is how a package integrates different
    physics than its receipt records.
    """

    for gap in SIGNATURE_GAPS:
        if gap.module != module or gap.name != name:
            continue
        missing = gap.missing()
        if not missing:
            return
        raise MissingEngineSymbol(
            f"the installed engine's {gap.module}.{gap.name} does not take "
            f"{', '.join(missing)}, which stops {gap.stops}.\n"
            "This is a signature, not a name: the symbol resolves, so an "
            "install reports success and the call fails inside the run.\n"
            f"The remedy is an engine whose {gap.name} takes "
            f"{', '.join(gap.keywords)}.")
    raise KeyError(f"{module}.{name} is not a measured signature gap")
