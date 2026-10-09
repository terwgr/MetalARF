"""Fixtures and paths for a suite that runs against an INSTALLED package.

The suite is not run from inside the package's source tree by accident and
then declared to prove something about the wheel.  `woof.globe` is imported
the way a user imports it -- out of site-packages -- and every config it reads
comes from `woof.globe.configs_dir.config_root()`, which is the directory
INSIDE the installed package.  A test that read a repository-relative config
path would pass in a checkout and prove nothing about the artefact.

What the repository still supplies is `tools/`: the driver scripts live
outside the wheel on purpose (they are instruments, not library code), and a
handful of tests drive them. Those tests reach them through `TOOLS` below.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

#: This repository.  Used for `tools/` and for the packaging declaration, and
#: for nothing else: no test may read a config or a data file from here.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The driver scripts, outside the wheel.
TOOLS = REPO_ROOT / "tools"

# The repository root goes on the path so `from tools import ...` reaches THIS
# tree's drivers.  It is inserted at the front deliberately: the engine also
# ships a `tools` package, and without the precedence a driver of this
# package's would silently resolve to the engine's directory of the same name.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def tools_dir() -> Path:
    return TOOLS


@pytest.fixture(scope="session")
def config_root() -> Path:
    """The shipped experiments, inside the installed package."""

    from woof.globe.configs_dir import config_root as root

    return root()


#: Every spelling that reaches a card, because one of them is not enough.
#: `"import cupy" in text` misses `from cupy import ...` and it misses
#: `pytest.importorskip("cupy")` entirely, and importorskip is exactly how
#: the tree this suite came from lost thirty-four device gates to a
#: CPU-only invocation: gated on importorskip, carrying no marker, and an
#: unmarked test is not excluded by `not gpu`.
_DEVICE_SPELLINGS = ("import cupy", "from cupy", 'importorskip("cupy")',
                     "importorskip('cupy')", "import cupyx", "from cupyx")


def _opens_a_device(text: str) -> bool:
    return any(spelling in text for spelling in _DEVICE_SPELLINGS)


def pytest_collection_modifyitems(config, items):
    """Auto-mark anything that opens a device, so a bare run stays CPU-only.

    A module that imports cupy at module scope cannot be collected on a
    machine without it, and a test that reaches a card should not run in a
    default `pytest` on a laptop.  Marking is done here rather than by hand on
    every test, because a mark that has to be remembered is a mark that gets
    forgotten and the run dies on a machine that never claimed to have a card.
    """

    for item in items:
        module = getattr(item, "module", None)
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        try:
            text = Path(source).read_text(encoding="utf-8")
        except OSError:  # pragma: no cover
            continue
        if _opens_a_device(text) and "gpu" not in item.keywords:
            item.add_marker(pytest.mark.gpu)
        if NO_LOCAL_GPU and item.get_closest_marker("gpu") is not None:
            item.add_marker(_SKIP_LOCAL)


# ------------------------------------------------- the engine's own gaps

def engine_symbols_or_skip(module_name: str, *names: str, patch_item: str):
    """Import `names` out of an engine module, or skip this test module.

    THE BREAKAGE THIS TURNS INTO A SENTENCE.  Three test modules import a
    symbol straight out of the engine rather than through
    `woof.globe.engine_compat`, and each of those symbols is one the
    published engine does not carry yet.  A module-scope ImportError is a
    COLLECTION ERROR: pytest abandons the whole run, so a suite that is
    otherwise entirely green reports nothing at all and the CI job's
    verdict describes the engine's release schedule rather than this
    package.  Measured on a clean venv against a 2.7.0 wheel: three
    collection errors, zero tests run, out of a suite of 106 files.

    A skip that names the missing symbol and the patch item supplying it
    carries the same information without the collateral.  It cannot become
    permanent quietly: both CI jobs run pytest with `-rs`, so each of these
    prints its reason in the run log, and the skip disappears by itself on
    the day the engine publishes the symbol.

    Returns the symbols in the order asked for, or the module itself when
    no name is given: `hasattr(package, "submodule")` is true only when
    something has already imported it, so a submodule is asked for by
    importing its own dotted name rather than by attribute.
    """

    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        pytest.skip(
            f"the installed woof does not carry {module_name}: patch item "
            f"{patch_item} of the series the carve wrote for the engine",
            allow_module_level=True)
    if not names:
        return module
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        pytest.skip(
            f"the installed woof's {module_name} does not carry "
            f"{', '.join(missing)}: patch item {patch_item} of the series "
            "the carve wrote for the engine",
            allow_module_level=True)
    return tuple(getattr(module, name) for name in names)


# --------------------------------------------------------------- the card

#: The engine's never-open-the-local-device switch, by name.
NO_LOCAL_GPU_ENV = "GPUWM_NO_LOCAL_GPU"

#: Set by an operator to forbid this process from opening a local device.
NO_LOCAL_GPU = os.environ.get(NO_LOCAL_GPU_ENV, "").strip() not in ("", "0")

#: Applied by the collection hook to anything carrying the `gpu` marker
#: while the ban is on, so the marker implies the skip rather than relying
#: on the caller passing `-m "not gpu"`.
_SKIP_LOCAL = pytest.mark.skip(
    reason="GPUWM_NO_LOCAL_GPU=1: device work belongs on a machine that "
           "was asked for it")

if NO_LOCAL_GPU:
    # RUNTIME BACKSTOP, because source inspection is not a guarantee.  The
    # detector above marks a module whose OWN source reaches a card, and a
    # test can reach one through an intermediary it cannot see: a lazy
    # import inside a test body did exactly that on the tree this suite
    # came from and ran on the local card during a mandated CPU-only run.
    # No enumeration of intermediaries closes that, so the guarantee is
    # planted where every route converges.  The CUDA runtime reads this
    # variable at initialisation and "-1" is an invalid ordinal that leaves
    # NOTHING visible, so an escaped test fails loudly at its first device
    # call instead of quietly succeeding, and subprocesses inherit it.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@pytest.fixture(autouse=True)
def _cpu_only_marked_tests(request, monkeypatch):
    """`pytest.mark.cpu_only` sets the device switch for ONE test and puts it
    back, because a module that sets it at import time decides it for the
    whole session.

    THE BREAKAGE THIS PREVENTS, measured on a Linux host (RTX 5090, Python
    3.14.4) on 2026-09-10 against published woof 2.7.0.  Fourteen modules
    here used to run ``os.environ.setdefault(NO_LOCAL_GPU_ENV, "1")`` at
    module scope.  pytest imports every COLLECTED module before it runs any
    test, so the first of those imports switched the device off for the rest
    of the process -- including a card selection those fourteen modules are
    not even part of.  `pytest -m "gpu and not slow and not network" tests`
    reported 8 failed and 4 errors, nine of them nothing but this leak: four
    collection errors and five failures in
    `tests/test_global_spectral_sampling_device.py`, which passes 6/6 when
    its file is run alone.  A red card selection nobody can read is a card
    selection nobody runs, and that is how an unmarked test rode a cut.

    The switch is still the operator's to set for a whole run: an
    environment that already carries it is left exactly as it is, and both
    CI jobs set it in the job environment.
    """

    if request.node.get_closest_marker("cpu_only") is None:
        return
    monkeypatch.setenv(NO_LOCAL_GPU_ENV, "1")


def has_gpu() -> bool:
    """True when a usable CUDA device is present.

    Under `GPUWM_NO_LOCAL_GPU` this answers False WITHOUT importing cupy.
    Importing it and calling `getDeviceCount()` is itself device contact, and
    on the tree this suite came from it used to happen on every single pytest
    invocation, including runs that had explicitly excluded the card.
    """

    if NO_LOCAL_GPU:
        return False
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


HAS_GPU = has_gpu()

#: The mark the carved tests already use by name.
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU / cupy")


# ------------------------------------------- the static-fields rows variant

def _static_fields_speaks_rows() -> bool:
    """Whether the STAGED static-fields library knows the rows grid kind.

    A Gaussian grid is described by no WPS map projection, so this model
    hands the crate a grid kind of its own -- explicit row latitudes on a
    uniform longitude ring -- and that kind (`src/projection/rows.rs`) exists
    only on the tree this package was carved from.  The library in a
    published engine bundle answers a rows spec with

        grid spec JSON: unknown variant `rows`,
        expected one of `lambert`, `mercator`, `polar`

    which is a door SKEW rather than a Python defect: the same shape the
    surface observation door had, and it rides the same patch item.

    Asked by TRYING it, never by reading a version: the library on a machine
    was built from some checkout at some time, and the question is what these
    bytes accept.
    """

    try:
        from woof.static import rust_bridge
    except Exception:
        return False
    try:
        library = rust_bridge.load()
    except Exception:
        return False
    # The spec this package's own rows grid sends, never a hand-written
    # subset: a subset the crate's GridSpec refuses for a missing field
    # (`ref_lat`, at the 2.8.0 floor) reads here as "no rows kind".
    from woof.globe.statics_rows import RowsGrid

    spec = RowsGrid([-30.0, 0.0, 30.0], 0.0, 120.0, 3)._rust_spec()
    # The call itself is NOT guarded.  Until 0.1.2 this probe called
    # `grid_new(library, spec)`, a signature the engine's bridge does not
    # have at the 2.8.0 floor (`grid_new(spec)`), with a spec missing fields
    # the crate requires, and a bare `except` turned both into "this library
    # does not know the rows kind": six statics tests skipped on the very
    # engine that carries the kind, and the rows-grid build went unexercised
    # by the suite.  Only the library's own refusal of a well-formed spec is
    # the answer this probe asks for.
    try:
        handle = rust_bridge.grid_new(spec)
    except rust_bridge.StaticBridgeError:
        return False
    rust_bridge.grid_free(handle)
    return True


STATIC_FIELDS_SPEAKS_ROWS = _static_fields_speaks_rows()

#: Skip a test that needs a static-fields library which knows the rows kind.
requires_rows_static_fields = pytest.mark.skipif(
    not STATIC_FIELDS_SPEAKS_ROWS,
    reason="the static_fields library the installed engine loads does not "
           "know the `rows` grid kind a Gaussian grid crosses the seam as; "
           "the engine's own bundled library carries it from woof 2.8.0, so "
           "a library staged from an older build is shadowing it")


# ----------------------------------- a refusal from the compat seam is a skip

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Turn `MissingEngineSymbol` into a skip that carries its own sentence.

    `woof.globe.engine_compat` raises exactly one exception type, and it
    means exactly one thing: the installed engine does not carry a symbol this
    package needs, and this package will not reimplement it because a second
    instrument answering the same question is how two numbers get reported
    with equal confidence and one of them is wrong.

    A test that reaches that refusal has not found a defect here.  Failing it
    would put a red row in the suite whose verdict describes another project's
    release schedule, and thirty such rows are how a suite stops being read at
    all.  Skipping it keeps the sentence -- the refusal's own text names the
    symbol, the command it stops and the remedy -- and the skip disappears by
    itself on the day the engine publishes the symbol.

    This is deliberately narrow.  It catches ONE exception type, raised in ONE
    module, whose only job is to say the engine is behind; every other failure
    in this suite is still a failure.
    """

    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed or call.excinfo is None:
        return
    try:
        from woof.globe.engine_compat import MissingEngineSymbol
    except Exception:  # pragma: no cover - the package is what is under test
        return
    if not call.excinfo.errisinstance(MissingEngineSymbol):
        return
    report.outcome = "skipped"
    report.longrepr = (str(item.fspath), item.location[1] or 0,
                       str(call.excinfo.value).splitlines()[0])


# ------------------------------- engine metadata and signatures, not symbols

# RETIRED with the defect it guarded (2026-09-10).  `engine_has_source_mapping`
# asked the ENGINE's authority table whether it carried a source id, so a test
# naming one of the six global mappings could skip while the engine had none.
# The six now ride inside this package and one resolver answers every spec
# engine-first, so there is nothing left to skip on.  Its only caller,
# `requires_source_mapping`, went with the carve; this definition survived a
# three-way merge because one branch edited it while another deleted it, which
# is how a dead guard outlives the thing it guarded.


# RETIRED with the defect it guarded (2026-09-09).  `engine_callable_takes`
# read an engine callable's signature so the mark retired below could skip
# a test the installed engine would have raised TypeError on.  The
# carve ended both: every scheme those marks interrogated is carried in
# this package now.

# ------------------------------ the shapes the helpers above are used in

def requires_engine_module(module_name: str, patch_item: str):
    """Skip unless the installed engine carries a module, by name.

    The four helpers above answer questions; this and its two siblings are
    the MARKS the suite applies, so a module that needs an engine symbol
    says so once at the top instead of each of its tests discovering the
    same ImportError inside its own body.  A skip carries the patch item
    that supplies the symbol, so a reader of a run log can tell an engine
    that is behind from a package that is broken without leaving the log.
    """

    import importlib

    try:
        importlib.import_module(module_name)
        present = True
    except Exception:
        present = False
    return pytest.mark.skipif(
        not present,
        reason=f"the installed woof does not carry {module_name}: patch "
               f"item {patch_item} of the series the carve wrote for the "
               "engine")


def requires_engine_symbol(module_name: str, symbol: str, patch_item: str):
    """Skip unless an engine module carries a name.

    The symbol-level member of the set: `requires_engine_module` above asks
    whether a module imports, `requires_engine_keyword` below whether a
    callable accepts an argument, and this one whether a module carries a
    name.  NO MARK IN THIS SUITE USES IT TODAY, which is not the same as
    dead: the marks that did retired with the engine gaps they cited, when
    this package took the brightness-temperature schema family and the GDAS
    fetch names into itself.  It stays because a symbol the installed engine
    lacks is one of the three shapes an engine gap has, and a suite that can
    express two of the three pushes the third into an import error at
    collection time.
    """

    import importlib

    try:
        module = importlib.import_module(module_name)
        present = hasattr(module, symbol)
    except Exception:
        present = False
    return pytest.mark.skipif(
        not present,
        reason=f"the installed woof's {module_name} does not carry "
               f"{symbol}: patch item {patch_item} of the series the carve "
               "wrote for the engine")


# RETIRED with the defect it guarded (2026-09-09).  `requires_engine_keyword`
# skipped a test when the INSTALLED engine's copy of a physics callable did
# not accept a keyword this package passes.  Its four marks were repointed at
# the CARRIED modules when the carve landed, which made them unable to skip:
# the carried callables always take the keyword.  The skip text they could no
# longer print named `woof.globe.core.gf` as "the installed woof's", a
# module the engine does not ship.  Marks and helper are gone rather than
# repointed again.  A keyword skew against a module that is still the
# ENGINE's belongs in `SIGNATURE_GAPS`, which refuses at the door by name.

# ------------------------------------------------ a door that is not staged

def _netcdf_writer_reason() -> str | None:
    """Why the Rust NetCDF writer cannot be used here, or None.

    THE BREAKAGE THIS TURNS INTO A SENTENCE.  Product tapes are written by
    `libnetcdf_writer`, which is a RELEASE ASSET: it ships in the engine's
    bridge bundle and is staged into `~/.woof/bridges` by `woof
    fetch-bridges`.  A CI runner has no bundle, and neither does a replay
    run under a clean HOME, so every test that writes a tape failed there
    with a refusal that was CORRECT.  Four red rows saying the door is not
    installed teach a reader to stop reading the suite; a skip naming the
    binary and the command that stages it says the same thing and leaves
    the suite readable.

    It is asked by TRYING to resolve, never by reading a version.
    """

    try:
        from woof.io import nc_writer_bridge
    except Exception as failure:  # pragma: no cover - engine is what it is
        return f"woof.io.nc_writer_bridge does not import: {failure}"
    try:
        reason = nc_writer_bridge.unavailable_reason()
    except Exception as failure:  # pragma: no cover
        return f"the writer bridge could not be probed: {failure}"
    return reason


NETCDF_WRITER_REASON = _netcdf_writer_reason()

#: Skip a test that writes a product tape when the writer is not staged.
requires_netcdf_writer = pytest.mark.skipif(
    NETCDF_WRITER_REASON is not None,
    reason="the Rust NetCDF writer (libnetcdf_writer, artifact "
           "`netcdf_writer` in the engine's bridge bundle) is not staged "
           "here, so no product tape can be written: run `woof "
           "fetch-bridges`.  " + (NETCDF_WRITER_REASON or ""))
