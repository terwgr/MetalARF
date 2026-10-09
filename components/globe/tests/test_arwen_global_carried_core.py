"""The carried physics, as the runtime reaches it.

`tests/test_resync_core_carve.py` holds the CUT: that the files here are the
source tree's, byte for byte where it matters, and that the tool can re-cut
them.  This file holds the RUNTIME: that the package imports them, that the
loader binds this directory rather than the engine's, that the physics runtime
asks for the carried names, and that a receipt says whose physics integrated.

WHY THE LOADER BINDING IS ITS OWN TEST.  `woof/globe/core/kernels/__init__.py`
is byte-identical to the engine's.  The only thing that makes it read this
package's kernels is `_KDIR = Path(__file__).parent`, and the failure mode is
silent in the worst way: if a carried module resolved the ENGINE's loader, it
would compile the engine's `.cu` while every Python file in this package said
otherwise, and the run would produce numbers from a scheme nobody selected.

WHAT NEEDS A CARD AND WHAT DOES NOT.  `module_source` reads and assembles the
kernel string with no CUDA at all, so the binding is provable on any host.
Compiling it is not; that is the `gpu` mark, and the card it needs is named in
the skip.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest

from woof.globe.physics import provenance

CORE = Path(importlib.import_module("arwen_global").__file__).resolve().parent / "core"

#: Carried modules that import without a CUDA runtime.  The rest reach cupy at
#: module scope and are the card's business.
IMPORTABLE_ANYWHERE = ("ysu_contract", "physics_inventory", "landuse", "noah",
                       "gf", "ntiedtke", "rrtmgp", "npref")

#: Carried modules that reach cupy at module scope, directly or through what
#: they import.  Not a defect: they drive kernels.
NEEDS_CUPY = ("sfclay", "ysu", "morrison", "physics")


@pytest.mark.parametrize("name", IMPORTABLE_ANYWHERE)
def test_the_carried_module_imports_on_a_host_with_no_card(name):
    module = importlib.import_module(f"woof.globe.core.{name}")
    assert Path(module.__file__).resolve().parent == CORE.resolve()


def test_the_carried_modules_that_need_a_card_say_so_rather_than_hiding_it():
    """The split is a measurement, not a guess: it is re-read every run."""

    for name in NEEDS_CUPY:
        try:
            importlib.import_module(f"woof.globe.core.{name}")
        except ImportError as exc:
            assert "cupy" in str(exc), (name, exc)
        # On a card host the import succeeds, which is also correct.


# ------------------------------------------------------------- the loader

def test_the_carried_loader_binds_the_carried_kernel_directory():
    from woof.globe.core import kernels

    # RESOLVED ON BOTH SIDES.  `_KDIR` is `Path(__file__).parent`, unresolved
    # by design, and a venv reached through a symlinked home directory makes
    # the two spellings of one directory compare unequal.  Measured 2026-09-10
    # on a Linux host mounted that way: this test was the only red row in an
    # otherwise green suite there, and it was the test that was wrong, not the
    # binding.
    assert kernels._KDIR.resolve() == (CORE / "kernels").resolve()
    assert kernels.module_source.__module__ == "woof.globe.core.kernels"


def test_a_kernel_name_resolves_to_this_package_s_file():
    """The assembled source is the carried `.cu`, not the engine's copy."""

    from woof.globe.core import kernels

    source = kernels.module_source("sfclay")
    own = (CORE / "kernels" / "sfclay.cu").read_text(encoding="utf-8")
    assert source.endswith(own)
    # And the preamble is the engine's CUDA_DEFINES, which is a SEAM file:
    # it stays on the engine and reaches every carried kernel's source.
    from woof.core.constants import CUDA_DEFINES

    for key in CUDA_DEFINES:
        assert f"#define {key} " in source


def test_the_engine_loader_is_a_different_object_reading_a_different_directory():
    """Two loaders in one process, and neither shadows the other."""

    engine = pytest.importorskip("woof.core.kernels")
    from woof.globe.core import kernels

    assert engine is not kernels
    assert engine._KDIR != kernels._KDIR
    # The manifest namespace is deliberately SHARED: every receipt this model
    # published keys its kernels under it, and record_module files a second
    # differing image under a suffix rather than replacing the first.
    assert engine.MODULE_KEY_ROOT == kernels.MODULE_KEY_ROOT


def test_a_kernel_the_engine_also_ships_differs_where_it_was_measured_to():
    """sfclay.cu is one of the nine translation units that differ."""

    engine = pytest.importorskip("woof.core.kernels")
    from woof.globe.core import kernels

    ours = kernels.module_source("sfclay")
    theirs = engine.module_source("sfclay")
    assert ours != theirs, (
        "the carried sfclay kernel and the installed engine's are the same "
        "bytes; either the engine caught up (retire the carry) or the carve "
        "took the engine's copy")


# ------------------------------------------------------------ the runtime

def test_the_runtime_asks_for_the_carried_modules():
    from woof.globe.physics.native_runtime import CUMULUS_SCHEME_MODULES

    assert CUMULUS_SCHEME_MODULES["gf"][0] == "woof.globe.core.gf"
    assert CUMULUS_SCHEME_MODULES["ntiedtke"][0] == "woof.globe.core.ntiedtke"
    assert CUMULUS_SCHEME_MODULES["own"][0].startswith("woof.globe.physics.")

    source = Path(
        importlib.import_module("woof.globe.physics.native_runtime").__file__
    ).read_text(encoding="utf-8")
    assert 'self._module("woof.core.' not in source
    for name in ("rrtmgp", "sfclay", "noah", "ysu", "morrison"):
        assert f'self._module("woof.globe.core.{name}")' in source


def test_the_scorecards_grade_against_the_carried_mirror():
    """A mirror that is not the kernel it mirrors is a flawed instrument."""

    for module_name in ("woof.globe.radiation_scorecard",
                        "woof.globe.pbl_free_atmosphere"):
        source = Path(
            importlib.import_module(module_name).__file__
        ).read_text(encoding="utf-8")
        assert "from tests.physics_reference.npref import" not in source
        assert "woof.globe.core.npref" in source


def test_the_carried_mirror_takes_the_keywords_the_scorecards_pass():
    import inspect

    from woof.globe.core import npref

    assert "vegfra" in inspect.signature(npref.np_sfclay).parameters
    assert ("free_atmosphere_mixing_length"
            in inspect.signature(npref.np_ysu_column).parameters)
    paths = inspect.signature(npref.np_rrtmgp_hydrometeor_paths).parameters
    for keyword in ("size_bounds", "size_treatment", "ice_merge",
                    "sentinel_fallback"):
        assert keyword in paths, keyword


# ------------------------------------------------------------- the card

#: Compiling is the one claim this file cannot make without a device.  The
#: skip names the card the claim was measured on rather than saying "no GPU",
#: so a reader of a run log knows what a green run would have meant.
needs_a_card = pytest.mark.skipif(
    importlib.util.find_spec("cupy") is None,
    reason="needs a CUDA device and cupy; measured on an RTX 5090 "
           "(sm_120, CUDA 13 driver) on 2026-09-10, where the carried core "
           "compiled and integrated three T255 L40 steps")


@pytest.mark.gpu
@needs_a_card
def test_a_carried_kernel_compiles_through_the_carried_loader():
    """The binding, taken all the way to a compiled image on the device."""

    from woof.globe.core import kernels

    module = kernels.load_module("sfclay")
    assert module.get_function("sfclay_column") is not None

    # The manifest namespace is the ENGINE's, deliberately: every receipt this
    # model published keys its kernels under it, and record_module files a
    # second, differing image under a suffix rather than replacing the first.
    from woof.certify.kernel_manifest import kernel_manifest

    entries = kernel_manifest()
    keys = [key for key in entries
            if key.startswith("woof.core.kernels:sfclay")]
    assert keys, sorted(entries)[:10]
    recorded = entries[keys[0]]
    assert recorded["source_sha256"] ==         __import__("hashlib").sha256(
            kernels.module_source("sfclay").encode("utf-8")).hexdigest()


# ---------------------------------------------------------- the provenance

def test_a_resolved_module_is_recorded_with_its_origin_and_its_hash():
    provenance.reset()
    module = importlib.import_module("woof.globe.core.npref")
    provenance.note(module)
    recorded = provenance.integrated_physics_modules()
    row = recorded["woof.globe.core.npref"]
    assert row["origin"] == "package"
    digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    assert row["sha256"] == digest


def test_the_loader_row_carries_a_digest_of_the_device_sources():
    """A Python file's hash says nothing about the kernels it binds."""

    provenance.reset()
    from woof.globe.core import kernels

    provenance.note(kernels)
    row = provenance.integrated_physics_modules()["woof.globe.core.kernels"]
    assert row["kernels"] == "15"
    assert len(row["kernels_sha256"]) == 64
    assert row["kernels_sha256"] != row["sha256"]


def test_a_receipt_records_whose_physics_integrated():
    from woof.globe.receipt import finalize_receipt

    provenance.reset()
    provenance.note(importlib.import_module("woof.globe.core.gf"))
    payload = finalize_receipt({"run": "a test"})
    assert payload["physics_modules"]["woof.globe.core.gf"]["origin"] == "package"


def test_a_command_that_ran_no_physics_claims_none():
    """An empty table, never a row about code that did not execute."""

    from woof.globe.receipt import finalize_receipt

    provenance.reset()
    payload = finalize_receipt({"run": "a test"})
    assert payload["physics_modules"] == {}


def test_a_receipt_records_the_seam_beside_the_module_hashes():
    """The module hashes cannot see the half of the physics that stays.

    `woof/core/constants.py` supplies CUDA_DEFINES to every carried kernel's
    preamble, so a run against a 2.7.x that moved it writes a receipt whose
    module hashes are identical while every compiled kernel's assembled source
    has moved.  The seam verdict is what makes that visible in the artefact a
    reader still has months later.
    """

    from woof.globe.receipt import finalize_receipt

    provenance.reset()
    payload = finalize_receipt({"run": "a test"})
    seam = payload["engine_seam"]
    assert "unreadable" not in seam, seam
    assert seam["files"] >= 40
    assert seam["proven"] + len(seam["unproven"]) == seam["files"]
    assert seam["pinned_against"]
    assert seam["engine_resolved"]


def test_the_seam_verdict_never_costs_a_run_its_receipt(monkeypatch):
    """An unreadable seam is RECORDED as unreadable, not raised.

    A receipt that failed to write because a manifest could not be read would
    lose the run it was recording, which is a worse outcome than a receipt
    that says it could not check.
    """

    from woof.globe import receipt as receipt_module
    from woof.globe import engine_seam

    def explode() -> dict:
        raise OSError("the manifest is not readable here")

    monkeypatch.setattr(engine_seam, "load_manifest", explode)
    provenance.reset()
    payload = receipt_module.finalize_receipt({"run": "a test"})
    assert "unreadable" in payload["engine_seam"]
    assert payload["self_sha256"]
