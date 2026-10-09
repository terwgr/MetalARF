"""RRTMGP's trace-gas climatology comes from the engine's table, not a copy.

The carried column driver used to open the RFMIP clear-sky input NetCDF
from the engine's companion at every construction; the engine stops
shipping that file at 2.8.0 and ships the 136 numbers as a derived table
with its own pinned loader.  This package carried a copy of both until it
moved onto the 2.8 engine.  These tests hold what the move must keep: the
construction reads the engine's loader and never the NetCDF, the loader
this package calls is the engine's object rather than a second copy, and
the RFMIP oracle reaches the input file only through the engine's pinned
fetch route.
"""

from __future__ import annotations

import inspect

import numpy as np


def test_the_engine_table_loads_and_names_every_carried_gas():
    from woof.core.rrtmgp import load_trace_climatology

    from woof.globe.core.rrtmgp import _RFMIP_GAS_NAMES

    climatology = load_trace_climatology()
    assert set(climatology.trace_vmr) == set(_RFMIP_GAS_NAMES)
    assert all(0.0 < value < 1.0 for value in climatology.trace_vmr.values())
    assert climatology.pressure_layer_pa.shape == (60,)
    assert climatology.ozone_vmr.shape == (60,)
    assert climatology.pressure_layer_pa.dtype == np.float64
    assert np.all(np.isfinite(climatology.ozone_vmr))


def test_the_table_is_the_one_the_bit_identical_run_read():
    """The pin the ten-step before/after run was measured on, 2026-09-26."""

    from woof.core.rfmip_upstream import TRACE_CLIMATOLOGY_SHA256

    assert TRACE_CLIMATOLOGY_SHA256 == (
        "71d7f85758fda8cf05df66100ccd0a974fed71216a1b7facad2083bc3dd3b70e")


def test_radiation_construction_reads_the_engine_loader_not_the_file():
    from woof.globe.core.rrtmgp import RRTMGPRadiation

    body = inspect.getsource(RRTMGPRadiation.__post_init__)
    assert "rfmip-clear-sky-inputs.nc" not in body
    assert "from woof.core.rrtmgp import load_trace_climatology" in body
    assert "load_trace_climatology()" in body


def test_no_second_copy_of_the_table_or_loader_ships():
    from importlib.util import find_spec

    import woof.globe

    root = woof.globe.__path__[0]
    assert find_spec("arwen_global.trace_climatology") is None
    from pathlib import Path

    assert not list(Path(root).rglob("rrtmgp-trace-gas-climatology.json"))


def test_the_oracle_reaches_the_input_file_only_through_the_pinned_fetch():
    from woof.globe.core import rrtmgp

    profiles = inspect.getsource(rrtmgp._rfmip_profiles)
    assert "fetch_rfmip(\"rfmip-clear-sky-inputs.nc\", path=inputs)" in profiles
    assert "_table(\"rfmip-clear-sky-inputs.nc\")" not in profiles
    assert "inputs" in inspect.signature(rrtmgp.rfmip_clear_sky).parameters
