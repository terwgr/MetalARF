"""The polar bias tool's MSLP reduction is the harness instrument's.

tools/arwen_global_bias_vs_gfs.py inlines the reduction so it runs against
a tree whose harness predates it; these tests hold the two together on
synthetic sides, both directions, and prove the tool's row of record
never reads the 2 m diagnostic.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import engine_symbols_or_skip

# The whole of this module drives the engine's surface-bias instrument, and
# `woof/verify/harness/` is absent from a published 2.7 (patch item 04).  A
# named module skip says which symbol and which patch item; the module-scope
# ImportError it replaces was a collection error that abandoned the run.
harness = engine_symbols_or_skip(
    "woof.verify.harness.surface_bias", patch_item="04")

TREE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tool():
    os.environ.setdefault("ARWEN_TREE", str(TREE))
    spec = importlib.util.spec_from_file_location(
        "_arwen_global_bias_vs_gfs_tool", TREE / "tools" / "arwen_global_bias_vs_gfs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sides(seed=3, *, t2_plant=0.0, ps_plant=0.0):
    rng = np.random.default_rng(seed)
    lat = np.linspace(-88.0, 88.0, 23)
    lon = np.arange(0.0, 360.0, 15.0)
    lon2d, lat2d = np.meshgrid(lon, lat)
    hgt = np.clip(600.0 + 500.0 * np.cos(np.deg2rad(lat2d)) + 80.0 * rng.standard_normal(lat2d.shape), 0.0, None)
    t2 = 282.0 + 0.05 * lat2d + rng.standard_normal(lat2d.shape)
    ours = {
        "t2": t2 + t2_plant, "u10": 2.0 + 0.1 * rng.standard_normal(lat2d.shape),
        "v10": -1.0 + 0.1 * rng.standard_normal(lat2d.shape),
        "ps": 101325.0 - 11.5 * hgt + 200.0 * rng.standard_normal(lat2d.shape) + ps_plant,
        "hgt": hgt, "land": (np.abs(lat2d) < 55.0).astype(float),
        "t1": t2 - 0.25 + 0.2 * rng.standard_normal(lat2d.shape), "z1": np.full_like(hgt, 22.7),
    }
    theirs = {
        "t2": t2 + 0.5, "u10": ours["u10"] + 0.2, "v10": ours["v10"] - 0.1,
        "ps": ours["ps"] - ps_plant + 60.0 * rng.standard_normal(lat2d.shape),
        "hgt": hgt + 30.0 * rng.standard_normal(lat2d.shape), "land": ours["land"],
    }
    return lat2d, lon2d, ours, theirs


def test_the_tool_reduces_exactly_as_the_harness(tool):
    lat2d, lon2d, ours, theirs = _sides()
    assert tool.LAPSE_K_M == harness.LAPSE_K_M
    assert tool.G == harness.GRAVITY_M_S2 and tool.R == harness.GAS_CONSTANT_DRY_J_KG_K
    assert tool.TAPE_KAPPA == harness.TAPE_KAPPA and tool.TAPE_THETA_OFFSET_K == harness.TAPE_THETA_OFFSET_K
    assert tool.MSLP_REDUCTION == harness.MSLP_REDUCTION
    t_surface = tool.surface_air_temperature(ours["t1"], ours["z1"])
    np.testing.assert_array_equal(t_surface, harness.surface_air_temperature(ours["t1"], ours["z1"]))
    np.testing.assert_array_equal(
        tool.reduce_mslp(ours["ps"], ours["hgt"], t_surface), harness.reduce_mslp(ours["ps"], ours["hgt"], t_surface)
    )
    np.testing.assert_array_equal(
        tool.mslp_temperature_sensitivity(theirs["ps"], theirs["hgt"], t_surface),
        harness.mslp_temperature_sensitivity(theirs["ps"], theirs["hgt"], t_surface),
    )
    a_ours, a_theirs, a_sens = tool.reduce_sides(ours, theirs)
    b_ours, b_theirs, b_sens = harness.reduce_sides(ours, theirs)
    for key in ("mslp", "mslp_t2", "wspd"):
        np.testing.assert_array_equal(a_ours[key], b_ours[key])
        np.testing.assert_array_equal(a_theirs[key], b_theirs[key])
    for key in a_sens:
        np.testing.assert_array_equal(a_sens[key], b_sens[key])


def test_the_tool_column_readers_are_the_harness_readers(tool):
    theta = np.array([[-3.0, 1.5]])
    pressure = np.array([[97000.0, 100500.0]])
    np.testing.assert_array_equal(
        tool.lowest_level_temperature(theta, pressure), harness.lowest_level_temperature(theta, pressure)
    )
    hgt = np.array([[250.0, 0.0]])
    phi = np.stack([hgt * tool.G, (hgt + 44.0) * tool.G])
    np.testing.assert_array_equal(tool.lowest_level_height(phi, hgt), harness.lowest_level_height(phi, hgt))
    with pytest.raises(ValueError, match="not the ground"):
        tool.lowest_level_height(phi[::-1], hgt)


def test_the_tool_row_of_record_ignores_the_2m_diagnostic_and_reads_a_pressure_plant(tool):
    lat2d, lon2d, ours, theirs = _sides()
    regions = {"global_all": np.ones(lat2d.shape, dtype=bool), "land": ours["land"] > 0.5}
    base = tool.score_regions(ours, theirs, regions, lat2d)
    _, _, warmed, same = _sides(t2_plant=1.7)
    warm = tool.score_regions(warmed, same, regions, lat2d)
    for region in regions:
        assert warm[region]["mslp_hpa"] == base[region]["mslp_hpa"]
        assert warm[region]["mslp_t2_hpa"] != base[region]["mslp_t2_hpa"]
        assert warm[region]["t2_k"]["bias"] == pytest.approx(base[region]["t2_k"]["bias"] + 1.7, abs=1e-9)
    _, _, pushed, ref = _sides(ps_plant=100.0)
    push = tool.score_regions(pushed, ref, regions, lat2d)
    t_surface = tool.surface_air_temperature(pushed["t1"], pushed["z1"])
    factor = np.exp(tool.G * pushed["hgt"] / (tool.R * (t_surface + tool.LAPSE_K_M * pushed["hgt"] / 2.0)))
    expected = tool.weighted_stats(factor, np.cos(np.deg2rad(lat2d)), regions["global_all"])["bias"]
    assert push["global_all"]["mslp_hpa"]["bias"] - base["global_all"]["mslp_hpa"]["bias"] == pytest.approx(
        expected, rel=1e-9
    )
    # The rows and the sensitivities are all present for a reader of the JSON.
    entry = base["global_all"]
    assert set(entry) == {"t2_k", "wspd10_m_s", "mslp_hpa", "mslp_t2_hpa", "sensitivity"}
    assert set(entry["sensitivity"]) == {
        "mslp_hpa_per_k_of_shared_column", "mslp_t2_hpa_per_k_of_tape_t2", "terrain_mismatch_m",
    }


def test_an_empty_footprint_reads_incomplete_not_zero(tool):
    lat2d, lon2d, ours, theirs = _sides()
    empty = tool.score_regions(ours, theirs, {"nothing": np.zeros(lat2d.shape, dtype=bool)}, lat2d)["nothing"]
    assert empty["mslp_hpa"] == {"bias": None, "rmse": None, "n": 0}
    assert empty["sensitivity"]["terrain_mismatch_m"]["n"] == 0
