"""The surface-stretched vertical grid and the receipt that states it.

Audit 2026-09-01 (task 1a / DA-4 / VTW-6): the 20-level pressure_blend
grid put the lowest full level ~378 m AGL under an 83.5 hPa bottom layer,
and every surface scheme assumes tens of metres; its 1.96 ln-ratio top
layer carried a measured +267 m model-top geopotential error under the
level-only half-layer integration (34 m mean / 64 m rms after DN-3).
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from pathlib import Path

import numpy as np
import pytest

from woof.globe.config import VERTICAL_COORDINATES, load_config
from woof.globe.runner import (
    run,
    vertical_grid_receipt,
    vertical_grid_sentence,
)
from woof.globe.vertical import (
    SURFACE_STRETCHED_PURE_PRESSURE_ABOVE_PA,
    HybridCoordinate,
    standard_atmosphere_height_m,
)

SMOKE_CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
SHIPPED_GLOBAL_CONFIGS = (
    str(_shipped_configs() / "arwen_global_gdas_t533_24h.toml"),
    str(_shipped_configs() / "arwen_global_gdas_t255_native_24h.toml"),
    str(_shipped_configs() / "arwen_global_gdas_t63_48h.toml"),
    str(_shipped_configs() / "arwen_global_t255_quickstart.toml"),
)
PRESSURE_BLEND_HASHES = {
    # Recorded before the coordinate key existed: identity-locked archives
    # written under pressure_blend must keep reading.
    str(_shipped_configs() / "arwen_global_moist_smoke.toml"):
        "7cebe340c371837416aa3aa5fd488da8cc4d630823d5e81c131098b0ea61cdc5",
    str(_shipped_configs() / "arwen_global_t21_baroclinic_ten_day.toml"):
        "c09142124cc957e8f27c03b4a5e7807f0bdec9c5265e4d139ea360fff1f96b59",
    str(_shipped_configs() / "arwen_global_hybrid_6h_reference.toml"):
        "fe1928fcade512e1903e7a382049923b82815a978ca1b77734f418cdb9cd6c92",
    # Native configs carry the adapter's normalised options in their
    # identity, and since 2026-09-01 those include cumulus="gf" and
    # gf_ishallow=1 (the Grell-Freitas component, default on): a checkpoint
    # written by the five-scheme suite is a different trajectory and must
    # not restart into the six-scheme one, so this pin moved from
    # ef224fd0c9371821dc1bc21a2514f30529b280416fbf80f6001506f69b288f1d.
    # The coordinate part of the identity is unchanged (three pins above).
    # Moved again on 2026-09-02 from
    # cbba9b6bf233b505878dbeb35e8ff3700c4ad7bcaaeda89729d955804335b41a, for
    # two reasons that landed on separate lanes: the adapter retired its
    # vegetation_category/soil_category constants (the planet Noah runs on
    # now travels in the surface state, real WPS_GEOG statics or the
    # declared synthetic planet, and a checkpoint written under the
    # constants carries no static surface fields to restart from), and the
    # options gained cumulus_column_chunk (the identity carries every
    # normalised option, radiation_column_chunk included, so a memory chunk
    # moves the hash although it moves no output bit).
    # Moved on 2026-09-05 to
    # ea6c3181319f06bfdd121ea1f871e927a95d6ef924555fea51dab59ab3c95ae2
    # when the options gained gf_updraft_only_when_downdraft_dry with a
    # default of true, and back the same day when the timing lane's grade
    # made the default false (the switch joins the identity only under
    # cumulus="gf" and only when true, so the WRF-faithful default keeps
    # the hash every earlier Grell-Freitas checkpoint carried and a
    # config that turns the switch on cannot restart into a checkpoint
    # written without it).
    # Unmoved on 2026-09-05 when the options gained
    # ysu_free_atmosphere_mixing_length (woof.globe.core.ysu_contract): its
    # default "wrf-layer" is the arithmetic every earlier checkpoint was
    # written under and stays out of the identity, bare or spelled; only
    # "fixed" (the 30 m asymptotic length above the boundary layer, a
    # different trajectory) joins it and moves the hash
    # (tests/test_arwen_global_pbl_free_atmosphere.py).
    # gf_resolved_convergence_closure (the
    # coarse-column deep arm of Grell-Freitas, a different trajectory from
    # WRF's kernel) follows the same rule: an opt-in after the mass-flux
    # lane's grade of 2026-09-05, it joins the identity only when true
    # (d7e20bc49855dcc0c0f5fe6423ff462af69776394bb9a052b4de2266c24c3443
    # is the smoke config's hash with it on), so the WRF-faithful default
    # keeps this pin.
    str(_shipped_configs() / "arwen_global_level5_native_smoke.toml"):
        "94609125d28d920ea188969a54788e26ae02538252f08e246aeefec6eea0ee2e",
}


def _first_full_level_height_m(coordinate, ps=101_325.0):
    p_half = coordinate.a_half_pa + coordinate.b_half * ps
    p_full = np.sqrt(p_half[-1] * p_half[-2])
    return float(standard_atmosphere_height_m(p_full) - standard_atmosphere_height_m(ps))


def test_first_full_level_is_tens_of_metres_and_invariants_hold():
    coordinate = HybridCoordinate.surface_stretched(40, 100.0)
    assert coordinate.nlev == 40
    z1 = _first_full_level_height_m(coordinate)
    assert 15.0 <= z1 <= 35.0, z1
    # Invariants the class enforces, checked here explicitly.
    assert np.all(coordinate.a_half_pa >= 0.0)
    assert np.all((coordinate.b_half >= 0.0) & (coordinate.b_half <= 1.0))
    assert coordinate.a_half_pa[-1] == 0.0 and coordinate.b_half[-1] == 1.0
    assert coordinate.a_half_pa[0] == 100.0 and coordinate.b_half[0] == 0.0
    for ps in (50_000.0, 80_000.0, 101_325.0, 120_000.0):
        p = coordinate.a_half_pa + coordinate.b_half * ps
        assert np.all(np.diff(p) > 0.0), ps


def test_thickness_grows_from_the_surface_then_thins_in_ln_p_toward_the_top():
    coordinate = HybridCoordinate.surface_stretched(40, 100.0)
    p = coordinate.a_half_pa + coordinate.b_half * 101_325.0
    dp = np.diff(p)
    # 5-6 hPa bottom layer, a single interior maximum, thinner at the top.
    assert 500.0 <= dp[-1] <= 600.0
    peak = int(np.argmax(dp))
    assert 0 < peak < dp.size - 1
    assert np.all(np.diff(dp[peak:]) < 0.0)   # grows toward the surface
    assert np.all(np.diff(dp[:peak]) > 0.0)   # thins toward the top
    peak_pressure = np.sqrt(p[peak] * p[peak + 1])
    assert 30_000.0 <= peak_pressure <= 60_000.0
    # Uniform in ln p above the stretched region: a top layer far thinner
    # in ln ratio than pressure_blend's 1.96 (audit VTW-6).
    ln_ratio = np.log(p[1:] / p[:-1])
    assert ln_ratio[0] < 0.5
    np.testing.assert_allclose(ln_ratio[:10], ln_ratio[0], rtol=1.0e-9)


def test_stratosphere_is_pure_pressure_with_enough_levels_in_the_sponge():
    coordinate = HybridCoordinate.surface_stretched(40, 100.0)
    p_half = coordinate.a_half_pa + coordinate.b_half * 101_325.0
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    above_50hpa = int(np.count_nonzero(p_full < 5_000.0))
    assert above_50hpa >= 4
    # The dycore sponge takes ring-mean p_full below 5000 Pa (default base).
    assert above_50hpa >= 3
    pure = coordinate.b_half == 0.0
    assert np.all(pure[p_half <= SURFACE_STRETCHED_PURE_PRESSURE_ABOVE_PA])
    assert np.all(coordinate.a_half_pa[pure] == p_half[pure])
    assert not np.any(pure[p_half > SURFACE_STRETCHED_PURE_PRESSURE_ABOVE_PA])


def test_over_a_high_plateau_the_bottom_layer_stays_thin_and_positive():
    coordinate = HybridCoordinate.surface_stretched(40, 100.0)
    for ps, low, high in ((50_000.0, 100.0, 300.0), (60_000.0, 150.0, 400.0)):
        p = coordinate.a_half_pa + coordinate.b_half * ps
        dp = np.diff(p)
        assert low <= dp[-1] <= high, (ps, dp[-1])
        assert np.all(dp > 0.0)


def test_too_few_layers_are_refused_naming_the_hydrostatic_breakage():
    with pytest.raises(ValueError, match="VTW-6"):
        HybridCoordinate.surface_stretched(6, 100.0)
    with pytest.raises(ValueError, match="pressure_blend"):
        HybridCoordinate.surface_stretched(10, 100.0)
    # And the coarsest accepted grid still carries the invariants.
    coarse = HybridCoordinate.surface_stretched(13, 100.0)
    assert 15.0 <= _first_full_level_height_m(coarse) <= 35.0


def test_describe_states_the_grid_a_run_will_use():
    coordinate = HybridCoordinate.surface_stretched(40, 100.0)
    grid = coordinate.describe()
    assert grid["nlev"] == 40 and grid["p_top_pa"] == 100.0
    assert 15.0 <= grid["first_full_level_height_m"] <= 35.0
    assert grid["full_levels_above_50hpa"] >= 4
    assert len(grid["layers"]) == 40
    bottom = grid["layers"][-1]
    assert bottom["b_half_top"] < 1.0 and bottom["thickness_pa"] == pytest.approx(
        grid["bottom_layer_thickness_pa"]
    )
    assert bottom["z_full_m_agl"] == pytest.approx(grid["first_full_level_height_m"])
    heights = [layer["z_full_m_agl"] for layer in grid["layers"]]
    assert heights == sorted(heights, reverse=True)
    # The legacy grid describes itself the same way, and says why it was
    # retired: a lowest full level hundreds of metres up.
    legacy = HybridCoordinate.pressure_blend(20, 100.0).describe()
    assert legacy["first_full_level_height_m"] > 300.0
    assert legacy["bottom_layer_thickness_pa"] > 8_000.0


def test_standard_atmosphere_heights_hit_the_icao_bases():
    assert standard_atmosphere_height_m(101_325.0) == 0.0
    assert standard_atmosphere_height_m(22_632.06) == pytest.approx(11_000.0, abs=0.01)
    assert standard_atmosphere_height_m(5_474.889) == pytest.approx(20_000.0, abs=0.05)
    assert standard_atmosphere_height_m(100.0) == pytest.approx(47_820.0, abs=5.0)


def test_shipped_global_configs_default_to_surface_stretched_40():
    for path in SHIPPED_GLOBAL_CONFIGS:
        cfg = load_config(path)
        assert cfg.vertical_coordinate == "surface_stretched", path
        assert cfg.vertical.nlev == 40, path
        assert 15.0 <= _first_full_level_height_m(cfg.vertical) <= 35.0, path


def test_pressure_blend_stays_selectable_and_hash_stable():
    from dataclasses import replace

    for path, expected in PRESSURE_BLEND_HASHES.items():
        cfg = load_config(path)
        assert cfg.vertical_coordinate == "pressure_blend", path
        assert "vertical_coordinate" not in cfg.config_identity
        # The hashes were recorded under the external-only semi-implicit
        # scheme, the only one that existed; an archive of that era reads
        # back under [semi_implicit] scheme = "external", which keeps the
        # identity byte-for-byte.  The vertical-mode default is a different
        # trajectory and must not share the hash.
        archived = replace(cfg, semi_implicit_scheme="external")
        assert archived.config_hash == expected, path
        assert cfg.config_hash != expected, path


def test_the_vertical_table_defaults_and_refusals(tmp_path):
    base = Path(SMOKE_CONFIG).read_text(encoding="utf-8")
    assert 'coordinate = "pressure_blend"\nnlev = 4\n' in base

    def write(name, vertical_block):
        path = tmp_path / name
        path.write_text(
            base.replace('coordinate = "pressure_blend"\nnlev = 4\n', vertical_block),
            encoding="utf-8",
        )
        return path

    # Fixed means default: a [vertical] that names nothing gets the
    # stretched grid at 40 levels.
    bare = load_config(write("bare.toml", ""))
    assert bare.vertical_coordinate == "surface_stretched"
    assert bare.vertical.nlev == 40
    # The old default level count stays with the old coordinate.
    legacy = load_config(write("legacy.toml", 'coordinate = "pressure_blend"\n'))
    assert legacy.vertical.nlev == 6
    # nlev alone under the new default is refused by name when too coarse.
    with pytest.raises(ValueError, match="pressure_blend"):
        load_config(write("coarse.toml", "nlev = 4\n"))
    with pytest.raises(ValueError, match="vertical.coordinate must be one of"):
        load_config(write("unknown.toml", 'coordinate = "sigma"\nnlev = 4\n'))
    explicit = write(
        "explicit.toml",
        'coordinate = "pressure_blend"\na_half_pa = [100.0, 50.0, 0.0]\n'
        "b_half = [0.0, 0.5, 1.0]\n",
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        load_config(explicit)
    assert set(VERTICAL_COORDINATES) == {"surface_stretched", "pressure_blend", "jet_refined"}


def test_the_receipt_and_the_door_state_the_vertical_grid(tmp_path):
    cfg = load_config(SMOKE_CONFIG)
    sentence = vertical_grid_sentence(cfg)
    assert "vertical pressure_blend nlev=4" in sentence
    assert "first full level" in sentence
    receipt = run(cfg, tmp_path / "out")
    grid = receipt["vertical"]
    assert grid == vertical_grid_receipt(cfg)
    assert grid["coordinate"] == "pressure_blend"
    assert grid["nlev"] == 4 and len(grid["layers"]) == 4
    assert grid["first_full_level_height_m"] == pytest.approx(
        _first_full_level_height_m(cfg.vertical)
    )
