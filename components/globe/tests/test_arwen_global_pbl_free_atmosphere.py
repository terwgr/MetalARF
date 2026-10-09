"""The YSU free-atmosphere mixing-length mode and the probe that measured
the need for it (woof.globe.pbl_free_atmosphere, 2026-09-05).

Held here, on the float64 mirror of the kernel (the CUDA kernel's own
run of the same families is the calibration of record, taken on the
card and recorded with the lane's evidence):

* the option: ``"wrf-layer"`` (WRF's rule) is the native suite's default
  and drops out of the config identity so every checkpoint written before
  the option existed keeps its hash; ``"fixed"`` is selectable and joins
  the identity as a different trajectory; an unknown name is refused by
  name;
* the planted jet family in both directions: a column whose free
  atmosphere carries a known gradient Richardson number and shear reads
  that Ri and shear back through the probe's formula terms, the mirror's
  ``exch_m`` equals the formula's K under both modes, and the two modes
  differ by exactly the square of their Blackadar lengths;
* the zero-shear family: a stably stratified shear-free column reads
  exactly the floor (0.1 m2/s) and moves no momentum; a neutral
  shear-free layer reads the floor plus the scheme's own 1e-9 shear-floor
  leak ``rl^2 sqrt(1e-9)`` and nothing else;
* the mode is confined to the free atmosphere: on the WRF oracle fixture
  the two modes agree bit for bit at every interface inside the boundary
  layer and at every free-atmosphere interface thinner than 300 m, and
  differ only where the layer is thicker;
* the band projection: a single-degree rotational tendency is read in
  its band alone and a zero tendency as zero everywhere;
* the reader's plumbing on a synthetic capture (no card): the
  transcription check reads zero against its own formula and the
  per-level kinetic tendency closes on the bands.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs

import numpy as np
import pytest

# CPU only: the device switch is set for each test in this module by
# `conftest._cpu_only_marked_tests` and put back afterwards, because a
# module that set it at import time decided it for the whole session.
pytestmark = pytest.mark.cpu_only

from woof.globe.pbl_free_atmosphere import (  # noqa: E402
    MODES,
    PLANTED_JETS,
    STABLE_SURFACE,
    XKZMINM,
    calibrate,
    calibrate_band_projection,
    entrainment_zone,
    free_atmosphere_terms,
    read_capture,
    synthetic_column,
)
from woof.globe.physics.native_options import NativePhysicsOptions  # noqa: E402
from woof.globe.core.ysu_contract import (  # noqa: E402
    YSU_FIXED_ASYMPTOTIC_LENGTH_M,
    YSU_FREE_ATMOSPHERE_MIXING_LENGTHS,
    free_atmosphere_mixing_length_flag,
)
from woof.globe.core.npref import np_ysu_column  # noqa: E402
from tests.physics_reference.ysu_oracle import load_ysu_oracle  # noqa: E402


# RETIRED 2026-09-09.  Nine tests here skipped on a published engine,
# because the mode table and the kernel that switches on it were patch item
# 01 of the series the carve wrote for the engine's owner and no published
# engine carried them.  The carve took `ysu_contract`, `ysu.py` and `ysu.cu`
# into `woof.globe.core`, so the subject of these tests is now this
# package's own code and there is no engine that can be behind on it.  The
# skip is deleted rather than left true-by-accident: a skip that can never
# fire reads as coverage.


ACK = dict(acknowledgement="device-pending-arwen-native-physics-v1", start_time_utc="2026-09-01T00:00:00Z")


# --------------------------------------------------------------------------
# the option
# --------------------------------------------------------------------------


def test_the_native_default_is_wrf_layer_and_only_fixed_joins_the_identity():
    bare = NativePhysicsOptions.from_mapping(dict(ACK))
    assert bare.ysu_free_atmosphere_mixing_length == "wrf-layer"
    assert "ysu_free_atmosphere_mixing_length" not in bare.identity
    wrf = NativePhysicsOptions.from_mapping(dict(ACK, ysu_free_atmosphere_mixing_length="wrf-layer"))
    assert wrf.identity == bare.identity
    fixed = NativePhysicsOptions.from_mapping(dict(ACK, ysu_free_atmosphere_mixing_length="fixed"))
    assert fixed.identity["ysu_free_atmosphere_mixing_length"] == "fixed"
    # The bare identity is the identity every earlier checkpoint carries;
    # "fixed" is a different trajectory and joins.
    before = dict(fixed.identity)
    del before["ysu_free_atmosphere_mixing_length"]
    assert bare.identity == before
    with pytest.raises(ValueError, match="ysu_free_atmosphere_mixing_length must be one of"):
        NativePhysicsOptions.from_mapping(dict(ACK, ysu_free_atmosphere_mixing_length="layer"))


def test_the_flag_table_is_the_kernel_switch():
    assert YSU_FREE_ATMOSPHERE_MIXING_LENGTHS == {"wrf-layer": 0, "fixed": 1}
    assert free_atmosphere_mixing_length_flag("wrf-layer") == 0
    assert free_atmosphere_mixing_length_flag("fixed") == 1
    assert YSU_FIXED_ASYMPTOTIC_LENGTH_M == 30.0
    with pytest.raises(ValueError, match="must be one of"):
        free_atmosphere_mixing_length_flag("30m")
    with pytest.raises(ValueError, match="must be one of"):
        _mirror(synthetic_column(), "30m")


def _column_args(col):
    return (col["u"], col["v"], col["theta"], col["qv"], col["qc"], col["qi"], col["p"], col["p_interface"],
            col["exner"], col["dz"])


def _mirror(col, mode, dt=50.0):
    return np_ysu_column(*_column_args(col), dt=dt, free_atmosphere_mixing_length=mode, **STABLE_SURFACE)


# --------------------------------------------------------------------------
# the families
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def calibration():
    return calibrate("numpy")


def test_the_planted_jet_reads_its_richardson_number_shear_and_formula_under_both_modes(calibration):
    rows = calibration["families"]["planted_jet"]
    assert {(r["planted_ri"], r["planted_shear_per_s"]) for r in rows} == set(PLANTED_JETS)
    assert {r["mode"] for r in rows} == set(MODES)
    for row in rows:
        # The scheme adds 1e-9 to the squared shear before it divides, so the
        # read Ri sits below the planted one by 1e-9 / shear^2: 4e-5 at the
        # weakest planted shear.  That is the scheme's own floor, not the
        # reader's, and it is the whole error.
        floor_share = 1.0e-9 / row["planted_shear_per_s"] ** 2
        assert row["read_ri_max_rel_error"] <= floor_share * 1.0001, row
        assert row["read_shear_max_rel_error"] <= 0.5 * floor_share * 1.0001, row
        assert row["max_rel_error_kernel_vs_formula"] < 1.0e-12, row
        assert row["read_rlamdz_m"] == row["expected_rlamdz_m"]
        assert row["expected_rlamdz_m"] == (150.0 if row["mode"] == "wrf-layer" else 30.0)
        assert row["kpbl"] <= 12
    # Both directions: the two modes on the same planted column differ by
    # exactly the square of their Blackadar lengths above the floor.
    by_key = {(r["planted_ri"], r["mode"]): r for r in rows}
    for ri, _shear in PLANTED_JETS:
        wrf = np.asarray(by_key[(ri, "wrf-layer")]["kernel_exch_m_m2_s"]) - XKZMINM
        fixed = np.asarray(by_key[(ri, "fixed")]["kernel_exch_m_m2_s"]) - XKZMINM
        col = synthetic_column(dz_m=1500.0, shear_per_s=_shear, ri=ri)
        faces = np.arange(12, 20)
        zk = 0.4 * col["zq"][faces + 1]
        rl_wrf = zk * 150.0 / (150.0 + zk)
        rl_fixed = zk * 30.0 / (30.0 + zk)
        assert wrf / fixed == pytest.approx((rl_wrf / rl_fixed) ** 2, rel=1.0e-9)
        assert np.all(wrf / fixed > 20.0)
    # The measured shape of the defect: at Ri 0.25 and a 0.02 1/s shear over
    # 1500 m layers WRF's rule gives more than 100 m2/s of momentum
    # diffusivity in the free atmosphere; the fixed length gives under 6.
    assert by_key[(0.25, "wrf-layer")]["kernel_exch_m_m2_s"][0] > 100.0
    assert by_key[(0.25, "fixed")]["kernel_exch_m_m2_s"][0] < 6.0


def test_the_zero_shear_column_reads_the_floor_and_the_neutral_layer_reads_the_leak(calibration):
    rows = calibration["families"]["zero_shear"]
    stable = [r for r in rows if r["family"] == "stable"]
    neutral = [r for r in rows if r["family"] == "neutral_layer"]
    assert len(stable) == 4 and len(neutral) == 4
    for row in stable:
        assert row["reads_exactly_the_floor"], row
        assert row["kernel_exch_m_max_minus_floor"] < 1.0e-9
        assert row["shear_term_max_m2_s"] < 1.0e-9
        assert row["du_max_abs"] == 0.0
    for row in neutral:
        assert row["read_ri_max_abs"] == 0.0
        assert row["max_rel_error_kernel_vs_floor_plus_leak"] < 1.0e-12, row
        assert row["du_max_abs"] == 0.0
        # The leak: rl^2 sqrt(1e-9), 0.69 m2/s under WRF's rule at 1500 m
        # layers (seven times the floor), 0.028 under the fixed length.
        if row["dz_m"] == 1500.0:
            expected = 0.69 if row["mode"] == "wrf-layer" else 0.028
            assert row["expected_shear_floor_leak_m2_s"] == pytest.approx(expected, rel=0.02)


def test_the_mode_touches_only_free_atmosphere_interfaces_thicker_than_300_m():
    fixture = load_ysu_oracle()
    inputs = fixture.inputs
    ncase = len(fixture.cases)
    for case in range(ncase):
        col = {name: np.asarray(inputs[name][:, 0, case], dtype=np.float64) for name in
               ("u", "v", "theta", "qv", "qc", "qi", "p", "p_interface", "exner", "dz", "rthraten")}
        surface = {("psfc" if name == "psfc" else name): float(inputs[name][0, case]) for name in
                   ("psfc", "znt", "ust", "hfx", "qfx", "wspd", "br", "psim", "psih", "xland", "u10", "v10")}
        outs = {}
        for mode in MODES:
            outs[mode] = np_ysu_column(
                col["u"], col["v"], col["theta"], col["qv"], col["qc"], col["qi"], col["p"], col["p_interface"],
                col["exner"], col["dz"], rthraten=col["rthraten"], dt=fixture.dt,
                ysu_topdown_pblmix=fixture.topdown[case], free_atmosphere_mixing_length=mode, **surface,
            )
        kpbl = int(outs["wrf-layer"]["kpbl"])
        assert int(outs["fixed"]["kpbl"]) == kpbl
        assert outs["fixed"]["hpbl"] == outs["wrf-layer"]["hpbl"]
        if not np.any(outs["wrf-layer"]["exch_m"]):
            # ust = hfx = qfx = 0: the scheme short-circuits (ysu.cu:203) in both modes.
            assert not np.any(outs["fixed"]["exch_m"])
            continue
        nz = col["u"].size
        terms = free_atmosphere_terms(col["u"], col["v"], col["theta"], col["qv"], col["qc"], col["qi"],
                                      col["exner"], col["dz"], kpbl)
        zone = entrainment_zone(col["dz"], outs["wrf-layer"]["hpbl"], outs["wrf-layer"]["delta"], kpbl)
        assert terms["active"].shape == (nz - 1,) and zone.shape == (nz - 1,)
        for k in range(nz - 1):
            same = outs["fixed"]["exch_m"][k + 1] == outs["wrf-layer"]["exch_m"][k + 1]
            if not terms["active"][k]:
                assert same, (case, k, "boundary layer")
            elif terms["dza"][k] <= 300.0 and not zone[k]:
                # rlamdz = max(0.1 dz, 30) = 30 below 300 m: WRF's rule IS the fixed length there.
                assert same, (case, k, "thin layer")
        # The scheme's own inventory of the difference: the fixed mode never
        # exceeds WRF's rule, and is below it exactly where the layer is thicker.
        thick = terms["active"] & (terms["dza"] > 300.0) & ~zone & (terms["shear"] > 1.0e-3)
        if np.any(thick):
            faces = np.flatnonzero(thick) + 1
            assert np.all(outs["fixed"]["exch_m"][faces] < outs["wrf-layer"]["exch_m"][faces]), case
        assert np.all(outs["fixed"]["exch_m"] <= outs["wrf-layer"]["exch_m"] + 1.0e-12)


def test_the_band_projection_reads_one_band_and_zero_for_no_tendency():
    from woof.globe.spectral.transform import SphericalHarmonicTransform
    from woof.globe.spectral.vector import VorticityDivergenceOperator

    transform = SphericalHarmonicTransform.create(63, backend="numpy", precision="float64")
    vector = VorticityDivergenceOperator(transform)
    for degree, order, home in ((30, 5, "n021-060"), (8, 3, "n001-020"), (62, 0, "n061-063")):
        reading = calibrate_band_projection(transform, vector, degree=degree, order=order)
        assert reading["home_band"] == home
        assert reading["home_band_rel_error"] < 1.0e-12
        assert reading["max_abs_elsewhere"] < 1.0e-12 * reading["twice_kinetic_energy"]
        assert reading["zero_tendency_max_abs"] == 0.0
        assert reading["total_vs_bands_rel"] < 1.0e-12


def test_the_reader_closes_on_a_synthetic_capture():
    """No card: a T21 capture whose exch_m is the formula's own word and
    whose tendency is a uniform relaxation of the wind reads a zero
    transcription difference, the relaxation's kinetic tendency on every
    level, and bands that close on the total over the sphere."""
    from woof.globe.spectral.transform import SphericalHarmonicTransform
    from woof.globe.spectral.vector import VorticityDivergenceOperator

    transform = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
    vector = VorticityDivergenceOperator(transform)
    nlat, nlon = transform.grid.shape
    nz = 8
    rng = np.random.default_rng(5)
    lat = np.repeat(np.asarray(transform.grid.latitude_deg)[:, None], nlon, axis=1)
    lon = np.repeat(np.linspace(0.0, 360.0, nlon, endpoint=False)[None, :], nlat, axis=0)
    dz = np.full((nz, nlat, nlon), 1200.0) + 100.0 * rng.random((nz, nlat, nlon))
    zq = np.concatenate([np.zeros((1, nlat, nlon)), np.cumsum(dz, axis=0)])
    za = 0.5 * (zq[:-1] + zq[1:])
    theta = 300.0 + 0.004 * za
    p_half = 1.0e5 * np.exp(-zq / 8500.0)
    p_full = 0.5 * (p_half[:-1] + p_half[1:])
    exner = (p_full / 1.0e5) ** (287.0 / 1004.5)
    # A smooth wind with shear: zonal jet plus a low-order eddy.
    coefficients = np.zeros((nz, *transform.spectral_shape), dtype=np.complex128)
    for k in range(nz):
        coefficients[k, 3, 0] = 4.0e-5 * (k + 1)
        coefficients[k, 6, 2] = 2.0e-5 * (1.0 + 0.5j) * (k + 1)
    vort = transform.project(coefficients)
    u, v = vector.wind_from_vordiv(vort, np.zeros_like(vort))
    kpbl = np.full((nlat, nlon), 2, dtype=np.int32)
    # The capture holds the kernel's float32 inputs; the planted word is the
    # formula on those same float32 values, so the check reads exactly zero.
    u, v, theta, exner, dz = (a.astype(np.float32).astype(np.float64) for a in (u, v, theta, exner, dz))
    terms = free_atmosphere_terms(u, v, theta, np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta),
                                  exner, dz, kpbl)
    exch_m = np.zeros((nz, nlat, nlon))
    exch_m[1:] = terms["xkzm_wrf-layer"]
    exch_h = np.zeros((nz, nlat, nlon))
    exch_h[1:] = terms["xkzh_wrf-layer"]
    tau = 3600.0
    captured = {
        "inputs": {
            "u": u.astype(np.float32), "v": v.astype(np.float32), "theta": theta.astype(np.float32),
            "qv": np.zeros((nz, nlat, nlon), np.float32), "qc": np.zeros((nz, nlat, nlon), np.float32),
            "qi": np.zeros((nz, nlat, nlon), np.float32), "exner": exner.astype(np.float32),
            "p_full": p_full.astype(np.float32), "p_half": p_half.astype(np.float32),
            "dp": (p_half[:-1] - p_half[1:]).astype(np.float32), "temperature": (theta * exner).astype(np.float32),
            "latitude_deg": lat.astype(np.float32), "longitude_deg": lon.astype(np.float32),
            "dz": dz.astype(np.float32), "rthraten": np.zeros((nz, nlat, nlon), np.float32),
        },
        "outputs": {
            "exch_m": exch_m, "exch_h": exch_h, "du": -u / tau, "dv": -v / tau,
            "hpbl": np.full((nlat, nlon), 500.0), "kpbl": kpbl, "delta": np.zeros((nlat, nlon)),
            "wstar": np.zeros((nlat, nlon)),
        },
        "dt_s": 50.0, "time_s": 0.0, "mixing_length": "wrf-layer", "config_hash": "synthetic", "step": 0,
        "transform": transform, "vector": vector,
        "quadrature_weights": np.asarray(transform.grid.quadrature_weights),
        "latitude_rows_deg": np.asarray(transform.grid.latitude_deg),
    }
    reading = read_capture(captured, levels=(3, 5))
    check = reading["summary"]["transcription_check"]
    assert check["interfaces_checked"] > 0
    assert check["exch_m_max_rel_diff"] < 1.0e-12
    assert check["exch_m_rel_diff_percentiles"]["p99"] < 1.0e-12
    assert check["exch_m_interfaces_above_1e-3_rel"] == 0
    assert check["exch_m_interfaces_above_1e-2_rel"] == 0
    assert check["exch_h_max_abs_diff_m2_s"] < 1.0e-12
    assert check["interfaces_in_entrainment_zone"] == 0
    per_level = reading["per_level_kinetic_tendency_w_m2"]
    total = np.asarray(per_level["total"])
    ke = np.asarray(per_level["level_kinetic_energy_j_m2"])
    g_idx = per_level["regions"].index("global")
    # u du + v dv = -2 KE / tau on every level.
    assert total[:, g_idx] == pytest.approx(-2.0 * ke[:, g_idx] / tau, rel=1.0e-5)
    bands = reading["per_band"]
    summed = sum(np.asarray(b) for b in bands["per_band"])
    assert summed[:, g_idx] == pytest.approx(total[:, g_idx], rel=1.0e-6)
    # The planted wind lives in degrees 3 and 6: band n001-020 carries it all.
    assert bands["bands"][0] == "n001-020"
    assert np.asarray(bands["per_band"][0])[:, g_idx] == pytest.approx(total[:, g_idx], rel=1.0e-6)
    assert reading["summary"]["levels_of_interest"] == [3, 5]
    # Everything is free atmosphere above level 1: the pbl part is level 0 only.
    free = np.asarray(per_level["free_atmosphere"])
    assert free[1:, g_idx] == pytest.approx(total[1:, g_idx], rel=1.0e-9)
    assert np.asarray(per_level["boundary_layer"])[0, g_idx] == pytest.approx(total[0, g_idx], rel=1.0e-9)
    rows = reading["per_interface"]
    assert len(rows) == nz - 1
    assert rows[3]["regions"]["global"]["rlamdz_wrf-layer_mean"] == pytest.approx(
        rows[3]["regions"]["global"]["dza_mean"] * 0.1, rel=1.0e-6)
    assert rows[3]["regions"]["global"]["rlamdz_fixed_mean"] == 30.0


# --------------------------------------------------------------------------
# the door: the suite is built from the spelled rule, the hash from its identity
# --------------------------------------------------------------------------


SMOKE_CONFIG = str(_shipped_configs() / "arwen_global_level5_native_smoke.toml")
KEY = "ysu_free_atmosphere_mixing_length"


def _smoke_config_with(tmp_path, spelled: str | None):
    from pathlib import Path

    text = Path(SMOKE_CONFIG).read_text(encoding="utf-8")
    anchor = 'pbl = "ysu",'
    assert anchor in text and KEY not in text
    if spelled is not None:
        text = text.replace(anchor, f'{anchor} {KEY} = "{spelled}",')
    # The analysis path in the shipped config is relative to the tree.
    path = tmp_path / f"smoke-{spelled or 'bare'}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_door_builds_the_suite_from_the_spelled_rule_and_hashes_its_identity(tmp_path):
    """Measured 2026-09-05 on the control's f012 checkpoint: a config
    spelling "wrf-layer" restarted (the hash matched) and the kernel ran
    "fixed", because the config kept only the identity payload of the
    options, which drops the key under WRF's rule, and the suite built
    from that payload read the dropped key as its default.  The build
    payload now carries every normalized option; only the hash and the
    receipt carry the identity."""
    from woof.globe.config import load_config
    from woof.globe.physics.arwen_bridge import NativeArwenPhysicsBridge

    bare = load_config(_smoke_config_with(tmp_path, None))
    wrf = load_config(_smoke_config_with(tmp_path, "wrf-layer"))
    fixed = load_config(_smoke_config_with(tmp_path, "fixed"))
    # What the suite is built from: the spelled rule, the default when bare.
    assert bare.native_adapter_options[KEY] == "wrf-layer"
    assert wrf.native_adapter_options[KEY] == "wrf-layer"
    assert fixed.native_adapter_options[KEY] == "fixed"
    # What the hash carries: "fixed" joins, "wrf-layer" stays out, bare or spelled.
    assert KEY not in bare.config_identity["native_adapter_options"]
    assert KEY not in wrf.config_identity["native_adapter_options"]
    assert fixed.config_identity["native_adapter_options"][KEY] == "fixed"
    assert bare.config_hash == wrf.config_hash
    assert fixed.config_hash != bare.config_hash
    # The hash under WRF's rule is the hash of the identity WITHOUT the key:
    # the one every checkpoint written before the option existed carries.
    import hashlib
    import json

    identity = dict(fixed.config_identity)
    identity["native_adapter_options"] = {
        k: v for k, v in identity["native_adapter_options"].items() if k != KEY
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert bare.config_hash == hashlib.sha256(raw).hexdigest()
    # The bridge the runner builds reads the spelled rule (no device: the
    # runtime is built lazily), and its receipt identity carries the
    # trajectory's name, not the build payload.
    for cfg, expected in ((bare, "wrf-layer"), (wrf, "wrf-layer"), (fixed, "fixed")):
        bridge = NativeArwenPhysicsBridge(cfg.native_adapter_name, cfg.native_adapter_options)
        assert bridge.adapter.options.ysu_free_atmosphere_mixing_length == expected
        assert bridge.adapter_options[KEY] == expected
        assert (KEY in bridge.identity["options"]) == (expected == "fixed")
    # The New Tiedtke closure flag, the other identity-scoped option, keeps
    # the same split: built at its default, out of the Grell-Freitas hash.
    assert bare.native_adapter_options["ntiedtke_tiedtke_closure"] is False
    assert "ntiedtke_tiedtke_closure" not in bare.config_identity["native_adapter_options"]
