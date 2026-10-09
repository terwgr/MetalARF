"""The Grell-Freitas mass-flux lane: the closure reading, its census, and the coarse-column deep arm.

CPU rows: the census calibration (planted readings both directions), the
option plumbing (the default, its identity rule, the seam's construction),
the field lists (driver slots, oracle list, kernel enum in one order) and
the kernel source's guards.  The device rows (the kernel-side calibration
on the oracle fixture) live in tests/test_arwen_global_gf_massflux_cuda.py.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import re
import sys
from pathlib import Path

import numpy as np
import pytest


from woof.globe import massflux_diagnostic as md
from woof.globe.core import gf

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = _ROOT / "tools" / "gf_wrf461_oracle"
for _p in (str(_ROOT), str(_TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------
# the census, calibrated
# --------------------------------------------------------------------------


def test_census_calibration_rows_all_meet_their_bars():
    payload = md.calibrate()
    failed = [r for r in payload["rows"] if not r["ok"]]
    assert payload["all_ok"], failed
    families = {r["family"] for r in payload["rows"]}
    assert {"capped columns", "applied over request", "applied over request, nothing capped",
            "closure family count", "exit code count", "resolved convergence mm/h",
            "request would make mm/h", "empty mask", "heavy capped columns",
            "floored columns", "floored columns with zero request", "floor over request",
            "floor over request, nothing floored with a request", "floored applied over request",
            "capped columns beside a floor block", "downdraft exit 7 overridden count",
            "downdraft exit 51 overridden count", "downdraft massless-level histogram",
            "columns with massless levels"} <= families
    planted = sorted(r["planted"] for r in payload["rows"] if r["family"] == "capped fraction of requesting")
    assert planted[0] == 0.0 and planted[-1] == 1.0 and 0.0 < planted[1] < 1.0
    planted = sorted(r["planted"] for r in payload["rows"] if r["family"] == "floored columns")
    assert planted[0] == 0 and planted[-1] > 0


@pytest.mark.parametrize("floored_fraction,multiple,capped_factor,n_zero", [(0.0, 2.0, 1.0, 0), (0.3, 2.0, 1.0, 0), (0.25, 1.5, 0.5, 0), (0.2, 3.0, 1.0, 7), (0.0, 2.0, 1.0, 5)])
def test_planted_floor_rows_read_back_exactly_including_zero_requests_and_capped_floors(floored_fraction, multiple, capped_factor, n_zero):
    """The floored row is the kernel's own bound (floor > request on a column
    the deep arm carried): a column the diurnal-cycle term silenced (request
    0) counts, and so does a floored column neg_check then capped under its
    own request.  The earlier form (request > 0 and applied > request) read
    1,750 of 3,063 floored columns at the arm's hour 12."""
    reading, weights, exact = md.synthetic_reading(capped_fraction=0.1, cap_factor=0.5, floored_fraction=floored_fraction,
                                                   floor_multiple=multiple, floored_capped_factor=capped_factor,
                                                   zero_request_floored=n_zero)
    c = md.census(reading, weights, np.ones(weights.shape, dtype=bool))
    assert c["floored"]["columns"] == exact["floored"]
    assert c["floored"]["columns_with_zero_request"] == exact["floored_zero_request"]
    assert c["floored"]["fraction_of_active_columns"] == pytest.approx(exact["floored"] / exact["active"], abs=1e-12)
    if exact["floored_with_request"]:
        assert c["floored"]["floor_over_request"]["mean"] == pytest.approx(multiple, abs=1e-12)
        assert c["floored"]["applied_over_request"]["mean"] == pytest.approx(multiple * capped_factor, abs=1e-12)
    else:
        assert c["floored"]["floor_over_request"]["mean"] is None
    # the floor block leaves the capped row alone unless it was itself capped under the request
    expected_capped = exact["capped"] + (exact["floored_with_request"] if multiple * capped_factor < 1.0 else 0)
    assert c["capped"]["columns"] == expected_capped
    assert c["deep_active_columns"] == exact["active"]
    # the WRF-faithful direction: a reading with no floor anywhere reads zero
    plain, w, _ = md.synthetic_reading(capped_fraction=0.5, cap_factor=0.3)
    assert md.census(plain, w, np.ones(w.shape, dtype=bool))["floored"]["columns"] == 0


@pytest.mark.parametrize("dry7,dry51,massless", [(0, 0, None), (3, 4, None), (0, 0, {1: 6, 3: 2, 5: 1}), (2, 0, {2: 4})])
def test_planted_downdraft_rows_read_back_exactly(dry7, dry51, massless):
    reading, weights, exact = md.synthetic_reading(dry_exit_7=dry7, dry_exit_51=dry51, massless_levels=massless)
    c = md.census(reading, weights, np.ones(weights.shape, dtype=bool))
    hist = c["downdraft"]["dry_exit_overridden_histogram"]
    assert hist.get("7", 0) == dry7 and hist.get("51", 0) == dry51
    assert hist.get("0", 0) == weights.size - dry7 - dry51
    got = {k: v for k, v in c["downdraft"]["massless_levels_histogram"].items() if k != "0"}
    assert got == {str(k): v for k, v in (massless or {}).items()}
    assert c["downdraft"]["columns_with_massless_levels"] == sum((massless or {}).values())


@pytest.mark.parametrize("capped_fraction,cap_factor", [(0.0, 1.0), (0.25, 0.5), (1.0, 0.1)])
def test_planted_capped_fraction_and_factor_read_back_exactly(capped_fraction, cap_factor):
    reading, weights, exact = md.synthetic_reading(capped_fraction=capped_fraction, cap_factor=cap_factor)
    c = md.census(reading, weights, np.ones(weights.shape, dtype=bool))
    assert c["capped"]["columns"] == exact["capped"]
    assert c["capped"]["fraction_of_requesting_columns"] == pytest.approx(exact["capped_fraction"], abs=1e-12)
    if exact["capped"]:
        assert c["capped"]["applied_over_request"]["mean"] == pytest.approx(cap_factor, abs=1e-12)
        assert c["capped"]["neg_check_factor"]["mean"] == pytest.approx(cap_factor, abs=1e-12)
    else:
        assert c["capped"]["applied_over_request"]["mean"] is None


def test_column_rows_carry_every_reading_and_the_derived_rates():
    reading, weights, exact = md.synthetic_reading(mconv_mm_h=30.0)
    rows = md.column_rows(reading, [(0, 0), (15, 31)], grid_rain_mm_h=np.full(weights.shape, 2.5))
    assert set(md.READING_NAMES) <= set(rows[0])
    assert rows[0]["resolved_moisture_convergence_mm_h"] == pytest.approx(30.0)
    assert rows[0]["request_would_make_mm_h"] == pytest.approx(rows[0]["xmb_request"] * 0.01 * 3600.0)
    assert rows[0]["applied_over_request"] == pytest.approx(1.0)
    assert rows[0]["closure_family_name"] == "moisture_convergence"
    assert rows[0]["grid_rain_mm_h"] == 2.5
    assert rows[1]["ierr_deep"] == 2 and rows[1]["applied_over_request"] is None


def test_planet_masks_cover_the_planet_once_and_conus_is_land():
    lat, lon = md.gaussian_grid_coordinates(32, 64)
    land = np.zeros((32, 64), dtype=bool)
    land[20:26, 10:20] = True
    masks = md.planet_masks(lat, lon, land)
    assert masks["planet"].all()
    assert np.array_equal(masks["land"] | masks["ocean"], masks["planet"])
    assert not np.any(masks["land"] & masks["ocean"])
    assert not np.any(masks["conus_land"] & ~land)
    w = md.area_weights(lat, 64)
    assert w.shape == (32, 64) and np.all(w > 0) and w[0, 0] < w[16, 0]


# --------------------------------------------------------------------------
# the option and the seam
# --------------------------------------------------------------------------


def _options():
    return {
        "acknowledgement": "device-pending-arwen-native-physics-v1",
        "start_time_utc": "2024-05-21T00:00:00Z",
        "radiation": "rrtmgp", "radiation_interval_s": 10.0,
        "radiation_column_chunk": 512, "radiation_validation_mode": "fused",
        "surface_layer": "sfclay", "sfclay_option": 1,
        "land_surface": "noah", "land_surface_interval_s": 10.0,
        "pbl": "ysu", "ysu_topdown_pblmix": 1,
        "microphysics": "morrison", "morr_rimed_ice": 1, "dx_m": 50_000.0,
    }


def test_the_coarse_column_closure_is_an_opt_in_and_joins_the_identity_only_when_on():
    from woof.globe.physics.native_options import NativePhysicsOptions

    # off by default (the lane's grade of 2026-09-05: the storm cells go
    # but the northern 250 km energy ratio and the grid-scale rain move
    # past their bars), and the WRF-faithful state keeps the identity every
    # earlier Grell-Freitas checkpoint carried
    faithful = NativePhysicsOptions.from_mapping(_options())
    assert faithful.gf_resolved_convergence_closure is False
    assert "gf_resolved_convergence_closure" not in faithful.identity
    coarse = NativePhysicsOptions.from_mapping({**_options(), "gf_resolved_convergence_closure": True})
    assert coarse.gf_resolved_convergence_closure is True
    assert coarse.identity["gf_resolved_convergence_closure"] is True
    assert {k: v for k, v in coarse.identity.items() if k != "gf_resolved_convergence_closure"} == faithful.identity
    for scheme in ("ntiedtke", "own", "none"):
        other = NativePhysicsOptions.from_mapping({**_options(), "cumulus": scheme})
        assert "gf_resolved_convergence_closure" not in other.identity, scheme
    with pytest.raises(ValueError, match="gf_resolved_convergence_closure must be true or false"):
        NativePhysicsOptions.from_mapping({**_options(), "gf_resolved_convergence_closure": 1})


def test_the_config_hash_separates_the_two_kernels_and_the_wrf_state_keeps_the_old_hash(tmp_path):
    from woof.globe.config import load_config

    smoke = Path(str(_shipped_configs() / "arwen_global_level5_native_smoke.toml"))
    text = smoke.read_text(encoding="utf-8")
    on_path = tmp_path / "on.toml"
    on_path.write_text(text.replace(
        "native_adapter_options = { ",
        "native_adapter_options = { gf_resolved_convergence_closure = true, ",
    ), encoding="utf-8")
    wrf = load_config(smoke)
    on = load_config(on_path)
    # the config carries only the options the operator wrote: the default is absent, and False
    assert wrf.native_adapter_options.get("gf_resolved_convergence_closure", False) is False
    assert on.native_adapter_options["gf_resolved_convergence_closure"] is True
    assert "gf_resolved_convergence_closure" not in wrf.config_identity["native_adapter_options"]
    assert on.config_hash != wrf.config_hash
    # The WRF-faithful default is the hash the timing lane's grade left the
    # smoke config with (its pin before this option existed); the coarse
    # arm's hash is the one the lane's graded arms carried.
    from dataclasses import replace

    assert replace(wrf, semi_implicit_scheme="external").config_hash == (
        "94609125d28d920ea188969a54788e26ae02538252f08e246aeefec6eea0ee2e"
    )
    assert replace(on, semi_implicit_scheme="external").config_hash == (
        "d7e20bc49855dcc0c0f5fe6423ff462af69776394bb9a052b4de2266c24c3443"
    )


def test_the_seam_takes_the_switch_and_the_field_lists_agree_with_the_kernel():
    from gf_field_lists import DRV_ISCA_FIELDS, DRV_SCA_FIELDS

    seam = gf.GrellFreitas(resolved_convergence_closure=True)
    assert seam.resolved_convergence_closure is True
    assert gf.GrellFreitas().resolved_convergence_closure is False
    assert tuple(DRV_SCA_FIELDS) == gf._OUT_SCA
    assert tuple(DRV_ISCA_FIELDS) == gf._OUT_ISCA
    assert gf.CLOSURE_READING_SCALARS[0] == "xmb_request"
    assert gf.CLOSURE_READING_INTEGERS == ("ierr_deep", "downdraft_dry_exit", "closure_family", "downdraft_massless_levels")
    assert tuple(md.READING_NAMES) == gf.CLOSURE_READING_SCALARS + gf.CLOSURE_READING_INTEGERS[2:] + gf.CLOSURE_READING_INTEGERS[:2]
    assert gf.GrellFreitas(resolved_convergence_closure=True).resolved_convergence_floor_percent == 100
    with pytest.raises(ValueError, match="never binds"):
        gf.GrellFreitas(resolved_convergence_floor_percent=-1)
    # the defines the seam compiles with are all positive integers (the only
    # kind the loader takes): a floor percent of 0 is the DISABLED define,
    # not a zero the loader refuses, and the assembled source carries it
    assert gf._gf_defines(40) == ()
    assert gf._gf_defines(40, resolved_convergence_closure=True) == (("GF_RESOLVED_CONVERGENCE_CLOSURE", 1),)
    assert gf._gf_defines(40, resolved_convergence_closure=True, resolved_convergence_floor_percent=50) == (
        ("GF_RESOLVED_CONVERGENCE_CLOSURE", 1), ("GF_RESOLVED_CONVERGENCE_FLOOR_PERCENT", 50))
    off = gf._gf_defines(40, resolved_convergence_closure=True, resolved_convergence_floor_percent=0)
    assert off == (("GF_RESOLVED_CONVERGENCE_CLOSURE", 1), ("GF_RESOLVED_CONVERGENCE_FLOOR_DISABLED", 1))
    assert all(isinstance(v, int) and v >= 1 for _, v in off)
    from woof.globe.core.kernels import module_source_int_defines

    assert "#define GF_RESOLVED_CONVERGENCE_FLOOR_DISABLED 1\n" in module_source_int_defines("gf", off)
    source = (Path(gf.__file__).parent / "kernels" / "gf.cu").read_text(encoding="utf-8")
    assert "&& !GF_RESOLVED_CONVERGENCE_FLOOR_DISABLED" in source
    enum = source[source.index("DS_raincv"):source.index("GF_DRV_NSCA")]
    assert re.findall(r"\bDS_([a-z0-9_]+)", enum) == list(gf._OUT_SCA)
    enum = source[source.index("DI_ktop_deep"):source.index("GF_DRV_NISCA")]
    assert re.findall(r"\bDI_([a-z0-9_]+)", enum) == list(gf._OUT_ISCA)
    # The three parts of the coarse-column arm are all guarded by the one
    # define, and the WRF exits stay in the other branch.
    assert source.count("#if GF_RESOLVED_CONVERGENCE_CLOSURE") == 4
    assert "if (xmb_floor > xmb) xmb = xmb_floor;" in source
    assert "*ierr_io = 51; break;" in source
    assert "if (!is_shallow && thresh_resolved > thresh) thresh = thresh_resolved;" in source
    assert "(GF_UPDRAFT_ONLY_WHEN_DOWNDRAFT_DRY || GF_RESOLVED_CONVERGENCE_CLOSURE)" in source


def test_the_native_runtime_hands_the_seam_both_switches():
    # The module is asked for its OWN file rather than reached through the
    # engine's package directory.  The old path walked up from `woof.globe.core.gf`
    # into `woof/arwen_global/`, which was where this model lived before it
    # became a distribution of its own; against an installed engine that
    # directory does not exist and the read failed on a path in site-packages.
    from woof.globe.physics import native_runtime

    source = Path(native_runtime.__file__).read_text(encoding="utf-8")
    assert 'extra["resolved_convergence_closure"] = (' in source
    assert "self.options.gf_resolved_convergence_closure" in source
