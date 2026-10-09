"""The Grell-Freitas mass-flux lane, the kernel side, on the device.

The kernel-side calibration on the oracle fixture's 216 columns, under the
WRF-faithful kernel and the coarse-column one (gf.cu
GF_RESOLVED_CONVERGENCE_CLOSURE): a planted vertical velocity requests the
moisture convergence's own flux and the kernel's mconv is the host integral
of -g rho w dq over the forced column's cloud levels; a column over the
heating cap reads capped by the cap's own factor with its applied profile
sitting ON the cap; the applied flux is the request times that factor; and
every column the coarse-column arm does not touch (no downdraft exit
overridden, no massless level, no cap that was binding under WRF) is
bitwise the WRF kernel's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof.globe import massflux_diagnostic as md  # noqa: E402
from woof.globe.core import gf  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = _ROOT / "tools" / "gf_wrf461_oracle"
for _p in (str(_ROOT), str(_TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)



def _launch(module, fixture, k22_wrf_faithful=1, omeg_scale=None, qv_scale=None):
    from gf_field_lists import DRV_IN_LEV, DRV_IN_SCA, DRV_ISCA_FIELDS, DRV_LEV_FIELDS, DRV_SCA_FIELDS
    from tests.physics_reference.gf_oracle import GF_NZ

    gl, gs, n = fixture.levels, fixture.surface, fixture.ncol
    lvin = np.zeros((n, len(DRV_IN_LEV), GF_NZ), dtype=np.float32)
    scin = np.zeros((n, len(DRV_IN_SCA)), dtype=np.float32)
    iin = np.zeros((n, 3), dtype=np.int32)
    for j, name in enumerate(DRV_IN_LEV):
        lvin[:, j, :] = gl[name]
    if omeg_scale is not None:
        lvin[:, DRV_IN_LEV.index("w"), :] *= np.float32(omeg_scale)
    if qv_scale is not None:
        # a planted humidity: the vapor lane scaled toward saturation, so the
        # cloud layer's mean relative humidity crosses the floor's 0.9
        lvin[:, DRV_IN_LEV.index("qv"), :] *= np.float32(qv_scale)
    for j, name in enumerate(DRV_IN_SCA):
        scin[:, j] = gs[name].astype(np.float32)
    iin[:, 0] = gs["kpbl"].astype(np.int32)
    iin[:, 1] = gs["ishallow"].astype(np.int32)
    iin[:, 2] = gs["ichoice"].astype(np.int32)
    d = [cp.asarray(np.ascontiguousarray(a)) for a in (lvin, scin, iin)]
    d_lev = cp.zeros((n, len(DRV_LEV_FIELDS), GF_NZ), dtype=cp.float32)
    d_sca = cp.zeros((n, len(DRV_SCA_FIELDS)), dtype=cp.float32)
    d_isc = cp.zeros((n, len(DRV_ISCA_FIELDS)), dtype=cp.int32)
    fn = module.get_function("gf_gfdrv_stage")
    d_ws = cp.empty(gf.gf_workspace_floats(GF_NZ, n), dtype=cp.float32)
    fn(((n + 63) // 64,), (64,), (*d, d_lev, d_sca, d_isc, d_ws, np.int32(k22_wrf_faithful), np.int32(n), np.int32(GF_NZ)))
    cp.cuda.Stream.null.synchronize()
    out = dict(lev=cp.asnumpy(d_lev), sca=cp.asnumpy(d_sca), isc=cp.asnumpy(d_isc), lvin=lvin)
    out["s"] = {name: out["sca"][:, j] for j, name in enumerate(DRV_SCA_FIELDS)}
    out["i"] = {name: out["isc"][:, j] for j, name in enumerate(DRV_ISCA_FIELDS)}
    out["l"] = {name: out["lev"][:, j, :] for j, name in enumerate(DRV_LEV_FIELDS)}
    return out


@pytest.fixture(scope="module")
def fixture():
    pytest.importorskip("cupy")
    from tests.physics_reference.gf_oracle import load_gf_oracle

    return load_gf_oracle()


@pytest.fixture(scope="module")
def wrf(fixture):
    return _launch(gf._gf_module(40), fixture)


@pytest.fixture(scope="module")
def coarse(fixture):
    return _launch(gf._gf_module(40, resolved_convergence_closure=True), fixture)


@pytest.mark.gpu
def test_the_reading_is_internally_consistent_under_both_kernels(wrf, coarse):
    for run in (wrf, coarse):
        s, i = run["s"], run["i"]
        ran = (i["ierr_deep"] == 0) & (s["cuten"] > 0)
        assert np.any(ran)
        # applied = the flux the tendencies carry times neg_check's factor;
        # the request is above it only through the 100 kg/m2/s ceiling, and
        # the resolved-convergence floor (0 under WRF's kernel) is the one
        # thing that lifts the applied flux above the request
        assert np.all(s["xmb_applied"][ran] > 0)
        assert np.all(s["xmb_applied"][ran] <= np.maximum(s["xmb_request"][ran], s["xmb_floor"][ran]) * (1 + 1e-6))
        assert np.all(s["neg_check_factor"][ran] > 0) and np.all(s["neg_check_factor"][ran] <= 1.0)
        assert np.all(s["heating_cap_k_day"][ran] >= md.WRF_HEATING_CAP_K_DAY - 1e-3)
        assert np.all(s["xmb_applied"][~ran] == 0)
        assert np.all(i["closure_family"][ran] >= 1) and np.all(i["closure_family"][ran] <= 4)
        assert np.all(i["closure_family"][i["ierr_deep"] != 0] == 0)
        # the family with the largest request is the one named
        fam = np.stack([s["xf_quasi_equilibrium"], s["xf_omega"], s["xf_moisture_convergence"], s["xf_ecmwf"]], axis=1)
        assert np.array_equal(np.argmax(fam[ran], axis=1) + 1, i["closure_family"][ran])
        # the request is clos_wei * sig * max(0, the sixteen-member mean
        # less the diurnal-cycle term), the sixteen being the four families
        # spelled: quasi-equilibrium in members 1, 2, 3 and 16; cloud-base
        # omega in 4, 5, 6 and, times betajb = 1.5 once more, 14; Kuo
        # moisture convergence in 7, 8, 9 and, under a 1e-3 instead of a
        # 1e-5 floor on the precipitation efficiency, 15; ECMWF in 10 to 13
        # (module_cu_gf_deep.F cup_forcing_ens_3d; every fixture column
        # runs the full ensemble, ichoice 0)
        scale = s["sig"][ran] * (16.0 / np.maximum(1.0, s["closure_n"][ran]))
        qe, om, mc, ec = (fam[ran][:, m].astype(np.float64) for m in range(4))
        pr7 = s["pr_ens7"][ran].astype(np.float64)
        mc15 = np.where(pr7 > 0, s["mconv"][ran] / np.where(s["mconv_den"][ran] > 0, s["mconv_den"][ran], 1.0) / np.maximum(1e-3, pr7), 0.0)
        mc15 = np.maximum(0.0, mc15)
        mean16 = (4.0 * qe + 4.5 * om + 3.0 * mc + mc15 + 4.0 * ec) / 16.0
        expected = scale * np.maximum(0.0, mean16 - s["xf_dicycle"][ran])
        assert np.allclose(s["xmb_request"][ran], expected, rtol=2e-5, atol=1e-9)


@pytest.mark.gpu
def test_a_capped_column_reads_capped_by_the_caps_own_factor_and_sits_on_the_cap(fixture, wrf):
    # Both directions: the oracle fixture as recorded carries no column
    # over the 300.01 K/day cap (nothing reads capped, every factor is 1),
    # and the same columns under an eightfold planted vertical velocity
    # carry several (the request grows with the resolved forcing until the
    # cap binds).
    base_ran = (wrf["i"]["ierr_deep"] == 0) & (wrf["s"]["cuten"] > 0)
    assert np.any(base_ran)
    assert not np.any(base_ran & (wrf["s"]["neg_check_factor"] < md.CAPPED_BELOW))
    assert np.all(wrf["s"]["neg_check_factor"][base_ran] == 1.0)
    assert np.allclose(wrf["s"]["xmb_applied"][base_ran], wrf["s"]["xmb_request"][base_ran], rtol=1e-6)
    planted = _launch(gf._gf_module(40), fixture, omeg_scale=8.0)
    s, i, lev = planted["s"], planted["i"], planted["l"]
    ran = (i["ierr_deep"] == 0) & (s["cuten"] > 0)
    # the deep arm's own post-neg_check temperature tendency (the seam
    # field outt; rthcuten would fold the shallow arm in), K/day
    peak = np.max(lev["outt"], axis=1) * 86400.0
    trough = np.min(lev["outt"], axis=1) * 86400.0
    cap = s["heating_cap_k_day"]
    capped = ran & (s["neg_check_factor"] < md.CAPPED_BELOW)
    free = ran & (s["neg_check_factor"] >= md.CAPPED_BELOW)
    assert np.sum(capped) >= 4, "the planted forcing capped fewer than four columns: the cap check measured nothing"
    assert np.any(free)
    # capped: the applied profile sits ON the cap, on its heating limb (the
    # peak equals the cap) or its cooling limb (the trough equals -cap/2)
    on_heating = np.isclose(peak[capped], cap[capped], rtol=2e-4)
    on_cooling = np.isclose(trough[capped], -0.5 * cap[capped], rtol=2e-4)
    assert np.all(on_heating | on_cooling), (peak[capped], trough[capped], cap[capped])
    # free: the applied peak heating is under the cap and the trough above its cooling limb
    assert np.all(peak[free] <= cap[free] * (1 + 1e-4))
    assert np.all(trough[free] >= -0.5 * cap[free] * (1 + 1e-4))
    # applied over request is the factor exactly (float32 product)
    assert np.allclose(s["xmb_applied"][capped] / s["xmb_request"][capped], s["neg_check_factor"][capped], rtol=1e-5)
    # the direction: the raw profile (applied / factor) is over the cap on one limb
    raw_peak = peak[capped] / s["neg_check_factor"][capped]
    raw_trough = trough[capped] / s["neg_check_factor"][capped]
    assert np.all((raw_peak > cap[capped]) | (raw_trough < -0.5 * cap[capped]))


@pytest.mark.gpu
def test_a_planted_vertical_velocity_requests_the_moisture_convergences_own_flux(fixture):
    """Scaling w scales omeg and so mconv; the Kuo member is mconv / (den
    pr_ens7), and the reading's mconv reproduces the host integral of
    -g rho w dq over the cloud levels of the forced column."""
    base = _launch(gf._gf_module(40), fixture)
    twice = _launch(gf._gf_module(40), fixture, omeg_scale=2.0)
    s0, s2 = base["s"], twice["s"]
    i0, i2 = base["i"], twice["i"]
    both = (i0["ierr_deep"] == 0) & (i2["ierr_deep"] == 0) & (s0["mconv"] > 0) & (s2["mconv"] > 0)
    assert np.sum(both) >= 5
    # the Kuo member is exactly mconv / den / max(floor, pr7) on every running column
    for s in (s0, s2):
        ran = s["mconv_den"] > 0
        want = s["mconv"][ran] / s["mconv_den"][ran] / np.maximum(1e-5, s["pr_ens7"][ran])
        got = s["xf_moisture_convergence"][ran]
        assert np.allclose(got, np.maximum(0.0, want), rtol=1e-5, atol=1e-9)
    # host integral of the kernel's mconv2 on the forced column
    from gf_field_lists import DRV_IN_LEV, DRV_IN_SCA

    lv = base["lvin"]
    dt = fixture.surface[DRV_IN_SCA[4]].astype(np.float64)
    q = lv[:, DRV_IN_LEV.index("qv"), :].astype(np.float64)
    q = np.where(q < 1e-8, 1e-8, q)
    qo = q + (lv[:, DRV_IN_LEV.index("rqvften"), :] + lv[:, DRV_IN_LEV.index("rqvblten"), :]).astype(np.float64) * dt[:, None]
    qo = np.where(qo < 1e-8, 1e-8, qo)
    omeg = -9.81 * lv[:, DRV_IN_LEV.index("rho"), :].astype(np.float64) * lv[:, DRV_IN_LEV.index("w"), :].astype(np.float64)
    ktop = i0["ktop"]
    ok = 0
    for col in np.flatnonzero(both):
        kt = int(ktop[col])
        if kt < 2:
            continue
        qo_cup = np.empty(kt + 2)
        qo_cup[0] = qo[col, 0]
        for k in range(1, kt + 2):
            qo_cup[k] = 0.5 * (qo[col, k - 1] + qo[col, k])
        want = sum(omeg[col, k - 1] * (qo_cup[k] - qo_cup[k - 1]) / 9.81 for k in range(1, kt + 1))
        assert abs(want - s0["mconv"][col]) <= 2e-3 * abs(want) + 1e-8, (col, want, s0["mconv"][col])
        assert abs(2.0 * want - s2["mconv"][col]) <= 2e-3 * abs(2 * want) + 1e-8
        ok += 1
    assert ok >= 5


@pytest.mark.gpu
def test_the_coarse_column_arm_leaves_every_untouched_column_bitwise_and_names_the_rest(wrf, coarse):
    raised = coarse["s"]["heating_cap_k_day"] > md.WRF_HEATING_CAP_K_DAY + 1e-3
    # the floor binds where it exceeds the ensemble's request (the request
    # bounds the mass flux the ceiling leaves, so a floor under it is inert)
    floored = coarse["s"]["xmb_floor"] > coarse["s"]["xmb_request"] * (1 - 1e-6)
    # a raised cap moves a word only where WRF's cap was binding: the
    # thresholds only loosen, so a column WRF scaled by 1 is scaled by 1
    touched = (
        (coarse["i"]["downdraft_dry_exit"] != 0)
        | (coarse["i"]["downdraft_massless_levels"] != 0)
        | (raised & (wrf["s"]["neg_check_factor"] < 1.0))
        | floored
    )
    same = ~touched
    assert np.any(same)
    # the fixture as recorded carries no cloud layer over the floor's 0.9
    # critical humidity: the floor is absent everywhere and WRF's kernel
    # never reports one
    assert np.all(coarse["s"]["xmb_floor"] == 0) and np.all(coarse["s"]["floor_fraction"] == 0)
    assert np.all(wrf["s"]["xmb_floor"] == 0) and np.all(wrf["s"]["floor_fraction"] == 0)
    for name, a in wrf["l"].items():
        b = coarse["l"][name]
        assert np.array_equal(a[same].view(np.uint32), b[same].view(np.uint32)), name
    for name, a in wrf["s"].items():
        if name in ("heating_cap_k_day", "xmb_floor", "floor_fraction"):
            # readings the coarse kernel fills wherever the Kuo member
            # exists, binding or not; WRF's are zero (asserted below)
            continue
        b = coarse["s"][name]
        assert np.array_equal(a[same].view(np.uint32), b[same].view(np.uint32)), name
    assert np.array_equal(wrf["s"]["heating_cap_k_day"][same & ~raised], coarse["s"]["heating_cap_k_day"][same & ~raised])
    for name, a in wrf["i"].items():
        assert np.array_equal(a[same], coarse["i"][name][same]), name
    # the WRF kernel never reports a massless level or an overridden exit
    assert np.all(wrf["i"]["downdraft_massless_levels"] == 0)
    assert np.all(wrf["i"]["downdraft_dry_exit"] == 0)
    assert np.all(wrf["s"]["heating_cap_k_day"][wrf["s"]["cuten"] > 0] == np.float32(md.WRF_HEATING_CAP_K_DAY))
    # every overridden exit of the coarse kernel is one WRF rejected with 7
    # or 51, and the massless-level columns are ones WRF rejected with 51
    # or carried a NaN-free partial downdraft through
    wrf_exit = wrf["i"]["ierr_deep"]
    dry = coarse["i"]["downdraft_dry_exit"] != 0
    assert np.all(np.isin(wrf_exit[dry], (7, 51)))
    massless = coarse["i"]["downdraft_massless_levels"] != 0
    assert np.all(np.isin(wrf_exit[massless], (0, 7, 51, 18, 19, 17)))


@pytest.mark.gpu
def test_a_planted_humidity_brings_the_floor_in_by_the_partitions_own_share(fixture):
    """Both directions on the floor: the fixture as recorded has no cloud
    layer over 90 percent relative humidity and no floor; the vapor lane
    scaled toward saturation brings the Kuo-Anthes share in, larger for the
    wetter planting, and with the resolved forcing planted as well the floor
    binds, lifting the applied flux above WRF's in exactly the floored
    columns and nowhere else."""
    coarse_module, wrf_module = gf._gf_module(40, resolved_convergence_closure=True), gf._gf_module(40)
    base = _launch(coarse_module, fixture)
    assert np.all(base["s"]["floor_fraction"] == 0) and np.all(base["s"]["xmb_floor"] == 0)
    wetter, wettest = (_launch(coarse_module, fixture, omeg_scale=16.0, qv_scale=q) for q in (1.4, 1.5))
    wrf_wettest = _launch(wrf_module, fixture, omeg_scale=16.0, qv_scale=1.5)
    s_lo, s_hi, s_w = wetter["s"], wettest["s"], wrf_wettest["s"]
    carries_lo, carries_hi = s_lo["xmb_floor"] > 0, s_hi["xmb_floor"] > 0
    assert np.sum(carries_hi) >= 4, "the planted humidity brought no floor in: the floor check measured nothing"
    # the share is the partition's: in (0, 1], and higher for the wetter planting on every column that carries one under both
    both = carries_lo & carries_hi
    assert np.any(both)
    assert np.all(s_lo["floor_fraction"][carries_lo] > 0) and np.all(s_hi["floor_fraction"][carries_hi] <= 1.0 + 1e-6)
    assert np.all(s_hi["floor_fraction"][both] > s_lo["floor_fraction"][both])
    # a share of 1 - b with b = 10 (1 - RH): the wettest planting reaches the storm's end of the scale somewhere
    assert np.max(s_hi["floor_fraction"]) > 0.5
    # binding: the floor above the request lifts the applied flux above WRF's, and only there
    floored = s_hi["xmb_floor"] > s_hi["xmb_request"] * (1 - 1e-6)
    assert np.sum(floored) >= 4, "the planted forcing floored fewer than four columns"
    ran = (wettest["i"]["ierr_deep"] == 0) & (s_hi["cuten"] > 0)
    assert np.all(s_hi["xmb_applied"][floored] > s_w["xmb_applied"][floored] * (1 + 1e-6))
    untouched = ran & ~floored & (wettest["i"]["downdraft_dry_exit"] == 0) & (wettest["i"]["downdraft_massless_levels"] == 0) & (s_hi["heating_cap_k_day"] <= md.WRF_HEATING_CAP_K_DAY + 1e-3)
    assert np.allclose(s_hi["xmb_applied"][untouched], s_w["xmb_applied"][untouched], rtol=1e-6)
    assert np.all(s_w["xmb_floor"] == 0) and np.all(s_w["floor_fraction"] == 0)


@pytest.mark.gpu
def test_a_floor_percent_of_zero_compiles_and_removes_the_floor_and_nothing_else(fixture):
    """The documented no-floor setting: percent 0 compiles (it is spelt as
    the DISABLED define; the loader refuses a zero) and on the planted
    humidity and forcing that floor eight columns under the full form it
    floors none, lifts no applied flux above its request, and leaves every
    other word of the coarse kernel's reading as it was."""
    full = _launch(gf._gf_module(40, resolved_convergence_closure=True), fixture, omeg_scale=16.0, qv_scale=1.5)
    off = _launch(gf._gf_module(40, resolved_convergence_closure=True, resolved_convergence_floor_percent=0), fixture, omeg_scale=16.0, qv_scale=1.5)
    floored = full["s"]["xmb_floor"] > full["s"]["xmb_request"] * (1 - 1e-6)
    assert np.sum(floored) >= 4, "the planting floored fewer than four columns under the full form: the check measured nothing"
    assert np.all(off["s"]["xmb_floor"] == 0) and np.all(off["s"]["floor_fraction"] == 0)
    ran = (off["i"]["ierr_deep"] == 0) & (off["s"]["cuten"] > 0)
    assert np.all(off["s"]["xmb_applied"][ran] <= off["s"]["xmb_request"][ran] * (1 + 1e-6))
    # the columns the floor never bound are bitwise the full form's
    for name, a in full["l"].items():
        assert np.array_equal(a[~floored].view(np.uint32), off["l"][name][~floored].view(np.uint32)), name
    for name, a in full["s"].items():
        if name in ("xmb_floor", "floor_fraction"):
            continue
        assert np.array_equal(a[~floored].view(np.uint32), off["s"][name][~floored].view(np.uint32)), name
    for name, a in full["i"].items():
        assert np.array_equal(a[~floored], off["i"][name][~floored]), name
    # and the floored ones moved: the applied flux fell to the request
    assert np.all(off["s"]["xmb_applied"][floored] < full["s"]["xmb_applied"][floored])
