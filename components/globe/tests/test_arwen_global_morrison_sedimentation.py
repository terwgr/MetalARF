# tests/test_arwen_global_morrison_sedimentation.py
"""Morrison sedimentation reads level quantities, not sedimentation-time ones.

WRF v4.6.1 ``phys/module_mp_morr_two_moment.F`` builds two quantities once
per level inside the column loop and then spends them, unchanged, in the
sedimentation block that runs after that loop closes:

* ``ACN(K) = G*RHOW/(18.*MU(K))`` (:1438) from ``MU(K)`` (:1424), which is
  evaluated before the warm branch's small snow and graupel melt
  (:1504, :1511).  The cloud droplet fall speeds read it at :3440-3441.
* the particle-size-distribution reference density
  ``DUM = PRES(K)/(287.15*T3D(K))`` at :3405.  There is no ``T3D``
  assignment anywhere between :1511 and the tendency application at :3710,
  so the temperature this reads is the one the process section's own PSD
  reconstruction used at :1558 and :2182, not a sedimentation-time value.

Rebuilding either one at sedimentation time from the updated temperature is
wrong in opposite directions, so both are published by the process stage and
consumed unchanged.  The Stokes coefficient is worth about -0.288 percent per
kelvin, so a rebuild from the post-process temperature moved both cloud
droplet fall speeds by that much per kelvin of the step's own temperature
change, on every cloudy level.

This file asserts both ends, in the carried CUDA kernel source and on the
carried float64 mirror, and it states what the rebuild was worth rather than
only that it is gone.  The file is named for the global model because that
is the distribution whose physics this is; the same test is maintained in
the model's own source tree, where the block that names where the two
sources live differs.
"""
from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path

import numpy as np
import pytest

from woof.core import constants as c

ROOT = Path(__file__).resolve().parents[1]

# THIS REPOSITORY'S LAYOUT ONLY: `tests/test_suite_device_and_gap_marks.py`
# refuses a test that builds a path into the engine checkout this package
# was carved out of, because that directory is in no install.  The paths
# are named rather than taken from an import, because a checkout can have a
# DIFFERENT revision of the package installed beside it and then ``import``
# would answer for the wrong tree's files; the mirror fixture below holds
# what it imported to the file the source assertions read.
KERNEL_SRC = ROOT / "src" / "arwen_global" / "core" / "kernels" / "morrison.cu"
MIRROR_SRC = ROOT / "src" / "arwen_global" / "core" / "npref.py"
MIRROR_MOD = "woof.globe.core.npref"

assert KERNEL_SRC.is_file() and MIRROR_SRC.is_file(), (
    "this test reads the carried sources by repository path; run it from a "
    "checkout or an unpacked sdist of this package")

KERNEL = KERNEL_SRC.read_text(encoding="utf-8")

# WRF's own constants at the two sites, spelled here rather than imported so
# the expectation cannot move with the implementation.
WRF_RHOW = 997.0
WRF_BC = 2.0
WRF_PSD_R = 287.15


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def mirror():
    """The imported mirror, held by content to the file the source reads.

    A suite run from an unpacked sdist against that sdist's own install,
    which is how the release selection is measured on the CPU host, imports
    from site-packages while ``MIRROR_SRC`` points into the unpacked tree;
    the two files are byte for byte the same and a path comparison would
    refuse a correct run.  A different revision installed beside a checkout
    is the breakage, and the digest catches exactly that.
    """

    m = importlib.import_module(MIRROR_MOD)
    imported = Path(m.__file__)
    assert _digest(imported) == _digest(MIRROR_SRC), (
        "imported %s from %s (sha256 %s), but the source assertions read "
        "%s (sha256 %s); the two are different revisions of the same file"
        % (MIRROR_MOD, imported, _digest(imported)[:12],
           MIRROR_SRC, _digest(MIRROR_SRC)[:12]))
    return m


def _wrf_acn(temperature):
    """WRF :1424 and :1438."""
    mu = 1.496e-6 * temperature ** 1.5 / (temperature + 120.0)
    return c.G * WRF_RHOW / (18.0 * mu)


def _wrf_psd_density(pressure, temperature):
    """WRF :3405, and the same expression at :1558, :2182 and :3920."""
    return pressure / (WRF_PSD_R * temperature)


def _wrf_cloud_fall_speeds(acn, pgam, lam):
    """WRF :3440-3441 with BC = 2."""
    vn = acn * math.gamma(1.0 + WRF_BC + pgam) / (
        lam ** WRF_BC * math.gamma(pgam + 1.0))
    vm = acn * math.gamma(4.0 + WRF_BC + pgam) / (
        lam ** WRF_BC * math.gamma(pgam + 4.0))
    return vm, vn


def _moving_column():
    """One warm cloudy level whose temperature moves during the step.

    ``qs`` below 1e-6 kg/kg makes the warm branch's small-particle melt fire,
    so the pre-melt and post-melt temperatures differ as well.
    """
    q = {"qv": 0.014, "qc": 1.0e-3, "qr": 2.0e-4,
         "qi": 0.0, "qs": 5.0e-7, "qg": 0.0}
    n = {"nc": 2.0e8, "nr": 2.0e5, "ni": 0.0, "ns": 1.0e3, "ng": 0.0}
    temperature, pressure, dt = 285.0, 90000.0, 60.0
    return q, n, temperature, pressure, dt


def _thermo(mirror, q, temperature, pressure):
    ew = min(0.99 * pressure,
             float(mirror._np_morrison_polysvp(temperature, False)))
    ei = min(ew, 0.99 * pressure,
             float(mirror._np_morrison_polysvp(temperature, True)))
    qvs = c.EP2 * ew / (pressure - ew)
    qvi = c.EP2 * ei / (pressure - ei)
    xlv = 3.1484e6 - 2370.0 * temperature
    xls = 3.15e6 - 2370.0 * temperature + 0.3337e6
    cpm = c.CP * (1.0 + 0.887 * q["qv"])
    return qvs, qvi, xlv, xls, cpm


# ---------------------------------------------------------------------------
# Source contract, read out of the carried kernel.
# ---------------------------------------------------------------------------


def test_kernel_psd_density_is_built_from_the_current_temperature():
    """morr_bound spends PRES/(287.15*T3D), not the frozen entry density."""
    assert "real rho_cloud = pres / (287.15f * temp);" in KERNEL
    assert "rhoa * RD / 287.15f" not in KERNEL


def test_kernel_terminal_velocity_consumes_published_level_quantities():
    """The sedimentation stage may not rebuild either quantity."""
    start = KERNEL.index("void morr_terminal_velocity(")
    body = KERNEL[start:KERNEL.index("struct MorrRates", start)]
    assert "real rho_pgam, real acn," in body
    assert "1.496e-6f" not in body, "ACN rebuilt from a sedimentation-time T"
    assert "287.15f" not in body, "PSD density rebuilt at sedimentation time"
    assert "nn / 1.0e6f * rho_pgam" in body


def test_process_stage_publishes_both_level_quantities():
    """ACN is frozen above the melt, the PSD density below it."""
    start = KERNEL.index("void morr_process_level(")
    body = KERNEL[start:KERNEL.index("morr_sediment_nstep", start)]
    acn_at = body.index("*acn_out = G * MRHOW / (18.0f * mu);")
    melt_at = body.index("if (warm) {")
    psd_at = body.index("*pgam_rho_out = pressure / (287.15f * (*temp));")
    assert acn_at < melt_at < psd_at


# ---------------------------------------------------------------------------
# Numeric contract, float64 mirror.
# ---------------------------------------------------------------------------


def test_mirror_publishes_wrf_psd_density_and_stokes_coefficient(mirror):
    q, n, temperature, pressure, dt = _moving_column()
    qvs, qvi, xlv, xls, cpm = _thermo(mirror, q, temperature, pressure)
    xlf = xls - xlv
    # WRF's own pre-melt and post-melt temperatures for this level.
    t_premelt = temperature
    t_postmelt = temperature - q["qs"] * xlf / cpm

    (_, _, t_after, _, _, pgam_rho, acn) = mirror._np_morrison_apply_level(
        dict(q), dict(n), temperature, pressure,
        pressure / (c.RD * temperature), dt,
        qvs, qvi, xlv, xls, cpm, True)

    assert t_postmelt != t_premelt, "the melt has to fire for this column"
    assert abs(t_after - t_postmelt) > 1.0, (
        "the process section has to move the temperature for this column")

    assert acn == pytest.approx(_wrf_acn(t_premelt), rel=1e-15)
    assert pgam_rho == pytest.approx(
        _wrf_psd_density(pressure, t_postmelt), rel=1e-15)
    # The two values the implementation must not use.
    assert acn != pytest.approx(_wrf_acn(t_after), rel=1e-9)
    assert pgam_rho != pytest.approx(
        _wrf_psd_density(pressure, t_after), rel=1e-9)


def test_mirror_cloud_fall_speeds_use_the_published_stokes_coefficient(mirror):
    q, n, temperature, pressure, dt = _moving_column()
    qvs, qvi, xlv, xls, cpm = _thermo(mirror, q, temperature, pressure)
    rhoa = pressure / (c.RD * temperature)
    (qnew, nnew, t_after, _, sediment_nc,
     pgam_rho, acn) = mirror._np_morrison_apply_level(
        dict(q), dict(n), temperature, pressure, rhoa, dt,
        qvs, qvi, xlv, xls, cpm, True)

    q_short = {"c": np.array([qnew["qc"]]), "r": np.array([qnew["qr"]]),
               "i": np.array([qnew["qi"]]), "s": np.array([qnew["qs"]]),
               "g": np.array([qnew["qg"]])}
    n_short = {"c": np.array([sediment_nc]), "r": np.array([nnew["nr"]]),
               "i": np.array([nnew["ni"]]), "s": np.array([nnew["ns"]]),
               "g": np.array([nnew["ng"]])}
    rho = np.array([rhoa])
    t_col = np.array([t_after])
    pgam_rho_col = np.array([pgam_rho])
    acn_col = np.array([acn])

    vm, vn, _ = mirror._np_morrison_fall_speeds(
        "c", q_short, n_short, rho, t_col,
        pgam_rho=pgam_rho_col, acn=acn_col)

    lam, pgam, _ = mirror._np_morrison_slopes(
        q_short, n_short, rho, t_col, dens=pgam_rho_col,
        reset_cloud_number=False)
    expected_vm, expected_vn = _wrf_cloud_fall_speeds(
        acn, float(pgam[0]), float(lam["c"][0]))
    assert float(vm[0]) == pytest.approx(expected_vm, rel=1e-14)
    assert float(vn[0]) == pytest.approx(expected_vn, rel=1e-14)

    # What the sedimentation-time rebuild was worth on this column.
    stale_vm, stale_vn = _wrf_cloud_fall_speeds(
        _wrf_acn(float(t_after)), float(pgam[0]), float(lam["c"][0]))
    assert abs(stale_vm / expected_vm - 1.0) > 5.0e-3
    assert abs(stale_vn / expected_vn - 1.0) > 5.0e-3


def test_mirror_fall_speeds_refuse_to_guess_the_two_quantities(mirror):
    """The sedimentation path is handed both; it never falls back."""
    q = {name: np.zeros(1) for name in "crisg"}
    n = {name: np.zeros(1) for name in "crisg"}
    q["c"][0] = 1.0e-3
    n["c"][0] = 2.0e8
    with pytest.raises(TypeError):
        mirror._np_morrison_fall_speeds("c", q, n, np.ones(1), np.full(1, 280.0))
