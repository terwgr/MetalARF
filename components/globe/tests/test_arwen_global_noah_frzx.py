# tests/test_arwen_global_noah_frzx.py
"""Noah LSM: the frozen-ground infiltration limit is spent on FRZX.

WRF's REDPRM builds two different quantities, and SFLX passes the second
one down (``phys/module_sf_noahlsm.F``)::

    FRZFACT = (SMCMAX/SMCREF) * (0.412/0.468)                    :2477
    FRZX    = FRZK * FRZFACT                                     :2478

SFLX hands FRZX to NOPAC (:769) and to SNOPAC (:784).  The receiving
dummy argument is merely SPELLED FRZFACT in NOPAC, SNOPAC and SMFLX; SRT
names it back to FRZX and spends it as::

    ACRT = CVFRZ * FRZX / DICE                                   :3795

so the value that has to arrive at ACRT is FRZK*FRZFACT.  A transcription
that follows the dummy's NAME instead of the argument passes FRZFACT and
runs ACRT a factor 1/FRZK too large, which for the shipped GENPARM
FRZK = 0.15 is 6.67x.  The exponential in FCR then saturates, FCR goes to
one, and frozen ground stops limiting infiltration at all.  Measured on
the float64 mirror for one wet loam column at 266 K under 5 mm of rain:
FCR 0.9653 against the correct 0.0837, an infiltration limit 11.5 times
too permissive.

That substitution was made twice, in the CUDA kernel and in the float64
mirror, which is why they agreed with each other and with nothing else.
So this file asserts both ends, and asserts them two ways.

A  SOURCE.  The kernel inlines NOPAC and SNOPAC, so its three SMFLX call
   sites sit directly in the column routine; the mirror keeps the two
   routines, so its two call sites sit in SFLX.  Both are read for the
   argument standing in the FRZX slot, and the receiving SRT is read for
   its declaration and for the ACRT line, so a fix undone at either end
   fails here.  The mirror is read with ``ast`` rather than by pattern,
   because the dummy is still spelled ``frzfact`` where WRF spells it so,
   and a text rule would have to tell a call from a signature.

B  NUMBERS.  The mirror is run on a column with ice through the whole
   soil profile and rain falling on it.  The value SRT actually receives
   is recorded, the whole SRT infiltration path is recomputed by hand
   from the recorded inputs, and the column is re-run with the pre-fix
   value forced back in, so the test states what the substitution is
   worth rather than only that it is gone.

The file is named for the global model because that is the distribution
whose physics this is.  The same file is maintained in the model's own
source tree; only the block that names where the two sources live differs
between the copies.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import math
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

# THIS REPOSITORY'S LAYOUT ONLY, and that is a rule rather than a
# simplification: `tests/test_suite_device_and_gap_marks.py` refuses a test
# that builds a path into the engine checkout this package was carved out
# of, because that directory is in no install and a read of it can only
# work on one person's disk.  The file is maintained in the model's own
# source tree as well, where it names that tree's layout instead; the
# re-cut three-way merges the two copies, so this block is the one place
# they differ on purpose.
#
# The paths are named rather than taken from an import, because a checkout
# can have a DIFFERENT revision of the package installed beside it and then
# ``import`` would answer for the wrong tree's files.  The fixtures below
# assert that what they imported is what these paths point at.
KERNEL_SRC = ROOT / "src" / "arwen_global" / "core" / "kernels" / "noah.cu"
MIRROR_SRC = ROOT / "src" / "arwen_global" / "core" / "npref.py"
MIRROR_MOD = "woof.globe.core.npref"
NOAH_MOD = "woof.globe.core.noah"

assert KERNEL_SRC.is_file() and MIRROR_SRC.is_file(), (
    "the carried Noah sources are not under %s" % ROOT)

SIGMA = 5.67e-8
DZS = (0.1, 0.3, 0.6, 1.0)
DT = 60.0
SOILTYP = 6          # loam, present in every land mask this model runs
VEGTYP = 10          # grassland
CVFRZ = 3            # module_sf_noahlsm.F SRT, file literal


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def mirror():
    """The imported mirror, held to the file the source assertions read.

    THE BREAKAGE THIS PREVENTS is a checkout with a DIFFERENT revision of
    this package installed beside it: the source assertions below would
    read one tree's line numbers while the numerical fixtures ran the
    other tree's code, and both halves would report green.

    The hold is on CONTENT and not on the path, because the path is not
    the breakage and refusing on it refuses a correct run.  A suite run
    from an unpacked sdist against that sdist's own install -- which is
    how the release selection is measured on the CPU node -- imports from
    site-packages while these paths point into the unpacked tree, and the
    two files are byte for byte the same.  A path comparison errored all
    four nodes there and said nothing about either tree.
    """

    m = importlib.import_module(MIRROR_MOD)
    imported = Path(m.__file__)
    assert _digest(imported) == _digest(MIRROR_SRC), (
        "imported %s from %s (sha256 %s), but the source assertions read "
        "%s (sha256 %s); the two are different revisions of the same file, "
        "so the line numbers below do not describe the code that ran"
        % (MIRROR_MOD, imported, _digest(imported)[:12],
           MIRROR_SRC, _digest(MIRROR_SRC)[:12]))
    return m


@pytest.fixture(scope="module")
def noah():
    return importlib.import_module(NOAH_MOD)


@pytest.fixture(scope="module")
def params(noah):
    return noah.pack_params(noah.load_tables())


# ----------------------------------------------------------------- A: source

#: The three inlined-NOPAC/SNOPAC call sites in the kernel's column
#: routine.  Every one opens with the same eight arguments, so the
#: capture is the ninth, which is WRF's FRZX slot.
_KERNEL_CALL = re.compile(
    r"noah_smflx\(smc, cmc, dt, prcp1, zsoil, swc, slope, kdt,\s*"
    r"(?P<arg>\w+), smcmax,")


def test_kernel_smflx_call_sites_pass_frzx():
    text = KERNEL_SRC.read_text(encoding="utf-8")
    args = [m.group("arg") for m in _KERNEL_CALL.finditer(text)]
    assert len(args) == 3, (
        "expected three SMFLX call sites in %s, found %d: %s"
        % (KERNEL_SRC.name, len(args), args))
    assert args == ["frzx", "frzx", "frzx"], (
        "the kernel hands SMFLX %s; WRF's SFLX hands NOPAC and SNOPAC "
        "FRZX = FRZK*FRZFACT (module_sf_noahlsm.F:2478, :769, :784)"
        % args)


def test_kernel_derives_frzx_from_frzk_and_frzfact():
    text = KERNEL_SRC.read_text(encoding="utf-8")
    assert re.search(
        r"real frzfact = \(smcmax / smcref\) \* \(0\.412f / 0\.468f\);",
        text), "the kernel no longer builds frzfact as REDPRM does"
    assert re.search(r"real frzx = frzk \* frzfact;", text), (
        "the kernel no longer builds frzx = frzk*frzfact "
        "(module_sf_noahlsm.F:2478)")


def test_kernel_srt_declares_and_spends_frzx():
    text = KERNEL_SRC.read_text(encoding="utf-8")
    assert "real slope, real kdt, real frzx," in text, (
        "noah_srt no longer declares its frozen-ground argument as frzx")
    assert re.search(r"acrt = \(real\)cvfrz \* frzx / dice;", text), (
        "noah_srt no longer spends frzx at ACRT "
        "(module_sf_noahlsm.F:3795)")


def _call_arg_in_frz_slot(module_ast, caller, callee):
    """Name passed by ``caller`` in ``callee``'s frozen-ground slot.

    The slot is found by NAME in the callee's signature and by POSITION
    in the call, which is the pair WRF's renaming breaks: the dummy is
    spelled FRZFACT at every level below SFLX and holds FRZX.
    """
    defs = {n.name: n for n in ast.walk(module_ast)
            if isinstance(n, ast.FunctionDef)}
    assert callee in defs, "%s not found in %s" % (callee, MIRROR_SRC.name)
    names = [a.arg for a in defs[callee].args.args]
    assert "frzfact" in names, (
        "%s no longer spells its frozen-ground dummy frzfact, as WRF's "
        "NOPAC/SNOPAC/SMFLX do" % callee)
    slot = names.index("frzfact")
    calls = [n for n in ast.walk(defs[caller])
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == callee]
    assert len(calls) == 1, (
        "expected exactly one %s call inside %s, found %d"
        % (callee, caller, len(calls)))
    arg = calls[0].args[slot]
    assert isinstance(arg, ast.Name), (
        "%s passes a non-name in %s's frozen-ground slot" % (caller, callee))
    return arg.id


@pytest.mark.parametrize("callee", ["_noah_nopac", "_noah_snopac"])
def test_mirror_sflx_call_sites_pass_frzx(callee):
    tree = ast.parse(MIRROR_SRC.read_text(encoding="utf-8"))
    passed = _call_arg_in_frz_slot(tree, "_noah_sflx", callee)
    assert passed == "frzx", (
        "the mirror's SFLX hands %s its local %s; WRF's SFLX hands it "
        "FRZX = FRZK*FRZFACT (module_sf_noahlsm.F:2478, :769, :784)"
        % (callee, passed))


def test_mirror_srt_declares_and_spends_frzx():
    text = MIRROR_SRC.read_text(encoding="utf-8")
    tree = ast.parse(text)
    srt = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_noah_srt")
    assert "frzx" in [a.arg for a in srt.args.args], (
        "the mirror's SRT no longer declares frzx")
    assert re.search(r"acrt = cvfrz \* frzx / dice", text), (
        "the mirror's SRT no longer spends frzx at ACRT "
        "(module_sf_noahlsm.F:3795)")


# ---------------------------------------------------------------- B: numbers

def _qsat_mix(t, p):
    es = 611.2 * math.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / (p - 0.378 * es)


def _frozen_column(noah, params, rain=5.0):
    """A wet loam column frozen through its whole profile, under rain.

    Soil below freezing everywhere, so SH2O_INIT leaves ice in all four
    layers and SRT's DICE clears its 1e-2 threshold; air above freezing
    and no pack, so the step goes through NOPAC and the precipitation
    arrives at SMFLX as liquid.  This is rain on frozen ground, which is
    the state the limiter exists for.
    """
    row = params.soil[SOILTYP - 1]
    smcmax = row[noah.SOIL_COLS.index("smcmax")]
    sfctmp, sfcprs, dz8w1 = 276.0, 98000.0, 60.0
    psfc = sfcprs * (1.0 + 9.81 * 0.5 * dz8w1 / (287.0 * sfctmp))
    qgh = _qsat_mix(sfctmp, sfcprs)
    tslb = np.array([266.0, 266.5, 267.0, 268.0])
    smc = np.full(4, 0.95 * smcmax)
    sh2o = noah.sh2o_init(smc, tslb, SOILTYP, params)
    return dict(
        psfc=psfc, sfcprs=sfcprs, sfctmp=sfctmp,
        qv1=0.9 * qgh, qgh=qgh, dz8w1=dz8w1,
        glw=0.85 * SIGMA * sfctmp ** 4, swdown=200.0,
        rainbl=rain, sr=0.0, chs=0.01, cqs2=0.02, chs2=0.01, rib=0.0,
        ivgtyp=VEGTYP, isltyp=SOILTYP, vegfra=40.0, shdmin=10.0,
        shdmax=90.0, tmn=272.0, xland=1.0, xice=0.0, snoalb=0.65,
        embck=0.95, tsk=275.0, canwat=0.0, snow=0.0, snowh=0.0,
        snowc=0.0,
        smois=np.asarray(smc, np.float64),
        tslb=np.asarray(tslb, np.float64),
        sh2o=np.asarray(sh2o, np.float64),
        albedo=0.19, albbck=0.19, emiss=0.95, z0=0.1, znt=0.1,
        snotime=0.0, lai=2.0, sfcrunoff=0.0, udrunoff=0.0, acsnow=0.0,
        acsnom=0.0, snopcx=0.0, potevp=0.0, hfx=0.0, qfx=0.0, lh=0.0,
        grdflx=0.0, qsfc=0.0)


def _run_recording(mirror, col, params, substitute=None):
    """Run one column, recording every argument SRT was called with.

    ``substitute``, when given, replaces the frozen-ground argument on
    the way in, which is how the pre-fix answer is produced from a fixed
    tree without a second checkout.
    """
    original = mirror._noah_srt
    sig = inspect.signature(original)
    seen = []

    def spy(*a, **k):
        bound = sig.bind(*a, **k)
        if substitute is not None:
            bound.arguments["frzx"] = substitute(bound.arguments["frzx"])
        seen.append(dict(bound.arguments))
        return original(*bound.args, **bound.kwargs)

    mirror._noah_srt = spy
    try:
        out = mirror.np_noah_column(col, params, DT, DZS)
    finally:
        mirror._noah_srt = original
    assert seen, "SRT was never reached: the column did not run SMFLX"
    return out, seen[-1]


def _srt_runoff1(mirror, a):
    """SRT's surface runoff, recomputed by hand from its own inputs.

    Transcribed from module_sf_noahlsm.F SRT: the DD/DDT infiltration
    capacity, the CVFRZ series that builds FCR, the WDFCND floor and the
    PX/DT ceiling, then the excess over the limit.
    """
    zsoil, sice, sh2oa = a["zsoil"], a["sice"], a["sh2oa"]
    nsoil, dt, pcpdrp = a["nsoil"], a["dt"], a["pcpdrp"]
    smcmax, smcwlt = a["smcmax"], a["smcwlt"]
    smcav = smcmax - smcwlt
    dice = -zsoil[0] * sice[0]
    dd = -zsoil[0] * smcav * (
        1.0 - (sh2oa[0] + sice[0] - smcwlt) / smcav)
    for ks in range(1, nsoil):
        dice += (zsoil[ks - 1] - zsoil[ks]) * sice[ks]
        dd += (zsoil[ks - 1] - zsoil[ks]) * smcav * (
            1.0 - (sh2oa[ks] + sice[ks] - smcwlt) / smcav)
    ddt = dd * (1.0 - math.exp(-a["kdt"] * dt / 86400.0))
    px = max(pcpdrp * dt, 0.0)
    infmax = (px * (ddt / (px + ddt))) / dt
    assert dice > 1.0e-2, (
        "DICE = %g did not clear SRT's 1e-2 threshold, so the limiter "
        "never ran and this test proves nothing" % dice)
    acrt = CVFRZ * a["frzx"] / dice
    ssum = 1.0
    for j in range(1, CVFRZ):
        k = 1
        for jj in range(j + 1, CVFRZ):
            k *= jj
        ssum += acrt ** (CVFRZ - j) / float(k)
    fcr = 1.0 - math.exp(-acrt) * ssum
    infmax *= fcr
    _, wcnd = mirror._noah_wdfcnd(sh2oa[0], smcmax, a["bexp"], a["dksat"],
                                  a["dwsat"], max(sice))
    infmax = min(max(infmax, wcnd), px / dt)
    return (pcpdrp - infmax if pcpdrp > infmax else 0.0), fcr, acrt


def test_mirror_frozen_column_sends_frzx_not_frzfact(mirror, noah, params):
    """The value SRT receives is FRZK*FRZFACT, not FRZFACT."""
    out, a = _run_recording(mirror, _frozen_column(noah, params), params)
    assert out["skip"] == 0, "the column did not run as land"
    assert out["ebal_case"] == 0, "the column did not take the NOPAC branch"
    frzk = params.gen[noah.GEN["frzk"]]
    frzx = out["frzx"]
    frzfact = frzx / frzk
    assert a["frzx"] == pytest.approx(frzx, rel=0.0, abs=0.0), (
        "SRT received %.10g; REDPRM's FRZX for this column is %.10g and "
        "its FRZFACT is %.10g, which is %.2fx larger"
        % (a["frzx"], frzx, frzfact, 1.0 / frzk))


def test_mirror_frozen_column_infiltration_limit_matches_wrf(mirror, noah,
                                                             params):
    """Runoff off the frozen column is the FRZX answer, by hand."""
    out, a = _run_recording(mirror, _frozen_column(noah, params), params)
    frzk = params.gen[noah.GEN["frzk"]]
    hand, fcr, _ = _srt_runoff1(mirror, a)
    assert out["runoff1"] == pytest.approx(hand, rel=1e-12, abs=0.0), (
        "the mirror's surface runoff %.12g does not match the SRT path "
        "recomputed from its own inputs, %.12g" % (out["runoff1"], hand))

    wrong = dict(a)
    wrong["frzx"] = a["frzx"] / frzk          # what FRZFACT would have been
    hand_wrong, fcr_wrong, _ = _srt_runoff1(mirror, wrong)
    assert fcr_wrong > fcr, (
        "FRZFACT should give the more permissive FCR; got %.6g against "
        "%.6g" % (fcr_wrong, fcr))
    assert fcr < 0.2 and fcr_wrong > 0.9, (
        "this column no longer separates the two values: FCR %.4g on "
        "FRZX against %.4g on FRZFACT" % (fcr, fcr_wrong))
    assert hand_wrong != pytest.approx(hand, rel=1e-9, abs=0.0), (
        "the two values give the same runoff on this column, so it is "
        "not a witness any more")


def test_mirror_frozen_column_moves_when_the_wrong_value_is_forced(
        mirror, noah, params):
    """State after the step differs from the pre-fix state, and by how much."""
    frzk = params.gen[noah.GEN["frzk"]]
    good, ga = _run_recording(mirror, _frozen_column(noah, params), params)
    bad, ba = _run_recording(mirror, _frozen_column(noah, params), params,
                             substitute=lambda v: v / frzk)
    # The two arms are the two readings of WRF's argument, and nothing
    # else: SRT sees FRZX in one and FRZFACT in the other.
    assert ga["frzx"] == pytest.approx(good["frzx"], rel=0.0, abs=0.0)
    assert ba["frzx"] == pytest.approx(good["frzx"] / frzk, rel=0.0, abs=0.0)
    assert bad["sfcrunoff"] < good["sfcrunoff"], (
        "the permissive limit should infiltrate more and run off less: "
        "%.6g against %.6g" % (bad["sfcrunoff"], good["sfcrunoff"]))
    assert good["smois"][0] < bad["smois"][0], (
        "the top layer should end drier under the correct limit")
    # A single 60 s step on one column: small in absolute terms, and the
    # point is that it is not zero and does not change sign.
    assert good["sfcrunoff"] - bad["sfcrunoff"] > 1.0e-3
    assert bad["smois"][0] - good["smois"][0] > 1.0e-6


def test_unfrozen_column_is_untouched_by_the_value(mirror, noah, params):
    """No ice, no limiter: the two values give the same column."""
    base = _frozen_column(noah, params)
    base["tslb"] = np.full(4, 285.0)
    base["sh2o"] = np.array(base["smois"], np.float64)
    base["tsk"], base["tmn"] = 286.0, 285.0

    def fresh():
        c = dict(base)
        for k in ("smois", "tslb", "sh2o"):
            c[k] = np.array(base[k], np.float64)
        return c

    frzk = params.gen[noah.GEN["frzk"]]
    good, _ = _run_recording(mirror, fresh(), params)
    bad, _ = _run_recording(mirror, fresh(), params,
                            substitute=lambda v: v / frzk)
    assert good["sfcrunoff"] == bad["sfcrunoff"]
    assert np.array_equal(good["smois"], bad["smois"])
