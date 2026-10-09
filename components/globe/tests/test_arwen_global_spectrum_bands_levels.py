"""The banded-spectrum instrument reads its levels from the receipt, and
refuses rather than guesses.

Origin (2026-09-07, the equal-cost grade refutation): the receipt keeps the
half-level coefficients under ``config`` and the instrument looked under
``vertical``, so it never found them and fell back to hard-coded levels 26
and 16 as 500 and 250 hPa.  On the 40-level surface-stretched stack of
record those are 793 and 295 hPa at 1000 hPa surface pressure, and the
ratios published under the wrong labels disagreed with the pressure-surface
instrument by 1.6x to 2.5x on identical pairs.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "arwen_global_spectrum_bands", ROOT / "tools" / "arwen_global_spectrum_bands.py")
SB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SB)

NLEV = 40


def _stack():
    """A pure-sigma 40-level stack: half levels b = k / 40, a = 0, so full
    level k sits at (k + 0.5) / 40 of the surface pressure."""
    b = np.linspace(0.0, 1.0, NLEV + 1)
    a = np.zeros(NLEV + 1)
    return a.tolist(), b.tolist()


def test_levels_come_from_the_receipt_config():
    a, b = _stack()
    receipt = {"config": {"a_half_pa": a, "b_half": b},
               "vertical": {"reference_surface_pa": 100000.0, "nlev": NLEV}}
    pres = SB._level_pressures_pa(receipt, NLEV)
    assert pres is not None and pres.shape == (NLEV,)
    # full level k at (k + 0.5) / 40 of 1000 hPa: level 19 is 487.5 hPa,
    # level 20 is 512.5 hPa, so 500 hPa is equidistant and argmin takes 19
    assert abs(pres[19] - 48750.0) < 1e-6
    assert abs(pres[20] - 51250.0) < 1e-6
    assert int(np.argmin(np.abs(pres - 25000.0))) == 9   # 237.5 vs 262.5 hPa


def test_vertical_block_without_arrays_is_not_enough():
    # the shape the runner writes: a summary under vertical, arrays under config
    receipt = {"vertical": {"reference_surface_pa": 101325.0, "nlev": NLEV, "coordinate": "surface_stretched"}}
    assert SB._level_pressures_pa(receipt, NLEV) is None


def test_spectrum_refuses_to_guess_a_level(tmp_path):
    T = 8
    vort = np.zeros((NLEV, T + 1, T + 1), dtype=np.complex64)
    lnps = np.zeros((T + 1, T + 1), dtype=np.complex64)
    ck = tmp_path / "arwen_global_step00000010.npz"
    np.savez(ck, atmosphere__vorticity=vort, atmosphere__log_surface_pressure=lnps)
    # a receipt without the coefficients: the old code guessed level 26 here
    (tmp_path / "arwen-global-receipt.json").write_text(json.dumps(
        {"vertical": {"reference_surface_pa": 101325.0, "nlev": NLEV}}), encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        SB.spectrum(ck, [500.0])
    assert "no longer guesses" in str(err.value)


def test_spectrum_reports_the_level_and_its_pressure(tmp_path):
    T = 8
    vort = np.zeros((NLEV, T + 1, T + 1), dtype=np.complex64)
    lnps = np.zeros((T + 1, T + 1), dtype=np.complex64)
    ck = tmp_path / "arwen_global_step00000010.npz"
    np.savez(ck, atmosphere__vorticity=vort, atmosphere__log_surface_pressure=lnps)
    a, b = _stack()
    (tmp_path / "arwen-global-receipt.json").write_text(json.dumps(
        {"config": {"a_half_pa": a, "b_half": b},
         "vertical": {"reference_surface_pa": 100000.0, "nlev": NLEV}}), encoding="utf-8")
    out = SB.spectrum(ck, [500.0, 250.0])
    assert out["levels"]["500.0"]["index"] == 19
    assert abs(out["levels"]["500.0"]["reference_pressure_hpa"] - 487.5) < 1e-9
    assert out["levels"]["250.0"]["index"] == 9
    assert abs(out["levels"]["250.0"]["reference_pressure_hpa"] - 237.5) < 1e-9
