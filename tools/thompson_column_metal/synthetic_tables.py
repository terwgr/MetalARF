"""Deterministic synthetic wrf_44 coefficient tables of the loader's exact shapes."""
import numpy as np

SHAPES = {}
for n in ("tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr"):
    SHAPES[n] = (37, 37, 37, 37)
for n in ("tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2", "tcr_sacr1", "tms_sacr1",
          "tcr_sacr2", "tms_sacr2", "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2"):
    SHAPES[n] = (37, 9, 37, 37)
for n in ("tpi_qrfz", "tni_qrfz", "tpg_qrfz", "tnr_qrfz"):
    SHAPES[n] = (37, 37, 45, 55)
for n in ("tpi_qcfz", "tni_qcfz"):
    SHAPES[n] = (37, 100, 45, 55)
for n in ("tps_iaus", "tni_iaus", "tpi_ide"):
    SHAPES[n] = (64, 55)
for n in ("t_Efrw", "t_Efsw"):
    SHAPES[n] = (100, 100)

# Magnitudes: mass transfers ~1e-5 kg/kg, number transfers ~1e3, fractions in 0..1.
SCALE = {}
for n in SHAPES:
    SCALE[n] = (1.0e3 if n.startswith("tn") and "iaus" not in n else
                1.0 if n in ("tpi_ide", "t_Efrw", "t_Efsw") or n.startswith("tps_") else 1.0e-5)
SCALE["tni_iaus"] = 1.0e3
SCALE["tps_iaus"] = 0.5
for n in ("tpi_qrfz", "tpg_qrfz", "tpi_qcfz"):
    SCALE[n] = 1.0e-6


def table(name, seed):
    n = int(np.prod(SHAPES[name]))
    i = np.arange(n, dtype=np.uint64)
    h = (i * np.uint64(2654435761) + np.uint64(seed * 40503)) % np.uint64(1000003)
    unit = h.astype(np.float64) / 1000003.0
    smooth = 0.5 + 0.5 * np.sin(i.astype(np.float64) * 1.0e-3 + seed)
    return (SCALE[name] * (0.1 + 0.6 * smooth + 0.3 * unit)).astype(np.float64)


def tables():
    return {name: table(name, k + 1) for k, name in enumerate(SHAPES)}
