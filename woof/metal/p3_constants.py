"""Stable host FP32 constants shared by the P3 Metal stages."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np


P3_CONSTANT_NAMES = (
    "PI",
    "THRD",
    "SXTH",
    "PIOV3",
    "PIOV6",
    "MAX_TOTAL_NI",
    "NCCNST",
    "KC",
    "KR",
    "CP",
    "INV_CP",
    "G",
    "RD",
    "RV",
    "EP_2",
    "RHOSUR",
    "RHOSUI",
    "AR",
    "BR",
    "F1R",
    "F2R",
    "ECR",
    "RHOW",
    "CPW",
    "INV_RHOW",
    "MU_R_CONSTANT",
    "INV_DRMAX",
    "RHO_RIMEMIN",
    "RHO_RIMEMAX",
    "INV_RHO_RIMEMAX",
    "QSMALL",
    "NSMALL",
    "BSMALL",
    "BIMM",
    "AIMM",
    "RIN",
    "MI0",
    "ECI",
    "ERI",
    "BCN",
    "DBRK",
    "NMLTRATIO",
    "MU_I_INITIAL",
    "CONS1",
    "CONS2",
    "CONS3",
    "CONS4",
    "CONS5",
    "CONS6",
    "CONS7",
    "CONS8",
    "E0",
)

P3_CONSTANT_INDEX = MappingProxyType({
    name: np.uint32(index)
    for index, name in enumerate(P3_CONSTANT_NAMES)
})


def p3_latent_heat_values() -> tuple[np.float32, np.float32, np.float32]:
    """Return xxlv, xxls, and xlf using the CPU wrapper's float32 order."""
    xxlv = np.float32(3.1484e6) - np.float32(2370.0) * np.float32(273.15)
    xxls = xxlv + np.float32(0.3337e6)
    xlf = xxls - xxlv
    return xxlv, xxls, xlf


def p3_constant_values(core_module: Any | None = None) -> np.ndarray:
    """Return canonical P3 module constants in the stable Metal ABI order."""
    if core_module is None:
        from woof.core import p3 as core_module

    values = np.empty(len(P3_CONSTANT_NAMES), dtype=np.float32)
    for index, name in enumerate(P3_CONSTANT_NAMES):
        value = np.asarray(getattr(core_module, name))
        if value.shape != () or value.dtype != np.dtype(np.float32):
            raise TypeError(
                f"core P3 constant {name} must be a scalar float32"
            )
        values[index] = value
    values.flags.writeable = False
    return values
