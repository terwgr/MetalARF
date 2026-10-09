"""Physical and numerical constants for the Level-3 global spectral core."""
from __future__ import annotations

EARTH_RADIUS_M = 6_371_220.0
EARTH_ROTATION_RATE_S = 7.292e-5
GRAVITY_M_S2 = 9.80616
DRY_AIR_GAS_CONSTANT = 287.0
DRY_AIR_CP = 1004.0
KAPPA = DRY_AIR_GAS_CONSTANT / DRY_AIR_CP
REFERENCE_PRESSURE_PA = 100_000.0
SECONDS_PER_DAY = 86_400.0

RESEARCH_ACKNOWLEDGEMENT = "research-only-global-spectral-v1"
RUN_SCHEMA = "gpuwm.global-spectral-run/v1"
CHECKPOINT_SCHEMA = "gpuwm.global-spectral-checkpoint/v1"
RECEIPT_SCHEMA = "gpuwm.global-spectral-receipt/v1"
