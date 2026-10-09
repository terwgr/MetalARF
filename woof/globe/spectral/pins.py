"""Committed arithmetic identity for the Level-3 prototype."""
from __future__ import annotations

import hashlib
import json

PIN_DOCUMENT = {
    "schema": "woof.global-spectral-pins/v2",
    "relationship": (
        "additive standalone research core; existing WOOF regional dycore, "
        "Level-1 verification identities, and Level-2 regional operators are "
        "not replaced or silently invoked"
    ),
    "state": [
        "relative-vorticity",
        "divergence",
        "temperature",
        "log-surface-pressure",
    ],
    "horizontal_basis": "complex-orthonormal-spherical-harmonics-condon-shortley",
    "quadrature": "gauss-legendre-in-sin-latitude-plus-equispaced-longitude",
    "spectral_layout": "triangular-positive-m-packed-nm",
    "nonlinear_evaluation": "dealiased-gaussian-grid-pseudospectral",
    "vector_analysis": "integration-by-parts-vector-spherical-harmonics-v1",
    "vector_inversion": "streamfunction-and-velocity-potential-v1",
    "shallow_water": "vector-invariant-vorticity-divergence-geopotential-v1",
    "williamson2_admission": (
        "alpha-zero-only-with-independent-analytic-vorticity-until-tilted-axis-"
        "balance-is-verified"
    ),
    "primitive_dry": "sigma-hydrostatic-vector-invariant-explicit-v1",
    "hydrostatic_integration": "piecewise-isothermal-sigma-layer-v1",
    "vertical_continuity": "top-down-sigma-continuity-with-zero-boundary-flux-v1",
    "vertical_advection": "centered-sigma-advective-difference-v1",
    "time_integration": ["ssprk3", "rk4"],
    "diffusion": "exact-exponential-total-degree-hyperdiffusion-v1",
    "mass_fixer": "uniform-log-pressure-offset-preserving-global-mean-ps-v1",
    "positive_field_policy": (
        "log(max(x,floor)) analysis; negative input refused; optional "
        "multiplicative arithmetic-mean restoration"
    ),
    "compression": (
        "triangular coefficients; independent symmetric int16 scale per total "
        "degree and leading field; payload hashes; progressive degree decoding"
    ),
    "wind_compression": "vorticity-divergence-carriers-not-independent-u-v-scalars",
    "arbitrary_sampling": (
        "direct spherical-harmonic evaluation in automatically bounded point "
        "chunks; exact poles admitted for scalars and refused for vector gradients"
    ),
    "regional_handoff": (
        "hash-bound regular-latlon-cell-centre export; neutral artifact only, "
        "not a completed WOOF lateral-boundary adapter"
    ),
    "checkpoint": (
        "hash-bound-npz-state-plus-config-arithmetic-identity-and-run-trackers-v1"
    ),
    "receipt": "atomic-self-hashed-pass-or-failure-run-receipt-v1",
    "admission": "research-only-global-spectral-v1",
}


def pins_hash() -> str:
    raw = json.dumps(PIN_DOCUMENT, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


PINS_HASH = pins_hash()
#: The pin WOOF 1.0.0 wrote (the v1 document, whose relationship and
#: regional-handoff sentences named the engine's earlier name).  1.0.1
#: rewords them (v2) and moves no arithmetic, so a Level-3 checkpoint,
#: export, archive or receipt 1.0.0 wrote reads as this build's.  Written
#: by the assembly from the v1 document as 1.0.0 shipped it, not typed.
WOOF_1_0_0_PINS_HASH = (
    "807314e05e550e9219869e9a446cea6be8291cd5c35ba0b6555d3ca40a2f57e4"
)
#: Every pin a Level-3 artifact may carry and be read by this build.
ACCEPTED_PINS_HASHES = frozenset({PINS_HASH, WOOF_1_0_0_PINS_HASH})


def pins_receipt() -> dict:
    return {"pins": PIN_DOCUMENT, "sha256": PINS_HASH}
