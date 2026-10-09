"""Committed arithmetic/coupling identity for WOOF global Level 5.

One pin document per gravity-wave ARITHMETIC: the [semi_implicit] scheme
(which linear operator is treated implicitly) together with the [time]
integrator family (how that operator and the explicit right-hand side are
composed).  Every arithmetic is shipped and selectable:

* ``vertical_modes`` under the split-era steppers (``ssprk3`` / ``rk4``):
  the symmetric sqrt-Crank-Nicolson split, v2;
* ``external`` under the same steppers: the barotropic proxy's Lie
  split, v1, kept bit-identical for identity-locked archives.  Its pin
  string is exactly the one the single document carried when it was the
  only scheme, so every checkpoint, receipt and export written in that
  era keeps its pin under ``scheme = "external"`` (``pins_hash``
  reproduces the committed literal in tests/test_arwen_global_pins.py);
* either scheme under the IMEX integrator (imex.py): one Runge-Kutta
  pair with shared abscissae, v3 / v2.

Documents of different arithmetics hash apart, so a checkpoint of one
never resumes under another.  ``PINS_HASH`` / ``PIN_DOCUMENT`` are the
default arithmetic's; a door that knows its config asks for its own.
"""
from __future__ import annotations

import copy
import hashlib
import json

from woof.globe.spectral.pins import PINS_HASH as LEVEL3_PINS_HASH

from .imex import DEFAULT_INTEGRATOR as _DEFAULT_TIME_INTEGRATOR, IMEX_INTEGRATORS
from .semilag.pins import SEMILAG_PIN_OVERRIDES, SEMILAG_SEMI_IMPLICIT_PINS
from .semilag.step import SEMILAG_INTEGRATORS

LEVEL4_PINS_HASH = "0f4f5bb173813ca196b904de70ab3a21b82f0c28ead67e005fcc29a687b043d5"

DEFAULT_SEMI_IMPLICIT_SCHEME = "vertical_modes"
#: The [time] integrator the pin document describes when a caller names
#: none (the config default, imex.py); the split-era steppers share one
#: arithmetic family ("split").
DEFAULT_INTEGRATOR = _DEFAULT_TIME_INTEGRATOR
SPLIT_INTEGRATORS = ("ssprk3", "rk4")

# v2 (audit 2026-09-01 DN-1): every vertical gravity-wave mode is
# implicit through the isothermal-reference vertical-structure operator,
# applied as the square root of the off-centred Crank-Nicolson map on
# either side of the explicit step (symmetric split: a balanced rest
# state over a 2 km mountain keeps its winds under 1.6e-2 m/s over the
# first hour at dt=60 s, where the corrector-after-the-step order of the
# v1 proxy reached 2.7 m/s); the v1 barotropic proxy left every internal
# mode explicit (rest ceiling 104.7 s at T533 regardless of its reference
# speed) and stays selectable as [semi_implicit] scheme = "external"
# under its own v1 pin, which is the pin of record of every archive
# written before the vertical-mode scheme existed.  The v2 string is
# kept exactly as first committed (its "-by-config" tail included): it
# is the identity every vertical-mode checkpoint written since then
# carries, and an identity string is not prose.
SEMI_IMPLICIT_PINS = {
    "vertical_modes": (
        "vertical-mode-symmetric-sqrt-off-centred-crank-nicolson-helmholtz-"
        "v2-or-barotropic-lie-v1-by-config"
    ),
    "external": "barotropic-crank-nicolson-helmholtz-mode-v1",
}
#: The same two operators under the IMEX integrator (imex.py): the
#: operator is composed with the explicit right-hand side by one
#: Runge-Kutta pair with shared abscissae, so the split maps of the
#: strings above never run; a distinct arithmetic, a distinct pin.
IMEX_SEMI_IMPLICIT_PINS = {
    "vertical_modes": (
        "vertical-mode-imex-ssp3-shared-abscissae-helmholtz-stage-solves-v3"
    ),
    "external": "barotropic-imex-ssp3-shared-abscissae-stage-solves-v2",
}


def integrator_family(integrator: str) -> str:
    """``"split"`` for the pre/post-map steppers, the integrator's own
    name for an IMEX tableau or the semi-Lagrangian core."""
    key = str(integrator).lower()
    if key in SPLIT_INTEGRATORS:
        return "split"
    if key in SEMILAG_INTEGRATORS:
        return key
    if key in IMEX_INTEGRATORS:
        return key
    raise ValueError(
        f"unknown time integrator {integrator!r}; pinned: "
        + ", ".join(
            repr(name) for name in
            (*SPLIT_INTEGRATORS, *IMEX_INTEGRATORS, *SEMILAG_INTEGRATORS)
        )
    )


def arithmetic_label(semi_implicit_scheme: str, integrator: str = DEFAULT_INTEGRATOR) -> str:
    """The name a pin is reported under: the scheme alone for the split
    era, ``scheme/integrator`` under an IMEX integrator."""
    family = integrator_family(integrator)
    if family == "split":
        return str(semi_implicit_scheme)
    return f"{semi_implicit_scheme}/{family}"

_PIN_DOCUMENT_TEMPLATE = {
    "schema": "woof.global-pins/v3",
    "level3_parent_pins_sha256": LEVEL3_PINS_HASH,
    "level4_parent_pins_sha256": LEVEL4_PINS_HASH,
    "relationship": (
        "additive global research model and one-way parent source; ordinary "
        "regional WOOF numerics remain unchanged until an explicit install/"
        "attach call consumes validated artifacts"
    ),
    "state": [
        "relative-vorticity", "divergence", "potential-temperature",
        "log-surface-pressure", "qv", "qc", "qr", "qi", "qs", "qg",
        "nc", "nr", "ni", "ns", "ng",
    ],
    # v2 (finding 2026-09-02): the five condensate species and the five
    # number moments are grid-point fields on the Gaussian grid, not
    # spectral coefficients.  Under v1 every hydrometeor was a truncated
    # spectral tracer; the positivity machinery that kept those fields
    # nonnegative rescaled every column on a level by the level's
    # clipped-to-unclipped mass ratio, which moved 37 percent of the
    # planet's cloud water and 9-32 percent of its rain, ice and graupel
    # out of the columns that held them on every pass (four per step)
    # into clear air, where the microphysics evaporated it; total surface
    # precipitation equalled convective precipitation to four decimals.
    "representation": {
        "spectral": [
            "relative-vorticity", "divergence", "potential-temperature",
            "log-surface-pressure", "qv",
        ],
        "grid": ["qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng"],
        "version": "vapor-spectral-condensate-and-moments-gaussian-grid-v2",
    },
    "vertical_coordinate": "hydrostatic-A-plus-B-times-surface-pressure-v1",
    # v2 (audit 2026-09-01 DN-3): the half layer above each interface
    # integrates Tv linear in ln p (gradient from the neighbouring
    # levels) instead of the level temperature alone, and the momentum
    # pressure gradient is the exact full-level grad(ln p_k) from grad(ln
    # ps).  The v1 pair left a 1.06 m/s per hour top-level acceleration
    # at rest over a 2 km mountain; v2 measures 0.002.
    "hydrostatic_integration": (
        "piecewise-isothermal-full-layers-linear-ln-p-half-layer-virtual-"
        "temperature-v2"
    ),
    "mass_continuity": "layer-pressure-flux-form-with-top-zero-omega-v1",
    # v3 (audit 2026-09-01 DN-4): the vertical flux of theta and every
    # tracer is the van Leer limited second-order upwind reconstruction
    # instead of first-order donor cell (numerical diffusivity |omega|
    # dp/2, 1.31e-4 K/s against 4.19e-5 on the smooth-profile test).
    # v4 (finding 2026-09-02): theta and vapor keep the spectral flux
    # form of v3; the ten grid tracers are carried once per step by the
    # directionally split, van Leer limited, sub-cycled flux-form
    # transport on the Gaussian cells (transport.GridTracerTransport)
    # with the step's mass fluxes averaged between its start and its
    # advanced state, positive by construction.
    "scalar_transport": (
        "spectral-horizontal-pressure-mass-flux-and-vertical-van-leer-limited-"
        "upwind-theta-and-vapor-plus-grid-point-split-van-leer-flux-form-"
        "condensate-and-moments-v4"
    ),
    "moment_policy": "five-Morrison-number-moments-grid-point-nonnegative-v2",
    # The moments have no spectral coefficients to diffuse: the transport's
    # limiter is their only dissipation (v2 retires the v1 hyperdiffusion).
    "moment_diffusion": "none-grid-point-limiter-only-v2",
    "moment_migration": "level4-explicit-zero-seed-with-hash-bound-receipt-v1",
    "momentum": (
        "vector-invariant-vorticity-divergence-hybrid-pressure-midpoint-"
        "pressure-gradient-v2"
    ),
    # "semi_implicit" is filled per scheme by pin_document (SEMI_IMPLICIT_PINS).
    "physics_split": "transactional-strang-half-full-half-v2",
    # The lid absorber is dycore-owned (moved out of the reference suite
    # at its v8 bump): it compensates the rigid p_top lid, so it acts on
    # every physics half-step's grid winds in every physics mode - the
    # reference-suite copy left arwen-native runs unprotected and the
    # T255 five-scheme run died at hour 6.55 on the 140 K research bound.
    "top_absorber": (
        "dycore-ring-mean-gated-cos2-graded-anomaly-rayleigh-per-half-step-v1"
    ),
    "reference_physics": (
        "gray-radiation-bulk-surface-implicit-vertical-mixing-convective-"
        "adjustment-saturation-warm-rain-mixed-phase-fallout-v1"
    ),
    "native_physics": {
        "adapter": "woof-cuda-column-suite-v1",
        "order": ["rrtmgp", "sfclay", "noah", "ysu", "morrison"],
        "vertical_boundary": "global-top-to-surface-to-native-bottom-to-top-fp32-v1",
        "transaction": "copy-run-validate-return-no-caller-mutation-v1",
        "persistent_state": "checkpointed-physics-namespace-v1",
        # v2: the closure ledger counts Noah's runoff stores at
        # land_fraction pricing, so soil water SSTEP sheds into
        # sfcrunoff/udrunoff reads as moved, not destroyed (the v1 ledger
        # priced the first land call on saturated ice-sheet columns as a
        # 1072 kg/m2 repair and failed both T255 qualification receipts).
        # v3: the runtime clears every store move through the surface
        # reservoir at ledger pricing -- the land step debits the priced
        # store delta of the precipitation Noah consumes, and the PBL step
        # books the surface moisture flux its vapor bottom boundary injects
        # -- so the closure residual measures only unexplained movement.
        # Under v2 the reservoir kept its copy of consumed precipitation
        # and the repair read the heaviest one-interval land-column burst:
        # both T255 qualification receipts failed
        # physics_water_repair_max_step_kg_m2 at 1.0943603515625 kg/m2
        # against 5e-4, bit-identical across restart (2026-08-31), with
        # the unbooked qfx flux at 1.5869e-3 per half-call beneath it.
        # v4: runoff is a booked EXIT, not a held store.  The land step
        # still debits the reservoir by the priced runoff-store increment
        # (the precipitation credit funds it), but books the same amount
        # to the cumulative water_outflow_kg_m2 account and the closure
        # ledger counts held water plus outflow instead of the monotone
        # kernel accumulators, which under v3 grew inside the pinned
        # conservation total forever and squeezed the global
        # atmosphere+reservoir mean by every kg of runoff.
        "water_closure": (
            "finite-surface-reservoir-runtime-booked-store-flux-and-outflow-v4"
        ),
        "admission": "device-pending-until-hash-bound-target-device-evidence-v1",
    },
    # v3: the total-water ledger the global drift fixer measures counts
    # Noah's runoff stores at land_fraction pricing, matching the native
    # closure's ledger; the two ledgers counting different stores would
    # hand the fixer whatever the closure stops repairing.
    # v4, two changes with one root (the h37.7 T255 reservoir-floor death,
    # 2026-08-31): (a) the ledger counts runoff as the cumulative
    # water_outflow_kg_m2 exit account instead of held-store content,
    # matching the native closure's v4 ledger, so conservation is held
    # water plus booked exits; (b) the spectral clamp closures (exchange
    # clamp and positivity-repair projection delta) debit the reservoir
    # UNIFORMLY by the area-weighted global mean of the water they create,
    # not per column -- the per-column debit was a perpetual concentrated
    # fee with no return path (measured 0.07-0.08 kg/m2 per 60 s step at
    # the ring columns of one fresh T21 feature; ~0.22 at T255 convective
    # sharpness, which is 500 kg/m2 in exactly 37.7 h) and the fee belongs
    # to the truncated representation of the whole field, not the column
    # the ringing lands on.
    # v5 (audit 2026-09-01 VTW-1): the uniform levy was itself a
    # reservoir-to-atmosphere moisture source -- 2.508 mm/day sustained on
    # the shipped T63 GDAS reference config, 2.1x the model's own
    # precipitation -- so both clamps now close INSIDE the atmosphere: the
    # positive part of each (species, level) is rescaled so its
    # mass-weighted global integral is unchanged by the clip (hole
    # filling).  No clamp touches the reservoir; the fixer's per-step
    # relative magnitude is a receipted, gated measurement.
    # v6 (finding 2026-09-02): the v5 per-level global rescale is
    # retired.  Vapor, the one water field in the spectral basis, is
    # clipped and closed INSIDE ITS OWN COLUMN (each column's vapor
    # integral is unchanged by the clip; no other column and no reservoir
    # pays); the condensate species are grid tracers, nonnegative by
    # construction, and no clamp or rescale touches them.
    "water_repair": (
        "finite-grid-surface-and-soil-reservoir-column-local-vapor-hole-"
        "filling-grid-point-condensate-booked-outflow-and-global-drift-v6"
    ),
    "checkpoint": (
        "hash-bound-five-field-spectral-atmosphere-ten-grid-tracers-grid-"
        "surface-physics-state-and-eight-run-trackers-v3"
    ),
    "parent_export": (
        "hash-bound-regular-latlon-fifteen-field-dynamics-continuity-and-surface-"
        "spectral-sampled-dynamics-bilinear-grid-tracers-v3"
    ),
    "regional_translation": {
        "horizontal": "periodic-regular-latlon-bilinear-v1",
        "vertical": "independent-column-log-pressure-v1",
        "surface_pressure": "single-layer-hypsometric-virtual-temperature-v1",
        "wind": "earth-to-grid-rotation-then-nonperiodic-C-grid-average-v1",
        "geopotential": "target-terrain-hydrostatic-reintegration-v1",
        "vertical_velocity": "interpolated-parent-w-zero-boundary-flux-v1",
        "dry_mass": "sum-dp-over-one-plus-total-water-v1",
        "coupled_units": "existing-WOOF-WRF-u-v-theta-phi-mu-qv-v1",
        "boundary_order": "west-south-storage-east-north-outermost-first-v1",
        "attachment": "existing-eager-or-streaming-LateralBoundaries-only-v1",
    },
    "device_qualification": (
        "uninterrupted-versus-midpoint-restart-bit-exact-plus-device-identity-v1"
    ),
    "admission": "research-only-woof-global-v1",
}


def pin_document(
    semi_implicit_scheme: str = DEFAULT_SEMI_IMPLICIT_SCHEME,
    integrator: str = DEFAULT_INTEGRATOR,
) -> dict[str, object]:
    """The pin document of the arithmetic a config's scheme and
    integrator run.  The split-era document is byte-identical to the one
    every archive of that era carries; an IMEX integrator changes only the
    ``semi_implicit`` entry."""
    family = integrator_family(integrator)
    if family == "split":
        table = SEMI_IMPLICIT_PINS
    elif family in SEMILAG_INTEGRATORS:
        table = SEMILAG_SEMI_IMPLICIT_PINS
    else:
        table = IMEX_SEMI_IMPLICIT_PINS
    try:
        pin = table[semi_implicit_scheme]
    except KeyError:
        raise ValueError(
            f"unknown semi-implicit scheme {semi_implicit_scheme!r} for "
            f"integrator family {family!r}; pinned: "
            + ", ".join(repr(name) for name in table)
        ) from None
    document = copy.deepcopy(_PIN_DOCUMENT_TEMPLATE)
    document["semi_implicit"] = pin
    if family in SEMILAG_INTEGRATORS:
        # Four keys, not one.  The template is untouched, so every
        # split-era and IMEX document hashes exactly as before; what
        # changes here is that a semi-Lagrangian archive states its real
        # arithmetic instead of hiding a momentum, a transport and a
        # checkpoint change inside a gravity-wave string.
        document.update(SEMILAG_PIN_OVERRIDES)
    return document


def pins_hash(
    semi_implicit_scheme: str = DEFAULT_SEMI_IMPLICIT_SCHEME,
    integrator: str = DEFAULT_INTEGRATOR,
) -> str:
    raw = json.dumps(
        pin_document(semi_implicit_scheme, integrator),
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(raw).hexdigest()


PIN_DOCUMENT = pin_document()
PINS_HASH = pins_hash()
#: scheme -> pins hash under the split-era steppers (the pins every
#: archive written before the IMEX integrator carries).
PINS_HASH_BY_SCHEME = {scheme: pins_hash(scheme, "ssprk3") for scheme in SEMI_IMPLICIT_PINS}
#: arithmetic label (arithmetic_label) -> pins hash, every arithmetic this
#: build integrates: both schemes under the split steppers and under each
#: IMEX tableau.
PINS_HASH_BY_ARITHMETIC = {
    **{
        arithmetic_label(scheme, integrator): pins_hash(scheme, integrator)
        for integrator in ("ssprk3", *IMEX_INTEGRATORS)
        for scheme in SEMI_IMPLICIT_PINS
    },
    **{
        arithmetic_label(scheme, integrator): pins_hash(scheme, integrator)
        for integrator in SEMILAG_INTEGRATORS
        for scheme in SEMILAG_SEMI_IMPLICIT_PINS
    },
}
#: The pins hashes an artifact may carry and still be this build's arithmetic.
KNOWN_PINS_HASHES = frozenset(PINS_HASH_BY_ARITHMETIC.values())
#: The four pins of the spectral-tracer era (every archive written before
#: the grid tracers, 2026-09-02): the vertical-mode split document as
#: first committed (edec21516), the barotropic proxy's single-document
#: pin (19f5ba33c), and the two IMEX documents as they stood at the owner
#: tip 1bc5585d3.  Not this build's arithmetic -- their condensate
#: transport is the defect the grid tracers retire -- so no restart
#: resumes under them; a reader without a scheme may still INSPECT such
#: a checkpoint (the instruments read the pre-fix arms through this door).
SPECTRAL_TRACER_ERA_PINS_HASHES = frozenset({
    "d05af5c097381e39c0e932992e6afed1722e2e94d508fe4c2ef634038045a0be",
    "d5afac4ed5f20778197202390e1fb3a0c47d495b70a913175b861a7c23b842f1",
    "26533ea4cd81faab809ef55374907de39ca313025fa92f6e1f2807d502982a6b",
    "15b5d7012e190e33fa49384cabd0ec57263cb905570bc5f963f6221ab4cd934d",
})
#: The two RETIRED semi-Lagrangian pins (2026-09-06): v1, before the tracer
#: mass fixer's clip-deficit stage, and v2, which advected theta as a
#: deviation from the reference profile with the reference's material
#: tendency carried on the grid (the lid warming of 32 K a day, semilag.rhs).
#: Not this build's arithmetic, so no restart resumes under them; a reader
#: without a scheme may INSPECT such a checkpoint, which is how the arms
#: that measured the defect stay readable beside the arms that fixed it.
RETIRED_SEMILAG_PINS_HASHES = frozenset({
    "7082aea098df6c9fca5385020bbc8836ed46cbe2e91a5b15c423fc21ba84fd5b",
    "a3f927dccabd32e9ba93f2ee65f63ff528764833d23f7757ca858b53fa6acd5b",
})
#: Every pin a reader may inspect without integrating under it.
INSPECTABLE_RETIRED_PINS_HASHES = (
    SPECTRAL_TRACER_ERA_PINS_HASHES | RETIRED_SEMILAG_PINS_HASHES
)
#: The pins WOOF 1.0.0 wrote, one per arithmetic: the v2 document, whose
#: identity texts still named the engine's earlier name.  1.0.1 rewords
#: those texts (the v3 document above: its schema, the relationship
#: sentence, the adapter, coupled-units and admission ids) and moves no
#: arithmetic, so a checkpoint, receipt, export or migration record 1.0.0
#: wrote carries one of these and IS the arithmetic of its label.  Every
#: reader accepts both; every writer writes v3.  Written by the assembly
#: from the v2 document as 1.0.0 shipped it, not typed.
WOOF_1_0_0_PINS_HASH_BY_ARITHMETIC = {
    "external": (
        "f536e10061499732a08bd6e32cb45160820bb55519df5c0721be4c33fbf573a0"
    ),
    "external/imex_ssp3": (
        "6066104ed4e0004ea7e17a5e99fd014968ec97241a93dc112400fd074f909546"
    ),
    "vertical_modes": (
        "4c4340945258b8c3e5648d0350af6e84ceb1e3d69d17c8b7d4603d361cb92aa6"
    ),
    "vertical_modes/imex_ssp3": (
        "c5d0545d71c6b3fefd2aa5899e720161b0f0d38ff4522a50f66cdc44a52a992b"
    ),
    "vertical_modes/sl_si": (
        "d82dc8ae4b0b5ea75aadccbf8b2ef5f8b3330d5b36670f00234f28bfe6e9d5d8"
    ),
}
#: The 1.0.0 pins, as a set.
LEGACY_PINS_HASHES = frozenset(WOOF_1_0_0_PINS_HASH_BY_ARITHMETIC.values())
#: Every pin a checkpoint, export, receipt or migration record may carry
#: and still be one of this build's arithmetics.
ACCEPTED_PINS_HASHES = KNOWN_PINS_HASHES | LEGACY_PINS_HASHES


def accepted_pins_hashes(
    semi_implicit_scheme: str = DEFAULT_SEMI_IMPLICIT_SCHEME,
    integrator: str = DEFAULT_INTEGRATOR,
) -> frozenset[str]:
    """The pins an archive of this arithmetic may carry and resume under
    it: its v3 pin and, when 1.0.0 shipped the arithmetic, the v2 pin
    1.0.0 wrote for it."""
    current = pins_hash(semi_implicit_scheme, integrator)
    legacy = WOOF_1_0_0_PINS_HASH_BY_ARITHMETIC.get(
        arithmetic_label(semi_implicit_scheme, integrator))
    return frozenset({current} if legacy is None else {current, legacy})


def scheme_of_pins_hash(value: object) -> str | None:
    """The arithmetic label (``scheme`` or ``scheme/integrator``) whose
    pin ``value`` is, or None for a pin no shipped arithmetic carries (an
    earlier or later era of the document)."""
    for label, digest in PINS_HASH_BY_ARITHMETIC.items():
        if value == digest:
            return label
    # the pin WOOF 1.0.0 wrote for the same arithmetic (v2 document)
    for label, digest in WOOF_1_0_0_PINS_HASH_BY_ARITHMETIC.items():
        if value == digest:
            return label
    return None


def pins_receipt(
    semi_implicit_scheme: str = DEFAULT_SEMI_IMPLICIT_SCHEME,
    integrator: str = DEFAULT_INTEGRATOR,
) -> dict[str, object]:
    return {
        "pins": pin_document(semi_implicit_scheme, integrator),
        "sha256": pins_hash(semi_implicit_scheme, integrator),
    }


__all__ = [
    "ACCEPTED_PINS_HASHES",
    "DEFAULT_INTEGRATOR",
    "LEGACY_PINS_HASHES",
    "WOOF_1_0_0_PINS_HASH_BY_ARITHMETIC",
    "accepted_pins_hashes",
    "SEMILAG_PIN_OVERRIDES",
    "SEMILAG_SEMI_IMPLICIT_PINS",
    "DEFAULT_SEMI_IMPLICIT_SCHEME",
    "IMEX_SEMI_IMPLICIT_PINS",
    "KNOWN_PINS_HASHES",
    "SPECTRAL_TRACER_ERA_PINS_HASHES",
    "RETIRED_SEMILAG_PINS_HASHES",
    "INSPECTABLE_RETIRED_PINS_HASHES",
    "LEVEL4_PINS_HASH",
    "PINS_HASH",
    "PINS_HASH_BY_ARITHMETIC",
    "PINS_HASH_BY_SCHEME",
    "PIN_DOCUMENT",
    "SEMI_IMPLICIT_PINS",
    "SPLIT_INTEGRATORS",
    "arithmetic_label",
    "integrator_family",
    "pin_document",
    "pins_hash",
    "pins_receipt",
    "scheme_of_pins_hash",
]
