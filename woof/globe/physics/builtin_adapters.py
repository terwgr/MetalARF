"""Built-in fail-closed registrations for WOOF global native physics."""
from __future__ import annotations

import hashlib
from pathlib import Path

from woof.globe.physics.registry import (
    register_global_physics_adapter,
    registered_global_physics_adapters,
)

from ..constants import PHYSICS_ADAPTER_SCHEMA
from .native_options import NativePhysicsOptions
from .native_suite import ArwenCudaColumnSuite


ADAPTER_NAME = "arwen-cuda-column-suite-v1"
_ZERO_SHA = "0" * 64


def _arithmetic_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in (
        "native_options.py", "native_batch.py", "native_state.py",
        "native_runtime.py", "native_suite.py", "arwen_massflux.py",
    ):
        data = (root / name).read_bytes()
        digest.update(name.encode())
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _contract() -> dict[str, object]:
    # No evidence receipt exists for this adapter yet: the target-device
    # qualification battery is what produces one.  The zero digest is the only
    # value that says so; hashing a sentence about the missing receipt would
    # put a well-formed digest in a field nothing can verify.
    return {
        "schema": PHYSICS_ADAPTER_SCHEMA,
        "scheme_identity": {
            "radiation": "WOOF RTE+RRTMGP",
            "surface_layer": "WRF-v4.6.1 MM5 SFCLAY option 1/91",
            "land_surface": "WRF-v4.6.1 Noah LSM",
            "pbl": "WRF-v4.6.1 YSU",
            "cumulus": (
                "WRF-v4.6.1 Grell-Freitas (woof/globe/core/kernels/gf.cu, "
                "16-member ensemble closure, shallow arm per gf_ishallow); "
                "default on, cumulus='ntiedtke' selects WRF-v4.6.1 New "
                "Tiedtke (woof/globe/core/kernels/ntiedtke.cu, per-column "
                "Gaussian spacing, momentum coupled, classic closure per "
                "ntiedtke_tiedtke_closure) in the same slot, cumulus='own' "
                "selects arwen-massflux-v1 (woof/globe/physics/"
                "arwen_massflux.py: scale-aware deep/shallow mass flux with "
                "convective momentum transport), cumulus='none' removes the "
                "slot"
            ),
            "microphysics": "WRF-v4.6.1 Morrison two-moment",
        },
        "backend": "cupy-cuda",
        "precision": "float32",
        "required_fields": [
            "u", "v", "theta", "p_half", "p_full", "dp", "exner",
            "temperature", "geopotential",
            "qv", "qc", "qr", "qi", "qs", "qg",
            "nc", "nr", "ni", "ns", "ng", "surface", "physics_state",
        ],
        "pressure_convention": "Pa; native arrays bottom-to-top",
        "vertical_coordinate": (
            "global hybrid A+B*ps converted to native bottom-to-top pressure/"
            "geometric column inputs"
        ),
        "surface_state": (
            "Gaussian-grid SurfaceState (skin, reservoirs, soil columns, the "
            "analysed sea-ice fraction and thickness, and "
            "the static land-use/soil categories, vegetation fraction and "
            "its annual range, LAI, background and snow albedo, deep-soil "
            "temperature, albedo, emissivity and roughness seeded by "
            "woof.globe.statics: WPS_GEOG fields on the Gaussian "
            "grid by default for a real-data run, or the declared synthetic "
            "planet) plus checkpointed PhysicsState for SFCLAY/Noah/"
            "radiation/microphysics inout fields and the category "
            "convention row the land surface checks its tables against"
        ),
        "restart_contract": (
            "all persistent physics arrays and JSON-scalar scheduler state "
            "are checkpointed under the physics__ namespace"
        ),
        "budget_contract": (
            "local atmosphere+surface water closes through explicit finite "
            "surface-ledger residual; energy change is measured"
        ),
        "evidence_receipt_sha256": _ZERO_SHA,
        "arithmetic_sha256": _arithmetic_hash(),
        # Device-pending again as of 2026-09-01: the Grell-Freitas cumulus
        # component joined the suite (default on) after the qualification
        # battery that promoted it, so the evidence on file (native-device-
        # evidence.json self_sha256 c682dbcc73ea544af04d4892cbeea91af08dc8f7
        # f35f1161fe2485f9c53a04fe; continuous + midpoint-restart 6 h T255
        # campaigns on the target 5090, cupy 14.2.0 / driver 13030) measured
        # a five-scheme stack this registration no longer runs.  The
        # registry admits only the fixed contract keys, so that digest
        # lives in the limitations text as the superseded record, not in
        # device_evidence_sha256 as cover for the six-scheme stack; the
        # battery must be re-run with GF in the order before the field
        # carries a value again.  The earlier promotion found and fixed six
        # admission defects (97874305a, 89571245d, 7d7b528d0, 42c37383c,
        # 9c5d407ca, 6f88311d4); those fixes stand.
        "admission_status": "device-pending",
        "device_evidence_sha256": _ZERO_SHA,
        "limitations": [
            "no evidence receipt artifact exists; evidence_receipt_sha256 is the "
            "zero digest until the qualification battery emits one",
            "device qualification predates the Grell-Freitas component: the "
            "superseded evidence c682dbcc73ea544af04d4892cbeea91af08dc8f7f35f11"
            "61fe2485f9c53a04fe covers rrtmgp/sfclay/noah/ysu/morrison only and "
            "the battery must be re-run with cumulus='gf' in the order",
            "the cumulus advective forcing lanes (RTHFTEN/RQVFTEN) are the "
            "dynamics' theta and vapor tendencies over the previous dynamics "
            "interval, measured by the runtime between its calls, not the "
            "current step's RK stage-1 tendencies WRF forms",
            "Grell-Freitas convective momentum tendencies are not coupled "
            "(the CumulusResult carries none; MPAS-A v8.4.1 does not couple "
            "them either); cumulus='own' couples its own",
            "Grell-Freitas dx is the scalar dx_m option, not the per-column "
            "Gaussian spacing (woof.globe.core.gf accepts a per-column feed); "
            "New Tiedtke (cumulus='ntiedtke') is fed each column's "
            "sqrt(dx*dy) on the Gaussian grid and couples its momentum "
            "tendencies, and has no scored forecast comparison yet",
            "sea ice and snow are seeded from the analysis (arwen_global."
            "surface_seeding: ICEC, ICETK, WEASD and SNOD, units verified by "
            "value, a missing field refused by name); the columns Noah skips "
            "as sea ice or land ice run a four-node heat conduction column "
            "in their soil-temperature layers (physics/frozen_surface.py: "
            "the skin node under held radiation and the surface layer's "
            "fluxes, each node the snow or ice at its depth, conduction to "
            "the freezing point of sea water at the analysed ice bottom or "
            "to the deep-soil climatology at 8 m through firn, implicit, "
            "capped at melting; the column receives the fluxes of its own "
            "skin, the cell's corrected by the surface layer's exchange "
            "coefficient and implicit in the new skin, so on a partial pack "
            "what the open water exchanged with the air stays with the "
            "water held at the freezing point; a partial pack presents the "
            "fraction-weighted blend of the ice skin and open water at the "
            "freezing point to the air and WRF's fractional blend of the "
            "sea-ice albedo 0.65 and emissivity 0.98 with open water 0.08 "
            "and 0.98 to the radiation; the surface layer runs a partial pack "
            "as two tiles, the ice at its own skin and the leads as WRF's "
            "water surface at the freezing point with their own inout state, "
            "and the atmosphere receives the area composite of their fluxes; "
            "the pack carries the statics convention row's sea-ice roughness, "
            "1 cm, in place of LANDUSE.TBL's 0.1 cm, and land ice keeps the "
            "table's) in place of WRF's "
            "seaice_noah and SFLX_GLACIAL, which are not ported",
            "statics.source='synthetic' (the analytic planet's default, and "
            "explicit in smoke/test configs) drives Noah, sfclay and the "
            "radiation with the pre-2026-09-01 constant planet -- one "
            "vegetation class 7 and one soil class 8 on land, LAI 3.0, snow "
            "albedo 0.6, deep-soil temperature equal to the bottom soil "
            "layer, albedo and roughness from the land fraction (audit "
            "2026-09-01 NB-8) -- printed at the run door and recorded in "
            "the receipt; a real-data run defaults to the WPS_GEOG statics "
            "and NB-8 no longer applies to it",
            "no NSSL-2/Thompson/MYNN adapter in this registration",
            "native Strang half-step scheduling is not WRF RK3 held-tendency scheduling",
        ],
    }


def ensure_builtin_global_physics_adapters() -> None:
    if ADAPTER_NAME in registered_global_physics_adapters():
        return
    register_global_physics_adapter(
        ADAPTER_NAME,
        lambda options: ArwenCudaColumnSuite(options),
        _contract(),
        # The suite is BUILT from every normalized option; only the hash
        # and the receipt carry the identity (woof.core.global_physics_
        # registry.GlobalPhysicsAdapterRegistration.options_identity).
        options_validator=lambda raw: NativePhysicsOptions.from_mapping(raw).normalized,
        options_identity=lambda normalized: NativePhysicsOptions.from_mapping(normalized).identity,
    )


__all__ = ["ADAPTER_NAME", "ensure_builtin_global_physics_adapters"]
