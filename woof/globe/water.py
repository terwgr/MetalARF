"""Single accounting authority for WOOF global water reservoirs."""
from __future__ import annotations

from .constants import GRAVITY_M_S2, LIQUID_WATER_DENSITY, WATER_SPECIES
from .spill import resident

SOIL_LAYER_THICKNESS_M = (0.10, 0.30, 0.60, 1.00)
# Native Noah fields in the Level-5 PhysicsState.  They are stores, unlike
# accumulated precipitation diagnostics, and therefore enter total water.
NATIVE_CANOPY_STORE_NAMES = ("noah_canwat", "canwat")
NATIVE_SNOW_STORE_NAMES = ("noah_snow", "snow")
# Noah's runoff accumulators are the water's LAST address inside the column
# system: SSTEP saturation excess and subsurface drainage leave the soil
# column into udrunoff, and infiltration excess leaves into sfcrunoff
# (noah.cu:548-564, 1518-1519), and nothing in the column ever reads them
# back -- physically that water has left for the rivers.  The v3 ledger
# counted them as held water at land_fraction pricing, which kept every
# per-call closure exact but sequestered a monotone-growing store inside
# the pinned conservation total: with total held water fixed to the
# cold-start target, every kg the runoff accumulators grew squeezed the
# atmosphere+reservoir global mean by the same kg, forever.  v4 books each
# priced runoff-store increment to the cumulative outflow account below at
# the moment the land step debits the reservoir for it; the runoff stores
# themselves (still monotone, still per unit LAND area -- the kernel is
# untouched) are counted minus outflow, i.e. at zero.  The driven-negative
# refusal stays on them: the outflow booking prices increments of
# max(store, 0) * land_fraction, so a store driven negative would underbook
# the next exit and launder the loss through the outflow account.
NATIVE_SURFACE_RUNOFF_STORE_NAMES = ("noah_sfcrunoff", "sfcrunoff")
NATIVE_SUBSURFACE_RUNOFF_STORE_NAMES = ("noah_udrunoff", "udrunoff")
# Cumulative booked exits (kg/m2 per column), maintained by the native land
# step.  Water conservation is held + outflow = constant: the run-level
# total_water_relative_drift gate and the global water fixer target both
# read total_water_column, which counts this account as a booked exit term.
WATER_OUTFLOW_NAMES = ("water_outflow_kg_m2",)


# Store content below zero is invisible to this accounting: the clamp counts
# it as zero, the global-mean total drops by only the store's prior positive
# content, and the global water fixer then manufactures that loss as real
# surface water everywhere. Refusing above roundoff scale keeps a defect that
# drives a store negative visible instead of laundered.
# -1e-9 on the dimensionless soil fraction is >1e6 times float64 roundoff on
# an O(0.1) fraction and books at most 2e-6 kg/m2 over the 2 m soil stack,
# inside the 1e-8 relative total-water gate on the ~1e3 kg/m2 column.
SOIL_FRACTION_MINIMUM = -1.0e-9
# Native stores are kg/m2; -1e-8 matches the surface-water reservoir refusal
# threshold in physics/reference.py.
NATIVE_STORE_MINIMUM_KG_M2 = -1.0e-8


def _refuse_driven_negative(name, value, minimum, xp):
    # The floors above are derived for float64, but the only configuration
    # that owns these stores ships them at float32 (native physics refuses
    # any other precision), where one representable rounding step of an
    # O(0.1) fraction is ~1.8e-8 - already past the float64 floor.  The
    # refusal therefore never fires inside 64 ulps of the store's own dtype
    # at its own magnitude; a driven-negative defect sits orders beyond.
    eps = float(xp.finfo(value.dtype).eps) if hasattr(value, "dtype") else 2.3e-16
    scale = max(1.0, float(xp.max(xp.abs(value))))
    floor = min(float(minimum), -64.0 * eps * scale)
    smallest = float(xp.min(value))
    if smallest < floor:
        raise FloatingPointError(
            f"water store {name} driven negative: min={smallest:.9g} is below "
            f"{floor:g}; below-zero content is invisible to the water ledger "
            "and the global fixer would manufacture the loss as surface water"
        )


def _staged_arrays(xp, arrays):
    """The ledger's few named physics-namespace planes, on the card.

    The tier holds the whole native namespace (2.3 GiB at T533); the
    water ledger reads three planes out of it, so only those three are
    staged and the rest never touches the card for this call.
    """
    from .spill import spilled

    if not any(spilled(value) for value in arrays.values()):
        return arrays
    wanted = (
        *NATIVE_CANOPY_STORE_NAMES, *NATIVE_SNOW_STORE_NAMES,
        *NATIVE_SURFACE_RUNOFF_STORE_NAMES,
        *NATIVE_SUBSURFACE_RUNOFF_STORE_NAMES, *WATER_OUTFLOW_NAMES,
    )
    return {
        name: (resident(xp, value) if name in wanted else value)
        for name, value in arrays.items()
    }


def atmospheric_water_column(grid_state):
    return (
        sum(grid_state[name] for name in WATER_SPECIES)
        * grid_state["dp"]
        / GRAVITY_M_S2
    )


def soil_water_column(surface, xp):
    # The soil stack and the land mask may be held by the pinned host
    # tier; ``resident`` stages them for this reduction and they are
    # dropped with the call.  Four soil planes and one mask, 0.024 GiB at
    # T533: the ledger reads them once a step.
    fraction = resident(xp, surface.soil_water_fraction)
    land = resident(xp, surface.land_fraction)
    _refuse_driven_negative(
        "soil_water_fraction", fraction, SOIL_FRACTION_MINIMUM, xp,
    )
    thickness = xp.asarray(
        SOIL_LAYER_THICKNESS_M,
        dtype=fraction.dtype,
    )[:, None, None]
    return xp.sum(
        xp.maximum(fraction, 0.0)
        * thickness
        * LIQUID_WATER_DENSITY
        * land[None],
        axis=0,
    )


def native_store_column(physics_state, like, xp):
    """Held native-store water: canopy and snow, at full weight.

    The runoff accumulators are validated but counted at zero: every priced
    increment they take is booked to the outflow account by the land step
    (native_runtime._land_step), so counting the stores again would count
    exited water twice.  See the account notes above WATER_OUTFLOW_NAMES.
    """
    total = xp.zeros_like(like)
    arrays = _staged_arrays(xp, getattr(physics_state, "arrays", {}))
    exited = (
        NATIVE_SURFACE_RUNOFF_STORE_NAMES,
        NATIVE_SUBSURFACE_RUNOFF_STORE_NAMES,
    )
    for choices in (
        NATIVE_CANOPY_STORE_NAMES,
        NATIVE_SNOW_STORE_NAMES,
        *exited,
    ):
        name, value = next(
            ((name, arrays[name]) for name in choices if name in arrays),
            (None, None),
        )
        if value is None:
            continue
        if tuple(value.shape) != tuple(like.shape):
            raise ValueError(
                f"native water store has shape {value.shape}, expected {like.shape}"
            )
        _refuse_driven_negative(name, value, NATIVE_STORE_MINIMUM_KG_M2, xp)
        if choices in exited:
            continue
        total += xp.maximum(value, 0.0)
    return total


def water_outflow_column(physics_state, like, xp):
    """Cumulative booked exits (kg/m2 per column); zero when never booked.

    Monotone nonnegative by construction (the land step adds priced
    increments of monotone accumulators), so a negative value is a booking
    defect and refuses like a driven-negative store: below-zero outflow
    would raise the conservation total and the global fixer would drain the
    difference out of real surface water everywhere.
    """
    arrays = _staged_arrays(xp, getattr(physics_state, "arrays", {}))
    name, value = next(
        ((name, arrays[name]) for name in WATER_OUTFLOW_NAMES if name in arrays),
        (None, None),
    )
    if value is None:
        return xp.zeros_like(like)
    if tuple(value.shape) != tuple(like.shape):
        raise ValueError(
            f"water outflow account has shape {value.shape}, expected {like.shape}"
        )
    _refuse_driven_negative(name, value, NATIVE_STORE_MINIMUM_KG_M2, xp)
    return value


def total_water_column(bundle, grid_state, xp, atmospheric=None):
    """The five water stores of a bundle, per column.

    ``atmospheric`` lets a caller that assembled the atmospheric column
    itself -- a band pipeline, which never holds the levelled water
    volume at full size -- hand it in; the plane is the same one the
    level sum below produces, so the total is the same expression in the
    same order either way.
    """
    if atmospheric is None:
        atmospheric = xp.sum(atmospheric_water_column(grid_state), axis=0)
    explicit_surface = resident(xp, bundle.surface.water_kg_m2)
    soil = soil_water_column(bundle.surface, xp)
    native = native_store_column(bundle.physics_state, explicit_surface, xp)
    outflow = water_outflow_column(bundle.physics_state, explicit_surface, xp)
    return {
        "atmospheric": atmospheric,
        "surface": explicit_surface,
        "soil": soil,
        "native": native,
        "outflow": outflow,
        # Conservation statement: held water plus booked exits.  The global
        # water fixer target and the total_water_relative_drift gate read
        # this total, so runoff leaving the column system is a booked exit,
        # not a loss for the fixer to manufacture back.
        "total": atmospheric + explicit_surface + soil + native + outflow,
    }


__all__ = [
    "SOIL_LAYER_THICKNESS_M",
    "WATER_OUTFLOW_NAMES",
    "atmospheric_water_column",
    "native_store_column",
    "soil_water_column",
    "total_water_column",
    "water_outflow_column",
]
