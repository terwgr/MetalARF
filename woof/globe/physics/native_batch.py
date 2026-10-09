"""Vertical-order, precision and humidity-convention boundary for existing Arwen CUDA physics.

Humidity convention.  The global model carries every water field as
SPECIFIC humidity (kg per kg of moist air): the analysis init stores GDAS
SPFH (GRIB2 0/1/0) unconverted and the dycore's water ledger prices
column water as sum(q dp)/g.  The WRF-lineage kernels consume DRY MIXING
RATIO (kg per kg of dry air): noah.cu:982 forms q2k = qv1/(1+qv1), and
sfclay.cu:255-257 / morrison.cu:950 compare qv straight against
EP2*e/(p-e).  Handing q to them as if it were r read every saturation test
dry by q_v^2 -- 2.0% at q = 0.02, 1.0% at 0.01, a dry bias across the
tropical boundary layer (audit 2026-09-01 NB-3).  This module converts at
the bridge boundary, r_x = q_x / (1 - q_v) inbound for every water species
and q_x = r_x / (1 + r_v) outbound, so the kernels see mixing ratio while
the model keeps its convention.  Number concentrations stay per kg of
moist air: the dycore rebuilds an untouched moment as raw + tendency so it
round-trips exactly, and a per-kg-of-dry-air rescale would turn that exact
round-trip into a one-ulp walk per call for a 2% shift in the implied mean
particle mass.  Stated divergence, not converted.

Water ledger.  ``atmospheric_water_kg_m2`` prices the batch's column water
in the MODEL's metric (specific humidity times dp/g) from the kernel-side
mixing ratios, so the suite's closure and the runtime's reservoir bookings
compare like with like on both sides of every kernel call.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..constants import (
    DRY_AIR_GAS_CONSTANT,
    GRAVITY_M_S2,
    NUMBER_MOMENTS,
    WATER_SPECIES,
)
from ..spectral.backend import device_cache_key
from ..spill import prefetch, resident, spilled
from ..state import PhysicsState, SURFACE_ARRAY_NAMES, SurfaceState, _copy_json
from ..vertical import HybridCoordinate
from .exchange import PhysicsExchange


#: The fused column-water kernel, one entry per (module, DEVICE).  See
#: :func:`~woof.globe.spectral.backend.device_cache_key`; gate CARD-1.
_COLUMN_WATER_KERNEL: dict[tuple, object] = {}


def staged_surface(xp, surface):
    """The surface reservoirs on the card, whatever holds them.

    ``SurfaceState.copy`` is the batch's own private copy of the caller's
    surface; when the pinned host tier holds those planes, the copy IS
    the stage, one plane at a time, and the tier keeps the original.
    """
    if not any(
        spilled(getattr(surface, member)) for member in SURFACE_ARRAY_NAMES
    ):
        return surface.copy()
    def _one(member):
        value = getattr(surface, member)
        # A parked plane's stage IS the private copy; one already on the
        # card is copied as ``SurfaceState.copy`` copies it.
        return resident(xp, value) if spilled(value) else value.copy()

    return SurfaceState(**{
        member: _one(member) for member in SURFACE_ARRAY_NAMES
    })


def staged_physics_state(xp, physics_state):
    """The physics namespace on the card, whatever holds it.

    The same contract as :func:`staged_surface`: a suite that copies the
    caller's namespace before working on it gets the stage as its copy.
    """
    if not any(spilled(value) for value in physics_state.arrays.values()):
        return physics_state.copy()
    return PhysicsState(
        schema=physics_state.schema,
        arrays={
            name: (resident(xp, value) if spilled(value) else value.copy())
            for name, value in physics_state.arrays.items()
        },
        metadata=_copy_json(physics_state.metadata),
    )


def _column_water_kernel(xp):
    """``(sum_x r_x / (1 + r_v)) dp / g`` per cell in float64, fused.

    Specification: the numpy branch of
    :meth:`NativeColumnBatch.atmospheric_water_kg_m2`.  ``-fmad=false``
    keeps each multiply and add rounded separately, as the ufunc chain
    rounds them; the species order is ``WATER_SPECIES``.
    """
    key = device_cache_key(xp)
    kernel = _COLUMN_WATER_KERNEL.get(key)
    if kernel is None:
        kernel = xp.ElementwiseKernel(
            "float32 qv, float32 qc, float32 qr, float32 qi, float32 qs, "
            "float32 qg, float32 dp, float64 gravity",
            "float64 out",
            """
            const double moist = 1.0 + (double)qv;
            double total = 0.0;
            total = total + (double)qv / moist;
            total = total + (double)qc / moist;
            total = total + (double)qr / moist;
            total = total + (double)qi / moist;
            total = total + (double)qs / moist;
            total = total + (double)qg / moist;
            out = total * (double)dp / gravity;
            """,
            "arwen_native_column_water",
            options=("-fmad=false",),
        )
        _COLUMN_WATER_KERNEL[key] = kernel
    return kernel


def mixing_ratio_from_specific_humidity(species):
    """q -> r = q / (1 - q_v) for every per-kg water field in ``species``.

    ``species`` maps field names to arrays (or scalars) sharing one
    convention and must carry ``"qv"``.  Arithmetic stays in the inputs'
    dtype and array module.
    """
    dry = 1.0 - species["qv"]
    return {name: value / dry for name, value in species.items()}


def specific_humidity_from_mixing_ratio(species):
    """r -> q = r / (1 + r_v), the exact inverse of the conversion above."""
    moist = 1.0 + species["qv"]
    return {name: value / moist for name, value in species.items()}


@dataclass
class NativeColumnBatch:
    """Complete copy of one global exchange in native bottom-to-top FP32 order.

    Water species are held as dry mixing ratios (see the module docstring);
    everything else is the exchange's own convention.
    """

    xp: object
    arrays: dict[str, object]
    surface: object
    physics_state: object
    time_s: float
    dt_s: float
    #: The globe's rows this batch is (PhysicsExchange.band), or None for
    #: the whole grid; the runtime keys its per-grid state on it.
    band: tuple[int, int] | None = None
    #: The model-top pressure the exchange read over the whole grid
    #: (PhysicsExchange.model_top_pa), or None to take this batch's own.
    model_top_pa: float | None = None

    @classmethod
    def from_exchange(cls, exchange: PhysicsExchange, xp) -> "NativeColumnBatch":
        exchange.validate()

        # The exchange may carry arrays the pinned host tier holds rather
        # than the card (spill.SpilledArray): the grid tracers and the
        # surface reservoirs.  ``resident`` stages one, and the batch's
        # own copy below IS the device copy the caller would otherwise
        # have held for the whole run -- which is what makes the tier a
        # capacity mechanism rather than a second copy.  The staged array
        # dies with the expression that consumes it.
        def volume(value):
            staged = resident(xp, value)
            out = xp.ascontiguousarray(
                xp.asarray(staged, dtype=xp.float32)[::-1]
            )
            del staged
            return out

        def level(value):
            staged = resident(xp, value)
            out = xp.ascontiguousarray(
                xp.asarray(staged, dtype=xp.float32)[::-1]
            )
            del staged
            return out

        def plane(value):
            staged = resident(xp, value)
            out = xp.ascontiguousarray(xp.asarray(staged, dtype=xp.float32))
            del staged
            return out

        # No geopotential volume: the runtime never reads one (the cumulus
        # scheme's height coordinate is the terrain height built below and
        # the layer depths come from _dz), so at T533 float32 it was 0.19
        # GiB copied and held through every physics call for nothing.
        arrays = {
            "latitude_deg": plane(exchange.latitude_deg),
            "longitude_deg": plane(exchange.longitude_deg),
            "p_half": level(exchange.p_half),
            "p_full": volume(exchange.p_full),
            "dp": volume(exchange.dp),
            "exner": volume(exchange.exner),
            "temperature": volume(exchange.temperature),
            "theta": volume(exchange.theta),
            "virtual_temperature": volume(exchange.virtual_temperature),
            "u": volume(exchange.u),
            "v": volume(exchange.v),
        }
        # One species live at a time through the r = q / (1 - q_v)
        # conversion (mixing_ratio_from_specific_humidity, applied per
        # species with the same shared divisor): the six-volume specific
        # stack this replaced was a 1.15 GiB transient at T533 float32.
        qv = volume(exchange.qv)
        dry = 1.0 - qv
        # The tracers arrive in a known order, so each one's copy is
        # started while the one before it is being converted: the double
        # buffer, at the one loop that knows what it reads next.
        order = (*WATER_SPECIES, *NUMBER_MOMENTS)
        prefetch(getattr(exchange, order[1]))
        for index, name in enumerate(WATER_SPECIES):
            if index + 1 < len(order):
                prefetch(getattr(exchange, order[index + 1]))
            value = qv if name == "qv" else volume(getattr(exchange, name))
            arrays[name] = xp.ascontiguousarray(value / dry)
            del value
        del qv, dry
        for index, name in enumerate(NUMBER_MOMENTS):
            step = len(WATER_SPECIES) + index + 1
            if step < len(order):
                prefetch(getattr(exchange, order[step]))
            arrays[name] = volume(getattr(exchange, name))
        # Terrain height for the cumulus scheme's height coordinate
        # (gf.cu builds zo from ht upward and reads zo[kpbl] as WRF does).
        # The exchange carries full-level geopotential built hydrostatically
        # from the surface (dynamics._hydrostatic_geopotential), so the
        # surface value is the bottom layer's inverted with the same
        # half-layer integral (HybridCoordinate.half_layer_thickness, Tv
        # linear in ln p; DN-3): phi_s = phi_full - that thickness.  Taken
        # in the exchange's own precision before the float32 cast.
        phi_bottom = xp.asarray(resident(xp, exchange.geopotential))[-1]
        tv = xp.asarray(resident(xp, exchange.virtual_temperature))
        p_half = xp.asarray(resident(xp, exchange.p_half))
        p_full = xp.asarray(resident(xp, exchange.p_full))
        half = HybridCoordinate.half_layer_thickness(
            tv, p_full, xp.log(p_half[1:] / p_half[:-1]), xp
        )
        arrays["terrain_height_m"] = plane((phi_bottom - half[-1]) / GRAVITY_M_S2)
        if exchange.omega_half_pa_s is not None:
            arrays["omega_half"] = level(exchange.omega_half_pa_s)
        # The physics namespace is carried by reference in a private dict:
        # nothing on the batch writes into it (the persistent state the
        # runtime works on copies every array it touches first, see
        # native_state._array), so a deep copy here was one whole
        # namespace (1.5 GiB at T533 with 40 levels) held per call for
        # nothing.  The caller's arrays stay untouched, which the bridge's
        # fingerprint check proves after every call.
        return cls(
            xp=xp,
            arrays=arrays,
            surface=staged_surface(xp, exchange.surface),
            physics_state=PhysicsState(
                schema=exchange.physics_state.schema,
                arrays=dict(exchange.physics_state.arrays),
                metadata=_copy_json(exchange.physics_state.metadata),
            ),
            time_s=float(exchange.time_s),
            dt_s=float(exchange.dt_s),
            band=(None if exchange.band is None
                  else (int(exchange.band[0]), int(exchange.band[1]))),
            model_top_pa=(None if exchange.model_top_pa is None
                          else float(exchange.model_top_pa)),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.arrays["theta"].shape)

    @property
    def surface_shape(self) -> tuple[int, int]:
        return self.shape[1:]

    def to_global(self, value):
        return self.xp.ascontiguousarray(value[::-1])

    def specific_humidities(self, dtype=None) -> dict[str, object]:
        """The batch's water species in the model's convention (unreversed)."""
        xp = self.xp
        dtype = xp.float32 if dtype is None else dtype
        return specific_humidity_from_mixing_ratio(
            {name: xp.asarray(self.arrays[name], dtype=dtype) for name in WATER_SPECIES}
        )

    def atmospheric_water_kg_m2(self):
        """Column water in the model's metric, sum_x q_x dp / g, in float64.

        Every reservoir booking and the suite's closure read this, so a
        kernel-side change measured here is exactly what the model's own
        ledger (water.py atmospheric_water_column on the returned specific
        humidities) will see, up to the float32 cast of the return trip.
        """
        xp = self.xp
        if hasattr(xp, "ElementwiseKernel"):
            # One fused pass on the device: the same float64 chain as the
            # numpy specification below (1 + r_v; 0 + r_qv / moist, + r_qc
            # / moist, ... in WATER_SPECIES order; times dp, over g) with
            # multiply-add contraction off, so every rounding sits where
            # the specification rounds.  The species reduction over levels
            # stays the backend's own sum.  Why: the specification streams
            # six float32 volumes through six float64 temporaries per
            # call and the runtime calls this eight times per physics
            # call; the fused pass reads the seven inputs once.
            out = xp.empty(self.arrays["dp"].shape, dtype=xp.float64)
            _column_water_kernel(xp)(
                self.arrays["qv"], self.arrays["qc"], self.arrays["qr"],
                self.arrays["qi"], self.arrays["qs"], self.arrays["qg"],
                self.arrays["dp"], float(GRAVITY_M_S2), out,
            )
            return out.sum(axis=0)
        # The species sum is accumulated one float64 species at a time in
        # the order and association ``sum()`` over the specific-humidity
        # dict used (0 + q_qv, + q_qc, ...; each q_x = r_x / (1 + r_v) in
        # float64), so the bits are the same while at most one converted
        # species is live instead of six (2.3 GiB at T533 with 40 levels,
        # on a function the runtime calls seven times per physics call).
        # This numpy chain is the specification of the fused kernel above.
        moist = 1.0 + xp.asarray(self.arrays["qv"], dtype=xp.float64)
        total = 0
        for name in WATER_SPECIES:
            total = total + xp.asarray(self.arrays[name], dtype=xp.float64) / moist
        del moist
        dp = xp.asarray(self.arrays["dp"], dtype=xp.float64)
        return (total * dp / GRAVITY_M_S2).sum(axis=0)

    def global_prognostics(self, *, consume: bool = False) -> dict[str, object]:
        """The batch's prognostics back in the model's order and convention.

        ``consume=True`` releases each batch array as its global copy is
        made, for a caller that is done with the batch: the fourteen
        copies otherwise stood beside the fourteen originals (2.67 GiB
        each at T533 float32) at the end of every physics call.  Same
        values either way: q_x = r_x / (1 + r_v) per species with the
        shared divisor formed first, then the level flip.
        """
        xp = self.xp
        arrays = self.arrays
        moist = 1.0 + xp.asarray(arrays["qv"], dtype=xp.float32)

        def take(name):
            return arrays.pop(name) if consume else arrays[name]

        out = {}
        for name in ("u", "v", "theta"):
            out[name] = self.to_global(take(name))
        for name in WATER_SPECIES:
            out[name] = self.to_global(
                xp.asarray(take(name), dtype=xp.float32) / moist
            )
        for name in NUMBER_MOMENTS:
            out[name] = self.to_global(take(name))
        return out

    def validate(self) -> None:
        shape = self.shape
        for name in ("u", "v", "theta", "p_full", "dp", "exner",
                     "temperature", *WATER_SPECIES,
                     *NUMBER_MOMENTS):
            value = self.arrays[name]
            if tuple(value.shape) != shape:
                raise ValueError(f"native batch {name} shape mismatch")
            if value.dtype != self.xp.float32 or not value.flags.c_contiguous:
                raise TypeError(f"native batch {name} must be contiguous float32")
        if tuple(self.arrays["p_half"].shape) != (shape[0] + 1, *shape[1:]):
            raise ValueError("native batch p_half shape mismatch")
        omega = self.arrays.get("omega_half")
        if omega is not None and tuple(omega.shape) != (shape[0] + 1, *shape[1:]):
            raise ValueError("native batch omega_half shape mismatch")


__all__ = [
    "NativeColumnBatch",
    "staged_physics_state",
    "staged_surface",
    "mixing_ratio_from_specific_humidity",
    "specific_humidity_from_mixing_ratio",
]
