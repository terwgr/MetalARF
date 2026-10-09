"""Canonical gridpoint exchange between the global dycore and column physics."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from ..constants import (
    NUMBER_MOMENTS,
    PHYSICS_EXCHANGE_SCHEMA,
    WATER_SPECIES,
)
from ..state import PhysicsState, SurfaceState


@dataclass(frozen=True)
class PhysicsExchange:
    schema: str
    time_s: float
    dt_s: float
    latitude_deg: object
    longitude_deg: object
    p_half: object
    p_full: object
    dp: object
    exner: object
    temperature: object
    theta: object
    virtual_temperature: object
    geopotential: object
    u: object
    v: object
    qv: object
    qc: object
    qr: object
    qi: object
    qs: object
    qg: object
    nc: object
    nr: object
    ni: object
    ns: object
    ng: object
    surface: SurfaceState
    physics_state: PhysicsState
    #: Hydrostatic pressure velocity dp/dt (Pa/s) on the nlev+1 half levels,
    #: in the model's top-to-bottom order, zero at the top and at the
    #: surface (the continuity closure's own convention).  Cumulus schemes
    #: read it: Grell-Freitas' Brown vertical-velocity closure members and
    #: its moisture-convergence term run on omega, so a column state
    #: without it is not a complete cumulus input.  The dycore's exchange
    #: supplies it; a hand-built exchange may leave it None, and the
    #: consumer that needs it refuses.
    omega_half_pa_s: object = None
    #: The latitude rows of the GLOBE this exchange covers, ``(first,
    #: last)`` with ``last`` exclusive, or None for the whole grid.  The
    #: physics suite runs a band at a time (dynamics.apply_physics): every
    #: array above is that band's rows, and a suite that holds per-grid
    #: state (the radiation's solar geometry, the frozen-column masks, the
    #: cumulus grid spacing) keys it on this.  A hand-built whole-grid
    #: exchange leaves it None and nothing changes for it.
    band: tuple[int, int] | None = None
    #: The model-top pressure the radiation's above-model column is built
    #: on, read ONCE over the whole grid by the caller (the mean of the top
    #: half-level plane, in the batch's float32) so every band's radiation
    #: reads the same number the resident call read.  A band's own mean of
    #: that plane is a different float for a different band count; None
    #: (a hand-built exchange) lets the suite take its own mean.
    model_top_pa: float | None = None

    @classmethod
    def create(cls, **kwargs) -> "PhysicsExchange":
        kwargs.setdefault("schema", PHYSICS_EXCHANGE_SCHEMA)
        kwargs.setdefault("physics_state", PhysicsState())
        value = cls(**kwargs)
        value.validate()
        return value

    @property
    def nlev(self) -> int:
        return int(self.theta.shape[0])

    @property
    def grid_shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.theta.shape[-2:])

    def water(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in WATER_SPECIES}

    def moments(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in NUMBER_MOMENTS}

    def prognostics(self) -> dict[str, object]:
        return {
            "u": self.u,
            "v": self.v,
            "theta": self.theta,
            **self.water(),
            **self.moments(),
        }

    def validate(self) -> None:
        if self.schema != PHYSICS_EXCHANGE_SCHEMA:
            raise ValueError(f"physics exchange schema mismatch: {self.schema!r}")
        if not math.isfinite(float(self.time_s)) or float(self.time_s) < 0.0:
            raise ValueError("physics exchange time_s must be finite and nonnegative")
        if not math.isfinite(float(self.dt_s)) or float(self.dt_s) <= 0.0:
            raise ValueError("physics exchange dt_s must be finite and positive")
        shape = tuple(self.theta.shape)
        if len(shape) != 3 or shape[0] < 2:
            raise ValueError("physics exchange theta must be (nlev,nlat,nlon), nlev>=2")
        for name in (
            "p_full", "dp", "exner", "temperature", "virtual_temperature",
            "geopotential", "u", "v", *WATER_SPECIES, *NUMBER_MOMENTS,
        ):
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"physics exchange {name} shape {value.shape} != {shape}")
        if tuple(self.p_half.shape) != (shape[0] + 1, *shape[1:]):
            raise ValueError("physics exchange p_half has the wrong shape")
        if self.omega_half_pa_s is not None and tuple(
            self.omega_half_pa_s.shape
        ) != (shape[0] + 1, *shape[1:]):
            raise ValueError(
                "physics exchange omega_half_pa_s must sit on the nlev+1 half "
                f"levels {(shape[0] + 1, *shape[1:])}, got "
                f"{tuple(self.omega_half_pa_s.shape)}"
            )
        if (
            tuple(self.latitude_deg.shape) != shape[1:]
            or tuple(self.longitude_deg.shape) != shape[1:]
        ):
            raise ValueError("physics exchange latitude/longitude must match grid")
        if self.band is not None:
            first, last = (int(v) for v in self.band)
            if first < 0 or last <= first or last - first != shape[1]:
                raise ValueError(
                    f"physics exchange band {self.band} does not cover its "
                    f"{shape[1]} latitude rows: a band names the globe's rows "
                    "the exchange's arrays are, or the suite's per-grid "
                    "state is keyed on the wrong rows"
                )
        if self.model_top_pa is not None and not (
            math.isfinite(float(self.model_top_pa)) and float(self.model_top_pa) > 0.0
        ):
            raise ValueError("physics exchange model_top_pa must be finite and positive")
        for name, value in self.surface.arrays().items():
            if value.ndim == 2:
                expected = shape[1:]
            elif value.ndim == 3:
                expected = (value.shape[0], *shape[1:])
            else:
                raise ValueError(f"surface field {name} must be 2-D or 3-D")
            if tuple(value.shape) != expected:
                raise ValueError(f"surface field {name} has incompatible shape {value.shape}")
        self.physics_state.validate()


@dataclass
class PhysicsResult:
    u: object
    v: object
    theta: object
    qv: object
    qc: object
    qr: object
    qi: object
    qs: object
    qg: object
    nc: object
    nr: object
    ni: object
    ns: object
    ng: object
    surface: SurfaceState
    physics_state: PhysicsState
    diagnostics: dict[str, float]
    adapter_receipt: dict[str, object]
    #: Horizontal planes of this call's rows whose WHOLE-GRID reduction is
    #: a diagnostic (a grid mean, a path sum): the suite hands them back
    #: unreduced, the caller assembles the globe's plane out of the bands
    #: (bands.PlaneAccumulator) and the suite's ``finish`` reduces it once.
    #: A flat reduction folded band by band is a different number for a
    #: different band count; the assembled plane is the same plane whatever
    #: the schedule was.  Each value is ``(*lead, rows, nlon)``.
    planes: dict[str, object] = field(default_factory=dict)

    def water(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in WATER_SPECIES}

    def moments(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in NUMBER_MOMENTS}

    def prognostics(self) -> dict[str, object]:
        return {
            "u": self.u,
            "v": self.v,
            "theta": self.theta,
            **self.water(),
            **self.moments(),
        }


__all__ = ["PhysicsExchange", "PhysicsResult"]
