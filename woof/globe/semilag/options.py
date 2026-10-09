"""The ``[semilag]`` table: what the semi-Lagrangian core is allowed to do.

Every field here changes arithmetic, so every field joins the config
identity when ``integrator = "sl_si"`` and none of them exists in the
identity of any other integrator (config.ArwenGlobalConfig.config_identity).
A table present under another integrator is refused by name rather than
dropped: a flag parsed and silently ignored produces the same run whether
it is set or not, and the operator has no way to learn which happened.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .tracers import RETIRED_TRACER_FIXERS, TRACER_FIXERS

#: Horizontal interpolation of the DYNAMICAL bundle (the three Cartesian
#: wind components, theta' and ln ps).  ``cubic_lagrange`` is the tricubic
#: gather every kernel gate of record was measured on; ``quintic_lagrange``
#: reads the same bundle through six-point Lagrange weights in both
#: horizontal directions (interpolate.ORDERS), because an interpolation is
#: a filter applied once per step whatever the step is and the cubic one
#: at 288 passes a day kept 17 percent of the Eulerian core's 500 hPa
#: vorticity power at total wavenumbers 181 to 230 (MEASURED 2026-09-06,
#: T255, dt = 300 s).  Vapour and the ten grid tracers keep the cubic
#: gather under both names: they are the fields the quasi-monotone limiter
#: bounds, and the limiter's cell is the same inner cell either way.
HORIZONTAL_INTERPOLATIONS = ("cubic_lagrange", "quintic_lagrange")

#: The gather order (interpolate.ORDERS) each name selects.
HORIZONTAL_ORDER = {"cubic_lagrange": 4, "quintic_lagrange": 6}
VERTICAL_INTERPOLATIONS = ("cubic",)
EXTRAPOLATIONS = ("settls", "none")
TRACER_SCHEMES = ("semi_lagrangian", "flux_form")

#: Where the FIRST physics half's increment is applied.  The physics is
#: evaluated at grid points under every one of them, because a column
#: suite has nowhere else to run; what the option varies is where the
#: increment it produced is added, and that is the only axis these three
#: names sit on.
#:
#: ``advected``
#:     The increment is added to the state before the gather, so a parcel
#:     reads it at its DEPARTURE point along with everything else.  This
#:     is the plain Strang split and it costs nothing: the increment is
#:     never separated from the state at all.
#: ``arrival``
#:     The increment is taken back out of the departure-side bundle and
#:     added at the ARRIVAL point instead, so a freshly created field is
#:     never interpolated on the step that created it.  It costs one
#:     synthesis of the increment and one more gathered field per advected
#:     variable.
#: ``trajectory_average``
#:     Half of each, which is the trapezoidal average of the same
#:     time-level tendency field over the parcel's two endpoints.  Same
#:     cost as ``arrival``.
PHYSICS_COUPLINGS = ("advected", "arrival", "trajectory_average")

#: How much of the first physics half's increment moves from the
#: departure point to the arrival point, by coupling.
PHYSICS_ARRIVAL_WEIGHT = {
    "advected": 0.0, "arrival": 1.0, "trajectory_average": 0.5,
}


@dataclass(frozen=True)
class SemiLagrangianOptions:
    """Parsed ``[semilag]`` options, defaults included."""

    #: The six-point gather is the default (2026-09-07, the coupling lane's
    #: dry ladder, config.SEMILAG_DIFFUSION_ORDER): the cubic gather at 288
    #: passes a day kept 0.47 of the Eulerian core's 500 hPa vorticity
    #: power at n = 181-230 on a dry T255 day and the quintic one 0.77 at
    #: the same drain; "cubic_lagrange" stays selectable by name.
    horizontal_interpolation: str = "quintic_lagrange"
    vertical_interpolation: str = "cubic"
    #: Bound the interpolated value at the departure point, for the
    #: POSITIVE-DEFINITE species: vapour and the ten grid tracers.  Cloud
    #: water has no business exceeding the cloud water around it, and the
    #: limiter is what keeps a cloud edge from ringing.  ``False`` turns it
    #: off entirely, which is the counter-arm and not a shipping choice:
    #: MEASURED 2026-09-06, a six-hour T255 native arm with the limiter off
    #: and the mass fixer off refused at its FIRST step, on a cloud water
    #: of -1.49e-5 against a maximum of 1.02e-3.
    quasi_monotone: bool = True
    #: The same clip on the dynamical bundle (wind, ``theta'``, ``ln ps``).
    #: Off by default and named so the arm can be measured: the bundle is
    #: a state PLUS a tendency, so its neighbours' values are not its
    #: physical bounds, and clipping a wind to its neighbours takes the
    #: extremes of a field whose extremes are the flow, every step, with
    #: no conservation and no measurement of what was taken.
    quasi_monotone_dynamics: bool = False
    trajectory_iterations: int = 3
    trajectory_convergence_cells: float = 0.01
    extrapolation: str = "settls"
    tracer_scheme: str = "semi_lagrangian"
    tracer_fixer: str = "bermejo_conde"
    physics_coupling: str = "advected"
    #: Fields per gather launch.  One index and weight computation is
    #: shared by a launch's whole batch, and the batch's transient is one
    #: stacked copy of it, so this trades the weight arithmetic against
    #: device memory: eight fields is 378 MiB of transient at T255 and
    #: 1.61 GiB at T533 in float32.
    gather_batch: int = 8
    #: How many suite calls each Strang half is paid in, at ``dt / (2 n)``
    #: each.  One is the shipped arithmetic and the only value that costs
    #: nothing.  The others exist as the INSTRUMENT that separates the
    #: physics suite's own time-step sensitivity from the dynamics at a
    #: fixed dynamical step: five puts every scheme at the 30 s call the
    #: Eulerian core makes at dt = 60 s while the trajectory still takes
    #: 300 s, so a difference between that arm and the shipped one is the
    #: suite's dt and nothing else.  The land and radiation buckets keep
    #: their own intervals across the sub-calls.
    physics_substeps: int = 1

    def __post_init__(self) -> None:
        retired = RETIRED_TRACER_FIXERS.get(self.tracer_fixer)
        if retired is not None:
            raise ValueError(
                f"semilag.tracer_fixer = {self.tracer_fixer!r} is retired: "
                + retired
            )
        for name, allowed in (
            ("horizontal_interpolation", HORIZONTAL_INTERPOLATIONS),
            ("vertical_interpolation", VERTICAL_INTERPOLATIONS),
            ("extrapolation", EXTRAPOLATIONS),
            ("tracer_scheme", TRACER_SCHEMES),
            ("tracer_fixer", TRACER_FIXERS),
            ("physics_coupling", PHYSICS_COUPLINGS),
        ):
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(
                    f"semilag.{name} must be one of "
                    + ", ".join(repr(item) for item in allowed)
                    + f", got {value!r}"
                )
        for name in ("quasi_monotone", "quasi_monotone_dynamics"):
            value = getattr(self, name)
            if isinstance(value, str) or not isinstance(value, (bool, int)):
                raise ValueError(f"semilag.{name} must be a boolean")
        iterations = int(self.trajectory_iterations)
        if iterations < 1 or iterations > 8:
            raise ValueError(
                "semilag.trajectory_iterations must lie in 1..8; zero ships "
                "the explicit first guess, which is first order in dt and is "
                "not a trajectory, and past eight the search has long since "
                "reached the roundoff floor of the wind it reads"
            )
        limit = float(self.trajectory_convergence_cells)
        if not math.isfinite(limit) or not 0.0 < limit <= 1.0:
            raise ValueError(
                "semilag.trajectory_convergence_cells must lie in (0, 1]: it "
                "is a fraction of the local meridional grid length, and a "
                "limit of a whole cell admits a trajectory that never "
                "converged at all"
            )
        if int(self.gather_batch) < 1:
            raise ValueError("semilag.gather_batch must be >= 1")
        substeps = int(self.physics_substeps)
        if substeps < 1 or substeps > 16:
            raise ValueError(
                "semilag.physics_substeps must lie in 1..16: it is the number "
                "of suite calls each Strang half is paid in, one is the "
                "shipped arithmetic, and past sixteen a 300 s step is calling "
                "the suite at under 10 s, which no scheme in it was priced for"
            )

    @property
    def identity(self) -> dict[str, object]:
        payload = asdict(self)
        payload["quasi_monotone"] = bool(self.quasi_monotone)
        payload["quasi_monotone_dynamics"] = bool(self.quasi_monotone_dynamics)
        payload["trajectory_iterations"] = int(self.trajectory_iterations)
        payload["trajectory_convergence_cells"] = float(
            self.trajectory_convergence_cells
        )
        payload["physics_substeps"] = int(self.physics_substeps)
        # The batch size is a memory/launch trade and changes no bits: one
        # index and weight computation per launch produces the same value
        # at every point whatever the launch holds.  Dropped from the
        # identity so a T533 run that needs a smaller transient shares its
        # checkpoint lineage with the T255 run it was sized from.
        del payload["gather_batch"]
        return payload


def semilag_options_from_table(table: dict) -> SemiLagrangianOptions:
    """Parse a ``[semilag]`` TOML table, refusing unknown keys by name."""
    allowed = set(SemiLagrangianOptions.__dataclass_fields__)
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"unknown keys in [semilag]: {', '.join(unknown)}")
    kwargs: dict[str, object] = {}
    for name in allowed:
        if name not in table:
            continue
        value = table[name]
        field = SemiLagrangianOptions.__dataclass_fields__[name]
        if field.type in ("str", str):
            kwargs[name] = str(value).lower()
        elif field.type in ("bool", bool):
            if not isinstance(value, bool):
                raise ValueError(f"semilag.{name} must be a boolean")
            kwargs[name] = value
        elif field.type in ("int", int):
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"semilag.{name} must be an integer")
            kwargs[name] = int(value)
        else:
            if isinstance(value, bool):
                raise ValueError(f"semilag.{name} must be a finite number")
            kwargs[name] = float(value)
    return SemiLagrangianOptions(**kwargs)


__all__ = [
    "EXTRAPOLATIONS",
    "RETIRED_TRACER_FIXERS",
    "PHYSICS_ARRIVAL_WEIGHT",
    "HORIZONTAL_INTERPOLATIONS",
    "HORIZONTAL_ORDER",
    "PHYSICS_COUPLINGS",
    "SemiLagrangianOptions",
    "TRACER_FIXERS",
    "TRACER_SCHEMES",
    "VERTICAL_INTERPOLATIONS",
    "semilag_options_from_table",
]
