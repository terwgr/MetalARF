"""Semi-Lagrangian departure points and interpolation for WOOF global.

Everything the two-time-level semi-Lagrangian semi-implicit integrator needs
in order to answer "where did this arrival point come from, and what was the
field there" lives under this package, and nothing here reaches into the
integrator.  There is no dispatch seam in this module and no time integrator:
:mod:`woof.globe.dynamics` is untouched by the trajectory lane, so the
IMEX path is byte-identical by construction rather than by measurement.

The reading order:

*   :mod:`.tables` builds the grid geometry once: the pole-extended latitude
    table, the reciprocal Lagrange denominators of the true Gauss-Legendre
    nodes, and the bracket lookup.
*   :mod:`.trajectory` solves for the departure points in geocentric
    Cartesian coordinates and carries the two diagnostics that gate them,
    the Lipschitz number and the fixed-point convergence.
*   :mod:`.interpolate` reads any number of fields at those points through
    one shared index and weight computation, with the quasi-monotone clip as
    a compiled variant rather than a runtime flag.
*   :mod:`.cases` drives both against prescribed flows with an exact answer.
*   :mod:`.rhs` writes the advective-form tendencies and puts the
    semi-implicit linear operator where the trajectory needs it.
*   :mod:`.vectors` carries a vector from its departure point to its
    arrival point without shortening it.
*   :mod:`.tracers` rides the ten grid tracers on the same stencil and
    restores their mass.
*   :mod:`.state` is the second time level; :mod:`.options` the table
    that configures the core; :mod:`.step` the step itself.
"""
from __future__ import annotations

from .cases import (
    run_deformational_transport,
    CaseRun,
    TransportNorms,
    area_weights,
    deformational_wind,
    run_deformational,
    run_solid_body,
    solid_body_departure,
    solid_body_wind,
    transport_norms,
)
from .interpolate import Stencil, gather, gather_batch, zero_stencil
from .options import SemiLagrangianOptions, semilag_options_from_table
from .pins import SEMILAG_PIN_OVERRIDES, SEMILAG_SEMI_IMPLICIT_PINS
from .rhs import (
    ReferenceProfile,
    advective_tendencies,
    linear_grid_tendencies,
    reference_theta_faces,
)
from .state import TRAJECTORY_FIELDS, TrajectoryState
from .step import SEMILAG_INTEGRATORS, grid_tables, reference_profile, semilag_step
from .tables import SphericalGridTables
from .tracers import DEFICIT_FIXERS, TRACER_FIXERS, area_weights as tracer_area_weights, fix_mass
from .trajectory import (
    CartesianWind,
    LipschitzDiagnostics,
    TrajectoryDiagnostics,
    cartesian_wind,
    convergence,
    departure_points,
    level_rate_from_mass_flux,
    lipschitz,
    local_wind,
    refuse_beyond_lipschitz,
)

__all__ = [
    "SEMILAG_INTEGRATORS",
    "SEMILAG_PIN_OVERRIDES",
    "SEMILAG_SEMI_IMPLICIT_PINS",
    "TRAJECTORY_FIELDS",
    "CartesianWind",
    "CaseRun",
    "LipschitzDiagnostics",
    "SphericalGridTables",
    "Stencil",
    "TrajectoryDiagnostics",
    "ReferenceProfile",
    "SemiLagrangianOptions",
    "TrajectoryState",
    "TransportNorms",
    "advective_tendencies",
    "area_weights",
    "cartesian_wind",
    "convergence",
    "deformational_wind",
    "departure_points",
    "DEFICIT_FIXERS",
    "TRACER_FIXERS",
    "fix_mass",
    "tracer_area_weights",
    "gather",
    "gather_batch",
    "grid_tables",
    "level_rate_from_mass_flux",
    "linear_grid_tendencies",
    "lipschitz",
    "local_wind",
    "reference_profile",
    "reference_theta_faces",
    "refuse_beyond_lipschitz",
    "run_deformational",
    "run_deformational_transport",
    "run_solid_body",
    "solid_body_departure",
    "semilag_options_from_table",
    "semilag_step",
    "solid_body_wind",
    "transport_norms",
    "zero_stencil",
]
