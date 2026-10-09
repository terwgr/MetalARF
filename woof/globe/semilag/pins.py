"""The arithmetic identity of the semi-Lagrangian core.

A pin document names what the numbers WERE produced by, so an archive
never resumes under different arithmetic while claiming to be its own
continuation.  Four things change under ``integrator = "sl_si"`` and all
four are named here rather than hidden inside one string:

* the gravity-wave arithmetic (one arrival-point Helmholtz solve per
  step instead of the IMEX pair's two stage solves),
* the momentum (Cartesian-component semi-Lagrangian advection with
  parallel transport, where the Eulerian core is vector-invariant),
* the scalar transport (a tricubic or quintic-horizontal gather at
  departure points for the WHOLE thermodynamic variable, vapour and every
  grid tracer, where the Eulerian core has a spectral flux form and a
  split flux-form sweep; v3 on 2026-09-06, when theta stopped being
  advected as a deviation from the reference profile with the reference's
  material tendency carried on the grid, because that tendency warmed the
  model lid by 32 K a day where the trajectory is clamped, see
  semilag.rhs),
* the checkpoint (seven trajectory arrays that no other integrator
  writes at all).

``pins.pin_document`` builds every document by deep-copying one template
and then assigning; assigning four keys instead of one moves nothing for
any other family, because the TEMPLATE is untouched and those families
still assign only the one.  That is the real shape of the hash-stability
constraint: it binds the template, not the number of keys a family sets.
"""
from __future__ import annotations

#: ``[semi_implicit] scheme`` -> pin, under the semi-Lagrangian family.
#: Only ``vertical_modes`` appears: the barotropic proxy leaves every
#: internal vertical mode explicit (its measured rest ceiling was 104.7 s
#: at T533 whatever its reference speed) and a 300 s step has no explicit
#: budget for a 350 m/s mode, so the config door refuses that pairing by
#: name rather than pinning arithmetic nothing may run.
SEMILAG_SEMI_IMPLICIT_PINS = {
    "vertical_modes": (
        "vertical-mode-two-time-level-semi-lagrangian-off-centred-settls-"
        "single-helmholtz-arrival-solve-v1"
    ),
}

#: The template keys the semi-Lagrangian family overrides.
SEMILAG_PIN_OVERRIDES = {
    "momentum": (
        "cartesian-component-semi-lagrangian-advection-with-great-circle-"
        "parallel-transport-vorticity-divergence-reanalysis-hybrid-pressure-"
        "midpoint-pressure-gradient-v1"
    ),
    "scalar_transport": (
        "whole-theta-and-vapor-tricubic-or-quintic-horizontal-semi-lagrangian-"
        "plus-grid-point-quasi-monotone-tricubic-condensate-and-moments-with-"
        "zero-floored-bermejo-conde-mass-fixer-optionally-clip-deficit-"
        "additive-v3"
    ),
    "checkpoint": (
        "hash-bound-five-field-spectral-atmosphere-ten-grid-tracers-grid-"
        "surface-physics-state-eight-run-trackers-and-seven-trajectory-"
        "arrays-v4"
    ),
}

__all__ = ["SEMILAG_PIN_OVERRIDES", "SEMILAG_SEMI_IMPLICIT_PINS"]
