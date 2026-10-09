"""The two-time-level semi-Lagrangian semi-implicit step.

For each advected variable ``X``, with ``A`` an arrival point (which is a
grid point, always) and ``D`` its departure point::

    X^{n+1}_A - alpha dt (L X)^{n+1}_A = Q_X(D) + (dt/2) [2 N^n - N^{n-1}]_A
    Q_X = X^n + dt [ (1 - alpha) (L X)^n + (1/2) N^n ]

``L`` is the semi-implicit linear operator the shipped core already
carries and ``N = A - L`` the nonlinear residual of the advective-form
tendency.  This is the trapezoidal two-time-level scheme with the
arrival-side nonlinear term extrapolated to ``t^{n+1}`` (SETTLS-type) and
off-centred by ``alpha``.

For theta the material tendency ``A`` is zero (adiabatic flow), so
``N = -L theta`` and the WHOLE variable is gathered: a parcel's reference
change is then whatever the vertical stencil reads between its departure
and arrival levels, which is exactly consistent with what the same stencil
reads of the deviation from the reference.  Until 2026-09-06 the core
gathered ``theta - theta_ref(k)`` and carried the reference profile's
material tendency as a grid tendency in ``N``; :mod:`.rhs` records what
that did at the model lid and the measurement that retired it.

Pre-combining the whole departure-side bundle into ``Q_X`` before the
interpolation is what keeps this to ONE interpolated field per advected
variable: three Cartesian wind components, ``theta``, vapour, ten grid
tracers and one two-dimensional field for ``ln ps``.  Interpolating the
state and its tendencies separately would double that, and the gather is
the step's largest new cost.

The left-hand side is the shipped Helmholtz solve, unchanged.  Written
out, ``(I - tau L) y = R`` with ``tau = alpha dt`` is exactly
``VerticalModeSemiImplicit.solve_shifted``, the same routine the IMEX
integrator calls for its stage solves and with the same per-tau cached
inverse; the semi-Lagrangian core calls it ONCE per step where the IMEX
pair calls it twice.

What is NOT clipped, and why.  The quasi-monotone option bounds the
interpolated value by the values of the cell that surrounds the
departure point.  That is right for a positive-definite species: cloud
water has no business exceeding the cloud water around it.  It is wrong
for the dynamical bundle.  ``Q_u`` is a wind plus a tendency, so its
physical bounds are not its neighbours' values at all, and clipping a
wind to its neighbours removes the extremes of a field whose extremes
ARE the flow, every step, with no conservation and no measurement of
what was taken.  So the wind, ``theta'`` and ``ln ps`` are gathered
unlimited by default and vapour and the ten tracers are gathered
limited; ``[semilag] quasi_monotone_dynamics`` reaches the other arm so
the choice can be measured rather than argued.
"""
from __future__ import annotations

import numpy as np

from .interpolate import Stencil, gather_batch
from .options import HORIZONTAL_ORDER, PHYSICS_ARRIVAL_WEIGHT
from .rhs import ReferenceProfile, advective_tendencies
from .state import TrajectoryState
from .tables import SphericalGridTables
from .tracers import DEFICIT_FIXERS, area_weights, fix_mass
from .trajectory import (
    CartesianWind,
    cartesian_wind,
    convergence,
    departure_points,
    extrapolate_level_rate,
    level_rate_from_mass_flux,
    lipschitz,
    refuse_beyond_lipschitz,
)


def _extrapolate_component(current, previous, context):
    """Compute ``2 * current - previous`` without reading resident fields back."""
    from woof.metal.array import MetalArray, MetalArrayModule
    if not isinstance(current, MetalArray):
        return 2.0 * current - previous
    if current.buffer.context is not context or current.dtype != np.dtype(np.float64):
        raise TypeError("resident Global wind extrapolation needs FP64 on its Metal context")
    if isinstance(previous, MetalArray):
        if (previous.buffer.context is not context
                or previous.dtype != np.dtype(np.float64)
                or previous.shape != current.shape):
            raise TypeError("resident Global previous wind must match the current field")
        previous_flat = previous.view((previous.size,), dtype=np.float64)
    else:
        previous_host = np.ascontiguousarray(previous, dtype=np.float64)
        if previous_host.shape != current.shape:
            raise ValueError("Global previous wind shape differs from the current field")
        previous_flat = MetalArrayModule(context).asarray(previous_host.reshape(-1))
    from woof.metal.generated import kernel_specs
    metal = MetalArrayModule(context)
    count = int(current.size)
    current_flat = current.view((count,), dtype=np.float64)
    destination = metal.empty((count,), dtype=np.float64)
    one = metal.asarray(np.asarray([1.0], dtype=np.float64))
    context.submit_kernel(
        "global_semilag_add_scaled_fragment",
        {"base": current_flat, "increment": current_flat,
         "previous": previous_flat, "current_scale": one,
         "previous_scale": one, "destination": destination},
        {"count": count}, exact_policy=False, grid_size=count,
        threads_per_group=kernel_specs.GLOBAL_SEMILAG_ADD_SCALED_FRAGMENT_THREADS_PER_GROUP,
    )
    return destination.view(tuple(current.shape), dtype=np.float64)
from .vectors import transport_to_arrival

from ..constants import CONDENSATE_SPECIES, GRAVITY_M_S2, GRID_TRACERS
from ..profile import profiler_of
from ..spill import resident, spilled
from ..state import MoistHybridState

#: The integrator names this package answers to.
SEMILAG_INTEGRATORS = ("sl_si",)


def grid_tables(model) -> SphericalGridTables:
    """The grid geometry of this model's transform, built once."""
    cached = getattr(model, "_semilag_tables", None)
    backend = model.transform.backend
    if cached is None or cached.dtype != backend.float_dtype:
        cached = SphericalGridTables.create(
            model.transform.grid, xp=backend.xp, dtype=backend.float_dtype,
            metal_context=getattr(model, "metal_diffusion_context", None),
        )
        model._semilag_tables = cached
    return cached


def reference_profile(model) -> ReferenceProfile:
    """The semi-implicit reference column, built once."""
    cached = getattr(model, "_semilag_reference", None)
    if cached is None:
        cached = ReferenceProfile(model)
        model._semilag_reference = cached
    return cached


def _surface_stencil(stencil: Stencil, tables: SphericalGridTables):
    """The horizontal departure points of the LOWEST model level, as a
    four-level stencil with no vertical displacement.

    ``ln ps`` is two-dimensional and the gather is a tricubic on a volume,
    so the surface field is read through four identical levels with the
    departure level index pinned to an exact node.  Three of the four
    vertical Lagrange weights are then exactly zero and the fourth is
    exactly one, so the result is the plain bicubic, computed by the same
    compiled kernel every other field is read by rather than by a second
    interpolation with its own bracket search and its own polar map.  It
    costs a tenth of one volume field on a forty-level stack.
    """
    from woof.metal.array import MetalArray

    # The Metal gather's surface mode selects the lowest-level coordinates
    # directly, so retaining this stencil avoids the host copy and four-level
    # broadcast used by the NumPy/CUDA path.
    if isinstance(stencil.xi, MetalArray):
        return Stencil(xi=stencil.xi, phi=stencil.phi,
                       level=stencil.level, tables=tables)

    xp = tables.xp
    dtype = tables.dtype
    nlev, nlat, nlon = stencil.shape
    xi = xp.ascontiguousarray(
        xp.broadcast_to(stencil.xi[nlev - 1][None], (4, nlat, nlon))
    )
    phi = xp.ascontiguousarray(
        xp.broadcast_to(stencil.phi[nlev - 1][None], (4, nlat, nlon))
    )
    level = xp.ascontiguousarray(
        xp.broadcast_to(
            xp.arange(4, dtype=dtype)[:, None, None], (4, nlat, nlon)
        )
    )
    return Stencil(xi=xi, phi=phi, level=level, tables=tables)


def _host_stencil(stencil: Stencil) -> Stencil:
    """Materialize departure coordinates for the remaining NumPy consumers.

    The cubic gather accepts resident coordinates and should read the
    departure kernel's buffers directly.  Vector transport and the surface
    stencil still use NumPy array expressions, so they receive this explicit
    host view after the main gathers have consumed the resident stencil.
    """
    from woof.metal.array import MetalArray

    if not isinstance(stencil.xi, MetalArray):
        return stencil
    shape = stencil.shape
    return Stencil(
        xi=stencil.xi.to_numpy().reshape(shape),
        phi=stencil.phi.to_numpy().reshape(shape),
        level=stencil.level.to_numpy().reshape(shape),
        tables=stencil.tables,
    )


def _gather_surface(field, surface: Stencil, *, monotone: bool, order: int = 4,
                    keep_resident: bool = False):
    from woof.metal.array import MetalArray

    if (order in (4, 6) and isinstance(surface.xi, MetalArray)
            and surface.tables.metal_context is not None):
        if not isinstance(field, MetalArray):
            from woof.metal.array import MetalArrayModule
            field = MetalArrayModule(surface.tables.metal_context).asarray(
                np.ascontiguousarray(np.asarray(field, dtype=np.float64).reshape(-1))
            ).view(surface.tables.shape, dtype=np.float64)
        from .interpolate import gather_surface_metal
        return gather_surface_metal(field, surface, monotone=monotone,
                                    order=order, keep_resident=keep_resident)

    surface = _host_stencil(surface)
    xp = surface.tables.xp
    nlat, nlon = surface.tables.shape
    if isinstance(field, MetalArray):
        field = field.to_numpy().reshape((nlat, nlon))
    tiled = xp.ascontiguousarray(
        xp.broadcast_to(field[None], (4, nlat, nlon))
    )
    return gather_batch([tiled], surface, monotone=monotone, batch=1,
                        order=order)[0][1]


def _metal_theta_update(base, current, previous, current_scale, previous_scale):
    """Keep a pointwise SL residual update on its resident Metal base."""
    from woof.metal.array import MetalArray, MetalArrayModule
    from woof.metal.generated import kernel_specs

    if not isinstance(base, MetalArray):
        raise TypeError("resident Global update requires a MetalArray base")
    context = base.buffer.context
    if base.dtype != np.dtype(np.float64):
        raise ValueError("resident Global update base must be FP64")
    metal = MetalArrayModule(context)

    def resource(value):
        if value is None:
            return metal.zeros(base.shape, dtype=np.float64)
        if isinstance(value, MetalArray):
            if value.buffer.context is not context:
                raise ValueError("resident Global update input has a different context")
            if value.dtype != np.dtype(np.float64) or value.shape != base.shape:
                raise ValueError("resident Global update input has a different FP64 shape")
            return metal.asarray(value)
        host = np.asarray(value, dtype=np.float64)
        if host.size != base.size:
            raise ValueError("Global update host input has a different element count")
        return metal.asarray(np.ascontiguousarray(host.reshape(-1))).view(
            base.shape, dtype=np.float64
        )

    current_values = resource(current)
    previous_values = resource(previous)
    destination = metal.empty(base.shape, dtype=np.float64)
    context.submit_kernel(
        "global_semilag_add_scaled_fragment",
        {"base": metal.asarray(base).view((base.size,), dtype=np.float64),
         "increment": current_values.view((current_values.size,), dtype=np.float64),
         "previous": previous_values.view((previous_values.size,), dtype=np.float64),
         "current_scale": metal.asarray(np.asarray([current_scale], dtype=np.float64)),
         "previous_scale": metal.asarray(np.asarray([previous_scale], dtype=np.float64)),
         "destination": destination.view((destination.size,), dtype=np.float64)},
        {"count": base.size}, exact_policy=False, grid_size=base.size,
        threads_per_group=kernel_specs.GLOBAL_SEMILAG_ADD_SCALED_FRAGMENT_THREADS_PER_GROUP,
    )
    return destination


def _metal_apply_arrival_increment(base, increment, gathered, weight):
    """Apply ``base + weight * (increment - gathered)`` on resident FP64."""
    difference = _metal_theta_update(
        increment, gathered, gathered, -1.0, 0.0
    )
    return _metal_theta_update(
        base, difference, difference, weight, 0.0
    )


def _metal_complex_increment_max(after, before, tables):
    """Reduce a resident complex spectrum's difference without host copies."""
    from woof.metal.array import MetalArray, MetalArrayModule
    from woof.metal.generated import kernel_specs

    context = tables.metal_context
    if (not isinstance(after, MetalArray) or not isinstance(before, MetalArray)
            or context is None or after.buffer.context is not context
            or before.buffer.context is not context
            or after.dtype != np.dtype(np.complex128)
            or before.dtype != np.dtype(np.complex128)
            or after.shape != before.shape):
        raise TypeError("Global spectral increment diagnostic needs matching complex128 Metal fields")
    metal = MetalArrayModule(context)
    count = int(after.size)
    magnitudes = metal.empty((count,), dtype=np.float64)
    context.submit_kernel(
        "global_semilag_complex_difference_fragment",
        {"before": before.view((2 * count,), dtype=np.float64),
         "after": after.view((2 * count,), dtype=np.float64),
         "magnitudes": magnitudes},
        {"count": count}, exact_policy=False, grid_size=count,
        threads_per_group=(
            kernel_specs.GLOBAL_SEMILAG_COMPLEX_DIFFERENCE_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    from .trajectory import _diagnostic_max
    return _diagnostic_max(tables, magnitudes)


def _physics_increment(model, atmosphere, pre, tables, *, riding: bool):
    """The FIRST physics half's increment, in grid space.

    Taken as a difference in SPECTRAL space and synthesized ONCE, rather
    than as the difference of two grid syntheses: the increment is five
    spectral fields and one synthesis, where the two-synthesis form would
    pay the 4.15 ms T255 stacked synthesis twice for the same numbers.
    The ten grid tracers are already grid fields and their increment is a
    plain subtraction with no transform at all.
    """
    transform = model.transform
    xp = transform.backend.xp
    nlev = int(model.nlev)
    from woof.metal.array import MetalArray, MetalArrayModule

    context = getattr(model, "metal_diffusion_context", None)
    if context is not None:
        from .rhs import subtract_resident

        metal = MetalArrayModule(context)

        def complex_state(value):
            if isinstance(value, MetalArray):
                if (value.buffer.context is not context
                        or value.dtype != np.dtype(np.complex128)):
                    raise TypeError(
                        "Global resident physics spectra must be complex128 on one context"
                    )
                return value
            return metal.asarray(np.ascontiguousarray(
                value, dtype=np.complex128
            ))

        def spectral_difference(after, before):
            return subtract_resident(
                complex_state(after), complex_state(before), context
            )

        delta_vorticity = spectral_difference(
            atmosphere.vorticity, pre.vorticity
        )
        delta_divergence = spectral_difference(
            atmosphere.divergence, pre.divergence
        )
        d_u, d_v = model.vector.wind_from_vordiv(
            delta_vorticity, delta_divergence
        )
        delta_theta = spectral_difference(atmosphere.theta, pre.theta)
        delta_qv = spectral_difference(atmosphere.qv, pre.qv)
        grid_theta = transform.inverse(delta_theta, keep_resident=True)
        grid_qv = transform.inverse(delta_qv, keep_resident=True)
        lnps = transform.inverse(
            spectral_difference(
                atmosphere.log_surface_pressure,
                pre.log_surface_pressure,
            ),
            keep_resident=True,
        )
        d_x, d_y, d_z = cartesian_wind(
            d_u, d_v, tables, keep_resident=True
        )
        return {
            "u": d_u, "v": d_v, "cartesian": (d_x, d_y, d_z),
            "theta": grid_theta, "qv": grid_qv, "lnps": lnps,
            "tracers": _tracer_increment(
                atmosphere, pre, context=context
            ) if riding else {},
        }

    d_u, d_v = model.vector.wind_from_vordiv(
        atmosphere.vorticity - pre.vorticity,
        atmosphere.divergence - pre.divergence,
    )
    stacked = xp.concatenate([
        atmosphere.theta - pre.theta, atmosphere.qv - pre.qv,
    ], axis=0)
    grid = model._chunked(transform.inverse, stacked)
    del stacked
    d_x, d_y, d_z = cartesian_wind(d_u, d_v, tables)
    return {
        "u": d_u, "v": d_v, "cartesian": (d_x, d_y, d_z),
        "theta": grid[:nlev], "qv": grid[nlev:],
        "lnps": transform.inverse(
            atmosphere.log_surface_pressure - pre.log_surface_pressure
        ),
        "tracers": _tracer_increment(atmosphere, pre) if riding else {},
    }


def _tracer_increment(atmosphere, pre, *, context=None) -> dict:
    """The physics half's increment of the ten grid tracers.

    REFUSED while the pinned host tier holds them: the physics half-step
    writes a parked tracer's slot in place (dynamics.apply_physics,
    band by band), so the state the half started from no longer exists
    as an array to subtract, and a difference taken against the slot
    would read zero on every tracer and couple no physics at all.  The
    shipped coupling ("advected") never takes this difference; the two
    that do run with the tier off.
    """
    parked = [name for name in GRID_TRACERS
              if spilled(getattr(atmosphere, name)) or spilled(getattr(pre, name))]
    if parked:
        raise ValueError(
            "semilag physics_coupling needs the grid tracers' physics "
            f"increment and the pinned host tier holds {', '.join(parked)}: "
            "the physics half-step writes a parked tracer's slot in place, "
            "so the pre-physics tracers no longer exist to subtract and the "
            "increment would read zero.  Run with [memory] host_spill = "
            "\"off\" or with [semilag] physics_coupling = \"advected\"."
        )
    if context is not None:
        from woof.metal.array import MetalArray, MetalArrayModule

        metal = MetalArrayModule(context)

        def resident(value):
            if isinstance(value, MetalArray):
                if (value.buffer.context is not context
                        or value.dtype != np.dtype(np.float64)):
                    raise TypeError(
                        "Global resident physics tracers must be FP64 on one context"
                    )
                return value
            return metal.asarray(np.ascontiguousarray(
                value, dtype=np.float64
            ))

        increments = {}
        for name in GRID_TRACERS:
            after = resident(getattr(atmosphere, name))
            before = resident(getattr(pre, name))
            increments[name] = _metal_theta_update(
                after, before, before, -1.0, 0.0
            )
        return increments
    return {name: getattr(atmosphere, name) - getattr(pre, name)
            for name in GRID_TRACERS}


def semilag_step(model, atmosphere: MoistHybridState, dt_s: float, mark=None):
    """One two-time-level semi-Lagrangian semi-implicit step of the
    adiabatic core, tracers included.

    Returns ``(advanced, metrics)`` in the shape
    :meth:`woof.globe.dynamics.MoistHybridModel.integrate_dynamics`
    contracts for, with the trajectory's own diagnostics beside the
    semi-implicit increment.
    """
    options = model.semilag
    transform = model.transform
    backend = transform.backend
    xp = backend.xp
    scalar = backend.float_dtype
    dt = float(dt_s)
    alpha = float(model.semi_implicit.off_centring_weight)
    tables = grid_tables(model)
    reference = reference_profile(model)
    previous = model.trajectory_state()
    startup = previous is None
    prof = profiler_of(model)
    # How much of the first physics half's increment is moved from the
    # departure point to the arrival point (semilag.options): zero under
    # the plain Strang split, which is the default and pays nothing at
    # all, because the increment is then never separated from the state.
    arrival_weight = float(PHYSICS_ARRIVAL_WEIGHT[options.physics_coupling])
    # The dynamical bundle's horizontal stencil width (semilag.options
    # HORIZONTAL_INTERPOLATIONS); the limited species stay on the cubic.
    order = int(HORIZONTAL_ORDER[options.horizontal_interpolation])
    pre_physics = model.take_pre_physics() if arrival_weight else None
    if arrival_weight and pre_physics is None:
        raise ValueError(
            f"semilag.physics_coupling = {options.physics_coupling!r} needs "
            "the state as it stood before the first physics half, and the "
            "step did not hand one over; a coupling that silently fell back "
            "to 'advected' would produce a different run under the same "
            "config and say nothing"
        )

    with prof.section("grid_state"):
        # One set of sources feeds both the grid view and the mass-flux
        # pass, because the band loop keys its memo and its per-band
        # syntheses off the sources object rather than off a grid dict.
        sources = model.grid_sources(atmosphere)
        g, _stack, _names = model.grid_band(sources, model.whole_rows)
        _spec, _div_mass, ps_t, omega_half = model._mass_flux_and_omega(sources)

    with prof.section("trajectory"):
        level_rate = level_rate_from_mass_flux(
            omega_half, g["dp"], xp=xp,
            metal_context=getattr(model, "metal_diffusion_context", None),
            keep_resident=getattr(model, "metal_diffusion_context", None) is not None,
        )
        vx, vy, vz = cartesian_wind(
            g["u"], g["v"], tables,
            keep_resident=getattr(model, "metal_diffusion_context", None) is not None,
        )
        wind = CartesianWind(vx=vx, vy=vy, vz=vz, level_rate=level_rate)
        if startup or options.extrapolation == "none":
            extrapolated = wind
        else:
            metal_context = getattr(model, "metal_diffusion_context", None)
            ex_u = _extrapolate_component(g["u"], previous.u_prev, metal_context)
            ex_v = _extrapolate_component(g["v"], previous.v_prev, metal_context)
            ex_x, ex_y, ex_z = cartesian_wind(
                ex_u, ex_v, tables,
                keep_resident=getattr(model, "metal_diffusion_context", None) is not None,
            )
            del ex_u, ex_v
            extrapolated = CartesianWind(
                vx=ex_x, vy=ex_y, vz=ex_z,
                level_rate=extrapolate_level_rate(
                    level_rate, previous.s_prev,
                    metal_context=getattr(model, "metal_diffusion_context", None),
                    keep_resident=getattr(model, "metal_diffusion_context", None) is not None,
                ),
            )
        deformation = lipschitz(wind, tables, dt)
        refuse_beyond_lipschitz(deformation, model.maximum_lipschitz)
        stencil, trajectory = departure_points(
            wind, tables, dt,
            iterations=int(options.trajectory_iterations),
            extrapolated=extrapolated,
            keep_resident=(order in (4, 6)),
        )
        del extrapolated
        convergence(trajectory, options.trajectory_convergence_cells)

    half = scalar(0.5 * dt)
    lead = scalar((1.0 - alpha) * dt)
    with prof.section("tendencies"):
        tendencies = advective_tendencies(
            model, atmosphere, g, omega_half, ps_t, reference,
            bundle_lead=float(lead), bundle_half=float(half),
        )
        if "bundle_u" in tendencies:
            n_u, n_v = tendencies["n_u"], tendencies["n_v"]
            n_theta, n_lnps = tendencies["n_theta"], tendencies["n_lnps"]
            bundle_u, bundle_v = tendencies["bundle_u"], tendencies["bundle_v"]
            bundle_theta, bundle_lnps = (
                tendencies["bundle_theta"], tendencies["bundle_lnps"]
            )
        else:
            n_u = tendencies["a_u"] - tendencies["l_u"]
            n_v = tendencies["a_v"] - tendencies["l_v"]
            # Adiabatic flow: the material tendency of theta is ZERO, so its
            # nonlinear residual is minus the operator's own row and the
            # reference profile is read along the trajectory by the gather of
            # the whole variable (rhs.advective_tendencies says why the
            # reference-subtracted form was retired, with the measurement).
            n_theta = -tendencies["l_theta"]
            n_lnps = tendencies["a_lnps"] - tendencies["l_lnps"]
            bundle_u = g["u"] + lead * tendencies["l_u"] + half * n_u
            bundle_v = g["v"] + lead * tendencies["l_v"] + half * n_v
            bundle_theta = g["theta"] + lead * tendencies["l_theta"] + half * n_theta
            bundle_lnps = (
                g["logps"] + lead * tendencies["l_lnps"] + half * n_lnps
            )
        del tendencies
        bundle_x, bundle_y, bundle_z = cartesian_wind(
            bundle_u, bundle_v, tables,
            keep_resident=getattr(model, "metal_diffusion_context", None) is not None,
        )
        del bundle_u, bundle_v

    with prof.section("gather"):
        batch = int(options.gather_batch)
        gathered = gather_batch(
            [bundle_x, bundle_y, bundle_z, bundle_theta], stencil,
            monotone=bool(options.quasi_monotone_dynamics), batch=batch,
            order=order,
            keep_resident=getattr(model, "metal_diffusion_context", None) is not None,
        )
        del bundle_x, bundle_y, bundle_z, bundle_theta
        departure_x, departure_y, departure_z, departure_theta = gathered
        del gathered
        # Under tracer_scheme = "flux_form" the ten tracers are left for
        # dynamics.step()'s Eulerian sweep and only vapour is gathered
        # here; the trial state carries the originals through the solve
        # by reference, exactly as it does on the IMEX path.
        riding = options.tracer_scheme == "semi_lagrangian"
        # The additive fixer puts a species' mass back where the limiter
        # took it, so it needs the limiter's own record of that: the
        # gather reports it out of arithmetic it already does, and the
        # values it returns are bit for bit the plain limited gather's.
        limiter = "quasi_monotone" if options.quasi_monotone else "none"
        want_deficit = bool(
            riding and limiter != "none"
            and options.tracer_fixer in DEFICIT_FIXERS
        )
        # The grid tracers may live in the pinned host tier (spill): the
        # gather stages a parked field one batch at a time and the copy
        # dies with the batch (interpolate.gather_batch).  MEASURED
        # 2026-09-07, T533 L40 sl_si with all three slices parked on an
        # RTX 5070 Ti: handed the slot itself, the kernel refused it by
        # type at step 1; handed all ten staged at once, the card ran out
        # at 15.96 GB inside the gather.
        # A tracer the pinned host tier holds is staged onto the card
        # here, where the gather reads it; under ``flux_form`` the ten are
        # NOT gathered and must stay parked, because they then ride through
        # the solve by reference and dynamics.step's Eulerian sweep is what
        # stages and re-parks them.  Staging them here as well would take
        # the tier off the tracers for the rest of the run.
        keep_qv_resident = (
            getattr(model, "metal_diffusion_context", None) is not None
            and not want_deficit
        )
        if keep_qv_resident:
            departure_qv = gather_batch(
                [g["qv"]], stencil, monotone=limiter, batch=batch,
                order=4, keep_resident=True,
            )[0]
            tracer_fields = ([getattr(atmosphere, name) for name in GRID_TRACERS]
                             if riding else [])
            species = gather_batch(
                tracer_fields, stencil, monotone=limiter, batch=batch,
                order=4, keep_resident=bool(tracer_fields),
            ) if tracer_fields else []
            advected = (dict(zip(GRID_TRACERS, species)) if riding
                        else atmosphere.grid_tracers())
            deficits = None
        else:
            species = gather_batch(
                [g["qv"], *(
                    (getattr(atmosphere, name) for name in GRID_TRACERS)
                    if riding else ()
                )],
                stencil, monotone=limiter, batch=batch,
                deficit=want_deficit,
            )
            if want_deficit:
                departure_qv = species[0][0]
                advected = dict(zip(GRID_TRACERS,
                                    (row[0] for row in species[1:])))
                deficits = dict(zip(GRID_TRACERS,
                                    (row[1] for row in species[1:])))
            else:
                departure_qv = species[0]
                advected = (
                    dict(zip(GRID_TRACERS, species[1:])) if riding
                    else atmosphere.grid_tracers()
                )
                deficits = None
        del species
        consumer_stencil = stencil
        surface = _surface_stencil(stencil, tables)
        departure_lnps = _gather_surface(
            bundle_lnps, surface, monotone=bool(options.quasi_monotone_dynamics),
            order=order,
            keep_resident=getattr(model, "metal_diffusion_context", None) is not None,
        )
        del bundle_lnps
        arrival_u, arrival_v = transport_to_arrival(
            departure_x, departure_y, departure_z, consumer_stencil, tables,
            keep_resident=getattr(model, "metal_diffusion_context", None) is not None,
        )
        del departure_x, departure_y, departure_z

    with prof.section("physics_coupling"):
        # The first physics half ran at the grid points, which are the
        # arrival points, and its increment went into the state BEFORE the
        # gather, so under the default coupling a parcel reads it at its
        # departure point along with everything else.  The other two
        # couplings move a fraction of it to the arrival point, by taking
        # it back out where the gather put it and adding it where the
        # parcel lands:
        #
        #     X = gather(X_pre + I + ...)  +  w * (I_A - gather(I))
        #
        # with w = 1 for "arrival" and 1/2 for "trajectory_average".  What
        # it buys is that a field the physics has just created is not
        # interpolated on the step that created it; what it costs is one
        # synthesis of the increment and one more gathered field per
        # advected variable, and, for a physics SINK, that the removal
        # lands where it was not computed, which the positivity repair
        # then has to hold.  That is why the default is the plain split.
        if arrival_weight:
            inc = _physics_increment(
                model, atmosphere, pre_physics, tables, riding=riding
            )
            del pre_physics
            names = list(inc["tracers"])
            from woof.metal.array import MetalArray

            resident_fields = isinstance(arrival_u, MetalArray)
            # The dynamical rows read the bundle's own stencil width and
            # the species rows the cubic, so what is taken back out is
            # what the gather put in, field by field.
            rows = gather_batch(
                [*inc["cartesian"], inc["theta"]], stencil,
                monotone=False, batch=batch, order=order,
                keep_resident=resident_fields,
            ) + gather_batch(
                [inc["qv"], *(inc["tracers"][name] for name in names)],
                stencil, monotone=False, batch=batch,
                keep_resident=resident_fields,
            )
            gathered_u, gathered_v = transport_to_arrival(
                rows[0], rows[1], rows[2],
                consumer_stencil if resident_fields else _host_stencil(consumer_stencil),
                tables, keep_resident=resident_fields,
            )
            weight = scalar(arrival_weight)
            if resident_fields:
                arrival_u = _metal_apply_arrival_increment(
                    arrival_u, inc["u"], gathered_u, float(weight)
                )
                arrival_v = _metal_apply_arrival_increment(
                    arrival_v, inc["v"], gathered_v, float(weight)
                )
                departure_theta = _metal_apply_arrival_increment(
                    departure_theta, inc["theta"], rows[3], float(weight)
                )
                departure_qv = _metal_apply_arrival_increment(
                    departure_qv, inc["qv"], rows[4], float(weight)
                )
                for index, name in enumerate(names):
                    advected[name] = _metal_apply_arrival_increment(
                        advected[name], inc["tracers"][name],
                        rows[5 + index], float(weight),
                    )
                gathered_lnps = _gather_surface(
                    inc["lnps"], surface, monotone=False, order=order,
                    keep_resident=True,
                )
                departure_lnps = _metal_apply_arrival_increment(
                    departure_lnps, inc["lnps"], gathered_lnps,
                    float(weight),
                )
                del gathered_lnps
            else:
                arrival_u = arrival_u + weight * (inc["u"] - gathered_u)
                arrival_v = arrival_v + weight * (inc["v"] - gathered_v)
                departure_theta = departure_theta + weight * (
                    inc["theta"] - rows[3]
                )
                departure_qv = departure_qv + weight * (
                    inc["qv"] - rows[4]
                )
                for index, name in enumerate(names):
                    advected[name] = advected[name] + weight * (
                        inc["tracers"][name] - rows[5 + index]
                    )
                departure_lnps = departure_lnps + weight * (
                    inc["lnps"] - _gather_surface(
                        inc["lnps"], surface, monotone=False, order=order
                    )
                )
            del gathered_u, gathered_v
            del inc, rows, names
    del surface, stencil, consumer_stencil

    with prof.section("assemble"):
        from woof.metal.array import MetalArray
        if startup or options.extrapolation == "none":
            if isinstance(arrival_u, MetalArray):
                arrival_u = _metal_theta_update(arrival_u, n_u, None, float(half), 0.0)
                arrival_v = _metal_theta_update(arrival_v, n_v, None, float(half), 0.0)
            else:
                arrival_u = arrival_u + half * n_u
                arrival_v = arrival_v + half * n_v
            if isinstance(departure_theta, MetalArray) and isinstance(n_theta, MetalArray):
                departure_theta = _metal_theta_update(
                    departure_theta, n_theta, None, float(half), 0.0
                )
            else:
                departure_theta = departure_theta + half * n_theta
            if isinstance(departure_lnps, MetalArray):
                departure_lnps = _metal_theta_update(
                    departure_lnps, n_lnps, None, float(half), 0.0
                )
            else:
                departure_lnps = departure_lnps + half * n_lnps
        else:
            if isinstance(arrival_u, MetalArray):
                arrival_u = _metal_theta_update(
                    arrival_u, n_u, previous.n_u, float(2.0 * half), float(half)
                )
                arrival_v = _metal_theta_update(
                    arrival_v, n_v, previous.n_v, float(2.0 * half), float(half)
                )
            else:
                arrival_u = arrival_u + half * (2.0 * n_u - previous.n_u)
                arrival_v = arrival_v + half * (2.0 * n_v - previous.n_v)
            if isinstance(departure_theta, MetalArray) and isinstance(n_theta, MetalArray):
                departure_theta = _metal_theta_update(
                    departure_theta, n_theta, previous.n_theta,
                    float(2.0 * half), float(half),
                )
            else:
                departure_theta = departure_theta + half * (
                    2.0 * n_theta - previous.n_theta
                )
            if isinstance(departure_lnps, MetalArray):
                departure_lnps = _metal_theta_update(
                    departure_lnps, n_lnps, previous.n_lnps,
                    float(2.0 * half), float(half),
                )
            else:
                departure_lnps = departure_lnps + half * (
                    2.0 * n_lnps - previous.n_lnps
                )
        trial_zeta, trial_divergence = model.vector.vordiv_from_wind(
            arrival_u, arrival_v
        )
        del arrival_u, arrival_v
        if isinstance(departure_theta, MetalArray):
            theta_spectral = transform.forward(
                departure_theta, keep_resident=True
            )
            if isinstance(departure_qv, MetalArray):
                qv_spectral = transform.forward(
                    departure_qv, keep_resident=True
                )
            else:
                qv_spectral = model._chunked(
                    lambda block: transform.project(transform.forward(block)),
                    departure_qv,
                )
        else:
            stacked = xp.concatenate([departure_theta, departure_qv], axis=0)
            analysed = model._chunked(
                lambda block: transform.project(transform.forward(block)), stacked
            )
            del stacked
            theta_spectral = analysed[:int(model.nlev)]
            qv_spectral = analysed[int(model.nlev):]
            del analysed
        del departure_theta, departure_qv
        nlev = int(model.nlev)
        trial = MoistHybridState(
            vorticity=trial_zeta,
            divergence=trial_divergence,
            theta=theta_spectral,
            log_surface_pressure=(
                transform.forward(departure_lnps, keep_resident=True)
                if isinstance(departure_lnps, MetalArray)
                else transform.project(transform.forward(departure_lnps))
            ),
            qv=qv_spectral,
            **advected,
            time_s=atmosphere.time_s,
            step=atmosphere.step,
        )
        del theta_spectral, qv_spectral, departure_lnps, trial_zeta, previous
        if mark is not None:
            mark("semilag_transport", trial)

    with prof.section("solve"):
        advanced = model.semi_implicit.solve_shifted(
            trial, transform, model.vertical, alpha * dt,
            keep_resident=(getattr(model, "metal_diffusion_context", None) is not None),
        )
        from woof.metal.array import MetalArray
        if isinstance(advanced.divergence, MetalArray):
            increment = _metal_complex_increment_max(
                advanced.divergence, trial_divergence, tables
            )
        else:
            increment = float(
                xp.max(xp.abs(advanced.divergence - trial_divergence))
            )
        del trial_divergence

    with prof.section("tracers"):
        del trial
        fixer: dict[str, float] = {}
        water_relative = 0.0
        if riding:
            after = model.grid_state(advanced, only=("dp",))
            # The mass fixer measures the advected species against the
            # species as they stood, and the water reading below reads the
            # same ones, so the pinned host tier is read ONCE here and the
            # staged copy serves both.  It is read here rather than held
            # from the gather because ten grid volumes are 1.9 GiB at T533
            # and holding them across the trajectory and the solve would
            # put them beside the step's own peak; two lines of the same
            # section is a lifetime the peak does not see.  A resident run
            # gets its own arrays back and copies nothing.
            before = {name: resident(xp, value) for name, value
                      in atmosphere.grid_tracers().items()}
            fixed, fixer = fix_mass(
                advected, before, g["dp"], after["dp"],
                transform, scheme=options.tracer_fixer, deficits=deficits,
            )
            # The advanced ten stay on the card from here to the second
            # physics half, which returns them to the tier's own slots
            # (dynamics._physics_result_to_state, and the sponge-only pass
            # on the dry route).  Parking them here as well would write
            # 1.9 GiB back at T533 and read it again a few operators
            # later, for a peak that is unchanged: the step's peak is in
            # the physics half, where one device copy of the ten exists
            # either way.
            advanced = advanced.with_grid_tracers(fixed)
            # The fixer's NET correction as a fraction of the atmosphere's
            # own water, which is the quantity the breakage is about: a
            # correction of four percent of a species whose global mass is
            # a millionth of the column water is four hundredths of a
            # millionth of the water, and the per-species relative number
            # on its own cannot say that.  The sum is SIGNED, because a
            # species the fixer fed and a species it starved are the same
            # water moving between them and not two losses; a per-species
            # magnitude belongs to the per-species row.  Number moments are
            # counts per kilogram, not water, and are not in this sum.  The
            # 1/g that turns a Pa-weighted mass into kg/m2 divides both
            # sides and is written on both so the two are the same quantity.
            from woof.metal.array import MetalArray
            if isinstance(g["qv"], MetalArray):
                from woof.metal.array import MetalArrayModule
                from woof.metal.generated import kernel_specs

                context = g["qv"].buffer.context
                if context is not getattr(model, "metal_diffusion_context", None):
                    raise ValueError("Global water diagnostic belongs to another Metal context")
                shape = tuple(int(value) for value in g["qv"].shape)
                if (len(shape) != 3 or tuple(g["dp"].shape) != shape
                        or g["qv"].dtype != np.dtype(np.float64)):
                    raise TypeError("Global resident water diagnostic needs matching FP64 fields")
                nlev, nlat, nlon = shape
                count = nlev * nlat * nlon
                metal = MetalArrayModule(context)

                def field(value):
                    if (not isinstance(value, MetalArray)
                            or value.buffer.context is not context
                            or value.dtype != np.dtype(np.float64)
                            or tuple(value.shape) != shape):
                        raise TypeError("Global resident water diagnostic fields must share FP64 shape")
                    return value.view((count,), dtype=np.float64)

                water_column = metal.empty((nlat * nlon,), dtype=np.float64)
                context.submit_kernel(
                    "global_semilag_water_column_fragment",
                    {"qv": field(g["qv"]),
                     **{name: field(before[name]) for name in CONDENSATE_SPECIES},
                     "dp": field(g["dp"]),
                     "gravity": metal.asarray(np.asarray(
                         [GRAVITY_M_S2], dtype=np.float64)),
                     "water_column": water_column},
                    {"nlevel": nlev, "nlat": nlat, "nlon": nlon},
                    exact_policy=False, grid_size=nlat * nlon,
                    threads_per_group=(
                        kernel_specs.GLOBAL_SEMILAG_WATER_COLUMN_FRAGMENT_THREADS_PER_GROUP
                    ),
                )
                column = model._global_grid_mean(
                    water_column.view((nlat, nlon), dtype=np.float64)
                )
            else:
                cell = area_weights(transform)
                column = float(xp.sum(
                    (g["qv"] + sum(before[name]
                                   for name in CONDENSATE_SPECIES))
                    * g["dp"] * cell
                )) / GRAVITY_M_S2
            moved = sum(
                value for name, value in fixer.items()
                if name.startswith("semilag_tracer_mass_fixer_kg_m2__")
                and name.rsplit("__", 1)[1] in CONDENSATE_SPECIES
            ) / GRAVITY_M_S2
            water_relative = abs(moved) / max(column, 1.0e-30)
            del fixed, after, before
        del advected, deficits

    model.set_trajectory_state(TrajectoryState(
        u_prev=g["u"], v_prev=g["v"], s_prev=level_rate,
        n_u=n_u, n_v=n_v, n_theta=n_theta, n_lnps=n_lnps,
    ))
    del g, n_u, n_v, n_theta, n_lnps, level_rate, wind, vx, vy, vz

    if mark is not None:
        mark("semilag_implicit", advanced)
    # The largest RELATIVE correction, over species.  Named explicitly
    # rather than taken as the maximum of the fixer's whole record: that
    # record also carries the correction in kg/m2 and the clip share, and a
    # number moment's correction in counts per kilogram is a number like
    # 2.6e9, which a bare max over the values reports as a relative
    # magnitude of 2.6e9 and fails every gate that reads it.
    fixer_max = max(
        (value for name, value in fixer.items()
         if name.startswith("semilag_tracer_mass_fixer_relative__")),
        default=0.0,
    )
    # The row dynamics.step() reads where the flux-form sweep's own
    # metrics would be.  The names are kept so a receipt of either arm
    # has the same shape; ``scheme`` says which produced them and the
    # values mean what they say.  A semi-Lagrangian gather crosses the
    # cells it crosses in ONE pass -- that is the whole point -- so the
    # sub-cycle counts are one and the displacement is reported under the
    # Courant names it is the semi-Lagrangian analogue of.  There is no
    # pseudo-density here and therefore no second continuity
    # discretization to disagree with the first, so that gap is zero by
    # construction rather than by measurement.
    transport = {
        "scheme": "semi_lagrangian",
        "max_courant_x": float(trajectory.displacement_max_cells),
        "max_courant_y": float(trajectory.displacement_max_cells),
        "max_courant_z": float(trajectory.displacement_max_levels),
        "substeps_x": 1, "substeps_y": 1, "substeps_z": 1,
        "floor_clip_kg_m2": 0.0,
        "pseudo_density_mismatch_relative": 0.0,
        "pseudo_density_mismatch_mean_relative": 0.0,
        "mass_fixer_max_relative": float(fixer_max),
        **fixer,
    }
    metrics = {
        **({"tracer_transport": transport} if riding else {}),
        "semi_implicit_max_divergence_increment_s1": increment,
        "semilag_startup_step": bool(startup),
        "semilag_off_centring_weight": alpha,
        "semilag_tracer_mass_fixer_relative": float(fixer_max),
        "semilag_tracer_mass_fixer_water_relative": float(water_relative),
        "semilag_tracer_limiter": limiter,
        **fixer,
        **deformation.as_dict(),
        **trajectory.as_dict(),
    }
    return advanced, metrics


__all__ = ["SEMILAG_INTEGRATORS", "grid_tables", "reference_profile",
           "semilag_step"]
