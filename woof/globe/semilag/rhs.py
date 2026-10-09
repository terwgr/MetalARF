"""The advective-form right-hand side, and the linear operator on the grid.

The shipped Eulerian core evaluates its tendencies in FLUX form: the
momentum in the vector-invariant form whose divergence and curl the
spectral analysis takes directly, and theta and vapour as the divergence
of a mass flux.  A semi-Lagrangian step needs neither.  It needs the
material tendency at a point, what a parcel's wind, temperature and
surface pressure do while it travels, and it needs the semi-implicit
linear operator in the same place, so the two can be subtracted.

The identity that turns one into the other is the vector-invariant one::

    (u/(a cos phi)) du/dlam + (v/a) du/dphi = dK/dx - zeta v + u v tan(phi)/a

so the whole curvature term cancels against the vector-invariant form's
own and what is left is the classical advective momentum equation::

    Du/Dt = (f + u tan(phi)/a) v - dPhi/dx - R Tv factor_k dlnps/dx
    Dv/Dt = -(f + u tan(phi)/a) u - dPhi/dy - R Tv factor_k dlnps/dy

with ``factor_k`` the exact full-level ``grad(ln p_k)`` per unit
``grad(ln ps)`` the shipped core already builds (the DN-3 pairing with
the half-layer hydrostatic integration, which is what makes those two
terms balance pointwise at rest).

The ``tan(phi)`` metric terms do NOT appear below.  They are exactly the
terms that vanish when the momentum is carried as three geocentric
Cartesian components of a tangent field: writing ``V = u e_lam + v
e_phi`` and differentiating the basis along the trajectory,

    D V_c/Dt = [Du/Dt - uv tan(phi)/a] e_lam
             + [Dv/Dt + u^2 tan(phi)/a] e_phi - (|V|^2/a) r

and the two bracketed metric terms cancel against the ones above.  The
remaining radial term is the parallel transport that
:mod:`woof.globe.semilag.vectors` integrates exactly as a
rotation.  So this module returns the Coriolis and force terms alone and
the geometry is carried by the trajectory, rather than by an arithmetic
term that would have to be right at a pole where the basis it is written
in turns through 2*pi in 326 m.

The thermodynamic variable has NO explicit material tendency here.
Adiabatic flow conserves theta along a parcel, so the step gathers the
whole variable and its nonlinear residual is ``-L theta``; the semi-implicit
operator's thermodynamic row (the reference profile's vertical advection
by the linearized mass flux) stays implicit at the arrival point and
explicit at the departure point exactly as it does for every other row,
and the linear stability analysis of the step is unchanged, because the
reference change a trajectory reads between its departure and arrival
levels is the SETTLS trajectory average of that same term to second order.

Until 2026-09-06 the core gathered ``theta' = theta - theta_ref(k)`` and
carried ``-D theta_ref/Dt`` as a grid tendency in the operator's own flux
form (the reference interface values, the run's mass flux).  That form is
consistent with the gather in the interior and inconsistent at the two
boundary levels, where the flux form is one-sided and the trajectory is
CLAMPED: under descent at the lid the parcel arriving at level 0 departs
from level 0 (nothing lies above it), the gather reads no reference
change, and the tendency warmed it anyway by ``-omega_{1/2} (face_1 -
theta_ref_0) / dp_0`` per unit time, where the Eulerian core's upwind
flux carries the layer's own value out of it and warms nothing.  MEASURED
2026-09-06 on the Eulerian arm's own 24 h T255 state (the 16 GB host): one 300 s
step of the retired form changes the level-0 potential temperature by
+0.1127 K in the global mean (+0.206 K over the descending half of the
globe, +0.016 K over the ascending half), which is +32.5 K per forecast
day against the +33.6 K the semi-Lagrangian day of record read above the
Eulerian one at level 0, zonally uniform to 1.7 K and growing linearly
through the day (+19.2 K at 12 h, +26.6 K at 18 h); the same form at level
1 is 29 percent too steep for any displacement because the top layer's
zero-gradient reconstruction pulls the first interface value toward the
lid.  Reading the reference along the trajectory instead gives -0.043 K
per step at level 0 on the same state, the sign and half the size of the
Eulerian upwind flux form's -0.090 K.  That is the whole of the 4.7 K
lid disagreement the observation grade could not see, and it is why the
variable is gathered whole.  :func:`reference_theta_faces` stays because
the gate that rebuilds the operator's gamma matrix reads it.
"""
from __future__ import annotations

import numpy as np


def _negate_resident(value, context):
    from woof.metal.array import MetalArray, MetalArrayModule

    if not isinstance(value, MetalArray):
        return -value
    if value.buffer.context is not context or value.dtype != np.dtype(np.float64):
        raise TypeError("resident Global tendency sign change requires FP64 Metal")
    from woof.metal.generated import kernel_specs

    metal = MetalArrayModule(context)
    count = int(value.size)
    flat = value.view((count,), dtype=np.float64)
    destination = metal.empty((count,), dtype=np.float64)
    minus_two = metal.asarray(np.asarray([-2.0], dtype=np.float64))
    zero = metal.asarray(np.asarray([0.0], dtype=np.float64))
    context.submit_kernel(
        "global_semilag_add_scaled_fragment",
        {"base": flat, "increment": flat, "previous": flat,
         "current_scale": minus_two, "previous_scale": zero,
         "destination": destination},
        {"count": count}, exact_policy=False, grid_size=count,
        threads_per_group=(
            kernel_specs.GLOBAL_SEMILAG_ADD_SCALED_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    return destination.view(tuple(value.shape), dtype=np.float64)


def subtract_resident(value, other, context):
    """Subtract spectral fields with a resident Metal result."""
    from woof.metal.array import MetalArray, MetalArrayModule

    if not isinstance(value, MetalArray) and isinstance(other, MetalArray):
        if context is None:
            context = other.buffer.context
        if (other.buffer.context is not context
                or other.dtype != np.dtype(np.complex128)):
            raise TypeError(
                "resident Global subtraction requires matching complex128 Metal fields"
            )
        host = np.ascontiguousarray(value, dtype=np.complex128)
        if host.shape != other.shape:
            raise ValueError("Global subtraction operand shapes differ")
        value = MetalArrayModule(context).asarray(host)
    if not isinstance(value, MetalArray):
        return value - other
    if context is None:
        context = value.buffer.context
    if (value.buffer.context is not context
            or value.dtype != np.dtype(np.complex128)):
        raise TypeError(
            "resident Global subtraction requires matching complex128 Metal fields"
        )
    from woof.metal.generated import kernel_specs

    metal = MetalArrayModule(context)
    if isinstance(other, MetalArray):
        if (other.buffer.context is not context
                or other.dtype != np.dtype(np.complex128)
                or other.shape != value.shape):
            raise TypeError(
                "resident Global subtraction requires matching complex128 Metal fields"
            )
        rhs = other
    else:
        host = np.ascontiguousarray(other, dtype=np.complex128)
        if host.shape != value.shape:
            raise ValueError("Global subtraction operand shapes differ")
        rhs = metal.asarray(host)
    count = int(value.size * 2)
    lhs = value.view((count,), dtype=np.float64)
    rhs = rhs.view((count,), dtype=np.float64)
    destination = metal.empty((count,), dtype=np.float64)
    minus_one = metal.asarray(np.asarray([-1.0], dtype=np.float64))
    zero = metal.asarray(np.asarray([0.0], dtype=np.float64))
    context.submit_kernel(
        "global_semilag_add_scaled_fragment",
        {"base": lhs, "increment": rhs, "previous": rhs,
         "current_scale": minus_one, "previous_scale": zero,
         "destination": destination},
        {"count": count}, exact_policy=False, grid_size=count,
        threads_per_group=(
            kernel_specs.GLOBAL_SEMILAG_ADD_SCALED_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    return destination.view(tuple(value.shape), dtype=np.complex128)


def average_resident(left, right, context):
    """Average equal-shaped Metal fields through the resident FP64 kernel."""
    from woof.metal.array import MetalArray, MetalArrayModule

    if not isinstance(left, MetalArray) and not isinstance(right, MetalArray):
        return 0.5 * (left + right)
    resident = left if isinstance(left, MetalArray) else right
    if context is None:
        context = resident.buffer.context
    if resident.buffer.context is not context or resident.dtype != np.dtype(np.float64):
        raise TypeError("Global resident averages require FP64 Metal fields")
    metal = MetalArrayModule(context)

    def as_resident(value):
        if isinstance(value, MetalArray):
            if (value.buffer.context is not context
                    or value.dtype != np.dtype(np.float64)
                    or value.shape != resident.shape):
                raise TypeError("Global average operands must share FP64 Metal layout")
            return value
        host = np.ascontiguousarray(value, dtype=np.float64)
        if host.shape != resident.shape:
            raise ValueError("Global average operand shapes differ")
        return metal.asarray(host)

    lhs = as_resident(left)
    rhs = as_resident(right)
    from woof.metal.generated import kernel_specs

    count = int(resident.size)
    lhs_flat = lhs.view((count,), dtype=np.float64)
    rhs_flat = rhs.view((count,), dtype=np.float64)
    half = metal.asarray(np.asarray([0.5], dtype=np.float64))
    destination = metal.empty((count,), dtype=np.float64)
    context.submit_kernel(
        "global_semilag_add_scaled_fragment",
        {"base": lhs_flat, "increment": rhs_flat, "previous": lhs_flat,
         "current_scale": half, "previous_scale": half,
         "destination": destination},
        {"count": count}, exact_policy=False, grid_size=count,
        threads_per_group=(
            kernel_specs.GLOBAL_SEMILAG_ADD_SCALED_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    return destination.view(tuple(resident.shape), dtype=np.float64)


def reference_theta_faces(operator) -> np.ndarray:
    """The interface reference potential temperatures the vertical-mode
    operator built, ``(nlev+1,)``, zero at both boundaries.

    Recomputed from the operator's own ``theta_ref`` and pressures by the
    same limited reconstruction ``VerticalStructureOperator._build`` uses,
    rather than reaching into that method for a local: the operator is
    built once per (stack, reference) pair and cached, so this runs once
    per run, and a reconstruction that drifted from ``_build``'s would be
    caught by the unit test that rebuilds ``gamma`` from this array.
    """
    theta_ref = np.asarray(operator.theta_ref, dtype=np.float64)
    p_full = np.asarray(operator.p_full, dtype=np.float64)
    p_half = np.asarray(operator.p_half, dtype=np.float64)
    nlev = int(theta_ref.shape[0])
    gradient = np.zeros(nlev)
    above = (theta_ref[1:-1] - theta_ref[:-2]) / (p_full[1:-1] - p_full[:-2])
    below = (theta_ref[2:] - theta_ref[1:-1]) / (p_full[2:] - p_full[1:-1])
    product = above * below
    monotone = product > 0.0
    gradient[1:-1] = np.where(
        monotone, 2.0 * product / np.where(monotone, above + below, 1.0), 0.0
    )
    lower = np.minimum(theta_ref[:-1], theta_ref[1:])
    upper = np.maximum(theta_ref[:-1], theta_ref[1:])
    face_above = theta_ref[:-1] + gradient[:-1] * (p_half[1:nlev] - p_full[:-1])
    face_below = theta_ref[1:] + gradient[1:] * (p_half[1:nlev] - p_full[1:])
    face_above = np.minimum(np.maximum(face_above, lower), upper)
    face_below = np.minimum(np.maximum(face_below, lower), upper)
    face = np.zeros(nlev + 1)
    face[1:nlev] = 0.5 * (face_above + face_below)
    return face


class ReferenceProfile:
    """The reference column the semi-implicit operator linearizes about,
    built once per model: the operator itself, and its profile and
    interface values on the host for the gates that read them."""

    def __init__(self, model):
        operator = model.semi_implicit.operator(model.vertical)
        self.operator = operator
        self.theta_ref_host = np.asarray(operator.theta_ref, dtype=np.float64)
        self.face_host = reference_theta_faces(operator)
        self._metal_operator_resources = None
        self._metal_coriolis = None

    def resident_coriolis(self, context, coriolis, shape):
        """Keep the fixed surface Coriolis plane resident for SL tendencies."""
        shape = tuple(int(value) for value in shape)
        cached = self._metal_coriolis
        if (cached is not None and cached[0] is context
                and cached[1] == shape):
            return cached[2]

        from woof.metal.array import MetalArrayModule

        host = np.broadcast_to(
            np.asarray(coriolis, dtype=np.float64), shape
        )
        value = MetalArrayModule(context).asarray(
            np.ascontiguousarray(host)
        )
        self._metal_coriolis = (context, shape, value)
        return value

    def resident_operator_resources(self, context, backend, weight):
        """Stage the fixed linear-operator matrices once per Metal context."""
        key = (context, backend.name, np.dtype(backend.float_dtype), float(weight))
        cached = self._metal_operator_resources
        if cached is not None and cached[0] == key:
            return cached[1]

        from woof.metal.array import MetalArrayModule

        b_matrix, c_matrix, _structure = self.operator.device_matrices(backend)
        metal = MetalArrayModule(context)
        resources = {
            "b_matrix": metal.asarray(np.ascontiguousarray(
                b_matrix, dtype=np.float64)).view(
                    (b_matrix.size,), dtype=np.float64
                ),
            "c_matrix": metal.asarray(np.ascontiguousarray(
                c_matrix, dtype=np.float64)).view(
                    (c_matrix.size,), dtype=np.float64
                ),
            "weight": metal.asarray(np.asarray([weight], dtype=np.float64)),
        }
        self._metal_operator_resources = (key, resources)
        return resources


def linear_grid_tendencies(model, state, reference: ReferenceProfile):
    """The semi-implicit operator ``w L y`` where the trajectory needs it.

    Returns ``(phi_l_spectral, l_theta, l_lnps)``: the linearized
    geopotential-plus-gas-constant combination ``B x`` in the spectral
    basis (the momentum rows are its negated gradient, which the caller
    takes together with the geopotential's so both ride one contraction),
    and the thermodynamic rows ``C D`` on the grid.
    """
    transform = model.transform
    backend = transform.backend
    xp = backend.xp
    operator = reference.operator
    b_matrix, c_matrix, _structure = operator.device_matrices(backend)
    weight = backend.float_dtype(model.semi_implicit.divergence_weight)
    from woof.metal.array import MetalArray, MetalArrayModule
    if any(isinstance(value, MetalArray) for value in (
        state.theta, state.log_surface_pressure, state.divergence
    )):
        context = getattr(model, "metal_diffusion_context", None)
        if context is None or not all(isinstance(value, MetalArray) for value in (
            state.theta, state.log_surface_pressure, state.divergence
        )):
            raise TypeError("resident Global SL operator inputs must share Metal")
        nlevel, ndegree, mcount = map(int, state.theta.shape)
        expected = (ndegree, mcount)
        if (state.log_surface_pressure.shape != expected
                or state.divergence.shape != state.theta.shape):
            raise ValueError("resident Global SL operator input shapes do not match")
        if any(value.buffer.context is not context
               or value.dtype != np.dtype(np.complex128)
               for value in (state.theta, state.log_surface_pressure,
                             state.divergence)):
            raise TypeError("resident Global SL operator inputs must be complex128")
        from woof.metal.generated import kernel_specs
        metal = MetalArrayModule(context)
        phi_l = metal.empty(state.theta.shape, dtype=np.complex128)
        x_t = metal.empty((nlevel + 1, ndegree, mcount), dtype=np.complex128)
        coefficients = ndegree * mcount
        resources = {
            "theta": state.theta.view((2 * state.theta.size,), dtype=np.float64),
            "logps": state.log_surface_pressure.view(
                (2 * state.log_surface_pressure.size,), dtype=np.float64
            ),
            "divergence": state.divergence.view(
                (2 * state.divergence.size,), dtype=np.float64
            ),
            **reference.resident_operator_resources(context, backend, weight),
            "phi_l": phi_l.view((2 * phi_l.size,), dtype=np.float64),
            "x_t": x_t.view((2 * x_t.size,), dtype=np.float64),
        }
        context.submit_kernel(
            "global_semilag_operator_fragment", resources,
            {"nlevel": nlevel, "ndegree": ndegree, "mcount": mcount},
            exact_policy=False, grid_size=(2 * nlevel + 1) * coefficients,
            threads_per_group=(
                kernel_specs.GLOBAL_SEMILAG_OPERATOR_FRAGMENT_THREADS_PER_GROUP
            ),
        )
        grid = transform.inverse(x_t, keep_resident=True)
        return phi_l, grid[:nlevel], grid[nlevel]
    x = xp.concatenate(
        [state.theta, state.log_surface_pressure[None]], axis=0
    )
    phi_l = weight * xp.einsum("ij,jnm->inm", b_matrix, x)
    del x
    x_t = weight * xp.einsum("ij,jnm->inm", c_matrix, state.divergence)
    grid = transform.inverse(transform.project(x_t))
    del x_t
    return transform.project(phi_l), grid[:-1], grid[-1]


def advective_tendencies(
    model, state, g, omega_half, ps_t, reference: ReferenceProfile,
    *, bundle_lead=None, bundle_half=None,
):
    """The material tendencies and the linear operator, both on the grid.

    Returns a dict with ``a_u``, ``a_v``, ``a_lnps`` (the advective-form
    tendencies of the eastward and northward wind and of ``ln ps``; theta
    has none, see the module docstring) and ``l_u``, ``l_v``, ``l_theta``,
    ``l_lnps`` (the rows of ``w L y``).  They are returned apart rather
    than already differenced because the departure-side bundle needs ``L``
    on its own.
    """
    transform = model.transform
    xp = transform.backend.xp
    phi_l_spectral, l_theta, l_lnps = linear_grid_tendencies(
        model, state, reference
    )
    from woof.metal.array import MetalArray
    if isinstance(phi_l_spectral, MetalArray):
        geopotential_spectral = transform.forward(
            g["geopotential"], keep_resident=True
        )
        grad_phi_east, grad_phi_north = transform.gradient(
            geopotential_spectral
        )
        l_east, l_north = transform.gradient(phi_l_spectral)
        east = (grad_phi_east, l_east)
        north = (grad_phi_north, l_north)
    else:
        geopotential_spectral = transform.project(
            transform.forward(g["geopotential"])
        )
        east, north = transform.gradient(
            xp.stack([geopotential_spectral, phi_l_spectral])
        )
        grad_phi_east, grad_phi_north = east[0], north[0]
        l_east, l_north = east[1], north[1]
    del geopotential_spectral, phi_l_spectral
    sign_context = getattr(model, "metal_diffusion_context", None)
    l_u = _negate_resident(l_east, sign_context)
    l_v = _negate_resident(l_north, sign_context)

    grad_lnps_east, grad_lnps_north = transform.gradient(
        state.log_surface_pressure
    )
    factor = model._pressure_gradient_factor(g["ps"], g["p_half"])
    if isinstance(g["virtual_temperature"], MetalArray):
        context = getattr(model, "metal_diffusion_context", None)
        if context is None:
            raise TypeError("resident Global force requires a Metal context")
        from woof.metal.array import MetalArrayModule
        from woof.metal.generated import kernel_specs

        metal = MetalArrayModule(context)
        shape = tuple(g["virtual_temperature"].shape)
        count = int(g["virtual_temperature"].size)
        tv = g["virtual_temperature"]
        if tv.buffer.context is not context or tv.dtype != np.dtype(np.float64):
            raise TypeError("resident Global virtual temperature must be FP64 Metal")
        if isinstance(factor, MetalArray):
            if (factor.buffer.context is not context
                    or factor.dtype != np.dtype(np.float64)
                    or tuple(factor.shape) != shape):
                raise TypeError("resident Global pressure factor must match temperature")
            factor_flat = factor.view((count,), dtype=np.float64)
        else:
            factor_host = np.ascontiguousarray(factor, dtype=np.float64)
            if tuple(factor_host.shape) != shape:
                raise ValueError("Global pressure factor shape differs from temperature")
            factor_flat = metal.asarray(factor_host.reshape(-1))
        force_array = metal.empty((count,), dtype=np.float64)
        gas_constant = metal.asarray(np.asarray([model.gas_constant], dtype=np.float64))
        context.submit_kernel(
            "global_semilag_force_fragment",
            {"virtual_temperature": tv.view((count,), dtype=np.float64),
             "factor": factor_flat, "gas_constant": gas_constant,
             "force": force_array},
            {"count": count}, exact_policy=False, grid_size=count,
            threads_per_group=(
                kernel_specs.GLOBAL_SEMILAG_FORCE_FRAGMENT_THREADS_PER_GROUP
            ),
        )
        force = force_array.view(shape, dtype=np.float64)
    else:
        force = model.gas_constant * g["virtual_temperature"] * factor
    del factor
    assembled = None
    if getattr(model, "metal_diffusion_context", None) is not None and xp is np:
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.generated import kernel_specs

        context = model.metal_diffusion_context
        metal = MetalArrayModule(context)
        shape = tuple(g["u"].shape)
        nlevel, nlat, nlon = shape
        npoint = int(np.prod(shape))
        surface_shape = shape[1:]

        def upload(value, target_shape):
            if isinstance(value, MetalArray):
                if value.buffer.context is not context:
                    raise ValueError(
                        "Global SL tendency input belongs to another Metal context"
                    )
                if value.dtype != np.dtype(np.float64):
                    raise TypeError("Global SL tendency Metal inputs must be FP64")
                if tuple(value.shape) != tuple(target_shape):
                    raise ValueError(
                        f"resident Global SL tendency input is {value.shape}, "
                        f"expected {tuple(target_shape)}"
                    )
                result = metal.asarray(value)
                return (result.view((result.size,), dtype=np.float64)
                        if result.ndim != 1 else result)
            host = np.broadcast_to(np.asarray(value, dtype=np.float64), target_shape)
            return metal.asarray(np.ascontiguousarray(host.reshape(-1)))

        resources = {
            "coriolis": upload(
                reference.resident_coriolis(
                    context, model.coriolis, surface_shape
                ),
                surface_shape,
            ),
            "u": upload(g["u"], shape), "v": upload(g["v"], shape),
            "grad_phi_east": upload(grad_phi_east, shape),
            "grad_phi_north": upload(grad_phi_north, shape),
            "force": upload(force, shape),
            "grad_lnps_east": upload(grad_lnps_east, surface_shape),
            "grad_lnps_north": upload(grad_lnps_north, surface_shape),
            "ps_t": upload(ps_t, surface_shape), "ps": upload(g["ps"], surface_shape),
            "a_u": metal.empty((npoint,), dtype=np.float64),
            "a_v": metal.empty((npoint,), dtype=np.float64),
            "a_lnps": metal.empty((nlat * nlon,), dtype=np.float64),
        }
        context.submit_kernel(
            "global_semilag_tendencies_fragment", resources,
            {"nlevel": nlevel, "nlat": nlat, "nlon": nlon},
            exact_policy=False, grid_size=npoint,
            threads_per_group=kernel_specs.GLOBAL_SEMILAG_TENDENCIES_FRAGMENT_THREADS_PER_GROUP,
        )
        if bundle_lead is not None and bundle_half is not None:
            bundle_resources = {
                "u": resources["u"], "v": resources["v"],
                "theta": upload(g["theta"], shape),
                "logps": upload(g["logps"], surface_shape),
                "a_u": resources["a_u"], "a_v": resources["a_v"],
                "a_lnps": resources["a_lnps"],
                "l_u": upload(l_u, shape), "l_v": upload(l_v, shape),
                "l_theta": upload(l_theta, shape),
                "l_lnps": upload(l_lnps, surface_shape),
                "lead": metal.asarray(np.asarray([bundle_lead], dtype=np.float64)),
                "half_dt": metal.asarray(np.asarray([bundle_half], dtype=np.float64)),
                "n_u": metal.empty((npoint,), dtype=np.float64),
                "n_v": metal.empty((npoint,), dtype=np.float64),
                "n_theta": metal.empty((npoint,), dtype=np.float64),
                "n_lnps": metal.empty((nlat * nlon,), dtype=np.float64),
                "bundle_u": metal.empty((npoint,), dtype=np.float64),
                "bundle_v": metal.empty((npoint,), dtype=np.float64),
                "bundle_theta": metal.empty((npoint,), dtype=np.float64),
                "bundle_lnps": metal.empty((nlat * nlon,), dtype=np.float64),
            }
            context.submit_kernel(
                "global_semilag_assemble_fragment", bundle_resources,
                {"nlevel": nlevel, "nlat": nlat, "nlon": nlon},
                exact_policy=False, grid_size=npoint,
                threads_per_group=(
                    kernel_specs.GLOBAL_SEMILAG_ASSEMBLE_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            n_u = bundle_resources["n_u"].view(shape, dtype=np.float64)
            n_v = bundle_resources["n_v"].view(shape, dtype=np.float64)
            n_theta = bundle_resources["n_theta"].view(shape, dtype=np.float64)
            n_lnps = bundle_resources["n_lnps"].view(surface_shape, dtype=np.float64)
            assembled = {
                "n_u": n_u, "n_v": n_v, "n_theta": n_theta,
                "n_lnps": n_lnps,
                **{name: bundle_resources[name].view(target_shape, dtype=np.float64)
                   for name, target_shape in (
                       ("bundle_u", shape), ("bundle_v", shape),
                       ("bundle_theta", shape), ("bundle_lnps", surface_shape),
                   )},
            }
            a_u = resources["a_u"].view(shape, dtype=np.float64)
            a_v = resources["a_v"].view(shape, dtype=np.float64)
            a_lnps = resources["a_lnps"].view(surface_shape, dtype=np.float64)
        else:
            a_u = resources["a_u"].to_numpy().reshape(shape)
            a_v = resources["a_v"].to_numpy().reshape(shape)
            a_lnps = resources["a_lnps"].to_numpy().reshape(surface_shape)
    else:
        a_u = model.coriolis * g["v"] - grad_phi_east - force * grad_lnps_east
        a_v = -model.coriolis * g["u"] - grad_phi_north - force * grad_lnps_north
        # ln ps is tied to the lowest-level trajectory and includes the
        # Eulerian continuity closure plus horizontal advection.
        a_lnps = (
            ps_t / g["ps"] + g["u"][-1] * grad_lnps_east
            + g["v"][-1] * grad_lnps_north
        )
    del force, grad_phi_east, grad_phi_north, east, north

    # ln ps rides the LOWEST model level's trajectory, so its material
    # tendency is the Eulerian one plus the advection by that level's
    # wind.  The Eulerian part is the continuity closure the shipped core
    # already diagnoses, so both arms of the model see one surface
    # pressure tendency rather than two discretizations of it.
    del grad_lnps_east, grad_lnps_north

    # ``Dtheta/Dt`` is zero for adiabatic flow and the whole variable is
    # gathered (module docstring), so there is no ``a_theta`` row: the
    # step's nonlinear residual for theta is ``-l_theta``.
    return {
        "a_u": a_u, "a_v": a_v, "a_lnps": a_lnps,
        "l_u": l_u, "l_v": l_v, "l_theta": l_theta, "l_lnps": l_lnps,
        **(assembled or {}),
    }


__all__ = [
    "ReferenceProfile",
    "advective_tendencies",
    "linear_grid_tendencies",
    "reference_theta_faces",
]
