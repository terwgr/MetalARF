"""Semi-Lagrangian transport of the ten grid tracers, and its mass fixer.

The condensate species and number moments ride the SAME stencil as the
wind, the thermodynamic variable and the vapour: one departure-point
search and one index-and-weight computation serve every field of the
step, so ten tracers are ten more rows in a batched gather rather than a
second transport scheme with a second time step of its own.

That matters arithmetically as well as in cost.  The shipped flux-form
sweep is sub-cycled to a per-face Courant number of 0.25, and on the
outermost T255 ring, which is 326 m wide, a 10 m/s zonal wind at
dt = 300 s has a Courant number of 9.2 and needs 37 sub-steps; at T533
the same wind on a 75 m ring needs 160.  The flux form does not survive
the step this integrator exists to take.  It is not deleted, and
``transport.py`` is not edited: ``[semilag] tracer_scheme = "flux_form"``
still routes the tracers through it for the arm that compares the two.

What a semi-Lagrangian gather does not do is conserve mass.  It reads a
MIXING RATIO at a point; the layer thickness that mixing ratio is
measured against has changed underneath it, and nothing in the
interpolation knows that.  On top of that the quasi-monotone limiter cuts
the cubic's overshoot at every sharp maximum, and a condensate species is
almost all sharp maxima.

Both effects are MEASURED, on the T255 native arms of 2026-09-06 (the 16 GB host,
dt = 300 s, the full native suite).  The two arm sets run the tree's two
cases and the numbers below are not two points on one run: the six-hour
arms are the GDAS 2026-08-30 18Z case
(configs/verify/arwen_global_gdas_t255_native_sl_si_6h.toml) and the
forecast day is the GDAS 2026-09-01 00Z case.  The largest per-step
relative mass the fixer had to move was 5.96e-2 over six hours and 1.01e-1
over the forecast day, and the six-hour arm with the limiter TURNED OFF
reads 6.43e-2, which is the same number as the six-hour arm with it on.

That equality is the finding, and it took fixing the defect below to see
it.  The mass is not the limiter's; it is POSITIVITY's.  A cubic through a
field that is zero at three of its four stencil points undershoots below
zero on the shoulder of every maximum, and something has to lift it back,
because a negative mixing ratio is not a state this model has.  Turning
the limiter off does not stop the lift, it moves it: the negatives arrive
at the fixer's own floor instead of at the gather's clip, and the same
mass is created either way.  Turning BOTH off does stop it, and the run is
then refused at its first step on a cloud water of -1.49e-5 against a
maximum of 1.02e-3.

So the fixer's magnitude is the ADVECTION's error, it is the same to two
figures under every form below, and what separates the forms is not how
much they move but where they put it:

``bermejo_conde_additive``
    Two stages.  First the clip deficit, the signed mass the limiter moved
    at every point, which the gather reports at no extra arithmetic: the
    correction goes back exactly there, scaled by one global number and
    capped so it can never over-restore past the raw interpolated value or
    drive a point negative.  Whatever is left over closes multiplicatively
    as below.  MEASURED on the same arms: that stage carries 1 to 6
    percent of the correction under the default physics coupling and 100
    percent under ``physics_coupling = "arrival"``, where the correction is
    an ADDITION and the deficit points the right way.  The reason it is
    small under the default is the direction above: the limiter's mass came
    from lifting points to zero, and a point that now holds exactly zero
    cannot give any of it back without going negative.
``bermejo_conde`` (default)
    The multiplicative form alone: one uniform RELATIVE adjustment of the
    air that holds the species, weighted by the mass already present.
    Conservative, and it reshapes the field by its own weighting rather
    than by the flow.  On a non-negative field it IS the additive form
    with mass-proportional weights, because ``q (1 + d/W)`` is
    ``q + d q/W``; what the additive stage above adds is not a different
    algebra but a different WEIGHT, the clip deficit in place of the
    mass.  It is the default because the additive stage's
    measured share does not pay for the second device array per gathered
    tracer that reporting the deficit costs; it ships selectable with that
    number rather than being left unbuilt.
``proportional`` is RETIRED and refused by name.  It weighted the
correction by the signed advected value where ``bermejo_conde`` weights
it by the positive part, and once the positivity floor moved in front of
the mass measurement the two weightings became the same array at every
point of every field: MEASURED 2026-09-06 on the numpy path, bitwise
identical output with the limiter on AND off, on a spiky field and on a
cloudy one.  A door whose two values produce the same run and two
different config hashes is the flag-parsed-and-ignored failure this
package refuses everywhere else, so it is refused here rather than left
selectable.

All three floor the advected field at zero BEFORE they measure it, and
report the mass that floor created per species.  A negative mixing ratio
is not a state the model has, so the floor is not optional; measuring the
mass after it is what makes the correction close.

The refusal in the middle of both names its breakage.  A species whose
entire mass has left the model cannot have that mass restored by a
weighted fixer, and a fixer that fell back to spreading it uniformly
would create the mass in clear air, which is precisely the failure the
level-wide spectral rescale used to have: 37 percent of the planet's
cloud water moved out of its columns on every positivity pass, four per
step, and grid-scale precipitation never reached the ground (the pin
document's ``water_repair`` v6 entry).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .interpolate import Stencil, gather_batch
from ..spill import resident

#: The fixer forms the ``[semilag] tracer_fixer`` door accepts.  ``none``
#: reports the drift and closes nothing; it is the counter-arm that says
#: what the fixer is worth, not a shipping choice.
TRACER_FIXERS = ("bermejo_conde_additive", "bermejo_conde", "none")

#: Names the door once accepted and now refuses, each with the reason.
#: A retired name is refused rather than dropped, because a config that
#: still carries it would otherwise run under a form it did not ask for.
RETIRED_TRACER_FIXERS = {
    "proportional": (
        "it weighted the correction by the signed advected value where "
        "'bermejo_conde' weights it by the positive part, and every "
        "conservative form now floors the field at zero BEFORE it measures "
        "the mass, so the two weightings are the same array at every point: "
        "MEASURED 2026-09-06, bitwise identical output with the "
        "quasi-monotone limiter on and off, on a spiky field and on a cloudy "
        "one.  Selecting it produced the same run as 'bermejo_conde' under a "
        "different config hash, which is the parsed-and-ignored failure this "
        "package refuses everywhere else"
    ),
}

#: The forms that read the limiter's clip deficit and therefore need the
#: gather to report it.
DEFICIT_FIXERS = ("bermejo_conde_additive",)


def area_weights(transform):
    """Fractional area of one grid cell, summing to one over the sphere.

    The same convention ``_transport_grid_tracers`` uses for its own
    global reductions, so a mass computed here and a mass computed there
    are the same number.
    """
    backend = transform.backend
    return backend.asarray(
        transform.grid.quadrature_weights, dtype=backend.float_dtype
    )[None, :, None] / (2.0 * transform.grid.nlon)


def _mass(xp, field, dp, cell) -> float:
    return float(xp.sum(field * dp * cell))


def advect(
    tracers: dict[str, Any], stencil: Stencil, *, monotone: bool = True,
    batch: int = 8, deficit: bool = False,
) -> dict[str, Any]:
    """Every grid tracer read at the stencil's departure points.

    With ``deficit=True`` each value is a ``(field, clip_deficit)`` pair.
    """
    names = list(tracers)
    if not names:
        return {}
    values = gather_batch(
        [tracers[name] for name in names], stencil,
        monotone=monotone, batch=int(batch), deficit=bool(deficit),
    )
    return dict(zip(names, values))


def _refuse_empty(name: str, delta: float) -> None:
    raise ValueError(
        f"the semi-Lagrangian mass fixer cannot restore {abs(delta):.6g} "
        f"kg/m2 of {name}: the advected field holds no mass anywhere on "
        "the sphere, so there is no air to put it back in, and spreading "
        "it uniformly would create the species in clear air"
    )


def fix_mass(
    advected: dict[str, Any], before: dict[str, Any], dp_before, dp_after,
    transform, *, scheme: str = "bermejo_conde_additive",
    deficits: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Restore each species' mass after a semi-Lagrangian advection.

    ``before`` are the mixing ratios the step started from and
    ``dp_before`` the thickness they were measured against; ``advected``
    are the same species at the departure points and ``dp_after`` the
    thickness of the layers they now sit in.  ``deficits`` carries the
    limiter's signed clip amount per species and is required by, and only
    read by, the additive form.

    Returns the fixed tracers and, per species, the relative mass the
    fixer moved, that mass signed (positive where it put mass back), and
    the fraction of it the clip deficit carried.
    """
    if scheme in RETIRED_TRACER_FIXERS:
        raise ValueError(
            f"the semi-Lagrangian tracer fixer {scheme!r} is retired: "
            + RETIRED_TRACER_FIXERS[scheme]
        )
    if scheme not in TRACER_FIXERS:
        raise ValueError(
            "the semi-Lagrangian tracer fixer must be one of "
            + ", ".join(repr(item) for item in TRACER_FIXERS)
            + f", got {scheme!r}"
        )
    additive = scheme == "bermejo_conde_additive"
    if additive and deficits is None:
        raise ValueError(
            "the additive tracer fixer puts a species' mass back where the "
            "quasi-monotone limiter took it, so it cannot run without the "
            "limiter's clip deficit; gather with deficit=True or select "
            "tracer_fixer = 'bermejo_conde'"
        )
    xp = transform.backend.xp
    cell = area_weights(transform)
    metrics: dict[str, float] = {}
    fixed: dict[str, Any] = {}
    from woof.metal.array import MetalArray, MetalArrayModule
    metal_values = [value for value in advected.values()
                    if isinstance(value, MetalArray)]
    if metal_values:
        context = metal_values[0].buffer.context
        if (scheme == "bermejo_conde" and deficits is None
                and len(metal_values) == len(advected)):
            from woof.metal.generated import kernel_specs

            metal = MetalArrayModule(context)
            nlevel, nlat, nlon = map(int, metal_values[0].shape)
            columns = nlat * nlon

            def resource(value, shape=None):
                if isinstance(value, MetalArray):
                    if value.buffer.context is not context:
                        raise ValueError("Global tracer mass input uses another Metal context")
                    if value.dtype != np.dtype(np.float64):
                        raise TypeError("Global tracer mass input must be FP64")
                    resident = metal.asarray(value)
                else:
                    host = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
                    resident = metal.asarray(host.reshape(-1))
                if shape is not None and resident.size != int(np.prod(shape)):
                    raise ValueError("Global tracer mass input has an unexpected shape")
                return resident.view((resident.size,), dtype=np.float64)

            def scalar_sums(*partial_vectors):
                """Finish compact per-column sums on Metal, then read scalars."""
                columns = int(partial_vectors[0].size)
                packed = metal.empty(
                    (len(partial_vectors) * columns,), dtype=np.float64
                )
                copies = []
                for index, partial in enumerate(partial_vectors):
                    copies.append((
                        partial,
                        packed.view(
                            (columns,), dtype=np.float64,
                            byte_offset=packed.byte_offset + index * columns * 8,
                        ),
                    ))
                context.submit_bookkeeping(copies)
                totals = metal.empty((len(partial_vectors),), dtype=np.float64)
                context.submit_kernel(
                    "global_reductions_fragment",
                    {"values": packed, "partial_sums": totals},
                    {"count": len(partial_vectors), "chunk_size": columns},
                    exact_policy=False, grid_size=len(partial_vectors),
                    threads_per_group=(
                        kernel_specs.GLOBAL_REDUCTIONS_FRAGMENT_THREADS_PER_GROUP
                    ),
                )
                return totals.to_numpy()

            cell_rows = np.ascontiguousarray(
                np.asarray(cell, dtype=np.float64).reshape(1, nlat, 1)[0, :, 0]
            )
            cell_resource = metal.asarray(cell_rows)
            for name, value in advected.items():
                target_partial = metal.empty((columns,), dtype=np.float64)
                after_partial = metal.empty((columns,), dtype=np.float64)
                clamp_partial = metal.empty((columns,), dtype=np.float64)
                context.submit_kernel(
                    "global_semilag_tracer_mass_fragment",
                    {"before": resource(before[name], (nlevel, nlat, nlon)),
                     "dp_before": resource(dp_before, (nlevel, nlat, nlon)),
                     "advected": resource(value, (nlevel, nlat, nlon)),
                     "dp_after": resource(dp_after, (nlevel, nlat, nlon)),
                     "cell_weight": cell_resource,
                     "target_partial": target_partial,
                     "after_partial": after_partial,
                     "clamp_partial": clamp_partial},
                    {"columns": columns, "nlevel": nlevel, "nlon": nlon},
                    exact_policy=False, grid_size=columns,
                    threads_per_group=(
                        kernel_specs.GLOBAL_SEMILAG_TRACER_MASS_FRAGMENT_THREADS_PER_GROUP
                    ),
                )
                target, after, clamped = scalar_sums(
                    target_partial, after_partial, clamp_partial
                )
                target, after, clamped = map(float, (target, after, clamped))
                delta = target - after
                metrics[f"semilag_tracer_positivity_clamp_kg_m2__{name}"] = float(clamped)
                metrics[f"semilag_tracer_mass_fixer_relative__{name}"] = (
                    abs(delta) / max(abs(target), 1.0e-30)
                )
                metrics[f"semilag_tracer_mass_fixer_kg_m2__{name}"] = float(delta)
                metrics[f"semilag_tracer_clip_share__{name}"] = 0.0
                if delta == 0.0:
                    fixed[name] = value
                    continue
                if after == 0.0:
                    _refuse_empty(name, delta)
                scale = 1.0 + delta / after
                result = metal.empty(value.shape, dtype=np.float64)
                context.submit_kernel(
                    "global_semilag_tracer_scale_fragment",
                    {"advected": resource(value),
                     "scale": metal.asarray(np.asarray([scale], dtype=np.float64)),
                     "fixed": result.view((result.size,), dtype=np.float64)},
                    {"count": int(value.size)}, exact_policy=False,
                    grid_size=int(value.size),
                    threads_per_group=(
                        kernel_specs.GLOBAL_SEMILAG_TRACER_SCALE_FRAGMENT_THREADS_PER_GROUP
                    ),
                )
                fixed[name] = result
            return fixed, metrics
        def host(value):
            return value.to_numpy() if isinstance(value, MetalArray) else value
        advected = {name: host(value) for name, value in advected.items()}
        before = {name: host(value) for name, value in before.items()}
        dp_before, dp_after = host(dp_before), host(dp_after)
        deficits = ({name: host(value) for name, value in deficits.items()}
                    if deficits is not None else None)
    for name, value in advected.items():
        # The before-state may be the tier's (a parked tracer): staged
        # one species at a time for its mass and dropped.
        target = _mass(xp, resident(xp, before[name]), dp_before, cell)
        clamped = 0.0
        if scheme != "none":
            # The floor comes FIRST, and its mass is measured.
            #
            # Every conservative form here ends by clamping the mixing
            # ratio at zero, because a negative one is not a state the
            # model has.  Measuring the advected mass before that clamp
            # and correcting afterwards leaves the correction short by
            # exactly the negative part: the fixer then returns a field
            # whose mass is the target PLUS the mass the clamp created,
            # reports a small correction, and closes nothing.  It is
            # invisible while the limiter is on, because the gather is
            # then nonnegative and the clamp does nothing; it is the whole
            # error of the arm that runs with the limiter off, and that
            # arm is the counter-arm the limiter's value is read from.
            floored = xp.maximum(value, 0.0)
            clamped = (_mass(xp, floored, dp_after, cell)
                       - _mass(xp, value, dp_after, cell))
            value = floored
        metrics[f"semilag_tracer_positivity_clamp_kg_m2__{name}"] = float(clamped)
        after = _mass(xp, value, dp_after, cell)
        delta = target - after
        relative = abs(delta) / max(abs(target), 1.0e-30)
        metrics[f"semilag_tracer_mass_fixer_relative__{name}"] = float(relative)
        # SIGNED: positive where the fixer had to put mass back, negative
        # where it had to take mass away.  The two directions cancel across
        # species in the water budget and must not be added as magnitudes
        # there; the per-species accuracy row reads the relative number
        # above, which is a magnitude on purpose.
        metrics[f"semilag_tracer_mass_fixer_kg_m2__{name}"] = float(delta)
        metrics[f"semilag_tracer_clip_share__{name}"] = 0.0
        if scheme == "none" or delta == 0.0:
            fixed[name] = value
            continue
        sign = 1.0 if delta > 0.0 else -1.0
        if additive:
            # Stage one.  The limiter's own record of where the mass went:
            # the deficit is raw minus kept, so a positive entry is mass
            # the clip removed and a negative entry mass it added.  Take
            # only the entries pointing the way the correction has to go,
            # and never take more from a point than it holds, so the stage
            # cannot drive a mixing ratio negative.
            room = xp.maximum(sign * deficits[name], 0.0)
            if sign < 0.0:
                room = xp.minimum(room, xp.maximum(value, 0.0))
            capacity = _mass(xp, room, dp_after, cell)
            if capacity > 0.0:
                # Capped at one: restoring the whole clip amount returns
                # the raw interpolated value, and past it the fixer would
                # be undoing more than the limiter ever did.
                lam = min(abs(delta) / capacity, 1.0)
                value = value + (sign * lam) * room
                moved = sign * lam * capacity
                metrics[f"semilag_tracer_clip_share__{name}"] = float(
                    abs(moved) / max(abs(delta), 1.0e-30)
                )
                delta = delta - moved
            del room
            if delta == 0.0:
                fixed[name] = xp.maximum(value, 0.0)
                continue
            sign = 1.0 if delta > 0.0 else -1.0
        # Stage two, and the whole of the multiplicative forms.  Weights
        # proportional to the mass already present, zero in air that holds
        # none of the species: the correction is a uniform RELATIVE
        # adjustment of the air that has it.  max(q, 0) rather than
        # abs(q) so a roundoff-negative point cannot be handed positive
        # mass.
        # ``value`` is already nonnegative (the floor above), so the
        # weights are the mass present and the scaled field cannot leave
        # the floor.  max(q, 0) rather than abs(q) is therefore an
        # identity here and is written as the floor instead.
        weight = _mass(xp, value, dp_after, cell)
        if weight == 0.0:
            _refuse_empty(name, delta)
        fixed[name] = value * (1.0 + delta / weight)
    return fixed, metrics


__all__ = ["DEFICIT_FIXERS", "RETIRED_TRACER_FIXERS", "TRACER_FIXERS",
           "advect", "area_weights", "fix_mass"]
