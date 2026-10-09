"""Departure points on the sphere, and the diagnostics that gate them.

The trajectory is solved in geocentric Cartesian coordinates.  There is no
rotation matrix, no local basis and therefore no polar special case: the
arrival point is a unit vector, the wind is three Cartesian components of a
tangent field, the trial departure point is the chord from the arrival point
projected back onto the sphere, and the vertical coordinate is the continuous
full-level index moved by the level-index rate.

The iteration, for arrival point ``r_A`` and level index ``k_A``::

    r_D^(m+1) = normalize( r_A - (dt/2) [ V_ex(r_A) + V^n(r_D^(m)) ] / a )
    k_D^(m+1) = k_A - (dt/2) [ s_ex(r_A,k_A) + s^n(r_D^(m), k_D^(m)) ]
    r_D^(0)   = normalize( r_A - dt V^n(r_A) / a )
    k_D^(0)   = k_A - dt s^n(r_A, k_A)

``V_ex`` and ``s_ex`` are the extrapolated fields, read at the arrival point
and so needing no interpolation at all; a caller with no previous time level
(the first step of a fresh run) passes none and gets ``V^n`` at the arrival
point in their place, which is the non-extrapolated two-time-level start-up.

Two diagnostics ship with the solver because the two ways a semi-Lagrangian
trajectory goes wrong are both silent:

*   :func:`convergence` measures the last iteration's move.  An unconverged
    trajectory reads as a Rossby phase error and mislocated cyclones and
    produces no other symptom, so the iteration count is measured rather
    than trusted.
*   :func:`lipschitz` measures the flow deformation.  Above one, the
    trajectory map is not invertible, two arrival points share a departure
    point, the fixed-point iteration stops contracting and the interpolation
    samples a folded field.  The model does not blow up when this happens.
    It silently mislocates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .interpolate import Stencil, _is_cupy
from .tables import SphericalGridTables


@dataclass(frozen=True)
class CartesianWind:
    """The advecting flow, in the form the trajectory reads it.

    ``vx, vy, vz`` are the geocentric Cartesian components of the horizontal
    wind in m/s at full levels; ``level_rate`` is the vertical velocity in
    continuous level indices per second, also at full levels.
    """

    vx: Any
    vy: Any
    vz: Any
    level_rate: Any

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.vx.shape)  # type: ignore[return-value]

    def arrays(self) -> tuple:
        return (self.vx, self.vy, self.vz, self.level_rate)


def _geometry(tables: SphericalGridTables, nlev: int, *, xp, dtype):
    nlat, nlon = tables.shape
    lat = xp.asarray(tables.lat_ext_host[2:nlat + 2], dtype=dtype)
    lon = xp.asarray(np.arange(nlon) * tables.dlam, dtype=dtype)
    sin_lat = xp.sin(lat)[None, :, None]
    cos_lat = xp.cos(lat)[None, :, None]
    sin_lon = xp.sin(lon)[None, None, :]
    cos_lon = xp.cos(lon)[None, None, :]
    return sin_lat, cos_lat, sin_lon, cos_lon


def cartesian_wind(
    u, v, tables: SphericalGridTables, *, xp=None, dtype=None,
    keep_resident: bool = False,
):
    """Local east-north wind as geocentric Cartesian components."""
    xp = tables.xp if xp is None else xp
    dtype = tables.dtype if dtype is None else dtype
    nlev = int(u.shape[0])
    context = tables.metal_context
    if (context is not None and xp is np
            and np.dtype(dtype) == np.dtype(np.float64)):
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.generated import kernel_specs

        shape = tuple(int(value) for value in u.shape)
        if tuple(int(value) for value in v.shape) != shape:
            raise ValueError("Global Cartesian wind inputs must have equal shapes")
        nlev, nlat, nlon = shape
        metal = MetalArrayModule(context)

        def resource(value):
            if isinstance(value, MetalArray):
                if value.buffer.context is not context:
                    raise ValueError(
                        "Global Cartesian wind input belongs to another Metal context"
                    )
                if value.dtype != np.dtype(np.float64):
                    raise TypeError("Global Cartesian wind inputs must be FP64")
                resident = metal.asarray(value)
                return (resident.view((resident.size,), dtype=np.float64)
                        if resident.ndim != 1 else resident)
            return metal.asarray(np.ascontiguousarray(
                np.asarray(value, dtype=np.float64).reshape(-1)))

        lat = tables.lat_ext_host[2:nlat + 2]
        lon = np.arange(nlon, dtype=np.float64) * tables.dlam
        count = nlev * nlat * nlon
        outputs = {name: metal.empty((count,), dtype=np.float64)
                   for name in ("vx", "vy", "vz")}
        resources = {
            "u": resource(u), "v": resource(v),
            "sinlat": metal.asarray(np.ascontiguousarray(np.sin(lat), dtype=np.float64)),
            "coslat": metal.asarray(np.ascontiguousarray(np.cos(lat), dtype=np.float64)),
            "sinlon": metal.asarray(np.ascontiguousarray(np.sin(lon), dtype=np.float64)),
            "coslon": metal.asarray(np.ascontiguousarray(np.cos(lon), dtype=np.float64)),
            **outputs,
        }
        context.submit_kernel(
            "global_semilag_cartesian_wind_fragment", resources,
            {"nlevel": nlev, "nlat": nlat, "nlon": nlon},
            exact_policy=False, grid_size=count,
            threads_per_group=(
                kernel_specs.GLOBAL_SEMILAG_CARTESIAN_WIND_FRAGMENT_THREADS_PER_GROUP
            ),
        )
        if keep_resident:
            return tuple(outputs[name].view(shape, dtype=np.float64)
                         for name in ("vx", "vy", "vz"))
        return tuple(outputs[name].to_numpy().reshape(shape)
                     for name in ("vx", "vy", "vz"))
    sph, cph, sla, cla = _geometry(tables, nlev, xp=xp, dtype=dtype)
    vx = -u * sla - v * sph * cla
    vy = u * cla - v * sph * sla
    vz = v * cph + xp.zeros_like(u)
    return vx, vy, vz


def local_wind(vx, vy, vz, tables: SphericalGridTables, *, xp=None, dtype=None):
    """Geocentric Cartesian components back to the local east-north wind."""
    xp = tables.xp if xp is None else xp
    dtype = tables.dtype if dtype is None else dtype
    nlev = int(vx.shape[0])
    sph, cph, sla, cla = _geometry(tables, nlev, xp=xp, dtype=dtype)
    u = -vx * sla + vy * cla
    v = -vx * sph * cla - vy * sph * sla + vz * cph
    return u, v


def level_rate_from_mass_flux(
    omega_half, dp, *, xp=None, metal_context=None, keep_resident: bool = False,
):
    """The vertical rate in level indices per second, at full levels.

    ``omega_half`` is the eta-coordinate vertical mass flux ``etadot dp/deta``
    in Pa/s at the ``nlev+1`` interfaces, zero at BOTH boundaries, which is
    what :func:`woof.globe.vertical.continuity` produces.  Dividing
    it by the thickness per unit level index gives the rate in level indices,
    and the zero boundary values mean no trajectory leaves the model through
    the lid or the ground by construction of the interior flow.
    """
    xp = np if xp is None else xp
    nlev = int(dp.shape[0])
    if int(omega_half.shape[0]) != nlev + 1:
        raise ValueError(
            f"omega_half must have {nlev + 1} interfaces for {nlev} layers, "
            f"got {int(omega_half.shape[0])}"
        )
    context = metal_context
    if (context is not None and xp is np
            and np.dtype(dp.dtype) == np.dtype(np.float64)):
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.generated import kernel_specs

        shape = tuple(int(value) for value in dp.shape)
        nlev, nlat, nlon = shape
        if tuple(int(value) for value in omega_half.shape) != (nlev + 1, nlat, nlon):
            raise ValueError("Global vertical-rate mass flux has an invalid shape")
        metal = MetalArrayModule(context)

        def resource(value):
            if isinstance(value, MetalArray):
                if value.buffer.context is not context:
                    raise ValueError(
                        "Global vertical-rate input belongs to another Metal context"
                    )
                if value.dtype != np.dtype(np.float64):
                    raise TypeError("Global vertical-rate inputs must be FP64")
                resident = metal.asarray(value)
                return (resident.view((resident.size,), dtype=np.float64)
                        if resident.ndim != 1 else resident)
            return metal.asarray(np.ascontiguousarray(
                np.asarray(value, dtype=np.float64).reshape(-1)))

        count = nlev * nlat * nlon
        output = metal.empty((count,), dtype=np.float64)
        resources = {
            "omega_half": resource(omega_half), "dp": resource(dp),
            "level_rate": output,
        }
        context.submit_kernel(
            "global_semilag_level_rate_fragment", resources,
            {"nlevel": nlev, "nlat": nlat, "nlon": nlon},
            exact_policy=False, grid_size=count,
            threads_per_group=(
                kernel_specs.GLOBAL_SEMILAG_LEVEL_RATE_FRAGMENT_THREADS_PER_GROUP
            ),
        )
        if keep_resident:
            return output.view(shape, dtype=np.float64)
        return output.to_numpy().reshape(shape)
    s_half = xp.zeros_like(omega_half)
    thickness = 0.5 * (dp[:-1] + dp[1:])
    s_half[1:nlev] = omega_half[1:nlev] / thickness
    return 0.5 * (s_half[:nlev] + s_half[1:nlev + 1])


def extrapolate_level_rate(current, previous, *, metal_context=None,
                           keep_resident: bool = False):
    """SETTLS extrapolation of the vertical trajectory rate."""
    context = metal_context
    if context is None:
        return 2.0 * current - previous
    from woof.metal.array import MetalArray, MetalArrayModule
    from woof.metal.generated import kernel_specs

    shape = tuple(int(value) for value in current.shape)
    if tuple(int(value) for value in previous.shape) != shape:
        raise ValueError("Global vertical-rate history must have equal shapes")
    metal = MetalArrayModule(context)

    def resource(value):
        if isinstance(value, MetalArray):
            if value.buffer.context is not context:
                raise ValueError("Global vertical-rate history belongs to another context")
            if value.dtype != np.dtype(np.float64):
                raise TypeError("Global vertical-rate history must be FP64")
            resident = metal.asarray(value)
            return (resident.view((resident.size,), dtype=np.float64)
                    if resident.ndim != 1 else resident)
        return metal.asarray(np.ascontiguousarray(
            np.asarray(value, dtype=np.float64).reshape(-1)))

    count = int(np.prod(shape, dtype=np.int64))
    output = metal.empty((count,), dtype=np.float64)
    context.submit_kernel(
        "global_semilag_level_rate_extrapolate_fragment",
        {"current": resource(current), "previous": resource(previous),
         "extrapolated": output},
        {"count": count}, exact_policy=False, grid_size=count,
        threads_per_group=(
            kernel_specs.GLOBAL_SEMILAG_LEVEL_RATE_EXTRAPOLATE_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    if keep_resident:
        return output.view(shape, dtype=np.float64)
    return output.to_numpy().reshape(shape)


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    iterations: int
    dt_s: float
    #: largest last-iteration move, metres
    move_max_m: float
    #: the same, in local meridional grid lengths (the reference length that
    #: does not collapse at the pole the way the zonal one does)
    move_max_cells: float
    #: largest last-iteration move of the vertical coordinate, level indices
    move_max_levels: float
    #: largest horizontal displacement of the finished trajectory, metres
    displacement_max_m: float
    displacement_max_cells: float
    displacement_max_levels: float

    def as_dict(self) -> dict:
        return {
            "semilag_trajectory_iterations": self.iterations,
            "semilag_trajectory_dt_s": self.dt_s,
            "semilag_trajectory_move_max_m": self.move_max_m,
            "semilag_trajectory_move_max_cells": self.move_max_cells,
            "semilag_trajectory_move_max_levels": self.move_max_levels,
            "semilag_displacement_max_m": self.displacement_max_m,
            "semilag_displacement_max_cells": self.displacement_max_cells,
            "semilag_displacement_max_levels": self.displacement_max_levels,
        }


def meridional_grid_length_m(tables: SphericalGridTables) -> np.ndarray:
    """``a * dphi`` per latitude row, from the pole-extended table.

    The convergence gate divides by this rather than by the local ZONAL
    spacing.  The zonal spacing collapses to 326 m on the outermost T255 ring
    and 75 m at T533, so a threshold expressed in local zonal cells would
    refuse a converged trajectory over 0.26 percent of the grid and pass a
    diverged one everywhere else.  The meridional spacing of a Gaussian grid
    is uniform to a few percent, and at T255 it is 52.1 km, the same length
    the equatorial zonal cell has.
    """
    lat_ext = tables.lat_ext_host
    nlat = tables.nlat
    upper = lat_ext[3:nlat + 3]
    lower = lat_ext[1:nlat + 1]
    return tables.radius_m * 0.5 * (upper - lower)


def _diagnostic_max(tables: SphericalGridTables, values) -> float:
    """Reduce one diagnostic field on Metal, materializing only its partials."""
    context = tables.metal_context
    if context is None or tables.xp is not np or np.dtype(tables.dtype) != np.dtype(np.float64):
        return float(tables.xp.max(values))
    from woof.metal.array import MetalArray, MetalArrayModule
    from woof.metal.generated import kernel_specs

    metal = MetalArrayModule(context)
    if isinstance(values, MetalArray):
        if values.buffer.context is not context:
            raise ValueError("Global SL diagnostic input belongs to another Metal context")
        if values.dtype != np.dtype(np.float64):
            raise TypeError("Global SL diagnostic Metal input must be FP64")
        source = values.view((values.size,), dtype=np.float64)
    else:
        host = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1))
        source = metal.asarray(host)
    chunk = 256
    partial_count = max(1, (source.size + chunk - 1) // chunk)
    resources = {
        "values": source,
        "partial_max": metal.empty((partial_count,), dtype=np.float64),
    }
    context.submit_kernel(
        "global_semilag_max_fragment", resources,
        {"count": source.size, "chunk_size": chunk,
         "partial_count": partial_count},
        exact_policy=False, grid_size=partial_count,
        threads_per_group=kernel_specs.GLOBAL_SEMILAG_MAX_FRAGMENT_THREADS_PER_GROUP,
    )
    maximum = metal.empty((1,), dtype=np.float64)
    context.submit_kernel(
        "global_semilag_max_finalize_fragment",
        {"partial_max": resources["partial_max"], "maximum": maximum},
        {"partial_count": partial_count}, exact_policy=False, grid_size=1,
        threads_per_group=(
            kernel_specs.GLOBAL_SEMILAG_MAX_FINALIZE_FRAGMENT_THREADS_PER_GROUP
        ),
    )
    return float(maximum.to_numpy()[0])


def departure_points(
    wind: CartesianWind,
    tables: SphericalGridTables,
    dt_s: float,
    *,
    iterations: int = 3,
    extrapolated: CartesianWind | None = None,
    keep_resident: bool = False,
) -> tuple[Stencil, TrajectoryDiagnostics]:
    """Solve for the departure point of every arrival point of the grid."""
    xp = tables.xp
    dtype = np.dtype(tables.dtype)
    scalar = dtype.type
    shape = wind.shape
    nlev, nlat, nlon = shape
    if (nlat, nlon) != tables.shape:
        raise ValueError(
            f"the wind is {(nlat, nlon)} on a {tables.shape} grid"
        )
    if nlev < 4:
        raise ValueError(f"nlev >= 4 is required, got {nlev}")
    if int(iterations) < 1:
        raise ValueError(
            "the departure-point search needs at least one iteration; zero "
            "would ship the explicit first guess, which is first order in dt "
            "and is not a trajectory"
        )
    ex = wind if extrapolated is None else extrapolated
    for name, array in (("wind", wind), ("extrapolated", ex)):
        for field in array.arrays():
            if tuple(int(s) for s in field.shape) != shape:
                raise ValueError(f"{name} fields must all be {shape}")
            if np.dtype(field.dtype) != dtype:
                raise ValueError(
                    f"{name} is {np.dtype(field.dtype)} where the grid tables "
                    f"were built for {dtype}"
                )

    if tables.metal_context is not None and xp is np and dtype == np.dtype(np.float64):
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.generated import kernel_specs

        metal = MetalArrayModule(tables.metal_context)
        npoint = nlev * nlat * nlon

        def field_resource(field):
            if isinstance(field, MetalArray):
                if field.buffer.context is not tables.metal_context:
                    raise ValueError("Global SL wind belongs to another Metal context")
                resident = metal.asarray(field)
            else:
                resident = metal.asarray(np.ascontiguousarray(
                    np.asarray(field, dtype=np.float64).reshape(-1)))
            if resident.ndim != 1:
                resident = resident.view((resident.size,), dtype=np.float64)
            return resident

        lat = tables.lat_ext_host[2:nlat + 2]
        lon = np.arange(nlon, dtype=np.float64) * tables.dlam
        cph = np.cos(lat)[:, None]
        axes = (
            np.broadcast_to(cph * np.cos(lon)[None, :], (nlat, nlon)),
            np.broadcast_to(cph * np.sin(lon)[None, :], (nlat, nlon)),
            np.broadcast_to(np.sin(lat)[:, None], (nlat, nlon)),
        )
        axis_resources = {
            name: metal.asarray(np.ascontiguousarray(np.broadcast_to(
                axis[None, :, :], shape).reshape(-1)))
            for name, axis in zip(("axis_x", "axis_y", "axis_z"), axes)
        }
        resources = {
            **{f"wind_{name}": field_resource(field)
               for name, field in zip(("vx", "vy", "vz", "level"),
                                      wind.arrays())},
            **{f"extrapolated_{name}": field_resource(field)
               for name, field in zip(("vx", "vy", "vz", "level"),
                                      ex.arrays())},
            **axis_resources,
            "lat_ext": metal.asarray(np.ascontiguousarray(
                tables.lat_ext_host.astype(np.float64, copy=False))),
            "mlut": metal.asarray(np.ascontiguousarray(tables.mlut, dtype=np.int32)),
            "rowoff": metal.asarray(np.ascontiguousarray(tables.rowoff, dtype=np.int32)),
            "rowshift": metal.asarray(np.ascontiguousarray(tables.rowshift, dtype=np.int32)),
            "dt": metal.asarray(np.asarray([dt_s], dtype=np.float64)),
            "radius": metal.asarray(np.asarray([tables.radius_m], dtype=np.float64)),
            "inv_dlam": metal.asarray(np.asarray([tables.inv_dlam], dtype=np.float64)),
            "lut_scale": metal.asarray(np.asarray([tables.lut_scale], dtype=np.float64)),
        }
        for name in ("xi", "phi", "departure_level"):
            resources[name] = metal.empty((npoint,), dtype=np.float64)
        # move_m carries four diagnostic planes and move_k carries two.  The
        # typed ABI is unchanged (both are flat float64 resources); named
        # views below make the per-point planes independently consumable.
        resources["move_m"] = metal.empty((4 * npoint,), dtype=np.float64)
        resources["move_k"] = metal.empty((2 * npoint,), dtype=np.float64)
        tables.metal_context.submit_kernel(
            "global_semilag_departure_fragment", resources,
            {"iterations": int(iterations), "nlevel": nlev, "nlat": nlat,
             "nlon": nlon, "lookup_bins": tables.lookup_bins},
            exact_policy=False, grid_size=npoint,
            threads_per_group=kernel_specs.GLOBAL_SEMILAG_DEPARTURE_FRAGMENT_THREADS_PER_GROUP,
        )
        if keep_resident:
            xi, phi, lev = (
                resources[name].view(shape, dtype=np.float64)
                for name in ("xi", "phi", "departure_level")
            )
        else:
            xi, phi, lev = (
                resources[name].to_numpy().reshape(shape)
                for name in ("xi", "phi", "departure_level")
            )
        move_m = resources["move_m"].view(shape, dtype=np.float64)
        move_cells = resources["move_m"].view(
            shape, dtype=np.float64, byte_offset=npoint * np.dtype(np.float64).itemsize
        )
        displacement_m = resources["move_m"].view(
            shape, dtype=np.float64, byte_offset=2 * npoint * np.dtype(np.float64).itemsize
        )
        displacement_cells = resources["move_m"].view(
            shape, dtype=np.float64, byte_offset=3 * npoint * np.dtype(np.float64).itemsize
        )
        move_k = resources["move_k"].view(shape, dtype=np.float64)
        displacement_levels = resources["move_k"].view(
            shape, dtype=np.float64, byte_offset=npoint * np.dtype(np.float64).itemsize
        )
        device_diagnostics = (
            move_cells, displacement_m, displacement_cells, displacement_levels
        )
    elif _is_cupy(xp):
        from ._cuda import get_kernel

        name = "sl_departure_f32" if dtype == np.float32 else "sl_departure_f64"
        kernel = get_kernel(name)
        xi = xp.empty(shape, dtype=dtype)
        phi = xp.empty(shape, dtype=dtype)
        lev = xp.empty(shape, dtype=dtype)
        move_m = xp.empty(shape, dtype=dtype)
        move_k = xp.empty(shape, dtype=dtype)
        npoints = nlev * nlat * nlon
        threads = 256
        blocks = (npoints + threads - 1) // threads
        kernel(
            (blocks,), (threads,),
            (xp.ascontiguousarray(wind.vx), xp.ascontiguousarray(wind.vy),
             xp.ascontiguousarray(wind.vz),
             xp.ascontiguousarray(wind.level_rate),
             xp.ascontiguousarray(ex.vx), xp.ascontiguousarray(ex.vy),
             xp.ascontiguousarray(ex.vz),
             xp.ascontiguousarray(ex.level_rate),
             tables.lat_ext, tables.mlut, tables.rowoff, tables.rowshift,
             xi, phi, lev, move_m, move_k,
             np.int32(int(iterations)), scalar(dt_s),
             scalar(tables.radius_m), scalar(tables.dlam),
             np.int32(tables.lookup_bins), np.int32(nlev), np.int32(nlat),
             np.int32(nlon), scalar(tables.inv_dlam),
             scalar(tables.lut_scale)),
        )
        device_diagnostics = None
    else:
        xi, phi, lev, move_m, move_k = _departure_numpy(
            wind, ex, tables, dt_s, iterations=int(iterations)
        )
        device_diagnostics = None
    stencil = Stencil(xi=xi, phi=phi, level=lev, tables=tables)
    diagnostics = _trajectory_diagnostics(
        stencil, move_m, move_k, tables, dt_s, int(iterations),
        device_diagnostics=device_diagnostics,
    )
    return stencil, diagnostics


def _trajectory_diagnostics(
    stencil: Stencil, move_m, move_k, tables: SphericalGridTables,
    dt_s: float, iterations: int, *, device_diagnostics=None,
) -> TrajectoryDiagnostics:
    if device_diagnostics is not None:
        move_cells, displacement_m, displacement_cells, displacement_levels = (
            device_diagnostics
        )
        return TrajectoryDiagnostics(
            iterations=int(iterations), dt_s=float(dt_s),
            move_max_m=_diagnostic_max(tables, move_m),
            move_max_cells=_diagnostic_max(tables, move_cells),
            move_max_levels=_diagnostic_max(tables, move_k),
            displacement_max_m=_diagnostic_max(tables, displacement_m),
            displacement_max_cells=_diagnostic_max(tables, displacement_cells),
            displacement_max_levels=_diagnostic_max(tables, displacement_levels),
        )
    xp = tables.xp
    dtype = np.dtype(tables.dtype)
    nlev, nlat, nlon = stencil.shape
    cell = xp.asarray(meridional_grid_length_m(tables), dtype=dtype)
    cell = cell[None, :, None]

    lat = xp.asarray(tables.lat_ext_host[2:nlat + 2], dtype=dtype)[None, :, None]
    index = xp.arange(nlon, dtype=dtype)[None, None, :]
    lev0 = xp.arange(nlev, dtype=dtype)[:, None, None]
    # Great-circle distance between arrival and departure, through the CHORD.
    # The arccos of the dot product loses half its digits as the angle goes to
    # zero, and reads 0.095 m of displacement for a wind that is exactly zero
    # at T21 in float64, which would put a floor under every convergence
    # measurement this diagnostic is here to make.
    dlam = dtype.type(tables.dlam)
    dphi = dtype.type(0.5) * (stencil.phi - lat)
    dlon = dtype.type(0.5) * (stencil.xi - index) * dlam
    hav = (xp.sin(dphi) ** 2
           + xp.cos(lat) * xp.cos(stencil.phi) * xp.sin(dlon) ** 2)
    dist = (dtype.type(2.0 * tables.radius_m)
            * xp.arcsin(xp.sqrt(xp.clip(hav, dtype.type(0.0),
                                        dtype.type(1.0)))))

    return TrajectoryDiagnostics(
        iterations=int(iterations),
        dt_s=float(dt_s),
        move_max_m=_diagnostic_max(tables, move_m),
        move_max_cells=_diagnostic_max(tables, move_m / cell),
        move_max_levels=_diagnostic_max(tables, move_k),
        displacement_max_m=_diagnostic_max(tables, dist),
        displacement_max_cells=_diagnostic_max(tables, dist / cell),
        displacement_max_levels=_diagnostic_max(tables, xp.abs(stencil.level - lev0)),
    )


def _trilinear_numpy(fields, lam, phi, lev, tables: SphericalGridTables):
    """The trilinear gather the search uses, as the numpy specification."""
    nlev = int(fields[0].shape[0])
    nlat, nlon = tables.shape
    dtype = np.dtype(tables.dtype)
    scalar = dtype.type
    lat_ext = tables.lat_ext_host.astype(dtype)
    rowoff = tables.rowoff_host
    rowshift = tables.rowshift_host
    half_pi = scalar(np.pi / 2.0)

    p = np.clip(phi, -half_pi, half_pi)
    xk = np.clip(lev, scalar(0.0), scalar(nlev - 1))
    xi = lam * scalar(tables.inv_dlam)
    fi = np.floor(xi)
    ax = (xi - fi).astype(dtype)
    c0 = fi.astype(np.int64) % nlon
    c1 = (c0 + 1) % nlon
    d0 = (c0 + nlon // 2) % nlon
    d1 = (c1 + nlon // 2) % nlon

    m0 = np.clip(
        np.searchsorted(lat_ext.astype(np.float64), p.astype(np.float64),
                        side="right") - 1,
        1, nlat + 1,
    )
    la = lat_ext[m0]
    lb = lat_ext[m0 + 1]
    ay = ((p - la) / (lb - la)).astype(dtype)
    r0 = rowoff[m0] // nlon
    r1 = rowoff[m0 + 1] // nlon
    s0 = rowshift[m0] > 0
    s1 = rowshift[m0 + 1] > 0
    i00 = r0 * nlon + np.where(s0, d0, c0)
    i01 = r0 * nlon + np.where(s0, d1, c1)
    i10 = r1 * nlon + np.where(s1, d0, c0)
    i11 = r1 * nlon + np.where(s1, d1, c1)

    k0 = np.clip(np.floor(xk).astype(np.int64), 0, nlev - 2)
    az = (xk - k0).astype(dtype)
    plane = nlat * nlon
    bx, by, bz = 1.0 - ax, 1.0 - ay, 1.0 - az

    out = []
    for field in fields:
        flat = np.ascontiguousarray(field).reshape(-1)
        base0 = k0 * plane
        base1 = base0 + plane
        t0 = (by * (bx * flat[base0 + i00] + ax * flat[base0 + i01])
              + ay * (bx * flat[base0 + i10] + ax * flat[base0 + i11]))
        t1 = (by * (bx * flat[base1 + i00] + ax * flat[base1 + i01])
              + ay * (bx * flat[base1 + i10] + ax * flat[base1 + i11]))
        out.append((bz * t0 + az * t1).astype(dtype))
    return out


def _departure_numpy(wind, ex, tables, dt_s, *, iterations):
    nlev, nlat, nlon = wind.shape
    dtype = np.dtype(tables.dtype)
    scalar = dtype.type
    lat = tables.lat_ext_host[2:nlat + 2].astype(dtype)
    lon = (np.arange(nlon) * tables.dlam).astype(dtype)
    sph = np.sin(lat)[None, :, None]
    cph = np.cos(lat)[None, :, None]
    sla = np.sin(lon)[None, None, :]
    cla = np.cos(lon)[None, None, :]
    ax = np.broadcast_to(cph * cla, wind.shape).astype(dtype)
    ay = np.broadcast_to(cph * sla, wind.shape).astype(dtype)
    az = np.broadcast_to(np.broadcast_to(sph, (1, nlat, 1)) + 0.0 * cla,
                         wind.shape).astype(dtype)
    ak = np.broadcast_to(np.arange(nlev, dtype=dtype)[:, None, None],
                         wind.shape).astype(dtype)

    radius = scalar(tables.radius_m)
    inv_r = scalar(1.0) / radius
    dt = scalar(dt_s)
    hdt = scalar(0.5) * dt

    qx = ax - dt * wind.vx * inv_r
    qy = ay - dt * wind.vy * inv_r
    qz = az - dt * wind.vz * inv_r
    norm = np.sqrt(qx * qx + qy * qy + qz * qz)
    qx, qy, qz = qx / norm, qy / norm, qz / norm
    qk = np.clip(ak - dt * wind.level_rate, scalar(0.0), scalar(nlev - 1))

    move_m = np.zeros(wind.shape, dtype=dtype)
    move_k = np.zeros(wind.shape, dtype=dtype)
    for _ in range(iterations):
        lam = np.arctan2(qy, qx)
        phi = np.arctan2(qz, np.sqrt(qx * qx + qy * qy))
        w = _trilinear_numpy(
            [wind.vx, wind.vy, wind.vz, wind.level_rate], lam, phi, qk, tables
        )
        px = ax - hdt * (ex.vx + w[0]) * inv_r
        py = ay - hdt * (ex.vy + w[1]) * inv_r
        pz = az - hdt * (ex.vz + w[2]) * inv_r
        norm = np.sqrt(px * px + py * py + pz * pz)
        px, py, pz = px / norm, py / norm, pz / norm
        pk = np.clip(ak - hdt * (ex.level_rate + w[3]), scalar(0.0),
                     scalar(nlev - 1))
        chord = np.sqrt((px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2)
        move_m = radius * 2.0 * np.arcsin(
            np.clip(0.5 * chord, scalar(0.0), scalar(1.0)))
        move_k = np.abs(pk - qk)
        qx, qy, qz, qk = px, py, pz, pk

    xi = (np.arctan2(qy, qx) * scalar(tables.inv_dlam)).astype(dtype)
    phi = np.arctan2(qz, np.sqrt(qx * qx + qy * qy)).astype(dtype)
    return xi, phi, qk.astype(dtype), move_m.astype(dtype), move_k.astype(dtype)


# ---------------------------------------------------------------------------
# The Lipschitz diagnostic.


def _sym3_max_eigenvalue(xp, a00, a01, a02, a11, a12, a22):
    """Largest eigenvalue of a symmetric 3x3, in closed form.

    Eleven million singular value decompositions a step is not a diagnostic,
    it is a second model.  The largest singular value of J is the square root
    of the largest eigenvalue of J^T J, and a symmetric 3x3 has a closed-form
    spectrum through the trigonometric solution of its characteristic cubic.

    The matrix is SCALED to unit trace before the cubic is solved and the
    eigenvalue scaled back afterwards.  The unscaled form divides by the cube
    of a deviation that is 1e-6 on a real atmospheric Jacobian in float32,
    and 1e-18 underflows to exactly zero there, so a point whose deviation
    vanished returned 0/0 and one NaN made the whole reduction NaN.  That is
    a diagnostic reporting "not a number" for a flow with a perfectly good
    deformation, which is worse than reporting nothing.
    """
    trace = a00 + a11 + a22
    # A RELATIVE floor, not an absolute one.  The matrix below is scaled to
    # unit trace, so its deviation p is O(1) unless the point is isotropic,
    # and eps is the smallest deviation that means anything there.  An
    # absolute floor of 1e-30 cubes to 1e-90 and underflows to exactly zero
    # in float32, which turned an isotropic point into 0/0 and made one NaN
    # poison the whole reduction.
    eps = float(np.finfo(np.dtype(xp.asarray(trace).dtype)).eps)
    tiny = xp.asarray(eps, dtype=trace.dtype)
    scale = xp.maximum(trace, tiny)
    inv = 1.0 / scale
    a00, a01, a02 = a00 * inv, a01 * inv, a02 * inv
    a11, a12, a22 = a11 * inv, a12 * inv, a22 * inv
    q = (a00 + a11 + a22) / 3.0
    b00 = a00 - q
    b11 = a11 - q
    b22 = a22 - q
    p2 = (b00 * b00 + b11 * b11 + b22 * b22
          + 2.0 * (a01 * a01 + a02 * a02 + a12 * a12)) / 6.0
    p = xp.sqrt(xp.maximum(p2, 0.0))
    safe = xp.maximum(p, tiny)
    det = (b00 * (b11 * b22 - a12 * a12)
           - a01 * (a01 * b22 - a12 * a02)
           + a02 * (a01 * a12 - b11 * a02))
    r = xp.clip(det / (2.0 * safe * safe * safe), -1.0, 1.0)
    phi = xp.arccos(r) / 3.0
    return scale * (q + 2.0 * p * xp.cos(phi))


def _sym2_max_eigenvalue(xp, a00, a01, a11):
    half = 0.5 * (a00 + a11)
    rad = xp.sqrt(xp.maximum(0.25 * (a00 - a11) ** 2 + a01 * a01, 0.0))
    return half + rad


def _reflected_rows(xp, field, nlon):
    """``field`` with one reflected row added at each pole."""
    top = xp.roll(field[:, :1, :], nlon // 2, axis=2)
    bot = xp.roll(field[:, -1:, :], nlon // 2, axis=2)
    return xp.concatenate([top, field, bot], axis=1)


def _d_dphi(xp, field, lat_ext, nlon):
    """``d(field)/dphi`` on the non-uniform latitude nodes, poles included.

    The two virtual rows come from the pole reflection, which is exact for a
    Cartesian component of a tangent field and for any scalar, so the
    outermost rings get the same centred three-point formula as everything
    else instead of a one-sided difference that would read as shear.
    """
    nlat = int(field.shape[1])
    ext = _reflected_rows(xp, field, nlon)
    x0 = lat_ext[1:nlat + 1][None, :, None]
    x1 = lat_ext[2:nlat + 2][None, :, None]
    x2 = lat_ext[3:nlat + 3][None, :, None]
    f0 = ext[:, :-2, :]
    f1 = ext[:, 1:-1, :]
    f2 = ext[:, 2:, :]
    return (f0 * (x1 - x2) / ((x0 - x1) * (x0 - x2))
            + f1 * (2.0 * x1 - x0 - x2) / ((x1 - x0) * (x1 - x2))
            + f2 * (x1 - x0) / ((x2 - x0) * (x2 - x1)))


def _d_dlam(xp, field, dlam):
    return (xp.roll(field, -1, axis=2) - xp.roll(field, 1, axis=2)) / (2.0 * dlam)


def _d_dk(xp, field):
    out = xp.empty_like(field)
    out[1:-1] = 0.5 * (field[2:] - field[:-2])
    out[0] = field[1] - field[0]
    out[-1] = field[-1] - field[-2]
    return out


@dataclass(frozen=True)
class LipschitzDiagnostics:
    """The flow deformation, in every norm the gate and the receipt read.

    ``lipschitz`` is the GATED number, and it is the largest of the THREE
    quantities that name an actual fold and that no choice of units can
    change: the spectral norm of the 2x2 HORIZONTAL block of the flow
    Jacobian, the vertical rate's own derivative in level indices, and
    the spectral norm of the BALANCED 3x3, which is the whole Jacobian
    with the length that converts a model level into metres chosen per
    point so the two coupling blocks have equal magnitude.  Past one in
    any of them, two arrival points share a departure point.

    The third exists because the first two are not the only invariant
    folds.  The coupling entries are ``dV/dk`` and ``grad(level rate)``,
    and each of them alone moves with the length chosen for a level, but
    their PRODUCT does not: a matrix ``[[0, a], [b, 0]]`` has both
    diagonal blocks zero and eigenvalues ``+-sqrt(ab)``, so a gate that
    read only the blocks would read zero on a fold that is there.
    Balancing is the closed-form scaling that equalizes the two coupling
    blocks.  It is not exactly the minimizer of the norm -- MEASURED on
    2,000 random matrices of this shape, it sits up to 10.8 percent above
    the minimum over all scalings -- but it needs no search, it is an
    upper bound on the spectral radius for the same reason any norm is,
    and every submatrix of it (the horizontal block and the vertical
    entry included) is bounded by it, which is why one number gates.
    MEASURED 2026-09-06, two hours of a real T127 L40 GDAS run at
    dt = 300 s, physics off: the balanced norm reads 0.1086 where the
    horizontal fold reads 0.0984, the vertical 0.0957 and the invariant
    spectral radius of the same matrix 0.0957, so the coupling adds 10
    percent to the gated number on a real flow and the mixed-unit norm
    at the equatorial spacing reads 0.4341, four times the gate, on the
    same states.  The verdict does not move where it has been measured:
    the whole T127 ladder from 100 to 900 s passes on the new number, at
    0.0827 to 0.7397.  The T255 rung at dt = 600 s, which read 0.6945 of
    0.75 on the old number, has not been rerun on this one and could sit
    near the limit; that rerun is the named follow-up.

    The mixed-unit 3x3 norms are reported beside it and are NOT gated,
    because their value depends on a length chosen to convert one model
    level into metres and no such length is a property of the flow.
    MEASURED 2026-09-06 on a real T255 L40 forecast day (the 16 GB host, GDAS
    analysis, dt = 100 s, physics off): the mixed 3x3 spectral norm reads
    3.114e-3 per second where the horizontal block reads 1.033e-3, a
    factor of 3.01, and the difference is the vertical rate's HORIZONTAL
    gradient multiplied by the 52.1 km equatorial spacing.  On the
    outermost T255 ring the zonal spacing is 326 m, so that one length
    overweights the polar rows by 160x, and gating on the mixed norm
    would have refused the 300 s step this integrator exists to take on
    the strength of two grid rings that are 0.26 percent of the planet.
    The horizontal block's own 1.033e-3 per second sits beside the
    1.0970e-3 the departure-point lane measured on the same class of
    state with no vertical motion in it at all, which is the cross-check
    that says the two instruments agree about the flow and disagree only
    about what a level index is worth.
    """

    dt_s: float
    reference_length_m: float
    #: dt * max over the grid of max(horizontal spectral norm, |ds/dk|).
    #: The gate reads this one.
    lipschitz: float
    #: dt * max Frobenius norm of the mixed-unit 3x3, reported beside it
    #: in every receipt so a future disagreement about which norm was
    #: quoted cannot be had silently
    lipschitz_frobenius: float
    #: dt * max spectral norm of the 2x2 horizontal block
    lipschitz_horizontal: float
    #: the mixed-unit 3x3 norms, per second, without the time step
    jacobian_spectral_s: float
    jacobian_frobenius_s: float
    jacobian_horizontal_s: float
    #: dt * max |d(level rate)/d(level index)|: the vertical fold
    lipschitz_vertical: float = 0.0
    jacobian_vertical_s: float = 0.0
    #: dt * the mixed-unit 3x3 spectral norm, reported and not gated
    lipschitz_mixed: float = 0.0
    #: dt * the BALANCED 3x3 spectral norm: the same matrix with the
    #: level-to-metre length chosen per point so the two coupling blocks
    #: have equal magnitude, a closed-form scaling that leaves a number no
    #: choice of length can change.  It dominates
    #: both scale-free folds (each is a submatrix of it) and it is the one
    #: the gate reads.
    lipschitz_balanced: float = 0.0
    jacobian_balanced_s: float = 0.0

    @property
    def binding_direction(self) -> str:
        if (self.lipschitz_balanced
                > 1.000001 * max(self.lipschitz_horizontal,
                                 self.lipschitz_vertical)):
            return "coupling"
        return ("vertical"
                if self.lipschitz_vertical > self.lipschitz_horizontal
                else "horizontal")

    def as_dict(self) -> dict:
        return {
            "semilag_lipschitz": self.lipschitz,
            "semilag_lipschitz_frobenius": self.lipschitz_frobenius,
            "semilag_lipschitz_horizontal": self.lipschitz_horizontal,
            "semilag_lipschitz_vertical": self.lipschitz_vertical,
            "semilag_lipschitz_balanced": self.lipschitz_balanced,
            "semilag_lipschitz_mixed": self.lipschitz_mixed,
            "semilag_jacobian_balanced_s": self.jacobian_balanced_s,
            "semilag_jacobian_spectral_s": self.jacobian_spectral_s,
            "semilag_jacobian_frobenius_s": self.jacobian_frobenius_s,
            "semilag_jacobian_horizontal_s": self.jacobian_horizontal_s,
            "semilag_jacobian_vertical_s": self.jacobian_vertical_s,
            "semilag_lipschitz_reference_length_m": self.reference_length_m,
            "semilag_lipschitz_dt_s": self.dt_s,
        }


_NORM_KERNEL_CACHE: dict = {}

#: The largest singular value of a 3x3, written once as the square root of
#: the top eigenvalue of its symmetric product, in the closed form the
#: fused kernel calls twice: once for the mixed-unit matrix it reports and
#: once for the balanced one it gates on.
_NORM_PREAMBLE = """
template <typename T>
__device__ __forceinline__ T sl_sym3_top(T a00, T a01, T a02,
                                         T a11, T a12, T a22)
{
    const T eps = (T)(sizeof(T) == 4 ? 1.1920929e-7 : 2.220446049250313e-16);
    const T trace = a00 + a11 + a22;
    const T scale = max(trace, eps);
    const T inv = (T)1 / scale;
    a00 *= inv; a11 *= inv; a22 *= inv;
    a01 *= inv; a02 *= inv; a12 *= inv;
    const T q = (a00 + a11 + a22) / (T)3;
    const T b00 = a00 - q, b11 = a11 - q, b22 = a22 - q;
    const T p2 = (b00*b00 + b11*b11 + b22*b22
                  + (T)2 * (a01*a01 + a02*a02 + a12*a12)) / (T)6;
    const T pp = sqrt(max(p2, (T)0));
    const T safe = max(pp, eps);
    const T det = b00 * (b11*b22 - a12*a12)
                - a01 * (a01*b22 - a12*a02)
                + a02 * (a01*a12 - b11*a02);
    T r = det / ((T)2 * safe * safe * safe);
    r = min(max(r, (T)-1), (T)1);
    const T ang = acos(r) / (T)3;
    return sqrt(max(scale * (q + (T)2 * pp * cos(ang)), (T)0));
}
"""


def _norm_kernel(xp):
    """One launch for the projection and both closed-form spectra.

    Written as an ElementwiseKernel on the tree's standing convention that a
    fused device path replaces a numpy expression that stays in place as the
    specification; :func:`_norms_reference` below is that specification and
    the two are graded against each other by the unit tests.

    The reason it exists is measured: the array-operation form ran about a
    hundred grid-sized elementwise passes and cost 18.6 ms at T255 and
    147.2 ms at T533 on the 5070 Ti, against 7.7 ms and 29.8 ms for the
    fourteen-field tricubic gather it is a diagnostic on.  A gate that costs
    five times the arithmetic it gates is not a gate, it is a second model.
    """
    key = getattr(xp, "__name__", "numpy")
    kernel = _NORM_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = xp.ElementwiseKernel(
            "T dvxdx, T dvydx, T dvzdx, T dvxdy, T dvydy, T dvzdy, "
            "T dvxdk, T dvydk, T dvzdk, T dsdx, T dsdy, T dsdk, "
            "T sph, T cph, T sla, T cla, T lref",
            "T spectral, T frobenius, T horizontal, T balanced",
            """
            const T exx = -sla, exy = cla, exz = (T)0;
            const T eyx = -sph * cla, eyy = -sph * sla, eyz = cph;
            const T j00 = exx * dvxdx + exy * dvydx + exz * dvzdx;
            const T j01 = exx * dvxdy + exy * dvydy + exz * dvzdy;
            const T j02 = (exx * dvxdk + exy * dvydk + exz * dvzdk) / lref;
            const T j10 = eyx * dvxdx + eyy * dvydx + eyz * dvzdx;
            const T j11 = eyx * dvxdy + eyy * dvydy + eyz * dvzdy;
            const T j12 = (eyx * dvxdk + eyy * dvydk + eyz * dvzdk) / lref;
            const T j20 = lref * dsdx;
            const T j21 = lref * dsdy;
            const T j22 = dsdk;
            frobenius = sqrt(j00*j00 + j01*j01 + j02*j02
                           + j10*j10 + j11*j11 + j12*j12
                           + j20*j20 + j21*j21 + j22*j22);
            const T h00 = j00*j00 + j10*j10;
            const T h11 = j01*j01 + j11*j11;
            const T h01 = j00*j01 + j10*j11;
            const T half = (T)0.5 * (h00 + h11);
            const T rad = sqrt(max((T)0.25 * (h00 - h11) * (h00 - h11)
                                   + h01 * h01, (T)0));
            horizontal = sqrt(max(half + rad, (T)0));
            spectral = sl_sym3_top<T>(h00 + j20*j20,
                                      j00*j01 + j10*j11 + j20*j21,
                                      j00*j02 + j10*j12 + j20*j22,
                                      h11 + j21*j21,
                                      j01*j02 + j11*j12 + j21*j22,
                                      j02*j02 + j12*j12 + j22*j22);
            /* The BALANCED matrix: the same Jacobian with the one length
               that converts a model level into metres chosen per point so
               the two coupling blocks have equal magnitude, which is the
               scaling that equalizes them.  Its entries are
               sqrt(|dV/dk| |grad s|) whatever length the caller passed, so
               this number is the coupling's own invariant and the gate can
               read it.  Both blocks go to zero when either does, which is
               the block-triangular case whose fold IS max(horizontal,
               vertical). */
            const T e0 = j02 * lref, e1 = j12 * lref;
            const T d0 = dsdx, d1 = dsdy;
            const T emag = sqrt(e0*e0 + e1*e1);
            const T dmag = sqrt(d0*d0 + d1*d1);
            const T gmean = sqrt(emag * dmag);
            const T se = emag > (T)0 ? gmean / emag : (T)0;
            const T sd = dmag > (T)0 ? gmean / dmag : (T)0;
            const T c02 = e0 * se, c12 = e1 * se;
            const T c20 = d0 * sd, c21 = d1 * sd;
            balanced = sl_sym3_top<T>(h00 + c20*c20,
                                      j00*j01 + j10*j11 + c20*c21,
                                      j00*c02 + j10*c12 + c20*j22,
                                      h11 + c21*c21,
                                      j01*c02 + j11*c12 + c21*j22,
                                      c02*c02 + c12*c12 + j22*j22);
            """,
            "arwen_semilag_lipschitz_norms",
            preamble=_NORM_PREAMBLE,
        )
        _NORM_KERNEL_CACHE[key] = kernel
    return kernel


def _norms_reference(xp, dv, ds, sph, cph, sla, cla, lref):
    """The numpy specification of :func:`_norm_kernel`."""
    ex = (-sla, cla, xp.zeros_like(sla))
    ey = (-sph * cla, -sph * sla, cph)

    def project(basis, vec):
        return basis[0] * vec[0] + basis[1] * vec[1] + basis[2] * vec[2]

    dvdx, dvdy, dvdk = dv
    dsdx, dsdy, dsdk = ds
    j00 = project(ex, dvdx)
    j01 = project(ex, dvdy)
    j02 = project(ex, dvdk) / lref
    j10 = project(ey, dvdx)
    j11 = project(ey, dvdy)
    j12 = project(ey, dvdk) / lref
    j20 = lref * dsdx
    j21 = lref * dsdy
    j22 = dsdk
    frob = xp.sqrt(j00 * j00 + j01 * j01 + j02 * j02
                   + j10 * j10 + j11 * j11 + j12 * j12
                   + j20 * j20 + j21 * j21 + j22 * j22)
    h00 = j00 * j00 + j10 * j10
    h11 = j01 * j01 + j11 * j11
    h01 = j00 * j01 + j10 * j11
    horizontal = xp.sqrt(xp.maximum(
        _sym2_max_eigenvalue(xp, h00, h01, h11), 0.0))
    m00 = h00 + j20 * j20
    m11 = h11 + j21 * j21
    m22 = j02 * j02 + j12 * j12 + j22 * j22
    m01 = j00 * j01 + j10 * j11 + j20 * j21
    m02 = j00 * j02 + j10 * j12 + j20 * j22
    m12 = j01 * j02 + j11 * j12 + j21 * j22
    spectral = xp.sqrt(xp.maximum(
        _sym3_max_eigenvalue(xp, m00, m01, m02, m11, m12, m22), 0.0))
    # The balanced matrix: see the kernel's comment.  The coupling blocks
    # both become sqrt(|dV/dk| |grad s|), which no choice of length can
    # change, and both vanish when either does.
    e0, e1 = j02 * lref, j12 * lref
    d0, d1 = dsdx, dsdy
    emag = xp.sqrt(e0 * e0 + e1 * e1)
    dmag = xp.sqrt(d0 * d0 + d1 * d1)
    gmean = xp.sqrt(emag * dmag)
    se = xp.where(emag > 0.0, gmean / xp.where(emag > 0.0, emag, 1.0), 0.0)
    sd = xp.where(dmag > 0.0, gmean / xp.where(dmag > 0.0, dmag, 1.0), 0.0)
    c02, c12 = e0 * se, e1 * se
    c20, c21 = d0 * sd, d1 * sd
    balanced = xp.sqrt(xp.maximum(_sym3_max_eigenvalue(
        xp,
        h00 + c20 * c20,
        j00 * j01 + j10 * j11 + c20 * c21,
        j00 * c02 + j10 * c12 + c20 * j22,
        h11 + c21 * c21,
        j01 * c02 + j11 * c12 + c21 * j22,
        c02 * c02 + c12 * c12 + j22 * j22,
    ), 0.0))
    return spectral, frob, horizontal, balanced


#: Levels per pass of the Lipschitz diagnostic.  The full-volume form held
#: about twenty-five grid arrays at once and ran the 5070 Ti out of memory at
#: T533 L40 while the timing harness held the fourteen advected fields
#: (MEASURED 2026-09-06: OutOfMemoryError on a 205,312,512 B request with
#: 16,246,371,840 B already allocated).  Eight levels is 41 MiB an array
#: there, and the diagnostic is a reduction, so nothing but the running
#: maxima crosses a chunk boundary.
LIPSCHITZ_LEVEL_CHUNK = 8


def lipschitz(
    wind: CartesianWind,
    tables: SphericalGridTables,
    dt_s: float,
    *,
    reference_length_m: float | None = None,
    level_chunk: int = LIPSCHITZ_LEVEL_CHUNK,
    fused_norms: bool = True,
) -> LipschitzDiagnostics:
    """The flow deformation the trajectory has to stay inside.

    The Jacobian is built from the CARTESIAN components of the wind and then
    projected onto the local tangent basis.  Building it from the local
    east-north components instead would put a false 0.037 per second into the
    outermost T255 ring for a uniform flow past the pole, because the local
    basis rotates through 2*pi across 326 m of zonal spacing there, and the
    diagnostic would refuse every step for a deformation that does not exist.
    A Cartesian component of a tangent field is a plain smooth scalar on the
    sphere and its zonal derivative vanishes at the pole with the spacing.

    The vertical is the level index.  To put its entries in the same units as
    the horizontal ones the mixed terms are scaled by a single REFERENCE
    length, by default the equatorial zonal spacing ``a*2*pi/nlon``, so no
    entry inherits the polar collapse of the local spacing.  The number the
    gate reads therefore means: how much of one equatorial cell of relative
    displacement the flow deformation produces per second.
    """
    xp = tables.xp
    dtype = np.dtype(tables.dtype)
    nlev, nlat, nlon = wind.shape
    if (nlat, nlon) != tables.shape:
        raise ValueError(f"the wind is {(nlat, nlon)} on a {tables.shape} grid")
    lref = (tables.equatorial_spacing_m() if reference_length_m is None
            else float(reference_length_m))
    if not math.isfinite(lref) or lref <= 0.0:
        raise ValueError("reference_length_m must be finite and positive")
    chunk = max(1, int(level_chunk))

    lat = xp.asarray(tables.lat_ext_host[2:nlat + 2], dtype=dtype)[None, :, None]
    lon = xp.asarray(np.arange(nlon) * tables.dlam, dtype=dtype)[None, None, :]
    lat_ext = xp.asarray(tables.lat_ext_host, dtype=dtype)
    sph, cph = xp.sin(lat), xp.cos(lat)
    sla, cla = xp.sin(lon), xp.cos(lon)
    radius = float(tables.radius_m)

    # 1/(a cos phi) is unbounded at a pole in isolation; the zonal derivative
    # of a smooth Cartesian component vanishes there at the same rate, so the
    # product is bounded.  The floor keeps the intermediate finite on a grid
    # whose outermost node sits exactly on a pole, which a Gaussian grid never
    # does but a caller-supplied grid might.
    inv_x = 1.0 / (radius * xp.maximum(cph, dtype.type(1e-12)))
    inv_y = dtype.type(1.0 / radius)

    ex = (-sla, cla, xp.zeros_like(sla))
    ey = (-sph * cla, -sph * sla, cph)

    from woof.metal.array import MetalArray

    if (tables.metal_context is not None and xp is np
            and dtype == np.dtype(np.float64)):
        from woof.metal.array import MetalArrayModule
        from woof.metal.generated import kernel_specs

        context = tables.metal_context
        metal = MetalArrayModule(context)
        names = ("vx_dx", "vx_dy", "vx_dk", "vy_dx", "vy_dy", "vy_dk",
                 "vz_dx", "vz_dy", "vz_dk", "rate_dx", "rate_dy", "rate_dk")

        def resource(value):
            if isinstance(value, MetalArray):
                if value.buffer.context is not context:
                    raise ValueError("Global SL Lipschitz input belongs to another Metal context")
                if value.dtype != np.dtype(np.float64):
                    raise TypeError("Global SL Lipschitz inputs must be FP64")
                resident = metal.asarray(value)
            else:
                resident = metal.asarray(np.ascontiguousarray(
                    np.asarray(value, dtype=np.float64).reshape(-1)))
            return (resident.view((resident.size,), dtype=np.float64)
                    if resident.ndim != 1 else resident)

        inputs = {
            "vx": resource(wind.vx), "vy": resource(wind.vy),
            "vz": resource(wind.vz), "rate": resource(wind.level_rate),
            "lat_ext": resource(tables.lat_ext_host),
            "inv_x": resource(1.0 / (radius * np.maximum(np.cos(
                tables.lat_ext_host[2:nlat + 2]), 1e-12))),
            "inv_y": resource(np.asarray([1.0 / radius], dtype=np.float64)),
            "dlam": resource(np.asarray([tables.dlam], dtype=np.float64)),
        }
        sm = fm = hm = vm = bm = 0.0
        lat2 = tables.lat_ext_host[2:nlat + 2][None, :, None]
        sph, cph = np.sin(lat2), np.cos(lat2)
        lon = np.arange(nlon, dtype=np.float64)[None, None, :] * tables.dlam
        sla, cla = np.sin(lon), np.cos(lon)
        norm_geometry = {
            "sph": metal.asarray(np.ascontiguousarray(sph.reshape(-1))),
            "cph": metal.asarray(np.ascontiguousarray(cph.reshape(-1))),
            "sla": metal.asarray(np.ascontiguousarray(sla.reshape(-1))),
            "cla": metal.asarray(np.ascontiguousarray(cla.reshape(-1))),
            "lref": metal.asarray(np.asarray([lref], dtype=np.float64)),
        }
        for start in range(0, nlev, chunk):
            stop = min(start + chunk, nlev)
            npoint = (stop - start) * nlat * nlon
            outputs = {name: metal.empty((npoint,), dtype=np.float64)
                       for name in names}
            context.submit_kernel(
                "global_semilag_lipschitz_fragment",
                {**inputs, **outputs},
                {"nlevel": nlev, "nlat": nlat, "nlon": nlon,
                 "chunk_levels": stop - start, "start_level": start},
                exact_policy=False, grid_size=npoint,
                threads_per_group=(
                    kernel_specs.GLOBAL_SEMILAG_LIPSCHITZ_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            norms = {name: metal.empty((npoint,), dtype=np.float64)
                     for name in ("spectral", "frobenius", "horizontal", "balanced")}
            context.submit_kernel(
                "global_semilag_lipschitz_norms_fragment",
                {**outputs, **norm_geometry, **norms},
                {"nlat": nlat, "nlon": nlon, "chunk_levels": stop - start},
                exact_policy=False, grid_size=npoint,
                threads_per_group=(
                    kernel_specs.GLOBAL_SEMILAG_LIPSCHITZ_NORMS_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            sm = max(sm, _diagnostic_max(tables, norms["spectral"]))
            fm = max(fm, _diagnostic_max(tables, norms["frobenius"]))
            hm = max(hm, _diagnostic_max(tables, norms["horizontal"]))
            vm = max(vm, _diagnostic_max(tables, outputs["rate_dk"]))
            bm = max(bm, _diagnostic_max(tables, norms["balanced"]))
        dt = float(dt_s)
        return LipschitzDiagnostics(
            dt_s=dt, reference_length_m=lref,
            lipschitz=dt * max(hm, vm, bm),
            lipschitz_frobenius=dt * fm, lipschitz_horizontal=dt * hm,
            jacobian_spectral_s=sm, jacobian_frobenius_s=fm,
            jacobian_horizontal_s=hm, lipschitz_vertical=dt * vm,
            jacobian_vertical_s=vm, lipschitz_mixed=dt * sm,
            lipschitz_balanced=dt * bm, jacobian_balanced_s=bm,
        )

    def host(value):
        return value.to_numpy() if isinstance(value, MetalArray) else value

    # The non-Metal reference path takes one explicit host view.  The Metal
    # route above computes this stencil in chunks and preserves wind buffers
    # for the departure kernel that follows it.
    comps = tuple(host(value) for value in (wind.vx, wind.vy, wind.vz))
    rate = host(wind.level_rate)
    if fused_norms and not hasattr(xp, "ElementwiseKernel"):
        fused_norms = False
    sm = fm = hm = vm = bm = 0.0
    for start in range(0, nlev, chunk):
        stop = min(start + chunk, nlev)
        lo = max(start - 1, 0)
        hi = min(stop + 1, nlev)
        take = slice(start - lo, start - lo + (stop - start))

        dvdx = [inv_x * _d_dlam(xp, c[start:stop], tables.dlam) for c in comps]
        dvdy = [inv_y * _d_dphi(xp, c[start:stop], lat_ext, nlon)
                for c in comps]
        dvdk = [_d_dk(xp, c[lo:hi])[take] for c in comps]

        ds = (inv_x * _d_dlam(xp, rate[start:stop], tables.dlam),
              inv_y * _d_dphi(xp, rate[start:stop], lat_ext, nlon),
              _d_dk(xp, rate[lo:hi])[take])
        # The vertical fold's own number, in the level indices the
        # trajectory travels in: no reference length reaches it.
        vertical_rate_derivative = ds[2]

        if fused_norms:
            spectral, frob, horizontal, balanced = _norm_kernel(xp)(
                dvdx[0], dvdx[1], dvdx[2],
                dvdy[0], dvdy[1], dvdy[2],
                dvdk[0], dvdk[1], dvdk[2],
                ds[0], ds[1], ds[2],
                sph, cph, sla, cla, dtype.type(lref),
            )
        else:
            spectral, frob, horizontal, balanced = _norms_reference(
                xp, (dvdx, dvdy, dvdk), ds, sph, cph, sla, cla, lref)
        del dvdx, dvdy, dvdk
        ds = None
        sm = max(sm, _diagnostic_max(tables, spectral))
        fm = max(fm, _diagnostic_max(tables, frob))
        hm = max(hm, _diagnostic_max(tables, horizontal))
        vm = max(vm, _diagnostic_max(tables, xp.abs(vertical_rate_derivative)))
        bm = max(bm, _diagnostic_max(tables, balanced))
        del spectral, frob, horizontal, balanced, vertical_rate_derivative

    dt = float(dt_s)
    gated = max(hm, vm, bm)
    return LipschitzDiagnostics(
        dt_s=dt,
        reference_length_m=lref,
        lipschitz=dt * gated,
        lipschitz_frobenius=dt * fm,
        lipschitz_horizontal=dt * hm,
        jacobian_spectral_s=sm,
        jacobian_frobenius_s=fm,
        jacobian_horizontal_s=hm,
        lipschitz_vertical=dt * vm,
        jacobian_vertical_s=vm,
        lipschitz_mixed=dt * sm,
        lipschitz_balanced=dt * bm,
        jacobian_balanced_s=bm,
    )


def convergence(diagnostics: TrajectoryDiagnostics, limit_cells: float) -> None:
    """Refuse a trajectory whose last iteration still moved.

    Named breakage: an unconverged trajectory reads as a Rossby phase error
    and mislocated cyclones, and it produces no other symptom at all.  The
    move is measured every step rather than the iteration count being
    trusted, because the count that converges on an hour-0 state is not the
    count that converges on a day-2 jet.
    """
    if diagnostics.move_max_cells > float(limit_cells):
        raise ValueError(
            "the departure-point search did not converge: the last of "
            f"{diagnostics.iterations} iterations still moved a point by "
            f"{diagnostics.move_max_m:.3f} m, which is "
            f"{diagnostics.move_max_cells:.4f} of the local meridional grid "
            f"length against a limit of {float(limit_cells):.4f}; an "
            "unconverged trajectory mislocates the flow and shows no other "
            "symptom"
        )


def refuse_beyond_lipschitz(
    diagnostics: LipschitzDiagnostics, limit: float
) -> None:
    """Refuse a time step the flow deformation does not permit.

    Named breakage: above one the trajectory map is not invertible, two
    arrival points share a departure point, the fixed-point iteration stops
    contracting, and the interpolation samples a folded field.  The model
    does not blow up when this happens; it silently mislocates.
    """
    if not (0.0 < float(limit) <= 1.0):
        raise ValueError(
            "the Lipschitz limit must lie in (0, 1]; above one the "
            "trajectory map is not invertible at all"
        )
    if diagnostics.lipschitz > float(limit):
        raise ValueError(
            f"the Lipschitz number is {diagnostics.lipschitz:.4f} at "
            f"dt = {diagnostics.dt_s:g} s, binding in the "
            f"{diagnostics.binding_direction} (horizontal "
            f"{diagnostics.lipschitz_horizontal:.4f}, vertical "
            f"{diagnostics.lipschitz_vertical:.4f}, balanced "
            f"{diagnostics.lipschitz_balanced:.4f}; the mixed-unit 3x3 "
            f"norms, reported and not gated, read "
            f"{diagnostics.lipschitz_mixed:.4f} spectral and "
            f"{diagnostics.lipschitz_frobenius:.4f} Frobenius at a "
            f"reference length of {diagnostics.reference_length_m:.0f} m) "
            f"against a limit of {float(limit):.4f}; the trajectory map "
            "folds above one and mislocates silently below it"
        )
