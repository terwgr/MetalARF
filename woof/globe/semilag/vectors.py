"""Carrying a vector from its departure point to its arrival point.

A scalar interpolated at a departure point is the answer.  A VECTOR is
not: the three geocentric Cartesian components read at the departure
point describe a vector that lies in the tangent plane THERE, and the
arrival point's tangent plane is a different one.  Dropping the
difference by projecting is not free, and the size of what it drops is
measurable: a 34.4 km displacement (T255, dt = 300 s, MEASURED
2026-09-06 on a real state) tilts the plane by 5.4e-3 rad, and a plain
projection would shorten the wind by 1 - cos(5.4e-3) = 1.46e-5 per step,
which over the 288 steps of a forecast day is 4.2e-3 of the wind removed
by nothing but bookkeeping.

The vector is ROTATED instead, by the minimal rotation taking the
departure point to the arrival point.  That rotation is not an
approximation of the physics: the Cartesian momentum equation of a
parcel with no forces on it is ``D V/Dt = -(|V|^2/a) r``, whose solution
is the great circle with the velocity parallel-transported along it, and
the minimal rotation from ``r_D`` to ``r_A`` IS that parallel transport.
So the term the trajectory would otherwise have to carry explicitly is
integrated exactly, and the magnitude of the wind is preserved to the
last bit the arithmetic has.

For unit vectors ``p`` (departure) and ``q`` (arrival),

    R v = v - ((p + q).v / (1 + p.q)) (p + q) + 2 (p.v) q

which sends ``p`` to ``q``, fixes every vector orthogonal to both, and
needs no axis, no angle and no trigonometric reconstruction of either.
Its one singular point is ``p.q = -1``, the antipode; the largest
displacement in the ladder is 0.011 rad.

The projection onto the arrival point's local east-north basis is done in
the same launch, because the caller always wants it: the step assembles
its right-hand side in the local components the vector analysis reads.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .interpolate import Stencil, _is_cupy
from .tables import SphericalGridTables

_TRANSPORT_KERNEL_CACHE: dict = {}


def _transport_kernel(xp):
    """One launch for four sine/cosine pairs, the rotation and the
    projection.

    Written as an ElementwiseKernel on the tree's standing convention that
    a fused device path replaces a numpy expression that stays in place as
    the specification; :func:`_transport_reference` is that specification
    and the two are graded against each other by the unit tests.
    """
    key = getattr(xp, "__name__", "numpy")
    kernel = _TRANSPORT_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = xp.ElementwiseKernel(
            "T vx, T vy, T vz, T xi, T phid, T sphA, T cphA, T slaA, T claA, "
            "T dlam",
            "T uout, T vout",
            """
            const T lam = xi * dlam;
            const T sd = sin(lam), cd = cos(lam);
            const T spd = sin(phid), cpd = cos(phid);
            const T px = cpd * cd, py = cpd * sd, pz = spd;
            const T qx = cphA * claA, qy = cphA * slaA, qz = sphA;
            const T dot = px * qx + py * qy + pz * qz;
            const T den = max((T)1 + dot, (T)1e-6);
            const T sx = px + qx, sy = py + qy, sz = pz + qz;
            const T sv = (sx * vx + sy * vy + sz * vz) / den;
            const T pv = (T)2 * (px * vx + py * vy + pz * vz);
            const T wx = vx - sv * sx + pv * qx;
            const T wy = vy - sv * sy + pv * qy;
            const T wz = vz - sv * sz + pv * qz;
            uout = -wx * slaA + wy * claA;
            vout = -wx * sphA * claA - wy * sphA * slaA + wz * cphA;
            """,
            "arwen_semilag_parallel_transport",
        )
        _TRANSPORT_KERNEL_CACHE[key] = kernel
    return kernel


def _arrival_trigonometry(tables: SphericalGridTables, *, xp=None, dtype=None):
    """``(sin phi, cos phi, sin lambda, cos lambda)`` of the grid points,
    shaped to broadcast over a ``(nlev, nlat, nlon)`` volume."""
    xp = tables.xp if xp is None else xp
    dtype = tables.dtype if dtype is None else dtype
    nlat, nlon = tables.shape
    lat = xp.asarray(tables.lat_ext_host[2:nlat + 2], dtype=dtype)
    lon = xp.asarray(np.arange(nlon) * tables.dlam, dtype=dtype)
    return (
        xp.sin(lat)[None, :, None], xp.cos(lat)[None, :, None],
        xp.sin(lon)[None, None, :], xp.cos(lon)[None, None, :],
    )


def transport_to_arrival(
    vx: Any, vy: Any, vz: Any, stencil: Stencil, tables: SphericalGridTables,
    *, fused: bool = True, keep_resident: bool = False,
):
    """Rotate a departure-point Cartesian vector field onto the arrival
    points and return its local eastward and northward components."""
    xp = tables.xp
    dtype = np.dtype(tables.dtype)
    for name, field in (("vx", vx), ("vy", vy), ("vz", vz)):
        if tuple(int(s) for s in field.shape) != stencil.shape:
            raise ValueError(
                f"{name} is {tuple(int(s) for s in field.shape)} where the "
                f"stencil is {stencil.shape}"
            )
    sph, cph, sla, cla = _arrival_trigonometry(tables, xp=xp, dtype=dtype)
    context = tables.metal_context
    if (context is not None and xp is np
            and dtype == np.dtype(np.float64)):
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.generated import kernel_specs

        nlev, nlat, nlon = stencil.shape
        count = nlev * nlat * nlon
        metal = MetalArrayModule(context)

        def resource(value):
            if isinstance(value, MetalArray):
                if value.buffer.context is not context:
                    raise ValueError("Global SL transport input belongs to another Metal context")
                if value.dtype != np.dtype(np.float64):
                    raise TypeError("Global SL transport inputs must be FP64")
                resident = metal.asarray(value)
                return (resident.view((resident.size,), dtype=np.float64)
                        if resident.ndim != 1 else resident)
            return metal.asarray(np.ascontiguousarray(
                np.asarray(value, dtype=np.float64).reshape(-1)))

        if keep_resident and isinstance(stencil.xi, MetalArray):
            if (stencil.xi.buffer.context is not context
                    or stencil.phi.buffer.context is not context
                    or stencil.xi.dtype != np.dtype(np.float64)
                    or stencil.phi.dtype != np.dtype(np.float64)):
                raise ValueError("resident Global transport stencil needs its FP64 Metal context")
            nlevel, nlat, nlon = map(int, stencil.shape)
            count = nlevel * nlat * nlon
            lat = tables.lat_ext_host[2:nlat + 2]
            lon = np.arange(nlon, dtype=np.float64) * tables.dlam
            outputs = {
                "u_arrival": metal.empty((count,), dtype=np.float64),
                "v_arrival": metal.empty((count,), dtype=np.float64),
            }
            context.submit_kernel(
                "global_semilag_transport_resident_fragment",
                {"vx": resource(vx), "vy": resource(vy), "vz": resource(vz),
                 "xi": resource(stencil.xi), "phi": resource(stencil.phi),
                 "dlam": metal.asarray(np.asarray([tables.dlam], dtype=np.float64)),
                 "sin_lat": metal.asarray(np.ascontiguousarray(np.sin(lat))),
                 "cos_lat": metal.asarray(np.ascontiguousarray(np.cos(lat))),
                 "sin_lon": metal.asarray(np.ascontiguousarray(np.sin(lon))),
                 "cos_lon": metal.asarray(np.ascontiguousarray(np.cos(lon))),
                 **outputs},
                {"count": count, "nlat": nlat, "nlon": nlon},
                exact_policy=False, grid_size=count,
                threads_per_group=(
                    kernel_specs.GLOBAL_SEMILAG_TRANSPORT_RESIDENT_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            return (outputs["u_arrival"].view(stencil.shape, dtype=np.float64),
                    outputs["v_arrival"].view(stencil.shape, dtype=np.float64))

        def host(value):
            return value.to_numpy() if isinstance(value, MetalArray) else np.asarray(value)

        xi = host(stencil.xi)
        phi = host(stencil.phi)
        lam = xi * float(tables.dlam)
        px = np.cos(phi) * np.cos(lam)
        py = np.cos(phi) * np.sin(lam)
        pz = np.sin(phi)
        qx = np.broadcast_to(cph * cla, stencil.shape)
        qy = np.broadcast_to(cph * sla, stencil.shape)
        qz = np.broadcast_to(sph, stencil.shape)
        sin_lat = np.broadcast_to(sph, stencil.shape)
        cos_lat = np.broadcast_to(cph, stencil.shape)
        sin_latlon = np.broadcast_to(sla, stencil.shape)
        cos_latlon = np.broadcast_to(cla, stencil.shape)
        outputs = {
            "u_arrival": metal.empty((count,), dtype=np.float64),
            "v_arrival": metal.empty((count,), dtype=np.float64),
        }
        resources = {
            "vx": resource(vx), "vy": resource(vy), "vz": resource(vz),
            **{name: resource(value) for name, value in (
                ("px", px), ("py", py), ("pz", pz),
                ("qx", qx), ("qy", qy), ("qz", qz),
                ("sin_lat", sin_lat), ("cos_lat", cos_lat),
                ("sin_latlon", sin_latlon), ("cos_latlon", cos_latlon),
            )},
            **outputs,
        }
        context.submit_kernel(
            "global_semilag_transport_fragment", resources,
            {"count": count}, exact_policy=False, grid_size=count,
            threads_per_group=(
                kernel_specs.GLOBAL_SEMILAG_TRANSPORT_FRAGMENT_THREADS_PER_GROUP
            ),
        )
        if keep_resident:
            return (outputs["u_arrival"].view(stencil.shape, dtype=np.float64),
                    outputs["v_arrival"].view(stencil.shape, dtype=np.float64))
        return (outputs["u_arrival"].to_numpy().reshape(stencil.shape),
                outputs["v_arrival"].to_numpy().reshape(stencil.shape))
    if fused and hasattr(xp, "ElementwiseKernel"):
        return _transport_kernel(xp)(
            vx, vy, vz, stencil.xi, stencil.phi, sph, cph, sla, cla,
            dtype.type(tables.dlam),
        )
    return _transport_reference(
        xp, vx, vy, vz, stencil, sph, cph, sla, cla, dtype
    )


def _transport_reference(xp, vx, vy, vz, stencil, sph, cph, sla, cla, dtype):
    """The numpy specification of :func:`_transport_kernel`."""
    lam = stencil.xi * dtype.type(stencil.tables.dlam)
    sd, cd = xp.sin(lam), xp.cos(lam)
    spd, cpd = xp.sin(stencil.phi), xp.cos(stencil.phi)
    px, py, pz = cpd * cd, cpd * sd, spd
    qx, qy, qz = cph * cla, cph * sla, sph
    dot = px * qx + py * qy + pz * qz
    den = xp.maximum(1.0 + dot, dtype.type(1.0e-6))
    sx, sy, sz = px + qx, py + qy, pz + qz
    sv = (sx * vx + sy * vy + sz * vz) / den
    pv = 2.0 * (px * vx + py * vy + pz * vz)
    wx = vx - sv * sx + pv * qx
    wy = vy - sv * sy + pv * qy
    wz = vz - sv * sz + pv * qz
    u = -wx * sla + wy * cla
    v = -wx * sph * cla - wy * sph * sla + wz * cph
    return u, v


__all__ = ["transport_to_arrival"]
