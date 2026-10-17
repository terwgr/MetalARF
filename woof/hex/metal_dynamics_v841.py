"""Metal mirror of the v8.4.1 CUDA dynamics kernels and driver tendency launches.

``cuda_dynamics_v841`` is mirrored function for function.  The kernels that
``CudaDryDycoreDriver`` launches itself on the v8.4.1 dry path are exposed as
module-level functions named after the driver methods; each takes the driver
(any object carrying the same attributes as ``CudaDryDycoreDriver`` plus
``arrays``, a ``MetalArrayModule``) as its first argument.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import ConfigurationRefusal
from .metal_containers_v841 import MetalPrognosticState, MetalSavedDiagnostics
from .metal_v841_launch import launch

_PREFIX = "hexv841_"


def _context(array: Any) -> Any:
    return array.buffer.context


def _arrays(context: Any) -> Any:
    from woof.metal.array import MetalArrayModule

    return MetalArrayModule(context)


def _launch(name: str, count: int, context: Any, arrays: dict, params: dict) -> None:
    # The registered shape rule is any_1d: pass flat views of the same storage.
    flat = {key: (value.view((int(value.size),)) if value.ndim != 1 else value)
            for key, value in arrays.items()}
    launch(context, _PREFIX + name, int(count), flat, params)


def _require(name: str, value: Any, dtype: Any, shape: tuple[int, ...] | None = None) -> Any:
    if not hasattr(value, "buffer") or np.dtype(value.dtype) != np.dtype(dtype):
        raise ValueError(f"{name} must be a resident Metal {np.dtype(dtype)} array")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} shape {tuple(value.shape)} != {tuple(shape)}")
    return value


def _shape(value: Any, name: str) -> tuple[int, int]:
    array = _require(name, value, np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be C-contiguous (level,owner) float32")
    return int(array.shape[0]), int(array.shape[1])


def _i(value: Any) -> np.int32:
    return np.int32(value)


def _f(value: Any) -> np.float32:
    return np.float32(value)


def vector_momentum_tendency_metal_v841(
    mesh: Any,
    context: Any,
    *,
    normal_velocity: Any,
    rho_edge: Any,
    pv_edge: Any,
    kinetic_energy: Any,
    horizontal_divergence: Any,
) -> Any:
    nlev, nedges = _shape(normal_velocity, "normal_velocity")
    ncells = int(mesh.n_cells)
    if nedges != int(mesh.n_edges):
        raise ValueError("normal_velocity edge extent differs from mesh")
    for name, value, shape in (
        ("rho_edge", rho_edge, (nlev, nedges)),
        ("pv_edge", pv_edge, (nlev, nedges)),
        ("kinetic_energy", kinetic_energy, (nlev, ncells)),
        ("horizontal_divergence", horizontal_divergence, (nlev, ncells)),
    ):
        _require(name, value, np.float32, shape)
    mctx = _context(normal_velocity)
    out = _arrays(mctx).empty(normal_velocity.shape, np.float32)
    reference_u = reference_wind_edge_metal_v841(mesh, context, n_vert_levels=nlev)
    _launch(
        "vector_momentum_v841_f32", nlev * nedges, mctx,
        dict(u=normal_velocity, rho_edge=rho_edge, pv_edge=pv_edge,
             kinetic=kinetic_energy, mass_divergence=horizontal_divergence,
             cells_on_edge=mesh.cells_on_edge, edges_on_edge=mesh.edges_on_edge,
             n_edges_on_edge=mesh.n_edges_on_edge,
             weights_on_edge=mesh.weights_on_edge,
             inv_dc_edge=context.inv_dc_edge, reference_u=reference_u,
             f_edge=mesh.f_edge, result=out),
        dict(nlev=_i(nlev), ncells=_i(ncells), nedges=_i(nedges),
             max_edges2=_i(mesh.max_edges2)),
    )
    return out


_REFERENCE_WIND: dict[tuple[Any, ...], tuple[Any, ...]] = {}


def reference_wind_edge_metal_v841(
    mesh: Any, context: Any, *, n_vert_levels: int
) -> Any:
    """The (level, edge) perturbation-Coriolis reference wind, once per run."""

    nlev = int(n_vert_levels)
    nedges = int(mesh.n_edges)
    angle_edge = mesh.angle_edge
    key = (id(angle_edge), id(context.u_init), id(context.v_init), nlev, nedges)
    hit = _REFERENCE_WIND.get(key)
    if hit is not None and hit[0] is angle_edge and hit[1] is context.u_init \
            and hit[2] is context.v_init:
        return hit[3]
    _require("angle_edge", angle_edge, np.float32, (nedges,))
    _require("u_init", context.u_init, np.float32, (nlev,))
    _require("v_init", context.v_init, np.float32, (nlev,))
    mctx = _context(angle_edge)
    reference_u = _arrays(mctx).empty((nlev, nedges), np.float32)
    _launch(
        "reference_wind_edge_v841_f32", nlev * nedges, mctx,
        dict(angle_edge=angle_edge, u_init=context.u_init,
             v_init=context.v_init, reference_u=reference_u),
        dict(nlev=_i(nlev), nedges=_i(nedges)),
    )
    _REFERENCE_WIND[key] = (angle_edge, context.u_init, context.v_init, reference_u)
    return reference_u


def theta_finish_metal_v841(
    mesh: Any, context: Any, *, edge_flux: Any, vertical_flux: Any,
    rdzw: Any, n_vert_levels: int,
) -> Any:
    nlev = int(n_vert_levels)
    ncells = int(mesh.n_cells)
    nedges = int(mesh.n_edges)
    _require("edge_flux", edge_flux, np.float32, (nlev, nedges))
    _require("vertical_flux", vertical_flux, np.float32, (nlev + 1, ncells))
    _require("rdzw", rdzw, np.float32, (nlev,))
    mctx = _context(edge_flux)
    out = _arrays(mctx).empty((nlev, ncells), np.float32)
    _launch(
        "theta_finish_v841_f32", nlev * ncells, mctx,
        dict(n_edges_on_cell=mesh.n_edges_on_cell,
             edges_on_cell=mesh.edges_on_cell,
             acoustic_sign=mesh.edge_sign_on_cell,
             inv_area_cell=context.inv_area_cell, rdzw=rdzw,
             edge_flux=edge_flux, vertical_flux=vertical_flux, result=out),
        dict(nlev=_i(nlev), ncells=_i(ncells), nedges=_i(nedges),
             max_edges=_i(mesh.max_edges)),
    )
    return out


def w_finish_metal_v841(
    mesh: Any, context: Any, *, edge_flux: Any, vertical_flux: Any,
    rdzu: Any, n_vert_levels: int,
) -> Any:
    nlev = int(n_vert_levels)
    ncells = int(mesh.n_cells)
    nedges = int(mesh.n_edges)
    _require("edge_flux", edge_flux, np.float32, (nlev + 1, nedges))
    _require("vertical_flux", vertical_flux, np.float32, (nlev + 1, ncells))
    _require("rdzu", rdzu, np.float32, (nlev,))
    mctx = _context(edge_flux)
    out = _arrays(mctx).empty((nlev + 1, ncells), np.float32)
    _launch(
        "w_finish_v841_f32", (nlev + 1) * ncells, mctx,
        dict(n_edges_on_cell=mesh.n_edges_on_cell,
             edges_on_cell=mesh.edges_on_cell,
             acoustic_sign=mesh.edge_sign_on_cell,
             inv_area_cell=context.inv_area_cell, rdzu=rdzu,
             edge_flux=edge_flux, vertical_flux=vertical_flux, result=out),
        dict(nlev=_i(nlev), ncells=_i(ncells), nedges=_i(nedges),
             max_edges=_i(mesh.max_edges)),
    )
    return out


def enforce_rw_endpoints_metal_v841(rho_w: Any) -> Any:
    nlevels, ncells = _shape(rho_w, "rho_w")
    if nlevels < 2:
        raise ValueError("rho_w must contain bottom and top interfaces")
    _launch(
        "enforce_rw_endpoints_v841_f32", ncells, _context(rho_w),
        dict(rw=rho_w), dict(nlev=_i(nlevels - 1), ncells=_i(ncells)),
    )
    return rho_w


def validate_positive_state_metal_v841(
    rho: Any, rho_theta: Any, *, invalid_flag: Any | None = None
) -> Any:
    nlev, ncells = _shape(rho, "rho")
    _require("rho_theta", rho_theta, np.float32, (nlev, ncells))
    mctx = _context(rho)
    owns_flag = invalid_flag is None
    invalid = (
        _arrays(mctx).zeros((1,), np.int32)
        if owns_flag
        else _require("invalid_flag", invalid_flag, np.int32, (1,))
    )
    _launch(
        "validate_positive_state_v841_f32", nlev * ncells, mctx,
        dict(rho=rho, rho_theta=rho_theta, invalid=invalid),
        dict(nvalues=_i(nlev * ncells)),
    )
    if owns_flag and int(invalid.to_numpy()[0]) != 0:
        raise FloatingPointError(
            "v8.4.1 Metal rho and rho_theta must remain finite and positive"
        )
    return invalid


def validate_recovered_state_metal_v841(state: Any, saved: Any, *, invalid_flag: Any) -> Any:
    nlev, ncells = _shape(state.rho, "state.rho")
    nedges = int(state.rho_u.shape[1])
    requirements = {
        "state.rho_theta": (state.rho_theta, (nlev, ncells)),
        "state.rho_u": (state.rho_u, (nlev, nedges)),
        "state.rho_w": (state.rho_w, (nlev + 1, ncells)),
        "saved.theta_m": (saved.theta_m, (nlev, ncells)),
        "saved.exner": (saved.exner, (nlev, ncells)),
        "saved.density_perturbation": (saved.density_perturbation, (nlev, ncells)),
        "saved.rho_theta_perturbation": (saved.rho_theta_perturbation, (nlev, ncells)),
        "saved.pressure_perturbation": (saved.pressure_perturbation, (nlev, ncells)),
        "saved.normal_velocity": (saved.normal_velocity, (nlev, nedges)),
        "saved.vertical_velocity": (saved.vertical_velocity, (nlev + 1, ncells)),
    }
    for name, (value, shape) in requirements.items():
        _require(name, value, np.float32, shape)
    flag = _require("invalid_flag", invalid_flag, np.int32, (1,))
    _launch(
        "validate_recovered_v841_f32", max(ncells, nedges), _context(state.rho),
        dict(rho=state.rho, rho_theta=state.rho_theta, rho_u=state.rho_u,
             rho_w=state.rho_w, theta=saved.theta_m, exner=saved.exner,
             rho_p=saved.density_perturbation,
             rtheta_p=saved.rho_theta_perturbation,
             pressure_p=saved.pressure_perturbation,
             normal_velocity=saved.normal_velocity,
             vertical_velocity=saved.vertical_velocity, invalid=flag),
        dict(nlev=_i(nlev), ncells=_i(ncells), nedges=_i(nedges)),
    )
    return flag


def accumulate_finite_array_metal_v841(values: Any, *, invalid_flag: Any) -> Any:
    array = _require("values", values, np.float32)
    flag = _require("invalid_flag", invalid_flag, np.int32, (1,))
    _launch(
        "validate_finite_array_v841_f32", int(array.size), _context(array),
        dict(values=array, invalid=flag), dict(nvalues=_i(array.size)),
    )
    return flag


def accumulate_split_flux_metal_v841(current: Any, accumulator: Any | None) -> Any:
    nlevels, nowners = _shape(current, "current_split_flux")
    mctx = _context(current)
    if accumulator is None:
        result = _arrays(mctx).empty(current.shape, np.float32)
        kernel = "split_flux_first_v841_f32"
    else:
        levels, owners = _shape(accumulator, "split_flux_accumulator")
        if (levels, owners) != (nlevels, nowners):
            raise ValueError("split flux accumulator shape mismatch")
        result = accumulator
        kernel = "split_flux_add_v841_f32"
    _launch(kernel, nlevels * nowners, mctx,
            dict(current=current, accumulator=result),
            dict(nlevels=_i(nlevels), nowners=_i(nowners)))
    return result


def finish_split_flux_metal_v841(accumulator: Any, split_count: int) -> Any:
    if not isinstance(split_count, (int, np.integer)) or isinstance(
        split_count, (bool, np.bool_)
    ) or int(split_count) < 1:
        raise ValueError("split_count must be a positive integer")
    nlevels, nowners = _shape(accumulator, "split_flux_accumulator")
    mctx = _context(accumulator)
    result = _arrays(mctx).empty(accumulator.shape, np.float32)
    reciprocal = np.float32(1.0) / np.float32(split_count)
    _launch("split_flux_finish_v841_f32", nlevels * nowners, mctx,
            dict(accumulator=accumulator, average=result),
            dict(nlevels=_i(nlevels), nowners=_i(nowners), reciprocal=reciprocal))
    return result


# ---------------------------------------------------------------------------
# Kernels launched by CudaDryDycoreDriver itself (cuda_driver.py).
# ``driver`` carries the CudaDryDycoreDriver attributes (nlev, ncells, nedges,
# atmosphere, coefficients, config, v841_context) plus ``arrays``.
# ---------------------------------------------------------------------------


def _level_owner_shape(value: Any, name: str) -> tuple[int, int]:
    if getattr(value, "ndim", None) != 2:
        raise ValueError(f"{name} must be a (level,owner) array")
    return int(value.shape[0]), int(value.shape[1])


def _add(driver: Any, target: Any, increment: Any) -> None:
    nlevels, nowners = _level_owner_shape(target, "target")
    if tuple(increment.shape) != tuple(target.shape):
        raise ValueError("increment shape differs from target")
    _launch("add_inplace_f32", nlevels * nowners, _context(target),
            dict(increment=increment, target=target),
            dict(nlevels=_i(nlevels), nowners=_i(nowners)))


def _scale(driver: Any, source: Any, scale: float) -> Any:
    nlevels, nowners = _level_owner_shape(source, "source")
    result = driver.arrays.empty(source.shape, np.float32)
    _launch("scale_f32", nlevels * nowners, _context(source),
            dict(source=source, target=result),
            dict(nlevels=_i(nlevels), nowners=_i(nowners), scale=_f(scale)))
    return result


def _euler_w(driver: Any, pressure_perturbation: Any, dpdz: Any) -> Any:
    """The euler_w_f32 launch of the dry v8.4.1 subcycle."""

    vertical = driver.atmosphere.vertical
    result = driver.arrays.empty((driver.nlev + 1, driver.ncells), np.float32)
    _launch("euler_w_f32", driver.ncells * (driver.nlev + 1),
            _context(pressure_perturbation),
            dict(pressure_p=pressure_perturbation, dpdz=dpdz, rdzu=vertical.rdzu,
                 fzm=vertical.fzm, fzp=vertical.fzp, result=result),
            dict(nlev=_i(driver.nlev), ncells=_i(driver.ncells)))
    return result


def _vertical_u(driver: Any, u: Any, rw: Any) -> Any:
    vertical = driver.atmosphere.vertical
    flux = driver.arrays.empty((driver.nlev + 1, driver.nedges), np.float32)
    result = driver.arrays.empty((driver.nlev, driver.nedges), np.float32)
    mctx = _context(u)
    _launch("vertical_u_flux_f32", driver.nedges * (driver.nlev + 1), mctx,
            dict(u=u, rw=rw, cells_on_edge=driver.atmosphere.mesh.cells_on_edge,
                 fzm=vertical.fzm, fzp=vertical.fzp, flux=flux),
            dict(nlev=_i(driver.nlev), ncells=_i(driver.ncells),
                 nedges=_i(driver.nedges)))
    _launch("vertical_u_finish_f32", driver.nedges * driver.nlev, mctx,
            dict(rdzw=vertical.rdzw, flux=flux, result=result),
            dict(nlev=_i(driver.nlev), nedges=_i(driver.nedges)))
    return result


def _vector_momentum(driver: Any, u: Any, rho_edge: Any, pv_edge: Any,
                     kinetic: Any, mass_divergence: Any) -> Any:
    if driver.v841_context is None:
        raise ConfigurationRefusal(
            "v841_context", None, "the Metal momentum tendency is the v8.4.1 kernel",
            "a v8.4.1 context")
    return vector_momentum_tendency_metal_v841(
        driver.atmosphere.mesh, driver.v841_context, normal_velocity=u,
        rho_edge=rho_edge, pv_edge=pv_edge, kinetic_energy=kinetic,
        horizontal_divergence=mass_divergence)


def _theta_tendency(driver: Any, state: Any, theta: Any, saved_state: Any,
                    theta_saved: Any, rk_step: int) -> Any:
    if driver.v841_context is None:
        raise ConfigurationRefusal(
            "v841_context", None, "the Metal theta finish is the v8.4.1 kernel",
            "a v8.4.1 context")
    coefficient = driver.coefficients
    width = int(coefficient.adv_coefs.shape[1])
    mesh = driver.atmosphere.mesh
    vertical = driver.atmosphere.vertical
    mctx = _context(theta)
    edge_flux = driver.arrays.empty(state.rho_u.shape, np.float32)
    vertical_flux = driver.arrays.empty(state.rho_w.shape, np.float32)
    _launch("theta_edge_flux_f32", driver.nedges * driver.nlev, mctx,
            dict(theta=theta, theta_saved=theta_saved, ru=state.rho_u,
                 ru_saved=saved_state.rho_u, dv_edge=mesh.dv_edge,
                 cells_on_edge=mesh.cells_on_edge, adv=coefficient.adv_coefs,
                 adv3=coefficient.adv_coefs_3rd,
                 n_adv=coefficient.n_adv_cells_for_edge,
                 adv_cells=coefficient.adv_cells_for_edge, flux=edge_flux),
            dict(nlev=_i(driver.nlev), ncells=_i(driver.ncells),
                 nedges=_i(driver.nedges), width=_i(width), rk_step=_i(rk_step),
                 coefficient=_f(driver.config.config_coef_3rd_order)))
    _launch("theta_vertical_flux_f32", driver.ncells * (driver.nlev + 1), mctx,
            dict(theta=theta, theta_saved=theta_saved, rw=state.rho_w,
                 rw_saved=saved_state.rho_w, fzm=vertical.fzm, fzp=vertical.fzp,
                 flux=vertical_flux),
            dict(nlev=_i(driver.nlev), ncells=_i(driver.ncells),
                 coefficient=_f(driver.config.config_coef_3rd_order)))
    return theta_finish_metal_v841(
        mesh, driver.v841_context, edge_flux=edge_flux,
        vertical_flux=vertical_flux, rdzw=vertical.rdzw,
        n_vert_levels=driver.nlev)


def _w_tendency(driver: Any, state: Any, w: Any) -> Any:
    if driver.v841_context is None:
        raise ConfigurationRefusal(
            "v841_context", None, "the Metal w finish is the v8.4.1 kernel",
            "a v8.4.1 context")
    coefficient = driver.coefficients
    width = int(coefficient.adv_coefs.shape[1])
    mesh = driver.atmosphere.mesh
    vertical = driver.atmosphere.vertical
    mctx = _context(w)
    edge_flux = driver.arrays.empty((driver.nlev + 1, driver.nedges), np.float32)
    vertical_flux = driver.arrays.empty(state.rho_w.shape, np.float32)
    _launch("w_edge_flux_f32", driver.nedges * (driver.nlev + 1), mctx,
            dict(w=w, ru=state.rho_u, fzm=vertical.fzm, fzp=vertical.fzp,
                 adv=coefficient.adv_coefs, adv3=coefficient.adv_coefs_3rd,
                 n_adv=coefficient.n_adv_cells_for_edge,
                 adv_cells=coefficient.adv_cells_for_edge, flux=edge_flux),
            dict(nlev=_i(driver.nlev), ncells=_i(driver.ncells),
                 nedges=_i(driver.nedges), width=_i(width),
                 coefficient=_f(driver.config.config_coef_3rd_order)))
    _launch("w_vertical_flux_f32", driver.ncells * (driver.nlev + 1), mctx,
            dict(w=w, rw=state.rho_w, flux=vertical_flux),
            dict(nlev=_i(driver.nlev), ncells=_i(driver.ncells)))
    return w_finish_metal_v841(
        mesh, driver.v841_context, edge_flux=edge_flux,
        vertical_flux=vertical_flux, rdzu=vertical.rdzu,
        n_vert_levels=driver.nlev)


def _device_state(driver: Any, rho: Any, rtheta: Any, ru: Any, rw: Any,
                  scalars: Any) -> MetalPrognosticState:
    return MetalPrognosticState(
        rho, rtheta, ru, rw, scalars,
        driver.atmosphere.state.time_seconds)


def _device_saved(driver: Any, theta: Any, exner: Any, rho_p: Any, rtheta_p: Any,
                  pressure_p: Any, u: Any, w: Any) -> MetalSavedDiagnostics:
    return MetalSavedDiagnostics(
        theta, exner, rho_p, rtheta_p, pressure_p, u, w)


def _recover_velocities(driver: Any, state: MetalPrognosticState) -> tuple[Any, Any]:
    """recover_state(include_pressure=False): the u and w recovery kernels."""

    atmosphere = driver.atmosphere
    mesh = atmosphere.mesh
    vertical = atmosphere.vertical
    nlev = int(vertical.n_vert_levels)
    ncells = int(mesh.n_cells)
    nedges = int(mesh.n_edges)
    terrain = atmosphere.terrain
    if terrain is not None and nlev < 3:
        raise ValueError("terrain recovery requires nVertLevels>=3 for cf1/cf2/cf3")
    mctx = _context(state.rho)
    normal = driver.arrays.empty(state.rho_u.shape, np.float32)
    vertical_velocity = driver.arrays.empty(state.rho_w.shape, np.float32)
    _launch("recover_edge_velocity_f32", nlev * nedges, mctx,
            dict(rho=state.rho, ru=state.rho_u, cells_on_edge=mesh.cells_on_edge,
                 u=normal),
            dict(nlev=_i(nlev), ncells=_i(ncells), nedges=_i(nedges)))
    if terrain is None:
        _launch("recover_flat_w_f32", (nlev + 1) * ncells, mctx,
                dict(rho=state.rho, rw=state.rho_w, zz=vertical.zz,
                     fzm=vertical.fzm, fzp=vertical.fzp, w=vertical_velocity),
                dict(nlev=_i(nlev), ncells=_i(ncells)))
    else:
        _launch("recover_terrain_w_f32", (nlev + 1) * ncells, mctx,
                dict(rho=state.rho, ru=state.rho_u, rw=state.rho_w,
                     zz=vertical.zz, fzm=vertical.fzm, fzp=vertical.fzp,
                     edges_on_cell=mesh.edges_on_cell,
                     n_edges_on_cell=mesh.n_edges_on_cell,
                     edge_sign_on_cell=mesh.edge_sign_on_cell,
                     zb_cell=terrain.zb_cell, zb3_cell=terrain.zb3_cell,
                     w=vertical_velocity),
                dict(nlev=_i(nlev), ncells=_i(ncells), nedges=_i(nedges),
                     max_edges=_i(mesh.max_edges), cf1=_f(vertical.cf1),
                     cf2=_f(vertical.cf2), cf3=_f(vertical.cf3)))
    return normal, vertical_velocity


def _recover_candidate(driver: Any, saved_state: MetalPrognosticState,
                       saved_diag: MetalSavedDiagnostics, acoustic: Any,
                       stage: int, acoustic_steps: int, scalars: Any):
    regional = getattr(driver, "regional_v841", None)
    arrays = driver.arrays
    atmosphere = driver.atmosphere
    mctx = _context(saved_state.rho)
    rho = arrays.empty(saved_state.rho.shape, np.float32)
    rtheta = arrays.empty(saved_state.rho_theta.shape, np.float32)
    rho_p = arrays.empty(saved_state.rho.shape, np.float32)
    rtheta_p = arrays.empty(saved_state.rho_theta.shape, np.float32)
    theta = arrays.empty(saved_state.rho.shape, np.float32)
    exner = arrays.empty(saved_diag.exner.shape, np.float32)
    pressure_p = arrays.empty(saved_diag.pressure_perturbation.shape, np.float32)
    reference = atmosphere.reference
    _launch("recover_cells_f32", driver.ncells * driver.nlev, mctx,
            dict(rho_p_saved=saved_diag.density_perturbation,
                 rtheta_p_saved=saved_diag.rho_theta_perturbation,
                 rho_pp=acoustic.rho_pp, rtheta_pp=acoustic.rtheta_pp,
                 rho_base=reference.rho_base, rtheta_base=reference.rho_theta_base,
                 exner_base=reference.exner_base, zz=atmosphere.vertical.zz,
                 exner_saved=saved_diag.exner,
                 pressure_p_saved=saved_diag.pressure_perturbation,
                 rho=rho, rtheta=rtheta, rho_p=rho_p, rtheta_p=rtheta_p,
                 theta=theta, exner=exner, pressure_p=pressure_p),
            dict(nlev=_i(driver.nlev), ncells=_i(driver.ncells), stage=_i(stage),
                 rgas=_f(287.0), cp=_f(1004.5), p0=_f(100_000.0)))
    ru = arrays.empty(saved_state.rho_u.shape, np.float32)
    flux_u = arrays.empty(saved_state.rho_u.shape, np.float32)
    _launch("recover_edges_f32", driver.nedges * driver.nlev, mctx,
            dict(ru_saved=saved_state.rho_u, ru_p=acoustic.ru_p,
                 ru_avg_p=acoustic.ru_avg, ru=ru, flux_u=flux_u),
            dict(nlev=_i(driver.nlev), nedges=_i(driver.nedges),
                 acoustic_steps=_i(acoustic_steps)))
    rw = arrays.empty(saved_state.rho_w.shape, np.float32)
    flux_w = arrays.empty(saved_state.rho_w.shape, np.float32)
    _launch("recover_interfaces_f32", driver.ncells * (driver.nlev + 1), mctx,
            dict(rw_saved=saved_state.rho_w, rw_p=acoustic.rw_p,
                 ww_avg_p=acoustic.ww_avg, rw=rw, flux_w=flux_w),
            dict(nlev=_i(driver.nlev), ncells=_i(driver.ncells),
                 acoustic_steps=_i(acoustic_steps)))
    if driver.v841_context is not None:
        enforce_rw_endpoints_metal_v841(rw)
    regional_u_overwrite = None
    if regional is not None:
        # As CUDA: recover in the padded memory model with rho_zz's garbage
        # cell at 1.0, then the ru half of the specified-zone overwrite.
        regional.bind_state_rho(rho)
        regional_u_overwrite = regional.overwrite_speczone_rho_u(ru, stage)
    state = _device_state(driver, rho, rtheta, ru, rw, scalars)
    normal_velocity, vertical_velocity = _recover_velocities(driver, state)
    if regional is not None:
        # The u half with the same driving values, then the zero-gradient w.
        regional.overwrite_speczone_u(normal_velocity, regional_u_overwrite)
        regional.zero_speczone_w(vertical_velocity)
    diagnostics = _device_saved(driver, theta, exner, rho_p, rtheta_p, pressure_p,
                                normal_velocity, vertical_velocity)
    return state, diagnostics, flux_u, flux_w


def _copy_state(driver: Any, source: MetalPrognosticState, *,
                share_scalars: bool = False) -> MetalPrognosticState:
    return _device_state(
        driver, source.rho.copy(), source.rho_theta.copy(), source.rho_u.copy(),
        source.rho_w.copy(),
        source.scalars if share_scalars else source.scalars.copy())


def _copy_saved(driver: Any, source: MetalSavedDiagnostics) -> MetalSavedDiagnostics:
    return _device_saved(
        driver, source.theta_m.copy(), source.exner.copy(),
        source.density_perturbation.copy(),
        source.rho_theta_perturbation.copy(),
        source.pressure_perturbation.copy(), source.normal_velocity.copy(),
        source.vertical_velocity.copy())


__all__ = [
    "accumulate_finite_array_metal_v841",
    "accumulate_split_flux_metal_v841",
    "enforce_rw_endpoints_metal_v841",
    "finish_split_flux_metal_v841",
    "reference_wind_edge_metal_v841",
    "theta_finish_metal_v841",
    "validate_positive_state_metal_v841",
    "validate_recovered_state_metal_v841",
    "vector_momentum_tendency_metal_v841",
    "w_finish_metal_v841",
]
