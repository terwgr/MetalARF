"""Metal mirror of the MPAS-A v8.4.1 CUDA atmosphere scalar transport.

``advance_scalar_transport_metal_v841`` follows
``cuda_transport_v841.advance_scalar_transport_cuda_v841`` on the path the
v8.4.1 dry step uses (standard third-order stage update).  Every array is a
resident ``MetalArray``; the only readback is the validation flag the CUDA
function itself reads with ``cp.asnumpy``.  The monotonic (FCT) kernels are
not mirrored and the monotonic dispatch refuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .cuda_transport import _mesh_value, _validate_orders
from .errors import ConfigurationRefusal
from .metal_v841_launch import launch


def _array(value: Any, name: str, dtype: Any) -> Any:
    from woof.metal.array import MetalArray

    if not isinstance(value, MetalArray):
        raise TypeError(f"{name} must be a resident MetalArray")
    if value.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}")
    return value


def _f32(value: Any, name: str) -> Any:
    return _array(value, name, np.float32)


def _i32(value: Any) -> Any:
    return _array(value, "connectivity", np.int32)


@dataclass(frozen=True, slots=True)
class MetalAdvectionCoefficients:
    adv_coefs: Any
    adv_coefs_3rd: Any
    n_adv_cells_for_edge: Any
    adv_cells_for_edge: Any
    h2d_bytes: int
    horizontal_order: int = 3

    @classmethod
    def from_host(cls, context: Any, coefficients: Any) -> "MetalAdvectionCoefficients":
        """Upload the host advection coefficients once."""

        from woof.metal.array import MetalArrayModule

        if int(coefficients.horizontal_order) != 3:
            raise ConfigurationRefusal(
                "config_scalar_adv_order",
                coefficients.horizontal_order,
                "the Metal transport admits the frozen order-three path",
                "config_scalar_adv_order=3",
            )
        host_float = {
            "adv_coefs": np.ascontiguousarray(
                np.asarray(coefficients.adv_coefs, dtype=np.float32)
            ),
            "adv_coefs_3rd": np.ascontiguousarray(
                np.asarray(coefficients.adv_coefs_3rd, dtype=np.float32)
            ),
        }
        host_index = {
            "n_adv_cells_for_edge": np.ascontiguousarray(
                np.asarray(coefficients.n_adv_cells_for_edge, dtype=np.int32)
            ),
            "adv_cells_for_edge": np.ascontiguousarray(
                np.asarray(coefficients.adv_cells_for_edge, dtype=np.int32)
            ),
        }
        if host_float["adv_coefs"].ndim != 2:
            raise ValueError("adv_coefs must have shape (nEdges, stencilWidth)")
        nedges, width = host_float["adv_coefs"].shape
        expected = (nedges, width)
        if host_float["adv_coefs_3rd"].shape != expected:
            raise ValueError("adv_coefs_3rd shape differs from adv_coefs")
        if host_index["n_adv_cells_for_edge"].shape != (nedges,):
            raise ValueError("n_adv_cells_for_edge must have shape (nEdges,)")
        if host_index["adv_cells_for_edge"].shape != expected:
            raise ValueError("adv_cells_for_edge shape differs from adv_coefs")
        if not all(np.all(np.isfinite(value)) for value in host_float.values()):
            raise ValueError("advection coefficients contain non-finite values")
        module = MetalArrayModule(context)
        device = {
            name: module.asarray(value)
            for name, value in {**host_float, **host_index}.items()
        }
        result = cls(
            **device,
            h2d_bytes=sum(int(value.nbytes) for value in device.values()),
        )
        result.validate()
        return result

    def validate(self) -> None:
        shape = tuple(self.adv_coefs.shape)
        if len(shape) != 2:
            raise ValueError("adv_coefs must have shape (nEdges, stencilWidth)")
        for name, value, dtype, expected in (
            ("adv_coefs", self.adv_coefs, np.float32, shape),
            ("adv_coefs_3rd", self.adv_coefs_3rd, np.float32, shape),
            ("n_adv_cells_for_edge", self.n_adv_cells_for_edge, np.int32,
             (shape[0],)),
            ("adv_cells_for_edge", self.adv_cells_for_edge, np.int32, shape),
        ):
            _array(value, f"advection.{name}", dtype)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"advection.{name} must have shape {expected}, "
                    f"got {tuple(value.shape)}"
                )


@dataclass(frozen=True, slots=True)
class MetalScalarTransportResult:
    scalars: Any
    density: Any


def _module(context: Any) -> Any:
    from woof.metal.array import MetalArrayModule

    return MetalArrayModule(context)


def _inputs(
    mesh: Any,
    v841: Any,
    scalar_old: Any,
    scalar_stage: Any,
    rho_zz_old: Any,
    rho_zz_new: Any,
    uh_avg: Any,
    ww_avg: Any,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
) -> tuple[Any, ...]:
    old = _f32(scalar_old, "scalar_old")
    stage = _f32(scalar_stage, "scalar_stage")
    squeezed = old.ndim == 2
    if squeezed:
        old = old.view((1, *old.shape))
        stage = stage.view((1, *stage.shape))
    if old.ndim != 3 or old.shape != stage.shape:
        raise ValueError("scalar_old/stage must share (nTracers,nLevels,nCells)")
    ntracers, nlev, ncells = map(int, old.shape)
    horizontal = _f32(uh_avg, "uh_avg")
    vertical = _f32(ww_avg, "ww_avg")
    if horizontal.ndim != 2 or horizontal.shape[0] != nlev:
        raise ValueError("uh_avg must have shape (nLevels,nEdges)")
    nedges = int(horizontal.shape[1])
    if vertical.shape != (nlev + 1, ncells):
        raise ValueError("ww_avg must have shape (nLevels+1,nCells)")
    rho0 = _f32(rho_zz_old, "rho_zz_old")
    rho1 = _f32(rho_zz_new, "rho_zz_new")
    if rho0.shape != (nlev, ncells) or rho1.shape != rho0.shape:
        raise ValueError("rho_zz arrays disagree with scalar shape")
    values = (
        old, stage, squeezed, rho0, rho1, horizontal, vertical,
        _f32(fzm, "fzm"), _f32(fzp, "fzp"), _f32(rdzw, "rdzw"),
        ntracers, nlev, ncells, nedges,
    )
    validate = getattr(v841, "validate", None)
    if validate is not None:
        validate(
            n_vert_levels=nlev, n_cells=ncells, n_edges=nedges,
            n_vertices=int(mesh.n_vertices),
        )
    for name, value in (("fzm", values[7]), ("fzp", values[8]), ("rdzw", values[9])):
        if tuple(value.shape) != (nlev,):
            raise ValueError(f"{name} must have shape ({nlev},)")
    return values


def _geometry(mesh: Any) -> tuple[Any, ...]:
    counts = _i32(_mesh_value(mesh, "n_edges_on_cell"))
    edges = _i32(_mesh_value(mesh, "edges_on_cell"))
    neighbors = _i32(_mesh_value(mesh, "cells_on_cell"))
    cells_on_edge = _i32(_mesh_value(mesh, "cells_on_edge"))
    signs = _f32(_mesh_value(mesh, "edge_sign_on_cell"), "edge_sign_on_cell")
    return (
        counts, edges, neighbors, cells_on_edge, signs,
        _f32(_mesh_value(mesh, "dv_edge"), "dv_edge"),
        _f32(_mesh_value(mesh, "area_cell"), "area_cell"),
        int(edges.shape[1]),
    )


def _check_density(
    context: Any,
    density: Any,
    *,
    name: str,
    count: int,
    invalid_flag: Any | None,
) -> Any:
    module = _module(context)
    owns_flag = invalid_flag is None
    invalid = (
        module.zeros((1,), dtype=np.int32)
        if owns_flag
        else _i32(invalid_flag)
    )
    if tuple(invalid.shape) != (1,):
        raise ValueError("validation_flag must have shape (1,)")
    launch(
        context, "hexv841_validate_density_v841", count,
        {"density": density, "invalid": invalid},
        {"nvalues": int(count)},
    )
    if owns_flag and int(invalid.to_numpy()[0]) != 0:
        raise ValueError(f"{name} must remain finite and strictly positive")
    return invalid


def _validate_geometry_and_coefficients(
    context: Any,
    mesh: Any,
    coefficients: MetalAdvectionCoefficients,
    *,
    ncells: int,
    nedges: int,
    indices_prevalidated: bool,
) -> tuple[Any, ...]:
    coefficients.validate()
    coefficient_shape = tuple(coefficients.adv_coefs.shape)
    if len(coefficient_shape) != 2 or coefficient_shape[0] != nedges:
        raise ValueError("advection coefficient edge extent differs from mesh")
    width = int(coefficient_shape[1])
    geometry = _geometry(mesh)
    counts, edges, neighbors, cells_on_edge, signs, dv, area, max_edges = geometry
    expected = {
        "n_edges_on_cell": (counts, (ncells,)),
        "edges_on_cell": (edges, (ncells, max_edges)),
        "cells_on_cell": (neighbors, (ncells, max_edges)),
        "cells_on_edge": (cells_on_edge, (nedges, 2)),
        "edge_sign_on_cell": (signs, (ncells, max_edges)),
        "dv_edge": (dv, (nedges,)),
        "area_cell": (area, (ncells,)),
    }
    for name, (value, shape) in expected.items():
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} shape {tuple(value.shape)} != {shape}")
    if not indices_prevalidated:
        invalid = _module(context).zeros((1,), dtype=np.int32)
        launch(
            context, "hexv841_validate_transport_indices_v841",
            max(ncells, nedges),
            {
                "n_edges_on_cell": counts, "edges_on_cell": edges,
                "cells_on_cell": neighbors, "cells_on_edge": cells_on_edge,
                "n_adv": coefficients.n_adv_cells_for_edge,
                "adv_cells": coefficients.adv_cells_for_edge,
                "invalid": invalid,
            },
            {
                "ncells": ncells, "nedges": nedges, "max_edges": max_edges,
                "width": width,
            },
        )
        if int(invalid.to_numpy()[0]) != 0:
            raise ValueError(
                "active transport connectivity or stencil index is out of range"
            )
    return geometry


def _standard_high_fluxes(
    context: Any,
    stage: Any,
    velocity: Any,
    vertical_velocity: Any,
    coefficients: MetalAdvectionCoefficients,
    fzm: Any,
    fzp: Any,
    coefficient: float,
    ntracers: int,
    nlev: int,
    ncells: int,
    nedges: int,
) -> tuple[Any, Any]:
    module = _module(context)
    edge = module.empty((ntracers, nlev, nedges), dtype=np.float32)
    vertical = module.empty((ntracers, nlev + 1, ncells), dtype=np.float32)
    width = int(coefficients.adv_coefs.shape[1])
    launch(
        context, "hexv841_transport_edge_values", nedges,
        {
            "stage": stage, "velocity": velocity,
            "adv": coefficients.adv_coefs,
            "adv3": coefficients.adv_coefs_3rd,
            "n_adv": coefficients.n_adv_cells_for_edge,
            "adv_cells": coefficients.adv_cells_for_edge,
            "edge_values": edge,
        },
        {
            "ntracers": ntracers, "nlev": nlev, "ncells": ncells,
            "nedges": nedges, "width": width,
            "coefficient": float(np.float32(coefficient)),
        },
    )
    # The CUDA v8.4.1 launch covers ncells threads and the kernel's
    # grid-stride loop walks every element; one thread per element gives the
    # same per-element arithmetic.
    launch(
        context, "hexv841_transport_vertical_flux",
        ntracers * (nlev + 1) * ncells,
        {
            "stage": stage, "velocity": vertical_velocity, "fzm": fzm,
            "fzp": fzp, "vertical_flux": vertical,
        },
        {
            "ntracers": ntracers, "nlev": nlev, "ncells": ncells,
            "coefficient": float(np.float32(coefficient)),
        },
    )
    return edge, vertical


def _source(context: Any, old: Any, scalar_tendency: Any | None, *, squeezed: bool) -> Any:
    if scalar_tendency is None:
        return _module(context).zeros(old.shape, dtype=np.float32)
    source = _f32(scalar_tendency, "scalar_tendency")
    expected_ndim = 2 if squeezed else 3
    if source.ndim != expected_ndim:
        raise ValueError("scalar_tendency carrier rank differs from scalar_old")
    if squeezed:
        source = source.view((1, *source.shape))
    if source.shape != old.shape:
        raise ValueError("scalar_tendency shape differs from scalar_old")
    return source


def advance_scalars_metal_v841(
    mesh: Any,
    v841: Any,
    scalar_old: Any,
    scalar_stage: Any,
    rho_zz_old: Any,
    rho_zz_new: Any,
    uh_avg: Any,
    ww_avg: Any,
    dt: float,
    *,
    coefficients: MetalAdvectionCoefficients,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
    rk_step: int,
    config_time_integration_order: int,
    config_scalar_adv_order: int,
    config_scalar_vadv_order: int,
    config_coef_3rd_order: float,
    scalar_tendency: Any | None = None,
    advance_density: bool = False,
    validation_flag: Any | None = None,
    indices_prevalidated: bool = False,
    rho_old_prevalidated: bool = False,
    halo_exchange: Any | None = None,
) -> MetalScalarTransportResult:
    """Standard (non-monotonic) third-order RK stage update."""

    if halo_exchange is not None:
        raise ConfigurationRefusal(
            "halo_exchange", "set", "the Metal transport has no halo exchange",
            "halo_exchange=None",
        )
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    _validate_orders(
        config_scalar_adv_order, config_scalar_vadv_order,
        config_coef_3rd_order,
    )
    if rk_step not in (1, 2, 3) or config_time_integration_order != 3:
        raise ConfigurationRefusal(
            "config_time_integration_order", config_time_integration_order,
            "the v8.4.1 CUDA scalar path is RK3", "config_time_integration_order=3",
        )
    values = _inputs(
        mesh, v841, scalar_old, scalar_stage, rho_zz_old, rho_zz_new,
        uh_avg, ww_avg, fzm, fzp, rdzw,
    )
    (
        old, stage, squeezed, rho0, rho1, velocity, vertical_velocity,
        fnm, fnp, rdnw, ntracers, nlev, ncells, nedges,
    ) = values
    context = old.buffer.context
    module = _module(context)
    source = _source(context, old, scalar_tendency, squeezed=squeezed)
    counts, edges, _neighbors, _cells, signs, _dv, _area, max_edges = (
        _validate_geometry_and_coefficients(
            context, mesh, coefficients, ncells=ncells, nedges=nedges,
            indices_prevalidated=indices_prevalidated,
        )
    )
    if not rho_old_prevalidated:
        _check_density(
            context, rho0, name="rho_zz_old", count=nlev * ncells,
            invalid_flag=None,
        )
    edge_values, vertical_flux = _standard_high_fluxes(
        context, stage, velocity, vertical_velocity, coefficients, fnm, fnp,
        config_coef_3rd_order, ntracers, nlev, ncells, nedges,
    )
    weight = np.float32(
        1.0 / 3.0 if advance_density and rk_step == 1
        else 0.5 if advance_density and rk_step == 2
        else 1.0
    )
    target = module.empty(rho0.shape, dtype=np.float32)
    launch(
        context, "hexv841_transport_interpolate_target_v841", nlev * ncells,
        {"rho_old": rho0, "rho_new": rho1, "target": target},
        {"nvalues": nlev * ncells, "weight": float(weight)},
    )
    _check_density(
        context, target, name="target_density", count=nlev * ncells,
        invalid_flag=validation_flag,
    )
    output = module.empty(old.shape, dtype=np.float32)
    launch(
        context, "hexv841_transport_standard_finish_v841", ncells,
        {
            "n_edges_on_cell": counts, "edges_on_cell": edges,
            "acoustic_sign": signs, "inv_area_cell": v841.inv_area_cell,
            "velocity": velocity, "edge_values": edge_values,
            "vertical_flux": vertical_flux, "rdzw": rdnw, "old": old,
            "rho_old": rho0, "target_density": target, "source": source,
            "output": output,
        },
        {
            "ntracers": ntracers, "nlev": nlev, "ncells": ncells,
            "nedges": nedges, "max_edges": max_edges,
            "dt": float(np.float32(dt)),
        },
    )
    return MetalScalarTransportResult(output[0] if squeezed else output, target)


def advance_scalar_transport_metal_v841(
    mesh: Any,
    v841: Any,
    scalar_old: Any,
    scalar_stage: Any,
    rho_zz_old: Any,
    rho_zz_new: Any,
    uh_avg: Any,
    ww_avg: Any,
    dt: float,
    *,
    coefficients: MetalAdvectionCoefficients,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
    rk_step: int,
    config_scalar_advection: bool,
    config_monotonic: bool,
    config_positive_definite: bool,
    config_split_dynamics_transport: bool,
    config_time_integration_order: int,
    **kwargs: Any,
) -> MetalScalarTransportResult:
    if rk_step not in (1, 2, 3):
        raise ConfigurationRefusal(
            "rk_step", rk_step, "the v8.4.1 CUDA scalar RK has three stages",
            "rk_step=1, 2, or 3",
        )
    if config_time_integration_order != 3:
        raise ConfigurationRefusal(
            "config_time_integration_order", config_time_integration_order,
            "the v8.4.1 CUDA scalar lane mirrors RK3",
            "config_time_integration_order=3",
        )
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    stage = _f32(scalar_stage, "scalar_stage")
    rho_new = _f32(rho_zz_new, "rho_zz_new")
    if not config_scalar_advection:
        return MetalScalarTransportResult(stage.copy(), rho_new.copy())
    if rk_step >= 3 and (config_monotonic or config_positive_definite):
        raise ConfigurationRefusal(
            "config_monotonic", config_monotonic,
            "the Metal transport mirrors the non-monotonic path only",
            "config_monotonic=False, config_positive_definite=False",
        )
    return advance_scalars_metal_v841(
        mesh, v841, scalar_old, scalar_stage, rho_zz_old, rho_zz_new,
        uh_avg, ww_avg, dt, rk_step=rk_step,
        config_time_integration_order=config_time_integration_order,
        coefficients=coefficients, fzm=fzm, fzp=fzp, rdzw=rdzw,
        advance_density=config_split_dynamics_transport, **kwargs,
    )


__all__ = [
    "MetalAdvectionCoefficients",
    "MetalScalarTransportResult",
    "advance_scalar_transport_metal_v841",
    "advance_scalars_metal_v841",
]
