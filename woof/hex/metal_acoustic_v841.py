"""Metal mirror of the CUDA MPAS-A v8.4.1 acoustic step.

Each function follows its CUDA namesake in ``cuda_acoustic_v841.py`` (and
``convert_w_tendency_to_omega_cuda`` in ``cuda_acoustic.py``) call for call,
on resident ``MetalArray`` storage, with one ``hexv841_*`` shader per CUDA
kernel.  Arrays keep the logical ``(level, entity)`` order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from woof.metal.array import MetalArray, MetalArrayModule

from .metal_v841_launch import launch


def _float32(value: Any, name: str) -> MetalArray:
    if not isinstance(value, MetalArray):
        raise TypeError(f"{name} must be a resident MetalArray")
    if value.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} must have dtype float32")
    return value


def _int32(value: Any, name: str) -> MetalArray:
    if not isinstance(value, MetalArray):
        raise TypeError(f"{name} must be a resident MetalArray")
    if value.dtype != np.dtype(np.int32):
        raise TypeError(f"{name} must have dtype int32")
    return value


def _mesh_value(mesh: Any, name: str) -> Any:
    if hasattr(mesh, name):
        return getattr(mesh, name)
    camel = {
        "n_edges_on_cell": "nEdgesOnCell",
        "edges_on_cell": "edgesOnCell",
        "cells_on_edge": "cellsOnEdge",
        "edge_sign_on_cell": "edgeSignOnCell",
        "dv_edge": "dvEdge",
        "dc_edge": "dcEdge",
        "area_cell": "areaCell",
    }.get(name)
    if camel is not None and hasattr(mesh, camel):
        return getattr(mesh, camel)
    arrays = getattr(mesh, "arrays", None)
    if arrays is not None:
        if name in arrays:
            return arrays[name]
        if camel is not None and camel in arrays:
            return arrays[camel]
    raise AttributeError(f"device mesh has no field {name!r}")


def _module(*arrays: MetalArray) -> MetalArrayModule:
    return MetalArrayModule(arrays[0].buffer.context)


@dataclass(frozen=True, slots=True)
class MetalVerticalImplicitCoefficients:
    cofwr: Any
    cofwz: Any
    coftz: Any
    cofwt: Any
    cofrz: Any
    a_tri: Any
    b_tri: Any
    c_tri: Any
    alpha_tri: Any
    gamma_tri: Any


@dataclass(slots=True)
class MetalAcousticState:
    ru_p: Any
    rw_p: Any
    rtheta_pp: Any
    rtheta_pp_old: Any
    rho_pp: Any
    ru_avg: Any
    ww_avg: Any

    @classmethod
    def zeros(
        cls, arrays: MetalArrayModule, nlev: int, ncells: int, nedges: int
    ) -> "MetalAcousticState":
        return cls(
            ru_p=arrays.zeros((nlev, nedges), np.float32),
            rw_p=arrays.zeros((nlev + 1, ncells), np.float32),
            rtheta_pp=arrays.zeros((nlev, ncells), np.float32),
            rtheta_pp_old=arrays.zeros((nlev, ncells), np.float32),
            rho_pp=arrays.zeros((nlev, ncells), np.float32),
            ru_avg=arrays.zeros((nlev, nedges), np.float32),
            ww_avg=arrays.zeros((nlev + 1, ncells), np.float32),
        )

    def copy(self) -> "MetalAcousticState":
        return MetalAcousticState(
            **{name: getattr(self, name).copy() for name in self.__slots__}
        )


@dataclass(frozen=True, slots=True)
class MetalAcousticForcing:
    rho_zz: Any
    theta_m: Any
    zz: Any
    exner: Any
    cqu: Any
    zxu: Any
    dss: Any
    tend_ru: Any
    tend_rho: Any
    tend_rt: Any
    tend_rw: Any
    w: Any
    rw: Any
    rw_save: Any


def compute_vertical_implicit_coefficients_metal_v841(
    *,
    dts: float,
    context: Any,
    zz: Any,
    cqw: Any,
    exner: Any,
    theta: Any,
    rho_base: Any,
    rho_theta_base: Any,
    exner_base: Any,
    rho_theta_perturbation: Any,
    qtot: Any,
    rdzw: Any,
    fzm: Any,
    fzp: Any,
    rdzu: Any,
    gravity: float = 9.80616,
    rgas: float = 287.0,
    cp: float = 1004.5,
    validation_flag: Any | None = None,
) -> MetalVerticalImplicitCoefficients:
    """Build released unscaled v8.4.1 coefficients on the device."""

    if not np.isfinite(dts) or dts <= 0.0:
        raise ValueError("dts must be finite and positive")
    if (
        not all(np.isfinite(value) for value in (gravity, rgas, cp))
        or gravity <= 0.0
        or rgas <= 0.0
        or cp <= rgas
    ):
        raise ValueError("gravity, rgas, and cp must define a finite dry atmosphere")
    zz = _float32(zz, "zz")
    if zz.ndim != 2:
        raise ValueError("zz must have shape (nVertLevels,nCells)")
    arrays = _module(zz)
    nlev, ncells = map(int, zz.shape)
    cell_shape = (nlev, ncells)
    for name, shape in {
        "etp": (nlev,),
        "etm": (nlev,),
        "ewp": (nlev + 1,),
        "ewm": (nlev + 1,),
    }.items():
        value = _float32(getattr(context, name), f"context.{name}")
        if value.shape != shape:
            raise ValueError(f"context.{name} shape {value.shape} != {shape}")
    inputs: dict[str, Any] = {}
    for name, value in {
        "cqw": cqw,
        "exner": exner,
        "theta": theta,
        "rho_base": rho_base,
        "rho_theta_base": rho_theta_base,
        "exner_base": exner_base,
        "rho_theta_perturbation": rho_theta_perturbation,
        "qtot": qtot,
    }.items():
        array = _float32(value, name)
        if array.shape != cell_shape:
            raise ValueError(f"{name} shape {array.shape} != {cell_shape}")
        inputs[name] = array
    vertical: dict[str, Any] = {}
    for name, value in {"rdzw": rdzw, "fzm": fzm, "fzp": fzp, "rdzu": rdzu}.items():
        array = _float32(value, name)
        if array.shape != (nlev,):
            raise ValueError(f"{name} shape {array.shape} != ({nlev},)")
        vertical[name] = array

    def cell_array() -> Any:
        return arrays.empty(cell_shape, np.float32)

    result = MetalVerticalImplicitCoefficients(
        cofwr=cell_array(),
        cofwz=cell_array(),
        coftz=arrays.empty((nlev + 1, ncells), np.float32),
        cofwt=cell_array(),
        cofrz=arrays.empty((nlev,), np.float32),
        a_tri=cell_array(),
        b_tri=cell_array(),
        c_tri=cell_array(),
        alpha_tri=cell_array(),
        gamma_tri=cell_array(),
    )
    owns_flag = validation_flag is None
    if owns_flag:
        singular = arrays.zeros((1,), np.int32)
    else:
        singular = _int32(validation_flag, "validation_flag")
        if singular.shape != (1,):
            raise ValueError("validation_flag must have shape (1,)")
    ctx = zz.buffer.context
    launch(
        ctx, "hexv841_acoustic_cofrz_v841", nlev,
        {"rdzw": vertical["rdzw"], "cofrz": result.cofrz},
        {"nlev": np.int32(nlev)},
    )
    launch(
        ctx, "hexv841_acoustic_coefficients_v841", ncells,
        {
            "zz": zz, "cqw": inputs["cqw"], "exner": inputs["exner"],
            "theta": inputs["theta"], "rho_base": inputs["rho_base"],
            "rtheta_base": inputs["rho_theta_base"],
            "exner_base": inputs["exner_base"],
            "rtheta_p": inputs["rho_theta_perturbation"],
            "qtot": inputs["qtot"], "rdzw": vertical["rdzw"],
            "fzm": vertical["fzm"], "fzp": vertical["fzp"],
            "rdzu": vertical["rdzu"], "etp": context.etp, "ewp": context.ewp,
            "cofwr": result.cofwr, "cofwz": result.cofwz,
            "coftz": result.coftz, "cofwt": result.cofwt,
            "cofrz": result.cofrz, "a_tri": result.a_tri,
            "b_tri": result.b_tri, "c_tri": result.c_tri,
            "alpha_tri": result.alpha_tri, "gamma_tri": result.gamma_tri,
            "singular": singular,
        },
        {
            "nlev": np.int32(nlev), "ncells": np.int32(ncells),
            "dts": np.float32(dts), "gravity": np.float32(gravity),
            "rgas": np.float32(rgas), "cp": np.float32(cp),
        },
    )
    if owns_flag and int(singular.to_numpy()[0]) != 0:
        raise FloatingPointError("singular v8.4.1 vertical acoustic system")
    return result


def convert_w_tendency_to_omega_metal(
    mesh: Any,
    w_tendency: Any,
    u_tendency: Any,
    *,
    fzm: Any,
    fzp: Any,
    zz: Any,
    zb_cell: Any,
    zb3_cell: Any,
) -> Any:
    """Apply the terrain w-tendency coupling on-device."""

    out = _float32(w_tendency, "w_tendency").copy()
    u = _float32(u_tendency, "u_tendency")
    nlev, nedges = map(int, u.shape)
    if out.ndim != 2 or out.shape[0] != nlev + 1:
        raise ValueError("w_tendency must have shape (nVertLevels+1,nCells)")
    ncells = int(out.shape[1])
    edges = _int32(_mesh_value(mesh, "edges_on_cell"), "edges_on_cell")
    counts = _int32(_mesh_value(mesh, "n_edges_on_cell"), "n_edges_on_cell")
    signs = _float32(_mesh_value(mesh, "edge_sign_on_cell"), "edge_sign_on_cell")
    max_edges = int(edges.shape[1])
    expected_metric = (nlev + 1, ncells, max_edges)
    zb = _float32(zb_cell, "zb_cell")
    zb3 = _float32(zb3_cell, "zb3_cell")
    if zb.shape != expected_metric or zb3.shape != expected_metric:
        raise ValueError(f"terrain metrics must have shape {expected_metric}")
    launch(
        out.buffer.context, "hexv841_tendency_w_to_omega",
        (nlev - 1) * ncells,
        {
            "n_edges_on_cell": counts, "edges_on_cell": edges,
            "edge_sign_on_cell": signs, "u_tendency": u,
            "fzm": _float32(fzm, "fzm"), "fzp": _float32(fzp, "fzp"),
            "zz": _float32(zz, "zz"), "zb": zb, "zb3": zb3,
            "omega_tendency": out,
        },
        {
            "nlev": np.int32(nlev), "ncells": np.int32(ncells),
            "nedges": np.int32(nedges), "max_edges": np.int32(max_edges),
        },
    )
    return out


def advance_acoustic_step_metal_v841(
    mesh: Any,
    state: MetalAcousticState,
    forcing: MetalAcousticForcing,
    coefficients: MetalVerticalImplicitCoefficients,
    *,
    context: Any,
    dts: float,
    small_step: int,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
    gravity: float = 9.80616,
    rgas: float = 287.0,
    cp: float = 1004.5,
    in_place: bool = True,
) -> MetalAcousticState:
    """Advance one closed/global released v8.4.1 acoustic substep."""

    if small_step < 1:
        raise ValueError("small_step must be at least one")
    if not np.isfinite(dts) or dts <= 0.0:
        raise ValueError("dts must be finite and positive")
    if (
        not all(np.isfinite(value) for value in (gravity, rgas, cp))
        or gravity <= 0.0
        or rgas <= 0.0
        or cp <= rgas
    ):
        raise ValueError("gravity, rgas, and cp must define a finite dry atmosphere")
    out = state if in_place else state.copy()
    nlev, nedges = map(int, out.ru_p.shape)
    ncells = int(out.rho_pp.shape[1])
    if out.rho_pp.shape != (nlev, ncells) or out.rw_p.shape != (nlev + 1, ncells):
        raise ValueError("acoustic state shapes are inconsistent")
    if int(mesh.n_cells) != ncells or int(mesh.n_edges) != nedges:
        raise ValueError("acoustic state dimensions differ from device mesh")
    state_shapes = {
        "ru_p": (nlev, nedges),
        "ru_avg": (nlev, nedges),
        "rw_p": (nlev + 1, ncells),
        "ww_avg": (nlev + 1, ncells),
        "rho_pp": (nlev, ncells),
        "rtheta_pp": (nlev, ncells),
        "rtheta_pp_old": (nlev, ncells),
    }
    for name, shape in state_shapes.items():
        value = _float32(getattr(out, name), f"state.{name}")
        if value.shape != shape:
            raise ValueError(f"state.{name} shape {value.shape} != {shape}")
    forcing_shapes = {
        "rho_zz": (nlev, ncells),
        "theta_m": (nlev, ncells),
        "zz": (nlev, ncells),
        "exner": (nlev, ncells),
        "cqu": (nlev, nedges),
        "zxu": (nlev, nedges),
        "dss": (nlev, ncells),
        "tend_ru": (nlev, nedges),
        "tend_rho": (nlev, ncells),
        "tend_rt": (nlev, ncells),
        "tend_rw": (nlev + 1, ncells),
        "w": (nlev + 1, ncells),
        "rw": (nlev + 1, ncells),
        "rw_save": (nlev + 1, ncells),
    }
    for name, shape in forcing_shapes.items():
        value = _float32(getattr(forcing, name), f"forcing.{name}")
        if value.shape != shape:
            raise ValueError(f"forcing.{name} shape {value.shape} != {shape}")
    coefficient_shapes = {
        "cofwr": (nlev, ncells),
        "cofwz": (nlev, ncells),
        "coftz": (nlev + 1, ncells),
        "cofwt": (nlev, ncells),
        "cofrz": (nlev,),
        "a_tri": (nlev, ncells),
        "b_tri": (nlev, ncells),
        "c_tri": (nlev, ncells),
        "alpha_tri": (nlev, ncells),
        "gamma_tri": (nlev, ncells),
    }
    for name, shape in coefficient_shapes.items():
        value = _float32(getattr(coefficients, name), f"coefficients.{name}")
        if value.shape != shape:
            raise ValueError(f"coefficients.{name} shape {value.shape} != {shape}")
    cells_on_edge = _int32(_mesh_value(mesh, "cells_on_edge"), "cells_on_edge")
    edges_on_cell = _int32(_mesh_value(mesh, "edges_on_cell"), "edges_on_cell")
    counts = _int32(_mesh_value(mesh, "n_edges_on_cell"), "n_edges_on_cell")
    signs = _float32(_mesh_value(mesh, "edge_sign_on_cell"), "edge_sign_on_cell")
    dv_edge = _float32(_mesh_value(mesh, "dv_edge"), "dv_edge")
    max_edges = int(edges_on_cell.shape[1])
    fzm = _float32(fzm, "fzm")
    fzp = _float32(fzp, "fzp")
    rdzw = _float32(rdzw, "rdzw")
    for name, value in (("fzm", fzm), ("fzp", fzp), ("rdzw", rdzw)):
        if value.shape != (nlev,):
            raise ValueError(f"{name} shape {value.shape} != ({nlev},)")

    arrays = _module(out.rho_pp)
    ctx = out.rho_pp.buffer.context
    launch(
        ctx, "hexv841_acoustic_ru_v841", nedges,
        {
            "cells_on_edge": cells_on_edge, "inv_dc_edge": context.inv_dc_edge,
            "zz": forcing.zz, "exner": forcing.exner, "cqu": forcing.cqu,
            "zxu": forcing.zxu, "tend_ru": forcing.tend_ru,
            "rho_pp": out.rho_pp, "rtheta_pp": out.rtheta_pp,
            "ru_p": out.ru_p, "ru_avg": out.ru_avg,
        },
        {
            "nlev": np.int32(nlev), "nedges": np.int32(nedges),
            "ncells": np.int32(ncells), "small_step": np.int32(small_step),
            "dts": np.float32(dts), "gravity": np.float32(gravity),
            "rgas": np.float32(rgas), "cp": np.float32(cp),
        },
    )
    launch(
        ctx, "hexv841_acoustic_prepare_v841", ncells,
        {
            "rw_p": out.rw_p, "rtheta_pp": out.rtheta_pp,
            "rtheta_pp_old": out.rtheta_pp_old, "rho_pp": out.rho_pp,
            "ww_avg": out.ww_avg,
        },
        {
            "nlev": np.int32(nlev), "ncells": np.int32(ncells),
            "small_step": np.int32(small_step),
        },
    )
    rs = arrays.empty((nlev, ncells), np.float32)
    ts = arrays.empty((nlev, ncells), np.float32)
    launch(
        ctx, "hexv841_acoustic_rs_ts_v841", ncells,
        {
            "edge_sign_on_cell": signs, "dv_edge": dv_edge,
            "inv_area_cell": context.inv_area_cell,
            "theta_m": forcing.theta_m, "rdzw": rdzw,
            "cofrz": coefficients.cofrz, "coftz": coefficients.coftz,
            "ewm": context.ewm, "ru_p": out.ru_p, "rw_p": out.rw_p,
            "rho_pp": out.rho_pp, "rtheta_pp": out.rtheta_pp,
            "tend_rho": forcing.tend_rho, "tend_rt": forcing.tend_rt,
            "n_edges_on_cell": counts, "edges_on_cell": edges_on_cell,
            "cells_on_edge": cells_on_edge, "rs": rs, "ts": ts,
        },
        {
            "nlev": np.int32(nlev), "ncells": np.int32(ncells),
            "nedges": np.int32(nedges), "max_edges": np.int32(max_edges),
            "dts": np.float32(dts),
        },
    )
    launch(
        ctx, "hexv841_acoustic_column_solve_v841", ncells,
        {
            "zz": forcing.zz, "rho_zz": forcing.rho_zz, "fzm": fzm,
            "fzp": fzp, "rdzw": rdzw, "dss": forcing.dss, "w": forcing.w,
            "rw": forcing.rw, "rw_save": forcing.rw_save,
            "tend_rw": forcing.tend_rw, "rs": rs, "ts": ts,
            "cofwr": coefficients.cofwr, "cofwz": coefficients.cofwz,
            "coftz": coefficients.coftz, "cofwt": coefficients.cofwt,
            "cofrz": coefficients.cofrz, "a_tri": coefficients.a_tri,
            "alpha_tri": coefficients.alpha_tri,
            "gamma_tri": coefficients.gamma_tri,
            "etp": context.etp, "etm": context.etm, "ewp": context.ewp,
            "ewm": context.ewm, "rw_p": out.rw_p, "rho_pp": out.rho_pp,
            "rtheta_pp": out.rtheta_pp, "ww_avg": out.ww_avg,
        },
        {
            "nlev": np.int32(nlev), "ncells": np.int32(ncells),
            "dts": np.float32(dts),
        },
    )
    return out


__all__ = [
    "MetalAcousticForcing",
    "MetalAcousticState",
    "MetalVerticalImplicitCoefficients",
    "advance_acoustic_step_metal_v841",
    "compute_vertical_implicit_coefficients_metal_v841",
    "convert_w_tendency_to_omega_metal",
]
