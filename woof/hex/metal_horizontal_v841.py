"""Metal mirror of the v8.4.1 CUDA horizontal operators.

``MetalHorizontalV841`` follows ``CudaHorizontalV841`` (and the
``CudaHorizontal`` methods the v8.4.1 step calls) method for method.  Each
CUDA kernel has one ``hexv841_<kernel>`` shader.  Fields are resident
``MetalArray`` objects in level-major order, ``index = level * nEntity +
entity``; mesh-static arrays arrive resident in ``MetalMesh`` and
``MetalV841Context`` (``metal_containers_v841``).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from woof.metal.array import MetalArray, MetalArrayModule

from .errors import ConfigurationRefusal
from .metal_v841_launch import launch


@dataclass(frozen=True, slots=True)
class MetalRecoveredEdges:
    normal_velocity: Any
    rho_edge: Any


@dataclass(frozen=True, slots=True)
class MetalSolveDiagnostics:
    h_edge: Any
    normal_velocity: Any
    tangential_velocity: Any
    vorticity: Any
    divergence: Any
    kinetic_energy: Any
    pv_edge: Any
    pv_vertex: Any
    pv_cell: Any
    grad_pv_normal: Any
    grad_pv_tangential: Any


@dataclass(frozen=True, slots=True)
class MetalDryMixingTendencies:
    kdiff: Any
    h_mom_eddy_visc4: np.float32
    h_theta_eddy_visc4: np.float32
    tend_u_euler: Any
    tend_w_euler: Any
    tend_theta_euler: Any
    delsq_u: Any
    delsq_divergence: Any
    delsq_vorticity: Any
    delsq_w: Any
    delsq_theta: Any


class MetalHorizontalV841:
    """v8.4.1 stored-inverse operators on resident Metal arrays."""

    def __init__(
        self,
        mesh: Any,  # MetalMesh
        n_vert_levels: int,
        context: Any,  # MetalV841Context
    ) -> None:
        if n_vert_levels < 1:
            raise ValueError("nlev must be positive")
        self.mesh = mesh
        self.nlev = int(n_vert_levels)
        self.ncells = int(mesh.n_cells)
        self.nedges = int(mesh.n_edges)
        self.nvertices = int(mesh.n_vertices)
        self.max_edges = int(mesh.max_edges)
        self.max_edges2 = int(mesh.max_edges2)
        self.vertex_degree = int(mesh.vertex_degree)
        if self.vertex_degree != 3:
            raise ConfigurationRefusal(
                "vertexDegree", self.vertex_degree,
                "v8.4.1 Metal requires MPAS vertexDegree=3", "vertexDegree=3")
        self.v841 = context
        self.metal = mesh.cells_on_edge.buffer.context
        self.xp = MetalArrayModule(self.metal)
        self._deformation_v841: dict[str, Any] | None = None

    # -- helpers -----------------------------------------------------------
    def _launch(self, kernel: str, total: int, arrays: dict, params: dict) -> None:
        launch(self.metal, f"hexv841_{kernel}", int(total), arrays, params)

    def _field(self, name: str, value: Any, shape: tuple[int, ...]) -> Any:
        if not isinstance(value, MetalArray):
            raise TypeError(f"{name} must already be a resident MetalArray")
        if value.dtype != np.dtype(np.float32):
            raise TypeError(f"{name} must have dtype float32")
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} shape {value.shape} != {shape}")
        return value

    def _cell_field(self, name: str, value: Any) -> Any:
        return self._field(name, value, (self.nlev, self.ncells))

    def _edge_field(self, name: str, value: Any) -> Any:
        return self._field(name, value, (self.nlev, self.nedges))

    def _interface_field(self, name: str, value: Any) -> Any:
        return self._field(name, value, (self.nlev + 1, self.ncells))

    def _edge_state(self, rho: Any, rho_u: Any, normal_velocity: Any | None):
        xp = self.xp
        rho = self._cell_field("rho", rho)
        rho_u = self._edge_field("rho_u", rho_u)
        rho_edge = xp.empty(rho_u.shape, np.float32)
        ke_edge = xp.empty(rho_u.shape, np.float32)
        common = {
            "dc_edge": self.mesh.dc_edge,
            "dv_edge": self.mesh.dv_edge,
            "cells_on_edge": self.mesh.cells_on_edge,
        }
        params = {
            "nlev": np.int32(self.nlev),
            "ncells": np.int32(self.ncells),
            "nedges": np.int32(self.nedges),
        }
        if normal_velocity is None:
            normal = xp.empty(rho_u.shape, np.float32)
            self._launch(
                "recover_edge_f32", self.nedges,
                {"rho": rho, "rho_u": rho_u, **common,
                 "normal_velocity": normal, "rho_edge": rho_edge,
                 "ke_edge": ke_edge},
                params)
        else:
            normal = self._edge_field("normal_velocity", normal_velocity)
            self._launch(
                "edge_fields_from_saved_u_f32", self.nedges,
                {"rho": rho, "normal_velocity": normal, **common,
                 "rho_edge": rho_edge, "ke_edge": ke_edge},
                params)
        return normal, rho_edge, ke_edge

    # -- CudaHorizontal API --------------------------------------------------
    def recover_edge_fields(self, rho: Any, rho_u: Any) -> MetalRecoveredEdges:
        normal, rho_edge, _ = self._edge_state(rho, rho_u, None)
        return MetalRecoveredEdges(normal_velocity=normal, rho_edge=rho_edge)

    def mass_flux_divergence(self, rho_u: Any) -> Any:
        rho_u = self._edge_field("rho_u", rho_u)
        out = self.xp.empty((self.nlev, self.ncells), np.float32)
        self._launch(
            "mass_flux_divergence_v841_f32", self.nlev * self.ncells,
            {"rho_u": rho_u,
             "edges_on_cell": self.mesh.edges_on_cell,
             "n_edges_on_cell": self.mesh.n_edges_on_cell,
             "cells_on_edge": self.mesh.cells_on_edge,
             "dv_edge": self.mesh.dv_edge,
             "inv_area_cell": self.v841.inv_area_cell,
             "mass_divergence": out},
            {"nlev": np.int32(self.nlev), "ncells": np.int32(self.ncells),
             "nedges": np.int32(self.nedges),
             "max_edges": np.int32(self.max_edges)})
        return out

    def density_tendency(
        self,
        rho_u: Any,
        rho_w: Any,
        rdzw: Any,
        physics_tendency: Any | None = None,
    ) -> Any:
        mass_divergence = self.mass_flux_divergence(rho_u)
        rho_w = self._interface_field("rho_w", rho_w)
        rdzw = self._field("rdzw", rdzw, (self.nlev,))
        if physics_tendency is None:
            physics = mass_divergence
            has_physics = np.int32(0)
        else:
            physics = self._cell_field("physics_tendency", physics_tendency)
            has_physics = np.int32(1)
        out = self.xp.empty(mass_divergence.shape, np.float32)
        self._launch(
            "density_tendency_f32", self.ncells,
            {"mass_divergence": mass_divergence, "rho_w": rho_w,
             "rdzw": rdzw, "physics_tendency": physics, "tendency": out},
            {"has_physics": has_physics, "nlev": np.int32(self.nlev),
             "ncells": np.int32(self.ncells)})
        return out

    def solve_diagnostics(
        self,
        rho: Any,
        rho_u: Any,
        *,
        dt: float,
        apvm_upwinding: float = 0.0,
        normal_velocity: Any | None = None,
        cached_tangential_velocity: Any | None = None,
        rk_step: int | None = 3,
    ) -> MetalSolveDiagnostics:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(apvm_upwinding) or apvm_upwinding < 0.0:
            raise ValueError("apvm_upwinding must be finite and non-negative")
        xp = self.xp
        mesh = self.mesh
        normal, rho_edge, ke_edge = self._edge_state(rho, rho_u, normal_velocity)
        reconstruct_v = rk_step is None or rk_step == 3
        if reconstruct_v:
            tangential = xp.empty(normal.shape, np.float32)
            self._launch(
                "tangential_velocity_f32", self.nlev * self.nedges,
                {"normal_velocity": normal,
                 "edges_on_edge": mesh.edges_on_edge,
                 "n_edges_on_edge": mesh.n_edges_on_edge,
                 "weights_on_edge": mesh.weights_on_edge,
                 "tangential_velocity": tangential},
                {"nlev": np.int32(self.nlev), "nedges": np.int32(self.nedges),
                 "max_edges2": np.int32(self.max_edges2)})
        else:
            if cached_tangential_velocity is None:
                raise ValueError(
                    "RK1/RK2 diagnostics require cached tangential velocity")
            tangential = self._edge_field(
                "cached_tangential_velocity", cached_tangential_velocity)

        vorticity = xp.empty((self.nlev, self.nvertices), np.float32)
        ke_vertex = xp.empty((self.nlev, self.nvertices), np.float32)
        pv_vertex = xp.empty((self.nlev, self.nvertices), np.float32)
        self._launch(
            "vertex_diagnostics_v841_f32", self.nlev * self.nvertices,
            {"normal_velocity": normal, "ke_edge": ke_edge,
             "edges_on_vertex": mesh.edges_on_vertex,
             "vertices_on_edge": mesh.vertices_on_edge,
             "dc_edge": mesh.dc_edge,
             "inv_area_triangle": self.v841.inv_area_triangle,
             "f_vertex": mesh.f_vertex,
             "vorticity": vorticity, "ke_vertex": ke_vertex,
             "pv_vertex": pv_vertex},
            {"nlev": np.int32(self.nlev), "nedges": np.int32(self.nedges),
             "nvertices": np.int32(self.nvertices),
             "vertex_degree": np.int32(self.vertex_degree)})
        divergence = xp.empty((self.nlev, self.ncells), np.float32)
        kinetic_energy = xp.empty((self.nlev, self.ncells), np.float32)
        self._launch(
            "cell_diagnostics_v841_f32", self.nlev * self.ncells,
            {"normal_velocity": normal, "ke_edge": ke_edge,
             "ke_vertex": ke_vertex,
             "edges_on_cell": mesh.edges_on_cell,
             "n_edges_on_cell": mesh.n_edges_on_cell,
             "cells_on_edge": mesh.cells_on_edge,
             "vertices_on_cell": mesh.vertices_on_cell,
             "cells_on_vertex": mesh.cells_on_vertex,
             "dv_edge": mesh.dv_edge,
             "inv_area_cell": self.v841.inv_area_cell,
             "kite_areas": mesh.kite_areas_on_vertex,
             "divergence": divergence, "kinetic_energy": kinetic_energy},
            {"nlev": np.int32(self.nlev), "ncells": np.int32(self.ncells),
             "nedges": np.int32(self.nedges),
             "nvertices": np.int32(self.nvertices),
             "max_edges": np.int32(self.max_edges),
             "vertex_degree": np.int32(self.vertex_degree)})
        pv_edge = xp.empty(normal.shape, np.float32)
        self._launch(
            "pv_edge_base_f32", self.nedges,
            {"pv_vertex": pv_vertex, "vertices_on_edge": mesh.vertices_on_edge,
             "pv_edge": pv_edge},
            {"nlev": np.int32(self.nlev), "nedges": np.int32(self.nedges),
             "nvertices": np.int32(self.nvertices)})
        pv_cell = xp.zeros((self.nlev, self.ncells), np.float32)
        grad_normal = xp.zeros(normal.shape, np.float32)
        grad_tangential = xp.zeros(normal.shape, np.float32)
        if apvm_upwinding > 0.0:
            self._launch(
                "pv_cell_v841_f32", self.nlev * self.ncells,
                {"pv_vertex": pv_vertex,
                 "vertices_on_cell": mesh.vertices_on_cell,
                 "n_edges_on_cell": mesh.n_edges_on_cell,
                 "cells_on_vertex": mesh.cells_on_vertex,
                 "kite_areas": mesh.kite_areas_on_vertex,
                 "inv_area_cell": self.v841.inv_area_cell,
                 "pv_cell": pv_cell},
                {"nlev": np.int32(self.nlev), "ncells": np.int32(self.ncells),
                 "nvertices": np.int32(self.nvertices),
                 "max_edges": np.int32(self.max_edges),
                 "vertex_degree": np.int32(self.vertex_degree)})
            scale = np.float32(apvm_upwinding) * np.float32(dt)
            self._launch(
                "pv_apvm_v841_f32", self.nlev * self.nedges,
                {"normal_velocity": normal, "tangential_velocity": tangential,
                 "pv_vertex": pv_vertex, "pv_cell": pv_cell,
                 "vertices_on_edge": mesh.vertices_on_edge,
                 "cells_on_edge": mesh.cells_on_edge,
                 "inv_dc_edge": self.v841.inv_dc_edge,
                 "inv_dv_edge": self.v841.inv_dv_edge,
                 "pv_edge": pv_edge, "grad_pv_normal": grad_normal,
                 "grad_pv_tangential": grad_tangential},
                {"scale": scale, "nlev": np.int32(self.nlev),
                 "ncells": np.int32(self.ncells),
                 "nedges": np.int32(self.nedges),
                 "nvertices": np.int32(self.nvertices)})
        return MetalSolveDiagnostics(
            h_edge=rho_edge,
            normal_velocity=normal,
            tangential_velocity=tangential,
            vorticity=vorticity,
            divergence=divergence,
            kinetic_energy=kinetic_energy,
            pv_edge=pv_edge,
            pv_vertex=pv_vertex,
            pv_cell=pv_cell,
            grad_pv_normal=grad_normal,
            grad_pv_tangential=grad_tangential,
        )

    def attach_deformation_weights_v841(self, weights: Any) -> dict[str, Any]:
        """Upload host float32 v8.4.1 deformation weights (byte-identical)."""

        shape = (self.ncells, self.max_edges)
        arrays = {}
        receipts: dict[str, Any] = {}
        for name in ("coef_c2", "coef_s2", "coef_cs"):
            host = np.ascontiguousarray(
                np.asarray(getattr(weights, name), dtype=np.float32))
            if host.shape != shape:
                raise ValueError(f"{name} shape {host.shape} != {shape}")
            if not np.all(np.isfinite(host)):
                raise ValueError(f"{name} contains non-finite values")
            arrays[name] = self.xp.asarray(host)
            receipts[name] = {
                "shape": list(shape),
                "dtype": "float32",
                "nonzero": int(np.count_nonzero(host)),
                "sha256": hashlib.sha256(host.tobytes(order="C")).hexdigest(),
            }
        self._deformation_v841 = arrays
        return receipts

    def compute_dry_mixing_tendencies_v841(
        self,
        normal_velocity: Any,
        tangential_velocity: Any,
        vertical_velocity: Any,
        theta_m: Any,
        rho_edge: Any,
        divergence: Any,
        vorticity: Any,
        *,
        dt: float,
        config: Any | None = None,
    ) -> MetalDryMixingTendencies:
        """v8.4.1 2-D Smagorinsky mixing (``smagorinsky_2d``)."""

        from .mixing_v841 import V841MixingConfig

        cfg = V841MixingConfig() if config is None else config
        if not isinstance(cfg, V841MixingConfig):
            raise TypeError("config must be a mixing_v841.V841MixingConfig")
        cfg.validate()
        timestep = float(dt)
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("dt must be finite and positive")
        deformation = self._deformation_v841
        if deformation is None:
            raise RuntimeError(
                "v8.4.1 deformation weights are not attached; call "
                "attach_deformation_weights_v841 first")

        mesh = self.mesh
        xp = self.xp
        normal = self._edge_field("normal_velocity", normal_velocity)
        tangential = self._edge_field("tangential_velocity", tangential_velocity)
        vertical = self._interface_field("vertical_velocity", vertical_velocity)
        theta = self._cell_field("theta_m", theta_m)
        edge_rho = self._edge_field("rho_edge", rho_edge)
        div = self._cell_field("divergence", divergence)
        vort = self._field("vorticity", vorticity, (self.nlev, self.nvertices))

        configured_length = float(cfg.config_len_disp)
        length = (float(mesh.nominal_min_dc)
                  if configured_length == 0.0 else configured_length)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError(
                "config_len_disp or the mesh nominal_min_dc must be positive")
        length32 = np.float32(length)
        cs = np.float32(cfg.config_smagorinsky_coef)
        strain_scale = np.float32((cs * length32) * (cs * length32))
        inv_dt = np.float32(np.float32(1.0) / np.float32(timestep))
        ceiling = np.float32(
            (np.float32(0.01) * (length32 * length32)) * inv_dt)
        visc4 = np.float32(cfg.config_visc4_2dsmag)
        h4 = np.float32(visc4 * ((length32 * length32) * length32))

        nlev = np.int32(self.nlev)
        ncells = np.int32(self.ncells)
        nedges = np.int32(self.nedges)
        nvertices = np.int32(self.nvertices)
        max_edges = np.int32(self.max_edges)

        del2 = xp.empty((self.nedges,), np.float32)
        del4 = xp.empty((self.nedges,), np.float32)
        self._launch(
            "mixing_scaling_f32", self.nedges,
            {"cells_on_edge": mesh.cells_on_edge,
             "mesh_density": mesh.mesh_density, "del2": del2, "del4": del4},
            {"nedges": nedges,
             "scale_with_mesh": np.int32(bool(cfg.config_h_ScaleWithMesh))})

        kdiff = xp.empty((self.nlev, self.ncells), np.float32)
        self._launch(
            "smagorinsky_v841_f32", self.nlev * self.ncells,
            {"normal_velocity": normal, "tangential_velocity": tangential,
             "edges_on_cell": mesh.edges_on_cell,
             "n_edges_on_cell": mesh.n_edges_on_cell,
             "coef_c2": deformation["coef_c2"],
             "coef_s2": deformation["coef_s2"],
             "coef_cs": deformation["coef_cs"], "kdiff": kdiff},
            {"strain_scale": strain_scale, "ceiling": ceiling, "nlev": nlev,
             "ncells": ncells, "nedges": nedges, "max_edges": max_edges})

        delsq_u = xp.empty(normal.shape, np.float32)
        tend_u = xp.empty(normal.shape, np.float32)
        self._launch(
            "momentum_filter_lap2_f32", self.nedges,
            {"rho_edge": edge_rho, "divergence": div, "vorticity": vort,
             "kdiff": kdiff, "del2": del2,
             "cells_on_edge": mesh.cells_on_edge,
             "vertices_on_edge": mesh.vertices_on_edge,
             "dc_edge": mesh.dc_edge, "dv_edge": mesh.dv_edge,
             "delsq_u": delsq_u, "tendency": tend_u},
            {"nlev": nlev, "ncells": ncells, "nedges": nedges,
             "nvertices": nvertices})
        delsq_div = xp.zeros((self.nlev, self.ncells), np.float32)
        delsq_vort = xp.zeros((self.nlev, self.nvertices), np.float32)
        if h4 > np.float32(0.0):
            self._launch(
                "laplacian_divergence_f32", self.ncells * self.nlev,
                {"delsq_u": delsq_u, "edges_on_cell": mesh.edges_on_cell,
                 "n_edges_on_cell": mesh.n_edges_on_cell,
                 "cells_on_edge": mesh.cells_on_edge,
                 "dv_edge": mesh.dv_edge, "area_cell": mesh.area_cell,
                 "delsq_divergence": delsq_div},
                {"nlev": nlev, "ncells": ncells, "nedges": nedges,
                 "max_edges": max_edges})
            self._launch(
                "laplacian_vorticity_f32", self.nvertices,
                {"delsq_u": delsq_u, "edges_on_vertex": mesh.edges_on_vertex,
                 "vertices_on_edge": mesh.vertices_on_edge,
                 "dc_edge": mesh.dc_edge, "area_triangle": mesh.area_triangle,
                 "delsq_vorticity": delsq_vort},
                {"nlev": nlev, "nedges": nedges, "nvertices": nvertices,
                 "vertex_degree": np.int32(self.vertex_degree)})
            self._launch(
                "momentum_filter_lap4_f32", self.nedges,
                {"rho_edge": edge_rho, "delsq_divergence": delsq_div,
                 "delsq_vorticity": delsq_vort, "del4": del4,
                 "cells_on_edge": mesh.cells_on_edge,
                 "vertices_on_edge": mesh.vertices_on_edge,
                 "dc_edge": mesh.dc_edge, "dv_edge": mesh.dv_edge,
                 "tendency": tend_u},
                {"h4": h4,
                 "div_factor": np.float32(cfg.config_del4u_div_factor),
                 "nlev": nlev, "ncells": ncells, "nedges": nedges,
                 "nvertices": nvertices})

        delsq_theta = xp.empty(theta.shape, np.float32)
        tend_theta = xp.empty(theta.shape, np.float32)
        self._launch(
            "theta_filter_lap2_f32", self.ncells * self.nlev,
            {"theta_m": theta, "rho_edge": edge_rho, "kdiff": kdiff,
             "del2": del2, "edges_on_cell": mesh.edges_on_cell,
             "n_edges_on_cell": mesh.n_edges_on_cell,
             "cells_on_edge": mesh.cells_on_edge, "dc_edge": mesh.dc_edge,
             "dv_edge": mesh.dv_edge, "area_cell": mesh.area_cell,
             "delsq_theta": delsq_theta, "tendency": tend_theta},
            {"nlev": nlev, "ncells": ncells, "nedges": nedges,
             "max_edges": max_edges})
        if h4 > np.float32(0.0):
            self._launch(
                "theta_filter_lap4_f32", self.ncells * self.nlev,
                {"delsq_theta": delsq_theta, "del4": del4,
                 "edges_on_cell": mesh.edges_on_cell,
                 "n_edges_on_cell": mesh.n_edges_on_cell,
                 "cells_on_edge": mesh.cells_on_edge,
                 "dc_edge": mesh.dc_edge, "dv_edge": mesh.dv_edge,
                 "area_cell": mesh.area_cell, "tendency": tend_theta},
                {"h4": h4, "nlev": nlev, "ncells": ncells,
                 "max_edges": max_edges})

        delsq_w = xp.empty((self.nlev, self.ncells), np.float32)
        tend_w = xp.zeros((self.nlev + 1, self.ncells), np.float32)
        self._launch(
            "w_filter_lap2_f32", self.ncells * self.nlev,
            {"vertical_velocity": vertical, "rho_edge": edge_rho,
             "kdiff": kdiff, "del2": del2,
             "edges_on_cell": mesh.edges_on_cell,
             "n_edges_on_cell": mesh.n_edges_on_cell,
             "cells_on_edge": mesh.cells_on_edge, "dc_edge": mesh.dc_edge,
             "dv_edge": mesh.dv_edge, "area_cell": mesh.area_cell,
             "delsq_w": delsq_w, "tendency": tend_w},
            {"nlev": nlev, "ncells": ncells, "nedges": nedges,
             "max_edges": max_edges})
        if h4 > np.float32(0.0):
            self._launch(
                "w_filter_lap4_f32", (self.nlev - 1) * self.ncells,
                {"delsq_w": delsq_w, "del4": del4,
                 "edges_on_cell": mesh.edges_on_cell,
                 "n_edges_on_cell": mesh.n_edges_on_cell,
                 "cells_on_edge": mesh.cells_on_edge,
                 "dc_edge": mesh.dc_edge, "dv_edge": mesh.dv_edge,
                 "area_cell": mesh.area_cell, "tendency": tend_w},
                {"h4": h4, "nlev": nlev, "ncells": ncells,
                 "max_edges": max_edges})

        return MetalDryMixingTendencies(
            kdiff=kdiff,
            h_mom_eddy_visc4=h4,
            h_theta_eddy_visc4=h4,
            tend_u_euler=tend_u,
            tend_w_euler=tend_w,
            tend_theta_euler=tend_theta,
            delsq_u=delsq_u,
            delsq_divergence=delsq_div,
            delsq_vorticity=delsq_vort,
            delsq_w=delsq_w,
            delsq_theta=delsq_theta,
        )

    def pressure_gradient_euler_tendency(
        self,
        pressure_perturbation: Any,
        dpdz: Any,
        cqu: Any,
        zz: Any,
        zxu: Any,
    ) -> Any:
        pressure = self._cell_field("pressure_perturbation", pressure_perturbation)
        dpdz = self._cell_field("dpdz", dpdz)
        cqu = self._edge_field("cqu", cqu)
        zz = self._cell_field("zz", zz)
        zxu = self._edge_field("zxu", zxu)
        out = self.xp.empty(cqu.shape, np.float32)
        self._launch(
            "pressure_gradient_v841_f32", self.nedges,
            {"pressure_p": pressure, "dpdz": dpdz, "cqu": cqu, "zz": zz,
             "zxu": zxu, "cells_on_edge": self.mesh.cells_on_edge,
             "inv_dc_edge": self.v841.inv_dc_edge, "tendency": out},
            {"nlev": np.int32(self.nlev), "ncells": np.int32(self.ncells),
             "nedges": np.int32(self.nedges)})
        return out

    def capture_rtheta_pp_old(self, rtheta_pp: Any, *, small_step: int) -> Any:
        """Capture the resident pre-acoustic damping reference."""

        current = self._cell_field("rtheta_pp", rtheta_pp)
        step = int(small_step)
        if step < 1:
            raise ValueError("small_step is one-based and must be positive")
        if step == 1:
            return self.xp.zeros(current.shape, np.float32)
        return current.copy()

    def divergence_damping_3d(
        self,
        rho_u_perturbation: Any,
        theta_m: Any,
        rtheta_pp: Any,
        rtheta_pp_old: Any,
        *,
        dts: float,
        config_smdiv: float = 0.1,
        config_len_disp: float = 0.0,
        spec_zone_mask_edge: Any | None = None,
        n_cells_solve: int | None = None,
        in_place: bool = False,
    ) -> Any:
        """Apply the post-acoustic 3-D divergence damping update."""

        timestep = float(dts)
        smdiv = float(config_smdiv)
        configured_length = float(config_len_disp)
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("dts must be finite and positive")
        if not np.isfinite(smdiv) or smdiv < 0.0:
            raise ValueError("config_smdiv must be finite and non-negative")
        if not np.isfinite(configured_length) or configured_length < 0.0:
            raise ValueError("config_len_disp must be finite and non-negative")
        length = (float(self.mesh.nominal_min_dc)
                  if configured_length == 0.0 else configured_length)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError(
                "config_len_disp or the mesh nominal_min_dc must be positive")
        solve_count = self.ncells if n_cells_solve is None else int(n_cells_solve)
        if solve_count < 0 or solve_count > self.ncells:
            raise ValueError("n_cells_solve must lie in [0, nCells]")

        ru = self._edge_field("rho_u_perturbation", rho_u_perturbation)
        theta = self._cell_field("theta_m", theta_m)
        current = self._cell_field("rtheta_pp", rtheta_pp)
        previous = self._cell_field("rtheta_pp_old", rtheta_pp_old)
        if spec_zone_mask_edge is None:
            mask = self.mesh.spec_zone_mask_edge
        else:
            mask = self._field(
                "spec_zone_mask_edge", spec_zone_mask_edge, (self.nedges,))
        if not isinstance(in_place, (bool, np.bool_)):
            raise TypeError("in_place must be boolean")
        out = ru if bool(in_place) else ru.copy()
        coefficient = np.float32(
            np.float32(2.0) * np.float32(smdiv) * np.float32(length)
            / np.float32(timestep))
        self._launch(
            "divergence_damping_f32", self.nedges,
            {"theta_m": theta, "rtheta_pp": current,
             "rtheta_pp_old": previous,
             "cells_on_edge": self.mesh.cells_on_edge,
             "spec_zone_mask_edge": mask, "rho_u_perturbation": out},
            {"coefficient": coefficient, "n_cells_solve": np.int32(solve_count),
             "nlev": np.int32(self.nlev), "ncells": np.int32(self.ncells),
             "nedges": np.int32(self.nedges)})
        return out


__all__ = [
    "MetalDryMixingTendencies",
    "MetalHorizontalV841",
    "MetalRecoveredEdges",
    "MetalSolveDiagnostics",
]
