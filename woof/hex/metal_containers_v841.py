"""Resident Metal containers for the Hex v8.4.1 dycore.

Field for field, these are the CUDA containers of
``woof.hex.cuda_backend.containers`` and ``woof.hex.cuda_v841`` with
``MetalArray`` storage.  Host preparation reuses the CUDA module's own
conversions, so both backends upload the same values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .cuda_backend.containers import _as_float, _as_index, _mesh_value
from .dynamics_v841 import precomputed_mesh_inverse_v841
from .errors import ConfigurationRefusal

_F32 = np.dtype(np.float32)
_INDEX_NAMES = {
    "cells_on_edge": "cellsOnEdge",
    "edges_on_cell": "edgesOnCell",
    "n_edges_on_cell": "nEdgesOnCell",
    "cells_on_cell": "cellsOnCell",
    "vertices_on_edge": "verticesOnEdge",
    "edges_on_edge": "edgesOnEdge",
    "n_edges_on_edge": "nEdgesOnEdge",
    "vertices_on_cell": "verticesOnCell",
    "edges_on_vertex": "edgesOnVertex",
    "cells_on_vertex": "cellsOnVertex",
}
_FLOAT_NAMES = {
    "weights_on_edge": "weightsOnEdge",
    "dc_edge": "dcEdge",
    "dv_edge": "dvEdge",
    "area_cell": "areaCell",
    "area_triangle": "areaTriangle",
    "kite_areas_on_vertex": "kiteAreasOnVertex",
    "lat_cell": "latCell",
    "lon_cell": "lonCell",
    "lat_edge": "latEdge",
    "lon_edge": "lonEdge",
    "angle_edge": "angleEdge",
    "mesh_density": "meshDensity",
    "defc_a": "defc_a",
    "defc_b": "defc_b",
    "f_vertex": "fVertex",
    "f_edge": "fEdge",
}
_VERTICAL_NAMES = (
    "zw", "dzw", "rdzw", "zu", "dzu", "rdzu", "rdzwp", "rdzwm", "fzp", "fzm",
    "ah", "hx", "zgrid", "zz", "zxu", "dss",
)
_STATE_NAMES = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
_SAVED_NAMES = (
    "theta_m", "exner", "density_perturbation", "rho_theta_perturbation",
    "pressure_perturbation", "normal_velocity", "vertical_velocity",
)


def _module(context: Any) -> Any:
    from woof.metal.array import MetalArrayModule

    return MetalArrayModule(context)


def _require_float32(name: str, value: Any) -> np.ndarray:
    host = np.asarray(value)
    if host.dtype != np.dtype(np.float32):
        raise ConfigurationRefusal(
            name, host.dtype.str, "the Hex v8.4.1 Metal dycore is float32",
            "float32 host arrays")
    return np.ascontiguousarray(host)


@dataclass(slots=True)
class MetalMesh:
    cells_on_edge: Any
    edges_on_cell: Any
    n_edges_on_cell: Any
    cells_on_cell: Any
    vertices_on_edge: Any
    edges_on_edge: Any
    n_edges_on_edge: Any
    vertices_on_cell: Any
    edges_on_vertex: Any
    cells_on_vertex: Any
    edge_sign_on_cell: Any
    weights_on_edge: Any
    dc_edge: Any
    dv_edge: Any
    area_cell: Any
    area_triangle: Any
    kite_areas_on_vertex: Any
    lat_cell: Any
    lon_cell: Any
    lat_edge: Any
    lon_edge: Any
    angle_edge: Any
    mesh_density: Any
    defc_a: Any
    defc_b: Any
    f_vertex: Any
    f_edge: Any
    spec_zone_mask_edge: Any
    n_cells: int
    n_edges: int
    n_vertices: int
    max_edges: int
    max_edges2: int
    vertex_degree: int
    nominal_min_dc: float

    @classmethod
    def from_host(cls, context: Any, mesh: object) -> "MetalMesh":
        """Upload the mesh as ``DeviceMesh.from_host`` does, with int32 indices."""

        dtype = np.dtype(np.float32)
        index_dtype = np.dtype(np.int32)
        dims = getattr(mesh, "dimensions", {})
        n_cells = int(dims.get("nCells", np.asarray(_mesh_value(mesh, "latCell")).size))
        n_edges = int(dims.get("nEdges", np.asarray(_mesh_value(mesh, "dcEdge")).size))
        n_vertices = int(dims.get(
            "nVertices", np.asarray(_mesh_value(mesh, "areaTriangle")).size))
        max_edges = int(dims.get(
            "maxEdges", np.asarray(_mesh_value(mesh, "edgesOnCell")).shape[1]))
        max_edges2 = int(dims.get(
            "maxEdges2", np.asarray(_mesh_value(mesh, "edgesOnEdge")).shape[1]))
        vertex_degree = int(dims.get(
            "vertexDegree", np.asarray(_mesh_value(mesh, "cellsOnVertex")).shape[1]))
        index = {
            target: _as_index(source, _mesh_value(mesh, source), index_dtype)
            for target, source in _INDEX_NAMES.items()
        }
        signs = np.zeros((n_cells, max_edges), dtype=dtype)
        eoc, coe, counts = (
            index["edges_on_cell"], index["cells_on_edge"], index["n_edges_on_cell"])
        for cell in range(n_cells):
            for slot in range(int(counts[cell])):
                edge = int(eoc[cell, slot])
                signs[cell, slot] = 1.0 if int(coe[edge, 0]) == cell else -1.0
        floats = {
            target: _as_float(source, _mesh_value(mesh, source), dtype)
            for target, source in _FLOAT_NAMES.items()
        }
        zone = _mesh_value(mesh, "spec_zone_mask_edge")
        floats["spec_zone_mask_edge"] = _as_float(
            "spec_zone_mask_edge",
            np.zeros(n_edges, dtype=dtype) if zone is None else zone, dtype)
        nominal = _mesh_value(mesh, "nominalMinDc")
        if nominal is None:
            nominal = _mesh_value(mesh, "nominal_min_dc")
        if nominal is None or not np.isfinite(float(np.asarray(nominal))):
            raise ConfigurationRefusal(
                "nominalMinDc", nominal,
                "the scale-aware filters need the mesh's nominal spacing",
                "a mesh carrying nominalMinDc")
        metal = _module(context)
        return cls(
            **{name: metal.asarray(value) for name, value in index.items()},
            edge_sign_on_cell=metal.asarray(signs),
            **{name: metal.asarray(value) for name, value in floats.items()},
            n_cells=n_cells, n_edges=n_edges, n_vertices=n_vertices,
            max_edges=max_edges, max_edges2=max_edges2,
            vertex_degree=vertex_degree,
            nominal_min_dc=float(np.asarray(nominal)),
        )


@dataclass(slots=True)
class MetalVerticalGrid:
    zw: Any
    dzw: Any
    rdzw: Any
    zu: Any
    dzu: Any
    rdzu: Any
    rdzwp: Any
    rdzwm: Any
    fzp: Any
    fzm: Any
    ah: Any
    hx: Any
    zgrid: Any
    zz: Any
    zxu: Any
    dss: Any
    cf1: float
    cf2: float
    cf3: float
    first_height_level: int
    n_vert_levels: int

    @classmethod
    def from_host(cls, context: Any, vertical: object) -> "MetalVerticalGrid":
        metal = _module(context)
        return cls(
            **{
                name: metal.asarray(np.ascontiguousarray(
                    np.asarray(getattr(vertical, name), dtype=np.float32)))
                for name in _VERTICAL_NAMES
            },
            cf1=float(getattr(vertical, "cf1")),
            cf2=float(getattr(vertical, "cf2")),
            cf3=float(getattr(vertical, "cf3")),
            first_height_level=int(getattr(vertical, "first_height_level")),
            n_vert_levels=int(np.asarray(getattr(vertical, "zz")).shape[0]),
        )


@dataclass(slots=True)
class MetalReferenceState:
    rho_base: Any
    rho_theta_base: Any
    pressure_base: Any
    exner_base: Any

    @classmethod
    def from_host(cls, context: Any, reference: object) -> "MetalReferenceState":
        metal = _module(context)
        return cls(**{
            name: metal.asarray(_as_float(name, getattr(reference, name), np.float32))
            for name in ("rho_base", "rho_theta_base", "pressure_base", "exner_base")
        })


@dataclass(slots=True)
class MetalPrognosticState:
    rho: Any
    rho_theta: Any
    rho_u: Any
    rho_w: Any
    scalars: Any
    time_seconds: float

    @classmethod
    def from_host(cls, context: Any, state: object) -> "MetalPrognosticState":
        metal = _module(context)
        return cls(
            **{name: metal.asarray(_require_float32(name, getattr(state, name)))
               for name in _STATE_NAMES},
            time_seconds=float(getattr(state, "time_seconds", 0.0)),
        )

    def to_host(self) -> Any:
        from .state import PrognosticState

        return PrognosticState(
            **{name: getattr(self, name).to_numpy() for name in _STATE_NAMES},
            time_seconds=self.time_seconds,
        )


@dataclass(slots=True)
class MetalSavedDiagnostics:
    theta_m: Any
    exner: Any
    density_perturbation: Any
    rho_theta_perturbation: Any
    pressure_perturbation: Any
    normal_velocity: Any
    vertical_velocity: Any

    @classmethod
    def from_host(cls, context: Any, saved: object) -> "MetalSavedDiagnostics":
        metal = _module(context)
        return cls(**{
            name: metal.asarray(_require_float32(name, getattr(saved, name)))
            for name in _SAVED_NAMES
        })

    def to_host(self) -> Any:
        from .driver import DrySavedDiagnostics

        return DrySavedDiagnostics(
            **{name: getattr(self, name).to_numpy() for name in _SAVED_NAMES})


@dataclass(slots=True)
class MetalTerrainMetrics:
    zb_cell: Any
    zb3_cell: Any

    @classmethod
    def from_host(cls, context: Any, terrain: object) -> "MetalTerrainMetrics":
        metal = _module(context)
        return cls(
            metal.asarray(_as_float("zb_cell", getattr(terrain, "zb_cell"), np.float32)),
            metal.asarray(_as_float("zb3_cell", getattr(terrain, "zb3_cell"), np.float32)),
        )


@dataclass(slots=True)
class MetalV841Context:
    """The resident v8.4.1 arrays of ``CudaV841Context``."""

    inv_area_cell: Any
    inv_area_triangle: Any
    inv_dc_edge: Any
    inv_dv_edge: Any
    etp: Any
    etm: Any
    ewp: Any
    ewm: Any
    u_init: Any
    v_init: Any

    @classmethod
    def from_host(
        cls, context: Any, mesh: object, offcentering: Any, reference_wind: Any,
        *, n_vert_levels: int,
    ) -> "MetalV841Context":
        reference_wind.validate(n_vert_levels, _F32)
        metal = _module(context)
        host = {
            "inv_area_cell": precomputed_mesh_inverse_v841(mesh, "areaCell", _F32),
            "inv_area_triangle": precomputed_mesh_inverse_v841(
                mesh, "areaTriangle", _F32),
            "inv_dc_edge": precomputed_mesh_inverse_v841(mesh, "dcEdge", _F32),
            "inv_dv_edge": precomputed_mesh_inverse_v841(mesh, "dvEdge", _F32),
            **{name: _require_float32(name, getattr(offcentering, name))
               for name in ("etp", "etm", "ewp", "ewm")},
            "u_init": _require_float32("u_init", reference_wind.u_init),
            "v_init": _require_float32("v_init", reference_wind.v_init),
        }
        return cls(**{name: metal.asarray(np.ascontiguousarray(value))
                      for name, value in host.items()})


@dataclass(slots=True)
class MetalAtmosphere:
    mesh: MetalMesh
    state: MetalPrognosticState
    vertical: MetalVerticalGrid
    reference: MetalReferenceState
    saved: MetalSavedDiagnostics
    terrain: MetalTerrainMetrics | None

    @classmethod
    def from_host(
        cls, context: Any, mesh: object, state: object, vertical: object,
        reference: object, saved_diagnostics: object,
        terrain_metrics: object | None = None,
    ) -> "MetalAtmosphere":
        return cls(
            mesh=MetalMesh.from_host(context, mesh),
            state=MetalPrognosticState.from_host(context, state),
            vertical=MetalVerticalGrid.from_host(context, vertical),
            reference=MetalReferenceState.from_host(context, reference),
            saved=MetalSavedDiagnostics.from_host(context, saved_diagnostics),
            terrain=(None if terrain_metrics is None
                     else MetalTerrainMetrics.from_host(context, terrain_metrics)),
        )


__all__ = [
    "MetalAtmosphere", "MetalMesh", "MetalPrognosticState", "MetalReferenceState",
    "MetalSavedDiagnostics", "MetalTerrainMetrics", "MetalV841Context",
    "MetalVerticalGrid",
]
