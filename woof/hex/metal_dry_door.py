"""Public, bounded Metal route for the Hex dry dynamics preview."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


def _tiny_icosahedral_mesh():
    """A 12-cell closed icosahedral mesh for the public tiny Metal route."""
    from .driver import make_synthetic_x1_case

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    points = []
    for a in (-1.0, 1.0):
        for b in (-1.0, 1.0):
            points.extend(((0.0, a, b * phi), (a * phi, 0.0, b), (a, b * phi, 0.0)))
    vertices = np.asarray(points, dtype=np.float64)
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    edge_dot = 1.0 / np.sqrt(5.0)
    faces = np.asarray([
        tri for tri in itertools.combinations(range(len(vertices)), 3)
        if all(abs(float(np.dot(vertices[i], vertices[j])) - edge_dot) < 1e-8
               for i, j in itertools.combinations(tri, 2))
    ], dtype=np.int64)
    n_cells, n_vertices = len(vertices), len(faces)
    edge_cells = sorted({tuple(sorted(pair)) for face in faces
                         for pair in itertools.combinations(face, 2)})
    edge_vertices = [sorted(
        set(np.flatnonzero(np.any(faces == a, axis=1)))
        & set(np.flatnonzero(np.any(faces == b, axis=1)))
    ) for a, b in edge_cells]
    if any(len(pair) != 2 for pair in edge_vertices):
        raise ValueError("tiny icosahedral mesh has an invalid edge")
    cell_edges = [[i for i, edge in enumerate(edge_cells) if cell in edge]
                  for cell in range(n_cells)]
    cell_neighbors = [[edge_cells[edge][1] if edge_cells[edge][0] == cell
                       else edge_cells[edge][0] for edge in row]
                      for cell, row in enumerate(cell_edges)]
    # Primal icosahedron vertices are the cells and its faces the dual
    # vertices: a cell's vertices are its incident faces, a vertex's cells
    # the corners of its face.
    cell_vertices = [np.flatnonzero(np.any(faces == cell, axis=1)).tolist()
                     for cell in range(n_cells)]
    vertex_cells = faces.tolist()
    vertex_edges = [[i for i, pair in enumerate(edge_vertices) if vertex in pair]
                    for vertex in range(n_vertices)]
    edge_neighbors = [[j for j, edge in enumerate(edge_cells)
                       if j != i and bool(set(edge) & set(edge_cells[i]))]
                      for i in range(len(edge_cells))]

    def padded(rows, width):
        return np.asarray([list(row) + [-1] * (width - len(row)) for row in rows],
                          dtype=np.int64)

    max_cell_edges = max(map(len, cell_edges))
    max_edge_neighbors = max(map(len, edge_neighbors))
    max_vertex_cells = max(map(len, vertex_cells))
    max_vertex_edges = max(map(len, vertex_edges))
    cells_on_edge = np.asarray(edge_cells, dtype=np.int64)
    vertices_on_edge = np.asarray(edge_vertices, dtype=np.int64)
    earth_radius = 6.371e6
    area_cell = np.full(n_cells, 4.0 * np.pi * earth_radius**2 / n_cells)
    area_triangle = np.full(n_vertices, 4.0 * np.pi * earth_radius**2 / n_vertices)
    latitude = lambda xyz: np.arcsin(xyz[:, 2])
    longitude = lambda xyz: np.arctan2(xyz[:, 1], xyz[:, 0])
    edge_xyz = vertices[cells_on_edge].sum(axis=1)
    edge_xyz /= np.linalg.norm(edge_xyz, axis=1)[:, None]
    n_edges = len(edge_cells)
    mesh_arrays = {
        "areaCell": area_cell,
        "areaTriangle": area_triangle,
        "dcEdge": np.full(n_edges, 4e6),
        "dvEdge": np.full(n_edges, 4e6),
        "latCell": latitude(vertices),
        "lonCell": longitude(vertices),
        "latEdge": latitude(edge_xyz),
        "lonEdge": longitude(edge_xyz),
        "cellsOnEdge": cells_on_edge,
        "verticesOnEdge": vertices_on_edge,
        "edgesOnCell": padded(cell_edges, max_cell_edges),
        "nEdgesOnCell": np.asarray([len(row) for row in cell_edges]),
        "cellsOnCell": padded(cell_neighbors, max_cell_edges),
        "verticesOnCell": padded(cell_vertices, max_cell_edges),
        "cellsOnVertex": padded(vertex_cells, max_vertex_cells),
        "edgesOnVertex": padded(vertex_edges, max_vertex_edges),
        "edgesOnEdge": padded(edge_neighbors, max_edge_neighbors),
        "nEdgesOnEdge": np.asarray([len(row) for row in edge_neighbors]),
        "weightsOnEdge": np.full((n_edges, max_edge_neighbors),
                                 1.0 / max_edge_neighbors),
        "kiteAreasOnVertex": np.full((n_vertices, max_vertex_cells),
                                     area_triangle[0] / max_vertex_cells),
        "fVertex": np.zeros(n_vertices),
        "fEdge": np.zeros(n_edges),
        "angleEdge": np.zeros(n_edges),
        "meshDensity": np.ones(n_cells),
        "nominalMinDc": np.asarray(4e6, dtype=np.float32),
    }

    class Mesh:
        arrays = mesh_arrays

    return Mesh(), make_synthetic_x1_case(
        Mesh(), n_vert_levels=4, n_scalars=1, wind_speed=0.0,
        perturbation_amplitude=0.0)


def _configuration(mesh, case):
    from dataclasses import fields, replace

    from .config_v841 import V841DryDycoreConfig
    from .driver import TerrainMetrics
    from .dynamics_v841 import V841ReferenceWindProfiles
    from .transport import AdvectionCoefficients

    def float32_record(record):
        return replace(record, **{
            field.name: (np.asarray(getattr(record, field.name), dtype=np.float32)
                         if isinstance(getattr(record, field.name), np.ndarray)
                         else getattr(record, field.name))
            for field in fields(record)
        })

    n_edges = len(mesh.arrays["cellsOnEdge"])
    n_cells = len(mesh.arrays["areaCell"])
    n_levels = 4
    state = float32_record(case.state)
    vertical = float32_record(case.vertical_grid)
    # Level 0 of fzm/fzp is undefined in MPAS and built as NaN; native grid
    # files store 0 there, which the terrain-metric validation expects.
    vertical = replace(vertical, **{
        name: np.nan_to_num(getattr(vertical, name), nan=0.0)
        for name in ("fzm", "fzp")})
    reference = float32_record(case.reference)
    max_cell_edges = mesh.arrays["edgesOnCell"].shape[1]
    flux = np.zeros((n_edges, 10), dtype=np.float32)
    adv_cells = np.zeros((n_edges, 10), dtype=np.int64)
    for edge, (cell0, cell1) in enumerate(mesh.arrays["cellsOnEdge"]):
        flux[edge, :2] = 0.5
        adv_cells[edge, :2] = (cell0, cell1)
        adv_cells[edge, 2:] = cell0
    coefficients = AdvectionCoefficients(
        flux, np.zeros_like(flux), np.full(n_edges, 2, dtype=np.int64),
        adv_cells, 3)
    config = V841DryDycoreConfig(
        config_dt=0.25, config_time_integration_order=3,
        config_number_of_sub_steps=1, config_dynamics_split_steps=1,
        config_scalar_adv_order=3, config_scalar_vadv_order=3,
        config_split_dynamics_transport=True, config_moist_physics=True)
    wind = V841ReferenceWindProfiles(
        np.zeros(n_levels, dtype=np.float32),
        np.zeros(n_levels, dtype=np.float32))
    terrain = TerrainMetrics(
        np.zeros((n_levels + 1, n_cells, max_cell_edges), dtype=np.float32),
        np.zeros((n_levels + 1, n_cells, max_cell_edges), dtype=np.float32))
    return state, vertical, reference, coefficients, config, wind, terrain


def tiny_dry_steps(steps, *, metal_context=None, state=None,
                   saved_diagnostics=None):
    """Advance the 12-cell tiny mesh; NumPy when ``metal_context`` is None.

    Returns ``(mesh, state, saved_diagnostics)`` after a finite-checked step
    sequence through the public dry-dynamics driver.
    """
    mesh, case = _tiny_icosahedral_mesh()
    initial, vertical, reference, coefficients, config, wind, terrain = (
        _configuration(mesh, case))
    if state is None:
        state = initial
    from .driver import DryDycoreDriver

    driver = DryDycoreDriver(
        mesh, vertical, reference, config,
        advection_coefficients=coefficients, index_qv=0,
        reference_wind_profiles=wind, terrain_metrics=terrain,
        metal_context=metal_context)
    for _ in range(steps):
        result = driver.step(state, saved_diagnostics=saved_diagnostics)
        state, saved_diagnostics = result.state, result.saved_diagnostics
    state.validate(n_cells=12, n_edges=30, n_vert_levels=4)
    return mesh, state, saved_diagnostics


def run_metal_dry(arguments: Any) -> int:
    """Run tiny public Hex steps and optionally write a checkpoint."""
    if arguments.steps < 1 or arguments.steps > 4:
        raise ValueError("--steps must be from 1 through 4 for the tiny Hex route")
    state = saved_diagnostics = None
    if arguments.restart is not None:
        from .checkpoint import read_checkpoint

        state, saved_diagnostics = read_checkpoint(arguments.restart)

    from woof.metal.runtime import MetalContext
    history_path = None
    if arguments.history is not None:
        from datetime import datetime, timedelta, timezone

        from .output import write_dry_state_history

        initial_time = datetime(2026, 10, 8, tzinfo=timezone.utc)

    start = float(state.time_seconds) if state is not None else 0.0
    with MetalContext() as context:
        mesh, state, saved_diagnostics = tiny_dry_steps(
            arguments.steps, metal_context=context, state=state,
            saved_diagnostics=saved_diagnostics)
        if arguments.checkpoint is not None:
            from .checkpoint import write_checkpoint

            write_checkpoint(arguments.checkpoint, state,
                             saved_diagnostics=saved_diagnostics)
        if arguments.history is not None:
            valid_time = initial_time + timedelta(seconds=state.time_seconds)
            history_path = write_dry_state_history(
                arguments.history, mesh, state, valid_time,
                initial_time=initial_time)

    print(json.dumps({
        "backend": "metal",
        "steps": arguments.steps,
        "start_time_seconds": start,
        "end_time_seconds": float(state.time_seconds),
        "finite": True,
        "checkpoint": (str(Path(arguments.checkpoint).resolve())
                       if arguments.checkpoint is not None else None),
        "history": str(history_path) if history_path is not None else None,
    }, sort_keys=True))
    return 0


__all__ = ["run_metal_dry", "tiny_dry_steps"]
