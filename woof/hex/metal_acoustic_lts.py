"""Local-timestep acoustic support on Metal.

OPT-IN.  Mirrors ``cuda_acoustic_lts.py``: the rate-class launch lists with
their interface bookkeeping, and the launch helper for the ``hexv841_*_lts``
and ``hexv841_lts_*`` shaders.  Each shader is the Metal transcription of the
kernel of the same name in the CUDA module; the gathered ones run the
ancestor's body for ``active[slot]`` instead of the thread index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .cuda_acoustic_lts import (
    _effective_rate,
    refuse_unsupported_ladder,
    stage_rate_plan,
)
from .metal_acoustic_v841 import _float32, _int32, _mesh_value
from .metal_v841_launch import launch as _launch


def launch(context: Any, name: str, count: int, arrays: Any, params: Any) -> None:
    """Launch ``hexv841_<name>`` over ``count`` threads (skipped when empty)."""

    if int(count) <= 0:
        return
    _launch(context, f"hexv841_{name}", int(count), arrays, params)


@dataclass(frozen=True, slots=True)
class MetalRateClassing:
    """Device-resident launch lists and the interface bookkeeping."""

    rates: tuple[int, ...]
    cell_lists: tuple[Any, ...]
    edge_lists: tuple[Any, ...]
    cell_sizes: tuple[int, ...]
    edge_sizes: tuple[int, ...]
    iface_edges: Any
    iface_coarse_cell: Any
    iface_coarse_sign: Any
    iface_coarse_rate: Any
    n_iface: int
    settle_cells: Any
    settle_offsets: Any
    settle_edges: Any
    settle_signs: Any
    n_settle: int
    resid: Any
    iface_edges_host: np.ndarray
    iface_coarse_rate_host: np.ndarray
    edge_rate_host: np.ndarray
    cell_rate_host: np.ndarray

    @property
    def single_class(self) -> bool:
        return len(self.rates) == 1 and int(self.rates[0]) == 1


def build_device_classing(
    classing: Any,
    mesh: Any,
    *,
    n_vert_levels: int,
    arrays: Any,
) -> MetalRateClassing:
    """Move a :class:`~woof.hex.lts_v841.LocalTimestepClassing` to the device."""

    cells_on_edge = np.asarray(_int32(_mesh_value(mesh, "cells_on_edge"), "cells_on_edge").to_numpy())
    edges_on_cell = np.asarray(_int32(_mesh_value(mesh, "edges_on_cell"), "edges_on_cell").to_numpy())
    signs = np.asarray(_float32(_mesh_value(mesh, "edge_sign_on_cell"), "edge_sign_on_cell").to_numpy())
    counts = np.asarray(_int32(_mesh_value(mesh, "n_edges_on_cell"), "n_edges_on_cell").to_numpy())

    cell_rate = np.asarray(classing.cell_rate, dtype=np.int32)
    edge_rate = np.asarray(classing.edge_rate, dtype=np.int32)
    n_cells = int(cell_rate.size)
    n_edges = int(edge_rate.size)
    if cells_on_edge.shape != (n_edges, 2):
        raise ValueError("device cellsOnEdge does not match the classing")

    iface = np.asarray(classing.interface_edges, dtype=np.int32)
    if iface.size:
        c0 = cells_on_edge[iface, 0].astype(np.int64)
        c1 = cells_on_edge[iface, 1].astype(np.int64)
        take_c1 = cell_rate[c1] > cell_rate[c0]
        coarse = np.where(take_c1, c1, c0).astype(np.int32)
        coarse_sign = np.empty(iface.size, dtype=np.float32)
        for position, (edge, cell) in enumerate(zip(iface.tolist(), coarse.tolist())):
            slots = edges_on_cell[cell, : int(counts[cell])]
            match = np.flatnonzero(slots == edge)
            if match.size != 1:
                raise ValueError(
                    "an interface edge is not listed exactly once on its coarse cell"
                )
            coarse_sign[position] = signs[cell, int(match[0])]
        if not np.all(np.abs(np.abs(coarse_sign) - 1.0) < 1e-6):
            raise ValueError("edgeSignOnCell must be exactly +-1 for the reflux to balance")
        coarse_rate = cell_rate[coarse.astype(np.int64)].astype(np.int32)
    else:
        coarse = np.zeros(0, dtype=np.int32)
        coarse_sign = np.zeros(0, dtype=np.float32)
        coarse_rate = np.zeros(0, dtype=np.int32)

    if iface.size:
        order = np.argsort(coarse.astype(np.int64), kind="stable")
        sorted_cells = coarse.astype(np.int64)[order]
        settle_cells, starts = np.unique(sorted_cells, return_index=True)
        settle_offsets = np.append(starts, sorted_cells.size).astype(np.int32)
        settle_edges = iface[order].astype(np.int32)
        settle_signs = coarse_sign[order].astype(np.float32)
        settle_cells = settle_cells.astype(np.int32)
        if int(settle_offsets[-1]) != int(iface.size):
            raise ValueError("the reflux settle grouping lost an interface edge")
    else:
        settle_cells = np.zeros(0, dtype=np.int32)
        settle_offsets = np.zeros(1, dtype=np.int32)
        settle_edges = np.zeros(0, dtype=np.int32)
        settle_signs = np.zeros(0, dtype=np.float32)

    def to_device(value: np.ndarray, dtype: Any = np.int32) -> Any:
        host = np.ascontiguousarray(value, dtype=dtype)
        if host.size == 0:
            host = np.zeros(1, dtype=dtype)  # a launch over zero threads never reads it
        return arrays.asarray(host)

    rates = tuple(int(value) for value in classing.rates)
    cell_lists_host = tuple(np.flatnonzero(cell_rate == rate).astype(np.int32) for rate in rates)
    edge_lists_host = tuple(np.flatnonzero(edge_rate == rate).astype(np.int32) for rate in rates)
    if sum(int(v.size) for v in cell_lists_host) != n_cells:
        raise ValueError("the cell launch lists do not partition the extent")
    if sum(int(v.size) for v in edge_lists_host) != n_edges:
        raise ValueError("the edge launch lists do not partition the extent")
    return MetalRateClassing(
        rates=rates,
        cell_lists=tuple(to_device(v) for v in cell_lists_host),
        edge_lists=tuple(to_device(v) for v in edge_lists_host),
        cell_sizes=tuple(int(v.size) for v in cell_lists_host),
        edge_sizes=tuple(int(v.size) for v in edge_lists_host),
        iface_edges=to_device(iface),
        iface_coarse_cell=to_device(coarse),
        iface_coarse_sign=to_device(coarse_sign, np.float32),
        iface_coarse_rate=to_device(coarse_rate),
        n_iface=int(iface.size),
        settle_cells=to_device(settle_cells),
        settle_offsets=to_device(settle_offsets),
        settle_edges=to_device(settle_edges),
        settle_signs=to_device(settle_signs, np.float32),
        n_settle=int(settle_cells.size),
        resid=(arrays.zeros((int(n_vert_levels), n_edges), np.float32) if iface.size else None),
        iface_edges_host=iface,
        iface_coarse_rate_host=coarse_rate,
        edge_rate_host=edge_rate,
        cell_rate_host=cell_rate,
    )


def entity_step_counts(
    classing: MetalRateClassing, stage_acoustic_steps: int, *, arrays: Any
) -> tuple[Any, Any]:
    """Per-cell and per-edge executed sub-step counts for one RK stage."""

    total = int(stage_acoustic_steps)
    cell_effective = np.array(
        [_effective_rate(int(value), total) for value in classing.cell_rate_host], dtype=np.int32
    )
    edge_effective = np.array(
        [_effective_rate(int(value), total) for value in classing.edge_rate_host], dtype=np.int32
    )
    return (
        arrays.asarray(np.ascontiguousarray(total // cell_effective, dtype=np.int32)),
        arrays.asarray(np.ascontiguousarray(total // edge_effective, dtype=np.int32)),
    )


__all__ = [
    "MetalRateClassing",
    "build_device_classing",
    "entity_step_counts",
    "launch",
    "refuse_unsupported_ladder",
    "stage_rate_plan",
]
