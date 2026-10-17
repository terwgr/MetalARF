"""Metal mirror of ``cuda_physics_v841``: the MPAS side of the physics seam.

Kernels (one shader each, ``hexv841_<cuda_name>``): ``couple_cells``,
``project_edges``, ``clamp_wsm6``, ``recover_wsm6``, ``commit_wsm6_state`` and
``validate_wsm6_surface``.  The CUDA kernel ``nonnegative_qv_copy_v841_f32`` has no
Metal counterpart because the CUDA route never calls it.

Differences from the CUDA module, all forced by Metal:

* CUDA's device tables of per-array pointers (``const float *const *``) cannot
  exist on Metal.  Each becomes ONE stacked buffer whose slot ``s`` is the
  table's entry ``s``:
    - ``couple_cells.rates``      -> ``rates_stack[nspecies, nlev, ncells]`` in
      model species order; a species with no phase-one rate is zero-filled.
    - ``recover_wsm6.water``      -> ``[nspecies, nlev, ncells]``, species order.
    - ``recover_wsm6.radius``     -> ``[nradius, nlev, ncells]``, ``row.radius_names``.
    - ``validate_wsm6_surface.acc_p/inc_p/old_p`` -> ``[nbuckets, ncells]`` in
      ``row.surface_accumulators`` order (``inc`` from ``<name>v``).
  The stacks are filled by device-to-device copies.
* ``kernel_cache`` arguments are replaced by the context carried by the arrays.
* The regional garbage-column hold (``regional_garbage_cell``) runs through
  the ``hexv841_copy_column_v841`` kernel, since ``MetalArray`` cannot index
  ``array[..., column]``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import time
from types import MappingProxyType
from typing import Any

import numpy as np

from .cuda_backend.containers import TransferStats
from .cuda_physics_v841 import (
    CUDA_PHYSICS_V841_CONTRACT_SHA256,
    CUDA_PHYSICS_V841_SCHEMA,
    CudaHeldMpasPhysicsTendenciesV841,
    CudaPhaseOneExecutionProvenanceV841,
    CudaPhysicsGeometryV841,
    CudaPostRkWsm6RecoveryV841,
    CudaPostRkWsm6UpdateV841,
    CudaRawColumnPhysicsV841,
    RV_OVER_RD_F32,
    WSM6_RADIUS_NAMES,
    WSM6_SCALAR_NAMES,
    WSM6_SURFACE_NAMES,
    _host_rows,
    _host_vectors,
    _mesh_value,
    _scalar_names,
    _wsm6_sr_roundoff_limit_v841,
)
from .errors import ConfigurationRefusal
from .metal_dynamics_v841 import _arrays, _context, _f, _i, _launch, _require
from .species_row import species_row_for_names


def _module(context: Any) -> Any:
    return _arrays(context)


@dataclass(slots=True)
class MetalPhysicsGeometryV841(CudaPhysicsGeometryV841):
    """One-time uploaded MPAS ``east/north/edgeNormalVectors`` geometry."""

    @classmethod
    def from_host(
        cls,
        context: Any,
        mesh: object,
        *,
        east: Any | None = None,
        north: Any | None = None,
        edge_normal_vectors: Any | None = None,
    ) -> "MetalPhysicsGeometryV841":
        raw_cells = np.asarray(_mesh_value(mesh, "cellsOnEdge"))
        if raw_cells.ndim != 2 or 2 not in raw_cells.shape:
            raise ValueError("cellsOnEdge must be a two-dimensional pair table")
        n_edges = raw_cells.shape[0] if raw_cells.shape[1] == 2 else raw_cells.shape[1]
        cells = _host_rows(raw_cells, n_edges, 2, "cellsOnEdge")

        east_value = _mesh_value(mesh, "east") if east is None else east
        north_value = _mesh_value(mesh, "north") if north is None else north
        normal_value = (
            _mesh_value(mesh, "edgeNormalVectors")
            if edge_normal_vectors is None
            else edge_normal_vectors
        )
        if east_value is None or north_value is None:
            lon = np.asarray(_mesh_value(mesh, "lonCell"))
            lat = np.asarray(_mesh_value(mesh, "latCell"))
            if lon.ndim != 1 or lat.shape != lon.shape:
                raise ValueError(
                    "mesh needs source-derived east/north or matching lonCell/latCell"
                )
            from .vector import zonal_meridional_vectors

            east_value, north_value, _ = zonal_meridional_vectors(lon, lat)
        east_shape = np.asarray(east_value).shape
        if len(east_shape) != 2 or 3 not in east_shape:
            raise ValueError("east must be an nCells-by-3 source vector table")
        if east_shape[1] == 3:
            n_cells = int(east_shape[0])
        elif east_shape[0] == 3:
            n_cells = int(east_shape[1])
        else:
            raise ValueError("ambiguous east source vector table")
        east_host = _host_vectors(east_value, n_cells, "east")
        north_host = _host_vectors(north_value, n_cells, "north")
        if normal_value is None:
            from .vector import initialize_vector_geometry

            normal_value = initialize_vector_geometry(mesh).edge_normal_vectors
        normal_host = _host_vectors(normal_value, n_edges, "edgeNormalVectors")
        cells = np.ascontiguousarray(cells, dtype=np.int32)
        if np.any(cells < 0) or np.any(cells >= n_cells):
            raise ValueError(
                "tend_toEdges requires two valid cell endpoints on every edge"
            )

        garbage_value = _mesh_value(mesh, "garbage_cell")
        garbage_cell = None if garbage_value is None else int(garbage_value)
        if garbage_cell is not None and garbage_cell != n_cells - 1:
            raise ValueError(
                "a limited-area mesh's garbage element is the LAST column of "
                f"its padded extent; this view declares {garbage_cell} with "
                f"{n_cells} columns"
            )
        metal = _module(context)
        started = time.perf_counter()
        device_cells = metal.asarray(cells)
        device_east = metal.asarray(east_host)
        device_north = metal.asarray(north_host)
        device_normal = metal.asarray(normal_host)
        elapsed = time.perf_counter() - started
        result = cls(
            cells_on_edge=device_cells,
            east_cell=device_east,
            north_cell=device_north,
            edge_normal_vectors=device_normal,
            n_cells=n_cells,
            n_edges=n_edges,
            h2d=TransferStats(
                int(
                    device_cells.nbytes
                    + device_east.nbytes
                    + device_north.nbytes
                    + device_normal.nbytes
                ),
                elapsed,
            ),
            regional_garbage_cell=garbage_cell,
        )
        result.validate()
        return result

    def validate(self) -> None:
        _require("physics_geometry.cells_on_edge", self.cells_on_edge,
                 np.int32, (self.n_edges, 2))
        for name, value, shape in (
            ("east_cell", self.east_cell, (self.n_cells, 3)),
            ("north_cell", self.north_cell, (self.n_cells, 3)),
            ("edge_normal_vectors", self.edge_normal_vectors, (self.n_edges, 3)),
        ):
            _require(f"physics_geometry.{name}", value, np.float32, shape)


@dataclass(frozen=True, slots=True)
class MetalRawColumnPhysicsV841(CudaRawColumnPhysicsV841):
    """Uncoupled A-grid rates held once per MPAS model step."""

    def validate(self, *, n_vert_levels: int, n_cells: int) -> None:
        shape = (n_vert_levels, n_cells)
        for name in ("du", "dv", "dtheta"):
            _require(f"raw_physics.{name}", getattr(self, name), np.float32, shape)
        for name, value in self.dscalars.items():
            _require(f"raw_physics.dscalars[{name!r}]", value, np.float32, shape)


@dataclass(frozen=True, slots=True)
class MetalHeldMpasPhysicsTendenciesV841(CudaHeldMpasPhysicsTendenciesV841):
    """Conservative physics tendencies retained through every RK stage."""

    def validate(
        self, *, n_vert_levels: int, n_cells: int, n_edges: int, n_scalars: int
    ) -> None:
        from .cuda_physics_v841 import _valid_clock

        _valid_clock(self.time_seconds, "held physics time_seconds")
        if not isinstance(
            self.execution_provenance, CudaPhaseOneExecutionProvenanceV841
        ):
            raise TypeError(
                "held physics requires CudaPhaseOneExecutionProvenanceV841"
            )
        self.execution_provenance.validate()
        if not isinstance(self.validation_d2h, TransferStats):
            raise TypeError("held physics validation_d2h must be TransferStats")
        if self.validation_d2h.bytes != 4:
            raise ValueError("held physics coupling validation must transfer four bytes")
        for name, value, shape in (
            ("rho", self.rho, (n_vert_levels, n_cells)),
            ("rho_u", self.rho_u, (n_vert_levels, n_edges)),
            ("rho_theta", self.rho_theta, (n_vert_levels, n_cells)),
            ("scalars", self.scalars, (n_scalars, n_vert_levels, n_cells)),
        ):
            _require(f"held_physics.{name}", value, np.float32, shape)


@dataclass(frozen=True, slots=True)
class MetalPostRkWsm6UpdateV841(CudaPostRkWsm6UpdateV841):
    """Direct post-RK microphysics state; these are not RK tendencies."""

    def validate(self, *, n_vert_levels: int, n_cells: int) -> None:
        from .cuda_physics_v841 import _valid_clock

        _valid_clock(self.time_seconds, "post-RK WSM6 time_seconds")
        volume_shape = (n_vert_levels, n_cells)
        for name in ("theta", *self.species, *self.radii):
            _require(f"post_rk_wsm6.{name}", getattr(self, name), np.float32,
                     volume_shape)
        for name in self.surface:
            _require(f"post_rk_wsm6.{name}", getattr(self, name), np.float32,
                     (n_cells,))


@dataclass(frozen=True, slots=True)
class MetalPostRkWsm6RecoveryV841(CudaPostRkWsm6RecoveryV841):
    """Validated resident MPAS state plus WSM6 persistent output carriers."""


def _validation_word(flag: Any) -> tuple[int, TransferStats]:
    started = time.perf_counter()
    value = int(flag.to_numpy()[0])
    return value, TransferStats(int(flag.nbytes), time.perf_counter() - started)


def _stack(arrays: Any, shape: tuple[int, ...], parts: Sequence[Any]) -> Any:
    """One stacked buffer ``[len(parts), *shape]``: slot ``s`` is ``parts[s]``.

    ``parts[s]`` may be ``None`` for a slot that stays zero.  Filled by
    device-to-device copies; nothing crosses the host.
    """

    stacked = arrays.zeros((len(parts), *shape), np.float32)
    for slot, part in enumerate(parts):
        if part is not None:
            stacked[slot][...] = part
    return stacked


def couple_raw_column_physics_v841(
    raw: CudaRawColumnPhysicsV841,
    *,
    state: Any,
    scalar_names: Sequence[str],
    geometry: CudaPhysicsGeometryV841,
    rho_edge: Any,
) -> MetalHeldMpasPhysicsTendenciesV841:
    """Build one validated conservative source (couple_cells + project_edges)."""

    geometry.validate()
    n_vert_levels, n_cells = tuple(state.rho.shape)
    n_edges = geometry.n_edges
    if n_cells != geometry.n_cells:
        raise ValueError(
            f"state carries {n_cells} cells but geometry carries {geometry.n_cells}"
        )
    _require("state.rho", state.rho, np.float32, (n_vert_levels, n_cells))
    _require("state.rho_theta", state.rho_theta, np.float32, (n_vert_levels, n_cells))
    _require("state.rho_u", state.rho_u, np.float32, (n_vert_levels, n_edges))
    names = _scalar_names(scalar_names, int(state.scalars.shape[0]))
    _require("state.scalars", state.scalars, np.float32,
             (len(names), n_vert_levels, n_cells))
    species_row_for_names(names)
    if float(raw.time_seconds) != float(state.time_seconds):
        raise ValueError(
            "raw physics time must equal the candidate step start exactly: "
            f"{raw.time_seconds} != {state.time_seconds}"
        )
    raw.validate(n_vert_levels=n_vert_levels, n_cells=n_cells)
    _require("physics.rho_edge", rho_edge, np.float32, (n_vert_levels, n_edges))
    mctx = _context(state.rho)
    arrays = _arrays(mctx)
    rho = arrays.empty(state.rho.shape, np.float32)
    rho_theta = arrays.empty(state.rho_theta.shape, np.float32)
    rho_u = arrays.empty(state.rho_u.shape, np.float32)
    scalars = arrays.empty(state.scalars.shape, np.float32)
    invalid = arrays.zeros((1,), np.int32)
    # CUDA's rate_table of device pointers -> one stacked buffer, slot s = the
    # model's species s; species the phase-one seam publishes no tendency for
    # stay at the zero fill (CUDA points them at one shared zero array).
    rates = _stack(arrays, (n_vert_levels, n_cells),
                   [raw.dscalars.get(name) for name in names])
    _launch(
        "couple_cells_v841_f32", n_cells, mctx,
        dict(rho=state.rho, rho_theta=state.rho_theta, q=state.scalars,
             dtheta=raw.dtheta, rates=rates, out_rho=rho,
             out_rtheta=rho_theta, out_q=scalars, invalid=invalid),
        dict(nlev=_i(n_vert_levels), ncells=_i(n_cells), nspecies=_i(len(names)),
             rvord=RV_OVER_RD_F32),
    )
    # As on CUDA: native's physics tendency at the limited-area garbage
    # element is zero, and the projection is the one place a real (ring-7,
    # one-cell) edge reads it.  The zero is held across the projection only,
    # since raw.du/raw.dv are the seam's own buffers.
    from .metal_physics_prep_v841 import _copy_column

    held: list[tuple[Any, Any]] = []
    garbage = geometry.regional_garbage_cell
    try:
        if garbage is not None:
            zero = arrays.zeros((n_vert_levels,), np.float32)
            for field in (raw.du, raw.dv):
                kept = arrays.empty((n_vert_levels,), np.float32)
                _copy_column(mctx, field, kept, n_vert_levels, n_cells, int(garbage), 1, 0)
                held.append((field, kept))
                _copy_column(mctx, zero, field, n_vert_levels, 1, 0, n_cells, int(garbage))
        _launch(
            "project_edges_v841_f32", n_edges, mctx,
            dict(cells=geometry.cells_on_edge, east=geometry.east_cell,
                 north=geometry.north_cell, normal=geometry.edge_normal_vectors,
                 du=raw.du, dv=raw.dv, rho_edge=rho_edge, out_ru=rho_u,
                 invalid=invalid),
            dict(nlev=_i(n_vert_levels), ncells=_i(n_cells), nedges=_i(n_edges)),
        )
    finally:
        for field, kept in held:
            _copy_column(mctx, kept, field, n_vert_levels, 1, 0, n_cells, int(garbage))
    invalid_value, validation_d2h = _validation_word(invalid)
    if invalid_value != 0:
        candidates: dict[str, Any] = {
            "state.rho (must be > 0)": state.rho,
            "state.rho_theta": state.rho_theta,
            "state.scalars": state.scalars,
            "raw.dtheta": raw.dtheta,
            "raw.du": raw.du,
            "raw.dv": raw.dv,
            "rho_edge": rho_edge,
            "coupled.rho_theta": rho_theta,
            "coupled.rho_u": rho_u,
            "coupled.scalars": scalars,
        }
        for species, rate in raw.dscalars.items():
            candidates[f"raw.d{species}"] = rate
        findings: list[str] = []
        for name, array in candidates.items():
            host = array.to_numpy()
            bad = ~np.isfinite(host)
            if name.startswith("state.rho ("):
                bad = bad | (host <= np.float32(0.0))
            count = int(np.count_nonzero(bad))
            if count == 0:
                continue
            last = np.unique(np.argwhere(bad)[:, -1])
            findings.append(
                f"{name}: {count} value(s) over {last.size} column(s)/edge(s), "
                f"first {int(last[0])} of {int(host.shape[-1])}"
            )
        detail = "; ".join(findings) if findings else (
            "every input and output is finite on the host, so the refusal is "
            "one of the kernels' consistency laws rather than a value"
        )
        raise FloatingPointError(
            "non-finite/invalid raw physics or conservative coupling result: "
            + detail
        )
    result = MetalHeldMpasPhysicsTendenciesV841(
        rho=rho,
        rho_u=rho_u,
        rho_theta=rho_theta,
        scalars=scalars,
        time_seconds=float(raw.time_seconds),
        validation_d2h=validation_d2h,
        execution_provenance=raw.execution_provenance,
    )
    result.validate(
        n_vert_levels=n_vert_levels, n_cells=n_cells, n_edges=n_edges,
        n_scalars=len(names),
    )
    return result


def clamp_wsm6_scalars_in_place_v841(
    scalars: Any, *, scalar_names: Sequence[str]
) -> TransferStats:
    """Apply native post-transport +0 clamp to all species carriers."""

    names = _scalar_names(scalar_names, int(scalars.shape[0]))
    species_row_for_names(names)
    if scalars.ndim != 3:
        raise ValueError("state.scalars must be (species, level, cell)")
    _require("state.scalars", scalars, np.float32)
    mctx = _context(scalars)
    invalid = _arrays(mctx).zeros((1,), np.int32)
    count = int(scalars.size)
    # Validate in a separate pass so a refusal cannot partially clamp the live
    # prognostic buffer; the second pass runs only after the word is read.
    _launch("clamp_wsm6_v841_f32", count, mctx,
            dict(values=scalars, invalid=invalid),
            dict(count=_i(count), apply_clamp=_i(0)))
    invalid_value, transfer = _validation_word(invalid)
    if invalid_value != 0:
        raise FloatingPointError("non-finite WSM6 carrier refused before phase two")
    _launch("clamp_wsm6_v841_f32", count, mctx,
            dict(values=scalars, invalid=invalid),
            dict(count=_i(count), apply_clamp=_i(1)))
    return transfer


def recover_post_rk_wsm6_state_v841(
    state: Any,
    update: CudaPostRkWsm6UpdateV841,
    *,
    scalar_names: Sequence[str],
    phase2_dt_seconds: float,
    previous_surface_updates: Mapping[str, Any] | None = None,
) -> MetalPostRkWsm6RecoveryV841:
    """Validate candidate buffers completely, then commit live state once."""

    n_vert_levels, n_cells = tuple(state.rho.shape)
    _require("state.rho", state.rho, np.float32, (n_vert_levels, n_cells))
    _require("state.rho_theta", state.rho_theta, np.float32, (n_vert_levels, n_cells))
    sr_upper, _, _ = _wsm6_sr_roundoff_limit_v841(phase2_dt_seconds)
    names = _scalar_names(scalar_names, int(state.scalars.shape[0]))
    row = species_row_for_names(names)
    _require("state.scalars", state.scalars, np.float32,
             (len(names), n_vert_levels, n_cells))
    if float(update.time_seconds) != float(state.time_seconds):
        raise ValueError(
            "WSM6 update time must equal the candidate endpoint exactly: "
            f"{update.time_seconds} != {state.time_seconds}"
        )
    update.validate(n_vert_levels=n_vert_levels, n_cells=n_cells)

    mctx = _context(state.rho)
    arrays = _arrays(mctx)
    candidate_rho_theta = arrays.empty(state.rho_theta.shape, np.float32)
    candidate_scalars = arrays.empty(state.scalars.shape, np.float32)
    invalid = arrays.zeros((1,), np.int32)
    count = n_vert_levels * n_cells
    volume = (n_vert_levels, n_cells)
    # CUDA's water_table / radius_table -> stacked buffers (see module doc).
    water = _stack(arrays, volume, [update.species[name] for name in names])
    radius = _stack(arrays, volume, [update.radii[name] for name in row.radius_names])
    _launch(
        "recover_wsm6_v841_f32", count, mctx,
        dict(rho=state.rho, theta=update.theta, water=water, radius=radius,
             rho_theta=candidate_rho_theta, scalars=candidate_scalars,
             invalid=invalid),
        dict(count=_i(count), nspecies=_i(len(names)),
             nradius=_i(len(update.radii)), rvord=RV_OVER_RD_F32),
    )
    previous = {} if previous_surface_updates is None else previous_surface_updates
    bucket_names = tuple(item.name for item in row.surface_accumulators)
    old_values = [
        previous.get(name, update.surface[name]) for name in bucket_names]
    for name, value in zip(bucket_names, old_values):
        _require(f"previous_surface.{name}", value, np.float32, (n_cells,))
    surface = (n_cells,)
    acc = _stack(arrays, surface, [update.surface[name] for name in bucket_names])
    inc = _stack(arrays, surface,
                 [update.surface[f"{name}v"] for name in bucket_names])
    old = _stack(arrays, surface, old_values)
    _launch(
        "validate_wsm6_surface_v841_f32", n_cells, mctx,
        dict(acc_p=acc, inc_p=inc, old_p=old, sr=update.sr, invalid=invalid),
        dict(count=_i(n_cells), nbuckets=_i(len(bucket_names)), sr_upper=_f(sr_upper)),
    )
    invalid_value, transfer = _validation_word(invalid)
    if invalid_value != 0:
        raise FloatingPointError(
            "post-RK WSM6 numeric validation refused the candidate state"
        )
    _launch(
        "commit_wsm6_state_v841_f32", count, mctx,
        dict(candidate_rho_theta=candidate_rho_theta,
             candidate_scalars=candidate_scalars,
             rho_theta=state.rho_theta, scalars=state.scalars),
        dict(count=_i(count), nspecies=_i(len(names))),
    )
    return MetalPostRkWsm6RecoveryV841(
        state=state,
        surface_updates=MappingProxyType(dict(update.surface)),
        effective_radii=MappingProxyType(dict(update.radii)),
        time_seconds=float(update.time_seconds),
        validation_d2h=transfer,
    )


__all__ = [
    "CUDA_PHYSICS_V841_CONTRACT_SHA256",
    "CUDA_PHYSICS_V841_SCHEMA",
    "MetalHeldMpasPhysicsTendenciesV841",
    "MetalPhysicsGeometryV841",
    "MetalPostRkWsm6RecoveryV841",
    "MetalPostRkWsm6UpdateV841",
    "MetalRawColumnPhysicsV841",
    "RV_OVER_RD_F32",
    "WSM6_RADIUS_NAMES",
    "WSM6_SCALAR_NAMES",
    "WSM6_SURFACE_NAMES",
    "clamp_wsm6_scalars_in_place_v841",
    "couple_raw_column_physics_v841",
    "recover_post_rk_wsm6_state_v841",
]
