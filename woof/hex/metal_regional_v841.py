"""Metal mirror of the MPAS-A v8.4.1 regional (limited-area) CUDA runtime.

``MetalRegionalRuntimeV841`` follows ``CudaRegionalRuntimeV841``
(``cuda_regional_forecast_v841``) method for method on resident ``MetalArray``
storage, one ``hexv841_*`` shader per CUDA kernel.  Host-side lateral-boundary
file reading stays in :mod:`woof.hex.lbc`; only the driving-state arrays are
uploaded.

Every regional shader runs one thread per element, so the launch count is the
element count of the CUDA kernel's grid-stride loop (the CUDA forecast
launches fewer threads for a few kernels and lets the loop cover the rest).
The regional kernels take every array as a flat view of the same storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .cuda_regional_forecast_v841 import (
    REGIONAL_GARBAGE_POOL,
    PaddedRegionalHostMesh,
    _PaddedReference,
    _PaddedSaved,
    _PaddedState,
    _PaddedTerrain,
    _PaddedVertical,
    _pad_axis,
    _pad_last,
)
from .cuda_regional_v841 import DRIVING_FIELD_KIND
from .diagnostics import edge_signs_on_vertices
from .errors import ConfigurationRefusal
from .lbc import LbcAdmissionError, LbcFile, LbcInventory, LbcPool
from .metal_v841_launch import launch
from .regional_v841 import (
    N_BDY_ZONE,
    N_RELAX_ZONE,
    RVORD_F32,
    RegionalAdmissionError,
    RegionalMasks,
    compute_mesh_scaling_regional,
    derive_regional_masks,
    dynamics_time_offset,
    regional_bdy_checks,
    rk_timestep_f32,
    transport_rk_timestep_f32,
)

_PREFIX = "hexv841_"


def _module(context: Any) -> Any:
    from woof.metal.array import MetalArrayModule

    return MetalArrayModule(context)


def _flat(value: Any) -> Any:
    return value if value.ndim == 1 else value.view((int(value.size),))


def _launch(context: Any, name: str, count: int, arrays: Mapping[str, Any],
            params: Mapping[str, Any]) -> None:
    """Launch one regional shader; its registered shape rule is any_1d."""

    launch(
        context, _PREFIX + name, int(count),
        {key: _flat(value) for key, value in arrays.items()}, dict(params))


def _i(value: Any) -> np.int32:
    return np.int32(value)


def _f(value: Any) -> float:
    return float(np.float32(value))


# ---------------------------------------------------------------------------
# the padded resident atmosphere and v8.4.1 context
# ---------------------------------------------------------------------------


def build_regional_metal_atmosphere(
    context: Any,
    padded_mesh: PaddedRegionalHostMesh,
    state: object,
    vertical: object,
    reference: object,
    saved: object,
    terrain: object,
) -> Any:
    """Upload one regional atmosphere in native's padded memory model.

    Mirror of ``build_regional_device_atmosphere``: the same host padding
    shims, uploaded through the Metal containers.
    """

    from .metal_containers_v841 import MetalAtmosphere

    return MetalAtmosphere.from_host(
        context, padded_mesh, _PaddedState(state), _PaddedVertical(vertical),
        _PaddedReference(reference), _PaddedSaved(saved),
        _PaddedTerrain(terrain))


def regional_padded_metal_context_v841(
    context: Any, mesh: object, offcentering: object, reference_wind: object,
    *, n_vert_levels: int,
) -> Any:
    """The v8.4.1 resident inverses with native's garbage allocation.

    Mirror of ``regional_padded_context_v841``: the inverses are built on the
    unpadded mesh and padded with 0.0.
    """

    from .dynamics_v841 import precomputed_mesh_inverse_v841
    from .metal_containers_v841 import MetalV841Context

    selected = np.dtype(np.float32)
    reference_wind.validate(n_vert_levels, selected)
    host = {
        "inv_area_cell": _pad_last(
            precomputed_mesh_inverse_v841(mesh, "areaCell", selected), 0.0),
        "inv_area_triangle": _pad_last(
            precomputed_mesh_inverse_v841(mesh, "areaTriangle", selected), 0.0),
        "inv_dc_edge": _pad_last(
            precomputed_mesh_inverse_v841(mesh, "dcEdge", selected), 0.0),
        "inv_dv_edge": _pad_last(
            precomputed_mesh_inverse_v841(mesh, "dvEdge", selected), 0.0),
        "etp": np.ascontiguousarray(np.asarray(offcentering.etp, selected)),
        "etm": np.ascontiguousarray(np.asarray(offcentering.etm, selected)),
        "ewp": np.ascontiguousarray(np.asarray(offcentering.ewp, selected)),
        "ewm": np.ascontiguousarray(np.asarray(offcentering.ewm, selected)),
        "u_init": np.ascontiguousarray(
            np.asarray(reference_wind.u_init, selected)),
        "v_init": np.ascontiguousarray(
            np.asarray(reference_wind.v_init, selected)),
    }
    for name, value in host.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"regional v8.4.1 context {name} must be finite")
    module = _module(context)
    return MetalV841Context(
        **{name: module.asarray(value) for name, value in host.items()})


# ---------------------------------------------------------------------------
# the solve-region checks
# ---------------------------------------------------------------------------

_ENVELOPE_THREADS = 256
_ENVELOPE_MAX_BLOCKS = 512
_VALIDATE_THREADS = 256
_VALIDATE_MAX_BLOCKS = 512
_PARTIAL_WORDS = 5
_RESULT_WORDS = 4


def _bits(value: float) -> int:
    """The int32 bit pattern of a float32, as a Python int."""

    return int(np.asarray(np.float32(value)).view(np.int32))


def _from_bits(bits: int) -> float:
    return float(np.asarray(np.int64(bits)).astype(np.int32).view(np.float32))


def describe_field(array: Any, n_solve: int | None, *, name: str = "field"
                   ) -> tuple[int, int, int]:
    """``(rows, stride, n_solve)`` for one contiguous float32 MetalArray."""

    if np.dtype(array.dtype) != np.dtype(np.float32):
        raise TypeError(f"{name} must be float32 for the solve-region kernels")
    if array.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    stride = int(array.shape[-1])
    rows = int(array.size // stride) if stride else 0
    solve = stride if n_solve is None else int(n_solve)
    if solve < 0 or solve > stride:
        raise ValueError(f"{name}: n_solve={solve} is outside [0, {stride}]")
    return rows, stride, solve


class MetalSolveRegionKernels:
    """Mirror of ``SolveRegionKernels``: envelope, validation, scrub.

    The CUDA kernels read a table of raw device pointers, one row per field;
    Metal resources are bound by name, so each field is its own launch of the
    same kernel body (the CUDA ``blockIdx.y``).
    """

    def __init__(self, context: Any) -> None:
        self.context = context
        self._module = _module(context)

    def envelope(
        self, fields: Sequence[tuple[str, Any, int | None]]
    ) -> dict[str, tuple[float, float, bool, int]]:
        if not fields:
            return {}
        rows = []
        widest = 1
        for name, array, n_solve in fields:
            count, stride, solve = describe_field(array, n_solve, name=name)
            rows.append((count, stride, solve))
            widest = max(widest, solve)
        n_fields = len(rows)
        blocks = min(_ENVELOPE_MAX_BLOCKS,
                     (widest + _ENVELOPE_THREADS - 1) // _ENVELOPE_THREADS)
        blocks = max(blocks, 1)
        partials = self._module.zeros(
            (n_fields * blocks * _PARTIAL_WORDS,), np.int64)
        out = self._module.zeros((n_fields * _RESULT_WORDS,), np.int64)
        for index, ((_name, array, _n), (count, stride, solve)) in enumerate(
                zip(fields, rows)):
            launch(
                self.context, _PREFIX + "solve_region_envelope_v841",
                blocks * _ENVELOPE_THREADS,
                {"field": _flat(array), "partials": partials},
                {"rows": _i(count), "stride": _i(stride), "n": _i(solve),
                 "field_index": _i(index), "blocks_per_field": _i(blocks)})
        launch(
            self.context, _PREFIX + "solve_region_envelope_finish_v841",
            n_fields * _ENVELOPE_THREADS,
            {"partials": partials, "out": out},
            {"n_fields": _i(n_fields), "blocks_per_field": _i(blocks)})
        words = out.to_numpy().reshape(n_fields, _RESULT_WORDS)
        result: dict[str, tuple[float, float, bool, int]] = {}
        for index, (name, _array, _n) in enumerate(fields):
            low = _from_bits(int(words[index, 0]))
            high = _from_bits(int(words[index, 1]))
            has_nan = bool(words[index, 2])
            argmax = int(words[index, 3])
            if argmax == 0x7FFFFFFFFFFFFFFF:
                argmax = 0
            result[name] = (low, high, has_nan, argmax)
        return result

    def validate(
        self,
        flag: Any,
        *,
        positive: Sequence[tuple[Any, int | None]] = (),
        finite: Sequence[tuple[Any, int | None]] = (),
    ) -> None:
        """Set ``flag[0] = 1`` if any listed element fails its test."""

        if (np.dtype(flag.dtype) != np.dtype(np.int32)
                or tuple(flag.shape) != (1,)):
            raise TypeError("validation flag must be a resident int32[1]")
        rows = []
        widest = 1
        for array, n_solve in positive:
            count, stride, solve = describe_field(array, n_solve)
            rows.append((array, count, stride, solve, 1))
            widest = max(widest, solve)
        for array, n_solve in finite:
            count, stride, solve = describe_field(array, n_solve)
            rows.append((array, count, stride, solve, 0))
            widest = max(widest, solve)
        if not rows:
            return
        blocks = max(1, min(_VALIDATE_MAX_BLOCKS,
                            (widest + _VALIDATE_THREADS - 1) // _VALIDATE_THREADS))
        for array, count, stride, solve, mode in rows:
            launch(
                self.context, _PREFIX + "solve_region_validate_v841",
                blocks * _VALIDATE_THREADS,
                {"field": _flat(array), "flag": flag},
                {"rows": _i(count), "stride": _i(stride), "n": _i(solve),
                 "positive": _i(mode), "blocks": _i(blocks)})

    def scrub_columns(self, entries: Sequence[tuple[Any, int, float]]) -> None:
        """``array[..., solve] = pool`` for every ``(array, solve, pool)``."""

        for array, solve, pool in entries:
            count, stride, _ = describe_field(array, solve)
            launch(
                self.context, _PREFIX + "solve_region_scrub_columns_v841",
                count,
                {"field": _flat(array)},
                {"rows": _i(count), "stride": _i(stride), "solve": _i(solve),
                 "pool_bits": _i(_bits(pool))})


# ---------------------------------------------------------------------------
# masks, kernels and the lateral-boundary pool
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MetalRegionalMasks:
    """The 7-ring masks and their derived element lists, resident."""

    bdy_mask_cell: Any
    bdy_mask_edge: Any
    spec_zone_mask_cell: Any
    spec_zone_mask_edge: Any
    spec_cells: Any
    spec_edges: Any
    relax_cells: Any
    relax_edges: Any
    nudged_cells: Any
    n_cells: int
    n_edges: int

    @classmethod
    def from_host(cls, context: Any, masks: RegionalMasks) -> "MetalRegionalMasks":
        module = _module(context)
        payload = {
            "bdy_mask_cell": np.ascontiguousarray(masks.bdy_mask_cell, np.int32),
            "bdy_mask_edge": np.ascontiguousarray(masks.bdy_mask_edge, np.int32),
            "spec_zone_mask_cell": np.ascontiguousarray(
                masks.spec_zone_mask_cell, np.float32),
            "spec_zone_mask_edge": np.ascontiguousarray(
                masks.spec_zone_mask_edge, np.float32),
            "spec_cells": np.ascontiguousarray(masks.spec_cells, np.int32),
            "spec_edges": np.ascontiguousarray(masks.spec_edges, np.int32),
            "relax_cells": np.ascontiguousarray(masks.relax_cells, np.int32),
            "relax_edges": np.ascontiguousarray(masks.relax_edges, np.int32),
            "nudged_cells": np.ascontiguousarray(masks.nudged_cells, np.int32),
        }
        return cls(
            **{name: module.asarray(value) for name, value in payload.items()},
            n_cells=int(masks.n_cells), n_edges=int(masks.n_edges))

    def validate(self) -> None:
        if tuple(self.bdy_mask_cell.shape) != (self.n_cells,):
            raise ValueError("regional.bdy_mask_cell shape differs from nCells")
        if tuple(self.bdy_mask_edge.shape) != (self.n_edges,):
            raise ValueError("regional.bdy_mask_edge shape differs from nEdges")

    @property
    def n_spec_cells(self) -> int:
        return int(self.spec_cells.size)

    @property
    def n_spec_edges(self) -> int:
        return int(self.spec_edges.size)

    @property
    def n_relax_cells(self) -> int:
        return int(self.relax_cells.size)

    @property
    def n_relax_edges(self) -> int:
        return int(self.relax_edges.size)

    @property
    def n_nudged_cells(self) -> int:
        return int(self.nudged_cells.size)


class MetalRegionalKernels:
    """Launches the regional shaders of one Metal context by CUDA name."""

    def __init__(self, context: Any) -> None:
        self.context = context

    def launch(self, name: str, count: int, arrays: Mapping[str, Any],
               params: Mapping[str, Any]) -> None:
        _launch(self.context, name, count, arrays, params)


class MetalRegionalDrivingState:
    """The two-level LBC pool with its derived coupled fields, on device.

    File decoding stays on the host (:mod:`woof.hex.lbc`); the derived
    coupled fields, interval tendencies and in-step interpolation are device
    arithmetic.
    """

    def __init__(
        self,
        inventory: LbcInventory,
        masks: Any,
        *,
        context: Any,
        cells_on_edge: Any,
        zz: Any,
        n_vert_levels: int,
        kernels: MetalRegionalKernels,
        scalar_names: Sequence[str] = ("lbc_qv",),
    ) -> None:
        self._pool = LbcPool(inventory)
        self._context = context
        self._module = _module(context)
        self._kernels = kernels
        self._masks = masks
        self._scalar_names = tuple(scalar_names)
        if not self._scalar_names:
            raise ConfigurationRefusal(
                "scalar_names", (),
                "the lbc_scalars var_array always carries at least qv",
                "scalar_names=('lbc_qv', ...)")
        self._cells_on_edge = cells_on_edge
        self._zz = zz
        self.nlev = int(n_vert_levels)
        self.ncells = int(masks.n_cells)
        self.nedges = int(masks.n_edges)
        self._cell_shape = (self.nlev, self.ncells)
        self._edge_shape = (self.nlev, self.nedges)
        self._state: dict[str, Any] | None = None
        self._tend: dict[str, Any] | None = None
        self._state_scalars: Any | None = None
        self._tend_scalars: Any | None = None

    @property
    def interval_end(self) -> datetime:
        return self._pool.interval_end

    @property
    def scalar_names(self) -> tuple[str, ...]:
        return self._scalar_names

    def _slab(self, admitted: LbcFile, name: str) -> np.ndarray:
        return np.ascontiguousarray(
            np.asarray(admitted.fields[name], dtype=np.float32).T)

    def _upload(self, host: np.ndarray) -> Any:
        return self._module.asarray(np.ascontiguousarray(host))

    def _derive(self, admitted: LbcFile) -> tuple[dict[str, Any], Any]:
        u = self._upload(self._slab(admitted, "lbc_u"))
        w = self._upload(self._slab(admitted, "lbc_w"))
        rho = self._upload(self._slab(admitted, "lbc_rho"))
        theta = self._upload(self._slab(admitted, "lbc_theta"))
        scalars = self._upload(np.stack(
            [self._slab(admitted, name) for name in self._scalar_names], axis=0))
        qv = scalars[0]
        rho_zz = self._module.empty(self._cell_shape, np.float32)
        rtheta_m = self._module.empty(self._cell_shape, np.float32)
        self._kernels.launch(
            "regional_lbc_derive_v841", self.nlev * self.ncells,
            {"zz": self._zz, "lbc_rho": rho, "lbc_theta": theta,
             "lbc_qv": qv, "rho_zz": rho_zz, "rtheta_m": rtheta_m},
            {"nvalues": _i(self.nlev * self.ncells),
             "rvord": _f(RVORD_F32)})
        # The pool allocation value one-cell (ring-7) rho_edge slots keep.
        rho_edge = self._module.zeros(self._edge_shape, np.float32)
        ru = self._module.empty(self._edge_shape, np.float32)
        self._kernels.launch(
            "regional_lbc_rho_edge_v841", self.nlev * self.nedges,
            {"cells_on_edge": self._cells_on_edge, "rho_zz": rho_zz,
             "lbc_u": u, "rho_edge": rho_edge, "ru": ru},
            {"nlev": _i(self.nlev), "ncells": _i(self.ncells),
             "nedges": _i(self.nedges)})
        state = {
            "u": u, "ru": ru, "rho_edge": rho_edge, "w": w, "rho": rho,
            "rho_zz": rho_zz, "theta": theta, "rtheta_m": rtheta_m,
        }
        return state, scalars

    def start(self, when: datetime) -> None:
        admitted = self._pool.start(when)
        self._state, self._state_scalars = self._derive(admitted)
        self._tend = None
        self._tend_scalars = None

    def advance(self, when: datetime | None = None) -> None:
        if self._state is None or self._state_scalars is None:
            raise LbcAdmissionError(
                "advance was called on a driving state that never started")
        old_state = self._state
        old_scalars = self._state_scalars
        old_end = self._pool.interval_end
        admitted = self._pool.advance(when)
        new_state, new_scalars = self._derive(admitted)
        # mpas_atm_boundaries.F:265-272 -- the interval in whole days plus
        # seconds, in REAL(RKIND), then one reciprocal.
        delta = admitted.valid_time - old_end
        dt = np.float32(
            np.float32(86400.0) * np.float32(delta.days)
            + np.float32(delta.seconds))
        inv_dt = np.float32(np.float32(1.0) / dt)
        tend: dict[str, Any] = {}
        for name in DRIVING_FIELD_KIND:
            out = self._module.empty(new_state[name].shape, np.float32)
            self._kernels.launch(
                "regional_lbc_tendency_v841", int(out.size),
                {"new_state": new_state[name], "old_state": old_state[name],
                 "tend": out},
                {"nvalues": _i(out.size), "inv_dt": _f(inv_dt)})
            tend[name] = out
        tend_scalars = self._module.empty(new_scalars.shape, np.float32)
        self._kernels.launch(
            "regional_lbc_tendency_v841", int(tend_scalars.size),
            {"new_state": new_scalars, "old_state": old_scalars,
             "tend": tend_scalars},
            {"nvalues": _i(tend_scalars.size), "inv_dt": _f(inv_dt)})
        self._tend = tend
        self._tend_scalars = tend_scalars
        self._state = new_state
        self._state_scalars = new_scalars

    def _require_ready(self) -> None:
        if self._tend is None or self._tend_scalars is None:
            raise LbcAdmissionError(
                "the regional driving state holds no complete interval; "
                "start then advance must both run before boundary "
                "tendencies or interpolated state exist")

    def tendency(self, name: str) -> Any:
        self._require_ready()
        assert self._tend is not None
        if name == "scalars":
            return self._tend_scalars
        if name not in DRIVING_FIELD_KIND:
            raise ConfigurationRefusal(
                "field", name, "not a driving field of the regional lbc pool",
                f"one of {sorted(DRIVING_FIELD_KIND)} or 'scalars'")
        return self._tend[name]

    def remaining_seconds(self, step_start: datetime,
                          delta_t: np.float32) -> np.float32:
        """``dt_remaining`` of ``mpas_atm_get_bdy_state`` (F:491-551)."""

        delta = self._pool.interval_end - step_start
        remaining = np.float32(
            np.float32(86400.0) * np.float32(delta.days)
            + np.float32(delta.seconds))
        return np.float32(remaining - np.float32(delta_t))

    def state_at(self, name: str, step_start: datetime, delta_t: np.float32,
                 out: Any = None) -> Any:
        """Device-side ``mpas_atm_get_bdy_state``: state - dt*tend."""

        self._require_ready()
        assert self._state is not None and self._tend is not None
        dt = self.remaining_seconds(step_start, delta_t)
        if name == "scalars":
            state = self._state_scalars
            tend = self._tend_scalars
        else:
            if name not in DRIVING_FIELD_KIND:
                raise ConfigurationRefusal(
                    "field", name,
                    "not a driving field of the regional lbc pool",
                    f"one of {sorted(DRIVING_FIELD_KIND)} or 'scalars'")
            state = self._state[name]
            tend = self._tend[name]
        target = self._module.empty(state.shape, np.float32) if out is None else out
        self._kernels.launch(
            "regional_lbc_state_at_v841", int(state.size),
            {"state": state, "tend": tend, "out": target},
            {"nvalues": _i(state.size), "dt_remaining": _f(dt)})
        return target


class _PaddedMaskExtents:
    __slots__ = ("n_cells", "n_edges")

    def __init__(self, n_cells: int, n_edges: int) -> None:
        self.n_cells = int(n_cells)
        self.n_edges = int(n_edges)


class PaddedMetalRegionalDrivingState(MetalRegionalDrivingState):
    """The pool on native's padded stride: each file slab gains a garbage
    column at the native allocation 0.0 before upload."""

    def __init__(self, *args: Any, garbage_cell: int, garbage_edge: int,
                 **kwargs: Any) -> None:
        self._garbage_cell = int(garbage_cell)
        self._garbage_edge = int(garbage_edge)
        super().__init__(*args, **kwargs)

    def _slab(self, admitted: LbcFile, name: str) -> np.ndarray:
        return _pad_last(super()._slab(admitted, name), 0.0)


# ---------------------------------------------------------------------------
# the garbage-column discipline
# ---------------------------------------------------------------------------


class MetalRegionalGarbageDiscipline:
    """Restore native's pool value into every garbage column.

    Same registry as ``RegionalGarbageDiscipline``: immutable residency is
    never rewritten, ``rho_zz`` images are bound to the 1.0 pool, everything
    else gets 0.0.  Arrays are identified by their (buffer, offset).  Nothing
    calls :meth:`scrub` after a launch yet: Metal launches have no
    post-launch hook, so the caller decides where to wire it.
    """

    def __init__(self, *, n_cells_solve: int, n_edges_solve: int,
                 n_vertices_solve: int, solve_region: MetalSolveRegionKernels
                 ) -> None:
        self.n_cells_solve = int(n_cells_solve)
        self.n_edges_solve = int(n_edges_solve)
        self.n_vertices_solve = int(n_vertices_solve)
        self._extents = {
            self.n_cells_solve + 1: self.n_cells_solve,
            self.n_edges_solve + 1: self.n_edges_solve,
            self.n_vertices_solve + 1: self.n_vertices_solve,
        }
        if len(self._extents) != 3:
            raise ValueError(
                "a regional cull whose padded cell/edge/vertex extents "
                "collide cannot be disciplined by extent; this mesh has "
                f"nCells={n_cells_solve} nEdges={n_edges_solve} "
                f"nVertices={n_vertices_solve}")
        self.solve_region = solve_region
        self._immutable: set[tuple[int, int]] = set()
        self._unit_pool: dict[tuple[int, int], Any] = {}
        self._permanent_unit_pool: dict[tuple[int, int], Any] = {}
        self.scrubs = 0
        self.enabled = True
        self.last_kernel = ""

    @staticmethod
    def _key(array: Any) -> tuple[int, int]:
        return (id(array.buffer), int(array.byte_offset))

    def hold_immutable(self, *arrays: Any) -> None:
        for array in arrays:
            if array is None or getattr(array, "ndim", 0) == 0:
                continue
            self._immutable.add(self._key(array))

    def bind_unit_pool(self, array: Any, *, permanent: bool = False) -> Any:
        """Register ``rho_zz``: its garbage cell is 1.0, not 0.0."""

        if array is None:
            return array
        if permanent:
            self._permanent_unit_pool[self._key(array)] = array
        self._unit_pool[self._key(array)] = array
        self.solve_region.scrub_columns(
            [(array, self.n_cells_solve, REGIONAL_GARBAGE_POOL["rho_zz"])])
        return array

    def bound_to_unit_pool(self, array: Any) -> bool:
        return self._key(array) in self._unit_pool

    def release_unit_pool(self) -> None:
        self._unit_pool.clear()
        self._unit_pool.update(self._permanent_unit_pool)

    def scrub(self, name: str, arrays: Iterable[Any]) -> None:
        """Restore the pool value in every padded float32 array given."""

        if not self.enabled:
            return
        self.last_kernel = name
        batch = []
        for value in arrays:
            if (not hasattr(value, "buffer") or value.ndim == 0
                    or np.dtype(value.dtype) != np.dtype(np.float32)):
                continue
            solve = self._extents.get(int(value.shape[-1]))
            if solve is None or self._key(value) in self._immutable:
                continue
            pool = (REGIONAL_GARBAGE_POOL["rho_zz"]
                    if self._key(value) in self._unit_pool
                    else REGIONAL_GARBAGE_POOL["default"])
            batch.append((value, solve, pool))
            self.scrubs += 1
        self.solve_region.scrub_columns(batch)

    def receipt(self) -> dict[str, Any]:
        return {
            "n_cells_solve": self.n_cells_solve,
            "n_edges_solve": self.n_edges_solve,
            "n_vertices_solve": self.n_vertices_solve,
            "garbage_columns_restored": self.scrubs,
        }


# ---------------------------------------------------------------------------
# the regional acoustic substep and scalar transport
# ---------------------------------------------------------------------------


def advance_acoustic_step_regional_metal_v841(
    runtime: "MetalRegionalRuntimeV841",
    mesh: Any,
    state: Any,
    forcing: Any,
    coefficients: Any,
    *,
    context: Any,
    dts: float,
    small_step: int,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
    gravity: float = 9.80616,
    rgas: float = 287.0,
    cp_dry: float = 1004.5,
) -> Any:
    """One regional acoustic substep: three regional shaders plus the shared
    ``acoustic_prepare_v841`` shader, unchanged."""

    ctx = runtime.context_metal
    module = _module(ctx)
    out = state
    nlev, nedges = map(int, out.ru_p.shape)
    ncells = int(out.rho_pp.shape[1])
    max_edges = int(mesh.max_edges)
    runtime.kernels.launch(
        "acoustic_ru_regional_v841", nlev * nedges,
        {"cells_on_edge": runtime.cells_on_edge_sentinel,
         "inv_dc_edge": context.inv_dc_edge,
         "spec_zone_mask_edge": mesh.spec_zone_mask_edge,
         "zz": forcing.zz, "exner": forcing.exner, "cqu": forcing.cqu,
         "zxu": forcing.zxu, "tend_ru": forcing.tend_ru,
         "rho_pp": out.rho_pp, "rtheta_pp": out.rtheta_pp,
         "ru_p": out.ru_p, "ru_avg": out.ru_avg,
         "invalid": runtime.acoustic_invalid},
        {"nlev": _i(nlev), "nedges": _i(nedges), "ncells": _i(ncells),
         "small_step": _i(small_step), "dts": _f(dts),
         "gravity": _f(gravity), "rgas": _f(rgas), "cp": _f(cp_dry)})
    launch(
        ctx, "hexv841_acoustic_prepare_v841", ncells,
        {"rw_p": out.rw_p, "rtheta_pp": out.rtheta_pp,
         "rtheta_pp_old": out.rtheta_pp_old, "rho_pp": out.rho_pp,
         "ww_avg": out.ww_avg},
        {"nlev": _i(nlev), "ncells": _i(ncells),
         "small_step": _i(small_step)})
    rs = module.zeros((nlev, ncells), np.float32)
    ts = module.zeros((nlev, ncells), np.float32)
    runtime.kernels.launch(
        "acoustic_rs_ts_regional_v841", nlev * ncells,
        {"n_edges_on_cell": mesh.n_edges_on_cell,
         "edges_on_cell": mesh.edges_on_cell,
         "cells_on_edge": mesh.cells_on_edge,
         "edge_sign_on_cell": mesh.edge_sign_on_cell,
         "dv_edge": mesh.dv_edge, "inv_area_cell": context.inv_area_cell,
         "spec_zone_mask_cell": runtime.spec_zone_mask_cell,
         "theta_m": forcing.theta_m, "rdzw": rdzw,
         "cofrz": coefficients.cofrz, "coftz": coefficients.coftz,
         "ewm": context.ewm, "ru_p": out.ru_p, "rw_p": out.rw_p,
         "rho_pp": out.rho_pp, "rtheta_pp": out.rtheta_pp,
         "tend_rho": forcing.tend_rho, "tend_rt": forcing.tend_rt,
         "rs": rs, "ts": ts},
        {"nlev": _i(nlev), "ncells": _i(ncells), "nedges": _i(nedges),
         "max_edges": _i(max_edges), "dts": _f(dts)})
    runtime.kernels.launch(
        "acoustic_column_solve_regional_v841", ncells,
        {"zz": forcing.zz, "rho_zz": forcing.rho_zz, "fzm": fzm, "fzp": fzp,
         "rdzw": rdzw, "dss": forcing.dss, "w": forcing.w, "rw": forcing.rw,
         "rw_save": forcing.rw_save, "tend_rw": forcing.tend_rw,
         "tend_rho": forcing.tend_rho, "tend_rt": forcing.tend_rt,
         "rs": rs, "ts": ts,
         "spec_zone_mask_cell": runtime.spec_zone_mask_cell,
         "cofwr": coefficients.cofwr, "cofwz": coefficients.cofwz,
         "coftz": coefficients.coftz, "cofwt": coefficients.cofwt,
         "cofrz": coefficients.cofrz, "a_tri": coefficients.a_tri,
         "alpha_tri": coefficients.alpha_tri,
         "gamma_tri": coefficients.gamma_tri, "etp": context.etp,
         "etm": context.etm, "ewp": context.ewp, "ewm": context.ewm,
         "rw_p": out.rw_p, "rho_pp": out.rho_pp,
         "rtheta_pp": out.rtheta_pp, "ww_avg": out.ww_avg},
        {"nlev": _i(nlev), "ncells": _i(ncells), "dts": _f(dts)})
    return out


def advance_scalars_regional_metal_v841(
    runtime: "MetalRegionalRuntimeV841",
    mesh: Any,
    context: Any,
    scalar_old: Any,
    scalar_stage: Any,
    rho_zz_old: Any,
    rho_zz_new: Any,
    uh_avg: Any,
    ww_avg: Any,
    dt: float,
    *,
    coefficients: Any,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
    rk_step: int,
    config_coef_3rd_order: float,
    validation_flag: Any,
    scalar_tendency: Any | None = None,
) -> tuple[Any, Any]:
    """The regional split scalar transport, F:4764-4861.

    ``scalar_tendency`` is the held physics source (``tend_scalars``), added
    in the cell finish as the global transport adds it; none means zero.

    The edge-value zone split and the cell finish are the regional shaders;
    the vertical flux and the target-density interpolation are the shared
    Metal transport shaders, unchanged.
    """

    ctx = runtime.context_metal
    module = _module(ctx)
    ntracers, nlev, ncells = map(int, scalar_stage.shape)
    nedges = int(uh_avg.shape[1])
    max_edges = int(mesh.max_edges)
    width = int(coefficients.adv_coefs.shape[1])
    edge_values = module.zeros((ntracers, nlev, nedges), np.float32)
    runtime.kernels.launch(
        "transport_edge_values_regional_v841", ntracers * nlev * nedges,
        {"bdy_mask_edge": runtime.bdy_mask_edge, "dv_edge": mesh.dv_edge,
         "cells_on_edge": mesh.cells_on_edge, "stage": scalar_stage,
         "velocity": uh_avg, "adv": coefficients.adv_coefs,
         "adv3": coefficients.adv_coefs_3rd,
         "n_adv": coefficients.n_adv_cells_for_edge,
         "adv_cells": coefficients.adv_cells_for_edge,
         "edge_values": edge_values},
        {"ntracers": _i(ntracers), "nlev": _i(nlev), "ncells": _i(ncells),
         "nedges": _i(nedges), "width": _i(width),
         "coefficient": _f(config_coef_3rd_order)})
    vertical_flux = module.zeros((ntracers, nlev + 1, ncells), np.float32)
    launch(
        ctx, "hexv841_transport_vertical_flux",
        ntracers * (nlev + 1) * ncells,
        {"stage": scalar_stage, "velocity": ww_avg, "fzm": fzm, "fzp": fzp,
         "vertical_flux": vertical_flux},
        {"ntracers": _i(ntracers), "nlev": _i(nlev), "ncells": _i(ncells),
         "coefficient": _f(config_coef_3rd_order)})
    weight = np.float32(
        1.0 / 3.0 if rk_step == 1 else 0.5 if rk_step == 2 else 1.0)
    target = module.empty(rho_zz_old.shape, np.float32)
    launch(
        ctx, "hexv841_transport_interpolate_target_v841", nlev * ncells,
        {"rho_old": rho_zz_old, "rho_new": rho_zz_new, "target": target},
        {"nvalues": _i(nlev * ncells), "weight": _f(weight)})
    runtime.validate_density(target, validation_flag)
    output = module.empty(scalar_old.shape, np.float32)
    runtime.kernels.launch(
        "transport_standard_finish_regional_v841", ntracers * nlev * ncells,
        {"bdy_mask_cell": runtime.bdy_mask_cell,
         "n_edges_on_cell": mesh.n_edges_on_cell,
         "edges_on_cell": mesh.edges_on_cell,
         "acoustic_sign": mesh.edge_sign_on_cell,
         "inv_area_cell": context.inv_area_cell, "velocity": uh_avg,
         "edge_values": edge_values, "vertical_flux": vertical_flux,
         "rdzw": rdzw, "old": scalar_old, "rho_old": rho_zz_old,
         "target_density": target,
         "source": (module.zeros(scalar_old.shape, np.float32)
                    if scalar_tendency is None else scalar_tendency),
         "stage": scalar_stage, "output": output},
        {"ntracers": _i(ntracers), "nlev": _i(nlev), "ncells": _i(ncells),
         "nedges": _i(nedges), "max_edges": _i(max_edges), "dt": _f(dt)})
    return output, target


# ---------------------------------------------------------------------------
# the device-side regional runtime the driver hooks call
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MetalMoistDynamicsCoefficients:
    """Resident cqu/cqw/qtot held for one full dynamics step."""

    qtot: Any
    cqw: Any
    cqu: Any


@dataclass(slots=True)
class _StageOffsets:
    dynamics_substep: int = 1


class MetalRegionalRuntimeV841:
    """Everything the Metal whole-step driver needs for the regional branch."""

    def __init__(
        self,
        mesh: object,
        padded: PaddedRegionalHostMesh,
        *,
        context: Any,
        lbc_paths: Sequence[str],
        start_time: datetime,
        config_h_scale_with_mesh: bool,
        config_apply_lbcs: bool,
        outer_dt: float,
        dynamics_split: int,
        n_vert_levels: int,
        device_atmosphere: Any,
        v841_context: Any,
        config_relax_zone_divdamp_coef: float = 6.0,
        scalar_names: Sequence[str] = ("lbc_qv",),
    ) -> None:
        self.driven_scalars = int(len(tuple(scalar_names)))
        self.context_metal = context
        module = _module(context)
        self._module = module
        dtype = np.dtype(np.float32)
        self.masks_host = derive_regional_masks(mesh, dtype)
        regional_bdy_checks(
            self.masks_host,
            config_apply_lbcs=bool(config_apply_lbcs),
            lbc_input_interval_valid=True)
        self.masks = MetalRegionalMasks.from_host(context, self.masks_host)
        self.masks.validate()
        self.padded = padded
        self.n_cells_solve = padded.n_cells_solve
        self.n_edges_solve = padded.n_edges_solve
        self.nlev = int(n_vert_levels)
        self.outer_dt = float(outer_dt)
        self.dynamics_split = int(dynamics_split)
        self.config_relax_zone_divdamp_coef = float(
            config_relax_zone_divdamp_coef)
        self.kernels = MetalRegionalKernels(context)

        scaling_cell, scaling_edge = compute_mesh_scaling_regional(
            mesh, dtype, config_h_scale_with_mesh=config_h_scale_with_mesh)
        self.mesh_scaling_cell = module.asarray(
            np.ascontiguousarray(scaling_cell, dtype=np.float32))
        self.mesh_scaling_edge = module.asarray(
            np.ascontiguousarray(scaling_edge, dtype=np.float32))

        self.atmosphere = device_atmosphere
        self.context = v841_context
        device_mesh = device_atmosphere.mesh
        self.cells_on_edge = device_mesh.cells_on_edge
        self.cells_on_edge_sentinel = module.asarray(np.ascontiguousarray(
            padded.arrays["cellsOnEdgeSentinel"], dtype=np.int32))
        self.spec_zone_mask_cell = module.asarray(np.ascontiguousarray(
            padded.arrays["spec_zone_mask_cell"], dtype=np.float32))
        self.spec_zone_mask_edge = device_mesh.spec_zone_mask_edge
        # The zone masks the transport shaders index by thread carry the
        # garbage element, in the outermost ring.
        self.bdy_mask_cell = module.asarray(np.ascontiguousarray(np.append(
            np.asarray(self.masks_host.bdy_mask_cell, np.int32),
            np.int32(N_BDY_ZONE))))
        self.bdy_mask_edge = module.asarray(np.ascontiguousarray(np.append(
            np.asarray(self.masks_host.bdy_mask_edge, np.int32),
            np.int32(N_BDY_ZONE))))
        self.edge_sign_on_vertex = module.asarray(np.ascontiguousarray(
            _pad_axis(np.asarray(edge_signs_on_vertices(mesh),
                                 dtype=np.float32), 0, 0.0)))

        self.driving = PaddedMetalRegionalDrivingState(
            LbcInventory(list(lbc_paths)),
            _PaddedMaskExtents(padded.n_cells, padded.n_edges),
            context=context,
            garbage_cell=padded.garbage_cell,
            garbage_edge=padded.garbage_edge,
            cells_on_edge=self.cells_on_edge_sentinel,
            zz=device_atmosphere.vertical.zz,
            n_vert_levels=self.nlev,
            kernels=self.kernels,
            scalar_names=scalar_names)
        self.clock = start_time
        self._started = False
        self._offsets = _StageOffsets()
        self._u_scratch: Any = None

        self.solve_region = MetalSolveRegionKernels(context)
        self.discipline = MetalRegionalGarbageDiscipline(
            n_cells_solve=padded.n_cells_solve,
            n_edges_solve=padded.n_edges_solve,
            n_vertices_solve=padded.n_vertices_solve,
            solve_region=self.solve_region)
        self._hold_residency()
        # rho_zz's garbage cell is 1.0 for the whole run.
        self.discipline.bind_unit_pool(device_atmosphere.state.rho,
                                       permanent=True)
        self.acoustic_invalid = module.zeros((1,), np.int32)

    # -- residency ---------------------------------------------------------

    def _hold_residency(self) -> None:
        """Mesh geometry and the resident inverses are never rewritten."""

        mesh = self.atmosphere.mesh
        vertical = self.atmosphere.vertical
        reference = self.atmosphere.reference
        terrain = self.atmosphere.terrain
        self.discipline.hold_immutable(
            *(getattr(mesh, name) for name in (
                "weights_on_edge", "dc_edge", "dv_edge", "area_cell",
                "area_triangle", "kite_areas_on_vertex", "lat_cell",
                "lon_cell", "lat_edge", "lon_edge", "angle_edge",
                "mesh_density", "defc_a", "defc_b", "f_vertex", "f_edge",
                "edge_sign_on_cell", "spec_zone_mask_edge")),
            *(getattr(vertical, name)
              for name in ("hx", "zgrid", "zz", "zxu", "dss")),
            *(getattr(reference, name) for name in (
                "rho_base", "rho_theta_base", "pressure_base", "exner_base")),
            *(getattr(self.context, name) for name in (
                "inv_area_cell", "inv_area_triangle", "inv_dc_edge",
                "inv_dv_edge")),
            self.mesh_scaling_cell,
            self.mesh_scaling_edge,
            self.spec_zone_mask_cell,
            *(() if terrain is None else (terrain.zb_cell, terrain.zb3_cell)))

    def bind_state_rho(self, rho: Any) -> None:
        """Re-declare ``rho_zz``'s garbage cell after a state rebinding."""

        self.discipline.bind_unit_pool(rho)

    def begin_step(self, state: Any) -> None:
        """Everything the CPU authority does before the step's arithmetic."""

        self.discipline.release_unit_pool()
        self.discipline.bind_unit_pool(state.rho)
        self.ensure_interval()

    # -- clock and the lbc_in cadence -------------------------------------

    def ensure_interval(self) -> None:
        """``atm_core_run``'s ``lbc_in`` read cadence (F:735-781)."""

        if not self._started:
            self.driving.start(self.clock)
            self.driving.advance(self.clock)
            self._started = True
            return
        if self.clock >= self.driving.interval_end:
            self.driving.advance(self.clock)

    def advance_clock(self, seconds: float) -> None:
        from datetime import timedelta

        self.clock = self.clock + timedelta(seconds=seconds)

    def begin_dynamics_substep(self, substep: int) -> None:
        self._offsets.dynamics_substep = int(substep)

    def _stage_offset(self, rk_step: int) -> np.float32:
        return dynamics_time_offset(
            outer_dt=self.outer_dt,
            dynamics_split=self.dynamics_split,
            dynamics_substep=self._offsets.dynamics_substep,
            rk_timestep=rk_timestep_f32(
                outer_dt=self.outer_dt,
                dynamics_split=self.dynamics_split,
                rk_step=int(rk_step)))

    def driving_values(self, name: str, rk_step: int) -> Any:
        return self.driving.state_at(name, self.clock, self._stage_offset(rk_step))

    # -- graft: the tendency adjust pair (atm_srk3:2300-2351) --------------

    def adjust_dynamics_tendencies(
        self,
        *,
        tend_ru: Any,
        tend_rho: Any,
        tend_rt: Any,
        tend_omega: Any,
        rho_u: Any,
        theta_m: Any,
        rho_zz: Any,
        rk_step: int,
    ) -> None:
        nlev = self.nlev
        ncells = int(tend_rho.shape[-1])
        nedges = int(tend_ru.shape[-1])
        masks = self.masks
        self.kernels.launch(
            "regional_speczone_tend_cell_v841", nlev * masks.n_spec_cells,
            {"spec_cells": masks.spec_cells,
             "rho_driving_tend": self.driving.tendency("rho_zz"),
             "rt_driving_tend": self.driving.tendency("rtheta_m"),
             "tend_rho": tend_rho, "tend_rt": tend_rt, "tend_rw": tend_omega},
            {"nlev": _i(nlev), "ncells": _i(ncells),
             "nspec": _i(masks.n_spec_cells)})
        self.kernels.launch(
            "regional_speczone_tend_edge_v841", nlev * masks.n_spec_edges,
            {"spec_edges": masks.spec_edges,
             "ru_driving_tend": self.driving.tendency("ru"),
             "tend_ru": tend_ru},
            {"nlev": _i(nlev), "nedges": _i(nedges),
             "nspec": _i(masks.n_spec_edges)})
        values_ru = self.driving_values("ru", rk_step)
        values_rt = self.driving_values("rtheta_m", rk_step)
        values_rho = self.driving_values("rho_zz", rk_step)
        fifty_dt = np.float32(np.float32(50.0) * np.float32(self.outer_dt))
        ten_dt = np.float32(np.float32(10.0) * np.float32(self.outer_dt))
        relax_zone = np.float32(N_RELAX_ZONE)
        mesh = self.atmosphere.mesh
        self.kernels.launch(
            "regional_relaxzone_rayleigh_cell_v841",
            nlev * masks.n_relax_cells,
            {"relax_cells": masks.relax_cells,
             "bdy_mask_cell": masks.bdy_mask_cell,
             "mesh_scaling_cell": self.mesh_scaling_cell,
             "rho_zz": rho_zz, "theta_m": theta_m,
             "rho_driving": values_rho, "rt_driving": values_rt,
             "tend_rho": tend_rho, "tend_rt": tend_rt},
            {"nlev": _i(nlev), "ncells": _i(ncells),
             "nrelax": _i(masks.n_relax_cells), "fifty_dt": _f(fifty_dt),
             "relax_zone": _f(relax_zone)})
        self.kernels.launch(
            "regional_relaxzone_rayleigh_edge_v841",
            nlev * masks.n_relax_edges,
            {"relax_edges": masks.relax_edges,
             "bdy_mask_edge": masks.bdy_mask_edge,
             "mesh_scaling_edge": self.mesh_scaling_edge, "ru": rho_u,
             "ru_driving": values_ru, "tend_ru": tend_ru},
            {"nlev": _i(nlev), "nedges": _i(nedges),
             "nrelax": _i(masks.n_relax_edges), "fifty_dt": _f(fifty_dt),
             "relax_zone": _f(relax_zone)})
        self.kernels.launch(
            "regional_relaxzone_filter_cell_v841",
            nlev * masks.n_relax_cells,
            {"relax_cells": masks.relax_cells,
             "bdy_mask_cell": masks.bdy_mask_cell,
             "mesh_scaling_cell": self.mesh_scaling_cell,
             "n_edges_on_cell": mesh.n_edges_on_cell,
             "edges_on_cell": mesh.edges_on_cell,
             "cells_on_edge": mesh.cells_on_edge,
             "edge_sign_on_cell": mesh.edge_sign_on_cell,
             "dv_edge": mesh.dv_edge, "inv_dc_edge": self.context.inv_dc_edge,
             "rho_zz": rho_zz, "theta_m": theta_m,
             "rho_driving": values_rho, "rt_driving": values_rt,
             "tend_rho": tend_rho, "tend_rt": tend_rt},
            {"nlev": _i(nlev), "ncells": _i(ncells), "nedges": _i(nedges),
             "max_edges": _i(mesh.max_edges),
             "nrelax": _i(masks.n_relax_cells), "ten_dt": _f(ten_dt),
             "relax_zone": _f(relax_zone)})
        self.kernels.launch(
            "regional_relaxzone_filter_edge_v841",
            nlev * masks.n_relax_edges,
            {"relax_edges": masks.relax_edges,
             "bdy_mask_edge": masks.bdy_mask_edge,
             "mesh_scaling_edge": self.mesh_scaling_edge,
             "n_edges_on_cell": mesh.n_edges_on_cell,
             "edges_on_cell": mesh.edges_on_cell,
             "cells_on_edge": mesh.cells_on_edge,
             "vertices_on_edge": mesh.vertices_on_edge,
             "edges_on_vertex": mesh.edges_on_vertex,
             "edge_sign_on_cell": mesh.edge_sign_on_cell,
             "edge_sign_on_vertex": self.edge_sign_on_vertex,
             "dc_edge": mesh.dc_edge, "dv_edge": mesh.dv_edge,
             "inv_dc_edge": self.context.inv_dc_edge,
             "inv_dv_edge": self.context.inv_dv_edge,
             "inv_area_cell": self.context.inv_area_cell,
             "inv_area_triangle": self.context.inv_area_triangle,
             "ru": rho_u, "ru_driving": values_ru, "tend_ru": tend_ru},
            {"nlev": _i(nlev), "ncells": _i(ncells), "nedges": _i(nedges),
             "nvertices": _i(mesh.n_vertices), "max_edges": _i(mesh.max_edges),
             "vertex_degree": _i(mesh.vertex_degree),
             "nrelax": _i(masks.n_relax_edges), "ten_dt": _f(ten_dt),
             "divdamp": _f(self.config_relax_zone_divdamp_coef),
             "relax_zone": _f(relax_zone)})

    # -- graft: the specified-zone velocity overwrite ----------------------

    def overwrite_speczone_rho_u(self, rho_u: Any, rk_step: int) -> Any:
        """``atm_srk3:2442-2485``, the ``ru`` half (``u`` goes to a sink)."""

        values_u = self.driving_values("u", rk_step)
        values_ru = self.driving_values("ru", rk_step)
        masks = self.masks
        self.kernels.launch(
            "regional_speczone_u_ru_v841", self.nlev * masks.n_spec_edges,
            {"spec_edges": masks.spec_edges, "u_driving": values_u,
             "ru_driving": values_ru, "normal_velocity": self._u_sink(rho_u),
             "rho_u": rho_u},
            {"nlev": _i(self.nlev), "nedges": _i(int(rho_u.shape[-1])),
             "nspec": _i(masks.n_spec_edges)})
        return values_u

    def _u_sink(self, like: Any) -> Any:
        sink = self._u_scratch
        if sink is None or sink.shape != like.shape:
            sink = self._module.zeros(like.shape, np.float32)
            self._u_scratch = sink
        return sink

    def overwrite_speczone_u(self, normal_velocity: Any, values_u: Any) -> None:
        """``atm_srk3:2442-2485``, the ``u`` half, after recover."""

        masks = self.masks
        self.kernels.launch(
            "regional_speczone_u_ru_v841", self.nlev * masks.n_spec_edges,
            {"spec_edges": masks.spec_edges, "u_driving": values_u,
             "ru_driving": values_u, "normal_velocity": normal_velocity,
             "rho_u": self._u_sink(normal_velocity)},
            {"nlev": _i(self.nlev),
             "nedges": _i(int(normal_velocity.shape[-1])),
             "nspec": _i(masks.n_spec_edges)})

    def zero_speczone_w(self, vertical_velocity: Any) -> None:
        """``atm_zero_gradient_w_bdy`` (F:7868-7902)."""

        masks = self.masks
        self.kernels.launch(
            "regional_zero_speczone_w_v841",
            (self.nlev + 1) * masks.n_spec_cells,
            {"spec_cells": masks.spec_cells, "w": vertical_velocity},
            {"nrows": _i(self.nlev + 1),
             "ncells": _i(int(vertical_velocity.shape[-1])),
             "nspec": _i(masks.n_spec_cells)})

    # -- graft: the scalar boundary stages ---------------------------------

    def bdy_adjust_scalars(self, scalars: Any, rk_step: int) -> None:
        """``atm_srk3:2688-2717``."""

        dt_rk = transport_rk_timestep_f32(
            outer_dt=self.outer_dt, rk_step=int(rk_step))
        driving = self.driving.state_at("scalars", self.clock, dt_rk)
        masks = self.masks
        mesh = self.atmosphere.mesh
        ntracers = self._driven_tracer_count(scalars)
        ncells = int(scalars.shape[-1])
        nedges = int(mesh.n_edges)
        ten_dt = np.float32(np.float32(10.0) * np.float32(self.outer_dt))
        # The kernels index tracer planes by the leading axis, so the driven
        # species are the leading slice of the model's scalar block.
        driven = scalars[0:ntracers]
        updates = self._module.zeros(driven.shape, np.float32)
        self.kernels.launch(
            "regional_bdy_adjust_scalars_compute_v841",
            ntracers * self.nlev * (masks.n_relax_cells + masks.n_spec_cells),
            {"relax_cells": masks.relax_cells, "spec_cells": masks.spec_cells,
             "bdy_mask_cell": masks.bdy_mask_cell,
             "mesh_scaling_cell": self.mesh_scaling_cell,
             "n_edges_on_cell": mesh.n_edges_on_cell,
             "edges_on_cell": mesh.edges_on_cell,
             "cells_on_edge": mesh.cells_on_edge,
             "edge_sign_on_cell": mesh.edge_sign_on_cell,
             "dv_edge": mesh.dv_edge, "inv_dc_edge": self.context.inv_dc_edge,
             "scalars_new": driven, "scalars_driving": driving,
             "updates": updates},
            {"ntracers": _i(ntracers), "nlev": _i(self.nlev),
             "ncells": _i(ncells), "nedges": _i(nedges),
             "max_edges": _i(mesh.max_edges),
             "nrelax": _i(masks.n_relax_cells),
             "nspec": _i(masks.n_spec_cells), "ten_dt": _f(ten_dt),
             "dt_rk": _f(dt_rk), "relax_zone": _f(N_RELAX_ZONE)})
        self.kernels.launch(
            "regional_bdy_adjust_scalars_copyback_v841",
            ntracers * self.nlev * masks.n_nudged_cells,
            {"nudged_cells": masks.nudged_cells, "updates": updates,
             "scalars_new": driven},
            {"ntracers": _i(ntracers), "nlev": _i(self.nlev),
             "ncells": _i(ncells), "nnudged": _i(masks.n_nudged_cells)})

    def clamp_negative_scalars(self, scalars: Any) -> None:
        """``atm_srk3:2798-2800``."""

        self.kernels.launch(
            "regional_clamp_negative_scalars_v841", int(scalars.size),
            {"scalars": scalars}, {"nvalues": _i(scalars.size)})

    # -- graft: the end-of-step resets -------------------------------------

    def reset_speczone_values(
        self,
        *,
        theta_m: Any,
        rho_theta: Any,
        rho_theta_perturbation: Any,
        scalars: Any,
    ) -> None:
        """``atm_srk3:2828-2878`` plus the F:8238 perturbation write."""

        dt_f32 = np.float32(self.outer_dt)
        rt_values = self.driving.state_at("rtheta_m", self.clock, dt_f32)
        rho_values = self.driving.state_at("rho_zz", self.clock, dt_f32)
        masks = self.masks
        ncells = int(theta_m.shape[-1])
        self.kernels.launch(
            "regional_reset_speczone_values_v841",
            self.nlev * masks.n_spec_cells,
            {"spec_cells": masks.spec_cells, "rt_driving": rt_values,
             "rho_driving": rho_values,
             "rho_theta_base": self.atmosphere.reference.rho_theta_base,
             "theta_m": theta_m, "rho_theta": rho_theta,
             "rho_theta_perturbation": rho_theta_perturbation},
            {"nlev": _i(self.nlev), "ncells": _i(ncells),
             "nspec": _i(masks.n_spec_cells)})
        driving = self.driving.state_at("scalars", self.clock, dt_f32)
        ntracers = self._driven_tracer_count(scalars)
        self.kernels.launch(
            "regional_bdy_set_scalars_v841",
            ntracers * self.nlev * masks.n_spec_cells,
            {"spec_cells": masks.spec_cells, "scalars_driving": driving,
             "scalars_new": scalars[0:ntracers]},
            {"ntracers": _i(ntracers), "nlev": _i(self.nlev),
             "ncells": _i(int(scalars.shape[-1])),
             "nspec": _i(masks.n_spec_cells)})

    def _driven_tracer_count(self, scalars: Any) -> int:
        """How many leading species the boundary stream actually drives."""

        model = int(scalars.shape[0])
        if self.driven_scalars > model:
            raise RegionalAdmissionError(
                f"the lateral-boundary stream drives {self.driven_scalars} "
                f"scalar species and this run integrates {model}; the extra "
                "driving fields belong to species no scheme in this "
                "configuration carries")
        return self.driven_scalars

    def moist_coefficients(self, scalars: Any) -> MetalMoistDynamicsCoefficients:
        """``atm_compute_moist_coefficients`` (F:3188-3283) for one species.

        The CUDA runtime forms qtot/cqw/cqu with CuPy elementwise float32
        operations; these two shaders form them with the same operations in
        the same order.  The edge gather runs through the remapped
        connectivity on the padded block, so a one-cell ring-7 edge reads
        the garbage column's zero.
        """

        q = scalars[0]
        nlev, ncells = map(int, q.shape)
        coe = self.cells_on_edge
        nedges = int(coe.shape[0])
        qtot = self._module.empty((nlev, ncells), np.float32)
        cqw = self._module.empty((nlev, ncells), np.float32)
        cqu = self._module.empty((nlev, nedges), np.float32)
        self.kernels.launch(
            "regional_moist_cell_coefficients", nlev * ncells,
            {"q": q, "qtot": qtot, "cqw": cqw},
            {"nlev": _i(nlev), "ncells": _i(ncells)})
        self.kernels.launch(
            "regional_moist_edge_coefficients", nlev * nedges,
            {"q": q, "cells_on_edge": coe, "cqu": cqu},
            {"nlev": _i(nlev), "ncells": _i(ncells), "nedges": _i(nedges)})
        return MetalMoistDynamicsCoefficients(qtot=qtot, cqw=cqw, cqu=cqu)

    def validate_recovered(self, state: Any, saved: Any, flag: Any) -> None:
        """``validate_recovered_v841_f32``'s contract over the SOLVE region."""

        cells = self.n_cells_solve
        edges = self.n_edges_solve
        self.solve_region.validate(
            flag,
            positive=(
                (state.rho, cells),
                (state.rho_theta, cells),
                (saved.theta_m, cells),
                (saved.exner, cells),
            ),
            finite=(
                (saved.density_perturbation, cells),
                (saved.rho_theta_perturbation, cells),
                (saved.pressure_perturbation, cells),
                (state.rho_w, cells),
                (saved.vertical_velocity, cells),
                (state.rho_u, edges),
                (saved.normal_velocity, edges),
            ))

    def validate_density(self, density: Any, flag: Any) -> None:
        """``_check_density``'s contract, over the SOLVE region."""

        self.solve_region.validate(flag, positive=((density, self.n_cells_solve),))

    def history_slice(self, array: Any) -> np.ndarray:
        """Read a field back with the garbage element stripped (host copy)."""

        extent = int(array.shape[-1])
        host = array.to_numpy()
        if extent == self.n_cells_solve + 1:
            return host[..., : self.n_cells_solve]
        if extent == self.n_edges_solve + 1:
            return host[..., : self.n_edges_solve]
        return host

    def receipt(self) -> dict[str, Any]:
        return {
            "n_cells_solve": self.n_cells_solve,
            "n_edges_solve": self.n_edges_solve,
            "spec_cells": int(self.masks.n_spec_cells),
            "spec_edges": int(self.masks.n_spec_edges),
            "relax_cells": int(self.masks.n_relax_cells),
            "relax_edges": int(self.masks.n_relax_edges),
            "nudged_cells": int(self.masks.n_nudged_cells),
            "acoustic_one_cell_edge_flag": int(
                self.acoustic_invalid.to_numpy()[0]),
            "garbage": self.discipline.receipt(),
        }


__all__ = [
    "MetalMoistDynamicsCoefficients",
    "MetalRegionalDrivingState",
    "MetalRegionalGarbageDiscipline",
    "MetalRegionalKernels",
    "MetalRegionalMasks",
    "MetalRegionalRuntimeV841",
    "MetalSolveRegionKernels",
    "PaddedMetalRegionalDrivingState",
    "advance_acoustic_step_regional_metal_v841",
    "advance_scalars_regional_metal_v841",
    "build_regional_metal_atmosphere",
    "describe_field",
    "regional_padded_metal_context_v841",
]
