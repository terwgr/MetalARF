"""Metal mirror of ``cuda_physics_prep_v841``: MPAS state to physics columns.

Kernels (one shader each, ``hexv841_<cuda_name>``): ``prep_mass``,
``reconstruct_winds``, ``prep_noahmp_sounding``, ``prep_interfaces`` and
``validate_prep``.  The CPU oracles, constants and result field inventory are
the CUDA module's own; this module supplies the resident Metal flow.

Differences from the CUDA module, all forced by Metal:

* CUDA's device pointer table of per-species outputs (``float *const *qout``
  in ``prep_mass``, ``const float *const *qspecies`` in ``validate_prep``)
  becomes ONE stacked buffer ``[nspecies, nlev, ncells]`` in model species
  order (slot ``s`` is table entry ``s``).  The public per-species fields
  ``species_p[name]`` are leading-axis views of that stack.
* ``kernel_cache`` is dropped; the context comes with the geometry upload.
* ``regional_garbage_cell``'s lend-and-restore of one column runs through
  the ``hexv841_copy_column_v841`` kernel, since ``MetalArray`` cannot index
  ``array[..., column]``.
* ``reconstruct_winds`` uses ``precise::cos/sin`` and ``prep_interfaces`` /
  ``validate_prep`` use ``precise::exp/log`` in place of CUDA ``cosf, sinf,
  expf, logf``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import time
from types import MappingProxyType
from typing import Any

import numpy as np

from .cuda_backend.containers import TransferStats
from .cuda_physics_prep_v841 import (
    CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256,
    CudaMpasToPhysColumnsV841,
    CudaMpasToPhysGeometryV841,
    CudaNoahmpSoundingV841,
    CudaWsm6InputViewV841,
    GRAVITY_F32,
    RV_OVER_RD_F32,
    WSM6_SCALAR_NAMES,
    _GEOMETRY_SEAL,
    _INTERFACE_FIELDS,
    _MASS_NONSPECIES_FIELDS,
    _PREP_STRICTLY_POSITIVE,
    _SURFACE_FIELDS,
    _array_sha256,
    _canonical_coefficients,
    _canonical_edges_on_cell,
    _mass_fields,
    _mesh_value,
    _validate_scalar_names,
)
from .errors import ConfigurationRefusal
from .metal_dynamics_v841 import _arrays, _context, _i, _launch, _require

_NOAHMP_NAMES = ("dz8w", "qv_curr", "t_phy", "u_phy", "v_phy", "p8w")


@dataclass(frozen=True, slots=True)
class MetalMpasToPhysGeometryV841(CudaMpasToPhysGeometryV841):
    """One-time, source-checked upload of reconstruction geometry."""

    @classmethod
    def from_host(cls, context: Any, mesh: object) -> "MetalMpasToPhysGeometryV841":
        """Validate and upload the exact spherical reconstruction carriers."""

        lat = np.asarray(_mesh_value(mesh, "latCell"))
        lon = np.asarray(_mesh_value(mesh, "lonCell"))
        if lat.ndim != 1 or lon.shape != lat.shape or lat.size == 0:
            raise ValueError("latCell/lonCell must be matching non-empty vectors")
        n_cells = int(lat.size)
        counts_raw = np.asarray(_mesh_value(mesh, "nEdgesOnCell"))
        if counts_raw.shape != (n_cells,) or counts_raw.dtype.kind not in "iu":
            raise ValueError("nEdgesOnCell must be an integer nCells vector")
        counts64 = np.asarray(counts_raw, dtype=np.int64)

        raw_edges = np.asarray(_mesh_value(mesh, "edgesOnCell"))
        if raw_edges.ndim != 2 or n_cells not in raw_edges.shape:
            raise ValueError("edgesOnCell geometry is missing or has invalid rank")
        max_edges = int(
            raw_edges.shape[1] if raw_edges.shape[0] == n_cells else raw_edges.shape[0]
        )
        if max_edges <= 0 or np.any(counts64 < 0) or np.any(counts64 > max_edges):
            raise ValueError("nEdgesOnCell lies outside the stored edge-slot extent")
        counts = np.ascontiguousarray(counts64, dtype=np.int32)
        edges = _canonical_edges_on_cell(
            raw_edges, n_cells=n_cells, max_edges=max_edges
        )

        raw_cells_on_edge = np.asarray(_mesh_value(mesh, "cellsOnEdge"))
        if raw_cells_on_edge.ndim != 2 or 2 not in raw_cells_on_edge.shape:
            raise ValueError("cellsOnEdge is required to bind nEdges")
        n_edges = int(
            raw_cells_on_edge.shape[0]
            if raw_cells_on_edge.shape[1] == 2
            else raw_cells_on_edge.shape[1]
        )
        if n_edges <= 0:
            raise ValueError("reconstruction geometry must contain edges")
        slots = np.arange(max_edges, dtype=np.int64)[None, :]
        active = slots < counts64[:, None]
        if np.any(edges[active] < 0) or np.any(edges[active] >= n_edges):
            raise ValueError("active edgesOnCell entries must be zero-based in range")
        if np.any(edges[~active] != -1):
            raise ValueError("edgesOnCell padding must be canonical -1")

        coeffs = _canonical_coefficients(
            _mesh_value(mesh, "coeffs_reconstruct"),
            n_cells=n_cells,
            max_edges=max_edges,
        )
        if not np.any(coeffs[active]):
            raise ValueError("active coeffs_reconstruct entries are all zero")
        if np.any(coeffs[~active] != np.float32(0.0)):
            raise ValueError("coeffs_reconstruct padding must be bitwise zero")

        lat32 = np.ascontiguousarray(lat, dtype=np.float32)
        lon32 = np.ascontiguousarray(lon, dtype=np.float32)
        if not np.all(np.isfinite(lat32)) or not np.all(np.isfinite(lon32)):
            raise ValueError("latCell/lonCell contains non-finite values")
        if np.any(np.abs(lat32) > np.float32(np.pi / 2.0 + 1.0e-5)):
            raise ValueError("latCell must be in radians")
        on_sphere = _mesh_value(mesh, "on_a_sphere", True)
        if isinstance(on_sphere, str):
            on_sphere = on_sphere.strip().lower() in {"true", "t", "yes", "1"}
        if not bool(on_sphere):
            raise ValueError("the v8.4.1 physics seam requires spherical MPAS")

        canonical = {
            "edges_on_cell": edges,
            "n_edges_on_cell": counts,
            "coeffs_reconstruct": coeffs,
            "lat_cell": lat32,
            "lon_cell": lon32,
        }
        hashes = MappingProxyType(
            {name: _array_sha256(value) for name, value in canonical.items()}
        )
        garbage_value = _mesh_value(mesh, "garbage_cell")
        garbage_cell = None if garbage_value is None else int(garbage_value)
        if garbage_cell is not None and garbage_cell != n_cells - 1:
            raise ValueError(
                "a limited-area mesh's garbage element is the LAST column of "
                f"its padded extent; this view declares {garbage_cell} with "
                f"{n_cells} columns"
            )
        metal = _arrays(context)
        started = time.perf_counter()
        live = {f"_{name}": metal.asarray(value) for name, value in canonical.items()}
        sealed = {
            f"_sealed_{name}": metal.asarray(value)
            for name, value in canonical.items()
        }
        elapsed = time.perf_counter() - started
        sealed_bytes = int(sum(int(value.nbytes) for value in sealed.values()))
        result = cls(
            **live,
            **sealed,
            n_cells=n_cells,
            n_edges=n_edges,
            max_edges=max_edges,
            sealed_device_bytes=sealed_bytes,
            h2d=TransferStats(
                int(sum(int(value.nbytes) for value in (*live.values(), *sealed.values()))),
                elapsed,
            ),
            array_sha256=hashes,
            _seal=_GEOMETRY_SEAL,
            regional_garbage_cell=garbage_cell,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self._seal is not _GEOMETRY_SEAL:
            raise TypeError("geometry must be constructed by from_host")
        if self.n_cells <= 0 or self.n_edges <= 0 or self.max_edges <= 0:
            raise ValueError("geometry dimensions must be positive")
        if self.regional_garbage_cell is not None and (
            self.regional_garbage_cell != self.n_cells - 1
        ):
            raise ValueError(
                "the regional garbage column must be the last of the extent"
            )
        shapes = {
            "edges_on_cell": (self.n_cells, self.max_edges),
            "n_edges_on_cell": (self.n_cells,),
            "coeffs_reconstruct": (self.n_cells, self.max_edges, 3),
            "lat_cell": (self.n_cells,),
            "lon_cell": (self.n_cells,),
        }
        dtypes = {
            "edges_on_cell": np.int32,
            "n_edges_on_cell": np.int32,
            "coeffs_reconstruct": np.float32,
            "lat_cell": np.float32,
            "lon_cell": np.float32,
        }
        sealed_bytes = 0
        for name in shapes:
            live = getattr(self, f"_{name}")
            sealed = getattr(self, f"_sealed_{name}")
            _require(f"physics_prep_geometry.live.{name}", live, dtypes[name],
                     shapes[name])
            _require(f"physics_prep_geometry.sealed.{name}", sealed, dtypes[name],
                     shapes[name])
            if live.buffer is sealed.buffer:
                raise ValueError("live and sealed geometry buffers must not alias")
            sealed_bytes += int(sealed.nbytes)
        if self.sealed_device_bytes != sealed_bytes:
            raise ValueError("sealed geometry byte inventory changed")
        if self.h2d.bytes != 2 * sealed_bytes:
            raise ValueError("geometry upload must contain live and sealed copies")
        expected = set(shapes)
        if set(self.array_sha256) != expected or any(
            len(str(value)) != 64 for value in self.array_sha256.values()
        ):
            raise ValueError("geometry array hash inventory is incomplete")


@dataclass(frozen=True, slots=True)
class MetalNoahmpSoundingV841(CudaNoahmpSoundingV841):
    """Separate released NoahMP sounding; raw qv is part of its identity."""

    def validate(self, *, n_vert_levels: int, n_cells: int) -> None:
        shape = (n_vert_levels, n_cells)
        for name in _NOAHMP_NAMES:
            _require(f"noahmp_sounding.{name}", getattr(self, name), np.float32, shape)


@dataclass(frozen=True, slots=True)
class MetalWsm6InputViewV841(CudaWsm6InputViewV841):
    """Dry post-RK WSM6 view, valid only after candidate scalar clamping."""

    def validate(self) -> None:
        shape = tuple(self.rho_dry.shape)
        if len(shape) != 2 or shape[0] < 2 or shape[1] <= 0:
            raise ValueError("WSM6 view must be [level,cell]")
        for name in (
            "rho_dry", "theta_dry", "exner", "p_eos", "dz", "z", "w",
            *self.species,
        ):
            _require(f"wsm6_input.{name}", getattr(self, name), np.float32, shape)


@dataclass(frozen=True, slots=True)
class MetalMpasToPhysColumnsV841(CudaMpasToPhysColumnsV841):
    """Resident output using the exact native MPAS physics field names."""

    def wsm6_input_view(self) -> MetalWsm6InputViewV841:
        if not self.wsm6_ready:
            raise ValueError(
                "WSM6 input view requires post_rk_wsm6=True after scalar clamp"
            )
        result = MetalWsm6InputViewV841(
            rho_dry=self.rho_dry,
            theta_dry=self.th_p,
            exner=self.pi_p,
            p_eos=self.pres_p,
            dz=self.dz_p,
            z=self.z_p[:-1],
            w=self.w_p[:-1],
            species={
                name: self._source_scalars[index]
                for index, name in enumerate(self.species_p)
            },
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not math.isfinite(float(self.time_seconds)) or self.time_seconds < 0.0:
            raise ValueError("time_seconds must be finite and non-negative")
        nlev, ncells = tuple(self.rho_p.shape)
        if nlev < 2 or ncells <= 0:
            raise ValueError("prepared physics columns need at least two levels")
        for name in _mass_fields(self.species_p):
            _require(f"physics_prep.{name}", getattr(self, name), np.float32,
                     (nlev, ncells))
        for name in _INTERFACE_FIELDS:
            _require(f"physics_prep.{name}", getattr(self, name), np.float32,
                     (nlev + 1, ncells))
        for name in _SURFACE_FIELDS:
            _require(f"physics_prep.{name}", getattr(self, name), np.float32,
                     (ncells,))
        _require("physics_prep.source_scalars", self._source_scalars, np.float32,
                 (len(self.species_p), nlev, ncells))
        self.noahmp_sounding.validate(n_vert_levels=nlev, n_cells=ncells)
        if not isinstance(self.wsm6_ready, bool):
            raise TypeError("wsm6_ready must be bool")
        if self.wsm6_ready:
            self.wsm6_input_view().validate()
        if self.validation_d2h.bytes != 4:
            raise ValueError("physics preparation validation must transfer exactly 4 bytes")
        if self.geometry_receipt.get("contract_sha256") != self.contract_sha256:
            raise ValueError("prepared columns are not bound to their geometry contract")


def _diagnose_prep_failure(
    output: Mapping[str, Any], noahmp_output: Mapping[str, Any], *, n_cells: int,
) -> str:
    """Name the field and column the preparation validation refused."""

    findings: list[str] = []
    for carrier in (output, noahmp_output):
        for name, array in carrier.items():
            host = array.to_numpy()
            bad = ~np.isfinite(host)
            if name in _PREP_STRICTLY_POSITIVE:
                bad = bad | (host <= np.float32(0.0))
            count = int(np.count_nonzero(bad))
            if count == 0:
                continue
            columns = np.unique(np.argwhere(bad)[:, -1])
            findings.append(
                f"{name}: {count} value(s) at {columns.size} column(s), "
                f"first column {int(columns[0])}, "
                f"range [{float(np.nanmin(host))}, {float(np.nanmax(host))}]"
            )
    if not findings:
        return (
            "every prepared field is finite and positive on the host, so the "
            "kernel refused a geometry or consistency law rather than a value "
            "-- read validate_prep_v841_f32 for the exact test"
        )
    return "; ".join(findings)


def _copy_column(context: Any, src: Any, dst: Any, rows: int, src_stride: int,
                 src_col: int, dst_stride: int, dst_col: int) -> None:
    """``dst[r, dst_col] = src[r, src_col]`` over the flattened leading rows."""

    _launch("copy_column_v841", rows, context,
            dict(src=src.view((int(src.size),)), dst=dst.view((int(dst.size),))),
            dict(rows=_i(rows), src_stride=_i(src_stride), src_col=_i(src_col),
                 dst_stride=_i(dst_stride), dst_col=_i(dst_col)))


def prepare_mpas_to_phys_metal_v841(
    atmosphere: Any,
    *,
    scalar_names: Sequence[str],
    geometry: MetalMpasToPhysGeometryV841,
    post_rk_wsm6: bool = False,
    regional_garbage_cell: int | None = None,
) -> MetalMpasToPhysColumnsV841:
    """Prepare one resident MPAS state for the aggregate column backend."""

    if type(post_rk_wsm6) is not bool:
        raise TypeError("post_rk_wsm6 must be bool")
    if geometry is None or not isinstance(geometry, MetalMpasToPhysGeometryV841):
        raise TypeError("a validated MetalMpasToPhysGeometryV841 is required")
    geometry.validate()
    if hasattr(atmosphere, "validate"):
        atmosphere.validate()
    state = atmosphere.state
    saved = atmosphere.saved
    vertical = atmosphere.vertical
    reference = atmosphere.reference
    nlev, ncells = tuple(state.rho.shape)
    nedges = int(saved.normal_velocity.shape[1])
    if nlev < 2:
        raise ValueError("MPAS_to_physics requires at least two vertical levels")
    if (ncells, nedges) != (geometry.n_cells, geometry.n_edges):
        raise ValueError(
            "preparation geometry/state mismatch: "
            f"state={(ncells, nedges)}, geometry={(geometry.n_cells, geometry.n_edges)}"
        )
    species_names = _validate_scalar_names(
        scalar_names, int(state.scalars.shape[0]))

    mass_shape = (nlev, ncells)
    interface_shape = (nlev + 1, ncells)
    _require("state.rho", state.rho, np.float32, mass_shape)
    _require("state.scalars", state.scalars, np.float32,
             (len(species_names), nlev, ncells))
    for name, value in (
        ("saved.theta_m", saved.theta_m),
        ("saved.exner", saved.exner),
        ("saved.pressure_perturbation", saved.pressure_perturbation),
        ("vertical.zz", vertical.zz),
        ("reference.pressure_base", reference.pressure_base),
    ):
        _require(name, value, np.float32, mass_shape)
    _require("saved.normal_velocity", saved.normal_velocity, np.float32,
             (nlev, nedges))
    _require("saved.vertical_velocity", saved.vertical_velocity, np.float32,
             interface_shape)
    _require("vertical.zgrid", vertical.zgrid, np.float32, interface_shape)

    if regional_garbage_cell is None:
        regional_garbage_cell = geometry.regional_garbage_cell
    # The limited-area garbage column, lent a real column for the duration of
    # the preparation and given back after it, as on CUDA: the dycore's pool
    # values there are not a column, and the preparation's validation refuses
    # them.  Nothing real reads the garbage column's physics.
    lent: list[tuple[Any, Any, int, int]] = []
    if regional_garbage_cell is not None:
        garbage = int(regional_garbage_cell)
        if garbage != ncells - 1:
            raise ValueError(
                "the regional garbage column is the LAST element of the "
                f"padded extent; got {garbage} with {ncells} columns")
        lend_context = _context(state.rho)
        for source in (state.rho, state.scalars, saved.theta_m, saved.exner,
                       saved.pressure_perturbation, saved.vertical_velocity,
                       reference.pressure_base, vertical.zz, vertical.zgrid):
            rows = int(np.prod(source.shape[:-1]))
            kept = _arrays(lend_context).empty((rows,), np.float32)
            _copy_column(lend_context, source, kept, rows, ncells, garbage, 1, 0)
            lent.append((source, kept, rows, garbage))
            _copy_column(lend_context, source, source, rows, ncells, garbage - 1,
                         ncells, garbage)
    try:
        mctx = _context(state.rho)
        arrays = _arrays(mctx)
        # The species outputs are one stack; table entry s of the CUDA qout /
        # qspecies tables is slot s here, in model species order.
        qstack = arrays.empty((len(species_names), nlev, ncells), np.float32)
        output: dict[str, Any] = {
            f"{name}_p": qstack[index] for index, name in enumerate(species_names)
        }
        output.update({
            name: arrays.empty(mass_shape, np.float32)
            for name in _MASS_NONSPECIES_FIELDS
        })
        output.update({
            name: arrays.empty(interface_shape, np.float32) for name in _INTERFACE_FIELDS
        })
        output.update({
            name: arrays.empty((ncells,), np.float32) for name in _SURFACE_FIELDS
        })
        noahmp_output = {
            name: arrays.empty(mass_shape, np.float32) for name in _NOAHMP_NAMES
        }
        reconstructed_x = arrays.empty(mass_shape, np.float32)
        reconstructed_y = arrays.empty(mass_shape, np.float32)
        reconstructed_z = arrays.empty(mass_shape, np.float32)
        invalid = arrays.zeros((1,), np.int32)

        _launch(
            "prep_mass_v841_f32", ncells, mctx,
            dict(rho_zz=state.rho, theta_m=saved.theta_m, scalars=state.scalars,
                 exner=saved.exner, pressure_base=reference.pressure_base,
                 pressure_p=saved.pressure_perturbation, zgrid=vertical.zgrid,
                 zz=vertical.zz, qout=qstack, zz_p=output["zz_p"],
                 rho_dry=output["rho_dry"], rho_p=output["rho_p"],
                 th_p=output["th_p"], t_p=output["t_p"], pi_p=output["pi_p"],
                 pres_p=output["pres_p"], zmid_p=output["zmid_p"],
                 dz_p=output["dz_p"], invalid=invalid),
            dict(nlev=_i(nlev), ncells=_i(ncells), nspecies=_i(len(species_names)),
                 rvord=RV_OVER_RD_F32),
        )
        _launch(
            "reconstruct_winds_v841_f32", ncells, mctx,
            dict(normal_velocity=saved.normal_velocity,
                 edges_on_cell=geometry._edges_on_cell,
                 n_edges_on_cell=geometry._n_edges_on_cell,
                 coeffs_reconstruct=geometry._coeffs_reconstruct,
                 lat_cell=geometry._lat_cell, lon_cell=geometry._lon_cell,
                 sealed_edges_on_cell=geometry._sealed_edges_on_cell,
                 sealed_n_edges_on_cell=geometry._sealed_n_edges_on_cell,
                 sealed_coeffs_reconstruct=geometry._sealed_coeffs_reconstruct,
                 sealed_lat_cell=geometry._sealed_lat_cell,
                 sealed_lon_cell=geometry._sealed_lon_cell,
                 x=reconstructed_x, y=reconstructed_y, z=reconstructed_z,
                 u_p=output["u_p"], v_p=output["v_p"], invalid=invalid),
            dict(nlev=_i(nlev), ncells=_i(ncells), nedges=_i(nedges),
                 maxedges=_i(geometry.max_edges)),
        )
        _launch(
            "prep_noahmp_sounding_v841_f32", ncells, mctx,
            dict(raw_qv=state.scalars, theta_m=saved.theta_m, exner=saved.exner,
                 pressure_base=reference.pressure_base,
                 pressure_p=saved.pressure_perturbation, zgrid=vertical.zgrid,
                 u_p=output["u_p"], v_p=output["v_p"],
                 dz8w=noahmp_output["dz8w"], qv_curr=noahmp_output["qv_curr"],
                 t_phy=noahmp_output["t_phy"], u_phy=noahmp_output["u_phy"],
                 v_phy=noahmp_output["v_phy"], p8w=noahmp_output["p8w"],
                 invalid=invalid),
            dict(nlev=_i(nlev), ncells=_i(ncells), rvord=RV_OVER_RD_F32),
        )
        _launch(
            "prep_interfaces_v841_f32", ncells, mctx,
            dict(rho_zz=state.rho, zz=vertical.zz, qv_p=output["qv_p"],
                 pres_p=output["pres_p"], t_p=output["t_p"], rho_p=output["rho_p"],
                 dz_p=output["dz_p"], zgrid=vertical.zgrid,
                 w=saved.vertical_velocity, w_p=output["w_p"], z_p=output["z_p"],
                 t2_p=output["t2_p"], pres2_p=output["pres2_p"],
                 pres_hyd_p=output["pres_hyd_p"], pres2_hyd_p=output["pres2_hyd_p"],
                 pres_hydd_p=output["pres_hydd_p"],
                 pres2_hydd_p=output["pres2_hydd_p"], psfc_p=output["psfc_p"],
                 psfc_hyd_p=output["psfc_hyd_p"], psfc_hydd_p=output["psfc_hydd_p"],
                 znu_p=output["znu_p"], znu_hyd_p=output["znu_hyd_p"],
                 plrad=output["plrad"], invalid=invalid),
            dict(nlev=_i(nlev), ncells=_i(ncells), gravity=GRAVITY_F32),
        )
        validate_arrays = dict(
            edges_on_cell=geometry._edges_on_cell,
            n_edges_on_cell=geometry._n_edges_on_cell,
            coeffs_reconstruct=geometry._coeffs_reconstruct,
            lat_cell=geometry._lat_cell, lon_cell=geometry._lon_cell,
            source_scalars=state.scalars, source_theta_m=saved.theta_m,
            source_exner=saved.exner, qspecies=qstack,
            **{name: output[name] for name in _MASS_NONSPECIES_FIELDS},
            **{name: output[name] for name in _INTERFACE_FIELDS},
            **{name: output[name] for name in _SURFACE_FIELDS},
            **{f"noah_{name}": noahmp_output[name] for name in _NOAHMP_NAMES},
            invalid=invalid,
        )
        _launch(
            "validate_prep_v841_f32", ncells, mctx, validate_arrays,
            dict(nlev=_i(nlev), ncells=_i(ncells), nedges=_i(nedges),
                 maxedges=_i(geometry.max_edges),
                 wsm6_ready=_i(1 if post_rk_wsm6 else 0), gravity=GRAVITY_F32,
                 rvord=RV_OVER_RD_F32, nspecies=_i(len(species_names))),
        )
        started = time.perf_counter()
        invalid_value = int(invalid.to_numpy()[0])
        validation = TransferStats(int(invalid.nbytes), time.perf_counter() - started)
        if invalid_value != 0:
            raise FloatingPointError(
                "MPAS_to_physics Metal preparation failed numeric/geometry "
                "validation: "
                + _diagnose_prep_failure(output, noahmp_output, n_cells=ncells)
            )
        species_outputs = frozenset(f"{name}_p" for name in species_names)
        result = MetalMpasToPhysColumnsV841(
            species_p={name: output[f"{name}_p"] for name in species_names},
            **{
                name: value for name, value in output.items()
                if name not in species_outputs
            },
            noahmp_sounding=MetalNoahmpSoundingV841(**noahmp_output),
            wsm6_ready=post_rk_wsm6,
            _source_scalars=state.scalars,
            time_seconds=float(state.time_seconds),
            validation_d2h=validation,
            geometry_receipt=MappingProxyType(geometry.receipt()),
        )
        result.validate()
        return result
    finally:
        for source, kept, rows, garbage in lent:
            _copy_column(_context(source), kept, source, rows, 1, 0, ncells, garbage)


__all__ = [
    "CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256",
    "GRAVITY_F32",
    "MetalMpasToPhysColumnsV841",
    "MetalMpasToPhysGeometryV841",
    "MetalNoahmpSoundingV841",
    "MetalWsm6InputViewV841",
    "RV_OVER_RD_F32",
    "WSM6_SCALAR_NAMES",
    "prepare_mpas_to_phys_metal_v841",
]
