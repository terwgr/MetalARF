"""Metal mirror of ``cuda_gwdo_v841``: YSU orographic gravity-wave drag.

One shader, ``hexv841_bl_ysu_gwdo_v841_f32``, transcribes the CUDA kernel.  It
keeps every weld of the CUDA kernel: ``GWDO_NLEV`` 55, ``dt`` 120 s and an
effective ``len_disp`` of 25000 m.  CUDA ``powf`` is ``fp32_pressure_power``
(binary64 evaluation, one rounding); ``atan2f`` and ``floorf`` are
``precise::atan2`` and ``floor``.  The CPU oracle, constants and native
``dx`` helper are the CUDA module's own.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .cuda_backend.containers import TransferStats
from .cuda_gwdo_v841 import (
    CP_F32,
    CudaYsuGwdoColumnViewV841,
    CudaYsuGwdoResultV841,
    CudaYsuGwdoStaticV841,
    EP1_F32,
    GRAVITY_F32,
    PI_F32,
    RD_F32,
    X4_N_VERT_LEVELS,
    _STATIC_FIELDS,
    _STATIC_SEAL,
    _carrier_value,
    _host_f32_vector,
)
from . import cuda_gwdo_v841 as _cuda_gwdo
from .metal_dynamics_v841 import _arrays, _context, _i, _launch, _require
from .cuda_physics_prep_v841 import CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256


@dataclass(frozen=True, slots=True)
class MetalYsuGwdoColumnViewV841(CudaYsuGwdoColumnViewV841):
    """Exact GWD aliases from a validated ``MPAS_to_physics`` result."""

    def validate(self) -> None:
        from .cuda_gwdo_v841 import _COLUMN_VIEW_SEAL
        import math

        if self._seal is not _COLUMN_VIEW_SEAL:
            raise TypeError("GWD column view must be constructed by from_prepared")
        if self.prep_contract_sha256 != CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256:
            raise ValueError("GWD column view is not bound to v8.4.1 physics prep")
        if not math.isfinite(self.time_seconds) or self.time_seconds < 0.0:
            raise ValueError("GWD time_seconds must be finite and non-negative")
        shape = tuple(self.u_p.shape)
        if shape[0] != X4_N_VERT_LEVELS or shape[1] <= 0:
            raise ValueError(
                f"exact x4 YSU-GWDO needs [{X4_N_VERT_LEVELS},nCells]"
            )
        for name in ("u_p", "v_p", "t_p", "qv_p", "pres_hydd_p", "pi_p", "zmid_p"):
            _require(f"gwdo_columns.{name}", getattr(self, name), np.float32, shape)
        _require("gwdo_columns.pres2_hydd_p", self.pres2_hydd_p, np.float32,
                 (X4_N_VERT_LEVELS + 1, shape[1]))


@dataclass(frozen=True, slots=True)
class MetalYsuGwdoStaticV841(CudaYsuGwdoStaticV841):
    """Resident x4 mesh/orography carriers used by every GWD call."""

    @classmethod
    def from_host(
        cls, context: Any, carriers: object, *, config_len_disp_m: float = 0.0
    ) -> "MetalYsuGwdoStaticV841":
        """Validate and upload official x4 static carriers exactly once."""

        mesh_density = _host_f32_vector(
            _carrier_value(carriers, "meshDensity"), name="meshDensity"
        )
        n_cells = int(mesh_density.size)
        host: dict[str, np.ndarray] = {"mesh_density": mesh_density}
        for name in _STATIC_FIELDS[1:]:
            host[name] = _host_f32_vector(
                _carrier_value(carriers, name), name=name, count=n_cells
            )
        configured = np.float32(config_len_disp_m)
        if configured > np.float32(0.0):
            effective = configured
        else:
            nominal = np.asarray(_carrier_value(carriers, "nominalMinDc"))
            if nominal.dtype != np.float32 or nominal.shape != ():
                raise TypeError("nominalMinDc must be a host FP32 scalar")
            effective = np.float32(nominal)
        metal = _arrays(context)
        started = time.perf_counter()
        device = {name: metal.asarray(value) for name, value in host.items()}
        elapsed = time.perf_counter() - started
        result = cls(
            **device,
            effective_length_scale_m=float(effective),
            n_cells=n_cells,
            h2d=TransferStats(
                int(sum(int(value.nbytes) for value in device.values())), elapsed
            ),
            _seal=_STATIC_SEAL,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self._seal is not _STATIC_SEAL:
            raise TypeError("GWD static carriers require from_host/from_resident")
        if self.n_cells <= 0:
            raise ValueError("GWD static carriers require at least one cell")
        for name in _STATIC_FIELDS:
            _require(f"gwdo_static.{name}", getattr(self, name), np.float32,
                     (self.n_cells,))
        if np.float32(self.effective_length_scale_m).view(np.uint32) != (
            _cuda_gwdo.X4_NOMINAL_MIN_DC_M_F32.view(np.uint32)
        ):
            raise ValueError("exact x4 GWD requires effective len_disp=25000 m")


@dataclass(frozen=True, slots=True)
class MetalYsuGwdoResultV841(CudaYsuGwdoResultV841):
    """Transactional resident GWD/PBL tendency and diagnostic candidates."""

    def validate(self, *, n_vert_levels: int, n_cells: int) -> None:
        shape = (n_vert_levels, n_cells)
        for name in ("rublten", "rvblten", "dtaux3d", "dtauy3d", "rubldiff",
                     "rvbldiff"):
            _require(f"gwdo_result.{name}", getattr(self, name), np.float32, shape)
        for name in ("dusfcg", "dvsfcg"):
            _require(f"gwdo_result.{name}", getattr(self, name), np.float32,
                     (n_cells,))
        if self.validation_d2h.bytes != 4:
            raise ValueError("GWD validation must transfer exactly four bytes")


def run_bl_ysu_gwdo_metal_v841(
    columns: CudaYsuGwdoColumnViewV841,
    *,
    rublten: Any,
    rvblten: Any,
    static: CudaYsuGwdoStaticV841,
    dt_seconds: float,
) -> MetalYsuGwdoResultV841:
    """Run complete native YSU-GWDO into transactional resident candidates."""

    if not isinstance(columns, MetalYsuGwdoColumnViewV841):
        raise TypeError("columns must be a sealed MetalYsuGwdoColumnViewV841")
    if not isinstance(static, MetalYsuGwdoStaticV841):
        raise TypeError("static must be sealed MetalYsuGwdoStaticV841")
    columns.validate()
    static.validate()
    nlev = columns.n_vert_levels
    ncells = columns.n_cells
    if static.n_cells != ncells:
        raise ValueError(
            f"GWD static carriers have {static.n_cells} cells, columns have {ncells}"
        )
    shape = (nlev, ncells)
    _require("gwdo.rublten", rublten, np.float32, shape)
    _require("gwdo.rvblten", rvblten, np.float32, shape)
    dt = np.float32(dt_seconds)
    if dt.view(np.uint32) != _cuda_gwdo.X4_DT_SECONDS_F32.view(np.uint32):
        raise ValueError("exact x4 YSU-GWDO requires dt_seconds=120")
    length_scale = np.float32(static.effective_length_scale_m)

    mctx = _context(columns.u_p)
    arrays = _arrays(mctx)
    output = {
        "rublten": arrays.empty(shape, np.float32),
        "rvblten": arrays.empty(shape, np.float32),
        "dtaux3d": arrays.empty(shape, np.float32),
        "dtauy3d": arrays.empty(shape, np.float32),
        "dusfcg": arrays.empty((ncells,), np.float32),
        "dvsfcg": arrays.empty((ncells,), np.float32),
        "rubldiff": arrays.empty(shape, np.float32),
        "rvbldiff": arrays.empty(shape, np.float32),
    }
    invalid = arrays.zeros((1,), np.int32)
    _launch(
        "bl_ysu_gwdo_v841_f32", ncells, mctx,
        dict(u=columns.u_p, v=columns.v_p, temperature=columns.t_p,
             qv=columns.qv_p, p_mass_dry_hyd=columns.pres_hydd_p,
             p_interface_dry_hyd=columns.pres2_hydd_p, exner=columns.pi_p,
             zmid=columns.zmid_p, rublten_in=rublten, rvblten_in=rvblten,
             mesh_density=static.mesh_density, var2d=static.var2d,
             oc1=static.con, oa1=static.oa1, oa2=static.oa2, oa3=static.oa3,
             oa4=static.oa4, ol1=static.ol1, ol2=static.ol2, ol3=static.ol3,
             ol4=static.ol4, rublten_out=output["rublten"],
             rvblten_out=output["rvblten"], dtaux3d=output["dtaux3d"],
             dtauy3d=output["dtauy3d"], dusfcg=output["dusfcg"],
             dvsfcg=output["dvsfcg"], rubldiff=output["rubldiff"],
             rvbldiff=output["rvbldiff"], invalid=invalid),
        dict(nlev=_i(nlev), ncells=_i(ncells), dt=dt, len_disp=length_scale,
             gravity=GRAVITY_F32, cp=CP_F32, rd=RD_F32, fv=EP1_F32, pi=PI_F32),
    )
    started = time.perf_counter()
    invalid_value = int(invalid.to_numpy()[0])
    validation = TransferStats(int(invalid.nbytes), time.perf_counter() - started)
    if invalid_value != 0:
        raise FloatingPointError(
            "bl_ysu_gwdo Metal rejected non-finite or out-of-contract x4 input"
        )
    result = MetalYsuGwdoResultV841(
        **output,
        time_seconds=columns.time_seconds,
        validation_d2h=validation,
    )
    result.validate(n_vert_levels=nlev, n_cells=ncells)
    return result


__all__ = [
    "MetalYsuGwdoColumnViewV841",
    "MetalYsuGwdoResultV841",
    "MetalYsuGwdoStaticV841",
    "run_bl_ysu_gwdo_metal_v841",
]
