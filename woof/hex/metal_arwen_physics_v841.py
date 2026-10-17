"""Two-phase column physics for the resident Metal Hex dycore.

``MetalTwoPhasePhysicsBackendV841`` is the Metal sibling of
``PersistentTwoPhaseCudaPhysicsBackendV841``: the same transaction around the
column-batch seam (phase one on the step-start state, WSM6 on the clamped
endpoint, publication only after the dycore commits), on the Metal seam
``woof.core.mpas_column_batch_metal`` and the Metal preparation, YSU-GWDO and
coupling lanes.  :func:`execute_composite_step_metal_v841` runs one step in
the order of the CUDA ``execute_composite_step``.
"""

from __future__ import annotations

import math
from datetime import timedelta
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

import numpy as np

from . import metal_v841_launch
from .config_v841 import (
    V841_GWDO_KERNEL_SHA256,
    V841_GWDO_RESULT_CLASS,
    V841_GWDO_RESULT_MODULE,
)
from .cuda_physics_v841 import CudaPhaseOneExecutionProvenanceV841
from .metal_gwdo_v841 import (
    MetalYsuGwdoColumnViewV841,
    MetalYsuGwdoResultV841,
    MetalYsuGwdoStaticV841,
    run_bl_ysu_gwdo_metal_v841,
)
from .metal_physics_prep_v841 import (
    MetalMpasToPhysGeometryV841,
    prepare_mpas_to_phys_metal_v841,
)
from .metal_physics_v841 import (
    MetalPhysicsGeometryV841,
    MetalPostRkWsm6UpdateV841,
    MetalRawColumnPhysicsV841,
    clamp_wsm6_scalars_in_place_v841,
    couple_raw_column_physics_v841,
    recover_post_rk_wsm6_state_v841,
)
from .species_row import species_row_for_names

_ENGINE_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")
# The CUDA backend's diagnostic groups (cuda_arwen_physics_v841).
_REQUIRED_EXPORT_FIELDS = ("tsk", "smois", "tslb", "hfx", "qfx", "lh")
_OPTIONAL_EXPORT_FIELDS = (
    "t2", "q2", "pblh", "u10", "v10", "psfc", "swdown", "glw", "olr")
_SOIL_FIELDS = frozenset(("smois", "tslb"))
_GWDO_SURFACE_FIELDS = ("dusfcg", "dvsfcg")
_GWDO_LEVEL_FIELDS = ("dtaux3d", "dtauy3d", "rubldiff", "rvbldiff")


def _host(value: Any) -> np.ndarray:
    return np.asarray(value.to_numpy() if hasattr(value, "to_numpy") else value)


class MetalTwoPhasePhysicsBackendV841:
    """Transactional Metal column-batch seam implementing the MPAS protocol."""

    def __init__(
        self,
        *,
        constructor_values: Mapping[str, Any],
        prep_geometry: MetalMpasToPhysGeometryV841,
        array_module: Any,
        gwdo_static: MetalYsuGwdoStaticV841 | None = None,
    ) -> None:
        from woof.core.mpas_column_batch_metal import run_mpas_column_batch_metal

        prep_geometry.validate()
        self._values = dict(constructor_values)
        self._n_levels = int(self._values["n_levels"])
        self._n_columns = int(self._values["n_columns"])
        self._dt = float(self._values["dt"])
        if prep_geometry.n_cells != self._n_columns:
            raise ValueError(
                "preparation geometry and constructor column counts differ")
        if gwdo_static is not None:
            gwdo_static.validate()
            if gwdo_static.n_cells != self._n_columns:
                raise ValueError("GWDO statics and constructor column counts differ")
        self._xp = array_module
        self._factory = run_mpas_column_batch_metal
        self._prep_geometry = prep_geometry
        self._gwdo_static = gwdo_static
        self._seam = self._new_seam()
        self._phase = "boundary"
        self._snapshot: Mapping[str, Any] | None = None
        self._step_start: float | None = None
        self._scalar_target: Any = None
        self._scalar_backup: Any = None
        self._pending_gwdo = None
        self.last_gwdo = None
        self._pending_refl10cm = None
        self._committed_refl10cm = None
        self._gf_forcing_seen = False

    def _new_seam(self) -> Any:
        return self._factory(**self._values, array_module=self._xp)

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def seam(self) -> Any:
        return self._seam

    def _restore_boundary(self, snapshot: Mapping[str, Any]) -> None:
        try:
            fresh = self._new_seam()
            fresh.restore_state(snapshot)
            self._seam = fresh
        except Exception as error:
            self._phase = "broken"
            raise RuntimeError("failed to reconstruct the physics seam") from error
        self._phase = "boundary"
        self._snapshot = None
        self._step_start = None
        self._scalar_target = None
        self._scalar_backup = None
        self._pending_gwdo = None
        self._pending_refl10cm = None

    def _gf_dynamics_lanes(self, carrier: Any, *, start: float) -> tuple[Any, Any]:
        """The previous step's GF advective forcing, checked by its clock."""

        if carrier is None:
            if self._gf_forcing_seen:
                raise ValueError(
                    "GF dynamics forcing vanished mid-run; omitting it now "
                    "would silently restore zero forcing")
            return None, None
        self._gf_forcing_seen = True
        held = float(getattr(carrier, "time_seconds", float("nan")))
        if not math.isfinite(held) or held != start:
            raise ValueError(
                "GF dynamics forcing must come from the PREVIOUS dynamics "
                f"step, whose endpoint is this step's start t={start} s; "
                f"got a carrier stamped t={held} s")
        for name in ("rthdynten", "rqvdynten"):
            shape = tuple(getattr(carrier, name).shape)
            if shape != (self._n_levels, self._n_columns):
                raise ValueError(
                    f"GF {name} must be [level, cell]="
                    f"{(self._n_levels, self._n_columns)}, got {shape}")
        return carrier.rthdynten, carrier.rqvdynten

    def _check(self, scalar_names: Sequence[str], dt: float) -> None:
        species_row_for_names(tuple(scalar_names))
        if float(dt) != self._dt:
            raise ValueError(
                f"backend dt={dt} does not equal constructor dt={self._dt}")

    def begin_step(
        self, *, atmosphere: Any, scalar_names: Sequence[str], dt: float,
        dynamics_tendencies: Any = None,
    ) -> MetalRawColumnPhysicsV841:
        """Prepare the MPAS columns and run phase one exactly once."""

        if self._phase != "boundary":
            raise RuntimeError("begin_step requires a clean step boundary")
        self._check(scalar_names, dt)
        start = float(atmosphere.state.time_seconds)
        if not math.isfinite(start) or start < 0.0:
            raise ValueError("candidate start time must be finite and non-negative")
        if float(self._seam.elapsed_seconds) != start:
            raise ValueError(
                "physics and MPAS clocks differ at begin_step: "
                f"{self._seam.elapsed_seconds} != {start}")
        snapshot = self._seam.export_state()
        self._snapshot = snapshot
        self._step_start = start
        self._pending_gwdo = None
        try:
            prepared = prepare_mpas_to_phys_metal_v841(
                atmosphere, scalar_names=scalar_names,
                geometry=self._prep_geometry, post_rk_wsm6=False)
            if float(prepared.time_seconds) != start:
                raise ValueError(
                    "phase-one prep time changed from the exact MPAS start")
            rthdynten, rqvdynten = self._gf_dynamics_lanes(
                dynamics_tendencies, start=start)
            output = self._seam.run_phase1(
                dt=dt,
                u=prepared.u_p,
                v=prepared.v_p,
                theta=prepared.th_p,
                pressure=prepared.pres_hyd_p,
                pressure_interface=prepared.pres2_hyd_p,
                z_interface=prepared.z_p,
                w=prepared.w_p,
                rho_dry=prepared.rho_dry,
                **{name: prepared.species_p[name] for name in scalar_names},
                exner=prepared.pi_p,
                rthdynten=rthdynten,
                rqvdynten=rqvdynten,
            )
            if float(output.elapsed_seconds) != start:
                raise ValueError(
                    "phase-one output time does not equal the MPAS start")
            if int(output.step_index) != int(self._seam.step_index):
                raise ValueError("phase-one step index changed during phase one")
            shape = (self._n_levels, self._n_columns)
            for name in ("du", "dv", "dtheta", *(f"d{s}" for s in _ENGINE_SPECIES),
                         "h_diabatic"):
                value = getattr(output, name)
                if tuple(value.shape) != shape or np.dtype(value.dtype) != np.float32:
                    raise ValueError(
                        f"phase-one {name} must be float32 {shape}, got "
                        f"{np.dtype(value.dtype)} {tuple(value.shape)}")
            du, dv = output.du, output.dv
            provenance = _provenance(None)
            if self._gwdo_static is not None:
                gwdo = run_bl_ysu_gwdo_metal_v841(
                    MetalYsuGwdoColumnViewV841.from_prepared(prepared),
                    rublten=output.du, rvblten=output.dv,
                    static=self._gwdo_static, dt_seconds=dt)
                du, dv = gwdo.rublten, gwdo.rvblten
                self._pending_gwdo = gwdo
                provenance = _provenance(gwdo)
            raw = MetalRawColumnPhysicsV841(
                du=du, dv=dv, dtheta=output.dtheta,
                # Phase one publishes rates for the six bulk species; a row's
                # number and rime moments have none and couple as zero.
                dscalars={name: getattr(output, f"d{name}")
                          for name in _ENGINE_SPECIES if name in scalar_names},
                time_seconds=start,
                execution_provenance=provenance,
                scalar_names=tuple(scalar_names),
            )
            raw.validate(n_vert_levels=self._n_levels, n_cells=self._n_columns)
            self._phase = "begun"
            return raw
        except Exception:
            self._restore_boundary(snapshot)
            raise

    def finish_step(
        self, *, atmosphere: Any, scalar_names: Sequence[str], dt: float,
        refl_10cm_due: bool = False,
    ) -> MetalPostRkWsm6UpdateV841:
        """Run WSM6 once on the clamped endpoint and stage its outputs."""

        if self._phase != "begun" or self._snapshot is None:
            raise RuntimeError("finish_step requires one successful begin_step")
        snapshot = self._snapshot
        try:
            self._check(scalar_names, dt)
            endpoint = float(atmosphere.state.time_seconds)
            if endpoint != float(self._step_start) + self._dt:
                raise ValueError(
                    "post-RK candidate time must equal the exact step endpoint: "
                    f"{endpoint} != {float(self._step_start) + self._dt}")
        except Exception:
            self._restore_boundary(snapshot)
            raise
        backup = atmosphere.state.scalars.copy()
        try:
            prepared = prepare_mpas_to_phys_metal_v841(
                atmosphere, scalar_names=scalar_names,
                geometry=self._prep_geometry, post_rk_wsm6=True)
            if float(prepared.time_seconds) != endpoint:
                raise ValueError("phase-two prep time changed from the exact endpoint")
            wsm6 = prepared.wsm6_input_view()
            row = species_row_for_names(tuple(scalar_names))
            # rho_dry only for WSM6: P3 and mp=28 derive density themselves.
            receipt = self._seam.run_phase2(
                theta=prepared.th_p,
                pressure=prepared.pres_p,
                z_interface=prepared.z_p,
                refl_10cm_due=bool(refl_10cm_due),
                **({"rho_dry": prepared.rho_dry}
                   if row.engine_scheme == "wsm6" else {}),
                **{name: wsm6.species[name] for name in scalar_names},
            )
            if float(self._seam.elapsed_seconds) != endpoint:
                raise ValueError("phase two did not advance to the MPAS endpoint")
            cumulative = self._seam.accumulated_precipitation()
            required = {item.history_name for item in row.surface_accumulators} | {"RAINC"}
            if set(cumulative) != required:
                raise ValueError(
                    "the seam's cumulative precipitation keys changed: "
                    f"{sorted(cumulative)}, expected {sorted(required)}")
            surface = {"sr": receipt["sr"]}
            for item in row.surface_accumulators:
                surface[item.name] = cumulative[item.history_name]
                surface[f"{item.name}v"] = receipt[f"{item.name}v"]
            shape = (self._n_levels, self._n_columns)
            radii = {name: getattr(self._seam._state, name).view(shape).copy()
                     for name in row.radius_names}
            update = MetalPostRkWsm6UpdateV841(
                theta=prepared.th_p,
                species={name: wsm6.species[name] for name in scalar_names},
                surface=surface,
                radii=radii,
                time_seconds=endpoint,
            )
            update.validate(n_vert_levels=self._n_levels, n_cells=self._n_columns)
            if refl_10cm_due:
                refl = receipt.get("refl_10cm")
                if refl is None or tuple(refl.shape) != shape:
                    raise ValueError(
                        "the due step's microphysics returned no refl_10cm "
                        f"of shape {shape}")
                self._pending_refl10cm = refl
            self._phase = "finished"
            self._scalar_target = atmosphere.state.scalars
            self._scalar_backup = backup
            return update
        except Exception:
            restore_error = None
            try:
                atmosphere.state.scalars[...] = backup
            except Exception as error:
                restore_error = error
            self._restore_boundary(snapshot)
            if restore_error is not None:
                raise RuntimeError(
                    "failed to restore unpublished MPAS candidate scalars"
                ) from restore_error
            raise

    def abort_step(self) -> None:
        """Roll back a begun or finished-unpublished step."""

        if self._phase not in ("begun", "finished") or self._snapshot is None:
            raise RuntimeError("abort_step requires a begun or finished transaction")
        restore_error = None
        if self._phase == "finished":
            try:
                self._scalar_target[...] = self._scalar_backup
            except Exception as error:
                restore_error = error
        self._restore_boundary(self._snapshot)
        if restore_error is not None:
            raise RuntimeError(
                "the physics seam rolled back but the unpublished MPAS scalars "
                "were not restored") from restore_error

    def commit_step(self) -> None:
        """Publish a finished seam after the dycore commit succeeded."""

        if self._phase != "finished" or self._snapshot is None:
            raise RuntimeError("commit_step requires a finished-unpublished transaction")
        if self._gwdo_static is not None and self._pending_gwdo is None:
            self._restore_boundary(self._snapshot)
            raise RuntimeError("finished transaction lost its GWDO result")
        if self._pending_gwdo is not None:
            self.last_gwdo = self._pending_gwdo
        self._pending_gwdo = None
        if self._pending_refl10cm is not None:
            self._committed_refl10cm = self._pending_refl10cm
        self._pending_refl10cm = None
        self._phase = "boundary"
        self._snapshot = None
        self._step_start = None
        self._scalar_target = None
        self._scalar_backup = None

    def take_history_refl10cm(self) -> Any | None:
        """The last due step's committed reflectivity, handed out once."""

        if self._phase != "boundary":
            raise RuntimeError("take_history_refl10cm is legal only at a boundary")
        refl, self._committed_refl10cm = self._committed_refl10cm, None
        return refl

    def diagnostic_snapshot(self) -> dict[str, dict[str, np.ndarray]]:
        """Host copies of the committed surface, soil, precipitation and GWDO
        diagnostics, grouped and shaped as the CUDA backend's snapshot."""

        if self._phase != "boundary":
            raise RuntimeError("diagnostic_snapshot is legal only at a boundary")
        ncol, nlev = self._n_columns, self._n_levels
        exported = self._seam.export_state()["arrays"]
        surface: dict[str, np.ndarray] = {}
        soil: dict[str, np.ndarray] = {}
        for name in (*_REQUIRED_EXPORT_FIELDS, *_OPTIONAL_EXPORT_FIELDS):
            key = f"fields/{name}"
            if key not in exported:
                if name in _REQUIRED_EXPORT_FIELDS:
                    raise ValueError(f"the physics seam export lacks {key!r}")
                continue
            host = np.asarray(exported[key], dtype=np.float32)
            if name in _SOIL_FIELDS:
                soil[name] = np.ascontiguousarray(host.reshape(-1, ncol))
            else:
                surface[name] = np.ascontiguousarray(host.reshape(ncol))
        precipitation = {
            key.lower(): np.asarray(_host(value), dtype=np.float32).reshape(ncol)
            for key, value in self._seam.accumulated_precipitation().items()}
        gwdo: dict[str, np.ndarray] = {}
        for name in _GWDO_SURFACE_FIELDS:
            gwdo[name] = (np.zeros(ncol, np.float32) if self.last_gwdo is None
                          else _host(getattr(self.last_gwdo, name)).reshape(ncol))
        for name in _GWDO_LEVEL_FIELDS:
            gwdo[name] = (np.zeros((nlev, ncol), np.float32) if self.last_gwdo is None
                          else _host(getattr(self.last_gwdo, name)).reshape(nlev, ncol))
        return {"surface": surface, "soil": soil,
                "precipitation": precipitation, "gwdo": gwdo}

    def accumulated_precipitation(self) -> dict[str, Any]:
        """Committed bucket totals keyed as the post-WSM6 recovery reads them."""

        if self._phase != "boundary":
            raise RuntimeError("accumulated_precipitation is legal only at a boundary")
        return {key.lower(): value
                for key, value in self._seam.accumulated_precipitation().items()}

    def restart_state(self) -> Mapping[str, Any]:
        if self._phase != "boundary":
            raise RuntimeError("restart_state is legal only at a step boundary")
        return {"seam": self._seam.export_state(),
                "gf_forcing_seen": self._gf_forcing_seen}

    def restore_restart_state(self, payload: Mapping[str, Any]) -> None:
        if self._phase != "boundary":
            raise RuntimeError("restore_restart_state is legal only at a step boundary")
        if set(payload) != {"seam", "gf_forcing_seen"}:
            raise ValueError(f"restart payload keys changed: {sorted(payload)}")
        fresh = self._new_seam()
        fresh.restore_state(payload["seam"])
        self._seam = fresh
        self._gf_forcing_seen = bool(payload["gf_forcing_seen"])
        self.last_gwdo = None
        self._pending_gwdo = None
        self._committed_refl10cm = None


@dataclass(frozen=True, slots=True)
class MetalPhaseOneExecutionProvenanceV841(CudaPhaseOneExecutionProvenanceV841):
    """Phase-one provenance whose external YSU-GWDO ran on Metal.

    The Metal result mirrors the CUDA one field for field; the CUDA law
    pins the result module, class and CUDA kernel digest, so those three
    are checked here as the Metal identities and the rest by the CUDA law.
    """

    def validate(self) -> None:
        if not self.gwdo_executed:
            CudaPhaseOneExecutionProvenanceV841.validate(self)
            return
        metal = {
            "gwdo_result_module": MetalYsuGwdoResultV841.__module__,
            "gwdo_result_class": MetalYsuGwdoResultV841.__name__,
            "gwdo_kernel_sha256": None,
        }
        for name, required in metal.items():
            if getattr(self, name) != required:
                raise ValueError(
                    f"Metal GWDO provenance {name} must be exactly {required!r}")
        # Constructing the CUDA carrier runs the CUDA law on the rest.
        CudaPhaseOneExecutionProvenanceV841(**{
            **{item.name: getattr(self, item.name) for item in fields(self)},
            "gwdo_result_module": V841_GWDO_RESULT_MODULE,
            "gwdo_result_class": V841_GWDO_RESULT_CLASS,
            "gwdo_kernel_sha256": V841_GWDO_KERNEL_SHA256,
        })


def _provenance(gwdo: Any) -> MetalPhaseOneExecutionProvenanceV841:
    if gwdo is None:
        return MetalPhaseOneExecutionProvenanceV841.arwen_gwd_off(
            aggregate_executed=True)
    return MetalPhaseOneExecutionProvenanceV841.arwen_with_external_gwdo(
        aggregate_executed=True,
        gwdo_executed=True,
        gwdo_composed_once=True,
        gwdo_result_module=type(gwdo).__module__,
        gwdo_result_class=type(gwdo).__name__,
        gwdo_contract_sha256=gwdo.contract_sha256,
        gwdo_kernel_sha256=None,
        gwdo_validation_d2h=gwdo.validation_d2h,
        gwdo_input_du_is_arwen_output=True,
        gwdo_input_dv_is_arwen_output=True,
        raw_du_is_gwdo_output=True,
        raw_dv_is_gwdo_output=True,
    )


@dataclass(frozen=True)
class MetalCompositeStepV841:
    committed: Any
    dynamics_tendencies: Any
    surface_updates: Any


def execute_composite_step_metal_v841(
    *,
    driver: Any,
    backend: MetalTwoPhasePhysicsBackendV841,
    scalar_names: Sequence[str],
    physics_geometry: MetalPhysicsGeometryV841,
    previous_surface_updates: Mapping[str, Any] | None,
    dynamics_tendencies: Any = None,
    refl_10cm_due: bool = False,
) -> MetalCompositeStepV841:
    """One staged dycore-plus-physics step; nothing publishes on failure."""

    dt = float(driver.config.config_dt)
    start_state = driver.atmosphere.state
    start_time = float(start_state.time_seconds)
    candidate = None
    committed = None
    try:
        # The dycore's garbage discipline is scoped to the dycore launches;
        # the physics runs its padded column on well-posed duplicates.
        with metal_v841_launch.suspended():
            raw = backend.begin_step(
                atmosphere=driver.atmosphere, scalar_names=scalar_names, dt=dt,
                dynamics_tendencies=dynamics_tendencies)
        edges = driver.horizontal.recover_edge_fields(
            start_state.rho, start_state.rho_u)
        with metal_v841_launch.suspended():
            held = couple_raw_column_physics_v841(
                raw, state=start_state, scalar_names=scalar_names,
                geometry=physics_geometry, rho_edge=edges.rho_edge)
        candidate = driver.step_device_with_physics(held)
        clamp_wsm6_scalars_in_place_v841(
            candidate.atmosphere.state.scalars, scalar_names=scalar_names)
        with metal_v841_launch.suspended():
            update = backend.finish_step(
                atmosphere=candidate.atmosphere, scalar_names=scalar_names,
                dt=dt, refl_10cm_due=refl_10cm_due)
            recovery = recover_post_rk_wsm6_state_v841(
                candidate.atmosphere.state, update, scalar_names=scalar_names,
                phase2_dt_seconds=dt,
                previous_surface_updates=previous_surface_updates)
        committed = driver.commit_post_wsm6_candidate(candidate, recovery)
        backend.commit_step()
    except Exception as error:
        rollback_errors: list[str] = []
        if candidate is not None and committed is None:
            try:
                driver.abort_post_wsm6_candidate(candidate)
            except Exception as abort_error:
                rollback_errors.append(f"dycore abort failed: {abort_error}")
        if backend.phase in ("begun", "finished"):
            try:
                backend.abort_step()
            except Exception as abort_error:
                rollback_errors.append(f"physics rollback failed: {abort_error}")
        if committed is not None:
            raise RuntimeError(
                "the physics commit failed after the dycore published its "
                f"endpoint at {start_time + dt} s; the run cannot continue"
            ) from error
        if (driver.atmosphere.state is not start_state
                or float(driver.atmosphere.state.time_seconds) != start_time):
            rollback_errors.append("the dycore did not keep its exact start state")
        if rollback_errors:
            raise RuntimeError(
                f"composite step at {start_time} s failed and its rollback was "
                "incomplete: " + "; ".join(rollback_errors)) from error
        raise
    if float(committed.atmosphere.state.time_seconds) != start_time + dt:
        raise RuntimeError("composite step committed the wrong endpoint time")
    return MetalCompositeStepV841(
        committed=committed,
        dynamics_tendencies=committed.dynamics_tendencies,
        surface_updates=committed.surface_updates,
    )


def capture_snapshot_metal_v841(
    *,
    driver: Any,
    backend: MetalTwoPhasePhysicsBackendV841,
    prep_geometry: MetalMpasToPhysGeometryV841,
    scalar_names: Sequence[str],
    step: int,
    expect_refl10cm: bool,
    solve_cells: int | None = None,
    solve_edges: int | None = None,
) -> dict[str, Any]:
    """One committed boundary on the host, as the CUDA ``capture_snapshot``
    lays it out for ``write_snapshot_netcdf``."""

    if backend.phase != "boundary":
        raise RuntimeError("a snapshot is legal only at a committed boundary")
    expected = step * float(driver.config.config_dt)
    if float(driver.atmosphere.state.time_seconds) != expected:
        raise ValueError(
            f"snapshot step {step} expects model time {expected} s, the "
            f"dycore is at {driver.atmosphere.state.time_seconds} s")
    with metal_v841_launch.suspended():
        prepared = prepare_mpas_to_phys_metal_v841(
            driver.atmosphere, scalar_names=tuple(scalar_names),
            geometry=prep_geometry, post_rk_wsm6=False)
    state, saved = driver.atmosphere.state, driver.atmosphere.saved
    arrays: dict[str, np.ndarray] = {
        name: _host(state.scalars[index]).copy()
        for index, name in enumerate(scalar_names)}
    arrays.update({
        "u_zonal": _host(prepared.u_p),
        "v_meridional": _host(prepared.v_p),
        "normal_u": _host(saved.normal_velocity),
        "rho": _host(prepared.rho_dry),
        "theta": _host(prepared.th_p),
        "pressure": _host(prepared.pres_p),
        "surface_pressure": _host(prepared.psfc_p),
        "w": _host(saved.vertical_velocity),
    })
    for group in backend.diagnostic_snapshot().values():
        for name, value in group.items():
            if name in arrays:
                raise ValueError(f"duplicate snapshot field {name!r}")
            arrays[name] = value
    if expect_refl10cm:
        refl = backend.take_history_refl10cm()
        if refl is None:
            raise RuntimeError(
                f"history step {step} expected a committed refl10cm and the "
                "backend holds none")
        arrays["refl10cm"] = _host(refl)
    receipt = {
        "step": int(step),
        "time_seconds": float(state.time_seconds),
        "arrays": {name: {"shape": list(np.shape(value)),
                          "minimum": float(np.min(value)),
                          "maximum": float(np.max(value))}
                   for name, value in sorted(arrays.items())},
    }
    if solve_cells is not None:
        # A limited-area frame publishes the domain, not the padded allocation.
        keep = {int(solve_cells) + 1: int(solve_cells),
                int(solve_edges) + 1: int(solve_edges)}
        for name, value in list(arrays.items()):
            size = int(np.shape(value)[-1])
            if size in keep:
                arrays[name] = np.ascontiguousarray(np.asarray(value)[..., :keep[size]])
        receipt["arrays"] = {
            name: {"shape": list(np.shape(value)),
                   "minimum": float(np.min(value)), "maximum": float(np.max(value))}
            for name, value in sorted(arrays.items())}
    row = species_row_for_names(tuple(scalar_names))
    required = {"tsk", "smois", "tslb", "hfx", "qfx", "lh", "t2", "u10", "v10",
                "rainc", *(item.name for item in row.surface_accumulators),
                *_GWDO_SURFACE_FIELDS, *_GWDO_LEVEL_FIELDS}
    missing = sorted(required - set(arrays))
    if missing:
        raise RuntimeError(f"the history snapshot lacks required fields {missing}")
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f"snapshot field {name!r} is not finite")
    return {"arrays": arrays, "receipt": receipt}


def open_metal_physics_stack_v841(
    context: Any,
    host: Mapping[str, Any],
    *,
    state: Any | None = None,
    saved_diagnostics: Any | None = None,
) -> dict[str, Any]:
    """Build the dycore, seam, geometries and GWDO statics from the CPU host prep.

    ``host`` is the mapping ``prepare_forecast_host`` returns, so the Metal
    stack starts from the same prepared state and seam inputs as CUDA;
    ``state`` and ``saved_diagnostics`` replace the prepared ones on a restart.
    """

    from woof.metal.array import MetalArrayModule

    from .metal_driver_v841 import MetalDycoreV841, assemble_regional_metal_dycore_v841

    prepared = host["prepared"]
    regional = host.get("regional")
    selected_state = prepared.state if state is None else state
    selected_saved = (prepared.saved_diagnostics if saved_diagnostics is None
                      else saved_diagnostics)
    if regional is None:
        driver = MetalDycoreV841.from_host(
            context, prepared.mesh, selected_state, prepared.vertical,
            prepared.reference, host["config"],
            saved_diagnostics=selected_saved,
            terrain_metrics=prepared.terrain_metrics,
            reference_wind_profiles=prepared.reference_wind_profiles,
            advection_coefficients=prepared.advection_coefficients,
        )
        physics_mesh = prepared.mesh
    else:
        # As on CUDA: the physics columns are native's padded extent, one
        # garbage column past the solve region.
        driver = assemble_regional_metal_dycore_v841(
            context, prepared.mesh, selected_state, prepared.vertical,
            prepared.reference, selected_saved, prepared.terrain_metrics,
            host["config"],
            reference_wind_profiles=prepared.reference_wind_profiles,
            lbc_paths=regional["lbc_paths"],
            # The boundary clock advances with the steps it runs, so a
            # restart starts it at the restored model time.
            start_time=regional["start_time"] + timedelta(
                seconds=float(selected_state.time_seconds)),
            scalar_names=regional["driven_scalars"],
        )
        physics_mesh = driver.regional_v841.padded
    prep_geometry = MetalMpasToPhysGeometryV841.from_host(context, physics_mesh)
    physics_geometry = MetalPhysicsGeometryV841.from_host(context, physics_mesh)
    gwdo_static = MetalYsuGwdoStaticV841.from_host(context, host["gwdo_host"])
    backend = MetalTwoPhasePhysicsBackendV841(
        constructor_values=host["constructor_values"],
        prep_geometry=prep_geometry,
        array_module=MetalArrayModule(context),
        gwdo_static=gwdo_static,
    )
    return {
        "driver": driver,
        "backend": backend,
        "prep_geometry": prep_geometry,
        "physics_geometry": physics_geometry,
        "scalar_names": tuple(host["scalar_names"]),
        "solve_cells": (None if regional is None
                        else int(driver.regional_v841.n_cells_solve)),
        "solve_edges": (None if regional is None
                        else int(driver.regional_v841.n_edges_solve)),
    }


__all__ = [
    "MetalCompositeStepV841",
    "MetalPhaseOneExecutionProvenanceV841",
    "MetalTwoPhasePhysicsBackendV841",
    "capture_snapshot_metal_v841",
    "execute_composite_step_metal_v841",
    "open_metal_physics_stack_v841",
]
