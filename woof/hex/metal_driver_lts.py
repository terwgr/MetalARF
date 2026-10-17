"""Attach opt-in local time stepping to a built resident Metal v8.4.1 dycore.

OPT-IN.  ``attach_local_timestep_metal`` is a no-op unless the driver's config
carries ``config_local_timestep=True``.  It mirrors ``cuda_driver_lts.py``: it
rebinds the same call targets on a ``MetalDycoreV841`` without editing it.

* ``metal_driver_v841.advance_acoustic_step_metal_v841`` -- the acoustic
  sub-step, replaced by a launch per active rate class.  The module name is
  process-wide, so the replacement is a dispatcher that picks the attachment
  by the driver's device context and otherwise calls the original.
* ``metal_driver_v841._recover_candidate`` -- settles the reflux, then
  divides the time averages by the sub-step count each entity ran.  Also a
  dispatcher, keyed by the driver it is handed.
* ``driver.horizontal.divergence_damping_3d`` and ``driver._vertical_coefficients``
  -- instance attributes, per class.

``detach_local_timestep_metal`` puts them back.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import metal_driver_v841 as _driver_module
from . import metal_acoustic_lts as ltsk
from .config_lts import local_timestep_enabled
from .cuda_driver_lts import SUPPORTED_STAGE_ACOUSTIC_STEPS, _StageTracker, _refuse
from .errors import ConfigurationRefusal
from .lts_v841 import LocalTimestepClassing, classify_from_grid_file
from .metal_acoustic_v841 import (
    MetalVerticalImplicitCoefficients,
    _float32,
    _int32,
    _mesh_value,
)

_ATTACHMENTS: dict[int, "MetalLocalTimestepAttachment"] = {}
_ORIGINAL: dict[str, Any] = {}


def _dispatch_advance(mesh, state, forcing, coefficients, *, context, **kwargs):
    attachment = _ATTACHMENTS.get(id(context))
    if attachment is None:
        return _ORIGINAL["advance"](
            mesh, state, forcing, coefficients, context=context, **kwargs)
    return attachment.advance_acoustic_step(
        mesh, state, forcing, coefficients, context=context, **kwargs)


def _dispatch_recover(driver, *args, **kwargs):
    attachment = getattr(driver, "local_timestep", None)
    if attachment is None:
        return _ORIGINAL["recover"](driver, *args, **kwargs)
    return attachment.recover_candidate(*args, **kwargs)


class MetalLocalTimestepAttachment:
    """Live LTS state bound to one Metal driver instance."""

    def __init__(self, driver: Any, classing: LocalTimestepClassing, *,
                 stage_acoustic_steps: tuple[int, ...]) -> None:
        self.driver = driver
        self.classing = classing
        self.stage_acoustic_steps = tuple(int(v) for v in stage_acoustic_steps)
        self.arrays = driver.arrays
        self.context = driver.context
        if (int(driver.ncells) != int(classing.n_cells)
                or int(driver.nedges) != int(classing.n_edges)):
            raise ValueError("the classing is not the mesh the driver was built on")
        self.device = ltsk.build_device_classing(
            classing, driver.atmosphere.mesh, n_vert_levels=driver.nlev,
            arrays=self.arrays)
        self.tracker = _StageTracker(expected=self.stage_acoustic_steps)
        self._active_edge_lists: list[tuple[Any, int, int]] = []
        self._step_counts: dict[int, tuple[Any, Any]] = {}
        self._iface_by_edge_rate = self._group_interface(by="edge")
        self._iface_by_coarse_rate = self._group_interface(by="coarse")
        self.launch_counts = {"acoustic_substep_class_launches": 0, "reflux_settles": 0}
        self._saved: dict[str, Any] = {}

    def _group_interface(self, *, by: str) -> tuple[tuple[int, Any, int], ...]:
        if self.device.n_iface == 0:
            return ()
        edges = self.device.iface_edges_host
        if by == "edge":
            rates = self.device.edge_rate_host[edges.astype(np.int64)]
        else:
            rates = self.device.iface_coarse_rate_host
        groups = []
        for rate in sorted({int(v) for v in rates.tolist()}):
            mask = np.flatnonzero(rates == rate).astype(np.int32)
            groups.append((
                int(rate),
                self.arrays.asarray(np.ascontiguousarray(edges[mask], dtype=np.int32)),
                int(mask.size),
            ))
        return tuple(groups)

    def _effective_rate(self, rate: int, stage_steps: int) -> int:
        best = 1
        for candidate in range(1, int(stage_steps) + 1):
            if int(stage_steps) % candidate == 0 and candidate <= int(rate):
                best = candidate
        return best

    def _stage_steps_for_dts(self, dts: float) -> int:
        config = self.driver.config
        dt_dyn = float(config.config_dt) / int(config.config_dynamics_split_steps)
        substeps = int(config.config_number_of_sub_steps)
        if dts == dt_dyn / 3.0:
            return 1
        if dts == dt_dyn / substeps:
            return min(steps for steps in self.stage_acoustic_steps if steps > 1)
        raise ConfigurationRefusal(
            "acoustic_timestep", dts,
            "the acoustic timestep is not one the pinned RK3 schedule emits, "
            "so the rate classes cannot be told which stage they are in",
            f"dts in ({dt_dyn / 3.0}, {dt_dyn / substeps})",
        )

    def _active_classes(self, stage_steps: int, small_step: int) -> list[tuple[int, int]]:
        active = []
        for index, rate in enumerate(self.device.rates):
            effective = self._effective_rate(rate, stage_steps)
            if (int(small_step) - 1) % effective == 0:
                active.append((index, effective))
        return active

    def _counts_for(self, stage_steps: int) -> tuple[Any, Any]:
        key = int(stage_steps)
        if key not in self._step_counts:
            self._step_counts[key] = ltsk.entity_step_counts(
                self.device, key, arrays=self.arrays)
        return self._step_counts[key]

    # -- vertical coefficients ---------------------------------------------
    def vertical_coefficients(
        self, state: Any, dts: float, theta_m: Any, rtheta_p: Any, *,
        exner: Any | None = None, validation_flag: Any | None = None,
        cqw: Any | None = None, qtot: Any | None = None,
    ) -> MetalVerticalImplicitCoefficients:
        driver = self.driver
        arrays = self.arrays
        context = driver.v841_context
        if context is None:
            raise RuntimeError("local time stepping requires the v8.4.1 device context")
        atmosphere = driver.atmosphere
        selected_exner = atmosphere.saved.exner if exner is None else exner
        nlev, ncells = driver.nlev, driver.ncells

        def cell_array() -> Any:
            return arrays.empty((nlev, ncells), np.float32)

        result = MetalVerticalImplicitCoefficients(
            cofwr=cell_array(), cofwz=cell_array(),
            coftz=arrays.empty((nlev + 1, ncells), np.float32),
            cofwt=cell_array(), cofrz=arrays.empty((nlev,), np.float32),
            a_tri=cell_array(), b_tri=cell_array(), c_tri=cell_array(),
            alpha_tri=cell_array(), gamma_tri=cell_array(),
        )
        owns_flag = validation_flag is None
        singular = (arrays.zeros((1,), np.int32) if owns_flag
                    else _int32(validation_flag, "validation_flag"))
        ltsk.launch(self.context, "acoustic_cofrz_v841", nlev,
                    {"rdzw": atmosphere.vertical.rdzw, "cofrz": result.cofrz},
                    {"nlev": np.int32(nlev)})
        stage_steps = self._stage_steps_for_dts(float(dts))
        cqw_value = arrays.ones(state.rho.shape, np.float32) if cqw is None else cqw
        qtot_value = arrays.zeros(state.rho.shape, np.float32) if qtot is None else qtot
        vertical = atmosphere.vertical
        reference = atmosphere.reference
        for index, rate in enumerate(self.device.rates):
            effective = self._effective_rate(rate, stage_steps)
            size = self.device.cell_sizes[index]
            ltsk.launch(
                self.context, "acoustic_coefficients_lts", size,
                {
                    "zz": vertical.zz, "cqw": cqw_value, "exner": selected_exner,
                    "theta": theta_m, "rho_base": reference.rho_base,
                    "rtheta_base": reference.rho_theta_base,
                    "exner_base": reference.exner_base, "rtheta_p": rtheta_p,
                    "qtot": qtot_value, "rdzw": vertical.rdzw, "fzm": vertical.fzm,
                    "fzp": vertical.fzp, "rdzu": vertical.rdzu,
                    "etp": context.etp, "ewp": context.ewp,
                    "cofwr": result.cofwr, "cofwz": result.cofwz,
                    "coftz": result.coftz, "cofwt": result.cofwt,
                    "cofrz": result.cofrz, "a_tri": result.a_tri,
                    "b_tri": result.b_tri, "c_tri": result.c_tri,
                    "alpha_tri": result.alpha_tri, "gamma_tri": result.gamma_tri,
                    "singular": singular,
                    "lts_active": self.device.cell_lists[index],
                },
                {
                    "nlev": np.int32(nlev), "ncells": np.int32(ncells),
                    "dts": np.float32(dts * effective),
                    "gravity": np.float32(9.80616), "rgas": np.float32(287.0),
                    "cp": np.float32(1004.5),
                    "lts_n_active": np.int32(size),
                },
            )
        if owns_flag and int(singular.to_numpy()[0]) != 0:
            raise FloatingPointError("singular v8.4.1 vertical acoustic system")
        return result

    # -- acoustic sub-step --------------------------------------------------
    def advance_acoustic_step(
        self, mesh: Any, state: Any, forcing: Any, coefficients: Any, *,
        context: Any, dts: float, small_step: int, fzm: Any, fzp: Any, rdzw: Any,
        gravity: float = 9.80616, rgas: float = 287.0, cp: float = 1004.5,
        in_place: bool = True,
    ) -> Any:
        arrays = self.arrays
        ctx = self.context
        out = state if in_place else state.copy()
        nlev, nedges = map(int, out.ru_p.shape)
        ncells = int(out.rho_pp.shape[1])
        self.tracker.note(int(small_step))
        stage_steps = self.tracker.stage_steps()
        cells_on_edge = _int32(_mesh_value(mesh, "cells_on_edge"), "cells_on_edge")
        edges_on_cell = _int32(_mesh_value(mesh, "edges_on_cell"), "edges_on_cell")
        counts = _int32(_mesh_value(mesh, "n_edges_on_cell"), "n_edges_on_cell")
        signs = _float32(_mesh_value(mesh, "edge_sign_on_cell"), "edge_sign_on_cell")
        dv_edge = _float32(_mesh_value(mesh, "dv_edge"), "dv_edge")
        max_edges = int(edges_on_cell.shape[1])

        if int(small_step) == 1 and self.device.resid is not None:
            self.device.resid.fill(0.0)

        active = self._active_classes(stage_steps, int(small_step))
        if not active:
            raise RuntimeError("no rate class is active at an acoustic sub-step")

        # 1. normal-momentum update, per active EDGE class at that class's dts
        self._active_edge_lists = []
        for index, effective in active:
            edges = self.device.edge_lists[index]
            size = self.device.edge_sizes[index]
            if size == 0:
                continue
            self._active_edge_lists.append((edges, size, effective))
            ltsk.launch(
                ctx, "acoustic_ru_lts", size,
                {
                    "cells_on_edge": cells_on_edge,
                    "inv_dc_edge": context.inv_dc_edge,
                    "zz": forcing.zz, "exner": forcing.exner, "cqu": forcing.cqu,
                    "zxu": forcing.zxu, "tend_ru": forcing.tend_ru,
                    "rho_pp": out.rho_pp, "rtheta_pp": out.rtheta_pp,
                    "ru_p": out.ru_p, "ru_avg": out.ru_avg,
                    "lts_active": edges,
                },
                {
                    "nlev": np.int32(nlev), "nedges": np.int32(nedges),
                    "ncells": np.int32(ncells), "small_step": np.int32(small_step),
                    "dts": np.float32(dts * effective),
                    "gravity": np.float32(gravity), "rgas": np.float32(rgas),
                    "cp": np.float32(cp), "lts_n_active": np.int32(size),
                },
            )
            self.launch_counts["acoustic_substep_class_launches"] += 1

        # 2. refluxing bookkeeping at class interfaces
        if self.device.resid is not None:
            for sign, groups in ((1.0, self._iface_by_edge_rate),
                                 (-1.0, self._iface_by_coarse_rate)):
                for rate, edges, size in groups:
                    effective = self._effective_rate(rate, stage_steps)
                    if (int(small_step) - 1) % effective:
                        continue
                    ltsk.launch(
                        ctx, "lts_reflux_accumulate", size,
                        {"iface_edges": edges, "dv_edge": dv_edge,
                         "ru_p": out.ru_p, "resid": self.device.resid},
                        {"nlev": np.int32(nlev), "nedges": np.int32(nedges),
                         "n_iface": np.int32(size),
                         "dts": np.float32(dts * effective),
                         "sign_scale": np.float32(sign)},
                    )

        # 3. cell kernels, per active CELL class
        rs = arrays.empty((nlev, ncells), np.float32)
        ts = arrays.empty((nlev, ncells), np.float32)
        for index, effective in active:
            cells = self.device.cell_lists[index]
            size = self.device.cell_sizes[index]
            if size == 0:
                continue
            step_dts = np.float32(dts * effective)
            ltsk.launch(
                ctx, "acoustic_prepare_lts", size,
                {"rw_p": out.rw_p, "rtheta_pp": out.rtheta_pp,
                 "rtheta_pp_old": out.rtheta_pp_old, "rho_pp": out.rho_pp,
                 "ww_avg": out.ww_avg, "lts_active": cells},
                {"nlev": np.int32(nlev), "ncells": np.int32(ncells),
                 "small_step": np.int32(small_step),
                 "lts_n_active": np.int32(size)},
            )
            ltsk.launch(
                ctx, "acoustic_rs_ts_lts", size,
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
                    "lts_active": cells,
                },
                {"nlev": np.int32(nlev), "ncells": np.int32(ncells),
                 "nedges": np.int32(nedges), "max_edges": np.int32(max_edges),
                 "dts": step_dts, "lts_n_active": np.int32(size)},
            )
            ltsk.launch(
                ctx, "acoustic_column_solve_lts", size,
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
                    "lts_active": cells,
                },
                {"nlev": np.int32(nlev), "ncells": np.int32(ncells),
                 "dts": step_dts, "lts_n_active": np.int32(size)},
            )
        return out

    # -- divergence damping -------------------------------------------------
    def divergence_damping_3d(
        self, rho_u_perturbation: Any, theta_m: Any, rtheta_pp: Any,
        rtheta_pp_old: Any, *, dts: float, config_smdiv: float = 0.1,
        config_len_disp: float = 0.0, spec_zone_mask_edge: Any | None = None,
        n_cells_solve: int | None = None, in_place: bool = False,
    ) -> Any:
        horizontal = self.driver.horizontal
        if not self._active_edge_lists:
            raise RuntimeError(
                "divergence damping ran without a preceding local-timestep "
                "acoustic sub-step, so the active edge classes are unknown")
        configured_length = float(config_len_disp)
        length = (float(horizontal.mesh.nominal_min_dc)
                  if configured_length == 0.0 else configured_length)
        mask = (horizontal.mesh.spec_zone_mask_edge
                if spec_zone_mask_edge is None else spec_zone_mask_edge)
        solve_count = horizontal.ncells if n_cells_solve is None else int(n_cells_solve)
        out = rho_u_perturbation if bool(in_place) else rho_u_perturbation.copy()
        for edges, size, effective in self._active_edge_lists:
            coefficient = np.float32(
                np.float32(2.0) * np.float32(float(config_smdiv))
                * np.float32(length) / np.float32(float(dts) * effective))
            ltsk.launch(
                self.context, "divergence_damping_lts", size,
                {"theta_m": theta_m, "rtheta_pp": rtheta_pp,
                 "rtheta_pp_old": rtheta_pp_old,
                 "cells_on_edge": horizontal.mesh.cells_on_edge,
                 "spec_zone_mask_edge": mask, "rho_u_perturbation": out,
                 "lts_active": edges},
                {"coefficient": coefficient, "n_cells_solve": np.int32(solve_count),
                 "nlev": np.int32(horizontal.nlev),
                 "ncells": np.int32(horizontal.ncells),
                 "nedges": np.int32(horizontal.nedges),
                 "lts_n_active": np.int32(size)},
            )
        self._active_edge_lists = []
        return out

    # -- stage close: settle the reflux, then recover -----------------------
    def recover_candidate(
        self, saved_state: Any, saved_diag: Any, acoustic: Any, stage: int,
        acoustic_steps: int, scalars: Any,
    ) -> tuple[Any, Any, Any, Any]:
        driver = self.driver
        arrays = self.arrays
        nlev, ncells, nedges = driver.nlev, driver.ncells, driver.nedges
        if self.device.resid is not None:
            ltsk.launch(
                self.context, "lts_reflux_settle", self.device.n_settle,
                {
                    "coarse_cells": self.device.settle_cells,
                    "slot_offsets": self.device.settle_offsets,
                    "slot_edges": self.device.settle_edges,
                    "slot_signs": self.device.settle_signs,
                    "inv_area_cell": driver.v841_context.inv_area_cell,
                    "theta_m": saved_diag.theta_m,
                    "cells_on_edge": _int32(
                        _mesh_value(driver.atmosphere.mesh, "cells_on_edge"),
                        "cells_on_edge"),
                    "resid": self.device.resid, "rho_pp": acoustic.rho_pp,
                    "rtheta_pp": acoustic.rtheta_pp,
                },
                {"nlev": np.int32(nlev), "nedges": np.int32(nedges),
                 "ncells": np.int32(ncells),
                 "n_coarse": np.int32(self.device.n_settle)},
            )
            self.launch_counts["reflux_settles"] += 1
        state, diagnostics, flux_u, flux_w = self._stock_recover(
            driver, saved_state, saved_diag, acoustic, stage, acoustic_steps, scalars)
        if not self.device.single_class:
            cell_steps, edge_steps = self._counts_for(int(acoustic_steps))
            scratch_ru = arrays.empty(state.rho_u.shape, np.float32)
            ltsk.launch(
                self.context, "lts_recover_edges", nedges,
                {"edge_steps": edge_steps, "ru_saved": saved_state.rho_u,
                 "ru_p": acoustic.ru_p, "ru_avg_p": acoustic.ru_avg,
                 "ru": scratch_ru, "flux_u": flux_u},
                {"nlev": np.int32(nlev), "nedges": np.int32(nedges)},
            )
            scratch_rw = arrays.empty(state.rho_w.shape, np.float32)
            ltsk.launch(
                self.context, "lts_recover_interfaces", ncells,
                {"cell_steps": cell_steps, "rw_saved": saved_state.rho_w,
                 "rw_p": acoustic.rw_p, "ww_avg_p": acoustic.ww_avg,
                 "rw": scratch_rw, "flux_w": flux_w},
                {"nlev": np.int32(nlev), "ncells": np.int32(ncells)},
            )
        return state, diagnostics, flux_u, flux_w

    # -- (de)attachment -----------------------------------------------------
    def attach(self) -> None:
        driver = self.driver
        if not _ATTACHMENTS:
            _ORIGINAL["advance"] = _driver_module.advance_acoustic_step_metal_v841
            _ORIGINAL["recover"] = _driver_module._recover_candidate
            _driver_module.advance_acoustic_step_metal_v841 = _dispatch_advance
            _driver_module._recover_candidate = _dispatch_recover
        self._saved["damping"] = driver.horizontal.divergence_damping_3d
        self._stock_recover = _ORIGINAL["recover"]
        _ATTACHMENTS[id(driver.v841_context)] = self
        driver.horizontal.divergence_damping_3d = self.divergence_damping_3d
        driver._vertical_coefficients = self.vertical_coefficients
        driver.local_timestep = self

    def detach(self) -> None:
        driver = self.driver
        _ATTACHMENTS.pop(id(driver.v841_context), None)
        if not _ATTACHMENTS and _ORIGINAL:
            _driver_module.advance_acoustic_step_metal_v841 = _ORIGINAL.pop("advance")
            _driver_module._recover_candidate = _ORIGINAL.pop("recover")
        driver.horizontal.divergence_damping_3d = self._saved["damping"]
        for name in ("_vertical_coefficients", "local_timestep"):
            if name in driver.__dict__:
                del driver.__dict__[name]

    def receipt(self) -> dict[str, Any]:
        summary = self.classing.summary()
        summary.update({
            "stage_acoustic_steps": list(self.stage_acoustic_steps),
            "arithmetic_acoustic_saving": self.classing.arithmetic_acoustic_saving(
                self.stage_acoustic_steps),
            "launch_counts": dict(self.launch_counts),
            "conservation": "approximate: flux-matched, binary32 rounding",
            "settle_coarse_cells": int(self.device.n_settle),
            "settle_is_order_defined": True,
            "regional": False,
            "executed_path": (
                "single class: every launch list is the identity "
                "permutation and the arithmetic is the option-off path"
                if self.classing.identity_permutation
                else "rate classes launched over index lists"),
        })
        return summary


def attach_local_timestep_metal(
    driver: Any, *, grid_path: str | None = None,
    classing: LocalTimestepClassing | None = None,
) -> MetalLocalTimestepAttachment | None:
    """Turn the option on for one driver, or return ``None`` if it is off."""

    config = driver.config
    if not local_timestep_enabled(config):
        return None
    if driver.v841_context is None:
        _refuse("config_local_timestep", True,
                "local time stepping is implemented against the v8.4.1 acoustic "
                "solver and there is no v8.4.1 device context on this driver",
                "a v8.4.1 Metal run")
    if getattr(driver, "local_timestep", None) is not None:
        _refuse("config_local_timestep", True,
                "local time stepping is already attached to this driver",
                "one attachment per driver")
    if getattr(driver, "halo_exchanger_v841", None) is not None:
        _refuse("config_local_timestep", True,
                "a partitioned run exchanges halo rings on a schedule built for one "
                "global sub-step rate, so a coarse rank would damp and exchange "
                "stale ring-1 state",
                "a single-device run")
    if getattr(driver, "regional_v841", None) is not None:
        _refuse("config_local_timestep", True,
                "the Metal local time stepping has no limited-area acoustic "
                "entrypoints; the regional trio is not ported",
                "a global (non-regional) run")
    for knob in ("config_monotonic", "config_positive_definite"):
        if getattr(config, knob, False):
            _refuse(knob, True,
                    "a flux-corrected transport limiter clips against neighbours "
                    "that a rate boundary has not advanced to the same time, so it "
                    "would clip the very flux the reflux balances the interface "
                    "with; that combination has not been proved",
                    f"{knob}=False with config_local_timestep=True")
    order = int(getattr(config, "config_time_integration_order", 3))
    substeps = int(getattr(config, "config_number_of_sub_steps", 6))
    stages = (1, max(1, substeps // 2), substeps)
    if order != 3 or stages != SUPPORTED_STAGE_ACOUSTIC_STEPS:
        _refuse("config_number_of_sub_steps", substeps,
                "the rate ladder and the per-stage effective rates were derived "
                f"for the released {SUPPORTED_STAGE_ACOUSTIC_STEPS} acoustic "
                "schedule; another schedule admits a different ladder and has not "
                "been proved",
                "config_time_integration_order=3 with config_number_of_sub_steps=6")
    rates = tuple(int(v) for v in config.config_local_timestep_rates)
    ltsk.refuse_unsupported_ladder(rates, SUPPORTED_STAGE_ACOUSTIC_STEPS)
    if classing is None:
        if grid_path is None:
            _refuse("config_local_timestep", True,
                    "cells are classed from the grid file's own dcEdge array, and "
                    "no grid file was supplied; nominalMinDc is not a substitute "
                    "(it can read high, which is the unsafe direction)",
                    "a grid file path")
        classing = classify_from_grid_file(
            grid_path, rates=rates,
            buffer_rings=int(config.config_local_timestep_buffer_rings),
            safety_factor=float(config.config_local_timestep_safety_factor))
    if classing.n_cells != int(driver.ncells) or classing.n_edges != int(driver.nedges):
        _refuse("config_local_timestep", (classing.n_cells, classing.n_edges),
                "the classed grid file is not the mesh the driver was built on",
                f"a grid file with {driver.ncells} cells and {driver.nedges} edges")
    attachment = MetalLocalTimestepAttachment(
        driver, classing, stage_acoustic_steps=SUPPORTED_STAGE_ACOUSTIC_STEPS)
    attachment.attach()
    return attachment


def detach_local_timestep_metal(driver: Any) -> None:
    """Put every rebound target back; a no-op when nothing is attached."""

    attachment = getattr(driver, "local_timestep", None)
    if attachment is not None:
        attachment.detach()


__all__ = [
    "MetalLocalTimestepAttachment",
    "attach_local_timestep_metal",
    "detach_local_timestep_metal",
]
