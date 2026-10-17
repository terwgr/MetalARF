"""Resident v8.4.1 Hex dycore on Metal.

``MetalDycoreV841`` runs ``CudaDryDycoreDriver._step_device_v841`` and
``_advance_dynamics_subcycle_v841`` call for call on resident ``MetalArray``
fields, with one ``hexv841_*`` shader per CUDA kernel.  The state stays on the
device for the whole step; the only readback is the four-byte validation flag
at the end, as on CUDA.

This is the dry, global lane.  The regional, physics and halo hooks of the
CUDA step are refused by name until their Metal mirrors exist.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .cuda_driver import CudaDryDycoreDriver, _validate_v841_host_coefficients
from .damping_v841 import build_v841_vertical_velocity_damping
from .errors import ConfigurationRefusal
from .integration import RKSchedule
from .metal_acoustic_v841 import (
    MetalAcousticForcing,
    MetalAcousticState,
    advance_acoustic_step_metal_v841,
    compute_vertical_implicit_coefficients_metal_v841,
    convert_w_tendency_to_omega_metal,
)
from .metal_containers_v841 import MetalAtmosphere, MetalV841Context
from .metal_dynamics_v841 import (
    _add,
    _copy_saved,
    _copy_state,
    _euler_w,
    _recover_candidate,
    _scale,
    _theta_tendency,
    _vector_momentum,
    _vertical_u,
    _w_tendency,
    accumulate_finite_array_metal_v841,
    accumulate_split_flux_metal_v841,
    finish_split_flux_metal_v841,
    validate_recovered_state_metal_v841,
)
from .metal_horizontal_v841 import MetalHorizontalV841
from .metal_transport_v841 import (
    MetalAdvectionCoefficients,
    advance_scalar_transport_metal_v841,
)
from .offcentering_v841 import build_v841_acoustic_offcentering
from .transport import build_advection_coefficients


class MetalDycoreV841:
    """One resident Metal atmosphere and its v8.4.1 dry RK3 step."""

    regional_v841 = None
    halo_exchanger_v841 = None

    def __init__(
        self,
        context: Any,
        atmosphere: MetalAtmosphere,
        config: Any,
        coefficients: MetalAdvectionCoefficients,
        v841_context: MetalV841Context,
    ) -> None:
        from woof.metal.array import MetalArrayModule

        self.context = context
        self.arrays = MetalArrayModule(context)
        self.atmosphere = atmosphere
        self.config = config
        self.coefficients = coefficients
        self.v841_context = v841_context
        self.nlev = int(atmosphere.vertical.n_vert_levels)
        self.ncells = int(atmosphere.mesh.n_cells)
        self.nedges = int(atmosphere.mesh.n_edges)
        self.horizontal = MetalHorizontalV841(
            atmosphere.mesh, self.nlev, v841_context)
        self.mixing_config_v841 = None
        if config.config_horiz_mixing == "2d_smagorinsky":
            from .mixing_v841 import V841MixingConfig

            self.mixing_config_v841 = V841MixingConfig(
                config_horiz_mixing=config.config_horiz_mixing,
                config_len_disp=config.config_len_disp,
                config_visc4_2dsmag=config.config_visc4_2dsmag,
                config_smagorinsky_coef=config.config_smagorinsky_coef,
                config_del4u_div_factor=config.config_del4u_div_factor,
                config_h_ScaleWithMesh=config.config_h_ScaleWithMesh,
                config_mpas_cam_coef=config.config_mpas_cam_coef,
            )
            self.mixing_config_v841.validate()

    @classmethod
    def from_host(
        cls,
        context: Any,
        mesh: Any,
        state: Any,
        vertical: Any,
        reference: Any,
        config: Any,
        *,
        saved_diagnostics: Any,
        terrain_metrics: Any,
        reference_wind_profiles: Any,
        advection_coefficients: Any | None = None,
    ) -> "MetalDycoreV841":
        """Upload every field once, after the CUDA lane's admission."""

        config.validate()
        CudaDryDycoreDriver._validate_config(config)
        if getattr(config, "source_release", None) != "v8.4.1":
            raise ConfigurationRefusal(
                "source_release", getattr(config, "source_release", None),
                "the resident Metal dycore mirrors the v8.4.1 CUDA lane",
                "V841DryDycoreConfig")
        if saved_diagnostics is None:
            raise ConfigurationRefusal(
                "saved_diagnostics", None,
                "the step starts from the init's exact time-level-one sidecar",
                "load_mpas_initial_state(..., return_saved_diagnostics=True)")
        if terrain_metrics is None:
            raise ConfigurationRefusal(
                "terrain_metrics", None,
                "the omega conversion reads zb_cell and zb3_cell",
                "the init's native terrain metrics")
        if np.dtype(state.rho.dtype) != np.dtype(np.float32):
            raise ConfigurationRefusal(
                "state.rho", state.rho.dtype.str,
                "the v8.4.1 Metal dycore is float32", "float32 state")
        nlev = int(state.rho.shape[0])
        reference_wind_profiles.validate(nlev, np.dtype(np.float32))
        offcentering = build_v841_acoustic_offcentering(
            np.asarray(vertical.rdzw),
            minimum=config.config_epssm_minimum,
            maximum=config.config_epssm_maximum,
            transition_bottom_z=config.config_epssm_transition_bottom_z,
            transition_top_z=config.config_epssm_transition_top_z,
        )
        coefficients = advection_coefficients
        if coefficients is None:
            coefficients = build_advection_coefficients(
                mesh,
                config_scalar_adv_order=config.config_scalar_adv_order,
                n_vert_levels=nlev,
                source_order_v841=True,
            )
        _validate_v841_host_coefficients(
            coefficients,
            n_cells=int(state.rho.shape[1]),
            n_edges=int(state.rho_u.shape[1]),
        )
        # As on CUDA: the init's dss is a placeholder, so rebuild it from
        # config_xnutr/config_zd, and round cf1..cf3 to the float32 RKIND.
        selected_vertical = replace(
            vertical,
            dss=build_v841_vertical_velocity_damping(
                np.asarray(vertical.zgrid),
                xnutr=config.config_xnutr,
                damping_start_z=config.config_zd,
            ),
            **{name: float(np.float32(getattr(vertical, name)))
               for name in ("cf1", "cf2", "cf3")},
        )
        atmosphere = MetalAtmosphere.from_host(
            context, mesh, state, selected_vertical, reference,
            saved_diagnostics, terrain_metrics)
        v841_context = MetalV841Context.from_host(
            context, mesh, offcentering, reference_wind_profiles,
            n_vert_levels=nlev)
        driver = cls(
            context, atmosphere, config,
            MetalAdvectionCoefficients.from_host(context, coefficients),
            v841_context,
        )
        if driver.mixing_config_v841 is not None:
            from .mixing_v841 import initialize_deformation_weights_v841

            driver.horizontal.attach_deformation_weights_v841(
                initialize_deformation_weights_v841(mesh, dtype=np.float32))
        return driver

    def _vertical_coefficients(
        self, state: Any, dts: float, theta_m: Any, rtheta_p: Any, *,
        exner: Any, validation_flag: Any,
    ) -> Any:
        vertical = self.atmosphere.vertical
        reference = self.atmosphere.reference
        return compute_vertical_implicit_coefficients_metal_v841(
            dts=dts,
            context=self.v841_context,
            zz=vertical.zz,
            cqw=self.arrays.ones(state.rho.shape, np.float32),
            exner=exner,
            theta=theta_m,
            rho_base=reference.rho_base,
            rho_theta_base=reference.rho_theta_base,
            exner_base=reference.exner_base,
            rho_theta_perturbation=rtheta_p,
            qtot=self.arrays.zeros(state.rho.shape, np.float32),
            rdzw=vertical.rdzw,
            fzm=vertical.fzm,
            fzp=vertical.fzp,
            rdzu=vertical.rdzu,
            validation_flag=validation_flag,
        )

    def _advance_dynamics_subcycle(
        self, state: Any, *, time_level_one: Any, initial_diagnostics: Any,
        schedule: RKSchedule, outer_dt: float, validation_flag: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        if schedule.full_timestep != outer_dt or schedule.dynamics_splits != 3:
            raise ValueError("the v8.4.1 subcycle requires the outer split-three schedule")
        vertical = self.atmosphere.vertical
        saved_state = _copy_state(self, state, share_scalars=True)
        saved_diag = _copy_saved(self, time_level_one)
        current_state = saved_state
        current_diag = saved_diag
        diagnostics = initial_diagnostics
        cached_v = diagnostics.tangential_velocity

        coefficients = self._vertical_coefficients(
            saved_state, schedule.stages[0].acoustic_timestep,
            saved_diag.theta_m, saved_diag.rho_theta_perturbation,
            exner=saved_diag.exner, validation_flag=validation_flag)
        cqu = self.arrays.ones(saved_state.rho_u.shape, np.float32)
        dpdz = _scale(self, saved_diag.density_perturbation, -9.80616)
        saved_euler_ru = self.horizontal.pressure_gradient_euler_tendency(
            saved_diag.pressure_perturbation, dpdz, cqu, vertical.zz, vertical.zxu)
        tend_rho_saved = self.horizontal.density_tendency(
            saved_state.rho_u, saved_state.rho_w, vertical.rdzw)
        saved_euler_rw = _euler_w(self, saved_diag.pressure_perturbation, dpdz)
        saved_euler_theta = None
        if self.mixing_config_v841 is not None:
            # Native forms the mixing Euler tendencies once per dynamics
            # substep at rk_step 1 and reuses them for RK2 and RK3.
            saved_mixing = self.horizontal.compute_dry_mixing_tendencies_v841(
                saved_diag.normal_velocity,
                diagnostics.tangential_velocity,
                saved_diag.vertical_velocity,
                saved_diag.theta_m,
                diagnostics.h_edge,
                diagnostics.divergence,
                diagnostics.vorticity,
                dt=outer_dt,
                config=self.mixing_config_v841,
            )
            _add(self, saved_euler_ru, saved_mixing.tend_u_euler)
            _add(self, saved_euler_rw, saved_mixing.tend_w_euler)
            saved_euler_theta = saved_mixing.tend_theta_euler.copy()

        final_flux_u = final_flux_w = None
        for stage in schedule.stages:
            if stage.stage == 2:
                coefficients = self._vertical_coefficients(
                    current_state, stage.acoustic_timestep,
                    current_diag.theta_m, current_diag.rho_theta_perturbation,
                    exner=saved_diag.exner, validation_flag=validation_flag)
            mass_divergence = self.horizontal.mass_flux_divergence(
                current_state.rho_u)
            tend_ru = _vertical_u(self, current_diag.normal_velocity,
                                  current_state.rho_w)
            _add(self, tend_ru, _vector_momentum(
                self, current_diag.normal_velocity, diagnostics.h_edge,
                diagnostics.pv_edge, diagnostics.kinetic_energy, mass_divergence))
            _add(self, tend_ru, saved_euler_ru)
            tend_rt = _theta_tendency(
                self, current_state, current_diag.theta_m, saved_state,
                saved_diag.theta_m, stage.stage)
            if saved_euler_theta is not None:
                _add(self, tend_rt, saved_euler_theta)
            tend_rw = _w_tendency(self, current_state,
                                  current_diag.vertical_velocity)
            _add(self, tend_rw, saved_euler_rw)
            tend_omega = convert_w_tendency_to_omega_metal(
                self.atmosphere.mesh, tend_rw, tend_ru,
                fzm=vertical.fzm, fzp=vertical.fzp, zz=vertical.zz,
                zb_cell=self.atmosphere.terrain.zb_cell,
                zb3_cell=self.atmosphere.terrain.zb3_cell,
            )
            acoustic = MetalAcousticState.zeros(
                self.arrays, self.nlev, self.ncells, self.nedges)
            forcing = MetalAcousticForcing(
                rho_zz=current_state.rho,
                theta_m=saved_diag.theta_m,
                zz=vertical.zz,
                exner=saved_diag.exner,
                cqu=cqu,
                zxu=vertical.zxu,
                dss=vertical.dss,
                tend_ru=tend_ru,
                tend_rho=tend_rho_saved,
                tend_rt=tend_rt,
                tend_rw=tend_omega,
                w=current_diag.vertical_velocity,
                rw=current_state.rho_w,
                rw_save=saved_state.rho_w,
            )
            for small_step in range(1, stage.acoustic_steps + 1):
                rtheta_pp_old = self.horizontal.capture_rtheta_pp_old(
                    acoustic.rtheta_pp, small_step=small_step)
                acoustic = advance_acoustic_step_metal_v841(
                    self.atmosphere.mesh, acoustic, forcing, coefficients,
                    context=self.v841_context,
                    dts=stage.acoustic_timestep,
                    small_step=small_step,
                    fzm=vertical.fzm, fzp=vertical.fzp, rdzw=vertical.rdzw,
                    in_place=True,
                )
                # Native :2395-2406: 3-D divergence damping completes the
                # horizontal momentum at the end of each acoustic sub-step.
                acoustic.ru_p = self.horizontal.divergence_damping_3d(
                    acoustic.ru_p,
                    saved_diag.theta_m,
                    acoustic.rtheta_pp,
                    rtheta_pp_old,
                    dts=stage.acoustic_timestep,
                    config_smdiv=self.config.config_smdiv,
                    config_len_disp=self.config.config_len_disp,
                    in_place=True,
                )
            current_state, current_diag, flux_u, flux_w = _recover_candidate(
                self, saved_state, saved_diag, acoustic, stage.stage,
                stage.acoustic_steps, current_state.scalars)
            validate_recovered_state_metal_v841(
                current_state, current_diag, invalid_flag=validation_flag)
            if stage.stage == 3:
                final_flux_u, final_flux_w = flux_u, flux_w
            diagnostics = self.horizontal.solve_diagnostics(
                current_state.rho,
                current_state.rho_u,
                dt=outer_dt,
                apvm_upwinding=self.config.config_apvm_upwinding,
                normal_velocity=current_diag.normal_velocity,
                cached_tangential_velocity=cached_v,
                rk_step=stage.stage,
            )
            cached_v = diagnostics.tangential_velocity
        if final_flux_u is None or final_flux_w is None:
            raise RuntimeError("v8.4.1 dynamics subcycle did not reach RK3")
        return current_state, current_diag, diagnostics, final_flux_u, final_flux_w

    def step_device(self) -> None:
        """Advance the resident atmosphere one outer step."""

        config = self.config
        validation_flag = self.arrays.zeros((1,), np.int32)
        outer_saved = _copy_state(self, self.atmosphere.state)
        time_level_one = _copy_saved(self, self.atmosphere.saved)
        outer_dt = float(config.config_dt)
        splits = int(config.config_dynamics_split_steps)
        dynamics_schedule = RKSchedule.from_mpas(
            outer_dt, order=config.config_time_integration_order,
            acoustic_substeps=config.config_number_of_sub_steps,
            dynamics_splits=splits)
        scalar_schedule = RKSchedule.from_mpas(
            outer_dt, order=config.config_time_integration_order,
            acoustic_substeps=config.config_number_of_sub_steps,
            dynamics_splits=1)
        diagnostics = self.horizontal.solve_diagnostics(
            outer_saved.rho, outer_saved.rho_u, dt=outer_dt,
            apvm_upwinding=config.config_apvm_upwinding,
            normal_velocity=time_level_one.normal_velocity, rk_step=3)
        current, carried = outer_saved, time_level_one
        split_flux_u_sum = split_flux_w_sum = None
        for _subcycle in range(splits):
            current, carried, diagnostics, flux_u, flux_w = (
                self._advance_dynamics_subcycle(
                    current, time_level_one=carried,
                    initial_diagnostics=diagnostics,
                    schedule=dynamics_schedule, outer_dt=outer_dt,
                    validation_flag=validation_flag))
            split_flux_u_sum = accumulate_split_flux_metal_v841(
                flux_u, split_flux_u_sum)
            split_flux_w_sum = accumulate_split_flux_metal_v841(
                flux_w, split_flux_w_sum)
        split_flux_u = finish_split_flux_metal_v841(split_flux_u_sum, splits)
        split_flux_w = finish_split_flux_metal_v841(split_flux_w_sum, splits)

        if config.config_scalar_advection and current.scalars.shape[0] > 0:
            vertical = self.atmosphere.vertical
            scalar_stage = outer_saved.scalars.copy()
            for stage in scalar_schedule.stages:
                transported = advance_scalar_transport_metal_v841(
                    self.atmosphere.mesh, self.v841_context,
                    outer_saved.scalars, scalar_stage,
                    outer_saved.rho, current.rho,
                    split_flux_u, split_flux_w, stage.large_timestep,
                    coefficients=self.coefficients,
                    fzm=vertical.fzm, fzp=vertical.fzp, rdzw=vertical.rdzw,
                    rk_step=stage.stage,
                    config_scalar_advection=True,
                    config_monotonic=config.config_monotonic,
                    config_positive_definite=config.config_positive_definite,
                    config_split_dynamics_transport=True,
                    config_time_integration_order=3,
                    config_scalar_adv_order=3,
                    config_scalar_vadv_order=3,
                    config_coef_3rd_order=config.config_coef_3rd_order,
                    validation_flag=validation_flag,
                    indices_prevalidated=True,
                    rho_old_prevalidated=True,
                )
                scalar_stage = transported.scalars
                accumulate_finite_array_metal_v841(
                    scalar_stage, invalid_flag=validation_flag)
            current.scalars = scalar_stage

        if int(validation_flag.to_numpy()[0]) != 0:
            raise FloatingPointError(
                "v8.4.1 Metal validation flag refused the outer step before publish")
        current.time_seconds = outer_saved.time_seconds + outer_dt
        self.atmosphere = replace(self.atmosphere, state=current, saved=carried)

    def step(self) -> tuple[Any, Any]:
        """Advance one step and return the host state and saved diagnostics."""

        self.step_device()
        return self.atmosphere.state.to_host(), self.atmosphere.saved.to_host()


__all__ = ["MetalDycoreV841"]
