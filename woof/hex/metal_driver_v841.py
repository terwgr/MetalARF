"""Resident v8.4.1 Hex dycore on Metal.

``MetalDycoreV841`` runs ``CudaDryDycoreDriver._step_device_v841`` and
``_advance_dynamics_subcycle_v841`` call for call on resident ``MetalArray``
fields, with one ``hexv841_*`` shader per CUDA kernel.  The state stays on the
device for the whole step; the only readback is the four-byte validation flag
at the end, as on CUDA.

The dry lane runs global or limited-area (``regional_v841`` attached by
:func:`assemble_regional_metal_dycore_v841`).  The full-physics lane follows
CUDA's transaction: :meth:`MetalDycoreV841.step_device_with_physics` advances
to an uncommitted candidate with the held phase-one tendencies, and
:meth:`MetalDycoreV841.commit_post_wsm6_candidate` publishes it after WSM6.
The halo hooks of the CUDA step are refused by name.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    rebuild_saved_diagnostics_metal_v841,
    validate_recovered_state_metal_v841,
)
from .metal_horizontal_v841 import MetalHorizontalV841
from .metal_moist_v841 import (
    compute_moist_dynamics_coefficients_metal_v841,
    gf_dry_theta_dynamics_rate_metal_v841,
    modified_theta_dynamics_rate_metal_v841,
    moist_dpdz_metal_v841,
    moist_euler_w_metal_v841,
)
from .metal_transport_v841 import (
    MetalAdvectionCoefficients,
    advance_scalar_transport_metal_v841,
)
from .offcentering_v841 import build_v841_acoustic_offcentering
from .species_row import species_row_for_scheme
from .transport import build_advection_coefficients


@dataclass(frozen=True)
class MetalGfDynamicsTendenciesV841:
    """``CudaV841GfDynamicsTendencies``: endpoint dry-theta and qv rates."""

    rthdynten: Any
    rqvdynten: Any
    time_seconds: float


@dataclass(frozen=True)
class MetalPhysicsStepCandidateV841:
    """An uncommitted full-physics endpoint awaiting post-WSM6 commit."""

    atmosphere: MetalAtmosphere
    dynamics_tendencies: MetalGfDynamicsTendenciesV841
    start_time_seconds: float


@dataclass(frozen=True)
class MetalCommittedPhysicsStepV841:
    atmosphere: MetalAtmosphere
    dynamics_tendencies: MetalGfDynamicsTendenciesV841
    surface_updates: Any
    effective_radii: Any


def _full_physics(config: Any) -> bool:
    from .config_v841 import V841MpasColumnPhysicsConfig

    return isinstance(config, V841MpasColumnPhysicsConfig)


class MetalDycoreV841:
    """One resident Metal atmosphere and its v8.4.1 dry RK3 step."""

    regional_v841 = None
    halo_exchanger_v841 = None
    _pending_candidate = None

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
        exner: Any, validation_flag: Any, moist: Any | None = None,
    ) -> Any:
        vertical = self.atmosphere.vertical
        reference = self.atmosphere.reference
        cqw = (self.arrays.ones(state.rho.shape, np.float32)
               if moist is None else moist.cqw)
        qtot = (self.arrays.zeros(state.rho.shape, np.float32)
                if moist is None else moist.qtot)
        return compute_vertical_implicit_coefficients_metal_v841(
            dts=dts,
            context=self.v841_context,
            zz=vertical.zz,
            cqw=cqw,
            exner=exner,
            theta=theta_m,
            rho_base=reference.rho_base,
            rho_theta_base=reference.rho_theta_base,
            exner_base=reference.exner_base,
            rho_theta_perturbation=rtheta_p,
            qtot=qtot,
            rdzw=vertical.rdzw,
            fzm=vertical.fzm,
            fzp=vertical.fzp,
            rdzu=vertical.rdzu,
            validation_flag=validation_flag,
        )

    def _advance_dynamics_subcycle(
        self, state: Any, *, time_level_one: Any, initial_diagnostics: Any,
        schedule: RKSchedule, outer_dt: float, validation_flag: Any,
        physics: Any | None = None, moist: Any | None = None,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
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
            exner=saved_diag.exner, validation_flag=validation_flag,
            moist=moist)
        if moist is None:
            cqu = self.arrays.ones(saved_state.rho_u.shape, np.float32)
            dpdz = _scale(self, saved_diag.density_perturbation, -9.80616)
        else:
            cqu = moist.cqu
            dpdz = moist_dpdz_metal_v841(
                self, saved_diag.density_perturbation, moist.qtot,
                validation_flag)
        saved_euler_ru = self.horizontal.pressure_gradient_euler_tendency(
            saved_diag.pressure_perturbation, dpdz, cqu, vertical.zz, vertical.zxu)
        tend_rho_dynamics = self.horizontal.density_tendency(
            saved_state.rho_u, saved_state.rho_w, vertical.rdzw)
        tend_rho_saved = (
            tend_rho_dynamics if physics is None
            else self.horizontal.density_tendency(
                saved_state.rho_u, saved_state.rho_w, vertical.rdzw,
                physics_tendency=physics.rho))
        if self.regional_v841 is not None:
            # Native writes tend_rho only at rk_step 1 and the regional stages
            # adjust that pool again in RK2 and RK3; keep a private image.
            tend_rho_saved = tend_rho_saved.copy()
        if moist is None:
            saved_euler_rw = _euler_w(self, saved_diag.pressure_perturbation, dpdz)
        else:
            saved_euler_rw = moist_euler_w_metal_v841(
                self, saved_diag.pressure_perturbation, dpdz, moist.cqw,
                validation_flag)
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
        modified_theta_rate = None
        for stage in schedule.stages:
            if stage.stage == 2:
                coefficients = self._vertical_coefficients(
                    current_state, stage.acoustic_timestep,
                    current_diag.theta_m, current_diag.rho_theta_perturbation,
                    exner=saved_diag.exner, validation_flag=validation_flag,
                    moist=moist)
            mass_divergence = self.horizontal.mass_flux_divergence(
                current_state.rho_u)
            tend_ru = _vertical_u(self, current_diag.normal_velocity,
                                  current_state.rho_w)
            _add(self, tend_ru, _vector_momentum(
                self, current_diag.normal_velocity, diagnostics.h_edge,
                diagnostics.pv_edge, diagnostics.kinetic_energy, mass_divergence))
            _add(self, tend_ru, saved_euler_ru)
            if physics is not None:
                _add(self, tend_ru, physics.rho_u)
            tend_rt = _theta_tendency(
                self, current_state, current_diag.theta_m, saved_state,
                saved_diag.theta_m, stage.stage)
            if physics is not None and stage.stage == 3:
                # The GF dynamics rate reads tend_theta before the mixing and
                # physics increments land (native F:6941 then F:6948).
                modified_theta_rate = modified_theta_dynamics_rate_metal_v841(
                    self, tend_rt, tend_rho_dynamics, current_state.rho,
                    current_diag.theta_m, validation_flag)
            if saved_euler_theta is not None:
                _add(self, tend_rt, saved_euler_theta)
            if physics is not None:
                _add(self, tend_rt, physics.rho_theta)
            tend_rw = _w_tendency(self, current_state,
                                  current_diag.vertical_velocity)
            _add(self, tend_rw, saved_euler_rw)
            tend_omega = convert_w_tendency_to_omega_metal(
                self.atmosphere.mesh, tend_rw, tend_ru,
                fzm=vertical.fzm, fzp=vertical.fzp, zz=vertical.zz,
                zb_cell=self.atmosphere.terrain.zb_cell,
                zb3_cell=self.atmosphere.terrain.zb3_cell,
            )
            if self.regional_v841 is not None:
                # atm_srk3:2300-2351: specified-zone assignment and the
                # relaxation-zone Rayleigh/Laplacian stages.
                self.regional_v841.adjust_dynamics_tendencies(
                    tend_ru=tend_ru, tend_rho=tend_rho_saved, tend_rt=tend_rt,
                    tend_omega=tend_omega, rho_u=current_state.rho_u,
                    theta_m=current_diag.theta_m, rho_zz=current_state.rho,
                    rk_step=stage.stage)
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
                if self.regional_v841 is None:
                    acoustic = advance_acoustic_step_metal_v841(
                        self.atmosphere.mesh, acoustic, forcing, coefficients,
                        context=self.v841_context,
                        dts=stage.acoustic_timestep,
                        small_step=small_step,
                        fzm=vertical.fzm, fzp=vertical.fzp, rdzw=vertical.rdzw,
                        in_place=True,
                    )
                else:
                    from .metal_regional_v841 import (
                        advance_acoustic_step_regional_metal_v841,
                    )

                    acoustic = advance_acoustic_step_regional_metal_v841(
                        self.regional_v841, self.atmosphere.mesh, acoustic,
                        forcing, coefficients, context=self.v841_context,
                        dts=stage.acoustic_timestep, small_step=small_step,
                        fzm=vertical.fzm, fzp=vertical.fzp, rdzw=vertical.rdzw)
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
                    **({} if self.regional_v841 is None else
                       {"n_cells_solve": self.regional_v841.n_cells_solve}),
                )
            current_state, current_diag, flux_u, flux_w = _recover_candidate(
                self, saved_state, saved_diag, acoustic, stage.stage,
                stage.acoustic_steps, current_state.scalars)
            if self.regional_v841 is None:
                validate_recovered_state_metal_v841(
                    current_state, current_diag, invalid_flag=validation_flag)
            else:
                self.regional_v841.validate_recovered(
                    current_state, current_diag, validation_flag)
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
        return (current_state, current_diag, diagnostics, final_flux_u,
                final_flux_w, modified_theta_rate)

    def _advance(self, physics: Any | None) -> tuple[Any, Any, Any, Any]:
        """``_step_device_v841`` up to the four-byte validation readback."""

        config = self.config
        validation_flag = self.arrays.zeros((1,), np.int32)
        regional = self.regional_v841
        if regional is not None:
            # mpas_atm_core.F:735-781: read lbc_in and re-form the boundary
            # tendencies at the start of a step whose clock reached an
            # interval end, before any dynamics.
            regional.begin_step(self.atmosphere.state)
        outer_saved = _copy_state(self, self.atmosphere.state)
        time_level_one = _copy_saved(self, self.atmosphere.saved)
        if regional is not None:
            regional.bind_state_rho(outer_saved.rho)
            if config.config_moist_physics or physics is not None:
                raise ConfigurationRefusal(
                    "config_moist_physics", True,
                    "the regional moist coefficients join with the physics lane",
                    "config_moist_physics=False")
        moist = None
        if physics is not None:
            row = species_row_for_scheme(config.config_microp_scheme)
            if int(outer_saved.scalars.shape[0]) != row.n_species():
                raise ValueError(
                    f"{row.scheme_name!r} declares {row.n_species()} scalars "
                    f"and the state carries {int(outer_saved.scalars.shape[0])}")
            physics.validate(n_vert_levels=self.nlev, n_cells=self.ncells,
                             n_edges=self.nedges, n_scalars=row.n_species())
            if float(physics.time_seconds) != float(outer_saved.time_seconds):
                raise ValueError(
                    "held physics time must equal the candidate start exactly: "
                    f"{physics.time_seconds} != {outer_saved.time_seconds}")
            moist = compute_moist_dynamics_coefficients_metal_v841(
                self, outer_saved.scalars, scalar_names=row.names(),
                validation_flag=validation_flag)
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
        modified_theta_rate = None
        for _subcycle in range(splits):
            if regional is not None:
                regional.begin_dynamics_substep(_subcycle + 1)
            (current, carried, diagnostics, flux_u, flux_w,
             modified_theta_rate) = self._advance_dynamics_subcycle(
                current, time_level_one=carried,
                initial_diagnostics=diagnostics,
                schedule=dynamics_schedule, outer_dt=outer_dt,
                validation_flag=validation_flag, physics=physics, moist=moist)
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
                if regional is not None:
                    from .metal_regional_v841 import (
                        advance_scalars_regional_metal_v841,
                    )

                    scalar_stage, _target = advance_scalars_regional_metal_v841(
                        regional, self.atmosphere.mesh, self.v841_context,
                        outer_saved.scalars, scalar_stage, outer_saved.rho,
                        current.rho, split_flux_u, split_flux_w,
                        stage.large_timestep, coefficients=self.coefficients,
                        fzm=vertical.fzm, fzp=vertical.fzp, rdzw=vertical.rdzw,
                        rk_step=stage.stage,
                        config_coef_3rd_order=config.config_coef_3rd_order,
                        validation_flag=validation_flag)
                    regional.bdy_adjust_scalars(scalar_stage, stage.stage)
                    accumulate_finite_array_metal_v841(
                        scalar_stage, invalid_flag=validation_flag)
                    continue
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
                    **({} if physics is None
                       else {"scalar_tendency": physics.scalars}),
                )
                scalar_stage = transported.scalars
                accumulate_finite_array_metal_v841(
                    scalar_stage, invalid_flag=validation_flag)
            current.scalars = scalar_stage
        if regional is not None:
            # atm_srk3:2798-2878 in order: the clamp, the specified-zone
            # resets on the pre-advance clock, then the clock advance.
            if config.config_moist_physics:
                regional.clamp_negative_scalars(current.scalars)
            regional.reset_speczone_values(
                theta_m=carried.theta_m, rho_theta=current.rho_theta,
                rho_theta_perturbation=carried.rho_theta_perturbation,
                scalars=current.scalars)
            regional.advance_clock(outer_dt)

        rates = None
        if physics is not None:
            if modified_theta_rate is None:
                raise RuntimeError(
                    "full-physics split loop did not capture final RK dynamics heating")
            rates = gf_dry_theta_dynamics_rate_metal_v841(
                self, modified_theta_rate, carried.theta_m, current.scalars[0],
                validation_flag)
        if int(validation_flag.to_numpy()[0]) != 0:
            raise FloatingPointError(
                "v8.4.1 Metal validation flag refused the outer step before publish")
        current.time_seconds = outer_saved.time_seconds + outer_dt
        return current, carried, outer_saved, rates

    def step_device(self) -> None:
        """Advance the resident atmosphere one outer step."""

        if _full_physics(self.config):
            raise ConfigurationRefusal(
                "held_physics_tendencies", None,
                "the full-physics lane must compute and validate phase one before RK",
                "step_device_with_physics(MetalHeldMpasPhysicsTendenciesV841)")
        current, carried, _start, _rates = self._advance(None)
        self.atmosphere = replace(self.atmosphere, state=current, saved=carried)

    def step_device_with_physics(
        self, physics: Any
    ) -> MetalPhysicsStepCandidateV841:
        """Advance to an uncommitted endpoint using one held phase-one source."""

        if not _full_physics(self.config):
            raise ConfigurationRefusal(
                "config_physics_suite", self.config.config_physics_suite,
                "held column tendencies require the distinct full-physics config",
                "V841MpasColumnPhysicsConfig()")
        if self._pending_candidate is not None:
            raise RuntimeError(
                "a prior full-physics candidate still awaits post-WSM6 commit")
        current, carried, start, rates = self._advance(physics)
        candidate = MetalPhysicsStepCandidateV841(
            atmosphere=replace(self.atmosphere, state=current, saved=carried),
            dynamics_tendencies=MetalGfDynamicsTendenciesV841(
                rthdynten=rates[0], rqvdynten=rates[1],
                time_seconds=current.time_seconds),
            start_time_seconds=start.time_seconds,
        )
        self._pending_candidate = candidate
        return candidate

    def commit_post_wsm6_candidate(
        self, candidate: MetalPhysicsStepCandidateV841, recovery: Any
    ) -> MetalCommittedPhysicsStepV841:
        """Publish only the externally validated phase-two resident endpoint."""

        if candidate is not self._pending_candidate:
            raise ValueError("candidate is not the driver's pending transaction")
        endpoint = float(candidate.atmosphere.state.time_seconds)
        if endpoint != float(recovery.time_seconds):
            raise ValueError(
                "post-WSM6 recovery time must equal the candidate endpoint exactly")
        if recovery.state is not candidate.atmosphere.state:
            raise ValueError(
                "post-WSM6 recovery must validate the pending candidate in place")
        if float(self.atmosphere.state.time_seconds) != float(
                candidate.start_time_seconds):
            raise ValueError("resident driver start state changed during phase two")
        committed = replace(
            candidate.atmosphere, state=recovery.state,
            saved=rebuild_saved_diagnostics_metal_v841(self, recovery.state))
        flag = self.arrays.zeros((1,), np.int32)
        if self.regional_v841 is None:
            validate_recovered_state_metal_v841(
                committed.state, committed.saved, invalid_flag=flag)
        else:
            self.regional_v841.validate_recovered(
                committed.state, committed.saved, flag)
        accumulate_finite_array_metal_v841(committed.state.scalars, invalid_flag=flag)
        if int(flag.to_numpy()[0]) != 0:
            raise FloatingPointError(
                "final rebuilt v8.4.1 full-physics state failed numeric validation")
        self.atmosphere = committed
        self._pending_candidate = None
        return MetalCommittedPhysicsStepV841(
            atmosphere=committed,
            dynamics_tendencies=candidate.dynamics_tendencies,
            surface_updates=recovery.surface_updates,
            effective_radii=recovery.effective_radii,
        )

    def abort_post_wsm6_candidate(
        self, candidate: MetalPhysicsStepCandidateV841
    ) -> None:
        """Discard exactly the pending candidate without publishing any state."""

        if self._pending_candidate is None or candidate is not self._pending_candidate:
            raise ValueError("candidate is not the driver's pending transaction")
        self._pending_candidate = None

    def step(self) -> tuple[Any, Any]:
        """Advance one step and return the host state and saved diagnostics."""

        self.step_device()
        return self.atmosphere.state.to_host(), self.atmosphere.saved.to_host()



def assemble_regional_metal_dycore_v841(
    context: Any,
    mesh: Any,
    state: Any,
    vertical: Any,
    reference: Any,
    saved: Any,
    terrain: Any,
    config: Any,
    *,
    reference_wind_profiles: Any,
    lbc_paths: Any,
    start_time: Any,
    scalar_names: tuple[str, ...] = ("lbc_qv",),
) -> MetalDycoreV841:
    """Build the limited-area resident dycore, as ``assemble_regional_driver_v841``.

    The atmosphere and the v8.4.1 context use native's padded memory model,
    and the regional runtime's garbage discipline runs after every launch.
    """

    from . import metal_v841_launch
    from .cuda_regional_forecast_v841 import (
        PaddedRegionalHostMesh,
        _PaddedDeformationWeights,
        regional_padded_advection_coefficients,
    )
    from .metal_regional_v841 import (
        MetalRegionalRuntimeV841,
        build_regional_metal_atmosphere,
        regional_padded_metal_context_v841,
    )
    from .regional_v841 import derive_regional_masks

    config.validate()
    CudaDryDycoreDriver._validate_config(config)
    if config.config_moist_physics:
        raise ConfigurationRefusal(
            "config_moist_physics", True,
            "the regional moist coefficients join with the physics lane",
            "config_moist_physics=False")
    dtype = np.dtype(np.float32)
    padded = PaddedRegionalHostMesh(mesh, derive_regional_masks(mesh, dtype))
    nlev = int(np.asarray(state.rho).shape[0])
    offcentering = build_v841_acoustic_offcentering(
        np.asarray(vertical.rdzw),
        minimum=config.config_epssm_minimum,
        maximum=config.config_epssm_maximum,
        transition_bottom_z=config.config_epssm_transition_bottom_z,
        transition_top_z=config.config_epssm_transition_top_z,
    )
    # dss is built on the unpadded zgrid and padded, as on CUDA.
    selected_dss = build_v841_vertical_velocity_damping(
        np.asarray(vertical.zgrid),
        xnutr=config.config_xnutr,
        damping_start_z=config.config_zd,
    )

    class _VerticalWithDss:
        def __getattr__(self, name: str) -> Any:
            if name == "dss":
                return selected_dss
            if name in ("cf1", "cf2", "cf3"):
                return float(np.float32(getattr(vertical, name)))
            return getattr(vertical, name)

    atmosphere = build_regional_metal_atmosphere(
        context, padded, state, _VerticalWithDss(), reference, saved, terrain)
    v841_context = regional_padded_metal_context_v841(
        context, mesh, offcentering, reference_wind_profiles,
        n_vert_levels=nlev)
    host_coefficients = build_advection_coefficients(
        mesh,
        config_scalar_adv_order=config.config_scalar_adv_order,
        n_vert_levels=nlev,
        source_order_v841=True,
        allow_regional_sentinels=True,
    )
    coefficients = MetalAdvectionCoefficients.from_host(
        context, regional_padded_advection_coefficients(
            host_coefficients, padded.garbage_cell))
    driver = MetalDycoreV841(context, atmosphere, config, coefficients, v841_context)
    if driver.mixing_config_v841 is not None:
        from .mixing_v841 import initialize_deformation_weights_v841

        driver.horizontal.attach_deformation_weights_v841(
            _PaddedDeformationWeights(
                initialize_deformation_weights_v841(mesh, dtype=dtype)))
    runtime = MetalRegionalRuntimeV841(
        mesh, padded,
        context=context,
        lbc_paths=lbc_paths,
        start_time=start_time,
        config_h_scale_with_mesh=bool(config.config_h_ScaleWithMesh),
        config_apply_lbcs=bool(config.config_apply_lbcs),
        outer_dt=float(config.config_dt),
        dynamics_split=int(config.config_dynamics_split_steps),
        n_vert_levels=nlev,
        device_atmosphere=atmosphere,
        v841_context=v841_context,
        scalar_names=tuple(scalar_names),
    )
    driver.regional_v841 = runtime
    # CUDA: kernel_cache.post_launch = discipline.scrub.
    metal_v841_launch.POST_LAUNCH = runtime.discipline.scrub
    return driver


__all__ = ["MetalDycoreV841", "assemble_regional_metal_dycore_v841"]
