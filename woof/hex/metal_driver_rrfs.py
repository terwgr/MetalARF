"""Resident retained RRFS split-explicit dynamics and scalar composition."""
from __future__ import annotations
from dataclasses import fields, replace
from typing import Any
import numpy as np
from .metal_driver_v841 import MetalDycoreV841
from .metal_dynamics_v841 import (_copy_state, _copy_saved, _scale, _add,
    _euler_w, _vertical_u, _theta_tendency, _w_tendency, _recover_candidate,
    accumulate_finite_array_metal_v841, accumulate_split_flux_metal_v841,
    finish_split_flux_metal_v841, validate_recovered_state_metal_v841)
from .metal_acoustic_v841 import MetalAcousticState, MetalAcousticForcing, convert_w_tendency_to_omega_metal
from .metal_acoustic_rrfs import compute_vertical_implicit_coefficients_metal_rrfs, advance_acoustic_step_metal_rrfs
from .metal_moist_v841 import (moist_dpdz_metal_v841, moist_euler_w_metal_v841,
    modified_theta_dynamics_rate_metal_v841, compute_moist_dynamics_coefficients_metal_v841,
    gf_dry_theta_dynamics_rate_metal_v841)
from .metal_horizontal_rrfs import MetalHorizontalRRFS
from .metal_rrfs_launch import launch
from .metal_transport_rrfs import advance_scalars
from .integration import RKSchedule
from .species_row import species_row_for_scheme


class MetalRRFSDycore(MetalDycoreV841):
    def __init__(self, *args):
        super().__init__(*args)
        from .mixing_rrfs import RRFSMixingConfig
        self.horizontal = MetalHorizontalRRFS(self.atmosphere.mesh,self.nlev,self.v841_context)
        self.mixing_config_v841 = RRFSMixingConfig(**{f.name:getattr(self.config,f.name) for f in fields(RRFSMixingConfig)})

    @staticmethod
    def _validate_config(config):
        from .cuda_driver_rrfs import CudaRRFSDycoreDriver
        CudaRRFSDycoreDriver._validate_config(config)
        unsupported = [name for name in ('config_v_mom_eddy_visc2','config_v_theta_eddy_visc2',
            'config_scalar_eddy_mix','config_h_scalar_filter4','config_rayleigh_damp_u','config_mpas_cam_coef')
            if getattr(config,name)]
        if unsupported:
            raise ValueError('Metal RRFS does not implement '+', '.join(unsupported))
        if not config.config_split_dynamics_transport:
            raise ValueError('Metal RRFS requires config_split_dynamics_transport=true')

    @staticmethod
    def _vertical_damping(mesh, vertical, config):
        from .damping_rrfs import build_rrfs_vertical_velocity_damping
        return build_rrfs_vertical_velocity_damping(np.asarray(vertical.zgrid),np.asarray(mesh.mesh_density),
            xnutr=config.config_xnutr,damping_start_z=config.config_zd)

    def _vertical_coefficients(self,state,dts,theta_m,rtheta_p,*,exner,validation_flag,cqw=None,qtot=None):
        vertical,reference = self.atmosphere.vertical,self.atmosphere.reference
        return compute_vertical_implicit_coefficients_metal_rrfs(dts=dts,epssm=self.config.config_epssm,
            zz=vertical.zz,cqw=self.arrays.ones(state.rho.shape,np.float32) if cqw is None else cqw,
            pressure=exner,theta=theta_m,rho_base=reference.rho_base,rho_theta_base=reference.rho_theta_base,
            pressure_base=reference.exner_base,rho_theta_perturbation=rtheta_p,
            qtot=self.arrays.zeros(state.rho.shape,np.float32) if qtot is None else qtot,
            rdzw=vertical.rdzw,fzm=vertical.fzm,fzp=vertical.fzp,rdzu=vertical.rdzu,
            cell_solve_count=None if self.regional_v841 is None else self.regional_v841.n_cells_solve)

    def _vector_momentum(self,u,rho_edge,pv_edge,kinetic,divergence,seed):
        mesh=self.atmosphere.mesh
        out=self.arrays.empty(u.shape,np.float32)
        launch('dynamics','vector_momentum_rrfs_f32',self.nlev*self.nedges,
            (self.nlev,self.ncells,self.nedges,mesh.max_edges2,u,rho_edge,pv_edge,kinetic,divergence,
             mesh.cells_on_edge,mesh.edges_on_edge,mesh.n_edges_on_edge,mesh.weights_on_edge,
             self.v841_context.inv_dc_edge,seed,1,out))
        return out

    def _zero_recovered_boundary_w(self,w):
        regional=self.regional_v841
        masks=regional.masks
        regional.kernels.launch('regional_zero_speczone_w_v841',masks.n_spec_cells,
            dict(spec_cells=masks.spec_cells,w=w[:1]),
            dict(nrows=1,ncells=self.ncells,nspec=masks.n_spec_cells))

    def _w_boundary(self,w):
        regional=self.regional_v841
        nearest=regional.nearest_w_cell
        if nearest is None:
            nearest=regional.bdy_mask_cell
        launch('regional','w_boundary_rrfs',regional.n_cells_solve,
            (self.nlev,self.ncells,0,regional.n_cells_solve,5,int(self.config.config_lbc_w=='zero'),
             regional.bdy_mask_cell,nearest,w,w))

    def _advance_dynamics_subcycle(
        self, state: Any, *, time_level_one: Any, initial_diagnostics: Any,
        schedule: RKSchedule, outer_dt: float, validation_flag: Any,
        physics: Any | None = None, moist: Any | None = None,
        native_stage_callback=None, dynamics_substep: int = 1,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        if schedule.full_timestep != outer_dt:
            raise ValueError("RRFS subcycle schedule and outer timestep differ")
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
            cqw=None if moist is None else moist.cqw,
            qtot=None if moist is None else moist.qtot)
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
            tend_rho_saved = tend_rho_saved.copy()
        if moist is None:
            saved_euler_rw = _euler_w(self, saved_diag.pressure_perturbation, dpdz)
        else:
            saved_euler_rw = moist_euler_w_metal_v841(
                self, saved_diag.pressure_perturbation, dpdz, moist.cqw,
                validation_flag)
        saved_euler_theta = None
        if self.mixing_config_v841 is not None:
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
                initial_u_tendency=saved_euler_ru,
            )
            saved_euler_ru = saved_mixing.tend_u_euler
            saved_euler_rw = saved_mixing.tend_w_euler
            solve = self.ncells if self.regional_v841 is None else self.regional_v841.n_cells_solve
            launch('dynamics', 'euler_w_accumulate_rrfs_f32', (self.nlev-1)*solve,
                (self.nlev,self.ncells,solve,saved_diag.pressure_perturbation,dpdz,
                 self.arrays.ones(saved_state.rho.shape,np.float32) if moist is None else moist.cqw,
                 vertical.rdzu,vertical.fzm,vertical.fzp,saved_euler_rw))
            saved_euler_theta = saved_mixing.tend_theta_euler.copy()

        final_flux_u = final_flux_w = None
        modified_theta_rate = None
        for stage in schedule.stages:
            if native_stage_callback is not None:
                physics = native_stage_callback(rk_step=stage.stage,
                    dynamics_substep=dynamics_substep, state=current_state,
                    saved=current_diag, diagnostics=diagnostics)
                if physics is None:
                    raise RuntimeError("active native MPAS RK callback returned no sources")
                if stage.stage == 1:
                    tend_rho_saved = self.horizontal.density_tendency(
                        saved_state.rho_u, saved_state.rho_w, vertical.rdzw,
                        physics_tendency=physics.rho)
                    if self.regional_v841 is not None:
                        tend_rho_saved = tend_rho_saved.copy()
            if stage.stage == 2 and self.config.config_time_integration_order == 3:
                coefficients = self._vertical_coefficients(
                    current_state, stage.acoustic_timestep,
                    current_diag.theta_m, current_diag.rho_theta_perturbation,
                    exner=saved_diag.exner, validation_flag=validation_flag,
                    cqw=None if moist is None else moist.cqw,
                    qtot=None if moist is None else moist.qtot)
            mass_divergence = self.horizontal.mass_flux_divergence(
                current_state.rho_u)
            tend_ru = _vertical_u(self, current_diag.normal_velocity,
                                  current_state.rho_w)
            tend_ru = self._vector_momentum(current_diag.normal_velocity, diagnostics.h_edge,
                diagnostics.pv_edge, diagnostics.kinetic_energy, mass_divergence, tend_ru)
            _add(self, tend_ru, saved_euler_ru)
            if physics is not None:
                _add(self, tend_ru, physics.rho_u)
            tend_rt = _theta_tendency(
                self, current_state, current_diag.theta_m, saved_state,
                saved_diag.theta_m, stage.stage)
            if physics is not None and stage.stage == 3:
                rho_rate = tend_rho_saved if self.regional_v841 is not None else tend_rho_dynamics
                modified_theta_rate = modified_theta_dynamics_rate_metal_v841(
                    self, tend_rt, rho_rate, current_state.rho,
                    current_diag.theta_m, validation_flag)
            diabatic = (getattr(physics, "diabatic_theta", None)
                        if native_stage_callback is not None else None)
            if diabatic is not None:
                from .mpasphys.metal_lagged_v842 import add_lagged
                add_lagged(self, current_state.rho, diabatic, tend_rt)
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
                self.regional_v841.adjust_dynamics_tendencies(
                    tend_ru=tend_ru, tend_rho=tend_rho_saved, tend_rt=tend_rt,
                    tend_omega=tend_omega, rho_u=current_state.rho_u,
                    theta_m=current_diag.theta_m, rho_zz=current_state.rho,
                    rk_step=stage.stage)
                if diabatic is not None:
                    from .mpasphys.metal_lagged_v842 import zero_speczone
                    zero_speczone(self, diabatic)
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
                regional = self.regional_v841
                acoustic = advance_acoustic_step_metal_rrfs(
                    self.atmosphere.mesh, acoustic, forcing, coefficients,
                    context=self.v841_context, dts=stage.acoustic_timestep,
                    small_step=small_step, epssm=self.config.config_epssm,
                    fzm=vertical.fzm, fzp=vertical.fzp, rdzw=vertical.rdzw,
                    cell_solve_count=None if regional is None else regional.n_cells_solve,
                    spec_zone_mask_cell=None if regional is None else regional.spec_zone_mask_cell,
                    spec_zone_mask_edge=None if regional is None else regional.spec_zone_mask_edge)
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
                stage.acoustic_steps, current_state.scalars,
                native_recovery=native_stage_callback is not None,
                diabatic_theta=diabatic, large_timestep=stage.large_timestep)
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
            if self.regional_v841 is not None:
                self._w_boundary(current_diag.vertical_velocity)
            cached_v = diagnostics.tangential_velocity
        if final_flux_u is None or final_flux_w is None:
            raise RuntimeError("v8.4.1 dynamics subcycle did not reach RK3")
        return (current_state, current_diag, diagnostics, final_flux_u,
                final_flux_w, modified_theta_rate, physics)

    def _advance(self, physics: Any | None, *, transport_callback=None,
                 native_stage_callback=None, native_post_transport=None,
                 native_mass_loading=None) -> tuple[Any, Any, Any, Any]:
        """``_step_device_v841`` up to the four-byte validation readback."""

        config = self.config
        validation_flag = self.arrays.zeros((1,), np.int32)
        regional = self.regional_v841
        if regional is not None:
            regional.begin_step(self.atmosphere.state)
        outer_saved = _copy_state(self, self.atmosphere.state)
        time_level_one = _copy_saved(self, self.atmosphere.saved)
        if regional is not None:
            regional.bind_state_rho(outer_saved.rho)
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
        if native_stage_callback is not None and config.config_moist_physics:
            moist = compute_moist_dynamics_coefficients_metal_v841(
                self, outer_saved.scalars, n_mass_loading=native_mass_loading,
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
        if regional is not None and moist is None and config.config_moist_physics:
            moist = regional.moist_coefficients(outer_saved.scalars)
        if moist is None and (regional is None or native_mass_loading is not None):
            moist = compute_moist_dynamics_coefficients_metal_v841(
                self,outer_saved.scalars,n_mass_loading=(outer_saved.scalars.shape[0]
                    if native_mass_loading is None else native_mass_loading),validation_flag=validation_flag)
        current, carried = outer_saved, time_level_one
        split_flux_u_sum = split_flux_w_sum = None
        modified_theta_rate = None
        for _subcycle in range(splits):
            if regional is not None:
                regional.begin_dynamics_substep(_subcycle + 1)
            (current, carried, diagnostics, flux_u, flux_w,
             modified_theta_rate, physics) = self._advance_dynamics_subcycle(
                current, time_level_one=carried,
                initial_diagnostics=diagnostics,
                schedule=dynamics_schedule, outer_dt=outer_dt,
                validation_flag=validation_flag, physics=physics, moist=moist,
                native_stage_callback=native_stage_callback,
                dynamics_substep=_subcycle + 1)
            split_flux_u_sum = accumulate_split_flux_metal_v841(
                flux_u, split_flux_u_sum)
            split_flux_w_sum = accumulate_split_flux_metal_v841(
                flux_w, split_flux_w_sum)
        split_flux_u = finish_split_flux_metal_v841(split_flux_u_sum, splits)
        split_flux_w = finish_split_flux_metal_v841(split_flux_w_sum, splits)

        if config.config_scalar_advection and current.scalars.shape[0] > 0:
            vertical = self.atmosphere.vertical
            scalar_stage = outer_saved.scalars.copy()
            source = self.arrays.zeros(outer_saved.scalars.shape,np.float32) if physics is None else physics.scalars.copy()
            coefficients = replace(self.coefficients,
                adv_coefs_3rd=_scale(self,self.coefficients.adv_coefs_3rd,config.config_coef_3rd_order))
            for stage in scalar_schedule.stages:
                limited = stage.stage == 3 and (config.config_monotonic or config.config_positive_definite)
                transported = advance_scalars(self.atmosphere.mesh,self.v841_context,
                    outer_saved.scalars,scalar_stage,outer_saved.rho,current.rho,
                    split_flux_u,split_flux_w,stage.large_timestep,
                    coefficients=coefficients,fzm=vertical.fzm,fzp=vertical.fzp,rdzw=vertical.rdzw,
                    rk_step=stage.stage,config_time_integration_order=config.config_time_integration_order,
                    config_coef_3rd_order=config.config_coef_3rd_order,
                    config_monotonic=config.config_monotonic,config_positive_definite=config.config_positive_definite,
                    scalar_tendency=(physics.scalars if limited and regional is not None and physics is not None else source),
                    **({} if regional is None else dict(config_apply_lbcs=True,
                        bdy_mask_cell=regional.bdy_mask_cell,bdy_mask_edge=regional.bdy_mask_edge,
                        cell_solve_count=regional.n_cells_solve)))
                scalar_stage = transported.scalars
                if regional is not None:
                    regional.bdy_adjust_scalars(scalar_stage,stage.stage)
                accumulate_finite_array_metal_v841(scalar_stage,invalid_flag=validation_flag)
            current.scalars = scalar_stage
        if native_post_transport is not None:
            native_post_transport(state=current, saved=carried, outer_state=outer_saved,
                modified_theta_dynamics_rate=modified_theta_rate)
        if regional is not None:
            if config.config_moist_physics and native_post_transport is None:
                regional.clamp_negative_scalars(current.scalars)
            regional.reset_speczone_values(
                theta_m=carried.theta_m, rho_theta=current.rho_theta,
                rho_theta_perturbation=carried.rho_theta_perturbation,
                scalars=current.scalars)
            if transport_callback is None:
                regional.advance_clock(outer_dt)

        rates = None
        if physics is not None and native_stage_callback is None:
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
        if transport_callback is not None:
            transport_callback(current, carried, outer_saved, modified_theta_rate)
            if regional is not None:
                regional.advance_clock(outer_dt)
        return current, carried, outer_saved, rates
