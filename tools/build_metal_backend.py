#!/usr/bin/env python3
"""Build WOOF's native Metal runtime and packaged compute-kernel libraries.

Source builds need Xcode, an installed Metal Toolchain, and an Apple Silicon
host. The resulting wheel carries the dylib and metallib; import-time code
never compiles MSL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from woof import bridge_assets
from tools.metal_abi import canonical_metal_abi_version
from woof.metal.bundle_integrity import BUNDLE_SCHEMA, metal_source_provenance


SHADER = ROOT / "woof" / "metal" / "shaders" / "rk_bookkeeping.metal"
CASE_BOUNDARY_MASS_FLUX_SHADER = ROOT / "woof" / "metal" / "shaders" / "case_boundary_mass_flux.metal"
GABLS_COLUMN_SHADER = ROOT / "woof" / "metal" / "shaders" / "gabls_column.metal"
EXACT_SMALL_STEP_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_small_step.metal"
EXACT_ACOUSTIC_WPHI_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_acoustic_advance_w_phi.metal"
EXACT_ACOUSTIC_MU_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_acoustic_advance_mu_th.metal"
EXACT_ACOUSTIC_UV_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_acoustic_advance_uv.metal"
EXACT_ACOUSTIC_COEFS_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_acoustic_calc_coefs.metal"
ACOUSTIC_SHADER = ROOT / "woof" / "metal" / "shaders" / "acoustic_coefficients.metal"
ACOUSTIC_CALC_CQ_SHADER = ROOT / "woof" / "metal" / "shaders" / "acoustic_calc_cq.metal"
ACOUSTIC_UV_SHADER = ROOT / "woof" / "metal" / "shaders" / "acoustic_advance_uv.metal"
ACOUSTIC_MU_TH_SHADER = ROOT / "woof" / "metal" / "shaders" / "acoustic_advance_mu_th.metal"
ACOUSTIC_W_PHI_SHADER = ROOT / "woof" / "metal" / "shaders" / "acoustic_advance_w_phi.metal"
ACOUSTIC_BOUNDARY_FRAME_SHADER = ROOT / "woof" / "metal" / "shaders" / "acoustic_boundary_frame.metal"
EXACT_ACOUSTIC_EMDIV_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_acoustic_emdiv.metal"
ACOUSTIC_EMDIV_SHADER = ROOT / "woof" / "metal" / "shaders" / "acoustic_emdiv.metal"
ADVECTION_SCALAR_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_flux_div_scalar.metal"
ADVECTION_U_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_flux_div_u.metal"
ADVECTION_V_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_flux_div_v.metal"
ADVECTION_W_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_flux_div_w.metal"
ADVECTION_SCALAR_VORDER_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_flux_div_scalar_vorder.metal"
ADVECTION_U_VORDER_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_flux_div_u_vorder.metal"
ADVECTION_V_VORDER_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_flux_div_v_vorder.metal"
ADVECTION_W_VORDER_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_flux_div_w_vorder.metal"
EXACT_DIAGNOSTICS_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_diagnostics.metal"
DIAGNOSTICS_CALC_SHADER = ROOT / "woof" / "metal" / "shaders" / "diagnostics_calc_p_alpha.metal"
EXACT_STAGE_MASS_FLUXES_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_stage_mass_fluxes.metal"
STAGE_MASS_FLUXES_SHADER = ROOT / "woof" / "metal" / "shaders" / "stage_mass_fluxes.metal"
ADVECTION_PREPARE_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_prepare.metal"
DYCORE_ADVANCE_STAGE_SHADER = ROOT / "woof" / "metal" / "shaders" / "dycore_advance_stage.metal"
EXACT_SLOW_PGF_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_slow_pgf.metal"
EXACT_CALC_CQ_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_calc_cq.metal"
SLOW_PGF_SHADER = ROOT / "woof" / "metal" / "shaders" / "slow_pgf.metal"
DIFFUSION_SHADER = ROOT / "woof" / "metal" / "shaders" / "diffusion.metal"
EXACT_SLOW_BUOYANCY_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_slow_buoyancy.metal"
SLOW_BUOYANCY_SHADER = ROOT / "woof" / "metal" / "shaders" / "slow_buoyancy.metal"
EXACT_SLOW_GEOPOTENTIAL_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_slow_geopotential.metal"
SLOW_GEOPOTENTIAL_SHADER = ROOT / "woof" / "metal" / "shaders" / "slow_geopotential.metal"
SURFACE_W_SHADER = ROOT / "woof" / "metal" / "shaders" / "surface_w.metal"
SMALL_STEP_SHADER = ROOT / "woof" / "metal" / "shaders" / "small_step.metal"
CLOSE_PERIODIC_ALIAS_SHADER = ROOT / "woof" / "metal" / "shaders" / "close_periodic_alias.metal"
OPENBC_RADIATIVE_SHADER = ROOT / "woof" / "metal" / "shaders" / "openbc_radiative.metal"
RK_TOTAL_THETA_SHADER = ROOT / "woof" / "metal" / "shaders" / "rk_total_theta.metal"
CORIOLIS_CURVATURE_SHADER = ROOT / "woof" / "metal" / "shaders" / "coriolis_curvature.metal"
EXACT_PD_ON_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_pd_on.metal"
EXACT_PD_OFF_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_pd_off.metal"
MOIST_PD_SHADER = ROOT / "woof" / "metal" / "shaders" / "moist_pd.metal"
EXACT_MOIST_CAPTURE_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_moist_capture.metal"
EXACT_MOIST_TENDENCY_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_moist_tendency.metal"
EXACT_MOIST_SCALAR_UPDATE_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_moist_scalar_update.metal"
MOIST_SCALAR_UPDATE_SHADER = ROOT / "woof" / "metal" / "shaders" / "moist_scalar_update.metal"
NEST_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "nest_family.metal"
NEST_INTERP_FRAGMENT_SHADER = ROOT / "woof" / "metal" / "shaders" / "nest_interp_fragment.metal"
NEST_STREAM_TRANSFER_FRAGMENT_SHADER = ROOT / "woof" / "metal" / "shaders" / "nest_stream_transfer_fragment.metal"
IEVA_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "ieva_family.metal"
RADAR_TTEN_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "radar_tten_family.metal"
IEVA_SOLVE_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "ieva_solve_family.metal"
FACE_MASS_SHADER = ROOT / "woof" / "metal" / "shaders" / "face_mass.metal"
COMPLEX64_PRIMITIVES_SHADER = ROOT / "woof" / "metal" / "shaders" / "complex64_primitives.metal"
JACOBI_EIGH_FP64_SHADER = ROOT / "woof" / "metal" / "shaders" / "jacobi_eigh_fp64.metal"
SPECTRAL_SCALAR_SHADER = ROOT / "woof" / "metal" / "shaders" / "spectral_scalar.metal"
SFCLAY_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "sfclay_family.metal"
MYNN_SURFACE_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "mynn_surface_family.metal"
YSU_COLUMN_SHADER = ROOT / "woof" / "metal" / "shaders" / "ysu_column.metal"
YSU_480_COLUMN_SHADER = ROOT / "woof" / "metal" / "shaders" / "ysu_480_column.metal"
SHINHONG_COLUMN_SHADER = ROOT / "woof" / "metal" / "shaders" / "shinhong_column.metal"
MOIST_PD_SOURCE_FOLD_SHADER = ROOT / "woof" / "metal" / "shaders" / "moist_pd_source_fold.metal"
THOMPSON_COLD_START_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_cold_start_family.metal"
THOMPSON_COLD_START_SURFACE_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_cold_start_surface.metal"
THOMPSON_COLD_START_NUMBERS_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_cold_start_numbers.metal"
THOMPSON_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_family.metal"
THOMPSON_FROZEN_VAPOR_NETWORK_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_frozen_vapor_network.metal"
THOMPSON_WARM_FROZEN_SOURCE_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_warm_frozen_source.metal"
THOMPSON_CLOUD_SEDIMENTATION_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_cloud_sedimentation.metal"
THOMPSON_ICE_SEDIMENTATION_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_ice_sedimentation.metal"
THOMPSON_RAIN_EVAPORATION_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_rain_evaporation.metal"
THOMPSON_RAIN_SEDIMENTATION_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_rain_sedimentation.metal"
THOMPSON_SNOW_SEDIMENTATION_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_snow_sedimentation.metal"
THOMPSON_GRAUPEL_SEDIMENTATION_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_graupel_sedimentation.metal"
THOMPSON_FINAL_PHASE_CLEANUP_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_final_phase_cleanup.metal"
THOMPSON_EFFECTIVE_RADIUS_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_effective_radius.metal"
THOMPSON_REFLECTIVITY_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_reflectivity.metal"
SPEC_ZONE_RING_SHADER = ROOT / "woof" / "metal" / "shaders" / "spec_zone_ring.metal"
PHYSICS_ATMOSPHERE_PREPARE_SHADER = ROOT / "woof" / "metal" / "shaders" / "physics_atmosphere_prepare.metal"
COUPLED_BOUNDARY_FIELD_SHADER = ROOT / "woof" / "metal" / "shaders" / "coupled_boundary_field.metal"
LBC_TIME_SHADER = ROOT / "woof" / "metal" / "shaders" / "lbc_time.metal"
EXACT_NEST_UNCOUPLE_ON_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_nest_uncouple_on.metal"
EXACT_NEST_UNCOUPLE_OFF_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_nest_uncouple_off.metal"
EXACT_MYNN_RADIATION_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_mynn_radiation.metal"
EXACT_DUDHIA_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_dudhia_shortwave.metal"
EXACT_HELD_HEATING_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_held_heating.metal"
EXACT_NEST_SMOOTH_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_nest_smooth.metal"
EXACT_NEST_INTERP_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_nest_interp.metal"
EXACT_NEST_FEEDBACK_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_nest_feedback.metal"
EXACT_NEST_COUPLE_ON_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_nest_couple_on.metal"
EXACT_NEST_COUPLE_OFF_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_nest_couple_off.metal"
EXACT_NEST_ROUNDTRIP_SHADER = ROOT / "woof/metal/shaders/strict_nest_roundtrip_family.metal"
EXACT_LBC_FINAL_ON_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_lbc_final_on.metal"
EXACT_LBC_FINAL_OFF_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_lbc_final_off.metal"
EXACT_LBC_INSTALL_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_install_mu_boundary.metal"
EXACT_LBC_INTERIOR_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_lbc_interior.metal"
EXACT_LBC_ON_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_lbc_on.metal"
EXACT_LBC_OFF_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_lbc_off.metal"
LBC_STATE_SHADER = ROOT / "woof" / "metal" / "shaders" / "lbc_state.metal"
EXACT_LBC_FLOW_SHADER = ROOT / "woof" / "metal" / "shaders" / "wrf_exact_lbc_flow.metal"
LBC_FLOW_SHADER = ROOT / "woof" / "metal" / "shaders" / "lbc_flow.metal"
SURFACE_HUMIDITY_SHADER = ROOT / "woof" / "metal" / "shaders" / "surface_humidity.metal"
SURFACE_DIAGNOSTICS_SEED_SHADER = ROOT / "woof" / "metal" / "shaders" / "surface_diagnostics_seed.metal"
ACOUSTIC_SUMFLUX_SHADER = ROOT / "woof" / "metal" / "shaders" / "acoustic_sumflux.metal"
REAL_TOTAL_GEOPOTENTIAL_SHADER = ROOT / "woof" / "metal" / "shaders" / "real_total_geopotential.metal"
RRTMGP_PREPARE_SHADER = ROOT / "woof" / "metal" / "shaders" / "rrtmgp_prepare_family.metal"
RRTM_LW_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "rrtm_lw_family.metal"
RRTMG_LEGACY_LW_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "rrtmg_legacy_lw_family.metal"
RRTMGP_MCICA_SHADER = ROOT / "woof" / "metal" / "shaders" / "rrtmgp_mcica_family.metal"
HEX_MASS_FLUX_SHADER = ROOT / "woof" / "metal" / "shaders" / "hex_mass_flux_divergence.metal"
RRTMGP_GAS_SHADER = ROOT / "woof" / "metal" / "shaders" / "rrtmgp_gas_family.metal"
RRTMGP_CLOUD_SHADER = ROOT / "woof" / "metal" / "shaders" / "rrtmgp_cloud_family.metal"
RRTMGP_CLOUD_PREP_SHADER = ROOT / "woof" / "metal" / "shaders" / "rrtmgp_cloud_prep_family.metal"
RRTMGP_RTE_SHADER = ROOT / "woof" / "metal" / "shaders" / "rrtmgp_rte_family.metal"
KF_W0AVG_SHADER = ROOT / "woof" / "metal" / "shaders" / "kf_w0avg_update.metal"
GLOBAL_SPECTRAL_DIFFUSION_SHADER = ROOT / "woof" / "metal" / "shaders" / "global_spectral_diffusion.metal"
HEX_SOLVE_VERTICAL_SYSTEM_SHADER = ROOT / "woof" / "metal" / "shaders" / "hex_solve_vertical_system_fragment.metal"
GLOBAL_TRACER_FLUX_FRAGMENT_SHADER = ROOT / "woof" / "metal" / "shaders" / "global_tracer_flux_fragment.metal"
GLOBAL_TRACER_GAP_FRAGMENT_SHADER = ROOT / "woof" / "metal" / "shaders" / "global_tracer_gap_fragment.metal"
GLOBAL_LONGITUDE_TRANSFORM_FRAGMENT_SHADER = ROOT / "woof" / "metal" / "shaders" / "global_longitude_transform_fragment.metal"
GLOBAL_INVERSE_LONGITUDE_TRANSFORM_FRAGMENT_SHADER = ROOT / "woof" / "metal" / "shaders" / "global_inverse_longitude_transform_fragment.metal"
GLOBAL_MOMENTUM_FRAGMENT_SHADER = ROOT / "woof" / "metal" / "shaders" / "global_momentum_fragment.metal"
HEX_RK_TENDENCIES_FRAGMENT_SHADER = ROOT / "woof" / "metal" / "shaders" / "hex_rk_tendencies_fragment.metal"
HEX_INTERFACE_VELOCITY_TERRAIN_FRAGMENT_SHADER = ROOT / "woof" / "metal" / "shaders" / "hex_interface_velocity_terrain_fragment.metal"
HEX_MESH_DIAGNOSTICS_FRAGMENT_SHADER = ROOT / "woof" / "metal" / "shaders" / "hex_mesh_diagnostics_fragment.metal"
UWPBL_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "uwpbl_family.metal"
RUC_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "ruc_family.metal"
RUC_SFCTMP_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "ruc_sfctmp_family.metal"
RUC_DRIVER_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "ruc_driver_family.metal"
RUC_SFCLAY_SEAICE_SHADER = ROOT / "woof" / "metal" / "shaders" / "ruc_sfclay_seaice.metal"
LAKE_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "lake_family.metal"
HAIL_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "hail_family.metal"
SASE_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "sase_family.metal"
DUDHIA_SHORTWAVE_SHADER = ROOT / "woof" / "metal" / "shaders" / "dudhia_shortwave.metal"
KESSLER_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "kessler_family.metal"
STRICT_REGIONAL_KESSLER_PREP_SHADER = ROOT / "woof" / "metal" / "shaders" / "strict_regional_kessler_prep.metal"
STRICT_REGIONAL_KESSLER_COLUMN_SHADER = ROOT / "woof" / "metal" / "shaders" / "strict_regional_kessler.metal"
STRICT_REGIONAL_KESSLER_FINISH_SHADER = ROOT / "woof" / "metal" / "shaders" / "strict_regional_kessler_finish.metal"
WSM6_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "wsm6_family.metal"
WDM6_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "wdm6_family.metal"
MP_ZERO_OUT_SHADER = ROOT / "woof" / "metal" / "shaders" / "mp_zero_out_fragment.metal"
NOAH_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "noah_family.metal"
NOAH_SFCDIAGS_SHADER = ROOT / "woof" / "metal" / "shaders" / "noah_sfcdiags_family.metal"
NOAH_MOSAIC_SHADER = ROOT / "woof" / "metal" / "shaders" / "noah_mosaic_family.metal"
HEALTH_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "health_family.metal"
SMAG2D_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "smag2d_family.metal"
DIFF6_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "diff6_family.metal"
UH_DIAG_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "uh_diag_family.metal"
KF_COLUMN_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "kf_column_family.metal"
KF_OUTPUT_VALIDATION_SHADER = ROOT / "woof" / "metal" / "shaders" / "kf_output_validation.metal"
MORRISON_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "morrison_family.metal"
MORRISON_REFL_SHADER = ROOT / "woof" / "metal" / "shaders" / "morrison_refl.metal"
REFL_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "refl_family.metal"
CUMULUS_CLOCK_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "cumulus_clock_family.metal"
NOAH_FORCING_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "noah_forcing_family.metal"
MILBRANDT2_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "milbrandt2_family.metal"
MILBRANDT2_FIELDS_SHADER = ROOT / "woof" / "metal" / "shaders" / "milbrandt2_fields.metal"
MILBRANDT2_ZET_SHADER = ROOT / "woof" / "metal" / "shaders" / "milbrandt2_zet.metal"
WDM6_REFLECTIVITY_SHADER = ROOT / "woof" / "metal" / "shaders" / "wdm6_reflectivity.metal"
WSM6_REFLECTIVITY_SHADER = ROOT / "woof" / "metal" / "shaders" / "wsm6_reflectivity.metal"
NOAHMP_DRIVER_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "noahmp_driver_family.metal"
NOAHMP_SFLX_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "noahmp_sflx_family.metal"
NOAHMP_LEAVES_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "noahmp_leaves_family.metal"
NOAHMP_VEGPRECIP_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "noahmp_vegprecip_family.metal"
NOAHMP_THERMAL_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "noahmp_thermal_family.metal"
NOAHMP_BAREFLUX_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "noahmp_bareflux_family.metal"
NOAHMP_VEGEFLUX_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "noahmp_vegeflux_family.metal"
NOAHMP_WATER_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "noahmp_water_family.metal"
NOAHMP_RADIATION_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "noahmp_radiation_family.metal"
NOAHMP_GLACIER_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "noahmp_glacier_family.metal"
MYNN_PBL_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "mynn_pbl_family.metal"
MYJPBL_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "myjpbl_family.metal"
MYJSFC_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "myjsfc_family.metal"
MYJURB_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "myjurb_family.metal"
NTIEDTKE_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "ntiedtke_family.metal"
GF_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "gf_family.metal"
NSSL2_FUSED_GS_SHADER = ROOT / "woof" / "metal" / "shaders" / "nssl2_fused_gs_family.metal"
NSSL2_NUCOND_SHADER = ROOT / "woof" / "metal" / "shaders" / "nssl2_nucond.metal"
NSSL2_DRIVER_TRANSPORT_SHADER = ROOT / "woof" / "metal" / "shaders" / "nssl2_driver_transport.metal"
NSSL2_SEDIMENTATION_SHADER = ROOT / "woof" / "metal" / "shaders" / "nssl2_sedimentation.metal"
NSSL2_DIAGNOSTICS_SHADER = ROOT / "woof" / "metal" / "shaders" / "nssl2_diagnostics.metal"
NSSL2_RUNTIME_SHADER = ROOT / "woof" / "metal" / "shaders" / "nssl2_runtime.metal"
P3_ENTRY_SHADER = ROOT / "woof" / "metal" / "shaders" / "p3_entry.metal"
P3_KLOOP1_SHADER = ROOT / "woof" / "metal" / "shaders" / "p3_kloop1.metal"
P3_KLOOPMAIN_SHADER = ROOT / "woof" / "metal" / "shaders" / "p3_kloopmain.metal"
P3_SAVEOLD_PRECIP_SHADER = ROOT / "woof" / "metal" / "shaders" / "p3_saveold_precip.metal"
P3_HOMOFREEZE_SHADER = ROOT / "woof" / "metal" / "shaders" / "p3_homofreeze.metal"
P3_FINAL_SHADER = ROOT / "woof" / "metal" / "shaders" / "p3_final.metal"
P3_SED_CLOUD_SHADER = ROOT / "woof" / "metal" / "shaders" / "p3_sed_cloud.metal"
P3_SED_RAIN_SHADER = ROOT / "woof" / "metal" / "shaders" / "p3_sed_rain.metal"
P3_SED_ICE_SHADER = ROOT / "woof" / "metal" / "shaders" / "p3_sed_ice.metal"
THOMPSON_AEROSOL_STATE_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_aerosol_state_family.metal"
TOPO_RADIATION_SHADER = ROOT / "woof" / "metal" / "shaders" / "topo_radiation_family.metal"
TERRAIN_DRAG_SHADER = ROOT / "woof" / "metal" / "shaders" / "terrain_drag_family.metal"
THOMPSON_AEROSOL_SAT_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_aerosol_sat_family.metal"
THOMPSON_AEROSOL_WARM_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_aerosol_warm_family.metal"
THOMPSON_AEROSOL_COLD_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_aerosol_cold_family.metal"
THOMPSON_AEROSOL_SED_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_aerosol_sed_family.metal"
THOMPSON_AEROSOL_WRF39_SED_SHADER = ROOT / "woof" / "metal" / "shaders" / "thompson_aerosol_wrf39_sed_family.metal"
NATIVE_DIR = ROOT / "woof" / "metal" / "native"
GENERATED_DIR = ROOT / "woof" / "metal" / "generated"
def _run_uncached(args: list[str]) -> str:
    result = subprocess.run(args, cwd=ROOT, text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(args)}\n"
            f"{result.stdout}")
    return result.stdout


def _run(args: list[str]) -> str:
    from tools.metal_compile_cache import cached_air
    return cached_air(args, ROOT, _run_uncached)


def _metal_tool(name: str) -> Path:
    """Prefer Xcode's selected tools, then an already-mounted Metal cryptex."""
    try:
        candidate = Path(_run(["xcrun", "--sdk", "macosx", "--find", name])
                         .strip().splitlines()[-1])
        probe = subprocess.run([str(candidate), "--version"], text=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        if probe.returncode == 0:
            return candidate
    except (OSError, RuntimeError, subprocess.CalledProcessError, IndexError):
        pass

    mounted = Path("/var/run/com.apple.security.cryptexd/mnt")
    candidates = sorted(mounted.glob(
        "com.apple.MobileAsset.MetalToolchain-*/Metal.xctoolchain/usr/bin/" +
        name), reverse=True)
    for candidate in candidates:
        probe = subprocess.run([str(candidate), "--version"], text=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        if probe.returncode == 0:
            return candidate
    raise RuntimeError(
        f"Metal tool {name!r} is unavailable. Select Xcode and install its "
        "Metal Toolchain with `xcodebuild -downloadComponent MetalToolchain`.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_provenance() -> tuple[str, str, int]:
    """Commit plus fingerprint of the Metal implementation inputs."""

    commit = _run(["git", "rev-parse", "HEAD"]).strip()
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise RuntimeError("git rev-parse HEAD did not return a full commit id")
    digest, count = bridge_assets.metal_implementation_provenance(ROOT)
    return commit, digest, count


def build(output: Path) -> dict:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("the WOOF Metal bundle builds on macOS arm64 only")
    abi_version = canonical_metal_abi_version()
    if (abi_version >> 16 != bridge_assets.METAL_ABI_MAJOR
            or (abi_version & 0xFFFF) < bridge_assets.METAL_ABI_MINIMUM_MINOR):
        raise RuntimeError(
            "canonical Metal header ABI is incompatible with the package "
            f"runtime: 0x{abi_version:08x}")
    _run([sys.executable, str(ROOT / "tools" / "generate_metal_kernel_signatures.py")])
    output.mkdir(parents=True, exist_ok=True)
    metal = _metal_tool("metal")
    metallib = _metal_tool("metallib")
    with tempfile.TemporaryDirectory(prefix="woof-metal-build-") as temporary:
        temp = Path(temporary)
        rk_air = temp / "rk_bookkeeping.air"
        exact_small_step_air = temp / "wrf_exact_small_step.air"
        exact_acoustic_wphi_air = temp / "wrf_exact_acoustic_advance_w_phi.air"
        exact_acoustic_mu_air = temp / "wrf_exact_acoustic_advance_mu_th.air"
        exact_acoustic_uv_air = temp / "wrf_exact_acoustic_advance_uv.air"
        exact_acoustic_coefs_air = temp / "wrf_exact_acoustic_calc_coefs.air"
        acoustic_default_air = temp / "acoustic_default.air"
        acoustic_exact_air = temp / "acoustic_exact.air"
        acoustic_calc_cq_air = temp / "acoustic_calc_cq.air"
        uv_default_air = temp / "acoustic_uv_default.air"
        uv_exact_air = temp / "acoustic_uv_exact.air"
        mu_default_air = temp / "acoustic_mu_th_default.air"
        mu_exact_air = temp / "acoustic_mu_th_exact.air"
        wphi_default_air = temp / "acoustic_w_phi_default.air"
        wphi_exact_air = temp / "acoustic_w_phi_exact.air"
        advection_default_air = temp / "advection_scalar_default.air"
        advection_exact_air = temp / "advection_scalar_exact.air"
        advection_u_default_air = temp / "advection_u_default.air"
        advection_u_exact_air = temp / "advection_u_exact.air"
        advection_v_default_air = temp / "advection_v_default.air"
        advection_v_exact_air = temp / "advection_v_exact.air"
        advection_w_default_air = temp / "advection_w_default.air"
        advection_w_exact_air = temp / "advection_w_exact.air"
        advection_scalar_vorder_default_air = temp / "advection_scalar_vorder_default.air"
        advection_scalar_vorder_exact_air = temp / "advection_scalar_vorder_exact.air"
        advection_u_vorder_default_air = temp / "advection_u_vorder_default.air"
        advection_u_vorder_exact_air = temp / "advection_u_vorder_exact.air"
        advection_v_vorder_default_air = temp / "advection_v_vorder_default.air"
        advection_v_vorder_exact_air = temp / "advection_v_vorder_exact.air"
        advection_w_vorder_default_air = temp / "advection_w_vorder_default.air"
        advection_w_vorder_exact_air = temp / "advection_w_vorder_exact.air"
        diagnostics_default_air = temp / "diagnostics_default.air"
        diagnostics_exact_air = temp / "diagnostics_exact.air"
        stage_mass_fluxes_air = temp / "stage_mass_fluxes.air"
        exact_stage_mass_fluxes_air = temp / "wrf_exact_stage_mass_fluxes.air"
        advection_prepare_air = temp / "advection_prepare.air"
        advance_stage_air = temp / "dycore_advance_stage.air"
        slow_pgf_air = temp / "slow_pgf.air"
        exact_slow_pgf_air = temp / "wrf_exact_slow_pgf.air"
        exact_calc_cq_air = temp / "wrf_exact_calc_cq.air"
        exact_diagnostics_air = temp / "wrf_exact_diagnostics.air"
        diffusion_air = temp / "diffusion.air"
        slow_buoyancy_air = temp / "slow_buoyancy.air"
        exact_slow_buoyancy_air = temp / "wrf_exact_slow_buoyancy.air"
        slow_geopotential_air = temp / "slow_geopotential.air"
        exact_slow_geopotential_air = temp / "wrf_exact_slow_geopotential.air"
        acoustic_boundary_frame_air = temp / "acoustic_boundary_frame.air"
        surface_w_air = temp / "surface_w.air"
        exact_surface_w_air = temp / "wrf_exact_surface_w.air"
        small_step_air = temp / "small_step.air"
        close_periodic_alias_air = temp / "close_periodic_alias.air"
        exact_acoustic_emdiv_air = temp / "wrf_exact_acoustic_emdiv.air"
        acoustic_emdiv_air = temp / "acoustic_emdiv.air"
        openbc_radiative_air = temp / "openbc_radiative.air"
        rk_total_theta_air = temp / "rk_total_theta.air"
        coriolis_curvature_air = temp / "coriolis_curvature.air"
        module_cache_override = os.environ.get("WOOF_METAL_MODULE_CACHE")
        module_cache = (
            Path(module_cache_override).expanduser().resolve()
            if module_cache_override else temp / "module-cache"
        )
        module_cache.mkdir(parents=True, exist_ok=True)
        default_library_path = output / "woof_default.metallib"
        exact_library_path = output / "woof_exact.metallib"
        runtime_path = output / "libwoof_metal.dylib"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=precise",
              "-c", str(SHADER), "-o", str(rk_air)])
        for air_path, contraction in (
                (acoustic_default_air, "fast"),
                (acoustic_exact_air, "off")):
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise",
                  f"-ffp-contract={contraction}",
                  "-DGPUWM_WRF_EXACT=" + ("1" if contraction == "off" else "0"),
                  "-c", str(ACOUSTIC_SHADER), "-o", str(air_path)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ACOUSTIC_CALC_CQ_SHADER), "-o",
              str(acoustic_calc_cq_air)])
        for source, air_path, contraction in (
                (ACOUSTIC_UV_SHADER, uv_default_air, "fast"),
                (ACOUSTIC_UV_SHADER, uv_exact_air, "off"),
                (ACOUSTIC_MU_TH_SHADER, mu_default_air, "fast"),
                (ACOUSTIC_MU_TH_SHADER, mu_exact_air, "off"),
                (ACOUSTIC_W_PHI_SHADER, wphi_default_air, "fast"),
                (ACOUSTIC_W_PHI_SHADER, wphi_exact_air, "off")):
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise",
                  f"-ffp-contract={contraction}",
                  "-DGPUWM_WRF_EXACT=" + ("1" if contraction == "off" else "0"),
                  "-c", str(source), "-o", str(air_path)])
        for source, air_path, contraction, exact in (
                (ADVECTION_SCALAR_SHADER, advection_default_air, "fast", "0"),
                (ADVECTION_SCALAR_SHADER, advection_exact_air, "off", "1"),
                (ADVECTION_U_SHADER, advection_u_default_air, "fast", "0"),
                (ADVECTION_U_SHADER, advection_u_exact_air, "off", "1"),
                (ADVECTION_V_SHADER, advection_v_default_air, "fast", "0"),
                (ADVECTION_V_SHADER, advection_v_exact_air, "off", "1"),
                (ADVECTION_W_SHADER, advection_w_default_air, "fast", "0"),
                (ADVECTION_W_SHADER, advection_w_exact_air, "off", "1"),
                (ADVECTION_SCALAR_VORDER_SHADER,
                 advection_scalar_vorder_default_air, "fast", "0"),
                (ADVECTION_SCALAR_VORDER_SHADER,
                 advection_scalar_vorder_exact_air, "off", "1"),
                (ADVECTION_U_VORDER_SHADER,
                 advection_u_vorder_default_air, "fast", "0"),
                (ADVECTION_U_VORDER_SHADER,
                 advection_u_vorder_exact_air, "off", "1"),
                (ADVECTION_V_VORDER_SHADER,
                 advection_v_vorder_default_air, "fast", "0"),
                (ADVECTION_V_VORDER_SHADER,
                 advection_v_vorder_exact_air, "off", "1"),
                (ADVECTION_W_VORDER_SHADER,
                 advection_w_vorder_default_air, "fast", "0"),
                (ADVECTION_W_VORDER_SHADER,
                 advection_w_vorder_exact_air, "off", "1"),
                (DIAGNOSTICS_CALC_SHADER, diagnostics_default_air, "fast", "0"),
                (DIAGNOSTICS_CALC_SHADER, diagnostics_exact_air, "off", "1")):
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise",
                  f"-ffp-contract={contraction}",
                  "-DGPUWM_WRF_EXACT=" + exact,
                  "-DGPUWM_WRF_EXACT_C_ADVECTION=" + exact,
                  "-c", str(source), "-o", str(air_path)])
        exact_advection_air = []
        for mode in ("on", "off"):
            stem = f"wrf_exact_advection_{mode}"
            air = temp / (stem + ".air")
            exact_advection_air.append(air)
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
                  "-c", str(ROOT / "woof/metal/shaders" / (stem + ".metal")),
                  "-o", str(air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(STAGE_MASS_FLUXES_SHADER), "-o",
              str(stage_mass_fluxes_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(EXACT_STAGE_MASS_FLUXES_SHADER), "-o",
              str(exact_stage_mass_fluxes_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ADVECTION_PREPARE_SHADER), "-o",
              str(advection_prepare_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(DYCORE_ADVANCE_STAGE_SHADER), "-o",
              str(advance_stage_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(SLOW_PGF_SHADER), "-o", str(slow_pgf_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(EXACT_SLOW_PGF_SHADER), "-o", str(exact_slow_pgf_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(EXACT_ACOUSTIC_COEFS_SHADER), "-o", str(exact_acoustic_coefs_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(EXACT_ACOUSTIC_UV_SHADER), "-o", str(exact_acoustic_uv_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(EXACT_ACOUSTIC_MU_SHADER), "-o", str(exact_acoustic_mu_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(EXACT_ACOUSTIC_WPHI_SHADER), "-o", str(exact_acoustic_wphi_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(EXACT_SMALL_STEP_SHADER), "-o", str(exact_small_step_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(EXACT_CALC_CQ_SHADER), "-o", str(exact_calc_cq_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(EXACT_DIAGNOSTICS_SHADER), "-o", str(exact_diagnostics_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(DIFFUSION_SHADER), "-o", str(diffusion_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(SLOW_BUOYANCY_SHADER), "-o",
              str(slow_buoyancy_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(EXACT_SLOW_BUOYANCY_SHADER), "-o",
              str(exact_slow_buoyancy_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(SLOW_GEOPOTENTIAL_SHADER), "-o",
              str(slow_geopotential_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(EXACT_SLOW_GEOPOTENTIAL_SHADER), "-o",
              str(exact_slow_geopotential_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ACOUSTIC_BOUNDARY_FRAME_SHADER), "-o",
              str(acoustic_boundary_frame_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(SURFACE_W_SHADER), "-o", str(surface_w_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_surface_w.metal"),
              "-o", str(exact_surface_w_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(SMALL_STEP_SHADER), "-o", str(small_step_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(CLOSE_PERIODIC_ALIAS_SHADER), "-o",
              str(close_periodic_alias_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ACOUSTIC_EMDIV_SHADER), "-o",
              str(acoustic_emdiv_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(EXACT_ACOUSTIC_EMDIV_SHADER), "-o",
              str(exact_acoustic_emdiv_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(OPENBC_RADIATIVE_SHADER), "-o",
              str(openbc_radiative_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(RK_TOTAL_THETA_SHADER), "-o",
              str(rk_total_theta_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(CORIOLIS_CURVATURE_SHADER), "-o",
              str(coriolis_curvature_air)])
        # Ordinary retained heating is compiled into the default library.
        held_heating_air = temp / "held_heating.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/held_heating.metal"),
              "-o", str(held_heating_air)])
        # Ordinary positive-definite scalar transport; WRF-exact arithmetic
        # is refused by the Python submitters and is not part of this library.
        moist_pd_air = temp / "moist_pd.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(MOIST_PD_SHADER), "-o", str(moist_pd_air)])
        exact_pd_on_air = temp / "wrf_exact_pd_on.air"
        exact_pd_off_air = temp / "wrf_exact_pd_off.air"
        for shader, air in ((EXACT_PD_ON_SHADER, exact_pd_on_air),
                            (EXACT_PD_OFF_SHADER, exact_pd_off_air)):
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
                  "-c", str(shader), "-o", str(air)])
        moist_pd_source_fold_air = temp / "moist_pd_source_fold.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(MOIST_PD_SOURCE_FOLD_SHADER), "-o",
              str(moist_pd_source_fold_air)])
        moist_scalar_update_air = temp / "moist_scalar_update.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(MOIST_SCALAR_UPDATE_SHADER), "-o",
              str(moist_scalar_update_air)])
        exact_moist_scalar_update_air = temp / "wrf_exact_moist_scalar_update.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(EXACT_MOIST_SCALAR_UPDATE_SHADER), "-o",
              str(exact_moist_scalar_update_air)])
        exact_moist_tendency_air = temp / "wrf_exact_moist_tendency.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(EXACT_MOIST_TENDENCY_SHADER), "-o",
              str(exact_moist_tendency_air)])
        exact_moist_capture_air = temp / "wrf_exact_moist_capture.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(EXACT_MOIST_CAPTURE_SHADER), "-o",
              str(exact_moist_capture_air)])
        # Exact-policy nesting and IEVA preparation entrypoints. Their typed
        # signatures admit only the supported FP32 operators in each module.
        nest_family_air = temp / "nest_family.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-DGPUWM_WRF_EXACT=1", "-c", str(NEST_FAMILY_SHADER),
              "-o", str(nest_family_air)])
        nest_interp_fragment_air = temp / "nest_interp_fragment.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-DGPUWM_WRF_EXACT=1", "-c", str(NEST_INTERP_FRAGMENT_SHADER),
              "-o", str(nest_interp_fragment_air)])
        nest_stream_transfer_fragment_air = temp / "nest_stream_transfer_fragment.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-DGPUWM_WRF_EXACT=1", "-c", str(NEST_STREAM_TRANSFER_FRAGMENT_SHADER),
              "-o", str(nest_stream_transfer_fragment_air)])
        ieva_family_air = temp / "ieva_family.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-DGPUWM_WRF_EXACT=1", "-c", str(IEVA_FAMILY_SHADER),
              "-o", str(ieva_family_air)])
        radar_tten_family_air = temp / "radar_tten_family.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-DGPUWM_WRF_EXACT=1", "-c", str(RADAR_TTEN_FAMILY_SHADER),
              "-o", str(radar_tten_family_air)])
        ieva_solve_family_air = temp / "ieva_solve_family.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-DGPUWM_WRF_EXACT=1", "-c", str(IEVA_SOLVE_FAMILY_SHADER),
              "-o", str(ieva_solve_family_air)])
        # Periodic/edge face mass averaging shared by ordinary IEVA stage
        # preparation and state-free face helper callers.
        face_mass_air = temp / "face_mass.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(FACE_MASS_SHADER), "-o", str(face_mass_air)])
        complex64_default_air = temp / "complex64_primitives_default.air"
        complex64_exact_air = temp / "complex64_primitives_exact.air"
        for air_path, contraction in ((complex64_default_air, "fast"),
                                      (complex64_exact_air, "off")):
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise",
                  f"-ffp-contract={contraction}", "-c",
                  str(COMPLEX64_PRIMITIVES_SHADER), "-o", str(air_path)])
        # Binary64 Jacobi is an exact-policy numerical path in both shipped
        # libraries; software binary64 helpers make the arithmetic independent
        # of native Metal FP64 support.
        jacobi_eigh_fp64_default_air = temp / "jacobi_eigh_fp64_default.air"
        jacobi_eigh_fp64_exact_air = temp / "jacobi_eigh_fp64_exact.air"
        for air_path, contraction in ((jacobi_eigh_fp64_default_air, "fast"),
                                      (jacobi_eigh_fp64_exact_air, "off")):
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise",
                  f"-ffp-contract={contraction}", "-c",
                  str(JACOBI_EIGH_FP64_SHADER), "-o", str(air_path)])
        # Optional regional scalar spectral filter. This is a default-policy
        # FP32/Complex64 path and is admitted only for its typed entrypoints.
        spectral_scalar_air = temp / "spectral_scalar.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(SPECTRAL_SCALAR_SHADER), "-o",
              str(spectral_scalar_air)])
        sfclay_family_air = temp / "sfclay_family.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(SFCLAY_FAMILY_SHADER), "-o", str(sfclay_family_air)])
        mynn_surface_family_air = temp / "mynn_surface_family.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(MYNN_SURFACE_FAMILY_SHADER), "-o",
              str(mynn_surface_family_air)])
        # Ordinary WRF YSU and its standard surface-layer scalar source fold.
        # This kernel family is default-arithmetic only; exact-policy runs
        # remain refused by their public caller.
        ysu_column_air = temp / "ysu_column.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(YSU_COLUMN_SHADER), "-o", str(ysu_column_air)])
        ysu_480_column_air = temp / "ysu_480_column.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(YSU_480_COLUMN_SHADER), "-o", str(ysu_480_column_air)])
        # Ordinary Noah and the WRF Smagorinsky family use default-policy
        # arithmetic only. Exact-policy callers remain refused by their
        # Python submitters.
        physics_family_airs = []
        for name, source in (("noah_family", NOAH_FAMILY_SHADER),
                             ("noah_chklowq_myj", ROOT / "woof/metal/shaders/noah_chklowq_myj.metal"),
                             ("noah_forcing_family", NOAH_FORCING_FAMILY_SHADER),
                             ("smag2d_family", SMAG2D_FAMILY_SHADER),
                             ("lake_family", LAKE_FAMILY_SHADER),
                             ("hail_family", HAIL_FAMILY_SHADER)):
            air_path = temp / f"{name}.air"
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
                  "-c", str(source), "-o", str(air_path)])
            physics_family_airs.append(air_path)
        # The strict regional coefficient and tendency stages store binary32 as integer
        # words and is linked only into the exact-policy library.
        exact_smag_airs = []
        for stem in ("wrf_exact_smag_coefficients", "wrf_exact_smag_tendencies",
                     "wrf_exact_smag_vertical", "wrf_exact_smag_turbulence"):
            exact_smag_airs.append(temp / (stem + ".air"))
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
                  "-c", str(ROOT / "woof/metal/shaders" / (stem + ".metal")),
                  "-o", str(exact_smag_airs[-1])])
        noah_sfcdiags_air = temp / "noah_sfcdiags.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(NOAH_SFCDIAGS_SHADER), "-o", str(noah_sfcdiags_air)])
        health_family_air = temp / "health_family.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(HEALTH_FAMILY_SHADER), "-o", str(health_family_air)])
        # Ordinary monotonic sixth-order diffusion and its specified/open
        # staggered seam replacement kernels use default-policy arithmetic.
        diff6_family_air = temp / "diff6_family.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(DIFF6_FAMILY_SHADER), "-o", str(diff6_family_air)])
        exact_diff6_air = temp / "wrf_exact_diff6.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_diff6.metal"),
              "-o", str(exact_diff6_air)])
        exact_openbc_air = temp / "wrf_exact_openbc.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_openbc.metal"),
              "-o", str(exact_openbc_air)])
        # The UH column integral and its running-max fold retain the CUDA
        # exact expression order and are shipped only in the exact library.
        uh_diag_family_air = temp / "uh_diag_family.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-DGPUWM_WRF_EXACT=1", "-c", str(UH_DIAG_FAMILY_SHADER),
              "-o", str(uh_diag_family_air)])
        exact_uh_diag_air = temp / "wrf_exact_uh_diag.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_uh_diag.metal"),
              "-o", str(exact_uh_diag_air)])
        exact_noah_seam_air = temp / "wrf_exact_noah_seam.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_noah_seam.metal"),
              "-o", str(exact_noah_seam_air)])
        exact_noah_land_air = temp / "wrf_exact_noah_land.air"
        exact_cldfra1_air = temp / "wrf_exact_cldfra1.air"
        exact_goddard_airs = []
        for name in ("wrf_exact_goddard_family", "wrf_exact_goddard_grid", "wrf_exact_goddard_tiers",
                     "wrf_exact_direct_normal_flux"):
            air = temp / (name + ".air")
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
                  "-c", str(ROOT / "woof/metal/shaders" / (name + ".metal")),
                  "-o", str(air)])
            exact_goddard_airs.append(air)
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_cldfra1.metal"),
              "-o", str(exact_cldfra1_air)])
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_noah_land.metal"),
              "-o", str(exact_noah_land_air)])
        exact_noah_init_air = temp / "wrf_exact_noah_init.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_noah_init.metal"),
              "-o", str(exact_noah_init_air)])
        exact_noah_seaice_air = temp / "wrf_exact_noah_seaice.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_noah_seaice.metal"),
              "-o", str(exact_noah_seaice_air)])
        physics_atmosphere_prepare_air = temp / "physics_atmosphere_prepare.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(PHYSICS_ATMOSPHERE_PREPARE_SHADER), "-o",
              str(physics_atmosphere_prepare_air)])
        exact_physics_atmosphere_air = temp / "wrf_exact_physics_atmosphere.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_physics_atmosphere.metal"),
              "-o", str(exact_physics_atmosphere_air)])
        coupled_boundary_field_air = temp / "coupled_boundary_field.air"
        strict_coupled_boundary_field_air = temp / "strict_coupled_boundary_field.air"
        for shader, air in (
                (COUPLED_BOUNDARY_FIELD_SHADER, coupled_boundary_field_air),
                (ROOT / "woof/metal/shaders/strict_coupled_boundary_field.metal",
                 strict_coupled_boundary_field_air)):
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
                  "-c", str(shader), "-o", str(air)])
        surface_humidity_air = temp / "surface_humidity.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(SURFACE_HUMIDITY_SHADER), "-o",
              str(surface_humidity_air)])
        lbc_time_air = temp / "lbc_time.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(LBC_TIME_SHADER), "-o", str(lbc_time_air)])
        lbc_state_air = temp / "lbc_state.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(LBC_STATE_SHADER), "-o", str(lbc_state_air)])
        exact_lbc_on_air = temp / "wrf_exact_lbc_on.air"
        exact_lbc_off_air = temp / "wrf_exact_lbc_off.air"
        exact_lbc_interior_air = temp / "wrf_exact_lbc_interior.air"
        exact_lbc_install_air = temp / "wrf_exact_install_mu_boundary.air"
        exact_lbc_final_on_air = temp / "wrf_exact_lbc_final_on.air"
        exact_lbc_final_off_air = temp / "wrf_exact_lbc_final_off.air"
        exact_nest_uncouple_on_air = temp / "wrf_exact_nest_uncouple_on.air"
        exact_nest_uncouple_off_air = temp / "wrf_exact_nest_uncouple_off.air"
        exact_mynn_radiation_air = temp / "wrf_exact_mynn_radiation.air"
        exact_dudhia_air = temp / "wrf_exact_dudhia_shortwave.air"
        exact_held_heating_air = temp / "wrf_exact_held_heating.air"
        exact_nest_smooth_air = temp / "wrf_exact_nest_smooth.air"
        exact_nest_interp_air = temp / "wrf_exact_nest_interp.air"
        exact_nest_feedback_air = temp / "wrf_exact_nest_feedback.air"
        exact_nest_couple_on_air = temp / "wrf_exact_nest_couple_on.air"
        exact_nest_couple_off_air = temp / "wrf_exact_nest_couple_off.air"
        exact_nest_roundtrip_air = temp / "strict_nest_roundtrip_family.air"
        for shader, air in ((EXACT_MYNN_RADIATION_SHADER, exact_mynn_radiation_air),
                            (EXACT_DUDHIA_SHADER, exact_dudhia_air),
                            (EXACT_HELD_HEATING_SHADER, exact_held_heating_air),
                            (EXACT_NEST_SMOOTH_SHADER, exact_nest_smooth_air),
                            (EXACT_NEST_INTERP_SHADER, exact_nest_interp_air),
                            (EXACT_NEST_FEEDBACK_SHADER, exact_nest_feedback_air),
                            (EXACT_NEST_UNCOUPLE_ON_SHADER, exact_nest_uncouple_on_air),
                            (EXACT_NEST_UNCOUPLE_OFF_SHADER, exact_nest_uncouple_off_air),
                            (EXACT_NEST_COUPLE_ON_SHADER, exact_nest_couple_on_air),
                            (EXACT_NEST_COUPLE_OFF_SHADER, exact_nest_couple_off_air),
                            (EXACT_NEST_ROUNDTRIP_SHADER, exact_nest_roundtrip_air),
                            (EXACT_LBC_FINAL_ON_SHADER, exact_lbc_final_on_air),
                            (EXACT_LBC_FINAL_OFF_SHADER, exact_lbc_final_off_air),
                            (EXACT_LBC_INSTALL_SHADER, exact_lbc_install_air),
                            (EXACT_LBC_ON_SHADER, exact_lbc_on_air),
                            (EXACT_LBC_OFF_SHADER, exact_lbc_off_air),
                            (EXACT_LBC_INTERIOR_SHADER, exact_lbc_interior_air)):
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
                  "-c", str(shader), "-o", str(air)])
        lbc_flow_air = temp / "lbc_flow.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(LBC_FLOW_SHADER), "-o", str(lbc_flow_air)])
        exact_lbc_flow_air = temp / "wrf_exact_lbc_flow.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(EXACT_LBC_FLOW_SHADER), "-o", str(exact_lbc_flow_air)])
        surface_diagnostics_seed_air = temp / "surface_diagnostics_seed.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(SURFACE_DIAGNOSTICS_SEED_SHADER), "-o",
              str(surface_diagnostics_seed_air)])
        acoustic_sumflux_air = temp / "acoustic_sumflux.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ACOUSTIC_SUMFLUX_SHADER), "-o",
              str(acoustic_sumflux_air)])
        real_total_geopotential_air = temp / "real_total_geopotential.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(REAL_TOTAL_GEOPOTENTIAL_SHADER), "-o",
              str(real_total_geopotential_air)])
        rrtmgp_prepare_air = temp / "rrtmgp_prepare_family.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(RRTMGP_PREPARE_SHADER), "-o",
              str(rrtmgp_prepare_air)])
        exact_rrtm_t8w_air = temp / "wrf_exact_rrtm_t8w.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_rrtm_t8w.metal"),
              "-o", str(exact_rrtm_t8w_air)])
        exact_rrtm_mm5atm_air = temp / "wrf_exact_rrtm_mm5atm.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_rrtm_mm5atm.metal"),
              "-o", str(exact_rrtm_mm5atm_air)])
        exact_rrtm_setcoef_air = temp / "wrf_exact_rrtm_setcoef.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_rrtm_setcoef.metal"),
              "-o", str(exact_rrtm_setcoef_air)])
        exact_rrtm_band12_air = temp / "wrf_exact_rrtm_band12.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_rrtm_band12.metal"),
              "-o", str(exact_rrtm_band12_air)])
        exact_rrtm_band03_air = temp / "wrf_exact_rrtm_band03.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_rrtm_band03.metal"),
              "-o", str(exact_rrtm_band03_air)])
        exact_rrtm_band0405_air = temp / "wrf_exact_rrtm_band0405.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_rrtm_band0405.metal"),
              "-o", str(exact_rrtm_band0405_air)])
        exact_rrtm_band06_air = temp / "wrf_exact_rrtm_band06.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_rrtm_band06.metal"),
              "-o", str(exact_rrtm_band06_air)])
        exact_rrtm_band07_air = temp / "wrf_exact_rrtm_band07.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_rrtm_band07.metal"),
              "-o", str(exact_rrtm_band07_air)])
        exact_rrtm_band08_air = temp / "wrf_exact_rrtm_band08.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_rrtm_band08.metal"),
              "-o", str(exact_rrtm_band08_air)])
        exact_rrtm_band09_air = temp / "wrf_exact_rrtm_band09.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_rrtm_band09.metal"),
              "-o", str(exact_rrtm_band09_air)])
        exact_rrtm_band1011_air = temp / "wrf_exact_rrtm_band1011.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_rrtm_band1011.metal"),
              "-o", str(exact_rrtm_band1011_air)])
        exact_rrtm_band1216_air = temp / "wrf_exact_rrtm_band1216.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_rrtm_band1216.metal"),
              "-o", str(exact_rrtm_band1216_air)])
        exact_rrtm_itr_air = temp / "wrf_exact_rrtm_itr.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_rrtm_itr.metal"),
              "-o", str(exact_rrtm_itr_air)])
        exact_rrtm_rtrn_air = temp / "wrf_exact_rrtm_rtrn.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_rrtm_rtrn.metal"),
              "-o", str(exact_rrtm_rtrn_air)])
        rrtm_lw_family_air = temp / "rrtm_lw_family.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(RRTM_LW_FAMILY_SHADER), "-o",
              str(rrtm_lw_family_air)])
        rrtmg_legacy_lw_family_air = temp / "rrtmg_legacy_lw_family.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(RRTMG_LEGACY_LW_FAMILY_SHADER), "-o",
              str(rrtmg_legacy_lw_family_air)])
        rrtmgp_mcica_air = temp / "rrtmgp_mcica_family.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(RRTMGP_MCICA_SHADER), "-o", str(rrtmgp_mcica_air)])
        hex_mass_flux_air = temp / "hex_mass_flux_divergence.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(HEX_MASS_FLUX_SHADER), "-o", str(hex_mass_flux_air)])
        rrtmgp_family_airs = []
        for name, source in (
                ("rrtmgp_gas_family", RRTMGP_GAS_SHADER),
                ("rrtmgp_cloud_prep_family", RRTMGP_CLOUD_PREP_SHADER),
                ("rrtmgp_cloud_family", RRTMGP_CLOUD_SHADER),
                ("rrtmgp_rte_family", RRTMGP_RTE_SHADER),
                ("kessler_family", KESSLER_FAMILY_SHADER),
                ("strict_regional_kessler_prep", STRICT_REGIONAL_KESSLER_PREP_SHADER),
                ("strict_regional_kessler", STRICT_REGIONAL_KESSLER_COLUMN_SHADER),
                ("strict_regional_kessler_finish", STRICT_REGIONAL_KESSLER_FINISH_SHADER),
                ("wrf_exact_glue_add_positive", ROOT / "woof/metal/shaders/wrf_exact_glue_add_positive.metal"),
                ("strict_regional_wsm6_prep", ROOT / "woof/metal/shaders/strict_regional_wsm6_prep.metal"),
                ("strict_regional_wsm6", ROOT / "woof/metal/shaders/strict_regional_wsm6.metal"),
                ("strict_regional_wsm6_deep", ROOT / "woof/metal/shaders/strict_regional_wsm6_deep.metal"),
                ("wsm6_family", WSM6_FAMILY_SHADER),
                ("wdm6_family", WDM6_FAMILY_SHADER),
                ("physics_correct_mode", ROOT / "woof/metal/shaders/physics_correct_mode.metal"),
                ("nssl2_reduce_exact", ROOT / "woof/metal/shaders/nssl2_reduce_exact.metal")):
            air_path = temp / f"{name}.air"
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
                  "-c", str(source), "-o", str(air_path)])
            rrtmgp_family_airs.append(air_path)
        climate_exact_air = temp / "climate_diagnostics_exact.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/climate_diagnostics_exact.metal"),
              "-o", str(climate_exact_air)])
        sfclay_exact_air = temp / "sfclay_exact.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/sfclay_exact.metal"),
              "-o", str(sfclay_exact_air)])
        kf_w0avg_air = temp / "kf_w0avg_update.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(KF_W0AVG_SHADER), "-o", str(kf_w0avg_air)])
        # Ordinary KF, Morrison MP10, and their compact clock/status support
        # use only the default arithmetic Metal library. Exact callers remain
        # refused by the physics adapters.
        physics_column_airs = []
        noahmp_glacier_air = None
        for name, source in (
                ("gabls_column", GABLS_COLUMN_SHADER),
                ("case_boundary_mass_flux", CASE_BOUNDARY_MASS_FLUX_SHADER),
                ("kf_column_family", KF_COLUMN_FAMILY_SHADER),
                ("kf_output_validation", KF_OUTPUT_VALIDATION_SHADER),
                ("morrison_family", MORRISON_FAMILY_SHADER),
                ("mp_zero_out", MP_ZERO_OUT_SHADER),
                ("morrison_refl", MORRISON_REFL_SHADER),
                ("wsm6_reflectivity", WSM6_REFLECTIVITY_SHADER),
                ("refl_family", REFL_FAMILY_SHADER),
                ("noah_mosaic_family", NOAH_MOSAIC_SHADER),
                ("noah_mosaic_ucm_family", ROOT / "woof/metal/shaders/noah_mosaic_ucm_family.metal"),
                ("milbrandt2_family", MILBRANDT2_FAMILY_SHADER),
                ("milbrandt2_fields", MILBRANDT2_FIELDS_SHADER),
                ("milbrandt2_zet", MILBRANDT2_ZET_SHADER),
                ("wdm6_reflectivity", WDM6_REFLECTIVITY_SHADER),
                ("noahmp_driver_family", NOAHMP_DRIVER_FAMILY_SHADER),
                ("noahmp_sflx_family", NOAHMP_SFLX_FAMILY_SHADER),
                ("noahmp_leaves_family", NOAHMP_LEAVES_FAMILY_SHADER),
                ("noahmp_vegprecip_family", NOAHMP_VEGPRECIP_FAMILY_SHADER),
                ("noahmp_thermal_family", NOAHMP_THERMAL_FAMILY_SHADER),
                ("noahmp_bareflux_family", NOAHMP_BAREFLUX_FAMILY_SHADER),
                ("noahmp_vegeflux_family", NOAHMP_VEGEFLUX_FAMILY_SHADER),
                ("noahmp_water_family", NOAHMP_WATER_FAMILY_SHADER),
                ("noahmp_radiation_family", NOAHMP_RADIATION_FAMILY_SHADER),
                ("noahmp_glacier_family", NOAHMP_GLACIER_FAMILY_SHADER),
                ("ruc_family", RUC_FAMILY_SHADER),
                ("ruc_sfctmp_family", RUC_SFCTMP_FAMILY_SHADER),
                ("ruc_driver_family", RUC_DRIVER_FAMILY_SHADER),
                ("ruc_sfclay_seaice", RUC_SFCLAY_SEAICE_SHADER),
                ("sase_family", SASE_FAMILY_SHADER),
                ("mynn_pbl_family", MYNN_PBL_FAMILY_SHADER),
                ("myjpbl_family", MYJPBL_FAMILY_SHADER),
                ("myjsfc_family", MYJSFC_FAMILY_SHADER),
                ("myjurb_family", MYJURB_FAMILY_SHADER),
                ("ntiedtke_family", NTIEDTKE_FAMILY_SHADER),
                ("shinhong_column", SHINHONG_COLUMN_SHADER),
                ("gf_family", GF_FAMILY_SHADER),
                ("gf_family_k256", ROOT / "woof/metal/shaders/gf_family_k256.metal"),
                ("nssl2_fused_gs", NSSL2_FUSED_GS_SHADER),
                ("nssl2_nucond", NSSL2_NUCOND_SHADER),
                ("nssl2_driver_transport", NSSL2_DRIVER_TRANSPORT_SHADER),
                ("analytic_radiation", ROOT / "woof/metal/shaders/analytic_radiation.metal"),
                ("diff_opt1", ROOT / "woof/metal/shaders/diff_opt1.metal"),
                ("da_state", ROOT / "woof/metal/shaders/da_state.metal"),
                ("da_masked_wind", ROOT / "woof/metal/shaders/da_masked_wind.metal"),
                ("hexrrfs_w_boundary", ROOT / "woof/metal/shaders/hexrrfs_w_boundary.metal"),
                ("da_hydrostatic", ROOT / "woof/metal/shaders/da_hydrostatic.metal"),
                ("da_hotstart_pointwise", ROOT / "woof/metal/shaders/da_hotstart_pointwise.metal"),
                ("da_hotstart_height", ROOT / "woof/metal/shaders/da_hotstart_height.metal"),
                ("da_hotstart_ramp", ROOT / "woof/metal/shaders/da_hotstart_ramp.metal"),
                ("da_iau_rate", ROOT / "woof/metal/shaders/da_iau_rate.metal"),
                ("da_iau_couple", ROOT / "woof/metal/shaders/da_iau_couple.metal"),
                ("da_child_positivity", ROOT / "woof/metal/shaders/da_child_positivity.metal"),
                ("tke_budget", ROOT / "woof/metal/shaders/tke_budget.metal"),
                ("nssl2_sedimentation", NSSL2_SEDIMENTATION_SHADER),
                ("nssl2_diagnostics", NSSL2_DIAGNOSTICS_SHADER),
                ("nssl2_runtime", NSSL2_RUNTIME_SHADER),
                ("p3_entry", P3_ENTRY_SHADER),
                ("p3_kloop1", P3_KLOOP1_SHADER),
                ("p3_kloopmain", P3_KLOOPMAIN_SHADER),
                ("p3_saveold_precip", P3_SAVEOLD_PRECIP_SHADER),
                ("p3_homofreeze", P3_HOMOFREEZE_SHADER),
                ("p3_final", P3_FINAL_SHADER),
                ("p3_sed_cloud", P3_SED_CLOUD_SHADER),
                ("p3_sed_rain", P3_SED_RAIN_SHADER),
                ("p3_sed_ice", P3_SED_ICE_SHADER),
                ("uwpbl_family", UWPBL_FAMILY_SHADER),
                ("cumulus_clock_family", CUMULUS_CLOCK_FAMILY_SHADER),
                ("nudging_family", ROOT / "woof/metal/shaders/nudging_family.metal"),
                ("nest_microphysics_family", ROOT / "woof/metal/shaders/nest_microphysics_family.metal"),
                ("physics_params_family", ROOT / "woof/metal/shaders/physics_params_family.metal")):
            air_path = temp / f"{name}.air"
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
                  "-c", str(source), "-o", str(air_path)])
            physics_column_airs.append(air_path)
            if name == "noahmp_glacier_family":
                noahmp_glacier_air = air_path
        if noahmp_glacier_air is None:
            raise RuntimeError("Noah-MP glacier shader was not compiled")
        global_spectral_diffusion_air = temp / "global_spectral_diffusion.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(GLOBAL_SPECTRAL_DIFFUSION_SHADER), "-o",
              str(global_spectral_diffusion_air)])
        integration_fragment_airs = []
        for name, source in (
                ("wrf_exact_tke_budget", ROOT / "woof/metal/shaders/wrf_exact_tke_budget.metal"),
                ("wrf_exact_diff_opt1", ROOT / "woof/metal/shaders/wrf_exact_diff_opt1.metal"),
                ("solar_geometry", ROOT / "woof/metal/shaders/solar_geometry.metal"),
                ("ensemble_surface_recipe_fragment", ROOT / "woof/metal/shaders/ensemble_surface_recipe_fragment.metal"),
                ("hex_solve_vertical_system_fragment", HEX_SOLVE_VERTICAL_SYSTEM_SHADER),
                ("hex_acoustic_edge_update_fragment", ROOT / "woof/metal/shaders/hex_acoustic_edge_update_fragment.metal"),
                ("hex_acoustic_postsolve_damping_fragment", ROOT / "woof/metal/shaders/hex_acoustic_postsolve_damping_fragment.metal"),
                ("hex_theta_diagnostic_fragment", ROOT / "woof/metal/shaders/hex_theta_diagnostic_fragment.metal"),
                ("hex_exner_pressure_diagnostic_fragment", ROOT / "woof/metal/shaders/hex_exner_pressure_diagnostic_fragment.metal"),
                ("hex_rk_tendencies_fragment", HEX_RK_TENDENCIES_FRAGMENT_SHADER),
                ("hex_interface_velocity_terrain_fragment", HEX_INTERFACE_VELOCITY_TERRAIN_FRAGMENT_SHADER),
                ("hex_mesh_diagnostics_fragment", HEX_MESH_DIAGNOSTICS_FRAGMENT_SHADER),
                ("hex_tangential_velocity_fragment", ROOT / "woof/metal/shaders/hex_tangential_velocity_fragment.metal"),
                ("hex_kinematic_diagnostics_fragment", ROOT / "woof/metal/shaders/hex_kinematic_diagnostics_fragment.metal"),
                ("hex_pv_diagnostics_fragment", ROOT / "woof/metal/shaders/hex_pv_diagnostics_fragment.metal"),
                ("hex_vertical_momentum_transport_fragment", ROOT / "woof/metal/shaders/hex_vertical_momentum_transport_fragment.metal"),
                ("solar_albedo_fragment", ROOT / "woof/metal/shaders/solar_albedo_fragment.metal"),
                ("hex_vertical_velocity_tendency_fragment", ROOT / "woof/metal/shaders/hex_vertical_velocity_tendency_fragment.metal"),
                ("hex_vertical_coefficients_v841_fragment", ROOT / "woof/metal/shaders/hex_vertical_coefficients_v841_fragment.metal"),
                ("hex_moist_coefficients_v841_fragment", ROOT / "woof/metal/shaders/hex_moist_coefficients_v841_fragment.metal"),
                ("hex_vector_invariant_momentum_fragment", ROOT / "woof/metal/shaders/hex_vector_invariant_momentum_fragment.metal"),
                ("hex_vector_invariant_momentum_v841_fragment", ROOT / "woof/metal/shaders/hex_vector_invariant_momentum_v841_fragment.metal"),
                ("hex_omega_terrain_conversion_fragment", ROOT / "woof/metal/shaders/hex_omega_terrain_conversion_fragment.metal"),
                ("hex_dynamics", ROOT / "woof/metal/shaders/hex_pressure_gradient_fragment.metal"),
                ("hex_vertical_pressure_euler_fragment", ROOT / "woof/metal/shaders/hex_vertical_pressure_euler_fragment.metal"),
                ("hex_density_tendency_fragment", ROOT / "woof/metal/shaders/hex_density_tendency_fragment.metal"),
                ("hex_dpdz_tendency_fragment", ROOT / "woof/metal/shaders/hex_dpdz_tendency_fragment.metal"),
                ("hex_horizontal_theta_tendency_fragment", ROOT / "woof/metal/shaders/hex_horizontal_theta_tendency_fragment.metal"),
                ("hex_vertical_theta_tendency_fragment", ROOT / "woof/metal/shaders/hex_vertical_theta_tendency_fragment.metal"),
                ("hex_vertical_scalar_flux3_fragment", ROOT / "woof/metal/shaders/hex_vertical_scalar_flux3_fragment.metal"),
                ("hex_pressure_perturbation_fragment", ROOT / "woof/metal/shaders/hex_pressure_perturbation_fragment.metal"),
                ("global_tracer_flux_fragment", GLOBAL_TRACER_FLUX_FRAGMENT_SHADER),
                ("global_tracer_gap_fragment", GLOBAL_TRACER_GAP_FRAGMENT_SHADER),
                ("global_scalar_tendency_fragment", ROOT / "woof/metal/shaders/global_scalar_tendency_fragment.metal"),
                ("global_tracer_zonal_fragment", ROOT / "woof/metal/shaders/global_tracer_zonal_fragment.metal"),
                ("global_tracer_walled_fragment", ROOT / "woof/metal/shaders/global_tracer_walled_fragment.metal"),
                ("global_fixer_fragment", ROOT / "woof/metal/shaders/global_fixer_fragment.metal"),
                ("global_column_water_stack_fragment", ROOT / "woof/metal/shaders/global_column_water_stack_fragment.metal"),
                ("global_convective_pack_fragment", ROOT / "woof/metal/shaders/global_convective_pack_fragment.metal"),
                ("global_convective_adjust_fragment", ROOT / "woof/metal/shaders/global_convective_adjust_fragment.metal"),
                ("global_convective_count_finalize_fragment", ROOT / "woof/metal/shaders/global_convective_count_finalize_fragment.metal"),
                ("global_surface_flux_fragment", ROOT / "woof/metal/shaders/global_surface_flux_fragment.metal"),
                ("global_surface_emission_fragment", ROOT / "woof/metal/shaders/global_surface_emission_fragment.metal"),
                ("global_soil_temperature_relax_fragment", ROOT / "woof/metal/shaders/global_soil_temperature_relax_fragment.metal"),
                ("global_surface_fallout_cooling_fragment", ROOT / "woof/metal/shaders/global_surface_fallout_cooling_fragment.metal"),
                ("global_surface_energy_update_fragment", ROOT / "woof/metal/shaders/global_surface_energy_update_fragment.metal"),
                ("global_stratospheric_floor_fragment", ROOT / "woof/metal/shaders/global_stratospheric_floor_fragment.metal"),
                ("global_latitude_transform_fragment", ROOT / "woof/metal/shaders/global_latitude_transform_fragment.metal"),
                ("global_reductions_fragment", ROOT / "woof/metal/shaders/global_reductions_fragment.metal"),
                ("global_dynamics_fragment", ROOT / "woof/metal/shaders/global_dynamics_fragment.metal"),
                ("global_vertical_continuity_fragment", ROOT / "woof/metal/shaders/global_vertical_continuity_fragment.metal"),
                ("global_vertical_scalar_flux_fragment", ROOT / "woof/metal/shaders/global_vertical_scalar_flux_fragment.metal"),
                ("global_vertical_momentum_fragment", ROOT / "woof/metal/shaders/global_vertical_momentum_fragment.metal"),
                ("global_pressure_gradient_fragment", ROOT / "woof/metal/shaders/global_pressure_gradient_fragment.metal"),
                ("global_hybrid_pressure_setup_fragment", ROOT / "woof/metal/shaders/global_hybrid_pressure_setup_fragment.metal"),
                ("global_surface_pressure_tendency_fragment", ROOT / "woof/metal/shaders/global_surface_pressure_tendency_fragment.metal"),
                ("global_grid_mean_fragment", ROOT / "woof/metal/shaders/global_grid_mean_fragment.metal"),
                ("global_gray_radiation_fragment", ROOT / "woof/metal/shaders/global_gray_radiation_fragment.metal"),
                ("global_gray_solar_fragment", ROOT / "woof/metal/shaders/global_gray_solar_fragment.metal"),
                ("global_gray_radiation_optics_fragment", ROOT / "woof/metal/shaders/global_gray_radiation_optics_fragment.metal"),
                ("global_gray_radiation_update_fragment", ROOT / "woof/metal/shaders/global_gray_radiation_update_fragment.metal"),
                ("global_semilag_gather_fragment", ROOT / "woof/metal/shaders/global_semilag_gather_fragment.metal"),
                ("global_semilag_gather5_fragment", ROOT / "woof/metal/shaders/global_semilag_gather5_fragment.metal"),
                ("global_semilag_departure_fragment", ROOT / "woof/metal/shaders/global_semilag_departure_fragment.metal"),
                ("global_semilag_tendencies_fragment", ROOT / "woof/metal/shaders/global_semilag_tendencies_fragment.metal"),
                ("global_semilag_operator_fragment", ROOT / "woof/metal/shaders/global_semilag_operator_fragment.metal"),
                ("global_semilag_gradient_metric_fragment", ROOT / "woof/metal/shaders/global_semilag_gradient_metric_fragment.metal"),
                ("global_semilag_force_fragment", ROOT / "woof/metal/shaders/global_semilag_force_fragment.metal"),
                ("global_level_water_mass_finalize_fragment", ROOT / "woof/metal/shaders/global_level_water_mass_finalize_fragment.metal"),
                ("global_level_water_mass_total_fragment", ROOT / "woof/metal/shaders/global_level_water_mass_total_fragment.metal"),
                ("global_column_water_before_fragment", ROOT / "woof/metal/shaders/global_column_water_before_fragment.metal"),
                ("global_implicit_diffusion_pack_fragment", ROOT / "woof/metal/shaders/global_implicit_diffusion_pack_fragment.metal"),
                ("global_surface_flux_prep_fragment", ROOT / "woof/metal/shaders/global_surface_flux_prep_fragment.metal"),
                ("global_surface_flux_prep_resident_fragment", ROOT / "woof/metal/shaders/global_surface_flux_prep_resident_fragment.metal"),
                ("global_surface_water_add_fragment", ROOT / "woof/metal/shaders/global_surface_water_add_fragment.metal"),
                ("global_semilag_max_finalize_fragment", ROOT / "woof/metal/shaders/global_semilag_max_finalize_fragment.metal"),
                ("global_saturation_pair_max_fragment", ROOT / "woof/metal/shaders/global_saturation_pair_max_fragment.metal"),
                ("global_saturation_closure_fragment", ROOT / "woof/metal/shaders/global_saturation_closure_fragment.metal"),
                ("global_saturation_closure_finalize_fragment", ROOT / "woof/metal/shaders/global_saturation_closure_finalize_fragment.metal"),
                ("global_sum_finalize_fragment", ROOT / "woof/metal/shaders/global_sum_finalize_fragment.metal"),
                ("global_total_water_stores_fragment", ROOT / "woof/metal/shaders/global_total_water_stores_fragment.metal"),
                ("global_any_u32_fragment", ROOT / "woof/metal/shaders/global_any_u32_fragment.metal"),
                ("global_cfl_wind_fragment", ROOT / "woof/metal/shaders/global_cfl_wind_fragment.metal"),
                ("global_semilag_assemble_fragment", ROOT / "woof/metal/shaders/global_semilag_assemble_fragment.metal"),
                ("global_semilag_add_scaled_fragment", ROOT / "woof/metal/shaders/global_semilag_add_scaled_fragment.metal"),
                ("global_semilag_solve_shifted_fragment", ROOT / "woof/metal/shaders/global_semilag_solve_shifted_fragment.metal"),
                ("global_semilag_half_apply_fragment", ROOT / "woof/metal/shaders/global_semilag_half_apply_fragment.metal"),
                ("global_semilag_linear_tendencies_fragment", ROOT / "woof/metal/shaders/global_semilag_linear_tendencies_fragment.metal"),
                ("global_semilag_max_fragment", ROOT / "woof/metal/shaders/global_semilag_max_fragment.metal"),
                ("global_semilag_lipschitz_fragment", ROOT / "woof/metal/shaders/global_semilag_lipschitz_fragment.metal"),
                ("global_semilag_lipschitz_norms_fragment", ROOT / "woof/metal/shaders/global_semilag_lipschitz_norms_fragment.metal"),
                ("global_semilag_water_column_fragment", ROOT / "woof/metal/shaders/global_semilag_water_column_fragment.metal"),
                ("global_semilag_complex_difference_fragment", ROOT / "woof/metal/shaders/global_semilag_complex_difference_fragment.metal"),
                ("global_semilag_vector_fragment", ROOT / "woof/metal/shaders/global_semilag_vector_fragment.metal"),
                ("global_semilag_pressure_fragment", ROOT / "woof/metal/shaders/global_semilag_pressure_fragment.metal"),
                ("global_semilag_transport_fragment", ROOT / "woof/metal/shaders/global_semilag_transport_fragment.metal"),
                ("global_semilag_wind_pair_fragment", ROOT / "woof/metal/shaders/global_semilag_wind_pair_fragment.metal"),
                ("global_semilag_vector_analysis_fragment", ROOT / "woof/metal/shaders/global_semilag_vector_analysis_fragment.metal"),
                ("global_semilag_tracer_fix_fragment", ROOT / "woof/metal/shaders/global_semilag_tracer_fix_fragment.metal"),
                ("global_tracer_resident_fragment", ROOT / "woof/metal/shaders/global_tracer_resident_fragment.metal"),
                ("global_semilag_transport_resident_fragment", ROOT / "woof/metal/shaders/global_semilag_transport_resident_fragment.metal"),
                ("global_saturation_adjust_fragment", ROOT / "woof/metal/shaders/global_saturation_adjust_fragment.metal"),
                ("global_microphysics_fragment", ROOT / "woof/metal/shaders/global_microphysics_fragment.metal"),
                ("global_fallout_fragment", ROOT / "woof/metal/shaders/global_fallout_fragment.metal"),
                ("global_betts_miller_fragment", ROOT / "woof/metal/shaders/global_betts_miller_fragment.metal"),
                ("global_betts_miller_temperature_fragment", ROOT / "woof/metal/shaders/global_betts_miller_temperature_fragment.metal"),
                ("global_betts_miller_theta_fragment", ROOT / "woof/metal/shaders/global_betts_miller_theta_fragment.metal"),
                ("global_betts_miller_deep_finalize_fragment", ROOT / "woof/metal/shaders/global_betts_miller_deep_finalize_fragment.metal"),
                ("global_implicit_diffusion_fragment", ROOT / "woof/metal/shaders/global_implicit_diffusion_fragment.metal"),
                ("global_screen_diagnostics_fragment", ROOT / "woof/metal/shaders/global_screen_diagnostics_fragment.metal"),
                ("global_floor_grid_tracer_fragment", ROOT / "woof/metal/shaders/global_floor_grid_tracer_fragment.metal"),
                ("global_tracer_minmax_fragment", ROOT / "woof/metal/shaders/global_tracer_minmax_fragment.metal"),
                ("global_tracer_minmax_finalize_fragment", ROOT / "woof/metal/shaders/global_tracer_minmax_finalize_fragment.metal"),
                ("global_enforce_vapor_rows_fragment", ROOT / "woof/metal/shaders/global_enforce_vapor_rows_fragment.metal"),
                ("global_finite_fragment", ROOT / "woof/metal/shaders/global_finite_fragment.metal"),
                ("global_turbulence_geometry_fragment", ROOT / "woof/metal/shaders/global_turbulence_geometry_fragment.metal"),
                ("global_convective_mass_fragment", ROOT / "woof/metal/shaders/global_convective_mass_fragment.metal"),
                ("global_top_sponge_fragment", ROOT / "woof/metal/shaders/global_top_sponge_fragment.metal"),
                ("global_physics_exner_fragment", ROOT / "woof/metal/shaders/global_physics_exner_fragment.metal"),
                ("global_tracer_extrema_fragment", ROOT / "woof/metal/shaders/global_tracer_extrema_fragment.metal"),
                ("global_tracer_extrema_finalize_fragment", ROOT / "woof/metal/shaders/global_tracer_extrema_finalize_fragment.metal"),
                ("global_insitu_ledger_fragment", ROOT / "woof/metal/shaders/global_insitu_ledger_fragment.metal"),
                ("global_insitu_capture_fragment", ROOT / "woof/metal/shaders/global_insitu_capture_fragment.metal"),
                ("global_insitu_spectra_fragment", ROOT / "woof/metal/shaders/global_insitu_spectra_fragment.metal"),
                ("global_insitu_energy_fragment", ROOT / "woof/metal/shaders/global_insitu_energy_fragment.metal"),
                ("global_level_water_mass_rows_fragment", ROOT / "woof/metal/shaders/global_level_water_mass_rows_fragment.metal"),
                ("global_fix_total_water_fragment", ROOT / "woof/metal/shaders/global_fix_total_water_fragment.metal"),
                ("global_longitude_transform_fragment", GLOBAL_LONGITUDE_TRANSFORM_FRAGMENT_SHADER),
                ("global_longitude_transform_resident_fragment", ROOT / "woof/metal/shaders/global_longitude_transform_resident_fragment.metal"),
                ("global_latitude_transform_resident_fragment", ROOT / "woof/metal/shaders/global_latitude_transform_resident_fragment.metal"),
                ("global_inverse_longitude_transform_fragment", GLOBAL_INVERSE_LONGITUDE_TRANSFORM_FRAGMENT_SHADER),
                ("global_inverse_longitude_transform_resident_fragment", ROOT / "woof/metal/shaders/global_inverse_longitude_transform_resident_fragment.metal"),
                ("global_momentum_fragment", GLOBAL_MOMENTUM_FRAGMENT_SHADER),
                ("global_grid_plane_broadcast_fragment", ROOT / "woof/metal/shaders/global_grid_plane_broadcast_fragment.metal"),
                ("global_spectral_project_fragment", ROOT / "woof/metal/shaders/global_spectral_project_fragment.metal"),
                ("hex_scalar_fct_fragment", ROOT / "woof/metal/shaders/hex_scalar_fct_fragment.metal"),
                ("hex_scalar_rk_update_fragment", ROOT / "woof/metal/shaders/hex_scalar_rk_update_fragment.metal"),
                ("hex_fct_cell_scales_fragment", ROOT / "woof/metal/shaders/hex_fct_cell_scales_fragment.metal"),
                ("hex_fct_apply_fragment", ROOT / "woof/metal/shaders/hex_fct_apply_fragment.metal"),
                ("hex_fct_edge_flux_fragment", ROOT / "woof/metal/shaders/hex_fct_edge_flux_fragment.metal"),
                ("hex_interface_velocity_recovery_fragment", ROOT / "woof/metal/shaders/hex_interface_velocity_recovery_fragment.metal"),
                ("wrf_exact_rrtmg_lw_transfer", ROOT / "woof/metal/shaders/wrf_exact_rrtmg_lw_transfer.metal"),
                ("rrtmgp_exact_lw", ROOT / "woof/metal/shaders/rrtmgp_exact_lw.metal"),
                ("rrtmgp_exact_sw", ROOT / "woof/metal/shaders/rrtmgp_exact_sw.metal"),
                ("rrtmgp_exact_interpolation", ROOT / "woof/metal/shaders/rrtmgp_exact_interpolation.metal"),
                ("rrtmgp_exact_gas_optics", ROOT / "woof/metal/shaders/rrtmgp_exact_gas_optics.metal"),
                ("rrtmgp_exact_cloud_optics", ROOT / "woof/metal/shaders/rrtmgp_exact_cloud_optics.metal"),
                ("rrtmgp_exact_outputs", ROOT / "woof/metal/shaders/rrtmgp_exact_outputs.metal"),
                ("rrtmgp_exact_profile_prepare", ROOT / "woof/metal/shaders/rrtmgp_exact_profile_prepare.metal"),
                ("rrtmgp_exact_hydrometeor_paths", ROOT / "woof/metal/shaders/rrtmgp_exact_hydrometeor_paths.metal"),
                ("hexrrfs_mynnsfc_v600", ROOT / "woof/metal/shaders/hexrrfs_mynnsfc_v600.metal"),
                ("hexrrfs_mynnsfc_coupling", ROOT / "woof/metal/shaders/hexrrfs_mynnsfc_coupling.metal"),
                ("hexrrfs_mynnedmf_columns", ROOT / "woof/metal/shaders/hexrrfs_mynnedmf_columns.metal"),
                ("hexrrfs_mynnedmf_old_columns", ROOT / "woof/metal/shaders/hexrrfs_mynnedmf_old_columns.metal"),
                ("hexrrfs_mpas_monthly_surface_f32", ROOT / "woof/metal/shaders/hexrrfs_mpas_monthly_surface_f32.metal"),
                ("hexrrfs_tempo_columns", ROOT / "woof/metal/shaders/hexrrfs_tempo_columns.metal"),
                ("hexrrfs_tempo_native_prep", ROOT / "woof/metal/shaders/hexrrfs_tempo_native_prep.metal"),
                ("hexrrfs_tempo_native_tail", ROOT / "woof/metal/shaders/hexrrfs_tempo_native_tail.metal"),
                ("hexrrfs_tempo_native_diagnostics", ROOT / "woof/metal/shaders/hexrrfs_tempo_native_diagnostics.metal"),
                ("hexrrfs_tempo_native_init", ROOT / "woof/metal/shaders/hexrrfs_tempo_native_init.metal"),
                ("hexrrfs_ruc_boundaries", ROOT / "woof/metal/shaders/hexrrfs_ruc_boundaries.metal"),
                ("hexrrfs_ruc_tbq", ROOT / "woof/metal/shaders/hexrrfs_ruc_tbq.metal"),
                ("hexrrfs_ruc_ice", ROOT / "woof/metal/shaders/hexrrfs_ruc_ice.metal"),
                ("hexrrfs_ruc_diagnostics", ROOT / "woof/metal/shaders/hexrrfs_ruc_diagnostics.metal"),
                ("hexrrfs_ruc_initialize", ROOT / "woof/metal/shaders/hexrrfs_ruc_initialize.metal"),
                ("hexrrfs_ruc_land_preflight", ROOT / "woof/metal/shaders/hexrrfs_ruc_land_preflight.metal"),
                ("hexrrfs_ruc_land", ROOT / "woof/metal/shaders/hexrrfs_ruc_land.metal"),
                ("hexrrfs_init_pools", ROOT / "woof/metal/shaders/hexrrfs_init_pools.metal"),
                ("hexrrfs_cloudiness", ROOT / "woof/metal/shaders/hexrrfs_cloudiness.metal"),
                ("hexrrfs_ugwp_oro", ROOT / "woof/metal/shaders/hexrrfs_ugwp_oro.metal"),
                ("wrf_exact_rrtmg_lw_inatm", ROOT / "woof/metal/shaders/wrf_exact_rrtmg_lw_inatm.metal"),
                ("wrf_exact_rrtmg_lw_setcoef", ROOT / "woof/metal/shaders/wrf_exact_rrtmg_lw_setcoef.metal"),
                ("wrf_exact_rrtmg_lw_cldprmc", ROOT / "woof/metal/shaders/wrf_exact_rrtmg_lw_cldprmc.metal"),
                ("rrtmg_legacy_lw_strict_gas", ROOT / "woof/metal/shaders/rrtmg_legacy_lw_strict_gas.metal"),
                ("urban_ucm_noahmp_after_lsm_fragment", ROOT / "woof/metal/shaders/urban_ucm_noahmp_after_lsm_fragment.metal"),
                ("urban_ucm480_family", ROOT / "woof/metal/shaders/urban_ucm480_family.metal"),
                ("urban_ucm_columns_fragment", ROOT / "woof/metal/shaders/urban_ucm_columns_fragment.metal"),
                ("urban_bep_family", ROOT / "woof/metal/shaders/urban_bep_family.metal"),
                ("urban_bep_couple_family", ROOT / "woof/metal/shaders/urban_bep_couple_family.metal"),
                ("urban_bep_radiation_fragment", ROOT / "woof/metal/shaders/urban_bep_radiation_fragment.metal"),
                ("urban_bep_radiosity_fragment", ROOT / "woof/metal/shaders/urban_bep_radiosity_fragment.metal"),
                ("urban_bep_surface_temperature_fragment", ROOT / "woof/metal/shaders/urban_bep_surface_temperature_fragment.metal"),
                ("urban_bem_view_factors_fragment", ROOT / "woof/metal/shaders/urban_bem_view_factors_fragment.metal"),
                ("urban_bem_view_inputs_fragment", ROOT / "woof/metal/shaders/urban_bem_view_inputs_fragment.metal"),
                ("urban_bem_profile_init_fragment", ROOT / "woof/metal/shaders/urban_bem_profile_init_fragment.metal"),
                ("urban_bem_height_init_fragment", ROOT / "woof/metal/shaders/urban_bem_height_init_fragment.metal"),
                ("urban_bem_temperature_average_fragment", ROOT / "woof/metal/shaders/urban_bem_temperature_average_fragment.metal"),
                ("urban_bem_temperature_profile_average_fragment", ROOT / "woof/metal/shaders/urban_bem_temperature_profile_average_fragment.metal"),
                ("urban_bem_shadow_mas_fragment", ROOT / "woof/metal/shaders/urban_bem_shadow_mas_fragment.metal"),
                ("urban_bem_roof_radiation_fragment", ROOT / "woof/metal/shaders/urban_bem_roof_radiation_fragment.metal"),
                ("urban_bem_occupancy_fragment", ROOT / "woof/metal/shaders/urban_bem_occupancy_fragment.metal"),
                ("urban_bem_occupancy_inputs_fragment", ROOT / "woof/metal/shaders/urban_bem_occupancy_inputs_fragment.metal"),
                ("urban_bem_indoor_radiation_fragment", ROOT / "woof/metal/shaders/urban_bem_indoor_radiation_fragment.metal"),
                ("urban_bem_indoor_fluxes_fragment", ROOT / "woof/metal/shaders/urban_bem_indoor_fluxes_fragment.metal"),
                ("urban_bem_room_balance_inputs_fragment", ROOT / "woof/metal/shaders/urban_bem_room_balance_inputs_fragment.metal"),
                ("urban_bem_indoor_state_update_fragment", ROOT / "woof/metal/shaders/urban_bem_indoor_state_update_fragment.metal"),
                ("urban_bem_room_control_fragment", ROOT / "woof/metal/shaders/urban_bem_room_control_fragment.metal"),
                ("urban_bem_shortwave_fragment", ROOT / "woof/metal/shaders/urban_bem_shortwave_fragment.metal"),
                ("urban_bem_longwave_fragment", ROOT / "woof/metal/shaders/urban_bem_longwave_fragment.metal"),
                ("urban_bem_surface_temp_fragment", ROOT / "woof/metal/shaders/urban_bem_surface_temp_fragment.metal"),
                ("urban_bem_rain_update_fragment", ROOT / "woof/metal/shaders/urban_bem_rain_update_fragment.metal"),
                ("urban_bem_wall_flux_fragment", ROOT / "woof/metal/shaders/urban_bem_wall_flux_fragment.metal"),
                ("urban_bem_flat_flux_fragment", ROOT / "woof/metal/shaders/urban_bem_flat_flux_fragment.metal"),
                ("urban_bem_buildings_tendency_fragment", ROOT / "woof/metal/shaders/urban_bem_buildings_tendency_fragment.metal"),
                ("urban_bem_urban_meso_fragment", ROOT / "woof/metal/shaders/urban_bem_urban_meso_fragment.metal"),
                ("urban_bem_interp_length_fragment", ROOT / "woof/metal/shaders/urban_bem_interp_length_fragment.metal"),
                ("urban_bem_final_output_fragment", ROOT / "woof/metal/shaders/urban_bem_final_output_fragment.metal"),
                ("urban_bem_consump_total_fragment", ROOT / "woof/metal/shaders/urban_bem_consump_total_fragment.metal"),
                ("urban_bem_interior_flux_fragment", ROOT / "woof/metal/shaders/urban_bem_interior_flux_fragment.metal"),
                ("urban_bem_interior_radiation_fragment", ROOT / "woof/metal/shaders/urban_bem_interior_radiation_fragment.metal"),
                ("urban_bem_param_fragment", ROOT / "woof/metal/shaders/urban_bem_param_fragment.metal"),
                ("urban_bem_icbep_xy_fragment", ROOT / "woof/metal/shaders/urban_bem_icbep_xy_fragment.metal"),
                ("urban_bem_building_level_scan_fragment", ROOT / "woof/metal/shaders/urban_bem_building_level_scan_fragment.metal"),
                ("urban_bem_atmosphere_gather_fragment", ROOT / "woof/metal/shaders/urban_bem_atmosphere_gather_fragment.metal"),
                ("urban_bem_atmosphere_interp_fragment", ROOT / "woof/metal/shaders/urban_bem_atmosphere_interp_fragment.metal"),
                ("urban_bep_bem_albedo_fragment", ROOT / "woof/metal/shaders/urban_bep_bem_albedo_fragment.metal"),
                ("urban_bep_bem_flat_roof_flux_fragment", ROOT / "woof/metal/shaders/urban_bep_bem_flat_roof_flux_fragment.metal"),
                ("urban_bep_bem_hi_profile_gather_fragment", ROOT / "woof/metal/shaders/urban_bep_bem_hi_profile_gather_fragment.metal"),
                ("urban_bep_bem_profile_select_fragment", ROOT / "woof/metal/shaders/urban_bep_bem_profile_select_fragment.metal"),
                ("urban_bep_bem_state_gather_fragment", ROOT / "woof/metal/shaders/urban_bep_bem_state_gather_fragment.metal"),
                ("urban_bep_bem_state_scatter_fragment", ROOT / "woof/metal/shaders/urban_bep_bem_state_scatter_fragment.metal"),
                ("urban_bem_material_solve_fragment", ROOT / "woof/metal/shaders/urban_bem_material_solve_fragment.metal"),
                ("urban_bem_flat_flux_pack_fragment", ROOT / "woof/metal/shaders/urban_bem_flat_flux_pack_fragment.metal"),
                ("urban_bep_bem_upward_rad_fragment", ROOT / "woof/metal/shaders/urban_bep_bem_upward_rad_fragment.metal"),
                ("urban_bep_bem_wall_fragment", ROOT / "woof/metal/shaders/urban_bep_bem_wall_fragment.metal"),
                ("urban_bep_bem_class_init_fragment", ROOT / "woof/metal/shaders/urban_bep_bem_class_init_fragment.metal"),
                ("urban_bep_bem_green_roof_fragment", ROOT / "woof/metal/shaders/urban_bep_bem_green_roof_fragment.metal"),
                ("urban_bem_material_solve_green_roof_fragment", ROOT / "woof/metal/shaders/urban_bem_material_solve_green_roof_fragment.metal"),
                ("urban_bep_bem_pv_solve_fragment", ROOT / "woof/metal/shaders/urban_bep_bem_pv_solve_fragment.metal"),
                ("urban_bep_bem_finalize_fragment", ROOT / "woof/metal/shaders/urban_bep_bem_finalize_fragment.metal"),
                ("urban_bep_bem_class_profile_pack_fragment", ROOT / "woof/metal/shaders/urban_bep_bem_class_profile_pack_fragment.metal")):
            air_path = temp / f"{name}.air"
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
                  "-c", str(source), "-o", str(air_path)])
            integration_fragment_airs.append(air_path)
        for source in sorted((ROOT / "woof/metal/shaders").glob("hexv84[12]_*.metal")):
            air_path = temp / f"{source.stem}.air"
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
                  "-c", str(source), "-o", str(air_path)])
            integration_fragment_airs.append(air_path)
        dudhia_shortwave_air = temp / "dudhia_shortwave.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(DUDHIA_SHORTWAVE_SHADER), "-o",
              str(dudhia_shortwave_air)])
        thompson_airs = []
        for name, source in (
                ("thompson_family", THOMPSON_FAMILY_SHADER),
                ("thompson_aerosol_state_family", THOMPSON_AEROSOL_STATE_SHADER),
                ("thompson_aerosol_refl_family", ROOT / "woof/metal/shaders/thompson_aerosol_refl_family.metal"),
                ("thompson_aerosol_wrf39_refl_family", ROOT / "woof/metal/shaders/thompson_aerosol_wrf39_refl_family.metal"),
                ("thompson_aerosol_exner", ROOT / "woof/metal/shaders/thompson_aerosol_exner.metal"),
                ("thompson_aerosol_graupel_init", ROOT / "woof/metal/shaders/thompson_aerosol_graupel_init.metal"),
                ("thompson_aerosol_cloud_tendency", ROOT / "woof/metal/shaders/thompson_aerosol_cloud_tendency.metal"),
                ("thompson_aerosol_sat_family", THOMPSON_AEROSOL_SAT_SHADER),
                ("thompson_column_family", ROOT / "woof/metal/shaders/thompson_column_family.metal"),
                ("tempo_family", ROOT / "woof/metal/shaders/tempo_family.metal"),
                ("tempo_family_deep", ROOT / "woof/metal/shaders/tempo_family_deep.metal"),
                ("thompson_column_adapter", ROOT / "woof/metal/shaders/thompson_column_adapter.metal"),
                ("topo_radiation_family", TOPO_RADIATION_SHADER),
                ("terrain_drag_family", TERRAIN_DRAG_SHADER),
                ("rrtmg_legacy_sw_family", ROOT / "woof/metal/shaders/rrtmg_legacy_sw_family.metal"),
                ("swint_fragment", ROOT / "woof/metal/shaders/swint_fragment.metal"),
                ("rrtmg_aer3_sw_optics", ROOT / "woof/metal/shaders/rrtmg_aer3_sw_optics.metal"),
                ("wrf_w_cfl_stat", ROOT / "woof/metal/shaders/wrf_w_cfl_stat.metal"),
                ("precipitation_buckets", ROOT / "woof/metal/shaders/precipitation_buckets.metal"),
                ("lightning_family", ROOT / "woof/metal/shaders/lightning_family.metal"),
                ("gsd_diagnostics_family", ROOT / "woof/metal/shaders/gsd_diagnostics_family.metal"),
                ("add_open_ring", ROOT / "woof/metal/shaders/add_open_ring.metal"),
                ("lbc_time_rational", ROOT / "woof/metal/shaders/lbc_time_rational.metal"),
                ("goddard_family", ROOT / "woof/metal/shaders/goddard_family.metal"),
                ("tiedtke_family", ROOT / "woof/metal/shaders/tiedtke_family.metal"),
                ("nsas_family", ROOT / "woof/metal/shaders/nsas_family.metal"),
                ("g3_family", ROOT / "woof/metal/shaders/g3_family.metal"),
                ("goddard_grid", ROOT / "woof/metal/shaders/goddard_grid.metal"),
                ("eclipse_family", ROOT / "woof/metal/shaders/eclipse_family.metal"),
                ("ocean_family", ROOT / "woof/metal/shaders/ocean_family.metal"),
                ("seaice_family", ROOT / "woof/metal/shaders/seaice_family.metal"),
                ("sfire_core", ROOT / "woof/metal/shaders/sfire_core.metal"),
                ("sfire_phys", ROOT / "woof/metal/shaders/sfire_phys.metal"),
                ("sfire_moisture", ROOT / "woof/metal/shaders/sfire_moisture.metal"),
                ("sfire_wind", ROOT / "woof/metal/shaders/sfire_wind.metal"),
                ("sfire_atm", ROOT / "woof/metal/shaders/sfire_atm.metal"),
                ("sfire_coupling", ROOT / "woof/metal/shaders/sfire_coupling.metal"),
                ("sfire_spotting", ROOT / "woof/metal/shaders/sfire_spotting.metal"),
                ("sfire_ideal", ROOT / "woof/metal/shaders/sfire_ideal.metal"),
                ("sfire_ideal_atmos", ROOT / "woof/metal/shaders/sfire_ideal_atmos.metal"),
                ("pwp_family", ROOT / "woof/metal/shaders/pwp_family.metal"),
                ("lin_family", ROOT / "woof/metal/shaders/lin_family.metal"),
                ("lin_refl_family", ROOT / "woof/metal/shaders/lin_refl_family.metal"),
                ("fer_hires_family", ROOT / "woof/metal/shaders/fer_hires_family.metal"),
                ("wsm5_family", ROOT / "woof/metal/shaders/wsm5_family.metal"),
                ("wsm3_family", ROOT / "woof/metal/shaders/wsm3_family.metal"),
                ("wsm6r_family", ROOT / "woof/metal/shaders/wsm6r_family.metal"),
                ("wsm7_family", ROOT / "woof/metal/shaders/wsm7_family.metal"),
                ("fast_sbm_family", ROOT / "woof/metal/shaders/fast_sbm_family.metal"),
                ("fast_sbm_pack", ROOT / "woof/metal/shaders/fast_sbm_pack.metal"),
                ("satmedmf_family", ROOT / "woof/metal/shaders/satmedmf_family.metal"),
                ("thompson_aerosol_warm_family", THOMPSON_AEROSOL_WARM_SHADER),
                ("thompson_aerosol_cold_family", THOMPSON_AEROSOL_COLD_SHADER),
                ("thompson_aerosol_sed_family", THOMPSON_AEROSOL_SED_SHADER),
                ("thompson_aerosol_wrf39_sed_family", THOMPSON_AEROSOL_WRF39_SED_SHADER),
                ("thompson_frozen_vapor_network", THOMPSON_FROZEN_VAPOR_NETWORK_SHADER),
                ("thompson_warm_frozen_source", THOMPSON_WARM_FROZEN_SOURCE_SHADER),
                ("thompson_cloud_sedimentation", THOMPSON_CLOUD_SEDIMENTATION_SHADER),
                ("thompson_ice_sedimentation", THOMPSON_ICE_SEDIMENTATION_SHADER),
                ("thompson_rain_evaporation", THOMPSON_RAIN_EVAPORATION_SHADER),
                ("thompson_rain_sedimentation", THOMPSON_RAIN_SEDIMENTATION_SHADER),
                ("thompson_snow_sedimentation", THOMPSON_SNOW_SEDIMENTATION_SHADER),
                ("thompson_graupel_sedimentation", THOMPSON_GRAUPEL_SEDIMENTATION_SHADER),
                ("thompson_final_phase_cleanup", THOMPSON_FINAL_PHASE_CLEANUP_SHADER),
                ("thompson_vapour_fill", ROOT / "woof/metal/shaders/thompson_vapour_fill.metal"),
                ("thompson_effective_radius", THOMPSON_EFFECTIVE_RADIUS_SHADER),
                ("thompson_reflectivity", THOMPSON_REFLECTIVITY_SHADER),
                ("spec_zone_ring", SPEC_ZONE_RING_SHADER),
                ("thompson_cold_start_family", THOMPSON_COLD_START_FAMILY_SHADER),
                ("thompson_cold_start_surface", THOMPSON_COLD_START_SURFACE_SHADER),
                ("thompson_cold_start_numbers", THOMPSON_COLD_START_NUMBERS_SHADER)):
            air_path = temp / f"{name}.air"
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
                  "-c", str(source), "-o", str(air_path)])
            thompson_airs.append(air_path)
        # Rayleigh AIR candidate; library admission remains a separate step.
        rayleigh_air = temp / "rayleigh_family.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/rayleigh_family.metal"),
              "-o", str(rayleigh_air)])
        bandwidth_air = temp / "bandwidth_family.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/bandwidth_family.metal"),
              "-o", str(bandwidth_air)])
        exact_glue_add_air = temp / "wrf_exact_glue_add.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_glue_add.metal"),
              "-o", str(exact_glue_add_air)])
        exact_tendency_glue_air = temp / "wrf_exact_tendency_glue.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_tendency_glue.metal"),
              "-o", str(exact_tendency_glue_air)])
        exact_upper_wind_air = temp / "wrf_exact_upper_wind_limiter.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_upper_wind_limiter.metal"),
              "-o", str(exact_upper_wind_air)])
        exact_w_damp_air = temp / "wrf_exact_w_damp.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_w_damp.metal"),
              "-o", str(exact_w_damp_air)])
        exact_coriolis_airs = []
        for mode in ("on", "off"):
            stem = f"wrf_exact_coriolis_{mode}"
            air = temp / (stem + ".air")
            exact_coriolis_airs.append(air)
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
                  "-c", str(ROOT / "woof/metal/shaders" / (stem + ".metal")),
                  "-o", str(air)])
        exact_total_theta_air = temp / "wrf_exact_total_theta.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_total_theta.metal"),
              "-o", str(exact_total_theta_air)])
        exact_smag_map_air = temp / "wrf_exact_smag_map_factor.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/wrf_exact_smag_map_factor.metal"),
              "-o", str(exact_smag_map_air)])
        urban_ucm_air = temp / "urban_ucm.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/urban_ucm_family.metal"),
              "-o", str(urban_ucm_air)])
        tendency_coupling_air = temp / "tendency_coupling.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(ROOT / "woof/metal/shaders/tendency_coupling.metal"),
              "-o", str(tendency_coupling_air)])
        _run([str(metallib), str(rk_air), str(acoustic_default_air),
              str(acoustic_calc_cq_air),
              str(uv_default_air), str(mu_default_air), str(wphi_default_air),
              str(advection_default_air),
              str(advection_u_default_air),
              str(advection_v_default_air),
              str(advection_w_default_air),
              str(advection_scalar_vorder_default_air),
              str(advection_u_vorder_default_air),
              str(advection_v_vorder_default_air),
              str(advection_w_vorder_default_air),
              str(diagnostics_default_air),
              str(stage_mass_fluxes_air), str(exact_stage_mass_fluxes_air), str(advection_prepare_air),
              str(advance_stage_air), str(slow_pgf_air), str(exact_slow_pgf_air),
              str(exact_acoustic_coefs_air), str(exact_acoustic_uv_air),
              str(exact_acoustic_mu_air), str(exact_acoustic_wphi_air),
              str(exact_calc_cq_air), str(exact_diagnostics_air), str(diffusion_air),
              str(slow_buoyancy_air), str(exact_slow_buoyancy_air),
              str(slow_geopotential_air), str(exact_slow_geopotential_air),
              str(acoustic_boundary_frame_air),
              str(surface_w_air),
              str(small_step_air), str(exact_small_step_air),
              str(close_periodic_alias_air),
              str(acoustic_emdiv_air), str(exact_acoustic_emdiv_air),
              str(openbc_radiative_air),
              str(rk_total_theta_air),
              str(coriolis_curvature_air),
              str(rayleigh_air),
              str(bandwidth_air),
              str(held_heating_air), str(moist_pd_air), str(exact_pd_on_air), str(exact_pd_off_air),
              str(moist_pd_source_fold_air),
              str(moist_scalar_update_air), str(exact_moist_scalar_update_air), str(exact_moist_tendency_air), str(exact_moist_capture_air),
              str(face_mass_air),
              str(urban_ucm_air),
              str(tendency_coupling_air),
              str(complex64_default_air),
              str(jacobi_eigh_fp64_default_air),
              str(spectral_scalar_air), str(sfclay_family_air),
              str(mynn_surface_family_air),
              str(physics_atmosphere_prepare_air),
              str(coupled_boundary_field_air),
              str(surface_humidity_air),
              str(lbc_time_air),
              str(lbc_state_air), str(exact_lbc_on_air), str(exact_lbc_off_air), str(exact_lbc_interior_air), str(exact_lbc_install_air), str(exact_lbc_final_on_air), str(exact_lbc_final_off_air), str(exact_nest_couple_on_air), str(exact_nest_couple_off_air), str(exact_nest_uncouple_on_air), str(exact_nest_uncouple_off_air), str(exact_nest_feedback_air), str(exact_nest_interp_air), str(exact_nest_smooth_air),
              str(lbc_flow_air), str(exact_lbc_flow_air),
              str(surface_diagnostics_seed_air),
              str(acoustic_sumflux_air),
              str(real_total_geopotential_air), str(ysu_column_air),
              str(ysu_480_column_air),
              str(noah_sfcdiags_air),
              str(health_family_air),
              *map(str, physics_family_airs),
              str(diff6_family_air),
              str(rrtmgp_prepare_air), str(rrtm_lw_family_air),
              str(rrtmg_legacy_lw_family_air),
              str(rrtmgp_mcica_air),
              str(hex_mass_flux_air),
              str(kf_w0avg_air),
              *map(str, physics_column_airs),
              str(global_spectral_diffusion_air),
              *map(str, integration_fragment_airs),
              str(dudhia_shortwave_air),
              *map(str, rrtmgp_family_airs),
              *map(str, thompson_airs),
              "-o", str(default_library_path)])
        _run([str(metallib), str(exact_rrtm_t8w_air), str(climate_exact_air), str(rk_air), str(acoustic_exact_air), str(sfclay_exact_air), str(exact_rrtm_mm5atm_air), str(exact_rrtm_setcoef_air), str(exact_rrtm_band0405_air), str(exact_rrtm_band03_air), str(exact_rrtm_band12_air), str(exact_rrtm_band06_air), str(exact_rrtm_band08_air), str(exact_rrtm_band07_air), str(exact_rrtm_band1011_air), str(exact_rrtm_band09_air), str(exact_rrtm_band1216_air), str(exact_rrtm_itr_air), str(exact_rrtm_rtrn_air),
              *map(str, exact_smag_airs), str(exact_diff6_air), str(exact_openbc_air),
              str(acoustic_calc_cq_air),
              str(uv_exact_air), str(mu_exact_air), str(wphi_exact_air),
              str(advection_exact_air),
              str(advection_u_exact_air),
              str(advection_v_exact_air),
              str(advection_w_exact_air),
              str(advection_scalar_vorder_exact_air),
              str(advection_u_vorder_exact_air),
              str(advection_v_vorder_exact_air),
              str(advection_w_vorder_exact_air),
              *(str(path) for path in exact_advection_air),
              str(diagnostics_exact_air),
              str(stage_mass_fluxes_air), str(exact_stage_mass_fluxes_air), str(advection_prepare_air),
              str(advance_stage_air), str(slow_pgf_air), str(exact_slow_pgf_air),
              str(exact_acoustic_coefs_air), str(exact_acoustic_uv_air),
              str(exact_acoustic_mu_air), str(exact_acoustic_wphi_air),
              str(exact_calc_cq_air), str(exact_diagnostics_air), str(diffusion_air),
              str(slow_buoyancy_air), str(exact_slow_buoyancy_air),
              str(slow_geopotential_air), str(exact_slow_geopotential_air),
              str(acoustic_boundary_frame_air),
              str(surface_w_air),
              str(exact_surface_w_air),
              str(small_step_air), str(exact_small_step_air),
              str(close_periodic_alias_air),
              str(acoustic_emdiv_air), str(exact_acoustic_emdiv_air),
              str(openbc_radiative_air),
              str(rk_total_theta_air),
              str(coriolis_curvature_air),
              str(exact_glue_add_air),
              str(exact_tendency_glue_air),
              str(exact_w_damp_air), str(exact_upper_wind_air),
              *(str(path) for path in exact_coriolis_airs),
              str(exact_total_theta_air),
              str(exact_smag_map_air),
              str(tendency_coupling_air),
              str(exact_moist_scalar_update_air), str(exact_moist_tendency_air),
              str(exact_moist_capture_air), str(exact_pd_on_air), str(exact_pd_off_air),
              str(exact_lbc_on_air), str(exact_lbc_off_air),
              str(exact_lbc_interior_air), str(exact_lbc_install_air),
              str(exact_lbc_final_on_air), str(exact_lbc_final_off_air),
              str(exact_lbc_flow_air),
              str(exact_nest_couple_on_air), str(exact_nest_couple_off_air),
              str(exact_nest_roundtrip_air),
              str(exact_nest_uncouple_on_air), str(exact_nest_uncouple_off_air),
              str(exact_nest_feedback_air),
              str(exact_nest_interp_air),
              str(exact_nest_smooth_air),
              str(exact_held_heating_air), str(exact_dudhia_air), str(exact_mynn_radiation_air),
              str(nest_family_air), str(nest_interp_fragment_air),
              str(nest_stream_transfer_fragment_air), str(ieva_family_air),
              str(radar_tten_family_air),
              str(ieva_solve_family_air),
              str(face_mass_air),
              str(complex64_exact_air),
              str(jacobi_eigh_fp64_exact_air),
              str(sfclay_family_air), str(mynn_surface_family_air),
              str(physics_atmosphere_prepare_air), str(exact_physics_atmosphere_air),
              str(temp / "ruc_family.air"),
              str(temp / "ruc_sfctmp_family.air"),
              str(temp / "ruc_driver_family.air"),
              str(coupled_boundary_field_air),
              str(strict_coupled_boundary_field_air),
              str(surface_humidity_air),
              str(lbc_time_air),
              str(surface_diagnostics_seed_air),
              str(uh_diag_family_air), str(exact_uh_diag_air), str(exact_noah_seam_air), str(exact_noah_land_air), str(exact_cldfra1_air), *map(str, exact_goddard_airs), str(exact_noah_init_air), str(exact_noah_seaice_air),
              str(acoustic_sumflux_air),
              str(real_total_geopotential_air),
              str(rrtmgp_prepare_air), str(rrtm_lw_family_air),
              str(rrtmg_legacy_lw_family_air),
              str(rrtmgp_mcica_air),
              str(hex_mass_flux_air),
              str(kf_w0avg_air),
              str(noahmp_glacier_air),
              str(global_spectral_diffusion_air),
              *map(str, integration_fragment_airs),
              str(dudhia_shortwave_air),
              *map(str, rrtmgp_family_airs),
              *map(str, thompson_airs),
              "-o", str(exact_library_path)])
        _run(["xcrun", "--sdk", "macosx", "clang++", "-arch", "arm64",
              "-std=c++17", "-O3",
              "-fobjc-arc", "-dynamiclib", "-mmacosx-version-min=14.0",
              "-framework", "Foundation", "-framework", "Metal",
              "-framework", "MetalPerformanceShadersGraph",
              "-I", str(NATIVE_DIR), str(NATIVE_DIR / "woof_metal.mm"),
              "-o", str(runtime_path)])

    _run(["install_name_tool", "-id", "@rpath/libwoof_metal.dylib",
          str(runtime_path)])

    _build_commit, implementation_sha256, implementation_file_count = \
        _implementation_provenance()
    files = {}
    for path in (runtime_path, default_library_path, exact_library_path):
        files[path.name] = {
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
    source_provenance = metal_source_provenance(ROOT)
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "abi_version": abi_version,
        "platform": "darwin-arm64",
        "minimum_os": "14.0",
        "source_provenance": source_provenance,
        "metal_inputs_sha256": source_provenance["sha256"],
        "implementation_sha256": implementation_sha256,
        "implementation_file_count": implementation_file_count,
        "files": files,
    }
    (output / "METAL_BUNDLE.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="directory for the dylib, metallib, and manifest")
    args = parser.parse_args()
    manifest = build(args.output.resolve())
    # The full manifest is METAL_BUNDLE.json; report only where it went.
    print(json.dumps({"output": str(args.output), "keys": sorted(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
