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
ACOUSTIC_SHADER = ROOT / "woof" / "metal" / "shaders" / "acoustic_coefficients.metal"
ACOUSTIC_CALC_CQ_SHADER = ROOT / "woof" / "metal" / "shaders" / "acoustic_calc_cq.metal"
ACOUSTIC_UV_SHADER = ROOT / "woof" / "metal" / "shaders" / "acoustic_advance_uv.metal"
ACOUSTIC_MU_TH_SHADER = ROOT / "woof" / "metal" / "shaders" / "acoustic_advance_mu_th.metal"
ACOUSTIC_W_PHI_SHADER = ROOT / "woof" / "metal" / "shaders" / "acoustic_advance_w_phi.metal"
ACOUSTIC_BOUNDARY_FRAME_SHADER = ROOT / "woof" / "metal" / "shaders" / "acoustic_boundary_frame.metal"
ACOUSTIC_EMDIV_SHADER = ROOT / "woof" / "metal" / "shaders" / "acoustic_emdiv.metal"
ADVECTION_SCALAR_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_flux_div_scalar.metal"
ADVECTION_U_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_flux_div_u.metal"
ADVECTION_V_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_flux_div_v.metal"
ADVECTION_W_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_flux_div_w.metal"
ADVECTION_SCALAR_VORDER_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_flux_div_scalar_vorder.metal"
ADVECTION_U_VORDER_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_flux_div_u_vorder.metal"
ADVECTION_V_VORDER_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_flux_div_v_vorder.metal"
ADVECTION_W_VORDER_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_flux_div_w_vorder.metal"
DIAGNOSTICS_CALC_SHADER = ROOT / "woof" / "metal" / "shaders" / "diagnostics_calc_p_alpha.metal"
STAGE_MASS_FLUXES_SHADER = ROOT / "woof" / "metal" / "shaders" / "stage_mass_fluxes.metal"
ADVECTION_PREPARE_SHADER = ROOT / "woof" / "metal" / "shaders" / "advection_prepare.metal"
DYCORE_ADVANCE_STAGE_SHADER = ROOT / "woof" / "metal" / "shaders" / "dycore_advance_stage.metal"
SLOW_PGF_SHADER = ROOT / "woof" / "metal" / "shaders" / "slow_pgf.metal"
DIFFUSION_SHADER = ROOT / "woof" / "metal" / "shaders" / "diffusion.metal"
SLOW_BUOYANCY_SHADER = ROOT / "woof" / "metal" / "shaders" / "slow_buoyancy.metal"
SLOW_GEOPOTENTIAL_SHADER = ROOT / "woof" / "metal" / "shaders" / "slow_geopotential.metal"
SURFACE_W_SHADER = ROOT / "woof" / "metal" / "shaders" / "surface_w.metal"
SMALL_STEP_SHADER = ROOT / "woof" / "metal" / "shaders" / "small_step.metal"
CLOSE_PERIODIC_ALIAS_SHADER = ROOT / "woof" / "metal" / "shaders" / "close_periodic_alias.metal"
OPENBC_RADIATIVE_SHADER = ROOT / "woof" / "metal" / "shaders" / "openbc_radiative.metal"
RK_TOTAL_THETA_SHADER = ROOT / "woof" / "metal" / "shaders" / "rk_total_theta.metal"
CORIOLIS_CURVATURE_SHADER = ROOT / "woof" / "metal" / "shaders" / "coriolis_curvature.metal"
MOIST_PD_SHADER = ROOT / "woof" / "metal" / "shaders" / "moist_pd.metal"
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
LBC_STATE_SHADER = ROOT / "woof" / "metal" / "shaders" / "lbc_state.metal"
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
SASE_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "sase_family.metal"
DUDHIA_SHORTWAVE_SHADER = ROOT / "woof" / "metal" / "shaders" / "dudhia_shortwave.metal"
KESSLER_FAMILY_SHADER = ROOT / "woof" / "metal" / "shaders" / "kessler_family.metal"
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
        advection_prepare_air = temp / "advection_prepare.air"
        advance_stage_air = temp / "dycore_advance_stage.air"
        slow_pgf_air = temp / "slow_pgf.air"
        diffusion_air = temp / "diffusion.air"
        slow_buoyancy_air = temp / "slow_buoyancy.air"
        slow_geopotential_air = temp / "slow_geopotential.air"
        acoustic_boundary_frame_air = temp / "acoustic_boundary_frame.air"
        surface_w_air = temp / "surface_w.air"
        small_step_air = temp / "small_step.air"
        close_periodic_alias_air = temp / "close_periodic_alias.air"
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
              "-c", str(SLOW_GEOPOTENTIAL_SHADER), "-o",
              str(slow_geopotential_air)])
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
        # Ordinary Noah and the WRF Smagorinsky family use default-policy
        # arithmetic only. Exact-policy callers remain refused by their
        # Python submitters.
        physics_family_airs = []
        for name, source in (("noah_family", NOAH_FAMILY_SHADER),
                             ("noah_forcing_family", NOAH_FORCING_FAMILY_SHADER),
                             ("smag2d_family", SMAG2D_FAMILY_SHADER),
                             ("lake_family", LAKE_FAMILY_SHADER)):
            air_path = temp / f"{name}.air"
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
                  "-c", str(source), "-o", str(air_path)])
            physics_family_airs.append(air_path)
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
        # The UH column integral and its running-max fold retain the CUDA
        # exact expression order and are shipped only in the exact library.
        uh_diag_family_air = temp / "uh_diag_family.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-DGPUWM_WRF_EXACT=1", "-c", str(UH_DIAG_FAMILY_SHADER),
              "-o", str(uh_diag_family_air)])
        physics_atmosphere_prepare_air = temp / "physics_atmosphere_prepare.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(PHYSICS_ATMOSPHERE_PREPARE_SHADER), "-o",
              str(physics_atmosphere_prepare_air)])
        coupled_boundary_field_air = temp / "coupled_boundary_field.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(COUPLED_BOUNDARY_FIELD_SHADER), "-o",
              str(coupled_boundary_field_air)])
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
        lbc_flow_air = temp / "lbc_flow.air"
        _run([str(metal), f"-fmodules-cache-path={module_cache}",
              "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
              "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
              "-c", str(LBC_FLOW_SHADER), "-o", str(lbc_flow_air)])
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
                ("wsm6_family", WSM6_FAMILY_SHADER),
                ("wdm6_family", WDM6_FAMILY_SHADER)):
            air_path = temp / f"{name}.air"
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
                  "-c", str(source), "-o", str(air_path)])
            rrtmgp_family_airs.append(air_path)
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
                ("nssl2_fused_gs", NSSL2_FUSED_GS_SHADER),
                ("nssl2_nucond", NSSL2_NUCOND_SHADER),
                ("nssl2_driver_transport", NSSL2_DRIVER_TRANSPORT_SHADER),
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
                ("urban_ucm_noahmp_after_lsm_fragment", ROOT / "woof/metal/shaders/urban_ucm_noahmp_after_lsm_fragment.metal"),
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
                ("urban_bep_bem_class_profile_pack_fragment", ROOT / "woof/metal/shaders/urban_bep_bem_class_profile_pack_fragment.metal")):
            air_path = temp / f"{name}.air"
            _run([str(metal), f"-fmodules-cache-path={module_cache}",
                  "-std=macos-metal2.4", "-target", "air64-apple-macos14.0",
                  "-I", str(GENERATED_DIR), "-fmetal-math-mode=safe",
                  "-fmetal-math-fp32-functions=precise", "-ffp-contract=off",
                  "-c", str(source), "-o", str(air_path)])
            integration_fragment_airs.append(air_path)
        for source in sorted((ROOT / "woof/metal/shaders").glob("hexv841_*.metal")):
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
                ("thompson_aerosol_sat_family", THOMPSON_AEROSOL_SAT_SHADER),
                ("topo_radiation_family", TOPO_RADIATION_SHADER),
                ("terrain_drag_family", TERRAIN_DRAG_SHADER),
                ("rrtmg_legacy_sw_family", ROOT / "woof/metal/shaders/rrtmg_legacy_sw_family.metal"),
                ("swint_fragment", ROOT / "woof/metal/shaders/swint_fragment.metal"),
                ("rrtmg_aer3_sw_optics", ROOT / "woof/metal/shaders/rrtmg_aer3_sw_optics.metal"),
                ("wrf_w_cfl_stat", ROOT / "woof/metal/shaders/wrf_w_cfl_stat.metal"),
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
              str(stage_mass_fluxes_air), str(advection_prepare_air),
              str(advance_stage_air), str(slow_pgf_air), str(diffusion_air),
              str(slow_buoyancy_air),
              str(slow_geopotential_air),
              str(acoustic_boundary_frame_air),
              str(surface_w_air),
              str(small_step_air),
              str(close_periodic_alias_air),
              str(acoustic_emdiv_air),
              str(openbc_radiative_air),
              str(rk_total_theta_air),
              str(coriolis_curvature_air),
              str(rayleigh_air),
              str(bandwidth_air),
              str(held_heating_air), str(moist_pd_air),
              str(moist_pd_source_fold_air),
              str(moist_scalar_update_air),
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
              str(lbc_state_air),
              str(lbc_flow_air),
              str(surface_diagnostics_seed_air),
              str(acoustic_sumflux_air),
              str(real_total_geopotential_air), str(ysu_column_air),
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
        _run([str(metallib), str(rk_air), str(acoustic_exact_air),
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
              str(diagnostics_exact_air),
              str(stage_mass_fluxes_air), str(advection_prepare_air),
              str(advance_stage_air), str(slow_pgf_air), str(diffusion_air),
              str(slow_buoyancy_air),
              str(slow_geopotential_air),
              str(acoustic_boundary_frame_air),
              str(surface_w_air),
              str(small_step_air),
              str(close_periodic_alias_air),
              str(acoustic_emdiv_air),
              str(openbc_radiative_air),
              str(rk_total_theta_air),
              str(coriolis_curvature_air),
              str(tendency_coupling_air),
              str(nest_family_air), str(nest_interp_fragment_air),
              str(nest_stream_transfer_fragment_air), str(ieva_family_air),
              str(radar_tten_family_air),
              str(ieva_solve_family_air),
              str(face_mass_air),
              str(complex64_exact_air),
              str(jacobi_eigh_fp64_exact_air),
              str(sfclay_family_air), str(mynn_surface_family_air),
              str(physics_atmosphere_prepare_air),
              str(temp / "ruc_family.air"),
              str(temp / "ruc_sfctmp_family.air"),
              str(temp / "ruc_driver_family.air"),
              str(coupled_boundary_field_air),
              str(surface_humidity_air),
              str(lbc_time_air),
              str(surface_diagnostics_seed_air),
              str(uh_diag_family_air),
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
    print(json.dumps(build(args.output.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
