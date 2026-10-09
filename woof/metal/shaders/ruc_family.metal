#include <metal_stdlib>
#include "ruc_surface_parameters_bindings.metalh"

using namespace metal;

#pragma clang fp contract(off)

// WRF v4.6.1 phys/module_sf_ruclsm.F:soilvegin, dominant-category arm.
// Resource and operation order follow woof/core/kernels/ruc.cu. The caller
// validates category bounds before dispatch; table indices are therefore
// valid for every live column.
kernel void ruc_surface_parameters(
    constant RucSurfaceParametersResources& r [[buffer(0)]],
    constant RucSurfaceParametersParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    const int vegetation_index = r.ivgtyp[tid] - 1;
    const int soil_index = r.isltyp[tid] - 1;
    const int forest_class = r.ifortbl[vegetation_index];
    r.iforest_out[tid] = forest_class;

    const float green_range = r.shdmax[tid] - r.shdmin[tid];
    float factor;
    if (green_range < 1.0f) {
        factor = 1.0f;
    } else {
        const float numerator = r.vegfrac[tid] - r.shdmin[tid];
        const float denominator = max(1.0f, green_range);
        const float ratio = numerator / denominator;
        factor = 1.0f - max(0.0f, min(1.0f, ratio));
    }

    const float table_lai = r.laitbl[vegetation_index];
    const float scaled_lai = 0.8f * table_lai;
    float delta_lai = 0.0f;
    if (forest_class == 1) {
        delta_lai = min(0.2f, scaled_lai);
    } else if (forest_class == 2 || forest_class == 7) {
        delta_lai = min(0.5f, scaled_lai);
    } else if (forest_class == 3) {
        delta_lai = min(0.45f, scaled_lai);
    } else if (forest_class == 4) {
        delta_lai = min(0.75f, scaled_lai);
    } else if (forest_class == 5) {
        delta_lai = min(0.86f, scaled_lai);
    }

    float roughness = r.znt_in[tid];
    float leaf_area = r.lai_in[tid];
    if (r.ivgtyp[tid] == p.iswater) {
        if (!p.rdlai2d) leaf_area = table_lai;
    } else {
        if (!p.rdlai2d) leaf_area = table_lai - delta_lai * factor;
        roughness = r.z0tbl[vegetation_index];
        if (forest_class == 7) roughness = roughness - 0.125f * factor;
    }
    r.emiss_out[tid] = r.lemitbl[vegetation_index];
    r.pc_out[tid] = r.pctbl[vegetation_index];
    r.znt_out[tid] = roughness;
    r.lai_out[tid] = leaf_area;

    r.qwrtz_out[tid] = 0.0f;
    r.rhocs_out[tid] = 0.0f;
    r.bclh_out[tid] = 0.0f;
    r.dqm_out[tid] = 0.0f;
    r.ksat_out[tid] = 0.0f;
    r.psis_out[tid] = 0.0f;
    r.qmin_out[tid] = 0.0f;
    r.ref_out[tid] = 0.0f;
    r.wilt_out[tid] = 0.0f;
    if (r.isltyp[tid] == 14) return;

    r.qwrtz_out[tid] = r.qtz[soil_index];
    r.rhocs_out[tid] = r.hc[soil_index] * 1.0e6f;
    r.bclh_out[tid] = r.bb[soil_index];
    r.dqm_out[tid] = r.maxsmc[soil_index] - r.drysmc[soil_index];
    r.ksat_out[tid] = r.satdk[soil_index];
    r.psis_out[tid] = -r.satpsi[soil_index];
    r.qmin_out[tid] = r.drysmc[soil_index];
    r.ref_out[tid] = r.refsmc[soil_index];
    r.wilt_out[tid] = r.wltsmc[soil_index];
}
