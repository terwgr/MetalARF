#include <metal_stdlib>
#include "ruc_sfclay_seaice_bindings.metalh"

using namespace metal;

kernel void ruc_sfclay_prepare(
    constant RucSfclayPrepareResources& r [[buffer(0)]],
    constant RucSfclayPrepareParams& p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    float fraction = r.xice[i];
    bool component = fraction >= p.xice_threshold && fraction <= 1.0f;
    r.local_tsk[i] = component ? r.tsk_save[i] : r.tsk[i];
}

kernel void ruc_sfclay_scatter(
    constant RucSfclayScatterResources& r [[buffer(0)]],
    constant RucSfclayScatterParams& p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    float fraction = r.xice[i];
    if (!(fraction >= p.xice_threshold && fraction <= 1.0f)) return;
    uint n = uint(p.n);
    r.znt_sea[i] = r.raw[0u*n+i];
    r.ust_sea[i] = r.raw[1u*n+i];
    r.mol_sea[i] = r.raw[3u*n+i];
    r.zol_sea[i] = r.raw[7u*n+i];
    r.flhc_sea[i] = r.raw[22u*n+i];
    r.flqc_sea[i] = r.raw[23u*n+i];
    r.cpm_sea[i] = r.raw[29u*n+i];
    r.cqs2_sea[i] = r.raw[21u*n+i];
    r.chs2_sea[i] = r.raw[20u*n+i];
    r.chs_sea[i] = r.raw[19u*n+i];
    r.qsfc_sea[i] = r.raw[6u*n+i];
    r.qgh_sea[i] = r.raw[24u*n+i];
    r.hfx_sea[i] = r.raw[4u*n+i];
    r.qfx_sea[i] = r.raw[5u*n+i];
    r.lh_sea[i] = r.raw[13u*n+i];
}

// MYNN_SEAICE_WRAPPER inputs for both SFCLAY_mynn calls
// (module_surface_driver.F:5402-5506, get_local_ice_tsk :7158-7204).
kernel void mynn_seaice_prepare(
    constant MynnSeaicePrepareResources& r [[buffer(0)]],
    constant MynnSeaicePrepareParams& p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    const float fraction = r.xice[i];
    const bool component = fraction >= p.xice_threshold && fraction <= 1.0f;
    float sea = r.tsk_sea[i];
    float local = r.tsk[i];
    if (component) {
        float sst = fmax(sea, 271.4f);
        if (p.itimestep <= 3 && sst > 273.0f) {
            if (fraction >= 0.6f) sst = 271.4f;
            else if (fraction >= 0.4f) sst = 273.0f;
            else if (fraction >= 0.2f && sst > 275.0f) sst = 275.0f;
            else if (sst > 278.0f) sst = 278.0f;
        }
        sea = sst;
        const float diagnosed =
            (r.tsk[i] - (1.0f - fraction) * sea) / fraction;
        local = fmax(diagnosed, 221.4f);
        if (fraction < 0.2f && r.tsk[i] < 253.15f) local = 253.15f;
        if (fraction < 0.1f && r.tsk[i] < 263.15f) local = 263.15f;
        sea = fmax(sea, 271.4f);
        r.znt_sea[i] = 1.0e-4f;
    }
    r.tsk_sea[i] = sea;
    r.local_tsk[i] = local;
    r.sea_tsk[i] = component ? sea : local;
    r.sea_mavail[i] = component ? 1.0f : r.mavail[i];
    r.sea_xland[i] = component ? 2.0f : r.xland[i];
}

// :5508-5554 grid-cell blend; select=1 takes the open-water value (REGIME).
kernel void mynn_seaice_blend(
    constant MynnSeaiceBlendResources& r [[buffer(0)]],
    constant MynnSeaiceBlendParams& p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    const float fraction = r.xice[i];
    if (!(fraction >= p.xice_threshold && fraction <= 1.0f)) return;
    const float ice_part = r.value[i] * fraction;
    const float sea_part = (1.0f - fraction) * r.sea[i];
    r.value[i] = p.select ? r.sea[i] : ice_part + sea_part;
}
