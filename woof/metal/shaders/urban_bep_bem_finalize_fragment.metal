#include <metal_stdlib>
#include "urban_bep_bem_finalize_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

constant uint BEM_FIN_NBUI = 15u;
constant uint BEM_FIN_NZUM = 18u;
constant uint BEM_FIN_NDM = 2u;
constant uint BEM_FIN_NGR = 10u;
constant uint BEM_FIN_NURB = 11u;
constant uint BEM_FIN_BLDAC = 1107u;
constant uint BEM_FIN_COOLED = 1118u;

static inline float ubf_add(float a, float b) {
    return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubf_sub(float a, float b) {
    return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubf_mul(float a, float b) {
    return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubf_div(float a, float b) {
    return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)));
}

// module_sf_bep_bem.F:1081-1108. One lane finalizes one active column after
// its building solve. PV energy has already been calculated by the preceding
// per-building bem_hsfluxpv stage; green diagnostics retain source order.
kernel void urban_bep_bem_finalize_fragment(
    constant UrbanBepBemFinalizeFragmentResources &r [[buffer(0)]],
    constant UrbanBepBemFinalizeFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol) || r.active_cols[col] == 0) return;

    const ulong c = ulong(col);
    const int utype = r.utype[col] - 1;
    if (utype < 0 || utype >= int(BEM_FIN_NURB)) return;

    float meso = 0.0f;
    for (uint d = 0; d < BEM_FIN_NDM; ++d) {
        const uint other = 1u - d;
        const float numerator = ubf_mul(0.25f, r.fraction[col]);
        const float denominator = ubf_mul(
            ubf_add(r.bs[c * BEM_FIN_NDM + d], r.ws[c * BEM_FIN_NDM + d]),
            r.bs[c * BEM_FIN_NDM + other]);
        meso = ubf_add(meso, ubf_div(numerator, denominator));
    }
    const float meso_ac = ubf_mul(
        ubf_mul(meso, r.class_float[BEM_FIN_BLDAC + uint(utype)]),
        r.class_float[BEM_FIN_COOLED + uint(utype)]);
    float ep_pv = 0.0f;
    for (uint b = 0; b < BEM_FIN_NBUI; ++b) {
        const int top = r.building_nlev[c * BEM_FIN_NBUI + b];
        if (top <= 0 || top > int(BEM_FIN_NZUM)) continue;
        ep_pv = ubf_add(ep_pv, ubf_mul(
            ubf_mul(meso_ac, r.d_urb[c * BEM_FIN_NZUM + b]),
            r.pv_energy[c * BEM_FIN_NBUI + b]));
    }
    r.ep_pv[col] = ep_pv;

    float drain_green = 0.0f;
    float qgr = 0.0f;
    float tgr = 0.0f;
    if (p.green_roof == 1) {
        const float roof_fraction = ubf_div(
            ubf_mul(r.fraction[col], r.bs[c * BEM_FIN_NDM]),
            ubf_add(r.bs[c * BEM_FIN_NDM], r.ws[c * BEM_FIN_NDM]));
        for (uint d = 0; d < BEM_FIN_NDM; ++d) {
            for (uint z = 1; z < BEM_FIN_NZUM - 1u; ++z) {
                const ulong drain_i = (c * BEM_FIN_NZUM + z) * BEM_FIN_NDM + d;
                const float drain_area = ubf_mul(
                    r.d_urb[c * BEM_FIN_NZUM + z - 1ul], roof_fraction);
                drain_green = ubf_add(drain_green,
                    ubf_mul(ubf_mul(drain_area, r.drain[drain_i]), 1000.0f));
                for (uint g = 0; g < BEM_FIN_NGR; ++g) {
                    const ulong plant_i =
                        ((c * BEM_FIN_NGR + g) * BEM_FIN_NZUM + z) * BEM_FIN_NDM + d;
                    qgr = ubf_add(qgr, ubf_div(ubf_div(
                        ubf_div(r.qr[plant_i], 2.0f), 16.0f), 10.0f));
                    tgr = ubf_add(tgr, ubf_div(ubf_div(
                        ubf_div(r.trv[plant_i], 2.0f), 16.0f), 10.0f));
                }
            }
        }
    }
    r.drain_green[col] = drain_green;
    r.qgr[col] = qgr;
    r.tgr[col] = tgr;
}
