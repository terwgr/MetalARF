#include <metal_stdlib>
#include "urban_bem_final_output_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

constant uint BEM_FINAL_NBUI = 15u;
constant uint BEM_FINAL_NZUM = 18u;

static inline float ubfo_add(float a,float b){return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubfo_mul(float a,float b){return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubfo_div(float a,float b){return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b)));}

// Final PBL column aggregation and domain scatter. The preceding dedicated
// BEP+BEM finalizer owns PV and green-roof diagnostics.
kernel void urban_bem_final_output_fragment(
    constant UrbanBemFinalOutputFragmentResources &r [[buffer(0)]],
    constant UrbanBemFinalOutputFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol) || r.active_cols[col] == 0) return;

    const uint ncol = uint(p.ncol);
    const ulong c = ulong(col);
    float sf_ac = 0.0f;
    float lf_ac = 0.0f;
    float cm_ac = 0.0f;
    float sf_vent = 0.0f;
    float lf_vent = 0.0f;
    float meso = 0.0f;
    for (uint d = 0; d < 2u; ++d) {
        const float bs0 = r.bs[c * 2ul + d];
        const float ws0 = r.ws[c * 2ul + d];
        const float bs1 = r.bs[c * 2ul + (1u - d)];
        const float ws1 = r.ws[c * 2ul + (1u - d)];
        const float numerator = ubfo_mul(0.25f, r.fraction[col]);
        const float denominator = ubfo_mul(ubfo_add(bs0, ws0), bs1);
        meso = ubfo_add(meso, ubfo_div(numerator, denominator));
    }
    const int utype = r.utype[col] - 1;
    if (utype < 0 || utype >= 11) return;
    const float meso_ac = ubfo_mul(
        ubfo_mul(meso, r.class_float[1107ul + uint(utype)]),
        r.class_float[1118ul + uint(utype)]);
    for (uint b = 0; b < BEM_FINAL_NBUI; ++b) {
        const ulong base = (c * BEM_FINAL_NBUI + b) * BEM_FINAL_NZUM;
        const int top = r.building_nlev[c * BEM_FINAL_NBUI + b];
        if (top <= 0) continue;
        const float area = ubfo_mul(
            meso_ac, r.d_urb[c * BEM_FINAL_NZUM + b]);
        for (int z = 0; z < top && z < int(BEM_FINAL_NZUM); ++z) {
            const ulong i = base + uint(z);
            sf_ac = ubfo_add(sf_ac, ubfo_mul(area, r.sflev[i]));
            lf_ac = ubfo_add(lf_ac, ubfo_mul(area, r.lflev[i]));
            cm_ac = ubfo_add(cm_ac, ubfo_mul(area, r.consum[i]));
            sf_vent = ubfo_add(sf_vent, ubfo_mul(area, r.sfvlev[i]));
            lf_vent = ubfo_add(lf_vent, ubfo_mul(area, r.lfvlev[i]));
        }
    }
    r.sf_ac[col] = sf_ac;
    r.lf_ac[col] = lf_ac;
    r.cm_ac[col] = cm_ac;
    r.sf_vent[col] = sf_vent;
    r.lf_vent[col] = lf_vent;
    r.rl_up[col] = r.rl_up_in[col];
    r.rs_abs[col] = r.rs_abs_in[col];
    r.emiss[col] = r.emiss_in[col];
    r.grdflx[col] = r.grdflx_in[col];

    for (uint z = 0; z < uint(p.nz); ++z) {
        const ulong out = ulong(z) * ncol + c;
        const ulong in = c * uint(p.nz) + z;
        r.out_vl[out] = r.vl[in];
        r.out_a_u[out] = r.a_u[in]; r.out_a_v[out] = r.a_v[in];
        r.out_a_t[out] = r.a_t[in]; r.out_a_e[out] = r.a_e[in];
        r.out_b_u[out] = r.b_u[in]; r.out_b_v[out] = r.b_v[in];
        r.out_b_t[out] = r.b_t[in]; r.out_b_e[out] = r.b_e[in];
        r.out_b_q[out] = r.b_q[in]; r.out_dlg[out] = r.dlg[in];
        r.out_dl_u[out] = r.dl_u[in];
        r.out_sf[out] = r.sf[in];
    }
    r.out_sf[ulong(p.nz) * ncol + c] = r.sf[c * (uint(p.nz) + 1u) + uint(p.nz)];
}
