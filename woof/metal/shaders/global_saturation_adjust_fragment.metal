// Four-step FP64 moist-adiabatic saturation adjustment for Global columns.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
#include "global_saturation_adjust_fragment_bindings.metalh"

using namespace metal;

static inline ulong global_sat_max(ulong a, ulong b) {
    return soft64_greater(a,b) ? a : b;
}

kernel void global_saturation_adjust_fragment(
    constant GlobalSaturationAdjustFragmentResources &r [[buffer(0)]],
    constant GlobalSaturationAdjustFragmentParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    if(column>=uint(p.ncolumn))return;
    const ulong one=0x3ff0000000000000ul;
    const ulong zero=0ul;
    const ulong cp_ratio=soft64_div(r.latent_heat[0],r.dry_air_cp[0]);
    ulong total_positive=zero;
    for(int level=0;level<p.nlevel;++level) {
        const uint index=uint(level)*uint(p.ncolumn)+column;
        ulong temperature=soft64_mul(r.theta[index],r.exner[index]);
        ulong qv=r.qv[index], qc=r.qc[index];
        ulong total_phase=zero;
        for(int iteration=0;iteration<4;++iteration) {
            ulong t=temperature;
            const ulong tmin=0x4062c00000000000ul; // 150
            const ulong tmax=0x4075e00000000000ul; // 350
            if(soft64_less(t,tmin))t=tmin;
            if(soft64_greater(t,tmax))t=tmax;
            ulong exponent=soft64_div(
                soft64_mul(0x4031ab851eb851ecul,
                    soft64_sub(t,0x4071126666666666ul)),
                global_sat_max(soft64_sub(t,0x403da66666666666ul),one));
            ulong vapor=soft64_mul(0x408319999999999aul,soft64_plm_exp(exponent));
            const ulong vapor_cap=soft64_mul(0x3fee666666666666ul,r.pressure[index]);
            if(soft64_greater(vapor,vapor_cap))vapor=vapor_cap;
            const ulong dry=global_sat_max(soft64_sub(r.pressure[index],vapor),one);
            const ulong qsat=soft64_div(soft64_mul(r.epsilon[0],vapor),dry);
            const ulong dqsat=soft64_div(
                soft64_mul(soft64_mul(soft64_mul(qsat,soft64_div(r.pressure[index],dry)),
                    0x4031ab851eb851ecul),0x406e700000000000ul),
                soft64_mul(global_sat_max(soft64_sub(t,0x403da66666666666ul),one),
                           global_sat_max(soft64_sub(t,0x403da66666666666ul),one)));
            const ulong gamma=soft64_mul(cp_ratio,dqsat);
            ulong phase=soft64_div(soft64_sub(qv,qsat),soft64_add(one,gamma));
            const ulong lower=soft64_negate_bits(qc);
            if(soft64_less(phase,lower))phase=lower;
            qv=soft64_sub(qv,phase);
            qc=soft64_add(qc,phase);
            temperature=soft64_add(temperature,soft64_mul(cp_ratio,phase));
            total_phase=soft64_add(total_phase,phase);
        }
        r.theta_out[index]=soft64_div(temperature,r.exner[index]);
        r.qv_out[index]=qv;
        r.qc_out[index]=qc;
        if(soft64_greater(total_phase,zero))total_positive=soft64_add(total_positive,total_phase);
    }
    r.condensed[column]=total_positive;
}
