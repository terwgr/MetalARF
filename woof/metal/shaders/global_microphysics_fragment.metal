// FP64 pointwise Global reference microphysics conversion chain.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
#include "global_microphysics_fragment_bindings.metalh"

using namespace metal;
static inline ulong gmmax(ulong a,ulong b){return soft64_greater(a,b)?a:b;}
static inline ulong gmmin(ulong a,ulong b){return soft64_less(a,b)?a:b;}
static inline ulong gmclip(ulong x,ulong a,ulong b){return gmmin(gmmax(x,a),b);}
static inline ulong gmqsat(ulong t,ulong p,ulong epsilon){
    t=gmclip(t,0x4062c00000000000ul,0x4075e00000000000ul);
    ulong denom=gmmax(soft64_sub(t,0x403da66666666666ul),0x3ff0000000000000ul);
    ulong e=soft64_mul(0x408319999999999aul,soft64_plm_exp(soft64_div(
        soft64_mul(0x4031ab851eb851ecul,soft64_sub(t,0x4071126666666666ul)),denom)));
    e=gmmin(e,soft64_mul(0x3fee666666666666ul,p));
    return soft64_div(soft64_mul(epsilon,e),gmmax(soft64_sub(p,e),0x3ff0000000000000ul));
}
kernel void global_microphysics_fragment(
    constant GlobalMicrophysicsFragmentResources &r [[buffer(0)]],
    constant GlobalMicrophysicsFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.count))return;
    const ulong zero=0ul,one=0x3ff0000000000000ul;
    ulong t=soft64_mul(r.theta[tid],r.exner[tid]);
    ulong qv=r.qv[tid],qc=r.qc[tid],qr=r.qr[tid],qi=r.qi[tid],qs=r.qs[tid],qg=r.qg[tid];
    ulong qsat=gmqsat(t,r.pressure[tid],r.epsilon[0]);
    ulong deficit=gmmax(soft64_sub(qsat,qv),zero);
    ulong rain_evap=gmmin(qr,soft64_mul(deficit,r.rain_evap_fraction[0]));
    qr=soft64_sub(qr,rain_evap);qv=soft64_add(qv,rain_evap);
    t=soft64_sub(t,soft64_mul(r.lv_cp[0],rain_evap));
    ulong auto_c=soft64_mul(gmmax(soft64_sub(qc,r.auto_threshold[0]),zero),r.auto_fraction[0]);
    qc=soft64_sub(qc,auto_c);qr=soft64_add(qr,auto_c);
    ulong accrete=gmmin(qc,soft64_mul(soft64_mul(r.accrete_coefficient[0],qc),
        soft64_sqrt(gmmax(qr,zero))));
    qc=soft64_sub(qc,accrete);qr=soft64_add(qr,accrete);
    ulong freeze_w=gmclip(soft64_div(soft64_sub(0x4071126666666666ul,t),0x402e000000000000ul),zero,one);
    ulong freeze_frac=soft64_mul(freeze_w,r.freeze_fraction[0]);
    ulong freeze_cloud=soft64_mul(qc,freeze_frac),freeze_rain=soft64_mul(soft64_mul(qr,freeze_frac),0x3fe0000000000000ul);
    qc=soft64_sub(qc,freeze_cloud);qr=soft64_sub(qr,freeze_rain);
    qi=soft64_add(qi,freeze_cloud);qs=soft64_add(qs,freeze_rain);
    t=soft64_add(t,soft64_mul(r.lf_cp[0],soft64_add(freeze_cloud,freeze_rain)));
    ulong ice_snow=soft64_mul(soft64_mul(qi,r.ice_snow_fraction[0]),gmclip(soft64_div(qi,0x3f1a36e2eb1c432dul),zero,one));
    qi=soft64_sub(qi,ice_snow);qs=soft64_add(qs,ice_snow);
    ulong snow_gr=soft64_mul(soft64_mul(qs,r.snow_graupel_fraction[0]),gmclip(soft64_div(qc,0x3f40624dd2f1a9fcul),zero,one));
    qs=soft64_sub(qs,snow_gr);qg=soft64_add(qg,snow_gr);
    ulong melt_w=gmclip(soft64_div(soft64_sub(t,0x4071126666666666ul),0x4014000000000000ul),zero,one);
    ulong melt_frac=soft64_mul(melt_w,r.melt_fraction[0]);
    ulong melt_i=soft64_mul(qi,melt_frac),melt_s=soft64_mul(qs,melt_frac),melt_g=soft64_mul(qg,melt_frac);
    qi=soft64_sub(qi,melt_i);qc=soft64_add(qc,melt_i);
    qs=soft64_sub(qs,melt_s);qg=soft64_sub(qg,melt_g);qr=soft64_add(qr,soft64_add(melt_s,melt_g));
    t=soft64_sub(t,soft64_mul(r.lf_cp[0],soft64_add(melt_i,soft64_add(melt_s,melt_g))));
    qsat=gmqsat(t,r.pressure[tid],r.epsilon[0]);
    ulong cold_w=gmclip(soft64_div(soft64_sub(0x4071126666666666ul,t),0x4014000000000000ul),zero,one);
    ulong subl_frac=soft64_mul(cold_w,r.sublimation_fraction[0]);
    ulong latent_feedback=soft64_add(one,soft64_mul(r.ls_cp[0],soft64_div(
        soft64_mul(qsat,r.latent_heat_sub[0]),
        soft64_mul(r.rv[0],soft64_mul(t,t)))));
    ulong remaining=soft64_div(gmmax(soft64_sub(qsat,qv),zero),latent_feedback);
    ulong subl=gmmin(soft64_mul(qi,subl_frac),remaining);qi=soft64_sub(qi,subl);qv=soft64_add(qv,subl);remaining=soft64_sub(remaining,subl);t=soft64_sub(t,soft64_mul(r.ls_cp[0],subl));
    subl=gmmin(soft64_mul(qs,subl_frac),remaining);qs=soft64_sub(qs,subl);qv=soft64_add(qv,subl);remaining=soft64_sub(remaining,subl);t=soft64_sub(t,soft64_mul(r.ls_cp[0],subl));
    subl=gmmin(soft64_mul(qg,subl_frac),remaining);qg=soft64_sub(qg,subl);qv=soft64_add(qv,subl);t=soft64_sub(t,soft64_mul(r.ls_cp[0],subl));
    r.theta_out[tid]=soft64_div(t,r.exner[tid]);r.qv_out[tid]=qv;r.qc_out[tid]=qc;
    r.qr_out[tid]=qr;r.qi_out[tid]=qi;r.qs_out[tid]=qs;r.qg_out[tid]=qg;
}
