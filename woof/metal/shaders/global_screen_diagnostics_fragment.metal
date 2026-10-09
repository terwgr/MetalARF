// FP64 Monin-Obukhov 2 m / 10 m screen diagnostics.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
#include "global_screen_diagnostics_fragment_bindings.metalh"
#include "global_screen_diagnostics_resident_fragment_bindings.metalh"
using namespace metal;
static inline ulong gsmax(ulong a,ulong b){return soft64_greater(a,b)?a:b;}
static inline ulong gsmin(ulong a,ulong b){return soft64_less(a,b)?a:b;}
static inline ulong gsclip(ulong x,ulong a,ulong b){return gsmin(gsmax(x,a),b);}
static inline ulong gsatan_unit(ulong x){
    const ulong one=0x3ff0000000000000ul,pi4=0x3fe921fb54442d18ul,tanpi8=0x3fda827999fcef32ul;
    bool shifted=soft64_greater(x,tanpi8);
    if(shifted)x=soft64_div(soft64_sub(x,one),soft64_add(x,one));
    ulong x2=soft64_mul(x,x),term=x,sum=x;
    for(int n=1;n<24;++n){term=soft64_negate_bits(soft64_mul(term,x2));sum=soft64_add(sum,soft64_div(term,soft64_from_i32(2*n+1)));}
    return shifted?soft64_add(pi4,sum):sum;
}
static inline ulong gsatan(ulong x){
    const ulong pi2=0x3ff921fb54442d18ul,one=0x3ff0000000000000ul;
    return soft64_greater(x,one)?soft64_sub(pi2,gsatan_unit(soft64_div(one,x))):gsatan_unit(x);
}
static inline void gspsi(ulong z,thread ulong &pm,thread ulong &ph){
    const ulong zero=0ul,one=0x3ff0000000000000ul,two=0x4000000000000000ul;
    if(soft64_less(z,zero)){
        ulong x=soft64_plm_pow(soft64_sub(one,soft64_mul(0x4030000000000000ul,z)),0x3fd0000000000000ul);
        ulong a=soft64_mul(0x3fe0000000000000ul,soft64_add(one,x));
        ulong b=soft64_mul(0x3fe0000000000000ul,soft64_add(one,soft64_mul(x,x)));
        pm=soft64_add(soft64_sub(soft64_add(soft64_mul(two,soft64_plm_log(a)),soft64_plm_log(b)),soft64_mul(two,gsatan(x))),0x3ff921fb54442d18ul);
        ph=soft64_mul(two,soft64_plm_log(b));
    }else{pm=soft64_negate_bits(soft64_mul(0x4014000000000000ul,z));ph=pm;}
}
static inline void gsprofile(ulong z,ulong z0,ulong z0h,ulong zeta,ulong z1,
    thread ulong &f,thread ulong &g){
    ulong pmz,phz,pm0,ph0,dummy;
    gspsi(soft64_div(soft64_mul(zeta,z),z1),pmz,phz);
    gspsi(soft64_div(soft64_mul(zeta,z0),z1),pm0,dummy);
    gspsi(soft64_div(soft64_mul(zeta,z0h),z1),dummy,ph0);
    f=soft64_sub(soft64_sub(soft64_plm_log(soft64_div(z,z0)),pmz),soft64_negate_bits(pm0));
    g=soft64_sub(soft64_sub(soft64_plm_log(soft64_div(z,z0h)),phz),soft64_negate_bits(ph0));
}
kernel void global_screen_diagnostics_fragment(
    constant GlobalScreenDiagnosticsFragmentResources &r [[buffer(0)]],
    constant GlobalScreenDiagnosticsFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]){
    if(tid>=uint(p.count))return;
    const ulong zero=0ul,one=0x3ff0000000000000ul,two=0x4000000000000000ul;
    const ulong kappa=0x3fd24b795eda4351ul;
    ulong u=r.u[tid],v=r.v[tid],t=r.temperature[tid],q=r.qv[tid];
    ulong pf=r.p_full[tid],ps=r.p_surface[tid],skin=r.skin_temperature[tid];
    ulong qs=r.surface_humidity[tid],z0=gsmax(r.roughness[tid],0x3ee4f8b588e368f1ul);
    ulong theta=soft64_mul(t,soft64_plm_pow(soft64_div(0x40f86a0000000000ul,pf),kappa));
    ulong theta_skin=soft64_mul(skin,soft64_plm_pow(soft64_div(0x40f86a0000000000ul,ps),kappa));
    ulong virtual_t=soft64_mul(t,soft64_add(one,soft64_mul(0x3fe3851eb851eb85ul,q)));
    ulong z1=soft64_plm_log(soft64_div(ps,pf));
    z1=soft64_div(soft64_mul(soft64_mul(0x4071f00000000000ul,virtual_t),z1),0x40239cc100e6afcdul);
    z1=gsmax(z1,0x4014000000000000ul);
    z0=gsmin(z0,soft64_mul(0x3fd0000000000000ul,z1));
    ulong z0h=soft64_mul(0x3fb999999999999aul,z0);
    ulong speed=gsmax(soft64_sqrt(soft64_add(soft64_mul(u,u),soft64_mul(v,v))),0x3fb999999999999aul);
    ulong thv1=soft64_mul(theta,soft64_add(one,soft64_mul(0x3fe3851eb851eb85ul,q)));
    ulong thvs=soft64_mul(theta_skin,soft64_add(one,soft64_mul(0x3fe3851eb851eb85ul,qs)));
    ulong ri=soft64_div(soft64_mul(soft64_mul(0x40239cc100e6afcdul,z1),soft64_sub(thv1,thvs)),
        soft64_mul(soft64_mul(0x3fe0000000000000ul,soft64_add(thv1,thvs)),soft64_mul(speed,speed)));
    ulong lm=soft64_plm_log(soft64_div(z1,z0)),lh=soft64_plm_log(soft64_div(z1,z0h));
    ulong ris=gsclip(ri,zero,0x3fc851eb851eb852ul);
    ulong qa=soft64_mul(0x4014000000000000ul,soft64_sub(one,soft64_mul(0x4014000000000000ul,ris)));
    ulong qb=soft64_sub(lh,soft64_mul(soft64_mul(0x4024000000000000ul,ris),lm));
    ulong qc=soft64_negate_bits(soft64_mul(ris,soft64_mul(lm,lm)));
    ulong disc=gsmax(soft64_sub(soft64_mul(qb,qb),soft64_mul(0x4010000000000000ul,soft64_mul(qa,qc))),zero);
    ulong zs=gsmin(soft64_div(soft64_add(soft64_negate_bits(qb),soft64_sqrt(disc)),soft64_mul(two,qa)),0x4014000000000000ul);
    ulong ru=gsmin(ri,zero),zu=gsmax(soft64_div(soft64_mul(ru,soft64_mul(lm,lm)),lh),soft64_negate_bits(0x4024000000000000ul));
    for(int n=0;n<5;++n){ulong f1,g1;gsprofile(z1,z0,z0h,zu,z1,f1,g1);zu=gsmax(soft64_div(soft64_mul(ru,soft64_mul(f1,f1)),g1),soft64_negate_bits(0x4024000000000000ul));}
    ulong zeta=soft64_less(ri,zero)?zu:zs;
    ulong f1,g1,f10,unused,g2;
    gsprofile(z1,z0,z0h,zeta,z1,f1,g1);
    gsprofile(0x4024000000000000ul,z0,z0h,zeta,z1,f10,unused);
    gsprofile(two,z0,z0h,zeta,z1,unused,g2);
    ulong scalar=soft64_div(g2,g1),wind=soft64_div(f10,f1);
    ulong th2=soft64_add(theta_skin,soft64_mul(soft64_sub(theta,theta_skin),scalar));
    ulong q2=gsmax(soft64_add(qs,soft64_mul(soft64_sub(q,qs),scalar)),zero);
    ulong p2=soft64_mul(ps,soft64_plm_exp(soft64_negate_bits(soft64_div(
        soft64_mul(0x40239cc100e6afcdul,two),
        soft64_mul(0x4071f00000000000ul,soft64_mul(0x3fe0000000000000ul,soft64_add(virtual_t,skin)))))));
    r.t2[tid] = soft64_mul(th2, soft64_plm_pow(soft64_div(p2, 0x40f86a0000000000ul), kappa));
    r.th2[tid] = th2;
    r.q2[tid] = q2;
    r.u10[tid] = soft64_mul(u, wind);
    r.v10[tid] = soft64_mul(v, wind);
}

kernel void global_screen_diagnostics_resident_fragment(
    constant GlobalScreenDiagnosticsResidentFragmentResources &r [[buffer(0)]],
    constant GlobalScreenDiagnosticsResidentFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const ulong zero = 0ul;
    const ulong one = 0x3ff0000000000000ul;
    const ulong two = 0x4000000000000000ul;
    const ulong kappa = 0x3fd24b795eda4351ul;
    const ulong u = r.u[tid];
    const ulong v = r.v[tid];
    const ulong t = soft64_mul(r.theta[tid], r.exner[tid]);
    const ulong q = r.qv[tid];
    const ulong pf = r.p_full[tid];
    const ulong ps = r.p_surface[tid];
    const ulong skin = r.skin_temperature[tid];
    const ulong constants_vapor_limit = r.thermodynamic_constants[1];
    const ulong constants_vapor_pressure = r.thermodynamic_constants[2];
    const ulong constants_tetens_slope = r.thermodynamic_constants[3];
    const ulong constants_tetens_offset = r.thermodynamic_constants[4];
    const ulong constants_min_temperature = r.thermodynamic_constants[5];
    const ulong constants_max_temperature = r.thermodynamic_constants[6];
    const ulong epsilon = r.thermodynamic_constants[7];
    const ulong wetness_capacity = r.thermodynamic_constants[8];
    const ulong clipped_skin = gsclip(
        skin, constants_min_temperature, constants_max_temperature);
    const ulong exponent = soft64_div(
        soft64_mul(constants_tetens_slope,
                   soft64_sub(clipped_skin, constants_tetens_offset)),
        gsmax(soft64_sub(clipped_skin, r.thermodynamic_constants[9]), one));
    const ulong vapor_pressure = gsmin(
        soft64_mul(constants_vapor_pressure, soft64_plm_exp(exponent)),
        soft64_mul(constants_vapor_limit, ps));
    const ulong qsat = soft64_div(
        soft64_mul(epsilon, vapor_pressure),
        gsmax(soft64_sub(ps, vapor_pressure), one));
    const ulong land = r.land_fraction[tid];
    const ulong wetness = gsclip(
        soft64_div(r.soil_wetness[tid], wetness_capacity), zero, one);
    const ulong beta = soft64_add(soft64_sub(one, land), soft64_mul(land, wetness));
    const ulong qs = soft64_add(q, soft64_mul(beta, soft64_sub(qsat, q)));
    // Floor only; the 0.25 * z1 cap below is the host's upper bound.
    const ulong z0 = gsmax(r.roughness[tid], 0x3ee4f8b588e368f1ul);
    const ulong theta = soft64_mul(
        t, soft64_plm_pow(soft64_div(0x40f86a0000000000ul, pf), kappa));
    const ulong theta_skin = soft64_mul(
        skin, soft64_plm_pow(soft64_div(0x40f86a0000000000ul, ps), kappa));
    const ulong virtual_t = soft64_mul(
        t, soft64_add(one, soft64_mul(0x3fe3851eb851eb85ul, q)));
    ulong z1 = soft64_plm_log(soft64_div(ps, pf));
    z1 = soft64_div(
        soft64_mul(soft64_mul(0x4071f00000000000ul, virtual_t), z1),
        0x40239cc100e6afcdul);
    z1 = gsmax(z1, 0x4014000000000000ul);
    const ulong bounded_z0 = gsmin(z0, soft64_mul(0x3fd0000000000000ul, z1));
    const ulong z0h = soft64_mul(0x3fb999999999999aul, bounded_z0);
    const ulong speed = gsmax(
        soft64_sqrt(soft64_add(soft64_mul(u, u), soft64_mul(v, v))),
        0x3fb999999999999aul);
    const ulong thv1 = soft64_mul(
        theta, soft64_add(one, soft64_mul(0x3fe3851eb851eb85ul, q)));
    const ulong thvs = soft64_mul(
        theta_skin, soft64_add(one, soft64_mul(0x3fe3851eb851eb85ul, qs)));
    const ulong ri = soft64_div(
        soft64_mul(soft64_mul(0x40239cc100e6afcdul, z1), soft64_sub(thv1, thvs)),
        soft64_mul(soft64_mul(0x3fe0000000000000ul, soft64_add(thv1, thvs)),
                   soft64_mul(speed, speed)));
    const ulong lm = soft64_plm_log(soft64_div(z1, bounded_z0));
    const ulong lh = soft64_plm_log(soft64_div(z1, z0h));
    const ulong ris = gsclip(ri, zero, 0x3fc851eb851eb852ul);
    const ulong qa = soft64_mul(0x4014000000000000ul,
        soft64_sub(one, soft64_mul(0x4014000000000000ul, ris)));
    const ulong qb = soft64_sub(lh,
        soft64_mul(soft64_mul(0x4024000000000000ul, ris), lm));
    const ulong qc = soft64_negate_bits(soft64_mul(ris, soft64_mul(lm, lm)));
    const ulong disc = gsmax(soft64_sub(soft64_mul(qb, qb),
        soft64_mul(0x4010000000000000ul, soft64_mul(qa, qc))), zero);
    const ulong zs = gsmin(soft64_div(
        soft64_add(soft64_negate_bits(qb), soft64_sqrt(disc)),
        soft64_mul(two, qa)), 0x4014000000000000ul);
    const ulong ru = gsmin(ri, zero);
    ulong zu = gsmax(soft64_div(soft64_mul(ru, soft64_mul(lm, lm)), lh),
                     soft64_negate_bits(0x4024000000000000ul));
    for (int n = 0; n < 5; ++n) {
        ulong f1, g1;
        gsprofile(z1, bounded_z0, z0h, zu, z1, f1, g1);
        zu = gsmax(soft64_div(soft64_mul(ru, soft64_mul(f1, f1)), g1),
                   soft64_negate_bits(0x4024000000000000ul));
    }
    const ulong zeta = soft64_less(ri, zero) ? zu : zs;
    ulong f1, g1, f10, unused, g2;
    gsprofile(z1, bounded_z0, z0h, zeta, z1, f1, g1);
    gsprofile(0x4024000000000000ul, bounded_z0, z0h, zeta, z1, f10, unused);
    gsprofile(two, bounded_z0, z0h, zeta, z1, unused, g2);
    const ulong scalar = soft64_div(g2, g1);
    const ulong wind = soft64_div(f10, f1);
    const ulong th2 = soft64_add(theta_skin,
        soft64_mul(soft64_sub(theta, theta_skin), scalar));
    const ulong q2 = gsmax(
        soft64_add(qs, soft64_mul(soft64_sub(q, qs), scalar)), zero);
    const ulong p2 = soft64_mul(ps, soft64_plm_exp(soft64_negate_bits(
        soft64_div(soft64_mul(0x40239cc100e6afcdul, two),
                   soft64_mul(0x4071f00000000000ul,
                       soft64_mul(0x3fe0000000000000ul,
                                  soft64_add(virtual_t, skin)))))));
    r.t2[tid] = soft64_mul(
        th2, soft64_plm_pow(soft64_div(p2, 0x40f86a0000000000ul), kappa));
    r.th2[tid] = th2;
    r.q2[tid] = q2;
    r.u10[tid] = soft64_mul(u, wind);
    r.v10[tid] = soft64_mul(v, wind);
}
