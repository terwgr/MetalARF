#include <metal_stdlib>
#include "morrison_process_levels_bindings.metalh"
#include "morrison_prepare_fields_bindings.metalh"
#include "morrison_finish_bindings.metalh"
#include "morrison_validate_diagnostics_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_gfk_tgamma.metalh"
#include "soft_libm64.metalh"
#include "morrison_sediment_64_bindings.metalh"
#include "morrison_sediment_256_bindings.metalh"
#include "morrison_finalize_levels_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)


constant float MORR_MPI=3.14159265358979323846f;
constant float MORR_RHOW=997.0f;
constant float MORR_RHOI=500.0f;
constant float MORR_RHOS=100.0f;
constant float MORR_RHOSU=(85000.0f/(287.15f*273.15f));
constant float MORR_QSMALL=1.0e-14f;
constant float MORR_DCS=125.0e-6f;
constant float MORR_CI=(MORR_RHOI*MORR_MPI/6.0f);
constant float MORR_CS=(MORR_RHOS*MORR_MPI/6.0f);
constant float MORR_RD=287.0f;
constant float MORR_RV=461.6f;
constant float MORR_G=9.81f;
// Match CUDA_DEFINES' Python-computed EP2 literal rounded to float32.
constant float MORR_EP2=0.6217504143714905f;

// Positive-domain Morrison cbrt. Reduce the FP32 exponent modulo three, then
// perform seven Newton steps in software binary64 before the single final
// FP32 round. Subnormals are normalized by 2^24 first; zero and positive
// infinity retain their values.
inline float morr_msl_cbrtf_positive(float x) {
    if (isnan(x) || x < 0.0f) return as_type<float>(0x7fc00000u);
    if (x == 0.0f || isinf(x)) return x;
    uint bits=as_type<uint>(x);
    int raw_exp=int((bits>>23)&0xffu);
    if (raw_exp==0) {
        x*=16777216.0f;
        bits=as_type<uint>(x);
        raw_exp=int((bits>>23)&0xffu)-24;
    }
    const int exponent=raw_exp-127;
    int q=exponent/3;
    int rem=exponent-3*q;
    if (rem<0) { --q; rem+=3; }
    const uint mantissa_bits=(bits&0x007fffffu)|0x3f800000u;
    const float mantissa=as_type<float>(mantissa_bits)*float(1u<<uint(rem));
    const ulong m=soft64_from_f32_bits(as_type<uint>(mantissa));
    const float seed=1.0f+(mantissa-1.0f)*0.2f;
    ulong y=soft64_from_f32_bits(as_type<uint>(seed));
    for (uint i=0;i<7;++i) {
        const ulong yy=soft64_mul(y,y);
        y=soft64_div(soft64_add(soft64_mul(0x4000000000000000ul,y),
                                 soft64_div(m,yy)),
                     0x4008000000000000ul);
    }
    const ulong scale=ulong(1023+q)<<52;
    return as_type<float>(soft64_to_f32_bits(soft64_mul(y,scale)));
}

struct MorrMoments { float lc,lr,li,ls,lg,pg; };

inline float morr_polysvp(float t, bool ice) {
    float x=t-273.15f;
    if (ice) {
        float p=0.252751365e-14f;
        p=0.146898966e-11f+x*p; p=0.385852041e-9f+x*p;
        p=0.602588177e-7f+x*p; p=0.615021634e-5f+x*p;
        p=0.420895665e-3f+x*p; p=0.188439774e-1f+x*p;
        p=0.503160820f+x*p; p=6.11147274f+x*p;
        if (t>=195.8f) return p*100.0f;
        float ratio=273.16f/t;
        float exponent=-9.09718f*(ratio-1.0f);
        exponent=exponent-3.56654f*log10(ratio);
        exponent=exponent+0.876793f*(1.0f-t/273.16f);
        exponent=exponent+log10(6.1071f);
        return pow(10.0f,exponent)*100.0f;
    }
    float p=-0.976195544e-15f;
    p=-0.952447341e-13f+x*p; p=0.640689451e-10f+x*p;
    p=0.206739458e-7f+x*p; p=0.302950461e-5f+x*p;
    p=0.264847430e-3f+x*p; p=0.142986287e-1f+x*p;
    p=0.443987641f+x*p; p=6.11239921f+x*p;
    if (t>=202.0f) return p*100.0f;
    float ratio=373.16f/t;
    float exponent=-7.90298f*(ratio-1.0f);
    exponent=exponent+5.02808f*log10(ratio);
    float inner=11.344f*(1.0f-t/373.16f);
    exponent=exponent-1.3816e-7f*(pow(10.0f,inner)-1.0f);
    inner=-3.49149f*(ratio-1.0f);
    exponent=exponent+8.1328e-3f*(pow(10.0f,inner)-1.0f);
    exponent=exponent+log10(1013.246f);
    return pow(10.0f,exponent)*100.0f;
}

inline void morr_bound_one(float q,float six_c,float lo,float hi,
                           thread float &number,thread float &lambda) {
    number=fmax(number,0.0f);
    if (q<MORR_QSMALL) { number=0.0f; lambda=0.0f; return; }
    float raw=morr_msl_cbrtf_positive(six_c*number/q);
    lambda=fmin(fmax(raw,lo),hi);
    if (lambda!=raw) number=(q*lambda*lambda*lambda)/six_c;
}

inline MorrMoments morr_bound(float qc,float qr,float qi,float qs,float qg,
        float rhoa,float pres,float temp,thread float &nc,thread float &nr,
        thread float &ni,thread float &ns,thread float &ng,
        bool reset_cloud_number,float morr_rhog) {
    MorrMoments m={0.0f,0.0f,0.0f,0.0f,0.0f,2.0f};
    if (reset_cloud_number) nc=250.0e6f/rhoa;
    if (qc>=MORR_QSMALL) {
        float rho_cloud=pres/(287.15f*temp);
        float pp=0.0005714f*((nc/1.0e6f)*rho_cloud)+0.2714f;
        m.pg=fmin(fmax(1.0f/(pp*pp)-1.0f,2.0f),10.0f);
        float raw=morr_msl_cbrtf_positive((MORR_MPI/6.0f*MORR_RHOW*nc*soft_gfk_tgammaf(m.pg+4.0f))
                       /(qc*soft_gfk_tgammaf(m.pg+1.0f)));
        float lo=(m.pg+1.0f)/60.0e-6f, hi=(m.pg+1.0f)/1.0e-6f;
        m.lc=fmin(fmax(raw,lo),hi);
        if (m.lc!=raw)
            nc=m.lc*m.lc*m.lc*qc*soft_gfk_tgammaf(m.pg+1.0f)
               /((MORR_MPI/6.0f*MORR_RHOW)*soft_gfk_tgammaf(m.pg+4.0f));
    }
    morr_bound_one(qr,MORR_MPI*MORR_RHOW,1.0f/2800.0e-6f,
                   1.0f/20.0e-6f,nr,m.lr);
    morr_bound_one(qi,6.0f*MORR_CI,1.0f/(2.0f*MORR_DCS+100.0e-6f),
                   1.0f/1.0e-6f,ni,m.li);
    morr_bound_one(qs,6.0f*MORR_CS,1.0f/2000.0e-6f,
                   1.0f/10.0e-6f,ns,m.ls);
    float cg=morr_rhog*MORR_MPI/6.0f;
    morr_bound_one(qg,6.0f*cg,1.0f/2000.0e-6f,
                   1.0f/20.0e-6f,ng,m.lg);
    return m;
}

inline void morr_terminal_velocity(int kind,float q,float number,float rhoa,
        float temp,thread float &bounded_number,thread float &vm,
        thread float &vn,float ag,float bg,float rhog) {
    float nn=fmax(number,0.0f), ll=0.0f, pg=2.0f;
    if (kind==0) {
        if (q>=MORR_QSMALL) {
            float rho_cloud=(rhoa*MORR_RD)/287.15f;
            float pp=0.0005714f*((nn/1.0e6f)*rho_cloud)+0.2714f;
            pg=fmin(fmax(1.0f/(pp*pp)-1.0f,2.0f),10.0f);
            ll=morr_msl_cbrtf_positive((MORR_MPI/6.0f*MORR_RHOW*nn*soft_gfk_tgammaf(pg+4.0f))
                    /(q*soft_gfk_tgammaf(pg+1.0f)));
            ll=fmin(fmax(ll,(pg+1.0f)/60.0e-6f),(pg+1.0f)/1.0e-6f);
        }
    } else {
        float six_c,lo,hi;
        if(kind==1){six_c=MORR_MPI*MORR_RHOW;lo=1.0f/2800.0e-6f;hi=1.0f/20.0e-6f;}
        else if(kind==2){six_c=6.0f*MORR_CI;lo=1.0f/(2.0f*MORR_DCS+100.0e-6f);hi=1.0f/1.0e-6f;}
        else if(kind==3){six_c=6.0f*MORR_CS;lo=1.0f/2000.0e-6f;hi=1.0f/10.0e-6f;}
        else{six_c=rhog*MORR_MPI;lo=1.0f/2000.0e-6f;hi=1.0f/20.0e-6f;}
        morr_bound_one(q,six_c,lo,hi,nn,ll);
    }
    bounded_number=nn;
    if(q<MORR_QSMALL){vm=0.0f;vn=0.0f;}
    else if(kind==0){
        float mu=1.496e-6f*pow(temp,1.5f)/(temp+120.0f);
        float acn=MORR_G*MORR_RHOW/(18.0f*mu);
        vm=acn*soft_gfk_tgammaf(6.0f+pg)/(ll*ll*soft_gfk_tgammaf(pg+4.0f));
        vn=acn*soft_gfk_tgammaf(3.0f+pg)/(ll*ll*soft_gfk_tgammaf(pg+1.0f));
    } else {
        float aa,bb,cap,de;
        if(kind==1){aa=841.99667f;bb=0.8f;cap=9.1f;de=0.54f;}
        else if(kind==2){aa=700.0f;bb=1.0f;cap=1.2f;de=0.35f;}
        else if(kind==3){aa=11.72f;bb=0.41f;cap=1.2f;de=0.54f;}
        else{aa=ag;bb=bg;cap=20.0f;de=0.54f;}
        float an=aa*pow(MORR_RHOSU/rhoa,de);
        vm=an*soft_gfk_tgammaf(4.0f+bb)/(6.0f*pow(ll,bb));
        vn=an*soft_gfk_tgammaf(1.0f+bb)/pow(ll,bb);
        float vcap=cap*pow(MORR_RHOSU/rhoa,kind==2?0.35f:0.54f);
        vm=fmin(vm,vcap); vn=fmin(vn,vcap);
    }
}

// Process-level rate math transcribed from morrison.cu. Morrison's adapted
// mathematical Gamma is rounded to FP32 at each existing call boundary;
// agreement with CUDA's distinct tgammaf remains unverified.
inline float __fdiv_rn(float a,float b) { return a/b; }
constant float MORR_CP=1004.5f;
constant float MORR_MMI0=(4.0f/3.0f*MORR_MPI*MORR_RHOI*1.0e-15f);
constant float MORR_MMG0=1.6e-10f;
struct MorrRates {
    float prc, nprc, nprc1, pra, npra, nragg, pre;
    float pracs, npracs, pracg, npracg, psmlt, evpms, pgmlt, evpmg;
    float nsmlts, nsmltr, ngmltg, ngmltr, nsubr;
    float mnuccc, nnuccc, nsagg, psacws, npsacws, psacwi, npsacwi;
    float psacwg, npsacwg, qmults, nmults, qmultr, nmultr;
    float qmultg, nmultg, qmultrg, nmultrg, pgsacw, pgracs, psacr;
    float nscng, ngracs, mnuccr, nnuccr, prci, nprci, prai, nprai;
    float nnuccd, mnuccd, prd, prds, prdg, eprd, eprds, eprdg;
    float piacr, niacr, praci, piacrs, niacrs, pracis;
    float nsubi, nsubs, nsubg;
};

inline ulong morr_widen(float x) { return soft64_from_f32_bits(as_type<uint>(x)); }
inline float morr_narrow(ulong x) { return as_type<float>(soft64_to_f32_bits(x)); }
inline ulong morr_wconst(ulong bits) { return bits; }
inline ulong morr_wexp(ulong x) { return soft64_plm_exp(x); }
// The reachable Morrison cold-cloud recovery only raises its positive,
// bounded lambda to the exact integer powers 3, 4, and 6. Keep every product
// in software binary64, but avoid the general libm pow lowering here: Apple's
// MTLCompilerService aborts while legalizing that unused general-exponent path
// on this shader. As with other libm implementations, the product grouping can
// differ by a few binary64 low bits from CUDA's pow(double, double); the
// overflow-protection range and binary64 intermediates are retained.
inline ulong morr_wpow3(ulong x) {
    const ulong x2=soft64_mul(x,x);
    return soft64_mul(x2,x);
}
inline ulong morr_wpow4(ulong x) {
    const ulong x2=soft64_mul(x,x);
    return soft64_mul(x2,x2);
}
inline ulong morr_wpow6(ulong x) {
    const ulong x2=soft64_mul(x,x);
    const ulong x3=soft64_mul(x2,x);
    return soft64_mul(x3,x3);
}
inline ulong morr_wmin(ulong a, ulong b) {
    if (soft64_is_nan(a)) return b;
    if (soft64_is_nan(b)) return a;
    return soft64_less(a,b) ? a : b;
}

inline ulong morr_cloud_freezing(
        float nc,float lambda,float pg,float temp,float pressure,float mu,
        float dt,thread float* number) {
    const float nacnt_exponent=-2.80f+0.262f*(273.15f-temp);
    const float nacnt=exp(nacnt_exponent)*1000.0f;
    const float slip=(7.37f*temp/(288.0f*10.0f*pressure))/100.0f;
    const float rin=0.1e-6f;
    const float dap=(4.0f*MORR_MPI*1.38e-23f/(6.0f*MORR_MPI*rin)
                     *temp*(1.0f+slip/rin)/mu);
    const float cdist=nc/soft_gfk_tgammaf(pg+1.0f);
    const float exponent=0.66f*(273.15f-temp);
    const float bigg=exp(exponent)-1.0f;
    const float logdist=log(cdist),loglambda=log(lambda);
    const float contact_factor=exp(logdist+log(soft_gfk_tgammaf(pg+5.0f))-4.0f*loglambda);
    const float immersion_factor=exp(logdist+log(soft_gfk_tgammaf(pg+7.0f))-6.0f*loglambda);
    const float number_factor=exp(logdist+log(soft_gfk_tgammaf(pg+4.0f))-3.0f*loglambda);
    const float contact_coefficient=MORR_MPI*MORR_MPI/3.0f*MORR_RHOW;
    const float immersion_coefficient=MORR_MPI*MORR_MPI/36.0f*MORR_RHOW*100.0f;
    const float number_coefficient=MORR_MPI/6.0f*100.0f;
    const float contact=contact_coefficient*dap*nacnt*contact_factor;
    const float immersion=immersion_coefficient*immersion_factor*bigg;
    const float count=2.0f*MORR_MPI*dap*nacnt*cdist*soft_gfk_tgammaf(pg+2.0f)/lambda
                      +number_coefficient*number_factor*bigg;
    if(isfinite(contact+immersion)&&isfinite(count)&&contact_factor>0.0f
       &&immersion_factor>0.0f&&number_factor>0.0f&&isfinite(bigg)) {
        *number=min(count,nc/dt);
        return morr_widen(contact+immersion);
    }
    // CUDA's exceptional recovery widens these FP32 values and performs each
    // following operation in binary64. exp is the pinned soft64_plm_exp;
    // Morrison's mathematical Gamma is rounded to FP32 before widening.
    const ulong dnc=morr_widen(nc),dlam=morr_widen(lambda);
    const ulong dist=soft64_div(dnc,morr_widen(soft_gfk_tgammaf(pg+1.0f)));
    const ulong nuclei=soft64_mul(morr_wexp(morr_widen(nacnt_exponent)),
                                  0x408f400000000000ul); // * 1000
    const ulong freezing=soft64_sub(morr_wexp(morr_widen(exponent)),
                                    0x3ff0000000000000ul);
    const ulong contact_wide=soft64_div(
        soft64_mul(soft64_mul(soft64_mul(soft64_mul(morr_widen(contact_coefficient),
        morr_widen(dap)),nuclei),dist),morr_widen(soft_gfk_tgammaf(pg+5.0f))),
        morr_wpow4(dlam));
    ulong immersion_wide=soft64_mul(morr_widen(immersion_coefficient),dist);
    immersion_wide=soft64_mul(immersion_wide,morr_widen(soft_gfk_tgammaf(pg+7.0f)));
    immersion_wide=soft64_div(immersion_wide,morr_wpow6(dlam));
    immersion_wide=soft64_mul(immersion_wide,freezing);
    const ulong first=soft64_div(soft64_mul(soft64_mul(soft64_mul(soft64_mul(
        morr_widen(2.0f*MORR_MPI),morr_widen(dap)),nuclei),dist),
        morr_widen(soft_gfk_tgammaf(pg+2.0f))),dlam);
    ulong second=soft64_mul(morr_widen(number_coefficient),dist);
    second=soft64_mul(second,morr_widen(soft_gfk_tgammaf(pg+4.0f)));
    second=soft64_div(second,morr_wpow3(dlam));
    second=soft64_mul(second,freezing);
    const ulong count_wide=soft64_add(first,second);
    const ulong cap=soft64_div(dnc,morr_widen(dt));
    *number=morr_narrow(morr_wmin(count_wide,cap));
    return soft64_add(contact_wide,immersion_wide);
}

inline void morr_limit_cold_cloud(thread MorrRates& r,float qc,float dt,ulong freezing) {
    float loss=(r.prc+r.pra+r.mnuccc+r.psacws+r.psacwi+r.qmults+r.psacwg+r.pgsacw+r.qmultg)*dt;
    if(!isfinite(loss)) {
        ulong other=morr_widen(r.prc);
        other=soft64_add(other,morr_widen(r.pra)); other=soft64_add(other,morr_widen(r.psacws));
        other=soft64_add(other,morr_widen(r.psacwi)); other=soft64_add(other,morr_widen(r.qmults));
        other=soft64_add(other,morr_widen(r.psacwg)); other=soft64_add(other,morr_widen(r.pgsacw));
        other=soft64_add(other,morr_widen(r.qmultg));
        const ulong denom=soft64_add(other,freezing);
        const ulong ratio=morr_wmin(0x3ff0000000000000ul,
            soft64_div(soft64_div(morr_widen(qc),morr_widen(dt)),denom));
        r.prc=morr_narrow(soft64_mul(morr_widen(r.prc),ratio));
        r.pra=morr_narrow(soft64_mul(morr_widen(r.pra),ratio));
        r.mnuccc=morr_narrow(soft64_mul(freezing,ratio));
        r.psacws=morr_narrow(soft64_mul(morr_widen(r.psacws),ratio));
        r.psacwi=morr_narrow(soft64_mul(morr_widen(r.psacwi),ratio));
        r.qmults=morr_narrow(soft64_mul(morr_widen(r.qmults),ratio));
        r.qmultg=morr_narrow(soft64_mul(morr_widen(r.qmultg),ratio));
        r.psacwg=morr_narrow(soft64_mul(morr_widen(r.psacwg),ratio));
        r.pgsacw=morr_narrow(soft64_mul(morr_widen(r.pgsacw),ratio));
    } else if(loss>qc&&qc>=MORR_QSMALL) {
        float ratio=qc/loss;r.prc*=ratio;r.pra*=ratio;r.mnuccc*=ratio;
        r.psacws*=ratio;r.psacwi*=ratio;r.qmults*=ratio;r.qmultg*=ratio;
        r.psacwg*=ratio;r.pgsacw*=ratio;
    }
}

inline void morr_limit_cold_rain(thread MorrRates& r,float qr,float nr,float lambda,float temp,float dt) {
    if(!isfinite(r.mnuccr)) {
        const ulong exponent=morr_widen(0.66f*(273.15f-temp));
        const ulong lambda3=soft64_mul(soft64_mul(morr_widen(lambda),morr_widen(lambda)),morr_widen(lambda));
        const float coefficient=20.0f*MORR_MPI*MORR_MPI*MORR_RHOW*100.0f;
        const ulong freezing=soft64_div(soft64_div(
            soft64_mul(soft64_mul(morr_widen(coefficient),morr_widen(nr)),
            soft64_sub(morr_wexp(exponent),0x3ff0000000000000ul)),lambda3),lambda3);
        ulong other=soft64_negate_bits(morr_widen(r.pre));
        other=soft64_add(other,morr_widen(r.qmultr));other=soft64_add(other,morr_widen(r.qmultrg));
        other=soft64_add(other,morr_widen(r.pracs));other=soft64_add(other,morr_widen(r.piacr));
        other=soft64_add(other,morr_widen(r.piacrs));other=soft64_add(other,morr_widen(r.pgracs));
        other=soft64_add(other,morr_widen(r.pracg));
        ulong supply=soft64_add(soft64_div(morr_widen(qr),morr_widen(dt)),morr_widen(r.prc));
        supply=soft64_add(supply,morr_widen(r.pra));
        const ulong ratio=morr_wmin(0x3ff0000000000000ul,
            soft64_div(supply,soft64_add(other,freezing)));
        r.pre=morr_narrow(soft64_mul(morr_widen(r.pre),ratio));
        r.pracs=morr_narrow(soft64_mul(morr_widen(r.pracs),ratio));
        r.qmultr=morr_narrow(soft64_mul(morr_widen(r.qmultr),ratio));
        r.qmultrg=morr_narrow(soft64_mul(morr_widen(r.qmultrg),ratio));
        r.mnuccr=morr_narrow(soft64_mul(freezing,ratio));
        r.piacr=morr_narrow(soft64_mul(morr_widen(r.piacr),ratio));
        r.piacrs=morr_narrow(soft64_mul(morr_widen(r.piacrs),ratio));
        r.pgracs=morr_narrow(soft64_mul(morr_widen(r.pgracs),ratio));
        r.pracg=morr_narrow(soft64_mul(morr_widen(r.pracg),ratio));
    } else {
        float loss=((r.pracs-r.pre)+(r.qmultr+r.qmultrg-r.prc)+(r.mnuccr-r.pra)
                    +r.piacr+r.piacrs+r.pgracs+r.pracg)*dt;
        if(loss>qr&&qr>=MORR_QSMALL){float ratio=(qr/dt+r.prc+r.pra)/(-r.pre+r.qmultr+r.qmultrg+r.pracs+r.mnuccr+r.piacr+r.piacrs+r.pgracs+r.pracg);
            r.pre*=ratio;r.pracs*=ratio;r.qmultr*=ratio;r.qmultrg*=ratio;r.mnuccr*=ratio;
            r.piacr*=ratio;r.piacrs*=ratio;r.pgracs*=ratio;r.pracg*=ratio;}
    }
}

inline void morr_process_level(
        thread float* qv, thread float* qc, thread float* qr, thread float* qi, thread float* qs, thread float* qg,
        thread float* nc, thread float* nr, thread float* ni, thread float* ns, thread float* ng,
        thread float* temp, float pressure, float rhoa, float dt,
        float qvs, float qvi, float xlv, float xls, float cpm,
        bool warm, thread float* stale_lami, thread float* cloud_nc_for_sedimentation,
        float morr_ag, float morr_bg, float morr_rhog)
{
    float xlf = xls - xlv;
    // WRF 1424-1479 evaluates MU/DV/SC/AB before the warm branch's
    // small-snow/graupel melt (1498-1514).  The melt changes temperature,
    // but these transport/psychrometric coefficients stay stale.
    float mu = 1.496e-6f * pow(*temp, 1.5f) / (*temp + 120.0f);
    float dv = 8.794e-5f * pow(*temp, 1.81f) / pressure;
    float sc = mu / (rhoa * dv);
    float kap = 1.414e3f * mu;
    float ab = 1.0f + xlv * qvs / (MORR_RV * *temp * *temp) * xlv / cpm;
    float abi = 1.0f + xls * qvi / (MORR_RV * *temp * *temp) * xls / cpm;
    if (warm) {
        if (*qs < 1.0e-6f) {
            *qr += *qs; *nr += *ns; *temp -= *qs * xlf / cpm;
            *qs = 0.0f; *ns = 0.0f;
        }
        if (*qg < 1.0e-6f) {
            *qr += *qg; *nr += *ng; *temp -= *qg * xlf / cpm;
            *qg = 0.0f; *ng = 0.0f;
        }
    }
    MorrMoments m = morr_bound(*qc,*qr,*qi,*qs,*qg,rhoa,pressure,*temp,
                                *nc,*nr,*ni,*ns,*ng,true,morr_rhog);
    *stale_lami = warm ? 0.0f : m.li;
    // INUM=1: DUMFNC=NC3D, without NC3DTEN, at WRF 3367-3374.
    *cloud_nc_for_sedimentation = *nc;
    MorrRates r = {};
    float dens54 = pow(MORR_RHOSU / rhoa, 0.54f);
    float ain = 700.0f * pow(MORR_RHOSU / rhoa, 0.35f);
    float arn = 841.99667f * dens54, asn = 11.72f * dens54;
    float agn = morr_ag * dens54;
    float n0r = *nr * m.lr, n0i = *ni * m.li;
    float n0s = *ns * m.ls, n0g = *ng * m.lg;

    // Zeroed rate-vector evaluation shared by both branches.
    if (*qc >= 1.0e-6f) {
        r.prc = 1350.0f * pow(*qc, 2.47f)
                * pow(__fdiv_rn(*nc, 1.0e6f) * rhoa, -1.79f);
        r.nprc1 = __fdiv_rn(r.prc, (4.0f / 3.0f * MORR_MPI * MORR_RHOW
                           * 1.5625e-14f));
        r.nprc = fmin(r.prc / (*qc / *nc), *nc / dt);
        r.nprc1 = fmin(r.nprc1, r.nprc);
    }
    if (*qr >= 1.0e-8f && *qc >= 1.0e-8f) {
        r.pra = 67.0f * pow(*qc * *qr, 1.15f);
        r.npra = r.pra / (*qc / *nc);
    }
    if (*qr >= 1.0e-8f) {
        float fb = 1.0f / m.lr < 300.0e-6f ? 1.0f
                  : 2.0f - exp(2300.0f * (1.0f / m.lr - 300.0e-6f));
        r.nragg = -5.78f * fb * *nr * *qr * rhoa;
    }
    if (*qr >= MORR_QSMALL) {
        float epsr = 2.0f * MORR_MPI * n0r * rhoa * dv
                    * (0.78f / (m.lr * m.lr)
                       + 0.308f * sqrt(arn * rhoa / mu)
                       * morr_msl_cbrtf_positive(sc) * soft_gfk_tgammaf(2.9f) / pow(m.lr, 2.9f));
        if (*qv < qvs) r.pre = fmin(epsr * (*qv - qvs) / ab, 0.0f);
    }

    if (warm) {
        if (*qr >= 1.0e-8f && *qs >= 1.0e-8f) {
            float ums = fmin(asn * soft_gfk_tgammaf(4.41f) / (6.0f * pow(m.ls, 0.41f)),
                             1.2f * dens54);
            float umr = fmin(arn * soft_gfk_tgammaf(4.8f) / (6.0f * pow(m.lr, 0.8f)),
                             9.1f * dens54);
            r.pracs = MORR_MPI * MORR_MPI * MORR_RHOW
                      * sqrt((1.2f * umr - 0.95f * ums)
                              * (1.2f * umr - 0.95f * ums)
                              + 0.08f * ums * umr)
                      * rhoa * n0r * n0s / (m.lr * m.lr * m.lr)
                      * (5.0f / (pow(m.lr, 3.0f) * m.ls)
                         + 2.0f / (m.lr * m.lr * m.ls * m.ls)
                         + 0.5f / (m.lr * pow(m.ls, 3.0f)));
        }
        if (*qr >= 1.0e-8f && *qg >= 1.0e-8f) {
            float umg = fmin(agn * soft_gfk_tgammaf(4.0f + morr_bg)
                             / (6.0f * pow(m.lg, morr_bg)),
                             20.0f * dens54);
            float ung = fmin(agn * soft_gfk_tgammaf(1.0f + morr_bg)
                             / pow(m.lg, morr_bg),
                             20.0f * dens54);
            float umr = fmin(arn * soft_gfk_tgammaf(4.8f) / (6.0f * pow(m.lr, 0.8f)),
                             9.1f * dens54);
            float unr = fmin(arn * soft_gfk_tgammaf(1.8f) / pow(m.lr, 0.8f),
                             9.1f * dens54);
            float vrelm = sqrt((1.2f * umr - 0.95f * umg)
                               * (1.2f * umr - 0.95f * umg)
                               + 0.08f * umg * umr);
            r.pracg = MORR_MPI * MORR_MPI * MORR_RHOW * vrelm * rhoa * n0r * n0g
                      / pow(m.lr, 3.0f)
                      * (5.0f / (pow(m.lr, 3.0f) * m.lg)
                         + 2.0f / (m.lr * m.lr * m.lg * m.lg)
                         + 0.5f / (m.lr * pow(m.lg, 3.0f)));
            float vreln = sqrt(1.7f * (unr - ung) * (unr - ung)
                               + 0.3f * unr * ung);
            float collected_n = MORR_MPI / 2.0f * rhoa * vreln * n0r * n0g
                               * (1.0f / (pow(m.lr, 3.0f) * m.lg)
                                  + 1.0f / (m.lr * m.lr * m.lg * m.lg)
                                  + 1.0f / (m.lr * pow(m.lg, 3.0f)));
            r.npracg = collected_n - __fdiv_rn(r.pracg, 5.2e-7f);
        }
        if (*qs >= 1.0e-8f) {
            float accel = -4187.0f / xlf * (*temp - 273.15f) * r.pracs;
            r.psmlt = 2.0f * MORR_MPI * n0s * kap * (273.15f - *temp) / xlf
                      * (0.86f / (m.ls * m.ls)
                         + 0.28f * sqrt(asn * rhoa / mu) * morr_msl_cbrtf_positive(sc)
                         * soft_gfk_tgammaf(2.705f) / pow(m.ls, 2.705f)) + accel;
            if (*qv / qvs < 1.0f) {
                float epss = 2.0f * MORR_MPI * n0s * rhoa * dv
                            * (0.86f / (m.ls * m.ls)
                               + 0.28f * sqrt(asn * rhoa / mu) * morr_msl_cbrtf_positive(sc)
                               * soft_gfk_tgammaf(2.705f) / pow(m.ls, 2.705f));
                r.evpms = fmax((*qv - qvs) * epss / ab, r.psmlt);
                r.psmlt -= r.evpms;
            }
        }
        if (*qg >= 1.0e-8f) {
            float accel = -4187.0f / xlf * (*temp - 273.15f) * r.pracg;
            r.pgmlt = 2.0f * MORR_MPI * n0g * kap * (273.15f - *temp) / xlf
                      * (0.86f / (m.lg * m.lg)
                         + 0.28f * sqrt(agn * rhoa / mu) * morr_msl_cbrtf_positive(sc)
                         * soft_gfk_tgammaf(2.5f + 0.5f * morr_bg)
                         / pow(m.lg, 2.5f + 0.5f * morr_bg)) + accel;
            if (*qv / qvs < 1.0f) {
                float epsg = 2.0f * MORR_MPI * n0g * rhoa * dv
                            * (0.86f / (m.lg * m.lg)
                               + 0.28f * sqrt(agn * rhoa / mu) * morr_msl_cbrtf_positive(sc)
                               * soft_gfk_tgammaf(2.5f + 0.5f * morr_bg)
                               / pow(m.lg, 2.5f + 0.5f * morr_bg));
                r.evpmg = fmax((*qv - qvs) * epsg / ab, r.pgmlt);
                r.pgmlt -= r.evpmg;
            }
        }
        // WRF 1924-1930 collision reset, then mass-only donor ratios.
        r.pracg = 0.0f; r.pracs = 0.0f;
        float loss = (r.prc + r.pra) * dt;
        if (loss > *qc && *qc >= MORR_QSMALL) {
            float ratio = *qc / loss; r.prc *= ratio; r.pra *= ratio;
        }
        loss = (-r.psmlt - r.evpms + r.pracs) * dt;
        if (loss > *qs && *qs >= MORR_QSMALL) {
            float ratio = *qs / loss;
            r.psmlt *= ratio; r.evpms *= ratio; r.pracs *= ratio;
        }
        loss = (-r.pgmlt - r.evpmg + r.pracg) * dt;
        if (loss > *qg && *qg >= MORR_QSMALL) {
            float ratio = *qg / loss;
            r.pgmlt *= ratio; r.evpmg *= ratio; r.pracg *= ratio;
        }
        loss = (-r.pracs - r.pracg - r.pre - r.pra - r.prc
                + r.psmlt + r.pgmlt) * dt;
        if (loss > *qr && *qr >= MORR_QSMALL) {
            float ratio = (*qr / dt + r.pracs + r.pracg + r.pra + r.prc
                          - r.psmlt - r.pgmlt) / (-r.pre);
            r.pre *= ratio;
        }
        float tqv = -r.pre - r.evpms - r.evpmg;
        float tt = (r.pre * xlv + (r.evpms + r.evpmg) * xls
                   + (r.psmlt + r.pgmlt - r.pracs - r.pracg) * xlf) / cpm;
        float tqc = -r.pra - r.prc;
        float tqr = r.pre + r.pra + r.prc - r.psmlt - r.pgmlt
                   + r.pracs + r.pracg;
        float tqi = 0.0f, tqs = r.psmlt + r.evpms - r.pracs;
        float tqg = r.pgmlt + r.evpmg - r.pracg;
        if (r.pre < 0.0f)
            r.nsubr = fmax(-1.0f, r.pre * dt / *qr) * *nr / dt;
        if (r.evpms + r.psmlt < 0.0f)
            r.nsmlts = fmax(-1.0f, (r.evpms + r.psmlt) * dt / *qs) * *ns / dt;
        if (r.psmlt < 0.0f)
            r.nsmltr = fmax(-1.0f, r.psmlt * dt / *qs) * *ns / dt;
        if (r.evpmg + r.pgmlt < 0.0f)
            r.ngmltg = fmax(-1.0f, (r.evpmg + r.pgmlt) * dt / *qg) * *ng / dt;
        if (r.pgmlt < 0.0f)
            r.ngmltr = fmax(-1.0f, r.pgmlt * dt / *qg) * *ng / dt;
        float tnc = -r.npra - r.nprc;
        float tnr = r.nprc1 + r.nragg - r.npracg + r.nsubr
                   - r.nsmltr - r.ngmltr;
        float tni = 0.0f, tns = r.nsmlts, tng = r.ngmltg;

        float tpred = *temp + dt * tt, qvpred = *qv + dt * tqv;
        float qcpred = fmax(*qc + dt * tqc, 0.0f);
        float ew = fmin(0.99f * pressure, morr_polysvp(tpred, false));
        float qss = MORR_EP2 * ew / (pressure - ew);
        float pcc = (qvpred - qss)
                   / (1.0f + xlv * xlv * qss / (cpm * MORR_RV * tpred * tpred)) / dt;
        if (pcc * dt + qcpred < 0.0f) pcc = -qcpred / dt;
        tqv -= pcc; tqc += pcc; tt += pcc * xlv / cpm;
        *qv += dt * tqv; *qc += dt * tqc; *qr += dt * tqr;
        *qi += dt * tqi; *qs += dt * tqs; *qg += dt * tqg;
        *nc += dt * tnc; *nr += dt * tnr; *ni += dt * tni;
        *ns += dt * tns; *ng += dt * tng; *temp += dt * tt;
        return;
    }

    // Cold rate evaluation from the single begin-of-process snapshot.
    ulong cloud_freezing = 0ul;
    if (*qc >= MORR_QSMALL && *temp < 269.15f) {
        cloud_freezing = morr_cloud_freezing(*nc, m.lc, m.pg, *temp,
                                              pressure, mu, dt, &r.nnuccc);
        r.mnuccc = morr_narrow(cloud_freezing);
    }
    if (*qs >= 1.0e-8f) {
        float cons15 = __fdiv_rn(-1108.0f * 0.1f * pow(MORR_MPI, (1.0f - 0.41f) / 3.0f)
                      * pow(MORR_RHOS, (-2.0f - 0.41f) / 3.0f), (4.0f * 720.0f));
        r.nsagg = cons15 * asn * pow(rhoa, (2.0f + 0.41f) / 3.0f)
                  * pow(*qs, (2.0f + 0.41f) / 3.0f)
                  * pow(*ns * rhoa, (4.0f - 0.41f) / 3.0f) / rhoa;
    }
    float cons13 = soft_gfk_tgammaf(3.41f) * MORR_MPI / 4.0f * 0.7f;
    float cons14 = soft_gfk_tgammaf(3.0f + morr_bg) * MORR_MPI / 4.0f * 0.7f;
    float cons16 = soft_gfk_tgammaf(4.0f) * MORR_MPI / 4.0f * 0.7f;
    if (*qs >= 1.0e-8f && *qc >= MORR_QSMALL) {
        r.psacws = cons13 * asn * *qc * rhoa * n0s / pow(m.ls, 3.41f);
        r.npsacws = cons13 * asn * *nc * rhoa * n0s / pow(m.ls, 3.41f);
    }
    if (*qg >= 1.0e-8f && *qc >= MORR_QSMALL) {
        r.psacwg = cons14 * agn * *qc * rhoa * n0g
                   / pow(m.lg, 3.0f + morr_bg);
        r.npsacwg = cons14 * agn * *nc * rhoa * n0g
                    / pow(m.lg, 3.0f + morr_bg);
    }
    if (*qi >= 1.0e-8f && *qc >= MORR_QSMALL && 1.0f / m.li >= 100.0e-6f) {
        r.psacwi = cons16 * ain * *qc * rhoa * n0i / pow(m.li, 4.0f);
        r.npsacwi = cons16 * ain * *nc * rhoa * n0i / pow(m.li, 4.0f);
    }
    if (*qr >= 1.0e-8f && *qs >= 1.0e-8f) {
        float ums = fmin(asn * soft_gfk_tgammaf(4.41f) / (6.0f * pow(m.ls, 0.41f)),
                         1.2f * dens54);
        float uns = fmin(asn * soft_gfk_tgammaf(1.41f) / pow(m.ls, 0.41f),
                         1.2f * dens54);
        float umr = fmin(arn * soft_gfk_tgammaf(4.8f) / (6.0f * pow(m.lr, 0.8f)),
                         9.1f * dens54);
        float unr = fmin(arn * soft_gfk_tgammaf(1.8f) / pow(m.lr, 0.8f),
                         9.1f * dens54);
        float vrelm = sqrt((1.2f * umr - 0.95f * ums)
                           * (1.2f * umr - 0.95f * ums) + 0.08f * ums * umr);
        float vreln = sqrt(1.7f * (unr - uns) * (unr - uns) + 0.3f * unr * uns);
        r.pracs = fmin(MORR_MPI * MORR_MPI * MORR_RHOW * vrelm * rhoa * n0r * n0s
                        / pow(m.lr, 3.0f)
                        * (5.0f / (pow(m.lr, 3.0f) * m.ls)
                           + 2.0f / (m.lr * m.lr * m.ls * m.ls)
                           + 0.5f / (m.lr * pow(m.ls, 3.0f))), *qr / dt);
        r.npracs = MORR_MPI / 2.0f * rhoa * vreln * n0r * n0s
                   * (1.0f / (pow(m.lr, 3.0f) * m.ls)
                      + 1.0f / (m.lr * m.lr * m.ls * m.ls)
                      + 1.0f / (m.lr * pow(m.ls, 3.0f)));
        if (*qs >= 0.1e-3f && *qr >= 0.1e-3f) {
            r.psacr = MORR_MPI * MORR_MPI * MORR_RHOS * vrelm * rhoa * n0r * n0s
                      / pow(m.ls, 3.0f)
                      * (5.0f / (pow(m.ls, 3.0f) * m.lr)
                         + 2.0f / (m.ls * m.ls * m.lr * m.lr)
                         + 0.5f / (m.ls * pow(m.lr, 3.0f)));
        }
    }
    if (*qr >= 1.0e-8f && *qg >= 1.0e-8f) {
        float umg = fmin(agn * soft_gfk_tgammaf(4.0f + morr_bg)
                         / (6.0f * pow(m.lg, morr_bg)),
                         20.0f * dens54);
        float ung = fmin(agn * soft_gfk_tgammaf(1.0f + morr_bg)
                         / pow(m.lg, morr_bg),
                         20.0f * dens54);
        float umr = fmin(arn * soft_gfk_tgammaf(4.8f) / (6.0f * pow(m.lr, 0.8f)),
                         9.1f * dens54);
        float unr = fmin(arn * soft_gfk_tgammaf(1.8f) / pow(m.lr, 0.8f),
                         9.1f * dens54);
        float vrelm = sqrt((1.2f * umr - 0.95f * umg)
                           * (1.2f * umr - 0.95f * umg) + 0.08f * umg * umr);
        float vreln = sqrt(1.7f * (unr - ung) * (unr - ung) + 0.3f * unr * ung);
        r.pracg = fmin(MORR_MPI * MORR_MPI * MORR_RHOW * vrelm * rhoa * n0r * n0g
                        / pow(m.lr, 3.0f)
                        * (5.0f / (pow(m.lr, 3.0f) * m.lg)
                           + 2.0f / (m.lr * m.lr * m.lg * m.lg)
                           + 0.5f / (m.lr * pow(m.lg, 3.0f))), *qr / dt);
        r.npracg = MORR_MPI / 2.0f * rhoa * vreln * n0r * n0g
                   * (1.0f / (pow(m.lr, 3.0f) * m.lg)
                      + 1.0f / (m.lr * m.lr * m.lg * m.lg)
                      + 1.0f / (m.lr * pow(m.lg, 3.0f)));
    }

    // Hallett-Mossop rate subtraction, WRF 2601-2713.
    if (*temp > 265.16f && *temp < 270.16f) {
        float fmult = *temp > 268.16f ? (270.16f - *temp) / 2.0f
                                     : __fdiv_rn((*temp - 265.16f), 3.0f);
        float mmult = 4.0f / 3.0f * MORR_MPI * MORR_RHOI * 1.25e-16f;
        if (*qs >= 0.1e-3f && (*qc >= 0.5e-3f || *qr >= 0.1e-3f)) {
            if (r.psacws > 0.0f) {
                r.nmults = 35.0e4f * r.psacws * fmult * 1000.0f;
                r.qmults = fmin(r.nmults * mmult, r.psacws);
                r.psacws -= r.qmults;
            }
            if (r.pracs > 0.0f) {
                r.nmultr = 35.0e4f * r.pracs * fmult * 1000.0f;
                r.qmultr = fmin(r.nmultr * mmult, r.pracs);
                r.pracs -= r.qmultr;
            }
        }
        if (*qg >= 0.1e-3f && (*qc >= 0.5e-3f || *qr >= 0.1e-3f)) {
            if (r.psacwg > 0.0f) {
                r.nmultg = 35.0e4f * r.psacwg * fmult * 1000.0f;
                r.qmultg = fmin(r.nmultg * mmult, r.psacwg);
                r.psacwg -= r.qmultg;
            }
            if (r.pracg > 0.0f) {
                r.nmultrg = 35.0e4f * r.pracg * fmult * 1000.0f;
                r.qmultrg = fmin(r.nmultrg * mmult, r.pracg);
                r.pracg -= r.qmultrg;
            }
        }
    }
    // Ordered snow-to-graupel redirects, WRF 2719-2763.
    if (r.psacws > 0.0f && *qs >= 0.1e-3f && *qc >= 0.5e-3f) {
        float cons17 = 3.0f * MORR_RHOSU * MORR_MPI * 0.7f * 0.7f * soft_gfk_tgammaf(2.82f)
                      / (morr_rhog - MORR_RHOS);
        r.pgsacw = fmin(r.psacws, cons17 * dt * n0s * *qc * *qc * asn * asn
                         / (rhoa * pow(m.ls, 2.82f)));
        float embryo = fmax(MORR_RHOS / (morr_rhog - MORR_RHOS) * r.pgsacw, 0.0f);
        r.nscng = fmin(__fdiv_rn(embryo, MORR_MMG0) * rhoa, __fdiv_rn(*ns, dt));
        r.psacws -= r.pgsacw;
    }
    if (r.pracs > 0.0f && *qs >= 0.1e-3f && *qr >= 0.1e-3f) {
        float snow6 = MORR_RHOS * MORR_RHOS * pow(4.0f / m.ls, 6.0f);
        float rain6 = MORR_RHOW * MORR_RHOW * pow(4.0f / m.lr, 6.0f);
        float fs = fmin(fmax(snow6 / (snow6 + rain6), 0.0f), 1.0f);
        r.pgracs = (1.0f - fs) * r.pracs;
        r.ngracs = fmin((1.0f - fs) * r.npracs,
                         fmin(*nr / dt, *ns / dt));
        r.pracs -= r.pgracs; r.npracs -= r.ngracs; r.psacr *= 1.0f - fs;
    }

    if (*temp < 269.15f && *qr >= MORR_QSMALL) {
        float bigg = exp(0.66f * (273.15f - *temp)) - 1.0f;
        r.mnuccr = 20.0f * MORR_MPI * MORR_MPI * MORR_RHOW * 100.0f * *nr * bigg
                   / pow(m.lr, 6.0f);
        r.nnuccr = fmin(MORR_MPI * *nr * 100.0f * bigg / pow(m.lr, 3.0f),
                         *nr / dt);
    }
    if (*qi >= 1.0e-8f && *qv / qvi >= 1.0f) {
        // PRCI forms from the UNCAPPED NPRCI; the NI3D/DT limit applies to
        // the number rate only afterwards (F:2833-2836 statement order).
        r.nprci = 4.0f / (MORR_DCS * MORR_RHOI) * (*qv - qvi) * rhoa * n0i
                   * exp(-m.li * MORR_DCS) * dv / abi;
        r.prci = __fdiv_rn(MORR_MPI * MORR_RHOI * pow(MORR_DCS, 3.0f), 6.0f) * r.nprci;
        r.nprci = fmin(r.nprci, *ni / dt);
    }
    if (*qs >= 1.0e-8f && *qi >= MORR_QSMALL) {
        float cons23 = MORR_MPI / 4.0f * 0.1f * soft_gfk_tgammaf(3.41f);
        r.prai = cons23 * asn * *qi * rhoa * n0s / pow(m.ls, 3.41f);
        r.nprai = fmin(cons23 * asn * *ni * rhoa * n0s
                        / pow(m.ls, 3.41f), *ni / dt);
    }
    if (*qr >= 1.0e-8f && *qi >= 1.0e-8f) {
        float cons24 = MORR_MPI / 4.0f * soft_gfk_tgammaf(3.8f);
        float cons25 = MORR_MPI * MORR_MPI / 24.0f * MORR_RHOW * soft_gfk_tgammaf(6.8f);
        float niacr = fmin(cons24 * *ni * n0r * arn / pow(m.lr, 3.8f)
                           * rhoa, fmin(*nr / dt, *ni / dt));
        float piacr = cons25 * *ni * n0r * arn / pow(m.lr, 6.8f) * rhoa;
        float praci = cons24 * *qi * n0r * arn / pow(m.lr, 3.8f) * rhoa;
        if (*qr >= 0.1e-3f) {
            r.niacr = niacr; r.piacr = piacr; r.praci = praci;
        } else {
            r.niacrs = niacr; r.piacrs = piacr; r.pracis = praci;
        }
    }
    if ((*qv / qvs >= 0.999f && *temp <= 265.15f) || *qv / qvi >= 1.08f) {
        float target = fmin(0.005f * exp(0.304f * (273.15f - *temp))
                            * 1000.0f, 500.0e3f) / rhoa;
        if (target > *ni + *ns + *ng) {
            r.nnuccd = (target - *ni - *ns - *ng) / dt;
            r.mnuccd = r.nnuccd * MORR_MMI0;
            // Deposition nucleation may not consume vapor that is not
            // there.  Below ~159 K POLYSVP's extrapolated liquid curve
            // drops under its ice curve, EIS = min(EW, ...) makes
            // qvi == qvs, and the 0.999*QVS trigger fires with qv AT OR
            // BELOW ice saturation; the FUDGEF rescale below skips the
            // dum <= 0 / sum_dep > 0 sign pair, so the unbounded MNUCCD
            // drove a polar-night model-top level to qv = -1.6e-4 kg/kg
            // (T255 native-suite abort at hour 49 of a 384 h run,
            // 2026-09-01).  Bound the nucleated mass by the vapor excess
            // over ice saturation and scale the number moment with it:
            // both sides of the process shrink together, so no crystal
            // count appears whose seed mass never existed.  Documented
            // divergence: module_mp_morr_two_moment.F carries the same
            // one-sided limiter but has no 156 K levels to reach it with.
            // The bound also engages above the trap, in cold first
            // nucleation whose excess is under the Cooper embryo mass
            // (the 1.08 ice-supersaturation arm near 200 K, a reachable
            // WRF tropopause state): there the dum > 0 FUDGEF branch
            // already caps the MASS in both models, so qv and qi match
            // WRF and only the number moment shrinks -- crystals keep
            // MI0 seed mass instead of WRF's mass-starved count.
            float avail = fmax(*qv - qvi, 0.0f) / dt;
            if (r.mnuccd > avail) {
                r.nnuccd *= avail / r.mnuccd;
                r.mnuccd = avail;
            }
        }
    }

    // Harrington deposition tail and the bidirectional collective FUDGEF
    // rescale.  NNUCCD is deliberately not rescaled (WRF 2971-3030).
    float epsi = *qi >= MORR_QSMALL
                ? 2.0f * MORR_MPI * n0i * rhoa * dv / (m.li * m.li) : 0.0f;
    float epss = *qs >= MORR_QSMALL
                ? 2.0f * MORR_MPI * n0s * rhoa * dv
                  * (0.86f / (m.ls * m.ls)
                     + 0.28f * sqrt(asn * rhoa / mu) * morr_msl_cbrtf_positive(sc)
                     * soft_gfk_tgammaf(2.705f) / pow(m.ls, 2.705f)) : 0.0f;
    float epsg = *qg >= MORR_QSMALL
                ? 2.0f * MORR_MPI * n0g * rhoa * dv
                  * (0.86f / (m.lg * m.lg)
                     + 0.28f * sqrt(agn * rhoa / mu) * morr_msl_cbrtf_positive(sc)
                     * soft_gfk_tgammaf(2.5f + 0.5f * morr_bg)
                     / pow(m.lg, 2.5f + 0.5f * morr_bg)) : 0.0f;
    float dep = (*qv - qvi) / abi;
    float tail = *qi >= MORR_QSMALL
                ? 1.0f - exp(-m.li * MORR_DCS) * (1.0f + m.li * MORR_DCS) : 0.0f;
    r.prd = epsi * dep * tail;
    if (*qs >= MORR_QSMALL) r.prds = epss * dep + epsi * dep * (1.0f - tail);
    else r.prd += epsi * dep * (1.0f - tail);
    r.prdg = epsg * dep;
    float sum_dep = r.prd + r.prds + r.prdg + r.mnuccd;
    float dum = (*qv - qvi) / dt;
    if (sum_dep != 0.0f
            && ((dum > 0.0f && sum_dep > dum * 0.9999f)
                || (dum < 0.0f && sum_dep < dum * 0.9999f))) {
        float ratio = 0.9999f * dum / sum_dep;
        r.prd *= ratio; r.prds *= ratio; r.prdg *= ratio; r.mnuccd *= ratio;
    }
    if (r.prd < 0.0f) { r.eprd = r.prd; r.prd = 0.0f; }
    if (r.prds < 0.0f) { r.eprds = r.prds; r.prds = 0.0f; }
    if (r.prdg < 0.0f) { r.eprdg = r.prdg; r.prdg = 0.0f; }

    // WRF 3086-3187 joint mass-only donor limiting.
    morr_limit_cold_cloud(r, *qc, dt, cloud_freezing);
    float loss = (-r.prd - r.mnuccc + r.prci + r.prai - r.qmults - r.qmultg
            - r.qmultr - r.qmultrg - r.mnuccd + r.praci + r.pracis
            - r.eprd - r.psacwi) * dt;
    if (loss > *qi && *qi >= MORR_QSMALL) {
        float ratio = (*qi / dt + r.prd + r.mnuccc + r.qmults + r.qmultg
                      + r.qmultr + r.qmultrg + r.mnuccd + r.psacwi)
                     / (r.prci + r.prai + r.praci + r.pracis - r.eprd);
        r.prci *= ratio; r.prai *= ratio; r.praci *= ratio;
        r.pracis *= ratio; r.eprd *= ratio;
    }
    morr_limit_cold_rain(r, *qr, *nr, m.lr, *temp, dt);
    loss = (-r.prds - r.psacws - r.prai - r.prci - r.pracs - r.eprds
            + r.psacr - r.piacrs - r.pracis) * dt;
    if (loss > *qs && *qs >= MORR_QSMALL) {
        float ratio = (*qs / dt + r.prds + r.psacws + r.prai + r.prci
                      + r.pracs + r.piacrs + r.pracis)
                     / (-r.eprds + r.psacr);
        r.eprds *= ratio; r.psacr *= ratio;
    }
    loss = (-r.psacwg - r.pracg - r.pgsacw - r.pgracs - r.prdg
            - r.mnuccr - r.eprdg - r.piacr - r.praci - r.psacr) * dt;
    if (loss > *qg && *qg >= MORR_QSMALL) {
        float ratio = (*qg / dt + r.psacwg + r.pracg + r.pgsacw + r.pgracs
                      + r.prdg + r.mnuccr + r.psacr + r.piacr + r.praci)
                     / (-r.eprdg);
        r.eprdg *= ratio;
    }

    float tqv = -r.pre - r.prd - r.prds - r.mnuccd - r.eprd - r.eprds
               - r.prdg - r.eprdg;
    float tt = (r.pre * xlv
               + (r.prd + r.prds + r.mnuccd + r.eprd + r.eprds
                  + r.prdg + r.eprdg) * xls
               + (r.psacws + r.psacwi + r.mnuccc + r.mnuccr + r.qmults
                  + r.qmultg + r.qmultr + r.qmultrg + r.pracs + r.psacwg
                  + r.pracg + r.pgsacw + r.pgracs + r.piacr + r.piacrs)
                 * xlf) / cpm;
    float tqc = -r.pra - r.prc - r.mnuccc - r.psacws - r.psacwi
               - r.qmults - r.qmultg - r.psacwg - r.pgsacw;
    float tqi = r.prd + r.eprd + r.psacwi + r.mnuccc - r.prci - r.prai
               + r.qmults + r.qmultg + r.qmultr + r.qmultrg + r.mnuccd
               - r.praci - r.pracis;
    float tqr = r.pre + r.pra + r.prc - r.pracs - r.mnuccr - r.qmultr
               - r.qmultrg - r.piacr - r.piacrs - r.pracg - r.pgracs;
    float tqs = r.prai + r.psacws + r.prds + r.pracs + r.prci + r.eprds
               - r.psacr + r.piacrs + r.pracis;
    float tqg = r.pracg + r.psacwg + r.pgsacw + r.pgracs + r.prdg
               + r.eprdg + r.mnuccr + r.piacr + r.praci + r.psacr;
    float tnc = -r.nnuccc - r.npsacws - r.npra - r.nprc
               - r.npsacwi - r.npsacwg;
    float tni = r.nnuccc - r.nprci - r.nprai + r.nmults + r.nmultg
               + r.nmultr + r.nmultrg + r.nnuccd - r.niacr - r.niacrs;
    float tnr = r.nprc1 - r.npracs - r.nnuccr + r.nragg - r.niacr
               - r.niacrs - r.npracg - r.ngracs;
    float tns = r.nsagg + r.nprci - r.nscng - r.ngracs + r.niacrs;
    float tng = r.nscng + r.ngracs + r.nnuccr + r.niacr;
    if (r.eprd < 0.0f) r.nsubi = fmax(-1.0f, r.eprd * dt / *qi) * *ni / dt;
    if (r.eprds < 0.0f) r.nsubs = fmax(-1.0f, r.eprds * dt / *qs) * *ns / dt;
    if (r.pre < 0.0f) r.nsubr = fmax(-1.0f, r.pre * dt / *qr) * *nr / dt;
    if (r.eprdg < 0.0f) r.nsubg = fmax(-1.0f, r.eprdg * dt / *qg) * *ng / dt;
    tni += r.nsubi; tns += r.nsubs; tnr += r.nsubr; tng += r.nsubg;

    // One collective update, including saturation adjustment diagnosed from
    // the collectively predicted state (WRF 3191-3267).
    float tpred = *temp + dt * tt, qvpred = *qv + dt * tqv;
    float qcpred = fmax(*qc + dt * tqc, 0.0f);
    float ew = fmin(0.99f * pressure, morr_polysvp(tpred, false));
    float qss = MORR_EP2 * ew / (pressure - ew);
    float pcc = (qvpred - qss)
               / (1.0f + xlv * xlv * qss / (cpm * MORR_RV * tpred * tpred)) / dt;
    if (pcc * dt + qcpred < 0.0f) pcc = -qcpred / dt;
    tqv -= pcc; tqc += pcc; tt += pcc * xlv / cpm;
    *qv += dt * tqv; *qc += dt * tqc; *qr += dt * tqr;
    *qi += dt * tqi; *qs += dt * tqs; *qg += dt * tqg;
    *nc += dt * tnc; *nr += dt * tnr; *ni += dt * tni;
    *ns += dt * tns; *ng += dt * tng; *temp += dt * tt;
}

kernel void morrison_process_levels(
    constant MorrisonProcessLevelsResources &r [[buffer(0)]],
    constant MorrisonProcessLevelsParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.ncell)) return;
    const uint ncol=uint(p.ny)*uint(p.nx), k=idx/ncol, rem=idx-k*ncol;
    const uint j=rem/uint(p.nx), i=rem-j*uint(p.nx);
    if(j<uint(p.j_start)||j>=uint(p.j_stop)
       ||i<uint(p.i_start)||i>=uint(p.i_stop)) return;
    const float piik=r.pii[idx], pres=r.pressure[idx];
    float temp=r.theta[idx]*piik;
    const float rhoa=pres/(MORR_RD*temp);
    float qvk=fmax(r.qv[idx],0.0f),qck=fmax(r.qc[idx],0.0f);
    float qrk=fmax(r.qr[idx],0.0f),qik=fmax(r.qi[idx],0.0f);
    float qsk=fmax(r.qs[idx],0.0f),qgk=fmax(r.qg[idx],0.0f);
    float nck=fmax(r.nc[idx],0.0f),nrk=fmax(r.nr[idx],0.0f);
    float nik=fmax(r.ni[idx],0.0f),nsk=fmax(r.ns[idx],0.0f),ngk=fmax(r.ng[idx],0.0f);
    const float xlv=3.1484e6f-2370.0f*temp;
    const float xls=3.15e6f-2370.0f*temp+0.3337e6f;
    const float cpm=MORR_CP*(1.0f+0.887f*qvk);
    r.effc[idx]=xlv; r.effi[idx]=cpm;
    const float ew=fmin(0.99f*pres,morr_polysvp(temp,false));
    const float ei=fmin(ew,fmin(0.99f*pres,morr_polysvp(temp,true)));
    const float qvs=MORR_EP2*ew/(pres-ew),qvi=MORR_EP2*ei/(pres-ei);
    if(p.has_cu_tendencies) {
        const float qrcu=r.qrcuten[idx];
        if(qrcu>=1.0e-10f) nrk+=1.8e5f*pow(qrcu*p.dt/(MORR_MPI*MORR_RHOW*rhoa*rhoa*rhoa),0.25f);
        const float qscu=r.qscuten[idx];
        if(qscu>=1.0e-10f) nsk+=3.0e5f*pow(qscu*p.dt/(100.0f*MORR_MPI*rhoa*rhoa*rhoa),0.25f);
        const float qicu=r.qicuten[idx];
        if(qicu>=1.0e-10f) nik+=__fdiv_rn(qicu*p.dt,(MORR_CI*80.0e-6f*80.0e-6f*80.0e-6f));
    }
    if(qvk/qvs<0.9f) {
        if(qrk<1.0e-8f){qvk+=qrk;temp-=qrk*xlv/cpm;qrk=0.0f;}
        if(qck<1.0e-8f){qvk+=qck;temp-=qck*xlv/cpm;qck=0.0f;}
    }
    if(qvk/qvi<0.9f) {
        if(qik<1.0e-8f){qvk+=qik;temp-=qik*xls/cpm;qik=0.0f;}
        if(qsk<1.0e-8f){qvk+=qsk;temp-=qsk*xls/cpm;qsk=0.0f;}
        if(qgk<1.0e-8f){qvk+=qgk;temp-=qgk*xls/cpm;qgk=0.0f;}
    }
    if(qck<MORR_QSMALL){qck=0.0f;nck=0.0f;}
    if(qrk<MORR_QSMALL){qrk=0.0f;nrk=0.0f;}
    if(qik<MORR_QSMALL){qik=0.0f;nik=0.0f;}
    if(qsk<MORR_QSMALL){qsk=0.0f;nsk=0.0f;}
    if(qgk<MORR_QSMALL){qgk=0.0f;ngk=0.0f;}
    const bool warm=temp>=273.15f;
    const float qi_begin=qik;
    float stale_lami=0.0f,cloud_nc=0.0f;
    morr_process_level(&qvk,&qck,&qrk,&qik,&qsk,&qgk,&nck,&nrk,&nik,&nsk,&ngk,
        &temp,pres,rhoa,p.dt,qvs,qvi,xlv,xls,cpm,warm,&stale_lami,&cloud_nc,
        p.morr_ag,p.morr_bg,p.morr_rhog);
    const bool ice_to_snow=(!warm&&qi_begin>=MORR_QSMALL&&stale_lami>=1.0e-10f
                             &&1.0f/stale_lami>=2.0f*MORR_DCS);
    r.rho_in[idx]=rhoa;
    r.ice_to_snow_scratch[idx]=ice_to_snow?1.0f:0.0f;
    r.effs[idx]=cloud_nc;
    r.theta[idx]=temp/piik;
    r.qv[idx]=qvk;r.qc[idx]=qck;r.qr[idx]=qrk;r.qi[idx]=qik;r.qs[idx]=qsk;r.qg[idx]=qgk;
    r.nc[idx]=nck;r.nr[idx]=nrk;r.ni[idx]=nik;r.ns[idx]=nsk;r.ng[idx]=ngk;
}

kernel void morrison_prepare_fields(
    constant MorrisonPrepareFieldsResources &r [[buffer(0)]],
    constant MorrisonPrepareFieldsParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx>=uint(p.ncell)) return;
    const uint ncol=uint(p.ny)*uint(p.nx), k=idx/ncol;
    const uint j=idx%ncol/uint(p.nx), i=idx%uint(p.nx);
    if(j<uint(p.j_start)||j>=uint(p.j_stop)
       ||i<uint(p.i_start)||i>=uint(p.i_stop)) {
        // WRF's clipped moist-physics finish never writes ring heating.
        r.h_diabatic[idx]=0.0f;
        return;
    }
    const uint above=idx+ncol;
    r.theta[idx]=r.thb[p.thb_full?idx:k]+r.thp[idx];
    r.pii[idx]=pow(r.pressure[idx]/p.p0,p.rcp);
    const float zb=(r.phb[p.phb_full?idx:k]+r.php[idx])/p.gravity;
    const float za=(r.phb[p.phb_full?above:k+1]+r.php[above])/p.gravity;
    r.dz[idx]=za-zb;
    r.h_diabatic[idx]=r.thb[p.thb_full?idx:k]+r.thp[idx];
}

kernel void morrison_finish(
    constant MorrisonFinishResources &r [[buffer(0)]],
    constant MorrisonFinishParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if(idx>=uint(p.ncell))return;
    const uint ncol=uint(p.ny)*uint(p.nx);
    const uint j=idx%ncol/uint(p.nx), i=idx%uint(p.nx);
    if(j<uint(p.j_start)||j>=uint(p.j_stop)
       ||i<uint(p.i_start)||i>=uint(p.i_stop)) {
        r.h_diabatic[idx]=0.0f;
        return;
    }
    if(p.no_mp_heating){r.h_diabatic[idx]=0.0f;return;}
    float mpten=r.theta[idx]-r.h_diabatic[idx];
    mpten=fmin(p.limit,mpten);mpten=fmax(-p.limit,mpten);
    r.thp[idx]+=mpten;
    r.h_diabatic[idx]=mpten/p.dt;
}


inline ulong morr_index(int k,int j,int i,int ny,int nx) {
    return (ulong(k)*ulong(ny)+ulong(j))*ulong(nx)+ulong(i);
}

inline void morr_sediment_column(
 device float *qc,device float *qr,device float *qi,device float *qs,device float *qg,
 device float *nc,device float *nr,device float *ni,device float *ns,device float *ng,
 device const float *cloud_nc,device const float *theta,device const float *pii,
 device const float *pressure,device const float *rho_in,device const float *dz,
 device float *rainnc,device float *rainncv,device float *snownc,device float *snowncv,
 device float *graupelnc,device float *graupelncv,device float *sr,
 float dt,float ag,float bg,float rhog,int nz,int ny,int nx,uint col) {
    const int j=int(col/uint(nx)), i=int(col-uint(j)*uint(nx));
    float qd[256],nd[256],nd0[256],vm[256],vn[256];
    float max_courant=0.0f;
    // CUDA reduces the largest category/level Courant number before any
    // category is advanced. Recompute speeds in the transport pass below.
    for(int kind=0;kind<5;++kind){
        device float *mass=kind==0?qc:(kind==1?qr:(kind==2?qi:(kind==3?qs:qg)));
        device float *number=kind==0?nc:(kind==1?nr:(kind==2?ni:(kind==3?ns:ng)));
        for(int k=nz-1;k>=0;--k){
            ulong x=morr_index(k,j,i,ny,nx); float temp=theta[x]*pii[x];
            float rhoa=fabs(rho_in[x]), q=fmax(mass[x],0.0f), nn, v0,v1;
            float nsed=kind==0?cloud_nc[x]:number[x];
            morr_terminal_velocity(kind,q,nsed,rhoa,temp,nn,v0,v1,ag,bg,rhog);
            if(k<nz-1){if(v0<1.0e-10f)v0=vm[k+1];if(v1<1.0e-10f)v1=vn[k+1];}
            vm[k]=v0;vn[k]=v1;
            max_courant=fmax(max_courant,fmax(v0,v1)*dt/dz[x]);
        }
    }
    const int nstep=max(int(max_courant+1.0f),1);
    const float dts=dt/float(nstep);
    float exported[5]={0.0f,0.0f,0.0f,0.0f,0.0f};
    for(int kind=0;kind<5;++kind){
        device float *mass=kind==0?qc:(kind==1?qr:(kind==2?qi:(kind==3?qs:qg)));
        device float *number=kind==0?nc:(kind==1?nr:(kind==2?ni:(kind==3?ns:ng)));
        int ktop=-1;
        for(int k=nz-1;k>=0;--k){
            ulong x=morr_index(k,j,i,ny,nx); float temp=theta[x]*pii[x];
            float rhoa=fabs(rho_in[x]),q=fmax(mass[x],0.0f),nn;
            float nsed=kind==0?cloud_nc[x]:number[x];
            morr_terminal_velocity(kind,q,nsed,rhoa,temp,nn,vm[k],vn[k],ag,bg,rhog);
            if(k<nz-1){if(vm[k]<1.0e-10f)vm[k]=vm[k+1];if(vn[k]<1.0e-10f)vn[k]=vn[k+1];}
            if(ktop<0&&(vm[k]!=0.0f||vn[k]!=0.0f))ktop=k;
            qd[k]=q*rhoa;nd[k]=fmax(nsed,0.0f)*rhoa;nd0[k]=nd[k];
        }
        for(int sub=0;ktop>=0&&sub<nstep;++sub){
            float fm_above=0.0f,fn_above=0.0f;
            for(int k=ktop;k>=0;--k){
                ulong x=morr_index(k,j,i,ny,nx);
                float fm_here=vm[k]*qd[k],fn_here=vn[k]*nd[k];
                if(k==0)exported[kind]+=fm_here*dts;
                qd[k]+=(fm_above-fm_here)*dts/dz[x];
                nd[k]+=(fn_above-fn_here)*dts/dz[x];
                fm_above=fm_here;fn_above=fn_here;
            }
        }
        for(int k=0;k<nz;++k){
            ulong x=morr_index(k,j,i,ny,nx);float rhoa=rho_in[x];
            mass[x]=fmax(qd[k]/rhoa,0.0f);float sedimented=fmax(nd[k]/rhoa,0.0f);
            if(kind==0)number[x]=fmax(number[x]+(nd[k]-nd0[k])/rhoa,0.0f);
            else number[x]=sedimented;
        }
    }
    ulong s=ulong(j)*ulong(nx)+ulong(i);
    float total=exported[0]+exported[1]+exported[2]+exported[3]+exported[4];
    float snow=exported[2]+exported[3];
    rainncv[s]=total;snowncv[s]=snow;graupelncv[s]=exported[4];
    rainnc[s]+=total;snownc[s]+=snow;graupelnc[s]+=exported[4];
    sr[s]=(snow+exported[4])/(total+1.0e-12f);
}

#define MORR_SEDIMENT_PASS morr_sediment_column( \
 r.qc,r.qr,r.qi,r.qs,r.qg,r.nc,r.nr,r.ni,r.ns,r.ng,r.cloud_nc,r.theta,r.pii, \
 r.pressure,r.rho_in,r.dz,r.rainnc,r.rainncv,r.snownc,r.snowncv, \
 r.graupelnc,r.graupelncv,r.sr,p.dt,p.morr_ag,p.morr_bg,p.morr_rhog, \
 p.nz,p.ny,p.nx,flat)

kernel void morrison_sediment_64(constant MorrisonSediment64Resources &r [[buffer(0)]],
 constant MorrisonSediment64Params &p [[buffer(1)]],uint flat [[thread_position_in_grid]]) {
    uint count=uint(p.ny)*uint(p.nx);if(flat>=count)return;
    uint j=flat/uint(p.nx),i=flat-j*uint(p.nx);
    if(j<uint(p.j_start)||j>=uint(p.j_stop)
       ||i<uint(p.i_start)||i>=uint(p.i_stop))return;
    MORR_SEDIMENT_PASS;
}
kernel void morrison_sediment_256(constant MorrisonSediment256Resources &r [[buffer(0)]],
 constant MorrisonSediment256Params &p [[buffer(1)]],uint flat [[thread_position_in_grid]]) {
    uint count=uint(p.ny)*uint(p.nx);if(flat>=count)return;
    uint j=flat/uint(p.nx),i=flat-j*uint(p.nx);
    if(j<uint(p.j_start)||j>=uint(p.j_stop)
       ||i<uint(p.i_start)||i>=uint(p.i_stop))return;
    MORR_SEDIMENT_PASS;
}
#undef MORR_SEDIMENT_PASS

inline void morrison_finalize_column(
 device float *theta,device float *qv,device float *qc,device float *qr,
 device float *qi,device float *qs,device float *qg,device float *nc,
 device float *nr,device float *ni,device float *ns,device float *ng,
 device const float *rho_in,device const float *pii,device const float *pressure,
 device const float *ice_to_snow_scratch,device float *effc,device float *effi,
 device float *effs,device float *effr,float morr_rhog,uint idx) {
    float temp=theta[idx]*pii[idx];bool ice_to_snow=ice_to_snow_scratch[idx]!=0.0f;
    float rhoa=rho_in[idx],qvk=qv[idx],qck=qc[idx],qrk=qr[idx],qik=qi[idx],qsk=qs[idx],qgk=qg[idx];
    if(qvk>0.0f&&qck==0.0f&&qrk==0.0f&&qik==0.0f&&qsk==0.0f&&qgk==0.0f){
        qc[idx]=qr[idx]=qi[idx]=qs[idx]=qg[idx]=0.0f;nr[idx]=ns[idx]=ng[idx]=0.0f;
        ni[idx]=fmin(0.0f,0.3e6f/rhoa);nc[idx]=250.0e6f/rhoa;
        effc[idx]=effr[idx]=effi[idx]=effs[idx]=25.0f;theta[idx]=temp/pii[idx];return;
    }
    float nck=nc[idx],nrk=nr[idx],nik=ni[idx],nsk=ns[idx],ngk=ng[idx];
    if(ice_to_snow&&qik>0.0f){qsk+=qik;nsk+=nik;qik=0.0f;nik=0.0f;}
    float xlv=effc[idx],xls=xlv+0.3353e6f,xlf=xls-xlv,cpm=effi[idx];
    float ew=fmin(0.99f*pressure[idx],morr_polysvp(temp,false));
    float ei=fmin(ew,fmin(0.99f*pressure[idx],morr_polysvp(temp,true)));
    float qvs=MORR_EP2*ew/(pressure[idx]-ew),qvi=MORR_EP2*ei/(pressure[idx]-ei);
    if(qvk/qvs<0.9f){if(qrk<1.0e-8f){qvk+=qrk;temp-=qrk*xlv/cpm;qrk=0.0f;}
        if(qck<1.0e-8f){qvk+=qck;temp-=qck*xlv/cpm;qck=0.0f;}}
    if(qvk/qvi<0.9f){if(qik<1.0e-8f){qvk+=qik;temp-=qik*xls/cpm;qik=0.0f;}
        if(qsk<1.0e-8f){qvk+=qsk;temp-=qsk*xls/cpm;qsk=0.0f;}
        if(qgk<1.0e-8f){qvk+=qgk;temp-=qgk*xls/cpm;qgk=0.0f;}}
    if(qck<MORR_QSMALL){qck=0.0f;nck=0.0f;}if(qrk<MORR_QSMALL){qrk=0.0f;nrk=0.0f;}
    if(qik<MORR_QSMALL){qik=0.0f;nik=0.0f;}if(qsk<MORR_QSMALL){qsk=0.0f;nsk=0.0f;}
    if(qgk<MORR_QSMALL){qgk=0.0f;ngk=0.0f;}
    if(qik>=MORR_QSMALL&&temp>=273.15f){qrk+=qik;nrk+=nik;temp-=qik*xlf/cpm;qik=0.0f;nik=0.0f;}
    if(temp<=233.15f&&qck>=MORR_QSMALL){qik+=qck;nik+=nck;temp+=qck*xlf/cpm;qck=0.0f;nck=0.0f;}
    if(temp<=233.15f&&qrk>=MORR_QSMALL){qgk+=qrk;ngk+=nrk;temp+=qrk*xlf/cpm;qrk=0.0f;nrk=0.0f;}
    qv[idx]=qvk;qc[idx]=qck;qr[idx]=qrk;qi[idx]=qik;qs[idx]=qsk;qg[idx]=qgk;
    nc[idx]=nck;nr[idx]=nrk;ni[idx]=nik;ns[idx]=nsk;ng[idx]=ngk;
    MorrMoments m=morr_bound(qc[idx],qr[idx],qi[idx],qs[idx],qg[idx],rhoa,
        pressure[idx],temp,nck,nrk,nik,nsk,ngk,false,morr_rhog);
    nc[idx]=nck;nr[idx]=nrk;ni[idx]=nik;ns[idx]=nsk;ng[idx]=ngk;
    effc[idx]=qc[idx]>=MORR_QSMALL?(m.pg+3.0f)/(2.0f*m.lc)*1.0e6f:25.0f;
    effr[idx]=qr[idx]>=MORR_QSMALL?1.5e6f/m.lr:25.0f;
    effi[idx]=qi[idx]>=MORR_QSMALL?1.5e6f/m.li:25.0f;
    effs[idx]=qs[idx]>=MORR_QSMALL?1.5e6f/m.ls:25.0f;
    ni[idx]=fmin(ni[idx],0.3e6f/rhoa);nc[idx]=250.0e6f/rhoa;theta[idx]=temp/pii[idx];
}

kernel void morrison_finalize_levels(constant MorrisonFinalizeLevelsResources &r [[buffer(0)]],
 constant MorrisonFinalizeLevelsParams &p [[buffer(1)]],uint idx [[thread_position_in_grid]]) {
    if(idx>=uint(p.ncell))return;
    const uint ncol=uint(p.ny)*uint(p.nx);
    const uint j=idx%ncol/uint(p.nx), i=idx%uint(p.nx);
    if(j<uint(p.j_start)||j>=uint(p.j_stop)
       ||i<uint(p.i_start)||i>=uint(p.i_stop)) return;
    morrison_finalize_column(r.theta,r.qv,r.qc,r.qr,r.qi,r.qs,r.qg,r.nc,r.nr,r.ni,r.ns,r.ng,
        r.rho_in,r.pii,r.pressure,r.ice_to_snow_scratch,r.effc,r.effi,r.effs,r.effr,p.morr_rhog,idx);
}

kernel void morrison_validate_diagnostics(
    constant MorrisonValidateDiagnosticsResources &r [[buffer(0)]],
    constant MorrisonValidateDiagnosticsParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if(idx>=uint(p.ncell))return;
    uint flags=0u;
    const float rainnc=r.rainnc[idx],rainncv=r.rainncv[idx];
    const float snownc=r.snownc[idx],snowncv=r.snowncv[idx];
    const float graupelnc=r.graupelnc[idx],graupelncv=r.graupelncv[idx];
    const float sr=r.sr[idx];
    if(!isfinite(rainnc))flags|=0x001u;
    if(!isfinite(rainncv))flags|=0x002u;
    if(!isfinite(snownc))flags|=0x004u;
    if(!isfinite(snowncv))flags|=0x008u;
    if(!isfinite(graupelnc))flags|=0x010u;
    if(!isfinite(graupelncv))flags|=0x020u;
    if(!isfinite(sr))flags|=0x040u;
    if(rainncv<0.0f)flags|=0x080u;
    if(sr<0.0f)flags|=0x100u;
    if(sr>p.sr_upper)flags|=0x200u;
    if(flags!=0u) {
        device atomic_uint *status=reinterpret_cast<device atomic_uint*>(r.status);
        atomic_fetch_or_explicit(status,flags,memory_order_relaxed);
    }
}
