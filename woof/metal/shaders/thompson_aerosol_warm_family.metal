#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "soft_binary64.metalh"
#include "soft_gfk_pow.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_log.metalh"
#include "soft_libm64.metalh"
#include "thompson_aerosol_warm_family_bindings.metalh"

using namespace metal;

static inline float thompson_aa_add(float a, float b)
{ return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float thompson_aa_sub(float a, float b)
{ return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float thompson_aa_mul(float a, float b)
{ return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float thompson_aa_div(float a, float b)
{ return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }
static inline ulong thompson_aa_widen(float x)
{ return soft64_from_f32_bits(as_type<uint>(x)); }
static inline float thompson_aa_narrow(ulong x)
{ return as_type<float>(soft64_to_f32_bits(x)); }
static inline ulong thompson_aa_pow64(ulong x, ulong y)
{ return soft64_plm_pow(x, y); }

constant float thompson_aa_cce2[16] = {
    0.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
    12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f};
constant float thompson_aa_ccg1[16] = {
    0.0f, 1.0f, 2.0f, 6.0f, 24.0f, 120.0000076f, 720.0000610f,
    5040.001953f, 40319.99609f, 362879.9688f, 3628801.750f,
    39916800.0f, 479001856.0f, 6227022336.0f, 87178297344.0f,
    1307673886720.0f};
constant float thompson_aa_ccg2[16] = {
    0.0f, 24.0f, 120.0000076f, 720.0000610f, 5040.001953f,
    40319.99609f, 362879.9688f, 3628801.750f, 39916800.0f,
    479001856.0f, 6227022336.0f, 87178297344.0f, 1307673886720.0f,
    20922782191616.0f, 355687448240128.0f, 6402383730776064.0f};
constant float thompson_aa_ocg1[16] = {
    0.0f, 1.0f, 0.5f, 0.1666666716f, 0.04166666791f,
    0.008333332837f, 0.001388888806f, 0.0001984126284f,
    0.00002480158946f, 0.000002755732112f, 0.0000002755730577f,
    0.00000002505210794f, 0.000000002087674478f,
    0.0000000001605904021f, 0.00000000001147074449f,
    0.0000000000007647166320f};
constant float thompson_aa_ocg2[16] = {
    0.0f, 0.04166666791f, 0.008333332837f, 0.001388888806f,
    0.0001984126284f, 0.00002480158946f, 0.000002755732112f,
    0.0000002755730577f, 0.00000002505210794f,
    0.000000002087674478f, 0.0000000001605904021f,
    0.00000000001147074449f, 0.0000000000007647166320f,
    0.00000000000004779478950f, 0.000000000000002811457147f,
    0.0000000000000001561918299f};


struct ThompsonAaD64 {
    ulong bits;
    ThompsonAaD64() : bits(0ul) {}
    ThompsonAaD64(float value) : bits(soft64_from_f32_bits(as_type<uint>(value))) {}
    ThompsonAaD64(int value) : bits(soft64_from_i32(value)) {}
    operator float() const { return as_type<float>(soft64_to_f32_bits(bits)); }
    static ThompsonAaD64 from_bits(ulong value) {
        ThompsonAaD64 result; result.bits = value; return result;
    }
};
static inline ThompsonAaD64 operator+(ThompsonAaD64 a, ThompsonAaD64 b)
{ return ThompsonAaD64::from_bits(soft64_add(a.bits, b.bits)); }
static inline ThompsonAaD64 operator-(ThompsonAaD64 a, ThompsonAaD64 b)
{ return ThompsonAaD64::from_bits(soft64_sub(a.bits, b.bits)); }
static inline ThompsonAaD64 operator*(ThompsonAaD64 a, ThompsonAaD64 b)
{ return ThompsonAaD64::from_bits(soft64_mul(a.bits, b.bits)); }
static inline ThompsonAaD64 operator/(ThompsonAaD64 a, ThompsonAaD64 b)
{ return ThompsonAaD64::from_bits(soft64_div(a.bits, b.bits)); }
static inline ThompsonAaD64 operator-(ThompsonAaD64 a)
{ return ThompsonAaD64::from_bits(a.bits ^ 0x8000000000000000ul); }
static inline bool operator<(ThompsonAaD64 a, ThompsonAaD64 b)
{ return soft64_less(a.bits, b.bits); }
static inline bool operator>(ThompsonAaD64 a, ThompsonAaD64 b)
{ return soft64_greater(a.bits, b.bits); }
static inline bool operator<=(ThompsonAaD64 a, ThompsonAaD64 b)
{ return !soft64_greater(a.bits, b.bits); }
static inline bool operator>=(ThompsonAaD64 a, ThompsonAaD64 b)
{ return !soft64_less(a.bits, b.bits); }
static inline ThompsonAaD64 thompson_aa_dmin(ThompsonAaD64 a, ThompsonAaD64 b)
{ return a < b ? a : b; }
static inline ThompsonAaD64 thompson_aa_dmax(ThompsonAaD64 a, ThompsonAaD64 b)
{ return a > b ? a : b; }
static inline ThompsonAaD64 thompson_aa_dmin_mode(ThompsonAaD64 a, ThompsonAaD64 b,
                                                  bool wrf39)
{
    if (!wrf39) return thompson_aa_dmin(a, b);
    if (soft64_is_nan(a.bits)) return b;
    if (soft64_is_nan(b.bits)) return a;
    if (soft64_is_zero(a.bits) && soft64_is_zero(b.bits))
        return ThompsonAaD64::from_bits((a.bits | b.bits) & 0x8000000000000000ul);
    return a < b ? a : b;
}
static inline ThompsonAaD64 thompson_aa_dmax_mode(ThompsonAaD64 a, ThompsonAaD64 b,
                                                  bool wrf39)
{
    if (!wrf39) return thompson_aa_dmax(a, b);
    if (soft64_is_nan(a.bits)) return b;
    if (soft64_is_nan(b.bits)) return a;
    if (soft64_is_zero(a.bits) && soft64_is_zero(b.bits))
        return ThompsonAaD64::from_bits((a.bits & b.bits) & 0x8000000000000000ul);
    return a > b ? a : b;
}
static inline ThompsonAaD64 thompson_aa_dpow(ThompsonAaD64 a, ThompsonAaD64 b)
{ return ThompsonAaD64::from_bits(soft64_plm_pow(a.bits, b.bits)); }
static inline float thompson_aa_powf(float a, float b)
{ return as_type<float>(soft_gfk_pow_bits(as_type<uint>(a), as_type<uint>(b))); }
static inline float thompson_aa_exp(float a)
{ return as_type<float>(soft_gfk_exp(as_type<uint>(a))); }
static inline float thompson_aa_log(float a)
{ return as_type<float>(soft_gfk_log(as_type<uint>(a))); }
// WRF39's *_cr helpers compute the transcendental in binary64 and then round
// once to binary32. Keep the WRF461 GFK implementation above unchanged.
static inline float thompson_aa_wrf39_powf_cr(float a, float b)
{ return thompson_aa_narrow(soft64_plm_pow(thompson_aa_widen(a), thompson_aa_widen(b))); }
static inline float thompson_aa_wrf39_expf_cr(float a)
{ return thompson_aa_narrow(soft64_plm_exp(thompson_aa_widen(a))); }
static inline float thompson_aa_wrf39_logf_cr(float a)
{ return thompson_aa_narrow(soft64_plm_log(thompson_aa_widen(a))); }
static inline float thompson_aa_wrf39_exp_native(float a)
{ return exp(a); }
static inline float thompson_aa_powf_mode(float a, float b, bool wrf39)
{ return wrf39 ? thompson_aa_wrf39_powf_cr(a, b) : thompson_aa_powf(a, b); }
static inline float thompson_aa_expf_mode(float a, bool wrf39)
{ return wrf39 ? thompson_aa_wrf39_expf_cr(a) : thompson_aa_exp(a); }
static inline float thompson_aa_logf_mode(float a, bool wrf39)
{ return wrf39 ? thompson_aa_wrf39_logf_cr(a) : thompson_aa_log(a); }
static inline float thompson_aa_sqrt(float a)
{ return as_type<float>(soft32_sqrt(as_type<uint>(a))); }
static inline float thompson_aa_rslf(float p,float t) {
    const float x=max(-80.0f,t-273.16f);
    float e=611.583699f+x*(44.4606896f+x*(1.43177157f+x*(0.0264224321f+x*(0.000299291081f+x*(0.00000203154182f+x*(0.00000000702620698f+x*(0.0000000000379534310f+x*(-0.0000000000000321582393f))))))));
    e=min(e,p*0.15f);return 0.622f*e/(p-e);
}
static inline float thompson_aa_thetae(float p,float t,float w,float tl) {
    const float rr=w+1.e-8f;
    return t*precise::pow(100000.f/p,0.2854f*(1.f-0.28f*rr))
        * precise::exp((3.376f/tl-0.00254f)*(rr*1000.f*(1.f+0.81f*rr)));
}
static inline float thompson_aa_tdew(float p,float w) {
    const float rr=w+1.e-8f;const float el=precise::log(p*rr/(0.622f+rr));
    return (35.86f*el-4947.2325f)/(el-23.6837f);
}
static inline float thompson_aa_tlcl(float t,float td) {
    return 1.f/(1.f/(td-56.f)+precise::log(t/td)/800.f)+56.f;
}
static inline float thompson_aa_twetb(float te) {
    ThompsonAaD64 x(min(475.f,te));
    ThompsonAaD64 c=ThompsonAaD64::from_bits(0xbddbbdc662f6cbbcul)+x*(ThompsonAaD64::from_bits(0xbe4fc5619ce33efdul)+x*(ThompsonAaD64::from_bits(0xbebd07e833dbb204ul)+x*(ThompsonAaD64::from_bits(0xbf20a86ec98c5086ul)+x*(ThompsonAaD64::from_bits(0x3eb36fb6c22856e6ul)+x*(ThompsonAaD64::from_bits(0xbe2e38d10dd467c4ul)+x*ThompsonAaD64::from_bits(0x3d8f7046774fa371ul))))));
    ThompsonAaD64 d=ThompsonAaD64(0)+x*(ThompsonAaD64::from_bits(0xbdf83496000e13b7ul)+x*(ThompsonAaD64::from_bits(0xbe6ebc7db224cadcul)+x*(ThompsonAaD64::from_bits(0xbed87e636ac48256ul)+x*(ThompsonAaD64::from_bits(0x3e695d3ca4020994ul)+x*(ThompsonAaD64::from_bits(0xbddf2c040abc83f0ul)+x*ThompsonAaD64::from_bits(0x3d3892535affcbeaul))))));
    return float((te<=335.5f?c:d)+ThompsonAaD64::from_bits(0x4071126666666666ul));
}
static inline float thompson_aa_fromte(float target,float p) {
    float guess=(target-0.5f*precise::pow(max(target-270.f,0.f),1.05f))*precise::pow(p/100000.f,0.2f);
    for(int i=0;i<100;i++){
        float w1=thompson_aa_rslf(p,guess),w2=thompson_aa_rslf(p,guess+1.f);
        float a=thompson_aa_thetae(p,guess,w1,guess),b=thompson_aa_thetae(p,guess+1.f,w2,guess+1.f);
        float den=b-a;if(fabs(den)<1.e-12f)break;
        float correction=(target-a)/den;guess+=correction;if(fabs(correction)<.01f)return guess;
    }
    return thompson_aa_twetb(target)*precise::pow(p/100000.f,.286f);
}
static inline float thompson_aa_wetbulb(float p,float t,float w) {
    if(w/thompson_aa_rslf(p,t)>=.999f)return t;
    float dew=min(t-.001f,thompson_aa_tdew(p,w));
    return min(t,thompson_aa_fromte(thompson_aa_thetae(p,t,w,thompson_aa_tlcl(t,dew)),p));
}
static inline float thompson_aa_log10_cr(float a)
{ return thompson_aa_narrow(soft64_div(
    soft64_plm_log(thompson_aa_widen(a)), 0x40026bb1bbb55516ul)); }

constant float thompson_aa_cce1[16] = {
    0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
constant float thompson_aa_ccg3[16] = {
    0, 5040.001953f, 40319.99609f, 362879.9688f, 3628801.75f,
    39916800.0f, 479001856.0f, 6227022336.0f, 87178297344.0f,
    1307673886720.0f, 20922782191616.0f, 355687448240128.0f,
    6402383730776064.0f, 121645285000000000.0f,
    2432903398000000000.0f, 51090914450000000000.0f};
constant float thompson_aa_ccg4[16] = {
    0, 6, 24, 120.0000076f, 720.0000610f, 5040.001953f,
    40319.99609f, 362879.9688f, 3628801.75f, 39916800.0f,
    479001856.0f, 6227022336.0f, 87178297344.0f,
    1307673886720.0f, 20922782191616.0f, 355687448240128.0f};
constant float thompson_aa_ccg5[16] = {
    0, 720.0000610f, 5040.001953f, 40319.99609f, 362879.9688f,
    3628801.75f, 39916800.0f, 479001856.0f, 6227022336.0f,
    87178297344.0f, 1307673886720.0f, 20922782191616.0f,
    355687448240128.0f, 6402383730776064.0f,
    121645285000000000.0f, 2432903398000000000.0f};
constant float thompson_aa_cse6[16] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
constant float thompson_aa_cse9[16] = {
    0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
constant float thompson_aa_cse11[16] = {
    0, 0.5f, 1, 1.5f, 2, 2.5f, 3, 3.5f, 4, 4.5f, 5,
    5.5f, 6, 6.5f, 7, 7.5f};

struct ThompsonAaWarmRain {
    bool l_qc, rain_active;
    int nu_c;
    float nc_m3, mvd_c, xdc, nwfa_m3, nifa_m3, nr_m3, mvd_r;
    ThompsonAaD64 lamc, lamr, n0_r;
    ThompsonAaD64 prr_wau, pnr_wau, pnc_wau, prr_rcw, pnc_rcw;
    ThompsonAaD64 pnr_rcr, pna_rca, pnd_rcd;
};

struct ThompsonAaFrozen {
    bool l_qs, l_qg;
    float rs, rg, ng_m3, smo0, smo1, smof;
    ulong ilamg, n0_g;
    ThompsonAaD64 prs_scw, pnc_scw, prg_gcw, pnc_gcw;
    ThompsonAaD64 pna_sca, pnd_scd, pna_gca, pnd_gcd;
};

static inline float thompson_aa_field_a(float tc, float moment, bool wrf39) {
    const float tc2 = thompson_aa_mul(tc, tc);
    const float m2 = thompson_aa_mul(moment, moment);
    float v = 5.065339f;
    v = thompson_aa_add(v, thompson_aa_mul(-0.062659f, tc));
    v = thompson_aa_add(v, thompson_aa_mul(-3.032362f, moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(0.029469f, tc), moment));
    v = thompson_aa_add(v, thompson_aa_mul(-0.000285f, tc2));
    v = thompson_aa_add(v, thompson_aa_mul(0.31255f, m2));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(0.000204f, tc2), moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(0.003199f, tc), m2));
    v = thompson_aa_add(v, thompson_aa_mul(-0.015952f, thompson_aa_mul(m2, moment)));
    return thompson_aa_powf_mode(10.0f, v, wrf39);
}
static inline float thompson_aa_field_b(float tc, float moment) {
    const float tc2 = thompson_aa_mul(tc, tc);
    const float m2 = thompson_aa_mul(moment, moment);
    float v = 0.476221f;
    v = thompson_aa_add(v, thompson_aa_mul(-0.015896f, tc));
    v = thompson_aa_add(v, thompson_aa_mul(0.165977f, moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(0.007468f, tc), moment));
    v = thompson_aa_add(v, thompson_aa_mul(-0.000141f, tc2));
    v = thompson_aa_add(v, thompson_aa_mul(0.060366f, m2));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(0.000079f, tc2), moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(0.000594f, tc), m2));
    return thompson_aa_add(v, thompson_aa_mul(-0.003577f, thompson_aa_mul(m2, moment)));
}
static inline int thompson_aa_nu_ci(float nc_m3) {
    return min(15, int(floor(thompson_aa_add(
        thompson_aa_div(1.0e9f, nc_m3), 0.5f))) + 2);
}

static inline float thompson_aa_cloud_dist(float rc, float nc_per_kg,
                                            float rho, thread int &nu_entry,
                                            bool wrf39) {
    float nc = max(2.0f, min(thompson_aa_mul(nc_per_kg, rho), 1999.0e6f));
    const int nu = thompson_aa_nu_ci(nc);
    nu_entry = nu;
    float arg = thompson_aa_mul(nc, 5.235988159e2f);
    arg = thompson_aa_mul(arg, thompson_aa_ccg2[nu]);
    arg = thompson_aa_mul(arg, thompson_aa_ocg1[nu]);
    arg = thompson_aa_div(arg, rc);
    ulong lambda = thompson_aa_widen(thompson_aa_powf_mode(arg, 3.333333433e-1f, wrf39));
    const float dnum = thompson_aa_add(thompson_aa_add(3.0f, float(nu)), 1.0f);
    const float diameter = thompson_aa_narrow(soft64_div(
        thompson_aa_widen(dnum), lambda));
    if (diameter < 1.0e-6f) lambda = thompson_aa_widen(
        thompson_aa_div(thompson_aa_cce2[nu], 1.0e-6f));
    else if (diameter > 1.0e-4f) lambda = thompson_aa_widen(
        thompson_aa_div(thompson_aa_cce2[nu], 1.0e-4f));
    float prefactor = thompson_aa_mul(thompson_aa_ccg1[nu], thompson_aa_ocg2[nu]);
    prefactor = thompson_aa_mul(prefactor, rc);
    prefactor = thompson_aa_div(prefactor, 5.235988159e2f);
    ulong diagnosed = soft64_mul(thompson_aa_widen(prefactor),
        soft64_mul(soft64_mul(lambda, lambda), lambda));
    const ulong cap = thompson_aa_widen(1999.0e6f);
    if (soft64_greater(diagnosed, cap)) diagnosed = cap;
    return thompson_aa_narrow(diagnosed);
}

static inline bool thompson_aa_entry_rain(float rain_per_kg,
        float rain_number_per_kg, float rho, thread float &number,
        thread ulong &lambda, thread float &mvd, thread ulong &n0,
        bool wrf39) {
    constexpr float R1 = 1.0e-12f, R2 = 1.0e-6f;
    constexpr float AM_R = 5.235988159e2f, OBMR = 3.333333433e-1f;
    float rr, nr; bool active;
    if (rain_per_kg > R1) {
        active = true; rr = thompson_aa_mul(rain_per_kg, rho);
        nr = max(R2, thompson_aa_mul(rain_number_per_kg, rho));
        if (nr <= R2) {
            const ulong lam = thompson_aa_widen(thompson_aa_div(3.672f, 1.0e-3f));
            nr = thompson_aa_narrow(soft64_mul(
                thompson_aa_widen(thompson_aa_div(thompson_aa_mul(1.0f/6.0f, rr), AM_R)),
                soft64_mul(soft64_mul(lam, lam), lam)));
        }
        const float arg = thompson_aa_div(
            thompson_aa_mul(thompson_aa_mul(AM_R, 6.0f), nr), rr);
        lambda = thompson_aa_widen(thompson_aa_powf_mode(arg, OBMR, wrf39));
        mvd = thompson_aa_narrow(soft64_div(thompson_aa_widen(3.672f), lambda));
        if (mvd > 2.5e-3f || mvd < 37.5e-6f) {
            mvd=clamp(mvd,37.5e-6f,2.5e-3f);
            lambda=thompson_aa_widen(thompson_aa_div(3.672f,mvd));
            const ulong lam3=soft64_mul(soft64_mul(lambda,lambda),lambda);
            const float prefix=thompson_aa_div(
                thompson_aa_mul(1.0f/6.0f,rr),AM_R);
            nr=thompson_aa_narrow(soft64_mul(thompson_aa_widen(prefix),lam3));
        }
    } else { active = false; rr=R1; nr=R2; }
    const float arg = thompson_aa_div(thompson_aa_mul(thompson_aa_mul(AM_R,6.0f),nr),rr);
    lambda = thompson_aa_widen(thompson_aa_powf_mode(arg,OBMR,wrf39));
    mvd = thompson_aa_narrow(soft64_div(thompson_aa_widen(3.672f),lambda));
    n0 = soft64_mul(thompson_aa_widen(nr),lambda);
    number=nr;
    return active;
}

static inline int thompson_aa_index(float value, int first, int count) {
    int e = int(floor(thompson_aa_log10_cr(value)));
    for (int q=e-1; q<=e+1; ++q) {
        const float scale=thompson_aa_powf(10.0f,float(q));
        const float m=thompson_aa_div(value,scale);
        if (m>=1.0f && m<10.0f) { e=q; break; }
    }
    const float scale=thompson_aa_powf(10.0f,float(e));
    return clamp(int(thompson_aa_div(value,scale))+9*(e-first)-1,0,count-1);
}
static inline int thompson_aa_index64(ulong value, int first, int count) {
    const ulong log10v=soft64_div(soft64_plm_log(value),
                                  0x40026bb1bbb55516ul);
    int e=int(floor(thompson_aa_narrow(log10v)));
    for(int q=e-1;q<=e+1;++q){
        const float scale=thompson_aa_powf(10.0f,float(q));
        const ulong m=soft64_div(value,thompson_aa_widen(scale));
        if(soft64_greater_equal(m,thompson_aa_widen(1.0f))
            && soft64_less(m,thompson_aa_widen(10.0f))){e=q;break;}
    }
    const float scale=thompson_aa_powf(10.0f,float(e));
    const int mant=int(thompson_aa_narrow(soft64_div(value,thompson_aa_widen(scale))));
    return clamp(mant+9*(e-first)-1,0,count-1);
}
static inline float thompson_aa_eff_aero(float d, float da, float visc,
        float rho, float temp, int species, bool wrf39) {
    float vt=1.0f;
    if(species==0){
        vt=thompson_aa_add(-0.1021f,thompson_aa_mul(4.932e3f,d));
        vt=thompson_aa_sub(vt,thompson_aa_mul(thompson_aa_mul(0.9551e6f,d),d));
        vt=thompson_aa_add(vt,thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(0.07934e9f,d),d),d));
        vt=thompson_aa_sub(vt,thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(0.002362e12f,d),d),d),d));
    } else if(species==1) vt=thompson_aa_mul(40.0f,thompson_aa_powf_mode(d,0.55f,wrf39));
    else vt=thompson_aa_mul(442.0f,thompson_aa_powf_mode(d,0.89f,wrf39));
    const float boltz=1.3806503e-23f, meanpath=0.0256e-6f;
    const float slip=thompson_aa_add(1.257f,thompson_aa_mul(0.4f,
        thompson_aa_expf_mode(thompson_aa_div(thompson_aa_mul(-0.55f,da),meanpath),wrf39)));
    const float cc=thompson_aa_add(1.0f,thompson_aa_mul(
        thompson_aa_div(thompson_aa_mul(2.0f,meanpath),da),slip));
    const float diff=thompson_aa_div(thompson_aa_mul(thompson_aa_mul(boltz,temp),cc),
        thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(3.0f,3.1415926536f),visc),da));
    const float re=thompson_aa_div(thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(0.5f,rho),d),vt),visc);
    const float sc=thompson_aa_div(visc,thompson_aa_mul(rho,diff));
    const float st=thompson_aa_div(thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(da,da),vt),1000.0f),
        thompson_aa_mul(thompson_aa_mul(9.0f,visc),d));
    const float aval=thompson_aa_add(1.0f,thompson_aa_logf_mode(thompson_aa_add(1.0f,re),wrf39));
    const float st2=thompson_aa_div(thompson_aa_add(1.2f,thompson_aa_mul(1.0f/12.0f,aval)),thompson_aa_add(1.0f,aval));
    const float sr=thompson_aa_sqrt(re);
    float brown=thompson_aa_add(1.0f,thompson_aa_mul(thompson_aa_mul(0.4f,sr),thompson_aa_powf_mode(sc,0.3333f,wrf39)));
    brown=thompson_aa_add(brown,thompson_aa_mul(thompson_aa_mul(0.16f,sr),thompson_aa_sqrt(sc)));
    const float term1=thompson_aa_mul(thompson_aa_div(4.0f,thompson_aa_mul(re,sc)),brown);
    const float ratio=thompson_aa_div(da,d);
    const float term2=thompson_aa_mul(thompson_aa_div(thompson_aa_mul(4.0f,da),d),
        thompson_aa_add(0.02f,thompson_aa_mul(ratio,thompson_aa_add(1.0f,thompson_aa_mul(2.0f,sr)))));
    float eff=thompson_aa_add(term1,term2);
    if(st>st2){const float excess=thompson_aa_sub(st,st2);eff=thompson_aa_add(eff,
        thompson_aa_powf_mode(thompson_aa_div(excess,thompson_aa_add(excess,0.666667f)),1.5f,wrf39));}
    return max(1.0e-5f,min(eff,1.0f));
}

static inline void thompson_aa_warm_rates(float qc,float nc_entry,float qr,float nr,
        float nwfa,float nifa,float rho,float temp,float rho_factor,float invdt,
        bool wrf39,
        device const ulong* eff_table,thread ThompsonAaWarmRain& o) {
    constexpr float R1=1.e-12f,AM_R=5.235988159e2f,OBMR=3.333333433e-1f;
    const float cloud_mass=max(0.0f,thompson_aa_mul(qc,rho));
    const float rain_mass=max(0.0f,thompson_aa_mul(qr,rho));
    const float tempc=thompson_aa_sub(temp,273.15f);
    o.l_qc=qc>R1; o.nu_c=0; o.nc_m3=2.0f; o.mvd_c=1.e-6f; o.xdc=1.e-6f;
    o.prr_wau=ThompsonAaD64(0);o.pnr_wau=ThompsonAaD64(0);o.pnc_wau=ThompsonAaD64(0);
    o.prr_rcw=ThompsonAaD64(0);o.pnc_rcw=ThompsonAaD64(0);o.pnr_rcr=ThompsonAaD64(0);
    o.pna_rca=ThompsonAaD64(0);o.pnd_rcd=ThompsonAaD64(0);
    o.nwfa_m3=max(11.1e6f,min(9999.e6f,thompson_aa_mul(nwfa,rho)));
    o.nifa_m3=max(5.e3f,min(9999.e6f,thompson_aa_mul(nifa,rho)));
    if(o.l_qc){int entry_nu=0;o.nc_m3=thompson_aa_cloud_dist(cloud_mass,nc_entry,rho,entry_nu,wrf39);
        o.nu_c=thompson_aa_nu_ci(o.nc_m3);
        float xdc=thompson_aa_mul(thompson_aa_powf_mode(thompson_aa_div(cloud_mass,thompson_aa_mul(AM_R,o.nc_m3)),OBMR,wrf39),1.e6f);
        o.xdc=max(1.0f,xdc);
        float arg=thompson_aa_div(thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(o.nc_m3,AM_R),thompson_aa_ccg2[o.nu_c]),thompson_aa_ocg1[o.nu_c]),cloud_mass);
        o.lamc=ThompsonAaD64(thompson_aa_powf_mode(arg,OBMR,wrf39));
        const float mvd=thompson_aa_narrow(soft64_div(thompson_aa_widen(thompson_aa_add(thompson_aa_add(3.0f,float(o.nu_c)),0.672f)),o.lamc.bits));
        o.mvd_c=wrf39?mvd:max(1.e-6f,min(mvd,50.e-6f));
    }
    float rain_number, rain_mvd; ulong rain_lambda,n0;
    o.rain_active=thompson_aa_entry_rain(qr,nr,rho,rain_number,rain_lambda,rain_mvd,n0,wrf39);
    o.nr_m3=rain_number;o.lamr=ThompsonAaD64::from_bits(rain_lambda);o.mvd_r=rain_mvd;o.n0_r=ThompsonAaD64::from_bits(n0);
    if(cloud_mass>1.e-5f){
        const float dcg=thompson_aa_mul(thompson_aa_narrow(soft64_div(
            thompson_aa_widen(thompson_aa_powf_mode(thompson_aa_mul(thompson_aa_ccg3[o.nu_c],thompson_aa_ocg2[o.nu_c]),OBMR,wrf39)),o.lamc.bits)),1.e6f);
        const float xdc3=thompson_aa_mul(thompson_aa_mul(o.xdc,o.xdc),o.xdc);
        const float arg=thompson_aa_sub(thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(xdc3,dcg),dcg),dcg),thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(xdc3,o.xdc),o.xdc),o.xdc));
        const float dcb=thompson_aa_powf_mode(max(0.0f,arg),1.0f/6.0f,wrf39);
        const float zterm=thompson_aa_sub(thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(6.25e-6f,o.xdc),dcb),thompson_aa_mul(dcb,dcb)),0.4f);
        const float zeta=thompson_aa_mul(thompson_aa_mul(0.027f,cloud_mass),thompson_aa_mul(0.5f,thompson_aa_add(zterm,fabs(zterm))));
        const float taud=thompson_aa_add(thompson_aa_mul(0.5f,thompson_aa_add(thompson_aa_sub(thompson_aa_mul(0.5f,dcb),7.5f),fabs(thompson_aa_sub(thompson_aa_mul(0.5f,dcb),7.5f)))),R1);
        const float tau=thompson_aa_div(3.72f,thompson_aa_mul(cloud_mass,taud));
        o.prr_wau=thompson_aa_dmin_mode(ThompsonAaD64(thompson_aa_mul(cloud_mass,invdt)),ThompsonAaD64(thompson_aa_div(zeta,tau)),wrf39);
        const float number_diameter=wrf39?50.e-6f:thompson_aa_mul(10.0f,50.e-6f);
        o.pnr_wau=o.prr_wau/ThompsonAaD64(thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(AM_R,float(o.nu_c)),number_diameter),thompson_aa_mul(50.e-6f,50.e-6f)));
        o.pnc_wau=thompson_aa_dmin_mode(ThompsonAaD64(thompson_aa_mul(o.nc_m3,invdt)),o.prr_wau/ThompsonAaD64(thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(AM_R,o.mvd_c),o.mvd_c),o.mvd_c)),wrf39);
    }
    if(!o.rain_active)return;
    if(rain_mvd>50.e-6f){
        if(wrf39){const float arg=thompson_aa_mul(2300.0f,thompson_aa_sub(rain_mvd,1950.e-6f));o.pnr_rcr=ThompsonAaD64((1.0f-thompson_aa_wrf39_exp_native(arg))*2.0f*rain_number*rain_mass);}
        else o.pnr_rcr=ThompsonAaD64((1.0f-thompson_aa_exp(thompson_aa_mul(2300.0f,thompson_aa_sub(rain_mvd,1950.e-6f))))*2.0f*rain_number*rain_mass);
    }
    if(!(rain_mvd>50.e-6f))return;
    const ThompsonAaD64 fall=thompson_aa_dpow(o.lamr+ThompsonAaD64(195.0f),ThompsonAaD64(-4));
    constexpr float t1=3.1415926536f*0.25f*4854.0f*6.0f;
    if(o.l_qc&&o.mvd_c>1.e-6f){
        const ulong first=0x3f0ad333f9742941ul; // 5.1164649614037726e-05
        const ulong last=0x3f740389985ae9dbul;  // 0.004886186104779057
        const ulong log_ratio=soft64_div(
            soft64_plm_log(soft64_div(thompson_aa_widen(rain_mvd),first)),
            soft64_plm_log(soft64_div(last,first)));
        const int rb=min(100,1+soft64_to_i32_rz(soft64_mul(
            soft64_from_i32(100),log_ratio)).value);
        const int cb=wrf39?min(100,int(thompson_aa_mul(o.mvd_c,1.e6f))):int(thompson_aa_mul(o.mvd_c,1.e6f));
        const float efficiency=thompson_aa_narrow(eff_table[(rb-1)+100*(cb-1)]);
        o.prr_rcw=thompson_aa_dmin_mode(ThompsonAaD64(thompson_aa_mul(cloud_mass,invdt)),ThompsonAaD64(thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(rho_factor,t1),efficiency),cloud_mass))*o.n0_r*fall,wrf39);
        o.pnc_rcw=thompson_aa_dmin_mode(ThompsonAaD64(thompson_aa_mul(o.nc_m3,invdt)),ThompsonAaD64(thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(rho_factor,t1),efficiency),o.nc_m3))*o.n0_r*fall,wrf39);
    }
    const float visc=(tempc>=0.0f)?(1.718f+0.0049f*tempc)*1.e-5f:(1.718f+0.0049f*tempc-1.2e-5f*tempc*tempc)*1.e-5f;
    const float efccn=thompson_aa_eff_aero(rain_mvd,0.04e-6f,visc,rho,temp,0,wrf39);
    const float efin=thompson_aa_eff_aero(rain_mvd,0.8e-6f,visc,rho,temp,0,wrf39);
    o.pna_rca=thompson_aa_dmin_mode(ThompsonAaD64(thompson_aa_mul(o.nwfa_m3,invdt)),ThompsonAaD64(thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(rho_factor,t1),efccn),o.nwfa_m3))*o.n0_r*fall,wrf39);
    o.pnd_rcd=thompson_aa_dmin_mode(ThompsonAaD64(thompson_aa_mul(o.nifa_m3,invdt)),ThompsonAaD64(thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(rho_factor,t1),efin),o.nifa_m3))*o.n0_r*fall,wrf39);
}

static inline float thompson_aa_snow_moment(float tc,float moment,float mass,
                                            bool wrf39) {
    const float smob=thompson_aa_mul(mass,14.49275398f);
    return thompson_aa_mul(thompson_aa_field_a(tc,moment,wrf39),
        thompson_aa_powf_mode(smob,thompson_aa_field_b(tc,moment),wrf39));
}
static inline ulong thompson_aa_wrf39_table_intercept(float rg, ulong ilamg,
                                                       float am_g) {
    const ulong lamg=soft64_div(thompson_aa_widen(1.0f),ilamg);
    const float prefix=thompson_aa_div(thompson_aa_mul(1.0f/6.0f,rg),am_g);
    return soft64_mul(thompson_aa_widen(prefix),
                      soft64_plm_pow(lamg,0x4010000000000000ul));
}
static inline float thompson_aa_wrf39_am_g() {
    return thompson_aa_div(thompson_aa_mul(3.1415926536f,500.0f),6.0f);
}
static inline void thompson_aa_frozen_rates(float qs,float qg,float ng_perkg,
        bool lqc,float cloud_mass,float nc,float cloud_mvd,float nwfa,float nifa,
        float rho,float temp,float tw,float rho_factor,float invdt,
        bool wrf39,device const ulong* snow_eff,thread ThompsonAaFrozen& o) {
    constexpr float R1=1.e-12f,R2=1.e-6f,D0C=1.e-6f,D0R=50.e-6f;
    constexpr float PI=3.1415926536f,AM_G=PI*400.0f/6.0f;
    constexpr float CGG9=5.23476267f,CGG6=20.3632278f,OGG3=1.0f/6.0f;
    constexpr float CGE9=3.8899998664855957f;
    const float tc=thompson_aa_sub(temp,273.15f);
    const float t1sq=PI*0.25f*40.0f;
    const float t1gq=PI*0.25f*442.0f*CGG9;
    const float am_g=wrf39?thompson_aa_wrf39_am_g():AM_G;
    o.l_qs=qs>R1;o.l_qg=qg>R1;
    o.rs=o.l_qs?max(0.0f,thompson_aa_mul(qs,rho)):R1;
    o.rg=o.l_qg?max(0.0f,thompson_aa_mul(qg,rho)):R1;
    const float snowtc=min(-0.1f,tc);
    const float snow_d0=wrf39?200.0e-6f:300.0e-6f;
    const float frozen_first=wrf39?1.0e-5f:R2;
    const float smob=o.l_qs?thompson_aa_mul(o.rs,14.49275398f):0.0f;
    const float smo0=o.l_qs?thompson_aa_snow_moment(snowtc,0.0f,o.rs,wrf39):0.0f;
    o.smo0=smo0;
    o.smo1=o.l_qs?thompson_aa_snow_moment(snowtc,1.0f,o.rs,wrf39):0.0f;
    o.smof=o.l_qs?thompson_aa_snow_moment(snowtc,1.775f,o.rs,wrf39):0.0f;
    const float smoc=o.l_qs?thompson_aa_snow_moment(snowtc,3.0f,o.rs,wrf39):0.0f;
    const float smoe=o.l_qs?thompson_aa_snow_moment(snowtc,2.55f,o.rs,wrf39):0.0f;
    const float xds=o.l_qs?thompson_aa_div(smoc,smob):0.0f;
    float ng=R2; float mvdg=0.0f; ulong lamg=thompson_aa_widen(1.0f);
    ulong ilam, n0;
    if(wrf39){
        const ThompsonAaD64 slope_base=ThompsonAaD64(ng_perkg)
            * ThompsonAaD64(am_g) * ThompsonAaD64(6) / ThompsonAaD64(o.rg);
        const ulong fork_lam=thompson_aa_dpow(slope_base,ThompsonAaD64(0.25f)).bits;
        ilam=soft64_div(thompson_aa_widen(1.0f),fork_lam);
        n0=soft64_mul(soft64_div(thompson_aa_widen(ng_perkg),fork_lam),
                      fork_lam);
        ng=0.0f;
    } else {
        if(o.l_qg){
            ng=max(R2,thompson_aa_mul(ng_perkg,rho));
            if(ng<=R2){mvdg=1.5e-3f;lamg=thompson_aa_widen(thompson_aa_div(3.671999931f,mvdg));
                const ulong lam3=soft64_mul(soft64_mul(lamg,lamg),lamg);
                ng=thompson_aa_narrow(soft64_div(soft64_mul(
                    thompson_aa_widen(thompson_aa_mul(1.0f/6.0f,o.rg)),lam3),thompson_aa_widen(AM_G)));
            }
            const float arg=thompson_aa_div(thompson_aa_mul(thompson_aa_mul(AM_G,6.0f),ng),o.rg);
            lamg=thompson_aa_widen(thompson_aa_powf(arg,1.0f/3.0f));
            mvdg=thompson_aa_narrow(soft64_div(thompson_aa_widen(3.671999931f),lamg));
            if(mvdg>25.4e-3f){mvdg=25.4e-3f;lamg=thompson_aa_widen(thompson_aa_div(3.671999931f,mvdg));
                const ulong lam3=soft64_mul(soft64_mul(lamg,lamg),lamg);
                ng=thompson_aa_narrow(soft64_div(soft64_mul(thompson_aa_widen(thompson_aa_mul(1.0f/6.0f,o.rg)),lam3),thompson_aa_widen(AM_G)));}
            else if(mvdg<D0R){mvdg=D0R;lamg=thompson_aa_widen(thompson_aa_div(3.671999931f,mvdg));
                const ulong lam3=soft64_mul(soft64_mul(lamg,lamg),lamg);
                ng=thompson_aa_narrow(soft64_div(soft64_mul(thompson_aa_widen(thompson_aa_mul(1.0f/6.0f,o.rg)),lam3),thompson_aa_widen(AM_G)));}
        }
        const float lgarg=thompson_aa_div(thompson_aa_mul(thompson_aa_mul(AM_G,6.0f),ng),o.rg);
        const ulong lgfinal=thompson_aa_widen(thompson_aa_powf(lgarg,1.0f/3.0f));
        ilam=soft64_div(thompson_aa_widen(1.0f),lgfinal);
        n0=soft64_mul(thompson_aa_widen(ng),lgfinal);
    }
    o.ng_m3=ng;o.ilamg=ilam;o.n0_g=n0;
    o.prs_scw=ThompsonAaD64(0);o.pnc_scw=ThompsonAaD64(0);
    o.prg_gcw=ThompsonAaD64(0);o.pnc_gcw=ThompsonAaD64(0);
    o.pna_sca=ThompsonAaD64(0);o.pnd_scd=ThompsonAaD64(0);
    o.pna_gca=ThompsonAaD64(0);o.pnd_gcd=ThompsonAaD64(0);
    const float visco=tc>=0.0f?(1.718f+0.0049f*tc)*1.e-5f:(1.718f+0.0049f*tc-1.2e-5f*tc*tc)*1.e-5f;
    if(lqc&&cloud_mvd>D0C){
        if(xds>snow_d0){
            ulong f,l;
            if(wrf39){
                const ulong d0=thompson_aa_widen(snow_d0);
                const ulong lr=soft64_plm_log(soft64_div(thompson_aa_widen(0.02f),d0));
                f=soft64_mul(d0,soft64_plm_exp(soft64_mul(thompson_aa_widen(0.005f),lr)));
                l=soft64_mul(d0,soft64_plm_exp(soft64_mul(thompson_aa_widen(0.995f),lr)));
            }else{
                const ulong lr=soft64_plm_log(soft64_div(
                    0x3f947ae147ae147bul,0x3f33a92a30553261ul));
                f=soft64_mul(0x3f33a92a30553261ul,
                    soft64_plm_exp(soft64_mul(0x3f747ae147ae147bul,lr)));
                l=soft64_mul(0x3f33a92a30553261ul,
                    soft64_plm_exp(soft64_mul(0x3fefd70a3d70a3d7ul,lr)));
            }
            const ulong log_ratio=soft64_plm_log(
                soft64_div(thompson_aa_widen(xds),f));
            const ulong log_span=soft64_plm_log(soft64_div(l,f));
            int sb=1+int(thompson_aa_narrow(soft64_div(
                soft64_mul(0x4059000000000000ul,log_ratio),log_span)));
            sb=max(1,min(sb,100));
            const int cb=max(1,min(int(thompson_aa_mul(cloud_mvd,1.e6f)),100));
            const float esw=thompson_aa_narrow(snow_eff[(sb-1)+100*(cb-1)]);
            const float efac=thompson_aa_mul(thompson_aa_mul(rho_factor,t1sq),esw);
            o.prs_scw=thompson_aa_dmin_mode(ThompsonAaD64(thompson_aa_mul(cloud_mass,invdt)),ThompsonAaD64(efac*cloud_mass*smoe),wrf39);
            o.pnc_scw=thompson_aa_dmin_mode(ThompsonAaD64(thompson_aa_mul(nc,invdt)),ThompsonAaD64(efac*nc*smoe),wrf39);
        }
        if(o.rg>=frozen_first){
            const float xdg=thompson_aa_narrow(soft64_mul(thompson_aa_widen(4.0f),ilam));
            const float vtg=thompson_aa_narrow(soft64_mul(thompson_aa_widen(rho_factor*442.0f*CGG6*OGG3),
                soft64_plm_pow(ilam,thompson_aa_widen(0.89f))));
            const float stoke=cloud_mvd*cloud_mvd*vtg*1000.0f/(9.0f*visco*xdg);
            float egw=0.0f;
            if(stoke>=0.4f&&stoke<=10.0f)egw=thompson_aa_mul(0.55f,thompson_aa_log10_cr(thompson_aa_mul(2.51f,stoke)));
            else if(stoke>10.0f)egw=0.77f;
            if(wrf39){if(!(xdg>250.0e-6f))egw=0.0f;}
            else if(tw>273.15f)egw*=0.1f;
            const ulong fall=soft64_plm_pow(ilam,thompson_aa_widen(CGE9));
            const ThompsonAaD64 factor=ThompsonAaD64(rho_factor*t1gq*egw*cloud_mass)*ThompsonAaD64::from_bits(n0)*ThompsonAaD64::from_bits(fall);
            o.prg_gcw=factor;
            o.pnc_gcw=thompson_aa_dmin_mode(ThompsonAaD64(thompson_aa_mul(nc,invdt)),ThompsonAaD64(rho_factor*t1gq*egw*nc)*ThompsonAaD64::from_bits(n0)*ThompsonAaD64::from_bits(fall),wrf39);
        }
    }
    if(o.rs>frozen_first){
        const float efccn=thompson_aa_eff_aero(xds,0.04e-6f,visco,rho,temp,1,wrf39);
        const float efin=thompson_aa_eff_aero(xds,0.8e-6f,visco,rho,temp,1,wrf39);
        const float factor=rho_factor*t1sq*smoe;
        o.pna_sca=thompson_aa_dmin_mode(ThompsonAaD64(nwfa*invdt),ThompsonAaD64(factor*efccn*nwfa),wrf39);
        o.pnd_scd=thompson_aa_dmin_mode(ThompsonAaD64(nifa*invdt),ThompsonAaD64(factor*efin*nifa),wrf39);
    }
    if(o.rg>frozen_first){
        const float xdg=thompson_aa_narrow(soft64_mul(thompson_aa_widen(4.0f),ilam));
        const ulong fall=soft64_plm_pow(ilam,thompson_aa_widen(CGE9));
        const float efccn=thompson_aa_eff_aero(xdg,0.04e-6f,visco,rho,temp,2,wrf39);
        const float efin=thompson_aa_eff_aero(xdg,0.8e-6f,visco,rho,temp,2,wrf39);
        o.pna_gca=thompson_aa_dmin_mode(ThompsonAaD64(nwfa*invdt),ThompsonAaD64(rho_factor*t1gq*efccn*nwfa)*ThompsonAaD64::from_bits(n0)*ThompsonAaD64::from_bits(fall),wrf39);
        o.pnd_gcd=thompson_aa_dmin_mode(ThompsonAaD64(nifa*invdt),ThompsonAaD64(rho_factor*t1gq*efin*nifa)*ThompsonAaD64::from_bits(n0)*ThompsonAaD64::from_bits(fall),wrf39);
    }
}

kernel void thompson_aa_warm_source_network(
    constant ThompsonAaWarmSourceNetworkResources &r [[buffer(0)]],
    constant ThompsonAaWarmSourceNetworkParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if(idx>=uint(p.n))return;
    const bool entry_warm=r.graupel_melt_marker[idx]!=0.0f;
    r.graupel_melt_marker[idx]=0.0f;r.snow_melt_marker[idx]=0.0f;
    if(!entry_warm)return;
    const float temp0=r.temperature[idx],pressure0=r.pressure[idx];
    const float qv0=max(1.e-10f,r.qv[idx]);
    const float rho=thompson_aa_div(thompson_aa_mul(0.622f,pressure0),
        thompson_aa_mul(thompson_aa_mul(287.04f,temp0),thompson_aa_add(qv0,0.622f)));
    const float orho=thompson_aa_div(1.0f,rho),invdt=thompson_aa_div(1.0f,p.dt);
    const float tempc=thompson_aa_sub(temp0,273.15f);
    const float tw=thompson_aa_wetbulb(pressure0,temp0,qv0);
    const float rhof=thompson_aa_sqrt(thompson_aa_div(101325.0f/(287.05f*298.0f),rho));
    const float cloud_mass=max(0.0f,thompson_aa_mul(r.qc[idx],rho));
    const float rain_mass=max(0.0f,thompson_aa_mul(r.qr[idx],rho));
    ThompsonAaWarmRain warm;
    const bool wrf39=p.wrf39!=0;
    thompson_aa_warm_rates(r.qc[idx],r.nc_entry[idx],r.qr[idx],r.nr[idx],
        r.nwfa_entry[idx],r.nifa_entry[idx],rho,temp0,rhof,invdt,wrf39,
        r.rain_cloud_efficiency,warm);
    ThompsonAaFrozen frozen;
    thompson_aa_frozen_rates(r.qs[idx],r.qg[idx],r.graupel_number_per_kg[idx],
        warm.l_qc,cloud_mass,warm.nc_m3,warm.mvd_c,warm.nwfa_m3,warm.nifa_m3,
        rho,temp0,tw,rhof,invdt,wrf39,r.snow_cloud_efficiency,frozen);

    ThompsonAaD64 prs_r=ThompsonAaD64(0),prs_s=ThompsonAaD64(0),prs_g=ThompsonAaD64(0),prs_n=ThompsonAaD64(0);
    ThompsonAaD64 prg_r=ThompsonAaD64(0),prg_g=ThompsonAaD64(0),prg_n=ThompsonAaD64(0),prg_rn=ThompsonAaD64(0);
    if(rain_mass>=1.e-6f){
        const int rb=thompson_aa_index(rain_mass,-6,37);
        const ThompsonAaD64 rain_intercept=ThompsonAaD64(thompson_aa_div(thompson_aa_mul(1.0f/6.0f,rain_mass),523.5988159f))*warm.lamr*warm.lamr*warm.lamr*warm.lamr;
        const int rib=thompson_aa_index64(rain_intercept.bits,6,37);
        const float snow_mass=frozen.rs;
        const int ntb=wrf39?28:37, sg_exp0=wrf39?-5:-6;
        const float sg_first=wrf39?1.0e-5f:1.0e-6f;
        if(snow_mass>=sg_first){
            const int sb=thompson_aa_index(snow_mass,sg_exp0,ntb);
            const int raw=int(thompson_aa_div(thompson_aa_sub(tempc,2.5f),5.0f))-1;
            const int tb=min(9,max(1,-raw))-1;
            const uint k=uint(sb+ntb*(tb+9*(rib+37*rb)));
            if(!wrf39&&tw<273.15f){
                prs_r=-(ThompsonAaD64::from_bits(r.tmr_racs2[k])+ThompsonAaD64::from_bits(r.tcr_sacr2[k])+ThompsonAaD64::from_bits(r.tmr_racs1[k])+ThompsonAaD64::from_bits(r.tcr_sacr1[k]));
                prs_s=ThompsonAaD64::from_bits(r.tmr_racs2[k])+ThompsonAaD64::from_bits(r.tcr_sacr2[k])-ThompsonAaD64::from_bits(r.tcs_racs1[k])-ThompsonAaD64::from_bits(r.tms_sacr1[k]);
                prs_g=ThompsonAaD64::from_bits(r.tmr_racs1[k])+ThompsonAaD64::from_bits(r.tcr_sacr1[k])+ThompsonAaD64::from_bits(r.tcs_racs1[k])+ThompsonAaD64::from_bits(r.tms_sacr1[k]);
                prs_r=thompson_aa_dmax_mode(ThompsonAaD64(-rain_mass*invdt),prs_r,wrf39);
                prs_s=thompson_aa_dmax_mode(ThompsonAaD64(-snow_mass*invdt),prs_s,wrf39);
                prs_g=thompson_aa_dmin_mode(ThompsonAaD64((rain_mass+snow_mass)*invdt),prs_g,wrf39);
                prs_n=thompson_aa_dmin_mode(ThompsonAaD64(warm.nr_m3*invdt),ThompsonAaD64::from_bits(r.tnr_racs1[k])+ThompsonAaD64::from_bits(r.tnr_racs2[k])+ThompsonAaD64::from_bits(r.tnr_sacr1[k])+ThompsonAaD64::from_bits(r.tnr_sacr2[k]),wrf39);
            } else {
                prs_s=-ThompsonAaD64::from_bits(r.tcs_racs1[k])-ThompsonAaD64::from_bits(r.tms_sacr1[k])+ThompsonAaD64::from_bits(r.tmr_racs2[k])+ThompsonAaD64::from_bits(r.tcr_sacr2[k]);
                prs_s=thompson_aa_dmax_mode(ThompsonAaD64(-snow_mass*invdt),prs_s,wrf39);prs_r=-prs_s;
                if(wrf39)prs_n=thompson_aa_dmin_mode(ThompsonAaD64(warm.nr_m3*invdt),ThompsonAaD64::from_bits(r.tnr_racs2[k])+ThompsonAaD64::from_bits(r.tnr_sacr2[k]),wrf39);
            }
        }
        const float gm=frozen.rg;
        if(wrf39){
            if(gm>=sg_first){
                const int gmb=thompson_aa_index(gm,sg_exp0,ntb);
                const ulong intercept=thompson_aa_wrf39_table_intercept(
                    gm,frozen.ilamg,thompson_aa_wrf39_am_g());
                const int gib=thompson_aa_index64(intercept,4,ntb);
                const uint k=uint(gib+ntb*(gmb+ntb*(rib+37*rb)));
                prg_r=thompson_aa_dmin_mode(ThompsonAaD64(gm*invdt),ThompsonAaD64::from_bits(r.tcg_racg[k]),wrf39);
                prg_g=-prg_r;
                prg_rn=ThompsonAaD64(-5.0f)*ThompsonAaD64::from_bits(r.tnr_gacr[k]);
            }
        }else if(gm>=1.e-6f){
            const int gmb=thompson_aa_index(gm,-6,37);
            const int gib=thompson_aa_index64(frozen.n0_g,2,37);
            const uint nominal=uint(gib+37*(gmb+37*(0+1*(rib+37*rb))));
            const uint k=nominal+uint(4*37*37),ts=uint(37*37*1*37*37);
            if(k<ts){
                if(tw<273.15f){prg_g=thompson_aa_dmin_mode(ThompsonAaD64(rain_mass*invdt),ThompsonAaD64::from_bits(r.tmr_racg[k])+ThompsonAaD64::from_bits(r.tcr_gacr[k]),wrf39);prg_r=-prg_g;prg_rn=thompson_aa_dmin_mode(ThompsonAaD64(warm.nr_m3*invdt),ThompsonAaD64::from_bits(r.tnr_racg[k])+ThompsonAaD64::from_bits(r.tnr_gacr[k]),wrf39);}
                else{prg_r=thompson_aa_dmin_mode(ThompsonAaD64(gm*invdt),ThompsonAaD64::from_bits(r.tcg_racg[k]),wrf39);prg_g=-prg_r;prg_n=thompson_aa_dmin_mode(ThompsonAaD64(frozen.ng_m3*invdt),ThompsonAaD64::from_bits(r.tnr_racg[k]),wrf39);prg_rn=ThompsonAaD64(-1.5f)*ThompsonAaD64::from_bits(r.tnr_gacr[k]);}
            }
        }
    }

    const float diffusivity=2.11e-5f*thompson_aa_powf(thompson_aa_div(temp0,273.15f),1.94f)*(101325.0f/pressure0);
    const float viscosity=(1.718f+0.0049f*tempc)*1.e-5f;
    const float conductivity=(5.69f+0.0168f*tempc)*1.e-5f*418.936f;
    const float rho_factor_sqrt=thompson_aa_sqrt(rhof),viscosity_factor=thompson_aa_sqrt(rho/viscosity);
    const float vapor_deficit=max(0.0f,thompson_aa_rslf(pressure0,273.15f)-qv0);
    const float invfusion=1.0f/334000.0f,schmidt=thompson_aa_powf(0.632f,1.0f/3.0f);
    ThompsonAaD64 smelt(0),smeltn(0),gmelt(0),gmeltn(0);
    if(frozen.l_qs){
        const float mm=(3.1415926536f*4.0f*0.15f*invfusion*0.86f)*frozen.smo1+(3.1415926536f*4.0f*0.15f*invfusion*0.28f*schmidt*thompson_aa_sqrt(40.0f))*rho_factor_sqrt*viscosity_factor*frozen.smof;
        smelt=ThompsonAaD64((tempc*conductivity-2.5e6f*diffusivity*vapor_deficit)*mm);
        if(wrf39)smelt=smelt+ThompsonAaD64(4218.0f*invfusion*tempc)*(prs_r+frozen.prs_scw);
        else if(smelt>ThompsonAaD64(0))smelt=smelt+ThompsonAaD64(4218.0f)*ThompsonAaD64(invfusion)*ThompsonAaD64(tw-273.15f)*(prs_r+warm.prr_rcw+frozen.prs_scw);
        smelt=thompson_aa_dmin_mode(ThompsonAaD64(frozen.rs*invdt),thompson_aa_dmax_mode(ThompsonAaD64(0),smelt,wrf39),wrf39);
        if(wrf39){
            smeltn=ThompsonAaD64(frozen.smo0)/ThompsonAaD64(frozen.rs)*smelt*ThompsonAaD64(thompson_aa_powf_mode(10.0f,-0.25f*tempc,wrf39));
            smeltn=thompson_aa_dmin_mode(ThompsonAaD64(frozen.smo0*invdt),smeltn,wrf39);
        }else if(smelt>ThompsonAaD64(0))smeltn=ThompsonAaD64(frozen.smo0)/ThompsonAaD64(frozen.rs)*smelt*ThompsonAaD64(thompson_aa_powf(10.0f,-0.25f*(tw-273.15f)));
    }
    if(frozen.l_qg){
        const ThompsonAaD64 inverse=ThompsonAaD64::from_bits(frozen.ilamg),intercept=ThompsonAaD64::from_bits(frozen.n0_g);
        ThompsonAaD64 mint=intercept;if(!wrf39&&frozen.rg*frozen.ng_m3<1.e-4f)mint=ThompsonAaD64(1.e-4f/frozen.rg)/inverse;
        const ThompsonAaD64 mm=mint*(ThompsonAaD64(3.1415926536f*4.0f*0.5f*invfusion*0.86f)*inverse*inverse+ThompsonAaD64(3.1415926536f*4.0f*0.5f*invfusion*0.28f*schmidt*thompson_aa_sqrt(442.0f)*1.9021706581115723f)*ThompsonAaD64(rho_factor_sqrt*viscosity_factor)*thompson_aa_dpow(inverse,ThompsonAaD64(2.944999933f)));
        gmelt=ThompsonAaD64(tempc*conductivity-2.5e6f*diffusivity*vapor_deficit)*mm;
        gmelt=thompson_aa_dmin_mode(ThompsonAaD64(frozen.rg*invdt),thompson_aa_dmax_mode(ThompsonAaD64(0),gmelt,wrf39),wrf39);
        if(wrf39)gmeltn=intercept*inverse/ThompsonAaD64(frozen.rg)*gmelt*ThompsonAaD64(thompson_aa_powf_mode(10.0f,-0.5f*tempc,wrf39));
        else if(gmelt>ThompsonAaD64(0))gmeltn=gmelt*ThompsonAaD64(frozen.ng_m3)/ThompsonAaD64(frozen.rg)*ThompsonAaD64(thompson_aa_powf(10.0f,-0.33f*(tw-273.15f)));
    }
    const float qvsi=thompson_aa_rslf(pressure0,temp0);float ssati=qv0/qvsi-1.0f;if(fabs(ssati)<1.e-15f)ssati=0.0f;
    ThompsonAaD64 sv(0),gv(0),gvn(0);
    const bool snow_sublimates=wrf39?frozen.l_qs:(frozen.l_qs&&smelt<=ThompsonAaD64(0));
    const bool graupel_sublimates=wrf39?frozen.l_qg:(frozen.l_qg&&gmelt<=ThompsonAaD64(0));
    if(ssati<0.0f&&(snow_sublimates||graupel_sublimates)){
        const float it=1.0f/temp0,sd=rho*qvsi;
        const float st=it*(2.834e6f*it*(1.0f/461.5f)-1.0f);
        const float slope=sd*st,curv=sd*(st*st-2.0f*2.834e6f*it*it*it*(1.0f/461.5f)+it*it);
        const float gamma=2.834e6f*diffusivity/conductivity*slope,gr=gamma/(1.0f+gamma);
        const float alpha=max(1.e-9f,0.5f*gr*gr*curv/slope*sd/slope),xs=ssati*ssati,alpha2=alpha*alpha;
        const float geom=4.0f*3.1415926536f*(1.0f-alpha*ssati+2.0f*alpha2*xs-5.0f*alpha2*alpha*xs*ssati)/(1.0f+gamma);
        if(snow_sublimates){const float vent=0.28f*schmidt*thompson_aa_sqrt(40.0f)*rho_factor_sqrt*viscosity_factor;const float cap=wrf39?0.5f:0.15f;sv=ThompsonAaD64(cap*geom*diffusivity*ssati*sd*(0.86f*frozen.smo1+vent*frozen.smof));sv=thompson_aa_dmax_mode(ThompsonAaD64(-frozen.rs*invdt),sv,wrf39);}
        if(graupel_sublimates){const ThompsonAaD64 il=ThompsonAaD64::from_bits(frozen.ilamg);const float vent=0.28f*schmidt*thompson_aa_sqrt(442.0f)*1.9021706581115723f*rho_factor_sqrt*viscosity_factor;gv=ThompsonAaD64(0.5f*geom*diffusivity*ssati*sd)*ThompsonAaD64::from_bits(frozen.n0_g)*(ThompsonAaD64(0.86f)*il*il+ThompsonAaD64(vent)*thompson_aa_dpow(il,ThompsonAaD64(2.944999933f)));gv=thompson_aa_dmax_mode(ThompsonAaD64(-frozen.rg*invdt),gv,wrf39);gvn=gv*ThompsonAaD64(frozen.ng_m3)/ThompsonAaD64(frozen.rg);}
        const ThompsonAaD64 vs=sv+gv,vlim=ThompsonAaD64((qv0-qvsi)*(wrf39?1.0f:rho)*invdt*0.999f);
        if(vs<ThompsonAaD64(-1.e-15f)&&vs<vlim){const ThompsonAaD64 ratio=vlim/vs;sv=sv*ratio;gv=gv*ratio;}
    }
    ThompsonAaD64 aut=warm.prr_wau,autn=warm.pnr_wau,rcw=warm.prr_rcw,scw=frozen.prs_scw,gcw=frozen.prg_gcw;
    if(p.dt>120.0f){rcw=rcw+scw+gcw;scw=ThompsonAaD64(0);gcw=ThompsonAaD64(0);}
    ThompsonAaD64 cloud_sink=aut+rcw+scw+gcw;
    if(cloud_sink>ThompsonAaD64(cloud_mass*invdt)&&cloud_sink>ThompsonAaD64(0)){const ThompsonAaD64 ratio=ThompsonAaD64(cloud_mass*invdt)/cloud_sink;aut=aut*ratio;rcw=rcw*ratio;scw=scw*ratio;gcw=gcw*ratio;cloud_sink=ThompsonAaD64(cloud_mass*invdt);}
    ThompsonAaD64 rain_sum=prs_r+prg_r;
    if(rain_mass>1.0e-12f&&rain_sum<ThompsonAaD64(-rain_mass*invdt)){const ThompsonAaD64 ratio=ThompsonAaD64(-rain_mass*invdt)/rain_sum;prs_r=prs_r*ratio;prg_r=prg_r*ratio;}
    ThompsonAaD64 snow_sum=sv+prs_s-smelt;if(frozen.l_qs&&snow_sum<ThompsonAaD64(-frozen.rs*invdt)){const ThompsonAaD64 ratio=ThompsonAaD64(-frozen.rs*invdt)/snow_sum;sv=sv*ratio;prs_s=prs_s*ratio;smelt=smelt*ratio;}
    ThompsonAaD64 gsum=gv+prg_g-gmelt;if(frozen.l_qg&&gsum<ThompsonAaD64(-frozen.rg*invdt)){const ThompsonAaD64 ratio=ThompsonAaD64(-frozen.rg*invdt)/gsum;gv=gv*ratio;prg_g=prg_g*ratio;gmelt=gmelt*ratio;}
    if(wrf39?temp0>273.15f:tw>273.15f){ThompsonAaD64 paired=thompson_aa_dmin_mode(ThompsonAaD64::from_bits(soft64_abs_bits(prs_r.bits)),ThompsonAaD64::from_bits(soft64_abs_bits(prs_s.bits)),wrf39);prs_r=ThompsonAaD64::from_bits((paired.bits&0x7ffffffffffffffful)|(prs_r.bits&0x8000000000000000ul));prs_s=-prs_r;}
    ThompsonAaD64 paired=thompson_aa_dmin_mode(ThompsonAaD64::from_bits(soft64_abs_bits(prg_r.bits)),ThompsonAaD64::from_bits(soft64_abs_bits(prg_g.bits)),wrf39);prg_r=ThompsonAaD64::from_bits((paired.bits&0x7ffffffffffffffful)|(prg_r.bits&0x8000000000000000ul));prg_g=-prg_r;
    r.snow_melt_marker[idx]=smelt>ThompsonAaD64(0)?1.0f:0.0f;r.graupel_melt_marker[idx]=gmelt>ThompsonAaD64(0)?1.0f:0.0f;
    const ThompsonAaD64 rainrate=aut+rcw+smelt+gmelt+prs_r+prg_r;
    const ThompsonAaD64 nrr=autn+smeltn+gmeltn-warm.pnr_rcr-prs_n-prg_rn;
    const ThompsonAaD64 snowrate=sv+scw+prs_s-smelt,graupelrate=gv+gcw+prs_g+prg_g-gmelt;
    const ThompsonAaD64 ngr=gvn+prs_n-prg_n-gmeltn,vapor=sv+gv;

    const float qr_entry=r.qr[idx]>1.e-12f?r.qr[idx]:0.0f,nr_entry=r.qr[idx]>1.e-12f?r.nr[idx]:0.0f;
    const float qg_entry=r.qg[idx]>1.e-12f?r.qg[idx]:0.0f,ng_entry=r.qg[idx]>1.e-12f?r.graupel_number_per_kg[idx]:0.0f;
    r.qc[idx]=max(0.0f,thompson_aa_sub(r.qc[idx],thompson_aa_mul(thompson_aa_narrow(soft64_mul(cloud_sink.bits,thompson_aa_widen(orho))),p.dt)));
    r.qr[idx]=max(0.0f,thompson_aa_add(r.qr[idx],thompson_aa_mul(thompson_aa_narrow(soft64_mul(rainrate.bits,thompson_aa_widen(orho))),p.dt)));
    r.nr[idx]=max(0.0f,thompson_aa_add(r.nr[idx],thompson_aa_mul(thompson_aa_narrow(soft64_mul(nrr.bits,thompson_aa_widen(orho))),p.dt)));
    r.qs[idx]=max(0.0f,thompson_aa_add(r.qs[idx],thompson_aa_mul(thompson_aa_narrow(soft64_mul(snowrate.bits,thompson_aa_widen(orho))),p.dt)));
    r.qg[idx]=max(0.0f,thompson_aa_add(r.qg[idx],thompson_aa_mul(thompson_aa_narrow(soft64_mul(graupelrate.bits,thompson_aa_widen(orho))),p.dt)));
    if(!wrf39)r.graupel_number_per_kg[idx]=thompson_aa_add(r.graupel_number_per_kg[idx],thompson_aa_mul(thompson_aa_narrow(soft64_mul(ngr.bits,thompson_aa_widen(orho))),p.dt));
    r.qv[idx]=thompson_aa_sub(r.qv[idx],thompson_aa_mul(thompson_aa_narrow(soft64_mul(vapor.bits,thompson_aa_widen(orho))),p.dt));
    float nrnew=r.nr[idx];
    if(thompson_aa_mul(r.qr[idx],rho)<=1.e-12f)nrnew=0.0f;
    else {float rm=thompson_aa_mul(r.qr[idx],rho),n=max(1.e-6f,thompson_aa_mul(nrnew,rho));float lam=thompson_aa_powf_mode(thompson_aa_div(thompson_aa_mul(thompson_aa_mul(523.5988159f,6.0f),n),rm),1.0f/3.0f,wrf39);float mvd=3.672f/lam;if(mvd>2.5e-3f||mvd<37.5e-6f){mvd=clamp(mvd,37.5e-6f,2.5e-3f);lam=3.672f/mvd;n=thompson_aa_div(thompson_aa_mul(thompson_aa_mul(1.0f/6.0f,rm),thompson_aa_mul(thompson_aa_mul(lam,lam),lam)),523.5988159f);nrnew=n/rho;}}
    r.nr[idx]=nrnew;
    if(!(thompson_aa_mul(r.qr[idx],rho)>1.e-12f)){r.qr[idx]=thompson_aa_add(qr_entry,thompson_aa_mul(thompson_aa_mul(-qr_entry,invdt),p.dt));r.nr[idx]=thompson_aa_add(nr_entry,thompson_aa_mul(thompson_aa_mul(-nr_entry,invdt),p.dt));}
    if(!wrf39){
        const float xrg=thompson_aa_mul(r.qg[idx],rho);
        if(!(xrg>1.e-12f)){r.qg[idx]=thompson_aa_add(qg_entry,thompson_aa_mul(thompson_aa_mul(-qg_entry,invdt),p.dt));r.graupel_number_per_kg[idx]=thompson_aa_add(ng_entry,thompson_aa_mul(thompson_aa_mul(-ng_entry,invdt),p.dt));}
        else{const float amg=3.1415926536f*400.0f/6.0f,xng=max(1.e-6f,r.graupel_number_per_kg[idx]*rho);ulong lam=thompson_aa_widen(thompson_aa_powf(thompson_aa_div(thompson_aa_mul(thompson_aa_mul(amg,6.0f),xng),xrg),1.0f/3.0f));float mvd=thompson_aa_narrow(soft64_div(thompson_aa_widen(3.672f),lam));bool bounded=false;if(mvd>25.4e-3f){lam=thompson_aa_widen(thompson_aa_div(3.672f,25.4e-3f));bounded=true;}else if(mvd<50.e-6f){lam=thompson_aa_widen(thompson_aa_div(3.672f,50.e-6f));bounded=true;}if(bounded){const ulong lam3=soft64_mul(soft64_mul(lam,lam),lam);const float xnb=thompson_aa_narrow(soft64_div(soft64_mul(thompson_aa_widen((1.0f/6.0f)*xrg),lam3),thompson_aa_widen(amg)));const float ngten=(xnb-thompson_aa_mul(ng_entry,rho))*invdt*(1.0f/rho);r.graupel_number_per_kg[idx]=ng_entry+ngten*p.dt;}}
    }
    const ThompsonAaD64 pnc=((( -warm.pnc_wau-warm.pnc_rcw)-ThompsonAaD64(0))-frozen.pnc_scw)-frozen.pnc_gcw;
    r.ncten[idx]=thompson_aa_narrow(soft64_add(thompson_aa_widen(r.ncten[idx]),soft64_mul(pnc.bits,thompson_aa_widen(orho))));
    const ThompsonAaD64 ccn=(warm.pna_rca+frozen.pna_sca)+frozen.pna_gca;
    const ThompsonAaD64 in=(warm.pnd_rcd+frozen.pnd_scd)+frozen.pnd_gcd;
    r.nwfaten[idx]=thompson_aa_narrow(soft64_sub(thompson_aa_widen(r.nwfaten[idx]),soft64_mul(ccn.bits,thompson_aa_widen(orho))));
    r.nifaten[idx]=thompson_aa_narrow(soft64_sub(thompson_aa_widen(r.nifaten[idx]),soft64_mul(in.bits,thompson_aa_widen(orho))));
    const float invcp=1.0f/(1004.0f*(1.0f+0.887f*qv0));
    const ThompsonAaD64 fusion=-smelt-gmelt-prs_r-prg_r;
    const ThompsonAaD64 heat=(ThompsonAaD64(334000.0f)*fusion)+(ThompsonAaD64(2.834e6f)*vapor);
    const ThompsonAaD64 tempdelta=heat*ThompsonAaD64(invcp)*ThompsonAaD64(orho)*ThompsonAaD64(p.dt);
    r.temperature[idx]=thompson_aa_add(temp0,thompson_aa_narrow(tempdelta.bits));
}


static inline int thompson_aa_nu_c(float nc_m3)
{
    return min(15, int(floor(thompson_aa_add(
        thompson_aa_div(1000000000.0f, nc_m3), 0.5f))) + 2);
}

kernel void thompson_aa_ncten_balance(
    constant ThompsonAaNctenBalanceResources &r [[buffer(0)]],
    constant ThompsonAaNctenBalanceParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;

    constexpr float R1 = 1.0e-12f;
    constexpr float NC_FLOOR = 2.0f;
    constexpr float NT_C_MAX = 1999.0e6f;
    constexpr float AM_R = 5.235988159e+02f;
    constexpr float BM_R = 3.0f;
    constexpr float OBMR = 3.333333433e-01f;
    constexpr float D0C = 1.0e-6f;
    constexpr float D0R = 50.0e-6f;

    const float rho = r.density[tid];
    const float orho = thompson_aa_div(1.0f, rho);
    const float odts = thompson_aa_div(1.0f, p.dt);
    float nc1d;
    float rc;
    if (r.qc_entry[tid] > R1) {
        nc1d = r.nc_entry[tid];
        rc = thompson_aa_mul(r.qc_entry[tid], rho);
    } else {
        nc1d = 0.0f;
        rc = R1;
    }

    float tendency = r.ncten[tid];
    const float xrc = max(R1, thompson_aa_mul(r.qc_after[tid], rho));
    float xnc = max(NC_FLOOR, thompson_aa_mul(
        thompson_aa_add(nc1d, thompson_aa_mul(tendency, p.dt)), rho));

    if (xrc > R1) {
        const int nu_c = thompson_aa_nu_c(xnc);
        const float arg = thompson_aa_div(
            thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(xnc, AM_R),
                                             thompson_aa_ccg2[nu_c]),
                            thompson_aa_ocg1[nu_c]), rc);
        ulong lamc = thompson_aa_widen(as_type<float>(soft_gfk_pow_bits(
            as_type<uint>(arg), as_type<uint>(OBMR))));
        const float xdc = thompson_aa_narrow(soft64_div(
            thompson_aa_widen(thompson_aa_add(
                thompson_aa_add(BM_R, float(nu_c)), 1.0f)), lamc));
        bool clamped = false;
        if (xdc < D0C) {
            lamc = thompson_aa_widen(thompson_aa_div(
                thompson_aa_cce2[nu_c], D0C));
            clamped = true;
        } else if (xdc > D0R * 2.0f) {
            lamc = thompson_aa_widen(thompson_aa_div(
                thompson_aa_cce2[nu_c], D0R * 2.0f));
            clamped = true;
        }
        if (clamped) {
            const float prefactor = thompson_aa_div(
                thompson_aa_mul(thompson_aa_mul(
                    thompson_aa_ccg1[nu_c], thompson_aa_ocg2[nu_c]), xrc),
                AM_R);
            const ulong raised = thompson_aa_pow64(
                lamc, thompson_aa_widen(BM_R));
            xnc = thompson_aa_narrow(soft64_mul(
                thompson_aa_widen(prefactor), raised));
            tendency = thompson_aa_mul(
                thompson_aa_mul(thompson_aa_sub(
                    xnc, thompson_aa_mul(nc1d, rho)), odts), orho);
        }
    } else {
        tendency = thompson_aa_mul(-nc1d, odts);
    }

    xnc = max(0.0f, thompson_aa_mul(
        thompson_aa_add(nc1d, thompson_aa_mul(tendency, p.dt)), rho));
    if (xnc > NT_C_MAX) {
        tendency = thompson_aa_mul(
            thompson_aa_mul(thompson_aa_sub(
                NT_C_MAX, thompson_aa_mul(nc1d, rho)), odts), orho);
    }
    r.ncten[tid] = tendency;
}
