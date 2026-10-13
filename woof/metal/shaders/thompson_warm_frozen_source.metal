#include <metal_stdlib>
#include "thompson_warm_frozen_source_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
using namespace metal;
#pragma clang fp contract(off)

struct D64 { ulong b; D64():b(0){} D64(float x):b(soft64_from_f32_bits(as_type<uint>(x))){} D64(int x):b(soft64_from_i32(x)){} operator float() const { return as_type<float>(soft64_to_f32_bits(b)); } static D64 fromBits(ulong x){D64 r;r.b=x;return r;} };
static inline D64 operator+(D64 a,D64 b){return D64::fromBits(soft64_add(a.b,b.b));}
static inline D64 operator-(D64 a,D64 b){return D64::fromBits(soft64_sub(a.b,b.b));}
static inline D64 operator*(D64 a,D64 b){return D64::fromBits(soft64_mul(a.b,b.b));}
static inline D64 operator/(D64 a,D64 b){return D64::fromBits(soft64_div(a.b,b.b));}
static inline D64 operator-(D64 a){return D64::fromBits(a.b^0x8000000000000000ul);}
static inline bool operator<(D64 a,D64 b){return soft64_less(a.b,b.b);}
static inline bool operator>(D64 a,D64 b){return soft64_greater(a.b,b.b);}
static inline bool operator<=(D64 a,D64 b){return !soft64_greater(a.b,b.b);}
static inline bool operator>=(D64 a,D64 b){return !soft64_less(a.b,b.b);}
static inline bool operator==(D64 a,D64 b){return soft64_equal(a.b,b.b);}
static inline thread D64& operator+=(thread D64& a,D64 b){a=a+b;return a;}
static inline thread D64& operator-=(thread D64& a,D64 b){a=a-b;return a;}
static inline thread D64& operator*=(thread D64& a,D64 b){a=a*b;return a;}
static inline thread D64& operator/=(thread D64& a,D64 b){a=a/b;return a;}
static inline D64 dmin(D64 a,D64 b){return a<b?a:b;} static inline D64 dmin(D64 a,float b){return dmin(a,D64(b));} static inline D64 dmin(float a,D64 b){return dmin(D64(a),b);}
static inline D64 dmax(D64 a,D64 b){return a>b?a:b;} static inline D64 dmax(D64 a,float b){return dmax(a,D64(b));} static inline D64 dmax(float a,D64 b){return dmax(D64(a),b);}
static inline D64 dabs(D64 a){return D64::fromBits(soft64_abs_bits(a.b));}
static inline D64 dcopysign(D64 a,D64 b){return D64::fromBits((a.b&0x7ffffffffffffffful)|(b.b&0x8000000000000000ul));}
// All Thompson call sites use a positive finite base; evaluate x^y through
// the existing soft64 log/exp path to keep this kernel pipeline tractable.
__attribute__((noinline)) static D64 dpow(D64 a,D64 b){return D64::fromBits(soft64_plm_pow(a.b,b.b));}
__attribute__((noinline)) static D64 dlog(D64 a){return D64::fromBits(soft64_plm_log(a.b));}
__attribute__((noinline)) static D64 dlog10(D64 a){return D64::fromBits(soft64_div(soft64_plm_log(a.b),0x40026bb1bbb55516ul));}
__attribute__((noinline)) static D64 dexp(D64 a){return D64::fromBits(soft64_plm_exp(a.b));}
static inline int dround(D64 a){bool neg=soft64_sign(a.b);ulong x=soft64_abs_bits(a.b);int v=soft64_to_i32_rz(soft64_add(x,0x3fe0000000000000ul)).value;return neg?-v:v;}
static inline D64 dconst(ulong x){return D64::fromBits(x);}
static inline ulong t64(float x){return soft64_from_f32_bits(as_type<uint>(x));}
static inline float t32(ulong x){return as_type<float>(soft64_to_f32_bits(x));}
static inline float tp(float a,float b){return precise::pow(a,b);}
static inline float fdiv_rn(float a,float b){return a/b;}
static inline float tthetae(float p,float t,float w,float tl){float rr=w+1.e-8f;float power=.2854f*(1.f-.28f*rr);return t*precise::pow(100000.f/p,power)*precise::exp((3.376f/tl-.00254f)*(rr*1000.f*(1.f+.81f*rr)));}
static inline float tdew(float p,float w){float rr=w+1.e-8f;float el=precise::log(p*rr/(.622f+rr));return (35.86f*el-4947.2325f)/(el-23.6837f);}
static inline float tlcl(float t,float td){return 1.f/(1.f/(td-56.f)+precise::log(t/td)/800.f)+56.f;}
static inline float twetb(float te){D64 x=D64(min(475.f,te));D64 c=D64::fromBits(0xbddbbdc662f6cbbcul)+x*(D64::fromBits(0xbe4fc5619ce33efdul)+x*(D64::fromBits(0xbebd07e833dbb204ul)+x*(D64::fromBits(0xbf20a86ec98c5086ul)+x*(D64::fromBits(0x3eb36fb6c22856e6ul)+x*(D64::fromBits(0xbe2e38d10dd467c4ul)+x*D64::fromBits(0x3d8f7046774fa371ul))))));D64 d=D64::fromBits(0ul)+x*(D64::fromBits(0xbdf83496000e13b7ul)+x*(D64::fromBits(0xbe6ebc7db224cadcul)+x*(D64::fromBits(0xbed87e636ac48256ul)+x*(D64::fromBits(0x3e695d3ca4020994ul)+x*(D64::fromBits(0xbddf2c040abc83f0ul)+x*D64::fromBits(0x3d3892535affcbea))))));return float((te<=335.5f?c:d)+D64::fromBits(0x4071126666666666ul));}
static inline float trslf(float p,float t);
__attribute__((noinline)) static float tfromte(float target,float p){float guess=(target-.5f*precise::pow(max(target-270.f,0.f),1.05f))*precise::pow(p/100000.f,.2f);for(int i=0;i<100;++i){float w1=trslf(p,guess),w2=trslf(p,guess+1.f),a=tthetae(p,guess,w1,guess),b=tthetae(p,guess+1.f,w2,guess+1.f),den=b-a;if(fabs(den)<1.e-12f)break;float correction=(target-a)/den;guess+=correction;if(fabs(correction)<.01f)return guess;}return twetb(target)*precise::pow(p/100000.f,.286f);}
__attribute__((noinline)) static float twet(float p,float t,float w){if(w/trslf(p,t)>=.999f)return t;float dew=min(t-.001f,tdew(p,w));return min(t,tfromte(tthetae(p,t,w,tlcl(t,dew)),p));}

kernel void thompson_warm_source_wet_bulb(
    constant ThompsonWarmSourceWetBulbResources& r [[buffer(0)]],
    constant ThompsonWarmSourceWetBulbParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    if (r.graupel_melt_marker[idx] == 0.0f) {
        r.wet_bulb[idx] = 0.0f;
        return;
    }
    const float qv0 = fmax(1.0e-10f, r.qv[idx]);
    r.wet_bulb[idx] = twet(r.pressure[idx], r.temperature[idx], qv0);
}

static inline float trslf(float p,float t){const float x=max(-80.f,t-273.16f);float e=611.583699f+x*(44.4606896f+x*(1.43177157f+x*(.0264224321f+x*(.000299291081f+x*(.00000203154182f+x*(.00000000702620698f+x*(0.379534310e-11f+x*(-.0000000000000321582393f))))))));e=min(e,p*.15f);return .622f*e/(p-e);}
static inline float tfield_a(float tc,float m){const float tc2=tc*tc,m2=m*m;return tp(10.f,5.065339f-.062659f*tc-3.032362f*m+.029469f*tc*m-.000285f*tc2+.31255f*m2+.000204f*tc2*m+.003199f*tc*m2-.015952f*m2*m);}
static inline float tfield_b(float tc,float m){const float tc2=tc*tc,m2=m*m;return .476221f-.015896f*tc+.165977f*m+.007468f*tc*m-.000141f*tc2+.060366f*m2+.000079f*tc2*m+.000594f*tc*m2-.003577f*m2*m;}
static inline int tindex(float value,int first,int n){int c=int(precise::round(precise::log10(value))),e=c;for(int q=c-1;q<=c+1;++q){float scale=tp(10.f,float(q)),mant=value/scale;if(mant>=1.f&&mant<10.f){e=q;break;}}float scale=tp(10.f,float(e));return clamp(int(value/scale)+9*(e-first)-1,0,n-1);}
static inline int tindex64(D64 value,int first,int n){int c=dround(dlog10(value)),e=c;for(int q=c-1;q<=c+1;++q){float scale=tp(10.f,float(q));D64 m=value/D64(scale);if(m>=D64(1)&&m<D64(10)){e=q;break;}}float scale=tp(10.f,float(e));return clamp(int(float(value/D64(scale)))+9*(e-first)-1,0,n-1);}
static inline bool train(float qr,float nr,float rho,thread float& number,thread D64& lambda,thread float& mvd){number=1.e-6f;lambda=D64(1);mvd=0.f;if(qr<=1.e-12f)return false;float mass=qr*rho,am=3.1415926536f*1000.f/6.f;number=max(1.e-6f,nr*rho);if(number<=1.e-6f){float l=3.672f/1.e-3f;number=fdiv_rn((1.f/6.f)*mass,am)*l*l*l;}float lf=tp(am*6.f*number/mass,1.f/3.f);mvd=3.672f/lf;if(mvd>2.5e-3f){mvd=2.5e-3f;lf=3.672f/mvd;number=fdiv_rn((1.f/6.f)*mass,am)*lf*lf*lf;}else if(mvd<37.5e-6f){mvd=37.5e-6f;lf=3.672f/mvd;number=fdiv_rn((1.f/6.f)*mass,am)*lf*lf*lf;}lambda=D64(lf);return true;}
static inline float tbound_rain(float mass,float rho,float nr){if(mass<=1.e-12f)return 0.f;float am=3.1415926536f*1000.f/6.f,number=max(1.e-6f,nr*rho),lambda=tp(am*6.f*number/mass,1.f/3.f),mvd=3.672f/lambda;if(mvd>2.5e-3f)mvd=2.5e-3f;else if(mvd<37.5e-6f)mvd=37.5e-6f;else return nr;lambda=3.672f/mvd;number=(1.f/6.f)*mass/am*lambda*lambda*lambda;return number/rho;}

kernel void thompson_warm_source_collision_rates(
    constant ThompsonWarmSourceCollisionRatesResources& r [[buffer(0)]],
    constant ThompsonWarmSourceCollisionRatesParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* out = r.collision_rates;
    const uint n = (uint)p.n;
    if (r.graupel_melt_marker[idx] == 0.0f) {
        for (uint slot = 0; slot < 8; ++slot) out[slot * n + idx] = 0ul;
        return;
    }
    const float temp0 = r.temperature[idx];
    const float pressure0 = r.pressure[idx];
    const float qv0 = fmax(1.0e-10f, r.qv[idx]);
    const float rho = 0.622f * pressure0
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float tempc = temp0 - 273.15f;
    const float wet_bulb = r.wet_bulb[idx];
    const float inverse_dt = 1.0f / p.dt;
    const float pi = 3.1415926536f;
    const float am_r = pi * 1000.0f / 6.0f;
    const float am_g = pi * 400.0f / 6.0f;
    const float rain_mass = fmax(0.0f, r.qr[idx] * rho);
    const float snow_mass = fmax(0.0f, r.qs[idx] * rho);
    const float graupel_mass = fmax(0.0f, r.qg[idx] * rho);
    float rain_number;
    D64 rain_lambda;
    float rain_mvd;
    const bool rain_active = train(
        r.qr[idx], r.nr[idx], rho, rain_number, rain_lambda, rain_mvd);
    const bool l_qg = r.qg[idx] > 1.0e-12f;
    float graupel_number = l_qg
        ? fmax(1.0e-6f, r.graupel_number_per_kg[idx] * rho) : 0.0f;
    D64 graupel_lambda = l_qg
        ? (D64)precise::pow(am_g * 6.0f * graupel_number / graupel_mass,
                       1.0f / 3.0f)
        : D64::fromBits(0x3ff0000000000000ul);
    if (l_qg) {
        float diameter = (float)(D64::fromBits(0x400d604189374bc7ul) / graupel_lambda);
        if (diameter > 25.4e-3f) {
            graupel_lambda = D64::fromBits(0x400d604189374bc7ul) / D64::fromBits(0x3f9a027525460aa6ul);
        } else if (diameter < 50.0e-6f) {
            graupel_lambda = D64::fromBits(0x400d604189374bc7ul) / D64::fromBits(0x3f0a36e2eb1c432dul);
        }
        graupel_number = (float)((D64)(fdiv_rn((1.0f / 6.0f)
            * graupel_mass,am_g)) * graupel_lambda * graupel_lambda
            * graupel_lambda);
    }
    D64 rain_snow_rain_rate = D64::fromBits(0x0000000000000000ul);
    D64 rain_snow_snow_rate = D64::fromBits(0x0000000000000000ul);
    D64 rain_snow_graupel_rate = D64::fromBits(0x0000000000000000ul);
    D64 rain_snow_number_rate = D64::fromBits(0x0000000000000000ul);
    D64 rain_graupel_rain_rate = D64::fromBits(0x0000000000000000ul);
    D64 rain_graupel_graupel_rate = D64::fromBits(0x0000000000000000ul);
    D64 rain_graupel_number_rate = D64::fromBits(0x0000000000000000ul);
    D64 rain_graupel_rain_number_rate = D64::fromBits(0x0000000000000000ul);
    if (rain_mass >= 1.0e-6f) {
        const int rain_bin = tindex(rain_mass, -6, 37);
        const D64 rain_intercept = (D64)(fdiv_rn((1.0f / 6.0f)
            * rain_mass,am_r)) * rain_lambda * rain_lambda
            * rain_lambda * rain_lambda;
        const int rain_intercept_bin = tindex64(rain_intercept, 6, 37);
        if (snow_mass >= 1.0e-6f) {
            const int snow_bin = tindex(snow_mass, -6, 37);
            const int raw_temp_bin = (int)(fdiv_rn((tempc - 2.5f),5.0f)) - 1;
            const int temp_bin = min(9, max(1, -raw_temp_bin)) - 1;
            const uint table_idx = (uint)snow_bin
                + (uint)37 * ((uint)temp_bin
                + (uint)9 * ((uint)rain_intercept_bin
                + (uint)37 * (uint)rain_bin));
            if (wet_bulb < 273.15f) {
                rain_snow_rain_rate = -(D64::fromBits(r.tmr_racs2[table_idx])
                    + D64::fromBits(r.tcr_sacr2[table_idx]) + D64::fromBits(r.tmr_racs1[table_idx])
                    + D64::fromBits(r.tcr_sacr1[table_idx]));
                rain_snow_snow_rate = D64::fromBits(r.tmr_racs2[table_idx])
                    + D64::fromBits(r.tcr_sacr2[table_idx]) - D64::fromBits(r.tcs_racs1[table_idx])
                    - D64::fromBits(r.tms_sacr1[table_idx]);
                rain_snow_graupel_rate = D64::fromBits(r.tmr_racs1[table_idx])
                    + D64::fromBits(r.tcr_sacr1[table_idx]) + D64::fromBits(r.tcs_racs1[table_idx])
                    + D64::fromBits(r.tms_sacr1[table_idx]);
                rain_snow_rain_rate = dmax((D64)(-rain_mass * inverse_dt), rain_snow_rain_rate);
                rain_snow_snow_rate = dmax((D64)(-snow_mass * inverse_dt), rain_snow_snow_rate);
                rain_snow_graupel_rate = dmin((D64)((rain_mass + snow_mass) * inverse_dt), rain_snow_graupel_rate);
                rain_snow_number_rate = D64::fromBits(r.tnr_racs1[table_idx])
                    + D64::fromBits(r.tnr_racs2[table_idx]) + D64::fromBits(r.tnr_sacr1[table_idx])
                    + D64::fromBits(r.tnr_sacr2[table_idx]);
                rain_snow_number_rate = dmin((D64)(rain_number * inverse_dt), rain_snow_number_rate);
            } else {
                rain_snow_snow_rate = -D64::fromBits(r.tcs_racs1[table_idx])
                    - D64::fromBits(r.tms_sacr1[table_idx]) + D64::fromBits(r.tmr_racs2[table_idx])
                    + D64::fromBits(r.tcr_sacr2[table_idx]);
                rain_snow_snow_rate = dmax((D64)(-snow_mass * inverse_dt), rain_snow_snow_rate);
                rain_snow_rain_rate = -rain_snow_snow_rate;
            }
        }
        if (graupel_mass >= 1.0e-6f) {
            const int graupel_mass_bin = tindex(graupel_mass, -6, 37);
            const D64 intercept = (D64)graupel_number * graupel_lambda;
            const int graupel_intercept_bin = tindex64(intercept, 2, 37);
            const uint nominal_idx = (uint)graupel_intercept_bin
                + (uint)37 * ((uint)graupel_mass_bin
                + (uint)37 * ((uint)0
                + (uint)1 * ((uint)rain_intercept_bin
                + (uint)37 * (uint)rain_bin)));
            const uint table_idx = nominal_idx + (uint)4 * 37 * 37;
            const uint table_size = (uint)37 * 37 * 1 * 37 * 37;
            if (table_idx < table_size) {
                if (wet_bulb < 273.15f) {
                    rain_graupel_graupel_rate = dmin(
                        (D64)(rain_mass * inverse_dt),
                        D64::fromBits(r.tmr_racg[table_idx]) + D64::fromBits(r.tcr_gacr[table_idx]));
                    rain_graupel_rain_rate = -rain_graupel_graupel_rate;
                    rain_graupel_rain_number_rate = dmin(
                        (D64)(rain_number * inverse_dt),
                        D64::fromBits(r.tnr_racg[table_idx]) + D64::fromBits(r.tnr_gacr[table_idx]));
                } else {
                    rain_graupel_rain_rate = dmin(
                        (D64)(graupel_mass * inverse_dt), D64::fromBits(r.tcg_racg[table_idx]));
                    rain_graupel_graupel_rate = -rain_graupel_rain_rate;
                    rain_graupel_number_rate = dmin(
                        (D64)(graupel_number * inverse_dt), D64::fromBits(r.tnr_racg[table_idx]));
                    rain_graupel_rain_number_rate =
                        -D64::fromBits(0x3ff8000000000000ul) * D64::fromBits(r.tnr_gacr[table_idx]);
                }
            }
        }
    }
    out[0 * n + idx] = rain_snow_rain_rate.b;
    out[1 * n + idx] = rain_snow_snow_rate.b;
    out[2 * n + idx] = rain_snow_graupel_rate.b;
    out[3 * n + idx] = rain_snow_number_rate.b;
    out[4 * n + idx] = rain_graupel_rain_rate.b;
    out[5 * n + idx] = rain_graupel_graupel_rate.b;
    out[6 * n + idx] = rain_graupel_number_rate.b;
    out[7 * n + idx] = rain_graupel_rain_number_rate.b;
}

enum ThompsonWarmSourceCloudSlot {
    THOMPSON_WSC_TEMP0 = 0, THOMPSON_WSC_PRESSURE0 = 1, THOMPSON_WSC_QV0 = 2, THOMPSON_WSC_RHO = 3, THOMPSON_WSC_ORHO = 4, THOMPSON_WSC_INVERSE_DT = 5, THOMPSON_WSC_TEMPC = 6, THOMPSON_WSC_WET_BULB = 7, THOMPSON_WSC_PI = 8, THOMPSON_WSC_AM_R = 9, THOMPSON_WSC_AM_G = 10, THOMPSON_WSC_RHO_FACTOR = 11, THOMPSON_WSC_CLOUD_MASS = 12, THOMPSON_WSC_RAIN_MASS = 13, THOMPSON_WSC_SNOW_MASS = 14, THOMPSON_WSC_GRAUPEL_MASS = 15, THOMPSON_WSC_L_QC = 16, THOMPSON_WSC_L_QS = 17, THOMPSON_WSC_L_QG = 18, THOMPSON_WSC_CLOUD_MVD = 19, THOMPSON_WSC_RAIN_NUMBER = 20, THOMPSON_WSC_RAIN_LAMBDA = 21, THOMPSON_WSC_RAIN_MVD = 22, THOMPSON_WSC_RAIN_ACTIVE = 23, THOMPSON_WSC_GRAUPEL_NUMBER = 24, THOMPSON_WSC_GRAUPEL_LAMBDA = 25, THOMPSON_WSC_SNOW_SECOND_MOMENT = 26, THOMPSON_WSC_SNOW_NUMBER = 27, THOMPSON_WSC_SNOW_FIRST_MOMENT = 28, THOMPSON_WSC_SNOW_VENTILATION_MOMENT = 29, THOMPSON_WSC_AUTOCONVERSION_RATE = 30, THOMPSON_WSC_AUTOCONVERSION_NUMBER_RATE = 31, THOMPSON_WSC_RAIN_SELF_NUMBER_RATE = 32, THOMPSON_WSC_RAIN_CLOUD_RATE = 33, THOMPSON_WSC_SNOW_CLOUD_RATE = 34, THOMPSON_WSC_GRAUPEL_CLOUD_RATE = 35, THOMPSON_WSC_GRAUPEL_CLOUD_INVERSE_LAMBDA = 36, THOMPSON_WSC_GRAUPEL_CLOUD_COEFFICIENT = 37, THOMPSON_WSC_GRAUPEL_VELOCITY_POWER = 38,
    THOMPSON_WSC_POW_LOG_HI = 39, THOMPSON_WSC_POW_LOG_LO = 40, THOMPSON_WSC_POW_PRODUCT_HI = 41, THOMPSON_WSC_POW_PRODUCT_LO = 42, THOMPSON_WSC_POW_SCALE = 43, THOMPSON_WSC_POW_STATUS = 44, THOMPSON_WSC_POW_SPECIAL = 45, THOMPSON_WSC_POW_T_H = 46, THOMPSON_WSC_POW_T_L = 47, THOMPSON_WSC_POW_S_L = 48, THOMPSON_WSC_POW_LOG_STATUS = 49, THOMPSON_WSC_POW_CP_PRODUCT_HI = 50, THOMPSON_WSC_POW_CP_PRODUCT_LO = 51, THOMPSON_WSC_GRAUPEL_INVERSE_LAMBDA_POW_2945 = 52, THOMPSON_WSC_GRAUPEL_INVERSE_LAMBDA_SQUARE = 53, THOMPSON_WSC_GRAUPEL_LAMBDA_POW_BASE = 54, THOMPSON_WSC_GRAUPEL_LAMBDA_POW_ACTIVE = 55, THOMPSON_WSC_GRAUPEL_LAMBDA_POW_3 = 56, THOMPSON_WSC_GRAUPEL_ENTRY_NUMBER = 57, THOMPSON_WSC_WARM_ENTRY_ACTIVE = 58, THOMPSON_WSC_SNOW_MELT_FUSION = 59, THOMPSON_WSC_GRAUPEL_MELT_FUSION = 60, THOMPSON_WSC_RAIN_SNOW_FUSION = 61, THOMPSON_WSC_RAIN_GRAUPEL_FUSION = 62, THOMPSON_WSC_VAPOR_HEAT = 63,
    THOMPSON_WSC_COUNT = 64
};

kernel void thompson_warm_source_cloud_rates(
    constant ThompsonWarmSourceCloudRatesResources& r [[buffer(0)]],
    constant ThompsonWarmSourceCloudRatesParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* out = r.cloud_state;
    const uint n = (uint)p.n;
    if (r.graupel_melt_marker[idx] == 0.0f) {
        for (uint slot = 0; slot < THOMPSON_WSC_COUNT; ++slot) out[slot * n + idx] = 0ul;
        return;
    }
    const float temp0 = r.temperature[idx];
    const float pressure0 = r.pressure[idx];
    const float qv0 = fmax(1.0e-10f, r.qv[idx]);
    const float rho = 0.622f * pressure0
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float orho = 1.0f / rho;
    const float inverse_dt = 1.0f / p.dt;
    const float tempc = temp0 - 273.15f;
    const float wet_bulb = r.wet_bulb[idx];
    const float pi = 3.1415926536f;
    const float am_r = pi * 1000.0f / 6.0f;
    const float am_g = pi * 400.0f / 6.0f;
    const float rho_factor = precise::sqrt(
        (101325.0f / (287.05f * 298.0f)) * orho);

    const float cloud_mass = fmax(0.0f, r.qc[idx] * rho);
    const float rain_mass = fmax(0.0f, r.qr[idx] * rho);
    const float snow_mass = fmax(0.0f, r.qs[idx] * rho);
    const float graupel_mass = fmax(0.0f, r.qg[idx] * rho);






    const bool l_qc = r.qc[idx] > 1.0e-12f;
    const bool l_qs = r.qs[idx] > 1.0e-12f;
    const bool l_qg = r.qg[idx] > 1.0e-12f;

    const float cloud_number = 100.0e6f;
    const float gamma_mass = 1.30767389e12f;
    const float inverse_gamma_number = 2.08767448e-9f;
    const float gamma_higher = 6.40238373e15f;
    const float inverse_gamma_mass = 7.64716632e-13f;
    float cloud_lambda = 0.0f;
    float cloud_mvd = 1.0e-6f;
    if (l_qc) {
        cloud_lambda = precise::pow(
            cloud_number * am_r * gamma_mass * inverse_gamma_number
                / cloud_mass,
            1.0f / 3.0f);
        cloud_mvd = fmax(1.0e-6f, fmin(
            15.672f / cloud_lambda, 50.0e-6f));
    }

    float rain_number;
    D64 rain_lambda;
    float rain_mvd;
    const bool rain_active = train(
        r.qr[idx], r.nr[idx], rho, rain_number, rain_lambda, rain_mvd);

    float graupel_number = l_qg
        ? fmax(1.0e-6f, r.graupel_number_per_kg[idx] * rho) : 0.0f;
    D64 graupel_lambda = l_qg
        ? (D64)precise::pow(am_g * 6.0f * graupel_number / graupel_mass,
                       1.0f / 3.0f)
        : D64::fromBits(0x3ff0000000000000ul);
    if (l_qg) {
        float diameter = (float)(D64::fromBits(0x400d604189374bc7ul) / graupel_lambda);
        if (diameter > 25.4e-3f) {
            graupel_lambda = D64::fromBits(0x400d604189374bc7ul) / D64::fromBits(0x3f9a027525460aa6ul);
        } else if (diameter < 50.0e-6f) {
            graupel_lambda = D64::fromBits(0x400d604189374bc7ul) / D64::fromBits(0x3f0a36e2eb1c432dul);
        }
        graupel_number = (float)((D64)(fdiv_rn((1.0f / 6.0f)
            * graupel_mass,am_g)) * graupel_lambda * graupel_lambda
            * graupel_lambda);
    }




    D64 autoconversion_rate = D64::fromBits(0x0000000000000000ul);
    D64 autoconversion_number_rate = D64::fromBits(0x0000000000000000ul);
    if (cloud_mass > 0.01e-3f) {
        const float xdc = fmax(1.0f,
            precise::pow(fdiv_rn(cloud_mass,am_r * cloud_number), 1.0f / 3.0f)
                * 1.0e6f);
        const float dcg = precise::pow(
            gamma_higher * inverse_gamma_mass, 1.0f / 3.0f)
            / cloud_lambda * 1.0e6f;
        const float xdc3 = xdc * xdc * xdc;
        const float dcg3 = dcg * dcg * dcg;
        const float dcb = precise::pow(fmax(
            0.0f, xdc3 * dcg3 - xdc3 * xdc3), 1.0f / 6.0f);
        const float zeta_term =
            6.25e-6f * xdc * dcb * dcb * dcb - 0.4f;
        const float zeta1 = 0.5f * (zeta_term + fabs(zeta_term));
        const float zeta = 0.027f * cloud_mass * zeta1;
        const float tau_diameter = 0.5f * dcb - 7.5f;
        const float taud = 0.5f
            * (tau_diameter + fabs(tau_diameter)) + 1.0e-12f;
        const float tau = 3.72f / (cloud_mass * taud);
        autoconversion_rate = dmin(
            (D64)(cloud_mass * inverse_dt), (D64)(zeta / tau));
        autoconversion_number_rate = autoconversion_rate
            / (D64)(am_r * 12.0f * 10.0f
                       * 50.0e-6f * 50.0e-6f * 50.0e-6f);
    }

    D64 rain_self_number_rate = D64::fromBits(0x0000000000000000ul);
    D64 rain_cloud_rate = D64::fromBits(0x0000000000000000ul);
    if (rain_active) {
        if (rain_mvd > 50.0e-6f) {
            const float efficiency = 1.0f
                - precise::exp(2300.0f * (rain_mvd - 1950.0e-6f));
            rain_self_number_rate = (D64)(
                efficiency * 2.0f * rain_number * rain_mass);
        }
        if (l_qc && rain_mvd > 50.0e-6f
                && cloud_mvd > 1.0e-6f) {
            const D64 first = D64::fromBits(0x3f0ad333f9742941ul);
            const D64 last = D64::fromBits(0x3f740389985ae9dbul);
            int rain_bin = 1 + (int)(D64::fromBits(0x4059000000000000ul)
                * dlog((D64)rain_mvd / first) / dlog(last / first));
            rain_bin = min(rain_bin, 100);
            const int cloud_bin = (int)(cloud_mvd * 1.0e6f);
            const float efficiency = (float)D64::fromBits(r.rain_cloud_efficiency[
                (rain_bin - 1) + 100 * (cloud_bin - 1)]);
            const float coefficient = pi * 0.25f * 4854.0f * 6.0f;
            const float intercept = rain_number * (float)rain_lambda;
            rain_cloud_rate = dmin(
                (D64)(cloud_mass * inverse_dt),
                (D64)(rho_factor * coefficient * efficiency
                    * cloud_mass * intercept
                    * precise::pow((float)rain_lambda + 195.0f, -4.0f)));
        }
    }




    D64 snow_cloud_rate = D64::fromBits(0x0000000000000000ul);
    float snow_second_moment = 0.0f;
    float snow_number = 0.0f;
    float snow_first_moment = 0.0f;
    float snow_ventilation_moment = 0.0f;
    if (l_qs) {
        const float tc0 = fmin(-0.1f, tempc);
        snow_second_moment = snow_mass * (1.0f / 0.069f);
        snow_number = tfield_a(tc0, 0.0f)
            * precise::pow(snow_second_moment, tfield_b(tc0, 0.0f));
        snow_first_moment = tfield_a(tc0, 1.0f)
            * precise::pow(snow_second_moment, tfield_b(tc0, 1.0f));
        snow_ventilation_moment = tfield_a(tc0, 1.775f)
            * precise::pow(snow_second_moment, tfield_b(tc0, 1.775f));
        const float snow_moment_3 = tfield_a(tc0, 3.0f)
            * precise::pow(snow_second_moment, tfield_b(tc0, 3.0f));
        const float snow_diameter = snow_moment_3 / snow_second_moment;
    }

    D64 graupel_cloud_rate = D64::fromBits(0x0000000000000000ul);
    out[0 * n + idx] = D64(temp0).b;
    out[1 * n + idx] = D64(pressure0).b;
    out[2 * n + idx] = D64(qv0).b;
    out[3 * n + idx] = D64(rho).b;
    out[4 * n + idx] = D64(orho).b;
    out[5 * n + idx] = D64(inverse_dt).b;
    out[6 * n + idx] = D64(tempc).b;
    out[7 * n + idx] = D64(wet_bulb).b;
    out[8 * n + idx] = D64(pi).b;
    out[9 * n + idx] = D64(am_r).b;
    out[10 * n + idx] = D64(am_g).b;
    out[11 * n + idx] = D64(rho_factor).b;
    out[12 * n + idx] = D64(cloud_mass).b;
    out[13 * n + idx] = D64(rain_mass).b;
    out[14 * n + idx] = D64(snow_mass).b;
    out[15 * n + idx] = D64(graupel_mass).b;
    out[16 * n + idx] = D64((int)l_qc).b;
    out[17 * n + idx] = D64((int)l_qs).b;
    out[18 * n + idx] = D64((int)l_qg).b;
    out[19 * n + idx] = D64(cloud_mvd).b;
    out[20 * n + idx] = D64(rain_number).b;
    out[21 * n + idx] = rain_lambda.b;
    out[22 * n + idx] = D64(rain_mvd).b;
    out[23 * n + idx] = D64((int)rain_active).b;
    out[24 * n + idx] = D64(graupel_number).b;
    out[25 * n + idx] = graupel_lambda.b;
    out[26 * n + idx] = D64(snow_second_moment).b;
    out[27 * n + idx] = D64(snow_number).b;
    out[28 * n + idx] = D64(snow_first_moment).b;
    out[29 * n + idx] = D64(snow_ventilation_moment).b;
    out[30 * n + idx] = autoconversion_rate.b;
    out[31 * n + idx] = autoconversion_number_rate.b;
    out[32 * n + idx] = rain_self_number_rate.b;
    out[33 * n + idx] = rain_cloud_rate.b;
    out[34 * n + idx] = snow_cloud_rate.b;
    out[35 * n + idx] = graupel_cloud_rate.b;
    out[36 * n + idx] = D64(1).b;
    out[37 * n + idx] = D64(0).b;
    out[38 * n + idx] = D64(0).b;
    for (uint slot = 39; slot < THOMPSON_WSC_COUNT; ++slot) out[slot * n + idx] = 0ul;
}

kernel void thompson_warm_source_snow_cloud_rate(
    constant ThompsonWarmSourceSnowCloudRateResources& r [[buffer(0)]],
    constant ThompsonWarmSourceSnowCloudRateParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint n = (uint)p.n;
    const float tempc = float(D64::fromBits(state[6 * n + idx]));
    const float inverse_dt = float(D64::fromBits(state[5 * n + idx]));
    const float rho_factor = float(D64::fromBits(state[11 * n + idx]));
    const float pi = float(D64::fromBits(state[8 * n + idx]));
    const float cloud_mass = float(D64::fromBits(state[12 * n + idx]));
    const float cloud_mvd = float(D64::fromBits(state[19 * n + idx]));
    const bool l_qc = float(D64::fromBits(state[16 * n + idx])) != 0.0f;
    const bool l_qs = float(D64::fromBits(state[17 * n + idx])) != 0.0f;
    const float snow_second_moment = float(D64::fromBits(state[26 * n + idx]));
    D64 snow_cloud_rate = D64::fromBits(0x0000000000000000ul);
    if (l_qs) {
        const float tc0 = fmin(-0.1f, tempc);
        const float snow_moment_3 = tfield_a(tc0, 3.0f)
            * precise::pow(snow_second_moment, tfield_b(tc0, 3.0f));
        const float snow_diameter = snow_moment_3 / snow_second_moment;
        if (l_qc && cloud_mvd > 1.0e-6f
                && snow_diameter > 300.0e-6f) {
            const D64 diameter_ratio = D64::fromBits(0x3f947ae147ae147bul) / D64::fromBits(0x3f33a92a30553261ul);
            const D64 log_ratio = dlog(diameter_ratio);
            const D64 first = D64::fromBits(0x3f33a92a30553261ul)
                * dexp(D64::fromBits(0x3fe0000000000000ul) / D64::fromBits(0x4059000000000000ul) * log_ratio);
            const D64 last = D64::fromBits(0x3f33a92a30553261ul)
                * dexp(D64::fromBits(0x4058e00000000000ul) / D64::fromBits(0x4059000000000000ul) * log_ratio);
            int snow_bin = 1 + (int)(D64::fromBits(0x4059000000000000ul)
                * dlog((D64)snow_diameter / first) / dlog(last / first));
            snow_bin = max(1, min(snow_bin, 100));
            const int cloud_bin = max(1, min((int)(cloud_mvd * 1.0e6f), 100));
            const float efficiency = (float)D64::fromBits(r.snow_cloud_efficiency[
                (snow_bin - 1) + 100 * (cloud_bin - 1)]);
            const float moment = tfield_a(tc0, 2.55f)
                * precise::pow(snow_second_moment, tfield_b(tc0, 2.55f));
            snow_cloud_rate = dmin(
                (D64)(cloud_mass * inverse_dt),
                (D64)(rho_factor * pi * 0.25f * 40.0f
                    * efficiency * cloud_mass * moment));
        }
    }
    state[34 * n + idx] = snow_cloud_rate.b;
}

kernel void thompson_warm_source_graupel_pow_log_reduce(
    constant ThompsonWarmSourceGraupelPowLogReduceResources& r [[buffer(0)]],
    constant ThompsonWarmSourceGraupelPowLogReduceParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint ncell = (uint)p.n;
    ulong ax;
    if (p.mode == 0) {
        const bool l_qg = float(D64::fromBits(state[18 * ncell + idx])) != 0.0f;
        const D64 graupel_lambda = D64::fromBits(state[25 * ncell + idx]);
        const ulong inverse_lambda = l_qg
            ? (D64::fromBits(0x3ff0000000000000ul) / graupel_lambda).b
            : 0x3ff0000000000000ul;
        state[36 * ncell + idx] = inverse_lambda;
        ax = inverse_lambda;
    } else if (p.mode == 3) {
        ax = state[54 * ncell + idx];
    } else {
        ax = state[36 * ncell + idx];
    }
    if (ax == 0x3ff0000000000000ul) {
        state[39 * ncell + idx] = 0ul;
        state[40 * ncell + idx] = 0ul;
        state[49 * ncell + idx] = 1ul;
        return;
    }
    const ulong plm_BP[2] = {0x3ff0000000000000ul, 0x3ff8000000000000ul};
    const ulong plm_TWO53 = 0x4340000000000000ul;
    int ix = (int)soft64_plm_hi(ax) & 0x7fffffff;
    int n = 0;
    int j;
    if (ix < 0x00100000) {
        ax = soft64_mul(ax, plm_TWO53);
        n -= 53;
        ix = (int)soft64_plm_hi(ax);
    }
    n += (ix >> 20) - 0x3ff;
    j = ix & 0x000fffff;
    int k;
    ix = j | 0x3ff00000;
    if (j <= 0x3988e) k = 0;
    else if (j < 0xbb67a) k = 1;
    else { k = 0; n += 1; ix -= 0x00100000; }
    ax = soft64_plm_sethi(ax, (uint)ix);
    ulong u = soft64_sub(ax, plm_BP[k]);
    ulong v = soft64_div(0x3ff0000000000000ul, soft64_add(ax, plm_BP[k]));
    ulong ss = soft64_mul(u, v);
    ulong s_h = soft64_plm_setlo(ss, 0);
    ulong t_h = soft64_plm_sethi(0ul,
        (((uint)ix >> 1) | 0x20000000u) + 0x00080000u + ((uint)k << 18));
    ulong t_l = soft64_sub(ax, soft64_sub(t_h, plm_BP[k]));
    ulong s_l = soft64_mul(v, soft64_sub(
        soft64_sub(u, soft64_mul(s_h, t_h)), soft64_mul(s_h, t_l)));
    state[39 * ncell + idx] = ax;
    state[40 * ncell + idx] = (ulong)(uint)n;
    state[41 * ncell + idx] = (ulong)(uint)k;
    state[42 * ncell + idx] = u;
    state[43 * ncell + idx] = v;
    state[44 * ncell + idx] = ss;
    state[45 * ncell + idx] = s_h;
    state[46 * ncell + idx] = t_h;
    state[47 * ncell + idx] = t_l;
    state[48 * ncell + idx] = s_l;
    state[49 * ncell + idx] = 0ul;
}

kernel void thompson_warm_source_graupel_pow_log_poly(
    constant ThompsonWarmSourceGraupelPowLogPolyResources& r [[buffer(0)]],
    constant ThompsonWarmSourceGraupelPowLogPolyParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint ncell = (uint)p.n;
    if (state[49 * ncell + idx] != 0ul) return;
    const ulong plm_L1 = 0x3fe3333333333303ul;
    const ulong plm_L2 = 0x3fdb6db6db6fabfful;
    const ulong plm_L3 = 0x3fd55555518f264dul;
    const ulong plm_L4 = 0x3fd17460a91d4101ul;
    const ulong plm_L5 = 0x3fcd864a93c9db65ul;
    const ulong plm_L6 = 0x3fca7e284a454eeful;
    const ulong ss = state[44 * ncell + idx];
    const ulong s_h = state[45 * ncell + idx];
    const ulong s_l = state[48 * ncell + idx];
    ulong s2 = soft64_mul(ss, ss);
    ulong rpoly = soft64_mul(soft64_mul(s2, s2), soft64_add(plm_L1,
        soft64_mul(s2, soft64_add(plm_L2,
        soft64_mul(s2, soft64_add(plm_L3,
        soft64_mul(s2, soft64_add(plm_L4,
        soft64_mul(s2, soft64_add(plm_L5,
        soft64_mul(s2, plm_L6)))))))))));
    rpoly = soft64_add(rpoly, soft64_mul(s_l, soft64_add(s_h, ss)));
    s2 = soft64_mul(s_h, s_h);
    const ulong t_h = soft64_plm_setlo(soft64_add(
        soft64_add(0x4008000000000000ul, s2), rpoly), 0);
    const ulong t_l = soft64_sub(rpoly, soft64_sub(
        soft64_sub(t_h, 0x4008000000000000ul), s2));
    state[46 * ncell + idx] = t_h;
    state[47 * ncell + idx] = t_l;
}

kernel void thompson_warm_source_graupel_pow_log_compensate_product(
    constant ThompsonWarmSourceGraupelPowLogCompensateProductResources& r [[buffer(0)]],
    constant ThompsonWarmSourceGraupelPowLogCompensateProductParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint ncell = (uint)p.n;
    if (state[49 * ncell + idx] != 0ul) return;
    const ulong ss = state[44 * ncell + idx];
    const ulong s_h = state[45 * ncell + idx];
    const ulong s_l = state[48 * ncell + idx];
    const ulong t_h = state[46 * ncell + idx];
    const ulong t_l = state[47 * ncell + idx];
    const ulong u = soft64_mul(s_h, t_h);
    const ulong v = soft64_add(soft64_mul(s_l, t_h), soft64_mul(t_l, ss));
    const ulong p_h = soft64_plm_setlo(soft64_add(u, v), 0);
    const ulong p_l = soft64_sub(v, soft64_sub(p_h, u));
    state[50 * ncell + idx] = p_h;
    state[51 * ncell + idx] = p_l;
}

kernel void thompson_warm_source_graupel_pow_log_compensate_sum(
    constant ThompsonWarmSourceGraupelPowLogCompensateSumResources& r [[buffer(0)]],
    constant ThompsonWarmSourceGraupelPowLogCompensateSumParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint ncell = (uint)p.n;
    if (state[49 * ncell + idx] != 0ul) return;
    const ulong plm_CP = 0x3feec709dc3a03fdul;
    const ulong plm_CP_H = 0x3feec709e0000000ul;
    const ulong plm_CP_L = 0xbe3e2fe0145b01f5ul;
    const ulong plm_DP_H[2] = {0ul, 0x3fe2b80340000000ul};
    const ulong plm_DP_L[2] = {0ul, 0x3e4cfdeb43cfd006ul};
    const int k = as_type<int>((uint)state[41 * ncell + idx]);
    const int n = as_type<int>((uint)state[40 * ncell + idx]);
    const ulong p_h = state[50 * ncell + idx];
    const ulong p_l = state[51 * ncell + idx];
    const ulong z_h = soft64_mul(plm_CP_H, p_h);
    const ulong z_l = soft64_add(soft64_add(soft64_mul(plm_CP_L, p_h),
        soft64_mul(p_l, plm_CP)), plm_DP_L[k]);
    const ulong nt = soft64_from_i32(n);
    const ulong log_hi = soft64_plm_setlo(soft64_add(
        soft64_add(soft64_add(z_h, z_l), plm_DP_H[k]), nt), 0);
    const ulong log_lo = soft64_sub(z_l, soft64_sub(
        soft64_sub(soft64_sub(log_hi, nt), plm_DP_H[k]), z_h));
    state[39 * ncell + idx] = log_hi;
    state[40 * ncell + idx] = log_lo;
}

kernel void thompson_warm_source_graupel_pow_multiply(
    constant ThompsonWarmSourceGraupelPowMultiplyResources& r [[buffer(0)]],
    constant ThompsonWarmSourceGraupelPowMultiplyParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint ncell = (uint)p.n;
    const ulong y = p.mode == 0 ? 0x3fec7ae147ae147bul
        : (p.mode == 1 ? 0x400f1eb851eb851ful
           : (p.mode == 2 ? 0x40078f5c28f5c28ful : 0x4008000000000000ul));
    const ulong plm_HUGE = 0x7e37e43c8800759cul;
    const ulong plm_TINY = 0x01a56e1fc2f8f359ul;
    const ulong plm_OVT = 0x3c971547652b82feul;
    ulong t1 = state[39 * ncell + idx];
    ulong t2 = state[40 * ncell + idx];
    const ulong y1 = soft64_plm_setlo(y, 0);
    ulong p_l = soft64_add(soft64_mul(soft64_sub(y, y1), t1), soft64_mul(y, t2));
    ulong p_h = soft64_mul(y1, t1);
    const ulong z = soft64_add(p_l, p_h);
    const int j = (int)soft64_plm_hi(z);
    const int i = (int)soft64_plm_word(z);
    int status = 0;
    ulong special = 0ul;
    if (j >= 0x40900000) {
        if (((j - 0x40900000) | i) != 0
                || soft64_greater(soft64_add(p_l, plm_OVT), soft64_sub(z, p_h))) {
            status = 1;
            special = soft64_mul(soft64_mul(0x3ff0000000000000ul, plm_HUGE), plm_HUGE);
        }
    } else if ((j & 0x7fffffff) >= 0x4090cc00) {
        if ((((uint)j - 0xc090cc00u) | (uint)i) != 0
                || soft64_less_equal(p_l, soft64_sub(z, p_h))) {
            status = 2;
            special = soft64_mul(soft64_mul(0x3ff0000000000000ul, plm_TINY), plm_TINY);
        }
    }
    int n = 0;
    if (status == 0) {
        const int iz = j & 0x7fffffff;
        const int k = (iz >> 20) - 0x3ff;
        if (iz > 0x3fe00000) {
            n = j + (0x00100000 >> (k + 1));
            const int nk = ((n & 0x7fffffff) >> 20) - 0x3ff;
            const ulong t = soft64_plm_sethi(0ul, (uint)(n & ~(0x000fffff >> nk)));
            n = ((n & 0x000fffff) | 0x00100000) >> (20 - nk);
            if (j < 0) n = -n;
            p_h = soft64_sub(p_h, t);
        }
    }
    state[41 * ncell + idx] = p_h;
    state[42 * ncell + idx] = p_l;
    state[43 * ncell + idx] = (ulong)(uint)n;
    state[44 * ncell + idx] = (ulong)status;
    state[45 * ncell + idx] = special;
}

kernel void thompson_warm_source_graupel_pow_exp_t(
    constant ThompsonWarmSourceGraupelPowExpTResources& r [[buffer(0)]],
    constant ThompsonWarmSourceGraupelPowExpTParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint ncell = (uint)p.n;
    if (state[44 * ncell + idx] != 0ul) return;
    const ulong p_h = state[41 * ncell + idx];
    const ulong p_l = state[42 * ncell + idx];
    state[48 * ncell + idx] = soft64_plm_setlo(
        soft64_add(p_l, p_h), 0);
}

kernel void thompson_warm_source_graupel_pow_exp_u(
    constant ThompsonWarmSourceGraupelPowExpUResources& r [[buffer(0)]],
    constant ThompsonWarmSourceGraupelPowExpUParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint ncell = (uint)p.n;
    if (state[44 * ncell + idx] != 0ul) return;
    const ulong plm_LG2_H = 0x3fe62e4300000000ul;
    state[49 * ncell + idx] = soft64_mul(
        state[48 * ncell + idx], plm_LG2_H);
}

kernel void thompson_warm_source_graupel_pow_exp_v(
    constant ThompsonWarmSourceGraupelPowExpVResources& r [[buffer(0)]],
    constant ThompsonWarmSourceGraupelPowExpVParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint ncell = (uint)p.n;
    if (state[44 * ncell + idx] != 0ul) return;
    const ulong plm_LG2 = 0x3fe62e42fefa39eful;
    const ulong plm_LG2_L = 0xbe205c610ca86c39ul;
    const ulong p_h = state[41 * ncell + idx];
    const ulong p_l = state[42 * ncell + idx];
    const ulong t = state[48 * ncell + idx];
    const ulong v = soft64_add(
        soft64_mul(soft64_sub(p_l, soft64_sub(t, p_h)), plm_LG2),
        soft64_mul(t, plm_LG2_L));
    state[50 * ncell + idx] = v;
}

kernel void thompson_warm_source_graupel_pow_exp_z(
    constant ThompsonWarmSourceGraupelPowExpZResources& r [[buffer(0)]],
    constant ThompsonWarmSourceGraupelPowExpZParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint ncell = (uint)p.n;
    if (state[44 * ncell + idx] != 0ul) return;
    state[51 * ncell + idx] = soft64_add(
        state[49 * ncell + idx], state[50 * ncell + idx]);
}

kernel void thompson_warm_source_graupel_pow_exp_w(
    constant ThompsonWarmSourceGraupelPowExpWResources& r [[buffer(0)]],
    constant ThompsonWarmSourceGraupelPowExpWParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint ncell = (uint)p.n;
    if (state[44 * ncell + idx] != 0ul) return;
    const ulong v = state[50 * ncell + idx];
    const ulong z = state[51 * ncell + idx];
    const ulong u = state[49 * ncell + idx];
    state[46 * ncell + idx] = z;
    state[47 * ncell + idx] = soft64_sub(
        v, soft64_sub(z, u));
}

kernel void thompson_warm_source_graupel_pow_exp_poly(
    constant ThompsonWarmSourceGraupelPowExpPolyResources& r [[buffer(0)]],
    constant ThompsonWarmSourceGraupelPowExpPolyParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint ncell = (uint)p.n;
    if (state[44 * ncell + idx] != 0ul) return;
    const ulong plm_P1 = 0x3fc555555555553eul;
    const ulong plm_P2 = 0xbf66c16c16bebd93ul;
    const ulong plm_P3 = 0x3f11566aaf25de2cul;
    const ulong plm_P4 = 0xbebbbd41c5d26bf1ul;
    const ulong plm_P5 = 0x3e66376972bea4d0ul;
    const ulong z = state[46 * ncell + idx];
    const ulong t = soft64_mul(z, z);
    const ulong t1 = soft64_sub(z, soft64_mul(t, soft64_add(plm_P1,
        soft64_mul(t, soft64_add(plm_P2,
        soft64_mul(t, soft64_add(plm_P3,
        soft64_mul(t, soft64_add(plm_P4,
        soft64_mul(t, plm_P5))))))))));
    state[48 * ncell + idx] = t;
    state[49 * ncell + idx] = t1;
}

kernel void thompson_warm_source_graupel_pow_exp_rational(
    constant ThompsonWarmSourceGraupelPowExpRationalResources& r [[buffer(0)]],
    constant ThompsonWarmSourceGraupelPowExpRationalParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint ncell = (uint)p.n;
    if (state[44 * ncell + idx] != 0ul) return;
    const ulong z = state[46 * ncell + idx];
    const ulong w = state[47 * ncell + idx];
    const ulong t = state[48 * ncell + idx];
    const ulong t1 = state[49 * ncell + idx];
    const ulong rpoly = soft64_sub(
        soft64_div(soft64_mul(z, t1), soft64_sub(t1, 0x4000000000000000ul)),
        soft64_add(w, soft64_mul(z, w)));
    state[46 * ncell + idx] = soft64_sub(
        0x3ff0000000000000ul, soft64_sub(rpoly, z));
}

kernel void thompson_warm_source_graupel_pow_exp_scale(
    constant ThompsonWarmSourceGraupelPowExpScaleResources& r [[buffer(0)]],
    constant ThompsonWarmSourceGraupelPowExpScaleParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint ncell = (uint)p.n;
    ulong z;
    const ulong status = state[44 * ncell + idx];
    if (status != 0ul) {
        z = state[45 * ncell + idx];
    } else {
        z = state[46 * ncell + idx];
        const int n = as_type<int>((uint)state[43 * ncell + idx]);
        const int j = (int)((uint)soft64_plm_hi(z) + ((uint)n << 20));
        z = (j >> 20) <= 0 ? soft64_plm_scalbn(z, n)
                            : soft64_plm_sethi(z, (uint)j);
        z = soft64_mul(0x3ff0000000000000ul, z);
    }
    if (p.mode == 0) state[38 * ncell + idx] = z;
    else if (p.mode == 1) state[35 * ncell + idx] = soft64_mul(state[37 * ncell + idx], z);
    else if (p.mode == 2) state[52 * ncell + idx] = z;
    else state[56 * ncell + idx] = z;
}

kernel void thompson_warm_source_graupel_pow_square(
    constant ThompsonWarmSourceGraupelPowSquareResources& r [[buffer(0)]],
    constant ThompsonWarmSourceGraupelPowSquareParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint ncell = (uint)p.n;
    const ulong inverse_lambda = state[36 * ncell + idx];
    state[53 * ncell + idx] = soft64_mul(inverse_lambda, inverse_lambda);
}

kernel void thompson_warm_source_graupel_cloud_rate_base(
    constant ThompsonWarmSourceGraupelCloudRateBaseResources& r [[buffer(0)]],
    constant ThompsonWarmSourceGraupelCloudRateBaseParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint n = (uint)p.n;
    const float tempc = float(D64::fromBits(state[6 * n + idx]));
    const float wet_bulb = float(D64::fromBits(state[7 * n + idx]));
    const float pi = float(D64::fromBits(state[8 * n + idx]));
    const float rho_factor = float(D64::fromBits(state[11 * n + idx]));
    const float cloud_mass = float(D64::fromBits(state[12 * n + idx]));
    const float graupel_mass = float(D64::fromBits(state[15 * n + idx]));
    const bool l_qc = float(D64::fromBits(state[16 * n + idx])) != 0.0f;
    const bool l_qg = float(D64::fromBits(state[18 * n + idx])) != 0.0f;
    const float cloud_mvd = float(D64::fromBits(state[19 * n + idx]));
    const float graupel_number = float(D64::fromBits(state[24 * n + idx]));
    const D64 graupel_lambda = D64::fromBits(state[25 * n + idx]);
    const D64 inverse_lambda = l_qg
        ? D64::fromBits(0x3ff0000000000000ul) / graupel_lambda
        : D64::fromBits(0x3ff0000000000000ul);
    D64 coefficient = D64::fromBits(0x0000000000000000ul);
    if (l_qc && l_qg && cloud_mvd > 1.0e-6f && graupel_mass >= 1.0e-6f) {
        const D64 intercept = (D64)graupel_number * graupel_lambda;
        const float viscosity = (1.718f + 0.0049f * tempc) * 1.0e-5f;
        const float diameter = (float)(D64::fromBits(0x4010000000000000ul) * inverse_lambda);
        const float velocity = (float)(
            (D64)(rho_factor * 442.0f * 20.3632278f * (1.0f / 6.0f))
            * D64::fromBits(state[38 * n + idx]));
        const float stokes = cloud_mvd * cloud_mvd * velocity * 1000.0f
            / (9.0f * viscosity * diameter);
        float efficiency = 0.0f;
        if (stokes >= 0.4f && stokes <= 10.0f) {
            efficiency = 0.55f * precise::log10(2.51f * stokes);
        } else if (stokes > 10.0f) {
            efficiency = 0.77f;
        }
        if (wet_bulb > 273.15f) efficiency *= 0.1f;
        const float prefactor = pi * 0.25f * 442.0f * 5.23476267f;
        coefficient = (D64)(rho_factor * prefactor * efficiency * cloud_mass)
            * intercept;
    }
    state[36 * n + idx] = inverse_lambda.b;
    state[37 * n + idx] = coefficient.b;
}

kernel void thompson_warm_frozen_source_network_melt(
    constant ThompsonWarmFrozenSourceNetworkMeltResources& r [[buffer(0)]],
    constant ThompsonWarmFrozenSourceNetworkMeltParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint n = (uint)p.n;
    const bool entry_warm = r.graupel_melt_marker[idx] != 0.0f;
    r.graupel_melt_marker[idx] = 0.0f;
    r.snow_melt_marker[idx] = 0.0f;
    state[58 * n + idx] = entry_warm ? 1ul : 0ul;
    if (!entry_warm) return;

    const float temp0 = float(D64::fromBits(state[0 * n + idx]));
    const float pressure0 = float(D64::fromBits(state[1 * n + idx]));
    const float qv0 = float(D64::fromBits(state[2 * n + idx]));
    const float rho = float(D64::fromBits(state[3 * n + idx]));
    const float inverse_dt = float(D64::fromBits(state[5 * n + idx]));
    const float tempc = float(D64::fromBits(state[6 * n + idx]));
    const float wet_bulb = float(D64::fromBits(state[7 * n + idx]));
    const float pi = float(D64::fromBits(state[8 * n + idx]));
    const float rho_factor = float(D64::fromBits(state[11 * n + idx]));
    const float cloud_mass = float(D64::fromBits(state[12 * n + idx]));
    const float snow_mass = float(D64::fromBits(state[14 * n + idx]));
    const float graupel_mass = float(D64::fromBits(state[15 * n + idx]));
    const bool l_qs = float(D64::fromBits(state[17 * n + idx])) != 0.0f;
    const bool l_qg = float(D64::fromBits(state[18 * n + idx])) != 0.0f;
    const float graupel_number = float(D64::fromBits(state[24 * n + idx]));
    const D64 graupel_lambda = D64::fromBits(state[25 * n + idx]);
    const float snow_number = float(D64::fromBits(state[27 * n + idx]));
    const float snow_first_moment = float(D64::fromBits(state[28 * n + idx]));
    const float snow_ventilation_moment = float(D64::fromBits(state[29 * n + idx]));
    const D64 snow_cloud_rate = D64::fromBits(state[34 * n + idx]);
    const D64 rain_snow_rain_rate = D64::fromBits(r.collision_rates[0 * n + idx]);

    const float diffusivity = 2.11e-5f
        * precise::pow(fdiv_rn(temp0,273.15f), 1.94f) * (101325.0f / pressure0);
    const float viscosity = (1.718f + 0.0049f * tempc) * 1.0e-5f;
    const float conductivity = (5.69f + 0.0168f * tempc)
        * 1.0e-5f * 418.936f;
    const float rho_factor_sqrt = precise::sqrt(rho_factor);
    const float viscosity_factor = precise::sqrt(rho / viscosity);
    const float vapor_deficit = fmax(
        0.0f, trslf(pressure0, 273.15f) - qv0);
    const float inverse_fusion = 1.0f / 334000.0f;
    const float schmidt_cuberoot = precise::pow(0.632f, 1.0f / 3.0f);

    D64 snow_melt_rate = D64::fromBits(0x0000000000000000ul);
    D64 snow_melt_number_rate = D64::fromBits(0x0000000000000000ul);
    if (l_qs) {
        const float melt_moment =
            (pi * 4.0f * 0.15f * inverse_fusion * 0.86f)
                * snow_first_moment
            + (pi * 4.0f * 0.15f * inverse_fusion * 0.28f
               * schmidt_cuberoot * precise::sqrt(40.0f))
                * rho_factor_sqrt * viscosity_factor
                * snow_ventilation_moment;
        snow_melt_rate = (D64)((tempc * conductivity
            - 2.5e6f * diffusivity * vapor_deficit) * melt_moment);
        if (snow_melt_rate > D64::fromBits(0x0000000000000000ul)) {
            snow_melt_rate += D64::fromBits(0x40b07a0000000000ul) * (D64)inverse_fusion
                * (D64)(wet_bulb - 273.15f)
                * (rain_snow_rain_rate + snow_cloud_rate);
        }
        snow_melt_rate = dmin(
            (D64)(snow_mass * inverse_dt),
            dmax(D64::fromBits(0x0000000000000000ul), snow_melt_rate));
        if (snow_melt_rate > D64::fromBits(0x0000000000000000ul)) {
            snow_melt_number_rate =
                (D64)snow_number / (D64)snow_mass * snow_melt_rate
                * (D64)precise::pow(
                    10.0f, -0.25f * (wet_bulb - 273.15f));
        }
    }

    D64 graupel_melt_rate = D64::fromBits(0x0000000000000000ul);
    D64 graupel_melt_number_rate = D64::fromBits(0x0000000000000000ul);
    if (l_qg) {
        const D64 inverse_lambda = D64::fromBits(state[36 * n + idx]);
        const D64 graupel_intercept =
            (D64)graupel_number * graupel_lambda;
        D64 melt_intercept = graupel_intercept;
        if (graupel_mass * graupel_number < 1.0e-4f) {
            melt_intercept = (D64)(1.0e-4f / graupel_mass)
                * graupel_lambda;
        }
        const D64 melt_moment = melt_intercept * (
            (D64)(pi * 4.0f * 0.5f * inverse_fusion * 0.86f)
                * inverse_lambda * inverse_lambda
            + (D64)(pi * 4.0f * 0.5f * inverse_fusion * 0.28f
                       * schmidt_cuberoot * precise::sqrt(442.0f)
                       * 1.9021706581115723f)
                * (D64)(rho_factor_sqrt * viscosity_factor)
                * D64::fromBits(state[52 * n + idx]));
        graupel_melt_rate = (D64)(tempc * conductivity
            - 2.5e6f * diffusivity * vapor_deficit) * melt_moment;
        graupel_melt_rate = dmin(
            (D64)(graupel_mass * inverse_dt),
            dmax(D64::fromBits(0x0000000000000000ul), graupel_melt_rate));
        if (graupel_melt_rate > D64::fromBits(0x0000000000000000ul)) {
            graupel_melt_number_rate = graupel_melt_rate
                * (D64)graupel_number / (D64)graupel_mass
                * (D64)precise::pow(
                    10.0f, -0.33f * (wet_bulb - 273.15f));
        }
    }
    state[39 * n + idx] = snow_melt_rate.b;
    state[40 * n + idx] = snow_melt_number_rate.b;
    state[41 * n + idx] = graupel_melt_rate.b;
    state[42 * n + idx] = graupel_melt_number_rate.b;
}

kernel void thompson_warm_frozen_source_network_vapor(
    constant ThompsonWarmFrozenSourceNetworkVaporResources& r [[buffer(0)]],
    constant ThompsonWarmFrozenSourceNetworkVaporParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint n = (uint)p.n;
    if (state[58 * n + idx] == 0ul) return;
    const float temp0 = float(D64::fromBits(state[0 * n + idx]));
    const float pressure0 = float(D64::fromBits(state[1 * n + idx]));
    const float qv0 = float(D64::fromBits(state[2 * n + idx]));
    const float rho = float(D64::fromBits(state[3 * n + idx]));
    const float inverse_dt = float(D64::fromBits(state[5 * n + idx]));
    const float rho_factor = float(D64::fromBits(state[11 * n + idx]));
    const float snow_mass = float(D64::fromBits(state[14 * n + idx]));
    const float graupel_mass = float(D64::fromBits(state[15 * n + idx]));
    const bool l_qs = float(D64::fromBits(state[17 * n + idx])) != 0.0f;
    const bool l_qg = float(D64::fromBits(state[18 * n + idx])) != 0.0f;
    const float snow_first_moment = float(D64::fromBits(state[28 * n + idx]));
    const float snow_ventilation_moment = float(D64::fromBits(state[29 * n + idx]));
    const D64 snow_melt_rate = D64::fromBits(state[39 * n + idx]);
    const D64 graupel_melt_rate = D64::fromBits(state[41 * n + idx]);
    const float pi = float(D64::fromBits(state[8 * n + idx]));
    const float diffusivity = 2.11e-5f
        * precise::pow(fdiv_rn(temp0,273.15f), 1.94f) * (101325.0f / pressure0);
    const float conductivity = (5.69f + 0.0168f
        * float(D64::fromBits(state[6 * n + idx]))) * 1.0e-5f * 418.936f;
    const float rho_factor_sqrt = precise::sqrt(rho_factor);
    const float viscosity = (1.718f + 0.0049f
        * float(D64::fromBits(state[6 * n + idx]))) * 1.0e-5f;
    const float viscosity_factor = precise::sqrt(rho / viscosity);
    const float schmidt_cuberoot = precise::pow(0.632f, 1.0f / 3.0f);
    const float qvsi = trslf(pressure0, temp0);
    float ssati = qv0 / qvsi - 1.0f;
    if (fabs(ssati) < 1.0e-15f) ssati = 0.0f;
    D64 snow_vapor_rate = D64::fromBits(0x0000000000000000ul);
    D64 graupel_vapor_rate = D64::fromBits(0x0000000000000000ul);
    D64 graupel_vapor_number_rate = D64::fromBits(0x0000000000000000ul);
    if (ssati < 0.0f
            && (snow_melt_rate <= D64::fromBits(0x0000000000000000ul)
                || graupel_melt_rate <= D64::fromBits(0x0000000000000000ul))) {
        const float inverse_temp = 1.0f / temp0;
        const float saturated_density = rho * qvsi;
        const float slope_term = inverse_temp
            * (2.834e6f * inverse_temp * (1.0f / 461.5f) - 1.0f);
        const float saturation_slope = saturated_density * slope_term;
        const float saturation_curvature = saturated_density
            * (slope_term * slope_term
               - 2.0f * 2.834e6f * inverse_temp * inverse_temp
                 * inverse_temp * (1.0f / 461.5f)
               + inverse_temp * inverse_temp);
        const float gamma = 2.834e6f * diffusivity / conductivity
            * saturation_slope;
        const float gamma_ratio = gamma / (1.0f + gamma);
        const float alpha = fmax(1.0e-9f,
            0.5f * gamma_ratio * gamma_ratio
            * saturation_curvature / saturation_slope
            * saturated_density / saturation_slope);
        const float xsat2 = ssati * ssati;
        const float alpha2 = alpha * alpha;
        const float geometry = 4.0f * pi
            * (1.0f - alpha * ssati
               + 2.0f * alpha2 * xsat2
               - 5.0f * alpha2 * alpha * xsat2 * ssati)
            / (1.0f + gamma);
        if (l_qs && snow_melt_rate <= D64::fromBits(0x0000000000000000ul)) {
            const float ventilation = 0.28f * schmidt_cuberoot
                * precise::sqrt(40.0f) * rho_factor_sqrt * viscosity_factor;
            snow_vapor_rate = (D64)(0.15f * geometry * diffusivity
                * ssati * saturated_density
                * (0.86f * snow_first_moment
                   + ventilation * snow_ventilation_moment));
            snow_vapor_rate = dmax(
                (D64)(-snow_mass * inverse_dt), snow_vapor_rate);
        }
        if (l_qg && graupel_melt_rate <= D64::fromBits(0x0000000000000000ul)) {
            const D64 inverse_lambda = D64::fromBits(state[36 * n + idx]);
            const D64 graupel_intercept =
                (D64)float(D64::fromBits(state[24 * n + idx]))
                * D64::fromBits(state[25 * n + idx]);
            const float ventilation = 0.28f * schmidt_cuberoot
                * precise::sqrt(442.0f) * 1.9021706581115723f
                * rho_factor_sqrt * viscosity_factor;
            graupel_vapor_rate = (D64)(0.5f * geometry * diffusivity
                * ssati * saturated_density) * graupel_intercept
                * (D64::fromBits(0x3feb851eb851eb85ul)
                    * D64::fromBits(state[53 * n + idx])
                   + (D64)ventilation
                    * D64::fromBits(state[52 * n + idx]));
            graupel_vapor_rate = dmax(
                (D64)(-graupel_mass * inverse_dt),
                graupel_vapor_rate);
            graupel_vapor_number_rate = graupel_vapor_rate
                * (D64)float(D64::fromBits(state[24 * n + idx]))
                / (D64)graupel_mass;
        }
        const D64 vapor_sum = snow_vapor_rate + graupel_vapor_rate;
        const D64 vapor_limit = (D64)((qv0 - qvsi) * rho
            * inverse_dt * 0.999f);
        if (vapor_sum < -D64::fromBits(0x3cd203af9ee75616ul)
                && vapor_sum < vapor_limit) {
            const D64 ratio = vapor_limit / vapor_sum;
            snow_vapor_rate *= ratio;
            graupel_vapor_rate *= ratio;
        }
    }
    state[43 * n + idx] = snow_vapor_rate.b;
    state[44 * n + idx] = graupel_vapor_rate.b;
    state[45 * n + idx] = graupel_vapor_number_rate.b;
}

kernel void thompson_warm_frozen_source_network_limits(
    constant ThompsonWarmFrozenSourceNetworkLimitsResources& r [[buffer(0)]],
    constant ThompsonWarmFrozenSourceNetworkLimitsParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint n = (uint)p.n;
    if (state[58 * n + idx] == 0ul) return;
    const float cloud_mass = float(D64::fromBits(state[12 * n + idx]));
    const float rain_mass = float(D64::fromBits(state[13 * n + idx]));
    const float snow_mass = float(D64::fromBits(state[14 * n + idx]));
    const float graupel_mass = float(D64::fromBits(state[15 * n + idx]));
    const float inverse_dt = float(D64::fromBits(state[5 * n + idx]));
    const float wet_bulb = float(D64::fromBits(state[7 * n + idx]));
    const bool l_qs = float(D64::fromBits(state[17 * n + idx])) != 0.0f;
    const bool l_qg = float(D64::fromBits(state[18 * n + idx])) != 0.0f;
    const bool rain_active = float(D64::fromBits(state[23 * n + idx])) != 0.0f;
    D64 autoconversion_rate = D64::fromBits(state[30 * n + idx]);
    D64 autoconversion_number_rate = D64::fromBits(state[31 * n + idx]);
    D64 rain_self_number_rate = D64::fromBits(state[32 * n + idx]);
    D64 rain_cloud_rate = D64::fromBits(state[33 * n + idx]);
    D64 snow_cloud_rate = D64::fromBits(state[34 * n + idx]);
    D64 graupel_cloud_rate = D64::fromBits(state[35 * n + idx]);
    D64 rain_snow_rain_rate = D64::fromBits(r.collision_rates[0 * n + idx]);
    D64 rain_snow_snow_rate = D64::fromBits(r.collision_rates[1 * n + idx]);
    D64 rain_snow_graupel_rate = D64::fromBits(r.collision_rates[2 * n + idx]);
    D64 rain_snow_number_rate = D64::fromBits(r.collision_rates[3 * n + idx]);
    D64 rain_graupel_rain_rate = D64::fromBits(r.collision_rates[4 * n + idx]);
    D64 rain_graupel_graupel_rate = D64::fromBits(r.collision_rates[5 * n + idx]);
    D64 rain_graupel_number_rate = D64::fromBits(r.collision_rates[6 * n + idx]);
    D64 rain_graupel_rain_number_rate = D64::fromBits(r.collision_rates[7 * n + idx]);
    D64 snow_melt_rate = D64::fromBits(state[39 * n + idx]);
    D64 snow_melt_number_rate = D64::fromBits(state[40 * n + idx]);
    D64 graupel_melt_rate = D64::fromBits(state[41 * n + idx]);
    D64 graupel_melt_number_rate = D64::fromBits(state[42 * n + idx]);
    D64 snow_vapor_rate = D64::fromBits(state[43 * n + idx]);
    D64 graupel_vapor_rate = D64::fromBits(state[44 * n + idx]);
    D64 graupel_vapor_number_rate = D64::fromBits(state[45 * n + idx]);

    if (p.dt > 120.0f) {
        rain_cloud_rate += snow_cloud_rate + graupel_cloud_rate;
        snow_cloud_rate = D64::fromBits(0x0000000000000000ul);
        graupel_cloud_rate = D64::fromBits(0x0000000000000000ul);
    }
    D64 cloud_sink = autoconversion_rate + rain_cloud_rate
        + snow_cloud_rate + graupel_cloud_rate;
    if (cloud_sink > (D64)(cloud_mass * inverse_dt)
            && cloud_sink > D64::fromBits(0x0000000000000000ul)) {
        const D64 ratio = (D64)(cloud_mass * inverse_dt) / cloud_sink;
        autoconversion_rate *= ratio;
        rain_cloud_rate *= ratio;
        snow_cloud_rate *= ratio;
        graupel_cloud_rate *= ratio;
        cloud_sink = (D64)(cloud_mass * inverse_dt);
    }
    D64 rain_sum = rain_snow_rain_rate + rain_graupel_rain_rate;
    if (rain_active && rain_sum < (D64)(-rain_mass * inverse_dt)) {
        const D64 ratio = (D64)(-rain_mass * inverse_dt) / rain_sum;
        rain_snow_rain_rate *= ratio;
        rain_graupel_rain_rate *= ratio;
    }
    D64 snow_sum = snow_vapor_rate
        + rain_snow_snow_rate - snow_melt_rate;
    if (l_qs && snow_sum < (D64)(-snow_mass * inverse_dt)) {
        const D64 ratio = (D64)(-snow_mass * inverse_dt) / snow_sum;
        snow_vapor_rate *= ratio;
        rain_snow_snow_rate *= ratio;
        snow_melt_rate *= ratio;
    }
    D64 graupel_sum = graupel_vapor_rate
        + rain_graupel_graupel_rate - graupel_melt_rate;
    if (l_qg && graupel_sum < (D64)(-graupel_mass * inverse_dt)) {
        const D64 ratio = (D64)(-graupel_mass * inverse_dt) / graupel_sum;
        graupel_vapor_rate *= ratio;
        rain_graupel_graupel_rate *= ratio;
        graupel_melt_rate *= ratio;
    }
    D64 paired;
    if (wet_bulb > 273.15f) {
        paired = dmin(dabs(rain_snow_rain_rate), dabs(rain_snow_snow_rate));
        rain_snow_rain_rate = dcopysign(paired, rain_snow_rain_rate);
        rain_snow_snow_rate = -rain_snow_rain_rate;
    }
    paired = dmin(dabs(rain_graupel_rain_rate), dabs(rain_graupel_graupel_rate));
    rain_graupel_rain_rate = dcopysign(paired, rain_graupel_rain_rate);
    rain_graupel_graupel_rate = -rain_graupel_rain_rate;
    r.snow_melt_marker[idx] = snow_melt_rate > D64::fromBits(0x0000000000000000ul) ? 1.0f : 0.0f;
    r.graupel_melt_marker[idx] = graupel_melt_rate > D64::fromBits(0x0000000000000000ul) ? 1.0f : 0.0f;
    const D64 rain_rate = autoconversion_rate + rain_cloud_rate
        + snow_melt_rate + graupel_melt_rate
        + rain_snow_rain_rate + rain_graupel_rain_rate;
    const D64 rain_number_rate = autoconversion_number_rate
        + snow_melt_number_rate + graupel_melt_number_rate
        - rain_self_number_rate - rain_snow_number_rate
        - rain_graupel_rain_number_rate;
    const D64 snow_rate = snow_vapor_rate
        + snow_cloud_rate + rain_snow_snow_rate - snow_melt_rate;
    const D64 graupel_rate = graupel_vapor_rate + graupel_cloud_rate
        + rain_snow_graupel_rate + rain_graupel_graupel_rate
        - graupel_melt_rate;
    const D64 graupel_number_rate = graupel_vapor_number_rate
        + rain_snow_number_rate - rain_graupel_number_rate
        - graupel_melt_number_rate;
    const D64 vapor_rate = snow_vapor_rate + graupel_vapor_rate;
    state[59 * n + idx] = snow_melt_rate.b;
    state[60 * n + idx] = graupel_melt_rate.b;
    state[61 * n + idx] = rain_snow_rain_rate.b;
    state[62 * n + idx] = rain_graupel_rain_rate.b;
    state[63 * n + idx] = vapor_rate.b;
    state[39 * n + idx] = cloud_sink.b;
    state[40 * n + idx] = rain_rate.b;
    state[41 * n + idx] = rain_number_rate.b;
    state[42 * n + idx] = snow_rate.b;
    state[43 * n + idx] = graupel_rate.b;
    state[44 * n + idx] = graupel_number_rate.b;
    state[45 * n + idx] = vapor_rate.b;
}

kernel void thompson_warm_frozen_source_network_apply(
    constant ThompsonWarmFrozenSourceNetworkApplyResources& r [[buffer(0)]],
    constant ThompsonWarmFrozenSourceNetworkApplyParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint n = (uint)p.n;
    state[54 * n + idx] = 0x3ff0000000000000ul;
    state[55 * n + idx] = 0ul;
    state[57 * n + idx] = 0ul;
    if (state[58 * n + idx] == 0ul) return;
    const float rho = float(D64::fromBits(state[3 * n + idx]));
    const float orho = float(D64::fromBits(state[4 * n + idx]));
    const float inverse_dt = float(D64::fromBits(state[5 * n + idx]));
    const float qv0 = float(D64::fromBits(state[2 * n + idx]));
    const D64 cloud_sink = D64::fromBits(state[39 * n + idx]);
    const D64 rain_rate = D64::fromBits(state[40 * n + idx]);
    const D64 rain_number_rate = D64::fromBits(state[41 * n + idx]);
    const D64 snow_rate = D64::fromBits(state[42 * n + idx]);
    const D64 graupel_rate = D64::fromBits(state[43 * n + idx]);
    const D64 graupel_number_rate = D64::fromBits(state[44 * n + idx]);
    const D64 vapor_rate = D64::fromBits(state[63 * n + idx]);
    const float qr_entry_wrf = r.qr[idx] > 1.0e-12f ? r.qr[idx] : 0.0f;
    const float nr_entry_wrf = r.qr[idx] > 1.0e-12f ? r.nr[idx] : 0.0f;
    const float qg_entry_wrf = r.qg[idx] > 1.0e-12f ? r.qg[idx] : 0.0f;
    const float ng_entry_wrf = r.qg[idx] > 1.0e-12f
        ? r.graupel_number_per_kg[idx] : 0.0f;
    r.qc[idx] = fmax(0.0f,
        r.qc[idx] - (float)(cloud_sink * (D64)orho) * p.dt);
    r.qr[idx] = fmax(0.0f,
        r.qr[idx] + (float)(rain_rate * (D64)orho) * p.dt);
    r.nr[idx] = fmax(0.0f,
        r.nr[idx] + (float)(rain_number_rate * (D64)orho) * p.dt);
    r.qs[idx] = fmax(0.0f,
        r.qs[idx] + (float)(snow_rate * (D64)orho) * p.dt);
    r.qg[idx] = fmax(0.0f,
        r.qg[idx] + (float)(graupel_rate * (D64)orho) * p.dt);
    r.graupel_number_per_kg[idx] +=
        (float)(graupel_number_rate * (D64)orho) * p.dt;
    r.qv[idx] = r.qv[idx] - (float)(vapor_rate * (D64)orho) * p.dt;
    r.nr[idx] = tbound_rain(r.qr[idx] * rho, rho, r.nr[idx]);
    if (!(r.qr[idx] * rho > 1.0e-12f)) {
        r.qr[idx] = qr_entry_wrf + (-qr_entry_wrf * inverse_dt) * p.dt;
        r.nr[idx] = nr_entry_wrf + (-nr_entry_wrf * inverse_dt) * p.dt;
    }
    const float xrg = r.qg[idx] * rho;
    if (!(xrg > 1.0e-12f)) {
        r.qg[idx] = qg_entry_wrf + (-qg_entry_wrf * inverse_dt) * p.dt;
        r.graupel_number_per_kg[idx] =
            ng_entry_wrf + (-ng_entry_wrf * inverse_dt) * p.dt;
    } else {
        const float am_g = 3.1415926536f * 400.0f / 6.0f;
        const float mvd_numerator = 3.0f + 0.0f + 0.672f;
        const float xng = fmax(1.0e-6f, r.graupel_number_per_kg[idx] * rho);
        D64 lamg = (D64)precise::pow(
            am_g * 6.0f * xng / xrg, 1.0f / 3.0f);
        const float mvd_g = (float)((D64)mvd_numerator / lamg);
        bool bounded = false;
        if (mvd_g > 25.4e-3f) {
            lamg = (D64)(mvd_numerator / 25.4e-3f);
            bounded = true;
        } else if (mvd_g < 50.0e-6f) {
            lamg = (D64)(mvd_numerator / 50.0e-6f);
            bounded = true;
        }
        if (bounded) {
            state[54 * n + idx] = lamg.b;
            state[55 * n + idx] = 1ul;
            state[57 * n + idx] = (ulong)as_type<uint>(ng_entry_wrf);
        }
    }
}

kernel void thompson_warm_frozen_source_network_finalize(
    constant ThompsonWarmFrozenSourceNetworkFinalizeResources& r [[buffer(0)]],
    constant ThompsonWarmFrozenSourceNetworkFinalizeParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= p.n) return;
    device ulong* state = r.cloud_state;
    const uint n = (uint)p.n;
    if (state[58 * n + idx] == 0ul) return;
    if (state[55 * n + idx] != 0ul) {
        const float rho = float(D64::fromBits(state[3 * n + idx]));
        const float inverse_dt = float(D64::fromBits(state[5 * n + idx]));
        const float xrg = r.qg[idx] * rho;
        const float ng_entry_wrf = as_type<float>((uint)state[57 * n + idx]);
        const float am_g = 3.1415926536f * 400.0f / 6.0f;
        const float xng_bounded = (float)(
            (D64)((1.0f / 6.0f) * xrg) * D64::fromBits(state[56 * n + idx])
            / (D64)am_g);
        const float ngten = (xng_bounded - ng_entry_wrf * rho)
            * inverse_dt * (1.0f / rho);
        r.graupel_number_per_kg[idx] = ng_entry_wrf + ngten * p.dt;
    }
    const float temp0 = float(D64::fromBits(state[0 * n + idx]));
    const float qv0 = float(D64::fromBits(state[2 * n + idx]));
    const float orho = float(D64::fromBits(state[4 * n + idx]));
    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
    const D64 snow_melt_rate = D64::fromBits(state[59 * n + idx]);
    const D64 graupel_melt_rate = D64::fromBits(state[60 * n + idx]);
    const D64 rain_snow_rain_rate = D64::fromBits(state[61 * n + idx]);
    const D64 rain_graupel_rain_rate = D64::fromBits(state[62 * n + idx]);
    const D64 fusion_rate = -snow_melt_rate - graupel_melt_rate
        - rain_snow_rain_rate - rain_graupel_rain_rate;
    const D64 vapor_rate = D64::fromBits(state[45 * n + idx]);
    r.temperature[idx] = temp0 + (float)(
        ((D64::fromBits(0x411462c000000000ul) * fusion_rate)
         + (D64::fromBits(0x41459f2800000000ul) * vapor_rate))
        * (D64)inverse_cp * (D64)orho * (D64)p.dt);
}

kernel void thompson_warm_frozen_source_network(
    constant ThompsonWarmFrozenSourceNetworkResources& r [[buffer(0)]],
    constant ThompsonWarmFrozenSourceNetworkParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {







    if (idx >= p.n) return;
    const bool entry_warm = r.graupel_melt_marker[idx] != 0.0f;



    r.graupel_melt_marker[idx] = 0.0f;
    r.snow_melt_marker[idx] = 0.0f;
    if (!entry_warm) return;

    const float temp0 = float(D64::fromBits(r.cloud_state[0 * p.n + idx]));
    const float pressure0 = float(D64::fromBits(r.cloud_state[1 * p.n + idx]));
    const float qv0 = float(D64::fromBits(r.cloud_state[2 * p.n + idx]));
    const float rho = float(D64::fromBits(r.cloud_state[3 * p.n + idx]));
    const float orho = float(D64::fromBits(r.cloud_state[4 * p.n + idx]));
    const float inverse_dt = float(D64::fromBits(r.cloud_state[5 * p.n + idx]));
    const float tempc = float(D64::fromBits(r.cloud_state[6 * p.n + idx]));
    const float wet_bulb = float(D64::fromBits(r.cloud_state[7 * p.n + idx]));
    const float pi = float(D64::fromBits(r.cloud_state[8 * p.n + idx]));
    const float am_r = float(D64::fromBits(r.cloud_state[9 * p.n + idx]));
    const float am_g = float(D64::fromBits(r.cloud_state[10 * p.n + idx]));
    const float rho_factor = float(D64::fromBits(r.cloud_state[11 * p.n + idx]));
    const float cloud_mass = float(D64::fromBits(r.cloud_state[12 * p.n + idx]));
    const float rain_mass = float(D64::fromBits(r.cloud_state[13 * p.n + idx]));
    const float snow_mass = float(D64::fromBits(r.cloud_state[14 * p.n + idx]));
    const float graupel_mass = float(D64::fromBits(r.cloud_state[15 * p.n + idx]));
    const bool l_qc = float(D64::fromBits(r.cloud_state[16 * p.n + idx])) != 0.0f;
    const bool l_qs = float(D64::fromBits(r.cloud_state[17 * p.n + idx])) != 0.0f;
    const bool l_qg = float(D64::fromBits(r.cloud_state[18 * p.n + idx])) != 0.0f;
    const float cloud_mvd = float(D64::fromBits(r.cloud_state[19 * p.n + idx]));
    const float rain_number = float(D64::fromBits(r.cloud_state[20 * p.n + idx]));
    D64 rain_lambda = D64::fromBits(r.cloud_state[21 * p.n + idx]);
    const float rain_mvd = float(D64::fromBits(r.cloud_state[22 * p.n + idx]));
    const bool rain_active = float(D64::fromBits(r.cloud_state[23 * p.n + idx])) != 0.0f;
    const float graupel_number = float(D64::fromBits(r.cloud_state[24 * p.n + idx]));
    D64 graupel_lambda = D64::fromBits(r.cloud_state[25 * p.n + idx]);
    const float snow_second_moment = float(D64::fromBits(r.cloud_state[26 * p.n + idx]));
    const float snow_number = float(D64::fromBits(r.cloud_state[27 * p.n + idx]));
    const float snow_first_moment = float(D64::fromBits(r.cloud_state[28 * p.n + idx]));
    const float snow_ventilation_moment = float(D64::fromBits(r.cloud_state[29 * p.n + idx]));
    D64 autoconversion_rate = D64::fromBits(r.cloud_state[30 * p.n + idx]);
    D64 autoconversion_number_rate = D64::fromBits(r.cloud_state[31 * p.n + idx]);
    D64 rain_self_number_rate = D64::fromBits(r.cloud_state[32 * p.n + idx]);
    D64 rain_cloud_rate = D64::fromBits(r.cloud_state[33 * p.n + idx]);
    D64 snow_cloud_rate = D64::fromBits(r.cloud_state[34 * p.n + idx]);
    D64 graupel_cloud_rate = D64::fromBits(r.cloud_state[35 * p.n + idx]);
    D64 rain_snow_rain_rate = D64::fromBits(r.collision_rates[0 * p.n + idx]);
    D64 rain_snow_snow_rate = D64::fromBits(r.collision_rates[1 * p.n + idx]);
    D64 rain_snow_graupel_rate = D64::fromBits(r.collision_rates[2 * p.n + idx]);
    D64 rain_snow_number_rate = D64::fromBits(r.collision_rates[3 * p.n + idx]);
    D64 rain_graupel_rain_rate = D64::fromBits(r.collision_rates[4 * p.n + idx]);
    D64 rain_graupel_graupel_rate = D64::fromBits(r.collision_rates[5 * p.n + idx]);
    D64 rain_graupel_number_rate = D64::fromBits(r.collision_rates[6 * p.n + idx]);
    D64 rain_graupel_rain_number_rate = D64::fromBits(r.collision_rates[7 * p.n + idx]);

    const float diffusivity = 2.11e-5f
        * precise::pow(fdiv_rn(temp0,273.15f), 1.94f) * (101325.0f / pressure0);
    const float viscosity = (1.718f + 0.0049f * tempc) * 1.0e-5f;
    const float conductivity = (5.69f + 0.0168f * tempc)
        * 1.0e-5f * 418.936f;
    const float rho_factor_sqrt = precise::sqrt(rho_factor);
    const float viscosity_factor = precise::sqrt(rho / viscosity);
    const float vapor_deficit = fmax(
        0.0f, trslf(pressure0, 273.15f) - qv0);
    const float inverse_fusion = 1.0f / 334000.0f;
    const float schmidt_cuberoot = precise::pow(0.632f, 1.0f / 3.0f);

    D64 snow_melt_rate = D64::fromBits(0x0000000000000000ul);
    D64 snow_melt_number_rate = D64::fromBits(0x0000000000000000ul);
    if (l_qs) {
        const float melt_moment =
            (pi * 4.0f * 0.15f * inverse_fusion * 0.86f)
                * snow_first_moment
            + (pi * 4.0f * 0.15f * inverse_fusion * 0.28f
               * schmidt_cuberoot * precise::sqrt(40.0f))
                * rho_factor_sqrt * viscosity_factor
                * snow_ventilation_moment;
        snow_melt_rate = (D64)((tempc * conductivity
            - 2.5e6f * diffusivity * vapor_deficit) * melt_moment);
        if (snow_melt_rate > D64::fromBits(0x0000000000000000ul)) {
            snow_melt_rate += D64::fromBits(0x40b07a0000000000ul) * (D64)inverse_fusion
                * (D64)(wet_bulb - 273.15f)
                * (rain_snow_rain_rate + snow_cloud_rate);
        }
        snow_melt_rate = dmin(
            (D64)(snow_mass * inverse_dt),
            dmax(D64::fromBits(0x0000000000000000ul), snow_melt_rate));
        if (snow_melt_rate > D64::fromBits(0x0000000000000000ul)) {
            snow_melt_number_rate =
                (D64)snow_number / (D64)snow_mass * snow_melt_rate
                * (D64)precise::pow(
                    10.0f, -0.25f * (wet_bulb - 273.15f));
        }
    }

    D64 graupel_melt_rate = D64::fromBits(0x0000000000000000ul);
    D64 graupel_melt_number_rate = D64::fromBits(0x0000000000000000ul);
    if (l_qg) {
        const D64 inverse_lambda = D64::fromBits(0x3ff0000000000000ul) / graupel_lambda;
        const D64 graupel_intercept =
            (D64)graupel_number * graupel_lambda;
        D64 melt_intercept = graupel_intercept;
        if (graupel_mass * graupel_number < 1.0e-4f) {
            melt_intercept = (D64)(1.0e-4f / graupel_mass)
                * graupel_lambda;
        }
        const D64 melt_moment = melt_intercept * (
            (D64)(pi * 4.0f * 0.5f * inverse_fusion * 0.86f)
                * inverse_lambda * inverse_lambda
            + (D64)(pi * 4.0f * 0.5f * inverse_fusion * 0.28f
                       * schmidt_cuberoot * precise::sqrt(442.0f)
                       * 1.9021706581115723f)
                * (D64)(rho_factor_sqrt * viscosity_factor)
                * D64::fromBits(r.cloud_state[52 * p.n + idx]));
        graupel_melt_rate = (D64)(tempc * conductivity
            - 2.5e6f * diffusivity * vapor_deficit) * melt_moment;
        graupel_melt_rate = dmin(
            (D64)(graupel_mass * inverse_dt),
            dmax(D64::fromBits(0x0000000000000000ul), graupel_melt_rate));
        if (graupel_melt_rate > D64::fromBits(0x0000000000000000ul)) {
            graupel_melt_number_rate = graupel_melt_rate
                * (D64)graupel_number / (D64)graupel_mass
                * (D64)precise::pow(
                    10.0f, -0.33f * (wet_bulb - 273.15f));
        }
    }




    const float qvsi = trslf(pressure0, temp0);
    float ssati = qv0 / qvsi - 1.0f;
    if (fabs(ssati) < 1.0e-15f) ssati = 0.0f;
    D64 snow_vapor_rate = D64::fromBits(0x0000000000000000ul);
    D64 graupel_vapor_rate = D64::fromBits(0x0000000000000000ul);
    D64 graupel_vapor_number_rate = D64::fromBits(0x0000000000000000ul);
    if (ssati < 0.0f
            && (snow_melt_rate <= D64::fromBits(0x0000000000000000ul) || graupel_melt_rate <= D64::fromBits(0x0000000000000000ul))) {
        const float inverse_temp = 1.0f / temp0;
        const float saturated_density = rho * qvsi;
        const float slope_term = inverse_temp
            * (2.834e6f * inverse_temp * (1.0f / 461.5f) - 1.0f);
        const float saturation_slope = saturated_density * slope_term;
        const float saturation_curvature = saturated_density
            * (slope_term * slope_term
               - 2.0f * 2.834e6f * inverse_temp * inverse_temp
                 * inverse_temp * (1.0f / 461.5f)
               + inverse_temp * inverse_temp);
        const float gamma = 2.834e6f * diffusivity / conductivity
            * saturation_slope;
        const float gamma_ratio = gamma / (1.0f + gamma);
        const float alpha = fmax(1.0e-9f,
            0.5f * gamma_ratio * gamma_ratio
            * saturation_curvature / saturation_slope
            * saturated_density / saturation_slope);
        const float xsat2 = ssati * ssati;
        const float alpha2 = alpha * alpha;
        const float geometry = 4.0f * pi
            * (1.0f - alpha * ssati
               + 2.0f * alpha2 * xsat2
               - 5.0f * alpha2 * alpha * xsat2 * ssati)
            / (1.0f + gamma);
        if (l_qs && snow_melt_rate <= D64::fromBits(0x0000000000000000ul)) {
            const float ventilation = 0.28f * schmidt_cuberoot
                * precise::sqrt(40.0f) * rho_factor_sqrt * viscosity_factor;
            snow_vapor_rate = (D64)(0.15f * geometry * diffusivity
                * ssati * saturated_density
                * (0.86f * snow_first_moment
                   + ventilation * snow_ventilation_moment));
            snow_vapor_rate = dmax(
                (D64)(-snow_mass * inverse_dt), snow_vapor_rate);
        }
        if (l_qg && graupel_melt_rate <= D64::fromBits(0x0000000000000000ul)) {
            const D64 inverse_lambda = D64::fromBits(0x3ff0000000000000ul) / graupel_lambda;
            const D64 graupel_intercept =
                (D64)graupel_number * graupel_lambda;
            const float ventilation = 0.28f * schmidt_cuberoot
                * precise::sqrt(442.0f) * 1.9021706581115723f
                * rho_factor_sqrt * viscosity_factor;
            graupel_vapor_rate = (D64)(0.5f * geometry * diffusivity
                * ssati * saturated_density) * graupel_intercept
                * (D64::fromBits(0x3feb851eb851eb85ul) * D64::fromBits(r.cloud_state[53 * p.n + idx])
                   + (D64)ventilation * D64::fromBits(r.cloud_state[52 * p.n + idx]));
            graupel_vapor_rate = dmax(
                (D64)(-graupel_mass * inverse_dt),
                graupel_vapor_rate);
            graupel_vapor_number_rate = graupel_vapor_rate
                * (D64)graupel_number / (D64)graupel_mass;
        }

        const D64 vapor_sum = snow_vapor_rate + graupel_vapor_rate;
        const D64 vapor_limit = (D64)((qv0 - qvsi) * rho
            * inverse_dt * 0.999f);
        if (vapor_sum < -D64::fromBits(0x3cd203af9ee75616ul) && vapor_sum < vapor_limit) {
            const D64 ratio = vapor_limit / vapor_sum;
            snow_vapor_rate *= ratio;
            graupel_vapor_rate *= ratio;


        }
    }

    if (p.dt > 120.0f) {


        rain_cloud_rate += snow_cloud_rate + graupel_cloud_rate;
        snow_cloud_rate = D64::fromBits(0x0000000000000000ul);
        graupel_cloud_rate = D64::fromBits(0x0000000000000000ul);
    }




    D64 cloud_sink = autoconversion_rate + rain_cloud_rate
        + snow_cloud_rate + graupel_cloud_rate;
    if (cloud_sink > (D64)(cloud_mass * inverse_dt)
            && cloud_sink > D64::fromBits(0x0000000000000000ul)) {
        const D64 ratio = (D64)(cloud_mass * inverse_dt) / cloud_sink;
        autoconversion_rate *= ratio;
        rain_cloud_rate *= ratio;
        snow_cloud_rate *= ratio;
        graupel_cloud_rate *= ratio;
        cloud_sink = (D64)(cloud_mass * inverse_dt);
    }

    D64 rain_sum = rain_snow_rain_rate + rain_graupel_rain_rate;
    if (rain_active
            && rain_sum < (D64)(-rain_mass * inverse_dt)) {
        const D64 ratio =
            (D64)(-rain_mass * inverse_dt) / rain_sum;
        rain_snow_rain_rate *= ratio;
        rain_graupel_rain_rate *= ratio;
    }

    D64 snow_sum = snow_vapor_rate
        + rain_snow_snow_rate - snow_melt_rate;
    if (l_qs
            && snow_sum < (D64)(-snow_mass * inverse_dt)) {
        const D64 ratio = (D64)(-snow_mass * inverse_dt) / snow_sum;
        snow_vapor_rate *= ratio;
        rain_snow_snow_rate *= ratio;
        snow_melt_rate *= ratio;
    }
    D64 graupel_sum =
        graupel_vapor_rate + rain_graupel_graupel_rate
        - graupel_melt_rate;
    if (l_qg
            && graupel_sum < (D64)(-graupel_mass * inverse_dt)) {
        const D64 ratio =
            (D64)(-graupel_mass * inverse_dt) / graupel_sum;
        graupel_vapor_rate *= ratio;
        rain_graupel_graupel_rate *= ratio;
        graupel_melt_rate *= ratio;
    }
    D64 paired;
    if (wet_bulb > 273.15f) {
        paired = dmin(
            dabs(rain_snow_rain_rate), dabs(rain_snow_snow_rate));
        rain_snow_rain_rate = dcopysign(paired, rain_snow_rain_rate);
        rain_snow_snow_rate = -rain_snow_rain_rate;
    }
    paired = dmin(
        dabs(rain_graupel_rain_rate),
        dabs(rain_graupel_graupel_rate));
    rain_graupel_rain_rate = dcopysign(
        paired, rain_graupel_rain_rate);
    rain_graupel_graupel_rate = -rain_graupel_rain_rate;




    r.snow_melt_marker[idx] = snow_melt_rate > D64::fromBits(0x0000000000000000ul) ? 1.0f : 0.0f;
    r.graupel_melt_marker[idx] = graupel_melt_rate > D64::fromBits(0x0000000000000000ul) ? 1.0f : 0.0f;

    const D64 rain_rate = autoconversion_rate + rain_cloud_rate
        + snow_melt_rate + graupel_melt_rate
        + rain_snow_rain_rate + rain_graupel_rain_rate;
    const D64 rain_number_rate = autoconversion_number_rate
        + snow_melt_number_rate + graupel_melt_number_rate
        - rain_self_number_rate - rain_snow_number_rate
        - rain_graupel_rain_number_rate;
    const D64 snow_rate = snow_vapor_rate
        + snow_cloud_rate + rain_snow_snow_rate - snow_melt_rate;
    const D64 graupel_rate = graupel_vapor_rate + graupel_cloud_rate
        + rain_snow_graupel_rate + rain_graupel_graupel_rate
        - graupel_melt_rate;
    const D64 graupel_number_rate = graupel_vapor_number_rate
        + rain_snow_number_rate - rain_graupel_number_rate
        - graupel_melt_number_rate;




    const float qr_entry_wrf = r.qr[idx] > 1.0e-12f ? r.qr[idx] : 0.0f;
    const float nr_entry_wrf = r.qr[idx] > 1.0e-12f ? r.nr[idx] : 0.0f;
    const float qg_entry_wrf = r.qg[idx] > 1.0e-12f ? r.qg[idx] : 0.0f;
    const float ng_entry_wrf = r.qg[idx] > 1.0e-12f
        ? r.graupel_number_per_kg[idx] : 0.0f;
    r.qc[idx] = fmax(0.0f,
        r.qc[idx] - (float)(cloud_sink * (D64)orho) * p.dt);
    r.qr[idx] = fmax(0.0f,
        r.qr[idx] + (float)(rain_rate * (D64)orho) * p.dt);
    r.nr[idx] = fmax(0.0f,
        r.nr[idx] + (float)(rain_number_rate * (D64)orho) * p.dt);
    r.qs[idx] = fmax(0.0f,
        r.qs[idx] + (float)(snow_rate * (D64)orho) * p.dt);
    r.qg[idx] = fmax(0.0f,
        r.qg[idx] + (float)(graupel_rate * (D64)orho) * p.dt);



    r.graupel_number_per_kg[idx] +=
        (float)(graupel_number_rate * (D64)orho) * p.dt;
    const D64 vapor_rate = snow_vapor_rate + graupel_vapor_rate;




    r.qv[idx] = r.qv[idx] - (float)(vapor_rate * (D64)orho) * p.dt;
    r.nr[idx] = tbound_rain(r.qr[idx] * rho, rho, r.nr[idx]);




    if (!(r.qr[idx] * rho > 1.0e-12f)) {
        r.qr[idx] = qr_entry_wrf + (-qr_entry_wrf * inverse_dt) * p.dt;
        r.nr[idx] = nr_entry_wrf + (-nr_entry_wrf * inverse_dt) * p.dt;
    }






    {
        const float xrg = r.qg[idx] * rho;
        if (!(xrg > 1.0e-12f)) {
            r.qg[idx] = qg_entry_wrf + (-qg_entry_wrf * inverse_dt) * p.dt;
            r.graupel_number_per_kg[idx] =
                ng_entry_wrf + (-ng_entry_wrf * inverse_dt) * p.dt;
        } else {

            const float am_g = 3.1415926536f * 400.0f / 6.0f;
            const float mvd_numerator = 3.0f + 0.0f + 0.672f;
            const float xng = fmax(1.0e-6f, r.graupel_number_per_kg[idx] * rho);
            D64 lamg = (D64)precise::pow(
                am_g * 6.0f * xng / xrg, 1.0f / 3.0f);
            const float mvd_g = (float)((D64)mvd_numerator / lamg);
            bool bounded = false;
            if (mvd_g > 25.4e-3f) {
                lamg = (D64)(mvd_numerator / 25.4e-3f);
                bounded = true;
            } else if (mvd_g < 50.0e-6f) {
                lamg = (D64)(mvd_numerator / 50.0e-6f);
                bounded = true;
            }
            if (bounded) {
                const float xng_bounded = (float)(
                    (D64)((1.0f / 6.0f) * xrg) * dpow(lamg, D64::fromBits(0x4008000000000000ul))
                    / (D64)am_g);
                const float ngten = (xng_bounded - ng_entry_wrf * rho)
                    * inverse_dt * (1.0f / rho);
                r.graupel_number_per_kg[idx] = ng_entry_wrf + ngten * p.dt;
            }
        }
    }

    const float inverse_cp = 1.0f
        / (1004.0f * (1.0f + 0.887f * qv0));
    const D64 fusion_rate = -snow_melt_rate - graupel_melt_rate
        - rain_snow_rain_rate - rain_graupel_rain_rate;
    r.temperature[idx] = temp0 + (float)(
        ((D64::fromBits(0x411462c000000000ul) * fusion_rate) + (D64::fromBits(0x41459f2800000000ul) * vapor_rate))
        * (D64)inverse_cp * (D64)orho * (D64)p.dt);
}
