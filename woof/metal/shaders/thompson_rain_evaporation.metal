#include <metal_stdlib>
using namespace metal;
#include "thompson_rain_evaporation_with_density_history_and_presence_bindings.metalh"
#include "soft_binary64.metalh"

// Thompson classic mp=8 Srivastava-Coen rain evaporation. CUDA's float
// operation order and branch/marker behavior are retained. Its explicit
// binary64 islands use soft64 below; precise Metal FP32 libm is not asserted
// bit-identical to CUDA's device libm.
static inline float rslf(float pressure, float temp) {
    const float c0=0.611583699e3f, c1=0.444606896e2f;
    const float c2=0.143177157e1f, c3=0.264224321e-1f;
    const float c4=0.299291081e-3f, c5=0.203154182e-5f;
    const float c6=0.702620698e-8f, c7=0.379534310e-11f;
    const float c8=-0.321582393e-13f;
    const float x=fmax(-80.0f,temp-273.16f);
    float esl=c0+x*(c1+x*(c2+x*(c3+x*(c4+x*(c5+x*(c6+x*(c7+x*c8)))))));
    esl=fmin(esl,pressure*0.15f);
    return 0.622f*esl/(pressure-esl);
}
static inline ulong dmul(ulong a, ulong b) { return soft64_mul(a,b); }
static inline ulong ddiv(ulong a, ulong b) { return soft64_div(a,b); }
static inline ulong dadd(ulong a, ulong b) { return soft64_add(a,b); }
static inline ulong dfromf(float x) { return soft64_from_f32_bits(as_type<uint>(x)); }
static inline float dtof(ulong x) { return as_type<float>(soft64_to_f32_bits(x)); }
static inline ulong dmin(ulong a, ulong b) {
    if (soft64_is_nan(a)) return b;
    if (soft64_is_nan(b)) return a;
    return soft64_less(a,b) ? a : b;
}
static inline ulong pow_minus_three(ulong x) {
    ulong cube=dmul(dmul(x,x),x);
    return ddiv(0x3ff0000000000000ul,cube);
}

kernel void thompson_rain_evaporation_with_density_history_and_presence(
    constant ThompsonRainEvaporationWithDensityHistoryAndPresenceResources& r [[buffer(0)]],
    constant ThompsonRainEvaporationWithDensityHistoryAndPresenceParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    uint n=uint(p.nz*p.ny*p.nx);
    if (idx>=n) return;
    const float temp0=r.temperature[idx];
    const float qv0=fmax(1.0e-10f,r.qv[idx]);
    const float rho=0.622f*r.pressure[idx]
        /(287.04f*temp0*(qv0+0.622f));
    const bool melting_graupel=r.graupel_melt_marker[idx]!=0.0f;
    const float rain_density=r.source_density[idx];
    r.reference_density[idx]=rain_density;
    if (r.qr[idx]<=1.0e-12f) {
        r.reference_density[idx]=0.0f;
        return;
    }
    if (r.condensation_marker[idx]!=0.0f) return;

    const float qvs=rslf(r.pressure[idx],temp0);
    float ssatw=qv0/qvs-1.0f;
    if (fabs(ssatw)<1.0e-15f) ssatw=0.0f;
    if (ssatw>=-1.0e-15f) return;

    const float orho=1.0f/rho;
    const float rr=r.qr[idx]*rain_density;
    const float rain_number=fmax(1.0e-6f,r.nr[idx]*rain_density);
    const float pi=3.1415926536f;
    const float am_r=pi*1000.0f/6.0f;
    const float lambda_f=pow(am_r*6.0f*rain_number/rr,1.0f/3.0f);
    const ulong lambda=dfromf(lambda_f);
    const ulong inverse_lambda=ddiv(0x3ff0000000000000ul,lambda);
    const ulong intercept=dmul(dfromf(rain_number),lambda);

    const float tempc=temp0-273.15f;
    const float inverse_temp=1.0f/temp0;
    const float rho_factor=sqrt((101325.0f/(287.05f*298.0f))*orho);
    const float rho_factor_sqrt=sqrt(rho_factor);
    const float diffusivity=2.11e-5f*pow(temp0/273.15f,1.94f)
        *(101325.0f/r.pressure[idx]);
    const float viscosity=tempc>=0.0f
        ? (1.718f+0.0049f*tempc)*1.0e-5f
        : (1.718f+0.0049f*tempc-1.2e-5f*tempc*tempc)*1.0e-5f;
    const float viscosity_factor=sqrt(rho/viscosity);
    const float latent_heat=2.5e6f+(2106.0f-4218.0f)*tempc;
    const float conductivity=(5.69f+0.0168f*tempc)*1.0e-5f*418.936f;
    const float inverse_cp=1.0f/(1004.0f*(1.0f+0.887f*qv0));

    const float saturated_density=rho*qvs;
    const float saturation_slope=saturated_density*inverse_temp
        *(latent_heat*inverse_temp*(1.0f/461.5f)-1.0f);
    const float slope_term=inverse_temp
        *(latent_heat*inverse_temp*(1.0f/461.5f)-1.0f);
    const float saturation_curvature=saturated_density
        *(slope_term*slope_term
          +(-2.0f*latent_heat*inverse_temp*inverse_temp*inverse_temp
            *(1.0f/461.5f))
          +inverse_temp*inverse_temp);
    const float gamma=latent_heat*diffusivity/conductivity*saturation_slope;
    const float gamma_ratio=gamma/(1.0f+gamma);
    const float alpha=fmax(1.0e-9f,
        0.5f*gamma_ratio*gamma_ratio*saturation_curvature/saturation_slope
        *saturated_density/saturation_slope);
    const float xsat=fmin(-1.0e-9f,ssatw);
    const float xsat2=xsat*xsat;
    const float alpha2=alpha*alpha;
    const float evaporation_geometry=2.0f*pi
        *(1.0f-alpha*xsat+2.0f*alpha2*xsat2
          -5.0f*alpha2*alpha*xsat2*xsat)/(1.0f+gamma);

    const float schmidt_cuberoot=pow(0.632f,1.0f/3.0f);
    const float ventilation_two=0.308f*schmidt_cuberoot*sqrt(4854.0f)*2.0f;
    const float rate_prefix=evaporation_geometry*diffusivity*(-ssatw);
    const ulong diffusion_term=dmul(0x3fe8f5c28f5c28f6ul,
                                     dmul(inverse_lambda,inverse_lambda));
    const ulong ventilation_x=dadd(lambda,0x4058600000000000ul); // 97.5
    const float ventilation_prefix=ventilation_two*viscosity_factor*rho_factor_sqrt;
    const ulong ventilation_term=dmul(dfromf(ventilation_prefix),
                                       pow_minus_three(ventilation_x));
    ulong evaporation_rate=dmul(dmul(dfromf(rate_prefix),intercept),
        dmul(dfromf(saturated_density),dadd(diffusion_term,ventilation_term)));

    const float inverse_dt=1.0f/p.dt;
    if (qv0/qvs<0.95f && rr*orho<=1.0e-8f) {
        evaporation_rate=dfromf(rr*orho*inverse_dt);
    } else {
        const float maximum_rate=fmin(rr*orho*inverse_dt,(qvs-qv0)*inverse_dt);
        evaporation_rate=dmin(dfromf(maximum_rate),
                              dmul(evaporation_rate,dfromf(orho)));
        if (melting_graupel) {
            const float evaporation_factor=fmin(1.0f,
                0.01f+(0.99f-0.01f)*(tempc/20.0f));
            evaporation_rate=dmul(evaporation_rate,dfromf(evaporation_factor));
        }
    }
    const ulong number_rate=dmin(
        dfromf(rain_number*0.99f*orho*inverse_dt),
        ddiv(dmul(evaporation_rate,dfromf(rain_number)),dfromf(rr)));
    const float qr_tendency=dtof(soft64_negate_bits(evaporation_rate));
    const float qv_tendency=dtof(evaporation_rate);
    const float nr_tendency=dtof(soft64_negate_bits(number_rate));
    const float thermal_factor=latent_heat*inverse_cp;
    const float temperature_tendency=dtof(
        soft64_negate_bits(dmul(dfromf(thermal_factor),evaporation_rate)));

    r.reference_density[idx]=-rho;
    r.qr[idx]+=qr_tendency*p.dt;
    r.qv[idx]=r.qv[idx]+qv_tendency*p.dt;
    r.nr[idx]+=nr_tendency*p.dt;
    r.temperature[idx]=temp0+temperature_tendency*p.dt;
}
