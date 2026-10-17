// Production MYNN default-family kernels and ordered driver stages.
#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_log.metalh"
#include "soft_gfk_pow.metalh"
#include "soft_gfk_atan.metalh"
#include "mynn_driver_surface_columns_bindings.metalh"
#include "mynn_driver_surface_gsd41_columns_bindings.metalh"
#include "mynn_predict_default_columns_bindings.metalh"
#include "mynn_spp_diffusivity_bindings.metalh"
#include "mynn_driver_prep_columns_bindings.metalh"
#include "mynn_wrapper_to_specific_bindings.metalh"
#include "mynn_wrapper_from_specific_bindings.metalh"
#include "mynn_driver_exchange_columns_bindings.metalh"
#include "mynn_level2_pairs_bindings.metalh"
#include "mynn_level2_columns_bindings.metalh"
#include "mynn_pblh_scale_columns_bindings.metalh"
#include "mynn_turbulence_default_interfaces_bindings.metalh"
#include "mynn_condensation_default_columns_bindings.metalh"
#include "mynn_driver_diss_heat_columns_bindings.metalh"
#include "mynn_mixlength_default_columns_bindings.metalh"
#include "mynn_mixlength_gsd41_columns_bindings.metalh"
#include "mynn_initialize_default_columns_bindings.metalh"
#include "mynn_initialize_gsd41_columns_bindings.metalh"
#include "mynn_tendencies_columns_bindings.metalh"
#include "mynn_dmp_mf_columns_bindings.metalh"
#include "mynn_dmp_mf_gsd41_columns_bindings.metalh"
#include "mynn_condensation_gsd41_columns_bindings.metalh"
#include "mynn_scalar_pblmix_columns_bindings.metalh"
#include "mynn_mix_scalar_columns_bindings.metalh"
#include "mynn_pblh_scale_gsd41_columns_bindings.metalh"
#include "mynn_turbulence_gsd41_interfaces_bindings.metalh"
#include "mynn_predict_gsd41_columns_bindings.metalh"
#include "mynn_driver_diss_heat_gsd41_columns_bindings.metalh"
#include "mynn_tendencies_gsd41_columns_bindings.metalh"
#include "mynn_gsd41_thvl_columns_bindings.metalh"
#include "mynn_gsd41_cloud_decay_bindings.metalh"

using namespace metal;

#pragma clang fp contract(off)

static inline float mynn_f32_add(float a, float b)
{
    return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)));
}

static inline float mynn_f32_sub(float a, float b)
{
    return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b)));
}

static inline float mynn_f32_mul(float a, float b)
{
    return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b)));
}

static inline float mynn_f32_div(float a, float b)
{
    return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)));
}

static inline float mynn_f32_sqrt(float a)
{
    return as_type<float>(soft32_sqrt(as_type<uint>(a)));
}

static inline ulong mynn_f64_from_f32(float a)
{
    return soft64_from_f32_bits(as_type<uint>(a));
}

static inline float mynn_f64_to_f32(ulong a)
{
    return as_type<float>(soft64_to_f32_bits(a));
}

static inline ulong mynn_f64_negate(ulong a)
{
    return a ^ 0x8000000000000000ul;
}

static inline float mynn_f32_max2(float a, float b)
{
    return soft32_greater(as_type<uint>(b), as_type<uint>(a)) ? b : a;
}

static inline float mynn_f32_min2(float a, float b)
{
    return soft32_greater(as_type<uint>(a), as_type<uint>(b)) ? b : a;
}

static inline float mynn_f32_negate(float value)
{
    return as_type<float>(as_type<uint>(value) ^ 0x80000000u);
}

static inline float mynn_gfk_logf(float value)
{
    return as_type<float>(soft_gfk_log(as_type<uint>(value)));
}

static inline float mynn_gfk_atanf(float value)
{
    return as_type<float>(soft_gfk_atanf(as_type<uint>(value)));
}

static inline float mynn_gfk_expf(float value)
{
    return as_type<float>(soft_gfk_exp(as_type<uint>(value)));
}

static inline float mynn_gfk_powf(float x, float y)
{
    return as_type<float>(soft_gfk_pow_bits(as_type<uint>(x),
                                            as_type<uint>(y)));
}

static inline bool mynn_stfunc_stable_arm(float zet)
{
    const uint bits = as_type<uint>(zet);
    if ((bits << 1u) > 0xff000000u) return false;
    if ((bits << 1u) == 0u) return true;
    return (bits & 0x80000000u) == 0u;
}

static inline float mynn_phi_stable(float zet, float a_st, float b_st,
                                    float rb_st)
{
    const float dummy_0 = mynn_f32_add(1.0f, mynn_gfk_powf(zet, b_st));
    const float dummy_1 = mynn_f32_add(
        zet, mynn_gfk_powf(dummy_0, rb_st));
    const float dummy_11 = mynn_f32_add(1.0f, mynn_f32_mul(
        mynn_gfk_powf(dummy_0, mynn_f32_sub(rb_st, 1.0f)),
        mynn_gfk_powf(zet, mynn_f32_sub(b_st, 1.0f))));
    const float dummy_2 = mynn_f32_mul(
        mynn_f32_div(mynn_f32_negate(a_st), dummy_1), dummy_11);
    return mynn_f32_sub(1.0f, mynn_f32_mul(zet, dummy_2));
}

static inline float mynn_phi_unstable(float zet, float phi,
                                      float dummy_psi, float a_unst)
{
    float dummy_0 = mynn_f32_sub(
        1.0f, mynn_f32_mul(a_unst, zet));
    const float dummy_1 = mynn_gfk_powf(dummy_0, 0.333333f);
    const float dummy_11 = mynn_f32_mul(
        mynn_f32_mul(mynn_f32_negate(0.33333f), a_unst),
        mynn_gfk_powf(dummy_0, -0.6666667f));
    const float dummy_2 = mynn_f32_mul(0.33333f, mynn_f32_add(
        mynn_f32_add(mynn_gfk_powf(dummy_1, 2.0f), dummy_1), 1.0f));
    const float dummy_22 = mynn_f32_mul(
        mynn_f32_mul(0.3333f, dummy_11),
        mynn_f32_add(mynn_f32_mul(2.0f, dummy_1), 1.0f));
    const float dummy_3 = mynn_f32_mul(
        0.57735f, mynn_f32_add(mynn_f32_mul(2.0f, dummy_1), 1.0f));
    const float dummy_33 = mynn_f32_mul(1.1547f, dummy_11);
    const float dummy_4 = mynn_f32_add(mynn_f32_sub(
        mynn_f32_mul(1.5f, mynn_gfk_logf(dummy_2)),
        mynn_f32_mul(1.73205f, mynn_gfk_atanf(dummy_3))), 1.813799364f);
    const float dummy_44 = mynn_f32_sub(
        mynn_f32_mul(mynn_f32_div(1.5f, dummy_2), dummy_22),
        mynn_f32_div(mynn_f32_mul(1.73205f, dummy_33),
            mynn_f32_add(1.0f, mynn_f32_mul(dummy_3, dummy_3))));

    dummy_0 = mynn_f32_mul(zet, zet);
    const float dummy_1b = mynn_f32_div(1.0f, mynn_f32_add(1.0f, dummy_0));
    const float dummy_11b = mynn_f32_mul(2.0f, zet);
    float dummy_2b = mynn_f32_add(
        mynn_f32_add(mynn_f32_div(
            mynn_f32_sub(1.0f, phi), zet),
            mynn_f32_mul(dummy_11b, dummy_4)),
        mynn_f32_mul(dummy_0, dummy_44));
    dummy_2b = mynn_f32_mul(dummy_2b, dummy_1b);
    const float dummy_22b = mynn_f32_mul(
        mynn_f32_negate(mynn_f32_mul(dummy_11b,
            mynn_f32_add(dummy_psi, mynn_f32_mul(dummy_0, dummy_4)))),
        mynn_f32_mul(dummy_1b, dummy_1b));
    return mynn_f32_sub(1.0f,
        mynn_f32_mul(zet, mynn_f32_add(dummy_2b, dummy_22b)));
}

static inline float mynn_phim(float zet)
{
    if (mynn_stfunc_stable_arm(zet))
        return mynn_phi_stable(zet, 6.1f, 2.5f,
                               mynn_f32_div(1.0f, 2.5f));
    const float dummy_0 = mynn_gfk_powf(
        mynn_f32_sub(1.0f, mynn_f32_mul(16.0f, zet)), 0.25f);
    const float phi_m = mynn_f32_div(1.0f, dummy_0);
    const float psi_first_log = mynn_f32_mul(2.0f, mynn_gfk_logf(
        mynn_f32_mul(0.5f, mynn_f32_add(1.0f, dummy_0))));
    const float psi_second_log = mynn_gfk_logf(mynn_f32_mul(
        0.5f, mynn_f32_add(1.0f, mynn_f32_mul(dummy_0, dummy_0))));
    const float psi_atan = mynn_f32_mul(2.0f, mynn_gfk_atanf(dummy_0));
    const float dummy_psi = mynn_f32_add(
        mynn_f32_sub(mynn_f32_add(psi_first_log, psi_second_log), psi_atan),
        1.570796f);
    return mynn_phi_unstable(zet, phi_m, dummy_psi, 10.0f);
}

static inline float mynn_phih(float zet)
{
    if (mynn_stfunc_stable_arm(zet))
        return mynn_phi_stable(zet, 5.3f, 1.1f,
                               mynn_f32_div(1.0f, 1.1f));
    const float dummy_0 = mynn_gfk_powf(
        mynn_f32_sub(1.0f, mynn_f32_mul(16.0f, zet)), 0.5f);
    const float phh = mynn_f32_div(1.0f, dummy_0);
    const float dummy_psi = mynn_f32_mul(2.0f, mynn_gfk_logf(
        mynn_f32_mul(0.5f, mynn_f32_add(1.0f, dummy_0))));
    return mynn_phi_unstable(zet, phh, dummy_psi, 34.0f);
}

static inline float mynn_esat_liquid(float xc)
{
    float value = -0.321582393e-13f;
    value = mynn_f32_add(0.379534310e-11f, mynn_f32_mul(xc, value));
    value = mynn_f32_add(0.702620698e-8f, mynn_f32_mul(xc, value));
    value = mynn_f32_add(0.203154182e-5f, mynn_f32_mul(xc, value));
    value = mynn_f32_add(0.299291081e-3f, mynn_f32_mul(xc, value));
    value = mynn_f32_add(0.264224321e-1f, mynn_f32_mul(xc, value));
    value = mynn_f32_add(0.143177157e1f, mynn_f32_mul(xc, value));
    value = mynn_f32_add(0.444606896e2f, mynn_f32_mul(xc, value));
    return mynn_f32_add(0.611583699e3f, mynn_f32_mul(xc, value));
}

static inline float mynn_esat_ice(float xc)
{
    float value = 0.161444444e-12f;
    value = mynn_f32_add(0.105785160e-9f, mynn_f32_mul(xc, value));
    value = mynn_f32_add(0.307839583e-7f, mynn_f32_mul(xc, value));
    value = mynn_f32_add(0.521693933e-5f, mynn_f32_mul(xc, value));
    value = mynn_f32_add(0.565392987e-3f, mynn_f32_mul(xc, value));
    value = mynn_f32_add(0.402737184e-1f, mynn_f32_mul(xc, value));
    value = mynn_f32_add(0.184672631e1f, mynn_f32_mul(xc, value));
    value = mynn_f32_add(0.499320233e2f, mynn_f32_mul(xc, value));
    return mynn_f32_add(0.609868993e3f, mynn_f32_mul(xc, value));
}

static inline float mynn_qsat_blend(float t, float p)
{
    const float t0c = 273.15f, tice = 240.0f;
    const float t0cm6 = mynn_f32_sub(t0c, 6.0f);
    const float xc = mynn_f32_max2(-80.0f, mynn_f32_sub(t, t0c));
    const float ceiling = mynn_f32_mul(p, 0.15f);
    if (t >= t0cm6) {
        const float esl = mynn_f32_min2(mynn_esat_liquid(xc), ceiling);
        return mynn_f32_div(mynn_f32_mul(0.622f, esl),
            mynn_f32_max2(mynn_f32_sub(p, esl), 1.0e-5f));
    }
    if (t <= tice) {
        const float esi = mynn_f32_min2(mynn_esat_ice(xc), ceiling);
        return mynn_f32_div(mynn_f32_mul(0.622f, esi),
            mynn_f32_max2(mynn_f32_sub(p, esi), 1.0e-5f));
    }
    const float esl = mynn_f32_min2(mynn_esat_liquid(xc), ceiling);
    const float esi = mynn_f32_min2(mynn_esat_ice(xc), ceiling);
    const float rslf = mynn_f32_div(mynn_f32_mul(0.622f, esl),
        mynn_f32_max2(mynn_f32_sub(p, esl), 1.0e-5f));
    const float rsif = mynn_f32_div(mynn_f32_mul(0.622f, esi),
        mynn_f32_max2(mynn_f32_sub(p, esi), 1.0e-5f));
    const float chi = mynn_f32_div(
        mynn_f32_sub(t0cm6, t), mynn_f32_sub(t0cm6, tice));
    return mynn_f32_add(mynn_f32_mul(mynn_f32_sub(1.0f, chi), rslf),
                        mynn_f32_mul(chi, rsif));
}

static inline float mynn_xl_blend(float t)
{
    const float t0c = 273.15f, tice = 240.0f;
    const float cpv = mynn_f32_mul(4.0f, 461.6f);
    const float cpv_cliq = mynn_f32_sub(cpv, 4190.0f);
    const float cpv_cice = mynn_f32_sub(cpv, 2106.0f);
    if (t >= t0c) return mynn_f32_add(2.5e6f,
        mynn_f32_mul(cpv_cliq, mynn_f32_sub(t, t0c)));
    if (t <= tice) return mynn_f32_add(2.85e6f,
        mynn_f32_mul(cpv_cice, mynn_f32_sub(t, t0c)));
    const float xlvt = mynn_f32_add(2.5e6f,
        mynn_f32_mul(cpv_cliq, mynn_f32_sub(t, t0c)));
    const float xlst = mynn_f32_add(2.85e6f,
        mynn_f32_mul(cpv_cice, mynn_f32_sub(t, t0c)));
    const float chi = mynn_f32_div(
        mynn_f32_sub(t0c, t), mynn_f32_sub(t0c, tice));
    return mynn_f32_add(mynn_f32_mul(mynn_f32_sub(1.0f, chi), xlvt),
                        mynn_f32_mul(chi, xlst));
}

static inline float mynn_gsd41_poly(float x, thread const float *c)
{
    float value = mynn_f32_add(c[7], mynn_f32_mul(x, c[8]));
    for (int i = 6; i >= 0; --i)
        value = mynn_f32_add(c[i], mynn_f32_mul(x, value));
    return value;
}

static inline float mynn_gsd41_esl(float xc)
{
    const float c[9] = {0.611583699e3f, 0.444606896e2f, 0.143177157e1f,
        0.264224321e-1f, 0.299291081e-3f, 0.203154182e-5f,
        0.702620698e-8f, 0.379534310e-11f, -0.321582393e-13f};
    return mynn_gsd41_poly(xc, c);
}

static inline float mynn_gsd41_esi(float xc)
{
    const float c[9] = {0.609868993e3f, 0.499320233e2f, 0.184672631e1f,
        0.402737184e-1f, 0.565392987e-3f, 0.521693933e-5f,
        0.307839583e-7f, 0.105785160e-9f, 0.161444444e-12f};
    return mynn_gsd41_poly(xc, c);
}

static inline float mynn_gsd41_esat_blend(float t)
{
    const float xc = mynn_f32_max2(-80.0f, mynn_f32_sub(t, 273.16f));
    if (t >= 273.16f) return mynn_gsd41_esl(xc);
    if (t <= 253.0f) return mynn_gsd41_esi(xc);
    const float chi = mynn_f32_div(mynn_f32_sub(273.16f, t), 20.16f);
    return mynn_f32_add(mynn_f32_mul(mynn_f32_sub(1.0f, chi),
                                     mynn_gsd41_esl(xc)),
                        mynn_f32_mul(chi, mynn_gsd41_esi(xc)));
}

static inline float mynn_gsd41_qsat_blend(float t, float p)
{
    const float xc = mynn_f32_max2(-80.0f, mynn_f32_sub(t, 273.16f));
    if (t >= 273.16f) {
        const float esl = mynn_gsd41_esl(xc);
        return mynn_f32_div(mynn_f32_mul(0.622f, esl), mynn_f32_sub(p, esl));
    }
    if (t <= 253.0f) {
        const float esi = mynn_gsd41_esi(xc);
        return mynn_f32_div(mynn_f32_mul(0.622f, esi), mynn_f32_sub(p, esi));
    }
    const float esl = mynn_gsd41_esl(xc), esi = mynn_gsd41_esi(xc);
    const float rslf = mynn_f32_div(mynn_f32_mul(0.622f, esl), mynn_f32_sub(p, esl));
    const float rsif = mynn_f32_div(mynn_f32_mul(0.622f, esi), mynn_f32_sub(p, esi));
    const float chi = mynn_f32_div(mynn_f32_sub(273.16f, t), 20.16f);
    return mynn_f32_add(mynn_f32_mul(mynn_f32_sub(1.0f, chi), rslf),
                        mynn_f32_mul(chi, rsif));
}

static inline float mynn_gsd41_xl_blend(float t)
{
    const float cpv = mynn_f32_mul(4.0f, 461.6f);
    const float cpv_cliq = mynn_f32_sub(cpv, 4190.0f);
    const float cpv_cice = mynn_f32_sub(cpv, 2106.0f);
    const float dt = mynn_f32_sub(t, 273.16f);
    if (t >= 273.16f) return mynn_f32_add(2.5e6f, mynn_f32_mul(cpv_cliq, dt));
    if (t <= 253.0f) return mynn_f32_add(2.85e6f, mynn_f32_mul(cpv_cice, dt));
    const float xlvt = mynn_f32_add(2.5e6f, mynn_f32_mul(cpv_cliq, dt));
    const float xlst = mynn_f32_add(2.85e6f, mynn_f32_mul(cpv_cice, dt));
    const float chi = mynn_f32_div(mynn_f32_sub(273.16f, t), 20.16f);
    return mynn_f32_add(mynn_f32_mul(mynn_f32_sub(1.0f, chi), xlvt),
                        mynn_f32_mul(chi, xlst));
}

kernel void mynn_driver_surface_columns(
    constant MynnDriverSurfaceColumnsResources &r [[buffer(0)]],
    constant MynnDriverSurfaceColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    if (column >= uint(q.ncol) || q.nz <= 0) return;
    const float xlvcp = mynn_f32_div(2500000.0f, 1004.5f);
    const float p608 = mynn_f32_sub(
        mynn_f32_div(461.6f, 287.0f), 1.0f);
    const float kgtr = mynn_f32_mul(
        0.4f, mynn_f32_div(9.81f, 300.0f));

    const float rho0 = r.rho[column];
    const float exner0 = r.exner[column];
    const float ustar_value = r.ust[column];
    const float cpm = mynn_f32_mul(
        1004.5f, mynn_f32_add(1.0f, mynn_f32_mul(0.84f, r.qv1[column])));
    const float qflux = mynn_f32_div(r.qfx[column], rho0);
    const float cflux = as_type<float>(0u);
    const float surface_theta = mynn_f32_div(r.ts[column], exner0);
    const float tflux = mynn_f32_sub(
        mynn_f32_div(r.hfx[column], mynn_f32_mul(rho0, cpm)),
        mynn_f32_div(mynn_f32_mul(xlvcp, cflux), exner0));
    const float ustar3 = mynn_f32_mul(
        mynn_f32_mul(ustar_value, ustar_value), ustar_value);
    const float buoyancy_flux = mynn_f32_add(
        tflux, mynn_f32_mul(mynn_f32_mul(qflux, p608), surface_theta));
    const float inverse_l = mynn_f32_negate(mynn_f32_div(
        mynn_f32_mul(kgtr, buoyancy_flux),
        mynn_f32_max2(ustar3, 1.0e-6f)));
    const float stability = mynn_f32_mul(
        mynn_f32_mul(0.5f, r.dz[column]), inverse_l);

    r.flqv[column] = qflux;
    r.flqc[column] = cflux;
    r.flq[column] = mynn_f32_add(qflux, cflux);
    r.flt[column] = tflux;
    r.fltv[column] = mynn_f32_add(tflux,
        mynn_f32_mul(mynn_f32_mul(qflux, p608), surface_theta));
    r.th_sfc[column] = surface_theta;
    r.rmol[column] = inverse_l;
    const float clamped = mynn_f32_min2(
        mynn_f32_max2(stability, -20.0f), 20.0f);
    r.zet[column] = clamped;
    r.pmz[column] = mynn_f32_sub(mynn_phim(clamped), clamped);
    r.phh[column] = mynn_phih(clamped);
}

// GSD v4.1 preserves the surface-layer 1/L supplied by SFCLAY and uses its
// own Kansas stability forms, as in module_bl_mynn.F:4412-4419.
kernel void mynn_driver_surface_gsd41_columns(
    constant MynnDriverSurfaceGsd41ColumnsResources &r [[buffer(0)]],
    constant MynnDriverSurfaceGsd41ColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    if (column >= uint(q.ncol) || q.nz <= 0) return;
    const float xlvcp = mynn_f32_div(2500000.0f, 1004.5f);
    const float p608 = mynn_f32_sub(mynn_f32_div(461.6f, 287.0f), 1.0f);
    const float rho0 = r.rho[column];
    const float exner0 = r.exner[column];
    const float cpm = mynn_f32_mul(1004.5f,
        mynn_f32_add(1.0f, mynn_f32_mul(0.84f, r.qv1[column])));
    const float qflux = mynn_f32_div(r.qfx[column], rho0);
    const float surface_theta = mynn_f32_div(r.ts[column], exner0);
    const float tflux = mynn_f32_sub(
        mynn_f32_div(r.hfx[column], mynn_f32_mul(rho0, cpm)),
        mynn_f32_div(mynn_f32_mul(xlvcp, as_type<float>(0u)), exner0));
    const float fltv = mynn_f32_add(tflux,
        mynn_f32_mul(mynn_f32_mul(qflux, p608), surface_theta));
    const float z = mynn_f32_mul(mynn_f32_mul(0.5f, r.dz[column]),
                                  r.rmol_in[column]);
    r.flqv[column] = qflux;
    r.flqc[column] = as_type<float>(0u);
    r.flq[column] = qflux;
    r.flt[column] = tflux;
    r.fltv[column] = fltv;
    r.th_sfc[column] = surface_theta;
    r.rmol[column] = r.rmol_in[column];
    r.zet[column] = z;
    if (z >= 0.0f) {
        r.pmz[column] = mynn_f32_add(1.0f, mynn_f32_mul(4.0f, z));
        r.phh[column] = mynn_f32_add(1.0f, mynn_f32_mul(5.0f, z));
    } else {
        const float stability = mynn_f32_sub(1.0f,
            mynn_f32_mul(16.0f, z));
        r.pmz[column] = mynn_f32_sub(
            mynn_f32_div(1.0f, mynn_gfk_powf(stability, 0.25f)), z);
        r.phh[column] = mynn_f32_div(1.0f,
            mynn_f32_sqrt(stability));
    }
}

kernel void mynn_wrapper_to_specific(
    constant MynnWrapperToSpecificResources &r [[buffer(0)]],
    constant MynnWrapperToSpecificParams &q [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= uint(q.count)) return;
    const float denominator = mynn_f32_add(1.0f, r.qv[index]);
    r.sqv[index] = mynn_f32_div(r.qv[index], denominator);
    r.sqc[index] = mynn_f32_div(r.qc[index], denominator);
    r.sqi[index] = mynn_f32_div(r.qi[index], denominator);
    r.sqs[index] = mynn_f32_div(r.qs[index], denominator);
}

kernel void mynn_wrapper_from_specific(
    constant MynnWrapperFromSpecificResources &r [[buffer(0)]],
    constant MynnWrapperFromSpecificParams &q [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= uint(q.count)) return;
    const float denominator = mynn_f32_sub(1.0f, r.sqv[index]);
    r.rqvblten[index] = mynn_f32_div(r.rqvblten[index], denominator);
    r.rqcblten[index] = mynn_f32_div(r.rqcblten[index], denominator);
    r.rqiblten[index] = mynn_f32_div(r.rqiblten[index], denominator);
    r.qc_bl[index] = mynn_f32_div(r.qc_bl[index], denominator);
    r.qi_bl[index] = mynn_f32_div(r.qi_bl[index], denominator);
}

kernel void mynn_driver_exchange_columns(
    constant MynnDriverExchangeColumnsResources &r [[buffer(0)]],
    constant MynnDriverExchangeColumnsParams &q [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= uint(q.count)) return;
    const uint leading_dimension = uint(q.count) / uint(q.nz);
    const uint level = index / leading_dimension;
    if (level == 0u) {
        r.k_m[index] = as_type<float>(0u);
        r.k_h[index] = as_type<float>(0u);
        return;
    }
    const float dzk = mynn_f32_mul(
        0.5f, mynn_f32_add(r.dz[index], r.dz[index - leading_dimension]));
    r.k_m[index] = mynn_f32_mul(r.dfm[index], dzk);
    r.k_h[index] = mynn_f32_mul(r.dfh[index], dzk);
}

kernel void mynn_driver_prep_columns(
    constant MynnDriverPrepColumnsResources &r [[buffer(0)]],
    constant MynnDriverPrepColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    if (column >= uint(q.ncol)) return;

    // Recreate the CUDA compile-time constants through the shared RN
    // operations. Their source compiler-folded bit patterns remain unverified.
    const float xlvcp = mynn_f32_div(2500000.0f, 1004.5f);
    const float xlscp = mynn_f32_div(
        mynn_f32_add(2500000.0f, 350000.0f), 1004.5f);
    const float p608 = mynn_f32_sub(
        mynn_f32_div(461.6f, 287.0f), 1.0f);

    r.zw[column] = as_type<float>(0u);
    for (int k = 0; k < q.nz; ++k) {
        const ulong index = ulong(k) * ulong(q.ncol) + ulong(column);
        const ulong previous_zw = ulong(k) * ulong(q.ncol) + ulong(column);
        const ulong next_zw = ulong(k + 1) * ulong(q.ncol) + ulong(column);
        r.zw[next_zw] = mynn_f32_add(r.zw[previous_zw], r.dz[index]);
    }

    const float ustar = r.ust[column];
    const float lead = mynn_f32_mul(5.0f, ustar);
    const float reach = mynn_f32_mul(ustar, 700.0f);
    const float denom = mynn_f32_mul(
        mynn_f32_max2(ustar, 0.01f), 700.0f);
    for (int k = 0; k < q.nz; ++k) {
        const ulong index = ulong(k) * ulong(q.ncol) + ulong(column);
        const float e = r.exner[index];
        const float qc = r.sqc[index];
        const float qi = r.sqi[index];
        const float qv_value = r.sqv[index];
        const float theta = r.th[index];
        r.qv1[index] = mynn_f32_div(qv_value,
                                  mynn_f32_sub(1.0f, qv_value));
        r.sqw[index] = mynn_f32_add(
            mynn_f32_add(qv_value, qc), qi);
        r.thl[index] = mynn_f32_sub(
            mynn_f32_sub(theta,
                mynn_f32_mul(mynn_f32_div(xlvcp, e), qc)),
            mynn_f32_mul(mynn_f32_div(xlscp, e), qi));
        r.thetav[index] = mynn_f32_mul(
            theta, mynn_f32_add(1.0f, mynn_f32_mul(p608, qv_value)));
        r.qke_seed[index] = mynn_f32_mul(lead, mynn_f32_max2(
            mynn_f32_div(mynn_f32_sub(reach, r.zw[index]), denom), 0.01f));
    }
}

// gsd41 selects the GSD MYNN v4.1 mym_predict: unweighted diffusion with no
// EDMF floor, an x(top)=x(top-1) upper row, and a 1e-4 TKE floor.
template <typename Resources, typename Params>
static inline void mynn_predict_column(
    constant Resources &r, constant Params &q, uint column, bool gsd41)
{
    if (column >= uint(q.ncol) || q.nz < 2) return;
    (void)r.phh;
    const int nz = q.nz;
    const ulong ncol = ulong(q.ncol);
    const float step = r.delt[column];

    const ulong base = ulong(column);
    const ulong last = ulong(nz - 1) * ncol + ulong(column);
    const ulong top = last;
    float kqdz_top, kmdz_top;
    if (gsd41) {
        for (int k = 0; k < nz; ++k) {
            const ulong idx = ulong(k) * ncol + ulong(column);
            r.qkw[idx] = mynn_f32_sqrt(max(r.qke_in[idx], 0.0f));
            r.rhoinv[idx] = 1.0f;
            r.kqdz[idx] = mynn_f32_mul(3.0f, r.dfq[idx]);
            r.kmdz[idx] = r.dfq[idx];
        }
        kqdz_top = mynn_f32_mul(3.0f, r.dfq[last]);
        kmdz_top = r.dfq[last];
    } else {
        for (int k = 0; k < nz; ++k) {
            const ulong idx = ulong(k) * ncol + ulong(column);
            r.qkw[idx] = mynn_f32_sqrt(max(r.qke_in[idx], 0.0f));
            r.rhoinv[idx] = 1.0f / max(r.rho[idx], 1.0e-4f);
        }
        r.rhoinv[base] = 1.0f / r.rho[base];
        float rhoz = r.rho[base];
        r.kqdz[base] = rhoz * (3.0f * r.dfq[base]);
        r.kmdz[base] = rhoz * r.dfq[base];
        for (int k = 1; k < nz; ++k) {
            const ulong idx = ulong(k) * ncol + ulong(column);
            rhoz = (r.rho[idx] * r.dz[idx - ncol]
                  + r.rho[idx - ncol] * r.dz[idx])
                 / (r.dz[idx - ncol] + r.dz[idx]);
            rhoz = max(rhoz, 1.0e-4f);
            r.kqdz[idx] = rhoz * (3.0f * r.dfq[idx]);
            r.kmdz[idx] = rhoz * r.dfq[idx];
        }
        kqdz_top = rhoz * (3.0f * r.dfq[last]);
        kmdz_top = rhoz * r.dfq[last];
        for (int k = 1; k < nz - 1; ++k) {
            const ulong idx = ulong(k) * ncol + ulong(column);
            r.kqdz[idx] = max(r.kqdz[idx], 0.5f * r.s_aw[idx]);
            r.kqdz[idx] = max(r.kqdz[idx],
                -0.5f * (r.s_aw[idx] - r.s_aw[idx + ncol]));
            r.kmdz[idx] = max(r.kmdz[idx], 0.5f * r.s_aw[idx]);
            r.kmdz[idx] = max(r.kmdz[idx],
                -0.5f * (r.s_aw[idx] - r.s_aw[idx + ncol]));
        }
    }

    if (gsd41) {
        const float vkz = mynn_f32_mul(mynn_f32_mul(0.4f, 0.5f), r.dz[base]);
        const float ustar = r.ust[column];
        const float ustar3 = mynn_f32_mul(mynn_f32_mul(ustar, ustar), ustar);
        const float pdk1 = mynn_f32_div(mynn_f32_mul(
            mynn_f32_mul(2.0f, ustar3), r.pmz[column]), vkz);
        const float pdk_bottom = mynn_f32_sub(pdk1, r.pdk[base + ncol]);
        for (int k = 0; k < nz - 1; ++k) {
            const ulong idx = ulong(k) * ncol + ulong(column);
            const float next_kqdz = k == nz - 2 ? kqdz_top : r.kqdz[idx + ncol];
            const float b1l = mynn_f32_mul(mynn_f32_mul(24.0f, 0.5f),
                mynn_f32_add(r.el[idx + ncol], r.el[idx]));
            const float bp = mynn_f32_div(mynn_f32_mul(2.0f, r.qkw[idx]), b1l);
            const float pdk_here = k == 0 ? pdk_bottom : r.pdk[idx];
            const float rp = mynn_f32_add(r.pdk[idx + ncol], pdk_here);
            const float dtz = mynn_f32_div(step, r.dz[idx]);
            r.a[idx] = mynn_f32_negate(mynn_f32_mul(dtz, r.kqdz[idx]));
            r.b[idx] = mynn_f32_add(mynn_f32_add(1.0f, mynn_f32_mul(dtz,
                mynn_f32_add(r.kqdz[idx], next_kqdz))),
                mynn_f32_mul(bp, step));
            r.c[idx] = mynn_f32_negate(mynn_f32_mul(dtz, next_kqdz));
            r.d[idx] = mynn_f32_add(mynn_f32_mul(rp, step), r.qke_in[idx]);
        }
        r.a[top] = -1.0f; r.b[top] = 1.0f; r.c[top] = 0.0f; r.d[top] = 0.0f;
        r.cp[base] = mynn_f32_div(r.c[base], r.b[base]);
        r.dp[base] = mynn_f32_div(r.d[base], r.b[base]);
        for (int k = 1; k < nz; ++k) {
            const ulong idx = ulong(k) * ncol + ulong(column);
            const float m = mynn_f32_sub(r.b[idx],
                mynn_f32_mul(r.cp[idx - ncol], r.a[idx]));
            r.cp[idx] = mynn_f32_div(r.c[idx], m);
            r.dp[idx] = mynn_f32_div(mynn_f32_sub(r.d[idx],
                mynn_f32_mul(r.dp[idx - ncol], r.a[idx])), m);
        }
    } else {
        const float vkz = 0.4f * 0.5f * r.dz[base];
        const float ustar = r.ust[column];
        const float pdk1 = 2.0f * (ustar * ustar * ustar) * r.pmz[column] / vkz;
        const float pdk_bottom = pdk1 - r.pdk[base + ncol];
        for (int k = 0; k < nz - 1; ++k) {
            const ulong idx = ulong(k) * ncol + ulong(column);
            const float next_kqdz = k == nz - 2 ? kqdz_top : r.kqdz[idx + ncol];
            const float b1l = 24.0f * 0.5f * (r.el[idx + ncol] + r.el[idx]);
            const float bp = 2.0f * r.qkw[idx] / b1l;
            const float pdk_here = k == 0 ? pdk_bottom : r.pdk[idx];
            const float rp = r.pdk[idx + ncol] + pdk_here;
            const float dtz = step / r.dz[idx];
            r.a[idx] = -dtz * r.kqdz[idx] * r.rhoinv[idx];
            r.b[idx] = 1.0f + dtz * (r.kqdz[idx] + next_kqdz) * r.rhoinv[idx]
                + bp * step;
            r.c[idx] = -dtz * next_kqdz * r.rhoinv[idx];
            r.d[idx] = rp * step + r.qke_in[idx];
        }
        r.a[top] = 0.0f; r.b[top] = 1.0f; r.c[top] = 0.0f;
        r.d[top] = r.qke_in[top];
        r.cp[base] = r.c[base] / r.b[base];
        r.dp[base] = r.d[base] / r.b[base];
        for (int k = 1; k < nz; ++k) {
            const ulong idx = ulong(k) * ncol + ulong(column);
            const float m = r.b[idx] - r.cp[idx - ncol] * r.a[idx];
            r.cp[idx] = r.c[idx] / m;
            r.dp[idx] = (r.d[idx] - r.dp[idx - ncol] * r.a[idx]) / m;
        }
    }
    r.d[top] = r.dp[top];
    for (int k = nz - 2; k >= 0; --k) {
        const ulong idx = ulong(k) * ncol + ulong(column);
        r.d[idx] = gsd41 ? mynn_f32_sub(r.dp[idx],
            mynn_f32_mul(r.cp[idx], r.d[idx + ncol]))
            : r.dp[idx] - r.cp[idx] * r.d[idx + ncol];
    }
    for (int k = 0; k < nz; ++k) {
        const ulong idx = ulong(k) * ncol + ulong(column);
        r.qke_out[idx] = gsd41 ? mynn_f32_max2(r.d[idx], 1.0e-4f)
            : min(max(r.d[idx], 1.0e-3f), 150.0f);
    }

    for (int k = 0; k < nz - 1; ++k) {
        const ulong idx = ulong(k) * ncol + ulong(column);
        const float next_kmdz = k == nz - 2 ? kmdz_top : r.kmdz[idx + ncol];
        const float b2l = 15.0f * 0.5f * (r.el[idx + ncol] + r.el[idx]);
        const float bp = 2.0f * r.qkw[idx] / b2l;
        const float pdq_here = k == 0 ? r.pdq[base + ncol] : r.pdq[idx];
        const float rp = r.pdq[idx + ncol] + pdq_here;
        const float dtz = step / r.dz[idx];
        r.a[idx] = -dtz * r.kmdz[idx] * r.rhoinv[idx];
        r.b[idx] = 1.0f + dtz * (r.kmdz[idx] + next_kmdz) * r.rhoinv[idx]
            + bp * step;
        r.c[idx] = -dtz * next_kmdz * r.rhoinv[idx];
        r.d[idx] = rp * step + r.qsq_in[idx];
    }
    r.a[top] = -1.0f; r.b[top] = 1.0f; r.c[top] = 0.0f; r.d[top] = 0.0f;
    r.cp[base] = r.c[base] / r.b[base];
    r.dp[base] = r.d[base] / r.b[base];
    for (int k = 1; k < nz; ++k) {
        const ulong idx = ulong(k) * ncol + ulong(column);
        const float m = r.b[idx] - r.cp[idx - ncol] * r.a[idx];
        r.cp[idx] = r.c[idx] / m;
        r.dp[idx] = (r.d[idx] - r.dp[idx - ncol] * r.a[idx]) / m;
    }
    r.d[top] = r.dp[top];
    for (int k = nz - 2; k >= 0; --k) {
        const ulong idx = ulong(k) * ncol + ulong(column);
        r.d[idx] = r.dp[idx] - r.cp[idx] * r.d[idx + ncol];
    }
    for (int k = 0; k < nz; ++k) {
        const ulong idx = ulong(k) * ncol + ulong(column);
        r.qsq_out[idx] = max(r.d[idx], 1.0e-17f);
    }

    for (int k = 0; k < nz - 1; ++k) {
        const ulong idx = ulong(k) * ncol + ulong(column);
        const float b2l = r.qkw[idx] <= 0.0f ? 0.0f
            : 15.0f * 0.25f * (r.el[idx + ncol] + r.el[idx]) / r.qkw[idx];
        const float pdt_here = k == 0 ? r.pdt[base + ncol] : r.pdt[idx];
        const float pdc_here = k == 0 ? r.pdc[base + ncol] : r.pdc[idx];
        r.tsq_out[idx] = b2l * (r.pdt[idx + ncol] + pdt_here);
        r.cov_out[idx] = b2l * (r.pdc[idx + ncol] + pdc_here);
    }
    r.tsq_out[top] = r.tsq_out[top - ncol];
    r.cov_out[top] = r.cov_out[top - ncol];
}

kernel void mynn_predict_default_columns(
    constant MynnPredictDefaultColumnsResources &r [[buffer(0)]],
    constant MynnPredictDefaultColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    mynn_predict_column(r, q, column, false);
}

kernel void mynn_predict_gsd41_columns(
    constant MynnPredictGsd41ColumnsResources &r [[buffer(0)]],
    constant MynnPredictGsd41ColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    mynn_predict_column(r, q, column, true);
}

kernel void mynn_level2_pairs(
    constant MynnLevel2PairsResources &r [[buffer(0)]],
    constant MynnLevel2PairsParams &q [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= uint(q.n)) return;

    const float pr = q.prandtl, g1 = 0.235f, b1 = 24.0f, b2 = 15.0f;
    const float c2 = 0.729f, c3 = 0.340f, c5 = 0.2f;
    const float a1 = mynn_f32_div(
        mynn_f32_mul(b1, mynn_f32_sub(1.0f,
            mynn_f32_mul(3.0f, g1))), 6.0f);
    const float c1 = mynn_f32_sub(g1, mynn_f32_div(1.0f,
        mynn_f32_mul(mynn_f32_mul(3.0f, a1), 2.88449914061481660f)));
    const float a2 = mynn_f32_div(
        mynn_f32_mul(a1, mynn_f32_sub(g1, c1)),
        mynn_f32_mul(g1, pr));
    const float g2 = mynn_f32_add(
        mynn_f32_mul(mynn_f32_div(b2, b1), mynn_f32_sub(1.0f, c3)),
        mynn_f32_mul(mynn_f32_div(mynn_f32_mul(2.0f, a1), b1),
            mynn_f32_sub(3.0f, mynn_f32_mul(2.0f, c2))));
    const float tv0 = mynn_f32_mul(
        mynn_f32_sub(mynn_f32_div(461.6f, 287.0f), 1.0f), 300.0f);
    const float gtr = mynn_f32_div(9.81f, 300.0f);

    const float dz = r.dz_a[index], dz_prev = r.dz_prev_a[index];
    const float dz_sum = mynn_f32_add(dz, dz_prev);
    const float dzk = mynn_f32_mul(0.5f, dz_sum);
    const float afk = mynn_f32_div(dz, dz_sum);
    const float abk = mynn_f32_sub(1.0f, afk);
    const float du = mynn_f32_sub(r.u_a[index], r.u_prev_a[index]);
    const float dv = mynn_f32_sub(r.v_a[index], r.v_prev_a[index]);
    const float duz = mynn_f32_div(
        mynn_f32_add(mynn_f32_mul(du, du), mynn_f32_mul(dv, dv)),
        mynn_f32_mul(dzk, dzk));
    const float dtz = mynn_f32_div(
        mynn_f32_sub(r.thl_a[index], r.thl_prev_a[index]), dzk);
    const float dqz = mynn_f32_div(
        mynn_f32_sub(r.qw_a[index], r.qw_prev_a[index]), dzk);
    const float vtt = mynn_f32_add(
        mynn_f32_add(1.0f, mynn_f32_mul(r.vt_a[index], abk)),
        mynn_f32_mul(r.vt_prev_a[index], afk));
    const float vqq = mynn_f32_add(
        mynn_f32_add(tv0, mynn_f32_mul(r.vq_a[index], abk)),
        mynn_f32_mul(r.vq_prev_a[index], afk));
    const float dtq = mynn_f32_add(mynn_f32_mul(vtt, dtz),
                                   mynn_f32_mul(vqq, dqz));
    const float gh = mynn_f32_negate(mynn_f32_mul(dtq, gtr));
    const float ri = mynn_f32_div(mynn_f32_negate(gh),
        mynn_f32_max2(duz, 1.0e-10f));
    const float a2fac = mynn_f32_div(1.0f,
        mynn_f32_add(1.0f, mynn_f32_max2(ri, 0.0f)));

    const float rfc = mynn_f32_div(g1, mynn_f32_add(g1, g2));
    const float f1 = mynn_f32_add(
        mynn_f32_add(mynn_f32_mul(b1, mynn_f32_sub(g1, c1)),
            mynn_f32_mul(
                mynn_f32_mul(
                    mynn_f32_mul(mynn_f32_mul(3.0f, a2), a2fac),
                    mynn_f32_sub(1.0f, c2)),
                mynn_f32_sub(1.0f, c5))),
        mynn_f32_mul(mynn_f32_mul(2.0f, a1),
            mynn_f32_sub(3.0f, mynn_f32_mul(2.0f, c2))));
    const float f2 = mynn_f32_sub(
        mynn_f32_mul(b1, mynn_f32_add(g1, g2)),
        mynn_f32_mul(mynn_f32_mul(3.0f, a1),
            mynn_f32_sub(1.0f, c2)));
    const float rf1 = mynn_f32_div(
        mynn_f32_mul(b1, mynn_f32_sub(g1, c1)), f1);
    const float rf2 = mynn_f32_div(mynn_f32_mul(b1, g1), f2);
    const float smc = mynn_f32_div(
        mynn_f32_mul(mynn_f32_div(a1,
            mynn_f32_mul(a2, a2fac)), f1), f2);
    const float shc = mynn_f32_mul(
        mynn_f32_mul(3.0f, mynn_f32_mul(a2, a2fac)),
        mynn_f32_add(g1, g2));
    const float ri1 = mynn_f32_div(0.5f, smc);
    const float ri2 = mynn_f32_mul(rf1, smc);
    const float ri3 = mynn_f32_sub(
        mynn_f32_mul(mynn_f32_mul(4.0f, rf2), smc),
        mynn_f32_mul(2.0f, ri2));
    const float ri4 = mynn_f32_mul(ri2, ri2);
    const float radical = mynn_f32_add(
        mynn_f32_sub(mynn_f32_mul(ri, ri), mynn_f32_mul(ri3, ri)), ri4);
    const float rf = mynn_f32_min2(mynn_f32_mul(ri1,
        mynn_f32_sub(mynn_f32_add(ri, ri2), mynn_f32_sqrt(radical))), rfc);
    const float sh = mynn_f32_div(
        mynn_f32_mul(shc, mynn_f32_sub(rfc, rf)),
        mynn_f32_sub(1.0f, rf));
    const float sm = mynn_f32_mul(
        mynn_f32_div(mynn_f32_mul(smc, mynn_f32_sub(rf1, rf)),
                     mynn_f32_sub(rf2, rf)), sh);

    r.dtl_o[index] = dtz;
    r.dqw_o[index] = dqz;
    r.dtv_o[index] = dtq;
    r.gm_o[index] = duz;
    r.gh_o[index] = gh;
    r.sm_o[index] = sm;
    r.sh_o[index] = sh;
}

// module_bl_mynn.F:1766-1820, using the zeroed VT/VQ locals from
// mym_initialize. Keep the original adjacent-level loop and operation order;
// level-zero tendency/gradient slots are zeroed by the caller before the
// Fortran routine, while the surface SM/SH values are preserved.
kernel void mynn_level2_columns(
    constant MynnLevel2ColumnsResources &r [[buffer(0)]],
    constant MynnLevel2ColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    if (column >= uint(q.ncol) || q.nz < 2) return;
    const ulong ncol = ulong(q.ncol);
    const float pr = q.prandtl, b1 = 24.0f, b2 = 15.0f, g1 = 0.235f;
    const float c2 = 0.729f, c3 = 0.340f, c5 = 0.2f;
    const float a1 = mynn_f32_div(
        mynn_f32_mul(b1, mynn_f32_sub(1.0f, mynn_f32_mul(3.0f, g1))), 6.0f);
    const float c1 = mynn_f32_sub(g1, mynn_f32_div(1.0f,
        mynn_f32_mul(mynn_f32_mul(3.0f, a1), 2.88449914061481660f)));
    const float a2 = mynn_f32_div(
        mynn_f32_mul(a1, mynn_f32_sub(g1, c1)), mynn_f32_mul(g1, pr));
    const float g2 = mynn_f32_add(
        mynn_f32_mul(mynn_f32_div(b2, b1), mynn_f32_sub(1.0f, c3)),
        mynn_f32_mul(mynn_f32_div(mynn_f32_mul(2.0f, a1), b1),
            mynn_f32_sub(3.0f, mynn_f32_mul(2.0f, c2))));
    const float tv0 = mynn_f32_mul(
        mynn_f32_sub(mynn_f32_div(461.6f, 287.0f), 1.0f), 300.0f);
    const float gtr = mynn_f32_div(9.81f, 300.0f);

    const ulong surface = ulong(column);
    r.dtl[surface] = as_type<float>(0u);
    r.dqw[surface] = as_type<float>(0u);
    r.dtv[surface] = as_type<float>(0u);
    r.gm[surface] = as_type<float>(0u);
    r.gh[surface] = as_type<float>(0u);
    for (int k = 1; k < q.nz; ++k) {
        const ulong idx = ulong(k) * ncol + ulong(column);
        const ulong prev = idx - ncol;
        const float dz = r.dz[idx], dz_prev = r.dz[prev];
        const float dz_sum = mynn_f32_add(dz, dz_prev);
        const float dzk = mynn_f32_mul(0.5f, dz_sum);
        const float afk = mynn_f32_div(dz, dz_sum);
        const float abk = mynn_f32_sub(1.0f, afk);
        const float du = mynn_f32_sub(r.u[idx], r.u[prev]);
        const float dv = mynn_f32_sub(r.v[idx], r.v[prev]);
        const float duz = mynn_f32_div(
            mynn_f32_add(mynn_f32_mul(du, du), mynn_f32_mul(dv, dv)),
            mynn_f32_mul(dzk, dzk));
        const float dtz = mynn_f32_div(
            mynn_f32_sub(r.thl[idx], r.thl[prev]), dzk);
        const float dqz = mynn_f32_div(
            mynn_f32_sub(r.qw[idx], r.qw[prev]), dzk);
        const float vtt = mynn_f32_add(
            mynn_f32_add(1.0f, mynn_f32_mul(r.vt[idx], abk)),
            mynn_f32_mul(r.vt[prev], afk));
        const float vqq = mynn_f32_add(
            mynn_f32_add(tv0, mynn_f32_mul(r.vq[idx], abk)),
            mynn_f32_mul(r.vq[prev], afk));
        const float dtq = mynn_f32_add(mynn_f32_mul(vtt, dtz),
                                       mynn_f32_mul(vqq, dqz));
        const float level_gh = mynn_f32_negate(mynn_f32_mul(dtq, gtr));
        const float ri = mynn_f32_div(mynn_f32_negate(level_gh),
            mynn_f32_max2(duz, 1.0e-10f));
        const float a2fac = mynn_f32_div(1.0f,
            mynn_f32_add(1.0f, mynn_f32_max2(ri, 0.0f)));
        const float rfc = mynn_f32_div(g1, mynn_f32_add(g1, g2));
        const float f1 = mynn_f32_add(
            mynn_f32_add(mynn_f32_mul(b1, mynn_f32_sub(g1, c1)),
                mynn_f32_mul(
                    mynn_f32_mul(
                        mynn_f32_mul(mynn_f32_mul(3.0f, a2), a2fac),
                        mynn_f32_sub(1.0f, c2)),
                    mynn_f32_sub(1.0f, c5))),
            mynn_f32_mul(mynn_f32_mul(2.0f, a1),
                mynn_f32_sub(3.0f, mynn_f32_mul(2.0f, c2))));
        const float f2 = mynn_f32_sub(
            mynn_f32_mul(b1, mynn_f32_add(g1, g2)),
            mynn_f32_mul(mynn_f32_mul(3.0f, a1),
                mynn_f32_sub(1.0f, c2)));
        const float rf1 = mynn_f32_div(
            mynn_f32_mul(b1, mynn_f32_sub(g1, c1)), f1);
        const float rf2 = mynn_f32_div(mynn_f32_mul(b1, g1), f2);
        const float smc = mynn_f32_div(
            mynn_f32_mul(mynn_f32_div(a1,
                mynn_f32_mul(a2, a2fac)), f1), f2);
        const float shc = mynn_f32_mul(
            mynn_f32_mul(3.0f, mynn_f32_mul(a2, a2fac)),
            mynn_f32_add(g1, g2));
        const float ri1 = mynn_f32_div(0.5f, smc);
        const float ri2 = mynn_f32_mul(rf1, smc);
        const float ri3 = mynn_f32_sub(
            mynn_f32_mul(mynn_f32_mul(4.0f, rf2), smc),
            mynn_f32_mul(2.0f, ri2));
        const float ri4 = mynn_f32_mul(ri2, ri2);
        const float radical = mynn_f32_add(
            mynn_f32_sub(mynn_f32_mul(ri, ri), mynn_f32_mul(ri3, ri)), ri4);
        const float rf = mynn_f32_min2(mynn_f32_mul(ri1,
            mynn_f32_sub(mynn_f32_add(ri, ri2), mynn_f32_sqrt(radical))), rfc);
        const float sh = mynn_f32_div(
            mynn_f32_mul(shc, mynn_f32_sub(rfc, rf)),
            mynn_f32_sub(1.0f, rf));
        const float sm = mynn_f32_mul(
            mynn_f32_div(mynn_f32_mul(smc, mynn_f32_sub(rf1, rf)),
                         mynn_f32_sub(rf2, rf)), sh);
        r.dtl[idx] = dtz;
        r.dqw[idx] = dqz;
        r.dtv[idx] = dtq;
        r.gm[idx] = duz;
        r.gh[idx] = level_gh;
        r.sm[idx] = sm;
        r.sh[idx] = sh;
    }
}

// gsd41: KPBL blends the theta-v and TKE levels with the height weight, and
// the scale-awareness ratio uses max(dx, 10) without the 2.5 factor.
template <typename Resources, typename Params>
static inline void mynn_pblh_scale_column(
    constant Resources &r, constant Params &q, uint column, bool gsd41)
{
    if (column >= uint(q.ncol) || q.nz < 2) return;
    const ulong ncol = ulong(q.ncol);
    float minthv = 9.0e9f;
    int k = 1;
    while (r.zw[ulong(k) * ncol + ulong(column)] <= 200.0f && k < q.nz) {
        minthv = mynn_f32_min2(minthv,
            r.thetav[ulong(k) * ncol + ulong(column)]);
        ++k;
    }

    const float delta = r.landsea[column] >= 1.5f ? 1.0f : 1.25f;
    float zi = 0.0f;
    for (k = 1; k < q.nz - 1; ++k) {
        const ulong idx = ulong(k) * ncol + ulong(column);
        const float current = r.thetav[idx];
        const float threshold = mynn_f32_add(minthv, delta);
        if (current >= threshold) {
            const float denominator = mynn_f32_max2(
                mynn_f32_sub(current, r.thetav[idx - ncol]), 1.0e-6f);
            const float fraction = mynn_f32_div(
                mynn_f32_sub(current, threshold), denominator);
            zi = mynn_f32_sub(r.zw[idx], mynn_f32_mul(
                r.dz[idx - ncol], mynn_f32_min2(fraction, 1.0f)));
        }
        if (k == q.nz - 2) zi = r.zw[ncol + ulong(column)];
        if (zi != 0.0f) break;
    }

    const float maxqke = mynn_f32_max2(r.qke[column], 0.0f);
    const float tkeeps = mynn_f32_max2(
        mynn_f32_div(maxqke, 40.0f), 0.02f);
    float pblh_tke = 0.0f;
    for (k = 1; k < q.nz - 1; ++k) {
        const ulong idx = ulong(k) * ncol + ulong(column);
        const float qtke = mynn_f32_max2(
            mynn_f32_div(r.qke[idx], 2.0f), 0.0f);
        const float qtkem1 = mynn_f32_max2(
            mynn_f32_div(r.qke[idx - ncol], 2.0f), 0.0f);
        if (qtke <= tkeeps) {
            const float fraction = mynn_f32_div(
                mynn_f32_sub(tkeeps, qtke),
                mynn_f32_max2(mynn_f32_sub(qtkem1, qtke), 1.0e-6f));
            pblh_tke = mynn_f32_sub(r.zw[idx], mynn_f32_mul(
                r.dz[idx - ncol], mynn_f32_min2(fraction, 1.0f)));
            pblh_tke = mynn_f32_max2(
                pblh_tke, r.zw[ncol + ulong(column)]);
        }
        if (k == q.nz - 2) pblh_tke = r.zw[ncol + ulong(column)];
        if (pblh_tke != 0.0f) break;
    }
    pblh_tke = mynn_f32_min2(
        pblh_tke, mynn_f32_add(zi, 350.0f));
    pblh_tke = mynn_f32_max2(pblh_tke,
        mynn_f32_max2(mynn_f32_sub(zi, 350.0f), 10.0f));
    const float tanh_arg = mynn_f32_div(
        mynn_f32_sub(zi, 200.0f), 400.0f);
    const float weight = mynn_f32_add(
        mynn_f32_mul(0.5f, tanh(tanh_arg)), 0.5f);
    if (maxqke > 0.05f) {
        zi = mynn_f32_add(
            mynn_f32_mul(pblh_tke, mynn_f32_sub(1.0f, weight)),
            mynn_f32_mul(zi, weight));
    }

    int kzi = 2;
    if (gsd41) {
        int kzi2 = 2;
        float zi_th = 0.0f;
        for (k = 1; k < q.nz - 1; ++k) {
            const ulong idx = ulong(k) * ncol + ulong(column);
            const float current = r.thetav[idx];
            const float threshold = mynn_f32_add(minthv, delta);
            if (current >= threshold) {
                const float fraction = mynn_f32_div(
                    mynn_f32_sub(current, threshold), mynn_f32_max2(
                        mynn_f32_sub(current, r.thetav[idx - ncol]), 1.0e-6f));
                zi_th = mynn_f32_sub(r.zw[idx], mynn_f32_mul(
                    r.dz[idx - ncol], mynn_f32_min2(fraction, 1.0f)));
                kzi = max(k, 1) + int(round(mynn_f32_div(
                    mynn_f32_sub(zi_th, r.zw[idx - ncol]), r.dz[idx - ncol])));
            }
            if (k == q.nz - 2) zi_th = r.zw[ncol + ulong(column)];
            if (zi_th != 0.0f) break;
        }
        float tke_h = 0.0f;
        for (k = 1; k < q.nz - 1; ++k) {
            const ulong idx = ulong(k) * ncol + ulong(column);
            const float qtke = mynn_f32_max2(
                mynn_f32_div(r.qke[idx], 2.0f), 0.0f);
            const float qtkem1 = mynn_f32_max2(
                mynn_f32_div(r.qke[idx - ncol], 2.0f), 0.0f);
            if (qtke <= tkeeps) {
                const float fraction = mynn_f32_div(
                    mynn_f32_sub(tkeeps, qtke),
                    mynn_f32_max2(mynn_f32_sub(qtkem1, qtke), 1.0e-6f));
                tke_h = mynn_f32_sub(r.zw[idx], mynn_f32_mul(
                    r.dz[idx - ncol], mynn_f32_min2(fraction, 1.0f)));
                tke_h = mynn_f32_max2(tke_h, r.zw[ncol + ulong(column)]);
                kzi2 = max(k, 1) + int(round(mynn_f32_div(
                    mynn_f32_sub(tke_h, r.zw[idx - ncol]), r.dz[idx - ncol])));
            }
            if (k == q.nz - 2) tke_h = r.zw[ncol + ulong(column)];
            if (tke_h != 0.0f) break;
        }
        kzi = max(int(mynn_f32_add(
            mynn_f32_mul(float(kzi2), mynn_f32_sub(1.0f, weight)),
            mynn_f32_mul(float(kzi), weight))), 1);
    } else {
        for (k = 1; k < q.nz - 1; ++k) {
            if (r.zw[ulong(k) * ncol + ulong(column)] >= zi) {
                kzi = k;
                break;
            }
        }
    }

    const float dx_floor = gsd41 ? mynn_f32_max2(r.dx[column], 10.0f)
        : mynn_f32_max2(mynn_f32_mul(2.5f, r.dx[column]), 10.0f);
    float dxdh = mynn_f32_div(dx_floor,
        mynn_f32_min2(zi, 3000.0f));
    float power = mynn_gfk_powf(dxdh, 0.667f);
    float square = mynn_f32_mul(dxdh, dxdh);
    const float psig_bl = mynn_f32_div(
        mynn_f32_add(square, mynn_f32_mul(0.106f, power)),
        mynn_f32_add(mynn_f32_add(square,
            mynn_f32_mul(0.066f, power)), 0.071f));
    dxdh = mynn_f32_div(dx_floor,
        mynn_f32_min2(mynn_f32_add(zi, 500.0f), 3500.0f));
    power = mynn_gfk_powf(dxdh, 0.667f);
    square = mynn_f32_mul(dxdh, dxdh);
    const float psig_shcu = mynn_f32_div(
        mynn_f32_add(square, mynn_f32_mul(0.145f, power)),
        mynn_f32_add(mynn_f32_add(square,
            mynn_f32_mul(0.172f, power)), 0.170f));

    r.zi[column] = zi;
    r.kzi[column] = kzi;
    r.psig_bl[column] = mynn_f32_min2(
        mynn_f32_max2(psig_bl, 0.0f), 1.0f);
    r.psig_shcu[column] = mynn_f32_min2(
        mynn_f32_max2(psig_shcu, 0.0f), 1.0f);
}

kernel void mynn_pblh_scale_columns(
    constant MynnPblhScaleColumnsResources &r [[buffer(0)]],
    constant MynnPblhScaleColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    mynn_pblh_scale_column(r, q, column, false);
}

kernel void mynn_pblh_scale_gsd41_columns(
    constant MynnPblhScaleGsd41ColumnsResources &r [[buffer(0)]],
    constant MynnPblhScaleGsd41ColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    mynn_pblh_scale_column(r, q, column, true);
}

// gsd41 selects the GSD MYNN v4.1 mym_turbulence: no sh floor before the
// closure, no sh/sm clips after it, and a 0.03 cloud floor.
template <typename Resources, typename Params>
static inline void mynn_turbulence_interface(
    constant Resources &r, constant Params &q, uint index, bool gsd41)
{
    if (index >= uint(q.count) || q.nz <= 0) return;
    const uint ld = uint(q.count / q.nz);
    if (ld == 0u) return;
    const uint k = index / ld;
    if (k == 0u) return;
    const ulong idx = ulong(index);
    const ulong prev = idx - ulong(ld);

    const float pr = q.prandtl, g1 = 0.235f, b1 = 24.0f, b2 = 15.0f;
    const float c2 = 0.729f, c3 = 0.340f, c5 = 0.2f;
    const float a1 = mynn_f32_div(
        mynn_f32_mul(b1, mynn_f32_sub(1.0f,
            mynn_f32_mul(3.0f, g1))), 6.0f);
    const float c1 = mynn_f32_sub(g1, mynn_f32_div(1.0f,
        mynn_f32_mul(mynn_f32_mul(3.0f, a1), 2.88449914061481660f)));
    const float a2 = mynn_f32_div(
        mynn_f32_mul(a1, mynn_f32_sub(g1, c1)),
        mynn_f32_mul(g1, pr));
    const float cc2 = mynn_f32_sub(1.0f, c2);
    const float cc3 = mynn_f32_sub(1.0f, c3);
    const float e1c = mynn_f32_mul(
        mynn_f32_mul(mynn_f32_mul(3.0f, a2), b2), cc3);
    const float e2c = mynn_f32_mul(
        mynn_f32_mul(mynn_f32_mul(9.0f, a1), a2), cc2);
    const float e3c = mynn_f32_mul(
        mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(9.0f, a2), a2), cc2),
        mynn_f32_sub(1.0f, c5));
    const float e4c = mynn_f32_mul(
        mynn_f32_mul(mynn_f32_mul(12.0f, a1), a2), cc2);
    const float e5c = mynn_f32_mul(mynn_f32_mul(6.0f, a1), a1);

    const float dzk = mynn_f32_mul(0.5f,
        mynn_f32_add(r.dz[idx], r.dz[prev]));
    const float el_square_f32 = mynn_f32_mul(r.el[idx], r.el[idx]);
    const float qkw_square_f32 = mynn_f32_mul(r.qkw[idx], r.qkw[idx]);
    const float source_f32 = mynn_f32_add(
        mynn_f32_mul(r.sm[idx], r.gm[idx]),
        mynn_f32_mul(r.sh[idx], r.gh[idx]));
    const ulong elsq = mynn_f64_from_f32(el_square_f32);
    ulong q3sq = mynn_f64_from_f32(qkw_square_f32);
    const ulong q2sq = soft64_mul(soft64_mul(
        mynn_f64_from_f32(b1), elsq), mynn_f64_from_f32(source_f32));
    if (!gsd41) r.sh[idx] = mynn_f32_max2(r.sh[idx], 1.0e-5f);

    const float du = mynn_f32_sub(r.u[idx], r.u[prev]);
    const float dv = mynn_f32_sub(r.v[idx], r.v[prev]);
    const float duz = mynn_f32_div(
        mynn_f32_add(mynn_f32_mul(du, du), mynn_f32_mul(dv, dv)),
        mynn_f32_mul(dzk, dzk));
    const float ri = mynn_f32_div(mynn_f32_negate(r.gh[idx]),
        mynn_f32_max2(duz, 1.0e-10f));
    const float a2fac = mynn_f32_div(1.0f,
        mynn_f32_add(1.0f, mynn_f32_max2(ri, 0.0f)));
    const ulong a2fac_sq = mynn_f64_from_f32(
        mynn_f32_mul(a2fac, a2fac));
    const float three_c1 = mynn_f32_mul(3.0f, c1);
    const float three_c1_e5c = mynn_f32_mul(three_c1, e5c);
    const ulong gmel = soft64_mul(mynn_f64_from_f32(r.gm[idx]), elsq);
    const ulong ghel = soft64_mul(mynn_f64_from_f32(r.gh[idx]), elsq);
    const ulong gh64 = mynn_f64_from_f32(r.gh[idx]);
    if (soft64_less(soft64_div(q3sq, elsq), mynn_f64_negate(gh64)))
        q3sq = mynn_f64_negate(soft64_mul(elsq, gh64));

    ulong qdiv;
    ulong e1, e2, e3, e4, eden;
    if (soft64_less(q3sq, q2sq)) {
        qdiv = soft64_sqrt(soft64_div(q3sq, q2sq));
        r.sh[idx] = mynn_f64_to_f32(soft64_mul(
            mynn_f64_from_f32(r.sh[idx]), qdiv));
        r.sm[idx] = mynn_f64_to_f32(soft64_mul(
            mynn_f64_from_f32(r.sm[idx]), qdiv));
        const ulong qdiv2 = soft64_mul(qdiv, qdiv);
        e1 = soft64_sub(q3sq, soft64_mul(soft64_mul(
            soft64_mul(mynn_f64_from_f32(e1c), ghel),
            mynn_f64_from_f32(a2fac)), qdiv2));
        e2 = soft64_sub(q3sq, soft64_mul(soft64_mul(
            soft64_mul(mynn_f64_from_f32(e2c), ghel),
            mynn_f64_from_f32(a2fac)), qdiv2));
        e3 = soft64_add(e1, soft64_mul(soft64_mul(
            soft64_mul(mynn_f64_from_f32(e3c), ghel), a2fac_sq), qdiv2));
        e4 = soft64_sub(e1, soft64_mul(soft64_mul(
            soft64_mul(mynn_f64_from_f32(e4c), ghel),
            mynn_f64_from_f32(a2fac)), qdiv2));
        eden = soft64_add(soft64_mul(e2, e4), soft64_mul(
            soft64_mul(soft64_mul(e3, mynn_f64_from_f32(e5c)), gmel),
            qdiv2));
        if (soft64_less(eden, 0x3bc79ca10c924223ul))
            eden = 0x3bc79ca10c924223ul;
    } else {
        qdiv = 0x3ff0000000000000ul;
        e1 = soft64_sub(q3sq, soft64_mul(soft64_mul(
            mynn_f64_from_f32(e1c), ghel), mynn_f64_from_f32(a2fac)));
        e2 = soft64_sub(q3sq, soft64_mul(soft64_mul(
            mynn_f64_from_f32(e2c), ghel), mynn_f64_from_f32(a2fac)));
        e3 = soft64_add(e1, soft64_mul(soft64_mul(
            mynn_f64_from_f32(e3c), ghel), a2fac_sq));
        e4 = soft64_sub(e1, soft64_mul(soft64_mul(
            mynn_f64_from_f32(e4c), ghel), mynn_f64_from_f32(a2fac)));
        eden = soft64_add(soft64_mul(e2, e4), soft64_mul(
            soft64_mul(e3, mynn_f64_from_f32(e5c)), gmel));
        if (soft64_less(eden, 0x3bc79ca10c924223ul))
            eden = 0x3bc79ca10c924223ul;
        const ulong sm_num = soft64_mul(soft64_mul(q3sq,
            mynn_f64_from_f32(a1)), soft64_sub(e3, soft64_mul(
                mynn_f64_from_f32(three_c1), e4)));
        r.sm[idx] = mynn_f64_to_f32(soft64_div(sm_num, eden));
        const float a2fac_product = mynn_f32_mul(a2, a2fac);
        const ulong sh_num = soft64_mul(soft64_mul(q3sq,
            mynn_f64_from_f32(a2fac_product)), soft64_add(e2,
                soft64_mul(mynn_f64_from_f32(three_c1_e5c), gmel)));
        r.sh[idx] = mynn_f64_to_f32(soft64_div(sh_num, eden));
    }

    if (!gsd41) {
        r.sh[idx] = mynn_f32_min2(
            mynn_f32_max2(r.sh[idx], 0.0f), 4.0f);
        r.sm[idx] = mynn_f32_min2(r.sm[idx],
            mynn_f32_mul(5.0f, mynn_f32_max2(r.sh[idx], 0.02f)));
    }
    const float cldavg = mynn_f32_mul(0.5f,
        mynn_f32_add(r.cldfra[prev], r.cldfra[idx]));
    if (r.edmf_a[idx] > 0.001f || cldavg > 0.02f) {
        const float plume_floor = mynn_f32_mul(0.03f,
            mynn_f32_min2(mynn_f32_mul(
                mynn_f32_mul(10.0f, r.edmf_a[idx]), r.edmf_w[idx]), 1.0f));
        const float cloud_floor = mynn_f32_mul(gsd41 ? 0.03f : 0.05f,
            mynn_f32_min2(cldavg, 1.0f));
        const float floor_value = mynn_f32_max2(plume_floor, cloud_floor);
        r.sm[idx] = mynn_f32_max2(r.sm[idx], floor_value);
        r.sh[idx] = mynn_f32_max2(r.sh[idx], floor_value);
    }

    const float elq = mynn_f32_mul(r.el[idx], r.qkw[idx]);
    const float elh = mynn_f64_to_f32(soft64_mul(
        mynn_f64_from_f32(elq), qdiv));
    r.pdk[idx] = mynn_f32_add(mynn_f32_mul(elq,
        mynn_f32_add(mynn_f32_mul(r.sm[idx], r.gm[idx]),
                     mynn_f32_mul(r.sh[idx], r.gh[idx]))),
        mynn_f32_mul(0.5f, r.tkeprodtd[idx]));
    r.pdt[idx] = mynn_f32_mul(
        mynn_f32_mul(elh, mynn_f32_mul(r.sh[idx], r.dtl[idx])), r.dtl[idx]);
    r.pdq[idx] = mynn_f32_mul(
        mynn_f32_mul(elh, mynn_f32_mul(r.sh[idx], r.dqw[idx])), r.dqw[idx]);
    r.pdc[idx] = mynn_f32_add(
        mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(elh,
            mynn_f32_mul(r.sh[idx], r.dtl[idx])), r.dqw[idx]), 0.5f),
        mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(elh,
            mynn_f32_mul(r.sh[idx], r.dqw[idx])), r.dtl[idx]), 0.5f));
    r.tcd[idx] = 0.0f;
    r.qcd[idx] = 0.0f;
    r.dfm[idx] = mynn_f32_div(
        mynn_f32_mul(elq, r.sm[idx]), dzk);
    r.dfh[idx] = mynn_f32_div(
        mynn_f32_mul(elq, r.sh[idx]), dzk);
    r.dfq[idx] = r.dfm[idx];
}

// WRF module_bl_mynn.F:3135-3140 under spp_pbl=1 (spp_kernel_sources.py
// mynn_spp_diffusivity). DFQ keeps the unperturbed momentum diffusivity.
kernel void mynn_spp_diffusivity(
    constant MynnSppDiffusivityResources &r [[buffer(0)]],
    constant MynnSppDiffusivityParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.count)) return;
    const float exponent = mynn_f32_div(
        -mynn_f32_max2(mynn_f32_sub(r.zw[i], 8000.0f), 0.0f), 2000.0f);
    const float taper = mynn_f32_max2(mynn_gfk_expf(exponent), 0.001f);
    r.dfm[i] = mynn_f32_add(r.dfm[i], mynn_f32_mul(mynn_f32_mul(
        mynn_f32_mul(r.dfm[i], r.rstoch[i]), 1.5f), taper));
    r.dfh[i] = mynn_f32_add(r.dfh[i], mynn_f32_mul(mynn_f32_mul(
        mynn_f32_mul(r.dfh[i], r.rstoch[i]), 1.5f), taper));
}

kernel void mynn_turbulence_default_interfaces(
    constant MynnTurbulenceDefaultInterfacesResources &r [[buffer(0)]],
    constant MynnTurbulenceDefaultInterfacesParams &q [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    mynn_turbulence_interface(r, q, index, false);
}

kernel void mynn_turbulence_gsd41_interfaces(
    constant MynnTurbulenceGsd41InterfacesResources &r [[buffer(0)]],
    constant MynnTurbulenceGsd41InterfacesParams &q [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    mynn_turbulence_interface(r, q, index, true);
}

kernel void mynn_condensation_default_columns(
    constant MynnCondensationDefaultColumnsResources &r [[buffer(0)]],
    constant MynnCondensationDefaultColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    if (column >= uint(q.ncol) || q.nz < 2) return;
    const ulong ncol = ulong(q.ncol);
    const float rd = 287.0f, rv = 461.6f;
    const float cpd = mynn_f32_div(mynn_f32_mul(7.0f, rd), 2.0f);
    const float cpv = mynn_f32_mul(4.0f, rv);
    const float tice = 240.0f, tliq = 269.0f;
    const float tv0 = mynn_f32_mul(
        mynn_f32_sub(mynn_f32_div(461.6f, 287.0f), 1.0f), 300.0f);
    const float rhcrit = 0.83f, rhmax = 1.02f;
    const float qpct_sfc = 0.025f, qpct_pbl = 0.030f, qpct_trp = 0.040f;
    const float exp_m1 = 0.36787944117144233f;

    int found = 0;
    for (int level = q.nz - 3; level >= 1; --level) {
        const ulong low = ulong(level - 1) * ncol + ulong(column);
        const ulong high = ulong(level + 1) * ncol + ulong(column);
        const float theta1 = r.th[low];
        const float theta2 = r.th[high];
        const float ratio1 = mynn_f32_div(r.p[low], 101325.0f);
        const float ratio2 = mynn_f32_div(r.p[high], 101325.0f);
        const float ht1 = mynn_f32_mul(44307.692f,
            mynn_f32_sub(1.0f, mynn_gfk_powf(ratio1, 0.190f)));
        const float ht2 = mynn_f32_mul(44307.692f,
            mynn_f32_sub(1.0f, mynn_gfk_powf(ratio2, 0.190f)));
        const float slope = mynn_f32_div(
            mynn_f32_sub(theta2, theta1), mynn_f32_sub(ht2, ht1));
        if (slope < mynn_f32_div(10.0f, 1500.0f)
                && ht1 < 19000.0f && ht1 > 4000.0f) {
            found = level;
            break;
        }
    }
    const int k_tropo = max(3, found + 2);
    const float pblh2 = mynn_f32_max2(10.0f, r.pblh[column]);
    float zagl = 0.0f, dzm1 = 0.0f;
    for (int k = 0; k < q.nz - 1; ++k) {
        const ulong idx = ulong(k) * ncol + ulong(column);
        zagl = mynn_f32_add(zagl, mynn_f32_mul(0.5f,
            mynn_f32_add(r.dz[idx], dzm1)));
        dzm1 = r.dz[idx];

        const float t = mynn_f32_mul(r.th[idx], r.exner[idx]);
        const float xl = mynn_xl_blend(t);
        const float qsat_tk = mynn_qsat_blend(t, r.p[idx]);
        float rh = mynn_f32_max2(mynn_f32_min2(rhmax,
            mynn_f32_div(r.qw[idx], mynn_f32_max2(1.0e-10f, qsat_tk))),
            0.001f);

        const float rsl = mynn_f32_div(
            mynn_f32_mul(xl, qsat_tk),
            mynn_f32_mul(rv, mynn_f32_mul(t, t)));
        const float cpm = mynn_f32_add(cpd, mynn_f32_mul(r.qw[idx], cpv));
        const float a = mynn_f32_div(1.0f, mynn_f32_add(1.0f,
            mynn_f32_div(mynn_f32_mul(xl, rsl), cpm)));
        const float b = mynn_f32_mul(a, rsl);

        // spp_pbl=1 perturbs the saturation deficit; the deterministic
        // source keeps its vanishing term, operation for operation.
        const float qw_spp = mynn_f32_mul(
            mynn_f32_mul(r.qw[idx], 0.5f), r.rstoch[idx]);
        const float qw_pert = mynn_f32_add(r.qw[idx], q.spp_pbl != 0
            ? qw_spp : mynn_f32_mul(qw_spp, 0.0f));
        const float qmq = mynn_f32_sub(qw_pert, qsat_tk);
        const float qsq_floor = mynn_f32_max2(r.qsq[idx], 0.0f);
        const float sgm_root = mynn_f64_to_f32(soft64_sqrt(
            mynn_f64_from_f32(qsq_floor)));
        float sgm_k = mynn_f32_min2(sgm_root,
            mynn_f32_mul(qsat_tk, 0.666f));
        float wt = mynn_f32_div(mynn_f32_max2(
            mynn_f32_sub(500.0f,
                mynn_f32_max2(mynn_f32_sub(r.dz[idx], 100.0f), 0.0f)),
            0.0f), 500.0f);
        sgm_k = mynn_f32_add(sgm_k, mynn_f32_mul(
            mynn_f32_mul(sgm_k, 0.2f), mynn_f32_sub(1.0f, wt)));
        float qpct = mynn_f32_add(mynn_f32_mul(qpct_pbl, wt),
            mynn_f32_mul(qpct_trp, mynn_f32_sub(1.0f, wt)));
        qpct = mynn_f32_min2(qpct,
            mynn_f32_max2(qpct_sfc, mynn_f32_div(
                mynn_f32_mul(qpct_pbl, zagl), 500.0f)));
        sgm_k = mynn_f32_max2(sgm_k, mynn_f32_mul(qsat_tk, qpct));
        r.sgm[idx] = sgm_k;

        float q1 = mynn_f32_div(qmq, sgm_k);
        const float wt2 = mynn_f32_min2(mynn_f32_div(
            mynn_f32_max2(mynn_f32_sub(zagl, pblh2), 0.0f), 300.0f), 1.0f);
        const float frozen = mynn_f32_add(r.qi[idx], r.qs[idx]);
        if (frozen > 1.0e-9f && zagl > pblh2) {
            const float rh_hack = mynn_f32_min2(rhmax, mynn_f32_add(
                rhcrit, mynn_f32_mul(mynn_f32_mul(wt2, 0.045f),
                    mynn_f32_add(9.0f, log10(frozen)))));
            rh = mynn_f32_max2(rh, rh_hack);
            const float q1_rh = mynn_f32_add(-3.0f, mynn_f32_div(
                mynn_f32_mul(3.0f, mynn_f32_sub(rh, rhcrit)),
                mynn_f32_sub(1.0f, rhcrit)));
            q1 = mynn_f32_max2(q1_rh, q1);
        }
        if (r.qc[idx] > 1.0e-6f && zagl > pblh2) {
            const float rh_hack = mynn_f32_min2(rhmax, mynn_f32_add(
                rhcrit, mynn_f32_mul(mynn_f32_mul(wt2, 0.08f),
                    mynn_f32_add(6.0f, log10(r.qc[idx])))));
            rh = mynn_f32_max2(rh, rh_hack);
            const float q1_rh = mynn_f32_add(-3.0f, mynn_f32_div(
                mynn_f32_mul(3.0f, mynn_f32_sub(rh, rhcrit)),
                mynn_f32_sub(1.0f, rhcrit)));
            q1 = mynn_f32_max2(q1_rh, q1);
        }

        const float q1k = q1;
        const float atan_arg = mynn_f32_mul(1.8f,
            mynn_f32_add(q1, 0.2f));
        float cldfra_k = mynn_f32_max2(0.0f,
            mynn_f32_min2(1.0f, mynn_f32_add(0.5f,
                mynn_f32_mul(0.36f, mynn_gfk_atanf(atan_arg)))));
        const float maxqc = mynn_f32_max2(
            mynn_f32_sub(r.qw[idx], qsat_tk), 0.0f);
        float ql_water, ql_ice;
        if (q1k < 0.0f) {
            const float exp_arg = mynn_f32_sub(
                mynn_f32_mul(1.2f, q1k), 1.0f);
            ql_water = mynn_f32_mul(sgm_k, mynn_gfk_expf(exp_arg));
            ql_ice = ql_water;
        } else if (q1k > 2.0f) {
            ql_water = mynn_f32_min2(mynn_f32_mul(sgm_k, q1k), maxqc);
            ql_ice = mynn_f32_mul(sgm_k, q1k);
        } else {
            const float shape = mynn_f32_mul(sgm_k,
                mynn_f32_add(mynn_f32_add(exp_m1,
                    mynn_f32_mul(0.66f, q1k)),
                    mynn_f32_mul(0.086f, mynn_f32_mul(q1k, q1k))));
            ql_water = mynn_f32_min2(shape, maxqc);
            ql_ice = shape;
        }
        if (cldfra_k < 0.001f) {
            ql_ice = 0.0f;
            ql_water = 0.0f;
            cldfra_k = 0.0f;
        }
        const float liq_frac = mynn_f32_min2(1.0f,
            mynn_f32_max2(0.0f, mynn_f32_div(
                mynn_f32_sub(t, tice), mynn_f32_sub(tliq, tice))));
        r.qc_bl[idx] = mynn_f32_mul(liq_frac, ql_water);
        r.qi_bl[idx] = mynn_f32_mul(
            mynn_f32_sub(1.0f, liq_frac), ql_ice);
        if (k + 1 >= k_tropo) {
            cldfra_k = 0.0f;
            r.qc_bl[idx] = 0.0f;
            r.qi_bl[idx] = 0.0f;
        }

        const float q1_floor = mynn_f32_sub(r.xland[column], 1.5f) >= 0.0f
            ? mynn_f32_max2(q1, -2.5f) : mynn_f32_max2(q1, -2.0f);
        float fng;
        if (q1_floor >= 1.0f) fng = 1.0f;
        else if (q1_floor >= -1.7f) {
            fng = mynn_gfk_expf(mynn_f32_mul(-0.4f,
                mynn_f32_sub(q1_floor, 1.0f)));
        } else if (q1_floor >= -2.5f) {
            fng = mynn_f32_add(3.0f, mynn_gfk_expf(mynn_f32_mul(-3.8f,
                mynn_f32_add(q1_floor, 1.7f))));
        } else {
            fng = mynn_f32_min2(mynn_f32_add(23.9f,
                mynn_gfk_expf(mynn_f32_mul(-1.6f,
                    mynn_f32_add(q1_floor, 2.5f)))), 60.0f);
        }

        float cfmax = mynn_f32_min2(cldfra_k, 0.6f);
        const float zsl = mynn_f32_min2(mynn_f32_max2(25.0f,
            mynn_f32_mul(0.1f, pblh2)), 100.0f);
        wt = mynn_f32_min2(mynn_f32_div(zagl, zsl), 1.0f);
        cfmax = mynn_f32_mul(cfmax, wt);
        const float bb = mynn_f32_div(
            mynn_f32_mul(b, t), r.th[idx]);
        const float qww = mynn_f32_add(1.0f,
            mynn_f32_mul(0.61f, r.qw[idx]));
        const float alpha = mynn_f32_mul(0.61f, r.th[idx]);
        const float beta = mynn_f32_sub(mynn_f32_mul(
            mynn_f32_div(r.th[idx], t), mynn_f32_div(xl, cpd)),
            mynn_f32_mul(1.61f, r.th[idx]));
        r.vt[idx] = mynn_f32_sub(mynn_f32_sub(qww,
            mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(
                cfmax, beta), bb), fng)), 1.0f);
        r.vq[idx] = mynn_f32_sub(mynn_f32_add(alpha,
            mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(
                cfmax, beta), a), fng)), tv0);

        const float fac_damp = mynn_f32_min2(
            mynn_f32_mul(zagl, 0.0025f), 1.0f);
        const float excess = mynn_f32_div(mynn_f32_max2(
            0.0f, mynn_f32_sub(rh, 0.92f)), 0.145f);
        const float cld_factor = mynn_f32_add(1.0f,
            mynn_f32_mul(fac_damp, mynn_f32_min2(
                mynn_f32_mul(excess, excess), 0.37f)));
        r.cldfra[idx] = mynn_f32_min2(1.0f,
            mynn_f32_mul(cld_factor, cldfra_k));
    }

    const ulong top = ulong(q.nz - 1) * ncol + ulong(column);
    r.vt[top] = r.vt[top - ncol];
    r.vq[top] = r.vq[top - ncol];
    r.sgm[top] = r.sgm_in[top];
    r.qc_bl[top] = 0.0f;
    r.qi_bl[top] = 0.0f;
    r.cldfra[top] = 0.0f;
}

// GSD MYNN v4.1's CASE(2) condensation lane. The work resource is laid out
// as five level-major planes: q1, rh, a, b, and cloud fraction.
kernel void mynn_condensation_gsd41_columns(
    constant MynnCondensationGsd41ColumnsResources &r [[buffer(0)]],
    constant MynnCondensationGsd41ColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    if (column >= uint(q.ncol) || q.nz < 2) return;
    const ulong ncol = ulong(q.ncol);
    const float rd = 287.0f, rv = 461.6f, g = 9.81f;
    const float cp = mynn_f32_div(mynn_f32_mul(7.0f, rd), 2.0f);
    const float cpv = mynn_f32_mul(4.0f, rv);
    const float rcp = mynn_f32_div(rd, cp);
    const float ep2 = mynn_f32_div(rd, rv);
    const float ep3 = mynn_f32_sub(1.0f, ep2);
    const float p608 = mynn_f32_sub(mynn_f32_div(rv, rd), 1.0f);
    const float tv0 = mynn_f32_mul(p608, 300.0f);
    const float exp_m1 = 0.36787944117144233f;

    int found = 0;
    for (int level = q.nz - 3; level >= 1; --level) {
        const ulong lo = ulong(level - 1) * ncol + ulong(column);
        const ulong hi = ulong(level + 1) * ncol + ulong(column);
        const float ht1 = mynn_f32_mul(44307.692f, mynn_f32_sub(1.0f,
            mynn_gfk_powf(mynn_f32_div(r.p[lo], 101325.0f), 0.190f)));
        const float ht2 = mynn_f32_mul(44307.692f, mynn_f32_sub(1.0f,
            mynn_gfk_powf(mynn_f32_div(r.p[hi], 101325.0f), 0.190f)));
        const float slope = mynn_f32_div(
            mynn_f32_sub(r.th[hi], r.th[lo]), mynn_f32_sub(ht2, ht1));
        if (slope < mynn_f32_div(10.0f, 1500.0f)
                && ht1 < 19000.0f && ht1 > 4000.0f) {
            found = level;
            break;
        }
    }
    const int k_tropo = max(3, found + 2);
    const float lfac = mynn_f32_min2(mynn_f32_add(4.25f,
        mynn_f32_div(r.dx[column], 4000.0f)), 6.0f);
    const float hfx = r.hfx[column], pblh = r.pblh[column];

    float zagl = 0.0f;
    for (int k = 0; k < q.nz - 1; ++k) {
        const ulong idx = ulong(k) * ncol + ulong(column);
        const int km = k > 0 ? k - 1 : 0;
        const ulong prev = ulong(km) * ncol + ulong(column);
        const ulong next = ulong(k + 1) * ncol + ulong(column);
        const float t = mynn_f32_mul(r.th[idx], r.exner[idx]);
        const float esat = mynn_gsd41_esat_blend(t);
        const float qsl = mynn_f32_div(mynn_f32_mul(ep2, esat),
            mynn_f32_max2(1.0e-4f,
                mynn_f32_sub(r.p[idx], mynn_f32_mul(ep3, esat))));
        const float rh = mynn_f32_max2(mynn_f32_min2(1.0f,
            mynn_f32_div(r.qw[idx], mynn_f32_max2(1.0e-8f, qsl))),
            0.001f);
        const float xl = mynn_gsd41_xl_blend(t);
        const float exn = mynn_gfk_powf(
            mynn_f32_div(r.p[idx], 100000.0f), rcp);
        const float tl = mynn_f32_mul(r.thl[idx], exn);
        const float qsat_tl = mynn_gsd41_qsat_blend(tl, r.p[idx]);
        const float rsl = mynn_f32_div(mynn_f32_mul(xl, qsat_tl),
            mynn_f32_mul(rv, mynn_f32_mul(tl, tl)));
        const float cpm = mynn_f32_add(cp, mynn_f32_mul(r.qw[idx], cpv));
        const float a = mynn_f32_div(1.0f, mynn_f32_add(1.0f,
            mynn_f32_div(mynn_f32_mul(xl, rsl), cpm)));
        const float qmq = mynn_f32_mul(a,
            mynn_f32_sub(r.qw[idx], qsat_tl));
        const float b = mynn_f32_mul(a, rsl);
        const float tl_up = mynn_f32_mul(r.thl[next], mynn_gfk_powf(
            mynn_f32_div(r.p[next], 100000.0f), rcp));
        const float tl_dn = mynn_f32_mul(r.thl[prev], mynn_gfk_powf(
            mynn_f32_div(r.p[prev], 100000.0f), rcp));
        const float dtl = mynn_f32_sub(mynn_f32_mul(0.5f,
            mynn_f32_add(tl_up, tl)), mynn_f32_mul(0.5f,
            mynn_f32_add(tl, tl_dn)));
        const float dqw = mynn_f32_sub(mynn_f32_mul(0.5f,
            mynn_f32_add(r.qw[next], r.qw[idx])), mynn_f32_mul(0.5f,
            mynn_f32_add(r.qw[idx], r.qw[prev])));
        const float dzk = k == 0 ? mynn_f32_mul(0.5f, r.dz[idx])
            : mynn_f32_mul(0.5f, mynn_f32_add(r.dz[idx],
                r.dz[idx - ncol]));
        const float cdhdz = mynn_f32_add(mynn_f32_div(dtl, dzk),
            mynn_f32_mul(mynn_f32_div(g, cpm),
                mynn_f32_add(1.0f, r.qw[idx])));
        zagl = mynn_f32_add(zagl, r.dz[idx]);
        const float ls_min0 = mynn_f32_add(300.0f, mynn_f32_min2(
            mynn_f32_mul(3.0f, mynn_f32_max2(hfx, 0.0f)), 300.0f));
        float ls_min = mynn_f32_min2(mynn_f32_max2(zagl, 25.0f), ls_min0);
        if (zagl > mynn_f32_add(pblh, 2000.0f))
            ls_min = mynn_f32_max2(mynn_f32_add(ls_min,
                mynn_f32_mul(0.5f, mynn_f32_sub(
                    mynn_f32_add(pblh, 2000.0f), zagl))), 300.0f);
        const float ls = mynn_f32_max2(mynn_f32_min2(
            mynn_f32_mul(lfac, r.el[idx]), 600.0f), ls_min);
        const float t1 = mynn_f32_div(mynn_f32_mul(a, dqw), dzk);
        const float term = mynn_f32_add(mynn_f32_sub(mynn_f32_mul(t1, t1),
            mynn_f32_div(mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(
                mynn_f32_mul(2.0f, a), b), cdhdz), dqw), dzk)),
            mynn_f32_mul(mynn_f32_mul(b, b),
                mynn_f32_mul(cdhdz, cdhdz)));
        const float sg = mynn_f32_max2(1.0e-10f,
            mynn_f32_mul(mynn_f32_mul(0.225f, ls),
                mynn_f64_to_f32(soft64_sqrt(mynn_f64_from_f32(
                    mynn_f32_max2(0.0f, term))))));
        const float q1 = mynn_f32_div(qmq, sg);
        r.sgm[idx] = sg;
        r.work[idx] = q1;
        r.work[idx + ulong(q.nz) * ncol] = rh;
        r.work[idx + ulong(2 * q.nz) * ncol] = a;
        r.work[idx + ulong(3 * q.nz) * ncol] = b;
        r.work[idx + ulong(4 * q.nz) * ncol] = mynn_f32_max2(0.0f,
            mynn_f32_min2(1.0f, mynn_f32_add(0.5f,
                mynn_f32_mul(0.36f,
                    mynn_gfk_atanf(mynn_f32_mul(1.55f, q1))))));
    }

    zagl = 0.0f;
    const float pblh2 = mynn_f32_max2(10.0f, pblh);
    for (int k = 0; k < q.nz - 1; ++k) {
        const ulong idx = ulong(k) * ncol + ulong(column);
        const float t = mynn_f32_mul(r.th[idx], r.exner[idx]);
        const float q1 = r.work[idx];
        const float rh = r.work[idx + ulong(q.nz) * ncol];
        const float a = r.work[idx + ulong(2 * q.nz) * ncol];
        const float b = r.work[idx + ulong(3 * q.nz) * ncol];
        float cld = r.work[idx + ulong(4 * q.nz) * ncol];
        const float sg = r.sgm[idx];
        zagl = mynn_f32_add(zagl, r.dz[idx]);
        float ql;
        if (q1 < 0.0f)
            ql = mynn_f32_mul(sg, mynn_gfk_expf(
                mynn_f32_sub(mynn_f32_mul(1.2f, q1), 1.0f)));
        else if (q1 > 2.0f)
            ql = mynn_f32_mul(sg, q1);
        else
            ql = mynn_f32_mul(sg, mynn_f32_add(mynn_f32_add(exp_m1,
                mynn_f32_mul(0.66f, q1)),
                mynn_f32_mul(0.086f, mynn_f32_mul(q1, q1))));
        if (k + 1 >= k_tropo - 1) {
            cld = 0.0f;
            ql = 0.0f;
        }
        const float q1c = mynn_f32_max2(q1, -5.0f);
        float fng;
        if (q1c >= 1.0f) fng = 1.0f;
        else if (q1c >= -1.7f)
            fng = mynn_gfk_expf(mynn_f32_mul(-0.4f,
                mynn_f32_sub(q1c, 1.0f)));
        else if (q1c >= -2.5f)
            fng = mynn_f32_add(3.0f, mynn_gfk_expf(mynn_f32_mul(-3.8f,
                mynn_f32_add(q1c, 1.7f))));
        else fng = mynn_f32_min2(mynn_f32_add(23.9f,
            mynn_gfk_expf(mynn_f32_mul(-1.6f,
                mynn_f32_add(q1c, 2.5f)))), 60.0f);
        fng = mynn_f32_min2(fng, 20.0f);
        const float xl = mynn_gsd41_xl_blend(t);
        const float bb = mynn_f32_div(mynn_f32_mul(b, t), r.th[idx]);
        const float qww = mynn_f32_add(1.0f,
            mynn_f32_mul(0.61f, r.qw[idx]));
        const float alpha = mynn_f32_mul(0.61f, r.th[idx]);
        const float beta = mynn_f32_sub(mynn_f32_mul(
            mynn_f32_div(r.th[idx], t), mynn_f32_div(xl, cp)),
            mynn_f32_mul(1.61f, r.th[idx]));
        const float cw = mynn_f32_min2(cld, 0.99f);
        r.vt[idx] = mynn_f32_sub(mynn_f32_sub(qww,
            mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(cw, beta), bb), fng)),
            1.0f);
        r.vq[idx] = mynn_f32_sub(mynn_f32_add(alpha,
            mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(cw, beta), a), fng)),
            tv0);
        const ulong ztrop = ulong(k_tropo - 1) * ncol + ulong(column);
        const float base_h = mynn_f32_add(pblh2, 1000.0f);
        const float fac_damp = mynn_f32_sub(1.0f, mynn_f32_min2(
            mynn_f32_div(mynn_f32_max2(mynn_f32_sub(zagl, base_h), 0.0f),
                mynn_f32_max2(mynn_f32_sub(r.zw[ztrop], base_h), 500.0f)),
            1.0f));
        const float rhterm = mynn_f32_max2(0.0f,
            mynn_f32_div(mynn_f32_sub(rh, 0.75f), 0.26f));
        const float cld_factor = mynn_f32_add(1.0f,
            mynn_f32_mul(fac_damp, mynn_gfk_powf(rhterm, 1.9f)));
        cld = mynn_f32_min2(1.0f, mynn_f32_mul(cld_factor, cld));
        r.cldfra[idx] = cld;
        r.qc_bl[idx] = ql;
        r.qi_bl[idx] = 0.0f;
    }
    const ulong top = ulong(q.nz - 1) * ncol + ulong(column);
    r.vt[top] = r.vt[top - ncol];
    r.vq[top] = r.vq[top - ncol];
    r.sgm[top] = r.sgm_in[top];
    r.qc_bl[top] = 0.0f;
    r.qi_bl[top] = 0.0f;
    r.cldfra[top] = 0.0f;
}

// gsd41: factor 0.5, held to 0..2e-5 K/s, no pressure taper.
template <typename Resources, typename Params>
static inline void mynn_diss_heat_column(
    constant Resources &r, constant Params &q, uint column, bool gsd41)
{
    if (column >= uint(q.ncol) || q.nz < 1) return;
    const ulong ncol = ulong(q.ncol);
    for (int k = 0; k < q.nz - 1; ++k) {
        const ulong idx = ulong(k) * ncol + ulong(column);
        const float blend = mynn_f32_max2(mynn_f32_mul(0.5f,
            mynn_f32_add(r.el[idx], r.el[idx + ncol])), 1.0f);
        const float powered = mynn_gfk_powf(r.qke[idx], 1.5f);
        const float value = mynn_f32_div(mynn_f32_div(
            mynn_f32_mul(gsd41 ? 0.5f : 1.0f, powered),
            mynn_f32_mul(24.0f, blend)), 1004.5f);
        if (gsd41) {
            r.diss_heat[idx] = mynn_f32_min2(
                mynn_f32_max2(value, 0.0f), 0.00002f);
            continue;
        }
        const float bounded = mynn_f32_min2(
            mynn_f32_max2(value, 0.0f), 0.002f);
        const float damping_arg = mynn_f32_negate(mynn_f32_div(
            10000.0f, mynn_f32_max2(r.p[idx], 1.0f)));
        r.diss_heat[idx] = mynn_f32_mul(
            bounded, mynn_gfk_expf(damping_arg));
    }
    const ulong top = ulong(q.nz - 1) * ncol + ulong(column);
    r.diss_heat[top] = 0.0f;
}

kernel void mynn_driver_diss_heat_columns(
    constant MynnDriverDissHeatColumnsResources &r [[buffer(0)]],
    constant MynnDriverDissHeatColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    mynn_diss_heat_column(r, q, column, false);
}

kernel void mynn_driver_diss_heat_gsd41_columns(
    constant MynnDriverDissHeatGsd41ColumnsResources &r [[buffer(0)]],
    constant MynnDriverDissHeatGsd41ColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    mynn_diss_heat_column(r, q, column, true);
}

// GSD MYNN v4.1 module_bl_mynn.F:4084-4096, :4236-4264: theta-v of the
// liquid-water theta, with the carried subgrid cloud (in-cloud QC_BL times
// CLDFRA_BL, split by (T-254)/15) standing in where no resolved cloud is.
kernel void mynn_gsd41_thvl_columns(
    constant MynnGsd41ThvlColumnsResources &r [[buffer(0)]],
    constant MynnGsd41ThvlColumnsParams &q [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= uint(q.count)) return;
    const float xlvcp = mynn_f32_div(2500000.0f, 1004.5f);
    const float xlscp = mynn_f32_div(
        mynn_f32_add(2500000.0f, 350000.0f), 1004.5f);
    float sqc9 = r.sqc[index], sqi9 = r.sqi[index];
    if (sqc9 < 1.0e-6f && sqi9 < 1.0e-8f && r.cldfra_bl[index] > 0.001f) {
        const float liq = mynn_f32_min2(1.0f, mynn_f32_max2(0.0f,
            mynn_f32_div(mynn_f32_sub(r.tk[index], 254.0f), 15.0f)));
        sqc9 = mynn_f32_mul(mynn_f32_mul(r.qc_bl[index], liq),
                            r.cldfra_bl[index]);
        sqi9 = mynn_f32_mul(mynn_f32_mul(r.qc_bl[index],
            mynn_f32_sub(1.0f, liq)), r.cldfra_bl[index]);
    }
    const float thlsg = mynn_f32_sub(mynn_f32_sub(r.th[index],
        mynn_f32_mul(mynn_f32_div(xlvcp, r.exner[index]), sqc9)),
        mynn_f32_mul(mynn_f32_div(xlscp, r.exner[index]), sqi9));
    r.thvl[index] = mynn_f32_mul(thlsg, mynn_f32_add(1.0f,
        mynn_f32_mul(0.61f, r.sqv[index])));
}

// GSD MYNN v4.1 module_bl_mynn.F:4690-4713, at the write-back: subgrid cloud
// fraction may fall by at most 0.25*delt/ts_decay per call, ts_decay =
// MIN(1800 s, 3*dx/MAX(|V|, 1 m/s)); a decayed fraction under 0.005 clears
// both fields, and QC_BL is floored at 1e-8 where the fraction exceeds 0.005.
kernel void mynn_gsd41_cloud_decay(
    constant MynnGsd41CloudDecayResources &r [[buffer(0)]],
    constant MynnGsd41CloudDecayParams &q [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= uint(q.nz) * uint(q.ncol)) return;
    const uint column = index % uint(q.ncol);
    float cf = r.cldfra_bl[index];
    const float old = r.cldfra_old[index];
    if (cf < old) {
        const float speed = mynn_f32_sqrt(mynn_f32_add(
            mynn_f32_mul(r.u[index], r.u[index]),
            mynn_f32_mul(r.v[index], r.v[index])));
        const float ts_decay = mynn_f32_min2(1800.0f, mynn_f32_div(
            mynn_f32_mul(3.0f, r.dx[column]), mynn_f32_max2(speed, 1.0f)));
        cf = mynn_f32_max2(cf, mynn_f32_sub(old, mynn_f32_div(
            mynn_f32_mul(0.25f, r.delt[column]), ts_decay)));
        r.cldfra_bl[index] = cf;
        if (cf < 0.005f) {
            r.cldfra_bl[index] = 0.0f;
            r.qc_bl[index] = 0.0f;
        }
    }
    if (r.qc_bl[index] < 1.0e-8f && r.cldfra_bl[index] > 0.005f)
        r.qc_bl[index] = 1.0e-8f;
}

// Source helpers for mynn_tendencies_columns. They are kept separate from an
// export entry point until the complete tendency resource/scratch ABI is
// ported. Storage remains level-major, with one sequential vertical walk per
// column, as in the CUDA MynnColumn accessors.
static inline ulong mynn_column_index(int level, uint ncol, uint column)
{
    return ulong(level) * ulong(ncol) + ulong(column);
}

static inline void mynn_tridiag2_column(
    device const float *a, device const float *b, device const float *c,
    device const float *d, device float *cpw, device float *dpw,
    device float *x, uint ncol, uint column, int n)
{
    ulong i0 = mynn_column_index(0, ncol, column);
    cpw[i0] = mynn_f32_div(c[i0], b[i0]);
    dpw[i0] = mynn_f32_div(d[i0], b[i0]);
    for (int k = 1; k < n; ++k) {
        const ulong idx = mynn_column_index(k, ncol, column);
        const ulong prev = k != 0 ? idx - ulong(ncol) : idx;
        const float m = mynn_f32_sub(b[idx], mynn_f32_mul(cpw[prev], a[idx]));
        cpw[idx] = mynn_f32_div(c[idx], m);
        dpw[idx] = mynn_f32_div(
            mynn_f32_sub(d[idx], mynn_f32_mul(dpw[prev], a[idx])), m);
    }
    const ulong last = mynn_column_index(n - 1, ncol, column);
    x[last] = dpw[last];
    for (int k = n - 2; k >= 0; --k) {
        const ulong idx = mynn_column_index(k, ncol, column);
        const ulong next = idx + ulong(ncol);
        x[idx] = mynn_f32_sub(dpw[idx], mynn_f32_mul(cpw[idx], x[next]));
    }
}

static inline void mynn_moisture_check_column(
    float delt, float xlvcp, float xlscp,
    device const float *dp, device const float *exner,
    device float *qv, device float *qc, device float *qi, device float *qs,
    device float *th, device float *dqv, device float *dqc,
    device float *dqi, device float *dqs, device float *dth,
    uint ncol, uint column, int nz)
{
    const float qvmin = 1.0e-20f;
    const float qcmin = 0.0f, qimin = 0.0f;
    float dqv2 = 0.0f;
    for (int k = nz - 1; k >= 0; --k) {
        const ulong idx = mynn_column_index(k, ncol, column);
        const ulong prev = k != 0 ? idx - ulong(ncol) : idx;
        const float dqc2 = mynn_f32_max2(
            0.0f, mynn_f32_sub(qcmin, qc[idx]));
        const float dqi2 = mynn_f32_max2(
            0.0f, mynn_f32_sub(qimin, qi[idx]));
        const float dqs2 = mynn_f32_max2(
            0.0f, mynn_f32_sub(qimin, qs[idx]));
        const float xlvcp_ex = mynn_f32_div(xlvcp, exner[idx]);
        const float xlscp_ex = mynn_f32_div(xlscp, exner[idx]);
        dqc[idx] = mynn_f32_add(dqc[idx], mynn_f32_div(dqc2, delt));
        dqi[idx] = mynn_f32_add(dqi[idx], mynn_f32_div(dqi2, delt));
        dqs[idx] = mynn_f32_add(dqs[idx], mynn_f32_div(dqs2, delt));
        dqv[idx] = mynn_f32_sub(dqv[idx], mynn_f32_div(
            mynn_f32_add(mynn_f32_add(dqc2, dqi2), dqs2), delt));
        dth[idx] = mynn_f32_add(mynn_f32_add(dth[idx], mynn_f32_mul(
            xlvcp_ex, mynn_f32_div(dqc2, delt))), mynn_f32_mul(
                xlscp_ex, mynn_f32_div(mynn_f32_add(dqi2, dqs2), delt)));
        qc[idx] = mynn_f32_add(qc[idx], dqc2);
        qi[idx] = mynn_f32_add(qi[idx], dqi2);
        qs[idx] = mynn_f32_add(qs[idx], dqs2);
        qv[idx] = mynn_f32_sub(mynn_f32_sub(mynn_f32_sub(
            qv[idx], dqc2), dqi2), dqs2);
        th[idx] = mynn_f32_add(mynn_f32_add(th[idx], mynn_f32_mul(
            xlvcp_ex, dqc2)), mynn_f32_mul(xlscp_ex,
                mynn_f32_add(dqi2, dqs2)));
        dqv2 = mynn_f32_max2(0.0f, mynn_f32_sub(qvmin, qv[idx]));
        dqv[idx] = mynn_f32_add(dqv[idx], mynn_f32_div(dqv2, delt));
        qv[idx] = mynn_f32_add(qv[idx], dqv2);
        if (k != 0) {
            const float borrow = mynn_f32_div(
                mynn_f32_mul(dqv2, dp[idx]), dp[prev]);
            qv[prev] = mynn_f32_sub(qv[prev], borrow);
            dqv[prev] = mynn_f32_sub(
                dqv[prev], mynn_f32_div(borrow, delt));
        }
        qv[idx] = mynn_f32_max2(qv[idx], qvmin);
        qc[idx] = mynn_f32_max2(qc[idx], qcmin);
        qi[idx] = mynn_f32_max2(qi[idx], qimin);
        qs[idx] = mynn_f32_max2(qs[idx], qimin);
    }
    if (dqv2 > 1.0e-20f) {
        float total = 0.0f;
        for (int k = 0; k < nz; ++k) {
            const ulong idx = mynn_column_index(k, ncol, column);
            if (qv[idx] > mynn_f32_mul(2.0f, qvmin))
                total = mynn_f32_add(total, mynn_f32_mul(qv[idx], dp[idx]));
        }
        const ulong bottom = mynn_column_index(0, ncol, column);
        const float aa = mynn_f32_div(mynn_f32_mul(dqv2, dp[bottom]),
            mynn_f32_max2(1.0e-20f, total));
        if (aa < 0.5f) {
            for (int k = 0; k < nz; ++k) {
                const ulong idx = mynn_column_index(k, ncol, column);
                if (qv[idx] > mynn_f32_mul(2.0f, qvmin)) {
                    const float dum = mynn_f32_mul(aa, qv[idx]);
                    qv[idx] = mynn_f32_sub(qv[idx], dum);
                    dqv[idx] = mynn_f32_sub(
                        dqv[idx], mynn_f32_div(dum, delt));
                }
            }
        }
    }
}

static inline float mynn_expm1f(float x)
{
    const float ln2_hi = as_type<float>(0x3F317180u);
    const float ln2_lo = as_type<float>(0x3717F7D1u);
    const float invln2 = as_type<float>(0x3FB8AA3Bu);
    const float q1 = as_type<float>(0xBD088889u);
    const float q2 = as_type<float>(0x3AD00D01u);
    const float q3 = as_type<float>(0xB8A670CDu);
    const float q4 = as_type<float>(0x36867E54u);
    const float q5 = as_type<float>(0xB457EDBBu);
    const float tiny = 1.0e-30f;
    const uint word = as_type<uint>(x);
    const uint sign = word & 0x80000000u;
    const uint magnitude = word & 0x7fffffffu;
    if (magnitude >= 0x4195b844u) {
        if (magnitude >= 0x42b17218u) {
            if (magnitude > 0x7f800000u)
                return mynn_f32_add(x, x);
            if (magnitude == 0x7f800000u)
                return sign == 0u ? x : -1.0f;
            if (x > 8.8721679688e01f)
                return as_type<float>(0x7f800000u);
        }
        if (sign != 0u) return mynn_f32_sub(tiny, 1.0f);
    }
    int k;
    float correction;
    if (magnitude > 0x3eb17218u) {
        float hi, lo;
        if (magnitude < 0x3f851592u) {
            if (sign == 0u) {
                hi = mynn_f32_sub(x, ln2_hi);
                lo = ln2_lo;
                k = 1;
            } else {
                hi = mynn_f32_add(x, ln2_hi);
                lo = mynn_f32_negate(ln2_lo);
                k = -1;
            }
        } else {
            k = int(mynn_f32_add(mynn_f32_mul(invln2, x),
                                  sign == 0u ? 0.5f : -0.5f));
            const float scale = float(k);
            hi = mynn_f32_sub(x, mynn_f32_mul(scale, ln2_hi));
            lo = mynn_f32_mul(scale, ln2_lo);
        }
        x = mynn_f32_sub(hi, lo);
        correction = mynn_f32_sub(mynn_f32_sub(hi, x), lo);
    } else if (magnitude < 0x33000000u) {
        return x;
    } else {
        k = 0;
        correction = 0.0f;
    }

    const float hfx = mynn_f32_mul(0.5f, x);
    const float hxs = mynn_f32_mul(x, hfx);
    float polynomial = mynn_f32_add(q4, mynn_f32_mul(hxs, q5));
    polynomial = mynn_f32_add(q3, mynn_f32_mul(hxs, polynomial));
    polynomial = mynn_f32_add(q2, mynn_f32_mul(hxs, polynomial));
    polynomial = mynn_f32_add(q1, mynn_f32_mul(hxs, polynomial));
    const float r1 = mynn_f32_add(1.0f,
        mynn_f32_mul(hxs, polynomial));
    float t = mynn_f32_sub(3.0f, mynn_f32_mul(r1, hfx));
    float e = mynn_f32_mul(hxs, mynn_f32_div(
        mynn_f32_sub(r1, t),
        mynn_f32_sub(6.0f, mynn_f32_mul(x, t))));
    if (k == 0) return mynn_f32_sub(x,
        mynn_f32_sub(mynn_f32_mul(x, e), hxs));
    e = mynn_f32_sub(mynn_f32_mul(x,
        mynn_f32_sub(e, correction)), correction);
    e = mynn_f32_sub(e, hxs);
    if (k == -1) return mynn_f32_sub(
        mynn_f32_mul(0.5f, mynn_f32_sub(x, e)), 0.5f);
    if (k == 1) {
        if (x < -0.25f)
            return mynn_f32_mul(-2.0f,
                mynn_f32_sub(e, mynn_f32_add(x, 0.5f)));
        return mynn_f32_add(1.0f,
            mynn_f32_mul(2.0f, mynn_f32_sub(x, e)));
    }
    float y;
    if (k <= -2 || k > 56) {
        y = mynn_f32_sub(1.0f, mynn_f32_sub(e, x));
        y = as_type<float>(as_type<uint>(y) + (uint(k) << 23));
        return mynn_f32_sub(y, 1.0f);
    }
    if (k < 23) {
        t = as_type<float>(0x3f800000u - (0x01000000u >> uint(k)));
        y = mynn_f32_sub(t, mynn_f32_sub(e, x));
    } else {
        t = as_type<float>(uint(0x7f - k) << 23);
        y = mynn_f32_add(mynn_f32_sub(x, mynn_f32_add(e, t)), 1.0f);
    }
    return as_type<float>(as_type<uint>(y) + (uint(k) << 23));
}

static inline float mynn_tanhf(float x)
{
    const float tiny = 1.0e-30f;
    const uint word = as_type<uint>(x);
    const uint magnitude = word & 0x7fffffffu;
    float z;
    if (magnitude < 0x41b00000u) {
        if (magnitude < 0x24000000u)
            return mynn_f32_mul(x, mynn_f32_add(1.0f, x));
        const float ax = as_type<float>(magnitude);
        if (magnitude >= 0x3f800000u) {
            const float t = mynn_expm1f(mynn_f32_mul(2.0f, ax));
            z = mynn_f32_sub(1.0f, mynn_f32_div(
                2.0f, mynn_f32_add(t, 2.0f)));
        } else {
            const float t = mynn_expm1f(mynn_f32_mul(-2.0f, ax));
            z = mynn_f32_div(mynn_f32_negate(t),
                             mynn_f32_add(t, 2.0f));
        }
    } else {
        z = mynn_f32_sub(1.0f, tiny);
    }
    return as_type<float>((word & 0x80000000u) == 0u
        ? as_type<uint>(z) : (as_type<uint>(z) ^ 0x80000000u));
}

static inline void mynn_boulac_elblavg(
    device const float *zw, device const float *dz,
    device const float *qtke, device const float *thetaw,
    device float *elblavg, device float *dlu, device float *dld,
    uint ncol, int nz, uint column)
{
    const float beta = mynn_f32_div(9.81f, 300.0f);
    for (int iz = 0; iz < nz; ++iz) {
        const ulong idx = ulong(iz) * ulong(ncol) + ulong(column);
        float zup = 0.0f, zzz = 0.0f, zup_inf = 0.0f;
        dlu[idx] = mynn_f32_sub(mynn_f32_sub(
            zw[ulong(nz) * ulong(ncol) + ulong(column)], zw[idx]),
            mynn_f32_mul(dz[idx], 0.5f));
        if (iz < nz - 1) {
            int izz = iz;
            bool found = false;
            while (!found) {
                if (izz < nz - 1) {
                    const ulong izz_idx = ulong(izz) * ulong(ncol) + ulong(column);
                    const ulong izz_next = izz_idx + ulong(ncol);
                    const float dzt = dz[izz_idx];
                    zup = mynn_f32_sub(zup,
                        mynn_f32_mul(mynn_f32_mul(beta, thetaw[idx]), dzt));
                    zup = mynn_f32_add(zup, mynn_f32_mul(
                        mynn_f32_mul(mynn_f32_mul(beta,
                            mynn_f32_add(thetaw[izz_next], thetaw[izz_idx])),
                            dzt), 0.5f));
                    zzz = mynn_f32_add(zzz, dzt);
                    if (qtke[idx] < zup && qtke[idx] >= zup_inf) {
                        const float bbb = mynn_f32_div(
                            mynn_f32_sub(thetaw[izz_next], thetaw[izz_idx]), dzt);
                        float tl;
                        if (bbb != 0.0f) {
                            const float b = mynn_f32_mul(beta,
                                mynn_f32_sub(thetaw[izz_idx], thetaw[idx]));
                            const float radical = mynn_f32_max2(mynn_f32_add(
                                mynn_f32_mul(b, b), mynn_f32_mul(
                                    mynn_f32_mul(mynn_f32_mul(2.0f, bbb), beta),
                                    mynn_f32_sub(qtke[idx], zup_inf))), 0.0f);
                            tl = mynn_f32_div(mynn_f32_div(
                                mynn_f32_add(mynn_f32_negate(b),
                                    mynn_f32_sqrt(radical)), bbb), beta);
                        } else if (thetaw[izz_idx] != thetaw[idx]) {
                            tl = mynn_f32_div(mynn_f32_sub(qtke[idx], zup_inf),
                                mynn_f32_mul(beta,
                                    mynn_f32_sub(thetaw[izz_idx], thetaw[idx])));
                        } else {
                            tl = 0.0f;
                        }
                        dlu[idx] = mynn_f32_add(mynn_f32_sub(zzz, dzt), tl);
                        found = true;
                    }
                    zup_inf = zup;
                    ++izz;
                } else {
                    found = true;
                }
            }
        }

        float zdo = 0.0f, zdo_sup = 0.0f;
        dld[idx] = zw[idx];
        zzz = 0.0f;
        if (iz > 0) {
            int izz = iz;
            bool found = false;
            while (!found) {
                if (izz > 0) {
                    const ulong izz_idx = ulong(izz) * ulong(ncol) + ulong(column);
                    const ulong izz_prev = izz_idx - ulong(ncol);
                    const float dzt = dz[izz_prev];
                    zdo = mynn_f32_add(zdo,
                        mynn_f32_mul(mynn_f32_mul(beta, thetaw[idx]), dzt));
                    zdo = mynn_f32_sub(zdo, mynn_f32_mul(
                        mynn_f32_mul(mynn_f32_mul(beta,
                            mynn_f32_add(thetaw[izz_prev], thetaw[izz_idx])),
                            dzt), 0.5f));
                    zzz = mynn_f32_add(zzz, dzt);
                    if (qtke[idx] < zdo && qtke[idx] >= zdo_sup) {
                        const float bbb = mynn_f32_div(
                            mynn_f32_sub(thetaw[izz_idx], thetaw[izz_prev]), dzt);
                        float tl;
                        if (bbb != 0.0f) {
                            const float b = mynn_f32_mul(beta,
                                mynn_f32_sub(thetaw[izz_idx], thetaw[idx]));
                            const float radical = mynn_f32_max2(mynn_f32_add(
                                mynn_f32_mul(b, b), mynn_f32_mul(
                                    mynn_f32_mul(mynn_f32_mul(2.0f, bbb), beta),
                                    mynn_f32_sub(qtke[idx], zdo_sup))), 0.0f);
                            tl = mynn_f32_div(mynn_f32_div(
                                mynn_f32_add(b, mynn_f32_sqrt(radical)), bbb), beta);
                        } else if (thetaw[izz_idx] != thetaw[idx]) {
                            tl = mynn_f32_div(mynn_f32_sub(qtke[idx], zdo_sup),
                                mynn_f32_mul(beta,
                                    mynn_f32_sub(thetaw[izz_idx], thetaw[idx])));
                        } else {
                            tl = 0.0f;
                        }
                        dld[idx] = mynn_f32_add(mynn_f32_sub(zzz, dzt), tl);
                        found = true;
                    }
                    zdo_sup = zdo;
                    --izz;
                } else {
                    found = true;
                }
            }
        }
        dld[idx] = mynn_f32_min2(dld[idx],
            zw[ulong(iz + 1) * ulong(ncol) + ulong(column)]);
        const float up = mynn_f32_max2(mynn_f32_min2(dlu[idx], 1000.0f), 0.1f);
        const float down = mynn_f32_max2(mynn_f32_min2(dld[idx], 1000.0f), 0.1f);
        float value = mynn_f32_sqrt(mynn_f32_mul(up, down));
        value = mynn_f32_div(value,
            mynn_f32_add(1.0f, mynn_f32_div(value, 2000.0f)));
        elblavg[idx] = iz == nz - 1
            ? elblavg[idx - ulong(ncol)] : value;
    }
}

static inline void mynn_mym_length(
    device const float *dz, device const float *zw,
    device const float *u, device const float *v,
    device const float *qke, device const float *dtv,
    device const float *theta, device const float *edmf_w,
    device const float *edmf_a, device float *el, device float *qkw,
    device float *qtke, device float *thetaw, device float *elblavg,
    device float *dlu, device float *dld, float rmo, float fltv,
    float zi, float psig_bl, uint ncol, int nz, uint column,
    float cns, float alp1)
{
    const float gtr = mynn_f32_div(9.81f, 300.0f);
    const ulong base = ulong(column);
    float ugrid = mynn_f32_sqrt(mynn_f32_add(
        mynn_f32_mul(u[base], u[base]), mynn_f32_mul(v[base], v[base])));
    const float wt_u = mynn_f32_sub(1.0f, mynn_f32_min2(
        mynn_f32_div(mynn_f32_max2(mynn_f32_sub(ugrid, 15.0f), 0.0f), 30.0f),
        0.5f));
    const float alp3 = mynn_f32_mul(2.5f, wt_u);
    const float zi2 = mynn_f32_max2(zi, 300.0f);
    const float h1 = mynn_f32_min2(
        mynn_f32_max2(mynn_f32_mul(0.3f, zi2), 300.0f), 600.0f);
    const float h2 = mynn_f32_div(h1, 2.0f);
    qtke[base] = mynn_f32_max2(mynn_f32_mul(0.5f, qke[base]), 0.5e-3f);
    thetaw[base] = theta[base];
    qkw[base] = mynn_f32_sqrt(mynn_f32_max2(qke[base], 1.0e-3f));
    for (int k = 1; k < nz; ++k) {
        const ulong idx = ulong(k) * ulong(ncol) + ulong(column);
        const ulong prev = idx - ulong(ncol);
        const float afk = mynn_f32_div(dz[idx],
            mynn_f32_add(dz[idx], dz[prev]));
        const float abk = mynn_f32_sub(1.0f, afk);
        qkw[idx] = mynn_f32_sqrt(mynn_f32_max2(mynn_f32_add(
            mynn_f32_mul(qke[idx], abk), mynn_f32_mul(qke[prev], afk)),
            1.0e-3f));
        qtke[idx] = mynn_f32_max2(mynn_f32_mul(0.5f,
            mynn_f32_mul(qkw[idx], qkw[idx])), 0.005f);
        thetaw[idx] = mynn_f32_add(mynn_f32_mul(theta[idx], abk),
                                   mynn_f32_mul(theta[prev], afk));
    }

    float elt = 1.0e-5f, vsc_sum = 1.0e-5f;
    int k = 1;
    while (k < nz && zw[ulong(k) * ulong(ncol) + ulong(column)]
                    <= mynn_f32_add(zi2, h1)) {
        const ulong idx = ulong(k) * ulong(ncol) + ulong(column);
        const ulong prev = idx - ulong(ncol);
        const float dzk = mynn_f32_mul(0.5f,
            mynn_f32_add(dz[idx], dz[prev]));
        const float qdz = mynn_f32_mul(
            mynn_f32_min2(mynn_f32_max2(qkw[idx], 0.01f), 30.0f), dzk);
        elt = mynn_f32_add(elt, mynn_f32_mul(qdz,
            zw[ulong(k) * ulong(ncol) + ulong(column)]));
        vsc_sum = mynn_f32_add(vsc_sum, qdz);
        ++k;
    }
    elt = mynn_f32_min2(mynn_f32_max2(mynn_f32_div(
        mynn_f32_mul(alp1, elt), vsc_sum), 8.0f), 400.0f);
    const float vsc = mynn_gfk_powf(mynn_f32_mul(mynn_f32_mul(gtr, elt),
        mynn_f32_max2(fltv, 0.0f)), mynn_f32_div(1.0f, 3.0f));
    mynn_boulac_elblavg(zw, dz, qtke, thetaw, elblavg, dlu, dld,
                        ncol, nz, column);

    el[base] = 0.0f;
    for (k = 1; k < nz; ++k) {
        const ulong idx = ulong(k) * ulong(ncol) + ulong(column);
        const ulong prev = idx - ulong(ncol);
        const float zwk = zw[idx];
        float elb, elf;
        if (dtv[idx] > 0.0f) {
            const float bv = mynn_f32_max2(mynn_f32_sqrt(
                mynn_f32_mul(gtr, dtv[idx])), 0.0001f);
            const float numerator = mynn_f32_max2(
                mynn_f32_mul(0.3f, mynn_f32_max2(qkw[idx], 0.018f)),
                mynn_f32_mul(mynn_f32_mul(50.0f, edmf_a[prev]), edmf_w[prev]));
            elb = mynn_f32_mul(mynn_f32_div(numerator, bv), mynn_f32_add(
                1.0f, mynn_f32_mul(alp3, mynn_f32_sqrt(mynn_f32_div(
                    vsc, mynn_f32_mul(bv, elt))))));
            elb = mynn_f32_min2(elb, zwk);
            elf = mynn_f32_div(mynn_f32_max2(qkw[idx], 0.018f), bv);
            elblavg[idx] = mynn_f32_max2(elblavg[idx], mynn_f32_div(
                mynn_f32_mul(mynn_f32_mul(50.0f, edmf_a[prev]), edmf_w[prev]),
                bv));
        } else {
            elb = 1.0e10f;
            elf = elb;
        }
        float els;
        if (rmo > 0.0f) {
            els = mynn_f32_div(mynn_f32_mul(0.4f, zwk), mynn_f32_add(1.0f,
                mynn_f32_mul(cns,
                    mynn_f32_min2(mynn_f32_mul(zwk, rmo), 1.0f))));
        } else {
            els = mynn_f32_mul(mynn_f32_mul(0.4f, zwk),
                mynn_gfk_powf(mynn_f32_sub(1.0f,
                    mynn_f32_mul(mynn_f32_mul(5.0f, zwk), rmo)), 0.2f));
        }
        const float weight = mynn_f32_add(mynn_f32_mul(0.5f,
            mynn_tanhf(mynn_f32_div(mynn_f32_sub(zwk,
                mynn_f32_add(zi2, h1)), h2))), 0.5f);
        float value = mynn_f32_sqrt(mynn_f32_div(
            mynn_f32_mul(els, els), mynn_f32_add(1.0f,
                mynn_f32_div(mynn_f32_mul(els, els),
                             mynn_f32_mul(elt, elt)))));
        value = mynn_f32_min2(mynn_f32_min2(value, elb), elf);
        value = mynn_f32_add(mynn_f32_mul(value,
            mynn_f32_sub(1.0f, weight)), mynn_f32_mul(
                mynn_f32_mul(0.3f, elblavg[idx]), weight));
        el[idx] = mynn_f32_mul(value, psig_bl);
    }
}

// WRF v4.6.1 module_bl_mynn.F:2100-2232 mym_length CASE(2).
// See licenses/LICENSE-WRF-public-domain.txt.
static inline void mynn_mym_length_local(
    device const float *dz, device const float *zw,
    device const float *qke, device const float *dtv,
    device const float *edmf_w, device const float *edmf_a,
    float rmo, float fltv, float zi, float psig_bl,
    device float *el, device float *qkw, device float *qtke,
    uint ncol, int nz, uint column)
{
    const ulong base = ulong(column), stride = ulong(ncol);
    const float gtr = mynn_f32_div(9.81f, 300.0f);
    const float onethird = mynn_f32_div(1.0f, 3.0f);
    float zi2 = mynn_f32_max2(zi, 300.0f);
    float h1 = mynn_f32_min2(mynn_f32_max2(mynn_f32_mul(0.3f, zi2), 300.0f), 600.0f);
    float h2 = mynn_f32_mul(h1, 0.5f);
    qtke[base] = mynn_f32_max2(mynn_f32_mul(0.5f, qke[base]), mynn_f32_mul(0.5f, 1.0e-3f));
    qkw[base] = mynn_f32_sqrt(mynn_f32_max2(qke[base], 1.0e-3f));
    for (int k = 1; k < nz; ++k) {
        const ulong idx = ulong(k) * stride + base, prev = idx - stride;
        float afk = mynn_f32_div(dz[idx], mynn_f32_add(dz[idx], dz[prev]));
        float abk = mynn_f32_sub(1.0f, afk);
        qkw[idx] = mynn_f32_sqrt(mynn_f32_max2(mynn_f32_add(
            mynn_f32_mul(qke[idx], abk), mynn_f32_mul(qke[prev], afk)), 1.0e-3f));
        qtke[idx] = mynn_f32_mul(0.5f, mynn_f32_mul(qkw[idx], qkw[idx]));
    }
    float elt = 1.0e-5f, vsc_sum = 1.0e-5f;
    float pblh_plus_ent = mynn_f32_max2(mynn_f32_add(zi, h1), 100.0f);
    for (int k = 1; k < nz && zw[ulong(k) * stride + base] <= pblh_plus_ent; ++k) {
        const ulong idx = ulong(k) * stride + base, prev = idx - stride;
        float dzk = mynn_f32_mul(0.5f, mynn_f32_add(dz[idx], dz[prev]));
        float qdz = mynn_f32_mul(mynn_f32_min2(mynn_f32_max2(qkw[idx], 0.03f), 30.0f), dzk);
        elt = mynn_f32_add(elt, mynn_f32_mul(qdz, zw[idx]));
        vsc_sum = mynn_f32_add(vsc_sum, qdz);
    }
    elt = mynn_f32_min2(mynn_f32_max2(mynn_f32_div(mynn_f32_mul(0.22f, elt), vsc_sum),
                              10.0f), 400.0f);
    float vsc = mynn_gfk_powf(mynn_f32_mul(mynn_f32_mul(gtr, elt), mynn_f32_max2(fltv, 0.0f)),
                         onethird);
    el[base] = 0.0f;
    for (int k = 1; k < nz; ++k) {
        const ulong idx = ulong(k) * stride + base, prev = idx - stride;
        float zwk = zw[idx];
        float dzk = mynn_f32_mul(0.5f, mynn_f32_add(dz[idx], dz[prev]));
        float wstar = mynn_f32_mul(1.25f, mynn_gfk_powf(
            mynn_f32_mul(mynn_f32_mul(gtr, zi), mynn_f32_max2(fltv, 1.0e-4f)), onethird));
        float weight = mynn_f32_add(mynn_f32_mul(0.5f, mynn_tanhf(
            mynn_f32_div(mynn_f32_sub(zwk, mynn_f32_add(zi2, h1)), h2))), 0.5f);
        float elb_mf, elf;
        if (dtv[idx] > 0.0f) {
            float bv = mynn_f32_max2(mynn_f32_sqrt(mynn_f32_mul(gtr, dtv[idx])), 0.001f);
            float numerator = mynn_f32_max2(mynn_f32_mul(0.30f, qkw[idx]),
                mynn_f32_mul(mynn_f32_mul(50.0f, edmf_a[prev]), edmf_w[prev]));
            elb_mf = mynn_f32_mul(mynn_f32_div(numerator, bv), mynn_f32_add(1.0f,
                mynn_f32_mul(2.0f, mynn_f32_sqrt(mynn_f32_div(vsc, mynn_f32_mul(bv, elt))))));
            float tau = mynn_f32_min2(mynn_f32_max2(
                mynn_f32_div(mynn_f32_mul(1000.0f, wstar), 9.81f), 30.0f), 150.0f);
            tau = mynn_f32_add(mynn_f32_mul(tau, mynn_f32_sub(1.0f, weight)),
                           mynn_f32_mul(50.0f, weight));
            elf = mynn_f32_min2(mynn_f32_max2(
                mynn_f32_mul(tau, mynn_f32_sqrt(mynn_f32_min2(qtke[idx], 40.0f))),
                mynn_f32_div(mynn_f32_mul(mynn_f32_mul(50.0f, edmf_a[idx]), edmf_w[idx]), bv)),
                zwk);
        } else {
            float tau = mynn_f32_min2(mynn_f32_max2(
                mynn_f32_div(mynn_f32_mul(1000.0f, wstar), 9.81f), 50.0f), 200.0f);
            tau = mynn_f32_add(mynn_f32_mul(tau, mynn_f32_sub(1.0f, weight)),
                mynn_f32_mul(mynn_f32_max2(100.0f, mynn_f32_mul(dzk, 0.25f)), weight));
            elf = mynn_f32_min2(mynn_f32_mul(tau, mynn_f32_sqrt(mynn_f32_min2(qtke[idx], 40.0f))), zwk);
            elb_mf = elf;
        }
        elf = mynn_f32_div(elf, mynn_f32_add(1.0f, mynn_f32_div(elf, 800.0f)));
        elb_mf = mynn_f32_max2(elb_mf, 0.01f);
        float els;
        if (rmo > 0.0f) {
            els = mynn_f32_div(mynn_f32_mul(0.4f, zwk), mynn_f32_add(1.0f,
                mynn_f32_mul(3.5f, mynn_f32_min2(mynn_f32_mul(zwk, rmo), 1.0f))));
        } else {
            els = mynn_f32_mul(mynn_f32_mul(0.4f, zwk), mynn_gfk_powf(mynn_f32_sub(1.0f,
                mynn_f32_mul(mynn_f32_mul(5.0f, zwk), rmo)), 0.2f));
        }
        float els2 = mynn_f32_mul(els, els);
        float value = mynn_f32_sqrt(mynn_f32_div(els2, mynn_f32_add(mynn_f32_add(1.0f,
            mynn_f32_div(els2, mynn_f32_mul(elt, elt))),
            mynn_f32_div(els2, mynn_f32_mul(elb_mf, elb_mf)))));
        value = mynn_f32_add(mynn_f32_mul(value, mynn_f32_sub(1.0f, weight)),
                         mynn_f32_mul(elf, weight));
        float el_les = mynn_f32_min2(mynn_f32_div(els,
            mynn_f32_add(1.0f, mynn_f32_div(els, 12.0f))), elb_mf);
        el[idx] = mynn_f32_add(mynn_f32_mul(value, psig_bl),
                         mynn_f32_mul(mynn_f32_sub(1.0f, psig_bl), el_les));
    }
}

// WRF v4.1 module_bl_mynn.F:mym_length CASE(2), the GSD experimental
// cloud-aware mixing length. Keep this separate from the RAP/HRRR CASE(1)
// helper above: its cloud-regime buoyancy length, cloud turnover length,
// and LES/asymptotic scale blend are a distinct formulation.
static inline void mynn_mym_length_gsd41(
    device const float *dz, device const float *zw,
    device const float *qke, device const float *dtv,
    device const float *theta, device const float *edmf_w,
    device const float *edmf_a, device const float *cldfra,
    device const float *vt, device const float *vq,
    device float *el, device float *qkw, device float *qtke,
    device float *thetaw, float rmo, float flt, float flq,
    float zi, float psig_bl, uint ncol, int nz, uint column,
    bool zero_profile_inputs = false)
{
    const ulong base = ulong(column);
    const float gtr = mynn_f32_div(9.81f, 300.0f);
    const float cns = 3.5f, alp2 = 0.3f;
    const float alp3 = 2.0f, alp4 = 20.0f, alp5 = 0.3f;
    const float alp6 = 50.0f, qmin = 0.0f, zmax = 1.0f;
    const float alp1 = mynn_f32_add(0.25f, mynn_f32_div(mynn_f32_mul(0.02f,
        mynn_f32_min2(mynn_f32_max2(mynn_f32_sub(zi, 200.0f), 0.0f),
                      1000.0f)), 1000.0f));
    const float zi2 = mynn_f32_max2(zi, 100.0f);
    const float h1 = mynn_f32_min2(mynn_f32_max2(
        mynn_f32_mul(0.3f, zi2), 300.0f), 750.0f);
    const float h2 = mynn_f32_mul(0.5f, h1);
    qtke[base] = mynn_f32_max2(mynn_f32_mul(0.5f, qke[base]), 0.01f);
    thetaw[base] = theta[base];
    qkw[base] = mynn_f32_sqrt(mynn_f32_max2(qke[base], 1.0e-10f));
    for (int k = 1; k < nz; ++k) {
        const ulong idx = ulong(k) * ulong(ncol) + ulong(column);
        const ulong prev = idx - ulong(ncol);
        const float dzsum = mynn_f32_add(dz[idx], dz[prev]);
        const float afk = mynn_f32_div(dz[idx], dzsum);
        const float abk = mynn_f32_sub(1.0f, afk);
        qkw[idx] = mynn_f32_sqrt(mynn_f32_max2(mynn_f32_add(
            mynn_f32_mul(qke[idx], abk), mynn_f32_mul(qke[prev], afk)),
            1.0e-3f));
        qtke[idx] = mynn_f32_mul(0.5f, mynn_f32_mul(qkw[idx], qkw[idx]));
        thetaw[idx] = mynn_f32_add(mynn_f32_mul(theta[idx], abk),
                                   mynn_f32_mul(theta[prev], afk));
    }

    const float pblh_plus_entr = mynn_f32_max2(
        mynn_f32_add(zi, h1), 100.0f);
    float elt = 1.0e-5f, vsc_sum = 1.0e-5f;
    for (int k = 1; k < nz; ++k) {
        const ulong idx = ulong(k) * ulong(ncol) + ulong(column);
        if (zw[idx] > pblh_plus_entr) break;
        const ulong prev = idx - ulong(ncol);
        const float dzk = mynn_f32_mul(0.5f,
            mynn_f32_add(dz[idx], dz[prev]));
        const float qdz = mynn_f32_mul(mynn_f32_max2(
            mynn_f32_sub(qkw[idx], qmin), 0.03f), dzk);
        elt = mynn_f32_add(elt, mynn_f32_mul(qdz, zw[idx]));
        vsc_sum = mynn_f32_add(vsc_sum, qdz);
    }
    elt = mynn_f32_max2(mynn_f32_div(mynn_f32_mul(alp1, elt), vsc_sum),
                        10.0f);
    const float tv0 = mynn_f32_mul(
        mynn_f32_sub(mynn_f32_div(461.6f, 287.0f), 1.0f), 300.0f);
    const float vflx = zero_profile_inputs ? 0.0f : mynn_f32_add(
        mynn_f32_mul(mynn_f32_add(vt[base], 1.0f), flt), mynn_f32_mul(
        mynn_f32_add(vq[base], tv0), flq));
    const float vsc = mynn_gfk_powf(mynn_f32_mul(mynn_f32_mul(gtr, elt),
        mynn_f32_max2(vflx, 0.0f)), 0.33333f);
    el[base] = 0.0f;
    for (int k = 1; k < nz; ++k) {
        const ulong idx = ulong(k) * ulong(ncol) + ulong(column);
        const ulong prev = idx - ulong(ncol);
        const float zwk = zw[idx];
        float elb_mf, elf;
        if (dtv[idx] > 0.0f) {
            const float bv = mynn_f32_sqrt(mynn_f32_mul(gtr, dtv[idx]));
            const float plume = zero_profile_inputs ? 0.0f : mynn_f32_mul(
                mynn_f32_mul(alp6, edmf_a[idx]), edmf_w[idx]);
            const float numerator = mynn_f32_max2(
                mynn_f32_mul(alp2, qkw[idx]), plume);
            elb_mf = mynn_f32_mul(mynn_f32_div(numerator, bv),
                mynn_f32_add(1.0f, mynn_f32_mul(alp3,
                    mynn_f32_sqrt(mynn_f32_div(vsc,
                        mynn_f32_mul(bv, elt))))));
            const float elb = mynn_f32_min2(mynn_f32_div(
                mynn_f32_mul(alp5, qkw[idx]), bv), zwk);
            elf = mynn_f32_div(elb, mynn_f32_add(1.0f,
                                                  mynn_f32_div(elb, 600.0f)));
        } else {
            const float zi_flux = mynn_f32_mul(mynn_f32_mul(gtr, zi),
                mynn_f32_max2(flt, 1.0e-4f));
            const float tau_raw = mynn_f32_div(mynn_f32_mul(0.5f, zi),
                mynn_gfk_powf(zi_flux, 0.3333f));
            float tau_cloud = mynn_f32_min2(mynn_f32_max2(tau_raw,
                                                           50.0f), 150.0f);
            const float weight = mynn_f32_add(mynn_f32_mul(0.5f,
                mynn_tanhf(mynn_f32_div(mynn_f32_sub(zwk,
                    mynn_f32_add(zi2, h1)), h2))), 0.5f);
            tau_cloud = mynn_f32_add(mynn_f32_mul(tau_cloud,
                mynn_f32_sub(1.0f, weight)), mynn_f32_mul(50.0f, weight));
            elb_mf = mynn_f32_min2(mynn_f32_mul(tau_cloud,
                mynn_f32_sqrt(mynn_f32_min2(qtke[idx], 30.0f))), zwk);
            elf = elb_mf;
        }
        float els;
        float els1;
        const float zm = mynn_f32_max2(mynn_f32_sub(zwk, 4.0f), 0.0f);
        if (rmo > 0.0f) {
            const float den = mynn_f32_add(1.0f, mynn_f32_mul(cns,
                mynn_f32_min2(mynn_f32_mul(zwk, rmo), zmax)));
            els = mynn_f32_div(mynn_f32_mul(0.4f, zwk), den);
            els1 = mynn_f32_div(mynn_f32_mul(0.4f, zm), den);
        } else {
            const float stable = mynn_gfk_powf(mynn_f32_sub(1.0f,
                mynn_f32_mul(mynn_f32_mul(alp4, zwk), rmo)), 0.2f);
            els = mynn_f32_mul(mynn_f32_mul(0.4f, zwk), stable);
            els1 = mynn_f32_mul(mynn_f32_mul(0.4f, zm), stable);
        }
        const float weight = mynn_f32_add(mynn_f32_mul(0.5f,
            mynn_tanhf(mynn_f32_div(mynn_f32_sub(zwk,
                mynn_f32_add(zi2, h1)), h2))), 0.5f);
        const float el_unstab = mynn_f32_div(els,
            mynn_f32_add(1.0f, mynn_f32_div(els1, elt)));
        float value = mynn_f32_min2(el_unstab, elb_mf);
        value = mynn_f32_add(mynn_f32_mul(value,
            mynn_f32_sub(1.0f, weight)), mynn_f32_mul(elf, weight));
        const float el_les = mynn_f32_min2(mynn_f32_div(els,
            mynn_f32_add(1.0f, mynn_f32_div(els1, 12.0f))), elb_mf);
        el[idx] = mynn_f32_add(mynn_f32_mul(value, psig_bl),
            mynn_f32_mul(mynn_f32_sub(1.0f, psig_bl), el_les));
    }
}

kernel void mynn_mixlength_default_columns(
    constant MynnMixlengthDefaultColumnsResources &r [[buffer(0)]],
    constant MynnMixlengthDefaultColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    if (column >= uint(q.ncol) || q.nz < 2) return;
    if (q.mixlength == 2) {
        mynn_mym_length_local(r.dz, r.zw, r.qke, r.dtv, r.edmf_w,
            r.edmf_a, r.rmo[column], r.fltv[column], r.zi[column],
            r.psig_bl[column], r.el, r.qkw, r.qtke, uint(q.ncol), q.nz, column);
    } else {
    mynn_mym_length(r.dz, r.zw, r.u, r.v, r.qke, r.dtv, r.theta,
        r.edmf_w, r.edmf_a, r.el, r.qkw, r.qtke, r.thetaw, r.elblavg,
        r.dlu, r.dld, r.rmo[column], r.fltv[column], r.zi[column],
        r.psig_bl[column], uint(q.ncol), q.nz, column, q.cns, q.alp1);
    }
}

kernel void mynn_mixlength_gsd41_columns(
    constant MynnMixlengthGsd41ColumnsResources &r [[buffer(0)]],
    constant MynnMixlengthGsd41ColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    if (column >= uint(q.ncol) || q.nz < 2) return;
    mynn_mym_length_gsd41(r.dz, r.zw, r.qke, r.dtv, r.theta,
        r.edmf_w, r.edmf_a, r.cldfra, r.vt, r.vq, r.el, r.qkw, r.qtke,
        r.thetaw, r.rmo[column], r.flt[column], r.flq[column],
        r.zi[column], r.psig_bl[column], uint(q.ncol), q.nz, column);
}

// Cold-start proposal for module_bl_mynn.F:mym_initialize. This deliberately
// uses the pinned portable double pow candidate and narrows only once. The
// candidate is not asserted equivalent to CUDA libdevice's binary64 pow.
static inline float mynn_initialize_powf_candidate(float x, float y)
{
    return mynn_f64_to_f32(soft64_plm_pow(mynn_f64_from_f32(x),
                                          mynn_f64_from_f32(y)));
}

template <typename Resources, typename Params>
// Length laws: 1 is WRF nonlocal, 2 is GSD 4.1, 3 is WRF local.
static inline void mynn_initialize_column(
    constant Resources &r, constant Params &q, uint column, int mixlength)
{
    if (column >= uint(q.ncol) || q.nz < 2) return;
    const int nz = q.nz;
    const ulong ncol = ulong(q.ncol);
    const ulong base = ulong(column);
    device float *qkw = r.scratch + 0 * nz * ncol;
    device float *qtke = r.scratch + 1 * nz * ncol;
    device float *thetaw = r.scratch + 2 * nz * ncol;
    device float *elblavg = r.scratch + 3 * nz * ncol;
    device float *dlu = r.scratch + 4 * nz * ncol;
    device float *dld = r.scratch + 5 * nz * ncol;
    device float *dtl = r.scratch + 6 * nz * ncol;
    device float *dqw = r.scratch + 7 * nz * ncol;
    device float *dtv = r.scratch + 8 * nz * ncol;
    device float *gm = r.scratch + 9 * nz * ncol;
    device float *gh = r.scratch + 10 * nz * ncol;
    device float *pdk = r.scratch + 11 * nz * ncol;
    device float *pdt = r.scratch + 12 * nz * ncol;
    device float *pdq = r.scratch + 13 * nz * ncol;
    device float *pdc = r.scratch + 14 * nz * ncol;
    const float b1 = 24.0f, b2 = 15.0f, g1 = 0.235f;
    const float pr = q.prandtl, c2 = 0.729f, c3 = 0.340f, c5 = 0.2f;
    const float a1 = mynn_f32_div(
        mynn_f32_mul(b1, mynn_f32_sub(1.0f, mynn_f32_mul(3.0f, g1))), 6.0f);
    const float c1 = mynn_f32_sub(g1, mynn_f32_div(1.0f,
        mynn_f32_mul(mynn_f32_mul(3.0f, a1), 2.88449914061481660f)));
    const float a2 = mynn_f32_div(mynn_f32_mul(a1, mynn_f32_sub(g1, c1)),
                                  mynn_f32_mul(g1, pr));
    const float g2 = mynn_f32_add(
        mynn_f32_mul(mynn_f32_div(b2, b1), mynn_f32_sub(1.0f, c3)),
        mynn_f32_mul(mynn_f32_div(mynn_f32_mul(2.0f, a1), b1),
                     mynn_f32_sub(3.0f, mynn_f32_mul(2.0f, c2))));
    const float tv0 = mynn_f32_mul(
        mynn_f32_sub(mynn_f32_div(461.6f, 287.0f), 1.0f), 300.0f);
    const float gtr = mynn_f32_div(9.81f, 300.0f);
    const float ust = r.ust[base], rmo = r.rmo[base], zi = r.zi[base];
    const float psig = r.psig_bl[base];
    for (int k = 0; k < nz; ++k) {
        const ulong idx = ulong(k) * ncol + base;
        r.sm[idx] = r.sm_in[idx]; r.sh[idx] = r.sh_in[idx];
        r.qke[idx] = r.qke_in[idx];
        r.el[idx] = 0.0f; r.tsq[idx] = 0.0f;
        r.qsq[idx] = 0.0f; r.cov[idx] = 0.0f;
        dtl[idx] = 0.0f; dqw[idx] = 0.0f; dtv[idx] = 0.0f;
        gm[idx] = 0.0f; gh[idx] = 0.0f;
        pdk[idx] = 0.0f; pdt[idx] = 0.0f;
        pdq[idx] = 0.0f; pdc[idx] = 0.0f;
    }
    // Level-2 closure from mym_initialize uses zero-weighted VT/VQ locals.
    const float rfc = mynn_f32_div(g1, mynn_f32_add(g1, g2));
    for (int k = 1; k < nz; ++k) {
        const ulong idx = ulong(k) * ncol + base, prev = idx - ncol;
        const float dz_sum = mynn_f32_add(r.dz[idx], r.dz[prev]);
        const float dzk = mynn_f32_mul(0.5f, dz_sum);
        const float afk = mynn_f32_div(r.dz[idx], dz_sum);
        const float abk = mynn_f32_sub(1.0f, afk);
        const float du = mynn_f32_sub(r.u[idx], r.u[prev]);
        const float dv = mynn_f32_sub(r.v[idx], r.v[prev]);
        const float duz = mynn_f32_div(mynn_f32_add(
            mynn_f32_mul(du, du), mynn_f32_mul(dv, dv)),
            mynn_f32_mul(dzk, dzk));
        const float dtz = mynn_f32_div(mynn_f32_sub(r.thl[idx], r.thl[prev]), dzk);
        const float dqz = mynn_f32_div(mynn_f32_sub(r.qw[idx], r.qw[prev]), dzk);
        const float vtt = mynn_f32_add(mynn_f32_add(1.0f,
            mynn_f32_mul(0.0f, abk)), mynn_f32_mul(0.0f, afk));
        const float vqq = mynn_f32_add(mynn_f32_add(tv0,
            mynn_f32_mul(0.0f, abk)), mynn_f32_mul(0.0f, afk));
        const float dtq = mynn_f32_add(mynn_f32_mul(vtt, dtz),
                                       mynn_f32_mul(vqq, dqz));
        const float level_gh = mynn_f32_negate(mynn_f32_mul(dtq, gtr));
        const float ri = mynn_f32_div(mynn_f32_negate(level_gh),
            mynn_f32_max2(duz, 1.0e-10f));
        const float a2fac = mynn_f32_div(1.0f, mynn_f32_add(1.0f,
            mynn_f32_max2(ri, 0.0f)));
        const float f1 = mynn_f32_add(mynn_f32_add(mynn_f32_mul(b1,
            mynn_f32_sub(g1, c1)), mynn_f32_mul(mynn_f32_mul(
            mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(3.0f, a2), a2fac),
            mynn_f32_sub(1.0f, c2)), mynn_f32_sub(1.0f, c5)), 1.0f)),
            mynn_f32_mul(mynn_f32_mul(2.0f, a1),
            mynn_f32_sub(3.0f, mynn_f32_mul(2.0f, c2))));
        const float f2 = mynn_f32_sub(mynn_f32_mul(b1,
            mynn_f32_add(g1, g2)), mynn_f32_mul(mynn_f32_mul(3.0f, a1),
            mynn_f32_sub(1.0f, c2)));
        const float rf1 = mynn_f32_div(mynn_f32_mul(b1,
            mynn_f32_sub(g1, c1)), f1);
        const float rf2 = mynn_f32_div(mynn_f32_mul(b1, g1), f2);
        const float smc = mynn_f32_div(mynn_f32_mul(mynn_f32_div(a1,
            mynn_f32_mul(a2, a2fac)), f1), f2);
        const float shc = mynn_f32_mul(mynn_f32_mul(3.0f,
            mynn_f32_mul(a2, a2fac)), mynn_f32_add(g1, g2));
        const float ri1 = mynn_f32_div(0.5f, smc);
        const float ri2 = mynn_f32_mul(rf1, smc);
        const float ri3 = mynn_f32_sub(mynn_f32_mul(mynn_f32_mul(4.0f, rf2),
            smc), mynn_f32_mul(2.0f, ri2));
        const float ri4 = mynn_f32_mul(ri2, ri2);
        const float radical = mynn_f32_add(mynn_f32_sub(
            mynn_f32_mul(ri, ri), mynn_f32_mul(ri3, ri)), ri4);
        const float rf = mynn_f32_min2(mynn_f32_mul(ri1,
            mynn_f32_sub(mynn_f32_add(ri, ri2), mynn_f32_sqrt(radical))), rfc);
        const float sh = mynn_f32_div(mynn_f32_mul(shc,
            mynn_f32_sub(rfc, rf)), mynn_f32_sub(1.0f, rf));
        const float sm = mynn_f32_mul(mynn_f32_div(mynn_f32_mul(smc,
            mynn_f32_sub(rf1, rf)), mynn_f32_sub(rf2, rf)), sh);
        dtl[idx] = dtz; dqw[idx] = dqz; dtv[idx] = dtq;
        gm[idx] = duz; gh[idx] = level_gh; r.sm[idx] = sm; r.sh[idx] = sh;
    }
    if (q.initialize_qke != 0) {
        r.qke[base] = mynn_f32_mul(mynn_f32_mul(1.5f,
            mynn_f32_mul(ust, ust)), mynn_initialize_powf_candidate(
                24.0f, mynn_f32_div(2.0f, 3.0f)));
        for (int k = 1; k < nz; ++k) {
            const ulong idx = ulong(k) * ncol + base;
            const float taper = mynn_f32_div(mynn_f32_sub(
                mynn_f32_mul(ust, 700.0f), r.zw[idx]),
                mynn_f32_mul(mynn_f32_max2(ust, 0.01f), 700.0f));
            r.qke[idx] = mynn_f32_mul(r.qke[base],
                                      mynn_f32_max2(taper, 0.01f));
        }
    }
    const float phm0 = mynn_f32_div(15.0f,
        mynn_initialize_powf_candidate(24.0f, mynn_f32_div(1.0f, 3.0f)));
    r.tsq[base] = mynn_f32_mul(phm0, mynn_f32_mul(
        mynn_f32_div(0.0f, ust), mynn_f32_div(0.0f, ust)));
    r.qsq[base] = r.tsq[base];
    r.cov[base] = mynn_f32_mul(mynn_f32_mul(phm0,
        mynn_f32_div(0.0f, ust)), mynn_f32_div(0.0f, ust));
    for (int k = 1; k < nz; ++k) {
        const ulong idx = ulong(k) * ncol + base;
        const float vkz = mynn_f32_mul(0.4f, r.zw[idx]);
        r.el[idx] = mynn_f32_div(vkz, mynn_f32_add(1.0f,
                                                   mynn_f32_div(vkz, 100.0f)));
    }
    const float qkemin = mixlength == 2 ? 1.0e-12f : 1.0e-3f;
    for (int iteration = 0; iteration < 5; ++iteration) {
        if (mixlength == 3) {
            mynn_mym_length_local(r.dz, r.zw, r.qke, dtv, r.edmf_w,
                r.edmf_a, rmo, 0.0f, zi, psig, r.el, qkw, qtke,
                uint(q.ncol), nz, column);
        } else if (mixlength == 2) {
            mynn_mym_length_gsd41(r.dz, r.zw, r.qke, dtv, r.theta,
                nullptr, nullptr, nullptr, nullptr, nullptr,
                r.el, qkw, qtke, thetaw, rmo, 0.0f, 0.0f, zi, psig,
                uint(q.ncol), nz, column, true);
        } else {
            mynn_mym_length(r.dz, r.zw, r.u, r.v, r.qke, dtv, r.theta,
                r.edmf_w, r.edmf_a, r.el, qkw, qtke, thetaw, elblavg, dlu, dld,
                rmo, 0.0f, zi, psig, uint(q.ncol), nz, column, q.cns, q.alp1);
        }
        for (int k = 1; k < nz; ++k) {
            const ulong idx = ulong(k) * ncol + base;
            const float elq = mynn_f32_mul(r.el[idx], qkw[idx]);
            pdk[idx] = mynn_f32_mul(elq, mynn_f32_add(
                mynn_f32_mul(r.sm[idx], gm[idx]),
                mynn_f32_mul(r.sh[idx], gh[idx])));
            pdt[idx] = mynn_f32_mul(mynn_f32_mul(elq, r.sh[idx]),
                                     mynn_f32_mul(dtl[idx], dtl[idx]));
            pdq[idx] = mynn_f32_mul(mynn_f32_mul(elq, r.sh[idx]),
                                     mynn_f32_mul(dqw[idx], dqw[idx]));
            pdc[idx] = mynn_f32_mul(mynn_f32_mul(
                mynn_f32_mul(elq, r.sh[idx]), dtl[idx]), dqw[idx]);
        }
        const float vkz = mynn_f32_mul(mynn_f32_mul(0.4f, 0.5f), r.dz[base]);
        const float elv = mynn_f32_div(mynn_f32_mul(0.5f,
            mynn_f32_add(r.el[ncol + base], r.el[base])), vkz);
        if (q.initialize_qke != 0) {
            const float uf = mynn_f32_max2(ust, 0.02f);
            r.qke[base] = mynn_f32_mul(mynn_f32_mul(uf, uf),
                mynn_initialize_powf_candidate(mynn_f32_mul(24.0f, elv),
                    mynn_f32_div(2.0f, 3.0f)));
        }
        const float phm = mynn_f32_div(15.0f,
            mynn_initialize_powf_candidate(mynn_f32_div(24.0f,
                mynn_f32_mul(elv, elv)), mynn_f32_div(1.0f, 3.0f)));
        r.tsq[base] = mynn_f32_mul(phm, mynn_f32_mul(
            mynn_f32_div(0.0f, ust), mynn_f32_div(0.0f, ust)));
        r.qsq[base] = r.tsq[base];
        r.cov[base] = mynn_f32_mul(mynn_f32_mul(phm,
            mynn_f32_div(0.0f, ust)), mynn_f32_div(0.0f, ust));
        for (int k = 1; k < nz - 1; ++k) {
            const ulong idx = ulong(k) * ncol + base;
            const float b1l = mynn_f32_mul(mynn_f32_mul(24.0f, 0.25f),
                mynn_f32_add(r.el[idx + ncol], r.el[idx]));
            const float tmpq = mynn_f32_min2(mynn_f32_max2(mynn_f32_mul(
                b1l, mynn_f32_add(pdk[idx + ncol], pdk[idx])), qkemin), 125.0f);
            if (q.initialize_qke != 0)
                r.qke[idx] = mynn_initialize_powf_candidate(tmpq,
                                                  mynn_f32_div(2.0f, 3.0f));
            const float b2l = r.qke[idx] <= 0.0f ? 0.0f : mynn_f32_div(
                mynn_f32_mul(15.0f, mynn_f32_div(b1l, 24.0f)),
                mynn_f32_sqrt(r.qke[idx]));
            r.tsq[idx] = mynn_f32_mul(b2l,
                mynn_f32_add(pdt[idx + ncol], pdt[idx]));
            r.qsq[idx] = mynn_f32_mul(b2l,
                mynn_f32_add(pdq[idx + ncol], pdq[idx]));
            r.cov[idx] = mynn_f32_mul(b2l,
                mynn_f32_add(pdc[idx + ncol], pdc[idx]));
        }
    }
    if (q.initialize_qke != 0) {
        r.qke[base] = mynn_f32_mul(0.5f,
            mynn_f32_add(r.qke[base], r.qke[ncol + base]));
        r.qke[ulong(nz - 1) * ncol + base] =
            r.qke[ulong(nz - 2) * ncol + base];
    }
    r.tsq[ulong(nz - 1) * ncol + base] = r.tsq[ulong(nz - 2) * ncol + base];
    r.qsq[ulong(nz - 1) * ncol + base] = r.qsq[ulong(nz - 2) * ncol + base];
    r.cov[ulong(nz - 1) * ncol + base] = r.cov[ulong(nz - 2) * ncol + base];
}

kernel void mynn_initialize_default_columns(
    constant MynnInitializeDefaultColumnsResources &r [[buffer(0)]],
    constant MynnInitializeDefaultColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    mynn_initialize_column(r, q, column, q.mixlength == 2 ? 3 : 1);
}

kernel void mynn_initialize_gsd41_columns(
    constant MynnInitializeGsd41ColumnsResources &r [[buffer(0)]],
    constant MynnInitializeGsd41ColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    mynn_initialize_column(r, q, column, 2);
}

// A non-null qc_orig/qi_orig selects the GSD MYNN v4.1 mynn_tendencies: no
// density weighting, no mass-flux floors, no limit on a negative surface
// vapour flux, and mixing-ratio tendencies rebuilt from the original
// mixing ratios (r.qv is then the original vapour mixing ratio).
template <typename Resources, typename Params>
static inline void mynn_tendencies_column(
    constant Resources &r, constant Params &q, uint column,
    device const float *qc_orig, device const float *qi_orig)
{
    const bool gsd41 = qc_orig != nullptr;
    if (column >= uint(q.ncol) || q.nz < 2) return;
    const int nz = q.nz;
    const int ncol = q.ncol;
    // Fifteen internal arrays share one level-major allocation. The common
    // pitch is nz+1 so interface fields retain their native vertical extent;
    // mass-level fields use only their first nz planes.
    const ulong scratch_pitch = ulong(nz + 1) * ulong(ncol);
    device float *dtz = r.scratch + 0 * scratch_pitch;
    device float *rhoinv = r.scratch + 1 * scratch_pitch;
    device float *delp = r.scratch + 2 * scratch_pitch;
    device float *khdz = r.scratch + 3 * scratch_pitch;
    device float *kmdz = r.scratch + 4 * scratch_pitch;
    device float *a = r.scratch + 5 * scratch_pitch;
    device float *b = r.scratch + 6 * scratch_pitch;
    device float *c = r.scratch + 7 * scratch_pitch;
    device float *d = r.scratch + 8 * scratch_pitch;
    device float *cpw = r.scratch + 9 * scratch_pitch;
    device float *dpw = r.scratch + 10 * scratch_pitch;
    device float *sqv2 = r.scratch + 11 * scratch_pitch;
    device float *sqc2 = r.scratch + 12 * scratch_pitch;
    device float *sqi2 = r.scratch + 13 * scratch_pitch;
    device float *sqs2 = r.scratch + 14 * scratch_pitch;

    const int base = 0;
    const int zbase = 0;
    const int top = base + nz - 1;

    const float p608 = q.p608;
    const float xlvcp = q.xlvcp;
    const float xlscp = q.xlscp;

    const float delt = r.delt[column];
    const float psfc = r.psfc[column];
    const float ust = r.ust[column];
    const float wspd = r.wspd[column];
    const float uoce = r.uoce[column];
    const float voce = r.voce[column];
    const float flt = r.flt[column];
    const float flqv = r.flqv[column];
    const float flqc = r.flqc[column];

    // ---- module_bl_mynn.F:4136-4163 diffusion "constants" ----------------
    float rhosfc = gsd41 ? 1.0f : mynn_f32_div(
        psfc,
        mynn_f32_mul(q.rd, mynn_f32_add(r.tk[ulong(base) * ulong(ncol) + column], mynn_f32_mul(p608, r.qv[ulong(base) * ulong(ncol) + column]))));
    dtz[ulong(base) * ulong(ncol) + column] = mynn_f32_div(delt, r.dz[ulong(base) * ulong(ncol) + column]);
    float rhoz = r.rho[ulong(base) * ulong(ncol) + column];
    rhoinv[ulong(base) * ulong(ncol) + column] = gsd41 ? 1.0f : mynn_f32_div(1.0f, r.rho[ulong(base) * ulong(ncol) + column]);
    khdz[ulong(zbase) * ulong(ncol) + column] = gsd41 ? 0.0f : mynn_f32_mul(rhoz, r.dfh[ulong(base) * ulong(ncol) + column]);
    kmdz[ulong(zbase) * ulong(ncol) + column] = gsd41 ? 0.0f : mynn_f32_mul(rhoz, r.dfm[ulong(base) * ulong(ncol) + column]);
    delp[ulong(base) * ulong(ncol) + column] = mynn_f32_sub(
        psfc,
        mynn_f32_div(mynn_f32_add(mynn_f32_mul(r.p[ulong(base + 1) * ulong(ncol) + column], r.dz[ulong(base) * ulong(ncol) + column]),
                          mynn_f32_mul(r.p[ulong(base) * ulong(ncol) + column], r.dz[ulong(base + 1) * ulong(ncol) + column])),
                 mynn_f32_add(r.dz[ulong(base) * ulong(ncol) + column], r.dz[ulong(base + 1) * ulong(ncol) + column])));
    for (int k = 1; k < nz; ++k) {
        int idx = base + k;
        dtz[ulong(idx) * ulong(ncol) + column] = mynn_f32_div(delt, r.dz[ulong(idx) * ulong(ncol) + column]);
        rhoz = mynn_f32_div(
            mynn_f32_add(mynn_f32_mul(r.rho[ulong(idx) * ulong(ncol) + column], r.dz[ulong(idx - 1) * ulong(ncol) + column]),
                     mynn_f32_mul(r.rho[ulong(idx - 1) * ulong(ncol) + column], r.dz[ulong(idx) * ulong(ncol) + column])),
            mynn_f32_add(r.dz[ulong(idx - 1) * ulong(ncol) + column], r.dz[ulong(idx) * ulong(ncol) + column]));
        rhoz = mynn_f32_max2(rhoz, 1.0e-4f);
        rhoinv[ulong(idx) * ulong(ncol) + column] = gsd41 ? 1.0f : mynn_f32_div(1.0f, mynn_f32_max2(r.rho[ulong(idx) * ulong(ncol) + column], 1.0e-4f));
        khdz[ulong(zbase + k) * ulong(ncol) + column] = gsd41 ? r.dfh[ulong(idx) * ulong(ncol) + column] : mynn_f32_mul(rhoz, r.dfh[ulong(idx) * ulong(ncol) + column]);
        kmdz[ulong(zbase + k) * ulong(ncol) + column] = gsd41 ? r.dfm[ulong(idx) * ulong(ncol) + column] : mynn_f32_mul(rhoz, r.dfm[ulong(idx) * ulong(ncol) + column]);
    }
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k;
        delp[ulong(idx) * ulong(ncol) + column] = mynn_f32_sub(
            mynn_f32_div(mynn_f32_add(mynn_f32_mul(r.p[ulong(idx) * ulong(ncol) + column], r.dz[ulong(idx - 1) * ulong(ncol) + column]),
                              mynn_f32_mul(r.p[ulong(idx - 1) * ulong(ncol) + column], r.dz[ulong(idx) * ulong(ncol) + column])),
                     mynn_f32_add(r.dz[ulong(idx) * ulong(ncol) + column], r.dz[ulong(idx - 1) * ulong(ncol) + column])),
            mynn_f32_div(mynn_f32_add(mynn_f32_mul(r.p[ulong(idx + 1) * ulong(ncol) + column], r.dz[ulong(idx) * ulong(ncol) + column]),
                              mynn_f32_mul(r.p[ulong(idx) * ulong(ncol) + column], r.dz[ulong(idx + 1) * ulong(ncol) + column])),
                     mynn_f32_add(r.dz[ulong(idx) * ulong(ncol) + column], r.dz[ulong(idx + 1) * ulong(ncol) + column])));
    }
    delp[ulong(top) * ulong(ncol) + column] = delp[ulong(top - 1) * ulong(ncol) + column];
    // rhoz(kte+1)=rhoz(kte): rhoz still holds the top mass-level value.
    khdz[ulong(zbase + nz) * ulong(ncol) + column] = gsd41 ? r.dfh[ulong(top) * ulong(ncol) + column] : mynn_f32_mul(rhoz, r.dfh[ulong(top) * ulong(ncol) + column]);
    kmdz[ulong(zbase + nz) * ulong(ncol) + column] = gsd41 ? r.dfm[ulong(top) * ulong(ncol) + column] : mynn_f32_mul(rhoz, r.dfm[ulong(top) * ulong(ncol) + column]);

    // module_bl_mynn.F:4165-4171 mass-flux stability floors.  Inert while
    // s_aw is zero, but they are why khdz/kmdz cannot simply be reused from
    // mym_predict once DMP_mf is live.
    for (int k = 1; !gsd41 && k < nz - 1; ++k) {
        int zk = zbase + k;
        khdz[ulong(zk) * ulong(ncol) + column] = mynn_f32_max2(khdz[ulong(zk) * ulong(ncol) + column], mynn_f32_mul(0.5f, r.s_aw[ulong(zk) * ulong(ncol) + column]));
        khdz[ulong(zk) * ulong(ncol) + column] = mynn_f32_max2(
            khdz[ulong(zk) * ulong(ncol) + column], -mynn_f32_mul(0.5f, mynn_f32_sub(r.s_aw[ulong(zk) * ulong(ncol) + column], r.s_aw[ulong(zk + 1) * ulong(ncol) + column])));
        kmdz[ulong(zk) * ulong(ncol) + column] = mynn_f32_max2(kmdz[ulong(zk) * ulong(ncol) + column], mynn_f32_mul(0.5f, r.s_aw[ulong(zk) * ulong(ncol) + column]));
        kmdz[ulong(zk) * ulong(ncol) + column] = mynn_f32_max2(
            kmdz[ulong(zk) * ulong(ncol) + column], -mynn_f32_mul(0.5f, mynn_f32_sub(r.s_aw[ulong(zk) * ulong(ncol) + column], r.s_aw[ulong(zk + 1) * ulong(ncol) + column])));
    }

    // 0.5*dtz(k)*rhoinv(k) and dtz(k)*rhoinv(k) are the shared prefactors of
    // every mass-flux term; the surface pair is reused by all six systems.
    const float hdz0 = mynn_f32_mul(mynn_f32_mul(0.5f, dtz[ulong(base) * ulong(ncol) + column]), rhoinv[ulong(base) * ulong(ncol) + column]);
    const float dzinv0 = mynn_f32_mul(dtz[ulong(base) * ulong(ncol) + column], rhoinv[ulong(base) * ulong(ncol) + column]);

    // ---- module_bl_mynn.F:4176-4296 momentum -----------------------------
    float drag = mynn_f32_div(mynn_f32_mul(rhosfc, mynn_f32_mul(ust, ust)), wspd);
    a[ulong(base) * ulong(ncol) + column] = -mynn_f32_mul(mynn_f32_mul(dtz[ulong(base) * ulong(ncol) + column], kmdz[ulong(zbase) * ulong(ncol) + column]), rhoinv[ulong(base) * ulong(ncol) + column]);
    b[ulong(base) * ulong(ncol) + column] = mynn_f32_sub(
        mynn_f32_add(1.0f,
                 mynn_f32_mul(mynn_f32_mul(dtz[ulong(base) * ulong(ncol) + column],
                                   mynn_f32_add(kmdz[ulong(zbase + 1) * ulong(ncol) + column], drag)),
                          rhoinv[ulong(base) * ulong(ncol) + column])),
        mynn_f32_mul(mynn_f32_mul(hdz0, r.s_aw[ulong(zbase + 1) * ulong(ncol) + column]), q.onoff));
    b[ulong(base) * ulong(ncol) + column] = mynn_f32_sub(b[ulong(base) * ulong(ncol) + column],
                       mynn_f32_mul(mynn_f32_mul(hdz0, r.sd_aw[ulong(zbase + 1) * ulong(ncol) + column]), q.onoff));
    c[ulong(base) * ulong(ncol) + column] = mynn_f32_sub(
        -mynn_f32_mul(mynn_f32_mul(dtz[ulong(base) * ulong(ncol) + column], kmdz[ulong(zbase + 1) * ulong(ncol) + column]), rhoinv[ulong(base) * ulong(ncol) + column]),
        mynn_f32_mul(mynn_f32_mul(hdz0, r.s_aw[ulong(zbase + 1) * ulong(ncol) + column]), q.onoff));
    c[ulong(base) * ulong(ncol) + column] = mynn_f32_sub(c[ulong(base) * ulong(ncol) + column],
                       mynn_f32_mul(mynn_f32_mul(hdz0, r.sd_aw[ulong(zbase + 1) * ulong(ncol) + column]), q.onoff));
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k, zk = zbase + k;
        float hdz = mynn_f32_mul(mynn_f32_mul(0.5f, dtz[ulong(idx) * ulong(ncol) + column]), rhoinv[ulong(idx) * ulong(ncol) + column]);
        a[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(
            -mynn_f32_mul(mynn_f32_mul(dtz[ulong(idx) * ulong(ncol) + column], kmdz[ulong(zk) * ulong(ncol) + column]), rhoinv[ulong(idx) * ulong(ncol) + column]),
            mynn_f32_mul(mynn_f32_mul(hdz, r.s_aw[ulong(zk) * ulong(ncol) + column]), q.onoff));
        a[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(a[ulong(idx) * ulong(ncol) + column],
                          mynn_f32_mul(mynn_f32_mul(hdz, r.sd_aw[ulong(zk) * ulong(ncol) + column]), q.onoff));
        b[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(
            mynn_f32_add(1.0f,
                     mynn_f32_mul(mynn_f32_mul(dtz[ulong(idx) * ulong(ncol) + column],
                                       mynn_f32_add(kmdz[ulong(zk) * ulong(ncol) + column], kmdz[ulong(zk + 1) * ulong(ncol) + column])),
                              rhoinv[ulong(idx) * ulong(ncol) + column])),
            mynn_f32_mul(mynn_f32_mul(hdz, mynn_f32_sub(r.s_aw[ulong(zk) * ulong(ncol) + column], r.s_aw[ulong(zk + 1) * ulong(ncol) + column])), q.onoff));
        b[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(
            b[ulong(idx) * ulong(ncol) + column],
            mynn_f32_mul(mynn_f32_mul(hdz, mynn_f32_sub(r.sd_aw[ulong(zk) * ulong(ncol) + column], r.sd_aw[ulong(zk + 1) * ulong(ncol) + column])),
                     q.onoff));
        c[ulong(idx) * ulong(ncol) + column] = mynn_f32_sub(
            -mynn_f32_mul(mynn_f32_mul(dtz[ulong(idx) * ulong(ncol) + column], kmdz[ulong(zk + 1) * ulong(ncol) + column]), rhoinv[ulong(idx) * ulong(ncol) + column]),
            mynn_f32_mul(mynn_f32_mul(hdz, r.s_aw[ulong(zk + 1) * ulong(ncol) + column]), q.onoff));
        c[ulong(idx) * ulong(ncol) + column] = mynn_f32_sub(c[ulong(idx) * ulong(ncol) + column],
                          mynn_f32_mul(mynn_f32_mul(hdz, r.sd_aw[ulong(zk + 1) * ulong(ncol) + column]), q.onoff));
    }
    a[ulong(top) * ulong(ncol) + column] = 0.0f;
    b[ulong(top) * ulong(ncol) + column] = 1.0f;
    c[ulong(top) * ulong(ncol) + column] = 0.0f;

    d[ulong(base) * ulong(ncol) + column] = mynn_f32_sub(
        mynn_f32_add(r.u[ulong(base) * ulong(ncol) + column],
                 mynn_f32_div(mynn_f32_mul(mynn_f32_mul(dtz[ulong(base) * ulong(ncol) + column], uoce),
                                   mynn_f32_mul(ust, ust)),
                          wspd)),
        mynn_f32_mul(mynn_f32_mul(dzinv0, r.s_awu[ulong(zbase + 1) * ulong(ncol) + column]), q.onoff));
    d[ulong(base) * ulong(ncol) + column] = mynn_f32_add(d[ulong(base) * ulong(ncol) + column],
                       mynn_f32_mul(mynn_f32_mul(dzinv0, r.sd_awu[ulong(zbase + 1) * ulong(ncol) + column]), q.onoff));
    d[ulong(base) * ulong(ncol) + column] = mynn_f32_add(mynn_f32_add(d[ulong(base) * ulong(ncol) + column], mynn_f32_mul(r.sub_u[ulong(base) * ulong(ncol) + column], delt)),
                       mynn_f32_mul(r.det_u[ulong(base) * ulong(ncol) + column], delt));
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k, zk = zbase + k;
        float dzinv = mynn_f32_mul(dtz[ulong(idx) * ulong(ncol) + column], rhoinv[ulong(idx) * ulong(ncol) + column]);
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(
            r.u[ulong(idx) * ulong(ncol) + column],
            mynn_f32_mul(mynn_f32_mul(dzinv, mynn_f32_sub(r.s_awu[ulong(zk) * ulong(ncol) + column], r.s_awu[ulong(zk + 1) * ulong(ncol) + column])),
                     q.onoff));
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_sub(
            d[ulong(idx) * ulong(ncol) + column],
            mynn_f32_mul(mynn_f32_mul(dzinv, mynn_f32_sub(r.sd_awu[ulong(zk) * ulong(ncol) + column], r.sd_awu[ulong(zk + 1) * ulong(ncol) + column])),
                     q.onoff));
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(mynn_f32_add(d[ulong(idx) * ulong(ncol) + column], mynn_f32_mul(r.sub_u[ulong(idx) * ulong(ncol) + column], delt)),
                          mynn_f32_mul(r.det_u[ulong(idx) * ulong(ncol) + column], delt));
    }
    d[ulong(top) * ulong(ncol) + column] = r.u[ulong(top) * ulong(ncol) + column];
    mynn_tridiag2_column(a, b, c, d, cpw, dpw, r.du, uint(ncol), column, nz);
    for (int k = 0; k < nz; ++k)
        r.du[ulong(base + k) * ulong(ncol) + column] = mynn_f32_div(mynn_f32_sub(r.du[ulong(base + k) * ulong(ncol) + column], r.u[ulong(base + k) * ulong(ncol) + column]), delt);

    d[ulong(base) * ulong(ncol) + column] = mynn_f32_sub(
        mynn_f32_add(r.v[ulong(base) * ulong(ncol) + column],
                 mynn_f32_div(mynn_f32_mul(mynn_f32_mul(dtz[ulong(base) * ulong(ncol) + column], voce),
                                   mynn_f32_mul(ust, ust)),
                          wspd)),
        mynn_f32_mul(mynn_f32_mul(dzinv0, r.s_awv[ulong(zbase + 1) * ulong(ncol) + column]), q.onoff));
    d[ulong(base) * ulong(ncol) + column] = mynn_f32_add(d[ulong(base) * ulong(ncol) + column],
                       mynn_f32_mul(mynn_f32_mul(dzinv0, r.sd_awv[ulong(zbase + 1) * ulong(ncol) + column]), q.onoff));
    d[ulong(base) * ulong(ncol) + column] = mynn_f32_add(mynn_f32_add(d[ulong(base) * ulong(ncol) + column], mynn_f32_mul(r.sub_v[ulong(base) * ulong(ncol) + column], delt)),
                       mynn_f32_mul(r.det_v[ulong(base) * ulong(ncol) + column], delt));
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k, zk = zbase + k;
        float dzinv = mynn_f32_mul(dtz[ulong(idx) * ulong(ncol) + column], rhoinv[ulong(idx) * ulong(ncol) + column]);
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(
            r.v[ulong(idx) * ulong(ncol) + column],
            mynn_f32_mul(mynn_f32_mul(dzinv, mynn_f32_sub(r.s_awv[ulong(zk) * ulong(ncol) + column], r.s_awv[ulong(zk + 1) * ulong(ncol) + column])),
                     q.onoff));
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_sub(
            d[ulong(idx) * ulong(ncol) + column],
            mynn_f32_mul(mynn_f32_mul(dzinv, mynn_f32_sub(r.sd_awv[ulong(zk) * ulong(ncol) + column], r.sd_awv[ulong(zk + 1) * ulong(ncol) + column])),
                     q.onoff));
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(mynn_f32_add(d[ulong(idx) * ulong(ncol) + column], mynn_f32_mul(r.sub_v[ulong(idx) * ulong(ncol) + column], delt)),
                          mynn_f32_mul(r.det_v[ulong(idx) * ulong(ncol) + column], delt));
    }
    d[ulong(top) * ulong(ncol) + column] = r.v[ulong(top) * ulong(ncol) + column];
    mynn_tridiag2_column(a, b, c, d, cpw, dpw, r.dv, uint(ncol), column, nz);
    for (int k = 0; k < nz; ++k)
        r.dv[ulong(base + k) * ulong(ncol) + column] = mynn_f32_div(mynn_f32_sub(r.dv[ulong(base + k) * ulong(ncol) + column], r.v[ulong(base + k) * ulong(ncol) + column]), delt);

    // ---- module_bl_mynn.F:4318-4335 shared heat/moisture matrix ----------
    a[ulong(base) * ulong(ncol) + column] = -mynn_f32_mul(mynn_f32_mul(dtz[ulong(base) * ulong(ncol) + column], khdz[ulong(zbase) * ulong(ncol) + column]), rhoinv[ulong(base) * ulong(ncol) + column]);
    b[ulong(base) * ulong(ncol) + column] = mynn_f32_sub(
        mynn_f32_add(1.0f,
                 mynn_f32_mul(mynn_f32_mul(dtz[ulong(base) * ulong(ncol) + column],
                                   mynn_f32_add(khdz[ulong(zbase + 1) * ulong(ncol) + column], khdz[ulong(zbase) * ulong(ncol) + column])),
                          rhoinv[ulong(base) * ulong(ncol) + column])),
        mynn_f32_mul(hdz0, r.s_aw[ulong(zbase + 1) * ulong(ncol) + column]));
    b[ulong(base) * ulong(ncol) + column] = mynn_f32_sub(b[ulong(base) * ulong(ncol) + column], mynn_f32_mul(hdz0, r.sd_aw[ulong(zbase + 1) * ulong(ncol) + column]));
    c[ulong(base) * ulong(ncol) + column] = mynn_f32_sub(
        -mynn_f32_mul(mynn_f32_mul(dtz[ulong(base) * ulong(ncol) + column], khdz[ulong(zbase + 1) * ulong(ncol) + column]), rhoinv[ulong(base) * ulong(ncol) + column]),
        mynn_f32_mul(hdz0, r.s_aw[ulong(zbase + 1) * ulong(ncol) + column]));
    c[ulong(base) * ulong(ncol) + column] = mynn_f32_sub(c[ulong(base) * ulong(ncol) + column], mynn_f32_mul(hdz0, r.sd_aw[ulong(zbase + 1) * ulong(ncol) + column]));
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k, zk = zbase + k;
        float hdz = mynn_f32_mul(mynn_f32_mul(0.5f, dtz[ulong(idx) * ulong(ncol) + column]), rhoinv[ulong(idx) * ulong(ncol) + column]);
        a[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(
            -mynn_f32_mul(mynn_f32_mul(dtz[ulong(idx) * ulong(ncol) + column], khdz[ulong(zk) * ulong(ncol) + column]), rhoinv[ulong(idx) * ulong(ncol) + column]),
            mynn_f32_mul(hdz, r.s_aw[ulong(zk) * ulong(ncol) + column]));
        a[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(a[ulong(idx) * ulong(ncol) + column], mynn_f32_mul(hdz, r.sd_aw[ulong(zk) * ulong(ncol) + column]));
        b[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(
            mynn_f32_add(1.0f,
                     mynn_f32_mul(mynn_f32_mul(dtz[ulong(idx) * ulong(ncol) + column],
                                       mynn_f32_add(khdz[ulong(zk) * ulong(ncol) + column], khdz[ulong(zk + 1) * ulong(ncol) + column])),
                              rhoinv[ulong(idx) * ulong(ncol) + column])),
            mynn_f32_mul(hdz, mynn_f32_sub(r.s_aw[ulong(zk) * ulong(ncol) + column], r.s_aw[ulong(zk + 1) * ulong(ncol) + column])));
        b[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(b[ulong(idx) * ulong(ncol) + column],
                          mynn_f32_mul(hdz, mynn_f32_sub(r.sd_aw[ulong(zk) * ulong(ncol) + column], r.sd_aw[ulong(zk + 1) * ulong(ncol) + column])));
        c[ulong(idx) * ulong(ncol) + column] = mynn_f32_sub(
            -mynn_f32_mul(mynn_f32_mul(dtz[ulong(idx) * ulong(ncol) + column], khdz[ulong(zk + 1) * ulong(ncol) + column]), rhoinv[ulong(idx) * ulong(ncol) + column]),
            mynn_f32_mul(hdz, r.s_aw[ulong(zk + 1) * ulong(ncol) + column]));
        c[ulong(idx) * ulong(ncol) + column] = mynn_f32_sub(c[ulong(idx) * ulong(ncol) + column], mynn_f32_mul(hdz, r.sd_aw[ulong(zk + 1) * ulong(ncol) + column]));
    }
    a[ulong(top) * ulong(ncol) + column] = 0.0f;
    b[ulong(top) * ulong(ncol) + column] = 1.0f;
    c[ulong(top) * ulong(ncol) + column] = 0.0f;

    // ---- module_bl_mynn.F:4298-4372 liquid-water potential temperature ---
    d[ulong(base) * ulong(ncol) + column] = mynn_f32_add(
        mynn_f32_add(r.thl_in[ulong(base) * ulong(ncol) + column],
                 mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(dtz[ulong(base) * ulong(ncol) + column], rhosfc), flt),
                          rhoinv[ulong(base) * ulong(ncol) + column])),
        mynn_f32_mul(r.tcd[ulong(base) * ulong(ncol) + column], delt));
    d[ulong(base) * ulong(ncol) + column] = mynn_f32_sub(
        mynn_f32_sub(d[ulong(base) * ulong(ncol) + column], mynn_f32_mul(dzinv0, r.s_awthl[ulong(zbase + 1) * ulong(ncol) + column])),
        mynn_f32_mul(dzinv0, r.sd_awthl[ulong(zbase + 1) * ulong(ncol) + column]));
    d[ulong(base) * ulong(ncol) + column] = mynn_f32_add(
        mynn_f32_add(mynn_f32_add(d[ulong(base) * ulong(ncol) + column], mynn_f32_mul(r.diss_heat[ulong(base) * ulong(ncol) + column], delt)),
                 mynn_f32_mul(r.sub_thl[ulong(base) * ulong(ncol) + column], delt)),
        mynn_f32_mul(r.det_thl[ulong(base) * ulong(ncol) + column], delt));
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k, zk = zbase + k;
        float dzinv = mynn_f32_mul(dtz[ulong(idx) * ulong(ncol) + column], rhoinv[ulong(idx) * ulong(ncol) + column]);
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(r.thl_in[ulong(idx) * ulong(ncol) + column], mynn_f32_mul(r.tcd[ulong(idx) * ulong(ncol) + column], delt));
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(
            d[ulong(idx) * ulong(ncol) + column],
            mynn_f32_mul(dzinv, mynn_f32_sub(r.s_awthl[ulong(zk) * ulong(ncol) + column], r.s_awthl[ulong(zk + 1) * ulong(ncol) + column])));
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(
            d[ulong(idx) * ulong(ncol) + column],
            mynn_f32_mul(dzinv, mynn_f32_sub(r.sd_awthl[ulong(zk) * ulong(ncol) + column], r.sd_awthl[ulong(zk + 1) * ulong(ncol) + column])));
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(
            mynn_f32_add(mynn_f32_add(d[ulong(idx) * ulong(ncol) + column], mynn_f32_mul(r.diss_heat[ulong(idx) * ulong(ncol) + column], delt)),
                     mynn_f32_mul(r.sub_thl[ulong(idx) * ulong(ncol) + column], delt)),
            mynn_f32_mul(r.det_thl[ulong(idx) * ulong(ncol) + column], delt));
    }
    d[ulong(top) * ulong(ncol) + column] = r.thl_in[ulong(top) * ulong(ncol) + column];
    mynn_tridiag2_column(a, b, c, d, cpw, dpw, r.thl, uint(ncol), column, nz);

    // ---- module_bl_mynn.F:4432-4489 cloud water --------------------------
    d[ulong(base) * ulong(ncol) + column] = mynn_f32_add(
        mynn_f32_add(r.sqc[ulong(base) * ulong(ncol) + column],
                 mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(dtz[ulong(base) * ulong(ncol) + column], rhosfc), flqc),
                          rhoinv[ulong(base) * ulong(ncol) + column])),
        mynn_f32_mul(r.qcd[ulong(base) * ulong(ncol) + column], delt));
    d[ulong(base) * ulong(ncol) + column] = mynn_f32_sub(
        mynn_f32_sub(d[ulong(base) * ulong(ncol) + column], mynn_f32_mul(dzinv0, r.s_awqc[ulong(zbase + 1) * ulong(ncol) + column])),
        mynn_f32_mul(dzinv0, r.sd_awqc[ulong(zbase + 1) * ulong(ncol) + column]));
    d[ulong(base) * ulong(ncol) + column] = mynn_f32_add(d[ulong(base) * ulong(ncol) + column], mynn_f32_mul(r.det_sqc[ulong(base) * ulong(ncol) + column], delt));
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k, zk = zbase + k;
        float dzinv = mynn_f32_mul(dtz[ulong(idx) * ulong(ncol) + column], rhoinv[ulong(idx) * ulong(ncol) + column]);
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(r.sqc[ulong(idx) * ulong(ncol) + column], mynn_f32_mul(r.qcd[ulong(idx) * ulong(ncol) + column], delt));
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(
            d[ulong(idx) * ulong(ncol) + column], mynn_f32_mul(dzinv, mynn_f32_sub(r.s_awqc[ulong(zk) * ulong(ncol) + column], r.s_awqc[ulong(zk + 1) * ulong(ncol) + column])));
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(
            d[ulong(idx) * ulong(ncol) + column], mynn_f32_mul(dzinv, mynn_f32_sub(r.sd_awqc[ulong(zk) * ulong(ncol) + column], r.sd_awqc[ulong(zk + 1) * ulong(ncol) + column])));
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(d[ulong(idx) * ulong(ncol) + column], mynn_f32_mul(r.det_sqc[ulong(idx) * ulong(ncol) + column], delt));
    }
    d[ulong(top) * ulong(ncol) + column] = r.sqc[ulong(top) * ulong(ncol) + column];
    mynn_tridiag2_column(a, b, c, d, cpw, dpw, sqc2, uint(ncol), column, nz);

    // ---- module_bl_mynn.F:4491-4553 water vapour -------------------------
    // module_bl_mynn.F:4514-4518 limits an unreasonably large *negative*
    // surface moisture flux.  For any positive sqv(kts) the MIN collapses to
    // 0.0, so a downward flux is not limited but deleted.
    float qvflux = flqv;
    if (!gsd41 && qvflux < 0.0f) {
        qvflux = mynn_f32_max2(
            qvflux,
            mynn_f32_div(
                mynn_f32_min2(mynn_f32_sub(mynn_f32_mul(0.9f, r.sqv[ulong(base) * ulong(ncol) + column]), 1.0e-8f), 0.0f),
                dtz[ulong(base) * ulong(ncol) + column]));
    }
    d[ulong(base) * ulong(ncol) + column] = mynn_f32_add(
        mynn_f32_add(r.sqv[ulong(base) * ulong(ncol) + column],
                 mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(dtz[ulong(base) * ulong(ncol) + column], rhosfc), qvflux),
                          rhoinv[ulong(base) * ulong(ncol) + column])),
        mynn_f32_mul(r.qcd[ulong(base) * ulong(ncol) + column], delt));
    d[ulong(base) * ulong(ncol) + column] = mynn_f32_sub(
        mynn_f32_sub(d[ulong(base) * ulong(ncol) + column], mynn_f32_mul(dzinv0, r.s_awqv[ulong(zbase + 1) * ulong(ncol) + column])),
        mynn_f32_mul(dzinv0, r.sd_awqv[ulong(zbase + 1) * ulong(ncol) + column]));
    d[ulong(base) * ulong(ncol) + column] = mynn_f32_add(mynn_f32_add(d[ulong(base) * ulong(ncol) + column], mynn_f32_mul(r.sub_sqv[ulong(base) * ulong(ncol) + column], delt)),
                       mynn_f32_mul(r.det_sqv[ulong(base) * ulong(ncol) + column], delt));
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k, zk = zbase + k;
        float dzinv = mynn_f32_mul(dtz[ulong(idx) * ulong(ncol) + column], rhoinv[ulong(idx) * ulong(ncol) + column]);
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(r.sqv[ulong(idx) * ulong(ncol) + column], mynn_f32_mul(r.qcd[ulong(idx) * ulong(ncol) + column], delt));
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(
            d[ulong(idx) * ulong(ncol) + column], mynn_f32_mul(dzinv, mynn_f32_sub(r.s_awqv[ulong(zk) * ulong(ncol) + column], r.s_awqv[ulong(zk + 1) * ulong(ncol) + column])));
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(
            d[ulong(idx) * ulong(ncol) + column], mynn_f32_mul(dzinv, mynn_f32_sub(r.sd_awqv[ulong(zk) * ulong(ncol) + column], r.sd_awqv[ulong(zk + 1) * ulong(ncol) + column])));
        d[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(mynn_f32_add(d[ulong(idx) * ulong(ncol) + column], mynn_f32_mul(r.sub_sqv[ulong(idx) * ulong(ncol) + column], delt)),
                          mynn_f32_mul(r.det_sqv[ulong(idx) * ulong(ncol) + column], delt));
    }
    d[ulong(top) * ulong(ncol) + column] = r.sqv[ulong(top) * ulong(ncol) + column];
    mynn_tridiag2_column(a, b, c, d, cpw, dpw, sqv2, uint(ncol), column, nz);

    // ---- module_bl_mynn.F:4566-4611 cloud ice: pure diffusion ------------
    a[ulong(base) * ulong(ncol) + column] = -mynn_f32_mul(mynn_f32_mul(dtz[ulong(base) * ulong(ncol) + column], khdz[ulong(zbase) * ulong(ncol) + column]), rhoinv[ulong(base) * ulong(ncol) + column]);
    b[ulong(base) * ulong(ncol) + column] = mynn_f32_add(
        1.0f,
        mynn_f32_mul(mynn_f32_mul(dtz[ulong(base) * ulong(ncol) + column], mynn_f32_add(khdz[ulong(zbase + 1) * ulong(ncol) + column], khdz[ulong(zbase) * ulong(ncol) + column])),
                 rhoinv[ulong(base) * ulong(ncol) + column]));
    c[ulong(base) * ulong(ncol) + column] = -mynn_f32_mul(mynn_f32_mul(dtz[ulong(base) * ulong(ncol) + column], khdz[ulong(zbase + 1) * ulong(ncol) + column]), rhoinv[ulong(base) * ulong(ncol) + column]);
    d[ulong(base) * ulong(ncol) + column] = r.sqi[ulong(base) * ulong(ncol) + column];
    for (int k = 1; k < nz - 1; ++k) {
        int idx = base + k, zk = zbase + k;
        a[ulong(idx) * ulong(ncol) + column] = -mynn_f32_mul(mynn_f32_mul(dtz[ulong(idx) * ulong(ncol) + column], khdz[ulong(zk) * ulong(ncol) + column]), rhoinv[ulong(idx) * ulong(ncol) + column]);
        b[ulong(idx) * ulong(ncol) + column] = mynn_f32_add(
            1.0f,
            mynn_f32_mul(mynn_f32_mul(dtz[ulong(idx) * ulong(ncol) + column], mynn_f32_add(khdz[ulong(zk) * ulong(ncol) + column], khdz[ulong(zk + 1) * ulong(ncol) + column])),
                     rhoinv[ulong(idx) * ulong(ncol) + column]));
        c[ulong(idx) * ulong(ncol) + column] = -mynn_f32_mul(mynn_f32_mul(dtz[ulong(idx) * ulong(ncol) + column], khdz[ulong(zk + 1) * ulong(ncol) + column]), rhoinv[ulong(idx) * ulong(ncol) + column]);
        d[ulong(idx) * ulong(ncol) + column] = r.sqi[ulong(idx) * ulong(ncol) + column];
    }
    a[ulong(top) * ulong(ncol) + column] = 0.0f;
    b[ulong(top) * ulong(ncol) + column] = 1.0f;
    c[ulong(top) * ulong(ncol) + column] = 0.0f;
    d[ulong(top) * ulong(ncol) + column] = r.sqi[ulong(top) * ulong(ncol) + column];
    mynn_tridiag2_column(a, b, c, d, cpw, dpw, sqi2, uint(ncol), column, nz);
    // Snow mixing is hard-disabled at module_bl_mynn.F:4618.
    for (int k = 0; k < nz; ++k) sqs2[ulong(base + k) * ulong(ncol) + column] = r.sqs[ulong(base + k) * ulong(ncol) + column];

    // ---- module_bl_mynn.F:4931-4993 species tendencies -------------------
    for (int k = 0; k < nz; ++k) {
        int idx = base + k;
        r.dqv[ulong(idx) * ulong(ncol) + column] = mynn_f32_div(mynn_f32_sub(sqv2[ulong(idx) * ulong(ncol) + column], r.sqv[ulong(idx) * ulong(ncol) + column]), delt);
        r.dqc[ulong(idx) * ulong(ncol) + column] = mynn_f32_div(mynn_f32_sub(sqc2[ulong(idx) * ulong(ncol) + column], r.sqc[ulong(idx) * ulong(ncol) + column]), delt);
        r.dqi[ulong(idx) * ulong(ncol) + column] = mynn_f32_div(mynn_f32_sub(sqi2[ulong(idx) * ulong(ncol) + column], r.sqi[ulong(idx) * ulong(ncol) + column]), delt);
        r.dqs[ulong(idx) * ulong(ncol) + column] = 0.0f;
        // module_bl_mynn.F:4173 zeroes dth so moisture_check can accumulate
        // into it; the theta block below then overwrites it, but the repair's
        // mutation of thl, sqc2, and sqi2 still reaches the answer.  That
        // asymmetry is WRF's, not a transcription slip.
        r.dth[ulong(idx) * ulong(ncol) + column] = 0.0f;
    }
    mynn_moisture_check_column(delt, xlvcp, xlscp, delp, r.exner, sqv2, sqc2, sqi2, sqs2, r.thl, r.dqv, r.dqc, r.dqi, r.dqs, r.dth, uint(ncol), column, nz);

    if (gsd41) {
        // Fork :3404, :3417 and :3449: the updated specific humidity over
        // (1 - updated vapour), less the original mixing ratio.
        for (int k = 0; k < nz; ++k) {
            const ulong idx = ulong(k) * ulong(ncol) + column;
            const float denominator = mynn_f32_sub(1.0f, sqv2[idx]);
            if (denominator == 0.0f) {
                r.dqv[idx] = 0.0f; r.dqc[idx] = 0.0f; r.dqi[idx] = 0.0f;
            } else {
                r.dqv[idx] = mynn_f32_div(mynn_f32_sub(
                    mynn_f32_div(sqv2[idx], denominator), r.qv[idx]), delt);
                r.dqc[idx] = mynn_f32_div(mynn_f32_sub(
                    mynn_f32_div(sqc2[idx], denominator), qc_orig[idx]), delt);
                r.dqi[idx] = mynn_f32_div(mynn_f32_sub(
                    mynn_f32_div(sqi2[idx], denominator), qi_orig[idx]), delt);
            }
        }
    }

    // ---- module_bl_mynn.F:5024-5031 ozone --------------------------------
    for (int k = 0; k < nz; ++k) {
        int idx = base + k;
        r.dozone[ulong(idx) * ulong(ncol) + column] = 0.0f;
        if (mynn_f32_add(mynn_f32_mul(r.dozone[ulong(idx) * ulong(ncol) + column], delt), r.ozone[ulong(idx) * ulong(ncol) + column]) < 0.0f)
            r.dozone[ulong(idx) * ulong(ncol) + column] = mynn_f32_div(-mynn_f32_mul(r.ozone[ulong(idx) * ulong(ncol) + column], 0.99f), delt);
    }

    // ---- module_bl_mynn.F:5033-5046 theta --------------------------------
    for (int k = 0; k < nz; ++k) {
        int idx = base + k;
        r.dth[ulong(idx) * ulong(ncol) + column] = mynn_f32_div(
            mynn_f32_sub(
                mynn_f32_add(
                    mynn_f32_add(r.thl[ulong(idx) * ulong(ncol) + column],
                             mynn_f32_mul(mynn_f32_div(xlvcp, r.exner[ulong(idx) * ulong(ncol) + column]),
                                      sqc2[ulong(idx) * ulong(ncol) + column])),
                    mynn_f32_mul(mynn_f32_div(xlscp, r.exner[ulong(idx) * ulong(ncol) + column]), sqi2[ulong(idx) * ulong(ncol) + column])),
                r.th[ulong(idx) * ulong(ncol) + column]),
            delt);
    }

}

kernel void mynn_tendencies_columns(
    constant MynnTendenciesColumnsResources &r [[buffer(0)]],
    constant MynnTendenciesColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    mynn_tendencies_column(r, q, column, nullptr, nullptr);
}

kernel void mynn_tendencies_gsd41_columns(
    constant MynnTendenciesGsd41ColumnsResources &r [[buffer(0)]],
    constant MynnTendenciesGsd41ColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    mynn_tendencies_column(r, q, column, r.qc_original, r.qi_original);
}

// module_bl_mynn.F:5679-6823, default EDMF mass-flux branch. The column
// scratch is caller-owned and sized for the source's eight updrafts. Shared
// software FP32 operations preserve the CUDA source's explicit operation
// boundaries; libm candidates remain subject to the family's experimental
// opt-in until the wider MYNN lane is admitted.
static inline float mynn_dmp_up(float a0, float a1, float dz0, float dz1)
{
    return mynn_f32_div(
        mynn_f32_add(mynn_f32_mul(a0, dz1), mynn_f32_mul(a1, dz0)),
        mynn_f32_add(dz1, dz0));
}

static inline void mynn_dmp_condensation(float qt, float thl, float p,
                                          float zagl, thread float &qc_io,
                                          thread float &thv_o, bool gsd41)
{
    const float xlvcp = mynn_f32_div(2.5e6f, 1004.5f);
    const float rcp = mynn_f32_div(287.0f, 1004.5f);
    const float rvovrd = mynn_f32_div(461.6f, 287.0f);
    float qc = qc_io;
    const float exn = mynn_gfk_powf(mynn_f32_div(p, 100000.0f), rcp);
    for (int it = 0; it < 50; ++it) {
        const float t = mynn_f32_add(mynn_f32_mul(exn, thl),
                                     mynn_f32_mul(xlvcp, qc));
        const float qs = gsd41 ? mynn_gsd41_qsat_blend(t, p)
                               : mynn_qsat_blend(t, p);
        const float old = qc;
        qc = mynn_f32_add(mynn_f32_mul(0.5f, qc),
            mynn_f32_mul(0.5f, mynn_f32_max2(mynn_f32_sub(qt, qs), 0.0f)));
        if (fabs(mynn_f32_sub(qc, old)) < (gsd41 ? 2.0e-5f : 1.0e-6f)) break;
    }
    const float t = mynn_f32_add(mynn_f32_mul(exn, thl),
                                 mynn_f32_mul(xlvcp, qc));
    const float qs = gsd41 ? mynn_gsd41_qsat_blend(t, p)
                           : mynn_qsat_blend(t, p);
    qc = mynn_f32_max2(mynn_f32_sub(qt, qs), 0.0f);
    if (zagl < 100.0f) qc = 0.0f;
    qc_io = qc;
    thv_o = mynn_f32_mul(
        mynn_f32_add(thl, mynn_f32_mul(xlvcp, qc)),
        mynn_f32_sub(
            mynn_f32_add(1.0f, mynn_f32_mul(qt,
                mynn_f32_sub(rvovrd, 1.0f))),
            mynn_f32_mul(rvovrd, qc)));
}

#define DMP_L(name, k) r.name[ulong(k) * ulong(q.ncol) + column]
#define DMP_I(name, k) r.name[ulong(k) * ulong(q.ncol) + column]
#define DMP_PLUME(v, i, k) r.plume_scratch[plume_base + ulong(v) * ulong(nup) * ulong(q.nz + 1) + ulong(i) * ulong(q.nz + 1) + ulong(k)]
#define DMP_WORK(v, k) r.work_scratch[work_base + ulong(v) * ulong(q.nz) + ulong(k)]
#define DMP_ENT(i, k) r.work_scratch[work_base + ulong(3 + (i)) * ulong(q.nz) + ulong(k)]

// bl_mynn_mixscalars=1 inputs/outputs of the DMP body: the four number
// scalars' layer values, their s_awqn interface fluxes and the up_qn plume
// scratch (twin of mynn_dmp_qn_flux_columns in mynn_scalar_mix.cu).
struct MynnDmpQn {
    device const float *qn[4];
    device float *s_awqn[4];
    device float *up_qn;
    int on;
};

template <typename Resources, typename Params>
static inline void mynn_dmp_mf_body(constant Resources &r,
                                    constant Params &q, uint column,
                                    bool gsd41,
                                    device const float *sgm_gsd41,
                                    MynnDmpQn mq)
{
    const int nz = q.nz;
    const uint ncol = uint(q.ncol);
    if (column >= ncol || nz < 5) return;
    const int nup = gsd41 ? 10 : 8;
    const ulong plume_stride = ulong(8) * ulong(nup) * ulong(nz + 1);
    const ulong work_stride = ulong(3 + nup) * ulong(nz);
    const ulong plume_base = ulong(column) * plume_stride;
    const ulong work_base = ulong(column) * work_stride;
    const float p608 = mynn_f32_sub(mynn_f32_div(461.6f, 287.0f), 1.0f);
    const float grav = 9.81f;
    const float gtr = mynn_f32_div(9.81f, 300.0f);
    const float tv0 = mynn_f32_mul(mynn_f32_sub(
        mynn_f32_div(461.6f, 287.0f), 1.0f), 300.0f);
    const float xlvcp = mynn_f32_div(2.5e6f, 1004.5f);
    const float cpv = mynn_f32_mul(4.0f, 461.6f);
    const float onethird = mynn_f32_div(1.0f, 3.0f);
    const float atot = 0.10f, lmax = 1000.0f, lmin = 300.0f;
    const float dcut = 1.2f, fluxportion = 0.75f, cf_thresh = 0.5f;
    const float dpow = -1.9f;
    const float flt = r.flt[column];
    float fltv = r.fltv[column];
    const float flq = r.flq[column], pblh = r.pblh[column];
    const float dx = r.dx[column], landsea = r.landsea[column];
    const float ts = r.ts[column], psig_shcu = r.psig_shcu[column];

    for (int k = 0; k < nz; ++k) {
        DMP_L(edmf_a, k) = 0.0f; DMP_L(edmf_w, k) = 0.0f;
        DMP_L(edmf_qt, k) = 0.0f; DMP_L(edmf_thl, k) = 0.0f;
        DMP_L(edmf_ent, k) = 0.0f; DMP_L(edmf_qc, k) = 0.0f;
        DMP_L(qc_bl, k) = DMP_L(qc_bl_in, k);
        DMP_L(cldfra_bl, k) = DMP_L(cldfra_bl_in, k);
        DMP_L(vt, k) = DMP_L(vt_in, k); DMP_L(vq, k) = DMP_L(vq_in, k);
        DMP_WORK(0, k) = 0.0f; DMP_WORK(1, k) = 0.0f;
        DMP_WORK(2, k) = 0.0f;
        for (int i = 0; i < nup; ++i) DMP_ENT(i, k) = 0.001f;
    }
    for (int k = 0; k <= nz; ++k) {
        DMP_I(s_aw, k) = 0.0f; DMP_I(s_awthl, k) = 0.0f;
        DMP_I(s_awqt, k) = 0.0f; DMP_I(s_awqv, k) = 0.0f;
        DMP_I(s_awqc, k) = 0.0f; DMP_I(s_awu, k) = 0.0f;
        DMP_I(s_awv, k) = 0.0f;
        if (mq.on) for (int n = 0; n < 4; ++n) mq.s_awqn[n][ulong(k) * ulong(q.ncol) + column] = 0.0f;
        for (int i = 0; i < nup; ++i)
            for (int v = 0; v < 8; ++v) DMP_PLUME(v, i, k) = 0.0f;
    }

    float nup2 = float(nup), maxw = 0.0f, cloud_base = 9000.0f;
    int k50 = 1;
    for (int k = 0; k < nz - 1; ++k) {
        if (DMP_I(zw, k) > mynn_f32_add(pblh, 500.0f)) break;
        float wpbl = DMP_L(w, k);
        if (wpbl < 0.0f) wpbl = mynn_f32_mul(2.0f, wpbl);
        maxw = mynn_f32_max2(maxw, fabs(wpbl));
        if (DMP_I(zw, k) <= 50.0f) k50 = k + 1;
        const float qc_sgs = mynn_f32_max2(DMP_L(qc, k), DMP_L(qc_bl, k));
        if (((gsd41 && DMP_L(qc, k) > 1.0e-5f)
                || (!gsd41 && qc_sgs > 1.0e-5f
                    && DMP_L(cldfra_bl, k) >= 0.5f))
            && cloud_base == 9000.0f)
            cloud_base = mynn_f32_mul(0.5f,
                mynn_f32_add(DMP_I(zw, k), DMP_I(zw, k + 1)));
    }
    float psig_w;
    float fltv2;
    if (gsd41) {
        maxw = mynn_f32_max2(0.0f, mynn_f32_sub(maxw, 0.5f));
        psig_w = mynn_f32_max2(0.0f,
                               mynn_f32_sub(1.0f, mynn_f32_div(maxw, 0.5f)));
        fltv = mynn_f32_add(flt, mynn_f32_mul(0.6112f, flq));
    } else {
        maxw = mynn_f32_max2(0.0f, mynn_f32_sub(maxw, 1.0f));
        psig_w = mynn_f32_max2(0.0, mynn_f32_sub(1.0f, maxw));
    }
    psig_w = mynn_f32_min2(psig_w, psig_shcu);
    fltv2 = fltv;
    if (psig_w == 0.0f && fltv > 0.0f)
        fltv2 = mynn_f32_mul(-1.0f, fltv);

    int superadiabatic = 0;
    const float hux = mynn_f32_sub(landsea, 1.5f) >= 0.0f
        ? (gsd41 ? -0.002f : -0.001f) : -0.005f;
    const float tvs = gsd41 ? ts : mynn_f32_mul(ts,
        mynn_f32_add(1.0f, mynn_f32_mul(p608, DMP_L(qv, 0))));
    const int nsuper = max(k50 - 1, 1);
    for (int k = 0; k < nsuper; ++k) {
        const float gradient = k == 0
            ? mynn_f32_div(mynn_f32_sub(gsd41 ? DMP_L(th, 0) : DMP_L(thv, 0), tvs),
                           mynn_f32_mul(0.5f, DMP_L(dz, 0)))
            : mynn_f32_div(mynn_f32_sub(gsd41 ? DMP_L(th, k) : DMP_L(thv, k),
                                       gsd41 ? DMP_L(th, k - 1) : DMP_L(thv, k - 1)),
                mynn_f32_mul(0.5f, mynn_f32_add(DMP_L(dz, k),
                                                DMP_L(dz, k - 1))));
        if (gradient < hux) superadiabatic = 1;
        else { superadiabatic = 0; break; }
    }
    float maxwidth;
    float minwidth;
    float wspd_pbl = 0.0f;
    if (gsd41) {
        const int class_count = int(mynn_f32_div(
            mynn_f32_mul(dx, dcut), 100.0f));
        nup2 = float(max(1, min(nup, class_count)));
        maxwidth = mynn_f32_min2(mynn_f32_mul(1.2f, pblh), cloud_base);
        if (mynn_f32_sub(landsea, 1.5f) < 0.0f) {
            const float width_shape = mynn_f32_add(0.5f,
                mynn_f32_mul(0.6f, mynn_tanhf(
                    mynn_f32_div(mynn_f32_sub(flt, 0.050f), 0.03f))));
            const float width_flx = mynn_f32_max2(0.0f,
                mynn_f32_min2(1000.0f, mynn_f32_mul(1000.0f, width_shape)));
            maxwidth = mynn_f32_min2(maxwidth, width_flx);
        }
        const float width100 = mynn_f32_sub(maxwidth,
                                             fmod(maxwidth, 100.0f));
        nup2 = mynn_f32_min2(float(max(int(mynn_f32_div(width100, 100.0f)), 0)),
                             nup2);
        minwidth = 100.0f;
    } else {
        maxwidth = mynn_f32_min2(mynn_f32_mul(dx, dcut), lmax);
        maxwidth = mynn_f32_min2(maxwidth, mynn_f32_mul(1.1f, pblh));
        maxwidth = mynn_f32_min2(maxwidth,
            mynn_f32_mul(mynn_f32_sub(landsea, 1.5f) < 0.0f ? 0.5f : 0.9f,
                         cloud_base));
        wspd_pbl = mynn_f32_sqrt(mynn_f32_max2(
            mynn_f32_add(mynn_f32_mul(DMP_L(u, 0), DMP_L(u, 0)),
                         mynn_f32_mul(DMP_L(v, 0), DMP_L(v, 0))), 0.01f));
        const float width_arg = mynn_f32_div(
            mynn_f32_sub(fltv, mynn_f32_sub(landsea, 1.5f) < 0.0f
                         ? 0.040f : 0.007f),
            mynn_f32_sub(landsea, 1.5f) < 0.0f ? 0.04f : 0.02f);
        const float width_flx = mynn_f32_max2(0.0f, mynn_f32_min2(1000.0f,
            mynn_f32_mul(1000.0f, mynn_f32_add(0.5f,
                mynn_f32_mul(0.6f, mynn_tanhf(width_arg))))));
        maxwidth = mynn_f32_min2(maxwidth, width_flx);
        minwidth = lmin;
        if (maxwidth <= minwidth) { nup2 = 0.0f; maxwidth = 0.0f; }
    }
    int ktop = 0;
    float ztop = 0.0f, maxmf = 0.0f;

    if (fltv2 > 0.002f
        && (gsd41 ? nup2 >= 1.0f : maxwidth > minwidth)
        && superadiabatic) {
        float cn = 0.0f;
        const float dl = gsd41 ? 100.0f :
            mynn_f32_div(mynn_f32_sub(maxwidth, minwidth), 7.0f);
        for (int i = 0; i < nup; ++i) {
            if (gsd41 && i >= int(nup2)) break;
            const float len = mynn_f32_add(minwidth, mynn_f32_mul(dl, float(i)));
            cn = mynn_f32_add(cn, mynn_f32_mul(mynn_f32_div(
                mynn_f32_mul(mynn_gfk_powf(len, dpow),
                             mynn_f32_mul(len, len)), mynn_f32_mul(dx, dx)), dl));
        }
        const float c_norm = mynn_f32_div(atot, cn);
        const float ac_wsp = (gsd41 || wspd_pbl <= 10.0f) ? 1.0f : mynn_f32_sub(
            1.0f, mynn_f32_min2(mynn_f32_div(
                mynn_f32_sub(wspd_pbl, 10.0f), 15.0f), 1.0f));
        float acfac = mynn_f32_add(0.5f,
            mynn_f32_mul(0.5f, mynn_tanhf(mynn_f32_div(
                mynn_f32_sub(fltv2, 0.02f), gsd41 ? 0.09f : 0.05f))));
        if (!gsd41) acfac = mynn_f32_mul(acfac, ac_wsp);
        for (int i = 0; i < nup; ++i) {
            if (gsd41 && i >= int(nup2)) break;
            const float len = mynn_f32_add(minwidth, mynn_f32_mul(dl, float(i)));
            const float number = mynn_f32_mul(c_norm, mynn_gfk_powf(len, dpow));
            const float area = mynn_f32_mul(mynn_f32_div(
                mynn_f32_mul(mynn_f32_mul(number, len), len),
                mynn_f32_mul(dx, dx)), dl);
            DMP_PLUME(5, i, 0) = mynn_f32_mul(area, acfac);
        }
        const float wstar = mynn_f32_max2(1.0e-2f,
            mynn_gfk_powf(mynn_f32_mul(mynn_f32_mul(
                gsd41 ? mynn_f32_div(grav, DMP_L(thv, 0)) : gtr,
                fltv2), pblh), onethird));
        const float qstar = mynn_f32_div(mynn_f32_max2(flq, 1.0e-5f), wstar);
        const float thstar = mynn_f32_div(flt, wstar);
        const float exc_fac = gsd41 ? 0.58f : mynn_f32_mul(
            mynn_f32_sub(landsea, 1.5f) >= 0.0f ? 2.32f : 0.58f, ac_wsp);
        const float zratio = mynn_gfk_powf(mynn_f32_div(50.0f, pblh), onethird);
        const float sigma_w = mynn_f32_mul(mynn_f32_mul(1.34f, wstar),
            mynn_f32_mul(zratio, mynn_f32_sub(1.0f,
                mynn_f32_div(mynn_f32_mul(0.8f, 50.0f), pblh))));
        const float scalar_zratio = gsd41
            ? mynn_gfk_powf(mynn_f32_div(50.0f, pblh), -onethird) : zratio;
        const float sigma_qt = mynn_f32_mul(mynn_f32_mul(1.34f, qstar), scalar_zratio);
        const float sigma_th = mynn_f32_mul(mynn_f32_mul(1.34f, thstar), scalar_zratio);
        const float wmin = mynn_f32_min2(mynn_f32_mul(sigma_w, 0.1f), 0.1f);
        const float wmax = mynn_f32_min2(mynn_f32_mul(sigma_w, 0.4f),
                                         gsd41 ? 0.3f : 0.5f);
        for (int i = 0; i < nup; ++i) {
            if (gsd41 && i >= int(nup2)) break;
            const int k = 0;
            DMP_PLUME(0, i, k) = mynn_f32_add(wmin, mynn_f32_mul(
                mynn_f32_div(float(i + 1), float(nup)), mynn_f32_sub(wmax, wmin)));
            DMP_PLUME(6, i, k) = mynn_dmp_up(DMP_L(u, 0), DMP_L(u, 1),
                                             DMP_L(dz, 0), DMP_L(dz, 1));
            DMP_PLUME(7, i, k) = mynn_dmp_up(DMP_L(v, 0), DMP_L(v, 1),
                                             DMP_L(dz, 0), DMP_L(dz, 1));
            DMP_PLUME(4, i, k) = 0.0f;
            const float exc_heat = mynn_f32_div(mynn_f32_mul(
                mynn_f32_mul(exc_fac, DMP_PLUME(0, i, k)), sigma_th), sigma_w);
            DMP_PLUME(2, i, k) = mynn_f32_add(mynn_dmp_up(
                DMP_L(thv, 0), DMP_L(thv, 1), DMP_L(dz, 0), DMP_L(dz, 1)), exc_heat);
            DMP_PLUME(1, i, k) = mynn_f32_add(mynn_dmp_up(
                DMP_L(thl, 0), DMP_L(thl, 1), DMP_L(dz, 0), DMP_L(dz, 1)), exc_heat);
            const float exc_moist = mynn_f32_div(mynn_f32_mul(
                mynn_f32_mul(exc_fac, DMP_PLUME(0, i, k)), sigma_qt), sigma_w);
            DMP_PLUME(3, i, k) = mynn_f32_add(mynn_dmp_up(
                DMP_L(qt, 0), DMP_L(qt, 1), DMP_L(dz, 0), DMP_L(dz, 1)), exc_moist);
        }
        if (!gsd41) {
            for (int k = 0; k < nz - 1; ++k)
                DMP_WORK(0, k) = mynn_dmp_up(DMP_L(rho, k), DMP_L(rho, k + 1),
                                             DMP_L(dz, k), DMP_L(dz, k + 1));
            DMP_WORK(0, nz - 1) = DMP_L(rho, nz - 1);
        }

        for (int i = 0; i < nup; ++i) {
            if (gsd41 && i >= int(nup2)) break;
            float plume_qc = 0.0f;
            const float len = mynn_f32_add(minwidth, mynn_f32_mul(dl, float(i)));
            int overshoot = 0;
            for (int k = 1; k < nz - 1; ++k) {
                const float ent_wmin = mynn_f32_add(0.3f,
                                                     mynn_f32_mul(len, 0.0005f));
                float e = gsd41
                    ? mynn_f32_div(0.35f, mynn_f32_mul(mynn_f32_min2(
                        mynn_f32_max2(DMP_PLUME(0, i, k - 1), 0.75f), 1.9f), len))
                    : mynn_f32_div(q.edmf_entrainment, mynn_f32_mul(
                        mynn_f32_min2(mynn_f32_max2(DMP_PLUME(0, i, k - 1),
                            ent_wmin), 0.9f), len));
                e = mynn_f32_max2(e, 0.0003f);
                const float ramp = mynn_f32_min2(mynn_f32_add(pblh, 1500.0f),
                                                   gsd41 ? 3500.0f : 4000.0f);
                if (DMP_I(zw, k) >= ramp)
                    e = mynn_f32_add(e, mynn_f32_mul(
                        mynn_f32_sub(DMP_I(zw, k), ramp), 5.0e-6f));
                e = mynn_f32_mul(e, mynn_f32_sub(1.0f, DMP_L(rstoch, k)));
                e = mynn_f32_min2(e, mynn_f32_div(0.9f,
                    mynn_f32_sub(DMP_I(zw, k + 1), DMP_I(zw, k))));
                DMP_ENT(i, k) = e;
                const float ent_exp = mynn_f32_mul(e,
                    mynn_f32_sub(DMP_I(zw, k + 1), DMP_I(zw, k)));
                const float ent_exm = gsd41 ? ent_exp
                                             : mynn_f32_mul(ent_exp, 0.3333f);
                const float qtn = mynn_f32_add(mynn_f32_mul(
                    DMP_PLUME(3, i, k - 1), mynn_f32_sub(1.0f, ent_exp)),
                    mynn_f32_mul(DMP_L(qt, k), ent_exp));
                const float thln = mynn_f32_add(mynn_f32_mul(
                    DMP_PLUME(1, i, k - 1), mynn_f32_sub(1.0f, ent_exp)),
                    mynn_f32_mul(DMP_L(thl, k), ent_exp));
                const float un = mynn_f32_add(mynn_f32_mul(
                    DMP_PLUME(6, i, k - 1), mynn_f32_sub(1.0f, ent_exm)),
                    mynn_f32_mul(DMP_L(u, k), ent_exm));
                const float vn = mynn_f32_add(mynn_f32_mul(
                    DMP_PLUME(7, i, k - 1), mynn_f32_sub(1.0f, ent_exm)),
                    mynn_f32_mul(DMP_L(v, k), ent_exm));
                const float pk = mynn_dmp_up(DMP_L(p, k), DMP_L(p, k + 1),
                                             DMP_L(dz, k), DMP_L(dz, k + 1));
                float thvn;
                mynn_dmp_condensation(qtn, thln, pk, DMP_I(zw, k + 1),
                                      plume_qc, thvn, gsd41);
                const float thvk = mynn_dmp_up(DMP_L(thv, k), DMP_L(thv, k + 1),
                                               DMP_L(dz, k), DMP_L(dz, k + 1));
                const float buoyancy = mynn_f32_mul(grav,
                    mynn_f32_sub(mynn_f32_div(thvn, thvk), 1.0f));
                const float bcoeff = buoyancy > 0.0f ? 0.15f : 0.2f;
                const float previous = DMP_PLUME(0, i, k - 1);
                const float step = mynn_f32_min2(mynn_f32_sub(
                    DMP_I(zw, k), DMP_I(zw, k - 1)), 250.0f);
                const float divisor = previous < 0.2f
                    ? mynn_f32_max2(previous, 0.2f) : previous;
                float wn = mynn_f32_add(previous, mynn_f32_mul(
                    mynn_f32_add(mynn_f32_mul(mynn_f32_mul(-2.0f, e), previous),
                        mynn_f32_div(mynn_f32_mul(bcoeff, buoyancy), divisor)), step));
                const float limit = mynn_f32_min2(mynn_f32_div(
                    mynn_f32_mul(1.25f, mynn_f32_sub(DMP_I(zw, k),
                    DMP_I(zw, k - 1))), 200.0f), 2.0f);
                wn = mynn_f32_min2(mynn_f32_max2(wn,
                    mynn_f32_sub(previous, limit)), mynn_f32_add(previous, limit));
                wn = mynn_f32_min2(mynn_f32_max2(wn, 0.0f), 3.0f);
                if (gsd41) {
                    if (fltv2 > 0.05f && wn <= 0.0f && overshoot == 0) {
                        overshoot = 1;
                        const float thvkm1 = mynn_dmp_up(DMP_L(thv, k - 1),
                            DMP_L(thv, k), DMP_L(dz, k - 1), DMP_L(dz, k));
                        if (mynn_f32_sub(thvk, thvkm1) > 0.0f) {
                            const float bvf = mynn_f32_sqrt(mynn_f32_div(
                                mynn_f32_mul(gtr, mynn_f32_sub(thvk, thvkm1)),
                                DMP_L(dz, k)));
                            const float frz = mynn_f32_div(previous,
                                mynn_f32_mul(bvf, DMP_L(dz, k)));
                            if (frz >= 0.5f)
                                wn = mynn_f32_mul(mynn_f32_min2(frz, 1.0f), previous);
                        }
                    } else if (fltv2 > 0.05f && overshoot == 1) {
                        wn = 0.0f;
                    }
                    const float decay_z = mynn_f32_max2(mynn_f32_sub(
                        DMP_I(zw, k + 1), mynn_f32_min2(
                            mynn_f32_add(pblh, 2000.0f), 3000.0f)), 0.0f);
                    wn = mynn_f32_mul(wn, mynn_gfk_expf(
                        -mynn_f32_div(decay_z, 1000.0f)));
                    if (DMP_I(zw, k + 1) >= mynn_f32_min2(
                            mynn_f32_add(pblh, 3000.0f), 4500.0f))
                        wn = 0.0f;
                } else if (k == 1 && wn == 0.0f) {
                    nup2 = 0.0f;
                    break;
                }
                if (wn <= 0.0f) break;
                DMP_PLUME(0, i, k) = wn; DMP_PLUME(2, i, k) = thvn;
                DMP_PLUME(1, i, k) = thln; DMP_PLUME(3, i, k) = qtn;
                DMP_PLUME(4, i, k) = plume_qc;
                DMP_PLUME(6, i, k) = un; DMP_PLUME(7, i, k) = vn;
                DMP_PLUME(5, i, k) = DMP_PLUME(5, i, k - 1);
                ktop = max(ktop, k + 1);
            }
        }
    } else nup2 = 0.0f;
    if (ktop > nz - 1) ktop = nz - 1;
    ztop = ktop == 0 ? 0.0f : DMP_I(zw, ktop - 1);

    if (nup2 > 0.0f) {
        for (int i = 0; i < nup; ++i) {
          if (gsd41 && i >= int(nup2)) break;
          for (int k = 0; k < nz - 1; ++k) {
            if (gsd41 && k >= ktop) break;
            const float raw = gsd41 ? DMP_PLUME(5, i, k)
                : mynn_f32_mul(DMP_WORK(0, k), DMP_PLUME(5, i, k));
            const float aw = mynn_f32_mul(raw, DMP_PLUME(0, i, k));
            DMP_I(s_aw, k + 1) = mynn_f32_add(DMP_I(s_aw, k + 1),
                mynn_f32_mul(aw, psig_w));
            DMP_I(s_awthl, k + 1) = mynn_f32_add(DMP_I(s_awthl, k + 1),
                mynn_f32_mul(mynn_f32_mul(aw, DMP_PLUME(1, i, k)), psig_w));
            DMP_I(s_awqt, k + 1) = mynn_f32_add(DMP_I(s_awqt, k + 1),
                mynn_f32_mul(mynn_f32_mul(aw, DMP_PLUME(3, i, k)), psig_w));
            DMP_I(s_awqc, k + 1) = mynn_f32_add(DMP_I(s_awqc, k + 1),
                mynn_f32_mul(mynn_f32_mul(aw, DMP_PLUME(4, i, k)), psig_w));
            DMP_I(s_awqv, k + 1) = mynn_f32_sub(DMP_I(s_awqt, k + 1),
                                                DMP_I(s_awqc, k + 1));
            DMP_I(s_awu, k + 1) = mynn_f32_add(DMP_I(s_awu, k + 1),
                mynn_f32_mul(mynn_f32_mul(aw, DMP_PLUME(6, i, k)), psig_w));
            DMP_I(s_awv, k + 1) = mynn_f32_add(DMP_I(s_awv, k + 1),
                mynn_f32_mul(mynn_f32_mul(aw, DMP_PLUME(7, i, k)), psig_w));
          }
        }
        if (mq.on) {
            // mynn_dmp_qn_flux_columns: s_awqn from the PRE-limiter up_a.
            const ulong nc = ulong(q.ncol);
            device float *upq = mq.up_qn + ulong(column) * ulong(nup) * ulong(nz + 1);
#define DMP_UPQ(i, k) upq[ulong(i) * ulong(nz + 1) + ulong(k)]
            for (int n = 0; n < 4; ++n) {
                device const float *qn = mq.qn[n];
                device float *saw = mq.s_awqn[n];
                for (int s = 0; s < nup * (nz + 1); ++s) upq[s] = 0.0f;
                for (int i = 0; i < nup; ++i) {
                    // WRF :6140-6144 surface value, :6213/:6217 entrainment.
                    DMP_UPQ(i, 0) = mynn_f32_div(mynn_f32_add(
                        mynn_f32_mul(qn[column], DMP_L(dz, 1)),
                        mynn_f32_mul(qn[nc + column], DMP_L(dz, 0))),
                        mynn_f32_add(DMP_L(dz, 0), DMP_L(dz, 1)));
                    for (int k = 1; k < nz - 1; ++k) {
                        if (!(DMP_PLUME(0, i, k) > 0.0f)) continue;
                        const float ent_exp = mynn_f32_mul(DMP_ENT(i, k),
                            mynn_f32_sub(DMP_I(zw, k + 1), DMP_I(zw, k)));
                        DMP_UPQ(i, k) = mynn_f32_add(
                            mynn_f32_mul(DMP_UPQ(i, k - 1),
                                         mynn_f32_sub(1.0f, ent_exp)),
                            mynn_f32_mul(qn[ulong(k) * nc + column], ent_exp));
                    }
                }
                // WRF :6447-6456, k outer and plume inner.
                for (int k = 0; k < nz; ++k)
                    for (int i = 0; i < nup; ++i) {
                        device float &acc = saw[ulong(k + 1) * nc + column];
                        acc = mynn_f32_add(acc, mynn_f32_mul(mynn_f32_mul(
                            mynn_f32_mul(mynn_f32_mul(DMP_WORK(0, k),
                                DMP_PLUME(5, i, k)), DMP_PLUME(0, i, k)),
                            DMP_UPQ(i, k)), psig_w));
                    }
            }
#undef DMP_UPQ
        }
        DMP_WORK(1, 0) = mynn_f32_mul(0.5f,
            mynn_f32_add(DMP_L(dz, 0), DMP_L(dz, 1)));
        const float flx1 = DMP_I(s_aw, 1) != 0.0f
            ? mynn_f32_max2(mynn_f32_div(mynn_f32_mul(DMP_I(s_aw, 1),
                mynn_f32_sub(DMP_L(th, 0), DMP_L(th, 1))), DMP_WORK(1, 0)),
                1.0e-5f) : 0.0f;
        const float threshold = mynn_f32_div(mynn_f32_mul(fluxportion,
            gsd41 ? flt : mynn_f32_max2(flt, 0.0f)), DMP_L(dz, 0));
        if (flx1 > threshold && flx1 > 0.0f) {
            const float adjust = mynn_f32_div(threshold, flx1);
            for (int k = 0; k <= nz; ++k) {
                DMP_I(s_aw, k) = mynn_f32_mul(DMP_I(s_aw, k), adjust);
                DMP_I(s_awthl, k) = mynn_f32_mul(DMP_I(s_awthl, k), adjust);
                DMP_I(s_awqt, k) = mynn_f32_mul(DMP_I(s_awqt, k), adjust);
                DMP_I(s_awqv, k) = mynn_f32_mul(DMP_I(s_awqv, k), adjust);
                DMP_I(s_awqc, k) = mynn_f32_mul(DMP_I(s_awqc, k), adjust);
                DMP_I(s_awu, k) = mynn_f32_mul(DMP_I(s_awu, k), adjust);
                DMP_I(s_awv, k) = mynn_f32_mul(DMP_I(s_awv, k), adjust);
                if (mq.on) for (int n = 0; n < 4; ++n) {
                    device float *saw = mq.s_awqn[n];
                    saw[ulong(k) * ulong(q.ncol) + column] = mynn_f32_mul(
                        saw[ulong(k) * ulong(q.ncol) + column], adjust);
                }
                for (int i = 0; i < nup; ++i)
                    DMP_PLUME(5, i, k) = mynn_f32_mul(DMP_PLUME(5, i, k), adjust);
            }
        }
        for (int k = 0; k < nz - 1; ++k) for (int i = 0; i < nup; ++i) {
            if (gsd41 && k >= ktop) break;
            const float a = DMP_PLUME(5, i, k);
            DMP_L(edmf_a, k) = mynn_f32_add(DMP_L(edmf_a, k), a);
            DMP_L(edmf_w, k) = mynn_f32_add(DMP_L(edmf_w, k),
                mynn_f32_mul(a, DMP_PLUME(0, i, k)));
            DMP_L(edmf_qt, k) = mynn_f32_add(DMP_L(edmf_qt, k),
                mynn_f32_mul(a, DMP_PLUME(3, i, k)));
            DMP_L(edmf_thl, k) = mynn_f32_add(DMP_L(edmf_thl, k),
                mynn_f32_mul(a, DMP_PLUME(1, i, k)));
            DMP_L(edmf_ent, k) = mynn_f32_add(DMP_L(edmf_ent, k),
                mynn_f32_mul(a, DMP_ENT(i, k)));
            DMP_L(edmf_qc, k) = mynn_f32_add(DMP_L(edmf_qc, k),
                mynn_f32_mul(a, DMP_PLUME(4, i, k)));
        }
        for (int k = 0; k < nz - 1; ++k) if (!(gsd41 && k >= ktop)
                                          && DMP_L(edmf_a, k) > 0.0f) {
            DMP_L(edmf_w, k) = mynn_f32_div(DMP_L(edmf_w, k), DMP_L(edmf_a, k));
            DMP_L(edmf_qt, k) = mynn_f32_div(DMP_L(edmf_qt, k), DMP_L(edmf_a, k));
            DMP_L(edmf_thl, k) = mynn_f32_div(DMP_L(edmf_thl, k), DMP_L(edmf_a, k));
            DMP_L(edmf_ent, k) = mynn_f32_div(DMP_L(edmf_ent, k), DMP_L(edmf_a, k));
            DMP_L(edmf_qc, k) = mynn_f32_div(DMP_L(edmf_qc, k), DMP_L(edmf_a, k));
            DMP_L(edmf_a, k) = mynn_f32_mul(DMP_L(edmf_a, k), psig_w);
            maxmf = mynn_f32_max2(maxmf, mynn_f32_mul(
                DMP_L(edmf_a, k), DMP_L(edmf_w, k)));
        }
        for (int k = 0; k < nz - 1; ++k) {
            const float exneri = mynn_dmp_up(DMP_L(exner, k), DMP_L(exner, k + 1),
                                             DMP_L(dz, k), DMP_L(dz, k + 1));
            DMP_WORK(2, k) = mynn_f32_add(DMP_L(edmf_thl, k),
                mynn_f32_mul(mynn_f32_div(xlvcp, exneri), DMP_L(edmf_qc, k)));
            DMP_WORK(1, k) = mynn_f32_mul(0.5f,
                mynn_f32_add(DMP_L(dz, k), DMP_L(dz, k + 1)));
        }
        for (int k = 1; k < nz - 2; ++k) {
            if (k + 1 > ktop) break;
            if (gsd41) {
                if (mynn_f32_mul(0.5f, mynn_f32_add(
                        DMP_L(edmf_qc, k), DMP_L(edmf_qc, k - 1))) > 0.0f) {
                    const float rv = 461.6f;
                    const float cp = mynn_f32_div(mynn_f32_mul(7.0f, 287.0f), 2.0f);
                    const float rcp = mynn_f32_div(287.0f, cp);
                    const float ep2 = mynn_f32_div(287.0f, rv);
                    const float ep3 = mynn_f32_sub(1.0f, ep2);
                    const float cpv_g = mynn_f32_mul(4.0f, rv);
                    const float xlvcp_g = mynn_f32_div(2.5e6f, cp);
                    const float tv0_g = mynn_f32_mul(
                        mynn_f32_sub(mynn_f32_div(rv, 287.0f), 1.0f), 300.0f);
                    const float thp = mynn_dmp_up(DMP_WORK(2, k), DMP_WORK(2, k - 1),
                                                  DMP_WORK(1, k), DMP_WORK(1, k - 1));
                    const float qtp = mynn_dmp_up(DMP_L(edmf_qt, k),
                        DMP_L(edmf_qt, k - 1), DMP_WORK(1, k), DMP_WORK(1, k - 1));
                    const float tp = mynn_f32_mul(thp, DMP_L(exner, k));
                    const float esat = mynn_gsd41_esat_blend(tp);
                    const float qsl = mynn_f32_div(mynn_f32_mul(ep2, esat),
                        mynn_f32_max2(1.0e-4f, mynn_f32_sub(DMP_L(p, k),
                                                          mynn_f32_mul(ep3, esat))));
                    const float qcp = (DMP_L(edmf_qc, k) > 0.0f
                                       && DMP_L(edmf_qc, k - 1) > 0.0f)
                        ? mynn_f32_mul(0.5f, mynn_f32_add(
                            DMP_L(edmf_qc, k), DMP_L(edmf_qc, k - 1)))
                        : mynn_f32_max2(0.0f, mynn_f32_sub(qtp, qsl));
                    const float xl = mynn_gsd41_xl_blend(DMP_L(tk, k));
                    const float tlk = mynn_f32_mul(DMP_L(thl, k),
                        mynn_gfk_powf(mynn_f32_div(DMP_L(p, k), 100000.0f), rcp));
                    const float qsat_tl = mynn_gsd41_qsat_blend(tlk, DMP_L(p, k));
                    const float rsl = mynn_f32_div(mynn_f32_mul(xl, qsat_tl),
                        mynn_f32_mul(rv, mynn_f32_mul(tlk, tlk)));
                    const float cpm = mynn_f32_add(cp, mynn_f32_mul(DMP_L(qt, k), cpv_g));
                    const float acb = mynn_f32_div(1.0f,
                        mynn_f32_add(1.0f, mynn_f32_div(mynn_f32_mul(xl, rsl), cpm)));
                    const float b9 = mynn_f32_mul(acb, rsl);
                    const float asum = mynn_f32_add(DMP_L(edmf_a, k), DMP_L(edmf_a, k - 1));
                    const float q2p = mynn_f32_div(xlvcp_g, DMP_L(exner, k));
                    const float pt = mynn_f32_add(DMP_L(thl, k),
                        mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(q2p, qcp), 0.5f), asum));
                    const float bb = mynn_f32_div(mynn_f32_mul(b9, DMP_L(tk, k)), pt);
                    const float qww = mynn_f32_add(1.0f,
                                      mynn_f32_mul(0.61f, DMP_L(qt, k)));
                    const float alpha = mynn_f32_mul(0.61f, pt);
                    const float tt = mynn_f32_mul(DMP_L(th, k), DMP_L(exner, k));
                    const float beta = mynn_f32_sub(mynn_f32_div(mynn_f32_mul(pt, xl),
                        mynn_f32_mul(tt, cp)), mynn_f32_mul(1.61f, pt));
                    const float f = acb > 0.0f
                        ? mynn_f32_min2(mynn_f32_div(1.0f, acb), 4.0f) : 1.0f;
                    float sigq = mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(
                        mynn_f32_mul(mynn_f32_mul(9.0e-3f, 0.5f), asum), 0.5f),
                        mynn_f32_add(DMP_L(edmf_w, k), DMP_L(edmf_w, k - 1))), f);
                    const float sgm = sgm_gsd41[ulong(k) * ulong(q.ncol) + column];
                    sigq = mynn_f32_sqrt(mynn_f32_add(mynn_f32_mul(sigq, sigq),
                                                       mynn_f32_mul(sgm, sgm)));
                    const float qmq = mynn_f32_mul(acb,
                        mynn_f32_sub(DMP_L(qt, k), qsat_tl));
                    const float mf_cf = mynn_f32_min2(mynn_f32_max2(
                        mynn_f32_add(0.5f, mynn_f32_mul(0.36f,
                            mynn_gfk_atanf(mynn_f32_mul(1.55f,
                                mynn_f32_div(qmq, sigq))))), 0.01f), 0.6f);
                    if (DMP_L(cldfra_bl, k) < 0.5f) {
                        const float ahalf = mynn_f32_mul(0.5f, asum);
                        if (mf_cf > ahalf) {
                            DMP_L(cldfra_bl, k) = mf_cf;
                            DMP_L(qc_bl, k) = mynn_f32_div(
                                mynn_f32_mul(mynn_f32_mul(qcp, 0.5f), asum), mf_cf);
                        } else {
                            DMP_L(cldfra_bl, k) = ahalf;
                            DMP_L(qc_bl, k) = qcp;
                        }
                    }
                    float q1 = mynn_f32_max2(mynn_f32_div(qmq,
                        mynn_f32_max2(sigq, 1.0e-10f)), -5.0f);
                    float fng = q1 >= 1.0f ? 1.0f :
                        (q1 >= -1.7f ? mynn_gfk_expf(mynn_f32_mul(-0.4f,
                            mynn_f32_sub(q1, 1.0f))) :
                         (q1 >= -2.5f ? mynn_f32_add(3.0f, mynn_gfk_expf(
                            mynn_f32_mul(-3.8f, mynn_f32_add(q1, 1.7f)))) :
                          mynn_f32_min2(mynn_f32_add(23.9f, mynn_gfk_expf(
                            mynn_f32_mul(-1.6f, mynn_f32_add(q1, 2.5f)))), 60.0f)));
                    const float cw = mynn_f32_min2(0.4f, DMP_L(cldfra_bl, k));
                    DMP_L(vt, k) = mynn_f32_sub(mynn_f32_sub(qww,
                        mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(cw, beta), bb), fng)), 1.0f);
                    DMP_L(vq, k) = mynn_f32_sub(mynn_f32_add(alpha,
                        mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(cw, beta), acb), fng)), tv0_g);
                }
                continue;
            }
            if (!(mynn_f32_mul(0.5f, mynn_f32_add(DMP_L(edmf_qc, k),
                    DMP_L(edmf_qc, k - 1))) > 0.0f
                  && DMP_L(cldfra_bl, k) < cf_thresh)) continue;
            const float aup = mynn_dmp_up(DMP_L(edmf_a, k), DMP_L(edmf_a, k - 1),
                                          DMP_WORK(1, k), DMP_WORK(1, k - 1));
            const float qtp = mynn_dmp_up(DMP_L(edmf_qt, k), DMP_L(edmf_qt, k - 1),
                                          DMP_WORK(1, k), DMP_WORK(1, k - 1));
            const float qcp = DMP_L(edmf_qc, k) > 0.0f && DMP_L(edmf_qc, k - 1) > 0.0f
                ? mynn_dmp_up(DMP_L(edmf_qc, k), DMP_L(edmf_qc, k - 1),
                              DMP_WORK(1, k), DMP_WORK(1, k - 1))
                : mynn_f32_max2(DMP_L(edmf_qc, k), DMP_L(edmf_qc, k - 1));
            const float xl = mynn_xl_blend(DMP_L(tk, k));
            const float qsat_tk = mynn_qsat_blend(DMP_L(tk, k), DMP_L(p, k));
            const float rsl = mynn_f32_div(mynn_f32_mul(xl, qsat_tk),
                mynn_f32_mul(461.6f, mynn_f32_mul(DMP_L(tk, k), DMP_L(tk, k))));
            const float cpm = mynn_f32_add(1004.5f, mynn_f32_mul(DMP_L(qt, k), cpv));
            const float acb = mynn_f32_div(1.0f,
                mynn_f32_add(1.0f, mynn_f32_div(mynn_f32_mul(xl, rsl), cpm)));
            const float b9 = mynn_f32_mul(acb, rsl);
            const float q2p = mynn_f32_div(xlvcp, DMP_L(exner, k));
            const float pt = mynn_f32_add(DMP_L(thl, k),
                mynn_f32_mul(mynn_f32_mul(q2p, qcp), aup));
            const float bb = mynn_f32_div(mynn_f32_mul(b9, DMP_L(tk, k)), pt);
            const float qww = mynn_f32_add(1.0f, mynn_f32_mul(0.61f, DMP_L(qt, k)));
            const float alpha = mynn_f32_mul(0.61f, pt);
            const float beta = mynn_f32_sub(mynn_f32_div(mynn_f32_mul(pt, xl),
                mynn_f32_mul(DMP_L(tk, k), 1004.5f)), mynn_f32_mul(1.61f, pt));
            float sigq = mynn_f32_mul(mynn_f32_mul(10.0f, aup),
                                      mynn_f32_sub(qtp, DMP_L(qt, k)));
            sigq = mynn_f32_min2(mynn_f32_max2(sigq,
                mynn_f32_mul(qsat_tk, 0.02f)), mynn_f32_mul(qsat_tk, 0.25f));
            float q1 = mynn_f32_div(mynn_f32_mul(acb,
                mynn_f32_sub(DMP_L(qt, k), qsat_tk)), sigq);
            float mf_cf = mynn_f32_min2(mynn_f32_max2(mynn_f32_add(0.5f,
                mynn_f32_mul(0.36f, mynn_gfk_atanf(mynn_f32_mul(1.55f, q1)))),
                0.01f), 0.6f);
            mf_cf = mynn_f32_max2(mf_cf, mynn_f32_mul(
                mynn_f32_sub(landsea, 1.5f) >= 0.0f ? 1.2f : 1.8f, aup));
            mf_cf = mynn_f32_min2(mf_cf, mynn_f32_mul(5.0f, aup));
            const float qc_plume = mynn_f32_mul(qcp, aup);
            DMP_L(qc_bl, k) = qc_plume > 5.0e-5f
                ? mynn_f32_sub(mynn_f32_mul(1.86f, qc_plume), 2.2e-5f)
                : mynn_f32_mul(1.18f, qc_plume);
            DMP_L(cldfra_bl, k) = mf_cf;
            q1 = mynn_f32_max2(q1, -2.25f);
            float fng = q1 >= 1.0f ? 1.0f :
                (q1 >= -1.7f ? mynn_gfk_expf(mynn_f32_mul(-0.4f,
                    mynn_f32_sub(q1, 1.0f))) :
                 (q1 >= -2.5f ? mynn_f32_add(3.0f, mynn_gfk_expf(
                    mynn_f32_mul(-3.8f, mynn_f32_add(q1, 1.7f)))) :
                  mynn_f32_min2(mynn_f32_add(23.9f, mynn_gfk_expf(
                    mynn_f32_mul(-1.6f, mynn_f32_add(q1, 2.5f)))), 60.0f)));
            DMP_L(vt, k) = mynn_f32_sub(mynn_f32_sub(qww,
                mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(
                    1.5f, aup), beta), bb), fng)), 1.0f);
            DMP_L(vq, k) = mynn_f32_sub(mynn_f32_add(alpha,
                mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(mynn_f32_mul(
                    1.5f, aup), beta), acb), fng)), tv0);
        }
    }
    if (ktop > 0) {
        float maxqc = DMP_L(edmf_qc, 0);
        for (int k = 1; k < ktop; ++k)
            maxqc = mynn_f32_max2(maxqc, DMP_L(edmf_qc, k));
        if (maxqc < 1.0e-8f) maxmf = mynn_f32_mul(-1.0f, maxmf);
    }
    r.maxwidth[column] = maxwidth; r.ktop[column] = ktop;
    r.ztop[column] = ztop; r.maxmf[column] = maxmf;
}

kernel void mynn_dmp_mf_columns(
    constant MynnDmpMfColumnsResources &r [[buffer(0)]],
    constant MynnDmpMfColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    MynnDmpQn mq = {{r.qnc, r.qni, r.qnwfa, r.qnifa},
        {r.s_awqnc, r.s_awqni, r.s_awqnwfa, r.s_awqnifa}, r.up_qn,
        q.mixscalars};
    mynn_dmp_mf_body(r, q, column, false, r.dz, mq);
}

kernel void mynn_dmp_mf_gsd41_columns(
    constant MynnDmpMfGsd41ColumnsResources &r [[buffer(0)]],
    constant MynnDmpMfGsd41ColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    MynnDmpQn mq = {{nullptr, nullptr, nullptr, nullptr},
        {nullptr, nullptr, nullptr, nullptr}, nullptr, 0};
    mynn_dmp_mf_body(r, q, column, true, r.sgm, mq);
}

#undef DMP_L
#undef DMP_I
#undef DMP_PLUME
#undef DMP_WORK
#undef DMP_ENT

// WRF v4.6.1 phys/module_pbl_driver.F:2598-2844 diff4d/diff/invert: the
// scalar_pblmix=1 local diffusion of one number scalar, after MYNN.  Metal
// twin of scalar_pblmix_columns in woof/core/kernels/mynn_scalar_mix.cu,
// op for op on the soft32 RN layer.  Arrays are level-major (nz, ncol);
// exch_h is MYNN's diffusivity at each layer's lower interface.  Bottom flux
// is zero, the top value is prescribed.  The interface coefficient cddz is
// recomputed rolling instead of stored, which is the same arithmetic.
static inline float mynn_scalar_pblmix_cddz(
    constant MynnScalarPblmixColumnsResources &r, uint k, uint column,
    uint ncol, int nz)
{
    if (k == 0u || int(k) >= nz) return 0.0f;
    const uint lo = (k - 1u) * ncol + column;
    const uint hi = k * ncol + column;
    const float rhoz = mynn_f32_div(
        mynn_f32_add(mynn_f32_mul(r.rho[hi], r.dz[lo]),
                     mynn_f32_mul(r.rho[lo], r.dz[hi])),
        mynn_f32_add(r.dz[lo], r.dz[hi]));
    return mynn_f32_div(mynn_f32_mul(mynn_f32_mul(2.0f, rhoz), r.exch_h[hi]),
                        mynn_f32_add(r.dz[hi], r.dz[lo]));
}

kernel void mynn_scalar_pblmix_columns(
    constant MynnScalarPblmixColumnsResources &r [[buffer(0)]],
    constant MynnScalarPblmixColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    if (column >= uint(q.ncol)) return;
    const int nz = q.nz;
    const uint ncol = uint(q.ncol);
    const float dt = q.dt;
    device float *a = r.scratch;
    device float *b = r.scratch + uint(nz) * ncol;
    device float *c = b + uint(nz) * ncol;
    device float *rhs = c + uint(nz) * ncol;
#define SPM(k) (uint(k) * ncol + column)
    float lower = 0.0f;
    for (int k = 0; k < nz - 1; ++k) {
        const float upper = mynn_scalar_pblmix_cddz(r, uint(k + 1), column, ncol, nz);
        const float dzk = r.dz[SPM(k)];
        const float rhok = r.rho[SPM(k)];
        a[SPM(k)] = mynn_f32_div(mynn_f32_div(mynn_f32_mul(-lower, dt), dzk), rhok);
        b[SPM(k)] = mynn_f32_add(1.0f, mynn_f32_div(mynn_f32_div(
            mynn_f32_mul(dt, mynn_f32_add(lower, upper)), dzk), rhok));
        c[SPM(k)] = mynn_f32_div(mynn_f32_div(mynn_f32_mul(-upper, dt), dzk), rhok);
        rhs[SPM(k)] = r.qn[SPM(k)];
        lower = upper;
    }
    a[SPM(nz - 1)] = 0.0f;
    b[SPM(nz - 1)] = 1.0f;
    c[SPM(nz - 1)] = 0.0f;
    rhs[SPM(nz - 1)] = r.qn[SPM(nz - 1)];
    // WRF invert: top-down elimination, bottom-up substitution, division.
    for (int k = nz - 2; k >= 0; --k) {
        rhs[SPM(k)] = mynn_f32_sub(rhs[SPM(k)], mynn_f32_div(
            mynn_f32_mul(c[SPM(k)], rhs[SPM(k + 1)]), b[SPM(k + 1)]));
        b[SPM(k)] = mynn_f32_sub(b[SPM(k)], mynn_f32_div(
            mynn_f32_mul(c[SPM(k)], a[SPM(k + 1)]), b[SPM(k + 1)]));
    }
    for (int k = 1; k < nz; ++k)
        rhs[SPM(k)] = mynn_f32_sub(rhs[SPM(k)], mynn_f32_div(
            mynn_f32_mul(a[SPM(k)], rhs[SPM(k - 1)]), b[SPM(k - 1)]));
    for (int k = 0; k < nz; ++k) {
        const float solved = mynn_f32_div(rhs[SPM(k)], b[SPM(k)]);
        r.rate[SPM(k)] = mynn_f32_div(mynn_f32_sub(solved, r.qn[SPM(k)]), dt);
    }
#undef SPM
}

// WRF v4.6.1 module_bl_mynn.F:4654-4689 qn-family tridiagonal solve with the
// mass-flux terms: Metal twin of mynn_mix_scalar_columns in
// woof/core/kernels/mynn_scalar_mix.cu, op for op.  Level-major (nz, ncol);
// scratch holds khdz (nz+1) then a, b, c, d, cpw, dpw (nz each).
kernel void mynn_mix_scalar_columns(
    constant MynnMixScalarColumnsResources &r [[buffer(0)]],
    constant MynnMixScalarColumnsParams &q [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    if (column >= uint(q.ncol)) return;
    const int nz = q.nz;
    const ulong nc = ulong(q.ncol);
#define MS(p, k) p[ulong(k) * nc + column]
    device float *khdz = r.scratch;
    device float *a = khdz + ulong(nz + 1) * nc;
    device float *b = a + ulong(nz) * nc;
    device float *c = b + ulong(nz) * nc;
    device float *d = c + ulong(nz) * nc;
    device float *cpw = d + ulong(nz) * nc;
    device float *dpw = cpw + ulong(nz) * nc;
    const float delt = r.delt[column];
    float rhoz = MS(r.rho, 0);
    MS(khdz, 0) = mynn_f32_mul(rhoz, MS(r.dfh, 0));
    for (int k = 1; k < nz; ++k) {
        rhoz = mynn_f32_div(
            mynn_f32_add(mynn_f32_mul(MS(r.rho, k), MS(r.dz, k - 1)),
                         mynn_f32_mul(MS(r.rho, k - 1), MS(r.dz, k))),
            mynn_f32_add(MS(r.dz, k - 1), MS(r.dz, k)));
        rhoz = mynn_f32_max2(rhoz, 1.0e-4f);
        MS(khdz, k) = mynn_f32_mul(rhoz, MS(r.dfh, k));
    }
    MS(khdz, nz) = mynn_f32_mul(rhoz, MS(r.dfh, nz - 1));
    for (int k = 1; k < nz - 1; ++k) {
        MS(khdz, k) = mynn_f32_max2(MS(khdz, k),
            mynn_f32_mul(0.5f, MS(r.s_aw, k)));
        MS(khdz, k) = mynn_f32_max2(MS(khdz, k), -mynn_f32_mul(0.5f,
            mynn_f32_sub(MS(r.s_aw, k), MS(r.s_aw, k + 1))));
    }
    {
        const float dtz0 = mynn_f32_div(delt, MS(r.dz, 0));
        const float rhoinv0 = mynn_f32_div(1.0f, MS(r.rho, 0));
        const float hdz0 = mynn_f32_mul(mynn_f32_mul(0.5f, dtz0), rhoinv0);
        const float dzinv0 = mynn_f32_mul(dtz0, rhoinv0);
        MS(a, 0) = -mynn_f32_mul(mynn_f32_mul(dtz0, MS(khdz, 0)), rhoinv0);
        MS(b, 0) = mynn_f32_sub(
            mynn_f32_add(1.0f, mynn_f32_mul(mynn_f32_mul(dtz0,
                mynn_f32_add(MS(khdz, 1), MS(khdz, 0))), rhoinv0)),
            mynn_f32_mul(mynn_f32_mul(hdz0, MS(r.s_aw, 1)), 1.0f));
        MS(c, 0) = mynn_f32_sub(
            -mynn_f32_mul(mynn_f32_mul(dtz0, MS(khdz, 1)), rhoinv0),
            mynn_f32_mul(mynn_f32_mul(hdz0, MS(r.s_aw, 1)), 1.0f));
        MS(d, 0) = mynn_f32_sub(MS(r.qn, 0),
            mynn_f32_mul(mynn_f32_mul(dzinv0, MS(r.s_awqn, 1)), 1.0f));
    }
    for (int k = 1; k < nz - 1; ++k) {
        const float dtz = mynn_f32_div(delt, MS(r.dz, k));
        const float rhoinv = mynn_f32_div(1.0f,
            mynn_f32_max2(MS(r.rho, k), 1.0e-4f));
        const float hdz = mynn_f32_mul(mynn_f32_mul(0.5f, dtz), rhoinv);
        const float dzinv = mynn_f32_mul(dtz, rhoinv);
        MS(a, k) = mynn_f32_add(
            -mynn_f32_mul(mynn_f32_mul(dtz, MS(khdz, k)), rhoinv),
            mynn_f32_mul(mynn_f32_mul(hdz, MS(r.s_aw, k)), 1.0f));
        MS(b, k) = mynn_f32_add(
            mynn_f32_add(1.0f, mynn_f32_mul(mynn_f32_mul(dtz,
                mynn_f32_add(MS(khdz, k), MS(khdz, k + 1))), rhoinv)),
            mynn_f32_mul(mynn_f32_mul(hdz, mynn_f32_sub(MS(r.s_aw, k),
                MS(r.s_aw, k + 1))), 1.0f));
        MS(c, k) = mynn_f32_sub(
            -mynn_f32_mul(mynn_f32_mul(dtz, MS(khdz, k + 1)), rhoinv),
            mynn_f32_mul(mynn_f32_mul(hdz, MS(r.s_aw, k + 1)), 1.0f));
        MS(d, k) = mynn_f32_add(MS(r.qn, k),
            mynn_f32_mul(mynn_f32_mul(dzinv, mynn_f32_sub(MS(r.s_awqn, k),
                MS(r.s_awqn, k + 1))), 1.0f));
    }
    MS(a, nz - 1) = 0.0f;
    MS(b, nz - 1) = 1.0f;
    MS(c, nz - 1) = 0.0f;
    MS(d, nz - 1) = MS(r.qn, nz - 1);
    // module_bl_mynn.F:5422 tridiag2.
    MS(cpw, 0) = mynn_f32_div(MS(c, 0), MS(b, 0));
    MS(dpw, 0) = mynn_f32_div(MS(d, 0), MS(b, 0));
    for (int k = 1; k < nz; ++k) {
        const float m = mynn_f32_sub(MS(b, k),
            mynn_f32_mul(MS(cpw, k - 1), MS(a, k)));
        MS(cpw, k) = mynn_f32_div(MS(c, k), m);
        MS(dpw, k) = mynn_f32_div(mynn_f32_sub(MS(d, k),
            mynn_f32_mul(MS(dpw, k - 1), MS(a, k))), m);
    }
    MS(r.qn2, nz - 1) = MS(dpw, nz - 1);
    for (int k = nz - 2; k >= 0; --k)
        MS(r.qn2, k) = mynn_f32_sub(MS(dpw, k),
            mynn_f32_mul(MS(cpw, k), MS(r.qn2, k + 1)));
    for (int k = 0; k < nz; ++k)
        MS(r.dqn, k) = mynn_f32_div(
            mynn_f32_sub(MS(r.qn2, k), MS(r.qn, k)), delt);
#undef MS
}
