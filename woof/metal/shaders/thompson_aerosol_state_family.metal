#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "soft_binary64.metalh"
#include "soft_gfk_pow.metalh"
#include "soft_libm64.metalh"
#include "thompson_aerosol_state_family_bindings.metalh"

using namespace metal;

static inline float thompson_aa_add(float a, float b);
static inline float thompson_aa_sub(float a, float b);
static inline float thompson_aa_mul(float a, float b);
static inline float thompson_aa_div(float a, float b);

// calc_effectRad's cloud-size ladder and snow fits. These stay local to this
// shader family because no compiled Metal helper currently exposes them.
constant float THOMPSON_AA_EFFRAD_G_RATIO[16] = {
    0.0f, 24.0f, 60.0f, 120.0f, 210.0f, 336.0f, 504.0f, 720.0f,
    990.0f, 1320.0f, 1716.0f, 2184.0f, 2730.0f, 3360.0f, 4080.0f,
    4896.0f};
constant float THOMPSON_AA_EFFRAD_SA[10] = {
    5.065339f, -0.062659f, -3.032362f, 0.029469f, -0.000285f,
    0.31255f, 0.000204f, 0.003199f, 0.0f, -0.015952f};
constant float THOMPSON_AA_EFFRAD_SB[10] = {
    0.476221f, -0.015896f, 0.165977f, 0.007468f, -0.000141f,
    0.060366f, 0.000079f, 0.000594f, 0.0f, -0.003577f};

static inline ulong thompson_aa_effrad_widen(float x) {
    return soft64_from_f32_bits(as_type<uint>(x));
}

static inline float thompson_aa_effrad_narrow(ulong x) {
    return as_type<float>(soft64_to_f32_bits(x));
}

// WRF REAL(4)**REAL(4), lowered by gfortran to a correctly rounded powf.
static inline float thompson_aa_effrad_powf(float x, float y) {
    return thompson_aa_effrad_narrow(soft64_plm_pow(
        thompson_aa_effrad_widen(x), thompson_aa_effrad_widen(y)));
}

static inline int thompson_aa_effrad_inu_c(float nc_m3) {
    if (nc_m3 < 100.0f) return 15;
    if (nc_m3 > 1.0e10f) return 2;
    return min(15, int(floor(thompson_aa_add(
        thompson_aa_div(1000000000.0f, nc_m3), 0.5f))) + 2);
}

static inline float thompson_aa_effrad_cloud(float rc, float nc_m3) {
    constexpr float AM_R = 5.235988159e+02f;
    constexpr float OBMR = 3.333333433e-01f;
    const int inu_c = thompson_aa_effrad_inu_c(nc_m3);
    const float argument = thompson_aa_div(
        thompson_aa_mul(thompson_aa_mul(nc_m3, AM_R),
                        THOMPSON_AA_EFFRAD_G_RATIO[inu_c]), rc);
    const ulong lambda = thompson_aa_effrad_widen(
        thompson_aa_effrad_powf(argument, OBMR));
    const float numerator = thompson_aa_add(3.0f, float(inu_c));
    const ulong diagnosed = soft64_div(
        soft64_mul(0x3fe0000000000000ul,
                   thompson_aa_effrad_widen(numerator)), lambda);
    return fmax(2.51e-6f, fmin(thompson_aa_effrad_narrow(diagnosed),
                               50.0e-6f));
}

static inline float thompson_aa_effrad_ice(float ri, float ni) {
    constexpr float AM_I = 4.660029297e+02f;
    constexpr float CIG2 = 6.0f;
    constexpr float OIG1 = 1.0f;
    constexpr float OBMI = 3.333333433e-01f;
    constexpr float MU_I = 0.0f;
    float argument = thompson_aa_mul(AM_I, CIG2);
    argument = thompson_aa_mul(argument, OIG1);
    argument = thompson_aa_mul(argument, ni);
    argument = thompson_aa_div(argument, ri);
    const ulong lambda = thompson_aa_effrad_widen(
        thompson_aa_effrad_powf(argument, OBMI));
    const float numerator = thompson_aa_add(3.0f, MU_I);
    const ulong diagnosed = soft64_div(
        soft64_mul(0x3fe0000000000000ul,
                   thompson_aa_effrad_widen(numerator)), lambda);
    return fmax(2.51e-6f, fmin(thompson_aa_effrad_narrow(diagnosed),
                               125.0e-6f));
}

static inline float thompson_aa_effrad_field_a(float tc, float moment) {
    float v = THOMPSON_AA_EFFRAD_SA[0];
    v = thompson_aa_add(v, thompson_aa_mul(THOMPSON_AA_EFFRAD_SA[1], tc));
    v = thompson_aa_add(v, thompson_aa_mul(THOMPSON_AA_EFFRAD_SA[2], moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        THOMPSON_AA_EFFRAD_SA[3], tc), moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        THOMPSON_AA_EFFRAD_SA[4], tc), tc));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        THOMPSON_AA_EFFRAD_SA[5], moment), moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
        THOMPSON_AA_EFFRAD_SA[6], tc), tc), moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
        THOMPSON_AA_EFFRAD_SA[7], tc), moment), moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
        THOMPSON_AA_EFFRAD_SA[8], tc), tc), tc));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
        THOMPSON_AA_EFFRAD_SA[9], moment), moment), moment));
    return thompson_aa_effrad_powf(10.0f, v);
}

static inline float thompson_aa_effrad_field_b(float tc, float moment) {
    float v = THOMPSON_AA_EFFRAD_SB[0];
    v = thompson_aa_add(v, thompson_aa_mul(THOMPSON_AA_EFFRAD_SB[1], tc));
    v = thompson_aa_add(v, thompson_aa_mul(THOMPSON_AA_EFFRAD_SB[2], moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        THOMPSON_AA_EFFRAD_SB[3], tc), moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        THOMPSON_AA_EFFRAD_SB[4], tc), tc));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(
        THOMPSON_AA_EFFRAD_SB[5], moment), moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
        THOMPSON_AA_EFFRAD_SB[6], tc), tc), moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
        THOMPSON_AA_EFFRAD_SB[7], tc), moment), moment));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
        THOMPSON_AA_EFFRAD_SB[8], tc), tc), tc));
    v = thompson_aa_add(v, thompson_aa_mul(thompson_aa_mul(thompson_aa_mul(
        THOMPSON_AA_EFFRAD_SB[9], moment), moment), moment));
    return v;
}

static inline float thompson_aa_effrad_snow(float rs, float t_k) {
    constexpr float OAMS = 1.449275398e+01f;
    const float tc0 = fmin(-0.1f, thompson_aa_sub(t_k, 273.15f));
    const float smob = thompson_aa_mul(rs, OAMS);
    const float smoc = thompson_aa_mul(
        thompson_aa_effrad_field_a(tc0, 3.0f),
        thompson_aa_effrad_powf(smob,
                                thompson_aa_effrad_field_b(tc0, 3.0f)));
    const float diagnosed = thompson_aa_mul(
        0.5f, thompson_aa_div(smoc, smob));
    return fmax(5.01e-6f, fmin(diagnosed, 999.0e-6f));
}

static inline float thompson_aa_effrad_density(float pressure,
                                                float temperature, float qv) {
    const float numerator = thompson_aa_mul(0.622f, pressure);
    const float qmix = thompson_aa_add(qv, 0.622f);
    const float denominator = thompson_aa_mul(
        thompson_aa_mul(287.04f, temperature), qmix);
    return thompson_aa_div(numerator, denominator);
}

static inline float thompson_aa_effrad_clamp_nc(float nc) {
    return fmax(2.0f, fmin(nc, 1999.0e6f));
}

constant float THOMPSON_AA_CCG1[16] = {
    0.0f, 1.0f, 2.0f, 6.0f, 24.0f, 120.0000076f, 720.0000610f,
    5040.001953f, 40319.99609f, 362879.9688f, 3628801.75f,
    39916800.0f, 479001856.0f, 6227022336.0f, 87178297344.0f,
    1307673886720.0f};
constant float THOMPSON_AA_CCG2[16] = {
    0.0f, 24.0f, 120.0000076f, 720.0000610f, 5040.001953f,
    40319.99609f, 362879.9688f, 3628801.75f, 39916800.0f,
    479001856.0f, 6227022336.0f, 87178297344.0f, 1307673886720.0f,
    20922782191616.0f, 355687448240128.0f, 6402383730776064.0f};
constant float THOMPSON_AA_OCG1[16] = {
    0.0f, 1.0f, 0.5f, 0.1666666716f, 0.04166666791f, 0.008333332837f,
    0.001388888806f, 0.0001984126284f, 0.00002480158946f,
    0.000002755732112f, 0.0000002755730577f, 0.00000002505210794f,
    0.000000002087674478f, 0.0000000001605904021f,
    0.00000000001147074449f, 0.0000000000007647166320f};
constant float THOMPSON_AA_OCG2[16] = {
    0.0f, 0.04166666791f, 0.008333332837f, 0.001388888806f,
    0.0001984126284f, 0.00002480158946f, 0.000002755732112f,
    0.0000002755730577f, 0.00000002505210794f,
    0.000000002087674478f, 0.0000000001605904021f,
    0.00000000001147074449f, 0.0000000000007647166320f,
    0.00000000000004779478950f, 0.000000000000002811457147f,
    0.0000000000000001561918299f};
constant float THOMPSON_AA_CCE2[16] = {
    0.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
    12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f};

static inline float thompson_aa_add(float a, float b) {
    return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)));
}

static inline float thompson_aa_mul(float a, float b) {
    return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b)));
}

static inline float thompson_aa_sub(float a, float b) {
    return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b)));
}

static inline float thompson_aa_div(float a, float b) {
    return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)));
}

static inline int thompson_aa_nu_c(float nc_m3) {
    const float quotient = thompson_aa_div(1000.0e6f, nc_m3);
    const int rounded = int(floor(thompson_aa_add(quotient, 0.5f)));
    return min(15, rounded + 2);
}

static inline float thompson_aa_cloud_distribution(
        float rc, float nc_per_kg, float rho, thread int &nu_c_entry) {
    float nc = fmax(2.0f, fmin(
        thompson_aa_mul(nc_per_kg, rho), 1999.0e6f));
    const int nu_c = thompson_aa_nu_c(nc);
    nu_c_entry = nu_c;

    float lambda_argument = thompson_aa_mul(nc, 5.235988159e2f);
    lambda_argument = thompson_aa_mul(lambda_argument,
                                       THOMPSON_AA_CCG2[nu_c]);
    lambda_argument = thompson_aa_mul(lambda_argument,
                                       THOMPSON_AA_OCG1[nu_c]);
    lambda_argument = thompson_aa_div(lambda_argument, rc);
    const float lambda_f = as_type<float>(soft_gfk_pow_bits(
        as_type<uint>(lambda_argument), as_type<uint>(3.333333433e-1f)));
    ulong lambda = soft64_from_f32_bits(as_type<uint>(lambda_f));

    const float diameter_numerator = thompson_aa_add(
        thompson_aa_add(3.0f, float(nu_c)), 1.0f);
    const float diameter = as_type<float>(soft64_to_f32_bits(soft64_div(
        soft64_from_f32_bits(as_type<uint>(diameter_numerator)), lambda)));
    if (diameter < 1.0e-6f) {
        lambda = soft64_from_f32_bits(as_type<uint>(
            thompson_aa_div(THOMPSON_AA_CCE2[nu_c], 1.0e-6f)));
    } else if (diameter > 1.0e-4f) {
        lambda = soft64_from_f32_bits(as_type<uint>(
            thompson_aa_div(THOMPSON_AA_CCE2[nu_c], 1.0e-4f)));
    }

    float prefactor = thompson_aa_mul(THOMPSON_AA_CCG1[nu_c],
                                      THOMPSON_AA_OCG2[nu_c]);
    prefactor = thompson_aa_mul(prefactor, rc);
    prefactor = thompson_aa_div(prefactor, 5.235988159e2f);
    const ulong lambda2 = soft64_mul(lambda, lambda);
    const ulong lambda3 = soft64_mul(lambda2, lambda);
    ulong diagnosed = soft64_mul(
        soft64_from_f32_bits(as_type<uint>(prefactor)), lambda3);
    const ulong cap = soft64_from_f32_bits(as_type<uint>(1999.0e6f));
    if (soft64_greater(diagnosed, cap)) diagnosed = cap;
    return as_type<float>(soft64_to_f32_bits(diagnosed));
}

static inline float thompson_aa_terminal_cloud_number(
        float qc, float nc_new, float rho, thread int &nu_c) {
    nu_c = thompson_aa_nu_c(thompson_aa_mul(nc_new, rho));
    float lambda_argument = thompson_aa_mul(5.235988159e2f,
                                             THOMPSON_AA_CCG2[nu_c]);
    lambda_argument = thompson_aa_mul(lambda_argument,
                                       THOMPSON_AA_OCG1[nu_c]);
    lambda_argument = thompson_aa_mul(lambda_argument, nc_new);
    lambda_argument = thompson_aa_div(lambda_argument, qc);
    const float lambda_f = as_type<float>(soft_gfk_pow_bits(
        as_type<uint>(lambda_argument), as_type<uint>(3.333333433e-1f)));
    ulong lambda = soft64_from_f32_bits(as_type<uint>(lambda_f));

    const float diameter_numerator = thompson_aa_add(
        thompson_aa_add(3.0f, float(nu_c)), 1.0f);
    const float diameter = as_type<float>(soft64_to_f32_bits(soft64_div(
        soft64_from_f32_bits(as_type<uint>(diameter_numerator)), lambda)));
    if (diameter < 1.0e-6f) {
        lambda = soft64_from_f32_bits(as_type<uint>(
            thompson_aa_div(THOMPSON_AA_CCE2[nu_c], 1.0e-6f)));
    } else if (diameter > 1.0e-4f) {
        lambda = soft64_from_f32_bits(as_type<uint>(
            thompson_aa_div(THOMPSON_AA_CCE2[nu_c], 1.0e-4f)));
    }

    float prefactor = thompson_aa_mul(THOMPSON_AA_CCG1[nu_c],
                                      THOMPSON_AA_OCG2[nu_c]);
    prefactor = thompson_aa_mul(prefactor, qc);
    prefactor = thompson_aa_div(prefactor, 5.235988159e2f);
    const ulong lambda2 = soft64_mul(lambda, lambda);
    const ulong lambda3 = soft64_mul(lambda2, lambda);
    const ulong product = soft64_mul(
        soft64_from_f32_bits(as_type<uint>(prefactor)), lambda3);
    const ulong cap = soft64_div(
        soft64_from_f32_bits(as_type<uint>(1999.0e6f)),
        soft64_from_f32_bits(as_type<uint>(rho)));
    const ulong diagnosed = soft64_less(product, cap) ? product : cap;
    return as_type<float>(soft64_to_f32_bits(diagnosed));
}

// module_mp_thompson.F:5378-5446. Preserve the default-REAL Horner tree and
// round each operation separately, as in the source helper used by the CUDA
// column classifier.
static inline float thompson_aa_saturation_mixing_ratio(
        float pressure, float temperature, bool over_ice) {
    const float x = fmax(-80.0f, thompson_aa_sub(temperature, 273.16f));
    float es = over_ice ? 0.161444444e-12f : -0.321582393e-13f;
    es = thompson_aa_add(over_ice ? 0.105785160e-9f : 0.379534310e-11f,
                         thompson_aa_mul(x, es));
    es = thompson_aa_add(over_ice ? 0.307839583e-7f : 0.702620698e-8f,
                         thompson_aa_mul(x, es));
    es = thompson_aa_add(over_ice ? 0.521693933e-5f : 0.203154182e-5f,
                         thompson_aa_mul(x, es));
    es = thompson_aa_add(over_ice ? 0.565392987e-3f : 0.299291081e-3f,
                         thompson_aa_mul(x, es));
    es = thompson_aa_add(over_ice ? 0.402737184e-1f : 0.264224321e-1f,
                         thompson_aa_mul(x, es));
    es = thompson_aa_add(over_ice ? 0.184672631e1f : 0.143177157e1f,
                         thompson_aa_mul(x, es));
    es = thompson_aa_add(over_ice ? 0.499320233e2f : 0.444606896e2f,
                         thompson_aa_mul(x, es));
    es = thompson_aa_add(over_ice ? 0.609868993e3f : 0.611583699e3f,
                         thompson_aa_mul(x, es));
    es = fmin(es, thompson_aa_mul(pressure, 0.15f));
    const float denominator = over_ice
        ? fmax(1.0e-4f, thompson_aa_sub(pressure, es))
        : thompson_aa_sub(pressure, es);
    return thompson_aa_div(thompson_aa_mul(0.622f, es), denominator);
}

// module_mp_thompson.F:1802 / thompson_aerosol_state.cu helper.
// Keep the source's float operation order and do not contract adjacent work.
static inline float thompson_aa_density_msl(float pressure,
                                             float temperature, float qv) {
    const float numerator = 0.622f * pressure;
    const float qmix = qv + 0.622f;
    const float denominator = (287.04f * temperature) * qmix;
    return numerator / denominator;
}

kernel void thompson_aa_entry_snapshot(
    constant ThompsonAaEntrySnapshotResources &r [[buffer(0)]],
    constant ThompsonAaEntrySnapshotParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;

    const float qv_local = fmax(1.0e-10f, r.qv[tid]);
    const float rho = thompson_aa_density_msl(
        r.pressure[tid], r.temperature[tid], qv_local);
    r.rho_out[tid] = rho;

    const float nwfa_m3 = r.nwfa[tid] * rho;
    const float nifa_m3 = r.nifa[tid] * rho;
    r.nwfa_entry_m3[tid] = fmax(11.1e6f, fmin(9999.0e6f, nwfa_m3));
    r.nifa_entry_m3[tid] = fmax(5.0e3f, fmin(9999.0e6f, nifa_m3));
}

kernel void thompson_aa_tau1_density(
    constant ThompsonAaTau1DensityResources &r [[buffer(0)]],
    constant ThompsonAaTau1DensityParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;

    const float qv_local = fmax(1.0e-10f, r.qv[tid]);
    r.rho_out[tid] = thompson_aa_density_msl(
        r.pressure[tid], r.temperature[tid], qv_local);
}

// module_mp_thompson.F:1310-1327, after mp_thompson returns. The scalar
// tendency is multiplied by dt before addition, with each FP32 operation
// rounded separately. Only the lowest model level is updated and WRF does
// not apply the terminal aerosol ceiling again here.
kernel void thompson_aa_surface_emission(
    constant ThompsonAaSurfaceEmissionResources &r [[buffer(0)]],
    constant ThompsonAaSurfaceEmissionParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncolumns)) return;
    const float nwfa_source = thompson_aa_mul(r.nwfa2d[col], p.dt);
    const float nifa_source = thompson_aa_mul(r.nifa2d[col], p.dt);
    r.nwfa[col] = thompson_aa_add(r.nwfa[col], nwfa_source);
    r.nifa[col] = thompson_aa_add(r.nifa[col], nifa_source);
}

// module_mp_thompson.F:3211. This is the working aerosol number used by
// activation: update the read-only per-kg entry number by its accumulated
// tendency, convert with the source's TAU+1 density, then apply only the
// 11.1e6 lower bound (there is no upper clamp at this point).
kernel void thompson_aa_working_number(
    constant ThompsonAaWorkingNumberResources &r [[buffer(0)]],
    constant ThompsonAaWorkingNumberParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    const float tendency = thompson_aa_mul(r.nwfaten[tid], p.dt);
    const float nwfa_kg = thompson_aa_add(r.nwfa[tid], tendency);
    const float nwfa_m3 = thompson_aa_mul(nwfa_kg, r.rho[tid]);
    r.nwfa_work_m3[tid] = (isnan(nwfa_m3) || nwfa_m3 < 11.1e6f)
        ? 11.1e6f : nwfa_m3;
}

// module_mp_thompson.F:1646, 1827-2020. A column is live when any entry
// condensate exceeds R1 or a level is supersaturated over ice. The output is
// one float flag per horizontal column; all inputs are read-only.
kernel void thompson_aa_micro_columns(
    constant ThompsonAaMicroColumnsResources &r [[buffer(0)]],
    constant ThompsonAaMicroColumnsParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= uint(p.ncol)) return;
    float micro = 0.0f;
    for (int k = 0; k < p.nz; ++k) {
        const uint idx = uint(k) * uint(p.ncol) + column;
        if (r.qc[idx] > 1.0e-12f || r.qi[idx] > 1.0e-12f
                || r.qr[idx] > 1.0e-12f || r.qs[idx] > 1.0e-12f
                || r.qg[idx] > 1.0e-12f) {
            micro = 1.0f;
            break;
        }
        const float temperature = r.temperature[idx];
        const float qv = fmax(1.0e-10f, r.qv[idx]);
        const float qv_sat = thompson_aa_saturation_mixing_ratio(
            r.pressure[idx], temperature,
            thompson_aa_sub(temperature, 273.15f) <= 0.0f);
        float supersaturation = thompson_aa_sub(
            thompson_aa_div(qv, qv_sat), 1.0f);
        if (fabs(supersaturation) < 1.0e-15f) supersaturation = 0.0f;
        if (supersaturation > 0.0f) {
            micro = 1.0f;
            break;
        }
    }
    r.micro_columns[column] = micro;
}

// module_mp_thompson.F:1826-1848. Entry cloud number diagnosis, including
// the source dry-level rewrite of qc/nc and the WRF cloud-size distribution.
kernel void thompson_aa_entry_cloud_number(
    constant ThompsonAaEntryCloudNumberResources &r [[buffer(0)]],
    constant ThompsonAaEntryCloudNumberParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    const float qc = r.qc[tid];
    const float rho = r.rho[tid];
    if (qc > 1.0e-12f) {
        const float rc = thompson_aa_mul(qc, rho);
        int nu_c = 0;
        const float nc = thompson_aa_cloud_distribution(
            rc, r.nc[tid], rho, nu_c);
        r.rc_out[tid] = rc;
        r.nc_entry_m3[tid] = nc;
        r.nu_c_out[tid] = nu_c;
        r.l_qc_out[tid] = 1;
    } else {
        r.qc[tid] = 0.0f;
        r.nc[tid] = 0.0f;
        r.rc_out[tid] = 1.0e-12f;
        r.nc_entry_m3[tid] = 2.0f;
        r.nu_c_out[tid] = thompson_aa_nu_c(2.0f);
        r.l_qc_out[tid] = 0;
    }
}

// module_mp_thompson.F:3972-4021, with the :2020 column return and :3974
// vapour floor. In inactive columns the source returns before accumulator
// application; active columns update aerosol state, then apply the terminal
// cloud-number rediagnosis.
kernel void thompson_aa_state_finalize_with_columns(
    constant ThompsonAaStateFinalizeWithColumnsResources &r [[buffer(0)]],
    constant ThompsonAaStateFinalizeWithColumnsParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    const uint column = tid % uint(p.ncol);
    if (r.micro_columns[column] == 0.0f) {
        r.nc_out[tid] = r.nc[tid];
        r.nwfa_out[tid] = r.nwfa[tid];
        r.nifa_out[tid] = r.nifa[tid];
        return;
    }

    r.qv[tid] = fmax(1.0e-10f, r.qv[tid]);
    const float rho = r.rho[tid];
    const float nc_accumulated = thompson_aa_add(
        r.nc[tid], thompson_aa_mul(r.ncten[tid], p.dt));
    float nc_new = fmax(thompson_aa_div(2.0f, rho),
                        fmin(nc_accumulated, 1999.0e6f));

    const float nwfa_updated = thompson_aa_add(
        r.nwfa[tid], thompson_aa_mul(r.nwfaten[tid], p.dt));
    const float nifa_updated = thompson_aa_add(
        r.nifa[tid], thompson_aa_mul(r.nifaten[tid], p.dt));
    r.nwfa_out[tid] = fmax(11.1e6f, fmin(nwfa_updated, 9999.0e6f));
    r.nifa_out[tid] = fmax(5.0e3f, fmin(nifa_updated, 9999.0e6f));

    const float qc = r.qc[tid];
    if (qc <= 1.0e-12f) {
        r.qc[tid] = 0.0f;
        r.nc_out[tid] = 0.0f;
        return;
    }
    int nu_c = 0;
    nc_new = thompson_aa_terminal_cloud_number(qc, nc_new, rho, nu_c);
    r.nc_out[tid] = nc_new;
}

// calc_effectRad (:5624-5695), followed by mp_gt_driver's final radius
// bounds (:1476-1478) and gpuwm's metre-to-micron convention.
kernel void thompson_aa_effective_radius(
    constant ThompsonAaEffectiveRadiusResources &r [[buffer(0)]],
    constant ThompsonAaEffectiveRadiusParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;

    constexpr float R1 = 1.0e-12f;
    constexpr float R2 = 1.0e-6f;
    const float rho = thompson_aa_effrad_density(
        r.pressure[tid], r.temperature[tid], r.qv[tid]);
    const float rc = fmax(R1, thompson_aa_mul(r.qc[tid], rho));
    const float nc_m3 = thompson_aa_effrad_clamp_nc(
        thompson_aa_mul(r.nc[tid], rho));
    const float ri = fmax(R1, thompson_aa_mul(r.qi[tid], rho));
    const float ni_m3 = fmax(R2, thompson_aa_mul(r.ni[tid], rho));
    const float rs = fmax(R1, thompson_aa_mul(r.qs[tid], rho));

    float reqc = 2.49e-6f;
    float reqi = 4.99e-6f;
    float reqs = 9.99e-6f;
    if (rc > R1 && nc_m3 > R2)
        reqc = thompson_aa_effrad_cloud(rc, nc_m3);
    if (ri > R1 && ni_m3 > R2)
        reqi = thompson_aa_effrad_ice(ri, ni_m3);
    if (rs > R1)
        reqs = thompson_aa_effrad_snow(rs, r.temperature[tid]);

    r.effc[tid] = fmax(2.49e-6f, fmin(reqc, 50.0e-6f)) * 1.0e6f;
    r.effi[tid] = fmax(4.99e-6f, fmin(reqi, 125.0e-6f)) * 1.0e6f;
    r.effs[tid] = fmax(9.99e-6f, fmin(reqs, 999.0e-6f)) * 1.0e6f;
}

// thompson_init's synthetic CCN/IN profile fill. The transcendental calls
// use Metal's native float overloads; surrounding REAL(4) operations retain
// their source order and are rounded with the local soft32 arithmetic.
kernel void thompson_aa_init_profile(
    constant ThompsonAaInitProfileResources &r [[buffer(0)]],
    constant ThompsonAaInitProfileParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncolumns)) return;

    const float hgt1 = r.hgt[col];
    const float hgt2 = r.hgt[uint(p.ncolumns) + col];
    float h_01;
    if (hgt1 <= 1000.0f) {
        h_01 = 0.8f;
    } else if (hgt1 >= 2500.0f) {
        h_01 = 0.01f;
    } else {
        const float cosine_argument = thompson_aa_sub(
            thompson_aa_mul(hgt1, 0.001f), 1.0f);
        h_01 = thompson_aa_mul(0.8f, cos(cosine_argument));
    }

    const float z1 = thompson_aa_sub(hgt2, hgt1);
    if (p.fill_ccn != 0) {
        const float niCCN3 = thompson_aa_div(
            thompson_aa_mul(-1.0f, log(thompson_aa_div(
                50.0e6f, 300.0e6f))), h_01);
        const float first = thompson_aa_add(
            50.0e6f,
            thompson_aa_mul(300.0e6f, exp(thompson_aa_mul(
                -thompson_aa_div(z1, 1000.0f), niCCN3))));
        r.nwfa[col] = first;
        r.nwfa2d[col] = thompson_aa_mul(
            thompson_aa_mul(first, 0.000196f),
            thompson_aa_div(50.0f, z1));
        for (int k = 1; k < p.nz; ++k) {
            const uint index = uint(k) * uint(p.ncolumns) + col;
            const float dz = thompson_aa_sub(r.hgt[index], hgt1);
            r.nwfa[index] = thompson_aa_add(
                50.0e6f,
                thompson_aa_mul(300.0e6f, exp(thompson_aa_mul(
                    -thompson_aa_div(dz, 1000.0f), niCCN3))));
        }
    }

    if (p.fill_in != 0) {
        const float niIN3 = thompson_aa_div(
            thompson_aa_mul(-1.0f, log(thompson_aa_div(
                0.5e6f, 1.5e6f))), h_01);
        r.nifa[col] = thompson_aa_add(
            0.5e6f,
            thompson_aa_mul(1.5e6f, exp(thompson_aa_mul(
                -thompson_aa_div(z1, 1000.0f), niIN3))));
        for (int k = 1; k < p.nz; ++k) {
            const uint index = uint(k) * uint(p.ncolumns) + col;
            const float dz = thompson_aa_sub(r.hgt[index], hgt1);
            r.nifa[index] = thompson_aa_add(
                0.5e6f,
                thompson_aa_mul(1.5e6f, exp(thompson_aa_mul(
                    -thompson_aa_div(dz, 1000.0f), niIN3))));
        }
    }
}

// module_mp_thompson.F:3213-3221 and :3484-3488. Refresh the cloud working
// values from the immutable entry fields plus their accumulated tendencies.
kernel void thompson_aa_working_cloud(
    constant ThompsonAaWorkingCloudResources &r [[buffer(0)]],
    constant ThompsonAaWorkingCloudParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;

    constexpr float R1 = 1.0e-12f;
    constexpr float NC_FLOOR = 2.0f;
    constexpr float NC_MAX = 1999.0e6f;
    const float qc_updated = thompson_aa_add(
        r.qc[tid], thompson_aa_mul(r.qcten[tid], p.dt));
    if (qc_updated > R1) {
        r.rc_work[tid] = thompson_aa_mul(qc_updated, r.rho[tid]);
        const float nc_updated = thompson_aa_add(
            r.nc[tid], thompson_aa_mul(r.ncten[tid], p.dt));
        const float nc_m3 = thompson_aa_mul(nc_updated, r.rho[tid]);
        r.nc_work_m3[tid] = fmax(NC_FLOOR, fmin(nc_m3, NC_MAX));
        r.l_qc_out[tid] = 1;
    } else {
        r.rc_work[tid] = R1;
        r.nc_work_m3[tid] = NC_FLOOR;
        r.l_qc_out[tid] = 0;
    }
}

// WRF's domain-wide MAXVAL test that precedes thompson_init's CCN/IN
// profile fills. Initialization needs one boolean per aerosol field; a
// single GPU lane performs the reduction so the state caller transfers only
// the one-word decision, never the field itself.
kernel void thompson_aa_profile_needs_fill(
    constant ThompsonAaProfileNeedsFillResources &r [[buffer(0)]],
    constant ThompsonAaProfileNeedsFillParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    bool below_threshold = true;
    for (int index = 0; index < p.size; ++index) {
        if (!(r.field[index] < p.threshold)) {
            below_threshold = false;
            break;
        }
    }
    r.needs_fill[0] = below_threshold ? 1 : 0;
}

// Build z8w=(phb+php)/g for thompson_init's profile fill. The caller uses a
// contiguous view of z8w[:nz], avoiding unsupported MetalArray slicing.
kernel void thompson_aa_profile_height(
    constant ThompsonAaProfileHeightResources &r [[buffer(0)]],
    constant ThompsonAaProfileHeightParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint size = uint(p.nz + 1) * uint(p.ny) * uint(p.nx);
    if (tid >= size) return;
    const uint columns = uint(p.ny) * uint(p.nx);
    const uint k = tid / columns;
    const float base = r.phb[p.phb_full != 0 ? tid : k];
    r.z8w[tid] = thompson_aa_div(
        thompson_aa_add(base, r.php[tid]), p.gravity);
}

// mp_gt_driver:1308's MP28 diagnostic. Preserve its explicit epsilon
// quotient: there is no small-rain guard and no MIN clamp in WRF here.
kernel void thompson_aa_surface_rain_fraction(
    constant ThompsonAaSurfaceRainFractionResources &r [[buffer(0)]],
    constant ThompsonAaSurfaceRainFractionParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncolumns)) return;
    const float frozen = thompson_aa_add(
        r.snowncv[tid], r.graupelncv[tid]);
    const float denominator = thompson_aa_add(r.rainncv[tid], 1.0e-12f);
    r.sr[tid] = thompson_aa_div(frozen, denominator);
}

// NOAA-EMC WRF 3.9 module_mp_thompson.F:3755. The fork carries no
// prognostic graupel number and clears trace graupel during terminal apply.
kernel void thompson_aa_wrf39_graupel_finalize(
    constant ThompsonAaWrf39GraupelFinalizeResources &r [[buffer(0)]],
    constant ThompsonAaWrf39GraupelFinalizeParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    if (r.qg[tid] <= p.r1) r.qg[tid] = 0.0f;
}

// NOAA-EMC WRF 3.9 module_mp_thompson.F:587-603. The fork initializes
// surface CCN emission from the analyzed lowest-level CCN profile.
kernel void thompson_aa_wrf39_start_emission(
    constant ThompsonAaWrf39StartEmissionResources &r [[buffer(0)]],
    constant ThompsonAaWrf39StartEmissionParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncolumns)) return;
    const float spacing = sqrt(thompson_aa_mul(p.dx, p.dy));
    const float relative_spacing = thompson_aa_div(spacing, 20000.0f);
    float scale = 0.875f;
    if (relative_spacing < 1.0f) {
        scale = thompson_aa_mul(
            thompson_aa_add(0.875f, thompson_aa_mul(0.125f,
                thompson_aa_div(thompson_aa_sub(20000.0f, spacing),
                                16000.0f))), relative_spacing);
    }
    const float number = r.nwfa[tid];
    if (!(number > 0.0f)) {
        r.nwfa2d[tid] = 0.0f;
        return;
    }
    const float exponent = thompson_aa_sub(
        log10(thompson_aa_mul(number, 1.0e-6f)), 3.69897f);
    const float emission = thompson_aa_effrad_powf(10.0f, exponent);
    r.nwfa2d[tid] = thompson_aa_mul(
        thompson_aa_mul(emission, scale), 1.0e6f);
}

// Fork module_mp_thompson.F:5246-5315 calc_effectRad bounds. This shares
// the WRF-derived distribution functions above but applies NOAA WRF39's
// uncapped cloud number and its larger ice/snow effective-radius floors.
kernel void thompson_aa_wrf39_effective_radius(
    constant ThompsonAaEffectiveRadiusResources &r [[buffer(0)]],
    constant ThompsonAaEffectiveRadiusParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    constexpr float R1 = 1.0e-12f;
    constexpr float R2 = 1.0e-6f;
    const float rho = thompson_aa_effrad_density(
        r.pressure[tid], r.temperature[tid], r.qv[tid]);
    const float rc = fmax(R1, thompson_aa_mul(r.qc[tid], rho));
    const float nc_m3 = fmax(R2, thompson_aa_mul(r.nc[tid], rho));
    const float ri = fmax(R1, thompson_aa_mul(r.qi[tid], rho));
    const float ni_m3 = fmax(R2, thompson_aa_mul(r.ni[tid], rho));
    const float rs = fmax(R1, thompson_aa_mul(r.qs[tid], rho));
    float reqc = 2.49e-6f;
    float reqi = 4.99e-6f;
    float reqs = 9.99e-6f;
    if (rc > R1 && nc_m3 > R2)
        reqc = thompson_aa_effrad_cloud(rc, nc_m3);
    if (ri > R1 && ni_m3 > R2)
        reqi = fmax(5.01e-6f, thompson_aa_effrad_ice(ri, ni_m3));
    if (rs > R1)
        reqs = fmax(10.0e-6f,
                    thompson_aa_effrad_snow(rs, r.temperature[tid]));
    r.effc[tid] = fmax(2.49e-6f, fmin(reqc, 50.0e-6f)) * 1.0e6f;
    r.effi[tid] = fmax(4.99e-6f, fmin(reqi, 125.0e-6f)) * 1.0e6f;
    r.effs[tid] = fmax(9.99e-6f, fmin(reqs, 999.0e-6f)) * 1.0e6f;
}

static inline ulong thompson_aa_wrf39_log10(float x) {
    const ulong value = soft64_from_f32_bits(as_type<uint>(float(x)));
    const ulong ten = soft64_from_f32_bits(as_type<uint>(10.0f));
    return soft64_div(soft64_plm_log(value), soft64_plm_log(ten));
}

// NOAA-EMC WRF 3.9 fork :2031-2054/:3110-3133/:5486-5510. One thread
// owns a column and performs the fork's top-down running-minimum pass.
kernel void thompson_aa_wrf39_graupel_intercept(
    constant ThompsonAaWrf39GraupelInterceptResources &r [[buffer(0)]],
    constant ThompsonAaWrf39GraupelInterceptParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= uint(p.ny * p.nx)) return;
    const uint j = column / uint(p.nx);
    const uint i = column - j * uint(p.nx);
    int k0 = 0;
    for (int k = p.nz - 1; k >= 0; --k) {
        const uint index = (uint(k) * uint(p.ny) + j) * uint(p.nx) + i;
        if (r.temperature[index] >= 270.65f) k0 = max(k0, k);
    }
    const float am_g = thompson_aa_div(
        thompson_aa_mul(p.pi, p.rho_g), 6.0f);
    float n0_min = p.gonv_max;
    for (int k = p.nz - 1; k >= 0; --k) {
        const uint index = (uint(k) * uint(p.ny) + j) * uint(p.nx) + i;
        const float qv = fmax(p.qv_floor, r.qv[index]);
        const float rho = thompson_aa_density_msl(
            r.pressure[index], r.temperature[index], qv);
        const bool has_rain = r.qr[index] > p.r1;
        float mvd_r = 0.0f;
        if (has_rain) {
            const float rain_mass = thompson_aa_mul(r.qr[index], rho);
            float rain_number = fmax(p.r2,
                thompson_aa_mul(r.nr[index], rho));
            if (p.mode == 1 && rain_number <= p.r2) {
                const ulong lamr_seed = soft64_from_f32_bits(as_type<uint>(
                    thompson_aa_div(p.mvd_numerator, 1.0e-3f)));
                const ulong lamr3 = soft64_mul(soft64_mul(lamr_seed, lamr_seed),
                                               lamr_seed);
                const ulong seeded = soft64_div(soft64_mul(
                    soft64_from_f32_bits(as_type<uint>(thompson_aa_mul(
                        0.16666667163372040f, rain_mass))), lamr3),
                    soft64_from_f32_bits(as_type<uint>(p.am_r)));
                rain_number = as_type<float>(soft64_to_f32_bits(seeded));
            }
            float lambda_base = thompson_aa_mul(p.am_r, 6.0f);
            lambda_base = thompson_aa_mul(lambda_base, rain_number);
            lambda_base = thompson_aa_div(lambda_base, rain_mass);
            const float lambda_r = thompson_aa_effrad_powf(
                lambda_base, p.obmr);
            if (p.mode == 2) {
                const ulong mvd = soft64_mul(
                    soft64_from_f32_bits(as_type<uint>(p.mvd_numerator)),
                    soft64_div(soft64_from_f32_bits(as_type<uint>(1.0f)),
                               soft64_from_f32_bits(as_type<uint>(lambda_r))));
                mvd_r = as_type<float>(soft64_to_f32_bits(mvd));
            } else {
                const ulong mvd = soft64_div(
                    soft64_from_f32_bits(as_type<uint>(p.mvd_numerator)),
                    soft64_from_f32_bits(as_type<uint>(lambda_r)));
                mvd_r = as_type<float>(soft64_to_f32_bits(mvd));
                if (mvd_r > 2.5e-3f) mvd_r = 2.5e-3f;
                else if (mvd_r < thompson_aa_mul(p.d0r, 0.75f))
                    mvd_r = thompson_aa_mul(p.d0r, 0.75f);
            }
        }
        const float graupel_floor = p.mode == 2 ? p.r2 : p.r1;
        const float rg = r.qg[index] > graupel_floor
            ? thompson_aa_mul(r.qg[index], rho) : p.r1;
        float xslw1 = 0.01f;
        if (k > k0 && has_rain && mvd_r > 100.0e-6f) {
            const float log_mvd = as_type<float>(soft64_to_f32_bits(
            thompson_aa_wrf39_log10(mvd_r)));
            xslw1 = thompson_aa_add(4.01f, log_mvd);
        }
        const float rg_log_arg = fmax(5.0e-5f, rg);
        const float log_rg = as_type<float>(soft64_to_f32_bits(
            thompson_aa_wrf39_log10(rg_log_arg)));
        const float ygra1 = thompson_aa_add(4.31f, log_rg);
        const float numerator = thompson_aa_mul(
            thompson_aa_mul(300.0f, xslw1), ygra1);
        const float denominator = thompson_aa_add(
            thompson_aa_add(thompson_aa_div(10.0f, xslw1), 1.0f),
            thompson_aa_mul(0.25f, ygra1));
        const float sum = thompson_aa_add(
            thompson_aa_add(thompson_aa_div(numerator, denominator), 30.0f),
            thompson_aa_mul(10.0f, ygra1));
        const float zans1 = thompson_aa_add(
            3.1f, thompson_aa_div(100.0f, sum));
        ulong n0_exp = soft64_plm_pow(
            soft64_from_f32_bits(as_type<uint>(10.0f)),
            soft64_from_f32_bits(as_type<uint>(zans1)));
        const ulong n0_floor = soft64_from_f32_bits(as_type<uint>(p.gonv_min));
        const ulong n0_ceil = soft64_from_f32_bits(as_type<uint>(p.gonv_max));
        if (soft64_less(n0_exp, n0_floor)) n0_exp = n0_floor;
        if (soft64_greater(n0_exp, n0_ceil)) n0_exp = n0_ceil;
        const float n0_level = as_type<float>(soft64_to_f32_bits(n0_exp));
        n0_min = fmin(n0_level, n0_min);
        if (p.mode == 2) {
            ulong slope_num = soft64_mul(
                soft64_from_f32_bits(as_type<uint>(n0_min)),
                soft64_from_f32_bits(as_type<uint>(am_g)));
            slope_num = soft64_mul(slope_num,
                soft64_from_f32_bits(as_type<uint>(6.0f)));
            const ulong slope_base = soft64_div(slope_num,
                soft64_from_f32_bits(as_type<uint>(rg)));
            const ulong lamg = soft64_plm_pow(
                slope_base, soft64_from_f32_bits(as_type<uint>(0.25f)));
            const ulong ilamg = soft64_div(
                soft64_from_f32_bits(as_type<uint>(1.0f)), lamg);
            const ulong n0_g = soft64_mul(
                soft64_div(soft64_from_f32_bits(as_type<uint>(n0_min)), lamg),
                lamg);
            const ulong number_per_kg = soft64_div(
                soft64_mul(n0_g, ilamg),
                soft64_from_f32_bits(as_type<uint>(rho)));
            r.n0_out[index] = as_type<float>(soft64_to_f32_bits(number_per_kg));
        } else {
            r.n0_out[index] = n0_min;
        }
    }
}
