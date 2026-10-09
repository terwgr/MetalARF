#include <metal_stdlib>
#include "thompson_family_bindings.metalh"
#include "soft_binary64.metalh"

using namespace metal;

// Thompson's scalar kernels in this stage use WRF default REAL (FP32).
#pragma clang fp contract(off)

static inline float thompson_rslf(float pressure, float temp) {
    const float c0 = 0.611583699e3f;
    const float c1 = 0.444606896e2f;
    const float c2 = 0.143177157e1f;
    const float c3 = 0.264224321e-1f;
    const float c4 = 0.299291081e-3f;
    const float c5 = 0.203154182e-5f;
    const float c6 = 0.702620698e-8f;
    const float c7 = 0.379534310e-11f;
    const float c8 = -0.321582393e-13f;
    const float x = fmax(-80.0f, temp - 273.16f);
    float esl = c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4
        + x * (c5 + x * (c6 + x * (c7 + x * c8)))))));
    esl = fmin(esl, pressure * 0.15f);
    return 0.622f * esl / (pressure - esl);
}

static inline void thompson_cloud_saturation_adjust_impl(
    device float *temperature, device const float *pressure,
    device float *qv, device float *qc,
    device float *reference_density,
    device float *reference_temperature,
    device float *condensation_marker,
    device float *cloud_presence, uint idx) {
    const float temp0 = temperature[idx];
    const float qv0 = fmax(1.0e-10f, qv[idx]);
    const float qc0 = qc[idx];
    if (condensation_marker != nullptr) condensation_marker[idx] = 0.0f;
    if (cloud_presence != nullptr) {
        cloud_presence[idx] = qc0 > 1.0e-12f ? 1.0f : 0.0f;
    }
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    if (reference_density != nullptr) reference_density[idx] = rho;
    if (reference_temperature != nullptr) reference_temperature[idx] = temp0;

    const float qvs = thompson_rslf(pressure[idx], temp0);
    float ssatw = qv0 / qvs - 1.0f;
    if (fabs(ssatw) < 1.0e-15f) ssatw = 0.0f;
    if (!(ssatw > 1.0e-15f
            || (ssatw < -1.0e-15f && qc0 > 1.0e-12f))) return;

    const float rc = qc0 > 1.0e-12f ? qc0 * rho : 1.0e-12f;
    const float tempc = temp0 - 273.15f;
    const float lvap = 2.5e6f + (2106.0f - 4218.0f) * tempc;
    const float ocp = 1.0f / (1004.0f * (1.0f + 0.887f * qv0));
    const float inverse_temp = 1.0f / temp0;
    const float lvt2 = lvap * lvap * ocp * (1.0f / 461.5f)
        * inverse_temp * inverse_temp;
    float clap = (qv0 - qvs) / (1.0f + lvt2 * qvs);
    for (int iteration = 0; iteration < 3; ++iteration) {
        const float exponential = exp(lvt2 * clap);
        const float fcd = qvs * exponential - qv0 + clap;
        const float dfcd = qvs * lvt2 * exponential + 1.0f;
        clap -= fcd / dfcd;
    }
    if (rc + clap * rho <= 1.0e-12f) clap = -rc / rho;
    if (condensation_marker != nullptr) {
        condensation_marker[idx] = clap > 0.0f ? 1.0f : 0.0f;
    }
    qv[idx] = qv[idx] - clap;
    qc[idx] = fmax(0.0f, qc0 + clap);
    if (cloud_presence != nullptr && !(qc[idx] * rho > 1.0e-12f)) {
        cloud_presence[idx] = 0.0f;
    }
    temperature[idx] = temp0 + lvap * ocp * clap;
}

kernel void thompson_warm_saturation_adjust(
    constant ThompsonWarmSaturationAdjustResources &r [[buffer(0)]],
    constant ThompsonWarmSaturationAdjustParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    const float temp0 = r.temperature[idx];
    const float qv0 = r.qv[idx];
    const float qc0 = r.qc[idx];
    if (temp0 <= 273.15f) return;
    const float qvs = thompson_rslf(r.pressure[idx], temp0);
    float ssatw = qv0 / qvs - 1.0f;
    if (fabs(ssatw) < 1.0e-15f) ssatw = 0.0f;
    if (!(ssatw > 1.0e-15f
            || (ssatw < -1.0e-15f && qc0 > 1.0e-12f))) return;
    const float rho = 0.622f * r.pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float rc = qc0 > 1.0e-12f ? qc0 * rho : 1.0e-12f;
    const float tempc = temp0 - 273.15f;
    const float lvap = 2.5e6f + (2106.0f - 4218.0f) * tempc;
    const float ocp = 1.0f / (1004.0f * (1.0f + 0.887f * qv0));
    const float inv_temp = 1.0f / temp0;
    const float lvt2 = lvap * lvap * ocp * (1.0f / 461.5f)
        * inv_temp * inv_temp;
    float clap = (qv0 - qvs) / (1.0f + lvt2 * qvs);
    for (int iteration = 0; iteration < 3; ++iteration) {
        const float exponential = exp(lvt2 * clap);
        const float fcd = qvs * exponential - qv0 + clap;
        const float dfcd = qvs * lvt2 * exponential + 1.0f;
        clap -= fcd / dfcd;
    }
    if (rc + clap * rho <= 1.0e-12f) clap = -rc / rho;
    r.qv[idx] = fmax(1.0e-10f, qv0 - clap);
    r.qc[idx] = fmax(0.0f, qc0 + clap);
    r.temperature[idx] = temp0 + lvap * ocp * clap;
}

kernel void thompson_exner_power(
    constant ThompsonExnerPowerResources &r [[buffer(0)]],
    constant ThompsonExnerPowerParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.size)) return;
    r.exner[idx] = precise::pow(r.pressure[idx] / p.reference_pressure,
                                p.exponent);
}

#define THOMPSON_CLOUD_KERNEL(NAME, TYPE, RD, RT, CM, CP) \
kernel void NAME(constant TYPE##Resources &r [[buffer(0)]], \
                 constant TYPE##Params &p [[buffer(1)]], \
                 uint idx [[thread_position_in_grid]]) { \
    if (idx >= uint(p.n)) return; \
    thompson_cloud_saturation_adjust_impl( \
        r.temperature, r.pressure, r.qv, r.qc, RD, RT, CM, CP, idx); \
}

THOMPSON_CLOUD_KERNEL(thompson_cloud_saturation_adjust,
                      ThompsonCloudSaturationAdjust,
                      nullptr, nullptr, nullptr, nullptr)
THOMPSON_CLOUD_KERNEL(thompson_cloud_saturation_adjust_with_density,
                      ThompsonCloudSaturationAdjustWithDensity,
                      r.reference_density, nullptr, nullptr, nullptr)
THOMPSON_CLOUD_KERNEL(thompson_cloud_saturation_adjust_with_state,
                      ThompsonCloudSaturationAdjustWithState,
                      r.reference_density, r.reference_temperature,
                      nullptr, nullptr)
THOMPSON_CLOUD_KERNEL(thompson_cloud_saturation_adjust_with_history,
                      ThompsonCloudSaturationAdjustWithHistory,
                      r.reference_density, r.reference_temperature,
                      r.condensation_marker, nullptr)
THOMPSON_CLOUD_KERNEL(thompson_cloud_saturation_adjust_history_marker_only,
                      ThompsonCloudSaturationAdjustHistoryMarkerOnly,
                      nullptr, nullptr, r.condensation_marker, nullptr)
THOMPSON_CLOUD_KERNEL(thompson_cloud_saturation_adjust_history_density_marker,
                      ThompsonCloudSaturationAdjustHistoryDensityMarker,
                      r.reference_density, nullptr,
                      r.condensation_marker, nullptr)
THOMPSON_CLOUD_KERNEL(thompson_cloud_saturation_adjust_with_presence,
                      ThompsonCloudSaturationAdjustWithPresence,
                      r.reference_density, r.reference_temperature,
                      r.condensation_marker, r.cloud_presence)

#undef THOMPSON_CLOUD_KERNEL

// CUDA's classic graupel wrapper deliberately mixes FP32 state/libm
// operations with binary64 lambda and moment arithmetic. Keep that boundary
// explicit here: precise Metal pow/log10 retain FP32 arguments and results,
// but are not asserted to be bit-identical to CUDA's device libm. The
// binary64 portions are represented as IEEE bit words and use the shared
// integer-only soft64 operations.
static inline ulong thompson_soft64_from_f32(float value) {
    return soft64_from_f32_bits(as_type<uint>(value));
}

static inline float thompson_soft64_to_f32(ulong value) {
    return as_type<float>(soft64_to_f32_bits(value));
}

static inline float thompson_classic_graupel_number(
    device const float *qg, device const float *temperature,
    device const float *pressure, device const float *qv, uint idx) {
    if (qg[idx] <= 1.0e-12f) return 0.0f;

    const float qvk = fmax(1.0e-10f, qv[idx]);
    const float rho = 0.622f * pressure[idx]
        / (287.04f * temperature[idx] * (qvk + 0.622f));
    const float rg = qg[idx] * rho;
    const float pi = 3.1415926536f;
    const float am_g = pi * 400.0f / 6.0f;
    const float intercept_power = fmax(2.0f, fmin(
        3.0f + (2.0f / 7.0f) * (precise::log10(fmax(1.0e-9f, rg)) + 8.0f),
        6.0f));
    const float intercept = precise::pow(10.0f, intercept_power);
    const float lambda_base = intercept * am_g * 6.0f / rg;
    const ulong lambda = thompson_soft64_from_f32(
        precise::pow(lambda_base, 0.25f));
    const float sixth_rg = (1.0f / 6.0f) * rg;
    const float moment_division = sixth_rg / am_g;
    ulong number_bits = soft64_mul(thompson_soft64_from_f32(moment_division), lambda);
    number_bits = soft64_mul(number_bits, lambda);
    number_bits = soft64_mul(number_bits, lambda);
    float number = thompson_soft64_to_f32(number_bits);

    const float mvd = thompson_soft64_to_f32(soft64_div(
        0x400d604189374bc7ul, lambda));
    if (mvd > 25.4e-3f) {
        const ulong bounded_lambda = soft64_div(
            0x400d604189374bc7ul, 0x3f9a027525460aa6ul);
        number_bits = soft64_mul(thompson_soft64_from_f32(moment_division),
                                 bounded_lambda);
        number_bits = soft64_mul(number_bits, bounded_lambda);
        number_bits = soft64_mul(number_bits, bounded_lambda);
        number = thompson_soft64_to_f32(number_bits);
    } else if (mvd < 50.0e-6f) {
        const ulong bounded_lambda = soft64_div(
            0x400d604189374bc7ul, 0x3f0a36e2eb1c432dul);
        number_bits = soft64_mul(thompson_soft64_from_f32(moment_division),
                                 bounded_lambda);
        number_bits = soft64_mul(number_bits, bounded_lambda);
        number_bits = soft64_mul(number_bits, bounded_lambda);
        number = thompson_soft64_to_f32(number_bits);
    }
    return number / rho;
}

kernel void thompson_classic_graupel_number_init(
    constant ThompsonClassicGraupelNumberInitResources &r [[buffer(0)]],
    constant ThompsonClassicGraupelNumberInitParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    r.graupel_number_per_kg[idx] = thompson_classic_graupel_number(
        r.qg, r.temperature, r.pressure, r.qv, idx);
}

kernel void thompson_classic_graupel_number_finalize(
    constant ThompsonClassicGraupelNumberFinalizeResources &r [[buffer(0)]],
    constant ThompsonClassicGraupelNumberFinalizeParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    if (r.qg[idx] <= 1.0e-12f) {
        r.qg[idx] = 0.0f;
        r.graupel_number_per_kg[idx] = 0.0f;
        return;
    }

    const float qvk = fmax(1.0e-10f, r.qv[idx]);
    const float rho = 0.622f * r.pressure[idx]
        / (287.04f * r.temperature[idx] * (qvk + 0.622f));
    const float pi = 3.1415926536f;
    const float am_g = pi * 400.0f / 6.0f;
    float number_per_kg = fmax(1.0e-6f / rho,
                                r.graupel_number_per_kg[idx]);
    const float lambda_base = am_g * 6.0f * number_per_kg / r.qg[idx];
    ulong lambda = thompson_soft64_from_f32(
        precise::pow(lambda_base, 1.0f / 3.0f));
    const float mvd = thompson_soft64_to_f32(soft64_div(
        0x400d604189374bc7ul, lambda));
    if (mvd > 25.4e-3f) {
        lambda = soft64_div(0x400d604189374bc7ul, 0x3f9a027525460aa6ul);
        const float sixth_mass = (1.0f / 6.0f) * r.qg[idx];
        const float moment_division = sixth_mass / am_g;
        ulong number_bits = soft64_mul(
            thompson_soft64_from_f32(moment_division), lambda);
        number_bits = soft64_mul(number_bits, lambda);
        number_bits = soft64_mul(number_bits, lambda);
        number_per_kg = thompson_soft64_to_f32(number_bits);
    } else if (mvd < 50.0e-6f) {
        lambda = soft64_div(0x400d604189374bc7ul, 0x3f0a36e2eb1c432dul);
        const float sixth_mass = (1.0f / 6.0f) * r.qg[idx];
        const float moment_division = sixth_mass / am_g;
        ulong number_bits = soft64_mul(
            thompson_soft64_from_f32(moment_division), lambda);
        number_bits = soft64_mul(number_bits, lambda);
        number_bits = soft64_mul(number_bits, lambda);
        number_per_kg = thompson_soft64_to_f32(number_bits);
    }
    r.graupel_number_per_kg[idx] = number_per_kg;
}

static inline float thompson_rsif(float pressure, float temp) {
    const float c0 = 0.609868993e3f;
    const float c1 = 0.499320233e2f;
    const float c2 = 0.184672631e1f;
    const float c3 = 0.402737184e-1f;
    const float c4 = 0.565392987e-3f;
    const float c5 = 0.521693933e-5f;
    const float c6 = 0.307839583e-7f;
    const float c7 = 0.105785160e-9f;
    const float c8 = 0.161444444e-12f;
    const float x = fmax(-80.0f, temp - 273.16f);
    float esi = c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4
        + x * (c5 + x * (c6 + x * (c7 + x * c8)))))));
    esi = fmin(esi, pressure * 0.15f);
    return 0.622f * esi / fmax(1.0e-4f, pressure - esi);
}

kernel void thompson_graupel_fallout_column_mask(
    constant ThompsonGraupelFalloutColumnMaskResources &r [[buffer(0)]],
    constant ThompsonGraupelFalloutColumnMaskParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint columns = uint(p.ny) * uint(p.nx);
    if (column >= columns) return;
    float active = 0.0f;
    for (int k = 0; k < p.nz; ++k) {
        const ulong idx = (ulong(k) * ulong(p.ny) + ulong(column / uint(p.nx)))
                        * ulong(p.nx) + ulong(column % uint(p.nx));
        if (r.entry_active[idx] != 0.0f && r.qg[idx] > 1.0e-12f) {
            active = 1.0f;
            break;
        }
    }
    r.active_columns[column] = active;
}

kernel void thompson_hydrometeor_column_mask(
    constant ThompsonHydrometeorColumnMaskResources &r [[buffer(0)]],
    constant ThompsonHydrometeorColumnMaskParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint columns = uint(p.ny) * uint(p.nx);
    if (column >= columns) return;
    float active = 0.0f;
    const uint j = column / uint(p.nx), i = column % uint(p.nx);
    for (int k = 0; k < p.nz; ++k) {
        const ulong idx = (ulong(k) * ulong(p.ny) + ulong(j)) * ulong(p.nx) + ulong(i);
        if (r.mixing_ratio[idx] > 1.0e-12f) {
            active = 1.0f;
            break;
        }
    }
    r.active_columns[column] = active;
}

static inline bool thompson_level_has_microphysics(
    device const float *qc, device const float *qi,
    device const float *qr, device const float *qs,
    device const float *qg, device const float *temperature,
    device const float *pressure, device const float *qv, ulong idx) {
    if (qc[idx] > 1.0e-12f || qi[idx] > 1.0e-12f || qr[idx] > 1.0e-12f
            || qs[idx] > 1.0e-12f || qg[idx] > 1.0e-12f) return true;
    const float temp = temperature[idx];
    const float qv_local = fmax(1.0e-10f, qv[idx]);
    const float qvsi = temp - 273.15f <= 0.0f
        ? thompson_rsif(pressure[idx], temp)
        : thompson_rslf(pressure[idx], temp);
    float ssati = qv_local / qvsi - 1.0f;
    if (fabs(ssati) < 1.0e-15f) ssati = 0.0f;
    return ssati > 0.0f;
}

kernel void thompson_microphysics_columns(
    constant ThompsonMicrophysicsColumnsResources &r [[buffer(0)]],
    constant ThompsonMicrophysicsColumnsParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint columns = uint(p.ny) * uint(p.nx);
    if (column >= columns) return;
    const uint j = column / uint(p.nx), i = column % uint(p.nx);
    float micro = 0.0f;
    for (int k = 0; k < p.nz; ++k) {
        const ulong idx = (ulong(k) * ulong(p.ny) + ulong(j)) * ulong(p.nx) + ulong(i);
        if (thompson_level_has_microphysics(
                r.qc,r.qi,r.qr,r.qs,r.qg,r.temperature,r.pressure,r.qv,idx)) {
            micro = 1.0f;
            break;
        }
    }
    r.micro_columns[column] = micro;
}

kernel void thompson_adapter_masks(
    constant ThompsonAdapterMasksResources &r [[buffer(0)]],
    constant ThompsonAdapterMasksParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= uint(p.columns)) return;
    float rain = 0.0f, graupel = 0.0f;
    for (int k = 0; k < p.nz && (rain == 0.0f || graupel == 0.0f); ++k) {
        const ulong idx = ulong(k) * ulong(p.columns) + ulong(column);
        if (r.qr[idx] > 1.0e-12f) rain = 1.0f;
        if (r.entry_graupel[idx] != 0.0f && r.qg[idx] > 1.0e-12f) {
            graupel = 1.0f;
        }
    }
    r.rain_columns[column] = rain;
    r.graupel_columns[column] = graupel;
}

kernel void thompson_adapter_finish(
    constant ThompsonAdapterFinishResources &r [[buffer(0)]],
    constant ThompsonAdapterFinishParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.size)) return;
    const float theta = r.temperature[idx] / r.pii[idx];
    r.th[idx] = theta;
    if (!p.no_mp_heating) {
        float increment = theta - r.h_diabatic[idx];
        if (isnan(increment) || isnan(p.limit)) increment = as_type<float>(0x7fffffffu);
        else increment = min(p.limit, increment);
        if (isnan(increment) || isnan(p.limit)) increment = as_type<float>(0x7fffffffu);
        else increment = max(-p.limit, increment);
        r.thp[idx] = r.thp[idx] + increment;
        r.h_diabatic[idx] = increment / p.dt;
    } else {
        r.h_diabatic[idx] = 0.0f;
    }
    if (idx < uint(p.columns)) {
        const float frozen = r.snowncv[idx] + r.graupelncv[idx];
        const float ratio = frozen / r.rainncv[idx];
        const float bounded = isnan(ratio)
            ? as_type<float>(0x7fffffffu) : min(1.0f, ratio);
        r.sr[idx] = r.rainncv[idx] > 1.0e-12f ? bounded : 0.0f;
    }
}

kernel void thompson_adapter_prepare(
    constant ThompsonAdapterPrepareResources &r [[buffer(0)]],
    constant ThompsonAdapterPrepareParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.size)) return;
    const int k = int(idx / uint(p.columns));
    const float theta = r.thb[p.thb_full ? idx : uint(k)] + r.thp[idx];
    const float temp = theta * r.pii[idx];
    r.th[idx] = theta;
    r.temperature[idx] = temp;
    r.saved_theta[idx] = theta;
    const float lower = (r.phb[p.phb_full ? idx : uint(k)] + r.php[idx]) / p.gravity;
    const float upper = (r.phb[p.phb_full ? idx + uint(p.columns) : uint(k + 1)]
                         + r.php[idx + uint(p.columns)]) / p.gravity;
    r.dz[idx] = upper - lower;
    r.entry_graupel[idx] = r.qg[idx] > 1.0e-12f ? 1.0f : 0.0f;
    r.warm_entry[idx] = temp >= 273.15f ? 1.0f : 0.0f;
    if (idx < uint(p.columns)) {
        r.graupelncv[idx] = 0.0f;
        r.micro_columns[idx] = 0.0f;
    }
    if (!(r.qc[idx] > 1.0e-12f)) r.qc[idx] = 0.0f;
    if (!(r.qi[idx] > 1.0e-12f)) { r.qi[idx] = 0.0f; r.ni[idx] = 0.0f; }
    if (!(r.qr[idx] > 1.0e-12f)) { r.qr[idx] = 0.0f; r.nr[idx] = 0.0f; }
    if (!(r.qs[idx] > 1.0e-12f)) r.qs[idx] = 0.0f;
    if (!(r.qg[idx] > 1.0e-12f)) r.qg[idx] = 0.0f;
}

kernel void thompson_adapter_entry(
    constant ThompsonAdapterEntryResources &r [[buffer(0)]],
    constant ThompsonAdapterEntryParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= uint(p.columns)) return;
    bool has_microphysics = false;
    for (int k = 0; k < p.nz; ++k) {
        const uint idx = uint(k) * uint(p.columns) + column;
        if (idx >= uint(p.size)) return;
        r.graupel_number_per_kg[idx] = thompson_classic_graupel_number(
            r.qg, r.temperature, r.pressure, r.qv, idx);
        has_microphysics = has_microphysics || thompson_level_has_microphysics(
            r.qc, r.qi, r.qr, r.qs, r.qg, r.temperature,
            r.pressure, r.qv, ulong(idx));
    }
    r.micro_columns[column] = has_microphysics ? 1.0f : 0.0f;
}
