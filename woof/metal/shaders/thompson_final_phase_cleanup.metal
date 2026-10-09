#include <metal_stdlib>
#include "thompson_final_phase_cleanup_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"

using namespace metal;
#pragma clang fp contract(off)

static inline ulong cleanup64(float x) {
    return soft64_from_f32_bits(as_type<uint>(x));
}
static inline float cleanup32(ulong x) {
    return as_type<float>(soft64_to_f32_bits(x));
}

static inline void cleanup_bound_ice(float ice_mass, float density,
                                     thread float& ice_number_per_kg) {
    if (ice_mass <= 1.0e-12f) {
        ice_number_per_kg = 0.0f;
        return;
    }
    const float am_i = 3.1415926536f * 890.0f / 6.0f;
    float ice_number = fmax(1.0e-6f, ice_number_per_kg * density);
    ulong lambda = cleanup64(precise::pow(
        am_i * 6.0f * ice_number / ice_mass, 1.0f / 3.0f));
    const float diameter = cleanup32(soft64_div(cleanup64(4.0f), lambda));
    if (diameter < 5.0e-6f) {
        lambda = cleanup64(4.0f / 5.0e-6f);
        const float sixth_over_mass = (1.0f / 6.0f) * ice_mass / am_i;
        float bounded = sixth_over_mass * cleanup32(
            soft64_mul(soft64_mul(lambda, lambda), lambda));
        ice_number = fmin(999.0e3f, bounded);
    } else if (diameter > 300.0e-6f) {
        lambda = cleanup64(4.0f / 300.0e-6f);
        const float sixth_over_mass = (1.0f / 6.0f) * ice_mass / am_i;
        ice_number = sixth_over_mass * cleanup32(
            soft64_mul(soft64_mul(lambda, lambda), lambda));
    }
    ice_number_per_kg = fmin(ice_number, 999.0e3f) / density;
}

kernel void thompson_final_phase_cleanup_with_columns(
    constant ThompsonFinalPhaseCleanupWithColumnsResources& r [[buffer(0)]],
    constant ThompsonFinalPhaseCleanupWithColumnsParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.size)) return;
    const uint ncol = uint(p.ncol);
    const uint column = idx % ncol;
    if (r.micro_columns[column] == 0.0f) {
        if (r.qc[idx] <= 1.0e-12f) r.qc[idx] = 0.0f;
        if (r.qi[idx] <= 1.0e-12f) {
            r.qi[idx] = 0.0f;
            r.ni[idx] = 0.0f;
        }
        return;
    }

    r.qv[idx] = fmax(1.0e-10f, r.qv[idx]);
    const float temp0 = r.temperature[idx];
    const float qv0 = fmax(1.0e-10f, r.qv[idx]);
    const float rho = 0.622f * r.pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float inverse_cp = 1.0f / (1004.0f * (1.0f + 0.887f * qv0));

    if (temp0 > 273.15f && r.qi[idx] > 0.0f) {
        const float transferred = fmax(0.0f, r.qi[idx]);
        r.qc[idx] += transferred;
        r.qi[idx] = 0.0f;
        r.ni[idx] = 0.0f;
        r.temperature[idx] -= 334000.0f * inverse_cp * transferred;
    } else if (temp0 < 235.16f && r.qc[idx] > 0.0f) {
        const float transferred = fmax(0.0f, r.qc[idx]);
        const float latent_vapor = 2.5e6f
            + (2106.0f - 4218.0f) * (temp0 - 273.15f);
        const float latent_fusion = 2.834e6f - latent_vapor;
        r.qc[idx] = 0.0f;
        r.qi[idx] += transferred;
        r.ni[idx] += 100.0e6f / rho;
        r.temperature[idx] += latent_fusion * inverse_cp * transferred;
    }

    if (r.qc[idx] <= 1.0e-12f) r.qc[idx] = 0.0f;
    if (r.qi[idx] <= 1.0e-12f) {
        r.qi[idx] = 0.0f;
        r.ni[idx] = 0.0f;
    } else if (r.qi[idx] * rho > 1.0e-12f) {
        float ni = r.ni[idx];
        cleanup_bound_ice(r.qi[idx] * rho, rho, ni);
        r.ni[idx] = ni;
    } else {
        const float am_i = 3.1415926536f * 890.0f / 6.0f;
        const float qi_local = r.qi[idx];
        const float ni_local = fmax(1.0e-6f / rho, r.ni[idx]);
        const float lambda_arg = am_i * 6.0f * ni_local / qi_local;
        ulong lambda = cleanup64(precise::pow(lambda_arg, 1.0f / 3.0f));
        const ulong inverse_lambda = soft64_div(cleanup64(1.0f), lambda);
        const float xdi = cleanup32(soft64_mul(cleanup64(4.0f), inverse_lambda));
        if (xdi < 5.0e-6f) {
            lambda = cleanup64(4.0f / 5.0e-6f);
        } else if (xdi > 300.0e-6f) {
            lambda = cleanup64(4.0f / 300.0e-6f);
        }
        const float sixth_over_mass = ((1.0f / 6.0f) * qi_local) / am_i;
        const ulong lambda3 = soft64_plm_pow(lambda, 0x4008000000000000ul);
        const ulong uncapped = soft64_mul(cleanup64(sixth_over_mass), lambda3);
        const ulong cap = soft64_div(cleanup64(999.0e3f), cleanup64(rho));
        const ulong number = soft64_less(uncapped, cap) ? uncapped : cap;
        r.ni[idx] = cleanup32(soft64_to_f32_bits(number));
    }
}
