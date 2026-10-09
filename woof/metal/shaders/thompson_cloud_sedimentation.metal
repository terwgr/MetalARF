#include <metal_stdlib>
#include "thompson_cloud_sediment_with_density_and_masks_bindings.metalh"
#include "soft_binary64.metalh"

using namespace metal;
#pragma clang fp contract(off)

// The active mp=8 caller selects the held-density, post-source-mask variant.
// One thread owns a column, retaining CUDA's per-column serial traversal and
// its ordered height scan while covering both CUDA KMAX dispatch ranges.
static inline ulong cs64(float x) {
    return soft64_from_f32_bits(as_type<uint>(x));
}

static inline float cs32(ulong x) {
    return as_type<float>(soft64_to_f32_bits(x));
}

static inline bool cs_empty_column(device const float *qc,
        device const float *temperature, device const float *pressure,
        device const float *qv, device const float *dz,
        float dt, uint column, uint nz, uint ny, uint nx) {
    if (!(dt >= 0.0f && dt <= 100000.0f)) return false;
    for (uint k = 0; k < nz; ++k) {
        const uint idx = k * ny * nx + column;
        if (!(qc[idx] >= -1.0f && qc[idx] <= 1.0e-12f)) return false;
        if (!(temperature[idx] >= 100.0f && temperature[idx] <= 400.0f
                && pressure[idx] >= 1.0f && pressure[idx] <= 120000.0f
                && qv[idx] >= -1.0f && qv[idx] <= 1.0f
                && dz[idx] >= 1.0f && dz[idx] <= 100000.0f)) return false;
    }
    return true;
}

kernel void thompson_cloud_sediment_with_density_and_masks(
    constant ThompsonCloudSedimentWithDensityAndMasksResources &r [[buffer(0)]],
    constant ThompsonCloudSedimentWithDensityAndMasksParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny * p.nx);
    if (column >= ncol) return;
    if (r.cloud_active_columns[column] == 0.0f) return;
    const uint j = column / uint(p.nx);
    const uint i = column - j * uint(p.nx);

    // Preserve the upstream conservative exact shortcut, including its
    // terminal add-zero writes and signed-zero behavior.
    if (cs_empty_column(r.qc, r.temperature, r.pressure, r.qv, r.dz,
            p.dt, column, uint(p.nz), uint(p.ny), uint(p.nx))) {
        for (uint k = 0; k < uint(p.nz); ++k) {
            const uint idx = k * uint(p.ny) * uint(p.nx) + column;
            r.qc[idx] = r.qc[idx] + 0.0f * p.dt;
        }
        return;
    }

    float density[256];
    float cloud_mass[256];
    float mass_flux[256];
    float qc_tendency[256];
    const uint nz = uint(p.nz), ny = uint(p.ny), nx = uint(p.nx);
    const float am_r = 3.1415926536f * 1000.0f / 6.0f;
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const bool rain_refreshes_rhof = r.rain_active_columns[column] != 0.0f;

    for (uint k = 0; k < nz; ++k) {
        const uint idx = k * ny * nx + j * nx + i;
        const float qvk = fmax(1.0e-10f, r.qv[idx]);
        density[k] = 0.622f * r.pressure[idx]
            / (287.04f * r.temperature[idx] * (qvk + 0.622f));
        qc_tendency[k] = 0.0f;
        cloud_mass[k] = r.qc[idx] > 1.0e-12f
            ? r.qc[idx] * r.reference_density[idx] : 1.0e-12f;
        mass_flux[k] = 0.0f; // initially the CUDA mass_velocity array
    }

    int sediment_top = 0;
    float height_agl = 0.0f;
    for (uint k = 0; k < nz - 1; ++k) {
        const uint idx = k * ny * nx + j * nx + i;
        if (cloud_mass[k] > 1.0e-6f) sediment_top = int(k);
        height_agl += r.dz[idx];
        if (height_agl > 500.0f) break;
    }

    for (int k = sediment_top; k >= 0; --k) {
        const uint idx = uint(k) * ny * nx + j * nx + i;
        if (cloud_mass[k] > 1.0e-12f
                && r.vertical_velocity[idx] < 1.0e-1f) {
            const float lambda_arg = 100.0e6f * am_r * 2730.0f
                / cloud_mass[k];
            // CUDA widens the FP32 powf result, computes the reciprocal and
            // two products in binary64, then rounds back to FP32.
            const float lambda_f = precise::pow(lambda_arg, 1.0f / 3.0f);
            const ulong lambda = cs64(lambda_f);
            const ulong inverse_lambda = soft64_div(0x3ff0000000000000ul,
                                                     lambda);
            const float velocity_density = rain_refreshes_rhof
                ? density[k] : r.reference_density[idx];
            const float prefix = precise::sqrt(rho_not / velocity_density)
                * 0.316946e8f * 272.0f;
            const ulong velocity_wide = soft64_mul(
                soft64_mul(cs64(prefix), inverse_lambda), inverse_lambda);
            mass_flux[k] = cs32(velocity_wide);
        }
    }

    for (uint k = 0; k < nz; ++k) {
        mass_flux[k] = mass_flux[k] * cloud_mass[k];
    }
    for (int k = sediment_top; k >= 0; --k) {
        const uint idx = uint(k) * ny * nx + j * nx + i;
        const float divergence = mass_flux[k + 1] - mass_flux[k];
        const float inv_dz = 1.0f / r.dz[idx];
        qc_tendency[k] += divergence * inv_dz / density[k];
        cloud_mass[k] = fmax(1.0e-12f,
            cloud_mass[k] + divergence * inv_dz * p.dt);
    }

    for (uint k = 0; k < nz; ++k) {
        const uint idx = k * ny * nx + j * nx + i;
        r.qc[idx] = r.qc[idx] + qc_tendency[k] * p.dt;
    }
}
