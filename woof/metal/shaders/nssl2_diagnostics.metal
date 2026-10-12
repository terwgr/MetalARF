// WRF v4.6.1 NSSL option-18 S-band reflectivity and radius diagnostics.
#include <metal_stdlib>
#include "nssl2_diagnostics_bindings.metalh"
#include "nssl2_soft64.metalh"
using namespace metal;
#pragma clang fp contract(off)
kernel void nssl2_radardd02(constant Nssl2Radardd02Resources &r [[buffer(0)]], constant Nssl2Radardd02Params &p [[buffer(1)]], uint thread_position_in_grid [[thread_position_in_grid]]) {
    const int idx = thread_position_in_grid;
    if (idx >= p.n) return;

    const float rho = r.air_density[idx];
    const float pi = 3.14159265358979323846f;
    float total = 0.0f;
    (void)r.temperature_k;

    // Rain: imurain=1, alphar=0, sixth moment reconstructed from q and N.
    const float number_scale = p.concentration_space ? 1.0f : rho;
    const float volume_scale = p.concentration_space ? 1.0f : rho;
    const float rain_number = r.qnr[idx] * number_scale;
    if (r.qr[idx] >= 1.0e-12f && rain_number > 1.0e-3f) {
        const Nssl2Soft64 g1(20.0f);
        const Nssl2Soft64 mass = Nssl2Soft64(rho) * Nssl2Soft64(r.qr[idx]);
        const Nssl2Soft64 zx = g1 * mass * mass / Nssl2Soft64(rain_number);
        const Nssl2Soft64 scale = Nssl2Soft64(6.0f)
            / (Nssl2Soft64(pi) * Nssl2Soft64(1000.0f));
        const Nssl2Soft64 ze = nssl2_soft64_const(0x43abc16d674ec800ul)
            * zx * scale * scale;
        total = (float)ze;
    }

    // Snow: the active dry Cox (1988) m=p*V^(2/3) relationship.
    const float snow_number = r.qns[idx] * number_scale;
    if (r.qs[idx] >= 1.0e-13f && snow_number > 1.0e-7f) {
        const float snu = -0.8f;
        const float gsnow1 = 4.590843677520752f;
        const float gsnow73 = 0.8877619504928589f;
        const float snow = 1.0e18f * 323.3226f
            * (0.106214f * 0.106214f)
            * (0.189f * r.qs[idx]) * r.qs[idx] * rho * rho * gsnow73
            / (snow_number * (917.0f * 917.0f) * gsnow1
               * pow(1.0f + snu, 4.0f / 3.0f));
        total += snow;
    }

    // Cloud ice: spherical 900-kg/m3 particles and cinu=0.
    const float ice_number = r.qni[idx] * number_scale;
    if (r.qi[idx] > 1.0e-13f && ice_number > 1.0f) {
        const float mean_volume =
            rho * r.qi[idx] / (900.0f * ice_number);
        const float ice = 0.224f * 3.6e18f * 2.0f
            * ice_number * mean_volume * mean_volume * (0.9f * 0.9f);
        total += ice;
    }

    // Variable-density graupel: alphah=0, 0.3--20-mm volume bounds.
    const float graupel_number = r.qng[idx] * number_scale;
    if (r.qg[idx] >= 1.0e-12f && graupel_number >= 1.0e-8f) {
        float density = 500.0f;
        const float volume_mixing_ratio = r.qvolg[idx] * volume_scale;
        if (volume_mixing_ratio > 0.0f) {
            density = rho * r.qg[idx] / volume_mixing_ratio;
            density = fmin(900.0f, fmax(100.0f, density));
        }
        float local_number = graupel_number;
        float mean_volume = rho * r.qg[idx]
            / (density * fmax(1.0e-3f, local_number));
        const float minimum_volume =
            0.523599f * (0.3e-3f * 0.3e-3f * 0.3e-3f);
        const float maximum_volume =
            0.523599f * (20.0e-3f * 20.0e-3f * 20.0e-3f);
        if (mean_volume < minimum_volume || mean_volume > maximum_volume) {
            mean_volume = fmin(
                maximum_volume, fmax(minimum_volume, mean_volume));
            local_number = rho * r.qg[idx] / (mean_volume * density);
        }
        if (local_number > 0.0f) {
            const Nssl2Soft64 g1(20.0f);
            const Nssl2Soft64 zx = g1 * Nssl2Soft64(rho)
                * Nssl2Soft64(rho)
                * (nssl2_soft64_const(0x3fccac083126e979ul)
                   * Nssl2Soft64(r.qg[idx])) * Nssl2Soft64(r.qg[idx])
                / Nssl2Soft64(local_number);
            const Nssl2Soft64 scale = Nssl2Soft64(6.0f)
                / (Nssl2Soft64(pi) * Nssl2Soft64(1000.0f));
            total += (float)(nssl2_soft64_const(0x43abc16d674ec800ul)
                             * zx * scale * scale);
        }
    }

    // Variable-density large hail: alphahl=1, 0.3--40-mm bounds.
    const float hail_number = r.qnh[idx] * number_scale;
    if (r.qh[idx] >= 1.0e-12f && hail_number > 0.0f) {
        float density = 900.0f;
        const float volume_mixing_ratio = r.qvolh[idx] * volume_scale;
        if (volume_mixing_ratio > 0.0f) {
            density = rho * r.qh[idx] / volume_mixing_ratio;
            density = fmin(900.0f, fmax(300.0f, density));
        }
        float local_number = hail_number;
        float mean_volume = rho * r.qh[idx]
            / (density * fmax(1.0e-9f, local_number));
        const float minimum_volume =
            0.523599f * (0.3e-3f * 0.3e-3f * 0.3e-3f);
        const float maximum_volume =
            0.523599f * (40.0e-3f * 40.0e-3f * 40.0e-3f);
        if (mean_volume < minimum_volume || mean_volume > maximum_volume) {
            mean_volume = fmin(
                maximum_volume, fmax(minimum_volume, mean_volume));
            local_number = rho * r.qh[idx] / (mean_volume * density);
        }
        if (local_number > 0.0f) {
            const Nssl2Soft64 g1(8.75f);
            const Nssl2Soft64 zx = g1 * Nssl2Soft64(rho)
                * Nssl2Soft64(rho)
                * (nssl2_soft64_const(0x3fccac083126e979ul)
                   * Nssl2Soft64(r.qh[idx])) * Nssl2Soft64(r.qh[idx])
                / Nssl2Soft64(local_number);
            const Nssl2Soft64 scale = Nssl2Soft64(6.0f)
                / (Nssl2Soft64(pi) * Nssl2Soft64(1000.0f));
            total += (float)(nssl2_soft64_const(0x43abc16d674ec800ul)
                             * zx * scale * scale);
        }
    }

    r.refl_10cm[idx] = total > 0.0f
        ? fmax(0.0f, 10.0f * log10(total))
        : 0.0f;
}

struct Nssl2EffectiveRadii {
    float cloud;
    float ice;
    float snow;
};

inline Nssl2EffectiveRadii nssl2_effective_radii(
        float rho, float qc, float cloud_number, float qi, float ice_number,
        float qs, float snow_number, bool registry_unit_numbers) {
    Nssl2EffectiveRadii radii = {2.51e-6f, 10.01e-6f, 25.0e-6f};
    const float number_scale = registry_unit_numbers ? rho : 1.0f;
    const float pi_over_six = 3.14159265358979323846f / 6.0f;
    const float one_third = 1.0f / 3.0f;
    const float cxmin = 1.0e-8f;
    const float qxmin = 1.0e-13f;

    const float cloud_mass = fmax(qc, 0.0f);
    const float nc = fmax(cloud_number * number_scale, 0.0f);
    if (cloud_mass > qxmin && nc > cxmin) {
        const float lambda = pow(
            (nc * pi_over_six * 1000.0f) / (cloud_mass * rho), one_third);
        const float raw = 0.5f * 1.1077321767807007f / lambda;
        radii.cloud = fmax(2.51e-6f, fmin(raw, 50.0e-6f));
    }

    const float ice_mass = fmax(qi, 0.0f);
    const float ni = fmax(ice_number * number_scale, 0.0f);
    if (ice_mass > qxmin && ni > cxmin) {
        const float lambda = pow(
            (ni * pi_over_six * 900.0f) / (ice_mass * rho), one_third);
        const float raw = 0.5f * 1.1077321767807007f / lambda;
        radii.ice = fmax(10.01e-6f, fmin(raw, 125.0e-6f));
    }

    const float snow_mass = fmax(qs, 0.0f);
    const float ns = fmax(snow_number * number_scale, 0.0f);
    if (snow_mass > qxmin && ns > cxmin) {
        const float gamma_mass = 0.91816872358322144f;
        const float gamma_number = 4.5908436775207520f;
        const float radius_factor = 0.83693999052047729f;
        const float lambda = pow(
            (ns * pi_over_six * 100.0f * gamma_mass)
                / (snow_mass * rho * gamma_number), one_third);
        const float raw = 0.5f * radius_factor / lambda;
        radii.snow = fmax(25.0e-6f, fmin(raw, 999.0e-6f));
    }
    return radii;
}

kernel void nssl2_effective_radius_concentration(constant Nssl2EffectiveRadiusConcentrationResources &r [[buffer(0)]], constant Nssl2EffectiveRadiusConcentrationParams &p [[buffer(1)]], uint thread_position_in_grid [[thread_position_in_grid]]) {
    const int idx = thread_position_in_grid;
    if (idx >= p.n) return;
    const Nssl2EffectiveRadii radii = nssl2_effective_radii(
        r.air_density[idx], r.qc[idx], r.cloud_number[idx], r.qi[idx],
        r.ice_number[idx], r.qs[idx], r.snow_number[idx], false);
    r.re_cloud[idx] = radii.cloud;
    r.re_ice[idx] = radii.ice;
    r.re_snow[idx] = radii.snow;
}
kernel void nssl2_effective_radius_registry_units(constant Nssl2EffectiveRadiusRegistryUnitsResources &r [[buffer(0)]], constant Nssl2EffectiveRadiusRegistryUnitsParams &p [[buffer(1)]], uint thread_position_in_grid [[thread_position_in_grid]]) {
    const int idx = thread_position_in_grid;
    if (idx >= p.n) return;
    const Nssl2EffectiveRadii radii = nssl2_effective_radii(
        r.air_density[idx], r.qc[idx], r.cloud_number[idx], r.qi[idx],
        r.ice_number[idx], r.qs[idx], r.snow_number[idx],
        p.registry_unit_numbers != 0);
    r.re_cloud[idx] = radii.cloud;
    r.re_ice[idx] = radii.ice;
    r.re_snow[idx] = radii.snow;
}
