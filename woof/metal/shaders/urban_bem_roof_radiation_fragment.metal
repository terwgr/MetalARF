#include <metal_stdlib>
#include "urban_bem_roof_radiation_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmr_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmr_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmr_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmr_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmr_pow4(float x) { float x2 = ubmr_mul(x, x); return ubmr_mul(x2, x2); }

// WRF module_sf_bem.F bem_radfluxs/bem_radfluxspv (2537-2585). One item
// computes a roof/opaque-surface net radiative flux. PV roofs include the
// source's panel-to-roof radiative exchange and validation of the selected
// panel level. Source-only: caller and typed ABI need integration.
kernel void urban_bem_roof_radiation_fragment(
    constant UrbanBemRoofRadiationFragmentResources &r [[buffer(0)]],
    constant UrbanBemRoofRadiationFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    device const float *albedo = r.albedo;
    device const float *rs = r.rs;
    device const float *emissivity = r.emissivity;
    device const float *rl = r.rl;
    device const float *surface_temp = r.surface_temp;
    device const float *pv_fraction = r.pv_fraction;
    device const float *panel_temp = r.panel_temp;
    device float *radflux = r.radflux;
    device int *error = r.error;
    const int nitem = p.nitem;
    const int has_pv = p.has_pv;
    const int level = p.level;
    const int nz = p.nz;
    const float sigma = p.sigma;
    if (item >= uint(nitem)) return;
    if (has_pv == 0) {
        radflux[item] = ubmr_sub(ubmr_add(
            ubmr_mul(ubmr_sub(1.0f, albedo[item]), rs[item]),
            ubmr_mul(emissivity[item], rl[item])),
            ubmr_mul(ubmr_mul(emissivity[item], sigma), ubmr_pow4(surface_temp[item])));
        error[item] = 0;
        return;
    }
    if (level + 1 > nz) { radflux[item] = 0.0f; error[item] = 5; return; }
    constexpr float panel_emissivity = 0.95f;
    const float hrdenom = ubmr_add(ubmr_add(
        ubmr_div(ubmr_sub(1.0f, panel_emissivity), panel_emissivity), 1.0f),
        ubmr_div(ubmr_sub(1.0f, emissivity[item]), emissivity[item]));
    const float hrad = ubmr_div(sigma, hrdenom);
    const float frac = pv_fraction[item];
    const float tw4 = ubmr_pow4(surface_temp[item]);
    const float panel_exchange = ubmr_mul(ubmr_mul(frac, hrad),
        ubmr_sub(ubmr_pow4(panel_temp[item]), tw4));
    const float shortwave = ubmr_mul(ubmr_mul(ubmr_sub(1.0f, albedo[item]),
        ubmr_sub(1.0f, frac)), rs[item]);
    const float longwave = ubmr_mul(ubmr_mul(emissivity[item],
        ubmr_sub(1.0f, frac)), rl[item]);
    const float roof_emission = ubmr_mul(ubmr_mul(ubmr_mul(emissivity[item], sigma),
        ubmr_sub(1.0f, frac)), tw4);
    radflux[item] = ubmr_sub(ubmr_add(ubmr_add(shortwave, longwave), panel_exchange), roof_emission);
    error[item] = 0;
}
