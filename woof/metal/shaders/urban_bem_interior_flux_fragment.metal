#include <metal_stdlib>
#include "urban_bem_interior_flux_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmif_add(float a, float b) {
    return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubmif_sub(float a, float b) {
    return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubmif_mul(float a, float b) {
    return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b)));
}

// WRF module_sf_bep_bem.F bem_hsinsflux/bem_fluxcond/bem_fluxroo.
// One item is one occupied building column and indoor level. wall_surface_temp
// stores the four canyon walls followed by the lower and upper horizontal
// surfaces; window_surface_temp stores the four window temperatures. The
// caller gathers the correct ground/floor/roof temperatures into slots 4/5.
kernel void urban_bem_interior_flux_fragment(
    constant UrbanBemInteriorFluxFragmentResources &r [[buffer(0)]],
    constant UrbanBemInteriorFluxFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    device const float *tlev = r.tlev;
    device const float *wall_surface_temp = r.wall_surface_temp;
    device const float *window_surface_temp = r.window_surface_temp;
    device const float *surface_area = r.surface_area;
    device const float *hseqocc = r.hseqocc;
    device const float *hleqocc = r.hleqocc;
    device const float *hsvent = r.hsvent;
    device const float *hlvent = r.hlvent;
    device float *hswalins = r.hswalins;
    device float *hswinins = r.hswinins;
    device float *hscond = r.hscond;
    device float *hslev = r.hslev;
    device float *hllev = r.hllev;
    device int *error = r.error;

    const int ncol = p.ncol;
    const int nlev = p.nlev;
    const float pwin = p.pwin;
    if (item >= uint(ncol * nlev)) return;
    if (ncol < 1 || nlev < 1 || nlev > 18) {
        error[item] = 3810;
        return;
    }
    if (!(pwin >= 0.0f && pwin <= 1.0f) || !isfinite(pwin)) {
        error[item] = 3811;
        return;
    }

    const ulong base10 = item * 10ul;
    const ulong base6 = item * 6ul;
    const ulong base4 = item * 4ul;
    const float indoor = tlev[item];
    const float wall_hc = ubmif_mul(5.678f, 1.09f);
    const float window_hc = ubmif_mul(5.678f, 0.99f);
    float wall_flux[6];
    float window_flux[4];
    for (uint i = 0; i < 4; ++i) {
        wall_flux[i] = ubmif_mul(wall_hc, ubmif_sub(indoor, wall_surface_temp[base10 + i]));
        window_flux[i] = ubmif_mul(window_hc, ubmif_sub(indoor, window_surface_temp[base4 + i]));
    }
    wall_flux[4] = ubmif_mul(wall_hc, ubmif_sub(indoor, wall_surface_temp[base10 + 4ul]));
    wall_flux[5] = ubmif_mul(wall_hc, ubmif_sub(indoor, wall_surface_temp[base10 + 5ul]));

    float cond = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        const float area = surface_area[base6 + i];
        const float opaque = ubmif_mul(ubmif_mul(area, ubmif_sub(1.0f, pwin)), wall_flux[i]);
        cond = ubmif_add(cond, opaque);
        const float glazed = ubmif_mul(ubmif_mul(area, pwin), window_flux[i]);
        cond = ubmif_add(cond, glazed);
    }
    cond = ubmif_add(cond, ubmif_mul(surface_area[base6 + 4ul], wall_flux[4]));
    cond = ubmif_add(cond, ubmif_mul(surface_area[base6 + 5ul], wall_flux[5]));
    cond = ubmif_mul(-1.0f, cond);

    for (uint i = 0; i < 6; ++i) hswalins[base6 + i] = wall_flux[i];
    for (uint i = 0; i < 4; ++i) hswinins[base4 + i] = window_flux[i];
    hscond[item] = cond;
    hslev[item] = ubmif_add(ubmif_add(hseqocc[item], hsvent[item]), cond);
    hllev[item] = ubmif_add(hleqocc[item], hlvent[item]);
    error[item] = 0;
}
