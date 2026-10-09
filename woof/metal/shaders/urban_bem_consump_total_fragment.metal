#include <metal_stdlib>
#include "urban_bem_consump_total_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmct_add(float a, float b) {
    return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubmct_div(float a, float b) {
    return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)));
}

// WRF module_sf_bem.F bem_consump_total (2317-2362). One item handles one
// building. WRF sums levels 1..nlev, reports the sensible output sum without
// scaling, and converts the energy-use sum from joules to kWh.
kernel void urban_bem_consump_total_fragment(
    constant UrbanBemConsumpTotalFragmentResources &r [[buffer(0)]],
    constant UrbanBemConsumpTotalFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    device const float *hsout = r.hsout;
    device const float *consump = r.consump;
    device const int *nlev = r.nlev;
    device float *hsoutbuild = r.hsoutbuild;
    device float *consumpbuild = r.consumpbuild;
    device int *error = r.error;
    const int nitem = p.nitem;
    const int nzcanm = p.nzcanm;
    if (item >= uint(nitem)) return;
    if (nitem < 1 || nzcanm < 1 || nzcanm > 18 ||
        nlev[item] < 0 || nlev[item] > nzcanm) {
        hsoutbuild[item] = 0.0f;
        consumpbuild[item] = 0.0f;
        error[item] = 2155;
        return;
    }
    float total_hs = 0.0f;
    float total_consumption = 0.0f;
    const ulong base = ulong(item) * ulong(nzcanm);
    for (int level = 0; level < nlev[item]; ++level) {
        total_consumption = ubmct_add(total_consumption,
                                      consump[base + uint(level)]);
        total_hs = ubmct_add(total_hs, hsout[base + uint(level)]);
    }
    consumpbuild[item] = ubmct_div(total_consumption, 3.6e6f);
    hsoutbuild[item] = total_hs;
    error[item] = 0;
}
