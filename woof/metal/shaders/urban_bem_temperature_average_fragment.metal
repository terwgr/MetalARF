#include <metal_stdlib>
#include "urban_bem_temperature_average_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubma_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubma_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubma_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubma_pow4(float x) { float x2 = ubma_mul(x, x); return ubma_mul(x2, x2); }

// WRF module_sf_bep_bem.F bb_averaging_temp (2183-2227). One item handles
// one urban column. Surface and indoor-wall temperatures are averaged in
// fourth-power space; sensible and window fluxes use the same weighted
// building fractions. Fixed extents are 2 directions, 4 wall faces, 18
// urban levels, and 15 active building slots. State uses the actual BEM
// row-major packing: tw[col,building,wall-layer,level,face], while twlev,
// sfw, and sfwin use [col,building,level,face]. The surface wall temperature
// is the outermost WRF wall layer (nwr_u=5).
kernel void urban_bem_temperature_average_fragment(
    constant UrbanBemTemperatureAverageFragmentResources &r [[buffer(0)]],
    constant UrbanBemTemperatureAverageFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    device const float *ss = r.ss;
    device const float *pb = r.pb;
    device const float *tw = r.tw;
    device const float *twlev = r.twlev;
    device const float *sfw = r.sfw;
    device const float *sfwin = r.sfwin;
    device float *tw_av = r.tw_av;
    device float *twlev_av = r.twlev_av;
    device float *sfw_av = r.sfw_av;
    device float *sfwind_av = r.sfwind_av;
    device int *error = r.error;
    const int ncol = p.ncol;
    if (col >= uint(ncol)) return;
    const ulong profile = ulong(col) * 19ul;
    const ulong cumulative = ulong(col) * 19ul;
    const ulong faces = ulong(col) * 72ul;
    for (int face = 0; face < 4; ++face)
        for (int iz = 0; iz < 18; ++iz) {
            const ulong out = faces + uint(face * 18 + iz);
            tw_av[out] = 0.0f; twlev_av[out] = 0.0f;
            sfw_av[out] = 0.0f; sfwind_av[out] = 0.0f;
        }
    int nactive = 0;
    float building_fraction[18];
    int top_layer[18];
    for (int iz = 0; iz < 18; ++iz) {
        building_fraction[iz] = 0.0f;
        top_layer[iz] = 0;
        if (ss[profile + uint(iz)] > 0.0f) {
            building_fraction[nactive] = ss[profile + uint(iz)];
            top_layer[nactive] = iz;
            ++nactive;
        }
    }
    for (int face = 0; face < 4; ++face) {
        for (int iz = 0; iz < 17; ++iz) {
            const float remaining = pb[cumulative + uint(iz + 1)];
            if (!(remaining > 0.0f)) continue;
            const ulong out = faces + uint(face * 18 + iz);
            for (int ib = 0; ib < nactive; ++ib) {
                // CUDA uses one-based iz <= nlev(ib), with nlev=source_iz-1.
                if (iz + 1 > top_layer[ib]) continue;
                const float weight = ubma_div(building_fraction[ib], remaining);
                const ulong wall = (((ulong(col) * 15ul + uint(ib)) * 5ul + 4ul)
                    * 18ul + uint(iz)) * 4ul + uint(face);
                const ulong building_level =
                    (ulong(col) * 15ul + uint(ib)) * 72ul
                    + uint(iz * 4 + face);
                tw_av[out] = ubma_add(tw_av[out], ubma_mul(weight, ubma_pow4(tw[wall])));
                twlev_av[out] = ubma_add(twlev_av[out], ubma_mul(weight, ubma_pow4(twlev[building_level])));
                sfw_av[out] = ubma_add(sfw_av[out], ubma_mul(weight, sfw[building_level]));
                sfwind_av[out] = ubma_add(sfwind_av[out], ubma_mul(weight, sfwin[building_level]));
            }
            tw_av[out] = powr(tw_av[out], 0.25f);
            twlev_av[out] = powr(twlev_av[out], 0.25f);
        }
    }
    error[col] = 0;
}
