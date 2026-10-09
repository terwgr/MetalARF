#include <metal_stdlib>
#include "urban_bem_wall_flux_fragment_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_pow.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubw_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubw_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubw_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubw_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubw_pow(float x, float y) { return as_type<float>(soft_gfk_pow_bits(as_type<uint>(x), as_type<uint>(y))); }

// WRF bb_flux_wall (module_sf_bep_bem.F:4157-4198). Each item is one
// [column, building, urban-level, wall-face] tuple. Faces 0/1 use street
// direction 0; faces 2/3 use direction 1. Temperatures passed to this stage
// are already converted from potential to actual temperature by the kernel.
kernel void urban_bem_wall_flux_fragment(
    constant UrbanBemWallFluxFragmentResources &r [[buffer(0)]],
    constant UrbanBemWallFluxFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= uint(p.nitem)) return;
    device const int *active = r.active_cols;
    device const int *nlev = r.building_nlev;
    device const float *interp = r.interp;
    device const float *tw = r.tw;
    device const float *twlev = r.twlev;
    device const float *pb = r.pb;
    device const float *bs = r.bs;
    device const float *ws = r.ws;
    device const float *drst = r.drst;
    device float *uva = r.uva, *vva = r.vva, *uvb = r.uvb, *vvb = r.vvb;
    device float *sfw = r.sfw, *sfwin = r.sfwin, *evb = r.evb;
    device int *error = r.error;
    const uint face = item % 4u;
    const uint iz = (item / 4u) % 18u;
    const uint building = (item / 72u) % 15u;
    const uint col = item / (15u * 18u * 4u);
    error[item] = 0;
    if (col >= uint(p.ncol) || active[col] == 0 || iz >= uint(max(nlev[ulong(col) * 15ul + building], 0))) {
        uva[item] = 0.0f; vva[item] = 0.0f; uvb[item] = 0.0f; vvb[item] = 0.0f;
        sfw[item] = 0.0f; sfwin[item] = 0.0f; evb[item] = 0.0f;
        return;
    }
    const uint direction = face / 2u;
    const ulong atmos = ulong(col) * 7ul * 18ul;
    const float ua = interp[atmos + iz];
    const float va = interp[atmos + 18ul + iz];
    const float pt = interp[atmos + 36ul + iz];
    const float pr = interp[atmos + 72ul + iz];
    const float da = interp[atmos + 90ul + iz];
    const float angle = drst[ulong(col) * 2ul + direction];
    const float sine = precise::sin(angle), cosine = precise::cos(angle);
    const float speed = ubw_pow(ubw_add(ubw_mul(ua, ua), ubw_mul(va, va)), 0.5f);
    const float u_ort = fabs(ubw_sub(ubw_mul(cosine, ua), ubw_mul(sine, va)));
    const float ratio = ubw_div(pr, p.p0);
    const float temperature_scale = ubw_pow(ratio, -p.rcp_u);
    const ulong wall = (((ulong(col) * 15ul + building) * 10ul + 9ul) * 18ul + iz) * 4ul + face;
    const ulong window = ((ulong(col) * 15ul + building) * 18ul + iz) * 4ul + face;
    const float ptw = ubw_mul(tw[wall], temperature_scale);
    const float ptwin = ubw_mul(twlev[window], temperature_scale);
    float alpha = 0.0f;
    for (uint d = 0; d < 2u; ++d) {
        const float denom = ubw_add(ws[ulong(col) * 2ul + d], bs[ulong(col) * 2ul + d]);
        if (!(denom > 0.0f)) { error[item] = 7410; return; }
        alpha = ubw_add(alpha, ubw_mul(ubw_div(bs[ulong(col) * 2ul + d], denom),
                                      pb[ulong(col) * 19ul + iz]));
    }
    alpha = ubw_div(alpha, 2.0f);
    const float cdrag = alpha < 0.29f ? ubw_mul(3.32f, ubw_pow(alpha, 0.47f)) : 1.85f;
    uva[item] = ubw_mul(ubw_mul(ubw_div(ubw_mul(-cdrag, u_ort), 2.0f), cosine), cosine);
    vva[item] = ubw_mul(ubw_mul(ubw_div(ubw_mul(-cdrag, u_ort), 2.0f), sine), sine);
    uvb[item] = ubw_mul(ubw_mul(ubw_mul(ubw_div(ubw_mul(cdrag, u_ort), 2.0f), sine), cosine), va);
    vvb[item] = ubw_mul(ubw_mul(ubw_mul(ubw_div(ubw_mul(cdrag, u_ort), 2.0f), sine), cosine), ua);
    const float limit = ubw_div(ubw_mul(da, p.cp_u), p.dt);
    float hc = speed < 4.88f
        ? ubw_mul(5.678f, ubw_add(1.09f, ubw_mul(0.23f, ubw_div(speed, 0.3048f))))
        : ubw_mul(ubw_mul(5.678f, 0.53f), ubw_pow(ubw_div(speed, 0.3048f), 0.78f));
    hc = min(hc, limit);
    float hcwin = speed < 4.88f
        ? ubw_mul(5.678f, ubw_add(0.99f, ubw_mul(0.21f, ubw_div(speed, 0.3048f))))
        : ubw_mul(ubw_mul(5.678f, 0.50f), ubw_pow(ubw_div(speed, 0.3048f), 0.78f));
    hcwin = min(hcwin, limit);
    sfw[item] = ubw_mul(hc, ubw_sub(pt, ptw));
    sfwin[item] = ubw_mul(hcwin, ubw_sub(pt, ptwin));
    evb[item] = ubw_div(ubw_mul(cdrag, ubw_pow(u_ort, 3.0f)), 2.0f);
}
