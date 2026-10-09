#include <metal_stdlib>
#include "urban_bem_view_factors_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmv_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmv_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmv_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmv_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmv_sqrt(float x) { return precise::sqrt(x); }
static inline float ubmv_log(float x) { return precise::log(x); }
static inline float ubmv_atan(float x) { return precise::atan(x); }

// WRF module_sf_bep_bem.F bb_view_factors/bb_fprls/bb_fnrms
// (4875-5013, 5021-5048). The output is a per-column/per-direction canyon
// factor set. The source's logarithm/power choices still require review at
// integration; this uses Metal precise math and source-ordered FP32 ops.
static inline float ubmv_fprls(float a, float b, float c) {
    const float x = ubmv_div(a, c), y = ubmv_div(b, c);
    if (a == 0.0f || b == 0.0f) return 0.0f;
    const float x2 = ubmv_mul(x, x), y2 = ubmv_mul(y, y);
    const float root = ubmv_sqrt(ubmv_div(ubmv_mul(ubmv_add(1.0f, x2), ubmv_add(1.0f, y2)), ubmv_add(ubmv_add(1.0f, x2), y2)));
    const float t1 = ubmv_log(root);
    const float t2 = ubmv_mul(ubmv_mul(y, ubmv_sqrt(ubmv_add(1.0f, x2))), ubmv_atan(ubmv_div(y, ubmv_sqrt(ubmv_add(1.0f, x2)))));
    const float t3 = ubmv_mul(ubmv_mul(x, ubmv_sqrt(ubmv_add(1.0f, y2))), ubmv_atan(ubmv_div(x, ubmv_sqrt(ubmv_add(1.0f, y2)))));
    const float v = ubmv_sub(ubmv_sub(ubmv_add(ubmv_add(t1, t2), t3), ubmv_mul(y, ubmv_atan(y))), ubmv_mul(x, ubmv_atan(x)));
    return ubmv_div(ubmv_mul(v, 2.0f), ubmv_mul(ubmv_mul(3.141592653f, x), y));
}
static inline float ubmv_fnrm(float a, float b, float c) {
    const float x = ubmv_div(a, b), y = ubmv_div(c, b);
    const float x2 = ubmv_mul(x, x), y2 = ubmv_mul(y, y);
    const float z = ubmv_add(x2, y2);
    if (y == 0.0f || x == 0.0f) return 0.0f;
    const float a1 = ubmv_log(ubmv_div(ubmv_mul(ubmv_add(1.0f, x2), ubmv_add(1.0f, y2)), ubmv_add(1.0f, z)));
    const float a2 = ubmv_mul(y2, ubmv_log(ubmv_div(ubmv_div(ubmv_mul(y2, ubmv_add(1.0f, z)), z), ubmv_add(1.0f, y2))));
    const float a3 = ubmv_mul(x2, ubmv_log(ubmv_div(ubmv_div(ubmv_mul(x2, ubmv_add(1.0f, z)), z), ubmv_add(1.0f, x2))));
    const float a4 = ubmv_mul(y, ubmv_atan(ubmv_div(1.0f, y)));
    const float a5 = ubmv_mul(x, ubmv_atan(ubmv_div(1.0f, x)));
    const float rootz = ubmv_sqrt(z);
    const float a6 = ubmv_mul(rootz, ubmv_atan(ubmv_div(1.0f, rootz)));
    float v = ubmv_sub(ubmv_add(ubmv_add(ubmv_mul(0.25f, ubmv_add(ubmv_add(a1, a2), a3)), a4), a5), a6);
    return ubmv_div(v, ubmv_mul(3.141592653f, y));
}

kernel void urban_bem_view_factors_fragment(
    constant UrbanBemViewFactorsFragmentResources &r [[buffer(0)]],
    constant UrbanBemViewFactorsFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    device const float *z = r.z;
    device const float *dxy = r.dxy;
    device const float *street_width = r.street_width;
    device float *fww = r.fww;
    device float *fwg = r.fwg;
    device float *fgw = r.fgw;
    device float *fsw = r.fsw;
    device float *fws = r.fws;
    device float *fsg = r.fsg;
    device int *error = r.error;
    const int ncol = p.ncol, nz_u = p.nz_u, nd = p.nd;
    if (item >= uint(ncol * nd)) return;
    const int col = int(item) / nd;
    if (nz_u < 1 || nz_u > 18 || nd < 1 || nd > 2) { error[item] = 4875; return; }
    const ulong zbase = ulong(col) * 19ul;
    const ulong vectorbase = item * 18ul;
    const ulong matrixbase = item * 324ul;
    const float width = dxy[item], ws = street_width[item];
    const float hut = z[zbase + uint(nz_u)];
    for (int jz = 0; jz < nz_u; ++jz) {
        for (int iz = 0; iz < nz_u; ++iz) {
            const float f123 = ubmv_fprls(width, fabs(ubmv_sub(z[zbase + uint(jz + 1)], z[zbase + uint(iz)])), ws);
            const float f23 = ubmv_fprls(width, fabs(ubmv_sub(z[zbase + uint(jz + 1)], z[zbase + uint(iz + 1)])), ws);
            const float f12 = ubmv_fprls(width, fabs(ubmv_sub(z[zbase + uint(jz)], z[zbase + uint(iz)])), ws);
            const float f2 = ubmv_fprls(width, fabs(ubmv_sub(z[zbase + uint(jz)], z[zbase + uint(iz + 1)])), ws);
            const float a123 = ubmv_mul(width, fabs(ubmv_sub(z[zbase + uint(jz + 1)], z[zbase + uint(iz)])));
            const float a12 = ubmv_mul(width, fabs(ubmv_sub(z[zbase + uint(jz)], z[zbase + uint(iz)])));
            const float a23 = ubmv_mul(width, fabs(ubmv_sub(z[zbase + uint(jz + 1)], z[zbase + uint(iz + 1)])));
            const float a1 = ubmv_mul(width, fabs(ubmv_sub(z[zbase + uint(iz + 1)], z[zbase + uint(iz)])));
            const float a2 = ubmv_mul(width, fabs(ubmv_sub(z[zbase + uint(jz)], z[zbase + uint(iz + 1)])));
            const float a3 = ubmv_mul(width, fabs(ubmv_sub(z[zbase + uint(jz + 1)], z[zbase + uint(jz)])));
            const float numerator = ubmv_add(ubmv_sub(ubmv_sub(ubmv_mul(a123, f123), ubmv_mul(a23, f23)), ubmv_mul(a12, f12)), ubmv_mul(a2, f2));
            const float ftot = ubmv_div(ubmv_mul(0.5f, numerator), a1);
            fww[matrixbase + uint(iz * 18 + jz)] = ubmv_div(ubmv_mul(ftot, a1), a3);
        }
        const float f12g = ubmv_fnrm(z[zbase + uint(jz + 1)], width, ws);
        const float f1g = ubmv_fnrm(z[zbase + uint(jz)], width, ws);
        const float a = ubmv_mul(ws, width);
        const float area = ubmv_mul(ubmv_sub(z[zbase + uint(jz + 1)], z[zbase + uint(jz)]), width);
        const float ftg = ubmv_div(ubmv_sub(ubmv_mul(a, f12g), ubmv_mul(a, f1g)), a);
        fgw[vectorbase + uint(jz)] = ubmv_div(ubmv_mul(ftg, a), area);

        const float f12s = ubmv_fnrm(ws, dxy[item], ubmv_sub(hut, z[zbase + uint(jz)]));
        const float f1s = ubmv_fnrm(ws, dxy[item], ubmv_sub(hut, z[zbase + uint(jz + 1)]));
        const float a1s = ubmv_mul(ubmv_sub(z[zbase + uint(jz + 1)], z[zbase + uint(jz)]), dxy[item]);
        const float a2s = ubmv_mul(ubmv_sub(hut, z[zbase + uint(jz + 1)]), dxy[item]);
        const float a12s = ubmv_mul(ubmv_sub(hut, z[zbase + uint(jz)]), dxy[item]);
        const float a4s = ubmv_mul(width, dxy[item]);
        const float fts = ubmv_div(ubmv_sub(ubmv_mul(a12s, f12s), ubmv_mul(a2s, f1s)), a1s);
        fws[vectorbase + uint(jz)] = ubmv_div(ubmv_mul(fts, a1s), a4s);

        const float f12gnd = ubmv_fnrm(ws, dxy[item], ubmv_sub(hut, z[zbase + uint(jz)]));
        const float f1gnd = ubmv_fnrm(ws, dxy[item], ubmv_sub(hut, z[zbase + uint(jz + 1)]));
        const float ag = ubmv_mul(ubmv_sub(z[zbase + uint(jz + 1)], z[zbase + uint(jz)]), dxy[item]);
        const float a2g = ubmv_mul(z[zbase + uint(jz)], dxy[item]);
        const float a12g = ubmv_mul(z[zbase + uint(jz + 1)], dxy[item]);
        const float a4g = ubmv_mul(width, dxy[item]);
        const float ftgnd = ubmv_div(ubmv_sub(ubmv_mul(a12g, f12gnd), ubmv_mul(a2g, f1gnd)), ag);
        fwg[vectorbase + uint(jz)] = ubmv_div(ubmv_mul(ftgnd, ag), a4g);

        const float fs12 = ubmv_fnrm(ubmv_sub(hut, z[zbase + uint(jz)]), dxy[item], ws);
        const float fs1 = ubmv_fnrm(ubmv_sub(hut, z[zbase + uint(jz + 1)]), dxy[item], ws);
        const float fsarea = ubmv_mul(ubmv_sub(z[zbase + uint(jz + 1)], z[zbase + uint(jz)]), dxy[item]);
        const float fsa = ubmv_mul(ws, dxy[item]);
        const float fsft = ubmv_div(ubmv_sub(ubmv_mul(fsa, fs12), ubmv_mul(fsa, fs1)), fsa);
        fsw[vectorbase + uint(jz)] = ubmv_div(ubmv_mul(fsft, fsa), fsarea);
    }
    fsg[item] = ubmv_fprls(dxy[item], ws, hut);
    error[item] = 0;
}
