#include <metal_stdlib>
#include "urban_bem_shortwave_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmsw_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmsw_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmsw_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmsw_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }

static inline bool ubmsw_solve(thread float *a, thread float *b, int n) {
    int ipiv[37];
    for (int j = 0; j < n; ++j) ipiv[j] = 0;
    for (int i = 0; i < n; ++i) {
        float big = 0.0f; int irow = -1, icol = -1;
        for (int j = 0; j < n; ++j) if (ipiv[j] != 1)
            for (int k = 0; k < n; ++k) {
                if (ipiv[k] == 0) { const float v = fabs(a[j * 37 + k]); if (v >= big) { big = v; irow = j; icol = k; } }
                else if (ipiv[k] > 1) return false;
            }
        if (irow < 0 || icol < 0) return false;
        ++ipiv[icol];
        if (irow != icol) {
            for (int l = 0; l < n; ++l) { const float t = a[irow * 37 + l]; a[irow * 37 + l] = a[icol * 37 + l]; a[icol * 37 + l] = t; }
            const float t = b[irow]; b[irow] = b[icol]; b[icol] = t;
        }
        const float pivot = a[icol * 37 + icol];
        if (pivot == 0.0f) return false;
        const float inv = ubmsw_div(1.0f, pivot);
        a[icol * 37 + icol] = 1.0f;
        for (int l = 0; l < n; ++l) a[icol * 37 + l] = ubmsw_mul(a[icol * 37 + l], inv);
        b[icol] = ubmsw_mul(b[icol], inv);
        for (int ll = 0; ll < n; ++ll) if (ll != icol) {
            const float d = a[ll * 37 + icol]; a[ll * 37 + icol] = 0.0f;
            for (int l = 0; l < n; ++l) a[ll * 37 + l] = ubmsw_sub(a[ll * 37 + l], ubmsw_mul(a[icol * 37 + l], d));
            b[ll] = ubmsw_sub(b[ll], ubmsw_mul(b[icol], d));
        }
    }
    return true;
}

// WRF module_sf_bep_bem.F bb_short_rad_dd/bb_gaussj (3551-3662, 3670+).
// Solves diffuse shortwave exchange across two canyon walls and the ground.
// A thread handles one column and street direction. Needs integration.
kernel void urban_bem_shortwave_fragment(
    constant UrbanBemShortwaveFragmentResources &r [[buffer(0)]],
    constant UrbanBemShortwaveFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    device const float *pb = r.pb;
    device const int *utype = r.utype;
    device const int *class_int = r.class_int;
    device const float *class_float = r.class_float;
    device const int *nzu_class = r.nzu_class;
    device const float *fwg = r.fwg;
    device const float *fww = r.fww;
    device const float *fgw = r.fgw;
    device const float *fsw = r.fsw;
    device const float *fsg = r.fsg;
    device float *rsw = r.rsw;
    device float *rsg = r.rsg;
    device const float *rsdif = r.rsdif;
    device int *error = r.error;
    const int ncol = p.ncol;
    const int nurbm = p.nurbm;
    if (item >= uint(ncol * 2)) return;
    const int col = int(item) / 2, dir = int(item) % 2;
    error[item] = 0;
    const int type = utype[col];
    if (nurbm < 1 || nurbm > 11 || type < 1 || type > nurbm) { error[item] = 3604; return; }
    const int nz_u = nzu_class[uint(type - 1)];
    const int nd = class_int[uint(type)];
    if (nz_u < 1 || nz_u > 18 || nd < 1 || nd > 2) { error[item] = 3604; return; }
    if (dir >= nd) return;
    const float pwin = class_float[1084u + uint(type - 1)];
    const float albwin = class_float[1051u + uint(type - 1)];
    const float albw = ubmsw_add(ubmsw_mul(pwin, albwin),
        ubmsw_mul(ubmsw_sub(1.0f, pwin), class_float[76u + uint(type - 1)]));
    const float albg = class_float[65u + uint(type - 1)];
    const ulong pbase = ulong(col) * 19ul;
    const ulong vbase = (ulong(col) * 2ul + uint(dir)) * 18ul;
    const ulong wbase = ulong(col) * 72ul + uint(dir * 2) * 18ul;
    const ulong ffbase = (ulong(col) * 2ul + uint(dir)) * 324ul;
    const int n = 2 * nz_u + 1, ground = 2 * nz_u;
    float a[37 * 37], b[37];
    for (int q = 0; q < n * 37; ++q) a[q] = 0.0f;
    for (int i = 0; i < nz_u; ++i) {
        a[i * 37 + i] = 1.0f;
        for (int j = 0; j < nz_u; ++j)
            a[i * 37 + nz_u + j] = ubmsw_mul(ubmsw_mul(-albw,
                fww[ffbase + ulong(j) * 18ul + uint(i)]), pb[pbase + uint(j + 1)]);
        a[i * 37 + ground] = ubmsw_mul(-albg, fgw[vbase + uint(i)]);
        b[i] = ubmsw_add(rsw[wbase + uint(i)], ubmsw_mul(fsw[vbase + uint(i)], rsdif[col]));
    }
    for (int i = 0; i < nz_u; ++i) {
        const int row = nz_u + i;
        for (int j = 0; j < nz_u; ++j)
            a[row * 37 + j] = ubmsw_mul(ubmsw_mul(-albw,
                fww[ffbase + ulong(j) * 18ul + uint(i)]), pb[pbase + uint(j + 1)]);
        a[row * 37 + row] = 1.0f;
        a[row * 37 + ground] = ubmsw_mul(-albg, fgw[vbase + uint(i)]);
        b[row] = ubmsw_add(rsw[wbase + 18ul + uint(i)], ubmsw_mul(fsw[vbase + uint(i)], rsdif[col]));
    }
    for (int j = 0; j < nz_u; ++j) {
        const float vf = fwg[vbase + uint(j)];
        a[ground * 37 + j] = ubmsw_mul(ubmsw_mul(-albw, vf), pb[pbase + uint(j + 1)]);
        a[ground * 37 + nz_u + j] = ubmsw_mul(ubmsw_mul(-albw, vf), pb[pbase + uint(j + 1)]);
    }
    a[ground * 37 + ground] = 1.0f;
    b[ground] = ubmsw_add(rsg[ulong(col) * 2ul + uint(dir)], ubmsw_mul(fsg[ulong(col) * 2ul + uint(dir)], rsdif[col]));
    if (!ubmsw_solve(a, b, n)) { error[item] = 3649; return; }
    for (int i = 0; i < nz_u; ++i) {
        rsw[wbase + uint(i)] = b[i];
        rsw[wbase + 18ul + uint(i)] = b[nz_u + i];
    }
    rsg[ulong(col) * 2ul + uint(dir)] = b[ground];
    error[item] = 0;
}
