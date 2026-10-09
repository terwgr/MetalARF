#include <metal_stdlib>
#include "urban_bem_longwave_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmlw_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmlw_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmlw_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmlw_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmlw_pow4(float x) { float x2 = ubmlw_mul(x, x); return ubmlw_mul(x2, x2); }

static inline bool ubmlw_solve(thread float *a, thread float *b, int n) {
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
        const float inv = ubmlw_div(1.0f, pivot);
        a[icol * 37 + icol] = 1.0f;
        for (int l = 0; l < n; ++l) a[icol * 37 + l] = ubmlw_mul(a[icol * 37 + l], inv);
        b[icol] = ubmlw_mul(b[icol], inv);
        for (int ll = 0; ll < n; ++ll) if (ll != icol) {
            const float d = a[ll * 37 + icol]; a[ll * 37 + icol] = 0.0f;
            for (int l = 0; l < n; ++l) a[ll * 37 + l] = ubmlw_sub(a[ll * 37 + l], ubmlw_mul(a[icol * 37 + l], d));
            b[ll] = ubmlw_sub(b[ll], ubmlw_mul(b[icol], d));
        }
    }
    return true;
}

// WRF module_sf_bep_bem.F bb_long_rad/bb_gaussj (3455-3544, 3670+).
// Window-aware longwave radiosity for two walls plus ground. Inputs use
// per-column/per-direction packed canyon view factors and temperatures.
// Source-only; needs integration with bb_modif_rad and the column caller.
kernel void urban_bem_longwave_fragment(
    constant UrbanBemLongwaveFragmentResources &r [[buffer(0)]],
    constant UrbanBemLongwaveFragmentParams &p [[buffer(1)]],
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
    device const float *tg_av = r.tg_av;
    device const float *tw = r.tw;
    device const float *twlev = r.twlev;
    device const float *rl_field = r.rl;
    device float *rlg = r.rlg;
    device float *rlw = r.rlw;
    device int *error = r.error;
    const int ncol = p.ncol;
    const float sigma = p.sigma;
    const int nurbm = p.nurbm;
    if (item >= uint(ncol * 2)) return;
    const int col = int(item) / 2, dir = int(item) % 2;
    error[item] = 0;
    const int type = utype[col];
    if (nurbm < 1 || nurbm > 11 || type < 1 || type > nurbm) { error[item] = 3455; return; }
    const int nz_u = nzu_class[uint(type - 1)];
    const int nd = class_int[uint(type)];
    if (nz_u < 1 || nz_u > 18 || nd < 1 || nd > 2) { error[item] = 3455; return; }
    if (dir >= nd) return;
    const float emw = class_float[109u + uint(type - 1)];
    const float emg = class_float[98u + uint(type - 1)];
    const float emwin = class_float[1062u + uint(type - 1)];
    const float pwin = class_float[1084u + uint(type - 1)];
    const float rl = rl_field[col];
    const ulong pbase = ulong(col) * 19ul;
    const ulong vbase = (ulong(col) * 2ul + uint(dir)) * 18ul;
    const ulong walls = ulong(col) * 72ul + uint(dir * 2) * 18ul;
    const ulong views = (ulong(col) * 2ul + uint(dir)) * 324ul;
    const int n = 2 * nz_u + 1, ground = 2 * nz_u;
    const float refl = ubmlw_sub(ubmlw_sub(1.0f,
        ubmlw_mul(emw, ubmlw_sub(1.0f, pwin))), ubmlw_mul(emwin, pwin));
    float a[37 * 37], b[37];
    for (int q = 0; q < n * 37; ++q) a[q] = 0.0f;
    for (int i = 0; i < nz_u; ++i) {
        a[i * 37 + i] = 1.0f;
        for (int j = 0; j < nz_u; ++j)
            a[i * 37 + nz_u + j] = ubmlw_mul(ubmlw_mul(-refl,
                fww[views + ulong(j) * 18ul + uint(i)]), pb[pbase + uint(j + 1)]);
        a[i * 37 + ground] = ubmlw_mul(-(1.0f - emg), fgw[vbase + uint(i)]);
        float rhs = ubmlw_add(ubmlw_mul(fsw[vbase + uint(i)], rl),
            ubmlw_mul(ubmlw_mul(ubmlw_mul(emg, fgw[vbase + uint(i)]), sigma), ubmlw_pow4(tg_av[ulong(col) * 2ul + uint(dir)])));
        for (int j = 0; j < nz_u; ++j) {
            const float vf = fww[views + ulong(j) * 18ul + uint(i)];
            const float window_emit = ubmlw_mul(ubmlw_mul(emw, ubmlw_sub(1.0f, pwin)), ubmlw_pow4(tw[walls + 18ul + uint(j)]));
            const float indoor_emit = ubmlw_mul(ubmlw_mul(emwin, pwin), ubmlw_pow4(twlev[walls + 18ul + uint(j)]));
            const float emit = ubmlw_mul(ubmlw_mul(ubmlw_mul(pb[pbase + uint(j + 1)], sigma), vf), ubmlw_add(window_emit, indoor_emit));
            const float transmit = ubmlw_mul(ubmlw_mul(vf, rl), ubmlw_sub(1.0f, pb[pbase + uint(j + 1)]));
            rhs = ubmlw_add(ubmlw_add(rhs, emit), transmit);
        }
        b[i] = rhs;
    }
    for (int i = 0; i < nz_u; ++i) {
        const int row = nz_u + i;
        for (int j = 0; j < nz_u; ++j)
            a[row * 37 + j] = ubmlw_mul(ubmlw_mul(-refl,
                fww[views + ulong(j) * 18ul + uint(i)]), pb[pbase + uint(j + 1)]);
        a[row * 37 + row] = 1.0f;
        a[row * 37 + ground] = ubmlw_mul(-(1.0f - emg), fgw[vbase + uint(i)]);
        float rhs = ubmlw_add(ubmlw_mul(fsw[vbase + uint(i)], rl),
            ubmlw_mul(ubmlw_mul(ubmlw_mul(emg, fgw[vbase + uint(i)]), sigma), ubmlw_pow4(tg_av[ulong(col) * 2ul + uint(dir)])));
        for (int j = 0; j < nz_u; ++j) {
            const float vf = fww[views + ulong(j) * 18ul + uint(i)];
            const float wall_emit = ubmlw_mul(ubmlw_mul(emw, ubmlw_sub(1.0f, pwin)), ubmlw_pow4(tw[walls + uint(j)]));
            const float indoor_emit = ubmlw_mul(ubmlw_mul(emwin, pwin), ubmlw_pow4(twlev[walls + uint(j)]));
            const float emit = ubmlw_mul(ubmlw_mul(ubmlw_mul(pb[pbase + uint(j + 1)], sigma), vf), ubmlw_add(wall_emit, indoor_emit));
            const float transmit = ubmlw_mul(ubmlw_mul(vf, rl), ubmlw_sub(1.0f, pb[pbase + uint(j + 1)]));
            rhs = ubmlw_add(ubmlw_add(rhs, emit), transmit);
        }
        b[row] = rhs;
    }
    for (int j = 0; j < nz_u; ++j) {
        const float vf = fwg[vbase + uint(j)];
        a[ground * 37 + j] = ubmlw_mul(ubmlw_mul(-refl, vf), pb[pbase + uint(j + 1)]);
        a[ground * 37 + nz_u + j] = ubmlw_mul(ubmlw_mul(-refl, vf), pb[pbase + uint(j + 1)]);
    }
    a[ground * 37 + ground] = 1.0f;
    float rhs = ubmlw_mul(fsg[ulong(col) * 2ul + uint(dir)], rl);
    for (int j = 0; j < nz_u; ++j) {
        const float vf = fwg[vbase + uint(j)];
        const float walls_emit = ubmlw_mul(ubmlw_mul(emw, ubmlw_sub(1.0f, pwin)),
            ubmlw_add(ubmlw_pow4(tw[walls + uint(j)]), ubmlw_pow4(tw[walls + 18ul + uint(j)])));
        const float indoor_emit = ubmlw_mul(ubmlw_mul(emwin, pwin),
            ubmlw_add(ubmlw_pow4(twlev[walls + uint(j)]), ubmlw_pow4(twlev[walls + 18ul + uint(j)])));
        const float emit = ubmlw_mul(ubmlw_mul(ubmlw_mul(sigma, vf), pb[pbase + uint(j + 1)]), ubmlw_add(walls_emit, indoor_emit));
        const float transmit = ubmlw_mul(ubmlw_mul(ubmlw_mul(2.0f, vf), ubmlw_sub(1.0f, pb[pbase + uint(j + 1)])), rl);
        rhs = ubmlw_add(ubmlw_add(rhs, emit), transmit);
    }
    b[ground] = rhs;
    if (!ubmlw_solve(a, b, n)) { error[item] = 3532; return; }
    for (int i = 0; i < nz_u; ++i) {
        rlw[walls + uint(i)] = b[i];
        rlw[walls + 18ul + uint(i)] = b[nz_u + i];
    }
    rlg[ulong(col) * 2ul + uint(dir)] = b[ground];
    error[item] = 0;
}
