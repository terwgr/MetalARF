#include <metal_stdlib>
#include "urban_bem_interior_radiation_fragment_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_binary64.metalh"
#include "soft_gfk_atan.metalh"
#include "soft_gfk_log.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmir_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmir_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmir_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmir_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmir_sqrt(float a) { return as_type<float>(soft32_sqrt(as_type<uint>(a))); }
static inline float ubmir_atan(float a) { return as_type<float>(soft_gfk_atanf(as_type<uint>(a))); }
static inline float ubmir_log(float a) { return as_type<float>(soft_gfk_log(as_type<uint>(a))); }
static inline float ubmir_pow4(float a) { const float h = ubmir_mul(a, a); return ubmir_mul(h, h); }

// WRF module_sf_bem.F bem_fprl_ints (2601) and bem_fnrm_ints (2625).
// The operation nesting follows the source's explicit RN binary32 operations.
static inline float ubmir_fprl(float vx, float vy) {
    const float vx2 = ubmir_mul(vx, vx), vy2 = ubmir_mul(vy, vy);
    const float sx = ubmir_sqrt(ubmir_add(1.0f, vx2));
    const float sy = ubmir_sqrt(ubmir_add(1.0f, vy2));
    const float denom = ubmir_add(ubmir_add(1.0f, vx2), vy2);
    const float inside = ubmir_div(ubmir_mul(ubmir_add(1.0f, vx2), ubmir_add(1.0f, vy2)), denom);
    float g = ubmir_log(ubmir_sqrt(inside));
    g = ubmir_add(g, ubmir_mul(ubmir_mul(vy, sx), ubmir_atan(ubmir_div(vy, sx))));
    g = ubmir_add(g, ubmir_mul(ubmir_mul(vx, sy), ubmir_atan(ubmir_div(vx, sy))));
    g = ubmir_sub(g, ubmir_mul(vy, ubmir_atan(vy)));
    g = ubmir_sub(g, ubmir_mul(vx, ubmir_atan(vx)));
    return ubmir_mul(ubmir_div(2.0f, ubmir_mul(ubmir_mul(3.141592653f, vx), vy)), g);
}

static inline float ubmir_fnrm(float wx, float wy, float wz) {
    const float wx2 = ubmir_mul(wx, wx), wy2 = ubmir_mul(wy, wy);
    const float rootz = ubmir_sqrt(wz);
    float g = ubmir_add(ubmir_mul(wy, ubmir_atan(ubmir_div(1.0f, wy))),
                        ubmir_mul(wx, ubmir_atan(ubmir_div(1.0f, wx))));
    g = ubmir_sub(g, ubmir_mul(rootz, ubmir_atan(ubmir_div(1.0f, rootz))));
    const float log0 = ubmir_log(ubmir_div(ubmir_mul(ubmir_add(1.0f, wx2), ubmir_add(1.0f, wy2)), ubmir_add(1.0f, wz)));
    const float logy = ubmir_log(ubmir_div(ubmir_mul(wy2, ubmir_add(1.0f, wz)), ubmir_mul(wz, ubmir_add(1.0f, wy2))));
    const float logx = ubmir_log(ubmir_div(ubmir_mul(wx2, ubmir_add(1.0f, wz)), ubmir_mul(wz, ubmir_add(1.0f, wx2))));
    const float logs = ubmir_add(ubmir_add(log0, ubmir_mul(wy2, logy)), ubmir_mul(wx2, logx));
    g = ubmir_add(g, ubmir_mul(0.25f, logs));
    return ubmir_mul(ubmir_div(1.0f, ubmir_mul(3.141592653f, wy)), g);
}

static inline bool ubmir_solve(thread float *a, thread float *b) {
    int ipiv[6];
    for (int j = 0; j < 6; ++j) ipiv[j] = 0;
    for (int i = 0; i < 6; ++i) {
        float big = 0.0f; int irow = -1, icol = -1;
        for (int j = 0; j < 6; ++j) if (ipiv[j] != 1)
            for (int k = 0; k < 6; ++k) {
                if (ipiv[k] == 0) {
                    const float v = fabs(a[j * 6 + k]);
                    if (v >= big) { big = v; irow = j; icol = k; }
                } else if (ipiv[k] > 1) return false;
            }
        if (irow < 0 || icol < 0) return false;
        ++ipiv[icol];
        if (irow != icol) {
            for (int l = 0; l < 6; ++l) { const float t = a[irow * 6 + l]; a[irow * 6 + l] = a[icol * 6 + l]; a[icol * 6 + l] = t; }
            const float t = b[irow]; b[irow] = b[icol]; b[icol] = t;
        }
        const float pivot = a[icol * 6 + icol];
        if (pivot == 0.0f || !isfinite(pivot)) return false;
        const float inv = ubmir_div(1.0f, pivot);
        a[icol * 6 + icol] = 1.0f;
        for (int l = 0; l < 6; ++l) a[icol * 6 + l] = ubmir_mul(a[icol * 6 + l], inv);
        b[icol] = ubmir_mul(b[icol], inv);
        for (int ll = 0; ll < 6; ++ll) if (ll != icol) {
            const float d = a[ll * 6 + icol]; a[ll * 6 + icol] = 0.0f;
            for (int l = 0; l < 6; ++l) a[ll * 6 + l] = ubmir_sub(a[ll * 6 + l], ubmir_mul(a[icol * 6 + l], d));
            b[ll] = ubmir_sub(b[ll], ubmir_mul(b[icol], d));
        }
    }
    return true;
}

// WRF module_sf_bem.F bem_algebra_short (1422-1516). `a` is row-major here;
// its mathematical entries and all source assignment dependencies are kept.
static inline void ubmir_algebra_short(float rstr, float albwal, float albwin,
                                       float aw, float bw, float zw, float pwin,
                                       thread float *a, thread float *b) {
    for (int i = 0; i < 36; ++i) a[i] = 0.0f;
    for (int i = 0; i < 6; ++i) b[i] = -rstr;
    const float albm = ubmir_add(ubmir_mul(pwin, albwin), ubmir_mul(ubmir_sub(1.0f, pwin), albwal));
    float q;
    a[0] = -1.0f;
    a[1] = ubmir_mul(albm, ubmir_fprl(ubmir_div(aw, bw), ubmir_div(zw, bw)));
    q = ubmir_fnrm(ubmir_div(aw, zw), ubmir_div(bw, zw), ubmir_div(ubmir_add(ubmir_mul(aw, aw), ubmir_mul(bw, bw)), ubmir_mul(zw, zw)));
    a[2] = ubmir_mul(ubmir_mul(albm, ubmir_div(bw, aw)), q); a[3] = a[2];
    q = ubmir_fnrm(ubmir_div(zw, aw), ubmir_div(bw, aw), ubmir_div(ubmir_add(ubmir_mul(bw, bw), ubmir_mul(zw, zw)), ubmir_mul(aw, aw)));
    a[4] = ubmir_mul(ubmir_mul(albwal, ubmir_div(bw, zw)), q); a[5] = a[4];
    a[6] = a[1]; a[7] = -1.0f; a[8] = a[2]; a[9] = a[3]; a[10] = a[4]; a[11] = a[5];
    q = ubmir_fnrm(ubmir_div(bw, zw), ubmir_div(aw, zw), ubmir_div(ubmir_add(ubmir_mul(bw, bw), ubmir_mul(aw, aw)), ubmir_mul(zw, zw)));
    a[12] = ubmir_mul(ubmir_mul(albm, ubmir_div(aw, bw)), q); a[13] = a[12]; a[14] = -1.0f;
    a[15] = ubmir_mul(albm, ubmir_fprl(ubmir_div(zw, aw), ubmir_div(bw, aw)));
    q = ubmir_fnrm(ubmir_div(zw, bw), ubmir_div(aw, bw), ubmir_div(ubmir_add(ubmir_mul(aw, aw), ubmir_mul(zw, zw)), ubmir_mul(bw, bw)));
    a[16] = ubmir_mul(ubmir_mul(albwal, ubmir_div(aw, zw)), q); a[17] = a[16];
    a[18] = a[12]; a[19] = a[13]; a[20] = a[15]; a[21] = -1.0f; a[22] = a[16]; a[23] = a[17];
    q = ubmir_fnrm(ubmir_div(bw, aw), ubmir_div(zw, aw), ubmir_div(ubmir_add(ubmir_mul(bw, bw), ubmir_mul(zw, zw)), ubmir_mul(aw, aw)));
    a[24] = ubmir_mul(ubmir_mul(albm, ubmir_div(zw, bw)), q); a[25] = a[24];
    q = ubmir_fnrm(ubmir_div(aw, bw), ubmir_div(zw, bw), ubmir_div(ubmir_add(ubmir_mul(aw, aw), ubmir_mul(zw, zw)), ubmir_mul(bw, bw)));
    a[26] = ubmir_mul(ubmir_mul(albm, ubmir_div(zw, aw)), q); a[27] = a[26]; a[28] = -1.0f;
    a[29] = ubmir_mul(albwal, ubmir_fprl(ubmir_div(aw, zw), ubmir_div(bw, zw)));
    a[30] = a[24]; a[31] = a[24]; a[32] = a[26]; a[33] = a[27]; a[34] = a[29]; a[35] = -1.0f;
}

// WRF module_sf_bem.F bem_algebra_long (1579-1820), expressed with source
// 1-based wall indices translated to zero-based packed arrays.
static inline void ubmir_algebra_long(float emwal, float emwin, float sigma,
                                      thread const float *twal, thread const float *twin,
                                      float pwin, float aw, float bw, float zw,
                                      thread float *a, thread float *b) {
    float wall[6], wind[6], t4w[6], t4n[4];
    for (int i = 0; i < 36; ++i) a[i] = 0.0f;
    for (int i = 0; i < 6; ++i) { wall[i] = 0.0f; wind[i] = 0.0f; t4w[i] = ubmir_pow4(twal[i]); }
    for (int i = 0; i < 4; ++i) t4n[i] = ubmir_pow4(twin[i]);
    const float emwal_av = ubmir_mul(ubmir_sub(1.0f, pwin), emwal);
    const float emwin_av = ubmir_mul(pwin, emwin);
    const float em_av = ubmir_add(emwal_av, emwin_av);
    float fp, fx, fy, sum;

    // Source rows 1 and 2: wall 5/6 facing the opposite wall pair and windows.
    fp = ubmir_fprl(ubmir_div(aw, zw), ubmir_div(bw, zw));
    fx = ubmir_fnrm(ubmir_div(aw, bw), ubmir_div(zw, bw), ubmir_div(ubmir_add(ubmir_mul(aw, aw), ubmir_mul(zw, zw)), ubmir_mul(bw, bw)));
    fy = ubmir_fnrm(ubmir_div(bw, aw), ubmir_div(zw, aw), ubmir_div(ubmir_add(ubmir_mul(bw, bw), ubmir_mul(zw, zw)), ubmir_mul(aw, aw)));
    for (int row = 0; row < 2; ++row) {
        const int wall_id = 4 + row;
        const float term1 = ubmir_mul(ubmir_mul(ubmir_mul(emwal, sigma), t4w[wall_id]), fp);
        sum = ubmir_add(ubmir_mul(emwal_av, t4w[2]), ubmir_mul(emwal_av, t4w[3]));
        const float term2 = ubmir_mul(ubmir_mul(ubmir_mul(sigma, sum), ubmir_div(zw, aw)), fx);
        sum = ubmir_add(ubmir_mul(emwal_av, t4w[0]), ubmir_mul(emwal_av, t4w[1]));
        const float term3 = ubmir_mul(ubmir_mul(ubmir_mul(sigma, sum), ubmir_div(zw, bw)), fy);
        wall[row] = ubmir_add(ubmir_add(term1, term2), term3);
    }
    // Rows 3 and 4.
    fp = ubmir_fprl(ubmir_div(zw, aw), ubmir_div(bw, aw));
    fx = ubmir_fnrm(ubmir_div(bw, zw), ubmir_div(aw, zw), ubmir_div(ubmir_add(ubmir_mul(bw, bw), ubmir_mul(aw, aw)), ubmir_mul(zw, zw)));
    fy = ubmir_fnrm(ubmir_div(zw, bw), ubmir_div(aw, bw), ubmir_div(ubmir_add(ubmir_mul(aw, aw), ubmir_mul(zw, zw)), ubmir_mul(bw, bw)));
    for (int row = 2; row < 4; ++row) {
        const int wall_id = row == 2 ? 3 : 2;
        const float term1 = ubmir_mul(ubmir_mul(ubmir_mul(emwal_av, sigma), t4w[wall_id]), fp);
        sum = ubmir_add(ubmir_mul(emwal_av, t4w[1]), ubmir_mul(emwal_av, t4w[0]));
        const float term2 = ubmir_mul(ubmir_mul(ubmir_mul(sigma, sum), ubmir_div(aw, bw)), fx);
        sum = ubmir_add(ubmir_mul(emwal, t4w[4]), ubmir_mul(emwal, t4w[5]));
        const float term3 = ubmir_mul(ubmir_mul(ubmir_mul(sigma, sum), ubmir_div(aw, zw)), fy);
        wall[row] = ubmir_add(ubmir_add(term1, term2), term3);
    }
    // Rows 5 and 6.
    fp = ubmir_fprl(ubmir_div(aw, bw), ubmir_div(zw, bw));
    fx = ubmir_fnrm(ubmir_div(aw, zw), ubmir_div(bw, zw), ubmir_div(ubmir_add(ubmir_mul(aw, aw), ubmir_mul(bw, bw)), ubmir_mul(zw, zw)));
    fy = ubmir_fnrm(ubmir_div(zw, aw), ubmir_div(bw, aw), ubmir_div(ubmir_add(ubmir_mul(bw, bw), ubmir_mul(zw, zw)), ubmir_mul(aw, aw)));
    for (int row = 4; row < 6; ++row) {
        const int wall_id = row == 4 ? 1 : 0;
        const float term1 = ubmir_mul(ubmir_mul(ubmir_mul(emwal_av, sigma), t4w[wall_id]), fp);
        sum = ubmir_add(ubmir_mul(emwal_av, t4w[2]), ubmir_mul(emwal_av, t4w[3]));
        const float term2 = ubmir_mul(ubmir_mul(ubmir_mul(sigma, sum), ubmir_div(bw, aw)), fx);
        sum = ubmir_add(ubmir_mul(emwal, t4w[4]), ubmir_mul(emwal, t4w[5]));
        const float term3 = ubmir_mul(ubmir_mul(ubmir_mul(sigma, sum), ubmir_div(bw, zw)), fy);
        wall[row] = ubmir_add(ubmir_add(term1, term2), term3);
    }

    // Window-source terms for rows 1 and 2.
    fx = ubmir_fnrm(ubmir_div(aw, bw), ubmir_div(zw, bw), ubmir_div(ubmir_add(ubmir_mul(aw, aw), ubmir_mul(zw, zw)), ubmir_mul(bw, bw)));
    fy = ubmir_fnrm(ubmir_div(bw, aw), ubmir_div(zw, aw), ubmir_div(ubmir_add(ubmir_mul(bw, bw), ubmir_mul(zw, zw)), ubmir_mul(aw, aw)));
    for (int row = 0; row < 2; ++row) {
        sum = ubmir_add(ubmir_mul(emwin_av, t4n[2]), ubmir_mul(emwin_av, t4n[3]));
        const float term1 = ubmir_mul(ubmir_mul(ubmir_mul(sigma, sum), ubmir_div(zw, aw)), fx);
        sum = ubmir_add(ubmir_mul(emwin_av, t4n[0]), ubmir_mul(emwin_av, t4n[1]));
        const float term2 = ubmir_mul(ubmir_mul(ubmir_mul(sigma, sum), ubmir_div(zw, bw)), fy);
        wind[row] = ubmir_add(term1, term2);
    }
    // Window-source terms for rows 3 and 4.
    fp = ubmir_fprl(ubmir_div(zw, aw), ubmir_div(bw, aw));
    fx = ubmir_fnrm(ubmir_div(bw, zw), ubmir_div(aw, zw), ubmir_div(ubmir_add(ubmir_mul(aw, aw), ubmir_mul(bw, bw)), ubmir_mul(zw, zw)));
    for (int row = 2; row < 4; ++row) {
        const int window_id = row == 2 ? 3 : 2;
        const float term1 = ubmir_mul(ubmir_mul(ubmir_mul(emwin_av, sigma), t4n[window_id]), fp);
        sum = ubmir_add(ubmir_mul(emwin_av, t4n[1]), ubmir_mul(emwin_av, t4n[0]));
        const float term2 = ubmir_mul(ubmir_mul(ubmir_mul(sigma, sum), ubmir_div(aw, bw)), fx);
        wind[row] = ubmir_add(term1, term2);
    }
    // Window-source terms for rows 5 and 6.
    fp = ubmir_fprl(ubmir_div(aw, bw), ubmir_div(zw, bw));
    fy = ubmir_fnrm(ubmir_div(aw, zw), ubmir_div(bw, zw), ubmir_div(ubmir_add(ubmir_mul(aw, aw), ubmir_mul(bw, bw)), ubmir_mul(zw, zw)));
    for (int row = 4; row < 6; ++row) {
        const int window_id = row == 4 ? 1 : 0;
        const float term1 = ubmir_mul(ubmir_mul(ubmir_mul(emwin_av, sigma), t4n[window_id]), fp);
        sum = ubmir_add(ubmir_mul(emwin_av, t4n[2]), ubmir_mul(emwin_av, t4n[3]));
        const float term2 = ubmir_mul(ubmir_mul(ubmir_mul(sigma, sum), ubmir_div(bw, aw)), fy);
        wind[row] = ubmir_add(term1, term2);
    }
    for (int i = 0; i < 6; ++i) b[i] = ubmir_add(wall[i], wind[i]);

    // Interior radiosity coefficient matrix, source rows 1-6.
    fy = ubmir_fnrm(ubmir_div(bw, aw), ubmir_div(zw, aw), ubmir_div(ubmir_add(ubmir_mul(bw, bw), ubmir_mul(zw, zw)), ubmir_mul(aw, aw)));
    const float m11 = ubmir_mul(ubmir_mul(ubmir_sub(em_av, 1.0f), ubmir_div(zw, bw)), fy);
    fx = ubmir_fnrm(ubmir_div(aw, bw), ubmir_div(zw, bw), ubmir_div(ubmir_add(ubmir_mul(aw, aw), ubmir_mul(zw, zw)), ubmir_mul(bw, bw)));
    const float m13 = ubmir_mul(ubmir_mul(ubmir_sub(em_av, 1.0f), ubmir_div(zw, aw)), fx);
    fp = ubmir_fprl(ubmir_div(aw, zw), ubmir_div(bw, zw));
    const float m15 = ubmir_mul(ubmir_sub(emwal, 1.0f), fp);
    a[0]=m11; a[1]=m11; a[2]=m13; a[3]=m13; a[4]=m15; a[5]=1.0f;
    a[6]=m11; a[7]=m11; a[8]=m13; a[9]=m13; a[10]=1.0f; a[11]=m15;

    fx = ubmir_fnrm(ubmir_div(bw, zw), ubmir_div(aw, zw), ubmir_div(ubmir_add(ubmir_mul(bw, bw), ubmir_mul(aw, aw)), ubmir_mul(zw, zw)));
    const float m31 = ubmir_mul(ubmir_mul(ubmir_sub(em_av, 1.0f), ubmir_div(aw, bw)), fx);
    fp = ubmir_fprl(ubmir_div(zw, aw), ubmir_div(bw, aw));
    const float m34 = ubmir_mul(ubmir_sub(em_av, 1.0f), fp);
    fy = ubmir_fnrm(ubmir_div(zw, bw), ubmir_div(aw, bw), ubmir_div(ubmir_add(ubmir_mul(aw, aw), ubmir_mul(zw, zw)), ubmir_mul(bw, bw)));
    const float m35 = ubmir_mul(ubmir_mul(ubmir_sub(emwal, 1.0f), ubmir_div(aw, zw)), fy);
    a[12]=m31; a[13]=m31; a[14]=1.0f; a[15]=m34; a[16]=m35; a[17]=m35;
    a[18]=m31; a[19]=m31; a[20]=m34; a[21]=1.0f; a[22]=m35; a[23]=m35;

    fp = ubmir_fprl(ubmir_div(aw, bw), ubmir_div(zw, bw));
    const float m52 = ubmir_mul(ubmir_sub(em_av, 1.0f), fp);
    fx = ubmir_fnrm(ubmir_div(aw, zw), ubmir_div(bw, zw), ubmir_div(ubmir_add(ubmir_mul(aw, aw), ubmir_mul(bw, bw)), ubmir_mul(zw, zw)));
    const float m53 = ubmir_mul(ubmir_mul(ubmir_sub(em_av, 1.0f), ubmir_div(bw, aw)), fx);
    fy = ubmir_fnrm(ubmir_div(zw, aw), ubmir_div(bw, aw), ubmir_div(ubmir_add(ubmir_mul(bw, bw), ubmir_mul(zw, zw)), ubmir_mul(aw, aw)));
    const float m55 = ubmir_mul(ubmir_mul(ubmir_sub(emwal, 1.0f), ubmir_div(bw, zw)), fy);
    a[24]=1.0f; a[25]=m52; a[26]=m53; a[27]=m53; a[28]=m55; a[29]=m55;
    a[30]=m52; a[31]=1.0f; a[32]=m53; a[33]=m53; a[34]=m55; a[35]=m55;
}

// WRF module_sf_bem.F bem_int_rsrad/bem_int_rlrad, with nested algebra and
// Gauss-Jordan solves. One item is one building column at one indoor level;
// all surface arrays are packed in WRF's 4-window/6-wall order. Geometry must
// be finite and strictly positive, pwin is [0,1], and surface areas must sum
// to a positive value. gfk_log/atan use the repository's pinned soft helpers.
kernel void urban_bem_interior_radiation_fragment(
    constant UrbanBemInteriorRadiationFragmentResources &r [[buffer(0)]],
    constant UrbanBemInteriorRadiationFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    device const float *rswal = r.rswal;
    device const float *surwal = r.surwal;
    device const float *twal_int = r.twal_int;
    device const float *twin = r.twin;
    device float *rsint = r.rsint;
    device float *rlint = r.rlint;
    device int *error = r.error;
    if (item >= uint(p.nitem)) return;
    const float aw = p.bw, bw = p.bl, zw = p.zw, pwin = p.pwin;
    if (!(aw > 0.0f && bw > 0.0f && zw > 0.0f) || !isfinite(aw) || !isfinite(bw) || !isfinite(zw) ||
        !(pwin >= 0.0f && pwin <= 1.0f) || !isfinite(pwin)) {
        error[item] = 3812;
        return;
    }
    const ulong base4 = ulong(item) * 4ul, base6 = ulong(item) * 6ul;
    float area_total = 0.0f, rstr = 0.0f;
    const float transmit = ubmir_sub(1.0f, p.albwin);
    for (uint i = 0; i < 4; ++i) {
        const float term = ubmir_mul(ubmir_mul(surwal[base6 + i], pwin),
                                     ubmir_mul(transmit, rswal[base4 + i]));
        rstr = ubmir_add(rstr, term);
    }
    for (uint i = 0; i < 6; ++i) area_total = ubmir_add(area_total, surwal[base6 + i]);
    if (!(area_total > 0.0f) || !isfinite(area_total)) { error[item] = 3813; return; }
    rstr = ubmir_div(rstr, area_total);

    float asw[36], bsw[6];
    ubmir_algebra_short(rstr, p.albwal, p.albwin, aw, bw, zw, pwin, asw, bsw);
    if (!ubmir_solve(asw, bsw)) { error[item] = 3814; return; }

    float alw[36], blw[6];
    thread float local_wall[6], local_window[4];
    for (uint i = 0; i < 6; ++i) local_wall[i] = twal_int[base6 + i];
    for (uint i = 0; i < 4; ++i) local_window[i] = twin[base4 + i];
    ubmir_algebra_long(p.emwal, p.emwin, p.sigma, local_wall, local_window, pwin, aw, bw, zw, alw, blw);
    if (!ubmir_solve(alw, blw)) { error[item] = 3815; return; }
    for (uint i = 0; i < 6; ++i) { rsint[base6 + i] = bsw[i]; rlint[base6 + i] = blw[i]; }
    error[item] = 0;
}
