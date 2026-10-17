#include <metal_stdlib>
#include "rtt_background_bindings.metalh"
#include "rtt_adapt_bindings.metalh"
#include "rtt_pbl_height_bindings.metalh"
#include "rtt_cone_fill_bindings.metalh"
#include "rtt_probable_convection_bindings.metalh"
#include "rtt_tendency_bindings.metalh"
#include "rtt_smooth_interior_bindings.metalh"
#include "rtt_smooth_edges_bindings.metalh"
#include "rtt_store_bindings.metalh"
#include "rtt_column_flag_bindings.metalh"
#include "rtt_vinterp_bindings.metalh"
#include "rtt_vinterp_edges_bindings.metalh"
#include "rtt_apply_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_binary64.metalh"
#include "soft_glibc_pow64.metalh"

using namespace metal;

// Port of the radar latent heating kernels in woof/da/radar_tten.py.  Float32
// nodes are single round-to-nearest operations (contraction off, division
// through soft32_div); every CUDA double node is the matching soft64
// operation on binary64 bits, and glibc_pow is soft_glibc_pow64.  Double
// arrays and scalars arrive as binary64 words.
#pragma clang fp contract(off)

static inline float fdiv_rn(float a, float b) {
    return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)));
}
static inline ulong d_of(float v) { return soft64_from_f32_bits(as_type<uint>(v)); }
static inline float f_of(ulong d) { return as_type<float>(soft64_to_f32_bits(d)); }
static inline ulong d_neg(ulong a) { return a ^ soft64_sign_mask(); }
static inline ulong d_abs(ulong a) { return a & ~soft64_sign_mask(); }
static inline ulong d_fmax(ulong a, ulong b) {
    if (soft64_is_nan(a)) return b;
    if (soft64_is_nan(b)) return a;
    return soft64_less(a, b) ? b : a;
}
static inline ulong d_fmin(ulong a, ulong b) {
    if (soft64_is_nan(a)) return b;
    if (soft64_is_nan(b)) return a;
    return soft64_less(b, a) ? b : a;
}
// Binary64 constants that are not exact binary32 values.
#define RTT_D_0_1 0x3fb999999999999aul
#define RTT_D_M9999_9 0xc0c387f333333333ul

static inline float rtt_thetav(float q, float t) {
    const float qsp = fdiv_rn(q, 1.0f + q);
    return t * (1.0f + (0.61f * qsp));
}

static inline float rtt_temperature(float theta, float p_hpa, ulong rd_over_cp) {
    const ulong ratio = soft64_div(d_of(p_hpa), d_of(1000.0f));
    return f_of(soft64_mul(d_of(theta), soft_glibc_pow64(ratio, rd_over_cp)));
}

static inline int rtt_krad_bot(float krad_bot_in, float pblh) {
    return int(fmax(krad_bot_in, pblh) + 0.5f);
}

kernel void rtt_background(constant RttBackgroundResources &r [[buffer(0)]],
                           constant RttBackgroundParams &p [[buffer(1)]],
                           uint tid [[thread_position_in_grid]]) {
    const ulong ncol = ulong(p.ny) * ulong(p.nx);
    const ulong n = ulong(p.nz) * ncol;
    const ulong idx = ulong(tid);
    if (idx >= n) return;
    const ulong k = idx / ncol;
    const ulong col = idx - k * ncol;
    r.theta[idx] = r.thb[p.thb_full != 0 ? idx : k] + r.thp[idx];
    r.p_hpa[idx] = fdiv_rn(r.p[idx], 100.0f);
    const float z0 = fdiv_rn(r.phb[p.phb_full != 0 ? col : 0ul] + r.php[col], p.g);
    const float zlo = fdiv_rn(r.phb[p.phb_full != 0 ? idx : k] + r.php[idx], p.g);
    const float zhi = fdiv_rn(
        r.phb[p.phb_full != 0 ? idx + ncol : k + 1] + r.php[idx + ncol], p.g);
    r.h_agl[idx] = (0.5f * (zlo + zhi)) - z0;
}

kernel void rtt_adapt(constant RttAdaptResources &r [[buffer(0)]],
                      constant RttAdaptParams &p [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {
    const ulong n = ulong(p.nz) * ulong(p.ny) * ulong(p.nx);
    if (ulong(tid) >= n) return;
    float out = -99999.0f;
    if (r.z_mask[tid] != 0) out = r.z_obs[tid];
    else if (p.has_clear != 0 && r.z0_mask[tid] != 0) out = -99.0f;
    r.ref[tid] = out;
}

kernel void rtt_pbl_height(constant RttPblHeightResources &r [[buffer(0)]],
                           constant RttPblHeightParams &p [[buffer(1)]],
                           uint tid [[thread_position_in_grid]]) {
    const int ncol = p.ny * p.nx;
    const int col = int(tid);
    if (col >= ncol) return;
    const float thsfc = rtt_thetav(r.q[col], r.t[col]);
    const float lim = thsfc + 1.0f;
    float out = 0.0f;
    int k = 1;
    while (fabs(out) < 0.0001f && k < p.nz - 2) {
        const ulong at = ulong(k - 1) * ulong(ncol) + ulong(col);
        const float thk = rtt_thetav(r.q[at], r.t[at]);
        if (thk > lim) {
            const float thm = rtt_thetav(r.q[at - ulong(ncol)], r.t[at - ulong(ncol)]);
            out = float(k) - fdiv_rn(thk - lim, fmax(thk - thm, 0.01f));
        }
        k += 1;
    }
    if (fabs(out) < 0.0001f) out = 2.0f;
    r.pblh[col] = out;
}

static inline ulong rtt_cone_profile(float height, ulong column_max, int mref,
                                     device const ulong *levels,
                                     device const ulong *profiles) {
    const ulong hg = d_of(height);
    if (!(soft64_greater_equal(hg, levels[2]) && soft64_less(hg, d_of(12000.0f))))
        return RTT_D_M9999_9;
    int ilvl = 0;
    for (int k = 0; k < 30; ++k)
        if (soft64_greater_equal(hg, levels[k]) && soft64_less(hg, levels[k + 1]))
            ilvl = k;
    device const ulong *row = profiles + (mref - 1) * 31;
    const ulong upref = soft64_mul(row[ilvl + 1], column_max);
    const ulong downref = soft64_mul(row[ilvl], column_max);
    const ulong wght = soft64_div(soft64_sub(hg, levels[ilvl]),
                                  soft64_sub(levels[ilvl + 1], levels[ilvl]));
    return soft64_add(soft64_mul(soft64_sub(d_of(1.0f), wght), downref),
                      soft64_mul(wght, upref));
}

kernel void rtt_cone_fill(constant RttConeFillResources &r [[buffer(0)]],
                          constant RttConeFillParams &p [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {
    const int nz = p.nz, nx = p.nx;
    const int ncol = p.ny * nx;
    const int col = int(tid);
    if (col >= ncol) return;
    const int j = col / nx, i = col - j * nx;
    if (i < 1 || i > nx - 2 || j < 1 || j > p.ny - 2) return;
#define RTT_REF(kf) r.ref[ulong((kf) - 1) * ulong(ncol) + ulong(col)]
#define RTT_HGT(kf) r.h[ulong((kf) - 1) * ulong(ncol) + ulong(col)]
    const int krad_bot = int(r.pblh[col] + 0.5f);
    const int klo = krad_bot < 1 ? 1 : krad_bot;
    const int khi = krad_bot > nz ? nz : krad_bot;
    int ifmissing = 0;
    ulong maxref = d_of(-9999.0f);
    for (int k2 = nz / 2; k2 >= klo; --k2) {
        if (RTT_REF(k2 + 1) >= 20.0f && RTT_REF(k2) < -100.0f) ifmissing = k2;
        if (soft64_greater_equal(d_of(RTT_REF(k2)), maxref)) maxref = d_of(RTT_REF(k2));
    }
    if (ifmissing > 1) {
        for (int k2 = khi; k2 >= 1; --k2)
            if (soft64_greater(d_of(RTT_REF(k2)), maxref)) maxref = d_of(RTT_REF(k2));
    }
    if (ifmissing > 1 && soft64_greater(maxref, d_of(19.0f))) {
        int mref = soft64_to_i32_rz(soft64_div(soft64_sub(maxref, d_of(20.0f)),
                                               d_of(5.0f))).value + 1;
        if (mref > 6) mref = 6;
        if (mref < 1) mref = 1;
        const float above = RTT_REF(ifmissing + 1);
        const ulong diff = soft64_sub(d_of(above), rtt_cone_profile(
            RTT_HGT(ifmissing + 1), maxref, mref, r.levels, r.profiles));
        if (soft64_less(d_abs(diff), d_of(10.0f))) {
            for (int k2 = ifmissing; k2 >= klo; --k2)
                RTT_REF(k2) = f_of(soft64_add(rtt_cone_profile(
                    RTT_HGT(k2), maxref, mref, r.levels, r.profiles), diff));
        } else {
            for (int k2 = ifmissing; k2 >= klo; --k2) RTT_REF(k2) = above;
        }
    }
    for (int k2 = 1; k2 <= khi; ++k2) RTT_REF(k2) = -99999.0f;
#undef RTT_REF
#undef RTT_HGT
}

kernel void rtt_probable_convection(
        constant RttProbableConvectionResources &r [[buffer(0)]],
        constant RttProbableConvectionParams &p [[buffer(1)]],
        uint tid [[thread_position_in_grid]]) {
    const int nx = p.nx;
    const int ncol = p.ny * nx;
    const int col = int(tid);
    if (col >= ncol) return;
    const int j = col / nx, i = col - j * nx;
    const ulong threshold = r.dp[0], warm_k = r.dp[1];
    const ulong cold_depth = r.dp[2], rd_over_cp = r.dp[3];
    int hit = 0;
    if (i >= 1 && i <= nx - 2 && j >= 1 && j <= p.ny - 2) {
        ulong dpint = 0ul;
        for (int kf = 2; kf <= p.nz - 1; ++kf) {
            const ulong at = ulong(kf - 1) * ulong(ncol) + ulong(col);
            const float tbk = rtt_temperature(r.t[at], r.p[at], rd_over_cp);
            const ulong rv = d_of(r.ref[at]);
            if (soft64_greater_equal(d_of(tbk), warm_k)
                    && soft64_greater_equal(rv, threshold)) hit = 1;
            if (soft64_less(d_of(tbk), warm_k)
                    && soft64_greater_equal(rv, threshold))
                dpint = soft64_add(dpint, soft64_mul(d_of(0.5f), d_of(
                    r.p[at - ulong(ncol)] - r.p[at + ulong(ncol)])));
        }
        if (soft64_greater_equal(dpint, cold_depth)) hit = 1;
    }
    r.probable[col] = hit;
}

kernel void rtt_tendency(constant RttTendencyResources &r [[buffer(0)]],
                         constant RttTendencyParams &p [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {
    const int nx = p.nx, nz = p.nz;
    const ulong ncol = ulong(p.ny) * ulong(nx);
    const ulong n = ulong(nz) * ncol;
    const ulong idx = ulong(tid);
    if (idx >= n) return;
    const int k = int(idx / ncol);
    const int col = int(idx - ulong(k) * ncol);
    const int j = col / nx, i = col - j * nx;
    const ulong echo_floor = r.dp[0], warm_k = r.dp[1], warm_min_dbz = r.dp[2];
    const ulong z_scale = r.dp[3], z_divisor = r.dp[4], z_factor = r.dp[5];
    const ulong inv_cpovr = r.dp[6], latent = r.dp[7], period_min = r.dp[8];
    const ulong cpd = r.dp[9], cap = r.dp[10], rd_over_cp = r.dp[11];
    ulong out = 0ul;
    if (k >= 1 && k <= nz - 2 && i >= 1 && i <= nx - 2 && j >= 1 && j <= p.ny - 2) {
        const float rf = r.ref[idx];
        const ulong rd = d_of(rf);
        if (soft64_less(rd, echo_floor) && rf > -100.0f) {
            out = 0ul;
        } else if (soft64_greater_equal(rd, echo_floor)) {
            const float tbk = rtt_temperature(r.t[idx], r.p[idx], rd_over_cp);
            const bool skip = soft64_greater(d_of(tbk), warm_k)
                              && soft64_less(rd, warm_min_dbz);
            if (!skip && (k + 1) >= rtt_krad_bot(p.krad_bot_in, r.pblh[col])) {
                const ulong addsnow = soft64_mul(soft64_div(
                    soft_glibc_pow64(d_of(10.0f), soft64_div(rd, z_scale)),
                    z_divisor), z_factor);
                const ulong tt = soft64_mul(
                    soft_glibc_pow64(soft64_div(d_of(1000.0f), d_of(r.p[idx])),
                                     inv_cpovr),
                    soft64_div(soft64_mul(latent, addsnow),
                               soft64_mul(soft64_mul(period_min, d_of(60.0f)), cpd)));
                out = d_fmin(cap, d_fmax(d_neg(cap), tt));
            }
        }
    }
    r.tten[idx] = out;
}

kernel void rtt_smooth_interior(constant RttSmoothInteriorResources &r [[buffer(0)]],
                                constant RttSmoothInteriorParams &p [[buffer(1)]],
                                uint tid [[thread_position_in_grid]]) {
    const int nx = p.nx;
    const ulong ncol = ulong(p.ny) * ulong(nx);
    const ulong n = ulong(p.nz) * ncol;
    const ulong idx = ulong(tid);
    if (idx >= n) return;
    const int col = int(idx % ncol);
    const int j = col / nx, i = col - j * nx;
    if (i < 1 || i > nx - 2 || j < 1 || j > p.ny - 2) {
        r.dst[idx] = r.src[idx];
        return;
    }
    device const ulong *f = r.src + idx;
    const ulong sum1 = soft64_add(soft64_add(soft64_add(
        f[nx - 1], f[-nx - 1]), f[nx + 1]), f[-nx + 1]);
    const ulong sum2 = soft64_add(soft64_add(soft64_add(
        f[nx], f[1]), f[-nx]), f[-1]);
    r.dst[idx] = soft64_add(soft64_add(soft64_mul(r.dp[0], sum1),
                                       soft64_mul(r.dp[1], sum2)),
                            soft64_mul(r.dp[2], f[0]));
}

kernel void rtt_smooth_edges(constant RttSmoothEdgesResources &r [[buffer(0)]],
                             constant RttSmoothEdgesParams &p [[buffer(1)]],
                             uint tid [[thread_position_in_grid]]) {
    const int nx = p.nx, ny = p.ny;
    if (int(tid) >= p.nz * 4) return;
    const int k = int(tid) / 4, edge = int(tid) - k * 4;
    const ulong s4 = r.dp[0], s5 = r.dp[1];
    device ulong *g = r.field + ulong(k) * ulong(ny) * ulong(nx);
    if (edge < 2) {
        device ulong *row = g + (edge == 0 ? 0ul : ulong(ny - 1) * ulong(nx));
        for (int i = 1; i <= nx - 2; ++i)
            row[i] = soft64_add(soft64_mul(s4, row[i]),
                                soft64_mul(s5, soft64_add(row[i - 1], row[i + 1])));
    } else {
        device ulong *colp = g + (edge == 2 ? 0 : nx - 1);
        for (int j = 1; j <= ny - 2; ++j) {
            const long at = long(j) * long(nx);
            colp[at] = soft64_add(soft64_mul(s4, colp[at]),
                                  soft64_mul(s5, soft64_add(colp[at - nx], colp[at + nx])));
        }
    }
}

kernel void rtt_store(constant RttStoreResources &r [[buffer(0)]],
                      constant RttStoreParams &p [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {
    const ulong n = ulong(p.nz) * ulong(p.ny) * ulong(p.nx);
    if (ulong(tid) >= n) return;
    float out = f_of(r.tten[tid]);
    if (r.ref[tid] <= -200.0f) out = -99999.0f;
    r.ges[tid] = out;
}

kernel void rtt_column_flag(constant RttColumnFlagResources &r [[buffer(0)]],
                            constant RttColumnFlagParams &p [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {
    const int nx = p.nx, nz = p.nz;
    const int ncol = p.ny * nx;
    const int col = int(tid);
    if (col >= ncol) return;
    const int j = col / nx, i = col - j * nx;
    const ulong coverage_depth = r.dp[0], nearby = r.dp[1];
    ulong radmax = 0ul, dpint = 0ul;
    for (int kf = 2; kf <= nz - 1; ++kf) {
        const ulong at = ulong(kf - 1) * ulong(ncol) + ulong(col);
        ulong v = r.tten[at];
        if (r.ref[at] <= -200.0f) v = d_of(-99999.0f);
        if (soft64_greater(v, d_of(-15.0f))) {
            dpint = soft64_add(dpint, soft64_mul(d_of(0.5f), d_of(
                r.p[at - ulong(ncol)] - r.p[at + ulong(ncol)])));
            radmax = d_fmax(radmax, v);
        }
    }
    float radyn = -10.0f;
    if (soft64_greater_equal(dpint, coverage_depth)) {
        radyn = 0.0f;
        if (soft64_greater(radmax, nearby)) radyn = 1.0f;
        if (fabs(radyn) < 0.00001f) {
            int kb = rtt_krad_bot(p.krad_bot_in, r.pblh[col]);
            if (kb < 1) kb = 1;
            for (int kf = kb; kf <= nz - 1; ++kf)
                r.ges[ulong(kf - 1) * ulong(ncol) + ulong(col)] = 0.0f;
        }
    }
    const ulong top = ulong(nz - 1) * ulong(ncol) + ulong(col);
    r.ges[top] = radyn;
    if (p.convection_only != 0 && i >= 1 && i <= nx - 2 && j >= 1 && j <= p.ny - 2
            && radyn > 0.9f && r.probable[col] == 0) {
        r.ges[top] = -10.0f;
        for (int kf = 2; kf <= nz - 1; ++kf)
            r.ges[ulong(kf - 1) * ulong(ncol) + ulong(col)] = -99999.0f;
    }
    for (int kf = 1; kf <= nz; ++kf) {
        const ulong at = ulong(kf - 1) * ulong(ncol) + ulong(col);
        if (r.ges[at] <= -200.0f) r.ges[at] = -20.0f;
    }
}

kernel void rtt_vinterp(constant RttVinterpResources &r [[buffer(0)]],
                        constant RttVinterpParams &p [[buffer(1)]],
                        uint tid [[thread_position_in_grid]]) {
    const int nx = p.nx, nlevels = p.nlevels;
    const ulong ncol = ulong(p.ny) * ulong(nx);
    const ulong n = ulong(p.nz) * ncol;
    const ulong idx = ulong(tid);
    if (idx >= n) return;
    const int k = int(idx / ncol);
    const int col = int(idx - ulong(k) * ncol);
    const int j = col / nx, i = col - j * nx;
    if (i < 1 || i > nx - 2 || j < 1 || j > p.ny - 2) {
        r.ref[idx] = -99999.0f;
        return;
    }
    const ulong hg = d_of(r.h_agl[idx] + r.terrain[col]);
    ulong out = d_of(-99999.0f);
    if (soft64_greater_equal(hg, r.levels[0])
            && soft64_less(hg, r.levels[nlevels - 1])) {
        int ilvl = 0;
        for (int m = 0; m < nlevels - 1; ++m)
            if (soft64_greater_equal(hg, r.levels[m]) && soft64_less(hg, r.levels[m + 1]))
                ilvl = m;
        const ulong up = d_of(r.mosaic[ulong(ilvl + 1) * ncol + ulong(col)]);
        const ulong down = d_of(r.mosaic[ulong(ilvl) * ncol + ulong(col)]);
        ulong value;
        if (soft64_less(d_abs(up), d_of(90.0f)) && soft64_less(d_abs(down), d_of(90.0f))) {
            const ulong wght = soft64_div(soft64_sub(hg, r.levels[ilvl]),
                                          soft64_sub(r.levels[ilvl + 1], r.levels[ilvl]));
            value = soft64_add(soft64_mul(soft64_sub(d_of(1.0f), wght), down),
                               soft64_mul(wght, up));
        } else if (soft64_less(d_abs(soft64_add(up, d_of(99.0f))), RTT_D_0_1)
                   || soft64_less(d_abs(soft64_add(down, d_of(99.0f))), RTT_D_0_1)) {
            value = d_of(-99.0f);
        } else {
            value = d_of(-99999.0f);
        }
        out = d_fmax(d_of(-99999.0f), value);
    }
    r.ref[idx] = f_of(out);
}

kernel void rtt_vinterp_edges(constant RttVinterpEdgesResources &r [[buffer(0)]],
                              constant RttVinterpEdgesParams &p [[buffer(1)]],
                              uint tid [[thread_position_in_grid]]) {
    const int nx = p.nx, ny = p.ny;
    const int per_level = 2 * (nx - 2) + 2 * (ny - 2) + 3;
    if (int(tid) >= p.nz * per_level) return;
    const int k = int(tid) / per_level;
    int e = int(tid) - k * per_level;
    device float *g = r.ref + ulong(k) * ulong(ny) * ulong(nx);
#define RTT_AT(jj, ii) g[ulong(jj) * ulong(nx) + ulong(ii)]
    if (e < nx - 2) { const int i = e + 1; RTT_AT(0, i) = RTT_AT(1, i); return; }
    e -= nx - 2;
    if (e < nx - 2) { const int i = e + 1; RTT_AT(ny - 1, i) = RTT_AT(ny - 2, i); return; }
    e -= nx - 2;
    if (e < ny - 2) { const int j = e + 1; RTT_AT(j, 0) = RTT_AT(j, 1); return; }
    e -= ny - 2;
    if (e < ny - 2) { const int j = e + 1; RTT_AT(j, nx - 1) = RTT_AT(j, nx - 2); return; }
    e -= ny - 2;
    if (e == 0) RTT_AT(ny - 1, nx - 1) = RTT_AT(ny - 2, nx - 2);
    else if (e == 1) RTT_AT(0, nx - 1) = RTT_AT(1, nx - 2);
    else RTT_AT(ny - 1, 0) = RTT_AT(1, 1);
#undef RTT_AT
}

kernel void rtt_apply(constant RttApplyResources &r [[buffer(0)]],
                      constant RttApplyParams &p [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {
    const int nx = p.nx, ring = p.ring;
    const ulong ncol = ulong(p.ny) * ulong(nx);
    const ulong n = ulong(p.nz) * ncol;
    const ulong idx = ulong(tid);
    if (idx >= n) return;
    const int k = int(idx / ncol);
    const int col = int(idx - ulong(k) * ncol);
    const int j = col / nx, i = col - j * nx;
    if (k >= p.nz - 1 || i < ring || i >= nx - ring || j < ring || j >= p.ny - ring)
        return;
    const float v = r.slot[idx];
    if (v >= -1.0f && v <= 1.0f)
        r.thp[idx] = r.thp_before[idx] + (v * p.dt);
}
