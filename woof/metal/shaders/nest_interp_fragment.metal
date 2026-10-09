// Nest interpolation kernels for the Metal tree route. The generated typed
// bindings carry exact int32 geometry resources; the arithmetic below follows
// core/kernels/nest.cu;
// boundary tendencies use the existing software binary64 helpers so their
// REAL*8 reciprocal and product do not fall back to native Metal double.

#include <metal_stdlib>
#include "nest_sint_fragment_bindings.metalh"
#include "nest_bdy_interp1_fragment_bindings.metalh"
#include "soft_binary64.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline ulong nest_interp_i3(int k, int j, int i, int ny, int nx) {
    return (ulong(k) * ulong(ny) + ulong(j)) * ulong(nx) + ulong(i);
}

static inline float nest_interp_donor(float y1, float y2, float a) {
    const float sign = copysign(1.0f, a);
    return (y1 * fmax(0.0f, sign) - y2 * fmin(0.0f, sign)) * a;
}

static inline float nest_interp_tr4(float ym1, float y0, float yp1,
                                    float yp2, float a) {
    const float one12 = 1.0f / 12.0f;
    const float one24 = 1.0f / 24.0f;
    return a * one12 * (7.0f * (yp1 + y0) - (yp2 + ym1))
         - a * a * one24 * (15.0f * (yp1 - y0) - (yp2 - ym1))
         - a * a * a * one12 * ((yp1 + y0) - (yp2 + ym1))
         + a * a * a * a * one24 * (3.0f * (yp1 - y0) - (yp2 - ym1));
}

static inline float nest_interp_pass(float ym2, float ym1, float y0,
                                     float yp1, float yp2, float a) {
    const float ep = 1.0e-10f;
    const float fl0 = nest_interp_donor(ym1, y0, a);
    const float fl1 = nest_interp_donor(y0, yp1, a);
    const float w = y0 - (fl1 - fl0);
    const float mxm = fmax(fmax(ym1, y0), fmax(yp1, w));
    const float mn = fmin(fmin(ym1, y0), fmin(yp1, w));
    const float f0 = nest_interp_tr4(ym2, ym1, y0, yp1, a) - fl0;
    const float f1 = nest_interp_tr4(ym1, y0, yp1, yp2, a) - fl1;
    const float pp0 = fmax(0.0f, f0), pn0 = fmin(0.0f, f0);
    const float pp1 = fmax(0.0f, f1), pn1 = fmin(0.0f, f1);
    const float ov = (mxm - w) / (-pn1 + pp0 + ep);
    const float un = (w - mn) / (pp1 - pn0 + ep);
    const float c0 = pp0 * fmin(1.0f, ov) + pn0 * fmin(1.0f, un);
    const float c1 = pp1 * fmin(1.0f, un) + pn1 * fmin(1.0f, ov);
    return w - (c1 - c0);
}

static inline float nest_interp_point(
    device const float *parent, int k, int cj, int ci, float ax, float ay,
    int nyp, int nxp) {
    float rows[5];
    for (int j = -2; j <= 2; ++j) {
        const ulong start = nest_interp_i3(k, cj + j, ci - 2, nyp, nxp);
        rows[j + 2] = nest_interp_pass(parent[start], parent[start + 1],
            parent[start + 2], parent[start + 3], parent[start + 4], ax);
    }
    return nest_interp_pass(rows[0], rows[1], rows[2], rows[3], rows[4], ay);
}

kernel void nest_sint_fragment(
    constant NestSintFragmentResources &r [[buffer(0)]],
    constant NestSintFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const ulong count = ulong(p.nz) * ulong(p.nyc) * ulong(p.nxc);
    if (ulong(tid) >= count) return;
    const int i = int(tid % uint(p.nxc));
    const int j = int((tid / uint(p.nxc)) % uint(p.nyc));
    const int k = int(tid / (ulong(p.nxc) * ulong(p.nyc)));
    r.child_a[tid] = nest_interp_point(r.parent_a, k, r.cj_map_a[j],
        r.ci_map_a[i], r.xig_a[r.ip_map_a[i]],
        r.xjg_a[r.jp_map_a[j]], p.nyp, p.nxp);
}

kernel void nest_bdy_interp1_fragment(
    constant NestBdyInterp1FragmentResources &r [[buffer(0)]],
    constant NestBdyInterp1FragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int longn = p.side < 2 ? p.nyc : p.nxc;
    const ulong count = ulong(p.nz) * ulong(longn) * ulong(p.sz);
    if (ulong(tid) >= count) return;
    const int w = int(tid % uint(p.sz));
    const int l = int((tid / uint(p.sz)) % uint(longn));
    const int k = int(tid / (ulong(p.sz) * ulong(longn)));
    int i, j;
    ulong slot;
    if (p.side == 0) {
        i = w; j = l; slot = nest_interp_i3(k, l, w, p.nyc, p.sz);
    } else if (p.side == 1) {
        i = p.nxc - 1 - w; j = l;
        slot = nest_interp_i3(k, l, w, p.nyc, p.sz);
    } else if (p.side == 2) {
        i = l; j = w; slot = nest_interp_i3(k, w, l, p.sz, p.nxc);
    } else {
        i = l; j = p.nyc - 1 - w;
        slot = nest_interp_i3(k, w, l, p.sz, p.nxc);
    }
    const float psca = nest_interp_point(r.parent_a, k, r.cj_map_a[j],
        r.ci_map_a[i], r.xig_a[r.ip_map_a[i]],
        r.xjg_a[r.jp_map_a[j]], p.nyp, p.nxp);
    const float child_value = r.child_a[
        nest_interp_i3(k, j, i, p.nyc, p.nxc)];
    const ulong rdt = soft64_div(
        0x3ff0000000000000ul,
        soft64_from_f32_bits(as_type<uint>(p.cdt)));
    const ulong difference = soft64_from_f32_bits(
        as_type<uint>(psca - child_value));
    r.tendency_a[slot] = as_type<float>(
        soft64_to_f32_bits(soft64_mul(rdt, difference)));
    r.value_a[slot] = child_value;
}
