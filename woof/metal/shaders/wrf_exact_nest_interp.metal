// Nest interpolation kernels for the Metal tree route. The generated typed
// bindings carry exact int32 geometry resources; the arithmetic below follows
// core/kernels/nest.cu;
// boundary tendencies use the existing software binary64 helpers so their
// REAL*8 reciprocal and product do not fall back to native Metal double.

#include <metal_stdlib>
#include "wrf_exact_nest_interp_bindings.metalh"
#include "exact_binary32.metalh"
#include "soft_binary64.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline ulong nest_interp_i3(int k, int j, int i, int ny, int nx) {
    return (ulong(k) * ulong(ny) + ulong(j)) * ulong(nx) + ulong(i);
}

static inline Exact32 nest_interp_donor(Exact32 y1, Exact32 y2, Exact32 a) {
    const Exact32 sign = Exact32::from_bits(0x3f800000u | (a.bits & 0x80000000u));
    return (y1 * exact_max(Exact32(0.0f), sign) - y2 * exact_min(Exact32(0.0f), sign)) * a;
}

static inline Exact32 nest_interp_tr4(Exact32 ym1, Exact32 y0, Exact32 yp1,
                                    Exact32 yp2, Exact32 a) {
    const Exact32 one12 = Exact32(1.0f) / Exact32(12.0f);
    const Exact32 one24 = Exact32(1.0f) / Exact32(24.0f);
    return a * one12 * (Exact32(7.0f) * (yp1 + y0) - (yp2 + ym1))
         - a * a * one24 * (Exact32(15.0f) * (yp1 - y0) - (yp2 - ym1))
         - a * a * a * one12 * ((yp1 + y0) - (yp2 + ym1))
         + a * a * a * a * one24 * (Exact32(3.0f) * (yp1 - y0) - (yp2 - ym1));
}

static inline Exact32 nest_interp_pass(Exact32 ym2, Exact32 ym1, Exact32 y0,
                                     Exact32 yp1, Exact32 yp2, Exact32 a) {
    const Exact32 ep = Exact32(1.0e-10f);
    const Exact32 fl0 = nest_interp_donor(ym1, y0, a);
    const Exact32 fl1 = nest_interp_donor(y0, yp1, a);
    const Exact32 w = y0 - (fl1 - fl0);
    const Exact32 mxm = exact_max(exact_max(ym1, y0), exact_max(yp1, w));
    const Exact32 mn = exact_min(exact_min(ym1, y0), exact_min(yp1, w));
    const Exact32 f0 = nest_interp_tr4(ym2, ym1, y0, yp1, a) - fl0;
    const Exact32 f1 = nest_interp_tr4(ym1, y0, yp1, yp2, a) - fl1;
    const Exact32 pp0 = exact_max(Exact32(0.0f), f0), pn0 = exact_min(Exact32(0.0f), f0);
    const Exact32 pp1 = exact_max(Exact32(0.0f), f1), pn1 = exact_min(Exact32(0.0f), f1);
    const Exact32 ov = (mxm - w) / (-pn1 + pp0 + ep);
    const Exact32 un = (w - mn) / (pp1 - pn0 + ep);
    const Exact32 c0 = pp0 * exact_min(Exact32(1.0f), ov) + pn0 * exact_min(Exact32(1.0f), un);
    const Exact32 c1 = pp1 * exact_min(Exact32(1.0f), un) + pn1 * exact_min(Exact32(1.0f), ov);
    return w - (c1 - c0);
}

static inline Exact32 nest_interp_point(
    device const Exact32 *parent, int k, int cj, int ci, Exact32 ax, Exact32 ay,
    int nyp, int nxp) {
    Exact32 rows[5];
    for (int j = -2; j <= 2; ++j) {
        const ulong start = nest_interp_i3(k, cj + j, ci - 2, nyp, nxp);
        rows[j + 2] = nest_interp_pass(parent[start], parent[start + 1],
            parent[start + 2], parent[start + 3], parent[start + 4], ax);
    }
    return nest_interp_pass(rows[0], rows[1], rows[2], rows[3], rows[4], ay);
}

kernel void wrf_exact_nest_sint_fragment(
    constant WrfExactNestSintFragmentResources &r [[buffer(0)]],
    constant WrfExactNestSintFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const ulong count = ulong(p.nz) * ulong(p.nyc) * ulong(p.nxc);
    if (ulong(tid) >= count) return;
    const int i = int(tid % uint(p.nxc));
    const int j = int((tid / uint(p.nxc)) % uint(p.nyc));
    const int k = int(tid / (ulong(p.nxc) * ulong(p.nyc)));
    reinterpret_cast<device Exact32 *>(r.child_a)[tid] = nest_interp_point(reinterpret_cast<device const Exact32 *>(r.parent_a), k, r.cj_map_a[j],
        r.ci_map_a[i], reinterpret_cast<device const Exact32 *>(r.xig_a)[r.ip_map_a[i]],
        reinterpret_cast<device const Exact32 *>(r.xjg_a)[r.jp_map_a[j]], p.nyp, p.nxp);
}

kernel void wrf_exact_nest_bdy_interp1_fragment(
    constant WrfExactNestBdyInterp1FragmentResources &r [[buffer(0)]],
    constant WrfExactNestBdyInterp1FragmentParams &p [[buffer(1)]],
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
    const Exact32 psca = nest_interp_point(reinterpret_cast<device const Exact32 *>(r.parent_a), k, r.cj_map_a[j],
        r.ci_map_a[i], reinterpret_cast<device const Exact32 *>(r.xig_a)[r.ip_map_a[i]],
        reinterpret_cast<device const Exact32 *>(r.xjg_a)[r.jp_map_a[j]], p.nyp, p.nxp);
    const Exact32 child_value = reinterpret_cast<device const Exact32 *>(r.child_a)[
        nest_interp_i3(k, j, i, p.nyc, p.nxc)];
    const ulong rdt = soft64_div(
        0x3ff0000000000000ul,
        soft64_from_f32_bits(as_type<uint>(p.cdt)));
    const ulong difference = soft64_from_f32_bits(
        (psca - child_value).bits);
    r.tendency_a[slot] = as_type<float>(
        soft64_to_f32_bits(soft64_mul(rdt, difference)));
    reinterpret_cast<device Exact32 *>(r.value_a)[slot] = child_value;
}
