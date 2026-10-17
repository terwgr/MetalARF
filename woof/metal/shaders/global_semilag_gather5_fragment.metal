// FP64 six-point horizontal semi-Lagrangian gather, with a quasi-monotone clip.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_gather5_fragment_bindings.metalh"

using namespace metal;

static inline int slg5_floor_i32(ulong x) {
    Soft64I32Result r = soft64_to_i32_rz(x);
    if (soft64_less(x, soft64_from_i32(r.value))) --r.value;
    return r.value;
}

kernel void global_semilag_gather5_fragment(
    constant GlobalSemilagGather5FragmentResources &r [[buffer(0)]],
    constant GlobalSemilagGather5FragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint npoint = uint(p.nlevel) * uint(p.nlat) * uint(p.nlon);
    const uint plane = uint(p.nlat) * uint(p.nlon);
    const bool surface_mode = p.surface_mode != 0;
    const uint total = surface_mode ? plane : npoint * uint(p.nfield);
    if (tid >= total) return;
    const uint field = surface_mode ? 0u : tid / npoint;
    const uint gid = surface_mode ? tid : tid - field * npoint;
    const uint stencil_gid = surface_mode
        ? (uint(p.nlevel) - 1u) * plane + gid : gid;
    const uint k = surface_mode ? 0u : gid / plane;
    const ulong half_pi = 0x3ff921fb54442d18ul;
    const ulong one = 0x3ff0000000000000ul;
    const ulong two = 0x4000000000000000ul;
    const ulong three = 0x4008000000000000ul;
    const ulong half_bits = 0x3fe0000000000000ul;
    const ulong sixth = soft64_div(one, 0x4018000000000000ul);

    ulong phi = r.phi[stencil_gid];
    if (soft64_less(phi, soft64_negate_bits(half_pi))) phi = soft64_negate_bits(half_pi);
    if (soft64_greater(phi, half_pi)) phi = half_pi;
    ulong xk = surface_mode ? 0ul : r.level[stencil_gid];
    const ulong top = soft64_from_i32(p.nlevel - 1);
    if (soft64_less(xk, 0ul)) xk = 0ul;
    if (soft64_greater(xk, top)) xk = top;
    const ulong xi = r.xi[stencil_gid];

    // Six equally spaced zonal nodes, wrapped through the pole-shift map.
    const int i0raw = slg5_floor_i32(xi) - 2;
    int a[6];
    const int half_lon = p.nlon / 2;
    int b[6];
    for (int q = 0; q < 6; ++q) {
        int col = (i0raw + q) % p.nlon;
        if (col < 0) col += p.nlon;
        a[q] = col;
        b[q] = (col + half_lon) % p.nlon;
    }
    const ulong fx = soft64_sub(xi, soft64_from_i32(i0raw));
    ulong wx[6];
    const int denominator[6] = {-120, 24, -12, 12, -24, 120};
    for (int m = 0; m < 6; ++m) {
        ulong product = one;
        for (int q = 0; q < 6; ++q) {
            if (q != m) product = soft64_mul(
                product, soft64_sub(fx, soft64_from_i32(q)));
        }
        wx[m] = soft64_div(product, soft64_from_i32(denominator[m]));
    }
    ulong sum = 0ul;
    for (int q = 0; q < 6; ++q) sum = soft64_add(sum, wx[q]);
    for (int q = 0; q < 6; ++q) wx[q] = soft64_div(wx[q], sum);

    // Six-row Gaussian Lagrange stencil, using the precomputed denominators.
    const ulong lut_pos = soft64_mul(soft64_add(phi, half_pi), r.lut_scale[0]);
    const int bin = clamp(slg5_floor_i32(lut_pos), 0, p.lookup_bins - 1);
    int m0 = r.mlut[bin];
    while (m0 > 2 && soft64_less(phi, r.lat_ext[m0])) --m0;
    while (m0 < p.nlat + 2 && !soft64_less(phi, r.lat_ext[m0 + 1])) ++m0;
    const int sst = m0 - 2;
    ulong wy[6];
    for (int m = 0; m < 6; ++m) {
        ulong product = r.mrden[6 * sst + m];
        for (int q = 0; q < 6; ++q) {
            if (q != m) product = soft64_mul(
                product, soft64_sub(phi, r.lat_ext[sst + q]));
        }
        wy[m] = product;
    }
    sum = 0ul;
    for (int m = 0; m < 6; ++m) sum = soft64_add(sum, wy[m]);
    for (int m = 0; m < 6; ++m) wy[m] = soft64_div(wy[m], sum);

    // The vertical stencil remains four-point cubic.
    int kb = clamp(slg5_floor_i32(xk), 0, p.nlevel - 2);
    int k0 = clamp(kb - 1, 0, p.nlevel - 4);
    const int kbox = kb - k0;
    const ulong fz = soft64_sub(xk, soft64_from_i32(k0));
    ulong wz[4] = {
        soft64_mul(soft64_mul(soft64_mul(
            soft64_negate_bits(soft64_sub(fz, one)),
            soft64_sub(fz, two)), soft64_sub(fz, three)), sixth),
        soft64_mul(soft64_mul(soft64_mul(fz,
            soft64_sub(fz, two)), soft64_sub(fz, three)), half_bits),
        soft64_mul(soft64_mul(soft64_mul(
            soft64_negate_bits(fz), soft64_sub(fz, one)),
            soft64_sub(fz, three)), half_bits),
        soft64_mul(soft64_mul(soft64_mul(fz,
            soft64_sub(fz, one)), soft64_sub(fz, two)), sixth)
    };
    sum = soft64_add(soft64_add(wz[0], wz[1]), soft64_add(wz[2], wz[3]));
    for (int c = 0; c < 4; ++c) wz[c] = soft64_div(wz[c], sum);

    ulong acc = 0ul;
    ulong lo = 0x7ff0000000000000ul;
    ulong hi = 0xfff0000000000000ul;
    const uint field_offset = field * npoint;
    for (int c = 0; c < 4; ++c) {
        ulong rowsum = 0ul;
        for (int m = 0; m < 6; ++m) {
            const int row = sst + m;
            ulong colsum = 0ul;
            ulong vals[6];
            for (int q = 0; q < 6; ++q) {
                const uint horizontal = uint(r.rowoff[row] +
                    ((r.rowshift[row] != 0) ? b[q] : a[q]));
                const uint index = surface_mode ? horizontal
                    : uint(k0 + c) * plane + horizontal;
                vals[q] = r.source[field_offset + index];
                colsum = soft64_add(colsum, soft64_mul(wx[q], vals[q]));
            }
            rowsum = soft64_add(rowsum, soft64_mul(wy[m], colsum));
            if (p.quasi_monotone != 0 && (c == kbox || c == kbox + 1)
                    && (m == 2 || m == 3)) {
                for (int q = 2; q < 4; ++q) {
                    if (soft64_less(vals[q], lo)) lo = vals[q];
                    if (soft64_greater(vals[q], hi)) hi = vals[q];
                }
            }
        }
        acc = soft64_add(acc, soft64_mul(wz[c], rowsum));
    }
    if (p.quasi_monotone != 0) {
        if (soft64_less(acc, lo)) acc = lo;
        if (soft64_greater(acc, hi)) acc = hi;
    }
    r.destination[tid] = acc;
}
