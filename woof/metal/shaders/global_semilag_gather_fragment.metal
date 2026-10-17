// FP64 cubic semi-Lagrangian gather, with the optional quasi-monotone clip.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_gather_fragment_bindings.metalh"

using namespace metal;

static inline int slg_floor_i32(ulong x) {
    Soft64I32Result r = soft64_to_i32_rz(x);
    int i = r.value;
    if (soft64_less(x, soft64_from_i32(i))) --i;
    return i;
}

static inline ulong slg_weight(ulong a, ulong b, ulong c, ulong scale) {
    return soft64_mul(soft64_mul(soft64_mul(a, b), c), scale);
}

kernel void global_semilag_gather_fragment(
    constant GlobalSemilagGatherFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagGatherFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint plane = uint(p.nlat) * uint(p.nlon);
    const uint npoint = uint(p.nlevel) * plane;
    const bool surface_mode = p.surface_mode != 0;
    const uint total = surface_mode ? plane : npoint * uint(p.nfield);
    if (tid >= total) return;
    const uint field = surface_mode ? 0u : tid / npoint;
    const uint gid = surface_mode ? tid : tid - field * npoint;
    const uint stencil_gid = surface_mode
        ? (uint(p.nlevel) - 1u) * plane + gid : gid;
    const uint k = surface_mode ? 0u : gid / plane;
    const uint rem = surface_mode ? gid : gid - k * plane;
    const uint j = rem / uint(p.nlon);
    const uint i = rem - j * uint(p.nlon);
    const ulong half_pi = 0x3ff921fb54442d18ul;
    const ulong one = 0x3ff0000000000000ul;
    const ulong two = 0x4000000000000000ul;
    const ulong three = 0x4008000000000000ul;
    const ulong six = 0x4018000000000000ul;
    const ulong half_bits = 0x3fe0000000000000ul;
    const ulong sixth = soft64_div(one, six);
    const ulong pi2 = soft64_mul(two, half_pi);

    ulong phi = r.phi[stencil_gid];
    if (soft64_less(phi, soft64_negate_bits(half_pi))) phi = soft64_negate_bits(half_pi);
    if (soft64_greater(phi, half_pi)) phi = half_pi;
    ulong xk = surface_mode ? 0ul : r.level[stencil_gid];
    const ulong top = soft64_from_i32(p.nlevel - 1);
    if (soft64_less(xk, 0ul)) xk = 0ul;
    if (soft64_greater(xk, top)) xk = top;
    const ulong xi = r.xi[stencil_gid];

    const int i0raw = slg_floor_i32(xi) - 1;
    int a0 = i0raw % p.nlon;
    if (a0 < 0) a0 += p.nlon;
    int a[4];
    for (uint q = 0; q < 4; ++q) a[q] = (a0 + int(q)) % p.nlon;
    const int half_lon = p.nlon / 2;
    int b[4];
    for (uint q = 0; q < 4; ++q) b[q] = (a[q] + half_lon) % p.nlon;

    ulong fx = soft64_sub(xi, soft64_from_i32(i0raw));
    ulong x0 = fx;
    ulong x1 = soft64_sub(fx, one);
    ulong x2 = soft64_sub(fx, two);
    ulong x3 = soft64_sub(fx, three);
    ulong wx[4];
    wx[0] = soft64_mul(soft64_mul(soft64_mul(soft64_negate_bits(x1), x2), x3), sixth);
    wx[1] = soft64_mul(soft64_mul(soft64_mul(x0, x2), x3), half_bits);
    wx[2] = soft64_mul(soft64_mul(soft64_mul(soft64_negate_bits(x0), x1), x3), half_bits);
    wx[3] = soft64_mul(soft64_mul(soft64_mul(x0, x1), x2), sixth);
    ulong wsum = soft64_add(soft64_add(wx[0], wx[1]), soft64_add(wx[2], wx[3]));
    for (uint q = 0; q < 4; ++q) wx[q] = soft64_div(wx[q], wsum);

    ulong lut_pos = soft64_mul(soft64_add(phi, half_pi), r.lut_scale[0]);
    int bin = clamp(slg_floor_i32(lut_pos), 0, p.lookup_bins - 1);
    int m0 = r.mlut[bin];
    while (m0 > 1 && soft64_less(phi, r.lat_ext[m0])) --m0;
    while (m0 < p.nlat + 1 && !soft64_less(phi, r.lat_ext[m0 + 1])) ++m0;
    const int sst = m0 - 1;
    ulong wy[4];
    ulong d[4];
    for (uint m = 0; m < 4; ++m) d[m] = soft64_sub(phi, r.lat_ext[sst + int(m)]);
    for (uint m = 0; m < 4; ++m) {
        ulong prod = r.mrden[4 * sst + int(m)];
        for (uint q = 0; q < 4; ++q) if (q != m) prod = soft64_mul(prod, d[q]);
        wy[m] = prod;
    }
    wsum = soft64_add(soft64_add(wy[0], wy[1]), soft64_add(wy[2], wy[3]));
    for (uint m = 0; m < 4; ++m) wy[m] = soft64_div(wy[m], wsum);

    int kb = clamp(slg_floor_i32(xk), 0, p.nlevel - 2);
    int k0 = clamp(kb - 1, 0, p.nlevel - 4);
    const int kbox = kb - k0;
    ulong fz = soft64_sub(xk, soft64_from_i32(k0));
    ulong z0 = fz;
    ulong z1 = soft64_sub(fz, one);
    ulong z2 = soft64_sub(fz, two);
    ulong z3 = soft64_sub(fz, three);
    ulong wz[4] = {
        soft64_mul(soft64_mul(soft64_mul(soft64_negate_bits(z1), z2), z3), sixth),
        soft64_mul(soft64_mul(soft64_mul(z0, z2), z3), half_bits),
        soft64_mul(soft64_mul(soft64_mul(soft64_negate_bits(z0), z1), z3), half_bits),
        soft64_mul(soft64_mul(soft64_mul(z0, z1), z2), sixth)
    };
    wsum = soft64_add(soft64_add(wz[0], wz[1]), soft64_add(wz[2], wz[3]));
    for (uint c = 0; c < 4; ++c) wz[c] = soft64_div(wz[c], wsum);

    ulong acc = 0ul;
    ulong lo = 0x7ff0000000000000ul;
    ulong hi = 0xfff0000000000000ul;
    const uint field_offset = field * npoint;
    for (int c = 0; c < 4; ++c) {
        ulong rowsum = 0ul;
        for (int m = 0; m < 4; ++m) {
            const int row = sst + m;
            ulong colsum = 0ul;
            ulong vals[4];
            for (int q = 0; q < 4; ++q) {
                const uint horizontal = uint(r.rowoff[row] +
                    ((r.rowshift[row] != 0) ? b[q] : a[q]));
                const uint index = surface_mode ? horizontal
                    : uint(k0 + c) * plane + horizontal;
                vals[q] = r.source[field_offset + index];
                colsum = soft64_add(colsum, soft64_mul(wx[q], vals[q]));
            }
            rowsum = soft64_add(rowsum, soft64_mul(wy[m], colsum));
            if (p.quasi_monotone != 0 && (c == kbox || c == kbox + 1) && (m == 1 || m == 2)) {
                const ulong vmin = soft64_less(vals[1], vals[2]) ? vals[1] : vals[2];
                const ulong vmax = soft64_greater(vals[1], vals[2]) ? vals[1] : vals[2];
                if (soft64_less(vmin, lo)) lo = vmin;
                if (soft64_greater(vmax, hi)) hi = vmax;
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
