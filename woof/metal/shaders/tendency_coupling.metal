#include <metal_stdlib>
#include "couple_mass_rates_bindings.metalh"
#include "couple_faces_bindings.metalh"
#include "soft_binary32.metalh"

using namespace metal;

// Binary32 operations call only the shared public software-RN primitives.
// No native float arithmetic is used in the coupling expression trees.
static inline float cp_add(float a, float b) {
    return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)));
}
static inline float cp_mul(float a, float b) {
    return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b)));
}
static inline float cp_div(float a, float b) {
    return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)));
}

static inline void cp_rate(
        device const float *input, device float *output, ulong at,
        float chm, bool present, bool ring, bool theta, bool has_msf,
        float msft) {
    float value = present ? cp_mul(chm, input[at]) : 0.0f;
    if (ring) value = 0.0f;
    if (theta && has_msf) value = cp_div(value, msft);
    output[at] = value;
}

kernel void couple_mass_rates(
        constant CoupleMassRatesResources &r [[buffer(0)]],
        constant CoupleMassRatesParams &p [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong plane = ulong(p.ny) * ulong(p.nx);
    const ulong count = ulong(p.nz) * plane;
    const ulong at = ulong(gid);
    if (at >= count) return;
    const ulong k = at / plane;
    const ulong cell = at - k * plane;
    const ulong j = cell / ulong(p.nx);
    const ulong i = cell - j * ulong(p.nx);

    // chm = RN(RN(c1h[k] * mut[cell]) + c2h[k]).
    const float product = cp_mul(r.c1h[k], r.mut[cell]);
    const float chm = cp_add(product, r.c2h[k]);
    if (p.momentum != 0) {
        r.mass_u[at] = cp_mul(chm, r.du[at]);
        r.mass_v[at] = cp_mul(chm, r.dv[at]);
    }

    const bool ring = p.mask_ring != 0
        && (j == 0ul || j == ulong(p.ny - 1)
            || i == 0ul || i == ulong(p.nx - 1));
    const uint wanted = as_type<uint>(p.wanted);
    const uint present = as_type<uint>(p.present);
    const float msft = r.msft[cell];
    if (p.nslot > 0 && (wanted & 0x01u))
        cp_rate(r.in0, r.out0, at, chm, (present & 0x01u) != 0u,
                ring, p.theta_slot == 0, p.has_msf != 0, msft);
    if (p.nslot > 1 && (wanted & 0x02u))
        cp_rate(r.in1, r.out1, at, chm, (present & 0x02u) != 0u,
                ring, p.theta_slot == 1, p.has_msf != 0, msft);
    if (p.nslot > 2 && (wanted & 0x04u))
        cp_rate(r.in2, r.out2, at, chm, (present & 0x04u) != 0u,
                ring, p.theta_slot == 2, p.has_msf != 0, msft);
    if (p.nslot > 3 && (wanted & 0x08u))
        cp_rate(r.in3, r.out3, at, chm, (present & 0x08u) != 0u,
                ring, p.theta_slot == 3, p.has_msf != 0, msft);
    if (p.nslot > 4 && (wanted & 0x10u))
        cp_rate(r.in4, r.out4, at, chm, (present & 0x10u) != 0u,
                ring, p.theta_slot == 4, p.has_msf != 0, msft);
    if (p.nslot > 5 && (wanted & 0x20u))
        cp_rate(r.in5, r.out5, at, chm, (present & 0x20u) != 0u,
                ring, p.theta_slot == 5, p.has_msf != 0, msft);
}

kernel void couple_faces(
        constant CoupleFacesResources &r [[buffer(0)]],
        constant CoupleFacesParams &p [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong nz = ulong(p.nz), ny = ulong(p.ny), nx = ulong(p.nx);
    const ulong nu = nz * ny * (nx + 1ul);
    const ulong nv = nz * (ny + 1ul) * nx;
    const ulong at = ulong(gid);
    if (at < nu) {
        const ulong row = nx + 1ul;
        const ulong k = at / (ny * row);
        const ulong rem = at - k * ny * row;
        const ulong j = rem / row;
        const ulong f = rem - j * row;
        const ulong e = f == nx ? 0ul : f;
        const ulong w = e == 0ul ? nx - 1ul : e - 1ul;
        const ulong base = (k * ny + j) * nx;
        float value = cp_mul(0.5f,
            cp_add(r.mass_u[base + e], r.mass_u[base + w]));
        if (p.mask_ring != 0) {
            if (j == 0ul || j == ny - 1ul || f == 0ul || f == nx)
                value = 0.0f;
        } else if (p.open_x != 0 && (f == 0ul || f == nx)) {
            value = 0.0f;
        }
        if (p.has_msf != 0) {
            const bool closed = !(p.mask_ring != 0 || p.open_x != 0)
                && f == nx;
            value = cp_div(value, r.msfu[j * row + (closed ? 0ul : f)]);
        }
        r.ru[at] = value;
        return;
    }

    const ulong t = at - nu;
    if (t >= nv) return;
    const ulong k = t / ((ny + 1ul) * nx);
    const ulong rem = t - k * (ny + 1ul) * nx;
    const ulong g = rem / nx;
    const ulong i = rem - g * nx;
    const ulong e = g == ny ? 0ul : g;
    const ulong south = e == 0ul ? ny - 1ul : e - 1ul;
    const ulong base = k * ny * nx;
    float value = cp_mul(0.5f,
        cp_add(r.mass_v[base + e * nx + i],
               r.mass_v[base + south * nx + i]));
    if (p.mask_ring != 0) {
        if (g == 0ul || g == ny || i == 0ul || i == nx - 1ul)
            value = 0.0f;
    } else if (p.open_y != 0 && (g == 0ul || g == ny)) {
        value = 0.0f;
    }
    if (p.has_msf != 0) {
        const bool closed = !(p.mask_ring != 0 || p.open_y != 0)
            && g == ny;
        const float metric = r.msfv[(closed ? 0ul : g) * nx + i];
        if (p.wrf_exact != 0)
            value = cp_mul(value, cp_div(1.0f, metric));
        else
            value = cp_div(value, metric);
    }
    r.rv[t] = value;
}
