#include <metal_stdlib>
#include "stage_face_masses_bindings.metalh"
#include "couple_ru_bindings.metalh"
#include "couple_rv_bindings.metalh"
#include "omega_column_scan_bindings.metalh"

using namespace metal;

// These kernels mirror WOOF's explicit FP32 operation boundaries in the
// source implementation; keep contraction disabled for both library modes.
#pragma clang fp contract(off)

static inline ulong mass_index(int j, int i, int nx) {
    return ulong(j) * ulong(nx) + ulong(i);
}

static inline float stage_face_mass(constant StageFaceMassesResources &r,
                                    constant StageFaceMassesParams &p,
                                    int j, int i, bool x_axis) {
    const int n = x_axis ? p.nx : p.ny;
    const int input_face = x_axis ? i : j;
    const int other = x_axis ? j : i;
    const bool boundary_axis = x_axis ? p.boundary_x != 0
                                      : p.boundary_y != 0;
    // The duplicate high periodic face is computed as face zero; never read
    // another thread's just-written output to copy that value.
    const int face = (!boundary_axis && input_face == n) ? 0 : input_face;
    const int current_face = face == n ? n - 1 : face;
    const int current_i = x_axis ? current_face : other;
    const int current_j = x_axis ? other : current_face;
    const bool edge = boundary_axis && (face == 0 || face == n);
    const int previous = face == 0 ? n - 1 : face - 1;
    const int previous_i = x_axis ? previous : other;
    const int previous_j = x_axis ? other : previous;
    const ulong current = mass_index(current_j, current_i, p.nx);
    const ulong left = mass_index(previous_j, previous_i, p.nx);

    if (edge) {
        const float mp = r.mup[current];
        const float mb = r.mub2d[current];
        if (p.wrf_exact)
            return 0.5f * (((mp + mp) + mb) + mb);
        return mb + mp;
    }

    if (p.wrf_exact) {
        // WRF calc_mu_uv grouping: perturbations are summed first, then the
        // base profile words, followed by the final half multiply.
        return 0.5f * (((r.mup[current] + r.mup[left])
                        + r.mub2d[current]) + r.mub2d[left]);
    }

    // stage_face_masses consumes state.total_mu() = mub2d + mup before the
    // face average; preserve those two FP32 rounds and operand order.
    const float mu_current = r.mub2d[current] + r.mup[current];
    const float mu_previous = r.mub2d[left] + r.mup[left];
    return 0.5f * (mu_current + mu_previous);
}

// One thread per mass cell writes each low-side face and the final high-side
// face. For periodic axes the final face duplicates face zero; nonperiodic
// (open, specified, or nested) axes use the adjacent boundary-cell mass.
kernel void stage_face_masses(
    constant StageFaceMassesResources &r [[buffer(0)]],
    constant StageFaceMassesParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny) * uint(p.nx);
    if (col >= ncol) return;
    const int j = int(col) / p.nx;
    const int i = int(col) - j * p.nx;
    const ulong u0 = ulong(j) * ulong(p.nx + 1) + ulong(i);
    const ulong v0 = ulong(j) * ulong(p.nx) + ulong(i);

    r.mux[u0] = stage_face_mass(r, p, j, i, true);
    r.muy[v0] = stage_face_mass(r, p, j, i, false);
    if (i == p.nx - 1) {
        r.mux[u0 + 1] = stage_face_mass(r, p, j, p.nx, true);
    }
    if (j == p.ny - 1) {
        r.muy[v0 + ulong(p.nx)] = stage_face_mass(r, p, p.ny, i, false);
    }
}

kernel void couple_ru(constant CoupleRuResources &r [[buffer(0)]],
                      constant CoupleRuParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const uint nface = uint(p.ny) * uint(p.nx + 1);
    const uint count = uint(p.nz) * nface;
    if (flat >= count) return;
    const uint k = flat / nface;
    const uint face = flat - k * nface;
    float value = (r.c1h[k] * r.mux[face] + r.c2h[k]) * r.u[flat];
    if (p.has_msf) value = value / r.msfu[face];
    r.ru[flat] = value;
}

kernel void couple_rv(constant CoupleRvResources &r [[buffer(0)]],
                      constant CoupleRvParams &p [[buffer(1)]],
                      uint flat [[thread_position_in_grid]]) {
    const uint nface = uint(p.ny + 1) * uint(p.nx);
    const uint count = uint(p.nz) * nface;
    if (flat >= count) return;
    const uint k = flat / nface;
    const uint face = flat - k * nface;
    float value = (r.c1h[k] * r.muy[face] + r.c2h[k]) * r.v[flat];
    if (p.has_msf) {
        if (p.reciprocal)
            value = value * (1.0f / r.msfv[face]);
        else
            value = value / r.msfv[face];
    }
    r.rv[flat] = value;
}

kernel void omega_column_scan(constant OmegaColumnScanResources &r [[buffer(0)]],
                              constant OmegaColumnScanParams &p [[buffer(1)]],
                              uint col [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny) * uint(p.nx);
    if (col >= ncol) return;
    const int j = int(col) / p.nx;
    const int i = int(col) - j * p.nx;
    float dmdt = 0.0f;

    // Pass one: source-order horizontal divergence and sequential column sum.
    for (int k = 0; k < p.nz; ++k) {
        const int u0 = (k * p.ny + j) * (p.nx + 1) + i;
        const int v0 = (k * (p.ny + 1) + j) * p.nx + i;
        const float du = r.ru[u0 + 1] - r.ru[u0];
        const float dv = r.rv[v0 + p.nx] - r.rv[v0];
        const float bracket = p.rdx * du + p.rdy * dv;
        const float weight = p.has_msf ? r.msft[col] * r.dnw[k]
                                       : r.dnw[k];
        const float divv = weight * bracket;
        dmdt = dmdt + divv;
        r.ww[(k + 1) * ncol + col] = divv;
    }

    // Pass two: recurrence has two distinct subtractions in the WOOF/CUDA
    // source order. The surface and top fluxes are explicitly zero.
    float w = 0.0f;
    r.ww[col] = w;
    for (int k = 1; k < p.nz; ++k) {
        const int at = k * ncol + col;
        const float level_weight = r.dnw[k - 1] * r.c1h[k - 1];
        w = w - level_weight * dmdt;
        w = w - r.ww[at];
        r.ww[at] = w;
    }
    r.ww[p.nz * ncol + col] = 0.0f;
}
