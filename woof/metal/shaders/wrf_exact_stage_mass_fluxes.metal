#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_stage_face_masses_bindings.metalh"
#include "wrf_exact_couple_ru_bindings.metalh"
#include "wrf_exact_couple_rv_bindings.metalh"
#include "wrf_exact_omega_column_scan_bindings.metalh"

using namespace metal;

static inline device const Exact32 *exact_words(device const float *p) {
    return reinterpret_cast<device const Exact32 *>(p);
}
static inline device Exact32 *exact_words(device float *p) {
    return reinterpret_cast<device Exact32 *>(p);
}

// Strict stage mass and Omega operations retain every source binary32 round.
// The ordinary stage kernels remain separate and unchanged.
#pragma clang fp contract(off)

static inline ulong mass_index(int j, int i, int nx) {
    return ulong(j) * ulong(nx) + ulong(i);
}

static inline Exact32 stage_face_mass(constant WrfExactStageFaceMassesResources &r,
                                    constant WrfExactStageFaceMassesParams &p,
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

    if (p.omega) {
        // calc_ww_cp forms one column's base+perturbation before its neighbor.
        const ulong neighbor = edge ? current : left;
        Exact32 mp_current = exact_words(r.mup)[current];
        Exact32 mp_neighbor = exact_words(r.mup)[neighbor];
        if (p.use_halo && edge) {
            if (face == 0) mp_neighbor = exact_words(r.halo)[current];
            else mp_current = exact_words(r.halo)[current];
        }
        return 0.5f * (((mp_current + exact_words(r.mub2d)[current])
                        + mp_neighbor) + exact_words(r.mub2d)[neighbor]);
    }

    if (edge) {
        const Exact32 mp = exact_words(r.mup)[current];
        const Exact32 mb = exact_words(r.mub2d)[current];
        if (p.wrf_exact)
            return 0.5f * (((mp + mp) + mb) + mb);
        return mb + mp;
    }

    if (p.wrf_exact) {
        // WRF calc_mu_uv grouping: perturbations are summed first, then the
        // base profile words, followed by the final half multiply.
        return 0.5f * (((exact_words(r.mup)[current] + exact_words(r.mup)[left])
                        + exact_words(r.mub2d)[current]) + exact_words(r.mub2d)[left]);
    }

    // stage_face_masses consumes state.total_mu() = mub2d + mup before the
    // face average; preserve those two FP32 rounds and operand order.
    const Exact32 mu_current = exact_words(r.mub2d)[current] + exact_words(r.mup)[current];
    const Exact32 mu_previous = exact_words(r.mub2d)[left] + exact_words(r.mup)[left];
    return 0.5f * (mu_current + mu_previous);
}

// One thread per mass cell writes each low-side face and the final high-side
// face. For periodic axes the final face duplicates face zero; nonperiodic
// (open, specified, or nested) axes use the adjacent boundary-cell mass.
kernel void wrf_exact_stage_face_masses(
    constant WrfExactStageFaceMassesResources &r [[buffer(0)]],
    constant WrfExactStageFaceMassesParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny) * uint(p.nx);
    if (col >= ncol) return;
    const int j = int(col) / p.nx;
    const int i = int(col) - j * p.nx;
    const ulong u0 = ulong(j) * ulong(p.nx + 1) + ulong(i);
    const ulong v0 = ulong(j) * ulong(p.nx) + ulong(i);

    exact_words(r.mux)[u0] = stage_face_mass(r, p, j, i, true);
    exact_words(r.muy)[v0] = stage_face_mass(r, p, j, i, false);
    if (i == p.nx - 1) {
        exact_words(r.mux)[u0 + 1] = stage_face_mass(r, p, j, p.nx, true);
    }
    if (j == p.ny - 1) {
        exact_words(r.muy)[v0 + ulong(p.nx)] = stage_face_mass(r, p, p.ny, i, false);
    }
}

kernel void wrf_exact_couple_ru(constant WrfExactCoupleRuResources &r [[buffer(0)]],
                      constant WrfExactCoupleRuParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const uint nface = uint(p.ny) * uint(p.nx + 1);
    const uint count = uint(p.nz) * nface;
    if (flat >= count) return;
    const uint k = flat / nface;
    const uint face = flat - k * nface;
    Exact32 value = (exact_words(r.c1h)[k] * exact_words(r.mux)[face] + exact_words(r.c2h)[k]) * exact_words(r.u)[flat];
    if (p.has_msf) value = value / exact_words(r.msfu)[face];
    exact_words(r.ru)[flat] = value;
}

kernel void wrf_exact_couple_rv(constant WrfExactCoupleRvResources &r [[buffer(0)]],
                      constant WrfExactCoupleRvParams &p [[buffer(1)]],
                      uint flat [[thread_position_in_grid]]) {
    const uint nface = uint(p.ny + 1) * uint(p.nx);
    const uint count = uint(p.nz) * nface;
    if (flat >= count) return;
    const uint k = flat / nface;
    const uint face = flat - k * nface;
    Exact32 value = (exact_words(r.c1h)[k] * exact_words(r.muy)[face] + exact_words(r.c2h)[k]) * exact_words(r.v)[flat];
    if (p.has_msf) {
        if (p.reciprocal)
            value = value * exact_words(r.msfv)[face];
        else
            value = value / exact_words(r.msfv)[face];
    }
    exact_words(r.rv)[flat] = value;
}

kernel void wrf_exact_omega_column_scan(constant WrfExactOmegaColumnScanResources &r [[buffer(0)]],
                              constant WrfExactOmegaColumnScanParams &p [[buffer(1)]],
                              uint col [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny) * uint(p.nx);
    if (col >= ncol) return;
    const int j = int(col) / p.nx;
    const int i = int(col) - j * p.nx;
    Exact32 dmdt = 0.0f;

    // Pass one: source-order horizontal divergence and sequential column sum.
    for (int k = 0; k < p.nz; ++k) {
        const int u0 = (k * p.ny + j) * (p.nx + 1) + i;
        const int v0 = (k * (p.ny + 1) + j) * p.nx + i;
        const Exact32 du = exact_words(r.ru)[u0 + 1] - exact_words(r.ru)[u0];
        const Exact32 dv = exact_words(r.rv)[v0 + p.nx] - exact_words(r.rv)[v0];
        const Exact32 bracket = Exact32(p.rdx) * du + Exact32(p.rdy) * dv;
        const Exact32 weight = p.has_msf ? exact_words(r.msft)[col] * exact_words(r.dnw)[k]
                                       : exact_words(r.dnw)[k];
        const Exact32 divv = weight * bracket;
        dmdt = dmdt + divv;
        exact_words(r.ww)[(k + 1) * ncol + col] = divv;
    }

    // Pass two: recurrence has two distinct subtractions in the WOOF/CUDA
    // source order. The surface and top fluxes are explicitly zero.
    Exact32 w = 0.0f;
    exact_words(r.ww)[col] = w;
    for (int k = 1; k < p.nz; ++k) {
        const int at = k * ncol + col;
        const Exact32 level_weight = exact_words(r.dnw)[k - 1] * exact_words(r.c1h)[k - 1];
        w = w - level_weight * dmdt;
        w = w - exact_words(r.ww)[at];
        exact_words(r.ww)[at] = w;
    }
    exact_words(r.ww)[p.nz * ncol + col] = 0.0f;
}
