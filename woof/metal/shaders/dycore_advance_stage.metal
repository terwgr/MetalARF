#include <metal_stdlib>
#include "advance_stage_theta_bindings.metalh"
#include "advance_stage_u_bindings.metalh"
#include "advance_stage_v_bindings.metalh"
#include "advance_stage_w_bindings.metalh"

using namespace metal;

// _advance_stage is expressed as separate CuPy ufuncs in dycore.py. Preserve
// those FP32 operation boundaries in both Metal pipeline policies.
#pragma clang fp contract(off)

static inline ulong cell_index(uint k, uint j, uint i, uint ny, uint nx) {
    return (ulong(k) * ulong(ny) + ulong(j)) * ulong(nx) + ulong(i);
}

static inline float cell_mu(device const float *mub2d,
                            device const float *mup,
                            ulong cell) {
    return mub2d[cell] + mup[cell];
}

static inline float u_face_mu(device const float *mub2d,
                              device const float *mup,
                              uint j, uint face, uint ny, uint nx) {
    const uint i = face % nx;
    const uint previous_i = (i + nx - 1) % nx;
    const ulong here = ulong(j) * ulong(nx) + ulong(i);
    const ulong previous = ulong(j) * ulong(nx) + ulong(previous_i);
    const float mu_here = cell_mu(mub2d, mup, here);
    const float mu_previous = cell_mu(mub2d, mup, previous);
    return 0.5f * (mu_here + mu_previous);
}

static inline float v_face_mu(device const float *mub2d,
                              device const float *mup,
                              uint face, uint i, uint ny, uint nx) {
    const uint j = face % ny;
    const uint previous_j = (j + ny - 1) % ny;
    const ulong here = ulong(j) * ulong(nx) + ulong(i);
    const ulong previous = ulong(previous_j) * ulong(nx) + ulong(i);
    const float mu_here = cell_mu(mub2d, mup, here);
    const float mu_previous = cell_mu(mub2d, mup, previous);
    return 0.5f * (mu_here + mu_previous);
}

kernel void advance_stage_theta(
    constant AdvanceStageThetaResources &r [[buffer(0)]],
    constant AdvanceStageThetaParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny) * uint(p.nx);
    const uint count = uint(p.nz) * ncol;
    if (flat >= count) return;
    const uint k = flat / ncol;
    const uint j = (flat / uint(p.nx)) % uint(p.ny);
    const uint i = flat % uint(p.nx);
    const ulong cell = cell_index(k, j, i, uint(p.ny), uint(p.nx));
    const ulong horizontal = ulong(j) * ulong(p.nx) + ulong(i);
    const float thb = p.base3d ? r.thb[cell] : r.thb[k];
    const float mu0 = r.mub2d[horizontal] + r.mup0[horizontal];
    const float mu = r.mub2d[horizontal] + r.mup[horizontal];
    const float c1_mu0 = r.c1h[k] * mu0;
    const float coef_old = c1_mu0 + r.c2h[k];
    const float theta0 = thb + r.thp0[cell];
    const float numerator_old = coef_old * theta0;
    const float tendency = p.has_msf ? r.rth_t[cell] * r.msft[horizontal]
                                     : r.rth_t[cell];
    const float numerator = numerator_old + p.dt_eff * tendency;
    const float coef_new = r.c1h[k] * mu + r.c2h[k];
    r.thp[cell] = numerator / coef_new - thb;
}

kernel void advance_stage_u(
    constant AdvanceStageUResources &r [[buffer(0)]],
    constant AdvanceStageUParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const uint width = uint(p.nx + 1);
    const uint nface = uint(p.ny) * width;
    const uint count = uint(p.nz) * nface;
    if (flat >= count) return;
    const uint k = flat / nface;
    const uint face = flat - k * nface;
    const uint j = face / width;
    const uint i = face - j * width;
    const float mu0 = u_face_mu(r.mub2d, r.mup0, j, i,
                                uint(p.ny), uint(p.nx));
    const float mu = u_face_mu(r.mub2d, r.mup, j, i,
                               uint(p.ny), uint(p.nx));
    const float coef_old = r.c1h[k] * mu0 + r.c2h[k];
    const float tendency = p.has_msf ? r.ru_t[flat] * r.msfu[face]
                                     : r.ru_t[flat];
    const float numerator = coef_old * r.u0[flat] + p.dt_eff * tendency;
    const float coef_new = r.c1h[k] * mu + r.c2h[k];
    r.u[flat] = numerator / coef_new;
}

kernel void advance_stage_v(
    constant AdvanceStageVResources &r [[buffer(0)]],
    constant AdvanceStageVParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const uint width = uint(p.nx);
    const uint height = uint(p.ny + 1);
    const uint nface = height * width;
    const uint count = uint(p.nz) * nface;
    if (flat >= count) return;
    const uint k = flat / nface;
    const uint face = flat - k * nface;
    const uint j = face / width;
    const uint i = face - j * width;
    const float mu0 = v_face_mu(r.mub2d, r.mup0, j, i,
                                uint(p.ny), uint(p.nx));
    const float mu = v_face_mu(r.mub2d, r.mup, j, i,
                               uint(p.ny), uint(p.nx));
    const float coef_old = r.c1h[k] * mu0 + r.c2h[k];
    const float tendency = p.has_msf ? r.rv_t[flat] * r.msfv[face]
                                     : r.rv_t[flat];
    const float numerator = coef_old * r.v0[flat] + p.dt_eff * tendency;
    const float coef_new = r.c1h[k] * mu + r.c2h[k];
    r.v[flat] = numerator / coef_new;
}

kernel void advance_stage_w(
    constant AdvanceStageWResources &r [[buffer(0)]],
    constant AdvanceStageWParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny) * uint(p.nx);
    const uint count = uint(p.nz + 1) * ncol;
    if (flat >= count) return;
    const uint k = flat / ncol;
    const uint horizontal = flat - k * ncol;
    const float mu0 = r.mub2d[horizontal] + r.mup0[horizontal];
    const float mu = r.mub2d[horizontal] + r.mup[horizontal];
    const float coef_old = r.c1f[k] * mu0 + r.c2f[k];
    const float tendency = p.has_msf ? r.rw_t[flat] * r.msft[horizontal]
                                     : r.rw_t[flat];
    const float numerator = coef_old * r.w0[flat] + p.dt_eff * tendency;
    const float coef_new = r.c1f[k] * mu + r.c2f[k];
    r.w[flat] = numerator / coef_new;
}
