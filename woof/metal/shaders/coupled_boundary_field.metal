#include <metal_stdlib>
#include "coupled_boundary_field_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float total_mu_at(device const float *mub,
                                device const float *mup,
                                ulong q) {
    return mub[q] + mup[q];
}

static inline float hybrid_mass(device const float *mub,
                                device const float *mup,
                                float c1, float c2, int j, int i,
                                int ny, int nx) {
    ulong q = ulong(j) * ulong(nx) + ulong(i);
    float mu = total_mu_at(mub, mup, q);
    float weighted = c1 * mu;
    return weighted + c2;
}

static inline float u_face_mass(device const float *mub,
                                device const float *mup,
                                float c1, float c2, int j, int i,
                                int ny, int nx) {
    int cell_i;
    float mu;
    if (i == 0) {
        cell_i = 0;
        mu = total_mu_at(mub, mup, ulong(j) * ulong(nx) + ulong(cell_i));
    } else if (i == nx) {
        cell_i = nx - 1;
        mu = total_mu_at(mub, mup, ulong(j) * ulong(nx) + ulong(cell_i));
    } else {
        ulong right = ulong(j) * ulong(nx) + ulong(i);
        float right_mu = total_mu_at(mub, mup, right);
        float left_mu = total_mu_at(mub, mup, right - 1);
        float sum = right_mu + left_mu;
        mu = 0.5f * sum;
    }
    float weighted = c1 * mu;
    return weighted + c2;
}

static inline float v_face_mass(device const float *mub,
                                device const float *mup,
                                float c1, float c2, int j, int i,
                                int ny, int nx) {
    float mu;
    if (j == 0) {
        mu = total_mu_at(mub, mup, ulong(i));
    } else if (j == ny) {
        mu = total_mu_at(mub, mup, ulong(ny - 1) * ulong(nx) + ulong(i));
    } else {
        ulong north = ulong(j) * ulong(nx) + ulong(i);
        float north_mu = total_mu_at(mub, mup, north);
        float south_mu = total_mu_at(mub, mup, north - ulong(nx));
        float sum = north_mu + south_mu;
        mu = 0.5f * sum;
    }
    float weighted = c1 * mu;
    return weighted + c2;
}

kernel void coupled_boundary_field(
        constant CoupledBoundaryFieldResources &r [[buffer(0)]],
        constant CoupledBoundaryFieldParams &q [[buffer(1)]],
        uint tid [[thread_position_in_grid]]) {
    ulong count = ulong(q.nz) * ulong(q.ny) * ulong(q.nx);
    if (ulong(tid) >= count) return;
    int k = int(ulong(tid) / (ulong(q.ny) * ulong(q.nx)));
    ulong rem = ulong(tid) - ulong(k) * ulong(q.ny) * ulong(q.nx);
    int j = int(rem / ulong(q.nx));
    int i = int(rem - ulong(j) * ulong(q.nx));
    ulong out_index = ulong(tid);
    // Map factors are horizontal 2-D tables. `out_index` includes the model
    // level and would walk beyond msfu/msfv as soon as k > 0.
    const ulong horizontal_index = ulong(j) * ulong(q.nx) + ulong(i);
    float value = r.target[out_index];
    float coupled;
    if (q.kind == 0) {
        coupled = u_face_mass(r.mub2d, r.mup, r.c1h[k], r.c2h[k],
                               j, i, q.mny, q.mnx);
        float product = coupled * value;
        if (q.has_msf) product = product / r.msfu[horizontal_index];
        r.out[out_index] = product;
        return;
    }
    if (q.kind == 1) {
        coupled = v_face_mass(r.mub2d, r.mup, r.c1h[k], r.c2h[k],
                               j, i, q.mny, q.mnx);
        float product = coupled * value;
        if (q.has_msf) product = product / r.msfv[horizontal_index];
        r.out[out_index] = product;
        return;
    }
    if (q.kind == 3) {
        coupled = hybrid_mass(r.mub2d, r.mup, r.c1f[k], r.c2f[k],
                              j, i, q.mny, q.mnx);
        r.out[out_index] = coupled * value;
        return;
    }
    coupled = hybrid_mass(r.mub2d, r.mup, r.c1h[k], r.c2h[k],
                          j, i, q.mny, q.mnx);
    if (q.kind == 2) {
        ulong base_index = q.thb_3d
            ? (ulong(k) * ulong(q.mny) + ulong(j)) * ulong(q.mnx) + ulong(i)
            : ulong(k);
        float total_theta = r.thb[base_index] + value;
        float theta_perturbation = total_theta - 300.0f;
        r.out[out_index] = coupled * theta_perturbation;
        return;
    }
    // Generic external scalar boundary fields use half-level mu coupling.
    r.out[out_index] = coupled * value;
}
