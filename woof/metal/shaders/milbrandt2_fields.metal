#include <metal_stdlib>
#include "milbrandt2_prepare_bindings.metalh"
#include "milbrandt2_finish_bindings.metalh"

using namespace metal;

// Match the CUDA/Fortran float operation sequence. In particular, keep each
// addition and multiplication as a separate FP32 operation.
#pragma clang fp contract(off)

kernel void milbrandt2_prepare(
    constant Milbrandt2PrepareResources &r [[buffer(0)]],
    constant Milbrandt2PrepareParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const int plane = p.ny * p.nx;
    if (col >= uint(plane)) return;

    // WRF moist_physics_prep_em: thb+thp, Exner, and absolute temperature.
    for (int k = 0; k < p.nz; ++k) {
        const int id = k * plane + int(col);
        const int thb_id = p.thb_3d != 0 ? id : k;
        const float full_theta = r.thb[thb_id] + r.thp[id];
        const float exner = pow(r.p[id] / p.p0, p.rcp);
        r.theta[id] = full_theta;
        r.pii[id] = exner;
        r.temperature[id] = full_theta * exner;
        r.h_diabatic[id] = full_theta;
    }

    // WRF module_big_step_utilities_em.F: p8w(i,kms,j), interpolated from
    // the lowest mass pressure and the lowest three full geopotential heights.
    const int phb0 = p.phb_3d != 0 ? int(col) : 0;
    const int phb1 = p.phb_3d != 0 ? plane + int(col) : 1;
    const int phb2 = p.phb_3d != 0 ? 2 * plane + int(col) : 2;
    const float z0 = (r.phb[phb0] + r.php[int(col)]) / p.gravity;
    const float z1 = (r.phb[phb1] + r.php[plane + int(col)]) / p.gravity;
    const float z2 = (r.phb[phb2] + r.php[2 * plane + int(col)]) / p.gravity;
    const float zh0 = 0.5f * (z0 + z1);
    const float zh1 = 0.5f * (z1 + z2);
    const float w1 = (z0 - zh1) / (zh0 - zh1);
    const float w2 = 1.0f - w1;
    r.psfc[col] = w1 * r.p[int(col)] + w2 * r.p[plane + int(col)];
}

kernel void milbrandt2_finish(
    constant Milbrandt2FinishResources &r [[buffer(0)]],
    constant Milbrandt2FinishParams &p [[buffer(1)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= uint(p.nz * p.ny * p.nx)) return;

    // The driver receives temperature and reconstructs theta = T/pii.
    const float theta = r.temperature[id] / r.pii[id];
    r.theta[id] = theta;
    if (p.no_mp_heating != 0) {
        r.h_diabatic[id] = 0.0f;
        return;
    }
    float mpten = theta - r.h_diabatic[id];
    mpten = min(p.limit, mpten);
    mpten = max(-p.limit, mpten);
    r.thp[id] += mpten;
    r.h_diabatic[id] = mpten / p.dt;
}
