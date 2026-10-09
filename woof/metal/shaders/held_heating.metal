#include <metal_stdlib>
#include "held_heating_bindings.metalh"

using namespace metal;

// CUDA expresses every step with an explicit round-to-nearest FP32 intrinsic.
// Keep these as separate primitive operations and prevent multiply-add
// contraction so each statement has the corresponding operation boundary.
#pragma clang fp contract(off)

kernel void held_heating(
    constant HeldHeatingResources &r [[buffer(0)]],
    constant HeldHeatingParams &q [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong flat = ulong(gid);
    const ulong nz = ulong(q.nz), ny = ulong(q.ny), nx = ulong(q.nx);
    const ulong plane = ny * nx;
    const ulong count = nz * plane;
    if (flat >= count) return;

    // CUDA maps consecutive threads across x, then y, then eta level.
    const ulong k = flat / plane;
    const ulong col = flat - k * plane;

    // Preserve CUDA's __fadd_rn / __fmul_rn / __fdiv_rn sequence:
    // mu = mub + mup; coupling = c1*mu + c2; heating = coupling*hd;
    // optionally divide by msft; then add to the existing tendency.
    const float mu = r.mub[col] + r.mup[col];
    const float c1mu = r.c1[k] * mu;
    const float coupling = c1mu + r.c2[k];
    const float heating = coupling * r.hd[flat];
    const float mapped = q.has_msf ? heating / r.msft[col] : heating;
    r.rth[flat] = r.rth[flat] + mapped;
}
