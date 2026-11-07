#include <metal_stdlib>
#include "wsm_support.metalh"
#include "physics_correct_mode_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void thompson_cloud_column_water(
    constant ThompsonCloudColumnWaterResources &r [[buffer(0)]],
    constant ThompsonCloudColumnWaterParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny * p.nx);
    if (gid >= ncol) return;
    Soft8 total(0);
    for (int k = 0; k < p.nz; ++k) {
        const uint id = uint(k) * ncol + gid;
        const Sf input_qv(r.qv[id]);
        const Sf qv = soft32_is_nan(as_type<uint>(r.qv[id])) || input_qv > Sf(1.0e-10f) ? input_qv : Sf(1.0e-10f);
        const float density = aa_div(Sf(0.622f) * Sf(r.pressure[id]),
            Sf(287.04f) * Sf(r.temperature[id]) * (qv + Sf(0.622f)));
        total = total + (Soft8(r.qc[id]) * Soft8(density)) * Soft8(r.dz[id]);
    }
    r.water[gid] = total.bits;
}

kernel void thompson_cloud_book(
    constant ThompsonCloudBookResources &r [[buffer(0)]],
    constant ThompsonCloudBookParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny * p.nx);
    if (gid >= ncol) return;
    const int i = int(gid) % p.nx, j = int(gid) / p.nx;
    const bool ring = i < p.spec_zone || i >= p.nx - p.spec_zone
                   || j < p.spec_zone || j >= p.ny - p.spec_zone;
    const Soft8 factor = SD8(ulong(uint(p.factor_lo)) | (ulong(uint(p.factor_hi)) << 32));
    const float settled = ring ? 0.0f : float((SD8(r.before[gid]) - SD8(r.after[gid])) * factor);
    r.cloudncv[gid] = settled;
    r.cloudnc[gid] = Sf(r.cloudnc[gid]) + Sf(settled);
}

kernel void ntiedtke_correct_forcing(
    constant NtiedtkeCorrectForcingResources &r [[buffer(0)]],
    constant NtiedtkeCorrectForcingParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.nz * p.ncol)) return;
    const Sf frame = Sf(r.qvf[gid]) + Sf(1.0f);
    r.pqte[gid] = aa_div(Sf(r.pqte[gid]), frame * frame);
}

kernel void ntiedtke_correct_water(
    constant NtiedtkeCorrectWaterResources &r [[buffer(0)]],
    constant NtiedtkeCorrectWaterParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.nz * p.ncol)) return;
    const uint k = gid / uint(p.ncol), i = gid % uint(p.ncol);
    const uint source = (uint(p.nz) - 1u - k) * uint(p.ncol) + i;
    const Sf frame = Sf(r.qv3d[gid]) + Sf(1.0f);
    r.rqvcuten[gid] = (Sf(r.pqte[source]) - Sf(r.zqq[source])) * frame;
    r.rqccuten[gid] = Sf(r.rqccuten[gid]) * frame;
    r.rqicuten[gid] = Sf(r.rqicuten[gid]) * frame;
}
