#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "thompson_column_adapter_bindings.metalh"

using namespace metal;

#pragma clang fp contract(off)

// Entry thermodynamics of woof/core/thompson_column.py: full theta, the
// pre-microphysics theta parked in h_diabatic, temperature and layer depth.
kernel void thompson_column_prepare(
    constant ThompsonColumnPrepareResources &r [[buffer(0)]],
    constant ThompsonColumnPrepareParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.size)) return;
    const uint k = idx / uint(p.columns);
    const float theta = r.thb[p.thb_full != 0 ? idx : k] + r.thp[idx];
    r.th[idx] = theta;
    r.h_diabatic[idx] = theta;
    r.temperature[idx] = theta * r.pii[idx];
    const uint lo = p.phb_full != 0 ? idx : k;
    const uint hi = p.phb_full != 0 ? idx + uint(p.columns) : k + 1u;
    const float z_lo = as_type<float>(soft32_div(
        as_type<uint>(r.phb[lo] + r.php[idx]), as_type<uint>(p.gravity)));
    const float z_hi = as_type<float>(soft32_div(
        as_type<uint>(r.phb[hi] + r.php[idx + uint(p.columns)]),
        as_type<uint>(p.gravity)));
    r.dz[idx] = z_hi - z_lo;
}

// Exit: theta from temperature, the clamped increment into theta' and the
// heating rate kept for the next step's tendency.
kernel void thompson_column_finish(
    constant ThompsonColumnFinishResources &r [[buffer(0)]],
    constant ThompsonColumnFinishParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.size)) return;
    const float theta = as_type<float>(soft32_div(
        as_type<uint>(r.temperature[idx]), as_type<uint>(r.pii[idx])));
    r.th[idx] = theta;
    if (p.no_mp_heating != 0) {
        r.h_diabatic[idx] = 0.0f;
        return;
    }
    float increment = theta - r.h_diabatic[idx];
    if (increment < -p.limit) increment = -p.limit;
    else if (increment > p.limit) increment = p.limit;
    r.thp[idx] = r.thp[idx] + increment;
    r.h_diabatic[idx] = as_type<float>(soft32_div(
        as_type<uint>(increment), as_type<uint>(p.dt)));
}
