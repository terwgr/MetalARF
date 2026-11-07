#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "wrf_exact_total_theta_bindings.metalh"
#include "wrf_exact_adopt_theta_bindings.metalh"
using namespace metal;

// bandwidth_glue.cu glue_total_theta uses __fadd_rn(thb, thp).
kernel void wrf_exact_total_theta(
    constant WrfExactTotalThetaResources &r [[buffer(0)]],
    constant WrfExactTotalThetaParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    ulong plane = ulong(p.ny) * ulong(p.nx);
    if (ulong(gid) >= ulong(p.nz) * plane) return;
    ulong base = p.base3d ? ulong(gid) : ulong(gid) / plane;
    reinterpret_cast<device uint *>(r.out)[gid] = soft32_add(
        as_type<uint>(r.thb[base]), as_type<uint>(r.thp[gid]));
}

kernel void wrf_exact_adopt_theta(
    constant WrfExactAdoptThetaResources &r [[buffer(0)]],
    constant WrfExactAdoptThetaParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    ulong plane = ulong(p.ny) * ulong(p.nx);
    if (ulong(gid) >= ulong(p.nz) * plane) return;
    ulong base = p.base3d ? ulong(gid) : ulong(gid) / plane;
    device uint *thp = reinterpret_cast<device uint *>(r.thp);
    uint total = soft32_add(as_type<uint>(r.thb[base]), thp[gid]);
    thp[gid] = soft32_sub(total, as_type<uint>(300.0f));
}
