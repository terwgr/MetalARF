#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_held_heating_bindings.metalh"
#include "wrf_exact_radiation_sum_bindings.metalh"
#include "wrf_exact_radiation_mass_bindings.metalh"
using namespace metal;

kernel void wrf_exact_held_heating(
    constant WrfExactHeldHeatingResources &r [[buffer(0)]],
    constant WrfExactHeldHeatingParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong plane = ulong(p.ny) * ulong(p.nx), at = ulong(gid);
    if (at >= ulong(p.nz) * plane) return;
    const ulong col = at % plane, k = at / plane;
#define A(name) reinterpret_cast<device const Exact32 *>(r.name)
    const Exact32 mu = A(mub)[col] + A(mup)[col];
    const Exact32 coupling = A(c1)[k] * mu + A(c2)[k];
    Exact32 heating = coupling * A(hd)[at];
    if (p.has_msf) heating = heating / A(msft)[col];
    reinterpret_cast<device Exact32 *>(r.rth)[at] = A(rth)[at] + heating;
#undef A
}

// _radiation_theta_total: round the sum before publishing the SW residual.
kernel void wrf_exact_radiation_sum(
    constant WrfExactRadiationSumResources &r [[buffer(0)]],
    constant WrfExactRadiationSumParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.count)) return;
    const Exact32 lw = reinterpret_cast<device const Exact32 *>(r.longwave)[gid];
    device Exact32 *sw = reinterpret_cast<device Exact32 *>(r.shortwave);
    const Exact32 total = lw + sw[gid];
    reinterpret_cast<device Exact32 *>(r.total)[gid] = total;
    if (p.publish_residual) sw[gid] = total - lw;
}

// PhysicsDriver forms c1h * mut + c2h before scaling the held rate.
kernel void wrf_exact_radiation_mass(
    constant WrfExactRadiationMassResources &r [[buffer(0)]],
    constant WrfExactRadiationMassParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.count)) return;
    const uint k = gid / uint(p.ncol), col = gid % uint(p.ncol);
#define A(name) reinterpret_cast<device const Exact32 *>(r.name)
    const Exact32 mass = A(c1)[k] * A(mut)[col] + A(c2)[k];
    const Exact32 rate = A(rate)[gid];
    reinterpret_cast<device Exact32 *>(r.output)[gid] =
        p.divide ? rate / mass : rate * mass;
#undef A
}
