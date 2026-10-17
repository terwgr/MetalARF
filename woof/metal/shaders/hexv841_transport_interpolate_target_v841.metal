// Hex v8.4.1 scalar transport: time-interpolated target density (cuda_transport_v841.py transport_interpolate_target_v841).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_transport_interpolate_target_v841_bindings.metalh"
#pragma clang fp contract(off)

kernel void hexv841_transport_interpolate_target_v841(
    constant Hexv841TransportInterpolateTargetV841Resources &r [[buffer(0)]],
    constant Hexv841TransportInterpolateTargetV841Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int index = int(tid);
    if (index >= p.nvalues) return;
    const float weight = p.weight;
    r.target[index] = mpas_add(
        mpas_mul(mpas_sub(1.0f, weight), r.rho_old[index]),
        mpas_mul(weight, r.rho_new[index]));
}
