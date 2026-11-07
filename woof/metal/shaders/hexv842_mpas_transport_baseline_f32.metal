#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_mpas_transport_baseline_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexv842_mpas_transport_baseline_f32(
    constant Hexv842MpasTransportBaselineF32Resources &r [[buffer(0)]],
    constant Hexv842MpasTransportBaselineF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int q = int(tid), plane = p.nlev * p.ncells;
    if (q >= p.nspecies * plane) return;
    r.baseline[q] = q % p.ncells < p.nsolve
        ? mpas_add(r.before[q], mpas_div(mpas_mul(p.dt, r.source[q]), r.rho_old[q % plane]))
        : r.before[q];
}
