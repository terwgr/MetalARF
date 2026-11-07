#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_mpas_transport_handoff_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexv842_mpas_transport_handoff_f32(
    constant Hexv842MpasTransportHandoffF32Resources &r [[buffer(0)]],
    constant Hexv842MpasTransportHandoffF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int i = int(tid), plane = p.nlev * p.ncells;
    if (i >= plane) return;
    const int qi = p.iqv * plane + i;
    if (p.handoff) {
        const float dq = p.monotonic ? mpas_div(mpas_sub(r.after[qi], r.before[qi]), p.dt) : 0.0f;
        r.rqvdynten[i] = dq;
        if (i % p.ncells < p.nsolve) {
            const float fac = mpas_div(1.0f, mpas_add(1.0f, mpas_mul(p.rvord, r.after[qi])));
            const float theta = mpas_mul(r.theta_m[i], fac);
            const float vapor = mpas_mul(mpas_div(mpas_mul(theta, p.rv), p.rd), dq);
            r.rthdynten[i] = mpas_mul(fac, mpas_sub(r.rthdynten[i], vapor));
        }
    }
    for (int s = 0; s < p.nspecies; ++s) {
        const int si = s * plane + i;
        const uint word = as_type<uint>(r.after[si]);
        const uint mag = word & 0x7fffffffu;
        if ((word & 0x80000000u) && mag != 0u && mag <= 0x7f800000u) r.after[si] = 0.0f;
    }
}
