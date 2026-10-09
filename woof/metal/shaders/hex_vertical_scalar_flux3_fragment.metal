#include <metal_stdlib>
#include "hex_vertical_scalar_flux3_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

// Atmosphere v8.2.3 flux3 uses the positive third-order correction. This is
// the column-independent arithmetic primitive; boundary/stencil selection is
// prepared by the host transport route.
kernel void hex_vertical_scalar_flux3(
        constant HexVerticalScalarFlux3Resources &r [[buffer(0)]],
        constant HexVerticalScalarFlux3Params &q [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    if (q.nitems <= 0 || gid >= uint(q.nitems)) return;

    const float w = r.vertical_velocity[gid];
    if (q.third_order_coefficient < 0.0f) {
        // A negative coefficient selects the source's second-order
        // near-boundary/interface formula. For this mode q_im2/q_ip1 carry
        // fnm/fnp, while q_im1/q_i carry tracer(i)/tracer(i-1).
        r.flux[gid] = w * (r.q_im2[gid] * r.q_im1[gid]
            + r.q_ip1[gid] * r.q_i[gid]);
        return;
    }
    const float q0 = r.q_im2[gid];
    const float q1 = r.q_im1[gid];
    const float q2 = r.q_i[gid];
    const float q3 = r.q_ip1[gid];
    const float base = w * (7.0f * (q2 + q1) - (q3 + q0)) / 12.0f;
    const float correction = q.third_order_coefficient * abs(w)
        * ((q3 - q0) - 3.0f * (q2 - q1)) / 12.0f;
    r.flux[gid] = base + correction;
}
