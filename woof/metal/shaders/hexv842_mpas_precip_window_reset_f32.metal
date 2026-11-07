#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_mpas_precip_window_reset_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexv842_mpas_precip_window_reset_f32(constant Hexv842MpasPrecipWindowResetF32Resources &r [[buffer(0)]],constant Hexv842MpasPrecipWindowResetF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {

    const int i = int(gid);
    if (i >= p.n) return;
    r.rainncv_bl[i] = 0.0f; r.raincv_bl[i] = 0.0f; r.snowncv_bl[i] = 0.0f; r.graupelncv_bl[i] = 0.0f;
    r.frozen_bl[i] = 0.0f; r.nsteps_bl[i] = 0;
}
