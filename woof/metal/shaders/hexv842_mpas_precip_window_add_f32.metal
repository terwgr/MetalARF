#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_mpas_precip_window_add_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexv842_mpas_precip_window_add_f32(constant Hexv842MpasPrecipWindowAddF32Resources &r [[buffer(0)]],constant Hexv842MpasPrecipWindowAddF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {

    const int i = int(gid);
    if (i >= p.n) return;
    r.nsteps_bl[i] = r.nsteps_bl[i] + 1;
    if (p.have_nc) {
        r.rainncv_bl[i] = mpas_add(r.rainncv_bl[i],r.rainncv[i]);
        const float frozen = mpas_mul(r.sr[i],r.rainncv[i]);
        r.frozen_bl[i] = mpas_add(r.frozen_bl[i],frozen);
    }
    if (p.have_c) r.raincv_bl[i] = mpas_add(r.raincv_bl[i],r.raincv[i]);
    if (p.have_sg) {
        r.snowncv_bl[i] = mpas_add(r.snowncv_bl[i],r.snowncv[i]);
        r.graupelncv_bl[i] = mpas_add(r.graupelncv_bl[i],r.graupelncv[i]);
    }
    float ratio = r.sr[i];
    if (r.nsteps_bl[i] > 1 && soft32_less(0u,as_type<uint>(r.rainncv_bl[i]))) {
        const float q = mpas_div(r.frozen_bl[i],r.rainncv_bl[i]);
        ratio = soft32_less(as_type<uint>(q),as_type<uint>(1.0f)) ? q : 1.0f;
    }
    r.sr_bl[i] = ratio;
}
