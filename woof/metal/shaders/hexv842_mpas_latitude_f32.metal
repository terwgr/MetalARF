#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_mpas_latitude_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexv842_mpas_latitude_f32(
    constant Hexv842MpasLatitudeF32Resources &r [[buffer(0)]],
    constant Hexv842MpasLatitudeF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int i=int(tid);
    if(i>=p.ncells) return;
    r.out[i]=mpas_div(r.lat[i],p.degrad);
}
