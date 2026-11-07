#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_mpas_ozone_update_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexv842_mpas_ozone_update_f32(
    constant Hexv842MpasOzoneUpdateF32Resources &r [[buffer(0)]],
    constant Hexv842MpasOzoneUpdateF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int i=int(tid),plane=p.nlev*p.ncells;
    if(i>=plane || i%p.ncells>=p.nsolve) return;
    r.out[i]=mpas_div(mpas_add(mpas_mul(r.monthly[p.month2*plane+i],float(p.d1)),
                        mpas_mul(r.monthly[p.month1*plane+i],float(p.d2))),float(p.d1+p.d2));
}
