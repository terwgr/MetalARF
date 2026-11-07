#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_sw_layout_f32_bindings.metalh"
kernel void hexv842_sw_layout_f32(constant Hexv842SwLayoutF32Resources &r [[buffer(0)]],
 constant Hexv842SwLayoutF32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 int t=int(tid);if(t>=p.nc*p.nl)return;int k=t/p.nc,c=t%p.nc;
 int column=c*p.nl+k;float v=p.reverse?r.src[column]:r.src[t];
 if(p.divide)v=mpas_div(v,p.divisor);
 r.dst[p.reverse?t:column]=v;
}
