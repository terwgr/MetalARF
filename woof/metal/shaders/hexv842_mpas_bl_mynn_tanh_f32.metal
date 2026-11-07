#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_mpas_bl_mynn_tanh.metalh"
#include "hexv842_mpas_bl_mynn_tanh_f32_bindings.metalh"
kernel void hexv842_mpas_bl_mynn_tanh_f32(constant Hexv842MpasBlMynnTanhF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnTanhF32Params &p [[buffer(1)]],uint i [[thread_position_in_grid]]) {if(i<uint(p.count))r.output[i]=as_type<float>(mynn_tanh_bits(as_type<uint>(r.input[i])));}
