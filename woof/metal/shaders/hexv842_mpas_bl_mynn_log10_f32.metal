#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_bl_mynn_log10_f32_bindings.metalh"
#include "hexv842_mpas_bl_mynn_log10.metalh"
kernel void hexv842_mpas_bl_mynn_log10_f32(constant Hexv842MpasBlMynnLog10F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnLog10F32Params &p [[buffer(1)]],uint i [[thread_position_in_grid]]){if(i<uint(p.count))r.output[i]=as_type<float>(mynn_log10_bits(as_type<uint>(r.input[i])));}
