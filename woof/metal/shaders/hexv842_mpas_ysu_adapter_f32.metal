#include <metal_stdlib>
using namespace metal;
#include "soft_binary32.metalh"
#include "hexv842_mpas_ysu_adapter_f32_bindings.metalh"
#pragma clang fp contract(off)

kernel void hexv842_mpas_ysu_radiation_f32(constant Hexv842MpasYsuRadiationF32Resources &r [[buffer(0)]], constant Hexv842MpasYsuRadiationF32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.count)) return;
    r.rthraten[gid] = as_type<float>(soft32_add(as_type<uint>(r.lw[gid]), as_type<uint>(r.sw[gid])));
}

kernel void hexv842_mpas_ysu_theta_f32(constant Hexv842MpasYsuThetaF32Resources &r [[buffer(0)]], constant Hexv842MpasYsuThetaF32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.count)) return;
    r.rthblten[gid] = as_type<float>(soft32_div(as_type<uint>(r.ttnp[gid]), as_type<uint>(r.pi[gid])));
}
