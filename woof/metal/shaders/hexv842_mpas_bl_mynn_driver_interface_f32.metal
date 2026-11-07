#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_bl_mynn_driver_interface_f32_bindings.metalh"
kernel void hexv842_mpas_bl_mynn_driver_interface_f32(constant Hexv842MpasBlMynnDriverInterfaceF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnDriverInterfaceF32Params &p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    uint nl=uint(p.nl), nc=uint(p.nc);
    if(i>=nc*nl)return;
    uint col=i/nl,k=i%nl;
    r.output[i]=r.input[col*(nl+1)+k];
}
