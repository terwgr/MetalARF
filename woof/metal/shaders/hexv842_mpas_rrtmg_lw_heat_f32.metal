#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_mpas_rrtmg_lw_heat_f32_bindings.metalh"
kernel void hexv842_mpas_rrtmg_lw_heat_f32(constant Hexv842MpasRrtmgLwHeatF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwHeatF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(ulong(gid)>=ulong(p.nc)*p.nl)return;
 uint col=gid/p.nl,k=gid%p.nl;
 r.heating[gid]=r.htr[ulong(col)*(p.nl+1)+k];
}
