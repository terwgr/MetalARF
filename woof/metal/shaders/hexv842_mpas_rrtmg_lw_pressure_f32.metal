#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_mpas_rrtmg_lw_pressure_f32_bindings.metalh"
kernel void hexv842_mpas_rrtmg_lw_pressure_f32(constant Hexv842MpasRrtmgLwPressureF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwPressureF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(gid>=uint(p.nc))return;
 ulong at=ulong(gid)*12*p.nl;
 r.status[gid]=soft32_less(as_type<uint>(r.a[at]),as_type<uint>(r.a[at+1]))?1:0;
}
