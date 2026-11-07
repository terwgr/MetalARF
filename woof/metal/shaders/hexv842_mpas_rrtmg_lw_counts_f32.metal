#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#pragma clang fp contract(off)
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_DV(a,b) cu_div(a,b)
#include "hexv842_mpas_rrtmg_lw_counts_f32_bindings.metalh"
kernel void hexv842_mpas_rrtmg_lw_counts_f32(constant Hexv842MpasRrtmgLwCountsF32Resources &r [[buffer(0)]], constant Hexv842MpasRrtmgLwCountsF32Params &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {
if(c>=uint(p.nc))return;
float top=RLW_DV(r.interfaces[ulong(c)*2*(p.nm+1)+p.nm],100.0f);
r.counts[c]=p.nm+max(int(RLW_AD(RLW_DV(top,4.0f),0.5f)),1);
}
