#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_rrtmg_lw_slotcounts_f32_bindings.metalh"
kernel void hexv842_mpas_rrtmg_lw_slotcounts_f32(constant Hexv842MpasRrtmgLwSlotcountsF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwSlotcountsF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(gid>=uint(p.nc))return;
float top=cu_div(r.p8w[ulong(p.nm)*p.nc+gid],100.0f);
r.counts[gid]=p.nm+max(int(mpas_add(cu_div(top,4.0f),0.5f)),1);
}
