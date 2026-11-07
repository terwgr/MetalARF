#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_rrtmg_lw_ozstart_f32_bindings.metalh"
kernel void hexv842_mpas_rrtmg_lw_ozstart_f32(constant Hexv842MpasRrtmgLwOzstartF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwOzstartF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(gid>0)return;
int kk=p.nz-1;for(int c=0;c<p.nc;++c)kk=min(kk,r.upper[c]);r.start[0]=kk;
}
