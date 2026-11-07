#include <metal_stdlib>
using namespace metal;
#include "hexv842_sw_aerosol_gather_f32_bindings.metalh"
kernel void hexv842_sw_aerosol_gather_f32(constant Hexv842SwAerosolGatherF32Resources &r [[buffer(0)]],
 constant Hexv842SwAerosolGatherF32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 int t=int(tid),w=14*p.nl;if(t>=p.nd*w)return;int c=t/w,j=t%w,src=r.indices[c];
 r.tau_out[t]=r.tau_in[src*w+j];r.ssa_out[t]=r.ssa_in[src*w+j];r.asy_out[t]=r.asy_in[src*w+j];
}
