#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_rrtmg_lw_slotscatter_f32_bindings.metalh"
kernel void hexv842_mpas_rrtmg_lw_slotscatter_f32(constant Hexv842MpasRrtmgLwSlotscatterF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwSlotscatterF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(ulong(gid)>=ulong(p.ng)*p.nm)return;
uint row=gid/p.nm,k=gid%p.nm,col=uint(r.indices[row]);ulong dst=ulong(k)*p.nc+col;
r.tend_out[dst]=r.tendency[ulong(row)*p.nm+k];
if(p.write_radii){r.re_cloud_out[dst]=p.radii?r.radii[(ulong(row)*3)*p.nm+k]:0.0f;r.re_ice_out[dst]=p.radii?r.radii[(ulong(row)*3+1)*p.nm+k]:0.0f;r.re_snow_out[dst]=p.radii?r.radii[(ulong(row)*3+2)*p.nm+k]:0.0f;}
if(k==0){
r.glw[col]=r.flux[ulong(row)*11+0];
r.olr[col]=r.flux[ulong(row)*11+1];
r.lwcf[col]=r.flux[ulong(row)*11+2];
r.lwupt[col]=r.flux[ulong(row)*11+3];
r.lwuptc[col]=r.flux[ulong(row)*11+4];
r.lwdnt[col]=r.flux[ulong(row)*11+5];
r.lwdntc[col]=r.flux[ulong(row)*11+6];
r.lwupb[col]=r.flux[ulong(row)*11+7];
r.lwupbc[col]=r.flux[ulong(row)*11+8];
r.lwdnb[col]=r.flux[ulong(row)*11+9];
r.lwdnbc[col]=r.flux[ulong(row)*11+10];
}
}
