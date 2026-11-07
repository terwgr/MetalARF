#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_rrtmg_lw_slotpack_f32_bindings.metalh"
kernel void hexv842_mpas_rrtmg_lw_slotpack_f32(constant Hexv842MpasRrtmgLwSlotpackF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwSlotpackF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(ulong(gid)>=ulong(p.ng)*p.nm)return;
uint row=gid/p.nm,k=gid%p.nm,col=uint(r.indices[row]);ulong src=ulong(k)*p.nc+col,base=ulong(row)*12*p.nm;
r.model[base+0*p.nm+k]=r.p3d[src];
r.model[base+1*p.nm+k]=r.t3d[src];
r.model[base+2*p.nm+k]=r.dz[src];
r.model[base+3*p.nm+k]=r.qv[src];
r.model[base+4*p.nm+k]=r.qc[src];
r.model[base+5*p.nm+k]=r.qi[src];
r.model[base+6*p.nm+k]=r.qs[src];
r.model[base+7*p.nm+k]=r.cldfra[src];
r.model[base+8*p.nm+k]=p.radii?r.re_cloud[src]:0.0f;
r.model[base+9*p.nm+k]=p.radii?r.re_ice[src]:0.0f;
r.model[base+10*p.nm+k]=p.radii?r.re_snow[src]:0.0f;
r.model[base+11*p.nm+k]=r.pi[src];
ulong out=ulong(row)*2*(p.nm+1)+k;
r.interfaces[out]=r.p8w[src];r.interfaces[out+p.nm+1]=r.t8w[src];
if(k==0){
 r.interfaces[ulong(row)*2*(p.nm+1)+p.nm]=r.p8w[ulong(p.nm)*p.nc+col];
 r.interfaces[ulong(row)*2*(p.nm+1)+2*p.nm+1]=r.t8w[ulong(p.nm)*p.nc+col];
r.surface[ulong(row)*6+0]=r.tsk[col];
r.surface[ulong(row)*6+1]=r.emiss[col];
r.surface[ulong(row)*6+2]=r.xland[col];
r.surface[ulong(row)*6+3]=r.xice[col];
r.surface[ulong(row)*6+4]=r.snow[col];
r.surface[ulong(row)*6+5]=r.xlat[col];
for(int j=0;j<p.nz;++j)r.ozone[ulong(row)*p.nz+j]=r.o3clim[ulong(j)*p.nc+col];
}
}
