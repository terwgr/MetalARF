#include <metal_stdlib>
using namespace metal;
#include "soft_binary32.metalh"
#include "hexv842_sw_day_select_i32_bindings.metalh"
#include "hexv842_sw_day_gather_f32_bindings.metalh"
#include "hexv842_sw_day_scatter_f32_bindings.metalh"
kernel void hexv842_sw_day_select_i32(constant Hexv842SwDaySelectI32Resources &r [[buffer(0)]],
 constant Hexv842SwDaySelectI32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 if(tid)return;int n=0;for(int c=0;c<p.nc;c++)if(soft32_greater(as_type<uint>(r.cosine[c]),0u))r.indices[n++]=c;r.count[0]=n;
}
kernel void hexv842_sw_day_gather_f32(constant Hexv842SwDayGatherF32Resources &r [[buffer(0)]],
 constant Hexv842SwDayGatherF32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 int t=int(tid),nc=p.nc,nd=p.nd,nz=p.nz,nl=nz+1;if(t>=nd*(nl+1))return;
 int c=t/(nl+1),k=t%(nl+1),src=r.indices[c];
 if(k<nl){for(int f=0;f<15;f++)r.layers_out[(f*nd+c)*nl+k]=r.layers_in[(f*nc+src)*nl+k];r.height_out[c*nl+k]=r.height_in[src*nl+k];}
 for(int f=0;f<2;f++)r.interfaces_out[(f*nd+c)*(nl+1)+k]=r.interfaces_in[(f*nc+src)*(nl+1)+k];
 if(k<nz)r.exner_out[c*nz+k]=r.fields_in[(3*nc+src)*nz+k];
 if(k==0){r.cosine_out[c]=r.cosine_in[src];r.albedo_out[c]=r.surface_in[2*nc+src];r.latitude_out[c]=r.surface_in[src];}
}
kernel void hexv842_sw_day_scatter_f32(constant Hexv842SwDayScatterF32Resources &r [[buffer(0)]],
 constant Hexv842SwDayScatterF32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 int t=int(tid),nc=p.nc,nd=p.nd,nz=p.nz,nl=nz+2;if(t>=nd*nl)return;
 int c=t/nl,k=t%nl,dst=r.indices[c];
 if(k==0)for(int f=0;f<14;f++)r.scalars_out[f*nc+dst]=r.scalars_in[f*nd+c];
 if(k<nz)r.tendency_out[dst*nz+k]=r.tendency_in[c*nz+k];
 r.profiles_out[(0*nc+dst)*nl+k]=r.up[c*nl+k];
 r.profiles_out[(1*nc+dst)*nl+k]=r.upc[c*nl+k];
 r.profiles_out[(2*nc+dst)*nl+k]=r.down[c*nl+k];
 r.profiles_out[(3*nc+dst)*nl+k]=r.downc[c*nl+k];
}
