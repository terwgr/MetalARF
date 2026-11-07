#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_sw_outputs_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexv842_sw_outputs_f32(constant Hexv842SwOutputsF32Resources &r [[buffer(0)]],
    constant Hexv842SwOutputsF32Params &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
 int c=int(tid),nc=p.nc,nz=p.nz;if(c>=nc)return;int nlev=nz+2;
 #define V(a,k) r.a[c*nlev+(k)]
 #define S(i) r.scalars[(i)*nc+c]
 S(0)=r.cosine[c];if(soft32_less_equal(as_type<uint>(r.cosine[c]),0u))return;
 S(1)=mpas_sub(V(down,0),V(up,0));S(2)=mpas_sub(mpas_sub(V(down,nlev-1),V(up,nlev-1)),mpas_sub(V(downc,nlev-1),V(upc,nlev-1)));
 S(3)=V(up,nlev-1);S(4)=V(upc,nlev-1);S(5)=V(down,nlev-1);S(6)=V(downc,nlev-1);
 S(7)=V(up,0);S(8)=V(upc,0);S(9)=V(down,0);S(10)=V(downc,0);
 S(11)=V(direct,0);S(12)=mpas_div(S(11),r.cosine[c]);S(13)=mpas_sub(V(down,0),V(direct,0));
 for(int k=0;k<nz;k++)r.tendency[c*nz+k]=mpas_div(mpas_div(r.heat[c*(nz+1)+k],86400.f),r.exner[c*nz+k]);
 #undef V
 #undef S
}
