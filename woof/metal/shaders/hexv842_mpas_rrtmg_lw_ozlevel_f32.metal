#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_rrtmg_lw_ozlevel_f32_bindings.metalh"
kernel void hexv842_mpas_rrtmg_lw_ozlevel_f32(constant Hexv842MpasRrtmgLwOzlevelF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwOzlevelF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(gid>=uint(p.nc))return;
float pressure=cu_div(r.p3d[ulong(p.k)*p.nc+gid],100.0f);int u=r.upper[gid];
for(int kk=r.start[0];kk<p.nz-1;++kk){
 if(soft32_less(as_type<uint>(r.pin[kk]),as_type<uint>(pressure)) &&
    soft32_less_equal(as_type<uint>(pressure),as_type<uint>(r.pin[kk+1]))){u=kk;break;}
}
r.upper[gid]=u;
float value;
if(soft32_less(as_type<uint>(pressure),as_type<uint>(r.pin[0])))value=cu_div(mpas_mul(r.ozone[gid],pressure),r.pin[0]);
else if(soft32_greater(as_type<uint>(pressure),as_type<uint>(r.pin[p.nz-1])))value=r.ozone[ulong(p.nz-1)*p.nc+gid];
else {
 float dpu=mpas_sub(pressure,r.pin[u]),dpl=mpas_sub(r.pin[u+1],pressure);
 value=cu_div(mpas_add(mpas_mul(r.ozone[ulong(u)*p.nc+gid],dpl),mpas_mul(r.ozone[ulong(u+1)*p.nc+gid],dpu)),mpas_add(dpl,dpu));
}
r.o3vmr[ulong(p.k)*p.nc+gid]=value;
}
