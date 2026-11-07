#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_mpas_ozone_latitude_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexv842_mpas_ozone_latitude_f32(
    constant Hexv842MpasOzoneLatitudeF32Resources &r [[buffer(0)]],
    constant Hexv842MpasOzoneLatitudeF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int idx=int(tid);
    if(idx>=p.ncells*p.nlev*12) return;
    const int col=idx%p.ncells, lev=(idx/p.ncells)%p.nlev, month=idx/(p.ncells*p.nlev);
    const float y=mpas_div(r.lat[col],3.1415926f/180.f);
    int i1=0,i2=0;
    if(y>r.lat_ozone[63]) i1=i2=63;
    else if(y<r.lat_ozone[0]) i1=i2=0;
    else {
        for(int i=0;i<63;i++) if(y>=r.lat_ozone[i] && y<r.lat_ozone[i+1]) {i1=i;i2=i+1;break;}
    }
    const float d=mpas_sub(r.lat_ozone[i2],r.lat_ozone[i1]),dy=mpas_sub(y,r.lat_ozone[i1]);
    const float a=r.table[(month*64+i1)*p.nlev+lev],b=r.table[(month*64+i2)*p.nlev+lev];
    r.monthly[idx]=d==0.f ? a : mpas_add(a,mpas_div(mpas_mul(mpas_sub(b,a),dy),d));
}
