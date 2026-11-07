#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_mpas_ozone_min_upper_f32_bindings.metalh"
#include "hexv842_mpas_ozone_vertical_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexv842_mpas_ozone_min_upper_f32(
    constant Hexv842MpasOzoneMinUpperF32Resources &r [[buffer(0)]],
    constant Hexv842MpasOzoneMinUpperF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid) return;
    int low=p.nlev;
    for(int i=0;i<p.nsolve;i++) low=min(low,r.upper[i]);
    r.start[0]=low;
}
kernel void hexv842_mpas_ozone_vertical_f32(
    constant Hexv842MpasOzoneVerticalF32Resources &r [[buffer(0)]],
    constant Hexv842MpasOzoneVerticalF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    int c=int(tid);
    if(c>=p.nsolve) return;
    float pressure=r.pmid[p.k*p.ncells+c];
    if(p.pascal) pressure=mpas_div(pressure,100.0f);
    int upper=r.upper[c];
    for(int kk=r.start[0];kk<p.nlev-1;kk++)
        if(r.pin[kk]<pressure && pressure<=r.pin[kk+1]) {upper=kk;break;}
    r.upper[c]=upper;
    float value;
    if(pressure<r.pin[0]) value=mpas_div(mpas_mul(r.ozone[c],pressure),r.pin[0]);
    else if(pressure>r.pin[p.nlev-1]) value=r.ozone[(p.nlev-1)*p.ncells+c];
    else {
        float dpu=mpas_sub(pressure,r.pin[upper]),dpl=mpas_sub(r.pin[upper+1],pressure);
        value=mpas_div(mpas_add(mpas_mul(r.ozone[upper*p.ncells+c],dpl),
                                mpas_mul(r.ozone[(upper+1)*p.ncells+c],dpu)),mpas_add(dpl,dpu));
    }
    r.out[p.k*p.ncells+c]=value;
}
