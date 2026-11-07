#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_noah_landuse_f32_bindings.metalh"
#pragma clang fp contract(off)

kernel void hexv842_mpas_noah_landuse_f32(
    constant Hexv842MpasNoahLanduseF32Resources &r [[buffer(0)]],
    constant Hexv842MpasNoahLanduseF32Params &p [[buffer(1)]],
    uint c [[thread_position_in_grid]]) {
    if(c>=uint(p.n)) return;
    device float *x=r.floats+c*14;
    device const int *ix=r.integers+c*9;
    r.status[c]=0;
    if(ix[4]) return;
int season=(ix[1]<105 || ix[1]>=288)?2:1;
if(x[0]<0.0f)season=3-season;
int category=ix[0]==0?ix[2]:ix[0];
// mpas_atmphys_landuse.F:268-273: native table category reads are unchecked.
if(category<1 || category>100 || ((x[2]>0.5f || x[3]>=x[13]) && (ix[3]<1 || ix[3]>100))) { r.status[c]=-1; return; }
int is=(season-1)*100+category-1,ice=(season-1)*100+ix[3]-1;
if(!ix[5])x[4]=cu_div(r.tables[0+(is)],100.0f);
x[8]=x[4];
if(x[2]>0.5f){
 x[4]=cu_div(r.tables[0+(ice)],100.0f);
 x[8]=ix[5]?x[1]:cu_div(x[4],mpas_add(1.0f,r.tables[6000+(is)]));
}
x[10]=cu_div(r.tables[4800+(is)],100.0f);x[5]=cu_div(r.tables[3600+(is)],100.0f);x[6]=x[5];
x[7]=r.tables[2400+(is)];x[9]=x[7];
if(ix[7])x[11]=r.tables[1200+(is)];
if(ix[8])x[12]=0.0001f;
if(x[3]>=x[13]){
 x[4]=cu_div(r.tables[0+(ice)],100.0f);x[7]=r.tables[2400+(ice)];
 if(ix[6]){
  x[8]=mpas_add(mpas_mul(x[3],x[4]),mpas_mul(mpas_sub(1.0f,x[3]),0.08f));
  x[9]=mpas_add(mpas_mul(x[3],x[7]),mpas_mul(mpas_sub(1.0f,x[3]),0.98f));
 }else{x[8]=x[4];x[9]=x[7];}
 x[10]=cu_div(r.tables[4800+(ice)],100.0f);x[5]=cu_div(r.tables[3600+(ice)],100.0f);x[6]=x[5];
 if(ix[7])x[11]=r.tables[1200+(ice)];
}
}
