#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#pragma clang fp contract(off)
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_DV(a,b) cu_div(a,b)
#include "hexv842_mpas_rrtmg_lw_outputs_f32_bindings.metalh"
#define M(f,k) m[(f)*nm+(k)]
#define S(f) surface[(f)]
#define A(f,k) a[(f)*nl+(k)]
#define P(k) lev[(k)]
#define T(k) lev[nl+1+(k)]

static inline void mpas_lw_outputs_one(int nm,int nl,int icloud,device const float* m,device const float* a,device const float* u,device const float* d,device const float* uc,device const float* dc,device const float* hr,device float* tendency,device float* radii,device float* flux){
    // MPAS: module_ra_rrtmg_lw.F:12198-12254; driver _to_MPAS:613-719.
    flux[0]=d[0];flux[1]=u[nl];flux[2]=RLW_SU(uc[nl],u[nl]);
    flux[3]=u[nl];flux[4]=uc[nl];flux[5]=d[nl];flux[6]=dc[nl];
    flux[7]=u[0];flux[8]=uc[0];flux[9]=d[0];flux[10]=dc[0];
    for(int k=0;k<nm;++k){tendency[k]=RLW_DV(RLW_DV(hr[k],86400.0f),M(11,k));
        for(int j=0;j<3;++j)radii[j*nm+k]=icloud>0&&soft32_greater(as_type<uint>(M(7,k)),0u)?A(9+j,k):0.0f;}
}
kernel void hexv842_mpas_rrtmg_lw_outputs_f32(constant Hexv842MpasRrtmgLwOutputsF32Resources &r [[buffer(0)]], constant Hexv842MpasRrtmgLwOutputsF32Params &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {
if(c>=uint(p.nc))return;
mpas_lw_outputs_one(p.nm,p.nl,p.icloud,r.m+ulong(c)*12*p.nm,r.a+ulong(c)*12*p.nl,r.u+ulong(c)*(p.nl+1),r.d+ulong(c)*(p.nl+1),r.uc+ulong(c)*(p.nl+1),r.dc+ulong(c)*(p.nl+1),r.hr+ulong(c)*p.nl,r.tendency+ulong(c)*p.nm,r.radii+ulong(c)*3*p.nm,r.flux+ulong(c)*11);
}
