#include <metal_stdlib>
#include "urban_bem_flat_flux_pack_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)
static inline float ubmp_add(float a,float b){return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubmp_sub(float a,float b){return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubmp_mul(float a,float b){return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)));}

constant uint BEM_PACK_NZ = 18u;
constant uint BEM_PACK_B = 15u;

// Repack bb_flux_flat's [column,direction,level] outputs into BEP_BEM's
// column-major prognostic fields and select the source-order roof terms
// sfr_avb(1,ibui)/lfrb(1,ibui) at each building's nlev.
kernel void urban_bem_flat_flux_pack_fragment(
    constant UrbanBemFlatFluxPackFragmentResources &r [[buffer(0)]],
    constant UrbanBemFlatFluxPackFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if(col>=uint(p.ncol))return;
    device const int *active=r.active_cols,*nlev=r.building_nlev;
    device const float *sf=r.sf,*lf=r.lf;
    device float *sfr=r.sfr,*lfr=r.lfr,*sfg=r.sfg,*lfg=r.lfg;
    device float *hsrof=r.hsrof,*lsrof=r.lsrof;
    device int *error=r.error;
    error[col]=0;
    if(active[col]==0)return;
    if(p.gather_only==0)for(uint z=0u;z<BEM_PACK_NZ;++z)
        for(uint id=0u;id<2u;++id){
            const ulong src=(ulong(col)*2ul+id)*BEM_PACK_NZ+z;
            const ulong dst=(ulong(col)*BEM_PACK_NZ+z)*2ul+id;
            if(z==0u){sfg[ulong(col)*2ul+id]=sf[src];lfg[ulong(col)*2ul+id]=lf[src];}
            else {sfr[dst]=sf[src];lfr[dst]=lf[src];}
        }
    const int cls=r.utype[col]-1;
    if(cls<0||cls>=11){error[col]=7431;return;}
    const float green_fraction=r.class_float[1206ul+uint(cls)];
    for(uint b=0u;b<BEM_PACK_B;++b){
        const int level=nlev[ulong(col)*BEM_PACK_B+b];
        const ulong out=ulong(col)*BEM_PACK_B+b;
        hsrof[out]=0.0f;lsrof[out]=0.0f;
        if(level<0||level>=int(BEM_PACK_NZ)){error[col]=7430;return;}
        if(level>0){
            const ulong src=(ulong(col)*BEM_PACK_NZ+uint(level))*2ul;
            hsrof[out]=ubmp_add(ubmp_mul(ubmp_sub(1.0f,green_fraction),sfr[src]),
                               ubmp_mul(green_fraction,r.sfrv[src]));
            lsrof[out]=lfr[src];
        }
    }
}
