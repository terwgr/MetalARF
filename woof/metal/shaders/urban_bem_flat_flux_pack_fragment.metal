#include <metal_stdlib>
#include "urban_bem_flat_flux_pack_fragment_bindings.metalh"
using namespace metal;

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
    for(uint z=0u;z<BEM_PACK_NZ;++z)
        for(uint id=0u;id<2u;++id){
            const ulong src=(ulong(col)*2ul+id)*BEM_PACK_NZ+z;
            const ulong dst=(ulong(col)*BEM_PACK_NZ+z)*2ul+id;
            if(z==0u){sfg[ulong(col)*2ul+id]=sf[src];lfg[ulong(col)*2ul+id]=lf[src];}
            else {sfr[dst]=sf[src];lfr[dst]=lf[src];}
        }
    for(uint b=0u;b<BEM_PACK_B;++b){
        const int level=nlev[ulong(col)*BEM_PACK_B+b];
        const ulong out=ulong(col)*BEM_PACK_B+b;
        hsrof[out]=0.0f;lsrof[out]=0.0f;
        if(level<0||level>=int(BEM_PACK_NZ)){error[col]=7430;return;}
        if(level>0){
            const ulong src=(ulong(col)*2ul)*BEM_PACK_NZ+uint(level);
            hsrof[out]=sf[src];lsrof[out]=lf[src];
        }
    }
}
