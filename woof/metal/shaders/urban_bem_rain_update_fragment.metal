#include <metal_stdlib>
#include "urban_bem_rain_update_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmru_add(float a,float b){return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubmru_mul(float a,float b){return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubmru_div(float a,float b){return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b)));}

// WRF bb_column's post-radiation dg/dgr accumulation and physical bounds.
kernel void urban_bem_rain_update_fragment(
    constant UrbanBemRainUpdateFragmentResources &r [[buffer(0)]],
    constant UrbanBemRainUpdateFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if(col>=uint(p.ncol))return;
    device const int *active=r.active_cols;
    device const int *utype=r.utype;
    device const int *class_int=r.class_int;
    device const float *rain=r.rainbl;
    device const float *lfg=r.lfg;
    device const float *lfr=r.lfr;
    device float *dg=r.dg;
    device float *dgr=r.dgr;
    device int *error=r.error;
    error[col]=0;
    if(active[col]==0)return;
    int type=utype[col];
    if(p.nurbm<1||p.nurbm>11||type<1||type>p.nurbm){error[col]=7301;return;}
    int nd=class_int[uint(type)];
    if(nd<1||nd>2){error[col]=7302;return;}
    const ulong base=ulong(col)*2ul;
    for(int dir=0;dir<nd;++dir){
        ulong gi=base+uint(dir);
        if(dg[gi]<=1.0f)dg[gi]=ubmru_add(dg[gi],ubmru_add(rain[col],ubmru_div(ubmru_mul(lfg[gi],p.dt),p.latent)));
        if(dg[gi]<0.0f)dg[gi]=0.0f;
        if(dg[gi]>1.0f)dg[gi]=1.0f;
        for(int iz=1;iz<18;++iz){
            ulong ri=(ulong(col)*18ul+uint(iz))*2ul+uint(dir);
            if(dgr[ri]<=1.0f)dgr[ri]=ubmru_add(dgr[ri],ubmru_add(rain[col],ubmru_div(ubmru_mul(lfr[ri],p.dt),p.latent)));
            if(dgr[ri]<0.0f)dgr[ri]=0.0f;
            if(dgr[ri]>1.0f)dgr[ri]=1.0f;
        }
    }
}
