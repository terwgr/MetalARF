#include <metal_stdlib>
#include "urban_bem_indoor_state_update_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubms_add(float a,float b){return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubms_sub(float a,float b){return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubms_mul(float a,float b){return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubms_div(float a,float b){return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b)));}

// WRF bem_bem module_sf_bem.F:809-811. One item is one building floor.
// This writes staged air-temperature/humidity outputs only; the caller must
// scatter them with the full BEM state transaction after all material solves.
kernel void urban_bem_indoor_state_update_fragment(
    constant UrbanBemIndoorStateUpdateFragmentResources &r [[buffer(0)]],
    constant UrbanBemIndoorStateUpdateFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    device const int *active=r.active_cols,*nlev=r.building_nlev;
    device const float *tlev=r.tlev,*humlev=r.humlev,*qb=r.qb,*vollev=r.vollev;
    device const float *rhoint=r.rhoint,*latent=r.latent,*hsroo=r.hsroo,*hlroo=r.hlroo;
    device const float *hsneed=r.hsneed,*hlneed=r.hlneed;
    device float *tlev_next=r.tlev_next,*humlev_next=r.humlev_next;device int *error=r.error;
    if(item>=uint(p.nitem))return;
    const uint lev=item%18u,b=(item/18u)%15u,col=item/(18u*15u);
    error[item]=0;
    if(col>=uint(p.ncol))return;
    const ulong source=(ulong(col)*15ul+b)*18ul+lev;
    tlev_next[item]=tlev[item];humlev_next[item]=humlev[item];
    if(active[col]==0||nlev[ulong(col)*15ul+b]<=int(lev))return;
    if(!(qb[item]>0.0f&&vollev[item]>0.0f&&rhoint[item]>0.0f&&latent[item]>0.0f)||
       !isfinite(qb[item])||!isfinite(vollev[item])||!isfinite(rhoint[item])||!isfinite(latent[item])){
        error[item]=3841;return;
    }
    tlev_next[item]=ubms_add(tlev[source],ubms_mul(ubms_div(p.dt,qb[item]),ubms_sub(hsroo[item],hsneed[item])));
    humlev_next[item]=ubms_add(humlev[source],ubms_mul(ubms_div(p.dt,ubms_mul(ubms_mul(vollev[item],rhoint[item]),latent[item])),ubms_sub(hlroo[item],hlneed[item])));
    if(!isfinite(tlev_next[item])||!isfinite(humlev_next[item]))error[item]=3842;
}
