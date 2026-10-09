// One thread per column for the reference Betts-Miller ascent and closure.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
#include "global_betts_miller_fragment_bindings.metalh"
using namespace metal;
static inline ulong gbmax(ulong a,ulong b){return soft64_greater(a,b)?a:b;}
static inline ulong gbmin(ulong a,ulong b){return soft64_less(a,b)?a:b;}
static inline ulong gbqsat(ulong t,ulong p,ulong eps){
    t=gbmin(gbmax(t,0x4062c00000000000ul),0x4075e00000000000ul);
    ulong denom=gbmax(soft64_sub(t,0x403da66666666666ul),0x3ff0000000000000ul);
    ulong e=soft64_mul(0x408319999999999aul,soft64_plm_exp(soft64_div(
        soft64_mul(0x4031ab851eb851ecul,soft64_sub(t,0x4071126666666666ul)),denom)));
    e=gbmin(e,soft64_mul(0x3fee666666666666ul,p));
    return soft64_div(soft64_mul(eps,e),gbmax(soft64_sub(p,e),0x3ff0000000000000ul));
}
kernel void global_betts_miller_fragment(
    constant GlobalBettsMillerFragmentResources &r [[buffer(0)]],
    constant GlobalBettsMillerFragmentParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    if(column>=uint(p.ncolumn))return;
    const ulong zero=0ul,one=0x3ff0000000000000ul;
    thread ulong parcel[128];
    const uint plane=uint(p.ncolumn);
    const int bottom=p.nlevel-1;
    const uint ib=uint(bottom)*plane+column;
    parcel[bottom]=r.temperature[ib];
    ulong qsat_bot=gbqsat(parcel[bottom],r.pressure[ib],r.epsilon[0]);
    ulong pq=gbmin(r.qv[ib],qsat_bot);
    bool saturated=soft64_greater_equal(pq,soft64_sub(qsat_bot,0x3d719799812dea11ul));
    int lcl=saturated?bottom:-1;
    for(int k=bottom-1;k>=0;--k){
        const uint idx=uint(k)*plane+column, below=idx+plane;
        ulong pk=r.pressure[idx],pk1=r.pressure[below],ptbelow=parcel[k+1];
        ulong ratio=soft64_div(pk,pk1);
        ulong dlnp=soft64_plm_log(ratio);
        ulong dry_t=soft64_mul(ptbelow,soft64_plm_pow(ratio,r.rd_cp[0]));
        ulong qs_prev=gbqsat(ptbelow,pk1,r.epsilon[0]);
        ulong numerator=soft64_add(soft64_mul(r.rd[0],ptbelow),soft64_mul(r.lv[0],qs_prev));
        ulong latent=soft64_div(soft64_mul(soft64_mul(r.lv2[0],qs_prev),r.epsilon[0]),
            soft64_mul(r.rd[0],soft64_mul(ptbelow,ptbelow)));
        ulong moist_rate=soft64_div(numerator,soft64_add(r.cp[0],latent));
        ulong moist_t=soft64_add(ptbelow,soft64_mul(moist_rate,dlnp));
        bool is_sat=saturated||soft64_greater_equal(pq,gbqsat(dry_t,pk,r.epsilon[0]));
        parcel[k]=is_sat?moist_t:dry_t;
        if(is_sat)pq=gbqsat(parcel[k],pk,r.epsilon[0]);
        if(is_sat&&!saturated)lcl=k;
        saturated=is_sat;
    }
    ulong qfac=soft64_add(one,soft64_mul(0x3fe3851eb851eb85ul,pq));
    int cloud_top=0; bool has_cloud=false;
    for(int k=0;k<p.nlevel;++k){
        uint idx=uint(k)*plane+column;
        bool buoyant=soft64_greater(soft64_mul(parcel[k],qfac),r.virtual_temperature[idx])&&k<=lcl&&lcl>=0;
        if(buoyant&&!has_cloud){cloud_top=k;has_cloud=true;}
    }
    int base=clamp(lcl,0,p.nlevel-1);
    ulong pbase=r.pressure[uint(base)*plane+column],ptop=r.pressure[uint(cloud_top)*plane+column];
    bool deep=has_cloud&&soft64_greater_equal(soft64_sub(pbase,ptop),r.minimum_depth[0]);
    ulong drying_neg=zero,moistening=zero,heating_raw=zero,cloud_mass=zero;
    for(int k=0;k<p.nlevel;++k){
        if(!deep||k<cloud_top||k>lcl)continue;
        uint idx=uint(k)*plane+column;
        ulong mass=soft64_div(r.dp[idx],r.gravity[0]);
        ulong dtr=soft64_mul(soft64_sub(parcel[k],r.temperature[idx]),r.relax[0]);
        ulong refq=soft64_mul(r.rh[0],gbqsat(parcel[k],r.pressure[idx],r.epsilon[0]));
        ulong dq=soft64_mul(soft64_sub(refq,r.qv[idx]),r.relax[0]);
        drying_neg=soft64_add(drying_neg,soft64_mul(soft64_less(dq,zero)?dq:zero,mass));
        moistening=soft64_add(moistening,soft64_mul(soft64_greater(dq,zero)?dq:zero,mass));
        heating_raw=soft64_add(heating_raw,soft64_mul(dtr,mass));
        cloud_mass=soft64_add(cloud_mass,mass);
    }
    ulong drying=soft64_negate_bits(drying_neg);
    bool precipitating=deep&&soft64_greater(drying,moistening);
    ulong scale=one;
    if(deep&&!precipitating&&soft64_greater(moistening,zero))
        scale=soft64_div(drying,gbmax(moistening,0x01b56e1fc2f8f359ul));
    ulong dq_mass=zero;
    for(int k=0;k<p.nlevel;++k){
        uint idx=uint(k)*plane+column;
        r.temperature_out[idx]=r.temperature[idx];
        r.qv_out[idx]=r.qv[idx];
        if(!deep||k<cloud_top||k>lcl)continue;
        ulong refq=soft64_mul(r.rh[0],gbqsat(parcel[k],r.pressure[idx],r.epsilon[0]));
        ulong dq=soft64_mul(soft64_sub(refq,r.qv[idx]),r.relax[0]);
        if(soft64_greater(dq,zero))dq=soft64_mul(dq,scale);
        dq_mass=soft64_add(dq_mass,soft64_mul(dq,soft64_div(r.dp[idx],r.gravity[0])));
        r.qv_out[idx]=soft64_add(r.qv[idx],dq);
    }
    ulong precip_col=precipitating?soft64_negate_bits(dq_mass):zero;
    ulong correction=zero;
    if(deep){
        ulong numerator=soft64_sub(soft64_mul(r.lv[0],precip_col),soft64_mul(r.cp[0],heating_raw));
        correction=soft64_div(numerator,soft64_mul(r.cp[0],gbmax(cloud_mass,0x3eb0c6f7a0b5ed8dul)));
    }
    for(int k=0;k<p.nlevel;++k){
        if(!deep||k<cloud_top||k>lcl)continue;
        uint idx=uint(k)*plane+column;
        ulong dtr=soft64_mul(soft64_sub(parcel[k],r.temperature[idx]),r.relax[0]);
        r.temperature_out[idx]=soft64_add(r.temperature[idx],soft64_add(dtr,correction));
    }
    r.precip[column]=precip_col;
    r.deep[column]=deep?1u:0u;
}
