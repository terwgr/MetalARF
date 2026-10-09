#include <metal_stdlib>
#include "urban_bem_room_balance_inputs_fragment_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_pow.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmi_add(float a,float b){return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubmi_sub(float a,float b){return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubmi_mul(float a,float b){return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubmi_div(float a,float b){return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubmi_pow(float a,float b){return as_type<float>(soft_gfk_pow_bits(as_type<uint>(a),as_type<uint>(b)));}

// Prepare the exact per-floor inputs to bem_fluxvent/regtemp/reghum/air_cond.
// hsroo/hlroo already include occupancy, ventilation and the preceding
// bem_fluxcond result; the existing room-control kernel recomputes the same
// ventilation terms for its returned diagnostic/state fields.
kernel void urban_bem_room_balance_inputs_fragment(
    constant UrbanBemRoomBalanceInputsFragmentResources &r [[buffer(0)]],
    constant UrbanBemRoomBalanceInputsFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    device const int *active=r.active_cols,*utype=r.utype,*nlev=r.building_nlev,*class_int=r.class_int;
    device const float *cf=r.class_float,*bs=r.bs,*tlev_src=r.tlev_src,*hum_src=r.hum_src;
    device const float *interp=r.interp,*occ_s=r.occupancy_sensible,*occ_l=r.occupancy_latent,*hscond=r.hscond;
    device float *cpint=r.cpint,*rhoint=r.rhoint,*vollev=r.vollev,*tlev=r.tlev,*tout=r.tout;
    device float *latent=r.latent,*humout=r.humout,*rhoout=r.rhoout,*humlev=r.humlev,*beta=r.beta;
    device float *qb=r.qb,*hsroo=r.hsroo,*volroo=r.volroo,*hlroo=r.hlroo,*shumroo=r.shumroo;
    device float *cop=r.cop,*timeon=r.timeon,*timeoff=r.timeoff,*targtemp=r.targtemp,*gaptemp=r.gaptemp;
    device float *targhum=r.targhum,*gaphum=r.gaphum;device int *swcond=r.swcond;device int *error=r.error;
    if(item>=uint(p.nitem))return;
    const uint lev=item%18u,b=(item/18u)%15u,col=item/(18u*15u);error[item]=0;
    if(col>=uint(p.ncol)||active[col]==0||nlev[ulong(col)*15ul+b]<=int(lev)){
        cpint[item]=1004.0f;rhoint[item]=0.0f;vollev[item]=1.0f;tlev[item]=273.15f;tout[item]=273.15f;
        latent[item]=2.45e6f;humout[item]=0.0f;rhoout[item]=1.0f;humlev[item]=0.0f;beta[item]=0.0f;
        qb[item]=1.0f;hsroo[item]=0.0f;volroo[item]=1.0f;hlroo[item]=0.0f;shumroo[item]=0.0f;
        cop[item]=1.0f;timeon[item]=0.0f;timeoff[item]=0.0f;targtemp[item]=0.0f;gaptemp[item]=0.0f;
        targhum[item]=0.0f;gaphum[item]=0.0f;swcond[item]=0;return;
    }
    const int cls=utype[col]-1;if(cls<0||cls>=p.nurbm){error[item]=3831;return;}
    const ulong bld=ulong(col)*15ul+b,lev_i=ulong(col)*15ul*18ul+b*18ul+lev;
    const float tin=tlev_src[lev_i],hum=hum_src[lev_i],bw=bs[ulong(col)*2ul],bl=bs[ulong(col)*2ul+1ul];
    const ulong at=ulong(col)*7ul*18ul;
    const float theta=interp[at+2ul*18ul+lev],pressure=interp[at+4ul*18ul+lev];
    const float rhoout_v=interp[at+5ul*18ul+lev],humout_v=interp[at+6ul*18ul+lev];
    if(!(tin>0.0f&&bw>0.0f&&bl>0.0f&&pressure>0.0f)||!isfinite(tin)||!isfinite(hum)||!isfinite(bw)||!isfinite(bl)||!isfinite(pressure)) {error[item]=3832;return;}
    const float dry=ubmi_div(hum,ubmi_sub(1.0f,hum));
    const float rhoi=ubmi_div(pressure,ubmi_mul(ubmi_mul(287.0f,ubmi_add(1.0f,ubmi_mul(0.61f,dry))),tin));
    const float cpi=ubmi_mul(1004.0f,ubmi_add(1.0f,ubmi_mul(0.84f,dry)));
    const float vol=ubmi_mul(ubmi_mul(bw,bl),p.dz_u);
    const float temp_out=ubmi_mul(theta,ubmi_pow(ubmi_div(pressure,100000.0f),ubmi_div(287.0f,1004.0f)));
    const float beta_v=cf[1096+cls];
    const float hvent=ubmi_mul(ubmi_mul(ubmi_mul(ubmi_mul(ubmi_sub(1.0f,beta_v),cpi),rhoi),ubmi_div(vol,3600.0f)),ubmi_sub(temp_out,tin));
    const float lvent=ubmi_mul(ubmi_mul(ubmi_mul(ubmi_mul(ubmi_sub(1.0f,beta_v),2.45e6f),rhoi),ubmi_div(vol,3600.0f)),ubmi_sub(humout_v,hum));
    const float hgain=occ_s[bld],lgain=occ_l[bld];
    cpint[item]=cpi;rhoint[item]=rhoi;vollev[item]=vol;tlev[item]=tin;tout[item]=temp_out;
    latent[item]=2.45e6f;humout[item]=humout_v;rhoout[item]=rhoout_v;humlev[item]=hum;beta[item]=beta_v;
    qb[item]=ubmi_mul(ubmi_mul(rhoi,cpi),vol);hsroo[item]=ubmi_add(ubmi_add(hgain,hvent),hscond[item]);
    volroo[item]=vol;hlroo[item]=ubmi_add(lgain,lvent);shumroo[item]=hum;
    cop[item]=cf[1074+cls];timeon[item]=cf[1129+cls];timeoff[item]=cf[1140+cls];
    targtemp[item]=cf[1151+cls];gaptemp[item]=cf[1162+cls];targhum[item]=cf[1173+cls];gaphum[item]=cf[1184+cls];
    swcond[item]=class_int[23+cls];
}
