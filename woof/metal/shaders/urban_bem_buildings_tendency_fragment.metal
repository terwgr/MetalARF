#include <metal_stdlib>
#include "urban_bem_buildings_tendency_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubt_add(float a,float b){return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubt_sub(float a,float b){return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubt_mul(float a,float b){return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubt_div(float a,float b){return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b)));}

// WRF bb_buildings' per-direction building loop, module_sf_bep_bem.F:2832-2888.
// One item is (column, street direction, urban level). It accumulates the two
// wall faces in building order, preserving the source FP32 add/sub/div order.
// Flat ground/roof tendencies are the preceding bb_flux_flat outputs and are
// copied into the exact arrays consumed by bb_urban_meso.
kernel void urban_bem_buildings_tendency_fragment(
    constant UrbanBemBuildingsTendencyFragmentResources &r [[buffer(0)]],
    constant UrbanBemBuildingsTendencyFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= uint(p.nitem)) return;
    device const int *active=r.active_cols,*nlev=r.building_nlev,*utype=r.utype;
    device const float *durb=r.d_urb,*pb=r.pb,*bs=r.bs,*cf=r.class_float;
    device const float *interp=r.interp;
    device const float *sflev=r.sflev,*lflev=r.lflev,*sfvlev=r.sfvlev,*lfvlev=r.lfvlev;
    device const float *wall_uva=r.wall_uva,*wall_vva=r.wall_vva;
    device const float *wall_uvb=r.wall_uvb,*wall_vvb=r.wall_vvb;
    device const float *wall_sfw=r.wall_sfw,*wall_sfwin=r.wall_sfwin,*wall_evb=r.wall_evb;
    device const float *flat_uhb=r.flat_uhb,*flat_vhb=r.flat_vhb;
    device const float *flat_thb=r.flat_thb,*flat_ehb=r.flat_ehb,*flat_qhb=r.flat_qhb;
    device float *uva=r.uva,*vva=r.vva,*uvb=r.uvb,*vvb=r.vvb;
    device float *tvb=r.tvb,*tvb_ac=r.tvb_ac,*evb=r.evb,*qvb=r.qvb;
    device float *uhb=r.uhb,*vhb=r.vhb,*thb=r.thb,*ehb=r.ehb,*qhb=r.qhb_flat;
    device int *error=r.error;

    const uint lev=item%18u;
    const uint direction=(item/18u)%2u;
    const uint col=item/36u;
    error[item]=0;
    const ulong flat=(ulong(col)*2ul+direction)*18ul+lev;
    uhb[flat]=flat_uhb[flat]; vhb[flat]=flat_vhb[flat];
    thb[flat]=flat_thb[flat]; ehb[flat]=flat_ehb[flat]; qhb[flat]=flat_qhb[flat];
    const ulong face0=(ulong(col)*4ul+ulong(direction)*2ul)*18ul+lev;
    const ulong face1=face0+18ul;
    uva[face0]=0.0f;uva[face1]=0.0f;
    vva[face0]=0.0f;vva[face1]=0.0f;
    uvb[face0]=0.0f;uvb[face1]=0.0f;
    vvb[face0]=0.0f;vvb[face1]=0.0f;
    tvb[face0]=0.0f;tvb[face1]=0.0f;
    tvb_ac[face0]=0.0f;tvb_ac[face1]=0.0f;
    evb[face0]=0.0f;evb[face1]=0.0f;
    qvb[face0]=0.0f;qvb[face1]=0.0f;
    if(col>=uint(p.ncol)||active[col]==0)return;

    const int cls=utype[col]-1;
    if(cls<0||cls>=p.nurbm){error[item]=7411;return;}
    const float pwin=cf[1085+uint(cls)];
    const float width=bs[ulong(col)*2ul+direction];
    const float density=pb[ulong(col)*19ul+lev+1ul];
    const ulong output[2]={face0,face1};
    float acc_uva[2]={0.0f,0.0f},acc_vva[2]={0.0f,0.0f};
    float acc_uvb[2]={0.0f,0.0f},acc_vvb[2]={0.0f,0.0f};
    float acc_tvb[2]={0.0f,0.0f},acc_tac[2]={0.0f,0.0f};
    float acc_evb[2]={0.0f,0.0f},acc_qvb[2]={0.0f,0.0f};
    if(!(pwin>=0.0f&&pwin<=1.0f)||!isfinite(pwin)||!(width>0.0f)||!isfinite(width)){
        error[item]=7410;return;
    }
    if(density>0.0f){
        const float da=interp[ulong(col)*7ul*18ul+5ul*18ul+lev];
        if(!(da>0.0f)||!isfinite(da)){error[item]=7410;return;}
        for(uint b=0u;b<15u;++b){
            if(int(lev)>=nlev[ulong(col)*15ul+b])continue;
            const float depth=durb[ulong(col)*18ul+b];
            if(!(depth>=0.0f)||!isfinite(depth)){error[item]=7410;return;}
            const float weight=ubt_div(depth,density);
            for(uint side=0u;side<2u;++side){
                const uint face=direction*2u+side;
                const ulong wall=(((ulong(col)*15ul+b)*18ul+lev)*4ul)+face;
                const uint local=side;
                acc_uva[local]=ubt_add(acc_uva[local],ubt_mul(weight,wall_uva[wall]));
                acc_vva[local]=ubt_add(acc_vva[local],ubt_mul(weight,wall_vva[wall]));
                acc_uvb[local]=ubt_add(acc_uvb[local],ubt_mul(weight,wall_uvb[wall]));
                acc_vvb[local]=ubt_add(acc_vvb[local],ubt_mul(weight,wall_vvb[wall]));
                acc_evb[local]=ubt_add(acc_evb[local],ubt_mul(weight,wall_evb[wall]));

                const float wall_flux=ubt_add(
                    ubt_mul(wall_sfw[wall],ubt_sub(1.0f,pwin)),
                    ubt_mul(wall_sfwin[wall],pwin));
                const float wall_term=ubt_div(ubt_div(ubt_mul(weight,wall_flux),da),p.cp_u);
                acc_tvb[local]=ubt_sub(acc_tvb[local],wall_term);

                float transfer=ubt_mul(ubt_div(1.0f,4.0f),weight);
                transfer=ubt_mul(transfer,ubt_sub(
                    sfvlev[(ulong(col)*15ul+b)*18ul+lev],
                    sflev[(ulong(col)*15ul+b)*18ul+lev]));
                transfer=ubt_div(transfer,ubt_mul(p.dz_u,width));
                transfer=ubt_div(transfer,da);
                transfer=ubt_div(transfer,p.cp_u);
                acc_tvb[local]=ubt_sub(acc_tvb[local],transfer);

                float acoustic=ubt_mul(ubt_div(1.0f,4.0f),weight);
                acoustic=ubt_mul(acoustic,-sflev[(ulong(col)*15ul+b)*18ul+lev]);
                acoustic=ubt_div(acoustic,ubt_mul(p.dz_u,width));
                acoustic=ubt_div(acoustic,da);
                acoustic=ubt_div(acoustic,p.cp_u);
                acc_tac[local]=ubt_sub(acc_tac[local],acoustic);

                float moisture=ubt_mul(ubt_div(1.0f,4.0f),weight);
                moisture=ubt_mul(moisture,ubt_sub(
                    lfvlev[(ulong(col)*15ul+b)*18ul+lev],
                    lflev[(ulong(col)*15ul+b)*18ul+lev]));
                moisture=ubt_div(moisture,ubt_mul(p.dz_u,width));
                moisture=ubt_div(moisture,da);
                moisture=ubt_div(moisture,p.latent);
                acc_qvb[local]=ubt_sub(acc_qvb[local],moisture);
            }
        }
    }
    uva[output[0]]=acc_uva[0];uva[output[1]]=acc_uva[1];
    vva[output[0]]=acc_vva[0];vva[output[1]]=acc_vva[1];
    uvb[output[0]]=acc_uvb[0];uvb[output[1]]=acc_uvb[1];
    vvb[output[0]]=acc_vvb[0];vvb[output[1]]=acc_vvb[1];
    tvb[output[0]]=acc_tvb[0];tvb[output[1]]=acc_tvb[1];
    tvb_ac[output[0]]=acc_tac[0];tvb_ac[output[1]]=acc_tac[1];
    evb[output[0]]=acc_evb[0];evb[output[1]]=acc_evb[1];
    qvb[output[0]]=acc_qvb[0];qvb[output[1]]=acc_qvb[1];
}
