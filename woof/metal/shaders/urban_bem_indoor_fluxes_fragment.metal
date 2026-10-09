#include <metal_stdlib>
#include "urban_bem_indoor_fluxes_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmf_add(float a,float b){return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubmf_sub(float a,float b){return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubmf_mul(float a,float b){return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubmf_pow4(float a){const float h=ubmf_mul(a,a);return ubmf_mul(h,h);}

// WRF bem_bem source order after bem_int_rsrad/rlrad: exterior wall/window
// radiation, roof radiation for the ordinary non-PV profile, interior
// wall/window radiation, then bem_hsinsflux and bem_fluxcond.
// One item is (column, building slot, urban level), packed row-major.
kernel void urban_bem_indoor_fluxes_fragment(
    constant UrbanBemIndoorFluxesFragmentResources &r [[buffer(0)]],
    constant UrbanBemIndoorFluxesFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    device const int *active=r.active_cols,*utype=r.utype,*nlev=r.building_nlev;
    device const float *cf=r.class_float,*bs=r.bs,*rsw=r.rsw,*rlw=r.rlw;
    device const float *rsint=r.rsint,*rlint=r.rlint,*tw=r.tw,*twlev=r.twlev;
    device const float *tflev=r.tflev,*tglev=r.tglev,*tr=r.tr,*tlev=r.tlev;
    device const float *tr_av=r.tr_av,*swdown=r.swdown,*glw=r.glw;
    device float *hrwalout=r.hrwalout,*hrwinout=r.hrwinout,*hrrof=r.hrrof;
    device float *hrwalins=r.hrwalins,*hrwinins=r.hrwinins;
    device float *hswalins=r.hswalins,*hswinins=r.hswinins,*surwal=r.surwal,*hscond=r.hscond;
    device int *error=r.error;
    if(item>=uint(p.nitem))return;
    const uint lev=item%18u,b=(item/18u)%15u,col=item/(18u*15u);
    error[item]=0;
    const ulong b4=ulong(item)*4ul,b6=ulong(item)*6ul;
    for(uint i=0;i<4;++i){hrwalout[b4+i]=0.0f;hrwinout[b4+i]=0.0f;hswinins[b4+i]=0.0f;}
    for(uint i=0;i<6;++i){hrwalins[b6+i]=0.0f;hswalins[b6+i]=0.0f;surwal[b6+i]=0.0f;}
    hscond[item]=0.0f;
    if(col>=uint(p.ncol)||active[col]==0)return;
    const int levels=nlev[ulong(col)*15ul+b];if(levels<=int(lev))return;
    const int cls=utype[col]-1;if(cls<0||cls>=p.nurbm){error[item]=3821;return;}
    const float bw=bs[ulong(col)*2ul],bl=bs[ulong(col)*2ul+1ul],dz=p.dz_u;
    const float albw=cf[77+cls],emw=cf[110+cls],albr=cf[88+cls],emr=cf[121+cls];
    const float albwin=cf[1052+cls],emwin=cf[1063+cls],pvfrac=cf[1217+cls],pwin=cf[1085+cls];
    if(!(bw>0.0f&&bl>0.0f&&dz>0.0f)||!isfinite(bw)||!isfinite(bl)||!isfinite(dz)||!(pwin>=0.0f&&pwin<=1.0f)||!isfinite(pwin)){error[item]=3822;return;}
    if(pvfrac!=0.0f){error[item]=3820;return;}
    const ulong base_tw=((((ulong(col)*15ul+b)*10ul)*18ul+lev)*4ul);
    const ulong base_outer=((((ulong(col)*15ul+b)*10ul+9ul)*18ul+lev)*4ul);
    const ulong base_twin=(((ulong(col)*15ul+b)*18ul+lev)*4ul);
    const ulong i6=ulong(item)*6ul;
    thread float wall_t[6],window_t[4],area[6];
    for(uint f=0;f<4;++f){
        wall_t[f]=tw[base_tw+f];window_t[f]=twlev[base_twin+f];
        const float sw=rsw[(ulong(col)*4ul+f)*18ul+lev];
        const float lw=rlw[(ulong(col)*4ul+f)*18ul+lev];
        const float surf=tw[base_outer+f];
        hrwalout[b4+f]=ubmf_sub(ubmf_add(ubmf_mul(ubmf_sub(1.0f,albw),sw),ubmf_mul(emw,lw)),ubmf_mul(ubmf_mul(emw,5.67e-8f),ubmf_pow4(surf)));
        hrwinout[b4+f]=ubmf_sub(ubmf_mul(emwin,lw),ubmf_mul(ubmf_mul(emwin,5.67e-8f),ubmf_pow4(window_t[f])));
        hrwalins[b6+f]=ubmf_sub(ubmf_add(ubmf_mul(0.7f,rsint[i6+f]),ubmf_mul(0.9f,rlint[i6+f])),ubmf_mul(ubmf_mul(0.9f,5.67e-8f),ubmf_pow4(wall_t[f])));
        hrwinins[b4+f]=ubmf_sub(ubmf_mul(emwin,rlint[i6+f]),ubmf_mul(ubmf_mul(emwin,5.67e-8f),ubmf_pow4(window_t[f])));
        hswalins[b6+f]=ubmf_mul(ubmf_mul(5.678f,1.09f),ubmf_sub(tlev[(ulong(col)*15ul+b)*18ul+lev],wall_t[f]));
        hswinins[b4+f]=ubmf_mul(ubmf_mul(5.678f,0.99f),ubmf_sub(tlev[(ulong(col)*15ul+b)*18ul+lev],window_t[f]));
        area[f]=ubmf_mul(dz,(f<2u)?bw:bl);surwal[b6+f]=area[f];
    }
    if(lev==0u){
        wall_t[4]=tglev[((ulong(col)*15ul+b)*10ul+9ul)*2ul+1ul];
        wall_t[5]=(levels==1)?tr[(ulong(col)*10ul)*18ul*2ul+1ul]:tflev[(((ulong(col)*15ul+b)*10ul)*17ul)*2ul+1ul];
    }else if(lev+1u==uint(levels)){
        wall_t[4]=tflev[(((ulong(col)*15ul+b)*10ul+9ul)*17ul+(lev-1u))*2ul+1ul];
        wall_t[5]=tr[(ulong(col)*10ul)*18ul*2ul+1ul];
    }else{
        wall_t[4]=tflev[(((ulong(col)*15ul+b)*10ul+9ul)*17ul+(lev-1u))*2ul+1ul];
        wall_t[5]=tflev[(((ulong(col)*15ul+b)*10ul)*17ul+lev)*2ul+1ul];
    }
    area[4]=area[5]=ubmf_mul(bw,bl);surwal[b6+4ul]=area[4];surwal[b6+5ul]=area[5];
    for(uint f=4;f<6;++f)
        hswalins[b6+f]=ubmf_mul(ubmf_mul(5.678f,1.09f),ubmf_sub(tlev[(ulong(col)*15ul+b)*18ul+lev],wall_t[f]));
    float cond=0.0f;
    for(uint f=0;f<4;++f){
        cond=ubmf_add(ubmf_add(cond,ubmf_mul(ubmf_mul(area[f],ubmf_sub(1.0f,pwin)),hswalins[b6+f])),ubmf_mul(ubmf_mul(area[f],pwin),hswinins[b4+f]));
    }
    for(uint f=4;f<6;++f)cond=ubmf_add(cond,ubmf_mul(area[f],hswalins[b6+f]));
    hscond[item]=ubmf_mul(-1.0f,cond);
    const float roof_t=tr_av[(ulong(col)*18ul+17ul)*2ul];
    if(lev==0u)hrrof[ulong(col)*15ul+b]=ubmf_sub(ubmf_add(ubmf_mul(ubmf_sub(1.0f,albr),swdown[col]),ubmf_mul(emr,glw[col])),ubmf_mul(ubmf_mul(emr,5.67e-8f),ubmf_pow4(roof_t)));
    error[item]=0;
}
