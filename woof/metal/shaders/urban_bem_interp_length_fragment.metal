#include <metal_stdlib>
#include "urban_bem_interp_length_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubl_add(float a,float b){return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubl_sub(float a,float b){return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubl_mul(float a,float b){return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubl_div(float a,float b){return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b)));}

// WRF bb_interp_length, module_sf_bep_bem.F:3120-3199. Each thread owns a
// column; urban levels are accumulated in ascending WRF order. The vertical
// domain z profile is edge based [ncol,nz+1], while dlg/dl_u are mass levels.
kernel void urban_bem_interp_length_fragment(
    constant UrbanBemInterpLengthFragmentResources &r [[buffer(0)]],
    constant UrbanBemInterpLengthFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if(col>=uint(p.ncol))return;
    device const int *active=r.active_cols;
    device const float *z=r.z,*zu=r.z_u,*ss=r.ss,*ws=r.ws,*bs=r.bs;
    device float *dlg=r.dlg,*dlu=r.dl_u;device int *error=r.error;
    error[col]=0;
    const ulong edge=ulong(col)*(uint(p.nz)+1ul), mass=ulong(col)*uint(p.nz);
    for(uint k=0;k<uint(p.nz);++k){dlg[mass+k]=0.0f;dlu[mass+k]=0.0f;}
    if(active[col]==0)return;

    // WRF first computes dl_u from building fractions above each mass level.
    for(uint k=0;k<uint(p.nz);++k){
        float ulu=0.0f,ssl=0.0f;
        for(uint id=0;id<2u;++id){
            for(uint iu=1u;iu<18u;++iu){
                if(zu[iu]>z[edge+k]){
                    ulu=ubl_add(ulu,ubl_div(ubl_div(ss[ulong(col)*19ul+iu],zu[iu]),2.0f));
                    ssl=ubl_add(ssl,ubl_div(ss[ulong(col)*19ul+iu],2.0f));
                }
            }
        }
        dlu[mass+k]=(ulu!=0.0f)?ubl_div(ssl,ulu):0.0f;
        if(!isfinite(dlu[mass+k])){error[col]=7701;return;}
    }

    // WRF next computes dlg. Preserve its operation order: wall width seeds
    // both sums, then each qualifying building level contributes in order.
    for(uint k=0;k<uint(p.nz);++k){
        const float zmid=ubl_div(ubl_add(z[edge+k],z[edge+k+1ul]),2.0f);
        float inverse=0.0f;
        for(uint id=0;id<2u;++id){
            float sftot=ws[ulong(col)*2ul+id];
            float dlgtmp=ubl_div(ws[ulong(col)*2ul+id],zmid);
            for(uint iu=0;iu<18u;++iu){
                if(zmid>zu[iu]){
                    const float frac=ss[ulong(col)*19ul+iu];
                    const float building=bs[ulong(col)*2ul+id];
                    dlgtmp=ubl_add(dlgtmp,ubl_div(ubl_mul(frac,building),ubl_sub(zmid,zu[iu])));
                    sftot=ubl_add(sftot,ubl_mul(frac,building));
                }
            }
            if(!(sftot!=0.0f)||!isfinite(sftot)){error[col]=7702;return;}
            inverse=ubl_add(inverse,ubl_div(ubl_div(dlgtmp,sftot),2.0f));
        }
        if(!(inverse!=0.0f)||!isfinite(inverse)){error[col]=7702;return;}
        dlg[mass+k]=ubl_div(1.0f,inverse);
        if(!isfinite(dlg[mass+k])){error[col]=7702;return;}
    }
}
