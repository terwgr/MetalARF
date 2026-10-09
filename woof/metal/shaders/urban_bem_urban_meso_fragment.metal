#include <metal_stdlib>
#include "urban_bem_urban_meso_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubx_add(float a,float b){return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubx_sub(float a,float b){return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubx_mul(float a,float b){return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubx_div(float a,float b){return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubx_min(float a,float b){return a<b?a:b;}
static inline float ubx_max(float a,float b){return a>b?a:b;}

// WRF bb_urban_meso, module_sf_bep_bem.F:2978-3114. One thread owns a full
// column and walks directions, urban levels, and mass levels in WRF order.
// The domain z profile uses edge levels [ncol,nz+1]; dz is the original
// WRF dz8w view [nz,ncol] (level-major, column-minor).
kernel void urban_bem_urban_meso_fragment(
    constant UrbanBemUrbanMesoFragmentResources &r [[buffer(0)]],
    constant UrbanBemUrbanMesoFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if(col>=uint(p.ncol))return;
    device const int *active=r.active_cols;
    device const float *z=r.z,*dz=r.dz,*zu=r.z_u,*pb=r.pb,*ss=r.ss;
    device const float *bs=r.bs,*ws=r.ws;
    device const float *uva=r.uva,*vva=r.vva,*uvb=r.uvb,*vvb=r.vvb;
    device const float *tvb=r.tvb,*tvb_ac=r.tvb_ac,*evb=r.evb;
    device const float *uhb=r.uhb,*vhb=r.vhb,*thb=r.thb,*ehb=r.ehb;
    device const float *qhb=r.qhb,*qvb=r.qvb;
    device float *a_u=r.a_u,*a_v=r.a_v,*a_t=r.a_t,*a_e=r.a_e;
    device float *b_u=r.b_u,*b_v=r.b_v,*b_t=r.b_t,*b_e=r.b_e,*b_q=r.b_q,*b_ac=r.b_ac;
    device float *sf=r.sf,*vl=r.vl;device int *error=r.error;
    error[col]=0;
    const ulong edge=ulong(col)*(uint(p.nz)+1ul);
    const ulong mass=ulong(col)*uint(p.nz);
    for(uint k=0;k<uint(p.nz);++k){
        a_u[mass+k]=0.0f;a_v[mass+k]=0.0f;a_t[mass+k]=0.0f;a_e[mass+k]=0.0f;
        b_u[mass+k]=0.0f;b_v[mass+k]=0.0f;b_t[mass+k]=0.0f;
        b_e[mass+k]=0.0f;b_q[mass+k]=0.0f;b_ac[mass+k]=0.0f;
        sf[edge+k]=0.0f;vl[mass+k]=0.0f;
    }
    sf[edge+uint(p.nz)]=0.0f;
    if(active[col]==0)return;

    // Source lines 2999-3022: urban plan fraction at interfaces and
    // atmospheric volume fraction at mass levels.
    for(uint id=0;id<2u;++id){
        const float width=ws[ulong(col)*2ul+id],building=bs[ulong(col)*2ul+id];
        const float denom=ubx_add(width,building);
        if(!(width>0.0f&&building>=0.0f&&denom>0.0f)||!isfinite(denom)){
            error[col]=7601;return;
        }
        for(uint k=1u;k<=uint(p.nz);++k){
            float sr=0.0f;
            for(uint iu=1u;iu<18u;++iu){
                if(z[edge+k]<zu[iu]&&z[edge+k]>=zu[iu-1ul])
                    sr=pb[ulong(col)*19ul+iu];
            }
            const float numerator=ubx_add(width,ubx_mul(ubx_sub(1.0f,sr),building));
            const float share=ubx_div(ubx_div(numerator,denom),2.0f);
            sf[edge+k]=ubx_add(sf[edge+k],share);
        }
        for(uint k=0u;k<uint(p.nz);++k){
            const float dzv=dz[ulong(k)*uint(p.ncol)+col];
            if(!(dzv>0.0f)||!isfinite(dzv)){error[col]=7602;return;}
            float vtot=0.0f;
            for(uint iu=0u;iu<18u;++iu){
                const float upper=ubx_min(zu[iu+1u],z[edge+k+1ul]);
                const float lower=ubx_max(zu[iu],z[edge+k]);
                const float overlap=ubx_max(ubx_sub(upper,lower),0.0f);
                vtot=ubx_add(vtot,ubx_mul(pb[ulong(col)*19ul+iu+1ul],overlap));
            }
            vtot=ubx_div(vtot,dzv);
            const float share=ubx_div(ubx_sub(1.0f,
                ubx_div(ubx_mul(vtot,building),denom)),2.0f);
            vl[mass+k]=ubx_add(vl[mass+k],share);
        }
    }

    // Ground/lowest-level source terms (module_sf_bep_bem.F:3026-3033).
    const float vl0=vl[mass],dz0=dz[col];
    if(!(vl0>0.0f&&dz0>0.0f)){error[col]=7603;return;}
    for(uint id=0;id<2u;++id){
        const float width=ws[ulong(col)*2ul+id],building=bs[ulong(col)*2ul+id];
        float fact=ubx_div(1.0f,vl0);
        fact=ubx_div(fact,dz0);
        fact=ubx_mul(fact,width);
        fact=ubx_div(fact,ubx_add(width,building));
        fact=ubx_div(fact,2.0f);
        b_t[mass]=ubx_add(b_t[mass],ubx_mul(thb[ulong(col)*2ul*18ul+ulong(id)*18ul],fact));
        b_u[mass]=ubx_add(b_u[mass],ubx_mul(uhb[ulong(col)*2ul*18ul+ulong(id)*18ul],fact));
        b_v[mass]=ubx_add(b_v[mass],ubx_mul(vhb[ulong(col)*2ul*18ul+ulong(id)*18ul],fact));
        const float dzurban=ubx_sub(zu[1],zu[0]);
        b_e[mass]=ubx_add(b_e[mass],ubx_mul(ubx_mul(ehb[ulong(col)*2ul*18ul+ulong(id)*18ul],fact),dzurban));
        b_q[mass]=ubx_add(b_q[mass],ubx_mul(qhb[ulong(col)*2ul*18ul+ulong(id)*18ul],fact));
    }

    // Building-to-mass-level overlap tendencies, source lines 3035-3057.
    for(uint k=0u;k<uint(p.nz);++k){
        const float dzv=dz[ulong(k)*uint(p.ncol)+col],vlv=vl[mass+k];
        if(!(dzv>0.0f&&vlv>0.0f)){error[col]=7603;return;}
        for(uint id=0u;id<2u;++id){
            float st=0.0f,su=0.0f,sv=0.0f,se=0.0f,sq=0.0f;
            for(uint iu=1u;iu<18u;++iu){
                if(z[edge+k]<=zu[iu]&&z[edge+k+1ul]>zu[iu]){
                    const float profile=ss[ulong(col)*19ul+iu];
                    const ulong flat=(ulong(col)*2ul+id)*18ul+iu;
                    st=ubx_add(st,ubx_mul(profile,thb[flat]));
                    su=ubx_add(su,ubx_mul(profile,uhb[flat]));
                    sv=ubx_add(sv,ubx_mul(profile,vhb[flat]));
                    se=ubx_add(se,ubx_mul(ubx_mul(profile,ehb[flat]),
                        ubx_sub(zu[iu+1u],zu[iu])));
                    sq=ubx_add(sq,ubx_mul(profile,qhb[flat]));
                }
            }
            float fact=ubx_div(bs[ulong(col)*2ul+id],
                ubx_add(ws[ulong(col)*2ul+id],bs[ulong(col)*2ul+id]));
            fact=ubx_div(fact,vlv);fact=ubx_div(fact,dzv);fact=ubx_div(fact,2.0f);
            b_t[mass+k]=ubx_add(b_t[mass+k],ubx_mul(st,fact));
            b_u[mass+k]=ubx_add(b_u[mass+k],ubx_mul(su,fact));
            b_v[mass+k]=ubx_add(b_v[mass+k],ubx_mul(sv,fact));
            b_e[mass+k]=ubx_add(b_e[mass+k],ubx_mul(se,fact));
            b_q[mass+k]=ubx_add(b_q[mass+k],ubx_mul(sq,fact));
        }
    }

    // Canyon-wall and acoustic tendencies, source lines 3062-3108.
    for(uint k=0u;k<uint(p.nz);++k){
        const float dzv=dz[ulong(k)*uint(p.ncol)+col],vlv=vl[mass+k];
        if(!(dzv>0.0f&&vlv>0.0f)){error[col]=7603;return;}
        float vtb=0.0f,vtb_ac=0.0f,vta=0.0f,vua=0.0f,vub=0.0f;
        float sum_vva=0.0f,sum_vvb=0.0f,veb=0.0f,vqb=0.0f;
        for(uint id=0u;id<2u;++id){
            const float denom=ubx_add(ws[ulong(col)*2ul+id],bs[ulong(col)*2ul+id]);
            for(uint iu=0u;iu<18u;++iu){
                const float upper=ubx_min(zu[iu+1u],z[edge+k+1ul]);
                const float lower=ubx_max(zu[iu],z[edge+k]);
                const float overlap=ubx_max(ubx_sub(upper,lower),0.0f);
                const float fact=ubx_div(overlap,denom);
                const ulong face0=(ulong(col)*4ul+ulong(id)*2ul)*18ul+iu;
                const ulong face1=face0+18ul;
                const float pair_t=ubx_add(tvb[face0],tvb[face1]);
                vtb=ubx_add(vtb,ubx_mul(ubx_mul(pb[ulong(col)*19ul+iu+1ul],pair_t),fact));
                const float pair_ac=ubx_add(tvb_ac[face0],tvb_ac[face1]);
                vtb_ac=ubx_add(vtb_ac,ubx_mul(ubx_mul(pb[ulong(col)*19ul+iu+1ul],pair_ac),fact));
                const float pair_a=0.0f;
                vta=ubx_add(vta,ubx_mul(ubx_mul(pb[ulong(col)*19ul+iu+1ul],pair_a),fact));
                vua=ubx_add(vua,ubx_mul(ubx_mul(pb[ulong(col)*19ul+iu+1ul],ubx_add(uva[face0],uva[face1])),fact));
                sum_vva=ubx_add(sum_vva,ubx_mul(ubx_mul(pb[ulong(col)*19ul+iu+1ul],ubx_add(vva[face0],vva[face1])),fact));
                vub=ubx_add(vub,ubx_mul(ubx_mul(pb[ulong(col)*19ul+iu+1ul],ubx_add(uvb[face0],uvb[face1])),fact));
                sum_vvb=ubx_add(sum_vvb,ubx_mul(ubx_mul(pb[ulong(col)*19ul+iu+1ul],ubx_add(vvb[face0],vvb[face1])),fact));
                veb=ubx_add(veb,ubx_mul(ubx_mul(pb[ulong(col)*19ul+iu+1ul],ubx_add(evb[face0],evb[face1])),fact));
                vqb=ubx_add(vqb,ubx_mul(ubx_mul(pb[ulong(col)*19ul+iu+1ul],ubx_add(qvb[face0],qvb[face1])),fact));
            }
        }
        float fact=ubx_div(1.0f,vlv);fact=ubx_div(fact,dzv);fact=ubx_div(fact,2.0f);
        b_t[mass+k]=ubx_add(b_t[mass+k],ubx_mul(vtb,fact));
        b_ac[mass+k]=ubx_add(b_ac[mass+k],ubx_mul(vtb_ac,fact));
        a_t[mass+k]=ubx_add(a_t[mass+k],ubx_mul(vta,fact));
        a_u[mass+k]=ubx_add(a_u[mass+k],ubx_mul(vua,fact));
        a_v[mass+k]=ubx_add(a_v[mass+k],ubx_mul(sum_vva,fact));
        b_u[mass+k]=ubx_add(b_u[mass+k],ubx_mul(vub,fact));
        b_v[mass+k]=ubx_add(b_v[mass+k],ubx_mul(sum_vvb,fact));
        b_e[mass+k]=ubx_add(b_e[mass+k],ubx_mul(veb,fact));
        b_q[mass+k]=ubx_add(b_q[mass+k],ubx_mul(vqb,fact));
    }
}
