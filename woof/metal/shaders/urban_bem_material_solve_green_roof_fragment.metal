#include <metal_stdlib>
#include "urban_bem_material_solve_green_roof_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubm_add(float a,float b){return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubm_sub(float a,float b){return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubm_mul(float a,float b){return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubm_div(float a,float b){return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b)));}

// WRF module_sf_bem.F:645-743. One item is a (column, building) solve. The
// profile arrays use the bb_param per-column packing; temperatures are the
// exact state-gather layouts. This fragment is the ordinary roof path. Green
// roof and PV are deliberately outside its contract and remain refused by
// the caller until their source-order terms are implemented.
static inline bool ubm_wall_solve(thread float *temp, thread const float *dz,
                                 thread const float *cap, thread const float *k,
                                 float dt, float flux0, float flux1,
                                 uint n) {
    if (n == 0u || n > 10u) return false;
    thread float k1[10], k2[10], kc[10], kf[10];
    thread float lo[10], diag[10], hi[10], rhs[10];
    for (uint i=0u;i<n;++i) {
        if (!(dz[i]>0.0f && cap[i]>0.0f) || !isfinite(dz[i]) ||
            !isfinite(cap[i]) || !isfinite(k[i]) || !(k[i]>0.0f)) return false;
        kc[i]=ubm_div(dt,ubm_mul(dz[i],cap[i]));
    }
    kf[0]=ubm_div(ubm_mul(2.0f,k[0]),dz[0]);
    for(uint i=1u;i<n;++i)
        kf[i]=ubm_div(ubm_mul(2.0f,k[i]),ubm_add(dz[i],dz[i-1u]));
    for(uint i=0u;i<n;++i) k1[i]=ubm_mul(kc[i],kf[i]);
    for(uint i=0u;i+1u<n;++i)
        k2[i]=ubm_div(ubm_mul(ubm_mul(kc[i],kf[i+1u]),cap[i]),cap[i+1u]);
    lo[0]=0.0f;diag[0]=ubm_add(1.0f,k2[0]);hi[0]=-k2[0];
    rhs[0]=ubm_add(temp[0],ubm_mul(flux0,kc[0]));
    for(uint i=1u;i+1u<n;++i){
        lo[i]=-k1[i];diag[i]=ubm_add(ubm_add(1.0f,k1[i]),k2[i]);
        hi[i]=-k2[i];rhs[i]=temp[i];
    }
    const uint last=n-1u;
    lo[last]=-k1[last];diag[last]=ubm_add(1.0f,k1[last]);hi[last]=0.0f;
    rhs[last]=ubm_add(temp[last],ubm_mul(flux1,kc[last]));
    // bem_tridia's exact backward elimination, forward elimination, then
    // division order (module_sf_bem.F:2396-2407).
    for(int i=int(n)-2;i>=0;--i){
        rhs[i]=ubm_sub(rhs[i],ubm_div(ubm_mul(hi[i],rhs[i+1]),diag[i+1]));
        diag[i]=ubm_sub(diag[i],ubm_div(ubm_mul(hi[i],lo[i+1]),diag[i+1]));
    }
    for(uint i=1u;i<n;++i)
        rhs[i]=ubm_sub(rhs[i],ubm_div(ubm_mul(lo[i],rhs[i-1u]),diag[i-1u]));
    for(uint i=0u;i<n;++i){
        if (!(diag[i] != 0.0f) || !isfinite(diag[i])) return false;
        temp[i]=ubm_div(rhs[i],diag[i]);
        if (!isfinite(temp[i])) return false;
    }
    return true;
}

kernel void urban_bem_material_solve_fragment(
    constant UrbanBemMaterialSolveFragmentResources &r [[buffer(0)]],
    constant UrbanBemMaterialSolveFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    if(item>=uint(p.nitem))return;
    device const int *active=r.active_cols,*nlev=r.building_nlev;
    device const float *dzw=r.dzw,*csw=r.csw,*kw=r.alaw;
    device const float *dzf=r.dzf,*csf=r.csf,*kf=r.alaf;
    device const float *dzg=r.dzgb,*csg=r.csgb,*kg=r.alagb;
    device const float *dzr=r.dzr,*csr=r.csr,*kr=r.alar;
    device const float *hswi=r.hswalins,*hrwi=r.hrwalins;
    device const float *hswo=r.hswalout,*hrwo=r.hrwalout;
    device const float *hsini=r.hswinins,*hrini=r.hrwinins;
    device const float *hsino=r.hswinout,*hrino=r.hrwinout;
    device const float *hsrof=r.hsrof,*hrrof=r.hrrof,*lsrof=r.lsrof;
    device float *tw=r.tw,*twin=r.twlev,*tflev=r.tflev,*tglev=r.tglev,*tr=r.tr;
    device float *gswal=r.gswal,*gsrof=r.gsrof;
    device int *error=r.error;
    const uint b=item%15u,col=item/15u;error[item]=0;
    if(col>=uint(p.ncol)||active[col]==0)return;
    const int nlev_i=nlev[ulong(col)*15ul+b];
    if(nlev_i<=0)return;
    if(nlev_i>18){error[item]=3850;return;}
    const uint nlev_u=uint(nlev_i);
    thread float dz[10],cap[10],k[11],temp[10];
    const ulong cbase=ulong(col)*10ul;
    const ulong itembase=(ulong(col)*15ul+b);
    // Wall stacks, followed by each window, preserving WRF's loop order.
    for(uint lev=0u;lev<nlev_u;++lev){
        for(uint face=0u;face<4u;++face){
            for(uint z=0u;z<10u;++z){
                dz[z]=dzw[cbase+z];cap[z]=csw[cbase+z];k[z]=kw[cbase+z];
                temp[z]=tw[((((itembase*10ul+z)*18ul+lev)*4ul)+face)];
            }
            k[10]=kw[cbase+10ul];
            const ulong fbase=(itembase*18ul+lev)*6ul+face;
            const float f0=ubm_add(hswi[fbase],hrwi[fbase]);
            const float f1=ubm_add(hswo[(itembase*18ul+lev)*4ul+face],hrwo[(itembase*18ul+lev)*4ul+face]);
            gswal[(itembase*18ul+lev)*4ul+face]=f1;
            if(!ubm_wall_solve(temp,dz,cap,k,p.dt,f0,f1,10u)){error[item]=3851;return;}
            for(uint z=0u;z<10u;++z)
                tw[((((itembase*10ul+z)*18ul+lev)*4ul)+face)]=temp[z];
        }
        for(uint face=0u;face<4u;++face){
            const ulong flux=(itembase*18ul+lev)*4ul+face;
            const float total=ubm_add(ubm_add(hsini[flux],hrini[flux]),ubm_add(hsino[flux],hrino[flux]));
            const ulong w=(itembase*18ul+lev)*4ul+face;
            twin[w]=ubm_add(twin[w],ubm_mul(ubm_div(p.dt,2.268e6f*0.006f),total));
        }
    }
    if(nlev_u>1u){
        for(uint lev=0u;lev+1u<nlev_u;++lev){
            const ulong base=(itembase*18ul+lev)*6ul;
            const float f0=ubm_add(hswi[base+5ul],hrwi[base+5ul]);
            const float f1=ubm_add(hswi[base+10ul],hrwi[base+10ul]);
            for(uint id=0u;id<2u;++id){
                for(uint z=0u;z<10u;++z){dz[z]=dzf[cbase+z];cap[z]=csf[cbase+z];k[z]=kf[cbase+z];
                    temp[z]=tflev[((((itembase*10ul+z)*17ul+lev)*2ul)+id)];}
                k[10]=kf[cbase+10ul];
                if(!ubm_wall_solve(temp,dz,cap,k,p.dt,f0,f1,10u)){error[item]=3852;return;}
                for(uint z=0u;z<10u;++z)tflev[((((itembase*10ul+z)*17ul+lev)*2ul)+id)]=temp[z];
            }
        }
    }
    const ulong first=(itembase*18ul)*6ul;
    for(uint id=0u;id<2u;++id){
        for(uint z=0u;z<10u;++z){dz[z]=dzg[cbase+z];cap[z]=csg[cbase+z];k[z]=kg[cbase+z];temp[z]=tglev[(itembase*10ul+z)*2ul+id];}
        k[10]=kg[cbase+10ul];
        if(!ubm_wall_solve(temp,dz,cap,k,p.dt,0.0f,ubm_add(hswi[first+4ul],hrwi[first+4ul]),10u)){error[item]=3853;return;}
        for(uint z=0u;z<10u;++z)tglev[(itembase*10ul+z)*2ul+id]=temp[z];
    }
    const ulong top=(itembase*18ul+(nlev_u-1u))*6ul;
    const float roof_flux=ubm_add(ubm_add(hsrof[item],hrrof[item]),lsrof[item]);
    gsrof[item]=roof_flux;
    for(uint id=0u;id<2u;++id){
        for(uint z=0u;z<10u;++z){dz[z]=dzr[cbase+z];cap[z]=csr[cbase+z];k[z]=kr[cbase+z];temp[z]=tr[(cbase+z)*18ul*2ul+(nlev_u-1u)*2ul+id];}
        k[10]=kr[cbase+10ul];
        if(!ubm_wall_solve(temp,dz,cap,k,p.dt,ubm_add(hswi[top+5ul],hrwi[top+5ul]),roof_flux,10u)){error[item]=3854;return;}
        for(uint z=0u;z<10u;++z)tr[(cbase+z)*18ul*2ul+(nlev_u-1u)*2ul+id]=temp[z];
    }
}

static inline bool ubm_wall_gr_solve(thread float *temp, thread const float *dz,
                                     thread const float *cap, thread const float *k,
                                     float dt, float flux0, float flux1,
                                     float hfgr, float green_fraction, uint n) {
    if (n == 0u || n > 10u) return false;
    thread float k1[10], k2[10], kc[10], kf[10];
    thread float lo[10], diag[10], hi[10], rhs[10];
    for (uint i=0u;i<n;++i) {
        if (!(dz[i]>0.0f && cap[i]>0.0f) || !isfinite(dz[i]) ||
            !isfinite(cap[i]) || !isfinite(k[i]) || !(k[i]>0.0f)) return false;
        kc[i]=ubm_div(dt,ubm_mul(dz[i],cap[i]));
    }
    kf[0]=ubm_div(ubm_mul(2.0f,k[0]),dz[0]);
    for(uint i=1u;i<n;++i) kf[i]=ubm_div(ubm_mul(2.0f,k[i]),ubm_add(dz[i],dz[i-1u]));
    for(uint i=0u;i<n;++i) k1[i]=ubm_mul(kc[i],kf[i]);
    for(uint i=0u;i+1u<n;++i)
        k2[i]=ubm_div(ubm_mul(ubm_mul(kc[i],kf[i+1u]),cap[i]),cap[i+1u]);
    lo[0]=0.0f; diag[0]=ubm_add(1.0f,k2[0]); hi[0]=-k2[0];
    rhs[0]=ubm_add(temp[0],ubm_mul(flux0,kc[0]));
    for(uint i=1u;i+1u<n;++i){
        lo[i]=-k1[i];
        if(i==4u){
            const float nongreen=ubm_sub(1.0f,green_fraction);
            diag[i]=ubm_add(ubm_add(1.0f,k1[i]),ubm_mul(nongreen,k2[i]));
            hi[i]=-ubm_mul(nongreen,k2[i]);
            rhs[i]=ubm_add(temp[i],ubm_div(ubm_mul(ubm_mul(green_fraction,hfgr),dt),dz[i]));
        } else {
            diag[i]=ubm_add(ubm_add(1.0f,k1[i]),k2[i]);
            hi[i]=-k2[i]; rhs[i]=temp[i];
        }
    }
    const uint last=n-1u;
    lo[last]=-k1[last];diag[last]=ubm_add(1.0f,k1[last]);hi[last]=0.0f;
    rhs[last]=ubm_add(temp[last],ubm_mul(flux1,kc[last]));
    for(int i=int(n)-2;i>=0;--i){
        rhs[i]=ubm_sub(rhs[i],ubm_div(ubm_mul(hi[i],rhs[i+1]),diag[i+1]));
        diag[i]=ubm_sub(diag[i],ubm_div(ubm_mul(hi[i],lo[i+1]),diag[i+1]));
    }
    for(uint i=1u;i<n;++i) rhs[i]=ubm_sub(rhs[i],ubm_div(ubm_mul(lo[i],rhs[i-1u]),diag[i-1u]));
    for(uint i=0u;i<n;++i){
        if (!(diag[i] != 0.0f) || !isfinite(diag[i])) return false;
        temp[i]=ubm_div(rhs[i],diag[i]);
        if (!isfinite(temp[i])) return false;
    }
    return true;
}

kernel void urban_bem_material_solve_green_roof_fragment(
    constant UrbanBemMaterialSolveGreenRoofFragmentResources &r [[buffer(0)]],
    constant UrbanBemMaterialSolveGreenRoofFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    if(item>=uint(p.nitem))return;
    device const int *active=r.active_cols,*nlev=r.building_nlev;
    device const int *utype=r.utype;
    device const float *class_float=r.class_float,*hfgr=r.hfgroof;
    device const float *dzw=r.dzw,*csw=r.csw,*kw=r.alaw;
    device const float *dzf=r.dzf,*csf=r.csf,*kf=r.alaf;
    device const float *dzg=r.dzgb,*csg=r.csgb,*kg=r.alagb;
    device const float *dzr=r.dzr,*csr=r.csr,*kr=r.alar;
    device const float *hswi=r.hswalins,*hrwi=r.hrwalins;
    device const float *hswo=r.hswalout,*hrwo=r.hrwalout;
    device const float *hsini=r.hswinins,*hrini=r.hrwinins;
    device const float *hsino=r.hswinout,*hrino=r.hrwinout;
    device const float *hsrof=r.hsrof,*hrrof=r.hrrof,*lsrof=r.lsrof;
    device float *tw=r.tw,*twin=r.twlev,*tflev=r.tflev,*tglev=r.tglev,*tr=r.tr;
    device float *gswal=r.gswal,*gsrof=r.gsrof;
    device int *error=r.error;
    const uint b=item%15u,col=item/15u;error[item]=0;
    if(col>=uint(p.ncol)||active[col]==0)return;
    const int nlev_i=nlev[ulong(col)*15ul+b];
    if(nlev_i<=0)return;
    if(nlev_i>18){error[item]=3850;return;}
    const uint nlev_u=uint(nlev_i);
    const int class_id=utype[col]-1;
    if(class_id<0||class_id>=11){error[item]=3855;return;}
    const float green_fraction=class_float[1206ul+ulong(class_id)];
    thread float dz[10],cap[10],k[11],temp[10];
    const ulong cbase=ulong(col)*10ul;
    const ulong itembase=(ulong(col)*15ul+b);
    // Wall stacks, followed by each window, preserving WRF's loop order.
    for(uint lev=0u;lev<nlev_u;++lev){
        for(uint face=0u;face<4u;++face){
            for(uint z=0u;z<10u;++z){
                dz[z]=dzw[cbase+z];cap[z]=csw[cbase+z];k[z]=kw[cbase+z];
                temp[z]=tw[((((itembase*10ul+z)*18ul+lev)*4ul)+face)];
            }
            k[10]=kw[cbase+10ul];
            const ulong fbase=(itembase*18ul+lev)*6ul+face;
            const float f0=ubm_add(hswi[fbase],hrwi[fbase]);
            const float f1=ubm_add(hswo[(itembase*18ul+lev)*4ul+face],hrwo[(itembase*18ul+lev)*4ul+face]);
            gswal[(itembase*18ul+lev)*4ul+face]=f1;
            if(!ubm_wall_solve(temp,dz,cap,k,p.dt,f0,f1,10u)){error[item]=3851;return;}
            for(uint z=0u;z<10u;++z)
                tw[((((itembase*10ul+z)*18ul+lev)*4ul)+face)]=temp[z];
        }
        for(uint face=0u;face<4u;++face){
            const ulong flux=(itembase*18ul+lev)*4ul+face;
            const float total=ubm_add(ubm_add(hsini[flux],hrini[flux]),ubm_add(hsino[flux],hrino[flux]));
            const ulong w=(itembase*18ul+lev)*4ul+face;
            twin[w]=ubm_add(twin[w],ubm_mul(ubm_div(p.dt,2.268e6f*0.006f),total));
        }
    }
    if(nlev_u>1u){
        for(uint lev=0u;lev+1u<nlev_u;++lev){
            const ulong base=(itembase*18ul+lev)*6ul;
            const float f0=ubm_add(hswi[base+5ul],hrwi[base+5ul]);
            const float f1=ubm_add(hswi[base+10ul],hrwi[base+10ul]);
            for(uint id=0u;id<2u;++id){
                for(uint z=0u;z<10u;++z){dz[z]=dzf[cbase+z];cap[z]=csf[cbase+z];k[z]=kf[cbase+z];
                    temp[z]=tflev[((((itembase*10ul+z)*17ul+lev)*2ul)+id)];}
                k[10]=kf[cbase+10ul];
                if(!ubm_wall_solve(temp,dz,cap,k,p.dt,f0,f1,10u)){error[item]=3852;return;}
                for(uint z=0u;z<10u;++z)tflev[((((itembase*10ul+z)*17ul+lev)*2ul)+id)]=temp[z];
            }
        }
    }
    const ulong first=(itembase*18ul)*6ul;
    for(uint id=0u;id<2u;++id){
        for(uint z=0u;z<10u;++z){dz[z]=dzg[cbase+z];cap[z]=csg[cbase+z];k[z]=kg[cbase+z];temp[z]=tglev[(itembase*10ul+z)*2ul+id];}
        k[10]=kg[cbase+10ul];
        if(!ubm_wall_solve(temp,dz,cap,k,p.dt,0.0f,ubm_add(hswi[first+4ul],hrwi[first+4ul]),10u)){error[item]=3853;return;}
        for(uint z=0u;z<10u;++z)tglev[(itembase*10ul+z)*2ul+id]=temp[z];
    }
    const ulong top=(itembase*18ul+(nlev_u-1u))*6ul;
    const float roof_flux=ubm_add(ubm_add(hsrof[item],hrrof[item]),lsrof[item]);
    gsrof[item]=roof_flux;
    for(uint id=0u;id<2u;++id){
        for(uint z=0u;z<10u;++z){dz[z]=dzr[cbase+z];cap[z]=csr[cbase+z];k[z]=kr[cbase+z];temp[z]=tr[(cbase+z)*18ul*2ul+(nlev_u-1u)*2ul+id];}
        k[10]=kr[cbase+10ul];
        const float roof_hfgr=hfgr[(ulong(col)*18ul+(nlev_u-1u))*2ul+id];
        const bool roof_ok=p.green_roof==1
            ? ubm_wall_gr_solve(temp,dz,cap,k,p.dt,
                ubm_add(hswi[top+5ul],hrwi[top+5ul]),roof_flux,
                roof_hfgr,green_fraction,10u)
            : ubm_wall_solve(temp,dz,cap,k,p.dt,
                ubm_add(hswi[top+5ul],hrwi[top+5ul]),roof_flux,10u);
        if(!roof_ok){error[item]=3854;return;}
        for(uint z=0u;z<10u;++z)tr[(cbase+z)*18ul*2ul+(nlev_u-1u)*2ul+id]=temp[z];
    }
}
