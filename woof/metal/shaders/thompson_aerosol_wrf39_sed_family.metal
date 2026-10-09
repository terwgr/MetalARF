#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#include "thompson_aerosol_wrf39_sed_family_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"

static inline ulong wrf39_sed64(float x) {
    return soft64_from_f32_bits(as_type<uint>(x));
}
static inline float wrf39_sed32(ulong x) {
    return as_type<float>(soft64_to_f32_bits(x));
}
static inline ulong wrf39_sed_pow64(ulong x, ulong y) {
    return soft64_plm_pow(x, y);
}
static inline ulong wrf39_sed_pow64f(ulong x, float y) {
    return soft64_plm_pow(x, wrf39_sed64(y));
}
static inline float wrf39_rain_speed(float rain_qr, float rain_nr,
                                     float carried_density, float rho,
                                     float velocity_above) {
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const float ogg3 = 1.0f / 6.0f;
    const float am_r = 3.1415926536f * 1000.0f / 6.0f;
    const float rain_density = fabs(carried_density);
    const bool l_qr = carried_density != 0.0f;
    const bool rewritten = carried_density < 0.0f;
    const float rr = !l_qr ? 1.0e-12f
        : rewritten ? fmax(1.0e-12f, rain_qr * rain_density)
                    : rain_qr * rain_density;
    if (rr <= 1.0e-12f) return velocity_above;
    const float nr = fmax(1.0e-6f, rain_nr * rain_density);
    const float lam_arg = am_r * 6.0f * nr / rr;
    const float lam_f = precise::pow(lam_arg, 0.33333334326744080f);
    const ulong lam = wrf39_sed64(lam_f);
    const float rhof = precise::sqrt(rho_not / rho);
    const float prefix = rhof * 4854.0f * 24.0f * ogg3;
    const ulong first = soft64_mul(wrf39_sed64(prefix),
        wrf39_sed_pow64f(lam, 4.0f));
    const ulong shifted = soft64_add(lam, wrf39_sed64(195.0f));
    return wrf39_sed32(soft64_mul(first,
        wrf39_sed_pow64f(shifted, -5.0f)));
}

template<typename R, typename P>
static inline void wrf39_ice_column(R r, P p, uint column) {
    const uint ncol = uint(p.ny * p.nx);
    if (column >= ncol) return;
    const int nz=p.nz, ny=p.ny, nx=p.nx, j=int(column)/nx, i=int(column)-j*nx;
    float density[256], mass[256], number[256], vm[256], vn[256], fm[256], fn[256];
    float qt[256], nt[256], qi0[256], ni0[256];
    const float am_i=3.1415926536f*890.0f/6.0f, oig2=1.0f/6.0f;
    const float rho_not=101325.0f/(287.05f*298.0f);
    const float ni_max=499.0e3f;
    int sediment_top=0, nstep=0;
    float above_m=0.0f, above_n=0.0f;
    for(int k=nz-1;k>=0;--k){
        const uint x=uint(k*ny*nx+j*nx+i);
        const float qvk=fmax(1.0e-10f,r.qv[x]);
        const float rho=0.622f*r.pressure[x]/(287.04f*r.temperature[x]*(qvk+0.622f));
        density[k]=rho; qi0[k]=r.qi[x]; ni0[k]=r.ni[x]; qt[k]=0.0f; nt[k]=0.0f;
        const float srho=r.reference_density[x];
        if(r.qi[x]>1.0e-12f && r.qi[x]*srho>1.0e-12f){
            const float ri=r.qi[x]*srho; float nn=fmax(1.0e-6f,r.ni[x]*srho);
            if(nn<=1.0e-6f){
                const ulong lam=wrf39_sed64(4.0f/5.0e-6f);
                const float pref=(oig2*ri)/am_i;
                nn=fmin(ni_max,wrf39_sed32(soft64_mul(wrf39_sed64(pref),
                    wrf39_sed_pow64(lam,0x4008000000000000ul))));
            }
            const float la=am_i*6.0f*nn/ri;
            ulong lam=wrf39_sed64(precise::pow(la,0.33333334326744080f));
            float d=wrf39_sed32(soft64_div(0x4010000000000000ul,lam));
            if(d<5.0e-6f){ d=5.0e-6f; lam=soft64_div(0x4010000000000000ul,wrf39_sed64(d));
                const float pref=(oig2*ri)/am_i; nn=fmin(ni_max,wrf39_sed32(soft64_mul(wrf39_sed64(pref),wrf39_sed_pow64(lam,0x4008000000000000ul))));
            } else if(d>300.0e-6f){ d=300.0e-6f; lam=soft64_div(0x4010000000000000ul,wrf39_sed64(d));
                const float pref=(oig2*ri)/am_i; nn=wrf39_sed32(soft64_mul(wrf39_sed64(pref),wrf39_sed_pow64(lam,0x4008000000000000ul)));
            }
            mass[k]=ri; number[k]=nn; const float rhof=precise::sqrt(rho_not/rho);
            const float mp=rhof*1847.5f*24.0f*oig2;
            vm[k]=wrf39_sed32(soft64_mul(wrf39_sed64(mp),soft64_div(0x3ff0000000000000ul,lam)));
            const float np=(rhof*1847.5f*3.3233511f)/1.3293403f;
            vn[k]=wrf39_sed32(soft64_mul(wrf39_sed64(np),soft64_div(0x3ff0000000000000ul,lam)));
        }else{
            const bool lqi=r.qi[x]>1.0e-12f; mass[k]=lqi?r.qi[x]*srho:1.0e-12f;
            number[k]=lqi?fmax(1.0e-6f,r.ni[x]*srho):1.0e-6f; vm[k]=above_m; vn[k]=above_n;
        }
        above_m=vm[k]; above_n=vn[k];
        if(vm[k]>1.0e-3f){sediment_top=max(sediment_top,k);nstep=max(nstep,int(float(p.dt)/(r.dz[x]/vm[k])+1.0f));}
    }
    if(sediment_top==nz-1)sediment_top=nz-2; nstep=max(nstep,1);
    const float onstep=1.0f/float(nstep), dts=float(p.dt)*onstep; float exported=0.0f;
    for(int step=0;step<nstep;++step){
        for(int k=nz-1;k>=0;--k){fm[k]=vm[k]*mass[k];fn[k]=vn[k]*number[k];}
        int k=nz-1; uint x=uint(k*ny*nx+j*nx+i); float iz=1.0f/r.dz[x], ir=1.0f/density[k];
        qt[k]-=fm[k]*iz*onstep*ir; nt[k]-=fn[k]*iz*onstep*ir;
        mass[k]=fmax(1.0e-12f,mass[k]-fm[k]*iz*dts); number[k]=fmax(1.0e-6f,number[k]-fn[k]*iz*dts);
        for(k=sediment_top;k>=0;--k){x=uint(k*ny*nx+j*nx+i);iz=1.0f/r.dz[x];ir=1.0f/density[k];
            const float dm=fm[k+1]-fm[k], dn=fn[k+1]-fn[k]; qt[k]+=dm*iz*onstep*ir;nt[k]+=dn*iz*onstep*ir;
            mass[k]=fmax(1.0e-12f,mass[k]+dm*iz*dts);number[k]=fmax(1.0e-6f,number[k]+dn*iz*dts);}
        if(mass[0]>1.0e-11f)exported+=fm[0]*dts;
    }
    for(int k=0;k<nz;++k){const uint x=uint(k*ny*nx+j*nx+i);const float qnew=qi0[k]+qt[k]*float(p.dt);float nnew=fmax(1.0e-6f/density[k],ni0[k]+nt[k]*float(p.dt));
        if(qnew<=1.0e-12f){r.qi[x]=qnew;r.ni[x]=ni0[k]+nt[k]*float(p.dt);continue;}
        const float la=am_i*6.0f*nnew/qnew; ulong lam=wrf39_sed64(precise::pow(la,0.33333334326744080f)); float d=wrf39_sed32(soft64_div(0x4010000000000000ul,lam));
        if(d<5.0e-6f)d=5.0e-6f;else if(d>300.0e-6f)d=300.0e-6f;lam=soft64_div(0x4010000000000000ul,wrf39_sed64(d));
        const float pref=(oig2*qnew)/am_i; nnew=fmin(wrf39_sed32(soft64_mul(wrf39_sed64(pref),wrf39_sed_pow64(lam,0x4008000000000000ul))),ni_max/density[k]);r.qi[x]=qnew;r.ni[x]=nnew;}
    r.rainncv[column]=exported;r.snowncv[column]=exported;r.rainnc[column]+=exported;r.snownc[column]+=exported;
}

kernel void thompson_aa_wrf39_ice_sediment_64(
    constant ThompsonAaWrf39IceSediment64Resources& r [[buffer(0)]],
    constant ThompsonAaWrf39IceSediment64Params& p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) { wrf39_ice_column(r,p,column); }
kernel void thompson_aa_wrf39_ice_sediment_256(
    constant ThompsonAaWrf39IceSediment256Resources& r [[buffer(0)]],
    constant ThompsonAaWrf39IceSediment256Params& p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) { wrf39_ice_column(r,p,column); }

static inline float wrf39_graupel_speed(float n0, float rg, float rho,
                                        float temperature, float rain_speed) {
    const float am_g = 3.1415926536f * 500.0f / 6.0f;
    ulong base = soft64_mul(wrf39_sed64(n0), wrf39_sed64(am_g));
    base = soft64_mul(base, 0x4018000000000000ul);
    base = soft64_div(base, wrf39_sed64(rg));
    const ulong lam = soft64_plm_pow(base, wrf39_sed64(0.25f));
    const ulong ilam = soft64_div(0x3ff0000000000000ul, lam);
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const float prefix = precise::sqrt(rho_not / rho) * 442.0f
        * 20.3632278f * (1.0f / 6.0f);
    float v = wrf39_sed32(soft64_mul(wrf39_sed64(prefix),
        soft64_plm_pow(ilam, wrf39_sed64(0.89f))));
    if (temperature > 273.15f) v = fmax(v, rain_speed);
    return v;
}

template<typename R, typename P>
static inline void wrf39_graupel_column(R r, P p, uint column) {
    const uint ncol=uint(p.ny*p.nx); if(column>=ncol || r.active_columns[column]==0.0f)return;
    const int nz=p.nz,ny=p.ny,nx=p.nx,j=int(column)/nx,i=int(column)-j*nx;
    float density[256],mass[256],velocity[256],flux[256],tendency[256],initial[256];
    const float rho_not=101325.0f/(287.05f*298.0f); const float ogg3=1.0f/6.0f;
    int sediment_top=0,nstep=0;float above=0.0f,rain_above=0.0f;
    for(int k=nz-1;k>=0;--k){
        const uint x=uint(k*ny*nx+j*nx+i);const float qvk=fmax(1.0e-10f,r.qv[x]);
        const float rho=0.622f*r.pressure[x]/(287.04f*r.temperature[x]*(qvk+0.622f));density[k]=rho;
        initial[k]=r.qg[x];tendency[k]=0.0f;
        const float rainv=wrf39_rain_speed(r.melt_rain_qr[x],r.melt_rain_nr[x],r.melt_rain_density[x],rho,rain_above);rain_above=rainv;
        if(r.qg[x]>1.0e-12f){const float srho=r.reference_density[x];const float rg=r.qg[x]*srho;
            velocity[k]=wrf39_graupel_speed(r.graupel_intercept[x],rg,rho,r.temperature[x],rainv);mass[k]=rg;
        }else{mass[k]=1.0e-12f;velocity[k]=above;}
        above=velocity[k];if(velocity[k]>1.0e-3f){sediment_top=max(sediment_top,k);nstep=max(nstep,int(float(p.dt)/(r.dz[x]/velocity[k])+1.0f));}
    }
    if(sediment_top==nz-1)sediment_top=nz-2;nstep=max(nstep,1);const float onstep=1.0f/float(nstep),dts=float(p.dt)*onstep;float exported=0.0f;
    for(int step=0;step<nstep;++step){for(int k=nz-1;k>=0;--k)flux[k]=velocity[k]*mass[k];
        int k=nz-1;uint x=uint(k*ny*nx+j*nx+i);float iz=1.0f/r.dz[x],ir=1.0f/density[k];tendency[k]-=flux[k]*iz*onstep*ir;mass[k]=fmax(1.0e-12f,mass[k]-flux[k]*iz*dts);
        for(k=sediment_top;k>=0;--k){x=uint(k*ny*nx+j*nx+i);iz=1.0f/r.dz[x];ir=1.0f/density[k];const float div=flux[k+1]-flux[k];tendency[k]+=div*iz*onstep*ir;mass[k]=fmax(1.0e-12f,mass[k]+div*iz*dts);}
        if(mass[0]>1.0e-11f)exported+=flux[0]*dts;
    }
    for(int k=0;k<nz;++k){const uint x=uint(k*ny*nx+j*nx+i);const float qnew=initial[k]+tendency[k]*float(p.dt);r.qg[x]=qnew<=1.0e-12f?0.0f:qnew;}
    r.rainncv[column]+=exported;r.graupelncv[column]+=exported;r.rainnc[column]+=exported;r.graupelnc[column]+=exported;
}
kernel void thompson_aa_wrf39_graupel_sediment_64(constant ThompsonAaWrf39GraupelSediment64Resources& r [[buffer(0)]], constant ThompsonAaWrf39GraupelSediment64Params& p [[buffer(1)]], uint c [[thread_position_in_grid]]) {wrf39_graupel_column(r,p,c);}
kernel void thompson_aa_wrf39_graupel_sediment_256(constant ThompsonAaWrf39GraupelSediment256Resources& r [[buffer(0)]], constant ThompsonAaWrf39GraupelSediment256Params& p [[buffer(1)]], uint c [[thread_position_in_grid]]) {wrf39_graupel_column(r,p,c);}

static inline float wrf39_snow_speed(float rs,float tc,float rho,float t_k,
        float rain_rr,float rain_v,float boost,bool melting,int singular) {
    const float rho_not=101325.0f/(287.05f*298.0f);const float smob=rs*(1.0f/0.069f);
    const float tc0=fmin(-0.1f,tc-273.15f),m=3.0f,tc02=tc0*tc0,m2=m*m;
    const float loga=5.065339f-0.062659f*tc0-3.032362f*m+0.029469f*tc0*m-0.000285f*tc02+0.31255f*m2+0.000204f*tc02*m+0.003199f*tc0*m2-0.015952f*m2*m;
    const float exponent=0.476221f-0.015896f*tc0+0.165977f*m+0.007468f*tc0*m-0.000141f*tc02+0.060366f*m2+0.000079f*tc02*m+0.000594f*tc0*m2-0.003577f*m2*m;
    const float smoc=precise::pow(10.0f,loga)*precise::pow(smob,exponent),mr=smob/smoc;
    float ils1=1.0f/(mr*20.78f+100.0f),ils2=1.0f/(mr*3.29f+100.0f);const float rp=precise::pow(mr,0.6357f);
    const float n1=490.6f*3.51325202f*precise::pow(ils1,3.55f),n2=17.46f*rp*7.61279917f*precise::pow(ils2,4.1857f);
    ils1=1.0f/(mr*20.78f);ils2=1.0f/(mr*3.29f);
    const float d1=490.6f*2.0f*precise::pow(ils1,3.0f),d2=17.46f*rp*3.87160635f*precise::pow(ils2,3.6357f);
    const float vts=precise::sqrt(rho_not/rho)*40.0f*(n1+n2)/(d1+d2);float vs;
    if(singular){if(t_k>273.25f)vs=fmax(vts*boost,vts*((rain_v-vts*boost)/(t_k-273.15f)));else vs=vts*boost;}
    else if(melting){const float sr=rs/(rs+rain_rr);vs=vts*boost*sr+rain_v*(1.0f-sr);}else vs=vts*boost;
    return vs;
}
template<typename R, typename P>
static inline void wrf39_snow_column(R r,P p,uint column){
    const uint ncol=uint(p.ny*p.nx);if(column>=ncol)return;const int nz=p.nz,ny=p.ny,nx=p.nx,j=int(column)/nx,i=int(column)-j*nx;
    float density[256],mass[256],velocity[256],flux[256],tendency[256],initial[256];int sediment_top=0,nstep=0;float above=0.0f,rain_above=0.0f;
    for(int k=nz-1;k>=0;--k){const uint x=uint(k*ny*nx+j*nx+i);const float qvk=fmax(1.0e-10f,r.qv[x]);const float rho=0.622f*r.pressure[x]/(287.04f*r.temperature[x]*(qvk+0.622f));density[k]=rho;initial[k]=r.qs[x];tendency[k]=0.0f;
        const float carried=r.melt_rain_density[x],rd=fabs(carried);const bool lqr=carried!=0.0f,rewritten=carried<0.0f;const float rr=!lqr?1.0e-12f:rewritten?fmax(1.0e-12f,r.melt_rain_qr[x]*rd):r.melt_rain_qr[x]*rd;
        const float rv=wrf39_rain_speed(r.melt_rain_qr[x],r.melt_rain_nr[x],carried,rho,rain_above);rain_above=rv;
        const float srho=r.reference_density[x];if(r.qs[x]>1.0e-12f && r.qs[x]*srho>1.0e-12f){const float rs=r.qs[x]*srho;velocity[k]=wrf39_snow_speed(rs,r.reference_temperature[x],rho,r.temperature[x],rr,rv,r.velocity_boost[x],r.snow_melt_marker[x]!=0.0f,p.singular_fall);mass[k]=rs;}else{mass[k]=r.qs[x]>1.0e-12f?r.qs[x]*srho:1.0e-12f;velocity[k]=above;}
        above=velocity[k];if(velocity[k]>1.0e-3f){sediment_top=max(sediment_top,k);nstep=max(nstep,int(float(p.dt)/(r.dz[x]/velocity[k])+1.0f));}}
    if(sediment_top==nz-1)sediment_top=nz-2;nstep=max(nstep,1);const float onstep=1.0f/float(nstep),dts=float(p.dt)*onstep;float exported=0.0f;
    for(int step=0;step<nstep;++step){for(int k=nz-1;k>=0;--k)flux[k]=velocity[k]*mass[k];int k=nz-1;uint x=uint(k*ny*nx+j*nx+i);float iz=1.0f/r.dz[x],ir=1.0f/density[k];tendency[k]-=flux[k]*iz*onstep*ir;mass[k]=fmax(1.0e-12f,mass[k]-flux[k]*iz*dts);
        for(k=sediment_top;k>=0;--k){x=uint(k*ny*nx+j*nx+i);iz=1.0f/r.dz[x];ir=1.0f/density[k];const float div=flux[k+1]-flux[k];tendency[k]+=div*iz*onstep*ir;mass[k]=fmax(1.0e-12f,mass[k]+div*iz*dts);}if(mass[0]>1.0e-11f)exported+=flux[0]*dts;}
    for(int k=0;k<nz;++k){const uint x=uint(k*ny*nx+j*nx+i);const float qnew=initial[k]+tendency[k]*float(p.dt);r.qs[x]=qnew<=1.0e-12f?0.0f:qnew;}
    r.rainncv[column]+=exported;r.snowncv[column]+=exported;r.rainnc[column]+=exported;r.snownc[column]+=exported;
}
kernel void thompson_aa_wrf39_snow_sediment_64(constant ThompsonAaWrf39SnowSediment64Resources& r [[buffer(0)]],constant ThompsonAaWrf39SnowSediment64Params& p [[buffer(1)]],uint c [[thread_position_in_grid]]){wrf39_snow_column(r,p,c);}
kernel void thompson_aa_wrf39_snow_sediment_256(constant ThompsonAaWrf39SnowSediment256Resources& r [[buffer(0)]],constant ThompsonAaWrf39SnowSediment256Params& p [[buffer(1)]],uint c [[thread_position_in_grid]]){wrf39_snow_column(r,p,c);}

kernel void thompson_aa_wrf39_warm_snow_boost(constant ThompsonAaWrf39WarmSnowBoostResources& r [[buffer(0)]],constant ThompsonAaWrf39WarmSnowBoostParams& p [[buffer(1)]],uint idx [[thread_position_in_grid]]){
    if(idx>=uint(p.n))return;if(r.entry_warm_mask[idx]!=0.0f)r.velocity_boost[idx]=1.5f;
}
