#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#include "thompson_aerosol_wrf39_sed_family_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
#include "soft_glibc_pow64.metalh"
#include "soft_gfk_log.metalh"
#include "thompson_aerosol_exner_libm.metalh"

static inline ulong wrf39_sed64(float x) {
    return soft64_from_f32_bits(as_type<uint>(x));
}
static inline float wrf39_sed32(ulong x) {
    return as_type<float>(soft64_to_f32_bits(x));
}
// thompson_aa_pow (glibc binary64 pow) and thompson_aa_powf (the FMA
// powf core), the pinned words every mp=28 unit evaluates.
static inline ulong wrf39_sed_pow64(ulong x, ulong y) {
    return soft_glibc_pow64(x, y);
}
static inline float wrf39_powf(float x, float y) {
    return as_type<float>(thompson_exner_powf_bits(as_type<uint>(x),
                                                   as_type<uint>(y)));
}
static inline ulong wrf39_sed_pow64f(ulong x, float y) {
    return wrf39_sed_pow64(x, wrf39_sed64(y));
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
    const float lam_f = wrf39_powf(lam_arg, 0.33333334326744080f);
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
            ulong lam=wrf39_sed64(wrf39_powf(la,0.33333334326744080f));
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
        const float la=am_i*6.0f*nnew/qnew; ulong lam=wrf39_sed64(wrf39_powf(la,0.33333334326744080f)); float d=wrf39_sed32(soft64_div(0x4010000000000000ul,lam));
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
    const ulong lam = wrf39_sed_pow64(base, wrf39_sed64(0.25f));
    const ulong ilam = soft64_div(0x3ff0000000000000ul, lam);
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const float prefix = precise::sqrt(rho_not / rho) * 442.0f
        * 20.3632278f * (1.0f / 6.0f);
    float v = wrf39_sed32(soft64_mul(wrf39_sed64(prefix),
        wrf39_sed_pow64(ilam, wrf39_sed64(0.89f))));
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
    const float smoc=wrf39_powf(10.0f,loga)*wrf39_powf(smob,exponent),mr=smob/smoc;
    float ils1=1.0f/(mr*20.78f+100.0f),ils2=1.0f/(mr*3.29f+100.0f);const float rp=wrf39_powf(mr,0.6357f);
    const float n1=490.6f*3.51325202f*wrf39_powf(ils1,3.55f),n2=17.46f*rp*7.61279917f*wrf39_powf(ils2,4.1857f);
    ils1=1.0f/(mr*20.78f);ils2=1.0f/(mr*3.29f);
    const float d1=490.6f*2.0f*wrf39_powf(ils1,3.0f),d2=17.46f*rp*3.87160635f*wrf39_powf(ils2,3.6357f);
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

template<typename R, typename P>
static inline void wrf39_rain_column(R r, P p, uint column) {
    const uint ncol = uint(p.ny * p.nx);
    if (column >= ncol) return;
    const int nz=p.nz, ny=p.ny, nx=p.nx, j=int(column)/nx, i=int(column)-j*nx;
    float density[256], mass[256], number[256], vm[256], vn[256], fm[256], fn[256];
    float qt[256], nt[256], qr0[256], nr0[256];
    const float am_r=3.1415926536f*1000.0f/6.0f, org3=1.0f/6.0f;
    const float rho_not=101325.0f/(287.05f*298.0f);
    const float dt=float(p.dt);
    int sediment_top=0, nstep=0;
    float above_m=0.0f, above_n=0.0f;
    for(int k=nz-1;k>=0;--k){
        const uint x=uint(k*ny*nx+j*nx+i);
        const float qvk=fmax(1.0e-10f,r.qv[x]);
        const float rho=0.622f*r.pressure[x]/(287.04f*r.temperature[x]*(qvk+0.622f));
        const float carried=r.reference_density[x], rd=fabs(carried);
        density[k]=rho; qr0[k]=r.qr[x]; nr0[k]=r.nr[x]; qt[k]=0.0f; nt[k]=0.0f;
        const bool lqr=carried!=0.0f, rewritten=carried<0.0f;
        const float rr=!lqr?1.0e-12f:rewritten?fmax(1.0e-12f,r.qr[x]*rd):r.qr[x]*rd;
        if(rr>1.0e-12f){
            const float nn=fmax(1.0e-6f,r.nr[x]*rd);
            const ulong lam=wrf39_sed64(wrf39_powf(am_r*6.0f*nn/rr,1.0f/3.0f));
            mass[k]=rr; number[k]=nn;
            const float rhof=precise::sqrt(rho_not/rho);
            const ulong shifted=soft64_add(lam,0x4068600000000000ul);
            const float mp=rhof*4854.0f*24.0f*org3;
            vm[k]=wrf39_sed32(soft64_mul(soft64_mul(wrf39_sed64(mp),
                wrf39_sed_pow64(lam,0x4010000000000000ul)),
                wrf39_sed_pow64(shifted,0xc014000000000000ul)));
            const float np=(rhof*4854.0f*3.3233511f)/1.3293403f;
            vn[k]=wrf39_sed32(soft64_mul(soft64_mul(wrf39_sed64(np),
                wrf39_sed_pow64(lam,0x4004000000000000ul)),
                wrf39_sed_pow64(shifted,0xc00c000000000000ul)));
        }else{
            mass[k]=rr; number[k]=lqr?fmax(1.0e-6f,r.nr[x]*rd):1.0e-6f;
            vm[k]=above_m; vn[k]=above_n;
        }
        above_m=vm[k]; above_n=vn[k];
        const float vmax=fmax(vm[k],vn[k]);
        if(vmax>1.0e-3f){sediment_top=max(sediment_top,k);nstep=max(nstep,int(dt/(r.dz[x]/vmax)+1.0f));}
    }
    if(sediment_top==nz-1)sediment_top=nz-2; nstep=max(nstep,1);
    const float onstep=1.0f/float(nstep), dts=dt*onstep; float exported=0.0f;
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
    for(int k=0;k<nz;++k){
        const uint x=uint(k*ny*nx+j*nx+i);
        const float qnew=qr0[k]+qt[k]*dt;
        float nnew=fmax(1.0e-6f/density[k],nr0[k]+nt[k]*dt);
        if(qnew<=1.0e-12f){r.qr[x]=0.0f;r.nr[x]=0.0f;continue;}
        ulong lam=wrf39_sed64(wrf39_powf(am_r*6.0f*nnew/qnew,1.0f/3.0f));
        float mvd=wrf39_sed32(soft64_div(0x400d604189374bc7ul,lam));
        if(mvd>2.5e-3f)mvd=2.5e-3f;else if(mvd<37.5e-6f)mvd=37.5e-6f;
        lam=soft64_div(0x400d604189374bc7ul,wrf39_sed64(mvd));
        const float pref=(org3*qnew)/am_r;
        nnew=wrf39_sed32(soft64_mul(wrf39_sed64(pref),wrf39_sed_pow64(lam,0x4008000000000000ul)));
        r.qr[x]=qnew; r.nr[x]=nnew;
    }
    r.rainncv[column]+=exported; r.rainnc[column]+=exported;
}
kernel void thompson_aa_wrf39_rain_sediment_64(
    constant ThompsonAaWrf39RainSediment64Resources& r [[buffer(0)]],
    constant ThompsonAaWrf39RainSediment64Params& p [[buffer(1)]],
    uint c [[thread_position_in_grid]]) { wrf39_rain_column(r,p,c); }
kernel void thompson_aa_wrf39_rain_sediment_256(
    constant ThompsonAaWrf39RainSediment256Resources& r [[buffer(0)]],
    constant ThompsonAaWrf39RainSediment256Params& p [[buffer(1)]],
    uint c [[thread_position_in_grid]]) { wrf39_rain_column(r,p,c); }

kernel void thompson_aa_wrf39_warm_snow_boost(constant ThompsonAaWrf39WarmSnowBoostResources& r [[buffer(0)]],constant ThompsonAaWrf39WarmSnowBoostParams& p [[buffer(1)]],uint idx [[thread_position_in_grid]]){
    if(idx>=uint(p.n))return;if(r.entry_warm_mask[idx]!=0.0f)r.velocity_boost[idx]=1.5f;
}

// ---------------------------------------------------------------------------
// WRF v4.6.1 accumulator-form fallout, thompson_aerosol_sed.cu's
// thompson_aa_{rain,ice}_sediment_accumulate, thompson_aa_snow_sediment,
// thompson_aa_graupel_sediment and thompson_aa_surface_precipitation, with the
// column kernels the host oracle runs.
// ---------------------------------------------------------------------------
#define W461_D(x) wrf39_sed64(x)
#define W461_F(x) wrf39_sed32(x)
#define W461_ONE 0x3ff0000000000000ul
#define W461_THREE 0x4008000000000000ul
#define W461_FOUR 0x4010000000000000ul
#define W461_RHO_NOT 1.18452108f
#define W461_AM_R 5.235988159e+02f
#define W461_OBMR 3.333333433e-01f
#define W461_AM_I 4.660029297e+02f
#define W461_D0R 50.0e-6f
#define W461_AM_G5 209.439514f

static inline float w461_air_density(float pressure, float temperature, float qv) {
    return (0.622f * pressure) / ((287.04f * temperature) * (qv + 0.622f));
}

static inline float w461_tau1_rain_number(float rr, float nr_m3) {
    float nr_work = fmax(1.0e-6f, nr_m3);
    const ulong lamr = W461_D(wrf39_powf(
        (((W461_AM_R * 6.0f) * 1.0f) * nr_work) / rr, W461_OBMR));
    const float mvd_num = (3.0f + 0.0f) + 0.672f;
    float mvd_r = W461_F(soft64_div(W461_D(mvd_num), lamr));
    const float d0r_low = W461_D0R * 0.75f;
    const bool high = mvd_r > 2.5e-3f;
    const bool low = mvd_r < d0r_low;
    if (high || low) {
        mvd_r = high ? 2.5e-3f : d0r_low;
        const ulong lamr_bounded = W461_D(mvd_num / mvd_r);
        nr_work = W461_F(soft64_div(soft64_mul(
            W461_D((1.0f * (1.0f / 6.0f)) * rr),
            wrf39_sed_pow64(lamr_bounded, W461_THREE)), W461_D(W461_AM_R)));
    }
    return nr_work;
}

static inline bool w461_rain_pair(float qr1d, float nr1d, float qrten,
        float nrten, float carried, float dt, thread float &rr, thread float &nn) {
    const float mass_working = qr1d + qrten * dt;
    const float number_working = nr1d + nrten * dt;
    if (carried == 0.0f) { rr = 1.0e-12f; nn = 1.0e-6f; return false; }
    if (carried < 0.0f) {
        rr = fmax(1.0e-12f, mass_working * -carried);
        nn = fmax(1.0e-6f, number_working * -carried);
        return true;
    }
    rr = mass_working * carried;
    nn = w461_tau1_rain_number(rr, number_working * carried);
    return true;
}

static inline ulong w461_rain_lamr(float rr, float nn) {
    return W461_D(wrf39_powf((((W461_AM_R * 6.0f) * 1.0f) * nn) / rr, W461_OBMR));
}

static inline float w461_rain_mass_speed(float rhof, ulong lamr) {
    return W461_F(soft64_mul(soft64_mul(
        W461_D(((rhof * 4854.0f) * 24.0f) * (1.0f / 6.0f)),
        wrf39_sed_pow64(lamr, W461_FOUR)),
        wrf39_sed_pow64(soft64_add(lamr, 0x4068600000000000ul), 0xc014000000000000ul)));
}

template<typename R, typename P>
static inline void w461_rain_accumulate_column(R r, P p, uint column) {
    if (column >= uint(p.ny * p.nx)) return;
    const int nz = p.nz, ny = p.ny, nx = p.nx, j = int(column) / nx, i = int(column) - j * nx;
    float density[256], rain_mass[256], rain_number[256], mv[256], nv[256];
    float mf[256], nf[256], mt[256], nt[256];
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const float dt = p.dt;
    bool any_rain = false;
    int sediment_top = 0, nstep = 0;
    float above_m = 0.0f, above_n = 0.0f;
    for (int k = nz - 1; k >= 0; --k) {
        const uint x = uint(k * ny * nx + j * nx + i);
        const float qvk = fmax(1.0e-10f, r.qv[x]);
        const float rho = w461_air_density(r.pressure[x], r.temperature[x], qvk);
        density[k] = rho; mt[k] = r.qrten[x]; nt[k] = r.nrten[x];
        float rr, nn;
        if (w461_rain_pair(r.qr1d[x], r.nr1d[x], r.qrten[x], r.nrten[x],
                           r.reference_density[x], dt, rr, nn)) any_rain = true;
        rain_mass[k] = rr; rain_number[k] = nn;
        const float rhof = precise::sqrt(rho_not / rho);
        if (rr > 1.0e-12f) {
            const ulong lamr = w461_rain_lamr(rr, nn);
            mv[k] = w461_rain_mass_speed(rhof, lamr);
            nv[k] = W461_F(soft64_mul(soft64_mul(
                W461_D(((rhof * 4854.0f) * 3.3233511f) / 1.3293403f),
                wrf39_sed_pow64(lamr, 0x4004000000000000ul)),
                wrf39_sed_pow64(soft64_add(lamr, 0x4068600000000000ul), 0xc00c000000000000ul)));
        } else { mv[k] = above_m; nv[k] = above_n; }
        above_m = mv[k]; above_n = nv[k];
        const float vmax = fmax(mv[k], nv[k]);
        if (vmax > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = r.dz[x] / vmax;
            nstep = max(nstep, int((dt / delta_tp) + 1.0f));
        }
    }
    float exported = 0.0f;
    if (any_rain) {
        if (sediment_top == nz - 1) sediment_top = nz - 2;
        const float onstep = nstep > 0 ? 1.0f / float(nstep) : 1.0f;
        const int steps = int(rint(1.0f / onstep));
        for (int step = 0; step < steps; ++step) {
            for (int k = nz - 1; k >= 0; --k) { mf[k] = mv[k] * rain_mass[k]; nf[k] = nv[k] * rain_number[k]; }
            int k = nz - 1; uint x = uint(k * ny * nx + j * nx + i);
            float odzq = 1.0f / r.dz[x], orho = 1.0f / density[k];
            mt[k] = mt[k] - ((mf[k] * odzq) * onstep) * orho;
            nt[k] = nt[k] - ((nf[k] * odzq) * onstep) * orho;
            rain_mass[k] = fmax(1.0e-12f, rain_mass[k] - ((mf[k] * odzq) * dt) * onstep);
            rain_number[k] = fmax(1.0e-6f, rain_number[k] - ((nf[k] * odzq) * dt) * onstep);
            for (k = sediment_top; k >= 0; --k) {
                x = uint(k * ny * nx + j * nx + i);
                odzq = 1.0f / r.dz[x]; orho = 1.0f / density[k];
                const float md = mf[k + 1] - mf[k], nd = nf[k + 1] - nf[k];
                mt[k] = mt[k] + ((md * odzq) * onstep) * orho;
                nt[k] = nt[k] + ((nd * odzq) * onstep) * orho;
                rain_mass[k] = fmax(1.0e-12f, rain_mass[k] + ((md * odzq) * dt) * onstep);
                rain_number[k] = fmax(1.0e-6f, rain_number[k] + ((nd * odzq) * dt) * onstep);
            }
            if (rain_mass[0] > 1.0e-12f * 1000.0f) exported = exported + (mf[0] * dt) * onstep;
        }
        for (int k = 0; k < nz; ++k) {
            const uint x = uint(k * ny * nx + j * nx + i);
            r.qrten[x] = mt[k]; r.nrten[x] = nt[k];
        }
    }
    if (p.accumulate_surface == 2) { r.rainncv[column] = exported; return; }
    if (p.accumulate_surface != 0) r.rainncv[column] += exported;
    else r.rainncv[column] = exported;
    r.rainnc[column] += exported;
}
kernel void thompson_aa_rain_sediment_accumulate_64(
    constant ThompsonAaRainSedimentAccumulate64Resources& r [[buffer(0)]],
    constant ThompsonAaRainSedimentAccumulate64Params& p [[buffer(1)]],
    uint c [[thread_position_in_grid]]) { w461_rain_accumulate_column(r, p, c); }
kernel void thompson_aa_rain_sediment_accumulate_256(
    constant ThompsonAaRainSedimentAccumulate256Resources& r [[buffer(0)]],
    constant ThompsonAaRainSedimentAccumulate256Params& p [[buffer(1)]],
    uint c [[thread_position_in_grid]]) { w461_rain_accumulate_column(r, p, c); }

template<typename R, typename P>
static inline void w461_ice_accumulate_column(R r, P p, uint column) {
    if (column >= uint(p.ny * p.nx)) return;
    const int nz = p.nz, ny = p.ny, nx = p.nx, j = int(column) / nx, i = int(column) - j * nx;
    float density[256], ice_mass[256], ice_number[256], mv[256], nv[256];
    float mf[256], nf[256], mt[256], nt[256];
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const float dt = p.dt;
    const bool rain_refreshes_rhof = r.rain_active_columns[column] != 0.0f;
    bool any_ice = false;
    int sediment_top = 0, nstep = 0;
    float above_m = 0.0f, above_n = 0.0f;
    for (int k = nz - 1; k >= 0; --k) {
        const uint x = uint(k * ny * nx + j * nx + i);
        const float qvk = fmax(1.0e-10f, r.qv[x]);
        const float rho = w461_air_density(r.pressure[x], r.temperature[x], qvk);
        density[k] = rho; mt[k] = r.qiten[x]; nt[k] = r.niten[x];
        const float held = r.reference_density[x];
        const float mass_working = r.qi1d[x] + r.qiten[x] * dt;
        float ri, nn;
        if (mass_working > 1.0e-12f) {
            ri = mass_working * held;
            nn = fmax(1.0e-6f, (r.ni1d[x] + r.niten[x] * dt) * held);
            any_ice = true;
        } else { ri = 1.0e-12f; nn = 1.0e-6f; }
        ice_mass[k] = ri; ice_number[k] = nn;
        const float rhof = precise::sqrt(rho_not / (rain_refreshes_rhof ? rho : held));
        if (ri > 1.0e-12f) {
            const ulong lami = W461_D(wrf39_powf(
                ((((W461_AM_I * 6.0f) * 1.0f) * nn) / ri), W461_OBMR));
            const ulong ilami = soft64_div(W461_ONE, lami);
            mv[k] = W461_F(soft64_mul(W461_D(((rhof * 1493.9f) * 24.0f) * (1.0f / 6.0f)), ilami));
            nv[k] = W461_F(soft64_mul(W461_D(((rhof * 1493.9f) * 3.3233511f) / 1.3293403f), ilami));
        } else { mv[k] = above_m; nv[k] = above_n; }
        above_m = mv[k]; above_n = nv[k];
        if (mv[k] > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = r.dz[x] / mv[k];
            nstep = max(nstep, int((dt / delta_tp) + 1.0f));
        }
    }
    float exported = 0.0f;
    if (any_ice) {
        if (sediment_top == nz - 1) sediment_top = nz - 2;
        const float onstep = nstep > 0 ? 1.0f / float(nstep) : 1.0f;
        const int steps = int(rint(1.0f / onstep));
        for (int step = 0; step < steps; ++step) {
            for (int k = nz - 1; k >= 0; --k) { mf[k] = mv[k] * ice_mass[k]; nf[k] = nv[k] * ice_number[k]; }
            int k = nz - 1; uint x = uint(k * ny * nx + j * nx + i);
            float odzq = 1.0f / r.dz[x], orho = 1.0f / density[k];
            mt[k] = mt[k] - ((mf[k] * odzq) * onstep) * orho;
            nt[k] = nt[k] - ((nf[k] * odzq) * onstep) * orho;
            ice_mass[k] = fmax(1.0e-12f, ice_mass[k] - ((mf[k] * odzq) * dt) * onstep);
            ice_number[k] = fmax(1.0e-6f, ice_number[k] - ((nf[k] * odzq) * dt) * onstep);
            for (k = sediment_top; k >= 0; --k) {
                x = uint(k * ny * nx + j * nx + i);
                odzq = 1.0f / r.dz[x]; orho = 1.0f / density[k];
                const float md = mf[k + 1] - mf[k], nd = nf[k + 1] - nf[k];
                mt[k] = mt[k] + ((md * odzq) * onstep) * orho;
                nt[k] = nt[k] + ((nd * odzq) * onstep) * orho;
                ice_mass[k] = fmax(1.0e-12f, ice_mass[k] + ((md * odzq) * dt) * onstep);
                ice_number[k] = fmax(1.0e-6f, ice_number[k] + ((nd * odzq) * dt) * onstep);
            }
            if (ice_mass[0] > 1.0e-12f * 1000.0f) exported = exported + (mf[0] * dt) * onstep;
        }
        for (int k = 0; k < nz; ++k) {
            const uint x = uint(k * ny * nx + j * nx + i);
            r.qiten[x] = mt[k]; r.niten[x] = nt[k];
        }
    }
    r.snowncv[column] = exported;
    if (p.export_only != 0) return;
    r.rainncv[column] = exported;
    r.rainnc[column] += exported;
    r.snownc[column] += exported;
}
kernel void thompson_aa_ice_sediment_accumulate_64(
    constant ThompsonAaIceSedimentAccumulate64Resources& r [[buffer(0)]],
    constant ThompsonAaIceSedimentAccumulate64Params& p [[buffer(1)]],
    uint c [[thread_position_in_grid]]) { w461_ice_accumulate_column(r, p, c); }
kernel void thompson_aa_ice_sediment_accumulate_256(
    constant ThompsonAaIceSedimentAccumulate256Resources& r [[buffer(0)]],
    constant ThompsonAaIceSedimentAccumulate256Params& p [[buffer(1)]],
    uint c [[thread_position_in_grid]]) { w461_ice_accumulate_column(r, p, c); }

static inline float w461_field_loga(float tc, float mom) {
    float v = 5.065339f;
    v = v + (-0.062659f) * tc;
    v = v + (-3.032362f) * mom;
    v = v + ((0.029469f * tc) * mom);
    v = v + ((-0.000285f * tc) * tc);
    v = v + ((0.31255f * mom) * mom);
    v = v + (((0.000204f * tc) * tc) * mom);
    v = v + (((0.003199f * tc) * mom) * mom);
    v = v + (((0.0f * tc) * tc) * tc);
    v = v + (((-0.015952f * mom) * mom) * mom);
    return v;
}
static inline float w461_field_b(float tc, float mom) {
    float v = 0.476221f;
    v = v + (-0.015896f) * tc;
    v = v + 0.165977f * mom;
    v = v + ((0.007468f * tc) * mom);
    v = v + ((-0.000141f * tc) * tc);
    v = v + ((0.060366f * mom) * mom);
    v = v + (((0.000079f * tc) * tc) * mom);
    v = v + (((0.000594f * tc) * mom) * mom);
    v = v + (((0.0f * tc) * tc) * tc);
    v = v + (((-0.003577f * mom) * mom) * mom);
    return v;
}

template<typename R, typename P>
static inline void w461_snow_column(R r, P p, uint column) {
    if (column >= uint(p.ny * p.nx)) return;
    const int nz = p.nz, ny = p.ny, nx = p.nx, j = int(column) / nx, i = int(column) - j * nx;
    float density[256], snow_mass[256], velocity[256], flux[256], tendency[256];
    const float dt = p.dt;
    bool any_rain = false;
    for (int k = 0; k < nz; ++k)
        if (r.rain_density[uint(k * ny * nx + j * nx + i)] != 0.0f) { any_rain = true; break; }
    bool any_snow = false;
    int sediment_top = 0, nstep = 0;
    float rain_speed = 0.0f, speed_above = 0.0f;
    for (int k = nz - 1; k >= 0; --k) {
        const uint x = uint(k * ny * nx + j * nx + i);
        const float rho = w461_air_density(r.pressure[x], r.temperature[x], fmax(1.0e-10f, r.qv[x]));
        density[k] = rho; tendency[k] = p.accumulate ? r.qsten[x] : 0.0f;
        const float held = r.reference_density[x];
        float rain_rr = 1.0e-12f;
        if (any_rain) {
            float rr, nn;
            w461_rain_pair(r.qr1d[x], r.nr1d[x], r.qrten[x], r.nrten[x],
                           r.rain_density[x], dt, rr, nn);
            rain_rr = rr;
            if (rr > 1.0e-12f) {
                const float rhof_r = precise::sqrt(W461_RHO_NOT / rho);
                rain_speed = w461_rain_mass_speed(rhof_r, w461_rain_lamr(rr, nn));
            }
        }
        const float working = p.accumulate ? r.qs[x] + r.qsten[x] * dt : r.qs[x];
        float rs = 1.0e-12f;
        if (working > 1.0e-12f) { rs = working * held; any_snow = true; }
        snow_mass[k] = rs;
        if (rs > 1.0e-12f) {
            const float tc0 = fmin(-0.1f, r.reference_temperature[x] - 273.15f);
            const float smob = rs * 1.449275398e+01f;
            const float smoc = wrf39_powf(10.0f, w461_field_loga(tc0, 3.0f))
                * wrf39_powf(smob, w461_field_b(tc0, 3.0f));
            const float xds = smoc / smob;
            const float mrat = 1.0f / xds;
            float ils1 = 1.0f / ((mrat * 20.78f) + 100.0f);
            float ils2 = 1.0f / ((mrat * 3.29f) + 100.0f);
            const float mrat_mu = wrf39_powf(mrat, 0.6357f);
            const float t1 = (490.6f * 3.51325202f) * wrf39_powf(ils1, 3.54999995f);
            const float t2 = ((17.46f * mrat_mu) * 7.61279917f) * wrf39_powf(ils2, 4.18569994f);
            ils1 = 1.0f / (mrat * 20.78f);
            ils2 = 1.0f / (mrat * 3.29f);
            const float t3 = (490.6f * 2.0f) * wrf39_powf(ils1, 3.0f);
            const float t4 = ((17.46f * mrat_mu) * 3.87160635f) * wrf39_powf(ils2, 3.63569999f);
            const float rhof = precise::sqrt(W461_RHO_NOT / (any_rain ? rho : held));
            const float vts = ((rhof * 40.0f) * (t1 + t2)) / (t3 + t4);
            if (r.snow_melt_marker[x] != 0.0f) {
                const float sr = rs / (rs + rain_rr);
                velocity[k] = (vts * sr) + ((1.0f - sr) * rain_speed);
            } else {
                velocity[k] = vts * r.velocity_boost[x];
            }
        } else { velocity[k] = speed_above; }
        speed_above = velocity[k];
        if (velocity[k] > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = r.dz[x] / velocity[k];
            nstep = max(nstep, int((dt / delta_tp) + 1.0f));
        }
    }
    float precipitation = 0.0f;
    if (any_snow) {
        if (sediment_top == nz - 1) sediment_top = nz - 2;
        const float onstep = nstep > 0 ? 1.0f / float(nstep) : 1.0f;
        const float inv = 1.0f / onstep;
        const int steps = int(floor(inv + 0.5f));
        for (int step = 0; step < steps; ++step) {
            for (int k = nz - 1; k >= 0; --k) flux[k] = velocity[k] * snow_mass[k];
            int k = nz - 1; uint x = uint(k * ny * nx + j * nx + i);
            float odzq = 1.0f / r.dz[x], orho = 1.0f / density[k];
            tendency[k] = tendency[k] - ((flux[k] * odzq) * onstep) * orho;
            snow_mass[k] = fmax(1.0e-12f, snow_mass[k] - ((flux[k] * odzq) * dt) * onstep);
            for (k = sediment_top; k >= 0; --k) {
                x = uint(k * ny * nx + j * nx + i);
                odzq = 1.0f / r.dz[x]; orho = 1.0f / density[k];
                const float divergence = flux[k + 1] - flux[k];
                tendency[k] = tendency[k] + ((divergence * odzq) * onstep) * orho;
                snow_mass[k] = fmax(1.0e-12f, snow_mass[k] + ((divergence * odzq) * dt) * onstep);
            }
            if (snow_mass[0] > 1.0e-12f * 1000.0f) precipitation = precipitation + (flux[0] * dt) * onstep;
        }
    }
    for (int k = 0; k < nz; ++k) {
        const uint x = uint(k * ny * nx + j * nx + i);
        const float updated = r.qs[x] + tendency[k] * dt;
        r.qs[x] = updated <= 1.0e-12f ? 0.0f : updated;
        if (p.accumulate) r.qsten[x] = tendency[k];
    }
    r.snow_precipitation[column] = precipitation;
}
kernel void thompson_aa_snow_sediment_64(
    constant ThompsonAaSnowSediment64Resources& r [[buffer(0)]],
    constant ThompsonAaSnowSediment64Params& p [[buffer(1)]],
    uint c [[thread_position_in_grid]]) { w461_snow_column(r, p, c); }
kernel void thompson_aa_snow_sediment_256(
    constant ThompsonAaSnowSediment256Resources& r [[buffer(0)]],
    constant ThompsonAaSnowSediment256Params& p [[buffer(1)]],
    uint c [[thread_position_in_grid]]) { w461_snow_column(r, p, c); }

// thompson_aa_log10f on float words.
static inline float w461_log10f(float xf) {
    uint hx = as_type<uint>(xf);
    const uint x = hx;
    int k = 0;
    if (hx < 0x00800000u || (hx & 0x80000000u)) {
        if ((hx & 0x7fffffffu) == 0u) return as_type<float>(0xff800000u);
        if (hx & 0x80000000u) return as_type<float>(0x7fc00000u);
        k -= 25;
        hx = soft32_mul(x, 0x4c000000u);
    }
    if (hx > 0x7f7fffffu) return as_type<float>(soft32_add(x, x));
    k += int(hx >> 23) - 127;
    const int i = int((uint(k) & 0x80000000u) >> 31);
    hx = (hx & 0x007fffffu) | (uint(0x7f - i) << 23);
    const uint y = soft64_to_f32_bits(soft64_from_i32(k + i));
    const uint z = soft32_add(soft32_mul(0x3ede5bd9u, soft_gfk_log(hx)),
                              soft32_mul(y, 0x355427dbu));
    return as_type<float>(soft32_add(z, soft32_mul(y, 0x3e9a2080u)));
}

// :3289-3297, the classic graupel number from the working content rg.
static inline float w461_graupel_number_m3(float rg) {
    const float ygra1 = w461_log10f(fmax(1.0e-9f, rg));
    float zans1 = 3.0f + ((2.0f / 7.0f) * (ygra1 + 8.0f));
    zans1 = fmax(2.0f, fmin(zans1, 6.0f));
    const ulong n0_exp = W461_D(wrf39_powf(10.0f, zans1));
    const ulong lam_exp = wrf39_sed_pow64(soft64_div(soft64_mul(soft64_mul(
        n0_exp, W461_D(W461_AM_G5)), W461_D(6.0f)), W461_D(rg)), W461_D(0.25f));
    const ulong lamg = soft64_mul(lam_exp, W461_D(wrf39_powf(
        ((6.0f * 1.0f) * 0.166666672f), 0.333333343f)));
    return W461_F(soft64_div(soft64_mul(
        W461_D((1.0f * 0.166666672f) * rg), wrf39_sed_pow64(lamg, W461_THREE)),
        W461_D(W461_AM_G5)));
}

template<typename R, typename P>
static inline void w461_graupel_column(R r, P p, uint column) {
    if (column >= uint(p.ny * p.nx)) return;
    const int nz = p.nz, ny = p.ny, nx = p.nx, j = int(column) / nx, i = int(column) - j * nx;
    const float dt = p.dt;
    if (r.active_columns[column] == 0.0f) {
        r.graupel_precipitation[column] = 0.0f;
        if (p.accumulate) for (int k = 0; k < nz; ++k) {
            const uint x = uint(k * ny * nx + j * nx + i);
            r.qg[x] = r.qg[x] + r.qgten[x] * dt;
            r.graupel_number[x] = r.graupel_number[x] + r.ngten[x] * dt;
        }
        return;
    }
    float density[256], graupel_mass[256], number[256], mv[256], nv[256];
    float mf[256], nf[256], mt[256], nt[256];
    bool any_rain = false;
    for (int k = 0; k < nz; ++k)
        if (r.rain_density[uint(k * ny * nx + j * nx + i)] != 0.0f) { any_rain = true; break; }
    int sediment_top = 0, nstep = 0;
    float above_m = 0.0f, above_n = 0.0f;
    for (int k = nz - 1; k >= 0; --k) {
        const uint x = uint(k * ny * nx + j * nx + i);
        const float rho = w461_air_density(r.pressure[x], r.temperature[x], fmax(1.0e-10f, r.qv[x]));
        density[k] = rho;
        mt[k] = p.accumulate ? r.qgten[x] : 0.0f;
        nt[k] = p.accumulate ? r.ngten[x] : 0.0f;
        const float held = r.reference_density[x];
        const float working = p.accumulate ? r.qg[x] + r.qgten[x] * dt : r.qg[x];
        float rg = 1.0e-12f, ng = 1.0e-6f;
        if (working > 1.0e-12f) { rg = working * held; ng = w461_graupel_number_m3(rg); }
        graupel_mass[k] = rg; number[k] = ng;
        if (rg > 1.0e-12f) {
            const ulong lamg = W461_D(wrf39_powf(
                ((((W461_AM_G5 * 6.0f) * 1.0f) * ng) / rg), 0.333333343f));
            const ulong ilamg = soft64_div(W461_ONE, lamg);
            const ulong fall = wrf39_sed_pow64(ilamg, W461_D(0.89f));
            const float rhof = precise::sqrt(W461_RHO_NOT / (any_rain ? rho : held));
            const float rhof_afall = rhof * 442.0f;
            mv[k] = W461_F(soft64_mul(W461_D((rhof_afall * 20.3632278f) * 0.166666672f), fall));
            nv[k] = W461_F(soft64_mul(W461_D((rhof_afall * 2.94954014f) / 1.32934034f), fall));
        } else { mv[k] = above_m; nv[k] = above_n; }
        above_m = mv[k]; above_n = nv[k];
        if (mv[k] > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = r.dz[x] / mv[k];
            nstep = max(nstep, int((dt / delta_tp) + 1.0f));
        }
    }
    if (sediment_top == nz - 1) sediment_top = nz - 2;
    const float onstep = nstep > 0 ? 1.0f / float(nstep) : 1.0f;
    const int steps = int(floor((1.0f / onstep) + 0.5f));
    float precipitation = 0.0f;
    for (int step = 0; step < steps; ++step) {
        for (int k = nz - 1; k >= 0; --k) { mf[k] = mv[k] * graupel_mass[k]; nf[k] = nv[k] * number[k]; }
        int k = nz - 1; uint x = uint(k * ny * nx + j * nx + i);
        float odzq = 1.0f / r.dz[x], orho = 1.0f / density[k];
        mt[k] = mt[k] - ((mf[k] * odzq) * onstep) * orho;
        nt[k] = nt[k] - ((nf[k] * odzq) * onstep) * orho;
        graupel_mass[k] = fmax(1.0e-12f, graupel_mass[k] - ((mf[k] * odzq) * dt) * onstep);
        number[k] = fmax(1.0e-6f, number[k] - ((nf[k] * odzq) * dt) * onstep);
        for (k = sediment_top; k >= 0; --k) {
            x = uint(k * ny * nx + j * nx + i);
            odzq = 1.0f / r.dz[x]; orho = 1.0f / density[k];
            const float md = mf[k + 1] - mf[k], nd = nf[k + 1] - nf[k];
            mt[k] = mt[k] + ((md * odzq) * onstep) * orho;
            nt[k] = nt[k] + ((nd * odzq) * onstep) * orho;
            graupel_mass[k] = fmax(1.0e-12f, graupel_mass[k] + ((md * odzq) * dt) * onstep);
            number[k] = fmax(1.0e-6f, number[k] + ((nd * odzq) * dt) * onstep);
        }
        if (graupel_mass[0] > 1.0e-12f * 1000.0f) precipitation = precipitation + (mf[0] * dt) * onstep;
    }
    for (int k = 0; k < nz; ++k) {
        const uint x = uint(k * ny * nx + j * nx + i);
        r.qg[x] = r.qg[x] + mt[k] * dt;
        r.graupel_number[x] = r.graupel_number[x] + nt[k] * dt;
        if (p.accumulate) { r.qgten[x] = mt[k]; r.ngten[x] = nt[k]; }
    }
    r.graupel_precipitation[column] = precipitation;
}
kernel void thompson_aa_graupel_sediment_64(
    constant ThompsonAaGraupelSediment64Resources& r [[buffer(0)]],
    constant ThompsonAaGraupelSediment64Params& p [[buffer(1)]],
    uint c [[thread_position_in_grid]]) { w461_graupel_column(r, p, c); }
kernel void thompson_aa_graupel_sediment_256(
    constant ThompsonAaGraupelSediment256Resources& r [[buffer(0)]],
    constant ThompsonAaGraupelSediment256Params& p [[buffer(1)]],
    uint c [[thread_position_in_grid]]) { w461_graupel_column(r, p, c); }

kernel void thompson_aa_surface_precipitation(
    constant ThompsonAaSurfacePrecipitationResources& r [[buffer(0)]],
    constant ThompsonAaSurfacePrecipitationParams& p [[buffer(1)]],
    uint c [[thread_position_in_grid]]) {
    if (c >= uint(p.ncol)) return;
    const float pptrain = r.sr[c], pptsnow = r.rainncv[c];
    const float pptice = r.snowncv[c], pptgraul = r.graupelncv[c];
    const float total = ((pptrain + pptsnow) + pptgraul) + pptice;
    r.rainncv[c] = total;
    r.rainnc[c] = (((r.rainnc[c] + pptrain) + pptsnow) + pptgraul) + pptice;
    r.snowncv[c] = pptsnow + pptice;
    r.snownc[c] = (r.snownc[c] + pptsnow) + pptice;
    r.graupelncv[c] = pptgraul;
    r.graupelnc[c] = r.graupelnc[c] + pptgraul;
    r.sr[c] = ((pptsnow + pptgraul) + pptice) / (total + 1.0e-12f);
}

kernel void thompson_aa_entry_warm_mask(
    constant ThompsonAaEntryWarmMaskResources& r [[buffer(0)]],
    constant ThompsonAaEntryWarmMaskParams& p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= uint(p.ncol)) return;
    bool at_or_below_melting = false;
    for (int k = p.nz - 1; k >= 0; --k) {
        const uint x = uint(k) * uint(p.ncol) + column;
        const float temp = r.temperature[x];
        if (temp - 273.15f > 0.0f) at_or_below_melting = true;
        r.warm_mask[x] = temp >= 273.15f ? (at_or_below_melting ? 1.0f : 2.0f) : 0.0f;
    }
}

kernel void thompson_aa_terminal_rain_ice(
    constant ThompsonAaTerminalRainIceResources& r [[buffer(0)]],
    constant ThompsonAaTerminalRainIceParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    if (r.micro_columns[idx % uint(p.ncol)] == 0.0f) return;
    const float dt = p.dt;
    const float rho = r.rho[idx];
    const float qi_new = r.qi[idx] + r.qiten[idx] * dt;
    float ni_new = fmax(1.0e-6f / rho, r.ni[idx] + r.niten[idx] * dt);
    if (qi_new <= 1.0e-12f) {
        r.qi[idx] = 0.0f; r.ni[idx] = 0.0f;
    } else {
        ulong lami = W461_D(wrf39_powf(
            ((((W461_AM_I * 6.0f) * 1.0f) * ni_new) / qi_new), W461_OBMR));
        const ulong ilami = soft64_div(W461_ONE, lami);
        const float xdi = W461_F(soft64_mul(W461_FOUR, ilami));
        if (xdi < 5.0e-6f) lami = W461_D(4.0f / 5.0e-6f);
        else if (xdi > 300.0e-6f) lami = W461_D(4.0f / 300.0e-6f);
        const ulong value = soft64_mul(W461_D((((1.0f * (1.0f / 6.0f)) * qi_new) / W461_AM_I)),
                                       wrf39_sed_pow64(lami, W461_THREE));
        const ulong ceiling = soft64_div(0x412e7cb000000000ul, W461_D(rho));
        ni_new = W461_F(soft64_less(ceiling, value) ? ceiling : value);
        r.qi[idx] = qi_new; r.ni[idx] = ni_new;
    }
    const float qr_new = r.qr[idx] + r.qrten[idx] * dt;
    float nr_new = fmax(1.0e-6f / rho, r.nr[idx] + r.nrten[idx] * dt);
    if (qr_new <= 1.0e-12f) {
        r.qr[idx] = 0.0f; r.nr[idx] = 0.0f;
    } else {
        ulong lamr = W461_D(wrf39_powf(
            ((((W461_AM_R * 6.0f) * 1.0f) * nr_new) / qr_new), W461_OBMR));
        const float mvd_num = (3.0f + 0.0f) + 0.672f;
        float mvd_r = W461_F(soft64_div(W461_D(mvd_num), lamr));
        if (mvd_r > 2.5e-3f) mvd_r = 2.5e-3f;
        else if (mvd_r < W461_D0R * 0.75f) mvd_r = W461_D0R * 0.75f;
        lamr = W461_D(mvd_num / mvd_r);
        nr_new = W461_F(soft64_div(soft64_mul(
            W461_D((1.0f * (1.0f / 6.0f)) * qr_new),
            wrf39_sed_pow64(lamr, W461_THREE)), W461_D(W461_AM_R)));
        r.qr[idx] = qr_new; r.nr[idx] = nr_new;
    }
}

kernel void thompson_aa_graupel_number_finalize(
    constant ThompsonAaGraupelNumberFinalizeResources& r [[buffer(0)]],
    constant ThompsonAaGraupelNumberFinalizeParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    const float rho = r.terminal_density[idx];
    const float q = r.qg[idx];
    float number = fmax(1.0e-6f / rho, r.graupel_number[idx]);
    if (q <= 1.0e-12f) { r.qg[idx] = 0.0f; r.graupel_number[idx] = 0.0f; return; }
    ulong lamg = W461_D(wrf39_powf(
        (((W461_AM_G5 * 6.0f) * 1.0f) * number) / q, 0.333333343f));
    const float mvd_numerator = (3.0f + 0.0f) + 0.672f;
    float mvd = W461_F(soft64_div(W461_D(mvd_numerator), lamg));
    if (mvd > 25.4e-3f) mvd = 25.4e-3f;
    else if (mvd < W461_D0R) mvd = W461_D0R;
    lamg = W461_D(mvd_numerator / mvd);
    number = W461_F(soft64_div(soft64_mul(
        W461_D((1.0f * 0.166666672f) * q), wrf39_sed_pow64(lamg, W461_THREE)),
        W461_D(W461_AM_G5)));
    r.graupel_number[idx] = number;
}

// thompson_aa_wrf39_bound_ice_number: the fork's 499.D3 ceiling.
static inline float w461_bound_ice_number(float ice_mass, float density, float ni_per_kg, float ni_max) {
    if (ice_mass <= 1.0e-12f) return 0.0f;
    float ice_number = fmax(1.0e-6f, ni_per_kg * density);
    ulong lambda = W461_D(wrf39_powf(((W461_AM_I * 6.0f) * ice_number) / ice_mass,
                                     0.33333334326744080f));
    const float diameter = W461_F(soft64_div(W461_FOUR, lambda));
    if (diameter < 5.0e-6f || diameter > 300.0e-6f) {
        lambda = soft64_div(W461_FOUR, diameter < 5.0e-6f
            ? 0x3ed4f8b588e368f1ul : 0x3f33a92a30553261ul);
        const float scaled = ((0.16666667163372040f * ice_mass) / W461_AM_I)
            * W461_F(soft64_mul(soft64_mul(lambda, lambda), lambda));
        ice_number = diameter < 5.0e-6f ? fmin(ni_max, scaled) : scaled;
    }
    return fmin(ice_number, ni_max) / density;
}

kernel void thompson_aa_final_phase_cleanup(
    constant ThompsonAaFinalPhaseCleanupResources& r [[buffer(0)]],
    constant ThompsonAaFinalPhaseCleanupParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.size)) return;
    const bool accumulate_cloud = p.accumulate_cloud != 0;
    const bool accumulate_ice = p.accumulate_ice != 0;
    const float ni_max = p.wrf39 ? 499.0e3f : 999.0e3f;
    const bool accumulate_heat = p.accumulate_heat != 0;
    const float dt = p.dt;
    const float temp0 = r.temperature[idx];
    const float qv0 = fmax(1.0e-10f, r.qv[idx]);
    const float rho = w461_air_density(r.pressure[idx], temp0, qv0);
    const float inverse_cp = 1.0f / (1004.0f * (1.0f + 0.887f * qv0));
    const float odt = 1.0f / dt;
    const float qi_working = accumulate_ice ? r.qi[idx] + r.qiten[idx] * dt : r.qi[idx];
    if (temp0 > 273.15f && qi_working > 0.0f) {
        const float transferred = fmax(0.0f, qi_working);
        if (accumulate_cloud) r.qcten[idx] = r.qcten[idx] + transferred * odt;
        else r.qc[idx] += transferred;
        r.cloud_number_tendency[idx] = r.cloud_number_tendency[idx] + r.ice_number_entry[idx] * odt;
        if (accumulate_ice) {
            r.qiten[idx] = r.qiten[idx] - transferred * odt;
            r.niten[idx] = (-r.ice_number_entry[idx]) * odt;
        } else {
            r.qi[idx] = 0.0f; r.ni[idx] = 0.0f;
        }
        if (accumulate_heat) {
            r.tten[idx] = r.tten[idx] - ((334000.0f * r.heat_ocp[idx]) * transferred) * odt;
        } else {
            r.temperature[idx] -= (334000.0f * inverse_cp) * transferred;
        }
    }
    const float qc_working = accumulate_cloud ? r.qc[idx] + r.qcten[idx] * dt : r.qc[idx];
    if (temp0 < 235.16f && qc_working > 0.0f) {
        const float transferred = fmax(0.0f, qc_working);
        const float latent_vapor = 2.5e6f + (2106.0f - 4218.0f) * (temp0 - 273.15f);
        const float latent_fusion = 2.834e6f - latent_vapor;
        const float xnc = r.cloud_number_entry[idx] + r.cloud_number_tendency[idx] * dt;
        if (accumulate_cloud) r.qcten[idx] = r.qcten[idx] - transferred * odt;
        else r.qc[idx] = 0.0f;
        if (accumulate_ice) {
            r.qiten[idx] = r.qiten[idx] + transferred * odt;
            r.niten[idx] = r.niten[idx] + xnc * odt;
        } else {
            r.qi[idx] += transferred;
            r.ni[idx] += xnc;
        }
        r.cloud_number_tendency[idx] = r.cloud_number_tendency[idx] - xnc * odt;
        if (accumulate_heat) {
            const float lfus2 = 2.834e6f - r.heat_lvap[idx];
            r.tten[idx] = r.tten[idx] + ((lfus2 * r.heat_ocp[idx]) * transferred) * odt;
        } else {
            r.temperature[idx] += (latent_fusion * inverse_cp) * transferred;
        }
    }
    if (accumulate_heat)
        r.temperature[idx] = r.theta[idx] * r.exner[idx] + r.tten[idx] * dt;
    if (!accumulate_cloud && r.qc[idx] <= 1.0e-12f) r.qc[idx] = 0.0f;
    if (accumulate_ice) return;
    const float qi_local = r.qi[idx];
    if (qi_local <= 1.0e-12f) {
        r.qi[idx] = 0.0f; r.ni[idx] = 0.0f;
    } else if (qi_local * rho > 1.0e-12f) {
        r.ni[idx] = w461_bound_ice_number(qi_local * rho, rho, r.ni[idx], ni_max);
    } else {
        const float ni_local = fmax(1.0e-6f / rho, r.ni[idx]);
        ulong lami = W461_D(wrf39_powf(((W461_AM_I * 6.0f) * ni_local) / qi_local,
                                       1.0f / 3.0f));
        const float xdi = W461_F(soft64_mul(W461_FOUR, soft64_div(W461_ONE, lami)));
        if (xdi < 5.0e-6f) lami = W461_D(4.0f / 5.0e-6f);
        else if (xdi > 300.0e-6f) lami = W461_D(4.0f / 300.0e-6f);
        const ulong value = soft64_mul(W461_D(((1.0f / 6.0f) * qi_local) / W461_AM_I),
                                       wrf39_sed_pow64(lami, W461_THREE));
        const ulong ceiling = soft64_div(W461_D(ni_max), W461_D(rho));
        r.ni[idx] = W461_F(soft64_less(ceiling, value) ? ceiling : value);
    }
}
