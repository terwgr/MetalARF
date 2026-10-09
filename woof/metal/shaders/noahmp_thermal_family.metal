#include <metal_stdlib>
#include "noahmp_thermal_family_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

// Pinned WRF v4.6.1 leaves. PHASECHANGE and FRH2O depend on broader libm
// behavior; THERMOPROP is admitted with native Metal transcendental calls.
constant int NMP_NSOIL = 4;
constant int NMP_NSNOW = 3;
constant int NMP_NLAY = 7;
constant int NMP_OFF = 2;
constant float NMP_TFRZ = 273.16f;
constant float NMP_CWAT = 4.188e6f;
constant float NMP_CICE = 2.094e6f;
constant float NMP_CPAIR = 1004.64f;
constant float NMP_TKWAT = 0.6f;
constant float NMP_TKICE = 2.2f;
constant float NMP_DENH2O = 1000.0f;
constant float NMP_DENICE = 917.0f;

static inline float fadd(float a,float b){return a+b;}
static inline float fsub(float a,float b){return a-b;}
static inline float fmul(float a,float b){return a*b;}
static inline float fdiv(float a,float b){return a/b;}

kernel void noahmp_thermal_thermoprop(
    constant NoahmpThermalThermopropResources &r [[buffer(0)]],
    constant NoahmpThermalThermopropParams &q [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(q.ncase)) return;
    ulong xb=ulong(idx)*ulong(q.n_in), yb=ulong(idx)*ulong(q.n_out);
    int isnow=r.ixs[ulong(idx)*ulong(q.n_int)];
    int ist=r.ixs[ulong(idx)*ulong(q.n_int)+1];
    bool urban=r.ixs[ulong(idx)*ulong(q.n_int)+3] != 0;
    float dz[NMP_NLAY], snice[NMP_NSNOW], snliq[NMP_NSNOW];
    float smc[NMP_NSOIL], sh2o[NMP_NSOIL], stc[NMP_NLAY];
    float smcmax[NMP_NSOIL], quartz[NMP_NSOIL];
    for(int i=0;i<NMP_NLAY;++i) dz[i]=r.xs[xb+ulong(i)];
    for(int i=0;i<NMP_NSNOW;++i){
        snice[i]=r.xs[xb+ulong(NMP_NLAY+i)];
        snliq[i]=r.xs[xb+ulong(NMP_NLAY+NMP_NSNOW+i)];
    }
    int base=NMP_NLAY+2*NMP_NSNOW;
    for(int i=0;i<NMP_NSOIL;++i){smc[i]=r.xs[xb+ulong(base+i)];sh2o[i]=r.xs[xb+ulong(base+NMP_NSOIL+i)];}
    base+=2*NMP_NSOIL;
    for(int i=0;i<NMP_NLAY;++i)stc[i]=r.xs[xb+ulong(base+i)];
    base+=NMP_NLAY;
    float snowh=r.xs[xb+ulong(base)], dt=r.xs[xb+ulong(base+1)];
    base+=7;
    for(int i=0;i<NMP_NSOIL;++i)smcmax[i]=r.xs[xb+ulong(base+i)];
    float csoil=r.xs[xb+ulong(base+NMP_NSOIL)];
    for(int i=0;i<NMP_NSOIL;++i)quartz[i]=r.xs[xb+ulong(base+NMP_NSOIL+1+i)];
    float df[NMP_NLAY], hcpct[NMP_NLAY], fact[NMP_NLAY];
    float snicev[NMP_NSNOW], snliqv[NMP_NSNOW], epore[NMP_NSNOW];
    for(int i=0;i<NMP_NLAY;++i){df[i]=0.0f;hcpct[i]=0.0f;fact[i]=0.0f;}
    for(int i=0;i<NMP_NSNOW;++i){snicev[i]=0.0f;snliqv[i]=0.0f;epore[i]=0.0f;}
    for(int k=isnow+1;k<=0;++k){
        int j=k+NMP_OFF;
        snicev[j]=min(1.0f,fdiv(snice[j],fmul(dz[j],NMP_DENICE)));
        epore[j]=fsub(1.0f,snicev[j]);
        snliqv[j]=min(epore[j],fdiv(snliq[j],fmul(dz[j],NMP_DENH2O)));
    }
    for(int k=isnow+1;k<=0;++k){int j=k+NMP_OFF;float bds=fdiv(fadd(snice[j],snliq[j]),dz[j]);hcpct[j]=fadd(fmul(NMP_CICE,snicev[j]),fmul(NMP_CWAT,snliqv[j]));df[j]=fmul(3.2217e-6f,pow(bds,2.0f));}
    for(int k=1;k<=NMP_NSOIL;++k){
        int j=k+NMP_OFF,s=k-1; float sice=fsub(smc[s],sh2o[s]);
        float acc=fmul(sh2o[s],NMP_CWAT);
        acc=fadd(acc,fmul(fsub(1.0f,smcmax[s]),csoil));
        acc=fadd(acc,fmul(fsub(smcmax[s],smc[s]),NMP_CPAIR));
        acc=fadd(acc,fmul(sice,NMP_CICE)); hcpct[j]=acc;
        float sat=fdiv(smc[s],smcmax[s]), xu=fmul(smcmax[s],smc[s]>0.0f?fdiv(sh2o[s],smc[s]):1.0f);
        float thks=fmul(pow(7.7f,quartz[s]),pow(2.0f,fsub(1.0f,quartz[s])));
        float thksat=fmul(fmul(pow(thks,fsub(1.0f,smcmax[s])),pow(NMP_TKICE,fsub(smcmax[s],xu))),pow(0.57f,xu));
        float gammd=fmul(fsub(1.0f,smcmax[s]),2700.0f);
        float thkdry=fdiv(fadd(fmul(0.135f,gammd),64.7f),fsub(2700.0f,fmul(0.947f,gammd)));
        float ake=(fadd(sh2o[s],0.0005f)<smc[s])?sat:(sat>0.1f?fadd(log10(sat),1.0f):0.0f);
        df[j]=fadd(fmul(ake,fsub(thksat,thkdry)),thkdry);
    }
    if(urban)for(int k=1;k<=NMP_NSOIL;++k)df[k+NMP_OFF]=3.24f;
    if(ist==2)for(int k=1;k<=NMP_NSOIL;++k){int j=k+NMP_OFF;if(stc[j]>NMP_TFRZ){hcpct[j]=NMP_CWAT;df[j]=NMP_TKWAT;}else{hcpct[j]=NMP_CICE;df[j]=NMP_TKICE;}}
    for(int k=isnow+1;k<=NMP_NSOIL;++k){int j=k+NMP_OFF;fact[j]=fdiv(dt,fmul(hcpct[j],dz[j]));}
    int one=1+NMP_OFF,zero=NMP_OFF;
    if(isnow==0)df[one]=fdiv(fadd(fmul(df[one],dz[one]),fmul(0.35f,snowh)),fadd(snowh,dz[one]));
    else df[one]=fdiv(fadd(fmul(df[one],dz[one]),fmul(df[zero],dz[zero])),fadd(dz[zero],dz[one]));
    for(int i=0;i<NMP_NLAY;++i){r.ys[yb+ulong(i)]=df[i];r.ys[yb+ulong(NMP_NLAY+i)]=hcpct[i];r.ys[yb+ulong(2*NMP_NLAY+3*NMP_NSNOW+i)]=fact[i];}
    for(int i=0;i<NMP_NSNOW;++i){r.ys[yb+ulong(2*NMP_NLAY+i)]=snicev[i];r.ys[yb+ulong(2*NMP_NLAY+NMP_NSNOW+i)]=snliqv[i];r.ys[yb+ulong(2*NMP_NLAY+2*NMP_NSNOW+i)]=epore[i];}
}

static inline void noahmp_hrt_core(
    int isnow, thread const float *zsnso, thread const float *stc,
    float tbot,float zbot,thread const float *df,thread const float *hcpct,
    float ssoil,thread const float *phi,thread float *ai,thread float *bi,
    thread float *ci,thread float *rhsts,thread float &botflx) {
    float denom[NMP_NLAY],ddz[NMP_NLAY],dtsdz[NMP_NLAY],eflux[NMP_NLAY];
    for(int i=0;i<NMP_NLAY;++i){ai[i]=0.0f;bi[i]=0.0f;ci[i]=0.0f;rhsts[i]=0.0f;denom[i]=0.0f;ddz[i]=0.0f;dtsdz[i]=0.0f;eflux[i]=0.0f;}
    botflx=0.0f;
    for(int k=isnow+1;k<=NMP_NSOIL;++k){
        int j=k+NMP_OFF;
        if(k==isnow+1){
            denom[j]=-fmul(zsnso[j],hcpct[j]);
            float temp1=-zsnso[j+1];
            ddz[j]=fdiv(2.0f,temp1);
            dtsdz[j]=fdiv(fmul(2.0f,fsub(stc[j],stc[j+1])),temp1);
            eflux[j]=fsub(fsub(fmul(df[j],dtsdz[j]),ssoil),phi[j]);
        }else if(k<NMP_NSOIL){
            denom[j]=fmul(fsub(zsnso[j-1],zsnso[j]),hcpct[j]);
            float temp1=fsub(zsnso[j-1],zsnso[j+1]);
            ddz[j]=fdiv(2.0f,temp1);
            dtsdz[j]=fdiv(fmul(2.0f,fsub(stc[j],stc[j+1])),temp1);
            eflux[j]=fsub(fsub(fmul(df[j],dtsdz[j]),fmul(df[j-1],dtsdz[j-1])),phi[j]);
        }else{
            denom[j]=fmul(fsub(zsnso[j-1],zsnso[j]),hcpct[j]);
            dtsdz[j]=fdiv(fsub(stc[j],tbot),fsub(fmul(0.5f,fadd(zsnso[j-1],zsnso[j])),zbot));
            botflx=-fmul(df[j],dtsdz[j]);
            eflux[j]=fsub(fsub(-botflx,fmul(df[j-1],dtsdz[j-1])),phi[j]);
        }
    }
    for(int k=isnow+1;k<=NMP_NSOIL;++k){
        int j=k+NMP_OFF;
        if(k==isnow+1){ai[j]=0.0f;ci[j]=-fdiv(fmul(df[j],ddz[j]),denom[j]);bi[j]=-ci[j];}
        else if(k<NMP_NSOIL){ai[j]=-fdiv(fmul(df[j-1],ddz[j-1]),denom[j]);ci[j]=-fdiv(fmul(df[j],ddz[j]),denom[j]);bi[j]=-fadd(ai[j],ci[j]);}
        else{ai[j]=-fdiv(fmul(df[j-1],ddz[j-1]),denom[j]);ci[j]=0.0f;bi[j]=-fadd(ai[j],ci[j]);}
        rhsts[j]=fdiv(eflux[j],-denom[j]);
    }
}

static inline void noahmp_rosr12_core(
    thread float *p,thread const float *a,thread const float *b,
    thread float *c,thread const float *d,thread float *delta,int ntop){
    c[NMP_NSOIL+NMP_OFF]=0.0f;
    int t=ntop+NMP_OFF;
    p[t]=fdiv(-c[t],b[t]);
    delta[t]=fdiv(d[t],b[t]);
    for(int k=ntop+1;k<=NMP_NSOIL;++k){
        int i=k+NMP_OFF;
        float recip=fdiv(1.0f,fadd(b[i],fmul(a[i],p[i-1])));
        p[i]=fmul(-c[i],recip);
        delta[i]=fmul(fsub(d[i],fmul(a[i],delta[i-1])),recip);
    }
    p[NMP_NSOIL+NMP_OFF]=delta[NMP_NSOIL+NMP_OFF];
    for(int k=ntop+1;k<=NMP_NSOIL;++k){
        int kk=NMP_NSOIL-k+(ntop-1)+1;
        int i=kk+NMP_OFF;
        p[i]=fadd(fmul(p[i],p[i+1]),delta[i]);
    }
}

static inline void noahmp_hstep_core(
    int isnow,float dt,thread float *ai,thread float *bi,thread float *ci,
    thread float *rhsts,thread float *stc){
    int ntop=isnow+1;
    for(int k=ntop;k<=NMP_NSOIL;++k){
        int j=k+NMP_OFF;
        rhsts[j]=fmul(rhsts[j],dt);ai[j]=fmul(ai[j],dt);
        bi[j]=fadd(1.0f,fmul(bi[j],dt));ci[j]=fmul(ci[j],dt);
    }
    float rhstsin[NMP_NLAY],ciin[NMP_NLAY];
    for(int i=0;i<NMP_NLAY;++i){rhstsin[i]=rhsts[i];ciin[i]=ci[i];}
    noahmp_rosr12_core(ci,ai,bi,ciin,rhstsin,rhsts,ntop);
    for(int k=ntop;k<=NMP_NSOIL;++k){int j=k+NMP_OFF;stc[j]=fadd(stc[j],ci[j]);}
}

kernel void noahmp_thermal_hrt(constant NoahmpThermalHrtResources &r [[buffer(0)]],
                               constant NoahmpThermalHrtParams &q [[buffer(1)]],
                               uint idx [[thread_position_in_grid]]){
    if(idx>=uint(q.ncase))return;
    ulong xb=ulong(idx)*ulong(q.n_in),yb=ulong(idx)*ulong(q.n_out);
    int isnow=r.ixs[ulong(idx)*ulong(q.n_int)];
    float zsnso[NMP_NLAY],stc[NMP_NLAY],df[NMP_NLAY],hcpct[NMP_NLAY],phi[NMP_NLAY];
    for(int i=0;i<NMP_NLAY;++i){zsnso[i]=r.xs[xb+ulong(i)];stc[i]=r.xs[xb+NMP_NLAY+ulong(i)];df[i]=r.xs[xb+2*NMP_NLAY+ulong(i)];hcpct[i]=r.xs[xb+3*NMP_NLAY+ulong(i)];phi[i]=r.xs[xb+4*NMP_NLAY+ulong(i)];}
    float tbot=r.xs[xb+5*NMP_NLAY],zbot=r.xs[xb+5*NMP_NLAY+1],ssoil=r.xs[xb+5*NMP_NLAY+3];
    float ai[NMP_NLAY],bi[NMP_NLAY],ci[NMP_NLAY],rhsts[NMP_NLAY],botflx;
    noahmp_hrt_core(isnow,zsnso,stc,tbot,zbot,df,hcpct,ssoil,phi,ai,bi,ci,rhsts,botflx);
    for(int i=0;i<NMP_NLAY;++i){r.ys[yb+ulong(i)]=ai[i];r.ys[yb+NMP_NLAY+ulong(i)]=bi[i];r.ys[yb+2*NMP_NLAY+ulong(i)]=ci[i];r.ys[yb+3*NMP_NLAY+ulong(i)]=rhsts[i];}
    r.ys[yb+4*NMP_NLAY]=botflx;
}



kernel void noahmp_thermal_hstep(constant NoahmpThermalHstepResources &r [[buffer(0)]],
                                constant NoahmpThermalHstepParams &q [[buffer(1)]],
                                uint idx [[thread_position_in_grid]]){
    if(idx>=uint(q.ncase))return;
    ulong xb=ulong(idx)*ulong(q.n_in),yb=ulong(idx)*ulong(q.n_out);
    int isnow=r.ixs[ulong(idx)*ulong(q.n_int)];
    float ai[NMP_NLAY],bi[NMP_NLAY],ci[NMP_NLAY],rhsts[NMP_NLAY],stc[NMP_NLAY];
    for(int i=0;i<NMP_NLAY;++i){ai[i]=r.xs[xb+ulong(i)];bi[i]=r.xs[xb+NMP_NLAY+ulong(i)];ci[i]=r.xs[xb+2*NMP_NLAY+ulong(i)];rhsts[i]=r.xs[xb+3*NMP_NLAY+ulong(i)];stc[i]=r.xs[xb+4*NMP_NLAY+ulong(i)];}
    float dt=r.xs[xb+5*NMP_NLAY];
    noahmp_hstep_core(isnow,dt,ai,bi,ci,rhsts,stc);
    for(int i=0;i<NMP_NLAY;++i){r.ys[yb+ulong(i)]=ai[i];r.ys[yb+NMP_NLAY+ulong(i)]=bi[i];r.ys[yb+2*NMP_NLAY+ulong(i)]=ci[i];r.ys[yb+3*NMP_NLAY+ulong(i)]=rhsts[i];r.ys[yb+4*NMP_NLAY+ulong(i)]=stc[i];}
}

kernel void noahmp_thermal_tsnosoi(constant NoahmpThermalTsnosoiResources &r [[buffer(0)]],
                                   constant NoahmpThermalTsnosoiParams &q [[buffer(1)]],
                                   uint idx [[thread_position_in_grid]]){
    if(idx>=uint(q.ncase))return;
    ulong xb=ulong(idx)*ulong(q.n_in),yb=ulong(idx)*ulong(q.n_out);
    int isnow=r.ixs[ulong(idx)*ulong(q.n_int)];
    float zsnso[NMP_NLAY],stc[NMP_NLAY],df[NMP_NLAY],hcpct[NMP_NLAY],phi[NMP_NLAY];
    for(int i=0;i<NMP_NLAY;++i){zsnso[i]=r.xs[xb+ulong(i)];stc[i]=r.xs[xb+NMP_NLAY+ulong(i)];df[i]=r.xs[xb+2*NMP_NLAY+ulong(i)];hcpct[i]=r.xs[xb+3*NMP_NLAY+ulong(i)];phi[i]=0.0f;}
    ulong base=5*NMP_NLAY;
    float tbot=r.xs[xb+base],ssoil=r.xs[xb+base+1],dt=r.xs[xb+base+3],snowh=r.xs[xb+base+4],zbot=r.xs[xb+base+6];
    float ai[NMP_NLAY],bi[NMP_NLAY],ci[NMP_NLAY],rhsts[NMP_NLAY],eflxb;
    noahmp_hrt_core(isnow,zsnso,stc,tbot,fsub(zbot,snowh),df,hcpct,ssoil,phi,ai,bi,ci,rhsts,eflxb);
    noahmp_hstep_core(isnow,dt,ai,bi,ci,rhsts,stc);
    for(int i=0;i<NMP_NLAY;++i)r.ys[yb+ulong(i)]=stc[i];
    r.ys[yb+NMP_NLAY]=eflxb;
}

kernel void noahmp_thermal_phasechange(constant NoahmpThermalPhasechangeResources &r [[buffer(0)]],
                                       constant NoahmpThermalPhasechangeParams &q [[buffer(1)]],
                                       uint idx [[thread_position_in_grid]]) {
    if(idx>=uint(q.ncase))return;
    ulong xb=ulong(idx)*ulong(q.n_in),yb=ulong(idx)*ulong(q.n_out);
    int isnow=r.ixs[ulong(idx)*ulong(q.n_int)],ist=r.ixs[ulong(idx)*ulong(q.n_int)+1];
    float fact[NMP_NLAY],dz[NMP_NLAY],stc[NMP_NLAY];
    float snice[NMP_NSNOW],snliq[NMP_NSNOW],smc[NMP_NSOIL],sh2o[NMP_NSOIL];
    float smcmax[NMP_NSOIL],psisat[NMP_NSOIL],bexp[NMP_NSOIL];
    float supercool[NMP_NLAY],mice[NMP_NLAY],mliq[NMP_NLAY],wice0[NMP_NLAY],wmass0[NMP_NLAY],hm[NMP_NLAY],xm[NMP_NLAY];
    int imelt[NMP_NLAY];
    for(int i=0;i<NMP_NLAY;++i){fact[i]=r.xs[xb+ulong(i)];dz[i]=r.xs[xb+ulong(NMP_NLAY+i)];stc[i]=r.xs[xb+ulong(3*NMP_NLAY+i)];supercool[i]=mice[i]=mliq[i]=wice0[i]=wmass0[i]=hm[i]=xm[i]=0.0f;imelt[i]=0;}
    int base=4*NMP_NLAY;
    for(int i=0;i<NMP_NSNOW;++i){snice[i]=r.xs[xb+ulong(base+i)];snliq[i]=r.xs[xb+ulong(base+NMP_NSNOW+i)];}
    base+=2*NMP_NSNOW;
    for(int i=0;i<NMP_NSOIL;++i){smc[i]=r.xs[xb+ulong(base+i)];sh2o[i]=r.xs[xb+ulong(base+NMP_NSOIL+i)];}
    base+=2*NMP_NSOIL;float sneqv=r.xs[xb+ulong(base)],snowh=r.xs[xb+ulong(base+1)],dt=r.xs[xb+ulong(base+2)];base+=3;
    for(int i=0;i<NMP_NSOIL;++i){smcmax[i]=r.xs[xb+ulong(base+i)];psisat[i]=r.xs[xb+ulong(base+NMP_NSOIL+i)];bexp[i]=r.xs[xb+ulong(base+2*NMP_NSOIL+i)];}
    float qmelt=0.0f,ponding=0.0f;
    for(int j=isnow+1;j<=0;++j){int s=j+NMP_OFF;mice[s]=snice[s];mliq[s]=snliq[s];}
    for(int j=1;j<=NMP_NSOIL;++j){int s=j+NMP_OFF;mliq[s]=fmul(fmul(sh2o[j-1],dz[s]),1000.0f);mice[s]=fmul(fmul(fsub(smc[j-1],sh2o[j-1]),dz[s]),1000.0f);}
    for(int j=isnow+1;j<=NMP_NSOIL;++j){int s=j+NMP_OFF;wice0[s]=mice[s];wmass0[s]=fadd(mice[s],mliq[s]);}
    if(ist==1)for(int j=1;j<=NMP_NSOIL;++j){int s=j+NMP_OFF;if(stc[s]<NMP_TFRZ){float smp=fdiv(fmul(0.3336e6f,fsub(NMP_TFRZ,stc[s])),fmul(9.80616f,stc[s]));supercool[s]=fmul(fmul(smcmax[j-1],pow(fdiv(smp,psisat[j-1]),-fdiv(1.0f,bexp[j-1]))),fmul(dz[s],1000.0f));}}
    for(int j=isnow+1;j<=NMP_NSOIL;++j){int s=j+NMP_OFF;if(mice[s]>0.0f&&stc[s]>=NMP_TFRZ)imelt[s]=1;if(mliq[s]>supercool[s]&&stc[s]<NMP_TFRZ)imelt[s]=2;if(isnow==0&&sneqv>0.0f&&j==1&&stc[s]>=NMP_TFRZ)imelt[s]=1;}
    for(int j=isnow+1;j<=NMP_NSOIL;++j){int s=j+NMP_OFF;if(imelt[s]>0){hm[s]=fdiv(fsub(stc[s],NMP_TFRZ),fact[s]);stc[s]=NMP_TFRZ;}if(imelt[s]==1&&hm[s]<0.0f){hm[s]=0.0f;imelt[s]=0;}if(imelt[s]==2&&hm[s]>0.0f){hm[s]=0.0f;imelt[s]=0;}xm[s]=fdiv(fmul(hm[s],dt),0.3336e6f);}
    int one=1+NMP_OFF;
    if(isnow==0&&sneqv>0.0f&&xm[one]>0.0f){float temp1=sneqv;sneqv=max(0.0f,fsub(temp1,xm[one]));float propor=fdiv(sneqv,temp1);snowh=max(0.0f,fmul(propor,snowh));snowh=min(max(snowh,fdiv(sneqv,500.0f)),fdiv(sneqv,50.0f));float heatr=fsub(hm[one],fdiv(fmul(0.3336e6f,fsub(temp1,sneqv)),dt));if(heatr>0.0f){xm[one]=fdiv(fmul(heatr,dt),0.3336e6f);hm[one]=heatr;}else{xm[one]=hm[one]=0.0f;}qmelt=fdiv(max(0.0f,fsub(temp1,sneqv)),dt);ponding=fsub(temp1,sneqv);}
    for(int j=isnow+1;j<=NMP_NSOIL;++j){int s=j+NMP_OFF;if(imelt[s]>0&&fabs(hm[s])>0.0f){float heatr=0.0f;if(xm[s]>0.0f){mice[s]=max(0.0f,fsub(wice0[s],xm[s]));heatr=fsub(hm[s],fdiv(fmul(0.3336e6f,fsub(wice0[s],mice[s])),dt));}else if(xm[s]<0.0f){if(j<=0)mice[s]=min(wmass0[s],fsub(wice0[s],xm[s]));else if(wmass0[s]<supercool[s])mice[s]=0.0f;else mice[s]=max(min(fsub(wmass0[s],supercool[s]),fsub(wice0[s],xm[s])),0.0f);heatr=fsub(hm[s],fdiv(fmul(0.3336e6f,fsub(wice0[s],mice[s])),dt));}mliq[s]=max(0.0f,fsub(wmass0[s],mice[s]));if(fabs(heatr)>0.0f){stc[s]=fadd(stc[s],fmul(fact[s],heatr));if(j<=0){if(fmul(mliq[s],mice[s])>0.0f)stc[s]=NMP_TFRZ;if(mice[s]==0.0f){stc[s]=NMP_TFRZ;hm[s+1]=fadd(hm[s+1],heatr);xm[s+1]=fdiv(fmul(hm[s+1],dt),0.3336e6f);}}}if(j<1)qmelt=fadd(qmelt,fdiv(max(0.0f,fsub(wice0[s],mice[s])),dt));}}
    for(int j=isnow+1;j<=0;++j){int s=j+NMP_OFF;snliq[s]=mliq[s];snice[s]=mice[s];}
    for(int j=1;j<=NMP_NSOIL;++j){int s=j+NMP_OFF;sh2o[j-1]=fdiv(mliq[s],fmul(1000.0f,dz[s]));smc[j-1]=fdiv(fadd(mliq[s],mice[s]),fmul(1000.0f,dz[s]));}
    for(int i=0;i<NMP_NLAY;++i)r.ys[yb+ulong(i)]=stc[i];base=NMP_NLAY;
    for(int i=0;i<NMP_NSNOW;++i){r.ys[yb+ulong(base+i)]=snice[i];r.ys[yb+ulong(base+NMP_NSNOW+i)]=snliq[i];}base+=2*NMP_NSNOW;
    for(int i=0;i<NMP_NSOIL;++i){r.ys[yb+ulong(base+i)]=smc[i];r.ys[yb+ulong(base+NMP_NSOIL+i)]=sh2o[i];}base+=2*NMP_NSOIL;
    r.ys[yb+ulong(base)]=sneqv;r.ys[yb+ulong(base+1)]=snowh;r.ys[yb+ulong(base+2)]=qmelt;r.ys[yb+ulong(base+3)]=ponding;base+=4;
    for(int i=0;i<NMP_NLAY;++i)r.ys[yb+ulong(base+i)]=float(imelt[i]);
}
